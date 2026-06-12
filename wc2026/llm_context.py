"""
OPTIONAL LLM layer. The LLM is not the probability engine (it isn't calibrated
and can't beat Elo/Poisson at the maths); it supplies what historical data can't
see, recent news (injuries, suspensions, form, fatigue) turned into a small,
bounded Elo delta per team, plus team-name wrangling and readable previews.

Provider-neutral: it calls any OpenAI-compatible chat endpoint, configured via
env vars LLM_API_KEY, LLM_MODEL and (optionally) LLM_BASE_URL. Discipline: deltas
are clamped (default +/-60 Elo) so context nudges but never dominates; everything
degrades to "no adjustment" without a key; and you must A/B test it on the
walk-forward backtest, keeping it only if it actually helps.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class TeamAdjustment:
    team: str
    elo_delta: float
    confidence: float
    reason: str


# ---------------------------------------------------------------------------
def _llm_chat(system: str, user: str, max_tokens: int,
              model: Optional[str] = None) -> Optional[str]:
    """
    One chat-completion call against any OpenAI-compatible endpoint. Configured
    via env: LLM_API_KEY (required), LLM_MODEL (required, or pass `model`),
    LLM_BASE_URL (default https://api.openai.com/v1). Returns the reply text, or
    None if unconfigured or on error (the feature then degrades to no-op).
    """
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("[llm_context] LLM_API_KEY not set -> no adjustments.")
        return None
    model = model or os.environ.get("LLM_MODEL")
    if not model:
        print("[llm_context] LLM_MODEL not set -> no adjustments.")
        return None
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = json.dumps({
        "model": model, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[llm_context] LLM request failed: {e}")
        return None


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1:] if "\n" in text else text
    # grab the outermost JSON array/object
    for open_c, close_c in (("[", "]"), ("{", "}")):
        i, j = text.find(open_c), text.rfind(close_c)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
def get_team_adjustments(teams: List[str], as_of: Optional[str] = None,
                         model: Optional[str] = None, use_web_search: bool = True,
                         max_delta: float = 60.0,
                         client=None) -> Dict[str, TeamAdjustment]:
    """
    Ask the LLM for a bounded Elo delta per team based on *short-term* context
    (injuries, suspensions, form, manager, fatigue) relative to baseline
    strength. Returns {} if the LLM is not configured. (`use_web_search`/`client`
    are accepted for compatibility but unused by the generic endpoint.)
    """
    as_of = as_of or "the current date"
    sys = (
        "You are a football analyst feeding a quantitative World Cup model. "
        "The model already encodes long-run team strength via Elo. Your job is "
        "ONLY to express short-term context as a small Elo adjustment per team: "
        "injuries/suspensions to key players, current form, a new manager, "
        "morale, fatigue or long travel. Do NOT restate overall strength. "
        f"Each elo_delta must be between {-max_delta:.0f} and {max_delta:.0f}; "
        "use 0 when there is no notable recent news. Be conservative."
    )
    instr = (
        f"As of {as_of}, assess these teams for the 2026 World Cup:\n"
        f"{', '.join(teams)}\n\n"
        "Respond with ONLY a JSON array, one object per team:\n"
        '[{"team": "...", "elo_delta": <number>, '
        '"confidence": <0..1>, "reason": "<one short sentence>"}]'
    )
    text = _llm_chat(sys, instr, max_tokens=4000, model=model)
    if text is None:
        return {}

    data = _parse_json(text)
    if not isinstance(data, list):
        print("[llm_context] could not parse adjustments JSON -> none applied.")
        return {}

    out: Dict[str, TeamAdjustment] = {}
    for rec in data:
        try:
            team = str(rec["team"]).strip()
            delta = float(rec.get("elo_delta", 0.0))
            delta = max(-max_delta, min(max_delta, delta))  # clamp
            out[team] = TeamAdjustment(
                team=team, elo_delta=delta,
                confidence=float(rec.get("confidence", 0.5)),
                reason=str(rec.get("reason", "")).strip(),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def apply_adjustments(elo, adjustments: Dict[str, TeamAdjustment]):
    """Return a NEW EloRatings with clamped deltas added (original untouched)."""
    from .elo import EloRatings
    from . import data as _data

    new = EloRatings(default_rating=elo.default_rating,
                     home_advantage=elo.home_advantage)
    for t, r in elo.as_dict().items():
        new.set_rating(t, r)
    for team, adj in adjustments.items():
        canon = _data.normalize_team(team)
        new.set_rating(canon, new.rating(canon) + adj.elo_delta)
    return new


# ---------------------------------------------------------------------------
def explain_match(home: str, away: str, probs, top_scorelines=None,
                  model: Optional[str] = None, client=None) -> str:
    """Readable preview generated FROM the model's numbers (no new prediction)."""
    ph, pd_, pa = probs
    fallback = (f"{home} {ph:.0%} win / {pd_:.0%} draw / {pa:.0%} {away} win"
                + (f"; likeliest scoreline {top_scorelines[0]}" if top_scorelines else ""))
    lines = (f"Model output for {home} vs {away} (neutral venue): "
             f"home win {ph:.1%}, draw {pd_:.1%}, away win {pa:.1%}.")
    if top_scorelines:
        lines += " Most likely scorelines: " + ", ".join(
            f"{s} ({p:.1%})" for s, p in top_scorelines)
    text = _llm_chat(
        "Write a 2-3 sentence match preview that faithfully reflects the supplied "
        "probabilities. Do not invent new numbers.", lines, max_tokens=300, model=model)
    return (text or "").strip() or fallback


def narrate_tournament(prob_table, top_n: int = 8,
                       model: Optional[str] = None, client=None) -> str:
    """Opta-style write-up generated from the Monte Carlo probability table."""
    head = prob_table.head(top_n)
    summary = "\n".join(
        f"{r.team}: title {r.win_title:.1%}, reach final {r.reach_final:.1%}, "
        f"reach SF {r.reach_sf:.1%}" for r in head.itertuples(index=False))
    fallback = "Title-odds leaders:\n" + summary
    text = _llm_chat(
        "Summarise these model-derived World Cup probabilities in a short, "
        "engaging paragraph. Reflect the numbers exactly; add no predictions of "
        "your own.", summary, max_tokens=600, model=model)
    return (text or "").strip() or fallback


# ---------------------------------------------------------------------------
# Caching adjustments to disk
# ---------------------------------------------------------------------------
# One LLM call for 48 teams takes a while and costs tokens, and
# injuries/suspensions don't change by the hour. So fetch once,
# cache to JSON, and reuse for every simulation that day.

def save_adjustments(adjustments: Dict[str, TeamAdjustment], path: str) -> str:
    payload = {
        "timestamp": time.time(),
        "fetched_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "adjustments": [asdict(a) for a in adjustments.values()],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_adjustments(path: str, max_age_hours: Optional[float] = 24.0
                     ) -> Dict[str, TeamAdjustment]:
    """Load cached adjustments; returns {} if missing, unreadable, or stale."""
    if not os.path.exists(path):
        print(f"[llm_context] no adjustment cache at {path}")
        return {}
    try:
        with open(path) as f:
            payload = json.load(f)
        age_h = (time.time() - float(payload["timestamp"])) / 3600.0
        if max_age_hours is not None and age_h > max_age_hours:
            print(f"[llm_context] cache is {age_h:.1f}h old (> {max_age_hours:.0f}h) "
                  f"-> ignoring; refresh with: python run.py update --refresh-llm")
            return {}
        out = {}
        for rec in payload.get("adjustments", []):
            out[rec["team"]] = TeamAdjustment(
                team=rec["team"], elo_delta=float(rec["elo_delta"]),
                confidence=float(rec.get("confidence", 0.5)),
                reason=str(rec.get("reason", "")))
        if out:
            print(f"[llm_context] loaded {len(out)} cached adjustments "
                  f"(fetched {payload.get('fetched_at', '?')})")
        return out
    except Exception as e:
        print(f"[llm_context] could not read cache: {e}")
        return {}
