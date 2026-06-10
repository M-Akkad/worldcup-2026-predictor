"""
Everything static about the tournament, plus result loading and cleaning.

The single most underrated part of an international-football model is name
normalisation: the historical results dataset and the draw use slightly
different spellings ("Turkey" vs "Türkiye", "Cabo Verde" vs "Cape Verde",
"Côte d'Ivoire" vs "Ivory Coast", ...). Everything here is normalised to one
canonical form so the strength model and the bracket line up.
"""

from __future__ import annotations

import os
import unicodedata
from typing import Dict, List

import numpy as np
import pandas as pd


# --- Name normalisation ---
# Map every alias to one canonical name; both the historical data and the group
# lists are normalised through this, so matching is always on the canonical form.

_ALIASES: Dict[str, str] = {
    # host / common short forms
    "usa": "United States",
    "us": "United States",
    "united states of america": "United States",
    # spellings that differ between sources
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "korea dpr": "North Korea",
    "iran": "Iran",
    "ir iran": "Iran",
    "cape verde": "Cape Verde",
    "cabo verde": "Cape Verde",
    "ivory coast": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "czechia": "Czech Republic",
    "czech republic": "Czech Republic",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    "dr congo": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "congo dr": "DR Congo",
    "curacao": "Curacao",
    "curaçao": "Curacao",
    "china pr": "China",
    "republic of ireland": "Ireland",
    "the gambia": "Gambia",
}


def normalize_team(name: str) -> str:
    """Return the canonical team name for any reasonable spelling."""
    if name is None:
        return name
    key = name.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    # strip accents and retry (handles "Türkiye"/"Côte d'Ivoire" if not listed)
    stripped = "".join(
        c for c in unicodedata.normalize("NFKD", key) if not unicodedata.combining(c)
    )
    if stripped in _ALIASES:
        return _ALIASES[stripped]
    # normalise "&" to "and" and collapse spaces, then retry (feeds vary on this)
    amp = " ".join(key.replace("&", " and ").split())
    if amp in _ALIASES:
        return _ALIASES[amp]
    # default: return the original spelling, leaving already-correct names intact
    return name.strip()


# --- The confirmed 2026 group draw (FIFA, 5 Dec 2025; playoffs resolved 2026) ---
GROUPS_2026: Dict[str, List[str]] = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curacao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

# Co-hosts play their group games at home -> apply home advantage rather than
# treating those matches as neutral.
HOST_TEAMS = {"United States", "Canada", "Mexico"}


def all_tournament_teams() -> List[str]:
    return [t for g in GROUPS_2026.values() for t in g]


# --- Knockout bracket template (Round of 32 -> Final) ---
# Slot ids:
#   "W_A".."W_L"  group winners
#   "R_A".."R_L"  group runners-up
#   "T1".."T8"    the eight best third-placed teams, ranked best (T1) to 8th
#
# IMPORTANT / honest caveat: FIFA assigns the eight third-placed teams to slots
# via an official combination table that depends on *which* group letters those
# thirds come from (there are C(12,8)=495 combinations). Reproducing that exact
# table is fiddly and easy to get wrong. This template instead routes the eight
# thirds by rank (T1 = best third). The group-stage itself is exact; this only
# affects *who meets whom* in the R32, a second-order effect on the headline
# advancement/title probabilities. To use FIFA's exact routing, edit R32_SLOTS.
#
# The template below is a valid 32-team single-elimination tree and avoids any
# R32 match between two teams from the same group (checked in tests).
R32_SLOTS = [
    ("W_A", "T1"),  # M1
    ("W_B", "T2"),  # M2
    ("W_C", "T3"),  # M3
    ("W_D", "T4"),  # M4
    ("W_E", "T5"),  # M5
    ("W_F", "T6"),  # M6
    ("W_G", "T7"),  # M7
    ("W_H", "T8"),  # M8
    ("W_I", "R_A"),  # M9
    ("W_J", "R_B"),  # M10
    ("W_K", "R_C"),  # M11
    ("W_L", "R_D"),  # M12
    ("R_E", "R_F"),  # M13
    ("R_G", "R_H"),  # M14
    ("R_I", "R_J"),  # M15
    ("R_K", "R_L"),  # M16
]
# Consecutive R32 matches feed one R16 match (M1+M2 -> R16_1, ...), then the
# standard binary tree up to the final.


def load_results(path: str) -> pd.DataFrame:
    """
    Load the canonical Kaggle dataset
    'International football results from 1872 to 2026' (martj42).

    Expected columns: date, home_team, away_team, home_score, away_score,
    tournament, city, country, neutral.

    Returns a cleaned DataFrame with normalised team names, datetime `date`,
    integer scores, and boolean `neutral`, sorted chronologically.
    """
    df = pd.read_csv(path)
    needed = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"results file is missing columns: {sorted(missing)}")

    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    if "tournament" not in df.columns:
        df["tournament"] = "Friendly"
    if "neutral" not in df.columns:
        df["neutral"] = False
    else:
        df["neutral"] = _parse_bool(df["neutral"])

    return df.sort_values("date").reset_index(drop=True)


def _parse_bool(s: pd.Series) -> pd.Series:
    """
    Robustly coerce a CSV boolean column. A naive astype(bool) is dangerous:
    on an object-dtype column the string "FALSE" and even NaN are truthy, which
    would silently mark matches as neutral and corrupt the home-advantage fit.
    Missing/unrecognised values default to False (a home match).
    """
    if s.dtype == bool:
        return s
    mapping = {"true": True, "false": False, "1": True, "0": False,
               "yes": True, "no": False, "t": True, "f": False}

    def conv(v):
        if isinstance(v, (bool,)):
            return bool(v)
        if pd.isna(v):
            return False
        if isinstance(v, (int, float)):
            return bool(int(v))
        return mapping.get(str(v).strip().lower(), False)

    return s.map(conv).astype(bool)


def check_coverage(df: pd.DataFrame) -> List[str]:
    """
    Return tournament teams that have no matches in the data (almost always a
    name-mismatch problem). Call this right after load_results to catch issues.
    """
    seen = set(df["home_team"]).union(df["away_team"])
    return [t for t in all_tournament_teams() if t not in seen]


def append_results(path: str, new_rows: pd.DataFrame,
                   default_tournament: str = "FIFA World Cup") -> Dict[str, int]:
    """
    Merge new match results into the results CSV, keeping the model current.

    * team names are normalised, dates coerced to YYYY-MM-DD,
      `neutral` parsed robustly;
    * a result whose (date + home + away) already exists as an unplayed *fixture*
      (missing score) fills that fixture's score, so the full 2026 schedule
      being pre-loaded no longer blocks live results from landing;
    * a result that duplicates an already-scored match is skipped, so re-running
      an import is safe (idempotent);
    * the file is kept sorted by date.

    Returns {"added": n, "updated": n, "skipped": n_duplicates, "total": rows}.
    """
    cols = ["date", "home_team", "away_team", "home_score", "away_score",
            "tournament", "city", "country", "neutral"]
    needed = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = needed - set(new_rows.columns)
    if missing:
        raise ValueError(f"new results are missing columns: {sorted(missing)}")

    new = new_rows.copy()
    for c, default in (("tournament", default_tournament), ("city", ""),
                       ("country", ""), ("neutral", True)):
        if c not in new.columns:
            new[c] = default
    new["home_team"] = new["home_team"].map(normalize_team)
    new["away_team"] = new["away_team"].map(normalize_team)
    new["date"] = pd.to_datetime(new["date"]).dt.strftime("%Y-%m-%d")
    new["home_score"] = new["home_score"].astype(int)
    new["away_score"] = new["away_score"].astype(int)
    new["neutral"] = _parse_bool(new["neutral"])
    new = new[cols]

    if os.path.exists(path):
        old = pd.read_csv(path)
    else:
        old = pd.DataFrame(columns=cols)

    def _key(df: pd.DataFrame) -> pd.Series:
        if len(df) == 0:
            return pd.Series([], dtype=str)
        return (pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d") + "|"
                + df["home_team"].map(normalize_team) + "|"
                + df["away_team"].map(normalize_team))

    old_key = _key(old).to_numpy()
    has_score = (old["home_score"].notna() & old["away_score"].notna()).to_numpy(copy=True) \
        if len(old) else np.array([], dtype=bool)

    added = updated = skipped = 0
    append_rows = []
    for (_, nrow), k in zip(new.iterrows(), _key(new)):
        match = old_key == k
        if not match.any():
            append_rows.append(nrow)
            added += 1
            continue
        fill = match & ~has_score          # fill any unplayed fixture(s) with this score
        if fill.any():
            idx = old.index[fill]
            old.loc[idx, "home_score"] = nrow["home_score"]
            old.loc[idx, "away_score"] = nrow["away_score"]
            old.loc[idx, "neutral"] = nrow["neutral"]
            has_score[fill] = True         # don't let a later duplicate re-fill it
            updated += 1
        else:
            skipped += 1

    parts = [old] + ([pd.DataFrame(append_rows)] if append_rows else [])
    out = pd.concat(parts, ignore_index=True)
    out["_d"] = pd.to_datetime(out["date"])
    out = out.sort_values("_d").drop(columns="_d")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
    return {"added": added, "updated": updated, "skipped": skipped, "total": len(out)}


def upcoming_fixtures(path: str, since: "str | None" = None) -> pd.DataFrame:
    """
    The not-yet-played World Cup matches from the schedule (rows with a missing
    score), normalised, with the venue `neutral` flag from the schedule (host
    group games are already marked non-neutral). Sorted by date; optionally only
    those on/after `since`. This is the forward-looking counterpart of
    `load_results`, which drops these unplayed rows.
    """
    df = pd.read_csv(path)
    needed = {"date", "home_team", "away_team", "home_score", "away_score"}
    if needed - set(df.columns):
        raise ValueError(f"results file is missing columns: {sorted(needed - set(df.columns))}")
    unplayed = df["home_score"].isna() | df["away_score"].isna()
    label = df["tournament"].astype(str).str.lower() if "tournament" in df.columns \
        else pd.Series("", index=df.index)
    is_wc = label.str.contains("world cup") & ~label.str.contains("qual")
    out = df[unplayed & is_wc].copy()
    # parse dates on the small fixture subset, tolerating mixed source formats
    out["date"] = pd.to_datetime(out["date"], errors="coerce", format="mixed")
    out = out.dropna(subset=["date"])
    if since is not None:
        out = out[out["date"] >= pd.Timestamp(since)]
    out["home_team"] = out["home_team"].map(normalize_team)
    out["away_team"] = out["away_team"].map(normalize_team)
    out["neutral"] = _parse_bool(out["neutral"]) if "neutral" in out.columns else False
    return (out[["date", "home_team", "away_team", "neutral", "tournament"]]
            .sort_values("date").reset_index(drop=True))


# --- Keeping the simulation conditioned on reality ---
# Official 2026 calendar: group stage 11-27 June, knockout from 28 June.
GROUP_STAGE_END = "2026-06-27"


def extract_played_tournament_matches(df: pd.DataFrame,
                                      start_date: str = "2026-06-01"):
    """
    Find 2026 World Cup matches that are already in the results data, so the
    Monte Carlo can fix them instead of re-simulating them.

    Returns (group_results, ko_winners):
      group_results : [(home, away, home_goals, away_goals)] for matches between
                      two teams of the same group during the group-stage window
      ko_winners    : {frozenset({a, b}): winner} for *decisive* matches after
                      the group stage (or between teams of different groups).
                      90'-draws decided on penalties are skipped, the basic
                      schema doesn't record the shootout winner; add those by
                      hand if needed.
    """
    start = pd.Timestamp(start_date)
    group_end = pd.Timestamp(GROUP_STAGE_END)
    team_group = {t: g for g, ts in GROUPS_2026.items() for t in ts}
    group_results, ko_winners = [], {}

    for row in df[df["date"] >= start].itertuples(index=False):
        label = str(getattr(row, "tournament", "")).lower()
        if "world cup" not in label or "qual" in label:
            continue
        h, a = row.home_team, row.away_team
        gh, ga = team_group.get(h), team_group.get(a)
        if gh is None or ga is None:
            continue
        hs, as_ = int(row.home_score), int(row.away_score)
        if gh == ga and row.date <= group_end:
            group_results.append((h, a, hs, as_))
        elif hs != as_:
            ko_winners[frozenset((h, a))] = h if hs > as_ else a
    return group_results, ko_winners


# Free, public-domain results feed (no API key): the openfootball project
# publishes the full 2026 schedule with scores filled in as matches are played.
OPENFOOTBALL_2026_URL = ("https://raw.githubusercontent.com/openfootball/"
                         "worldcup.json/master/2026/worldcup.json")


def parse_openfootball(payload: dict) -> pd.DataFrame:
    """Turn an openfootball worldcup.json payload into append_results rows
    (played matches only; fixtures without a score are skipped)."""
    rows = []
    for m in payload.get("matches", []):
        score = None
        sc = m.get("score")
        if isinstance(sc, dict) and sc.get("ft"):
            score = sc["ft"]
        elif m.get("score1") is not None and m.get("score2") is not None:
            score = [m["score1"], m["score2"]]
        if not score:
            continue
        h = normalize_team(m["team1"])
        a = normalize_team(m["team2"])
        rows.append({"date": m["date"], "home_team": h, "away_team": a,
                     "home_score": int(score[0]), "away_score": int(score[1]),
                     "tournament": "FIFA World Cup",
                     "city": m.get("ground", ""), "country": "",
                     "neutral": h not in HOST_TEAMS})
    return pd.DataFrame(rows)


def fetch_openfootball_results(url: str = OPENFOOTBALL_2026_URL,
                               timeout: float = 20.0) -> pd.DataFrame:
    """Download played 2026 World Cup matches from openfootball (free, no key,
    updated roughly daily by its maintainers)."""
    import json as _json
    from urllib.request import urlopen
    with urlopen(url, timeout=timeout) as r:
        payload = _json.loads(r.read().decode("utf-8"))
    return parse_openfootball(payload)
