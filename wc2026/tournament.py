"""
One simulated 2026 World Cup, with the real rules.

This module is pure logic: it operates on pre-built scoreline matrices (so the
Monte Carlo driver can cache them once) and the structures defined in data.py.

Implemented exactly:
  * round-robin group stage (4 teams, 6 games)
  * FIFA tiebreakers: points -> goal difference -> goals for -> head-to-head
    (points/GD/GF among the tied teams) -> [fair-play omitted: not modelled] ->
    random draw
  * the eight best third-placed teams advance alongside the 24 top-two finishers
  * single-elimination R32 -> R16 -> QF -> SF -> Final, with draws after 90'
    resolved by a strength-weighted coin (extra-time + penalties proxy)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from . import data
from .match_model import outcome_probs, sample_score

# furthest stage a team reaches, as an ordered code
STAGES = ["GROUP", "R32", "R16", "QF", "SF", "FINAL", "WINNER"]
STAGE_IX = {s: i for i, s in enumerate(STAGES)}

GroupMatch = Tuple  # (home, away, score_matrix) or (home, away, P|None, (hg, ag))
TeamStats = Dict[str, Dict[str, int]]


# ---------------------------------------------------------------------------
# Group stage
# ---------------------------------------------------------------------------
def _blank_stats(teams: List[str]) -> TeamStats:
    return {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0} for t in teams}


def _apply_result(stats: TeamStats, h: str, a: str, hg: int, ag: int) -> None:
    stats[h]["gf"] += hg; stats[h]["ga"] += ag
    stats[a]["gf"] += ag; stats[a]["ga"] += hg
    if hg > ag:
        stats[h]["pts"] += 3
    elif hg < ag:
        stats[a]["pts"] += 3
    else:
        stats[h]["pts"] += 1; stats[a]["pts"] += 1
    stats[h]["gd"] = stats[h]["gf"] - stats[h]["ga"]
    stats[a]["gd"] = stats[a]["gf"] - stats[a]["ga"]


def _head_to_head_order(tied: List[str],
                        results: List[Tuple[str, str, int, int]],
                        rng: np.random.Generator) -> List[str]:
    """Order a set of tied teams using only the matches played between them."""
    mini = _blank_stats(tied)
    tied_set = set(tied)
    for h, a, hg, ag in results:
        if h in tied_set and a in tied_set:
            _apply_result(mini, h, a, hg, ag)
    # sort by H2H pts, gd, gf; remaining ties -> random
    jitter = {t: rng.random() for t in tied}
    return sorted(
        tied,
        key=lambda t: (mini[t]["pts"], mini[t]["gd"], mini[t]["gf"], jitter[t]),
        reverse=True,
    )


def play_group(matches: List[GroupMatch], rng: np.random.Generator
               ) -> Tuple[List[str], TeamStats]:
    """
    Simulate one group. Returns (ranked_teams [1st..4th], stats_by_team).
    `matches` are the 6 fixed pairings as (home, away, score_matrix).
    """
    teams = sorted({t for m in matches for t in (m[0], m[1])})
    stats = _blank_stats(teams)
    results: List[Tuple[str, str, int, int]] = []
    for m in matches:
        h, a, P = m[0], m[1], m[2]
        fixed = m[3] if len(m) > 3 else None
        if fixed is not None:                  # match already played in reality
            hg, ag = int(fixed[0]), int(fixed[1])
        else:
            hg, ag = sample_score(P, rng)
        _apply_result(stats, h, a, hg, ag)
        results.append((h, a, hg, ag))

    # primary order: pts -> gd -> gf
    order = sorted(teams,
                   key=lambda t: (stats[t]["pts"], stats[t]["gd"], stats[t]["gf"]),
                   reverse=True)

    # resolve any blocks still level on (pts, gd, gf) via head-to-head
    ranked: List[str] = []
    i = 0
    while i < len(order):
        j = i + 1
        key_i = (stats[order[i]]["pts"], stats[order[i]]["gd"], stats[order[i]]["gf"])
        while j < len(order) and (
            stats[order[j]]["pts"], stats[order[j]]["gd"], stats[order[j]]["gf"]
        ) == key_i:
            j += 1
        block = order[i:j]
        ranked.extend(block if len(block) == 1
                      else _head_to_head_order(block, results, rng))
        i = j
    return ranked, stats


# ---------------------------------------------------------------------------
# Knockout
# ---------------------------------------------------------------------------
def play_ko(home: str, away: str, P: np.ndarray, rng: np.random.Generator) -> str:
    """Return the winner. A 90' draw is resolved by a strength-weighted coin."""
    hg, ag = sample_score(P, rng)
    if hg > ag:
        return home
    if ag > hg:
        return away
    ph, _, pa = outcome_probs(P)
    p_home = ph / (ph + pa) if (ph + pa) > 0 else 0.5
    return home if rng.random() < p_home else away


def _rank_thirds(thirds: List[Tuple[str, Dict[str, int]]],
                 n: int, rng: np.random.Generator) -> List[str]:
    jitter = {t: rng.random() for t, _ in thirds}
    ordered = sorted(
        thirds,
        key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"], jitter[x[0]]),
        reverse=True,
    )
    return [t for t, _ in ordered[:n]]


def simulate_tournament(group_matches: Dict[str, List[GroupMatch]],
                        ko_matrix: Callable[[str, str], np.ndarray],
                        rng: np.random.Generator,
                        fixed_ko: Dict[frozenset, str] | None = None
                        ) -> Dict[str, object]:
    """
    Play one whole tournament.

    group_matches : {group_letter: [6 GroupMatch]} (each possibly pre-played)
    ko_matrix     : callable (home, away) -> neutral score matrix, for knockouts
    fixed_ko      : optional {frozenset({a, b}): winner} for knockout matches
                    already decided in reality; consulted whenever that exact
                    pairing arises in the simulated bracket.

    Returns a dict with:
      'reached'   : {team: stage_code}  furthest stage each team reached
      'champion', 'runner_up', 'semifinalists', 'quarterfinalists'
    """
    def ko_play(h: str, a: str) -> str:
        if fixed_ko:
            w = fixed_ko.get(frozenset((h, a)))
            if w is not None:
                return w
        return play_ko(h, a, ko_matrix(h, a), rng)

    reached: Dict[str, int] = {}
    slot: Dict[str, str] = {}
    thirds: List[Tuple[str, Dict[str, int]]] = []

    for g, matches in group_matches.items():
        ranked, stats = play_group(matches, rng)
        for t in ranked:
            reached[t] = STAGE_IX["GROUP"]
        slot[f"W_{g}"] = ranked[0]
        slot[f"R_{g}"] = ranked[1]
        thirds.append((ranked[2], stats[ranked[2]]))

    best_thirds = _rank_thirds(thirds, 8, rng)
    for i, t in enumerate(best_thirds, start=1):
        slot[f"T{i}"] = t

    # everyone who reached a slot is in the Round of 32
    qualifiers = [slot[s] for pair in data.R32_SLOTS for s in pair]
    for t in qualifiers:
        reached[t] = STAGE_IX["R32"]

    # Round of 32 -> winners
    r32 = [(slot[a], slot[b]) for a, b in data.R32_SLOTS]
    winners = [ko_play(h, a) for h, a in r32]
    _promote(winners, reached, "R16")

    winners = _play_round(winners, ko_play)          # R16 -> 8
    _promote(winners, reached, "QF")
    quarterfinalists = list(winners)

    winners = _play_round(winners, ko_play)          # QF -> 4
    _promote(winners, reached, "SF")
    semifinalists = list(winners)

    winners = _play_round(winners, ko_play)          # SF -> 2 (finalists)
    _promote(winners, reached, "FINAL")
    finalists = list(winners)

    h, a = finalists
    champion = ko_play(h, a)
    reached[champion] = STAGE_IX["WINNER"]
    runner_up = a if champion == h else h

    return {
        "reached": {t: STAGES[i] for t, i in reached.items()},
        "champion": champion,
        "runner_up": runner_up,
        "semifinalists": semifinalists,
        "quarterfinalists": quarterfinalists,
        "group_winners": [slot[f"W_{g}"] for g in group_matches],
        "qualified": qualifiers,
    }


def _play_round(teams: List[str], play: Callable[[str, str], str]) -> List[str]:
    """Pair consecutive teams and play; return winners (binary-tree advance)."""
    out = []
    for i in range(0, len(teams), 2):
        out.append(play(teams[i], teams[i + 1]))
    return out


def _promote(teams: List[str], reached: Dict[str, int], stage: str) -> None:
    for t in teams:
        reached[t] = max(reached.get(t, 0), STAGE_IX[stage])
