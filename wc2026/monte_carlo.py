"""
Run the tournament many times and aggregate the per-team probabilities.

The expensive objects (scoreline matrices) are built once and reused across
all simulations:
  * group matches are fixed pairings; a co-host playing at home gets home
    advantage, every other group game is neutral
  * knockout matches are always neutral; matrices are memoised per matchup
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from . import data, tournament
from .elo import EloRatings
from .match_model import MatchModel
from .tournament import STAGE_IX


def _order_pair(t1: str, t2: str):
    """Decide (home, away, neutral) for a group pairing: a host plays at home."""
    if t1 in data.HOST_TEAMS and t2 not in data.HOST_TEAMS:
        return t1, t2, False
    if t2 in data.HOST_TEAMS and t1 not in data.HOST_TEAMS:
        return t2, t1, False
    return t1, t2, True


def build_group_matches(model: MatchModel,
                        fixed_results=None) -> Dict[str, List[tournament.GroupMatch]]:
    """Build the 72 group matches; pairings already played in reality (passed as
    [(home, away, hg, ag)]) are fixed instead of getting a probability matrix."""
    fixed = {}
    for h, a, hg, ag in (fixed_results or []):
        fixed[frozenset((h, a))] = (h, a, int(hg), int(ag))

    out: Dict[str, List[tournament.GroupMatch]] = {}
    for g, teams in data.GROUPS_2026.items():
        matches = []
        for t1, t2 in combinations(teams, 2):
            key = frozenset((t1, t2))
            if key in fixed:
                h, a, hg, ag = fixed[key]
                matches.append((h, a, None, (hg, ag)))
            else:
                h, a, neutral = _order_pair(t1, t2)
                P = model.score_matrix(h, a, neutral=neutral)
                matches.append((h, a, P, None))
        out[g] = matches
    return out


def make_ko_matrix(model: MatchModel) -> Callable[[str, str], np.ndarray]:
    """Memoised neutral-venue score matrix provider for knockout matches."""
    cache: Dict[tuple, np.ndarray] = {}

    def ko(home: str, away: str) -> np.ndarray:
        key = (home, away)
        P = cache.get(key)
        if P is None:
            P = model.score_matrix(home, away, neutral=True)
            cache[key] = P
        return P

    return ko


def run(model: MatchModel, n_sims: int = 10_000, seed: int = 0,
        elo: EloRatings | None = None,
        fixed_group=None, fixed_ko=None) -> pd.DataFrame:
    """
    Run `n_sims` tournaments and return a per-team probability table sorted by
    title probability. `fixed_group` ([(home, away, hg, ag)]) and `fixed_ko`
    ({frozenset({a,b}): winner}) condition every simulation on matches already
    played in reality. Columns are probabilities in [0, 1]:
      win_group, reach_ko (top-2 or best-third), reach_r16, reach_qf,
      reach_sf, reach_final, win_title , plus group and (if given) elo.
    """
    rng = np.random.default_rng(seed)
    group_matches = build_group_matches(model, fixed_results=fixed_group)
    ko_matrix = make_ko_matrix(model)

    teams = data.all_tournament_teams()
    cnt = {t: dict(win_group=0, reach_ko=0, reach_r16=0, reach_qf=0,
                   reach_sf=0, reach_final=0, win_title=0) for t in teams}

    for _ in range(n_sims):
        res = tournament.simulate_tournament(group_matches, ko_matrix, rng,
                                             fixed_ko=fixed_ko)
        reached = res["reached"]
        for t in res["group_winners"]:
            cnt[t]["win_group"] += 1
        cnt[res["champion"]]["win_title"] += 1
        for t, stage in reached.items():
            ix = STAGE_IX[stage]
            if ix >= STAGE_IX["R32"]:
                cnt[t]["reach_ko"] += 1
            if ix >= STAGE_IX["R16"]:
                cnt[t]["reach_r16"] += 1
            if ix >= STAGE_IX["QF"]:
                cnt[t]["reach_qf"] += 1
            if ix >= STAGE_IX["SF"]:
                cnt[t]["reach_sf"] += 1
            if ix >= STAGE_IX["FINAL"]:
                cnt[t]["reach_final"] += 1

    team_to_group = {t: g for g, ts in data.GROUPS_2026.items() for t in ts}
    rows = []
    for t in teams:
        row = {"team": t, "group": team_to_group[t]}
        if elo is not None:
            row["elo"] = round(elo.rating(t), 1)
        for k, v in cnt[t].items():
            row[k] = v / n_sims
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("win_title", ascending=False)
    return df.reset_index(drop=True)
