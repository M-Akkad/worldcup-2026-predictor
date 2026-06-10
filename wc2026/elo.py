"""
World-Football-Elo style rating engine. Elo suits international football: it
needs only results (no per-match stats), is naturally recency-weighted, and
predicts at least as well as the FIFA ranking. Following the public scheme,
K scales with match importance, a goal-difference multiplier rewards bigger
wins (diminishing returns), and a home-advantage constant is added on
non-neutral ground. Updates are incremental, so a backtest can walk forward
and predict each match strictly before learning its result.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

DEFAULT_RATING = 1500.0
HOME_ADVANTAGE = 100.0  # Elo points added to the home side on non-neutral ground

_CONTINENTAL = (
    "euro", "copa am", "copa ame", "africa cup", "african cup",
    "asian cup", "gold cup", "confederations", "oceania nations",
)


@dataclass
class KFactors:
    """Tunable K-factors (rating step per competition) and margin sensitivity.

    The defaults reproduce the public World Football Elo scheme exactly, so an
    ``EloRatings`` built without an explicit ``KFactors`` behaves as before. The
    fields are exposed so a search (see ``optimize.py``) can calibrate them on
    the leakage-free backtest instead of leaving them hand-set.
    """
    world_cup: float = 60.0
    qualifier: float = 40.0
    nations_league: float = 40.0
    continental: float = 50.0
    friendly: float = 20.0
    other: float = 30.0
    # Scales the goal-difference multiplier's *effect* above 1.0: 0 ignores the
    # margin entirely, 1 reproduces the classic curve, >1 rewards blowouts more.
    gd_scale: float = 1.0

    def weight(self, name: str) -> float:
        """K-factor by competition importance (World Football Elo scheme)."""
        if not isinstance(name, str):
            return self.other
        n = name.lower()
        if "world cup" in n and "qual" not in n:
            return self.world_cup
        if "qual" in n:  # World Cup / continental qualifiers
            return self.qualifier
        if "nations league" in n:
            return self.nations_league
        if any(c in n for c in _CONTINENTAL):
            return self.continental
        if "friendly" in n:
            return self.friendly
        return self.other  # other competitive tournaments

    def gd_multiplier(self, goal_diff: int) -> float:
        """Bigger margins move ratings more, with diminishing returns, scaled."""
        return 1.0 + self.gd_scale * (goal_diff_multiplier(goal_diff) - 1.0)


_DEFAULT_K = KFactors()


def tournament_weight(name: str) -> float:
    """K-factor by competition importance (World Football Elo scheme)."""
    return _DEFAULT_K.weight(name)


def goal_diff_multiplier(goal_diff: int) -> float:
    """Bigger margins move ratings more, with diminishing returns."""
    g = abs(int(goal_diff))
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


def expected_score(elo_home: float, elo_away: float,
                   home_adv: float = HOME_ADVANTAGE, neutral: bool = False) -> float:
    """Expected match 'score' for the home team in [0, 1]."""
    adv = 0.0 if neutral else home_adv
    diff = (elo_home + adv) - elo_away
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


class EloRatings:
    """Mutable Elo table built incrementally from match results."""

    def __init__(self, default_rating: float = DEFAULT_RATING,
                 home_advantage: float = HOME_ADVANTAGE,
                 k_factors: Optional[KFactors] = None):
        self.default_rating = default_rating
        self.home_advantage = home_advantage
        self.k = k_factors or _DEFAULT_K
        self._r: Dict[str, float] = defaultdict(lambda: default_rating)
        self.last_update: Dict[str, pd.Timestamp] = {}

    # -- access -------------------------------------------------------------
    def rating(self, team: str) -> float:
        return self._r[team]

    def as_dict(self) -> Dict[str, float]:
        return dict(self._r)

    def set_rating(self, team: str, value: float) -> None:
        self._r[team] = float(value)

    # -- learning -----------------------------------------------------------
    def update_match(self, home: str, away: str, hs: int, as_: int,
                     tournament: str = "Friendly", neutral: bool = False,
                     date: Optional[pd.Timestamp] = None) -> None:
        eh, ea = self._r[home], self._r[away]
        we = expected_score(eh, ea, self.home_advantage, neutral)
        if hs > as_:
            w = 1.0
        elif hs < as_:
            w = 0.0
        else:
            w = 0.5
        k = self.k.weight(tournament) * self.k.gd_multiplier(hs - as_)
        delta = k * (w - we)
        self._r[home] = eh + delta
        self._r[away] = ea - delta
        if date is not None:
            self.last_update[home] = date
            self.last_update[away] = date

    def fit(self, df: pd.DataFrame, regress_each_year: float = 0.0) -> "EloRatings":
        """
        Build ratings by replaying `df` (must be chronologically sorted).

        regress_each_year: optional mean-reversion. At each new calendar year,
        every rating is pulled this fraction toward the mean (a light way to
        forget stale strength). 0 disables it; ~0.05-0.10 is a reasonable try.
        """
        last_year = None
        for row in df.itertuples(index=False):
            if regress_each_year > 0:
                yr = row.date.year
                if last_year is not None and yr != last_year:
                    self._regress_to_mean(regress_each_year)
                last_year = yr
            self.update_match(
                row.home_team, row.away_team,
                int(row.home_score), int(row.away_score),
                getattr(row, "tournament", "Friendly"),
                bool(getattr(row, "neutral", False)),
                row.date,
            )
        return self

    def _regress_to_mean(self, frac: float) -> None:
        if not self._r:
            return
        mean = sum(self._r.values()) / len(self._r)
        for t in list(self._r):
            self._r[t] += frac * (mean - self._r[t])


def build_from_results(df: pd.DataFrame, **kwargs) -> EloRatings:
    """Convenience: fit an EloRatings on the full results frame."""
    return EloRatings(**{k: v for k, v in kwargs.items()
                         if k in ("default_rating", "home_advantage",
                                  "k_factors")}).fit(
        df, regress_each_year=kwargs.get("regress_each_year", 0.0)
    )
