"""
Turn team strength into a full scoreline distribution. The core object every
model produces is a score matrix P where P[i, j] = Prob(home scores i, away j),
since the tournament needs scorelines, not just W/D/L.

Models: EloMatchModel (Elo gap -> Poisson goal rates -> Dixon-Coles low-score
correction), OrderedLogitModel (proportional-odds logit on the Elo gap, strongest
single 2018 WC model per Robberechts & Davis 2018; made simulator-compatible via
an inverse goals bridge), and EnsembleMatchModel (a weighted average, usually
better calibrated than either).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, factorial
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import fsolve, minimize

from .elo import EloRatings, HOME_ADVANTAGE

MAX_GOALS = 10  # scoreline grid is 0..MAX_GOALS for each team


def _poisson_pmf(lmbda: float, kmax: int) -> np.ndarray:
    ks = np.arange(kmax + 1)
    # stable enough for our small lambdas / k range
    return np.array([exp(-lmbda) * lmbda ** k / factorial(k) for k in ks])


def _dixon_coles_tau(P: np.ndarray, lh: float, la: float, rho: float) -> np.ndarray:
    """Apply the DC dependence correction to the four lowest scorelines."""
    if rho == 0.0:
        return P
    P = P.copy()
    P[0, 0] *= 1.0 - lh * la * rho
    P[0, 1] *= 1.0 + lh * rho
    P[1, 0] *= 1.0 + la * rho
    P[1, 1] *= 1.0 - rho
    return P


def dc_score_matrix(lh: float, la: float, rho: float = 0.0) -> np.ndarray:
    """
    Normalised Dixon-Coles-corrected score matrix from two goal rates.
    Cells are clamped at zero: an aggressive rho combined with large goal rates
    can push a low-score cell negative, which would break sampling downstream.
    """
    P = np.outer(_poisson_pmf(lh, MAX_GOALS), _poisson_pmf(la, MAX_GOALS))
    P = np.clip(_dixon_coles_tau(P, lh, la, rho), 0.0, None)
    s = P.sum()
    return P / s if s > 0 else P


@dataclass
class GoalsParams:
    """Tunable knobs of the Elo->goals bridge (calibrate these on the backtest)."""
    base_total_goals: float = 2.6   # expected total goals in an even match
    sup_per_100_elo: float = 0.40   # goal supremacy added per 100 Elo of edge
    rho: float = -0.08              # Dixon-Coles low-score / draw correction
    home_advantage: float = HOME_ADVANTAGE
    min_rate: float = 0.15          # floor on a team's expected goals


class MatchModel:
    """Interface: every model returns a normalised score matrix."""

    def score_matrix(self, home: str, away: str, neutral: bool = True,
                     extra_home_adv: float = 0.0) -> np.ndarray:
        raise NotImplementedError


class EloMatchModel(MatchModel):
    def __init__(self, elo: EloRatings, params: Optional[GoalsParams] = None):
        self.elo = elo
        self.p = params or GoalsParams()

    def goal_rates(self, home: str, away: str, neutral: bool,
                   extra_home_adv: float) -> Tuple[float, float]:
        adv = (0.0 if neutral else self.p.home_advantage) + extra_home_adv
        diff = (self.elo.rating(home) + adv) - self.elo.rating(away)
        sup = diff / 100.0 * self.p.sup_per_100_elo
        lh = max(self.p.min_rate, (self.p.base_total_goals + sup) / 2.0)
        la = max(self.p.min_rate, (self.p.base_total_goals - sup) / 2.0)
        return lh, la

    def score_matrix(self, home: str, away: str, neutral: bool = True,
                     extra_home_adv: float = 0.0) -> np.ndarray:
        lh, la = self.goal_rates(home, away, neutral, extra_home_adv)
        return dc_score_matrix(lh, la, self.p.rho)


class EnsembleMatchModel(MatchModel):
    def __init__(self, members: Sequence[Tuple[MatchModel, float]]):
        if not members:
            raise ValueError("ensemble needs at least one (model, weight) member")
        tot = sum(w for _, w in members)
        if tot <= 0:
            raise ValueError("ensemble weights must sum to a positive number")
        self.members = [(m, w / tot) for m, w in members]

    def score_matrix(self, home: str, away: str, neutral: bool = True,
                     extra_home_adv: float = 0.0) -> np.ndarray:
        P = None
        for m, w in self.members:
            Pi = m.score_matrix(home, away, neutral, extra_home_adv)
            P = Pi * w if P is None else P + Pi * w
        s = P.sum()
        return P / s if s > 0 else P


# --- reading information out of a score matrix ------------------------------
def outcome_probs(P: np.ndarray) -> Tuple[float, float, float]:
    home = float(np.tril(P, -1).sum())   # home_goals > away_goals
    draw = float(np.trace(P))
    away = float(np.triu(P, 1).sum())
    return home, draw, away


def expected_goals(P: np.ndarray) -> Tuple[float, float]:
    """Expected goals for (home, away) from the score matrix."""
    g = np.arange(P.shape[0])
    eh = float((P.sum(axis=1) * g).sum())
    ea = float((P.sum(axis=0) * g).sum())
    return eh, ea


def expected_goal_diff(P: np.ndarray) -> float:
    eh, ea = expected_goals(P)
    return eh - ea


def total_goals_dist(P: np.ndarray) -> np.ndarray:
    """
    Distribution of the match total (home + away goals): index t holds
    Prob(total == t), for t in 0..2*MAX_GOALS. Summing the anti-diagonals of P.
    """
    n = P.shape[0]
    dist = np.zeros(2 * (n - 1) + 1)
    for i in range(n):
        for j in range(P.shape[1]):
            dist[i + j] += P[i, j]
    return dist


def prob_over(P: np.ndarray, line: float = 2.5) -> float:
    """Prob(total goals > line). Lines are half-integers (2.5, 1.5, …)."""
    dist = total_goals_dist(P)
    ts = np.arange(len(dist))
    return float(dist[ts > line].sum())


def prob_btts(P: np.ndarray) -> float:
    """Prob(both teams score), i.e. home >= 1 and away >= 1."""
    return float(P[1:, 1:].sum())


def sample_score(P: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
    """
    Sample one (home_goals, away_goals) from the score matrix. Implemented via
    cumulative sum + searchsorted rather than rng.choice: identical semantics,
    several times faster (this is the Monte Carlo hot path), and robust to tiny
    floating-point drift in the matrix sum.
    """
    flat = P.ravel()
    c = np.cumsum(flat)
    idx = int(np.searchsorted(c, rng.random() * c[-1], side="right"))
    if idx >= flat.size:  # guard against u == c[-1] edge
        idx = flat.size - 1
    n = P.shape[1]
    return idx // n, idx % n


def blend_with_market(model_probs: Tuple[float, float, float],
                      market_probs: Tuple[float, float, float],
                      market_weight: float = 0.5) -> Tuple[float, float, float]:
    """
    Convex blend of model and (de-vigged) bookmaker probabilities. Market odds
    are the single strongest public predictor of football matches, so blending
    toward them is the cheapest big accuracy win available. `market_weight` in
    [0,1]; 0 = ignore market, 1 = trust market only.
    """
    w = market_weight
    out = tuple((1 - w) * m + w * k for m, k in zip(model_probs, market_probs))
    s = sum(out)
    return tuple(x / s for x in out)


# Ordered-logit model: a proportional-odds logit on the pre-match Elo difference
# plus a home-field term (away < draw < home), with the slope and cutpoints
# *learned* from data rather than hand-set:
#     eta     = beta_elo*(eloH - eloA)/scale + beta_home*home_flag
#     P(away) = sigma(theta1 - eta);  P(home) = 1 - sigma(theta2 - eta)
# Strongest single 2018 WC model (Robberechts & Davis 2018), so a good ensemble
# partner for Elo/Dixon-Coles. score_matrix() stays simulator-compatible via an
# inverse bridge: it solves for the two goal rates whose Dixon-Coles matrix
# reproduces the model's home/away-win probabilities.


def _logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class OrderedLogitParams:
    beta_elo: float
    beta_home: float
    theta1: float            # cutpoint between away win and draw
    theta2: float            # cutpoint between draw and home win
    elo_scale: float = 400.0
    base_total_goals: float = 2.6   # used only by the inverse score-matrix bridge
    rho: float = -0.08


def fit_ordered_logit(elo_diff, home_flag, y, elo_scale: float = 400.0
                      ) -> OrderedLogitParams:
    """
    Maximum-likelihood fit of the proportional-odds model.
    `y` uses 0=home win, 1=draw, 2=away win (the project convention).
    The cutpoint gap is reparameterised as exp(.) to guarantee theta1 < theta2.
    """
    x = np.asarray(elo_diff, dtype=float) / elo_scale
    h = np.asarray(home_flag, dtype=float)
    y = np.asarray(y, dtype=int)

    def nll(theta):
        be, bh, t1, loggap = theta
        t2 = t1 + np.exp(loggap)
        eta = be * x + bh * h
        c1 = _logistic(t1 - eta)
        c2 = _logistic(t2 - eta)
        p = np.empty((len(y), 3))
        p[:, 0] = np.clip(1.0 - c2, 1e-12, 1.0)   # home
        p[:, 1] = np.clip(c2 - c1, 1e-12, 1.0)    # draw
        p[:, 2] = np.clip(c1, 1e-12, 1.0)         # away
        return -np.sum(np.log(p[np.arange(len(y)), y]))

    res = minimize(
        nll, np.array([1.0, 0.3, -0.4, np.log(0.8)]), method="L-BFGS-B",
        bounds=[(-10.0, 10.0),   # beta_elo
                (-5.0, 5.0),     # beta_home
                (-10.0, 10.0),   # theta1
                (np.log(1e-3), np.log(10.0))],  # log cutpoint gap
        options={"maxiter": 500},
    )
    be, bh, t1, loggap = res.x
    return OrderedLogitParams(beta_elo=float(be), beta_home=float(bh),
                              theta1=float(t1), theta2=float(t1 + np.exp(loggap)),
                              elo_scale=elo_scale)


def _rates_for_probs(p_home: float, p_away: float, total_goals: float,
                     rho: float) -> Tuple[float, float]:
    """Inverse bridge: goal rates whose DC matrix gives these win probabilities."""
    import warnings

    s0 = float(np.clip((p_home - p_away) * total_goals, -0.9 * total_goals,
                       0.9 * total_goals))
    lh0 = max(0.1, (total_goals + s0) / 2.0)
    la0 = max(0.1, (total_goals - s0) / 2.0)

    def f(logl):
        lh, la = np.exp(logl)
        ph, _, pa = outcome_probs(dc_score_matrix(lh, la, rho))
        return [ph - p_home, pa - p_away]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # fsolve chatters when stuck
            sol = fsolve(f, np.log([lh0, la0]), xtol=1e-8)
        lh, la = np.exp(sol)
        if not (np.isfinite(lh) and np.isfinite(la)):
            raise ValueError
        # accept only if the solver actually hit the target probabilities
        resid = f(np.log([lh, la]))
        if max(abs(resid[0]), abs(resid[1])) > 1e-4:
            raise ValueError
        return float(np.clip(lh, 0.05, 12.0)), float(np.clip(la, 0.05, 12.0))
    except Exception:
        return lh0, la0


class OrderedLogitModel(MatchModel):
    def __init__(self, elo: EloRatings, params: OrderedLogitParams):
        self.elo = elo
        self.p = params

    def proba(self, home: str, away: str, neutral: bool = True
              ) -> Tuple[float, float, float]:
        x = (self.elo.rating(home) - self.elo.rating(away)) / self.p.elo_scale
        hflag = 0.0 if neutral else 1.0
        eta = self.p.beta_elo * x + self.p.beta_home * hflag
        c1 = float(_logistic(self.p.theta1 - eta))
        c2 = float(_logistic(self.p.theta2 - eta))
        p_home = max(1.0 - c2, 1e-9)
        p_draw = max(c2 - c1, 1e-9)
        p_away = max(c1, 1e-9)
        s = p_home + p_draw + p_away
        return p_home / s, p_draw / s, p_away / s

    def score_matrix(self, home: str, away: str, neutral: bool = True,
                     extra_home_adv: float = 0.0) -> np.ndarray:
        p_home, _, p_away = self.proba(home, away, neutral)
        lh, la = _rates_for_probs(p_home, p_away, self.p.base_total_goals, self.p.rho)
        return dc_score_matrix(lh, la, self.p.rho)

    @classmethod
    def fit_from_history(cls, df, home_advantage: float = HOME_ADVANTAGE,
                         base_total_goals: float = 2.6, rho: float = -0.08
                         ) -> "OrderedLogitModel":
        """
        Build pre-match Elo-difference covariates by walking the data once
        (leakage-free), fit the ordered logit, and keep the final Elo for
        predicting 2026 matchups.
        """
        el = EloRatings(home_advantage=home_advantage)
        diffs, flags, ys = [], [], []
        for row in df.itertuples(index=False):
            neutral = bool(getattr(row, "neutral", False))
            diffs.append(el.rating(row.home_team) - el.rating(row.away_team))
            flags.append(0.0 if neutral else 1.0)
            hs, as_ = int(row.home_score), int(row.away_score)
            ys.append(0 if hs > as_ else (1 if hs == as_ else 2))
            el.update_match(row.home_team, row.away_team, hs, as_,
                            getattr(row, "tournament", "Friendly"), neutral, row.date)
        params = fit_ordered_logit(diffs, flags, ys)
        params.base_total_goals = base_total_goals
        params.rho = rho
        return cls(el, params)
