"""
OPTIONAL full Dixon-Coles maximum-likelihood goals model.

This fits an attack and defence strength per team directly to goals, plus a home
advantage and the Dixon-Coles low-score correction rho, with a time-decay weight
so recent matches matter more.

When to use it: as a second opinion to ensemble with the Elo model. For
international football the Elo model is usually the more robust default because
many teams play few matches; here we add L2 regularisation (toward average) and
a recent-match window to keep the fit stable, and unknown teams fall back to
average strength.

It is intentionally separate and not used in the walk-forward backtest by
default, because re-fitting an MLE before every historical match is far too slow,
Elo is the right tool for walk-forward evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from .match_model import MatchModel, MAX_GOALS, _dixon_coles_tau


def _poisson_logpmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    return k * np.log(lam) - lam - gammaln(k + 1.0)


class DixonColesModel(MatchModel):
    def __init__(self):
        self.teams: List[str] = []
        self.idx: Dict[str, int] = {}
        self.attack: Optional[np.ndarray] = None
        self.defense: Optional[np.ndarray] = None
        self.gamma: float = 0.25   # home advantage (log scale)
        self.rho: float = -0.08
        self.fitted: bool = False

    # -- fitting ------------------------------------------------------------
    def fit(self, df, as_of=None, years: float = 8.0,
            half_life_days: float = 540.0, reg: float = 0.5,
            maxiter: int = 200) -> "DixonColesModel":
        """
        Fit on matches within `years` of `as_of` (default: last date in df),
        weighting match m by 0.5 ** (age_days / half_life_days).
        """
        d = df.copy()
        as_of = d["date"].max() if as_of is None else as_of
        cutoff = as_of - np.timedelta64(int(years * 365), "D")
        d = d[(d["date"] >= cutoff) & (d["date"] <= as_of)]
        if len(d) < 50:
            raise ValueError("not enough matches in window to fit Dixon-Coles")

        self.teams = sorted(set(d["home_team"]).union(d["away_team"]))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        hi = d["home_team"].map(self.idx).to_numpy()
        ai = d["away_team"].map(self.idx).to_numpy()
        x = d["home_score"].to_numpy(dtype=float)
        y = d["away_score"].to_numpy(dtype=float)
        neutral = d["neutral"].to_numpy(dtype=bool) if "neutral" in d else np.zeros(len(d), bool)
        age_days = (as_of - d["date"]).dt.days.to_numpy(dtype=float)
        w = 0.5 ** (age_days / half_life_days)

        # low-score correction masks
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)

        def unpack(theta):
            a = theta[:n]
            de = theta[n:2 * n]
            gamma = theta[2 * n]
            rho = theta[2 * n + 1]
            return a, de, gamma, rho

        def nll(theta):
            a, de, gamma, rho = unpack(theta)
            log_lh = a[hi] - de[ai] + np.where(neutral, 0.0, gamma)
            log_la = a[ai] - de[hi]
            lh = np.exp(np.clip(log_lh, -4, 4))
            la = np.exp(np.clip(log_la, -4, 4))
            ll = _poisson_logpmf(x, lh) + _poisson_logpmf(y, la)
            # Dixon-Coles tau (guarded positive)
            tau = np.ones_like(lh)
            tau = np.where(m00, 1.0 - lh * la * rho, tau)
            tau = np.where(m01, 1.0 + lh * rho, tau)
            tau = np.where(m10, 1.0 + la * rho, tau)
            tau = np.where(m11, 1.0 - rho, tau)
            ll = ll + np.log(np.clip(tau, 1e-6, None))
            penalty = reg * (np.sum(a ** 2) + np.sum(de ** 2))
            return -np.sum(w * ll) + penalty

        theta0 = np.zeros(2 * n + 2)
        theta0[2 * n] = 0.25      # gamma
        theta0[2 * n + 1] = -0.08  # rho
        bounds = [(-3, 3)] * (2 * n) + [(-1.0, 1.0), (-0.99, 0.99)]
        res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter})

        a, de, gamma, rho = unpack(res.x)
        # centre attack for interpretability (does not change predictions much)
        a = a - a.mean()
        self.attack, self.defense, self.gamma, self.rho = a, de, float(gamma), float(rho)
        self.fitted = True
        return self

    # -- prediction ---------------------------------------------------------
    def _strength(self, team: str):
        i = self.idx.get(team)
        if i is None:  # unknown team -> average
            return 0.0, 0.0
        return self.attack[i], self.defense[i]

    def score_matrix(self, home: str, away: str, neutral: bool = True,
                     extra_home_adv: float = 0.0) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("call fit() before predicting")
        ah, dh = self._strength(home)
        aa, da = self._strength(away)
        gamma = 0.0 if neutral else self.gamma
        lh = float(np.exp(np.clip(ah - da + gamma, -4, 4)))
        la = float(np.exp(np.clip(aa - dh, -4, 4)))
        ks = np.arange(MAX_GOALS + 1)
        ph = np.exp(_poisson_logpmf(ks.astype(float), np.full_like(ks, lh, dtype=float)))
        pa = np.exp(_poisson_logpmf(ks.astype(float), np.full_like(ks, la, dtype=float)))
        P = np.outer(ph, pa)
        P = _dixon_coles_tau(P, lh, la, self.rho)
        s = P.sum()
        return P / s if s > 0 else P
