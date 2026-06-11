"""
Turn bookmaker odds into fair probabilities. Raw inverse odds overstate every
outcome because of the bookmaker margin (the "vig"), and unevenly so, longshots
are shaded more than favourites (the favourite-longshot bias). `devig_proportional`
just removes the margin; `devig_shin` (Shin 1992/1993) models the margin as
protection against insiders and removes it while correcting the bias, the
recommended way to extract probabilities before blending with a model.

Usage:
    probs = devig_shin([2.10, 3.30, 3.90])   # decimal odds H/D/A -> fair probs
    blended = blend(model_probs, probs, market_weight=0.5)
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def implied(odds: Sequence[float]) -> np.ndarray:
    """Raw inverse odds (sum > 1 by the bookmaker margin)."""
    o = np.asarray(odds, dtype=float)
    if (o <= 1.0).any():
        raise ValueError("decimal odds must be > 1")
    return 1.0 / o


def devig_proportional(odds: Sequence[float]) -> np.ndarray:
    """Margin removed by simple normalisation (keeps favourite-longshot bias)."""
    pi = implied(odds)
    return pi / pi.sum()


def devig_shin(odds: Sequence[float], tol: float = 1e-12) -> np.ndarray:
    """
    Shin's method. Solves for the insider fraction z such that the implied
    'true' probabilities sum to one:

        p_i(z) = ( sqrt(z^2 + 4 (1 - z) pi_i^2 / s) - z ) / ( 2 (1 - z) )

    with pi_i the inverse odds and s their sum. Falls back to proportional
    normalisation when the book has no margin (s <= 1) or bisection cannot
    bracket a root (degenerate inputs).
    """
    pi = implied(odds)
    s = float(pi.sum())
    if s <= 1.0 + 1e-12:
        return pi / s

    def p_of(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / s) - z) / (2.0 * (1.0 - z))

    lo, hi = 0.0, 0.999
    f_lo = p_of(lo).sum() - 1.0       # = sqrt(s) - 1 > 0 when s > 1
    f_hi = p_of(hi).sum() - 1.0
    if f_lo <= 0 or f_hi >= 0:
        return devig_proportional(odds)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = p_of(mid).sum() - 1.0
        if abs(f_mid) < tol:
            lo = hi = mid
            break
        if f_mid > 0:
            lo = mid
        else:
            hi = mid
    p = p_of(0.5 * (lo + hi))
    p = np.clip(p, 1e-9, None)
    return p / p.sum()


def parse_odds(spec: str) -> Tuple[float, float, float]:
    """Parse "2.10,3.30,3.90" (decimal odds: home, draw, away)."""
    parts = [float(x) for x in spec.split(",")]
    if len(parts) != 3:
        raise ValueError('odds must be "HOME,DRAW,AWAY", e.g. "2.10,3.30,3.90"')
    return parts[0], parts[1], parts[2]
