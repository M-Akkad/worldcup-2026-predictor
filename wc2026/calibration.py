"""
Isotonic recalibration of W/D/L probabilities.

A model can rank matches well but still be systematically over- or
under-confident; recalibration fixes the *level* of the probabilities without
changing the ordering, and typically improves Brier/log-loss/ECE more cheaply
than a smarter model. We fit one isotonic regression per outcome class
(one-vs-rest) on a validation slice and renormalise to the simplex.

Discipline: the calibrator must be fit on data the model has already been
*scored* on out-of-sample, and evaluated on a later, untouched slice,
`calibrated_backtest` does exactly that by splitting the walk-forward scoring
window chronologically in half.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from . import evaluate
from .match_model import GoalsParams


class ProbCalibrator:
    """
    One-vs-rest calibration over 3-class probability vectors.

    method="auto" uses isotonic regression when there is enough data to support
    its step function (>= min_isotonic fit rows) and falls back to Platt
    (sigmoid/logistic) scaling on small samples, where isotonic is known to
    overfit its knots and *hurt* out-of-sample log-loss.
    """

    def __init__(self, method: str = "auto", min_isotonic: int = 1500):
        self.method = method
        self.min_isotonic = min_isotonic
        self.used: Optional[str] = None
        self._models = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "ProbCalibrator":
        probs = np.asarray(probs, dtype=float)
        y = np.asarray(y, dtype=int)
        method = self.method
        if method == "auto":
            method = "isotonic" if len(y) >= self.min_isotonic else "platt"
        self.used = method
        self._models = []
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            for k in range(probs.shape[1]):
                ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                ir.fit(probs[:, k], (y == k).astype(float))
                self._models.append(ir)
        elif method == "platt":
            from sklearn.linear_model import LogisticRegression
            for k in range(probs.shape[1]):
                lr = LogisticRegression(C=1e6, max_iter=1000)
                lr.fit(self._logit(probs[:, k]).reshape(-1, 1),
                       (y == k).astype(int))
                self._models.append(lr)
        else:
            raise ValueError(f"unknown method: {method}")
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if self._models is None:
            raise RuntimeError("fit the calibrator first")
        probs = np.asarray(probs, dtype=float)
        cols = []
        for k, m in enumerate(self._models):
            if self.used == "isotonic":
                cols.append(m.predict(probs[:, k]))
            else:
                x = self._logit(probs[:, k]).reshape(-1, 1)
                cols.append(m.predict_proba(x)[:, 1])
        q = np.clip(np.column_stack(cols), 1e-6, None)
        return q / q.sum(axis=1, keepdims=True)


def calibrated_backtest(df, model: str = "elo", start_date: str = "2022-01-01",
                        params: Optional[GoalsParams] = None,
                        home_advantage: float = 100.0) -> Dict[str, object]:
    """
    Walk-forward once, then split the scoring window chronologically in half:
    fit the calibrator on the first half, evaluate raw vs calibrated on the
    second half. Returns {"raw": metrics, "calibrated": metrics, "ece_raw",
    "ece_calibrated", "n_fit", "n_eval"}.
    """
    res = evaluate.walk_forward(df, params=params, start_date=start_date,
                                home_advantage=home_advantage, model=model)
    probs, y = res["_probs"], res["_y"]
    split = len(y) // 2
    if split < 100:
        raise ValueError("too few scored matches to calibrate; use an earlier start")
    cal = ProbCalibrator().fit(probs[:split], y[:split])
    raw_p, cal_p = probs[split:], cal.transform(probs[split:])
    y2 = y[split:]
    return {
        "raw": evaluate._metrics(raw_p, y2),
        "calibrated": evaluate._metrics(cal_p, y2),
        "ece_raw": evaluate.reliability(raw_p, y2)["ece"],
        "ece_calibrated": evaluate.reliability(cal_p, y2)["ece"],
        "n_fit": int(split), "n_eval": int(len(y2)), "method": cal.used,
    }
