"""
Optuna (TPE) search over the whole forecasting pipeline, the Bayesian upgrade of
`evaluate.tune_goals_params`. Calibrates everything otherwise hand-set: the Elo
K-factors and margin sensitivity, home advantage, yearly mean-reversion, the
Elo->goals bridge knobs, and (for the ensemble) the blend weight.

Nothing is fitted on the scoring window, every trial runs the same leakage-free
`evaluate.walk_forward`, so a config is adopted only if it beats the current
default out-of-sample (RPS by default; log_loss/brier optional).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import pandas as pd

from .elo import KFactors
from .evaluate import walk_forward
from .match_model import GoalsParams

# Search bounds. Wide enough to matter, tight enough to stay physically sane
# (e.g. home advantage can't be negative, rho stays in the DC-stable region).
BOUNDS = {
    "base_total_goals": (2.2, 3.1),
    "sup_per_100_elo": (0.20, 0.60),
    "rho": (-0.20, 0.05),
    "min_rate": (0.05, 0.30),
    "home_advantage": (40.0, 140.0),
    "regress_each_year": (0.0, 0.15),
    "k_world_cup": (30.0, 90.0),
    "k_qualifier": (20.0, 60.0),
    "k_nations_league": (20.0, 60.0),
    "k_continental": (25.0, 70.0),
    "k_friendly": (5.0, 40.0),
    "k_other": (15.0, 50.0),
    "gd_scale": (0.0, 2.0),
    "ens_weight": (0.0, 1.0),
}


@dataclass
class Config:
    """A concrete point in the search space -> arguments for walk_forward."""
    goals_params: GoalsParams
    k_factors: KFactors
    home_advantage: float
    regress_each_year: float
    ens_weight: float

    def as_flat_dict(self) -> Dict[str, float]:
        return {
            "base_total_goals": self.goals_params.base_total_goals,
            "sup_per_100_elo": self.goals_params.sup_per_100_elo,
            "rho": self.goals_params.rho,
            "min_rate": self.goals_params.min_rate,
            "home_advantage": self.home_advantage,
            "regress_each_year": self.regress_each_year,
            "k_world_cup": self.k_factors.world_cup,
            "k_qualifier": self.k_factors.qualifier,
            "k_nations_league": self.k_factors.nations_league,
            "k_continental": self.k_factors.continental,
            "k_friendly": self.k_factors.friendly,
            "k_other": self.k_factors.other,
            "gd_scale": self.k_factors.gd_scale,
            "ens_weight": self.ens_weight,
        }


@dataclass
class TuneResult:
    model: str
    metric: str
    n_trials: int
    best_value: float                 # metric at the tuned config (lower=better)
    baseline_value: float             # metric at the current default config
    improved: bool                    # did the tuned config beat the default?
    best_config: Dict[str, float]     # flat params, ready to copy into defaults
    best_metrics: Dict[str, float]    # full metric dict (rps/log_loss/brier/...)
    baseline_metrics: Dict[str, float]
    history: list = field(default_factory=list)  # best-so-far per trial


def _config_from_params(p: Dict[str, float], model: str) -> Config:
    """Build a Config from a flat parameter dict (Optuna or defaults)."""
    gp = GoalsParams(
        base_total_goals=p["base_total_goals"],
        sup_per_100_elo=p["sup_per_100_elo"],
        rho=p["rho"],
        home_advantage=p["home_advantage"],
        min_rate=p["min_rate"],
    )
    kf = KFactors(
        world_cup=p["k_world_cup"],
        qualifier=p["k_qualifier"],
        nations_league=p["k_nations_league"],
        continental=p["k_continental"],
        friendly=p["k_friendly"],
        other=p["k_other"],
        gd_scale=p["gd_scale"],
    )
    ens_weight = p["ens_weight"] if model == "ensemble" else 0.5
    return Config(gp, kf, p["home_advantage"], p["regress_each_year"], ens_weight)


def _default_params() -> Dict[str, float]:
    """The current shipped defaults, as a flat dict (the baseline to beat)."""
    gp, kf = GoalsParams(), KFactors()
    return {
        "base_total_goals": gp.base_total_goals,
        "sup_per_100_elo": gp.sup_per_100_elo,
        "rho": gp.rho,
        "min_rate": gp.min_rate,
        "home_advantage": gp.home_advantage,
        "regress_each_year": 0.0,
        "k_world_cup": kf.world_cup,
        "k_qualifier": kf.qualifier,
        "k_nations_league": kf.nations_league,
        "k_continental": kf.continental,
        "k_friendly": kf.friendly,
        "k_other": kf.other,
        "gd_scale": kf.gd_scale,
        "ens_weight": 0.5,
    }


def _score(df, cfg: Config, model: str, metric: str,
           start_date: str, end_date: Optional[str]) -> Dict[str, object]:
    return walk_forward(
        df, params=cfg.goals_params, start_date=start_date, end_date=end_date,
        home_advantage=cfg.home_advantage, regress_each_year=cfg.regress_each_year,
        model=model, k_factors=cfg.k_factors, ens_weight=cfg.ens_weight,
    )


def optimize(df: pd.DataFrame, *,
             model: str = "ensemble",
             n_trials: int = 40,
             metric: str = "rps",
             start_date: str = "2022-01-01",
             end_date: Optional[str] = None,
             seed: int = 0,
             progress: Optional[Callable[[int, int, float], None]] = None
             ) -> TuneResult:
    """
    Run a TPE search and return the best config plus a head-to-head against the
    current shipped defaults. `metric` is any key in walk_forward's metric dict
    (rps, log_loss, brier); all are minimised. `progress(done, total, best)` is
    called after each trial if supplied (used by the web UI for a progress bar).
    """
    import optuna

    if model not in ("elo", "ordered_logit", "ensemble"):
        raise ValueError(f"unknown model: {model}")
    if metric not in ("rps", "log_loss", "brier"):
        raise ValueError(f"metric must be rps/log_loss/brier, got {metric}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # The ordered-logit ignores the goals bridge, so only tune what it uses.
    tune_goals = model in ("elo", "ensemble")
    tune_blend = model == "ensemble"

    def suggest(trial) -> Config:
        base = _default_params()
        p = dict(base)
        # Elo engine + K-factors are always in play (Elo drives every model).
        for key in ("home_advantage", "regress_each_year", "k_world_cup",
                    "k_qualifier", "k_nations_league", "k_continental",
                    "k_friendly", "k_other", "gd_scale"):
            lo, hi = BOUNDS[key]
            p[key] = trial.suggest_float(key, lo, hi)
        if tune_goals:
            for key in ("base_total_goals", "sup_per_100_elo", "rho", "min_rate"):
                lo, hi = BOUNDS[key]
                p[key] = trial.suggest_float(key, lo, hi)
        if tune_blend:
            lo, hi = BOUNDS["ens_weight"]
            p["ens_weight"] = trial.suggest_float("ens_weight", lo, hi)
        return _config_from_params(p, model)

    history: list = []

    def objective(trial):
        cfg = suggest(trial)
        res = _score(df, cfg, model, metric, start_date, end_date)
        return res["model"][metric]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def _cb(study, trial):
        history.append(float(study.best_value))
        if progress:
            progress(trial.number + 1, n_trials, float(study.best_value))

    study.optimize(objective, n_trials=n_trials, callbacks=[_cb])

    # Full metrics at the winning config and at the shipped defaults.
    best_cfg = _config_from_params({**_default_params(), **study.best_params}, model)
    best_res = _score(df, best_cfg, model, metric, start_date, end_date)
    base_cfg = _config_from_params(_default_params(), model)
    base_res = _score(df, base_cfg, model, metric, start_date, end_date)

    best_metrics = {**best_res["model"], "ece": best_res["reliability"]["ece"]}
    base_metrics = {**base_res["model"], "ece": base_res["reliability"]["ece"]}
    best_value = best_metrics[metric]
    baseline_value = base_metrics[metric]

    return TuneResult(
        model=model, metric=metric, n_trials=n_trials,
        best_value=best_value, baseline_value=baseline_value,
        improved=best_value < baseline_value,
        best_config=best_cfg.as_flat_dict(),
        best_metrics=best_metrics, baseline_metrics=base_metrics,
        history=history,
    )
