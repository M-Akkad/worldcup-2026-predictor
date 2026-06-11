"""
Honest, leakage-free backtesting. The only test that matters is walk-forward:
walk history in date order, predict each match using only information available
before kickoff, then reveal the result and update, scoring the probabilistic
forecast, not just the named winner.

Metrics: RPS (the field standard, since outcomes are ordinal home>draw>away;
Constantinou & Fenton 2012), log-loss (strictly proper, local), multiclass Brier,
and accuracy (weakest; reported for intuition only). A reliability diagram / ECE
checks calibration, of the matches called ~30%, did ~30% happen? Every metric is
compared against the base-rate baseline; beating it means real skill.

Models, backtested under identical conditions: "elo" (Elo -> Dixon-Coles bridge),
"ordered_logit" (Robberechts 2018), "ensemble" (average of the two).
"""

from __future__ import annotations

from dataclasses import asdict
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .elo import EloRatings, KFactors
from .match_model import (EloMatchModel, GoalsParams, OrderedLogitModel,
                          expected_goals, fit_ordered_logit, outcome_probs,
                          prob_btts, prob_over, total_goals_dist)

EPS = 1e-12


def _result_class(hs: int, as_: int) -> int:
    return 0 if hs > as_ else (1 if hs == as_ else 2)  # home / draw / away


def _rps(probs: np.ndarray, y: np.ndarray) -> float:
    """Ranked probability score for ordered outcomes [home, draw, away]."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    cp = np.cumsum(probs, axis=1)[:, :-1]    # cumulative forecast (r-1 terms)
    ce = np.cumsum(onehot, axis=1)[:, :-1]
    r = probs.shape[1]
    return float(np.mean(np.sum((cp - ce) ** 2, axis=1) / (r - 1)))


def _metrics(probs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    p_actual = probs[np.arange(len(y)), y].clip(EPS, 1.0)
    logloss = float(-np.mean(np.log(p_actual)))
    acc = float(np.mean(probs.argmax(axis=1) == y))
    return {"rps": _rps(probs, y), "brier": brier, "log_loss": logloss,
            "accuracy": acc, "n": int(len(y))}


def reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict[str, object]:
    """
    Pooled one-vs-rest calibration: treat each (match, outcome) pair as a binary
    prediction, bin by predicted probability, and compare mean predicted vs
    observed frequency. Returns a per-bin table and the Expected Calibration
    Error (ECE = sum over bins of (bin share) * |mean_pred - obs_freq|).
    """
    probs = np.asarray(probs)
    y = np.asarray(y)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    p = probs.ravel()
    o = onehot.ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)

    table, ece, N = [], 0.0, len(p)
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            table.append({"bin_low": float(edges[b]), "bin_high": float(edges[b + 1]),
                          "n": 0, "mean_pred": float("nan"), "obs_freq": float("nan")})
            continue
        mp, of = float(p[m].mean()), float(o[m].mean())
        ece += (n / N) * abs(mp - of)
        table.append({"bin_low": float(edges[b]), "bin_high": float(edges[b + 1]),
                      "n": n, "mean_pred": mp, "obs_freq": of})
    return {"table": table, "ece": float(ece), "n_points": N}


def plot_reliability(probs: np.ndarray, y: np.ndarray, path: str,
                     n_bins: int = 10) -> Optional[str]:
    """Save a reliability diagram as PNG. Needs matplotlib (optional)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[evaluate] matplotlib not installed -> skipping plot "
              "(pip install matplotlib).")
        return None
    rel = reliability(probs, y, n_bins)
    xs = [r["mean_pred"] for r in rel["table"] if r["n"] > 0]
    ys = [r["obs_freq"] for r in rel["table"] if r["n"] > 0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    ax.plot(xs, ys, "o-", label=f"model (ECE={rel['ece']:.3f})")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability diagram (one-vs-rest, pooled)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"reliability diagram -> {path}")
    return path


def walk_forward(df: pd.DataFrame,
                 params: Optional[GoalsParams] = None,
                 start_date: str = "2022-01-01",
                 end_date: Optional[str] = None,
                 home_advantage: float = 100.0,
                 regress_each_year: float = 0.0,
                 model: str = "elo",
                 k_factors: Optional[KFactors] = None,
                 ens_weight: float = 0.5) -> Dict[str, object]:
    """
    Walk-forward backtest. Warms up on everything before `start_date`, then scores
    every match in [start_date, end_date]. `model` selects the forecaster; the
    ordered-logit cutpoints are fit once on the warm-up period (leakage-free) and
    then held fixed while Elo keeps updating through the scoring window.

    `k_factors` overrides the Elo K-factor scheme (None = the default scheme).
    `ens_weight` is the weight on the Elo model in the ensemble blend (the
    ordered-logit gets 1 - ens_weight); ignored for the single-model runs.
    """
    if model not in ("elo", "ordered_logit", "ensemble"):
        raise ValueError(f"unknown model: {model}")
    params = params or GoalsParams(home_advantage=home_advantage)
    elo = EloRatings(home_advantage=home_advantage, k_factors=k_factors)
    elo_model = EloMatchModel(elo, params)
    needs_olr = model in ("ordered_logit", "ensemble")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else None

    warm_diff: List[float] = []
    warm_home: List[float] = []
    warm_y: List[int] = []
    olr_model: Optional[OrderedLogitModel] = None

    probs: List[List[float]] = []
    ys: List[int] = []
    last_year = None

    def elo_probs(h, a, neutral):
        return outcome_probs(elo_model.score_matrix(h, a, neutral=neutral))

    for row in df.itertuples(index=False):
        d = row.date
        neutral = bool(getattr(row, "neutral", False))
        in_window = d >= start and (end is None or d <= end)

        if regress_each_year > 0:
            if last_year is not None and d.year != last_year:
                elo._regress_to_mean(regress_each_year)
            last_year = d.year

        if needs_olr and not in_window:
            warm_diff.append(elo.rating(row.home_team) - elo.rating(row.away_team))
            warm_home.append(0.0 if neutral else 1.0)
            warm_y.append(_result_class(int(row.home_score), int(row.away_score)))

        if in_window:
            if needs_olr and olr_model is None:
                if len(warm_y) < 200:
                    raise ValueError("too few warm-up matches to fit ordered logit; "
                                     "use an earlier --start or more data")
                p_fit = fit_ordered_logit(warm_diff, warm_home, warm_y)
                p_fit.base_total_goals = params.base_total_goals
                p_fit.rho = params.rho
                olr_model = OrderedLogitModel(elo, p_fit)

            if model == "elo":
                p = elo_probs(row.home_team, row.away_team, neutral)
            elif model == "ordered_logit":
                p = olr_model.proba(row.home_team, row.away_team, neutral)
            else:  # ensemble (weighted average of the two probability vectors)
                pe = np.array(elo_probs(row.home_team, row.away_team, neutral))
                po = np.array(olr_model.proba(row.home_team, row.away_team, neutral))
                p = tuple(ens_weight * pe + (1.0 - ens_weight) * po)
            probs.append(list(p))
            ys.append(_result_class(int(row.home_score), int(row.away_score)))

        elo.update_match(row.home_team, row.away_team,
                         int(row.home_score), int(row.away_score),
                         getattr(row, "tournament", "Friendly"), neutral, d)

    probs_arr = np.array(probs)
    y_arr = np.array(ys)
    out: Dict[str, object] = {"model_name": model,
                              "model": _metrics(probs_arr, y_arr),
                              "reliability": reliability(probs_arr, y_arr)}

    base = np.bincount(y_arr, minlength=3) / len(y_arr)
    base_probs = np.tile(base, (len(y_arr), 1))
    out["baseline"] = _metrics(base_probs, y_arr)
    out["base_rates"] = {"home": float(base[0]), "draw": float(base[1]),
                         "away": float(base[2])}
    out["_probs"] = probs_arr
    out["_y"] = y_arr
    return out


# ---------------------------------------------------------------------------
# Goals-layer backtest, validate the *scoreline* predictions, not just W/D/L
# ---------------------------------------------------------------------------
# The W/D/L walk-forward says nothing about whether "2.3 expected goals" or
# "55% over 2.5" can be trusted. This evaluates the goals layer on its own terms,
# leakage-free, and, crucially, reports *calibration*: of the matches called
# ~60% over 2.5, did ~60% actually go over? A well-calibrated goals layer is the
# definition of "you can trust the number".

TOTAL_CAP = 7  # total-goals classes are 0,1,..,6,7+ (8 classes) for the log-loss


def _binary_scores(p, y) -> Dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return {"brier": brier, "log_loss": logloss, "obs_rate": float(y.mean()),
            "mean_pred": float(p.mean()), "n": int(len(y))}


def _binary_baseline(y) -> Dict[str, float]:
    """Always predict the in-window base rate, the bar the model must clear."""
    y = np.asarray(y, dtype=float)
    return _binary_scores(np.full(len(y), y.mean()), y)


def _binary_reliability(p, y, n_bins: int = 10) -> Dict[str, object]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    table, ece, N = [], 0.0, len(p)
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        mp, of = float(p[m].mean()), float(y[m].mean())
        ece += (n / N) * abs(mp - of)
        table.append({"bin_low": float(edges[b]), "bin_high": float(edges[b + 1]),
                      "n": n, "mean_pred": mp, "obs_freq": of})
    return {"table": table, "ece": float(ece)}


def walk_forward_goals(df: pd.DataFrame,
                       params: Optional[GoalsParams] = None,
                       start_date: str = "2022-01-01",
                       end_date: Optional[str] = None,
                       home_advantage: float = 100.0,
                       regress_each_year: float = 0.0,
                       model: str = "ensemble",
                       k_factors: Optional[KFactors] = None) -> Dict[str, object]:
    """
    Leakage-free walk-forward of the *goals* layer. For each scored match it
    builds the model's full score matrix and records over/under 2.5, both-teams-
    to-score, the predicted total-goals pmf, and expected total goals; then it
    scores those against what actually happened, with calibration (ECE) and a
    base-rate baseline for each. `model` is elo / ordered_logit / ensemble.
    """
    if model not in ("elo", "ordered_logit", "ensemble"):
        raise ValueError(f"unknown model: {model}")
    from .match_model import EnsembleMatchModel
    params = params or GoalsParams(home_advantage=home_advantage)
    elo = EloRatings(home_advantage=home_advantage, k_factors=k_factors)
    elo_model = EloMatchModel(elo, params)
    needs_olr = model in ("ordered_logit", "ensemble")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else None

    warm_diff: List[float] = []
    warm_home: List[float] = []
    warm_y: List[int] = []
    olr_model: Optional[OrderedLogitModel] = None
    last_year = None

    over_p: List[float] = []; over_y: List[int] = []
    btts_p: List[float] = []; btts_y: List[int] = []
    exp_tot: List[float] = []; act_tot: List[int] = []
    tg_pmf: List[np.ndarray] = []; tg_cls: List[int] = []

    def matrix(h, a, neutral):
        if model == "elo":
            return elo_model.score_matrix(h, a, neutral=neutral)
        if model == "ordered_logit":
            return olr_model.score_matrix(h, a, neutral=neutral)
        return EnsembleMatchModel([(elo_model, 0.5),
                                   (olr_model, 0.5)]).score_matrix(h, a, neutral=neutral)

    def capped_pmf(P):
        dist = total_goals_dist(P)
        pmf = np.zeros(TOTAL_CAP + 1)
        pmf[:TOTAL_CAP] = dist[:TOTAL_CAP]
        pmf[TOTAL_CAP] = dist[TOTAL_CAP:].sum()
        return pmf

    for row in df.itertuples(index=False):
        d = row.date
        neutral = bool(getattr(row, "neutral", False))
        in_window = d >= start and (end is None or d <= end)

        if regress_each_year > 0:
            if last_year is not None and d.year != last_year:
                elo._regress_to_mean(regress_each_year)
            last_year = d.year

        if needs_olr and not in_window:
            warm_diff.append(elo.rating(row.home_team) - elo.rating(row.away_team))
            warm_home.append(0.0 if neutral else 1.0)
            warm_y.append(_result_class(int(row.home_score), int(row.away_score)))

        if in_window:
            if needs_olr and olr_model is None:
                if len(warm_y) < 200:
                    raise ValueError("too few warm-up matches to fit ordered logit; "
                                     "use an earlier --start or more data")
                p_fit = fit_ordered_logit(warm_diff, warm_home, warm_y)
                p_fit.base_total_goals = params.base_total_goals
                p_fit.rho = params.rho
                olr_model = OrderedLogitModel(elo, p_fit)

            P = matrix(row.home_team, row.away_team, neutral)
            hs, as_ = int(row.home_score), int(row.away_score)
            tot = hs + as_
            over_p.append(prob_over(P, 2.5)); over_y.append(int(tot > 2.5))
            btts_p.append(prob_btts(P)); btts_y.append(int(hs >= 1 and as_ >= 1))
            eh, ea = expected_goals(P)
            exp_tot.append(eh + ea); act_tot.append(tot)
            tg_pmf.append(capped_pmf(P)); tg_cls.append(min(tot, TOTAL_CAP))

        elo.update_match(row.home_team, row.away_team,
                         int(row.home_score), int(row.away_score),
                         getattr(row, "tournament", "Friendly"), neutral, d)

    n = len(over_y)
    over_y_a = np.array(over_y); act_tot_a = np.array(act_tot)
    tg_pmf_a = np.clip(np.array(tg_pmf), EPS, 1.0)
    tg_cls_a = np.array(tg_cls)

    # total-goals log-loss vs the empirical (base-rate) total-goals distribution
    base_pmf = np.bincount(tg_cls_a, minlength=TOTAL_CAP + 1) / n
    model_tg_ll = float(-np.mean(np.log(tg_pmf_a[np.arange(n), tg_cls_a])))
    base_tg_ll = float(-np.mean(np.log(np.clip(base_pmf[tg_cls_a], EPS, 1.0))))

    return {
        "model_name": model, "n": n,
        "over25": {**_binary_scores(over_p, over_y),
                   "baseline": _binary_baseline(over_y),
                   "reliability": _binary_reliability(over_p, over_y)},
        "btts": {**_binary_scores(btts_p, btts_y),
                 "baseline": _binary_baseline(btts_y),
                 "reliability": _binary_reliability(btts_p, btts_y)},
        "total_goals": {
            "model_log_loss": model_tg_ll, "baseline_log_loss": base_tg_ll,
            "mean_pred": float(np.mean(exp_tot)),
            "mean_actual": float(act_tot_a.mean()),
            "bias": float(np.mean(exp_tot) - act_tot_a.mean()),
            "mae": float(np.mean(np.abs(np.array(exp_tot) - act_tot_a))),
        },
    }


def tournament_scorecard(df: pd.DataFrame,
                         model: str = "ensemble",
                         since: str = "2026-06-01",
                         params: Optional[GoalsParams] = None,
                         home_advantage: float = 100.0,
                         k_factors: Optional[KFactors] = None) -> Dict[str, object]:
    """
    Predicted-vs-actual report for already-played World Cup matches.

    Each match is forecast *leakage-free*: with the Elo state as it was the
    instant before kickoff (the ordered-logit cutpoints are fit once on
    everything before `since`). For each match we record what the model said
    (W/D/L probabilities, expected goals, most likely score) and what actually
    happened, plus the gap, the probability the model gave to the real result,
    the per-match RPS, and whether the model's favourite came through.
    """
    if model not in ("elo", "ordered_logit", "ensemble"):
        raise ValueError(f"unknown model: {model}")
    from .match_model import EnsembleMatchModel, expected_goals
    params = params or GoalsParams(home_advantage=home_advantage)
    elo = EloRatings(home_advantage=home_advantage, k_factors=k_factors)
    elo_model = EloMatchModel(elo, params)
    needs_olr = model in ("ordered_logit", "ensemble")
    since_ts = pd.Timestamp(since)

    warm_diff: List[float] = []; warm_home: List[float] = []; warm_y: List[int] = []
    olr_model: Optional[OrderedLogitModel] = None
    matches: List[dict] = []

    def matrix(h, a, neutral):
        if model == "elo":
            return elo_model.score_matrix(h, a, neutral=neutral)
        if model == "ordered_logit":
            return olr_model.score_matrix(h, a, neutral=neutral)
        return EnsembleMatchModel([(elo_model, 0.5),
                                   (olr_model, 0.5)]).score_matrix(h, a, neutral=neutral)

    for row in df.itertuples(index=False):
        d = row.date
        neutral = bool(getattr(row, "neutral", False))
        in_window = d >= since_ts
        label = str(getattr(row, "tournament", "")).lower()
        is_wc = "world cup" in label and "qual" not in label

        if needs_olr and not in_window:
            warm_diff.append(elo.rating(row.home_team) - elo.rating(row.away_team))
            warm_home.append(0.0 if neutral else 1.0)
            warm_y.append(_result_class(int(row.home_score), int(row.away_score)))

        if in_window and is_wc:
            if needs_olr and olr_model is None:
                if len(warm_y) < 200:
                    raise ValueError("too few warm-up matches to fit ordered logit")
                p_fit = fit_ordered_logit(warm_diff, warm_home, warm_y)
                p_fit.base_total_goals = params.base_total_goals
                p_fit.rho = params.rho
                olr_model = OrderedLogitModel(elo, p_fit)

            h, a = row.home_team, row.away_team
            eh_pre, ea_pre = elo.rating(h), elo.rating(a)
            P = matrix(h, a, neutral)
            probs = list(outcome_probs(P))               # [home, draw, away]
            xg_h, xg_a = expected_goals(P)
            flat = [(i, j, float(P[i, j])) for i in range(P.shape[0])
                    for j in range(P.shape[1])]
            top = max(flat, key=lambda t: t[2])
            hs, as_ = int(row.home_score), int(row.away_score)
            y = _result_class(hs, as_)
            cp = np.cumsum(probs)[:-1]                    # [p_home, p_home+p_draw]
            ce = np.array([1.0 if y == 0 else 0.0, 1.0 if y <= 1 else 0.0])
            rps_match = float(np.sum((cp - ce) ** 2) / 2.0)
            matches.append({
                "date": d.strftime("%Y-%m-%d"), "home": h, "away": a,
                "neutral": neutral, "elo_home": eh_pre, "elo_away": ea_pre,
                "pred": {"home": probs[0], "draw": probs[1], "away": probs[2]},
                "pred_xg": {"home": xg_h, "away": xg_a},
                "pred_score": {"home": top[0], "away": top[1], "prob": top[2]},
                "actual": {"home": hs, "away": as_, "outcome": y},
                "p_actual": float(probs[y]),            # prob the model gave reality
                "rps": rps_match,
                "favourite_correct": bool(int(np.argmax(probs)) == y),
            })

        elo.update_match(row.home_team, row.away_team,
                         int(row.home_score), int(row.away_score),
                         getattr(row, "tournament", "Friendly"), neutral, d)

    n = len(matches)
    summary = {
        "n": n,
        "mean_rps": float(np.mean([m["rps"] for m in matches])) if n else None,
        "mean_p_actual": float(np.mean([m["p_actual"] for m in matches])) if n else None,
        "favourite_hit_rate": float(np.mean([m["favourite_correct"]
                                             for m in matches])) if n else None,
    }
    return {"model": model, "since": since, "summary": summary, "matches": matches}


def fixture_predictions(played_df: pd.DataFrame, fixtures_df: pd.DataFrame,
                        model: str = "ensemble",
                        params: Optional[GoalsParams] = None,
                        home_advantage: float = 100.0,
                        k_factors: Optional[KFactors] = None) -> List[dict]:
    """
    Forecast a set of upcoming fixtures (who-plays-whom, no score yet) with the
    current model: who wins, the W/D/L probabilities, expected goals and the most
    likely scoreline for each. `played_df` is the scored history the ratings are
    built from; `fixtures_df` has columns date/home_team/away_team/neutral
    (see `data.upcoming_fixtures`).
    """
    if model not in ("elo", "ordered_logit", "ensemble"):
        raise ValueError(f"unknown model: {model}")
    from .match_model import EnsembleMatchModel, expected_goals
    params = params or GoalsParams(home_advantage=home_advantage)
    elo = EloRatings(home_advantage=home_advantage, k_factors=k_factors).fit(played_df)
    elo_model = EloMatchModel(elo, params)
    olr_model = None
    if model in ("ordered_logit", "ensemble"):
        olr_model = OrderedLogitModel.fit_from_history(
            played_df, home_advantage=home_advantage,
            base_total_goals=params.base_total_goals, rho=params.rho)

    def matrix(h, a, neutral):
        if model == "elo":
            return elo_model.score_matrix(h, a, neutral=neutral)
        if model == "ordered_logit":
            return olr_model.score_matrix(h, a, neutral=neutral)
        return EnsembleMatchModel([(elo_model, 0.5),
                                   (olr_model, 0.5)]).score_matrix(h, a, neutral=neutral)

    out = []
    for r in fixtures_df.itertuples(index=False):
        h, a = r.home_team, r.away_team
        neutral = bool(getattr(r, "neutral", False))
        P = matrix(h, a, neutral)
        probs = list(outcome_probs(P))                  # [home, draw, away]
        xg_h, xg_a = expected_goals(P)
        flat = [(i, j, float(P[i, j])) for i in range(P.shape[0])
                for j in range(P.shape[1])]
        top = max(flat, key=lambda t: t[2])
        idx = int(np.argmax(probs))
        favourite = "Gelijkspel" if idx == 1 else (h if idx == 0 else a)
        out.append({
            "date": pd.Timestamp(r.date).strftime("%Y-%m-%d"),
            "home": h, "away": a, "neutral": neutral,
            "elo_home": elo.rating(h), "elo_away": elo.rating(a),
            "pred": {"home": probs[0], "draw": probs[1], "away": probs[2]},
            "pred_xg": {"home": xg_h, "away": xg_a},
            "pred_score": {"home": top[0], "away": top[1], "prob": top[2]},
            "favourite": favourite, "favourite_prob": float(max(probs)),
        })
    return out


def tune_goals_params(df: pd.DataFrame,
                      start_date: str = "2022-01-01",
                      sup_grid=(0.30, 0.40, 0.50),
                      total_grid=(2.4, 2.6, 2.8),
                      rho_grid=(-0.12, -0.08, 0.0),
                      hfa_grid=(80.0, 100.0, 120.0),
                      regress_grid=(0.0, 0.05),
                      metric: str = "rps") -> Tuple[Dict, Dict]:
    """
    Small grid search over the goals-bridge knobs (Elo model). Returns
    (best_config, best_metrics). Default scoring metric is RPS; default grid is
    modest (162 fits), widen as you like.
    """
    best_cfg, best_metrics, best_score = None, None, np.inf
    for sup, tot, rho, hfa, reg in product(
        sup_grid, total_grid, rho_grid, hfa_grid, regress_grid
    ):
        params = GoalsParams(base_total_goals=tot, sup_per_100_elo=sup,
                             rho=rho, home_advantage=hfa)
        res = walk_forward(df, params=params, start_date=start_date,
                           home_advantage=hfa, regress_each_year=reg, model="elo")
        score = res["model"][metric]
        if score < best_score:
            best_score = score
            best_cfg = {**asdict(params), "regress_each_year": reg}
            best_metrics = res["model"]
    return best_cfg, best_metrics
