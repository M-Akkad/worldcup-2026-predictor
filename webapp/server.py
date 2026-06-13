"""
FastAPI backend (run via `python run_web.py`): serves the API under /api/* and
the frontend from webapp/static/. SQLite is the source of truth (storage.py;
auto-imports data/results.csv when empty). Simulations run on background threads
and the UI polls /api/jobs/{id}; each one conditions on already-played 2026 WC
matches and optionally applies the stored Elo adjustments.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from wc2026 import data as wc_data
from wc2026 import elo as wc_elo
from wc2026 import evaluate, monte_carlo
from wc2026.match_model import (EloMatchModel, EnsembleMatchModel, GoalsParams,
                                OrderedLogitModel, blend_with_market,
                                expected_goals, outcome_probs, prob_btts, prob_over)
from . import storage

app = FastAPI(title="WC2026 Odds Engine", version="1.0")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
DEFAULT_CSV = os.path.join(storage.ROOT, "data", "results.csv")

_jobs: dict = {}
_jobs_lock = threading.Lock()

MODELS = ("elo", "ordered_logit", "ensemble")


# ------------------------------------------------------------ helpers
def _matches_df():
    df = storage.fetch_matches_df()
    if df.empty:
        raise HTTPException(status_code=409, detail=(
            "Nog geen wedstrijddata. Importeer eerst de Kaggle-CSV via "
            "POST /api/import of zet hem klaar als data/results.csv."))
    return df


def _adjusted_elo(df, use_adjustments: bool, hfa: float = 100.0):
    elo = wc_elo.build_from_results(df, home_advantage=hfa)
    applied = []
    if use_adjustments:
        for a in storage.get_adjustments():
            elo.set_rating(a["team"], elo.rating(a["team"]) + a["delta"])
            applied.append(a)
    return elo, applied


def _build_model(name: str, df, elo, use_adjustments: bool):
    params = GoalsParams()
    if name == "elo":
        return EloMatchModel(elo, params)
    olr = OrderedLogitModel.fit_from_history(df)
    if use_adjustments:
        for a in storage.get_adjustments():
            olr.elo.set_rating(a["team"], olr.elo.rating(a["team"]) + a["delta"])
    if name == "ordered_logit":
        return olr
    return EnsembleMatchModel([(EloMatchModel(elo, params), 0.5), (olr, 0.5)])


# ------------------------------------------------------------- schemas
class SimulateBody(BaseModel):
    model: str = "ensemble"
    n_sims: int = Field(default=10000, ge=100, le=200000)
    seed: int = 0
    use_adjustments: bool = True


class TuneBody(BaseModel):
    model: str = "ensemble"
    n_trials: int = Field(default=40, ge=5, le=300)
    metric: str = "rps"
    start: str = "2022-01-01"


class ResultBody(BaseModel):
    home: str
    away: str
    home_score: int = Field(ge=0, le=30)
    away_score: int = Field(ge=0, le=30)
    date: Optional[str] = None
    neutral: bool = True
    tournament: str = "FIFA World Cup"


class AdjustmentBody(BaseModel):
    team: str
    delta: float = Field(ge=-100, le=100)
    reason: str = ""


class ImportBody(BaseModel):
    path: str = DEFAULT_CSV


# ------------------------------------------------------------ lifecycle
@app.on_event("startup")
def _startup():
    storage.init_db()
    if storage.match_count() == 0 and os.path.exists(DEFAULT_CSV):
        stats = storage.import_csv(DEFAULT_CSV)
        print(f"[startup] imported {stats['added']} matches from {DEFAULT_CSV}")
    if os.environ.get("WC_AUTO_FETCH") == "1":
        t = threading.Thread(target=_auto_fetch_loop, daemon=True)
        t.start()
        print("[startup] auto-fetch enabled (every 6h)")


def _auto_fetch_loop():
    while True:
        try:
            fetched = wc_data.fetch_openfootball_results()
            if len(fetched):
                stats = storage.insert_matches(fetched)
                if stats["added"]:
                    print(f"[auto-fetch] +{stats['added']} new results")
        except Exception as e:  # pragma: no cover
            print(f"[auto-fetch] failed: {e}")
        time.sleep(6 * 3600)


# ------------------------------------------------------------- routes
@app.get("/api/status")
def status():
    df = storage.fetch_matches_df()
    latest = storage.latest_run()
    fixed_group, fixed_ko = ([], {})
    if len(df):
        fixed_group, fixed_ko = wc_data.extract_played_tournament_matches(df)
    return {
        "matches": int(len(df)),
        "last_match_date": df["date"].max().strftime("%Y-%m-%d") if len(df) else None,
        "played_group_matches": len(fixed_group),
        "decided_ko_matches": len(fixed_ko),
        "adjustments": len(storage.get_adjustments()),
        "latest_run": ({"id": latest["id"], "created_at": latest["created_at"],
                        "model": latest["model"], "n_sims": latest["n_sims"]}
                       if latest else None),
        "tournament_start": "2026-06-11",
    }


@app.get("/api/teams")
def teams():
    return {"teams": sorted(wc_data.all_tournament_teams()),
            "groups": wc_data.GROUPS_2026, "hosts": sorted(wc_data.HOST_TEAMS)}


@app.get("/api/groups")
def groups():
    latest = storage.latest_run()
    probs = {r["team"]: r for r in (latest["table"] if latest else [])}
    df = storage.fetch_matches_df()
    played, _ = (wc_data.extract_played_tournament_matches(df)
                 if len(df) else ([], {}))
    played_by_group = {}
    team_group = {t: g for g, ts in wc_data.GROUPS_2026.items() for t in ts}
    for h, a, hs, as_ in played:
        played_by_group.setdefault(team_group[h], []).append(
            {"home": h, "away": a, "home_score": hs, "away_score": as_})
    out = []
    for g, ts in wc_data.GROUPS_2026.items():
        out.append({"group": g,
                    "teams": [{"team": t,
                               "host": t in wc_data.HOST_TEAMS,
                               **{k: probs.get(t, {}).get(k) for k in
                                  ("elo", "win_group", "reach_ko", "win_title")}}
                              for t in ts],
                    "played": played_by_group.get(g, [])})
    return {"groups": out, "has_run": latest is not None}


@app.get("/api/runs/latest")
def runs_latest():
    latest = storage.latest_run()
    if latest is None:
        raise HTTPException(404, "Nog geen simulatie gedraaid.")
    return latest


@app.get("/api/runs/history")
def runs_history(limit: int = 60):
    return {"runs": storage.run_history(limit)}


@app.get("/api/predict")
def predict(home: str, away: str, model: str = "ensemble",
            neutral: bool = True, odds: Optional[str] = None,
            market_weight: float = 0.5, use_adjustments: bool = True):
    if model not in MODELS:
        raise HTTPException(400, f"model moet een van {MODELS} zijn")
    df = _matches_df()
    h = wc_data.normalize_team(home)
    a = wc_data.normalize_team(away)
    elo, _ = _adjusted_elo(df, use_adjustments)
    m = _build_model(model, df, elo, use_adjustments)
    P = m.score_matrix(h, a, neutral=neutral)
    ph, pdr, pa = outcome_probs(P)

    market = None
    if odds:
        from wc2026.market import devig_shin, parse_odds
        try:
            market = [float(x) for x in devig_shin(parse_odds(odds))]
        except ValueError as e:
            raise HTTPException(400, str(e))
        ph, pdr, pa = blend_with_market((ph, pdr, pa), tuple(market),
                                        market_weight=market_weight)

    flat = [((i, j), float(P[i, j])) for i in range(P.shape[0])
            for j in range(P.shape[1])]
    flat.sort(key=lambda x: -x[1])
    eh, ea = expected_goals(P)
    return {
        "home": h, "away": a, "neutral": neutral, "model": model,
        "elo": {"home": elo.rating(h), "away": elo.rating(a)},
        "probs": {"home": ph, "draw": pdr, "away": pa},
        "market": market,
        "goals": {
            "exp_home": eh, "exp_away": ea, "exp_total": eh + ea,
            "over25": prob_over(P, 2.5), "over15": prob_over(P, 1.5),
            "over35": prob_over(P, 3.5), "btts": prob_btts(P),
        },
        "top_scorelines": [{"home_goals": i, "away_goals": j, "prob": p}
                           for (i, j), p in flat[:8]],
        "matrix": [[float(P[i, j]) for j in range(7)] for i in range(7)],
    }


@app.post("/api/import")
def import_data(body: ImportBody):
    if not os.path.exists(body.path):
        raise HTTPException(404, f"bestand niet gevonden: {body.path}")
    return storage.import_csv(body.path)


@app.post("/api/update/fetch")
def update_fetch():
    try:
        fetched = wc_data.fetch_openfootball_results()
    except Exception as e:
        raise HTTPException(502, f"openfootball niet bereikbaar: {e}")
    if len(fetched) == 0:
        return {"added": 0, "skipped": 0, "total": storage.match_count(),
                "note": "Upstream heeft nog geen uitslagen (wordt ~dagelijks bijgewerkt)."}
    return storage.insert_matches(fetched)


@app.post("/api/update/result")
def update_result(body: ResultBody):
    import pandas as pd
    date = body.date or pd.Timestamp.today().strftime("%Y-%m-%d")
    df = pd.DataFrame([{
        "date": date, "home_team": body.home, "away_team": body.away,
        "home_score": body.home_score, "away_score": body.away_score,
        "tournament": body.tournament, "neutral": body.neutral,
    }])
    return storage.insert_matches(df)


@app.get("/api/adjustments")
def adjustments_list():
    return {"adjustments": storage.get_adjustments()}


@app.post("/api/adjustments")
def adjustments_set(body: AdjustmentBody):
    storage.set_adjustment(body.team, body.delta, body.reason)
    return {"ok": True, "adjustments": storage.get_adjustments()}


@app.delete("/api/adjustments/{team}")
def adjustments_delete(team: str):
    storage.delete_adjustment(team)
    return {"ok": True, "adjustments": storage.get_adjustments()}


@app.get("/api/scorecard")
def scorecard(model: str = "ensemble", since: str = "2026-06-01"):
    if model not in MODELS:
        raise HTTPException(400, f"model moet een van {MODELS} zijn")
    df = _matches_df()
    return evaluate.tournament_scorecard(df, model=model, since=since)


@app.get("/api/fixtures")
def fixtures(model: str = "ensemble", since: Optional[str] = None, limit: int = 0):
    if model not in MODELS:
        raise HTTPException(400, f"model moet een van {MODELS} zijn")
    df = _matches_df()
    if not os.path.exists(DEFAULT_CSV):
        return {"model": model, "fixtures": []}
    fx = wc_data.upcoming_fixtures(DEFAULT_CSV, since=since)
    # hide fixtures whose result is already in the DB (e.g. entered manually
    # before the free CSV feed caught up), keyed on date + the unordered team
    # pair so an unrelated historical friendly never masks a real fixture
    played = {(r.date.strftime("%Y-%m-%d"), frozenset((r.home_team, r.away_team)))
              for r in df.itertuples()}
    if len(fx):
        fx = fx[~fx.apply(lambda r: (r["date"].strftime("%Y-%m-%d"),
                                     frozenset((r["home_team"], r["away_team"])))
                          in played, axis=1)]
    if limit and limit > 0:
        fx = fx.head(limit)
    preds = evaluate.fixture_predictions(df, fx, model=model)
    return {"model": model, "fixtures": preds}


@app.get("/api/backtest")
def backtest(model: str = "elo", start: str = "2022-01-01"):
    if model not in MODELS:
        raise HTTPException(400, f"model moet een van {MODELS} zijn")
    df = _matches_df()
    res = evaluate.walk_forward(df, start_date=start, model=model)
    return {"model_name": model, "start": start,
            "model": res["model"], "baseline": res["baseline"],
            "ece": res["reliability"]["ece"], "base_rates": res["base_rates"]}


# ------------------------------------------------------- simulation jobs
def _run_simulation(job_id: str, body: SimulateBody):
    try:
        df = storage.fetch_matches_df()
        elo, applied = _adjusted_elo(df, body.use_adjustments)
        model = _build_model(body.model, df, elo, body.use_adjustments)
        fixed_group, fixed_ko = wc_data.extract_played_tournament_matches(df)
        table = monte_carlo.run(model, n_sims=body.n_sims, seed=body.seed,
                                elo=elo, fixed_group=fixed_group or None,
                                fixed_ko=fixed_ko or None)
        run_id = storage.save_run(
            body.model, body.n_sims, body.seed, bool(applied),
            len(fixed_group), len(fixed_ko), table,
            meta={"adjustments": [a["team"] for a in applied]})
        with _jobs_lock:
            _jobs[job_id] = {"status": "done", "run_id": run_id}
    except Exception as e:  # pragma: no cover
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "error": str(e)}


# ------------------------------------------------------- tuning jobs
def _run_tune(job_id: str, body: TuneBody):
    try:
        from wc2026 import optimize as wc_opt
        df = storage.fetch_matches_df()

        def _progress(done, total, best):
            with _jobs_lock:
                _jobs[job_id] = {"status": "running", "done": done,
                                 "total": total, "best": best,
                                 "metric": body.metric}

        res = wc_opt.optimize(df, model=body.model, n_trials=body.n_trials,
                              metric=body.metric, start_date=body.start,
                              progress=_progress)
        with _jobs_lock:
            _jobs[job_id] = {
                "status": "done", "model": res.model, "metric": res.metric,
                "n_trials": res.n_trials, "improved": res.improved,
                "best_value": res.best_value, "baseline_value": res.baseline_value,
                "best_config": res.best_config,
                "best_metrics": res.best_metrics,
                "baseline_metrics": res.baseline_metrics,
            }
    except Exception as e:  # pragma: no cover
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "error": str(e)}


@app.post("/api/simulate")
def simulate(body: SimulateBody):
    if body.model not in MODELS:
        raise HTTPException(400, f"model moet een van {MODELS} zijn")
    _matches_df()  # 409 when empty
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"status": "running"}
    threading.Thread(target=_run_simulation, args=(job_id, body),
                     daemon=True).start()
    return {"job_id": job_id}


@app.post("/api/tune")
def tune(body: TuneBody):
    if body.model not in MODELS:
        raise HTTPException(400, f"model moet een van {MODELS} zijn")
    if body.metric not in ("rps", "log_loss", "brier"):
        raise HTTPException(400, "metric moet rps, log_loss of brier zijn")
    try:
        import optuna  # noqa: F401
    except ImportError:
        raise HTTPException(503, "optuna is niet geïnstalleerd "
                                 "(pip install optuna)")
    _matches_df()  # 409 when empty
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "done": 0,
                         "total": body.n_trials, "metric": body.metric}
    threading.Thread(target=_run_tune, args=(job_id, body),
                     daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "onbekende job")
    return job


# ------------------------------------------------------------- frontend
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
