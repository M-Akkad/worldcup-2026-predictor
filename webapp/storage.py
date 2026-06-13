"""
SQLite persistence for the web app.

Tables:
  matches      -- the results history (source of truth for the web app);
                  unique on (date, home_team, away_team) so imports are idempotent
  runs         -- one row per Monte Carlo simulation run (model, config, counts)
  run_probs    -- per-team probabilities for each run (powers the history chart)
  adjustments  -- bounded Elo deltas (manual or LLM), applied at simulation time

The CLI keeps working on CSV; the web app reads/writes this database. Import a
CSV once via storage.import_csv (the server does it automatically on first
start when the database is empty and data/results.csv exists).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from wc2026 import data as wc_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("WC_DB", os.path.join(ROOT, "data", "wc2026.db"))

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
  id INTEGER PRIMARY KEY,
  date TEXT NOT NULL,
  home_team TEXT NOT NULL,
  away_team TEXT NOT NULL,
  home_score INTEGER NOT NULL,
  away_score INTEGER NOT NULL,
  tournament TEXT NOT NULL DEFAULT 'Friendly',
  city TEXT DEFAULT '',
  country TEXT DEFAULT '',
  neutral INTEGER NOT NULL DEFAULT 0,
  UNIQUE(date, home_team, away_team)
);
CREATE TABLE IF NOT EXISTS adjustments (
  team TEXT PRIMARY KEY,
  delta REAL NOT NULL,
  reason TEXT DEFAULT '',
  source TEXT DEFAULT 'manual',
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY,
  created_at REAL NOT NULL,
  model TEXT NOT NULL,
  n_sims INTEGER NOT NULL,
  seed INTEGER NOT NULL,
  used_adjustments INTEGER NOT NULL DEFAULT 0,
  conditioned_group INTEGER NOT NULL DEFAULT 0,
  conditioned_ko INTEGER NOT NULL DEFAULT 0,
  meta TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS run_probs (
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  team TEXT NOT NULL,
  grp TEXT NOT NULL,
  elo REAL,
  win_group REAL, reach_ko REAL, reach_r16 REAL, reach_qf REAL,
  reach_sf REAL, reach_final REAL, win_title REAL,
  PRIMARY KEY (run_id, team)
);
"""


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init_db() -> None:
    with _lock, _conn() as c:
        c.executescript(_SCHEMA)


# ---------------------------------------------------------------- matches
def insert_matches(df: pd.DataFrame) -> Dict[str, int]:
    """
    Insert or refresh match rows, idempotent on (date, home, away). A new match
    is inserted; an existing match whose score or venue changed is *updated* (so a
    mistyped result can be corrected by re-entering it); an unchanged duplicate is
    left as-is. Team names are normalised.

    Returns {"added": n, "updated": n, "skipped": n, "total": rows}.
    """
    rows = []
    for r in df.itertuples(index=False):
        rows.append((
            pd.Timestamp(r.date).strftime("%Y-%m-%d"),
            wc_data.normalize_team(str(r.home_team)),
            wc_data.normalize_team(str(r.away_team)),
            int(r.home_score), int(r.away_score),
            str(getattr(r, "tournament", "Friendly")),
            str(getattr(r, "city", "") or ""),
            str(getattr(r, "country", "") or ""),
            1 if bool(getattr(r, "neutral", False)) else 0,
        ))
    with _lock, _conn() as c:
        before = c.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        c.executemany(
            "INSERT INTO matches "
            "(date, home_team, away_team, home_score, away_score, tournament,"
            " city, country, neutral) VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(date, home_team, away_team) DO UPDATE SET "
            "  home_score=excluded.home_score, away_score=excluded.away_score, "
            "  neutral=excluded.neutral, tournament=excluded.tournament "
            "WHERE matches.home_score IS NOT excluded.home_score "
            "   OR matches.away_score IS NOT excluded.away_score "
            "   OR matches.neutral IS NOT excluded.neutral", rows)
        after = c.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        changed = c.total_changes          # inserts + rows actually updated
    added = after - before
    updated = changed - added
    return {"added": added, "updated": updated,
            "skipped": len(rows) - added - updated, "total": after}


def import_csv(path: str) -> Dict[str, int]:
    return insert_matches(wc_data.load_results(path))


def fetch_matches_df() -> pd.DataFrame:
    with _lock, _conn() as c:
        df = pd.read_sql_query(
            "SELECT date, home_team, away_team, home_score, away_score,"
            " tournament, city, country, neutral FROM matches ORDER BY date", c)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
        df["neutral"] = df["neutral"].astype(bool)
    return df


def match_count() -> int:
    with _lock, _conn() as c:
        return int(c.execute("SELECT COUNT(*) FROM matches").fetchone()[0])


def recent_wc_matches(limit: int = 30) -> List[dict]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT date, home_team, away_team, home_score, away_score, neutral"
            " FROM matches WHERE date >= '2026-06-01' AND tournament LIKE"
            " '%World Cup%' AND tournament NOT LIKE '%qual%'"
            " ORDER BY date DESC, id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------- adjustments
def set_adjustment(team: str, delta: float, reason: str = "",
                   source: str = "manual") -> None:
    team = wc_data.normalize_team(team)
    delta = max(-100.0, min(100.0, float(delta)))
    with _lock, _conn() as c:
        c.execute("INSERT INTO adjustments (team, delta, reason, source, updated_at)"
                  " VALUES (?,?,?,?,?) ON CONFLICT(team) DO UPDATE SET"
                  " delta=excluded.delta, reason=excluded.reason,"
                  " source=excluded.source, updated_at=excluded.updated_at",
                  (team, delta, reason, source, time.time()))


def delete_adjustment(team: str) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM adjustments WHERE team = ?",
                  (wc_data.normalize_team(team),))


def get_adjustments() -> List[dict]:
    with _lock, _conn() as c:
        rows = c.execute("SELECT team, delta, reason, source, updated_at"
                         " FROM adjustments ORDER BY ABS(delta) DESC").fetchall()
    return [dict(r) for r in rows]


# -------------------------------------------------------------------- runs
_PROB_COLS = ["win_group", "reach_ko", "reach_r16", "reach_qf",
              "reach_sf", "reach_final", "win_title"]


def save_run(model: str, n_sims: int, seed: int, used_adjustments: bool,
             conditioned_group: int, conditioned_ko: int,
             table: pd.DataFrame, meta: Optional[dict] = None) -> int:
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO runs (created_at, model, n_sims, seed,"
            " used_adjustments, conditioned_group, conditioned_ko, meta)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (time.time(), model, int(n_sims), int(seed),
             1 if used_adjustments else 0, int(conditioned_group),
             int(conditioned_ko), json.dumps(meta or {})))
        run_id = cur.lastrowid
        rows = []
        for r in table.itertuples(index=False):
            rows.append((run_id, r.team, r.group,
                         float(getattr(r, "elo", 0.0) or 0.0),
                         *[float(getattr(r, k)) for k in _PROB_COLS]))
        c.executemany(
            "INSERT INTO run_probs (run_id, team, grp, elo, win_group, reach_ko,"
            " reach_r16, reach_qf, reach_sf, reach_final, win_title)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    return int(run_id)


def latest_run() -> Optional[dict]:
    with _lock, _conn() as c:
        run = c.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        if run is None:
            return None
        probs = c.execute("SELECT * FROM run_probs WHERE run_id = ?"
                          " ORDER BY win_title DESC", (run["id"],)).fetchall()
    out = dict(run)
    out["meta"] = json.loads(out.get("meta") or "{}")
    out["table"] = [dict(p) for p in probs]
    return out


def run_history(limit: int = 60) -> List[dict]:
    """Per run: timestamp + {team: win_title}, oldest first (for the chart)."""
    with _lock, _conn() as c:
        runs = c.execute("SELECT id, created_at, model, n_sims FROM runs"
                         " ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for run in runs:
            probs = c.execute("SELECT team, win_title FROM run_probs"
                              " WHERE run_id = ?", (run["id"],)).fetchall()
            out.append({"id": run["id"], "created_at": run["created_at"],
                        "model": run["model"], "n_sims": run["n_sims"],
                        "win_title": {p["team"]: p["win_title"] for p in probs}})
    return list(reversed(out))
