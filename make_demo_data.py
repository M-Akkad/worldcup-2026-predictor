"""
make_demo_data.py, generate a synthetic results file matching the schema of
the Kaggle dataset 'International football results from 1872 to 2026'.

This lets you run the entire pipeline end-to-end without downloading anything.
Each team gets a hidden 'true strength'; matches are simulated with Poisson
goals so that stronger teams win more and a sensible favourite ordering emerges.

Usage:  python make_demo_data.py            # writes data/demo_results.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from wc2026 import data as wc_data

EXTRA_TEAMS = [
    "Italy", "Nigeria", "Cameroon", "Serbia", "Denmark", "Poland", "Chile",
    "Peru", "Venezuela", "Costa Rica", "Mali", "Burkina Faso", "Ukraine",
    "Romania", "Greece", "Hungary", "Slovakia", "Slovenia", "Wales", "Ireland",
    "Russia", "China", "United Arab Emirates", "Honduras", "Jamaica",
    "El Salvador", "Bolivia", "Guinea", "Cameroon", "Zambia", "Israel",
]


def generate(out_path: str = "data/demo_results.csv", seed: int = 7,
             start_year: int = 2012, n_matches: int = 12000) -> str:
    rng = np.random.default_rng(seed)
    teams = sorted(set(wc_data.all_tournament_teams()) | set(EXTRA_TEAMS))

    # hidden strengths on a ~Elo-like scale, with a few clear elites
    strength = {t: rng.normal(0.0, 0.45) for t in teams}
    for elite, val in {"Spain": 1.5, "France": 1.45, "Argentina": 1.4,
                       "England": 1.35, "Brazil": 1.3, "Germany": 1.2,
                       "Portugal": 1.2, "Netherlands": 1.15}.items():
        strength[elite] = val

    dates = pd.to_datetime(
        rng.integers(pd.Timestamp(f"{start_year}-01-01").value // 10**9,
                     pd.Timestamp("2026-05-01").value // 10**9, n_matches),
        unit="s",
    )
    base, k, hfa = 1.35, 0.9, 0.35
    rows = []
    tourneys = (["Friendly"] * 5 + ["FIFA World Cup qualification"] * 3
                + ["UEFA Nations League"] + ["UEFA Euro"])
    for d in dates:
        h, a = rng.choice(teams, size=2, replace=False)
        neutral = bool(rng.random() < 0.25)
        sh, sa = strength[h], strength[a]
        adv = 0.0 if neutral else hfa
        lh = base * np.exp(k * (sh - sa + adv) / 2)
        la = base * np.exp(k * (sa - sh - adv) / 2)
        hg, ag = int(rng.poisson(lh)), int(rng.poisson(la))
        rows.append((d.date().isoformat(), h, a, hg, ag,
                     str(rng.choice(tourneys)), "", "", neutral))

    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                     "home_score", "away_score", "tournament",
                                     "city", "country", "neutral"])
    df = df.sort_values("date")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} synthetic matches -> {out_path}")
    return out_path


if __name__ == "__main__":
    generate()
