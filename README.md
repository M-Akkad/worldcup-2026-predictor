# WorldCup 2026 Predictor

Transparent, tunable probabilities for the **2026 FIFA World Cup**, both
per-match odds and whole-tournament title/advancement chances, backed by
honest, leakage-free evaluation. Ships with a CLI and a small FastAPI + SQLite
web app (bilingual UI, NL/EN).

> **Reality check.** No public model beats the betting market, draws are
> genuinely hard, and a single tournament is high-variance. Treat any one number
> here as a single draw from a wide distribution, not a prophecy.

---

## What it does

- **Match model**, a recency-weighted **Elo** rating turned into a full 11×11
  **scoreline distribution** via a Poisson + **Dixon–Coles** low-score
  correction, with a results-based **ordered-logit** alternative and an
  **ensemble** of the two.
- **Tournament simulation**, a **Monte Carlo** engine plays the real 2026
  bracket (12 groups of 4, FIFA tiebreakers, 8 best third-placed teams, R32 →
  final) tens of thousands of times. Hosts (USA/Canada/Mexico) get home
  advantage; already-played matches are pinned to reality.
- **Honest evaluation**, leakage-free **walk-forward backtesting** scored with
  the **ranked probability score** (the field standard for ordinal W/D/L
  outcomes), log-loss, Brier, plus a reliability diagram / **ECE** for
  calibration. A separate **goals backtest** validates over/under and
  both-teams-to-score.
- **Tuning**, an **Optuna** search calibrates the whole pipeline (Elo
  K-factors, home advantage, the goals bridge, ensemble weight) against the
  backtest.
- **Optional LLM context**, bounded Elo nudges from news (injuries, form) via
  any OpenAI-compatible LLM endpoint; context that adjusts, never dominates.

Current ensemble accuracy (walk-forward from 2022, real data): **RPS 0.171**
vs a 0.228 base-rate baseline, well calibrated (ECE ≈ 0.009).

---

## Quickstart

```bash
# 1. clone
git clone https://github.com/M-Akkad/worldcup-2026-predictor.git
cd worldcup-2026-predictor

# 2. install
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. get data, the real Kaggle CSV (below) or a synthetic stand-in:
python make_demo_data.py                              # writes data/results.csv

# 4. run the web app  ->  http://localhost:8000
python run_web.py
```

Only `numpy`, `pandas`, `scipy`, `fastapi`, `uvicorn` are required.
`scikit-learn`, `optuna`, `matplotlib` are optional (calibration, tuning, plots).
The optional LLM layer needs no extra package (it uses the standard library).

### Get the data (you download it yourself)

The match dataset is **not bundled** with this repo, you fetch it yourself under
its own licence. Two options:

**A. Real data (recommended).** Download Kaggle's *International football results
from 1872 to 2026* by **martj42** and save it as `data/results.csv`. Either grab
the zip from the dataset page and unzip `results.csv` into `data/`, or use the
Kaggle CLI (needs a free Kaggle API token):

```bash
pip install kaggle
kaggle datasets download -d martj42/international-football-results-from-1872-to-2017 -p data --unzip
# confirm the file landed at data/results.csv (check the slug on the dataset page)
```

**B. No Kaggle account?** Generate a synthetic stand-in to try everything end to
end (numbers are indicative, not real):

```bash
python make_demo_data.py        # writes data/results.csv
```

Once `data/results.csv` exists, live scores and the 2026 schedule are pulled
free (no API key) from the openfootball feed, see *Keeping it current*.

---

## The web app

Five tabs, served from `webapp/static` (vanilla JS, no build step). The UI is
bilingual; toggle between Dutch and English with the **EN/NL** button top-right.

| Tab | What it shows |
|-----|---------------|
| **Dashboard** | Title odds, the full 48-team advancement table, a title-race-over-time chart |
| **Groups** | The twelve groups with per-team group-win chances |
| **Results** | Upcoming fixtures with predictions **and** a predicted-vs-actual scorecard for played matches |
| **Predictor** | One-off match predictor: W/D/L, expected goals, over/under, BTTS, scoreline heatmap, optional bookmaker-odds blend |
| **Data & model** | Fetch/enter results, news adjustments, run a simulation, backtest, and run the Optuna tuner |

Simulations run as background jobs; the UI polls for completion.

---

## Command line

```bash
python run.py simulate  --data data/results.csv --n-sims 20000 --model ensemble
python run.py fixtures  --data data/results.csv --model ensemble      # predict upcoming matches
python run.py scorecard --data data/results.csv --model ensemble      # predicted vs actual (played)
python run.py predict   "Netherlands" "Brazil"  --data data/results.csv
python run.py backtest  --data data/results.csv --start 2022-01-01 --model ensemble
python run.py backtest  --data data/results.csv --model elo --goals   # validate the goals layer
python run.py tune      --data data/results.csv --optuna --model ensemble --trials 40
python run.py update    --data data/results.csv --fetch               # free live-results feed
```

`--model` is one of `elo`, `ordered_logit`, `ensemble`. Use `elo` for goal
markets (best-calibrated on goals); `ensemble` is sharpest on W/D/L.

---

## How it works

Three layers:

1. **Match models** (`wc2026/match_model.py`), every model returns a normalised
   11×11 score matrix. `EloMatchModel` maps an Elo gap to goal rates → Poisson →
   Dixon–Coles; `OrderedLogitModel` is a proportional-odds logit on the Elo gap
   (made simulator-compatible via an inverse goals bridge); `EnsembleMatchModel`
   averages them. `elo.py` builds the ratings (importance-weighted K-factors,
   goal-difference multiplier, home advantage).
2. **Tournament** (`tournament.py`, `monte_carlo.py`), exact 2026 rules and
   FIFA tiebreakers; conditions on already-played matches found in the data.
3. **Web app** (`webapp/`), SQLite is the source of truth (`storage.py`),
   FastAPI serves the API and runs simulations on background threads
   (`server.py`), the frontend is dependency-free.

Supporting modules: `evaluate.py` (walk-forward backtests + goals validation +
predicted-vs-actual), `optimize.py` (Optuna), `calibration.py`, `market.py`
(Shin de-vig), `llm_context.py`, `data.py` (groups, bracket, name
normalisation, the openfootball feed).

---

## Evaluation discipline

Every change to the models is judged the same way, no exceptions:

1. Run the **leakage-free walk-forward backtest** before and after.
2. Compare on **RPS first**, then log-loss, then ECE. Accuracy is reported but
   never decides.
3. A change is adopted only if it beats the **base-rate baseline** and the
   current default out-of-sample, without materially worsening calibration.

Nothing is ever fit on data inside the scoring window. The test suite enforces
the probability-sum and leakage invariants (`pytest -q`).

---

## Keeping it current during the tournament

```bash
python run.py update --data data/results.csv --fetch      # pull played matches (free, no key)
```

Or in the web app: **Data & model → Fetch results**. The free openfootball
feed updates roughly daily; enter a result by hand for anything it hasn't caught
up to yet (re-entering corrects a mistake). Re-run a simulation afterwards and
the title race reflects reality. Set `WC_AUTO_FETCH=1` to poll every six hours.

---

## Optional LLM context

Set `LLM_API_KEY` and `LLM_MODEL` (any OpenAI-compatible endpoint; override
`LLM_BASE_URL` if your provider differs), then `run.py update --refresh-llm`
fetches injury / suspension / form news and converts it to **bounded** Elo deltas
(clamped to ±60), applied at simulation time. It is context, not an oracle: you
are expected to prove it helps on the backtest before trusting it.

---

## Project layout

```
wc2026/          match model, Elo, tournament, Monte Carlo, evaluation, tuning
webapp/          FastAPI backend, SQLite storage, static frontend
data/            results.csv (history + schedule), wc2026.db (web app)
docs/            accuracy-analysis.md, goals-validation.md
run.py           command-line interface
run_web.py       start the web app
make_demo_data.py   synthetic data when the Kaggle CSV is absent
```

## Testing

```bash
pytest -q        # fast suite; enforces probability sums and leakage discipline
```

## Docker

```bash
docker build -t wc26 . && docker run -p 8000:8000 wc26
```

## Caveats & limitations

- Results-only models are near their ceiling (RPS ≈ 0.171); real accuracy gains
  need new information (market odds, xG), see `docs/accuracy-analysis.md`.
- The R32 routing of third-placed teams is a rank-based approximation, not
  FIFA's official 495-combination table. The group stage itself is exact.
- Knockout matches drawn after 90′ aren't auto-conditioned (the schema lacks
  shootout winners), pin them by hand if needed.

## License

Released under the MIT License, see [`LICENSE`](LICENSE). Match data comes from
the Kaggle dataset by martj42 and the openfootball project, under their own
terms.
