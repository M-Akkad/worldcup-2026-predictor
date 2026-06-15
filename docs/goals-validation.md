# Goals-layer validation, can you trust the score / goal predictions?

_Leakage-free walk-forward from 2022-01-01 on `data/results.csv` (n=4568 scored
matches). Run it yourself: `python run.py backtest --model <elo|ensemble> --goals`.
Produced 2026-06-13._

"Trust" for a probability is **calibration**: when the model says 60% over 2.5,
~60% of those matches should actually go over. Football outcomes are never
certain (even the single most likely scoreline is usually ~14%), so this
validates that the *probabilities are honest*, not that any one match is called
right.

## Results

**Over/Under 2.5 goals** (model vs base-rate baseline; lower Brier/log-loss = skill):

| model | Brier | log-loss | predicted | observed | ECE |
|---|---:|---:|---:|---:|---:|
| elo | 0.2486 | 0.6904 | 48.4% | 49.3% | **0.0082** |
| ensemble | **0.2455** | **0.6838** | 53.9% | 49.3% | 0.0489 |
| base-rate | 0.2499 | 0.6930 |, | 49.3% |, |

**Both teams to score:**

| model | Brier | log-loss | predicted | observed | ECE |
|---|---:|---:|---:|---:|---:|
| elo | 0.2415 | 0.6757 | 44.9% | 43.4% | 0.0370 |
| ensemble | 0.2440 | 0.6809 | 49.1% | 43.4% | 0.0564 |
| base-rate | 0.2457 | 0.6845 |, | 43.4% |, |

**Total goals (full 0,1,…,6,7+ distribution):**

| model | log-loss | mean predicted | actual | bias |
|---|---:|---:|---:|---:|
| elo | 1.9055 | 2.61 | 2.72 | −0.107 |
| ensemble | 1.8911 | 2.91 | 2.72 | +0.195 |
| base-rate | 1.9023 |, | 2.72 |, |

## What you can trust

- **Win / draw / loss**, yes. RPS 0.1714, ECE 0.0085 (see `accuracy-analysis.md`).
- **Over/Under 2.5**, yes, and **best with the `elo` model**: it beats the base
  rate and is almost perfectly calibrated (predicted 48.4% vs observed 49.3%,
  ECE 0.008).
- **Both teams to score**, reasonably; beats the base rate, `elo` again better
  calibrated than the ensemble.
- **Expected goals**, trust the `elo` number (bias only −0.11 goal). The
  ensemble runs ~0.2 goal high.
- **Exact scoreline / full total-goals distribution**, *indicative only*. No
  model beats the base-rate distribution by much (total-goals log-loss ~1.90 for
  all), because the exact number of goals is barely predictable from team
  strength. Use expected goals and over/under, not "the exact score".

## Why the ensemble over-predicts goals (and why a constant can't fix it)

The ordered-logit is a **W/D/L model**; its scoreline distribution is a by-product
of an inverse bridge (`_rates_for_probs`) that *solves* for the two goal rates
whose Dixon-Coles matrix reproduces the model's win/draw/loss probabilities. That
solution is pinned by the W/D/L probabilities, `base_total_goals` only seeds the
solver, it does not move the answer (a scan over it confirmed: zero change). So
the ensemble's higher goal total is structural to matching the OLR's W/D/L, and
**cannot be dialled down with a single constant** without breaking the W/D/L fit.

The `elo` / Dixon-Coles model *is* the purpose-built goals model and is already
well calibrated, so the recommendation is simply: **use `elo` for goal markets**.
The predictor UI surfaces this hint when a non-elo model is selected; the
expected-goals / over-under / BTTS numbers are shown for every model so you can
compare. No parameter change was adopted, by the evaluation discipline, nothing
beat the current defaults on RPS → log-loss → ECE.
