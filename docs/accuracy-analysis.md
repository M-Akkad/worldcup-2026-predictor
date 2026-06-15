# Accuracy analysis, where the model loses, and what would actually help

_Walk-forward from 2022-01-01 on `data/results.csv` (n=4568 scored matches),
ensemble model. Produced 2026-06-13._

The only honest test is the leakage-free walk-forward (RPS first, then log-loss,
then ECE, never accuracy). Everything below comes from that backtest, sliced by
match type so we can see *where* the error concentrates rather than just the
headline number.

## Headline

| model | RPS | log-loss | Brier | ECE | top-pick acc |
|---|---:|---:|---:|---:|---:|
| elo | 0.1720 | 0.8781 | 0.5158 | 0.0179 | 60.1% |
| ordered_logit | 0.1718 | 0.8766 | 0.5154 | 0.0138 | 59.8% |
| **ensemble** | **0.1714** | **0.8749** | **0.5142** | **0.0085** | 59.9% |
| base-rate baseline | 0.2283 | 1.0502 | 0.6333 |, | 47.8% |

The three forecasters cluster tightly (~0.171–0.172 RPS); the ensemble is both
the most accurate and the best calibrated. The model already adds large skill
over the base rate and is well calibrated (ECE 0.0085).

## Finding 1, error concentrates exactly where it matters and where it's hardest

**By |Elo gap| (effective, incl. home advantage):**

| gap bucket | n | RPS | draw rate |
|---|---:|---:|---:|
| < 50 | 625 | 0.2278 | 0.286 |
| 50–150 | 1210 | 0.2184 | 0.279 |
| 150–300 | 1347 | 0.1812 | 0.262 |
| > 300 | 1385 | **0.0954** | 0.129 |

**By tournament class:**

| class | n | RPS | log-loss |
|---|---:|---:|---:|
| qualifier | 1683 | 0.1535 | 0.8016 |
| friendly | 1273 | 0.1739 | 0.8995 |
| other | 520 | 0.1861 | 0.9122 |
| continental | 1028 | 0.1870 | 0.9339 |
| **WC-finals** | 64 | **0.2248** | 1.0651 |

**By venue:** home 0.1665 vs **neutral 0.1807**.

The model is strong on mismatches (RPS 0.095 when the gap > 300) and weak on
evenly-matched games (RPS 0.23 when the gap < 50). That weak segment *is* a World
Cup: near-equal elite teams, on neutral ground. Qualifiers are easy (lopsided);
the actual finals matches, small Elo gaps, neutral venue, are the worst-scored
segment (WC-finals RPS 0.2248, n=64, directionally clear though small). A pure
Elo-gap model has little signal left to separate two near-equal teams.

## Finding 2, the suspected weakness (draw under-prediction) is NOT present

The classic football-model failure mode is absent here:

| | value |
|---|---:|
| actual draw rate | 0.229 |
| mean predicted draw | 0.224 |
| close games (|gap|<75): actual vs predicted draw | **0.284 vs 0.285** |

The Dixon-Coles low-score correction plus the ensemble already calibrate draws
almost perfectly. Draws carry a high per-match RPS (0.1835) and log-loss (1.43)
not because they are miscalibrated but because a draw is an intrinsically
high-entropy outcome. **Tweaking the draw model would not help.**

## Finding 3, it's an information problem, not a math problem

Two confirmations that the functional form is near its ceiling on results-only
data:

- **Calibration does not help.** On the holdout split (fit 2284 / eval 2284,
  isotonic), RPS went 0.1657 → 0.1666 and log-loss 0.8589 → 0.8922, slightly
  *worse*. The model is already calibrated, so there is no headroom here.
  (The model-evaluation skill's note about calibration halving ECE was on
  synthetic demo data; it does not hold on the real CSV.)
- **Tuning is marginal.** An 80-trial Optuna search over the whole pipeline
  (incl. Elo K-factors) reached RPS 0.1707 vs the 0.1714 default, a ~0.4%
  relative gain.

Where the model loses (close, neutral, tournament games), what's missing is
information that discriminates between near-equal teams: current form, squad
availability/injuries, tactical match-ups, and tournament-specific over/under-
performance. Elo gap alone barely moves in that segment.

## Prioritised plan (ranked by evidence, not intuition)

1. **Blend market odds into `simulate`**, highest ROI. The bookmaker line is the
   one public source that *does* separate near-equal elite teams (it prices in
   form, injuries, tactics), exactly the segment we lose. Infra is half-built:
   `market.devig_shin` and `match_model.blend_with_market` exist and `predict`
   already blends; `simulate` does not yet (open roadmap item). Needs a source of
   per-match WC odds (import + manual entry).
2. **xG-based strength** (FBref/StatsBomb) as an independent ensemble member,
   separates "genuinely good" from "lucky" among the elite; helps most in the
   hard < 150-Elo segment. New data pipeline; medium-high effort.
3. **Form / recency features** (last-N results, momentum), cheaper, smaller
   effect, same target segment.
4. **Tournament-performance adjustment**, some nations systematically over/under-
   perform their Elo at World Cups; the WC-finals weakness hints at this, but
   n=64 is too small to conclude. Revisit once 2026 group games are played.

**Explicitly not worth doing** (measured above): draw-model changes, probability
calibration, more hyperparameter tuning, swapping the single model class.

## How any change is judged

Same procedure as the rest of the repo (see the `model-evaluation` skill): run
`python run.py backtest --data data/results.csv --start 2022-01-01 --model ensemble`
before and after, compare on RPS → log-loss → ECE against the base-rate column,
and adopt only what beats the current **RPS 0.1714** out-of-sample without
materially worsening ECE.
