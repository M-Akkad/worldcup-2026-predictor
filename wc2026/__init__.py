"""
wc2026, a World Cup 2026 prediction toolkit.

Two layers:
  1) A match model that turns team strength into a full scoreline
     probability matrix (and from it W/D/L probabilities).
  2) A Monte Carlo tournament simulator that plays the real 48-team,
     12-group, 104-match bracket thousands of times.

Public entry points:
  data            -- load/clean results, the 2026 groups, the bracket
  elo             -- World-Football-Elo style rating engine
  match_model     -- MatchModel interface, Elo->goals, ensemble, market blend
  dixon_coles     -- optional full Dixon-Coles MLE goals model
  tournament      -- single-tournament simulation with the 2026 rules
  monte_carlo     -- many simulations -> per-team probabilities
  evaluate        -- walk-forward, leakage-free backtest (Brier / log-loss)
  llm_context     -- optional LLM contextual adjustments + narratives
"""

from . import data, elo, match_model, tournament, monte_carlo, evaluate

__all__ = [
    "data",
    "elo",
    "match_model",
    "tournament",
    "monte_carlo",
    "evaluate",
]

__version__ = "0.1.0"
