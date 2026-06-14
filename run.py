#!/usr/bin/env python3
"""
Command-line interface for the wc2026 toolkit: backtest, simulate, fixtures,
scorecard, predict, tune and update. Run `python run.py <command> --help` for
options, or see the README for examples.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from wc2026 import data as wc_data
from wc2026 import elo as wc_elo
from wc2026 import evaluate, monte_carlo
from wc2026.match_model import (EloMatchModel, EnsembleMatchModel, GoalsParams,
                                OrderedLogitModel, expected_goal_diff, expected_goals,
                                outcome_probs, prob_btts, prob_over)

pd.set_option("display.width", 120)
pd.set_option("display.max_rows", 60)


def _load(path: str) -> pd.DataFrame:
    df = wc_data.load_results(path)
    missing = wc_data.check_coverage(df)
    if missing:
        print(f"[warn] {len(missing)} tournament teams have no matches "
              f"(likely a name mismatch): {missing}\n", file=sys.stderr)
    return df


def _params(args) -> GoalsParams:
    return GoalsParams(base_total_goals=args.total, sup_per_100_elo=args.sup,
                       rho=args.rho, home_advantage=args.hfa)


def _build_model(args, df, elo, adjustments=None):
    """Construct the chosen MatchModel for the tournament simulation."""
    params = _params(args)
    if args.model == "elo":
        return EloMatchModel(elo, params)
    # ordered_logit / ensemble both need a fitted OLR
    olr = OrderedLogitModel.fit_from_history(df, home_advantage=args.hfa,
                                             base_total_goals=args.total, rho=args.rho)
    if adjustments:
        from wc2026 import llm_context
        olr.elo = llm_context.apply_adjustments(olr.elo, adjustments)
    if args.model == "ordered_logit":
        return olr
    return EnsembleMatchModel([(EloMatchModel(elo, params), 0.5), (olr, 0.5)])


def cmd_backtest(args):
    df = _load(args.data)
    if getattr(args, "goals", False):
        return _backtest_goals(args, df)
    res = evaluate.walk_forward(df, params=_params(args), start_date=args.start,
                                home_advantage=args.hfa, regress_each_year=args.regress,
                                model=args.model)
    m, b = res["model"], res["baseline"]
    print(f"Walk-forward from {args.start}  model='{args.model}'  (n={m['n']} matches)\n")
    print(f"{'metric':<12}{'model':>12}{'base-rate':>12}")
    for key in ("rps", "log_loss", "brier", "accuracy"):
        print(f"{key:<12}{m[key]:>12.4f}{b[key]:>12.4f}")
    print(f"{'ECE':<12}{res['reliability']['ece']:>12.4f}")
    br = res["base_rates"]
    print(f"\nbase rates  home {br['home']:.1%}  draw {br['draw']:.1%}  away {br['away']:.1%}")
    print("Lower RPS/log-loss/Brier than the base-rate column = the model adds skill.")
    print("RPS is the field-standard metric (ordinal outcomes); ECE measures calibration.\n")

    print("Reliability (pooled one-vs-rest):")
    print(f"{'bin':>11}{'n':>8}{'pred':>9}{'obs':>9}")
    for r in res["reliability"]["table"]:
        if r["n"] > 0:
            print(f"  {r['bin_low']:.1f}-{r['bin_high']:.1f}{r['n']:>8}"
                  f"{r['mean_pred']:>9.3f}{r['obs_freq']:>9.3f}")
    if args.plot_reliability:
        evaluate.plot_reliability(res["_probs"], res["_y"], args.plot_reliability)

    if args.calibrate:
        from wc2026.calibration import calibrated_backtest
        out = calibrated_backtest(df, model=args.model, start_date=args.start,
                                  params=_params(args), home_advantage=args.hfa)
        print(f"\nCalibration holdout (fit {out['n_fit']} / eval {out['n_eval']}, "
              f"method={out['method']}):")
        print(f"{'':<12}{'raw':>12}{'calibrated':>12}")
        for key in ("rps", "log_loss", "brier"):
            print(f"{key:<12}{out['raw'][key]:>12.4f}{out['calibrated'][key]:>12.4f}")
        print(f"{'ECE':<12}{out['ece_raw']:>12.4f}{out['ece_calibrated']:>12.4f}")


def _backtest_goals(args, df):
    print(f"Goals walk-forward from {args.start}  model='{args.model}' "
          f"(building full score matrices - ordered_logit/ensemble are slower) ...")
    g = evaluate.walk_forward_goals(df, params=_params(args), start_date=args.start,
                                    home_advantage=args.hfa, regress_each_year=args.regress,
                                    model=args.model)
    print(f"\nScored {g['n']} matches.\n")

    def binblock(title, b):
        print(f"{title}")
        print(f"  {'':<12}{'model':>10}{'base-rate':>12}")
        print(f"  {'Brier':<12}{b['brier']:>10.4f}{b['baseline']['brier']:>12.4f}")
        print(f"  {'log-loss':<12}{b['log_loss']:>10.4f}{b['baseline']['log_loss']:>12.4f}")
        print(f"  predicted {b['mean_pred']:.1%}  vs  observed {b['obs_rate']:.1%}"
              f"   ECE {b['reliability']['ece']:.4f}")
        print(f"  {'bin':>9}{'n':>7}{'pred':>8}{'obs':>8}")
        for r in b["reliability"]["table"]:
            print(f"  {r['bin_low']:.1f}-{r['bin_high']:.1f}{r['n']:>7}"
                  f"{r['mean_pred']:>8.2f}{r['obs_freq']:>8.2f}")
        print()

    binblock("Over/Under 2.5 goals:", g["over25"])
    binblock("Both teams to score:", g["btts"])
    t = g["total_goals"]
    print("Total goals (0,1,..,6,7+):")
    print(f"  log-loss   model {t['model_log_loss']:.4f}   base-rate {t['baseline_log_loss']:.4f}")
    print(f"  mean total predicted {t['mean_pred']:.2f}  vs  actual {t['mean_actual']:.2f}"
          f"  (bias {t['bias']:+.3f}, MAE {t['mae']:.2f})")
    print("\nLower Brier/log-loss than base-rate = real skill; ECE near 0 and "
          "pred~=obs per bin = trustworthy, well-calibrated probabilities.")


def cmd_update(args):
    """Keep the model current: ingest new results and/or refresh LLM context."""
    import os
    did_something = False

    if args.fetch:
        print("[fetch] downloading played matches from openfootball (free feed) ...")
        try:
            fetched = wc_data.fetch_openfootball_results()
        except Exception as e:
            sys.exit(f"[fetch] failed ({e}); check your connection or use --result")
        if len(fetched) == 0:
            print("[fetch] upstream has no recorded scores yet (updated ~daily).")
        else:
            stats = wc_data.append_results(args.data, fetched)
            print(f"[fetch] +{stats['added']} added, {stats['updated']} fixtures filled, "
                  f"{stats['skipped']} duplicates skipped, {stats['total']} total "
                  f"in {args.data}")
        did_something = True

    new = None
    if args.add:
        new = pd.read_csv(args.add)
    if args.result:
        rows = []
        for r in args.result:
            parts = [p.strip() for p in r.split(",")]
            if len(parts) != 4:
                sys.exit(f'--result expects "HOME,AWAY,HOME_GOALS,AWAY_GOALS", got: {r}')
            h, a, hs, as_ = parts
            rows.append({"date": args.date, "home_team": h, "away_team": a,
                         "home_score": int(hs), "away_score": int(as_),
                         "tournament": args.tournament, "neutral": args.neutral})
        rows = pd.DataFrame(rows)
        new = rows if new is None else pd.concat([new, rows], ignore_index=True)

    if new is not None:
        stats = wc_data.append_results(args.data, new,
                                       default_tournament=args.tournament)
        print(f"results: +{stats['added']} added, {stats['updated']} fixtures filled, "
              f"{stats['skipped']} duplicates skipped, {stats['total']} total "
              f"in {args.data}")
        # show the updated strength of the teams involved
        df = _load(args.data)
        elo = wc_elo.build_from_results(df, home_advantage=args.hfa)
        teams = sorted({wc_data.normalize_team(t)
                        for t in list(new["home_team"]) + list(new["away_team"])})
        print("updated Elo: " + ", ".join(f"{t} {elo.rating(t):.0f}" for t in teams))
        did_something = True

    if args.adjust:
        from wc2026 import llm_context
        existing = (llm_context.load_adjustments(args.llm_cache, max_age_hours=None)
                    if os.path.exists(args.llm_cache) else {})
        for spec in args.adjust:
            parts = [p.strip() for p in spec.split(",", 2)]
            if len(parts) < 2:
                sys.exit(f'--adjust expects "TEAM,DELTA[,reason]", got: {spec}')
            team = wc_data.normalize_team(parts[0])
            delta = max(-100.0, min(100.0, float(parts[1])))
            reason = parts[2] if len(parts) > 2 else "manual adjustment"
            existing[team] = llm_context.TeamAdjustment(team, delta, 1.0, reason)
        llm_context.save_adjustments(existing, args.llm_cache)
        print(f"[adjust] saved {len(args.adjust)} manual adjustment(s) -> "
              f"{args.llm_cache} (use with: simulate --llm-cache {args.llm_cache})")
        did_something = True

    if args.refresh_llm:
        from wc2026 import llm_context
        print("[llm] fetching fresh contextual adjustments ...")
        adj = llm_context.get_team_adjustments(
            wc_data.all_tournament_teams(), use_web_search=not args.no_web_search)
        if adj:
            path = llm_context.save_adjustments(adj, args.llm_cache)
            print(f"[llm] cached {len(adj)} adjustments -> {path}; biggest movers:")
            for a in sorted(adj.values(), key=lambda x: -abs(x.elo_delta))[:8]:
                print(f"    {a.team:<22}{a.elo_delta:+6.0f}  {a.reason}")
        else:
            print("[llm] no adjustments fetched (no API key/SDK?) -> cache unchanged.")
        did_something = True

    if not did_something:
        sys.exit("nothing to do: pass --fetch or --result/--add for new scores, "
                 "--adjust for manual context, and/or --refresh-llm")


def cmd_simulate(args):
    df = _load(args.data)
    elo = wc_elo.build_from_results(df, home_advantage=args.hfa,
                                    regress_each_year=args.regress)
    adjustments = None
    if args.use_llm:
        from wc2026 import llm_context
        print("[llm] requesting contextual adjustments ...")
        adjustments = llm_context.get_team_adjustments(
            wc_data.all_tournament_teams(), use_web_search=not args.no_web_search)
        if adjustments and args.llm_cache:
            llm_context.save_adjustments(adjustments, args.llm_cache)
            print(f"[llm] cached -> {args.llm_cache}")
    elif args.llm_cache:
        from wc2026 import llm_context
        adjustments = llm_context.load_adjustments(args.llm_cache,
                                                   max_age_hours=args.llm_max_age)
    if adjustments:
        print(f"[llm] applying {len(adjustments)} adjustments; biggest movers:")
        for a in sorted(adjustments.values(), key=lambda x: -abs(x.elo_delta))[:10]:
            print(f"    {a.team:<22}{a.elo_delta:+6.0f}  {a.reason}")
        from wc2026 import llm_context
        elo = llm_context.apply_adjustments(elo, adjustments)
    elif args.use_llm:
        print("[llm] no adjustments applied (continuing with base Elo).")

    model = _build_model(args, df, elo, adjustments)

    fixed_group, fixed_ko = [], {}
    if not args.no_condition:
        fixed_group, fixed_ko = wc_data.extract_played_tournament_matches(df)
        if fixed_group or fixed_ko:
            msg = f"[conditioning] fixing {len(fixed_group)} played group matches"
            if fixed_ko:
                msg += f" and {len(fixed_ko)} decided knockout results"
            print(msg + " found in the data (disable with --no-condition)")

    print(f"\nRunning {args.n_sims} simulations (model='{args.model}') ...")
    table = monte_carlo.run(model, n_sims=args.n_sims, seed=args.seed, elo=elo,
                            fixed_group=fixed_group or None,
                            fixed_ko=fixed_ko or None)

    show = table.copy()
    pct = ["win_group", "reach_ko", "reach_r16", "reach_qf", "reach_sf",
           "reach_final", "win_title"]
    for c in pct:
        show[c] = (show[c] * 100).round(1)
    print("\nTitle-odds leaders:")
    print(show.head(20).to_string(index=False))

    if args.save:
        table.to_csv(args.save, index=False)
        print(f"\nfull table -> {args.save}")

    if args.use_llm:
        from wc2026 import llm_context
        print("\n" + llm_context.narrate_tournament(table))


def cmd_predict(args):
    df = _load(args.data)
    elo = wc_elo.build_from_results(df, home_advantage=args.hfa)
    model = _build_model(args, df, elo)
    home = wc_data.normalize_team(args.home)
    away = wc_data.normalize_team(args.away)
    P = model.score_matrix(home, away, neutral=not args.home_advantage)
    ph, pd_, pa = outcome_probs(P)

    if args.odds:
        from wc2026.market import devig_shin, parse_odds
        from wc2026.match_model import blend_with_market
        market = devig_shin(parse_odds(args.odds))
        ph, pd_, pa = blend_with_market((ph, pd_, pa), tuple(market),
                                        market_weight=args.market_weight)
        print(f"\n[market] Shin de-vig of {args.odds} -> "
              f"({market[0]:.3f}, {market[1]:.3f}, {market[2]:.3f}); "
              f"blended at weight {args.market_weight}")

    print(f"\n{home} (Elo {elo.rating(home):.0f}) vs "
          f"{away} (Elo {elo.rating(away):.0f})"
          f"  [{'home venue' if args.home_advantage else 'neutral'}; "
          f"model='{args.model}']\n")
    print(f"  {home} win : {ph:.1%}")
    print(f"  draw       : {pd_:.1%}")
    print(f"  {away} win : {pa:.1%}")
    print(f"  expected goal diff (home - away): {expected_goal_diff(P):+.2f}\n")

    eh, ea = expected_goals(P)
    print("  goals:")
    print(f"    expected   {home} {eh:.2f} - {ea:.2f} {away}   (total {eh + ea:.2f})")
    print(f"    over 2.5   {prob_over(P, 2.5):.1%}   under 2.5  {1 - prob_over(P, 2.5):.1%}")
    print(f"    both teams to score: {prob_btts(P):.1%}\n")

    flat = [((i, j), float(P[i, j])) for i in range(P.shape[0])
            for j in range(P.shape[1])]
    flat.sort(key=lambda x: -x[1])
    print("  most likely scorelines:")
    for (i, j), p in flat[:6]:
        print(f"    {home} {i}-{j} {away}   {p:.1%}")


def cmd_fixtures(args):
    played = _load(args.data)
    fix = wc_data.upcoming_fixtures(args.data, since=args.since)
    if fix.empty:
        print("No upcoming World Cup fixtures found in the data.")
        return
    if args.limit:
        fix = fix.head(args.limit)
    preds = evaluate.fixture_predictions(played, fix, model=args.model,
                                         params=_params(args), home_advantage=args.hfa)
    print(f"Upcoming fixtures  model='{args.model}'  ({len(preds)} matches)\n")
    print(f"{'date':<11}{'match':<34}{'pred H/D/A':<18}{'xG':<10}"
          f"{'likely':<9}{'favourite':<22}")
    for m in preds:
        p, s = m["pred"], m["pred_score"]
        match = f"{m['home']} v {m['away']}"
        hda = f"{p['home']:.0%}/{p['draw']:.0%}/{p['away']:.0%}"
        xg = f"{m['pred_xg']['home']:.1f}-{m['pred_xg']['away']:.1f}"
        likely = f"{s['home']}-{s['away']}"
        fav = f"{m['favourite']} {m['favourite_prob']:.0%}"
        print(f"{m['date']:<11}{match[:33]:<34}{hda:<18}{xg:<10}{likely:<9}{fav:<22}")
    print("\npred H/D/A = home win / draw / away win; 'likely' = single most "
          "probable scoreline (always a low %, football is high-variance).")


def cmd_scorecard(args):
    df = _load(args.data)
    sc = evaluate.tournament_scorecard(df, model=args.model, since=args.since,
                                       params=_params(args), home_advantage=args.hfa)
    s = sc["summary"]
    if not s["n"]:
        print(f"No played World Cup matches found since {args.since} yet "
              f"(run `update --fetch` once games kick off).")
        return
    print(f"Predicted vs actual  model='{args.model}'  ({s['n']} played WC matches "
          f"since {args.since})\n")
    print(f"{'date':<11}{'match':<40}{'pred H/D/A':<18}{'xG':<10}"
          f"{'result':<8}{'P(real)':>8}  fav")
    for m in sc["matches"]:
        p, ac = m["pred"], m["actual"]
        match = f"{m['home']} v {m['away']}"
        hda = f"{p['home']:.0%}/{p['draw']:.0%}/{p['away']:.0%}"
        xg = f"{m['pred_xg']['home']:.1f}-{m['pred_xg']['away']:.1f}"
        res = f"{ac['home']}-{ac['away']}"
        fav = "OK" if m["favourite_correct"] else "miss"
        print(f"{m['date']:<11}{match[:39]:<40}{hda:<18}{xg:<10}{res:<8}"
              f"{m['p_actual']:>7.0%}  {fav}")
    print(f"\nmean RPS {s['mean_rps']:.3f}   mean P(real) {s['mean_p_actual']:.0%}"
          f"   favourite hit-rate {s['favourite_hit_rate']:.0%}")
    print("P(real) = probability the model gave to what actually happened "
          "(higher = better); RPS lower = better.")


def cmd_tune(args):
    df = _load(args.data)
    if not args.optuna:
        print("Grid-searching goals-bridge knobs by RPS (this can take a minute) ...")
        cfg, metrics = evaluate.tune_goals_params(df, start_date=args.start)
        print("\nbest config:")
        for k, v in cfg.items():
            print(f"  {k:<18}{v}")
        print(f"\nbacktest metrics at best config: {metrics}")
        return

    from wc2026 import optimize as wc_opt
    print(f"Optuna search: model='{args.model}', metric='{args.metric}', "
          f"{args.trials} trials, walk-forward from {args.start} ...")

    def _progress(done, total, best):
        print(f"\r  trial {done:>3}/{total}   best {args.metric}={best:.4f}",
              end="", flush=True)

    res = wc_opt.optimize(df, model=args.model, n_trials=args.trials,
                          metric=args.metric, start_date=args.start,
                          seed=args.seed, progress=_progress)
    print("\n\nbest config (copy into the model defaults if adopted):")
    for k, v in res.best_config.items():
        print(f"  {k:<20}{v:.4f}")
    arrow = "better" if res.improved else "NOT better"
    print(f"\n{args.metric}: tuned {res.best_value:.4f}  vs  default "
          f"{res.baseline_value:.4f}   -> {arrow} than the current default")
    print(f"{'metric':<12}{'tuned':>12}{'default':>12}")
    for key in ("rps", "log_loss", "brier", "accuracy", "ece"):
        if key in res.best_metrics:
            print(f"{key:<12}{res.best_metrics[key]:>12.4f}"
                  f"{res.baseline_metrics[key]:>12.4f}")
    if res.improved:
        print("\nAdopt only after `pytest -q` stays green and ECE is not materially "
              "worse (see the model-evaluation skill).")
    else:
        print("\nKeep the current defaults: the search did not beat them "
              "out-of-sample. Try more --trials or a different --model.")


def build_parser():
    p = argparse.ArgumentParser(description="World Cup 2026 predictor")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--data", default="data/results.csv")
        sp.add_argument("--hfa", type=float, default=100.0, help="home advantage (Elo)")
        sp.add_argument("--sup", type=float, default=0.40, help="goal supremacy / 100 Elo")
        sp.add_argument("--total", type=float, default=2.6, help="base total goals")
        sp.add_argument("--rho", type=float, default=-0.08, help="Dixon-Coles rho")
        sp.add_argument("--regress", type=float, default=0.0, help="yearly mean reversion")

    MODELS = ("elo", "ordered_logit", "ensemble")

    b = sub.add_parser("backtest"); common(b)
    b.add_argument("--start", default="2022-01-01")
    b.add_argument("--model", choices=MODELS, default="elo")
    b.add_argument("--plot-reliability", default=None,
                   help="save a reliability diagram PNG to this path (needs matplotlib)")
    b.add_argument("--calibrate", action="store_true",
                   help="fit a calibrator on the first half of the window and "
                        "report raw vs calibrated metrics on the second half")
    b.add_argument("--goals", action="store_true",
                   help="validate the goals layer instead of W/D/L: over/under 2.5, "
                        "both-teams-to-score and total-goals calibration")
    b.set_defaults(func=cmd_backtest)

    s = sub.add_parser("simulate"); common(s)
    s.add_argument("--n-sims", type=int, default=10000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--model", choices=MODELS, default="elo")
    s.add_argument("--save", default=None, help="optional CSV output path")
    s.add_argument("--use-llm", action="store_true",
                   help="fetch fresh LLM adjustments (needs LLM_API_KEY + LLM_MODEL)")
    s.add_argument("--llm-cache", default=None,
                   help="JSON cache of adjustments: load it (without --use-llm) "
                        "or refresh it (with --use-llm)")
    s.add_argument("--llm-max-age", type=float, default=24.0,
                   help="max cache age in hours before it is ignored")
    s.add_argument("--no-web-search", action="store_true")
    s.add_argument("--no-condition", action="store_true",
                   help="do not fix already-played 2026 WC matches found in the data")
    s.set_defaults(func=cmd_simulate)

    u = sub.add_parser("update", help="ingest new results / refresh LLM context")
    u.add_argument("--data", default="data/results.csv")
    u.add_argument("--hfa", type=float, default=100.0)
    u.add_argument("--fetch", action="store_true",
                   help="pull played WC matches from the free openfootball feed "
                        "(no API key)")
    u.add_argument("--result", action="append", metavar='"HOME,AWAY,HG,AG"',
                   help="one match result; repeat the flag for several")
    u.add_argument("--add", default=None,
                   help="CSV file with new matches (same columns as results.csv)")
    u.add_argument("--adjust", action="append", metavar='"TEAM,DELTA,reason"',
                   help="manual Elo adjustment written to the cache "
                        "(free alternative to --refresh-llm); repeatable")
    u.add_argument("--date", default=pd.Timestamp.today().strftime("%Y-%m-%d"),
                   help="date for --result entries (default: today)")
    u.add_argument("--tournament", default="FIFA World Cup",
                   help='tournament label for --result entries (sets Elo K-factor)')
    u.add_argument("--neutral", action=argparse.BooleanOptionalAction, default=True,
                   help="venue for --result entries; use --no-neutral for a "
                        "host playing at home")
    u.add_argument("--refresh-llm", action="store_true",
                   help="fetch fresh LLM adjustments and cache them")
    u.add_argument("--llm-cache", default="data/adjustments.json")
    u.add_argument("--no-web-search", action="store_true")
    u.set_defaults(func=cmd_update)

    pr = sub.add_parser("predict"); common(pr)
    pr.add_argument("home"); pr.add_argument("away")
    pr.add_argument("--model", choices=MODELS, default="elo")
    pr.add_argument("--home-advantage", action="store_true",
                    help="treat as a home game for the first team (default neutral)")
    pr.add_argument("--odds", default=None, metavar='"H,D,A"',
                    help='decimal bookmaker odds to blend in, e.g. "2.10,3.30,3.90"')
    pr.add_argument("--market-weight", type=float, default=0.5,
                    help="weight on the (de-vigged) market in the blend [0..1]")
    pr.set_defaults(func=cmd_predict)

    fx = sub.add_parser("fixtures", help="predict upcoming WC fixtures (who vs who)")
    common(fx)
    fx.add_argument("--model", choices=MODELS, default="ensemble")
    fx.add_argument("--since", default=None,
                    help="only fixtures on/after this date (default: all upcoming)")
    fx.add_argument("--limit", type=int, default=None, help="show at most N fixtures")
    fx.set_defaults(func=cmd_fixtures)

    sc = sub.add_parser("scorecard", help="predicted vs actual for played WC matches")
    common(sc)
    sc.add_argument("--model", choices=MODELS, default="ensemble")
    sc.add_argument("--since", default="2026-06-01",
                    help="only matches on/after this date (tournament window)")
    sc.set_defaults(func=cmd_scorecard)

    t = sub.add_parser("tune"); common(t)
    t.add_argument("--start", default="2022-01-01")
    t.add_argument("--optuna", action="store_true",
                   help="Bayesian (TPE) search over the full pipeline incl. Elo "
                        "K-factors, instead of the small fixed grid")
    t.add_argument("--trials", type=int, default=40,
                   help="number of Optuna trials (with --optuna)")
    t.add_argument("--metric", choices=("rps", "log_loss", "brier"), default="rps",
                   help="objective to minimise (with --optuna)")
    t.add_argument("--model", choices=MODELS, default="ensemble",
                   help="forecaster to tune (with --optuna)")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_tune)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
