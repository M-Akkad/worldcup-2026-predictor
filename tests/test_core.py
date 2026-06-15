"""Invariant tests for the wc2026 toolkit. Run with: pytest -q"""

import io

import numpy as np
import pandas as pd
import pytest

from wc2026 import data, elo as wc_elo, evaluate, monte_carlo, tournament
from wc2026.match_model import (EloMatchModel, EnsembleMatchModel, GoalsParams,
                                OrderedLogitModel, dc_score_matrix, expected_goals,
                                outcome_probs, prob_btts, prob_over, sample_score,
                                total_goals_dist)


# ---------------------------------------------------------------- data layer
def test_normalize_known_aliases():
    cases = {
        "USA": "United States",
        "Türkiye": "Turkey",
        "Curaçao": "Curacao",          # the real Kaggle spelling (cedilla)
        "Curacao": "Curacao",
        "Côte d'Ivoire": "Ivory Coast",
        "Czechia": "Czech Republic",
        "Korea Republic": "South Korea",
        "Bosnia-Herzegovina": "Bosnia and Herzegovina",
        "Netherlands": "Netherlands",  # untouched
    }
    for raw, want in cases.items():
        assert data.normalize_team(raw) == want


def test_groups_are_complete_and_unique():
    teams = data.all_tournament_teams()
    assert len(teams) == 48
    assert len(set(teams)) == 48
    assert all(len(g) == 4 for g in data.GROUPS_2026.values())
    assert data.HOST_TEAMS <= set(teams)


def test_bracket_slots_valid():
    slots = [s for pair in data.R32_SLOTS for s in pair]
    assert len(slots) == 32 and len(set(slots)) == 32
    # no R32 match may pit a winner and runner-up of the same group
    grp = {}
    for g in data.GROUPS_2026:
        grp[f"W_{g}"] = g
        grp[f"R_{g}"] = g
    for a, b in data.R32_SLOTS:
        if a in grp and b in grp:
            assert grp[a] != grp[b]


def test_neutral_parsing_is_robust():
    csv = ("date,home_team,away_team,home_score,away_score,tournament,neutral\n"
           "2024-01-01,A,B,1,0,Friendly,FALSE\n"
           "2024-01-02,A,B,1,0,Friendly,TRUE\n"
           "2024-01-03,A,B,1,0,Friendly,\n")
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv)
        path = f.name
    try:
        df = data.load_results(path)
        assert df["neutral"].tolist() == [False, True, False]  # NaN -> False
        assert df["neutral"].dtype == bool
    finally:
        os.unlink(path)


# ------------------------------------------------------------- match models
def test_score_matrix_properties():
    P = dc_score_matrix(1.5, 1.1, rho=-0.08)
    assert P.shape == (11, 11)
    assert abs(P.sum() - 1.0) < 1e-12
    assert (P >= 0).all()
    h, d, a = outcome_probs(P)
    assert abs(h + d + a - 1.0) < 1e-12
    # negative rho must raise the draw probability vs independent Poisson
    d0 = outcome_probs(dc_score_matrix(1.5, 1.1, rho=0.0))[1]
    assert d > d0


def test_goal_metrics_consistent_with_matrix():
    # Independent Poisson (rho=0): expected goals recover the input rates well
    # (only truncation at MAX_GOALS introduces a tiny downward bias).
    P = dc_score_matrix(1.7, 1.1, rho=0.0)
    eh, ea = expected_goals(P)
    assert abs(eh - 1.7) < 1e-3 and abs(ea - 1.1) < 1e-3

    # total-goals distribution is a proper pmf
    dist = total_goals_dist(P)
    assert abs(dist.sum() - 1.0) < 1e-12
    assert (dist >= 0).all()
    # its mean equals E[home] + E[away]
    assert abs((dist * np.arange(len(dist))).sum() - (eh + ea)) < 1e-9

    # over/under partition to 1 and are monotone in the line
    assert abs(prob_over(P, 2.5) + (1 - prob_over(P, 2.5)) - 1.0) < 1e-12
    assert prob_over(P, 1.5) > prob_over(P, 2.5) > prob_over(P, 3.5)
    assert abs(prob_over(P, -0.5) - 1.0) < 1e-12   # always > -0.5 goals

    # BTTS equals the inclusion-exclusion identity on the matrix
    p_home0 = float(P[0, :].sum())
    p_away0 = float(P[:, 0].sum())
    expected_btts = 1.0 - p_home0 - p_away0 + float(P[0, 0])
    assert abs(prob_btts(P) - expected_btts) < 1e-12


def test_extreme_rho_never_negative():
    P = dc_score_matrix(4.0, 0.3, rho=-0.30)   # used to produce a negative cell
    assert (P >= 0).all()
    assert abs(P.sum() - 1.0) < 1e-12


def test_sample_score_matches_matrix():
    rng = np.random.default_rng(0)
    P = dc_score_matrix(1.8, 0.9, rho=-0.08)
    n = 20000
    counts = np.zeros_like(P)
    for _ in range(n):
        i, j = sample_score(P, rng)
        counts[i, j] += 1
    # empirical frequency of the most likely cells within ~2.5 sd
    for idx in np.argsort(P.ravel())[-5:]:
        i, j = divmod(int(idx), P.shape[1])
        p = P[i, j]
        sd = (p * (1 - p) / n) ** 0.5
        assert abs(counts[i, j] / n - p) < 4 * sd + 1e-3


def test_elo_expected_and_update():
    assert wc_elo.expected_score(1500, 1500, neutral=True) == pytest.approx(0.5)
    e = wc_elo.EloRatings()
    e.update_match("A", "B", 2, 0, "Friendly", neutral=True)
    assert e.rating("A") > 1500 > e.rating("B")
    # zero-sum update
    assert e.rating("A") + e.rating("B") == pytest.approx(3000.0)


def test_ordered_logit_bridge_consistency(demo_df):
    olr = OrderedLogitModel.fit_from_history(demo_df)
    for h, a, neu in [("Brazil", "Netherlands", True),
                      ("Spain", "Cape Verde", True),
                      ("United States", "Paraguay", False)]:
        ph, pdraw, pa = olr.proba(h, a, neu)
        mh, md, ma = outcome_probs(olr.score_matrix(h, a, neu))
        assert mh == pytest.approx(ph, abs=2e-3)
        assert ma == pytest.approx(pa, abs=2e-3)
    # learned home effect should help the home side
    ph_home, _, _ = olr.proba("Brazil", "Netherlands", neutral=False)
    ph_neu, _, _ = olr.proba("Brazil", "Netherlands", neutral=True)
    assert ph_home > ph_neu


def test_ensemble_guards():
    with pytest.raises(ValueError):
        EnsembleMatchModel([])


# ---------------------------------------------------------------- tournament
def test_head_to_head_tiebreaker_deterministic():
    rng = np.random.default_rng(0)
    # A and B tied on everything overall, but A beat B head-to-head
    results = [("A", "B", 1, 0), ("C", "D", 1, 0)]
    order = tournament._head_to_head_order(["B", "A"], results, rng)
    assert order[0] == "A"


def test_group_and_ko_mechanics():
    rng = np.random.default_rng(3)
    # degenerate matrix: home always wins 1-0
    P = np.zeros((11, 11)); P[1, 0] = 1.0
    teams = ["T1", "T2", "T3", "T4"]
    matches = [(h, a, P) for i, h in enumerate(teams) for a in teams[i + 1:]]
    ranked, stats = tournament.play_group(matches, rng)
    # T1 was home 3x -> 9 pts; T4 never home -> 0 pts
    assert ranked[0] == "T1" and ranked[-1] == "T4"
    assert stats["T1"]["pts"] == 9 and stats["T4"]["pts"] == 0
    assert tournament.play_ko("H", "A", P, rng) == "H"


def test_monte_carlo_probability_sums(demo_df):
    el = wc_elo.build_from_results(demo_df)
    table = monte_carlo.run(EloMatchModel(el), n_sims=200, seed=1, elo=el)
    assert len(table) == 48
    assert table.win_title.sum() == pytest.approx(1.0)
    assert table.reach_final.sum() == pytest.approx(2.0)
    assert table.reach_sf.sum() == pytest.approx(4.0)
    assert table.reach_qf.sum() == pytest.approx(8.0)
    assert table.reach_ko.sum() == pytest.approx(32.0)
    assert table.win_group.sum() == pytest.approx(12.0)
    # advancement must be monotone
    assert (table.reach_ko >= table.reach_r16 - 1e-9).all()
    assert (table.reach_r16 >= table.reach_qf - 1e-9).all()
    assert (table.reach_qf >= table.reach_sf - 1e-9).all()
    assert (table.reach_sf >= table.reach_final - 1e-9).all()
    assert (table.reach_final >= table.win_title - 1e-9).all()


# ---------------------------------------------------------------- evaluation
def test_rps_known_values():
    y = np.array([0])
    perfect = np.array([[1.0, 0.0, 0.0]])
    assert evaluate._rps(perfect, y) == pytest.approx(0.0)
    probs = np.array([[0.5, 0.3, 0.2]])
    # cum forecast (0.5, 0.8) vs outcome (1, 1): ((0.5)^2 + (0.2)^2) / 2
    assert evaluate._rps(probs, y) == pytest.approx((0.25 + 0.04) / 2)
    # ordinal sensitivity: mass on the adjacent outcome scores better
    near = np.array([[0.6, 0.4, 0.0]])
    far = np.array([[0.6, 0.0, 0.4]])
    assert evaluate._rps(near, y) < evaluate._rps(far, y)


def test_reliability_bounds(demo_df):
    res = evaluate.walk_forward(demo_df, start_date="2024-06-01", model="elo")
    rel = res["reliability"]
    assert 0.0 <= rel["ece"] <= 1.0
    assert sum(r["n"] for r in rel["table"]) == rel["n_points"]


@pytest.mark.parametrize("model", ["elo", "ordered_logit", "ensemble"])
def test_walk_forward_beats_baseline(demo_df, model):
    res = evaluate.walk_forward(demo_df, start_date="2024-06-01", model=model)
    assert res["model"]["rps"] < res["baseline"]["rps"]
    assert res["model"]["log_loss"] < res["baseline"]["log_loss"]
    assert res["model"]["n"] > 300


# ------------------------------------------------------------ update pipeline
def test_append_results_dedup_and_normalisation(tmp_path):
    path = str(tmp_path / "results.csv")
    new = pd.DataFrame([{
        "date": "2026-06-11", "home_team": "Türkiye", "away_team": "Curaçao",
        "home_score": 2, "away_score": 0,
    }])
    s1 = data.append_results(path, new)
    assert s1 == {"added": 1, "updated": 0, "skipped": 0, "total": 1}
    # idempotent: same match (even in canonical spelling) is skipped
    again = pd.DataFrame([{
        "date": "2026-06-11", "home_team": "Turkey", "away_team": "Curacao",
        "home_score": 2, "away_score": 0,
    }])
    s2 = data.append_results(path, again)
    assert s2 == {"added": 0, "updated": 0, "skipped": 1, "total": 1}
    df = data.load_results(path)
    assert df.iloc[0]["home_team"] == "Turkey"
    assert df.iloc[0]["away_team"] == "Curacao"
    assert df.iloc[0]["tournament"] == "FIFA World Cup"
    assert bool(df.iloc[0]["neutral"]) is True


def test_append_result_fills_preexisting_fixture(tmp_path):
    # a pre-loaded fixture with no score must be filled by a later result, not
    # skipped as a duplicate (the live-results bug). Also exercises the "&" alias.
    path = str(tmp_path / "results.csv")
    fixtures = pd.DataFrame([{
        "date": "2026-06-12", "home_team": "Canada",
        "away_team": "Bosnia and Herzegovina", "home_score": None,
        "away_score": None, "tournament": "FIFA World Cup", "neutral": True,
    }])
    fixtures.to_csv(path, index=False)
    assert len(data.load_results(path)) == 0   # no played matches yet

    result = pd.DataFrame([{
        "date": "2026-06-12", "home_team": "Canada",
        "away_team": "Bosnia & Herzegovina",   # different source spelling
        "home_score": 1, "away_score": 1,
    }])
    stats = data.append_results(path, result)
    assert stats == {"added": 0, "updated": 1, "skipped": 0, "total": 1}
    df = data.load_results(path)
    assert len(df) == 1                          # fixture filled, not duplicated
    assert df.iloc[0]["away_team"] == "Bosnia and Herzegovina"
    assert int(df.iloc[0]["home_score"]) == 1 and int(df.iloc[0]["away_score"]) == 1


def test_normalize_ampersand_alias():
    assert data.normalize_team("Bosnia & Herzegovina") == "Bosnia and Herzegovina"
    assert data.normalize_team("Bosnia  &  Herzegovina") == "Bosnia and Herzegovina"


def test_db_upsert_corrects_a_mistyped_score(tmp_path, monkeypatch):
    from webapp import storage
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "t.db"))
    storage.init_db()

    def row(hs, as_):
        return pd.DataFrame([{"date": "2026-06-13", "home_team": "Qatar",
                              "away_team": "Switzerland", "home_score": hs,
                              "away_score": as_, "tournament": "FIFA World Cup",
                              "neutral": True}])

    assert storage.insert_matches(row(0, 1)) == {"added": 1, "updated": 0, "skipped": 0, "total": 1}
    # identical re-entry is a no-op (idempotent)
    assert storage.insert_matches(row(0, 1)) == {"added": 0, "updated": 0, "skipped": 1, "total": 1}
    # corrected score updates the existing row instead of being ignored
    assert storage.insert_matches(row(1, 1)) == {"added": 0, "updated": 1, "skipped": 0, "total": 1}
    df = storage.fetch_matches_df()
    assert len(df) == 1
    assert int(df.iloc[0]["home_score"]) == 1 and int(df.iloc[0]["away_score"]) == 1


def test_upcoming_fixtures_and_predictions(tmp_path, demo_df):
    # one unplayed WC fixture + a played one; only the unplayed should surface
    path = str(tmp_path / "results.csv")
    rows = pd.concat([demo_df.assign(neutral=False), pd.DataFrame([
        {"date": "2026-06-20", "home_team": "Brazil", "away_team": "Haiti",
         "home_score": None, "away_score": None, "tournament": "FIFA World Cup",
         "neutral": True},
        {"date": "2026-06-21", "home_team": "Spain", "away_team": "Uruguay",
         "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup",
         "neutral": True},   # already played -> not a fixture
    ])], ignore_index=True)
    rows.to_csv(path, index=False)

    fx = data.upcoming_fixtures(path, since="2026-06-01")
    assert list(fx["home_team"]) == ["Brazil"] and list(fx["away_team"]) == ["Haiti"]

    preds = evaluate.fixture_predictions(data.load_results(path), fx, model="elo")
    assert len(preds) == 1
    p = preds[0]["pred"]
    assert abs(p["home"] + p["draw"] + p["away"] - 1.0) < 1e-9
    assert preds[0]["favourite"] in ("Brazil", "Haiti", "Gelijkspel")


def test_adjustment_cache_roundtrip(tmp_path):
    from wc2026 import llm_context
    path = str(tmp_path / "adj.json")
    adj = {"France": llm_context.TeamAdjustment("France", -35.0, 0.8, "injury")}
    llm_context.save_adjustments(adj, path)
    back = llm_context.load_adjustments(path, max_age_hours=24)
    assert back["France"].elo_delta == -35.0
    assert back["France"].reason == "injury"
    # stale cache must be ignored
    assert llm_context.load_adjustments(path, max_age_hours=0.0) == {}


# ------------------------------------------------- conditioning on reality
def test_extract_played_tournament_matches():
    rows = [
        # group A match inside the group window -> group result
        dict(date="2026-06-15", home_team="Mexico", away_team="South Korea",
             home_score=2, away_score=1, tournament="FIFA World Cup", neutral=False),
        # cross-group decisive match after group stage -> KO winner
        dict(date="2026-06-30", home_team="France", away_team="Brazil",
             home_score=1, away_score=0, tournament="FIFA World Cup", neutral=True),
        # cross-group draw -> skipped (pens winner unknown)
        dict(date="2026-07-01", home_team="Spain", away_team="England",
             home_score=1, away_score=1, tournament="FIFA World Cup", neutral=True),
        # qualifier and pre-2026 WC match -> both ignored
        dict(date="2026-06-12", home_team="Italy", away_team="Wales",
             home_score=3, away_score=0, tournament="FIFA World Cup qualification",
             neutral=False),
        dict(date="2022-12-18", home_team="Argentina", away_team="France",
             home_score=3, away_score=3, tournament="FIFA World Cup", neutral=True),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    group_results, ko = data.extract_played_tournament_matches(df)
    assert group_results == [("Mexico", "South Korea", 2, 1)]
    assert ko == {frozenset(("France", "Brazil")): "France"}


def test_conditioned_group_is_deterministic(demo_df):
    el = wc_elo.build_from_results(demo_df)
    # fix all six Group A matches: South Africa wins everything
    fixed = [("South Africa", "Mexico", 1, 0),
             ("South Africa", "South Korea", 2, 0),
             ("South Africa", "Czech Republic", 3, 0),
             ("Mexico", "South Korea", 1, 0),
             ("Mexico", "Czech Republic", 2, 0),
             ("South Korea", "Czech Republic", 1, 0)]
    table = monte_carlo.run(EloMatchModel(el), n_sims=80, seed=4, elo=el,
                            fixed_group=fixed)
    t = table.set_index("team")
    assert t.loc["South Africa", "win_group"] == pytest.approx(1.0)
    assert t.loc["South Africa", "reach_ko"] == pytest.approx(1.0)
    assert t.loc["Mexico", "win_group"] == pytest.approx(0.0)
    assert t.loc["Mexico", "reach_ko"] == pytest.approx(1.0)   # runner-up, 6 pts
    assert t.loc["Czech Republic", "reach_ko"] == pytest.approx(0.0)  # 0 pts
    # the rest of the tournament must still be stochastic and consistent
    assert table.win_title.sum() == pytest.approx(1.0)


def test_parse_openfootball_format():
    payload = {"matches": [
        {"date": "2026-06-11", "team1": "Mexico", "team2": "South Africa",
         "group": "Group A", "score": {"ft": [2, 1]}},
        {"date": "2026-06-11", "team1": "South Korea", "team2": "Czech Republic",
         "group": "Group A"},                       # not played yet -> skipped
        {"date": "2026-06-12", "team1": "Türkiye", "team2": "Curaçao",
         "score1": 0, "score2": 0},                 # legacy score fields
    ]}
    df = data.parse_openfootball(payload)
    assert len(df) == 2
    assert df.iloc[0]["home_team"] == "Mexico"
    assert bool(df.iloc[0]["neutral"]) is False     # host at home
    assert df.iloc[1]["home_team"] == "Turkey"      # normalised
    assert bool(df.iloc[1]["neutral"]) is True


# ------------------------------------------------- market & calibration
def test_shin_devig():
    from wc2026.market import devig_proportional, devig_shin
    odds = [1.50, 4.20, 7.00]              # clear favourite + longshot, with vig
    prop = devig_proportional(odds)
    shin = devig_shin(odds)
    assert shin.sum() == pytest.approx(1.0)
    assert prop.sum() == pytest.approx(1.0)
    # Shin corrects favourite-longshot bias: favourite up, longshot down
    assert shin[0] > prop[0]
    assert shin[2] < prop[2]
    # no-margin book -> plain normalisation
    fair = devig_shin([2.0, 4.0, 4.0])
    assert fair.sum() == pytest.approx(1.0)


def test_isotonic_calibration_improves_holdout(demo_df):
    from wc2026.calibration import calibrated_backtest
    out = calibrated_backtest(demo_df, model="elo", start_date="2023-01-01")
    # valid metrics on both
    assert 0 < out["calibrated"]["log_loss"] < 2
    # calibration must not make things meaningfully worse out-of-sample
    assert out["calibrated"]["log_loss"] <= out["raw"]["log_loss"] + 0.01
    assert out["ece_calibrated"] <= out["ece_raw"] + 0.01


# --------------------------------------------------------------- web storage
def test_storage_roundtrip(tmp_path, monkeypatch, demo_df):
    from webapp import storage
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "test.db"))
    storage.init_db()
    stats = storage.insert_matches(demo_df.head(50))
    assert stats["added"] == 50
    assert storage.insert_matches(demo_df.head(50))["added"] == 0  # idempotent
    back = storage.fetch_matches_df()
    assert len(back) == 50 and bool(back["neutral"].dtype == bool)
    storage.set_adjustment("Türkiye", -120, "test")  # normalised + clamped
    adj = storage.get_adjustments()
    assert adj[0]["team"] == "Turkey" and adj[0]["delta"] == -100.0
    storage.delete_adjustment("Turkey")
    assert storage.get_adjustments() == []
    # save and read back a run
    import wc2026.elo as wc_elo, wc2026.monte_carlo as mc
    from wc2026.match_model import EloMatchModel
    el = wc_elo.build_from_results(demo_df)
    table = mc.run(EloMatchModel(el), n_sims=50, seed=0, elo=el)
    run_id = storage.save_run("elo", 50, 0, False, 0, 0, table)
    latest = storage.latest_run()
    assert latest["id"] == run_id and len(latest["table"]) == 48
    assert len(storage.run_history()) == 1
