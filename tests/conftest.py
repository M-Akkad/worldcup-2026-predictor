"""Shared fixtures: make the package importable and build one small synthetic
results dataset per test session (fast, deterministic)."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def demo_df(tmp_path_factory):
    from make_demo_data import generate
    from wc2026 import data

    out = tmp_path_factory.mktemp("data") / "results.csv"
    generate(out_path=str(out), seed=11, start_year=2016, n_matches=5000)
    return data.load_results(str(out))
