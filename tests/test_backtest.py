"""Out-of-sample backtest engine (ml_models/backtest.py), on synthetic Yahoo data."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# data_engine uses top-level imports (cleaner, stock), so backend/ must be on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import data_engine  # noqa: E402
from ml_models import backtest  # noqa: E402


def _fake_yahoo(n_days=1260, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.018, n_days))
    idx = pd.date_range("2021-10-25", periods=n_days, freq="B", name="Date")
    return pd.DataFrame({
        "Open": close * 0.998, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": rng.integers(20_000_000, 80_000_000, n_days),
    }, index=idx)


@pytest.fixture
def raw(monkeypatch):
    data = _fake_yahoo()
    monkeypatch.setattr(data_engine.yf, "download", lambda *a, **k: data.copy())
    return data


@pytest.fixture
def result(raw):
    return backtest.run_backtest("TEST")


def test_uses_the_production_feature_pipeline():
    """Same features the deployed models see (fixed RSI, shared z-score), not a private copy."""
    assert backtest._fetch_production_features is data_engine.fetch_and_prepare_data
    assert backtest.FEATURE_COLS == data_engine.FEATURE_COLUMNS


def test_default_window_is_five_years():
    assert backtest.DATA_PERIOD == "5y"


def test_test_period_is_strictly_after_training(result):
    metrics, _ = result
    assert metrics.train_period_end < metrics.test_period_start
    assert metrics.train_days > metrics.test_days > 300


def test_last_bar_without_a_next_day_is_not_scored(raw, result):
    """Its 'tomorrow' is unknown, so it must not be labelled 'down' and scored."""
    _, equity = result
    assert pd.Timestamp(equity["date"].iloc[-1]) == raw.index[-2]


def test_positions_lag_predictions_by_one_day(result):
    """T+1 execution: today's signal is only traded on the next bar."""
    _, equity = result
    assert equity["position"].iloc[0] == 0
    assert (equity["position"].iloc[1:].to_numpy() == equity["prediction"].iloc[:-1].to_numpy()).all()


def test_equity_curves_start_from_initial_capital(result):
    _, equity = result
    first_bh = equity["buyhold_equity"].iloc[0]
    assert first_bh == pytest.approx(backtest.INITIAL_CAPITAL)


def test_metrics_are_valid_json_with_provenance(result):
    metrics, _ = result
    payload = json.loads(json.dumps(asdict(metrics), allow_nan=False))
    assert payload["features"] == data_engine.FEATURE_COLUMNS
    assert payload["data_period"] == "5y"
    assert payload["generated_at"]


def test_profit_factor_without_losses_is_null_not_infinity(raw, monkeypatch, tmp_path):
    metrics, equity = backtest.run_backtest("TEST")
    metrics.profit_factor = None  # what run_backtest writes when there are no losing days
    monkeypatch.setattr(backtest, "RESULTS_DIR", tmp_path)

    backtest.export_results("TEST", metrics, equity)

    saved = json.loads((tmp_path / "TEST_backtest_metrics.json").read_text())
    assert saved["profit_factor"] is None


def test_exported_curve_has_the_columns_the_api_reads(raw, monkeypatch, tmp_path):
    metrics, equity = backtest.run_backtest("TEST")
    monkeypatch.setattr(backtest, "RESULTS_DIR", tmp_path)
    backtest.export_results("TEST", metrics, equity)

    curve = pd.read_csv(tmp_path / "TEST_equity_curve.csv")
    assert {"date", "strategy_equity", "buyhold_equity"} <= set(curve.columns)
