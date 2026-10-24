import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# data_engine uses top-level imports (cleaner, stock), so backend/ must be on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import data_engine  # noqa: E402
from data_engine import FEATURE_COLUMNS, calculate_rsi, fetch_and_prepare_data  # noqa: E402


def test_rsi_is_100_when_window_has_no_losses():
    """A 14-day window of only up-days must give RSI 100, not NaN."""
    close = pd.Series(np.r_[np.linspace(100, 90, 30), np.linspace(90, 110, 20)])
    rsi = calculate_rsi(close)
    assert rsi.iloc[-1] == 100.0


def test_rsi_is_50_when_price_is_flat():
    """No gains and no losses means neutral RSI, not NaN."""
    rsi = calculate_rsi(pd.Series([100.0] * 30))
    assert rsi.iloc[-1] == 50.0


def test_rsi_is_0_when_window_has_no_gains():
    rsi = calculate_rsi(pd.Series(np.linspace(110, 90, 30)))
    assert rsi.iloc[-1] == 0.0


def test_rsi_matches_formula_for_mixed_moves():
    close = pd.Series([100.0, 102.0, 101.0, 104.0, 103.0])
    rsi = calculate_rsi(close, window=4)
    # last window deltas: +2, -1, +3, -1 -> avg gain 1.25, avg loss 0.5
    expected = 100 - 100 / (1 + 1.25 / 0.5)
    assert np.isclose(rsi.iloc[-1], expected)


def test_rsi_warmup_rows_stay_nan():
    rsi = calculate_rsi(pd.Series(np.linspace(100, 120, 30)), window=14)
    assert rsi.iloc[:14].isna().all()
    assert rsi.iloc[14:].notna().all()


def test_rsi_never_leaves_0_to_100_range():
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 500)))
    rsi = calculate_rsi(close).dropna()
    assert rsi.between(0, 100).all()


def _fake_yahoo_history(n_days: int, rally_days: int) -> pd.DataFrame:
    """Noisy history that ends in an unbroken rally (every day closes higher)."""
    rng = np.random.default_rng(7)
    noisy = 100 * np.cumprod(1 + rng.normal(0, 0.01, n_days - rally_days))
    rally = noisy[-1] * np.cumprod(np.full(rally_days, 1.01))
    close = np.r_[noisy, rally]
    idx = pd.date_range("2025-01-01", periods=n_days, freq="B", name="Date")
    return pd.DataFrame({
        "Open": close * 0.995,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n_days),
    }, index=idx)


def test_newest_bar_is_kept_after_a_long_rally(monkeypatch):
    """Regression: a 20-day rally used to make RSI NaN, and dropna() deleted the newest bar."""
    raw = _fake_yahoo_history(n_days=120, rally_days=20)
    monkeypatch.setattr(data_engine.yf, "download", lambda *a, **k: raw.copy())

    df = fetch_and_prepare_data("TEST")

    assert df["date"].iloc[-1] == raw.index[-1]
    assert df["rsi"].iloc[-1] == 100.0


def test_only_warmup_rows_are_dropped(monkeypatch):
    raw = _fake_yahoo_history(n_days=120, rally_days=20)
    monkeypatch.setattr(data_engine.yf, "download", lambda *a, **k: raw.copy())

    df = fetch_and_prepare_data("TEST")

    # sma_50 is the longest warm-up: first complete row is the 50th bar
    assert len(df) == 120 - 49
    assert df[FEATURE_COLUMNS].notna().all().all()


# --- 3.4: backfill every bar missing since the last stored date -------------

def _run_pipeline(monkeypatch, last_stored):
    """Run the daily pipeline for one fake ticker; return (result, rows passed to the DB)."""
    raw = _fake_yahoo_history(n_days=120, rally_days=5)
    monkeypatch.setattr(data_engine.yf, "download", lambda *a, **k: raw.copy())
    monkeypatch.setattr(data_engine, "get_last_stored_date", lambda ticker: last_stored)

    stored = {}

    def fake_store(df, ticker, update_on_conflict=False):
        stored["df"] = df.copy()
        stored["update_on_conflict"] = update_on_conflict

    monkeypatch.setattr(data_engine, "store_data_in_db", fake_store)
    result = data_engine.run_daily_pipeline(tickers=["TEST"])
    return result, stored, raw


def test_missed_days_are_backfilled(monkeypatch):
    """Worker was down for 3 trading days: all 3 are stored, not just the latest one."""
    raw = _fake_yahoo_history(n_days=120, rally_days=5)
    last_stored = raw.index[-4].date()

    result, stored, raw = _run_pipeline(monkeypatch, last_stored)

    dates = list(stored["df"]["date"])
    missed = list(raw.index[-3:])
    assert dates[-3:] == missed
    assert len(dates) == 3 + data_engine.OVERLAP_BARS
    assert stored["update_on_conflict"] is True
    assert result["succeeded"]["TEST"].startswith("ok: 3 new bar(s)")


def test_overlap_bars_are_reupserted(monkeypatch):
    raw = _fake_yahoo_history(n_days=120, rally_days=5)
    last_stored = raw.index[-4].date()

    _, stored, raw = _run_pipeline(monkeypatch, last_stored)

    overlap = list(stored["df"]["date"])[: data_engine.OVERLAP_BARS]
    assert overlap == list(raw.index[-3 - data_engine.OVERLAP_BARS : -3])


def test_up_to_date_ticker_only_refreshes_overlap(monkeypatch):
    raw = _fake_yahoo_history(n_days=120, rally_days=5)
    last_stored = raw.index[-1].date()

    result, stored, _ = _run_pipeline(monkeypatch, last_stored)

    assert len(stored["df"]) == data_engine.OVERLAP_BARS
    assert result["succeeded"]["TEST"].startswith("up to date")


def test_empty_database_gets_full_history(monkeypatch):
    result, stored, _ = _run_pipeline(monkeypatch, last_stored=None)

    assert len(stored["df"]) == 120 - 49  # everything after warm-up
    assert result["succeeded"]["TEST"].startswith(f"ok: {120 - 49} new bar(s)")


def test_gap_older_than_fetch_window_is_logged(monkeypatch, caplog):
    """If the last stored bar predates the fetched history, store what we can and warn."""
    with caplog.at_level("WARNING", logger="streamquant.data_engine"):
        _, stored, _ = _run_pipeline(monkeypatch, last_stored=datetime.date(2020, 1, 1))

    assert len(stored["df"]) == 120 - 49
    assert "cannot be backfilled" in caplog.text
