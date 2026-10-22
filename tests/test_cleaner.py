import numpy as np
import pandas as pd
import pytest
from backend.cleaner import (
    prune_metadata_columns,
    handle_missing_values,
    calculate_volume_zscore,
    clean_and_enrich
)


def test_prune_metadata_columns():
    """Verify non-essential Yahoo Finance metadata columns are dropped."""
    df = pd.DataFrame({
        "Open": [100.0], "High": [105.0], "Low": [95.0], "Close": [102.0], "Volume": [1000],
        "Dividends": [0.5], "Stock Splits": [0.0], "Capital Gains": [0.0]
    })
    pruned = prune_metadata_columns(df)
    assert "Dividends" not in pruned.columns
    assert "Stock Splits" not in pruned.columns
    assert "Close" in pruned.columns


def test_handle_missing_values_ffill():
    """Verify that intermediate NaN ticks are forward-filled from previous ticks."""
    df = pd.DataFrame({
        "Open": [100.0, np.nan, 102.0],
        "High": [105.0, 105.0, 106.0],
        "Low": [95.0, 95.0, 96.0],
        "Close": [102.0, np.nan, 104.0],
        "Volume": [1000, 1000, 1200]
    })
    cleaned = handle_missing_values(df)
    assert cleaned["Close"].iloc[1] == 102.0
    assert not cleaned.isna().any().any()


def test_missing_required_column_raises_key_error():
    """Verify that missing required OHLCV columns raise a KeyError."""
    df = pd.DataFrame({"Open": [100.0], "High": [105.0]})  # Missing Low, Close, Volume
    with pytest.raises(KeyError):
        handle_missing_values(df)


def test_calculate_volume_zscore_handles_flat_variance():
    """Verify volume Z-score computation handles zero standard deviation gracefully without crashing."""
    flat_volume = pd.Series([1000] * 30)
    z = calculate_volume_zscore(flat_volume, window=20)
    assert len(z) == 30
    assert not np.isinf(z).any()
