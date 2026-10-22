import numpy as np
import pandas as pd
from ml_models.anomaly import detect_volume_anomalies


def test_detect_volume_anomaly_spike():
    """Verify that an unnatural 10x volume spike (> 3.0 sigma) is correctly flagged."""
    n_days = 40
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    volume = [1_000_000] * n_days
    volume[30] = 10_000_000  # Inject volume anomaly at index 30

    df = pd.DataFrame({"date": dates, "volume": volume})
    result = detect_volume_anomalies(df, window=20, threshold=3.0)

    assert result.loc[30, "is_volume_anomaly"] == True
    assert result.loc[25, "is_volume_anomaly"] == False


def test_insufficient_history_rows_unflagged():
    """Verify that the first (window - 1) rows with insufficient baseline are never flagged."""
    n_days = 25
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n_days),
        "volume": [1_000_000] * n_days
    })
    result = detect_volume_anomalies(df, window=20)
    # The first 19 rows lack rolling 20-day baseline
    assert result.loc[:18, "is_volume_anomaly"].sum() == 0
    assert result.loc[:18, "volume_z_score"].isna().all()
