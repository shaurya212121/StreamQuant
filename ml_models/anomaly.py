from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, update  # noqa: E402

from db.database import SessionLocal, engine  # noqa: E402
from db.models import StockPrice  # noqa: E402

logger = logging.getLogger("streamquant.anomaly")

DEFAULT_WINDOW = 20
DEFAULT_THRESHOLD = 3.0


def fetch_recent_data(ticker: str, lookback_days: int = 90) -> pd.DataFrame:
    
    query = (
        select(StockPrice.date, StockPrice.volume, StockPrice.volume_z_score)
        .where(StockPrice.ticker == ticker)
        .order_by(StockPrice.date.desc())
        .limit(lookback_days)
    )

    try:
        df = pd.read_sql(query, con=engine)
    except Exception as exc:
        logger.error("Failed to fetch data for %s: %s", ticker, exc)
        raise

    return df.sort_values("date").reset_index(drop=True)


def calculate_rolling_zscore(series: pd.Series, window: int = DEFAULT_WINDOW) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()

    safe_std = rolling_std.replace(0, np.nan)

    return (series - rolling_mean) / safe_std


def flag_anomalies(
    df: pd.DataFrame,
    zscore_col: str = "volume_z_score",
    threshold: float = DEFAULT_THRESHOLD,
    two_sided: bool = False,
) -> pd.DataFrame:
    flagged = df.copy()
    z = flagged[zscore_col]

    flagged["is_volume_anomaly"] = z.abs() > threshold if two_sided else z > threshold

    return flagged


def detect_volume_anomalies(
    df: pd.DataFrame,
    volume_col: str = "volume",
    zscore_col: str = "volume_z_score",
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    two_sided: bool = False,
) -> pd.DataFrame:
    result = df.copy()

    needs_computation = zscore_col not in result.columns or result[zscore_col].isna().all()
    if needs_computation:
        result[zscore_col] = calculate_rolling_zscore(result[volume_col], window=window)

    return flag_anomalies(result, zscore_col=zscore_col, threshold=threshold, two_sided=two_sided)


def persist_anomaly_scores(df: pd.DataFrame, ticker: str) -> None:
    session = SessionLocal()
    try:
        for _, row in df.iterrows():
            z_value = None if pd.isna(row["volume_z_score"]) else float(row["volume_z_score"])
            session.execute(
                update(StockPrice)
                .where(StockPrice.ticker == ticker, StockPrice.date == row["date"])
                .values(volume_z_score=z_value)
            )
        session.commit()
        logger.info("Persisted %d volume_z_score value(s) for %s.", len(df), ticker)
    except Exception as exc:
        session.rollback()
        logger.error("Failed to persist anomaly scores for %s: %s", ticker, exc)
        raise
    finally:
        session.close()


def run_anomaly_detection(
    ticker: str,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    lookback_days: int = 90,
    persist: bool = False,
) -> pd.DataFrame:
    logger.info("Running volume anomaly detection for %s", ticker)
    raw_df = fetch_recent_data(ticker, lookback_days=lookback_days)

    if raw_df.empty:
        logger.warning("No data found for %s; skipping anomaly detection.", ticker)
        return raw_df

    if len(raw_df) < window:
        logger.warning(
            "Only %d row(s) available for %s but window=%d; rolling Z-scores "
            "will be NaN until enough history accumulates.",
            len(raw_df), ticker, window,
        )

    result_df = detect_volume_anomalies(raw_df, window=window, threshold=threshold)

    anomaly_count = int(result_df["is_volume_anomaly"].sum())
    if anomaly_count:
        logger.warning("%d volume anomaly(ies) detected for %s.", anomaly_count, ticker)

    if persist:
        persist_anomaly_scores(result_df, ticker)

    return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    rng = np.random.default_rng(seed=42)
    n_days = 40
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    volume = rng.normal(loc=1_000_000, scale=50_000, size=n_days)

    SPIKE_INDEX = 30
    volume[SPIKE_INDEX] = 10_000_000

    synthetic_df = pd.DataFrame({
        "date": dates,
        "volume": volume.astype(int),
    })

    WINDOW = DEFAULT_WINDOW
    THRESHOLD = DEFAULT_THRESHOLD

    result = detect_volume_anomalies(synthetic_df, window=WINDOW, threshold=THRESHOLD)

    print(f"\n--- Rows around the injected spike (index {SPIKE_INDEX}) ---")
    print(result.loc[SPIKE_INDEX - 3:SPIKE_INDEX + 3, ["date", "volume", "volume_z_score", "is_volume_anomaly"]])

    print(f"\n--- Edge case: first {WINDOW - 1} rows lack enough history for a rolling baseline ---")
    print(result.loc[:WINDOW - 2, ["date", "volume", "volume_z_score", "is_volume_anomaly"]])

    assert result.loc[SPIKE_INDEX, "is_volume_anomaly"], "Known spike was NOT flagged!"
    assert result.loc[:WINDOW - 2, "is_volume_anomaly"].sum() == 0, (
        "Early insufficient-history rows were incorrectly flagged!"
    )
    assert result.loc[:WINDOW - 2, "volume_z_score"].isna().all(), (
        "Early insufficient-history rows should have NaN Z-scores, not a number!"
    )

    print(
        "\n✅ Synthetic spike correctly detected at index "
        f"{SPIKE_INDEX}, and the first {WINDOW - 1} rows (insufficient "
        "history) were correctly left un-flagged with NaN Z-scores."
    )
