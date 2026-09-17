from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("streamquant.cleaner")

_NON_ESSENTIAL_COLUMNS: tuple[str, ...] = ("Dividends", "Stock Splits", "Capital Gains")

_REQUIRED_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


def prune_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=list(_NON_ESSENTIAL_COLUMNS), errors="ignore")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    missing_columns = [col for col in _REQUIRED_COLUMNS if col not in cleaned.columns]
    if missing_columns:
        raise KeyError(f"DataFrame is missing required columns: {missing_columns}")

    pre_ffill_na_count = int(cleaned[list(_REQUIRED_COLUMNS)].isna().sum().sum())
    if pre_ffill_na_count:
        logger.info("Forward-filling %d NaN cell(s) in required OHLCV columns.", pre_ffill_na_count)
        cleaned[list(_REQUIRED_COLUMNS)] = cleaned[list(_REQUIRED_COLUMNS)].ffill()

    still_corrupt_mask = cleaned[list(_REQUIRED_COLUMNS)].isna().any(axis=1)
    if still_corrupt_mask.any():
        dropped_rows = cleaned.loc[still_corrupt_mask]
        logger.warning(
            "Dropping %d row(s) with unrecoverable NaN values (no prior value to "
            "forward-fill from). Dropped indices: %s",
            len(dropped_rows),
            list(dropped_rows.index),
        )
        cleaned = cleaned.loc[~still_corrupt_mask]

    return cleaned.reset_index(drop=True)


def calculate_sma(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def calculate_volume_zscore(volume: pd.Series) -> pd.Series:
    mean = volume.mean()
    std = volume.std()

    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=volume.index)

    return (volume - mean) / std


def clean_and_enrich(df: pd.DataFrame, sma_window: int = 20) -> pd.DataFrame:
    pruned = prune_metadata_columns(df)
    cleaned = handle_missing_values(pruned)

    cleaned[f"sma_{sma_window}"] = calculate_sma(cleaned["Close"], window=sma_window)
    cleaned["volume_zscore"] = calculate_volume_zscore(cleaned["Volume"])

    return cleaned