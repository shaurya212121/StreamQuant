import datetime
import logging

import pandas as pd
import numpy as np
import yfinance as yf
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import OperationalError
from db.database import SessionLocal, engine
from db.models import StockPrice, Base
from cleaner import calculate_volume_zscore, handle_missing_values, prune_metadata_columns
from stock import StockRecordSchema

logger = logging.getLogger("streamquant.data_engine")

FEATURE_COLUMNS = [
    "sma_ratio", "macd", "rsi", "daily_return", "volume_z_score",
    "rolling_sharpe", "rolling_max_drawdown", "atr_20",
]

# Rolling means of exact zeros can come back as ~1e-17 instead of 0.0 because
# pandas uses running sums; treat anything this small as zero.
_ZERO_TOLERANCE = 1e-10


def init_db():
    """Creates tables in PostgreSQL if they do not exist."""
    Base.metadata.create_all(bind=engine)


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """14-day RSI using simple rolling means of gains and losses.

    Edge cases a naive gain/loss ratio gets wrong:
    - no losses in the window (pure rally) -> 100, not NaN
    - no gains and no losses (flat price)   -> 50
    The first `window` rows have no full window and stay NaN (warm-up).
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()

    gain = gain.mask(gain.abs() < _ZERO_TOLERANCE, 0.0)
    loss = loss.mask(loss.abs() < _ZERO_TOLERANCE, 0.0)

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)
    return rsi.clip(lower=0.0, upper=100.0)


def drop_warmup_rows(df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    """Drop only the leading rows where rolling features are still warming up.

    Everything from the first fully-populated row onwards is kept, including
    the newest bar. A NaN that appears after the warm-up is logged and kept
    (stored as NULL) instead of silently deleting the row.
    """
    complete = df[feature_cols].notna().all(axis=1)
    if not complete.any():
        return df.iloc[0:0]

    first_complete = complete.to_numpy().argmax()
    trimmed = df.iloc[first_complete:]

    late_gaps = trimmed[feature_cols].isna().any(axis=1)
    if late_gaps.any():
        logger.warning(
            "Keeping %d post-warm-up row(s) with missing features (stored as NULL): %s",
            int(late_gaps.sum()),
            [str(d)[:10] for d in trimmed.loc[late_gaps, "date"]],
        )
    return trimmed


def add_advanced_features(df):
    daily_returns = df['Close'].pct_change()
    rolling_mean = daily_returns.rolling(window=20).mean()
    rolling_std = daily_returns.rolling(window=20).std()
    
    # Explicitly cast to float to prevent object dtype errors
    df['rolling_sharpe'] = ((rolling_mean / (rolling_std + 1e-6)) * (252 ** 0.5)).astype(float)

    rolling_max = df['Close'].rolling(window=20).max()
    df['rolling_max_drawdown'] = ((df['Close'] - rolling_max) / rolling_max).astype(float)

    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_20'] = tr.rolling(window=20).mean().astype(float)

    return df


def fetch_and_prepare_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    print(f"Fetching data for {ticker}...")
    df = yf.download(
        ticker, period=period, interval="1d", multi_level_index=False
    )

    if df.empty:
        print(f"Error: No data found for ticker '{ticker}'.")
        return df

    # Safety fallback to ensure 1D column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. FIX: Convert the date index into an explicit column immediately
    df = df.reset_index()
    
    # Identify and standardize the date column name
    date_col = None
    for col in df.columns:
        if str(col).lower() in ['date', 'datetime', 'index', 'level_0']:
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]
        
    df = df.rename(columns={date_col: 'date'})

    # 1a. Retain full OHLCV data required by PostgreSQL models & candlestick charting
    df = df[["date", "Open", "High", "Low", "Close", "Volume"]].copy()

    # 1b. Prune non-essential metadata
    df = prune_metadata_columns(df)
    df = handle_missing_values(df)

    # 2. Trend features (Upgraded to fix downward bias)
    sma_10 = df["Close"].rolling(window=10).mean()
    sma_50 = df["Close"].rolling(window=50).mean()
    df["sma_ratio"] = (sma_10 - sma_50) / sma_50
    
    # Calculate MACD (Moving Average Convergence Divergence)
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    
    # Calculate RSI (14-day Relative Strength Index)
    df["rsi"] = calculate_rsi(df["Close"], window=14)
    df["daily_return"] = df["Close"].pct_change()

    # 3. Anomaly features (rolling volume stats)
    df["volume_z_score"] = calculate_volume_zscore(df["Volume"], window=20)

    # 4. Inject advanced quantitative risk & volatility features
    df = add_advanced_features(df)

    # 5. Normalize timestamps: convert to UTC then strip timezone for naive storage
    if pd.api.types.is_datetime64tz_dtype(df["date"]):
        df["date"] = df["date"].dt.tz_convert("UTC").dt.tz_localize(None)

    # 6. Lowercase all column headers to match PostgreSQL table schema
    df.columns = [c.lower() for c in df.columns]

    # 7. Drop only the warm-up rows at the start; never the newest bar
    df = df.replace([np.inf, -np.inf], np.nan)
    return drop_warmup_rows(df).reset_index(drop=True)


def store_data_in_db(df: pd.DataFrame, ticker: str, update_on_conflict: bool = False):
    """Validates and upserts cleaned DataFrame records into PostgreSQL."""
    if df.empty:
        print("DataFrame is empty. Skipping database insert.")
        return

    init_db()

    records_df = df.copy()

    # Ensure date column exists and is formatted properly
    if "date" in records_df.columns:
        records_df["date"] = pd.to_datetime(records_df["date"]).dt.date
    else:
        raise KeyError("Could not identify a 'date' column in the dataframe.")

    # Explicitly add the ticker column
    records_df["ticker"] = ticker

    # Define exactly what the database expects (including our 3 new risk features)
    db_columns = [
        "date", "ticker", "open", "high", "low", "close", "volume",
        "sma_ratio", "macd", "rsi", "daily_return", "volume_z_score",
        "rolling_sharpe", "rolling_max_drawdown", "atr_20"
    ]

    # Drop any extra intermediate columns
    final_cols = [c for c in db_columns if c in records_df.columns]
    records_df = records_df[final_cols]

    # NaN/inf would be stored as Postgres 'NaN' and later break JSON responses; store NULL instead
    records_df = records_df.replace([np.inf, -np.inf], np.nan)
    records_df = records_df.astype(object).where(records_df.notna(), None)

    print("DEBUG - Rows ready for insert:", len(records_df))
    print(records_df.head(2))

    # Final QC gate: validate each row's OHLCV against StockRecordSchema
    validated_rows = []
    for record in records_df.to_dict(orient="records"):
        try:
            StockRecordSchema(
                ticker=record["ticker"],
                date=record["date"],
                open=record["open"],
                high=record["high"],
                low=record["low"],
                close=record["close"],
                volume=record["volume"],
            )
            validated_rows.append(record)
        except ValueError as e:
            print(f"Dropping corrupt row for {ticker} on {record.get('date')}: {e}")

    data_dicts = validated_rows
    if not data_dicts:
        print(f"No valid rows remained for {ticker} after validation. Skipping insert.")
        return

    session = SessionLocal()
    try:
        # Build the insert statement
        stmt = insert(StockPrice).values(data_dicts)

        # Apply the Upsert rule
        if update_on_conflict:
            update_cols = {
                c.name: stmt.excluded[c.name]
                for c in StockPrice.__table__.columns
                if c.name not in ("id", "ticker", "date")
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["ticker", "date"], set_=update_cols
            )
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=["ticker", "date"])

        # Execute and commit
        result = session.execute(stmt)
        session.commit()

        print(f"Successfully processed {len(data_dicts)} rows for {ticker}.")
        print(f"Actually inserted: {result.rowcount} new rows.")

    except Exception as e:
        session.rollback()
        print(f"Error inserting into PostgreSQL: {e}")
        raise e
    finally:
        session.close()

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA"]

# Recent bars re-upserted on every run so late revisions from Yahoo are picked up
OVERLAP_BARS = 5


def get_last_stored_date(ticker: str) -> datetime.date | None:
    """Most recent bar date stored for `ticker`, or None if it has no rows yet."""
    session = SessionLocal()
    try:
        return session.execute(
            select(func.max(StockPrice.date)).where(StockPrice.ticker == ticker)
        ).scalar()
    finally:
        session.close()


def select_rows_to_store(
    df: pd.DataFrame, last_stored: datetime.date | None, overlap_bars: int = OVERLAP_BARS
) -> pd.DataFrame:
    """Every bar after `last_stored`, plus the `overlap_bars` bars before it.

    With no stored history, the whole frame is returned (initial load). Market
    holidays need no special handling: Yahoo only returns trading days, so a
    holiday simply produces no new rows.
    """
    if last_stored is None:
        return df

    is_new = df["date"] > pd.Timestamp(last_stored)
    first_new = int(is_new.to_numpy().argmax()) if is_new.any() else len(df)
    return df.iloc[max(0, first_new - overlap_bars):]


class DataFetchError(RuntimeError):
    """Yahoo Finance returned no usable history for a ticker we know exists.

    yf.download swallows rate-limit (HTTP 429) and network errors and returns an
    empty frame instead of raising, so for our fixed ticker universe an empty
    result is treated as a transient failure worth retrying.
    """


# Failures that are likely to succeed on a later attempt. Anything else (a bug,
# a validation error) fails immediately instead of being retried.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    DataFetchError,
    ConnectionError,
    TimeoutError,
    OperationalError,
)


def ingest_ticker(symbol: str, overlap_bars: int = OVERLAP_BARS) -> str:
    """Fetch, enrich and store every bar missing for one ticker. Raises on failure.

    A worker outage of N days is backfilled on the next run instead of leaving
    permanent gaps. Features are computed on the full 2y history so rolling
    windows are correct, but only the missing (and overlap) bars are written.
    """
    print(f"\n--- Daily job: {symbol} ---")
    df = fetch_and_prepare_data(symbol, period="2y")
    if df.empty:
        raise DataFetchError(
            f"Yahoo Finance returned no data for {symbol} (likely rate-limited or a network error)."
        )

    last_stored = get_last_stored_date(symbol)
    if last_stored is not None and df["date"].iloc[0] > pd.Timestamp(last_stored):
        logger.warning(
            "%s: last stored bar is %s but fetched history starts %s; "
            "bars in between cannot be backfilled from this fetch window.",
            symbol, last_stored, str(df["date"].iloc[0])[:10],
        )

    to_store = select_rows_to_store(df, last_stored, overlap_bars)
    new_bars = to_store if last_stored is None else to_store[to_store["date"] > pd.Timestamp(last_stored)]

    store_data_in_db(to_store, ticker=symbol, update_on_conflict=True)

    if new_bars.empty:
        return f"up to date ({last_stored})"
    return (
        f"ok: {len(new_bars)} new bar(s) "
        f"{str(new_bars['date'].iloc[0])[:10]} to {str(new_bars['date'].iloc[-1])[:10]}"
    )


def run_daily_pipeline(tickers: list[str] | None = None, overlap_bars: int = OVERLAP_BARS) -> dict:
    """Run ingest_ticker for each ticker in sequence, continuing past failures.

    For manual runs and scripts. The scheduled Celery job instead runs one
    retryable task per ticker (see backend/worker.py).
    """
    results: dict = {}
    failures: dict = {}

    for symbol in tickers or TICKERS:
        try:
            results[symbol] = ingest_ticker(symbol, overlap_bars)
        except Exception as exc:
            print(f"Daily job failed for {symbol}: {exc}")
            failures[symbol] = str(exc)

    if failures:
        logger.error(
            "Daily pipeline partial failure: %s. Succeeded: %s", failures, results
        )
    return {"succeeded": results, "failed": failures}


if __name__ == "__main__":
    tickers = TICKERS
    
    for symbol in tickers:
        print(f"\n--- Fetching & Storing {symbol} ---")
        df = fetch_and_prepare_data(symbol, period="2y")
        
        print(f"\nProcessed Features Preview for {symbol}:")
        print(df[["date", "close", "sma_ratio", "macd", "rsi", "volume_z_score", "rolling_sharpe"]].tail())
        
        print(f"\nStoring {symbol} into PostgreSQL...")
        store_data_in_db(df, ticker=symbol)