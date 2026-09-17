import pandas as pd
import yfinance as yf
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from sqlalchemy.dialects.postgresql import insert
from db.database import SessionLocal, engine
from db.models import StockPrice, Base
from cleaner import prune_metadata_columns, handle_missing_values
from stock import StockRecordSchema


def init_db():
    """Creates tables in PostgreSQL if they do not exist."""
    Base.metadata.create_all(bind=engine)


def store_data_in_db(df: pd.DataFrame, ticker: str):
    """Validates and upserts cleaned DataFrame records into PostgreSQL."""
    if df.empty:
        print("DataFrame is empty. Skipping database insert.")
        return

    init_db()

    # 1. Reset index to turn Date into a standard column
    records_df = df.reset_index()

    # 2. Force EVERY column name to lowercase to match PostgreSQL exactly
    records_df.columns = [c.lower() for c in records_df.columns]

    # 3. Explicitly add the ticker column
    records_df["ticker"] = ticker

    # 4. Standardize the date objects
    records_df["date"] = pd.to_datetime(records_df["date"]).dt.date

    # 5. Define exactly what the database expects
    db_columns = [
        "date", "ticker", "open", "high", "low", "close", "volume",
        "sma_10", "sma_50", "daily_return", "volume_z_score"
    ]

    # 6. Drop any extra intermediate columns
    final_cols = [c for c in db_columns if c in records_df.columns]
    records_df = records_df[final_cols]

    # 7. Final QC gate: validate each row's OHLCV against StockRecordSchema.
    #    Drops and logs anything corrupt (e.g. high < low) instead of
    #    letting it reach Postgres.
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

        # Apply the Upsert rule (ON CONFLICT DO NOTHING)
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


def fetch_and_prepare_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
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

  # 1. Retain full OHLCV data required by PostgreSQL models & candlestick charting
  df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

  # 1b. Prune non-essential metadata (defensive; also handles gaps from
  #     market holidays/halts via forward-fill, dropping/logging anything
  #     unrecoverable) BEFORE any feature math runs on top of it.
  df = prune_metadata_columns(df)
  df = handle_missing_values(df)

  # 2. Trend features
  df["sma_10"] = df["Close"].rolling(window=10).mean()
  df["sma_50"] = df["Close"].rolling(window=50).mean()
  df["daily_return"] = df["Close"].pct_change()

  # 3. Anomaly features (rolling volume stats)
  vol_mean = df["Volume"].rolling(window=20).mean()
  vol_std = df["Volume"].rolling(window=20).std()
  df["volume_z_score"] = (df["Volume"] - vol_mean) / vol_std

  # 4. Normalize timestamps: remove timezone info for SQL Date compatibility
  if df.index.tz is not None:
    df.index = df.index.tz_localize(None)
  df.index.name = "date"

  # 5. Lowercase all column headers to match PostgreSQL table schema
  df.columns = [c.lower() for c in df.columns]

  return df.dropna()


if __name__ == "__main__":
    ticker_symbol = "TSLA"
    stock_df = fetch_and_prepare_data(ticker_symbol, period="6mo")

    print("\nProcessed Features Preview:")
    print(stock_df[["close", "sma_10", "volume", "volume_z_score"]].tail())

    print("\nStoring into PostgreSQL...")
    store_data_in_db(stock_df, ticker=ticker_symbol)