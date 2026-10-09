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

    
def add_advanced_features(df):
    daily_returns = df['Close'].pct_change()
    rolling_mean = daily_returns.rolling(window=20).mean()
    rolling_std = daily_returns.rolling(window=20).std()
    
    # Explicitly cast to float to prevent object dtype errors
    df['rolling_sharpe'] = ((rolling_mean / (rolling_std + 1e-6)) * (252 ** 0.5)).astype(float)

    rolling_max = df['Close'].rolling(window=20).max()
    df['rolling_max_drawdown'] = ((df['Close'] - rolling_max) / rolling_max).astype(float)

    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
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
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["daily_return"] = df["Close"].pct_change()

    # 3. Anomaly features (rolling volume stats)
    vol_mean = df["Volume"].rolling(window=20).mean()
    vol_std = df["Volume"].rolling(window=20).std()
    df["volume_z_score"] = (df["Volume"] - vol_mean) / vol_std

    # 4. Inject advanced quantitative risk & volatility features
    df = add_advanced_features(df)

    # 5. Normalize timestamps: remove timezone info safely if it's a DatetimeIndex / datetime column
    if pd.api.types.is_datetime64tz_dtype(df["date"]):
        df["date"] = df["date"].dt.tz_localize(None)

    # 6. Lowercase all column headers to match PostgreSQL table schema
    df.columns = [c.lower() for c in df.columns]

    return df.dropna()


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


def run_daily_pipeline(tickers: list[str] | None = None, latest_days: int = 1) -> dict:
    results: dict = {}
    failures: dict = {}

    for symbol in tickers or TICKERS:
        try:
            print(f"\n--- Daily job: {symbol} ---")
            df = fetch_and_prepare_data(symbol, period="2y")
            if df.empty:
                results[symbol] = "no data"
                continue

            latest = df.tail(latest_days)
            store_data_in_db(latest, ticker=symbol, update_on_conflict=True)
            results[symbol] = f"ok ({str(latest['date'].iloc[-1])[:10]})"
        except Exception as exc:
            print(f"Daily job failed for {symbol}: {exc}")
            failures[symbol] = str(exc)

    if failures:
        raise RuntimeError(f"Daily pipeline failed for: {failures}. Succeeded: {results}")
    return results


if __name__ == "__main__":
    tickers = TICKERS
    
    for symbol in tickers:
        print(f"\n--- Fetching & Storing {symbol} ---")
        df = fetch_and_prepare_data(symbol, period="2y")
        
        print(f"\nProcessed Features Preview for {symbol}:")
        print(df[["date", "close", "sma_ratio", "macd", "rsi", "volume_z_score", "rolling_sharpe"]].tail())
        
        print(f"\nStoring {symbol} into PostgreSQL...")
        store_data_in_db(df, ticker=symbol)