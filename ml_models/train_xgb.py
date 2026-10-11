import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sqlalchemy import select
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import SessionLocal
from db.models import StockPrice

def pull_stock_history(ticker_symbol: str) -> pd.DataFrame:
    db = SessionLocal()
    try:
        query = select(StockPrice).where(StockPrice.ticker == ticker_symbol).order_by(StockPrice.date)
        return pd.read_sql(query, db.bind)
    finally:
        db.close()

def prepare_data(df: pd.DataFrame):
    df['tomorrow_close'] = df['close'].shift(-1)
    df['price_went_up'] = (df['tomorrow_close'] > df['close']).astype(int)

    clean_df = df.dropna(subset=['price_went_up', 'sma_ratio', 'macd', 'rsi', 'daily_return', 'rolling_sharpe', 'rolling_max_drawdown', 'atr_20'])
    X = clean_df[['sma_ratio', 'macd', 'rsi', 'daily_return', 'volume_z_score', 'rolling_sharpe', 'rolling_max_drawdown', 'atr_20']]
    y = clean_df['price_went_up']

    return X, y

def train_model(ticker_symbol: str):
    stock_df = pull_stock_history(ticker_symbol)
    X, y = prepare_data(stock_df)

    if len(X) < 50:
        print(f"Skipping {ticker_symbol}: Insufficient data points ({len(X)})")
        return

    tscv = TimeSeriesSplit(n_splits=5)
    
    train_idx, val_idx = list(tscv.split(X))[-1]
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    ai_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    ai_model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    #Evaluate validation performance out-of-sample
    val_preds = ai_model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"[{ticker_symbol}] Out-of-Sample Validation Accuracy: {acc:.4f}")

    export_path = Path(__file__).resolve().parent / f"{ticker_symbol}_xgb_model.json"
    ai_model.save_model(export_path)
    print(f"Model saved to {export_path}")

if __name__ == "__main__":
    master_session = SessionLocal()
    try:
        query = select(StockPrice.ticker).distinct()
        all_tickers = master_session.execute(query).scalars().all()

        print(f"Indexed tickers: {len(all_tickers)} stocks in the database")
        for ticker in all_tickers:
            print(f"Training model for {ticker}...")
            train_model(ticker)

    finally:
        master_session.close()