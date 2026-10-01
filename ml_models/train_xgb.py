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

def pull_stock_history(ticker_symbol: str)->pd.DataFrame:
    db = SessionLocal()
    try:
        query = select(StockPrice).where(StockPrice.ticker == ticker_symbol).order_by(StockPrice.date)
        return pd.read_sql(query, db.bind)
    finally:
        db.close()

def prepare_data(df: pd.DataFrame):
    df['tomorrow_close'] = df['close'].shift(-1)
    df['price_went_up'] = (df['tomorrow_close']>df['close']).astype(int)

    clean_df = df.dropna(subset=['price_went_up','sma_10','sma_50','daily_return'])
    X = clean_df[['sma_10','sma_50','daily_return']]
    y = clean_df['price_went_up']

    return X, y

def train_model(ticker_symbol:str):
    stock_df = pull_stock_history(ticker_symbol)
    X, y = prepare_data(stock_df)

    time_splitter = TimeSeriesSplit(n_splits=5)

    ai_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )

    ai_model.fit(X, y)
    export_path = Path(__file__).resolve().parent / f"{ticker_symbol}_xgb_model.json"
    ai_model.save_model(export_path)
    print(f" model saved to {export_path}")

if __name__ == "__main__":
   master_session = SessionLocal()
   try:
       query = select(StockPrice.ticker).distinct()
       all_tickers = master_session.execute(query).scalars().all()

       print(f"indexed tickers {len(all_tickers)} stocks in the database")
       for ticker in all_tickers:
           print(f"Training model for {ticker}...")
           train_model(ticker)

   finally:
         master_session.close()

         
   
 