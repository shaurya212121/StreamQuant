import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import select

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import SessionLocal
from db.models import StockPrice

FEATURE_COLS = [
    'sma_ratio',
    'macd',
    'rsi',
    'daily_return',
    'volume_z_score',
    'rolling_sharpe',
    'rolling_max_drawdown',
    'atr_20'
]

def pull_stock_history(ticker_symbol: str) -> pd.DataFrame:
    """Pulls historical bar data chronologically from Supabase."""
    db = SessionLocal()
    try:
        query = (
            select(StockPrice)
            .where(StockPrice.ticker == ticker_symbol)
            .order_by(StockPrice.date)
        )
        return pd.read_sql(query, db.bind)
    finally:
        db.close()

def prepare_data(df: pd.DataFrame):
    """Engineers the binary target and isolates the 8 institutional features."""
    df['tomorrow_close'] = df['close'].shift(-1)
    df['price_went_up'] = (df['tomorrow_close'] > df['close']).astype(int)

    # Drop NaNs across required features and target
    clean_df = df.dropna(subset=['price_went_up'] + FEATURE_COLS)
    
    X = clean_df[FEATURE_COLS]
    y = clean_df['price_went_up']
    return X, y

def train_model(ticker_symbol: str):
    print(f"\n--- Processing Ticker: {ticker_symbol} ---")
    stock_df = pull_stock_history(ticker_symbol)
    X, y = prepare_data(stock_df)

    if len(X) < 100:
        print(f"Skipping {ticker_symbol}: Insufficient historical data points ({len(X)})")
        return

    tscv = TimeSeriesSplit(n_splits=5)
    
    oof_preds = []
    oof_true = []
    best_iterations = []

    # 2. Walk-Forward Cross-Validation Loop across all splits
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=15
        )

        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        best_iterations.append(model.best_iteration)

        preds = model.predict(X_val)
        oof_preds.extend(preds)
        oof_true.extend(y_val)

    # 3. Compute Out-of-Fold (OOF) Metrics
    mean_acc = accuracy_score(oof_true, oof_preds)
    mean_prec = precision_score(oof_true, oof_preds, zero_division=0)
    mean_f1 = f1_score(oof_true, oof_preds, zero_division=0)

    print(f"[{ticker_symbol}] OOF Metrics -> Accuracy: {mean_acc:.4f} | Precision: {mean_prec:.4f} | F1: {mean_f1:.4f}")

    # 4. Final Refit on 100% of historical data using average optimal estimators
    avg_best_iter = int(np.mean(best_iterations)) if best_iterations else 50
    print(f"[{ticker_symbol}] Refitting final model on 100% dataset (n_estimators={max(avg_best_iter, 10)})...")

    final_model = xgb.XGBClassifier(
        n_estimators=max(avg_best_iter, 10),
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    
    final_model.fit(X, y)

    # 5. Export Production Artifact
    export_path = Path(__file__).resolve().parent / f"{ticker_symbol}_xgb_model.json"
    final_model.save_model(export_path)
    print(f"Production model artifact successfully exported to {export_path}")

if __name__ == "__main__":
    master_session = SessionLocal()
    try:
        query = select(StockPrice.ticker).distinct()
        all_tickers = master_session.execute(query).scalars().all()

        print(f"Indexed tickers: {len(all_tickers)} stocks in database.")
        for ticker in all_tickers:
            train_model(ticker)
    finally:
        master_session.close()