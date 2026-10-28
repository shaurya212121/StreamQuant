import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import select

# Set up paths so we can import from db
sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import SessionLocal
from db.models import StockPrice

# The 8 institutional features engineered by StreamQuant
FEATURES = [
    'sma_ratio',
    'macd',
    'rsi',
    'daily_return',
    'volume_z_score',
    'rolling_sharpe',
    'rolling_max_drawdown',
    'atr_20'
]

def load_stock_data(ticker: str) -> pd.DataFrame:
    """Pulls full chronological price & feature history from PostgreSQL."""
    session = SessionLocal()
    try:
        query = (
            select(StockPrice)
            .where(StockPrice.ticker == ticker.upper())
            .order_by(StockPrice.date.asc())
        )
        df = pd.read_sql(query, session.bind)
        return df
    finally:
        session.close()

def run_honest_backtest(ticker: str, train_ratio: float = 0.7, fee_pct: float = 0.0005) -> dict:
    """
    Executes a walk-forward out-of-sample backtest with strict T+1 execution
    to guarantee zero lookahead bias / data leakage.
    """
    print(f"\n==========================================")
    print(f"[BACKTEST] Running Honest Backtest for {ticker.upper()}")
    print(f"==========================================")

    df = load_stock_data(ticker)
    if df.empty:
        print(f"[ERROR] No data found in database for {ticker}")
        return {}

    # Target: Did price go UP tomorrow?
    df['tomorrow_close'] = df['close'].shift(-1)
    df['target'] = (df['tomorrow_close'] > df['close']).astype(int)

    # Drop rows missing features or target
    clean_df = df.dropna(subset=FEATURES + ['target', 'close']).copy().reset_index(drop=True)

    if len(clean_df) < 100:
        print(f"[ERROR] Insufficient data for {ticker}: only {len(clean_df)} bars.")
        return {}

    # 1. STRICT TRAIN / TEST SPLIT (No data leakage)
    split_idx = int(len(clean_df) * train_ratio)
    train_df = clean_df.iloc[:split_idx].copy()
    test_df = clean_df.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"[INFO] Total Clean Bars:     {len(clean_df)}")
    print(f"[INFO] Training Window:      {len(train_df)} bars ({train_df['date'].iloc[0]} to {train_df['date'].iloc[-1]})")
    print(f"[INFO] Out-of-Sample Test:   {len(test_df)} bars ({test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]})")

    # 2. TRAIN XGBOOST ONLY ON THE PAST
    X_train = train_df[FEATURES]
    y_train = train_df['target']

    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 3. PREDICT ON UNSEEN TEST DATA
    X_test = test_df[FEATURES]
    test_df['pred_signal'] = model.predict(X_test)  # 1 = Bullish, 0 = Bearish

    # 4. PREVENT LOOKAHEAD BIAS: SHIFT SIGNALS BY 1 DAY
    # Signal generated at Day T close is traded on Day T+1!
    test_df['position'] = test_df['pred_signal'].shift(1).fillna(0)

    # 5. CALCULATE DAILY RETURNS & REAL-WORLD COSTS
    test_df['stock_return'] = test_df['close'].pct_change().fillna(0)

    # Trading cost applied only when changing position (buying or selling)
    test_df['trade_occurred'] = (test_df['position'] != test_df['position'].shift(1)).astype(int)
    test_df['fees'] = test_df['trade_occurred'] * fee_pct

    # Strategy daily return
    test_df['strategy_return'] = (test_df['position'] * test_df['stock_return']) - test_df['fees']

    # Cumulative growth
    test_df['portfolio_equity'] = (1 + test_df['strategy_return']).cumprod()
    test_df['benchmark_equity'] = (1 + test_df['stock_return']).cumprod()

    # 6. CALCULATE HONEST INSTITUTIONAL METRICS
    total_strategy_return = float((test_df['portfolio_equity'].iloc[-1] - 1.0) * 100)
    total_benchmark_return = float((test_df['benchmark_equity'].iloc[-1] - 1.0) * 100)

    # Win Rate: Percentage of days we held stock and made positive returns
    active_days = test_df[test_df['position'] == 1]
    if len(active_days) > 0:
        winning_days = (active_days['stock_return'] > 0).sum()
        win_rate = float((winning_days / len(active_days)) * 100)
    else:
        win_rate = 0.0

    # Annualized Sharpe Ratio
    strat_returns = test_df['strategy_return']
    daily_mean = strat_returns.mean()
    daily_std = strat_returns.std()
    sharpe_ratio = float((daily_mean / (daily_std + 1e-6)) * (252 ** 0.5))

    # Max Drawdown
    rolling_peak = test_df['portfolio_equity'].cummax()
    drawdown = (test_df['portfolio_equity'] - rolling_peak) / rolling_peak
    max_drawdown = float(drawdown.min() * 100)

    results = {
        "ticker": ticker.upper(),
        "test_period": f"{test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]}",
        "total_test_days": len(test_df),
        "strategy_return_pct": round(total_strategy_return, 2),
        "benchmark_buy_and_hold_pct": round(total_benchmark_return, 2),
        "win_rate_pct": round(win_rate, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "total_trades_executed": int(test_df['trade_occurred'].sum())
    }

    print("\n--- FINAL REALISTIC RESULTS ---")
    print(f"[RESULT] Strategy Return:        {results['strategy_return_pct']}%")
    print(f"[RESULT] Buy & Hold Return:      {results['benchmark_buy_and_hold_pct']}%")
    print(f"[RESULT] Win Rate:               {results['win_rate_pct']}% (Real-world: 50%-58%)")
    print(f"[RESULT] Sharpe Ratio:           {results['sharpe_ratio']} (Real-world: 1.0-2.5)")
    print(f"[RESULT] Max Drawdown:           {results['max_drawdown_pct']}%")
    print(f"[RESULT] Total Trades Executed:  {results['total_trades_executed']}")

    # Save to JSON
    output_dir = Path(__file__).resolve().parent / "backtest_results"
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / f"{ticker.upper()}_honest_backtest.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[SAVED] Saved results to: {out_file}\n")

    return results

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "TSLA"
    run_honest_backtest(ticker)
