"""
StreamQuant Out-of-Sample Backtest Engine
==========================================
Strict 70% Train / 30% Unseen Test split with 0.05% trade slippage.
Produces honest, uncontaminated performance metrics and equity curves.

Data Source: yfinance (self-contained, no DB dependency for backtesting)

Output:
  ml_models/backtest_results/{TICKER}_backtest_metrics.json
  ml_models/backtest_results/{TICKER}_equity_curve.csv
  ml_models/backtest_results/all_tickers_summary.json
"""

import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, f1_score

logger = logging.getLogger("streamquant.backtest")

# ── Constants ────────────────────────────────────────────────────────
FEATURE_COLS = [
    "sma_ratio", "macd", "rsi", "daily_return",
    "volume_z_score", "rolling_sharpe", "rolling_max_drawdown", "atr_20"
]

TRAIN_RATIO = 0.70               # 70% train, 30% unseen test
SLIPPAGE_PCT = 0.0005            # 0.05% per trade (realistic institutional cost)
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_ANNUAL = 0.045    # 4.5% annualized risk-free rate
INITIAL_CAPITAL = 10_000.0       # $10,000 starting capital

RESULTS_DIR = Path(__file__).resolve().parent / "backtest_results"

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA", "TSLA"]


@dataclass
class BacktestMetrics:
    """Clean out-of-sample performance metrics."""
    ticker: str
    test_period_start: str
    test_period_end: str
    test_days: int
    total_trades: int

    # Classification metrics (on test set only)
    win_rate_pct: float
    accuracy_pct: float
    precision_pct: float
    f1_score: float

    # Portfolio metrics (on test set only)
    total_return_pct: float
    buy_hold_return_pct: float
    excess_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    annualized_volatility_pct: float


def fetch_and_prepare_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance and compute all 8 features locally.
    Self-contained — no DB dependency.
    """
    import yfinance as yf

    logger.info("Fetching %s data from yfinance (%s) ...", ticker, period)
    df = yf.download(ticker, period=period, interval="1d", multi_level_index=False)

    if df.empty:
        raise ValueError(f"No data returned from yfinance for {ticker}")

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Standardize date column name
    date_col = None
    for col in df.columns:
        if str(col).lower() in ["date", "datetime", "index", "level_0"]:
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})

    df = df[["date", "Open", "High", "Low", "Close", "Volume"]].copy()

    # Strip timezone if present
    if pd.api.types.is_datetime64tz_dtype(df["date"]):
        df["date"] = df["date"].dt.tz_convert("UTC").dt.tz_localize(None)

    # ── Feature Engineering ──────────────────────────────────────────
    # SMA Ratio
    sma_10 = df["Close"].rolling(window=10).mean()
    sma_50 = df["Close"].rolling(window=50).mean()
    df["sma_ratio"] = (sma_10 - sma_50) / sma_50

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26

    # RSI (14-day)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Daily return
    df["daily_return"] = df["Close"].pct_change()

    # Volume Z-score
    vol_mean = df["Volume"].rolling(window=20).mean()
    vol_std = df["Volume"].rolling(window=20).std()
    df["volume_z_score"] = (df["Volume"] - vol_mean) / vol_std

    # Rolling Sharpe
    daily_returns = df["Close"].pct_change()
    rolling_mean = daily_returns.rolling(window=20).mean()
    rolling_std = daily_returns.rolling(window=20).std()
    df["rolling_sharpe"] = ((rolling_mean / (rolling_std + 1e-6)) * (252 ** 0.5)).astype(float)

    # Rolling Max Drawdown
    rolling_max = df["Close"].rolling(window=20).max()
    df["rolling_max_drawdown"] = ((df["Close"] - rolling_max) / rolling_max).astype(float)

    # ATR-20
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_20"] = tr.rolling(window=20).mean().astype(float)

    # Lowercase columns to match expected feature names
    df.columns = [c.lower() for c in df.columns]

    return df.dropna().reset_index(drop=True)


def run_backtest(ticker: str) -> tuple[BacktestMetrics, pd.DataFrame]:
    """
    Execute a strict out-of-sample backtest for the given ticker.

    1. Fetch all historical data from yfinance
    2. Engineer target (next-day direction)
    3. Split 70% train / 30% test chronologically (NO shuffle)
    4. Train XGBoost on train set ONLY
    5. Generate predictions on unseen test set
    6. Simulate portfolio with T+1 execution lag + 0.05% slippage
    7. Return metrics and equity curve DataFrame
    """
    logger.info("Running OOS backtest for %s ...", ticker)

    # ── 1. Load & Prepare Data ───────────────────────────────────────
    df = fetch_and_prepare_data(ticker)
    if len(df) < 100:
        raise ValueError(f"Insufficient data for {ticker}: only {len(df)} rows")

    # ── 2. Engineer Target ───────────────────────────────────────────
    df["tomorrow_close"] = df["close"].shift(-1)
    df["price_went_up"] = (df["tomorrow_close"] > df["close"]).astype(int)
    clean_df = df.dropna(subset=["price_went_up"] + FEATURE_COLS).copy()
    clean_df = clean_df.reset_index(drop=True)

    # ── 3. Strict 70/30 Chronological Split ──────────────────────────
    split_idx = int(len(clean_df) * TRAIN_RATIO)
    train_df = clean_df.iloc[:split_idx]
    test_df = clean_df.iloc[split_idx:].copy()

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["price_went_up"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["price_went_up"]

    logger.info(
        "[%s] Train: %d rows (%s to %s) | Test: %d rows (%s to %s)",
        ticker, len(train_df),
        str(train_df["date"].iloc[0])[:10], str(train_df["date"].iloc[-1])[:10],
        len(test_df),
        str(test_df["date"].iloc[0])[:10], str(test_df["date"].iloc[-1])[:10]
    )

    # ── 4. Train Model on Train Set ONLY ─────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=15
    )

    # Use last 20% of train set as validation for early stopping
    val_split = int(len(X_train) * 0.80)
    model.fit(
        X_train.iloc[:val_split], y_train.iloc[:val_split],
        eval_set=[(X_train.iloc[val_split:], y_train.iloc[val_split:])],
        verbose=False
    )

    # ── 5. Generate Predictions on UNSEEN Test Set ───────────────────
    test_df.loc[:, "prediction"] = model.predict(X_test)

    # Classification metrics
    y_pred = test_df["prediction"].values
    y_true = y_test.values
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Win rate: % of trades where predicted direction was correct
    buy_signals = test_df[test_df["prediction"] == 1]
    if len(buy_signals) > 0:
        correct_buys = (buy_signals["price_went_up"] == 1).sum()
        win_rate = correct_buys / len(buy_signals) * 100
    else:
        win_rate = 0.0

    # ── 6. Portfolio Simulation (T+1 lag + slippage) ─────────────────
    test_df.loc[:, "asset_return"] = test_df["close"].pct_change().fillna(0)

    # T+1 execution lag: today's signal applied to tomorrow's return
    test_df.loc[:, "position"] = test_df["prediction"].shift(1).fillna(0)

    # Turnover and slippage
    test_df.loc[:, "turnover"] = test_df["position"].diff().abs().fillna(0)
    test_df.loc[:, "slippage_cost"] = test_df["turnover"] * SLIPPAGE_PCT

    # Net strategy return
    test_df.loc[:, "strategy_return"] = (
        test_df["position"] * test_df["asset_return"]
    ) - test_df["slippage_cost"]

    # Equity curves starting at $10,000
    test_df.loc[:, "strategy_equity"] = (
        INITIAL_CAPITAL * (1 + test_df["strategy_return"]).cumprod()
    )
    test_df.loc[:, "buyhold_equity"] = (
        INITIAL_CAPITAL * (1 + test_df["asset_return"]).cumprod()
    )

    # ── 7. Compute Portfolio Metrics ─────────────────────────────────
    strat_returns = test_df["strategy_return"]

    total_trades = int(test_df["turnover"].gt(0).sum())

    # Total returns
    total_return = (test_df["strategy_equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    bh_return = (test_df["buyhold_equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100

    # Sharpe ratio
    rf_daily = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
    excess = strat_returns - rf_daily
    sharpe = (
        (excess.mean() / strat_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
        if strat_returns.std() > 0 else 0.0
    )

    # Sortino ratio
    downside = strat_returns[strat_returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-6
    sortino = (
        (excess.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
        if downside_std > 0 else 0.0
    )

    # Max drawdown
    cum_wealth = (1 + strat_returns).cumprod()
    peak = cum_wealth.cummax()
    drawdown = (cum_wealth - peak) / peak
    max_dd = drawdown.min() * 100

    # Profit factor
    gains = strat_returns[strat_returns > 0].sum()
    losses = abs(strat_returns[strat_returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else float("inf")

    # Annualized volatility
    ann_vol = strat_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    metrics = BacktestMetrics(
        ticker=ticker,
        test_period_start=str(test_df["date"].iloc[0])[:10],
        test_period_end=str(test_df["date"].iloc[-1])[:10],
        test_days=len(test_df),
        total_trades=total_trades,
        win_rate_pct=round(float(win_rate), 2),
        accuracy_pct=round(float(acc * 100), 2),
        precision_pct=round(float(prec * 100), 2),
        f1_score=round(float(f1), 4),
        total_return_pct=round(float(total_return), 2),
        buy_hold_return_pct=round(float(bh_return), 2),
        excess_return_pct=round(float(total_return - bh_return), 2),
        sharpe_ratio=round(float(sharpe), 2),
        sortino_ratio=round(float(sortino), 2),
        max_drawdown_pct=round(float(max_dd), 2),
        profit_factor=round(float(profit_factor), 2),
        annualized_volatility_pct=round(float(ann_vol), 2),
    )

    # Build equity curve export DataFrame
    equity_df = test_df[["date", "close", "prediction", "position",
                          "asset_return", "strategy_return",
                          "strategy_equity", "buyhold_equity"]].copy()

    return metrics, equity_df


def export_results(ticker: str, metrics: BacktestMetrics, equity_df: pd.DataFrame):
    """Write metrics JSON and equity curve CSV to backtest_results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Export metrics JSON
    metrics_path = RESULTS_DIR / f"{ticker}_backtest_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(asdict(metrics), f, indent=2)
    logger.info("Exported metrics -> %s", metrics_path)

    # Export equity curve CSV
    curve_path = RESULTS_DIR / f"{ticker}_equity_curve.csv"
    equity_df.to_csv(curve_path, index=False)
    logger.info("Exported equity curve -> %s", curve_path)


def run_all_backtests():
    """Run backtest for all 6 tickers and export clean results."""
    print("\n" + "=" * 60)
    print("  StreamQuant Out-of-Sample Backtest Engine")
    print("  70% Train / 30% Unseen Test | 0.05% Slippage")
    print("=" * 60)

    all_metrics = {}

    for ticker in TICKERS:
        try:
            metrics, equity_df = run_backtest(ticker)
            export_results(ticker, metrics, equity_df)
            all_metrics[ticker] = metrics

            print(f"\n--- {ticker} (OOS: {metrics.test_period_start} -> {metrics.test_period_end}) ---")
            print(f"  Win Rate:     {metrics.win_rate_pct}%")
            print(f"  Accuracy:     {metrics.accuracy_pct}%")
            print(f"  Sharpe Ratio: {metrics.sharpe_ratio}")
            print(f"  Sortino:      {metrics.sortino_ratio}")
            print(f"  Total Return: {metrics.total_return_pct}%  (B&H: {metrics.buy_hold_return_pct}%)")
            print(f"  Max Drawdown: {metrics.max_drawdown_pct}%")
            print(f"  Trades:       {metrics.total_trades}")
        except Exception as e:
            print(f"\n[ERROR] {ticker}: {e}")

    # Export combined summary
    if all_metrics:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = RESULTS_DIR / "all_tickers_summary.json"
        summary = {t: asdict(m) for t, m in all_metrics.items()}
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[OK] Combined summary exported -> {summary_path}")

    return all_metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    run_all_backtests()
