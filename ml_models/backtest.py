"""
StreamQuant Out-of-Sample Backtest Engine
==========================================
Strict 70% Train / 30% Unseen Test split with 0.05% trade slippage.
Produces honest, uncontaminated performance metrics and equity curves.

Data Source: yfinance (no DB dependency for backtesting). Features come from the
same pipeline the live system uses (backend/data_engine.py), so the backtest
measures exactly the features the deployed models see.

Note: this trains its own model on the first 70% of the window, so it measures
the modelling approach out-of-sample, not the deployed model files.

Output:
  ml_models/backtest_results/{TICKER}_backtest_metrics.json
  ml_models/backtest_results/{TICKER}_equity_curve.csv
  ml_models/backtest_results/all_tickers_summary.json
"""

import sys
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, f1_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
# data_engine uses top-level imports (cleaner, stock), so backend/ must be on sys.path
sys.path.insert(0, str(ROOT_DIR / "backend"))
from data_engine import FEATURE_COLUMNS, fetch_and_prepare_data as _fetch_production_features  # noqa: E402

logger = logging.getLogger("streamquant.backtest")

# ── Constants ────────────────────────────────────────────────────────
FEATURE_COLS = FEATURE_COLUMNS

DATA_PERIOD = "5y"               # ~1,200 bars -> ~360 unseen test days per ticker
TRAIN_RATIO = 0.70               # 70% train, 30% unseen test
SLIPPAGE_PCT = 0.0005            # 0.05% per trade (realistic institutional cost)
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_ANNUAL = 0.045    # 4.5% annualized risk-free rate
INITIAL_CAPITAL = 10_000.0       # $10,000 starting capital

RESULTS_DIR = Path(__file__).resolve().parent / "backtest_results"

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA", "TSLA"]

MODEL_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    early_stopping_rounds=15,
)


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


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
    profit_factor: Optional[float]   # None when the strategy had no losing days (would be Infinity)
    annualized_volatility_pct: float

    # Provenance: what these numbers describe
    train_period_start: str
    train_period_end: str
    train_days: int
    data_period: str
    features: list
    model_params: dict
    slippage_pct: float
    generated_at: str
    git_commit: Optional[str]


def fetch_and_prepare_data(ticker: str, period: str = DATA_PERIOD) -> pd.DataFrame:
    """OHLCV + all 8 features, computed by the live ingestion pipeline (backend/data_engine.py)."""
    logger.info("Fetching %s data from yfinance (%s) ...", ticker, period)
    df = _fetch_production_features(ticker, period=period)
    if df.empty:
        raise ValueError(f"No data returned from yfinance for {ticker}")
    return df


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
    # The last bar has no "tomorrow"; leave its label NaN so it is dropped, instead of
    # (NaN > close) silently becoming 0 = "price went down"
    df["price_went_up"] = (df["tomorrow_close"] > df["close"]).astype(float)
    df.loc[df["tomorrow_close"].isna(), "price_went_up"] = np.nan
    clean_df = df.dropna(subset=["price_went_up"] + FEATURE_COLS).copy()
    clean_df["price_went_up"] = clean_df["price_went_up"].astype(int)
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
    model = xgb.XGBClassifier(**MODEL_PARAMS)

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
    # None instead of Infinity: Infinity is not valid JSON and breaks the /backtest API
    profit_factor = round(float(gains / losses), 2) if losses > 0 else None

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
        profit_factor=profit_factor,
        annualized_volatility_pct=round(float(ann_vol), 2),
        train_period_start=str(train_df["date"].iloc[0])[:10],
        train_period_end=str(train_df["date"].iloc[-1])[:10],
        train_days=len(train_df),
        data_period=DATA_PERIOD,
        features=list(FEATURE_COLS),
        model_params=dict(MODEL_PARAMS),
        slippage_pct=SLIPPAGE_PCT,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git_commit(),
    )

    # Build equity curve export DataFrame
    equity_df = test_df[["date", "close", "prediction", "position",
                          "asset_return", "strategy_return",
                          "strategy_equity", "buyhold_equity"]].copy()

    return metrics, equity_df


def export_results(ticker: str, metrics: BacktestMetrics, equity_df: pd.DataFrame):
    """Write metrics JSON and equity curve CSV to backtest_results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Export metrics JSON (allow_nan=False: fail loudly rather than write invalid JSON)
    metrics_path = RESULTS_DIR / f"{ticker}_backtest_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(asdict(metrics), f, indent=2, allow_nan=False)
    logger.info("Exported metrics -> %s", metrics_path)

    # Export equity curve CSV
    curve_path = RESULTS_DIR / f"{ticker}_equity_curve.csv"
    equity_df.to_csv(curve_path, index=False)
    logger.info("Exported equity curve -> %s", curve_path)


def run_all_backtests():
    """Run backtest for all 6 tickers and export clean results."""
    print("\n" + "=" * 60)
    print("  StreamQuant Out-of-Sample Backtest Engine")
    print(f"  {DATA_PERIOD} data | 70% Train / 30% Unseen Test | 0.05% Slippage")
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
            json.dump(summary, f, indent=2, allow_nan=False)
        print(f"\n[OK] Combined summary exported -> {summary_path}")

    return all_metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    run_all_backtests()
