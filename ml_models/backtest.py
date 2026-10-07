"""
StreamQuant — Historical Backtester
====================================
Walks the trained XGBoost model across historical data and produces
quantitative performance metrics that can be surfaced on the Streamlit dashboard.

Outputs
-------
- Cumulative Strategy Return vs. Buy‑and‑Hold Return
- Win Rate  (% of trades that were profitable)
- Max Drawdown
- Annualised Sharpe Ratio

Usage
-----
    python -m ml_models.backtest              # backtest every available ticker
    python -m ml_models.backtest TSLA         # backtest a single ticker
    python -m ml_models.backtest TSLA AAPL    # backtest specific tickers
"""

from __future__ import annotations

import json
import logging
import sys

# Ensure Unicode output works on Windows consoles (cp1252 → UTF-8)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import select

# ── project imports ──────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import SessionLocal, engine  # noqa: E402
from db.models import StockPrice               # noqa: E402

logger = logging.getLogger("streamquant.backtest")

# ── constants ────────────────────────────────────────────────────────
FEATURE_COLS: List[str] = ["sma_ratio", "macd", "rsi", "daily_return", "volume_z_score"]
MODEL_DIR: Path = Path(__file__).resolve().parent
LOOKBACK_YEARS: int = 2
TRADING_DAYS_PER_YEAR: int = 252
RISK_FREE_RATE: float = 0.0   # assume 0 for simplicity; adjust if needed


# ═══════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_backtest_data(ticker: str, lookback_years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    """
    Pull *at least* ``lookback_years`` of history from the database for
    the given ticker and return it sorted by date ascending.
    """
    cutoff_date = datetime.now().date() - timedelta(days=lookback_years * 365)

    query = (
        select(StockPrice)
        .where(StockPrice.ticker == ticker.upper())
        .where(StockPrice.date >= cutoff_date)
        .order_by(StockPrice.date)
    )

    df = pd.read_sql(query, con=engine)

    if df.empty:
        logger.warning("No data found for %s going back %d years.", ticker, lookback_years)

    return df


# ═══════════════════════════════════════════════════════════════════════
# 2.  CORE BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(ticker: str, lookback_years: int = LOOKBACK_YEARS) -> Dict:
    """
    Walk-forward backtest using the pre-trained XGBoost model.

    Strategy logic (long-only, fully invested):
    - Each day the model predicts whether tomorrow's close will be
      **higher** (label 1 → BULLISH) or lower (label 0 → BEARISH).
    - If BULLISH → hold the position overnight (capture the daily return).
    - If BEARISH → stay flat (daily return = 0 for strategy).

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (must have a trained model and DB data).
    lookback_years : int
        Number of years of history to simulate over.

    Returns
    -------
    dict
        {
            "ticker":                str,
            "total_trading_days":    int,
            "strategy_return_pct":   float,
            "buyhold_return_pct":    float,
            "win_rate_pct":          float,
            "max_drawdown_pct":      float,
            "sharpe_ratio":          float,
            "equity_curve":          pd.DataFrame   # date, strategy_equity, buyhold_equity
        }
    """

    # ── load model ───────────────────────────────────────────────────
    model_path = MODEL_DIR / f"{ticker.upper()}_xgb_model.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            f"Run train_xgb.py first for {ticker}."
        )
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    # ── load data ────────────────────────────────────────────────────
    df = load_backtest_data(ticker, lookback_years)
    if df.empty or len(df) < 10:
        raise ValueError(f"Not enough data for {ticker} to run a meaningful backtest.")

    # Drop rows where features are NaN (e.g. early SMA warm-up period)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"All feature rows are NaN for {ticker} — cannot backtest.")

    # ── compute forward return (actual next-day % change) ────────────
    df["next_day_return"] = df["close"].pct_change().shift(-1)
    # The last row has no "tomorrow" → drop it
    df = df.iloc[:-1].reset_index(drop=True)

    # ── generate predictions ─────────────────────────────────────────
    X = df[FEATURE_COLS]
    predictions = model.predict(X)       # 1 = bullish, 0 = bearish
    df["signal"] = predictions

    # ── daily strategy return ────────────────────────────────────────
    # If signal == 1 (bullish) we capture the next-day return;
    # if signal == 0 (bearish) we sit in cash → 0% return.
    df["strategy_return"] = df["signal"] * df["next_day_return"]

    # ── cumulative equity curves ─────────────────────────────────────
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
    df["buyhold_equity"]  = (1 + df["next_day_return"]).cumprod()

    # ── metric calculations ──────────────────────────────────────────
    total_strategy_return = float(df["strategy_equity"].iloc[-1] - 1) * 100
    total_buyhold_return  = float(df["buyhold_equity"].iloc[-1] - 1) * 100

    # Win rate: among days we were IN the market (signal=1), how many
    # had a positive next-day return?
    in_market = df[df["signal"] == 1]
    if len(in_market) > 0:
        winning_trades = (in_market["next_day_return"] > 0).sum()
        win_rate = float(winning_trades / len(in_market)) * 100
    else:
        win_rate = 0.0

    # Max drawdown (strategy equity curve)
    cummax = df["strategy_equity"].cummax()
    drawdowns = (df["strategy_equity"] - cummax) / cummax
    max_drawdown = float(drawdowns.min()) * 100   # negative number

    # Annualised Sharpe ratio
    daily_excess = df["strategy_return"] - (RISK_FREE_RATE / TRADING_DAYS_PER_YEAR)
    sharpe = 0.0
    if daily_excess.std() != 0:
        sharpe = float(
            (daily_excess.mean() / daily_excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

    # ── build equity-curve DataFrame for charting ────────────────────
    equity_curve = df[["date", "strategy_equity", "buyhold_equity"]].copy()
    equity_curve["date"] = pd.to_datetime(equity_curve["date"])
    equity_curve = equity_curve.set_index("date")

    results = {
        "ticker":              ticker.upper(),
        "total_trading_days":  len(df),
        "strategy_return_pct": round(total_strategy_return, 2),
        "buyhold_return_pct":  round(total_buyhold_return, 2),
        "win_rate_pct":        round(win_rate, 2),
        "max_drawdown_pct":    round(max_drawdown, 2),
        "sharpe_ratio":        round(sharpe, 4),
        "equity_curve":        equity_curve,
    }

    return results


# ═══════════════════════════════════════════════════════════════════════
# 3.  PRETTY REPORT  (console output)
# ═══════════════════════════════════════════════════════════════════════

_DIVIDER = "═" * 60

def print_report(results: Dict) -> None:
    """Print a clean, human-readable backtest report to stdout."""
    ticker = results["ticker"]
    strat  = results["strategy_return_pct"]
    bnh    = results["buyhold_return_pct"]
    alpha  = round(strat - bnh, 2)

    print(f"\n{_DIVIDER}")
    print(f"  📊  BACKTEST REPORT — {ticker}")
    print(_DIVIDER)
    print(f"  Trading Days Simulated  :  {results['total_trading_days']}")
    print(f"  Strategy Return         :  {strat:+.2f}%")
    print(f"  Buy‑and‑Hold Return     :  {bnh:+.2f}%")
    print(f"  Alpha (Strategy − B&H)  :  {alpha:+.2f}%")
    print(f"  Win Rate                :  {results['win_rate_pct']:.2f}%")
    print(f"  Max Drawdown            :  {results['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio (ann.)     :  {results['sharpe_ratio']:.4f}")
    print(_DIVIDER)

    if strat > bnh:
        print(f"  ✅  Model OUTPERFORMED buy-and-hold by {alpha:+.2f}%")
    else:
        print(f"  ⚠️  Model UNDERPERFORMED buy-and-hold by {alpha:.2f}%")
    print(f"{_DIVIDER}\n")


# ═══════════════════════════════════════════════════════════════════════
# 4.  EXPORT RESULTS (JSON + CSV for the dashboard)
# ═══════════════════════════════════════════════════════════════════════

def export_results(results: Dict, output_dir: Optional[Path] = None) -> Path:
    """
    Persist the metrics as JSON and the equity curve as CSV
    so the Streamlit dashboard can consume them without hitting
    the database again.

    Returns the directory where files were saved.
    """
    if output_dir is None:
        output_dir = MODEL_DIR / "backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    ticker = results["ticker"]

    # ── metrics JSON ─────────────────────────────────────────────────
    metrics_payload = {k: v for k, v in results.items() if k != "equity_curve"}
    metrics_path = output_dir / f"{ticker}_backtest_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)
    logger.info("Saved metrics → %s", metrics_path)

    # ── equity curve CSV ─────────────────────────────────────────────
    curve_path = output_dir / f"{ticker}_equity_curve.csv"
    results["equity_curve"].to_csv(curve_path)
    logger.info("Saved equity curve → %s", curve_path)

    return output_dir


# ═══════════════════════════════════════════════════════════════════════
# 5.  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def _discover_tickers() -> List[str]:
    """Return all ticker symbols that have both a trained model AND DB data."""
    model_files = list(MODEL_DIR.glob("*_xgb_model.json"))
    return sorted(p.stem.replace("_xgb_model", "") for p in model_files)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Parse CLI args: optional list of tickers
    requested_tickers = [t.upper() for t in sys.argv[1:]] if len(sys.argv) > 1 else None

    if requested_tickers is None:
        tickers = _discover_tickers()
        if not tickers:
            print("❌  No trained models found. Run train_xgb.py first.")
            sys.exit(1)
        print(f"🔎  Auto-discovered {len(tickers)} model(s): {', '.join(tickers)}")
    else:
        tickers = requested_tickers

    all_results: List[Dict] = []

    for tkr in tickers:
        try:
            result = run_backtest(tkr)
            print_report(result)
            export_results(result)
            all_results.append(result)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Skipping %s: %s", tkr, exc)

    # ── summary table ────────────────────────────────────────────────
    if all_results:
        summary = pd.DataFrame([
            {
                "Ticker":           r["ticker"],
                "Strategy %":       r["strategy_return_pct"],
                "Buy&Hold %":       r["buyhold_return_pct"],
                "Alpha %":          round(r["strategy_return_pct"] - r["buyhold_return_pct"], 2),
                "Win Rate %":       r["win_rate_pct"],
                "Max Drawdown %":   r["max_drawdown_pct"],
                "Sharpe":           r["sharpe_ratio"],
            }
            for r in all_results
        ])
        print("\n══════════════════════════════════════════════════════")
        print("  📋  CROSS-TICKER SUMMARY")
        print("══════════════════════════════════════════════════════")
        print(summary.to_string(index=False))
        print()
