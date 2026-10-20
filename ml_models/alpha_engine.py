"""
StreamQuant Quantitative Alpha & Risk Engine
Path: c:\streamquant\ml_models\alpha_engine.py
Description: Evaluates active AI strategies against passive Buy & Hold and DCA,
computing institutional risk metrics, CAPM Beta, and Jensen's Alpha.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sqlalchemy import select
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import SessionLocal
from db.models import StockPrice

# Institutional constants
INITIAL_CAPITAL = 100000.0
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_ANNUAL = 0.045  # 4.5% Risk-free rate baseline
TRANSACTION_FEE_BPS = 0.0010   # 10 bps slippage & commission per turnover unit

@dataclass
class PerformanceMetrics:
    total_return_pct: float
    cagr_pct: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    capm_beta: float
    jensens_alpha: float

@dataclass
class AlphaEngineResult:
    ticker: str
    metrics: dict  # Contains dict mapping strategy name -> PerformanceMetrics
    equity_curves: pd.DataFrame

class AlphaEngine:
    def __init__(self, ticker: str, initial_capital: float = INITIAL_CAPITAL):
        self.ticker = ticker
        self.initial_capital = initial_capital

    def _load_data_and_model(self) -> tuple[pd.DataFrame, xgb.XGBClassifier]:
        """Loads historical price bars from Supabase and the trained model artifact."""
        db = SessionLocal()
        try:
            query = (
                select(StockPrice)
                .where(StockPrice.ticker == self.ticker)
                .order_by(StockPrice.date)
            )
            df = pd.read_sql(query, db.bind)
        finally:
            db.close()

        model_path = Path(__file__).resolve().parent / f"{self.ticker}_xgb_model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model artifact not found at {model_path}. Run train_xgb.py first.")
        
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return df, model

    def run_backtest(self) -> AlphaEngineResult:
        df, model = self._load_data_and_model()
        
        feature_cols = [
            'sma_ratio', 'macd', 'rsi', 'daily_return', 
            'volume_z_score', 'rolling_sharpe', 'rolling_max_drawdown', 'atr_20'
        ]
        
        df = df.dropna(subset=feature_cols + ['close']).copy()
        X = df[feature_cols]

        # 1. Generate Raw Model Signals
        df['raw_signal'] = model.predict(X)

        # 2. Enforce Strict T+1 Execution Lag (Preventing Lookahead Bias)
        df['position'] = df['raw_signal'].shift(1).fillna(0)

        # 3. Calculate Asset Returns & Strategy Turnover Frictions
        df['asset_return'] = df['close'].pct_change().fillna(0)
        df['turnover'] = df['position'].diff().abs().fillna(0)
        df['friction'] = df['turnover'] * TRANSACTION_FEE_BPS

        df['strategy_return'] = (df['position'] * df['asset_return']) - df['friction']

        # 5. Construct Parallel NAV Equity Curves ($100,000 Base)
        df['nav_ai'] = self.initial_capital * (1 + df['strategy_return']).cumprod()
        df['nav_bh'] = self.initial_capital * (1 + df['asset_return']).cumprod()

        # Dollar-Cost Averaging (DCA) NAV simulation
        dca_shares = 0.0
        dca_navs = []
        periodic_allocation = self.initial_capital / len(df) if len(df) > 0 else 0
        
        for price, ret in zip(df['close'], df['asset_return']):
            dca_shares += periodic_allocation / price if price > 0 else 0
            current_portfolio_val = dca_shares * price
            dca_navs.append(current_portfolio_val)
        
        df['nav_dca'] = dca_navs

        # 6. Compute Institutional Metrics for Each Strategy
        strategies = {
            "Active_AI": df['strategy_return'],
            "Buy_Hold": df['asset_return'],
            "DCA": df['nav_dca'].pct_change().fillna(0)
        }

        metrics_dict = {}
        rf_daily = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
        benchmark_returns = df['asset_return']

        for name, ret_series in strategies.items():
            if name == "Active_AI":
                metrics_dict[name] = self._calculate_metrics(ret_series, benchmark_returns, rf_daily, is_benchmark=False)
            elif name == "Buy_Hold":
                # Buy & Hold is the benchmark itself: Beta = 1.0, Alpha = 0.0
                mets = self._calculate_metrics(ret_series, benchmark_returns, rf_daily, is_benchmark=True)
                mets.capm_beta = 1.0
                mets.jensens_alpha = 0.0
                metrics_dict[name] = mets
            else:  # DCA (Passive cash deployment, neutral benchmark mapping)
                mets = self._calculate_metrics(ret_series, benchmark_returns, rf_daily, is_benchmark=True)
                mets.capm_beta = 0.0
                mets.jensens_alpha = 0.0
                metrics_dict[name] = mets

        equity_curves_df = df[['date', 'close', 'nav_ai', 'nav_bh', 'nav_dca']].copy()

        return AlphaEngineResult(
            ticker=self.ticker,
            metrics=metrics_dict,
            equity_curves=equity_curves_df
        )

    def _calculate_metrics(self, returns: pd.Series, benchmark: pd.Series, rf_daily: float, is_benchmark: bool = False) -> PerformanceMetrics:
        n = len(returns)
        if n == 0:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

        total_return = (1 + returns).prod() - 1
        years = n / TRADING_DAYS_PER_YEAR
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

        volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        excess_returns = returns - rf_daily
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if returns.std() > 0 else 0.0

        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside_returns) > 0 else 1e-6
        sortino = (excess_returns.mean() / downside_vol) * np.sqrt(TRADING_DAYS_PER_YEAR) if downside_vol > 0 else 0.0

        # Maximum Drawdown calculation
        cum_wealth = (1 + returns).cumprod()
        peak = cum_wealth.cummax()
        drawdown = (cum_wealth - peak) / peak
        max_dd = drawdown.min()

        # Profit Factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')

        # CAPM Beta and Jensen's Alpha (Calculated dynamically for Active AI)
        if not is_benchmark:
            cov_matrix = np.cov(returns, benchmark)
            beta = cov_matrix[0, 1] / (benchmark.var() if benchmark.var() > 0 else 1e-6)
            
            expected_market_return = benchmark.mean() * TRADING_DAYS_PER_YEAR
            expected_strat_return = returns.mean() * TRADING_DAYS_PER_YEAR
            alpha = expected_strat_return - (RISK_FREE_RATE_ANNUAL + beta * (expected_market_return - RISK_FREE_RATE_ANNUAL))
        else:
            beta = 0.0
            alpha = 0.0

        return PerformanceMetrics(
            total_return_pct=round(float(total_return * 100), 2),
            cagr_pct=round(float(cagr * 100), 2),
            annualized_volatility=round(float(volatility * 100), 2),
            sharpe_ratio=round(float(sharpe), 2),
            sortino_ratio=round(float(sortino), 2),
            max_drawdown_pct=round(float(max_dd * 100), 2),
            profit_factor=round(float(profit_factor), 2),
            capm_beta=round(float(beta), 2),
            jensens_alpha=round(float(alpha * 100), 2)
        )

if __name__ == "__main__":
    master_session = SessionLocal()
    try:
        query = select(StockPrice.ticker).distinct()
        all_tickers = master_session.execute(query).scalars().all()

        print(f"\n==========================================")
        print(f"Running Alpha Engine Across Universe: {all_tickers}")
        print(f"==========================================")

        for ticker in all_tickers:
            model_path = Path(__file__).resolve().parent / f"{ticker}_xgb_model.json"
            if not model_path.exists():
                print(f"\n[SKIP] {ticker}: No trained model artifact found. Run train_xgb.py first.")
                continue

            try:
                engine = AlphaEngine(ticker)
                result = engine.run_backtest()
                
                print(f"\n--- Performance Report: {result.ticker} ---")
                for strat, mets in result.metrics.items():
                    print(f"  Strategy: {strat:<10} | Tot Ret: {mets.total_return_pct:>6}% | CAGR: {mets.cagr_pct:>6}% | Sharpe: {mets.sharpe_ratio:>5} | Alpha: {mets.jensens_alpha:>5}% | Max DD: {mets.max_drawdown_pct:>6}%")
            
            except Exception as e:
                print(f"[ERROR] Failed to run backtest for {ticker}: {str(e)}")

    finally:
        master_session.close()