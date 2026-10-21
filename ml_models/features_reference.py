"""
=============================================================================
StreamQuant XGBoost Feature Reference & Education Script
Path: ml_models/features_reference.py
=============================================================================
This script serves as a permanent reference explaining and computing every
mathematical feature fed into the StreamQuant XGBoost classifier.

Run this script anytime in terminal:
    python ml_models/features_reference.py
"""

import numpy as np
import pandas as pd


# =============================================================================
# 1. SIMPLE MOVING AVERAGE RATIO (sma_ratio)
# =============================================================================
def calculate_sma_ratio(close_series: pd.Series, short_window: int = 10, long_window: int = 50) -> pd.Series:
    """
    Q: What is a Simple Moving Average (SMA)?
       It calculates the unweighted mean price over the last N days.

    Q: Why is it used?
       Stock prices bounce randomly every day. Moving averages act as low-pass
       filters, smoothing out noise to show the true underlying momentum.

    Q: On what values is it calculated?
       On daily 'Close' prices:
       - sma_10: Rolling 10-day mean (short-term momentum)
       - sma_50: Rolling 50-day mean (long-term baseline)

    Q: Why do we use a RATIO instead of raw dollar prices?
       If TSLA is $250 and NVDA is $900, raw dollar averages confuse tree models.
       The ratio (sma_10 - sma_50) / sma_50 normalizes the momentum percentage
       across all stocks regardless of price level:
       - > 0 (+0.04): Bullish Golden Cross (short-term is 4% above long-term)
       - < 0 (-0.03): Bearish Death Cross (short-term is dragging below long-term)
    """
    sma_short = close_series.rolling(window=short_window).mean()
    sma_long = close_series.rolling(window=long_window).mean()
    sma_ratio = (sma_short - sma_long) / sma_long
    return sma_ratio


# =============================================================================
# 2. MOVING AVERAGE CONVERGENCE DIVERGENCE (macd)
# =============================================================================
def calculate_macd(close_series: pd.Series, fast_span: int = 12, slow_span: int = 26) -> pd.Series:
    """
    Q: What is MACD?
       A momentum oscillator measuring the gap between a fast (12-day) and
       slow (26-day) Exponential Moving Average (EMA).

    Q: On what values is it calculated?
       On daily 'Close' prices using exponential weighting (recent days have more weight).

    Q: What does it do?
       Measures whether buying pressure is accelerating or decelerating:
       - macd > 0: Fast momentum is accelerating above slow momentum.
       - macd < 0: Selling momentum is accelerating downward.
    """
    ema_fast = close_series.ewm(span=fast_span, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow_span, adjust=False).mean()
    return ema_fast - ema_slow


# =============================================================================
# 3. RELATIVE STRENGTH INDEX (rsi)
# =============================================================================
def calculate_rsi(close_series: pd.Series, window: int = 14) -> pd.Series:
    """
    Q: What is RSI?
       A momentum oscillator bounded strictly between 0 and 100 that measures
       the magnitude of recent gains vs. recent losses over 14 days.

    Q: On what values is it calculated?
       On daily price changes: delta = Close[t] - Close[t-1].
       Separates green days (gains) and red days (losses):
       RS = Average Gain / Average Loss
       RSI = 100 - (100 / (1 + RS))

    Q: What do the 70 and 30 thresholds mean?
       - RSI > 70 ("OVERBOUGHT"): Like an exhausted runner who sprinted 2 miles at top speed.
         Buyers bought too fast, cash is exhausted, and price is prone to pulling back down.
       - RSI < 30 ("OVERSOLD"): Like a rubber band stretched down to the floor.
         Panic sellers exhausted their supply, stock is heavily discounted, and it is
         primed to SNAP BACK UP.
       - RSI ~ 50: Neutral equilibrium.
    """
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


# =============================================================================
# 4. DAILY PERCENTAGE RETURN (daily_return)
# =============================================================================
def calculate_daily_return(close_series: pd.Series) -> pd.Series:
    """
    Q: What is daily_return?
       The 1-day percentage movement: (Close[t] - Close[t-1]) / Close[t-1].

    Q: Why is it used?
       Provides immediate feedback on today's short-term price velocity and impulse.
    """
    return close_series.pct_change().fillna(0.0)


# =============================================================================
# 5. VOLUME Z-SCORE (volume_z_score)
# =============================================================================
def calculate_volume_zscore(volume_series: pd.Series, window: int = 20) -> pd.Series:
    """
    Q: What is Volume Z-Score?
       A standardized Gaussian score measuring how many standard deviations today's
       trading volume is above or below its 20-day backward-looking average.

    Q: On what values is it calculated?
       On daily 'Volume' using rolling 20-day mean and standard deviation:
       Z = (Volume - Rolling Mean) / Rolling StdDev

    Q: What does it do?
       - Z ~ 0.0: Normal, everyday retail volume.
       - Z > 3.0 (WHALE ALERT): Extreme volume spike occurring in <0.3% of normal days.
         Signals institutional funds or banks actively entering or exiting.
    """
    rolling_mean = volume_series.rolling(window=window, min_periods=window).mean()
    rolling_std = volume_series.rolling(window=window, min_periods=window).std()
    zscore = (volume_series - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore.fillna(0.0)


# =============================================================================
# 6. ROLLING SHARPE RATIO (rolling_sharpe)
# =============================================================================
def calculate_rolling_sharpe(daily_returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Q: What is Rolling Sharpe Ratio?
       Measures return divided by volatility over a sliding 20-day window,
       annualized using sqrt(252 trading days).

    Q: Why is it used as an AI feature? (The "Market Weather Meter")
       A stock can rise in two different ways:
       - Scenario A: Calm highway (smooth steady +1% daily gains) -> Rolling Sharpe > +1.5.
       - Scenario B: Stormy gravel (wild swings: -8%, +11%, -7%) -> Rolling Sharpe < 0.0.
       XGBoost uses this as a regime filter: it trusts trend signals during smooth weather
       and becomes defensive during turbulent market storms.
    """
    rolling_mean = daily_returns.rolling(window=window).mean()
    rolling_std = daily_returns.rolling(window=window).std()
    rolling_sharpe = (rolling_mean / (rolling_std + 1e-6)) * np.sqrt(252)
    return rolling_sharpe.fillna(0.0)


# =============================================================================
# 7. ROLLING MAXIMUM DRAWDOWN (rolling_max_drawdown)
# =============================================================================
def calculate_rolling_max_drawdown(close_series: pd.Series, window: int = 20) -> pd.Series:
    """
    Q: What is Rolling Maximum Drawdown?
       Measures the percentage drop of today's closing price from the highest price
       reached over the last 20 days:
       (Close[t] - RollingMax[20]) / RollingMax[20]

    Q: What does it do?
       - Value = 0.0: Today is sitting at a brand-new 20-day high!
       - Value = -0.12: Stock is currently 12% below its recent 20-day peak.
       Tells the AI whether the asset is in a deep correction or breaking out.
    """
    rolling_max = close_series.rolling(window=window).max()
    drawdown = (close_series - rolling_max) / rolling_max
    return drawdown.fillna(0.0)


# =============================================================================
# 8. AVERAGE TRUE RANGE (atr_20)
# =============================================================================
def calculate_atr(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Q: What is Average True Range (ATR)?
       Measures absolute daily trading span in dollars ($), accounting for overnight gaps.

    Q: On what values is it calculated?
       True Range (TR) = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
       ATR_20 = 20-day rolling mean of True Range.

    Q: What does it do?
       Tells the AI the raw dollar volatility bandwidth of the stock:
       - If TSLA ATR = $12.50, it typically swings $12.50 per day between high and low.
    """
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window).mean().fillna(0.0)


# =============================================================================
# SUMMARY PRINT DEMONSTRATION
# =============================================================================
def print_feature_guide():
    guide = """
========================================================================================
             STREAMQUANT XGBOOST FEATURE CHEAT SHEET (ALL 8 FEATURES)
========================================================================================
1. sma_ratio            | Trend Momentum spread: (SMA_10 - SMA_50) / SMA_50.
                        | >0 = Golden cross / Bullish; <0 = Death cross / Bearish.
                        
2. macd                 | Momentum acceleration: EMA_12 - EMA_26.
                        | Measures whether buying speed is expanding or shrinking.

3. rsi                  | 14-day oscillator bounded between 0 and 100.
                        | >70 = Overbought (exhausted rally); <30 = Oversold (snap-back rally).

4. daily_return         | 1-day percentage movement: (Close_today - Close_yesterday) / Close_yesterday.
                        | Immediate velocity and short-term price impulse.

5. volume_z_score       | Standardized volume vs. 20-day baseline: (Volume - Mean) / StdDev.
                        | Z > 3.0 flags institutional whale accumulation/dumping.

6. rolling_sharpe       | 20-day annualized Sharpe ratio (mean return / volatility).
                        | Market Weather: >1.5 = Calm highway trend; <0 = Stormy gravel.

7. rolling_max_drawdown | Percentage drop from 20-day peak: (Close - Peak) / Peak (<= 0.0).
                        | Identifies deep pullbacks vs. all-time high breakouts.

8. atr_20               | 20-day Average True Range in dollars ($).
                        | Raw dollar volatility width and typical daily trading span.
----------------------------------------------------------------------------------------
TARGET: price_went_up   | 1 if Tomorrow Close > Today Close else 0.
========================================================================================
Full documentation available at: ml_models/FEATURE_DICTIONARY.md
"""
    print(guide)


if __name__ == "__main__":
    print_feature_guide()
