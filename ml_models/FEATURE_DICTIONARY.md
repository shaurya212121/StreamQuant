# 📘 StreamQuant XGBoost Feature Dictionary & Quant Guide

> **Official Developer Reference Document**  
> *A comprehensive, plain-English reference explaining every technical indicator, ratio, and formula used in the XGBoost classification model.*

---

## 📌 Table of Contents
1. [Target Variable (`price_went_up`)](#1-target-variable-price_went_up)
2. [Simple Moving Average Ratio (`sma_ratio`)](#2-simple-moving-average-ratio-sma_ratio)
3. [Moving Average Convergence Divergence (`macd`)](#3-moving-average-convergence-divergence-macd)
4. [Relative Strength Index (`rsi`)](#4-relative-strength-index-rsi)
5. [Daily Percentage Return (`daily_return`)](#5-daily-percentage-return-daily_return)
6. [Volume Z-Score (`volume_z_score`)](#6-volume-z-score-volume_z_score)
7. [Rolling Sharpe Ratio (`rolling_sharpe`)](#7-rolling-sharpe-ratio-rolling_sharpe)
8. [Rolling Maximum Drawdown (`rolling_max_drawdown`)](#8-rolling-maximum-drawdown-rolling_max_drawdown)
9. [Average True Range (`atr_20`)](#9-average-true-range-atr_20)
10. [Summary Feature Matrix Table](#10-summary-feature-matrix-table)

---

## 1. Target Variable (`price_went_up`)

* **What it is:** The ground-truth answer key the XGBoost model tries to predict for tomorrow.
* **On what values is it calculated:** The **`Close`** price of today vs. the **`Close`** price of tomorrow.
* **Mathematical Formula:**
  $$\text{Tomorrow's Close} = \text{Close}_{t+1}$$
  $$\text{Target } y_t = \begin{cases} 1 & \text{if } \text{Close}_{t+1} > \text{Close}_t \quad (\text{BULLISH / Price went UP}) \\ 0 & \text{if } \text{Close}_{t+1} \le \text{Close}_t \quad (\text{BEARISH / Price went DOWN}) \end{cases}$$
* **Why it's used:** It frames stock forecasting as a **Supervised Binary Classification** task ($y \in \{0, 1\}$) instead of predicting raw dollar values, which is much more stable and less prone to noise.

---

## 2. Simple Moving Average Ratio (`sma_ratio`)

### A. What is a Simple Moving Average (SMA)?
A Simple Moving Average takes the average closing price over the last $N$ days.
* **10-day SMA (`sma_10`):** The average price of the last 10 trading days (measures **short-term momentum**).
* **50-day SMA (`sma_50`):** The average price of the last 50 trading days (measures **long-term trend**).

### B. Why is it used?
Stock prices bounce around randomly every single day. The moving average acts as a **low-pass filter**: it smooths out the chaotic daily noise and reveals the underlying direction.

### C. On what values is it calculated?
On the daily **`Close`** price series:
$$\text{SMA}_{10} = \frac{1}{10} \sum_{i=0}^{9} \text{Close}_{t-i}, \quad \text{SMA}_{50} = \frac{1}{50} \sum_{i=0}^{49} \text{Close}_{t-i}$$

### D. Why do we feed `sma_ratio` instead of raw `sma_10` and `sma_50`?
* **The Dollar Trap:** If Tesla is $\$200$ and Nvidia is $\$900$, raw dollar SMAs confuse the model across different stocks.
* **The Ratio Fix:** We calculate the **percentage spread**:
  $$\text{sma\_ratio} = \frac{\text{SMA}_{10} - \text{SMA}_{50}}{\text{SMA}_{50}}$$
* **How to interpret it:**
  * **$\text{sma\_ratio} > 0$ (e.g., $+0.04$):** Short-term average is 4% above long-term average $\rightarrow$ **Bullish Golden Cross / Upward Momentum**.
  * **$\text{sma\_ratio} < 0$ (e.g., $-0.03$):** Short-term average is below long-term average $\rightarrow$ **Bearish Death Cross / Downward Drag**.

---

## 3. Moving Average Convergence Divergence (`macd`)

### A. What is it?
MACD is a momentum oscillator that tracks the difference between two **Exponential Moving Averages (EMA)**. Unlike a simple average, an Exponential Moving Average gives **more weight to the most recent days**.

### B. On what values is it calculated?
On the daily **`Close`** price using 12-day and 26-day spans:
$$\text{EMA}_{12} = \text{Exponential Moving Average over 12 days}$$
$$\text{EMA}_{26} = \text{Exponential Moving Average over 26 days}$$
$$\text{MACD} = \text{EMA}_{12} - \text{EMA}_{26}$$

### C. What does it do?
It tells the AI whether **buying pressure is speeding up or slowing down**:
* When `macd` turns positive and grows larger: Buyers are aggressively pushing prices up.
* When `macd` turns negative: Sellers have taken control.

---

## 4. Relative Strength Index (`rsi`)

### A. What is it?
A momentum oscillator bounded strictly between **0 and 100** that measures the speed and magnitude of recent price changes over a 14-day lookback window.

### B. On what values is it calculated?
On daily closing price changes ($\Delta = \text{Close}_t - \text{Close}_{t-1}$):
$$\text{Average Gain} = \text{Rolling 14-day average of positive gains}$$
$$\text{Average Loss} = \text{Rolling 14-day average of negative drops}$$
$$\text{RS} = \frac{\text{Average Gain}}{\text{Average Loss}}$$
$$\text{RSI} = 100 - \left( \frac{100}{1 + \text{RS}} \right)$$

### C. The Thresholds (70 to 30 Rule):
* **RSI > 70 ("Overbought"):** 
  * *Analogy:* An exhausted marathon runner sprinting at top speed. 
  * *Meaning:* Buyers bought too aggressively. The rally is overheated, buyers are running out of cash, and the price is prone to pulling back down.
* **RSI < 30 ("Oversold"):** 
  * *Analogy:* A rubber band stretched all the way to the floor. 
  * *Meaning:* Sellers panic-dumped too hard. Selling pressure is exhausted, the stock is heavily discounted, and it is primed to **snap back UP**.
* **RSI ~ 50:** Neutral equilibrium.

---

## 5. Daily Percentage Return (`daily_return`)

### A. What is it?
The standard 1-day percentage movement of the stock from yesterday's close to today's close.

### B. On what values is it calculated?
$$\text{daily\_return} = \frac{\text{Close}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}$$

### C. Why is it used?
It gives the model immediate feedback on **today's short-term impulse**. A sudden $+5\%$ jump or $-4\%$ drop signals high immediate volatility that influences tomorrow's price continuation or reversal.

---

## 6. Volume Z-Score (`volume_z_score`)

### A. What is it?
A standardized statistical score that measures whether today's trading volume is normal or abnormally massive compared to the last 20 days.

### B. On what values is it calculated?
On the daily **`Volume`** column using a backward-looking 20-day rolling window:
$$\mu_{20} = \text{Rolling 20-day Mean Volume}$$
$$\sigma_{20} = \text{Rolling 20-day Standard Deviation of Volume}$$
$$\text{volume\_z\_score} = \frac{\text{Volume}_t - \mu_{20}}{\sigma_{20}}$$

### C. What does it do?
* **$Z \approx 0.0$:** Completely normal, everyday retail trading volume.
* **$Z > +3.0\sigma$ (Institutional Whale Alert):**
  * Less than $0.3\%$ of days produce volume 3 standard deviations above normal.
  * When $Z > 3.0$, it indicates that a massive institutional fund, bank, or "whale" is aggressively accumulating or dumping millions of shares.

---

## 7. Rolling Sharpe Ratio (`rolling_sharpe`)

### A. What is it?
The Sharpe Ratio calculated continuously over a **sliding 20-day backward window**, annualized using $\sqrt{252}$ trading days.

### B. On what values is it calculated?
On the 20-day series of **`daily_return`**:
$$\text{rolling\_mean} = \text{Rolling 20-day mean of daily returns}$$
$$\text{rolling\_std} = \text{Rolling 20-day standard deviation of daily returns}$$
$$\text{rolling\_sharpe} = \left( \frac{\text{rolling\_mean}}{\text{rolling\_std} + 10^{-6}} \right) \times \sqrt{252}$$

### C. Why is it needed in the model?
It acts as the AI's **"Market Weather Meter"**:
* **High Rolling Sharpe ($> +1.5$):** The price is climbing on a smooth, calm highway with low volatility. Trends are healthy and predictable.
* **Low / Negative Rolling Sharpe ($< 0.0$):** The market is choppy, violent, and erratic. The model learns to be defensive and distrust fake breakout rallies during stormy weather.

---

## 8. Rolling Maximum Drawdown (`rolling_max_drawdown`)

### A. What is it?
Measures the percentage drop of today's closing price from the **highest price recorded over the last 20 days**.

### B. On what values is it calculated?
On the daily **`Close`** price series:
$$\text{rolling\_max}_{20} = \max(\text{Close}_{t-19}, \dots, \text{Close}_t)$$
$$\text{rolling\_max\_drawdown} = \frac{\text{Close}_t - \text{rolling\_max}_{20}}{\text{rolling\_max}_{20}}$$

### C. What does it do?
* It is always **$\le 0$** (a negative number or zero).
* **$\text{Value} = 0.0$:** Today's price is at a new 20-day high!
* **$\text{Value} = -0.15$:** The stock is currently down 15% from its recent peak.
* *Why the AI needs it:* Helps the model detect if the stock is in a steep correction or if it is breaking out into all-time highs.

---

## 9. Average True Range (`atr_20`)

### A. What is it?
An institutional volatility indicator developed by J. Welles Wilder that measures the **absolute daily trading span (in dollars)**, accounting for overnight price gaps.

### B. On what values is it calculated?
Uses today's **`High`**, **`Low`**, and yesterday's **`Close`**:
$$\text{True Range (TR)} = \max\Big( \text{High}_t - \text{Low}_t, \; |\text{High}_t - \text{Close}_{t-1}|, \; |\text{Low}_t - \text{Close}_{t-1}| \Big)$$
$$\text{ATR}_{20} = \text{Rolling 20-day average of True Range}$$

### C. What does it do?
* If Tesla has an $\text{ATR}_{20} = \$12.50$, it means on a typical day, Tesla swings by $\$12.50$ between its peak and trough.
* *Why the AI needs it:* Informs the model about the **raw volatility bandwidth** of the asset, helping it adjust confidence and distinguish quiet consolidation from explosive volatility.

---

## 10. Summary Feature Matrix Table

| Feature Name | Category | Inputs Used | Output Range | What XGBoost Learns |
| :--- | :--- | :--- | :---: | :--- |
| **`sma_ratio`** | Trend Momentum | `Close` (10d & 50d SMA) | Unlimited (e.g. $-0.10$ to $+0.15$) | Golden cross vs. Death cross trend spread |
| **`macd`** | Trend Acceleration | `Close` (12d & 26d EMA) | Continuous float | Is buying/selling momentum accelerating |
| **`rsi`** | Reversal / Oscillator | `Close` (14d gains/losses) | **0 to 100** | Overbought ($>70$) vs. Oversold ($<30$) snap-backs |
| **`daily_return`** | Short-term Impulse | `Close` ($t$ vs. $t-1$) | Percentage (e.g. $-0.03$ to $+0.04$) | Immediate 1-day price velocity |
| **`volume_z_score`** | Institutional Flow | `Volume` (20d mean/std) | Standard deviation $\sigma$ ($Z$) | Flags institutional whale accumulation ($Z > 3.0$) |
| **`rolling_sharpe`**| Regime Quality | `daily_return` (20d mean/std) | $-3.0$ to $+5.0$ | Calm highway trend vs. chaotic gravel storm |
| **`rolling_max_drawdown`** | Pullback Depth | `Close` (20d max) | $\le 0.0$ (e.g. $-0.18$) | How deeply the stock has fallen from recent highs |
| **`atr_20`** | Absolute Volatility | `High`, `Low`, `Close` (20d) | Positive dollar amount ($\$$) | Daily trading range and swing volatility width |
