import yfinance as yf
import pandas as pd

def fetch_and_prepare_data(ticker: str, period: str = "6mo"):
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, period=period, interval="1d")
    
    if df.empty:
        print("Error: No data found.")
        return df

    # We only need Close price and Volume for now
    df = df[['Close', 'Volume']].copy()
    
    # ---------------------------------------------------
    # FEATURE 1: For the Trend Prediction Model (XGBoost)
    # ---------------------------------------------------
    # Calculate 10-day and 50-day Simple Moving Averages (SMA)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate daily percentage return
    df['Daily_Return'] = df['Close'].pct_change()

    # ---------------------------------------------------
    # FEATURE 2: For the Anomaly Detection Model (Z-Score)
    # ---------------------------------------------------
    # Calculate 20-day rolling volume average and standard deviation
    df['Vol_20_Mean'] = df['Volume'].rolling(window=20).mean()
    df['Vol_20_Std'] = df['Volume'].rolling(window=20).std()
    
    # Z-Score: (Current Volume - Mean Volume) / Std Dev
    # If Z-Score > 3, it's a massive anomaly (spike in trading)
    df['Volume_Z_Score'] = (df['Volume'] - df['Vol_20_Mean']) / df['Vol_20_Std']
    
    # Drop the empty rows (NaNs) created by the rolling windows
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    # Let's test it on Tesla (TSLA)
    stock_df = fetch_and_prepare_data("TSLA")
    
    print("\n--- Processed ML Features ---")
    # Print the last 5 days of our calculated features
    print(stock_df[['Close', 'SMA_10', 'Volume', 'Volume_Z_Score']].tail())
