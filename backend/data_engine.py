import yfinance as yf
import pandas as pd

def fetch_and_prepare_data(ticker: str, period: str = "6mo"):
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, period=period, interval="1d")
    
    if df.empty:
        print("Error: No data found.")
        return df

    df = df[['Close', 'Volume']].copy()
    
    # Trend features
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Daily_Return'] = df['Close'].pct_change()

    # Anomaly features
    df['Vol_20_Mean'] = df['Volume'].rolling(window=20).mean()
    df['Vol_20_Std'] = df['Volume'].rolling(window=20).std()
    df['Volume_Z_Score'] = (df['Volume'] - df['Vol_20_Mean']) / df['Vol_20_Std']
    
    return df.dropna()

if __name__ == "__main__":
    stock_df = fetch_and_prepare_data("TSLA")
    print("\nProcessed Features (TSLA):")
    print(stock_df[['Close', 'SMA_10', 'Volume', 'Volume_Z_Score']].tail())
