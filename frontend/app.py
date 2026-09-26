import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/api/stocks"

# MUST BE FIRST LINE: Makes the app take up the whole screen
st.set_page_config(page_title="StreamQuant Terminal", page_icon="⚡", layout="wide")

# --- 1. THE SIDEBAR (Moves controls out of the way) ---
with st.sidebar:
    st.title("⚡ StreamQuant")
    st.markdown("Engineered for low-latency financial intelligence.")
    st.divider()
    ticker = st.text_input("🔍 Search Ticker:", "TSLA").upper()
    fetch_btn = st.button("Run Intelligence Pipeline", use_container_width=True)

# --- 2. MAIN DASHBOARD ---
if fetch_btn:
    with st.spinner(f"Querying PostgreSQL for {ticker}..."):
        try:
            response = requests.get(f"{API_URL}/{ticker}?limit=50")
            
            if response.status_code == 200:
                data = response.json()["data"]
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').set_index('date')
                
                st.header(f"📊 {ticker} Intelligence Report")
                
                # --- 3. KPI METRICS (The "Fintech" Look) ---
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest
                
                price_change = latest['close'] - prev['close']
                pct_change = (price_change / prev['close']) * 100
                
                # Creates 4 columns at the top for big numbers
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Closing Price", f"${latest['close']:.2f}", f"{price_change:.2f} ({pct_change:.2f}%)")
                
                # Logic for trend indicator
                trend = "Bullish" if latest['close'] > latest['sma_10'] else "Bearish"
                col2.metric("10-Day Trend (SMA)", f"${latest['sma_10']:.2f}", trend, 
                            delta_color="normal" if trend == "Bullish" else "inverse")
                
                col3.metric("Volume", f"{int(latest['volume']):,}")
                
                # Z-Score Anomaly Logic
                z_score = latest['volume_z_score']
                col4.metric("Volume Z-Score", f"{z_score:.2f}σ", "ANOMALY" if z_score > 2.5 else "Normal", 
                            delta_color="inverse" if z_score > 2.5 else "normal")
                
                # Show flashing red alert if anomaly is detected!
                if z_score > 2.5:
                    st.error(f"🚨 **ANOMALY DETECTED:** Trading volume is {z_score:.2f} standard deviations above normal. High probability of institutional activity.")

                # --- 4. ADVANCED CHARTS ---
                st.divider()
                st.subheader("Price Action vs. Machine Learning Moving Averages")
                
                # Plot the Close price AND the moving averages on the exact same chart
                chart_data = df[['close', 'sma_10', 'sma_50']]
                st.line_chart(chart_data, color=["#FFFFFF", "#00FF00", "#FF0000"]) # White, Green, Red lines
                
                # --- 5. TABS (Hiding the ugly data) ---
                st.divider()
                tab1, tab2 = st.tabs(["Raw Feature Data", "API Developer View"])
                with tab1:
                    st.dataframe(df[['open', 'high', 'low', 'close', 'volume', 'sma_10', 'sma_50', 'volume_z_score']], use_container_width=True)
                with tab2:
                    st.json(response.json())
                    
            elif response.status_code == 404:
                st.warning(f"No data found in the database for {ticker}.")
        except requests.exceptions.ConnectionError:
            st.error("Backend server error. Is FastAPI running?")
else:
    st.title("⚡ Welcome to the StreamQuant Terminal")
    st.markdown("Use the sidebar on the left to search for an asset and run the ML pipeline.")