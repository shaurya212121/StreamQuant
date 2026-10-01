import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/api/stocks"

# 1. Page Configuration
st.set_page_config(page_title="StreamQuant Terminal", page_icon="⚡", layout="wide")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("⚡ StreamQuant")
    st.markdown("Distributed Financial Intelligence & ML Inference Pipeline")
    st.divider()
    ticker = st.text_input("🔍 Search Ticker:", "TSLA").upper()
    fetch_btn = st.button("Run Intelligence Pipeline", use_container_width=True)

# --- MAIN DASHBOARD ---
if fetch_btn:
    with st.spinner(f"Running Distributed Pipeline for {ticker}..."):
        try:
            # 1. Fetch Historical Price Data from FastAPI
            hist_response = requests.get(f"{API_URL}/{ticker}?limit=50")
            
            # 2. Fetch Live XGBoost Prediction & Anomaly Signal from FastAPI
            pred_response = requests.get(f"{API_URL}/{ticker}/prediction")
            
            if hist_response.status_code == 200 and pred_response.status_code == 200:
                hist_data = hist_response.json()["data"]
                pred_data = pred_response.json()
                
                df = pd.DataFrame(hist_data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').set_index('date')
                
                st.header(f"📊 {ticker} Intelligence Report")
                
                # ==========================================
                # NEW: AI SIGNAL & ANOMALY BANNER
                # ==========================================
                signal = pred_data["trend_prediction"]
                confidence = pred_data["confidence_pct"]
                is_anomaly = pred_data["is_volume_anomaly"]
                z_score = pred_data["volume_z_score"]
                
                if signal == "BULLISH":
                    st.success(f"🤖 **AI SIGNAL: BULLISH (UPWARD BIAS)** | Model Confidence: **{confidence}%**")
                else:
                    st.error(f"🤖 **AI SIGNAL: BEARISH (DOWNWARD BIAS)** | Model Confidence: **{confidence}%**")
                    
                if is_anomaly:
                    st.warning(f"🚨 **STATISTICAL ANOMALY DETECTED:** Volume Z-Score is **{z_score:.2f}σ** (Institutional Volume Spike!)")
                
                st.divider()
                
                # --- KPI METRIC CARDS ---
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest
                
                price_change = latest['close'] - prev['close']
                pct_change = (price_change / prev['close']) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Closing Price", f"${latest['close']:.2f}", f"{price_change:.2f} ({pct_change:.2f}%)")
                
                trend = "Bullish" if latest['close'] > latest['sma_10'] else "Bearish"
                col2.metric("10-Day SMA Trend", f"${latest['sma_10']:.2f}", trend, 
                            delta_color="normal" if trend == "Bullish" else "inverse")
                
                col3.metric("Daily Volume", f"{int(latest['volume']):,}")
                
                col4.metric("Volume Z-Score", f"{z_score:.2f}σ" if z_score else "N/A", 
                            "ANOMALY" if is_anomaly else "Normal", 
                            delta_color="inverse" if is_anomaly else "normal")

                # --- MULTI-LINE PRICE & SMA CHART ---
                st.subheader("Price Action vs. Quantitative Moving Averages")
                chart_data = df[['close', 'sma_10', 'sma_50']]
                st.line_chart(chart_data, color=["#FFFFFF", "#00FF00", "#FF0000"])
                
                # --- DEVELOPER TABS ---
                st.divider()
                tab1, tab2 = st.tabs(["Engineered ML Features", "Raw Model Inference Payload"])
                with tab1:
                    st.dataframe(df[['open', 'high', 'low', 'close', 'volume', 'sma_10', 'sma_50', 'volume_z_score']], use_container_width=True)
                with tab2:
                    st.json(pred_data)
                    
            elif hist_response.status_code == 404:
                st.warning(f"No data found in the database for {ticker}.")
            else:
                st.error("Error communicating with backend services.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI. Make sure your backend server is running on port 8000!")
else:
    st.title("⚡ Welcome to the StreamQuant Terminal")
    st.markdown("Use the sidebar on the left to query an asset and trigger the distributed ML pipeline.")