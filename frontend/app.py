import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

API_BASE = "http://127.0.0.1:8000/api"

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
            # 1. Fetch Unified Intelligence from Gateway
            intel_response = requests.get(f"{API_BASE}/stocks/{ticker}/intelligence")
            # 2. Fetch Historical Price Data
            hist_response = requests.get(f"{API_BASE}/stocks/{ticker}?limit=50")
            
            if hist_response.status_code != 200:
                error_detail = hist_response.json().get("detail", "Unknown error") if hist_response.headers.get("content-type", "").startswith("application/json") else hist_response.text
                st.warning(f"No historical data found for {ticker}: {error_detail}")
            else:
                hist_data = hist_response.json()["data"]
                df = pd.DataFrame(hist_data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').set_index('date')
                
                # Parse Intelligence or fallback to legacy prediction
                if intel_response.status_code == 200:
                    intel_data = intel_response.json()
                    ml_layer = intel_data["ml_trend_layer"]
                    confluence = intel_data["confluence_decision"]
                    anomaly_layer = intel_data["statistical_anomaly_layer"]
                    sentiment_layer = intel_data["nlp_sentiment_layer"]
                    
                    signal = ml_layer["prediction"]
                    confidence = ml_layer["confidence_pct"]
                    top_drivers = ml_layer.get("top_drivers", [])
                    is_anomaly = anomaly_layer["is_anomaly"]
                    z_score = anomaly_layer["volume_z_score"]
                    confluence_signal = confluence["signal"]
                    confluence_summary = confluence["summary"]
                else:
                    # Fallback to direct prediction endpoint
                    pred_res = requests.get(f"{API_BASE}/stocks/{ticker}/prediction")
                    if pred_res.status_code != 200:
                        error_detail = pred_res.json().get("detail", "Unknown error") if pred_res.headers.get("content-type", "").startswith("application/json") else pred_res.text
                        st.error(f"Prediction unavailable for {ticker}: {error_detail}")
                    pred_data = pred_res.json() if pred_res.status_code == 200 else {}
                    signal = pred_data.get("trend_prediction", "NEUTRAL")
                    confidence = pred_data.get("confidence_pct", 50.0)
                    is_anomaly = pred_data.get("is_volume_anomaly", False)
                    z_score = pred_data.get("volume_z_score", 0.0)
                    confluence_signal = signal
                    confluence_summary = "Signal generated directly from XGBoost trend model."
                    top_drivers = []
                    intel_data = pred_data
                    sentiment_layer = {"status": "UNAVAILABLE", "sentiment_score": 0.0}

                st.header(f"📊 {ticker} Intelligence Report")
                
                # --- CONFLUENCE BANNER ---
                if "BUY" in confluence_signal:
                    st.success(f"🎯 **UNIFIED SIGNAL: {confluence_signal}** | Model Confidence: **{confidence}%**\n\n_{confluence_summary}_")
                elif "SELL" in confluence_signal:
                    st.error(f"⚠️ **UNIFIED SIGNAL: {confluence_signal}** | Model Confidence: **{confidence}%**\n\n_{confluence_summary}_")
                else:
                    st.info(f"⚖️ **UNIFIED SIGNAL: {confluence_signal}** | Model Confidence: **{confidence}%**\n\n_{confluence_summary}_")
                    
                if is_anomaly:
                    st.warning(f"🚨 **STATISTICAL ANOMALY DETECTED:** Volume Z-Score is **{z_score:.2f}σ** (Institutional Whale Activity!)")
                
                st.divider()
                
                # --- KPI METRIC CARDS ---
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest
                
                price_change = latest['close'] - prev['close']
                pct_change = (price_change / prev['close']) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Closing Price", f"${latest['close']:.2f}", f"{price_change:.2f} ({pct_change:.2f}%)")
                
                sma_spread = float(latest['sma_ratio']) if 'sma_ratio' in latest and pd.notna(latest['sma_ratio']) else 0.0
                trend = "Bullish" if sma_spread > 0 else "Bearish"
                col2.metric("SMA Momentum Ratio", f"{sma_spread:.2%}", trend, 
                            delta_color="normal" if trend == "Bullish" else "inverse")
                
                col3.metric("Daily Volume", f"{int(latest['volume']):,}")
                
                col4.metric("Volume Z-Score", f"{z_score:.2f}σ" if z_score else "N/A", 
                            "ANOMALY" if is_anomaly else "Normal", 
                            delta_color="inverse" if is_anomaly else "normal")

                # --- PRICE & MOMENTUM CHART (PLOTLY) ---
                st.subheader("Price Action History")
                try:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df.index, 
                        y=df['close'], 
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#00FFAA', width=2.5)
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0E1117",
                        plot_bgcolor="#0E1117",
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=400,
                        xaxis=dict(showgrid=True, gridcolor="#262730"),
                        yaxis=dict(showgrid=True, gridcolor="#262730", tickprefix="$")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.line_chart(df['close'])
                
                # --- DEVELOPER & AI EXPLAINABILITY TABS ---
                st.divider()
                tab1, tab2, tab3 = st.tabs(["Engineered ML Features (v2)", "SHAP Explainability Drivers", "Raw Gateway Payload"])
                with tab1:
                    feature_cols = [c for c in ['open', 'high', 'low', 'close', 'volume', 'sma_ratio', 'macd', 'rsi', 'volume_z_score', 'rolling_sharpe', 'rolling_max_drawdown', 'atr_20'] if c in df.columns]
                    st.dataframe(df[feature_cols], use_container_width=True)
                with tab2:
                    if top_drivers:
                        st.markdown("#### Top Features Driving Today's AI Decision")
                        for d in top_drivers:
                            feat_name = d.get('feature', 'N/A')
                            val = d.get('current_value', 0.0)
                            impact = d.get('shap_impact', 0.0)
                            arrow = "🟢 Bullish Driver" if impact > 0 else "🔴 Bearish Drag"
                            st.write(f"- **{feat_name.upper()}** (Value: `{val}`): Contribution = `{impact:+.4f}` ({arrow})")
                    else:
                        st.info("SHAP drivers will appear once models are evaluated with explainability.")
                with tab3:
                    st.json(intel_data)
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI. Make sure your backend server is running on port 8000!")
else:
    st.title("⚡ Welcome to the StreamQuant Terminal")
    st.markdown("Use the sidebar on the left to query an asset and trigger the distributed ML pipeline.")
