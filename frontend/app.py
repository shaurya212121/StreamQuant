import streamlit as st
import requests
import pandas as pd

# The URL of your running FastAPI server
API_URL = "http://127.0.0.1:8000/api/stocks"

st.set_page_config(page_title="StreamQuant", layout="wide")

st.title("📈 StreamQuant Intelligence Dashboard")
st.markdown("Real-time financial signals and anomaly detection.")

# 1. Create a search bar for the user
ticker = st.text_input("Enter Stock Ticker (e.g., TSLA, AAPL):", "TSLA").upper()

if st.button("Fetch Data"):
    with st.spinner(f"Fetching data for {ticker} from FastAPI..."):
        
        # 2. Talk to your FastAPI backend
        try:
            response = requests.get(f"{API_URL}/{ticker}?limit=30")
            
            if response.status_code == 200:
                # 3. Convert the JSON into a Pandas table
                data = response.json()["data"]
                df = pd.DataFrame(data)
                
                # Make sure the dates are in order for the chart
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                st.success(f"Successfully loaded data for {ticker}!")
                
                # 4. Draw a beautiful Line Chart of the Closing Price
                st.subheader(f"{ticker} - 30 Day Closing Price")
                
                # Set the date as the index so the chart looks nice
                chart_data = df.set_index('date')
                st.line_chart(chart_data['close'])
                
                # 5. Show the raw mathematical features
                st.subheader("Raw ML Features (Engineered Data)")
                st.dataframe(df[['date', 'close', 'sma_10', 'sma_50', 'volume_z_score']].tail(5))
                
            elif response.status_code == 404:
                st.warning(f"No data found in the database for {ticker}. Please run the data engine first!")
            else:
                st.error("Backend server error. Is FastAPI running?")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI. Please make sure you are running 'uvicorn backend.main:app' in another terminal!")