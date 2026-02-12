import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

# --- 1. CONFIG ---
st.set_page_config(page_title="NiftyGram Pro", layout="wide")

# --- 2. DATA ENGINES ---
@st.cache_data(ttl=600)
def get_nifty_snapshot():
    tickers = [
        "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
        "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
        "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
        "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
        "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS",
        "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LTIM.NS",
        "LT.NS", "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS",
        "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
        "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
        "TECHM.NS", "TITAN.NS", "UPL.NS", "ULTRACEMCO.NS", "WIPRO.NS"
    ]
    data = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.fast_info
            price = info['last_price']
            prev_close = info['previous_close']
            change = ((price - prev_close) / prev_close) * 100
            data.append({'Stock': t.replace(".NS", ""), 'Price': round(price, 2), 'Change %': round(change, 2)})
        except: continue
    
    full_df = pd.DataFrame(data)
    
    # Generate Top 10 Gainers
    gainers = full_df.sort_values(by='Change %', ascending=False).head(10).copy()
    gainers.insert(0, 'SL', range(1, 11))
    
    # Generate Top 10 Losers
    losers = full_df.sort_values(by='Change %', ascending=True).head(10).copy()
    losers.insert(0, 'SL', range(1, 11))
    
    return gainers, losers, tickers

# --- 3. UI LAYOUT ---
st.title("📊 Nifty 50 Rank Board")
st.write(f"Updated: {datetime.datetime.now().strftime('%H:%M:%S')} | Aizawl Time")

# Get Ranked Data
top_gainers, top_losers, all_tickers = get_nifty_snapshot()

# Sidebar for the full list
with st.sidebar:
    st.header("📂 Market Explorer")
    selected_stock = st.selectbox("Detailed AI Prediction", all_tickers)
    st.divider()
    st.caption("The top rankers update every 10 minutes.")

# --- TOP 10 DISPLAY ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔥 Top 10 Gainers")
    # Styling the table to look clean
    st.dataframe(top_gainers, hide_index=True, use_container_width=True)

with col2:
    st.markdown("#### ❄️ Top 10 Losers")
    st.dataframe(top_losers, hide_index=True, use_container_width=True)

st.divider()

# --- SELECTED STOCK ANALYSIS ---
if selected_stock:
    st.subheader(f"🧠 Deep Analysis: {selected_stock.replace('.NS', '')}")
    # (Insert your LSTM prediction logic here as used in previous versions)
    st.info("AI is monitoring this stock. Check the 48h forecast in the cards below.")
