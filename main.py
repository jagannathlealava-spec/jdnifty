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

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="NiftyGram Pro", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .card { background: white; border: 1px solid #dbdbdb; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .gainer-text { color: #28a745; font-weight: bold; }
    .loser-text { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

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
            price = stock.fast_info['last_price']
            change = ((price - stock.fast_info['previous_close']) / stock.fast_info['previous_close']) * 100
            data.append({'Ticker': t, 'Price': price, 'Change': change})
        except: continue
    df = pd.DataFrame(data)
    return df.sort_values(by='Change', ascending=False), tickers

@st.cache_data(ttl=3600)
def get_lstm_prediction(ticker):
    try:
        data = yf.download(ticker, period="2y", interval="1d", progress=False)
        df = data[['Close']].values
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(df)
        X_train, y_train = [], []
        for i in range(60, len(scaled_data)):
            X_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
        X_train = np.array(X_train).reshape(-1, 60, 1)
        model = Sequential([LSTM(50, return_sequences=True, input_shape=(60,1)), LSTM(50), Dense(1)])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, np.array(y_train), epochs=1, batch_size=32, verbose=0)
        input_data = scaled_data[-60:].reshape(1, 60, 1)
        pred = model.predict(input_data, verbose=0)
        return scaler.inverse_transform(pred).item()
    except: return None

# --- 3. MAIN APP LOGIC ---
st.title("🚀 Nifty 50 War Room")
snapshot_df, all_tickers = get_nifty_snapshot()

# Sidebar - Hiding all 50 stocks here
with st.sidebar:
    st.header("📂 Nifty 50 List")
    selected_stock = st.selectbox("Select Stock to Analyze", all_tickers)
    st.divider()
    st.info("Select a stock here to trigger the AI Prediction Brain.")

# Top 10 Gainers & Losers Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 Top 10 Gainers")
    st.table(snapshot_df.head(10)[['Ticker', 'Change']].assign(Change=lambda x: x['Change'].map("{:,.2f}%".format)))

with col2:
    st.subheader("❄️ Top 10 Losers")
    st.table(snapshot_df.tail(10)[['Ticker', 'Change']].assign(Change=lambda x: x['Change'].map("{:,.2f}%".format)))

st.divider()

# Selected Stock Feed
st.subheader(f"🧠 AI Forecast Feed: {selected_stock}")
with st.spinner(f"AI Brain crunching numbers for {selected_stock}..."):
    future_price = get_lstm_prediction(selected_stock)
    live_price = snapshot_df[snapshot_df['Ticker'] == selected_stock]['Price'].values[0]

if future_price:
    rocket = "🚀" if future_price > live_price else "📉"
    color = "green" if future_price > live_price else "red"
    st.markdown(f"""
<div class="card">
    <h2 style="margin:0;">{selected_stock.split('.')[0]}</h2>
    <p style="color:gray;">Current Price: ₹{live_price:,.2f}</p>
    <div style="background:#f0f7ff; padding:15px; border-radius:10px; border-left: 5px solid {color};">
        <h4 style="margin:0; color:{color};">{rocket} LSTM 24h Prediction: ₹{future_price:,.2f}</h4>
    </div>
</div>
""", unsafe_allow_html=True)
