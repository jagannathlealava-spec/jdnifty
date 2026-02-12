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

# --- 1. AI PREDICTION ENGINE (LSTM) ---
@st.cache_data(ttl=3600)
def get_lstm_prediction(ticker):
    try:
        # Fetch 2 years of history for training
        data = yf.download(ticker, period="2y", interval="1d", progress=False)
        if len(data) < 100: return None
        
        # Prepare Data
        df = data[['Close']].values
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(df)

        X_train, y_train = [], []
        for i in range(60, len(scaled_data)):
            X_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
        
        X_train = np.array(X_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build Optimized LSTM Model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        # We use 2 epochs to prevent the Streamlit server from timing out
        model.fit(X_train, np.array(y_train), epochs=2, batch_size=32, verbose=0)

        # Predict next 2 days
        input_data = scaled_data[-60:].reshape(1, 60, 1)
        pred_days = []
        for _ in range(2):
            p = model.predict(input_data, verbose=0)
            pred_days.append(p[0,0])
            # Slide the window forward
            input_data = np.append(input_data[:, 1:, :], p.reshape(1,1,1), axis=1)

        res = scaler.inverse_transform(np.array(pred_days).reshape(-1, 1))
        return res.flatten().tolist()
    except:
        return None

# --- 2. LIVE GOOGLE PRICE ENGINE ---
def get_google_price(ticker):
    try:
        symbol = ticker.replace(".NS", "")
        url = f"https://www.google.com/finance/quote/{symbol}:NSE"
        headers = {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X)'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        price_text = soup.find("div", {"class": "YMlKec fxKbKc"}).text
        return float(price_text.replace("₹", "").replace(",", ""))
    except:
        return None

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="NiftyGram AI Pro", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #fafafa; }
    .card {
        background: white; border: 1px solid #dbdbdb;
        border-radius: 15px; padding: 20px; margin-bottom: 25px;
    }
    .price-today { color: #262626; font-size: 32px; font-weight: bold; margin: 0; }
    .pred-box { 
        background: #f0f7ff; border-left: 5px solid #007bff; 
        border-radius: 8px; padding: 15px; margin-top: 15px; 
    }
</style>
""", unsafe_allow_html=True)

st.title("📸 NiftyGram AI Pro")
st.write(f"📍 Aizawl, Mizoram | {datetime.date.today().strftime('%d %b %Y')}")

tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

# The Feed
for t in tickers:
    display_name = t.split('.')[0]
    live_price = get_google_price(t)
    
    with st.spinner(f'🧠 AI Training for {display_name}...'):
        future_preds = get_lstm_prediction(t)

    if live_price and future_preds:
        st.markdown(f"""
<div class="card">
<h3 style="margin:0; color:#e1306c;">{display_name}</h3>
<p style="color:gray; font-size:12px; margin-bottom:5px;">Live NSE Price</p>
<p class="price-today">₹{live_price:,.2f}</p>
<div class="pred-box">
<p style="margin:0; font-size:14px; font-weight:bold; color:#007bff;">🚀 LSTM 2-Day Forecast</p>
<table style="width:100%; margin-top:8px; font-size:15px;">
<tr>
<td>Tomorrow:</td>
<td style="text-align:right; font-weight:bold;">₹{future_preds[0]:,.2f}</td>
</tr>
<tr>
<td>Day After:</td>
<td style="text-align:right; font-weight:bold;">₹{future_preds[1]:,.2f}</td>
</tr>
</table>
</div>
</div>
""", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        c1.button(f"Trade {display_name}", key=f"t_{t}", use_container_width=True)
        c2.button(f"Details", key=f"d_{t}", use_container_width=True)
    else:
        st.error(f"⚠️ Connection lag for {display_name}. Refreshing...")

st.divider()
st.caption("Disclaimer: LSTM predictions use historical trends and are not guaranteed financial advice.")
