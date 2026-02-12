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
@st.cache_data(ttl=3600) # Cache the model for 1 hour to keep it fast
def get_lstm_prediction(ticker):
    try:
        # Get 2 years of data for deep learning
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

        # Build LSTM Model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, np.array(y_train), epochs=2, batch_size=32, verbose=0)

        # Predict next 2 days
        input_data = scaled_data[-60:].reshape(1, 60, 1)
        pred_days = []
        for _ in range(2):
            p = model.predict(input_data, verbose=0)
            pred_days.append(p[0,0])
            # Update input data with the new prediction for the next step
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
        headers = {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        price = soup.find("div", {"class": "YMlKec fxKbKc"}).text.replace("₹", "").replace(",", "")
        return float(price)
    except:
        return None

# --- 3. UI DASHBOARD ---
st.set_page_config(page_title="NiftyGram AI Pro", layout="centered")

st.markdown(f"""
<div class="card">
<h3 style="margin:0; color:#e1306c;">{t.split('.')[0]}</h3>
<p style="color:gray; font-size:12px; margin-bottom:10px;">Today's Live NSE Price</p>
<p class="price-today">₹{live_price:,.2f}</p>
<div class="pred-box">
<p style="margin:0; font-size:13px; font-weight:bold; color:#007bff;">🚀 AI 48-Hour Forecast</p>
<table style="width:100%; margin-top:5px;">
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

st.title("📸 NiftyGram AI Pro")
st.write(f"📍 Aizawl, Mizoram | {datetime.date.today().strftime('%d %b %Y')}")

tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

for t in tickers:
    with st.container():
        # Fetching Data
        live_price = get_google_price(t)
        
        with st.spinner(f'Analyzing {t} patterns...'):
            future_preds = get_lstm_prediction(t)

        if live_price and future_preds:
            st.markdown(f"""
                <div class="card">
                    <h3 style="margin:0; color:#e1306c;">{t.split('.')[0]}</h3>
                    <p style="color:gray; font-size:12px; margin-bottom:10px;">Today's Live NSE Price</p>
                    <p class="price-today">₹{live_price:,.2f}</p>
                    
                    <div class="pred-box">
                        <p style="margin:0; font-size:13px; font-weight:bold; color:#007bff;">🚀 AI 48-Hour Forecast</p>
                        <table style="width:100%; margin-top:5px;">
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
            
            # Action Buttons
            c1, c2 = st.columns(2)
            c1.button(f"Trade {t.split('.')[0]}", key=f"t_{t}", use_container_width=True)
            c2.button(f"Set Alert", key=f"a_{t}", use_container_width=True)
        else:
            st.error(f"Waiting for {t} data stream...")

st.divider()
st.caption("Disclaimer: AI predictions are for educational purposes. Markets involve risk.")
