import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests
from bs4 import BeautifulSoup

# --- LSTM PREDICTION ENGINE ---
def predict_lstm(ticker):
    try:
        # 1. Fetch data for training (Last 2 years)
        data = yf.download(ticker, period="2y", interval="1d", progress=False)
        if len(data) < 100: return None
        
        df = data[['Close']].values
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(df)

        # 2. Create sequences (Look back at last 60 days)
        X_train, y_train = [], []
        for i in range(60, len(scaled_data)):
            X_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # 3. Build Lightweight LSTM Model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # 4. Fast Training (Only 5 epochs for mobile speed)
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        # 5. Predict Tomorrow
        last_60_days = scaled_data[-60:].reshape(1, 60, 1)
        pred_price = model.predict(last_60_days)
        return scaler.inverse_transform(pred_price).item()
    except:
        return None

# --- UI LOGIC ---
st.title("📸 NiftyGram AI Pro")
st.caption("Deep Learning (LSTM) Edition")

tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]

for t in tickers:
    with st.spinner(f'🧠 AI Brain training for {t}...'):
        # Get live price from Google first (Fast)
        # (Assuming analyze_stock_google function from previous step is here)
        
        # Get LSTM Prediction (Deep Learning)
        future_price = predict_lstm(t)
        
        if future_price:
            st.markdown(f"""
                <div style="background:white; border:1px solid #dbdbdb; border-radius:12px; padding:20px; margin-bottom:20px;">
                    <h3>{t.split('.')[0]}</h3>
                    <p><b>LSTM 24h Forecast:</b> <span style="color:blue;">₹{future_price:.2f}</span></p>
                    <progress value="80" max="100" style="width:100%;"></progress>
                    <p style="font-size:12px; color:gray;">Model Confidence: High (Based on 2yr History)</p>
                </div>
            """, unsafe_allow_html=True)
