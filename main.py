import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

# --- 1. CONFIG ---
st.set_page_config(page_title="NiftyGram AI Pro", layout="wide")

# --- 2. AI PREDICTION ENGINE (LSTM) ---
@st.cache_data(ttl=3600)
def get_quick_pred(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        df = data[['Close']].values
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(df)
        
        X_train = np.array([scaled_data[-61:-1, 0]]).reshape(1, 60, 1)
        y_train = np.array([scaled_data[-1, 0]])
        
        model = Sequential([LSTM(30, input_shape=(60, 1)), Dense(1)])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=1, verbose=0)
        
        input_data = scaled_data[-60:].reshape(1, 60, 1)
        p1 = model.predict(input_data, verbose=0)
        new_input = np.append(input_data[:, 1:, :], p1.reshape(1,1,1), axis=1)
        p2 = model.predict(new_input, verbose=0)
        
        res = scaler.inverse_transform(np.array([p1[0,0], p2[0,0]]).reshape(-1, 1))
        return res.flatten().tolist()
    except:
        return [0, 0]

# --- 3. DATA SNAPSHOT ---
@st.cache_data(ttl=600)
def get_ranked_forecasts():
    tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS"
    ]
    data = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            price = stock.fast_info['last_price']
            change_pct = ((price - stock.fast_info['previous_close']) / stock.fast_info['previous_close']) * 100
            
            preds = get_quick_pred(t)
            day2_change = ((preds[1] - price) / price) * 100
            
            if day2_change >= 2.0: signal = "✅ BUY"
            elif day2_change <= -2.0: signal = "🚨 SELL"
            else: signal = "⏳ WAIT"
            
            data.append({
                'Ticker': t,
                'Company': t.replace(".NS", ""),
                'Day 2 Price': round(preds[1], 2),
                'Live Price': round(price, 2),
                'Change %': round(change_pct, 2),
                'Signal': signal
            })
        except: continue
    return pd.DataFrame(data)

# --- 4. UI LAYOUT ---
st.title("📊 Nifty 50 Execution Dashboard")

# Sidebar Capital Input
with st.sidebar:
    st.header("💰 Capital Allocator")
    user_budget = st.number_input("Enter Trading Capital (₹)", min_value=0, value=500000, step=10000)
    st.divider()
    st.info("Input your total budget to see the share quantity for BUY signals.")

with st.spinner("Analyzing AI Signals..."):
    full_data = get_ranked_forecasts()
    # Rank them for display
    gainers = full_data.sort_values(by='Change %', ascending=False).head(10).copy()
    gainers.insert(0, 'SL', range(1, len(gainers) + 1))

# Display Tables
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### 🔥 AI Strategy Board")
    st.dataframe(
        gainers[['SL', 'Company', 'Day 2 Price', 'Live Price', 'Signal']], 
        hide_index=True, use_container_width=True
    )

with col2:
    st.markdown("#### ⚡ Execution Plan")
    # Filter for BUY signals
    buy_list = gainers[gainers['Signal'] == "✅ BUY"]
    
    if not buy_list.empty:
        selected_buy = st.selectbox("Pick a BUY Stock", buy_list['Company'])
        stock_row = buy_list[buy_list['Company'] == selected_buy].iloc[0]
        
        live_p = stock_row['Live Price']
        qty = int(user_budget // live_p)
        total_inv = qty * live_p
        
        st.success(f"**Strategy for {selected_buy}**")
        st.metric("Quantity to Buy", f"{qty} Shares")
        st.metric("Total Investment", f"₹{total_inv:,.2f}")
        st.write(f"Leftover Cash: ₹{user_budget - total_inv:,.2f}")
    else:
        st.warning("No ✅ BUY signals found in the top list currently.")

st.divider()
st.caption("Disclaimer: Calculations are based on current market prices and AI forecasts. Brokerage charges not included.")
