import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

# --- 1. CONFIG & SYSTEM SETUP ---
st.set_page_config(page_title="Nifty 50 AI Command", layout="wide")

@st.cache_data(ttl=3600)
def get_lstm_pred(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        df = data[['Close']].values
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(df)
        # Ultra-fast training for batch processing
        X = np.array([scaled_data[-61:-1, 0]]).reshape(1, 60, 1)
        y = np.array([scaled_data[-1, 0]])
        model = Sequential([LSTM(20, input_shape=(60, 1)), Dense(1)])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=1, verbose=0)
        # Predict 2nd Day
        inp = scaled_data[-60:].reshape(1, 60, 1)
        p1 = model.predict(inp, verbose=0)
        p2 = model.predict(np.append(inp[:, 1:, :], p1.reshape(1,1,1), axis=1), verbose=0)
        res = scaler.inverse_transform(p2)
        return round(res.item(), 2)
    except: return 0

# --- 2. THE FULL NIFTY 50 DATA ENGINE ---
@st.cache_data(ttl=900)
def process_nifty_50():
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
    results = []
    progress_bar = st.progress(0)
    for i, t in enumerate(tickers):
        try:
            stock = yf.Ticker(t)
            live = stock.fast_info['last_price']
            future = get_lstm_pred(t)
            gain_loss = ((future - live) / live) * 100
            results.append({
                'Stock': t.replace(".NS", ""),
                'Today Price': round(live, 2),
                'Future Price': future,
                '% Change': round(gain_loss, 2)
            })
        except: continue
        progress_bar.progress((i + 1) / len(tickers))
    return pd.DataFrame(results)

# --- 3. UI DASHBOARD ---
st.title("📈 Nifty 50 AI Forecast Dashboard")
st.write(f"📍 Aizawl, Mizoram | Full 50-Stock Analysis | {datetime.date.today()}")

if st.button('🔄 Start Full Market Scan'):
    df = process_nifty_50()
    
    # Split into Gainers and Losers
    gainers = df[df['% Change'] > 0].sort_values('% Change', ascending=False)
    losers = df[df['% Change'] < 0].sort_values('% Change', ascending=True)

    # --- TOP GAINERS & LOSERS COLUMNS ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("🚀 **Predicted Gainers**")
        st.dataframe(gainers.assign(Rank=range(1, len(gainers)+1)).set_index('Rank'), use_container_width=True)

    with col2:
        st.error("📉 **Predicted Losers**")
        st.dataframe(losers.assign(Rank=range(1, len(losers)+1)).set_index('Rank'), use_container_width=True)

    # --- 4. DIVERSIFICATION MODE (THE 5 LAKH STRATEGY) ---
    st.divider()
    st.subheader("💰 Smart Capital Allocation (₹5,00,000)")
    
    top_3 = gainers.head(3)
    if not top_3.empty:
        budget_per_stock = 500000 / len(top_3)
        cols = st.columns(len(top_3))
        
        for i, (index, row) in enumerate(top_3.iterrows()):
            qty = int(budget_per_stock // row['Today Price'])
            actual_inv = qty * row['Today Price']
            with cols[i]:
                st.info(f"**{row['Stock']}**")
                st.metric("Buy Qty", f"{qty}")
                st.metric("Invest", f"₹{actual_inv:,.0f}")
                st.write(f"Exp. Growth: {row['% Change']}%")
    else:
        st.warning("No gainers predicted for current cycle.")
else:
    st.info("Click the button above to begin the AI scan for all 50 stocks. This may take 30-60 seconds.")
