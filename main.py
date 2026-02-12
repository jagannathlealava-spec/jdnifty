import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# --- 1. CONFIG ---
st.set_page_config(page_title="Nifty 50 Turbo", layout="wide")

# --- 2. FAST DATA ENGINE ---
@st.cache_data(ttl=300) # Fast 5-minute cache
def get_turbo_analysis():
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
    
    # Fast Bulk Download
    data = yf.download(tickers, period="30d", interval="1d", group_by='ticker', progress=False)
    
    results = []
    for t in tickers:
        try:
            hist = data[t]['Close'].dropna()
            live_price = hist.iloc[-1]
            
            # Fast Vectorized Trend (Linear Slope)
            y = hist.values
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            
            # Predict Day 1 and Day 2
            pred_day1 = slope * (len(y)) + intercept
            pred_day2 = slope * (len(y) + 1) + intercept
            
            gain_loss = ((pred_day2 - live_price) / live_price) * 100
            
            results.append({
                'Stock': t.replace(".NS", ""),
                'Today Price': round(live_price, 2),
                'Day 1 Pred': round(pred_day1, 2),
                'Day 2 Pred': round(pred_day2, 2),
                '% Forecast': round(gain_loss, 2)
            })
        except: continue
    return pd.DataFrame(results)

# --- 3. UI DASHBOARD ---
st.title("⚡ Nifty 50 Turbo Forecast")
st.write(f"📍 Aizawl | All 50 Stocks | {datetime.date.today()}")

if st.button('🚀 Run High-Speed Market Scan'):
    with st.spinner('Calculating momentum for 50 stocks...'):
        df = get_turbo_analysis()
        
        gainers = df[df['% Forecast'] > 0].sort_values('% Forecast', ascending=False)
        losers = df[df['% Forecast'] < 0].sort_values('% Forecast', ascending=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.success("📈 **Fast Gainers**")
        st.dataframe(gainers.assign(Rank=range(1, len(gainers)+1)).set_index('Rank'), use_container_width=True)

    with col2:
        st.error("📉 **Fast Losers**")
        st.dataframe(losers.assign(Rank=range(1, len(losers)+1)).set_index('Rank'), use_container_width=True)

    # DIVERSIFICATION MODE (₹5,00,000)
    st.divider()
    st.subheader("💰 Smart Allocation (₹5,00,000)")
    top_3 = gainers.head(3)
    
    if not top_3.empty:
        budget = 500000 / len(top_3)
        cols = st.columns(3)
        for i, (idx, row) in enumerate(top_3.iterrows()):
            qty = int(budget // row['Today Price'])
            with cols[i]:
                st.info(f"**{row['Stock']}**")
                st.metric("Buy Quantity", f"{qty}")
                st.metric("Total Cost", f"₹{qty * row['Today Price']:,.0f}")
                st.write(f"Signal: +{row['% Forecast']}%")
else:
    st.info("The Turbo Engine is ready. Click the button to analyze the full market in seconds.")
