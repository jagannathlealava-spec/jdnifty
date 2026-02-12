import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import datetime

# --- 1. CONFIG & SYSTEM SETUP ---
st.set_page_config(page_title="Nifty 50 Turbo", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #fafafa; }
    .card { background: white; border: 1px solid #dbdbdb; border-radius: 12px; padding: 15px; margin-bottom: 20px; }
    .stMetric { background: #f8f9fa; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 2. THE TURBO DATA ENGINE ---
@st.cache_data(ttl=300)
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
    
    # Bulk Fetch 30 days of data in ONE go
    data = yf.download(tickers, period="30d", interval="1d", group_by='ticker', progress=False)
    
    results = []
    for t in tickers:
        try:
            hist = data[t]['Close'].dropna()
            live_price = hist.iloc[-1]
            
            # Vectorized Linear Regression (Slope)
            y = hist.values
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            
            # Predict Days
            p1 = slope * (len(y)) + intercept
            p2 = slope * (len(y) + 1) + intercept
            gain_loss = ((p2 - live_price) / live_price) * 100
            
            results.append({
                'Stock': t.replace(".NS", ""),
                'Today Price': round(live_price, 2),
                'Day 1 Price': round(p1, 2),
                'Day 2 Price': round(p2, 2),
                '% Forecast': round(gain_loss, 2)
            })
        except: continue
    return pd.DataFrame(results)

# --- 3. UI DASHBOARD & SESSION STATE ---
st.title("⚡ Nifty 50 Turbo AI Dashboard")
st.write(f"📍 Aizawl, Mizoram | Full 50-Stock Scan | {datetime.date.today()}")

if st.button('🚀 RUN HIGH-SPEED SCAN'):
    with st.spinner('Ranking 50 stocks by momentum...'):
        full_df = get_turbo_analysis()
        
        # Save results to Session State so the Spin Game can find them
        st.session_state.gainers = full_df[full_df['% Forecast'] > 0].sort_values('% Forecast', ascending=False).head(10).copy()
        st.session_state.gainers.insert(0, 'SL', range(1, len(st.session_state.gainers) + 1))

        st.session_state.losers = full_df[full_df['% Forecast'] < 0].sort_values('% Forecast', ascending=True).head(10).copy()
        st.session_state.losers.insert(0, 'SL', range(1, len(st.session_state.losers) + 1))

# --- 4. DISPLAY TABLES ---
if 'gainers' in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.success("📈 **Top 10 Ranked Gainers**")
        st.dataframe(st.session_state.gainers, hide_index=True, use_container_width=True)
    with col2:
        st.error("📉 **Top 10 Ranked Losers**")
        st.dataframe(st.session_state.losers, hide_index=True, use_container_width=True)

    # --- 5. THE SPIN GAME (WHICH TO BUY?) ---
    st.divider()
    st.subheader("🎡 The Nifty Fortune Wheel")
    st.write("Let AI pick which stock to buy with your ₹5,00,000!")

    if st.button("🎰 SPIN THE WHEEL"):
        potential_buys = st.session_state.gainers['Stock'].tolist()
        
        # Simulated Spin Animation
        placeholder = st.empty()
        for i in range(12):
            temp_pick = np.random.choice(potential_buys)
            placeholder.markdown(f"<h1 style='text-align: center; color: #e1306c;'>🎡 {temp_pick}</h1>", unsafe_allow_html=True)
            time.sleep(0.1)
        
        # Final Selection
        final_pick = np.random.choice(potential_buys)
        placeholder.markdown(f"<h1 style='text-align: center; color: #28a745;'>✅ BUY: {final_pick}</h1>", unsafe_allow_html=True)
        st.balloons()
        
        # Calculation for ₹5 Lakhs
        win_data = st.session_state.gainers[st.session_state.gainers['Stock'] == final_pick].iloc[0]
        qty = int(500000 // win_data['Today Price'])
        st.success(f"**Strategy for {final_pick}:** Buy **{qty}** shares at ₹{win_data['Today Price']}. Predicted 48h trend: **+{win_data['% Forecast']}%**")

else:
    st.info("The engine is warm! Click 'Run High-Speed Scan' to start your session.")
