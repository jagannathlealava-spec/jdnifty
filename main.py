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
    
    /* Wheel Container Styling */
    .wheel-container { position: relative; width: 300px; height: 300px; margin: 30px auto; }
    .pointer {
        position: absolute; top: -15px; left: 50%; transform: translateX(-50%);
        width: 0; height: 0; border-left: 15px solid transparent;
        border-right: 15px solid transparent; border-top: 30px solid #28a745; z-index: 10;
    }
    .wheel-graphic {
        width: 100%; height: 100%; border-radius: 50%; border: 5px solid #333;
        background: conic-gradient(#e1306c 0 45deg, #fff 45deg 90deg, #e1306c 90deg 135deg, #fff 135deg 180deg, #e1306c 180deg 225deg, #fff 225deg 270deg, #e1306c 270deg 315deg, #fff 315deg 360deg);
        display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px;
    }
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
    data = yf.download(tickers, period="30d", interval="1d", group_by='ticker', progress=False)
    results = []
    for t in tickers:
        try:
            hist = data[t]['Close'].dropna()
            live_price = hist.iloc[-1]
            y = hist.values
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            p1 = slope * (len(y)) + intercept
            p2 = slope * (len(y) + 1) + intercept
            gain_loss = ((p2 - live_price) / live_price) * 100
            results.append({
                'Stock': t.replace(".NS", ""),
                'Today Price': round(live_price, 2),
                'Day 2 Price': round(p2, 2),
                '% Forecast': round(gain_loss, 2)
            })
        except: continue
    return pd.DataFrame(results)

# --- 3. UI DASHBOARD ---
st.title("⚡ Nifty 50 Turbo Pro Dashboard")
st.write(f"📍 Aizawl, Mizoram | {datetime.date.today()}")

if st.button('🚀 RUN HIGH-SPEED SCAN'):
    with st.spinner('Ranking 50 stocks...'):
        full_df = get_turbo_analysis()
        st.session_state.gainers = full_df[full_df['% Forecast'] > 0].sort_values('% Forecast', ascending=False).head(10).copy()
        st.session_state.gainers.insert(0, 'SL', range(1, len(st.session_state.gainers) + 1))
        st.session_state.losers = full_df[full_df['% Forecast'] < 0].sort_values('% Forecast', ascending=True).head(10).copy()
        st.session_state.losers.insert(0, 'SL', range(1, len(st.session_state.losers) + 1))

# Display Tables
if 'gainers' in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.success("📈 **Top 10 Ranked Gainers**")
        st.dataframe(st.session_state.gainers, hide_index=True, use_container_width=True)
    with col2:
        st.error("📉 **Top 10 Ranked Losers**")
        st.dataframe(st.session_state.losers, hide_index=True, use_container_width=True)

    # --- 4. THE SPINNING WHEEL GAME ---
    st.divider()
    st.subheader("🎡 The Nifty Fortune Wheel")
    st.write("Spin to pick one of the Top 10 Gainers to Buy!")

    wheel_placeholder = st.empty()
    
    # Static Wheel View
    wheel_placeholder.markdown("""
        <div class="wheel-container">
            <div class="pointer"></div>
            <div class="wheel-graphic">NIFTY 50</div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🎰 SPIN FOR A BUY"):
        top_gainers = st.session_state.gainers['Stock'].tolist()
        
        # Simulated Spin Animation
        for i in range(15):
            pick = np.random.choice(top_gainers)
            wheel_placeholder.markdown(f"""
                <div class="wheel-container">
                    <div class="pointer"></div>
                    <div class="wheel-graphic" style="transform: rotate({i*45}deg);">{pick}</div>
                </div>
            """, unsafe_allow_html=True)
            time.sleep(0.1)
        
        # Final Result
        final_pick = np.random.choice(top_gainers)
        wheel_placeholder.markdown(f"""
            <div class="wheel-container">
                <div class="pointer"></div>
                <div class="wheel-graphic" style="border: 5px solid #28a745; background: #f0fff4;">{final_pick}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.balloons()
        win_data = st.session_state.gainers[st.session_state.gainers['Stock'] == final_pick].iloc[0]
        qty = int(500000 // win_data['Today Price'])
        st.success(f"✅ **AI Pick: {final_pick}** | Buy **{qty}** shares at ₹{win_data['Today Price']}. Target: ₹{win_data['Day 2 Price']}")
else:
    st.info("Click 'Run High-Speed Scan' to fill the wheel with today's winners!")
