import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="Nifty 50 Ranked Turbo", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #fafafa; }
    .card { background: white; border: 1px solid #dbdbdb; border-radius: 12px; padding: 15px; margin-bottom: 20px; }
    .gainer-val { color: #22c55e; font-weight: bold; }
    .loser-val { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. FAST VECTORIZED ENGINE ---
@st.cache_data(ttl=300)
def get_ranked_analysis():
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
    
    # Bulk Fetch 30 days of data
    data = yf.download(tickers, period="30d", interval="1d", group_by='ticker', progress=False)
    
    results = []
    for t in tickers:
        try:
            hist = data[t]['Close'].dropna()
            live_price = hist.iloc[-1]
            
            # High-speed Trend Math
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
                'Day 1 Price': round(pred_day1, 2),
                'Day 2 Price': round(pred_day2, 2),
                '% Forecast': round(gain_loss, 2)
            })
        except: continue
    return pd.DataFrame(results)

# --- 3. UI DASHBOARD ---
st.title("⚡ Ranked Nifty 50 Turbo Forecast")
st.write(f"📍 Aizawl | Sorted by Momentum | {datetime.date.today().strftime('%d %b %Y')}")

if st.button('🚀 Execute High-Speed Scan'):
    with st.spinner('Ranking 50 stocks by trend strength...'):
        full_df = get_ranked_analysis()
        
        # 1. Rank Gainers (Highest % Forecast first)
        gainers = full_df[full_df['% Forecast'] > 0].sort_values('% Forecast', ascending=False).head(10).copy()
        gainers.insert(0, 'SL', range(1, len(gainers) + 1))

        # 2. Rank Losers (Most Negative % Forecast first)
        losers = full_df[full_df['% Forecast'] < 0].sort_values('% Forecast', ascending=True).head(10).copy()
        losers.insert(0, 'SL', range(1, len(losers) + 1))

    col1, col2 = st.columns(2)
    
    with col1:
        st.success("📈 **Top 10 Ranked Gainers**")
        st.dataframe(gainers, hide_index=True, use_container_width=True)

    with col2:
        st.error("📉 **Top 10 Ranked Losers**")
        st.dataframe(losers, hide_index=True, use_container_width=True)

    # --- 4. THE SPIN GAME (WHICH ONE TO BUY?) ---
st.divider()
st.subheader("🎡 The Nifty Fortune Wheel")
st.write("Can't decide where to put your ₹5 Lakhs? Let the AI spin for you!")

if not gainers.empty:
    if st.button("🎰 SPIN THE WHEEL"):
        # Get list of top 10 gainers
        potential_buys = gainers['Stock'].tolist()
        
        # 1. Animation Loop
        with st.empty():
            for i in range(15):  # Flash different stocks to simulate spinning
                random_pick = np.random.choice(potential_buys)
                st.markdown(f"<h1 style='text-align: center; color: #e1306c;'>🎡 {random_pick}</h1>", unsafe_allow_html=True)
                import time
                time.sleep(0.1) # Fast spin
            
            # 2. The Final Result
            final_pick = np.random.choice(potential_buys)
            st.markdown(f"<h1 style='text-align: center; color: #28a745;'>✅ BUY: {final_pick}</h1>", unsafe_allow_html=True)
            st.balloons() # Victory celebration!
            
            # 3. Quick Trading Card for the winner
            win_row = gainers[gainers['Stock'] == final_pick].iloc[0]
            st.info(f"**AI Recommendation:** Buying **{final_pick}** today shows a predicted trend of **{win_row['% Forecast']}%**. With ₹5 Lakhs, you can grab approximately **{int(500000 // win_row['Today Price'])}** shares.")
else:
    st.warning("No gainers found to spin. Market might be in a heavy dip!")

    # SMART CAPITAL ALLOCATOR
    st.divider()
    st.subheader("💰 Strategy Execution (₹5,00,000 Capital)")
    top_3 = gainers.head(3)
    
    if not top_3.empty:
        budget = 500000 / len(top_3)
        cols = st.columns(3)
        for i, (idx, row) in enumerate(top_3.iterrows()):
            qty = int(budget // row['Today Price'])
            with cols[i]:
                st.info(f"**#{i+1} Target: {row['Stock']}**")
                st.metric("Shares to Buy", f"{qty}")
                st.metric("Required Cash", f"₹{qty * row['Today Price']:,.0f}")
                st.write(f"Forecast: +{row['% Forecast']}%")
    else:
        st.warning("Market trend is currently sideways. No strong Buy signals found.")

st.divider()
st.caption("Turbo Engine: Uses Linear Least Squares for momentum detection. Processing time: < 3 seconds.")
