import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# 1. Initialize AI Brains
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# 2. Page Config
st.set_page_config(page_title="NiftyGram AI", layout="centered")

# 3. --- APP STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #fafafa; }
    .stock-card {
        background-color: white; 
        border: 1px solid #dbdbdb;
        border-radius: 8px; 
        padding: 15px; 
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 4. --- DATA ENGINE ---
@st.cache_data(ttl=600)
def analyze_stock(ticker):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1y", interval="1d", auto_adjust=True)
        if hist.empty: return None
        
        # Math Prediction (Linear Regression)
        y = hist['Close'].values.reshape(-1, 1)
        X = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        pred = model.predict([[len(y)]]).item()
        
        # Sentiment Analysis
        news = t.news[:3]
        sent = sum([sia.polarity_scores(n['title'])['compound'] for n in news]) / 3 if news else 0
        
        return {"price": y[-1].item(), "pred": pred, "sent": sent}
    except Exception as e:
        return None

# 5. --- UI HEADER & STORIES ---
st.title("📸 NiftyGram")
# Note: These must be in quotes!
tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

# Story Bar (Visual Circle Tickers)
cols = st.columns(len(tickers))
for i, t in enumerate(tickers):
    with cols[i]:
        name = t.split('.')[0]
        st.markdown(f"""
            <div style='border:2px solid #e1306c; border-radius:50%; width:55px; height:55px; 
                        display:flex; align-items:center; justify-content:center; margin:auto; 
                        font-size:10px; font-weight:bold; background:white;'>
                {name}
            </div>
        """, unsafe_allow_html=True)

st.divider()

# 6. --- UI THE FEED ---
for t in tickers:
    res = analyze_stock(t)
    if res:
        with st.container():
            st.markdown(f"""
                <div class="stock-card">
                    <h3 style="margin:0;">{t.split('.')[0]}</h3>
                    <p style="color:gray; font-size:12px;">Aizawl, Mizoram • AI Prediction Active</p>
                    <hr>
                    <h2 style="color:{'green' if res['pred'] > res['price'] else 'red'};">
                        ₹{res['price']:.2f} 
                        <span style="font-size:15px;">({'▲' if res['pred'] > res['price'] else '▼'} Forecast)</span>
                    </h2>
                    <p><b>Market Mood:</b> {'Bullish 🟢' if res['sent'] > 0 else 'Bearish 🔴'}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Action Buttons
            c1, c2 = st.columns(2)
            if c1.button(f"Buy {t.split('.')[0]}", icon="❤️", key=f"buy_{t}"):
                st.confetti()
                st.success(f"Signal sent for {t}!")

            if c2.button("Analysis", icon="📊", key=f"anal_{t}"):
                st.info(f"Analyzing {t} momentum...")
