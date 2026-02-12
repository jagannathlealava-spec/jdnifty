import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# 1. Setup
@st.cache_resource
def setup_sia():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = setup_sia()

st.set_page_config(page_title="NiftyGram AI", layout="centered")

# 2. --- APP STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #fafafa; }
    .stock-card {
        background-color: white; border: 1px solid #dbdbdb;
        border-radius: 12px; padding: 20px; margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. --- DATA ENGINE ---
def analyze_stock(ticker):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1y", interval="1d", auto_adjust=True)
        if hist.empty: return None
        
        y = hist['Close'].values.reshape(-1, 1)
        X = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        pred = model.predict([[len(y)]]).item()
        
        news = t.news
        sent = 0
        if news:
            scores = [sia.polarity_scores(n['title'])['compound'] for n in news[:3]]
            sent = sum(scores) / len(scores)
            
        return {"price": y[-1].item(), "pred": pred, "sent": sent}
    except:
        return None

# 4. --- UI ---
st.title("📸 NiftyGram")
tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

# Story Bar
cols = st.columns(len(tickers))
for i, t in enumerate(tickers):
    with cols[i]:
        st.markdown(f"<div style='border:2px solid #e1306c; border-radius:50%; width:50px; height:50px; display:flex; align-items:center; justify-content:center; margin:auto; font-size:10px; font-weight:bold; background:white;'>{t.split('.')[0]}</div>", unsafe_allow_html=True)

st.divider()

# The Feed
for t in tickers:
    # --- THIS PART FIXES THE NAMEERROR ---
    analysis_result = analyze_stock(t) 
    
    if analysis_result is not None:
        color = "green" if analysis_result['pred'] > analysis_result['price'] else "red"
        st.markdown(f"""
            <div class="stock-card">
                <h3>{t.split('.')[0]}</h3>
                <h2 style="color:{color};">₹{analysis_result['price']:.2f}</h2>
                <p>AI Target: <b>₹{analysis_result['pred']:.2f}</b></p>
                <p>Mood: {'🟢 Bullish' if analysis_result['sent'] > 0 else '🔴 Bearish'}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"Could not load {t}. Check connection.")
