import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import requests
import os

# 1. CRITICAL: Fix for yfinance caching on Streamlit Cloud
import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

# 2. Setup AI Mood Reader
@st.cache_resource
def load_sia():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_sia()

st.set_page_config(page_title="NiftyGram AI", layout="centered")

# 3. --- DATA ENGINE (The Browser-Mimic Version) ---
def analyze_stock(ticker):
    try:
        # Create a session that looks like a real Chrome browser
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
        })
        
        t = yf.Ticker(ticker, session=session)
        
        # Use a 6-month window (More reliable than 1 year for quick loads)
        hist = t.history(period="6mo", interval="1d")
        
        if hist.empty:
            return None
        
        # Prediction Logic
        prices = hist['Close'].values.reshape(-1, 1)
        X = np.arange(len(prices)).reshape(-1, 1)
        model = LinearRegression().fit(X, prices)
        pred = model.predict([[len(prices)]]).item()
        
        # Sentiment Logic
        news = t.news
        sent = 0
        if news:
            scores = [sia.polarity_scores(n['title'])['compound'] for n in news[:3]]
            sent = sum(scores) / len(scores)
            
        return {"price": prices[-1].item(), "pred": pred, "sent": sent}
    except Exception as e:
        return None

# 4. --- APP UI ---
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
    data = analyze_stock(t)
    
    if data:
        color = "green" if data['pred'] > data['price'] else "red"
        st.markdown(f"""
            <div style="background:white; border:1px solid #dbdbdb; border-radius:12px; padding:20px; margin-bottom:20px;">
                <h3 style="margin:0;">{t.split('.')[0]}</h3>
                <h2 style="color:{color}; margin:10px 0;">₹{data['price']:.2f}</h2>
                <p style="margin:0;">AI Target: <b>₹{data['pred']:.2f}</b></p>
                <p style="margin:0; font-size:14px;">Mood: {'🟢 Bullish' if data['sent'] > 0 else '🔴 Bearish'}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"⚠️ Yahoo Finance blocked {t}. Try again in 1 minute.")
