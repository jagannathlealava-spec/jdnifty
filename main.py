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
import requests

def analyze_stock(ticker):
    try:
        # 1. Add 'User-Agent' to mimic a real browser
        # This prevents Yahoo from blocking the request
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        t = yf.Ticker(ticker, session=session)
        
        # 2. Try fetching a smaller window of data first (1 month)
        hist = t.history(period="1mo", interval="1d", auto_adjust=True)
        
        # If 1mo is empty, the ticker might be wrong or the server is blocked
        if hist.empty:
            return None
        
        # Math: Linear Regression for Trend
        y = hist['Close'].values.reshape(-1, 1)
        X = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        pred = model.predict([[len(y)]]).item()
        
        # Sentiment logic
        news = t.news
        sent = 0
        if news:
            scores = [sia.polarity_scores(n['title'])['compound'] for n in news[:3]]
            sent = sum(scores) / len(scores)
            
        return {"price": y[-1].item(), "pred": pred, "sent": sent}
    except Exception as e:
        # This will print the exact error in your Streamlit logs
        print(f"Error for {ticker}: {e}")
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
