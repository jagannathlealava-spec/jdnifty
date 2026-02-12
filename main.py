import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# 1. Initialize AI
@st.cache_resource
def load_nltk():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_nltk()

# 2. Page Config
st.set_page_config(page_title="NiftyGram AI", layout="centered")

# 3. --- APP STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #fafafa; }
    .stock-card {
        background-color: white; 
        border: 1px solid #dbdbdb;
        border-radius: 12px; 
        padding: 20px; 
        margin-bottom: 25px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# 4. --- DATA ENGINE ---
def analyze_stock(ticker):
    try:
        t = yf.Ticker(ticker)
        # Fetch 1 year of data
        hist = t.history(period="1y", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 10:
            return None
        
        # Math Prediction
        y = hist['Close'].values.reshape(-1, 1)
        X = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        pred = model.predict([[len(y)]]).item()
        
        # Safe Sentiment Check
        news = t.news
        if news and len(news) > 0:
            scores = [sia.polarity_scores(n['title'])['compound'] for n in news[:3]]
            sent = sum(scores) / len(scores)
        else:
            sent = 0
        
        return {"price": y[-1].item(), "pred": pred, "sent": sent}
    except Exception as e:
        return None

# 5. --- UI HEADER & STORIES ---
st.title("📸 NiftyGram")
tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

# Story Bar
cols = st.columns(len(tickers))
for i, t in enumerate(tickers):
    with cols[i]:
        st.markdown(f"""
            <div style='border:2px solid #e1306c; border-radius:50%; width:55px; height:55px; 
                        display:flex; align-items:center; justify-content:center; margin:auto; 
                        font-size:10px; font-weight:bold; background:white;'>
                {t.split('.')[0]}
            </div>
        """, unsafe_allow_html=True)

st.divider()

# 6. --- UI THE FEED ---
st.subheader("Your AI Forecast Feed")

for t in tickers:
    with st.spinner(f'AI Analyzing {t}...'):
        res
