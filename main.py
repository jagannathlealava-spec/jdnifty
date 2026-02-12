import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Initialize AI Brains
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title=NiftyGram AI, layout=centered)

# --- APP STYLING ---
st.markdown(
    style
    .stApp { background-color #fafafa; }
    .stock-card {
        # --- APP STYLING ---
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
    }
    style
, unsafe_allow_html=True)
}

# --- DATA ENGINE ---
@st.cache_data(ttl=600) # Cache data for 10 minutes to save speed
def analyze_stock(ticker)
    try
        t = yf.Ticker(ticker)
        hist = t.history(period=1y, interval=1d, auto_adjust=True)
        if hist.empty return None
        
        # Math Prediction
        y = hist['Close'].values.reshape(-1, 1)
        X = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        pred = model.predict([[len(y)]]).item()
        
        # Sentiment
        news = t.news[3]
        sent = sum([sia.polarity_scores(n['title'])['compound'] for n in news])  3 if news else 0
        
        return {price y[-1].item(), pred pred, sent sent}
    except return None

# --- UI HEADER & STORIES ---
st.title(📸 NiftyGram)
tickers = [RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS]

# Story Bar (Top Gainers)
cols = st.columns(len(tickers))
for i, t in enumerate(tickers)
    with cols[i]
        st.markdown(fdiv style='border2px solid #e1306c; border-radius50%; width50px; height50px; displayflex; align-itemscenter; justify-contentcenter; marginauto; font-size10px; font-weightbold;'{t.split('.')[0]}div, unsafe_allow_html=True)

st.divider()

# --- UI THE FEED ---
for t in tickers
    res = analyze_stock(t)
    if res
        with st.container()
            st.markdown(f
                div class=stock-card
                    h3 style=margin0;{t.split('.')[0]}h3
                    p style=colorgray; font-size12px;Aizawl, Mizoram • AI Prediction Activep
                    hr
                    h2 style=color{'green' if res['pred']  res['price'] else 'red'};
                        ₹{res['price'].2f} 
                        span style=font-size15px;({'▲' if res['pred']  res['price'] else '▼'} Expected)span
                    h2
                    pbMarket Moodb {'Bullish 🟢' if res['sent']  0 else 'Bearish 🔴'}p
                div
            , unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            if c1.button(f❤️ Buy {t.split('.')[0]})
                st.confetti()
                st.success(fOrder Simulated for {t}!)
