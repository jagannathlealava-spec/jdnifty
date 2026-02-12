import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# 1. Setup AI Sentiment
@st.cache_resource
def load_sia():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_sia()

st.set_page_config(page_title="NiftyGram AI", layout="centered")

# 2. --- GOOGLE FINANCE ENGINE ---
def analyze_stock_google(ticker):
    try:
        # Format for Google (e.g., RELIANCE:NSE)
        symbol = ticker.replace(".NS", "")
        url = f"https://www.google.com/finance/quote/{symbol}:NSE"
        
        # Mimic a real mobile browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Google Finance live price class
        price_element = soup.find("div", {"class": "YMlKec fxKbKc"})
        if not price_element:
            return None
        
        current_price = float(price_element.text.replace("₹", "").replace(",", ""))
        
        # Get historical change to simulate AI trend
        change_element = soup.find("div", {"class": "Jw7Cyc"})
        change_val = 0
        if change_element:
            # Extract percentage from text like "+1.20%"
            change_text = change_element.text.split('%')[0].replace('+', '').replace('-', '')
            change_val = float(change_text) if change_text else 0

        # AI Target: Simple trend-following logic for speed
        target_price = current_price * (1 + (change_val/100 if change_val > 0 else 0.015))
        
        return {
            "price": current_price,
            "pred": target_price,
            "sent": 0.1 if change_val > 0 else -0.1
        }
    except Exception as e:
        return None

# 3. --- UI APP INTERFACE ---
st.title("📸 NiftyGram")
st.caption("Powered by Google Finance Data")

tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

# Story Bar
cols = st.columns(len(tickers))
for i, t in enumerate(tickers):
    with cols[i]:
        st.markdown(f"""
            <div style='border:2px solid #e1306c; border-radius:50%; width:50px; height:50px; 
            display:flex; align-items:center; justify-content:center; margin:auto; 
            font-size:10px; font-weight:bold; background:white;'>
                {t.split('.')[0]}
            </div>
        """, unsafe_allow_html=True)

st.divider()

# The Feed
for t in tickers:
    data = analyze_stock_google(t)
    
    if data:
        color = "green" if data['pred'] > data['price'] else "red"
        st.markdown(f"""
            <div style="background:white; border:1px solid #dbdbdb; border-radius:12px; padding:20px; margin-bottom:20px;">
                <h3 style="margin:0;">{t.split('.')[0]}</h3>
                <p style="color:gray; font-size:12px;">Live from Google Finance • Aizawl</p>
                <h2 style="color:{color}; margin:10px 0;">₹{data['price']:.2f}</h2>
                <p style="margin:0;">AI Potential Target: <b>₹{data['pred']:.2f}</b></p>
                <p style="margin:0; font-size:14px;">Mood: {'🟢 Bullish' if data['sent'] > 0 else '🔴 Bearish'}</p>
            </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        c1.button(f"Trade {t.split('.')[0]}", key=f"t_{t}")
        c2.button(f"Alerts", key=f"a_{t}")
    else:
        st.warning(f"Searching for {t} data on Google...")
