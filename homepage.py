import streamlit as st
import requests
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

#set tittle page
st.title("Stock Predictor App")

# Create tabs
tab1, tab2 = st.tabs(["S&P 500 Stock Info", "Price Prediction"])

# Load an image
image_path = "stock_market_banner_970x250.png"
image = Image.open(image_path)
st.image(image, use_container_width=True)


st.write("""
This is a stock price and prediction modelling app that will assist investors on buying or selling the stocks. The stocks are based on all companies listed on S&P 500 and the price are up to date linking from Yahoo Finance server.
The app will display the following:
- Historical stock price trend up to the latest.
- Showcase stock price forecast (up to 10days).

The main purpose of this application is providing price guidance for investors on the price risks and market risks of the S&P 500 stocks.""")



