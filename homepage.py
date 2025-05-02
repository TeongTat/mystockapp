import streamlit as st
import requests
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import PIL
import datetime
from PIL import Image
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

#set tittle page
st.title("Stock Predictor App")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Introduction", "S&P 500 Stock Info", "Price Prediction"])

import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from PIL import Image
from statsmodels.tsa.arima.model import ARIMA
import datetime

st.set_page_config(layout="wide")
st.title("Stock Predictor App")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Introduction", "S&P 500 Stock Info", "Price Prediction"])

# Introduction Tab
with tab1:
    image = Image.open("stock_market_banner_970x250.png")
    st.image(image, use_container_width=True)
    st.write("""
This app helps investors analyze and predict S&P 500 stock trends.

Features:
- View historical price trends for S&P 500 companies
- Predict next 5 days’ prices using ARIMA

Data is sourced from Yahoo Finance.
""")

# Tab 2: S&P 500 Info
with tab2:
    st.subheader("S&P 500 Stock Info:")

    @st.cache_data
    def load_sp500_tickers_names():
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)[0]
        tickers_names = dict(zip(table["Symbol"], table["Security"]))
        return tickers_names

    def get_stock_data(ticker):
        ticker = ticker.replace(".", "-")
        for _ in range(3):
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            if not hist.empty:
                return hist
        return pd.DataFrame()

    tickers_names = load_sp500_tickers_names()
    selected = st.selectbox("Select a stock ticker:", [f"{t} - {n}" for t, n in tickers_names.items()])
    ticker_symbol = selected.split(" - ")[0]

    if ticker_symbol:
        st.subheader(f"Stock Data for {ticker_symbol}")
        df = get_stock_data(ticker_symbol)
        if df.empty:
            st.error("No data found. Please try another stock.")
        else:
            st.write(df)
            st.subheader("Closing Price Chart")
            st.line_chart(df["Close"])
            st.subheader("Volume Chart")
            st.line_chart(df["Volume"])

# Tab 3: Prediction
with tab3:
    st.subheader("Price Prediction: Select and Click-Predict")
    st.image("bull_bear_01.jpg", use_container_width=True)

    def load_sp500_stocks():
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)[0]
        return dict(zip(table['Security'], table['Symbol']))

    sp500_stocks = load_sp500_stocks()
    stock_name = st.selectbox("Select a stock:", list(sp500_stocks.keys()))
    stock_symbol = sp500_stocks[stock_name].replace(".", "-")

    start_date = st.date_input("Select start date", datetime.date(2023, 1, 1))
    end_date = st.date_input("Select end date", datetime.date.today())

    @st.cache_data
    def fetch_stock_data(symbol, start, end, retries=3):
        for _ in range(retries):
            data = yf.download(symbol, start=start, end=end)
            if not data.empty:
                return data
        return pd.DataFrame()

    if st.button("Predict"):
        st.subheader(f"Fetching Data for {stock_symbol}...")
        data = fetch_stock_data(stock_symbol, start_date, end_date)

        if data.empty:
            st.error("No data found! Try selecting a different stock or date range.")
        else:
            st.write("Last 5 rows of historical data:")
            st.write(data.tail())

            stock_prices = data[['Close', 'High', 'Low']].dropna()

            def train_arima(series):
                model = ARIMA(series, order=(5, 1, 0))
                return model.fit()

            model_close = train_arima(stock_prices['Close'])
            model_high = train_arima(stock_prices['High'])
            model_low = train_arima(stock_prices['Low'])

            forecast_close = model_close.forecast(steps=5)
            forecast_high = model_high.forecast(steps=5)
            forecast_low = model_low.forecast(steps=5)

            future_dates = pd.date_range(stock_prices.index[-1], periods=6)[1:]
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Close Price': forecast_close,
                'Predicted High Price': forecast_high,
                'Predicted Low Price': forecast_low
            }).set_index("Date")

            st.subheader("Forecast for Next 5 Days")
            st.write(forecast_df)

            st.subheader("Forecast Chart")
            fig, ax = plt.subplots(figsize=(10, 5))
            stock_prices['Close'][-50:].plot(ax=ax, label="Historical Close", color="blue")
            forecast_df["Predicted Close Price"].plot(ax=ax, label="Forecast Close", color="red", linestyle="--")
            forecast_df["Predicted High Price"].plot(ax=ax, label="Forecast High", color="green", linestyle="--")
            forecast_df["Predicted Low Price"].plot(ax=ax, label="Forecast Low", color="orange", linestyle="--")
            ax.set_title(f"Stock Price Prediction for {stock_symbol}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)
