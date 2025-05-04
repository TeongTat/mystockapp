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
from alpha_vantage.timeseries import TimeSeries

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(layout="wide")  # MUST be the first Streamlit command

API_KEY = "ad0c312fed0549689ae22f4aec9cacd5"  # Replace with your actual key
ts = TimeSeries(key=API_KEY, output_format='pandas')

# -----------------------
# PAGE TITLE
# -----------------------
st.title("Stock Predictor App")

# -----------------------
# TAB SETUP
# -----------------------
tab1, tab2, tab3 = st.tabs(["Introduction", "S&P 500 Stock Info", "Price Prediction"])

# -----------------------
# TAB 1: INTRODUCTION
# -----------------------
with tab1:
    image_path = "stock_market_banner_970x250.png"
    image = Image.open(image_path)
    st.image(image, use_container_width=True)
    st.write("""
    This is a stock price and prediction modelling app that will assist investors on buying or selling the stocks. The stocks are based on all companies listed on S&P 500 and the price are up to date linking from Alpha Vantage API..
  The app will display the following:
  - Historical stock price trend up to the latest - shown on S&P 500 stock info tab
  - Forecast next 5 days using ARIMA model (up to 5days) - shown on Price Prediction tab

The main purpose of this application is providing price guidance for investors on the price risks and market risks of the S&P 500 stocks.
    """)

# -----------------------
# Load S&P 500 tickers
# -----------------------
@st.cache_data
def load_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return dict(zip(table['Security'], table['Symbol']))

sp500_dict = load_sp500_tickers()

# -----------------------
# TAB 2: STOCK INFO
# -----------------------
with tab2:
    st.subheader("S&P 500 Stock Info")

    selected = st.selectbox("Select a stock:", [f"{k} ({v})" for k, v in sp500_dict.items()])
    selected_name, selected_symbol = selected.rsplit("(", 1)
    selected_symbol = selected_symbol.strip(")")

    st.write(f"Fetching data for **{selected_symbol}**...")

    try:
        data, _ = ts.get_daily(symbol=selected_symbol, outputsize='compact')
        data = data.rename(columns={
            '1. open': 'Open', '2. high': 'High',
            '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'
        })
        data = data.sort_index()

        st.line_chart(data['Close'])
        st.line_chart(data['Volume'])
        st.write("Latest data:")
        st.write(data.tail())

    except Exception as e:
        st.error(f"Error fetching stock data: {e}")

# -----------------------
# TAB 3: PREDICTION
# -----------------------
with tab3:
    st.subheader("Price Prediction: ARIMA Model (Close, High, Low)")

    selected_name = st.selectbox("Choose stock for prediction", list(sp500_dict.keys()))
    symbol = sp500_dict[selected_name]

    try:
        st.write(f"Fetching closing price data for **{symbol}**...")
        full_data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        full_data = full_data.rename(columns={
            '1. open': 'Open', '2. high': 'High',
            '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'
        })
        full_data = full_data.sort_index()

        close_series = full_data['Close'].dropna()
        high_series = full_data['High'].dropna()
        low_series = full_data['Low'].dropna()

        st.line_chart(close_series.tail(100))

        # Forecasting function
        def forecast_series(series, label):
            #st.write(f"Training ARIMA model for **{label}** price...")
            model = ARIMA(series, order=(5, 1, 0))
            fitted_model = model.fit()
            return fitted_model.forecast(steps=5)

        # Forecast each component
        forecast_close = forecast_series(close_series, "Close")
        forecast_high = forecast_series(high_series, "High")
        forecast_low = forecast_series(low_series, "Low")

        # Prepare forecast DataFrame
        forecast_dates = pd.date_range(start=close_series.index[-1], periods=6, freq='B')[1:]
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast Close': forecast_close,
            'Forecast High': forecast_high,
            'Forecast Low': forecast_low
        })
        forecast_df.set_index("Date", inplace=True)

        st.write("5-Day Forecast for **{symbol}** (Close, High, Low)")
        st.write(forecast_df)

        st.write("Graphical for **{symbol}**")
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        close_series.tail(100).plot(ax=ax, label="Historical Close", color='blue')
        forecast_df['Forecast Close'].plot(ax=ax, label="Forecast Close", linestyle='--', color='red')
        forecast_df['Forecast High'].plot(ax=ax, label="Forecast High", linestyle='--', color='green')
        forecast_df['Forecast Low'].plot(ax=ax, label="Forecast Low", linestyle='--', color='orange')
        ax.set_title(f"{symbol} Price Forecast (Next 5 Days)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to fetch or model data: {e}")






