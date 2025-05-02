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
tab1, tab2, tab3 = st.tabs(["Introduction", "S&P 500 Stock Info", "Price Prediction"])

# Introduction Tab (Tab 1)
with tab1:

     # Load an image
    image_path = "stock_market_banner_970x250.png"
    image = Image.open(image_path)
    
    # Display the banner
    st.image(image, use_container_width=True)
    
    st.write("""
  This is a stock price and prediction modelling app that will assist investors on buying or selling the stocks. The stocks are based on all companies listed on S&P 500 and the price are up to date linking from Yahoo Finance server.
  The app will display the following:
  - Historical stock price trend up to the latest - shown on S&P 500 stock info tab
  - Showcase stock price forecast (up to 5days) - shown on Price Prediction tab

The main purpose of this application is providing price guidance for investors on the price risks and market risks of the S&P 500 stocks.""")
  
 
# Add content to Tab 2
with tab2:
  st.subheader("S&P 500 Stock Info:")
  
  # Load S&P 500 tickers and company names
  @st.cache_data  # Cache the data to avoid reloading on every interaction
  def load_sp500_tickers_names():
      # Scrape S&P 500 tickers and company names from Wikipedia
      url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
      table = pd.read_html(url)[0]  # Load the first table on the page
      # Create a dictionary with Ticker as key and Name as value
      tickers_names = dict(zip(table["Symbol"], table["Security"]))
      return tickers_names
       
   # Function to get stock data
   def get_stock_data(ticker):
      stock = yf.Ticker(ticker)
      return stock.history(period="1y")  # adjust the period as needed
  
  # Load tickers and company names
  tickers_names = load_sp500_tickers_names()
  
  # Dropdown for ticker selection with company names
  selected_ticker = st.selectbox(
      "Select a stock ticker:", 
      options=[f"{ticker} - {name}" for ticker, name in tickers_names.items()]
  )
  
  # Extract the actual ticker from the selected option
  ticker_symbol = selected_ticker.split(" - ")[0]
  
  # Display stock data
  if ticker_symbol:
      st.subheader(f"Stock Data for {ticker_symbol}")
      stock_data = get_stock_data(ticker_symbol)
      st.write(stock_data)
      
      # Display charts
      st.subheader(f"S&P 500 Charts for {ticker_symbol} closing price:")
      st.line_chart(stock_data['Close'], width=0, height=0)

      st.subheader(f"S&P 500 Charts for {ticker_symbol} trading volume:")
      st.line_chart(stock_data['Volume'], width=0, height=0)

# Add content to Tab 3:
with tab3:
  st.subheader("Price Prediction: Select and Click-Predict")

  # Load an image
  image_path_tab3 = "bull_bear_01.jpg"
  image_tab3 = Image.open(image_path_tab3)
  st.image(image_tab3, use_container_width=True)

  # Load full S&P 500 stock list
  def load_sp500_stocks():
      url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
      table = pd.read_html(url)[0]
      return dict(zip(table['Security'], table['Symbol']))
  
  sp500_stocks = load_sp500_stocks()

  
  # Dropdown for stock selection
  stock_name = st.selectbox("Select a stock:", list(sp500_stocks.keys()))
  stock_symbol = sp500_stocks[stock_name]
  
  # Date input for historical data range
  start_date = st.date_input("Select start date for historical data", pd.to_datetime("2023-01-01"))
  end_date = st.date_input("Select end date", pd.to_datetime("today"))
  
  @st.cache_data
  def fetch_stock_data(symbol, start, end):
      return yf.download(symbol, start=start, end=end)
  
  # Button to fetch and predict
  if st.button("Predict"):
      st.subheader(f"Fetching Data for {stock_symbol}...")
      
      # Fetch stock data
      stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
      
      if stock_data.empty:
          st.error("No data found! Try selecting a different date range.")
      else:
          st.write("Last 5 rows of historical data:")
          st.write(stock_data.tail())
  
          # Use 'Close', 'High', and 'Low' prices for forecasting
          stock_prices = stock_data[['Close', 'High', 'Low']].dropna()
  
          def train_arima(series):
              model = ARIMA(series, order=(5, 1, 0))
              return model.fit()
  
          model_close = train_arima(stock_prices['Close'])
          model_high = train_arima(stock_prices['High'])
          model_low = train_arima(stock_prices['Low'])
  
          # Forecast next 5 days
          forecast_close = model_close.forecast(steps=5)
          forecast_high = model_high.forecast(steps=5)
          forecast_low = model_low.forecast(steps=5)
  
          # Display predictions
          future_dates = pd.date_range(stock_prices.index[-1], periods=6)[1:]
          forecast_df = pd.DataFrame({
              'Date': future_dates, 
              'Predicted Close Price': forecast_close,
              'Predicted High Price': forecast_high,
              'Predicted Low Price': forecast_low
          })
          forecast_df.set_index("Date", inplace=True)
  
          st.subheader(f"Predicted {stock_symbol} Prices for Next 5 Days")
          st.write(forecast_df)
  
          # Plot results
          st.subheader(f"Predicted {stock_symbol} Chart:")
          fig, ax = plt.subplots(figsize=(10, 5))
          stock_prices['Close'][-50:].plot(ax=ax, label="Historical Close Prices", color="blue")
          forecast_df["Predicted Close Price"].plot(ax=ax, label="Forecast Close", linestyle="dashed", color="red")
          forecast_df["Predicted High Price"].plot(ax=ax, label="Forecast High", linestyle="dashed", color="green")
          forecast_df["Predicted Low Price"].plot(ax=ax, label="Forecast Low", linestyle="dashed", color="orange")
          ax.set_title(f"Stock Price Prediction for {stock_symbol}")
          ax.set_xlabel("Date")
          ax.set_ylabel("Price")
          ax.legend()
          st.pyplot(fig)
