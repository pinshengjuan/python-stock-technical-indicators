import os
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_config():
  return {
    'TICKER': os.getenv('TICKER', 'GOOGL'),
    'DAYS': int(os.getenv('DAYS', 365)),
    'SAVE_DATA_TO_FILE':int(os.getenv('SAVE_DATA_TO_FILE', 0)),
  }

def get_multiple_days_price(ticker, days):
  end_date = datetime.now()
  start_date = end_date - timedelta(days=days)
  stock = yf.Ticker(ticker)
  data_set = stock.history(start=start_date, end=end_date)

  return data_set

def get_current_price(data_set) -> float:
  return data_set['Close'].iloc[-1]

def wilder_smoothing(series, period=14):
  smoothed = np.zeros(len(series))
  smoothed[period-1] = series[:period].mean()
  for i in range(period, len(series)):
    smoothed[i] = (smoothed[i-1] * (period-1) + series.iloc[i]) / period
  smoothed[:period-1] = np.nan

  return pd.Series(smoothed, index=series.index)

def write_to_file(data_set, filename):
  dir = str(os.path.abspath(os.path.dirname(__file__))) + '/data/'
  if os.path.isdir(dir) == False:
    os.mkdir(dir)
  filename = dir + filename + '.txt'
  pd.set_option('display.max_rows', None)
  open(filename, 'w', encoding='utf8').writelines(str(data_set))