import numpy as np
from technical_indicators import sma, rsi, adx14, is_macd_golden_cross, bollinger_position, fib_retracement, aroon
from utils import load_config, get_multiple_days_price, get_current_price, write_to_file

def main():
  config = load_config()
  ticker = config['TICKER']
  days = config['DAYS']
  print(f'Retrieve the equity data of {ticker} for the last {days} days')

  data_set = get_multiple_days_price(ticker, days)
  price_current = get_current_price(data_set)
  print(f'{ticker} latest price: {price_current}')

  sma_150 = sma(data_set, 150)
  rsi_50 = rsi(data_set, 50)
  adx_14 = adx14(data_set, 14)
  macd_now = is_macd_golden_cross(data_set).iloc[-1]
  boll_band = bollinger_position(data_set, window = 30)
  boll_now = boll_band.iloc[-1]
  fib_90 = fib_retracement(data_set, 90)
  aroon_up_25, aroon_down_25, aroon_osc_25 = aroon(data_set, 25)
  aroon_up_14, aroon_down_14, aroon_osc_14 = aroon(data_set, 14)

  print(boll_now)
  print(f'latest aroon 14: {aroon_osc_14.iloc[-1]}')
  print(f'latest aroon 25: {aroon_osc_25.iloc[-1]}')

  if config['SAVE_DATA_TO_FILE']:
    write_to_file(data_set, f'data_set_{ticker}')
    write_to_file(adx14, f'ADX_{ticker}')
    write_to_file(fib_90, f'Fibonacci_{ticker}')
    write_to_file(boll_band, f'bollinger_{ticker}')

if __name__ == '__main__':
  main()