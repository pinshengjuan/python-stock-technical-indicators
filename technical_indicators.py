import pandas as pd
import numpy as np
from utils import wilder_smoothing

def sma(data: pd.DataFrame, periods: int) -> pd.Series:
  close = data['Close']
  return np.round(close.rolling(window=periods).mean().iloc[-1], 2)

def rsi(data: pd.DataFrame, periods: int = 14) -> pd.Series:
  if len(data) <= periods:
    raise ValueError(f"Data length ({len(data)}) must be > periods ({periods})")
  
  close = data['Close']
  delta = close.diff()
  gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
  rs = gain / loss
  rsi = 100 - (100 / (1 + rs))
  
  rsi = rsi.where(loss != 0, 100.0)  # RSI = 100 when loss is 0
  return rsi

def adx14(data: pd.DataFrame, periods: int = 14) -> pd.Series:
  high = data['High']
  low = data['Low']
  close = data['Close']

  tr = pd.DataFrame(index=high.index)
  tr['h_l'] = np.round((high - low), 2)
  tr['h_pc'] = abs(np.round((high - close.shift(1)), 2))
  tr['l_pc'] = abs(np.round((low - close.shift(1)), 2))
  tr['tr'] = tr[['h_l', 'h_pc', 'l_pc']].max(axis=1)
  
  dm_plus = high - high.shift(1)
  dm_minus = low.shift(1) - low
  dm_plus = dm_plus.where((dm_plus > 0) & (dm_plus > dm_minus), 0)
  dm_minus = dm_minus.where((dm_minus > 0) & (dm_minus > dm_plus), 0)

  smoothed_dm_plus = wilder_smoothing(dm_plus, periods)
  smoothed_dm_minus = wilder_smoothing(dm_minus, periods)
  smoothed_tr = wilder_smoothing(tr['tr'], periods)

  di_plus = (smoothed_dm_plus / smoothed_tr) * 100
  di_minus = (smoothed_dm_minus / smoothed_tr) * 100

  dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100

  adx = wilder_smoothing(dx.fillna(0), periods)

  return adx

def macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, 
         zero_threshold: float = 0.1, level_thresholds: tuple = (0.5, 0.2)) -> pd.DataFrame:
  close = data['Close']
  ema_fast = close.ewm(span=fast, adjust=False).mean()
  ema_slow = close.ewm(span=slow, adjust=False).mean()
  macd = ema_fast - ema_slow
  signal_line = macd.ewm(span=signal, adjust=False).mean()
  
  # Golden cross: MACD crosses above signal line
  is_golden_cross = (macd > signal_line) & (macd.shift() <= signal_line.shift())
  
  # Check if MACD is near zero
  is_near_zero = abs(macd) < zero_threshold
  
  # Classify MACD levels
  strong_threshold, mild_threshold = level_thresholds
  def classify_macd_level(macd_value):
    if macd_value > strong_threshold:
      return "Strong Bullish"
    elif macd_value > mild_threshold:
      return "Bullish"
    elif macd_value < -strong_threshold:
      return "Strong Bearish"
    elif macd_value < -mild_threshold:
      return "Bearish"
    else:
      return "Neutral"
  
  macd_level = macd.apply(classify_macd_level)
  
  # Combine results into a DataFrame
  result = pd.DataFrame({
    'is_golden_cross': is_golden_cross,
    'is_near_zero': is_near_zero,
    'macd_level': macd_level,
    'macd_value': macd
  })
  
  return result

def bollinger_position(data: pd.DataFrame, window: int = 20, k: float = 2, return_all_zones: bool = False) -> pd.Series:
  if window <= 0 or k <= 0:
    raise ValueError("Window and k must be positive")
  
  # Calculate Bollinger Bands
  close = data['Close']
  sma = close.rolling(window=window).mean()
  std_n = close.rolling(window=window).std()
  upper_band = sma + (std_n * k)
  lower_band = sma - (std_n * k)
  
  # Define percentage-based boundaries
  upper_0_3 = upper_band * 0.97  # 3% below upper
  upper_3_5 = upper_band * 0.95  # 5% below upper
  upper_5_10 = upper_band * 0.90  # 10% below upper
  lower_0_3 = lower_band * 1.03  # 3% above lower
  lower_3_5 = lower_band * 1.05  # 5% above lower
  lower_5_10 = lower_band * 1.10  # 10% above lower
  basis_plus_3 = sma * 1.03  # 3% above SMA
  basis_minus_3 = sma * 0.97  # 3% below SMA
  
  # Initialize position Series
  if return_all_zones:
    position = pd.Series([[] for _ in close], index=close.index, dtype=object)
  else:
    position = pd.Series(index=close.index, dtype=str)
  
  # Helper function to assign zones
  def get_zones(price, u, u_0_3, u_3_5, u_5_10, s, b_plus, b_minus, l_5_10, l_3_5, l_0_3, l):
    zones = []
    if price > u:
      zones.append('Above Upper')
    elif price <= u and price >= l:
      if price >= s:
        zones.append('Between Upper and SMA')
        if price <= u and price >= u_0_3:
          zones.append('Below Upper 0-3%')
        if price < u_0_3 and price >= u_3_5:
          zones.append('Below Upper 3-5%')
        if price < u_3_5 and price >= u_5_10:
          zones.append('Below Upper 5-10%')
      else:
        zones.append('Between SMA and Lower')
        if price <= l_0_3 and price > l:
          zones.append('Above Lower 0-3%')
        if price <= l_3_5 and price > l_0_3:
          zones.append('Above Lower 3-5%')
        if price <= l_5_10 and price > l_3_5:
          zones.append('Above Lower 5-10%')
      if price <= b_plus and price >= b_minus:
        zones.append('Basis ±3%')
    else:
      zones.append('Below Lower')
    return zones
  
  # Assign positions
  for i in close.index:
    if pd.isna(close[i]) or pd.isna(sma[i]) or pd.isna(upper_band[i]) or pd.isna(lower_band[i]):
      continue
    zones = get_zones(
      close[i],
      upper_band[i], upper_0_3[i], upper_3_5[i], upper_5_10[i],
      sma[i], basis_plus_3[i], basis_minus_3[i],
      lower_5_10[i], lower_3_5[i], lower_0_3[i], lower_band[i]
    )
    if return_all_zones:
      position[i] = zones
    else:
      # Priority order for single zone: extreme positions > percentage zones > general zones
      if 'Above Upper' in zones:
        position[i] = 'Above Upper'
      elif 'Below Lower' in zones:
        position[i] = 'Below Lower'
      elif 'Below Upper 0-3%' in zones:
        position[i] = 'Below Upper 0-3%'
      elif 'Below Upper 3-5%' in zones:
        position[i] = 'Below Upper 3-5%'
      elif 'Below Upper 5-10%' in zones:
        position[i] = 'Below Upper 5-10%'
      elif 'Above Lower 0-3%' in zones:
        position[i] = 'Above Lower 0-3%'
      elif 'Above Lower 3-5%' in zones:
        position[i] = 'Above Lower 3-5%'
      elif 'Above Lower 5-10%' in zones:
        position[i] = 'Above Lower 5-10%'
      elif 'Basis ±3%' in zones:
        position[i] = 'Basis ±3%'
      elif 'Between Upper and SMA' in zones:
        position[i] = 'Between Upper and SMA'
      else:
        position[i] = 'Between SMA and Lower'

  return position

def fib_retracement(data: pd.DataFrame, periods: int = 90) -> pd.Series:
  if len(data) <= periods:
    raise ValueError(f"Data length ({len(data)}) must be > periods ({periods})")
  close = data['Close']
  max_price = close.rolling(window=periods).max()
  min_price = close.rolling(window=periods).min()
  diff = max_price - min_price
  
  levels = {
      '100.0%': max_price,
      '61.8%': max_price - (diff * 0.382),
      '50.0%': max_price - (diff * 0.5),
      '38.2%': max_price - (diff * 0.618),
      '0.0%': min_price
  }
  level_values = list(levels.values())
  level_keys = list(levels.keys())

  position = pd.Series(index=close.index, dtype=str)
  for i in range(len(close)):
    price = close.iloc[i]
    if pd.isna(max_price.iloc[i]) or pd.isna(min_price.iloc[i]):
      continue
    if price > max_price.iloc[i]:
      position.iloc[i] = 'Above or equals 100.0%'
      continue
    elif price == max_price.iloc[i]:
      position.iloc[i] = 'Equals 100.0%'
      continue
    elif price < min_price.iloc[i]:
      position.iloc[i] = 'Below or equals 0.0%'
      continue
    elif price == min_price.iloc[i]:
      position.iloc[i] = 'Equals 0.0%'
      continue
    for j in range(len(levels) - 1):
      level_high = level_values[j].iloc[i]
      level_second_high = level_values[j + 1].iloc[i]
      if level_second_high < price < level_high:
        position.iloc[i] = f'Between {level_keys[j+1]} and {level_keys[j]}'

  return position

def aroon(data: pd.DataFrame, periods: int = 25) -> tuple:
  high = data['High']
  low = data['Low']
  aroon_up = pd.Series(index=high.index, dtype=float)
  aroon_down = pd.Series(index=low.index, dtype=float)
  aroon_osc = pd.Series(index=high.index, dtype=float)
  
  for i in range(periods, len(high)):
    high_window = high[i-periods:i+1]
    low_window = low[i-periods:i+1]
    high_idx = high_window.idxmax()
    low_idx = low_window.idxmin()
    periods_since_high = i - high.index.get_loc(high_idx)
    periods_since_low = i - low.index.get_loc(low_idx)
    aroon_up.iloc[i] = ((periods - periods_since_high) / periods) * 100
    aroon_down.iloc[i] = ((periods - periods_since_low) / periods) * 100
    aroon_osc.iloc[i] = aroon_up.iloc[i] - aroon_down.iloc[i]
  
  return aroon_up, aroon_down, aroon_osc