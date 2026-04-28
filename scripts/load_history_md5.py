import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

mt5.initialize()

symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_M5

utc_from = datetime(2025, 4, 1)
utc_to = datetime(2026, 4, 1)

rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

df.to_csv("xauusd_1y_m5.csv", index=False)

mt5.shutdown()