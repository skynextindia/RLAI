import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd

if not mt5.initialize():
    print("MT5 init failed")
    quit()

print("MT5 initialized")
symbol = "EURUSD"
symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    print(f"{symbol} not found")
else:
    print(f"Symbol {symbol} found")
    
end = datetime.now()
start = end - timedelta(days=1)
ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
if ticks is None:
    print("No ticks found")
else:
    print(f"Downloaded {len(ticks)} ticks")

mt5.shutdown()
