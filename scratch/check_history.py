import MetaTrader5 as mt5
from datetime import datetime, timedelta

if not mt5.initialize():
    print("Failed to initialize MT5")
    quit()

# Fetch history from yesterday
from_date = datetime.now() - timedelta(days=1)
to_date = datetime.now()

history = mt5.history_deals_get(from_date, to_date)
if history is None:
    print("No history found.")
else:
    print(f"Found {len(history)} deals from yesterday.")
    for deal in history:
        print(f"Ticket: {deal.ticket} | Symbol: {deal.symbol} | Profit: {deal.profit} | Time: {datetime.fromtimestamp(deal.time)}")

mt5.shutdown()
