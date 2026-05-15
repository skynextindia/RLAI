import MetaTrader5 as mt5
from src.mt5_bridge.zmq_client import MT5ZMQClient
import time
import os

def run_test():
    print(">>> Initializing Execution Test (Using Port 5565/5566 to avoid conflict)...")
    # Use different ports so we don't conflict with the running engine
    client = MT5ZMQClient(pull_port=5565, push_port=5566)
    
    symbol = "BTCUSDm"
    lot_size = 0.01
    
    if not mt5.initialize():
        print("MT5 Initialization Failed")
        return

    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Failed to get price for {symbol}")
        return
        
    current_price = tick.ask
    sl_dist = 50.0 # Fixed distance for crypto test
    tp_dist = 100.0
    
    print(f"\n--- TEST 1: BUY ORDER ---")
    sl = current_price - sl_dist
    tp = current_price + tp_dist
    print(f"[TEST] Sending BUY {lot_size} @ {current_price} | SL: {sl} | TP: {tp}")
    
    ticket = client.send_order("BUY", symbol, lot_size, sl=sl, tp=tp)
    if ticket:
        print(f"SUCCESS: BUY Ticket {ticket} opened.")
        time.sleep(3)
        print(f"\n--- TEST 2: CLOSE ORDER ---")
        client.send_order("CLOSE_ALL", symbol, lot_size)
    else:
        print("FAILURE: BUY Order failed. Check terminal logs.")

    time.sleep(2)

    tick = mt5.symbol_info_tick(symbol)
    if tick:
        current_price = tick.bid
        print(f"\n--- TEST 3: SELL ORDER ---")
        sl = current_price + sl_dist
        tp = current_price - tp_dist
        print(f"[TEST] Sending SELL {lot_size} @ {current_price} | SL: {sl} | TP: {tp}")
        
        ticket = client.send_order("SELL", symbol, lot_size, sl=sl, tp=tp)
        if ticket:
            print(f"SUCCESS: SELL Ticket {ticket} opened.")
            time.sleep(3)
            print(f"\n--- TEST 4: CLOSE ORDER ---")
            client.send_order("CLOSE_ALL", symbol, lot_size)
        else:
            print("FAILURE: SELL Order failed. Check terminal logs.")

    mt5.shutdown()

if __name__ == "__main__":
    run_test()
