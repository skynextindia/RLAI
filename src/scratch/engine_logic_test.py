import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from src.mt5_bridge.zmq_client import MT5ZMQClient
from src.env.trading_env import MT5TradingEnv
import time

def run_engine_logic_test():
    print(">>> Initializing Engine Logic Simulation...")
    client = MT5ZMQClient(pull_port=5575, push_port=5576)
    env = MT5TradingEnv(initial_balance=10000)
    
    symbol = "BTCUSDm"
    lot_size = 0.01
    
    if not mt5.initialize():
        print("MT5 Initialization Failed")
        return

    # Simulate ATR from actual market data
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 50)
    if rates is None or len(rates) < 14:
        print("Failed to get H4 data for ATR calculation")
        return
        
    df = pd.DataFrame(rates)
    # Simple ATR calculation
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    
    tick = mt5.symbol_info_tick(symbol)
    current_price = tick.ask
    
    # --- START ENGINE LOGIC EMULATION ---
    # Logic from main_scalper.py:678-690
    sl_dist = (atr * 1.5) if atr > 0 else (current_price * 0.005)
    tp_dist = (atr * 3.0) if atr > 0 else (current_price * 0.01)
    
    print(f"\n[ENGINE EMULATION] Testing BUY Entry Logic...")
    
    # 1. Diagnostic Pulse (The [RISK] line)
    print(f"[RISK] sl_dist={sl_dist:.5f} tp_dist={tp_dist:.5f} atr={atr:.5f}")

    # 2. Hard Guard check
    if sl_dist <= 0 or sl_dist < current_price * 0.001:
        print(f"[RISK BLOCK] Order REJECTED — invalid sl_dist={sl_dist:.5f}. No order sent.")
        return

    # 3. Execution
    sl = current_price - sl_dist
    tp = current_price + tp_dist
    
    print(f"[ENGINE] Attempting BUY {lot_size} @ {current_price} | sl_dist={sl_dist:.2f} | tp_dist={tp_dist:.2f}")
    
    ticket = client.send_order("BUY", symbol, lot_size, sl=sl, tp=tp)
    
    if ticket:
        print(f"\n>>> [TEST SUCCESS] Engine Logic produced a valid execution with SL/TP.")
        print(f">>> Waiting 5 seconds, then closing...")
        time.sleep(5)
        client.send_order("CLOSE_ALL", symbol, lot_size)
    else:
        print(f"\n>>> [TEST FAILURE] Order rejected. Check MT5 retcode above.")

    mt5.shutdown()

if __name__ == "__main__":
    run_engine_logic_test()
