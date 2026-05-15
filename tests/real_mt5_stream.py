import MetaTrader5 as mt5
import zmq
import json
import time

def stream_real_data():
    if not mt5.initialize():
        print("MT5 initialization failed. Ensure MetaTrader 5 is open.")
        return

    context = zmq.Context()
    push_socket = context.socket(zmq.PUSH)
    push_socket.connect("tcp://127.0.0.1:5555")

    # Exness feed requires Uppercase for ticks
    symbol = "BTCUSDm" 
    
    # Ensure symbol is visible
    mt5.symbol_select(symbol, True)
    
    print(f"Streaming REAL tick data for {symbol} to ZMQ...")
    
    last_time_msc = 0
    
    try:
        while True:
            tick = mt5.symbol_info_tick(symbol)
            if tick and tick.time_msc != last_time_msc:
                data = {
                    "symbol": symbol,
                    "ask": tick.ask,
                    "bid": tick.bid,
                    "volume": tick.volume,
                }
                push_socket.send_string(json.dumps(data))
                last_time_msc = tick.time_msc
                
            time.sleep(0.01) # High-frequency polling
    except KeyboardInterrupt:
        pass
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    stream_real_data()
