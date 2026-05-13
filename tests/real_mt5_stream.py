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

    # Update to your active symbol in MT5 Market Watch
    symbol = "BTCUSDm" 
    
    # Ensure symbol is visible
    mt5.symbol_select(symbol, True)
    
    print(f"Streaming REAL tick data for {symbol} to ZMQ...")
    
    try:
        while True:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                data = {
                    "symbol": symbol,
                    "ask": tick.ask,
                    "bid": tick.bid,
                    "volume": tick.volume,
                }
                push_socket.send_string(json.dumps(data))
                print(f"Pushed: {data}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    stream_real_data()
