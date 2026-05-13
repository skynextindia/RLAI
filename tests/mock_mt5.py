import zmq
import json
import time
import random

def run_mock():
    context = zmq.Context()
    push_socket = context.socket(zmq.PUSH)
    push_socket.connect("tcp://127.0.0.1:5555")

    print("Mock MT5 Publisher streaming to tcp://127.0.0.1:5555...")
    try:
        while True: # Infinite stream for testing
            data = {
                "symbol": "BTCUSDm",
                "ask": round(79000.50 + random.uniform(-10, 10), 2),
                "bid": round(79000.00 + random.uniform(-10, 10), 2),
                "volume": random.randint(1, 10)
            }
            push_socket.send_string(json.dumps(data))
            print(f"Sent tick: {data['bid']}") # Feedback to user
            time.sleep(0.1) # 10 ticks per second
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    run_mock()
