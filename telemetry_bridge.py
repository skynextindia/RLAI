
import zmq
import json
import time
import os

ZMQ_ADDR = "tcp://127.0.0.1:5555"
# Target file in the dashboard's public folder so it can be fetched easily
PULSE_FILE = "dashboard/public/telemetry.json"

def bridge():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(ZMQ_ADDR)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, 100)
    
    print(f"PULSE_BRIDGE_START -> {PULSE_FILE}")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(PULSE_FILE), exist_ok=True)

    while True:
        try:
            msg_str = sock.recv_string()
            msg = json.loads(msg_str)
            
            # Atomic write with persistent retry to break Windows locks
            temp_file = PULSE_FILE + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(msg, f)
            
            success = False
            for attempt in range(20):
                try:
                    if os.path.exists(PULSE_FILE):
                        os.remove(PULSE_FILE)
                    os.rename(temp_file, PULSE_FILE)
                    success = True
                    break
                except OSError:
                    time.sleep(0.005) # Faster retry
            
            if not success:
                print(f"PULSE_FAIL")
        except zmq.Again:
            pass
        except Exception as e:
            print(f"PULSE_ERR: {e}")
        
        time.sleep(0.005) # Faster loop

if __name__ == "__main__":
    bridge()
