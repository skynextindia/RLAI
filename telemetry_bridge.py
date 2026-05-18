import zmq
import json
import time
import threading
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

ZMQ_ADDR = "tcp://127.0.0.1:5555"
current_pulse = {"status": "BRIDGE_SYNCED", "timestep": 0, "equity": 10000}
equity_history_rolling = []
last_processed_step = 0

class TelemetryHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/telemetry':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Combine current pulse with rolling equity history
            response_payload = dict(current_pulse)
            response_payload['equity_history'] = equity_history_rolling
            self.wfile.write(json.dumps(response_payload).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return

def run_server():
    try:
        # Binding to 0.0.0.0 to ensure maximum reachability
        server = HTTPServer(('0.0.0.0', 8080), TelemetryHandler)
        print("HTTP_SERVER_LIVE: http://127.0.0.1:8080/telemetry", flush=True)
        server.serve_forever()
    except Exception as e:
        print(f"HTTP_ERR: {e}", flush=True)

def bridge():
    global current_pulse, equity_history_rolling, last_processed_step
    print("STARTING_BRIDGE_SERVICE...", flush=True)
    
    # 1. Start HTTP Server First
    threading.Thread(target=run_server, daemon=True).start()
    
    # 2. Setup ZMQ
    print(f"CONNECTING_TO_ZMQ: {ZMQ_ADDR}", flush=True)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(ZMQ_ADDR)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, 500)
    print("ZMQ_CONNECTED_WAITING_FOR_DATA...", flush=True)
    
    while True:
        try:
            msg_str = sock.recv_string()
            current_pulse = json.loads(msg_str)
            current_pulse['server_time'] = time.time()
            
            # Update equity history rolling buffer
            step = current_pulse.get('step', 0)
            equity = current_pulse.get('equity', 10000)
            pnl = current_pulse.get('pnl', 0)
            win_rate = current_pulse.get('win_rate', 0.0)
            
            # Reset history if a new episode starts (step goes backward)
            if step < last_processed_step:
                print("NEW_EPISODE_RESET: Clearing rolling equity history", flush=True)
                equity_history_rolling = []
                
            last_processed_step = step
            
            if step > 0:
                last_step = equity_history_rolling[-1]['step'] if equity_history_rolling else -1
                is_trade_step = (step == current_pulse.get('recent_trades', [{}])[-1].get('step', -2) if current_pulse.get('recent_trades') else False)
                
                # Record if:
                # 1. First step
                # 2. Or advanced by at least 150 steps
                # 3. Or a trade was completed at this exact step
                if last_step == -1 or (step - last_step >= 150) or is_trade_step:
                    if not equity_history_rolling or equity_history_rolling[-1]['step'] != step:
                        equity_history_rolling.append({
                            'step': step,
                            'equity': equity,
                            'pnl': pnl,
                            'win_rate': float(win_rate) * 100
                        })
                        if len(equity_history_rolling) > 2000:
                            equity_history_rolling = equity_history_rolling[-2000:]
        except zmq.Again:
            pass
        except Exception as e:
            # Re-init on error to prevent silence
            print(f"ZMQ_ERR: {e}", flush=True)
            time.sleep(1)
        time.sleep(0.01)

if __name__ == "__main__":
    bridge()
