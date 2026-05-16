import zmq
import json
import time
import threading
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

ZMQ_ADDR = "tcp://127.0.0.1:5555"
current_pulse = {"status": "BRIDGE_SYNCED", "timestep": 0, "equity": 10000}

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
            self.wfile.write(json.dumps(current_pulse).encode())
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
    global current_pulse
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
        except zmq.Again:
            pass
        except Exception as e:
            # Re-init on error to prevent silence
            print(f"ZMQ_ERR: {e}", flush=True)
            time.sleep(1)
        time.sleep(0.01)

if __name__ == "__main__":
    bridge()
