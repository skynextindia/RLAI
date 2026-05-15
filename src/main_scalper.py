
# Start File Pulse listener (High-reliability Mailbox pattern)
def file_pulse_listener():
    import time
    import os
    import json
    print(">>> [PULSE] Atomic File Listener Active (20Hz) <<<")
    while True:
        try:
            if os.path.exists("pulse.json"):
                with open("pulse.json", "r") as f:
                    data = json.load(f)
                # "Consume" the pulse immediately
                try:
                    os.remove("pulse.json")
                except:
                    pass
                data_callback(data)
        except Exception as e:
            pass
        time.sleep(0.05) # 20Hz Poll

def main():
    init_db()
    
    # Start heartbeat threads
    threading.Thread(target=file_pulse_listener, daemon=True).start()
    threading.Thread(target=telemetry_loop, daemon=True).start()
    
    # --- TRUE MT5 STATE RECOVERY ---
    import MetaTrader5 as mt5
    if mt5.initialize():
        symbol = "BTCUSDm"
        positions = mt5.positions_get(symbol=symbol)
        if positions and len(positions) > 0:
            pos = positions[0]
            # Initialize global state from existing position
            global last_lot_size, max_pnl_reached, last_start_time, last_reasoning
            last_lot_size = pos.volume
            max_pnl_reached = max(0.0, pos.profit)
            from datetime import timezone
            last_start_time = dt.fromtimestamp(pos.time, tz=timezone.utc).isoformat()
            last_reasoning = "ENGINE BOOT: Syncing with active position."
            print(f">>> [SYNC] Found Active Position: {pos.type} at {pos.price_open}")

    # Start ZMQ listener (Legacy support)
    client = MT5ZMQClient(Config.ZMQ_PULL_PORT, Config.ZMQ_PUSH_PORT)
    thread = threading.Thread(target=client.stream_market_data, args=(data_callback,))
    thread.daemon = True
    thread.start()
    
    # Start background historical sync
    threading.Thread(target=sync_historical_data, args=("BTCUSDm",), daemon=True).start()

    print("System Running... Press Ctrl+C to stop.")
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        import os
        os._exit(0)

if __name__ == "__main__":
    main()
