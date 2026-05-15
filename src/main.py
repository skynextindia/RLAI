import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import zmq
import json
import time
import threading
import os
import sqlite3
from datetime import datetime as dt
from config import Config
from data.mtf_aggregator import MTFAggregator
from env.trading_env import MT5TradingEnv
from models.ppo_agent import PPOAgent
from training.ppo_trainer import PPOTrainer
from models.transformer import TimeSeriesTransformer
from models.xgboost_model import XGBoostModel
from execution.risk_engine import RiskEngine
from mt5_bridge.zmq_client import MT5ZMQClient
import requests

# --- TACTICAL GLOBAL STATE ---
TELEMETRY_FILE = "telemetry.json"
COMMAND_FILE = "commands.json"
LOG_FILE = "axon_ai.log"
DB_FILE = "axon_trades.db"
SYSTEM_PAUSED = False

last_price = 0.0
last_confidence = 0.0
last_sl = 0.0
last_tp = 0.0
last_start_time = "--:--"
last_reasoning = "INITIALIZING SENSORS..."
last_probs = [0.33, 0.33, 0.33]
last_smoothed_probs = None
last_regime = "STABLE"
last_tf_matrix = {}
last_lot_size = 0.0
last_pnl = 0.0
max_pnl_reached = 0.0
last_tps = 0.0
tick_counter = 0
last_processed_timestamp = None

def ai_log(message):
    timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Create the final 10-column institutional schema
    c.execute('''CREATE TABLE IF NOT EXISTS trades
                 (ticket INTEGER PRIMARY KEY,
                  symbol TEXT,
                  action TEXT,
                  profit REAL,
                  reward REAL,
                  timestamp TEXT,
                  entry REAL DEFAULT 0.0,
                  peak REAL DEFAULT 0.0,
                  max_dd REAL DEFAULT 0.0,
                  reasoning TEXT DEFAULT 'N/A')''')
    conn.commit()
    conn.close()

def sync_account_history():
    """Pulls last 24h of MT5 history into local DB for UI visibility"""
    import MetaTrader5 as mt5
    from datetime import datetime, timedelta
    if not mt5.initialize(): return
    
    # Deep History Recall: Fetch last 7 days of trades
    from_date = datetime.now() - timedelta(days=7)
    history = mt5.history_deals_get(from_date, datetime.now())
    
    if history:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        for deal in history:
            if "btcusd" in deal.symbol.lower() and deal.entry == mt5.DEAL_ENTRY_OUT:
                try:
                    c.execute("""INSERT OR IGNORE INTO trades 
                                 (ticket, symbol, action, profit, reward, timestamp, entry, peak, max_dd, reasoning) 
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                              (deal.ticket, deal.symbol, 
                               "LONG" if deal.type == mt5.DEAL_TYPE_SELL else "SHORT", 
                               float(deal.profit), 0.0,
                               datetime.fromtimestamp(deal.time).isoformat(),
                               0.0, 0.0, 0.0, "Historical Sync"))
                except Exception as e:
                    print(f">>> [SYNC ERROR] Ticket {deal.ticket}: {e}")
        conn.commit()
        conn.close()
    print(f">>> [SYNC] Account history integration complete.")

def sync_historical_data(symbol):
    symbol = "BTCUSDm" # Force Uppercase for Data
    globals()['IS_SYNCING'] = True
    ai_log(f"Initiating Historical Backfill for {symbol}...")
    if not mt5.initialize():
        globals()['IS_SYNCING'] = False
        return
    
    timeframes = {
        '1min': mt5.TIMEFRAME_M1,
        '5min': mt5.TIMEFRAME_M5,
        '15min': mt5.TIMEFRAME_M15,
        '1h': mt5.TIMEFRAME_H1,
        '4h': mt5.TIMEFRAME_H4
    }
    
    for tf_name, mt5_tf in timeframes.items():
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, 500)
        if rates is not None:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df = aggregator.engineer.calculate_atr(df)
            aggregator.seed_historical_data(tf_name, df)
            ai_log(f"| Sync Complete: {tf_name} ({len(df)} candles)")
            
    globals()['IS_SYNCING'] = False
    ai_log(">>> Historical Synchronization 100% Complete. AI is now LUNID (Live & Synchronized).")

# Global State for Neural Memory
last_active_state = None
last_active_action = None
last_active_sequence = None

def process_commands():
    global SYSTEM_PAUSED, client
    if os.path.exists(COMMAND_FILE):
        try:
            with open(COMMAND_FILE, "r") as f:
                cmd = json.load(f)
            action = cmd.get("action")
            if action == "CLOSE_ALL":
                positions = mt5.positions_get()
                if positions:
                    for p in positions:
                        client.close_position(p.symbol)
                    ai_log("[WEB IO] ALL POSITIONS CLOSED.")
            elif action == "PAUSE": SYSTEM_PAUSED = True
            elif action == "RESUME": SYSTEM_PAUSED = False
            elif action == "KILL": os._exit(0)
            os.remove(COMMAND_FILE)
        except Exception: pass

def data_callback(data):
    # Tick Activity Indicator
    print(".", end="", flush=True)
    try:
        global last_processed_timestamp, client, active_rl_trade, SYSTEM_PAUSED, last_price, last_confidence, last_regime, last_tf_matrix, last_probs, last_reasoning, last_sl, last_tp, last_start_time, last_lot_size, last_pnl, max_pnl_reached, tick_counter, last_tps
        
        # --- INSTANT TELEMETRY UPDATE ---
        current_price = data['bid']
        last_price = current_price
        globals()['last_spread'] = data['ask'] - data['bid']
        
        process_commands()
        aggregator.add_tick(data)
        features = aggregator.aggregate()
        
        # 1. Position Sync (Adaptive Case)
        all_positions = mt5.positions_get()
        positions = [p for p in all_positions if "btcusd" in p.symbol.lower()] if all_positions else []
        actual_pnl = 0.0
        
        if positions:
            pos = positions[0]
            current_ticket = pos.ticket
            
            # Reset forensics ONLY if this is a brand new ticket
            if current_ticket != globals().get('last_active_ticket'):
                globals()['last_active_ticket'] = current_ticket
                max_pnl_reached = 0.0
                globals()['worst_pnl_reached'] = 0.0
                print(f">>> [FORENSICS] Tracking New Ticket: {current_ticket}")
            
            env.position = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
            env.entry_price = pos.price_open
            actual_pnl = pos.profit
            last_sl, last_tp, last_lot_size, last_pnl = pos.sl, pos.tp, pos.volume, actual_pnl
            
            # Strictly cumulative Peak/DD
            if actual_pnl > max_pnl_reached: 
                max_pnl_reached = actual_pnl
            
            worst = globals().get('worst_pnl_reached', 0.0)
            if actual_pnl < worst: 
                worst = actual_pnl
            globals()['worst_pnl_reached'] = worst
            
            # Continuous Reward Tracking for Neural Memory
            globals()['current_trade_reward'] = actual_pnl
        else:
            # Check if a trade was JUST closed
            old_ticket = globals().get('last_active_ticket')
            if old_ticket:
                try:
                    # Match the 10-column schema: ticket, symbol, action, profit, reward, timestamp, entry, peak, max_dd, reasoning
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    
                    final_pnl = float(last_pnl)
                    # Normalized reward for RL (-1 to 1 scale)
                    normalized_reward = np.clip(final_pnl / 10.0, -1.0, 1.0)
                    
                    c.execute("""INSERT OR REPLACE INTO trades 
                                 (ticket, symbol, action, profit, reward, timestamp, entry, peak, max_dd, reasoning) 
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                              (old_ticket, "BTCUSDm", "LONG" if env.position == 1 else "SHORT",
                               final_pnl, float(normalized_reward), dt.now().isoformat(),
                               float(env.entry_price), float(max_pnl_reached), float(globals().get('worst_pnl_reached', 0.0)),
                               str(last_reasoning)))
                    conn.commit()
                    conn.close()
                    
                    # --- NEURAL FEEDBACK LOOP ---
                    # Push to PPO Buffer for learning
                    if globals().get('last_active_state') is not None:
                        trainer.store_transition(
                            globals()['last_active_state'],
                            globals()['last_active_action'],
                            normalized_reward,
                            env._extract_state(features, tps=tps, spread=spread, tick_delta=tick_delta),
                            True # Done
                        )
                        print(f">>> [BRAIN] Experience Saved. Reward: {normalized_reward:+.2f} | Buffer: {len(trainer.buffer)}/32")
                        
                        # Trigger training if buffer is full
                        if len(trainer.buffer) >= 32:
                            print(">>> [BRAIN] Optimizing Neural Weights...")
                            trainer.train(batch_size=32)
                            torch.save(agent.state_dict(), "ppo_agent_h4_pretrained.pth")
                            print(">>> [BRAIN] Neural Weights Updated & Saved.")
                    
                    ai_log(f"[DB] FORENSICS SAVED FOR TICKET {old_ticket}")
                except Exception as e:
                    print(f"!!! DB ERROR (10-col): {e}")
                
                globals()['last_active_ticket'] = None
                globals()['last_active_state'] = None
                globals()['last_close_time'] = dt.now().isoformat() # Trigger Cooldown

            env.position = 0
            env.entry_price = 0.0
            last_pnl = 0.0
            globals()['worst_pnl_reached'] = 0.0


        if features is None: return # Wait for more data for neural analysis

        # Heartbeat TPS
        if 'tick_timestamps' not in globals(): globals()['tick_timestamps'] = []
        current_time = dt.now().timestamp()
        globals()['tick_timestamps'].append(current_time)
        if len(globals()['tick_timestamps']) > 20: globals()['tick_timestamps'].pop(0)
        tps = 0.0
        if len(globals()['tick_timestamps']) > 1:
            dur = globals()['tick_timestamps'][-1] - globals()['tick_timestamps'][0]
            if dur > 0: tps = len(globals()['tick_timestamps']) / dur
        
        # Neural State
        spread = globals().get('last_spread', 0.0)
        last_spread = globals().get('last_spread_ref', spread)
        spread_delta = spread - last_spread
        globals()['last_spread_ref'] = spread

        # Calculate tick delta for AI learning
        last_p = globals().get('last_tick_price_ref', current_price)
        tick_delta = current_price - last_p
        globals()['last_tick_price_ref'] = current_price

        state = env._extract_state(features, tps=tps, spread=spread, tick_delta=tick_delta)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        sequence = env._extract_sequence(features, timeframe='4h', seq_len=50)
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

        # --- NEURAL BRAIN TRIGGER ---
        _compute_neural_policy(state, seq_tensor, state_tensor, spread, tick_delta)
        
        # Intra-Candle Execution Engine
        h4_df = features.get('4h')
        process_h4_strategy(h4_df, current_price, features, tps, spread, tick_delta, state_tensor, data)

        # --- TERMINAL VITALITY ---
        tick_counter += 1
        last_tps = tps
        spinner = ['|', '/', '-', '\\'][tick_counter % 4]
        
        # Calculate Neural Bias for display
        probs = globals().get('ensemble_probs', [1.0, 0.0, 0.0])
        conf = globals().get('last_confidence', 0.0)
        bias = "BUY" if probs[1] > probs[2] else "SELL"
        if probs[0] > 0.8: bias = "HOLD"
        
        if tick_counter % 2 == 0:
            print(f"\r[{spinner} AXON APEX] BTCUSD @ {data['bid']:.2f} | {bias}: {conf:.1%} | TPS: {tps:.1f} | PnL: ${actual_pnl:.2f}", end="", flush=True)

        # TF Matrix
        tf_matrix = {}
        for tf_name in ['1min', '5min', '15min', '1h', '4h']:
            df_tf = features.get(tf_name)
            if df_tf is not None and len(df_tf) > 14:
                tf_state = env._extract_state(features={'tf': df_tf}) # Standardized call
                with torch.no_grad():
                    tf_probs = xgb_filter.predict_confidence(tf_state.reshape(1, -1))
                probs = tf_probs[0]
                tf_action = np.argmax(probs)
                tf_certainty = float(probs[tf_action])
                entropy = -np.sum(probs * np.log(probs + 1e-9))
                tf_matrix[tf_name] = {
                    "atr": float(df_tf['ATR'].iloc[-1]),
                    "trend": ["NEUTRAL", "LONG", "SHORT"][tf_action],
                    "certainty": tf_certainty,
                    "regime": "STABLE" if entropy < 0.8 else "TURBULENT",
                    "alignment": "HIGH" if tf_certainty > 0.6 else "LOW"
                }
        last_price, last_tf_matrix = current_price, tf_matrix

    except Exception as e: print(f"Callback Error: {e}")

def _compute_neural_policy(state, seq_tensor, state_tensor, spread, tick_delta):
    try:
        with torch.no_grad():
            # Dimension Validation
            if state_tensor.shape[1] != 16:
                raise ValueError(f"State Dimension Mismatch: Expected 16, got {state_tensor.shape[1]}")
            
            action_probs, _ = agent(state_tensor)
            trans_probs = torch.softmax(transformer(seq_tensor), dim=-1)
            xgb_probs = xgb_filter.predict_confidence(state.reshape(1, -1))
            
            w_ppo, w_trans, w_xgb = 0.40, 0.35, 0.25
            raw_probs = (action_probs.numpy()[0] * w_ppo + trans_probs.numpy()[0] * w_trans + xgb_probs[0] * w_xgb)
            
            # Strategic Bias: Only consider Buy/Sell for confidence display
            conf = float(np.max(raw_probs[1:])) 
            
            globals()['ensemble_probs'] = raw_probs
            globals()['last_confidence'] = conf
            globals()['last_probs'] = [float(p) for p in raw_probs]
            
    except Exception as e:
        print(f"\n[BRAIN ERROR] {e}")
        globals()['ensemble_probs'] = np.array([1.0, 0.0, 0.0])
        globals()['last_confidence'] = 0.0

def telemetry_loop():
    while True:
        try:
            performance = {"win_rate": 0.0, "profit_factor": 0.0, "drawdown": 0.0}
            conn = sqlite3.connect(DB_FILE)
            df_history = pd.read_sql_query("SELECT profit FROM trades ORDER BY timestamp ASC", conn)
            conn.close()
            if not df_history.empty:
                wins = len(df_history[df_history['profit'] > 0])
                performance["win_rate"] = (wins / len(df_history)) * 100
                gross_p = float(df_history[df_history['profit'] > 0]['profit'].sum())
                gross_l = abs(float(df_history[df_history['profit'] < 0]['profit'].sum()))
                performance["profit_factor"] = (gross_p / gross_l) if gross_l > 0 else gross_p
                
                # Drawdown
                profits = [0.0] + df_history['profit'].tolist() + [float(globals().get('last_pnl', 0.0))]
                p_curve = pd.Series(profits).cumsum()
                performance["drawdown"] = float((p_curve.cummax() - p_curve).max())

            probs_raw = globals().get('last_probs', [0.3333, 0.3333, 0.3334])
            if len(probs_raw) < 3 or any(np.isnan(p) for p in probs_raw): 
                probs_raw = [0.3333, 0.3333, 0.3334]
            
            # Calculate Entropy for UI
            entropy = -np.sum([p * np.log(p + 1e-9) for p in probs_raw])
            
            # RL Performance Metrics
            # Calculate actual Time Decay penalty (e.g., -0.001 per minute held)
            time_decay = 0.0
            if env.position != 0 and globals().get('last_start_time'):
                try:
                    # Handle both ISO formats (with and without microseconds)
                    start_dt = dt.fromisoformat(globals()['last_start_time'].split('+')[0]) 
                    time_held_seconds = (dt.now() - start_dt).total_seconds()
                    time_decay = -0.001 * (time_held_seconds / 60.0)
                except Exception:
                    pass
            
            live_pnl = float(globals().get('last_pnl', 0.0))
            current_equity = float(env.balance + live_pnl)

            telemetry = {
                "balance": current_equity,
                "pnl": live_pnl,
                "confidence": float(globals().get('last_confidence', 0.0)),
                "action": int(env.position),
                "symbol": "BTCUSDm",
                "price": float(last_price),
                "spread": float(globals().get('last_spread', 0.0)),
                "tps": float(last_tps),
                "performance": performance,
                "diagnostics": {
                    "total_ticks": int(tick_counter),
                    "buffer": len(trainer.buffer), # Actual RL Memory Buffer
                    "decay": time_decay, # Use actual calculated decay
                    "entropy": float(entropy),
                    "last_brain_sync": globals().get('last_brain_sync_time', 'NEVER'),
                    "is_syncing": globals().get('IS_SYNCING', False),
                    "regime": last_regime
                },
                "trade_forensics": {
                    "entry": float(env.entry_price),
                    "sl": float(last_sl),
                    "tp": float(last_tp),
                    "peak_pnl": float(max_pnl_reached),
                    "max_dd": float(globals().get('worst_pnl_reached', 0.0)),
                    "start_time": str(last_start_time),
                    "lot_size": float(last_lot_size),
                    "reasoning": str(last_reasoning)
                },
                "probs": [float(p) for p in probs_raw[:3]],
                "tf_matrix": last_tf_matrix
            }
            
            # RL Buffer tracking for UI
            prev_buffer_len = globals().get('last_buffer_len', 0)
            current_buffer_len = len(trainer.buffer)
            if current_buffer_len == 0 and prev_buffer_len > 0:
                globals()['last_brain_sync_time'] = dt.now().strftime("%H:%M:%S")
            globals()['last_buffer_len'] = current_buffer_len
            
            def sanitize_json(obj):
                if isinstance(obj, dict):
                    return {k: sanitize_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [sanitize_json(i) for i in obj]
                elif isinstance(obj, float):
                    if np.isnan(obj) or np.isinf(obj): return 0.0
                return obj

            telemetry = sanitize_json(telemetry)
            
            # Use POST for primary telemetry (Faster than Disk I/O)
            try: 
                requests.post("http://127.0.0.1:8000/api/telemetry", json=telemetry, timeout=0.5)
            except Exception as e:
                pass # Silently handle server offline cases
                
        except Exception as e:
            print(f">>> [TELEMETRY ERROR] {e}")
            
        time.sleep(0.2) # 5Hz Neural Refresh

def process_h4_strategy(h4_df, current_price, features, tps, spread, tick_delta, state_tensor, data):
    global last_processed_timestamp, last_reasoning, last_start_time, max_pnl_reached, tick_counter
    
    # CHAOS MODE: No Cooldowns. No Gatekeeping.
    
    if env.position != 0:
        last_reasoning = f"MONITORING | Neural Conviction: {globals().get('last_confidence', 0.0):.1%}"
    else:
        # 1. Extract Neural Decision
        state = env._extract_state(features, tps=tps, spread=spread, tick_delta=tick_delta)
        _compute_neural_policy(state, torch.tensor(env._extract_sequence(features, '4h', 50)).unsqueeze(0).float(), state_tensor, spread, tick_delta)
        
        probs = globals()['ensemble_probs']
        action = np.argmax(probs)
        confidence = float(probs[action])
        globals()['last_confidence'] = confidence
        
        # 2. INJECT RANDOMNESS (20% Exploration)
        if random.random() < 0.20:
            action = random.choice([1, 2]) # Force Buy or Sell
            last_reasoning = "EXPLORATION | Random Neural Probe"
        else:
            last_reasoning = f"EXPLOITATION | Signal: {confidence:.1%}"

        # 3. NON-STOP EXECUTION (Threshold = 0.0)
        if action != 0:
            target_symbol = "BTCUSDm" 
            lot = risk_engine.calculate_lot_size(env.balance, confidence, current_price)
            
            # Fixed 1:2 Chaos Exits (Approx 100 pips SL / 200 pips TP for BTC)
            sl_val = current_price - 150.0 if action == 1 else current_price + 150.0
            tp_val = current_price + 300.0 if action == 1 else current_price - 300.0

            ticket = client.send_order("BUY" if action == 1 else "SELL", target_symbol, lot, sl=sl_val, tp=tp_val)
            if ticket:
                last_start_time = dt.now().isoformat()
                max_pnl_reached = 0.0
                last_reasoning = f"NEURAL ENTRY | Confidence: {confidence:.1%}"
                
                globals()['last_active_state'] = state
                globals()['last_active_action'] = action
                print(f"\n>>> [EXECUTION] High-Conviction Re-entry: {last_reasoning}")

def main():
    global client, env, agent, risk_engine, xgb_filter, transformer, trainer, last_lot_size, last_sl, last_tp, last_start_time, max_pnl_reached
    init_db()
    sync_account_history()
    if not mt5.initialize(): return
    
    # Real Balance Sync
    acc = mt5.account_info()
    if acc:
        env.balance = acc.balance
        print(f">>> [REAL SYNC] Account Balance Verified: ${env.balance:,.2f}")

    print(">>> [SYNC] Adopting active Bitcoin trades...")
    adopted_pos = None
    for _ in range(10):
        all_pos = mt5.positions_get()
        if all_pos:
            matches = [p for p in all_pos if "btcusd" in p.symbol.lower()]
            if matches:
                adopted_pos = matches[0]
                break
        time.sleep(0.5)

    if adopted_pos:
        env.position = 1 if adopted_pos.type == mt5.ORDER_TYPE_BUY else -1
        env.entry_price = adopted_pos.price_open
        last_lot_size, last_sl, last_tp = adopted_pos.volume, adopted_pos.sl, adopted_pos.tp
        last_start_time = dt.fromtimestamp(adopted_pos.time).isoformat()
        globals()['last_active_ticket'] = adopted_pos.ticket
        print(f">>> [SYNC] Adopted {adopted_pos.symbol} (Ticket: {adopted_pos.ticket})")
        
        # --- RECONSTRUCT HISTORICAL MAE/MFE ---
        try:
            rates = mt5.copy_rates_from(adopted_pos.symbol, mt5.TIMEFRAME_M1, dt.now(), 5000)
            if rates is not None and len(rates) > 0:
                valid_rates = [r for r in rates if r['time'] >= adopted_pos.time]
                if valid_rates:
                    highest_price = float(max(r['high'] for r in valid_rates))
                    lowest_price = float(min(r['low'] for r in valid_rates))
                    
                    tick = mt5.symbol_info_tick(adopted_pos.symbol)
                    current_price = tick.bid if env.position == 1 else tick.ask
                    
                    # Contract-aware scaling for forensics
                    multiplier = adopted_pos.volume 
                    
                    if env.position == 1:
                        mfe_diff = highest_price - env.entry_price
                        mae_diff = lowest_price - env.entry_price
                    else:
                        mfe_diff = env.entry_price - lowest_price
                        mae_diff = env.entry_price - highest_price
                        
                    max_pnl_reached = max(adopted_pos.profit, mfe_diff * multiplier)
                    globals()['worst_pnl_reached'] = min(adopted_pos.profit, mae_diff * multiplier)
                    print(f">>> [FORENSICS] Reconstructed Historical MFE (Peak): ${max_pnl_reached:.2f} | MAE (Max DD): ${globals()['worst_pnl_reached']:.2f}")
        except Exception as e:
            print(f">>> [SYNC WARNING] Could not reconstruct historical MAE/MFE: {e}")
    else:
        print(">>> [SYNC] No active trades found.")

    client = MT5ZMQClient(Config.ZMQ_PULL_PORT, Config.ZMQ_PUSH_PORT)
    
    # 1. Start Telemetry FIRST (Ensures Dashboard isn't 0.00)
    threading.Thread(target=telemetry_loop, daemon=True).start()
    
    # 2. Start Live Stream SECOND
    threading.Thread(target=client.stream_market_data, args=(data_callback,), daemon=True).start()
    
    # 3. Start Historical Backfill THIRD (In background)
    threading.Thread(target=sync_historical_data, args=("BTCUSDm",), daemon=True).start()
    
    print("\n>>> [STATUS] AI ENGINE ONLINE.\n")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        ai_log(">>> [SHUTDOWN] Initiating Clean Neural Exit...")
        if client:
            client.pull_socket.close()
            client.push_socket.close()
            client.context.term()
        print(">>> [STATUS] Neural Port Released. Axon Safe-Sleep.")
        return

if __name__ == "__main__":
    # GLOBAL REGISTRY FOR THREAD-SAFE TELEMETRY
    registry_lock = threading.Lock()
    
    aggregator = MTFAggregator()
    env = MT5TradingEnv()
    
    # ENSURE UNIFIED 16-DIM ARCHITECTURE
    state_dim = 16 
    agent = PPOAgent(state_dim=state_dim, action_dim=3)
    
    # PERSISTENT LEARNING LOAD
    weights_path = "ppo_agent_h4_pretrained.pth"
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            # Intelligent shape verification
            model_layers = agent.state_dict()
            if state_dict['fc1.weight'].shape[1] == state_dim:
                agent.load_state_dict(state_dict)
                print(f">>> [ARCH] Successfully loaded 16-dim neural weights. Learning is PERSISTENT.")
            else:
                print(f">>> [ARCH] Weight dimension mismatch ({state_dict['fc1.weight'].shape[1]} vs {state_dim}). Initializing Fresh 16-dim matrix.")
        except Exception as e:
            print(f">>> [ARCH] Weight load error: {e}. Resetting weights.")
            
    trainer = PPOTrainer(agent)
    transformer = TimeSeriesTransformer(input_dim=16)
    xgb_filter = XGBoostModel()
    risk_engine = RiskEngine(max_dd_pct=Config.MAX_DRAWDOWN_PCT, risk_per_trade_pct=Config.RISK_PER_TRADE_PCT)
    main()
