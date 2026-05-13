from config import Config
from mt5_bridge.zmq_client import MT5ZMQClient
from data.mtf_aggregator import MTFAggregator
from env.trading_env import MT5TradingEnv
from models.ppo_agent import PPOAgent
from models.transformer import TimeSeriesTransformer
from models.xgboost_model import XGBoostModel
from execution.risk_engine import RiskEngine
from training.ppo_trainer import PPOTrainer
import threading
import torch
import numpy as np
import json
import os
import pandas as pd
import sqlite3
from datetime import datetime as dt

# Initialize SQLite Database
DB_FILE = "axon_trades.db"
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades
                 (ticket INTEGER PRIMARY KEY,
                  symbol TEXT,
                  action TEXT,
                  profit REAL,
                  reward REAL,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

TELEMETRY_FILE = "telemetry.json"
COMMAND_FILE = "commands.json"
LOG_FILE = "axon_ai.log"
SYSTEM_PAUSED = False

def ai_log(message):
    timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")

# Clear logs on startup
with open(LOG_FILE, "w") as f:
    f.write(f"--- Axon Engine Session Started at {dt.now()} ---\n")

def process_commands():
    global SYSTEM_PAUSED, client
    if os.path.exists(COMMAND_FILE):
        try:
            with open(COMMAND_FILE, "r") as f:
                cmd = json.load(f)
            
            action = cmd.get("action")
            print(f"\n[WEB IO] Received Command: {action}")
            
            if action == "CLOSE_ALL":
                import MetaTrader5 as mt5
                if mt5.initialize():
                    positions = mt5.positions_get()
                    if positions:
                        for p in positions:
                            client.close_position(p.symbol)
                        ai_log("[WEB IO] ALL POSITIONS CLOSED SUCCESSFULLY.")
            
            elif action == "PAUSE":
                SYSTEM_PAUSED = True
                ai_log("[WEB IO] AI ENGINE PAUSED.")
            
            elif action == "RESUME":
                SYSTEM_PAUSED = False
                ai_log("[WEB IO] AI ENGINE RESUMED.")
            
            elif action == "KILL":
                ai_log("[WEB IO] EMERGENCY SHUTDOWN INITIATED.")
                os._exit(0)
            
            # Wipe command after processing
            os.remove(COMMAND_FILE)
        except Exception as e:
            print(f"[WEB IO] Error: {e}")

aggregator = MTFAggregator()
env = MT5TradingEnv()
agent = PPOAgent(state_dim=12, action_dim=3)

# Load Pre-trained H4 Model if exists
if os.path.exists("ppo_agent_h4_pretrained.pth"):
    agent.load_state_dict(torch.load("ppo_agent_h4_pretrained.pth", weights_only=True))
    print(">>> Loaded Pre-trained H4 Model Weights <<<")

trainer = PPOTrainer(agent)
transformer = TimeSeriesTransformer(input_dim=5)
xgb_filter = XGBoostModel()
risk_engine = RiskEngine(max_dd_pct=Config.MAX_DRAWDOWN_PCT, risk_per_trade_pct=Config.RISK_PER_TRADE_PCT)

last_processed_timestamp = None
state_history = []
tick_counter = 0

# Global Client Reference
client = None

# --- INSTITUTIONAL HISTORICAL SYNC ---
def sync_historical_data(symbol):
    ai_log(f"Initiating Historical Backfill for {symbol}...")
    import MetaTrader5 as mt5
    if not mt5.initialize():
        ai_log("Sync Error: MT5 Initialize Failed.")
        return
    
    global IS_SYNCING
    IS_SYNCING = True
    
    # We backfill 4H, 1H, 15M, 5M, 1M
    tfs = {
        '4h': mt5.TIMEFRAME_H4,
        '1h': mt5.TIMEFRAME_H1,
        '15m': mt5.TIMEFRAME_M15,
        '5m': mt5.TIMEFRAME_M5,
        '1m': mt5.TIMEFRAME_M1
    }
    
    for tf_name, tf_mt5 in tfs.items():
        rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, 500)
        if rates is not None:
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('timestamp', inplace=True)
            df.rename(columns={'real_volume': 'volume'}, inplace=True)
            
            # Feed into engineer for indicators
            df = aggregator.engineer.calculate_atr(df)
            df = aggregator.engineer.detect_choch_bos(df)
            
            # Seed the aggregator
            aggregator.seed_historical_data(tf_name, df)
            ai_log(f"| Sync Complete: {tf_name} ({len(df)} candles)")
            
    IS_SYNCING = False
    ai_log(">>> Historical Synchronization 100% Complete. AI is now LUNID (Live & Synchronized).")

# RL Trade Memory Persistence
MEMORY_FILE = "active_trade.json"
active_rl_trade = {"ticket": None, "state": None, "action": None, "start_time": None}
IS_SYNCING = False

def save_memory():
    try:
        # Deep cast everything to standard Python types for JSON
        ticket = active_rl_trade.get("ticket")
        action = active_rl_trade.get("action")
        state = active_rl_trade.get("state")
        
        mem_to_save = {
            "ticket": int(ticket) if ticket is not None else None,
            "action": int(action) if action is not None else None,
            "state": state.tolist() if state is not None else None,
            "start_time": active_rl_trade.get("start_time")
        }
        with open(MEMORY_FILE, "w") as f:
            json.dump(mem_to_save, f)
    except Exception as e:
        print(f"[MEMORY] Save Error: {e}")

def load_memory():
    global active_rl_trade
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
            
            # Reconstruct tensors
            if data.get("state") is not None:
                data["state"] = torch.tensor(data["state"], dtype=torch.float32)
            
            active_rl_trade = data
            if active_rl_trade.get("ticket"):
                print(f"[MEMORY] Restored Tracking for Trade: {active_rl_trade['ticket']}")
        except Exception as e:
            print(f"[MEMORY] Load Error (Resetting): {e}")
            active_rl_trade = {"ticket": None, "state": None, "action": None, "start_time": None}

load_memory()

def data_callback(data):
    try:
        global last_processed_timestamp, client, active_rl_trade, SYSTEM_PAUSED
        
        # Handle Web IO Commands
        process_commands()
        
        aggregator.add_tick(data)
        features = aggregator.aggregate()
        
        if features is None:
            return

        current_price = data['bid']
        
        # --- TRUE MT5 STATE REFLECTION ---
        import MetaTrader5 as mt5
        actual_pnl = 0.0
        sl_target = 0.0
        tp_target = 0.0
        
        if mt5.initialize():
            # 1. Sync True Balance
            account_info = mt5.account_info()
            if account_info:
                env.balance = account_info.balance
                
            # 2. Sync True Positions
            positions = mt5.positions_get(symbol=data['symbol'])
            if positions and len(positions) > 0:
                pos = positions[0]
                env.position = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
                env.entry_price = pos.price_open
                actual_pnl = pos.profit
                sl_target = pos.sl
                tp_target = pos.tp
            else:
                env.position = 0
                env.entry_price = 0.0

            # 3. RL Feedback Loop Listener (Did the trade close?)
            if active_rl_trade["ticket"] is not None and env.position == 0:
                ticket = active_rl_trade["ticket"]
                deals = mt5.history_deals_get(position=ticket)
                if deals and len(deals) > 0:
                    # The exit deal is usually the last one
                    exit_deal = deals[-1]
                    profit = exit_deal.profit
                    
                    # Assign RL Reward
                    reward = 2.0 if profit > 0 else -1.0
                    
                    # --- TIME-DECAY PENALTY (PRO) ---
                    if active_rl_trade.get("start_time"):
                        start_dt = dt.fromisoformat(active_rl_trade["start_time"])
                        duration_hours = (dt.now() - start_dt).total_seconds() / 3600
                        # Penalize 0.1 for every 24 hours open
                        decay = (duration_hours / 24) * 0.1
                        reward -= decay
                        print(f"[RL ENGINE] Trade Duration: {duration_hours:.1f}h | Time-Decay Penalty: -{decay:.2f}")

                    print(f"\n[RL ENGINE] Trade {ticket} Closed! Profit: ${profit:.2f} -> Final Reward: {reward:.2f}")
                    
                    # Extract current state as 'next_state'
                    obs = env._extract_state(features)
                    next_state_tensor = torch.tensor(obs, dtype=torch.float32)
                    
                    # Store Transition & Update Weights
                    trainer.store_transition(active_rl_trade["state"], active_rl_trade["action"], reward, next_state_tensor, True)
                    
                    # Save to SQLite Database
                    action_str = "BUY" if active_rl_trade["action"] == 1 else "SELL"
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute("INSERT OR REPLACE INTO trades (ticket, symbol, action, profit, reward, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                              (ticket, data['symbol'], action_str, profit, reward, dt.now().isoformat()))
                    conn.commit()
                    conn.close()
                    print(f"[RL ENGINE] Trade {ticket} logged to SQLite Database.")

                    if len(trainer.buffer) >= 32:
                        print(f"[RL ENGINE] Buffer Full. Updating Neural Weights...")
                        trainer.update()
                        torch.save(agent.state_dict(), "ppo_agent_h4_pretrained.pth")
                        global last_weight_update
                        last_weight_update = dt.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[RL ENGINE] Weights Successfully Saved to Disk at {last_weight_update}")
                
                # Wipe memory after learning
                active_rl_trade = {"ticket": None, "state": None, "action": None}
                save_memory()
                
        # Tick Feedback (Faster heartbeat: every 5 ticks)
        global tick_counter
        tick_counter += 1
        if tick_counter % 5 == 0:
            pos_str = "FLAT" if env.position == 0 else ("LONG" if env.position == 1 else "SHORT")
            print(f"\r[HEARTBEAT] {data['symbol']} @ {current_price:.2f} | {pos_str} | PnL: ${actual_pnl:.2f} | Bal: ${env.balance:.2f}", end="", flush=True)

        # --- LIVE NEURAL INFERENCE (Every Tick) ---
        state = env._extract_state(features)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        sequence = env._extract_sequence(features, timeframe='4h', seq_len=50)
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, _ = agent(state_tensor)
            trans_probs = torch.softmax(transformer(seq_tensor), dim=-1)
            xgb_probs = xgb_filter.predict_confidence(state.reshape(1, -1))
        
        global ensemble_probs, confidence
        ensemble_probs = (action_probs.numpy()[0] + trans_probs.numpy()[0] + xgb_probs[0]) / 3.0
        if np.isnan(ensemble_probs).any(): ensemble_probs = np.array([0.33, 0.33, 0.34])
        confidence = float(ensemble_probs[np.argmax(ensemble_probs)])
        probs_list = ensemble_probs.tolist()

        # 2. Extract metrics from H4 for Institutional Context
        h4_df = features.get('4h')
        
        live_atr = float(h4_df['ATR'].iloc[-1]) if h4_df is not None and not h4_df.empty else 0.0
        live_bos = bool(h4_df['BOS_Bullish'].iloc[-1]) if h4_df is not None and not h4_df.empty else False
        live_choch = bool(h4_df['CHOCH_Bearish'].iloc[-1]) if h4_df is not None and not h4_df.empty else False

        # Sanitize NaNs for JSON stability
        safe_atr = live_atr if not np.isnan(live_atr) else 0.0

        # --- INSTITUTIONAL DIAGNOSTICS ---
        regime = "STABLE"
        if h4_df is not None and not h4_df.empty and 'ATR' in h4_df.columns:
            avg_atr = h4_df['ATR'].rolling(50).mean().iloc[-1]
            if safe_atr > (avg_atr * 1.5): regime = "VOLATILE"
            elif safe_atr < (avg_atr * 0.5): regime = "COMPRESSED"
        
        buffer_len = len(trainer.buffer) if hasattr(trainer, 'buffer') else 0
        last_update = globals().get('last_weight_update', 'NEVER')
        
        est_decay = 0.0
        start_time_str = active_rl_trade.get("start_time")
        if start_time_str:
            start_dt = dt.fromisoformat(start_time_str)
            duration_hours = (dt.now() - start_dt).total_seconds() / 3600
            est_decay = (duration_hours / 24) * 0.1

        # 3. Multi-Timeframe Sensor Matrix (For "Whole Data" Monitoring)
        tf_matrix = {}
        for tf_name in ['1min', '5min', '15min', '1h', '4h']:
            df_tf = features.get(tf_name)
            # PROTECTIVE GUARD: Only process if we have enough bars for indicators
            if df_tf is not None and len(df_tf) > 14:
                tf_atr = float(df_tf['ATR'].iloc[-1]) if 'ATR' in df_tf.columns else 0.0
                if np.isnan(tf_atr): tf_atr = 0.0 # Sanitize for JSON compatibility
                
                # Neural Sentiment (Pure AI Inference per Timeframe)
                tf_state = env._extract_state({'tf': df_tf}) 
                tf_probs = xgb_filter.predict_confidence(tf_state.reshape(1, -1))
                
                # Ensure tf_probs is treated as 2D
                probs = tf_probs[0] if len(tf_probs.shape) > 1 else tf_probs
                
                tf_action = np.argmax(probs)
                tf_certainty = float(probs[tf_action])
                
                bias = ["NEUTRAL", "LONG", "SHORT"][tf_action]

                tf_matrix[tf_name] = {
                    "atr": tf_atr,
                    "bos": bool(df_tf['BOS_Bullish'].iloc[-1]) if 'BOS_Bullish' in df_tf.columns else False,
                    "choch": bool(df_tf['CHOCH_Bearish'].iloc[-1]) if 'CHOCH_Bearish' in df_tf.columns else False,
                    "trend": bias,
                    "certainty": tf_certainty
                }

        # --- TACTICAL PERFORMANCE ENGINE ---
        performance = {"win_rate": 0.0, "profit_factor": 0.0, "drawdown": 0.0}
        try:
            conn = sqlite3.connect(DB_FILE)
            df_history = pd.read_sql_query("SELECT profit FROM trades", conn)
            conn.close()
            
            if not df_history.empty:
                wins = len(df_history[df_history['profit'] > 0])
                total = len(df_history)
                performance["win_rate"] = (wins / total) * 100
                
                gross_profit = df_history[df_history['profit'] > 0]['profit'].sum()
                gross_loss = abs(df_history[df_history['profit'] < 0]['profit'].sum())
                performance["profit_factor"] = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit
                
                # Simple Drawdown (Cumulative)
                cum_profit = df_history['profit'].cumsum()
                max_profit = cum_profit.expanding().max()
                drawdown = (max_profit - cum_profit).max()
                performance["drawdown"] = float(drawdown)
        except Exception as e:
            print(f"Stats Error: {e}")

        # Machine Reasoning Justification
        reasoning = "SCANNING MARKET STRUCTURE..."
        if env.position != 0:
            bias_str = "BULLISH" if env.position == 1 else "BEARISH"
            reasoning = f"INSTITUTIONAL {bias_str} FLOW DETECTED | CONFIDENCE: {confidence*100:.1f}% | H4 STRUCTURE: {'BOS' if live_bos else 'STABLE'}"

        telemetry = {
            "balance": float(env.balance),
            "pnl": float(actual_pnl),
            "confidence": float(confidence),
            "action": int(env.position),
            "symbol": data['symbol'],
            "price": float(current_price),
            "performance": performance,
            "trade_forensics": {
                "entry": float(env.entry_price),
                "sl": float(sl_target),
                "tp": float(tp_target),
                "start_time": start_time_str,
                "lot_size": globals().get('last_lot_size', 0.0),
                "reasoning": reasoning
            },
            "probs": probs_list,
            "diagnostics": {
                "regime": regime,
                "buffer": buffer_len,
                "decay": round(est_decay, 3),
                "entropy": round(float(np.std(probs_list)), 4),
                "last_brain_sync": last_update,
                "is_syncing": globals().get('IS_SYNCING', False)
            },
            "tf_matrix": tf_matrix,
            "indicators": {
                "ATR": safe_atr,
                "BOS": live_bos,
                "CHOCH": live_choch
            }
        }
        with open(TELEMETRY_FILE, "w") as f:
            json.dump(telemetry, f)

        # 3. ONLY run AI Model for new entries on 4H Candle closure AND if FLAT
        if h4_df is not None and not h4_df.empty:
            current_ts = h4_df.index[-1]
            if current_ts != last_processed_timestamp:
                last_processed_timestamp = current_ts
                
                if SYSTEM_PAUSED:
                    ai_log(f"H4 Candle Triggered, but ENGINE IS PAUSED. Skipping decision.")
                    return

                ai_log(f"--- [H4 STRATEGY] New 4H Candle Triggered: {current_ts} ---")

                if mt5.initialize():
                    positions = mt5.positions_get(symbol=data['symbol'])
                    if not positions or len(positions) == 0:
                        if env.position != 0:
                            env.position = 0

                if env.position != 0:
                    print(f"[{data['symbol']}] Holding existing {'BUY' if env.position == 1 else 'SELL'} trade.")
                else:
                    action = np.argmax(ensemble_probs)
                    
                    # --- NEURAL EXPLAINER (PRO) ---
                    action_name = ['HOLD','BUY','SELL'][action]
                    ai_log(f"AI Decision: {action_name}")
                    
                    if h4_df is not None and not h4_df.empty:
                        last_h4 = h4_df.iloc[-1]
                        prev_h4 = h4_df.iloc[-2] if len(h4_df) > 1 else last_h4
                        price_change = ((last_h4['close'] - prev_h4['close']) / prev_h4['close']) * 100
                        vol_change = ((last_h4['volume'] - prev_h4['volume']) / prev_h4['volume']) * 100 if prev_h4['volume'] > 0 else 0
                        ai_log(f"| Sensory Input 1: 4H Momentum is {price_change:+.2f}%")
                        ai_log(f"| Sensory Input 2: Volume Delta is {vol_change:+.2f}%")

                    lot_size = risk_engine.calculate_lot_size(env.balance, confidence, current_price)
                    next_obs, reward, done, _, info = env.step(action, current_price, features, lot_size=lot_size, contract_size=globals().get('contract_size', 1.0))
                    
                    atr = float(h4_df['ATR'].iloc[-1]) if h4_df is not None and not h4_df.empty else 0.0
                    sl_dist = (atr * 1.5) if atr > 0 else (current_price * 0.005)
                    tp_dist = (atr * 3.0) if atr > 0 else (current_price * 0.01)

                    if action == 1: # BUY
                        ticket = client.send_order("BUY", data['symbol'], lot_size, sl=current_price - sl_dist, tp=current_price + tp_dist)
                    elif action == 2: # SELL
                        ticket = client.send_order("SELL", data['symbol'], lot_size, sl=current_price + sl_dist, tp=current_price - tp_dist)
                    
                    if action != 0 and 'ticket' in locals() and ticket is not None:
                        active_rl_trade["ticket"] = ticket
                        active_rl_trade["state"] = state_tensor.squeeze(0)
                        active_rl_trade["action"] = action
                        active_rl_trade["start_time"] = dt.now().isoformat()
                        save_memory()

    except Exception as e:
        print(f"Engine Critical Error in Callback: {e}")


def main():
    global client
    print("Starting Institutional AI Trading Infrastructure...")
    
    # --- SYNC STATE WITH MT5 ---
    import MetaTrader5 as mt5
    global contract_size
    contract_size = 1.0 # Default fallback
    if mt5.initialize():
        symbol_info = mt5.symbol_info("BTCUSDm")
        if symbol_info:
            contract_size = symbol_info.trade_contract_size
            print(f">>> [SYNC] MT5 Contract Size for BTCUSDm: {contract_size}")
            
        positions = mt5.positions_get(symbol="BTCUSDm")
        if positions and len(positions) > 0:
            pos = positions[0] # Take the first active position
            env.position = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
            env.entry_price = pos.price_open
            
            # Capture Lot Size for UI
            global last_lot_size
            last_lot_size = pos.volume
            
            pos_type_str = "BUY" if env.position == 1 else "SELL"
            print(f">>> [SYNC] Found Active MT5 Position: {pos_type_str} at {env.entry_price} (Lots: {last_lot_size}). AI will MONITOR, not open a new trade.")
        else:
            print(">>> [SYNC] No active positions found. AI is FLAT and ready to enter.")
    # ---------------------------

    client = MT5ZMQClient(Config.ZMQ_PULL_PORT, Config.ZMQ_PUSH_PORT)
    
    # --- ASYNC AUTO-SYNC (PRO) ---
    # We run this in a background thread so the engine starts instantly
    threading.Thread(target=sync_historical_data, args=("BTCUSDm",), daemon=True).start()
    
    # Start ZMQ listener
    thread = threading.Thread(target=client.stream_market_data, args=(data_callback,))
    thread.daemon = True
    thread.start()
    
    print("System Running... Press Ctrl+C to stop.")
    try:
        while True:
            if not thread.is_alive():
                print(">>> [CRITICAL] ZMQ Thread Died. Attempting restart...")
                thread = threading.Thread(target=client.stream_market_data, args=(data_callback,))
                thread.daemon = True
                thread.start()
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Institutional AI Engine...")
        import os
        os._exit(0) # Force exit to prevent ZMQ thread hanging terminal

if __name__ == "__main__":
    main()

