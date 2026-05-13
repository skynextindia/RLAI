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
import time
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

# --- TACTICAL GLOBAL STATE (For Background Heartbeat) ---
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
agent = PPOAgent(state_dim=13, action_dim=3)

# Load Pre-trained H4 Model if exists
if os.path.exists("ppo_agent_h4_pretrained.pth"):
    try:
        agent.load_state_dict(torch.load("ppo_agent_h4_pretrained.pth", weights_only=True))
        print(">>> [NEURAL SYNC] Loaded Pre-trained H4 Model Weights (13-Dim) <<<")
    except Exception as e:
        print(f">>> [NEURAL EVOLUTION] Dimension Mismatch Detected. Initializing Fresh 13-Dim Brain. (Old brain saved as legacy) <<<")

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
    globals()['IS_SYNCING'] = True
    ai_log(f"Initiating Historical Backfill for {symbol}...")
    import MetaTrader5 as mt5
    if not mt5.initialize():
        ai_log("Sync Error: MT5 Initialize Failed.")
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
            df.rename(columns={'real_volume': 'volume'}, inplace=True)
            
            # Feed into engineer for indicators
            df = aggregator.engineer.calculate_atr(df)
            df = aggregator.engineer.detect_choch_bos(df)
            
            # Seed the aggregator
            aggregator.seed_historical_data(tf_name, df)
            ai_log(f"| Sync Complete: {tf_name} ({len(df)} candles)")
            
    globals()['IS_SYNCING'] = False
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
            "start_time": active_rl_trade.get("start_time"),
            "max_pnl_reached": float(globals().get('max_pnl_reached', 0.0))
        }
        with open(MEMORY_FILE, "w") as f:
            json.dump(mem_to_save, f)
    except Exception as e:
        print(f"[MEMORY] Save Error: {e}")

def load_memory():
    global active_rl_trade, max_pnl_reached
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
            
            # Restore peak profit tracker
            max_pnl_reached = float(data.pop('max_pnl_reached', 0.0))
            
            # Reconstruct tensors
            if data.get("state") is not None:
                data["state"] = torch.tensor(data["state"], dtype=torch.float32)
            
            active_rl_trade = data
            if active_rl_trade.get("ticket"):
                print(f"[MEMORY] Restored Tracking for Trade: {active_rl_trade['ticket']} (Peak PnL: ${max_pnl_reached:.2f})")
        except Exception as e:
            print(f"[MEMORY] Load Error (Resetting): {e}")
            active_rl_trade = {"ticket": None, "state": None, "action": None, "start_time": None}

load_memory()

def data_callback(data):
    try:
        global last_processed_timestamp, client, active_rl_trade, SYSTEM_PAUSED, last_price, last_confidence, last_regime, last_tf_matrix, last_probs, last_reasoning, last_sl, last_tp, last_start_time, last_lot_size, last_pnl, max_pnl_reached, tick_counter, last_tps
        
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
                
                # Latch for UI
                last_sl = pos.sl
                last_tp = pos.tp
                last_lot_size = pos.volume
                last_pnl = actual_pnl
                
                # MFE: Track Peak Profit reached before closure
                if actual_pnl > max_pnl_reached:
                    max_pnl_reached = actual_pnl
                
                # Extract Start Time from MT5 Metadata
                from datetime import timezone
                open_time = dt.fromtimestamp(pos.time, tz=timezone.utc).isoformat()
                last_start_time = open_time
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
                globals()['max_pnl_reached'] = 0.0
                
        # --- NEURAL PULSE SENSOR (TPS) ---
        global tick_timestamps
        if 'tick_timestamps' not in globals(): tick_timestamps = []
        
        current_time = dt.now().timestamp()
        tick_timestamps.append(current_time)
        
        # Keep only the last 20 ticks for velocity calculation
        if len(tick_timestamps) > 20: tick_timestamps.pop(0)
        
        # Calculate Ticks Per Second (TPS)
        tps = 0.0
        if len(tick_timestamps) > 1:
            duration = tick_timestamps[-1] - tick_timestamps[0]
            if duration > 0:
                tps = len(tick_timestamps) / duration

        # Tick Feedback (Faster heartbeat: every 5 ticks)
        global tick_counter, last_tps
        tick_counter += 1
        last_tps = tps  # Latch rolling TPS for dashboard heartbeat
        if tick_counter % 5 == 0:
            pos_str = "FLAT" if env.position == 0 else ("LONG" if env.position == 1 else "SHORT")
            print(f"\r[HEARTBEAT] {data['symbol']} @ {current_price:.2f} | TPS: {tps:.1f} | PnL: ${actual_pnl:.2f} | Bal: ${env.balance:.2f}", end="", flush=True)

        # --- NEURAL STATE EXTRACTION (Every Tick for strategy use) ---
        state = env._extract_state(features)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        sequence = env._extract_sequence(features, timeframe='4h', seq_len=50)
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

        # --- NEURAL POLICY (Computed ONCE on startup, then only on H4 candle) ---
        # This prevents the "dancing bars" — conviction only changes when the AI
        # actually receives new strategic data (4H candle close or weight update)
        if globals().get('_neural_policy_initialized') is None:
            _compute_neural_policy(state, seq_tensor, state_tensor)
            globals()['_neural_policy_initialized'] = True


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
        
        # Latch these for UI
        globals()['buffer_len'] = len(trainer.buffer) if hasattr(trainer, 'buffer') else 0
        globals()['last_update'] = globals().get('last_weight_update', 'NEVER')
        globals()['last_regime'] = regime
        
        est_decay = 0.0
        start_time_str = active_rl_trade.get("start_time")
        if start_time_str:
            start_dt = dt.fromisoformat(start_time_str)
            duration_hours = (dt.now() - start_dt).total_seconds() / 3600
            est_decay = (duration_hours / 24) * 0.1
        globals()['last_decay'] = est_decay

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
        
        # Latch data for background heartbeat
        last_price = current_price
        last_regime = regime
        last_tf_matrix = tf_matrix
        
        # Execute Strategic Logic (H4 Structure + Neural Entry)
        # Using the globally latched ensemble_probs
        if 'ensemble_probs' in globals():
            process_h4_strategy(h4_df, current_price, features, tps, globals()['ensemble_probs'], state_tensor, data)

    except Exception as e:
        print(f"Engine Critical Error in Callback: {e}")

# --- NEURAL POLICY COMPUTATION (Called on H4 candle + startup) ---
def _compute_neural_policy(state, seq_tensor, state_tensor):
    """Recompute the weighted ensemble. Only called on strategic events."""
    with torch.no_grad():
        action_probs, _ = agent(state_tensor)
        trans_probs = torch.softmax(transformer(seq_tensor), dim=-1)
        xgb_probs = xgb_filter.predict_confidence(state.reshape(1, -1))
    
    w_ppo, w_trans, w_xgb = 0.40, 0.35, 0.25
    raw_probs = (action_probs.numpy()[0] * w_ppo + 
                 trans_probs.numpy()[0] * w_trans + 
                 xgb_probs[0] * w_xgb)
    
    if np.isnan(raw_probs).any():
        raw_probs = np.array([0.33, 0.33, 0.34])
    
    globals()['last_probs'] = [float(p) if not np.isnan(p) else 0.33 for p in raw_probs]
    globals()['last_confidence'] = float(raw_probs[np.argmax(raw_probs)])
    globals()['ensemble_probs'] = raw_probs
    globals()['last_smoothed_probs'] = raw_probs

# --- INDEPENDENT TELEMETRY HEARTBEAT ---
def telemetry_loop():
    global tick_counter, last_lot_size, SYSTEM_PAUSED, env
    last_processed_ticks = 0
    
    while True:
        try:
            # Use the rolling TPS latched by data_callback (accurate market velocity)
            tps = float(globals().get('last_tps', 0.0))

            # Calculate Performance Stats
            performance = {"win_rate": 0.0, "profit_factor": 0.0, "drawdown": 0.0}
            try:
                conn = sqlite3.connect(DB_FILE)
                df_history = pd.read_sql_query("SELECT profit FROM trades", conn)
                conn.close()
                if not df_history.empty:
                    # Win Rate
                    wins = len(df_history[df_history['profit'] > 0])
                    performance["win_rate"] = (wins / len(df_history)) * 100
                    
                    # Profit Factor (Guarded against Zero Division)
                    gross_p = float(df_history[df_history['profit'] > 0]['profit'].sum())
                    gross_l = abs(float(df_history[df_history['profit'] < 0]['profit'].sum()))
                    performance["profit_factor"] = (gross_p / gross_l) if gross_l > 0 else gross_p
                    
                    # Max Drawdown (Guarded against Empty Series)
                    initial_balance = 10000.0
                    equity_curve = initial_balance + df_history['profit'].cumsum()
                    running_max = equity_curve.cummax()
                    drawdowns = running_max - equity_curve
                    performance["drawdown"] = float(drawdowns.max())
            except Exception as pe:
                print(f"[TELEMETRY ERROR] Performance Calc Failed: {pe}")

            # Compute decay based on active trade start time
            decay = 0.0
            start_time_str = globals().get('last_start_time')
            if start_time_str:
                try:
                    start_dt = dt.fromisoformat(start_time_str)
                    duration_hours = (dt.now() - start_dt).total_seconds() / 3600
                    decay = (duration_hours / 24) * 0.1
                except Exception:
                    decay = 0.0

            # Compute entropy from last_probs (simple Shannon entropy)
            probs = globals().get('last_probs', [0.33, 0.33, 0.34])
            entropy = 0.0
            try:
                entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            except Exception:
                entropy = 0.0

            # Extract indicator snapshot from latest H4 (if available)
            indicators = {"ATR": 0.0, "BOS": False, "CHOCH": False}
            h4_df = None
            try:
                # Assuming the latest processed features contain '4h' df in globals
                h4_df = globals().get('last_tf_matrix')  # fallback, will be empty dict
            except Exception:
                h4_df = None

            # Guard: env must be initialized before telemetry can be built
            if env is None:
                time.sleep(1.0)
                continue

            # Build Telemetry
            telemetry = {
                "balance": float(env.balance if env.balance else 0.0),
                "pnl": float(globals().get('last_pnl', 0.0)),
                "confidence": float(globals().get('last_confidence', 0.0)) if not np.isnan(float(globals().get('last_confidence', 0.0))) else 0.0,
                "action": int(env.position if env.position else 0),
                "symbol": "BTCUSDm",
                "price": float(globals().get('last_price', 0.0)),
                "tps": float(tps),
                "performance": performance,
                "trade_forensics": {
                    "entry": float(env.entry_price if env.entry_price else 0.0),
                    "sl": float(globals().get('last_sl', 0.0)),
                    "tp": float(globals().get('last_tp', 0.0)),
                    "peak_pnl": float(globals().get('max_pnl_reached', 0.0)),
                    "start_time": str(globals().get('last_start_time', "--:--")),
                    "lot_size": float(globals().get('last_lot_size', 0.0)),
                    "reasoning": str(globals().get('last_reasoning', "SCANNING..."))
                },
                "probs": [float(p) for p in globals().get('last_probs', [0.33, 0.33, 0.34])],
                "diagnostics": {
                    "regime": str(globals().get('last_regime', "STABLE")),
                    "buffer": int(globals().get('buffer_len', 0)),
                    "decay": float(globals().get('last_decay', 0.0)),
                    "entropy": float(entropy),
                    "total_ticks": int(tick_counter),
                    "last_brain_sync": str(globals().get('last_update', 'NEVER')),
                    "is_syncing": bool(globals().get('IS_SYNCING', False))
                },
                "tf_matrix": globals().get('last_tf_matrix', {}),
                "indicators": indicators
            }

            with open(TELEMETRY_FILE, "w") as f:
                json.dump(telemetry, f)
                
        except Exception as e:
            print(f"[TELEMETRY CRASH] {e}") # Expose crash, never silently die
        
        time.sleep(1.0) # 1Hz Heartbeat

# Heartbeat thread is started inside main() after env is initialized

def process_h4_strategy(h4_df, current_price, features, tps, ensemble_probs, state_tensor, data):
    global active_rl_trade, last_confidence, last_sl, last_tp, last_start_time, last_reasoning, last_processed_timestamp, SYSTEM_PAUSED, client, env, risk_engine
    
    # Ensure timestamp is defined
    if 'last_processed_timestamp' not in globals(): globals()['last_processed_timestamp'] = None

    # 3. ONLY run AI Model for new entries on 4H Candle closure AND if FLAT
    if h4_df is not None and not h4_df.empty:
        current_ts = h4_df.index[-1]
        if current_ts != globals()['last_processed_timestamp']:
            globals()['last_processed_timestamp'] = current_ts
            
            if SYSTEM_PAUSED:
                ai_log(f"H4 Candle Triggered, but ENGINE IS PAUSED. Skipping decision.")
                return

            ai_log(f"--- [H4 STRATEGY] New 4H Candle Triggered: {current_ts} ---")

            # Recompute Neural Policy on new strategic data
            state = env._extract_state(features)
            state_tensor_h4 = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            sequence = env._extract_sequence(features, timeframe='4h', seq_len=50)
            seq_tensor_h4 = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            _compute_neural_policy(state, seq_tensor_h4, state_tensor_h4)
            ensemble_probs = globals()['ensemble_probs']


            # Check if we have active positions from MT5 directly
            import MetaTrader5 as mt5
            if mt5.initialize():
                positions = mt5.positions_get(symbol=data['symbol'])
                if not positions or len(positions) == 0:
                    if env.position != 0:
                        env.position = 0 # Sync flat state if MT5 is empty

            if env.position != 0:
                direction = 'BUY' if env.position == 1 else 'SELL'
                conf = float(globals().get('last_confidence', 0.0)) * 100
                globals()['last_reasoning'] = f"HOLDING {direction} | Neural Conviction: {conf:.1f}% | Monitoring for exit signals on next H4 close."
                print(f"[{data['symbol']}] Holding existing {direction} trade.")
            else:
                action = np.argmax(ensemble_probs)
                
                # --- NEURAL EXPLAINER (PRO) ---
                action_name = ['HOLD','BUY','SELL'][action]
                ai_log(f"AI Decision: {action_name}")
                
                # Sensory Feedback for reasoning
                last_h4 = h4_df.iloc[-1]
                prev_h4 = h4_df.iloc[-2] if len(h4_df) > 1 else last_h4
                price_change = ((last_h4['close'] - prev_h4['close']) / prev_h4['close']) * 100
                vol_change = ((last_h4['volume'] - prev_h4['volume']) / prev_h4['volume']) * 100 if prev_h4['volume'] > 0 else 0
                
                reasoning = f"H4 {action_name} | MOMENTUM: {price_change:+.1f}% | VOL: {vol_change:+.1f}% | TPS: {tps:.1f}"
                globals()['last_reasoning'] = reasoning

                if action != 0:
                    confidence = float(np.max(ensemble_probs))
                    lot_size = risk_engine.calculate_lot_size(env.balance, confidence, current_price)
                    
                    atr = float(h4_df['ATR'].iloc[-1]) if h4_df is not None and not h4_df.empty else 0.0
                    sl_dist = (atr * 1.5) if atr > 0 else (current_price * 0.005)
                    tp_dist = (atr * 3.0) if atr > 0 else (current_price * 0.01)

                    if action == 1: # BUY
                        ticket = client.send_order("BUY", data['symbol'], lot_size, sl=current_price - sl_dist, tp=current_price + tp_dist)
                        last_sl = current_price - sl_dist
                        last_tp = current_price + tp_dist
                    elif action == 2: # SELL
                        ticket = client.send_order("SELL", data['symbol'], lot_size, sl=current_price + sl_dist, tp=current_price - tp_dist)
                        last_sl = current_price + sl_dist
                        last_tp = current_price - tp_dist
                    
                    if 'ticket' in locals() and ticket is not None:
                        # Finalize Env Step
                        env.step(action, current_price, features, lot_size=lot_size, tps=tps)
                        
                        active_rl_trade["ticket"] = ticket
                        active_rl_trade["state"] = state_tensor.squeeze(0)
                        active_rl_trade["action"] = action
                        active_rl_trade["start_time"] = dt.now().isoformat()
                        globals()['last_start_time'] = active_rl_trade["start_time"]
                        globals()['last_lot_size'] = lot_size
                        save_memory()


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
            
            # Capture Tactical State for UI
            global last_lot_size, last_sl, last_tp, last_start_time
            last_lot_size = pos.volume
            last_sl = pos.sl
            last_tp = pos.tp
            
            from datetime import timezone
            last_start_time = dt.fromtimestamp(pos.time, tz=timezone.utc).isoformat()
            
            # --- HISTORICAL PEAK PROFIT (MFE) RECONSTRUCTION ---
            global max_pnl_reached
            try:
                open_time = dt.fromtimestamp(pos.time)
                rates = mt5.copy_rates_range("BTCUSDm", mt5.TIMEFRAME_M1, open_time, dt.now())
                if rates is not None and len(rates) > 0:
                    import pandas as _pd
                    hist = _pd.DataFrame(rates)
                    if env.position == -1:  # SELL: profit at lowest price
                        best_price = float(hist['low'].min())
                        hist_peak_pnl = (env.entry_price - best_price) * pos.volume * contract_size
                    else:  # BUY: profit at highest price
                        best_price = float(hist['high'].max())
                        hist_peak_pnl = (best_price - env.entry_price) * pos.volume * contract_size
                    
                    # Take the higher of persisted vs historically reconstructed
                    max_pnl_reached = max(max_pnl_reached, hist_peak_pnl)
                    print(f">>> [MFE] Historical Peak Profit Reconstructed: ${max_pnl_reached:.2f} (Best Price: {best_price:.2f})")
            except Exception as mfe_err:
                print(f">>> [MFE] Reconstruction failed: {mfe_err}")
            
            pos_type_str = "BUY" if env.position == 1 else "SELL"
            globals()['last_reasoning'] = f"ENGINE BOOT: Detected active {pos_type_str} position from MT5. Monitoring trade."
            print(f">>> [SYNC] Found Active MT5 Position: {pos_type_str} at {env.entry_price} (Lots: {last_lot_size}). AI will MONITOR, not open a new trade.")
        else:
            print(">>> [SYNC] No active positions found. AI is FLAT and ready to enter.")
    # ---------------------------

    client = MT5ZMQClient(Config.ZMQ_PULL_PORT, Config.ZMQ_PUSH_PORT)
    
    # --- ASYNC AUTO-SYNC (PRO) ---
    # We run this in a background thread so the engine starts instantly
    threading.Thread(target=sync_historical_data, args=("BTCUSDm",), daemon=True).start()
    
    # Start the telemetry heartbeat AFTER env is synced
    threading.Thread(target=telemetry_loop, daemon=True).start()
    print(">>> [HEARTBEAT] Independent Telemetry Thread Started (1Hz) <<<")
    
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

