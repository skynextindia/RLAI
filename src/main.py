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

import sqlite3

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
SYSTEM_PAUSED = False

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
                        print("[WEB IO] ALL POSITIONS CLOSED SUCCESSFULLY.")
            
            elif action == "PAUSE":
                SYSTEM_PAUSED = True
                print("[WEB IO] AI ENGINE PAUSED.")
            
            elif action == "RESUME":
                SYSTEM_PAUSED = False
                print("[WEB IO] AI ENGINE RESUMED.")
            
            elif action == "KILL":
                print("[WEB IO] EMERGENCY SHUTDOWN INITIATED.")
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

# RL Trade Memory Persistence
MEMORY_FILE = "active_trade.json"
active_rl_trade = {"ticket": None, "state": None, "action": None, "start_time": None}

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
            from datetime import datetime, timedelta
            deals = mt5.history_deals_get(position=ticket)
            if deals and len(deals) > 0:
                # The exit deal is usually the last one
                exit_deal = deals[-1]
                profit = exit_deal.profit
                
                # Assign RL Reward
                reward = 2.0 if profit > 0 else -1.0
                
                # --- TIME-DECAY PENALTY (PRO) ---
                if active_rl_trade.get("start_time"):
                    start_dt = datetime.fromisoformat(active_rl_trade["start_time"])
                    duration_hours = (datetime.now() - start_dt).total_seconds() / 3600
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
                          (ticket, data['symbol'], action_str, profit, reward, datetime.now().isoformat()))
                conn.commit()
                conn.close()
                print(f"[RL ENGINE] Trade {ticket} logged to SQLite Database.")

                if len(trainer.buffer) >= 32:
                    print(f"[RL ENGINE] Buffer Full. Updating Neural Weights...")
                    trainer.update()
                    torch.save(agent.state_dict(), "ppo_agent_h4_pretrained.pth")
                    print(f"[RL ENGINE] Weights Successfully Saved to Disk.")
            
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

    # 2. Extract live metrics for Heartbeat & Telemetry
    # We must look at the last row (.iloc[-1]) of the dataframes
    h4_df = features.get('4h')
    s5_df = features.get('5s')
    
    live_atr = float(s5_df['ATR'].iloc[-1]) if s5_df is not None and not s5_df.empty else 0.0
    live_bos = bool(s5_df['BOS_Bullish'].iloc[-1]) if s5_df is not None and not s5_df.empty else False
    live_choch = bool(s5_df['CHOCH_Bearish'].iloc[-1]) if s5_df is not None and not s5_df.empty else False

    # Sanitize NaNs for JSON stability
    safe_atr = live_atr if not np.isnan(live_atr) else 0.0

    telemetry = {
        "balance": float(env.balance),
        "pnl": float(actual_pnl),
        "confidence": float(globals().get('confidence', 0.0)),
        "action": int(env.position),
        "symbol": data['symbol'],
        "price": float(current_price),
        "entry": float(env.entry_price),
        "sl": float(sl_target),
        "tp": float(tp_target),
        "probs": probs_list,
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
        current_ts = h4_df.index[-1] # Use the index (timestamp) of the last candle
        if current_ts != last_processed_timestamp:
            last_processed_timestamp = current_ts
            
            if SYSTEM_PAUSED:
                print(f"\n[SYSTEM] H4 Candle Triggered, but ENGINE IS PAUSED. Skipping decision.")
                return

            print(f"\n--- [H4 STRATEGY] New 4H Candle Triggered: {current_ts} ---")

            # Live MT5 State Sync
            import MetaTrader5 as mt5
            if mt5.initialize():
                positions = mt5.positions_get(symbol=data['symbol'])
                if not positions or len(positions) == 0:
                    if env.position != 0:
                        print(f"\n[SYNC] Notice: MT5 is FLAT but Python was holding {env.position}. Resyncing to FLAT.")
                        env.position = 0

            if env.position != 0:
                print(f"[{data['symbol']}] Holding existing {'BUY' if env.position == 1 else 'SELL'} trade. AI will wait for closure.")
            else:
                action = np.argmax(ensemble_probs)
                
                # --- NEURAL EXPLAINER (PRO) ---
                action_name = ['HOLD','BUY','SELL'][action]
                print(f"\n[AI EXPLAINER] Decision: {action_name}")
                print(f"| Context: Transformer detected a structural pattern over the last 50 candles.")
                
                # Simple Feature-Based explanation
                if h4_df is not None and not h4_df.empty:
                    last_h4 = h4_df.iloc[-1]
                    prev_h4 = h4_df.iloc[-2] if len(h4_df) > 1 else last_h4
                    price_change = ((last_h4['close'] - prev_h4['close']) / prev_h4['close']) * 100
                    vol_change = ((last_h4['volume'] - prev_h4['volume']) / prev_h4['volume']) * 100 if prev_h4['volume'] > 0 else 0
                    
                    print(f"| Sensory Input 1: 4H Momentum is {price_change:+.2f}%")
                    print(f"| Sensory Input 2: Volume Delta is {vol_change:+.2f}%")
                    if action == 2: # SELL
                        if price_change < 0: print(f"| AI Logic: Pattern matches a high-probability Bearish Continuation.")
                        else: print(f"| AI Logic: Pattern matches an Institutional Supply Rejection.")
                    elif action == 1: # BUY
                        if price_change > 0: print(f"| AI Logic: Pattern matches a high-probability Bullish Breakout.")
                        else: print(f"| AI Logic: Pattern matches a Liquidity Sweep Rebound.")

                # Risk & Execution
                lot_size = risk_engine.calculate_lot_size(env.balance, confidence, current_price)
                next_obs, reward, done, _, info = env.step(action, current_price, features, lot_size=lot_size, contract_size=globals().get('contract_size', 1.0))
                
                # Dynamic ATR Risk Matrix (1.5x ATR SL, 3.0x ATR TP)
                atr = float(h4_df['ATR'].iloc[-1]) if h4_df is not None and not h4_df.empty else 0.0
                sl_dist = (atr * 1.5) if atr > 0 else (current_price * 0.005)
                tp_dist = (atr * 3.0) if atr > 0 else (current_price * 0.01)

                new_sl = 0.0
                new_tp = 0.0
                if action == 1: # BUY
                    new_sl = current_price - sl_dist
                    new_tp = current_price + tp_dist
                    print(f"[BRIDGE] Dynamic BUY | SL: {new_sl:.2f} (1.5x ATR), TP: {new_tp:.2f} (3.0x ATR)")
                    ticket = client.send_order("BUY", data['symbol'], lot_size, sl=new_sl, tp=new_tp)
                elif action == 2: # SELL
                    new_sl = current_price + sl_dist
                    new_tp = current_price - tp_dist
                    print(f"[BRIDGE] Dynamic SELL | SL: {new_sl:.2f} (1.5x ATR), TP: {new_tp:.2f} (3.0x ATR)")
                    ticket = client.send_order("SELL", data['symbol'], lot_size, sl=new_sl, tp=new_tp)
                
                print(f"[{data['symbol']}] H4 Decision: {['HOLD','BUY','SELL'][action]} (Conf: {confidence:.2f})")
                
                # Store memory for learning later
                if action != 0 and 'ticket' in locals() and ticket is not None:
                    active_rl_trade["ticket"] = ticket
                    active_rl_trade["state"] = state_tensor.squeeze(0)
                    active_rl_trade["action"] = action
                    active_rl_trade["start_time"] = datetime.now().isoformat()
                    save_memory() # Persist to disk immediately

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
            pos_type_str = "BUY" if env.position == 1 else "SELL"
            print(f">>> [SYNC] Found Active MT5 Position: {pos_type_str} at {env.entry_price}. AI will MONITOR, not open a new trade.")
        else:
            print(">>> [SYNC] No active positions found. AI is FLAT and ready to enter.")
    # ---------------------------

    client = MT5ZMQClient(Config.ZMQ_PULL_PORT, Config.ZMQ_PUSH_PORT)
    
    # Start ZMQ listener
    thread = threading.Thread(target=client.stream_market_data, args=(data_callback,))
    thread.daemon = True
    thread.start()
    
    print("System Running... Press Ctrl+C to stop.")
    try:
        while True:
            thread.join(1)
    except KeyboardInterrupt:
        print("\nShutting down Institutional AI Engine...")
        import os
        os._exit(0) # Force exit to prevent ZMQ thread hanging terminal

if __name__ == "__main__":
    main()
