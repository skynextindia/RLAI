import MetaTrader5 as mt5
import pandas as pd
import sqlite3
import torch
import sys
import os

# Add src to path so we can import modules when running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.trading_env import MT5TradingEnv
from models.ppo_agent import PPOAgent
from training.ppo_trainer import PPOTrainer
from data.feature_engineering import FeatureEngineer

DB_PATH = "market_data.db"

def fetch_and_store_history(symbol="BTCUSDm", timeframe=mt5.TIMEFRAME_H4, num_candles=5000):
    if not mt5.initialize():
        print("MT5 Init Failed")
        return
    
    print(f">>> Fetching {num_candles} candles from MT5...")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    mt5.shutdown()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Store in SQLite
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("historic_4h", conn, if_exists="replace", index=False)
    conn.close()
    print(f">>> Data stored in {DB_PATH}")

def train_from_db():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM historic_4h", conn)
    conn.close()
    
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Feature Engineering
    engineer = FeatureEngineer()
    df = engineer.calculate_atr(df)
    df = engineer.detect_choch_bos(df)
    df.fillna(0, inplace=True)
    
    # RL Init
    env = MT5TradingEnv()
    agent = PPOAgent(state_dim=12, action_dim=3)
    trainer = PPOTrainer(agent)
    
    print(f">>> Training on {len(df)} candles...")
    for epoch in range(10): # 10 full passes over history
        env.reset()
        for i in range(20, len(df)):
            current_row = df.iloc[i]
            features = {'5s': current_row.to_dict(), '1min': current_row.to_dict(), '5min': current_row.to_dict()}
            state = env._extract_state(features)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                action_probs, _ = agent(state_tensor)
                action = torch.argmax(action_probs).item()
            
            next_obs, reward, done, _, info = env.step(action, current_row['close'], features)
            next_state_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            
            trainer.store_transition(state_tensor.squeeze(0), action, reward, next_state_tensor.squeeze(0), done)
            if len(trainer.buffer) >= 64:
                trainer.update()
        
        print(f"Epoch {epoch+1} complete. Balance: ${env.balance:.2f}")
    
    torch.save(agent.state_dict(), "ppo_agent_h4_pretrained.pth")
    print(">>> Pre-training finished. Model saved.")

if __name__ == "__main__":
    fetch_and_store_history()
    train_from_db()
