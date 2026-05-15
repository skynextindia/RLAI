# main.py

import argparse
import yaml
import torch
# from data.pipeline import TickPipeline
# from data.downloader import download_ticks
from model.pretrain import pretrain
from agent.ppo import PPOTrainer
from agent.actor_critic import TradingActorCritic
from model.encoder import MarketEncoder
from sim.env import TradingEnvironment
from datetime import datetime, timedelta

def load_config(path="config/base.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_pipeline(symbol, config):
    pipeline = TickPipeline(symbol, config['database']['conn_str'])
    pipeline.start()

def run_download(symbol, days, config):
    end = datetime.now()
    start = end - timedelta(days=days)
    df = download_ticks(symbol, start, end)
    path = f"data/historical/{symbol}_ticks.parquet"
    df.to_parquet(path)
    print(f"Saved to {path}")

def run_pretrain(symbol, config):
    # This assumes we have a parquet file ready
    data_path = f"data/historical/{symbol}_ticks.parquet"
    pretrain(
        data_path = data_path,
        save_path = config['pretrain']['save_path'],
        epochs    = config['pretrain']['epochs'],
        batch_size= config['pretrain']['batch_size'],
        device    = config['ppo']['device']
    )

def run_train(symbol, config):
    print(f"Starting Phase 4 for {symbol}...", flush=True)
    config['sim_symbol'] = symbol
    print("Initializing environment...", flush=True)
    env = TradingEnvironment(config)
    print("Initializing encoder...", flush=True)
    encoder = MarketEncoder()
    # Load pre-trained weights if they exist
    try:
        print(f"Loading weights from {config['pretrain']['save_path']}...", flush=True)
        encoder.load_state_dict(torch.load(config['pretrain']['save_path']))
        print("Loaded pre-trained encoder.", flush=True)
    except Exception as e:
        print(f"Starting with untrained encoder. Reason: {e}", flush=True)
        
    print("Initializing Agent...", flush=True)
    agent = TradingActorCritic(encoder)
    print("Initializing Trainer...", flush=True)
    trainer = PPOTrainer(agent, env, config['ppo'])
    print("Starting training loop...", flush=True)
    trainer.train(total_timesteps=config['ppo']['total_timesteps'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6], help="Phase to run")
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--download_days", type=int, default=365)
    args = parser.parse_args()
    
    config = load_config()
    
    if args.phase == 1:
        # run_pipeline(args.symbol, config)
        run_download(args.symbol, args.download_days, config)
    elif args.phase == 3:
        run_pretrain(args.symbol, config)
    elif args.phase == 4:
        run_train(args.symbol, config)
    else:
        print("Specify a phase (e.g., --phase 1)")
