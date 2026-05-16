
import sqlite3
import pandas as pd
import torch
import numpy as np
import yaml
import sys
import os

# Add root and src to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "src"))

from sim.env import TradingEnvironment
from agent.actor_critic import TradingActorCritic
from training.ppo_trainer import PPOTrainer

def run_oos_validation():
    # 1. Load Config
    with open("config/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> [OOS PROTOCOL v3.3] Institutional Validation (Device: {device}) <<<")

    # Initialize Environment & Agent
    config['database']['conn_str'] = "postgresql://postgres:postgres@localhost:5432/axon_market" 
    env = TradingEnvironment(config)
    agent = TradingActorCritic(latent_dim=128, hidden_dim=256, n_actions=6).to(device)
    
    weights_path = "models/ppo_agent_latest.pt"
    if os.path.exists(weights_path):
        agent.load_state_dict(torch.load(weights_path, map_location=device))
        print(f">>> Weights Verified: {weights_path}")

    # 2. Execution Audit State
    obs, _ = env.reset()
    results = {
        'rewards': [], 'pnl': [], 'pnl_ratio': [],
        'decomposition': {'pnl': 0, 'stability': 0, 'persistence': 0, 'risk': 0, 'bonus': 0},
        'actions': {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
    }
    
    print("\n>>> EXECUTION: 50,000 Step Institutional Stress-Test <<<")
    total_steps = 0
    while total_steps < 50000:
        obs, _ = env.reset()
        done = False
        while not done and total_steps < 50000:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action(state_tensor)
                action = action.item()
            
            obs, reward, done, _, info = env.step(action)
            total_steps += 1
            
            # Accumulate Audit Data
            results['rewards'].append(reward)
            results['pnl'].append(info.get('equity') - 10000)
            results['actions'][action] += 1
            
            if hasattr(env, 'last_decomposition'):
                for k, v in env.last_decomposition.items():
                    results['decomposition'][k] += v
                results['pnl_ratio'].append(env.last_pnl_ratio)

            if total_steps % 5000 == 0:
                avg_ratio = np.mean(results['pnl_ratio'][-5000:]) if results['pnl_ratio'] else 0
                print(f"Step {total_steps:5d} | Equity: ${info['equity']:8.2f} | PnL Ratio: {avg_ratio:.1%} | Action: {action}")
            
    print(">>> [HALT] 50,000 Step Protocol Complete.")

    # 3. Final Metric Computation
    final_equity = info['equity']
    equity_delta = final_equity - 10000.0
    
    # Institutional Expectancy Audit (Realized Only)
    trades = env.perf_metrics['trade_metrics']['realized_pnl']
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    
    total_trades = len(trades)
    realized_pnl_sum = sum(trades)
    
    expectancy = realized_pnl_sum / max(1, total_trades)
    profit_factor = sum(wins) / abs(sum(losses) + 1e-9)
    
    # Reward vs PnL Forensic
    total_abs_sum = sum(abs(v) for v in results['decomposition'].values()) + 1e-9
    decomp_pct = {k: (v / total_abs_sum) * 100 for k, v in results['decomposition'].items()}

    print("\n" + "="*50)
    print("INSTITUTIONAL OOS AUDIT (50,000 STEPS)")
    print("="*50)
    print(f"Final Equity:      ${final_equity:.2f}")
    print(f"Equity Delta:      ${equity_delta:+.2f}")
    print(f"Realized PnL:      ${realized_pnl_sum:+.2f}")
    print(f"Expectancy (Edge): ${expectancy:+.4f} / trade")
    print(f"Profit Factor:     {profit_factor:.2f}")
    print(f"PnL Signal Ratio:  {np.mean(results['pnl_ratio']):.1%}")
    
    print("\n--- Integrity Gates ---")
    # Gate A: PF vs Expectancy
    gate_a = "PASS" if not (profit_factor > 1.1 and expectancy < 0) else "REJECT (Divergence)"
    # Gate B: PnL vs Equity
    gate_b = "PASS" if abs(realized_pnl_sum - equity_delta) < 100 else "REJECT (Drift)"
    
    print(f"Gate A (PF/Exp):   {gate_a}")
    print(f"Gate B (PnL/Eq):   {gate_b}")

    print("\n--- Reward Decomposition (%) ---")
    for k, v in decomp_pct.items():
        print(f"{k.capitalize():12}: {v:+.1f}%")
    
    print("\n--- Conclusion ---")
    if gate_a != "PASS" or gate_b != "PASS":
        print("[CRITICAL] Metric Integrity Failure. Audit corrupted.")
    elif np.mean(results['pnl_ratio']) < 0.5:
        print("[REJECT] Reward Optimization Drift (Shaping dominates PnL)")
    elif expectancy <= 0:
        print("[STABLE] No edge detected; behavior is defensive but not profitable.")
    else:
        print("[ALPHA] Alignment and Integrity verified. Edge detected in OOS.")
    print("="*50)

if __name__ == "__main__":
    run_oos_validation()
