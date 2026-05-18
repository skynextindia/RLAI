import sqlite3
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import yaml
import sys
import os

# Add root and src to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "src"))

from sim.env import TradingEnvironment

class LegacyInstitutionalEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        
        self.tick_processor = nn.Sequential(
            nn.Linear(240, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64)
        )
        self.m1_processor = nn.Sequential(nn.Linear(200, 64), nn.GELU())
        self.m5_processor = nn.Sequential(nn.Linear(200, 64), nn.GELU())
        self.m15_processor = nn.Sequential(nn.Linear(200, 64), nn.GELU())
        self.h1_processor = nn.Sequential(nn.Linear(100, 32), nn.GELU())
        
        self.scalar_processor = nn.Sequential(
            nn.Linear(10, 16),
            nn.GELU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(304, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 950)
        t60     = x[:, 0:240]
        m1      = x[:, 240:440]
        m5      = x[:, 440:640]
        m15     = x[:, 640:840]
        h1      = x[:, 840:940]
        scalars = x[:, 940:950]
        
        self.last_activations = {
            'tick':   self.tick_processor(t60),
            'm1':     self.m1_processor(m1),
            'm5':     self.m5_processor(m5),
            'm15':    self.m15_processor(m15),
            'h1':     self.h1_processor(h1),
            'scalar': self.scalar_processor(scalars)
        }
        
        combined = torch.cat(list(self.last_activations.values()), dim=-1)
        latent   = self.fusion(combined)
        return latent

    def get_state_vector(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

class LegacyTradingActorCritic(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256, n_actions: int = 6):
        super().__init__()
        self.encoder = LegacyInstitutionalEncoder(latent_dim=latent_dim)
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, obs: torch.Tensor):
        z = self.encoder.get_state_vector(obs)
        logits = self.actor(z)
        value = self.critic(z)
        return logits, value

    def get_action(self, obs: torch.Tensor, threshold: float = 0.55):
        logits, value = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        max_prob, max_action = torch.max(probs, dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        if max_prob.item() < threshold:
            action = torch.zeros_like(action)
        log_prob = dist.log_prob(action)
        return action, log_prob, value, dist.entropy()

def get_legacy_obs(env):
    idx = env.step_count + env.SEQUENCE_LEN - 1
    t60 = env.tick_features[env.step_count : env.step_count + env.SEQUENCE_LEN].flatten()
    mtf = env.state_builder.get_mtf_slice(idx)
    metrics = env.state_builder.get_market_metrics(idx)
    
    hour = env._get_hour(env.ticks[idx].get('time', 0))
    scalars = np.array([
        np.clip(metrics['market_speed'] / 1000.0, -10, 10),
        np.clip(metrics['volatility'] / 100.0, -10, 10),
        np.clip(metrics['imbalance'], -1, 1),
        np.clip(env.position.size * 10.0, -1, 1),
        np.clip(env.position.floating_pnl / 100.0, -20, 20),
        env.regime.current_code / 10.0,
        np.clip((env.account_equity / env.initial_equity) - 1.0, -1, 1),
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        env.exec.base_slippage * 1000.0
    ], dtype=np.float32)

    return np.concatenate([t60, mtf, scalars])

def run_oos_validation():
    with open("config/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> [LEGACY OOS PROTOCOL v3.3] Institutional Validation (Device: {device}) <<<")

    # Initialize Environment & Agent
    config['database']['conn_str'] = "postgresql://postgres:postgres@localhost:5432/axon_market" 
    env = TradingEnvironment(config)
    agent = LegacyTradingActorCritic(latent_dim=128, hidden_dim=256, n_actions=6).to(device)
    
    weights_path = "models/ppo_agent_behavioral_stable.pt"
    if os.path.exists(weights_path):
        agent.load_state_dict(torch.load(weights_path, map_location=device))
        print(f">>> Weights Verified: {weights_path}")

    # 2. Execution Audit State
    env.reset()
    obs = get_legacy_obs(env)
    results = {
        'rewards': [], 'pnl': [], 'pnl_ratio': [],
        'decomposition': {'pnl': 0, 'stability': 0, 'persistence': 0, 'risk': 0, 'bonus': 0},
        'actions': {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
    }
    
    print("\n>>> EXECUTION: 50,000 Step Institutional Stress-Test <<<")
    total_steps = 0
    while total_steps < 50000:
        env.reset()
        obs = get_legacy_obs(env)
        done = False
        while not done and total_steps < 50000:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action(state_tensor)
                action = action.item()
            
            _, reward, done, _, info = env.step(action)
            obs = get_legacy_obs(env)
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
    
    trades = env.perf_metrics['trade_metrics']['realized_pnl']
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    
    total_trades = len(trades)
    realized_pnl_sum = sum(trades)
    
    expectancy = realized_pnl_sum / max(1, total_trades)
    profit_factor = sum(wins) / abs(sum(losses) + 1e-9)
    
    total_abs_sum = sum(abs(v) for v in results['decomposition'].values()) + 1e-9
    decomp_pct = {k: (v / total_abs_sum) * 100 for k, v in results['decomposition'].items()}

    print("\n" + "="*50)
    print("INSTITUTIONAL OOS AUDIT (50,000 STEPS) - BEHAVIORAL STABLE")
    print("="*50)
    print(f"Final Equity:      ${final_equity:.2f}")
    print(f"Equity Delta:      ${equity_delta:+.2f}")
    print(f"Realized PnL:      ${realized_pnl_sum:+.2f}")
    print(f"Expectancy (Edge): ${expectancy:+.4f} / trade")
    print(f"Profit Factor:     {profit_factor:.2f}")
    print(f"PnL Signal Ratio:  {np.mean(results['pnl_ratio']):.1%}")
    
    print("\n--- Integrity Gates ---")
    gate_a = "PASS" if not (profit_factor > 1.1 and expectancy < 0) else "REJECT (Divergence)"
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
