
import torch
import numpy as np
from sim.env import TradingEnvironment
from agent.actor_critic import TradingActorCritic
import yaml
import os

def run_ablation_test(mask_name, start_idx, end_idx):
    print(f"\n>>> TESTING ABLATION: {mask_name} (Masking indices {start_idx}:{end_idx})")
    
    with open('config/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    env = TradingEnvironment(config)
    agent = TradingActorCritic(latent_dim=128, n_actions=3)
    
    if os.path.exists('models/ppo_agent_latest.pt'):
        agent.load_state_dict(torch.load('models/ppo_agent_latest.pt', map_location='cpu'))
        print(">>> Model weights loaded.")
    
    obs, _ = env.reset()
    
    # Validation Loop (10,000 steps)
    for i in range(10000):
        # Apply Ablation Mask
        if start_idx is not None:
            obs[start_idx:end_idx] = 0.0
            
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        regime_tensor = torch.LongTensor([env.regime.current_code])
        with torch.no_grad():
            action, _, _, _ = agent.get_action(obs_tensor, regime_tensor, threshold=0.10)
            
        obs, reward, done, _, info = env.step(action.item())
        
        if done:
            obs, _ = env.reset()
            
    # Audit Results
    metrics = env.perf_metrics['trade_metrics']
    trades = metrics['realized_pnl']
    pnl = sum(trades)
    pf = sum([t for t in trades if t > 0]) / (abs(sum([t for t in trades if t < 0])) + 1e-9)
    exp = pnl / max(1, len(trades))
    
    print(f"RESULTS [{mask_name}]:")
    print(f"  PNL:        ${pnl:+.2f}")
    print(f"  Trades:     {len(trades)}")
    print(f"  PF:         {pf:.2f}")
    print(f"  Expectancy: ${exp:.4f}/trade")
    
    return {'name': mask_name, 'pnl': pnl, 'pf': pf, 'exp': exp, 'trades': len(trades)}

if __name__ == "__main__":
    results = []
    # A: Full Features (Baseline)
    results.append(run_ablation_test("Full Features", None, None))
    # B: No Ticks (indices 0-240)
    results.append(run_ablation_test("No Ticks", 0, 240))
    # C: No MTF (indices 240-940)
    results.append(run_ablation_test("No MTF", 240, 940))
    # D: No Scalars (indices 940-950)
    results.append(run_ablation_test("No Scalars", 940, 950))
    
    print("\n" + "="*50)
    print("FINAL ABLATION SUMMARY")
    print("="*50)
    for r in results:
        print(f"{r['name']:15} | PNL: {r['pnl']:+8.2f} | PF: {r['pf']:.2f} | EXP: {r['exp']:+8.4f} | T: {r['trades']}")
    print("="*50)
