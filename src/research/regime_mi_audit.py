
import os
import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sim.env import TradingEnvironment
from model.institutional_encoder import InstitutionalEncoder
from sklearn.cluster import HDBSCAN

def run_regime_audit():
    with open('config/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['sim_symbol'] = 'BTCUSDm'
    env = TradingEnvironment(config)
    
    samples = 20000
    print(f"Collecting {samples} samples for Regime-Conditioned Audit...")
    obs_buffer = []
    prices = []
    
    obs, _ = env.reset()
    for i in range(samples):
        obs_buffer.append(obs)
        idx = env.step_count + env.SEQUENCE_LEN - 1
        prices.append(env.ticks[idx].get('bid', 1.0))
        obs, _, done, _, _ = env.step(0)
        if done: obs, _ = env.reset()

    obs_array = np.array(obs_buffer)
    prices = np.array(prices)
    future_returns = np.log(pd.Series(prices).shift(-100) / prices).fillna(0).values

    # 1. Latent Regime Discovery
    print("Partitioning Latent Space into Regimes...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = InstitutionalEncoder(config['encoder']['d_model']).to(device)
    obs_t = torch.from_numpy(obs_array).float().to(device)
    
    with torch.no_grad():
        latents = encoder(obs_t).cpu().numpy()
    
    clusterer = HDBSCAN(min_cluster_size=100)
    regime_labels = clusterer.fit_predict(latents)
    
    unique_regimes = np.unique(regime_labels)
    
    # 2. Define Feature Families (710 Dimensions)
    # 0-199: M1 (Returns, Mom)
    # 200-399: M5
    # 400-599: M15
    # 600-699: H1
    # 700-709: Scalars
    
    families = {
        'MTF_Returns': [i for i in range(0, 700, 2)],
        'MTF_Momentum': [i for i in range(1, 700, 2)],
        'Scalars': [i for i in range(700, 710)]
    }

    print("\n" + "="*80)
    print(f"{'REGIME':<15} | {'COUNT':<6} | {'RETURNS MI':<12} | {'MOMENTUM MI':<12} | {'SCALAR MI':<10}")
    print("-" * 80)

    for r in unique_regimes:
        mask = (regime_labels == r)
        r_obs = obs_array[mask]
        r_ret = future_returns[mask]
        
        if len(r_obs) < 500: continue # Skip tiny clusters
        
        name = "NOISE" if r == -1 else f"REGIME_{r}"
        
        # Calculate MI for family representatives
        scores = {}
        for fam_name, indices in families.items():
            subset = r_obs[:, indices[:10]] # Take 10 reps to save time
            scores[fam_name] = np.mean(mutual_info_regression(subset, r_ret))
            
        print(f"{name:<15} | {len(r_obs):<6} | {scores['MTF_Returns']:<12.6f} | {scores['MTF_Momentum']:<12.6f} | {scores['Scalars']:<10.6f}")

    print("="*80)
    print("\nCONCLUSION:")
    print("If MI scores are consistent across regimes -> The 710-dim state is STABLE.")
    print("If MI drops to 0 in certain regimes -> The agent is BLIND in those states.")

if __name__ == "__main__":
    run_regime_audit()
