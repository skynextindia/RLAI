
import os
import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sim.env import TradingEnvironment

def run_utility_audit():
    with open('config/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['sim_symbol'] = 'BTCUSDm'
    env = TradingEnvironment(config)
    
    samples = 30000
    print(f"Collecting {samples} samples for Deep Signal Audit...")
    obs_buffer = []
    prices = []
    
    obs, _ = env.reset()
    for i in range(samples):
        # Current price for return calculation
        idx = env.step_count + env.SEQUENCE_LEN - 1
        prices.append(env.ticks[idx].get('bid', 1.0))
        
        obs_buffer.append(obs)
        obs, _, done, _, _ = env.step(0) # Observe passive drift
        if done: obs, _ = env.reset()

    obs_array = np.array(obs_buffer)
    prices = np.array(prices)
    
    # Define Feature Indices
    # MTF Blocks: Returns=0, Vol=1, VWAP=2, Mom=3 (repeated every 4)
    mtf_start = 240
    vwap_indices = []
    mom_indices = []
    vol_indices = []
    ret_indices = []
    
    for i in range(mtf_start, 1640, 4):
        ret_indices.append(i)
        vol_indices.append(i+1)
        vwap_indices.append(i+2)
        mom_indices.append(i+3)

    results = []
    horizons = [20, 100, 500]
    
    print("\n" + "="*60)
    print(f"{'FEATURE FAMILY':<25} | {'t+20 MI':<10} | {'t+100 MI':<10} | {'t+500 MI':<10}")
    print("-" * 60)

    def get_mi(indices, target):
        # Sample subset to speed up
        subset = obs_array[:, indices]
        if subset.shape[1] > 20:
            subset = subset[:, ::subset.shape[1]//10] # Take 10 representatives
        return np.mean(mutual_info_regression(subset, target))

    for h in horizons:
        future_returns = np.log(pd.Series(prices).shift(-h) / prices).fillna(0).values
        
        row = {'h': h}
        row['ret'] = get_mi(ret_indices, future_returns)
        row['vol'] = get_mi(vol_indices, future_returns)
        row['vwap'] = get_mi(vwap_indices, future_returns)
        row['mom'] = get_mi(mom_indices, future_returns)
        row['tick'] = get_mi(np.arange(0, 240, 12), future_returns)
        row['scalar'] = get_mi(np.arange(1640, 1650), future_returns)
        results.append(row)

    families = [
        ('MTF Returns', 'ret'),
        ('MTF Vol-Norm', 'vol'),
        ('VWAP Distance', 'vwap'),
        ('Momentum Slope', 'mom'),
        ('Raw Tick Block', 'tick'),
        ('Environment Scalars', 'scalar')
    ]

    for label, key in families:
        mi_vals = [r[key] for r in results]
        print(f"{label:<25} | {mi_vals[0]:.6f}   | {mi_vals[1]:.6f}    | {mi_vals[2]:.6f}")

    print("="*60)
    
    # Check for Curse of Dimensionality
    max_mi = max([r['vwap'] for r in results])
    if max_mi < 0.01:
        print("\nWARNING: Structural Signal Density is Critically Low (<0.01 MI).")
        print("Dimensionality pruning highly recommended to prevent noise-memorization.")
    else:
        print("\nCONFIRMED: Structural features contain predictive information.")

if __name__ == "__main__":
    run_utility_audit()
