
import os
import yaml
import torch
import numpy as np
from sklearn.cluster import HDBSCAN
from model.institutional_encoder import InstitutionalEncoder
from sim.env import TradingEnvironment

def discover_regimes():
    with open('config/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Precise Latent Dim extraction
    latent_dim = config['encoder'].get('d_model', 128)
    if isinstance(latent_dim, dict):
        latent_dim = 128 # Fallback if nested
        
    print(f"Initializing InstitutionalEncoder with LatentDim={latent_dim}")
    encoder = InstitutionalEncoder(latent_dim).to(device)
    
    # Load latest checkpoint if exists
    ckpt_path = "models/ppo_agent_latest.pt" # Check standard path first
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        # Handle policy state dict vs raw encoder weights
        if 'encoder_state_dict' in state:
            encoder.load_state_dict(state['encoder_state_dict'])
        elif 'model_state_dict' in state:
            # Try to extract encoder part
            encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in state['model_state_dict'].items() if 'encoder' in k})
        else:
            print("Warning: Could not map weights directly. Using randomized init.")
    
    encoder.eval()
    
    config['sim_symbol'] = 'BTCUSDm'
    env = TradingEnvironment(config)
    
    samples = 5000
    print(f"Profiling {samples} latent states for Regime Coherence...")
    latents = []
    metadata = []
    
    obs, _ = env.reset()
    for i in range(samples):
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            latent = encoder(obs_t)
            latents.append(latent.cpu().numpy().flatten())
        
        # Market profile (Vol + Momentum)
        idx = env.step_count + env.SEQUENCE_LEN - 1
        vol = env.state_builder.get_market_metrics(idx)['volatility']
        metadata.append(vol)
        
        obs, _, done, _, _ = env.step(0)
        if done: obs, _ = env.reset()

    latents = np.array(latents)
    
    print(f"Running HDBSCAN on {latents.shape} latent space...")
    clusterer = HDBSCAN(min_cluster_size=100, min_samples=15)
    labels = clusterer.fit_predict(latents)
    
    unique_labels = np.unique(labels)
    print("\n" + "="*50)
    print("REGIME DISCOVERY AUDIT")
    print("="*50)
    print(f"Clusters Detected: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")
    
    for label in unique_labels:
        name = "TRANSITION/NOISE" if label == -1 else f"REGIME_{label}"
        mask = (labels == label)
        avg_vol = np.mean(np.array(metadata)[mask])
        print(f"  - {name:<18} | Pts: {np.sum(mask):>4} | Avg Vol: {avg_vol:.6f}")

    if len(unique_labels) > 2:
        print("\nSTATUS: Coherent Regime Separation Found.")
        print("Agent is successfully partitioning the 1,650-dim state.")
    else:
        print("\nSTATUS: Latent Collapse Detected.")
        print("The 1,650-dim input is mapping to a single undifferentiated state.")

if __name__ == "__main__":
    discover_regimes()
