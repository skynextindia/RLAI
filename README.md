# Axon Institutional RLAI Engine

![System Status](https://img.shields.io/badge/System-LIVE-green?style=for-the-badge)
![Neural Models](https://img.shields.io/badge/Neural_Ensemble-Transformer_%2B_PPO_%2B_XGBoost-blue?style=for-the-badge)

**Axon RLAI** is an institutional-grade autonomous trading infrastructure designed for the BTCUSDm market on MetaTrader 5. It replaces human-heuristic trading with a pure, unsupervised Reinforcement Learning (RL) feedback loop.

## 🧠 Neural Architecture: The Triple-Helix Ensemble
The engine does not rely on a single model. It uses three distinct neural viewpoints to calculate every move:
1. **Transformer (The Historian)**: Ingests a raw 50-candle sequence (200 hours of 4H data) to detect structural market patterns via Attention heads.
2. **PPO Agent (The Tactician)**: A Proximal Policy Optimization reinforcement learning agent that optimizes for raw equity growth.
3. **XGBoost (The Statistician)**: Acts as a high-confidence probability filter, ensuring the ensemble only attacks when statistical certainty is high.

## 🚀 Key Institutional Features
- **Pure AI Self-Learning**: Zero human indicators. The AI learns directly from raw OHLCV fabric and PnL-based rewards.
- **Time-Decay Reward Logic**: The AI is penalized for holding trades too long, forcing it to discover high-velocity profit setups.
- **Persistent RL Memory**: A custom `.json` persistence layer ensures the AI never "forgets" a trade, even across system restarts or crashes.
- **Dynamic ATR Risk Matrix**: Real-time volatility-adjusted Stop Loss (1.5x ATR) and Take Profit (3.0x ATR).
- **Glassmorphic Web IO**: A high-performance FastAPI/JS dashboard with real-time telemetry and a "Command Control Center" for manual overrides.

## 🛠 Tech Stack
- **Engine**: Python 3.10+ (PyTorch, Gymnasium, Pandas, NumPy, XGBoost)
- **Bridge**: ZeroMQ + MetaTrader 5 Native API
- **Backend**: FastAPI (Async Telemetry & Command IO)
- **Frontend**: Vanilla JS + TailwindCSS (Glassmorphism UI)
- **Database**: SQLite (Trade Audit Logs)

## 📡 Web Terminal IO (Localhost:8000)
- **Real-time Neural Matrix**: Visualize the probability distribution (HOLD/LONG/SHORT) on every tick.
- **Command Overrides**: `CLOSE_ALL`, `PAUSE`, `RESUME`, and `KILL SWITCH` directly from the UI.
- **Lifetime Equity Curve**: Continuous tracking of all MT5 executions.

## 🚦 Getting Started
1. **MT5 Setup**: Install the `ZeroMQ_Bridge.mq5` into your MT5 Experts folder.
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Launch Bridge**: Run the MT5 Expert Advisor on a BTCUSDm chart.
4. **Launch Engine**: `python src/main.py`
5. **Launch Terminal**: `python src/dashboard/server.py`

## 🛤 Implementation Roadmap
- [x] Phase 1: MT5 ZeroMQ Native Bridge
- [x] Phase 2: Neural Ensemble (Transformer + PPO)
- [x] Phase 3: Persistent RL Memory & Time-Decay
- [x] Phase 4: Glassmorphic Web IO Control Center
- [ ] Phase 5: Multi-Asset Expansion (ETH, Gold, Forex)
- [ ] Phase 6: Dockerization & Cloud Deployment

---
**Institutional AI Engine | Designed by Antigravity AI**
