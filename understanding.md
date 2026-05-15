# Axon Algo | Understanding the Project v3.1

## 🎯 Project Aim
The core objective of **Axon Algo** is to engineer an institutional-grade, fully autonomous trading infrastructure powered by **Reinforcement Learning (RL)**. 

### Key Goals:
- **Pure RL Execution**: Transitioning away from legacy indicator-based rules to a neural policy that optimizes for PnL and risk-adjusted returns.
- **Neural Transparency**: Providing real-time strategic justifications and an "AI Monologue" for every decision.
- **Institutional Resilience**: Implementing professional risk management, multi-timeframe fractal analysis, and ultra-low latency execution via a custom MT5 ZeroMQ bridge.
- **On-Policy Evolution**: Continuous learning where the agent updates its neural weights with institutional-grade safety guards (KL-Divergence).

---

## 🏗️ Architecture: The Apex Engine v3.1

### 1. Neural Core (The Brain)
Axon uses a **Strategic Ensemble** for decision-making:
- **PPO Agent (Proximal Policy Optimization)**: Primary RL actor-critic model with clipped objectives.
- **Time-Series Transformer**: Analyzes long-range dependencies in H4 price action.
- **XGBoost Classifier**: Provides a "Neural Sentiment Matrix" across fractal layers.

### 2. Market Environment (`MT5TradingEnv`)
- **State Space**: **17-dimensional vector** (Updated). Now includes **Spread Ratio** (Current/Avg) for real-time liquidity awareness.
- **Reward Function (v3.1 Audit)**:
    - **Rolling Sharpe Signal**: Primary reward based on risk-adjusted returns (Window=20).
    - **Friction-Aware**: Proportional spread penalties that scale with market conditions.
    - **Completion Logic**: Discrete bonuses for closed trades based on realized PnL.
    - **Flat-State Soft Penalty**: Replaced legacy time-decay with a soft inactivity signal for stagnant "flat" periods.

### 3. Training & Safety Pipeline
- **Welford Normalization**: Online reward normalization to ensure stable gradients across market regimes.
- **KL-Divergence Guard**: Hard safety limit (threshold = 0.02) that aborts neural updates if the policy shift is too radical, preventing "catastrophic forgetting".
- **Synchronized Feedback**: Reward calculation is fully wired between the MT5 history stream and the PPO buffer.

### 4. Data Pipeline & Sync
- **Ultra-Low Latency Bridge**: ZeroMQ EA streaming raw ticks directly to the Python kernel.
- **LUNID State**: "Live & Synchronized" backfilling of historical data for immediate fractal context on boot.

---

## 📈 Progress Report (Post-Audit v3.1)

### ✅ Completed Milestones:
- **Surgical Audit & Wiring**: Fully disconnected legacy hardcoded rewards and integrated the new institutional reward engine.
- **Liquidity Awareness**: Integrated the `spread_ratio` feature to prevent entries during high-friction periods.
- **Safety Kernel**: Implemented the KL-Divergence guard and Welford normalization in the training loop.
- **Neural Pulse Kernel**: Integrated real-time TPS monitoring to detect high-volatility "neural surges".
- **H4 Institutional Logic**: Locked execution to H4 candles while maintaining 1m tick surveillance.
- **Forensics Reconstruction**: Automated MFE/MAE calculation for post-trade analysis.

### 🔄 In-Progress:
- **Unsupervised Regime Discovery**: Developing an auto-encoder to cluster market states (Trending vs. Mean-Reverting).
- **Dynamic Kelly Optimization**: Refining lot-sizing based on neural confidence and account volatility.

---

## 🛠 Tech Stack
| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **AI Framework** | PyTorch, Gymnasium, XGBoost |
| **Data Streaming** | ZeroMQ (ZMQ) |
| **Database** | SQLite3 (Trade Logs), JSON (Telemetry) |
| **Safety** | KL-Divergence Guard, Welford Normalization |
| **Execution** | MetaTrader 5 (MT5) |

---
> **Status**: *Audit Verified & Synchronized*
> **Audit Checkpoint (SHA256)**:
> - `src/env/trading_env.py`: `21CA5F35C398180E30BF5FEF21A2C1AF7D95E8201C7C9C2C8D8DAC389F96C7AC`
> - `src/main_scalper.py`: `6B442698719CF7A77B8A0375DBBFE979CC17C6795E26E98488993D9369E6B1F6`
> - `src/training/ppo_trainer.py`: `B81AA084C522E52989B09C06AA7EF62E50DD5F34771F879F58645563375545BB`
> **Current Neural Pulse**: *Active*
