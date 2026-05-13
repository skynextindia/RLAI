# AXON ALGO | Institutional Neural Execution v3.0

![Axon Terminal](https://raw.githubusercontent.com/skynextindia/RLAI/main/docs/dashboard_preview.png)

Axon is a high-fidelity, Reinforcement Learning (RL) powered trading infrastructure designed for institutional-grade execution on **BTCUSDm**. It integrates the precision of **Transformer-based time-series analysis** with the transparency of **Smart Money Concepts (SMC)**.

## 🚀 Core Architecture: The Apex Suite v3.0

### 1. Neural Sentiment Matrix
The engine doesn't just read price; it senses **Machine Bias**. Using a multi-timeframe array (1m, 5m, 15m, 1h, 4h), Axon feeds raw sensory data through an **XGBoost Classifier** to determine the neural sentiment across all fractal layers simultaneously.

### 2. H4 Institutional Strategy
Axon is locked to the **H4 Timeframe** for trade execution to eliminate micro-tick noise. 
- **Fractal BOS/CHOCH**: Automatic detection of Break of Structure and Change of Character using 5-bar fractal pivots.
- **Fair Value Gaps (FVG)**: Integrated liquidity void analysis for high-probability entries.

### 3. Machine Strategic Reasoning
Unlike "black box" bots, Axon provides **Neural Transparency**. For every trade and scanning cycle, the AI outputs its **Strategic Justification**—a human-readable breakdown of the neural weights and structural conditions that triggered the action.

### 4. Apex War Room Dashboard
A professional-grade web interface for real-time surveillance:
- **Live Trade Forensics**: Entry, SL, TP, and Neural Reasonings.
- **Tactical Performance Array**: Real-time Win Rate, Profit Factor, and Max Drawdown tracking.
- **AI Monologue**: A live stream of the machine's "internal thoughts" and sensory logs.

## 🛠 Tech Stack
- **Brain**: PyTorch (PPO + Transformer), XGBoost.
- **Sensors**: MT5 ZeroMQ Bridge (Ultra-Low Latency).
- **Execution**: MetaTrader 5 Terminal.
- **Intelligence**: Multi-Timeframe MTFAggregator.
- **Surveillance**: FastAPI + TailwindCSS Dashboard.

## 📦 Installation & Deployment

1. **MetaTrader 5**: Install the `ZeroMQ_Bridge.mq5` Expert Advisor.
2. **Environment**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Execution**:
   - Start MT5 and the ZMQ Bridge.
   - Launch the AI Engine: `python src/main.py`
   - Launch the Command Center: `python src/dashboard/server.py`

## ⚖️ Institutional Risk Engine
Axon utilizes a dynamic **Kelly Criterion**-inspired risk model, automatically adjusting lot sizes based on neural confidence and current account volatility to ensure maximum capital preservation.

---
*Developed by skynextindia for the next generation of institutional neural trading.*
