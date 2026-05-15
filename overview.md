# Axon Algo | System Overview v3.2 (Audit Verified)

## 🛡️ Core Infrastructure: The Institutional Guard
The Axon Algo has been upgraded with a multi-layered safety and execution kernel, verified through live execution audits.

### 1. Risk-Controlled Execution
*   **Hard SL Guard**: Every order is vetted before submission. Invalid or too-tight Stop Losses are blocked at the engine level.
*   **Precision Kernel**: Automatic price rounding based on broker-specific digits (`symbol_info.digits`) to ensure 100% acceptance for high-precision symbols (BTCUSDm).
*   **Two-Stage Telemetry**: 
    *   `[RISK]`: Pre-flight audit of ATR, SL distance, and TP distance.
    *   `[MT5 VERIFY]`: Post-wire audit of exactly what was sent to the terminal and the broker's return code.

### 2. Neural Reward Architecture
*   **Rolling Sharpe Engine**: The agent is rewarded for risk-adjusted returns, not just raw PnL.
*   **Welford Normalization**: Cold-start seeded with a conservative prior (Mean: 0, Std: 0.01) and windowed to the last 500 rewards for regime reactivity.
*   **Friction Awareness**: Proportional spread penalties scale with market liquidity to prevent overtrading in high-friction environments.

### 3. Neural Policy Ensemble
The system uses a multi-model consensus (Weights TBD / Unvalidated):
*   **PPO Actor-Critic**: On-policy execution logic.
*   **H4 Transformer**: Long-range structural dependency analysis.
*   **XGBoost Sentiment**: Fractal market regime classification.
*   **Decision Logic**: Evaluation occurs on H4 candle closes or during Instant Re-Entry protocols.

---

## 🏗️ Technical Architecture

| Component | Role | Status |
| :--- | :--- | :--- |
| **`main_scalper.py`** | AI Orchestrator & Strategy Engine | **Operational** |
| **`trading_env.py`** | 17-Dim Gymnasium Environment | **Audited** |
| **`ppo_trainer.py`** | RL Training with KL-Divergence Guard | **Audited** |
| **`zmq_client.py`** | Precision Execution Bridge | **Verified** |
| **`server.py`** | Apex War Room Dashboard (FastAPI) | **Live** |

---

## 📊 Audit Checkpoint (SHA256)
Verified local file states as of 2026-05-15:
*   `src/env/trading_env.py`: `8FCB26C9D438ED0E0E98AC34E394F8BBE4E772808B68120F34E1D22AF248BF6F`
*   `src/main_scalper.py`: `1654DAE98DA2C93C576BC2511F5891AFDF7602721C737DFE7A47DBEC58564992`
*   `src/training/ppo_trainer.py`: `B81AA084C522E52989B09C06AA7EF62E50DD5F34771F879F58645563375545BB`
*   `src/mt5_bridge/zmq_client.py`: `1FEA036FA2045589385FF75EEE0FED305B96A2BF87C7A11C86ECC20A3DD0B7D0`

---
> **Status**: *Audit Verified & Institutional-Ready*
> **Current Neural Pulse**: *Active (LUNID State)*
