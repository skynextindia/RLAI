# Axon RL: Institutional Trading Engine

## Current Status: PHASE 5.4 - ALPHA SNIPING
- [x] **PHASE 5.1**: Institutional Foundation (950-dim MTF)
- [x] **PHASE 5.2**: Behavioral Stability (Verified via 50k OOS)
- [x] **PHASE 5.3**: Reward Alignment (PnL Ratio 96.9%)
- [ ] **PHASE 5.4**: Alpha Sniping (10.2M Tick Retraining)
- [ ] **PHASE 6**: Regime Discovery & Live Shadow-Mode

### Recent Breakthroughs
- **PnL Signal Densification**: Transitioned to continuous Floating PnL Delta rewards, increasing PnL signal dominance from **2.4% to 96.9%**.
- **Historical Expansion**: Integrated 10.2 million ticks (Feb-May 2026) for institutional-grade training.
- **OOS Forensic Audit**: Implemented expectancy and profit factor tracking to verify trading edge.
- **Overfitting Guard**: No catastrophic degradation observed on blind datasets.
- **Institutional Patience**: Hold times are controlled; churn is minimized.

**Not Yet Proven**
- **Persistent Positive Alpha**: OOS results currently hover near breakeven ($9,981.69).
- **Generalization**: AlphaSniping policy requires further regime-specific calibration to convert stability into profit.

### 🛠️ Institutional Audit Metrics
| Metric | Status | Details |
| :--- | :--- | :--- |
| **Architecture** | 🟢 100% | Multi-Horizon institutional kernel verified. |
| **Metric Integrity**| 🟢 100% | Reward Decomposition & PnL Ratio tracking active. |
| **Behavioral Sync** | 🟢 98% | Transitioned from scalping to trend-riding. |
| **OOS Stability** | 🟡 80% | No large losses; near-flat equity on blind data. |
| **Alpha Verification**| 🔴 45% | Edge on 2025 BTCUSDm not yet confirmed. |

### ⚠️ Active Investigations
1. **Reward Optimization Drift**: Auditing the `pnl_ratio`. If PnL contribution < 50%, shaping is overpowering the trading signal.
2. **Behavior vs. Profit**: Resolving the mismatch where RL Rewards are positive (+24) but Equity is slightly negative (-0.18%).

## 📈 Next Steps: OOS Protocol v3.3
1. **Extended OOS Test**: Running 50,000 steps on 2025 H1/H2 partitions.
2. **Reward Forensic Audit**: Analyzing the rolling contribution of the `persistence` vs `pnl` components.
3. **Regime Discovery**: implementing auto-encoder for Trending vs Mean-Reverting clustering before Live Readiness.

---
*Last Updated: 2026-05-16*
