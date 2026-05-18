# Axon RL Engine: Project Progress & Handover (Phase 4.0 Institutional Calibration)

## 📌 Project Status Summary
We have successfully initiated and stabilized **Institutional Phase 4 Calibration for EURUSDm** running on a high-fidelity 710-dimensional state representation. The system has been fully overhauled to enforce strict standard Stop-Loss/Take-Profit targets and provides state-of-the-art telemetry directly to the React-based WebUI.

---

### ✅ Completed Milestones

1.  **Pure Fixed-Target Risk Model**:
    *   Completely deactivated and removed all breakeven (BE) and trailing stop-loss elements.
    *   Enforced standard **Stop-Loss (10 pips / -0.00100)** and **Take-Profit (15 pips / +0.00150)** limits to maximize the PPO model's clean exploration behavior.
    *   Standardized trade outcome logs to support strict **`TP`**, **`SL`**, or **`EXPIRED`** events.

2.  **Premium WebUI Overhaul**:
    *   **Time Tracking**: Added a absolute market time column (**`TIME`**) in HH:MM:SS format mapping exact tick timestamps.
    *   **Trade Durations in Minutes**: Upgraded outcome holding logs to show actual trade duration in minutes (e.g., `(46.0m)`) with automated legacy tick fallbacks.
    *   **Dynamic Profitability Stages**: Programmed real-time Stage and Window trackers directly into the loading bar matching the neural calibration phases:
        *   `STAGE 1: SURVIVAL & DISCOVERY` (Windows 1-20 | Steps 0 - 200,000)
        *   `STAGE 2: TREND ALIGNMENT` (Windows 21-50 | Steps 200,000 - 500,000)
        *   `STAGE 3: REGIME CONDITIONING` (Windows 51-80 | Steps 500,000 - 800,000)
        *   `STAGE 4: INSTITUTIONAL EXPLOITATION` (Windows 81-100 | Steps 800,000 - 1,000,000)
    *   **Dynamic Gradient flow**: The loading bar now visually pulses with a gradient flowing into the active Stage color (Slate, Gold, Cyan, or Emerald Green).

3.  **Real-Data Pipeline Verification**:
    *   Confirmed the engine runs strictly on high-fidelity historical market ticks loaded from `data/historical/EURUSDm_ticks.parquet` (107 MB), with zero synthetic or mock overrides.

---

### 📂 Key Files
*   **`sim/env.py`**: Environment kernel with pure fixed SL/TP checks and absolute duration-in-minutes calculation.
*   **`dashboard/src/App.jsx`**: Upgraded React frontend rendering time columns, minute holding metrics, stage trackers, and flowing gradient progress bar.
*   **`telemetry_bridge.py`**: ZeroMQ-to-HTTP live streaming server linking python and UI.
