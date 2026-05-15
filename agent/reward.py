# agent/reward.py

import numpy as np

class RewardFunction:
    """
    Multi-component reward function for trading agents.
    """
    def __init__(self, config: dict):
        self.config = config

    def compute(self, exec_result, account_state, action) -> float:
        # Realised component
        r_pnl = exec_result.realised_pnl

        # Penalties
        p_spread     = -exec_result.spread_paid * 2.0
        p_drawdown   = -max(0, account_state.max_equity - account_state.equity) * 0.5
        p_overtrade  = -0.001 if action != 0 else 0

        # Consistency bonus
        recent_returns = np.array(account_state.recent_pnl_log[-20:])
        b_consistency  = 0.0
        if len(recent_returns) >= 10:
            sharpe_approx = (
                recent_returns.mean() /
                (recent_returns.std() + 1e-8)
            )
            b_consistency = sharpe_approx * 0.01

        return r_pnl + p_spread + p_drawdown + p_overtrade + b_consistency
