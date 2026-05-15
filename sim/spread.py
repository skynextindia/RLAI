# sim/spread.py

import numpy as np

class SpreadModel:
    """
    Stochastic model for market spread.
    In reality, spreads widen during low liquidity or high volatility.
    """
    def __init__(self, config: dict):
        self.base_spread = config.get('baseline_spread', 0.0001)
        self.vol_factor = config.get('spread_vol_factor', 2.0)

    def get_spread(self, base_tick_spread: float, regime_code: int) -> float:
        """
        Adjusts the raw tick spread based on market regime.
        """
        # Regime 4 (Illiquid) and 5 (Panic) cause major widening
        if regime_code == 5:
            return base_tick_spread * np.random.uniform(5, 15)
        if regime_code == 4:
            return base_tick_spread * np.random.uniform(2, 5)
        
        # Normal regimes might still have some stochastic widening
        return base_tick_spread * np.random.uniform(1.0, 1.2)
