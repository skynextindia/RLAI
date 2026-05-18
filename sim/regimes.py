# sim/regimes.py

import numpy as np
from collections import deque

class RegimeStateMachine:
    """
    Tracks which market regime we're currently in.
    """

    def __init__(self, config: dict):
        self.window        = deque(maxlen=config.get('regime_window', 200))
        self.current_code  = 0
        self._step         = 0

    def update(self, tick) -> int:
        price_delta = getattr(tick, 'price_delta', None)
        if price_delta is not None:
            self.window.append({
                'price_delta': price_delta,
                'spread':      getattr(tick, 'spread', 0.0001),
                'time_delta':  getattr(tick, 'time_delta_ms', 100) or 100,
                'volume':      getattr(tick, 'volume', 1),
            })

        if len(self.window) >= 50 and self._step % 20 == 0:
            self.current_code = self._classify()

        self._step += 1
        return self.current_code

    def _classify(self) -> int:
        deltas   = np.array([w['price_delta'] for w in self.window])
        spreads  = np.array([w['spread']      for w in self.window])
        times    = np.array([w['time_delta']  for w in self.window])

        vol      = np.std(deltas)
        drift    = abs(np.mean(deltas))
        spread_ratio = np.mean(spreads) / (spreads[0] + 1e-8)
        
        if vol > 0.000150:       return 5   # panic
        if spread_ratio > 2.0:   return 4   # illiquid
        if vol > 0.000100:       return 2   # volatile
        if drift > 0.000015:     return 0   # trending
        if vol < 0.000060:       return 3   # accumulation
        return 1                            # ranging
