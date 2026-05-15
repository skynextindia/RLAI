# sim/chaos.py

import numpy as np
from dataclasses import replace

class ChaosInjector:
    """
    Injects realistic market pathologies into the tick stream.
    The agent must learn to survive these. Not avoid them.
    """

    def __init__(self, config: dict):
        self.flash_crash_prob   = config.get('flash_crash_prob',   0.0001)
        self.stop_hunt_prob     = config.get('stop_hunt_prob',     0.002)
        self.spread_spike_prob  = config.get('spread_spike_prob',  0.005)
        self.gap_prob           = config.get('gap_prob',           0.0005)
        self._crash_active      = 0     # duration counter

    def transform(self, tick, step: int):
        """Apply stochastic market pathologies to the tick."""
        tick = self._maybe_flash_crash(tick, step)
        tick = self._maybe_stop_hunt(tick)
        tick = self._maybe_spread_spike(tick)
        tick = self._maybe_gap(tick)
        return tick

    def _maybe_flash_crash(self, tick, step):
        if self._crash_active > 0:
            self._crash_active -= 1
            crash_move  = np.random.normal(-0.003, 0.001)
            # Use replace if it's a dataclass, otherwise setattr
            if hasattr(tick, '__dataclass_fields__'):
                return replace(tick,
                    ask   = tick.ask * (1 + crash_move),
                    bid   = tick.bid * (1 + crash_move * 1.1),
                    spread= tick.spread * np.random.uniform(3, 8),
                )
            else:
                tick.ask *= (1 + crash_move)
                tick.bid *= (1 + crash_move * 1.1)
                tick.spread *= np.random.uniform(3, 8)
                return tick
        elif np.random.random() < self.flash_crash_prob:
            self._crash_active = np.random.randint(20, 100)
        return tick

    def _maybe_stop_hunt(self, tick):
        if np.random.random() < self.stop_hunt_prob:
            hunt_move = np.random.choice([-1, 1]) * np.random.uniform(0.0005, 0.002)
            if hasattr(tick, '__dataclass_fields__'):
                return replace(tick,
                    ask    = tick.ask * (1 + hunt_move),
                    bid    = tick.bid * (1 + hunt_move),
                    spread = tick.spread * np.random.uniform(2, 5),
                )
            else:
                tick.ask *= (1 + hunt_move)
                tick.bid *= (1 + hunt_move)
                tick.spread *= np.random.uniform(2, 5)
                return tick
        return tick

    def _maybe_spread_spike(self, tick):
        if np.random.random() < self.spread_spike_prob:
            if hasattr(tick, '__dataclass_fields__'):
                return replace(tick, spread=tick.spread * np.random.uniform(3, 15))
            else:
                tick.spread *= np.random.uniform(3, 15)
                return tick
        return tick

    def _maybe_gap(self, tick):
        if np.random.random() < self.gap_prob:
            gap = np.random.choice([-1, 1]) * np.random.uniform(0.001, 0.005)
            if hasattr(tick, '__dataclass_fields__'):
                return replace(tick,
                    ask = tick.ask * (1 + gap),
                    bid = tick.bid * (1 + gap),
                )
            else:
                tick.ask *= (1 + gap)
                tick.bid *= (1 + gap)
                return tick
        return tick
