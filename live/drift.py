# live/drift.py

import numpy as np
from collections import deque
from scipy.stats import ks_2samp

class DriftDetector:
    """
    Three independent drift signals.
    Any one firing triggers the response protocol.
    """

    def __init__(self, config: dict):
        ref_size  = config.get('reference_window', 5000)
        live_size = config.get('live_window',       500)

        self.reference_buffer = deque(maxlen=ref_size)
        self.live_buffer      = deque(maxlen=live_size)
        self.td_error_buffer  = deque(maxlen=100)
        self.reward_buffer    = deque(maxlen=200)

        self.ks_threshold     = config.get('ks_threshold',     0.18)
        self.td_threshold     = config.get('td_threshold',     0.55)
        self.cusum_threshold  = config.get('cusum_threshold',  8.0)
        self.expected_reward  = config.get('expected_reward',  0.04)

        self._cusum_sum = 0.0
        self.alerts     = []

    def update(self, tick_features: np.ndarray, td_error: float, reward: float):
        """Call on every live tick. Returns list of active alerts."""
        self.live_buffer.append(tick_features)
        self.td_error_buffer.append(td_error)
        self.reward_buffer.append(reward)
        self._cusum_sum = max(0, self._cusum_sum + (self.expected_reward - reward))

        self.alerts = []

        # 1. Feature drift (KS test)
        if len(self.reference_buffer) >= 100 and len(self.live_buffer) >= 50:
            ref  = np.array(self.reference_buffer)[:, 0]   # test first feature
            live = np.array(self.live_buffer)[:, 0]
            ks_stat, _ = ks_2samp(ref, live)
            if ks_stat > self.ks_threshold:
                self.alerts.append({'type': 'FEATURE_DRIFT', 'value': ks_stat})

        # 2. Concept drift (TD error rising)
        if len(self.td_error_buffer) >= 20:
            recent_td = np.mean(list(self.td_error_buffer)[-20:])
            if recent_td > self.td_threshold:
                self.alerts.append({'type': 'CONCEPT_DRIFT', 'value': recent_td})

        # 3. Reward drift (CUSUM)
        if self._cusum_sum > self.cusum_threshold:
            self.alerts.append({'type': 'REWARD_DRIFT', 'value': self._cusum_sum})

        return self.alerts

    def seed_reference(self, historical_features: np.ndarray):
        """Populate reference buffer from training data distribution."""
        for row in historical_features[-len(self.reference_buffer.maxlen):]:
            self.reference_buffer.append(row)
