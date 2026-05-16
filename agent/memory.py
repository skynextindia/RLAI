
import numpy as np
import torch
from collections import deque
import random

class PrioritizedReplayBuffer:
    """
    Institutional Experience Memory with Priority Sampling.
    Prioritizes 'Crisis' events and rare regime shifts.
    """
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0=uniform, 1=full)
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.pos = 0

    def push(self, state, action, reward, next_state, done, info):
        # Calculate base priority
        # We prioritize: 
        # 1. Large absolute rewards (big wins/losses)
        # 2. Regime changes (if info['regime'] != last_regime)
        # 3. Rare events (flash crashes)
        
        priority = abs(reward) + 1.0
        if info.get('is_rare', False):
            priority *= 5.0
            
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities.append(priority ** self.alpha)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == 0:
            return None
            
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate Importance Sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(weights, dtype=np.float32),
            indices
        )

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = (abs(error) + 1e-6) ** self.alpha

    def __len__(self):
        return len(self.buffer)
