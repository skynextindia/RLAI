# sim/env.py

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass

from sim.execution   import ExecutionEngine
from sim.spread      import SpreadModel
from sim.regimes     import RegimeStateMachine
from sim.chaos       import ChaosInjector
from data.storage    import TickDataLoader


@dataclass
class Position:
    size:        float   = 0.0    # positive=long, negative=short
    entry_price: float   = 0.0
    entry_time:  int     = 0
    floating_pnl:float   = 0.0


class Tick:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class TradingEnvironment(gym.Env):
    """
    The core training environment.
    This is the gym.Env your PPO agent will train inside.
    """

    SEQUENCE_LEN = 60      # 60 ticks of context fed to encoder
    N_FEATURES   = 8       # features per tick
    MAX_STEPS    = 10_000  # episode length

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.symbol = config.get('sim_symbol', 'BTCUSDm')
        self.loader  = TickDataLoader(config['database']['conn_str'])
        self.exec    = ExecutionEngine(config['simulation'])
        self.spread_model  = SpreadModel(config['simulation'])
        self.regime  = RegimeStateMachine(config['simulation'])
        self.chaos   = ChaosInjector(config['simulation'])

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low  = -np.inf,
            high =  np.inf,
            shape= (self.SEQUENCE_LEN, self.N_FEATURES),
            dtype= np.float32
        )

        self._reset_state()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._reset_state()
        # Load a random episode from historical data
        self.ticks = self.loader.sample_episode(
            length=self.MAX_STEPS + self.SEQUENCE_LEN,
            symbol=self.config.get('sim_symbol', 'EURUSD')
        )
        
        if not self.ticks:
            # Fallback for testing/empty DB
            self.ticks = [{'price_delta': 0, 'volume_delta': 0, 'spread': 0.0001, 'time_delta_ms': 100, 'session': 0, 'ask': 1.1, 'bid': 1.0999, 'last': 1.1, 'volume': 1} for _ in range(self.MAX_STEPS + self.SEQUENCE_LEN)]
        
        # Pre-process ticks into a feature matrix for speed
        self.tick_features = np.zeros((len(self.ticks), 5), dtype=np.float32)
        for i, t in enumerate(self.ticks):
            self.tick_features[i] = [
                t.get('price_delta', 0) or 0,
                t.get('volume_delta', 0) or 0,
                t.get('spread', 0) or 0,
                (t.get('time_delta_ms', 0) or 100) / 1000.0,
                (t.get('session', 0) or 0) / 3.0
            ]
            
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        tick = self.ticks[self.step_count + self.SEQUENCE_LEN - 1]
        
        # Ensure tick is a simple object for the rest of the engine
        if isinstance(tick, dict):
            tick = Tick(**tick)
        
        # Chaos injection (stop hunts, flash crashes, spread spikes)
        tick = self.chaos.transform(tick, self.step_count)

        # Regime update
        regime_code = self.regime.update(tick)

        # Execute action through realistic execution engine
        execution_result = self.exec.execute(
            action      = action,
            tick        = tick,
            position    = self.position,
            regime_code = regime_code,
        )

        # Update position and account state
        self._update_position(execution_result)

        # Update floating PnL for current position based on current price
        if self.position.size != 0:
            current_tick = self.ticks[self.step_count + self.SEQUENCE_LEN]
            last_p = current_tick.get('last', 0)
            if last_p == 0:
                last_p = (current_tick.get('bid', 0) + current_tick.get('ask', 0)) / 2
                
            self.position.floating_pnl = (
                (last_p - self.position.entry_price) 
                * self.position.size * self.exec.lot_size
            )
        else:
            self.position.floating_pnl = 0.0

        # Compute reward
        reward = self._compute_reward(execution_result, tick, action)

        # Episode termination conditions
        done = (
            self.step_count >= self.MAX_STEPS
            or self.account_equity < self.initial_equity * 0.90  # 10% ruin
        )
        if done:
            print(f"Episode Done: Step={self.step_count}, Equity={self.account_equity:.2f}, Ruin={self.account_equity < self.initial_equity * 0.90}", flush=True)

        info = {
            'pnl':          self.realised_pnl,
            'spread':       execution_result.spread_paid,
            'slippage':     execution_result.slippage,
            'regime':       regime_code,
            'equity':       self.account_equity,
            'action':       action,
            'last_pnl':     execution_result.realised_pnl,
        }

        return self._get_obs(), reward, done, False, info

    def _compute_reward(self, exec_result, tick, action) -> float:
        # Realised component (Scaled to 0.0001 per $)
        r_pnl = exec_result.realised_pnl * 0.0001

        # Penalties (Normalized to RL-stable range)
        p_spread     = -exec_result.spread_paid * 0.0002
        p_drawdown   = -max(0, self.max_equity - self.account_equity) * 0.0001
        p_overtrade  = -0.0001 if action != 0 else 0
        
        # Ruin penalty (Major shock for going bust)
        p_ruin = -1.0 if self.account_equity < self.initial_equity * 0.90 else 0.0

        return r_pnl + p_spread + p_drawdown + p_overtrade + p_ruin

    def _get_obs(self) -> np.ndarray:
        """Build the (SEQUENCE_LEN, N_FEATURES) observation tensor."""
        start = self.step_count
        end   = start + self.SEQUENCE_LEN
        
        # Take pre-calculated tick features
        tick_slice = self.tick_features[start:end] # (SEQ, 5)
        
        obs = np.zeros((self.SEQUENCE_LEN, self.N_FEATURES), dtype=np.float32)
        obs[:, :5] = tick_slice
        
        # Add stateful features
        obs[:, 5] = self.position.size
        obs[:, 6] = self.position.floating_pnl
        obs[:, 7] = self.regime.current_code / 5.0
        
        return obs

    def _reset_state(self):
        self.step_count      = 0
        self.position        = Position()
        self.account_equity  = self.config.get('initial_equity', 10_000.0)
        self.initial_equity  = self.account_equity
        self.max_equity      = self.account_equity
        self.realised_pnl    = 0.0
        self.recent_pnl_log  = []
        self.ticks           = []

    def _update_position(self, result):
        self.position = result.new_position
        self.account_equity += result.realised_pnl
        self.max_equity = max(self.max_equity, self.account_equity)
        self.realised_pnl += result.realised_pnl
        self.recent_pnl_log.append(result.realised_pnl)
