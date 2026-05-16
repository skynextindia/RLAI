# sim/env.py

import gymnasium as gym
import numpy as np
import pandas as pd
import time
from collections import deque
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass

from sim.execution   import ExecutionEngine
from sim.spread      import SpreadModel
from sim.regimes     import RegimeStateMachine
from sim.chaos       import ChaosInjector
from sim.state       import MTFStateBuilder
from data.storage    import TickDataLoader


@dataclass
class Position:
    size:        float   = 0.0    # positive=long, negative=short
    entry_price: float   = 0.0
    entry_time:  int     = 0
    floating_pnl:float   = 0.0


class Tick:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def get(self, key, default):
        return getattr(self, key, default)


class TradingEnvironment(gym.Env):
    """
    Institutional RL Environment for BTCUSDm.
    Includes MTF State Awareness, Realistic Execution, and Regime Tracking.
    """

    SEQUENCE_LEN = 60      
    MTF_FEATURES = 1650     
    MAX_STEPS    = 5000    

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
            low=-np.inf, high=np.inf, shape=(self.MTF_FEATURES,), dtype=np.float32
        )

        self.initial_equity = config.get('initial_equity', 10_000.0)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.account_equity = self.initial_equity
        self.max_equity = self.initial_equity
        self.position = Position()
        self.realised_pnl = 0.0
        self.recent_pnl_log = deque(maxlen=100)
        self.last_reward_audit = {}

        # 1. Load fresh episode
        self.ticks_list = self.loader.sample_episode(
            length=self.MAX_STEPS + self.SEQUENCE_LEN + 1000,
            symbol=self.symbol
        )
        if not self.ticks_list:
            self.ticks_list = [{'price_delta': 0, 'volume_delta': 0, 'spread': 0.0001, 'time_delta_ms': 100, 'ask': 1.1, 'bid': 1.0999, 'last': 1.1, 'volume': 1, 'time': time.time()} for _ in range(7000)]
        
        self.ticks = self.ticks_list
        ticks_df = pd.DataFrame(self.ticks_list)
        self.state_builder = MTFStateBuilder(ticks_df, self.config['simulation'])

        # 2. Pre-process T60 features
        self.tick_features = np.zeros((len(self.ticks), 4), dtype=np.float32)
        for i, t in enumerate(self.ticks):
            self.tick_features[i] = [
                t.get('price_delta', 0) or 0,
                t.get('volume_delta', 0) or 0,
                t.get('spread', 0) or 0,
                (t.get('time_delta_ms', 0) or 100) / 1000.0
            ]

        # 3. Reset Institutional Performance Audit
        self.perf_metrics = {
            'wins': 0, 'total_trades': 0, 
            'pnl_100': deque(maxlen=100),
            'pnl_500': deque(maxlen=500),
            'pnl_global': [],
            'regime_stats': {},
            'transition_stats': {},
            'prev_regime': None,
            'moving_costs': deque(maxlen=100),
            'moving_profit': deque(maxlen=100),
            'action_counts': {0:0, 1:0, 2:0, 3:0, 4:0, 5:0},
            'path_audit': {'close': 0, 'flip': 0, 'hold': 0, 'trade': 0},
            'hold_durations': [],
            'last_trade_step': 0,
            'consecutive_reversals': 0,
            'last_side': 0, # 1 for long, -1 for short
            'trade_metrics': {
                'realized_pnl': [],
                'durations': [],
                'wins': 0,
                'total': 0
            },
            'step_metrics': {
                'rewards': [],
                'floating_pnl': [],
                'pnl_ratios': []
            }
        }
        print(f"ENV_RESET: Step={self.step_count}, Trades={self.perf_metrics['total_trades']}, Equity={self.account_equity}", flush=True)
        
        return self._get_obs(), {}

    def step(self, action: int):
        self.step_count += 1
        idx = self.step_count + self.SEQUENCE_LEN - 1
        tick = self.ticks[idx]
        if isinstance(tick, dict): tick = Tick(**tick)
        
        # Track Action Path & Duration
        current_side = 1 if self.position.size > 0 else (-1 if self.position.size < 0 else 0)
        
        if action == 0: 
            self.perf_metrics['path_audit']['hold'] += 1
        else:
            # Check for Reversals
            if current_side != 0:
                new_side = current_side
                if action == 2 or action == 5: new_side = -1
                if action == 1 or action == 3: new_side = 1
                if action == 4: new_side = 0
                
                if new_side != 0 and new_side != current_side:
                    self.perf_metrics['consecutive_reversals'] += 1
                else:
                    self.perf_metrics['consecutive_reversals'] = 0

            # Churn Tracking
            steps_since_last = self.step_count - self.perf_metrics['last_trade_step']
            if steps_since_last < 10:
                self.perf_metrics['path_audit']['churn_event'] = self.perf_metrics['path_audit'].get('churn_event', 0) + 1
            
            if action == 4: # Close
                self.perf_metrics['path_audit']['close'] += 1
                if self.position.entry_time > 0:
                    self.perf_metrics['hold_durations'].append(self.step_count - self.position.entry_time)
            elif action == 5: # Flip
                self.perf_metrics['path_audit']['flip'] += 1
                if self.position.entry_time > 0:
                    self.perf_metrics['hold_durations'].append(self.step_count - self.position.entry_time)
            else:
                self.perf_metrics['path_audit']['trade'] += 1
            
            self.perf_metrics['last_trade_step'] = self.step_count

        self.perf_metrics['action_counts'][action] += 1

        # Physics
        tick = self.chaos.transform(tick, self.step_count)
        regime_code = self.regime.update(tick)

        # Execution
        execution_result = self.exec.execute(
            action=action, tick=tick, position=self.position, regime_code=regime_code,
            step_count=self.step_count
        )
        self._update_position(execution_result)

        # Reward & Audit
        reward, audit = self._compute_reward_with_audit(execution_result, tick, action)
        
        done = (self.step_count >= self.MAX_STEPS or self.account_equity < self.initial_equity * 0.85)
        
        info = {
            'pnl': self.realised_pnl,
            'regime': regime_code,
            'equity': self.account_equity + self.position.floating_pnl,
            'action': action,
            'reward_audit': audit,
            'session': self._get_session_code(getattr(tick, 'time', 0))
        }

        return self._get_obs(), reward, done, False, info

    def _compute_reward_with_audit(self, exec_result, tick, action):
        r_pnl = exec_result.realised_pnl * 0.40
        
        regime_code = getattr(self.regime, 'current_code', 0)
        prev_regime = self.perf_metrics['prev_regime']
        
        if regime_code not in self.perf_metrics['regime_stats']:
            self.perf_metrics['regime_stats'][regime_code] = {'pnl': [], 'trades': 0}

        if exec_result.realised_pnl != 0:
            self.perf_metrics['total_trades'] += 1
            self.perf_metrics['trade_metrics']['total'] += 1
            self.perf_metrics['trade_metrics']['realized_pnl'].append(exec_result.realised_pnl)
            if exec_result.realised_pnl > 0:
                self.perf_metrics['trade_metrics']['wins'] += 1
                self.perf_metrics['wins'] += 1
            
            # Regime-specific trade tracking
            if regime_code not in self.perf_metrics['regime_stats']:
                self.perf_metrics['regime_stats'][regime_code] = {'pnl': [], 'trades': 0}
            self.perf_metrics['regime_stats'][regime_code]['trades'] += 1
            
            # Transition tracking
            if prev_regime is not None and prev_regime != regime_code:
                trans_key = f"{prev_regime}_to_{regime_code}"
                if trans_key not in self.perf_metrics['transition_stats']:
                    self.perf_metrics['transition_stats'][trans_key] = {'pnl': [], 'trades': 0}
                self.perf_metrics['transition_stats'][trans_key]['trades'] += 1
                self.perf_metrics['transition_stats'][trans_key]['pnl'].append(exec_result.realised_pnl)

        self.perf_metrics['pnl_100'].append(exec_result.realised_pnl)
        self.perf_metrics['pnl_500'].append(exec_result.realised_pnl)
        self.perf_metrics['pnl_global'].append(exec_result.realised_pnl)
        self.perf_metrics['regime_stats'][regime_code]['pnl'].append(exec_result.realised_pnl)
        self.perf_metrics['prev_regime'] = regime_code

        # 2. Bounded Costs with Selective Friction
        vol = getattr(self.regime, 'current_volatility', 0.001)
        vol_factor = np.clip(1.0 / (vol * 1000 + 0.1), 0.5, 5.0)
        
        base_friction = -0.05 * vol_factor if action != 0 else 0
        
        # Selective Penalties
        flip_penalty = -0.15 * vol_factor if action == 5 else 0
        
        # Churn Penalty (Holding < 10 steps)
        churn_penalty = 0
        if action != 0:
            steps_since_last = self.step_count - self.perf_metrics['last_trade_step']
            if steps_since_last < 10:
                churn_penalty = -0.10 * (10 - steps_since_last) / 10.0

        # Stability Tax (Excessive direction changes)
        stability_tax = -0.10 * vol_factor if self.perf_metrics['consecutive_reversals'] > 2 else 0

        # 3. Persistence Reward (Patience)
        patience_reward = 0
        if action == 0 and self.position.size != 0:
            hold_dur = self.step_count - self.position.entry_time
            patience_reward = min(hold_dur * 0.002, 0.03)

        p_cost = (-exec_result.spread_paid * 1.0) + base_friction + flip_penalty + churn_penalty + stability_tax
        
        self.perf_metrics['moving_costs'].append(p_cost)
        self.perf_metrics['moving_profit'].append(r_pnl)

        # 4. Efficiency & Risk
        current_dd = (self.max_equity - self.account_equity) / self.initial_equity
        p_drawdown = -current_dd * 0.30
        
        floating_pnl = self.position.floating_pnl
        sl_penalty = -0.05 if self.position.size != 0 and floating_pnl < -10.0 else 0.0
        tp_bonus = 0.2 if action in [3, 4, 5] and floating_pnl > 5.0 else 0.0

        # --- PHASE 5.3: REWARD ALIGNMENT (v3.6 Blended) ---
        # 1. Blended PnL Signal (Outcome Quality vs Noise)
        curr_total_pnl = self.position.floating_pnl + self.realised_pnl
        prev_total_pnl = getattr(self, '_last_total_pnl', curr_total_pnl)
        delta_floating = curr_total_pnl - prev_total_pnl
        self._last_total_pnl = curr_total_pnl
        
        # Outcome attribution blend (30% Noise / 70% Result)
        pnl_signal = (0.3 * delta_floating) + (0.7 * exec_result.realised_pnl)
        
        pnl_vol = np.std(self.perf_metrics['trade_metrics']['realized_pnl'][-50:]) if len(self.perf_metrics['trade_metrics']['realized_pnl']) > 5 else 0.1
        pnl_norm = np.tanh(pnl_signal / (pnl_vol + 1e-6))
        
        # 2. Component Normalization
        stability_norm = np.clip(p_cost, -1.0, 1.0)
        persistence_norm = np.clip(patience_reward, -0.5, 0.5)
        risk_norm = np.clip(p_drawdown + sl_penalty, -1.0, 0.0)

        decomposition = {
            "pnl": float(0.60 * pnl_norm),
            "stability": float(0.15 * stability_norm),
            "persistence": float(0.10 * persistence_norm),
            "risk": float(0.15 * risk_norm),
            "bonus": float(tp_bonus)
        }
        
        total_abs = sum(abs(v) for v in decomposition.values()) + 1e-9
        self.last_pnl_ratio = abs(decomposition['pnl']) / total_abs
        self.perf_metrics['step_metrics']['pnl_ratios'].append(self.last_pnl_ratio)

        # Gate C: Noise Suppression (Limit floating reward contribution)
        if exec_result.realised_pnl == 0 and abs(decomposition['pnl']) > 0.3 * total_abs:
             decomposition['pnl'] *= 0.5 

        self.last_decomposition = decomposition

        def calc_sharpe(history, n=30):
            if len(history) < n: return 0.0
            arr = np.array(history)
            return np.mean(arr) / (np.std(arr) + 1e-6)

        def get_dur_stat(p):
            if not self.perf_metrics['hold_durations']: return 0
            return np.percentile(self.perf_metrics['hold_durations'], p)

        audit = {
            'step_profit': float(exec_result.realised_pnl),
            'moving_costs': float(np.mean(self.perf_metrics['moving_costs'])),
            'win_rate': self.perf_metrics['wins'] / max(1, self.perf_metrics['total_trades']),
            'trade_freq': (self.perf_metrics['total_trades'] / max(1, self.step_count)) * 100,
            'expectancy': np.mean(self.perf_metrics['trade_metrics']['realized_pnl']) if self.perf_metrics['trade_metrics']['realized_pnl'] else 0.0,
            'holding_duration': {
                'p25': float(get_dur_stat(25)),
                'median': float(get_dur_stat(50)),
                'p75': float(get_dur_stat(75))
            },
            'path': self.perf_metrics['path_audit']
        }
        self.last_reward_audit = audit
        total_reward = sum(decomposition.values())
        return np.clip(total_reward, -20.0, 20.0), audit

    def _get_obs(self):
        idx = self.step_count + self.SEQUENCE_LEN - 1
        mtf = self.state_builder.get_mtf_slice(idx)
        metrics = self.state_builder.get_market_metrics(idx)
        
        if not hasattr(self, '_dim_audit_done'):
            print(f"DIMENSION_AUDIT: MTF={mtf.shape}, Total={mtf.shape[0]+10}", flush=True)
            self._dim_audit_done = True

        hour = self._get_hour(self.ticks[idx].get('time', 0))
        scalars = np.array([
            np.clip(metrics['market_speed'] / 1000.0, -10, 10),
            np.clip(metrics['volatility'] / 100.0, -10, 10),
            np.clip(metrics['imbalance'], -1, 1),
            np.clip(self.position.size * 10.0, -1, 1),
            np.clip(self.position.floating_pnl / 100.0, -20, 20),
            self.regime.current_code / 10.0,
            np.clip((self.account_equity / self.initial_equity) - 1.0, -1, 1),
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            self.exec.base_slippage * 1000.0
        ], dtype=np.float32)

        obs = np.concatenate([mtf, scalars])
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    def _update_position(self, result):
        self.position = result.new_position
        self.account_equity += result.realised_pnl
        self.max_equity = max(self.max_equity, self.account_equity)
        self.realised_pnl += result.realised_pnl
        self.recent_pnl_log.append(result.realised_pnl)

    def _get_hour(self, t):
        try: return time.gmtime(float(t)).tm_hour
        except: return 0

    def _get_session_code(self, t):
        hour = self._get_hour(t)
        if 0 <= hour < 7: return 1
        if 8 <= hour < 12: return 2
        if 13 <= hour < 17: return 3
        return 4
