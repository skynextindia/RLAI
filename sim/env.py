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
    entry_mid:   float   = 0.0


class Tick:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def get(self, key, default):
        return getattr(self, key, default)


class TradingEnvironment(gym.Env):
    """
    Institutional RL Environment for EurUsdm.
    Includes MTF State Awareness, Realistic Execution, and Regime Tracking.
    """

    SEQUENCE_LEN = 60      
    MTF_FEATURES = 120     
    MAX_STEPS    = 30000    

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.symbol = config.get('sim_symbol', 'EURUSDm')
        self.loader  = TickDataLoader(config['database']['conn_str'])
        self.exec    = ExecutionEngine(config['simulation'])
        self.spread_model  = SpreadModel(config['simulation'])
        self.regime  = RegimeStateMachine(config['simulation'])
        self.chaos   = ChaosInjector(config['simulation'])

        self.action_space = gym.spaces.Discrete(3)
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
        self.recent_trades = []


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
        # 1. Capture exact starting net liquidation equity
        prev_equity = self.account_equity + self.position.floating_pnl

        self.step_count += 1
        idx = self.step_count + self.SEQUENCE_LEN - 1
        tick = self.ticks[idx]
        if isinstance(tick, dict): tick = Tick(**tick)

        # Recalibrate regime code
        tick = self.chaos.transform(tick, self.step_count)
        regime_code = self.regime.update(tick)

        # Exits are environment-controlled (TP/SL/Expiration)
        exec_action = 0  # Default to hold/do nothing
        
        if self.position.size != 0:
            # We are in a position, ignore new entries and check TP/SL/Expiration
            entry = self.position.entry_price
            entry_mid = getattr(self.position, 'entry_mid', 0.0)
            size = self.position.size
            duration = self.step_count - self.position.entry_time
            
            tp_hit = False
            sl_hit = False
            expired = False
            
            # Use symmetric mid-price relative change if entry_mid is available
            if entry_mid > 0:
                mid_current = (tick.bid + tick.ask) / 2
                pnl_pips = (mid_current - entry_mid) if size > 0 else (entry_mid - mid_current)
            else:
                pnl_pips = (tick.bid - entry) if size > 0 else (entry - tick.ask)
                
            if pnl_pips >= 0.00150:
                tp_hit = True
            elif pnl_pips <= -0.00100:
                sl_hit = True
                    
            if duration >= 500:
                expired = True
                
            if tp_hit or sl_hit or expired:
                exec_action = 4 # Force Close Position!
        else:
            # Flat, allow entry signals from the agent
            if action == 1:
                exec_action = 1 # Open Long
            elif action == 2:
                exec_action = 2 # Open Short
            else:
                exec_action = 0 # Stay Flat

        # Audit path counts
        if exec_action == 0:
            self.perf_metrics['path_audit']['hold'] = self.perf_metrics['path_audit'].get('hold', 0) + 1
        elif exec_action == 4:
            self.perf_metrics['path_audit']['close'] = self.perf_metrics['path_audit'].get('close', 0) + 1
            if self.position.entry_time > 0:
                self.perf_metrics['hold_durations'].append(self.step_count - self.position.entry_time)
        else:
            self.perf_metrics['path_audit']['trade'] = self.perf_metrics['path_audit'].get('trade', 0) + 1

        self.perf_metrics['action_counts'][action] = self.perf_metrics['action_counts'].get(action, 0) + 1

        # Execute order
        execution_result = self.exec.execute(
            action=exec_action, tick=tick, position=self.position, regime_code=regime_code,
            step_count=self.step_count
        )
        self._update_position(execution_result)

        # Reward & Audit
        reward, audit = self._compute_reward_with_audit(execution_result, tick, action, prev_equity)
        
        if exec_action in (1, 2, 4):
            self.perf_metrics['last_trade_step'] = self.step_count
            
        net_equity = self.account_equity + self.position.floating_pnl
        done = (self.step_count >= self.MAX_STEPS or net_equity < 9500.0)
        
        info = {
            'pnl': self.realised_pnl,
            'regime': regime_code,
            'equity': net_equity,
            'action': action,
            'reward_audit': audit,
            'session': self._get_session_code(getattr(tick, 'time', 0))
        }

        return self._get_obs(), reward, done, False, info

    def _compute_reward_with_audit(self, exec_result, tick, action, prev_equity):
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
            
            if regime_code not in self.perf_metrics['regime_stats']:
                self.perf_metrics['regime_stats'][regime_code] = {'pnl': [], 'trades': 0}
            self.perf_metrics['regime_stats'][regime_code]['trades'] += 1
            
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

        # Pure PnL / Equity Delta
        curr_equity = self.account_equity + self.position.floating_pnl
        total_reward = curr_equity - prev_equity

        # Churn/hold path audits to keep metrics happy
        if action == 4: # Close
            self.perf_metrics['path_audit']['close'] += 1
            if self.position.entry_time > 0:
                self.perf_metrics['hold_durations'].append(self.step_count - self.position.entry_time)
        elif action == 5: # Flip
            self.perf_metrics['path_audit']['flip'] += 1
            if self.position.entry_time > 0:
                self.perf_metrics['hold_durations'].append(self.step_count - self.position.entry_time)
        elif action != 0:
            self.perf_metrics['path_audit']['trade'] += 1

        self.perf_metrics['moving_costs'].append(float(-exec_result.spread_paid))
        self.perf_metrics['moving_profit'].append(float(exec_result.realised_pnl))

        def get_dur_stat(p):
            if not self.perf_metrics['hold_durations']: return 0
            return np.percentile(self.perf_metrics['hold_durations'], p)

        # Decompose the active step reward
        pnl_val = float(exec_result.realised_pnl) if exec_result.realised_pnl != 0 else float(self.position.floating_pnl)
        fric_val = float(-exec_result.spread_paid - exec_result.slippage * abs(self.position.size) * self.exec.lot_size) if exec_result.realised_pnl != 0 else 0.0
        
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
            'path': self.perf_metrics['path_audit'],
            'decomposition': {
                'pnl_reward': pnl_val,
                'friction_cost': fric_val,
                'net_reward': float(total_reward)
            }
        }
        self.last_reward_audit = audit
        return np.clip(total_reward, -10.0, 10.0), audit

    def _get_obs(self):
        start_idx = self.step_count
        end_idx = self.step_count + self.SEQUENCE_LEN
        
        # Extract sequence of shape (60, 2) consisting of [price_delta, spread]
        seq_features = self.tick_features[start_idx:end_idx, [0, 2]]
        obs = seq_features.flatten() # Shape (120,)
        
        if not hasattr(self, '_dim_audit_done'):
            print(f"DIMENSION_AUDIT: Observation space stripped to minimal features (Price Return + Spread), Shape={obs.shape}", flush=True)
            self._dim_audit_done = True
            
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    def _update_position(self, result):
        if result.realised_pnl != 0:
            idx = self.step_count + self.SEQUENCE_LEN - 1
            tick = self.ticks[idx]
            exit_price = getattr(result, 'executed_price', 0.0)
            if exit_price == 0.0:
                slippage = getattr(result, 'slippage', 0.0)
                if self.position.size > 0:
                    exit_price = tick.get('bid', 0.0) - slippage
                else:
                    exit_price = tick.get('ask', 0.0) + slippage
                
            duration = self.step_count - self.position.entry_time
            if duration >= 500:
                outcome = 'EXPIRED'
            elif result.realised_pnl > 0:
                outcome = 'TP'
            else:
                outcome = 'SL'
                
            tick_time = tick.get('time', 0.0)
            if tick_time > 0:
                time_str = time.strftime('%H:%M:%S', time.localtime(tick_time))
            else:
                time_str = 'N/A'
                
            entry_idx = self.position.entry_time + self.SEQUENCE_LEN - 1
            if entry_idx < len(self.ticks):
                entry_tick = self.ticks[entry_idx]
                entry_time = entry_tick.get('time', 0.0)
            else:
                entry_time = 0.0
                
            if entry_time > 0 and tick_time > entry_time:
                duration_min = round((tick_time - entry_time) / 60.0, 1)
            else:
                duration_min = 0.0
                
            trade_log = {
                'step': int(self.step_count),
                'time': time_str,
                'side': 'LONG' if self.position.size > 0 else 'SHORT',
                'entry': float(self.position.entry_price),
                'exit': float(exit_price),
                'pnl': float(result.realised_pnl),
                'outcome': outcome,
                'duration': int(duration),
                'duration_min': float(duration_min)
            }
            if not hasattr(self, 'recent_trades'):
                self.recent_trades = []
            self.recent_trades.append(trade_log)
            self.recent_trades = self.recent_trades[-1000:]
            
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
        if 22 <= hour or hour < 8:
            return 0  # Sydney & Tokyo Overlap (Pacific/Asian Range)
        if hour == 8:
            return 1  # London/Tokyo Overlap (London Morning Open)
        if 9 <= hour < 13:
            return 2  # Pure London Session
        if 13 <= hour < 17:
            return 3  # London/New York Overlap (High-Volume Merger)
        return 4      # Pure New York Afternoon & Close
