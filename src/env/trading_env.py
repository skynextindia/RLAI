import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MT5TradingEnv(gym.Env):
    def __init__(self, initial_balance=10000.0, commission=0.0001, slippage=0.0001):
        super(MT5TradingEnv, self).__init__()
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        
        # Risk-Adjusted Return State
        self.returns_window = 20
        self.rolling_returns = []
        self.spread_history = []
        self.steps_flat = 0
        self.max_steps_flat = 100
        
        self.reset()
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0  
        self.entry_price = 0.0
        self.last_tick_price = 0.0
        self.last_spread = 0.0
        self.unrealized_pnl = 0.0 # Initialization Fix
        return np.zeros(17), {}

    def step(self, action, current_price, features, lot_size=0.01, contract_size=1.0, tps=0.0, spread=0.0):
        reward = 0.0
        done = False
        info = { 'closed_this_tick': False }
        
        tick_delta = current_price - self.last_tick_price if self.last_tick_price > 0 else 0.0
        self.last_tick_price = current_price
        self.last_spread = spread
        
        # 2. Execution Logic (Compute PnL first)
        price_change = (current_price - self.entry_price) * self.position if self.position != 0 else 0.0
        pnl = price_change * lot_size * contract_size
        is_closed = info.get('closed_this_tick', False)

        if self.position == 0 and action != 0:
            if action == 1: # BUY
                self.position = 1
                self.entry_price = current_price * (1 + self.slippage)
            elif action == 2: # SELL
                self.position = -1
                self.entry_price = current_price * (1 - self.slippage)
        
        # 3. New Reward Function (Surgical Audit Rewrite)
        reward = self._calculate_reward(pnl, action, spread, is_closed)
        
        # 4. Final State Calculation
        self.unrealized_pnl = pnl
        obs = self._extract_state(features, tps=tps, spread=spread, tick_delta=tick_delta)
        
        info.update({
            "balance": self.balance,
            "unrealized_pnl": self.unrealized_pnl,
            "position": self.position,
            "reward_pulse": reward
        })
        
        return obs, reward, done, False, info
        
    def _calculate_reward(self, pnl, action, spread, is_closed):
        reward = 0.0
        
        # 1. Risk-Adjusted Return (Sharpe Component)
        if self.position != 0:
            step_return = pnl / self.initial_balance
            self.rolling_returns.append(step_return)
            if len(self.rolling_returns) > self.returns_window:
                self.rolling_returns.pop(0)
            
            if len(self.rolling_returns) >= 5:
                std = np.std(self.rolling_returns)
                reward = (np.mean(self.rolling_returns) / (std + 1e-6)) * 2.0 
        
        # 2. Trade Completion Bonus
        if is_closed:
            reward += (0.5 if pnl > 0 else -0.2)
        
        # 3. Scaled Overtrading Penalty (Proportional to Spread)
        if self.position == 0 and action != 0:
            friction_cost = (spread * 0.01) / self.initial_balance
            reward -= friction_cost * 5.0
        
        # 4. Inactivity Signal
        if self.position == 0:
            self.steps_flat += 1
            if self.steps_flat > self.max_steps_flat:
                reward -= 0.01
        else:
            self.steps_flat = 0

        # TEMPORARY AUDIT LOG
        print(f"[REWARD] pnl={pnl:.4f} sharpe_n={len(self.rolling_returns)} reward={reward:.4f} is_closed={is_closed}")
            
        return reward

    def _extract_sequence(self, features, timeframe='4h', seq_len=50):
        if features is None or timeframe not in features:
            return np.zeros((seq_len, 16), dtype=np.float32)
        
        df = features[timeframe].fillna(0).replace([np.inf, -np.inf], 0)
        # Standard columns for sequence learning
        cols = ['close', 'volume', 'ATR', 'RSI', 'MACD']
        available_cols = [c for c in cols if c in df.columns]
        data = df[available_cols].tail(seq_len).values
        
        if len(data) < seq_len:
            pad = np.zeros((seq_len - len(data), len(available_cols)))
            data = np.vstack([pad, data])
            
        # Pad width to 16
        if data.shape[1] < 16:
            data = np.pad(data, ((0, 0), (0, 16 - data.shape[1])))
        return data[:, :16].astype(np.float32)

    def _extract_state(self, features, tps=0.0, spread=0.0, tick_delta=0.0):
        if features is None: return np.zeros(16, dtype=np.float32)
        
        # Scrub all features
        for tf in features:
            features[tf] = features[tf].fillna(0).replace([np.inf, -np.inf], 0)

        state_vec = []
        # Core Market Features (1min and 4h)
        for tf in ['1min', '4h']:
            df = features.get(tf)
            if df is not None and not df.empty:
                last = df.iloc[-1]
                state_vec.extend([
                    float(last.get('close', 0) / 80000.0), # Normalize BTC price
                    float(last.get('RSI', 50) / 100.0),
                    float(last.get('ATR', 0) / 1000.0),
                    float(last.get('MACD', 0) / 100.0)
                ])
            else:
                state_vec.extend([0.0, 0.5, 0.0, 0.0])

        # System & PnL Features
        state_vec.extend([
            float(self.position),
            float(self.unrealized_pnl / 100.0),
            float(tps / 20.0),
            float(spread / 100.0),
            float(tick_delta / 10.0),
            float(self.balance / self.initial_balance)
        ])

        # Spread Ratio (Forward-Looking Feature)
        self.spread_history.append(spread)
        if len(self.spread_history) > 20: self.spread_history.pop(0)
        avg_spread = np.mean(self.spread_history) if self.spread_history else spread
        spread_ratio = spread / (avg_spread + 1e-6)
        state_vec.append(float(spread_ratio))
        
        return np.array(state_vec[:17], dtype=np.float32)
