# sim/execution.py

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionResult:
    new_position:  object
    realised_pnl:  float
    spread_paid:   float
    slippage:      float
    partial_fill:  float   # 0.0 to 1.0
    executed_price: float = 0.0


class ExecutionEngine:
    """
    Institutional Execution Engine.
    Models slippage, spread, and liquidity impact.
    """

    def __init__(self, config: dict):
        self.base_slippage    = config.get('base_slippage', 0.00005)
        self.impact_factor    = config.get('impact_factor', 0.0002)
        self.partial_fill_min = config.get('partial_fill_min', 0.7)
        self.lot_size         = config.get('lot_size', 100_000)

    def execute(
        self,
        action:      int,
        tick:        object,
        position:    object,
        regime_code: int,
        step_count:  int, # Added for duration audit
    ) -> ExecutionResult:

        if action == 0:   # HOLD
            return self._hold(position, tick)

        order_size = self._get_order_size(action, position)
        direction  = 1.0 if action in (1, 3) else -1.0
        if action in (4, 5): # Close or Flip reverses side
            direction = -1.0 if position.size > 0 else 1.0

        # 1. Spread cost
        spread = getattr(tick, 'spread', 0.0001)
        spread_paid = spread * abs(order_size) * self.lot_size

        # 2. Slippage
        volatility_mult = 1.0 + regime_code * 0.5
        slippage = (
            self.base_slippage
            + self.impact_factor * abs(order_size) * volatility_mult
            + np.random.exponential(self.base_slippage * 0.5)
        )

        # 3. Partial fill
        fill_ratio = 1.0 if action in (4, 5) else self._partial_fill_ratio(order_size, regime_code)

        # 4. Actual executed price
        executed_price = (
            tick.ask + slippage if direction > 0
            else tick.bid - slippage
        )
        filled_size = round((order_size * direction) * fill_ratio, 2)

        # 5. PnL calculation
        realised_pnl = self._calc_pnl(
            position, action, filled_size, executed_price
        )

        new_position = self._update_position(
            position, action, filled_size, executed_price, step_count, tick
        )

        # HARD POSITION GUARD — no code path can exceed this
        MAX_POS = 0.01
        if abs(new_position.size) > MAX_POS:
            new_position.size = MAX_POS if new_position.size > 0 else -MAX_POS

        return ExecutionResult(
            new_position = new_position,
            realised_pnl = realised_pnl,
            spread_paid  = spread_paid,
            slippage     = slippage,
            partial_fill = fill_ratio,
            executed_price = executed_price,
        )

    def _partial_fill_ratio(self, size: float, regime: int) -> float:
        base_fill = 1.0
        if regime >= 4:
            base_fill = np.random.uniform(self.partial_fill_min, 1.0)
        elif regime >= 2:
            base_fill = np.random.uniform(0.85, 1.0)
        return base_fill

    def _get_order_size(self, action: int, position) -> float:
        MAX_POS = 0.01  # Hard cap: 0.01 Lots (1000 Euro exposure)
        cur = abs(position.size)
        if action in (1, 2):    # Open
            return 0.01 if cur < MAX_POS else 0.0
        if action == 3:         # Add
            return 0.01 if cur < MAX_POS else 0.0
        if action == 4:         # Close
            return cur
        if action == 5:         # Flip = close + open opposite at base size
            return cur + 0.01 if position.size != 0 else 0.01
        return 0.0

    def _hold(self, position, tick) -> ExecutionResult:
        from sim.env import Position
        new_p = Position(
            size         = position.size,
            entry_price  = position.entry_price,
            entry_time   = position.entry_time,
            floating_pnl = position.floating_pnl,
            entry_mid    = getattr(position, 'entry_mid', 0.0),
        )
        if position.size != 0:
            if position.size > 0:
                new_p.floating_pnl = (tick.bid - position.entry_price) * position.size * self.lot_size
            else:
                new_p.floating_pnl = (tick.ask - position.entry_price) * position.size * self.lot_size
        return ExecutionResult(new_p, 0.0, 0.0, 0.0, 1.0, 0.0)

    def _calc_pnl(self, position, action, filled_size, exec_price) -> float:
        if action in (4, 5) and position.size != 0:
            # PnL based on actually closed size (filled_size)
            # filled_size is negative for closing long, positive for closing short
            return (exec_price - position.entry_price) * (-filled_size) * self.lot_size
        return 0.0

    def _update_position(self, position, action, filled_size, exec_price, step, tick):
        from sim.env import Position
        p = Position(
            size         = position.size,
            entry_price  = position.entry_price,
            entry_time   = position.entry_time,
            floating_pnl = 0.0,
            entry_mid    = getattr(position, 'entry_mid', 0.0),
        )
        
        old_size = p.size
        p.size += filled_size
        
        # If opening or reversing, update entry_price and entry_time
        if old_size == 0 and p.size != 0:
            p.entry_price = exec_price
            p.entry_time = step
            p.entry_mid = (tick.bid + tick.ask) / 2
        elif (old_size > 0 and p.size < 0) or (old_size < 0 and p.size > 0):
            # Reversed (Flip)
            p.entry_price = exec_price
            p.entry_time = step
            p.entry_mid = (tick.bid + tick.ask) / 2
        elif action == 3:
            # Add to position (Weighted average price) with Safe Division
            if abs(p.size) > 1e-9:
                p.entry_price = (p.entry_price * old_size + exec_price * filled_size) / p.size
            else:
                p.size = 0.0
                p.entry_price = 0.0
        
        # Numerical Sanitization
        if abs(p.size) < 1e-9:
            p.size = 0.0
            p.entry_price = 0.0
            p.entry_time = 0
            p.entry_mid = 0.0
        
        if p.size == 0:
            p.entry_price = 0.0
            p.entry_time = 0
            p.entry_mid = 0.0
            
        return p
