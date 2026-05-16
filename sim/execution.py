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
        if action == 5: # Flip reverses side
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
        fill_ratio = self._partial_fill_ratio(order_size, regime_code)

        # 4. Actual executed price
        executed_price = (
            tick.ask + slippage if direction > 0
            else tick.bid - slippage
        )
        filled_size = (order_size * direction) * fill_ratio

        # 5. PnL calculation
        realised_pnl = self._calc_pnl(
            position, action, filled_size, executed_price
        )

        new_position = self._update_position(
            position, action, filled_size, executed_price, step_count
        )

        return ExecutionResult(
            new_position = new_position,
            realised_pnl = realised_pnl - spread_paid,
            spread_paid  = spread_paid,
            slippage     = slippage,
            partial_fill = fill_ratio,
        )

    def _partial_fill_ratio(self, size: float, regime: int) -> float:
        base_fill = 1.0
        if regime >= 4:
            base_fill = np.random.uniform(self.partial_fill_min, 1.0)
        elif regime >= 2:
            base_fill = np.random.uniform(0.85, 1.0)
        return base_fill

    def _get_order_size(self, action: int, position) -> float:
        if action in (1, 2):    return 0.01   # New position
        if action == 3:         return 0.01   # Add
        if action == 4:         return abs(position.size) # Full Close (updated from 0.5)
        if action == 5:         return 2.0 * abs(position.size) if position.size != 0 else 0.01
        return 0.0

    def _hold(self, position, tick) -> ExecutionResult:
        from sim.env import Position
        new_p = Position(
            size         = position.size,
            entry_price  = position.entry_price,
            entry_time   = position.entry_time,
            floating_pnl = position.floating_pnl,
        )
        if position.size != 0:
            last_p = getattr(tick, 'last', 0)
            if last_p == 0: last_p = (tick.bid + tick.ask) / 2
            new_p.floating_pnl = (last_p - position.entry_price) * position.size * self.lot_size
        return ExecutionResult(new_p, 0.0, 0.0, 0.0, 1.0)

    def _calc_pnl(self, position, action, filled_size, exec_price) -> float:
        if action in (4, 5) and position.size != 0:
            # PnL on the closed portion
            closed_size = abs(position.size) if action == 4 else abs(position.size)
            return (exec_price - position.entry_price) * (position.size if position.size > 0 else position.size) * self.lot_size
        return 0.0

    def _update_position(self, position, action, filled_size, exec_price, step):
        from sim.env import Position
        p = Position(
            size         = position.size,
            entry_price  = position.entry_price,
            entry_time   = position.entry_time,
            floating_pnl = 0.0,
        )
        
        old_size = p.size
        p.size += filled_size
        
        # If opening or reversing, update entry_price and entry_time
        if old_size == 0 and p.size != 0:
            p.entry_price = exec_price
            p.entry_time = step
        elif (old_size > 0 and p.size < 0) or (old_size < 0 and p.size > 0):
            # Reversed (Flip)
            p.entry_price = exec_price
            p.entry_time = step
        elif action == 3:
            # Add to position (Weighted average price)
            if p.size != 0:
                p.entry_price = (p.entry_price * old_size + exec_price * filled_size) / p.size
        
        if p.size == 0:
            p.entry_price = 0.0
            p.entry_time = 0
            
        return p
