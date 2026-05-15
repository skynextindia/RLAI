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
    Models the difference between textbook trading and real trading.
    This is where most simulators lie and where most agents fail.
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
    ) -> ExecutionResult:

        if action == 0:   # HOLD
            return self._hold(position, tick)

        order_size = self._get_order_size(action, position)
        direction  = 1.0 if action in (1, 3) else -1.0

        # 1. Spread cost (you always pay the spread)
        spread = getattr(tick, 'spread', 0.0001)
        spread_paid = spread * abs(order_size) * self.lot_size

        # 2. Slippage (worsens with order size and volatility regime)
        volatility_mult = 1.0 + regime_code * 0.5   # worse in volatile regimes
        slippage = (
            self.base_slippage
            + self.impact_factor * abs(order_size) * volatility_mult
            + np.random.exponential(self.base_slippage * 0.5)  # random component
        )

        # 3. Partial fill (large orders don't always complete)
        fill_ratio = self._partial_fill_ratio(order_size, regime_code)

        # 4. Actual executed price
        executed_price = (
            tick.ask + slippage if direction > 0
            else tick.bid - slippage
        )
        filled_size = order_size * fill_ratio

        # 5. PnL calculation
        realised_pnl = self._calc_pnl(
            position, action, filled_size, executed_price
        )

        new_position = self._update_position(
            position, action, filled_size, executed_price
        )

        return ExecutionResult(
            new_position = new_position,
            realised_pnl = realised_pnl - spread_paid,
            spread_paid  = spread_paid,
            slippage     = slippage,
            partial_fill = fill_ratio,
        )

    def _partial_fill_ratio(self, size: float, regime: int) -> float:
        """In illiquid or volatile conditions, orders partially fill."""
        base_fill = 1.0
        if regime >= 4:   # illiquid or panic regime
            base_fill = np.random.uniform(self.partial_fill_min, 1.0)
        elif regime >= 2:  # volatile regime
            base_fill = np.random.uniform(0.85, 1.0)
        return base_fill

    def _get_order_size(self, action: int, position) -> float:
        if action in (1, 2):    return 0.01   # standard lot
        if action == 3:         return 0.01   # add to position
        if action == 4:         return abs(position.size) * 0.5
        if action == 5:         return abs(position.size)
        return 0.0

    def _hold(self, position, tick) -> ExecutionResult:
        """No action. Update floating PnL only."""
        from sim.env import Position
        new_p = Position(
            size         = position.size,
            entry_price  = position.entry_price,
            entry_time   = position.entry_time,
            floating_pnl = position.floating_pnl,
        )
        if position.size != 0:
            new_p.floating_pnl = (
                (tick.last - position.entry_price)
                * position.size * self.lot_size
            )
        return ExecutionResult(new_p, 0.0, 0.0, 0.0, 1.0)

    def _calc_pnl(self, position, action, filled_size, exec_price) -> float:
        if action in (4, 5) and position.size != 0:
            return (exec_price - position.entry_price) * filled_size * self.lot_size
        return 0.0

    def _update_position(self, position, action, filled_size, exec_price):
        from sim.env import Position
        p = Position(
            size         = position.size,
            entry_price  = position.entry_price,
            entry_time   = position.entry_time,
            floating_pnl = position.floating_pnl,
        )
        if action == 1:   
            p.size += filled_size
            p.entry_price = exec_price
        elif action == 2: 
            p.size -= filled_size
            p.entry_price = exec_price
        elif action == 3: 
            p.size += filled_size
            # Weighted average entry price?
            # For simplicity, keeping original entry price or just updating
            p.entry_price = (p.entry_price * (p.size - filled_size) + exec_price * filled_size) / p.size if p.size != 0 else exec_price
        elif action == 4: 
            p.size *= 0.5
        elif action == 5: 
            p.size = 0.0
            p.entry_price = 0.0
            p.floating_pnl = 0.0
        return p
