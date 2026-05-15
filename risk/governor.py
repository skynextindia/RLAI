# risk/governor.py

from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AccountState:
    equity:              float
    daily_start_equity:  float
    open_position_size:  float
    consecutive_losses:  int
    last_trade_time:     Optional[datetime]
    current_spread:      float
    baseline_spread:     float
    realised_vol_1h:     float   # 1-hour realised volatility
    baseline_vol:        float


class RiskGovernor:
    """
    Hard rules. The agent sends actions here first.
    If the governor rejects the action, it becomes HOLD.
    """

    def __init__(self, config: dict):
        self.max_daily_dd       = config.get('max_daily_dd',       -0.02)   # -2%
        self.max_position_pct   = config.get('max_position_pct',    0.01)   # 1% of equity
        self.max_spread_mult    = config.get('max_spread_mult',     3.0)    # 3× baseline
        self.max_vol_mult       = config.get('max_vol_mult',        2.5)    # 2.5× baseline vol
        self.max_consec_losses  = config.get('max_consec_losses',   6)
        self.news_blackout_mins = config.get('news_blackout_mins',  5)
        self.kill_switch_active = False
        self._news_events       = []   # loaded from economic calendar

    def approve(self, action: int, account: AccountState) -> tuple[int, str]:
        """
        Returns (approved_action, reason).
        If action is rejected, returns (0, reason) — HOLD.
        """

        if self.kill_switch_active:
            return 0, "KILL_SWITCH_ACTIVE"

        # Rule 1: Daily drawdown limit
        daily_dd = (account.equity - account.daily_start_equity) / account.daily_start_equity
        if daily_dd <= self.max_daily_dd:
            self.kill_switch_active = True
            logger.critical(f"KILL SWITCH: daily DD {daily_dd:.2%} exceeded limit {self.max_daily_dd:.2%}")
            return 0, f"DAILY_DD_LIMIT: {daily_dd:.2%}"

        # Only check entry rules for new entries (1=BUY, 2=SELL, 3=BUY_MORE)
        if action in (1, 2, 3):

            # Rule 2: Position size limit
            proposed_size = 0.01   # one micro lot
            max_size = (account.equity * self.max_position_pct) / 100_000
            if proposed_size > max_size:
                return 0, f"POSITION_SIZE_LIMIT"

            # Rule 3: Spread filter
            spread_ratio = account.current_spread / (account.baseline_spread + 1e-8)
            if spread_ratio > self.max_spread_mult:
                return 0, f"SPREAD_FILTER: {spread_ratio:.1f}× baseline"

            # Rule 4: Volatility lock
            vol_ratio = account.realised_vol_1h / (account.baseline_vol + 1e-8)
            if vol_ratio > self.max_vol_mult:
                return 0, f"VOL_LOCK: {vol_ratio:.1f}× baseline"

            # Rule 5: Consecutive loss circuit breaker
            if account.consecutive_losses >= self.max_consec_losses:
                return 0, f"CIRCUIT_BREAKER: {account.consecutive_losses} losses"

            # Rule 6: News blackout
            if self._near_news_event():
                return 0, "NEWS_BLACKOUT"

        return action, "APPROVED"

    def _near_news_event(self) -> bool:
        now = datetime.utcnow()
        for event_time in self._news_events:
            delta_mins = abs((now - event_time).total_seconds()) / 60
            if delta_mins < self.news_blackout_mins:
                return True
        return False

    def reset_daily(self):
        self.kill_switch_active = False
        logger.info("Daily risk governor reset")
