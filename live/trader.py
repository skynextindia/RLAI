# live/trader.py

import MetaTrader5 as mt5
import time
import logging
import torch
import numpy as np
from risk.governor    import RiskGovernor, AccountState
from live.drift       import DriftDetector
from model.encoder    import MarketEncoder
from agent.actor_critic import TradingActorCritic

logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Production execution engine.
    """

    def __init__(
        self,
        agent:    TradingActorCritic,
        governor: RiskGovernor,
        detector: DriftDetector,
        config:   dict,
    ):
        self.agent    = agent
        self.governor = governor
        self.detector = detector
        self.config   = config
        self.symbol   = config['symbol']
        self.seq_len  = config.get('seq_len', 60)
        self._tick_buffer = []
        self.agent.eval()

    def run(self):
        if not mt5.initialize():
            raise RuntimeError("MT5 failed to initialise")

        logger.info(f"LiveTrader started — {self.symbol} — MICRO LOTS ONLY")
        self.governor.reset_daily()

        while True:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                time.sleep(0.1)
                continue

            self._tick_buffer.append(self._normalise(tick))
            if len(self._tick_buffer) < self.seq_len:
                continue

            self._tick_buffer = self._tick_buffer[-self.seq_len:]

            obs = torch.FloatTensor(
                np.array(self._tick_buffer[-self.seq_len:])
            ).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, value, _ = self.agent.get_action(obs)
                td_error = abs(value.item())

            action_id = action.item()

            alerts = self.detector.update(
                np.array(self._tick_buffer[-1]), td_error, reward=0.0
            )
            if len(alerts) >= 2:
                logger.warning(f"Multiple drift alerts: {alerts}. Switching to HOLD.")
                action_id = 0

            account = self._get_account_state(tick)
            approved_action, reason = self.governor.approve(action_id, account)

            if approved_action != action_id:
                logger.info(f"Action {action_id} blocked: {reason}")

            if approved_action != 0:
                self._execute(approved_action, tick)

            time.sleep(0.1)

    def _execute(self, action: int, tick):
        action_map = {1: mt5.ORDER_TYPE_BUY, 2: mt5.ORDER_TYPE_SELL}
        if action not in action_map:
            return

        request = {
            "action":   mt5.TRADE_ACTION_DEAL,
            "symbol":   self.symbol,
            "volume":   0.01,
            "type":     action_map[action],
            "price":    tick.ask if action == 1 else tick.bid,
            "deviation":10,
            "magic":    20240101,
            "comment":  "rl_agent_v1",
            "type_time":mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment}")

    def _normalise(self, tick) -> list:
        # Should match training normalisation
        return [0.0] * 8

    def _get_account_state(self, tick) -> AccountState:
        info = mt5.account_info()
        return AccountState(
            equity             = info.equity,
            daily_start_equity = info.balance,
            open_position_size = 0.01,
            consecutive_losses = 0,
            last_trade_time    = None,
            current_spread     = tick.ask - tick.bid,
            baseline_spread    = self.config.get('baseline_spread', 0.0001),
            realised_vol_1h    = 0.001,
            baseline_vol       = self.config.get('baseline_vol', 0.0008),
        )
