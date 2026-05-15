# live/validator.py

from dataclasses import dataclass
from typing import Callable

@dataclass
class Gate:
    name:      str
    check:     Callable[[float], bool]
    current:   float = 0.0
    passed:    bool  = False
    required:  bool  = True


class LiveValidator:
    """
    All gates must pass before allocating real capital.
    """

    GATES = [
        Gate("30-day Sharpe ratio",     lambda v: v > 1.5),
        Gate("30-day max drawdown",     lambda v: v > -0.03),
        Gate("Sim-to-live slippage gap",lambda v: v < 0.0002),
        Gate("Drift alerts in 30 days", lambda v: v == 0),
        Gate("Win rate",                lambda v: v > 0.48),
        Gate("Max consecutive losses",  lambda v: v < 6),
        Gate("Daily PnL std deviation", lambda v: v < 0.005),
        Gate("Trade frequency",         lambda v: 1 <= v <= 50),
    ]

    def evaluate(self, metrics: dict) -> tuple[bool, list[str]]:
        failures = []

        for gate in self.GATES:
            value = metrics.get(gate.name)
            if value is None:
                failures.append(f"{gate.name}: NOT MEASURED")
                continue
            gate.current = value
            gate.passed  = gate.check(value)
            if not gate.passed:
                failures.append(f"{gate.name}: {value} (FAILED)")

        all_passed = len(failures) == 0

        if all_passed:
            print("\n✓ ALL GATES PASSED — system approved for micro-lot live trading\n")
        else:
            print("\n✗ GATES FAILED — do NOT go live:\n")
            for f in failures:
                print(f"  ✗ {f}")

        return all_passed, failures
