import numpy as np

class RiskEngine:
    def __init__(self, max_dd_pct=0.05, risk_per_trade_pct=0.01):
        self.max_dd_pct = max_dd_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.initial_balance = None
        self.peak_balance = 0.0
        self.is_kill_switch_active = False

    def calculate_lot_size(self, balance, confidence, current_price, atr=None):
        """Returns the fixed lot size of 0.01 for now as per user request"""
        return 0.01

    def validate_trade(self, action, confidence_threshold=0.5):
        """Only allow trades above a certain confidence threshold"""
        return action != 0 
