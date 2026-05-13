import os

class Config:
    ZMQ_PULL_PORT = 5555
    ZMQ_PUSH_PORT = 5556
    TIMEFRAMES = ["M1", "M5", "M15", "H1", "H4"]
    SYMBOLS = ["EURUSD", "BTCUSD"]
    MAX_DRAWDOWN_PCT = 0.05
    RISK_PER_TRADE_PCT = 0.01
    USE_GPU = True
