# AI Trading Infrastructure Architecture

```mermaid
graph TD
    MT5[MetaTrader 5] <-->|ZeroMQ| Bridge[ZMQ Client]
    Bridge --> Data[Data Engine]
    Data --> FE[Feature Engineering\nBOS, CHOCH, FVG, ATR]
    FE --> MTF[Multi-Timeframe Engine\nM1, M5, M15, H1, H4]
    MTF --> Env[Gymnasium Market Simulator]
    Env --> RL[PPO Agent]
    Env --> Seq[Transformer Model]
    Env --> Prob[XGBoost Model]
    RL --> Risk[Risk Engine]
    Seq --> Risk
    Prob --> Risk
    Risk --> Exec[Execution Engine]
    Exec --> Bridge
    Data --> DB[(PostgreSQL TimescaleDB)]
    DB --> Train[Continuous Learning Pipeline]
    Train --> RL
    DB --> Dash[Streamlit Dashboard]
```

## Implementation Roadmap
1. **Phase 1**: MT5 ZeroMQ Bridge & Core Config
2. **Phase 2**: Feature Engineering & MTF Aggregation
3. **Phase 3**: Gymnasium Environment & Simulation
4. **Phase 4**: PyTorch AI Models (Transformer, PPO, XGB)
5. **Phase 5**: Execution & Risk Engines
6. **Phase 6**: Streamlit Dashboard & Dockerization
