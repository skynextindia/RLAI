# d:\work\axon\RLBOT\src\scratch\test_env_pnl.py
import sys
sys.path.append(".")
import yaml
from sim.env import TradingEnvironment, Tick

def test_short_tp():
    with open("config/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    config['sim_symbol'] = "EURUSDm"
    
    env = TradingEnvironment(config)
    env.reset()
    
    # 1. Force a Short entry
    env.step(2) # Sell (Short) at step_count = 1
    print("Position after Sell (Short):", env.position)
    
    # 2. Modify the tick that will be loaded in the next step to hit TP for Short!
    # A Short position wins when the price goes DOWN.
    next_idx = 2 + env.SEQUENCE_LEN - 1
    next_tick_dict = env.ticks[next_idx].copy()
    next_tick_dict['ask'] = env.position.entry_price - 0.00200 # -20 pips!
    next_tick_dict['bid'] = env.position.entry_price - 0.00208
    env.ticks[next_idx] = next_tick_dict
    
    print("\n--- NEXT STEP: SHOULD HIT TP FOR SHORT ---")
    obs, reward, done, _, info = env.step(0) # Hold, but environment should auto close!
    print("Info:", info)
    print("Position after TP:", env.position)
    print("Realised PnL:", env.realised_pnl)
    print("Account Equity:", env.account_equity)

if __name__ == "__main__":
    test_short_tp()
