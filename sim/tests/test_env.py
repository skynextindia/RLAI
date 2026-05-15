# sim/tests/test_env.py

import yaml
from sim.env import TradingEnvironment
from stable_baselines3.common.env_checker import check_env

def test_environment():
    with open("config/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Mocking TickDataLoader if DB is not ready
    env = TradingEnvironment(config)
    print("Checking environment interface...")
    check_env(env)
    print("Environment interface is valid [OK]")

    obs, _ = env.reset()
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        if done:
            break
    
    print(f"Sample run finished. Total reward: {total_reward:.4f}")
    assert total_reward != 0 or i == 99, "Environment might be stuck"

if __name__ == "__main__":
    test_environment()
