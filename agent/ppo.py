import torch
import torch.optim as optim
import numpy as np
import zmq
import json
import time
import os
import gc
from collections import deque

from agent.memory import PrioritizedReplayBuffer


class PPOTrainer:
    """
    Institutional PPO Trainer with Windowed Confirmation Gates.
    """

    def __init__(self, agent, env, config: dict):
        self.agent  = agent
        self.env    = env
        self.config = config
        self.device = config.get('device', 'cpu')
        
        self.memory = PrioritizedReplayBuffer(capacity=50_000)
        
        self.agent  = agent.to(self.device)
        print(f"PPO_DEVICE: {self.device}")
        self.recent_trades = []
        self.current_step  = 0

        # Telemetry
        self.zmq_ctx = zmq.Context()
        self.socket  = self.zmq_ctx.socket(zmq.PUB)
        self.socket.bind("tcp://*:5555")

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, agent.parameters()),
            lr     = config.get('lr', 3e-4),
            eps    = 1e-5,
        )

        self.clip_eps    = config.get('clip_eps',    0.2)
        self.gamma       = config.get('gamma',       0.99)
        self.gae_lambda  = config.get('gae_lambda',  0.95)
        self.ent_coef    = config.get('ent_coef',    0.01)
        self.vf_coef     = config.get('vf_coef',     0.5)
        self.max_grad    = config.get('max_grad',    0.5)
        self.n_steps     = config.get('n_steps',     4096)
        self.n_epochs    = config.get('n_epochs',    10)
        self.batch_size  = config.get('batch_size',  64)

        self.reward_history = deque(maxlen=100)
        self.window_stats = []

    def train(self, total_timesteps: int):
        self.total_timesteps = total_timesteps
        obs, _ = self.env.reset()
        
        # 1. Load weights if existing
        ckpt = "models/ppo_agent_latest.pt"
        if os.path.exists(ckpt):
            self.agent.load_state_dict(torch.load(ckpt, map_location=self.device))
            print("Weights restored. Starting Institutional Confirmation Gate.")

        # 2. Window Configuration (10k per window)
        window_size = 10000
        n_windows = total_timesteps // window_size
        self.current_step = 0
        
        for window_idx in range(n_windows):
            print(f"\n--- WINDOW {window_idx+1}/{n_windows} (Steps {window_idx*window_size}-{(window_idx+1)*window_size}) ---", flush=True)
            
            start_equity = self.env.account_equity
            window_steps = 0
            
            while window_steps < window_size:
                # Rollout
                rollout, obs = self._collect_rollout(obs)
                
                # Update
                advantages, returns = self._compute_gae(rollout)
                metrics = self._update(rollout, advantages, returns)
                
                window_steps += self.n_steps
                
                # Check Circuit Breakers
                if self._check_circuit_breakers():
                    break

            # End of Window Audit
            end_equity = self.env.account_equity
            equity_delta = end_equity - start_equity
            audit = self.env.last_reward_audit
            
            window_report = {
                'window': window_idx + 1,
                'equity_delta': equity_delta,
                'trade_freq': audit.get('trade_freq', 0),
                'sharpe': audit.get('sharpe_global', 0),
                'hold_time': audit.get('holding_duration', {}).get('median', 0),
                'churn': audit.get('churn_ratio', 0)
            }
            self.window_stats.append(window_report)
            
            # Persist Progress
            torch.save(self.agent.state_dict(), ckpt)
            self._save_diagnostic_report(f"diagnostics_window_{window_idx+1}.json", torch.FloatTensor(obs).unsqueeze(0).to(self.device))
            
            print(f"WINDOW_AUDIT: Equity={end_equity:.2f}, Freq={window_report['trade_freq']:.1f}%, Sharpe={window_report['sharpe']:.2f}", flush=True)

        self._print_final_gate_report()

    def _collect_rollout(self, obs_start):
        rollout = {
            'obs': [], 'actions': [], 'log_probs': [],
            'values': [], 'rewards': [], 'dones': [],
            'episode_rewards': [],
        }
        obs = obs_start
        ep_reward = 0.0

        for i in range(self.n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob, value, _ = self.agent.get_action(
                    obs_tensor, threshold=self.config.get('confidence_threshold', 0.0)
                )

            next_obs, reward, done, _, info = self.env.step(action.item())
            
            self.env.last_reward_audit = info.get('reward_audit', {})
            rollout['obs'].append(obs)
            rollout['actions'].append(action.item())
            rollout['log_probs'].append(log_prob.item())
            rollout['values'].append(value.item())
            rollout['rewards'].append(reward)
            rollout['dones'].append(done)

            ep_reward += reward
            obs = next_obs
            self.current_step += 1

            if action != 0 or i % 10 == 0:
                self._broadcast_diagnostics({'entropy': 1.0}, obs_tensor)

            if done:
                rollout['episode_rewards'].append(ep_reward)
                ep_reward = 0.0
                obs, _ = self.env.reset()

        return rollout, obs

    def _compute_gae(self, rollout):
        rewards = np.array(rollout['rewards'])
        values = np.array(rollout['values'])
        dones = np.array(rollout['dones'])
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def _update(self, rollout, advantages, returns) -> dict:
        obs_t = torch.FloatTensor(np.array(rollout['obs'])).to(self.device)
        act_t = torch.LongTensor(rollout['actions']).to(self.device)
        logp_t = torch.FloatTensor(rollout['log_probs']).to(self.device)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        
        for _ in range(self.n_epochs):
            indices = torch.randperm(len(obs_t))
            for start in range(0, len(obs_t), self.batch_size):
                idx = indices[start : start + self.batch_size]
                log_probs, values, entropy = self.agent.evaluate_action(obs_t[idx], act_t[idx])
                ratio = torch.exp(log_probs - logp_t[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (ret_t[idx] - values.squeeze()).pow(2).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad)
                self.optimizer.step()
        return {'loss': loss.item(), 'entropy': entropy.mean().item()}

    def _broadcast_diagnostics(self, losses, obs, heavy=False):
        audit = self.env.last_reward_audit
        
        # Calculate Pulse Frequency (Hz)
        now = time.time()
        dt = now - getattr(self, '_last_broadcast_time', now - 0.1)
        self._last_broadcast_time = now
        fps = 1.0 / (dt + 1e-6)

        msg = {
            'task': 'DIAGNOSTIC',
            'step': self.current_step,
            'total_steps': self.total_timesteps,
            'equity': float(self.env.account_equity),
            'pnl': float(self.env.account_equity - self.env.initial_equity),
            'entropy': float(losses.get('entropy', 1.0)),
            'value_loss': float(losses.get('loss', 0.0)),
            'fps': fps,
            'sharpe_100': audit.get('sharpe_global', 0),
            'win_rate': audit.get('win_rate', 0),
            'last_price': float(self.env.last_tick.get('bid', 0)) if hasattr(self.env, 'last_tick') else 0,
            'pos_size': float(self.env.position.size),
            'pos_pnl': float(self.env.position.floating_pnl),
            'audit': audit
        }
        self.socket.send_string(json.dumps(msg))

    def _save_diagnostic_report(self, name, obs):
        path = os.path.join("diagnostics", name)
        audit = self.env.last_reward_audit
        with open(path, "w") as f:
            json.dump({
                'step': self.current_step,
                'metrics': {'equity': float(self.env.account_equity)},
                'reward_audit': audit
            }, f, indent=4)

    def _check_circuit_breakers(self) -> bool:
        if self.env.account_equity < self.env.initial_equity * 0.70:
            print("HALT: 30%_DRAWDOWN_BREACHED")
            return True
        return False

    def _print_final_gate_report(self):
        print("\n" + "="*50)
        print("CONVERGENCE CONFIRMATION GATE REPORT")
        print("="*50)
        print(f"{'Win':<5} | {'Eq Delta':<10} | {'Freq':<6} | {'Sharpe':<6} | {'Hold':<5} | {'Churn':<5}")
        print("-" * 50)
        for s in self.window_stats:
            print(f"{s['window']:<5} | {s['equity_delta']:>10.2f} | {s['trade_freq']:>5.1f}% | {s['sharpe']:>6.2f} | {s['hold_time']:>5.1f} | {s['churn']:>5.2f}")
        print("="*50)
        final = self.window_stats[-1]
        stable = abs(self.window_stats[-1]['trade_freq'] - self.window_stats[-2]['trade_freq']) < 20 if len(self.window_stats) > 1 else False
        print(f"Pass Criteria: Final Sharpe > 0.5? {'YES' if final['sharpe'] > 0.5 else 'NO'}")
        print(f"Pass Criteria: Stability? {'YES' if stable else 'NO'}")
