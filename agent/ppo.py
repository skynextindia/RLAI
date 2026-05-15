# agent/ppo.py

import torch
import torch.optim as optim
import numpy as np
import zmq
import json
import time
import os
import gc
from collections import deque

class PPOTrainer:
    """
    Standard PPO with clipped objective.
    Tuned for trading environments with sparse, noisy rewards.
    """

    def __init__(self, agent, env, config: dict):
        self.agent  = agent
        self.env    = env
        self.config = config
        self.device = config.get('device', 'cpu')
        if self.device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        
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

        # PPO hyperparameters
        self.clip_eps    = config.get('clip_eps',    0.2)
        self.gamma       = config.get('gamma',       0.99)
        self.gae_lambda  = config.get('gae_lambda',  0.95)
        self.ent_coef    = config.get('ent_coef',    0.01)
        self.vf_coef     = config.get('vf_coef',     0.5)
        self.max_grad    = config.get('max_grad',    0.5)
        self.n_steps     = config.get('n_steps',     2048)
        self.n_epochs    = config.get('n_epochs',    10)
        self.batch_size  = config.get('batch_size',  64)

        self.reward_history = deque(maxlen=100)

    def train(self, total_timesteps: int):
        # mlflow.set_experiment("ppo_agent")

        # with mlflow.start_run():
        #     mlflow.log_params(self.config)

            self.total_timesteps = total_timesteps
            obs, _      = self.env.reset()
            episode_num = 0

            # Auto-Resume
            ckpt = "models/ppo_agent_latest.pt"
            if os.path.exists(ckpt):
                print(f"LOADING_SAVED_MODEL: {ckpt}")
                self.agent.load_state_dict(torch.load(ckpt))

            while self.current_step < self.total_timesteps:
                self.start_time = time.time()
                # Collect rollout
                self.current_task = "COLLECTING"
                print(f"Collecting rollout ({self.current_step}/{self.total_timesteps})...", flush=True)
                rollout, obs = self._collect_rollout(obs)
                self.current_step += self.n_steps

                # Compute advantages (GAE)
                self.current_task = "OPTIMIZING"
                advantages, returns = self._compute_gae(rollout)

                # PPO update
                print(f"Updating policy...", flush=True)
                losses = self._update(rollout, advantages, returns)

                # Logging
                ep_rewards = rollout['episode_rewards']
                mean_reward = np.mean(ep_rewards) if ep_rewards else 0.0
                if ep_rewards:
                    self.reward_history.append(mean_reward)
                    episode_num += len(ep_rewards)

                print(
                    f"Step {self.current_step:>8,} | "
                    f"Loss P: {losses['policy']:>7.4f} | "
                    f"Loss V: {losses['value']:>7.4f} | "
                    f"Ep Rew: {mean_reward:>8.4f}",
                    flush=True
                )

                # Broadcast telemetry
                telemetry = {
                    'task': 'OPTIMIZING',
                    'total_steps': self.total_timesteps,
                    'timestep': self.current_step,
                    'policy_loss': float(losses['policy']),
                    'value_loss': float(losses['value']),
                    'reward': float(mean_reward),
                    'equity': float(self.env.account_equity),
                    'entropy': float(losses['entropy']),
                    'regime': int(self.env.regime.current_code),
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'fps': float(self.n_steps / (time.time() - self.start_time)),
                    'trades': self.recent_trades,
                    # Neural Reasoning Data
                    'pos_size': float(self.env.position.size),
                    'pos_pnl': float(self.env.position.floating_pnl),
                    'last_price': float(self.env.ticks[self.env.step_count + self.env.SEQUENCE_LEN - 1]['last']) if (self.env.ticks and self.env.step_count + self.env.SEQUENCE_LEN < len(self.env.ticks)) else 0.0,
                    'entropy': getattr(self, 'last_entropy', 0.5)
                }
                self.socket.send_string(json.dumps(telemetry))
                
                # MEMORY RECOVERY: Break data references and trigger GC
                self.recent_trades = []
                del rollout
                del advantages
                del returns
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                # Periodic Saving
                if (self.current_step // self.n_steps) % 5 == 0:
                    os.makedirs("models", exist_ok=True)
                    torch.save(self.agent.state_dict(), "models/ppo_agent_latest.pt")
                    print(f"CHECKPOINT_SAVED: {self.current_step} steps")

    def _collect_rollout(self, obs_start):
        rollout = {
            'obs':      [], 'actions': [], 'log_probs': [],
            'values':   [], 'rewards': [], 'dones':     [],
            'episode_rewards': [],
        }

        obs = obs_start
        ep_reward = 0.0

        for i in range(self.n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, value, _ = self.agent.get_action(obs_tensor)

            next_obs, reward, done, _, info = self.env.step(action.item())

            rollout['obs'].append(obs)
            rollout['actions'].append(action.item())
            rollout['log_probs'].append(log_prob.item())
            rollout['values'].append(value.item())
            rollout['rewards'].append(reward)
            rollout['dones'].append(done)

            ep_reward += reward
            obs = next_obs

            # LIVE TELEMETRY during collection
            if i % 50 == 0:
                try:
                    elapsed = time.time() - self.start_time
                    fps = (i + 1) / elapsed if elapsed > 0.1 else 30.0
                    
                    # Ensure we don't index out of bounds
                    idx = min(self.env.step_count + self.env.SEQUENCE_LEN - 1, len(self.env.ticks) - 1)
                    tick = self.env.ticks[idx] if self.env.ticks else {}
                    price = float(tick.get('last', 0))
                    if price == 0 and tick:
                        price = (float(tick.get('bid', 0)) + float(tick.get('ask', 0))) / 2

                    live_telemetry = {
                        'device': str(self.device),
                        'task': 'COLLECTING',
                        'total_steps': self.total_timesteps,
                        'symbol': getattr(self.env, 'symbol', 'BTCUSDm'),
                        'timestep': self.current_step + i,
                        'equity': float(self.env.account_equity),
                        'pos_size': float(self.env.position.size),
                        'pos_pnl': float(self.env.position.floating_pnl),
                        'last_price': price,
                        'regime': int(self.env.regime.current_code),
                        'reward': float(reward),
                        'fps': float(fps),
                        'value_loss': getattr(self, 'last_v_loss', 0.0),
                        'policy_loss': getattr(self, 'last_p_loss', 0.0),
                        'entropy': getattr(self, 'last_entropy', 1.0),
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'trades': [] # Don't send partial list to avoid flickering
                    }
                    self.socket.send_string(json.dumps(live_telemetry))
                except Exception as e:
                    print(f"TELEMETRY_ERROR: {e}")

            if info['action'] != 0:
                action_names = ["HOLD", "BUY", "SELL", "ADD", "REDUCE", "CLOSE"]
                # Ensure we don't index out of bounds
                idx = min(self.env.step_count + self.env.SEQUENCE_LEN - 1, len(self.env.ticks) - 1)
                tick = self.env.ticks[idx] if self.env.ticks else {}
                price = float(tick.get('last', 0))
                if price == 0 and tick:
                    price = (float(tick.get('bid', 0)) + float(tick.get('ask', 0))) / 2

                self.recent_trades.append({
                    "id": int(time.time() * 1000),
                    "type": action_names[info['action']],
                    "pnl": float(info['last_pnl']),
                    "equity": float(info['equity']),
                    "price": price,
                    "size": float(self.env.position.size)
                })

            if done:
                rollout['episode_rewards'].append(ep_reward)
                ep_reward = 0.0
                obs, _ = self.env.reset()

        return rollout, obs

    def _compute_gae(self, rollout):
        rewards    = np.array(rollout['rewards'])
        values     = np.array(rollout['values'])
        dones      = np.array(rollout['dones'])

        advantages = np.zeros_like(rewards)
        last_gae   = 0.0

        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] if t + 1 < len(values) else 0.0
            delta    = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def _update(self, rollout, advantages, returns) -> dict:
        obs_t   = torch.FloatTensor(np.array(rollout['obs'])).to(self.device)
        act_t   = torch.LongTensor(rollout['actions']).to(self.device)
        logp_t  = torch.FloatTensor(rollout['log_probs']).to(self.device)
        adv_t   = torch.FloatTensor(advantages).to(self.device)
        ret_t   = torch.FloatTensor(returns).to(self.device)

        total_policy_loss = total_value_loss = total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = torch.randperm(len(obs_t))

            for start in range(0, len(obs_t), self.batch_size):
                idx = indices[start : start + self.batch_size]

                log_probs, values, entropy = self.agent.evaluate_action(
                    obs_t[idx], act_t[idx]
                )

                ratio        = torch.exp(log_probs - logp_t[idx])
                surr1        = ratio * adv_t[idx]
                surr2        = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[idx]
                policy_loss  = -torch.min(surr1, surr2).mean()
                value_loss   = torch.nn.functional.mse_loss(values.squeeze(), ret_t[idx])
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.mean().item()
                n_updates         += 1

        self.last_p_loss = total_policy_loss / n_updates
        self.last_v_loss = total_value_loss / n_updates
        self.last_entropy = total_entropy / n_updates

        return {
            'policy':  self.last_p_loss,
            'value':   self.last_v_loss,
            'entropy': self.last_entropy,
        }
