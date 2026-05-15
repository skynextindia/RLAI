import torch
import torch.optim as optim
import torch.nn.functional as F

class PPOTrainer:
    def __init__(self, agent, lr=1e-4, gamma=0.99, eps_clip=0.2):
        self.agent = agent
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.buffer = []
        
        # Welford Online Normalization (Cold-Start Seeded v3.1)
        # Prior: mean=0, std=0.01, weight=500 steps
        self.reward_mean = 0.0
        self.reward_m2 = 0.05 # Calculated as (std^2 * count) = (0.01^2 * 500)
        self.reward_count = 500

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.buffer) < 32:
            return

        # 1. Windowed Welford Online Normalization (Last 500 rewards)
        raw_rewards = [t[2] for t in self.buffer]
        for r in raw_rewards:
            if self.reward_count >= 500:
                # Decay old stats to maintain a "moving" average effect
                self.reward_m2 *= 0.99
                self.reward_count = 499
                
            self.reward_count += 1
            delta = r - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = r - self.reward_mean
            self.reward_m2 += delta * delta2
        
        reward_std = (self.reward_m2 / self.reward_count)**0.5 if self.reward_count > 1 else 1.0
        
        # Process Buffer
        states = torch.stack([t[0] for t in self.buffer])
        actions = torch.tensor([t[1] for t in self.buffer])
        next_states = torch.stack([t[3] for t in self.buffer])
        dones = torch.tensor([t[4] for t in self.buffer], dtype=torch.float32)
        
        # Normalize Rewards for this update
        norm_rewards = (torch.tensor(raw_rewards, dtype=torch.float32) - self.reward_mean) / (reward_std + 1e-8)

        # Pre-compute old log probabilities for PPO/KL
        with torch.no_grad():
            old_probs, _ = self.agent(states)
            old_dist = torch.distributions.Categorical(old_probs)
            old_log_probs = old_dist.log_prob(actions)

        # 2. PPO Optimization Loop
        for epoch in range(5):
            action_probs, values = self.agent(states)
            _, next_values = self.agent(next_states)
            
            # KL-Divergence Guard
            dist = torch.distributions.Categorical(action_probs)
            kl_div = torch.distributions.kl_divergence(old_dist, dist).mean()
            if kl_div > 0.02:
                print(f">>> [SAFETY] KL Divergence Limit Exceeded ({kl_div:.4f}). Aborting Update at Epoch {epoch}. <<<")
                break

            # Compute TD targets and advantages
            targets = norm_rewards.unsqueeze(1) + self.gamma * next_values * (1 - dones.unsqueeze(1))
            advantages = (targets - values).detach()
            
            # PPO Actor Loss (Clipped Objective)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            
            surr1 = ratio * advantages.squeeze()
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.squeeze()
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic Loss (MSE)
            critic_loss = F.mse_loss(values, targets.detach())
            
            total_loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
        self.buffer = [] 
        print(f">>> Model Updated (KL: {kl_div:.4f}) | Regime Mean: {self.reward_mean:.4f} <<<")
