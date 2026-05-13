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

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.buffer) < 32: # Minimum batch size
            return

        states = torch.stack([t[0] for t in self.buffer])
        actions = torch.tensor([t[1] for t in self.buffer])
        rewards = torch.tensor([t[2] for t in self.buffer])
        next_states = torch.stack([t[3] for t in self.buffer])
        dones = torch.tensor([t[4] for t in self.buffer], dtype=torch.float32)

        # Basic PPO update logic
        for _ in range(5): # 5 epochs per update
            action_probs, values = self.agent(states)
            _, next_values = self.agent(next_states)
            
            # Compute TD targets and advantages
            targets = rewards.unsqueeze(1) + self.gamma * next_values * (1 - dones.unsqueeze(1))
            advantages = (targets - values).detach()
            
            # Actor loss
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            
            # (Simplified PPO loss)
            actor_loss = -(log_probs * advantages.squeeze()).mean()
            critic_loss = F.mse_loss(values, targets.detach())
            
            total_loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
        self.buffer = [] # Clear buffer after update
        print(">>> Model Updated via Continuous Learning Pipeline <<<")
