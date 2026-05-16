import torch
import torch.nn as nn
from model.institutional_encoder import InstitutionalEncoder


class TradingActorCritic(nn.Module):
    """
    Institutional PPO Actor-Critic.
    Uses partitioned feature fusion for multi-horizon situational awareness.
    """

    def __init__(
        self,
        latent_dim:  int = 128,
        hidden_dim:  int = 256,
        n_actions:   int = 6,
    ):
        super().__init__()
        self.encoder = InstitutionalEncoder(latent_dim=latent_dim)

        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Orthogonal init for actor (standard PPO practice)
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor):
        # obs: (batch, seq_len, n_features)
        z   = self.encoder.get_state_vector(obs)   # (batch, latent_dim)
        logits = self.actor(z)
        value  = self.critic(z)
        return logits, value

    def get_action(self, obs: torch.Tensor, threshold: float = 0.55):
        logits, value = self.forward(obs)
        probs  = torch.softmax(logits, dim=-1)
        max_prob, max_action = torch.max(probs, dim=-1)
        
        dist   = torch.distributions.Categorical(logits=logits)
        
        # Original Sampling Logic
        action = dist.sample()
        
        # Confidence Gate: If the BEST action is weak, we MUST HOLD.
        # This prevents "forced choices" in noisy states.
        if max_prob.item() < threshold:
            action = torch.zeros_like(action) # Force HOLD
            
        log_prob = dist.log_prob(action)
        return action, log_prob, value, dist.entropy()

    def evaluate_action(self, obs: torch.Tensor, action: torch.Tensor):
        logits, value = self.forward(obs)
        dist     = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return log_prob, value, entropy

    def unfreeze_encoder(self, lr: float = 1e-6):
        """Call after initial PPO convergence for joint fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print(f"Encoder unfrozen. Fine-tuning at lr={lr}")
