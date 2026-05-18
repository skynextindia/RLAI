
import torch
import torch.nn as nn

class InstitutionalEncoder(nn.Module):
    """
    Simplified Market Encoder for price returns and spread.
    Processes a sequence of 60 ticks of [price_return, spread] (120-dim input).
    """
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        
        self.processor = nn.Sequential(
            nn.Linear(120, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 120)
        latent = self.processor(x)
        return latent

    def get_stream_contributions(self) -> dict:
        return {'price_return': 50.0, 'spread': 50.0}

    def get_state_vector(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
