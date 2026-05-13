import torch
import torch.nn as nn
import math

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model) * 0.01)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, 3) # 3 actions: HOLD, BUY, SELL

    def forward(self, src):
        # src shape: (batch, seq_len, input_dim)
        src = self.embedding(src)
        src = src + self.pos_encoder[:, :src.size(1), :]
        output = self.transformer_encoder(src)
        return self.decoder(output[:, -1, :]) # Return last sequence element prediction
