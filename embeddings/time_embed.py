# embeddings/time_embed.py

import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
    def forward(self, t):
        # t is shape (batch,)
        t = t.float().unsqueeze(-1)  # => (batch,1)
        return self.mlp(t)          # => (batch,d_model)