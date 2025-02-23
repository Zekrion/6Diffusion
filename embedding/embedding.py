import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class IPv6Embedding(nn.Module):
    """
    Handles:
      1) nibble embedding (64-d),
      2) positional embedding (64-d),
      3) time embedding (64-d),
      4) up-projection to 512-d
    so that we end with [B, 32, 512].
    """
    def __init__(self, nibble_dim=64, seq_len=32, d_model=512):
        super().__init__()
        self.seq_len = seq_len
        self.nibble_emb = nn.Linear(1, nibble_dim)  # e.g. 1 -> 64
        self.pos_emb    = nn.Embedding(seq_len, nibble_dim)
        self.time_emb   = TimeEmbedding(nibble_dim)
        
        # final linear to go from 64 -> 512
        self.up_proj    = nn.Linear(nibble_dim, d_model)

        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, x, t):
        """
        x: [B, 32] float noised IPv6 (each nibble is 1 float).
        t: [B] integer diffusion step.
        returns: [B, 32, 512]
        """
        B, seq_len = x.shape
        # 1) nibble embedding => [B,32,64]
        x = x.unsqueeze(-1)              # [B,32,1]
        x = self.nibble_emb(x)          # => [B,32,64]
        
        # 2) add positional embedding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  
        # positions: shape [1,32]
        # self.pos_emb(positions): shape [1,32,64]
        x = x + self.pos_emb(positions)  # broadcast to [B,32,64]
        
        # 3) add time embedding
        t_e = self.time_emb(t)          # => [B,64]
        t_e = t_e.unsqueeze(1).expand(-1, seq_len, -1)  # => [B,32,64]
        x = x + t_e
        
        # 4) up-project to [B,32,512]
        x = self.up_proj(x)

        x_norm = self.norm1(x)
        
        return x_norm


class TimeEmbedding(nn.Module):
    """
    Encodes a diffusion timestep t -> 64-d vector (via sinusoidal + MLP).
    """
    def __init__(self, embed_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.SiLU(),
            nn.Linear(embed_dim*4, embed_dim)
        )
        # Initialize MLP
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        
        self.embed_dim = embed_dim

    def forward(self, t):
        """
        t: [B] integer
        => [B, embed_dim]
        """
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(0, half, device=t.device).float() / half
        )
        # freqs shape: [half]
        # broadcast with t => shape [B, half]
        freqs = t.unsqueeze(1).float() * freqs.unsqueeze(0)
        emb_sin = freqs.sin()
        emb_cos = freqs.cos()
        emb = torch.cat([emb_sin, emb_cos], dim=1)  # => [B, embed_dim=64]

        emb = self.mlp(emb)
        return emb