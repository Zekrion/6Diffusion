import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        
        # More sophisticated frequency initialization
        self.register_buffer('freqs', torch.exp(
            -math.log(10000) * 
            torch.linspace(0, 1, embed_dim // 2, dtype=torch.float32))
        )
        
        # Better MLP initialization for SiLU
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim))
        self._init_weights()

    def _init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('silu'))
                nn.init.normal_(layer.bias, std=0.02)

    def forward(self, t):
        """Embeds timesteps using learned sinusoidal projection"""
        # Shape preservation for odd dimensions
        half_dim = self.embed_dim // 2 + self.embed_dim % 2
        
        # Vectorized frequency calculation
        freqs = t.float()[:, None] * self.freqs[None, :half_dim]
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        
        # Handle odd dimensions
        if self.embed_dim % 2:
            emb = emb[:, :-1]
            
        return self.mlp(emb)

class IPv6Embedding(nn.Module):
    def __init__(self, nibble_dim=64, seq_len=32, d_model=512):
        super().__init__()
        
        # Learnable components
        self.nibble_emb = nn.Sequential(
            nn.Linear(1, nibble_dim),
            nn.Tanh()  # Helps bound initial embeddings
        )
        self.pos_emb = nn.Embedding(seq_len, nibble_dim)
        self.time_emb = TimeEmbedding(nibble_dim)
        
        # Projection with residual
        self.up_proj = nn.Linear(nibble_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize position indices once
        self.register_buffer('positions', torch.arange(seq_len))

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.nibble_emb[0].weight)
        nn.init.normal_(self.nibble_emb[0].bias, std=0.01)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.xavier_normal_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, t):
        """Embeds nibbles with position and time context"""
        # Input shape: [B, seq_len]
        B, seq_len = x.shape
        
        # Nibble embedding
        x = self.nibble_emb(x.unsqueeze(-1))  # [B, 32, 64]
        
        # Add position embeddings
        x += self.pos_emb(self.positions)  # Auto-broadcasted
        
        # Add time embeddings with broadcasting
        x += self.time_emb(t).unsqueeze(1)  # [B, 1, 64] -> [B, 32, 64]
        
        # Project and normalize
        return self.norm(self.up_proj(x))