# models/ipv6_denoiser.py

import torch
import torch.nn as nn

from glf_msa.encoder import GLFTransformerEncoder
from embeddings.time_embed import TimeEmbedding

class IPv6Denoiser(nn.Module):
    """
    Takes (batch,32) integer tokens in [0..15] + time-step t,
    outputs (batch,32) floats representing the “denoised” tokens.
    """
    def __init__(self, d_model=512):
        super().__init__()
        self.d_model = d_model

        # nybble -> 64
        self.nybble_embed = nn.Embedding(16, 64)
        # 64 -> 512
        self.proj_in = nn.Linear(64, d_model)

        # 10-layer GLF-Transformer
        self.encoder = GLFTransformerEncoder(d_model=d_model)

        # final projection 512->1 at each of the 32 positions
        self.proj_out = nn.Linear(d_model, 1)

        # time embedding
        self.t_embed = TimeEmbedding(d_model=d_model)

    def forward(self, x_tokens, t):
        """
        x_tokens: (batch,32) in [0..15]
        t       : (batch,) int
        returns : (batch,32) float
        """
        B, S = x_tokens.shape
        # embed discrete tokens
        x = self.nybble_embed(x_tokens)     # (B,32,64)
        x = self.proj_in(x)                # (B,32,512)

        # add time embedding
        t_vec = self.t_embed(t)            # (B,512)
        t_vec = t_vec.unsqueeze(1)         # (B,1,512)
        x = x + t_vec                      # broadcast add

        # pass through 10 layers
        x = self.encoder(x)                # (B,32,512)

        # final projection
        x = self.proj_out(x)               # (B,32,1)
        x = x.squeeze(-1)                  # (B,32)
        return x