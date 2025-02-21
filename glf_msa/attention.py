# glf_msa/attention.py

import torch
import torch.nn as nn

class WindowedMultiheadAttention(nn.Module):
    """
    Local MSA: each position can attend only to neighbors
    in [i - w, i + w].
    """
    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.window_size = window_size

    def forward(self, x):
        B, S, D = x.shape
        attn_mask = torch.zeros((S, S), device=x.device, dtype=torch.bool)
        for i in range(S):
            left  = max(0, i - self.window_size)
            right = min(S, i + self.window_size + 1)
            attn_mask[i, 0:left] = True
            attn_mask[i, right:] = True

        out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        return out


class GLFMSABlock(nn.Module):
    """
    One Transformer block with:
      - Global MSA (2 heads, top-down)
      - Local MSA (2 heads, window-based)
      - Fuse outputs by concatenation -> linear
      - Add & Norm, then FeedForward, then Add & Norm
    """
    def __init__(self, d_model=512, n_heads_global=2, n_heads_local=2, window_size=2):
        super().__init__()

        # "Global" MSA w/ top-down (causal) mask
        self.global_attn = nn.MultiheadAttention(d_model,
                                                 n_heads_global,
                                                 batch_first=True)

        # "Local" MSA
        self.local_attn = WindowedMultiheadAttention(d_model, n_heads_local, window_size)

        # fuse (512 + 512) -> 512
        self.fuse_linear = nn.Linear(2*d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)

        # feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, S, D = x.shape
        # top-down causal mask for global
        causal_mask = torch.ones((S, S), device=x.device, dtype=torch.bool)
        causal_mask = torch.triu(causal_mask, diagonal=1)  # upper triangle => True
        g_out, _ = self.global_attn(x, x, x, attn_mask=causal_mask)

        # local attention
        l_out = self.local_attn(x)

        # fuse
        fused = torch.cat([g_out, l_out], dim=-1)  # (B,S,2D)
        fused = self.fuse_linear(fused)           # (B,S,D)

        x = x + fused
        x = self.norm1(x)

        # feed forward
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm2(x)
        return x
