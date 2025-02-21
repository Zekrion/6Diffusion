import torch
import torch.nn as nn

class WindowedSelfAttention(nn.Module):
    """
    Properly splits into non-overlapping windows before computing attention.
    """
    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        self.window_size = window_size
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)

    def forward(self, x):
        B, S, D = x.shape
        assert S % self.window_size == 0, "Sequence length must be divisible by window size"

        # Reshape into windows
        x = x.view(B, S // self.window_size, self.window_size, D)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.window_size, D)  # merge batch & seq

        # Compute attention within each window
        out, _ = self.mha(x, x, x)

        # Restore shape
        out = out.view(B, S // self.window_size, self.window_size, D)
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
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

        # Global MSA (causal mask)
        self.global_attn = nn.MultiheadAttention(d_model, n_heads_global, batch_first=True, dropout=0.1)

        # Local MSA (window-based)
        self.local_attn = WindowedSelfAttention(d_model, n_heads_local, window_size)

        # LayerNorm before attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Fusion layer
        self.fuse_linear = nn.Linear(2*d_model, d_model)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        B, S, D = x.shape
        x_norm = self.norm1(x)

        causal_mask = torch.triu(torch.full((S, S), float('-inf'), device=x.device), diagonal=1)
        g_out, _ = self.global_attn(x_norm, x_norm, x_norm, attn_mask=causal_mask)
        l_out = self.local_attn(x_norm)

        fused = torch.cat([g_out, l_out], dim=-1)
        fused = self.fuse_linear(fused)

        x = x + fused
        x = self.norm2(x + self.ff(x))

        return x