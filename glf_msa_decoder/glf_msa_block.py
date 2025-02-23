import torch
import torch.nn as nn
from einops import rearrange

class WindowedSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        self.window_size = window_size
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)

    def forward(self, x):
        B, S, D = x.shape
        assert S % self.window_size == 0, "Sequence length must be divisible by window size"

        # Reshape into windows
        x = x.view(B, S // self.window_size, self.window_size, D)

        x = x.reshape(-1, self.window_size, D)  # merge batch & seq

        # Compute attention within each window
        out, w = self.mha(x, x, x, need_weights=True)

        # Restore shape
        out = out.view(B, S // self.window_size, self.window_size, D)

        out = out.reshape(B, S, D)

        return out
        

class GLFMSABlock(nn.Module):
    def __init__(self, d_model=512, n_heads_global=2, n_heads_local=2, window_size=2):
        super().__init__()
        # Pre-LayerNorm configuration
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Attention modules
        self.global_attn = nn.MultiheadAttention(
            d_model, n_heads_global, 
            batch_first=True, dropout=0.1
        )
        self.local_attn = WindowedSelfAttention(d_model, n_heads_local, window_size)
        
        # Fusion with gating instead of concatenation
        self.fuse_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # Improved FFN with GELU
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(0.1)
        )
        
        # Causal mask cache
        self.register_buffer("causal_mask", None)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Proper initialization for attention layers
        for module in [self.global_attn, self.local_attn.mha]:
            nn.init.xavier_uniform_(module.in_proj_weight)
            nn.init.constant_(module.in_proj_bias, 0.)
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.constant_(module.out_proj.bias, 0.)
            
        # Initialize fusion gate
        nn.init.kaiming_normal_(self.fuse_gate[0].weight)
        
    def _get_causal_mask(self, seq_len, device):
        if self.causal_mask is None or self.causal_mask.size(0) != seq_len:
            self.causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=device),
                diagonal=1
            )
        return self.causal_mask

    def forward(self, x):
        # Pre-LayerNorm setup
        x_norm = self.norm1(x)
        
        # Global attention with cached mask
        seq_len = x.size(1)
        causal_mask = self._get_causal_mask(seq_len, x.device)
        g_out, _ = self.global_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=causal_mask
        )
        
        # Local attention
        l_out = self.local_attn(x_norm)
        
        # Gated fusion instead of concatenation
        gate = self.fuse_gate(x_norm)
        fused = gate * g_out + (1 - gate) * l_out
        
        # Residual connection
        x = x + fused
        
        # Feed-forward with Pre-LN
        x = x + self.ff(self.norm2(x))
        
        return self.norm3(x)