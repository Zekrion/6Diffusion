import torch
import torch.nn as nn
from embedding.embedding import IPv6Embedding  # Import embedding
from glf_msa_decoder.glf_msa_block import GLFMSABlock     # Import GLF Transformer block

class IPv6Decoder(nn.Module):
    """
    Full pipeline:
    - Embedding -> [B,32,512]
    - 10 Transformer layers (GLFMSABlock)
    - Final projection -> [B,32] (predicted noise)
    """
    def __init__(self, d_model=512, embed_dim=64, num_layers=10):
        super().__init__()
        
        # 1. Embedding layer
        self.embedding = IPv6Embedding(nibble_dim=embed_dim, d_model=d_model)
        
        # 2. Transformer decoder with 10 GLF Blocks
        self.blocks = nn.ModuleList([
            GLFMSABlock(d_model=d_model, window_size=2**(i//2))  # Window size doubles every 2 layers
            for i in range(num_layers)
        ])

        # 3. Final projection to 1 float per nibble
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x, t):
        """
        x: [B,32], float noised IPv6.
        t: [B], diffusion step
        => returns predicted noise [B,32].
        """
        # 1) Embed to [B,32,512]
        x = self.embedding(x, t)

        # 2) Pass through 10 Transformer layers
        for blk in self.blocks:
            x = blk(x)  # [B,32,512]

        # 3) Project to [B,32,1] -> squeeze to [B,32]
        x = self.out_proj(x).squeeze(-1)
        
        return x