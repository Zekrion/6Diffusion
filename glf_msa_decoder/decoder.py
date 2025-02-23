import torch
import torch.nn as nn
from embedding.embedding import IPv6Embedding  # Import embedding
from glf_msa_decoder.glf_msa_block import GLFMSABlock     # Import GLF Transformer block

class IPv6Decoder(nn.Module):
    """
    Improved decoder architecture with:
    - Adaptive window sizing
    - Proper weight initialization
    - Optimized normalization
    - Enhanced projection layer
    """
    def __init__(self, d_model=512, embed_dim=64, num_layers=10):
        super().__init__()
        
        # 1. Embedding layer with pre-LN
        self.embedding = IPv6Embedding(nibble_dim=embed_dim, d_model=d_model)
        self.embed_ln = nn.LayerNorm(d_model)

        # 2. Transformer blocks with adaptive window sizing
        self.blocks = nn.ModuleList([
            GLFMSABlock(d_model=d_model, window_size=self._calculate_window_size(i, num_layers))
            for i in range(num_layers)
        ])

        # 3. Final projection with residual
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),  # Intermediate dimension
            nn.SiLU(),
            nn.Linear(d_model//2, 1)
        )
        
        self._init_weights()

    def _calculate_window_size(self, layer_idx, total_layers):
        """Adaptive window sizing with max sequence length protection"""
        base_size = 2 ** ((layer_idx // 2) + 1)
        return min(base_size, 32)  # Cap at sequence length

    def _init_weights(self):
        """Proper weight initialization"""
        # Initialize transformer blocks
        for block in self.blocks:
            if hasattr(block, '_init_weights'):
                block._init_weights()
        
        # Initialize final projection layers
        for layer in self.out_proj:
            if isinstance(layer, nn.Linear):  # Only initialize Linear layers
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, t):
        """
        Optimized forward pass:
        x: [B, 32] - Noised IPv6 nibbles
        t: [B] - Diffusion timesteps
        => [B, 32] - Predicted noise
        """
        # Embed with pre-normalization
        x = self.embed_ln(self.embedding(x, t))

        # Process through blocks
        for block in self.blocks:
            x = block(x)

        # Final projection
        return self.out_proj(x).squeeze(-1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)