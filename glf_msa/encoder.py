# glf_msa/encoder.py

import torch
import torch.nn as nn
from .attention import GLFMSABlock

class GLFTransformerEncoder(nn.Module):
    """
    The 10-layer stack with window sizes doubling every 2 layers:
    2,2,4,4,8,8,16,16,32,32
    """
    def __init__(self, d_model=512):
        super().__init__()
        window_sizes = [2,2,4,4,8,8,16,16,32,32]
        self.layers = nn.ModuleList()
        for ws in window_sizes:
            block = GLFMSABlock(d_model=d_model,
                                n_heads_global=2,
                                n_heads_local=2,
                                window_size=ws)
            self.layers.append(block)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x