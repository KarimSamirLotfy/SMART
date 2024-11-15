import torch
import torch.nn as nn

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization with zero initialization (adaLN-Zero) modulation.

    Applies learnable modulation parameters (shift, scale) to the normalized input based on a conditioning input.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        

    def forward(self, x, c):
        # Generate shift and scale parameters from conditioning input
        shift, scale = self.modulation(c).chunk(2, dim=1)
        # Apply LayerNorm and modulate with shift and scale
        x = self.norm(x)
        return x * (1+scale.unsqueeze(1)) + shift.unsqueeze(1)