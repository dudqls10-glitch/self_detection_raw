"""
Model B: Shared trunk + Per-channel heads

Architecture:
- Trunk (shared): Linear(18, 128) -> ReLU -> Dropout(0.1) -> Linear(128, 128) -> ReLU -> Dropout(0.1)
- Heads (8개, 각 채널 독립): Linear(128, 64) -> ReLU -> Linear(64, 1)
- 8개 head 출력을 concat -> 8D
"""

import torch
import torch.nn as nn


class ModelB(nn.Module):
    """
    Model B: Shared trunk + Per-channel heads
    
    Args:
        in_dim: Input dimension (12 for sin/cos only, joint velocities removed)
        trunk_hidden: Hidden dimension for trunk (default: 128)
        head_hidden: Hidden dimension for each head (default: 64)
        out_dim: Output dimension (8 for raw1..raw8)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        in_dim: int = 12,
        trunk_hidden: int = 128,
        head_hidden: int = 64,
        out_dim: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, trunk_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Per-channel heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_hidden, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1)
            )
            for _ in range(out_dim)
        ])
        
        self.out_dim = out_dim
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, 12) input features [sin(j1..j6), cos(j1..j6)]
            
        Returns:
            (batch, 8) output predictions
        """
        # Shared trunk
        z = self.trunk(x)  # (batch, trunk_hidden)
        
        # Per-channel heads
        outputs = [head(z) for head in self.heads]  # List of (batch, 1)
        
        # Concatenate
        out = torch.cat(outputs, dim=1)  # (batch, 8)
        
        return out


