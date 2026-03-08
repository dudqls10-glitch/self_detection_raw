import torch
import torch.nn as nn

class ModelBV(nn.Module):
    """
    Model B Version with Velocity support.
    Shared trunk + Per-channel heads.
    """
    def __init__(self, in_dim=18, trunk_hidden=128, head_hidden=64, out_dim=8, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Shared Trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, trunk_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Per-channel Heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_hidden, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1)
            ) for _ in range(out_dim)
        ])
        
    def forward(self, x):
        # x: (B, in_dim)
        feat = self.trunk(x) # (B, trunk_hidden)
        
        outputs = []
        for head in self.heads:
            out = head(feat) # (B, 1)
            outputs.append(out)
            
        # Concatenate along dim 1
        return torch.cat(outputs, dim=1) # (B, out_dim)