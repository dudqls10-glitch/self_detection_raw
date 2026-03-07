"""
Method B: MLP main + TCN residual.

Forward rule (important):
- x_current is always extracted from last frame in sequence: x_current = x_seq[:, -1, :]
- y_hat = y_main + y_res
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPMain(nn.Module):
    """Shared trunk + per-channel heads (same spirit as ModelB)."""

    def __init__(
        self,
        in_dim: int = 12,
        trunk_hidden: int = 128,
        head_hidden: int = 64,
        out_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, trunk_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_hidden, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1),
            )
            for _ in range(out_dim)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        return torch.cat([head(z) for head in self.heads], dim=1)


class CausalConv1d(nn.Module):
    """1D causal convolution via left padding."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad_left = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad_left, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.act(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.drop(out)
        return x + out


class TCNResidual(nn.Module):
    """Residual stream TCN: (B, T, D) -> (B, out_dim)."""

    def __init__(
        self,
        in_dim: int = 12,
        hidden_channels: int = 64,
        out_dim: int = 8,
        kernel_size: int = 3,
        dilations=(1, 2, 4, 8),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            TCNResidualBlock(hidden_channels, kernel_size=kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])
        self.head = nn.Linear(hidden_channels, out_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, D, T)
        x = x_seq.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x_last = x[:, :, -1]
        return self.head(x_last)


class MLP_TCN_ResidualModel(nn.Module):
    """
    Method B model:
    - main: MLP(x_current)
    - residual: TCN(x_seq)
    - final: y_hat = y_main + y_res
    """

    def __init__(
        self,
        in_dim: int = 12,
        out_dim: int = 8,
        trunk_hidden: int = 128,
        head_hidden: int = 64,
        tcn_hidden: int = 64,
        tcn_kernel: int = 3,
        tcn_dilations=(1, 2, 4, 8),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.main = MLPMain(
            in_dim=in_dim,
            trunk_hidden=trunk_hidden,
            head_hidden=head_hidden,
            out_dim=out_dim,
            dropout=dropout,
        )
        self.residual = TCNResidual(
            in_dim=in_dim,
            hidden_channels=tcn_hidden,
            out_dim=out_dim,
            kernel_size=tcn_kernel,
            dilations=tcn_dilations,
            dropout=dropout,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_seq: torch.Tensor, use_residual: bool = True):
        # Enforce consistency: x_current must be the last frame of x_seq.
        x_current = x_seq[:, -1, :]
        y_main = self.main(x_current)

        if use_residual:
            y_res = self.residual(x_seq)
        else:
            y_res = torch.zeros_like(y_main)

        y_hat = y_main + y_res
        return y_hat, y_res
