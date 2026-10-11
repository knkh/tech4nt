

# tcn_model.py

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Remove future-leaking padding from causal convolutions."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size,
                      padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for windowed OHLCV classification.

    Receptive field with default settings:
        kernel_size=4, 5 layers, dilations=[1,2,4,8,16]
        RF = (4-1) * (1+2+4+8+16) * 2 = 186 > 64-bar window — full coverage.

    Input:  (batch, window_size, num_features)   e.g. (64, 64, 32)
    Output: (batch, num_classes)                 logits, not probabilities
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 3,
        channels: tuple = (32, 64, 64, 64, 64),   # 5 layers, full window coverage
        kernel_size: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_ch = num_features
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i                      # 1,2,4,8,16
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, time, features) — matches windowed dataset shape directly.
        """
        x = x.transpose(1, 2)          # → (batch, features, time)
        y = self.network(x)
        y = y[:, :, -1]                 # last timestep only — causal, no leakage
        return self.classifier(y)