"""cldnn_v1: residual Conv1d stack + 2-layer BiLSTM + attention pooling.

Input: (batch, in_channels, T), e.g. I, Q, amplitude, phase with T = 128.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from rml.models.registry import register_model


def _conv_bn(c_in: int, c_out: int, k: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(c_in, c_out, k, padding=k // 2, bias=False), nn.BatchNorm1d(c_out))


class AttentionPool(nn.Module):
    """Additive attention over time: softmax(v^T tanh(W h_t)) weighted sum of h_t."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # h: (B, T, D)
        weights = torch.softmax(self.score(torch.tanh(self.proj(h))).squeeze(-1), dim=1)
        return torch.bmm(weights.unsqueeze(1), h).squeeze(1)


@register_model("cldnn_v1")
class CLDNNv1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_channels: Sequence[int] = (64, 64, 128),
        kernel_sizes: Sequence[int] = (7, 5, 3),
        pool: int = 2,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        attention_hidden: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        c1, c2, c3 = conv_channels
        k1, k2, k3 = kernel_sizes
        if c1 != c2:
            raise ValueError("Residual block requires conv_channels[0] == conv_channels[1]")
        self.conv1 = _conv_bn(in_channels, c1, k1)
        self.conv2 = _conv_bn(c1, c2, k2)  # residual: relu(conv2(a1) + a1)
        self.conv3 = _conv_bn(c2, c3, k3)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.lstm = nn.LSTM(
            c3,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.attention = AttentionPool(2 * lstm_hidden, attention_hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a1 = torch.relu(self.conv1(x))
        a2 = torch.relu(self.conv2(a1) + a1)
        a3 = self.pool(torch.relu(self.conv3(a2)))  # (B, C, T/pool)
        h, _ = self.lstm(a3.transpose(1, 2))  # (B, T/pool, 2H)
        return self.classifier(self.dropout(self.attention(h)))
