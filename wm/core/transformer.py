from __future__ import annotations

import torch
from torch import nn


class SharedTransformerCore(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        layers: int = 6,
        heads: int = 4,
        dim_feedforward: int = 3072,
        dropout: float = 0.0,
    ):
        super().__init__()
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(block, num_layers=layers)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
        mask = None
        if causal:
            seq_len = x.shape[1]
            mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device), diagonal=1)
        return self.final_norm(self.net(x, mask=mask))
