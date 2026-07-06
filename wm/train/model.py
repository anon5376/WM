from __future__ import annotations

import torch
from torch import nn

from wm.adapters.modules import (
    MODALITY_IDS,
    ByteInverseHead,
    GridInverseHead,
    GridPatchEmbed,
    SignalFrameEmbed,
    SignalInverseHead,
    TextEventByteEmbedder,
    TypePositionEmbeddings,
)
from wm.core.transformer import SharedTransformerCore


class MultiModalPredictor(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        d_model = model_cfg["d_model"]
        self.byte_embed = TextEventByteEmbedder(d_model)
        self.grid_embed = GridPatchEmbed(d_model)
        self.signal_embed = SignalFrameEmbed(d_model)
        self.type_pos = TypePositionEmbeddings(d_model, max_len=model_cfg["max_len"])
        self.core = SharedTransformerCore(
            d_model=d_model,
            layers=model_cfg["layers"],
            heads=model_cfg["heads"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg.get("dropout", 0.0),
        )
        self.byte_head = ByteInverseHead(d_model)
        self.grid_head = GridInverseHead(d_model)
        self.signal_head = SignalInverseHead(d_model)

    def forward_textlike(self, tokens: torch.Tensor, name: str, *, causal: bool = False) -> torch.Tensor:
        x = self.byte_embed(tokens)
        x = self.type_pos(x, MODALITY_IDS[name])
        return self.byte_head(self.core(x, causal=causal))

    def forward_grid(self, frames: torch.Tensor) -> torch.Tensor:
        x = self.grid_embed(frames)
        x = self.type_pos(x, MODALITY_IDS["grid"])
        return self.grid_head(self.core(x))

    def forward_signal(self, frames: torch.Tensor) -> torch.Tensor:
        x = self.signal_embed(frames)
        x = self.type_pos(x, MODALITY_IDS["signal"])
        return self.signal_head(self.core(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
