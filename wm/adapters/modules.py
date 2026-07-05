from __future__ import annotations

import torch
from torch import nn

from wm.data.grid import GRID_CHANNELS, VIEW_H, VIEW_W


MODALITY_IDS = {"text": 0, "events": 1, "grid": 2, "signal": 3}


class TypePositionEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, modality_count: int = 4):
        super().__init__()
        self.type_embed = nn.Embedding(modality_count, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor, modality_id: int) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)
        types = torch.full((batch, seq_len), modality_id, dtype=torch.long, device=x.device)
        return x + self.pos_embed(pos) + self.type_embed(types)


class TextEventByteEmbedder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(256, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids.long().clamp(0, 255))


class ByteInverseHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class GridPatchEmbed(nn.Module):
    def __init__(self, d_model: int, channels: int = GRID_CHANNELS, height: int = VIEW_H, width: int = VIEW_W):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.proj = nn.Linear(channels * height * width, d_model)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels, height, width = frames.shape
        if (channels, height, width) != (self.channels, self.height, self.width):
            raise ValueError("unexpected grid frame shape")
        return self.proj(frames.reshape(batch, seq_len, -1).float())


class GridInverseHead(nn.Module):
    def __init__(self, d_model: int, channels: int = GRID_CHANNELS, height: int = VIEW_H, width: int = VIEW_W):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.proj = nn.Linear(d_model, channels * height * width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return self.proj(x).reshape(batch, seq_len, self.channels, self.height, self.width)


class SignalFrameEmbed(nn.Module):
    def __init__(self, d_model: int, features: int = 2):
        super().__init__()
        self.features = features
        self.proj = nn.Linear(features, d_model)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[-1] != self.features:
            raise ValueError("unexpected signal frame width")
        return self.proj(frames.float())


class SignalInverseHead(nn.Module):
    def __init__(self, d_model: int, features: int = 2):
        super().__init__()
        self.proj = nn.Linear(d_model, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

