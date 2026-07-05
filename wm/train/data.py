from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any

import torch

from wm.adapters.byte_tokenizer import ByteTokenizer
from wm.data.schema import read_jsonl


@dataclass
class PreparedSample:
    modality: str
    inputs: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor | None = None


def load_training_rows(data_root: str | Path, split: str, modalities: list[str]) -> list[dict[str, Any]]:
    root = Path(data_root) / "v1" / split
    rows: list[dict[str, Any]] = []
    for modality in modalities:
        rows.extend(read_jsonl(root / f"{modality}.jsonl"))
    return rows


class SequentialSampler:
    def __init__(self, rows: list[dict[str, Any]], seed: int):
        self.rows = list(rows)
        random.Random(seed).shuffle(self.rows)
        self.cursor = 0

    def state_dict(self) -> dict[str, Any]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.cursor = int(state["cursor"])

    def next(self) -> dict[str, Any]:
        row = self.rows[self.cursor % len(self.rows)]
        self.cursor += 1
        return row


def prepare_sample(row: dict[str, Any], max_seq_len: int, device: torch.device) -> PreparedSample:
    modality = row["modality"]
    if modality in {"text", "events"}:
        tok = ByteTokenizer()
        ids = tok.encode(row["tokens"])
        if len(ids) < 2:
            ids = ids + [10]
        ids = ids[: max_seq_len + 1]
        inputs = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        targets = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
        return PreparedSample(modality, inputs, targets)
    if modality == "grid":
        frames = torch.tensor(row["tensor"], dtype=torch.float32, device=device)
        frames = frames[:max_seq_len]
        if frames.shape[0] == 0:
            raise ValueError("empty grid sample")
        mask = (torch.arange(frames.shape[0], device=device) % 4) == 0
        masked = frames.clone()
        masked[mask] = 0.0
        return PreparedSample(modality, masked.unsqueeze(0), frames.unsqueeze(0), mask.unsqueeze(0))
    if modality == "signal":
        frames = torch.tensor(row["tensor"], dtype=torch.float32, device=device)
        frames = frames[: max_seq_len + 1]
        if frames.shape[0] < 2:
            frames = torch.cat([frames, frames[-1:].clone()], dim=0)
        return PreparedSample(modality, frames[:-1].unsqueeze(0), frames[1:].unsqueeze(0))
    raise ValueError(f"unknown modality {modality}")

