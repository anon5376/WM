from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

from wm.train.data import prepare_sample


def scheduled_eval(trainer: Any, label: str) -> dict[str, Any]:
    trainer.model.eval()
    by_modality: dict[str, list[float]] = {}
    probe_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with torch.no_grad():
        for row in trainer.sampler.rows:
            modality = row["modality"]
            if modality in seen:
                continue
            max_len = int(trainer.config["data"]["max_seq_len"][modality])
            prepared = prepare_sample(row, max_len, trainer.device)
            loss = float(trainer.loss_for(prepared).detach().cpu())
            by_modality.setdefault(modality, []).append(loss)
            probe_rows.append(
                {
                    "step": trainer.step,
                    "label": label,
                    "modality": modality,
                    "probe": "scheduled_loss",
                    "value": loss,
                }
            )
            seen.add(modality)
            if len(seen) == len(trainer.config["data"]["modalities"]):
                break
    metrics = {
        "label": label,
        "step": trainer.step,
        "wall_time_unix": time.time(),
        "loss_by_modality": {name: sum(vals) / len(vals) for name, vals in sorted(by_modality.items())},
    }
    eval_dir = Path(trainer.run_path) / "eval"
    probe_dir = Path(trainer.run_path) / "probes"
    eval_dir.mkdir(parents=True, exist_ok=True)
    probe_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"{label}_step_{trainer.step:06d}.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    with (probe_dir / f"{label}_step_{trainer.step:06d}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "label", "modality", "probe", "value"])
        writer.writeheader()
        writer.writerows(probe_rows)
    return metrics

