from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wm.eval.scheduled import scheduled_eval
from wm.runtime import select_device
from wm.train.config import config_hash, run_dir
from wm.train.data import PreparedSample, SequentialSampler, load_training_rows, prepare_sample
from wm.train.model import MultiModalPredictor, count_parameters


class Trainer:
    def __init__(self, config: dict[str, Any], *, run_path: Path | None = None):
        self.config = config
        self.hash = config_hash(config)
        self.run_path = run_path or run_dir(config)
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.device_info = select_device(config.get("device", "auto"))
        self.device = torch.device(self.device_info.selected)
        self.seed = int(config["seed"])
        self._seed_all(self.seed)
        rows = load_training_rows(config["data"]["root"], config["data"]["split"], config["data"]["modalities"])
        self.sampler = SequentialSampler(rows, self.seed)
        self.model = MultiModalPredictor(config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(config["train"]["lr"]))
        self.step = 0
        self.history: list[dict[str, Any]] = []

    def _seed_all(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)

    def metadata(self) -> dict[str, Any]:
        return {
            "run_id": self.hash,
            "device": self.device_info.__dict__,
            "parameter_count": count_parameters(self.model),
            "config_hash": self.hash,
        }

    def save_config(self) -> None:
        import yaml

        (self.run_path / "config.yaml").write_text(yaml.safe_dump(self.config, sort_keys=True))
        (self.run_path / "metadata.json").write_text(json.dumps(self.metadata(), indent=2, sort_keys=True) + "\n")

    def checkpoint_path(self, step: int | None = None) -> Path:
        step = self.step if step is None else step
        return self.run_path / f"checkpoint_step_{step:06d}.pt"

    def latest_checkpoint(self) -> Path | None:
        checkpoints = sorted(self.run_path.glob("checkpoint_step_*.pt"))
        return checkpoints[-1] if checkpoints else None

    def save_checkpoint(self) -> Path:
        path = self.checkpoint_path()
        torch.save(
            {
                "config_hash": self.hash,
                "step": self.step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "sampler": self.sampler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "history": self.history,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if ckpt["config_hash"] != self.hash:
            raise ValueError("checkpoint config hash does not match current config")
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.sampler.load_state_dict(ckpt["sampler"])
        self.step = int(ckpt["step"])
        self.history = list(ckpt.get("history", []))
        torch.set_rng_state(ckpt["torch_rng_state"])
        random.setstate(ckpt["python_rng_state"])
        np.random.set_state(ckpt["numpy_rng_state"])

    def loss_for(self, prepared: PreparedSample) -> torch.Tensor:
        if prepared.modality in {"text", "events"}:
            logits = self.model.forward_textlike(prepared.inputs, prepared.modality)
            return F.cross_entropy(logits.reshape(-1, 256), prepared.targets.reshape(-1))
        if prepared.modality == "grid":
            pred = self.model.forward_grid(prepared.inputs)
            mask = prepared.mask
            assert mask is not None
            return F.mse_loss(pred[mask], prepared.targets[mask])
        if prepared.modality == "signal":
            pred = self.model.forward_signal(prepared.inputs)
            return F.mse_loss(pred, prepared.targets)
        raise ValueError(prepared.modality)

    def train_step(self) -> dict[str, Any]:
        self.model.train()
        row = self.sampler.next()
        max_len = int(self.config["data"]["max_seq_len"][row["modality"]])
        prepared = prepare_sample(row, max_len, self.device)
        raw_loss = self.loss_for(prepared)
        weight = float(self.config["loss_weights"][prepared.modality])
        loss = raw_loss * weight
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.step += 1
        metric = {
            "step": self.step,
            "modality": prepared.modality,
            "loss": float(raw_loss.detach().cpu()),
            "weighted_loss": float(loss.detach().cpu()),
            "cursor": self.sampler.cursor,
        }
        self.history.append(metric)
        return metric

    def train(self, *, stop_after_steps: int | None = None) -> list[dict[str, Any]]:
        self.save_config()
        max_steps = int(self.config["train"]["max_steps"])
        checkpoint_every = int(self.config["train"]["checkpoint_every_steps"])
        log_every = int(self.config["train"]["log_every_steps"])
        wall_clock_seconds = self.config["train"].get("wall_clock_seconds")
        eval_every_seconds = self.config["train"].get("eval_every_seconds")
        start_time = time.time()
        deadline = None if wall_clock_seconds is None else start_time + float(wall_clock_seconds)
        next_eval_time = start_time if eval_every_seconds is not None else None
        stop_at = max_steps if stop_after_steps is None else min(max_steps, self.step + stop_after_steps)
        metrics_path = self.run_path / "metrics.jsonl"
        while self.step < stop_at:
            if deadline is not None and time.time() >= deadline:
                break
            if next_eval_time is not None and time.time() >= next_eval_time:
                scheduled_eval(self, "scheduled")
                next_eval_time = time.time() + float(eval_every_seconds)
            metric = self.train_step()
            now = time.time()
            metric["wall_time_unix"] = now
            metric["elapsed_seconds"] = now - start_time
            metric["steps_per_second"] = self.step / max(now - start_time, 1e-9)
            if self.step % log_every == 0 or self.step == 1:
                with metrics_path.open("a") as f:
                    f.write(json.dumps(metric, sort_keys=True) + "\n")
            if self.step % checkpoint_every == 0:
                self.save_checkpoint()
        if eval_every_seconds is not None:
            scheduled_eval(self, "final")
        self.save_checkpoint()
        return self.history
