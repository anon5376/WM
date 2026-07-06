from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from wm.adapters.byte_tokenizer import ByteTokenizer
from wm.adapters.modules import MODALITY_IDS
from wm.data.schema import read_jsonl
from wm.train.config import load_config
from wm.train.data import load_training_rows, prepare_sample
from wm.train.trainer import Trainer


def rows_by_modality(data_root: str | Path, split: str, modalities: list[str], version: str = "v1") -> dict[str, list[dict[str, Any]]]:
    root = Path(data_root) / version / split
    return {modality: read_jsonl(root / f"{modality}.jsonl") for modality in modalities}


def byte_unigram(rows: list[dict[str, Any]]) -> np.ndarray:
    tok = ByteTokenizer()
    counts = np.ones(256, dtype=np.float64)
    for row in rows:
        counts[tok.encode(row["tokens"])] += 1
    return counts / counts.sum()


def byte_unigram_loss(row: dict[str, Any], probs: np.ndarray) -> float:
    tok = ByteTokenizer()
    ids = tok.encode(row["tokens"])
    if len(ids) < 2:
        return 0.0
    targets = np.array(ids[1:], dtype=np.int64)
    return float(-np.log(probs[targets]).mean())


def byte_persistence_loss(row: dict[str, Any]) -> float:
    tok = ByteTokenizer()
    ids = tok.encode(row["tokens"])
    if len(ids) < 2:
        return 0.0
    eps = 1e-4
    losses = []
    for prev, target in zip(ids[:-1], ids[1:]):
        prob = 1.0 - eps if prev == target else eps / 255.0
        losses.append(-math.log(prob))
    return float(np.mean(losses))


def grid_frequency_frame(rows: list[dict[str, Any]]) -> torch.Tensor:
    frames = []
    for row in rows:
        frames.extend(row["tensor"])
    return torch.tensor(frames, dtype=torch.float32).mean(dim=0)


def signal_frequency_frame(rows: list[dict[str, Any]]) -> torch.Tensor:
    frames = []
    for row in rows:
        frames.extend(row["tensor"])
    return torch.tensor(frames, dtype=torch.float32).mean(dim=0)


def model_loss(trainer: Trainer, row: dict[str, Any]) -> float:
    max_len = int(trainer.config["data"]["max_seq_len"][row["modality"]])
    prepared = prepare_sample(row, max_len, trainer.device)
    with torch.no_grad():
        return float(trainer.loss_for(prepared).detach().cpu())


def shuffled_model_loss(trainer: Trainer, row: dict[str, Any], seed: int) -> float:
    max_len = int(trainer.config["data"]["max_seq_len"][row["modality"]])
    prepared = prepare_sample(row, max_len, trainer.device)
    gen = torch.Generator().manual_seed(seed)
    seq_len = prepared.inputs.shape[1]
    perm = torch.randperm(seq_len, generator=gen).to(trainer.device)
    prepared.inputs = prepared.inputs[:, perm]
    with torch.no_grad():
        return float(trainer.loss_for(prepared).detach().cpu())


def grid_persistence_loss(row: dict[str, Any]) -> float:
    frames = torch.tensor(row["tensor"], dtype=torch.float32)
    if frames.shape[0] < 2:
        return 0.0
    return float(F.mse_loss(frames[:-1], frames[1:]).cpu())


def signal_persistence_loss(row: dict[str, Any]) -> float:
    frames = torch.tensor(row["tensor"], dtype=torch.float32)
    if frames.shape[0] < 2:
        return 0.0
    return float(F.mse_loss(frames[:-1], frames[1:]).cpu())


def grid_frequency_loss(row: dict[str, Any], mean_frame: torch.Tensor) -> float:
    frames = torch.tensor(row["tensor"], dtype=torch.float32)
    return float(F.mse_loss(mean_frame.expand_as(frames), frames).cpu())


def signal_frequency_loss(row: dict[str, Any], mean_frame: torch.Tensor) -> float:
    frames = torch.tensor(row["tensor"], dtype=torch.float32)
    return float(F.mse_loss(mean_frame.expand_as(frames), frames).cpu())


def latent_vector(trainer: Trainer, row: dict[str, Any]) -> np.ndarray:
    max_len = int(trainer.config["data"]["max_seq_len"][row["modality"]])
    prepared = prepare_sample(row, max_len, trainer.device)
    model = trainer.model
    with torch.no_grad():
        if row["modality"] in {"text", "events"}:
            x = model.byte_embed(prepared.inputs)
            x = model.type_pos(x, MODALITY_IDS[row["modality"]])
        elif row["modality"] == "grid":
            x = model.grid_embed(prepared.inputs)
            x = model.type_pos(x, MODALITY_IDS["grid"])
        elif row["modality"] == "signal":
            x = model.signal_embed(prepared.inputs)
            x = model.type_pos(x, MODALITY_IDS["signal"])
        else:
            raise ValueError(row["modality"])
        z = model.core(x).mean(dim=1).squeeze(0).detach().cpu().numpy()
    return z.astype(np.float64)


def label_for(row: dict[str, Any], probe: str) -> str:
    meta = row["meta"]
    if probe == "grid_position":
        states = meta.get("states", [])
        pos = states[-1]["pos"] if states else [0, 0]
        return f"x{pos[0]}"
    if probe == "grid_class":
        return str(meta["rule_table"].get("0", meta["rule_table"].get(0, 0)))
    if probe == "text_nonterminal":
        return meta["nonterminal"]
    if probe == "signal_frequency_band":
        return meta["frequency_band"]
    if probe == "event_verb":
        return meta["verbs"][0]
    if probe == "event_source":
        return meta["sources"][0]
    raise ValueError(probe)


def linear_probe_accuracy(train_x: np.ndarray, train_y: list[str], test_x: np.ndarray, test_y: list[str]) -> float:
    labels = sorted(set(train_y) | set(test_y))
    label_to_i = {label: i for i, label in enumerate(labels)}
    y = np.zeros((len(train_y), len(labels)), dtype=np.float64)
    for i, label in enumerate(train_y):
        y[i, label_to_i[label]] = 1.0
    x = np.concatenate([train_x, np.ones((train_x.shape[0], 1))], axis=1)
    xt = np.concatenate([test_x, np.ones((test_x.shape[0], 1))], axis=1)
    reg = 1e-3 * np.eye(x.shape[1])
    weights = np.linalg.solve(x.T @ x + reg, x.T @ y)
    pred = (xt @ weights).argmax(axis=1)
    gold = np.array([label_to_i[label] for label in test_y])
    return float((pred == gold).mean())


def run_probes(trainer: Trainer, train_rows: dict[str, list[dict[str, Any]]], heldout_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    probe_map = {
        "grid": ["grid_position", "grid_class"],
        "text": ["text_nonterminal"],
        "signal": ["signal_frequency_band"],
        "events": ["event_verb", "event_source"],
    }
    results = []
    for modality, probes in probe_map.items():
        tr = train_rows[modality][:16]
        te = heldout_rows[modality][:16]
        train_x = np.stack([latent_vector(trainer, row) for row in tr])
        test_x = np.stack([latent_vector(trainer, row) for row in te])
        for probe in probes:
            train_y = [label_for(row, probe) for row in tr]
            test_y = [label_for(row, probe) for row in te]
            acc = linear_probe_accuracy(train_x, train_y, test_x, test_y)
            results.append({"modality": modality, "probe": probe, "accuracy": acc})
    return results


def evaluate_prediction(trainer: Trainer, train_rows: dict[str, list[dict[str, Any]]], eval_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    byte_probs = {
        "text": byte_unigram(train_rows["text"]),
        "events": byte_unigram(train_rows["events"]),
    }
    grid_mean = grid_frequency_frame(train_rows["grid"]).to(trainer.device)
    signal_mean = signal_frequency_frame(train_rows["signal"]).to(trainer.device)
    out: dict[str, Any] = {}
    for modality, rows in eval_rows.items():
        rows = rows[:12]
        model_losses = [model_loss(trainer, row) for row in rows]
        shuffled_losses = [shuffled_model_loss(trainer, row, seed=1000 + i) for i, row in enumerate(rows)]
        if modality in {"text", "events"}:
            persistence = [byte_persistence_loss(row) for row in rows]
            unigram = [byte_unigram_loss(row, byte_probs[modality]) for row in rows]
        elif modality == "grid":
            persistence = [grid_persistence_loss(row) for row in rows]
            unigram = [grid_frequency_loss(row, grid_mean.cpu()) for row in rows]
        elif modality == "signal":
            persistence = [signal_persistence_loss(row) for row in rows]
            unigram = [signal_frequency_loss(row, signal_mean.cpu()) for row in rows]
        else:
            raise ValueError(modality)
        out[modality] = {
            "model_loss": float(np.mean(model_losses)),
            "persistence_loss": float(np.mean(persistence)),
            "unigram_or_frequency_loss": float(np.mean(unigram)),
            "shuffled_input_loss": float(np.mean(shuffled_losses)),
        }
    return out


def transfer_eval(trainer: Trainer, heldout_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    transfer: dict[str, Any] = {}
    for modality, rows in heldout_rows.items():
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["meta"]["family"], []).append(row)
        by_family: dict[str, list[float]] = {}
        for fam, family_rows in sorted(grouped.items()):
            by_family[fam] = [model_loss(trainer, row) for row in family_rows[:12]]
        transfer[modality] = {fam: float(np.mean(vals)) for fam, vals in sorted(by_family.items())}
    return transfer


def apply_killtests(metrics: dict[str, Any], killtests_path: str | Path) -> dict[str, Any]:
    spec = yaml.safe_load(Path(killtests_path).read_text())
    thresholds = spec["thresholds"]
    rows = []
    for modality, vals in metrics["prediction"].items():
        rows.append(
            {
                "test": f"{modality}_prediction_loss_max",
                "value": vals["model_loss"],
                "threshold": thresholds["prediction_loss_max"][modality],
                "passed": vals["model_loss"] <= thresholds["prediction_loss_max"][modality],
            }
        )
        for baseline in thresholds["required_baselines"]:
            key = "unigram_or_frequency_loss" if baseline == "unigram_or_frequency" else f"{baseline}_loss"
            rows.append({"test": f"{modality}_{baseline}_present", "value": vals[key], "threshold": "present", "passed": key in vals})
        ratio = vals["model_loss"] / max(vals["shuffled_input_loss"], 1e-12)
        rows.append(
            {
                "test": f"{modality}_shuffled_loss_ratio_min",
                "value": ratio,
                "threshold": thresholds["shuffled_loss_ratio_min"][modality],
                "passed": ratio >= thresholds["shuffled_loss_ratio_min"][modality],
            }
        )
    for row in metrics["probes"]:
        rows.append(
            {
                "test": f"{row['modality']}_{row['probe']}_accuracy_min",
                "value": row["accuracy"],
                "threshold": thresholds["probe_accuracy_min"],
                "passed": row["accuracy"] >= thresholds["probe_accuracy_min"],
            }
        )
    for modality, families in metrics["transfer"].items():
        for family, value in families.items():
            rows.append(
                {
                    "test": f"{modality}_{family}_transfer_loss_max",
                    "value": value,
                    "threshold": thresholds["transfer_loss_max"][modality],
                    "passed": value <= thresholds["transfer_loss_max"][modality],
                }
            )
    return {"spec": spec, "rows": rows, "passed": all(row["passed"] for row in rows)}


def run_battery(config_path: str | Path, checkpoint_path: str | Path, output_dir: str | Path, killtests_path: str | Path = "killtests.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    trainer = Trainer(config)
    trainer.load_checkpoint(checkpoint_path)
    trainer.model.eval()
    version = config["data"].get("version", "v1")
    train_rows = rows_by_modality(config["data"]["root"], "train", config["data"]["modalities"], version)
    heldout_rows = rows_by_modality(config["data"]["root"], "heldout", config["data"]["modalities"], version)
    metrics = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "step": trainer.step,
        "prediction": evaluate_prediction(trainer, train_rows, heldout_rows),
        "probes": run_probes(trainer, train_rows, heldout_rows),
        "transfer": transfer_eval(trainer, heldout_rows),
    }
    metrics["killtests"] = apply_killtests(metrics, killtests_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    with (output_dir / "probe_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["modality", "probe", "accuracy"])
        writer.writeheader()
        writer.writerows(metrics["probes"])
    with (output_dir / "killtest_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test", "value", "threshold", "passed"])
        writer.writeheader()
        writer.writerows(metrics["killtests"]["rows"])
    return metrics
