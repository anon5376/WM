from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from wm.adapters.byte_tokenizer import ByteTokenizer
from wm.data.schema import read_jsonl
from wm.generate import generate_text
from wm.train.config import load_config
from wm.train.data import prepare_sample
from wm.train.trainer import Trainer


def _rows(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    root = Path(config["data"]["root"]) / config["data"].get("version", "v1") / split / "text.jsonl"
    return read_jsonl(root)


def unigram_probs(rows: list[dict[str, Any]]) -> np.ndarray:
    tok = ByteTokenizer()
    counts = np.ones(256, dtype=np.float64)
    for row in rows:
        counts[tok.encode(row["tokens"])] += 1
    return counts / counts.sum()


def bigram_probs(rows: list[dict[str, Any]]) -> np.ndarray:
    tok = ByteTokenizer()
    counts = np.ones((256, 256), dtype=np.float64)
    for row in rows:
        ids = tok.encode(row["tokens"])
        for prev, nxt in zip(ids[:-1], ids[1:]):
            counts[prev, nxt] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def unigram_loss(row: dict[str, Any], probs: np.ndarray) -> float:
    tok = ByteTokenizer()
    ids = tok.encode(row["tokens"])
    if len(ids) < 2:
        return 0.0
    targets = np.array(ids[1:], dtype=np.int64)
    return float(-np.log(probs[targets]).mean())


def bigram_loss(row: dict[str, Any], probs: np.ndarray) -> float:
    tok = ByteTokenizer()
    ids = tok.encode(row["tokens"])
    if len(ids) < 2:
        return 0.0
    vals = [-np.log(probs[prev, nxt]) for prev, nxt in zip(ids[:-1], ids[1:])]
    return float(np.mean(vals))


def model_text_loss(trainer: Trainer, row: dict[str, Any]) -> float:
    max_len = int(trainer.config["data"]["max_seq_len"]["text"])
    prepared = prepare_sample(row, max_len, trainer.device)
    with torch.no_grad():
        return float(trainer.loss_for(prepared).detach().cpu())


def char_bigrams(text: str) -> set[str]:
    return {text[i : i + 2] for i in range(max(len(text) - 1, 0))}


def bigram_overlap(text: str, reference: set[str]) -> float:
    grams = char_bigrams(text)
    if not grams:
        return 0.0
    return len(grams & reference) / len(grams)


def run_decode_eval(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    killtests_path: str | Path = "killtests.yaml",
    max_eval_rows: int = 256,
) -> dict[str, Any]:
    config = load_config(config_path)
    trainer = Trainer(config)
    trainer.load_checkpoint(checkpoint_path)
    trainer.model.eval()
    train_rows = _rows(config, "train")
    heldout_rows = _rows(config, "heldout")[:max_eval_rows]
    uni = unigram_probs(train_rows)
    bi = bigram_probs(train_rows)
    model_losses = [model_text_loss(trainer, row) for row in heldout_rows]
    unigram_losses = [unigram_loss(row, uni) for row in heldout_rows]
    bigram_losses = [bigram_loss(row, bi) for row in heldout_rows]
    prompts = [" ".join(row["tokens"].split()[:4]) for row in heldout_rows[:8]]
    generations = [
        generate_text(config_path, checkpoint_path, prompt, max_tokens=80, temperature=0.0)
        for prompt in prompts
    ]
    reference_bigrams = set()
    for row in train_rows[:2000] + heldout_rows:
        reference_bigrams |= char_bigrams(row["tokens"])
    overlaps = [bigram_overlap(str(item["continuation"]), reference_bigrams) for item in generations]
    thresholds = yaml.safe_load(Path(killtests_path).read_text())["decode_thresholds"]
    metrics = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "step": trainer.step,
        "loss": {
            "model": float(np.mean(model_losses)),
            "unigram": float(np.mean(unigram_losses)),
            "bigram": float(np.mean(bigram_losses)),
            "model_margin_vs_unigram": float(np.mean(unigram_losses) - np.mean(model_losses)),
            "model_margin_vs_bigram": float(np.mean(bigram_losses) - np.mean(model_losses)),
        },
        "coherence": {
            "char_bigram_overlap_mean": float(np.mean(overlaps)),
            "char_bigram_overlap_by_sample": overlaps,
        },
        "generations": generations,
    }
    rows = [
        {
            "test": "decode_model_margin_vs_unigram_min",
            "value": metrics["loss"]["model_margin_vs_unigram"],
            "threshold": thresholds["byte_loss_margin_min"]["unigram"],
            "passed": metrics["loss"]["model_margin_vs_unigram"] >= thresholds["byte_loss_margin_min"]["unigram"],
        },
        {
            "test": "decode_model_margin_vs_bigram_min",
            "value": metrics["loss"]["model_margin_vs_bigram"],
            "threshold": thresholds["byte_loss_margin_min"]["bigram"],
            "passed": metrics["loss"]["model_margin_vs_bigram"] >= thresholds["byte_loss_margin_min"]["bigram"],
        },
        {
            "test": "decode_coherence_bigram_overlap_min",
            "value": metrics["coherence"]["char_bigram_overlap_mean"],
            "threshold": thresholds["coherence_bigram_overlap_min"],
            "passed": metrics["coherence"]["char_bigram_overlap_mean"] >= thresholds["coherence_bigram_overlap_min"],
        },
    ]
    metrics["killtests"] = {"rows": rows, "passed": all(row["passed"] for row in rows)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decode_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    transcript = []
    for i, item in enumerate(generations):
        transcript.append(f"## sample {i}\nPROMPT: {item['prompt']}\nOUTPUT: {item['text']}\n")
    (output_dir / "decode_samples.md").write_text("\n".join(transcript))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--killtests", default="killtests.yaml")
    args = parser.parse_args()
    metrics = run_decode_eval(args.config, args.checkpoint, args.output_dir, killtests_path=args.killtests)
    print(json.dumps({"step": metrics["step"], "killtests_passed": metrics["killtests"]["passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
