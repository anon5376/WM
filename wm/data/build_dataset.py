from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from wm.data.events import EVENT_FAMILIES, generate_event_sample
from wm.data.grid import generate_grid_episode
from wm.data.schema import write_jsonl
from wm.data.signal import SIGNAL_FAMILIES, generate_signal_sample
from wm.data.text import GRAMMAR_FAMILIES, generate_text_sample


DEFAULT_SEEDS = {
    "grid_train_rule_seeds": [11, 12],
    "grid_heldout_rule_seeds": [101, 102],
    "sample_seed_base": 7000,
}
TRAIN_GRAMMAR = ["grammar_0", "grammar_1"]
HELDOUT_GRAMMAR = ["grammar_2", "grammar_3"]
TRAIN_SIGNAL = ["signal_0", "signal_1"]
HELDOUT_SIGNAL = ["signal_2", "signal_3"]
TRAIN_EVENTS = ["event_0", "event_1"]
HELDOUT_EVENTS = ["event_2", "event_3"]


def build_rows(samples_per_family: int = 12) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}

    def add(split: str, modality: str, row: dict[str, Any]) -> None:
        rows.setdefault(f"{split}/{modality}.jsonl", []).append(row)

    base = DEFAULT_SEEDS["sample_seed_base"]
    for split, rule_seeds in [
        ("train", DEFAULT_SEEDS["grid_train_rule_seeds"]),
        ("heldout", DEFAULT_SEEDS["grid_heldout_rule_seeds"]),
    ]:
        for rule_seed in rule_seeds:
            for i in range(samples_per_family):
                add(split, "grid", generate_grid_episode(base + rule_seed * 100 + i, rule_seed, split))

    for split, families in [("train", TRAIN_GRAMMAR), ("heldout", HELDOUT_GRAMMAR)]:
        for fam_i, family in enumerate(families):
            assert family in GRAMMAR_FAMILIES
            for i in range(samples_per_family * 2):
                add(split, "text", generate_text_sample(base + fam_i * 1000 + i, family, split))

    for split, families in [("train", TRAIN_SIGNAL), ("heldout", HELDOUT_SIGNAL)]:
        for fam_i, family in enumerate(families):
            assert family in SIGNAL_FAMILIES
            for i in range(samples_per_family * 2):
                add(split, "signal", generate_signal_sample(base + fam_i * 1000 + i, family, split))

    for split, families in [("train", TRAIN_EVENTS), ("heldout", HELDOUT_EVENTS)]:
        for fam_i, family in enumerate(families):
            assert family in EVENT_FAMILIES
            for i in range(samples_per_family * 2):
                add(split, "events", generate_event_sample(base + fam_i * 1000 + i, family, split))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset(output: Path, samples_per_family: int = 12) -> dict[str, Any]:
    version_dir = output / "v1"
    rows = build_rows(samples_per_family=samples_per_family)
    shards = []
    for rel, shard_rows in sorted(rows.items()):
        path = version_dir / rel
        digest = write_jsonl(path, shard_rows)
        shards.append(
            {
                "path": str(path.relative_to(output)),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "samples": len(shard_rows),
                "split": rel.split("/")[0],
                "modality": Path(rel).stem,
            }
        )
    manifest = {
        "dataset": "wm001-v1",
        "schema": "{modality,tokens|tensor,meta,target?}",
        "external_data": [],
        "seeds": DEFAULT_SEEDS,
        "splits": {
            "heldout_grammar_families": HELDOUT_GRAMMAR,
            "heldout_signal_families": HELDOUT_SIGNAL,
            "heldout_grid_rule_seeds": DEFAULT_SEEDS["grid_heldout_rule_seeds"],
        },
        "counts": {
            "total_samples": sum(s["samples"] for s in shards),
            "by_shard": {s["path"]: s["samples"] for s in shards},
        },
        "shards": shards,
        "total_bytes": sum(s["bytes"] for s in shards),
    }
    manifest_path = output / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--samples-per-family", type=int, default=12)
    args = parser.parse_args()
    manifest = build_dataset(args.output, samples_per_family=args.samples_per_family)
    print(json.dumps({"total_samples": manifest["counts"]["total_samples"], "total_bytes": manifest["total_bytes"]}, sort_keys=True))


if __name__ == "__main__":
    main()

