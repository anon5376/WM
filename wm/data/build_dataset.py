from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from wm.data.events import EVENT_FAMILIES, EVENT_V2_FAMILIES, generate_event_sample, generate_event_v2_sample
from wm.data.grid import generate_grid_episode
from wm.data.schema import write_jsonl
from wm.data.signal import SIGNAL_FAMILIES, generate_signal_sample
from wm.data.text import GRAMMAR_FAMILIES, TEXT_V2_FAMILIES, generate_text_sample, generate_text_v2_sample


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
TRAIN_TEXT_V2 = ["textv2_0", "textv2_1", "textv2_2", "textv2_3", "textv2_4", "textv2_5"]
HELDOUT_TEXT_V2 = ["textv2_6", "textv2_7"]
TRAIN_EVENTS_V2 = ["eventv2_0", "eventv2_1", "eventv2_2", "eventv2_3", "eventv2_4", "eventv2_5"]
HELDOUT_EVENTS_V2 = ["eventv2_6", "eventv2_7"]


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


def build_text_events_v2_rows(samples_per_train_family: int = 2000, samples_per_heldout_family: int = 1000) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}

    def add(split: str, modality: str, row: dict[str, Any]) -> None:
        rows.setdefault(f"{split}/{modality}.jsonl", []).append(row)

    base = DEFAULT_SEEDS["sample_seed_base"] + 200_000
    for split, families, samples_per_family in [
        ("train", TRAIN_TEXT_V2, samples_per_train_family),
        ("heldout", HELDOUT_TEXT_V2, samples_per_heldout_family),
    ]:
        for fam_i, family in enumerate(families):
            assert family in TEXT_V2_FAMILIES
            for i in range(samples_per_family):
                add(split, "text", generate_text_v2_sample(base + fam_i * 100_000 + i, family, split))

    event_base = base + 1_000_000
    for split, families, samples_per_family in [
        ("train", TRAIN_EVENTS_V2, samples_per_train_family),
        ("heldout", HELDOUT_EVENTS_V2, samples_per_heldout_family),
    ]:
        for fam_i, family in enumerate(families):
            assert family in EVENT_V2_FAMILIES
            for i in range(samples_per_family):
                add(split, "events", generate_event_v2_sample(event_base + fam_i * 100_000 + i, family, split))
    return rows


def build_text_events_v2_dataset(
    output: Path,
    samples_per_train_family: int = 2000,
    samples_per_heldout_family: int = 1000,
) -> dict[str, Any]:
    version_dir = output / "v2"
    rows = build_text_events_v2_rows(
        samples_per_train_family=samples_per_train_family,
        samples_per_heldout_family=samples_per_heldout_family,
    )
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
        "dataset": "wm001-v2-text-events",
        "schema": "{modality,tokens,meta,target?}",
        "external_data": [],
        "seeds": {
            "sample_seed_base": DEFAULT_SEEDS["sample_seed_base"] + 200_000,
            "event_seed_base": DEFAULT_SEEDS["sample_seed_base"] + 1_200_000,
        },
        "splits": {
            "train_text_families": TRAIN_TEXT_V2,
            "heldout_text_families": HELDOUT_TEXT_V2,
            "train_event_families": TRAIN_EVENTS_V2,
            "heldout_event_families": HELDOUT_EVENTS_V2,
        },
        "generation": {
            "samples_per_train_family": samples_per_train_family,
            "samples_per_heldout_family": samples_per_heldout_family,
            "modalities": ["text", "events"],
        },
        "counts": {
            "total_samples": sum(s["samples"] for s in shards),
            "train_samples": sum(s["samples"] for s in shards if s["split"] == "train"),
            "heldout_samples": sum(s["samples"] for s in shards if s["split"] == "heldout"),
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
    parser.add_argument("--version", choices=["v1", "v2-text-events"], default="v1")
    parser.add_argument("--samples-per-train-family", type=int, default=2000)
    parser.add_argument("--samples-per-heldout-family", type=int, default=1000)
    args = parser.parse_args()
    if args.version == "v1":
        manifest = build_dataset(args.output, samples_per_family=args.samples_per_family)
    else:
        manifest = build_text_events_v2_dataset(
            args.output,
            samples_per_train_family=args.samples_per_train_family,
            samples_per_heldout_family=args.samples_per_heldout_family,
        )
    print(json.dumps({"total_samples": manifest["counts"]["total_samples"], "total_bytes": manifest["total_bytes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
