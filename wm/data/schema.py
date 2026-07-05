from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def canonical_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with path.open("wb") as f:
        for row in rows:
            line = (canonical_json(row) + "\n").encode("utf-8")
            f.write(line)
            h.update(line)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sample(modality: str, *, meta: dict[str, Any], tokens: Any = None, tensor: Any = None, target: Any = None) -> dict[str, Any]:
    row: dict[str, Any] = {"modality": modality, "meta": meta}
    if tokens is not None:
        row["tokens"] = tokens
    if tensor is not None:
        row["tensor"] = tensor
    if target is not None:
        row["target"] = target
    if ("tokens" in row) == ("tensor" in row):
        raise ValueError("sample must provide exactly one of tokens or tensor")
    return row

