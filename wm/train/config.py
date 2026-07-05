from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def stable_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config)


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(stable_config(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def run_dir(config: dict[str, Any]) -> Path:
    return Path(config.get("run_root", "runs")) / config_hash(config)

