from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"checkpoint_step_(\d+)\.pt$")


def checkpoint_steps(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run_dir.glob("checkpoint_step_*.pt")):
        match = STEP_RE.search(path.name)
        if match:
            rows.append({"step": int(match.group(1)), "mtime": path.stat().st_mtime, "path": str(path)})
    return rows


def eval_events(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "eval").glob("*.json")):
        data = json.loads(path.read_text())
        rows.append({"step": int(data["step"]), "wall_time_unix": float(data["wall_time_unix"]), "path": str(path)})
    return sorted(rows, key=lambda row: (row["wall_time_unix"], row["step"]))


def summarize_throughput(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    checkpoints = checkpoint_steps(run_dir)
    evals = eval_events(run_dir)
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "checkpoint_count": len(checkpoints),
        "eval_count": len(evals),
        "checkpoints": checkpoints,
        "evals": evals,
    }
    if len(checkpoints) >= 2:
        first = checkpoints[0]
        last = checkpoints[-1]
        elapsed = max(last["mtime"] - first["mtime"], 1e-9)
        summary["checkpoint_window"] = {
            "first_step": first["step"],
            "last_step": last["step"],
            "elapsed_seconds": elapsed,
            "steps_per_second": (last["step"] - first["step"]) / elapsed,
        }
    if len(evals) >= 2:
        first = evals[0]
        last = evals[-1]
        elapsed = max(last["wall_time_unix"] - first["wall_time_unix"], 1e-9)
        summary["eval_window"] = {
            "first_step": first["step"],
            "last_step": last["step"],
            "elapsed_seconds": elapsed,
            "steps_per_second": (last["step"] - first["step"]) / elapsed,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = summarize_throughput(args.run_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary.get(k) for k in ["checkpoint_count", "eval_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
