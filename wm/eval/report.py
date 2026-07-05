from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def latest_metric_step(metrics_jsonl: Path) -> int:
    last = 0
    if not metrics_jsonl.exists():
        return last
    for line in metrics_jsonl.read_text().splitlines():
        if line.strip():
            last = int(json.loads(line)["step"])
    return last


def write_report(
    *,
    manifest_path: str | Path,
    run_dir: str | Path,
    final_battery_dir: str | Path,
    throughput_path: str | Path,
    output_path: str | Path,
) -> None:
    manifest = read_json(manifest_path)
    run_dir = Path(run_dir)
    battery_dir = Path(final_battery_dir)
    throughput = read_json(throughput_path)
    metrics = read_json(battery_dir / "metrics.json")
    kill_rows = csv_rows(battery_dir / "killtest_table.csv")
    probe_rows = csv_rows(battery_dir / "probe_summary.csv")
    metadata = read_json(run_dir / "metadata.json")
    eval_files = sorted((run_dir / "eval").glob("*.json"))
    probe_files = sorted((run_dir / "probes").glob("*.csv"))
    checkpoint_files = sorted(run_dir.glob("checkpoint_step_*.pt"))
    latest_step = latest_metric_step(run_dir / "metrics.jsonl")

    lines = [
        "# BUILD REPORT WM N2R",
        "",
        "## Scope",
        "",
        "Night 2R replaced the blocked WM0 absorption path with a fresh local bootstrap. No external or real-world data was used.",
        "",
        "## Environment And Seeds",
        "",
        f"- Run id: `{metadata['run_id']}`",
        f"- Config hash: `{metadata['config_hash']}`",
        f"- Device: `{metadata['device']['selected']}` ({metadata['device']['reason']})",
        f"- Device metadata: `{metadata['device']}`",
        f"- Seed: `{manifest['seeds']['sample_seed_base']}` data base; train seed is recorded in `{run_dir / 'config.yaml'}`",
        f"- Parameter count: `{metadata['parameter_count']}`",
        "",
        "## Dataset",
        "",
        f"- Dataset: `{manifest['dataset']}`",
        f"- Total samples: `{manifest['counts']['total_samples']}`",
        f"- Total bytes: `{manifest['total_bytes']}`",
        f"- External data: `{manifest['external_data']}`",
        f"- Held-out grammar families: `{manifest['splits']['heldout_grammar_families']}`",
        f"- Held-out signal families: `{manifest['splits']['heldout_signal_families']}`",
        f"- Held-out grid rule seeds: `{manifest['splits']['heldout_grid_rule_seeds']}`",
        "",
        "## Training Run",
        "",
        f"- Metrics: `{run_dir / 'metrics.jsonl'}`",
        f"- Latest logged metric step: `{latest_step}`",
        f"- Checkpoints written: `{len(checkpoint_files)}` (not committed; run metrics/probes are tracked)",
        f"- Eval JSON files: `{len(eval_files)}`",
        f"- Probe CSV files: `{len(probe_files)}`",
        f"- Throughput summary: `{throughput_path}`",
    ]
    if "checkpoint_window" in throughput:
        cw = throughput["checkpoint_window"]
        lines.append(f"- Checkpoint-window throughput: `{cw['steps_per_second']:.3f}` steps/sec from step {cw['first_step']} to {cw['last_step']}")
    if "eval_window" in throughput:
        ew = throughput["eval_window"]
        lines.append(f"- Eval-window throughput: `{ew['steps_per_second']:.3f}` steps/sec from step {ew['first_step']} to {ew['last_step']}")

    lines.extend(["", "## Prediction And Baselines", ""])
    for modality, vals in metrics["prediction"].items():
        lines.append(
            f"- {modality}: model `{vals['model_loss']:.6g}`, persistence `{vals['persistence_loss']:.6g}`, "
            f"unigram/frequency `{vals['unigram_or_frequency_loss']:.6g}`, shuffled `{vals['shuffled_input_loss']:.6g}`"
        )

    lines.extend(["", "## Probe Summary", ""])
    for row in probe_rows:
        lines.append(f"- {row['modality']} {row['probe']}: accuracy `{float(row['accuracy']):.3f}`")

    lines.extend(["", "## K Table", ""])
    for row in kill_rows:
        lines.append(f"- {row['test']}: value `{row['value']}`, threshold `{row['threshold']}`, pass `{row['passed']}`")

    lines.extend(["", "## WG Compliance Notes", ""])
    lines.extend(
        [
            "- WG1: data is procedural; manifest records no external sources.",
            "- WG2: checkpoints include model, optimizer, sampler cursor, and RNG states; kill-and-resume test is committed.",
            "- WG3: run ids are content hashes of committed config files.",
            "- WG4: S3 model is pinned at d_model 256, 6 layers, 4 heads, with parameter count in range.",
            "- WG5: persistence, unigram/frequency, and shuffled-input baselines are reported.",
            "- WG7: core lint forbids modality conditionals in `wm/core`.",
            "- WG9: scheduled eval JSON and probe CSV artifacts are written by the training harness.",
            "- WG10/WG11: no RL and no AM imports were added.",
            "- WG12: final archive is generated from committed `HEAD`.",
        ]
    )

    lines.extend(["", "## Honest Limitations", ""])
    lines.extend(
        [
            "- Dataset v1 is intentionally small for a local bootstrap; it is not a capability claim beyond S1-S4 plumbing.",
            "- Kill-test thresholds are provisional v1 calibration; failing rows remain reported instead of weakened.",
            "- Checkpoints are not tracked because M0R explicitly gitignored `runs/` except metrics JSON/CSV artifacts.",
        ]
    )

    lines.extend(["", "## Next Stages", ""])
    lines.extend(
        [
            "- S5+: improve mixed precision and throughput reporting now that the eval schedule is reusable.",
            "- S6: only then consider the permitted scale-step study with before/after tables.",
            "- S7: external data remains gated and must be licensed/manifested before use.",
        ]
    )
    Path(output_path).write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/MANIFEST.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--final-battery-dir", required=True)
    parser.add_argument("--throughput", required=True)
    parser.add_argument("--output", default="docs/BUILD_REPORT_WM_N2R.md")
    args = parser.parse_args()
    write_report(
        manifest_path=args.manifest,
        run_dir=args.run_dir,
        final_battery_dir=args.final_battery_dir,
        throughput_path=args.throughput,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
