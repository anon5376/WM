import json
import os
from pathlib import Path

from wm.train.throughput import summarize_throughput


def test_throughput_summary_from_checkpoints_and_eval_events(tmp_path: Path):
    run = tmp_path / "run"
    eval_dir = run / "eval"
    eval_dir.mkdir(parents=True)
    ckpt_a = run / "checkpoint_step_000500.pt"
    ckpt_b = run / "checkpoint_step_001000.pt"
    ckpt_a.write_bytes(b"a")
    ckpt_b.write_bytes(b"b")
    os.utime(ckpt_a, (1000, 1000))
    os.utime(ckpt_b, (1010, 1010))
    (eval_dir / "scheduled_step_000000.json").write_text(json.dumps({"step": 0, "wall_time_unix": 1000.0}))
    (eval_dir / "final_step_001000.json").write_text(json.dumps({"step": 1000, "wall_time_unix": 1020.0}))
    summary = summarize_throughput(run)
    assert summary["checkpoint_window"]["steps_per_second"] == 50.0
    assert summary["eval_window"]["steps_per_second"] == 50.0

