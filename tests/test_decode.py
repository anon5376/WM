import json
from pathlib import Path
import subprocess
import sys

import yaml

from wm.eval.decode import bigram_loss, bigram_probs, run_decode_eval
from wm.generate import generate_text
from wm.train.config import load_config
from wm.train.trainer import Trainer


def test_bigram_baseline_prefers_seen_transition():
    rows = [{"tokens": "aaaa"}, {"tokens": "aaab"}]
    probs = bigram_probs(rows)
    assert bigram_loss({"tokens": "aaaa"}, probs) < bigram_loss({"tokens": "azzz"}, probs)


def test_generate_cli_returns_prompt_plus_bytes(tmp_path: Path):
    config = load_config("configs/test_decode.yaml")
    config["run_root"] = str(tmp_path / "run")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    trainer = Trainer(config)
    trainer.train()
    checkpoint = trainer.latest_checkpoint()
    assert checkpoint is not None
    out = generate_text(config_path, checkpoint, "agent", max_tokens=4)
    assert out["text"].startswith("agent")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "wm.generate",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--prompt",
            "agent",
            "--max-tokens",
            "4",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["prompt"] == "agent"


def test_decode_eval_writes_metrics(tmp_path: Path):
    config = load_config("configs/test_decode.yaml")
    config["run_root"] = str(tmp_path / "run")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    trainer = Trainer(config)
    trainer.train()
    checkpoint = trainer.latest_checkpoint()
    assert checkpoint is not None
    out_dir = tmp_path / "decode_eval"
    metrics = run_decode_eval(config_path, checkpoint, out_dir, max_eval_rows=4)
    assert "bigram" in metrics["loss"]
    assert metrics["generations"]
    assert (out_dir / "decode_metrics.json").exists()
    assert (out_dir / "decode_samples.md").exists()
