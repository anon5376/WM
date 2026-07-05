from pathlib import Path

import yaml

from wm.eval.battery import run_battery
from wm.train.config import load_config
from wm.train.trainer import Trainer


def test_eval_battery_outputs_baselines_probes_and_killtests(tmp_path: Path):
    config = load_config("configs/test_tiny.yaml")
    config["run_root"] = str(tmp_path / "run")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    trainer = Trainer(config)
    trainer.train(stop_after_steps=2)
    checkpoint = trainer.latest_checkpoint()
    assert checkpoint is not None
    out = tmp_path / "eval"
    metrics = run_battery(config_path, checkpoint, out)
    assert set(metrics["prediction"]) == {"grid", "text", "signal", "events"}
    for vals in metrics["prediction"].values():
        assert "persistence_loss" in vals
        assert "unigram_or_frequency_loss" in vals
        assert "shuffled_input_loss" in vals
    assert metrics["probes"]
    assert metrics["transfer"]
    assert len(metrics["transfer"]["text"]) >= 2
    assert len(metrics["transfer"]["signal"]) >= 2
    assert len(metrics["transfer"]["grid"]) >= 2
    assert (out / "metrics.json").exists()
    assert (out / "probe_summary.csv").exists()
    assert (out / "killtest_table.csv").exists()
