import json
from pathlib import Path

import torch
import yaml

from wm.train.config import load_config
from wm.train.model import MultiModalPredictor, count_parameters
from wm.train.trainer import Trainer


def test_s3_parameter_count_is_on_pinned_scale():
    config = load_config("configs/s3.yaml")
    model = MultiModalPredictor(config)
    params = count_parameters(model)
    assert 10_000_000 <= params <= 15_000_000


def test_checkpoint_resume_matches_continuous_run(tmp_path: Path):
    base = load_config("configs/test_tiny.yaml")
    continuous_cfg = json.loads(json.dumps(base))
    continuous_cfg["run_root"] = str(tmp_path / "continuous")
    interrupted_cfg = json.loads(json.dumps(base))
    interrupted_cfg["run_root"] = str(tmp_path / "interrupted")

    continuous = Trainer(continuous_cfg)
    continuous.train()
    continuous_state = {k: v.detach().clone() for k, v in continuous.model.state_dict().items()}

    interrupted = Trainer(interrupted_cfg)
    interrupted.train(stop_after_steps=200)
    ckpt = interrupted.latest_checkpoint()
    assert ckpt is not None

    resumed = Trainer(interrupted_cfg)
    resumed.load_checkpoint(ckpt)
    assert resumed.step == 200
    resumed.train()

    assert resumed.step == continuous.step == 205
    assert resumed.sampler.cursor == continuous.sampler.cursor
    for key, value in resumed.model.state_dict().items():
        assert torch.allclose(value, continuous_state[key], atol=1e-6, rtol=1e-5), key


def test_checkpoint_rejects_config_mismatch(tmp_path: Path):
    config = load_config("configs/test_tiny.yaml")
    config["run_root"] = str(tmp_path / "run")
    trainer = Trainer(config)
    trainer.train(stop_after_steps=1)
    ckpt = trainer.latest_checkpoint()
    assert ckpt is not None
    changed = json.loads(json.dumps(config))
    changed["train"]["lr"] = config["train"]["lr"] * 2
    with (tmp_path / "changed.yaml").open("w") as f:
        yaml.safe_dump(changed, f)
    other = Trainer(changed)
    try:
        other.load_checkpoint(ckpt)
    except ValueError as exc:
        assert "config hash" in str(exc)
    else:
        raise AssertionError("config mismatch was accepted")


def test_checkpoint_load_accepts_rng_state_tensor_on_selected_device(tmp_path: Path):
    config = load_config("configs/test_tiny.yaml")
    config["run_root"] = str(tmp_path / "run")
    trainer = Trainer(config)
    trainer.train(stop_after_steps=1)
    ckpt = trainer.latest_checkpoint()
    assert ckpt is not None
    payload = torch.load(ckpt, map_location=trainer.device, weights_only=False)
    payload["torch_rng_state"] = payload["torch_rng_state"].to(trainer.device)
    altered = tmp_path / "rng_device_checkpoint.pt"
    torch.save(payload, altered)
    restored = Trainer(config)
    restored.load_checkpoint(altered)
    assert restored.step == 1
