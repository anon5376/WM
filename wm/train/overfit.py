from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from wm.train.config import load_config
from wm.train.data import load_training_rows, prepare_sample
from wm.train.trainer import Trainer


def overfit_sanity(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    base_config = load_config(config_path)
    rows = load_training_rows(base_config["data"]["root"], base_config["data"]["split"], base_config["data"]["modalities"])
    selected = {name: next(row for row in rows if row["modality"] == name) for name in base_config["data"]["modalities"]}
    results: dict[str, Any] = {"config": str(config_path), "modalities": {}}
    for modality, row in selected.items():
        config = json.loads(json.dumps(base_config))
        config["data"]["modalities"] = [modality]
        trainer = Trainer(config, run_path=Path("runs") / f"overfit_{modality}")
        max_len = int(config["data"]["max_seq_len"][modality])
        prepared = prepare_sample(row, max_len, trainer.device)
        with torch.no_grad():
            initial = float(trainer.loss_for(prepared).detach().cpu())
        losses = [initial]
        for _ in range(int(config["train"]["max_steps"])):
            trainer.model.train()
            loss = trainer.loss_for(prepared)
            trainer.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
            trainer.optimizer.step()
        with torch.no_grad():
            final = float(trainer.loss_for(prepared).detach().cpu())
        losses.append(final)
        results["modalities"][modality] = {
            "initial_loss": initial,
            "final_loss": final,
            "fall_ratio": initial / max(final, 1e-12),
            "passed": final <= initial / 5.0,
        }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/overfit.yaml")
    parser.add_argument("--output", default="artifacts/m3_overfit_sanity.json")
    args = parser.parse_args()
    results = overfit_sanity(args.config, args.output)
    print(json.dumps(results["modalities"], sort_keys=True))


if __name__ == "__main__":
    main()

