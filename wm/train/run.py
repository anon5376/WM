from __future__ import annotations

import argparse
import json
from pathlib import Path

from wm.train.config import load_config
from wm.train.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    trainer = Trainer(config)
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    trainer.train()
    print(json.dumps({"run_id": trainer.hash, "step": trainer.step, "run_path": str(trainer.run_path)}, sort_keys=True))


if __name__ == "__main__":
    main()

