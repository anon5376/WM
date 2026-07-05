from __future__ import annotations

import argparse
import json

from wm.eval.battery import run_battery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--killtests", default="killtests.yaml")
    args = parser.parse_args()
    metrics = run_battery(args.config, args.checkpoint, args.output_dir, args.killtests)
    print(json.dumps({"step": metrics["step"], "killtests_passed": metrics["killtests"]["passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()

