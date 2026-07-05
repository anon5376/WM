# S5 Long-Run Infra

Reusable long-run config:

```sh
.venv/bin/python -m wm.train.run --config configs/s5_longrun.yaml
```

The schedule is encoded in config, not CLI-only flags:

- `train.max_steps: 150000`
- `train.wall_clock_seconds: 14400`
- `train.eval_every_seconds: 1800`
- `train.checkpoint_every_steps: 500`
- `train.log_every_steps: 20`

Training writes:

- `runs/<run_id>/metrics.jsonl`
- `runs/<run_id>/eval/*.json`
- `runs/<run_id>/probes/*.csv`
- checkpoints for resume safety, intentionally not committed

After a run, write throughput evidence from the run directory:

```sh
.venv/bin/python -m wm.train.throughput --run-dir runs/<run_id> --output runs/<run_id>/throughput.json
```

