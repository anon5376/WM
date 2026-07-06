# WM Text Interface Session 01 Build Report

Date: 2026-07-06
Repo: `https://github.com/anon5376/WM`
Final code branch: `main`

## Boundary

WM's core is a transformer. Training it to generate text makes this a small, from-scratch language model. "Not an LLM" only means no web-scale pretraining, no external corpus here, and fully inspectable/procedural data. There is no separate grounding layer in this WM-only path, so no-hallucination is not guaranteed by construction. Any low-fabrication claim must be measured later through grounded decoding plus abstention. Point this model at open-world questions and it will fabricate.

## Environment

- Device: MPS selected automatically on this Mac.
- Torch: pinned in `pyproject.toml` as `torch==2.2.2`.
- Parameter count: 11,727,027, unchanged S3-scale core.
- Hard constraints observed: no external data, no pretrained weights, no RL, no AM imports, no core scale increase.
- Final verification: `.venv/bin/python -m pytest -q` passed, 33 tests, 11 warnings.

## T0 Unblock

The missing 0.1.0 checkpoint was uploaded as a GitHub release asset.

- Release: `https://github.com/anon5376/WM/releases/tag/wm-checkpoints-f6d279d3809d252a`
- Asset: `checkpoint_step_150000.pt`
- Asset URL: `https://github.com/anon5376/WM/releases/download/wm-checkpoints-f6d279d3809d252a/checkpoint_step_150000.pt`
- Size: 150,033,703 bytes
- SHA256: `61e2cb61907d7e2ebbb61c7c77e8f7383c71617a7ecdfafefdb0c2039e260295`
- Release asset count after upload: 301
- Evidence: `artifacts/t0_checkpoint_publish.json`

Fresh clone proof was recorded in `artifacts/t0_fresh_clone_pytest.json`: editable install on the GitHub repo, torch 2.2.2, `python -m pytest` green with 24 passed and 6 warnings at that point.

All repo `torch.load(...)` checkpoint loads now pass `weights_only=False` where full optimizer/RNG checkpoint state is required.

## T1 Scaled Text+Events

Dataset v2 was generated with procedural text and event streams only.

- Dataset: `wm001-v2-text-events`
- External data: none
- Train samples: 24,000
- Heldout samples: 4,000
- Total samples: 28,000
- Total bytes: 116,896,835
- Train shards: 12,000 text, 12,000 events
- Heldout shards: 2,000 text, 2,000 events
- Heldout families: `textv2_6`, `textv2_7`, `eventv2_6`, `eventv2_7`
- Manifest: `data/MANIFEST.json`

Training run:

- Config: `configs/text_events_v2.yaml`
- Run id: `5928db1b03a07a60`
- Train rows: 24,000
- Max epochs: 3
- Final checkpoint: `runs/5928db1b03a07a60/checkpoint_step_072000.pt` local, ignored by git
- Final step: 72,000
- Final sampler cursor: 72,000
- Epoch proof: 72,000 / 24,000 = 3.0 epochs
- Note: first segment stopped at step 36,930 due wall-clock guard, then resumed from `checkpoint_step_036930.pt` and reached the hard epoch cap.

T1 final killtest result: failed honestly under tightened thresholds.

| Test | Value | Threshold | Pass |
|---|---:|---:|---|
| text prediction loss max | 0.0000159904 | 5.75 | pass |
| text shuffled loss ratio min | 0.0000010607 | 0.75 | fail |
| events prediction loss max | 0.0000142589 | 5.75 | pass |
| events shuffled loss ratio min | 0.0000011641 | 0.75 | fail |
| text nonterminal probe accuracy min | 0.6875 | 0.30 | pass |
| event verb probe accuracy min | 0.0 | 0.30 | fail |
| event source probe accuracy min | 0.0 | 0.30 | fail |
| text heldout transfer loss max | <= 0.0000900 | 6.25 | pass |
| events heldout transfer loss max | <= 0.0042534 | 6.25 | pass |

Evidence:

- `runs/5928db1b03a07a60/metrics.jsonl`
- `runs/5928db1b03a07a60/t1_final/metrics.json`
- `runs/5928db1b03a07a60/t1_final/killtest_table.csv`
- `runs/5928db1b03a07a60/t1_final/probe_summary.csv`

Interpretation: the model fits the text/events prediction target extremely well, but the shuffled-input ratio and event probes expose that the representation is not robust enough. This is a red result, not a packaging failure.

## T2 Decode Head

Implemented a causal autoregressive text path:

- Boolean causal attention mask in the shared transformer core.
- Causal text training path controlled by `model.causal_text: true`.
- `wm generate --prompt "<text>" --max-tokens N` CLI with greedy and temperature sampling.
- Decode eval with unigram and bigram byte baselines plus a committed coherence proxy.

The first T2 attempt produced NaN loss at step 1 because the MPS attention kernel did not tolerate the `-inf` float mask path. That failed run was stopped and preserved under `artifacts/t2_decode_nan_failed_run/`. The mask was changed to a boolean causal mask and covered by `tests/test_decode.py::test_causal_forward_is_finite`.

Final decode training:

- Config: `configs/text_decode_v2.yaml`
- Run id: `eb2b204d7a033057`
- Train rows: 12,000 text rows
- Max epochs: 3
- Final checkpoint: `runs/eb2b204d7a033057/checkpoint_step_036000.pt` local, ignored by git
- Final step: 36,000
- Final sampler cursor: 36,000
- Epoch proof: 36,000 / 12,000 = 3.0 epochs

Decode killtest result: failed because the model did not beat the bigram baseline.

| Test | Value | Threshold | Pass |
|---|---:|---:|---|
| model margin vs unigram min | 0.4561601 | 0.25 | pass |
| model margin vs bigram min | -0.4158222 | 0.05 | fail |
| coherence char-bigram overlap min | 0.8311944 | 0.55 | pass |

Heldout byte losses:

| Model | Loss |
|---|---:|
| WM causal model | 2.7581273 |
| Unigram baseline | 3.2142874 |
| Bigram baseline | 2.3423051 |

Sample from the CLI:

```text
Prompt: uma counts 6 bucket
Output: uma counts 6 bucket plus 4 beacon equals 10 .eter .low .event .st .st .n . . .s
```

Sample interpretation: it can often continue the local procedural arithmetic pattern at the start, then degrades into fragment repetition. The char-bigram coherence proxy is too weak to prove real generation quality; the bigram baseline failure is the controlling result.

Evidence:

- `runs/eb2b204d7a033057/metrics.jsonl`
- `runs/eb2b204d7a033057/decode_final/decode_metrics.json`
- `runs/eb2b204d7a033057/decode_final/decode_samples.md`

## Status

T0 complete.
T1 complete with honest red killtest rows.
T2 implementation complete and trained, but decode killtest failed because the model did not beat the bigram baseline.
T3 was not started because the T2 stop condition says to stop and document instead of tuning past the registered test.

## What It Does Right Now

The repo can:

- Generate deterministic procedural text/event data.
- Train the shared WM transformer on scaled text/events under a hard epoch cap.
- Train a causal next-byte text model from scratch on procedural text.
- Run `wm generate` to emit continuations from a prompt.
- Score decode output against unigram and bigram byte baselines.

It does not yet:

- Reliably respond as a useful assistant.
- Beat a bigram baseline on heldout procedural text.
- Abstain.
- Attribute answers to context.
- Prevent fabrication.
- Handle open-world text.

## Next Work

1. Improve the decode objective and evaluation before any interface claims:
   - stronger causal batching,
   - explicit BOS/EOS handling,
   - train/heldout prompt templates that make continuation boundaries less ambiguous,
   - compare byte-level vs small procedural-token-level decoding.
2. Make the coherence metric harder:
   - exact field validity,
   - arithmetic correctness,
   - grammar parse success,
   - repetition penalty.
3. Only after beating bigram, start T3:
   - extractive/constrained decoding,
   - abstention head,
   - faithfulness and calibration evals.
4. Keep the boundary visible: closed-domain only until licensed data and grounding/abstention work exist.

## Archive

Final full-tree archive is generated in `artifacts/` after this report is committed. As usual, a `git archive HEAD` snapshot cannot include the archive file that is created from that same HEAD, but it includes all tracked files and artifacts present in HEAD at archive time.
