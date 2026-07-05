# WM PROGRAM v1 — General Predictive Network (constitution; commit as docs/WM_PROGRAM.md)
Supersedes WM0_SPEC scope while inheriting its culture. Target artifact: one small unified network that ingests and predicts across virtually any information type available to this project — spatial observations, text-like token streams, continuous signals, event/log streams — through modality adapters into a shared predictive core, with memory, uncertainty, inspection tooling, and a kill-test battery. Built across ~20 stages; any single night completes a contiguous prefix of the earliest unfinished stages. Total program ≈ 4 Codex reset-cycles.

## Laws WG1–WG12 (WL1–WL10 remain in force except where revised)
WG1 (revises WL1): No pretrained weights ever. Data is self-generated/procedural by default; real-world corpora become legal ONLY at Stage 7+ and only public-domain or explicitly-licensed, recorded in `data/MANIFEST.json` with source, license, sha256. Nothing scraped ad hoc.
WG2 All training runs are resume-safe: checkpoint + optimizer + RNG state + dataloader cursor every N steps; a killed run resumes bit-compatibly in config terms (tolerance rules WL2 apply to metrics).
WG3 Every run has a committed config file; no CLI-only hyperparameters. Run IDs are content-hashes of config.
WG4 Scale ladder: parameter count may only increase at designated stages (S6, S13), each time with a before/after eval table. No silent growth.
WG5 Baselines everywhere: persistence, unigram/frequency, and modality-shuffled-input, as applicable. A model number without its baselines is an unreported number.
WG6 Kill-test battery grows monotonically; thresholds freeze after one calibration per test (WL4 discipline). Transfer tests (held-out families, rule-flips) are mandatory per modality.
WG7 The shared core is modality-blind: adapters may know their modality; the core may not (no modality-conditional branches inside core blocks except the standard embedding/type token). A test greps for violations.
WG8 Uncertainty is first-class from S12 on: predictive heads expose variance; calibration is measured, not asserted.
WG9 Inspection artifacts (probe CSVs, attention/latent dumps) ship with every report from S4 on.
WG10 No RL, no reward optimization anywhere in this program. Prediction, representation, memory, adaptation only.
WG11 One repo, `wm/`; WM0's generator and encoder are absorbed as Stage-1 components; nothing imports AM.
WG12 Full-tree `git archive HEAD` per session; numbers only from committed scripts (WL3).

## Stages
Phase 1 — Foundation
S1 Multimodal data substrate: four self-generated modalities under one sample schema `{modality, tokens|tensor, meta}` — (a) gridworld observation sequences (port WM0 gen), (b) procedural text: a context-free grammar family producing world-log narrations + arithmetic/relational sentences with controllable vocab, (c) continuous 1D signals: parameterized oscillator mixtures with embedded event markers, (d) event streams: EG-1-shaped synthetic JSONL. Manifests, splits, held-out FAMILIES per modality for transfer tests.
S2 Adapter layer: per-modality tokenizers/embedders into shared d_model tokens — byte-level path for text/events, patch embed for grids, frame embed for signals; type/position embeddings; inverse heads per modality.
S3 Shared core v0: pre-norm transformer, d_model 256, 6 layers, 4 heads (~10–15M params), multi-task objective = next-token / masked-patch / next-frame per modality, mixed batches, MPS with CPU fallback.
S4 Eval+probe suite v0: per-modality prediction vs WG5 baselines; linear probes (grid: position/class; text: grammar nonterminal; signal: frequency band; events: verb/source); transfer evals on held-out families; killtests.yaml v1.
Phase 2 — Capability
S5 Long-run infra: schedulers, mixed precision where MPS allows, throughput logging, resumable multi-hour runs, eval-every-30-min harness.
S6 Scale step 1 (~30–60M) + curriculum mixing ratios study (committed configs, ablation table).
S7 Licensed real-data ingestion (WG1): public-domain text (e.g., Gutenberg subset), open signal datasets; manifest law enforced by test.
S8 Cross-modal alignment: paired samples (grid episode ↔ its narration ↔ its event log), contrastive/alignment objective, cross-modal retrieval evals.
Phase 3 — Memory & adaptation
S9 Persistent state module: recurrent latent carried across long streams; document-level and episode-level memory evals.
S10 Online adaptation: fast-weights or per-stream cache; measured adaptation-speed benefit.
S11 Continual-learning eval: sequential-task protocol, forgetting metrics, report — no mitigation claims without the table.
S12 Uncertainty heads: per-prediction variance, calibration curves, abstention thresholding.
Phase 4 — Interaction & unification
S13 Action-conditioned branch: WM0's world-model becomes one head of the shared core; scale step 2 permitted.
S14 Goal-conditioned prediction (conditioning tokens; still no RL).
S15 AM event-stream modality first-class: model predicts EG-1 streams; provenance-aware masking.
S16 Optional AM pairing experiment behind a flag: core latents exported per the WM0 interface contract; remains decoupled; skippable without penalty.
Phase 5 — Proof
S17 Unified checkpoint + model card + one-command reproduction script.
S18 Full kill-test battery: all modalities, transfer, ablations (adapter-swap, core-freeze), calibration.
S19 Static inspection dashboard: probes, per-modality errors, calibration, retrieval demos — generated HTML, no server.
S20 Final report: honest capability map — what it predicts well, where it fails, measured, with baselines.

## Tonight (Night 2) = S1→S4 complete + S5 started, per CODEX_GOAL_NIGHT2_WM.md. Program position after tonight ≈ 4/20.


