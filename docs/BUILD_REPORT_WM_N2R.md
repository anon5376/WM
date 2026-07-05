# BUILD REPORT WM N2R

## Scope

Night 2R replaced the blocked WM0 absorption path with a fresh local bootstrap. No external or real-world data was used.

## Environment And Seeds

- Run id: `f6d279d3809d252a`
- Config hash: `f6d279d3809d252a`
- Device: `mps` (auto selected mps)
- Device metadata: `{'mps_available': True, 'reason': 'auto selected mps', 'requested': 'auto', 'selected': 'mps'}`
- Seed: `7000` data base; train seed is recorded in `runs/f6d279d3809d252a/config.yaml`
- Parameter count: `11727027`

## Dataset

- Dataset: `wm001-v1`
- Total samples: `336`
- Total bytes: `10380431`
- External data: `[]`
- Held-out grammar families: `['grammar_2', 'grammar_3']`
- Held-out signal families: `['signal_2', 'signal_3']`
- Held-out grid rule seeds: `[101, 102]`
- Grid train rule tables: `{11: {0: 0, 1: 1}, 12: {0: 0, 1: 1}}`
- Grid held-out rule-flip tables: `{101: {0: 1, 1: 0}, 102: {0: 1, 1: 0}}`

## Training Run

- Metrics: `runs/f6d279d3809d252a/metrics.jsonl`
- Latest logged metric step: `150000`
- Checkpoints written: `300` (not committed; run metrics/probes are tracked)
- Eval JSON files: `4`
- Probe CSV files: `4`
- Throughput summary: `runs/f6d279d3809d252a/throughput_final.json`
- Checkpoint-window throughput: `33.238` steps/sec from step 500 to 150000
- Eval-window throughput: `33.219` steps/sec from step 0 to 150000

## Prediction And Baselines

- events: model `1.25132`, persistence `14.309`, unigram/frequency `3.87255`, shuffled `6.89663`
- grid: model `0.0204974`, persistence `0.0222516`, unigram/frequency `0.0297737`, shuffled `0.0314103`
- signal: model `0.0147504`, persistence `0.0528952`, unigram/frequency `0.23714`, shuffled `0.463717`
- text: model `4.87514`, persistence `14.3856`, unigram/frequency `3.38206`, shuffled `10.3839`

## Probe Summary

- grid grid_position: accuracy `0.438`
- grid grid_class: accuracy `0.000`
- text text_nonterminal: accuracy `0.688`
- signal signal_frequency_band: accuracy `0.000`
- events event_verb: accuracy `0.000`
- events event_source: accuracy `0.000`

## K Table

- grid_prediction_loss_max: value `0.020497381609554093`, threshold `0.75`, pass `True`
- grid_persistence_present: value `0.022251637109244864`, threshold `present`, pass `True`
- grid_unigram_or_frequency_present: value `0.02977374444405238`, threshold `present`, pass `True`
- grid_shuffled_input_present: value `0.03141028806567192`, threshold `present`, pass `True`
- grid_shuffled_loss_ratio_min: value `0.652569042560152`, threshold `0.5`, pass `True`
- text_prediction_loss_max: value `4.875143369038899`, threshold `6.25`, pass `True`
- text_persistence_present: value `14.385605935180331`, threshold `present`, pass `True`
- text_unigram_or_frequency_present: value `3.3820575302373768`, threshold `present`, pass `True`
- text_shuffled_input_present: value `10.383911887804667`, threshold `present`, pass `True`
- text_shuffled_loss_ratio_min: value `0.46949005555069157`, threshold `0.5`, pass `False`
- signal_prediction_loss_max: value `0.014750419727837047`, threshold `1.0`, pass `True`
- signal_persistence_present: value `0.05289523210376501`, threshold `present`, pass `True`
- signal_unigram_or_frequency_present: value `0.23713957394162813`, threshold `present`, pass `True`
- signal_shuffled_input_present: value `0.46371713529030484`, threshold `present`, pass `True`
- signal_shuffled_loss_ratio_min: value `0.03180908921686216`, threshold `0.5`, pass `False`
- events_prediction_loss_max: value `1.2513192494710286`, threshold `6.25`, pass `True`
- events_persistence_present: value `14.30895730998895`, threshold `present`, pass `True`
- events_unigram_or_frequency_present: value `3.8725482338651087`, threshold `present`, pass `True`
- events_shuffled_input_present: value `6.896628061930339`, threshold `present`, pass `True`
- events_shuffled_loss_ratio_min: value `0.18143928282552463`, threshold `0.5`, pass `False`
- grid_grid_position_accuracy_min: value `0.4375`, threshold `0.25`, pass `True`
- grid_grid_class_accuracy_min: value `0.0`, threshold `0.25`, pass `False`
- text_text_nonterminal_accuracy_min: value `0.6875`, threshold `0.25`, pass `True`
- signal_signal_frequency_band_accuracy_min: value `0.0`, threshold `0.25`, pass `False`
- events_event_verb_accuracy_min: value `0.0`, threshold `0.25`, pass `False`
- events_event_source_accuracy_min: value `0.0`, threshold `0.25`, pass `False`
- grid_rule_seed_101_transfer_loss_max: value `0.020497381609554093`, threshold `1.0`, pass `True`
- grid_rule_seed_102_transfer_loss_max: value `0.016163147403858602`, threshold `1.0`, pass `True`
- text_grammar_2_transfer_loss_max: value `4.875143369038899`, threshold `6.75`, pass `True`
- text_grammar_3_transfer_loss_max: value `5.580457240343094`, threshold `6.75`, pass `True`
- signal_signal_2_transfer_loss_max: value `0.014750419727837047`, threshold `1.25`, pass `True`
- signal_signal_3_transfer_loss_max: value `0.04617215289423863`, threshold `1.25`, pass `True`
- events_event_2_transfer_loss_max: value `1.2513192494710286`, threshold `6.75`, pass `True`
- events_event_3_transfer_loss_max: value `1.3664660851160686`, threshold `6.75`, pass `True`

## WG Compliance Notes

- WG1: data is procedural; manifest records no external sources.
- WG2: checkpoints include model, optimizer, sampler cursor, and RNG states; kill-and-resume test is committed.
- WG3: run ids are content hashes of committed config files.
- WG4: S3 model is pinned at d_model 256, 6 layers, 4 heads, with parameter count in range.
- WG5: persistence, unigram/frequency, and shuffled-input baselines are reported.
- WG7: core lint forbids modality conditionals in `wm/core`.
- WG9: scheduled eval JSON and probe CSV artifacts are written by the training harness.
- WG10/WG11: no RL and no AM imports were added.
- WG12: final archive is generated from committed `HEAD`.

## Honest Limitations

- Dataset v1 is intentionally small for a local bootstrap; it is not a capability claim beyond S1-S4 plumbing.
- Kill-test thresholds are provisional v1 calibration; failing rows remain reported instead of weakened.
- Checkpoints are not tracked because M0R explicitly gitignored `runs/` except metrics JSON/CSV artifacts.

## Next Stages

- S5+: improve mixed precision and throughput reporting now that the eval schedule is reusable.
- S6: only then consider the permitted scale-step study with before/after tables.
- S7: external data remains gated and must be licensed/manifested before use.
