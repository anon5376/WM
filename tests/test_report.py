import json
from pathlib import Path

from wm.eval.report import write_report


def test_report_writer_uses_evidence_files(tmp_path: Path):
    manifest = {
        "dataset": "demo",
        "counts": {"total_samples": 1},
        "total_bytes": 10,
        "external_data": [],
        "seeds": {"sample_seed_base": 1, "grid_train_rule_seeds": [11]},
        "splits": {"heldout_grammar_families": [], "heldout_signal_families": [], "heldout_grid_rule_seeds": []},
    }
    run = tmp_path / "run"
    final = tmp_path / "final"
    run.mkdir()
    (run / "eval").mkdir()
    (run / "probes").mkdir()
    final.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (run / "metadata.json").write_text(json.dumps({"run_id": "r", "config_hash": "h", "device": {"selected": "cpu", "reason": "test"}, "parameter_count": 1}))
    (run / "config.yaml").write_text("seed: 1\n")
    (run / "metrics.jsonl").write_text(json.dumps({"step": 1}) + "\n")
    (final / "metrics.json").write_text(json.dumps({"prediction": {"text": {"model_loss": 1, "persistence_loss": 2, "unigram_or_frequency_loss": 3, "shuffled_input_loss": 4}}}))
    (final / "probe_summary.csv").write_text("modality,probe,accuracy\ntext,p,0.5\n")
    (final / "killtest_table.csv").write_text("test,value,threshold,passed\nt,1,2,True\n")
    (tmp_path / "throughput.json").write_text(json.dumps({"checkpoint_count": 0, "eval_count": 0}))
    out = tmp_path / "report.md"
    write_report(
        manifest_path=tmp_path / "manifest.json",
        run_dir=run,
        final_battery_dir=final,
        throughput_path=tmp_path / "throughput.json",
        output_path=out,
    )
    text = out.read_text()
    assert "# BUILD REPORT WM N2R" in text
    assert "Prediction And Baselines" in text
