from pathlib import Path

from wm.data.build_dataset import build_dataset
from wm.data.events import generate_event_sample, validate_event_stream
from wm.data.schema import read_jsonl
from wm.data.signal import generate_signal_sample, marker_positions
from wm.data.text import generate_text_sample, validate_text_sample


def test_text_grammar_derivation_is_valid():
    for family in ["grammar_0", "grammar_1", "grammar_2", "grammar_3"]:
        row = generate_text_sample(123, family, "train")
        assert row["modality"] == "text"
        assert validate_text_sample(row)
        assert row["meta"]["derivation"][0] == "S"


def test_signal_markers_at_parameterized_positions():
    row = generate_signal_sample(123, "signal_0", "train", length=32)
    assert marker_positions(row) == [8, 16, 24]
    assert row["meta"]["marker_positions"] == [8, 16, 24]
    assert row["tensor"][8][1] == 1.0


def test_event_stream_is_eg1_shape_valid():
    row = generate_event_sample(123, "event_0", "train", length=5)
    assert row["modality"] == "events"
    assert validate_event_stream(row)
    assert row["tokens"].splitlines()[0].startswith('{"event_id"')


def test_dataset_generation_is_byte_deterministic(tmp_path: Path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    manifest_one = build_dataset(one, samples_per_family=2)
    manifest_two = build_dataset(two, samples_per_family=2)
    assert manifest_one["counts"] == manifest_two["counts"]
    for shard in manifest_one["shards"]:
        rel = shard["path"]
        a = (one / rel).read_bytes()
        b = (two / rel).read_bytes()
        assert a == b
    assert len(manifest_one["splits"]["heldout_grammar_families"]) >= 2
    assert len(manifest_one["splits"]["heldout_signal_families"]) >= 2
    assert len(manifest_one["splits"]["heldout_grid_rule_seeds"]) >= 2
    rows = read_jsonl(one / "v1/train/text.jsonl")
    assert rows and rows[0]["modality"] == "text"

