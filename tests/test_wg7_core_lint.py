from pathlib import Path


FORBIDDEN_CORE_SNIPPETS = [
    "if modality",
    "elif modality",
    "modality ==",
    "modality !=",
    "modality in",
    "MODALITY_IDS",
]


def test_shared_core_has_no_modality_conditional_branches():
    core_files = sorted(Path("wm/core").glob("*.py"))
    assert core_files
    for path in core_files:
        text = path.read_text()
        for snippet in FORBIDDEN_CORE_SNIPPETS:
            assert snippet not in text, f"{snippet!r} found in {path}"

