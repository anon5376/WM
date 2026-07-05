from __future__ import annotations

import math
import random
from typing import Any

from wm.data.schema import sample


SIGNAL_FAMILIES = {
    "signal_0": {"freqs": [1.0, 3.0], "band": "low"},
    "signal_1": {"freqs": [2.0, 5.0], "band": "mid"},
    "signal_2": {"freqs": [4.0, 7.0], "band": "high"},
    "signal_3": {"freqs": [6.0, 9.0], "band": "upper"},
}


def generate_signal_sample(seed: int, family: str, split: str, length: int = 96) -> dict[str, Any]:
    rng = random.Random(seed)
    spec = SIGNAL_FAMILIES[family]
    phase = rng.random() * math.tau
    amp = 0.7 + rng.random() * 0.4
    marker_positions = [length // 4, length // 2, (3 * length) // 4]
    frames: list[list[float]] = []
    for t in range(length):
        x = t / length
        value = 0.0
        for i, freq in enumerate(spec["freqs"]):
            value += amp / (i + 1) * math.sin(math.tau * freq * x + phase / (i + 1))
        marker = 1.0 if t in marker_positions else 0.0
        frames.append([round(value, 6), marker])
    target = [row[0] for row in frames[1:]] + [frames[-1][0]]
    return sample(
        "signal",
        tensor=frames,
        target=target,
        meta={
            "seed": seed,
            "family": family,
            "split": split,
            "freqs": spec["freqs"],
            "frequency_band": spec["band"],
            "marker_positions": marker_positions,
            "length": length,
        },
    )


def marker_positions(row: dict[str, Any]) -> list[int]:
    return [i for i, frame in enumerate(row["tensor"]) if frame[1] == 1.0]

