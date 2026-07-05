from __future__ import annotations

import json
import random
from typing import Any

from wm.data.schema import sample


EVENT_FAMILIES = {
    "event_0": {"sources": ["grid.sim", "agent.ctrl"], "verbs": ["move", "sense", "wait"]},
    "event_1": {"sources": ["door.sys", "key.inv"], "verbs": ["pickup", "open", "drop"]},
    "event_2": {"sources": ["energy.sys", "hazard.map"], "verbs": ["charge", "drain", "mark"]},
    "event_3": {"sources": ["eval.probe", "trace.log"], "verbs": ["emit", "mask", "score"]},
}


def event_record(seed: int, family: str, index: int, rng: random.Random) -> dict[str, Any]:
    spec = EVENT_FAMILIES[family]
    verb = rng.choice(spec["verbs"])
    source = rng.choice(spec["sources"])
    return {
        "version": "EG-1",
        "event_id": f"{family}-{seed:06d}-{index:03d}",
        "ts": index,
        "source": source,
        "verb": verb,
        "object": f"obj_{rng.randint(0, 5)}",
        "value": rng.randint(0, 9),
        "meta": {"family": family, "seed": seed, "index": index},
    }


def generate_event_sample(seed: int, family: str, split: str, length: int = 12) -> dict[str, Any]:
    rng = random.Random(seed)
    events = [event_record(seed, family, i, rng) for i in range(length)]
    jsonl = "\n".join(json.dumps(ev, sort_keys=True, separators=(",", ":")) for ev in events)
    return sample(
        "events",
        tokens=jsonl,
        target=jsonl[1:] + "\n",
        meta={
            "seed": seed,
            "family": family,
            "split": split,
            "length": length,
            "verbs": [ev["verb"] for ev in events],
            "sources": [ev["source"] for ev in events],
        },
    )


def validate_event_stream(row: dict[str, Any]) -> bool:
    lines = row["tokens"].splitlines()
    if len(lines) != row["meta"]["length"]:
        return False
    for i, line in enumerate(lines):
        ev = json.loads(line)
        if ev.get("version") != "EG-1":
            return False
        for key in ["event_id", "ts", "source", "verb", "object", "value", "meta"]:
            if key not in ev:
                return False
        if ev["ts"] != i:
            return False
    return True

