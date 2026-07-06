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

EVENT_V2_FAMILIES = {
    "eventv2_0": {
        "sources": ["grid.sim", "agent.ctrl", "door.sys", "trace.core"],
        "verbs": ["move", "sense", "wait", "open", "block", "mark", "route", "score"],
        "objects": ["cell", "gate", "key", "exit", "path", "wall"],
    },
    "eventv2_1": {
        "sources": ["energy.sys", "hazard.map", "battery.mod", "field.scan"],
        "verbs": ["charge", "drain", "warn", "clear", "pulse", "cool", "heat", "stabilize"],
        "objects": ["meter", "hazard", "cell", "coil", "field", "sensor"],
    },
    "eventv2_2": {
        "sources": ["inventory.buf", "key.inv", "cache.mem", "lock.reg"],
        "verbs": ["pickup", "drop", "store", "release", "match", "reject", "bind", "unseal"],
        "objects": ["token", "key", "lock", "cache", "slot", "barrier"],
    },
    "eventv2_3": {
        "sources": ["eval.probe", "trace.log", "metric.bus", "mask.plan"],
        "verbs": ["emit", "mask", "score", "sample", "compare", "flag", "copy", "audit"],
        "objects": ["probe", "trace", "metric", "span", "field", "row"],
    },
    "eventv2_4": {
        "sources": ["route.net", "port.link", "packet.bus", "gate.ctrl"],
        "verbs": ["send", "receive", "forward", "block", "allow", "echo", "merge", "split"],
        "objects": ["packet", "port", "route", "channel", "signal", "frame"],
    },
    "eventv2_5": {
        "sources": ["grammar.gen", "token.flow", "byte.seq", "parse.tab"],
        "verbs": ["shift", "reduce", "append", "close", "open", "label", "count", "align"],
        "objects": ["token", "span", "rule", "byte", "node", "edge"],
    },
    "eventv2_6": {
        "sources": ["world.tick", "state.diff", "rule.seed", "sensor.grid"],
        "verbs": ["tick", "diff", "flip", "observe", "update", "freeze", "thaw", "reset"],
        "objects": ["state", "rule", "sensor", "grid", "agent", "clock"],
    },
    "eventv2_7": {
        "sources": ["qa.closed", "answer.buf", "context.win", "abstain.head"],
        "verbs": ["ask", "answer", "cite", "abstain", "copy", "rank", "verify", "deny"],
        "objects": ["query", "answer", "context", "span", "source", "claim"],
    },
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


def event_v2_record(seed: int, family: str, index: int, rng: random.Random, previous_id: str | None) -> dict[str, Any]:
    spec = EVENT_V2_FAMILIES[family]
    verb = rng.choice(spec["verbs"])
    source = rng.choice(spec["sources"])
    event_id = f"{family}-{seed:08d}-{index:04d}"
    value = rng.randint(-5, 15)
    status = rng.choice(["ok", "hold", "warn", "deny"])
    return {
        "version": "EG-1",
        "event_id": event_id,
        "ts": index,
        "source": source,
        "verb": verb,
        "object": rng.choice(spec["objects"]),
        "value": value,
        "status": status,
        "prev": previous_id,
        "meta": {
            "family": family,
            "seed": seed,
            "index": index,
            "phase": index % 4,
            "bucket": value // 3,
        },
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


def generate_event_v2_sample(seed: int, family: str, split: str, length: int | None = None) -> dict[str, Any]:
    rng = random.Random(seed)
    if length is None:
        length = rng.randint(8, 16)
    events = []
    previous_id = None
    for i in range(length):
        ev = event_v2_record(seed, family, i, rng, previous_id)
        events.append(ev)
        previous_id = ev["event_id"]
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
            "objects": [ev["object"] for ev in events],
            "version": "events-v2",
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


def validate_event_v2_stream(row: dict[str, Any]) -> bool:
    if row["meta"].get("version") != "events-v2":
        return False
    lines = row["tokens"].splitlines()
    if len(lines) != row["meta"]["length"]:
        return False
    previous_id = None
    for i, line in enumerate(lines):
        ev = json.loads(line)
        if ev.get("version") != "EG-1":
            return False
        for key in ["event_id", "ts", "source", "verb", "object", "value", "status", "prev", "meta"]:
            if key not in ev:
                return False
        if ev["ts"] != i or ev["prev"] != previous_id:
            return False
        previous_id = ev["event_id"]
    return True
