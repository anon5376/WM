from __future__ import annotations

import random
from typing import Any

from wm.data.schema import sample


GRAMMAR_FAMILIES = {
    "grammar_0": {
        "agents": ["agent", "scout"],
        "verbs": ["moves", "waits", "opens"],
        "objects": ["key", "gate", "exit"],
        "rels": ["left_of", "right_of"],
        "nums": ["zero", "one", "two", "three"],
    },
    "grammar_1": {
        "agents": ["bot", "walker"],
        "verbs": ["steps", "rests", "unlocks"],
        "objects": ["orb", "door", "goal"],
        "rels": ["above", "below"],
        "nums": ["nil", "single", "pair", "triple"],
    },
    "grammar_2": {
        "agents": ["unit", "runner"],
        "verbs": ["slides", "holds", "clears"],
        "objects": ["token", "barrier", "portal"],
        "rels": ["near", "far_from"],
        "nums": ["n0", "n1", "n2", "n3"],
    },
    "grammar_3": {
        "agents": ["drone", "mover"],
        "verbs": ["turns", "idles", "releases"],
        "objects": ["crystal", "lock", "finish"],
        "rels": ["north_of", "south_of"],
        "nums": ["z", "u", "d", "t"],
    },
}

TEXT_V2_FAMILIES = {
    "textv2_0": {
        "agents": ["agent", "scout", "runner", "mapper", "keeper", "pilot"],
        "verbs": ["moves", "waits", "opens", "marks", "checks", "carries"],
        "objects": ["key", "gate", "exit", "cell", "lever", "beacon", "tile", "cache"],
        "rels": ["left_of", "right_of", "above", "below", "near", "far_from"],
        "states": ["clear", "blocked", "charged", "empty", "locked", "open"],
        "reasons": ["path_clear", "signal_seen", "energy_low", "rule_match"],
    },
    "textv2_1": {
        "agents": ["bot", "walker", "seeker", "carrier", "reader", "unit"],
        "verbs": ["steps", "rests", "unlocks", "scans", "stores", "aligns"],
        "objects": ["orb", "door", "goal", "slot", "panel", "marker", "bridge", "crate"],
        "rels": ["north_of", "south_of", "east_of", "west_of", "touching", "apart_from"],
        "states": ["lit", "dark", "stable", "unstable", "ready", "spent"],
        "reasons": ["door_ready", "marker_found", "timer_done", "load_safe"],
    },
    "textv2_2": {
        "agents": ["unit", "runner", "drone", "mover", "sentinel", "actor"],
        "verbs": ["slides", "holds", "clears", "turns", "idles", "releases"],
        "objects": ["token", "barrier", "portal", "crystal", "lock", "finish", "node", "wire"],
        "rels": ["before", "after", "inside", "outside", "adjacent_to", "separate_from"],
        "states": ["cold", "warm", "active", "silent", "valid", "invalid"],
        "reasons": ["token_seen", "barrier_open", "phase_shift", "trace_valid"],
    },
    "textv2_3": {
        "agents": ["probe", "tracer", "worker", "clerk", "guide", "solver"],
        "verbs": ["records", "copies", "links", "tests", "counts", "routes"],
        "objects": ["record", "trace", "event", "field", "label", "packet", "queue", "index"],
        "rels": ["matches", "differs_from", "precedes", "follows", "contains", "omits"],
        "states": ["true", "false", "known", "unknown", "clean", "noisy"],
        "reasons": ["probe_pass", "trace_match", "label_known", "field_clean"],
    },
    "textv2_4": {
        "agents": ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"],
        "verbs": ["visits", "guards", "measures", "updates", "rejects", "accepts"],
        "objects": ["sector", "sensor", "vault", "switch", "token", "lane", "well", "tower"],
        "rels": ["closer_than", "farther_than", "parallel_to", "crossing", "beside", "between"],
        "states": ["low", "mid", "high", "red", "green", "blue"],
        "reasons": ["range_ok", "color_match", "sector_seen", "switch_safe"],
    },
    "textv2_5": {
        "agents": ["iris", "jax", "kilo", "luma", "mira", "nox"],
        "verbs": ["pushes", "pulls", "reads", "writes", "raises", "lowers"],
        "objects": ["flag", "anchor", "meter", "glyph", "button", "coil", "ring", "shaft"],
        "rels": ["same_as", "other_than", "aligned_with", "offset_from", "nested_in", "paired_with"],
        "states": ["small", "large", "thin", "wide", "heavy", "light"],
        "reasons": ["glyph_pair", "meter_low", "ring_set", "anchor_free"],
    },
    "textv2_6": {
        "agents": ["opal", "quinn", "rhea", "sable", "taro", "uma"],
        "verbs": ["selects", "skips", "merges", "splits", "sorts", "filters"],
        "objects": ["list", "bucket", "shard", "entry", "token", "sample", "window", "frame"],
        "rels": ["before", "after", "equal_to", "less_than", "greater_than", "inside"],
        "states": ["sorted", "mixed", "first", "last", "odd", "even"],
        "reasons": ["bucket_full", "sample_valid", "window_open", "frame_seen"],
    },
    "textv2_7": {
        "agents": ["voss", "wren", "xara", "yori", "zen", "nova"],
        "verbs": ["hears", "sends", "receives", "blocks", "allows", "echoes"],
        "objects": ["pulse", "message", "gate", "channel", "signal", "code", "route", "port"],
        "rels": ["upstream_of", "downstream_of", "linked_to", "blocked_by", "allowed_by", "echoes"],
        "states": ["muted", "loud", "open", "closed", "fresh", "stale"],
        "reasons": ["pulse_high", "code_match", "port_open", "route_clear"],
    },
}


def generate_text_sample(seed: int, family: str, split: str) -> dict[str, Any]:
    rng = random.Random(seed)
    vocab = GRAMMAR_FAMILIES[family]
    if rng.random() < 0.5:
        agent = rng.choice(vocab["agents"])
        verb = rng.choice(vocab["verbs"])
        obj = rng.choice(vocab["objects"])
        x = rng.randint(0, 8)
        y = rng.randint(0, 6)
        text = f"{agent} {verb} {obj} at {x} {y} ."
        derivation = ["S", "WORLD_LOG", "AGENT VERB OBJECT at INT INT ."]
        nonterminal = "WORLD_LOG"
    else:
        left = rng.choice(vocab["objects"])
        rel = rng.choice(vocab["rels"])
        right = rng.choice(vocab["objects"])
        text = f"{left} is {rel} {right} ."
        derivation = ["S", "REL_SENT", "OBJECT is REL OBJECT ."]
        nonterminal = "REL_SENT"
    return sample(
        "text",
        tokens=text,
        target=text[1:] + "\n",
        meta={
            "seed": seed,
            "family": family,
            "split": split,
            "nonterminal": nonterminal,
            "derivation": derivation,
            "vocab": vocab,
        },
    )


def generate_text_v2_sample(seed: int, family: str, split: str) -> dict[str, Any]:
    rng = random.Random(seed)
    vocab = TEXT_V2_FAMILIES[family]
    template_id = rng.randrange(5)
    if template_id == 0:
        agent = rng.choice(vocab["agents"])
        verb = rng.choice(vocab["verbs"])
        obj = rng.choice(vocab["objects"])
        x0, y0 = rng.randint(0, 8), rng.randint(0, 6)
        x1, y1 = rng.randint(0, 8), rng.randint(0, 6)
        reason = rng.choice(vocab["reasons"])
        text = f"{agent} {verb} {obj} from {x0} {y0} to {x1} {y1} because {reason} ."
        nonterminal = "MOVE_REASON"
        derivation = ["S", "MOVE_REASON", "AGENT VERB OBJECT from INT INT to INT INT because REASON ."]
    elif template_id == 1:
        left = rng.choice(vocab["objects"])
        middle = rng.choice([obj for obj in vocab["objects"] if obj != left])
        right = rng.choice(vocab["objects"])
        rel_a = rng.choice(vocab["rels"])
        rel_b = rng.choice(vocab["rels"])
        text = f"{left} is {rel_a} {middle} and {middle} is {rel_b} {right} ."
        nonterminal = "REL_CHAIN"
        derivation = ["S", "REL_CHAIN", "OBJECT is REL OBJECT and OBJECT is REL OBJECT ."]
    elif template_id == 2:
        obj = rng.choice(vocab["objects"])
        state = rng.choice(vocab["states"])
        agent = rng.choice(vocab["agents"])
        verb = rng.choice(vocab["verbs"])
        text = f"if {obj} is {state} then {agent} should {verb} ."
        nonterminal = "CONDITION"
        derivation = ["S", "CONDITION", "if OBJECT is STATE then AGENT should VERB ."]
    elif template_id == 3:
        agent = rng.choice(vocab["agents"])
        obj = rng.choice(vocab["objects"])
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        text = f"{agent} counts {a} {obj} plus {b} {obj} equals {a + b} ."
        nonterminal = "ARITH"
        derivation = ["S", "ARITH", "AGENT counts INT OBJECT plus INT OBJECT equals INT ."]
    else:
        agent = rng.choice(vocab["agents"])
        obj = rng.choice(vocab["objects"])
        state_a = rng.choice(vocab["states"])
        state_b = rng.choice([state for state in vocab["states"] if state != state_a])
        reason = rng.choice(vocab["reasons"])
        text = f"{agent} changes {obj} from {state_a} to {state_b} after {reason} ."
        nonterminal = "STATE_CHANGE"
        derivation = ["S", "STATE_CHANGE", "AGENT changes OBJECT from STATE to STATE after REASON ."]
    return sample(
        "text",
        tokens=text,
        target=text[1:] + "\n",
        meta={
            "seed": seed,
            "family": family,
            "split": split,
            "nonterminal": nonterminal,
            "derivation": derivation,
            "vocab": vocab,
            "version": "text-v2",
        },
    )


def validate_text_sample(row: dict[str, Any]) -> bool:
    meta = row["meta"]
    vocab = GRAMMAR_FAMILIES[meta["family"]]
    words = row["tokens"].split()
    if meta["nonterminal"] == "WORLD_LOG":
        return (
            len(words) == 7
            and words[0] in vocab["agents"]
            and words[1] in vocab["verbs"]
            and words[2] in vocab["objects"]
            and words[3] == "at"
            and words[6] == "."
        )
    if meta["nonterminal"] == "REL_SENT":
        return (
            len(words) == 5
            and words[0] in vocab["objects"]
            and words[1] == "is"
            and words[2] in vocab["rels"]
            and words[3] in vocab["objects"]
            and words[4] == "."
        )
    return False


def validate_text_v2_sample(row: dict[str, Any]) -> bool:
    meta = row["meta"]
    if meta.get("version") != "text-v2":
        return False
    vocab = TEXT_V2_FAMILIES[meta["family"]]
    words = row["tokens"].split()
    if meta["nonterminal"] == "MOVE_REASON":
        return (
            len(words) == 12
            and words[0] in vocab["agents"]
            and words[1] in vocab["verbs"]
            and words[2] in vocab["objects"]
            and words[3] == "from"
            and words[6] == "to"
            and words[9] == "because"
            and words[10] in vocab["reasons"]
            and words[11] == "."
        )
    if meta["nonterminal"] == "REL_CHAIN":
        return len(words) == 9 and words[1] == "is" and words[4] == "and" and words[6] == "is" and words[8] == "."
    if meta["nonterminal"] == "CONDITION":
        return len(words) == 9 and words[0] == "if" and words[2] == "is" and words[4] == "then" and words[6] == "should" and words[8] == "."
    if meta["nonterminal"] == "ARITH":
        return len(words) == 10 and words[1] == "counts" and words[4] == "plus" and words[7] == "equals" and words[9] == "."
    if meta["nonterminal"] == "STATE_CHANGE":
        return len(words) == 10 and words[1] == "changes" and words[3] == "from" and words[5] == "to" and words[7] == "after" and words[9] == "."
    return False
