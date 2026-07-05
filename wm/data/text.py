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

