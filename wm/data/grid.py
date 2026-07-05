from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

import numpy as np

from wm.data.schema import sample


WIDTH = 9
HEIGHT = 7
VIEW_W = 9
VIEW_H = 7
ENERGY_START = 10
TICK_CAP = 200
ACTIONS = ["N", "S", "E", "W", "PickUp", "Drop", "Open", "Wait"]
SHAPES = ["wall", "key", "barrier", "consumable", "hazard", "exit"]
SHAPE_TO_CHANNEL = {name: i for i, name in enumerate(SHAPES)}
GRID_CHANNELS = 6 + 1 + 6 + 1 + 1


@dataclass
class Entity:
    shape: str
    klass: int = 0


@dataclass
class GridWorld:
    rule_seed: int
    rng_seed: int = 0
    width: int = WIDTH
    height: int = HEIGHT
    self_pos: tuple[int, int] = (1, 1)
    held_key_class: int | None = None
    energy: int = ENERGY_START
    tick: int = 0
    done: bool = False
    grid: dict[tuple[int, int], Entity] = field(default_factory=dict)
    last_blocked: bool = False

    def __post_init__(self) -> None:
        self.rng = random.Random(self.rng_seed)
        self.rule_table = rule_table(self.rule_seed)
        if not self.grid:
            self.grid = self.default_grid()

    def default_grid(self) -> dict[tuple[int, int], Entity]:
        grid: dict[tuple[int, int], Entity] = {}
        for x in range(self.width):
            grid[(x, 0)] = Entity("wall")
            grid[(x, self.height - 1)] = Entity("wall")
        for y in range(self.height):
            grid[(0, y)] = Entity("wall")
            grid[(self.width - 1, y)] = Entity("wall")
        grid[(2, 1)] = Entity("key", klass=0)
        grid[(2, 4)] = Entity("key", klass=1)
        grid[(4, 2)] = Entity("consumable")
        grid[(4, 4)] = Entity("hazard")
        grid[(6, 5)] = Entity("barrier", klass=self.rule_table[0])
        grid[(7, 5)] = Entity("exit")
        return grid

    def entity_at(self, pos: tuple[int, int]) -> Entity | None:
        return self.grid.get(pos)

    def is_blocking(self, pos: tuple[int, int]) -> bool:
        ent = self.entity_at(pos)
        return bool(ent and ent.shape in {"wall", "barrier"})

    def move_delta(self, action: str) -> tuple[int, int]:
        return {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}[action]

    def adjacent_positions(self) -> list[tuple[int, int]]:
        x, y = self.self_pos
        return [(x, y - 1), (x, y + 1), (x + 1, y), (x - 1, y)]

    def choose_action(self) -> str:
        interact = []
        here = self.entity_at(self.self_pos)
        if here and here.shape in {"key", "consumable"}:
            interact.append("PickUp")
        for pos in self.adjacent_positions():
            ent = self.entity_at(pos)
            if ent and ent.shape == "barrier":
                interact.append("Open")
        if interact and self.rng.random() < 0.15:
            return self.rng.choice(interact)
        return self.rng.choice(ACTIONS)

    def step(self, action: str) -> dict[str, Any]:
        if self.done:
            return self.state(action, event="already_done")
        self.last_blocked = False
        event = "tick"
        if action in {"N", "S", "E", "W"}:
            dx, dy = self.move_delta(action)
            target = (self.self_pos[0] + dx, self.self_pos[1] + dy)
            if self.is_blocking(target):
                self.last_blocked = True
                event = "blocked"
            else:
                self.self_pos = target
                event = "moved"
        elif action == "PickUp":
            event = self.pick_up()
        elif action == "Drop":
            event = self.drop()
        elif action == "Open":
            event = self.open_barrier()
        elif action == "Wait":
            event = "waited"
        else:
            raise ValueError(f"unknown action {action}")

        here = self.entity_at(self.self_pos)
        if here and here.shape == "hazard":
            self.energy -= 2
            event += "+hazard"
        if here and here.shape == "exit":
            self.done = True
            event += "+exit"
        if self.energy <= 0:
            self.done = True
            event += "+dead"
        self.tick += 1
        if self.tick >= TICK_CAP:
            self.done = True
            event += "+cap"
        return self.state(action, event=event)

    def pick_up(self) -> str:
        ent = self.entity_at(self.self_pos)
        if not ent:
            return "pickup_empty"
        if ent.shape == "key":
            self.held_key_class = ent.klass
            del self.grid[self.self_pos]
            return "pickup_key"
        if ent.shape == "consumable":
            self.energy += 3
            del self.grid[self.self_pos]
            return "pickup_consumable"
        return "pickup_ignored"

    def drop(self) -> str:
        if self.held_key_class is None or self.self_pos in self.grid:
            return "drop_ignored"
        self.grid[self.self_pos] = Entity("key", klass=self.held_key_class)
        self.held_key_class = None
        return "drop_key"

    def open_barrier(self) -> str:
        for pos in self.adjacent_positions():
            ent = self.entity_at(pos)
            if ent and ent.shape == "barrier":
                if self.held_key_class is not None and self.rule_table[self.held_key_class] == ent.klass:
                    del self.grid[pos]
                    return "open_barrier"
                self.last_blocked = True
                return "open_failed"
        return "open_none"

    def state(self, action: str, event: str) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "pos": list(self.self_pos),
            "energy": self.energy,
            "held_key_class": self.held_key_class,
            "action": action,
            "event": event,
            "blocked": self.last_blocked,
            "done": self.done,
        }

    def observe(self) -> list[list[list[float]]]:
        arr = np.zeros((GRID_CHANNELS, VIEW_H, VIEW_W), dtype=np.float32)
        cx, cy = VIEW_W // 2, VIEW_H // 2
        sx, sy = self.self_pos
        for oy in range(VIEW_H):
            for ox in range(VIEW_W):
                wx = sx + ox - cx
                wy = sy + oy - cy
                ent = self.grid.get((wx, wy))
                if wx < 0 or wx >= self.width or wy < 0 or wy >= self.height:
                    ent = Entity("wall")
                if ent and ent.shape in SHAPE_TO_CHANNEL:
                    arr[SHAPE_TO_CHANNEL[ent.shape], oy, ox] = 1.0
        arr[6, cy, cx] = 1.0
        if self.held_key_class is not None:
            arr[7 + self.held_key_class, :, :] = 1.0
        arr[13, :, :] = min(self.energy, 20) / 20.0
        arr[14, :, :] = 1.0 if self.last_blocked else 0.0
        return arr.tolist()


def rule_table(rule_seed: int, classes: int = 2) -> dict[int, int]:
    rng = random.Random(rule_seed)
    barriers = list(range(classes))
    rng.shuffle(barriers)
    return {key_class: barriers[key_class] for key_class in range(classes)}


def generate_grid_episode(seed: int, rule_seed: int, split: str, max_steps: int = 48) -> dict[str, Any]:
    world = GridWorld(rule_seed=rule_seed, rng_seed=seed)
    frames = []
    states = []
    actions = []
    for _ in range(max_steps):
        frames.append(world.observe())
        action = world.choose_action()
        actions.append(ACTIONS.index(action))
        states.append(world.step(action))
        if world.done:
            break
    return sample(
        "grid",
        tensor=frames,
        target=actions,
        meta={
            "seed": seed,
            "rule_seed": rule_seed,
            "family": f"rule_seed_{rule_seed}",
            "split": split,
            "width": WIDTH,
            "height": HEIGHT,
            "channels": GRID_CHANNELS,
            "rule_table": rule_table(rule_seed),
            "states": states,
        },
    )
