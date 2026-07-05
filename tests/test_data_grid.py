from wm.data.grid import Entity, GridWorld, rule_table


def test_wall_blocks_movement():
    world = GridWorld(rule_seed=11, self_pos=(1, 1))
    state = world.step("W")
    assert state["blocked"] is True
    assert state["pos"] == [1, 1]


def test_key_opens_matching_barrier_and_removes_it():
    rule_seed = 11
    table = rule_table(rule_seed)
    world = GridWorld(rule_seed=rule_seed, self_pos=(5, 5), grid={(6, 5): Entity("barrier", klass=table[0])})
    world.held_key_class = 0
    state = world.step("Open")
    assert state["event"] == "open_barrier"
    assert (6, 5) not in world.grid


def test_wrong_key_class_does_not_open_barrier():
    rule_seed = 11
    table = rule_table(rule_seed)
    wrong_key = 1 if table[1] != table[0] else 0
    world = GridWorld(rule_seed=rule_seed, self_pos=(5, 5), grid={(6, 5): Entity("barrier", klass=table[0])})
    world.held_key_class = wrong_key
    state = world.step("Open")
    assert state["blocked"] is True
    assert (6, 5) in world.grid


def test_consumable_hazard_and_exit_mechanics():
    world = GridWorld(
        rule_seed=11,
        self_pos=(2, 2),
        grid={(2, 2): Entity("consumable"), (3, 2): Entity("hazard"), (4, 2): Entity("exit")},
    )
    state = world.step("PickUp")
    assert state["event"] == "pickup_consumable"
    assert state["energy"] == 13
    assert (2, 2) not in world.grid
    state = world.step("E")
    assert state["pos"] == [3, 2]
    assert state["energy"] == 11
    assert "hazard" in state["event"]
    state = world.step("E")
    assert state["done"] is True
    assert "exit" in state["event"]


def test_observation_shape_and_broadcasts():
    world = GridWorld(rule_seed=11, self_pos=(1, 1))
    world.last_blocked = True
    obs = world.observe()
    assert len(obs) == 15
    assert len(obs[0]) == 7
    assert len(obs[0][0]) == 9
    assert obs[6][3][4] == 1.0
    assert obs[13][0][0] == 0.5
    assert obs[14][0][0] == 1.0

