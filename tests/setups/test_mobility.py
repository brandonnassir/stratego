"""Initial-mobility library-quality rule: engine delegation and behaviour."""

import inspect
import random

from stratego.engine.constants import (
    BLUE,
    BOMB,
    FLAG,
    RED,
    TERMINAL_BOTH_NO_LEGAL_MOVE_DRAW,
)
from stratego.engine.legal_moves import has_legal_action
from stratego.engine.setup import random_setup
from stratego.engine.state import create_game
from stratego.setups import mobility
from stratego.setups.identity import canonical_index, orient_setup
from stratego.setups.mobility import setup_has_initial_mobility
from stratego.setups.traits import OPEN_FRONT_FILES


def _stranded_setup() -> tuple[int, ...]:
    """A legal setup with no initial legal move for its owner.

    Every open front-rank file (the six files whose forward square is not a
    lake) holds an immovable piece; the four lake-facing files are blocked by
    geometry regardless, and every non-front piece is boxed in by its own
    army. Built by swapping the six Bombs and nothing else, so inventory
    legality is untouched.
    """
    rng = random.Random(424242)
    cells = list(random_setup(rng))
    front = [canonical_index(3, file) for file in OPEN_FRONT_FILES]
    bomb_positions = [index for index, piece in enumerate(cells) if piece == BOMB]
    for target, source in zip(front, bomb_positions):
        cells[source], cells[target] = cells[target], cells[source]
    # The swap order can shuffle bombs already on the front rank; force the
    # final state explicitly and verify.
    for target in front:
        if cells[target] != BOMB:
            donor = next(
                index
                for index, piece in enumerate(cells)
                if piece == BOMB and index not in front
            )
            cells[donor], cells[target] = cells[target], cells[donor]
    assert all(cells[target] == BOMB for target in front)
    return tuple(cells)


def test_random_legal_setups_are_usually_mobile():
    rng = random.Random(20260813)
    verdicts = [setup_has_initial_mobility(random_setup(rng)) for _ in range(50)]
    # Stranding needs all six open front files immovable: ~1 in 548,340.
    assert all(verdicts)


def test_stranded_setup_is_detected():
    assert not setup_has_initial_mobility(_stranded_setup())


def test_verdict_matches_engine_has_legal_action_directly():
    rng = random.Random(99)
    samples = [random_setup(rng) for _ in range(10)] + [_stranded_setup()]
    for canonical in samples:
        state = create_game(
            orient_setup(canonical, RED),
            orient_setup(canonical, BLUE),
            game_id="delegation-check",
        )
        assert setup_has_initial_mobility(canonical) == has_legal_action(state, RED)


def test_stranded_mirror_game_is_terminal_at_creation_under_engine_1_2_0():
    canonical = _stranded_setup()
    state = create_game(
        orient_setup(canonical, RED),
        orient_setup(canonical, BLUE),
        game_id="stranded-mirror",
    )
    # Both mirrored copies are stranded, so the corrected engine must already
    # declare the both-stranded draw at creation; the library check and the
    # engine's own initial-terminal rule agree on the same authority.
    assert state.terminal
    assert state.terminal_reason == TERMINAL_BOTH_NO_LEGAL_MOVE_DRAW


def test_mobility_is_reflection_invariant():
    from stratego.setups.identity import reflect_canonical

    rng = random.Random(5)
    samples = [random_setup(rng) for _ in range(10)] + [_stranded_setup()]
    for canonical in samples:
        assert setup_has_initial_mobility(canonical) == setup_has_initial_mobility(
            reflect_canonical(canonical)
        )


def test_mobility_module_contains_no_competing_movement_implementation():
    # The library-quality rule must delegate to the frozen engine's legality
    # authority: the module may not import or reimplement movement geometry.
    source = inspect.getsource(mobility)
    assert "has_legal_action" in source
    assert "create_game" in source
    for forbidden in ("NEIGHBOURS", "RAYS", "generate_actions"):
        assert forbidden not in source


def test_mobility_uses_flag_and_bomb_constants_not_ordinals():
    source = inspect.getsource(mobility)
    # The module never hand-tests piece types at all; movability decisions
    # belong to the engine.
    assert "== 10" not in source and "== 11" not in source
    assert FLAG == 10 and BOMB == 11  # documents what the check guards against
