"""Global planes: lake mask, game progress and no-battle progress (124-126).

Covers `07_observation_validation_matrix.md` section 10.
"""

import numpy as np
import pytest

from stratego.engine.constants import (
    BLUE,
    EVALUATION_RULES,
    LAKE_SQUARES,
    RED,
    TRAINING_RULES,
)
from stratego.engine.observation import (
    CH_BATTLELESS_PROGRESS,
    CH_GAME_PROGRESS,
    CH_LAKE_MASK,
    build_observation,
)
from tests.helpers import cell, make_position, nonterminal_state, play


def base_state(rules=TRAINING_RULES, **kwargs):
    return make_position(
        red={"a1": "captain", "j1": "flag"},
        blue={"a10": "captain", "j10": "flag"},
        acting_player=RED,
        rules=rules,
        **kwargs,
    )


# -- channel 124 ------------------------------------------------------------


def test_lake_mask_has_exactly_eight_ones():
    observation = build_observation(base_state(), RED)
    plane = observation[CH_LAKE_MASK]
    assert plane.sum() == 8.0
    assert int((plane == 0.0).sum()) == 92


def test_lake_mask_marks_the_correct_squares():
    observation = build_observation(base_state(), RED)
    for name in ("c5", "d5", "g5", "h5", "c6", "d6", "g6", "h6"):
        assert cell(observation, CH_LAKE_MASK, name) == 1.0
    for name in ("b5", "e5", "f5", "i5", "c4", "c7"):
        assert cell(observation, CH_LAKE_MASK, name) == 0.0


def test_lake_mask_is_static_and_perspective_invariant():
    reference = build_observation(base_state(), RED)[CH_LAKE_MASK]
    for ply in (0, 12, 55, 120):
        state = nonterminal_state(ply)
        for observer in (RED, BLUE):
            assert np.array_equal(build_observation(state, observer)[CH_LAKE_MASK], reference)


# -- channel 125 ------------------------------------------------------------


@pytest.mark.parametrize(
    "total_moves,expected", [(0, 0.0), (1000, 0.25), (2000, 0.5), (4000, 1.0)]
)
def test_game_progress_values(total_moves, expected):
    state = base_state()
    state.total_moves = total_moves
    observation = build_observation(state, RED)
    assert observation[CH_GAME_PROGRESS].min() == pytest.approx(expected)
    assert observation[CH_GAME_PROGRESS].max() == pytest.approx(expected)


def test_game_progress_is_spatially_constant_and_clamped():
    state = base_state()
    state.total_moves = 9_999
    plane = build_observation(state, RED)[CH_GAME_PROGRESS]
    assert np.allclose(plane, 1.0)


def test_game_progress_increases_monotonically():
    state = base_state()
    values = []
    for _ in range(6):
        values.append(float(build_observation(state, RED)[CH_GAME_PROGRESS][0, 0]))
        if state.acting_player == RED:
            play(state, "a1 a2" if state.board[0] is not None else "a2 a1")
        else:
            play(state, "a10 a9" if state.board[90] is not None else "a9 a10")
    assert values == sorted(values)
    assert len(set(values)) > 1


# -- channel 126 ------------------------------------------------------------


@pytest.mark.parametrize(
    "battleless,expected", [(0, 0.0), (50, 0.5), (99, 0.99), (100, 1.0), (150, 1.0)]
)
def test_no_battle_progress_under_the_training_configuration(battleless, expected):
    state = base_state(TRAINING_RULES)
    state.battleless_moves = battleless
    plane = build_observation(state, RED)[CH_BATTLELESS_PROGRESS]
    assert plane.min() == pytest.approx(expected)
    assert np.allclose(plane, plane[0, 0])


@pytest.mark.parametrize(
    "battleless,expected", [(0, 0.0), (100, 0.5), (199, 0.995), (200, 1.0)]
)
def test_no_battle_progress_under_the_evaluation_configuration(battleless, expected):
    state = base_state(EVALUATION_RULES)
    state.battleless_moves = battleless
    plane = build_observation(state, RED)[CH_BATTLELESS_PROGRESS]
    assert plane.min() == pytest.approx(expected)


def test_no_battle_progress_resets_to_zero_after_combat():
    state = make_position(
        red={"e3": "marshal", "a1": "flag"},
        blue={"e4": "captain", "j10": "flag", "j9": "scout"},
        acting_player=RED,
        battleless_moves=60,
    )
    before = build_observation(state, RED)[CH_BATTLELESS_PROGRESS][0, 0]
    assert before == pytest.approx(0.6)

    play(state, "e3 e4")
    after = build_observation(state, RED)[CH_BATTLELESS_PROGRESS]
    assert np.allclose(after, 0.0)


def test_no_battle_progress_climbs_again_after_a_reset():
    state = make_position(
        red={"e3": "marshal", "a1": "flag"},
        blue={"e4": "captain", "j10": "flag", "j9": "scout"},
        acting_player=RED,
        battleless_moves=60,
    )
    play(state, "e3 e4", "j9 i9")
    plane = build_observation(state, RED)[CH_BATTLELESS_PROGRESS]
    assert np.allclose(plane, 0.01)
