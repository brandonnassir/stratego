"""Phase 17 Agent 2: sampled-not-argmax selection (gate G-C check C3)."""

from __future__ import annotations

import numpy as np
import pytest

from stratego.training.phase17.move_contract import game_id
from stratego.training.phase17.move_snapshot import (
    Phase17SeatingError,
    action_seed,
    action_sampling_uniform,
    reproduce_sample,
    sample_legal_action,
)

GAME = game_id("RUN-TEST-A", 0, 0)


def test_selection_is_not_argmax_over_many_decisions():
    """The distribution's most likely action is not always the one taken.

    Argmax would return action 44 every time. A categorical sample must
    produce every action with positive probability, and the empirical
    frequencies must track the distribution rather than merely differ from a
    constant.
    """
    probabilities = (0.1, 0.2, 0.7)
    legal = (3, 9, 44)
    draws = [
        sample_legal_action(probabilities, legal, GAME, ply)["action"]
        for ply in range(4000)
    ]
    counts = {action: draws.count(action) for action in legal}
    assert set(counts) == set(legal)
    assert all(count > 0 for count in counts.values())
    for action, probability in zip(legal, probabilities):
        assert counts[action] / len(draws) == pytest.approx(probability, abs=0.025)


def test_a_near_uniform_distribution_spreads_over_a_large_legal_set():
    legal = tuple(range(0, 400, 4))
    probabilities = tuple([1.0 / len(legal)] * len(legal))
    draws = {
        sample_legal_action(probabilities, legal, GAME, ply)["action"]
        for ply in range(1500)
    }
    assert len(draws) > len(legal) * 0.9


def test_the_draw_replays_from_the_game_and_ply_alone():
    probabilities = (0.25, 0.25, 0.5)
    legal = (1, 5, 7)
    first = sample_legal_action(probabilities, legal, GAME, 12)
    second = sample_legal_action(probabilities, legal, GAME, 12)
    assert first == second
    assert first["seed"] == action_seed(GAME, 12)
    assert first["uniform"] == pytest.approx(action_sampling_uniform(GAME, 12))


def test_the_draw_replays_from_the_stored_distribution_and_seed_alone():
    """A stored row must reproduce its own action without the game or the model."""
    probabilities = (0.05, 0.15, 0.3, 0.5)
    legal = (2, 8, 30, 91)
    for ply in range(200):
        draw = sample_legal_action(probabilities, legal, GAME, ply)
        assert reproduce_sample(probabilities, legal, seed=draw["seed"]) == draw["action"]
        assert legal[draw["index"]] == draw["action"]


def test_different_plies_and_games_draw_differently():
    probabilities = tuple([1.0 / 50] * 50)
    legal = tuple(range(50))
    by_ply = {
        sample_legal_action(probabilities, legal, GAME, ply)["action"]
        for ply in range(200)
    }
    other = game_id("RUN-TEST-A", 1, 0)
    by_game = {
        sample_legal_action(probabilities, legal, other, ply)["action"]
        for ply in range(200)
    }
    assert len(by_ply) > 20
    assert by_ply != by_game


def test_a_non_ascending_legal_list_is_refused():
    with pytest.raises(Phase17SeatingError, match="ascending"):
        sample_legal_action((0.5, 0.5), (9, 3), GAME, 0)


def test_a_length_mismatch_is_refused():
    with pytest.raises(Phase17SeatingError, match="probabilities for"):
        sample_legal_action((0.5, 0.5), (1, 2, 3), GAME, 0)


def test_an_empty_legal_set_is_refused():
    with pytest.raises(Phase17SeatingError, match="empty legal set"):
        sample_legal_action((), (), GAME, 0)


def test_a_float32_tail_shortfall_takes_the_last_legal_action():
    """The accepted rule: a distribution summing just under 1 never falls through."""
    probabilities = (0.5, 0.4999999)
    legal = (4, 6)
    seen = {
        sample_legal_action(probabilities, legal, GAME, ply)["action"]
        for ply in range(500)
    }
    assert seen <= {4, 6}
    assert 6 in seen


def test_the_uniform_stream_is_uniform():
    values = np.asarray(
        [action_sampling_uniform(GAME, ply) for ply in range(5000)], dtype=np.float64
    )
    assert float(values.min()) >= 0.0
    assert float(values.max()) < 1.0
    assert float(values.mean()) == pytest.approx(0.5, abs=0.02)
