"""Deterministic seeded execution.

Covers `03_game_engine_spec.md` section 18 and section 19 of the Phase Two
instructions.
"""

import random

import pytest

from stratego.engine.observation import build_observation
from stratego.engine.random_play import (
    generate_random_games,
    make_random_setups,
    play_random_game,
    select_random_action,
)
from stratego.engine.state import state_fingerprint
from tests.helpers import nonterminal_state


@pytest.mark.parametrize("seed", range(6))
def test_the_same_seed_reproduces_the_same_game(seed):
    first_state, first_record = play_random_game(seed)
    second_state, second_record = play_random_game(seed)

    assert first_record == second_record
    assert state_fingerprint(first_state) == state_fingerprint(second_state)


def test_different_seeds_produce_different_games():
    records = [play_random_game(seed)[1] for seed in range(8)]
    action_sequences = {tuple(record.actions) for record in records}
    assert len(action_sequences) == len(records)


def test_setups_are_derived_deterministically_from_the_seed():
    assert make_random_setups(17) == make_random_setups(17)
    assert make_random_setups(17) != make_random_setups(18)


def test_random_action_selection_is_seeded():
    state = nonterminal_state(25)
    first = [select_random_action(state, random.Random(9)) for _ in range(5)]
    second = [select_random_action(state, random.Random(9)) for _ in range(5)]
    assert first == second


def test_generated_game_batch_is_reproducible():
    first = [record.to_json() for _, record in generate_random_games(5, base_seed=100)]
    second = [record.to_json() for _, record in generate_random_games(5, base_seed=100)]
    assert first == second


def test_observations_are_bitwise_reproducible():
    state = nonterminal_state(45)
    first = build_observation(state)
    second = build_observation(state)
    assert first.tobytes() == second.tobytes()


def test_replay_record_serialisation_round_trips():
    from stratego.engine.replay import ReplayRecord

    _, record = play_random_game(31)
    restored = ReplayRecord.from_json(record.to_json())
    assert restored == record
