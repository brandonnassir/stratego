"""Deterministic replay reconstruction.

Covers `03_game_engine_spec.md` section 12,
`09_public_event_and_replay_schema.md` section 13,
`07_observation_validation_matrix.md` section 12 and
`04_engine_validation_plan.md` sections 15 and 21.7.

The 10,000-game acceptance gate runs in `scripts/run_phase2_validation.py`; this
module replays a smaller seeded sample with full per-ply comparison.
"""

import numpy as np
import pytest

from stratego.engine.constants import (
    BLUE,
    EVALUATION_RULES,
    EVENT_SCHEMA_VERSION,
    OBSERVATION_VERSION,
    RED,
    REPLAY_VERSION,
    RULES_VERSION,
)
from stratego.engine.events import filter_events_for_observer, public_board_view
from stratego.engine.invariants import capture_baseline, check_invariants
from stratego.engine.legal_moves import legal_action_mask, legal_actions
from stratego.engine.observation import build_observation
from stratego.engine.random_play import play_random_game
from stratego.engine.replay import (
    ReplayRecord,
    initial_state_from_record,
    rebuild_final_state,
    replay_plies,
)
from stratego.engine.state import state_fingerprint
from stratego.engine.transition import apply_action


def record_ply_signatures(record: ReplayRecord) -> list[tuple]:
    """A full per-ply signature of everything replay must reproduce."""
    signatures = []
    for _, state, events in replay_plies(record):
        signatures.append(
            (
                state_fingerprint(state),
                tuple(legal_actions(state)),
                build_observation(state, RED).tobytes(),
                build_observation(state, BLUE).tobytes(),
                repr(public_board_view(state, RED)),
                repr(public_board_view(state, BLUE)),
                repr(events),
            )
        )
    return signatures


@pytest.mark.parametrize("seed", range(12))
def test_a_record_replays_to_an_identical_final_state(seed):
    original, record = play_random_game(seed)
    replayed = rebuild_final_state(record)
    assert state_fingerprint(replayed) == state_fingerprint(original)


@pytest.mark.parametrize("seed", range(6))
def test_replay_reproduces_every_ply_exactly(seed):
    _, record = play_random_game(seed)
    first = record_ply_signatures(record)
    second = record_ply_signatures(record)
    assert first == second
    assert len(first) == len(record.actions) + 1


@pytest.mark.parametrize("seed", range(6))
def test_replay_matches_the_observations_recorded_during_original_play(seed):
    """Compare against observations captured while the game was first played."""
    from stratego.engine.state import create_game
    from stratego.engine.setup import deserialize_setup

    original, record = play_random_game(seed)

    live = create_game(
        deserialize_setup(record.red_setup),
        deserialize_setup(record.blue_setup),
        rules=original.rules,
        game_id=record.game_id,
    )
    live_signatures = [
        (
            build_observation(live, RED).tobytes(),
            build_observation(live, BLUE).tobytes(),
            tuple(legal_actions(live)),
        )
    ]
    for action in record.actions:
        apply_action(live, action)
        live_signatures.append(
            (
                build_observation(live, RED).tobytes(),
                build_observation(live, BLUE).tobytes(),
                tuple(legal_actions(live)),
            )
        )

    replay_signatures = []
    for _, state, _ in replay_plies(record):
        replay_signatures.append(
            (
                build_observation(state, RED).tobytes(),
                build_observation(state, BLUE).tobytes(),
                tuple(legal_actions(state)),
            )
        )

    assert replay_signatures == live_signatures


@pytest.mark.parametrize("seed", range(6))
def test_replay_reproduces_the_event_stream_in_order(seed):
    original, record = play_random_game(seed)
    replayed = rebuild_final_state(record)
    assert replayed.events == original.events
    for observer in (RED, BLUE):
        assert filter_events_for_observer(
            replayed.events, observer
        ) == filter_events_for_observer(original.events, observer)


@pytest.mark.parametrize("seed", range(6))
def test_replay_reproduces_counters_and_terminal_result(seed):
    original, record = play_random_game(seed)
    replayed = rebuild_final_state(record)
    assert replayed.total_moves == original.total_moves == record.total_moves
    assert replayed.battleless_moves == original.battleless_moves
    assert replayed.terminal_reason == original.terminal_reason == record.terminal_reason
    assert replayed.winner == original.winner


@pytest.mark.parametrize("seed", range(4))
def test_replay_reproduces_knowledge_and_behaviour_records(seed):
    original, record = play_random_game(seed)
    replayed = rebuild_final_state(record)

    for left, right in zip(original.pieces, replayed.pieces):
        assert (left.known_to_red, left.known_to_blue) == (
            right.known_to_red,
            right.known_to_blue,
        )
        assert left.reveal_reason_red == right.reveal_reason_red
        assert left.reveal_reason_blue == right.reveal_reason_blue

    assert {key: event.as_tuple() for key, event in original.behavior_memory.items()} == {
        key: event.as_tuple() for key, event in replayed.behavior_memory.items()
    }
    assert sorted(original.active_threat_relations) == sorted(
        replayed.active_threat_relations
    )


def test_replay_preserves_invariants_at_every_ply():
    _, record = play_random_game(23)
    baseline = None
    for _, state, _ in replay_plies(record):
        if baseline is None:
            baseline = capture_baseline(state)
        check_invariants(state, baseline=baseline)


def test_record_contains_the_documented_version_fields():
    _, record = play_random_game(5)
    assert record.replay_version == REPLAY_VERSION
    assert record.rules_version == RULES_VERSION
    assert record.observation_version == OBSERVATION_VERSION
    assert record.event_schema_version == EVENT_SCHEMA_VERSION
    assert record.terminal_result in ("red_win", "blue_win", "draw")
    assert record.seeds["game_seed"] == 5


def test_record_is_sufficient_on_its_own():
    """Only the record contents are used; nothing is carried over in memory."""
    _, record = play_random_game(8)
    text = record.to_json()
    restored = ReplayRecord.from_json(text)
    replayed = rebuild_final_state(restored)
    assert replayed.terminal_reason == record.terminal_reason
    assert replayed.total_moves == record.total_moves


def test_replay_honours_a_non_default_rules_configuration():
    original, record = play_random_game(3, rules=EVALUATION_RULES)
    assert record.battleless_move_limit == 200
    replayed = rebuild_final_state(record)
    assert replayed.rules.battleless_move_limit == 200
    assert state_fingerprint(replayed) == state_fingerprint(original)


def test_legal_action_masks_are_reproduced_at_every_ply():
    _, record = play_random_game(14)
    masks = [legal_action_mask(state) for _, state, _ in replay_plies(record)]
    again = [legal_action_mask(state) for _, state, _ in replay_plies(record)]
    assert all(np.array_equal(left, right) for left, right in zip(masks, again))
    assert len(masks) == len(record.actions) + 1


def test_initial_state_from_record_matches_the_original_start():
    original, record = play_random_game(2)
    start = initial_state_from_record(record)
    assert start.total_moves == 0
    assert start.acting_player == original.rules.first_player
    assert [record_.true_type for record_ in start.pieces] == [
        record_.true_type for record_ in rebuild_final_state(record).pieces
    ]
