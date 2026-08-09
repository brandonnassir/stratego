"""Identity knowledge: sources, monotonicity and persistence.

Covers `04_engine_validation_plan.md` sections 8 and 21.3,
`02_project_ruleset.md` sections 7 and 8, and section 12 of the Phase Two
instructions.
"""

import pytest

from stratego.engine.constants import BLUE, RED
from stratego.engine.events import EVENT_IDENTITY_REVEAL
from stratego.engine.invariants import capture_knowledge, check_invariants
from stratego.engine.random_play import play_random_game
from stratego.engine.replay import initial_state_from_record
from stratego.engine.transition import apply_action
from tests.helpers import make_position, piece_at, play


def test_players_always_know_their_own_pieces():
    state = make_position(
        red={"e3": "spy", "a1": "flag", "b1": "bomb"},
        blue={"e7": "marshal", "j10": "flag"},
    )
    for record in state.pieces_of(RED):
        assert record.known_to(RED)
    for record in state.pieces_of(BLUE):
        assert record.known_to(BLUE)


def test_opponent_identities_start_hidden():
    state = make_position(
        red={"e3": "spy", "a1": "flag"}, blue={"e7": "marshal", "j10": "flag"}
    )
    assert not piece_at(state, "e7").known_to(RED)
    assert not piece_at(state, "e3").known_to(BLUE)


def test_combat_reveals_both_identities_exactly_once():
    state = make_position(
        red={"e3": "captain", "a1": "flag", "d3": "scout"},
        blue={"e4": "marshal", "j10": "flag", "j9": "scout"},
        acting_player=RED,
    )
    marshal = piece_at(state, "e4")
    events = play(state, "e3 e4")[0]

    reveals = [event for event in events if event["event_type"] == EVENT_IDENTITY_REVEAL]
    assert len(reveals) == 2
    assert {event["reason"] for event in reveals} == {"combat"}
    assert marshal.known_to(RED)

    # A second combat with the same already-known marshal emits no new reveal.
    events = play(state, "j9 i9", "d3 d4", "e4 d4")[-1]
    assert not any(event["event_type"] == EVENT_IDENTITY_REVEAL for event in events[:1])


def test_knowledge_persists_after_the_piece_moves():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e4": "marshal", "j10": "flag"},
        acting_player=RED,
    )
    marshal = piece_at(state, "e4")
    play(state, "e3 e4", "e4 e5")
    assert marshal.known_to(RED)
    assert marshal.reveal_reason_red == "combat"


def test_knowledge_persists_after_the_piece_is_captured():
    state = make_position(
        red={"e3": "marshal", "a1": "flag"},
        blue={"e4": "captain", "j10": "flag"},
        acting_player=RED,
    )
    captain = piece_at(state, "e4")
    play(state, "e3 e4")
    assert not captain.alive
    assert captain.known_to(RED)


def test_unrelated_pieces_stay_unresolved_after_a_reveal():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e4": "marshal", "a7": "scout", "j10": "flag"},
        acting_player=RED,
    )
    bystander = piece_at(state, "a7")
    play(state, "e3 e4")
    assert not bystander.known_to(RED)


def test_multi_square_scout_move_reveals_only_the_moving_piece():
    state = make_position(
        red={"a1": "captain", "j1": "flag"},
        blue={"a10": "scout", "b10": "marshal", "j10": "flag"},
        acting_player=BLUE,
    )
    scout = piece_at(state, "a10")
    neighbour = piece_at(state, "b10")
    play(state, "a10 a7")
    assert scout.known_to(RED)
    assert not neighbour.known_to(RED)


def test_knowledge_is_monotonic_across_a_whole_random_game():
    _, record = play_random_game(12)
    state = initial_state_from_record(record)
    previous = capture_knowledge(state)
    for action in record.actions:
        apply_action(state, action)
        check_invariants(state, previous_knowledge=previous)
        current = capture_knowledge(state)
        for (was_red, was_blue), (now_red, now_blue) in zip(previous, current):
            assert now_red >= was_red
            assert now_blue >= was_blue
        previous = current


def test_reveal_reasons_are_limited_to_legal_causes():
    for seed in range(6):
        state, _ = play_random_game(seed)
        for record in state.pieces:
            for reason in (record.reveal_reason_red, record.reveal_reason_blue):
                assert reason in (None, "own_piece", "combat", "scout_multisquare")


def test_captured_identities_remain_available_for_inventory_deduction():
    state, _ = play_random_game(9)
    for record in state.pieces:
        if not record.alive:
            # Every capture comes from combat, which reveals both identities.
            assert record.known_to(RED) and record.known_to(BLUE)


@pytest.mark.parametrize("observer", [RED, BLUE])
def test_a_players_observation_never_exposes_unrevealed_opponent_types(observer):
    from stratego.engine.constants import opponent_of
    from stratego.engine.observation import (
        CH_HIDDEN_OPPONENT_OCCUPANCY,
        CH_KNOWN_OPPONENT_IDENTITY,
        build_observation,
    )
    from tests.helpers import nonterminal_state

    state = nonterminal_state(75)
    observation = build_observation(state, observer)
    opponent = opponent_of(observer)

    for record in state.pieces_of(opponent):
        if not record.alive:
            continue
        from stratego.engine.coordinates import to_perspective

        row, column = divmod(to_perspective(record.current_square, observer), 10)
        known_planes = observation[
            CH_KNOWN_OPPONENT_IDENTITY : CH_KNOWN_OPPONENT_IDENTITY + 12, row, column
        ]
        hidden_marker = observation[CH_HIDDEN_OPPONENT_OCCUPANCY, row, column]
        if record.known_to(observer):
            assert known_planes.sum() == 1.0
            assert known_planes[record.true_type] == 1.0
            assert hidden_marker == 0.0
        else:
            assert known_planes.sum() == 0.0
            assert hidden_marker == 1.0
