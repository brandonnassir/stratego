"""State invariants, checked in real games and deliberately violated.

Covers `08_internal_state_spec.md` section 18, `04_engine_validation_plan.md`
sections 16 and 21.1, and section 20 of the Phase Two instructions.
"""

import pytest

from stratego.engine.constants import BLUE, PIECE_TYPE_BY_NAME, RED
from stratego.engine.invariants import (
    InvariantViolation,
    capture_baseline,
    capture_knowledge,
    check_invariants,
)
from stratego.engine.legal_moves import legal_actions
from stratego.engine.random_play import play_random_game, select_random_action
from stratego.engine.replay import initial_state_from_record
from stratego.engine.transition import apply_action
from tests.helpers import make_position, piece_at, square


@pytest.mark.parametrize("seed", range(15))
def test_invariants_hold_at_every_ply_of_a_random_game(seed):
    import random

    state, record = play_random_game(seed)
    replay = initial_state_from_record(record)
    baseline = capture_baseline(replay)
    knowledge = capture_knowledge(replay)
    check_invariants(replay, baseline=baseline)

    for action in record.actions:
        apply_action(replay, action)
        check_invariants(replay, baseline=baseline, previous_knowledge=knowledge)
        knowledge = capture_knowledge(replay)

    assert replay.terminal == state.terminal


def test_random_games_conserve_forty_pieces_per_player():
    for seed in range(10):
        state, _ = play_random_game(seed)
        for player in (RED, BLUE):
            records = state.pieces_of(player)
            assert len(records) == 40
            alive = sum(1 for record in records if record.alive)
            captured = sum(1 for record in records if not record.alive)
            assert alive + captured == 40


def test_legal_actions_never_start_from_an_opponent_piece():
    import random

    from stratego.engine.actions import decode_action

    rng = random.Random(0)
    state, record = play_random_game(2)
    replay = initial_state_from_record(record)
    for action in record.actions:
        for candidate in legal_actions(replay):
            source = decode_action(candidate)[0]
            assert replay.pieces[replay.board[source]].owner == replay.acting_player
        apply_action(replay, action)


# ---------------------------------------------------------------------------
# Negative controls: a corrupted state must be detected
# ---------------------------------------------------------------------------


def corrupt_position():
    return make_position(
        red={"e3": "captain", "a1": "flag", "b1": "bomb"},
        blue={"e7": "captain", "a10": "flag"},
    )


def test_board_and_piece_disagreement_is_detected():
    state = corrupt_position()
    state.board[square("e3")] = None
    with pytest.raises(InvariantViolation, match="board_matches_piece_records"):
        check_invariants(state, check_setup_slots=False)


def test_duplicate_occupancy_is_detected():
    state = corrupt_position()
    captain = piece_at(state, "e3")
    state.board[square("e4")] = captain.piece_id
    with pytest.raises(InvariantViolation, match="single_square_per_piece"):
        check_invariants(state, check_setup_slots=False)


def test_piece_on_a_lake_is_detected():
    state = corrupt_position()
    captain = piece_at(state, "e3")
    state.board[square("e3")] = None
    state.board[square("c5")] = captain.piece_id
    captain.current_square = square("c5")
    with pytest.raises(InvariantViolation, match="live_piece_not_on_lake"):
        check_invariants(state, check_setup_slots=False)


def test_captured_piece_on_the_board_is_detected():
    state = corrupt_position()
    captain = piece_at(state, "e3")
    captain.alive = False
    captain.capture_ply = 1
    with pytest.raises(InvariantViolation, match="captured_piece_has_no_square"):
        check_invariants(state, check_setup_slots=False)


def test_moved_flag_is_detected():
    state = corrupt_position()
    flag = piece_at(state, "a1")
    flag.has_moved = True
    with pytest.raises(InvariantViolation, match="flag_never_moves"):
        check_invariants(state, check_setup_slots=False)


def test_moved_bomb_is_detected():
    state = corrupt_position()
    bomb = piece_at(state, "b1")
    bomb.has_moved = True
    with pytest.raises(InvariantViolation, match="bomb_never_moves"):
        check_invariants(state, check_setup_slots=False)


def test_owner_losing_knowledge_of_its_own_piece_is_detected():
    state = corrupt_position()
    piece_at(state, "e3").known_to_red = False
    with pytest.raises(InvariantViolation, match="player_knows_own_identities"):
        check_invariants(state, check_setup_slots=False)


def test_knowledge_reversal_is_detected():
    state = corrupt_position()
    captain = piece_at(state, "e3")
    captain.set_known_to(BLUE, "combat")
    knowledge = capture_knowledge(state)
    captain.known_to_blue = False
    captain.reveal_reason_blue = None
    with pytest.raises(InvariantViolation, match="knowledge_is_monotonic"):
        check_invariants(state, previous_knowledge=knowledge, check_setup_slots=False)


def test_illegal_reveal_cause_is_detected():
    state = corrupt_position()
    captain = piece_at(state, "e3")
    captain.known_to_blue = True
    captain.reveal_reason_blue = "peeked_at_hidden_state"
    with pytest.raises(InvariantViolation, match="legal_reveal_cause"):
        check_invariants(state, check_setup_slots=False)


def test_changed_true_type_is_detected():
    state = corrupt_position()
    baseline = capture_baseline(state)
    piece_at(state, "e3").true_type = PIECE_TYPE_BY_NAME["marshal"]
    with pytest.raises(InvariantViolation, match="true_type_never_changes"):
        check_invariants(state, baseline=baseline, check_setup_slots=False)


def test_changed_starting_square_is_detected():
    state = corrupt_position()
    baseline = capture_baseline(state)
    piece_at(state, "e3").starting_square = square("j10")
    with pytest.raises(InvariantViolation, match="starting_square_never_changes"):
        check_invariants(state, baseline=baseline, check_setup_slots=False)


def test_wrong_setup_slot_is_detected_in_real_games():
    from tests.helpers import known_good_game

    state = known_good_game()
    check_invariants(state)
    state.pieces[0].starting_square = square("j10")
    with pytest.raises(InvariantViolation, match="starting_square_matches_slot"):
        check_invariants(state)


def test_stale_threat_relation_is_detected():
    state = corrupt_position()
    red_captain = piece_at(state, "e3")
    blue_captain = piece_at(state, "e7")
    state.active_threat_relations = [(red_captain.piece_id, blue_captain.piece_id, 999)]
    with pytest.raises(InvariantViolation, match="threat_relations_are_current"):
        check_invariants(state, check_setup_slots=False)
