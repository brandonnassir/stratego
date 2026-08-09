"""Observer-filtered board views, setup views and public event streams.

Covers `09_public_event_and_replay_schema.md` sections 11, 12, 15 and
`04_engine_validation_plan.md` section 21.8.
"""

import json

import pytest

from stratego.engine.constants import BLUE, PIECE_TYPE_NAMES, RED
from stratego.engine.events import (
    filter_events_for_observer,
    public_board_view,
    public_setup_view,
)
from stratego.engine.pieces import piece_id_name
from stratego.engine.random_play import play_random_game
from tests.helpers import make_position, nonterminal_state, piece_at, play


@pytest.mark.parametrize("observer", [RED, BLUE])
def test_own_types_are_exact_and_opponent_types_are_hidden(observer):
    from stratego.engine.constants import opponent_of

    state = nonterminal_state(60)
    view = public_board_view(state, observer)
    opponent = opponent_of(observer)

    for entry in view["squares"]:
        if entry.get("lake") or entry.get("piece") is None:
            continue
        piece = entry["piece"]
        record = state.pieces[
            next(
                item.piece_id
                for item in state.pieces
                if piece_id_name(item.piece_id) == piece["piece_id"]
            )
        ]
        if record.owner == observer:
            assert piece["hidden"] is False
            assert piece["piece_type"] == PIECE_TYPE_NAMES[record.true_type]
        elif record.known_to(observer):
            assert piece["hidden"] is False
            assert piece["piece_type"] == PIECE_TYPE_NAMES[record.true_type]
        else:
            assert piece["hidden"] is True
            assert piece["piece_type"] is None


def test_hidden_opponent_pieces_still_expose_their_stable_identifier():
    state = make_position(
        red={"e3": "captain", "a1": "flag"}, blue={"e7": "marshal", "j10": "flag"}
    )
    view = public_board_view(state, RED)
    marshal_square = next(
        entry for entry in view["squares"] if entry["name"] == "e7"
    )
    assert marshal_square["piece"]["hidden"] is True
    assert marshal_square["piece"]["piece_type"] is None
    assert marshal_square["piece"]["piece_id"].startswith("blue:")


def test_known_opponent_identities_remain_visible_after_revelation():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e4": "marshal", "j10": "flag"},
        acting_player=RED,
    )
    play(state, "e3 e4", "e4 e5")
    entry = next(item for item in public_board_view(state, RED)["squares"] if item["name"] == "e5")
    assert entry["piece"]["piece_type"] == "marshal"
    assert entry["piece"]["hidden"] is False


def test_board_view_reports_lakes_and_counters():
    state = nonterminal_state(30)
    view = public_board_view(state, RED)
    lakes = [entry for entry in view["squares"] if entry.get("lake")]
    assert len(lakes) == 8
    assert {entry["name"] for entry in lakes} == {
        "c5", "d5", "g5", "h5", "c6", "d6", "g6", "h6"
    }
    assert view["total_moves"] == state.total_moves
    assert view["moves_since_last_combat"] == state.battleless_moves
    assert view["battleless_move_limit"] == state.rules.battleless_move_limit


def test_board_view_is_json_serialisable():
    state = nonterminal_state(45)
    text = json.dumps(public_board_view(state, RED))
    assert '"hidden": true' in text or '"hidden":true' in text


@pytest.mark.parametrize("observer", [RED, BLUE])
def test_setup_view_shows_only_the_observers_own_identities(observer):
    from stratego.engine.constants import opponent_of

    state = nonterminal_state(0)
    view = public_setup_view(state, observer)
    opponent = opponent_of(observer)

    assert len(view["own_setup"]) == 40
    assert len(view["opponent_setup_occupancy"]) == 40
    assert set(view["own_setup"].values()) <= set(PIECE_TYPE_NAMES)

    # The opponent block must contain identifiers only, never type names.
    for value in view["opponent_setup_occupancy"].values():
        assert value.startswith("blue:" if opponent == BLUE else "red:")
        assert value not in PIECE_TYPE_NAMES


def test_no_public_event_carries_an_unrevealed_type():
    state, _ = play_random_game(15)
    revealed_by_ply: dict[str, int] = {}
    for event in state.events:
        if event["event_type"] == "identity_reveal":
            revealed_by_ply[event["piece_id"]] = event["ply"]

    for observer in (RED, BLUE):
        for event in filter_events_for_observer(state.events, observer):
            if event["event_type"] == "move":
                assert "piece_type" not in event
                assert "true_type" not in event
            if event["event_type"] == "behavior":
                assert "piece_type" not in event
                assert not any(value in PIECE_TYPE_NAMES for value in event.values())


def test_combat_events_disclose_types_only_because_combat_is_public():
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e4": "marshal", "j10": "flag"},
        acting_player=RED,
    )
    events = play(state, "e3 e4")[0]
    combat = next(event for event in events if event["event_type"] == "combat")
    assert combat["attacker_type"] == "captain"
    assert combat["defender_type"] == "marshal"
    # Both players legally know both identities at this point.
    assert piece_at(state, "e4").known_to(RED)


def test_behaviour_events_never_contain_type_information():
    state, _ = play_random_game(19)
    for event in state.events:
        if event["event_type"] != "behavior":
            continue
        assert set(event) == {
            "event_type",
            "behavior_type",
            "ply",
            "actor_piece_id",
            "counterpart_piece_id",
            "actor_knew_counterpart_type",
            "context_piece_id",
        }


def test_filtered_event_stream_uses_a_field_whitelist():
    state, _ = play_random_game(21)
    for observer in (RED, BLUE):
        filtered = filter_events_for_observer(state.events, observer)
        assert len(filtered) == len(state.events)
        for original, projected in zip(state.events, filtered):
            assert projected["event_type"] == original["event_type"]
            assert set(projected) <= set(original)
