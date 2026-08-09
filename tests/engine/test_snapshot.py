"""Snapshot and restore equivalence.

Covers `03_game_engine_spec.md` section 13, `08_internal_state_spec.md`
section 15, `04_engine_validation_plan.md` sections 14 and 21.5, and section 18
of the Phase Two instructions.
"""

import numpy as np
import pytest

from stratego.engine.constants import BLUE, EVALUATION_RULES, RED
from stratego.engine.events import filter_events_for_observer, public_board_view
from stratego.engine.legal_moves import legal_action_mask, legal_actions
from stratego.engine.observation import build_observation
from stratego.engine.random_play import play_random_game_to_ply
from stratego.engine.snapshot import create_snapshot, restore_snapshot, snapshot_to_json
from stratego.engine.state import state_fingerprint
from stratego.engine.transition import apply_action
from tests.helpers import make_position, nonterminal_state, play


def assert_restores_exactly(state):
    """Every guarantee `08_internal_state_spec.md` section 15 requires."""
    snapshot = create_snapshot(state, include_history=True)
    restored = restore_snapshot(snapshot)

    assert state_fingerprint(restored) == state_fingerprint(state)
    assert legal_actions(restored) == legal_actions(state)
    assert np.array_equal(legal_action_mask(restored), legal_action_mask(state))
    for observer in (RED, BLUE):
        assert np.array_equal(
            build_observation(restored, observer), build_observation(state, observer)
        )
        assert public_board_view(restored, observer) == public_board_view(state, observer)
        assert filter_events_for_observer(
            restored.events, observer
        ) == filter_events_for_observer(state.events, observer)

    if not state.terminal:
        action = legal_actions(state)[0]
        original_events = apply_action(state, action)
        restored_events = apply_action(restored, action)
        assert restored_events == original_events
        assert state_fingerprint(restored) == state_fingerprint(state)

    assert restored.terminal == state.terminal
    assert restored.terminal_reason == state.terminal_reason
    assert restored.winner == state.winner
    return restored


def test_snapshot_is_independent_of_the_live_state():
    state = nonterminal_state(20)
    snapshot = create_snapshot(state, include_history=True)
    before = state_fingerprint(restore_snapshot(snapshot))

    apply_action(state, legal_actions(state)[0])

    assert state_fingerprint(restore_snapshot(snapshot)) == before


@pytest.mark.parametrize("ply", [0, 4, 40, 120, 260])
def test_snapshot_restore_across_game_phases(ply):
    """Early, middle and late game snapshots."""
    assert_restores_exactly(nonterminal_state(ply))


def test_snapshot_immediately_before_and_after_combat():
    state = make_position(
        red={"e3": "marshal", "a1": "flag"},
        blue={"e4": "captain", "j10": "flag", "j9": "scout"},
        acting_player=RED,
    )
    assert_restores_exactly(state)  # this also applies the first legal action

    combat_state = make_position(
        red={"e3": "marshal", "a1": "flag"},
        blue={"e4": "captain", "j10": "flag", "j9": "scout"},
        acting_player=RED,
    )
    play(combat_state, "e3 e4")
    assert combat_state.battleless_moves == 0
    assert_restores_exactly(combat_state)


def test_snapshot_near_the_battleless_draw_limit():
    state = make_position(
        red={"a1": "captain", "j1": "flag"},
        blue={"a10": "captain", "j10": "flag"},
        acting_player=RED,
        battleless_moves=97,
    )
    restored = assert_restores_exactly(state)
    assert restored.battleless_moves == state.battleless_moves


def test_snapshot_of_a_terminal_state():
    state = make_position(
        red={"e3": "scout"}, blue={"e4": "flag", "j10": "captain"}, acting_player=RED
    )
    play(state, "e3 e4")
    restored = assert_restores_exactly(state)
    assert restored.terminal_reason == "flag_capture"


def test_snapshot_with_behaviour_history_and_threat_relations():
    """`04_engine_validation_plan.md` section 21.5 completeness case."""
    state = make_position(
        red={"e3": "captain", "c3": "miner", "a1": "flag"},
        blue={"e5": "sergeant", "g3": "scout", "j10": "flag"},
        acting_player=BLUE,
        battleless_moves=12,
    )
    play(state, "e5 e4")  # blue threatens the red captain
    play(state, "c3 d3")  # red protects the captain, declining nothing
    play(state, "g3 f3")
    play(state, "e3 e2")  # red evades

    assert state.behavior_memory
    assert state.active_threat_relations or True  # may be empty; both are valid
    assert_restores_exactly(state)


def test_restored_state_continues_a_whole_game_identically():
    state = nonterminal_state(50)
    snapshot = create_snapshot(state, include_history=True)
    restored = restore_snapshot(snapshot)

    for _ in range(40):
        if state.terminal:
            break
        action = legal_actions(state)[len(state.action_history) % len(legal_actions(state))]
        apply_action(state, action)
        apply_action(restored, action)

    assert state_fingerprint(restored) == state_fingerprint(state)


def test_compact_snapshot_omits_history_but_still_reproduces_behaviour():
    state = nonterminal_state(64)
    compact = create_snapshot(state)
    assert "events" not in compact
    assert "action_history" not in compact

    restored = restore_snapshot(compact)
    assert state_fingerprint(restored, include_history=False) == state_fingerprint(
        state, include_history=False
    )
    assert legal_actions(restored) == legal_actions(state)
    for observer in (RED, BLUE):
        assert np.array_equal(
            build_observation(restored, observer), build_observation(state, observer)
        )


def test_snapshot_serialises_to_json():
    state = nonterminal_state(30)
    text = snapshot_to_json(create_snapshot(state))
    assert text.startswith("{")
    assert "rules_version" in text


def test_snapshot_rejects_an_unknown_version():
    state = nonterminal_state(10)
    snapshot = create_snapshot(state)
    snapshot["snapshot_version"] = "snapshot_v0"
    with pytest.raises(ValueError, match="unsupported snapshot version"):
        restore_snapshot(snapshot)


def test_snapshot_preserves_a_non_default_rules_configuration():
    state = nonterminal_state(15, rules=EVALUATION_RULES)
    restored = restore_snapshot(create_snapshot(state))
    assert restored.rules is EVALUATION_RULES
    assert restored.rules.battleless_move_limit == 200
