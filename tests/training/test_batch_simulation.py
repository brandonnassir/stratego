"""Batch-wrapper acceptance tests (Phase 3 Agent 1).

These run at a small deterministic scale so they stay part of the ordinary
pytest run. The full-scale gates -- at least 100,000 differential comparisons,
the reset-isolation trials and the illegal-action inertness trials -- live in
`scripts/run_phase3_agent01.py`, which reuses the same comparator from
`tests/training/differential.py`.
"""

import numpy as np
import pytest

from stratego.engine.constants import (
    ACTION_SPACE_SIZE,
    BLUE,
    EVALUATION_RULES,
    LAKE_SQUARES,
    NOT_TERMINAL,
    RED,
    TERMINAL_REASONS,
    RulesConfig,
)
from stratego.engine.legal_moves import legal_action_mask, legal_actions
from stratego.engine.observation import belief_target, build_observation
from stratego.engine.random_play import make_random_setups
from stratego.engine.replay import rebuild_final_state
from stratego.engine.snapshot import restore_snapshot
from stratego.engine.state import state_fingerprint
from stratego.engine.transition import apply_action
from stratego.training.batch_simulation import (
    NO_ACTING_PLAYER,
    BatchIllegalActionError,
    BatchSimulator,
    BatchTerminalStateError,
    UnknownEnvironmentError,
    derive_slot_seed,
    slot_game_id,
)
from tests.training.differential import choose_action, reference_game, run_differential


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def advance(simulator: BatchSimulator, slot: int, plies: int) -> int:
    """Play one slot forward with the shared deterministic policy.

    Returns the number of plies actually applied, which is smaller than `plies`
    if the game finishes first.
    """
    applied = 0
    for _ in range(plies):
        if simulator.is_terminal(slot):
            break
        action = choose_action(
            simulator.root_seed,
            simulator.environment_id(slot),
            simulator.generation(slot),
            simulator.game_state(slot).total_moves,
            simulator.legal_actions(slot),
        )
        simulator.step({slot: action})
        applied += 1
    return applied


def play_to_terminal(simulator: BatchSimulator, slot: int, limit: int = 6000) -> None:
    """Drive one slot to a terminal state, stepping no other slot."""
    advance(simulator, slot, limit)
    assert simulator.is_terminal(slot), "slot did not finish within the ply limit"


# ---------------------------------------------------------------------------
# Construction, identity and determinism
# ---------------------------------------------------------------------------


def test_creates_the_requested_number_of_independent_games():
    simulator = BatchSimulator(6, root_seed=11)

    assert len(simulator) == 6
    assert simulator.environment_ids() == (0, 1, 2, 3, 4, 5)
    assert simulator.generations() == (0,) * 6
    assert simulator.active_slots() == (0, 1, 2, 3, 4, 5)
    assert simulator.finished_slots() == ()
    assert simulator.num_active == 6

    # Independent games: distinct setups and distinct state fingerprints.
    fingerprints = {simulator.slot_fingerprint(slot) for slot in range(6)}
    assert len(fingerprints) == 6
    assert len({simulator.setups(slot) for slot in range(6)}) == 6


def test_rejects_an_empty_batch():
    with pytest.raises(ValueError):
        BatchSimulator(0)


def test_slot_seeding_is_deterministic_and_slot_local():
    assert derive_slot_seed(7, 3, 0) == derive_slot_seed(7, 3, 0)
    # Neighbouring slots and consecutive generations must not collide.
    seeds = {
        derive_slot_seed(7, environment_id, generation)
        for environment_id in range(8)
        for generation in range(4)
    }
    assert len(seeds) == 32
    assert derive_slot_seed(7, 3, 0) != derive_slot_seed(8, 3, 0)


def test_a_batch_is_reproducible_from_its_root_seed():
    first = BatchSimulator(4, root_seed=99)
    second = BatchSimulator(4, root_seed=99)
    assert first.batch_fingerprint() == second.batch_fingerprint()

    for slot in range(4):
        advance(first, slot, 5)
        advance(second, slot, 5)
    assert first.batch_fingerprint() == second.batch_fingerprint()

    other = BatchSimulator(4, root_seed=100)
    assert other.batch_fingerprint() != first.batch_fingerprint()


def test_each_slot_matches_an_independently_built_reference_game():
    simulator = BatchSimulator(5, root_seed=4)
    for slot in range(5):
        reference = reference_game(4, simulator.environment_id(slot), 0)
        assert simulator.slot_fingerprint(slot)[-1] == state_fingerprint(reference)
        assert simulator.game_id(slot) == slot_game_id(4, slot, 0)


def test_a_non_default_rules_configuration_is_used_by_every_slot():
    simulator = BatchSimulator(3, root_seed=2, rules=EVALUATION_RULES)
    for slot in range(3):
        assert simulator.game_state(slot).rules is EVALUATION_RULES


# ---------------------------------------------------------------------------
# Model-facing reads
# ---------------------------------------------------------------------------


def test_acting_players_match_the_engine_and_alternate():
    simulator = BatchSimulator(3, root_seed=5)
    assert list(simulator.acting_players()) == [RED, RED, RED]

    advance(simulator, 1, 1)
    assert list(simulator.acting_players()) == [RED, BLUE, RED]
    assert simulator.acting_player(1) == BLUE


def test_observations_are_stacked_engine_observations_for_the_acting_player():
    simulator = BatchSimulator(4, root_seed=6)
    advance(simulator, 2, 3)

    active = simulator.active_slots()
    stacked = simulator.observations(active)
    assert stacked.shape == (4, 127, 10, 10)
    assert stacked.dtype == np.float32

    for position, slot in enumerate(active):
        state = simulator.game_state(slot)
        assert np.array_equal(stacked[position], build_observation(state, state.acting_player))
        for observer in (RED, BLUE):
            assert np.array_equal(
                simulator.observation(slot, observer), build_observation(state, observer)
            )


def test_legal_action_lists_and_dense_masks_match_the_engine_and_each_other():
    simulator = BatchSimulator(4, root_seed=8)
    advance(simulator, 0, 7)
    advance(simulator, 3, 2)

    masks = simulator.legal_action_masks()
    assert masks.shape == (4, ACTION_SPACE_SIZE)
    assert masks.dtype == np.uint8

    lists = simulator.legal_action_lists()
    for position, slot in enumerate(simulator.active_slots()):
        expected = legal_actions(simulator.game_state(slot))
        assert lists[position] == expected
        assert expected == sorted(expected)
        assert np.array_equal(masks[position], legal_action_mask(simulator.game_state(slot)))
        assert np.flatnonzero(masks[position]).tolist() == expected


def test_selecting_slots_explicitly_preserves_the_requested_order():
    simulator = BatchSimulator(4, root_seed=9)
    advance(simulator, 1, 1)

    players = simulator.acting_players([3, 1, 0])
    assert list(players) == [RED, BLUE, RED]
    assert simulator.observations([3, 1]).shape == (2, 127, 10, 10)
    assert simulator.legal_action_lists([2]) == [legal_actions(simulator.game_state(2))]


def test_unknown_slots_are_rejected():
    simulator = BatchSimulator(2, root_seed=3)
    for bad in (2, -1, 99):
        with pytest.raises(UnknownEnvironmentError):
            simulator.legal_actions(bad)
        with pytest.raises(UnknownEnvironmentError):
            simulator.observations([bad])


# ---------------------------------------------------------------------------
# Bulk-synchronous stepping
# ---------------------------------------------------------------------------


def test_one_batch_step_applies_exactly_one_action_per_selected_slot():
    simulator = BatchSimulator(4, root_seed=12)
    actions = {slot: simulator.legal_actions(slot)[0] for slot in (0, 2, 3)}
    before = {slot: simulator.game_state(slot).total_moves for slot in range(4)}

    result = simulator.step(actions)

    assert result.stepped == (0, 2, 3)
    assert result.actions == actions
    for slot in (0, 2, 3):
        assert simulator.game_state(slot).total_moves == before[slot] + 1
        assert simulator.game_state(slot).action_history == [actions[slot]]
        assert result.events[slot][0]["event_type"] == "move"
    # An unselected slot is untouched.
    assert simulator.game_state(1).total_moves == before[1]
    assert simulator.game_state(1).action_history == []


def test_a_dense_action_vector_skips_negative_entries():
    simulator = BatchSimulator(3, root_seed=13)
    dense = [simulator.legal_actions(0)[0], -1, simulator.legal_actions(2)[0]]

    result = simulator.step(dense)

    assert result.stepped == (0, 2)
    assert simulator.game_state(1).total_moves == 0


def test_a_dense_action_vector_must_cover_the_whole_batch():
    simulator = BatchSimulator(3, root_seed=14)
    with pytest.raises(ValueError, match="dense action vector"):
        simulator.step([simulator.legal_actions(0)[0], -1])


def test_a_batch_step_equals_stepping_the_engine_directly():
    simulator = BatchSimulator(3, root_seed=15)
    references = {slot: reference_game(15, slot, 0) for slot in range(3)}

    for _ in range(12):
        actions = {}
        for slot in simulator.active_slots():
            actions[slot] = choose_action(
                15, slot, 0, references[slot].total_moves, simulator.legal_actions(slot)
            )
        result = simulator.step(actions)
        for slot, action in actions.items():
            expected_events = apply_action(references[slot], action)
            assert list(result.events[slot]) == expected_events
            assert simulator.slot_fingerprint(slot)[-1] == state_fingerprint(references[slot])


def test_stepping_a_finished_slot_is_refused():
    simulator = BatchSimulator(2, root_seed=16)
    play_to_terminal(simulator, 0)

    assert simulator.legal_actions(0) == []
    assert not simulator.legal_action_mask(0).any()
    assert simulator.acting_player(0) == NO_ACTING_PLAYER
    assert simulator.active_slots() == (1,)
    assert simulator.finished_slots() == (0,)

    with pytest.raises(BatchTerminalStateError):
        simulator.step({0: 0})


# ---------------------------------------------------------------------------
# Terminal results
# ---------------------------------------------------------------------------


def test_terminal_result_and_reason_are_exposed():
    simulator = BatchSimulator(2, root_seed=17)
    assert simulator.outcome(0).terminal_reason == NOT_TERMINAL
    assert simulator.outcome(0).result_for_red is None

    play_to_terminal(simulator, 0)
    outcome = simulator.outcome(0)
    state = simulator.game_state(0)

    assert outcome.terminal is True
    assert outcome.terminal_reason in TERMINAL_REASONS
    assert outcome.terminal_reason != NOT_TERMINAL
    assert outcome.winner == state.winner
    assert outcome.is_draw == state.is_draw
    assert outcome.total_moves == state.total_moves
    assert outcome.result_for_red == state.result_for(RED)
    assert outcome.result_for_blue == state.result_for(BLUE)
    assert outcome.result_for_red + outcome.result_for_blue == 0.0
    assert simulator.finished_outcomes() == {0: outcome}


@pytest.mark.parametrize(
    ("rules", "expected"),
    [
        (RulesConfig(battleless_move_limit=1), "battleless_move_limit_draw"),
        (RulesConfig(absolute_move_limit=2), "absolute_move_limit_draw"),
    ],
)
def test_draw_limit_terminal_reasons_are_reported_faithfully(rules, expected):
    """Tight rule limits make the two limit-driven draws reachable from a seed.

    `opponent_no_legal_move` and `both_no_legal_move_draw` need a constructed
    position rather than a seeded setup, so they stay covered by the engine's own
    terminal-condition tests. The batch layer reads whatever reason the engine
    set, which is what this checks.
    """
    simulator = BatchSimulator(8, root_seed=35, rules=rules)
    for slot in range(8):
        play_to_terminal(simulator, slot)

    reasons = {simulator.outcome(slot).terminal_reason for slot in range(8)}
    assert expected in reasons

    for slot in range(8):
        state = simulator.game_state(slot)
        outcome = simulator.outcome(slot)
        assert outcome.terminal_reason == state.terminal_reason
        assert outcome.winner == state.winner
        assert outcome.result_for_red == state.result_for(RED)
        if outcome.terminal_reason == expected:
            assert outcome.is_draw
            assert outcome.result_for_red == 0.0
            assert outcome.result_for_blue == 0.0


def test_a_step_reports_the_games_that_finished_during_it():
    simulator = BatchSimulator(2, root_seed=18)

    # Step slot 0 until the batch step itself reports the finish.
    newly_terminal = ()
    for _ in range(6000):
        action = choose_action(
            18, 0, 0, simulator.game_state(0).total_moves, simulator.legal_actions(0)
        )
        result = simulator.step({0: action})
        if result.newly_terminal:
            newly_terminal = result.newly_terminal
            assert result.outcomes[0].terminal is True
            assert result.events[0][-1]["event_type"] == "game_end"
            break
    assert newly_terminal == (0,)


# ---------------------------------------------------------------------------
# Independent reset and generation semantics
# ---------------------------------------------------------------------------


def test_resetting_one_slot_leaves_every_other_slot_byte_identical():
    simulator = BatchSimulator(5, root_seed=19)
    # Substantially different plies across the batch.
    for slot, plies in enumerate((0, 3, 17, 60, 140)):
        advance(simulator, slot, plies)

    untouched = [0, 1, 2, 4]
    before = {slot: simulator.slot_fingerprint(slot) for slot in untouched}
    before_observations = {slot: simulator.observation(slot) for slot in untouched}

    assert simulator.reset_slots([3]) == (1,)

    for slot in untouched:
        assert simulator.slot_fingerprint(slot) == before[slot]
        assert np.array_equal(simulator.observation(slot), before_observations[slot])
    assert simulator.generations() == (0, 0, 0, 1, 0)


def test_a_finished_slot_resets_independently_of_running_slots():
    simulator = BatchSimulator(4, root_seed=20)
    for slot, plies in enumerate((5, 40, 0, 90)):
        advance(simulator, slot, plies)
    play_to_terminal(simulator, 2)

    untouched = [0, 1, 3]
    plies = {slot: simulator.game_state(slot).total_moves for slot in untouched}
    assert len(set(plies.values())) == len(untouched), "slots sit at different plies"
    before = {slot: simulator.slot_fingerprint(slot) for slot in untouched}

    assert simulator.reset_slots([2]) == (1,)

    for slot in untouched:
        assert simulator.slot_fingerprint(slot) == before[slot]
        assert simulator.game_state(slot).total_moves == plies[slot]
    assert simulator.generations() == (0, 0, 1, 0)
    assert not simulator.is_terminal(2)


def test_reset_finished_resets_exactly_the_finished_slots():
    simulator = BatchSimulator(4, root_seed=20)
    advance(simulator, 1, 12)
    play_to_terminal(simulator, 0)
    play_to_terminal(simulator, 2)

    finished = simulator.finished_slots()
    running = tuple(slot for slot in range(4) if slot not in finished)
    assert {0, 2} <= set(finished)
    assert 3 in running, "a slot that was never stepped cannot be finished"
    before = {slot: simulator.slot_fingerprint(slot) for slot in running}
    generations = simulator.generations()

    assert simulator.reset_finished() == finished

    assert simulator.finished_slots() == ()
    for slot in running:
        assert simulator.slot_fingerprint(slot) == before[slot]
    assert simulator.generations() == tuple(
        generation + (1 if slot in finished else 0)
        for slot, generation in enumerate(generations)
    )


def test_reset_increments_generation_exactly_once_and_keeps_slot_identity():
    simulator = BatchSimulator(3, root_seed=21)
    keys = []
    for expected_generation in range(4):
        assert simulator.generation(1) == expected_generation
        assert simulator.environment_id(1) == 1
        keys.append(simulator.trajectory_key(1))
        simulator.reset_slots([1])

    assert keys == [(1, 0), (1, 1), (1, 2), (1, 3)]
    assert len(set(keys)) == 4
    assert simulator.environment_ids() == (0, 1, 2)
    assert simulator.generations() == (0, 4, 0)


def test_a_reset_slot_starts_a_fresh_legal_game_with_nothing_carried_over():
    simulator = BatchSimulator(2, root_seed=22)
    advance(simulator, 0, 60)
    old_state = simulator.game_state(0)
    assert old_state.recent_moves and old_state.behavior_memory and old_state.events

    simulator.reset_slots([0])
    state = simulator.game_state(0)

    assert state is not old_state
    assert state.total_moves == 0
    assert state.battleless_moves == 0
    assert state.terminal is False
    assert state.terminal_reason == NOT_TERMINAL
    assert state.winner is None
    assert state.acting_player == simulator.rules.first_player
    # No recent moves, behavioural events, counters or knowledge survive.
    assert list(state.recent_moves) == []
    assert state.behavior_memory == {}
    assert state.active_threat_relations == []
    assert state.events == []
    assert state.action_history == []
    for record in state.pieces:
        assert record.alive
        assert not record.has_moved
        assert record.current_square == record.starting_square
        assert record.capture_ply is None
        assert record.known_to(record.owner)
        assert not record.known_to(1 - record.owner)

    # The new game is a legal game, and exactly the one the new generation seeds.
    assert len(simulator.legal_actions(0)) > 0
    expected = reference_game(22, 0, 1)
    assert simulator.slot_fingerprint(0)[-1] == state_fingerprint(expected)
    assert simulator.setups(0) == make_random_setups(derive_slot_seed(22, 0, 1))


def test_resetting_an_unknown_slot_resets_nothing():
    simulator = BatchSimulator(3, root_seed=23)
    before = simulator.batch_fingerprint()

    with pytest.raises(UnknownEnvironmentError):
        simulator.reset_slots([1, 7])

    assert simulator.batch_fingerprint() == before
    assert simulator.generations() == (0, 0, 0)


# ---------------------------------------------------------------------------
# Illegal-action inertness
# ---------------------------------------------------------------------------


def illegal_action_candidates(simulator: BatchSimulator, slot: int) -> list[int]:
    """Assorted illegal action identifiers for one slot's current position."""
    state = simulator.game_state(slot)
    legal = set(simulator.legal_actions(slot))
    candidates: list[int] = []

    own_square = next(
        record.current_square
        for record in state.pieces_of(state.acting_player)
        if record.alive
    )
    opponent_square = next(
        record.current_square
        for record in state.pieces_of(1 - state.acting_player)
        if record.alive
    )
    lake = LAKE_SQUARES[0]

    candidates.append(100 * own_square + lake)  # into a lake
    candidates.append(100 * opponent_square + lake)  # opponent's piece, into a lake
    candidates.append(100 * own_square + own_square)  # stand still
    candidates.append(ACTION_SPACE_SIZE + 5)  # outside the action space
    candidates.append(100 * opponent_square + opponent_square)  # wrong player
    for candidate in range(0, ACTION_SPACE_SIZE, 997):  # a scan of unrelated ids
        if candidate not in legal:
            candidates.append(candidate)
            break
    return [candidate for candidate in candidates if candidate not in legal]


def test_a_rejected_illegal_action_cannot_mutate_any_slot():
    simulator = BatchSimulator(4, root_seed=24)
    for slot, plies in enumerate((0, 4, 25, 80)):
        advance(simulator, slot, plies)

    before = simulator.batch_fingerprint()
    tried = 0
    for slot in range(4):
        for illegal in illegal_action_candidates(simulator, slot):
            # Submit the illegal action alongside legal actions for the other
            # slots: the whole batch step must be rejected.
            actions = {
                other: simulator.legal_actions(other)[0]
                for other in range(4)
                if other != slot
            }
            actions[slot] = illegal
            with pytest.raises(BatchIllegalActionError):
                simulator.step(actions)
            assert simulator.batch_fingerprint() == before
            tried += 1

    assert tried >= 12
    # The batch is still fully usable afterwards.
    simulator.step({slot: simulator.legal_actions(slot)[0] for slot in range(4)})
    assert simulator.batch_fingerprint() != before


def test_a_rejected_step_reports_the_offending_slot():
    simulator = BatchSimulator(3, root_seed=25)
    with pytest.raises(BatchIllegalActionError, match="slot 2"):
        simulator.step({0: simulator.legal_actions(0)[0], 2: ACTION_SPACE_SIZE + 1})


def test_an_unknown_slot_in_a_step_mutates_nothing():
    simulator = BatchSimulator(2, root_seed=26)
    before = simulator.batch_fingerprint()
    with pytest.raises(UnknownEnvironmentError):
        simulator.step({0: simulator.legal_actions(0)[0], 5: 0})
    assert simulator.batch_fingerprint() == before


# ---------------------------------------------------------------------------
# Privileged and serialisable extras (used by Agent 3)
# ---------------------------------------------------------------------------


def test_belief_targets_match_the_engine_and_stay_out_of_the_observation():
    simulator = BatchSimulator(2, root_seed=27)
    advance(simulator, 0, 30)
    state = simulator.game_state(0)

    for observer in (RED, BLUE):
        targets = simulator.belief_targets(0, observer)
        assert targets == belief_target(state, observer)
        assert targets, "a mid-game position still hides opponent pieces"
    # The stacked observation is built for the acting player only and cannot
    # depend on the privileged labels.
    assert np.array_equal(
        simulator.observations([0])[0], build_observation(state, state.acting_player)
    )


def test_snapshot_restores_an_equivalent_slot_state():
    simulator = BatchSimulator(2, root_seed=28)
    advance(simulator, 1, 24)

    restored = restore_snapshot(simulator.snapshot(1, include_history=True))
    assert state_fingerprint(restored) == state_fingerprint(simulator.game_state(1))
    assert legal_actions(restored) == simulator.legal_actions(1)


def test_replay_record_reproduces_the_slot_and_carries_its_identity():
    simulator = BatchSimulator(2, root_seed=29)
    play_to_terminal(simulator, 0)

    record = simulator.replay_record(0)
    assert record.seeds["environment_id"] == 0
    assert record.seeds["generation"] == 0
    assert record.seeds["slot_seed"] == derive_slot_seed(29, 0, 0)
    assert record.seeds["root_seed"] == 29

    rebuilt = rebuild_final_state(record)
    assert state_fingerprint(rebuilt) == state_fingerprint(simulator.game_state(0))


def test_public_views_are_engine_views_and_hide_unresolved_identities():
    simulator = BatchSimulator(2, root_seed=30)
    advance(simulator, 0, 20)
    state = simulator.game_state(0)

    for observer in (RED, BLUE):
        view = simulator.public_board(0, observer)
        assert view["observer"] in ("red", "blue")
        hidden = [
            square["piece"]
            for square in view["squares"]
            if square.get("piece") and square["piece"]["hidden"]
        ]
        assert hidden, "the opponent still has unresolved pieces"
        assert all(entry["piece_type"] is None for entry in hidden)
        assert simulator.public_events(0, observer)
        assert simulator.public_setup(0, observer)["own_setup"]
    # The observer-limited rendering conceals unresolved opponent pieces.
    assert simulator.render(0, RED).count("?") > 0
    assert "?" not in simulator.render(0)


# ---------------------------------------------------------------------------
# Scaled-down differential equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("num_environments", "root_seed", "comparisons"),
    [(1, 31, 200), (8, 32, 700), (32, 33, 700)],
)
def test_batch_matches_independently_stepped_reference_games(
    num_environments, root_seed, comparisons
):
    report = run_differential(
        num_environments=num_environments,
        root_seed=root_seed,
        target_comparisons=comparisons,
    )

    assert report.mismatch_details == []
    assert report.mismatches == 0
    assert report.generation_errors == 0
    assert report.comparisons >= comparisons


def test_the_differential_run_exercises_the_required_situations():
    report = run_differential(
        num_environments=16, root_seed=34, target_comparisons=2000
    )

    assert report.mismatches == 0
    assert report.ordinary_moves > 0
    assert report.scout_multisquare_moves > 0
    assert report.combats > 0
    assert report.reveals > 0
    assert set(report.behavior_counts) == {
        "threat",
        "evade",
        "declined_attack",
        "protect",
        "was_protected",
    }
    assert report.games_completed >= 1
    assert report.resets == report.games_completed
    assert report.terminal_reason_counts
