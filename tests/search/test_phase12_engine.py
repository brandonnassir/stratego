"""Phase 12 search engine: decisions, boundaries, terminal override, determinism."""

import numpy as np
import pytest
import torch

from stratego.engine.constants import RulesConfig
from stratego.engine.legal_moves import generate_actions_for_player, legal_actions
from stratego.engine.observation import build_observation
from stratego.engine.transition import apply_action
from stratego.model.contract import expected_value
from stratego.model.policy_adapter import greedy_action
from stratego.model.tokenization import tokenize_numpy_observation
from stratego.search.phase12.contract import (
    Phase12SearchConfig,
    Phase12SearchError,
    SEARCH_PRESETS,
    search_preset,
)
from stratego.search.phase12.engine import (
    Phase12SearchEngine,
    _greedy_model_action,
    apply_assignment_in_place,
    materialize_world,
)
from stratego.search.phase12.providers import (
    OracleBeliefProvider,
    RemainingCountBeliefProvider,
)

from tests.helpers import full_inventory_setup
from tests.search.conftest import public_state_for

TEST_CONFIG = Phase12SearchConfig(
    "unit_test", worlds=3, rollout_depth=1, max_root_candidates=4
)


def _engine(model, provider=None, config=TEST_CONFIG, **kwargs):
    provider = RemainingCountBeliefProvider() if provider is None else provider
    return Phase12SearchEngine(model, provider, config, device="cpu", **kwargs)


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------


def test_materialized_world_preserves_the_public_surface(midgame_state, midgame_public):
    provider = RemainingCountBeliefProvider()
    assignment = provider.sample_assignments(midgame_public, 1, seed=5)[0]
    observer = midgame_state.acting_player
    world = materialize_world(midgame_state, observer, assignment)

    assert np.array_equal(
        build_observation(world, observer), build_observation(midgame_state, observer)
    )
    assert generate_actions_for_player(world, observer) == generate_actions_for_player(
        midgame_state, observer
    )
    for original, cloned in zip(midgame_state.pieces, world.pieces):
        if original.owner == observer or original.known_to(observer):
            assert cloned.true_type == original.true_type
        assert cloned.alive == original.alive
        assert cloned.current_square == original.current_square
        assert cloned.has_moved == original.has_moved
    # The original state is untouched.
    assert midgame_state.pieces is not world.pieces


def test_apply_assignment_rejects_bad_worlds(midgame_state, midgame_public):
    provider = RemainingCountBeliefProvider()
    assignment = provider.sample_assignments(midgame_public, 1, seed=5)[0]
    observer = midgame_state.acting_player

    missing = dict(assignment)
    missing.pop(next(iter(missing)))
    with pytest.raises(Phase12SearchError):
        materialize_world(midgame_state, observer, missing)

    moved_slots = [
        piece["piece_slot"]
        for piece in midgame_public.public_state_document["pieces"]
        if piece["owner_color"] != midgame_public.public_state_document["observer_color"]
        and piece["alive"]
        and not piece["known_to_observer"]
        and piece["has_moved"]
    ]
    if moved_slots:
        immovable = dict(assignment)
        immovable[moved_slots[0]] = 11  # bomb on a piece that publicly moved
        with pytest.raises(Phase12SearchError):
            materialize_world(midgame_state, observer, immovable)


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------


def test_engine_refuses_oracle_under_a_production_config(random_c1):
    oracle = OracleBeliefProvider(offline_diagnostic=True)
    production = Phase12SearchConfig("unit_test", worlds=2, rollout_depth=1)
    assert production.production is True
    with pytest.raises(Phase12SearchError):
        Phase12SearchEngine(random_c1, oracle, production, device="cpu")


def test_presets_are_the_instructed_budgets():
    assert SEARCH_PRESETS["TINY"].worlds == 8
    assert SEARCH_PRESETS["TINY"].rollout_depth == 4
    assert SEARCH_PRESETS["SMALL"].worlds == 16
    assert SEARCH_PRESETS["SMALL"].rollout_depth == 6
    assert SEARCH_PRESETS["MEDIUM"].worlds == 32
    assert SEARCH_PRESETS["MEDIUM"].rollout_depth == 8
    for config in SEARCH_PRESETS.values():
        assert config.max_root_candidates == 8
        assert config.production is True
    assert search_preset("TINY", production=False).production is False
    assert search_preset("TINY", worlds=2).preset_id == "TINY_modified"
    with pytest.raises(Phase12SearchError):
        search_preset("HUGE")


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def test_decision_contract_and_seed_determinism(random_c1, midgame_state):
    engine = _engine(random_c1)
    decision = engine.choose_action(midgame_state, seed=17)

    legal = set(legal_actions(midgame_state))
    assert decision.selected_action_id in legal
    assert decision.direct_action_id in legal
    assert decision.candidates[0].is_direct
    assert decision.candidates[0].absolute_action_id == decision.direct_action_id
    assert len(decision.candidates) == min(4, len(legal))
    assert decision.worlds_requested == 3
    assert sum(decision.world_weights) == 3
    assert decision.move_changed == (
        decision.selected_action_id != decision.direct_action_id
    )
    for candidate in decision.candidates:
        assert len(candidate.world_values) == decision.unique_worlds
        assert all(-1.0 <= value <= 1.0 for value in candidate.world_values)
        expected_q = float(
            np.dot(candidate.world_values, decision.world_weights) / 3.0
        )
        assert candidate.q_value == pytest.approx(expected_q, abs=1e-12)
        assert candidate.score == pytest.approx(
            candidate.q_value + candidate.log_prior_term, abs=1e-12
        )
    # The chosen action maximises the score (ties to the lowest model id).
    best = max(candidate.score for candidate in decision.candidates)
    winners = [
        candidate
        for candidate in decision.candidates
        if candidate.score == best
    ]
    expected_choice = min(winners, key=lambda item: item.model_action_id)
    assert decision.selected_action_id == expected_choice.absolute_action_id

    assert decision.c1_forwards == sum(decision.forward_batch_sizes)
    assert decision.forward_batch_sizes[0] == 1
    assert (
        decision.terminal_leaves + decision.value_leaves
        == len(decision.candidates) * decision.unique_worlds
    )

    again = engine.choose_action(midgame_state, seed=17)
    assert again.selected_action_id == decision.selected_action_id
    assert [c.world_values for c in again.candidates] == [
        c.world_values for c in decision.candidates
    ]
    # A fresh engine instance reproduces the decision too: no hidden state.
    fresh = _engine(random_c1).choose_action(midgame_state, seed=17)
    assert fresh.selected_action_id == decision.selected_action_id


def test_search_with_the_oracle_provider_offline(random_c1, midgame_state):
    config = Phase12SearchConfig(
        "unit_test_oracle",
        worlds=3,
        rollout_depth=1,
        max_root_candidates=3,
        production=False,
    )
    engine = _engine(random_c1, provider=OracleBeliefProvider(offline_diagnostic=True), config=config)
    decision = engine.choose_action(midgame_state, seed=1)
    # One true world, deduplicated from the requested three.
    assert decision.unique_worlds == 1
    assert decision.world_weights == (3,)
    assert decision.selected_action_id in set(legal_actions(midgame_state))


def test_terminal_results_override_and_stop_the_rollout(random_c1):
    from stratego.engine.state import create_game

    # A one-move universe: any root action reaches the absolute move limit,
    # so every world of every candidate ends in an exact draw with no
    # rollout forwards at all.
    rules = RulesConfig(
        battleless_move_limit=100, absolute_move_limit=1, context="unit-test"
    )
    setup = full_inventory_setup()
    state = create_game(setup, setup, rules=rules, game_id="one-move-universe")
    assert not state.terminal

    config = Phase12SearchConfig(
        "unit_test_terminal", worlds=2, rollout_depth=4, max_root_candidates=4
    )
    engine = _engine(random_c1, config=config)
    decision = engine.choose_action(state, seed=3)

    assert decision.c1_forwards == 1  # the root forward only
    assert decision.value_leaves == 0
    assert decision.terminal_leaves == len(decision.candidates) * decision.unique_worlds
    for candidate in decision.candidates:
        assert candidate.q_value == 0.0
        assert set(candidate.world_values) == {0.0}
    # With all Q equal, the policy prior decides: the direct action wins.
    assert decision.selected_action_id == decision.direct_action_id
    assert decision.rollout_plies_total == 0


def test_depth_zero_leaf_values_flip_to_the_root_perspective(random_c1, midgame_state):
    config = Phase12SearchConfig(
        "unit_test_leaf", worlds=2, rollout_depth=0, max_root_candidates=2
    )
    engine = _engine(random_c1, config=config)
    decision = engine.choose_action(midgame_state, seed=9)

    provider = RemainingCountBeliefProvider()
    public = public_state_for(midgame_state)
    assignments = provider.sample_assignments(public, 2, seed=9)
    # Reproduce the engine's first unique world (first-occurrence order).
    root = midgame_state.acting_player
    world = materialize_world(midgame_state, root, assignments[0])
    candidate = decision.candidates[0]
    apply_action(world, candidate.absolute_action_id)
    if world.terminal:
        expected = world.result_for(root)
    else:
        tokens = tokenize_numpy_observation(
            build_observation(world, world.acting_player)
        )
        with torch.no_grad():
            outputs = random_c1(tokens)
        leaf = float(expected_value(outputs.value_logits)[0])
        expected = leaf if world.acting_player == root else -leaf
    assert candidate.world_values[0] == pytest.approx(expected, abs=1e-4)


def test_no_hidden_pieces_collapses_to_one_world(random_c1):
    from tests.helpers import make_position

    state = make_position(
        red={"e5": "marshal", "a1": "flag"},
        blue={"e7": "scout", "j10": "flag"},
        revealed={"e7", "j10"},
    )
    config = Phase12SearchConfig(
        "unit_test_no_hidden", worlds=4, rollout_depth=1, max_root_candidates=3
    )
    engine = _engine(random_c1, config=config)
    decision = engine.choose_action(state, seed=2)
    assert decision.unique_worlds == 1
    assert decision.world_weights == (4,)
    assert decision.selected_action_id in set(legal_actions(state))


def test_fast_greedy_matches_the_accepted_rule():
    rng = np.random.default_rng(0)
    for _ in range(50):
        row = np.round(
            rng.normal(size=10000).astype(np.float32), 1
        )  # rounding forces ties
        count = int(rng.integers(1, 60))
        legal = np.sort(rng.choice(10000, size=count, replace=False)).astype(np.int64)
        fast = _greedy_model_action(row, legal)
        accepted = greedy_action(torch.from_numpy(row), [int(a) for a in legal])
        assert fast == accepted
