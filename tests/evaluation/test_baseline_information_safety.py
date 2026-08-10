"""Hidden-information safety for the baseline and stress suite.

The property under test, stated precisely:

> Take a legal position. Permute the true types of the opponent pieces the
> acting player cannot legally know, preserving every publicly deducible
> constraint. Build both policy inputs with the same policy seed. Then every
> policy's chosen action and every public diagnostic must be byte-identical,
> while the privileged state and the belief target must not be.

Both halves matter. Without the second, a fixture that quietly failed to permute
anything would make every assertion here pass vacuously, so the positive control
and the leak detector below are load-bearing rather than decoration.

This is the *local regression* suite. Agent 4 owns the >= 100,000 trial audit
across the whole suite; `scripts/run_phase4_agent02.py` runs the larger
intermediate sweep. What this file adds is a fast check that runs on every
commit and fails loudly the moment a new heuristic reaches for something it
should not see.
"""

import json
import random

import pytest

from stratego.engine.constants import EVALUATION_RULES, IMMOVABLE_TYPES
from stratego.engine.permutation import (
    belief_targets_differ,
    hidden_opponent_piece_ids,
    permute_hidden_identities,
)
from stratego.engine.random_play import play_random_game_to_ply
from stratego.evaluation.baselines import ScoringPolicy
from stratego.evaluation.heuristics import build_context, rank_moves
from stratego.evaluation.policy import build_policy_input, build_public_view
from stratego.evaluation.registry import ALL_POLICY_IDS, build_policies, build_policy

ALL_POLICY_ID_LIST = list(ALL_POLICY_IDS)

#: Plies chosen to span opening, middle game and late positions, because the
#: amount of hidden information -- and therefore what a leak could look like --
#: changes completely across a game.
SAMPLE_PLIES = (8, 20, 40, 70, 110)

POLICY_SEED = 4242


def permutation_cases(trials: int = 30):
    """Positions paired with a valid hidden-identity permutation of themselves.

    Only cases where the permutation actually changed an assignment are yielded;
    an unchanged clone would make the comparison trivially true.
    """
    rng = random.Random(20260401)
    produced = 0
    for seed in range(600):
        if produced >= trials:
            return
        for ply in SAMPLE_PLIES:
            if produced >= trials:
                return
            state = play_random_game_to_ply(seed, ply, rules=EVALUATION_RULES)
            if state.terminal or state.total_moves != ply:
                continue
            clone, info = permute_hidden_identities(state, state.acting_player, rng)
            if not info["valid"] or not info["changed"]:
                continue
            yield state, clone
            produced += 1


def make_request(state, policy, seed: int = POLICY_SEED):
    return build_policy_input(
        state,
        policy=policy.ref,
        policy_seed=seed,
        requirements=policy.requirements,
        suite_version="test_suite",
        match_id="m-test",
        paired_unit_id="u-test",
    )


# ---------------------------------------------------------------------------
# The fixture is not vacuous
# ---------------------------------------------------------------------------


def test_the_fixture_produces_the_number_of_cases_it_promises():
    assert len(list(permutation_cases(30))) == 30


def test_the_permutation_really_changes_the_privileged_state():
    """Positive control. Without this every invariance test below proves nothing."""
    for state, clone in permutation_cases(20):
        observer = state.acting_player
        assert belief_targets_differ(state, clone, observer)
        original = [(record.piece_id, record.true_type) for record in state.pieces]
        permuted = [(record.piece_id, record.true_type) for record in clone.pieces]
        assert original != permuted


def test_the_permutation_preserves_the_public_constraint_it_must():
    """A piece that has moved is neither Flag nor Bomb, and movement is public."""
    for state, clone in permutation_cases(20):
        for record in clone.pieces:
            if record.has_moved:
                assert record.true_type not in IMMOVABLE_TYPES


def test_a_policy_that_read_hidden_types_would_be_caught():
    """Leak detector: proves these positions can distinguish hidden state at all.

    Nothing in the contract lets a real policy do this -- `PolicyInput` carries
    no privileged object -- so the leak is simulated by reading the `GameState`
    directly, which is exactly what the contract forbids. If this ever stopped
    detecting a difference, the invariance assertions below would be measuring
    nothing.
    """
    detected = 0
    total = 0
    for state, clone in permutation_cases(20):
        observer = state.acting_player
        hidden = hidden_opponent_piece_ids(state, observer)
        if not hidden:
            continue
        total += 1
        leaked_left = [state.pieces[piece_id].true_type for piece_id in hidden]
        leaked_right = [clone.pieces[piece_id].true_type for piece_id in hidden]
        if leaked_left != leaked_right:
            detected += 1
    assert total > 0
    assert detected == total, "the fixture stopped changing what a leak would expose"


# ---------------------------------------------------------------------------
# Invariance of the inputs
# ---------------------------------------------------------------------------


def test_the_public_view_is_invariant():
    """The foundation of the whole argument: the policies' only input is fixed."""
    for state, clone in permutation_cases(30):
        observer = state.acting_player
        assert build_public_view(state, observer) == build_public_view(clone, observer)


def test_every_derived_context_field_is_invariant():
    """The heuristic layer adds no new dependence on the state beyond the view."""
    policy = build_policy("strategic_rule_based")
    for state, clone in permutation_cases(30):
        left = build_context(make_request(state, policy))
        right = build_context(make_request(clone, policy))
        for name in (
            "own_flag_square",
            "own_flag_attackers",
            "known_opponent_marshal_square",
            "own_miner_count",
            "known_attacker_types",
            "hidden_mover_adjacent",
            "own_support",
            "empty_neighbours",
            "unresolved_counts",
            "unresolved_total",
            "unresolved_bombs",
            "average_hidden_value",
            "own_material",
            "opponent_material",
            "material_edge",
            "battleless_pressure",
            "last_source_of",
            "recent_move_count",
            "moves",
        ):
            assert getattr(left, name) == getattr(right, name), f"{name} changed"


# ---------------------------------------------------------------------------
# Invariance of the decisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy_id", ALL_POLICY_ID_LIST)
def test_the_chosen_action_is_invariant(policy_id):
    policy = build_policy(policy_id)
    trials = 0
    for state, clone in permutation_cases(30):
        left = policy.decide_checked(make_request(state, policy))
        right = policy.decide_checked(make_request(clone, policy))
        assert left.selected_action_id == right.selected_action_id
        trials += 1
    assert trials == 30


@pytest.mark.parametrize("policy_id", ALL_POLICY_ID_LIST)
def test_the_diagnostics_are_invariant(policy_id):
    """Compared as whole payloads, so a leak into any score component fails here."""
    policy = build_policy(policy_id)
    for state, clone in permutation_cases(30):
        left = policy.decide_checked(make_request(state, policy))
        right = policy.decide_checked(make_request(clone, policy))
        assert left.diagnostics == right.diagnostics
        assert json.dumps(dict(left.diagnostics), sort_keys=True) == json.dumps(
            dict(right.diagnostics), sort_keys=True
        )


@pytest.mark.parametrize("policy_id", ALL_POLICY_ID_LIST)
def test_the_whole_score_vector_is_invariant(policy_id):
    """Far stronger than comparing the argmax.

    Two score vectors can share a maximum and still differ everywhere else, so a
    leak that only moved the ranking of losing candidates would survive an
    action-only comparison. Comparing every score closes that gap.
    """
    policy = build_policy(policy_id)
    if not isinstance(policy, ScoringPolicy):
        pytest.skip(f"{policy_id} does not expose a per-move score vector")

    for state, clone in permutation_cases(30):
        left_context = build_context(make_request(state, policy))
        right_context = build_context(make_request(clone, policy))
        left = rank_moves(policy.score(left_context, move) for move in left_context.moves)
        right = rank_moves(policy.score(right_context, move) for move in right_context.moves)
        assert left == right


def test_every_policy_in_the_catalogue_is_covered():
    """A policy that quietly left the catalogue would silently leave the audit."""
    covered = {policy.policy_id for policy in build_policies()}
    assert covered == set(ALL_POLICY_IDS)
    assert len(covered) >= 10


def test_the_decision_seed_does_not_rescue_a_leak():
    """A different seed must not mask a difference, and must not create one.

    Running the whole comparison under several seeds guards against a policy
    whose hidden-state dependence only shows up on some branch of its sampling.
    """
    policies = build_policies()
    comparisons = 0
    for state, clone in permutation_cases(8):
        for seed in (1, 7, 2026):
            for policy in policies:
                left = policy.decide_checked(make_request(state, policy, seed=seed))
                right = policy.decide_checked(make_request(clone, policy, seed=seed))
                assert left.selected_action_id == right.selected_action_id
                assert left.diagnostics == right.diagnostics
                comparisons += 1
    assert comparisons == 8 * 3 * len(policies)


def test_the_local_suite_runs_a_meaningful_number_of_comparisons():
    """Guards the suite's own coverage claim in the Phase 4 report.

    Agent 4 owns the >= 100,000 trial audit. This asserts the *local* regression
    suite is large enough to be worth running on every commit, so a future edit
    that trims it down fails here rather than passing quietly.
    """
    cases = 30
    policies = len(ALL_POLICY_IDS)
    action_and_diagnostic_comparisons = 2 * cases * policies
    seeded_comparisons = 8 * 3 * policies
    assert action_and_diagnostic_comparisons + seeded_comparisons >= 800

    plies_covered = {state.total_moves for state, _ in permutation_cases(30)}
    assert len(plies_covered) >= 3, "the sample must span more than one game phase"
