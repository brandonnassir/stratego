"""The observer-safe policy contract.

Three properties matter here:

1. a :class:`PolicyInput` carries no privileged object and no hidden identity;
2. a policy only receives what it declared, so an unused product is never built
   and therefore can never leak;
3. a decision is reproducible from `(public input, policy seed, ply)` alone.

The permutation tests at the end are the local version of the audit Agent 4 runs
at scale: permuting the true types of unresolved opponent pieces must change the
privileged state and the belief target while changing nothing a policy can see.
"""

import random

import numpy as np
import pytest

from stratego.engine.constants import (
    BLUE,
    EVALUATION_RULES,
    NUM_PIECE_TYPES,
    NUM_SQUARES,
    PIECE_COUNTS,
    RED,
)
from stratego.engine.legal_moves import legal_actions
from stratego.engine.observation import belief_target
from stratego.engine.permutation import (
    belief_targets_differ,
    hidden_opponent_piece_ids,
    permute_hidden_identities,
)
from stratego.engine.pieces import PieceRecord
from stratego.engine.random_play import play_random_game_to_ply
from stratego.engine.replay import ReplayRecord
from stratego.engine.state import BehaviorEvent, GameState, RecentMove
from stratego.evaluation.policy import (
    POLICY_INTERFACE_VERSION,
    FirstLegalActionPolicy,
    ObservationProbePolicy,
    Policy,
    PolicyContractError,
    PolicyInput,
    PolicyRef,
    PolicyRequirements,
    PolicyResult,
    PublicView,
    SeededUniformPolicy,
    build_policy_input,
    build_public_view,
    derive_decision_seed,
    validate_policy_result,
)
from tests.helpers import T, make_position, nonterminal_state, square

ALL_REQUIREMENTS = PolicyRequirements(
    observation=True,
    legal_action_mask=True,
    public_view=True,
    public_events=True,
    public_setup=True,
)

CONTRACT_POLICIES = (FirstLegalActionPolicy, SeededUniformPolicy, ObservationProbePolicy)

#: Objects that must never be reachable from a policy input.
PRIVILEGED_TYPES = (GameState, PieceRecord, BehaviorEvent, RecentMove, ReplayRecord)


def make_request(state: GameState, policy: Policy, seed: int = 4242, **overrides) -> PolicyInput:
    fields = {
        "policy": policy.ref,
        "policy_seed": seed,
        "requirements": policy.requirements,
        "suite_version": "test_suite",
        "match_id": "m-test",
        "paired_unit_id": "u-test",
    }
    fields.update(overrides)
    return build_policy_input(state, **fields)


# ---------------------------------------------------------------------------
# No privileged state reaches a policy
# ---------------------------------------------------------------------------


def _walk(value, seen: set[int], depth: int = 0):
    """Yield every object reachable from `value` through public structure."""
    if depth > 6 or id(value) in seen:
        return
    seen.add(id(value))
    yield value
    if isinstance(value, np.ndarray) or isinstance(value, (str, bytes, int, float, bool)):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(key, seen, depth + 1)
            yield from _walk(item, seen, depth + 1)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _walk(item, seen, depth + 1)
        return
    for attribute in getattr(value, "__dict__", {}).values():
        yield from _walk(attribute, seen, depth + 1)


def test_a_policy_input_reaches_no_privileged_object():
    state = nonterminal_state(30)
    request = make_request(state, FirstLegalActionPolicy(), requirements=ALL_REQUIREMENTS)
    reachable = list(_walk(request, set()))
    for item in reachable:
        assert not isinstance(item, PRIVILEGED_TYPES), (
            f"{type(item).__name__} is reachable from a policy input"
        )
    # Identity, not equality: a NumPy array makes `in` ambiguous.
    assert not any(item is state for item in reachable)


def test_a_policy_input_reaches_no_belief_target():
    state = nonterminal_state(30)
    request = make_request(state, FirstLegalActionPolicy(), requirements=ALL_REQUIREMENTS)
    target = belief_target(state, state.acting_player)
    reachable = list(_walk(request, set()))
    assert not any(item is target for item in reachable)
    assert not any(item is state.pieces for item in reachable)
    assert not any(item is state.board for item in reachable)
    assert not any(item is state.behavior_memory for item in reachable)


def test_the_public_view_masks_every_unresolved_opponent_type():
    state = nonterminal_state(40)
    observer = state.acting_player
    view = build_public_view(state, observer)
    hidden = set(hidden_opponent_piece_ids(state, observer))

    for record in state.pieces:
        entry = view.piece(record.piece_id)
        if record.piece_id in hidden:
            assert entry.piece_type is None
            assert entry.hidden
        elif record.known_to(observer):
            assert entry.piece_type == record.true_type
    assert set(view.unresolved_opponent_piece_ids) == hidden


def test_a_player_always_sees_their_own_pieces():
    state = nonterminal_state(40)
    view = build_public_view(state, state.acting_player)
    for piece_id in view.own_piece_ids:
        assert view.piece(piece_id).piece_type is not None
        assert view.piece(piece_id).known


def test_unresolved_counts_are_a_legal_public_deduction():
    state = nonterminal_state(60)
    view = build_public_view(state, state.acting_player)
    assert len(view.unresolved_opponent_counts) == NUM_PIECE_TYPES
    # Every opponent copy is either legally resolved or still unaccounted for.
    assert sum(view.unresolved_opponent_counts) == len(view.unresolved_opponent_piece_ids)
    for piece_type, count in enumerate(view.unresolved_opponent_counts):
        assert 0 <= count <= PIECE_COUNTS[piece_type]


def test_the_public_view_reports_which_own_pieces_are_exposed():
    state = make_position(
        red={"a4": "miner", "b4": "scout"},
        blue={"a5": "captain"},
        revealed={"a4", "a5"},
        acting_player=RED,
        rules=EVALUATION_RULES,
    )
    view = build_public_view(state, RED)
    exposed = {view.piece(piece_id).square for piece_id in view.own_piece_ids_known_to_opponent}
    assert square("a4") in exposed
    assert square("b4") not in exposed


def test_the_public_view_describes_the_board_and_the_clocks():
    state = nonterminal_state(30, rules=EVALUATION_RULES)
    view = build_public_view(state, state.acting_player)
    assert len(view.occupancy) == NUM_SQUARES
    assert view.ply == state.total_moves
    assert view.battleless_move_limit == EVALUATION_RULES.battleless_move_limit
    assert view.absolute_move_limit == EVALUATION_RULES.absolute_move_limit
    assert view.moves_until_battleless_draw == (
        EVALUATION_RULES.battleless_move_limit - state.battleless_moves
    )
    assert view.opponent != view.observer
    for square_index, piece_id in enumerate(view.occupancy):
        if piece_id is not None:
            assert view.piece_at(square_index).piece_id == piece_id


def test_build_public_view_rejects_an_unknown_observer():
    with pytest.raises(PolicyContractError):
        build_public_view(nonterminal_state(10), 7)


# ---------------------------------------------------------------------------
# Requirement-declared materialisation
# ---------------------------------------------------------------------------


def test_undeclared_products_are_absent():
    state = nonterminal_state(20)
    request = make_request(
        state, FirstLegalActionPolicy(), requirements=PolicyRequirements(public_view=True)
    )
    assert request.public_view is not None
    assert request.observation is None
    assert request.legal_action_mask is None
    assert request.public_events is None
    assert request.public_setup is None


def test_reading_an_undeclared_product_fails_loudly():
    state = nonterminal_state(20)
    request = make_request(
        state, FirstLegalActionPolicy(), requirements=PolicyRequirements(public_view=False)
    )
    for reader in ("require_observation", "require_legal_action_mask", "require_public_view"):
        with pytest.raises(PolicyContractError):
            getattr(request, reader)()


def test_declared_products_are_materialised():
    state = nonterminal_state(20)
    request = make_request(state, FirstLegalActionPolicy(), requirements=ALL_REQUIREMENTS)
    assert request.require_observation().shape == (127, 10, 10)
    assert request.require_legal_action_mask().shape == (10_000,)
    assert isinstance(request.require_public_view(), PublicView)
    assert request.public_events is not None
    assert request.public_setup is not None
    assert set(request.public_setup) == {"observer", "own_setup", "opponent_setup_occupancy"}


def test_materialised_arrays_are_read_only():
    """A policy must not be able to scribble on engine-derived memory."""
    request = make_request(nonterminal_state(20), ObservationProbePolicy())
    with pytest.raises(ValueError):
        request.require_observation()[0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        request.require_legal_action_mask()[0] = 1


def test_the_legality_mask_agrees_with_the_legal_action_list():
    state = nonterminal_state(25)
    request = make_request(state, ObservationProbePolicy())
    mask = request.require_legal_action_mask()
    assert set(np.flatnonzero(mask).tolist()) == set(request.legal_actions)
    assert request.legal_actions == tuple(legal_actions(state))


def test_a_terminal_state_never_produces_a_decision_request():
    state = play_random_game_to_ply(3, 10_000)
    assert state.terminal
    with pytest.raises(PolicyContractError):
        make_request(state, FirstLegalActionPolicy())


def test_the_request_carries_its_match_identity():
    state = nonterminal_state(20)
    request = make_request(state, FirstLegalActionPolicy(), seed=99, game_id="m-xyz")
    identity = request.identity()
    assert identity["match_id"] == "m-test"
    assert identity["paired_unit_id"] == "u-test"
    assert identity["game_id"] == "m-xyz"
    assert identity["ply"] == state.total_moves
    assert identity["acting_player"] == state.acting_player
    assert identity["policy_seed"] == 99
    assert identity["decision_seed"] == derive_decision_seed(99, state.total_moves)
    assert request.rules is state.rules


# ---------------------------------------------------------------------------
# Seeds and reproducibility
# ---------------------------------------------------------------------------


def test_decision_seeds_are_a_pure_function_of_seed_and_ply():
    assert derive_decision_seed(7, 3) == derive_decision_seed(7, 3)
    assert derive_decision_seed(7, 3) != derive_decision_seed(7, 4)
    assert derive_decision_seed(7, 3) != derive_decision_seed(8, 3)
    assert derive_decision_seed(7, 3) >= 0


def test_the_random_stream_is_fresh_on_every_call():
    request = make_request(nonterminal_state(20), SeededUniformPolicy())
    assert [request.random_stream().random() for _ in range(3)].count(
        request.random_stream().random()
    ) == 3


@pytest.mark.parametrize("policy_class", CONTRACT_POLICIES)
def test_the_same_public_input_and_seed_give_the_same_action(policy_class):
    policy = policy_class()
    state = nonterminal_state(30)
    first = policy.decide_checked(make_request(state, policy))
    second = policy.decide_checked(make_request(state, policy))
    assert first == second
    assert first.selected_action_id == second.selected_action_id


def test_a_stochastic_policy_follows_its_seed():
    policy = SeededUniformPolicy()
    state = nonterminal_state(30)
    choices = {
        policy.decide(make_request(state, policy, seed=seed)).selected_action_id
        for seed in range(40)
    }
    assert len(choices) > 1, "a seeded uniform policy must depend on its seed"


def test_a_deterministic_policy_ignores_the_seed():
    policy = FirstLegalActionPolicy()
    state = nonterminal_state(30)
    choices = {
        policy.decide(make_request(state, policy, seed=seed)).selected_action_id
        for seed in range(20)
    }
    assert len(choices) == 1


# ---------------------------------------------------------------------------
# Legality and the result contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy_class", CONTRACT_POLICIES)
def test_contract_policies_always_return_a_legal_action(policy_class):
    policy = policy_class()
    decisions = 0
    for seed in range(12):
        for ply in (0, 8, 24, 60):
            state = play_random_game_to_ply(seed, ply)
            if state.terminal:
                continue
            request = make_request(state, policy, seed=1000 + seed)
            result = policy.decide_checked(request)
            assert result.selected_action_id in request.legal_actions
            decisions += 1
    assert decisions >= 30


def test_an_illegal_selection_is_rejected():
    policy = FirstLegalActionPolicy()
    request = make_request(nonterminal_state(20), policy)
    illegal = PolicyResult(9_999, request.policy, request.decision_seed)
    assert illegal.selected_action_id not in request.legal_actions
    with pytest.raises(PolicyContractError):
        validate_policy_result(illegal, request)


def test_a_result_claiming_the_wrong_identity_is_rejected():
    request = make_request(nonterminal_state(20), FirstLegalActionPolicy())
    impostor = PolicyResult(
        request.legal_actions[0], PolicyRef("other", "1.0.0"), request.decision_seed
    )
    with pytest.raises(PolicyContractError):
        validate_policy_result(impostor, request)


def test_a_result_with_the_wrong_decision_seed_is_rejected():
    request = make_request(nonterminal_state(20), FirstLegalActionPolicy())
    stale = PolicyResult(request.legal_actions[0], request.policy, request.decision_seed + 1)
    with pytest.raises(PolicyContractError):
        validate_policy_result(stale, request)


def test_a_non_result_return_value_is_rejected():
    request = make_request(nonterminal_state(20), FirstLegalActionPolicy())
    with pytest.raises(PolicyContractError):
        validate_policy_result(request.legal_actions[0], request)


def test_a_request_addressed_to_another_policy_is_rejected():
    state = nonterminal_state(20)
    request = make_request(state, FirstLegalActionPolicy())
    with pytest.raises(PolicyContractError):
        SeededUniformPolicy().decide_checked(request)


def test_the_result_exposes_serialisable_metadata():
    policy = SeededUniformPolicy()
    request = make_request(nonterminal_state(20), policy)
    result = policy.decide_checked(request)
    payload = result.to_dict()
    assert payload["policy"] == {
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
    }
    assert payload["decision_seed"] == request.decision_seed
    assert result.policy_id == policy.policy_id
    assert result.policy_version == policy.policy_version
    assert payload["diagnostics"]["rule"] == "uniform_legal"


@pytest.mark.parametrize("policy_class", CONTRACT_POLICIES)
def test_every_policy_describes_itself(policy_class):
    described = policy_class().describe()
    assert described["interface_version"] == POLICY_INTERFACE_VERSION
    assert described["policy_id"].startswith("contract_")
    assert set(described["requirements"]) == {
        "observation",
        "legal_action_mask",
        "public_view",
        "public_events",
        "public_setup",
    }


def test_policy_ref_tokens_round_trip():
    ref = PolicyRef("some_policy", "1.4.2")
    assert PolicyRef.from_token(ref.token) == ref
    assert PolicyRef.from_dict(ref.to_dict()) == ref
    with pytest.raises(PolicyContractError):
        PolicyRef.from_token("no_version_here")


def test_a_policy_may_not_be_instantiated_without_decide():
    class Incomplete(Policy):
        policy_id = "incomplete"
        policy_version = "0"

    with pytest.raises(TypeError):
        Incomplete()


# ---------------------------------------------------------------------------
# Hidden-information permutation (local version of the Agent 4 audit)
# ---------------------------------------------------------------------------


def _permutation_cases(trials: int = 40):
    """Positions paired with a hidden-identity permutation of themselves."""
    rng = random.Random(20260401)
    produced = 0
    for seed in range(400):
        if produced >= trials:
            return
        for ply in (12, 30, 55, 90):
            state = play_random_game_to_ply(seed, ply, rules=EVALUATION_RULES)
            if state.terminal or state.total_moves != ply:
                continue
            clone, info = permute_hidden_identities(state, state.acting_player, rng)
            if not info["valid"] or not info["changed"]:
                continue
            yield state, clone
            produced += 1
            if produced >= trials:
                return


def test_the_permutation_fixture_actually_changes_the_privileged_state():
    """Positive control: without this, every invariance test below is vacuous."""
    cases = list(_permutation_cases(20))
    assert len(cases) == 20
    for state, clone in cases:
        observer = state.acting_player
        assert belief_targets_differ(state, clone, observer)
        privileged = [
            (record.piece_id, record.true_type) for record in state.pieces
        ]
        permuted = [(record.piece_id, record.true_type) for record in clone.pieces]
        assert privileged != permuted


def test_the_public_view_is_invariant_under_hidden_permutation():
    for state, clone in _permutation_cases():
        observer = state.acting_player
        assert build_public_view(state, observer) == build_public_view(clone, observer)


def test_the_whole_policy_input_is_invariant_under_hidden_permutation():
    for state, clone in _permutation_cases():
        policy = FirstLegalActionPolicy()
        first = make_request(state, policy, requirements=ALL_REQUIREMENTS)
        second = make_request(clone, policy, requirements=ALL_REQUIREMENTS)
        assert np.array_equal(first.observation, second.observation)
        assert np.array_equal(first.legal_action_mask, second.legal_action_mask)
        assert first.legal_actions == second.legal_actions
        assert first.public_view == second.public_view
        assert first.public_events == second.public_events
        assert first.public_setup == second.public_setup
        assert first.decision_seed == second.decision_seed


@pytest.mark.parametrize("policy_class", CONTRACT_POLICIES)
def test_policy_decisions_are_invariant_under_hidden_permutation(policy_class):
    policy = policy_class()
    trials = 0
    for state, clone in _permutation_cases():
        first = policy.decide_checked(make_request(state, policy))
        second = policy.decide_checked(make_request(clone, policy))
        assert first.selected_action_id == second.selected_action_id
        assert first.diagnostics == second.diagnostics
        trials += 1
    assert trials >= 20


def test_diagnostics_never_carry_a_hidden_opponent_type():
    state = nonterminal_state(40)
    observer = state.acting_player
    hidden = {
        state.pieces[piece_id].true_type
        for piece_id in hidden_opponent_piece_ids(state, observer)
    }
    assert hidden, "the fixture must contain unresolved opponent pieces"
    for policy_class in CONTRACT_POLICIES:
        policy = policy_class()
        diagnostics = policy.decide(make_request(state, policy)).diagnostics
        for key, value in diagnostics.items():
            assert "true_type" not in key
            assert not isinstance(value, (GameState, PieceRecord))


def test_permutation_invariance_holds_from_both_perspectives():
    """The acting player is not special; a policy for either colour is safe."""
    rng = random.Random(7)
    state = nonterminal_state(45, rules=EVALUATION_RULES)
    for observer in (RED, BLUE):
        clone, info = permute_hidden_identities(state, observer, rng)
        if not info["changed"]:
            continue
        assert build_public_view(state, observer) == build_public_view(clone, observer)
        assert belief_targets_differ(state, clone, observer)


def test_a_synthetic_position_with_a_revealed_piece_keeps_that_piece_visible():
    """Permutation must not disturb a legally revealed identity."""
    state = make_position(
        red={"a4": "marshal", "b4": "scout"},
        blue={"a5": "general", "c5": "captain"},
        revealed={"a5"},
        acting_player=RED,
        rules=EVALUATION_RULES,
    )
    view = build_public_view(state, RED)
    revealed = view.piece_at(square("a5"))
    assert revealed.piece_type == T["general"]
    concealed = view.piece_at(square("c5"))
    assert concealed.piece_type is None

    clone, _ = permute_hidden_identities(state, RED, random.Random(3))
    assert build_public_view(clone, RED) == view
