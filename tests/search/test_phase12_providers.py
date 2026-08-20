"""Phase 12 belief providers: interface, validity, determinism, oracle rules."""

import numpy as np
import pytest

from stratego.belief.phase11b.heads import ExistingBeliefHead
from stratego.belief.phase11b.interface import Phase11BBeliefModel
from stratego.evaluation.phase11_baselines import remaining_counts
from stratego.evaluation.phase11_public_state import hidden_opponent_pieces
from stratego.search.phase12.contract import (
    PROVIDER_AGENT1C,
    PROVIDER_ORACLE,
    PROVIDER_ORIGINAL_PHASE11,
    PROVIDER_REMAINING_COUNT,
    Phase12SearchError,
)
from stratego.search.phase12.providers import (
    AdapterNeuralBeliefProvider,
    OracleBeliefProvider,
    RemainingCountBeliefProvider,
    build_belief_provider,
)
from stratego.training.phase11_contract import IMMOVABLE_RANK_INDICES, RANK_COUNT

from tests.helpers import nonterminal_state
from tests.search.conftest import public_state_for


def _check_assignments(document, assignments, n):
    """Coverage, inventory and immovability of every assignment."""
    hidden = {int(piece["piece_slot"]): piece for piece in hidden_opponent_pieces(document)}
    counts = remaining_counts(document)
    assert len(assignments) == n
    for assignment in assignments:
        assert set(assignment) == set(hidden)
        observed = [0] * RANK_COUNT
        for slot, rank in assignment.items():
            observed[rank] += 1
            if hidden[slot]["has_moved"]:
                assert rank not in IMMOVABLE_RANK_INDICES
        assert tuple(observed) == tuple(counts)


def test_remaining_count_assignments_valid_and_seed_deterministic(midgame_public):
    provider = RemainingCountBeliefProvider()
    assert provider.provider_id == PROVIDER_REMAINING_COUNT
    assert provider.uses_hidden_truth is False

    first = provider.sample_assignments(midgame_public, 5, seed=7)
    again = provider.sample_assignments(midgame_public, 5, seed=7)
    other = provider.sample_assignments(midgame_public, 5, seed=8)
    assert first == again
    _check_assignments(midgame_public.public_state_document, first, 5)
    # A different seed starts a different ordinal walk. (Equality would not
    # be an error in principle, but for this position it would be absurd.)
    assert first != other

    marginals = provider.predict_marginals(midgame_public)
    assert set(marginals) == {
        int(piece["piece_slot"])
        for piece in hidden_opponent_pieces(midgame_public.public_state_document)
    }
    for row in marginals.values():
        assert row.shape == (RANK_COUNT,)
        assert abs(float(row.sum()) - 1.0) < 1e-12


def test_adapter_provider_runs_the_accepted_interface(random_c1, midgame_public):
    head = ExistingBeliefHead.from_accepted(random_c1)
    model = Phase11BBeliefModel(random_c1, head, candidate_id="test_head", device="cpu")
    provider = AdapterNeuralBeliefProvider(
        model, provider_id=PROVIDER_ORIGINAL_PHASE11, identity={"weights": "random"}
    )

    marginals = provider.predict_marginals(midgame_public)
    for row in marginals.values():
        assert row.shape == (RANK_COUNT,)
        assert np.isfinite(row).all()
        assert abs(float(row.sum()) - 1.0) < 1e-9

    first = provider.sample_assignments(midgame_public, 4, seed=11)
    again = provider.sample_assignments(midgame_public, 4, seed=11)
    assert first == again
    _check_assignments(midgame_public.public_state_document, first, 4)


def test_adapter_provider_rejects_unknown_id(random_c1):
    head = ExistingBeliefHead.from_accepted(random_c1)
    model = Phase11BBeliefModel(random_c1, head, candidate_id="test_head", device="cpu")
    with pytest.raises(Phase12SearchError):
        AdapterNeuralBeliefProvider(model, provider_id="mystery", identity={})


def test_factory_refuses_oracle_in_production():
    with pytest.raises(Phase12SearchError):
        build_belief_provider(PROVIDER_ORACLE, production=True)
    provider = build_belief_provider(PROVIDER_ORACLE, production=False)
    assert isinstance(provider, OracleBeliefProvider)
    assert provider.uses_hidden_truth is True


def test_factory_checks_the_original_head_digest(random_c1):
    # Random weights are not the accepted Phase 9 belief head; the factory
    # must refuse to call them `original_phase11`.
    with pytest.raises(Phase12SearchError):
        build_belief_provider(
            PROVIDER_ORIGINAL_PHASE11, encoder=random_c1, production=True
        )


def test_factory_requires_components():
    with pytest.raises(Phase12SearchError):
        build_belief_provider(PROVIDER_AGENT1C, production=True)
    with pytest.raises(Phase12SearchError):
        build_belief_provider("mystery_provider", production=True)


def test_oracle_requires_the_offline_flag():
    with pytest.raises(Phase12SearchError):
        OracleBeliefProvider()
    provider = OracleBeliefProvider(offline_diagnostic=True)
    with pytest.raises(Phase12SearchError):
        provider.predict_marginals(None)


def test_oracle_has_no_public_sampling_path(midgame_public):
    provider = OracleBeliefProvider(offline_diagnostic=True)
    with pytest.raises(Phase12SearchError):
        provider.sample_assignments(midgame_public, 3, seed=0)


def test_oracle_returns_the_true_hidden_ranks(midgame_state, midgame_public):
    provider = OracleBeliefProvider(offline_diagnostic=True)
    assignments = provider.sample_assignments_privileged(
        midgame_state, midgame_public, 3, seed=0
    )
    assert len(assignments) == 3
    assert assignments[0] == assignments[1] == assignments[2]
    _check_assignments(midgame_public.public_state_document, assignments, 3)
    truth = assignments[0]
    observer = midgame_state.acting_player
    for record in midgame_state.pieces:
        if record.owner == observer or not record.alive or record.known_to(observer):
            continue
        slot = record.piece_id % 40
        assert truth[slot] == record.true_type


def test_oracle_refuses_a_mismatched_state(midgame_public):
    provider = OracleBeliefProvider(offline_diagnostic=True)
    other_state = nonterminal_state(30, first_seed=50)
    with pytest.raises(Phase12SearchError):
        provider.sample_assignments_privileged(other_state, midgame_public, 1, seed=0)
