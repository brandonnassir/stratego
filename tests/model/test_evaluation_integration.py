"""The checkpoint policy inside the real Phase 4 evaluation harness.

Covers Phase 5 gates 17 (`greedy_and_seeded_modes_reproducible`) and the fast
slice of 22 (`phase4_gauntlet_pass`). The full four-baseline gauntlet runs in
`scripts/run_phase5.py`.

Nothing in this module changes Phase 4 semantics. The schedule is built from the
frozen `build_paired_schedule`, the setup bank is the accepted generator, the
pairing mode is `color_swap_same_board`, and the neural policy is handed to the
runner as an instance rather than registered in the Phase 4 catalogue -- so the
catalogue, its audits and its version checks are untouched.
"""

from __future__ import annotations

import pytest

from stratego.engine.constants import BLUE, RED
from stratego.evaluation.match_runner import (
    compare_results,
    play_match,
    replay_stored_match,
    reproduce_match,
    results_digest,
    run_schedule,
)
from stratego.evaluation.match_spec import PAIRING_COLOR_SWAP_SAME_BOARD, build_paired_schedule
from stratego.evaluation.registry import policy_ref
from stratego.evaluation.setup_bank import SetupBank

BASELINE = "random_legal"
PAIR_IDS = (0, 1)


@pytest.fixture(scope="module")
def bank() -> SetupBank:
    return SetupBank.generate(4)


def _units(candidate_ref):
    return build_paired_schedule(candidate_ref, policy_ref(BASELINE), PAIR_IDS)


def _run(policy, bank):
    units = _units(policy.ref)
    specs = [spec for unit in units for spec in unit.matches]
    return specs, [
        play_match(spec, bank=bank, policies={policy.ref.token: policy}) for spec in specs
    ]


# ---------------------------------------------------------------------------
# A paired unit runs cleanly
# ---------------------------------------------------------------------------


def test_the_greedy_policy_plays_a_paired_unit_with_no_failures(greedy_policy, bank):
    specs, results = _run(greedy_policy, bank)
    assert len(results) == 2 * len(PAIR_IDS)
    for result in results:
        assert not result.errored
        assert result.policy_error_category is None
        assert result.plies > 0
        assert result.candidate.token == greedy_policy.ref.token


def test_the_sampling_policy_plays_a_paired_unit_with_no_failures(sampling_policy, bank):
    _, results = _run(sampling_policy, bank)
    for result in results:
        assert not result.errored
        assert result.candidate.token == sampling_policy.ref.token


def test_the_candidate_plays_both_colours_on_the_same_board(greedy_policy, bank):
    units = _units(greedy_policy.ref)
    for unit in units:
        assert unit.pairing_mode == PAIRING_COLOR_SWAP_SAME_BOARD
        first, second = unit.matches
        assert first.candidate_color == RED
        assert second.candidate_color == BLUE
        assert first.setup_pair_id == second.setup_pair_id

        results = [
            play_match(spec, bank=bank, policies={greedy_policy.ref.token: greedy_policy})
            for spec in unit.matches
        ]
        # Same board in both games, colours swapped.
        assert results[0].red_setup == results[1].red_setup
        assert results[0].blue_setup == results[1].blue_setup
        assert results[0].candidate_color != results[1].candidate_color


# ---------------------------------------------------------------------------
# Reproduction
# ---------------------------------------------------------------------------


def test_a_greedy_match_reruns_identically(greedy_policy, bank):
    specs, first = _run(greedy_policy, bank)
    second = [
        play_match(spec, bank=bank, policies={greedy_policy.ref.token: greedy_policy})
        for spec in specs
    ]
    assert compare_results(first, second) == []
    assert results_digest(first) == results_digest(second)


def test_a_seeded_stochastic_match_reruns_identically(sampling_policy, bank):
    """Stochastic but reproducible: the seed contract fixes every draw."""
    specs, first = _run(sampling_policy, bank)
    second = [
        play_match(spec, bank=bank, policies={sampling_policy.ref.token: sampling_policy})
        for spec in specs
    ]
    assert compare_results(first, second) == []
    assert results_digest(first) == results_digest(second)


def test_the_two_modes_are_genuinely_different_policies(greedy_policy, sampling_policy, bank):
    """Otherwise the stochastic gate would be testing the greedy path twice."""
    assert greedy_policy.ref != sampling_policy.ref
    _, greedy_results = _run(greedy_policy, bank)
    _, sampled_results = _run(sampling_policy, bank)
    assert results_digest(greedy_results) != results_digest(sampled_results)


def test_a_stored_row_reproduces_without_the_bank(greedy_policy, bank):
    _, results = _run(greedy_policy, bank)
    for stored in results:
        rebuilt = reproduce_match(stored, policies={greedy_policy.ref.token: greedy_policy})
        assert compare_results([stored], [rebuilt]) == []


def test_every_stored_action_history_replays_through_the_engine(greedy_policy, bank):
    """Pure engine replay: no policy is consulted, so it checks the actions themselves."""
    _, results = _run(greedy_policy, bank)
    for stored in results:
        assert replay_stored_match(stored) == []


def test_a_schedule_run_reports_zero_illegal_actions(greedy_policy, bank):
    units = _units(greedy_policy.ref)
    specs = [spec for unit in units for spec in unit.matches]
    summary = run_schedule(
        specs, bank, policies={greedy_policy.ref.token: greedy_policy}, worker_count=1
    )
    assert summary.policy_errors == 0
    assert summary.illegal_policy_actions == 0
    assert summary.matches_run == len(specs)
    assert summary.paired_units_run == len(PAIR_IDS)


# ---------------------------------------------------------------------------
# Identity and metadata
# ---------------------------------------------------------------------------


def test_the_result_rows_name_the_checkpoint_policy_and_version(greedy_policy, bank):
    _, results = _run(greedy_policy, bank)
    for result in results:
        assert result.candidate.policy_id == "integration_model_v2_greedy"
        assert result.candidate.policy_version == greedy_policy.policy_version
        # The frame change altered which move these weights pick, so it had to
        # take the identity with it: a v1 row and a v2 row are different policies.
        assert result.candidate.policy_version == "0.2.0"
        assert result.opponent.token == policy_ref(BASELINE).token


def test_the_policy_description_carries_the_checkpoint_identity(greedy_policy):
    description = greedy_policy.describe()
    assert description["interface_version"] == "policy_interface_v1"
    assert description["decision_mode"] == "greedy"
    assert description["model_architecture_id"] == "integration_model_v1"
    checkpoint = description["checkpoint"]
    assert checkpoint["rules_version"] == "stratego_project_v1"
    assert checkpoint["observation_version"] == "observation_v2_1_127ch"
    assert checkpoint["state_dict_digest"]


def test_the_neural_policy_is_not_in_the_phase_4_catalogue():
    """Phase 4's catalogue, audits and league membership stay exactly as accepted."""
    from stratego.evaluation.registry import ALL_POLICY_IDS

    assert "integration_model_v1_greedy" not in ALL_POLICY_IDS
    assert "integration_model_v1_sampled" not in ALL_POLICY_IDS
    assert "integration_model_v2_greedy" not in ALL_POLICY_IDS
    assert "integration_model_v2_sampled" not in ALL_POLICY_IDS
