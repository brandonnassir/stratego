"""Regression: Agent 7's accepted evaluation artifacts stay self-consistent.

Agent 7 opens the sealed test split and the Phase 4 bank exactly once and
freezes the result. These tests pin what that freeze *means*:

- the four artifacts exist together, name the same checkpoint, and carry a
  status justified by their own recorded gates;
- every frozen threshold in the artifacts equals the live
  `warmstart_contract.acceptance_thresholds()` — a silently relaxed gate
  cannot survive the suite;
- the recorded measurements actually satisfy the gates the artifacts claim
  they satisfy, re-derived here from the stored numbers;
- the sealed-test statistics used the frozen test bootstrap seed and the
  train-fitted value prior, never a refit;
- the evaluation identities (bank digest, corpus digests, checkpoint SHA-256)
  match the accepted upstream records.

Everything is gated on the artifacts existing, so the suite is green both
before and after Agent 7 runs.
"""

import json
from pathlib import Path

import pytest

from stratego.training import warmstart_contract as wc
from stratego.training.warmstart_seed import TEST_BOOTSTRAP_SEED

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_8_data"

ARTIFACTS = {
    "heldout": DATA_DIRECTORY / "agent_07_heldout_metrics.json",
    "random": DATA_DIRECTORY / "agent_07_random_evaluation.json",
    "acceptance": DATA_DIRECTORY / "agent_07_final_acceptance.json",
    "handoff": DATA_DIRECTORY / "agent_07_phase9_handoff.json",
}

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in ARTIFACTS.values()),
    reason="Agent 7's evaluation artifacts have not been produced yet",
)


@pytest.fixture(scope="module")
def artifacts() -> dict:
    return {name: json.loads(path.read_text()) for name, path in ARTIFACTS.items()}


class TestArtifactCoherence:
    def test_every_artifact_names_phase_8_agent_7(self, artifacts):
        for name, payload in artifacts.items():
            assert payload["phase"] == 8, name
            assert payload["agent"] == 7, name

    def test_one_checkpoint_identity_across_all_artifacts(self, artifacts):
        manifest = json.loads(
            (DATA_DIRECTORY / "agent_06_checkpoint_manifest.json").read_text()
        )
        accepted_sha = manifest["checkpoint_sha256"]
        assert artifacts["heldout"]["checkpoint_sha256"] == accepted_sha
        assert (
            artifacts["acceptance"]["prerequisite_digests"]["checkpoint_sha256"]
            == accepted_sha
        )
        assert artifacts["handoff"]["frozen_checkpoint"]["sha256"] == accepted_sha
        assert (
            artifacts["handoff"]["frozen_checkpoint"]["canonical_untrained_checkpoint"][
                "sha256"
            ]
            == manifest["initial_checkpoint_sha256"]
        )

    def test_status_is_justified_by_the_recorded_gates(self, artifacts):
        acceptance = artifacts["acceptance"]
        gates = acceptance["completion_gates"]
        assert acceptance["gates_total"] == len(gates)
        assert acceptance["gates_true"] == sum(bool(value) for value in gates.values())
        expected = "PASS" if all(gates.values()) else "FAIL"
        assert acceptance["status"] == expected
        assert acceptance["recommendation"] == expected

    def test_handoff_readiness_matches_the_recommendation(self, artifacts):
        acceptance = artifacts["acceptance"]["status"]
        readiness = artifacts["handoff"]["phase_9_readiness"]
        assert readiness == ("READY TO PLAN" if acceptance == "PASS" else "BLOCKED")

    def test_handoff_names_every_frozen_phase_8_identity(self, artifacts):
        identities = artifacts["handoff"]["frozen_identities"]
        for version in (
            "warmstart_training_contract_v1",
            "synthetic_warmstart_corpus_v1",
            "warmstart_decision_sampler_v1",
            "warmstart_example_v1",
            "warmstart_trainer_v1",
            "warmstart_checkpoint_v1",
            "warmstart_train_config_v1",
            "warmstart_eval_v1",
        ):
            assert identities[version] == version

    def test_corpus_identity_matches_the_accepted_checkpoint_manifest(self, artifacts):
        manifest = json.loads(
            (DATA_DIRECTORY / "agent_06_checkpoint_manifest.json").read_text()
        )
        accepted = manifest["identities"]["corpus_digests"]
        recorded = artifacts["handoff"]["corpus"]["digests"]
        for field in ("content_digest", "metadata_digest", "commit_index_digest"):
            assert recorded[field] == accepted[field]
        assert (
            artifacts["handoff"]["corpus"]["resolver"]
            == "stratego.training.synthetic_corpus.default_corpus_root()"
        )


class TestFrozenThresholds:
    def test_acceptance_thresholds_equal_the_live_contract(self, artifacts):
        # The artifact stores the thresholds it was judged against; they must
        # be the frozen contract's, byte for byte, so a relaxed gate is loud.
        recorded = artifacts["acceptance"]["acceptance_thresholds"]
        live = json.loads(json.dumps(wc.acceptance_thresholds()))
        assert recorded == live

    def test_random_gate_thresholds_are_the_frozen_ones(self, artifacts):
        thresholds = artifacts["random"]["random_gate"]["thresholds"]
        frozen = wc.acceptance_thresholds()["playing_strength_vs_random"]
        for name in (
            "effective_win_rate_min",
            "red_effective_win_rate_min",
            "blue_effective_win_rate_min",
            "paired_bootstrap_lower_bound_exclusive",
        ):
            assert thresholds[name] == frozen[name]

    def test_vs_init_thresholds_are_the_frozen_ones(self, artifacts):
        thresholds = artifacts["random"]["final_vs_initialisation"]["thresholds"]
        frozen = wc.acceptance_thresholds()["improvement_over_initialization"]
        assert thresholds["effective_win_rate_min"] == frozen["effective_win_rate_min"]
        assert (
            thresholds["paired_bootstrap_lower_bound_exclusive"]
            == frozen["paired_bootstrap_lower_bound_exclusive"]
        )


class TestSealedTestEvaluation:
    def test_the_full_sealed_universe_was_evaluated(self, artifacts):
        contract = json.loads(
            (DATA_DIRECTORY / "agent_03_example_contract.json").read_text()
        )
        expected_examples = contract["universe"]["counts"]["test"]
        headline = artifacts["heldout"]["headline"]
        assert headline["split"] == "test"
        assert headline["examples"] == expected_examples
        assert headline["games"] == 4000

    def test_the_frozen_test_bootstrap_seed_was_used(self, artifacts):
        bootstrap = artifacts["heldout"]["bootstrap"]
        assert bootstrap["seed"] == TEST_BOOTSTRAP_SEED
        assert bootstrap["unit"] == "game"
        assert bootstrap["replicates"] == wc.BOOTSTRAP_REPLICATES
        for interval in bootstrap["intervals"].values():
            assert interval.get("seed") == TEST_BOOTSTRAP_SEED

    def test_the_value_prior_is_agent_3s_frozen_train_prior(self, artifacts):
        baselines = json.loads(
            (DATA_DIRECTORY / "agent_03_validation_baselines.json").read_text()
        )
        frozen_prior = baselines["value_prior"]["prior_win_draw_loss"]
        assert artifacts["heldout"]["value_prior"] == frozen_prior
        assert "not refit" in artifacts["heldout"]["value_prior_source"]

    def test_recorded_gates_rederive_from_the_recorded_metrics(self, artifacts):
        heldout = artifacts["heldout"]
        headline = heldout["headline"]
        thresholds = wc.acceptance_thresholds()
        gates = heldout["gates"]
        assert gates["policy_ce_ratio_at_most_0_90"] == (
            headline["policy"]["ce_ratio"]
            <= thresholds["policy_learning"]["ce_ratio_vs_uniform_legal_max"]
        )
        assert gates["policy_top1_beats_uniform_expected"] == (
            headline["policy"]["model_top1"] > headline["policy"]["baseline_expected_top1"]
        )
        assert gates["value_ce_ratio_at_most_0_98"] == (
            headline["value"]["ce_ratio"]
            <= thresholds["value_learning"]["ce_ratio_vs_train_prior_max"]
        )
        assert gates["value_brier_beats_train_prior"] == (
            headline["value"]["model_brier"] < headline["value"]["baseline_brier"]
        )
        assert gates["belief_ce_ratio_at_most_0_98"] == (
            headline["belief"]["ce_ratio"]
            <= thresholds["belief_learning"]["ce_ratio_vs_remaining_count_prior_max"]
        )
        assert gates["belief_top1_beats_remaining_count_prior"] == (
            headline["belief"]["model_top1"] > headline["belief"]["baseline_top1"]
        )

    def test_stability_gate_rederives_from_the_recorded_fraction(self, artifacts):
        stability = artifacts["heldout"]["stability"]
        threshold = wc.acceptance_thresholds()["stability"]
        assert stability["non_finite_examples"] == 0
        fraction = stability["max_legal_probability"]["fraction_above_0_999"]
        assert artifacts["heldout"]["gates"]["collapse_fraction_below_0_95"] == (
            fraction < threshold["fraction_above_threshold_max_exclusive"]
        )

    def test_belief_breakdowns_are_complete_partitions(self, artifacts):
        heldout = artifacts["heldout"]
        pieces = heldout["headline"]["belief"]["pieces"]
        by_type = heldout["belief_breakdown"]["counts_by_true_type"]
        assert sum(by_type.values()) == pieces
        by_bucket = heldout["belief_breakdown"]["metrics_by_progress_bucket"]
        assert sum(entry["pieces"] for entry in by_bucket.values()) == pieces

    def test_family_stratification_partitions_every_example(self, artifacts):
        strat = artifacts["heldout"]["family_stratification"]
        total = sum(entry["examples"] for entry in strat["by_family"].values())
        assert total == artifacts["heldout"]["headline"]["examples"]


class TestPlayingStrengthEvaluations:
    def test_random_gate_covered_the_whole_frozen_bank(self, artifacts):
        gate = artifacts["random"]["random_gate"]
        assert gate["harness"]["setup_bank_digest"] == wc.EXPECTED_PHASE4_BANK_DIGEST
        assert gate["summary"]["paired_units"] == 1024
        assert gate["summary"]["games"] == 2048

    def test_random_gate_results_rederive_their_gates(self, artifacts):
        gate = artifacts["random"]["random_gate"]
        summary = gate["summary"]
        thresholds = gate["thresholds"]
        assert gate["gates"]["effective_win_rate_at_least_0_950"] == (
            summary["effective_win_rate"] >= thresholds["effective_win_rate_min"]
        )
        assert gate["gates"]["paired_bootstrap_lower_bound_above_0_900"] == (
            summary["confidence_interval"]["lower"]
            > thresholds["paired_bootstrap_lower_bound_exclusive"]
        )
        assert gate["gates"]["model_failures_zero"] == (summary["policy_errors"] == 0)

    def test_effective_win_rate_matches_the_recorded_counts(self, artifacts):
        for section in ("random_gate", "final_vs_initialisation"):
            summary = artifacts["random"][section]["summary"]
            games = summary["wins"] + summary["draws"] + summary["losses"]
            assert games == summary["games"]
            expected = (summary["wins"] + 0.5 * summary["draws"]) / games
            assert abs(summary["effective_win_rate"] - expected) < 1e-12

    def test_vs_init_played_at_least_the_frozen_minimum(self, artifacts):
        summary = artifacts["random"]["final_vs_initialisation"]["summary"]
        frozen = wc.acceptance_thresholds()["improvement_over_initialization"]
        assert summary["paired_units"] >= frozen["paired_setup_cases_min"]
        assert summary["games"] >= frozen["games_min"]

    def test_additional_baselines_are_reported_not_gated(self, artifacts):
        baselines = artifacts["random"]["additional_baselines"]
        assert "hard gate" in baselines["role"] or "diagnostics" in baselines["role"]
        for tier in ("basic_heuristic", "tactical_rule_based", "strategic_rule_based"):
            assert tier in baselines["tiers"]
            assert baselines["tiers"][tier]["games"] >= 512


class TestDisciplineEvidence:
    def test_the_audit_recorded_zero_pre_agent_7_heldout_contact(self, artifacts):
        audit = artifacts["acceptance"]["training_discipline_audit"]
        assert audit["test_model_inference_before_agent_7"] == 0
        assert audit["phase4_neural_games_before_agent_7"] == 0
        assert audit["candidate_count"]["considered"] <= 6

    def test_agent_7s_own_test_access_covers_the_sealed_universe(self, artifacts):
        access = artifacts["heldout"]["model_input_access"]
        assert access["test_examples_evaluated_by_model"] == (
            artifacts["heldout"]["headline"]["examples"]
        )
        assert artifacts["heldout"]["authorized_access"]["purpose"] == "final_evaluation"
        assert artifacts["heldout"]["authorized_access"]["phase8_agent"] == 7
