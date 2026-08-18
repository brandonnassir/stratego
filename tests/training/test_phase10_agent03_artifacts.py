"""Regression: Agent 3's accepted utility-fit artifacts stay self-consistent.

Agent 3 produces two fitted utility models and their independent audit, and
selects nothing. These tests pin what that evidence has to keep saying:

- the acceptance status follows from the 19 completion gates, none false;
- every upstream identity the artifacts name equals the live frozen value —
  the sealed corpus digest, the contract bundle, the utility contract, the
  scaler, and the accepted Phase 9 checkpoint before *and* after;
- the exported `setup_utility_v1` artifact is byte-stable (file SHA), is a
  pure own-side scorer, and its coefficient digests recompute exactly;
- the independent audit recorded agreement at its frozen tolerances, the
  deterministic refits were bit-identical, and every negative control fired;
- nothing records a candidate selection, a held-out outcome access, or a
  Phase 9 optimizer step.

Everything is gated on the artifacts existing, so the suite is green both
before and after Agent 3 runs.
"""

import hashlib
import json
from pathlib import Path

import pytest

from stratego.training import phase10_contract as pc
from stratego.training import phase10_utility_fit as fit
from stratego.training.phase10_utility import document_digest
from tests.training.phase10_frozen_digests import (
    CONTRACT_BUNDLE_DIGEST,
    CONTRACT_DIGESTS,
    TRAIT_SCALER_DIGEST,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_10_data"

ARTIFACTS = {
    "models": DATA_DIRECTORY / "agent_03_utility_models.json",
    "audit": DATA_DIRECTORY / "agent_03_utility_audit.json",
    "acceptance": DATA_DIRECTORY / "agent_03_acceptance.json",
}

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in ARTIFACTS.values()),
    reason="Phase 10 Agent 3 artifacts have not been written yet",
)

ACCEPTED_CORPUS_CONTENT_DIGEST = (
    "1977bb6f5e2611b0498c7976f6129718fdfe7f6f44216f3b3f1932c8192b3c50"
)

EXPECTED_GATES = {
    "agents1_2_pass",
    "corpus_digest_verified",
    "corpus_train_only",
    "trait_vectors_reconstructed",
    "standardizer_train_only",
    "model_f_fit_complete",
    "model_t_fit_complete",
    "coefficients_finite",
    "objectives_finite",
    "independent_objective_audit_pass",
    "red_blue_orientation_audit_pass",
    "deterministic_refit_pass",
    "negative_controls_fire",
    "production_scorer_own_side_only",
    "no_validation_outcome_access",
    "no_test_outcome_access",
    "no_candidate_selection",
    "phase9_checkpoint_unchanged",
    "full_suite_green",
}


@pytest.fixture(scope="module")
def models():
    return json.loads(ARTIFACTS["models"].read_text())


@pytest.fixture(scope="module")
def audit():
    return json.loads(ARTIFACTS["audit"].read_text())


@pytest.fixture(scope="module")
def acceptance():
    return json.loads(ARTIFACTS["acceptance"].read_text())


@pytest.fixture(scope="module")
def exported(models):
    path = REPOSITORY_ROOT / models["fitted_artifact"]["path"]
    assert path.exists(), "the fitted setup_utility_v1 artifact is missing"
    return json.loads(path.read_text())


class TestAcceptance:
    def test_status_follows_from_the_gates(self, acceptance):
        gates = acceptance["completion_gates"]
        false_gates = sorted(name for name, value in gates.items() if not value)
        assert acceptance["false_gates"] == false_gates
        assert acceptance["status"] == ("PASS" if not false_gates else "FAIL")
        assert acceptance["gates_total"] == len(gates)
        assert acceptance["gates_true"] == sum(1 for value in gates.values() if value)

    def test_the_gate_set_is_exactly_the_frozen_one(self, acceptance):
        assert set(acceptance["completion_gates"]) == EXPECTED_GATES
        assert acceptance["gates_total"] == 19

    def test_accepted_run_has_no_false_gate(self, acceptance):
        assert acceptance["status"] == "PASS"
        assert acceptance["false_gates"] == []

    def test_frozen_inputs_match_the_live_freeze(self, acceptance):
        frozen = acceptance["frozen_inputs"]
        assert frozen["corpus_content_digest"] == ACCEPTED_CORPUS_CONTENT_DIGEST
        assert frozen["contract_bundle_digest"] == CONTRACT_BUNDLE_DIGEST
        assert frozen["utility_contract_digest"] == CONTRACT_DIGESTS["phase10_setup_utility_v1"]
        assert frozen["scaler_digest"] == TRAIT_SCALER_DIGEST
        assert frozen["phase9_checkpoint_sha256"] == pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256
        assert frozen["phase9_model_state_digest"] == pc.ACCEPTED_PHASE9_MODEL_STATE_DIGEST

    def test_discipline_counters_are_clean(self, acceptance):
        discipline = acceptance["discipline"]
        for counter in (
            "c1_optimizer_steps",
            "candidates_selected",
            "held_out_bases_in_fitting",
            "human_games_used",
            "neural_inference_on_either_bank",
            "games_played",
            "test_bank_outcome_access",
            "validation_bank_outcome_access",
            "hyperparameter_search_runs",
        ):
            assert discipline[counter] == 0, counter
        assert discipline["utility_models_fit"] == 2
        assert discipline["fits_per_model_canonical"] == 1

    def test_bank_access_was_structural_only(self, acceptance):
        for entry in acceptance["bank_access_log"]:
            assert entry["purpose"] == "digest_computation"
            assert entry["neural"] is False
            assert entry["outcomes"] is False

    def test_phase9_is_preserved(self, acceptance):
        preservation = acceptance["phase9_preservation"]
        assert preservation["unchanged"] is True
        for side in ("before", "after"):
            assert preservation[side]["sha256"] == pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256
            assert (
                preservation[side]["model_state_digest"]
                == pc.ACCEPTED_PHASE9_MODEL_STATE_DIGEST
            )

    def test_corpus_is_preserved(self, acceptance):
        preservation = acceptance["corpus_preservation"]
        assert preservation["before"] == ACCEPTED_CORPUS_CONTENT_DIGEST
        assert preservation["after"] == ACCEPTED_CORPUS_CONTENT_DIGEST
        assert preservation["state_after"] == "SEALED"
        assert preservation["byte_identical_content"] is True

    def test_the_suite_ran_and_was_green(self, acceptance):
        suite = acceptance["suite"]
        assert suite["returncode"] == 0
        assert suite["failed"] == 0
        assert suite["passed"] > 0

    def test_carried_forward_obligations_survive(self, acceptance):
        obligations = " ".join(
            entry["obligation"] for entry in acceptance["carried_forward_obligations"]
        )
        assert "selector_audit" in obligations
        assert "MPS" in obligations or "backend" in obligations

    def test_handoff_names_the_six_candidates_and_the_loader(self, acceptance):
        handoff = acceptance["handoff_to_agent_4"]
        candidate_ids = [entry["candidate_id"] for entry in handoff["six_candidates"]]
        assert candidate_ids == ["P10-A", "P10-B", "P10-C", "P10-D", "P10-E", "P10-F"]
        assert "own colour" in handoff["scoring_contract"]
        assert handoff["fitted_utility"]["scaler_digest"] == TRAIT_SCALER_DIGEST


class TestFittedArtifact:
    def test_file_sha_matches_the_acceptance_record(self, models, acceptance):
        path = REPOSITORY_ROOT / models["fitted_artifact"]["path"]
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == models["fitted_artifact"]["sha256"]
        assert observed == acceptance["new_digests"]["setup_utility_v1_file_sha256"]

    def test_the_artifact_is_a_pure_own_side_scorer(self, exported):
        assert fit.own_side_only_findings(exported) == []
        scorer = fit.SetupUtilityScorer(exported)
        assert scorer.utility("model_F", "red", "F00") is not None

    def test_coefficient_digests_recompute(self, exported, acceptance):
        for model_id in ("model_F", "model_T"):
            entry = exported["models"][model_id]
            document = {
                "utility_version": entry["utility_version"],
                "model_id": entry["model_id"],
                "colour_order": entry["colour_order"],
                "family_order": entry["family_order"],
                "feature_order": entry["feature_order"],
                "red_first_intercept": entry["red_first_intercept"],
                "family_offsets_raw": entry["family_offsets_raw"],
                "trait_weights": entry["trait_weights"],
            }
            observed = document_digest(document)
            assert observed == entry["coefficient_digest"]
            assert (
                observed
                == acceptance["new_digests"][f"{model_id}_coefficient_digest"]
            )

    def test_the_scaler_travels_with_the_artifact(self, exported):
        assert exported["scaler_digest"] == TRAIT_SCALER_DIGEST
        assert exported["scaler"]["base_count"] == 6400
        assert exported["scaler"]["split"] == "train"
        assert exported["scaler"]["ddof"] == 0
        assert exported["scaler"]["zero_std_features"] == []

    def test_effective_offsets_are_the_centered_raw_ones(self, exported):
        for entry in exported["models"].values():
            for raw_row, effective_row in zip(
                entry["family_offsets_raw"], entry["family_offsets_effective"]
            ):
                mean = sum(raw_row) / len(raw_row)
                for raw, effective in zip(raw_row, effective_row):
                    assert abs((raw - mean) - effective) <= 1e-15
                assert abs(mean) <= 1e-8  # the L2-forced self-centering

    def test_parameter_layouts_are_frozen(self, exported):
        model_f = exported["models"]["model_F"]
        model_t = exported["models"]["model_T"]
        assert model_f["trait_weights"] is None
        assert len(model_f["family_offsets_raw"]) == 2
        assert all(len(row) == 16 for row in model_f["family_offsets_raw"])
        assert len(model_t["trait_weights"]) == 2
        assert all(len(row) == 47 for row in model_t["trait_weights"])
        assert model_t["feature_order"][0] == "flag_rank"
        assert model_t["feature_order"][-1] == "unconventional_feature_count"

    def test_the_allowlist_is_recorded_and_was_respected(self, models):
        assert models["fitting_input_allowlist"] == {
            model_id: list(fields)
            for model_id, fields in fit.FIT_INPUT_ALLOWLIST.items()
        }
        for model_id, accessed in models["accessed_fields"].items():
            assert set(accessed) <= set(fit.FIT_INPUT_ALLOWLIST[model_id])
        assert "red_score" in models["forbidden_fitting_fields"]
        assert "red_final_fingerprint" in models["forbidden_fitting_fields"]

    def test_no_model_was_selected(self, models):
        assert set(models["models"]) == {"model_F", "model_T"}
        assert not any("selected" in key or "winner" in key for key in models)
        assert "rank nothing" in models["no_model_selection"]


class TestIndependentAudit:
    def test_every_record_was_audited_clean(self, audit):
        record_audit = audit["record_audit"]
        assert record_audit["records_audited"] == 16384
        assert record_audit["all_pass"] is True
        assert record_audit["violations"] == []
        counts = record_audit["result_counts"]
        assert counts["red_win"] + counts["draw"] + counts["red_loss"] == 16384

    def test_designs_agree_exactly(self, audit):
        for model_id, agreement in audit["design_agreement"].items():
            for key, value in agreement.items():
                if key.endswith("_exact"):
                    assert value is True, (model_id, key)
        assert audit["design_agreement"]["model_T"]["red_features_max_abs_difference"] == 0.0

    def test_model_audits_passed_at_the_frozen_tolerances(self, audit):
        for model_id in ("model_F", "model_T"):
            outcome = audit["model_audits"][model_id]
            assert outcome["all_pass"] is True, outcome["checks"]
            assert outcome["objective_abs_difference"] <= 1e-10
            assert outcome["gradient"]["max_abs"] <= 1e-6
            assert outcome["finite_difference"]["all_within_tolerance"] is True

    def test_every_negative_control_fired(self, audit):
        controls = audit["negative_controls"]
        assert len(controls) >= 6
        names = {control["control"] for control in controls}
        assert {
            "orientation_swap",
            "wrong_draw_target",
            "held_out_scaler",
            "permuted_trait_column",
            "altered_family_id",
            "altered_coefficient",
        } <= names
        for control in controls:
            assert control["detected"] is True, control["control"]

    def test_the_orientation_reversal_control_failed_loudly(self, audit):
        control = next(
            entry
            for entry in audit["negative_controls"]
            if entry["control"] == "orientation_swap"
        )
        assert control["objective_abs_difference"] > 1e-3
        assert control["logit_max_shift"] > 1e-3

    def test_draws_were_handled_exactly_as_frozen(self, audit):
        control = next(
            entry
            for entry in audit["negative_controls"]
            if entry["control"] == "wrong_draw_target"
        )
        assert control["draws_in_corpus"] == audit["record_audit"]["result_counts"]["draw"]
        assert control["target_mismatches_detected"] == control["draws_in_corpus"]

    def test_refits_were_bit_identical_across_processes(self, audit):
        refit = audit["deterministic_refit"]
        assert refit["all_identical"] is True
        for model_id in ("model_F", "model_T"):
            comparison = refit["comparisons"][model_id]
            assert comparison["identical"] is True
            assert comparison["max_abs_difference"] == 0.0
            assert comparison["fits"] == 3
            assert comparison["digests_identical"] is True
            assert len(set(comparison["objectives"])) == 1

    def test_the_scorer_decomposition_held(self, audit):
        decomposition = audit["scorer_decomposition"]
        assert decomposition["max_abs_difference"] <= 1e-10
        assert decomposition["samples"] >= 16

    def test_feature_reconstruction_covered_the_whole_library(self, audit):
        features = audit["feature_reconstruction"]
        assert features["library_entries_reconstructed"] == 8000
        assert features["stored_trait_mismatches"] == 0
        assert features["feature_names_match_frozen"] is True
        assert features["train_matrix_shape"] == [6400, 47]
        assert features["independent_scaler"]["mean_matches_frozen_exactly"] is True
        assert features["independent_scaler"]["std_matches_frozen_exactly"] is True
        assert features["unique_corpus_base_splits"] == ["train"]
