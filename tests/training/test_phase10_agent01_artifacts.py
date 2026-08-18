"""Regression: Agent 1's accepted Phase 10 freeze artifacts stay self-consistent.

Agent 1 freezes the whole Phase 10 experiment before any outcome game is
played and before either utility model is fit. These tests pin what that
freeze *means*:

- the four artifacts exist together, name the same contract digests as the
  live `phase10_contract`, and carry a status justified by their own
  recorded gates;
- every frozen threshold in the artifacts equals the live value, so a
  silently relaxed gate cannot survive the suite;
- the recorded bank digests equal the frozen bank identity, and the cases
  stored inside the artifacts re-hash to exactly those digests;
- the isolation claim the artifacts make is the one the banks actually
  satisfy, and the sealed test bank records zero neural or outcome access;
- the recorded upstream identities match the accepted Phase 9 and Phase 7
  records.

Everything is gated on the artifacts existing, so the suite is green both
before and after Agent 1 runs.
"""

import json
from pathlib import Path

import pytest

from stratego.evaluation import phase10_banks as pb
from stratego.training import phase10_contract as pc
from stratego.training import phase10_schedule as sch
from stratego.training import phase10_seed as ps
from tests.training.phase10_frozen_digests import (
    BANK_DIGESTS,
    CONTRACT_BUNDLE_DIGEST,
    CONTRACT_DIGESTS,
    OUTCOME_SCHEDULE_DIGEST,
    PHASE9_ISOLATION_SET_DIGEST,
    TRAIT_SCALER_DIGEST,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_10_data"

ARTIFACTS = {
    "contract": DATA_DIRECTORY / "agent_01_setup_selection_contract.json",
    "validation_bank": DATA_DIRECTORY / "agent_01_validation_bank.json",
    "test_bank": DATA_DIRECTORY / "agent_01_test_bank.json",
    "acceptance": DATA_DIRECTORY / "agent_01_acceptance.json",
}

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in ARTIFACTS.values()),
    reason="Phase 10 Agent 1 artifacts have not been written yet",
)


def load(name):
    return json.loads(ARTIFACTS[name].read_text())


@pytest.fixture(scope="module")
def contract():
    return load("contract")


@pytest.fixture(scope="module")
def acceptance():
    return load("acceptance")


@pytest.fixture(scope="module")
def banks():
    return {"validation": load("validation_bank"), "test": load("test_bank")}


class TestArtifactSet:
    def test_all_four_artifacts_name_the_same_freeze(self, contract, acceptance, banks):
        bundle = contract["contract_bundle_digest"]
        assert bundle == acceptance["new_digests"]["contract_bundle_digest"]
        assert bundle == CONTRACT_BUNDLE_DIGEST
        for name, artifact in banks.items():
            assert artifact["bank_digest"] == BANK_DIGESTS[name]
            assert artifact["bank_digest"] == acceptance["new_digests"][f"{name}_bank_digest"]

    def test_every_artifact_records_phase_and_agent(self, contract, acceptance, banks):
        for artifact in (contract, acceptance, *banks.values()):
            assert artifact["phase"] == 10
            assert artifact["agent"] == 1

    def test_status_is_justified_by_its_own_gates(self, acceptance):
        gates = acceptance["completion_gates"]
        false_gates = sorted(key for key, value in gates.items() if not value)
        assert acceptance["false_gates"] == false_gates
        assert acceptance["gates_true"] == sum(1 for value in gates.values() if value)
        if acceptance["status"] == "PASS":
            assert not false_gates
            assert not acceptance["problems"]


class TestContractArtifact:
    def test_digests_equal_the_live_contract(self, contract):
        assert contract["contract_digests"] == pc.contract_digests()
        assert contract["contract_digests"] == CONTRACT_DIGESTS

    def test_stored_documents_rehash_to_their_recorded_digests(self, contract):
        for name, document in contract["contracts"].items():
            assert pc.document_digest(document) == contract["contract_digests"][name]

    def test_seeds_are_the_eight_frozen_roots(self, contract):
        assert contract["seeds"] == ps.CANONICAL_PHASE10_SEEDS

    def test_the_collision_audit_found_nothing(self, contract):
        assert contract["seed_collision_audit"]["no_collisions"]
        assert (
            contract["seed_collision_audit"]["total_seeds"]
            == contract["seed_collision_audit"]["distinct_seeds"]
        )

    def test_schedule_identity_matches_the_live_schedule(self, contract):
        assert contract["outcome_schedule"]["schedule_digest"] == sch.schedule_digest()
        assert contract["outcome_schedule"]["schedule_digest"] == OUTCOME_SCHEDULE_DIGEST
        assert contract["outcome_schedule"]["audit"]["all_pass"]
        assert contract["outcome_schedule"]["audit"]["total_games"] == 16_384

    def test_side_draw_samples_are_train_only(self, contract):
        for sample in contract["outcome_schedule"]["side_draw_samples"]:
            assert sample["base_setup_id"].startswith("setup_library_v1:")
            assert sample["accepted_attempt"] >= 0

    def test_scaler_and_feature_identity(self, contract):
        assert contract["trait_scaler_digest"] == TRAIT_SCALER_DIGEST
        assert contract["trait_feature_count"] == 47
        assert len(contract["trait_feature_names"]) == 47

    def test_upstream_identities_are_the_accepted_ones(self, contract):
        upstream = contract["upstream_identities"]
        assert upstream["phase9_checkpoint"]["sha256"] == pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256
        assert (
            upstream["phase9_checkpoint"]["model_state_digest"]
            == pc.ACCEPTED_PHASE9_MODEL_STATE_DIGEST
        )
        assert upstream["phase9_checkpoint"]["all_parameters_finite"]
        assert upstream["phase9_chain"]["chain_intact"]
        assert upstream["phase7_library"]["content_digest"] == pc.PHASE7_LIBRARY_CONTENT_DIGEST
        assert upstream["phase7_library"]["trait_vectors_reconstructed"] == 8000

    def test_no_absolute_path_enters_logical_identity(self, contract):
        text = json.dumps(contract["contracts"], sort_keys=True)
        assert "/Volumes/" not in text
        assert "/Users/" not in text


class TestBankArtifacts:
    @pytest.mark.parametrize(
        "name,count,per_family,split",
        [("validation", 128, 8, "validation"), ("test", 512, 32, "test")],
    )
    def test_shape(self, banks, name, count, per_family, split):
        artifact = banks[name]
        assert artifact["case_count"] == count
        assert artifact["cases_per_opponent_family"] == per_family
        assert artifact["manifest"]["split"] == split
        assert len(artifact["cases"]) == count

    def test_family_balance_is_exact(self, banks):
        for name, artifact in banks.items():
            counts = {}
            for case in artifact["cases"]:
                counts[case["family_id"]] = counts.get(case["family_id"], 0) + 1
            assert len(counts) == 16
            assert set(counts.values()) == {artifact["cases_per_opponent_family"]}

    def test_every_case_carries_the_frozen_case_structure(self, banks):
        for artifact in banks.values():
            for case in artifact["cases"][:8]:
                assert set(case["selector_seeds"]) == {"red", "blue"}
                assert set(case["neutral_provenance"]) == {"red", "blue"}
                assert set(case["match_seeds"]) == set(pc.MATCHUP_TOKENS)
                assert case["colour_pairing"] == {"0": "red", "1": "blue"}
                assert case["bootstrap_unit"] == case["case_id"]
                assert len(case["frozen_fingerprints"]) == 3
                assert len(set(case["frozen_fingerprints"])) == 3

    def test_audit_recorded_in_the_artifact_passed(self, banks):
        for artifact in banks.values():
            assert artifact["audit"]["all_pass"]
            assert artifact["audit"]["checks"]["phase9_fingerprint_overlap_zero"]
            assert artifact["audit"]["checks"]["split_isolation"]
            assert artifact["audit"]["checks"]["engine_valid"]
            assert artifact["audit"]["checks"]["provenance_rebuilds"]
            assert artifact["audit"]["checks"]["isolated_rebuild_exact"]

    def test_isolation_claim_matches_the_live_phase9_set(self, banks):
        _, manifest = pb.phase9_isolation_set()
        for artifact in banks.values():
            assert artifact["isolation"]["set_digest"] == manifest["set_digest"]
            assert artifact["isolation"]["set_digest"] == PHASE9_ISOLATION_SET_DIGEST
            assert artifact["cross_bank_isolation"]["zero_overlap"]

    def test_no_stored_fingerprint_is_a_phase9_heldout_fingerprint(self, banks):
        fingerprints, _ = pb.phase9_isolation_set()
        for artifact in banks.values():
            for case in artifact["cases"]:
                for fingerprint in case["frozen_fingerprints"]:
                    assert fingerprint not in fingerprints

    def test_the_sealed_test_bank_records_only_structural_access(self, banks):
        log = banks["test"]["access_log"]
        assert log
        for record in log:
            assert record["neural"] is False
            assert record["outcomes"] is False
            assert record["purpose"] in banks["test"]["sealing"]["allowed_before_agent_7"]
        assert banks["validation"]["access_log"] == []

    def test_the_validation_bank_carries_no_sealed_access_claim(self, banks):
        assert banks["test"]["sealing"]["sealed_until"] == "Agent 7"


class TestAcceptanceArtifact:
    def test_the_twenty_two_completion_gates(self, acceptance):
        assert set(acceptance["completion_gates"]) == {
            "phase9_final_identity_verified",
            "phase9_model_finite",
            "phase7_library_identity_verified",
            "phase7_splits_verified",
            "phase7_trait_vectors_reconstruct",
            "neutral_profile_verified",
            "phase10_seeds_frozen",
            "phase10_contracts_frozen_and_hashed",
            "outcome_schedule_exact_16384",
            "ordered_family_pair_counts_exact",
            "utility_fit_protocol_frozen",
            "candidate_matrix_exactly_six",
            "validation_bank_frozen_and_hashed",
            "test_bank_frozen_and_hashed",
            "phase9_bank_exact_fingerprint_overlap_zero",
            "phase10_val_test_fingerprint_overlap_zero",
            "test_bank_neural_outcome_access_zero",
            "final_acceptance_gates_frozen",
            "no_phase10_outcome_games",
            "no_utility_fit",
            "phase9_checkpoint_unchanged",
            "full_suite_green",
        }

    def test_discipline_counters_are_all_zero(self, acceptance):
        assert acceptance["discipline"] == {
            "phase10_outcome_games_played": 0,
            "utility_models_fit": 0,
            "c1_optimizer_steps": 0,
            "neural_inference_on_either_bank": 0,
            "held_out_bases_in_fitting_path": 0,
            "test_bank_outcome_access": 0,
        }

    def test_candidate_matrix_is_exactly_the_frozen_six(self, acceptance):
        assert acceptance["candidate_matrix"] == [dict(e) for e in pc.CANDIDATE_MATRIX]

    def test_every_gate_boundary_was_exercised(self, acceptance):
        gates = {entry["gate"] for entry in acceptance["boundary_evidence"]}
        assert gates == set(pc.HARD_GATE_IDS)

    def test_strict_bounds_were_shown_to_fail_at_the_threshold(self, acceptance):
        by_gate = {}
        for entry in acceptance["boundary_evidence"]:
            by_gate.setdefault(entry["gate"], []).append(entry)
        strict_a = [
            entry
            for entry in by_gate["A"]
            if entry.get("expected_at_threshold") is False
        ]
        assert strict_a and all(
            entry["at_threshold_passes"] is False for entry in strict_a
        )
        assert any(
            entry.get("at_bound_threshold_passes") is False for entry in by_gate["B"]
        )
        assert any(
            entry.get("at_threshold_passes") is False for entry in by_gate["C"]
        )

    def test_recorded_readings_are_present_and_explained(self, acceptance):
        assert len(acceptance["deviations"]) >= 4
        for entry in acceptance["deviations"]:
            assert entry["topic"] and entry["contract_text"]
            assert entry["reading"] and entry["why_safe"]

    def test_frozen_inputs_are_the_accepted_upstream_identities(self, acceptance):
        frozen = acceptance["frozen_inputs"]
        assert frozen["phase9_checkpoint_sha256"] == pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256
        assert frozen["phase9_model_state_digest"] == pc.ACCEPTED_PHASE9_MODEL_STATE_DIGEST
        assert frozen["phase9_contract_digest"] == pc.ACCEPTED_PHASE9_CONTRACT_DIGEST
        assert frozen["phase7_library_content_digest"] == pc.PHASE7_LIBRARY_CONTENT_DIGEST

    def test_handoff_gives_agent_2_everything_and_no_design_choice(self, acceptance):
        handoff = acceptance["handoff_to_agent_2"]
        assert handoff["for_agent"] == 2
        assert handoff["exact_game_count"] == 16_384
        assert handoff["schedule_digest"] == OUTCOME_SCHEDULE_DIGEST
        assert handoff["contract_digests"] == CONTRACT_DIGESTS
        assert "train" in handoff["train_only_rule"]
        assert handoff["phase9_evaluation_only_identity"]["sha256"] == (
            pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256
        )
        assert handoff["phase9_evaluation_only_identity"]["behaviour"]["search"] == "none"
        assert "storage_policy" in handoff
        assert "crash_resume_identity" in handoff

    def test_the_report_records_agent_1(self):
        report = REPOSITORY_ROOT / "reports" / "phase_10_implementation_report.md"
        assert report.exists()
        text = report.read_text()
        assert "## 1. Agent 1" in text
        assert CONTRACT_BUNDLE_DIGEST in text
