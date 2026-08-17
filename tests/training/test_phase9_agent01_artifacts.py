"""Regression: Agent 1's accepted freeze artifacts stay self-consistent.

Agent 1 freezes the Phase 9 RL contract and the two evaluation banks before
any Phase 9 training exists. These tests pin what that freeze *means*:

- the four artifacts exist together, name the same contract digest, and
  carry a status justified by their own recorded gates;
- every frozen threshold in the artifacts equals the live
  `phase9_contract` value — a silently relaxed gate cannot survive the
  suite;
- the recorded bank digests equal the frozen bank identity and the banks
  stored inside the artifacts re-hash to exactly those digests;
- the anchor baseline was measured on the validation bank only, with the
  frozen seeds and zero safety violations;
- the recorded Phase 8 identities match the accepted upstream records.

Everything is gated on the artifacts existing, so the suite is green both
before and after Agent 1 runs.
"""

import json
from pathlib import Path

import pytest

from stratego.training import phase9_contract as pc
from stratego.training import phase9_seed as ps

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_9_data"

ARTIFACTS = {
    "contract": DATA_DIRECTORY / "agent_01_rl_contract.json",
    "acceptance": DATA_DIRECTORY / "agent_01_acceptance.json",
    "validation_bank": DATA_DIRECTORY / "agent_01_validation_bank.json",
    "test_bank": DATA_DIRECTORY / "agent_01_test_bank.json",
}

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in ARTIFACTS.values()),
    reason="Agent 1's freeze artifacts have not been produced yet",
)


@pytest.fixture(scope="module")
def artifacts() -> dict:
    return {name: json.loads(path.read_text()) for name, path in ARTIFACTS.items()}


class TestArtifactCoherence:
    def test_every_artifact_names_phase_9_agent_1(self, artifacts):
        for name, payload in artifacts.items():
            assert payload["phase"] == 9, name
            assert payload["agent"] == 1, name

    def test_status_is_justified_by_the_recorded_gates(self, artifacts):
        acceptance = artifacts["acceptance"]
        gates = acceptance["completion_gates"]
        assert acceptance["gates_total"] == len(gates) == 18
        assert acceptance["gates_true"] == sum(bool(value) for value in gates.values())
        if acceptance["status"] == "PASS":
            assert all(gates.values())
            assert not acceptance["problems"]

    def test_one_contract_digest_across_artifacts_and_live_source(self, artifacts):
        recorded = artifacts["acceptance"]["contract_digest"]
        assert artifacts["contract"]["contract_digest"] == recorded
        assert pc.contract_digest() == recorded

    def test_contract_document_matches_live_source(self, artifacts):
        assert artifacts["contract"]["contract"] == json.loads(
            json.dumps(pc.rl_contract_document())
        )

    def test_canonical_seeds_are_the_frozen_eight(self, artifacts):
        assert artifacts["contract"]["canonical_seeds"] == {
            name: int(value) for name, value in ps.CANONICAL_PHASE9_SEEDS.items()
        }


class TestFrozenIdentities:
    def test_phase8_checkpoint_identities(self, artifacts):
        loads = artifacts["acceptance"]["checkpoint_verification"]
        assert loads["accepted"]["sha256"] == pc.EXPECTED_PHASE8_CHECKPOINT_SHA256
        assert loads["accepted"]["parameters"] == 863959
        assert loads["accepted"]["global_step"] == 24000
        assert loads["canonical_untrained"]["sha256"] == pc.EXPECTED_PHASE8_INIT_SHA256
        assert (
            loads["canonical_untrained"]["model_state_checksum"]
            == pc.EXPECTED_PHASE8_INIT_STATE_CHECKSUM
        )

    def test_corpus_identity_matches_the_accepted_records(self, artifacts):
        corpus = artifacts["acceptance"]["corpus_verification"]
        assert corpus["observed_identity"] == corpus["accepted_identity"]
        assert corpus["accepted_identity"]["content_digest"] == (
            pc.EXPECTED_CORPUS_CONTENT_DIGEST
        )
        assert corpus["resolved_root_matches_accepted_location"] is True

    def test_final_gates_match_live_source(self, artifacts):
        assert artifacts["acceptance"]["final_gates"] == json.loads(
            json.dumps(pc.final_gates())
        )

    def test_pilot_matrix_matches_live_source(self, artifacts):
        assert artifacts["acceptance"]["pilot_matrix"] == json.loads(
            json.dumps(pc.pilot_matrix())
        )


class TestBankArtifacts:
    def test_recorded_digests_agree_everywhere(self, artifacts):
        acceptance = artifacts["acceptance"]["bank_digests"]
        assert (
            artifacts["validation_bank"]["bank_digest"]
            == acceptance["phase9_validation_bank_v1"]
        )
        assert artifacts["test_bank"]["bank_digest"] == acceptance["phase9_test_bank_v1"]

    def test_stored_banks_rehash_to_their_recorded_digests(self, artifacts):
        from stratego.evaluation.setup_bank import SetupBank, bank_digest

        for name in ("validation_bank", "test_bank"):
            payload = artifacts[name]
            bank = SetupBank.from_dict(payload["bank"])
            assert bank_digest(bank) == payload["bank_digest"], name

    def test_bank_shapes(self, artifacts):
        assert artifacts["validation_bank"]["case_count"] == 128
        assert artifacts["validation_bank"]["cases_per_family"] == 8
        assert artifacts["validation_bank"]["bank_version"] == "phase9_validation_bank_v1"
        assert artifacts["test_bank"]["case_count"] == 512
        assert artifacts["test_bank"]["cases_per_family"] == 32
        assert artifacts["test_bank"]["bank_version"] == "phase9_test_bank_v1"

    def test_recorded_audits_pass(self, artifacts):
        for name in ("validation_bank", "test_bank"):
            audit = artifacts[name]["audit"]
            assert audit["all_pass"], name
            assert all(audit["checks"].values()), name


class TestAnchorBaseline:
    def test_measured_on_the_validation_bank_only(self, artifacts):
        baseline = artifacts["acceptance"]["anchor_validation_baseline"]
        assert baseline["bank"] == "phase9_validation_bank_v1"
        assert baseline["paired_cases_per_opponent"] == 128
        assert baseline["games_per_opponent"] == 256

    def test_all_four_core_rule_opponents_are_recorded(self, artifacts):
        ewrs = artifacts["acceptance"]["anchor_validation_baseline"]["effective_win_rates"]
        assert set(ewrs) == {
            "random_legal",
            "basic_heuristic",
            "tactical_rule_based",
            "strategic_rule_based",
        }
        for opponent, value in ewrs.items():
            assert 0.0 <= value <= 1.0, opponent

    def test_zero_safety_violations(self, artifacts):
        safety = artifacts["acceptance"]["anchor_validation_baseline"]["safety"]
        assert safety["illegal_actions"] == 0
        assert safety["policy_errors"] == 0
        assert safety["inference_failures"] == 0
        assert safety["workers_importing_torch"] == 0
        assert safety["worker_checkpoint_loads"] == 0

    def test_no_neural_test_bank_access(self, artifacts):
        gates = artifacts["acceptance"]["completion_gates"]
        assert gates["test_bank_neural_access_zero"] is True
        probe = artifacts["acceptance"]["sealing_probe"]
        assert probe["test_bank_neural_purposes_refused"] == len(
            pc.TEST_BANK_PROHIBITED_BEFORE_8
        ) + len(pc.TEST_BANK_AGENT8_ONLY)


class TestNoTrainingHappened:
    def test_no_optimizer_steps_and_no_rollouts(self, artifacts):
        gates = artifacts["acceptance"]["completion_gates"]
        assert gates["no_phase9_optimizer_steps"] is True
        assert gates["no_trainable_phase9_rollouts"] is True
