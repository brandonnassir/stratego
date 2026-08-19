"""Phase 11 Agent 1 artifacts: the frozen record checks itself.

Every check here recomputes something from the tracked artifacts and the
live modules rather than trusting a stored summary. `full_suite_green` is a
claim about the suite that contains this test, so it is checked against the
recorded measurement rather than asserted (the accepted Phase 10 pattern).
"""

import json
from pathlib import Path

import pytest

from stratego.evaluation import phase11_banks as pb
from stratego.training import phase11_contract as pc
from stratego.training.phase11_seed import (
    CANONICAL_PHASE11_SEEDS,
    OPPONENT_STRATA,
    SETUP_SOURCES,
    parse_phase11_case_id,
)

from .phase11_frozen_digests import (
    BANK_DIGESTS,
    BANK_MANIFEST_DIGESTS,
    BELIEF_HEAD_DIGEST,
    CONTRACT_BUNDLE_DIGEST,
    CONTRACT_DIGESTS,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_11_data"
CONTRACT_PATH = DATA_DIRECTORY / "agent_01_phase11_contract.json"
VALIDATION_BANK_PATH = DATA_DIRECTORY / "agent_01_validation_bank.json"
TEST_BANK_PATH = DATA_DIRECTORY / "agent_01_test_bank.json"
ACCEPTANCE_PATH = DATA_DIRECTORY / "agent_01_acceptance.json"

#: The 28 completion gates of the Agent 1 instruction, plus the three added
#: gates (anchor, observation, ledger). Gates may be added, never weakened.
EXPECTED_GATES = (
    "upstream_phase10_closed",
    "phase9_identity_verified",
    "belief_head_identity_frozen",
    "phase10_selector_identity_verified",
    "phase7_identity_verified",
    "phase8_anchor_identity_verified",
    "observation_contract_verified",
    "eight_contracts_frozen",
    "contract_bundle_frozen",
    "eight_root_seeds_frozen",
    "randomness_domains_frozen",
    "seed_collision_audit_clean",
    "validation_bank_exact",
    "test_bank_exact",
    "validation_balance_exact",
    "test_balance_exact",
    "isolated_case_rebuild_pass",
    "bank_overlap_zero",
    "prediction_target_contract_frozen",
    "baselines_frozen",
    "sampler_math_frozen",
    "metrics_frozen",
    "bootstrap_frozen",
    "acceptance_gates_frozen",
    "classification_frozen",
    "ledger_initialized",
    "test_outcome_access_zero",
    "no_phase11_predictions_scored",
    "no_neural_updates",
    "phase9_checkpoint_unchanged",
    "full_suite_green",
)

SELF_REFERENTIAL_GATE = "full_suite_green"

pytestmark = pytest.mark.skipif(
    not ACCEPTANCE_PATH.exists(), reason="Agent 1 has not produced artifacts yet"
)


@pytest.fixture(scope="module")
def acceptance():
    return json.loads(ACCEPTANCE_PATH.read_text())


@pytest.fixture(scope="module")
def contract_artifact():
    return json.loads(CONTRACT_PATH.read_text())


@pytest.fixture(scope="module")
def bank_artifacts():
    return {
        "validation": json.loads(VALIDATION_BANK_PATH.read_text()),
        "test": json.loads(TEST_BANK_PATH.read_text()),
    }


def test_the_gate_set_is_exactly_the_instruction_plus_additions(acceptance):
    assert tuple(sorted(acceptance["completion_gates"])) == tuple(sorted(EXPECTED_GATES))


def test_status_follows_from_the_gates(acceptance):
    expected = "PASS" if all(acceptance["completion_gates"].values()) else "PENDING-SUITE"
    assert acceptance["status"] == expected
    non_suite = {
        name: value
        for name, value in acceptance["completion_gates"].items()
        if name != SELF_REFERENTIAL_GATE
    }
    assert all(non_suite.values())
    assert acceptance["problems"] == []


def test_the_suite_gate_agrees_with_the_recorded_measurement(acceptance):
    suite = acceptance["suite"]
    expected = (
        suite is not None
        and suite["returncode"] == 0
        and suite["failed"] == 0
        and suite["passed"] > 0
    )
    assert acceptance["completion_gates"][SELF_REFERENTIAL_GATE] == expected


def test_contract_digests_recompute_from_the_live_modules(acceptance, contract_artifact):
    live = pc.contract_digests()
    assert live == CONTRACT_DIGESTS
    assert live == contract_artifact["contract_digests"]
    assert live == acceptance["new_digests"]["contract_digests"]
    bundle = pc.contract_bundle_digest()
    assert bundle == CONTRACT_BUNDLE_DIGEST
    assert bundle == contract_artifact["contract_bundle_digest"]
    assert bundle == acceptance["new_digests"]["contract_bundle_digest"]


def test_contract_artifact_stores_the_documents_it_hashes(contract_artifact):
    documents = contract_artifact["documents"]
    assert tuple(sorted(documents)) == tuple(sorted(pc.CONTRACT_VERSIONS))
    for name, document in documents.items():
        assert pc.document_digest(document) == contract_artifact["contract_digests"][name]
    assert documents == pc.contract_documents()


def test_the_frozen_inputs_match_the_contract_constants(acceptance):
    frozen = acceptance["frozen_inputs"]
    assert frozen["phase9_checkpoint_sha256"] == pc.GATE_H["phase9_checkpoint_sha256"]
    assert frozen["phase9_model_state_digest"] == pc.GATE_H["phase9_model_state_digest"]
    assert frozen["phase9_parameters"] == 863_959
    assert frozen["phase9_global_optimizer_step"] == pc.ACCEPTED_GLOBAL_OPTIMIZER_STEP
    assert frozen["belief_head_digest"] == pc.ACCEPTED_BELIEF_HEAD_DIGEST
    assert frozen["belief_head_digest"] == BELIEF_HEAD_DIGEST
    assert tuple(frozen["belief_head_tensor_names"]) == pc.BELIEF_HEAD_TENSOR_NAMES
    assert frozen["selector_config_sha256"] == pc.ACCEPTED_SELECTOR_CONFIG_SHA256
    assert frozen["model_T_coefficient_digest"] == pc.ACCEPTED_UTILITY_COEFFICIENT_DIGEST
    assert frozen["trait_scaler_digest"] == pc.ACCEPTED_TRAIT_SCALER_DIGEST
    assert frozen["phase10_system_digest"] == pc.ACCEPTED_PHASE10_SYSTEM_DIGEST
    assert frozen["phase8_anchor_sha256"] == pc.ACCEPTED_ANCHOR_EXPORT_SHA256
    assert frozen["phase10_closure_commit"] == pc.PHASE10_CLOSURE_COMMIT


def test_the_root_seeds_are_recorded_exactly(acceptance):
    assert acceptance["seeds"] == dict(CANONICAL_PHASE11_SEEDS)
    audit = acceptance["seed_collision_audit"]
    assert audit["no_collisions"] and not audit["findings"]
    assert audit["total_seeds"] == audit["distinct_seeds"]


def test_bank_digests_are_pinned_and_manifests_rehash(bank_artifacts, acceptance):
    for bank, artifact in bank_artifacts.items():
        manifest = artifact["manifest"]
        assert manifest["bank_digest"] == BANK_DIGESTS[bank]
        assert manifest["manifest_digest"] == BANK_MANIFEST_DIGESTS[bank]
        assert pb.manifest_digest(manifest) == manifest["manifest_digest"]
        assert artifact["audit_all_pass"] is True
        assert all(artifact["audit_checks"].values())
    fresh = acceptance["new_digests"]
    assert fresh["validation_bank_digest"] == BANK_DIGESTS["validation"]
    assert fresh["test_bank_digest"] == BANK_DIGESTS["test"]


def test_stored_cases_rehash_to_the_bank_digest(bank_artifacts):
    """The artifact's case list is the digested content, byte for byte."""
    import hashlib

    from stratego.training.phase11_seed import PHASE11_MASTER_SEED

    for bank, artifact in bank_artifacts.items():
        payload = {
            "domain": pb.BANK_DIGEST_DOMAIN,
            "master_seed": PHASE11_MASTER_SEED,
            "cases": artifact["cases"],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        observed = hashlib.sha256(canonical.encode()).hexdigest()
        assert observed == BANK_DIGESTS[bank], bank


def test_bank_balance_and_ids_from_stored_cases(bank_artifacts):
    for bank, artifact in bank_artifacts.items():
        cases = artifact["cases"]
        expected_cell = pb.BANK_SPECIFICATIONS[bank]["cases_per_cell"]
        assert len(cases) == pb.BANK_SPECIFICATIONS[bank]["case_count"]
        cells: dict = {}
        for case in cases:
            fields = parse_phase11_case_id(case["case_id"])
            assert fields["bank_version"] == artifact["manifest"]["bank_version"]
            key = (case["stratum"], case["setup_source"])
            cells[key] = cells.get(key, 0) + 1
            assert case["games"]["0"]["observer_color"] == "red"
            assert case["games"]["1"]["observer_color"] == "blue"
        assert len(cells) == len(OPPONENT_STRATA) * len(SETUP_SOURCES)
        assert all(count == expected_cell for count in cells.values())


def test_a_sample_of_stored_cases_rebuilds_in_isolation(bank_artifacts):
    sources = pb.Phase11SetupSources()
    for bank, artifact in bank_artifacts.items():
        cases = artifact["cases"]
        for stored in cases[:: max(1, len(cases) // 4)]:
            rebuilt = pb.build_case(bank, stored["case_index"], sources)
            assert rebuilt.to_dict() == stored


def test_the_ledger_proves_structural_only_access(acceptance):
    entries = pb.read_ledger()
    sealed = pb.verify_test_bank_sealed(entries)
    assert sealed["test_bank_structural_only"]
    assert sealed["scored_prediction_total"] == 0
    assert sealed["privileged_truth_total"] == 0
    assert sealed["outcome_total"] == 0
    assert sealed["neural_inference_total"] == 0
    assert acceptance["ledger"]["test_bank_structural_only"]
    for entry in entries:
        assert entry["agent"] == 1
        assert entry["structural_only"] is True


def test_forbidden_operation_counters_are_zero(acceptance):
    counters = acceptance["forbidden_operation_counters"]
    assert set(counters.values()) == {0}
    assert "phase11_optimizer_steps" in counters
    assert "test_bank_scored_accesses" in counters


def test_the_handoff_names_the_frozen_identities(acceptance):
    handoff = acceptance["handoff_to_agent_2"]
    assert handoff["for_agent"] == 2
    assert handoff["contract_bundle_digest"] == CONTRACT_BUNDLE_DIGEST
    assert handoff["bank_identities"]["validation"]["bank_digest"] == BANK_DIGESTS["validation"]
    assert handoff["bank_identities"]["test"]["bank_digest"] == BANK_DIGESTS["test"]
    assert handoff["belief_head"]["digest"] == BELIEF_HEAD_DIGEST
    assert tuple(handoff["rank_indexing"]) == pc.RANK_NAMES
    assert handoff["evaluator_version"] == pc.EVALUATOR_VERSION
    assert "must not implement or tune belief_sampler_v1" in handoff["prohibition"]


def test_the_recorded_readings_include_the_load_bearing_ones(acceptance):
    readings = {entry["reading"] for entry in acceptance["recorded_deviations"]}
    assert {
        "bank_split_binding",
        "no_rejection_draws",
        "sampler_completion_feasibility_rule",
        "soak_namespace_frozen_now",
        "progress_bucket_thresholds",
        "phase10b_draft_drift_restored",
    } <= readings


def test_the_belief_head_digest_re_derives_from_the_live_checkpoint():
    """The one heavy check: live checkpoint bytes -> frozen head identity."""
    import hashlib

    import torch

    from stratego.training import phase9_checkpoint

    payload = phase9_checkpoint.read_phase9_payload(
        REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"
    )
    model = phase9_checkpoint.model_from_payload(payload)
    state_dict = model.state_dict()
    names = tuple(sorted(name for name in state_dict if name.startswith("belief_output.")))
    assert names == pc.BELIEF_HEAD_TENSOR_NAMES
    hasher = hashlib.sha256()
    for name in names:
        tensor = state_dict[name]
        assert tuple(tensor.shape) == pc.BELIEF_HEAD_TENSOR_SHAPES[name]
        hasher.update(name.encode())
        array = tensor.detach().to("cpu", torch.float32).contiguous().numpy()
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    assert hasher.hexdigest() == BELIEF_HEAD_DIGEST
    assert payload.get("global_optimizer_step") == pc.ACCEPTED_GLOBAL_OPTIMIZER_STEP
