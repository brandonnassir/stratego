"""Phase 10 Agent 6's three artifacts, checked against the frozen contract.

The soak itself is expensive and ran once; what these tests protect is the
record of it. Every claim below is recomputed from the artifacts' own
primitives — the production vector digests from the stored exact hex values,
the mixture from its two components, the system identity from its own
document — so an edited number fails here rather than being inherited by
Agent 7.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from stratego.setups.families import FAMILY_IDS
from stratego.training import phase10_soak as soak
from stratego.training.phase10_contract import (
    LEARNED_MIXTURE_WEIGHT,
    NEUTRAL_MIXTURE_WEIGHT,
    document_digest,
)
from stratego.training.phase10_seed import CANONICAL_PHASE10_SEEDS
from stratego.training.phase10_selector import DISTRIBUTION_DIGEST_DOMAIN

from .phase10_frozen_digests import BANK_DIGESTS, CONTRACT_BUNDLE_DIGEST, CONTRACT_DIGESTS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_10_data"
ACCEPTANCE_PATH = DATA_DIRECTORY / "agent_06_acceptance.json"
MANIFEST_PATH = DATA_DIRECTORY / "agent_06_production_selector_manifest.json"
SOAK_PATH = DATA_DIRECTORY / "agent_06_integration_soak.json"
CONFIG_PATH = DATA_DIRECTORY / "agent_05_frozen_selector_config.json"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_10_implementation_report.md"

ACCEPTED_PHASE9_CHECKPOINT_SHA256 = (
    "dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea"
)
ACCEPTED_PHASE9_MODEL_STATE_DIGEST = (
    "f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd"
)

#: The 21 completion gates of the Agent 6 instruction, in its order.
EXPECTED_GATES = (
    "agents1_5_pass",
    "selector_config_digest_verified",
    "production_red_distribution_frozen",
    "production_blue_distribution_frozen",
    "production_distribution_rebuild_exact",
    "phase10_system_v1_frozen",
    "soak_games_ge_8192",
    "all_16_families_seen_in_soak",
    "setup_legality_errors_zero",
    "stranded_sampled_setups_zero",
    "inventory_errors_zero",
    "provenance_mismatches_zero",
    "hidden_opponent_selector_inputs_zero",
    "restart_resume_pass",
    "duplicate_game_ids_zero",
    "missing_game_ids_zero",
    "phase9_checkpoint_unchanged",
    "no_c1_optimizer_steps",
    "no_reselection",
    "no_test_outcome_access",
    "full_suite_green",
)

#: `full_suite_green` is a claim about the suite that contains this test, so
#: it cannot be asserted directly; it is checked against the recorded suite
#: measurement embedded in the acceptance artifact instead.
SELF_REFERENTIAL_GATE = "full_suite_green"

pytestmark = pytest.mark.skipif(
    not ACCEPTANCE_PATH.exists(), reason="Agent 6 has not produced artifacts yet"
)


@pytest.fixture(scope="module")
def acceptance() -> dict:
    return json.loads(ACCEPTANCE_PATH.read_text())


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def soak_artifact() -> dict:
    return json.loads(SOAK_PATH.read_text())


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


# ---------------------------------------------------------------------------
# Acceptance: status, gates, and the frozen identity chain
# ---------------------------------------------------------------------------


def test_status_follows_from_the_gates(acceptance):
    gates = acceptance["completion_gates"]
    # The artifact serializes with sorted keys, so membership is exact and
    # order is the serializer's.
    assert set(gates) == set(EXPECTED_GATES) and len(gates) == len(EXPECTED_GATES)
    assert acceptance["gates_total"] == len(EXPECTED_GATES)
    assert acceptance["gates_true"] == sum(bool(value) for value in gates.values())
    assert acceptance["false_gates"] == sorted(
        name for name, value in gates.items() if not value
    )
    expected_status = "PASS" if not acceptance["false_gates"] else "FAIL"
    assert acceptance["status"] == expected_status
    for name, value in gates.items():
        if name != SELF_REFERENTIAL_GATE:
            assert value is True, f"gate {name} is not true"


def test_the_suite_gate_agrees_with_the_recorded_measurement(acceptance):
    suite = acceptance["suite"]
    assert acceptance["completion_gates"][SELF_REFERENTIAL_GATE] == (
        suite["returncode"] == 0 and suite["failed"] == 0
    )
    assert suite["command"] == ".venv/bin/python -m pytest tests -q"
    assert suite["passed"] > acceptance["suite_before"]["passed"]


def test_frozen_inputs_match_the_accepted_identities(acceptance):
    frozen = acceptance["frozen_inputs"]
    assert frozen["phase9_checkpoint_sha256"] == ACCEPTED_PHASE9_CHECKPOINT_SHA256
    assert frozen["phase9_model_state_digest"] == ACCEPTED_PHASE9_MODEL_STATE_DIGEST
    assert frozen["phase9_parameters"] == 863_959
    assert frozen["contract_bundle_digest"] == CONTRACT_BUNDLE_DIGEST
    assert frozen["validation_bank_digest"] == BANK_DIGESTS["validation"]
    assert frozen["test_bank_digest"] == BANK_DIGESTS["test"]
    assert frozen["selector_config_sha256"] == soak.SELECTED_CONFIG_SHA256


def test_selection_stayed_closed(acceptance, config):
    closed = acceptance["selection_closed"]
    assert closed["reopened"] is False
    assert closed["winner"]["candidate_id"] == "P10-D"
    assert closed["winner"] == {
        "candidate_id": config["winner"]["candidate_id"],
        "utility_model": config["winner"]["utility_model"],
        "temperature": config["winner"]["temperature"],
        "selector_identity": config["winner"]["selector_identity"],
    }


def test_phase9_preservation_holds_before_and_after(acceptance):
    preservation = acceptance["phase9_preservation"]
    for stamp in ("before", "after"):
        assert preservation[stamp]["sha256"] == ACCEPTED_PHASE9_CHECKPOINT_SHA256
        assert preservation[stamp]["model_state_digest"] == ACCEPTED_PHASE9_MODEL_STATE_DIGEST
    assert preservation["unchanged"] is True
    assert preservation["c1_optimizer_steps"] == 0
    assert acceptance["upstream_byte_preservation"]["all_unchanged"] is True


def test_test_bank_was_never_opened_for_outcomes(acceptance):
    unopened = acceptance["test_bank_unopened"]
    assert unopened["games"] == 0
    assert unopened["neural_inference"] == 0
    assert unopened["outcomes_read"] == 0
    for entry in acceptance["bank_access_log"]:
        if entry["bank"] == "phase10_test_bank_v1":
            assert not entry["neural"] and not entry["outcomes"]
    handoff = acceptance["handoff_to_agent_7"]
    assert handoff["final_test_outcome_access"]["games"] == 0
    assert handoff["final_test_outcome_access"]["outcomes_read"] == 0
    assert handoff["for_agent"] == 7


# ---------------------------------------------------------------------------
# Production manifest: the vectors prove their own digests
# ---------------------------------------------------------------------------


def _vector_digest(block: dict, label: str, hex_values: list) -> str:
    payload = "\n".join(
        [
            DISTRIBUTION_DIGEST_DOMAIN,
            label,
            block["candidate_id"],
            block["utility_model"],
            f"T={block['temperature']:.2f}",
            block["color"],
            block["split"],
            f"n={block['base_count']}",
            *[
                f"{base_id}:{value}"
                for base_id, value in zip(block["base_ids"], hex_values)
            ],
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@pytest.mark.parametrize("color", ("red", "blue"))
def test_production_vector_digests_recompute_from_stored_values(manifest, config, color):
    block = manifest["vectors"][color]
    assert block["split"] == "train"
    assert block["base_count"] == len(block["base_ids"]) == 6_400
    assert block["bases_per_family"] == 400
    assert block["family_order"] == list(FAMILY_IDS)
    observed = _vector_digest(block, "p_phase10", block["p_phase10_hex"])
    assert observed == block["probability_vector_digest"]
    assert observed == block["digests"]["p_phase10"]
    assert observed == manifest["frozen_train_digests"][color]
    assert observed == config["train_split_production_digests"][color]
    assert manifest["frozen_train_digests_match"][color] is True
    learned = _vector_digest(block, "p_learned", block["p_learned_hex"])
    assert learned == block["digests"]["p_learned"]


@pytest.mark.parametrize("color", ("red", "blue"))
def test_production_mixture_recomputes_exactly_from_hex(manifest, color):
    block = manifest["vectors"][color]
    neutral = float.fromhex(block["p_neutral_uniform_hex"])
    assert neutral == 1.0 / 6_400
    learned = [float.fromhex(value) for value in block["p_learned_hex"]]
    mixed = [float.fromhex(value) for value in block["p_phase10_hex"]]
    assert all(
        NEUTRAL_MIXTURE_WEIGHT * neutral + LEARNED_MIXTURE_WEIGHT * p_l == p_m
        for p_l, p_m in zip(learned, mixed)
    )
    assert abs(sum(mixed) - 1.0) < 1e-9
    assert block["mixture_exact"] is True
    assert block["diversity_evaluation"]["all_pass"] is True
    # Family aggregation: the stored per-family mass equals the block sums.
    for index, family_id in enumerate(FAMILY_IDS):
        mass = sum(mixed[index * 400 : (index + 1) * 400])
        assert abs(mass - block["family_probabilities"][family_id]) < 1e-12


def test_the_rebuild_was_independent_and_exact(manifest):
    rebuild = manifest["independent_rebuild"]
    assert rebuild["exact"] is True
    assert rebuild["rebuild_pid"] != rebuild["parent_pid"]
    assert (
        rebuild["canonical_serialization_sha256"]
        == manifest["canonical_serialization_sha256"]
    )


# ---------------------------------------------------------------------------
# phase10_system_v1: the frozen system document
# ---------------------------------------------------------------------------


def test_system_document_digest_recomputes(manifest, acceptance):
    system = manifest["phase10_system_v1"]
    digest = document_digest(system)
    assert digest == manifest["phase10_system_v1_digest"]
    assert digest == acceptance["phase10_system_v1"]["digest"]
    assert digest == acceptance["handoff_to_agent_7"]["phase10_system_v1_digest"]


def test_system_document_binds_every_required_identity(manifest, config):
    system = manifest["phase10_system_v1"]
    assert system["system_version"] == "phase10_system_v1"
    assert system["binding_schema_pinned_digest"] == CONTRACT_DIGESTS["phase10_system_v1"]
    assert system["binding_schema_digest"] == CONTRACT_DIGESTS["phase10_system_v1"]
    assert system["move_model"]["sha256"] == ACCEPTED_PHASE9_CHECKPOINT_SHA256
    assert system["move_model"]["model_state_digest"] == ACCEPTED_PHASE9_MODEL_STATE_DIGEST
    assert system["accepted_utility_model"]["model_id"] == "model_T"
    assert (
        system["accepted_utility_model"]["coefficient_digest"]
        == config["utility"]["coefficient_digests"]["model_T"]
    )
    assert system["accepted_trait_scaler"]["scaler_digest"] == config["utility"]["scaler_digest"]
    selected = system["selected_selector_config"]
    assert selected["candidate_id"] == "P10-D"
    assert selected["temperature"] == 0.75
    assert selected["mixture"] == {"neutral_weight": 0.35, "learned_weight": 0.65}
    assert selected["config_artifact_sha256"] == soak.SELECTED_CONFIG_SHA256
    assert system["production_distributions"]["red_digest"] == (
        config["train_split_production_digests"]["red"]
    )
    assert system["production_distributions"]["blue_digest"] == (
        config["train_split_production_digests"]["blue"]
    )
    assert system["neutral_v1"]["profile"] == "neutral_v1"
    assert system["neutral_v1"]["redefined"] is False
    assert system["phase10_root_seeds"] == dict(CANONICAL_PHASE10_SEEDS)
    assert system["evaluation_banks"]["validation"]["bank_digest"] == BANK_DIGESTS["validation"]
    assert system["evaluation_banks"]["test"]["bank_digest"] == BANK_DIGESTS["test"]
    assert system["contract_bundle_digest"] == CONTRACT_BUNDLE_DIGEST


def test_system_document_carries_no_filesystem_path(manifest):
    text = json.dumps(manifest["phase10_system_v1"])
    assert "/Volumes/" not in text
    assert "/Users/" not in text
    assert "/private/" not in text


# ---------------------------------------------------------------------------
# The soak artifact: scale, zero counters, restart, determinism
# ---------------------------------------------------------------------------


def test_soak_reached_scale_with_zero_counters(soak_artifact):
    assert soak_artifact["seal"]["committed_games"] >= 8_192
    assert soak_artifact["scheduled_games"] == soak_artifact["seal"]["committed_games"]
    assert soak_artifact["split"] == "train"
    counters = soak_artifact["audit_counters"]
    assert set(counters) == set(soak.SOAK_AUDIT_COUNTERS)
    assert all(value == 0 for value in counters.values()), counters
    assert soak_artifact["audit_findings"] == []
    checks = soak_artifact["identity_checks"]
    assert all(checks.values()), checks
    assert soak_artifact["seal_verification"]["all_pass"] is True


def test_soak_exercised_restart_and_parallelism(soak_artifact):
    legs = soak_artifact["legs"]
    assert len(legs) == 3
    killed = [leg for leg in legs if leg["killed"]]
    assert len(killed) == 1 and killed[0]["returncode"] < 0
    surviving = [leg for leg in legs if leg["collection"]]
    worker_counts = {leg["collection"]["worker_count"] for leg in surviving}
    assert len(worker_counts) >= 2, "the soak must exercise multiple worker topologies"
    restart = soak_artifact["restart_evidence"]
    assert restart["process_restarts"] >= 1
    committed = restart["committed_after_leg"]
    assert committed["A"] < committed["B"] < committed["C"]
    assert committed["C"] == soak_artifact["seal"]["committed_games"]
    replay = soak_artifact["replay_probe"]
    assert replay["all_identical"] is True
    assert replay["mismatches"] == []
    assert replay["replayed_games"] >= 128


def test_soak_diversity_diagnostics_cover_all_families(soak_artifact):
    diagnostics = soak_artifact["diagnostics"]
    for color in ("red", "blue"):
        per_color = diagnostics["per_color"][color]
        assert per_color["families_seen"] == 16
        assert per_color["draws"] == soak_artifact["seal"]["committed_games"]
        assert set(per_color["family_frequencies"]) == set(FAMILY_IDS)
        assert 0.0 < per_color["normalized_family_entropy"] <= 1.0
        assert 0.0 < per_color["reflection_rate"] < 1.0
        assert 0.0 < per_color["perturbation_rate"] < 1.0
        comparison = per_color["empirical_vs_exact"]
        assert comparison["family_total_variation"] >= 0.0
        assert comparison["sampling_noise_expectation"] > 0.0
    landings = diagnostics["phase9_fingerprint_landings"]
    assert landings["sides_checked"] == 2 * soak_artifact["seal"]["committed_games"]
    assert "report-only" in landings["role"]
    assert "report-only" in diagnostics["outcomes"]["role"]


def test_soak_identity_and_storage_are_recorded(soak_artifact, acceptance):
    assert soak_artifact["selector_identity"] == (
        "learned_setup_source_v1|k=P10-D|m=model_T|T=0.75"
    )
    assert soak_artifact["selector_config_sha256"] == soak.SELECTED_CONFIG_SHA256
    assert soak_artifact["storage"]["bytes_per_game"] > 0
    assert soak_artifact["throughput"]["mps_used"] is False
    assert soak_artifact["throughput"]["inference_failures"] == 0
    assert acceptance["soak_summary"]["content_digest"] == (
        soak_artifact["seal"]["content_digest"]
    )
    assert acceptance["soak_summary"]["committed_games"] == (
        soak_artifact["seal"]["committed_games"]
    )


def test_the_report_carries_the_agent_6_section():
    assert "## 6. Agent 6 — Integration Soak and Production Freeze" in REPORT_PATH.read_text(
        encoding="utf-8"
    )
