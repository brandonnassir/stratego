"""Phase 11 Agent 1: the frozen contract, gates and classification."""

import math

import pytest

from stratego.engine.constants import (
    IMMOVABLE_TYPES,
    PIECE_COUNTS,
    PIECE_TYPE_NAMES,
)
from stratego.training import phase11_contract as pc
from stratego.training.phase11_seed import (
    OPPONENT_STRATA,
    seed_derivation_document,
)

from .phase11_frozen_digests import CONTRACT_BUNDLE_DIGEST, CONTRACT_DIGESTS


# ---------------------------------------------------------------------------
# Contract digests
# ---------------------------------------------------------------------------


def test_the_eight_contract_digests_are_pinned():
    assert pc.contract_digests() == CONTRACT_DIGESTS


def test_the_bundle_digest_is_pinned():
    assert pc.contract_bundle_digest() == CONTRACT_BUNDLE_DIGEST


def test_the_contract_versions_are_exactly_the_common_contract_eight():
    assert pc.CONTRACT_VERSIONS == (
        "phase11_belief_contract_v1",
        "phase11_belief_baseline_v1",
        "phase11_belief_bank_v1",
        "phase11_belief_metrics_v1",
        "phase11_belief_sampler_v1",
        "phase11_information_safety_v1",
        "phase11_acceptance_v1",
        "phase11_system_v1",
    )


def test_documents_carry_their_own_version_tokens():
    documents = pc.contract_documents()
    version_keys = {
        "phase11_belief_contract_v1": "contract_version",
        "phase11_belief_baseline_v1": "contract_version",
        "phase11_belief_bank_v1": "contract_version",
        "phase11_belief_metrics_v1": "contract_version",
        "phase11_belief_sampler_v1": "contract_version",
        "phase11_information_safety_v1": "contract_version",
        "phase11_acceptance_v1": "acceptance_version",
        "phase11_system_v1": "system_version",
    }
    for name, key in version_keys.items():
        assert documents[name][key] == name


def test_the_seed_document_in_the_contract_is_the_seed_modules():
    document = pc.belief_contract_document()
    assert document["seeds"] == seed_derivation_document()


# ---------------------------------------------------------------------------
# Rank space — must be exactly the engine's, which is what the head trained on
# ---------------------------------------------------------------------------


def test_rank_order_is_exactly_the_engine_enumeration():
    assert pc.RANK_NAMES == PIECE_TYPE_NAMES
    assert pc.RANK_COUNT == 12


def test_rank_initial_counts_match_the_engine():
    assert pc.RANK_INITIAL_COUNTS == tuple(
        PIECE_COUNTS[index] for index in range(len(PIECE_TYPE_NAMES))
    )
    assert sum(pc.RANK_INITIAL_COUNTS) == 40


def test_immovable_ranks_match_the_engine():
    assert set(pc.IMMOVABLE_RANK_INDICES) == set(IMMOVABLE_TYPES)
    assert pc.RANK_NAMES[10] == "flag" and pc.RANK_NAMES[11] == "bomb"
    assert set(pc.MOVABLE_RANK_INDICES) | set(pc.IMMOVABLE_RANK_INDICES) == set(range(12))


# ---------------------------------------------------------------------------
# Target/event semantics
# ---------------------------------------------------------------------------


def test_progress_bucket_boundaries_are_exact():
    assert pc.progress_bucket(0) == "early"
    assert pc.progress_bucket(39) == "early"
    assert pc.progress_bucket(40) == "middle"
    assert pc.progress_bucket(119) == "middle"
    assert pc.progress_bucket(120) == "late"
    assert pc.progress_bucket(4000) == "late"
    with pytest.raises(pc.Phase11ContractError):
        pc.progress_bucket(-1)


def test_prediction_record_covers_every_common_contract_field():
    fields = set(pc.PREDICTION_RECORD_FIELDS)
    required = {
        "bank_version",
        "case_id",
        "game_id",
        "decision_index",
        "observer_color",
        "opponent_stratum",
        "opponent_setup_source",
        "public_state_identity",
        "piece_slot",
        "piece_square",
        "legal_rank_mask",
        "learned_probabilities",
        "baseline_probabilities",
        "true_rank_index",
        "progress_bucket",
        "piece_moved",
        "model_identity",
        "prediction_identity",
        "observation_sha256",
    }
    assert required <= fields
    assert pc.PRIVILEGED_RECORD_FIELDS == ("true_rank_index",)


def test_the_production_request_schema_carries_no_hidden_truth():
    for field in pc.ALLOWED_BELIEF_REQUEST_FIELDS:
        assert not any(
            token in field.lower() for token in pc.FORBIDDEN_BELIEF_REQUEST_TOKENS
        ), field
    for field in pc.ALLOWED_SAMPLER_REQUEST_FIELDS:
        assert not any(
            token in field.lower() for token in pc.FORBIDDEN_SAMPLER_REQUEST_TOKENS
        ), field
    assert "observation" in pc.ALLOWED_BELIEF_REQUEST_FIELDS
    assert "public_state_document" in pc.ALLOWED_BELIEF_REQUEST_FIELDS


def test_the_ten_prohibitions_are_frozen():
    assert len(pc.NON_GOALS) == 10
    joined = " ".join(pc.NON_GOALS)
    for fragment in ("calibrate", "127-channel", "P10-D", "Phase 12", "hidden opponent"):
        assert fragment in joined


# ---------------------------------------------------------------------------
# Baselines, sampler, metrics
# ---------------------------------------------------------------------------


def test_baseline_document_freezes_both_baselines():
    document = pc.baseline_document()
    assert document["baseline_count"] == 2
    remaining = document["remaining_count_belief_v1"]
    assert remaining["role"] == "primary predictive baseline"
    assert "c[r] * mask[r]" in remaining["per_piece_distribution"]
    world = document["count_uniform_world_sampler_v1"]
    assert "remaining_count" in world["algorithm"]
    assert "no learned factor" in world["algorithm"]


def test_sampler_document_freezes_the_twelve_steps_and_counters():
    document = pc.sampler_document()
    assert len(document["algorithm_steps"]) == 12
    assert document["weighting"] == "weight = learned_probability * remaining_count"
    assert tuple(document["zero_tolerance_counters"]) == pc.SAMPLER_ZERO_TOLERANCE_COUNTERS
    assert "dead_end_events" in document["zero_tolerance_counters"]
    assert document["feasibility_rule"]["status"].startswith("an Agent 1 design reading")
    assert len(document["validation_stack"]) == len(pc.WORLD_VALIDATION_STACK)
    assert document["audit_volumes"]["large_audit_min_worlds"] == 250_000


def test_metric_tokens_are_unique_and_statistics_are_frozen():
    assert len(set(pc.OVERALL_METRIC_TOKENS)) == len(pc.OVERALL_METRIC_TOKENS)
    assert pc.BOOTSTRAP_REPLICATES == 10_000
    assert pc.BOOTSTRAP_CONFIDENCE == 0.95
    assert pc.STATISTICS["replicates"] == 10_000
    assert pc.ECE_SPECIFICATION["bins"] == 15
    assert pc.LOG_PROBABILITY_FLOOR == 1e-12
    assert set(pc.DIAGNOSTIC_SLICES) == {
        "opponent_stratum",
        "observer_color",
        "progress_bucket",
        "piece_moved",
        "true_rank",
        "opponent_setup_source",
    }


def test_bank_arithmetic_is_exactly_the_common_contract():
    assert pc.VALIDATION_BANK_CASES == 512
    assert pc.VALIDATION_BANK_GAMES == 1_024
    assert pc.TEST_BANK_CASES == 2_048
    assert pc.TEST_BANK_GAMES == 4_096
    assert pc.VALIDATION_CASES_PER_CELL == 32
    assert pc.TEST_CASES_PER_CELL == 128
    document = pc.bank_document()
    strata = [entry["stratum"] for entry in document["strata"]]
    assert tuple(strata) == OPPONENT_STRATA
    for entry in document["strata"]:
        neural = entry["opponent_checkpoint_sha256"] is not None
        rule = entry["opponent_policy_id"] is not None
        assert neural != rule  # exactly one binding per stratum


# ---------------------------------------------------------------------------
# Gate boundary behaviour — strict vs non-strict, exactly
# ---------------------------------------------------------------------------


def test_gate_a_boundaries():
    assert pc.evaluate_gate_a(0.97, -1e-9)["passed"]  # non-strict ratio edge
    assert not pc.evaluate_gate_a(0.97 + 1e-9, -1e-9)["passed"]
    assert not pc.evaluate_gate_a(0.97, 0.0)["passed"]  # strict upper bound
    assert pc.evaluate_gate_a(0.5, -1e-12)["passed"]
    assert not pc.evaluate_gate_a(float("nan"), -1.0)["passed"]
    assert not pc.evaluate_gate_a(0.5, float("nan"))["passed"]


def test_gate_b_boundaries():
    assert pc.evaluate_gate_b(0.03, 1e-12)["passed"]  # non-strict delta edge
    assert not pc.evaluate_gate_b(0.03 - 1e-9, 1e-12)["passed"]
    assert not pc.evaluate_gate_b(0.03, 0.0)["passed"]  # strict lower bound
    assert not pc.evaluate_gate_b(float("inf"), float("nan"))["passed"]


def test_gate_c_boundaries():
    strata = {stratum: 0.12 for stratum in OPPONENT_STRATA}
    assert pc.evaluate_gate_c(0.08, strata, 0.01)["passed"]  # all edges non-strict
    assert not pc.evaluate_gate_c(0.08 + 1e-9, strata, 0.01)["passed"]
    worse = dict(strata)
    worse[OPPONENT_STRATA[0]] = 0.12 + 1e-9
    assert not pc.evaluate_gate_c(0.08, worse, 0.01)["passed"]
    assert not pc.evaluate_gate_c(0.08, strata, 0.01 + 1e-9)["passed"]
    missing = {s: 0.05 for s in OPPONENT_STRATA[:-1]}
    assert not pc.evaluate_gate_c(0.05, missing, 0.0)["passed"]


def test_gate_d_boundaries():
    strata = {stratum: 1.05 for stratum in OPPONENT_STRATA}
    assert pc.evaluate_gate_d(strata)["passed"]
    strata[OPPONENT_STRATA[3]] = 1.05 + 1e-9
    assert not pc.evaluate_gate_d(strata)["passed"]
    assert not pc.evaluate_gate_d({s: 1.0 for s in OPPONENT_STRATA[:-1]})["passed"]
    nan_strata = {stratum: 1.0 for stratum in OPPONENT_STRATA}
    nan_strata[OPPONENT_STRATA[0]] = float("nan")
    assert not pc.evaluate_gate_d(nan_strata)["passed"]


def test_gate_e_and_f_zero_tolerance():
    zeros = {name: 0 for name in pc.SAMPLER_ZERO_TOLERANCE_COUNTERS}
    assert pc.evaluate_gate_e(zeros)["passed"]
    one = dict(zeros)
    one["inventory_errors"] = 1
    assert not pc.evaluate_gate_e(one)["passed"]
    missing = {name: 0 for name in pc.SAMPLER_ZERO_TOLERANCE_COUNTERS[:-1]}
    assert not pc.evaluate_gate_e(missing)["passed"]
    boolean = dict(zeros)
    boolean["inventory_errors"] = False  # a boolean is not a counter
    assert not pc.evaluate_gate_e(boolean)["passed"]

    zeros_f = {name: 0 for name in pc.INFORMATION_SAFETY_ZERO_COUNTERS}
    assert pc.evaluate_gate_f(zeros_f)["passed"]
    bad_f = dict(zeros_f)
    bad_f["belief_output_differences"] = 2
    assert not pc.evaluate_gate_f(bad_f)["passed"]


def test_gate_g_boundaries():
    legs = {leg: True for leg in pc.REPRODUCIBILITY_TOPOLOGY_LEGS}
    assert pc.evaluate_gate_g(legs, 500.0)["passed"]  # non-strict ceiling edge
    assert not pc.evaluate_gate_g(legs, 500.0 + 1e-9)["passed"]
    broken = dict(legs)
    broken["workers_12"] = False
    assert not pc.evaluate_gate_g(broken, 100.0)["passed"]
    assert not pc.evaluate_gate_g(
        {leg: True for leg in pc.REPRODUCIBILITY_TOPOLOGY_LEGS[:-1]}, 100.0
    )["passed"]
    assert not pc.evaluate_gate_g(legs, float("nan"))["passed"]


def test_gate_h_requires_every_identity_exact():
    observed = {
        key: value for key, value in pc.GATE_H.items() if key not in ("gate", "name")
    }
    assert pc.evaluate_gate_h(observed)["passed"]
    moved = dict(observed)
    moved["belief_head_digest"] = "0" * 64
    assert not pc.evaluate_gate_h(moved)["passed"]
    stepped = dict(observed)
    stepped["phase11_optimizer_steps"] = 1
    assert not pc.evaluate_gate_h(stepped)["passed"]
    assert not pc.evaluate_gate_h({})["passed"]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classification_logic_is_total_and_exact():
    all_pass = {gate: True for gate in "ABCDEFGH"}
    assert pc.classify_phase11(all_pass) == "PASS-SEARCH-READY"
    for gate in "ABCDEFGH":
        one_fail = {name: name != gate for name in "ABCDEFGH"}
        assert pc.classify_phase11(one_fail) == "FAIL"
    assert pc.classify_phase11(all_pass, integrity_established=False) == "BLOCKED"
    assert pc.classify_phase11(all_pass, experiment_valid=False) == "BLOCKED"
    with pytest.raises(pc.Phase11ContractError):
        pc.classify_phase11({gate: True for gate in "ABCDEFG"})
    assert set(pc.CLASSIFICATIONS) == {"PASS-SEARCH-READY", "FAIL", "BLOCKED"}


def test_hard_gate_ids_are_a_through_h():
    assert pc.HARD_GATE_IDS == ("A", "B", "C", "D", "E", "F", "G", "H")
    assert pc.GATE_A["r_ce_max"] == 0.97
    assert pc.GATE_B["delta_top1_min"] == 0.03
    assert pc.GATE_C["ece_overall_max"] == 0.08
    assert pc.GATE_C["stratum_ece_max"] == 0.12
    assert pc.GATE_C["brier_delta_upper_max"] == 0.01
    assert pc.GATE_D["stratum_r_ce_max"] == 1.05
    assert pc.GATE_G["p95_forward_64_max_ms"] == 500.0


# ---------------------------------------------------------------------------
# System template and ledger schema
# ---------------------------------------------------------------------------


def test_system_template_binds_now_and_leaves_five_slots():
    document = pc.system_document()
    bound = document["bound_now"]
    assert bound["belief_model"]["sha256"] == pc.GATE_H["phase9_checkpoint_sha256"]
    assert bound["belief_model"]["belief_head_digest"] == pc.ACCEPTED_BELIEF_HEAD_DIGEST
    assert bound["setup_selector"]["config_sha256"] == pc.ACCEPTED_SELECTOR_CONFIG_SHA256
    slots = [entry["slot"] for entry in document["unbound_slots"]]
    assert slots == [
        "evaluator_implementation",
        "sampler_implementation",
        "information_safety_evidence",
        "runtime_benchmark",
        "bank_digests",
    ]
    assert all(entry["filled_by"] == "Agent 6" for entry in document["unbound_slots"])


def test_ledger_schema_is_frozen():
    assert pc.LEDGER_ENTRY_FIELDS == (
        "ledger_version",
        "agent",
        "stage",
        "bank_version",
        "purpose",
        "structural_only",
        "neural_inference_count",
        "scored_prediction_count",
        "privileged_truth_count",
        "outcome_count",
    )
    assert pc.LEDGER_RELATIVE_PATH.endswith("phase11_bank_access_ledger.jsonl")


def test_runtime_benchmark_configuration_is_frozen_before_any_measurement():
    configuration = pc.RUNTIME_BENCHMARK_CONFIGURATION
    assert configuration["backend"] == "cpu"
    assert configuration["dtype"] == "float32"
    assert configuration["torch_threads"] == 1
    assert configuration["ceiling_ms"] == 500.0
    assert configuration["state_count"] == 480
    assert configuration["gate_quantity"] == "p95(forward_plus_64_worlds)"
    assert len(pc.REPRODUCIBILITY_TOPOLOGY_LEGS) == 8


def test_belief_head_identity_constants_are_well_formed():
    assert pc.BELIEF_HEAD_TENSOR_NAMES == ("belief_output.bias", "belief_output.weight")
    assert pc.BELIEF_HEAD_TENSOR_SHAPES["belief_output.weight"] == (12, 128)
    assert pc.BELIEF_HEAD_TENSOR_SHAPES["belief_output.bias"] == (12,)
    assert len(pc.ACCEPTED_BELIEF_HEAD_DIGEST) == 64
    assert not math.isnan(pc.ACCEPTED_GLOBAL_OPTIMIZER_STEP)
    assert pc.ACCEPTED_GLOBAL_OPTIMIZER_STEP == 47_086
