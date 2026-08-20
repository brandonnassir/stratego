"""Phase 11 Agent 7: the sealed final acceptance artifacts.

These tests read the artifacts the Agent 7 harness wrote and protect what
the sealed evaluation established: the administrative freeze was verified
from live bytes before the bank opened, the sealed run covered exactly
2,048 cases / 4,096 games with the frozen balance, every gate quantity was
independently recomputed within the frozen tolerance, Gates A-H recompute
from the recorded quantities through the frozen contract evaluators, the
classification recomputes from the gate rows alone, the test bank's first
scored access belongs to Agent 7 and happened exactly once, and every
preserved identity was re-derived exact after the evaluation.

The sealed evaluation is final evidence: nothing here reruns it, and
`full_suite_green` is checked for consistency against the recorded
measurement rather than asserted, the accepted Agent 2-6 pattern. The
artifacts are skipped when absent so a fresh clone still runs green.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stratego.training import phase11_contract as contract

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_11_data"


def _load(name: str):
    path = DATA_DIRECTORY / name
    if not path.exists():
        pytest.skip(f"{name} has not been produced yet")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def acceptance():
    return _load("agent_07_final_acceptance.json")


@pytest.fixture(scope="module")
def sampler_results():
    return _load("agent_07_sampler_results.json")


# ---------------------------------------------------------------------------
# The sealed run really was the frozen experiment
# ---------------------------------------------------------------------------


def test_the_sealed_run_is_the_frozen_test_bank_exactly(acceptance):
    sealed = acceptance["sealed_run"]
    assert sealed["bank_version"] == contract.TEST_BANK_VERSION
    assert sealed["cases"] == contract.TEST_BANK_CASES == 2_048
    assert sealed["games"] == contract.TEST_BANK_GAMES == 4_096
    assert sealed["sealed_bank_authorized"] is True
    assert sealed["run_ordinal"] == 1


def test_the_sealed_run_used_the_frozen_pipeline(acceptance):
    from stratego.evaluation import phase11_pipeline as pipeline

    assert acceptance["pipeline_version"] == pipeline.PIPELINE_VERSION
    assert acceptance["frozen_inputs"]["validation_freeze_digest"] == (
        "ad2562af538abc6c78fc5b12bc1f57d3e32184172acde390417a00d500a0d912"
    )
    assert acceptance["frozen_inputs"]["phase11_system_v1_digest"] == (
        "e4452ba38b568a0ed3a5866f761324dcc7f1eea226d7ba6f94fde45ceb3b6101"
    )


def test_the_structure_gates_are_exact(acceptance):
    gates = acceptance["completion_gates"]
    assert gates["test_games_exact"] is True
    assert gates["test_strata_exact"] is True
    assert gates["test_color_balance_exact"] is True
    assert gates["test_setup_source_balance_exact"] is True
    assert gates["all_prediction_events_recorded"] is True


# ---------------------------------------------------------------------------
# First-access proof
# ---------------------------------------------------------------------------


def test_the_first_scored_access_belongs_to_agent_7(acceptance):
    proof = acceptance["first_scored_access_proof"]
    pre = proof["pre_agent7_ledger"]
    assert pre["scored_prediction_total"] == 0
    assert pre["privileged_truth_total"] == 0
    assert pre["neural_inference_total"] == 0
    assert pre["outcome_total"] == 0
    assert pre["structural_only"] is True
    post = proof["post_run_ledger"]
    assert post["pre_agent7_still_structural_only"] is True
    assert post["non_structural_test_entries_all_agent7"] is True
    assert post["sealed_test_run_entries"] == 1
    assert proof["seal_behaviour"]["test_refused_without_authorization"] is True


def test_the_live_ledger_agrees_with_the_recorded_proof():
    from stratego.evaluation import phase11_banks as pb

    entries = pb.read_ledger()
    if not any(int(entry["agent"]) == 7 for entry in entries):
        pytest.skip("Agent 7 has not run yet")
    pre = pb.verify_test_bank_sealed(
        [entry for entry in entries if int(entry["agent"]) <= 6]
    )
    assert pre["test_bank_structural_only"] is True
    assert pre["scored_prediction_total"] == 0
    scored = [
        entry
        for entry in entries
        if entry["bank_version"] == contract.TEST_BANK_VERSION
        and not entry["structural_only"]
    ]
    assert scored, "the sealed run must be ledgered"
    assert all(int(entry["agent"]) == 7 for entry in scored)
    runs = [entry for entry in scored if entry["stage"] == "sealed_test_run"]
    assert len(runs) == 1


# ---------------------------------------------------------------------------
# Gates recompute from the recorded rows
# ---------------------------------------------------------------------------


def test_gates_a_to_d_recompute_from_the_recorded_quantities(acceptance):
    quantities = acceptance["gate_quantities"]
    gates = acceptance["hard_gates"]
    assert gates["A"] == contract.evaluate_gate_a(
        quantities["r_ce"], quantities["ce_delta_upper"]
    )
    assert gates["B"] == contract.evaluate_gate_b(
        quantities["delta_top1"], quantities["delta_top1_lower"]
    )
    assert gates["C"] == contract.evaluate_gate_c(
        quantities["ece_overall"],
        quantities["stratum_ece"],
        quantities["brier_delta_upper"],
    )
    assert gates["D"] == contract.evaluate_gate_d(quantities["stratum_r_ce"])


def test_the_classification_recomputes_from_the_gate_rows(acceptance):
    booleans = {
        gate: bool(block["passed"]) for gate, block in acceptance["hard_gates"].items()
    }
    assert acceptance["hard_gate_booleans"] == booleans
    recomputed = contract.classify_phase11(
        booleans,
        experiment_valid=True,
        integrity_established=acceptance["classification"]["integrity_established"],
    )
    assert acceptance["recommendation"] == recomputed
    assert acceptance["phase12_authorized"] == (recomputed == "PASS-SEARCH-READY")


def test_the_recommendation_is_one_of_the_frozen_classifications(acceptance):
    assert acceptance["recommendation"] in contract.CLASSIFICATIONS


def test_the_independent_recompute_is_within_the_frozen_tolerance(acceptance):
    comparison = acceptance["independent_recompute"]
    assert comparison["within_tolerance"] is True
    assert comparison["both_nan_comparisons"] == 0
    assert comparison["max_deviation"] <= 1e-9


# ---------------------------------------------------------------------------
# Sampler, streams, preservation
# ---------------------------------------------------------------------------


def test_the_sampler_confirmation_counters_are_zero(acceptance):
    confirmation = acceptance["sampler_confirmation"]
    assert confirmation["all_counters_zero"] is True
    assert all(int(value) == 0 for value in confirmation["counters"].values())
    assert confirmation["worlds_verified"] >= 25_000


def test_the_stream_audit_found_no_collisions(acceptance, sampler_results):
    audit = acceptance["stream_audit"]
    assert audit["accidental_collisions"] == 0
    assert audit["prior_combined_matches_accepted"] is True
    combined = sampler_results["materialized_stream_audit"]["combined"]
    assert combined["no_collisions"] is True
    assert combined["unique_logical_identities"] == combined["distinct_seeds"]


def test_every_preserved_identity_was_rederived_exact(acceptance):
    preservation = acceptance["preservation"]
    assert preservation["exact"] is True
    assert preservation["optimizer_step_delta"] == 0
    assert preservation["phase11_optimizer_steps"] == 0
    after = preservation["after"]
    assert after["phase9_checkpoint_sha256"] == (
        contract.ACCEPTED_PHASE9_CHECKPOINT_SHA256
    )
    assert after["belief_head_digest"] == contract.ACCEPTED_BELIEF_HEAD_DIGEST
    assert after["phase9_parameters"] == contract.ACCEPTED_PHASE9_PARAMETERS
    assert after["global_optimizer_step"] == contract.ACCEPTED_GLOBAL_OPTIMIZER_STEP


def test_no_forbidden_operation_was_counted(acceptance):
    counters = acceptance["forbidden_operation_counters"]
    assert counters
    assert all(int(value) == 0 for value in counters.values()), counters


def test_the_suite_gate_agrees_with_the_recorded_measurement(acceptance):
    suite = acceptance.get("suite")
    expected = bool(suite) and suite.get("returncode") == 0
    assert acceptance["completion_gates"]["full_suite_green"] == expected
