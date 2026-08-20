"""Phase 11 Agent 2 artifacts: the validation evidence checks itself.

Every check recomputes something from the tracked artifacts and the live
modules rather than trusting a stored summary. `full_suite_green` is a claim
about the suite that contains this test, so it is checked against the
recorded measurement rather than asserted (the accepted Phase 10 pattern).
"""

import csv
import json
import math
from pathlib import Path

import pytest

from stratego.evaluation.phase11_audit import NEGATIVE_CONTROLS
from stratego.training import phase11_contract as pc
from stratego.training.phase11_seed import OPPONENT_STRATA, SETUP_SOURCES

from ..training.phase11_frozen_digests import (
    BANK_DIGESTS,
    BELIEF_HEAD_DIGEST,
    CONTRACT_BUNDLE_DIGEST,
    CONTRACT_DIGESTS,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_11_data"
ACCEPTANCE_PATH = DATA_DIRECTORY / "agent_02_acceptance.json"
METRICS_PATH = DATA_DIRECTORY / "agent_02_predictive_metrics.json"
STRATUM_CSV_PATH = DATA_DIRECTORY / "agent_02_stratum_metrics.csv"
BASELINE_AUDIT_PATH = DATA_DIRECTORY / "agent_02_baseline_audit.json"
LEDGER_PATH = DATA_DIRECTORY / "phase11_bank_access_ledger.jsonl"

#: The instruction's twenty-four minimum completion gates. Gates may be
#: added, never weakened.
EXPECTED_GATES = (
    "agent1_pass",
    "contracts_verified",
    "validation_bank_verified",
    "test_bank_structural_only",
    "public_privileged_boundary_pass",
    "prediction_schema_exact",
    "rank_order_exact",
    "remaining_count_baseline_complete",
    "baseline_negative_controls_fire",
    "count_uniform_world_baseline_complete",
    "validation_games_exact",
    "validation_strata_exact",
    "validation_color_balance_exact",
    "validation_setup_source_balance_exact",
    "all_required_prediction_events_recorded",
    "metrics_finite",
    "independent_metric_recompute_pass",
    "evaluator_negative_controls_fire",
    "no_test_prediction_access",
    "no_test_truth_access",
    "no_belief_updates",
    "phase9_checkpoint_unchanged",
    "belief_head_unchanged",
    "full_suite_green",
)

SELF_REFERENTIAL_GATE = "full_suite_green"

pytestmark = pytest.mark.skipif(
    not ACCEPTANCE_PATH.exists(), reason="Agent 2 has not produced artifacts yet"
)


@pytest.fixture(scope="module")
def acceptance():
    return json.loads(ACCEPTANCE_PATH.read_text())


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS_PATH.read_text())


@pytest.fixture(scope="module")
def audit():
    return json.loads(BASELINE_AUDIT_PATH.read_text())


# ---------------------------------------------------------------------------
# Gates and status
# ---------------------------------------------------------------------------


def test_the_gate_set_is_exactly_the_instruction_list(acceptance):
    assert tuple(sorted(acceptance["completion_gates"])) == tuple(sorted(EXPECTED_GATES))
    assert acceptance["gates_total"] == len(EXPECTED_GATES)


def test_status_follows_from_the_gates(acceptance):
    gates = acceptance["completion_gates"]
    expected = "PASS" if all(gates.values()) else "BLOCKED"
    assert acceptance["status"] == expected
    assert acceptance["gates_true"] == sum(1 for value in gates.values() if value)
    assert acceptance["false_gates"] == sorted(
        name for name, value in gates.items() if not value
    )


def test_the_suite_gate_agrees_with_the_recorded_measurement(acceptance):
    suite = acceptance["suite"]
    expected = bool(suite) and suite.get("returncode") == 0
    assert acceptance["completion_gates"][SELF_REFERENTIAL_GATE] == expected


# ---------------------------------------------------------------------------
# Frozen inputs
# ---------------------------------------------------------------------------


def test_the_frozen_inputs_are_the_agent_1_freeze(acceptance):
    frozen = acceptance["frozen_inputs"]
    assert frozen["contract_bundle_digest"] == CONTRACT_BUNDLE_DIGEST
    assert frozen["contract_digests"] == CONTRACT_DIGESTS
    assert frozen["validation_bank_digest"] == BANK_DIGESTS["validation"]
    assert frozen["test_bank_digest"] == BANK_DIGESTS["test"]
    assert frozen["belief_head_digest"] == BELIEF_HEAD_DIGEST
    assert frozen["phase9_sha256"] == pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256
    assert frozen["phase9_model_state_digest"] == pc.ACCEPTED_PHASE9_MODEL_STATE_DIGEST
    assert frozen["phase9_parameters"] == 863_959
    assert frozen["phase10_closure_commit"] == pc.PHASE10_CLOSURE_COMMIT


def test_the_contract_digests_still_recompute_live(acceptance):
    assert pc.contract_digests() == acceptance["frozen_inputs"]["contract_digests"]
    assert pc.contract_bundle_digest() == CONTRACT_BUNDLE_DIGEST


# ---------------------------------------------------------------------------
# Preservation
# ---------------------------------------------------------------------------


def test_nothing_upstream_moved(acceptance):
    preservation = acceptance["preservation"]
    assert preservation["checkpoint_unchanged"]
    assert preservation["belief_head_unchanged"]
    assert preservation["p10d_unchanged"]
    assert preservation["phase7_unchanged"]
    assert preservation["anchor_unchanged"]
    assert preservation["problems"] == []


def test_the_optimizer_counter_moved_by_exactly_zero(acceptance):
    preservation = acceptance["preservation"]
    assert preservation["optimizer_step_before"] == pc.ACCEPTED_GLOBAL_OPTIMIZER_STEP
    assert preservation["optimizer_step_after"] == pc.ACCEPTED_GLOBAL_OPTIMIZER_STEP
    assert preservation["optimizer_step_delta"] == 0
    assert preservation["optimizer_steps_run"] == 0


def test_every_forbidden_counter_is_zero(acceptance):
    counters = acceptance["forbidden_operation_counters"]
    assert set(counters) >= {
        "phase11_optimizer_steps",
        "belief_calibration_operations",
        "test_bank_scored_accesses",
        "test_bank_privileged_truth_reads",
        "hidden_truth_inputs_to_inference",
    }
    assert all(value == 0 for value in counters.values()), counters


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def test_the_run_covered_the_whole_validation_bank_exactly(acceptance):
    run = acceptance["run"]
    assert run["games"] == pc.VALIDATION_BANK_GAMES == 1_024
    balance = run["balance"]
    assert balance["cases"] == pc.VALIDATION_BANK_CASES == 512
    assert balance["games_per_case"] == [2]
    assert balance["by_color"] == {"red": 512, "blue": 512}
    assert sorted(balance["by_stratum"]) == sorted(OPPONENT_STRATA)
    assert set(balance["by_stratum"].values()) == {128}
    assert sorted(balance["by_source"]) == sorted(SETUP_SOURCES)
    assert set(balance["by_source"].values()) == {512}


def test_the_privileged_pass_verified_every_decision(acceptance):
    truth = acceptance["run"]["truth_pass"]
    assert truth["identity_mismatches"] == 0
    assert truth["alignment_mismatches"] == 0
    assert truth["count_mismatches"] == 0
    assert truth["mask_mismatches"] == 0
    assert truth["unlabelled_events"] == 0
    assert truth["verified_decisions"] == acceptance["run"]["observer_decisions"]


def test_game_outcomes_are_reported_but_rank_nothing(acceptance):
    outcomes = acceptance["run"]["outcomes"]
    assert sum(outcomes.values()) == acceptance["run"]["games"]
    assert set(outcomes) <= {"win", "draw", "loss"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_every_overall_metric_is_finite(metrics):
    assert metrics["metrics_finite"]
    assert metrics["nonfinite_paths"] == []
    for token in pc.OVERALL_METRIC_TOKENS:
        block = metrics["overall"]["metrics"][token]
        for key in ("point", "lower", "upper"):
            assert math.isfinite(block[key]), (token, key)


def test_the_metric_block_is_internally_consistent(metrics):
    block = metrics["overall"]["metrics"]
    assert block["r_ce"]["point"] == pytest.approx(
        block["ce_learned"]["point"] / block["ce_baseline"]["point"], rel=1e-12
    )
    for name in ("ce", "top1", "brier"):
        assert block[f"{name}_delta"]["point"] == pytest.approx(
            block[f"{name}_learned"]["point"] - block[f"{name}_baseline"]["point"],
            abs=1e-12,
        )
        assert block[f"{name}_delta"]["lower"] <= block[f"{name}_delta"]["point"]
        assert block[f"{name}_delta"]["point"] <= block[f"{name}_delta"]["upper"]


def test_the_bootstrap_used_the_frozen_parameters(metrics):
    for token in pc.OVERALL_METRIC_TOKENS:
        block = metrics["overall"]["metrics"][token]
        assert block["replicates"] == pc.BOOTSTRAP_REPLICATES == 10_000
        assert block["confidence"] == pc.BOOTSTRAP_CONFIDENCE == 0.95
        assert block["metric_token"] == token


def test_the_stratum_slice_covers_all_eight_strata(metrics):
    stratum = metrics["slices"]["opponent_stratum"]
    assert sorted(stratum) == sorted(OPPONENT_STRATA)
    for name, block in stratum.items():
        assert block["events"] > 0, name
        assert math.isfinite(block["r_ce"]["point"]), name
        assert math.isfinite(block["ece_learned"]["ece"]), name


def test_every_required_diagnostic_slice_is_present(metrics):
    assert sorted(metrics["slices"]) == sorted(pc.DIAGNOSTIC_SLICES)


def test_the_ece_uses_fifteen_bins_and_pooled_events(metrics):
    ece = metrics["overall"]["ece_learned"]
    assert len(ece["bins"]) == pc.ECE_SPECIFICATION["bins"] == 15
    assert ece["events"] == metrics["prediction_events"]
    assert sum(row["events"] for row in ece["bins"]) == ece["events"]


def test_the_stratum_csv_matches_the_metric_artifact(metrics):
    with open(STRATUM_CSV_PATH, newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(OPPONENT_STRATA)
    for row in rows:
        block = metrics["slices"]["opponent_stratum"][row["stratum"]]
        assert int(row["events"]) == block["events"]
        assert float(row["r_ce"]) == pytest.approx(block["r_ce"]["point"])
        assert float(row["ece_learned"]) == pytest.approx(block["ece_learned"]["ece"])


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def test_the_independent_formula_audit_covered_every_event(metrics):
    audit = metrics["independent_formula_audit"]
    assert audit["within_tolerance"]
    assert audit["coverage"] == "every scored event"
    assert all(value <= audit["tolerance"] for value in audit["max_deviation"].values())


def test_the_scalar_audit_agrees_with_the_primary_path(audit):
    scalar = audit["scalar_recompute"]
    assert scalar["within_tolerance"]
    assert scalar["events"] > 0 and scalar["cases"] > 0
    assert scalar["max_case_aggregate_deviation"] <= scalar["tolerance"]


def test_every_negative_control_fires(audit):
    fired = audit["negative_controls"]["fired"]
    assert sorted(fired) == sorted(NEGATIVE_CONTROLS)
    assert all(fired.values()), fired
    assert audit["negative_controls"]["all_fire"]


def test_the_baseline_audit_found_no_disagreement(audit):
    edge = audit["baseline_edge_cases"]
    assert edge["pass"]
    assert edge["count_mismatches"] == 0
    assert edge["mask_mismatches"] == 0
    assert edge["conservation_failures"] == 0
    assert edge["distribution_mismatches"] == 0
    assert edge["baseline_zero_on_true_rank"] == 0
    assert edge["decisions"] > 0 and edge["hidden_pieces"] > 0


def test_every_world_baseline_counter_is_zero(audit):
    worlds = audit["count_uniform_world_sampler"]
    assert worlds["all_counters_zero"], worlds["counters"]
    assert worlds["pass"]
    assert worlds["worlds"] > 0
    assert set(worlds["counters"]) >= set(pc.SAMPLER_ZERO_TOLERANCE_COUNTERS)


# ---------------------------------------------------------------------------
# The seal and the ledger
# ---------------------------------------------------------------------------


def test_the_test_bank_seal_held_until_agent_7():
    """Agent 2's seal invariant in its permanent time-scoped form: every
    ledger entry up to and including Agent 6 is structural-only with zero
    counters, and the only non-structural test-bank entries are Agent 7's
    single authorized sealed evaluation, which postdates this agent."""
    from stratego.evaluation import phase11_banks as pb

    entries = pb.read_ledger()
    sealing = pb.verify_test_bank_sealed(
        [entry for entry in entries if entry["agent"] <= 6]
    )
    assert sealing["test_bank_structural_only"]
    assert sealing["scored_prediction_total"] == 0
    assert sealing["privileged_truth_total"] == 0
    assert sealing["neural_inference_total"] == 0
    assert sealing["outcome_total"] == 0
    scored = [
        entry
        for entry in entries
        if entry["bank_version"] == "phase11_test_bank_v1"
        and not entry["structural_only"]
    ]
    assert all(entry["agent"] == 7 for entry in scored)


def test_the_ledger_records_the_validation_run():
    from stratego.evaluation import phase11_banks as pb

    entries = pb.read_ledger()
    agent2 = [entry for entry in entries if entry["agent"] == 2]
    assert agent2, "Agent 2 wrote no ledger entry"
    scored = [entry for entry in agent2 if not entry["structural_only"]]
    assert scored
    assert all(
        entry["bank_version"] == pc.VALIDATION_BANK_VERSION for entry in scored
    )


def test_the_handoff_names_what_agent_3_needs(acceptance):
    handoff = acceptance["handoff_to_agent_3"]
    assert handoff["for_agent"] == 3
    assert handoff["belief_api"]["request_version"] == pc.BELIEF_REQUEST_VERSION
    assert (
        handoff["public_state_identity"]["document_version"]
        == pc.PUBLIC_STATE_DOCUMENT_VERSION
    )
    assert handoff["sampler_contract"]["digest"] == CONTRACT_DIGESTS[
        "phase11_belief_sampler_v1"
    ]
    assert handoff["sampler_contract"]["agent2_built_only"] == pc.WORLD_BASELINE_VERSION
    assert handoff["validation_public_states"]["public_shards"] == 1_024


def test_the_readings_are_recorded_for_the_reviewer(acceptance):
    readings = acceptance["recorded_readings"]
    assert readings
    names = {reading["reading"] for reading in readings}
    assert "optimizer_step_baseline_is_a_delta" in names
    assert "cases_without_events_are_excluded_from_the_case_mean" in names
    for reading in readings:
        assert reading["statement"] and reading["impact"]
