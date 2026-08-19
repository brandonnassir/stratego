"""Phase 11 Agent 5: the four acceptance artifacts.

These tests read the artifacts the Agent 5 harness wrote and check that
they say what Agent 6 and Agent 7 will rely on: the integrated run covered
the whole validation bank through the frozen pipeline, the independent
recomputation agreed, the Agent 3 and Agent 4 evidence is bound by digest,
the leakage audit is clean, the implementation freeze is complete and
re-derivable from live bytes, and the sealed test bank was never scored.

`full_suite_green` is a claim about the suite that contains this test, so
it is checked for consistency against the recorded measurement rather than
asserted — the accepted Phase 10 / Agent 2 / Agent 3 / Agent 4 pattern. The
artifacts are skipped when absent so a fresh clone still runs green.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from stratego.evaluation import phase11_pipeline as pipeline
from stratego.evaluation import phase11_recompute as recompute
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
    return _load("agent_05_acceptance.json")


@pytest.fixture(scope="module")
def metrics():
    return _load("agent_05_validation_metrics.json")


@pytest.fixture(scope="module")
def freeze():
    return _load("agent_05_validation_freeze.json")


@pytest.fixture(scope="module")
def strata_rows():
    path = DATA_DIRECTORY / "agent_05_validation_strata.csv"
    if not path.exists():
        pytest.skip("agent_05_validation_strata.csv has not been produced yet")
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


def test_the_acceptance_status_agrees_with_its_own_gates(acceptance):
    gates = acceptance["completion_gates"]
    expected = "PASS" if all(gates.values()) else "FAIL"
    assert acceptance["status"] == expected
    assert acceptance["gates_true"] == sum(1 for value in gates.values() if value)
    assert acceptance["gates_total"] == len(gates)
    assert acceptance["false_gates"] == sorted(
        name for name, value in gates.items() if not value
    )


def test_every_gate_other_than_the_self_referential_one_is_true(acceptance):
    """The suite gate is about the run containing this test; the rest are not."""
    false_gates = set(acceptance["false_gates"])
    assert false_gates <= {"full_suite_green"}, sorted(false_gates)


def test_acceptance_carries_the_twenty_two_required_gates(acceptance):
    required = {
        "agents1_4_pass",
        "validation_bank_exact",
        "validation_games_exact",
        "full_pipeline_complete",
        "predictive_metrics_complete",
        "all_slices_complete",
        "bootstrap_complete",
        "independent_recompute_pass",
        "sampler_evidence_bound",
        "safety_evidence_bound",
        "reproducibility_evidence_bound",
        "runtime_evidence_bound",
        "validation_privileged_boundary_clean",
        "no_test_prediction_access",
        "no_test_truth_access",
        "no_threshold_change",
        "no_calibration",
        "no_belief_update",
        "no_sampler_change",
        "upstream_assets_unchanged",
        "final_implementation_freeze_complete",
        "full_suite_green",
    }
    assert required <= set(acceptance["completion_gates"])


def test_every_forbidden_operation_counter_is_zero(acceptance):
    nonzero = {
        name: value
        for name, value in acceptance["forbidden_operation_counters"].items()
        if value
    }
    assert nonzero == {}


def test_no_optimizer_step_and_no_belief_head_movement(acceptance):
    preservation = acceptance["preservation"]
    assert preservation["optimizer_step_delta"] == 0
    assert preservation["optimizer_steps_run"] == 0
    assert preservation["belief_head_unchanged"] is True
    assert preservation["checkpoint_unchanged"] is True
    assert preservation["sampler_identity_unchanged"] is True
    assert preservation["problems"] == []


def test_upstream_stack_is_unchanged(acceptance):
    preservation = acceptance["preservation"]
    assert preservation["p10d_unchanged"] is True
    assert preservation["anchor_unchanged"] is True
    assert preservation["phase7_unchanged"] is True
    assert preservation["bank_files_unchanged"] is True
    assert preservation["bound_evidence_unchanged"] is True


def test_the_test_bank_is_still_sealed(acceptance):
    sealing = acceptance["test_bank_sealing"]
    assert sealing["test_bank_structural_only"] is True
    assert sealing["violations"] == []
    assert sealing["scored_prediction_total"] == 0
    assert sealing["privileged_truth_total"] == 0
    assert sealing["neural_inference_total"] == 0
    assert sealing["outcome_total"] == 0


def test_the_frozen_inputs_match_the_accepted_identities(acceptance):
    frozen = acceptance["frozen_inputs"]
    assert frozen["belief_head_digest"] == contract.ACCEPTED_BELIEF_HEAD_DIGEST
    assert frozen["phase9_checkpoint_sha256"] == (
        contract.ACCEPTED_PHASE9_CHECKPOINT_SHA256
    )
    assert frozen["phase9_model_state_digest"] == (
        contract.ACCEPTED_PHASE9_MODEL_STATE_DIGEST
    )
    assert frozen["phase9_parameters"] == contract.ACCEPTED_PHASE9_PARAMETERS
    assert frozen["global_optimizer_step"] == contract.ACCEPTED_GLOBAL_OPTIMIZER_STEP
    assert frozen["selector_config_sha256"] == contract.ACCEPTED_SELECTOR_CONFIG_SHA256
    assert frozen["phase7_library_content_digest"] == (
        contract.PHASE7_LIBRARY_CONTENT_DIGEST
    )


def test_the_sampler_bytes_are_the_immutable_ones(acceptance):
    digests = acceptance["frozen_inputs"]["sampler_module_sha256"]
    assert digests["stratego/evaluation/phase11_sampler.py"] == (
        "a0119f0126a1100c3fd74a20a703ea47c183d43e7fb1b6822aa15c7e34b921e8"
    )
    live = hashlib.sha256(
        (REPOSITORY_ROOT / "stratego/evaluation/phase11_sampler.py").read_bytes()
    ).hexdigest()
    assert live == digests["stratego/evaluation/phase11_sampler.py"]


def test_the_agent5_run_never_authorized_the_sealed_bank(acceptance):
    access = acceptance["leakage_audit"]["no_test_access"]
    assert access["scored_prediction_total"] == 0
    assert access["privileged_truth_total"] == 0
    assert access["neural_inference_total"] == 0
    assert access["outcome_total"] == 0
    assert access["seal_behaviour"]["test_refused_without_authorization"] is True


def test_the_recorded_readings_name_the_gate_a_risk(acceptance):
    readings = {entry["reading"] for entry in acceptance["recorded_readings"]}
    assert "gate_a_would_fail_on_validation_nothing_retuned" in readings
    assert acceptance["forbidden_operation_counters"][
        "retuning_actions_after_r_ce_reading"
    ] == 0


def test_the_suite_gate_agrees_with_the_recorded_measurement(acceptance):
    suite = acceptance.get("suite")
    expected = bool(suite) and suite.get("returncode") == 0
    assert acceptance["completion_gates"]["full_suite_green"] == expected


# ---------------------------------------------------------------------------
# The integrated run
# ---------------------------------------------------------------------------


def test_the_run_covered_the_whole_validation_bank(metrics):
    assert metrics["bank_version"] == contract.VALIDATION_BANK_VERSION
    assert metrics["games"] == contract.VALIDATION_BANK_GAMES
    assert metrics["prediction_events"] > 0
    assert metrics["overall"]["cases_with_events"] == contract.VALIDATION_BANK_CASES


def test_the_run_reproduces_the_agent2_store_on_every_content_digest(metrics):
    """Content, not the frozen manifest digest — which embeds a duration.

    See the `store_manifest_digest_embeds_a_wall_clock_duration` reading:
    `phase11_records.manifest_digest` covers each game's `forward_seconds`,
    so two executions of the same frozen bank cannot agree on it. The
    content digest covers exactly what a replay determines.
    """
    agent2 = _load("agent_02_predictive_metrics.json")
    assert metrics["reproduces_agent2_store"] is True
    assert metrics["store_content_digest"] == metrics["agent2_store_content_digest"]
    assert metrics["prediction_events"] == agent2["prediction_events"]
    assert metrics["observer_decisions"] == agent2["observer_decisions"]


def test_the_store_content_digest_is_recomputable_from_both_stores(metrics):
    from stratego.evaluation import phase11_records as records

    agent5_root = Path(metrics["store_root"])
    if not (agent5_root / "manifest.json").exists():
        pytest.skip("the Agent 5 prediction store is not on this volume")
    live = pipeline.store_content_digest(records.read_manifest(agent5_root))
    assert live == metrics["store_content_digest"]


def test_the_frozen_manifest_digest_difference_is_reported_not_hidden(metrics, acceptance):
    """The defect must be surfaced, and must not be silently patched."""
    assert metrics["agent2_manifest_digest_matches"] is False
    readings = {entry["reading"] for entry in acceptance["recorded_readings"]}
    assert "store_manifest_digest_embeds_a_wall_clock_duration" in readings
    # The accepted module was not modified: its digest exclusions still
    # omit the per-game timing, which is exactly what was reported.
    assert records_manifest_exclusions() == (
        "store_root",
        "written_at",
        "duration_seconds",
    )


def records_manifest_exclusions():
    import inspect

    from stratego.evaluation import phase11_records

    source = inspect.getsource(phase11_records.manifest_digest)
    start = source.index('not in ("') + len('not in (')
    end = source.index(")", start)
    return tuple(
        part.strip().strip('"') for part in source[start:end].split(",") if part.strip()
    )


def test_the_metric_block_reproduces_the_agent2_readings(metrics):
    agent2 = _load("agent_02_predictive_metrics.json")
    for name, block in agent2["overall"]["metrics"].items():
        for key in ("point", "lower", "upper"):
            assert metrics["overall"]["metrics"][name][key] == pytest.approx(
                block[key], abs=1e-12
            )
    assert metrics["overall"]["ece_learned"]["ece"] == pytest.approx(
        agent2["overall"]["ece_learned"]["ece"], abs=1e-12
    )


def test_every_bootstrap_used_the_frozen_replicates_and_confidence(metrics):
    for block in metrics["overall"]["metrics"].values():
        assert block["replicates"] == contract.BOOTSTRAP_REPLICATES
        assert block["confidence"] == contract.BOOTSTRAP_CONFIDENCE


def test_every_mandatory_diagnostic_slice_is_present(metrics):
    assert sorted(metrics["slices"]) == sorted(contract.DIAGNOSTIC_SLICES)
    assert sorted(metrics["slices"]["opponent_stratum"]) == sorted(
        contract.OPPONENT_STRATA
    )
    assert sorted(metrics["slices"]["observer_color"]) == ["blue", "red"]
    assert sorted(metrics["slices"]["opponent_setup_source"]) == sorted(
        contract.SETUP_SOURCES
    )


def test_the_integrated_sampler_pass_is_clean(metrics):
    sampler = metrics["sampler_checks"]
    assert sampler["sampler_version"] == contract.BELIEF_SAMPLER_VERSION
    assert sampler["all_counters_zero"] is True
    assert sorted(sampler["counters"]) == sorted(
        contract.SAMPLER_ZERO_TOLERANCE_COUNTERS
    )
    assert all(value == 0 for value in sampler["counters"].values())
    assert sampler["world_ordinals_per_request"] == pipeline.SAMPLE_WORLD_ORDINALS


def test_the_integrated_schedule_took_every_slot_it_could(metrics):
    """The rule's guarantee: every eligible game contributes, nothing dropped."""
    sampler = metrics["sampler_checks"]
    accounting = sampler["schedule_accounting"]
    assert accounting["every_eligible_game_contributes"] is True
    assert accounting["realized_equals_attainable"] is True
    assert accounting["games"] == contract.VALIDATION_BANK_GAMES
    assert (
        accounting["games_with_eligible_decisions"]
        + accounting["games_without_eligible_decisions"]
        == accounting["games"]
    )
    assert accounting["schedule_slots_attainable"] <= accounting["schedule_slots_nominal"]
    assert sampler["worlds"] == (
        accounting["schedule_slots_realized"] * sampler["world_ordinals_per_request"]
    )


def test_the_world_floor_is_met_by_the_bound_and_combined_evidence(metrics, acceptance):
    """The 250,000 floor is Agent 3's, and Agent 5 reports its own shortfall."""
    sampler = metrics["sampler_checks"]
    bound = metrics["bound_evidence"]
    assert bound["sampler_audit_worlds"] >= contract.SAMPLER_AUDIT_MIN_WORLDS
    assert (
        sampler["worlds"] + bound["sampler_audit_worlds"]
        >= contract.SAMPLER_AUDIT_MIN_WORLDS
    )
    if sampler["worlds"] < contract.SAMPLER_AUDIT_MIN_WORLDS:
        assert sampler["meets_world_floor"] is False
        readings = {entry["reading"] for entry in acceptance["recorded_readings"]}
        assert "integrated_schedule_realizes_fewer_slots_than_nominal" in readings
    assert pipeline.SAMPLE_DECISIONS_PER_GAME == 4, (
        "the schedule rule must not be retuned to chase the world count"
    )


def test_the_bound_evidence_is_the_agent3_and_agent4_evidence(metrics):
    bound = metrics["bound_evidence"]
    assert all(value == 0 for value in bound["safety_counters"].values())
    assert all(value == 0 for value in bound["sampler_audit_counters"].values())
    assert bound["sampler_audit_worlds"] >= contract.SAMPLER_AUDIT_MIN_WORLDS
    assert sorted(bound["leg_exact"]) == sorted(
        contract.REPRODUCIBILITY_TOPOLOGY_LEGS
    )
    assert all(bound["leg_exact"].values())
    assert bound["p95_forward_64_ms"] <= contract.GATE_G["p95_forward_64_max_ms"]


def test_the_eight_gates_were_all_recomputed(metrics):
    assert sorted(metrics["gates"]) == list(contract.HARD_GATE_IDS)
    for gate in contract.HARD_GATE_IDS:
        assert "passed" in metrics["gates"][gate]
        assert metrics["gates"][gate]["checks"]


def test_gate_e_reads_the_sum_of_both_sampler_passes(metrics):
    combined = metrics["gates"]["E"]
    assert combined["passed"] is True
    sampler = metrics["sampler_checks"]["counters"]
    audit = metrics["bound_evidence"]["sampler_audit_counters"]
    assert all(
        sampler[name] + audit.get(name, 0) == 0
        for name in contract.SAMPLER_ZERO_TOLERANCE_COUNTERS
    )


def test_gates_f_g_and_h_pass_on_validation(metrics):
    for gate in ("F", "G", "H"):
        assert metrics["gates"][gate]["passed"] is True, gate


def test_the_known_r_ce_reading_is_reported_not_repaired(metrics):
    quantities = metrics["gate_quantities"]
    assert quantities["r_ce"] == pytest.approx(0.9750, abs=5e-4)
    # It exceeds the frozen ceiling, and the ceiling did not move.
    assert contract.GATE_A["r_ce_max"] == 0.97
    assert metrics["gates"]["A"]["checks"]["r_ce_le_0_97"] is False
    # The paired CE delta is still significantly negative.
    assert quantities["ce_delta_upper"] < 0.0
    assert metrics["gates"]["A"]["checks"]["ce_delta_upper_lt_0"] is True


# ---------------------------------------------------------------------------
# Independent recomputation
# ---------------------------------------------------------------------------


def test_the_independent_recomputation_agreed(metrics):
    comparison = metrics["independent_recompute"]["comparison"]
    assert comparison["recompute_version"] == recompute.RECOMPUTE_VERSION
    assert comparison["within_tolerance"] is True
    assert comparison["max_deviation"] <= comparison["tolerance"]
    assert comparison["both_nan_comparisons"] == 0
    assert comparison["quantities_compared"] >= 72


def test_the_independent_gate_quantities_match_the_primary(metrics):
    primary = metrics["gate_quantities"]
    independent = metrics["independent_recompute"]["independent_gate_quantities"]
    for name in (
        "r_ce",
        "ce_delta_upper",
        "delta_top1",
        "delta_top1_lower",
        "brier_delta_upper",
        "ece_overall",
    ):
        assert independent[name] == pytest.approx(primary[name], abs=1e-9)
    for stratum in contract.OPPONENT_STRATA:
        assert independent["stratum_r_ce"][stratum] == pytest.approx(
            primary["stratum_r_ce"][stratum], abs=1e-9
        )
        assert independent["stratum_ece"][stratum] == pytest.approx(
            primary["stratum_ece"][stratum], abs=1e-9
        )


def test_the_per_event_audit_agreed(metrics):
    deviations = metrics["per_event_audit_max_deviation"]
    assert deviations
    assert all(value <= 1e-9 for value in deviations.values())


# ---------------------------------------------------------------------------
# The strata CSV
# ---------------------------------------------------------------------------


def test_the_strata_csv_has_one_row_per_stratum(strata_rows):
    assert len(strata_rows) == len(contract.OPPONENT_STRATA)
    assert sorted(row["opponent_stratum"] for row in strata_rows) == sorted(
        contract.OPPONENT_STRATA
    )


def test_every_stratum_satisfies_gates_c_and_d(strata_rows):
    for row in strata_rows:
        assert float(row["r_ce"]) <= contract.GATE_D["stratum_r_ce_max"]
        assert float(row["ece_learned"]) <= contract.GATE_C["stratum_ece_max"]
        assert row["gate_d_r_ce_le_1_05"] == "true"
        assert row["gate_c_ece_le_0_12"] == "true"


def test_the_strata_csv_ratios_match_the_independent_path(strata_rows):
    for row in strata_rows:
        assert float(row["independent_r_ce"]) == pytest.approx(
            float(row["r_ce"]), abs=1e-6
        )
        assert float(row["independent_ece_learned"]) == pytest.approx(
            float(row["ece_learned"]), abs=1e-6
        )


def test_every_stratum_carries_events_and_cases(strata_rows):
    for row in strata_rows:
        assert int(row["events"]) > 0
        assert int(row["cases_with_events"]) == 64


# ---------------------------------------------------------------------------
# The implementation freeze
# ---------------------------------------------------------------------------


def test_the_freeze_names_the_frozen_versions(freeze):
    document = freeze["freeze"]
    assert document["freeze_version"] == pipeline.PIPELINE_VERSION
    assert document["final_test_entry_point"] == pipeline.FINAL_TEST_ENTRY_POINT
    assert document["evaluator_version"] == contract.EVALUATOR_VERSION
    assert document["remaining_count_baseline"] == (
        contract.REMAINING_COUNT_BASELINE_VERSION
    )
    assert document["world_baseline"] == contract.WORLD_BASELINE_VERSION
    assert document["sampler_version"] == contract.BELIEF_SAMPLER_VERSION
    assert document["information_safety_version"] == (
        contract.INFORMATION_SAFETY_VERSION
    )
    assert document["pipeline_stages"] == list(pipeline.PIPELINE_STAGES)
    assert document["sealed_banks"] == list(pipeline.SEALED_BANKS)


def test_the_freeze_digest_is_reproducible_from_its_own_document(freeze):
    document = dict(freeze["freeze"])
    recorded = document.pop("freeze_digest")
    assert pipeline.freeze_identity(document) == recorded


def test_the_freeze_module_digests_match_the_live_bytes(freeze):
    live = pipeline.module_sha256(REPOSITORY_ROOT)
    recorded = freeze["freeze"]["module_sha256"]
    assert sorted(recorded) == sorted(pipeline.FROZEN_IMPLEMENTATION_MODULES)
    for name, digest in recorded.items():
        assert live[name] == digest, name


def test_the_freeze_binds_every_upstream_evidence_artifact(freeze):
    bound = freeze["freeze"]["bound_evidence"]
    for name, digest in bound.items():
        path = DATA_DIRECTORY / name
        assert path.exists(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    assert "agent_04_information_safety.json" in bound
    assert "agent_04_reproducibility.json" in bound
    assert "agent_04_runtime.csv" in bound
    assert "agent_03_sampler_audit.json" in bound


def test_the_freeze_records_the_frozen_statistics_and_runtime(freeze):
    document = freeze["freeze"]
    statistics = document["statistics_version"]
    assert statistics["bootstrap_replicates"] == contract.BOOTSTRAP_REPLICATES
    assert statistics["bootstrap_confidence"] == contract.BOOTSTRAP_CONFIDENCE
    assert statistics["ece_bins"] == int(contract.ECE_SPECIFICATION["bins"])
    assert statistics["independent_recompute_version"] == recompute.RECOMPUTE_VERSION
    runtime = document["runtime_backend"]
    assert runtime["backend"] == "cpu"
    assert runtime["dtype"] == "float32"
    assert runtime["torch_threads"] == 1
    assert runtime["measured_p95_forward_64_ms"] <= runtime["ceiling_ms"] == 500.0


def test_the_freeze_carries_both_bank_digests(freeze):
    digests = freeze["freeze"]["bank_digests"]
    validation = _load("agent_01_validation_bank.json")["manifest"]["bank_digest"]
    test = _load("agent_01_test_bank.json")["manifest"]["bank_digest"]
    assert digests[contract.VALIDATION_BANK_VERSION] == validation
    assert digests[contract.TEST_BANK_VERSION] == test


def test_the_handoff_is_addressed_to_agent_6(freeze, acceptance):
    handoff = freeze["handoff_to_agent_6"]
    assert handoff["for_agent"] == 6
    assert handoff["freeze_digest"] == freeze["freeze"]["freeze_digest"]
    assert handoff["final_test_entry_point"] == pipeline.FINAL_TEST_ENTRY_POINT
    dependencies = handoff["immutable_dependencies"]
    assert dependencies["belief_head_digest"] == contract.ACCEPTED_BELIEF_HEAD_DIGEST
    assert dependencies["sampler_version"] == contract.BELIEF_SAMPLER_VERSION
    assert acceptance["handoff_to_agent_6"] == handoff


def test_the_freeze_was_validated_on_the_validation_bank_only(freeze):
    validated = freeze["validated_on"]
    assert validated["bank_version"] == contract.VALIDATION_BANK_VERSION
    assert validated["reproduces_agent2_store"] is True
    assert validated["prediction_events"] > 0
    assert validated["sampler_worlds"] > 0
