"""Phase 11 Agent 4 artifacts: the safety, topology and runtime evidence.

Every check recomputes something from the tracked artifacts and the live
contract rather than trusting a stored summary. `full_suite_green` is a
claim about the suite that contains this test, so it is checked against the
recorded measurement rather than asserted (the accepted Phase 10 / Agent 2 /
Agent 3 pattern).
"""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from stratego.training import phase11_contract as pc
from stratego.training.phase11_seed import (
    BENCHMARK_CELL_COUNT,
    BENCHMARK_STATE_COUNT,
    OPPONENT_STRATA,
    REPRO_REQUEST_COUNT,
    SAFETY_TRIAL_COUNT,
)

from ..training.phase11_frozen_digests import (
    BANK_DIGESTS,
    BELIEF_HEAD_DIGEST,
    CONTRACT_BUNDLE_DIGEST,
    CONTRACT_DIGESTS,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_11_data"
ACCEPTANCE_PATH = DATA_DIRECTORY / "agent_04_acceptance.json"
FROZEN_SETS_PATH = DATA_DIRECTORY / "agent_04_frozen_sets.json"
SAFETY_PATH = DATA_DIRECTORY / "agent_04_information_safety.json"
REPRO_PATH = DATA_DIRECTORY / "agent_04_reproducibility.json"
RUNTIME_PATH = DATA_DIRECTORY / "agent_04_runtime.csv"
STREAM_AUDIT_PATH = DATA_DIRECTORY / "agent_04_stream_audit.json"

#: The instruction's twenty-five minimum gates plus the Agent 4 additions
#: (round-robin exactness, the aggregate leg check, one rollup digest, the
#: detail counters, the cross-agent logit agreement, the two materialized
#: stream-identity gates, zero optimizer delta and upstream cleanliness).
#: Gates may be added, never weakened.
EXPECTED_GATES = (
    "agents1_3_pass",
    "hidden_truth_trials_ge_50k",
    "belief_output_changes_zero",
    "fixed_seed_sample_changes_zero",
    "forbidden_hidden_access_zero",
    "injection_controls_rejected",
    "safety_detail_counters_zero",
    "topology_request_set_frozen",
    "worker_1_exact",
    "worker_4_exact",
    "worker_12_exact",
    "forward_reverse_exact",
    "round_robin_sharded_exact",
    "fresh_process_exact",
    "restart_resume_exact",
    "all_topology_legs_exact",
    "recorded_logits_reproduce_exactly",
    "one_distinct_rollup_digest",
    "mutable_rng_absent",
    "agent4_materialized_stream_collisions_zero",
    "stream_universe_reconstruction_faithful",
    "benchmark_config_frozen",
    "benchmark_states_representative",
    "runtime_metrics_finite",
    "p95_64_worlds_recorded",
    "p95_64_worlds_le_500ms",
    "negative_controls_fire",
    "no_test_prediction_access",
    "belief_head_unchanged",
    "sampler_identity_unchanged",
    "phase9_checkpoint_unchanged",
    "no_belief_updates",
    "upstream_artifacts_unchanged",
    "full_suite_green",
)

SELF_REFERENTIAL_GATE = "full_suite_green"

SENSITIVITY_CONTROLS = (
    "private_truth_read",
    "belief_probability_perturbed",
    "sample_seed_changed",
    "mutable_global_rng",
    "provenance_corrupted",
)

pytestmark = pytest.mark.skipif(
    not ACCEPTANCE_PATH.exists(), reason="Agent 4 has not produced artifacts yet"
)


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


@pytest.fixture(scope="module")
def acceptance():
    return json.loads(ACCEPTANCE_PATH.read_text())


@pytest.fixture(scope="module")
def frozen_sets():
    return json.loads(FROZEN_SETS_PATH.read_text())


@pytest.fixture(scope="module")
def safety():
    return json.loads(SAFETY_PATH.read_text())


@pytest.fixture(scope="module")
def repro():
    return json.loads(REPRO_PATH.read_text())


@pytest.fixture(scope="module")
def stream_audit():
    return json.loads(STREAM_AUDIT_PATH.read_text())


@pytest.fixture(scope="module")
def runtime_rows():
    with open(RUNTIME_PATH, newline="") as stream:
        return list(csv.DictReader(stream))


# ---------------------------------------------------------------------------
# Gates and status
# ---------------------------------------------------------------------------


def test_the_gate_set_is_the_instruction_list_plus_the_agent_4_additions(acceptance):
    assert tuple(sorted(acceptance["completion_gates"])) == tuple(sorted(EXPECTED_GATES))
    assert acceptance["gates_total"] == len(EXPECTED_GATES)


def test_status_follows_from_the_gates(acceptance):
    gates = acceptance["completion_gates"]
    expected = "PASS" if all(gates.values()) else "FAIL"
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
    assert (
        frozen["information_safety_contract_digest"]
        == CONTRACT_DIGESTS["phase11_information_safety_v1"]
    )
    assert (
        frozen["sampler_contract_digest"]
        == CONTRACT_DIGESTS["phase11_belief_sampler_v1"]
    )
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


def test_the_sampler_bytes_are_still_the_agent_3_accepted_bytes(acceptance):
    recorded = acceptance["frozen_inputs"]["sampler_module_sha256"]
    agent3 = json.loads((DATA_DIRECTORY / "agent_03_acceptance.json").read_text())
    assert recorded == agent3["new_digests"]["implementation_sha256"]
    for module, digest in recorded.items():
        assert _file_sha256(REPOSITORY_ROOT / module) == digest, module


def test_the_agent_4_implementation_bytes_match_the_recorded_identity(acceptance):
    for module, digest in acceptance["new_digests"]["implementation_sha256"].items():
        assert _file_sha256(REPOSITORY_ROOT / module) == digest, module


def test_the_runtime_csv_bytes_match_the_recorded_digest(acceptance):
    assert acceptance["new_digests"]["runtime_csv_sha256"] == _file_sha256(RUNTIME_PATH)


# ---------------------------------------------------------------------------
# The frozen sets
# ---------------------------------------------------------------------------


def test_both_sets_were_frozen_before_any_measurement(frozen_sets):
    assert frozen_sets["frozen_before_any_measurement"] is True
    assert frozen_sets["source"]["validation_bank_digest"] == BANK_DIGESTS["validation"]


def test_the_topology_set_is_the_frozen_rule(frozen_sets):
    topology = frozen_sets["topology_request_set"]
    assert topology["request_count"] == REPRO_REQUEST_COUNT
    assert topology["distinct_public_states"] == REPRO_REQUEST_COUNT
    assert set(topology["by_stratum"]) == set(OPPONENT_STRATA)
    assert set(topology["by_stratum"].values()) == {topology["requests_per_stratum"]}
    assert set(topology["by_observer_color"]) == {"red", "blue"}
    assert topology["worlds_per_request"] == 64
    assert (
        topology["rule"] == pc.REPRODUCIBILITY_SPECIFICATION["request_set"]
    )


def test_the_benchmark_set_is_the_frozen_rule(frozen_sets):
    benchmark = frozen_sets["benchmark_state_set"]
    assert benchmark["state_count"] == BENCHMARK_STATE_COUNT
    assert benchmark["cells_total"] == BENCHMARK_CELL_COUNT
    assert benchmark["cells_short"] == []
    assert benchmark["backend"] == "cpu"
    assert benchmark["dtype"] == "float32"
    assert benchmark["torch_threads"] == 1
    assert list(benchmark["configurations"]) == list(
        pc.RUNTIME_BENCHMARK_CONFIGURATION["measured_configurations"]
    )
    assert benchmark["unresolved_pieces"]["distinct"] >= 10


def test_the_safety_pool_is_the_frozen_rule(frozen_sets):
    pool = frozen_sets["safety_candidate_pool"]
    assert pool["trials"] == SAFETY_TRIAL_COUNT
    assert pool["rule"] == pc.INFORMATION_SAFETY_ATTACK["state_pool"]
    assert pool["admitting"] > 0
    assert pool["candidates"] == pool["admitting"] + pool["non_admitting"]


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------


def test_the_attack_ran_the_frozen_trial_volume(safety):
    trials = safety["trials"]
    assert trials["executed"] >= SAFETY_TRIAL_COUNT
    assert trials["floor"] == SAFETY_TRIAL_COUNT
    assert trials["meets_floor"] is True
    assert trials["belief_forwards"] == 2 * trials["executed"]
    assert trials["sampled_worlds"] == 2 * trials["executed"]


def test_every_gate_f_counter_is_present_and_zero(safety):
    counters = safety["zero_tolerance_counters"]
    assert sorted(counters) == sorted(pc.INFORMATION_SAFETY_ZERO_COUNTERS)
    assert all(value == 0 for value in counters.values()), counters


def test_every_recorded_detail_counter_is_zero(safety):
    assert all(value == 0 for value in safety["detail_counters"].values()), safety[
        "detail_counters"
    ]
    for name in (
        "public_document_differences",
        "observation_differences",
        "public_mask_differences",
        "sampler_request_differences",
        "sampled_world_differences",
        "sampler_provenance_differences",
    ):
        assert name in safety["detail_counters"]


def test_gate_f_recomputes_from_the_recorded_counters(safety, acceptance):
    gate = pc.evaluate_gate_f(safety["zero_tolerance_counters"])
    assert gate["passed"] is True
    assert acceptance["diagnostic_gate_readings"]["gate_f"] == gate


def test_the_attack_covered_every_stratum_both_colours_and_all_buckets(safety):
    coverage = safety["coverage"]
    assert sorted(coverage["by_stratum"]) == sorted(OPPONENT_STRATA)
    assert sorted(coverage["by_observer_color"]) == ["blue", "red"]
    assert sorted(coverage["by_progress_bucket"]) == ["early", "late", "middle"]
    assert sum(coverage["by_stratum"].values()) == safety["trials"]["executed"]


def test_every_alternative_truth_actually_changed_something(safety):
    assert safety["permutation"]["changed_pieces"]["min"] >= 1
    assert safety["detail_counters"]["unchanged_alternative_truths"] == 0
    assert safety["detail_counters"]["illegal_alternative_truths"] == 0
    assert safety["detail_counters"]["inventory_changes"] == 0


def test_the_injection_controls_probed_both_boundaries_and_none_were_accepted(safety):
    injection = safety["injection_controls"]
    assert injection["states_probed"] >= 1
    assert injection["probes_total"] >= 36
    assert injection["injection_acceptances"] == 0
    assert injection["all_rejected"] is True
    for report in injection["reports"]:
        assert report["accepted_fields"] == []


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------


def test_every_frozen_leg_ran_over_the_complete_request_set(repro):
    assert sorted(repro["legs"]) == sorted(pc.REPRODUCIBILITY_TOPOLOGY_LEGS)
    for leg, detail in repro["legs"].items():
        assert detail["requests"] == REPRO_REQUEST_COUNT, leg


def test_all_legs_produced_one_rollup_digest(repro):
    assert repro["all_legs_exact"] is True
    assert len(repro["distinct_rollup_digests"]) == 1
    assert repro["distinct_rollup_digests"][0] == repro["reference_rollup_digest"]
    for leg, comparison in repro["comparison"].items():
        assert comparison["mismatches"] == 0, leg
        assert comparison["requests_compared"] == REPRO_REQUEST_COUNT


def test_the_restart_leg_really_killed_and_really_resumed(repro):
    leg = repro["legs"]["kill_resume_set_subtraction"]
    assert leg["kill_signal"] == "SIGKILL"
    assert leg["worker_returncode"] == -9
    assert leg["committed_before_kill"] > 0
    assert leg["resumed_requests"] > 0
    assert leg["recomputed_on_both_sides"] == 0
    assert leg["union_covers_set"] is True
    assert leg["committed_before_kill"] + leg["resumed_requests"] == REPRO_REQUEST_COUNT


def test_the_sharded_legs_used_the_worker_counts_they_claim(repro):
    assert repro["legs"]["workers_1"]["workers"] == 1
    assert repro["legs"]["workers_4"]["workers"] == 4
    assert repro["legs"]["workers_12"]["workers"] == 12
    assert repro["legs"]["round_robin_sharded"]["workers"] > 1
    assert (
        repro["legs"]["round_robin_sharded"]["assignment"]
        != repro["legs"]["workers_4"]["assignment"]
    )


def test_the_live_forward_reproduces_agent_2_recorded_logits(repro):
    agreement = repro["recorded_logit_agreement"]
    assert agreement["exact"] is True
    assert agreement["requests_compared"] == REPRO_REQUEST_COUNT
    assert agreement["rows_compared"] > 0
    assert agreement["row_mismatches"] == 0
    assert agreement["request_mismatches"] == 0
    assert agreement["public_state_identity_mismatches"] == 0


def test_no_mutable_rng_marker_appears_in_a_derivation(repro):
    scan = repro["purity_scan"]
    assert scan["findings"] == []
    assert scan["mutable_rng_absent"] is True
    for module in scan["scanned_modules"]:
        source = (REPOSITORY_ROOT / module).read_text()
        for marker in scan["markers"]:
            assert marker not in source, f"{module} names {marker}"


# ---------------------------------------------------------------------------
# Part C
# ---------------------------------------------------------------------------


def test_the_benchmark_ran_on_the_frozen_backend(acceptance):
    configuration = acceptance["runtime_summary"]["configuration"]
    assert configuration["backend"] == pc.RUNTIME_BENCHMARK_CONFIGURATION["backend"]
    assert configuration["dtype"] == pc.RUNTIME_BENCHMARK_CONFIGURATION["dtype"]
    assert (
        configuration["torch_threads"]
        == pc.RUNTIME_BENCHMARK_CONFIGURATION["torch_threads"]
    )
    assert acceptance["runtime_summary"]["states"] == BENCHMARK_STATE_COUNT


def test_the_gate_g_quantity_is_recorded_and_under_the_ceiling(acceptance):
    summary = acceptance["runtime_summary"]
    assert summary["ceiling_ms"] == pc.GATE_G["p95_forward_64_max_ms"] == 500.0
    assert isinstance(summary["p95_forward_64_ms"], float)
    # The gate quantity is the unrounded measurement; the per-configuration
    # table rounds to four decimals for the report, so they agree to that.
    assert summary["p95_forward_64_ms"] == pytest.approx(
        summary["summary"]["forward_plus_64_worlds"]["p95_ms"], abs=5e-5
    )
    assert summary["p95_forward_64_le_500ms"] is True
    assert summary["p95_forward_64_ms"] <= summary["ceiling_ms"]


def test_gate_g_recomputes_from_the_recorded_legs_and_p95(acceptance, repro):
    gate = pc.evaluate_gate_g(
        repro["leg_exact"], float(acceptance["runtime_summary"]["p95_forward_64_ms"])
    )
    assert gate["passed"] is True
    assert acceptance["diagnostic_gate_readings"]["gate_g"] == gate


def test_the_runtime_csv_holds_every_state_and_configuration(runtime_rows):
    configurations = {row["configuration"] for row in runtime_rows}
    assert configurations == set(
        pc.RUNTIME_BENCHMARK_CONFIGURATION["measured_configurations"]
    )
    assert len(runtime_rows) == BENCHMARK_STATE_COUNT * len(configurations)
    assert len({row["benchmark_state_id"] for row in runtime_rows}) == (
        BENCHMARK_STATE_COUNT
    )


def test_every_recorded_timing_is_finite_and_positive(runtime_rows):
    for row in runtime_rows:
        for column in ("document_ms", "forward_ms", "sampling_ms", "total_ms"):
            value = float(row[column])
            assert value == value and value not in (
                float("inf"),
                float("-inf"),
            ), f"{row['benchmark_state_id']} {column}"
            assert value >= 0.0
        assert float(row["total_ms"]) >= float(row["forward_ms"])


def test_the_recomputed_p95_matches_the_recorded_one(acceptance, runtime_rows):
    from stratego.evaluation.phase11_repro import timing_statistics

    values = [
        float(row["total_ms"])
        for row in runtime_rows
        if row["configuration"] == "forward_plus_64_worlds"
    ]
    recomputed = timing_statistics(values)["p95_ms"]
    assert recomputed == pytest.approx(
        acceptance["runtime_summary"]["p95_forward_64_ms"], rel=1e-12
    )


def test_more_worlds_cost_more_time(acceptance):
    summary = acceptance["runtime_summary"]["summary"]
    medians = [
        summary[name]["median_ms"]
        for name in pc.RUNTIME_BENCHMARK_CONFIGURATION["measured_configurations"]
    ]
    assert medians == sorted(medians)


def test_the_benchmark_spans_the_frozen_slices(acceptance, runtime_rows):
    strata = {row["opponent_stratum"] for row in runtime_rows}
    assert strata == set(OPPONENT_STRATA)
    assert {row["observer_color"] for row in runtime_rows} == {"red", "blue"}
    assert {row["progress_bucket"] for row in runtime_rows} == {
        "early",
        "middle",
        "late",
    }
    assert len({int(row["unresolved_pieces"]) for row in runtime_rows}) >= 10


# ---------------------------------------------------------------------------
# Materialized random-stream identities
# ---------------------------------------------------------------------------


def test_no_two_logical_identities_share_a_derived_seed(stream_audit):
    audit = stream_audit["collision_audit"]
    assert audit["accidental_collisions"] == 0
    assert audit["no_collisions"] is True
    assert audit["findings"] == []
    assert audit["distinct_seeds"] == audit["total_identities"]
    for name, entry in audit["per_domain"].items():
        assert entry["internal_duplicates"] == 0, name
        assert entry["distinct_seeds"] == entry["identities"], name


def test_the_per_domain_counts_sum_to_the_combined_total(stream_audit):
    audit = stream_audit["collision_audit"]
    assert sum(
        entry["identities"] for entry in audit["per_domain"].values()
    ) == audit["total_identities"]


def test_every_agent_4_materialized_domain_is_present_and_non_empty(stream_audit):
    from stratego.evaluation.phase11_streams import AGENT4_MATERIALIZED_DOMAINS

    per_domain = stream_audit["collision_audit"]["per_domain"]
    for domain in AGENT4_MATERIALIZED_DOMAINS:
        matching = [
            name
            for name in per_domain
            if name == domain or name.startswith(f"{domain}:")
        ]
        assert matching, domain
        assert all(per_domain[name]["identities"] > 0 for name in matching), domain
    assert stream_audit["agent4_materialized_identities"] > 0


def test_the_world_streams_dominate_and_are_child_balanced(stream_audit):
    per_domain = stream_audit["collision_audit"]["per_domain"]
    # One order key and one categorical draw per unresolved piece of every
    # token: the two child streams must be exactly the same size.
    assert per_domain["world_order"]["identities"] == (
        per_domain["world_categorical"]["identities"]
    )
    assert per_domain["world_sample"]["identities"] == (
        stream_audit["tokens"]["combined_distinct"]
    )


def test_the_agent_3_universe_reconstruction_matches_its_record(stream_audit):
    agent3 = json.loads((DATA_DIRECTORY / "agent_03_acceptance.json").read_text())
    recorded = agent3["audit_summary"]["world_stream_seed_counts"]
    for name, entry in stream_audit["agent3_reconstruction"].items():
        assert entry["matches"] is True, name
        assert entry["reconstructed"] == recorded[name]["count"], name
        assert entry["agent3_recorded"] == recorded[name]["count"], name


def test_the_fast_derivation_path_was_checked_against_the_public_helpers(
    stream_audit,
):
    check = stream_audit["fast_path_check"]
    assert check["exact"] is True
    assert check["mismatches"] == 0
    assert check["derivations_checked"] > 1_000


def test_intentional_reuse_is_deduplicated_not_counted_as_a_collision(stream_audit):
    tokens = stream_audit["tokens"]
    # Agent 4 reissues thousands of Agent 3's identities on purpose; they
    # appear once in the combined universe, not twice.
    assert tokens["agent4_shared_with_agent3"] > 0
    assert (
        tokens["agent4_new_to_agent4"] + tokens["agent4_shared_with_agent3"]
        == tokens["agent4_distinct"]
    )
    assert tokens["combined_distinct"] == (
        tokens["agent3_reconstructed"] + tokens["agent4_new_to_agent4"]
    )


def test_the_safety_draws_exceed_agent_1_draw_zero_enumeration(stream_audit):
    consumption = stream_audit["safety_draw_consumption"]
    by_purpose = consumption["by_purpose"]
    # Every trial consumes at least draw 0 of each purpose, and the retry
    # loop and skip walk consume more, so the attack's enumeration is a
    # strict superset of Agent 1's.
    assert by_purpose["sample_check"] == consumption["trials"]
    assert by_purpose["state_selection"] >= consumption["trials"]
    assert by_purpose["truth_permutation"] > consumption["trials"]
    assert sum(by_purpose.values()) > consumption["agent1_enumerated_draw0_only"]


def test_the_uninstantiated_domains_are_named_and_still_audited(
    stream_audit, acceptance
):
    per_domain = stream_audit["collision_audit"]["per_domain"]
    assert set(acceptance["stream_identity_summary"]["domains_not_instantiated"]) == {
        "repro_schedule",
        "benchmark",
    }
    # Their Agent 1 enumerable entries are still in the combined check.
    assert per_domain["repro_schedule:replay"]["identities"] > 0
    assert per_domain["benchmark:state_selection"]["identities"] > 0


def test_the_recomputed_attack_consumption_matches_the_recorded_run(
    stream_audit, safety
):
    assert (
        stream_audit["safety_draw_consumption"]["method_counts"]
        == safety["permutation"]["method_counts"]
    )
    assert stream_audit["safety_draw_consumption"]["trials"] == (
        safety["trials"]["executed"]
    )


def test_the_public_state_index_is_internally_consistent(stream_audit):
    index = stream_audit["public_state_index"]
    assert index["slot_set_disagreements"] == 0
    assert index["distinct_identities"] > 0
    assert index["repeated_identity_occurrences"] >= 0


def test_the_stream_audit_bytes_match_the_recorded_digest(acceptance):
    assert acceptance["new_digests"]["stream_audit_artifact_sha256"] == _file_sha256(
        STREAM_AUDIT_PATH
    )


# ---------------------------------------------------------------------------
# Part D, preservation and the seal
# ---------------------------------------------------------------------------


def test_every_sensitivity_control_fired(acceptance):
    controls = acceptance["sensitivity_controls"]
    assert sorted(controls) == sorted(SENSITIVITY_CONTROLS)
    assert all(controls.values()), controls


def test_nothing_upstream_moved(acceptance):
    preservation = acceptance["preservation"]
    assert preservation["problems"] == []
    assert preservation["checkpoint_unchanged"] is True
    assert preservation["belief_head_unchanged"] is True
    assert preservation["sampler_identity_unchanged"] is True
    assert preservation["p10d_unchanged"] is True
    assert preservation["phase7_unchanged"] is True
    assert preservation["prediction_store_unchanged"] is True
    assert preservation["optimizer_step_delta"] == 0
    assert preservation["optimizer_steps_run"] == 0
    assert preservation["optimizer_step_before"] == pc.ACCEPTED_GLOBAL_OPTIMIZER_STEP


def test_the_test_bank_is_still_sealed(acceptance):
    sealing = acceptance["test_bank_sealing"]
    assert sealing["test_bank_structural_only"] is True
    assert sealing["violations"] == []
    assert sealing["neural_inference_total"] == 0
    assert sealing["scored_prediction_total"] == 0
    assert sealing["privileged_truth_total"] == 0
    assert sealing["outcome_total"] == 0


def test_the_live_ledger_shows_the_seal_held_until_agent_7():
    """Agent 4's seal invariant in its permanent time-scoped form: the
    pre-Agent-7 ledger prefix is structural-only, and the only
    non-structural test-bank entries are Agent 7's single authorized
    sealed evaluation, which postdates this agent."""
    from stratego.evaluation import phase11_banks as banks

    entries = banks.read_ledger()
    sealing = banks.verify_test_bank_sealed(
        [entry for entry in entries if entry["agent"] <= 6]
    )
    assert sealing["test_bank_structural_only"] is True
    scored = [
        entry
        for entry in entries
        if entry["bank_version"] == "phase11_test_bank_v1"
        and not entry["structural_only"]
    ]
    assert all(entry["agent"] == 7 for entry in scored)
    agent4 = [entry for entry in entries if entry["agent"] == 4]
    assert agent4, "Agent 4 must record its bank access"
    assert any(
        entry["bank_version"] == "phase11_test_bank_v1" and entry["structural_only"]
        for entry in agent4
    )


def test_no_forbidden_operation_was_counted(acceptance):
    counters = acceptance["forbidden_operation_counters"]
    assert all(value == 0 for value in counters.values()), counters
    for name in (
        "phase11_optimizer_steps",
        "belief_head_writes",
        "belief_calibration_operations",
        "sampler_redesign_operations",
        "threshold_changes_after_evidence",
        "backend_changes_after_measurement",
        "test_bank_scored_accesses",
    ):
        assert name in counters


def test_the_handoff_names_agent_5_and_the_immutable_identities(acceptance):
    handoff = acceptance["handoff_to_agent_5"]
    assert handoff["for_agent"] == 5
    assert handoff["evaluator_identity"]["belief_head_digest"] == BELIEF_HEAD_DIGEST
    assert handoff["sampler_identity"]["sampler_version"] == "belief_sampler_v1"
    assert handoff["measured_runtime"]["backend"] == "cpu"
    assert handoff["measured_runtime"]["ceiling_ms"] == 500.0
    assert handoff["measured_runtime"]["headroom_factor"] > 1.0


def test_the_gate_a_risk_is_recorded_and_nothing_was_retuned(acceptance):
    readings = {reading["reading"] for reading in acceptance["recorded_readings"]}
    assert "gate_a_risk_acknowledged_nothing_retuned" in readings
    assert acceptance["forbidden_operation_counters"][
        "threshold_changes_after_evidence"
    ] == 0
