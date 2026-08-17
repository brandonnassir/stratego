"""Phase 9 Agent 3: the published artifacts say what the run actually did.

These tests read the reports rather than recompute them. Their job is to stop
a published artifact from drifting away from the frozen contract it claims to
satisfy — an acceptance file that recorded a passing gate it never measured,
a soak that quietly shrank below its floor, or a reproduction audit whose
"zero mismatches" came from auditing nothing.

The artifacts are only present after `scripts/run_phase9_agent03.py` has run,
so every test skips cleanly when they are absent rather than failing a fresh
checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stratego.training import phase9_collector as pc
from stratego.training import phase9_contract as contract
from stratego.training import phase9_rollout_store as store
from stratego.training import phase9_seed as seeds

DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "reports" / "phase_9_data"
ACCEPTANCE = DATA_DIRECTORY / "agent_03_acceptance.json"
STORE_REPORT = DATA_DIRECTORY / "agent_03_rollout_store.json"
SOAK_REPORT = DATA_DIRECTORY / "agent_03_collection_soak.json"
REPRODUCTION_REPORT = DATA_DIRECTORY / "agent_03_behavior_reproduction.json"

ANCHOR_SHA256 = "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca"
ACCEPTED_CONTRACT_DIGEST = (
    "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"
)

#: The soak floor the assignment fixes. A run below this is not a soak.
MINIMUM_SOAK_GAMES = 8192
MINIMUM_REPRODUCED_DECISIONS = 100_000


def _load(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"{path.name} has not been produced yet")
    return json.loads(path.read_text())


@pytest.fixture
def acceptance():
    return _load(ACCEPTANCE)


@pytest.fixture
def soak():
    return _load(SOAK_REPORT)


@pytest.fixture
def reproduction():
    return _load(REPRODUCTION_REPORT)


@pytest.fixture
def store_report():
    return _load(STORE_REPORT)


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


#: `full_suite_green` is deliberately excluded from the gate assertions below.
#: A test that runs *inside* the suite cannot soundly assert that the suite
#: passed: the assertion is evaluated before its own run finishes, so it can
#: only ever report on a previous run, and demanding green from the run that
#: records it has no fixed point. The suite's job is to verify what the
#: artifacts claim about the collection; whether the suite itself passed is
#: established by running it and reading the result, which is what
#: `--record-final-suite` writes into `tests_after`.
SELF_REFERENTIAL_GATE = "full_suite_green"


def test_acceptance_reports_every_measured_gate_true(acceptance):
    failed = [
        name
        for name, value in acceptance["completion_gates"].items()
        if not value and name != SELF_REFERENTIAL_GATE
    ]
    assert not failed, f"gates reported false: {failed}"
    assert not acceptance["problems"]
    assert acceptance["gates_total"] == len(acceptance["completion_gates"])


def test_the_recorded_suite_ran_with_the_artifacts_in_place(acceptance):
    """The recorded result must come from a pass that could see the artifacts.

    The harness runs the suite before it writes them, so in that pass every
    test in this file skips and the recorded green would mean nothing. Only
    the `--record-final-suite` pass sets this flag.
    """
    assert acceptance["tests_after"].get("covers_agent_03_artifact_tests"), (
        "the recorded suite ran before the artifacts existed, so it never "
        "exercised the artifact tests"
    )


def test_every_assigned_completion_gate_is_present(acceptance):
    """The gate list the assignment requires, checked name by name."""
    required = {
        "agents1_2_pass",
        "corpus_resolver_verified",
        "corpus_digests_match",
        "behavior_snapshot_immutable",
        "one_behavior_identity_per_iteration",
        "neural_actions_legal",
        "behavior_storage_matches_contract",
        "behavior_reproduction_ge_100k",
        "behavior_reproduction_mismatches_zero",
        "rollout_commit_protocol_pass",
        "crash_resume_converges",
        "orphan_records_zero",
        "duplicate_game_ids_zero",
        "unscheduled_games_zero",
        "replay_illegal_actions_zero",
        "setup_provenance_mismatches_zero",
        "observer_input_leaks_zero",
        "collection_soak_ge_8192_games",
        "no_rl_optimizer_steps",
        "full_suite_green",
    }
    assert required <= set(acceptance["completion_gates"])


def test_acceptance_pins_the_accepted_upstream_identities(acceptance):
    prerequisites = acceptance["prerequisites"]
    assert prerequisites["contract_digest"] == ACCEPTED_CONTRACT_DIGEST
    assert prerequisites["phase8_checkpoint_sha256"] == ANCHOR_SHA256
    assert prerequisites["c1_parameters"] == 863_959
    assert prerequisites["corpus"]["identity_matches"]
    assert prerequisites["corpus"]["observed_identity"] == {
        "corpus_version": contract.EXPECTED_CORPUS_VERSION,
        "content_digest": contract.EXPECTED_CORPUS_CONTENT_DIGEST,
        "metadata_digest": contract.EXPECTED_CORPUS_METADATA_DIGEST,
        "commit_index_digest": contract.EXPECTED_CORPUS_COMMIT_INDEX_DIGEST,
    }


def test_no_collection_module_hard_codes_an_absolute_data_path(acceptance):
    assert acceptance["prerequisites"]["corpus"]["modules_hard_coding_absolute_paths"] == []


def test_the_rollout_root_was_proved_to_be_a_real_external_volume(acceptance):
    """A directory of the same name on the boot disk must not pass for one."""
    storage = acceptance["storage"]
    assert storage["resolved_root_matches_expected"]
    assert storage["external_volume"]["is_external"]
    assert storage["external_volume"]["internal"] is False
    assert storage["external_volume"]["mounted"]
    assert storage["is_mount_point"]
    assert storage["distinct_from_boot_filesystem"]
    assert not storage["external_volume"]["volume_read_only"]
    assert storage["capacity_evaluation"]["recommended"]
    assert not storage["problems"]


# ---------------------------------------------------------------------------
# The soak
# ---------------------------------------------------------------------------


def test_the_soak_met_its_floor_with_every_bucket_represented(soak):
    totals = soak["totals"]
    assert totals["games_committed"] >= MINIMUM_SOAK_GAMES
    assert sorted(totals["buckets_represented"]) == [
        "current",
        "historical",
        "rule",
        "stress",
    ]
    assert totals["total_decisions"] > 0
    assert totals["neural_decisions"] > 0
    assert totals["learner_decisions"] > 0


def test_the_soak_collected_exactly_the_scheduled_games(soak):
    """No additions, no replacements: the schedule decides, not the collector."""
    for iteration in soak["iterations"]:
        namespace = iteration["namespace"]
        expected = contract.games_per_iteration(namespace)
        assert iteration["storage"]["committed_games"] == expected
        assert iteration["seal"]["scheduled_games"] == expected
        assert iteration["seal"]["missing_games"] == 0
        assert iteration["seal"]["unscheduled_games"] == 0
        assert iteration["seal"]["duplicate_game_ids"] == 0


def test_every_soak_iteration_used_one_behavior_snapshot(soak):
    for iteration in soak["iterations"]:
        assert iteration["seal"]["behavior_snapshot_identities"] == ["B001"]
        assert iteration["seal"]["behavior_checkpoint_digests"] == [ANCHOR_SHA256]
        assert iteration["behavior_checkpoint_sha256"] == ANCHOR_SHA256


def test_the_soak_needed_no_unresolved_future_archive_weights(soak):
    """Iteration 1 everywhere: the active window is the real Phase 8 anchor.

    This is why the soak schedule was chosen. An iteration past the archive
    cadence would have scheduled `H005` games with no checkpoint behind them,
    and the only way to collect those would have been to invent a digest.
    """
    for iteration in soak["iterations"]:
        assert iteration["iteration"] == 1
        assert iteration["active_history"]["identities"] == [contract.HISTORICAL_ANCHOR_ID]
        assert iteration["active_history"]["checkpoint_digests"] == {
            contract.HISTORICAL_ANCHOR_ID: ANCHOR_SHA256
        }


def test_all_seven_namespaces_were_collected(soak):
    assert {iteration["namespace"] for iteration in soak["iterations"]} == set(
        seeds.RUN_NAMESPACES
    )


def test_every_soak_iteration_sealed(soak):
    for iteration in soak["iterations"]:
        assert iteration["sealed"], iteration["seal"]["problems"]
        assert len(iteration["sealed_rollout_digest"]) == 64


def test_storage_density_is_measured_not_inherited(soak):
    """Agent 2's estimate is planning evidence; this is the measurement."""
    projection = soak["storage_projection"]
    assert projection["measured_bytes_per_game"] > 0
    assert projection["measured_bytes_per_decision"] > 0
    assert 0 < projection["measured_compression_ratio"] < 1
    assert projection["phase9_total_games"] == 60 * 2048 + 6 * 8 * 1024
    assert projection["agent_2_planning_estimate_gib"] > 0
    # Both figures are reported side by side; the measured one is the authority.
    assert projection["measured_over_planning_ratio"] > 0


def test_the_soak_reports_the_required_throughput_instruments(soak):
    totals = soak["totals"]
    for field in (
        "positions_per_second",
        "games_per_second",
        "decisions_per_second",
        "peak_rss_bytes",
        "mean_game_length_plies",
    ):
        assert totals[field] > 0, field
    storage = totals["storage"]
    for field in (
        "bytes_per_game",
        "bytes_per_decision",
        "bytes_per_position",
        "compression_ratio",
        "storage_per_hour_bytes",
    ):
        assert storage[field] > 0, field
    cpu = totals["cpu"]
    assert cpu["cpu_seconds"] > 0
    assert cpu["cores_busy"] > 0
    assert 0 < cpu["machine_utilization_fraction"] <= 1.0
    # MPS has no utilization counter; the report must say so rather than
    # invent a number.
    assert "no MPS device-utilization counter" in cpu["mps_utilization_note"]
    if totals["inference_device"] == "mps":
        assert totals["peak_mps_bytes"] > 0


# ---------------------------------------------------------------------------
# The reproduction audit
# ---------------------------------------------------------------------------


def test_the_reproduction_audit_met_its_floor_with_zero_mismatches(reproduction):
    learner = reproduction["learner"]
    assert learner["decisions"] >= MINIMUM_REPRODUCED_DECISIONS
    assert learner["mismatches"] == 0
    assert reproduction["legal_set_mismatches"] == 0
    assert (
        learner["max_abs_probability_difference"]
        <= contract.BEHAVIOR_PROBABILITY_ABS_TOLERANCE
    )
    assert reproduction["tolerance"] == contract.BEHAVIOR_PROBABILITY_ABS_TOLERANCE


def test_historical_decisions_were_verified_against_their_own_checkpoint(reproduction):
    """Per-side identity, measured: not everything checked against one digest."""
    historical = reproduction["historical"]
    assert historical["decisions"] > 0
    assert historical["mismatches"] == 0
    assert (
        historical["max_abs_probability_difference"]
        <= contract.BEHAVIOR_PROBABILITY_ABS_TOLERANCE
    )


def test_the_audit_is_capable_of_failing(reproduction):
    """An audit that cannot fail is not evidence. The negative control."""
    control = reproduction["cross_checkpoint_control"]
    assert control["control_holds"]
    assert control["decisions"] > 0
    assert control["verified_against_wrong_checkpoint"] == 0
    assert control["mismatches"] == control["decisions"]
    assert (
        control["max_abs_probability_difference"]
        > contract.BEHAVIOR_PROBABILITY_ABS_TOLERANCE
    )


def test_the_audit_covers_every_required_field(reproduction):
    assert set(reproduction["audited_fields"]) == {
        "acting player",
        "observation digest",
        "legal set",
        "action frame",
        "behavior distribution",
        "sampled action legality",
        "WDL output",
        "behavior snapshot identity",
    }
    assert reproduction["distinct_observations"] > 1000


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def test_the_store_audit_found_no_integrity_failures(store_report):
    assert store_report["duplicate_game_ids"] == 0
    assert store_report["unscheduled_games"] == 0
    assert store_report["orphan_records"] == 0
    assert store_report["replay_illegal_actions"] == 0
    assert store_report["setup_provenance_mismatches"] == 0
    assert store_report["games_replayed_legally"] > 0
    assert store_report["sealed_iterations"] == len(seeds.RUN_NAMESPACES)


def test_the_store_publishes_the_frozen_schema(store_report):
    assert store_report["store_version"] == contract.PHASE9_ROLLOUT_STORE_VERSION
    assert store_report["states"] == list(contract.ROLLOUT_STATES)
    assert store_report["metadata_fields"] == list(store.METADATA_FIELDS)
    assert store_report["collector_version"] == pc.PHASE9_COLLECTOR_VERSION


def test_every_scheduled_game_of_every_namespace_is_committed(store_report):
    assert store_report["distinct_game_ids"] == MINIMUM_SOAK_GAMES
    for iteration in store_report["iterations"]:
        assert iteration["state"] == "SEALED"
        assert iteration["missing_games"] == 0
        assert iteration["committed_games"] == iteration["scheduled_games"]


def test_the_recorded_collection_conditions_are_pinned_for_resume(store_report):
    """Device and batch shape are recorded, because a resume may not change them."""
    for iteration in store_report["iterations"]:
        assert iteration["inference_device"] in ("cpu", "mps")
        assert iteration["inference_batch_shape"] >= 1


# ---------------------------------------------------------------------------
# Crash safety and the boundary
# ---------------------------------------------------------------------------


def test_crash_resume_converged_on_the_production_store(acceptance):
    crash = acceptance["crash_resume"]
    assert crash["digests_converge"]
    assert crash["payload_bytes_identical"]
    assert crash["committed_before_crash"] > 0
    assert crash["worker_topology_changed"]
    assert set(crash["crash_stages_covered_by_suite"]) == set(store.CRASH_STAGES)


def test_the_observer_boundary_audit_ran_and_can_detect_a_planted_leak(acceptance):
    observer = acceptance["observer_safety"]
    assert observer["probes"] > 0
    assert observer["failures"] == 0
    assert observer["mean_hidden_opponent_pieces"] > 0
    control = observer["positive_control"]
    assert control["control_holds"]
    assert control["planted_leak_detected"]
    assert control["planted_leak_entries"] > 0
    assert control["frozen_builder_passes_the_same_check"]


def test_the_handoff_warns_agent_4_about_per_side_identity(acceptance):
    handoff = acceptance["handoff_to_agent_4"]
    assert "opponent_checkpoint_sha256" in handoff["per_side_identity_warning"]
    for key in (
        "sealed_rollout_reader",
        "random_access_reconstruction",
        "behavior_quantity_access",
        "behavior_wdl_outputs",
        "learner_control_masks",
        "privileged_target_only_state",
        "rollout_digests",
        "crash_safe_iteration_state",
        "independent_reproduction",
    ):
        assert handoff[key]


def test_the_run_took_no_optimizer_steps(acceptance):
    """Structurally, not by assertion: an infrastructure soak cannot prove
    "we did not train" from its own results, so the claim is checked against
    the collection modules and the live snapshot instead."""
    assert acceptance["completion_gates"]["no_rl_optimizer_steps"]
    audit = acceptance["no_optimizer_audit"]
    assert audit["no_optimizer_steps"]
    assert audit["symbol_findings"] == []
    assert audit["trainable_parameters_after_collection"] == 0
    assert audit["weights_unchanged"]
    assert audit["probe_game_decisions"] > 0
    assert set(audit["modules_scanned"]) == {
        "stratego/training/phase9_behavior.py",
        "stratego/training/phase9_collector.py",
        "stratego/training/phase9_rollout_store.py",
    }
    assert acceptance["soak_totals"]["games_committed"] >= MINIMUM_SOAK_GAMES
