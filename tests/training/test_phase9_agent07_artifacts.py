"""Phase 9 Agent 7: the published canonical-run artifacts stay honest.

These tests read the reports rather than recompute them. Their job is to stop
a published artifact from drifting away from the frozen experiment it claims
to have executed — a "60 iterations" claim with 58 rows, a validation pass off
the frozen cadence, an archive member that silently reused another member's
weights, a restart record whose continuity comparison was never actually
checked, a selection that is really "the last iteration", or a "final-test
untouched" claim with a test-bank matchup in the evidence.

The artifacts exist only after `scripts/run_phase9_agent07.py` has run, so
every test skips cleanly when they are absent rather than failing a fresh
checkout.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from stratego.training.phase9_amendment import (
    AMENDED_CEILING_SECONDS as V1_CEILING_SECONDS,
    AMENDED_CONTRACT_DIGEST,
    HISTORICAL_CEILING_SECONDS,
    amendment_digest,
)
from stratego.training.phase9_amendment_v2 import (
    AMENDED_CEILING_SECONDS,
    amendment_digest as amendment_v2_digest,
    verify_chain_untouched,
)
from stratego.training.phase9_contract import (
    ACTIVE_WINDOW_RECENT_SNAPSHOTS,
    ARCHIVE_CADENCE_ITERATIONS,
    CANONICAL_BUCKET_COUNTS,
    CANONICAL_GAMES_PER_ITERATION,
    CANONICAL_ITERATIONS,
    CANONICAL_MAX_SCHEDULED_GAMES,
    CLIP_FRACTION_HARD_LIMIT,
    EPOCHS_PER_ROLLOUT,
    EXPECTED_PHASE8_CHECKPOINT_SHA256,
    HISTORICAL_ANCHOR_ID,
    KL_HARD_LIMIT,
    MINIBATCH_SIZE,
    TEST_BANK_VERSION,
    VALIDATION_BANK_VERSION,
    VALIDATION_CADENCE_ITERATIONS,
    VALIDATION_REGRESSION_GUARDS,
    VALIDATION_SCORE_WEIGHTS,
    VALIDATION_TIE_BREAK,
    active_historical_window,
    archive_snapshot_id,
    contract_digest,
    validation_score,
)

DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "reports" / "phase_9_data"
RUN = DATA_DIRECTORY / "agent_07_canonical_run.json"
CURVE = DATA_DIRECTORY / "agent_07_training_curve.csv"
ARCHIVE = DATA_DIRECTORY / "agent_07_population_archive.json"
MANIFEST = DATA_DIRECTORY / "agent_07_checkpoint_manifest.json"

ACCEPTED_TRAINER_RUNTIME_IDENTITY_DIGEST = (
    "77af4d45dd8b64e7bf87a82499bc6e54e808320cb214e9b6c58545aa6617b036"
)
EXPECTED_START_MODEL_STATE_DIGEST = (
    "f2ec4fc24d72ca170341c2a176aec32c7bf7e75d3315bb39d365835a29d9dd8c"
)

#: A test that runs *inside* the suite cannot soundly assert that the suite
#: passed. That gate is established by `--record-final-suite`, which re-runs
#: the suite with the artifacts present.
SELF_REFERENTIAL_GATE = "full_suite_green"

CADENCE_ITERATIONS = list(
    range(VALIDATION_CADENCE_ITERATIONS, CANONICAL_ITERATIONS + 1, VALIDATION_CADENCE_ITERATIONS)
)


def _load(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"{path.name} is absent; run scripts/run_phase9_agent07.py")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def run() -> dict:
    return _load(RUN)


@pytest.fixture(scope="module")
def archive() -> dict:
    return _load(ARCHIVE)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return _load(MANIFEST)


@pytest.fixture(scope="module")
def curve_rows() -> list:
    if not CURVE.exists():
        pytest.skip("agent_07_training_curve.csv is absent")
    with open(CURVE, newline="") as stream:
        return list(csv.DictReader(stream))


# ---------------------------------------------------------------------------
# Identity: what the run says it executed
# ---------------------------------------------------------------------------


def test_run_pins_the_accepted_upstream_identities(run):
    identities = run["identities"]
    assert identities["contract_digest"] == contract_digest() == AMENDED_CONTRACT_DIGEST
    assert identities["operational_amendment_digest"] == amendment_digest()
    assert identities["operational_amendment_v2_digest"] == amendment_v2_digest()
    assert (
        identities["trainer_runtime_identity_digest"]
        == ACCEPTED_TRAINER_RUNTIME_IDENTITY_DIGEST
    )
    assert identities["train_config_document_executed"].startswith("config_amended_v2")
    digests = {
        identities["train_config_document_digest_accepted_12h"],
        identities["train_config_document_digest_amended_15h"],
        identities["train_config_document_digest_amended_24h"],
    }
    assert len(digests) == 3, "the three ceiling documents must stay distinct"
    assert identities["validation_bank_digest"] != identities["test_bank_digest"]


def test_the_ceiling_history_preserves_every_earlier_value(run):
    """Two amendments to one number is where an operational budget starts to
    look negotiable; every earlier value must still be addressable."""
    history = run["identities"]["ceiling_history"]
    assert [entry["seconds"] for entry in history] == [43_200, 54_000, 86_400]
    assert history[0]["digest"] == AMENDED_CONTRACT_DIGEST
    assert history[-1]["seconds"] == AMENDED_CEILING_SECONDS
    assert verify_chain_untouched() == []
    assert run["completion_gates"]["amendment_chain_untouched"] is True


def test_the_second_amendment_records_its_own_reconciliation(run):
    stage = run["operational_amendment_v2"]
    assert stage is not None, "the v2 amendment stage must be recorded"
    assert stage["problems"] == []
    assert stage["ceiling_seconds_in_force"] == AMENDED_CEILING_SECONDS
    reconciliation = stage["reconciliation"]
    assert reconciliation["only_the_wall_clock_ceiling_changed"] is True
    assert reconciliation["fields_compared"] == 39
    assert reconciliation["unchanged_field_count"] == 38
    assert stage["runtime_identity_effect"]["unchanged"] is True
    assert stage["amends"]["in_place_edit"] is False


def test_amendment_changed_only_the_operational_ceiling(run):
    amendment = run["operational_amendment"]
    assert amendment["base_contract_untouched"] is True
    assert amendment["historical_ceiling_seconds"] == HISTORICAL_CEILING_SECONDS
    assert amendment["amended_ceiling_seconds"] == V1_CEILING_SECONDS
    assert amendment["changed_field"] == "wall_clock_ceiling_hours"
    reconciliation = amendment["reconciliation"]
    assert reconciliation["only_the_wall_clock_ceiling_changed"] is True
    assert reconciliation["fields_compared"] == 39
    assert reconciliation["unchanged_field_count"] == 38
    assert amendment["runtime_identity_effect"]["unchanged"] is True


def test_the_frozen_configuration_is_p9c_and_nothing_was_retuned(run):
    configuration = run["frozen_configuration"]
    assert run["candidate_id"] == "P9-C"
    assert configuration["learning_rate"] == pytest.approx(3e-4)
    assert configuration["initial_kl_beta"] == pytest.approx(0.005)
    assert configuration["total_iterations"] == CANONICAL_ITERATIONS
    assert configuration["epochs_per_rollout"] == EPOCHS_PER_ROLLOUT
    assert configuration["minibatch_size"] == MINIBATCH_SIZE
    assert configuration["precision"] == "float32"
    assert configuration["topology"] == {
        "workers": 6,
        "prefetch": 2,
        "record_cache_size": 48,
    }


def test_the_legacy_scope_token_was_measured_to_be_inert(run):
    audit = run["frozen_configuration"]["scope_audit"]
    assert run["frozen_configuration"]["runtime_scope_token"] == "pilot_candidate"
    assert audit["changes_training_behaviour"] is False
    assert audit["learning_fields_that_differ_by_scope"] == []
    assert not audit["verdict"].startswith("BLOCKED")


# ---------------------------------------------------------------------------
# The fresh start
# ---------------------------------------------------------------------------


def test_the_learner_started_fresh_from_the_phase8_anchor(run):
    start = run["fresh_start"]
    assert start["checkpoint_sha256"] == EXPECTED_PHASE8_CHECKPOINT_SHA256
    assert start["model_state_digest"] == EXPECTED_START_MODEL_STATE_DIGEST
    assert start["pilot_checkpoint_loaded"] is False
    assert start["global_optimizer_step"] == 0
    assert start["kl_beta"] == pytest.approx(0.005)
    assert "fresh AdamW" in start["optimizer_state"]


def test_the_frozen_checkpoint_is_not_the_phase8_anchor(run, manifest):
    difference = run["validation"]["selection"]
    assert difference  # selection exists
    comparison = manifest["differs_from_phase8_anchor"]
    assert comparison["phase8_anchor_model_state_digest"] == EXPECTED_START_MODEL_STATE_DIGEST
    assert comparison["differs"] is True
    assert comparison["phase9_model_state_digest"] != EXPECTED_START_MODEL_STATE_DIGEST


# ---------------------------------------------------------------------------
# The executed budget
# ---------------------------------------------------------------------------


def test_the_whole_frozen_experiment_ran(run):
    execution = run["execution"]
    assert execution["iterations_committed"] == CANONICAL_ITERATIONS
    assert execution["games_scheduled"] == CANONICAL_MAX_SCHEDULED_GAMES
    assert run["validation"]["passes"] == run["validation"]["passes_expected"] == 12


def test_the_run_stopped_at_the_contract_not_at_the_ceiling(run):
    execution = run["execution"]
    assert execution["wall_clock_seconds"] <= AMENDED_CEILING_SECONDS
    assert execution["ceiling_seconds"] == AMENDED_CEILING_SECONDS
    assert execution["ceiling_authority"] == "phase9_operational_amendment_v2"
    assert execution["ceiling_headroom_seconds"] > 0
    assert "contracted 60 iterations" in execution["ended_on"]


def test_every_iteration_ran_the_full_frozen_schedule(curve_rows):
    assert len(curve_rows) == CANONICAL_ITERATIONS
    assert [int(row["iteration"]) for row in curve_rows] == list(
        range(1, CANONICAL_ITERATIONS + 1)
    )
    for row in curve_rows:
        assert int(row["games"]) == CANONICAL_GAMES_PER_ITERATION
        assert int(row["optimizer_updates"]) > 0


def test_curve_totals_agree_with_the_run_artifact(run, curve_rows):
    assert sum(int(row["games"]) for row in curve_rows) == run["execution"]["games_scheduled"]
    assert (
        sum(int(row["optimizer_updates"]) for row in curve_rows)
        == run["execution"]["optimizer_updates"]
    )
    assert sum(int(row["examples"]) for row in curve_rows) == run["execution"]["examples"]


def test_curve_carries_the_report_only_diagnostics_it_promises(curve_rows):
    required = {
        "collection_seconds",
        "collection_games_per_second",
        "ppo_loss",
        "value_loss",
        "belief_loss",
        "mean_behavior_kl",
        "kl_beta_after",
        "mean_clip_fraction",
        "mean_policy_entropy",
        "advantage_threshold",
        "filter_retention",
        "train_seconds",
        "rss_mib",
    }
    assert required <= set(curve_rows[0])
    for row in curve_rows:
        assert float(row["filter_retention"]) > 0.0
        assert float(row["mean_policy_entropy"]) > 0.0


# ---------------------------------------------------------------------------
# Safety and the frozen hard stops
# ---------------------------------------------------------------------------


def test_no_hard_stop_condition_ever_fired(run):
    counters = run["hard_stop_counters"]
    for key in (
        "non_finite_losses",
        "non_finite_gradients",
        "non_finite_parameters",
        "illegal_targets",
        "data_mismatches",
        "checkpoint_errors",
        "behavior_identity_mismatches",
        "rollout_identity_mismatches",
        "kl_hard_limit_breaches",
        "clip_fraction_hard_limit_breaches",
        "observer_probe_failures",
        "illegal_policy_actions_validation",
        "inference_failures_validation",
        "test_bank_model_access",
    ):
        assert int(counters[key]) == 0, key


def test_kl_and_clip_stayed_inside_the_frozen_hard_limits(run, curve_rows):
    counters = run["hard_stop_counters"]
    assert counters["kl_hard_limit"] == KL_HARD_LIMIT
    assert counters["clip_fraction_hard_limit"] == CLIP_FRACTION_HARD_LIMIT
    assert counters["max_epoch_mean_kl"] <= KL_HARD_LIMIT
    assert counters["max_epoch_clip_fraction"] <= CLIP_FRACTION_HARD_LIMIT
    for row in curve_rows:
        for column in ("epoch1_mean_kl", "epoch2_mean_kl"):
            if row[column]:
                assert float(row[column]) <= KL_HARD_LIMIT
        for column in ("epoch1_clip_fraction", "epoch2_clip_fraction"):
            if row[column]:
                assert float(row[column]) <= CLIP_FRACTION_HARD_LIMIT


def test_every_iteration_recorded_two_optimizer_epochs(curve_rows):
    assert EPOCHS_PER_ROLLOUT == 2
    for row in curve_rows:
        assert row["epoch1_mean_kl"] and row["epoch2_mean_kl"]


# ---------------------------------------------------------------------------
# Validation and selection
# ---------------------------------------------------------------------------


def test_validation_ran_only_on_the_frozen_cadence(run):
    history = run["validation"]["history"]
    assert [record["iteration"] for record in history] == CADENCE_ITERATIONS
    assert run["validation"]["bank_version"] == VALIDATION_BANK_VERSION
    assert run["validation"]["cadence_iterations"] == VALIDATION_CADENCE_ITERATIONS


def test_every_validation_score_recomputes_from_its_own_ewrs(run):
    assert run["validation"]["score_weights"] == dict(VALIDATION_SCORE_WEIGHTS)
    for record in run["validation"]["history"]:
        rates = record["effective_win_rates"]
        assert record["selection_score"] == pytest.approx(
            validation_score(
                rates["strategic_rule_based"],
                rates["tactical_rule_based"],
                rates["phase8_anchor"],
            )
        )


def test_regression_guards_were_evaluated_at_every_pass(run):
    for record in run["validation"]["history"]:
        guards = record["guards"]
        assert guards["random_min"] == VALIDATION_REGRESSION_GUARDS["random_legal_ewr_min"]
        assert guards["basic_min"] == VALIDATION_REGRESSION_GUARDS["basic_heuristic_ewr_min"]
        assert guards["random_pass"] == (guards["random_ewr"] >= guards["random_min"])
        assert guards["basic_pass"] == (guards["basic_ewr"] >= guards["basic_min"])


def test_the_selected_checkpoint_has_the_strictly_highest_score(run):
    selection = run["validation"]["selection"]
    scores = {int(key): value for key, value in selection["scores_by_iteration"].items()}
    assert set(scores) == set(CADENCE_ITERATIONS)
    best = selection["selected_iteration"]
    assert scores[best] == max(scores.values())
    assert selection["selection_score"] == scores[best]
    assert selection["tie_break"] == list(VALIDATION_TIE_BREAK)
    if selection["unique_on_score"]:
        assert [
            iteration for iteration, score in scores.items() if score == scores[best]
        ] == [best]


def test_selection_used_the_validation_bank_and_never_a_final_test_metric(run):
    selection = run["validation"]["selection"]
    assert "phase9_validation_bank_v1" in selection["selected_by"]
    assert run["final_test_bank"]["model_access_by_agent_7"] == 0
    assert run["final_test_bank"]["constructed_by_agent_7"] is False
    assert run["final_test_bank"]["version"] == TEST_BANK_VERSION
    assert run["hard_stop_counters"]["test_bank_model_access"] == 0


def test_the_selection_records_whether_the_last_iteration_won(run):
    """The final iteration is not automatically selected; the artifact has to
    say which it was rather than leave a reader to assume."""
    selection = run["validation"]["selection"]
    assert isinstance(selection["final_iteration_is_best"], bool)
    assert selection["final_iteration_is_best"] == (
        selection["selected_iteration"] == CANONICAL_ITERATIONS
    )


def test_the_frozen_checkpoint_reload_reproduced_its_metrics(run, manifest):
    reproduction = manifest["reload_reproduction"]
    assert reproduction["passed"] is True
    assert reproduction["selection_score"]["equal"] is True
    for metric, entry in reproduction["effective_win_rates"].items():
        assert entry["equal"] is True, metric
    assert reproduction["safety_clean"] is True


def test_the_frozen_checkpoint_carries_a_written_sha(manifest):
    frozen = manifest["frozen_phase9_checkpoint"]
    assert frozen["path"] == "checkpoints/phase9/selfplay_c1_v1.pt"
    assert len(frozen["sha256"]) == 64
    assert frozen["bytes_identical_to_source"] is True
    assert len(frozen["model_state_digest"]) == 64


# ---------------------------------------------------------------------------
# The historical league
# ---------------------------------------------------------------------------


def test_the_archive_is_exactly_the_frozen_cadence(archive):
    expected = [
        archive_snapshot_id(iteration)
        for iteration in range(
            ARCHIVE_CADENCE_ITERATIONS, CANONICAL_ITERATIONS + 1, ARCHIVE_CADENCE_ITERATIONS
        )
    ]
    assert archive["expected_members"] == expected
    assert [member["local_identity"] for member in archive["members"]] == expected
    assert archive["member_count"] == 12
    assert archive["archive_schedule_exact"] is True
    assert archive["cadence_iterations"] == ARCHIVE_CADENCE_ITERATIONS


def test_every_archive_member_is_a_distinct_immutable_object(archive):
    shas = [member["checkpoint_sha256"] for member in archive["members"]]
    digests = [member["state_dict_digest"] for member in archive["members"]]
    assert len(set(shas)) == len(shas)
    assert len(set(digests)) == len(digests)
    for member in archive["members"]:
        # Logical identity and checkpoint SHA are different objects.
        assert member["qualified_identity"] == f"canonical|{member['local_identity']}"
        assert member["local_identity"] != member["checkpoint_sha256"]
        assert member["created_after_iteration"] == int(member["local_identity"][1:])


def test_the_anchor_is_the_phase8_checkpoint_and_never_an_archive_member(archive):
    assert archive["anchor"]["identity"] == HISTORICAL_ANCHOR_ID
    assert archive["anchor"]["checkpoint_sha256"] == EXPECTED_PHASE8_CHECKPOINT_SHA256
    assert HISTORICAL_ANCHOR_ID not in [
        member["local_identity"] for member in archive["members"]
    ]


def test_the_active_window_follows_the_frozen_rule_at_every_iteration(archive):
    for iteration in range(1, CANONICAL_ITERATIONS + 1):
        recorded = archive["active_window_by_iteration"][str(iteration)]
        assert recorded == list(active_historical_window(iteration))
        assert recorded[0] == HISTORICAL_ANCHOR_ID
        assert len(recorded) <= ACTIVE_WINDOW_RECENT_SNAPSHOTS + 1
    assert archive["outcome_prioritized_sampling"] is False


def test_historical_actions_were_verified_against_the_acting_archive(archive):
    verifications = archive["historical_action_verification"]
    assert len(verifications) == CANONICAL_ITERATIONS
    for entry in verifications:
        assert entry["all_verified"] is True
        assert entry["active_window"] == sorted(active_historical_window(entry["iteration"]))
        for identity, summary in entry["verified"].items():
            assert summary["failed"] == 0
            assert summary["decisions"] > 0
            assert identity in entry["active_window"]


# ---------------------------------------------------------------------------
# The restart exercise
# ---------------------------------------------------------------------------


def test_at_least_one_genuine_process_restart_was_exercised(run):
    restarts = run["restart_exercise"]
    assert restarts, "the canonical run must include a genuine process restart"
    for record in restarts:
        assert record["kind"] == "scheduled_mid_epoch_process_restart"
        assert record["exited_pid"] != record["resumed_pid"]
        assert record["updates_before_exit"] > 0
        assert record["iteration"] > ARCHIVE_CADENCE_ITERATIONS // 2


def test_every_boundary_resume_also_proved_continuity(run):
    """Resumes between iterations carry no live pre-exit process, so the
    checkpoint itself is the authority: everything it recorded must be what
    the resumed trainer holds, weights included."""
    for record in run["boundary_resumes"]:
        assert record["kind"] == "committed_iteration_boundary_resume"
        assert record["passed"] is True
        assert record["model_state_digest_bitwise_equal"] is True
        failing = [key for key, ok in record["fields_equal"].items() if not ok]
        assert failing == []


def test_each_restart_proved_exact_logical_continuity(run):
    for record in run["restart_exercise"]:
        comparison = record["comparison"]
        assert comparison["passed"] is True
        assert comparison["criterion_id"] == "phase9_backend_aware_resume_equivalence_v1"
        checks = comparison["checks"]
        for name in (
            "logical_state_equal",
            "model_state_digest_bitwise_equal",
            "next_batch_identical",
            "probe_within_backend_tolerance",
            "active_history_equal",
            "validation_history_equal",
            "best_validation_equal",
            "sealed_rollout_identity_equal",
            "behavior_snapshot_equal",
        ):
            assert checks[name] is True, name


def test_a_restart_resumed_mid_epoch_rather_than_at_a_convenient_boundary(run):
    """A restart that lands on an iteration boundary would prove far less: the
    minibatch cursor and the KL controller's partial-epoch state are exactly
    what an exact logical resume has to carry."""
    mid_epoch = []
    for record in run["restart_exercise"]:
        cursor = record["before"]["state_summary"]["minibatch_cursor"]
        if cursor["minibatch_index"] > 0:
            mid_epoch.append(record["iteration"])
    assert mid_epoch, "no restart happened inside an epoch"


def test_a_restart_carried_a_non_empty_validation_and_archive_state(run):
    """The later restart has to cross a boundary where a best-validation record
    and real archive members already exist, or that continuity is untested."""
    loaded = [
        record
        for record in run["restart_exercise"]
        if record["before"]["validation_history"]
        and len(record["before"]["active_history"]["identities"]) > 1
    ]
    assert loaded, "no restart crossed a boundary with validation and archive state"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_production_rollout_storage_was_verified_on_every_process_start(run):
    checks = run["storage_checks"]
    assert len(checks) >= run["execution"]["process_launches"]
    for check in checks:
        assert check["on_external_volume"] is True
        assert check["mount_point"].startswith("/Volumes/")
    assert run["storage"]["resolver"].endswith("default_rollout_root()")
    assert run["storage"]["problems"] == []


# ---------------------------------------------------------------------------
# Completion gates
# ---------------------------------------------------------------------------


def test_completion_gates_cover_the_assignment(run):
    required = {
        "agents1_6_pass",
        "corpus_resolver_verified",
        "corpus_digests_match",
        "fresh_phase8_anchor_start",
        "pilot_checkpoint_loaded_no",
        "exact_frozen_config_used",
        "iterations_completed_60",
        "games_scheduled_122880",
        "rollout_identity_errors_zero",
        "illegal_actions_zero",
        "nonfinite_zero",
        "target_mismatches_zero",
        "observer_leaks_zero",
        "kl_hard_limit_never_exceeded",
        "clip_fraction_hard_limit_never_exceeded",
        "restart_path_exercised",
        "archive_schedule_exact",
        "validation_only_checkpoint_selection",
        "best_checkpoint_reload_reproduces",
        "final_checkpoint_sha_written",
        "final_test_model_access_zero",
        "full_suite_green",
    }
    assert required <= set(run["completion_gates"])


def test_every_gate_except_the_self_referential_one_is_true(run):
    failing = [
        name
        for name, value in run["completion_gates"].items()
        if not value and name != SELF_REFERENTIAL_GATE
    ]
    assert failing == []


def test_status_is_pass_only_when_every_gate_is_true(run):
    gates = run["completion_gates"]
    assert run["gates_total"] == len(gates)
    assert run["gates_true"] == sum(1 for value in gates.values() if value)
    if run["status"] == "PASS":
        assert all(gates.values())


# ---------------------------------------------------------------------------
# The handoff
# ---------------------------------------------------------------------------


def test_handoff_gives_agent_8_everything_it_needs_and_no_training(run, manifest):
    handoff = run["handoff_to_agent_8"]
    assert handoff["frozen_checkpoint_path"] == "checkpoints/phase9/selfplay_c1_v1.pt"
    assert handoff["frozen_checkpoint_sha256"] == manifest["frozen_phase9_checkpoint"]["sha256"]
    assert handoff["phase8_anchor_checkpoint_sha256"] == EXPECTED_PHASE8_CHECKPOINT_SHA256
    assert handoff["selected_iteration"] in CADENCE_ITERATIONS
    assert len(handoff["final_test_bank_digest"]) == 64
    assert handoff["agent_8_performs_no_training"] is True
