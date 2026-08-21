"""Phase 13 — Agent 4: the launch package, the status surface and the supervisor.

Task: `instructions/phase_13_final_training_integration/04_AGENT_4_IMMUTABLE_LAUNCH_PACKAGE.md`.

Four things are under test, and no training value is among them:

* the **code binding** — the launch manifest binds the revision, the content of
  the Phase 14 import closure and the presence of the accepted Agent 3
  worker-pool repair, because neither frozen digest can see that repair;
* the **authoritative committed-game total** — read from the rollout store's
  iteration manifests, with the process-local counter kept beside it as the
  diagnostic it is;
* **loader-pool health** — configured, live, rebuilds, and when and why the
  last rebuild happened;
* the **launch supervisor** — what it records, when it restarts, and the five
  states in which it must not.

Targeted tests only. The 90-minute rehearsal is not rerun: Agent 3 already
demonstrated the underlying recovery path at the frozen production population.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from stratego.training import phase14_contract as contract
from stratego.training.phase14_launch import (
    OperationalTopology,
    code_closure,
    Phase14LaunchError,
    assert_frozen_topology,
    assert_launch_code,
    build_launch_manifest,
    clear_emergency_stop,
    code_binding,
    emergency_stop_path,
    emergency_stop_state,
    final_training_config_document,
    integrity_failure_state,
    load_launch_manifest,
    record_integrity_failure,
    recovery_semantics,
    request_emergency_stop,
    rng_semantics,
    worker_repair_evidence,
)
from stratego.training.phase14_status import (
    committed_game_census,
    games_report,
    loader_health,
)
from stratego.training.phase14_storage import Phase14Storage
from stratego.training.phase14_supervisor import (
    ACTION_FINALIZE_ONLY,
    ACTION_RESTART,
    ACTION_STOP,
    DEFAULT_MAX_CONSECUTIVE_RESTARTS,
    Phase14SupervisorError,
    RestartConditions,
    SupervisorLog,
    SupervisorPolicy,
    assert_all_candidates_evaluated,
    deadline_state,
    exit_description,
    read_conditions,
    restart_decision,
    run_manifest_state,
    unevaluated_candidates,
)

START = "2026-09-01T00:00:00.000Z"


# ---------------------------------------------------------------------------
# 1. The frozen digests still match, and still cannot see the repair
# ---------------------------------------------------------------------------


def test_the_two_frozen_digests_are_unchanged_by_agent_4():
    """Agent 4 adds monitoring and a supervisor. It changes no training value."""
    from stratego.training.phase14_config import integrated_config_digest

    assert contract.contract_digest() == (
        "62ce6d4e04ffd25755717ef290f7486f2616927ddada59d8ea9fb05565c052b9"
    )
    assert integrated_config_digest() == (
        "9c2a38e4335762997adbb33731dc619615fff713c2c60840c7c8d74a2f29da5e"
    )


def test_the_worker_repair_is_proved_by_assertion_not_by_digest():
    """The reason section 1 asks for a code binding at all."""
    evidence = worker_repair_evidence()
    assert evidence["installed"] is True
    assert evidence["max_loader_pool_rebuilds"] == 16
    assert "BrokenExecutor" in evidence["recoverable_errors"]
    assert all(evidence["checks"].values()), evidence["checks"]


def test_the_frozen_metric_list_was_not_touched():
    """It lives in the contract document; moving it would move the digest."""
    from stratego.training.phase14_telemetry import (
        EXTENDED_METRIC_PATHS,
        METRIC_PATHS,
    )

    assert set(contract.FROZEN_METRIC_LIST) == set(METRIC_PATHS)
    assert not set(EXTENDED_METRIC_PATHS) & set(METRIC_PATHS)


# ---------------------------------------------------------------------------
# 2. Binding the code revision
# ---------------------------------------------------------------------------


def test_the_code_binding_covers_the_files_that_carry_the_repair():
    binding = code_binding()
    assert binding["closure_files"] > 100
    assert "stratego/training/phase14_trainer.py" in binding["file_sha256"]
    assert "stratego/training/phase14_runner.py" in binding["file_sha256"]
    assert binding["search_excluded"] is True


@pytest.fixture(scope="module")
def fresh_manifest():
    """A manifest built against the tree these tests are running on.

    The *on-disk* launch package deliberately goes stale the moment tracked
    code moves — that is the whole point of binding a revision — so the
    mechanism is tested against a freshly built manifest, and staleness is
    caught where it belongs: by the pre-launch check and by the launcher.
    """
    return build_launch_manifest()


def test_a_freshly_built_manifest_verifies_against_the_live_tree(fresh_manifest):
    report = assert_launch_code(fresh_manifest)
    assert report["verified"] is True
    assert report["worker_pool_repair_installed"] is True


def test_the_on_disk_launch_package_binds_a_revision_and_the_repair():
    manifest = load_launch_manifest()
    revision = manifest["code"]["git"]["revision"]
    assert isinstance(revision, str) and len(revision) == 40
    assert manifest["code"]["worker_pool_repair"]["installed"] is True
    assert manifest["code"]["closure_files"] > 100
    assert len(manifest["launch_manifest_digest"]) == 64


def test_launch_refuses_a_different_code_revision(fresh_manifest):
    altered = json.loads(json.dumps(fresh_manifest))
    altered["code"]["git"]["revision"] = "0" * 40
    with pytest.raises(Phase14LaunchError, match="is not the bound"):
        assert_launch_code(altered)


def test_launch_refuses_an_edited_tracked_file(fresh_manifest):
    """Same commit, different bytes: the digest map is what notices."""
    altered = json.loads(json.dumps(fresh_manifest))
    altered["code"]["file_sha256"]["stratego/training/phase14_trainer.py"] = "0" * 64
    altered["code"]["code_digest"] = "1" * 64
    with pytest.raises(Phase14LaunchError, match="code closure does not match"):
        assert_launch_code(altered)


def test_launch_refuses_a_different_working_tree_state(fresh_manifest):
    altered = json.loads(json.dumps(fresh_manifest))
    altered["code"]["git"]["dirty_tracked_files"] = ["some/other/file.py"]
    with pytest.raises(Phase14LaunchError, match="working-tree state differs"):
        assert_launch_code(altered)


def test_launch_refuses_a_replaced_launch_script(fresh_manifest):
    altered = json.loads(json.dumps(fresh_manifest))
    altered["scripts"]["launch"]["sha256"] = "0" * 64
    with pytest.raises(Phase14LaunchError, match="launch script"):
        assert_launch_code(altered)


def test_the_manifest_binds_every_operator_script():
    manifest = load_launch_manifest()
    for name in ("launch", "resume", "status", "emergency_stop", "candidate_evaluator"):
        entry = manifest["scripts"][name]
        assert (contract.repository_root() / entry["path"]).exists()
        assert len(entry["sha256"]) == 64


# ---------------------------------------------------------------------------
# 3. The authoritative committed-game total
# ---------------------------------------------------------------------------


def _fake_iteration(root, iteration: int, *, state: str, committed: int, manifest=True):
    directory = Path(root) / contract.PHASE14_NAMESPACE / f"iteration_{iteration:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "state.json").write_text(
        json.dumps({"state": state, "iteration": iteration, "namespace": "phase14"})
    )
    if manifest:
        (directory / "manifest.json").write_text(
            json.dumps({"committed_games": committed, "scheduled_games": committed})
        )
    return directory


def test_the_census_reproduces_the_rehearsal_disagreement(tmp_path):
    """Agent 3's exact observation: 8,192 on disk, 4,096 in the counter."""
    for iteration in (1, 2, 3, 4):
        _fake_iteration(tmp_path, iteration, state="COMMITTED", committed=2048)
    report = games_report(tmp_path, 4096)
    assert report["committed_games"] == 8192
    assert report["committed_games_authoritative"] is True
    assert report["process_counter_games"] == 4096
    assert report["process_counter_is_diagnostic"] is True
    assert report["process_counter_shortfall"] == 4096


def test_the_census_counts_an_unsealed_iteration_separately(tmp_path):
    _fake_iteration(tmp_path, 1, state="COMMITTED", committed=2048)
    collecting = _fake_iteration(tmp_path, 2, state="COLLECTING", committed=0, manifest=False)
    journal = collecting / "journal"
    journal.mkdir()
    journal.joinpath("w00.commit.jsonl").write_text(
        "".join(
            json.dumps({"phase9_game_id": f"g{index:04d}"}) + "\n" for index in range(17)
        )
    )
    census = committed_game_census(tmp_path)
    assert census["committed_games"] == 2048
    assert census["in_flight_games"] == 17
    assert census["sealed_iterations"] == 1


def test_the_census_does_not_double_count_a_duplicate_commit(tmp_path):
    collecting = _fake_iteration(tmp_path, 1, state="COLLECTING", committed=0, manifest=False)
    journal = collecting / "journal"
    journal.mkdir()
    journal.joinpath("w00.commit.jsonl").write_text(
        json.dumps({"phase9_game_id": "g0001"}) + "\n" + json.dumps({"phase9_game_id": "g0001"}) + "\n"
    )
    assert committed_game_census(tmp_path)["in_flight_games"] == 1


def test_the_census_is_empty_and_honest_before_the_run_starts(tmp_path):
    census = committed_game_census(tmp_path / "nothing_here")
    assert census["committed_games"] == 0
    assert census["iterations"] == []
    assert census["authoritative"] is True


# ---------------------------------------------------------------------------
# 4. Loader-pool health
# ---------------------------------------------------------------------------


def test_loader_health_reports_the_five_required_fields():
    health = loader_health(
        configured_workers=6,
        pool_open=True,
        rebuilds=2,
        last_rebuild_unix=1_787_000_000.0,
        last_rebuild_reason="BrokenProcessPool: a worker died",
        max_rebuilds=16,
    )
    assert health["configured_loader_workers"] == 6
    assert isinstance(health["live_loader_workers"], int)
    assert health["loader_pool_rebuilds"] == 2
    assert health["last_pool_rebuild_utc"].endswith("Z")
    assert "BrokenProcessPool" in health["last_pool_rebuild_reason"]
    assert health["max_loader_pool_rebuilds"] == 16


def test_an_idle_pool_is_not_reported_as_a_fault():
    """Zero workers during a collection is the healthy state of a healthy run."""
    idle = loader_health(configured_workers=6, pool_open=False)
    assert "idle" in idle["status"]
    assert idle["pool_open"] is False


def test_worker_status_is_a_health_sentence_not_a_constant():
    """The Agent 3 finding: the old field would not have shown the dead worker."""
    healthy = loader_health(configured_workers=0, pool_open=True)
    assert "0 of 0" in healthy["status"] or "live" in healthy["status"]
    degraded = loader_health(configured_workers=6, pool_open=True)
    # No pool is open in this process, so six configured workers read as absent.
    assert "no live loader workers" in degraded["status"]


def test_the_trainer_records_when_and_why_a_pool_was_rebuilt():
    """No process is killed here; the recorder is exercised directly."""
    from concurrent.futures import BrokenExecutor

    from stratego.training.phase14_trainer import (
        LOADER_POOL_EVENT_RETAIN,
        Phase14Trainer,
    )

    trainer = Phase14Trainer.__new__(Phase14Trainer)
    trainer.counters = {"loader_pool_rebuilds": 0}
    trainer.loader_pool_events = []
    trainer.global_step = 41
    trainer.rl_iteration = 3
    trainer.cursor = None
    trainer.counters["loader_pool_rebuilds"] = 1
    event = trainer._record_pool_rebuild(BrokenExecutor("a process in the pool died"))
    state = Phase14Trainer.loader_pool_state(trainer)
    assert event["rebuild_index"] == 1
    assert state["rebuilds"] == 1
    assert state["last_rebuild_utc"].endswith("Z")
    assert "died" in state["last_rebuild_reason"]
    assert state["max_rebuilds"] == 16
    for index in range(LOADER_POOL_EVENT_RETAIN * 2):
        trainer.counters["loader_pool_rebuilds"] = index + 2
        trainer._record_pool_rebuild(BrokenExecutor("again"))
    assert len(trainer.loader_pool_events) == LOADER_POOL_EVENT_RETAIN


def test_the_rebuild_cap_is_unchanged():
    from stratego.training.phase14_trainer import MAX_LOADER_POOL_REBUILDS

    assert MAX_LOADER_POOL_REBUILDS == 16


# ---------------------------------------------------------------------------
# 5. The restart policy
# ---------------------------------------------------------------------------


def _healthy() -> RestartConditions:
    return RestartConditions(resume_checkpoint_valid=True, deadline_known=True)


def test_an_unexpected_death_before_the_deadline_restarts():
    decision = restart_decision(_healthy())
    assert decision["action"] == ACTION_RESTART


@pytest.mark.parametrize(
    "field,fragment",
    [
        ("emergency_stop_active", "emergency stop is active"),
        ("run_closed", "training is closed"),
        ("integrity_failure_recorded", "unrecoverable integrity failure"),
    ],
)
def test_the_supervisor_refuses_to_restart(field, fragment):
    conditions = RestartConditions(
        resume_checkpoint_valid=True, deadline_known=True, **{field: True}
    )
    decision = restart_decision(conditions)
    assert decision["action"] == ACTION_STOP
    assert fragment in decision["reason"]


def test_no_valid_resume_checkpoint_means_no_restart():
    """A fresh start() would stamp a new 168-hour deadline. It never happens."""
    decision = restart_decision(RestartConditions(resume_checkpoint_valid=False))
    assert decision["action"] == ACTION_STOP
    assert "168-hour" in decision["reason"]


def test_a_post_deadline_death_gets_one_zero_step_closeout_then_stops():
    conditions = RestartConditions(
        resume_checkpoint_valid=True, deadline_known=True, deadline_passed=True
    )
    first = restart_decision(conditions, closeout_attempts=0)
    assert first["action"] == ACTION_FINALIZE_ONLY
    assert "zero optimizer steps" in first["reason"]
    exhausted = restart_decision(conditions, closeout_attempts=2, max_closeout_attempts=2)
    assert exhausted["action"] == ACTION_STOP
    # And a closed run past its deadline is simply over.
    closed = restart_decision(
        RestartConditions(
            resume_checkpoint_valid=True, deadline_passed=True, run_closed=True
        )
    )
    assert closed["action"] == ACTION_STOP


def test_the_consecutive_restart_bound_stops_a_crash_loop():
    for attempts in range(DEFAULT_MAX_CONSECUTIVE_RESTARTS):
        assert restart_decision(_healthy(), consecutive_restarts=attempts)["action"] == (
            ACTION_RESTART
        )
    stopped = restart_decision(
        _healthy(), consecutive_restarts=DEFAULT_MAX_CONSECUTIVE_RESTARTS
    )
    assert stopped["action"] == ACTION_STOP
    assert "bounded restart policy" in stopped["reason"]


def test_the_refusals_are_ordered_so_the_operator_reason_wins():
    """An emergency stop past the deadline reads as a stop, not a closeout."""
    conditions = RestartConditions(
        emergency_stop_active=True, resume_checkpoint_valid=True, deadline_passed=True
    )
    assert "emergency stop" in restart_decision(conditions)["reason"]


def test_exit_codes_and_signals_are_distinguished():
    assert exit_description(0)["exit_code"] == 0
    assert exit_description(3)["exit_code"] == 3
    killed = exit_description(-9)
    assert killed["signal"] == 9
    assert killed["signal_name"] == "SIGKILL"
    assert exit_description(None)["still_running"] is True


# ---------------------------------------------------------------------------
# 6. The supervisor reads its conditions off disk
# ---------------------------------------------------------------------------


@pytest.fixture()
def storage(tmp_path):
    layout = Phase14Storage.under(tmp_path / "external", hot_root=tmp_path / "hot")
    layout.prepare()
    return layout


def test_conditions_are_read_from_the_control_files(storage):
    conditions = read_conditions(storage)
    assert conditions.emergency_stop_active is False
    assert conditions.integrity_failure_recorded is False
    assert conditions.resume_checkpoint_valid is False

    request_emergency_stop(storage.external_root, reason="a test")
    assert read_conditions(storage).emergency_stop_active is True
    assert emergency_stop_state(storage.external_root)["reason"] == "a test"
    clear_emergency_stop(storage.external_root)
    assert read_conditions(storage).emergency_stop_active is False

    record_integrity_failure(storage.external_root, error="wrong starting model")
    assert read_conditions(storage).integrity_failure_recorded is True
    assert "wrong starting model" in integrity_failure_state(storage.external_root)["error"]


def test_an_unreadable_stop_file_still_stops(storage):
    emergency_stop_path(storage.external_root).write_text("{ not json")
    assert emergency_stop_state(storage.external_root)["active"] is True
    assert read_conditions(storage).emergency_stop_active is True


def test_a_closed_run_manifest_is_seen(storage):
    storage.run_state_path.write_text(
        json.dumps({"progress": {"closed": True, "close_reason": "deadline"}})
    )
    state = run_manifest_state(storage.run_state_path)
    assert state["closed"] is True
    assert read_conditions(storage).run_closed is True


def test_the_deadline_is_read_from_the_persisted_window():
    passed = deadline_state({"run_deadline_utc": "2020-01-01T00:00:00.000Z"})
    assert passed["passed"] is True
    ahead = deadline_state({"run_deadline_utc": "2099-01-01T00:00:00.000Z"})
    assert ahead["passed"] is False
    assert deadline_state({})["known"] is False


# ---------------------------------------------------------------------------
# 7. The supervisor never creates a deadline, and records what it did
# ---------------------------------------------------------------------------


def _supervisor(storage, manifest=None):
    from stratego.training.phase14_supervisor import Phase14Supervisor

    return Phase14Supervisor(
        storage,
        manifest=manifest or build_launch_manifest(),
        python="/usr/bin/true",
        learner_script=Path("/dev/null"),
        policy=SupervisorPolicy(poll_seconds=0.01, backoff_seconds=()),
    )


def test_the_supervisor_refuses_a_window_that_moved(storage):
    supervisor = _supervisor(storage)
    supervisor._observe_window({"run_window": {
        "run_start_utc": START,
        "run_deadline_utc": "2026-09-08T00:00:00.000Z",
        "transition_utc": "2026-09-06T12:00:00.000Z",
    }})
    with pytest.raises(Phase14SupervisorError, match="never creates a new one"):
        supervisor._observe_window({"run_window": {
            "run_start_utc": "2026-09-02T00:00:00.000Z",
            "run_deadline_utc": "2026-09-09T00:00:00.000Z",
            "transition_utc": "2026-09-07T12:00:00.000Z",
        }})


def test_the_supervisor_never_passes_a_deadline_to_the_learner(storage):
    arguments = " ".join(str(token) for token in _supervisor(storage)._learner_arguments("learner"))
    for forbidden in ("deadline", "rehearsal", "hours", "population-divisor"):
        assert forbidden not in arguments
    assert "--device mps" in arguments
    assert "--loader-workers 6" in arguments
    assert "--games-in-flight 96" in arguments


def test_the_supervisor_log_survives_and_reads_back(tmp_path):
    log = SupervisorLog(tmp_path / "supervisor.jsonl")
    log.emit("launch", learner_pid=4321, attempt=1, checkpoint_selected="hot_0001.pt")
    log.emit("unexpected_exit", signal=9, learner_pid=4321)
    records = log.read()
    assert [record["event"] for record in records] == ["launch", "unexpected_exit"]
    assert records[0]["learner_pid"] == 4321
    assert records[0]["utc"].endswith("Z")
    assert all("unix" in record for record in records)


def test_the_supervisor_records_every_field_the_task_requires():
    from stratego.training.phase14_supervisor import supervisor_semantics

    required = {
        "launch timestamp",
        "learner PID",
        "unexpected exit",
        "exit code / signal",
        "restart attempt",
        "checkpoint selected",
        "restart success/failure",
        "final process exit",
    }
    assert required <= set(supervisor_semantics()["records"])
    assert supervisor_semantics()["never"] == "creates a new training deadline"


def test_the_preflight_refuses_to_launch_over_an_emergency_stop(storage, fresh_manifest):
    supervisor = _supervisor(storage, fresh_manifest)
    request_emergency_stop(storage.external_root, reason="left over from yesterday")
    with pytest.raises(Phase14LaunchError, match="emergency stop is active"):
        supervisor.preflight()
    clear_emergency_stop(storage.external_root)
    record_integrity_failure(storage.external_root, error="pool digest mismatch")
    with pytest.raises(Phase14LaunchError, match="integrity failure"):
        supervisor.preflight()


# ---------------------------------------------------------------------------
# 8. The durable emergency stop reaches the run
# ---------------------------------------------------------------------------


def test_the_control_surface_honours_the_stop_file(tmp_path):
    from stratego.training.phase14_telemetry import ControlSurface

    stop = tmp_path / "phase14_emergency_stop.json"
    control = ControlSurface(stop_file=stop)
    assert control.should_continue() is True
    stop.write_text("{}")
    assert control.should_continue() is False
    assert control.status()["stop_requested"] is True
    assert control.status()["stop_file_present"] is True


def test_the_stop_file_does_not_become_a_settings_file(tmp_path):
    from stratego.training.phase14_telemetry import ControlSurface, Phase14TelemetryError

    control = ControlSurface(stop_file=tmp_path / "stop.json")
    for key in contract.IMMUTABLE_CONTROL_KEYS:
        with pytest.raises(Phase14TelemetryError):
            control.set(key, 1.0)
    assert len(control.refusals) == len(contract.IMMUTABLE_CONTROL_KEYS)


def test_a_production_runner_watches_the_stop_file_by_default(tmp_path):
    from stratego.training.phase14_clock import ManualClock
    from stratego.training.phase14_contract import Population
    from stratego.training.phase14_runner import MODE_TEST, Phase14Runner

    layout = Phase14Storage.under(tmp_path)
    runner = Phase14Runner(
        layout,
        clock=ManualClock(START),
        mode=MODE_TEST,
        device="cpu",
        inference_device="cpu",
        population=Population.scaled(512),
    )
    assert runner.control.stop_file == emergency_stop_path(layout.external_root)
    request_emergency_stop(layout.external_root, reason="stop now")
    assert runner.control.should_continue() is False


# ---------------------------------------------------------------------------
# 9. Candidate evaluation may not depend on memory
# ---------------------------------------------------------------------------


def test_pending_candidates_are_visible_on_disk(storage):
    from stratego.evaluation.phase14_candidates import CandidateLedger

    ledger = CandidateLedger.at(storage.evaluation_root)
    ledger.record_candidate(6, {"hour": 6, "snapshot_path": "archive_0001.pt"})
    ledger.record_candidate(12, {"hour": 12, "snapshot_path": "archive_0002.pt"})
    assert unevaluated_candidates(storage.evaluation_root) == [6, 12]


def test_a_failed_evaluation_preserves_the_candidate_and_is_rerunnable(storage):
    from stratego.evaluation.phase14_candidates import CandidateLedger

    ledger = CandidateLedger.at(storage.evaluation_root)
    ledger.record_candidate(6, {"hour": 6, "snapshot_path": "archive_0001.pt"})
    ledger.record_failure(6, "the bytes were on a volume that was not mounted")
    entry = ledger.read()["candidates"]["6"]
    assert entry["status"] == "failed"
    assert entry["rerunnable"] is True
    assert entry["mark"]["snapshot_path"] == "archive_0001.pt"
    assert unevaluated_candidates(storage.evaluation_root) == [6]


def test_the_hour_168_gate_refuses_an_incomplete_ledger(storage):
    from stratego.evaluation.phase14_candidates import CandidateLedger

    ledger = CandidateLedger.at(storage.evaluation_root)
    ledger.record_candidate(6, {"hour": 6})
    with pytest.raises(Phase14SupervisorError, match="no complete evaluation"):
        assert_all_candidates_evaluated(storage.evaluation_root)
    ledger.record_result(
        6,
        {
            "mean_ewr": 0.5,
            "min_stratum_ewr": 0.4,
            "strata": {},
            "games_played": 128,
            "complete": True,
            "pack_content_digest": contract.SELECTION_PACK_DIGEST,
            "seconds": 1.0,
        },
    )
    assert assert_all_candidates_evaluated(storage.evaluation_root)["candidates"] == 1


def test_candidate_evaluation_imports_no_search_and_no_trainer():
    """Structural: the evaluator cannot reach training, and cannot search."""
    import subprocess
    import sys

    finished = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, stratego.evaluation.phase14_candidates as module;"
            "print([n for n in sys.modules if n.startswith('stratego.search')"
            " or n.endswith('phase14_trainer') or n.endswith('phase14_runner')])",
        ],
        capture_output=True,
        text=True,
        cwd=str(contract.repository_root()),
        check=True,
    )
    assert finished.stdout.strip() == "[]"


# ---------------------------------------------------------------------------
# 10. The frozen operational topology
# ---------------------------------------------------------------------------


def test_the_frozen_topology_is_what_agent_3_rehearsed():
    topology = OperationalTopology.frozen()
    assert topology.device == "mps"
    assert topology.loader_workers == 6
    assert topology.games_in_flight == 96
    assert topology.to_dict()["games_per_iteration"] == 2048
    assert topology.to_dict()["in_logical_config_digest"] is False


def test_a_production_launch_refuses_an_unrehearsed_topology():
    with pytest.raises(Phase14LaunchError, match="frozen at"):
        assert_frozen_topology(OperationalTopology(loader_workers=12))


def test_the_manifest_carries_the_topology_outside_the_config_digest():
    manifest = load_launch_manifest()
    assert manifest["operational_topology"]["loader_workers"] == 6
    # The logical digest is the Agent 2 one, and it knows nothing about workers.
    assert manifest["integrated_config_digest"] == (
        "9c2a38e4335762997adbb33731dc619615fff713c2c60840c7c8d74a2f29da5e"
    )


# ---------------------------------------------------------------------------
# 11. The frozen config document says the true things
# ---------------------------------------------------------------------------


def test_the_final_config_binds_every_value_section_16_lists():
    document = final_training_config_document()
    assert document["starting_model"]["sha256"] == contract.STARTING_CHECKPOINT_SHA256
    assert document["starting_model"]["agent1c"]["used_as_phase14_policy_value"] is False
    assert document["learning_rate"]["main"] == 7.5e-05
    assert document["learning_rate"]["late"] == 3.75e-05
    assert document["transition"]["transition_seconds"] == 475200
    assert document["opponent_mixture"]["main"]["current"] == 1188
    assert document["opponent_mixture"]["late"]["historical"] == 984
    assert document["checkpoint_cadences"]["hot_seconds"] == 900
    assert document["checkpoint_cadences"]["archive_seconds"] == 7200
    assert document["checkpoint_cadences"]["candidate_seconds"] == 21600
    assert document["candidate_evaluation"]["pack_digest"] == contract.SELECTION_PACK_DIGEST
    assert document["candidate_evaluation"]["search_permitted"] is False
    assert document["deadline_semantics"]["deadline_seconds"] == 604800
    assert document["search_excluded"] is True
    assert document["objectives"]["detail"]["belief_loss_weight"] == 0.25


def test_the_checkpoint_age_semantics_are_not_overclaimed():
    """Section 7: do not say every crash loses at most exactly 15 minutes."""
    semantics = recovery_semantics()
    assert semantics["hot_checkpoint_cadence_seconds"] == 900
    assert semantics["cadence_is_nominal"] is True
    assert semantics["sealed_games_survive_a_later_learner_crash"] is True
    assert semantics["max_checkpoint_age_observed_seconds"] == 895.4
    assert "plus a collection" in semantics["read_checkpoint_age_as"]


def test_the_rng_documentation_is_preserved_exactly():
    """Section 8: captured yes, restored no, and the reason recorded."""
    semantics = rng_semantics()
    assert semantics["global_rng_state_captured"] is True
    assert semantics["global_rng_state_restored"] is False
    assert "explicit deterministic streams" in semantics["reason"]
    assert semantics["redesigned_by_agent_4"] is False


def test_building_the_manifest_twice_gives_the_same_identity():
    """A rebuild over unchanged code is the same manifest, digest included."""
    first = build_launch_manifest()
    second = build_launch_manifest()
    assert first["launch_manifest_digest"] == second["launch_manifest_digest"]
    assert first["launch_manifest_digest_excludes"] == ["built_utc"]
    first.pop("built_utc")
    second.pop("built_utc")
    assert first == second


def test_the_manifest_leaves_the_absolute_deadline_unstamped():
    """It cannot be known until launch, and the launch materializes it once."""
    manifest = load_launch_manifest()
    assert manifest["deadline"]["run_start_utc"] is None
    assert manifest["deadline"]["run_deadline_utc"] is None
    assert manifest["deadline"]["duration_hours"] == 168


# ---------------------------------------------------------------------------
# 12. The supervisor loop, end to end, against a stand-in learner
# ---------------------------------------------------------------------------


def _fake_learner(path: Path, *, exit_code: int, sleep_seconds: float = 0.0) -> Path:
    path.write_text(
        "import sys, time\n"
        f"time.sleep({sleep_seconds})\n"
        f"sys.exit({exit_code})\n"
    )
    return path


@pytest.fixture()
def supervisor_harness(storage, tmp_path, monkeypatch, fresh_manifest):
    """A supervisor whose learner is a stand-in and whose checkpoint is a fact.

    The training process is replaced, not simulated: the supervisor really
    launches a child, really watches it die, and really decides. What is faked
    is only the ten-megabyte checkpoint it would otherwise have to write.
    """
    import stratego.training.phase14_supervisor as supervisor_module

    window = {
        "run_start_utc": START,
        "run_deadline_utc": "2026-09-08T00:00:00.000Z",
        "transition_utc": "2026-09-06T12:00:00.000Z",
    }
    state = {"step": 100, "valid": True, "window": window}

    def fake_checkpoint(hot_root):
        return {
            "valid": state["valid"],
            "path": "hot_000007_step000000100.pt",
            "global_optimizer_step": state["step"],
            "iteration": 3,
            "run_window": state["window"],
        }

    monkeypatch.setattr(supervisor_module, "resume_checkpoint_state", fake_checkpoint)
    return supervisor_module, state


def test_the_supervisor_records_a_death_and_relaunches(supervisor_harness, storage, tmp_path):
    module, _state = supervisor_harness
    learner = _fake_learner(tmp_path / "learner.py", exit_code=1)
    supervisor = module.Phase14Supervisor(
        storage,
        manifest=build_launch_manifest(),
        python="python3",
        learner_script=learner,
        policy=SupervisorPolicy(
            poll_seconds=0.01,
            backoff_seconds=(),
            max_consecutive_restarts=2,
            candidate_poll_seconds=10_000.0,
            status_sample_seconds=10_000.0,
        ),
    )
    supervisor.launch(reason="test launch")
    final = supervisor.supervise(max_seconds=30.0)
    events = [record["event"] for record in supervisor.log.read()]
    assert "launch" in events
    assert "unexpected_exit" in events
    assert "restart_attempt" in events
    assert "final_process_exit" in events
    assert "bounded restart policy" in final["state"]["stopped_because"]
    # Three launches: the original plus the two the bound allows.
    assert final["state"]["launches"] == 3

    launches = [r for r in supervisor.log.read() if r["event"] == "launch"]
    for record in launches:
        assert record["learner_pid"] > 0
        assert record["launch_timestamp"].endswith("Z")
        assert record["checkpoint_selected"] == "hot_000007_step000000100.pt"
        assert record["checkpoint_step"] == 100
    deaths = [r for r in supervisor.log.read() if r["event"] == "unexpected_exit"]
    assert all(record["exit_code"] == 1 for record in deaths)


def test_the_supervisor_stops_when_the_run_closes_itself(supervisor_harness, storage, tmp_path):
    module, _state = supervisor_harness
    storage.run_state_path.write_text(
        json.dumps({"progress": {"closed": True, "close_reason": "deadline"}})
    )
    supervisor = module.Phase14Supervisor(
        storage,
        manifest=build_launch_manifest(),
        python="python3",
        learner_script=_fake_learner(tmp_path / "learner.py", exit_code=0),
        policy=SupervisorPolicy(
            poll_seconds=0.01,
            backoff_seconds=(),
            candidate_poll_seconds=10_000.0,
            status_sample_seconds=10_000.0,
        ),
    )
    supervisor.launch(reason="test launch")
    final = supervisor.supervise(max_seconds=30.0)
    assert final["state"]["launches"] == 1
    assert "training is closed" in final["state"]["stopped_because"]
    assert "expected_exit" in [record["event"] for record in supervisor.log.read()]


def test_a_restart_that_makes_progress_clears_the_consecutive_count(
    supervisor_harness, storage, tmp_path
):
    module, state = supervisor_harness
    supervisor = module.Phase14Supervisor(
        storage,
        manifest=build_launch_manifest(),
        python="python3",
        learner_script=_fake_learner(tmp_path / "learner.py", exit_code=0),
        policy=SupervisorPolicy(poll_seconds=0.01, backoff_seconds=()),
    )
    supervisor.launch(reason="test launch")
    supervisor.state.consecutive_restarts = 2
    state["step"] = 500
    supervisor._record_progress()
    assert supervisor.state.consecutive_restarts == 0
    assert "restart_success" in [record["event"] for record in supervisor.log.read()]


def test_the_supervisor_stops_when_the_checkpoint_disappears(
    supervisor_harness, storage, tmp_path
):
    module, state = supervisor_harness
    supervisor = module.Phase14Supervisor(
        storage,
        manifest=build_launch_manifest(),
        python="python3",
        learner_script=_fake_learner(tmp_path / "learner.py", exit_code=1),
        policy=SupervisorPolicy(
            poll_seconds=0.01,
            backoff_seconds=(),
            candidate_poll_seconds=10_000.0,
            status_sample_seconds=10_000.0,
        ),
    )
    supervisor.launch(reason="test launch")
    state["valid"] = False
    final = supervisor.supervise(max_seconds=30.0)
    assert final["state"]["launches"] == 1
    assert "no valid resume checkpoint" in final["state"]["stopped_because"]


def test_a_learner_that_dies_past_the_deadline_gets_one_closeout(
    supervisor_harness, storage, tmp_path
):
    module, state = supervisor_harness
    state["window"] = {
        "run_start_utc": "2020-01-01T00:00:00.000Z",
        "run_deadline_utc": "2020-01-08T00:00:00.000Z",
        "transition_utc": "2020-01-06T12:00:00.000Z",
    }
    supervisor = module.Phase14Supervisor(
        storage,
        manifest=build_launch_manifest(),
        python="python3",
        learner_script=_fake_learner(tmp_path / "learner.py", exit_code=1),
        policy=SupervisorPolicy(
            poll_seconds=0.01,
            backoff_seconds=(),
            max_closeout_attempts=1,
            candidate_poll_seconds=10_000.0,
            status_sample_seconds=10_000.0,
        ),
    )
    supervisor.launch(reason="test launch")
    final = supervisor.supervise(max_seconds=30.0)
    records = supervisor.log.read()
    closeouts = [r for r in records if r["event"] == "closeout_launch"]
    assert len(closeouts) == 1
    finalize_launches = [
        r for r in records if r["event"] == "launch" and r.get("role") == "finalize"
    ]
    assert len(finalize_launches) == 1
    assert final["state"]["closeout_attempts"] == 1
    assert "closeout launch has been attempted" in final["state"]["stopped_because"]


def test_the_supervisor_launches_the_evaluator_for_a_pending_candidate(
    supervisor_harness, storage, tmp_path
):
    from stratego.evaluation.phase14_candidates import CandidateLedger

    module, _state = supervisor_harness
    CandidateLedger.at(storage.evaluation_root).record_candidate(
        6, {"hour": 6, "snapshot_path": "archive_0001.pt"}
    )
    supervisor = module.Phase14Supervisor(
        storage,
        manifest=build_launch_manifest(),
        python="python3",
        learner_script=_fake_learner(tmp_path / "learner.py", exit_code=0, sleep_seconds=5),
        evaluator_script=_fake_learner(tmp_path / "evaluator.py", exit_code=0),
        policy=SupervisorPolicy(poll_seconds=0.01, backoff_seconds=()),
    )
    supervisor.launch(reason="test launch")
    record = supervisor._maybe_evaluate_candidates()
    assert record is not None
    assert record["pending_hours"] == [6]
    assert supervisor.evaluator is not None
    assert supervisor.evaluator.pid != supervisor.child.pid
    supervisor.stop_child()


def test_a_post_deadline_resume_takes_zero_optimizer_steps(tmp_path):
    """The runner-level guarantee behind the closeout launch.

    Agent 3 proved this nine hours past a real rehearsal deadline; repeated
    here at test scale so the supervisor's `finalize_only` branch rests on a
    checked property rather than on a remembered one.
    """
    from stratego.training.phase14_clock import ManualClock
    from stratego.training.phase14_contract import Population
    from stratego.training.phase14_runner import MODE_TEST, Phase14Runner

    layout = Phase14Storage.under(tmp_path)
    runner = Phase14Runner(
        layout,
        clock=ManualClock(START),
        mode=MODE_TEST,
        device="cpu",
        inference_device="cpu",
        population=Population.scaled(512),
    )
    started = runner.start()
    step_at_start = runner.trainer.global_step
    window_before = json.dumps(runner.controller.window.to_dict(), sort_keys=True)

    late = Phase14Runner(
        layout,
        clock=ManualClock("2026-09-09T00:00:00.000Z"),
        mode=MODE_TEST,
        device="cpu",
        inference_device="cpu",
        population=Population.scaled(512),
    )
    resumed = late.resume()
    assert resumed["past_deadline"] is True
    result = late.run()
    assert result["stopped_because"] == "deadline"
    assert late.trainer.global_step == step_at_start
    assert json.dumps(late.controller.window.to_dict(), sort_keys=True) == window_before
    assert started["run_deadline_utc"] == resumed["run_deadline_utc"]
    late.trainer.close()
    runner.trainer.close()


def test_the_code_closure_does_not_depend_on_the_calling_process():
    """It is computed in a clean interpreter, and that is load-bearing.

    A full-suite run has already imported `stratego.search` by the time this
    executes. If the closure were read from this process's `sys.modules`, the
    bound file set would grow with whatever else ran first, and two honest
    processes would compute two different manifests.
    """
    import stratego.search.phase12.engine  # noqa: F401 - deliberate pollution

    binding = code_binding()
    assert binding["search_excluded"] is True
    assert binding["search_modules_in_training_closure"] == []
    assert binding["closure_files"] == len(code_closure())


# ---------------------------------------------------------------------------
# 13. The host has to stay awake for 168 hours
# ---------------------------------------------------------------------------


def test_the_host_power_state_is_read_not_assumed():
    from stratego.training.phase14_launch import host_power_state

    state = host_power_state()
    assert state["known"] in (True, False)
    if state["known"]:
        assert "sleep" in state["settings"]
        assert isinstance(state["will_stay_awake"], bool)


def test_a_machine_that_will_sleep_is_refused(monkeypatch):
    """The deadline is wall-clock: sleeping at hour 3 loses the hours."""
    import stratego.training.phase14_launch as launch_module

    monkeypatch.setattr(
        launch_module,
        "host_power_state",
        lambda: {
            "known": True,
            "settings": {"sleep": 1},
            "idle_sleep_minutes": 1,
            "disk_sleep_minutes": 10,
            "sleep_prevented_by": "",
            "will_stay_awake": False,
            "disk_will_stay_spun_up": False,
        },
    )
    with pytest.raises(Phase14LaunchError, match="idle-sleeps"):
        launch_module.assert_host_stays_awake()


def test_a_power_assertion_satisfies_the_check(monkeypatch):
    """Which is how `caffeinate -dimsu` passes without a password."""
    import stratego.training.phase14_launch as launch_module

    monkeypatch.setattr(
        launch_module,
        "host_power_state",
        lambda: {
            "known": True,
            "settings": {"sleep": 1},
            "idle_sleep_minutes": 1,
            "sleep_prevented_by": "caffeinate",
            "will_stay_awake": True,
            "disk_will_stay_spun_up": True,
        },
    )
    assert launch_module.assert_host_stays_awake()["verified"] is True


def test_an_unreadable_pmset_does_not_block_a_launch(monkeypatch):
    """Unknown is not the same as unsafe; it is recorded rather than refused."""
    import stratego.training.phase14_launch as launch_module

    monkeypatch.setattr(
        launch_module, "host_power_state", lambda: {"known": False, "error": "no pmset"}
    )
    assert launch_module.assert_host_stays_awake()["verified"] is False


def test_a_preflight_does_not_create_a_run_identity(storage, tmp_path, fresh_manifest):
    """`--preflight-only` checks; it does not bring a Phase 14 run into being."""
    from stratego.training.phase14_supervisor import Phase14Supervisor

    empty = Phase14Storage.under(tmp_path / "untouched", hot_root=tmp_path / "untouched_hot")
    supervisor = Phase14Supervisor(
        empty,
        manifest=fresh_manifest,
        python="/usr/bin/true",
        learner_script=Path("/dev/null"),
    )
    assert not Path(empty.hot_root).exists()
    report = supervisor.preflight()
    assert report["code"]["verified"] is True
    assert not list(Path(empty.hot_root).glob("hot_*.pt"))
    assert not Path(empty.run_state_path).exists()
