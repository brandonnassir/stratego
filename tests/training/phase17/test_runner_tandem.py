"""Agent 4: the tandem runner's integration behaviour.

Everything here runs a real tandem iteration -- real move forward passes, real
boundary targets, a real move epoch, real setup generation and, where the queue
allows, five real setup epochs. Nothing is mocked: a mocked timing path would
prove that the mock works.

The population and budget are small so the mechanics are exercised quickly. A
small window says nothing about strength, and none of these tests claim any.
"""

from __future__ import annotations

import pytest

from stratego.training.phase17.export import (
    build_manifest,
    due_boundaries,
    verify_paired_export,
    write_paired_export,
)
from stratego.training.phase17.checkpoint import (
    read_joint_checkpoint,
    write_joint_checkpoint,
)
from stratego.training.phase17.queue import (
    Phase17BudgetError,
    SetupBudgetPolicy,
)
from stratego.training.phase17.runner import (
    PHASE17_SETUP_FAMILY,
    Phase17RunnerError,
    TandemConfig,
    TandemRunner,
)
from stratego.training.phase17.supervisor import MODE_INTEGRATION
from stratego.training.phase17.telemetry import (
    Phase17TelemetryError,
    TelemetryWriter,
    read_rows,
)


def tiny_config(run_id: str = "RUN-TEST-A") -> TandemConfig:
    return TandemConfig(
        run_id=run_id,
        total_iterations=20,
        move_budget=400,
        population=8,
        pool_size_per_side=16,
        setup_budget=4,
        setup_queue_capacity=64,
        setup_warm_up_minimum=4,
        setup_minibatch_episodes=4,
        move_minibatch_size=64,
    )


@pytest.fixture(scope="module")
def advanced():
    """One runner advanced three iterations, shared by the read-only tests."""
    runner = TandemRunner(tiny_config(), supervisor_mode=MODE_INTEGRATION)
    results = [runner.run_iteration() for _ in range(5)]
    return runner, results


# -- the iteration ---------------------------------------------------------


def test_the_window_lands_exactly_on_the_budget(advanced):
    _, results = advanced
    for result in results:
        assert result.window.transitions_harvested == 400
        assert len(result.window.rows) == 400


def test_both_finished_and_unfinished_games_are_present(advanced):
    _, results = advanced
    assert any(result.window.games_finished for result in results)
    assert all(result.window.active_games > 0 for result in results)
    assert any(result.window.boundary_rows for result in results)


def test_only_the_current_raw_policy_ever_acted(advanced):
    runner, _ = advanced
    ledger = runner.collector.participant_ledger()
    assert ledger["holds"]
    assert ledger["unknown_model_states"] == {}
    assert ledger["rule_or_stress_decisions"] == 0
    assert ledger["historical_participants"] == 0
    assert ledger["search_participants"] == 0
    assert ledger["seats"]["red"] == ledger["seats"]["blue"]


def test_the_move_policy_is_rebound_after_every_update(advanced):
    runner, results = advanced
    for result in results:
        assert result.rebind["changed"], "an update that did not move the weights"
    # Every iteration held a distinct acting model state.
    assert len(set(runner.cell.known_digests())) == len(runner.cell.known_digests())


def test_an_in_flight_game_plays_on_under_the_new_weights(advanced):
    """The Phase 16 defect, checked on games that survived a rebind."""
    runner, results = advanced
    digests = {
        row.behavior_model_state_digest
        for result in results
        for row in result.window.rows
    }
    assert len(digests) == len(results), "each window must record its own acting digest"
    survivors = {
        row.game_id for row in results[0].window.rows
    } & {row.game_id for row in results[-1].window.rows}
    assert survivors, "no game survived two rebinds; the test proves nothing"
    for game_id in survivors:
        first = {
            row.behavior_model_state_digest
            for row in results[0].window.rows
            if row.game_id == game_id
        }
        third = {
            row.behavior_model_state_digest
            for row in results[-1].window.rows
            if row.game_id == game_id
        }
        assert first.isdisjoint(third)


def test_setups_come_from_the_setup_policy_and_never_a_library(advanced):
    runner, _ = advanced
    assert runner.provider.setup_family == PHASE17_SETUP_FAMILY
    assert runner.provider.legality_failures == 0
    assert runner.provider.orientation_failures == 0
    assert runner.provider.fallback_attempts == 0
    telemetry = runner.provider.telemetry()
    assert telemetry["generated"] > 0
    # Every assigned game is either still open or has completed and been
    # enqueued. Nothing goes anywhere else.
    completed = runner.setup_trainer.queue.enqueued_count // 2
    assert telemetry["assigned"] == telemetry["open_episodes"] + completed


def test_every_finished_game_enqueued_both_of_its_episodes(advanced):
    runner, results = advanced
    finished = sum(result.window.games_finished for result in results)
    queue = runner.setup_trainer.queue
    assert queue.enqueued_count == 2 * finished
    assert not runner.enqueue_rejections


def test_a_setup_update_consumes_exactly_the_frozen_budget(advanced):
    runner, results = advanced
    real = [
        result
        for result in results
        if result.setup_update is not None and not result.setup_update.skipped
    ]
    assert real, "no setup update ran; the tandem path is unproven"
    for result in real:
        assert result.setup_update.episodes_consumed == 4


def test_five_setup_epochs_run_and_are_timed(advanced):
    _, results = advanced
    real = [
        result
        for result in results
        if result.setup_update is not None and not result.setup_update.skipped
    ]
    for result in real:
        assert len(result.setup_update.epochs) == 5
        assert result.seconds["setup_optimization"] > 0.0


def test_the_setup_controller_steps_once_per_iteration_on_the_final_epoch(advanced):
    """Decision D9-B section 4: not once per epoch, and not the epoch mean."""
    runner, results = advanced
    real = [
        result
        for result in results
        if result.setup_update is not None and not result.setup_update.skipped
    ]
    for result in real:
        update = result.setup_update
        assert len(update.per_epoch_kl) == 5
        assert update.control_kl == pytest.approx(update.per_epoch_kl[-1])
        # The final epoch, NOT the mean across epochs. These differ whenever
        # the KL climbs across epochs, which is the normal case: epoch 0 starts
        # on the behavior snapshot and its KL is near zero by construction.
        if len(set(update.per_epoch_kl)) > 1:
            assert update.control_kl != pytest.approx(update.mean_iteration_kl)
        # Exactly one beta move per iteration, across five epochs.
        assert len(update.epochs) == 5
        assert len(runner.setup_trainer.controller.history) == runner.setup_updates


def test_a_short_queue_skips_explicitly_rather_than_shrinking(advanced):
    _, results = advanced
    skipped = [result for result in results if result.setup_skipped]
    assert skipped, "the warm-up path never ran"
    for result in skipped:
        assert result.setup_skip_reason
        assert "queue holds" in result.setup_skip_reason


def test_a_real_setup_update_moves_the_setup_weights(advanced):
    _, results = advanced
    real = [
        result
        for result in results
        if result.setup_update is not None and not result.setup_update.skipped
    ]
    for result in real:
        assert result.setup_update.optimizer_steps > 0
        assert result.setup_update.digest_before != result.setup_update.digest_after


def test_the_horizon_is_never_extended_by_production_speed():
    runner = TandemRunner(
        TandemConfig(
            run_id="RUN-TEST-A",
            total_iterations=2,
            move_budget=60,
            population=2,
            pool_size_per_side=4,
            setup_budget=2,
            setup_queue_capacity=8,
            setup_warm_up_minimum=2,
            setup_minibatch_episodes=2,
            move_minibatch_size=32,
        ),
        supervisor_mode=MODE_INTEGRATION,
    )
    for _ in range(2):
        runner.run_iteration()
    with pytest.raises(Phase17RunnerError, match="outside the frozen horizon"):
        runner.run_iteration()


# -- persistence -----------------------------------------------------------


def _fingerprint(runner, result) -> dict:
    window = result.window
    return {
        "iteration": result.iteration,
        "harvested": window.transitions_harvested,
        "row_keys": [
            (row.game_id, int(row.color), int(row.ply), int(row.sampled_action))
            for row in window.rows
        ],
        "advantages": [round(float(row.advantage_target), 9) for row in window.rows],
        "wdl_targets": [
            tuple(round(float(v), 9) for v in row.wdl_target) for row in window.rows
        ],
        "provenance": [row.target_provenance for row in window.rows],
        "games_finished": window.games_finished,
        "terminal_results": dict(window.terminal_results),
        "move_raw": runner._move_digest(),
        "move_ema": runner.move_ema_digest(),
        "setup_raw": runner._setup_digest(),
        "setup_ema": runner.setup_ema_digest(),
        "queue_depth": result.queue_telemetry["depth"],
        "setup_skipped": result.setup_skipped,
    }


def test_a_round_trip_reproduces_the_next_iteration_exactly(tmp_path):
    """Common contract section 10: the active population survives EXACTLY.

    Not "approximately" and not "the same number of games": the same games, at
    the same plies, sampling the same actions and producing the same targets
    and the same optimizer update.
    """
    config = tiny_config("RUN-TEST-A")

    control = TandemRunner(config, supervisor_mode=MODE_INTEGRATION)
    for _ in range(2):
        control.run_iteration()
    expected = _fingerprint(control, control.run_iteration())

    interrupted = TandemRunner(config, supervisor_mode=MODE_INTEGRATION)
    for _ in range(2):
        interrupted.run_iteration()
    payload = interrupted.capture(
        checkpoint_generation=1,
        parent_checkpoint_identity={"path": None, "generation": 0},
        config_digest="cfg",
        source_digest="src",
        run_digest="run",
        telemetry_position={"path": str(tmp_path / "t.jsonl"), "records": 0, "offset": 0, "last_record_digest": None},
        next_export_boundary_seconds=1800.0,
    )
    write_joint_checkpoint(payload, tmp_path / "joint.pt")
    reread = read_joint_checkpoint(
        tmp_path / "joint.pt",
        run_id=config.run_id,
        config_digest="cfg",
        source_digest="src",
    )

    resumed = TandemRunner(config, supervisor_mode=MODE_INTEGRATION)
    report = resumed.restore(reread)
    assert report["games_restored"] == report["games_reseated"] > 0
    assert report["setup_episodes_restored"] == report["games_restored"]
    observed = _fingerprint(resumed, resumed.run_iteration())
    assert observed == expected


def test_a_resumed_game_keeps_the_setups_it_was_created_with(tmp_path):
    config = tiny_config("RUN-TEST-A")
    original = TandemRunner(config, supervisor_mode=MODE_INTEGRATION)
    original.run_iteration()
    before = {
        runner.game_id: (
            tuple(runner.builder.red_setup),
            tuple(runner.builder.blue_setup),
        )
        for runner in original.collector.slots
        if runner is not None
    }
    payload = original.capture(
        checkpoint_generation=1,
        parent_checkpoint_identity={},
        config_digest="cfg",
        source_digest="src",
        run_digest="run",
        telemetry_position={"path": str(tmp_path / "t.jsonl"), "records": 0, "offset": 0, "last_record_digest": None},
        next_export_boundary_seconds=1800.0,
    )
    resumed = TandemRunner(config, supervisor_mode=MODE_INTEGRATION)
    resumed.restore(payload)
    after = {
        runner.game_id: (
            tuple(runner.builder.red_setup),
            tuple(runner.builder.blue_setup),
        )
        for runner in resumed.collector.slots
        if runner is not None
    }
    assert after == before


def test_the_checkpoint_carries_every_active_game_and_its_episodes(tmp_path):
    config = tiny_config("RUN-TEST-A")
    runner = TandemRunner(config, supervisor_mode=MODE_INTEGRATION)
    runner.run_iteration()
    payload = runner.capture(
        checkpoint_generation=1,
        parent_checkpoint_identity={},
        config_digest="cfg",
        source_digest="src",
        run_digest="run",
        telemetry_position={"path": str(tmp_path / "t.jsonl"), "records": 0, "offset": 0, "last_record_digest": None},
        next_export_boundary_seconds=1800.0,
    )
    active = {entry["game_id"] for entry in payload["active_games"]}
    assert active == set(payload["active_game_setup_episodes"])
    assert active == {
        runner.game_id for runner in runner.collector.slots if runner is not None
    }
    for entry in payload["active_games"]:
        assert entry["engine_snapshot"]["action_history"]
        assert entry["builder_decisions"]
        assert entry["traces"]


def test_the_rng_namespaces_are_all_derived():
    runner = TandemRunner(tiny_config(), supervisor_mode=MODE_INTEGRATION)
    namespaces = runner.rng_namespaces()
    assert "not read" in namespaces["global_generators"]["torch"]
    assert namespaces["move_action_sampling"]["derived_from"] == ["game_id", "ply"]


# -- exports ---------------------------------------------------------------


def test_an_h0_export_carries_both_ema_halves_and_re_verifies(tmp_path):
    runner = TandemRunner(tiny_config(), supervisor_mode=MODE_INTEGRATION)
    manifest = build_manifest(
        run_id=runner.config.run_id,
        index=0,
        move_ema_digest=runner.move_ema_digest(),
        setup_ema_digest=runner.setup_ema_digest(),
        move_parameter_count=863959,
        setup_parameter_count=802320,
        start_identity=runner.start.identity["model_state_digest"] and {
            "model_state_digest": runner.start.identity["model_state_digest"]
        },
        parent_checkpoint={"generation": 0},
        config_digest="cfg",
        source_digest="src",
        elapsed_active_training_seconds=0.0,
        iteration=0,
    )
    candidate = write_paired_export(
        directory=tmp_path,
        manifest=manifest,
        move_ema_state=runner.start.ema.state_dict(),
        setup_ema_state=runner.setup_trainer.ema.state_dict(),
    )
    verified = verify_paired_export(candidate.path, expected_file_sha256=candidate.file_sha256)
    assert verified["verified"]
    assert verified["move_ema_model_state_digest"] == runner.move_ema_digest()
    assert verified["setup_ema_model_state_digest"] == runner.setup_ema_digest()
    assert manifest["lanes"]["move_only"]["consumes_setup"] is False
    assert manifest["lanes"]["joint_move_setup"]["consumes_setup"] is True


def test_creating_an_export_mutates_nothing(tmp_path):
    runner = TandemRunner(tiny_config(), supervisor_mode=MODE_INTEGRATION)
    runner.run_iteration()
    before = (
        runner._move_digest(),
        runner.move_ema_digest(),
        runner._setup_digest(),
        runner.setup_ema_digest(),
        runner.cell.digest,
        runner.iteration,
        runner.elapsed_active_training_seconds,
        runner.collector.draw_counts[:],
    )
    manifest = build_manifest(
        run_id=runner.config.run_id, index=1,
        move_ema_digest=runner.move_ema_digest(),
        setup_ema_digest=runner.setup_ema_digest(),
        move_parameter_count=863959, setup_parameter_count=802320,
        start_identity={}, parent_checkpoint={}, config_digest="cfg",
        source_digest="src", elapsed_active_training_seconds=1800.0, iteration=1,
    )
    write_paired_export(
        directory=tmp_path, manifest=manifest,
        move_ema_state=runner.start.ema.state_dict(),
        setup_ema_state=runner.setup_trainer.ema.state_dict(),
    )
    after = (
        runner._move_digest(),
        runner.move_ema_digest(),
        runner._setup_digest(),
        runner.setup_ema_digest(),
        runner.cell.digest,
        runner.iteration,
        runner.elapsed_active_training_seconds,
        runner.collector.draw_counts[:],
    )
    assert after == before


def test_the_cadence_never_drops_a_boundary():
    assert due_boundaries(0.0, 1799.0) == []
    assert due_boundaries(0.0, 1800.0) == [1]
    assert due_boundaries(1700.0, 3700.0) == [1, 2], "a long iteration must emit both"
    assert due_boundaries(12 * 3600 - 1, 13 * 3600) == [24]
    assert due_boundaries(12 * 3600, 13 * 3600) == []


# -- the injected stop -----------------------------------------------------


def test_an_injected_stop_records_its_reason_and_exits_safely(tmp_path):
    """Agent 4 instruction section 9: one injected event, proved end to end."""
    runner = TandemRunner(tiny_config(), supervisor_mode=MODE_INTEGRATION)
    runner.run_iteration()
    # Inject an absolute-floor failure, which is hard in every mode.
    verdict = runner.supervisor.observe_flag_support(2.0)
    assert verdict["fired"]
    assert runner.supervisor.should_stop

    payload = runner.capture(
        checkpoint_generation=99,
        parent_checkpoint_identity={},
        config_digest="cfg",
        source_digest="src",
        run_digest="run",
        telemetry_position={"path": str(tmp_path / "t.jsonl"), "records": 0, "offset": 0, "last_record_digest": None},
        next_export_boundary_seconds=1800.0,
    )
    identity = write_joint_checkpoint(payload, tmp_path / "stop.pt")
    record = runner.supervisor.stop_record()
    assert record["code"] == "P5"
    assert record["evidence"]["flag_effective_support"] == 2.0
    assert "no hyperparameter was changed" in record["action"]
    # The safe exit produced a loadable checkpoint, not a corpse.
    assert read_joint_checkpoint(
        tmp_path / "stop.pt", run_id=runner.config.run_id, config_digest="cfg"
    )["checkpoint_generation"] == 99
    assert identity.generation == 99


# -- the budget policy ------------------------------------------------------


def test_an_unsustainable_budget_is_refused_rather_than_frozen():
    with pytest.raises(Phase17BudgetError, match="completed no games"):
        SetupBudgetPolicy.freeze(games_per_iteration=0.0)


def test_the_budget_exceeds_the_arrival_rate_so_the_queue_cannot_grow():
    """The margin points UP: Agent 3's queue raises at capacity, never evicts.

    A budget below the arrival rate is not conservative -- it is a run that
    dies on an exception some hours in.
    """
    policy = SetupBudgetPolicy.freeze(games_per_iteration=100.0)
    arrivals = 200.0
    assert policy.budget > arrivals
    assert policy.sustainability_margin >= 1.10
    assert policy.capacity > policy.budget
    assert policy.warm_up_minimum >= policy.budget
    assert 0.0 < policy.document()["expected_skip_fraction"] < 0.2

    # Simulate the equilibrium: with a fixed arrival rate the depth must stay
    # bounded rather than march at the capacity.
    depth, peak = 0.0, 0.0
    for _ in range(400):
        depth += arrivals
        if policy.may_update(int(depth), warmed_up=True)["update"]:
            depth -= policy.budget
        peak = max(peak, depth)
    assert peak < policy.capacity, "the queue reached the capacity that raises"


def test_a_budget_below_the_arrival_rate_is_refused():
    from stratego.training.phase17.queue import SetupBudgetPolicy as Policy

    with pytest.raises(Phase17BudgetError, match="grow without bound"):
        Policy.freeze(games_per_iteration=100.0, margin=0.9)


def test_the_budget_never_prefers_short_games():
    policy = SetupBudgetPolicy.freeze(games_per_iteration=100.0)
    assert policy.may_update(policy.budget, warmed_up=True)["update"]
    short = policy.may_update(policy.budget - 1, warmed_up=True)
    assert not short["update"] and short["reason"] == "starved"
    # Warm-up is two budgets deep, so one budget is not yet enough.
    cold = policy.may_update(policy.budget, warmed_up=False)
    assert not cold["update"] and cold["reason"] == "warm_up"
    assert policy.may_update(policy.warm_up_minimum, warmed_up=False)["update"]


# -- telemetry --------------------------------------------------------------


def make_row(index: int) -> dict:
    from stratego.training.phase17.telemetry import (
        REQUIRED_MOVE_KEYS,
        REQUIRED_SETUP_KEYS,
        REQUIRED_SYSTEM_KEYS,
    )

    return {
        "move": {key: index for key in REQUIRED_MOVE_KEYS},
        "setup": {key: index for key in REQUIRED_SETUP_KEYS},
        "system": {key: index for key in REQUIRED_SYSTEM_KEYS},
    }


def test_a_row_missing_a_frozen_field_is_refused(tmp_path):
    writer = TelemetryWriter(path=tmp_path / "t.jsonl", run_id="RUN-TEST-A")
    row = make_row(0)
    del row["setup"]["queue"]
    with pytest.raises(Phase17TelemetryError, match="missing"):
        writer.append(row)


def test_appending_is_durable_and_resumable(tmp_path):
    path = tmp_path / "t.jsonl"
    writer = TelemetryWriter(path=path, run_id="RUN-TEST-A")
    for index in range(3):
        writer.append(make_row(index))
    position = writer.position()
    writer.close()

    resumed = TelemetryWriter.resume(position, run_id="RUN-TEST-A")
    resumed.append(make_row(3))
    resumed.close()
    rows = read_rows(path)
    assert [row["record_index"] for row in rows] == [0, 1, 2, 3]


def test_a_crash_between_the_fsync_and_the_checkpoint_is_truncated_back(tmp_path):
    path = tmp_path / "t.jsonl"
    writer = TelemetryWriter(path=path, run_id="RUN-TEST-A")
    writer.append(make_row(0))
    position = writer.position()
    writer.append(make_row(1))  # durable, but never checkpointed
    writer.close()

    resumed = TelemetryWriter.resume(position, run_id="RUN-TEST-A")
    assert len(read_rows(path)) == 1, "the uncheckpointed row must not survive twice"
    resumed.append(make_row(1))
    resumed.close()
    rows = read_rows(path)
    assert [row["record_index"] for row in rows] == [0, 1]


def test_a_different_log_at_the_same_offset_is_refused(tmp_path):
    path = tmp_path / "t.jsonl"
    writer = TelemetryWriter(path=path, run_id="RUN-TEST-A")
    writer.append(make_row(0))
    position = writer.position()
    writer.close()
    position = {**position, "last_record_digest": "0" * 64}
    with pytest.raises(Phase17TelemetryError, match="not the same log"):
        TelemetryWriter.resume(position, run_id="RUN-TEST-A")


def test_a_truncated_log_is_refused(tmp_path):
    path = tmp_path / "t.jsonl"
    writer = TelemetryWriter(path=path, run_id="RUN-TEST-A")
    for index in range(3):
        writer.append(make_row(index))
    position = writer.position()
    writer.close()
    path.write_text("")
    with pytest.raises(Phase17TelemetryError, match="truncated"):
        TelemetryWriter.resume(position, run_id="RUN-TEST-A")


# -- the production session's guard cadence ---------------------------------


def _session(tmp_path, **overrides):
    import sys as _sys

    _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "scripts"))
    from run_phase17_training import TrainingSession

    config = tiny_config("RUN-TEST-A")
    return TrainingSession(
        config,
        directory=tmp_path,
        supervisor_mode=MODE_INTEGRATION,
        **overrides,
    )


def test_the_session_feeds_p4_and_p5_on_the_reading_cadence(tmp_path):
    """Without this wiring the two setup-distribution predicates never fire.

    `TandemRunner._supervise` only sees quantities an iteration already
    produced; prefix entropy and flag effective support need a fresh sample of
    the setup policy, so the session has to take one.
    """
    session = _session(tmp_path, reading_every=1, reading_samples=8)
    session.export_hour_zero()
    step = session.step(checkpoint=False)
    session.close()

    supervisor = session.runner.supervisor
    assert step["reading"] is not None
    assert session.readings, "no concentration reading was taken"
    assert supervisor.predicates["P4"].last_evidence or (
        supervisor.predicates["P4"].consecutive == 0
    )
    # Both predicates were *observed*, which is what the wiring has to prove.
    codes = {verdict["code"] for verdict in supervisor.verdicts}
    assert {"P4", "P5"} <= codes
    assert step["row"]["setup"]["concentration"]["measured_this_iteration"]
    assert step["row"]["setup"]["concentration"]["last"]["flag_effective_support"] > 0


def test_the_session_feeds_p6_and_p7_every_iteration(tmp_path):
    session = _session(tmp_path, reading_every=0)
    session.export_hour_zero()
    session.step(checkpoint=False)
    session.close()
    codes = {verdict["code"] for verdict in session.runner.supervisor.verdicts}
    assert {"P6", "P7"} <= codes
    # P6 cannot trip before its first-hour median exists.
    assert not session.first_hour_median_set
    assert session.runner.supervisor.predicates["P6"].trips == 0


def test_hour_zero_must_precede_the_first_update(tmp_path):
    session = _session(tmp_path, reading_every=0)
    session.step(checkpoint=False)
    with pytest.raises(RuntimeError, match="before the first optimizer update"):
        session.export_hour_zero()
    session.close()


def test_a_session_step_writes_a_valid_telemetry_row_and_a_checkpoint(tmp_path):
    session = _session(tmp_path, reading_every=0)
    session.export_hour_zero()
    step = session.step()
    session.close()
    assert step["checkpoint"]["generation"] == 1
    assert step["receipt"]["record_index"] == 0
    rows = read_rows(session.telemetry.path)
    assert len(rows) == 1
    assert rows[0]["system"]["iteration"] == 1
    assert rows[0]["move"]["participant_ledger"]["holds"]
    assert rows[0]["move"]["transitions_harvested"] == session.config.move_budget


def test_the_backlog_alarm_fires_with_room_for_its_own_consecutive_count():
    """P8 needs three consecutive windows; the alarm must leave room for them.

    An alarm too close to the capacity can never complete: Agent 3's queue
    raises at capacity, so the run dies between the first and second reading.
    A rehearsal found this the hard way.
    """
    policy = SetupBudgetPolicy.freeze(games_per_iteration=260.0)
    assert policy.headroom_windows > policy.alarm_consecutive_windows

    # Walk it: from the alarm depth, at a DOUBLED arrival rate, three more
    # windows must still sit under the capacity.
    depth = policy.backlog_alarm_depth
    for _ in range(policy.alarm_consecutive_windows):
        depth += 2 * policy.measured_completions_per_iteration - policy.budget
        assert depth < policy.capacity


def test_an_alarm_that_cannot_complete_is_refused_at_freeze_time():
    from dataclasses import replace

    policy = SetupBudgetPolicy.freeze(games_per_iteration=260.0)
    with pytest.raises(Phase17BudgetError, match="could never complete"):
        replace(policy, backlog_alarm_depth=policy.capacity - 1)


def test_a_window_that_could_overflow_the_queue_is_refused_before_it_starts():
    """Stopping before the window costs nothing; raising inside it costs the window."""
    policy = SetupBudgetPolicy.freeze(games_per_iteration=4.0)
    runner = TandemRunner(
        tiny_config(), budget_policy=policy, supervisor_mode=MODE_INTEGRATION
    )
    # Fill the queue past the point where one more window fits.
    class _Full:
        def __len__(self):
            return policy.capacity

    runner.setup_trainer.queue = _Full()
    with pytest.raises(Phase17RunnerError, match="could reach the capacity"):
        runner.run_iteration()
    assert runner.collector.iteration == 0, "no window was collected"


def test_move_means_fails_loudly_on_an_unprefixed_name():
    """A guard fed a silent 0.0 is a guard that is switched off.

    Agent 2 stores every mean under a `mean_` prefix. `.get("behavior_kl", 0.0)`
    against that mapping returns 0.0 forever, which would make P2 (move KL above
    0.08) and P6 (move entropy collapse) unfireable while the telemetry claimed
    they were live. This caught exactly that.
    """
    from stratego.training.phase17.runner import move_means

    assert move_means({"mean_behavior_kl": 0.02}, "behavior_kl") == pytest.approx(0.02)
    with pytest.raises(Phase17RunnerError, match="has no 'mean_nonsense'"):
        move_means({"mean_behavior_kl": 0.02}, "nonsense")


def test_the_supervisor_sees_the_real_move_kl_not_a_zero(advanced):
    runner, results = advanced
    from stratego.training.phase17.runner import move_means

    for result in results:
        observed = move_means(result.move_update.means, "behavior_kl")
        assert observed > 0.0, "a real PPO update has nonzero behavior KL"
    p2 = [v for v in runner.supervisor.verdicts if v["code"] == "P2"]
    assert p2, "P2 was never observed"
    assert all(v["evidence"]["mean_kl"] > 0.0 for v in p2)


def test_the_session_records_a_nonzero_move_entropy(tmp_path):
    session = _session(tmp_path, reading_every=0)
    session.export_hour_zero()
    step = session.step(checkpoint=False)
    session.close()
    assert step["row"]["move"]["entropy"] > 0.0
    assert step["row"]["move"]["mean_kl"] > 0.0
    assert session.first_hour_move_entropies[-1] > 0.0


def test_the_config_digest_covers_every_field_that_changes_the_run():
    """A digest that misses a field calls two different runs the same run."""
    import dataclasses
    import json

    from stratego.training.phase17.checkpoint import json_digest

    config = TandemConfig(run_id="RUN-TEST-A", total_iterations=640)
    document = json.dumps(config.document())
    for field in dataclasses.fields(config):
        assert field.name.rsplit("_", 1)[-1] in document or field.name in document, (
            f"{field.name} is not represented in TandemConfig.document()"
        )

    # And changing any one of them changes the digest.
    baseline = json_digest(config.document())
    for name, value in (
        ("total_iterations", 641),
        ("move_budget", 32768),
        ("population", 128),
        ("pool_size_per_side", 1000),
        ("setup_budget", 573),
        ("setup_queue_capacity", 4577),
        ("setup_warm_up_minimum", 1145),
        ("setup_max_age_iterations", 9),
        ("setup_minibatch_episodes", 32),
        ("setup_model_seed", 18),
        ("move_minibatch_size", 256),
        ("move_device", "mps"),
    ):
        altered = dataclasses.replace(config, **{name: value})
        assert json_digest(altered.document()) != baseline, f"{name} is not digested"
