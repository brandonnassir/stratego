"""The tandem runner's integration behaviour, under operator decision D10.

Everything here runs a real tandem iteration -- real move forward passes, real
boundary targets, a real move epoch, real setup generation and, where a game
completes, five real setup epochs. Nothing is mocked: a mocked timing path
would prove that the mock works.

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
from stratego.training.phase17.runner import (
    PHASE17_SETUP_FAMILY,
    Phase17RunnerError,
    TandemConfig,
    TandemRunner,
)
from stratego.training.phase17.setup_contract import (
    PRODUCTION_RUN_ID,
    SETUP_RECIPE_VERSION,
    setup_alpha,
)
from stratego.training.phase17.setup_model import build_setup_model
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


def test_a_setup_update_consumes_every_episode_that_completed(advanced):
    """D10 section 4: the batch is exactly what arrived, not a fixed quota."""
    runner, results = advanced
    real = [
        result
        for result in results
        if result.setup_update is not None and not result.setup_update.skipped
    ]
    assert real, "no setup update ran; the tandem path is unproven"
    for result in real:
        assert result.setup_update.episodes_consumed == 2 * result.window.games_finished
        # And nothing was carried into the next iteration.
        assert result.buffer_telemetry["depth"] == 0
    # Across the whole run: every episode that was enqueued was consumed once.
    buffer = runner.setup_trainer.queue
    assert buffer.enqueued_count == buffer.consumed_count
    assert buffer.rejected_count == 0


def test_the_setup_pool_is_regenerated_from_the_live_snapshot_every_iteration(
    advanced,
):
    """D10 section 4: 512 fresh samples per side at every global iteration.

    Iteration 1 has nothing to discard. Every iteration after it discards the
    leftovers of the one before, because a stale candidate carries the OLD
    behavior probabilities and reusing it would misattribute the PPO ratio's
    denominator.
    """
    runner, results = advanced
    assert results[0].pool_discarded == 0
    assert all(result.pool_discarded > 0 for result in results[1:])
    # The snapshot each pool is bound to is the global iteration, and the
    # digest is the live raw setup model at the top of that iteration.
    assert runner.provider.snapshot_iteration == results[-1].iteration
    for result in results:
        assert result.provider_telemetry["snapshot_iteration"] == result.iteration


def test_the_setup_alpha_follows_the_shared_global_iteration(advanced):
    """A4-CF6, settled by D10: the same `n` the move schedule reads."""
    _, results = advanced
    for result in results:
        assert result.setup_update.alpha == pytest.approx(setup_alpha(result.iteration))


def test_the_setup_kl_coefficient_is_fixed_and_never_stepped(advanced):
    runner, results = advanced
    for result in results:
        update = result.setup_update
        assert update.behavior_kl_coefficient == 0.1
        if not update.skipped:
            assert {epoch["behavior_kl_coefficient"] for epoch in update.epochs} == {0.1}
    assert not hasattr(runner.setup_trainer, "controller")


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


def test_both_kl_readings_are_reported_as_telemetry(advanced):
    """Nothing consumes them, but the split still has to be readable."""
    _, results = advanced
    real = [
        result
        for result in results
        if result.setup_update is not None and not result.setup_update.skipped
    ]
    for result in real:
        update = result.setup_update
        assert len(update.per_epoch_kl) == 5
        assert update.final_epoch_kl == pytest.approx(update.per_epoch_kl[-1])
        if len(set(update.per_epoch_kl)) > 1:
            assert update.final_epoch_kl != pytest.approx(update.mean_iteration_kl)


def test_an_iteration_with_no_completed_game_skips_explicitly(advanced):
    """The only skip D10 leaves: nothing arrived."""
    _, results = advanced
    skipped = [result for result in results if result.setup_skipped]
    assert skipped, "no iteration skipped; the skip path is unproven"
    for result in skipped:
        assert result.window.games_finished == 0
        assert "no game completed" in result.setup_skip_reason


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
            setup_minibatch_episodes=2,
            move_minibatch_size=32,
        ),
        supervisor_mode=MODE_INTEGRATION,
    )
    for _ in range(2):
        runner.run_iteration()
    with pytest.raises(Phase17RunnerError, match="outside the frozen horizon"):
        runner.run_iteration()


# -- the D10 production identity -------------------------------------------


def test_the_config_and_checkpoint_carry_the_d10_recipe_identity():
    config = tiny_config()
    assert config.recipe == SETUP_RECIPE_VERSION == "phase17_simple_paper_tandem_v1"
    assert config.document()["recipe"] == SETUP_RECIPE_VERSION


def test_production_starts_from_phase_9_plus_a_freshly_random_setup_model():
    """D10 section 3: no rehearsal setup state may enter production."""
    runner = TandemRunner(tiny_config(PRODUCTION_RUN_ID))
    identity = runner.identity_document()
    assert identity["start_identity"]["model_state_digest"] == (
        "f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd"
    )
    # The setup half is the seeded from-scratch model and nothing else.
    fresh = build_setup_model(
        device="cpu", seed=runner.config.setup_model_seed
    )
    from stratego.training.phase9_behavior import state_dict_digest

    assert identity["setup_start_model_state_digest"] == state_dict_digest(fresh)
    assert runner._setup_digest() == runner.setup_start_digest
    assert runner.setup_trainer.updates == 0


def test_production_refuses_an_injected_setup_model():
    """The one way rehearsal weights could reach production, closed."""
    rehearsed = build_setup_model(device="cpu", seed=4242)
    with pytest.raises(Phase17RunnerError, match="build its own setup model"):
        TandemRunner(tiny_config(PRODUCTION_RUN_ID), setup_model=rehearsed)


def test_a_checkpoint_from_the_retired_recipe_is_refused(tmp_path):
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
    payload["recipe"] = "phase17_setup_update_v2"
    write_joint_checkpoint(payload, tmp_path / "old.pt")
    from stratego.training.phase17.checkpoint import Phase17CheckpointError

    with pytest.raises(Phase17CheckpointError, match="written under recipe"):
        read_joint_checkpoint(tmp_path / "old.pt", run_id=config.run_id)


def test_a_checkpoint_carrying_an_adaptive_controller_is_refused(tmp_path):
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
    assert payload["setup_behavior_kl"] == {
        "direction": "reverse_current_given_behavior",
        "coefficient": 0.1,
        "adaptive": False,
    }
    payload["setup_behavior_kl"] = dict(payload["setup_behavior_kl"], adaptive=True)
    write_joint_checkpoint(payload, tmp_path / "adaptive.pt")
    from stratego.training.phase17.checkpoint import Phase17CheckpointError

    with pytest.raises(Phase17CheckpointError, match="ADAPTIVE"):
        read_joint_checkpoint(tmp_path / "adaptive.pt", run_id=config.run_id)


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
        "buffer_depth": result.buffer_telemetry["depth"],
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
    # An integrity failure, which is what D10 leaves as a stop. A statistical
    # reading -- flag support, entropy, EWR -- could not do this any more.
    verdict = runner.supervisor.check_transition_count(harvested=1, budget=400)
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
    assert record["code"] == "I8"
    assert record["evidence"]["transitions_harvested"] == 1
    assert "no hyperparameter was changed" in record["action"]
    # The safe exit produced a loadable checkpoint, not a corpse.
    assert read_joint_checkpoint(
        tmp_path / "stop.pt", run_id=runner.config.run_id, config_digest="cfg"
    )["checkpoint_generation"] == 99
    assert identity.generation == 99


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
    del row["setup"]["completed_episode_buffer"]
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
    baseline = json_digest(config.document())
    for name, value in (
        ("run_id", "RUN-OTHER"),
        ("total_iterations", 641),
        ("move_budget", 32768),
        ("population", 128),
        ("pool_size_per_side", 1000),
        ("setup_minibatch_episodes", 32),
        ("setup_model_seed", 18),
        ("move_minibatch_size", 256),
        ("move_device", "mps"),
        ("setup_device", "mps"),
        ("work_package", "phase18"),
    ):
        altered = dataclasses.replace(config, **{name: value})
        assert json_digest(altered.document()) != baseline, f"{name} is not digested"

    # Every field of the dataclass is covered by the list above, so a field
    # added later fails here rather than silently escaping the digest.
    covered = {
        "run_id",
        "total_iterations",
        "move_budget",
        "population",
        "pool_size_per_side",
        "setup_minibatch_episodes",
        "setup_model_seed",
        "move_minibatch_size",
        "move_device",
        "setup_device",
        "work_package",
    }
    assert {field.name for field in dataclasses.fields(config)} == covered
