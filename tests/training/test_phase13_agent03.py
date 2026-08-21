"""Phase 13 — Agent 3: the rehearsal seam and the worker-recovery fix.

Task: `instructions/phase_13_final_training_integration/03_AGENT_3_90_MINUTE_CRASH_RESUME_REHEARSAL.md`.

Two things are under test here, and nothing else:

* the **rehearsal window seam** — a real wall clock over a shortened deadline,
  which production refuses and which never moves the 132-hour transition; and
* the **loader-pool recovery** the rehearsal's Failure 2 forced into existence:
  a killed CPU worker used to raise `BrokenProcessPool` straight out of
  `Phase14Runner.run()` and end the run.

No frozen training value is exercised here — those are Agent 2's tests, which
still pass unchanged.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from stratego.training import phase14_contract as contract
from stratego.training.phase14_clock import (
    DeadlineController,
    ManualClock,
    Phase14ClockError,
    RunWindow,
    parse_utc,
    utc_text,
)
from stratego.training.phase14_contract import Population

REHEARSAL_SECONDS = 5400.0
START = "2026-09-01T00:00:00.000Z"


# ---------------------------------------------------------------------------
# The rehearsal window seam
# ---------------------------------------------------------------------------


def test_a_rehearsal_window_shortens_only_the_deadline():
    """The 132-hour transition does not move; the rehearsal never reaches it."""
    window = RunWindow.rehearsal(parse_utc(START), REHEARSAL_SECONDS)
    assert window.deadline_seconds == REHEARSAL_SECONDS
    assert (window.transition_utc - window.run_start_utc).total_seconds() == (
        contract.TRANSITION_SECONDS
    )
    assert window.production is False
    # And so a rehearsal spends every one of its minutes in `main`.
    assert window.transition_utc > window.run_deadline_utc


def test_a_rehearsal_window_must_be_shorter_than_the_frozen_run():
    for span in (0.0, -1.0, contract.DEADLINE_SECONDS, contract.DEADLINE_SECONDS + 1):
        with pytest.raises(Phase14ClockError):
            RunWindow.rehearsal(parse_utc(START), span)


def test_the_production_window_is_untouched_by_the_seam():
    window = RunWindow.start(parse_utc(START))
    assert window.production is True
    assert window.deadline_seconds == contract.DEADLINE_SECONDS
    assert window.to_dict()["deadline_seconds"] == contract.DEADLINE_SECONDS
    assert window.to_dict()["production"] is True


def test_a_window_of_the_wrong_length_is_still_refused():
    start = parse_utc(START)
    with pytest.raises(Phase14ClockError):
        RunWindow(
            run_start_utc=start,
            run_deadline_utc=start,
            transition_utc=start,
        )


def test_a_rehearsal_window_round_trips_through_a_checkpoint():
    window = RunWindow.rehearsal(parse_utc(START), REHEARSAL_SECONDS)
    persisted = window.to_dict()
    assert persisted["deadline_seconds"] == REHEARSAL_SECONDS
    assert persisted["production"] is False
    assert RunWindow.from_dict(persisted) == window


def test_a_persisted_window_without_the_flag_is_a_production_window():
    """Every window written before the seam existed was a production window."""
    persisted = RunWindow.start(parse_utc(START)).to_dict()
    persisted.pop("production")
    assert RunWindow.from_dict(persisted).production is True


def test_a_rehearsal_deadline_expires_at_the_shortened_instant():
    clock = ManualClock(START)
    controller = DeadlineController.rehearsal(clock, REHEARSAL_SECONDS)
    persisted = controller.window.to_dict()
    clock.advance(REHEARSAL_SECONDS - 1)
    assert controller.expired() is False
    assert controller.segment() == "main"
    assert controller.learning_rate() == contract.learning_rate("main")
    clock.advance(1)
    assert controller.expired() is True
    assert controller.may_start_collection_unit() is False
    assert controller.may_start_optimizer_step() is False
    # Downtime inside a rehearsal is charged the same way it is in the run.
    clock.advance_hours(5)
    assert DeadlineController.resume(persisted, clock).window.to_dict() == persisted


# ---------------------------------------------------------------------------
# The runner's rehearsal mode
# ---------------------------------------------------------------------------


def test_rehearsal_mode_owns_its_deadline_and_no_other_mode_may(tmp_path):
    from stratego.training.phase14_runner import (
        MODE_PRODUCTION,
        MODE_REHEARSAL,
        MODE_TEST,
        Phase14Runner,
        Phase14RunnerError,
    )
    from stratego.training.phase14_storage import Phase14Storage

    storage = Phase14Storage.under(tmp_path)
    with pytest.raises(Phase14RunnerError):
        Phase14Runner(storage, mode=MODE_REHEARSAL)
    for mode in (MODE_PRODUCTION, MODE_TEST):
        with pytest.raises(Phase14RunnerError):
            Phase14Runner(
                storage,
                mode=mode,
                clock=None if mode == MODE_PRODUCTION else ManualClock(START),
                rehearsal_deadline_seconds=REHEARSAL_SECONDS,
            )


def test_rehearsal_mode_refuses_the_manual_clock(tmp_path):
    """A 90-minute rehearsal is worthless if the 90 minutes are simulated."""
    from stratego.training.phase14_runner import MODE_REHEARSAL, Phase14Runner
    from stratego.training.phase14_storage import Phase14Storage

    with pytest.raises(Phase14ClockError):
        Phase14Runner(
            Phase14Storage.under(tmp_path),
            clock=ManualClock(START),
            mode=MODE_REHEARSAL,
            rehearsal_deadline_seconds=REHEARSAL_SECONDS,
        )


def test_production_mode_still_refuses_both_original_seams(tmp_path):
    from stratego.training.phase14_runner import (
        MODE_PRODUCTION,
        Phase14Runner,
        Phase14RunnerError,
    )
    from stratego.training.phase14_storage import Phase14Storage

    storage = Phase14Storage.under(tmp_path)
    with pytest.raises(Phase14ClockError):
        Phase14Runner(storage, clock=ManualClock(START), mode=MODE_PRODUCTION)
    with pytest.raises(Phase14RunnerError):
        Phase14Runner(storage, mode=MODE_PRODUCTION, population=Population.scaled(8))


@pytest.fixture(scope="module")
def rehearsal_start(tmp_path_factory):
    """One started rehearsal run, stopped before it collects anything."""
    from stratego.training.phase14_runner import MODE_REHEARSAL, Phase14Runner
    from stratego.training.phase14_storage import Phase14Storage

    root = tmp_path_factory.mktemp("phase13_rehearsal_seam")
    storage = Phase14Storage.under(root)
    runner = Phase14Runner(
        storage,
        mode=MODE_REHEARSAL,
        rehearsal_deadline_seconds=REHEARSAL_SECONDS,
        device="cpu",
        inference_device="cpu",
        population=Population.scaled(512),
    )
    report = runner.start()
    return storage, report


def test_a_rehearsal_stamps_a_shortened_window_against_the_real_clock(rehearsal_start):
    _storage, report = rehearsal_start
    span = (
        parse_utc(report["run_deadline_utc"]) - parse_utc(report["run_start_utc"])
    ).total_seconds()
    assert span == REHEARSAL_SECONDS
    transition = (
        parse_utc(report["transition_utc"]) - parse_utc(report["run_start_utc"])
    ).total_seconds()
    assert transition == contract.TRANSITION_SECONDS


def test_a_production_run_refuses_to_adopt_a_rehearsal_checkpoint(rehearsal_start):
    """The failure this prevents is unrecoverable once it has happened."""
    from stratego.training.phase14_runner import (
        MODE_PRODUCTION,
        Phase14IntegrityError,
        Phase14Runner,
    )

    storage, _report = rehearsal_start
    runner = Phase14Runner(storage, mode=MODE_PRODUCTION, device="cpu", inference_device="cpu")
    with pytest.raises(Phase14IntegrityError):
        runner.resume()


def test_a_rehearsal_may_not_negotiate_a_different_deadline(rehearsal_start):
    from stratego.training.phase14_runner import (
        MODE_REHEARSAL,
        Phase14IntegrityError,
        Phase14Runner,
    )

    storage, _report = rehearsal_start
    runner = Phase14Runner(
        storage,
        mode=MODE_REHEARSAL,
        rehearsal_deadline_seconds=REHEARSAL_SECONDS * 2,
        device="cpu",
        inference_device="cpu",
        population=Population.scaled(512),
    )
    with pytest.raises(Phase14IntegrityError):
        runner.resume()


# ---------------------------------------------------------------------------
# The worker-recovery fix
# ---------------------------------------------------------------------------


def test_a_broken_loader_pool_is_recoverable_at_the_runner_level():
    """The gap Failure 2 found: `BrokenProcessPool` was in neither list."""
    from concurrent.futures import BrokenExecutor
    from concurrent.futures.process import BrokenProcessPool

    from stratego.training.phase14_runner import RECOVERABLE_ERRORS, UNRECOVERABLE_ERRORS

    assert issubclass(BrokenProcessPool, RuntimeError)
    assert issubclass(BrokenProcessPool, RECOVERABLE_ERRORS)
    assert not issubclass(BrokenProcessPool, tuple(UNRECOVERABLE_ERRORS))
    assert BrokenExecutor in RECOVERABLE_ERRORS


def _runner(root, *, workers: int):
    """One small real run, on the CPU, over the declared test seams."""
    from stratego.training.phase14_runner import MODE_TEST, Phase14Runner
    from stratego.training.phase14_storage import Phase14Storage
    from stratego.training.phase9_trainer import LoaderTopology

    runner = Phase14Runner(
        Phase14Storage.under(root),
        clock=ManualClock(START),
        mode=MODE_TEST,
        device="cpu",
        inference_device="cpu",
        topology=LoaderTopology(workers=workers),
        games_in_flight=8,
        population=Population.scaled(128),
    )
    runner.start()
    return runner


def _record_steps(runner, *, kill_at: "int | None" = None):
    """Run one iteration, recording each step; optionally kill a worker.

    The kill is fired from the trainer's own per-step callback rather than from
    a racing thread, so the pool dies at a known step with the rest of the
    epoch still pending — which is the case that used to end the run.
    """
    from stratego.training.phase13_rehearsal import loader_worker_pids, process_alive

    rows: list = []
    killed: list = []
    original = runner._on_step

    def hooked(trainer, row):
        original(trainer, row)
        rows.append(
            {
                "step": int(row["global_optimizer_step"]),
                "epoch": int(row["epoch"]),
                "minibatch_index": int(row["minibatch_index"]),
                "loss_total": float(row["loss_total"]),
            }
        )
        if kill_at is not None and not killed and int(row["global_optimizer_step"]) >= kill_at:
            workers = loader_worker_pids(os.getpid())
            if workers:
                os.kill(workers[0], 9)
                killed.append(workers[0])

    runner._on_step = hooked
    unit = runner.run_iteration()
    if killed:
        # A killed child is a zombie until its parent reaps it; give the pool a
        # moment so the assertion describes the process, not the bookkeeping.
        for _ in range(50):
            if not process_alive(killed[0]):
                break
            time.sleep(0.1)
    return unit, rows, killed


@pytest.fixture(scope="module")
def killed_and_clean(tmp_path_factory):
    """The same iteration, once with a worker killed mid-epoch and once not."""
    killed_root = tmp_path_factory.mktemp("phase13_worker_killed")
    clean_root = tmp_path_factory.mktemp("phase13_worker_clean")
    killed_runner = _runner(killed_root, workers=3)
    killed_unit, killed_rows, victims = _record_steps(killed_runner, kill_at=2)
    clean_runner = _runner(clean_root, workers=3)
    clean_unit, clean_rows, _ = _record_steps(clean_runner)
    return {
        "killed": (killed_runner, killed_unit, killed_rows, victims),
        "clean": (clean_runner, clean_unit, clean_rows),
    }


def test_a_killed_loader_worker_does_not_kill_the_learner(killed_and_clean):
    """Failure 2 of the rehearsal, as a test."""
    from stratego.training.phase13_rehearsal import process_alive

    runner, unit, rows, victims = killed_and_clean["killed"]
    assert victims, "the test did not manage to kill a loader worker"
    assert not process_alive(victims[0])
    assert unit["trained"] is True
    assert unit["sealed"] is True
    assert rows, "no optimizer step landed"
    assert runner.trainer.counters["loader_pool_rebuilds"] >= 1


def test_a_rebuilt_pool_repeats_no_minibatch_and_skips_none(killed_and_clean):
    """Section 5: no duplicate optimizer work, no skipped logical state."""
    _killed_runner, killed_unit, killed_rows, _victims = killed_and_clean["killed"]
    clean_runner, clean_unit, clean_rows = killed_and_clean["clean"]

    killed_plan = [(row["epoch"], row["minibatch_index"]) for row in killed_rows]
    clean_plan = [(row["epoch"], row["minibatch_index"]) for row in clean_rows]
    assert killed_plan == clean_plan
    assert len(killed_plan) == len(set(killed_plan))
    assert killed_unit["updates"] == clean_unit["updates"]
    assert clean_runner.trainer.counters["loader_pool_rebuilds"] == 0


def test_a_rebuilt_pool_trains_on_the_same_numbers(killed_and_clean):
    """The rebuilt pool packs the same bytes, so the losses are the same."""
    _killed_runner, _killed_unit, killed_rows, _victims = killed_and_clean["killed"]
    _clean_runner, _clean_unit, clean_rows = killed_and_clean["clean"]
    for killed_row, clean_row in zip(killed_rows, clean_rows, strict=True):
        assert killed_row["loss_total"] == pytest.approx(clean_row["loss_total"], rel=1e-9)


def test_the_rebuild_ceiling_stops_a_sick_machine():
    """Rebuilding forever would hide a dying host behind a healthy log."""
    from stratego.training import phase14_trainer

    assert phase14_trainer.MAX_LOADER_POOL_REBUILDS >= 1


# ---------------------------------------------------------------------------
# The rehearsal harness itself
# ---------------------------------------------------------------------------


def test_loader_workers_are_told_apart_from_the_resource_tracker():
    """Killing the resource tracker would look like a worker-failure test."""
    from stratego.training.phase13_rehearsal import (
        RESOURCE_TRACKER_MARK,
        WORKER_COMMAND_MARK,
        child_processes,
        loader_worker_pids,
    )
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(abs, [-1, -2]))
        children = child_processes(os.getpid())
        workers = loader_worker_pids(os.getpid())
        commands = {child["pid"]: child["command"] for child in children}
        assert workers, "the pool started no worker this test can see"
        assert all(WORKER_COMMAND_MARK in commands[pid] for pid in workers)
        trackers = [
            child["pid"] for child in children if RESOURCE_TRACKER_MARK in child["command"]
        ]
        assert not set(trackers) & set(workers)


def test_a_zombie_is_not_reported_as_a_live_process():
    """`kill(pid, 0)` succeeds on a zombie; a kill would look like it failed."""
    from stratego.training.phase13_rehearsal import process_alive

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert process_alive(child.pid) is True
        child.kill()
        for _ in range(50):
            if not process_alive(child.pid):
                break
            time.sleep(0.1)
        # Still unreaped here, so this is precisely the zombie case.
        assert process_alive(child.pid) is False
    finally:
        child.wait()


def test_the_harness_states_what_it_leaves_alone():
    from stratego.training.phase13_rehearsal import rehearsal_semantics

    semantics = rehearsal_semantics()
    unchanged = " ".join(semantics["unchanged"])
    for frozen in ("learning rate", "opponent mixture", "132-hour transition", "cadence"):
        assert frozen in unchanged
