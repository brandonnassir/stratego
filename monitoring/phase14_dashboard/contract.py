"""The frozen Phase 14 values the dashboard displays, mirrored deliberately.

Why these are copied rather than imported
-----------------------------------------
Importing any ``stratego.training.phase14_*`` module pulls in
``phase14_contract`` -> ``phase9_contract`` -> ``warmstart_contract`` ->
``stratego.model`` -> ``torch``: about 205 MB of resident memory and a CUDA/MPS
capable runtime, inside a process whose entire job is to read seven JSON files
and print them. The task forbids importing the model for dashboard purposes,
and the cheapest way to *guarantee* that is to make it impossible: this package
imports nothing from :mod:`stratego` at all, so no refactor upstream can
quietly give the monitor a model.

The cost of that choice is duplication, and duplication of a frozen number is
how a dashboard ends up confidently displaying last month's deadline.
:mod:`tests.monitoring.test_phase14_dashboard` therefore imports the real
frozen modules — in a test process, where torch is free — and asserts every
value below equals its source. If Phase 14's contract ever moves, the test
fails rather than the dashboard lying.

Nothing here is ever written anywhere. These are display values.
"""

from __future__ import annotations

# -- identity ---------------------------------------------------------------

DASHBOARD_VERSION = "phase14_dashboard_v1"

#: mirrors stratego.training.phase14_contract.PHASE14_NAMESPACE
PHASE14_NAMESPACE = "phase14"

# -- storage layout ---------------------------------------------------------

#: mirrors stratego.training.phase14_contract.EXTERNAL_VOLUME
EXTERNAL_VOLUME = "/Volumes/Brandon_Washington"
#: mirrors stratego.training.phase14_contract.EXTERNAL_RUN_DIRECTORY
EXTERNAL_RUN_DIRECTORY = "/Volumes/Brandon_Washington/stratego_phase14"
#: mirrors stratego.training.phase14_contract.HOT_CHECKPOINT_DIRECTORY
HOT_CHECKPOINT_DIRECTORY = "checkpoints/phase14/hot"
#: mirrors stratego.training.phase14_contract.ROLLOUT_SUBDIRECTORY
ROLLOUT_SUBDIRECTORY = "rollouts"
#: mirrors stratego.training.phase14_contract.DURABLE_ARCHIVE_SUBDIRECTORY
DURABLE_ARCHIVE_SUBDIRECTORY = "archive"
#: mirrors stratego.training.phase14_contract.LOG_SUBDIRECTORY
LOG_SUBDIRECTORY = "logs"
#: mirrors stratego.training.phase14_contract.EVALUATION_SUBDIRECTORY
EVALUATION_SUBDIRECTORY = "evaluations"

RUN_STATE_FILENAME = "phase14_run_state.json"
#: mirrors stratego.training.phase14_launch.EMERGENCY_STOP_FILENAME
EMERGENCY_STOP_FILENAME = "phase14_emergency_stop.json"
#: mirrors stratego.training.phase14_launch.INTEGRITY_FAILURE_FILENAME
INTEGRITY_FAILURE_FILENAME = "phase14_integrity_failure.json"
TELEMETRY_FILENAME = "phase14_telemetry.jsonl"
SUPERVISOR_FILENAME = "phase14_supervisor.jsonl"
CANDIDATE_LEDGER_FILENAME = "phase14_candidate_ledger.json"

# -- the immutable wall clock ----------------------------------------------

#: mirrors stratego.training.phase14_contract.TOTAL_HOURS
TOTAL_HOURS = 168
#: mirrors stratego.training.phase14_contract.MAIN_SEGMENT_HOURS
MAIN_SEGMENT_HOURS = 132
#: mirrors stratego.training.phase14_contract.LATE_SEGMENT_HOURS
LATE_SEGMENT_HOURS = 36
#: mirrors stratego.training.phase14_contract.TRANSITION_SECONDS
TRANSITION_SECONDS = MAIN_SEGMENT_HOURS * 3600
#: mirrors stratego.training.phase14_contract.DEADLINE_SECONDS
DEADLINE_SECONDS = TOTAL_HOURS * 3600

#: mirrors stratego.training.phase14_contract.SEGMENT_MAIN / SEGMENT_LATE
SEGMENT_MAIN = "main"
SEGMENT_LATE = "late"

#: mirrors stratego.training.phase14_contract.MAIN_LEARNING_RATE
MAIN_LEARNING_RATE = 7.5e-5
#: mirrors stratego.training.phase14_contract.LATE_LEARNING_RATE
LATE_LEARNING_RATE = 3.75e-5

# -- cadences ---------------------------------------------------------------

#: mirrors stratego.training.phase14_contract.HOT_CHECKPOINT_SECONDS
HOT_CHECKPOINT_SECONDS = 15 * 60
#: mirrors stratego.training.phase14_contract.ARCHIVE_CADENCE_SECONDS
ARCHIVE_CADENCE_SECONDS = 2 * 3600
#: mirrors stratego.training.phase14_contract.CANDIDATE_CADENCE_SECONDS
CANDIDATE_CADENCE_SECONDS = 6 * 3600
#: mirrors stratego.training.phase14_contract.CANDIDATE_HOURS
CANDIDATE_HOURS = tuple(range(0, TOTAL_HOURS + 1, 6))

#: mirrors stratego.training.phase14_contract.GAMES_PER_ITERATION
GAMES_PER_ITERATION = 2048
#: mirrors stratego.training.phase14_contract.STORAGE_RESERVE_GIB
STORAGE_RESERVE_GIB = 120
#: mirrors stratego.training.phase14_trainer.MAX_LOADER_POOL_REBUILDS
MAX_LOADER_POOL_REBUILDS = 16

# -- reading the rollout store ---------------------------------------------

#: mirrors stratego.training.phase14_status.SEALED_OR_LATER
SEALED_OR_LATER = ("SEALED", "TRAINING", "EVALUATED", "COMMITTED")
#: mirrors stratego.training.phase14_status.WORKER_COMMAND_MARK
WORKER_COMMAND_MARK = "spawn_main"

# -- how the operator should read an age -----------------------------------

#: A hot checkpoint is written on a 900 s cadence, but only at a unit boundary,
#: so a collection (~300 s) can legitimately sit between two writes. The
#: runbook's own reading is "up to one cadence plus a collection"; anything
#: past this is worth a look and is *not* on its own an alarm.
HOT_CHECKPOINT_CONCERN_SECONDS = 1300

#: The learner publishes one telemetry row per iteration (~21 minutes
#: measured). A row older than two iterations means the learner has not
#: completed a unit in that time, which is the thing worth seeing.
TELEMETRY_STALE_SECONDS = 45 * 60

GIB = 1024 ** 3


def segment_for_elapsed(elapsed_seconds: float) -> str:
    """mirrors stratego.training.phase14_clock.segment_for_elapsed."""
    return SEGMENT_LATE if float(elapsed_seconds) >= TRANSITION_SECONDS else SEGMENT_MAIN


def learning_rate_for_elapsed(elapsed_seconds: float) -> float:
    """mirrors stratego.training.phase14_clock.learning_rate_for_elapsed."""
    return (
        MAIN_LEARNING_RATE
        if segment_for_elapsed(elapsed_seconds) == SEGMENT_MAIN
        else LATE_LEARNING_RATE
    )


def mirrored_values() -> dict:
    """Every mirrored constant, for the test that checks them against source."""
    return {
        "PHASE14_NAMESPACE": PHASE14_NAMESPACE,
        "EXTERNAL_VOLUME": EXTERNAL_VOLUME,
        "EXTERNAL_RUN_DIRECTORY": EXTERNAL_RUN_DIRECTORY,
        "HOT_CHECKPOINT_DIRECTORY": HOT_CHECKPOINT_DIRECTORY,
        "ROLLOUT_SUBDIRECTORY": ROLLOUT_SUBDIRECTORY,
        "DURABLE_ARCHIVE_SUBDIRECTORY": DURABLE_ARCHIVE_SUBDIRECTORY,
        "LOG_SUBDIRECTORY": LOG_SUBDIRECTORY,
        "EVALUATION_SUBDIRECTORY": EVALUATION_SUBDIRECTORY,
        "TOTAL_HOURS": TOTAL_HOURS,
        "MAIN_SEGMENT_HOURS": MAIN_SEGMENT_HOURS,
        "LATE_SEGMENT_HOURS": LATE_SEGMENT_HOURS,
        "TRANSITION_SECONDS": TRANSITION_SECONDS,
        "DEADLINE_SECONDS": DEADLINE_SECONDS,
        "SEGMENT_MAIN": SEGMENT_MAIN,
        "SEGMENT_LATE": SEGMENT_LATE,
        "MAIN_LEARNING_RATE": MAIN_LEARNING_RATE,
        "LATE_LEARNING_RATE": LATE_LEARNING_RATE,
        "HOT_CHECKPOINT_SECONDS": HOT_CHECKPOINT_SECONDS,
        "ARCHIVE_CADENCE_SECONDS": ARCHIVE_CADENCE_SECONDS,
        "CANDIDATE_CADENCE_SECONDS": CANDIDATE_CADENCE_SECONDS,
        "CANDIDATE_HOURS": CANDIDATE_HOURS,
        "GAMES_PER_ITERATION": GAMES_PER_ITERATION,
        "STORAGE_RESERVE_GIB": STORAGE_RESERVE_GIB,
        "MAX_LOADER_POOL_REBUILDS": MAX_LOADER_POOL_REBUILDS,
        "SEALED_OR_LATER": SEALED_OR_LATER,
        "WORKER_COMMAND_MARK": WORKER_COMMAND_MARK,
        "EMERGENCY_STOP_FILENAME": EMERGENCY_STOP_FILENAME,
        "INTEGRITY_FAILURE_FILENAME": INTEGRITY_FAILURE_FILENAME,
    }
