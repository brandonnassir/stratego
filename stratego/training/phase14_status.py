"""Phase 14: the authoritative operator status surface.

Specification source: `04_AGENT_4_IMMUTABLE_LAUNCH_PACKAGE.md` sections 2, 3
and 14, over the three monitoring gaps Phase 13 Agent 3 recorded and left for
this agent.

Why this module exists at all
-----------------------------
Agent 3 ran the real system for 90 minutes and found that two of its
operator-facing numbers were not what an operator would read them as:

1. ``games generated`` is a *process-local counter*, restored from the last hot
   checkpoint. A collection that completed but was never checkpointed before a
   crash is lost from it. The rehearsal committed **8,192** games and the
   counter reported **4,096**.
2. ``worker status`` was the configured worker count and a constant string. It
   would not have shown the dead loader worker in the rehearsal's second
   injected failure.

Neither is a training-correctness problem, and nothing here changes training.
What changes is where the number comes from.

The authoritative total is on disk, not in memory
-------------------------------------------------
The rollout store already writes a per-iteration ``manifest.json`` at seal
time carrying ``committed_games``, verified against the scheduled set, the
journals and every payload digest. That file survives a ``SIGKILL``; a counter
in a Python object does not. :func:`committed_game_census` therefore *reads the
store* and reports the process counter beside it, labelled, rather than
instead of it.

Worker health is asked of the operating system
-----------------------------------------------
:func:`loader_health` counts the learner's live ``spawn_main`` children the way
the rehearsal supervisor did — by asking the OS, not the runner. A worker the
runner has forgotten about is exactly the worker whose absence should show up.
Zero live workers is *not* an alarm on its own: the pool exists only while an
iteration trains, so the report says whether the pool is expected to be open.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .phase14_contract import PHASE14_NAMESPACE

PHASE14_STATUS_VERSION = "phase14_status_v1"

#: The command-line fingerprint of a `ProcessPoolExecutor` worker under the
#: `spawn` start method macOS uses, and of the pool's bookkeeping child. The
#: resource tracker is not a loader worker and is never counted as one.
WORKER_COMMAND_MARK = "spawn_main"
RESOURCE_TRACKER_MARK = "resource_tracker"

#: The state names the accepted rollout store moves an iteration through, in
#: order. Everything from SEALED onwards has a manifest.
SEALED_OR_LATER = ("SEALED", "TRAINING", "EVALUATED", "COMMITTED")


class Phase14StatusError(RuntimeError):
    """Raised when a status request cannot be answered honestly."""


def utc_text(unix: "float | None" = None) -> str:
    moment = (
        datetime.now(timezone.utc)
        if unix is None
        else datetime.fromtimestamp(float(unix), tz=timezone.utc)
    )
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ---------------------------------------------------------------------------
# The authoritative committed-game total
# ---------------------------------------------------------------------------


def _iteration_directories(rollout_root, namespace: str) -> list:
    root = Path(rollout_root) / str(namespace)
    if not root.exists():
        return []
    return sorted(
        (path for path in root.glob("iteration_*") if path.is_dir()),
        key=lambda path: path.name,
    )


def _iteration_number(directory: Path) -> int:
    _, _, digits = directory.name.partition("_")
    try:
        return int(digits)
    except ValueError:
        return -1


def _read_json(path: Path) -> "dict | None":
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # A torn file is a fact about the run, not a reason to refuse to
        # report a status; it is reported as an unreadable source below.
        return None


def _journal_committed_games(directory: Path) -> "int | None":
    """Distinct committed game ids in an iteration that has not sealed yet.

    Only used for the single in-flight iteration, where no manifest exists
    yet. Distinct ids rather than line count: a duplicate commit is a defect
    the store's own reconciliation reports, and counting it twice here would
    quietly inflate the authoritative total.
    """
    journal = directory / "journal"
    if not journal.exists():
        return None
    identifiers = set()
    for path in sorted(journal.glob("*.commit.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # The last line of a journal killed mid-write.
                        continue
                    game_id = record.get("phase9_game_id")
                    if game_id:
                        identifiers.add(game_id)
        except OSError:
            continue
    return len(identifiers)


def committed_game_census(rollout_root, namespace: str = PHASE14_NAMESPACE) -> dict:
    """The authoritative committed-game total, read from the rollout store.

    Per iteration, in order of preference:

    ``manifest.json``
        Written by :func:`seal_iteration` only after the scheduled set, the
        journals, every payload digest and every metadata sidecar agree. This
        is the authoritative number and the one the instruction names.
    ``state.json``
        Carries ``committed_games`` on the SEALED transition, and a
        ``seal_attempt`` summary when a seal was refused.
    the journals
        Only for an iteration still COLLECTING, which by definition has no
        manifest. Reported separately as ``in_flight_games`` and excluded from
        the sealed total, because those games are not yet committed *as an
        iteration*.
    """
    root = Path(rollout_root)
    iterations = []
    sealed_total = 0
    in_flight_total = 0
    unreadable = []
    for directory in _iteration_directories(root, namespace):
        number = _iteration_number(directory)
        state_document = _read_json(directory / "state.json") or {}
        state = state_document.get("state", "UNKNOWN")
        manifest = _read_json(directory / "manifest.json")
        games = None
        source = None
        if manifest is not None and manifest.get("committed_games") is not None:
            games = int(manifest["committed_games"])
            source = "manifest.json"
        elif state_document.get("committed_games") is not None:
            games = int(state_document["committed_games"])
            source = "state.json"
        elif isinstance(state_document.get("seal_attempt"), dict):
            attempt = state_document["seal_attempt"]
            if attempt.get("committed_games") is not None:
                games = int(attempt["committed_games"])
                source = "state.json seal_attempt"
        if games is None:
            games = _journal_committed_games(directory)
            source = "journal" if games is not None else None
        if games is None:
            unreadable.append(str(directory))
            continue
        sealed = state in SEALED_OR_LATER
        if sealed:
            sealed_total += games
        else:
            in_flight_total += games
        iterations.append(
            {
                "iteration": number,
                "state": state,
                "committed_games": int(games),
                "sealed": bool(sealed),
                "source": source,
                "sealed_rollout_digest": state_document.get("sealed_rollout_digest"),
            }
        )
    return {
        "status_version": PHASE14_STATUS_VERSION,
        "authoritative": True,
        "source": "rollout store iteration manifests",
        "rollout_root": str(root),
        "namespace": str(namespace),
        "committed_games": int(sealed_total),
        "in_flight_games": int(in_flight_total),
        "iterations_counted": len(iterations),
        "sealed_iterations": sum(1 for entry in iterations if entry["sealed"]),
        "unreadable_iteration_directories": unreadable,
        "iterations": iterations,
    }


def games_report(rollout_root, process_counter: int, namespace: str = PHASE14_NAMESPACE) -> dict:
    """The committed/process pair an operator reads, each labelled.

    Both numbers are reported, and the disagreement is reported too, because a
    silent divergence between them is precisely what the rehearsal found. The
    process counter can only ever be low or equal after a crash, so a positive
    ``process_counter_shortfall`` is the expected shape of the gap and not an
    alarm by itself.
    """
    census = committed_game_census(rollout_root, namespace)
    counter = int(process_counter)
    return {
        "committed_games": census["committed_games"],
        "committed_games_authoritative": True,
        "committed_games_source": census["source"],
        "in_flight_games": census["in_flight_games"],
        "process_counter_games": counter,
        "process_counter_is_diagnostic": True,
        "process_counter_shortfall": census["committed_games"] - counter,
        "census": census,
    }


# ---------------------------------------------------------------------------
# Loader worker health
# ---------------------------------------------------------------------------


def child_processes(pid: int) -> list:
    """Every direct OS child of `pid`, with its state and command line."""
    listed = subprocess.run(
        ["pgrep", "-P", str(int(pid))], capture_output=True, text=True, check=False
    )
    children = []
    for token in listed.stdout.split():
        try:
            child = int(token)
        except ValueError:
            continue
        described = subprocess.run(
            ["ps", "-o", "state=,command=", "-p", str(child)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        state, _, command = described.partition(" ")
        children.append({"pid": child, "state": state, "command": command.strip()})
    return children


def loader_worker_pids(pid: int) -> list:
    """The live CPU loader workers of one training process, as the OS sees them.

    The resource tracker is filtered out by command line: it is a child of the
    same parent and it is not a worker, so counting it would report a healthy
    pool that has lost every worker it had.
    """
    return [
        child["pid"]
        for child in child_processes(pid)
        if WORKER_COMMAND_MARK in child["command"] and child["state"] != "Z"
    ]


def process_alive(pid: "int | None") -> bool:
    """Whether `pid` is a *running* process.

    A killed child its parent has not reaped yet is a zombie, and
    `kill(pid, 0)` succeeds on a zombie. Reporting that as "alive" would make a
    dead learner look healthy in an operator status, so the state is checked
    too.
    """
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False
    state = subprocess.run(
        ["ps", "-o", "state=", "-p", str(int(pid))],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not state:
        return False
    return not state.startswith("Z")


def learner_process_state(log_root, name: str = "phase14_supervisor.jsonl") -> dict:
    """The current learner PID, from the supervisor's own log, and whether it lives.

    Read from the log rather than from a pidfile because the log is written
    anyway, is append-only, and records the launch that a pidfile would only
    summarise. A status read is the one moment when "currently live loader
    workers" means something, so the probe happens here and not in a telemetry
    row written between iterations, when the pool is legitimately closed.
    """
    path = Path(log_root) / name
    launch = None
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "launch":
                launch = record
    if launch is None:
        return {"known": False, "learner_pid": None, "alive": False}
    pid = launch.get("learner_pid")
    return {
        "known": True,
        "learner_pid": pid,
        "alive": process_alive(pid),
        "role": launch.get("role"),
        "launch_timestamp": launch.get("launch_timestamp"),
        "attempt": launch.get("attempt"),
        "checkpoint_selected": launch.get("checkpoint_selected"),
        "supervisor_log": str(path),
    }


def loader_health(
    *,
    pid: "int | None" = None,
    configured_workers: "int | None" = None,
    pool_open: bool = False,
    rebuilds: int = 0,
    last_rebuild_unix: "float | None" = None,
    last_rebuild_reason: str = "",
    max_rebuilds: "int | None" = None,
) -> dict:
    """Live loader-pool health, in the shape section 3 of the task requires.

    ``pool_open`` matters. A `ProcessPoolExecutor` exists only while an
    iteration is training, so zero live workers during a collection is the
    normal state of a healthy run. Reporting "0 of 6" as a fault for the four
    to five minutes of every iteration spent collecting would train an operator
    to ignore the field, which is how a real dead worker gets missed.
    """
    target = os.getpid() if pid is None else int(pid)
    try:
        live = loader_worker_pids(target)
    except OSError:
        live = []
    configured = None if configured_workers is None else int(configured_workers)
    count = len(live)
    if not pool_open:
        status = f"pool idle (collection or between units); {count} live worker processes"
    elif configured is None:
        status = f"{count} live loader workers"
    elif count == configured:
        status = f"healthy: {count} of {configured} loader workers live"
    elif count == 0:
        status = f"no live loader workers; {configured} configured"
    else:
        status = f"degraded: {count} of {configured} loader workers live"
    return {
        # `status` is the path the frozen metric "worker health" already reads.
        # It is now a health sentence rather than a constant string.
        "status": status,
        "observed_pid": target,
        "configured_loader_workers": configured,
        # Retained under its accepted name so nothing that already reads it
        # breaks; `configured_loader_workers` is the unambiguous spelling.
        "loader_workers": configured,
        "live_loader_workers": count,
        "live_loader_worker_pids": list(live),
        "pool_open": bool(pool_open),
        "loader_pool_rebuilds": int(rebuilds),
        "max_loader_pool_rebuilds": max_rebuilds,
        "last_pool_rebuild_unix": last_rebuild_unix,
        "last_pool_rebuild_utc": (
            None if last_rebuild_unix is None else utc_text(last_rebuild_unix)
        ),
        "last_pool_rebuild_reason": str(last_rebuild_reason or ""),
        "execution_model": "single-process bulk-synchronous loop with a CPU loader pool",
    }


def status_semantics() -> dict:
    """What this surface guarantees, in one readable place."""
    return {
        "status_version": PHASE14_STATUS_VERSION,
        "committed_games": "authoritative; read from rollout-store iteration manifests",
        "process_counter_games": "diagnostic only; process-local and lost across a crash",
        "worker_health": "live OS children of the learner, not a configured constant",
        "changes_training": False,
    }
