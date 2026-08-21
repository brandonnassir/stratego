"""Phase 13 Agent 3: the 90-minute crash/resume rehearsal harness.

Specification source: `03_AGENT_3_90_MINUTE_CRASH_RESUME_REHEARSAL.md`.

What this module is, and what it is not
---------------------------------------
It is a *supervisor*. It launches the real Phase 14 runner in a real child
process, watches it from outside, kills it the way a machine kills a process,
restarts it through the production recovery path, kills one of its CPU loader
workers, and samples storage and status the whole time. It contains no training
logic, no schedule logic and no configuration: everything it starts is the
frozen Phase 14 system, reached through :mod:`phase14_runner`.

It deliberately observes from outside the training process
-----------------------------------------------------------
Every fact this module records about the run — optimizer step, pool, archive,
window, storage — is read back off disk from the artifacts the production path
writes (hot checkpoints, the telemetry JSONL, the run manifest), never from the
runner object. That is the only way the evidence survives a ``SIGKILL``, and it
is also the honest test: it proves the run's *persisted* state is enough to
resume, rather than proving that a live Python object knew what it was doing.

The one deviation from the 168-hour configuration
--------------------------------------------------
The deadline. `MODE_REHEARSAL` stamps a shortened
:meth:`RunWindow.rehearsal` window against the real system clock; the frozen
132-hour transition, the 15-minute/2-hour/6-hour cadences, the learning rates,
both mixtures, the objective, the setup source and the storage policy are
untouched. Downtime inside the rehearsal counts against the rehearsal deadline
exactly as it counts against the 168 hours.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REHEARSAL_VERSION = "phase13_rehearsal_v1"

#: How often the supervisor samples the run from outside.
POLL_SECONDS = 5.0

#: How long a killed child is given to actually disappear before we complain.
REAP_SECONDS = 20.0


def utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ---------------------------------------------------------------------------
# The event log
# ---------------------------------------------------------------------------


class EventLog:
    """An append-and-flush JSONL log. Written so a SIGKILL cannot lose it.

    Every record is flushed and `fsync`-ed before the call returns, because the
    interesting records are precisely the ones written moments before something
    is killed.
    """

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields) -> dict:
        record = {"utc": utc_now_text(), "event": str(event), **fields}
        line = json.dumps(record, sort_keys=True, default=str) + "\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def read(self) -> list:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line is itself evidence, not a crash.
                records.append({"event": "unparseable", "raw": line})
        return records


# ---------------------------------------------------------------------------
# Reading the run from outside the run
# ---------------------------------------------------------------------------


def hot_checkpoint_files(hot_root) -> list:
    root = Path(hot_root)
    if not root.exists():
        return []
    return sorted(path for path in root.glob("hot_*.pt"))


def read_latest_valid(hot_root) -> "tuple | None":
    """The newest hot checkpoint that *validates*, read the production way."""
    from .phase14_checkpoint import HotCheckpointRing

    return HotCheckpointRing(Path(hot_root)).load_latest()


def observed_state(hot_root) -> dict:
    """The resume-relevant state, read off disk. Empty dict if none yet.

    The pool and the archive are *rebuilt through their own classes* rather
    than read as raw dictionaries, so the digests recorded here are the same
    digests the runner compares on resume, and `f(k)` is re-derived from the
    persisted archive on every sample.
    """
    from .phase14_pool import ActivePool, HistoricalArchive

    loaded = read_latest_valid(hot_root)
    if loaded is None:
        return {}
    path, payload = loaded
    trainer = payload["trainer_state"]
    schedule = payload["schedule_state"]
    progress = schedule.get("progress", {})
    archive = HistoricalArchive.from_dict(payload["historical_archive_state"])
    pool = ActivePool.from_dict(payload["active_historical_pool"])
    stat = Path(path).stat()
    return {
        "checkpoint": str(path),
        "checkpoint_bytes": stat.st_size,
        "checkpoint_mtime_unix": stat.st_mtime,
        "global_optimizer_step": trainer["global_optimizer_step"],
        "examples_consumed": trainer.get("examples_consumed"),
        "kl_beta": trainer.get("kl_beta"),
        "cursor": trainer.get("cursor"),
        "rl_iteration": trainer.get("rl_iteration"),
        "model_state_digest": payload["model_state_digest"],
        "optimizer_param_groups": len(payload["optimizer_state"].get("param_groups", [])),
        "optimizer_state_entries": len(payload["optimizer_state"].get("state", {})),
        "ema_state": payload.get("ema_state"),
        "rng_streams": sorted(
            key for key in payload.get("rng", {}) if key not in ("note", "device")
        ),
        "rng_digest": _rng_digest(payload.get("rng", {})),
        "run_window": payload["run_window"],
        "segment": schedule.get("segment"),
        "learning_rate": schedule.get("learning_rate"),
        "elapsed_seconds": schedule.get("elapsed_seconds"),
        "iteration": progress.get("iteration"),
        "iterations_completed": progress.get("iterations_completed"),
        "last_hot_index": progress.get("last_hot_index"),
        "last_archive_mark": progress.get("last_archive_mark"),
        "last_candidate_index": progress.get("last_candidate_index"),
        "progress": progress,
        "pool_members": list(pool.members()),
        "pool_digest": pool.digest(),
        "pool_recomputed_digest": ActivePool.for_archive(archive).digest(),
        "archive_k": archive.k,
        "archive_digest": archive.digest(),
        "shard_cursor": payload.get("shard_cursor", {}),
        "candidate_state": payload.get("candidate_evaluation_state", {}),
        "storage_state": {
            key: payload.get("storage_state", {}).get(key)
            for key in ("free_bytes", "free_gib", "used_bytes")
        },
        "mode": schedule.get("mode"),
        "clock": schedule.get("clock"),
        "population_schedule_state": payload.get("population_schedule_state", {}),
    }


def _rng_digest(rng: dict) -> str:
    """One comparable fingerprint of every persisted RNG stream.

    Hashed rather than stored: the torch CPU generator state alone is a
    multi-kilobyte byte string, and what a resume check needs is whether the
    streams came back identical, not what they were.
    """
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(rng):
        if key in ("note", "device"):
            continue
        digest.update(key.encode())
        digest.update(repr(rng[key]).encode())
    return digest.hexdigest()


def directory_bytes(path) -> int:
    total = 0
    root = Path(path)
    if not root.exists():
        return 0
    for entry in root.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def free_bytes(path) -> int:
    usage = os.statvfs(str(path))
    return usage.f_bavail * usage.f_frsize


# ---------------------------------------------------------------------------
# The child process
# ---------------------------------------------------------------------------


#: The command-line fingerprint of a `ProcessPoolExecutor` worker under the
#: `spawn` start method that macOS uses. The other child a pool creates is
#: `multiprocessing.resource_tracker`, which is bookkeeping and not a worker —
#: killing it would prove nothing about surviving a lost CPU worker.
WORKER_COMMAND_MARK = "spawn_main"
RESOURCE_TRACKER_MARK = "resource_tracker"


def child_processes(pid: int) -> list:
    """Every direct OS child of `pid`, with its state and command line."""
    listed = subprocess.run(
        ["pgrep", "-P", str(pid)], capture_output=True, text=True, check=False
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
    """The CPU loader workers of one training process, as the OS sees them.

    `ProcessPoolExecutor` workers are ordinary OS children of the training
    process, and they exist only while an iteration is training. Asking the OS
    rather than the runner is deliberate: a worker the runner has forgotten
    about is exactly the kind of worker whose death we want to survive.

    The resource tracker is filtered out by command line. It is a child of the
    same parent and it is *not* a worker; killing it would look like a
    successful worker-failure test while testing nothing.
    """
    return [
        child["pid"]
        for child in child_processes(pid)
        if WORKER_COMMAND_MARK in child["command"] and child["state"] != "Z"
    ]


def process_alive(pid: int) -> bool:
    """Whether `pid` is a *running* process.

    A killed child that its parent has not reaped yet is a zombie, and
    `kill(pid, 0)` succeeds on a zombie. Reporting that as "alive" would turn a
    successful worker kill into a failed one in the record, so the process
    state is checked as well.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    state = subprocess.run(
        ["ps", "-o", "state=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    if not state:
        return False
    return not state.startswith("Z")


@dataclass
class ChildProcess:
    """One launched training process, in its own process group."""

    popen: subprocess.Popen
    role: str
    launched_utc: str
    stdout_path: Path

    @property
    def pid(self) -> int:
        return self.popen.pid

    def alive(self) -> bool:
        return self.popen.poll() is None

    def kill_group(self, sig=signal.SIGKILL) -> dict:
        """Force-terminate the whole group: learner and its loader workers.

        A real process-level failure takes the workers with it, so the
        rehearsal's Failure 1 does too. Killing only the learner would leave
        orphaned workers holding the shard files and would make the restart
        easier than the one the run will actually face.
        """
        workers = loader_worker_pids(self.pid)
        try:
            os.killpg(os.getpgid(self.pid), sig)
        except (ProcessLookupError, PermissionError) as error:
            return {"killed": False, "error": f"{type(error).__name__}: {error}"}
        return {
            "killed": True,
            "signal": int(sig),
            "pid": self.pid,
            "loader_workers": workers,
        }

    def wait_gone(self, seconds: float = REAP_SECONDS) -> dict:
        deadline = time.monotonic() + float(seconds)
        while time.monotonic() < deadline:
            if self.popen.poll() is not None:
                break
            time.sleep(0.1)
        returncode = self.popen.poll()
        leftover = [pid for pid in loader_worker_pids(self.pid) if process_alive(pid)]
        return {
            "returncode": returncode,
            "gone": returncode is not None,
            "orphaned_workers": leftover,
        }


def launch_segment(
    *,
    python: str,
    script: Path,
    repository: Path,
    arguments: list,
    role: str,
    stdout_path: Path,
) -> ChildProcess:
    """Start one training process in its own session, output to a file.

    `start_new_session=True` gives the child its own process group, which is
    what makes "kill the training process and everything it spawned" a single
    signal rather than a race against a worker that has not been reaped yet.
    """
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    handle = open(stdout_path, "ab")
    popen = subprocess.Popen(
        [python, str(script), *arguments],
        cwd=str(repository),
        env=environment,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return ChildProcess(
        popen=popen, role=role, launched_utc=utc_now_text(), stdout_path=stdout_path
    )


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass
class RehearsalPlan:
    """Everything the supervisor decides in advance, in one object."""

    deadline_seconds: float = 5400.0
    crash_at_seconds: float = 1800.0
    worker_kill_at_seconds: float = 3300.0
    poll_seconds: float = POLL_SECONDS
    #: Grace after the deadline for the child to finalize on its own.
    finalize_grace_seconds: float = 900.0
    device: str = "mps"
    loader_workers: int = 6
    games_in_flight: int = 96
    population_divisor: "int | None" = None
    #: How often the (relatively expensive) directory walk is paid for.
    storage_sample_seconds: float = 30.0
    #: A crash loop must not spend the rehearsal relaunching; it is a finding.
    max_launches: int = 12

    def to_dict(self) -> dict:
        return {
            "rehearsal_version": REHEARSAL_VERSION,
            "deadline_seconds": self.deadline_seconds,
            "crash_at_seconds": self.crash_at_seconds,
            "worker_kill_at_seconds": self.worker_kill_at_seconds,
            "poll_seconds": self.poll_seconds,
            "finalize_grace_seconds": self.finalize_grace_seconds,
            "device": self.device,
            "loader_workers": self.loader_workers,
            "games_in_flight": self.games_in_flight,
            "population_divisor": self.population_divisor,
            "storage_sample_seconds": self.storage_sample_seconds,
            "max_launches": self.max_launches,
        }


@dataclass
class StorageSample:
    utc: str
    elapsed_seconds: float
    rollout_bytes: int
    archive_bytes: int
    hot_bytes: int
    log_bytes: int
    external_free_bytes: int
    internal_free_bytes: int

    def to_dict(self) -> dict:
        return {
            "utc": self.utc,
            "elapsed_seconds": self.elapsed_seconds,
            "rollout_bytes": self.rollout_bytes,
            "archive_bytes": self.archive_bytes,
            "hot_bytes": self.hot_bytes,
            "log_bytes": self.log_bytes,
            "external_free_bytes": self.external_free_bytes,
            "internal_free_bytes": self.internal_free_bytes,
        }


@dataclass
class RehearsalRecord:
    """What actually happened, accumulated as it happens."""

    plan: dict
    events: list = field(default_factory=list)
    storage_samples: list = field(default_factory=list)
    launches: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "plan": self.plan,
            "events": self.events,
            "storage_samples": [sample.to_dict() for sample in self.storage_samples],
            "launches": self.launches,
        }


def rehearsal_semantics() -> dict:
    """What this harness guarantees, in one readable place."""
    return {
        "rehearsal_version": REHEARSAL_VERSION,
        "observation": "every recorded fact is read back off disk, not from the runner",
        "failure_1": "SIGKILL of the training process group — learner and loader workers",
        "failure_2": "SIGKILL of one CPU loader worker while the learner trains",
        "recovery": "relaunch through the production start_or_resume() path",
        "deadline": (
            "one shortened RunWindow stamped once; every restart reuses the "
            "persisted window and downtime is charged against it"
        ),
        "unchanged": [
            "learning rates",
            "opponent mixtures",
            "132-hour transition",
            "15-minute / 2-hour / 6-hour cadences",
            "objective and loss weights",
            "setup source",
            "storage and retention policy",
        ],
    }
