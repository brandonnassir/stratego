"""Every read the dashboard performs, and the caching that keeps them cheap.

Read-only, and structurally so
------------------------------
:func:`_read_bytes` and :func:`_read_json` are the only ways this module
touches a file, and both open in binary read mode. There is no write path, no
`mkdir`, no `unlink`, and no import of anything that has one. The Phase 14
supervisor and runner keep every recovery behaviour they had; the monitor's
entire vocabulary is `stat`, `open('rb')`, `statvfs`, `pgrep` and `ps`.

Cheap enough to poll
--------------------
The naive version of this file re-reads everything on every request, and at
hour 160 that means parsing ~460 iteration manifests every five seconds. Three
things stop that:

``_TTLCache``
    Each source declares how long its answer stays true. The wall clock is
    arithmetic and recomputed every request; a `statvfs` is worth 30 s; the
    store census is worth 60 s.
:class:`_IterationCensus`
    A sealed iteration's ``committed_games`` never changes again, so it is read
    once and then re-validated with two `stat` calls instead of a JSON parse.
    Steady-state cost is a couple of stats per iteration, not a parse.
:class:`_TailReader`
    The telemetry and supervisor logs are append-only, so they are read from
    the last byte offset forward rather than from the top. A seven-day
    supervisor log is read once.

Nothing runs in the background. Every number here is computed inside a request,
which is why an unattended dashboard costs no CPU at all.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .contract import (
    CANDIDATE_LEDGER_FILENAME,
    EMERGENCY_STOP_FILENAME,
    DURABLE_ARCHIVE_SUBDIRECTORY,
    EVALUATION_SUBDIRECTORY,
    EXTERNAL_RUN_DIRECTORY,
    EXTERNAL_VOLUME,
    GIB,
    HOT_CHECKPOINT_DIRECTORY,
    INTEGRITY_FAILURE_FILENAME,
    LOG_SUBDIRECTORY,
    PHASE14_NAMESPACE,
    ROLLOUT_SUBDIRECTORY,
    RUN_STATE_FILENAME,
    SEALED_OR_LATER,
    SEGMENT_LATE,
    SEGMENT_MAIN,
    SUPERVISOR_FILENAME,
    TELEMETRY_FILENAME,
    WORKER_COMMAND_MARK,
)

#: Hot checkpoint file naming, mirrored from
#: `stratego.training.phase14_checkpoint` (HOT_PREFIX / HOT_SUFFIX). Only used
#: to *list and stat* files — the dashboard never opens a checkpoint. Reading
#: one means `torch.load` of a full model and optimizer state, which is what
#: `resume_checkpoint_state` does and why the dashboard does not call it.
HOT_PREFIX = "hot_"
HOT_SUFFIX = ".pt"
ARCHIVE_PREFIX = "archive_"
CANDIDATE_MARK_SUFFIX = ".candidate.json"


def utc_text(unix: "float | None" = None) -> str:
    moment = (
        datetime.now(timezone.utc)
        if unix is None
        else datetime.fromtimestamp(float(unix), tz=timezone.utc)
    )
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def parse_utc(text: str) -> datetime:
    """Parse the `...Z` timestamps Phase 14 writes."""
    value = str(text).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


# ---------------------------------------------------------------------------
# The only two file primitives
# ---------------------------------------------------------------------------


def _read_bytes(path: Path, offset: int = 0) -> bytes:
    with open(path, "rb") as stream:
        if offset:
            stream.seek(offset)
        return stream.read()


def _read_json(path: Path):
    """A JSON document, or None if it is absent or torn.

    A torn file is a fact about the run — usually a write interrupted by the
    kill the operator is trying to understand — so it becomes a missing source
    rather than a traceback in the browser.
    """
    try:
        return json.loads(_read_bytes(Path(path)).decode("utf-8"))
    except (OSError, ValueError):
        return None


def _stat(path: Path):
    try:
        return path.stat()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class _TTLCache:
    """One value with an expiry, and the clock injected for tests."""

    def __init__(self, seconds: float, now=time.monotonic) -> None:
        self.seconds = float(seconds)
        self._now = now
        self._value = None
        self._stamp = None
        self.hits = 0
        self.misses = 0

    def get(self, produce):
        moment = self._now()
        if self._stamp is not None and (moment - self._stamp) < self.seconds:
            self.hits += 1
            return self._value
        self.misses += 1
        self._value = produce()
        self._stamp = moment
        return self._value

    def invalidate(self) -> None:
        self._stamp = None


class _TailReader:
    """Incremental reader for one append-only JSONL file.

    Keeps the byte offset it stopped at and the last `retain` records. A file
    that shrank or was replaced (a new run under the same path) is detected by
    inode and size and re-read from the top rather than silently producing
    garbage from a stale offset.
    """

    def __init__(self, path, retain: int = 400) -> None:
        self.path = Path(path)
        self.retain = int(retain)
        self.records: list = []
        self._offset = 0
        self._inode = None
        self.total_records = 0
        self.unparseable = 0

    def read(self) -> list:
        info = _stat(self.path)
        if info is None:
            self.records = []
            self._offset = 0
            self._inode = None
            self.total_records = 0
            return self.records
        if info.st_ino != self._inode or info.st_size < self._offset:
            self.records = []
            self._offset = 0
            self._inode = info.st_ino
            self.total_records = 0
            self.unparseable = 0
        if info.st_size == self._offset:
            return self.records
        try:
            chunk = _read_bytes(self.path, self._offset)
        except OSError:
            return self.records
        text = chunk.decode("utf-8", errors="replace")
        # A record being appended right now has no trailing newline yet; leave
        # it in the file for the next read rather than parsing half of it.
        cut = text.rfind("\n")
        if cut < 0:
            return self.records
        self._offset += len(text[: cut + 1].encode("utf-8"))
        for line in text[:cut].splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self.records.append(json.loads(line))
                self.total_records += 1
            except ValueError:
                self.unparseable += 1
        if len(self.records) > self.retain:
            self.records = self.records[-self.retain :]
        return self.records

    def latest(self):
        records = self.read()
        return records[-1] if records else None


# ---------------------------------------------------------------------------
# The authoritative committed-game census
# ---------------------------------------------------------------------------


def _iteration_number(directory: Path) -> int:
    _, _, digits = directory.name.partition("_")
    try:
        return int(digits)
    except ValueError:
        return -1


class _IterationCensus:
    """The rollout store's committed-game total, read the durable way.

    Mirrors the accepted `stratego.training.phase14_status.committed_game_census`
    source preference — manifest, then state, then the journals for the one
    iteration still collecting — and adds the caching a five-second refresh
    over 480 iterations needs. Per iteration the cache key is the (inode, size,
    mtime) of `state.json` and `manifest.json`; if neither moved, the parse is
    skipped. A sealed iteration therefore costs two `stat` calls forever after.
    """

    def __init__(self) -> None:
        self._entries: dict = {}
        self._journals: dict = {}
        self.parses = 0

    @staticmethod
    def _key(state_info, manifest_info):
        def part(info):
            return None if info is None else (info.st_ino, info.st_size, info.st_mtime)

        return (part(state_info), part(manifest_info))

    def _journal_games(self, directory: Path) -> "int | None":
        """Distinct committed game ids for the iteration still collecting.

        Distinct ids rather than lines: a duplicate commit is a defect the
        store's own reconciliation reports, and counting it twice here would
        quietly inflate the authoritative total. Read incrementally — the
        journals are append-only, and re-reading 2,048 records every five
        seconds is exactly the expensive scan the task forbids.
        """
        journal = directory / "journal"
        if not journal.is_dir():
            return None
        state = self._journals.setdefault(str(directory), {"ids": set(), "offsets": {}})
        try:
            files = sorted(journal.glob("*.commit.jsonl"))
        except OSError:
            return None
        for path in files:
            info = _stat(path)
            if info is None:
                continue
            offset = state["offsets"].get(path.name, 0)
            if info.st_size < offset:
                offset = 0
            if info.st_size == offset:
                continue
            try:
                text = _read_bytes(path, offset).decode("utf-8", errors="replace")
            except OSError:
                continue
            cut = text.rfind("\n")
            if cut < 0:
                continue
            state["offsets"][path.name] = offset + len(text[: cut + 1].encode("utf-8"))
            for line in text[:cut].splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                identifier = record.get("phase9_game_id")
                if identifier:
                    state["ids"].add(identifier)
        return len(state["ids"])

    def _entry(self, directory: Path) -> "dict | None":
        state_path = directory / "state.json"
        manifest_path = directory / "manifest.json"
        state_info = _stat(state_path)
        manifest_info = _stat(manifest_path)
        key = self._key(state_info, manifest_info)
        cached = self._entries.get(str(directory))
        if cached is not None and cached["key"] == key and cached["entry"] is not None:
            entry = dict(cached["entry"])
            if entry["source"] == "journal":
                games = self._journal_games(directory)
                if games is not None:
                    entry["committed_games"] = int(games)
            return entry

        self.parses += 1
        document = _read_json(state_path) or {}
        manifest = _read_json(manifest_path)
        state = document.get("state", "UNKNOWN")
        games = None
        source = None
        if manifest is not None and manifest.get("committed_games") is not None:
            games = int(manifest["committed_games"])
            source = "manifest.json"
        elif document.get("committed_games") is not None:
            games = int(document["committed_games"])
            source = "state.json"
        elif isinstance(document.get("seal_attempt"), dict):
            attempt = document["seal_attempt"]
            if attempt.get("committed_games") is not None:
                games = int(attempt["committed_games"])
                source = "state.json seal_attempt"
        if games is None:
            games = self._journal_games(directory)
            source = "journal" if games is not None else None
        # The store stamps every state transition, so an iteration commit can
        # take its place in a time-ordered event stream instead of floating to
        # one end of it undated.
        history = document.get("history") or []
        state_unix = None
        for record in reversed(history):
            if isinstance(record, dict) and record.get("unix") is not None:
                state_unix = float(record["unix"])
                break
        entry = (
            None
            if games is None
            else {
                "iteration": _iteration_number(directory),
                "state": state,
                "committed_games": int(games),
                "sealed": state in SEALED_OR_LATER,
                "source": source,
                "state_unix": state_unix,
                "sealed_rollout_digest": document.get("sealed_rollout_digest"),
            }
        )
        self._entries[str(directory)] = {"key": key, "entry": entry}
        if entry is not None and entry["source"] != "journal":
            # The iteration sealed; its journal bookkeeping is dead weight.
            self._journals.pop(str(directory), None)
        return entry

    def read(self, rollout_root, namespace: str = PHASE14_NAMESPACE) -> dict:
        root = Path(rollout_root) / str(namespace)
        try:
            directories = sorted(
                (path for path in root.glob("iteration_*") if path.is_dir()),
                key=lambda path: path.name,
            )
        except OSError:
            directories = []
        entries = []
        sealed_total = 0
        in_flight_total = 0
        unreadable = []
        for directory in directories:
            entry = self._entry(directory)
            if entry is None:
                unreadable.append(str(directory))
                continue
            if entry["sealed"]:
                sealed_total += entry["committed_games"]
            else:
                in_flight_total += entry["committed_games"]
            entries.append(entry)
        return {
            "authoritative": True,
            "source": "rollout store iteration manifests",
            "rollout_root": str(Path(rollout_root)),
            "namespace": str(namespace),
            "committed_games": int(sealed_total),
            "in_flight_games": int(in_flight_total),
            "iterations_counted": len(entries),
            "sealed_iterations": sum(1 for entry in entries if entry["sealed"]),
            "unreadable_iteration_directories": unreadable,
            "iterations": entries,
            "manifest_parses": self.parses,
        }


# ---------------------------------------------------------------------------
# Process facts, asked of the operating system
# ---------------------------------------------------------------------------


def _run(command: list) -> str:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def process_alive(pid: "int | None") -> bool:
    """Whether `pid` is a *running* process, zombies excluded.

    `kill(pid, 0)` succeeds on a zombie, so a learner that died and has not
    been reaped would otherwise read as healthy in an operator status. That is
    the one case this field exists to catch.
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
    state = _run(["ps", "-o", "state=", "-p", str(int(pid))]).strip()
    if not state:
        return False
    return not state.startswith("Z")


def parent_pid(pid: "int | None") -> "int | None":
    """The parent of `pid`, or None if it cannot be asked."""
    if pid is None:
        return None
    text = _run(["ps", "-o", "ppid=", "-p", str(int(pid))]).strip()
    try:
        return int(text)
    except ValueError:
        return None


def loader_worker_pids(pid: "int | None") -> list:
    """Live CPU loader workers of one learner, as the OS sees them.

    The pool's resource tracker is a child of the same parent and is not a
    worker; counting it would report a healthy pool that has lost every worker
    it had. It is excluded by command line, which is how Phase 13's rehearsal
    supervisor counted them.
    """
    if pid is None:
        return []
    listed = _run(["pgrep", "-P", str(int(pid))])
    workers = []
    for token in listed.split():
        try:
            child = int(token)
        except ValueError:
            continue
        described = _run(["ps", "-o", "state=,command=", "-p", str(child)]).strip()
        state, _, command = described.partition(" ")
        if WORKER_COMMAND_MARK in command and state != "Z":
            workers.append(child)
    return workers


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------


def volume_usage(path) -> dict:
    """Capacity facts about the volume holding `path`, measured now."""
    target = Path(path).expanduser()
    probe = target
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    try:
        usage = shutil.disk_usage(probe)
        statvfs = os.statvfs(probe)
    except OSError as error:
        return {
            "requested_path": str(target),
            "available": False,
            "error": f"{type(error).__name__}: {error}",
            "external_volume_present": Path(EXTERNAL_VOLUME).exists(),
        }
    mount = probe.resolve()
    while not os.path.ismount(mount) and mount.parent != mount:
        mount = mount.parent
    return {
        "requested_path": str(target),
        "available": True,
        "mounted": target.exists(),
        "probed_path": str(probe),
        "mount_point": str(mount),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "free_gib": round(usage.free / GIB, 3),
        "used_gib": round(usage.used / GIB, 3),
        "total_gib": round(usage.total / GIB, 3),
        "read_only": bool(statvfs.f_flag & os.ST_RDONLY),
        "external_volume_present": Path(EXTERNAL_VOLUME).exists(),
    }


def _newest(paths: list) -> "tuple[Path, os.stat_result] | None":
    best = None
    for path in paths:
        info = _stat(path)
        if info is None:
            continue
        if best is None or info.st_mtime > best[1].st_mtime:
            best = (path, info)
    return best


def hot_checkpoint_state(hot_root) -> dict:
    """The hot ring described by `stat` alone — never by loading a checkpoint.

    `resume_checkpoint_state` in the accepted supervisor answers a stronger
    question ("does the newest file *validate*") and pays `torch.load` of a
    full model and optimizer state to answer it. That is right for a resume
    decision and wrong for a page that refreshes every ten seconds, so the
    dashboard reports what the file system knows: how many files, how large,
    and how long ago the newest one was written. Validity is the supervisor's
    call, and the runbook's.
    """
    root = Path(hot_root)
    try:
        files = sorted(root.glob(f"{HOT_PREFIX}*{HOT_SUFFIX}"))
    except OSError:
        files = []
    newest = _newest(files)
    if newest is None:
        return {
            "directory": str(root),
            "present": False,
            "files": 0,
            "latest_path": None,
            "latest_unix": None,
            "latest_utc": None,
            "age_seconds": None,
            "validated": False,
            "validation_note": "not validated by the dashboard; validity is the supervisor's call",
        }
    path, info = newest
    return {
        "directory": str(root),
        "present": True,
        "files": len(files),
        "latest_path": str(path),
        "latest_name": path.name,
        "latest_bytes": int(info.st_size),
        "latest_unix": info.st_mtime,
        "latest_utc": utc_text(info.st_mtime),
        "age_seconds": max(0.0, time.time() - info.st_mtime),
        "validated": False,
        "validation_note": "not validated by the dashboard; validity is the supervisor's call",
    }


def archive_state(archive_root) -> dict:
    """The durable archive snapshots and candidate marks, by `stat` and small JSON."""
    root = Path(archive_root)
    try:
        snapshots = sorted(root.glob(f"{ARCHIVE_PREFIX}*.pt"))
        marks = sorted(root.glob(f"*{CANDIDATE_MARK_SUFFIX}"))
    except OSError:
        snapshots, marks = [], []
    newest = _newest(snapshots)
    latest_mark = None
    if marks:
        # Candidate marks are ~700-byte JSON documents; only the newest is read.
        newest_mark = _newest(marks)
        if newest_mark is not None:
            document = _read_json(newest_mark[0]) or {}
            latest_mark = {
                "path": str(newest_mark[0]),
                "hour": document.get("hour"),
                "written_utc": document.get("written_utc"),
                "global_optimizer_step": document.get("global_optimizer_step"),
                "iteration": document.get("iteration"),
                "archive_position": document.get("archive_position"),
                "evaluation_status": document.get("evaluation_status"),
                "unix": newest_mark[1].st_mtime,
                "age_seconds": max(0.0, time.time() - newest_mark[1].st_mtime),
            }
    return {
        "directory": str(root),
        "snapshots": len(snapshots),
        "latest_snapshot_path": None if newest is None else str(newest[0]),
        "latest_snapshot_utc": None if newest is None else utc_text(newest[1].st_mtime),
        "latest_snapshot_unix": None if newest is None else newest[1].st_mtime,
        "latest_snapshot_age_seconds": (
            None if newest is None else max(0.0, time.time() - newest[1].st_mtime)
        ),
        "candidate_marks": len(marks),
        "latest_candidate": latest_mark,
    }


def candidate_ledger_state(evaluation_root) -> dict:
    """The candidate ledger, read as the small JSON document it is.

    Read directly rather than through `CandidateLedger`, whose module imports
    the evaluation stack and therefore torch. The dashboard needs the counts,
    not the evaluator.
    """
    path = Path(evaluation_root) / CANDIDATE_LEDGER_FILENAME
    document = _read_json(path)
    if document is None:
        return {"present": False, "path": str(path), "by_status": {}, "candidates": 0}
    entries = document.get("candidates", {}) or {}
    counts: dict = {}
    unevaluated = []
    for key, entry in entries.items():
        status = str(entry.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
        if status != "complete":
            try:
                unevaluated.append(int(entry.get("hour", key)))
            except (TypeError, ValueError):
                continue
    return {
        "present": True,
        "path": str(path),
        "by_status": counts,
        "candidates": len(entries),
        "unevaluated_hours": sorted(unevaluated),
        "selection_rule": document.get("selection_rule"),
    }


# ---------------------------------------------------------------------------
# The run window, and the immutable clock derived from it
# ---------------------------------------------------------------------------


def window_clock(window: dict, now: "datetime | None" = None) -> dict:
    """Elapsed, remaining and segment, derived from the *original* window.

    Everything is computed against `run_start_utc` and `run_deadline_utc` as
    persisted. Downtime is therefore lost time and shows as lost time: a crash
    that costs six hours costs six of the 168, moves the 132-hour transition
    not at all, and cannot produce a fresh deadline. The dashboard has no code
    path that restarts this clock, because it has no code path that computes a
    deadline from anything except the pair on disk.
    """
    start_text = (window or {}).get("run_start_utc")
    deadline_text = (window or {}).get("run_deadline_utc")
    if not start_text or not deadline_text:
        return {"known": False, "reason": "no persisted run window"}
    try:
        start = parse_utc(start_text)
        deadline = parse_utc(deadline_text)
    except (TypeError, ValueError):
        return {"known": False, "reason": f"unparseable window {start_text!r}/{deadline_text!r}"}
    moment = now or datetime.now(timezone.utc)
    span = (deadline - start).total_seconds()
    elapsed = (moment - start).total_seconds()
    remaining = (deadline - moment).total_seconds()
    transition_text = (window or {}).get("transition_utc")
    transition_seconds = float((window or {}).get("transition_seconds") or 0.0)
    if transition_text:
        try:
            transition_seconds = (parse_utc(transition_text) - start).total_seconds()
        except (TypeError, ValueError):
            pass
    segment = SEGMENT_LATE if elapsed >= transition_seconds else SEGMENT_MAIN
    return {
        "known": True,
        "now_utc": utc_text(moment.timestamp()),
        "run_start_utc": start_text,
        "run_deadline_utc": deadline_text,
        "transition_utc": transition_text,
        "deadline_seconds": span,
        "transition_seconds": transition_seconds,
        "elapsed_seconds": elapsed,
        "elapsed_hours": elapsed / 3600.0,
        "remaining_seconds": remaining,
        "remaining_hours": remaining / 3600.0,
        "progress_fraction": max(0.0, min(1.0, elapsed / span)) if span > 0 else 0.0,
        "segment": segment,
        "seconds_to_transition": transition_seconds - elapsed,
        "passed_deadline": moment >= deadline,
        "window_production": bool((window or {}).get("production", False)),
        "derived_from": "the original immutable run window on disk",
    }


# ---------------------------------------------------------------------------
# The run layout
# ---------------------------------------------------------------------------


class RunPaths:
    """Where one Phase 14 run keeps the things the dashboard reads."""

    def __init__(self, external_root=None, hot_root=None) -> None:
        self.external_root = Path(external_root or EXTERNAL_RUN_DIRECTORY)
        if hot_root is not None:
            self.hot_root = Path(hot_root)
        elif external_root is None:
            self.hot_root = Path(__file__).resolve().parents[2] / HOT_CHECKPOINT_DIRECTORY
        else:
            self.hot_root = self.external_root / "hot"

    @property
    def rollout_root(self) -> Path:
        return self.external_root / ROLLOUT_SUBDIRECTORY

    @property
    def archive_root(self) -> Path:
        return self.external_root / DURABLE_ARCHIVE_SUBDIRECTORY

    @property
    def log_root(self) -> Path:
        return self.external_root / LOG_SUBDIRECTORY

    @property
    def evaluation_root(self) -> Path:
        return self.external_root / EVALUATION_SUBDIRECTORY

    @property
    def run_state_path(self) -> Path:
        return self.external_root / RUN_STATE_FILENAME

    @property
    def telemetry_path(self) -> Path:
        return self.log_root / TELEMETRY_FILENAME

    @property
    def supervisor_path(self) -> Path:
        return self.log_root / SUPERVISOR_FILENAME

    @property
    def emergency_stop_path(self) -> Path:
        return self.external_root / EMERGENCY_STOP_FILENAME

    @property
    def integrity_failure_path(self) -> Path:
        return self.external_root / INTEGRITY_FAILURE_FILENAME

    def to_dict(self) -> dict:
        return {
            "external_root": str(self.external_root),
            "hot_root": str(self.hot_root),
            "rollout_root": str(self.rollout_root),
            "archive_root": str(self.archive_root),
            "log_root": str(self.log_root),
            "evaluation_root": str(self.evaluation_root),
            "run_state": str(self.run_state_path),
            "telemetry": str(self.telemetry_path),
            "supervisor": str(self.supervisor_path),
        }
