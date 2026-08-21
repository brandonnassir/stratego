"""One status document, assembled from the read-only sources.

What this module is responsible for
-----------------------------------
Choosing the *authoritative* source for each displayed value, and saying which
one it used. Phase 13's rehearsal found two operator numbers that were not what
an operator read them as, and the fix in both cases was where the number came
from, not what it was called. So every field that has a durable source and a
convenient one takes the durable source:

``committed games``
    the rollout store's iteration manifests, never the process-local counter,
    which is restored from the last hot checkpoint and is therefore low after
    any crash. The counter is displayed beside it, labelled.
``elapsed`` / ``remaining``
    the persisted run window, never accumulated active-training time.
``learner alive`` / ``live loader workers``
    probed against the OS now, never copied from a telemetry row that may have
    been written twenty minutes ago between iterations.
``checkpoint age``
    the file's mtime, never a counter in the runner's memory.

Health is only claimed where a contract backs it
------------------------------------------------
:func:`_health` grades exactly the conditions Phase 14 already defines: a
process being alive, a pool being whole, an age against a documented cadence,
free space against the frozen 120 GiB reserve, a non-finite counter being zero.
Loss movement is displayed and never graded — "policy loss went up, therefore
training is bad" is not in the frozen contract, and an alarm nobody agreed to
is an alarm an operator learns to ignore.

Degraded honestly
-----------------
Before launch there is no run directory, and during a crash there may be no
learner and a torn file. Both are ordinary states here: a missing source
becomes a stated absence, not an exception, because the moment an operator most
needs the page is the moment the run is in the worst shape.
"""

from __future__ import annotations

import time
from pathlib import Path

from .contract import (
    ARCHIVE_CADENCE_SECONDS,
    CANDIDATE_CADENCE_SECONDS,
    DASHBOARD_VERSION,
    GAMES_PER_ITERATION,
    HOT_CHECKPOINT_CONCERN_SECONDS,
    HOT_CHECKPOINT_SECONDS,
    LATE_LEARNING_RATE,
    LATE_SEGMENT_HOURS,
    MAIN_LEARNING_RATE,
    MAIN_SEGMENT_HOURS,
    MAX_LOADER_POOL_REBUILDS,
    SEGMENT_LATE,
    SEGMENT_MAIN,
    STORAGE_RESERVE_GIB,
    TELEMETRY_STALE_SECONDS,
    TOTAL_HOURS,
    TRANSITION_SECONDS,
)
from .sources import (
    RunPaths,
    parent_pid,
    _IterationCensus,
    _read_json,
    _stat,
    _TailReader,
    _TTLCache,
    archive_state,
    candidate_ledger_state,
    hot_checkpoint_state,
    loader_worker_pids,
    process_alive,
    utc_text,
    volume_usage,
    window_clock,
)

GREEN = "ok"
YELLOW = "watch"
RED = "bad"
UNKNOWN = "unknown"

#: Overall run states, in the vocabulary section 4 of the task names.
TRAINING = "TRAINING"
RECOVERING = "RECOVERING"
FINALIZING = "FINALIZING"
COMPLETE = "COMPLETE"
ERROR = "ERROR"
NOT_STARTED = "NOT STARTED"

#: How long each source's answer stays true. The wall clock is not here: it is
#: arithmetic over two persisted timestamps and is recomputed every request.
TTL_SECONDS = {
    "process": 5.0,
    "telemetry": 5.0,
    "supervisor": 5.0,
    "census": 60.0,
    "storage": 30.0,
    "checkpoints": 15.0,
    "run_state": 15.0,
    "candidates": 30.0,
}

#: The recent-history window the browser charts. One row per iteration at the
#: measured ~21 min/iteration is ~69 rows a day, so 400 rows is comfortably the
#: 6-24 hours the task asks for and bounds the JSON at the same time.
HISTORY_ROWS = 400

#: The training series the page plots. Presented, never graded.
HISTORY_SERIES = (
    ("policy_loss", ("training", "policy_loss")),
    ("value_loss", ("training", "value_loss")),
    ("belief_loss", ("training", "belief_loss")),
    ("grad_norm", ("training", "grad_norm")),
    ("learning_rate", ("training", "learning_rate")),
    ("advantage_retention", ("training", "advantage_retention")),
    ("examples_per_second", ("training", "examples_per_second")),
    ("games_per_second", ("collection", "games_per_second")),
    ("draw_rate", ("collection", "draw_rate")),
    ("mean_game_length", ("collection", "mean_game_length")),
)

#: Supervisor events that belong in an operator's event stream, mapped to the
#: sentence an operator should read. Anything not listed still appears, under
#: its own event name — an unrecognised event is information, not noise.
EVENT_SENTENCES = {
    "preflight": "preflight completed",
    "window_observed": "run window observed",
    "launch": "learner launched",
    "child_launched": "learner launched",
    "restart_decision": "restart decision recorded",
    "restart_failure": "restart failed",
    "launch_ceiling": "restart ceiling reached",
    "emergency_stop_seen": "emergency stop seen",
    "child_stopped": "learner stopped",
    "final_process_exit": "learner exited",
    "candidate_evaluation": "candidate evaluated",
    "worker_health": "worker health sampled",
    "progress": "progress recorded",
}


def _get(document, *path, default=None):
    cursor = document
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def _health(status, note: str, **extra) -> dict:
    return {"status": status, "note": note, **extra}


class DashboardState:
    """Reads one Phase 14 run and answers with one status document.

    Holds every cache, so a single instance is shared by all HTTP requests and
    a browser refreshing every ten seconds re-parses almost nothing. Nothing
    runs between requests: an unattended dashboard performs no reads at all.
    """

    def __init__(self, paths: RunPaths, ttl: "dict | None" = None) -> None:
        self.paths = paths
        seconds = {**TTL_SECONDS, **(ttl or {})}
        self.ttl = seconds
        self._caches = {name: _TTLCache(value) for name, value in seconds.items()}
        self._census = _IterationCensus()
        self._telemetry = _TailReader(paths.telemetry_path, retain=HISTORY_ROWS)
        self._supervisor = _TailReader(paths.supervisor_path, retain=HISTORY_ROWS)
        self.requests = 0
        self.started_unix = time.time()

    # -- cached sources ----------------------------------------------------

    def _cached(self, name: str, produce):
        return self._caches[name].get(produce)

    def telemetry_rows(self) -> list:
        return self._cached("telemetry", self._telemetry.read)

    def supervisor_events(self) -> list:
        return self._cached("supervisor", self._supervisor.read)

    def census(self) -> dict:
        return self._cached(
            "census", lambda: self._census.read(self.paths.rollout_root)
        )

    def run_state(self) -> dict:
        return self._cached(
            "run_state", lambda: _read_json(self.paths.run_state_path) or {}
        )

    def storage(self) -> dict:
        return self._cached(
            "storage", lambda: volume_usage(self.paths.external_root)
        )

    def checkpoints(self) -> dict:
        def produce():
            return {
                "hot": hot_checkpoint_state(self.paths.hot_root),
                "archive": archive_state(self.paths.archive_root),
            }

        return self._cached("checkpoints", produce)

    def candidates(self) -> dict:
        return self._cached(
            "candidates", lambda: candidate_ledger_state(self.paths.evaluation_root)
        )

    def processes(self) -> dict:
        """The learner PID from the supervisor log, and whether it lives now.

        The PID comes from the supervisor's own launch records rather than a
        pidfile, because the log is written anyway and records the launch a
        pidfile would only summarise. Liveness and the worker count are asked
        of the OS at read time — a status is the one moment where "currently
        live" means something.
        """

        def produce():
            launch = None
            for event in self.supervisor_events():
                if event.get("event") in ("launch", "child_launched"):
                    launch = event
            blank = {
                "known": False,
                "learner_pid": None,
                "learner_alive": False,
                "learner_orphaned": False,
                "live_loader_workers": 0,
                "live_loader_worker_pids": [],
                "launch_utc": None,
                "launch_attempt": None,
                "checkpoint_resumed_from": None,
                "launch_reason": None,
                "supervisor_pid": None,
                "supervisor_pid_source": "no launch recorded",
                "supervisor_alive": None,
            }
            if launch is None:
                return blank
            pid = launch.get("learner_pid") or launch.get("pid")
            alive = process_alive(pid)
            workers = loader_worker_pids(pid) if alive else []
            supervisor_pid, supervisor_source, orphaned = self._supervisor_pid(
                launch, pid, alive
            )
            return {
                **blank,
                "known": True,
                "learner_pid": pid,
                "learner_alive": alive,
                "learner_orphaned": orphaned,
                "live_loader_workers": len(workers),
                "live_loader_worker_pids": workers,
                "launch_utc": launch.get("utc", launch.get("launch_timestamp")),
                "launch_attempt": launch.get("attempt", launch.get("index")),
                "checkpoint_resumed_from": launch.get("checkpoint_selected"),
                "launch_reason": launch.get("reason", launch.get("label")),
                "supervisor_pid": supervisor_pid,
                "supervisor_pid_source": supervisor_source,
                "supervisor_alive": (
                    None if supervisor_pid is None else process_alive(supervisor_pid)
                ),
            }

        return self._cached("process", produce)

    @staticmethod
    def _supervisor_pid(launch: dict, learner_pid, alive: bool):
        """The supervisor's PID, derived from the learner when it is not logged.

        The accepted supervisor's `launch` record carries `learner_pid` and not
        its own PID, and changing it would move sealed Phase 14 code. It does
        not need to: `spawn` gives the learner a new *session*, not a new
        parent, so while the supervisor lives it is the learner's PPID.

        The derivation also answers a question a logged PID could not. A learner
        whose parent is PID 1 has been reparented to `launchd`, which means the
        supervisor died and left it running — exactly the state in which nothing
        is watching for the next crash, and exactly the state an operator would
        otherwise have to notice by hand.
        """
        logged = launch.get("supervisor_pid")
        if logged is not None:
            return logged, "supervisor log", False
        if not alive:
            # No live learner to ask, and no logged PID to fall back on.
            return None, "unavailable (learner not running)", False
        parent = parent_pid(learner_pid)
        if parent is None:
            return None, "unavailable", False
        if parent <= 1:
            return None, "learner reparented to launchd", True
        return parent, "derived from the learner's parent process", False

    # -- derived views -----------------------------------------------------

    def window(self) -> dict:
        """The run window, preferring the run manifest and falling back to telemetry."""
        manifest = self.run_state()
        window = manifest.get("window") or {}
        if not window.get("run_start_utc"):
            rows = self.telemetry_rows()
            if rows:
                clock = rows[-1].get("clock", {})
                window = {
                    "run_start_utc": clock.get("run_start_utc"),
                    "run_deadline_utc": clock.get("run_deadline_utc"),
                    "transition_utc": clock.get("transition_utc"),
                    "transition_seconds": clock.get("deadline_seconds"),
                    "production": clock.get("window_production", False),
                }
        return window

    def latest_row(self) -> dict:
        rows = self.telemetry_rows()
        return rows[-1] if rows else {}

    def _supervisor_counters(self) -> dict:
        """Restarts and the last exit, counted from the supervisor's own log.

        The supervisor is the only party that can record a hard kill, because a
        learner that took a SIGKILL wrote nothing. This is the counter section
        10 of the task asks for, and it is why the dashboard reads the
        supervisor log rather than only the telemetry.
        """
        launches = 0
        last_exit = None
        last_restart = None
        stops = 0
        for event in self.supervisor_events():
            name = event.get("event")
            if name in ("launch", "child_launched"):
                launches += 1
                last_restart = event.get("utc")
            elif name in ("final_process_exit", "child_stopped"):
                last_exit = {
                    "utc": event.get("utc"),
                    "returncode": event.get("returncode"),
                    "signal": event.get("signal") or event.get("signal_name"),
                    "description": event.get("description"),
                }
            elif name == "emergency_stop_seen":
                stops += 1
        return {
            "launches": launches,
            # The first launch is not a restart.
            "restarts": max(0, launches - 1),
            "last_restart_utc": last_restart if launches > 1 else None,
            "last_exit": last_exit,
            "emergency_stops_seen": stops,
            "events_read": self._supervisor.total_records,
            "log": str(self.paths.supervisor_path),
            "log_present": self.paths.supervisor_path.exists(),
        }

    def history(self) -> dict:
        """Bounded recent history for the browser's charts."""
        rows = self.telemetry_rows()
        series: dict = {name: [] for name, _ in HISTORY_SERIES}
        stamps = []
        for row in rows:
            stamps.append(row.get("unix"))
            for name, path in HISTORY_SERIES:
                value = _get(row, *path)
                series[name].append(value if isinstance(value, (int, float)) else None)
        return {
            "rows": len(rows),
            "unix": stamps,
            "series": series,
            "note": "one row per completed iteration (~21 minutes); presented, not graded",
        }

    def events(self) -> list:
        """A compact recent operational event stream."""
        stream = []
        for event in self.supervisor_events()[-60:]:
            name = str(event.get("event", "event"))
            stream.append(
                {
                    "utc": event.get("utc"),
                    "unix": event.get("unix"),
                    "event": name,
                    "text": EVENT_SENTENCES.get(name, name.replace("_", " ")),
                    "detail": event.get("reason")
                    or event.get("close_reason")
                    or event.get("error")
                    or event.get("label")
                    or "",
                    "source": "supervisor",
                }
            )
        # Iteration commits are a training event the supervisor does not log;
        # they come from the store, which is where the authoritative count is.
        for entry in self.census().get("iterations", [])[-20:]:
            if entry["sealed"]:
                stamp = entry.get("state_unix")
                stream.append(
                    {
                        "utc": None if stamp is None else utc_text(stamp),
                        "unix": stamp,
                        "event": "iteration_committed",
                        "text": (
                            f"population iteration {entry['iteration']} "
                            f"{entry['state'].lower()} ({entry['committed_games']} games)"
                        ),
                        "detail": entry.get("source") or "",
                        "source": "rollout store",
                    }
                )
        # Undated entries sort to the front, so a store that never stamped a
        # transition cannot push timestamped supervisor events out of the tail.
        stream.sort(key=lambda item: (item["unix"] is not None, item["unix"] or 0))
        return stream[-60:]

    # -- health ------------------------------------------------------------

    def _operational_health(self, clock, processes, workers, storage, hot, row) -> dict:
        checks = {}

        supervisor_alive = processes.get("supervisor_alive")
        if processes.get("learner_orphaned"):
            # The learner outlived its supervisor. Training continues, but
            # nothing is watching for the next crash.
            checks["supervisor"] = _health(
                RED,
                "not running — the learner was reparented to launchd and is "
                "unsupervised; see PHASE_14_RUNBOOK.md",
            )
        elif supervisor_alive is None:
            checks["supervisor"] = _health(
                UNKNOWN, processes.get("supervisor_pid_source") or "no supervisor PID"
            )
        else:
            checks["supervisor"] = _health(
                GREEN if supervisor_alive else RED,
                f"alive, pid {processes.get('supervisor_pid')}"
                if supervisor_alive
                else "not running",
            )

        learner_alive = processes.get("learner_alive")
        if not processes.get("known"):
            checks["learner"] = _health(UNKNOWN, "no launch recorded")
        elif learner_alive:
            checks["learner"] = _health(GREEN, f"alive, pid {processes.get('learner_pid')}")
        else:
            checks["learner"] = _health(
                RED, "not running — the supervisor restarts it; see PHASE_14_RUNBOOK.md"
            )

        configured = workers.get("configured_loader_workers")
        live = workers.get("live_loader_workers", 0)
        if not learner_alive:
            checks["loaders"] = _health(UNKNOWN, "learner not running")
        elif configured is None:
            checks["loaders"] = _health(UNKNOWN, f"{live} live; configured count not yet published")
        elif live == configured:
            checks["loaders"] = _health(GREEN, f"{live} of {configured} live")
        elif live == 0:
            # The pool exists only while an iteration trains. Zero live workers
            # during a collection is the normal state of a healthy run, and
            # grading it red for the ~5 minutes of every iteration spent
            # collecting is how an operator learns to ignore this field.
            checks["loaders"] = _health(
                GREEN, f"pool closed (collecting); {configured} configured"
            )
        else:
            checks["loaders"] = _health(YELLOW, f"degraded: {live} of {configured} live")

        rebuilds = workers.get("loader_pool_rebuilds")
        if rebuilds is None:
            checks["pool_rebuilds"] = _health(UNKNOWN, "not yet published")
        elif rebuilds == 0:
            checks["pool_rebuilds"] = _health(GREEN, "0")
        elif rebuilds < MAX_LOADER_POOL_REBUILDS:
            checks["pool_rebuilds"] = _health(
                YELLOW, f"{rebuilds} of {MAX_LOADER_POOL_REBUILDS} — recovered, but count them"
            )
        else:
            checks["pool_rebuilds"] = _health(
                RED, f"{rebuilds} at the frozen ceiling of {MAX_LOADER_POOL_REBUILDS}"
            )

        age = hot.get("age_seconds")
        if age is None:
            checks["checkpoint"] = _health(
                UNKNOWN if not learner_alive else YELLOW, "no hot checkpoint written yet"
            )
        elif age <= HOT_CHECKPOINT_CONCERN_SECONDS:
            checks["checkpoint"] = _health(GREEN, f"{age / 60.0:.1f} min old")
        else:
            checks["checkpoint"] = _health(
                YELLOW,
                f"{age / 60.0:.1f} min old — past one {HOT_CHECKPOINT_SECONDS}s cadence "
                "plus a collection",
            )

        free = storage.get("free_gib")
        if free is None:
            checks["storage"] = _health(RED, storage.get("error", "volume unreadable"))
        elif not storage.get("external_volume_present"):
            checks["storage"] = _health(RED, "external volume not mounted")
        elif free < STORAGE_RESERVE_GIB:
            checks["storage"] = _health(
                RED, f"{free:.1f} GiB free, under the frozen {STORAGE_RESERVE_GIB} GiB reserve"
            )
        elif free < STORAGE_RESERVE_GIB * 2:
            checks["storage"] = _health(YELLOW, f"{free:.1f} GiB free")
        else:
            checks["storage"] = _health(GREEN, f"{free:.1f} GiB free")

        counters = row.get("counters", {})
        nonfinite = sum(
            int(counters.get(key, 0) or 0)
            for key in ("non_finite_losses", "non_finite_gradients", "non_finite_parameters")
        )
        checks["nonfinite"] = (
            _health(GREEN, "0")
            if nonfinite == 0
            else _health(RED, f"{nonfinite} non-finite values recorded")
        )

        row_unix = row.get("unix")
        row_age = None if row_unix is None else max(0.0, time.time() - float(row_unix))
        if row_age is None:
            checks["telemetry"] = _health(UNKNOWN, "no telemetry row yet")
        elif not learner_alive:
            checks["telemetry"] = _health(UNKNOWN, f"{row_age / 60.0:.0f} min old; learner down")
        elif row_age <= TELEMETRY_STALE_SECONDS:
            checks["telemetry"] = _health(GREEN, f"{row_age / 60.0:.0f} min old")
        else:
            checks["telemetry"] = _health(
                YELLOW, f"{row_age / 60.0:.0f} min old — over two iterations"
            )
        return {"checks": checks, "telemetry_row_age_seconds": row_age}

    def overall(self, clock, processes, health, manifest, controls) -> dict:
        """One word for the top of the page, with the reason it was chosen."""
        if controls["integrity_failure"]["recorded"]:
            return {"state": ERROR, "reason": "an integrity failure is recorded"}
        if not self.paths.external_root.exists():
            return {"state": NOT_STARTED, "reason": "no Phase 14 run directory"}
        progress = manifest.get("progress", {})
        if progress.get("closed"):
            reason = progress.get("close_reason", "closed")
            return {"state": COMPLETE, "reason": f"run closed: {reason}"}
        if controls["emergency_stop"]["active"]:
            return {"state": FINALIZING, "reason": "emergency stop requested"}
        if not processes.get("known"):
            return {"state": NOT_STARTED, "reason": "no launch recorded"}
        if not processes.get("learner_alive"):
            return {
                "state": RECOVERING,
                "reason": "learner not running — the supervisor restarts it",
            }
        if clock.get("known") and clock.get("passed_deadline"):
            return {"state": FINALIZING, "reason": "past the absolute deadline"}
        statuses = [check["status"] for check in health["checks"].values()]
        if RED in statuses:
            return {"state": ERROR, "reason": "an operational check is red"}
        return {"state": TRAINING, "reason": "learner alive and inside the run window"}

    def controls(self) -> dict:
        """The two durable operator flags, read as files.

        Read and never written. Emergency stop is requested with
        `scripts/phase14_emergency_stop.py`, which is the accepted control
        path; showing the flag here does not make this a second one.
        """
        stop = _read_json(self.paths.emergency_stop_path)
        integrity = _read_json(self.paths.integrity_failure_path)
        return {
            "emergency_stop": {
                # Presence, not readability: an unreadable stop file is still a
                # stop file, and failing open here would show a run as running
                # that an operator has asked to stop.
                "active": self.paths.emergency_stop_path.exists(),
                "path": str(self.paths.emergency_stop_path),
                "reason": (stop or {}).get("reason", ""),
                "requested_utc": (stop or {}).get("requested_utc"),
                "requested_by_pid": (stop or {}).get("requested_by_pid"),
            },
            "integrity_failure": {
                "recorded": self.paths.integrity_failure_path.exists(),
                "path": str(self.paths.integrity_failure_path),
                "reason": (integrity or {}).get("reason", ""),
            },
            "note": "read-only; emergency stop is requested with scripts/phase14_emergency_stop.py",
        }

    # -- the document ------------------------------------------------------

    def status(self) -> dict:
        started = time.perf_counter()
        self.requests += 1

        row = self.latest_row()
        manifest = self.run_state()
        window = self.window()
        clock = window_clock(window)
        census = self.census()
        processes = self.processes()
        storage = self.storage()
        checkpoints = self.checkpoints()
        candidates = self.candidates()
        controls = self.controls()
        supervisor = self._supervisor_counters()

        persisted_workers = row.get("workers", {})
        workers = {
            "configured_loader_workers": persisted_workers.get(
                "configured_loader_workers", persisted_workers.get("loader_workers")
            ),
            "live_loader_workers": processes.get("live_loader_workers", 0),
            "live_loader_worker_pids": processes.get("live_loader_worker_pids", []),
            "loader_pool_rebuilds": persisted_workers.get("loader_pool_rebuilds"),
            "max_loader_pool_rebuilds": persisted_workers.get(
                "max_loader_pool_rebuilds", MAX_LOADER_POOL_REBUILDS
            ),
            "last_pool_rebuild_utc": persisted_workers.get("last_pool_rebuild_utc"),
            "last_pool_rebuild_reason": persisted_workers.get("last_pool_rebuild_reason", ""),
            "live_source": "OS children of the learner, probed now",
            "counter_source": "the learner's telemetry row",
        }

        health = self._operational_health(
            clock, processes, workers, storage, checkpoints["hot"], row
        )
        overall = self.overall(clock, processes, health, manifest, controls)

        elapsed = clock.get("elapsed_seconds", 0.0) if clock.get("known") else 0.0
        training = row.get("training", {})
        collection = row.get("collection", {})
        committed = census["committed_games"]
        games_per_hour = (committed / (elapsed / 3600.0)) if elapsed > 0 else None

        document = {
            "artifact": DASHBOARD_VERSION,
            "read_utc": utc_text(),
            "read_unix": time.time(),
            "overall": overall,
            "clock": clock,
            "schedule": {
                "segments": [
                    {
                        "name": SEGMENT_MAIN,
                        "from_hour": 0,
                        "to_hour": MAIN_SEGMENT_HOURS,
                        "learning_rate": MAIN_LEARNING_RATE,
                        "hours": MAIN_SEGMENT_HOURS,
                    },
                    {
                        "name": SEGMENT_LATE,
                        "from_hour": MAIN_SEGMENT_HOURS,
                        "to_hour": TOTAL_HOURS,
                        "learning_rate": LATE_LEARNING_RATE,
                        "hours": LATE_SEGMENT_HOURS,
                    },
                ],
                "transition_hour": MAIN_SEGMENT_HOURS,
                "total_hours": TOTAL_HOURS,
                "transition_seconds": TRANSITION_SECONDS,
                "population_segment": training.get("segment") or collection.get("segment"),
                "frozen": True,
                "note": "frozen values, displayed only — the dashboard has no control over them",
            },
            "training": {
                "global_optimizer_step": training.get("global_optimizer_step"),
                "learning_rate": training.get("learning_rate"),
                "segment": training.get("segment"),
                "policy_loss": training.get("policy_loss"),
                "value_loss": training.get("value_loss"),
                "belief_loss": training.get("belief_loss"),
                "total_loss": training.get("total_loss"),
                "grad_norm": training.get("grad_norm"),
                "kl": training.get("kl"),
                "policy_entropy": training.get("policy_entropy"),
                "clip_fraction": training.get("clip_fraction"),
                "advantage_retention": training.get("advantage_retention"),
                "examples_consumed": training.get("examples_consumed"),
                "examples_per_second": training.get("examples_per_second"),
                "cursor": training.get("cursor"),
            },
            "games": {
                "committed_games": committed,
                "committed_games_source": "rollout store iteration manifests",
                "committed_games_authoritative": True,
                "in_flight_games": census["in_flight_games"],
                "sealed_iterations": census["sealed_iterations"],
                "iteration": collection.get("iteration"),
                "games_per_iteration": GAMES_PER_ITERATION,
                "games_per_hour": games_per_hour,
                "games_per_second": collection.get("games_per_second"),
                "positions_generated": collection.get("positions_generated"),
                "draw_rate": collection.get("draw_rate"),
                "mean_game_length": collection.get("mean_game_length"),
                "process_counter_games": collection.get(
                    "process_counter_games", collection.get("games_generated")
                ),
                "process_counter_is_diagnostic": True,
                "process_counter_note": (
                    "process-local and restored from the last hot checkpoint; "
                    "low after a crash by design, and not an alarm"
                ),
                "unreadable_iteration_directories": census[
                    "unreadable_iteration_directories"
                ],
            },
            "population": row.get("population", {}),
            "workers": workers,
            "processes": processes,
            "supervisor": supervisor,
            "checkpoints": {
                "hot": {**checkpoints["hot"], "cadence_seconds": HOT_CHECKPOINT_SECONDS},
                "archive": {
                    **checkpoints["archive"],
                    "cadence_seconds": ARCHIVE_CADENCE_SECONDS,
                },
                "candidate": {
                    "cadence_seconds": CANDIDATE_CADENCE_SECONDS,
                    "latest": checkpoints["archive"].get("latest_candidate"),
                    "marks": checkpoints["archive"].get("candidate_marks", 0),
                    "note": "candidates are not evaluated by the dashboard",
                },
            },
            "candidates": candidates,
            "storage": {**storage, "reserve_gib": STORAGE_RESERVE_GIB},
            "counters": row.get("counters", {}),
            "failures": row.get("failures", {}),
            "controls": controls,
            "health": health,
            "events": self.events(),
            "history": self.history(),
            "sources": {
                **self.paths.to_dict(),
                "committed games": "rollout store iteration manifests",
                "process counter / losses / population": "phase14_telemetry.jsonl (one row per iteration)",
                "learner + loader liveness": "pgrep/ps against the learner PID, probed per request",
                "restarts and exits": "phase14_supervisor.jsonl",
                "checkpoint ages": "filesystem mtime; no checkpoint is opened",
                "disk": "statvfs on the external volume",
                "run window": "phase14_run_state.json, falling back to the telemetry clock",
            },
            "meta": {
                "read_only": True,
                "imports_model": False,
                "uses_mps": False,
                "telemetry_rows_held": len(self.telemetry_rows()),
                "telemetry_rows_read": self._telemetry.total_records,
                "manifest_parses": census["manifest_parses"],
                "requests_served": self.requests,
                "uptime_seconds": time.time() - self.started_unix,
                "ttl_seconds": self.ttl,
            },
        }
        document["meta"]["build_seconds"] = time.perf_counter() - started
        return document
