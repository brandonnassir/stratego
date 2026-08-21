"""Phase 14: the production launch supervisor.

Specification source: `04_AGENT_4_IMMUTABLE_LAUNCH_PACKAGE.md` sections 4 and 6.

Why a supervisor exists
-----------------------
Phase 13 Agent 3 established the fact this module answers: **a killed training
process cannot restart itself and cannot record its own death.** The rehearsal
survived a process-group ``SIGKILL`` only because a supervisor outside it
noticed and relaunched it; and its own telemetry showed ``failures: {}`` across
two process deaths, because the dead process wrote nothing and the resumed one
only knew about failures it had caught itself.

Over 168 unattended hours that is not a monitoring inconvenience — it is the
difference between an eight-minute gap and a run that quietly ended on Tuesday.

What it does and does not do
----------------------------
It launches the frozen Phase 14 runner in a child process, watches it from
outside, records every launch and every death, and relaunches through the
production ``start_or_resume()`` path. It contains no training logic, no
schedule and no configuration.

**It never creates a deadline.** It passes no window to the learner; the
learner resumes the window persisted in the checkpoint. Before every relaunch
the supervisor re-reads that window off disk and refuses to continue if it has
moved, so "restarted into a fresh 168 hours" is caught by the thing whose job
is to notice.

When it refuses to restart
--------------------------
Five conditions, checked in this order, and each one is a state where a
restart would do damage rather than recovery:

1. an **emergency stop** is active — the operator asked for this;
2. the **run manifest says training is closed** — restarting would reopen a
   finished run;
3. an **unrecoverable integrity failure** was recorded — the run is no longer
   the run its manifest describes, and it needs a human;
4. **no valid resume checkpoint exists** — there is nothing to resume, and a
   fresh ``start()`` would stamp a new 168-hour deadline;
5. the **deadline has passed** — training is over. One *closeout* launch is
   still permitted, and it is not a training restart: the runner resumes,
   observes ``past_deadline``, takes **zero optimizer steps** and finalizes.
   Agent 3 verified that path directly, nine hours past a real deadline.

Plus the bounded policy: after :data:`DEFAULT_MAX_CONSECUTIVE_RESTARTS`
restarts with no observed optimizer-step progress, the supervisor stops. A
machine that cannot get through one iteration is not a machine that should be
relaunched three hundred times overnight.

Candidate evaluation runs in its own lane
------------------------------------------
Every six-hour candidate must eventually receive the same frozen 128-game
direct-policy evaluation, and that must not depend on someone remembering. The
supervisor reads the candidate ledger off disk, and when a mark is pending it
launches the frozen evaluator as a **separate process**, one at a time. The
evaluator writes only to the ledger. It cannot stop training, change a
hyper-parameter or extend the deadline — the accepted control surface refuses
every frozen key by name — and a failed evaluation records a reason, preserves
the candidate and is retried later on the identical pack.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .phase14_launch import (
    Phase14LaunchError,
    emergency_stop_state,
    integrity_failure_state,
)
from .phase14_status import utc_text

PHASE14_SUPERVISOR_VERSION = "phase14_supervisor_v1"

#: Consecutive relaunches with no observed optimizer-step progress before the
#: supervisor concludes the machine, not the run, is the problem.
DEFAULT_MAX_CONSECUTIVE_RESTARTS = 5

#: Backoff between relaunches. Growing, so a fast crash loop costs minutes
#: rather than seconds of the deadline, and bounded so a single unlucky crash
#: does not idle the machine for an hour.
DEFAULT_RESTART_BACKOFF_SECONDS = (15.0, 60.0, 180.0, 600.0, 900.0)

#: How often the supervisor looks at its child and at the control files.
DEFAULT_POLL_SECONDS = 15.0

#: How often the pending-candidate ledger is consulted. Candidates appear every
#: six hours; polling it every fifteen seconds would be noise.
DEFAULT_CANDIDATE_POLL_SECONDS = 600.0

#: How long a killed or exiting child is given to actually disappear.
REAP_SECONDS = 30.0

ACTION_RESTART = "restart"
ACTION_FINALIZE_ONLY = "finalize_only"
ACTION_STOP = "stop"

ROLE_LEARNER = "learner"
ROLE_FINALIZE = "finalize"


class Phase14SupervisorError(RuntimeError):
    """Raised when the supervisor cannot proceed safely."""


# ---------------------------------------------------------------------------
# The event log
# ---------------------------------------------------------------------------


class SupervisorLog:
    """An append-and-fsync JSONL log of everything the supervisor did.

    Flushed and `fsync`-ed per record, because the interesting records are the
    ones written moments before something dies — including, on a power loss,
    the supervisor itself.
    """

    def __init__(self, path) -> None:
        self.path = Path(path)
        # The directory is created on the first write, not here: constructing a
        # supervisor to run `--preflight-only` should not bring the production
        # run directory into existence before anyone has decided to launch.
        self.records: list = []

    def emit(self, event: str, **fields) -> dict:
        record = {"utc": utc_text(), "unix": time.time(), "event": str(event), **fields}
        line = json.dumps(record, sort_keys=True, default=str) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)
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
                records.append({"event": "unparseable", "raw": line})
        return records


# ---------------------------------------------------------------------------
# Reading the run from outside the run
# ---------------------------------------------------------------------------


def resume_checkpoint_state(hot_root) -> dict:
    """The newest hot checkpoint that *validates*, described without loading a model.

    Read through the production ring so the supervisor's idea of "a valid
    resume checkpoint exists" is the runner's idea of it, not a second opinion.
    """
    from .phase14_checkpoint import HotCheckpointRing

    try:
        loaded = HotCheckpointRing(Path(hot_root)).load_latest()
    except Exception as error:  # noqa: BLE001 - an unreadable ring is a fact
        return {"valid": False, "error": f"{type(error).__name__}: {error}"}
    if loaded is None:
        return {"valid": False, "error": "no valid hot checkpoint"}
    path, payload = loaded
    trainer = payload.get("trainer_state", {})
    progress = payload.get("schedule_state", {}).get("progress", {})
    return {
        "valid": True,
        "path": str(path),
        "global_optimizer_step": int(trainer.get("global_optimizer_step", 0)),
        "iteration": int(progress.get("iteration", 0)),
        "run_window": dict(payload.get("run_window", {})),
        "model_state_digest": payload.get("model_state_digest"),
    }


def run_manifest_state(run_state_path) -> dict:
    path = Path(run_state_path)
    if not path.exists():
        return {"present": False, "closed": False}
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"present": True, "closed": False, "error": "unreadable run manifest"}
    progress = manifest.get("progress", {})
    return {
        "present": True,
        "closed": bool(progress.get("closed")),
        "close_reason": progress.get("close_reason", ""),
        "window": manifest.get("window", {}),
        "elapsed_hours": manifest.get("elapsed_hours"),
    }


def deadline_state(window: dict) -> dict:
    """Whether the persisted deadline has passed, from the window on disk."""
    from datetime import datetime, timezone

    from .phase14_clock import parse_utc

    text = (window or {}).get("run_deadline_utc")
    if not text:
        return {"known": False, "passed": False}
    try:
        deadline = parse_utc(text)
    except Exception:  # noqa: BLE001 - a malformed window is not a deadline
        return {"known": False, "passed": False, "error": f"unparseable deadline {text!r}"}
    now = datetime.now(timezone.utc)
    return {
        "known": True,
        "run_deadline_utc": text,
        "remaining_seconds": (deadline - now).total_seconds(),
        "passed": now >= deadline,
    }


# ---------------------------------------------------------------------------
# The restart policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RestartConditions:
    """Everything the restart decision depends on, all of it read off disk."""

    emergency_stop_active: bool = False
    run_closed: bool = False
    integrity_failure_recorded: bool = False
    resume_checkpoint_valid: bool = False
    deadline_passed: bool = False
    deadline_known: bool = False

    def to_dict(self) -> dict:
        return {
            "emergency_stop_active": self.emergency_stop_active,
            "run_closed": self.run_closed,
            "integrity_failure_recorded": self.integrity_failure_recorded,
            "resume_checkpoint_valid": self.resume_checkpoint_valid,
            "deadline_passed": self.deadline_passed,
            "deadline_known": self.deadline_known,
        }


def read_conditions(storage) -> RestartConditions:
    checkpoint = resume_checkpoint_state(storage.hot_root)
    manifest = run_manifest_state(storage.run_state_path)
    window = checkpoint.get("run_window") or manifest.get("window") or {}
    deadline = deadline_state(window)
    return RestartConditions(
        emergency_stop_active=emergency_stop_state(storage.external_root)["active"],
        run_closed=bool(manifest.get("closed")),
        integrity_failure_recorded=integrity_failure_state(storage.external_root)["recorded"],
        resume_checkpoint_valid=bool(checkpoint.get("valid")),
        deadline_passed=bool(deadline.get("passed")),
        deadline_known=bool(deadline.get("known")),
    )


def restart_decision(
    conditions: RestartConditions,
    *,
    consecutive_restarts: int = 0,
    max_consecutive_restarts: int = DEFAULT_MAX_CONSECUTIVE_RESTARTS,
    closeout_attempts: int = 0,
    max_closeout_attempts: int = 2,
) -> dict:
    """Whether to relaunch, close out, or stop — and why, in one sentence.

    A pure function of facts already read off disk, so the policy can be tested
    exhaustively without launching a process, and so the reason recorded in the
    log is the reason the decision was actually made.
    """
    if conditions.emergency_stop_active:
        return {"action": ACTION_STOP, "reason": "emergency stop is active"}
    if conditions.run_closed:
        return {"action": ACTION_STOP, "reason": "the run manifest says training is closed"}
    if conditions.integrity_failure_recorded:
        return {
            "action": ACTION_STOP,
            "reason": "an unrecoverable integrity failure has been recorded",
        }
    if not conditions.resume_checkpoint_valid:
        return {
            "action": ACTION_STOP,
            "reason": (
                "no valid resume checkpoint exists; a fresh start would stamp a new "
                "168-hour deadline and this supervisor never does that"
            ),
        }
    if conditions.deadline_passed:
        if closeout_attempts < int(max_closeout_attempts):
            return {
                "action": ACTION_FINALIZE_ONLY,
                "reason": (
                    "the deadline has passed; training does not restart. One closeout "
                    "launch resumes, takes zero optimizer steps and finalizes"
                ),
            }
        return {
            "action": ACTION_STOP,
            "reason": "the deadline has passed and the closeout launch has been attempted",
        }
    if int(consecutive_restarts) >= int(max_consecutive_restarts):
        return {
            "action": ACTION_STOP,
            "reason": (
                f"{consecutive_restarts} consecutive restarts made no optimizer-step "
                f"progress; the bounded restart policy stops here"
            ),
        }
    return {"action": ACTION_RESTART, "reason": "unexpected exit before the deadline"}


def exit_description(returncode: "int | None") -> dict:
    """Split a `Popen` return code into an exit code or a signal."""
    if returncode is None:
        return {"exit_code": None, "signal": None, "still_running": True}
    if returncode < 0:
        number = -int(returncode)
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = f"SIG{number}"
        return {"exit_code": None, "signal": number, "signal_name": name, "still_running": False}
    return {"exit_code": int(returncode), "signal": None, "still_running": False}


# ---------------------------------------------------------------------------
# The candidate-evaluation lane
# ---------------------------------------------------------------------------


def pending_candidates(evaluation_root) -> list:
    """Candidate marks with no complete evaluation, read off disk."""
    from ..evaluation.phase14_candidates import CandidateLedger

    try:
        return CandidateLedger.at(evaluation_root).pending()
    except Exception:  # noqa: BLE001 - a ledger problem never stops training
        return []


def unevaluated_candidates(evaluation_root) -> list:
    """Marked candidates still missing a complete 128-game result.

    The hour-168 gate: the frozen selection rule may not be applied until this
    list is empty, because a candidate scored on 40 games is not comparable
    with one scored on 128.
    """
    return [entry["hour"] for entry in pending_candidates(evaluation_root)]


def assert_all_candidates_evaluated(evaluation_root) -> dict:
    """Refuse the final selection while any candidate evaluation is missing."""
    from ..evaluation.phase14_candidates import CandidateLedger

    ledger = CandidateLedger.at(evaluation_root)
    missing = unevaluated_candidates(evaluation_root)
    if missing:
        raise Phase14SupervisorError(
            f"{len(missing)} candidate(s) still have no complete evaluation on the "
            f"frozen pack: hours {missing}. Run the out-of-band evaluator until the "
            "ledger is clear; the frozen selection rule compares complete results only"
        )
    return ledger.status_summary()


# ---------------------------------------------------------------------------
# The child process
# ---------------------------------------------------------------------------


@dataclass
class Child:
    """One launched process, in its own process group."""

    popen: subprocess.Popen
    role: str
    launched_utc: str
    launched_unix: float
    stdout_path: Path
    attempt: int = 0

    @property
    def pid(self) -> int:
        return self.popen.pid

    def alive(self) -> bool:
        return self.popen.poll() is None

    def terminate_group(self, sig=signal.SIGTERM) -> dict:
        try:
            os.killpg(os.getpgid(self.pid), sig)
        except (ProcessLookupError, PermissionError) as error:
            return {"signalled": False, "error": f"{type(error).__name__}: {error}"}
        return {"signalled": True, "signal": int(sig), "pid": self.pid}

    def wait_gone(self, seconds: float = REAP_SECONDS) -> "int | None":
        deadline = time.monotonic() + float(seconds)
        while time.monotonic() < deadline:
            if self.popen.poll() is not None:
                break
            time.sleep(0.2)
        return self.popen.poll()


def spawn(
    *,
    python: str,
    script,
    arguments: list,
    repository,
    role: str,
    stdout_path,
    attempt: int = 0,
) -> Child:
    """Start one child in its own session, output to a file.

    ``start_new_session=True`` gives the child its own process group, so a stop
    reaches the learner *and* its loader workers with one signal rather than
    racing a worker that has not been reaped.
    """
    stdout_path = Path(stdout_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    handle = open(stdout_path, "ab")
    popen = subprocess.Popen(
        [str(python), str(script), *[str(argument) for argument in arguments]],
        cwd=str(repository),
        env=environment,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return Child(
        popen=popen,
        role=role,
        launched_utc=utc_text(),
        launched_unix=time.time(),
        stdout_path=stdout_path,
        attempt=int(attempt),
    )


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------


@dataclass
class SupervisorPolicy:
    max_consecutive_restarts: int = DEFAULT_MAX_CONSECUTIVE_RESTARTS
    max_closeout_attempts: int = 2
    backoff_seconds: tuple = DEFAULT_RESTART_BACKOFF_SECONDS
    poll_seconds: float = DEFAULT_POLL_SECONDS
    candidate_poll_seconds: float = DEFAULT_CANDIDATE_POLL_SECONDS
    #: How often live loader-worker health is sampled and logged. The learner
    #: publishes telemetry once per iteration, when its pool is legitimately
    #: closed; a *live* worker count only means something sampled from outside
    #: while the learner is mid-epoch, which is what this does.
    status_sample_seconds: float = 60.0
    #: A ceiling on total launches over the whole run, independent of the
    #: consecutive bound: 168 hours of intermittent faults should still be
    #: visible as a number rather than as an unbounded log.
    max_total_launches: int = 64

    def backoff(self, consecutive: int) -> float:
        if not self.backoff_seconds:
            return 0.0
        index = min(int(consecutive), len(self.backoff_seconds) - 1)
        return float(self.backoff_seconds[index])

    def to_dict(self) -> dict:
        return {
            "max_consecutive_restarts": self.max_consecutive_restarts,
            "max_closeout_attempts": self.max_closeout_attempts,
            "backoff_seconds": list(self.backoff_seconds),
            "poll_seconds": self.poll_seconds,
            "candidate_poll_seconds": self.candidate_poll_seconds,
            "status_sample_seconds": self.status_sample_seconds,
            "max_total_launches": self.max_total_launches,
            "creates_a_new_deadline": False,
        }


@dataclass
class SupervisorState:
    launches: int = 0
    consecutive_restarts: int = 0
    closeout_attempts: int = 0
    evaluations_launched: int = 0
    step_at_last_launch: int = -1
    window: dict = field(default_factory=dict)
    stopped_because: str = ""

    def to_dict(self) -> dict:
        return {
            "launches": self.launches,
            "consecutive_restarts": self.consecutive_restarts,
            "closeout_attempts": self.closeout_attempts,
            "evaluations_launched": self.evaluations_launched,
            "step_at_last_launch": self.step_at_last_launch,
            "window": dict(self.window),
            "stopped_because": self.stopped_because,
        }


class Phase14Supervisor:
    """Launch the frozen Phase 14 runner, watch it, and restart it safely."""

    def __init__(
        self,
        storage,
        *,
        manifest: dict,
        python: str,
        learner_script,
        evaluator_script=None,
        repository=None,
        policy: "SupervisorPolicy | None" = None,
        log_path=None,
    ) -> None:
        from .phase14_contract import repository_root

        self.storage = storage
        self.manifest = dict(manifest)
        self.python = str(python)
        self.learner_script = Path(learner_script)
        self.evaluator_script = None if evaluator_script is None else Path(evaluator_script)
        self.repository = Path(repository or repository_root())
        self.policy = policy or SupervisorPolicy()
        self.log = SupervisorLog(
            log_path or Path(self.storage.log_root) / "phase14_supervisor.jsonl"
        )
        self.state = SupervisorState()
        self.child: "Child | None" = None
        self.evaluator: "Child | None" = None

    # -- preflight ---------------------------------------------------------

    def preflight(self) -> dict:
        """Everything that must be true before a single process is launched."""
        from .phase14_launch import (
            OperationalTopology,
            assert_frozen_topology,
            assert_host_stays_awake,
            assert_launch_code,
        )

        code = assert_launch_code(self.manifest)
        topology = assert_frozen_topology(OperationalTopology.from_manifest(self.manifest))
        power = assert_host_stays_awake()
        stop = emergency_stop_state(self.storage.external_root)
        if stop["active"]:
            raise Phase14LaunchError(
                f"an emergency stop is active at {stop['path']}; clear it deliberately "
                "before launching Phase 14"
            )
        integrity = integrity_failure_state(self.storage.external_root)
        if integrity["recorded"]:
            raise Phase14LaunchError(
                f"an unrecoverable integrity failure is recorded at {integrity['path']}; "
                "Phase 14 does not launch over one"
            )
        report = {
            "code": code,
            "topology": topology.to_dict(),
            "host_power": power,
            "manifest_digest": self.manifest.get("launch_manifest_digest"),
            "config_digest": self.manifest.get("phase14_final_training_config_digest"),
            "storage": self.storage.to_dict(),
        }
        self.log.emit("preflight", **report)
        return report

    # -- the window is never renegotiated ---------------------------------

    def _observe_window(self, checkpoint: dict) -> dict:
        """Record the persisted window, and refuse to watch it move."""
        window = dict(checkpoint.get("run_window") or {})
        if not window:
            return {}
        keys = ("run_start_utc", "run_deadline_utc", "transition_utc")
        if not self.state.window:
            self.state.window = window
            self.log.emit("window_observed", window=window)
            return window
        moved = {
            key: (self.state.window.get(key), window.get(key))
            for key in keys
            if self.state.window.get(key) != window.get(key)
        }
        if moved:
            raise Phase14SupervisorError(
                f"the persisted run window moved between launches: {moved}. A restart "
                "reuses the original deadline and never creates a new one"
            )
        return window

    # -- launching ---------------------------------------------------------

    def _learner_arguments(self, role: str) -> list:
        from .phase14_launch import OperationalTopology

        topology = OperationalTopology.from_manifest(self.manifest)
        return [
            "--role",
            role,
            "--external-root",
            str(self.storage.external_root),
            "--hot-root",
            str(self.storage.hot_root),
            "--device",
            topology.device,
            "--loader-workers",
            topology.loader_workers,
            "--games-in-flight",
            topology.games_in_flight,
            "--inference-batch-shape",
            topology.inference_batch_shape,
        ]

    def launch(self, *, role: str = ROLE_LEARNER, reason: str = "initial launch") -> Child:
        """Start one training (or closeout) process and record what was started."""
        checkpoint = resume_checkpoint_state(self.storage.hot_root)
        self._observe_window(checkpoint)
        self.state.launches += 1
        stdout = (
            Path(self.storage.log_root)
            / f"phase14_learner_{self.state.launches:03d}.log"
        )
        child = spawn(
            python=self.python,
            script=self.learner_script,
            arguments=self._learner_arguments(role),
            repository=self.repository,
            role=role,
            stdout_path=stdout,
            attempt=self.state.launches,
        )
        self.state.step_at_last_launch = int(checkpoint.get("global_optimizer_step", -1))
        self.child = child
        self.log.emit(
            "launch",
            role=role,
            reason=reason,
            attempt=self.state.launches,
            launch_timestamp=child.launched_utc,
            learner_pid=child.pid,
            checkpoint_selected=checkpoint.get("path"),
            checkpoint_step=checkpoint.get("global_optimizer_step"),
            checkpoint_valid=checkpoint.get("valid"),
            run_window=checkpoint.get("run_window"),
            stdout=str(stdout),
        )
        return child

    # -- the candidate lane ------------------------------------------------

    def _maybe_evaluate_candidates(self) -> "dict | None":
        """Start the out-of-band evaluator if a candidate is waiting.

        One at a time, in its own process, never inside the learner. The
        evaluator's results reach the ledger and nothing else.
        """
        if self.evaluator_script is None:
            return None
        if self.evaluator is not None and self.evaluator.alive():
            return None
        if self.evaluator is not None:
            returncode = self.evaluator.popen.poll()
            self.log.emit(
                "candidate_evaluator_exit",
                pid=self.evaluator.pid,
                **exit_description(returncode),
            )
            self.evaluator = None
        waiting = unevaluated_candidates(self.storage.evaluation_root)
        if not waiting:
            return None
        self.state.evaluations_launched += 1
        stdout = (
            Path(self.storage.log_root)
            / f"phase14_evaluator_{self.state.evaluations_launched:03d}.log"
        )
        self.evaluator = spawn(
            python=self.python,
            script=self.evaluator_script,
            arguments=[
                "--external-root",
                str(self.storage.external_root),
                "--hot-root",
                str(self.storage.hot_root),
            ],
            repository=self.repository,
            role="candidate_evaluator",
            stdout_path=stdout,
            attempt=self.state.evaluations_launched,
        )
        return self.log.emit(
            "candidate_evaluator_launched",
            pid=self.evaluator.pid,
            pending_hours=waiting,
            stdout=str(stdout),
            isolation="results reach the candidate ledger only; training is untouched",
        )

    # -- the loop ----------------------------------------------------------

    def _sample_worker_health(self) -> dict:
        """Live loader-worker health of the running learner, from outside it."""
        from .phase14_launch import OperationalTopology
        from .phase14_status import loader_health

        assert self.child is not None
        health = loader_health(
            pid=self.child.pid,
            configured_workers=OperationalTopology.from_manifest(
                self.manifest
            ).loader_workers,
            # The learner opens its pool only while an iteration trains, so the
            # supervisor cannot know from outside whether zero is a fault. It
            # reports the count and lets the sequence of samples say the rest.
            pool_open=False,
        )
        return self.log.emit(
            "worker_health_sample",
            learner_pid=self.child.pid,
            configured_loader_workers=health["configured_loader_workers"],
            live_loader_workers=health["live_loader_workers"],
            live_loader_worker_pids=health["live_loader_worker_pids"],
        )

    def _record_progress(self) -> dict:
        checkpoint = resume_checkpoint_state(self.storage.hot_root)
        step = int(checkpoint.get("global_optimizer_step", -1))
        if checkpoint.get("valid") and step > self.state.step_at_last_launch:
            if self.state.consecutive_restarts:
                self.log.emit(
                    "restart_success",
                    global_optimizer_step=step,
                    step_at_last_launch=self.state.step_at_last_launch,
                    consecutive_restarts_cleared=self.state.consecutive_restarts,
                )
            self.state.consecutive_restarts = 0
        return checkpoint

    def supervise(self, *, max_seconds: "float | None" = None) -> dict:
        """Watch the child, restart it when that is the right thing to do."""
        started = time.monotonic()
        last_candidate_poll = 0.0
        last_health_sample = 0.0
        while True:
            if max_seconds is not None and time.monotonic() - started >= float(max_seconds):
                self.state.stopped_because = "supervision limit"
                break
            assert self.child is not None
            if self.child.alive():
                if emergency_stop_state(self.storage.external_root)["active"]:
                    # The learner reads the same file and stops itself at a safe
                    # boundary; the supervisor only needs to stop relaunching.
                    self.log.emit("emergency_stop_seen", learner_pid=self.child.pid)
                now = time.monotonic()
                if now - last_candidate_poll >= self.policy.candidate_poll_seconds:
                    last_candidate_poll = now
                    self._maybe_evaluate_candidates()
                if now - last_health_sample >= self.policy.status_sample_seconds:
                    last_health_sample = now
                    self._sample_worker_health()
                self._record_progress()
                time.sleep(self.policy.poll_seconds)
                continue

            returncode = self.child.popen.poll()
            description = exit_description(returncode)
            conditions = read_conditions(self.storage)
            expected = conditions.run_closed or conditions.emergency_stop_active
            self.log.emit(
                "unexpected_exit" if not expected else "expected_exit",
                learner_pid=self.child.pid,
                role=self.child.role,
                attempt=self.child.attempt,
                lifetime_seconds=round(time.time() - self.child.launched_unix, 3),
                conditions=conditions.to_dict(),
                **description,
            )
            decision = restart_decision(
                conditions,
                consecutive_restarts=self.state.consecutive_restarts,
                max_consecutive_restarts=self.policy.max_consecutive_restarts,
                closeout_attempts=self.state.closeout_attempts,
                max_closeout_attempts=self.policy.max_closeout_attempts,
            )
            self.log.emit("restart_decision", **decision, conditions=conditions.to_dict())
            if decision["action"] == ACTION_STOP:
                self.state.stopped_because = decision["reason"]
                break
            if self.state.launches >= self.policy.max_total_launches:
                self.state.stopped_because = (
                    f"the supervisor has launched {self.state.launches} processes, its "
                    "ceiling for one run"
                )
                self.log.emit("launch_ceiling", **self.state.to_dict())
                break
            if decision["action"] == ACTION_FINALIZE_ONLY:
                self.state.closeout_attempts += 1
                self.log.emit(
                    "closeout_launch",
                    attempt=self.state.closeout_attempts,
                    note="resumes past the deadline; takes zero optimizer steps",
                )
                self.launch(role=ROLE_FINALIZE, reason=decision["reason"])
                continue
            self.state.consecutive_restarts += 1
            delay = self.policy.backoff(self.state.consecutive_restarts - 1)
            self.log.emit(
                "restart_attempt",
                attempt=self.state.consecutive_restarts,
                backoff_seconds=delay,
                reason=decision["reason"],
            )
            if delay:
                time.sleep(delay)
            try:
                self.launch(role=ROLE_LEARNER, reason=decision["reason"])
            except Phase14SupervisorError as error:
                self.log.emit("restart_failure", error=str(error))
                self.state.stopped_because = str(error)
                break
        final = {
            "supervisor_version": PHASE14_SUPERVISOR_VERSION,
            "state": self.state.to_dict(),
            "policy": self.policy.to_dict(),
            "conditions": read_conditions(self.storage).to_dict(),
            "candidates_outstanding": unevaluated_candidates(self.storage.evaluation_root),
        }
        self.log.emit("final_process_exit", **final)
        return final

    def stop_child(self) -> dict:
        """Ask the child to end, for a supervisor that is itself shutting down."""
        if self.child is None or not self.child.alive():
            return {"stopped": False}
        signalled = self.child.terminate_group()
        returncode = self.child.wait_gone()
        return self.log.emit("child_stopped", returncode=returncode, **signalled)


def supervisor_semantics() -> dict:
    return {
        "supervisor_version": PHASE14_SUPERVISOR_VERSION,
        "records": [
            "launch timestamp",
            "learner PID",
            "unexpected exit",
            "exit code / signal",
            "restart attempt",
            "checkpoint selected",
            "restart success/failure",
            "final process exit",
        ],
        "never": "creates a new training deadline",
        "refuses_to_restart_when": [
            "emergency stop is active",
            "the run manifest says training is closed",
            "an unrecoverable integrity failure has been recorded",
            "no valid resume checkpoint exists",
            "the deadline has passed (one zero-step closeout launch excepted)",
            "the bounded consecutive-restart policy is exhausted",
        ],
        "candidate_evaluation": (
            "pending marks are detected from the ledger on disk and evaluated in a "
            "separate process, one at a time, with no path back into training"
        ),
    }
