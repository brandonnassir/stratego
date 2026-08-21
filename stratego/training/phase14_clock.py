"""Phase 14: the wall clock, the scheduler seam and the absolute deadline.

Specification source: `02_AGENT_2_FINAL_TRAINING_INTEGRATION.md` sections 5, 11
and 12, over the frozen `wall_clock_contract`.

Why a clock is an object here
-----------------------------
Every long-horizon event of the run — the 2-hour archive, the 6-hour candidate,
the 132-hour main/late transition, the 168-hour stop — is defined against
elapsed wall-clock from the *original* start. A 90-minute rehearsal cannot
reach any of them, so those code paths would otherwise ship untested. The
scheduler therefore reads time through a :class:`Clock` object: production
passes :class:`SystemClock`, tests pass :class:`ManualClock`, and the *logic*
under both is one implementation.

The seam cannot leak into production
------------------------------------
:class:`ManualClock` carries ``production = False``, and
:func:`require_production_clock` — which the runner calls when it is started in
production mode — refuses it. A test clock is therefore not something the
production path can be talked into using; it is something the production path
rejects.

:meth:`RunWindow.rehearsal` is the second seam of the same shape, added for the
Phase 13 Agent 3 reliability rehearsal: a *real* wall clock over a deliberately
shortened deadline. The shortened window carries ``production = False``, that
flag travels in every checkpoint, and the runner refuses to resume a rehearsal
window as a production run or a production window as a rehearsal. Only the
deadline moves — the 132-hour transition stays exactly where it is, because a
rehearsal that reached the late segment early would be rehearsing a schedule
the real run will never follow.

Downtime counts
---------------
Elapsed is `now - run_start_utc` against the original start, not accumulated
compute time and not time since the latest restart. A crash that costs six
hours costs six hours of the 168, moves the main/late transition not at all,
and cannot produce a fresh deadline on resume — :meth:`RunWindow.resume`
reuses the persisted pair and re-derives nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .phase14_contract import (
    ARCHIVE_CADENCE_SECONDS,
    ARCHIVE_SNAPSHOTS_IN_RUN,
    CANDIDATE_CADENCE_SECONDS,
    CANDIDATE_COUNT,
    CANDIDATE_HOURS,
    DEADLINE_SECONDS,
    HOT_CHECKPOINT_SECONDS,
    SEGMENT_LATE,
    SEGMENT_MAIN,
    TRANSITION_SECONDS,
    learning_rate,
)


class Phase14ClockError(RuntimeError):
    """Raised when a Phase 14 time value or clock is not usable as given."""


# ---------------------------------------------------------------------------
# UTC helpers
# ---------------------------------------------------------------------------


def utc_text(moment: datetime) -> str:
    """One UTC instant as the run's canonical `...Z` string."""
    if moment.tzinfo is None:
        raise Phase14ClockError(f"{moment!r} is naive; Phase 14 times are UTC-aware")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def parse_utc(text: str) -> datetime:
    """The inverse of :func:`utc_text`, tolerant of a missing subsecond part."""
    raw = str(text)
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    moment = datetime.fromisoformat(raw)
    if moment.tzinfo is None:
        raise Phase14ClockError(f"{text!r} carries no timezone")
    return moment.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------


class Clock:
    """The two time readings the run makes, and nothing else.

    `now` decides schedule position and is the only reading that may be
    simulated. `monotonic` measures durations for throughput reporting; it is
    deliberately separate so a test that jumps `now` forward by 132 hours does
    not also claim the machine spent 132 hours computing.
    """

    production = False
    kind = "abstract"

    def now(self) -> datetime:  # pragma: no cover - interface
        raise NotImplementedError

    def monotonic(self) -> float:
        return time.perf_counter()

    def describe(self) -> dict:
        return {"kind": self.kind, "production": bool(self.production)}


class SystemClock(Clock):
    """The real UTC wall clock. The only clock Phase 14 production accepts."""

    production = True
    kind = "system"

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class ManualClock(Clock):
    """A controllable clock. Test-only, and refused by the production guard.

    Advancing time here is the only way a short test can stand at hour 132 or
    hour 168 and watch the *production* scheduler decide what happens there.
    """

    production = False
    kind = "manual"

    def __init__(self, start: "datetime | str") -> None:
        self._now = parse_utc(start) if isinstance(start, str) else start.astimezone(timezone.utc)
        self._monotonic = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, seconds: float) -> datetime:
        """Move both readings forward. Never backwards: time does not rewind."""
        if seconds < 0:
            raise Phase14ClockError(f"cannot advance by {seconds}s; time moves forward")
        self._now = self._now + timedelta(seconds=float(seconds))
        self._monotonic += float(seconds)
        return self._now

    def advance_hours(self, hours: float) -> datetime:
        return self.advance(float(hours) * 3600.0)

    def set(self, moment: "datetime | str") -> datetime:
        target = parse_utc(moment) if isinstance(moment, str) else moment.astimezone(timezone.utc)
        if target < self._now:
            raise Phase14ClockError(f"cannot set the clock back to {utc_text(target)}")
        return self.advance((target - self._now).total_seconds())


def production_clock() -> SystemClock:
    return SystemClock()


def require_production_clock(clock: Clock) -> Clock:
    """Refuse anything but the real wall clock in a production Phase 14 run."""
    if not getattr(clock, "production", False):
        raise Phase14ClockError(
            f"a {clock.describe()['kind']!r} clock may not drive a production "
            "Phase 14 run; the test scheduler seam is unavailable in production"
        )
    return clock


# ---------------------------------------------------------------------------
# The run window
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunWindow:
    """The three instants that define the run, computed once and persisted.

    `transition_utc` is derived here rather than recomputed at each check so a
    reader of a hot checkpoint sees the same instant the run has been using,
    and so a future change to the constant cannot silently move a live run's
    transition.
    """

    run_start_utc: datetime
    run_deadline_utc: datetime
    transition_utc: datetime
    #: `False` marks a *rehearsal* window — the same seam pattern as
    #: `ManualClock` and `Population.scaled`. It travels in the checkpoint so a
    #: shortened window can never be resumed as a production run, or the other
    #: way round.
    production: bool = True

    @staticmethod
    def start(now: datetime) -> "RunWindow":
        """Stamp a new window. Called once, immediately before the loop."""
        start = now.astimezone(timezone.utc)
        return RunWindow(
            run_start_utc=start,
            run_deadline_utc=start + timedelta(seconds=DEADLINE_SECONDS),
            transition_utc=start + timedelta(seconds=TRANSITION_SECONDS),
        )

    @staticmethod
    def rehearsal(now: datetime, deadline_seconds: float) -> "RunWindow":
        """Stamp a shortened rehearsal window. Test seam; production refuses it.

        Only the *deadline* is shortened. The late transition stays at the
        frozen 132 hours, which a rehearsal simply never reaches: moving it
        earlier would make the rehearsal exercise a schedule the real run will
        never follow, and section 3 of the rehearsal task forbids it. The
        main/late transition is verified instead through the manual-clock seam,
        against the same production code.
        """
        start = now.astimezone(timezone.utc)
        span = float(deadline_seconds)
        if not 0.0 < span < DEADLINE_SECONDS:
            raise Phase14ClockError(
                f"a rehearsal window spans {span}s; it must be positive and shorter "
                f"than the frozen {DEADLINE_SECONDS}s — a full-length window is the "
                "production window and is stamped by RunWindow.start"
            )
        return RunWindow(
            run_start_utc=start,
            run_deadline_utc=start + timedelta(seconds=span),
            transition_utc=start + timedelta(seconds=TRANSITION_SECONDS),
            production=False,
        )

    def __post_init__(self) -> None:
        deadline = (self.run_deadline_utc - self.run_start_utc).total_seconds()
        if self.production:
            if abs(deadline - DEADLINE_SECONDS) > 1e-6:
                raise Phase14ClockError(
                    f"the window spans {deadline}s, not the frozen {DEADLINE_SECONDS}s"
                )
        elif not 0.0 < deadline < DEADLINE_SECONDS:
            raise Phase14ClockError(
                f"the rehearsal window spans {deadline}s; it must be positive and "
                f"shorter than the frozen {DEADLINE_SECONDS}s"
            )
        transition = (self.transition_utc - self.run_start_utc).total_seconds()
        # Checked for a rehearsal window too: the transition never moves.
        if abs(transition - TRANSITION_SECONDS) > 1e-6:
            raise Phase14ClockError(
                f"the transition is at {transition}s, not the frozen {TRANSITION_SECONDS}s"
            )

    @property
    def deadline_seconds(self) -> float:
        """This window's own span. Equals the frozen constant in production."""
        return (self.run_deadline_utc - self.run_start_utc).total_seconds()

    def elapsed_seconds(self, now: datetime) -> float:
        return (now.astimezone(timezone.utc) - self.run_start_utc).total_seconds()

    def remaining_seconds(self, now: datetime) -> float:
        return (self.run_deadline_utc - now.astimezone(timezone.utc)).total_seconds()

    def to_dict(self) -> dict:
        return {
            "run_start_utc": utc_text(self.run_start_utc),
            "run_deadline_utc": utc_text(self.run_deadline_utc),
            "transition_utc": utc_text(self.transition_utc),
            "deadline_seconds": self.deadline_seconds,
            "transition_seconds": TRANSITION_SECONDS,
            "production": bool(self.production),
        }

    @staticmethod
    def from_dict(payload: dict) -> "RunWindow":
        """Rebuild a persisted window. The resume path; derives nothing new."""
        for key in ("run_start_utc", "run_deadline_utc", "transition_utc"):
            if key not in payload:
                raise Phase14ClockError(f"the persisted window has no {key!r}")
        return RunWindow(
            run_start_utc=parse_utc(payload["run_start_utc"]),
            run_deadline_utc=parse_utc(payload["run_deadline_utc"]),
            transition_utc=parse_utc(payload["transition_utc"]),
            # Absent means production: every window written before the
            # rehearsal seam existed was a full-length production window.
            production=bool(payload.get("production", True)),
        )


# ---------------------------------------------------------------------------
# The deadline / cadence controller
# ---------------------------------------------------------------------------


def segment_for_elapsed(elapsed_seconds: float) -> str:
    """The frozen segment of a launch instant. `>=` at the mark, by contract."""
    return SEGMENT_LATE if float(elapsed_seconds) >= TRANSITION_SECONDS else SEGMENT_MAIN


def learning_rate_for_elapsed(elapsed_seconds: float) -> float:
    return learning_rate(segment_for_elapsed(elapsed_seconds))


def archive_index_for_elapsed(elapsed_seconds: float) -> int:
    """The highest 2-hour archive mark crossed, capped at the run's last one."""
    index = int(max(0.0, float(elapsed_seconds)) // ARCHIVE_CADENCE_SECONDS)
    return min(index, ARCHIVE_SNAPSHOTS_IN_RUN)


def candidate_index_for_elapsed(elapsed_seconds: float) -> int:
    """The highest 6-hour candidate mark crossed; index 0 is hour 0."""
    index = int(max(0.0, float(elapsed_seconds)) // CANDIDATE_CADENCE_SECONDS)
    return min(index, CANDIDATE_COUNT - 1)


def hot_index_for_elapsed(elapsed_seconds: float) -> int:
    return int(max(0.0, float(elapsed_seconds)) // HOT_CHECKPOINT_SECONDS)


class DeadlineController:
    """The single authority on "what time is it in this run, and so what?".

    Everything the loop asks about time goes through one object holding one
    window and one clock, so there is no second place that could compute a
    different elapsed, a different segment or a different deadline.
    """

    def __init__(self, window: RunWindow, clock: "Clock | None" = None) -> None:
        self.window = window
        self.clock = clock or SystemClock()

    # -- construction ------------------------------------------------------

    @staticmethod
    def start(clock: "Clock | None" = None) -> "DeadlineController":
        clock = clock or SystemClock()
        return DeadlineController(RunWindow.start(clock.now()), clock)

    @staticmethod
    def rehearsal(clock: "Clock | None", deadline_seconds: float) -> "DeadlineController":
        """Stamp a shortened rehearsal window against a real wall clock."""
        clock = clock or SystemClock()
        return DeadlineController(
            RunWindow.rehearsal(clock.now(), deadline_seconds), clock
        )

    @staticmethod
    def resume(payload: dict, clock: "Clock | None" = None) -> "DeadlineController":
        """Reuse a persisted window. Never produces a new 168-hour duration."""
        return DeadlineController(RunWindow.from_dict(payload), clock or SystemClock())

    # -- readings ----------------------------------------------------------

    def now(self) -> datetime:
        return self.clock.now()

    def elapsed(self) -> float:
        return self.window.elapsed_seconds(self.now())

    def remaining(self) -> float:
        return self.window.remaining_seconds(self.now())

    def elapsed_hours(self) -> float:
        return self.elapsed() / 3600.0

    def segment(self) -> str:
        return segment_for_elapsed(self.elapsed())

    def learning_rate(self) -> float:
        return learning_rate(self.segment())

    def expired(self) -> bool:
        return self.now() >= self.window.run_deadline_utc

    # -- gates -------------------------------------------------------------

    def may_start_collection_unit(self) -> bool:
        """At or after the deadline, no new collection unit may be launched."""
        return not self.expired()

    def may_start_optimizer_step(self) -> bool:
        """At or after the deadline, no new optimizer step may begin.

        Deliberately the same reading as the collection gate rather than a
        softer one: the frozen behaviour lets an in-flight bulk unit finish its
        epochs *only* when those epochs began before the deadline, and the loop
        expresses that by asking before each step.
        """
        return not self.expired()

    # -- cadences ----------------------------------------------------------

    def archive_index_due(self) -> int:
        return archive_index_for_elapsed(self.elapsed())

    def candidate_index_due(self) -> int:
        return candidate_index_for_elapsed(self.elapsed())

    def candidate_hour_due(self) -> int:
        return CANDIDATE_HOURS[self.candidate_index_due()]

    def hot_index_due(self) -> int:
        return hot_index_for_elapsed(self.elapsed())

    def seconds_to_transition(self) -> float:
        return (self.window.transition_utc - self.now()).total_seconds()

    # -- reporting ---------------------------------------------------------

    def status(self) -> dict:
        elapsed = self.elapsed()
        return {
            "now_utc": utc_text(self.now()),
            "run_start_utc": utc_text(self.window.run_start_utc),
            "run_deadline_utc": utc_text(self.window.run_deadline_utc),
            "transition_utc": utc_text(self.window.transition_utc),
            "elapsed_seconds": elapsed,
            "elapsed_hours": elapsed / 3600.0,
            "remaining_seconds": self.remaining(),
            "remaining_hours": self.remaining() / 3600.0,
            "segment": segment_for_elapsed(elapsed),
            "learning_rate": learning_rate(segment_for_elapsed(elapsed)),
            "expired": self.expired(),
            "archive_index_due": archive_index_for_elapsed(elapsed),
            "candidate_index_due": candidate_index_for_elapsed(elapsed),
            "candidate_hour_due": CANDIDATE_HOURS[candidate_index_for_elapsed(elapsed)],
            "hot_index_due": hot_index_for_elapsed(elapsed),
            "clock": self.clock.describe(),
            "window_production": bool(self.window.production),
            "deadline_seconds": self.window.deadline_seconds,
        }

    def to_dict(self) -> dict:
        return self.window.to_dict()
