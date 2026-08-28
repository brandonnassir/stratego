"""Phase 17 Agent 4: the frozen setup-episode budget policy.

Specification sources: Agent 4 instruction section 4, common contract section 8.

What this module is *not*
-------------------------
It is not a second queue. :class:`stratego.training.phase17.setup_episode.SetupEpisodeQueue`
is Agent 3's, it is accepted, and its FIFO discipline, rejection accounting and
state document are consumed unchanged. Wrapping it would give Phase 17 two
places where an episode can be dropped, which is exactly the failure mode
section 8 forbids.

What this module *is*
---------------------
The arithmetic that turns Agent 3's measured throughput plus a measured game
completion rate into one **fixed** setup-sequence budget, together with the
capacity, warm-up, age and alarm constants that budget implies. Those numbers
are provisional in Agent 3's handoff (`PROVISIONAL_SETUP_QUEUE_CAPACITY`,
`PROVISIONAL_SETUP_QUEUE_MAX_AGE_ITERATIONS`); Agent 4 freezes them.

Why a fixed budget and not "consume whatever is ready"
------------------------------------------------------
"Consume whatever is ready" is a biased estimator. A window in which the
population happens to finish many games would train on a batch dominated by
*short* games, and short games are not a random sample of the setup
distribution -- a setup that loses quickly finishes sooner than one that grinds
to the move limit. Fixing the count and skipping explicitly when it cannot be
met keeps every consumed batch the same size, so the only thing that varies is
*whether* an update happens, which is a recorded event rather than a silent
reweighting.

Which way the margin has to point
----------------------------------
This is the part that is easy to get backwards, and the Agent 4 rehearsal got
it backwards first. Agent 3's `SetupEpisodeQueue` **raises** at capacity: it
never evicts, because silent dropping is what section 8 forbids. So a budget
*below* the arrival rate is not "conservative" -- it makes the queue grow
without bound and the run dies on an exception hours in.

Write the dynamics down. With arrivals `A` per iteration and a fixed budget
`B`, the depth moves as `D <- D + A - B` on an iteration that updates and
`D <- D + A` on one that skips. Then:

```text
B <  A    depth grows linearly            -> capacity overflow, run dies
B == A    depth is a zero-drift walk      -> overflows on variance alone
B >  A    depth is bounded; the system
          skips a fraction p = 1 - A/B    -> the intended equilibrium
```

So `B > A`, and the price is a skip rate of `1 - A/B`, which is a *counted
event*, never a silently shrunk update. `sustainability_margin` below is
`B / A`, and the frozen minimum is above 1.0 for exactly this reason.

Arrivals are close to deterministic here, which is why a small margin suffices:
in steady state a window advances the population by exactly the transition
budget, so `A = 2 * budget_transitions / mean_game_length`. What actually moves
`A` over twelve hours is the mean game length changing as the policy trains --
which is why the capacity carries several budgets of headroom and the backlog
alarm (`P8`) fires, as a clean supervisor stop, before the queue can reach the
capacity that would raise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .setup_contract import Phase17SetupError

#: Version of the frozen budget policy document.
SETUP_BUDGET_POLICY_VERSION = "phase17_setup_budget_policy_v1"

#: The smallest budget-to-arrivals ratio Agent 4 will freeze. 1.0 is a
#: zero-drift walk that overflows on variance alone; 1.10 bounds the depth at
#: the cost of skipping about one setup update in eleven.
MINIMUM_SUSTAINABILITY_MARGIN = 1.10

#: Queue capacity, as a multiple of the budget. Sized for the surplus a
#: *shortening* mean game length would produce: games getting shorter as the
#: policy trains raises the arrival rate, and this is the headroom the backlog
#: alarm gets to fire inside before the queue reaches the capacity that raises.
CAPACITY_RESERVE_MULTIPLE = 8

#: Warm-up: the run does not attempt a setup update until the queue holds this
#: many budgets' worth of completed episodes. Two rather than one, because the
#: first games to finish in a fresh population are the *shortest* games in it --
#: a first update sized to exactly one budget would train on that biased tail.
WARM_UP_BUDGET_MULTIPLE = 2

#: Backlog alarm: queue depth above this fraction of capacity for the frozen
#: consecutive count is stop predicate P8's backlog half.
#:
#: 0.5, not 0.9, and the difference is not cosmetic. P8 needs *three
#: consecutive* windows before it stops, so the alarm has to fire far enough
#: below the capacity that three more windows of the worst plausible growth
#: still fit underneath it. At 0.9 of an 8x-budget capacity there are 0.8
#: budgets left; a run whose arrival rate has doubled adds a budget per window
#: and reaches the capacity -- which *raises* -- before the second reading. At
#: 0.5 there are 4 budgets of headroom, so even a doubled arrival rate leaves
#: four windows, and P8 stops the run cleanly instead.
#:
#: This was found by a rehearsal that hit the capacity and died. See
#: `SetupBudgetPolicy.headroom_windows`.
BACKLOG_ALARM_FRACTION_OF_CAPACITY = 0.5


class Phase17BudgetError(Phase17SetupError):
    """A setup budget could not be frozen as an unbiased bounded policy."""


@dataclass(frozen=True)
class SetupBudgetPolicy:
    """One frozen, unbiased, bounded setup-episode consumption policy."""

    budget: int
    capacity: int
    warm_up_minimum: int
    max_age_iterations: int
    backlog_alarm_depth: int
    age_alarm_iterations: int
    alarm_consecutive_windows: int
    #: the measurement this budget was derived from
    measured_completions_per_iteration: float
    measured_games_per_iteration: float
    sustainability_margin: float
    epochs_per_iteration: int
    measured_five_epoch_seconds: float
    notes: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise Phase17BudgetError(f"budget must be >= 1, got {self.budget}")
        if self.capacity < self.budget:
            raise Phase17BudgetError(
                f"capacity {self.capacity} is below the budget {self.budget}; a "
                "queue that cannot hold one update's worth of episodes will "
                "overflow every iteration"
            )
        if self.warm_up_minimum < self.budget:
            raise Phase17BudgetError(
                f"warm-up minimum {self.warm_up_minimum} is below the budget "
                f"{self.budget}; the first update would consume fewer episodes "
                "than the fixed batch"
            )
        if self.max_age_iterations < 1:
            raise Phase17BudgetError("max_age_iterations must be >= 1")
        if self.headroom_windows <= self.alarm_consecutive_windows:
            raise Phase17BudgetError(
                f"the backlog alarm at depth {self.backlog_alarm_depth} leaves "
                f"{self.headroom_windows:.2f} windows of headroom below the "
                f"capacity {self.capacity}, which is not more than P8's "
                f"{self.alarm_consecutive_windows} consecutive readings: the "
                "alarm could never complete before Agent 3's queue raises"
            )
        if self.sustainability_margin < MINIMUM_SUSTAINABILITY_MARGIN:
            raise Phase17BudgetError(
                f"sustainability margin {self.sustainability_margin:.3f} is below "
                f"the frozen minimum {MINIMUM_SUSTAINABILITY_MARGIN}: at "
                f"{self.measured_completions_per_iteration:.2f} completed setup "
                f"episodes arriving per iteration, a budget of {self.budget} lets "
                "the queue grow without bound and Agent 3's queue raises at "
                "capacity rather than evicting. Raise the budget or stop for "
                "operator review; do not shrink an update silently."
            )

    # -- construction ------------------------------------------------------

    @classmethod
    def freeze(
        cls,
        *,
        games_per_iteration: float,
        epochs_per_iteration: int = 5,
        five_epoch_seconds: float = 0.0,
        margin: float = MINIMUM_SUSTAINABILITY_MARGIN,
        notes: "list | None" = None,
    ) -> "SetupBudgetPolicy":
        """Derive the fixed budget from a *measured* game-completion rate.

        `games_per_iteration` is the mean number of games the population
        actually finishes in one move iteration, measured in the bounded
        rehearsal. Each finished game contributes two completed setup episodes.
        """
        completions = 2.0 * float(games_per_iteration)
        if completions <= 0.0:
            raise Phase17BudgetError(
                "the rehearsal completed no games, so no sustainable setup "
                "budget exists; stop for operator review rather than freezing a "
                "budget the run cannot supply"
            )
        budget = int(math.ceil(completions * float(margin)))
        if budget < 1:
            raise Phase17BudgetError(
                f"a {margin}x margin over {completions:.2f} completed episodes per "
                "iteration leaves no whole episode to train on; the population or "
                "the move budget is too small for a tandem setup update"
            )
        capacity = int(budget * CAPACITY_RESERVE_MULTIPLE)
        warm_up = int(budget * WARM_UP_BUDGET_MULTIPLE)
        # In equilibrium the depth hovers around one budget, so a consumed
        # episode is typically one or two iterations old. The ceiling is set at
        # the capacity multiple: an episode that has sat for that many
        # iterations is evidence the queue is backing up, which is what P8
        # exists to catch, and the age reading is how it catches it.
        max_age = int(CAPACITY_RESERVE_MULTIPLE)
        return cls(
            budget=budget,
            capacity=capacity,
            warm_up_minimum=warm_up,
            max_age_iterations=max_age,
            backlog_alarm_depth=int(capacity * BACKLOG_ALARM_FRACTION_OF_CAPACITY),
            age_alarm_iterations=max_age,
            alarm_consecutive_windows=3,
            measured_completions_per_iteration=completions,
            measured_games_per_iteration=float(games_per_iteration),
            sustainability_margin=budget / completions,
            epochs_per_iteration=int(epochs_per_iteration),
            measured_five_epoch_seconds=float(five_epoch_seconds),
            notes=list(notes or []),
        )

    # -- runtime -----------------------------------------------------------

    def may_update(self, queue_depth: int, *, warmed_up: bool) -> dict:
        """Whether this iteration runs a setup update, and why not if it does not."""
        depth = int(queue_depth)
        if not warmed_up and depth < self.warm_up_minimum:
            return {
                "update": False,
                "reason": "warm_up",
                "detail": (
                    f"queue holds {depth} completed episodes; warm-up needs "
                    f"{self.warm_up_minimum}"
                ),
                "queue_depth": depth,
            }
        if depth < self.budget:
            return {
                "update": False,
                "reason": "starved",
                "detail": (
                    f"queue holds {depth} completed episodes; the fixed setup "
                    f"budget is {self.budget}"
                ),
                "queue_depth": depth,
            }
        return {"update": True, "reason": None, "detail": None, "queue_depth": depth}

    @property
    def headroom_windows(self) -> float:
        """Windows between the backlog alarm and the capacity that raises.

        Measured in *budgets per window*, i.e. assuming the arrival rate has
        doubled to twice the budget so the queue gains one whole budget per
        window. Must exceed P8's consecutive count, or the alarm can never
        complete before the raise.
        """
        return (self.capacity - self.backlog_alarm_depth) / float(self.budget)

    def would_overflow(self, queue_depth: int, *, arrivals: "float | None" = None) -> dict:
        """Whether one more window could reach the capacity that raises.

        Checked *before* a window is collected, so a run that has run out of
        queue can stop having lost nothing, rather than raising in the middle
        of a window and discarding its work. `arrivals` defaults to twice the
        measured rate: the point of the check is to survive the case where the
        rate has moved, not the case where it has not.
        """
        expected = float(
            arrivals if arrivals is not None else 2.0 * self.measured_completions_per_iteration
        )
        projected = int(queue_depth) + expected
        return {
            "queue_depth": int(queue_depth),
            "assumed_arrivals": expected,
            "projected_depth": projected,
            "capacity": self.capacity,
            "would_overflow": projected >= self.capacity,
        }

    def alarms(self, telemetry: dict) -> dict:
        """Backlog and age readings against the frozen ceilings.

        `depth` is required rather than defaulted. A missing depth defaulting to
        zero would report "not backed up" for a queue that is full, which is the
        one reading P8 exists to catch. `oldest_age` is legitimately `None` on an
        empty queue, so that one is a real absence rather than a missing field.
        """
        if "depth" not in telemetry:
            raise Phase17BudgetError(
                "the queue telemetry has no 'depth'; refusing to report a "
                "backlog reading that cannot be computed"
            )
        depth = int(telemetry["depth"])
        oldest = float(telemetry.get("oldest_age") or 0)
        return {
            "backlog": {
                "depth": depth,
                "ceiling": self.backlog_alarm_depth,
                "over": depth > self.backlog_alarm_depth,
            },
            "age": {
                "oldest_age_iterations": oldest,
                "ceiling": self.age_alarm_iterations,
                "over": oldest > self.age_alarm_iterations,
            },
        }

    def document(self) -> dict:
        return {
            "policy_version": SETUP_BUDGET_POLICY_VERSION,
            "budget": self.budget,
            "capacity": self.capacity,
            "warm_up_minimum": self.warm_up_minimum,
            "max_age_iterations": self.max_age_iterations,
            "backlog_alarm_depth": self.backlog_alarm_depth,
            "age_alarm_iterations": self.age_alarm_iterations,
            "alarm_consecutive_windows": self.alarm_consecutive_windows,
            "backlog_alarm_fraction_of_capacity": BACKLOG_ALARM_FRACTION_OF_CAPACITY,
            "headroom_windows_at_a_doubled_arrival_rate": self.headroom_windows,
            "epochs_per_iteration": self.epochs_per_iteration,
            "measured_games_per_iteration": self.measured_games_per_iteration,
            "measured_completions_per_iteration": self.measured_completions_per_iteration,
            "measured_five_epoch_seconds": self.measured_five_epoch_seconds,
            "sustainability_margin": self.sustainability_margin,
            "expected_skip_fraction": max(
                0.0, 1.0 - self.measured_completions_per_iteration / self.budget
            ),
            "minimum_sustainability_margin": MINIMUM_SUSTAINABILITY_MARGIN,
            "capacity_reserve_multiple": CAPACITY_RESERVE_MULTIPLE,
            "warm_up_budget_multiple": WARM_UP_BUDGET_MULTIPLE,
            "consumption": "each episode exactly once; a short queue SKIPS explicitly",
            "bias_rule": (
                "a fixed batch never prefers short games: the count is constant "
                "and only whether an update happens varies, which is a counted event"
            ),
            "overflow_rule": (
                "capacity overflow and age rejection are counted rejections in "
                "Agent 3's queue, never silent drops"
            ),
            "notes": list(self.notes),
        }


__all__ = [
    "BACKLOG_ALARM_FRACTION_OF_CAPACITY",
    "CAPACITY_RESERVE_MULTIPLE",
    "MINIMUM_SUSTAINABILITY_MARGIN",
    "Phase17BudgetError",
    "SETUP_BUDGET_POLICY_VERSION",
    "SetupBudgetPolicy",
    "WARM_UP_BUDGET_MULTIPLE",
]
