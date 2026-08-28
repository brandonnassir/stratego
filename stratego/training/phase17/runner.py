"""Phase 17 Agent 4: the bulk-synchronous tandem move/setup runner.

Specification sources: Agent 4 instruction sections 3, 4, 5 and 10, common
contract sections 5, 6, 7, 8 and 9, operator decision D9-B sections 3 and 5.

One iteration, in the contractual order
---------------------------------------
```text
1  bind the current raw move and setup snapshot identities
2  refill the setup pools under that setup snapshot
3  advance the persistent population until exactly the move budget lands
4  replacement games draw both setups from the bound raw setup distribution
   and attach both behavior episodes
5  a finished game enqueues both of its setup episodes
6  build boundary-bootstrapped targets and run ONE move epoch
7  consume exactly the frozen setup budget, or SKIP explicitly; five setup epochs
8  update the two independent KL controllers and both raw-to-EMA states
9  atomically checkpoint and emit the telemetry row
10 rebind every active game to the newly updated raw move snapshot
```

Steps 3-5 happen inside Agent 2's collector, which is why the enqueue hook is
an override of `_retire` rather than a scan afterwards: `_retire` is the single
point at which a game is known to be finished *and* still addressable, so
hooking it makes "a completed game always enqueues both episodes" structural.
Scanning for disappeared game ids afterwards would work until the first game
that finished and was replaced inside the same window.

Step 10 is one assignment
-------------------------
`CurrentMovePolicy` is a single mutable cell shared by the whole population, so
"rebind every active game" is `cell.rebind_from_model(...)`. There is nothing to
propagate and therefore nothing to forget. The rebind happens *after* the move
update and *before* the next window, which is what makes the recorded
`behavior_model_state_digest` of every row the policy that actually chose it.

Setups are bound at creation, moves are not
--------------------------------------------
Contract section 5. A setup is drawn once when the game is created and its
behavior probabilities stay attached until the outcome arrives, even if the
setup learner has updated ten times since. The move policy is the opposite: it
is resolved per decision. Mixing these up in either direction silently breaks
the PPO ratio of whichever half was got wrong, so the two are held by different
mechanisms -- an immutable `SetupEpisode` and a mutable policy cell.

The D7-B/D5 setup recipe is consumed, not reinterpreted
--------------------------------------------------------
`SetupTrainingConfig` already carries D7-B's `0.9 * alpha * (I/10)` bonus and
D5's reverse-KL controller at target `0.0018` with a **per-iteration** update
from the **final epoch's** KL. The stale `d5_resolution.controller_update_cadence`
field in Agent 3's handoff says "once per setup EPOCH"; decision D9-B section 4
records that as an upstream documentation irregularity and Agent 3's
implementation already does the right thing. Nothing here re-implements the
controller.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ...engine.constants import BLUE, RED
from ..phase9_behavior import state_dict_digest
from ..phase9_trainer import KLController
from ..setup_source import SetupAssignment
from .checkpoint import (
    JOINT_CHECKPOINT_SCHEMA_VERSION,
    capture_active_game,
    restore_active_game,
)
from .move_contract import (
    MOVE_EMA_DECAY,
    game_id as make_game_id,
    parse_game_id,
    Phase17MoveError,
    WINDOW_TRANSITIONS,
    reference_iteration,
)
from .move_snapshot import CurrentMovePolicy, snapshot_from_model
from .move_start import build_move_start
from .move_trainer import MoveWindowTrainer, state_mapping_digest
from .queue import SetupBudgetPolicy
from .setup_contract import SETUP_EQUATION_VERSION, SetupTrainingConfig
from .setup_episode import attach_setup_episodes
from .setup_learning import SetupTrainer
from .setup_metrics import diversity_profile
from .setup_model import build_setup_model
from .setup_sampling import SetupPool
from .supervisor import CollapseSupervisor, MODE_INTEGRATION, MODE_PRODUCTION
from .transition_collector import FixedTransitionCollector

TANDEM_RUNNER_VERSION = "phase17_tandem_runner_v1"


def _utc_now() -> str:
    from datetime import datetime, timezone

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

#: Agent 3's measured initial masked-model baseline. Read, never recomputed
#: here: recomputing it under a different sample would move the stop threshold,
#: which is exactly what decision D9-B forbids.
SETUP_ENTROPY_BASELINE_NATS = 1.542894478885798

#: Setup family label recorded on every Phase 17 trajectory. Named so a later
#: reader can never confuse these games with frozen-library games.
PHASE17_SETUP_FAMILY = "phase17_current_raw_setup_v1"


class Phase17RunnerError(Phase17MoveError):
    """The tandem runner was configured or driven outside its contract."""


def move_means(means: dict, name: str) -> float:
    """One `MoveUpdate.means` value by its unprefixed name, failing loudly.

    Agent 2 stores every mean under a `mean_` prefix. A `.get("behavior_kl", 0.0)`
    against that mapping returns 0.0 forever, and a stop predicate fed a constant
    zero can never fire -- a guard that is quietly switched off is worse than no
    guard, because the telemetry says it is on.
    """
    key = f"mean_{name}"
    if key not in means:
        raise Phase17RunnerError(
            f"MoveUpdate.means has no {key!r}; available: {sorted(means)}"
        )
    return float(means[key])


# ---------------------------------------------------------------------------
# The setup provider: Agent 3's generator behind Agent 2's protocol
# ---------------------------------------------------------------------------


class Phase17SetupProvider:
    """Draws both sides of a new game from the bound raw setup snapshot.

    Implements the `setup_provider` protocol Agent 2's collector requires
    (`assign(...) -> SetupAssignment` plus a `setup_family` attribute) on top
    of Agent 3's `SetupPool` and `attach_setup_episodes`.

    There is no fallback. Contract section 7: a generation or orientation
    failure is fatal and there is no frozen setup library in Phase 17 training.
    A caller that wanted a fallback would be asking for the one behavior the
    stop policy calls `I4`.
    """

    setup_family = PHASE17_SETUP_FAMILY

    def __init__(
        self,
        model,
        *,
        run_id: str,
        model_state_digest: str,
        snapshot_iteration: int,
        pool_size: int,
    ) -> None:
        self.run_id = run_id
        self.pool_size = int(pool_size)
        self.pools = {
            int(RED): SetupPool(
                model,
                run_id=run_id,
                color=int(RED),
                model_state_digest=model_state_digest,
                snapshot_iteration=int(snapshot_iteration),
                size=int(pool_size),
            ),
            int(BLUE): SetupPool(
                model,
                run_id=run_id,
                color=int(BLUE),
                model_state_digest=model_state_digest,
                snapshot_iteration=int(snapshot_iteration),
                size=int(pool_size),
            ),
        }
        #: open episodes, keyed by game id, until the outcome arrives
        self.open_episodes: dict = {}
        #: episodes restored from a checkpoint, waiting for their game to be
        #: re-seated. Held separately from `open_episodes` so a resumed episode
        #: can only ever be claimed by the game it already belongs to.
        self.resuming: dict = {}
        self.assigned = 0
        self.legality_failures = 0
        self.orientation_failures = 0
        self.fallback_attempts = 0
        self.discarded_on_rebind = 0
        self.last_samples: dict = {int(RED): [], int(BLUE): []}

    @property
    def model_state_digest(self) -> str:
        return self.pools[int(RED)].model_state_digest

    @property
    def snapshot_iteration(self) -> int:
        return self.pools[int(RED)].snapshot_iteration

    def rebind(self, model, *, model_state_digest: str, snapshot_iteration: int) -> int:
        """Adopt a new raw setup snapshot; discard every unused pool candidate.

        Unused candidates are discarded rather than relabelled: a stale entry
        carries the *old* behavior probabilities, and reusing it under a new
        digest would make the PPO ratio's denominator a distribution the model
        no longer has.
        """
        discarded = sum(
            pool.rebind(
                model,
                model_state_digest=model_state_digest,
                snapshot_iteration=int(snapshot_iteration),
            )
            for pool in self.pools.values()
        )
        self.discarded_on_rebind += discarded
        return discarded

    def prefetch(self, game_ids: "list[str]") -> None:
        for pool in self.pools.values():
            pool.prefetch(list(game_ids))

    def assign(self, *, root_seed, environment_id, generation, game_id) -> SetupAssignment:
        """One game's two setups, with both behavior episodes attached."""
        if game_id in self.open_episodes:
            raise Phase17RunnerError(
                f"{game_id}: a setup episode is already attached; a game id is "
                "never reused inside one run"
            )
        resumed = self.resuming.pop(game_id, None)
        if resumed is not None:
            # A resumed game keeps the setups it was created with, together
            # with their recorded behavior probabilities. Drawing fresh ones
            # would change the board the game is already being played on.
            self.open_episodes[game_id] = resumed
            self.assigned += 1
            red_setup, blue_setup = resumed.engine_setups()
            return SetupAssignment(
                red_setup=red_setup,
                blue_setup=blue_setup,
                provenance={
                    "source": "phase17_current_raw_setup_policy",
                    "setup_family": self.setup_family,
                    "setup_model_state_digest": resumed.red.setup_model_state_digest,
                    "setup_snapshot_iteration": int(
                        resumed.red.setup_snapshot_iteration
                    ),
                    "red_canonical_fingerprint": resumed.red.canonical_fingerprint,
                    "blue_canonical_fingerprint": resumed.blue.canonical_fingerprint,
                    "orientation_rule_version": resumed.red.orientation_rule_version,
                    "restored_from_checkpoint": True,
                },
            )
        red = self.pools[int(RED)].take(game_id)
        blue = self.pools[int(BLUE)].take(game_id)
        episodes = attach_setup_episodes(red, blue, run_id=self.run_id, game_id=game_id)
        self.open_episodes[game_id] = episodes
        self.assigned += 1
        red_setup, blue_setup = episodes.engine_setups()
        return SetupAssignment(
            red_setup=red_setup,
            blue_setup=blue_setup,
            provenance={
                "source": "phase17_current_raw_setup_policy",
                "setup_family": self.setup_family,
                "setup_model_state_digest": red.setup_model_state_digest,
                "setup_snapshot_iteration": int(red.setup_snapshot_iteration),
                "red_canonical_fingerprint": red.canonical_fingerprint,
                "blue_canonical_fingerprint": blue.canonical_fingerprint,
                "orientation_rule_version": red.orientation_rule_version,
            },
        )

    def complete(self, game_id: str, terminal_result: str) -> list:
        """Bind the outcome to both episodes of a finished game."""
        episodes = self.open_episodes.pop(game_id, None)
        if episodes is None:
            raise Phase17RunnerError(
                f"{game_id} finished but has no attached setup episodes; a setup "
                "episode may never be orphaned from its game"
            )
        return episodes.complete(terminal_result)

    def telemetry(self) -> dict:
        red = self.pools[int(RED)].telemetry()
        blue = self.pools[int(BLUE)].telemetry()
        return {
            "generated": int(red["generated"] + blue["generated"]),
            "refills": int(red["refills"] + blue["refills"]),
            "unused": int(red["unused"] + blue["unused"]),
            "consumed": int(red["consumed"] + blue["consumed"]),
            "discarded_on_rebind": int(self.discarded_on_rebind),
            "snapshot_iteration": int(self.snapshot_iteration),
            "setup_model_state_digest": self.model_state_digest,
            "open_episodes": len(self.open_episodes),
            "assigned": int(self.assigned),
            "red": red,
            "blue": blue,
        }

    def capture_open_episodes(self) -> dict:
        """Every in-flight game's two setup episodes, as documents."""
        return {
            game_id: {
                "red": episodes.red.to_document(),
                "blue": episodes.blue.to_document(),
            }
            for game_id, episodes in self.open_episodes.items()
        }

    def restore_open_episodes(self, payload: dict) -> int:
        """Stage the checkpointed episodes for the games about to be re-seated."""
        from .setup_episode import GameSetupEpisodes, SetupEpisode

        self.resuming = {}
        for game_id, sides in payload.items():
            red = SetupEpisode.from_document(sides["red"])
            blue = SetupEpisode.from_document(sides["blue"])
            if red.game_id != game_id or blue.game_id != game_id:
                raise Phase17RunnerError(
                    f"checkpointed setup episodes under {game_id!r} name games "
                    f"{red.game_id!r}/{blue.game_id!r}"
                )
            self.resuming[game_id] = GameSetupEpisodes(
                game_id=game_id, red=red, blue=blue
            )
        return len(self.resuming)

    def pool_identity(self) -> dict:
        """The setup-pool identity a paired checkpoint binds."""
        return {
            "setup_model_state_digest": self.model_state_digest,
            "snapshot_iteration": int(self.snapshot_iteration),
            "pool_size_per_side": int(self.pool_size),
            "unused_discard_rule": (
                "unused pool candidates are DISCARDED on resume, never carried: "
                "they were drawn under the checkpointed snapshot and the resume "
                "regenerates under the same snapshot from the same per-game seed, "
                "so the setup a game receives is unchanged"
            ),
        }


# ---------------------------------------------------------------------------
# The collector, with the completed-episode hook
# ---------------------------------------------------------------------------


class TandemCollector(FixedTransitionCollector):
    """Agent 2's collector plus the one hook the setup half needs.

    `_retire` is the single point at which a game is known finished and still
    addressable. Overriding it is additive: `super()._retire` runs unchanged
    and nothing about how a window is collected, budgeted or emitted moves.
    """

    def __init__(self, *args, provider: Phase17SetupProvider, sink, **kwargs) -> None:
        self.provider = provider
        self.sink = sink
        self.completed_this_window: list = []
        super().__init__(*args, **kwargs)

    def _retire(self, runner, result, finished) -> None:
        super()._retire(runner, result, finished)
        episodes = self.provider.complete(runner.game_id, runner.record.terminal_result)
        for episode in episodes:
            self.sink(episode)
        self.completed_this_window.append(
            {
                "game_id": runner.game_id,
                "terminal_result": runner.record.terminal_result,
                "final_ply": int(runner.record.final_ply),
            }
        )


# ---------------------------------------------------------------------------
# Frozen configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TandemConfig:
    """Everything frozen before h0. Nothing here is recomputed from run speed."""

    run_id: str
    total_iterations: int
    move_budget: int = WINDOW_TRANSITIONS
    population: int = 256
    pool_size_per_side: int = 512
    setup_budget: int = 320
    setup_queue_capacity: int = 1280
    setup_warm_up_minimum: int = 320
    setup_max_age_iterations: int = 4
    setup_minibatch_episodes: int = 64
    move_device: str = "cpu"
    setup_device: str = "cpu"
    move_minibatch_size: int = 512
    setup_model_seed: int = 17
    work_package: str = "phase17"

    def __post_init__(self) -> None:
        # >= 2 because the setup alpha re-horizon divides by ln(N), which is
        # zero at N = 1. Enforced here so a mis-measured horizon fails when the
        # config is built rather than inside the first setup update.
        if self.total_iterations < 2:
            raise Phase17RunnerError("the frozen horizon N must be >= 2")
        if self.setup_budget > self.setup_queue_capacity:
            raise Phase17RunnerError(
                f"setup budget {self.setup_budget} exceeds queue capacity "
                f"{self.setup_queue_capacity}"
            )

    @property
    def reference_iteration(self) -> int:
        return reference_iteration(self.total_iterations)

    @property
    def setup_p(self) -> float:
        return 0.3 * math.log(42376) / math.log(self.total_iterations)

    def setup_config(self) -> SetupTrainingConfig:
        """Agent 3's config, with only the fields Agent 4 owns filled in."""
        return SetupTrainingConfig(
            run_id=self.run_id,
            total_iterations=self.total_iterations,
            device=self.setup_device,
            minibatch_episodes=self.setup_minibatch_episodes,
            queue_capacity=self.setup_queue_capacity,
            queue_max_age_iterations=self.setup_max_age_iterations,
        )

    def document(self) -> dict:
        return {
            "runner_version": TANDEM_RUNNER_VERSION,
            "work_package": self.work_package,
            "run_id": self.run_id,
            "horizon": {
                "N": int(self.total_iterations),
                "n_ref": self.reference_iteration,
                "p_setup": self.setup_p,
                "frozen_before_h0": True,
                "never": "recomputed from changing production speed",
            },
            "move": {
                "budget_transitions": int(self.move_budget),
                "population": int(self.population),
                "epochs_per_iteration": 1,
                "minibatch_size": int(self.move_minibatch_size),
                "ema_decay": MOVE_EMA_DECAY,
                "device": self.move_device,
            },
            "setup": {
                "pool_size_per_side": int(self.pool_size_per_side),
                "budget_episodes": int(self.setup_budget),
                "queue_capacity": int(self.setup_queue_capacity),
                "warm_up_minimum": int(self.setup_warm_up_minimum),
                "max_age_iterations": int(self.setup_max_age_iterations),
                "minibatch_episodes": int(self.setup_minibatch_episodes),
                "epochs_per_iteration": 5,
                "device": self.setup_device,
                # In the config digest because it determines the initial masked
                # setup model, and therefore every diversity baseline the stop
                # policy is calibrated against. A digest that did not cover it
                # would call two different starting distributions the same run.
                "model_seed": int(self.setup_model_seed),
                "recipe": "phase17_setup_update_v2 (D7-B) with the D5 controller",
                "controller_cadence": "once per setup ITERATION, on the FINAL epoch's KL",
                "controller_cadence_note": (
                    "the stale d5_resolution.controller_update_cadence field in "
                    "Agent 3's handoff says once per epoch; decision D9-B section 4 "
                    "records that as an upstream documentation irregularity"
                ),
            },
        }


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


@dataclass
class IterationResult:
    """One tandem iteration, as the telemetry row and the report both read it."""

    iteration: int
    window: object = None
    move_update: object = None
    setup_update: object = None
    setup_skipped: bool = False
    setup_skip_reason: "str | None" = None
    rebind: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)
    provider_telemetry: dict = field(default_factory=dict)
    queue_telemetry: dict = field(default_factory=dict)
    verdicts: list = field(default_factory=list)


class TandemRunner:
    """The whole Phase 17 system: one move learner, one setup learner, one loop."""

    def __init__(
        self,
        config: TandemConfig,
        *,
        budget_policy: "SetupBudgetPolicy | None" = None,
        supervisor_mode: str = MODE_PRODUCTION,
        move_start_path: "str | Path | None" = None,
        root: "str | Path" = ".",
        setup_model=None,
    ) -> None:
        self.config = config
        self.budget_policy = budget_policy
        self.supervisor_mode = supervisor_mode

        self.start = build_move_start(
            total_iterations=config.total_iterations,
            path=move_start_path,
            device=config.move_device,
            root=root,
        )
        self.move_model = self.start.model
        self.cell = CurrentMovePolicy(
            snapshot_from_model(self.move_model, device=config.move_device),
            iteration=1,
        )
        self.move_trainer = MoveWindowTrainer(
            run_id=config.run_id,
            model=self.move_model,
            optimizer=self.start.optimizer,
            controller=self.start.controller,
            ema=self.start.ema,
            horizon=self.start.horizon,
            device=config.move_device,
            minibatch_size=config.move_minibatch_size,
        )

        self.setup_config = config.setup_config()
        self.setup_model = setup_model or build_setup_model(
            device=config.setup_device, seed=config.setup_model_seed
        )
        self.setup_trainer = SetupTrainer(self.setup_model, self.setup_config)

        self.provider = Phase17SetupProvider(
            self.setup_model,
            run_id=config.run_id,
            model_state_digest=self._setup_digest(),
            snapshot_iteration=0,
            pool_size=config.pool_size_per_side,
        )
        self.collector = TandemCollector(
            run_id=config.run_id,
            cell=self.cell,
            setup_provider=self.provider,
            population=config.population,
            budget=config.move_budget,
            provider=self.provider,
            sink=self._enqueue,
        )
        self.supervisor = CollapseSupervisor(
            run_id=config.run_id,
            mode=supervisor_mode,
            setup_entropy_baseline=SETUP_ENTROPY_BASELINE_NATS,
        )

        self.iteration = 0
        self.elapsed_active_training_seconds = 0.0
        self.warmed_up = False
        self.enqueue_rejections: list = []
        self.setup_updates = 0
        self.setup_skips = 0

    # -- identities --------------------------------------------------------

    def _setup_digest(self) -> str:
        return state_dict_digest(self.setup_model)

    def _move_digest(self) -> str:
        return state_dict_digest(self.move_model)

    def move_ema_digest(self) -> str:
        return state_mapping_digest(self.start.ema.state_dict())

    def setup_ema_digest(self) -> str:
        return state_mapping_digest(self.setup_trainer.ema.state_dict())

    # -- the completed-episode sink ---------------------------------------

    def _enqueue(self, episode) -> None:
        """Every completed setup episode, with refusals counted, never dropped."""
        if not self.setup_trainer.queue.enqueue(episode):
            self.enqueue_rejections.append(
                {
                    "identity": episode.identity(),
                    "state": episode.state,
                    "reason": episode.rejected_reason,
                }
            )

    # -- one iteration -----------------------------------------------------

    def run_iteration(self, *, should_continue=None) -> IterationResult:
        """The ten contractual steps, in order, once."""
        self.iteration += 1
        n = self.iteration
        if n > self.config.total_iterations:
            raise Phase17RunnerError(
                f"iteration {n} is outside the frozen horizon "
                f"1..{self.config.total_iterations}; the horizon is frozen before "
                "launch and never extended from production speed"
            )
        result = IterationResult(iteration=n)
        started = time.perf_counter()

        # 0  refuse a window the completed-episode queue could not absorb.
        #
        # Agent 3's queue raises at capacity rather than evicting, which is
        # correct -- silent dropping is what section 8 forbids -- but a raise
        # lands in the MIDDLE of a window, after tens of thousands of
        # transitions have been collected, and discards all of it. Checking
        # first means a run that has genuinely run out of queue stops having
        # lost nothing. P8's backlog alarm is sized to fire several windows
        # before this point; reaching here means the arrival rate moved faster
        # than the alarm could accumulate.
        if self.budget_policy is not None:
            headroom = self.budget_policy.would_overflow(
                len(self.setup_trainer.queue)
            )
            if headroom["would_overflow"]:
                raise Phase17RunnerError(
                    f"iteration {n}: the completed-setup queue holds "
                    f"{headroom['queue_depth']} episodes and one more window "
                    f"could reach the capacity {headroom['capacity']}, which "
                    "Agent 3's queue raises on. Stopping before the window "
                    "rather than losing it mid-collection. Evidence: "
                    f"{headroom}"
                )

        # 1  bind the current raw move and setup snapshot identities
        bound_move_digest = self.cell.digest
        bound_setup_digest = self.provider.model_state_digest

        # 2  refill the setup pools under the bound setup snapshot
        generation_started = time.perf_counter()
        self.provider.prefetch(self._upcoming_game_ids())
        result.seconds["setup_generation"] = time.perf_counter() - generation_started

        # 3-5  advance the population; replacements draw setups and attach
        #      episodes; finished games enqueue both episodes
        self.collector.completed_this_window = []
        collection_started = time.perf_counter()
        window = self.collector.collect_window(should_continue=should_continue)
        result.window = window
        result.seconds["collection"] = time.perf_counter() - collection_started
        # Pool generation happens inside collection, through the provider; the
        # measured cost is attributed here rather than to the move half.
        result.provider_telemetry = self.provider.telemetry()

        # 6  boundary-bootstrapped targets, one move epoch
        optimization_started = time.perf_counter()
        move_update = self.move_trainer.train_window(
            window.rows, iteration=n, cell=self.cell
        )
        result.move_update = move_update
        result.seconds["move_optimization"] = time.perf_counter() - optimization_started

        # 7  exactly the frozen setup budget, or an explicit skip; five epochs
        setup_started = time.perf_counter()
        depth = len(self.setup_trainer.queue)
        if not self.warmed_up and depth >= self.config.setup_warm_up_minimum:
            self.warmed_up = True
        gate = self._setup_gate(depth)
        if gate["update"]:
            update = self.setup_trainer.update(batch_episodes=self.config.setup_budget)
            result.setup_update = update
            result.setup_skipped = bool(update.skipped)
            result.setup_skip_reason = update.skip_reason
            if update.skipped:
                self.setup_skips += 1
            else:
                self.setup_updates += 1
        else:
            result.setup_skipped = True
            result.setup_skip_reason = gate["detail"]
            self.setup_skips += 1
        result.seconds["setup_optimization"] = time.perf_counter() - setup_started
        result.queue_telemetry = self.setup_trainer.queue.telemetry(
            self.setup_trainer.setup_iteration
        ).__dict__

        # 8  the two independent controllers and both EMAs are advanced by the
        #    two trainers themselves; nothing is re-stepped here. Re-stepping
        #    either would double-count exactly the quantity D5 was about.

        # 9  telemetry / checkpointing is the caller's, so a rehearsal and a
        #    production run share this method verbatim.

        # 10 rebind every active game to the newly updated raw move snapshot
        result.rebind = self.cell.rebind_from_model(
            self.move_model, iteration=n + 1, device=self.config.move_device
        )
        if not result.rebind["changed"] and move_update.steps:
            raise Phase17RunnerError(
                f"iteration {n}: {move_update.steps} optimizer steps ran "
                "but the raw move digest did not change; the population would "
                "keep playing under the pre-update weights"
            )
        # The setup snapshot rebinds only when the setup model actually moved.
        setup_digest_now = self._setup_digest()
        if setup_digest_now != bound_setup_digest:
            self.provider.rebind(
                self.setup_model,
                model_state_digest=setup_digest_now,
                snapshot_iteration=self.setup_trainer.setup_iteration,
            )

        result.seconds["total"] = time.perf_counter() - started
        self.elapsed_active_training_seconds += result.seconds["total"]
        result.verdicts = self._supervise(result)
        return result

    def _upcoming_game_ids(self) -> list:
        """The game ids the collector will ask for next, in the order it will.

        A game id is `(run, slot, draw)` and `draw` only ever increments, so
        the next few draws of every slot are predictable exactly. Prefetching
        those -- rather than letting `SetupPool.take` fall through to a
        one-sample generation per new game -- is the whole reason the pool is
        batched: a 512-wide forward pass costs about what one 1-wide pass does.

        The lookahead covers the empty slots plus the games that will finish
        mid-window, spread evenly, capped at the frozen pool size.
        """
        slots = len(self.collector.slots)
        if not slots:
            return []
        lookahead = max(1, math.ceil(self.config.pool_size_per_side / slots))
        upcoming = []
        for slot in range(slots):
            base = int(self.collector.draw_counts[slot])
            for step in range(lookahead):
                upcoming.append(make_game_id(self.config.run_id, slot, base + step))
        return upcoming[: self.config.pool_size_per_side]

    def _setup_gate(self, depth: int) -> dict:
        if self.budget_policy is not None:
            return self.budget_policy.may_update(depth, warmed_up=self.warmed_up)
        if not self.warmed_up:
            return {
                "update": False,
                "reason": "warm_up",
                "detail": (
                    f"queue holds {depth} completed episodes; warm-up needs "
                    f"{self.config.setup_warm_up_minimum}"
                ),
            }
        if depth < self.config.setup_budget:
            return {
                "update": False,
                "reason": "starved",
                "detail": (
                    f"queue holds {depth} completed episodes; the fixed setup "
                    f"budget is {self.config.setup_budget}"
                ),
            }
        return {"update": True, "reason": None, "detail": None}

    # -- guards ------------------------------------------------------------

    def _supervise(self, result: IterationResult) -> list:
        verdicts = []
        ledger = self.collector.participant_ledger()
        verdicts.extend(self.supervisor.check_participant_ledger(ledger))
        verdicts.append(
            self.supervisor.check_setup_generation(
                legality_failures=self.provider.legality_failures,
                orientation_failures=self.provider.orientation_failures,
                fallback_attempts=self.provider.fallback_attempts,
            )
        )
        # MoveUpdate.means keys carry a `mean_` prefix -- `mean_behavior_kl`, not
        # `behavior_kl`. Reading the unprefixed name silently yields 0.0, which
        # would feed the move-KL and move-entropy predicates a constant zero and
        # make P2 and P6 unfireable. `move_means` fails loudly instead.
        means = (result.move_update.means or {}) if result.move_update else {}
        verdicts.append(
            self.supervisor.check_finite(
                {
                    "move_loss_total": move_means(means, "loss_total"),
                    "move_policy_entropy": move_means(means, "policy_entropy"),
                    "move_behavior_kl": move_means(means, "behavior_kl"),
                    "learning_rate": result.move_update.learning_rate if result.move_update else None,
                }
            )
        )
        if result.move_update is not None:
            verdicts.append(
                self.supervisor.observe_move_kl(move_means(means, "behavior_kl"))
            )
        update = result.setup_update
        if update is not None and not update.skipped:
            verdicts.append(self.supervisor.observe_setup_kl(float(update.control_kl)))
        if self.budget_policy is not None:
            verdicts.append(
                self.supervisor.observe_queue(
                    self.budget_policy.alarms(result.queue_telemetry)
                )
            )
        return verdicts

    # -- the D9-B tandem concentration reading ----------------------------

    def concentration_reading(self, *, samples: int = 160, label: str = "tandem") -> dict:
        """Decision D9-B section 5, measured under the live raw setup snapshot.

        Drawn from the *current* raw setup policy and scored with exactly Agent
        3's metric functions, at exactly Agent 3's sample shape: its soak drew
        160 Red plus 160 Blue and profiled the 320 together, so the `matched`
        profile below is the one its trajectory is comparable with. The
        per-colour profiles are extra, because a Red-only or Blue-only collapse
        and a symmetric one are different failures and the pooled number hides
        which happened.

        Drawing is a *read*. It uses reading-only game ids that no game will
        ever hold, so no training seed is consumed, no pool entry is taken from
        a game that needs one, and no counter moves.
        """
        from .setup_sampling import generate_setups

        digest = self._setup_digest()
        drawn_by_color = {}
        for color, name in ((int(RED), "red"), (int(BLUE), "blue")):
            drawn_by_color[name] = generate_setups(
                self.setup_model,
                run_id=self.config.run_id,
                game_ids=[
                    f"{self.config.run_id}-reading-{label}-{name}-{index}"
                    for index in range(samples)
                ],
                color=color,
                model_state_digest=digest,
                snapshot_iteration=self.setup_trainer.setup_iteration,
            )

        def profile(drawn, tag):
            return diversity_profile(
                [sample.canonical_setup for sample in drawn],
                behavior_probabilities=np.stack(
                    [sample.behavior_probabilities for sample in drawn]
                ),
                suffix_information=np.stack(
                    [sample.suffix_information_content for sample in drawn]
                ),
                label=tag,
            )

        pooled = profile(
            drawn_by_color["red"] + drawn_by_color["blue"], f"{label}_matched"
        )
        by_color = {
            name: profile(drawn, f"{label}_{name}")
            for name, drawn in drawn_by_color.items()
        }
        entropy = float(pooled["mean_prefix_entropy_nats"])
        floor = SETUP_ENTROPY_BASELINE_NATS * 0.60
        return {
            "label": label,
            "samples_per_color": int(samples),
            "matched_sample_count": int(pooled["sample_count"]),
            "matched_to": "Agent 3's soak diversity check: 160 Red + 160 Blue, profiled together",
            "setup_model_state_digest": digest,
            "setup_iteration": int(self.setup_trainer.setup_iteration),
            "move_raw_model_state_digest": self._move_digest(),
            "move_iteration": int(self.iteration),
            "mean_prefix_entropy_nats": entropy,
            "baseline_nats": SETUP_ENTROPY_BASELINE_NATS,
            "percent_of_baseline": 100.0 * entropy / SETUP_ENTROPY_BASELINE_NATS,
            "relative_60_percent_floor_nats": floor,
            "crosses_relative_floor": entropy < floor,
            "flag_effective_support": float(pooled["flag_effective_support"]),
            "bomb_effective_support": float(pooled["bomb_effective_support"]),
            "flag_square_support": int(pooled["flag_square_support"]),
            "bomb_square_support": int(pooled["bomb_square_support"]),
            "reflection_class_unique_fraction": float(
                pooled["reflection_class_unique_fraction"]
            ),
            "mean_class_distance": float(pooled["mean_class_distance"]),
            "min_class_distance": float(pooled["min_class_distance"]),
            "mean_top_token_concentration": float(pooled["mean_top_token_concentration"]),
            "mean_per_square_entropy_bits": float(pooled["mean_per_square_entropy_bits"]),
            "sequence_information_mean_nats": float(
                pooled["sequence_information_mean_nats"]
            ),
            "matched": pooled,
            "by_color": by_color,
        }

    # -- exact joint persistence -------------------------------------------

    def rng_namespaces(self) -> dict:
        """Every RNG stream this run has, and the counter that positions it.

        Phase 17 has no global RNG to carry, and that is a design property
        rather than an omission. Every draw is derived: a move action from
        `(game_id, ply)`, a setup token from the game's root seed, a minibatch
        order from `(run_id, iteration, epoch)`. So the complete RNG state *is*
        the set of counters below, and a resumed process reproduces every
        stream from them without inheriting a `torch` or `numpy` generator.
        Dropout is 0.0 in both models, so no stochastic layer reads a global
        generator during an update either.
        """
        return {
            "move_action_sampling": {
                "domain": "phase17/action-sampling",
                "derived_from": ["game_id", "ply"],
                "counter": {"iteration": int(self.iteration)},
            },
            "setup_draw": {
                "domain": "phase17/setup-draw",
                "derived_from": ["game_id"],
                "counter": {
                    "draw_counts": [int(v) for v in self.collector.draw_counts],
                    "setup_snapshot_iteration": int(self.provider.snapshot_iteration),
                },
            },
            "move_minibatch_order": {
                "domain": "phase17/minibatch-order",
                "derived_from": ["run_id", "iteration"],
                "counter": {"iteration": int(self.iteration)},
            },
            "setup_minibatch_order": {
                "domain": "phase17/setup-shuffle",
                "derived_from": ["run_id", "setup_iteration", "epoch"],
                "counter": {
                    "setup_iteration": int(self.setup_trainer.setup_iteration)
                },
            },
            "global_generators": {
                "torch": "not read: dropout is 0.0 and every draw is derived",
                "numpy": "not read: the setup shuffle builds its own RandomState per call",
            },
        }

    def capture(
        self,
        *,
        checkpoint_generation: int,
        parent_checkpoint_identity: dict,
        config_digest: str,
        source_digest: str,
        run_digest: str,
        telemetry_position: dict,
        next_export_boundary_seconds: float,
    ) -> dict:
        """The complete paired checkpoint payload, taken between iterations."""
        active = []
        for slot, runner in enumerate(self.collector.slots):
            if runner is None:
                continue
            active.append(
                capture_active_game(
                    runner,
                    slot=slot,
                    draw=int(parse_game_id(runner.game_id)["draw"]),
                )
            )
        setup_state = self.setup_trainer.state_document()
        return {
            "schema_version": JOINT_CHECKPOINT_SCHEMA_VERSION,
            "run_id": self.config.run_id,
            "work_package": self.config.work_package,
            "iteration": int(self.iteration),
            "start_identity": {
                key: self.start.identity[key]
                for key in ("path", "file_sha256", "model_state_digest", "parameter_count")
            },
            "move_raw_state": {
                name: tensor.detach().to("cpu").clone()
                for name, tensor in self.move_model.state_dict().items()
            },
            "move_raw_model_state_digest": self._move_digest(),
            "move_ema_state": self.start.ema.state_dict(),
            "move_ema_model_state_digest": self.move_ema_digest(),
            "setup_raw_state": setup_state["setup_raw_state"],
            "setup_raw_model_state_digest": setup_state["setup_raw_model_state_digest"],
            "setup_ema_state": setup_state["setup_ema_state"],
            "setup_ema_model_state_digest": setup_state["setup_ema_model_state_digest"],
            "setup_ema_updates": int(setup_state["setup_ema_updates"]),
            "setup_config_digest": setup_state["config_digest"],
            "move_optimizer_state": self.start.optimizer.state_dict(),
            "setup_optimizer_state": setup_state["setup_optimizer_state"],
            "move_kl_controller_state": self.move_trainer.controller.to_dict(),
            "setup_kl_controller_state": setup_state["setup_kl_controller_state"],
            "move_scheduler_position": {
                "iteration": int(self.iteration),
                "N": int(self.config.total_iterations),
                "n_ref": self.config.reference_iteration,
                "lr": self.start.horizon.learning_rate(max(1, self.iteration)),
                "c_H": self.start.horizon.entropy_coefficient(max(1, self.iteration)),
            },
            "setup_scheduler_position": {
                "iteration": int(self.setup_trainer.setup_iteration),
                "N": int(self.config.total_iterations),
                "N_paper": 42376,
                "p": self.config.setup_p,
                "alpha": self.setup_config.alpha(
                    max(1, self.setup_trainer.setup_iteration)
                ),
            },
            "move_optimizer_step_count": int(self.move_trainer.global_step),
            "setup_optimizer_step_count": int(self.setup_trainer.optimizer_step_count),
            "rng_namespaces": self.rng_namespaces(),
            "active_games": active,
            "active_game_setup_episodes": self.provider.capture_open_episodes(),
            "boundary_carry_state": self.collector.state(),
            "completed_setup_queue": setup_state["completed_setup_queue"],
            "setup_pool_identity": self.provider.pool_identity(),
            "run_digest": run_digest,
            "config_digest": config_digest,
            "source_digest": source_digest,
            "elapsed_active_training_seconds": float(
                self.elapsed_active_training_seconds
            ),
            "written_utc": _utc_now(),
            "checkpoint_generation": int(checkpoint_generation),
            "parent_checkpoint_identity": dict(parent_checkpoint_identity),
            "next_export_boundary_seconds": float(next_export_boundary_seconds),
            "telemetry_position": dict(telemetry_position),
            "supervisor_state": self.supervisor.state_document(),
            "collector_counters": {
                "warmed_up": bool(self.warmed_up),
                "setup_updates": int(self.setup_updates),
                "setup_skips": int(self.setup_skips),
                "enqueue_rejections": list(self.enqueue_rejections),
            },
            "move_trainer_state": self.move_trainer.trainer_state(),
            "window_partial_state": {
                "hot_checkpoints_mid_window": False,
                "detail": (
                    "a checkpoint is taken between iterations, after the window "
                    "closed and every collected transition was emitted, so there "
                    "is no partial-window transition state to carry. The "
                    "collector drops any un-applied neural request at the "
                    "boundary by design, which is why a mid-window checkpoint "
                    "would have nothing extra to store either."
                ),
            },
        }

    def restore(self, payload: dict) -> dict:
        """Put a validated paired checkpoint back, completely or not at all."""
        self.move_model.load_state_dict(payload["move_raw_state"])
        observed_move = self._move_digest()
        if observed_move != payload["move_raw_model_state_digest"]:
            raise Phase17RunnerError(
                f"restored move weights digest to {observed_move}, not the "
                f"checkpointed {payload['move_raw_model_state_digest']}"
            )
        self.start.ema.load_state_dict(
            payload["move_ema_state"],
            updates=int(payload["move_trainer_state"]["ema"]["updates"]),
        )
        self.start.optimizer.load_state_dict(payload["move_optimizer_state"])
        self.move_trainer.restore_state(payload["move_trainer_state"])
        self.move_trainer.controller = KLController.from_dict(
            payload["move_kl_controller_state"]
        )

        self.setup_trainer.load_state_document(
            {
                **{
                    key: payload[key]
                    for key in (
                        "setup_raw_state",
                        "setup_raw_model_state_digest",
                        "setup_ema_state",
                        "setup_ema_model_state_digest",
                        "setup_optimizer_state",
                        "setup_kl_controller_state",
                        "completed_setup_queue",
                    )
                },
                "config_digest": payload["setup_config_digest"],
                "setup_contract_version": self.setup_config.document()[
                    "setup_contract_version"
                ],
                "setup_equation_version": SETUP_EQUATION_VERSION,
                "run_id": payload["run_id"],
                "setup_iteration": int(payload["setup_scheduler_position"]["iteration"]),
                "setup_optimizer_step_count": int(payload["setup_optimizer_step_count"]),
                "setup_ema_updates": int(payload["setup_ema_updates"]),
                "setup_scheduler_position": payload["setup_scheduler_position"],
            }
        )

        self.iteration = int(payload["iteration"])
        self.elapsed_active_training_seconds = float(
            payload["elapsed_active_training_seconds"]
        )
        counters = payload["collector_counters"]
        self.warmed_up = bool(counters["warmed_up"])
        self.setup_updates = int(counters["setup_updates"])
        self.setup_skips = int(counters["setup_skips"])
        self.enqueue_rejections = list(counters["enqueue_rejections"])
        self.supervisor.load_state_document(payload["supervisor_state"])

        # The cell must hold the RESTORED raw weights before any game is
        # re-seated, or the first decision after a resume would be taken under
        # the model this process happened to construct.
        self.cell.rebind(
            snapshot_from_model(self.move_model, device=self.config.move_device),
            iteration=self.iteration + 1,
        )
        self.provider.rebind(
            self.setup_model,
            model_state_digest=payload["setup_raw_model_state_digest"],
            snapshot_iteration=int(payload["setup_scheduler_position"]["iteration"]),
        )

        carry = payload["boundary_carry_state"]
        self.collector.restore_counters(carry)
        self.collector.restore_seating(carry)
        staged = self.provider.restore_open_episodes(
            payload["active_game_setup_episodes"]
        )
        seated = self.collector.fill()
        if self.provider.resuming:
            raise Phase17RunnerError(
                f"{len(self.provider.resuming)} checkpointed setup episodes were "
                "not claimed by a re-seated game: "
                f"{sorted(self.provider.resuming)}"
            )
        by_game = {entry["game_id"]: entry for entry in payload["active_games"]}
        restored = 0
        for runner in self.collector.slots:
            if runner is None:
                continue
            entry = by_game.get(runner.game_id)
            if entry is None:
                raise Phase17RunnerError(
                    f"{runner.game_id} was re-seated but is not in the checkpoint"
                )
            if tuple(entry["red_setup"]) != tuple(runner.builder.red_setup) or tuple(
                entry["blue_setup"]
            ) != tuple(runner.builder.blue_setup):
                raise Phase17RunnerError(
                    f"{runner.game_id}: the re-seated setups differ from the "
                    "checkpointed ones"
                )
            restore_active_game(runner, entry)
            restored += 1
        return {
            "iteration": int(self.iteration),
            "games_reseated": int(seated),
            "games_restored": int(restored),
            "setup_episodes_restored": int(staged),
            "move_raw_model_state_digest": observed_move,
            "setup_raw_model_state_digest": self._setup_digest(),
            "cell_digest": self.cell.digest,
            "queue_depth": len(self.setup_trainer.queue),
        }

    # -- documents ---------------------------------------------------------

    def identity_document(self) -> dict:
        return {
            "runner_version": TANDEM_RUNNER_VERSION,
            "run_id": self.config.run_id,
            "iteration": int(self.iteration),
            "move_raw_model_state_digest": self._move_digest(),
            "move_ema_model_state_digest": self.move_ema_digest(),
            "setup_raw_model_state_digest": self._setup_digest(),
            "setup_ema_model_state_digest": self.setup_ema_digest(),
            "cell_digest": self.cell.digest,
            "cell_iteration": int(self.cell.iteration),
            "setup_snapshot_iteration": int(self.provider.snapshot_iteration),
            "start_identity": {
                key: self.start.identity[key]
                for key in ("path", "file_sha256", "model_state_digest", "parameter_count")
            },
            "elapsed_active_training_seconds": float(
                self.elapsed_active_training_seconds
            ),
        }


__all__ = [
    "IterationResult",
    "move_means",
    "PHASE17_SETUP_FAMILY",
    "Phase17RunnerError",
    "Phase17SetupProvider",
    "SETUP_ENTROPY_BASELINE_NATS",
    "TANDEM_RUNNER_VERSION",
    "TandemCollector",
    "TandemConfig",
    "TandemRunner",
]
