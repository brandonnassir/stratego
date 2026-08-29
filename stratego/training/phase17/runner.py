"""Phase 17: the bulk-synchronous tandem move/setup runner.

Specification sources: operator decision D10 sections 3-7, common contract
sections 5, 6, 7, 8 and 9.

One iteration, in the contractual order
---------------------------------------
```text
1  bind the current raw move and setup snapshot identities
2  regenerate both setup pools from the current raw setup snapshot
3  advance the persistent population until exactly the move budget lands
4  replacement games draw both setups from the bound raw setup distribution
   and attach both behavior episodes
5  a finished game enqueues both of its setup episodes
6  build boundary-bootstrapped targets and run ONE move epoch
7  train five setup epochs on EVERY episode that completed, or SKIP explicitly
8  advance the move KL controller and both raw-to-EMA states
9  atomically checkpoint and emit the telemetry row
10 rebind every active game to the newly updated raw move snapshot
```

Steps 3-5 happen inside the collector, which is why the enqueue hook is an
override of `_retire` rather than a scan afterwards: `_retire` is the single
point at which a game is known to be finished *and* still addressable, so
hooking it makes "a completed game always enqueues both episodes" structural.
Scanning for disappeared game ids afterwards would work until the first game
that finished and was replaced inside the same window.

Step 2 is unconditional under D10
----------------------------------
Both pools are rebound to the live raw setup snapshot at the top of every
global iteration and their unused candidates discarded, rather than only when
the setup digest happens to have moved. D10 section 4: 512 fresh samples per
side, regenerated at every shared tandem iteration, and refilled within an
iteration only from that same snapshot. Discarding is not waste -- a leftover
candidate carries the OLD behavior probabilities, and reusing it under a new
digest would make the PPO ratio's denominator a distribution the model no
longer has.

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
setup learner has updated since. The move policy is the opposite: it is
resolved per decision. Mixing these up in either direction silently breaks the
PPO ratio of whichever half was got wrong, so the two are held by different
mechanisms -- an immutable `SetupEpisode` and a mutable policy cell.

What D10 removed from step 7
-----------------------------
The fixed setup quota, the two-budget warm-up, the max-age selection and the
backlog alarm are gone, along with the pre-window overflow check they needed.
Those existed to keep a *fixed-size* setup batch unbiased while episodes were
carried across iterations. Draining the buffer completely has no such problem:
the batch is exactly what arrived in this window, so there is no count to hold
constant and nothing to be starved of. The only skip left is "no game
completed", and it is recorded.
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
from .setup_contract import (
    PRODUCTION_RUN_ID,
    SETUP_EQUATION_VERSION,
    SETUP_POOL_SIZE_PER_SIDE,
    SETUP_RECIPE_VERSION,
    SetupTrainingConfig,
)
from .setup_episode import attach_setup_episodes
from .setup_learning import SetupTrainer
from .setup_metrics import diversity_profile
from .setup_model import build_setup_model
from .setup_sampling import SetupPool
from .supervisor import CollapseSupervisor, MODE_PRODUCTION
from .transition_collector import FixedTransitionCollector

TANDEM_RUNNER_VERSION = "phase17_tandem_runner_v2"


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
            "regeneration_cadence": "every global tandem iteration (D10 section 4)",
            "unused_discard_rule": (
                "unused pool candidates are DISCARDED at every regeneration and "
                "on resume, never carried: a candidate drawn under an older "
                "snapshot carries that snapshot's behavior probabilities, and a "
                "resume regenerates under the checkpointed snapshot from the same "
                "per-game seed, so the setup a game receives is unchanged"
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
    #: The move schedule's horizon. The setup half no longer reads it: D10's
    #: alpha is `0.1 * n**-0.3` with no dependence on the run length.
    total_iterations: int
    move_budget: int = WINDOW_TRANSITIONS
    population: int = 256
    pool_size_per_side: int = SETUP_POOL_SIZE_PER_SIDE
    setup_minibatch_episodes: int = 64
    move_device: str = "cpu"
    setup_device: str = "cpu"
    move_minibatch_size: int = 512
    setup_model_seed: int = 17
    work_package: str = "phase17"

    def __post_init__(self) -> None:
        if self.total_iterations < 1:
            raise Phase17RunnerError("the frozen horizon N must be >= 1")
        if self.population < 1:
            raise Phase17RunnerError("the population must be >= 1")
        if self.pool_size_per_side < 1:
            raise Phase17RunnerError("the setup pool must hold at least one side")

    @property
    def recipe(self) -> str:
        return SETUP_RECIPE_VERSION

    @property
    def is_production(self) -> bool:
        return self.run_id == PRODUCTION_RUN_ID

    @property
    def reference_iteration(self) -> int:
        return reference_iteration(self.total_iterations)

    def setup_config(self) -> SetupTrainingConfig:
        """The setup half's config, with only the fields the runner owns filled in."""
        return SetupTrainingConfig(
            run_id=self.run_id,
            device=self.setup_device,
            minibatch_episodes=self.setup_minibatch_episodes,
            pool_size_per_side=self.pool_size_per_side,
        )

    def document(self) -> dict:
        return {
            "recipe": SETUP_RECIPE_VERSION,
            "runner_version": TANDEM_RUNNER_VERSION,
            "work_package": self.work_package,
            "run_id": self.run_id,
            "horizon": {
                "N": int(self.total_iterations),
                "n_ref": self.reference_iteration,
                "applies_to": "the move LR and move entropy schedules only",
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
                "pool_cadence": "regenerated every global tandem iteration",
                "consumption": (
                    "every episode whose game completed in the iteration, both "
                    "sides, exactly once; no quota, warm-up, age selection or "
                    "backlog balancing"
                ),
                "minibatch_episodes": int(self.setup_minibatch_episodes),
                "epochs_per_iteration": 5,
                "device": self.setup_device,
                # In the config digest because it determines the initial random
                # setup model, and therefore every descriptive baseline the
                # telemetry is read against. A digest that did not cover it
                # would call two different starting distributions the same run.
                "model_seed": int(self.setup_model_seed),
                "alpha_formula": "0.1 * n**-0.3, n the global tandem iteration",
                "behavior_kl": "fixed reverse coefficient 0.1, no controller",
                "advantage": "(outcome - E[v]) + alpha(n) * (I - h_behavior)",
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
    buffer_telemetry: dict = field(default_factory=dict)
    pool_discarded: int = 0
    bound_digests: dict = field(default_factory=dict)
    verdicts: list = field(default_factory=list)


class TandemRunner:
    """The whole Phase 17 system: one move learner, one setup learner, one loop."""

    def __init__(
        self,
        config: TandemConfig,
        *,
        supervisor_mode: str = MODE_PRODUCTION,
        move_start_path: "str | Path | None" = None,
        root: "str | Path" = ".",
        setup_model=None,
    ) -> None:
        self.config = config
        self.supervisor_mode = supervisor_mode
        # D10 section 3: production reinitializes from Phase 9 plus a NEWLY
        # RANDOM setup model, and no setup state from a rehearsal may enter it.
        # `setup_model` exists so a test can inject a tiny model; the production
        # run ID refuses it outright rather than trusting a caller to pass a
        # fresh one.
        if setup_model is not None and config.is_production:
            raise Phase17RunnerError(
                f"run {config.run_id!r} is the D10 production lineage and must "
                "build its own setup model from scratch under the recorded "
                "seed; an injected setup model could carry rehearsal state"
            )

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
        #: The freshly initialised setup identity, before any update. Recorded
        #: so a reader can prove the run started from a random setup model
        #: rather than from a rehearsal's weights.
        self.setup_start_digest = self._setup_digest()

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
        self.enqueue_rejections: list = []
        #: Verdicts raised mid-window, drained into the iteration's verdict
        #: list by `_supervise`. Without this they would reach the supervisor
        #: but never the telemetry row, and a stop would appear in the run with
        #: no row saying which iteration armed it.
        self._mid_window_verdicts: list = []
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
        """Every completed setup episode, counted, and never silently dropped.

        Agent 4C correction 5. A refusal used to be appended to a list and the
        window carried on. But every refusal reason -- incomplete, duplicate,
        already consumed -- means a finished game's outcome will not reach the
        setup learner, so continuing produces a setup half that trained on a
        different set of games than its telemetry describes. The rejection is
        still recorded for diagnosis; what it now also does is arm `I7`, and
        the run stops at the end of this iteration after its safe checkpoint.
        """
        if self.setup_trainer.queue.enqueue(episode):
            return
        rejection = {
            "identity": episode.identity(),
            "state": episode.state,
            "reason": episode.rejected_reason,
            "iteration": int(self.iteration),
        }
        self.enqueue_rejections.append(rejection)
        self._mid_window_verdicts.append(
            self.supervisor.check_setup_outcome_accounting(
                rejected=True,
                evidence={
                    "rejected_episode": rejection,
                    "rejections_this_run": len(self.enqueue_rejections),
                    "rule": (
                        "D10 section 7: loss or duplication of a setup outcome "
                        "is an integrity stop, not a counted warning"
                    ),
                },
            )
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

        # 1  bind the current raw move and setup snapshot identities
        bound_move_digest = self.cell.digest
        bound_setup_digest = self._setup_digest()

        # 2  regenerate both setup pools from the current raw setup snapshot.
        #    Unconditional: D10 section 4 asks for a fresh 512-per-side pool at
        #    every global iteration, and refills within the iteration then come
        #    from this same snapshot because `SetupPool.take` falls back to the
        #    model and digest bound here.
        generation_started = time.perf_counter()
        result.pool_discarded = self.provider.rebind(
            self.setup_model,
            model_state_digest=bound_setup_digest,
            snapshot_iteration=n,
        )
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

        # 7  five setup epochs on everything that completed, or an explicit skip
        setup_started = time.perf_counter()
        update = self.setup_trainer.update(global_iteration=n)
        result.setup_update = update
        result.setup_skipped = bool(update.skipped)
        result.setup_skip_reason = update.skip_reason
        if update.skipped:
            self.setup_skips += 1
        else:
            self.setup_updates += 1
        result.seconds["setup_optimization"] = time.perf_counter() - setup_started
        result.buffer_telemetry = self.setup_trainer.queue.telemetry(
            self.setup_trainer.setup_iteration
        ).__dict__

        # 8  the move KL controller and both EMAs are advanced by the two
        #    trainers themselves; nothing is re-stepped here. The setup half has
        #    no controller to step -- its coefficient is fixed at 0.1.

        # 9  telemetry / checkpointing is the caller's, so a smoke and a
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
        # The setup snapshot is NOT rebound here. Step 2 of the next iteration
        # rebinds it, which is the single point at which the pool is
        # regenerated -- two rebind sites would give a game's setup two possible
        # provenances for the same iteration.
        result.bound_digests = {
            "move": bound_move_digest,
            "setup": bound_setup_digest,
        }

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

    # -- guards ------------------------------------------------------------

    def _supervise(self, result: IterationResult) -> list:
        """Every D10 section 7 reading, folded into the supervisor.

        The integrity family (`I*`) still stops the run. Everything statistical
        is a warning: D10 makes EWR decline, high but finite KL, entropy
        decline, low diversity and setup concentration telemetry, because the
        12-hour learning curve is the experiment.
        """
        # Anything raised inside the window (a refused completed setup episode)
        # is carried out here, so the iteration's row shows what armed a stop.
        verdicts = list(self._mid_window_verdicts)
        self._mid_window_verdicts = []
        ledger = self.collector.participant_ledger()
        verdicts.extend(self.supervisor.check_participant_ledger(ledger))
        verdicts.append(
            self.supervisor.check_setup_generation(
                legality_failures=self.provider.legality_failures,
                orientation_failures=self.provider.orientation_failures,
                fallback_attempts=self.provider.fallback_attempts,
            )
        )
        # A fixed-transition count violation is one of D10's named stops, and
        # it is checked against the window's own harvest rather than trusted:
        # a window that emitted the wrong number of rows has already trained on
        # them by the time this runs, so the run must not continue.
        harvested = int(result.window.transitions_harvested) if result.window else 0
        verdicts.append(
            self.supervisor.check_transition_count(
                harvested=harvested, budget=int(self.config.move_budget)
            )
        )
        # MoveUpdate.means keys carry a `mean_` prefix -- `mean_behavior_kl`, not
        # `behavior_kl`. Reading the unprefixed name silently yields 0.0, which
        # would feed the move-KL and move-entropy predicates a constant zero and
        # make P2 and P6 unreadable. `move_means` fails loudly instead.
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
            verdicts.append(
                self.supervisor.observe_setup_kl(float(update.final_epoch_kl))
            )
        return verdicts

    # -- the descriptive setup-concentration reading -----------------------

    def concentration_reading(self, *, samples: int = 160, label: str = "tandem") -> dict:
        """A descriptive reading of the live raw setup snapshot.

        Drawn from the *current* raw setup policy and scored with exactly the
        setup half's metric functions, at exactly Agent 3's sample shape: its
        soak drew 160 Red plus 160 Blue and profiled the 320 together, so the
        `matched` profile below is the one its trajectory is comparable with.
        The per-colour profiles are extra, because a Red-only or Blue-only
        collapse and a symmetric one are different failures and the pooled
        number hides which happened.

        Under D10 nothing here can stop a run. Concentration and entropy inform
        interpretation of the 12-hour curve and no longer make a checkpoint
        ineligible by themselves.

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
            "recipe": SETUP_RECIPE_VERSION,
            "run_id": self.config.run_id,
            "work_package": self.config.work_package,
            "iteration": int(self.iteration),
            "start_identity": {
                key: self.start.identity[key]
                for key in ("path", "file_sha256", "model_state_digest", "parameter_count")
            },
            "setup_start_model_state_digest": self.setup_start_digest,
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
            # A scalar plus its direction, never controller state: D10 section 1
            # requires the checkpoint to call this a fixed coefficient.
            "setup_behavior_kl": setup_state["setup_behavior_kl"],
            "move_scheduler_position": {
                "iteration": int(self.iteration),
                "N": int(self.config.total_iterations),
                "n_ref": self.config.reference_iteration,
                "lr": self.start.horizon.learning_rate(max(1, self.iteration)),
                "c_H": self.start.horizon.entropy_coefficient(max(1, self.iteration)),
            },
            "setup_scheduler_position": {
                "iteration": int(self.setup_trainer.setup_iteration),
                "alpha": self.setup_config.alpha(
                    max(1, self.setup_trainer.setup_iteration)
                ),
                "alpha_formula": "0.1 * n**-0.3",
            },
            "move_optimizer_step_count": int(self.move_trainer.global_step),
            "setup_optimizer_step_count": int(self.setup_trainer.optimizer_step_count),
            "rng_namespaces": self.rng_namespaces(),
            "active_games": active,
            "active_game_setup_episodes": self.provider.capture_open_episodes(),
            "boundary_carry_state": self.collector.state(),
            "completed_setup_buffer": setup_state["completed_setup_buffer"],
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
                        "setup_behavior_kl",
                        "completed_setup_buffer",
                    )
                },
                "config_digest": payload["setup_config_digest"],
                "recipe": payload["recipe"],
                "setup_contract_version": self.setup_config.document()[
                    "setup_contract_version"
                ],
                "setup_equation_version": SETUP_EQUATION_VERSION,
                "run_id": payload["run_id"],
                "setup_iteration": int(payload["setup_scheduler_position"]["iteration"]),
                "setup_optimizer_step_count": int(payload["setup_optimizer_step_count"]),
                "setup_updates": int(payload["collector_counters"]["setup_updates"]),
                "setup_skips": int(payload["collector_counters"]["setup_skips"]),
                "setup_ema_updates": int(payload["setup_ema_updates"]),
            }
        )

        self.iteration = int(payload["iteration"])
        self.elapsed_active_training_seconds = float(
            payload["elapsed_active_training_seconds"]
        )
        counters = payload["collector_counters"]
        self.setup_updates = int(counters["setup_updates"])
        self.setup_skips = int(counters["setup_skips"])
        self.enqueue_rejections = list(counters["enqueue_rejections"])
        self._mid_window_verdicts = []
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
            "completed_setup_buffer_depth": len(self.setup_trainer.queue),
        }

    # -- documents ---------------------------------------------------------

    def identity_document(self) -> dict:
        return {
            "recipe": SETUP_RECIPE_VERSION,
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
            "setup_start_model_state_digest": self.setup_start_digest,
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
