"""Phase 14: the bulk-synchronous PPO/KL learner.

Specification source: `02_AGENT_2_FINAL_TRAINING_INTEGRATION.md` sections 4, 5
and 16, over the frozen `phase9_retrieved_values` block.

The objective is not reimplemented
----------------------------------
Every number that decides what the gradient *is* comes from the accepted Phase
9 code: :func:`~stratego.training.phase9_loss.phase9_batch_loss` is the
objective — PPO clipped surrogate + 0.5 value + **0.25 belief auxiliary** +
beta*KL - c_H*H — :class:`~stratego.training.phase9_trainer.KLController` is the
damping controller, and the advantage construction, filter, standardization and
WDL/belief targets are the accepted ones reached through
:mod:`stratego.training.phase9_targets`. This module is the *schedule* around
them.

What is Phase 14's own
----------------------
Three things, all frozen before the first rollout:

1. the learning rate, which is 7.5e-5 while the launch instant is before the
   132-hour mark and 3.75e-5 after it — a wall-clock property of the bound
   iteration, never a function of the optimizer step;
2. the constant 0.001 entropy coefficient;
3. the minibatch shuffle stream, which descends from the Phase 14
   `training_order` domain.

Nothing else about the optimization differs, and the hard vetoes are the
accepted Phase 9 ones with the accepted abort semantics: a breached limit
raises, the affected update does not land, and the counter is incremented.

The segment travels with the iteration
--------------------------------------
An iteration bound under `main` completes its epochs under `main` even if the
transition passes mid-training. The segment is therefore an argument to
:meth:`Phase14Trainer.bind_iteration` and a field of the checkpointed cursor,
not something the trainer re-reads from a clock between minibatches.
"""

from __future__ import annotations

import random
import time
from concurrent.futures import BrokenExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .phase14_contract import (
    ACCEPTED_C1_CONFIG_DIGEST,
    ACCEPTED_C1_PARAMETERS,
    ENTROPY_COEFFICIENT,
    EPOCHS_PER_ROLLOUT,
    INITIAL_KL_BETA,
    MINIBATCH_SIZE,
    OPTIMIZER_CONSTRAINTS,
    PHASE14_CONTRACT_VERSION,
    PHASE14_NAMESPACE,
    PHASE14_POPULATION_VERSION,
    PHASE14_ROLLOUT_VERSION,
    PHASE14_SCHEDULE_VERSION,
    PHASE14_TRAINER_VERSION,
    PRODUCTION_POPULATION,
    STARTING_CHECKPOINT_SHA256,
    STARTING_MODEL_STATE_DIGEST,
    Population,
    contract_digest,
    learner_control_for,
    learning_rate as segment_learning_rate,
    require_segment,
)
from .phase14_schedule import iteration_game_ids
from .phase14_seed import parse_game_id, train_order_seed
from .phase9_contract import CLIP_FRACTION_HARD_LIMIT, KL_HARD_LIMIT
from .phase9_loss import Phase9LossError, phase9_batch_loss
from .phase9_rollout_store import (
    Phase9RolloutReader,
    read_iteration_state,
    sealed_rollout_digest,
    write_iteration_state,
)
from .phase9_targets import (
    collect_iteration_advantages,
    iteration_statistics,
    minibatch_slices,
    train_order_keys,
)
from .phase9_trainer import (
    KLController,
    LoaderTopology,
    _MinibatchPipeline,
    batch_digest,
    unpack_batch,
)


class Phase14TrainerError(RuntimeError):
    """Raised when a Phase 14 optimization step may not proceed."""


def rollout_identity(iteration: int) -> str:
    return f"{PHASE14_ROLLOUT_VERSION}|ns={PHASE14_NAMESPACE}|it={int(iteration):04d}"


# ---------------------------------------------------------------------------
# The train order
# ---------------------------------------------------------------------------


def epoch_order(keys, iteration: int, epoch: int) -> tuple:
    """Positions into `keys` for one optimizer epoch, in consumption order.

    `random.Random(train_order_seed(iteration, epoch)).shuffle` over the index
    list: the accepted Phase 9 shuffle *mechanism* on the Phase 14 stream.
    Returning indices rather than keys keeps the shuffle independent of what an
    example carries.
    """
    order = list(range(len(keys)))
    random.Random(train_order_seed(int(iteration), int(epoch))).shuffle(order)
    return tuple(order)


def minibatch_keys(keys, iteration: int, epoch: int, cursor_index: int, size: int) -> tuple:
    order = epoch_order(keys, iteration, epoch)
    slices = minibatch_slices(len(keys), size)
    if not 0 <= cursor_index < len(slices):
        raise Phase14TrainerError(
            f"minibatch index {cursor_index} is outside 0..{len(slices) - 1}"
        )
    start, stop = slices[cursor_index]
    return tuple(keys[position] for position in order[start:stop])


# ---------------------------------------------------------------------------
# The cursor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cursor:
    """Where one bound iteration's optimization has got to.

    Persisted whole in every hot checkpoint: "epoch 1, minibatch 37 of 96 over
    sealed digest X" is the only description of mid-iteration progress that a
    resume can act on without replaying updates or skipping them.
    """

    iteration: int
    segment: str
    sealed_rollout_digest: str
    total_examples: int
    minibatch_size: int
    epochs: int
    epoch: int = 0
    minibatch_index: int = 0
    examples_consumed: int = 0

    @property
    def minibatches_per_epoch(self) -> int:
        return len(minibatch_slices(self.total_examples, self.minibatch_size))

    @property
    def finished(self) -> bool:
        return self.epoch >= self.epochs

    def advance(self, consumed: int) -> "Cursor":
        index = self.minibatch_index + 1
        epoch = self.epoch
        if index >= self.minibatches_per_epoch:
            index = 0
            epoch += 1
        return Cursor(
            iteration=self.iteration,
            segment=self.segment,
            sealed_rollout_digest=self.sealed_rollout_digest,
            total_examples=self.total_examples,
            minibatch_size=self.minibatch_size,
            epochs=self.epochs,
            epoch=epoch,
            minibatch_index=index,
            examples_consumed=self.examples_consumed + int(consumed),
        )

    def to_dict(self) -> dict:
        return {
            "iteration": int(self.iteration),
            "segment": self.segment,
            "sealed_rollout_digest": self.sealed_rollout_digest,
            "total_examples": int(self.total_examples),
            "minibatch_size": int(self.minibatch_size),
            "epochs": int(self.epochs),
            "epoch": int(self.epoch),
            "minibatch_index": int(self.minibatch_index),
            "examples_consumed": int(self.examples_consumed),
            "minibatches_per_epoch": self.minibatches_per_epoch,
            "finished": self.finished,
        }

    @staticmethod
    def from_dict(payload: dict) -> "Cursor":
        return Cursor(
            iteration=int(payload["iteration"]),
            segment=require_segment(str(payload["segment"])),
            sealed_rollout_digest=str(payload["sealed_rollout_digest"]),
            total_examples=int(payload["total_examples"]),
            minibatch_size=int(payload["minibatch_size"]),
            epochs=int(payload["epochs"]),
            epoch=int(payload["epoch"]),
            minibatch_index=int(payload["minibatch_index"]),
            examples_consumed=int(payload["examples_consumed"]),
        )

    @staticmethod
    def start(
        *,
        iteration: int,
        segment: str,
        sealed_rollout_digest: str,
        total_examples: int,
        epochs: int = EPOCHS_PER_ROLLOUT,
        minibatch_size: int = MINIBATCH_SIZE,
    ) -> "Cursor":
        return Cursor(
            iteration=int(iteration),
            segment=require_segment(segment),
            sealed_rollout_digest=str(sealed_rollout_digest),
            total_examples=int(total_examples),
            minibatch_size=int(minibatch_size),
            epochs=int(epochs),
        )


# ---------------------------------------------------------------------------
# The sealed rollout
# ---------------------------------------------------------------------------


@dataclass
class SealedRollout:
    """One verified iteration, as a trainable view."""

    root: Path
    iteration: int
    segment: str
    sealed_rollout_digest: str
    behavior_snapshot_id: str
    behavior_checkpoint_sha256: str
    games: int
    keys: tuple
    statistics: object
    reader: object
    verifications: dict
    keys_set: set = field(default_factory=set)

    def __post_init__(self) -> None:
        self.keys_set = set(self.keys)

    @property
    def namespace(self) -> str:
        return PHASE14_NAMESPACE

    @property
    def rollout_id(self) -> str:
        return rollout_identity(self.iteration)

    @property
    def learner_decisions(self) -> int:
        return len(self.keys)

    def to_dict(self) -> dict:
        return {
            "namespace": PHASE14_NAMESPACE,
            "iteration": int(self.iteration),
            "segment": self.segment,
            "rollout_id": self.rollout_id,
            "sealed_rollout_digest": self.sealed_rollout_digest,
            "behavior_snapshot_id": self.behavior_snapshot_id,
            "behavior_checkpoint_sha256": self.behavior_checkpoint_sha256,
            "games": int(self.games),
            "learner_decisions": self.learner_decisions,
            "verifications": dict(self.verifications),
        }


def bind_sealed_rollout(
    root,
    iteration: int,
    *,
    segment: "str | None" = None,
    population: Population = PRODUCTION_POPULATION,
    behavior_snapshot=None,
    expected_model_state_digest: "str | None" = None,
    resuming: bool = False,
) -> SealedRollout:
    """Verify one iteration completely, then hand back a trainable view.

    The digest is recomputed from the committed journal rather than read from
    the manifest, so a state document that claims a rollout it does not hold
    cannot authorize training. `expected_model_state_digest` is the on-policy
    binding: the weights about to be optimized must be the weights that
    collected these games.
    """
    root = Path(root)
    state = read_iteration_state(root, PHASE14_NAMESPACE, iteration)
    if state is None:
        raise Phase14TrainerError(f"iteration {iteration} has no state document at {root}")
    acceptable = ("SEALED", "TRAINING") if resuming else ("SEALED",)
    if state["state"] not in acceptable:
        raise Phase14TrainerError(
            f"iteration {iteration} is {state['state']}, not one of {list(acceptable)}"
        )
    recorded_segment = state.get("segment")
    if segment is None:
        segment = recorded_segment
    if segment is None:
        raise Phase14TrainerError(
            f"iteration {iteration} records no segment; the mixture it was "
            "collected under cannot be inferred after the fact"
        )
    require_segment(segment)
    if recorded_segment is not None and recorded_segment != segment:
        raise Phase14TrainerError(
            f"iteration {iteration} was collected in the {recorded_segment} segment, "
            f"not {segment}"
        )

    reader = Phase9RolloutReader(root, PHASE14_NAMESPACE, iteration)
    recomputed = sealed_rollout_digest(reader.commits)
    if recomputed != state.get("sealed_rollout_digest"):
        raise Phase14TrainerError(
            f"iteration {iteration}: recomputed sealed digest {recomputed} != "
            f"recorded {state.get('sealed_rollout_digest')}"
        )

    expected_games = population.bucket_counts(segment)
    observed: dict = {}
    behavior_ids: set = set()
    behavior_digests: set = set()
    control_problems: list = []
    version_problems: list = []
    for game_id in reader.game_ids:
        metadata = reader.metadata[game_id]
        identity = parse_game_id(game_id)
        bucket = str(identity["bucket"])
        observed[bucket] = observed.get(bucket, 0) + 1
        behavior_ids.add(str(metadata["behavior_snapshot_id"]))
        behavior_digests.add(str(metadata["behavior_checkpoint_sha256"]))
        expected_control = learner_control_for(
            bucket, int(identity["iteration"]), int(identity["ordinal"])
        )
        if str(metadata["learner_control"]) != expected_control:
            control_problems.append(
                f"{game_id}: learner_control {metadata['learner_control']!r} != "
                f"{expected_control!r}"
            )
        for field_name, expected in (
            ("bucket", bucket),
            ("ordinal", int(identity["ordinal"])),
            ("population_version", PHASE14_POPULATION_VERSION),
            ("schedule_version", PHASE14_SCHEDULE_VERSION),
            ("contract_digest", contract_digest()),
            ("namespace", PHASE14_NAMESPACE),
            ("iteration", int(iteration)),
        ):
            if metadata[field_name] != expected:
                version_problems.append(
                    f"{game_id}: {field_name} {metadata[field_name]!r} != {expected!r}"
                )

    problems: list = []
    if observed != expected_games:
        problems.append(
            f"bucket counts {observed} != the frozen {segment} counts {expected_games}"
        )
    if len(behavior_ids) != 1 or len(behavior_digests) != 1:
        problems.append(
            f"iteration mixes behavior identities {sorted(behavior_ids)} / digests "
            f"{sorted(behavior_digests)}"
        )
    if behavior_digests and state.get("behavior_checkpoint_sha256") not in behavior_digests:
        problems.append(
            f"state names behavior checkpoint {state.get('behavior_checkpoint_sha256')!r}, "
            f"the games were collected under {sorted(behavior_digests)}"
        )
    problems.extend(control_problems[:5])
    problems.extend(version_problems[:5])
    if problems:
        raise Phase14TrainerError(
            f"iteration {iteration} failed ownership verification: {'; '.join(problems)}"
        )

    behavior_id = sorted(behavior_ids)[0]
    behavior_digest = sorted(behavior_digests)[0]
    on_policy = None
    if behavior_snapshot is not None:
        if behavior_snapshot.checkpoint_sha256 != behavior_digest:
            raise Phase14TrainerError(
                f"iteration {iteration} was collected under {behavior_digest}, but the "
                f"supplied snapshot is {behavior_snapshot.checkpoint_sha256}"
            )
        if behavior_snapshot.logical_identity != behavior_id:
            raise Phase14TrainerError(
                f"the supplied snapshot is {behavior_snapshot.logical_identity!r}, the "
                f"iteration was collected by {behavior_id!r}"
            )
        behavior_snapshot.assert_frozen()
        on_policy = behavior_snapshot.loaded_state_dict_digest
    if expected_model_state_digest is not None:
        if on_policy is None:
            raise Phase14TrainerError(
                "an on-policy check needs the behavior snapshot whose weights "
                "collected the iteration"
            )
        if on_policy != expected_model_state_digest:
            raise Phase14TrainerError(
                f"iteration {iteration} was collected by weights {on_policy}, but the "
                f"trainer holds {expected_model_state_digest}; PPO may not consume "
                "another policy's rollout as if it were its own"
            )

    advantages_by_key, sequences_by_game, target_problems = collect_iteration_advantages(reader)
    if target_problems:
        raise Phase14TrainerError(
            f"iteration {iteration}: target construction reported "
            f"{len(target_problems)} problem(s): {target_problems[:3]}"
        )
    statistics = iteration_statistics(
        advantages_by_key,
        namespace=PHASE14_NAMESPACE,
        iteration=iteration,
        sealed_rollout_digest=recomputed,
        games=len(reader.game_ids),
    )
    keys = train_order_keys(reader, sequences_by_game)
    if len(keys) != len(advantages_by_key):
        raise Phase14TrainerError(
            f"iteration {iteration}: {len(keys)} train-order keys but "
            f"{len(advantages_by_key)} advantages"
        )
    declared = sum(
        int(reader.metadata[game_id]["learner_decision_count"])
        for game_id in reader.game_ids
    )
    if declared != len(keys):
        raise Phase14TrainerError(
            f"iteration {iteration}: metadata declares {declared} learner decisions, "
            f"the rebuilt sequences hold {len(keys)}"
        )
    scheduled = set(iteration_game_ids(iteration, segment, population))
    if set(reader.game_ids) != scheduled:
        raise Phase14TrainerError(
            f"iteration {iteration}: the committed game set is not the scheduled one"
        )
    return SealedRollout(
        root=root,
        iteration=int(iteration),
        segment=segment,
        sealed_rollout_digest=recomputed,
        behavior_snapshot_id=behavior_id,
        behavior_checkpoint_sha256=behavior_digest,
        games=len(reader.game_ids),
        keys=keys,
        statistics=statistics,
        reader=reader,
        verifications={
            "state": state["state"],
            "segment": segment,
            "digest_recomputed_from_commits": True,
            "bucket_counts": observed,
            "single_behavior_identity": True,
            "learner_control_mismatches": len(control_problems),
            "version_mismatches": len(version_problems),
            "on_policy_state_dict_digest": on_policy,
            "declared_learner_decisions": declared,
            "scheduled_set_exact": True,
        },
    )


#: How many times one run may lose its CPU loader pool and rebuild it before
#: the losses stop looking like bad luck and start looking like a sick machine.
MAX_LOADER_POOL_REBUILDS = 16

#: How many rebuild events one checkpoint carries. Bounded because the useful
#: question at hour 140 is "when did this last happen and why", not a complete
#: history of a machine that has rebuilt its pool sixteen times.
LOADER_POOL_EVENT_RETAIN = 16


class Phase14Pipeline(_MinibatchPipeline):
    """The accepted multiprocess loader on the Phase 14 train order.

    The parent computes the plan in exactly one place; overriding it is the
    whole of the Phase 14 difference, so parallelism still cannot influence
    which examples a step consumes or in what order.
    """

    def _plan(self, cursor):
        return minibatch_keys(
            self.rollout.keys,
            self.rollout.iteration,
            cursor.epoch,
            cursor.minibatch_index,
            cursor.minibatch_size,
        )


# ---------------------------------------------------------------------------
# The trainer
# ---------------------------------------------------------------------------


class Phase14Trainer:
    """The Phase 14 optimization loop over sealed on-policy rollouts."""

    def __init__(
        self,
        model,
        *,
        device: str = "mps",
        topology: "LoaderTopology | None" = None,
        initial_kl_beta: float = INITIAL_KL_BETA,
        run_label: str = "",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(device=self.device, dtype=torch.float32)
        self.model.requires_grad_(True)
        self.model.train()
        self._verify_architecture()
        self.topology = topology or LoaderTopology()
        self.run_label = str(run_label)
        constraints = OPTIMIZER_CONSTRAINTS
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=segment_learning_rate("main"),
            betas=tuple(constraints["adam_betas"]),
            eps=float(constraints["adam_epsilon"]),
            weight_decay=float(constraints["weight_decay"]),
        )
        self.gradient_clip_norm = float(constraints["gradient_clip_norm"])
        self.controller = KLController(beta=float(initial_kl_beta))
        self.global_step = 0
        self.examples_consumed = 0
        self.rl_iteration = 0
        self.segment = "main"
        self.cursor: "Cursor | None" = None
        self.counters = {
            "non_finite_losses": 0,
            "non_finite_gradients": 0,
            "non_finite_parameters": 0,
            "illegal_targets": 0,
            "data_mismatches": 0,
            "checkpoint_errors": 0,
            "behavior_identity_mismatches": 0,
            "rollout_identity_mismatches": 0,
            "kl_hard_limit_breaches": 0,
            "clip_fraction_hard_limit_breaches": 0,
            "deadline_stops": 0,
            "loader_pool_rebuilds": 0,
        }
        self.wall_clock = {
            "train_seconds": 0.0,
            "data_wait_seconds": 0.0,
            "checkpoint_seconds": 0.0,
        }
        self.totals = {
            "games_generated": 0,
            "positions_generated": 0,
            "iterations_trained": 0,
        }
        self._pipeline: "Phase14Pipeline | None" = None
        self._bound: "SealedRollout | None" = None
        # A count answers "how sick is this machine"; it does not answer "when"
        # or "why". Both are needed by an operator reading a status at hour 140,
        # and neither survives a crash unless it rides in the checkpoint.
        self.loader_pool_events: list = []

    # -- construction ------------------------------------------------------

    def _verify_architecture(self) -> None:
        parameters = int(self.model.parameter_count())
        if parameters != ACCEPTED_C1_PARAMETERS:
            raise Phase14TrainerError(
                f"the model holds {parameters:,} parameters; Phase 14 continues C1 "
                f"with {ACCEPTED_C1_PARAMETERS:,}"
            )
        summary = self.model.architecture_summary()
        digest = summary.get("config_digest") or summary.get("configuration_digest")
        if digest is not None and digest != ACCEPTED_C1_CONFIG_DIGEST:
            raise Phase14TrainerError(
                f"C1 config digest {digest} != the accepted {ACCEPTED_C1_CONFIG_DIGEST}"
            )

    @property
    def model_state_digest(self) -> str:
        from .phase9_behavior import state_dict_digest

        return state_dict_digest(self.model)

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def schedule_identity(self) -> dict:
        """The frozen schedule position, for the checkpoint payload."""
        return {
            "contract_version": PHASE14_CONTRACT_VERSION,
            "trainer_version": PHASE14_TRAINER_VERSION,
            "namespace": PHASE14_NAMESPACE,
            "rl_iteration": int(self.rl_iteration),
            "segment": self.segment,
            "learning_rate": segment_learning_rate(self.segment),
            "entropy_coefficient": ENTROPY_COEFFICIENT,
            "minibatch_size": MINIBATCH_SIZE,
            "epochs_per_rollout": EPOCHS_PER_ROLLOUT,
            "optimizer": dict(OPTIMIZER_CONSTRAINTS),
            "learning_rate_schedule": "constant within a segment; 132h wall-clock step",
            "cursor": None if self.cursor is None else self.cursor.to_dict(),
        }

    # -- binding -----------------------------------------------------------

    def bind_iteration(
        self, rollout: SealedRollout, *, epochs: int = EPOCHS_PER_ROLLOUT
    ) -> SealedRollout:
        """Bind one sealed rollout and set this iteration's frozen schedule.

        The learning rate is set from the *rollout's* segment, so an iteration
        collected before the transition trains at the main rate even if the
        transition passes while its epochs run.
        """
        if rollout.learner_decisions == 0:
            raise Phase14TrainerError(
                f"iteration {rollout.iteration} holds no learner decisions"
            )
        self.close()
        self._bound = rollout
        self.rl_iteration = int(rollout.iteration)
        self.segment = require_segment(rollout.segment)
        self.cursor = Cursor.start(
            iteration=rollout.iteration,
            segment=rollout.segment,
            sealed_rollout_digest=rollout.sealed_rollout_digest,
            total_examples=rollout.learner_decisions,
            epochs=int(epochs),
        )
        rate = segment_learning_rate(self.segment)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        write_iteration_state(
            rollout.root,
            PHASE14_NAMESPACE,
            rollout.iteration,
            "TRAINING",
            sealed_rollout_digest=rollout.sealed_rollout_digest,
            behavior_snapshot_id=rollout.behavior_snapshot_id,
            behavior_checkpoint_sha256=rollout.behavior_checkpoint_sha256,
            learner_decisions=rollout.learner_decisions,
            segment=rollout.segment,
            learning_rate=rate,
            entropy_coefficient=ENTROPY_COEFFICIENT,
        )
        return rollout

    def resume_iteration(self, rollout: SealedRollout, cursor: Cursor) -> SealedRollout:
        """Re-enter a partially trained iteration at its checkpointed position.

        The counterpart of :meth:`bind_iteration` for the crash path: binding
        afresh would restart the iteration's epochs from zero and consume every
        example a second time, which is neither the accepted bulk-sync semantics
        nor honest about how many updates the run has made.
        """
        self._bound = rollout
        self.rl_iteration = int(rollout.iteration)
        self.restore_cursor(cursor, rollout)
        write_iteration_state(
            rollout.root,
            PHASE14_NAMESPACE,
            rollout.iteration,
            "TRAINING",
            sealed_rollout_digest=rollout.sealed_rollout_digest,
            behavior_snapshot_id=rollout.behavior_snapshot_id,
            behavior_checkpoint_sha256=rollout.behavior_checkpoint_sha256,
            learner_decisions=rollout.learner_decisions,
            segment=rollout.segment,
            learning_rate=segment_learning_rate(rollout.segment),
            entropy_coefficient=ENTROPY_COEFFICIENT,
            resumed_at_epoch=int(cursor.epoch),
            resumed_at_minibatch=int(cursor.minibatch_index),
        )
        return rollout

    def restore_cursor(self, cursor: Cursor, rollout: SealedRollout) -> None:
        """Point the trainer at a checkpointed position inside a bound rollout."""
        if cursor.sealed_rollout_digest != rollout.sealed_rollout_digest:
            raise Phase14TrainerError(
                f"the saved cursor names rollout {cursor.sealed_rollout_digest}, the "
                f"bound iteration is {rollout.sealed_rollout_digest}"
            )
        if cursor.total_examples != rollout.learner_decisions:
            raise Phase14TrainerError(
                f"the saved cursor counts {cursor.total_examples} examples, the bound "
                f"iteration holds {rollout.learner_decisions}"
            )
        self.close()
        self.cursor = cursor
        self.segment = require_segment(cursor.segment)
        rate = segment_learning_rate(self.segment)
        for group in self.optimizer.param_groups:
            group["lr"] = rate

    def mark_iteration_trained(self) -> dict:
        if self._bound is None or self.cursor is None:
            raise Phase14TrainerError("no iteration is bound")
        if not self.cursor.finished:
            raise Phase14TrainerError(
                f"iteration {self._bound.iteration} has not completed its epochs"
            )
        rollout = self._bound
        self.totals["iterations_trained"] += 1
        write_iteration_state(
            rollout.root,
            PHASE14_NAMESPACE,
            rollout.iteration,
            "COMMITTED",
            sealed_rollout_digest=rollout.sealed_rollout_digest,
            behavior_snapshot_id=rollout.behavior_snapshot_id,
            behavior_checkpoint_sha256=rollout.behavior_checkpoint_sha256,
            segment=rollout.segment,
            training_complete=True,
            global_optimizer_step=self.global_step,
            examples_consumed=self.examples_consumed,
        )
        return {
            "iteration": rollout.iteration,
            "segment": rollout.segment,
            "global_optimizer_step": self.global_step,
            "examples_consumed": self.examples_consumed,
        }

    def _next_minibatch(self, pipeline):
        """One minibatch, surviving the death of a CPU loader worker.

        A `ProcessPoolExecutor` whose worker is killed marks itself
        permanently broken and raises `BrokenProcessPool` from the *next*
        submit — a `RuntimeError`, which the runner's recoverable list
        (`OSError`, `TimeoutError`) does not catch and its unrecoverable list
        does not name either. One dead CPU worker therefore killed the whole
        learner, which the Phase 13 rehearsal demonstrated end to end.

        The pool is infrastructure, not state. The minibatch plan is a pure
        function of the cursor, so the pool is rebuilt *at the same cursor* and
        the identical minibatch is rebuilt with it: the optimizer step, the
        epoch, the KL controller and the examples consumed are all untouched,
        and only the processes that packed the bytes are different. A rebuild
        is counted, because a run that quietly rebuilds its pool every minute
        has a machine problem worth seeing in telemetry.
        """
        try:
            return pipeline.next(self.cursor), pipeline
        except BrokenExecutor as error:
            self.counters["loader_pool_rebuilds"] += 1
            self._record_pool_rebuild(error)
            if self.counters["loader_pool_rebuilds"] > MAX_LOADER_POOL_REBUILDS:
                raise Phase14TrainerError(
                    f"the CPU loader pool has been rebuilt "
                    f"{self.counters['loader_pool_rebuilds']} times; the workers are "
                    "not merely unlucky and the run is no longer making progress "
                    "through its epochs"
                ) from error
            pipeline.shutdown()
            self._pipeline = None
            rebuilt = self._ensure_pipeline()
            return rebuilt.next(self.cursor), rebuilt

    def _record_pool_rebuild(self, error: BaseException) -> dict:
        """Log one pool rebuild: when, why, and where in the epoch plan."""
        from .phase14_status import utc_text

        unix = time.time()
        event = {
            "unix": unix,
            "utc": utc_text(unix),
            "rebuild_index": int(self.counters["loader_pool_rebuilds"]),
            "reason": f"{type(error).__name__}: {error}"[:500],
            "global_optimizer_step": int(self.global_step),
            "rl_iteration": int(self.rl_iteration),
            "cursor": None if self.cursor is None else self.cursor.to_dict(),
        }
        self.loader_pool_events.append(event)
        del self.loader_pool_events[:-LOADER_POOL_EVENT_RETAIN]
        return event

    @property
    def loader_pool_open(self) -> bool:
        """Whether a CPU loader pool is currently expected to have workers."""
        return self._pipeline is not None

    def loader_pool_state(self) -> dict:
        """The rebuild history, in the shape the status surface reports."""
        last = self.loader_pool_events[-1] if self.loader_pool_events else {}
        return {
            "rebuilds": int(self.counters["loader_pool_rebuilds"]),
            "max_rebuilds": MAX_LOADER_POOL_REBUILDS,
            "last_rebuild_unix": last.get("unix"),
            "last_rebuild_utc": last.get("utc"),
            "last_rebuild_reason": last.get("reason", ""),
            "events": [dict(event) for event in self.loader_pool_events],
        }

    def _ensure_pipeline(self) -> Phase14Pipeline:
        if self._pipeline is None:
            self._pipeline = Phase14Pipeline(
                self._bound,
                self.cursor,
                topology=self.topology,
                epochs=self.cursor.epochs,
            )
        return self._pipeline

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.shutdown()
            self._pipeline = None

    def __enter__(self) -> "Phase14Trainer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- the loop ----------------------------------------------------------

    def _grad_norm(self) -> torch.Tensor:
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                total = total + parameter.grad.detach().float().pow(2).sum()
        return total.sqrt()

    def _parameter_norm(self) -> torch.Tensor:
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        for parameter in self.model.parameters():
            total = total + parameter.detach().float().pow(2).sum()
        return total.sqrt()

    def _verify_batch(self, keys, packed, rollout: SealedRollout) -> None:
        outside = [key for key in keys if key not in rollout.keys_set]
        if outside:
            self.counters["data_mismatches"] += 1
            raise Phase14TrainerError(
                f"{len(outside)} minibatch key(s) are outside the sealed iteration's "
                f"learner universe (e.g. {outside[:3]})"
            )
        digests = set(packed["behavior_checkpoint_sha256"])
        if digests != {rollout.behavior_checkpoint_sha256}:
            self.counters["behavior_identity_mismatches"] += 1
            raise Phase14TrainerError(
                f"minibatch carries behavior checkpoints {sorted(digests)}, the "
                f"iteration was collected under {rollout.behavior_checkpoint_sha256}"
            )

    def _check_hard_limits(self, *, iteration, epoch, mean_kl, clip_fraction) -> None:
        if mean_kl > KL_HARD_LIMIT:
            self.counters["kl_hard_limit_breaches"] += 1
            raise Phase14TrainerError(
                f"iteration {iteration} epoch {epoch}: mean behavior KL {mean_kl:.6f} "
                f"exceeds the frozen hard limit {KL_HARD_LIMIT}"
            )
        if clip_fraction > CLIP_FRACTION_HARD_LIMIT:
            self.counters["clip_fraction_hard_limit_breaches"] += 1
            raise Phase14TrainerError(
                f"iteration {iteration} epoch {epoch}: PPO clip fraction "
                f"{clip_fraction:.6f} exceeds the frozen hard limit "
                f"{CLIP_FRACTION_HARD_LIMIT}"
            )

    def train_iteration(
        self,
        *,
        updates: "int | None" = None,
        capture_batch_digests: bool = False,
        on_step=None,
        may_start_step=None,
    ) -> list:
        """Run the bound iteration's epochs; return one metric row per update.

        `may_start_step` is the deadline gate. It is consulted *before* each
        step rather than after, because the frozen rule is that no optimizer
        step may **begin** at or after the deadline; a step already begun
        completes and lands.
        """
        if self._bound is None:
            raise Phase14TrainerError("no iteration is bound")
        pipeline = self._ensure_pipeline()
        rollout = self._bound
        coefficient = float(ENTROPY_COEFFICIENT)
        rows: list = []
        performed = 0
        while not self.cursor.finished:
            if updates is not None and performed >= int(updates):
                break
            if may_start_step is not None and not bool(may_start_step()):
                self.counters["deadline_stops"] += 1
                break
            step_started = time.perf_counter()
            (keys, packed, waited), pipeline = self._next_minibatch(pipeline)
            arrays = unpack_batch(packed)
            self._verify_batch(keys, packed, rollout)
            row = {
                "namespace": PHASE14_NAMESPACE,
                "iteration": rollout.iteration,
                "segment": rollout.segment,
                "epoch": int(self.cursor.epoch),
                "minibatch_index": int(self.cursor.minibatch_index),
                "data_wait_seconds": waited,
            }
            if capture_batch_digests:
                row["batch_digest"] = batch_digest(packed)

            tensors = {
                name: torch.from_numpy(np.ascontiguousarray(value)).to(self.device)
                for name, value in arrays.items()
                if name != "learner_side"
            }
            outputs = self.model.forward_observation(tensors["observation"])
            try:
                loss = phase9_batch_loss(
                    outputs,
                    legal_mask=tensors["legal_mask"],
                    sampled_action_model=tensors["sampled_action_model"],
                    behavior_action_probability=tensors["behavior_action_probability"],
                    behavior_probabilities=tensors["behavior_probabilities"],
                    standardized_advantage=tensors["standardized_advantage"],
                    ppo_eligible=tensors["ppo_eligible"],
                    wdl_target=tensors["wdl_target"],
                    belief_target=tensors["belief_target"],
                    belief_mask=tensors["belief_mask"],
                    kl_beta=float(self.controller.beta),
                    entropy_coefficient=coefficient,
                )
            except Phase9LossError:
                self.counters["illegal_targets"] += 1
                raise
            if not loss.all_finite():
                self.counters["non_finite_losses"] += 1
                raise Phase14TrainerError(
                    f"non-finite loss at global step {self.global_step + 1}"
                )

            self.optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            rate = float(self.optimizer.param_groups[0]["lr"])
            pre_clip = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )
            post_clip = self._grad_norm()
            if not bool(torch.isfinite(pre_clip)) or not bool(torch.isfinite(post_clip)):
                self.counters["non_finite_gradients"] += 1
                raise Phase14TrainerError(
                    f"non-finite gradient norm at global step {self.global_step + 1}"
                )
            self.optimizer.step()
            parameter_norm = self._parameter_norm()
            if not bool(torch.isfinite(parameter_norm)):
                self.counters["non_finite_parameters"] += 1
                raise Phase14TrainerError(
                    f"non-finite parameters at global step {self.global_step + 1}"
                )

            size = len(keys)
            self.controller.observe(
                mean_kl=float(loss.kl.detach()),
                examples=size,
                clipped=int(loss.ppo_clipped),
                ppo_examples=int(loss.ppo_examples),
            )
            self.global_step += 1
            self.examples_consumed += size
            self.wall_clock["data_wait_seconds"] += waited
            row.update(loss.to_dict())
            row.update(
                {
                    "global_optimizer_step": int(self.global_step),
                    "examples": size,
                    "examples_consumed": int(self.examples_consumed),
                    "advantage_retention": float(loss.ppo_examples) / size,
                    "grad_norm_pre_clip": float(pre_clip),
                    "grad_norm_post_clip": float(post_clip),
                    "parameter_norm": float(parameter_norm),
                    "learning_rate": rate,
                    "entropy_coefficient": coefficient,
                    "kl_beta": float(self.controller.beta),
                    "step_seconds": time.perf_counter() - step_started,
                }
            )
            previous_epoch = int(self.cursor.epoch)
            self.cursor = self.cursor.advance(size)
            performed += 1
            self.wall_clock["train_seconds"] += row["step_seconds"]

            if int(self.cursor.epoch) != previous_epoch:
                entry = self.controller.update(
                    iteration=rollout.iteration, epoch=previous_epoch
                )
                row["epoch_mean_kl"] = entry["mean_epoch_kl"]
                row["epoch_clip_fraction"] = entry["epoch_clip_fraction"]
                row["kl_beta_after_epoch"] = float(self.controller.beta)
                self._check_hard_limits(
                    iteration=rollout.iteration,
                    epoch=previous_epoch,
                    mean_kl=entry["mean_epoch_kl"],
                    clip_fraction=entry["epoch_clip_fraction"],
                )
            rows.append(row)
            if on_step is not None:
                on_step(self, row)
        return rows

    # -- state -------------------------------------------------------------

    def trainer_state(self) -> dict:
        """Everything a hot checkpoint needs from the optimization side."""
        return {
            "global_optimizer_step": int(self.global_step),
            "examples_consumed": int(self.examples_consumed),
            "rl_iteration": int(self.rl_iteration),
            "segment": self.segment,
            "learning_rate": self.learning_rate,
            "entropy_coefficient": ENTROPY_COEFFICIENT,
            "kl_beta": float(self.controller.beta),
            "kl_controller_state": self.controller.to_dict(),
            "cursor": None if self.cursor is None else self.cursor.to_dict(),
            "counters": dict(self.counters),
            "loader_pool": self.loader_pool_state(),
            "wall_clock": dict(self.wall_clock),
            "totals": dict(self.totals),
        }

    def restore_state(self, payload: dict) -> None:
        """Restore optimizer, controller, counters and schedule position."""
        self.optimizer.load_state_dict(payload["optimizer_state"])
        trainer_state = payload.get("trainer_state", payload)
        self.controller = KLController.from_dict(trainer_state["kl_controller_state"])
        self.global_step = int(trainer_state["global_optimizer_step"])
        self.examples_consumed = int(trainer_state["examples_consumed"])
        self.rl_iteration = int(trainer_state["rl_iteration"])
        self.segment = require_segment(str(trainer_state.get("segment", "main")))
        self.counters.update(trainer_state.get("counters", {}))
        self.loader_pool_events = list(
            (trainer_state.get("loader_pool") or {}).get("events", [])
        )[-LOADER_POOL_EVENT_RETAIN:]
        self.wall_clock.update(trainer_state.get("wall_clock", {}))
        self.totals.update(trainer_state.get("totals", {}))
        cursor = trainer_state.get("cursor")
        self.cursor = Cursor.from_dict(cursor) if cursor else None
        for group in self.optimizer.param_groups:
            group["lr"] = segment_learning_rate(self.segment)


def load_starting_model(checkpoint_path=None, *, device: str = "mps"):
    """Load the accepted Phase 9 C1 checkpoint, digest-checked.

    Refuses anything whose file SHA-256 or model-state digest is not the
    accepted one. "Wrong starting model" is on the frozen list of unrecoverable
    integrity failures, so it is detected here rather than after an hour of
    training.
    """
    from .phase14_contract import STARTING_CHECKPOINT, file_sha256, repository_root
    from .phase9_checkpoint import model_from_payload, read_phase9_payload

    path = Path(checkpoint_path or (repository_root() / STARTING_CHECKPOINT))
    if not path.exists():
        raise Phase14TrainerError(f"the accepted starting checkpoint is missing at {path}")
    digest = file_sha256(path)
    if digest != STARTING_CHECKPOINT_SHA256:
        raise Phase14TrainerError(
            f"{path} has SHA-256 {digest}, not the accepted {STARTING_CHECKPOINT_SHA256}; "
            "Phase 14 may not start from another checkpoint"
        )
    payload = read_phase9_payload(path)
    model = model_from_payload(payload, device=device)
    from .phase9_behavior import state_dict_digest

    observed = state_dict_digest(model)
    if observed != STARTING_MODEL_STATE_DIGEST:
        raise Phase14TrainerError(
            f"the loaded model-state digest {observed} != the accepted "
            f"{STARTING_MODEL_STATE_DIGEST}"
        )
    return model


def trainer_semantics() -> dict:
    return {
        "trainer_version": PHASE14_TRAINER_VERSION,
        "contract_version": PHASE14_CONTRACT_VERSION,
        "objective": "stratego.training.phase9_loss.phase9_batch_loss, unchanged",
        "belief_auxiliary": "retained, weight 0.25, accepted targets",
        "controller": "stratego.training.phase9_trainer.KLController, unchanged",
        "targets": "stratego.training.phase9_targets, unchanged",
        "loader": "the accepted Phase 9 multiprocess pipeline, Phase 14 train order",
        "phase14_own": [
            "7.5e-5 main / 3.75e-5 late, by the iteration's launch segment",
            "constant 0.001 entropy coefficient",
            "training_order stream from the Phase 14 roots",
        ],
        "ema": "absent, as in the accepted Phase 9 system",
        "search": "absent; no search-derived supervision exists in this path",
        "hard_vetoes": {
            "kl_hard_limit": KL_HARD_LIMIT,
            "clip_fraction_hard_limit": CLIP_FRACTION_HARD_LIMIT,
            "abort_semantics": (
                "a breached limit raises at the epoch boundary, the affected update "
                "does not land, and the counter is incremented"
            ),
        },
    }
