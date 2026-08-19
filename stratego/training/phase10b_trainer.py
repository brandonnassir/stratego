"""Optional Phase 10B: the bounded PPO/KL fine-tuning loop.

Specification source: `OPTIONAL_PHASE_10B_SETUP_CONDITIONED_FINE_TUNING_AGENT.md`
sections 5, 9, 10, 11 and 13.

The objective is not reimplemented
----------------------------------
Every number that decides what the gradient *is* comes from the accepted
Phase 9 code: :func:`~stratego.training.phase9_loss.phase9_batch_loss` is the
objective, :class:`~stratego.training.phase9_trainer.KLController` is the
damping controller, and the advantage construction, filter, standardization
and WDL/belief targets are the accepted Phase 4/8/9 ones reached through
:mod:`stratego.training.phase9_targets`. This module is the *schedule* around
them, which is exactly the scope the plan gives Phase 10B.

What is Phase 10B's own
-----------------------
Three things, all frozen before the first rollout:

1. the linear learning-rate decay from 0.25x to 0.10x of the accepted Phase 9
   canonical starting rate, replacing Phase 9's constant schedule;
2. the 0.0010 -> 0.0005 entropy schedule;
3. the minibatch shuffle stream, which descends from the Phase 10B
   `training_order` domain.

Nothing else about the optimization differs, and the hard vetoes are the
accepted Phase 9 ones with the accepted abort semantics: a breached limit
raises, the affected update does not land, and the counter that Gate F reads
is incremented.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .phase10b_contract import (
    ACCEPTED_C1_CONFIG_DIGEST,
    ACCEPTED_PHASE9_PARAMETERS,
    EPOCHS_PER_ROLLOUT,
    MINIBATCH_SIZE,
    OPTIMIZER_CONSTRAINTS,
    PHASE10B_CONTRACT_VERSION,
    PHASE10B_NAMESPACE,
    PHASE10B_POPULATION_VERSION,
    PHASE10B_ROLLOUT_VERSION,
    PHASE10B_SCHEDULE_VERSION,
    PHASE10B_TRAINER_VERSION,
    PHASE9_CANONICAL_INITIAL_KL_BETA,
    Phase10BContractError,
    bucket_counts,
    contract_digest,
    entropy_coefficient,
    learner_control_for,
    learning_rate as scheduled_learning_rate,
)
from .phase10b_schedule import iteration_game_ids
from .phase10b_seed import parse_game_id, train_order_seed
from .phase9_contract import (
    CLIP_FRACTION_HARD_LIMIT,
    KL_HARD_LIMIT,
)
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


class Phase10BTrainerError(RuntimeError):
    """Raised when a Phase 10B optimization step may not proceed."""


def rollout_identity(iteration: int) -> str:
    return f"{PHASE10B_ROLLOUT_VERSION}|ns={PHASE10B_NAMESPACE}|it={int(iteration):03d}"


# ---------------------------------------------------------------------------
# The train order
# ---------------------------------------------------------------------------


def epoch_order(keys, iteration: int, epoch: int) -> tuple:
    """Positions into `keys` for one optimizer epoch, in consumption order.

    `random.Random(train_order_seed(iteration, epoch)).shuffle` over the index
    list, the accepted Phase 9 shuffle mechanism on the Phase 10B stream.
    Returning indices rather than keys keeps the shuffle independent of what
    an example carries.
    """
    import random

    order = list(range(len(keys)))
    random.Random(train_order_seed(int(iteration), int(epoch))).shuffle(order)
    return tuple(order)


def minibatch_keys(keys, iteration: int, epoch: int, cursor_index: int, size: int) -> tuple:
    order = epoch_order(keys, iteration, epoch)
    slices = minibatch_slices(len(keys), size)
    if not 0 <= cursor_index < len(slices):
        raise Phase10BTrainerError(
            f"minibatch {cursor_index} is outside 0..{len(slices) - 1}"
        )
    start, stop = slices[cursor_index]
    return tuple(keys[position] for position in order[start:stop])


@dataclass(frozen=True)
class Cursor:
    """Where a training pass is, in logical terms only."""

    iteration: int
    sealed_rollout_digest: str
    epoch: int
    minibatch_index: int
    examples_consumed: int
    total_examples: int
    minibatch_size: int = MINIBATCH_SIZE
    epochs: int = EPOCHS_PER_ROLLOUT

    @property
    def minibatches_per_epoch(self) -> int:
        return len(minibatch_slices(self.total_examples, self.minibatch_size))

    @property
    def finished(self) -> bool:
        return self.epochs > 0 and self.epoch >= self.epochs

    def advance(self, consumed: int) -> "Cursor":
        index = self.minibatch_index + 1
        epoch = self.epoch
        if index >= self.minibatches_per_epoch:
            index = 0
            epoch += 1
        return Cursor(
            iteration=self.iteration,
            sealed_rollout_digest=self.sealed_rollout_digest,
            epoch=epoch,
            minibatch_index=index,
            examples_consumed=self.examples_consumed + int(consumed),
            total_examples=self.total_examples,
            minibatch_size=self.minibatch_size,
            epochs=self.epochs,
        )

    def to_dict(self) -> dict:
        return {
            "train_order_version": "phase10b_train_order_v1",
            "namespace": PHASE10B_NAMESPACE,
            "iteration": int(self.iteration),
            "sealed_rollout_digest": self.sealed_rollout_digest,
            "epoch": int(self.epoch),
            "minibatch_index": int(self.minibatch_index),
            "examples_consumed": int(self.examples_consumed),
            "total_examples": int(self.total_examples),
            "minibatch_size": int(self.minibatch_size),
            "epochs": int(self.epochs),
        }

    @staticmethod
    def start(*, iteration, sealed_rollout_digest, total_examples, epochs=EPOCHS_PER_ROLLOUT,
              minibatch_size=MINIBATCH_SIZE) -> "Cursor":
        return Cursor(
            iteration=int(iteration),
            sealed_rollout_digest=str(sealed_rollout_digest),
            epoch=0,
            minibatch_index=0,
            examples_consumed=0,
            total_examples=int(total_examples),
            minibatch_size=int(minibatch_size),
            epochs=int(epochs),
        )


# ---------------------------------------------------------------------------
# The sealed rollout
# ---------------------------------------------------------------------------


@dataclass
class SealedRollout:
    """One verified, trainable Phase 10B iteration."""

    root: Path
    iteration: int
    sealed_rollout_digest: str
    behavior_snapshot_id: str
    behavior_checkpoint_sha256: str
    games: int
    keys: tuple
    statistics: object
    reader: Phase9RolloutReader = field(repr=False, default=None)
    verifications: dict = field(default_factory=dict)
    keys_set: frozenset = field(repr=False, default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys_set", frozenset(self.keys))

    @property
    def namespace(self) -> str:
        return PHASE10B_NAMESPACE

    @property
    def rollout_id(self) -> str:
        # The example builder stamps `rollout_identity(namespace, iteration)`
        # from the accepted Phase 9 targets module into every example, so the
        # bound identity has to be compared against that same spelling.
        from .phase9_targets import rollout_identity as phase9_rollout_identity

        return phase9_rollout_identity(PHASE10B_NAMESPACE, self.iteration)

    @property
    def learner_decisions(self) -> int:
        return len(self.keys)

    def to_dict(self) -> dict:
        import hashlib

        hasher = hashlib.sha256()
        for game_id, ply in self.keys:
            hasher.update(f"{game_id}|{ply}\n".encode())
        return {
            "namespace": PHASE10B_NAMESPACE,
            "iteration": int(self.iteration),
            "rollout_id": rollout_identity(self.iteration),
            "sealed_rollout_digest": self.sealed_rollout_digest,
            "behavior_snapshot_id": self.behavior_snapshot_id,
            "behavior_checkpoint_sha256": self.behavior_checkpoint_sha256,
            "games": int(self.games),
            "learner_decisions": int(self.learner_decisions),
            "train_order_keys_digest": hasher.hexdigest(),
            "statistics": self.statistics.to_dict(),
            "verifications": dict(self.verifications),
        }


def bind_sealed_rollout(
    root,
    iteration: int,
    *,
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
    state = read_iteration_state(root, PHASE10B_NAMESPACE, iteration)
    if state is None:
        raise Phase10BTrainerError(
            f"iteration {iteration} has no state document at {root}"
        )
    acceptable = ("SEALED", "TRAINING") if resuming else ("SEALED",)
    if state["state"] not in acceptable:
        raise Phase10BTrainerError(
            f"iteration {iteration} is {state['state']}, not one of {list(acceptable)}"
        )
    reader = Phase9RolloutReader(root, PHASE10B_NAMESPACE, iteration)
    recomputed = sealed_rollout_digest(reader.commits)
    if recomputed != state.get("sealed_rollout_digest"):
        raise Phase10BTrainerError(
            f"iteration {iteration}: recomputed sealed digest {recomputed} != "
            f"recorded {state.get('sealed_rollout_digest')}"
        )

    expected_games = bucket_counts()
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
            ("population_version", PHASE10B_POPULATION_VERSION),
            ("schedule_version", PHASE10B_SCHEDULE_VERSION),
            ("contract_digest", contract_digest()),
            ("namespace", PHASE10B_NAMESPACE),
            ("iteration", int(iteration)),
        ):
            if metadata[field_name] != expected:
                version_problems.append(
                    f"{game_id}: {field_name} {metadata[field_name]!r} != {expected!r}"
                )

    problems: list = []
    if observed != expected_games:
        problems.append(f"bucket counts {observed} != the frozen {expected_games}")
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
        raise Phase10BTrainerError(
            f"iteration {iteration} failed ownership verification: {'; '.join(problems)}"
        )

    behavior_id = sorted(behavior_ids)[0]
    behavior_digest = sorted(behavior_digests)[0]
    on_policy = None
    if behavior_snapshot is not None:
        if behavior_snapshot.checkpoint_sha256 != behavior_digest:
            raise Phase10BTrainerError(
                f"iteration {iteration} was collected under {behavior_digest}, but "
                f"the supplied snapshot is {behavior_snapshot.checkpoint_sha256}"
            )
        if behavior_snapshot.logical_identity != behavior_id:
            raise Phase10BTrainerError(
                f"the supplied snapshot is {behavior_snapshot.logical_identity!r}, "
                f"the iteration was collected by {behavior_id!r}"
            )
        behavior_snapshot.assert_frozen()
        on_policy = behavior_snapshot.loaded_state_dict_digest
    if expected_model_state_digest is not None:
        if on_policy is None:
            raise Phase10BTrainerError(
                "an on-policy check needs the behavior snapshot whose weights "
                "collected the iteration"
            )
        if on_policy != expected_model_state_digest:
            raise Phase10BTrainerError(
                f"iteration {iteration} was collected by weights {on_policy}, but "
                f"the trainer holds {expected_model_state_digest}; PPO may not "
                "consume another policy's rollout as if it were its own"
            )

    advantages_by_key, sequences_by_game, target_problems = collect_iteration_advantages(
        reader
    )
    if target_problems:
        raise Phase10BTrainerError(
            f"iteration {iteration}: target construction reported "
            f"{len(target_problems)} problem(s): {target_problems[:3]}"
        )
    statistics = iteration_statistics(
        advantages_by_key,
        namespace=PHASE10B_NAMESPACE,
        iteration=iteration,
        sealed_rollout_digest=recomputed,
        games=len(reader.game_ids),
    )
    keys = train_order_keys(reader, sequences_by_game)
    if len(keys) != len(advantages_by_key):
        raise Phase10BTrainerError(
            f"iteration {iteration}: {len(keys)} train-order keys but "
            f"{len(advantages_by_key)} advantages"
        )
    declared = sum(
        int(reader.metadata[game_id]["learner_decision_count"])
        for game_id in reader.game_ids
    )
    if declared != len(keys):
        raise Phase10BTrainerError(
            f"iteration {iteration}: metadata declares {declared} learner decisions, "
            f"the rebuilt sequences hold {len(keys)}"
        )
    scheduled = set(iteration_game_ids(iteration))
    if set(reader.game_ids) != scheduled:
        raise Phase10BTrainerError(
            f"iteration {iteration}: the committed game set is not the scheduled one"
        )
    return SealedRollout(
        root=root,
        iteration=int(iteration),
        sealed_rollout_digest=recomputed,
        behavior_snapshot_id=behavior_id,
        behavior_checkpoint_sha256=behavior_digest,
        games=len(reader.game_ids),
        keys=keys,
        statistics=statistics,
        reader=reader,
        verifications={
            "state": state["state"],
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


class Phase10BPipeline(_MinibatchPipeline):
    """The accepted multiprocess loader on the Phase 10B train order.

    The parent computes the plan in exactly one place; overriding it is the
    whole of the Phase 10B difference, so parallelism still cannot influence
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


class Phase10BTrainer:
    """One bounded Phase 10B optimization run over sealed on-policy rollouts."""

    def __init__(
        self,
        model,
        *,
        device: str = "mps",
        topology: "LoaderTopology | None" = None,
        initial_kl_beta: float = PHASE9_CANONICAL_INITIAL_KL_BETA,
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
            lr=scheduled_learning_rate(1),
            betas=tuple(constraints["adam_betas"]),
            eps=float(constraints["adam_epsilon"]),
            weight_decay=float(constraints["weight_decay"]),
        )
        self.gradient_clip_norm = float(constraints["gradient_clip_norm"])
        self.controller = KLController(beta=float(initial_kl_beta))
        self.global_step = 0
        self.examples_consumed = 0
        self.rl_iteration = 0
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
        }
        self.wall_clock = {
            "train_seconds": 0.0,
            "data_wait_seconds": 0.0,
            "checkpoint_seconds": 0.0,
        }
        self._pipeline: "Phase10BPipeline | None" = None
        self._bound: "SealedRollout | None" = None

    # -- construction ------------------------------------------------------

    def _verify_architecture(self) -> None:
        parameters = int(self.model.parameter_count())
        if parameters != ACCEPTED_PHASE9_PARAMETERS:
            raise Phase10BTrainerError(
                f"the model holds {parameters:,} parameters; Phase 10B fine-tunes "
                f"C1 with {ACCEPTED_PHASE9_PARAMETERS:,}"
            )
        summary = self.model.architecture_summary()
        digest = summary.get("config_digest") or summary.get("configuration_digest")
        if digest is not None and digest != ACCEPTED_C1_CONFIG_DIGEST:
            raise Phase10BTrainerError(
                f"C1 config digest {digest} != the frozen {ACCEPTED_C1_CONFIG_DIGEST}"
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
        iteration = max(1, min(self.rl_iteration, 30))
        return {
            "contract_version": PHASE10B_CONTRACT_VERSION,
            "trainer_version": PHASE10B_TRAINER_VERSION,
            "namespace": PHASE10B_NAMESPACE,
            "rl_iteration": int(self.rl_iteration),
            "learning_rate": scheduled_learning_rate(iteration),
            "entropy_coefficient": entropy_coefficient(iteration),
            "minibatch_size": MINIBATCH_SIZE,
            "epochs_per_rollout": EPOCHS_PER_ROLLOUT,
            "optimizer": dict(OPTIMIZER_CONSTRAINTS),
            "learning_rate_schedule": "linear decay across 30 iterations",
            "cursor": None if self.cursor is None else self.cursor.to_dict(),
        }

    # -- binding -----------------------------------------------------------

    def bind_iteration(
        self,
        rollout: SealedRollout,
        *,
        epochs: int = EPOCHS_PER_ROLLOUT,
    ) -> SealedRollout:
        """Bind one sealed rollout and set this iteration's frozen schedule."""
        if rollout.learner_decisions == 0:
            raise Phase10BTrainerError(
                f"iteration {rollout.iteration} holds no learner decisions"
            )
        self.close()
        self._bound = rollout
        self.rl_iteration = int(rollout.iteration)
        self.cursor = Cursor.start(
            iteration=rollout.iteration,
            sealed_rollout_digest=rollout.sealed_rollout_digest,
            total_examples=rollout.learner_decisions,
            epochs=int(epochs),
        )
        rate = scheduled_learning_rate(self.rl_iteration)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        write_iteration_state(
            rollout.root,
            PHASE10B_NAMESPACE,
            rollout.iteration,
            "TRAINING",
            sealed_rollout_digest=rollout.sealed_rollout_digest,
            behavior_snapshot_id=rollout.behavior_snapshot_id,
            behavior_checkpoint_sha256=rollout.behavior_checkpoint_sha256,
            learner_decisions=rollout.learner_decisions,
            learning_rate=rate,
            entropy_coefficient=entropy_coefficient(self.rl_iteration),
        )
        return rollout

    def mark_iteration_trained(self) -> dict:
        if self._bound is None or self.cursor is None:
            raise Phase10BTrainerError("no iteration is bound")
        if not self.cursor.finished:
            raise Phase10BTrainerError(
                f"iteration {self._bound.iteration} has not completed its epochs"
            )
        rollout = self._bound
        write_iteration_state(
            rollout.root,
            PHASE10B_NAMESPACE,
            rollout.iteration,
            "COMMITTED",
            sealed_rollout_digest=rollout.sealed_rollout_digest,
            behavior_snapshot_id=rollout.behavior_snapshot_id,
            behavior_checkpoint_sha256=rollout.behavior_checkpoint_sha256,
            training_complete=True,
            global_optimizer_step=self.global_step,
            examples_consumed=self.examples_consumed,
        )
        return {
            "iteration": rollout.iteration,
            "global_optimizer_step": self.global_step,
            "examples_consumed": self.examples_consumed,
        }

    def _ensure_pipeline(self) -> Phase10BPipeline:
        if self._pipeline is None:
            self._pipeline = Phase10BPipeline(
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

    def __enter__(self) -> "Phase10BTrainer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- the loop ----------------------------------------------------------

    def _sync(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":  # pragma: no cover - not this project
            torch.cuda.synchronize()

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
            raise Phase10BTrainerError(
                f"{len(outside)} minibatch key(s) are outside the sealed "
                f"iteration's learner universe (e.g. {outside[:3]})"
            )
        digests = set(packed["behavior_checkpoint_sha256"])
        if digests != {rollout.behavior_checkpoint_sha256}:
            self.counters["behavior_identity_mismatches"] += 1
            raise Phase10BTrainerError(
                f"minibatch carries behavior checkpoints {sorted(digests)}, the "
                f"iteration was collected under {rollout.behavior_checkpoint_sha256}"
            )
        identities = set(packed["rollout_ids"])
        if identities != {rollout.rollout_id}:
            self.counters["rollout_identity_mismatches"] += 1
            raise Phase10BTrainerError(
                f"minibatch carries rollout identities {sorted(identities)}, the "
                f"bound iteration is {rollout.rollout_id}"
            )

    def _check_hard_limits(self, *, iteration, epoch, mean_kl, clip_fraction) -> None:
        if mean_kl > KL_HARD_LIMIT:
            self.counters["kl_hard_limit_breaches"] += 1
            raise Phase10BTrainerError(
                f"iteration {iteration} epoch {epoch}: mean behavior KL "
                f"{mean_kl:.6f} exceeds the frozen hard limit {KL_HARD_LIMIT}"
            )
        if clip_fraction > CLIP_FRACTION_HARD_LIMIT:
            self.counters["clip_fraction_hard_limit_breaches"] += 1
            raise Phase10BTrainerError(
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
    ) -> list:
        """Run the bound iteration's epochs; return one metric row per update."""
        if self._bound is None:
            raise Phase10BTrainerError("no iteration is bound")
        pipeline = self._ensure_pipeline()
        rollout = self._bound
        coefficient = float(entropy_coefficient(max(1, min(self.rl_iteration, 30))))
        rows: list = []
        performed = 0
        while not self.cursor.finished:
            if updates is not None and performed >= int(updates):
                break
            step_started = time.perf_counter()
            keys, packed, waited = pipeline.next(self.cursor)
            arrays = unpack_batch(packed)
            self._verify_batch(keys, packed, rollout)
            row = {
                "namespace": PHASE10B_NAMESPACE,
                "iteration": rollout.iteration,
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
                raise Phase10BTrainerError(
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
                raise Phase10BTrainerError(
                    f"non-finite gradient norm at global step {self.global_step + 1}"
                )
            self.optimizer.step()
            parameter_norm = self._parameter_norm()
            if not bool(torch.isfinite(parameter_norm)):
                self.counters["non_finite_parameters"] += 1
                raise Phase10BTrainerError(
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

    # -- checkpoints -------------------------------------------------------

    def save(
        self,
        path,
        *,
        snapshot_role: str = "resume",
        rl_iteration: "int | None" = None,
        behavior_snapshot_identity: "str | None" = None,
        active_history_identities=(),
        history_checkpoint_digests: "dict | None" = None,
        diagnostics: "dict | None" = None,
    ) -> dict:
        from . import phase10b_checkpoint as checkpoints

        started = time.perf_counter()
        rollout = self._bound
        iteration = self.rl_iteration if rl_iteration is None else int(rl_iteration)
        payload = checkpoints.build_payload(
            model=self.model,
            optimizer=self.optimizer,
            snapshot_role=snapshot_role,
            rl_iteration=iteration,
            global_optimizer_step=self.global_step,
            examples_consumed=self.examples_consumed,
            behavior_snapshot_identity=(
                behavior_snapshot_identity
                if behavior_snapshot_identity is not None
                else ("" if rollout is None else rollout.behavior_snapshot_id)
            ),
            behavior_checkpoint_sha256=(
                "" if rollout is None else rollout.behavior_checkpoint_sha256
            ),
            sealed_rollout_digest=(
                "" if rollout is None else rollout.sealed_rollout_digest
            ),
            kl_beta=float(self.controller.beta),
            kl_controller_state=self.controller.to_dict(),
            learning_rate=self.learning_rate,
            entropy_coefficient=entropy_coefficient(max(1, min(self.rl_iteration, 30))),
            active_history_identities=active_history_identities,
            history_checkpoint_digests=history_checkpoint_digests or {},
            schedule_identity=self.schedule_identity(),
            counters=self.counters,
            wall_clock=self.wall_clock,
            device=str(self.device),
            diagnostics=diagnostics,
        )
        try:
            written = checkpoints.save(payload, path)
        except checkpoints.Phase10BCheckpointError:
            self.counters["checkpoint_errors"] += 1
            raise
        self.wall_clock["checkpoint_seconds"] += time.perf_counter() - started
        return written

    def restore(self, payload: dict) -> None:
        """Restore optimizer, controller and counters from a saved payload."""
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.controller = KLController.from_dict(payload["kl_controller_state"])
        self.global_step = int(payload["global_optimizer_step"])
        self.examples_consumed = int(payload["examples_consumed"])
        self.rl_iteration = int(payload["rl_iteration"])
        self.counters.update(payload.get("counters", {}))
        self.wall_clock.update(payload.get("wall_clock", {}))


def load_from_phase9(checkpoint_path, *, device: str = "mps", **kwargs) -> Phase10BTrainer:
    """Start a Phase 10B run from the accepted Phase 9 checkpoint.

    Fresh optimizer, fresh KL-controller state and fresh schedule position:
    Phase 10B adapts the accepted weights, it does not continue Phase 9's run.
    """
    from .phase10b_checkpoint import assert_phase9_untouched
    from .phase9_checkpoint import model_from_payload, read_phase9_payload

    path = Path(checkpoint_path)
    payload = read_phase9_payload(path)
    model = model_from_payload(payload, device=device)
    trainer = Phase10BTrainer(model, device=device, **kwargs)
    observed = trainer.model_state_digest
    from .phase10b_contract import ACCEPTED_PHASE9_STATE_DIGEST

    if observed != ACCEPTED_PHASE9_STATE_DIGEST:
        raise Phase10BTrainerError(
            f"the initialization's model-state digest {observed} != the accepted "
            f"Phase 9 {ACCEPTED_PHASE9_STATE_DIGEST}"
        )
    assert_phase9_untouched(path.resolve().parent.parent.parent)
    return trainer


def trainer_semantics() -> dict:
    return {
        "trainer_version": PHASE10B_TRAINER_VERSION,
        "contract_version": PHASE10B_CONTRACT_VERSION,
        "objective": "stratego.training.phase9_loss.phase9_batch_loss, unchanged",
        "controller": "stratego.training.phase9_trainer.KLController, unchanged",
        "targets": "stratego.training.phase9_targets, unchanged",
        "loader": "the accepted Phase 9 multiprocess pipeline, Phase 10B train order",
        "phase10b_own": [
            "linear learning-rate decay across 30 iterations",
            "0.0010 -> 0.0005 entropy schedule",
            "training_order stream from the Phase 10B roots",
        ],
        "hard_vetoes": {
            "kl_hard_limit": KL_HARD_LIMIT,
            "clip_fraction_hard_limit": CLIP_FRACTION_HARD_LIMIT,
            "abort_semantics": (
                "a breached limit raises at the epoch boundary, the affected "
                "update does not land, and the Gate F counter is incremented"
            ),
        },
        "replay": "none",
        "search": "never",
    }


__all__ = [
    "Cursor",
    "Phase10BPipeline",
    "Phase10BTrainer",
    "Phase10BTrainerError",
    "SealedRollout",
    "bind_sealed_rollout",
    "epoch_order",
    "load_from_phase9",
    "minibatch_keys",
    "rollout_identity",
    "trainer_semantics",
]
