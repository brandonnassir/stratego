"""Phase 9 Agent 5: `phase9_trainer_v1` — the MPS optimizer path.

Specification sources:

- `05_AGENT_5_PPO_TRAINER_AND_RESUME.md` ("Trainer", "Iteration ownership",
  "Checkpoint/resume", "Stability soak", "Throughput")
- `00_PHASE_9_SEQUENCE_AND_COMMON_CONTRACT.md` ("Full loss and common
  optimizer constraints", "Rollout state machine", "Learner-control
  semantics")
- Agent 1's `phase9_contract` (every learning constant), Agent 2's
  `phase9_schedule` (population and identity tokens), Agent 3's
  `phase9_rollout_store` (the sealed rollout), Agent 4's `phase9_targets`
  (examples, train order, cursor).

What this module owns
---------------------
Optimization, and only optimization. It selects nothing: the learning rate and
the initial KL beta arrive from Agent 1's frozen six-candidate matrix and
:class:`Phase9TrainConfig` refuses any other pair outside an explicitly
labelled non-selection scope. It redesigns nothing: targets, eligibility,
advantages and train order are read from Agent 4, and the rollout is read from
Agent 3.

Iteration ownership
-------------------
Only a `SEALED` rollout whose recomputed digest matches its own state record
may be consumed, and :func:`bind_sealed_rollout` re-derives that digest from
the committed bytes rather than believing the collector's bookkeeping. Six
verifications gate every iteration:

```text
sealed state + recomputed digest       are these the bytes that were sealed?
one behavior identity, matching state  was one snapshot responsible?
on-policy binding                      are these my weights?
population + schedule + contract       is this the league I am training in?
learner-control semantics              whose decisions may carry gradient?
example/target/advantage versions      do I mean by "example" what Agent 4 did?
```

Nothing is mutated: after two epochs the iteration is marked trained and the
rollout bytes are untouched.

Execution topology vs logical order
-----------------------------------
`phase9_train_order_v1` fixes the minibatch sequence as a pure function of
`(namespace, iteration, epoch)` and the sealed rollout's sorted key list.
Worker count, prefetch depth and the record cache change only *when* an
example is materialized, never *which* examples a minibatch holds or in what
order — `workers=1` builds through the same packing functions in-process and
is the bit-identical reference path.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..model.contract import POLICY_LOGIT_COUNT
from .phase9_behavior import BehaviorSnapshot, state_dict_digest
from .phase9_checkpoint import (
    PHASE9_TRAINER_VERSION,
    Phase9CheckpointError,
    build_phase9_checkpoint_payload,
    load_phase9_checkpoint,
    model_from_payload,
    read_phase9_payload,
    save_phase9_checkpoint,
    validate_phase9_payload,
)
from .phase9_contract import (
    BEHAVIOR_KL_TARGET,
    CLIP_FRACTION_HARD_LIMIT,
    EPOCHS_PER_ROLLOUT,
    EXPECTED_C1_CONFIG_DIGEST,
    EXPECTED_C1_PARAMETERS,
    KL_BETA_MAX,
    KL_BETA_MIN,
    KL_HARD_LIMIT,
    MINIBATCH_SIZE,
    OPTIMIZER_CONSTRAINTS,
    PHASE9_POPULATION_VERSION,
    PHASE9_ROLLOUT_SCHEDULE_VERSION,
    PILOT_CANDIDATES,
    PILOT_ITERATIONS,
    adaptive_kl_beta,
    bucket_counts,
    contract_digest,
    entropy_coefficient,
    learner_control_for,
)
from .phase9_loss import Phase9LossError, model_frame_table, phase9_batch_loss
from .phase9_rollout_store import (
    Phase9RolloutReader,
    read_iteration_state,
    sealed_rollout_digest,
    write_iteration_state,
)
from .phase9_targets import (
    MINIBATCH_SIZE as TARGET_MINIBATCH_SIZE,
    Phase9MinibatchCursor,
    build_example,
    build_sequences,
    collect_iteration_advantages,
    iteration_statistics,
    minibatch_keys,
    rollout_identity,
    train_order_keys,
)
from .phase9_seed import parse_phase9_game_id
from .reconstruction import iter_reconstructed_decisions
from .warmstart_checkpoint import CorpusIdentity, load_model_for_evaluation

#: Non-selection scopes. Agent 6 owns candidate selection; everything Agent 5
#: runs has to say out loud that it is not selecting anything.
SCOPE_PILOT = "pilot_candidate"
SCOPE_SOAK = "infrastructure_soak"
SCOPE_UNIT_TEST = "unit_test"
SCOPES = (SCOPE_PILOT, SCOPE_SOAK, SCOPE_UNIT_TEST)

#: The neutral middle configuration the mission names for the infrastructure
#: soak: the middle learning rate with the lower initial beta. Chosen solely so
#: the soak exercises the machinery — it is never compared with anything and
#: its weights never leave Agent 5.
SOAK_CANDIDATE_ID = "P9-C"

#: The architecture Phase 9 trains. Frozen: `C1`, and a checkpoint that says
#: anything else is a stop condition rather than a conversion.
MODEL_CANDIDATE = "C1"

STEP_METRIC_COLUMNS = (
    "namespace",
    "iteration",
    "epoch",
    "minibatch_index",
    "global_optimizer_step",
    "examples",
    "examples_consumed",
    "loss_total",
    "loss_ppo",
    "loss_value",
    "loss_belief",
    "behavior_kl",
    "policy_entropy",
    "policy_entropy_normalized",
    "kl_beta",
    "entropy_coefficient",
    "ppo_examples",
    "clip_fraction",
    "ratio_mean",
    "ratio_min",
    "ratio_max",
    "advantage_retention",
    "grad_norm_pre_clip",
    "grad_norm_post_clip",
    "parameter_norm",
    "learning_rate",
    "data_wait_seconds",
    "host_to_device_seconds",
    "forward_seconds",
    "loss_seconds",
    "backward_seconds",
    "optimizer_seconds",
    "step_seconds",
    "batch_digest",
)


class Phase9TrainerError(RuntimeError):
    """The trainer refused to train. Never repaired, always raised."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase9TrainConfig:
    """One logical Phase 9 optimization run's complete configuration.

    Only two numbers are free — the learning rate and the initial KL beta —
    and both are frozen per candidate by Agent 1. Everything else is read from
    `OPTIMIZER_CONSTRAINTS` at construction and re-checked, so a constant that
    drifted in the contract fails here rather than silently retraining under
    new physics.
    """

    namespace: str
    candidate_id: str
    scope: str
    learning_rate: float
    initial_kl_beta: float
    total_iterations: int
    device: str = "mps"
    model_candidate: str = MODEL_CANDIDATE
    minibatch_size: int = MINIBATCH_SIZE
    epochs_per_rollout: int = EPOCHS_PER_ROLLOUT
    weight_decay: float = float(OPTIMIZER_CONSTRAINTS["weight_decay"])
    adam_beta1: float = float(OPTIMIZER_CONSTRAINTS["adam_betas"][0])
    adam_beta2: float = float(OPTIMIZER_CONSTRAINTS["adam_betas"][1])
    adam_epsilon: float = float(OPTIMIZER_CONSTRAINTS["adam_epsilon"])
    gradient_clip_norm: float = float(OPTIMIZER_CONSTRAINTS["gradient_clip_norm"])
    precision: str = str(OPTIMIZER_CONSTRAINTS["precision"])

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise Phase9TrainerError(
                f"unknown scope {self.scope!r}; expected one of {list(SCOPES)}"
            )
        if self.precision != OPTIMIZER_CONSTRAINTS["precision"]:
            raise Phase9TrainerError(
                f"Phase 9 trains in {OPTIMIZER_CONSTRAINTS['precision']}, not "
                f"{self.precision!r}"
            )
        if self.model_candidate != MODEL_CANDIDATE:
            raise Phase9TrainerError(
                f"Phase 9 trains {MODEL_CANDIDATE}, not {self.model_candidate!r}"
            )
        if not KL_BETA_MIN <= self.initial_kl_beta <= KL_BETA_MAX:
            raise Phase9TrainerError(
                f"initial KL beta {self.initial_kl_beta} is outside the frozen "
                f"clamp [{KL_BETA_MIN}, {KL_BETA_MAX}]"
            )
        if self.learning_rate <= 0.0:
            raise Phase9TrainerError(
                f"learning rate must be positive, got {self.learning_rate}"
            )
        if self.total_iterations < 1:
            raise Phase9TrainerError(
                f"total_iterations must be >= 1, got {self.total_iterations}"
            )
        if self.scope != SCOPE_UNIT_TEST:
            for name, expected in (
                ("minibatch_size", OPTIMIZER_CONSTRAINTS["minibatch_size"]),
                ("epochs_per_rollout", OPTIMIZER_CONSTRAINTS["epochs_per_rollout"]),
                ("weight_decay", OPTIMIZER_CONSTRAINTS["weight_decay"]),
                ("gradient_clip_norm", OPTIMIZER_CONSTRAINTS["gradient_clip_norm"]),
            ):
                if getattr(self, name) != expected:
                    raise Phase9TrainerError(
                        f"{name} is frozen at {expected} for a Phase 9 run, got "
                        f"{getattr(self, name)}"
                    )
            frozen = {
                (candidate["learning_rate"], candidate["initial_kl_beta"])
                for candidate in PILOT_CANDIDATES
            }
            if (self.learning_rate, self.initial_kl_beta) not in frozen:
                raise Phase9TrainerError(
                    f"(learning_rate={self.learning_rate}, "
                    f"initial_kl_beta={self.initial_kl_beta}) is not one of the "
                    "six frozen candidates; only Agent 6 selects these, and only "
                    "from that matrix"
                )

    @property
    def selects_a_configuration(self) -> bool:
        """True only for a real pilot candidate run, which Agent 5 never does."""
        return self.scope == SCOPE_PILOT

    def identity(self) -> dict:
        """Everything a resume must find unchanged. No paths, no topology."""
        return {
            "trainer_version": PHASE9_TRAINER_VERSION,
            "namespace": self.namespace,
            "candidate_id": self.candidate_id,
            "scope": self.scope,
            "learning_rate": float(self.learning_rate),
            "initial_kl_beta": float(self.initial_kl_beta),
            "total_iterations": int(self.total_iterations),
            "device": self.device,
            "model_candidate": self.model_candidate,
            "minibatch_size": int(self.minibatch_size),
            "epochs_per_rollout": int(self.epochs_per_rollout),
            "weight_decay": float(self.weight_decay),
            "adam_beta1": float(self.adam_beta1),
            "adam_beta2": float(self.adam_beta2),
            "adam_epsilon": float(self.adam_epsilon),
            "gradient_clip_norm": float(self.gradient_clip_norm),
            "precision": self.precision,
            "optimizer": OPTIMIZER_CONSTRAINTS["optimizer"],
            "learning_rate_schedule": OPTIMIZER_CONSTRAINTS["learning_rate_schedule"],
            "contract_digest": contract_digest(),
        }

    def digest(self) -> str:
        parts = [f"{key}={value!r}" for key, value in sorted(self.identity().items())]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    @classmethod
    def for_candidate(
        cls,
        candidate_id: str,
        *,
        namespace: "str | None" = None,
        device: str = "mps",
        total_iterations: int = PILOT_ITERATIONS,
        scope: str = SCOPE_PILOT,
    ) -> "Phase9TrainConfig":
        """The frozen configuration of one of the six candidates."""
        index = {entry["candidate_id"]: entry for entry in PILOT_CANDIDATES}
        if candidate_id not in index:
            raise Phase9TrainerError(
                f"unknown pilot candidate {candidate_id!r}; the frozen matrix is "
                f"{sorted(index)}"
            )
        candidate = index[candidate_id]
        return cls(
            namespace=namespace or candidate["namespace"],
            candidate_id=candidate_id,
            scope=scope,
            learning_rate=float(candidate["learning_rate"]),
            initial_kl_beta=float(candidate["initial_kl_beta"]),
            total_iterations=int(total_iterations),
            device=device,
        )

    @classmethod
    def for_soak(
        cls,
        *,
        namespace: str,
        device: str = "mps",
        total_iterations: int,
    ) -> "Phase9TrainConfig":
        """The neutral middle configuration, labelled as non-selection.

        The mission asks for "a neutral middle pilot configuration chosen
        solely for infrastructure". `SCOPE_SOAK` is how that intent is carried
        in the artifacts: the run is a candidate's *numbers*, never a
        candidate's *result*.
        """
        config = cls.for_candidate(
            SOAK_CANDIDATE_ID,
            namespace=namespace,
            device=device,
            total_iterations=total_iterations,
            scope=SCOPE_SOAK,
        )
        return config

    @classmethod
    def for_unit_test(
        cls,
        *,
        namespace: str = "pilot_p9a",
        learning_rate: float = 1e-4,
        initial_kl_beta: float = 0.005,
        total_iterations: int = 1,
        device: str = "cpu",
        minibatch_size: int = 8,
        epochs_per_rollout: int = EPOCHS_PER_ROLLOUT,
    ) -> "Phase9TrainConfig":
        """A tiny configuration for tests. Never a run of anything."""
        return cls(
            namespace=namespace,
            candidate_id="unit-test",
            scope=SCOPE_UNIT_TEST,
            learning_rate=float(learning_rate),
            initial_kl_beta=float(initial_kl_beta),
            total_iterations=int(total_iterations),
            device=device,
            minibatch_size=int(minibatch_size),
            epochs_per_rollout=int(epochs_per_rollout),
        )


@dataclass(frozen=True)
class LoaderTopology:
    """Loader infrastructure. Tunable without touching any batch's identity."""

    workers: int = 6
    prefetch: int = 2
    record_cache_size: int = 64

    def __post_init__(self) -> None:
        if self.workers < 1 or self.prefetch < 1 or self.record_cache_size < 1:
            raise Phase9TrainerError("topology values must be >= 1")

    def to_dict(self) -> dict:
        return {
            "workers": int(self.workers),
            "prefetch": int(self.prefetch),
            "record_cache_size": int(self.record_cache_size),
        }


# ---------------------------------------------------------------------------
# The sealed rollout, verified
# ---------------------------------------------------------------------------


@dataclass
class SealedRollout:
    """One verified, trainable iteration. Built only by `bind_sealed_rollout`."""

    root: Path
    namespace: str
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
        # The membership test every minibatch runs, computed once: a key that
        # is not in the sealed iteration's learner universe is the shape an
        # opponent decision would take if one ever reached the optimizer.
        object.__setattr__(self, "keys_set", frozenset(self.keys))

    @property
    def rollout_id(self) -> str:
        return rollout_identity(self.namespace, self.iteration)

    @property
    def learner_decisions(self) -> int:
        return len(self.keys)

    def keys_digest(self) -> str:
        hasher = hashlib.sha256()
        for game_id, ply in self.keys:
            hasher.update(f"{game_id}|{ply}\n".encode())
        return hasher.hexdigest()

    def to_dict(self) -> dict:
        return {
            "namespace": self.namespace,
            "iteration": int(self.iteration),
            "rollout_id": self.rollout_id,
            "sealed_rollout_digest": self.sealed_rollout_digest,
            "behavior_snapshot_id": self.behavior_snapshot_id,
            "behavior_checkpoint_sha256": self.behavior_checkpoint_sha256,
            "games": int(self.games),
            "learner_decisions": int(self.learner_decisions),
            "train_order_keys_digest": self.keys_digest(),
            "statistics": self.statistics.to_dict(),
            "verifications": dict(self.verifications),
        }


def bind_sealed_rollout(
    root,
    namespace: str,
    iteration: int,
    *,
    behavior_snapshot: "BehaviorSnapshot | None" = None,
    expected_model_state_digest: "str | None" = None,
    require_full_schedule: bool = True,
    resuming: bool = False,
) -> SealedRollout:
    """Verify one iteration completely, then hand back a trainable view.

    The digest is recomputed from the committed journal rather than read from
    the manifest, so a state document that claims a rollout it does not hold
    cannot authorize training. `expected_model_state_digest` is the on-policy
    binding: the weights about to be optimized must be the weights that
    collected these games.

    `resuming=True` additionally accepts a `TRAINING` iteration, which is the
    state a crashed run left behind: the transition to `TRAINING` is durable
    precisely so a resume can find it, and refusing it would make the state
    machine unresumable at exactly the point it exists to survive.
    """
    root = Path(root)
    state = read_iteration_state(root, namespace, iteration)
    if state is None:
        raise Phase9TrainerError(
            f"{namespace} iteration {iteration} has no state document at {root}"
        )
    acceptable = ("SEALED", "TRAINING") if resuming else ("SEALED",)
    if state["state"] not in acceptable:
        raise Phase9TrainerError(
            f"{namespace} iteration {iteration} is {state['state']}, not one of "
            f"{list(acceptable)}; only a sealed rollout may be optimized"
        )
    if state["state"] == "TRAINING" and state.get("training_complete") and not resuming:
        raise Phase9TrainerError(
            f"{namespace} iteration {iteration} has already completed its epochs; "
            "a sealed rollout is trained once"
        )
    reader = Phase9RolloutReader(root, namespace, iteration)
    recomputed = sealed_rollout_digest(reader.commits)
    recorded = state.get("sealed_rollout_digest")
    if recomputed != recorded:
        raise Phase9TrainerError(
            f"{namespace} iteration {iteration}: recomputed sealed digest "
            f"{recomputed} != recorded {recorded}; the committed bytes are not "
            "the bytes that were sealed"
        )

    expected_games = bucket_counts(namespace)
    observed_buckets: dict = {}
    behavior_ids: set = set()
    behavior_digests: set = set()
    control_problems: list = []
    version_problems: list = []
    for game_id in reader.game_ids:
        metadata = reader.metadata[game_id]
        # The game id, not the sidecar, is the authority on which scheduled
        # game this is: a metadata block that agreed with itself about a
        # bucket it was never scheduled for would otherwise verify cleanly.
        identity = parse_phase9_game_id(game_id)
        bucket = str(identity["bucket"])
        observed_buckets[bucket] = observed_buckets.get(bucket, 0) + 1
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
        ):
            if metadata[field_name] != expected:
                version_problems.append(
                    f"{game_id}: metadata {field_name} {metadata[field_name]!r} != "
                    f"the id's {expected!r}"
                )
        for field_name, expected in (
            ("population_version", PHASE9_POPULATION_VERSION),
            ("schedule_version", PHASE9_ROLLOUT_SCHEDULE_VERSION),
            ("contract_digest", contract_digest()),
            ("namespace", namespace),
            ("iteration", int(iteration)),
        ):
            if metadata[field_name] != expected:
                version_problems.append(
                    f"{game_id}: {field_name} {metadata[field_name]!r} != {expected!r}"
                )
    problems: list = []
    if require_full_schedule and observed_buckets != expected_games:
        problems.append(
            f"bucket counts {observed_buckets} != the frozen {expected_games}"
        )
    if len(behavior_ids) != 1 or len(behavior_digests) != 1:
        problems.append(
            f"iteration mixes behavior identities {sorted(behavior_ids)} / "
            f"digests {sorted(behavior_digests)}"
        )
    if behavior_digests and state.get("behavior_checkpoint_sha256") not in behavior_digests:
        problems.append(
            f"state names behavior checkpoint {state.get('behavior_checkpoint_sha256')!r}, "
            f"the games were collected under {sorted(behavior_digests)}"
        )
    problems.extend(control_problems[:5])
    problems.extend(version_problems[:5])
    if problems:
        raise Phase9TrainerError(
            f"{namespace} iteration {iteration} failed iteration-ownership "
            f"verification: {'; '.join(problems)}"
        )

    behavior_id = sorted(behavior_ids)[0]
    behavior_digest = sorted(behavior_digests)[0]
    on_policy = None
    if behavior_snapshot is not None:
        if behavior_snapshot.checkpoint_sha256 != behavior_digest:
            raise Phase9TrainerError(
                f"{namespace} iteration {iteration} was collected under "
                f"{behavior_digest}, but the supplied snapshot is "
                f"{behavior_snapshot.checkpoint_sha256}"
            )
        if behavior_snapshot.logical_identity != behavior_id:
            raise Phase9TrainerError(
                f"the supplied snapshot is {behavior_snapshot.logical_identity!r}, "
                f"the iteration was collected by {behavior_id!r}"
            )
        behavior_snapshot.assert_frozen()
        on_policy = behavior_snapshot.loaded_state_dict_digest
    if expected_model_state_digest is not None:
        if on_policy is None:
            raise Phase9TrainerError(
                "an on-policy check needs the behavior snapshot whose weights "
                "collected the iteration"
            )
        if on_policy != expected_model_state_digest:
            raise Phase9TrainerError(
                f"{namespace} iteration {iteration} was collected by weights "
                f"{on_policy}, but the trainer holds {expected_model_state_digest}; "
                "PPO may not consume another policy's rollout as if it were its own"
            )

    advantages_by_key, sequences_by_game, target_problems = collect_iteration_advantages(
        reader
    )
    if target_problems:
        raise Phase9TrainerError(
            f"{namespace} iteration {iteration}: Agent 4 target construction "
            f"reported {len(target_problems)} problem(s): {target_problems[:3]}"
        )
    statistics = iteration_statistics(
        advantages_by_key,
        namespace=namespace,
        iteration=iteration,
        sealed_rollout_digest=recomputed,
        games=len(reader.game_ids),
    )
    keys = train_order_keys(reader, sequences_by_game)
    if len(keys) != len(advantages_by_key):
        raise Phase9TrainerError(
            f"{namespace} iteration {iteration}: {len(keys)} train-order keys but "
            f"{len(advantages_by_key)} advantages"
        )
    declared = sum(
        int(reader.metadata[game_id]["learner_decision_count"])
        for game_id in reader.game_ids
    )
    if declared != len(keys):
        raise Phase9TrainerError(
            f"{namespace} iteration {iteration}: metadata declares {declared} "
            f"learner decisions, the rebuilt sequences hold {len(keys)}"
        )
    return SealedRollout(
        root=root,
        namespace=namespace,
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
            "bucket_counts": observed_buckets,
            "single_behavior_identity": True,
            "learner_control_mismatches": len(control_problems),
            "version_mismatches": len(version_problems),
            "on_policy_state_dict_digest": on_policy,
            "declared_learner_decisions": declared,
        },
    )


# ---------------------------------------------------------------------------
# Example packing (the one path both topologies use)
# ---------------------------------------------------------------------------


def examples_for_keys(record, metadata, statistics, plies, sequences=None) -> dict:
    """`{ply: Phase9RLExample}` for one game's requested learner plies."""
    if sequences is None:
        sequences = build_sequences(record, metadata)
    by_ply = {
        int(ply): sequence for sequence in sequences.values() for ply in sequence.plies
    }
    wanted = sorted({int(ply) for ply in plies})
    missing = [ply for ply in wanted if ply not in by_ply]
    if missing:
        raise Phase9TrainerError(
            f"{record.game_id}: plies {missing[:3]} are not learner decisions; a "
            "minibatch may only contain learner-controlled decisions"
        )
    built = {}
    for rebuilt in iter_reconstructed_decisions(
        record,
        wanted,
        dense_mask=True,
        include_public_knowledge=False,
        copy_state=False,
    ):
        ply = int(rebuilt.ply)
        built[ply] = build_example(
            record, metadata, rebuilt, by_ply[ply], statistics
        )
    return built


def pack_examples(examples) -> dict:
    """Compact arrays for one minibatch, in the order given.

    The legal set and the behavior distribution are packed ragged — one
    concatenated run per field plus offsets — rather than as two `[B, 10000]`
    planes. It is the same information: `build_example` has already proved the
    replayed dense mask and the stored legal list describe the same position,
    and :func:`unpack_batch` re-densifies both from the one authority. The
    reason is throughput, not taste: the dense pair is 25 MB a minibatch and
    the ragged pair is under 200 KB.
    """
    items = list(examples)
    if not items:
        raise Phase9TrainerError("cannot pack an empty minibatch")
    legal_runs = []
    behavior_runs = []
    lengths = []
    for example in items:
        legal_model = np.flatnonzero(example.legal_mask).astype(np.int64)
        stored = _stored_legal_in_model_frame(example)
        if not np.array_equal(legal_model, stored[0]):
            raise Phase9TrainerError(
                f"{example.game_id} ply {example.decision_index}: the replayed "
                "legal mask and the stored legal set describe different positions"
            )
        legal_runs.append(stored[0])
        behavior_runs.append(stored[1])
        lengths.append(int(stored[0].size))
    return {
        "observation": np.stack(
            [np.ascontiguousarray(item.observation, dtype=np.float32) for item in items]
        ),
        "legal_actions_model": np.concatenate(legal_runs),
        "behavior_probabilities": np.concatenate(behavior_runs),
        "legal_lengths": np.asarray(lengths, dtype=np.int64),
        "sampled_action_model": np.asarray(
            [item.sampled_action_model for item in items], dtype=np.int64
        ),
        "behavior_action_probability": np.asarray(
            [item.behavior_action_probability for item in items], dtype=np.float32
        ),
        "standardized_advantage": np.asarray(
            [item.standardized_advantage for item in items], dtype=np.float32
        ),
        "advantage": np.asarray([item.advantage for item in items], dtype=np.float32),
        "ppo_eligible": np.asarray([item.ppo_eligible for item in items], dtype=bool),
        "wdl_target": np.asarray([item.wdl_target for item in items], dtype=np.float32),
        "belief_target": np.stack([item.belief_target for item in items]),
        "belief_mask": np.stack([item.belief_mask for item in items]),
        "learner_side": np.asarray(
            [item.learner_side for item in items], dtype=np.int64
        ),
        "game_ids": tuple(item.game_id for item in items),
        "decision_indices": tuple(int(item.decision_index) for item in items),
        "behavior_checkpoint_sha256": tuple(
            item.behavior_checkpoint_sha256 for item in items
        ),
        "rollout_ids": tuple(item.rollout_id for item in items),
    }


def _stored_legal_in_model_frame(example) -> tuple:
    """`(model-frame legal actions ascending, aligned probabilities)`.

    The stored pair is absolute-frame ascending; the model frame reorders it
    for blue, so the probabilities are permuted with the actions rather than
    re-sorted independently.
    """
    table = model_frame_table(example.learner_side)
    actions = table[np.asarray(example.behavior_legal_actions, dtype=np.int64)]
    probabilities = np.asarray(example.behavior_legal_probabilities, dtype=np.float32)
    order = np.argsort(actions, kind="stable")
    return actions[order], probabilities[order]


def unpack_batch(packed: dict) -> dict:
    """Re-densify a packed minibatch into the arrays the loss consumes."""
    size = int(packed["observation"].shape[0])
    lengths = packed["legal_lengths"]
    rows = np.repeat(np.arange(size, dtype=np.int64), lengths)
    columns = packed["legal_actions_model"]
    legal_mask = np.zeros((size, POLICY_LOGIT_COUNT), dtype=bool)
    legal_mask[rows, columns] = True
    behavior = np.zeros((size, POLICY_LOGIT_COUNT), dtype=np.float32)
    behavior[rows, columns] = packed["behavior_probabilities"]
    return {
        "observation": packed["observation"],
        "legal_mask": legal_mask,
        "behavior_probabilities": behavior,
        "sampled_action_model": packed["sampled_action_model"],
        "behavior_action_probability": packed["behavior_action_probability"],
        "standardized_advantage": packed["standardized_advantage"],
        "advantage": packed["advantage"],
        "ppo_eligible": packed["ppo_eligible"],
        "wdl_target": packed["wdl_target"],
        "belief_target": packed["belief_target"],
        "belief_mask": packed["belief_mask"],
        "learner_side": packed["learner_side"],
    }


def batch_digest(packed: dict) -> str:
    """A digest over one minibatch's identity and its numerical content.

    Two runs that produce equal digests at every step consumed the same
    examples in the same order with the same bytes — the logical equality a
    backend-aware resume proof rests on.
    """
    hasher = hashlib.sha256()
    for game_id, ply in zip(packed["game_ids"], packed["decision_indices"]):
        hasher.update(f"{game_id}|{ply}\n".encode())
    for name in (
        "observation",
        "legal_actions_model",
        "behavior_probabilities",
        "legal_lengths",
        "sampled_action_model",
        "behavior_action_probability",
        "standardized_advantage",
        "ppo_eligible",
        "wdl_target",
        "belief_target",
        "belief_mask",
        "learner_side",
    ):
        array = np.ascontiguousarray(packed[name])
        hasher.update(f"{name}|{array.shape}|{array.dtype}".encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _loader_init(options: dict) -> None:
    torch.set_num_threads(1)
    _WORKER["reader"] = Phase9RolloutReader(
        options["root"], options["namespace"], options["iteration"]
    )
    _WORKER["statistics"] = options["statistics"]
    _WORKER["cache"] = OrderedDict()
    _WORKER["cache_size"] = int(options["record_cache_size"])


def _worker_game(game_id: str):
    cache = _WORKER["cache"]
    if game_id in cache:
        cache.move_to_end(game_id)
        return cache[game_id]
    record, metadata = _WORKER["reader"].read_game(game_id)
    entry = (record, metadata, build_sequences(record, metadata))
    cache[game_id] = entry
    while len(cache) > _WORKER["cache_size"]:
        cache.popitem(last=False)
    return entry


def build_minibatch(keys, reader, statistics, game_source=None) -> dict:
    """Materialize one minibatch's examples and pack them, in key order.

    Games are visited in sorted order (so one replay serves every requested
    ply of a game) and the result is re-emitted in the *requested* order,
    because the shuffled order is what the frozen train order specifies and
    float summation is not commutative.
    """
    wanted: dict = {}
    for game_id, ply in keys:
        wanted.setdefault(game_id, []).append(int(ply))
    built: dict = {}
    for game_id in sorted(wanted):
        if game_source is None:
            record, metadata = reader.read_game(game_id)
            sequences = None
        else:
            record, metadata, sequences = game_source(game_id)
        for ply, example in examples_for_keys(
            record, metadata, statistics, wanted[game_id], sequences
        ).items():
            built[(game_id, ply)] = example
    return pack_examples(built[key] for key in keys)


def _loader_task(payload):
    index, keys = payload
    packed = build_minibatch(
        keys, _WORKER["reader"], _WORKER["statistics"], game_source=_worker_game
    )
    return index, packed


class _MinibatchPipeline:
    """Frozen minibatch plans in, identical packed batches out, strictly FIFO.

    The plan is computed here and only here, from the cursor; workers receive
    `(index, keys)` and nothing else, so parallelism can change arrival times
    and nothing else. `workers=1` builds in-process through the same
    :func:`build_minibatch`, which is the bit-identical reference path.
    """

    def __init__(self, rollout: SealedRollout, cursor, *, topology: LoaderTopology, epochs: int):
        self.rollout = rollout
        self.topology = topology
        self.epochs = int(epochs)
        self._cursor = cursor
        self._pending: deque = deque()
        self._pool = None
        self._cache: OrderedDict = OrderedDict()
        if topology.workers > 1:
            self._pool = ProcessPoolExecutor(
                max_workers=topology.workers,
                initializer=_loader_init,
                initargs=(
                    {
                        "root": str(rollout.root),
                        "namespace": rollout.namespace,
                        "iteration": rollout.iteration,
                        "statistics": rollout.statistics,
                        "record_cache_size": topology.record_cache_size,
                    },
                ),
            )
            self._fill()

    def _plan(self, cursor):
        return minibatch_keys(
            self.rollout.keys,
            self.rollout.namespace,
            self.rollout.iteration,
            cursor.epoch,
            cursor.minibatch_index,
            cursor.minibatch_size,
        )

    def _local_game(self, game_id: str):
        if game_id in self._cache:
            self._cache.move_to_end(game_id)
            return self._cache[game_id]
        record, metadata = self.rollout.reader.read_game(game_id)
        entry = (record, metadata, build_sequences(record, metadata))
        self._cache[game_id] = entry
        while len(self._cache) > self.topology.record_cache_size:
            self._cache.popitem(last=False)
        return entry

    def _fill(self) -> None:
        target = self.topology.workers * self.topology.prefetch
        while len(self._pending) < target and not self._cursor.finished:
            keys = self._plan(self._cursor)
            future = self._pool.submit(_loader_task, (len(self._pending), keys))
            self._pending.append((future, keys))
            self._cursor = self._cursor.advance(len(keys))

    def next(self, cursor) -> tuple:
        """`(keys, packed, wait_seconds)` for the minibatch `cursor` names."""
        if self._pool is None:
            keys = self._plan(cursor)
            started = time.perf_counter()
            packed = build_minibatch(
                keys,
                self.rollout.reader,
                self.rollout.statistics,
                game_source=self._local_game,
            )
            return keys, packed, time.perf_counter() - started
        future, keys = self._pending.popleft()
        expected = self._plan(cursor)
        if tuple(keys) != tuple(expected):
            raise Phase9TrainerError(
                "the pipeline produced a minibatch the cursor did not ask for; "
                "the prefetch plan and the training cursor have diverged"
            )
        started = time.perf_counter()
        _index, packed = future.result()
        waited = time.perf_counter() - started
        self._fill()
        return keys, packed, waited

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None
        self._pending.clear()
        self._cache.clear()


# ---------------------------------------------------------------------------
# The KL controller
# ---------------------------------------------------------------------------


@dataclass
class KLController:
    """The frozen adaptive-beta damping controller, with its own history.

    Beta is updated once after each optimizer epoch — never per minibatch —
    and the whole history rides in the checkpoint, so a resumed run damps from
    the same state an uninterrupted one would.

    The *partial* epoch also rides here, and that is the point: an epoch's mean
    KL is an example-weighted average over every minibatch of that epoch, so a
    run checkpointed halfway through one must carry the accumulated half. Left
    in a local variable it would silently reset on resume, and the resumed run
    would damp on the post-resume half alone — a divergence no parameter
    comparison at the resume boundary could ever reveal, because it only
    appears at the *next* epoch boundary.
    """

    beta: float
    target: float = BEHAVIOR_KL_TARGET
    history: list = field(default_factory=list)
    epoch_kl_sum: float = 0.0
    epoch_examples: int = 0
    epoch_clipped: int = 0
    epoch_ppo_examples: int = 0

    def observe(self, *, mean_kl: float, examples: int, clipped: int, ppo_examples: int) -> None:
        """Fold one minibatch into the epoch that is still being measured."""
        self.epoch_kl_sum += float(mean_kl) * int(examples)
        self.epoch_examples += int(examples)
        self.epoch_clipped += int(clipped)
        self.epoch_ppo_examples += int(ppo_examples)

    @property
    def epoch_mean_kl(self) -> float:
        return self.epoch_kl_sum / self.epoch_examples if self.epoch_examples else 0.0

    @property
    def epoch_clip_fraction(self) -> float:
        return (
            self.epoch_clipped / self.epoch_ppo_examples if self.epoch_ppo_examples else 0.0
        )

    def reset_epoch(self) -> None:
        self.epoch_kl_sum = 0.0
        self.epoch_examples = 0
        self.epoch_clipped = 0
        self.epoch_ppo_examples = 0

    def update(self, *, iteration: int, epoch: int) -> dict:
        """Close one epoch: apply the frozen rule, record it, start the next."""
        mean_epoch_kl = self.epoch_mean_kl
        clip_fraction = self.epoch_clip_fraction
        before = float(self.beta)
        after = adaptive_kl_beta(before, mean_epoch_kl)
        self.beta = after
        entry = {
            "iteration": int(iteration),
            "epoch": int(epoch),
            "mean_epoch_kl": mean_epoch_kl,
            "epoch_clip_fraction": clip_fraction,
            "epoch_examples": int(self.epoch_examples),
            "beta_before": before,
            "beta_after": after,
            "direction": (
                "increase" if after > before else "decrease" if after < before else "unchanged"
            ),
        }
        self.history.append(entry)
        self.reset_epoch()
        return entry

    def to_dict(self) -> dict:
        return {
            "beta": float(self.beta),
            "target": float(self.target),
            "updates": len(self.history),
            "history": [dict(entry) for entry in self.history],
            "clamp": [KL_BETA_MIN, KL_BETA_MAX],
            "epoch_kl_sum": float(self.epoch_kl_sum),
            "epoch_examples": int(self.epoch_examples),
            "epoch_clipped": int(self.epoch_clipped),
            "epoch_ppo_examples": int(self.epoch_ppo_examples),
        }

    @staticmethod
    def from_dict(payload: dict) -> "KLController":
        return KLController(
            beta=float(payload["beta"]),
            target=float(payload.get("target", BEHAVIOR_KL_TARGET)),
            history=[dict(entry) for entry in payload.get("history", [])],
            epoch_kl_sum=float(payload.get("epoch_kl_sum", 0.0)),
            epoch_examples=int(payload.get("epoch_examples", 0)),
            epoch_clipped=int(payload.get("epoch_clipped", 0)),
            epoch_ppo_examples=int(payload.get("epoch_ppo_examples", 0)),
        )


# ---------------------------------------------------------------------------
# The trainer
# ---------------------------------------------------------------------------


class Phase9Trainer:
    """One logical Phase 9 optimization run over sealed on-policy rollouts."""

    def __init__(
        self,
        config: Phase9TrainConfig,
        corpus_identity: CorpusIdentity,
        *,
        model,
        topology: "LoaderTopology | None" = None,
        run_label: str = "",
        fsync_checkpoints: bool = True,
        _restored: "dict | None" = None,
    ) -> None:
        if not isinstance(corpus_identity, CorpusIdentity):
            raise Phase9TrainerError(
                "the trainer requires a verified CorpusIdentity; measure it with "
                "warmstart_checkpoint.verify_corpus_identity before construction"
            )
        self.config = config
        self.corpus_identity = corpus_identity
        self.topology = topology or LoaderTopology()
        self.run_label = str(run_label)
        self.fsync_checkpoints = bool(fsync_checkpoints)
        self.device = torch.device(config.device)
        self.model = model.to(device=self.device, dtype=torch.float32)
        self.model.requires_grad_(True)
        self.model.train()
        self._verify_architecture()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
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
        self.validation_history: list = []
        self.best_validation_score = None
        self.best_checkpoint_identity = None
        self.active_historical_identities: tuple = ()
        self.historical_checkpoint_digests: dict = {}
        self._pipeline: "_MinibatchPipeline | None" = None
        self._bound: "SealedRollout | None" = None

        if _restored is None:
            self.global_step = 0
            self.examples_consumed = 0
            self.rl_iteration = 0
            self.controller = KLController(beta=float(config.initial_kl_beta))
            self.cursor = None
        else:
            self.optimizer.load_state_dict(_restored["optimizer_state"])
            self.scheduler.load_state_dict(_restored["scheduler_state"])
            self.global_step = int(_restored["global_optimizer_step"])
            self.examples_consumed = int(_restored["examples_consumed"])
            self.rl_iteration = int(_restored["rl_iteration"])
            self.controller = KLController.from_dict(_restored["kl_controller_state"])
            self.cursor = _cursor_from_dict(_restored["minibatch_cursor"])
            self.validation_history = list(_restored["validation_history"])
            self.best_validation_score = _restored["best_validation_score"]
            self.best_checkpoint_identity = _restored["best_checkpoint_identity"]
            self.wall_clock.update(_restored["wall_clock_counters"])

    # -- construction ---------------------------------------------------------

    def _verify_architecture(self) -> None:
        summary = self.model.architecture_summary()
        parameters = int(self.model.parameter_count())
        if parameters != EXPECTED_C1_PARAMETERS:
            raise Phase9TrainerError(
                f"the model holds {parameters:,} parameters; Phase 9 trains C1 "
                f"with {EXPECTED_C1_PARAMETERS:,}"
            )
        digest = summary.get("config_digest") or summary.get("configuration_digest")
        if digest is not None and digest != EXPECTED_C1_CONFIG_DIGEST:
            raise Phase9TrainerError(
                f"C1 config digest {digest} != the frozen {EXPECTED_C1_CONFIG_DIGEST}"
            )

    def _build_optimizer(self) -> torch.optim.AdamW:
        config = self.config
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """A constant-factor schedule.

        `OPTIMIZER_CONSTRAINTS` freezes "constant (no warmup, no decay)". The
        scheduler object exists anyway because `phase9_checkpoint_v1` requires
        scheduler state: a run that later needs a schedule inherits the same
        save/restore path instead of growing a new one.
        """
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _step: 1.0)

    @classmethod
    def from_phase8_checkpoint(
        cls,
        checkpoint_path: "str | Path",
        config: Phase9TrainConfig,
        corpus_identity: CorpusIdentity,
        **kwargs,
    ) -> "Phase9Trainer":
        """Start fresh from the accepted Phase 8 checkpoint."""
        model, _metadata = load_model_for_evaluation(checkpoint_path, device="cpu")
        return cls(config, corpus_identity, model=model, **kwargs)

    @classmethod
    def from_phase9_checkpoint(
        cls,
        checkpoint_path: "str | Path",
        config: Phase9TrainConfig,
        corpus_identity: CorpusIdentity,
        **kwargs,
    ) -> "Phase9Trainer":
        """Start a *new* run from a Phase 9 file's weights, carrying no state."""
        payload = read_phase9_payload(checkpoint_path)
        validate_phase9_payload(payload, source=str(checkpoint_path))
        return cls(
            config, corpus_identity, model=model_from_payload(payload), **kwargs
        )

    @classmethod
    def resume(
        cls,
        checkpoint_path: "str | Path",
        *,
        config: Phase9TrainConfig,
        corpus_identity: CorpusIdentity,
        topology: "LoaderTopology | None" = None,
        run_label: str = "",
        fsync_checkpoints: bool = True,
        **identity_expectations,
    ) -> "Phase9Trainer":
        """Continue the exact logical run a checkpoint froze.

        Every identity mismatch raises inside
        :func:`phase9_checkpoint.load_phase9_checkpoint` before any state is
        touched, so a rejected resume leaves this process unchanged.
        """
        restored = load_phase9_checkpoint(
            checkpoint_path,
            device=config.device,
            expected_train_config=config.identity(),
            expected_train_config_digest=config.digest(),
            expected_corpus_identity=corpus_identity,
            **identity_expectations,
        )
        return cls(
            config,
            corpus_identity,
            model=restored["model"],
            topology=topology,
            run_label=run_label,
            fsync_checkpoints=fsync_checkpoints,
            _restored=restored,
        )

    # -- state ----------------------------------------------------------------

    def model_state_digest(self) -> str:
        """The digest of the live weights, for the on-policy binding check."""
        return state_dict_digest(self.model)

    def parameter_snapshot(self) -> dict:
        return {
            name: parameter.detach().to("cpu", torch.float32).clone()
            for name, parameter in self.model.named_parameters()
        }

    def state_summary(self) -> dict:
        """Every logical quantity a resume must reproduce exactly."""
        return {
            "trainer_version": PHASE9_TRAINER_VERSION,
            "train_config_digest": self.config.digest(),
            "global_optimizer_step": int(self.global_step),
            "examples_consumed": int(self.examples_consumed),
            "rl_iteration": int(self.rl_iteration),
            "minibatch_cursor": self.cursor.to_dict() if self.cursor else None,
            "kl_beta": float(self.controller.beta),
            "kl_controller_updates": len(self.controller.history),
            "kl_controller_history": [dict(e) for e in self.controller.history],
            "kl_controller_partial_epoch": {
                "kl_sum": float(self.controller.epoch_kl_sum),
                "examples": int(self.controller.epoch_examples),
                "clipped": int(self.controller.epoch_clipped),
                "ppo_examples": int(self.controller.epoch_ppo_examples),
            },
            "entropy_schedule_position": self.entropy_position(),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "scheduler_last_epoch": int(self.scheduler.last_epoch),
            "optimizer_state_structure": _optimizer_structure(self.optimizer),
            "counters": dict(self.counters),
            "bound_rollout": None if self._bound is None else self._bound.rollout_id,
            "sealed_rollout_digest": (
                None if self._bound is None else self._bound.sealed_rollout_digest
            ),
        }

    def entropy_position(self) -> dict:
        iteration = max(1, int(self.rl_iteration))
        return {
            "iteration": int(self.rl_iteration),
            "total_iterations": int(self.config.total_iterations),
            "coefficient": float(
                entropy_coefficient(
                    min(iteration, self.config.total_iterations),
                    self.config.total_iterations,
                )
            ),
        }

    # -- iteration ownership ---------------------------------------------------

    def bind_iteration(
        self,
        rollout: SealedRollout,
        *,
        mark_training: bool = True,
    ) -> SealedRollout:
        """Take ownership of one sealed iteration and start its cursor."""
        if rollout.namespace != self.config.namespace:
            self.counters["rollout_identity_mismatches"] += 1
            raise Phase9TrainerError(
                f"rollout namespace {rollout.namespace!r} != the run's "
                f"{self.config.namespace!r}"
            )
        # A pipeline belongs to one iteration's cursor and rollout. The
        # previous iteration left its own exhausted, so binding a new one
        # must drop it rather than let a stale prefetch queue answer.
        self.close()
        self._bound = rollout
        self.rl_iteration = rollout.iteration
        self.cursor = Phase9MinibatchCursor.start(
            namespace=rollout.namespace,
            iteration=rollout.iteration,
            sealed_rollout_digest=rollout.sealed_rollout_digest,
            total_examples=rollout.learner_decisions,
            epochs=self.config.epochs_per_rollout,
            minibatch_size=self.config.minibatch_size,
        )
        if mark_training:
            write_iteration_state(
                rollout.root,
                rollout.namespace,
                rollout.iteration,
                "TRAINING",
                sealed_rollout_digest=rollout.sealed_rollout_digest,
                trainer_version=PHASE9_TRAINER_VERSION,
                train_config_digest=self.config.digest(),
                training_complete=False,
            )
        return rollout

    def rebind_iteration(self, rollout: SealedRollout) -> SealedRollout:
        """Re-attach a resumed run to the sealed rollout its cursor names."""
        if self.cursor is None:
            raise Phase9TrainerError("this trainer has no cursor to rebind")
        for label, mine, theirs in (
            ("namespace", self.cursor.namespace, rollout.namespace),
            ("iteration", self.cursor.iteration, rollout.iteration),
            (
                "sealed rollout digest",
                self.cursor.sealed_rollout_digest,
                rollout.sealed_rollout_digest,
            ),
            ("total examples", self.cursor.total_examples, rollout.learner_decisions),
        ):
            if mine != theirs:
                self.counters["rollout_identity_mismatches"] += 1
                raise Phase9TrainerError(
                    f"resumed cursor names {label} {mine!r}, the supplied rollout "
                    f"has {theirs!r}"
                )
        self.close()
        self._bound = rollout
        self.rl_iteration = rollout.iteration
        return rollout

    def mark_iteration_trained(self) -> dict:
        """Record that two epochs finished, without touching rollout bytes."""
        if self._bound is None:
            raise Phase9TrainerError("no iteration is bound")
        rollout = self._bound
        # Re-read the journals from disk rather than re-hashing the reader's
        # in-memory copy: the claim is that the bytes on disk did not move
        # while they were being trained on, and only a fresh read can say so.
        after = Phase9RolloutReader(rollout.root, rollout.namespace, rollout.iteration)
        if sealed_rollout_digest(after.commits) != rollout.sealed_rollout_digest:
            self.counters["rollout_identity_mismatches"] += 1
            raise Phase9TrainerError(
                "the sealed rollout digest changed during training; rollout bytes "
                "are immutable"
            )
        return write_iteration_state(
            rollout.root,
            rollout.namespace,
            rollout.iteration,
            "TRAINING",
            sealed_rollout_digest=rollout.sealed_rollout_digest,
            trainer_version=PHASE9_TRAINER_VERSION,
            train_config_digest=self.config.digest(),
            training_complete=True,
            epochs_completed=int(self.config.epochs_per_rollout),
            global_optimizer_step=int(self.global_step),
            examples_consumed=int(self.examples_consumed),
        )

    # -- the loop --------------------------------------------------------------

    def _ensure_pipeline(self) -> _MinibatchPipeline:
        if self._pipeline is None:
            if self._bound is None:
                raise Phase9TrainerError("no iteration is bound")
            self._pipeline = _MinibatchPipeline(
                self._bound,
                self.cursor,
                topology=self.topology,
                epochs=self.config.epochs_per_rollout,
            )
        return self._pipeline

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.shutdown()
            self._pipeline = None

    def __enter__(self) -> "Phase9Trainer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _sync(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()

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

    def train_iteration(
        self,
        *,
        updates: "int | None" = None,
        timing: bool = False,
        capture_batch_digests: bool = False,
        on_step=None,
        crash_hook=None,
    ) -> list:
        """Run the bound iteration's epochs; return one metric row per update.

        Stops early only when `updates` says so — which exists for the resume
        experiments and the soak, never as an optimization decision. The KL
        controller fires at each epoch boundary and the frozen hard limits are
        checked there, because they are epoch statistics.
        """
        if self._bound is None:
            raise Phase9TrainerError("no iteration is bound")
        pipeline = self._ensure_pipeline()
        rollout = self._bound
        coefficient = float(
            entropy_coefficient(
                min(max(1, self.rl_iteration), self.config.total_iterations),
                self.config.total_iterations,
            )
        )
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
                "namespace": rollout.namespace,
                "iteration": rollout.iteration,
                "epoch": int(self.cursor.epoch),
                "minibatch_index": int(self.cursor.minibatch_index),
                "data_wait_seconds": waited,
            }
            if capture_batch_digests:
                row["batch_digest"] = batch_digest(packed)

            marks = [time.perf_counter()]
            tensors = {
                name: torch.from_numpy(np.ascontiguousarray(value)).to(self.device)
                for name, value in arrays.items()
                if name != "learner_side"
            }
            if timing:
                self._sync()
            marks.append(time.perf_counter())

            outputs = self.model.forward_observation(tensors["observation"])
            if timing:
                self._sync()
            marks.append(time.perf_counter())

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
                raise Phase9TrainerError(
                    f"non-finite loss at global step {self.global_step + 1}"
                )
            if timing:
                self._sync()
            marks.append(time.perf_counter())

            self.optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            if timing:
                self._sync()
            marks.append(time.perf_counter())

            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            pre_clip = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_norm
            )
            post_clip = self._grad_norm()
            if not bool(torch.isfinite(pre_clip)) or not bool(torch.isfinite(post_clip)):
                self.counters["non_finite_gradients"] += 1
                raise Phase9TrainerError(
                    f"non-finite gradient norm at global step {self.global_step + 1}"
                )
            self.optimizer.step()
            self.scheduler.step()
            parameter_norm = self._parameter_norm()
            if not bool(torch.isfinite(parameter_norm)):
                self.counters["non_finite_parameters"] += 1
                raise Phase9TrainerError(
                    f"non-finite parameters at global step {self.global_step + 1}"
                )
            if timing:
                self._sync()
            marks.append(time.perf_counter())

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
                    "learning_rate": learning_rate,
                    "host_to_device_seconds": marks[1] - marks[0],
                    "forward_seconds": marks[2] - marks[1],
                    "loss_seconds": marks[3] - marks[2],
                    "backward_seconds": marks[4] - marks[3],
                    "optimizer_seconds": marks[5] - marks[4],
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
            if crash_hook is not None:
                crash_hook(self, row)
        return rows

    def _verify_batch(self, keys, packed, rollout: SealedRollout) -> None:
        """Refuse a minibatch that is not what the frozen order asked for.

        Three different mistakes, each of which would silently corrupt the run:
        a key outside the sealed iteration's learner universe (an opponent
        decision would enter here if it could), a behavior checkpoint that is
        not the iteration's, and a rollout identity that is not the bound one.
        """
        universe = rollout.keys_set
        outside = [key for key in keys if key not in universe]
        if outside:
            self.counters["data_mismatches"] += 1
            raise Phase9TrainerError(
                f"{len(outside)} minibatch key(s) are outside the sealed "
                f"iteration's learner universe (e.g. {outside[:3]})"
            )
        digests = set(packed["behavior_checkpoint_sha256"])
        if digests != {rollout.behavior_checkpoint_sha256}:
            self.counters["behavior_identity_mismatches"] += 1
            raise Phase9TrainerError(
                f"minibatch carries behavior checkpoints {sorted(digests)}, the "
                f"iteration was collected under {rollout.behavior_checkpoint_sha256}"
            )
        identities = set(packed["rollout_ids"])
        if identities != {rollout.rollout_id}:
            self.counters["rollout_identity_mismatches"] += 1
            raise Phase9TrainerError(
                f"minibatch carries rollout identities {sorted(identities)}, the "
                f"bound iteration is {rollout.rollout_id}"
            )

    def _check_hard_limits(
        self, *, iteration: int, epoch: int, mean_kl: float, clip_fraction: float
    ) -> None:
        if mean_kl > KL_HARD_LIMIT:
            self.counters["kl_hard_limit_breaches"] += 1
            raise Phase9TrainerError(
                f"iteration {iteration} epoch {epoch}: mean behavior KL {mean_kl:.6f} "
                f"exceeds the frozen hard limit {KL_HARD_LIMIT}"
            )
        if clip_fraction > CLIP_FRACTION_HARD_LIMIT:
            self.counters["clip_fraction_hard_limit_breaches"] += 1
            raise Phase9TrainerError(
                f"iteration {iteration} epoch {epoch}: PPO clip fraction "
                f"{clip_fraction:.6f} exceeds the frozen hard limit "
                f"{CLIP_FRACTION_HARD_LIMIT}"
            )

    # -- checkpointing ---------------------------------------------------------

    def checkpoint_payload(
        self,
        *,
        snapshot_role: str = "resume",
        diagnostics=None,
        behavior_snapshot_identity: "str | None" = None,
        rl_iteration: "int | None" = None,
    ) -> dict:
        """The current state as a `phase9_checkpoint_v1` payload.

        The two overrides exist for the snapshot roles. A behavior snapshot is
        frozen *after* one iteration's epochs and names the iteration it is
        about to collect, so its own identity and RL iteration are not the ones
        the trainer is currently sitting on — while its cursor, optimizer and
        controller state honestly record where the weights came from.
        """
        if self.cursor is None:
            raise Phase9TrainerError(
                "a Phase 9 checkpoint records a position in a sealed rollout; bind "
                "an iteration first"
            )
        rollout = self._bound
        extra = dict(diagnostics or {})
        extra.setdefault("device", str(self.device))
        extra.setdefault("run_label", self.run_label)
        extra.setdefault("topology", self.topology.to_dict())
        extra.setdefault("counters", dict(self.counters))
        extra.setdefault(
            "rollout_root", None if rollout is None else str(rollout.root)
        )
        return build_phase9_checkpoint_payload(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            train_config=self.config.identity(),
            train_config_digest=self.config.digest(),
            corpus_identity=self.corpus_identity,
            global_optimizer_step=self.global_step,
            rl_iteration=(
                self.rl_iteration if rl_iteration is None else int(rl_iteration)
            ),
            minibatch_cursor=self.cursor.to_dict(),
            examples_consumed=self.examples_consumed,
            behavior_snapshot_identity=(
                behavior_snapshot_identity
                if behavior_snapshot_identity is not None
                else ("" if rollout is None else rollout.behavior_snapshot_id)
            ),
            behavior_checkpoint_sha256=(
                "" if rollout is None else rollout.behavior_checkpoint_sha256
            ),
            rollout_iteration_identity=("" if rollout is None else rollout.rollout_id),
            sealed_rollout_digest=self.cursor.sealed_rollout_digest,
            kl_beta=float(self.controller.beta),
            kl_controller_state=self.controller.to_dict(),
            entropy_schedule_position=self.entropy_position(),
            active_historical_identities=self.active_historical_identities,
            historical_checkpoint_digests=self.historical_checkpoint_digests,
            best_validation_score=self.best_validation_score,
            best_checkpoint_identity=self.best_checkpoint_identity,
            validation_history=self.validation_history,
            wall_clock_counters=dict(self.wall_clock),
            diagnostics=extra,
            snapshot_role=snapshot_role,
        )

    def save_checkpoint(
        self,
        path: "str | Path",
        *,
        snapshot_role: str = "resume",
        diagnostics=None,
        crash_hook=None,
        **overrides,
    ) -> dict:
        started = time.perf_counter()
        try:
            written = save_phase9_checkpoint(
                self.checkpoint_payload(
                    snapshot_role=snapshot_role, diagnostics=diagnostics, **overrides
                ),
                path,
                fsync=self.fsync_checkpoints,
                crash_hook=crash_hook,
            )
        except Phase9CheckpointError:
            self.counters["checkpoint_errors"] += 1
            raise
        self.wall_clock["checkpoint_seconds"] += time.perf_counter() - started
        return written

    def save_behavior_snapshot(
        self, path: "str | Path", *, logical_identity: str, rl_iteration: int
    ) -> dict:
        """Freeze the current weights as the snapshot that collects `rl_iteration`.

        Written only after an iteration's epochs finish, which is the frozen
        order: "create the next behavior snapshot only after the iteration is
        committed". The file is a complete checkpoint, so the collector's
        binding check and the trainer's later on-policy check compare real
        recorded identities rather than trusting a stripped export.
        """
        return self.save_checkpoint(
            path,
            snapshot_role="behavior_snapshot",
            diagnostics={
                "produced_after_iteration": int(self.rl_iteration),
                "collects_iteration": int(rl_iteration),
            },
            behavior_snapshot_identity=str(logical_identity),
            rl_iteration=int(rl_iteration),
        )

    def archive_member_payload(self, *, local_identity: str) -> dict:
        """The payload for one immutable namespace-local archive member.

        The local identity is the frozen cadence's name for this slot
        (`archive_snapshot_id`); the namespace comes from the train config, and
        the two together are what makes `pilot_p9a|H005` a different object
        from `canonical|H005`.
        """
        return self.checkpoint_payload(
            snapshot_role="archive_member",
            diagnostics={
                "archived_after_iteration": int(self.rl_iteration),
                "local_identity": str(local_identity),
            },
        )


def _cursor_from_dict(payload: dict) -> Phase9MinibatchCursor:
    return Phase9MinibatchCursor(
        namespace=str(payload["namespace"]),
        iteration=int(payload["iteration"]),
        sealed_rollout_digest=str(payload["sealed_rollout_digest"]),
        epoch=int(payload["epoch"]),
        minibatch_index=int(payload["minibatch_index"]),
        examples_consumed=int(payload["examples_consumed"]),
        total_examples=int(payload["total_examples"]),
        minibatch_size=int(payload["minibatch_size"]),
        epochs=int(payload["epochs"]),
    )


def _optimizer_structure(optimizer) -> dict:
    """Shape-level description of the optimizer state, for resume comparison."""
    state = optimizer.state_dict()
    return {
        "param_groups": [
            {
                key: value
                for key, value in group.items()
                if key != "params"
            }
            for group in state["param_groups"]
        ],
        "state_entries": len(state["state"]),
        "step_values": sorted(
            {
                int(entry["step"]) if not isinstance(entry["step"], torch.Tensor)
                else int(entry["step"].item())
                for entry in state["state"].values()
                if "step" in entry
            }
        ),
    }


def trainer_semantics() -> dict:
    """The serializable statement of the trainer's frozen behavior."""
    return {
        "trainer_version": PHASE9_TRAINER_VERSION,
        "architecture": MODEL_CANDIDATE,
        "optimizer_constraints": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in OPTIMIZER_CONSTRAINTS.items()
        },
        "iteration_ownership": [
            "only a SEALED rollout may be consumed",
            "the sealed digest is recomputed from the committed journal",
            "one behavior identity per iteration, matching the state document",
            "the on-policy binding compares live weights with the collecting snapshot",
            "population, schedule and contract identities are checked per game",
            "learner-control semantics are re-derived per game",
            "Agent 4 example/advantage/train-order versions must match",
        ],
        "populations": {
            "policy": "ppo_eligible learner examples only",
            "value_belief_kl_entropy": "every learner example of the minibatch",
            "opponent": (
                "rule, stress and historical-opponent decisions are not members "
                "of the train-order universe, so they contribute exactly zero "
                "policy/value/belief gradient"
            ),
        },
        "kl_controller": {
            "cadence": "once after each optimizer epoch",
            "target": BEHAVIOR_KL_TARGET,
            "clamp": [KL_BETA_MIN, KL_BETA_MAX],
            "hard_limits": {
                "mean_epoch_kl": KL_HARD_LIMIT,
                "clip_fraction": CLIP_FRACTION_HARD_LIMIT,
            },
        },
        "train_order": {
            "universe": "the sealed iteration's learner decisions, sorted by (game_id, ply)",
            "shuffle": "train_order_seed(namespace, iteration, epoch)",
            "minibatch_size": TARGET_MINIBATCH_SIZE,
            "final_partial_minibatch": "consumed, never dropped",
        },
        "topology_rule": (
            "worker count, prefetch and record cache change only when an example "
            "is materialized; the minibatch plan is computed from the cursor "
            "alone and every batch is verified against it"
        ),
        "selection": (
            "Agent 5 selects nothing: the scope is recorded in the train config "
            "and only SCOPE_PILOT is a selection run"
        ),
    }


__all__ = [
    "MODEL_CANDIDATE",
    "SCOPES",
    "SCOPE_PILOT",
    "SCOPE_SOAK",
    "SCOPE_UNIT_TEST",
    "SOAK_CANDIDATE_ID",
    "STEP_METRIC_COLUMNS",
    "KLController",
    "LoaderTopology",
    "Phase9TrainConfig",
    "Phase9Trainer",
    "Phase9TrainerError",
    "SealedRollout",
    "batch_digest",
    "bind_sealed_rollout",
    "build_minibatch",
    "examples_for_keys",
    "pack_examples",
    "trainer_semantics",
    "unpack_batch",
]
