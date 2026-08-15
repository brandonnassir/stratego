"""Phase 8 Agent 4: `warmstart_trainer_v1` — the C1 float32 MPS trainer.

Specification sources:

- `04_AGENT_4_TRAINER_AND_RESUME.md` (mission, optimizer/scheduler,
  loader/trainer balance, resume equivalence)
- `00_PHASE_8_SEQUENCE_AND_COMMON_CONTRACT.md` sections 18-23
- Agent 1's frozen pilot matrix
  (:data:`stratego.training.warmstart_contract.PILOT_CANDIDATES`)

Only the predeclared candidates
-------------------------------
A production configuration exists exactly when it is one of Agent 1's six
frozen pilot candidates: :meth:`WarmstartTrainConfig.from_pilot_candidate` is
the only constructor that yields one, and direct construction with
off-matrix hyperparameters raises. The deliberately separate
``unit_test`` scope exists so the suite can train C0 for a handful of steps
on a mini corpus; its identifiers are forced to a ``unittest_`` prefix, so no
artifact can quietly present a test configuration as a pilot.

Where the data comes from
-------------------------
The corpus root comes from :func:`synthetic_corpus.default_corpus_root` — the
production resolver — unless a test passes its own mini-corpus root. The
trainer refuses to construct without a verified :class:`CorpusIdentity`
(digests, not a path), and every checkpoint it writes carries that identity.

What one update is
------------------
```text
fetch batch  (frozen plan: pure function of the data cursor)
h2d          observation + loss inputs to the device
forward      C1 float32
loss         warmstart_loss_v1 (frozen normalizations)
backward     autograd
clip         global-norm clip, pre/post norms recorded
step         AdamW, then the versioned warmup schedule
```

Worker count and prefetch depth affect only *when* a batch's bytes are ready,
never which batch is which: the pipeline submits frozen plans and yields
strictly in submission order, so its results are byte-identical to the
single-process loader — the property the topology benchmark asserts by
digest.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from ..model.production_model import build_candidate_model
from .synthetic_corpus import default_corpus_root, repository_relative
from .warmstart_checkpoint import (
    WARMSTART_CHECKPOINT_VERSION,
    WARMSTART_TRAINER_VERSION,
    CorpusIdentity,
    build_warmstart_checkpoint_payload,
    load_warmstart_checkpoint,
    save_warmstart_checkpoint,
)
from .warmstart_contract import (
    PILOT_CANDIDATES,
    PILOT_FIXED_CONTROLS,
    WARMSTART_EXAMPLE_VERSION,
)
from .warmstart_dataset import (
    ORDER_SEQUENTIAL,
    ORDER_SHUFFLE,
    TRAIN_ORDER_VERSION,
    DataCursor,
    WarmstartDataset,
    WarmstartDatasetError,
    _loader_init,
    _loader_task,
    batch_digest,
    batch_from_arrays,
    plan_batch,
)
from .warmstart_loss import (
    WARMSTART_LOSS_VERSION,
    WarmstartLossError,
    WarmstartLossWeights,
    warmstart_batch_loss,
)
from .warmstart_metrics import frozen_train_value_prior, run_validation
from .warmstart_seed import DECISION_SAMPLER_VERSION, SYNTHETIC_CORPUS_VERSION

#: The two configuration scopes. Everything an artifact may call a training
#: run is `pilot_candidate` scope; `unit_test` exists for the suite only.
SCOPE_PILOT = "pilot_candidate"
SCOPE_UNIT_TEST = "unit_test"

#: Metric columns every training step emits, in CSV order.
STEP_METRIC_COLUMNS = (
    "global_step",
    "epoch",
    "cursor_position",
    "batch_size",
    "keys_digest",
    "loss_total",
    "loss_policy",
    "loss_value",
    "loss_belief",
    "policy_supervised_decisions",
    "policy_weight_sum",
    "value_decisions",
    "belief_supervised_pieces",
    "legal_policy_entropy",
    "legal_policy_entropy_normalized",
    "learning_rate",
    "grad_norm_pre_clip",
    "grad_norm_post_clip",
    "parameter_norm",
    "data_wait_seconds",
    "h2d_seconds",
    "forward_seconds",
    "loss_seconds",
    "backward_seconds",
    "optimizer_seconds",
    "step_wall_seconds",
    "cache_hits",
    "cache_misses",
)


class WarmstartTrainerError(RuntimeError):
    """The trainer met a state it must not train through. Always raised."""


def keys_digest(keys: tuple) -> str:
    """SHA-256 over one batch's ordered `(game_id, decision_index)` identities."""
    hasher = hashlib.sha256()
    for game_id, index in keys:
        hasher.update(f"{game_id}|{index}\n".encode())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_FROZEN_CANDIDATES = {entry["candidate_id"]: entry for entry in PILOT_CANDIDATES}


@dataclass(frozen=True)
class WarmstartTrainConfig:
    """The semantic identity of one logical training run.

    Every field here participates in the train-config digest that checkpoints
    carry and resume compares. Loader topology (workers, prefetch, cache) is
    deliberately absent: it cannot change any batch, so it must not change the
    run's identity.
    """

    config_scope: str
    candidate_id: str
    model_candidate: str
    model_init_seed: int
    device: str
    precision: str
    batch_size: int
    learning_rate: float
    lambda_policy: float
    lambda_value: float
    lambda_belief: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    gradient_clip_norm: float
    warmup_steps: int
    split: str = "train"
    order: str = ORDER_SHUFFLE
    validation_split: str = "validation"
    validation_cadence_updates: int = 500
    validation_batches: "int | None" = None

    def __post_init__(self) -> None:
        if self.precision != "float32":
            raise WarmstartTrainerError(
                f"Phase 8 trains in float32 only, got {self.precision!r}"
            )
        if self.order not in (ORDER_SHUFFLE, ORDER_SEQUENTIAL):
            raise WarmstartTrainerError(f"unknown order {self.order!r}")
        if self.split == "test" or self.validation_split == "test":
            raise WarmstartTrainerError(
                "the sealed test split may never feed the trainer"
            )
        if self.batch_size < 1 or self.warmup_steps < 1:
            raise WarmstartTrainerError("batch_size and warmup_steps must be >= 1")
        if self.validation_cadence_updates < 1:
            raise WarmstartTrainerError("validation_cadence_updates must be >= 1")
        if self.config_scope == SCOPE_PILOT:
            self._check_pilot_candidate()
        elif self.config_scope == SCOPE_UNIT_TEST:
            if not self.candidate_id.startswith("unittest_"):
                raise WarmstartTrainerError(
                    "unit-test configurations must carry a 'unittest_' candidate id"
                )
            if self.model_candidate not in ("C0", "C1"):
                raise WarmstartTrainerError(
                    "unit tests may use C0 (preferred) or C1 only"
                )
        else:
            raise WarmstartTrainerError(f"unknown config scope {self.config_scope!r}")

    def _check_pilot_candidate(self) -> None:
        """A production config must *be* a frozen Agent 1 candidate."""
        frozen = _FROZEN_CANDIDATES.get(self.candidate_id)
        if frozen is None:
            raise WarmstartTrainerError(
                f"{self.candidate_id!r} is not one of Agent 1's frozen pilot "
                f"candidates: {sorted(_FROZEN_CANDIDATES)}"
            )
        controls = PILOT_FIXED_CONTROLS
        expected = {
            "model_candidate": controls["model"],
            "model_init_seed": controls["model_init_seed"],
            "precision": controls["precision"],
            "batch_size": controls["batch_size"],
            "learning_rate": frozen["learning_rate"],
            "lambda_policy": frozen["lambda_policy"],
            "lambda_value": frozen["lambda_value"],
            "lambda_belief": frozen["lambda_belief"],
            "weight_decay": controls["weight_decay"],
            "adam_beta1": controls["adam_betas"][0],
            "adam_beta2": controls["adam_betas"][1],
            "adam_epsilon": controls["adam_epsilon"],
            "gradient_clip_norm": controls["gradient_clip_norm"],
            "warmup_steps": 500,
            "split": "train",
            "order": ORDER_SHUFFLE,
            "validation_cadence_updates": controls["validation_cadence_updates"],
        }
        differing = [
            f"{name}: config {getattr(self, name)!r} vs frozen {value!r}"
            for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if differing:
            raise WarmstartTrainerError(
                "a pilot-scope configuration must match the frozen candidate "
                f"matrix exactly; {self.candidate_id} differs in: "
                + "; ".join(differing)
            )
        # The fixed controls name MPS as the pilot device; the CPU value exists
        # solely for the sanctioned exact-determinism reference run.
        if self.device not in ("mps", "cpu"):
            raise WarmstartTrainerError(
                f"pilot-scope device must be 'mps' (or 'cpu' for the reference "
                f"run), got {self.device!r}"
            )

    @staticmethod
    def from_pilot_candidate(
        candidate_id: str,
        *,
        device: str = "mps",
        validation_batches: "int | None" = None,
    ) -> "WarmstartTrainConfig":
        """The one constructor that yields a production configuration."""
        frozen = _FROZEN_CANDIDATES.get(candidate_id)
        if frozen is None:
            raise WarmstartTrainerError(
                f"{candidate_id!r} is not one of Agent 1's frozen pilot "
                f"candidates: {sorted(_FROZEN_CANDIDATES)}"
            )
        controls = PILOT_FIXED_CONTROLS
        return WarmstartTrainConfig(
            config_scope=SCOPE_PILOT,
            candidate_id=candidate_id,
            model_candidate=controls["model"],
            model_init_seed=int(controls["model_init_seed"]),
            device=device,
            precision=controls["precision"],
            batch_size=int(controls["batch_size"]),
            learning_rate=float(frozen["learning_rate"]),
            lambda_policy=float(frozen["lambda_policy"]),
            lambda_value=float(frozen["lambda_value"]),
            lambda_belief=float(frozen["lambda_belief"]),
            weight_decay=float(controls["weight_decay"]),
            adam_beta1=float(controls["adam_betas"][0]),
            adam_beta2=float(controls["adam_betas"][1]),
            adam_epsilon=float(controls["adam_epsilon"]),
            gradient_clip_norm=float(controls["gradient_clip_norm"]),
            warmup_steps=500,
            validation_cadence_updates=int(controls["validation_cadence_updates"]),
            validation_batches=validation_batches,
        )

    @property
    def lr_schedule(self) -> str:
        return f"linear_warmup_{self.warmup_steps}_steps_then_constant"

    def loss_weights(self) -> WarmstartLossWeights:
        return WarmstartLossWeights(
            lambda_policy=self.lambda_policy,
            lambda_value=self.lambda_value,
            lambda_belief=self.lambda_belief,
        )

    def identity(self) -> dict:
        """Every semantic field plus the frozen versions, JSON-safe."""
        return {
            "trainer_version": WARMSTART_TRAINER_VERSION,
            "loss_version": WARMSTART_LOSS_VERSION,
            "corpus_version": SYNTHETIC_CORPUS_VERSION,
            "example_version": WARMSTART_EXAMPLE_VERSION,
            "train_order_version": TRAIN_ORDER_VERSION,
            "sampler_version": DECISION_SAMPLER_VERSION,
            "config_scope": self.config_scope,
            "candidate_id": self.candidate_id,
            "model_candidate": self.model_candidate,
            "model_init_seed": int(self.model_init_seed),
            "device": self.device,
            "precision": self.precision,
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "lambda_policy": float(self.lambda_policy),
            "lambda_value": float(self.lambda_value),
            "lambda_belief": float(self.lambda_belief),
            "weight_decay": float(self.weight_decay),
            "adam_beta1": float(self.adam_beta1),
            "adam_beta2": float(self.adam_beta2),
            "adam_epsilon": float(self.adam_epsilon),
            "gradient_clip_norm": float(self.gradient_clip_norm),
            "optimizer": "AdamW",
            "lr_schedule": self.lr_schedule,
            "warmup_steps": int(self.warmup_steps),
            "split": self.split,
            "order": self.order,
            "validation_split": self.validation_split,
            "validation_cadence_updates": int(self.validation_cadence_updates),
            "validation_batches": (
                int(self.validation_batches)
                if self.validation_batches is not None
                else None
            ),
            "validation_selection": (
                "evenly_spread_v1" if self.validation_batches is not None else "full_split_sequential"
            ),
        }

    def digest(self) -> str:
        canonical = json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def pilot_candidate_ids() -> tuple:
    """The six frozen candidate identifiers, in Agent 1's order."""
    return tuple(entry["candidate_id"] for entry in PILOT_CANDIDATES)


def unit_test_config(
    *,
    candidate_id: str = "unittest_default",
    model_candidate: str = "C0",
    device: str = "cpu",
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    warmup_steps: int = 3,
    validation_cadence_updates: int = 1000000,
    validation_batches: "int | None" = 1,
    **overrides,
) -> WarmstartTrainConfig:
    """A tiny suite-only configuration. Never a production run."""
    return WarmstartTrainConfig(
        config_scope=SCOPE_UNIT_TEST,
        candidate_id=candidate_id,
        model_candidate=model_candidate,
        model_init_seed=overrides.pop("model_init_seed", 20260815),
        device=device,
        precision="float32",
        batch_size=batch_size,
        learning_rate=learning_rate,
        lambda_policy=overrides.pop("lambda_policy", 1.0),
        lambda_value=overrides.pop("lambda_value", 1.0),
        lambda_belief=overrides.pop("lambda_belief", 1.0),
        weight_decay=overrides.pop("weight_decay", 0.01),
        adam_beta1=overrides.pop("adam_beta1", 0.9),
        adam_beta2=overrides.pop("adam_beta2", 0.999),
        adam_epsilon=overrides.pop("adam_epsilon", 1e-8),
        gradient_clip_norm=overrides.pop("gradient_clip_norm", 1.0),
        warmup_steps=warmup_steps,
        validation_cadence_updates=validation_cadence_updates,
        validation_batches=validation_batches,
        **overrides,
    )


@dataclass(frozen=True)
class LoaderTopology:
    """Loader infrastructure. Tunable without touching any batch's identity."""

    workers: int = 8
    prefetch: int = 2
    record_cache_size: int = 512

    def __post_init__(self) -> None:
        if self.workers < 1 or self.prefetch < 1 or self.record_cache_size < 1:
            raise WarmstartTrainerError("topology values must be >= 1")

    def to_dict(self) -> dict:
        return {
            "workers": int(self.workers),
            "prefetch": int(self.prefetch),
            "record_cache_size": int(self.record_cache_size),
        }


# ---------------------------------------------------------------------------
# The persistent ordered pipeline
# ---------------------------------------------------------------------------


class _BatchPipeline:
    """Frozen batch plans in, byte-identical batches out, strictly in order.

    Plans are pure functions of the data cursor; workers receive
    `(batch_index, keys)` and results are consumed FIFO, so parallelism can
    only change arrival *times*. `workers=1` builds in-process through the
    same `batch_arrays`, which is the bit-identical reference path.
    """

    def __init__(
        self,
        dataset: WarmstartDataset,
        cursor: DataCursor,
        *,
        workers: int,
        prefetch: int,
    ) -> None:
        self.dataset = dataset
        self.workers = int(workers)
        self.prefetch = max(1, int(prefetch))
        self._universe = dataset.universe(cursor.split)
        self._plan_cursor = cursor
        self._pending: deque = deque()
        self._batch_index = 0
        self._pool = None
        if self.workers > 1:
            options = {
                "root": str(dataset.root),
                "splits": tuple(dataset.splits),
                "record_cache_size": dataset.record_cache_size,
            }
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_loader_init,
                initargs=(options,),
            )
            self._fill()

    def _fill(self) -> None:
        target = self.workers * self.prefetch
        while len(self._pending) < target:
            keys, cursor_after = plan_batch(self._universe, self._plan_cursor)
            future = self._pool.submit(_loader_task, (self._batch_index, keys))
            self._pending.append((future, cursor_after))
            self._plan_cursor = cursor_after
            self._batch_index += 1

    def next(self) -> tuple:
        """`(arrays, metadata, stats, cursor_after, wait_seconds)`."""
        if self._pool is None:
            keys, cursor_after = plan_batch(self._universe, self._plan_cursor)
            self._plan_cursor = cursor_after
            started = time.perf_counter()
            arrays, metadata, stats = self.dataset.batch_arrays(keys)
            return arrays, metadata, stats, cursor_after, time.perf_counter() - started
        future, cursor_after = self._pending.popleft()
        started = time.perf_counter()
        _index, arrays, metadata, stats = future.result()
        waited = time.perf_counter() - started
        self._fill()
        return arrays, metadata, stats, cursor_after, waited

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None
        self._pending.clear()


# ---------------------------------------------------------------------------
# The trainer
# ---------------------------------------------------------------------------


class WarmstartTrainer:
    """One logical Phase 8 training run over the accepted synthetic corpus."""

    def __init__(
        self,
        config: WarmstartTrainConfig,
        corpus_identity: CorpusIdentity,
        *,
        root: "str | Path | None" = None,
        topology: "LoaderTopology | None" = None,
        require_complete_split: bool = True,
        value_prior: "tuple | None" = None,
        run_label: str = "",
        fsync_checkpoints: bool = True,
        _restored: "dict | None" = None,
    ) -> None:
        if not isinstance(corpus_identity, CorpusIdentity):
            raise WarmstartTrainerError(
                "the trainer requires a verified CorpusIdentity; measure it with "
                "warmstart_checkpoint.verify_corpus_identity before construction"
            )
        self.config = config
        self.corpus_identity = corpus_identity
        self.topology = topology or LoaderTopology()
        self.run_label = str(run_label)
        self.fsync_checkpoints = bool(fsync_checkpoints)
        # The production resolver is the only default; tests pass mini roots.
        self.root = Path(root) if root is not None else default_corpus_root()
        self.dataset = WarmstartDataset(
            self.root,
            record_cache_size=self.topology.record_cache_size,
            require_complete_split=require_complete_split,
        )
        # Validation reads through its own dataset instance so a held-out pass
        # cannot evict the training loader's record cache.
        self.validation_dataset = WarmstartDataset(
            self.root,
            record_cache_size=self.topology.record_cache_size,
            require_complete_split=require_complete_split,
        )
        self.value_prior = (
            tuple(value_prior) if value_prior is not None else frozen_train_value_prior()
        )
        self.device = torch.device(config.device)
        self.counters = {
            "non_finite_losses": 0,
            "non_finite_gradients": 0,
            "non_finite_parameters": 0,
            "illegal_targets": 0,
            "data_mismatches": 0,
            "checkpoint_errors": 0,
        }
        self.validation_seconds = 0.0
        self._pipeline: "_BatchPipeline | None" = None

        if _restored is None:
            self.model = build_candidate_model(
                config.model_candidate,
                seed=config.model_init_seed,
                device=self.device,
                dtype=torch.float32,
            )
            self.model.train()
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
            self.cursor = DataCursor(
                split=config.split, batch_size=config.batch_size, order=config.order
            )
            self.global_step = 0
            self.examples_consumed = 0
            self.best_validation = {"score": None, "step": None}
            self.validation_history: list = []
        else:
            self.model = _restored["model"]
            self.model.train()
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
            self.optimizer.load_state_dict(_restored["optimizer_state"])
            self.scheduler.load_state_dict(_restored["scheduler_state"])
            self.cursor = _restored["cursor"]
            self.global_step = _restored["global_step"]
            self.examples_consumed = _restored["examples_consumed"]
            self.best_validation = _restored["best_validation"]
            self.validation_history = _restored["validation_history"]

    # -- construction helpers ------------------------------------------------

    def _build_optimizer(self) -> torch.optim.AdamW:
        config = self.config
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )

    def _warmup_factor(self, step: int) -> float:
        return min(1.0, (step + 1) / self.config.warmup_steps)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._warmup_factor)

    @classmethod
    def resume(
        cls,
        checkpoint_path: "str | Path",
        *,
        config: WarmstartTrainConfig,
        corpus_identity: CorpusIdentity,
        root: "str | Path | None" = None,
        topology: "LoaderTopology | None" = None,
        require_complete_split: bool = True,
        value_prior: "tuple | None" = None,
        run_label: str = "",
        fsync_checkpoints: bool = True,
    ) -> "WarmstartTrainer":
        """Continue the exact logical run a checkpoint froze.

        Every identity mismatch — train config, corpus digests, versions,
        model contract — raises inside
        :func:`warmstart_checkpoint.load_warmstart_checkpoint` before any
        state is touched.
        """
        restored = load_warmstart_checkpoint(
            checkpoint_path,
            expected_train_config=config.identity(),
            expected_train_config_digest=config.digest(),
            expected_corpus_identity=corpus_identity,
            device=config.device,
        )
        return cls(
            config,
            corpus_identity,
            root=root,
            topology=topology,
            require_complete_split=require_complete_split,
            value_prior=value_prior,
            run_label=run_label,
            fsync_checkpoints=fsync_checkpoints,
            _restored=restored,
        )

    # -- the pipeline ---------------------------------------------------------

    def _ensure_pipeline(self) -> _BatchPipeline:
        if self._pipeline is None:
            self._pipeline = _BatchPipeline(
                self.dataset,
                self.cursor,
                workers=self.topology.workers,
                prefetch=self.topology.prefetch,
            )
        return self._pipeline

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.shutdown()
            self._pipeline = None

    def __enter__(self) -> "WarmstartTrainer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- training -------------------------------------------------------------

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

    def train_updates(
        self,
        updates: int,
        *,
        timing: bool = False,
        capture_batch_digests: bool = False,
        best_checkpoint_path: "str | Path | None" = None,
        on_step=None,
    ) -> list:
        """Run `updates` optimizer updates; return one metric row per update.

        `timing=True` synchronizes the device between phases so the per-phase
        milliseconds mean what they say (benchmark mode). Validation runs
        whenever `global_step` reaches a multiple of the frozen cadence, so an
        interrupted-and-resumed run validates at the same global steps as an
        uninterrupted one.
        """
        pipeline = self._ensure_pipeline()
        rows = []
        for _ in range(int(updates)):
            step_started = time.perf_counter()
            arrays, metadata, stats, cursor_after, waited = pipeline.next()
            row = {
                "data_wait_seconds": waited,
                "cache_hits": stats.get("cache_hits", 0),
                "cache_misses": stats.get("cache_misses", 0),
            }
            if capture_batch_digests:
                row["batch_digest"] = batch_digest(arrays, metadata)
            try:
                batch = batch_from_arrays(arrays, metadata)
            except WarmstartDatasetError:
                self.counters["data_mismatches"] += 1
                raise
            if any(split != self.config.split for split in batch.corpus_splits):
                self.counters["data_mismatches"] += 1
                raise WarmstartTrainerError(
                    "a batch contains examples from outside the training split"
                )

            marks = [time.perf_counter()]

            observations = batch.model_input().to(self.device)
            targets = batch.targets
            legal_mask = targets.legal_mask.to(self.device)
            policy_actions = targets.policy_action_model.to(self.device)
            policy_weights = targets.policy_weight.to(self.device)
            value_targets = targets.value_target.to(self.device)
            belief_targets = targets.belief_target.to(self.device)
            belief_mask = targets.belief_mask.to(self.device)
            if timing:
                self._sync()
            marks.append(time.perf_counter())

            outputs = self.model.forward_observation(observations)
            if timing:
                self._sync()
            marks.append(time.perf_counter())

            try:
                loss = warmstart_batch_loss(
                    outputs,
                    legal_mask=legal_mask,
                    policy_actions=policy_actions,
                    policy_weights=policy_weights,
                    value_targets=value_targets,
                    belief_targets=belief_targets,
                    belief_mask=belief_mask,
                    weights=self.config.loss_weights(),
                )
            except WarmstartLossError:
                self.counters["illegal_targets"] += 1
                raise
            if not loss.all_finite():
                self.counters["non_finite_losses"] += 1
                raise WarmstartTrainerError(
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
                raise WarmstartTrainerError(
                    f"non-finite gradient norm at global step {self.global_step + 1}"
                )
            self.optimizer.step()
            self.scheduler.step()
            parameter_norm = self._parameter_norm()
            if not bool(torch.isfinite(parameter_norm)):
                self.counters["non_finite_parameters"] += 1
                raise WarmstartTrainerError(
                    f"non-finite parameters at global step {self.global_step + 1}"
                )
            if timing:
                self._sync()
            marks.append(time.perf_counter())

            self.cursor = cursor_after
            self.global_step += 1
            self.examples_consumed += batch.batch_size

            row.update(loss.to_dict())
            row.update(
                {
                    "global_step": self.global_step,
                    # The cursor now names the *next* batch; the epoch/position
                    # of the batch just consumed are what the row describes.
                    "epoch": self.cursor.epoch,
                    "cursor_position": self.cursor.position,
                    "keys_digest": keys_digest(batch.keys),
                    "learning_rate": learning_rate,
                    "grad_norm_pre_clip": float(pre_clip),
                    "grad_norm_post_clip": float(post_clip),
                    "parameter_norm": float(parameter_norm),
                    "h2d_seconds": marks[1] - marks[0],
                    "forward_seconds": marks[2] - marks[1],
                    "loss_seconds": marks[3] - marks[2],
                    "backward_seconds": marks[4] - marks[3],
                    "optimizer_seconds": marks[5] - marks[4],
                    "step_wall_seconds": time.perf_counter() - step_started,
                }
            )
            rows.append(row)
            if on_step is not None:
                on_step(row, batch)

            if self.global_step % self.config.validation_cadence_updates == 0:
                self.run_cadence_validation(best_checkpoint_path=best_checkpoint_path)
        return rows

    # -- validation -----------------------------------------------------------

    def run_cadence_validation(
        self, *, best_checkpoint_path: "str | Path | None" = None
    ) -> dict:
        """One held-out pass at the current step, plus best-checkpoint logic.

        Strictly lower selection score wins. The pass reads through the
        separate validation dataset with its own fresh sequential cursor, so
        the training cursor, optimizer, scheduler and model mode are untouched
        — properties the suite asserts directly.
        """
        result = run_validation(
            self.model,
            self.validation_dataset,
            split=self.config.validation_split,
            value_prior=self.value_prior,
            batches=self.config.validation_batches,
            batch_size=self.config.batch_size,
            device=self.device,
            # Cadence-sized passes sample evenly across the split: the frozen
            # sequential order is cell-major, so a prefix would see only the
            # policy-unsupervised random-vs-random cells.
            spread=self.config.validation_batches is not None,
        )
        entry = {
            "global_step": self.global_step,
            "examples_consumed": self.examples_consumed,
            "selection_score": result.selection_score,
            "policy_ce_ratio": result.policy["ce_ratio"],
            "value_ce_ratio": result.value["ce_ratio"],
            "belief_ce_ratio": result.belief["ce_ratio"],
            "policy_model_ce": result.policy["model_ce"],
            "value_model_ce": result.value["model_ce"],
            "belief_model_ce": result.belief["model_ce"],
            "examples": result.examples,
            "games": result.games,
            "batches": result.batches,
            "seconds": result.seconds,
            "is_best": False,
        }
        score = result.selection_score
        best = self.best_validation.get("score")
        if score is not None and (best is None or score < best):
            self.best_validation = {"score": float(score), "step": self.global_step}
            entry["is_best"] = True
            if best_checkpoint_path is not None:
                self.save_checkpoint(best_checkpoint_path)
        self.validation_history.append(entry)
        self.validation_seconds += result.seconds
        return entry

    # -- checkpointing ----------------------------------------------------------

    def diagnostics(self) -> dict:
        """Non-semantic facts recorded in every checkpoint, never compared."""
        return {
            "resolved_corpus_root": str(self.root),
            "resolved_corpus_root_repository_relative": repository_relative(self.root),
            "device": str(self.device),
            "run_label": self.run_label,
            "topology": self.topology.to_dict(),
            "wall_clock_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "warmstart_checkpoint_version": WARMSTART_CHECKPOINT_VERSION,
        }

    def save_checkpoint(self, path: "str | Path", *, crash_hook=None) -> dict:
        try:
            payload = build_warmstart_checkpoint_payload(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                train_config=self.config.identity(),
                train_config_digest=self.config.digest(),
                corpus_identity=self.corpus_identity,
                cursor=self.cursor,
                global_step=self.global_step,
                examples_consumed=self.examples_consumed,
                best_validation=self.best_validation,
                validation_history=self.validation_history,
                diagnostics=self.diagnostics(),
            )
            return save_warmstart_checkpoint(
                payload, path, fsync=self.fsync_checkpoints, crash_hook=crash_hook
            )
        except Exception:
            self.counters["checkpoint_errors"] += 1
            raise

    # -- comparison support -------------------------------------------------------

    def parameter_snapshot(self) -> dict:
        """Detached CPU float32 copies of every parameter, by name."""
        return {
            name: tensor.detach().to("cpu", torch.float32).clone()
            for name, tensor in self.model.state_dict().items()
        }

    def state_summary(self) -> dict:
        """The comparable logical state of the run, JSON-safe."""
        optimizer_state = self.optimizer.state_dict()
        structure = {
            str(index): sorted(entry)
            for index, entry in optimizer_state["state"].items()
        }
        return {
            "global_step": self.global_step,
            "examples_consumed": self.examples_consumed,
            "cursor": self.cursor.to_dict(),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "scheduler_last_epoch": int(self.scheduler.last_epoch),
            "best_validation": dict(self.best_validation),
            "validation_steps": [
                entry["global_step"] for entry in self.validation_history
            ],
            "validation_best_flags": [
                entry["is_best"] for entry in self.validation_history
            ],
            "optimizer_state_structure": structure,
            "counters": dict(self.counters),
        }


__all__ = [
    "LoaderTopology",
    "SCOPE_PILOT",
    "SCOPE_UNIT_TEST",
    "STEP_METRIC_COLUMNS",
    "WARMSTART_TRAINER_VERSION",
    "WarmstartTrainConfig",
    "WarmstartTrainer",
    "WarmstartTrainerError",
    "keys_digest",
    "pilot_candidate_ids",
    "unit_test_config",
]
