"""Phase 16 Agent 3: the window learner.

Specification source: `03_AGENT_3_TRAINING_LOOP_V2.md` sections 2.3-2.5.

The objective is not reimplemented
----------------------------------
Every number that decides what the gradient *is* comes from the accepted Phase
9 code: :func:`~stratego.training.phase9_loss.phase9_batch_loss` is the
objective -- PPO clipped surrogate + 0.5 value + 0.25 belief auxiliary +
beta*KL - c_H*H -- :class:`~stratego.training.phase9_trainer.KLController` is
the damping controller with its accepted thresholds, and the advantage filter,
standardization and W/D/L targets are the accepted ones. This module is the
*schedule* around them, plus two things Phase 14 did not have:

1. the per-iteration learning-rate and entropy schedules (`schedules.py`);
2. an exponential moving average of the weights, evaluation-side only.

Why the EMA never trains
------------------------
The EMA is a second set of tensors updated after each optimizer step and read
only by the evaluator. It is not in the optimizer, it never receives a
gradient, and the collector's behavior snapshot is taken from the *raw*
weights: a run whose behavior policy were the EMA would have a PPO denominator
that no longer matched the policy being updated. Checkpoints store both states
so an arm can be re-scored either way after the fact.

The window is the statistics unit
---------------------------------
`tau = max(Q_0.75(|A|), 0.01)` is a per-window statistic exactly as it was a
per-sealed-iteration statistic in Phase 9: PPO eligibility of one decision
depends on every other learner decision in the same window. That is why the
rows arrive as a list and the statistics are computed once, before the first
minibatch, rather than per batch.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..phase9_contract import (
    ADVANTAGE_FILTER_FLOOR,
    ADVANTAGE_FILTER_QUANTILE,
    ADVANTAGE_STANDARDIZATION_EPSILON,
    CLIP_FRACTION_HARD_LIMIT,
    KL_HARD_LIMIT,
    OPTIMIZER_CONSTRAINTS,
    advantage_filter_threshold,
)
from ..phase10b_contract import PHASE9_CANONICAL_INITIAL_KL_BETA as INITIAL_KL_BETA
from ..phase9_loss import Phase9LossError, behavior_probability_matrix, phase9_batch_loss
from ..phase9_trainer import KLController
from .contract import (
    DOMAIN_TRAINING_ORDER,
    PHASE16_TRAINER_VERSION,
    ArmConfig,
    Phase16TrainingError,
    derive_train_seed,
)
from .schedules import entropy_coefficient_for, learning_rate_for

GRADIENT_CLIP_NORM = float(OPTIMIZER_CONSTRAINTS.get("gradient_clip_norm", 1.0))


class Phase16TrainerError(Phase16TrainingError):
    """A Phase 16 update could not be performed as specified."""


# ---------------------------------------------------------------------------
# Per-window advantage statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowStatistics:
    """The `phase9_advantage_v1` state of one window, on the accepted formulas."""

    iteration: int
    rows: int
    threshold: float
    eligible: int
    retention_fraction: float
    mean_eligible: float
    std_eligible: float
    zero_variance: bool
    no_eligible: bool
    advantage_min: float
    advantage_max: float
    advantage_mean: float
    advantage_abs_mean: float

    def is_eligible(self, advantage: float) -> bool:
        return abs(float(advantage)) >= self.threshold

    def standardize(self, advantage: float) -> float:
        if self.no_eligible:
            return 0.0
        return (float(advantage) - self.mean_eligible) / (
            self.std_eligible + ADVANTAGE_STANDARDIZATION_EPSILON
        )

    def to_dict(self) -> dict:
        return {
            "iteration": int(self.iteration),
            "rows": int(self.rows),
            "quantile": ADVANTAGE_FILTER_QUANTILE,
            "floor": ADVANTAGE_FILTER_FLOOR,
            "threshold": self.threshold,
            "eligible": int(self.eligible),
            "retention_fraction": self.retention_fraction,
            "mean_eligible": self.mean_eligible,
            "std_eligible": self.std_eligible,
            "zero_variance": bool(self.zero_variance),
            "no_eligible": bool(self.no_eligible),
            "advantage_min": self.advantage_min,
            "advantage_max": self.advantage_max,
            "advantage_mean": self.advantage_mean,
            "advantage_abs_mean": self.advantage_abs_mean,
        }


def window_statistics(rows, *, iteration: int) -> WindowStatistics:
    """The filter threshold and standardization moments of one window."""
    if not rows:
        raise Phase16TrainerError("a window with no rows has no statistics")
    advantages = np.asarray([float(row.advantage) for row in rows], dtype=np.float64)
    threshold = advantage_filter_threshold(list(advantages))
    selected = advantages[np.abs(advantages) >= threshold]
    no_eligible = selected.size == 0
    mean = float(selected.mean()) if not no_eligible else 0.0
    std = float(selected.std()) if not no_eligible else 0.0
    return WindowStatistics(
        iteration=int(iteration),
        rows=len(rows),
        threshold=float(threshold),
        eligible=int(selected.size),
        retention_fraction=float(selected.size) / len(rows),
        mean_eligible=mean,
        std_eligible=std,
        zero_variance=bool(not no_eligible and std == 0.0),
        no_eligible=bool(no_eligible),
        advantage_min=float(advantages.min()),
        advantage_max=float(advantages.max()),
        advantage_mean=float(advantages.mean()),
        advantage_abs_mean=float(np.abs(advantages).mean()),
    )


def apply_statistics(rows, statistics: WindowStatistics) -> None:
    """Stamp eligibility and the standardized advantage onto every row."""
    for row in rows:
        row.ppo_eligible = bool(statistics.is_eligible(row.advantage))
        row.standardized_advantage = float(statistics.standardize(row.advantage))


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def train_order(rows, *, arm_id: str, iteration: int, epoch: int) -> list:
    """The shuffled row order of one epoch, from the Phase 16 order domain.

    Deterministic from `(arm, iteration, epoch)` alone, so a resumed window
    consumes its rows in the order the crashed one would have.
    """
    order = list(range(len(rows)))
    seed = derive_train_seed(DOMAIN_TRAINING_ORDER, arm_id, iteration, epoch)
    random.Random(seed).shuffle(order)
    return order


def build_arrays(rows) -> dict:
    """Collate one minibatch, keeping the accepted privilege boundary visible.

    `behavior_probabilities` is the accepted dense builder over the stored
    float32 bytes -- never recomputed, never renormalized -- and `observation`
    is the only model input in the mapping.
    """
    if not rows:
        raise Phase16TrainerError("cannot build a minibatch from no rows")
    return {
        "observation": np.stack(
            [np.ascontiguousarray(row.observation, dtype=np.float32) for row in rows]
        ),
        "legal_mask": np.stack([row.legal_mask for row in rows]),
        "sampled_action_model": np.asarray(
            [row.sampled_action_model for row in rows], dtype=np.int64
        ),
        "behavior_action_probability": np.asarray(
            [row.behavior_action_probability for row in rows], dtype=np.float32
        ),
        "behavior_probabilities": behavior_probability_matrix(rows),
        "standardized_advantage": np.asarray(
            [row.standardized_advantage for row in rows], dtype=np.float32
        ),
        "ppo_eligible": np.asarray([row.ppo_eligible for row in rows], dtype=bool),
        "wdl_target": np.asarray([row.wdl_target for row in rows], dtype=np.float32),
        "belief_target": np.stack([row.belief_target for row in rows]),
        "belief_mask": np.stack([row.belief_mask for row in rows]),
    }


# ---------------------------------------------------------------------------
# The EMA
# ---------------------------------------------------------------------------


class WeightEMA:
    """An exponential moving average of the model state. Evaluation-side only.

    Averages float tensors and *copies* everything else: an integer buffer such
    as a step counter has no meaningful average, and quietly casting one to
    float would produce an EMA state dict that no longer loads.
    """

    def __init__(self, model, decay: float) -> None:
        if not 0.0 < float(decay) < 1.0:
            raise Phase16TrainerError(f"ema decay must be in (0, 1): {decay!r}")
        self.decay = float(decay)
        self.updates = 0
        self.state = {
            name: tensor.detach().clone().to("cpu")
            for name, tensor in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model) -> None:
        decay = self.decay
        for name, tensor in model.state_dict().items():
            current = tensor.detach().to("cpu")
            stored = self.state[name]
            if stored.is_floating_point():
                stored.mul_(decay).add_(current, alpha=1.0 - decay)
            else:
                self.state[name] = current.clone()
        self.updates += 1

    def state_dict(self) -> dict:
        return {name: tensor.clone() for name, tensor in self.state.items()}

    def load_state_dict(self, payload: dict, *, updates: int = 0) -> None:
        self.state = {
            name: torch.as_tensor(value).clone() for name, value in payload.items()
        }
        self.updates = int(updates)

    def to_dict(self) -> dict:
        return {"present": True, "decay": self.decay, "updates": int(self.updates)}


# ---------------------------------------------------------------------------
# The learner
# ---------------------------------------------------------------------------


@dataclass
class WindowUpdate:
    """One window's optimization, as numbers the telemetry row can carry."""

    iteration: int
    steps: int = 0
    examples: int = 0
    seconds: float = 0.0
    learning_rate: float = 0.0
    entropy_coefficient: float = 0.0
    kl_beta: float = 0.0
    statistics: dict = field(default_factory=dict)
    epochs: list = field(default_factory=list)
    means: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "iteration": int(self.iteration),
            "optimizer_steps": int(self.steps),
            "examples": int(self.examples),
            "seconds": round(float(self.seconds), 3),
            "learning_rate": self.learning_rate,
            "entropy_coefficient": self.entropy_coefficient,
            "kl_beta": self.kl_beta,
            "advantage_statistics": dict(self.statistics),
            "epochs": list(self.epochs),
            **{key: value for key, value in self.means.items()},
        }


class WindowTrainer:
    """The bulk-synchronous PPO/KL learner over one window of rows."""

    def __init__(
        self,
        config: ArmConfig,
        model,
        *,
        device: "str | None" = None,
        optimizer=None,
        kl_beta: float = INITIAL_KL_BETA,
    ) -> None:
        self.config = config
        self.model = model
        self.device = torch.device(device or config.device)
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate_for(config, 1),
            betas=tuple(float(v) for v in OPTIMIZER_CONSTRAINTS["adam_betas"]),
            eps=float(OPTIMIZER_CONSTRAINTS["adam_epsilon"]),
            weight_decay=float(OPTIMIZER_CONSTRAINTS["weight_decay"]),
        )
        self.controller = KLController(beta=float(kl_beta))
        self.ema = WeightEMA(model, config.ema_decay) if config.ema else None
        self.global_step = 0
        self.examples_consumed = 0
        self.counters = {
            "non_finite_losses": 0,
            "non_finite_gradients": 0,
            "non_finite_parameters": 0,
            "illegal_targets": 0,
            "kl_vetoes": 0,
            "clip_vetoes": 0,
        }

    # -- limits ------------------------------------------------------------

    def _check_hard_limits(self, *, iteration: int, epoch: int, mean_kl: float, clip_fraction: float) -> None:
        """The accepted Phase 9 vetoes, with the accepted abort semantics."""
        if mean_kl > KL_HARD_LIMIT:
            self.counters["kl_vetoes"] += 1
            raise Phase16TrainerError(
                f"window {iteration} epoch {epoch}: mean behavior KL {mean_kl:.5f} "
                f"exceeds the hard limit {KL_HARD_LIMIT}"
            )
        if clip_fraction > CLIP_FRACTION_HARD_LIMIT:
            self.counters["clip_vetoes"] += 1
            raise Phase16TrainerError(
                f"window {iteration} epoch {epoch}: clip fraction {clip_fraction:.4f} "
                f"exceeds the hard limit {CLIP_FRACTION_HARD_LIMIT}"
            )

    def _grad_norm(self) -> torch.Tensor:
        norms = [
            parameter.grad.detach().norm(2)
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        if not norms:  # pragma: no cover - a step always has gradients
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norms), 2)

    # -- one window --------------------------------------------------------

    def train_window(self, rows, *, iteration: int, may_start_step=None) -> WindowUpdate:
        """Train on one window's rows. Returns what happened, never a claim."""
        if not rows:
            raise Phase16TrainerError(f"window {iteration} produced no rows to train on")
        rate = learning_rate_for(self.config, iteration)
        coefficient = entropy_coefficient_for(self.config, iteration)
        for group in self.optimizer.param_groups:
            group["lr"] = rate

        statistics = window_statistics(rows, iteration=iteration)
        apply_statistics(rows, statistics)

        update = WindowUpdate(
            iteration=int(iteration),
            learning_rate=rate,
            entropy_coefficient=coefficient,
            statistics=statistics.to_dict(),
        )
        collected: dict = {}
        started = time.perf_counter()
        size = int(self.config.minibatch_size)
        self.model.train()

        for epoch in range(1, int(self.config.epochs) + 1):
            order = train_order(
                rows, arm_id=self.config.arm_id, iteration=iteration, epoch=epoch
            )
            for start in range(0, len(order), size):
                if may_start_step is not None and not bool(may_start_step()):
                    break
                batch = [rows[index] for index in order[start : start + size]]
                if len(batch) < 2:  # a 1-row minibatch has no usable moments
                    continue
                loss = self._step(batch, coefficient)
                for key, value in loss.to_dict().items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        collected.setdefault(key, []).append(float(value))
                update.steps += 1
                update.examples += len(batch)
            entry = self.controller.update(iteration=int(iteration), epoch=epoch)
            self._check_hard_limits(
                iteration=iteration,
                epoch=epoch,
                mean_kl=float(entry["mean_epoch_kl"]),
                clip_fraction=float(entry["epoch_clip_fraction"]),
            )
            update.epochs.append(entry)

        update.seconds = time.perf_counter() - started
        update.kl_beta = float(self.controller.beta)
        update.means = {
            f"mean_{key}": float(np.mean(values)) for key, values in collected.items()
        }
        return update

    def _step(self, batch, coefficient: float):
        arrays = build_arrays(batch)
        tensors = {
            name: torch.from_numpy(np.ascontiguousarray(value)).to(self.device)
            for name, value in arrays.items()
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
                entropy_coefficient=float(coefficient),
            )
        except Phase9LossError:
            self.counters["illegal_targets"] += 1
            raise
        if not loss.all_finite():
            self.counters["non_finite_losses"] += 1
            raise Phase16TrainerError(
                f"non-finite loss at global step {self.global_step + 1}"
            )

        self.optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        pre_clip = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), GRADIENT_CLIP_NORM
        )
        if not bool(torch.isfinite(pre_clip)):
            self.counters["non_finite_gradients"] += 1
            raise Phase16TrainerError(
                f"non-finite gradient norm at global step {self.global_step + 1}"
            )
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model)
        self.controller.observe(
            mean_kl=float(loss.kl.detach()),
            examples=len(batch),
            clipped=int(loss.ppo_clipped),
            ppo_examples=int(loss.ppo_examples),
        )
        self.global_step += 1
        self.examples_consumed += len(batch)
        return loss

    # -- state -------------------------------------------------------------

    def trainer_state(self) -> dict:
        return {
            "trainer_version": PHASE16_TRAINER_VERSION,
            "global_optimizer_step": int(self.global_step),
            "examples_consumed": int(self.examples_consumed),
            "kl_controller": self.controller.to_dict(),
            "ema": self.ema.to_dict() if self.ema is not None else {"present": False},
            "counters": dict(self.counters),
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
        }

    def restore_state(self, payload: dict) -> None:
        self.global_step = int(payload.get("global_optimizer_step", 0))
        self.examples_consumed = int(payload.get("examples_consumed", 0))
        controller = payload.get("kl_controller") or {}
        self.controller.beta = float(controller.get("beta", self.controller.beta))
        self.controller.history = [dict(entry) for entry in controller.get("history", [])]
        self.counters.update(payload.get("counters") or {})


def trainer_semantics() -> dict:
    return {
        "trainer_version": PHASE16_TRAINER_VERSION,
        "objective": "stratego.training.phase9_loss.phase9_batch_loss, unchanged",
        "kl_controller": "stratego.training.phase9_trainer.KLController, unchanged",
        "filter": "the accepted tau = max(Q_0.75(|A|), 0.01), per window",
        "phase16_own": [
            "the per-iteration learning-rate and entropy schedules",
            "the evaluation-side weight EMA",
            "the window training order",
        ],
        "ema": "never trains, never collects; the behavior snapshot is the raw model",
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
    }


__all__ = [
    "GRADIENT_CLIP_NORM",
    "Phase16TrainerError",
    "WeightEMA",
    "WindowStatistics",
    "WindowTrainer",
    "WindowUpdate",
    "apply_statistics",
    "build_arrays",
    "train_order",
    "trainer_semantics",
    "window_statistics",
]
