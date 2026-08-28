"""Phase 17 Agent 2: the one-epoch move update.

Specification sources: Agent 2 instruction section 6, common contract section 9.

Shape
-----
```text
one window of transitions
  -> per-window advantage filter and standardization  (accepted phase9 formulas)
  -> ONE epoch of shuffled minibatches
  -> per minibatch: forward, phase17_batch_loss, backward, clip, step
  -> after each step: the evaluation-only EMA follows the raw weights
  -> after the epoch: the accepted KL controller closes and beta moves
```

The trainer returns what happened, never a claim
------------------------------------------------
`MoveUpdate` carries the schedule values it actually used, the filter
statistics, the epoch's KL entry, the counters, and the boundary/terminal
provenance split of the rows it trained on. Nothing in it is inferred.

Refusals
--------
Nonfinite loss, nonfinite gradient norm, mean epoch KL above the accepted hard
limit, clip fraction above the accepted hard limit, an iteration outside the
frozen horizon, and -- the Phase 17 addition -- a row whose behavior
model-state digest is not one the live cell has held. That last one is
immediate stop condition I2 ("any decision recorded under the wrong current
move-policy digest") checked at the moment the row would be trained on, which
is the last point at which it can still be caught.

RAW trains, EMA follows
-----------------------
The EMA is a second set of tensors updated after each optimizer step and read
only by evaluation. It never enters the training population: a run whose
behavior policy were the EMA would store a PPO denominator that no set of
weights in the run ever produced. `assert_ema_never_acted` is the check.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..phase16.trainer import WeightEMA
from ..phase9_contract import (
    ADVANTAGE_FILTER_FLOOR,
    ADVANTAGE_FILTER_QUANTILE,
    ADVANTAGE_STANDARDIZATION_EPSILON,
    CLIP_FRACTION_HARD_LIMIT,
    KL_HARD_LIMIT,
    MINIBATCH_SIZE,
    advantage_filter_threshold,
)
from ..phase9_loss import behavior_probability_matrix
from ..phase9_trainer import KLController
from .move_contract import (
    DOMAIN_TRAINING_ORDER,
    MOVE_EPOCHS_PER_ITERATION,
    MOVE_GRADIENT_CLIP_NORM,
    MOVE_TRAINER_VERSION,
    PROVENANCE_BOOTSTRAP,
    PROVENANCE_TERMINAL,
    MoveScheduleHorizon,
    Phase17MoveError,
    derive_move_seed,
    require_run_id,
)
from .move_loss import Phase17LossError, phase17_batch_loss
from .move_snapshot import CurrentMovePolicy


class Phase17TrainerError(Phase17MoveError):
    """A Phase 17 move update could not be performed as specified."""


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
        raise Phase17TrainerError("a window with no rows has no statistics")
    advantages = np.asarray([float(row.advantage_target) for row in rows], dtype=np.float64)
    if not np.isfinite(advantages).all():
        raise Phase17TrainerError("a non-finite advantage entered the window statistics")
    threshold = advantage_filter_threshold(list(advantages))
    selected = advantages[np.abs(advantages) >= threshold]
    no_eligible = selected.size == 0
    return WindowStatistics(
        iteration=int(iteration),
        rows=len(rows),
        threshold=float(threshold),
        eligible=int(selected.size),
        retention_fraction=float(selected.size) / len(rows),
        mean_eligible=float(selected.mean()) if not no_eligible else 0.0,
        std_eligible=float(selected.std()) if not no_eligible else 0.0,
        zero_variance=bool(not no_eligible and float(selected.std()) == 0.0),
        no_eligible=bool(no_eligible),
        advantage_min=float(advantages.min()),
        advantage_max=float(advantages.max()),
        advantage_mean=float(advantages.mean()),
        advantage_abs_mean=float(np.abs(advantages).mean()),
    )


def apply_statistics(rows, statistics: WindowStatistics) -> None:
    """Stamp eligibility and the standardized advantage onto every row."""
    for row in rows:
        row.ppo_eligible = bool(statistics.is_eligible(row.advantage_target))
        row.standardized_advantage = float(statistics.standardize(row.advantage_target))


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def train_order(rows, *, run_id: str, iteration: int, epoch: int) -> list:
    """The shuffled row order of one epoch, from Phase 17's order domain.

    Deterministic from `(run, iteration, epoch)` alone, so a resumed window
    consumes its rows in the order the interrupted one would have.
    """
    order = list(range(len(rows)))
    seed = derive_move_seed(DOMAIN_TRAINING_ORDER, run_id, int(iteration), int(epoch))
    random.Random(seed).shuffle(order)
    return order


def build_arrays(rows) -> dict:
    """Collate one minibatch, keeping the accepted privilege boundary visible.

    `observation` is the only model input in the mapping; the belief label the
    accepted collation carried is simply not built, because the Phase 17
    objective has no belief term to feed.
    """
    if not rows:
        raise Phase17TrainerError("cannot build a minibatch from no rows")
    return {
        "observation": np.stack(
            [np.ascontiguousarray(row.observation, dtype=np.float32) for row in rows]
        ),
        "legal_mask": np.stack([np.asarray(row.legal_mask, dtype=bool) for row in rows]),
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
        "value_row_weight": np.asarray(
            [row.value_row_weight for row in rows], dtype=np.float32
        ),
    }


# ---------------------------------------------------------------------------
# One update
# ---------------------------------------------------------------------------


@dataclass
class MoveUpdate:
    """What one window's update actually did."""

    iteration: int
    learning_rate: float
    entropy_coefficient: float
    statistics: dict
    rows: int = 0
    trained_rows: int = 0
    boundary_rows: int = 0
    terminal_rows: int = 0
    steps: int = 0
    examples: int = 0
    epochs: list = field(default_factory=list)
    kl_beta: float = 0.0
    means: dict = field(default_factory=dict)
    seconds: float = 0.0
    raw_digest_before: str = ""
    raw_digest_after: str = ""
    ema_updates: int = 0
    counters: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "trainer_version": MOVE_TRAINER_VERSION,
            "iteration": int(self.iteration),
            "learning_rate": float(self.learning_rate),
            "entropy_coefficient": float(self.entropy_coefficient),
            "epochs_per_iteration": MOVE_EPOCHS_PER_ITERATION,
            "transitions_harvested": int(self.rows),
            "transitions_trained": int(self.trained_rows),
            "boundary_bootstrapped_rows": int(self.boundary_rows),
            "terminal_rows": int(self.terminal_rows),
            "optimizer_steps": int(self.steps),
            "examples": int(self.examples),
            "kl_beta": float(self.kl_beta),
            "epochs": [dict(entry) for entry in self.epochs],
            "statistics": dict(self.statistics),
            "means": dict(self.means),
            "seconds": float(self.seconds),
            "raw_model_state_digest_before": self.raw_digest_before,
            "raw_model_state_digest_after": self.raw_digest_after,
            "raw_changed": self.raw_digest_before != self.raw_digest_after,
            "ema_updates": int(self.ema_updates),
            "counters": dict(self.counters),
            "belief_loss_weight": 0.0,
        }


class MoveWindowTrainer:
    """The one-epoch PPO/KL learner over one fixed-transition window."""

    def __init__(
        self,
        *,
        run_id: str,
        model,
        optimizer,
        controller: KLController,
        ema: WeightEMA,
        horizon: MoveScheduleHorizon,
        device: str = "cpu",
        minibatch_size: int = MINIBATCH_SIZE,
        epochs: int = MOVE_EPOCHS_PER_ITERATION,
    ) -> None:
        self.run_id = require_run_id(run_id)
        self.model = model
        self.optimizer = optimizer
        self.controller = controller
        self.ema = ema
        self.horizon = horizon
        self.device = torch.device(device)
        self.minibatch_size = int(minibatch_size)
        self.epochs = int(epochs)
        if self.epochs != MOVE_EPOCHS_PER_ITERATION:
            raise Phase17TrainerError(
                f"the Phase 17 move update is {MOVE_EPOCHS_PER_ITERATION} epoch "
                f"per iteration, not {self.epochs}"
            )
        self.global_step = 0
        self.examples_consumed = 0
        self.counters = {
            "non_finite_losses": 0,
            "non_finite_gradients": 0,
            "illegal_targets": 0,
            "kl_vetoes": 0,
            "clip_vetoes": 0,
            "stale_digest_refusals": 0,
        }

    # -- refusals ----------------------------------------------------------

    def assert_rows_current(self, rows, *, cell: CurrentMovePolicy) -> dict:
        """Refuse a window carrying a digest the live cell has never held.

        Immediate stop condition I2, checked at the last point it still can be:
        a stale-weights decision produces a PPO ratio near 1.0 and is otherwise
        invisible in the loss.
        """
        known = set(cell.known_digests())
        counts: dict = {}
        for row in rows:
            counts[row.behavior_model_state_digest] = (
                counts.get(row.behavior_model_state_digest, 0) + 1
            )
        unknown = {d: n for d, n in counts.items() if d not in known}
        if unknown:
            self.counters["stale_digest_refusals"] += 1
            raise Phase17TrainerError(
                "the window carries decisions recorded under model states the "
                f"current-policy cell has never held: {unknown}"
            )
        return {"distinct_model_states": len(counts), "transitions_by_model_state": counts}

    def _check_hard_limits(self, *, iteration: int, epoch: int, mean_kl: float, clip_fraction: float) -> None:
        """The accepted Phase 9 vetoes, with the accepted abort semantics."""
        if mean_kl > KL_HARD_LIMIT:
            self.counters["kl_vetoes"] += 1
            raise Phase17TrainerError(
                f"window {iteration} epoch {epoch}: mean behavior KL {mean_kl:.5f} "
                f"exceeds the hard limit {KL_HARD_LIMIT}"
            )
        if clip_fraction > CLIP_FRACTION_HARD_LIMIT:
            self.counters["clip_vetoes"] += 1
            raise Phase17TrainerError(
                f"window {iteration} epoch {epoch}: clip fraction {clip_fraction:.4f} "
                f"exceeds the hard limit {CLIP_FRACTION_HARD_LIMIT}"
            )

    def _raw_digest(self) -> str:
        from ..phase9_behavior import state_dict_digest

        return state_dict_digest(self.model)

    # -- one window --------------------------------------------------------

    def train_window(
        self,
        rows,
        *,
        iteration: int,
        cell: "CurrentMovePolicy | None" = None,
        may_start_step=None,
    ) -> MoveUpdate:
        """Train on one window's transitions. Returns what happened."""
        if not rows:
            raise Phase17TrainerError(f"window {iteration} produced no rows to train on")
        if not 1 <= int(iteration) <= self.horizon.total_iterations:
            raise Phase17TrainerError(
                f"iteration {iteration} is outside the frozen horizon "
                f"1..{self.horizon.total_iterations}; the horizon is frozen "
                "before launch and never recomputed from production speed"
            )
        if cell is not None:
            self.assert_rows_current(rows, cell=cell)

        rate = self.horizon.learning_rate(int(iteration))
        coefficient = self.horizon.entropy_coefficient(int(iteration))
        for group in self.optimizer.param_groups:
            group["lr"] = rate

        statistics = window_statistics(rows, iteration=int(iteration))
        apply_statistics(rows, statistics)

        update = MoveUpdate(
            iteration=int(iteration),
            learning_rate=rate,
            entropy_coefficient=coefficient,
            statistics=statistics.to_dict(),
            rows=len(rows),
            trained_rows=int(statistics.eligible),
            boundary_rows=sum(
                1 for row in rows if row.target_provenance == PROVENANCE_BOOTSTRAP
            ),
            terminal_rows=sum(
                1 for row in rows if row.target_provenance == PROVENANCE_TERMINAL
            ),
            raw_digest_before=self._raw_digest(),
        )
        collected: dict = {}
        started = time.perf_counter()
        size = self.minibatch_size
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            order = train_order(
                rows, run_id=self.run_id, iteration=int(iteration), epoch=epoch
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
                iteration=int(iteration),
                epoch=epoch,
                mean_kl=float(entry["mean_epoch_kl"]),
                clip_fraction=float(entry["epoch_clip_fraction"]),
            )
            update.epochs.append(entry)

        self.model.eval()
        update.seconds = time.perf_counter() - started
        update.kl_beta = float(self.controller.beta)
        update.means = {
            f"mean_{key}": float(np.mean(values)) for key, values in collected.items()
        }
        update.raw_digest_after = self._raw_digest()
        update.ema_updates = int(self.ema.updates)
        update.counters = dict(self.counters)
        return update

    def _step(self, batch, coefficient: float):
        arrays = build_arrays(batch)
        tensors = {
            name: torch.from_numpy(np.ascontiguousarray(value)).to(self.device)
            for name, value in arrays.items()
        }
        outputs = self.model.forward_observation(tensors["observation"])
        try:
            loss = phase17_batch_loss(
                outputs,
                legal_mask=tensors["legal_mask"],
                sampled_action_model=tensors["sampled_action_model"],
                behavior_action_probability=tensors["behavior_action_probability"],
                behavior_probabilities=tensors["behavior_probabilities"],
                standardized_advantage=tensors["standardized_advantage"],
                ppo_eligible=tensors["ppo_eligible"],
                wdl_target=tensors["wdl_target"],
                value_row_weight=tensors["value_row_weight"],
                kl_beta=float(self.controller.beta),
                entropy_coefficient=float(coefficient),
                boundary_rows=sum(
                    1 for row in batch if row.target_provenance == PROVENANCE_BOOTSTRAP
                ),
            )
        except Phase17LossError:
            self.counters["illegal_targets"] += 1
            raise
        if not loss.all_finite():
            self.counters["non_finite_losses"] += 1
            raise Phase17TrainerError(
                f"non-finite loss at global step {self.global_step + 1}"
            )

        self.optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        pre_clip = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), MOVE_GRADIENT_CLIP_NORM
        )
        if not bool(torch.isfinite(pre_clip)):
            self.counters["non_finite_gradients"] += 1
            raise Phase17TrainerError(
                f"non-finite gradient norm at global step {self.global_step + 1}"
            )
        self.optimizer.step()
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
            "trainer_version": MOVE_TRAINER_VERSION,
            "run_id": self.run_id,
            "global_optimizer_step": int(self.global_step),
            "examples_consumed": int(self.examples_consumed),
            "kl_controller": self.controller.to_dict(),
            "ema": self.ema.to_dict(),
            "counters": dict(self.counters),
            "gradient_clip_norm": MOVE_GRADIENT_CLIP_NORM,
            "minibatch_size": int(self.minibatch_size),
            "epochs_per_iteration": self.epochs,
            "horizon": self.horizon.to_dict(),
        }

    def restore_state(self, payload: dict) -> None:
        if payload.get("run_id") not in (None, self.run_id):
            raise Phase17TrainerError(
                f"trainer state belongs to run {payload.get('run_id')!r}, not "
                f"{self.run_id!r}"
            )
        self.global_step = int(payload["global_optimizer_step"])
        self.examples_consumed = int(payload["examples_consumed"])
        self.controller = KLController.from_dict(payload["kl_controller"])
        self.counters.update(payload.get("counters") or {})


def state_mapping_digest(state: dict) -> str:
    """The accepted `state_dict_digest` algorithm, over a plain mapping.

    `stratego.training.phase9_behavior.state_dict_digest` takes a live module.
    The EMA is a mapping of tensors and has no module to be read through, so
    the same walk -- sorted name, then the float32 shape, then the float32
    bytes -- is applied to the mapping directly. Wrapping the mapping in a
    throwaway module instead would have to mangle the dotted names, and a name
    that survived mangling differently would change the digest silently.
    """
    hasher = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        hasher.update(name.encode())
        array = torch.as_tensor(tensor).detach().to("cpu", torch.float32)
        array = array.contiguous().numpy()
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def assert_ema_never_acted(cell: CurrentMovePolicy, ema: WeightEMA) -> dict:
    """Prove the EMA is not the policy that generated any decision.

    The check the contract's "RAW generates all training data" clause needs:
    if the EMA's state ever equals a digest the cell has held, the two have
    been confused somewhere.
    """
    ema_digest = state_mapping_digest(ema.state_dict())
    acted = set(cell.known_digests())
    return {
        "ema_updates": int(ema.updates),
        "ema_model_state_digest": ema_digest,
        "raw_digests_that_acted": sorted(acted),
        "ema_ever_acted": ema_digest in acted,
        "holds": ema_digest not in acted or int(ema.updates) == 0,
    }


def trainer_semantics() -> dict:
    return {
        "trainer_version": MOVE_TRAINER_VERSION,
        "epochs_per_iteration": MOVE_EPOCHS_PER_ITERATION,
        "minibatch_size": MINIBATCH_SIZE,
        "gradient_clip_norm": MOVE_GRADIENT_CLIP_NORM,
        "advantage_filter": "the accepted per-window tau = max(Q75(|A|), 0.01)",
        "standardization": "the accepted mean/std over the eligible subset",
        "objective": "stratego.training.phase17.move_loss.phase17_batch_loss",
        "kl_controller": "the accepted Phase 9 controller, closed once per epoch",
        "ema": "updated after every optimizer step; evaluation-only, never acts",
        "refusals": [
            "non-finite loss",
            "non-finite gradient norm",
            "mean epoch KL above the accepted hard limit",
            "clip fraction above the accepted hard limit",
            "an iteration outside the frozen horizon",
            "a row whose behavior model-state digest the cell never held",
        ],
    }


__all__ = [
    "MoveUpdate",
    "MoveWindowTrainer",
    "Phase17TrainerError",
    "WindowStatistics",
    "apply_statistics",
    "assert_ema_never_acted",
    "build_arrays",
    "state_mapping_digest",
    "train_order",
    "trainer_semantics",
    "window_statistics",
]
