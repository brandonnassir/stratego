"""Phase 11B Agent 1 training: one small loop, two experiments.

Specification source: `01_AGENT_1_ATTACHED_BELIEF_HEAD.md` ("Use supervised
hidden-rank cross-entropy", "Do not train policy or value", "Record the
learning curve and save the best development checkpoint").

Supervised belief only
----------------------
The loss is `cross_entropy(logits, true_rank)` over hidden pieces and
nothing else. There is no policy term, no value term and no game outcome
anywhere in this module — Phase 11B is supervised belief prediction, not
reinforcement learning from outcomes.

Best-on-development, not last
-----------------------------
Every evaluation writes the full shared metric block, and the best
development cross-entropy is kept in memory as a state dict and written
once at the end. `time_to_best_seconds` is the wall clock at the epoch that
produced it, which is the "time-to-best checkpoint" the leaderboard wants —
not the total, which the run's stopping rule also determines.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn

from .contract import Phase11BError
from .metrics import baseline_probabilities, evaluate
from .seeds import training_seed

#: The trainer identity.
TRAINER_VERSION = "phase11b_attached_trainer_v1"


class Phase11BTrainError(Phase11BError):
    """A candidate could not be trained."""


@dataclass
class TrainConfig:
    """One candidate's complete training configuration.

    Chosen once and recorded, not searched. `01_AGENT_1` forbids a
    hyperparameter sweep, so these values are the experiment's declared
    settings and the report carries them verbatim.
    """

    candidate_id: str
    epochs: int = 24
    batch_size: int = 4096
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    optimizer: str = "adamw"
    schedule: str = "cosine"
    patience: int = 5
    device: str = "cpu"
    eval_batch_size: int = 65536
    max_seconds: float = 1800.0

    def to_dict(self) -> dict:
        return {**asdict(self), "trainer_version": TRAINER_VERSION}


def _optimizer(model: nn.Module, config: TrainConfig, groups=None):
    if config.optimizer != "adamw":  # pragma: no cover - one optimizer by design
        raise Phase11BTrainError(f"unknown optimizer {config.optimizer!r}")
    parameters = groups if groups is not None else model.parameters()
    return torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )


@torch.no_grad()
def predict_probabilities(
    model: nn.Module, features: np.ndarray, *, device: str = "cpu", batch_size: int = 65536
) -> np.ndarray:
    """`float64[M, 12]` raw softmax over a feature matrix, in corpus order."""
    model.eval()
    rows = int(features.shape[0])
    out = np.empty((rows, 12), dtype=np.float64)
    for start in range(0, rows, batch_size):
        stop = min(start + batch_size, rows)
        batch = torch.from_numpy(np.ascontiguousarray(features[start:stop])).to(device)
        # Move first, then widen: float64 does not exist on every backend, and
        # casting on-device silently degrades the result on Metal.
        logits = model(batch).detach().cpu().to(torch.float64)
        out[start:stop] = torch.softmax(logits, dim=1).numpy()
    return out


def train_attached_head(
    model: nn.Module,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    dev_features: np.ndarray,
    dev_data: dict,
    config: TrainConfig,
    *,
    progress=None,
) -> dict:
    """Train one attached head and return its run record.

    The frozen features are a fixed matrix, so an "epoch" is one shuffled
    pass over `train_features`. Evaluation happens once per epoch on the
    common development positions, through the same shared metric block
    every other Phase 11B candidate reports.
    """
    device = torch.device(config.device)
    model = model.to(device)
    optimizer = _optimizer(model, config)
    rows = int(train_features.shape[0])
    steps_per_epoch = max(1, (rows + config.batch_size - 1) // config.batch_size)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs * steps_per_epoch
        )
        if config.schedule == "cosine"
        else None
    )
    generator = np.random.default_rng(training_seed(config.candidate_id, "shuffle"))
    features = torch.from_numpy(np.ascontiguousarray(train_features)).to(device)
    labels = torch.from_numpy(np.ascontiguousarray(train_labels, dtype=np.int64)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    dev_baseline = baseline_probabilities(dev_data)

    curve: list[dict] = []
    best = {"dev_ce": float("inf"), "epoch": -1, "state": None, "seconds": 0.0, "metrics": None}
    started = time.perf_counter()
    stopped = "epochs_exhausted"
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        order = torch.from_numpy(generator.permutation(rows)).to(device)
        total = 0.0
        for start in range(0, rows, config.batch_size):
            index = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features[index]), labels[index])
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float(loss.detach()) * int(index.numel())
        train_loss = total / rows

        probabilities = predict_probabilities(
            model, dev_features, device=config.device, batch_size=config.eval_batch_size
        )
        metrics = evaluate(
            probabilities, dev_data, baseline=dev_baseline, bootstrap_resamples=200
        )
        elapsed = time.perf_counter() - started
        curve.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_ce": metrics["ce"],
                "dev_r_ce": metrics["r_ce"],
                "dev_top1": metrics["top1"],
                "seconds": round(elapsed, 3),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if progress is not None:
            progress(config.candidate_id, curve[-1])
        if metrics["ce"] < best["dev_ce"] - 1e-9:
            best = {
                "dev_ce": metrics["ce"],
                "epoch": epoch,
                "state": copy.deepcopy(
                    {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                ),
                "seconds": round(elapsed, 3),
                "metrics": metrics,
            }
        elif epoch - best["epoch"] >= int(config.patience):
            stopped = "patience"
            break
        if elapsed > float(config.max_seconds):
            stopped = "time_budget"
            break

    if best["state"] is None:  # pragma: no cover - the first epoch always improves
        raise Phase11BTrainError(f"{config.candidate_id} produced no checkpoint")
    model.load_state_dict(best["state"])
    model.eval()
    return {
        "candidate_id": config.candidate_id,
        "config": config.to_dict(),
        "curve": curve,
        "epochs_run": len(curve),
        # Patience can fire on the *last* scheduled epoch, which is not an
        # early stop. The label reports what actually happened.
        "stopped_because": (
            "epochs_exhausted" if len(curve) >= int(config.epochs) else stopped
        ),
        "best_epoch": best["epoch"],
        "best_state": best["state"],
        "time_to_best_seconds": best["seconds"],
        "training_seconds": round(time.perf_counter() - started, 3),
        "train_rows": rows,
        "dev_metrics": best["metrics"],
    }


# ---------------------------------------------------------------------------
# Experiment 1C — the last C1 block, unfrozen
# ---------------------------------------------------------------------------


def _sample_batches(data: dict, rows: np.ndarray, batch_size: int):
    """Yield `(sample rows, token rows, token squares, labels)` per batch.

    1C's loss is still exactly per hidden piece, so a batch of *positions*
    carries an index pair into its own `[B, 100, 128]` block rather than a
    padded square mask.
    """
    offsets = np.asarray(data["piece_offset"], dtype=np.int64)
    squares = np.asarray(data["perspective_square"], dtype=np.int64)
    labels = np.asarray(data["true_rank"], dtype=np.int64)
    for start in range(0, rows.size, batch_size):
        block = rows[start : start + batch_size]
        counts = offsets[block + 1] - offsets[block]
        token_rows = np.repeat(np.arange(block.size, dtype=np.int64), counts)
        piece_index = np.concatenate(
            [np.arange(offsets[row], offsets[row + 1], dtype=np.int64) for row in block]
        ) if block.size else np.zeros(0, dtype=np.int64)
        yield block, token_rows, squares[piece_index], labels[piece_index]


@torch.no_grad()
def predict_probabilities_1c(
    model: nn.Module, tokens: np.ndarray, data: dict, *, device: str = "cpu", batch_size: int = 256
) -> np.ndarray:
    """`float64[M, 12]` for a 1C model, in the corpus's own piece order."""
    model.eval()
    offsets = np.asarray(data["piece_offset"], dtype=np.int64)
    out = np.empty((int(data["pieces"]), 12), dtype=np.float64)
    rows = np.arange(int(data["samples"]), dtype=np.int64)
    for block, token_rows, token_squares, _labels in _sample_batches(data, rows, batch_size):
        batch = torch.from_numpy(np.array(tokens[block], dtype=np.float32, copy=True)).to(device)
        gather = (
            torch.from_numpy(token_rows).to(device),
            torch.from_numpy(token_squares).to(device),
        )
        logits = model(batch, gather).detach().cpu().to(torch.float64)
        probabilities = torch.softmax(logits, dim=1).numpy()
        cursor = 0
        for row in block:
            width = int(offsets[row + 1] - offsets[row])
            out[offsets[row] : offsets[row + 1]] = probabilities[cursor : cursor + width]
            cursor += width
    return out


def train_final_block(
    model: nn.Module,
    train_tokens: np.ndarray,
    train_data: dict,
    dev_tokens: np.ndarray,
    dev_data: dict,
    config: TrainConfig,
    *,
    block_learning_rate: float = 1.0e-4,
    progress=None,
) -> dict:
    """Train the unfrozen last C1 block together with the larger head.

    Two parameter groups, and the block's learning rate is an order of
    magnitude smaller than the head's: the block starts from accepted
    weights that already work, the head starts from noise.
    """
    device = torch.device(config.device)
    model = model.to(device)
    groups = [
        {
            "params": list(model.block.parameters()) + list(model.encoder_norm.parameters()),
            "lr": float(block_learning_rate),
        },
        {"params": list(model.head.parameters()), "lr": float(config.learning_rate)},
    ]
    optimizer = _optimizer(model, config, groups)
    samples = int(train_data["samples"])
    steps_per_epoch = max(1, (samples + config.batch_size - 1) // config.batch_size)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs * steps_per_epoch
        )
        if config.schedule == "cosine"
        else None
    )
    generator = np.random.default_rng(training_seed(config.candidate_id, "shuffle"))
    loss_fn = nn.CrossEntropyLoss()
    dev_baseline = baseline_probabilities(dev_data)

    curve: list[dict] = []
    best = {"dev_ce": float("inf"), "epoch": -1, "state": None, "seconds": 0.0, "metrics": None}
    started = time.perf_counter()
    stopped = "epochs_exhausted"
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        order = generator.permutation(samples).astype(np.int64)
        total = 0.0
        pieces = 0
        for block, token_rows, token_squares, labels in _sample_batches(
            train_data, order, config.batch_size
        ):
            # `block` is in shuffled order and `token_rows` indexes into it,
            # so the tokens are gathered in exactly that order.
            batch = torch.from_numpy(
                np.array(train_tokens[block], dtype=np.float32, copy=True)
            ).to(device)
            gather = (
                torch.from_numpy(token_rows).to(device),
                torch.from_numpy(token_squares).to(device),
            )
            target = torch.from_numpy(labels).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch, gather), target)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float(loss.detach()) * int(target.numel())
            pieces += int(target.numel())
        train_loss = total / max(pieces, 1)

        probabilities = predict_probabilities_1c(
            model, dev_tokens, dev_data, device=config.device
        )
        metrics = evaluate(
            probabilities, dev_data, baseline=dev_baseline, bootstrap_resamples=200
        )
        elapsed = time.perf_counter() - started
        curve.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_ce": metrics["ce"],
                "dev_r_ce": metrics["r_ce"],
                "dev_top1": metrics["top1"],
                "seconds": round(elapsed, 3),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if progress is not None:
            progress(config.candidate_id, curve[-1])
        if metrics["ce"] < best["dev_ce"] - 1e-9:
            best = {
                "dev_ce": metrics["ce"],
                "epoch": epoch,
                "state": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                },
                "seconds": round(elapsed, 3),
                "metrics": metrics,
            }
        elif epoch - best["epoch"] >= int(config.patience):
            stopped = "patience"
            break
        if elapsed > float(config.max_seconds):
            stopped = "time_budget"
            break

    if best["state"] is None:  # pragma: no cover - the first epoch always improves
        raise Phase11BTrainError(f"{config.candidate_id} produced no checkpoint")
    model.load_state_dict(best["state"])
    model.eval()
    return {
        "candidate_id": config.candidate_id,
        "config": {**config.to_dict(), "block_learning_rate": float(block_learning_rate)},
        "curve": curve,
        "epochs_run": len(curve),
        "stopped_because": (
            "epochs_exhausted" if len(curve) >= int(config.epochs) else stopped
        ),
        "best_epoch": best["epoch"],
        "best_state": best["state"],
        "time_to_best_seconds": best["seconds"],
        "training_seconds": round(time.perf_counter() - started, 3),
        "train_rows": int(train_data["pieces"]),
        "dev_metrics": best["metrics"],
    }


__all__ = [
    "TRAINER_VERSION",
    "Phase11BTrainError",
    "TrainConfig",
    "predict_probabilities",
    "predict_probabilities_1c",
    "train_final_block",
    "train_attached_head",
]
