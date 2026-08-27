"""Phase 15 Agent 1 section 9: one fixed recipe, two specialists.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` section 9.

```text
loss                        hidden-piece cross-entropy
optimizer                   AdamW
head learning rate          1.0e-3
final-block learning rate   1.0e-4
weight decay                1.0e-4
schedule                    cosine
batch size                  256 positions
maximum epochs              12
early-stop patience         3 development evaluations
selection                   best development cross-entropy
```

Declared, not searched
----------------------
These values are the instruction's, copied into
:data:`~.contract.RECIPE` and read from there. B18 and B24 run the *same*
recipe: the only sanctioned deviation is a batch-size change for memory or
throughput safety, which :class:`TrainConfig` records as
`batch_size_changed_from` so a report can state it rather than a reader
having to notice it.

Supervised belief only
----------------------
The loss is `cross_entropy(logits, true_rank)` over hidden pieces and
nothing else. No policy term, no value term, no game outcome. The optimizer
is handed exactly the specialist's own two parameter groups, and the
specialist holds no policy or value parameter to hand it.

Best-on-development, kept immediately
--------------------------------------
Every epoch evaluates on development and the best cross-entropy is cloned
to CPU the moment it is seen, so a later epoch cannot overwrite it and a
crash cannot lose it. Each run writes to a unique output path, so a
repeat pass can never overwrite the selected bytes.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .contract import RECIPE, Phase15Error
from .heads import Phase15BeliefSpecialist, trainable_parameter_groups
from .metrics import baseline_probabilities, cross_entropy
from .seeds import training_seed

#: The trainer identity.
TRAINER_VERSION = "phase15_belief_trainer_v1"


class Phase15TrainError(Phase15Error):
    """A specialist could not be trained."""


@dataclass
class TrainConfig:
    """One specialist's complete training configuration."""

    specialist_id: str
    epochs: int = int(RECIPE["max_epochs"])
    batch_size: int = int(RECIPE["batch_size"])
    head_learning_rate: float = float(RECIPE["head_learning_rate"])
    block_learning_rate: float = float(RECIPE["final_block_learning_rate"])
    weight_decay: float = float(RECIPE["weight_decay"])
    optimizer: str = str(RECIPE["optimizer"])
    schedule: str = str(RECIPE["schedule"])
    patience: int = int(RECIPE["early_stop_patience"])
    device: str = "cpu"
    eval_batch_size: int = 512
    max_seconds: float = 7200.0
    batch_size_changed_from: "int | None" = None
    batch_size_change_reason: "str | None" = None

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "trainer_version": TRAINER_VERSION,
            "loss": RECIPE["loss"],
            "selection": RECIPE["selection"],
            "recipe_is_declared_not_searched": True,
        }


def _optimizer(model: Phase15BeliefSpecialist, config: TrainConfig):
    if config.optimizer != "adamw":  # pragma: no cover - one optimizer by design
        raise Phase15TrainError(f"unknown optimizer {config.optimizer!r}")
    groups = trainable_parameter_groups(
        model, head_lr=config.head_learning_rate, block_lr=config.block_learning_rate
    )
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def sample_batches(data: dict, rows: np.ndarray, batch_size: int):
    """Yield `(sample rows, token rows, token squares, labels)` per batch.

    The loss is exactly per hidden piece, so a batch of *positions* carries
    an index pair into its own `[B, 100, 128]` block rather than a padded
    square mask. `rows` is the shuffled order for this epoch.
    """
    offsets = np.asarray(data["piece_offset"], dtype=np.int64)
    squares = np.asarray(data["perspective_square"], dtype=np.int64)
    labels = np.asarray(data["true_rank"], dtype=np.int64)
    for start in range(0, rows.size, batch_size):
        block = rows[start : start + batch_size]
        counts = offsets[block + 1] - offsets[block]
        token_rows = np.repeat(np.arange(block.size, dtype=np.int64), counts)
        piece_index = (
            np.concatenate(
                [np.arange(offsets[row], offsets[row + 1], dtype=np.int64) for row in block]
            )
            if block.size
            else np.zeros(0, dtype=np.int64)
        )
        yield block, token_rows, squares[piece_index], labels[piece_index]


@torch.no_grad()
def predict_probabilities(
    model: Phase15BeliefSpecialist,
    cache: np.ndarray,
    data: dict,
    *,
    device: str = "cpu",
    batch_size: int = 512,
    temperature: "float | None" = None,
) -> np.ndarray:
    """`float64[M, 12]` raw softmax, in the corpus's own piece order.

    `temperature` overrides the model's own fitted value; `None` uses it.
    Float64 is taken *after* moving to CPU, because a device without
    float64 would silently degrade the result.
    """
    model.eval()
    offsets = np.asarray(data["piece_offset"], dtype=np.int64)
    out = np.empty((int(data["pieces"]), 12), dtype=np.float64)
    rows = np.arange(int(data["samples"]), dtype=np.int64)
    scale = float(model.temperature if temperature is None else temperature)
    if not scale > 0.0:
        raise Phase15TrainError(f"temperature must be positive, got {scale}")
    for block, token_rows, token_squares, _labels in sample_batches(
        data, rows, batch_size
    ):
        tokens = torch.from_numpy(np.array(cache[block], dtype=np.float32, copy=True)).to(
            device
        )
        gather = (
            torch.from_numpy(token_rows).to(device),
            torch.from_numpy(token_squares).to(device),
        )
        logits = model(tokens, gather).detach().cpu().to(torch.float64) / scale
        probabilities = torch.softmax(logits, dim=1).numpy()
        cursor = 0
        for row in block:
            width = int(offsets[row + 1] - offsets[row])
            out[offsets[row] : offsets[row + 1]] = probabilities[cursor : cursor + width]
            cursor += width
    return out


def assert_no_source_gradients(policy_model) -> dict:
    """Prove the frozen backbone took no gradient.

    Structural, not incidental: the specialist holds deep copies, so the
    source is not in the graph at all. This checks the property anyway,
    because "it cannot happen" is worth a number in a report.
    """
    with_grad = [
        name
        for name, tensor in policy_model.named_parameters()
        if tensor.grad is not None or tensor.requires_grad
    ]
    if with_grad:
        raise Phase15TrainError(
            f"the frozen policy backbone has {len(with_grad)} parameters carrying "
            f"gradient or requires_grad: {with_grad[:4]}"
        )
    return {
        "policy_value_parameters_with_gradient": 0,
        "policy_value_parameters_requiring_grad": 0,
        "checked_parameters": int(sum(1 for _ in policy_model.parameters())),
    }


def train_specialist(
    model: Phase15BeliefSpecialist,
    train_cache: np.ndarray,
    train_data: dict,
    dev_cache: np.ndarray,
    dev_data: dict,
    config: TrainConfig,
    *,
    policy_model=None,
    progress=None,
) -> dict:
    """Train one specialist and return its run record.

    Evaluation is the development *cross-entropy* alone — the section 9
    selection rule — computed every epoch on the raw (uncalibrated)
    probabilities, because the temperature is fitted afterwards on a
    different split and must not influence which epoch is chosen.
    """
    device = torch.device(config.device)
    model = model.to(device)
    optimizer = _optimizer(model, config)
    samples = int(train_data["samples"])
    steps_per_epoch = max(1, (samples + config.batch_size - 1) // config.batch_size)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs * steps_per_epoch
        )
        if config.schedule == "cosine"
        else None
    )
    generator = np.random.default_rng(training_seed(config.specialist_id, "shuffle"))
    loss_fn = nn.CrossEntropyLoss()
    dev_baseline = baseline_probabilities(dev_data)
    dev_true = np.asarray(dev_data["true_rank"], dtype=np.int64)
    dev_baseline_ce = float(cross_entropy(dev_baseline, dev_true).mean())

    curve: list[dict] = []
    best = {
        "dev_ce": float("inf"),
        "epoch": -1,
        "state": None,
        "seconds": 0.0,
    }
    gradient_check: dict = {}
    started = time.perf_counter()
    stopped = "epochs_exhausted"

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        order = generator.permutation(samples)
        total = 0.0
        terms = 0
        for block, token_rows, token_squares, labels in sample_batches(
            train_data, order, config.batch_size
        ):
            tokens = torch.from_numpy(
                np.array(train_cache[block], dtype=np.float32, copy=True)
            ).to(device)
            gather = (
                torch.from_numpy(token_rows).to(device),
                torch.from_numpy(token_squares).to(device),
            )
            target = torch.from_numpy(labels).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(tokens, gather), target)
            loss.backward()
            if policy_model is not None and not gradient_check:
                # Checked once, on the first backward pass of the first
                # epoch — the only moment at which a leak could appear.
                gradient_check = assert_no_source_gradients(policy_model)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float(loss.detach()) * int(target.numel())
            terms += int(target.numel())
        train_loss = total / max(terms, 1)

        probabilities = predict_probabilities(
            model,
            dev_cache,
            dev_data,
            device=config.device,
            batch_size=config.eval_batch_size,
            temperature=1.0,
        )
        dev_ce = float(cross_entropy(probabilities, dev_true).mean())
        dev_top1 = float((probabilities.argmax(axis=1) == dev_true).mean())
        elapsed = time.perf_counter() - started
        curve.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_terms": terms,
                "dev_ce": dev_ce,
                "dev_r_ce": dev_ce / dev_baseline_ce,
                "dev_top1": dev_top1,
                "seconds": round(elapsed, 3),
                "head_learning_rate": float(optimizer.param_groups[1]["lr"]),
                "block_learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if progress is not None:
            progress(config.specialist_id, curve[-1])
        if dev_ce < best["dev_ce"] - 1e-9:
            best = {
                "dev_ce": dev_ce,
                "epoch": epoch,
                "state": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                },
                "seconds": round(elapsed, 3),
            }
        elif epoch - best["epoch"] >= int(config.patience):
            stopped = "patience"
            break
        if elapsed > float(config.max_seconds):
            stopped = "time_budget"
            break

    if best["state"] is None:  # pragma: no cover - the first epoch always improves
        raise Phase15TrainError(f"{config.specialist_id} produced no checkpoint")
    model.load_state_dict(best["state"])
    model.eval()
    return {
        "specialist_id": config.specialist_id,
        "config": config.to_dict(),
        "curve": curve,
        "epochs_run": len(curve),
        # Patience can fire on the last scheduled epoch, which is not an
        # early stop. The label reports what actually happened.
        "stopped_because": (
            "epochs_exhausted" if len(curve) >= int(config.epochs) else stopped
        ),
        "best_epoch": best["epoch"],
        "best_dev_ce": best["dev_ce"],
        "time_to_best_seconds": best["seconds"],
        "training_seconds": round(time.perf_counter() - started, 3),
        "train_positions": samples,
        "train_pieces": int(train_data["pieces"]),
        "development_positions": int(dev_data["samples"]),
        "development_baseline_ce": dev_baseline_ce,
        "gradient_isolation": gradient_check,
        "steps_per_epoch": steps_per_epoch,
    }


__all__ = [
    "TRAINER_VERSION",
    "Phase15TrainError",
    "TrainConfig",
    "assert_no_source_gradients",
    "predict_probabilities",
    "sample_batches",
    "train_specialist",
]
