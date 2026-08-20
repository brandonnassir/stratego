"""Phase 11B Agent 2 training: one CNN over the raw observation corpus.

Specification source: `02_AGENT_2_RAW_OBSERVATION_CNN.md` ("Training",
"Time Budget", "Evaluation").

Why this is not Agent 1's trainer
----------------------------------
Agent 1's candidates read a **frozen** C1 feature, so its trainer could
cache the encoder's output once and treat an epoch as a pass over a fixed
`[M, 128]` matrix. Agent 2 learns its own representation, so the
observations themselves are the training tensor and every epoch is a real
pass over `[N, 127, 10, 10]`. The batch unit is therefore a *position*, not
a piece, and the loss gathers the supervised squares out of each position's
logit field.

The gather is the accepted one
-------------------------------
`_sample_batches` — the same private helper Agent 1's Experiment 1C uses —
produces the `(positions, token rows, token squares, labels)` tuple. Reusing
it rather than re-deriving the slicing is deliberate: it guarantees Agent 2
supervises exactly the pieces, in exactly the order, that every other
Phase 11B candidate was scored on.

Supervised belief only
-----------------------
The loss is `cross_entropy(logits, true_rank)` over hidden pieces and
nothing else. No policy term, no value term, no game outcome. The model's
only input is the public observation.

Device policy
-------------
`02_AGENT_2` allows MPS "if the implementation is stable and materially
faster", so the device is measured rather than assumed: the harness runs a
throughput pilot on each available backend and trains on the winner. Two
consequences are handled here rather than at the call site — probabilities
are widened to float64 only **after** returning to the CPU, because Metal
has no float64, and the corpus observations are staged as one resident
tensor on the training device, because a 1.4 GB tensor that fits in unified
memory should be copied once rather than once per step.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .contract import Phase11BError, RANK_COUNT
from .metrics import baseline_probabilities, cross_entropy, evaluate
from .raw_cnn import CANDIDATE_2
from .seeds import training_seed
from .train import _sample_batches

#: The trainer identity.
RAW_TRAINER_VERSION = "phase11b_raw_cnn_trainer_v1"


class Phase11BRawTrainError(Phase11BError):
    """The raw-observation candidate could not be trained."""


@dataclass
class RawTrainConfig:
    """The complete declared configuration of one Agent 2 run.

    The defaults are the configuration Agent 2 declared first: Agent 1's own
    optimizer family (AdamW, cosine, `1e-3`, weight decay `1e-4`) and no
    dropout, so that the two experiments would differ in architecture rather
    than in tuning effort. `02_AGENT_2` instructs "choose one sensible
    configuration, no architecture sweep", and this was that choice.

    That run overfit from its second epoch — training cross-entropy
    collapsing while development cross-entropy rose monotonically — which is
    a statement about 3.9M parameters against 26,898 correlated positions
    rather than about the architecture. `run_id` exists because exactly one
    corrective configuration was then declared and run, and both are
    reported: see `DECLARED_RUNS` in `scripts/run_phase11b_agent02.py`. Two
    declared configurations is not a sweep, and the report says which is
    which.

    `epochs` is set by the harness from the *measured pilot throughput*
    before any development metric exists — a budget decision, not a tuned
    one.
    """

    candidate_id: str = CANDIDATE_2
    run_id: str = "run1_declared"
    epochs: int = 24
    batch_positions: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    block_dropout: float = 0.0
    readout_dropout: float = 0.0
    optimizer: str = "adamw"
    schedule: str = "cosine"
    gradient_clip: float = 1.0
    patience: int = 5
    device: str = "cpu"
    #: Evaluations per epoch. `02_AGENT_2` asks for the best development
    #: checkpoint, and both Agent 2 runs reach their development optimum
    #: *inside* their first epoch, so a once-per-epoch probe cannot find it.
    #: Sub-epoch probes score development cross-entropy only; the full shared
    #: metric block is computed once at the end, on the state that won.
    evaluations_per_epoch: int = 8
    eval_batch_positions: int = 512
    max_seconds: float = 3600.0
    stage_observations_on_device: bool = True

    def to_dict(self) -> dict:
        return {**asdict(self), "trainer_version": RAW_TRAINER_VERSION}


def stage_observations(
    data: dict, device: "str | torch.device", *, on_device: bool = True
) -> torch.Tensor:
    """The split's observations as one float32 tensor, ready to index.

    The corpus stores observations as a read-only memmap; `np.array(...,
    copy=True)` materializes them, which torch requires for a writable
    buffer. `on_device=False` keeps the tensor in host memory and leaves the
    per-batch transfer to the training loop — the fallback when a backend
    cannot hold the whole split.
    """
    array = np.array(data["observations"], dtype=np.float32, copy=True)
    tensor = torch.from_numpy(array)
    if on_device:
        tensor = tensor.to(torch.device(device))
    return tensor


def subset_split(data: dict, rows: "np.ndarray") -> dict:
    """A stored split restricted to `rows`, as a standalone split view.

    Used by the corpus-size diagnostic. The selection is by *sample row*, and
    the caller chooses those from whole games, because positions inside one
    game share an army and slicing positions rather than games would make a
    smaller corpus look more informative than it is.
    """
    rows = np.asarray(rows, dtype=np.int64)
    offsets = np.asarray(data["piece_offset"], dtype=np.int64)
    counts = offsets[rows + 1] - offsets[rows]
    piece_index = (
        np.concatenate([np.arange(offsets[row], offsets[row + 1]) for row in rows])
        if rows.size
        else np.zeros(0, dtype=np.int64)
    )
    view = dict(data)
    view["samples"] = int(rows.size)
    view["pieces"] = int(piece_index.size)
    view["piece_offset"] = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
    )
    view["observations"] = np.asarray(data["observations"])[rows]
    for name in (
        "perspective_square",
        "piece_slot",
        "piece_square",
        "piece_moved",
        "legal_rank_mask",
        "true_rank",
    ):
        if name in data:
            view[name] = np.asarray(data[name])[piece_index]
    for name in (
        "game_ordinal",
        "stratum",
        "setup_source",
        "observer_color",
        "decision_index",
        "total_moves",
        "remaining_counts",
        "target_mask",
    ):
        if name in data:
            view[name] = np.asarray(data[name])[rows]
    return view


@torch.no_grad()
def predict_probabilities_raw(
    model: nn.Module,
    observations: torch.Tensor,
    data: dict,
    *,
    device: str = "cpu",
    batch_positions: int = 512,
) -> np.ndarray:
    """`float64[M, 12]` raw softmax per hidden piece, in corpus piece order."""
    model.eval()
    target = torch.device(device)
    offsets = np.asarray(data["piece_offset"], dtype=np.int64)
    out = np.empty((int(data["pieces"]), RANK_COUNT), dtype=np.float64)
    rows = np.arange(int(data["samples"]), dtype=np.int64)
    for block, token_rows, token_squares, _labels in _sample_batches(
        data, rows, batch_positions
    ):
        batch = observations[torch.from_numpy(block).to(observations.device)].to(target)
        gather = (
            torch.from_numpy(token_rows).to(target),
            torch.from_numpy(token_squares).to(target),
        )
        # Move first, then widen: float64 does not exist on every backend,
        # and casting on-device silently degrades the result on Metal.
        logits = model.logits_at(batch, *gather).detach().cpu().to(torch.float64)
        probabilities = torch.softmax(logits, dim=1).numpy()
        cursor = 0
        for row in block:
            width = int(offsets[row + 1] - offsets[row])
            out[offsets[row] : offsets[row + 1]] = probabilities[cursor : cursor + width]
            cursor += width
    return out


def throughput_pilot(
    model: nn.Module,
    data: dict,
    *,
    device: str,
    batch_positions: int = 256,
    steps: int = 6,
    warmup: int = 2,
) -> dict:
    """A tiny sanity/throughput probe: does this backend work, and how fast?

    `02_AGENT_2` requires a pilot before the run. It answers three
    questions — the forward/backward path is numerically alive, the device
    is usable, and one epoch over the real corpus would cost roughly this
    long — without training anything that is kept.
    """
    target = torch.device(device)
    probe = copy.deepcopy(model).to(target)
    probe.train()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1.0e-4)
    loss_fn = nn.CrossEntropyLoss()
    rows = np.arange(min(int(data["samples"]), batch_positions * (steps + warmup)), dtype=np.int64)
    observations = stage_observations(
        {"observations": data["observations"][: rows.size]}, target, on_device=False
    )

    losses: list[float] = []
    pieces = 0
    # Assigned again at the end of the warm-up; initialized here so a split
    # too small to warm up still produces a (meaningless but finite) time.
    started = time.perf_counter()
    for index, (block, token_rows, token_squares, labels) in enumerate(
        _sample_batches(data, rows, batch_positions)
    ):
        if index >= steps + warmup:
            break
        if index == warmup:
            _synchronize(target)
            started = time.perf_counter()
        batch = observations[torch.from_numpy(block)].to(target)
        gather = (
            torch.from_numpy(token_rows).to(target),
            torch.from_numpy(token_squares).to(target),
        )
        target_ranks = torch.from_numpy(labels).to(target)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(probe.logits_at(batch, *gather), target_ranks)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if not np.isfinite(value):
            raise Phase11BRawTrainError(f"pilot loss on {device} is {value}")
        if index >= warmup:
            losses.append(value)
            pieces += int(target_ranks.numel())
    _synchronize(target)
    elapsed = time.perf_counter() - started
    measured = max(len(losses), 1)
    per_step = elapsed / measured
    positions = measured * batch_positions
    return {
        "device": str(device),
        "batch_positions": int(batch_positions),
        "steps_measured": measured,
        "seconds_per_step": round(per_step, 4),
        "positions_per_second": round(positions / max(elapsed, 1e-9), 1),
        "pieces_per_second": round(pieces / max(elapsed, 1e-9), 1),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "finite": True,
    }


def _synchronize(device: torch.device) -> None:
    """Make an asynchronous backend's queued work real before timing it."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":  # pragma: no cover - no CUDA on this host
        torch.cuda.synchronize()


def _optimizer(model: nn.Module, config: RawTrainConfig):
    if config.optimizer != "adamw":  # pragma: no cover - one optimizer by design
        raise Phase11BRawTrainError(f"unknown optimizer {config.optimizer!r}")
    return torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def train_raw_cnn(
    model: nn.Module,
    train_data: dict,
    dev_data: dict,
    config: RawTrainConfig,
    *,
    progress=None,
) -> dict:
    """Train the one Agent 2 architecture and return its run record.

    An epoch is one shuffled pass over the training positions. Development
    cross-entropy is probed `evaluations_per_epoch` times *within* each
    epoch as well as at its boundary, and the weights with the lowest
    development cross-entropy of the whole run are the checkpoint — this
    candidate's optimum arrives a fraction of an epoch in, so an
    epoch-granular probe would miss it by more than any other quantity this
    experiment measures. The probes are read-only: same batches, same order,
    same optimizer state, whether they run or not.

    The full shared Phase 11B metric block is computed once at the end, on
    the weights that won, so the reported metrics come from the same
    `metrics.evaluate` every other candidate in the sprint reports.
    """
    device = torch.device(config.device)
    model = model.to(device)
    optimizer = _optimizer(model, config)
    samples = int(train_data["samples"])
    steps_per_epoch = max(1, (samples + config.batch_positions - 1) // config.batch_positions)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(config.epochs) * steps_per_epoch
        )
        if config.schedule == "cosine"
        else None
    )
    generator = np.random.default_rng(
        training_seed(config.candidate_id, f"shuffle-{config.run_id}")
    )
    loss_fn = nn.CrossEntropyLoss()
    dev_baseline = baseline_probabilities(dev_data)
    baseline_ce = float(
        cross_entropy(dev_baseline, np.asarray(dev_data["true_rank"], dtype=np.int64)).mean()
    )

    staged = bool(config.stage_observations_on_device)
    train_observations = stage_observations(train_data, device, on_device=staged)
    dev_observations = stage_observations(dev_data, device, on_device=staged)
    dev_true = np.asarray(dev_data["true_rank"], dtype=np.int64)

    def dev_cross_entropy() -> tuple:
        """`(CE, top-1)` on the development pieces. The cheap probe.

        Deliberately not the full metric block: the bootstrap dominates that
        block's cost and contributes nothing to choosing a checkpoint.
        """
        probabilities = predict_probabilities_raw(
            model,
            dev_observations,
            dev_data,
            device=config.device,
            batch_positions=config.eval_batch_positions,
        )
        model.train()
        return (
            float(cross_entropy(probabilities, dev_true).mean()),
            float((probabilities.argmax(axis=1) == dev_true).mean()),
        )

    curve: list[dict] = []
    best = {"dev_ce": float("inf"), "epoch": -1, "step": 0, "state": None, "seconds": 0.0}
    started = time.perf_counter()
    stopped = "epochs_exhausted"
    probe_every = max(1, steps_per_epoch // max(int(config.evaluations_per_epoch), 1))
    global_step = 0

    def consider(epoch: int, train_loss: float, sub_epoch: bool) -> None:
        """Score the current weights and keep them if they are the best yet."""
        nonlocal best
        dev_ce, dev_top1 = dev_cross_entropy()
        elapsed = time.perf_counter() - started
        curve.append(
            {
                "epoch": epoch,
                "step": global_step,
                "epoch_fraction": round(global_step / steps_per_epoch, 4),
                "sub_epoch": sub_epoch,
                "train_loss": train_loss,
                "dev_ce": dev_ce,
                "dev_r_ce": dev_ce / baseline_ce,
                "dev_top1": dev_top1,
                "seconds": round(elapsed, 3),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if progress is not None and not sub_epoch:
            progress(config.candidate_id, curve[-1])
        if dev_ce < best["dev_ce"] - 1e-9:
            best = {
                "dev_ce": dev_ce,
                "epoch": epoch,
                "step": global_step,
                "state": {
                    name: tensor.detach().to("cpu").clone()
                    for name, tensor in model.state_dict().items()
                },
                "seconds": round(elapsed, 3),
            }

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        order = generator.permutation(samples).astype(np.int64)
        total = 0.0
        pieces = 0
        probe_seconds = 0.0
        epoch_started = time.perf_counter()
        for block, token_rows, token_squares, labels in _sample_batches(
            train_data, order, config.batch_positions
        ):
            # `block` is in shuffled order and `token_rows` indexes into it,
            # so the gathered squares follow the batch's own row order.
            batch = train_observations[
                torch.from_numpy(block).to(train_observations.device)
            ].to(device)
            gather = (
                torch.from_numpy(token_rows).to(device),
                torch.from_numpy(token_squares).to(device),
            )
            targets = torch.from_numpy(labels).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model.logits_at(batch, *gather), targets)
            loss.backward()
            if config.gradient_clip:
                nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float(loss.detach()) * int(targets.numel())
            pieces += int(targets.numel())
            global_step += 1
            if global_step % probe_every == 0 and global_step < epoch * steps_per_epoch:
                probe_started = time.perf_counter()
                consider(epoch, total / max(pieces, 1), sub_epoch=True)
                probe_seconds += time.perf_counter() - probe_started
        train_loss = total / max(pieces, 1)
        _synchronize(device)
        # Pure optimizer time: the sub-epoch probes ran inside this window and
        # are subtracted, so the number means "what did the training cost".
        train_seconds = time.perf_counter() - epoch_started - probe_seconds
        consider(epoch, train_loss, sub_epoch=False)
        curve[-1]["epoch_train_seconds"] = round(train_seconds, 3)

        elapsed = time.perf_counter() - started
        if epoch - best["epoch"] >= int(config.patience):
            stopped = "patience"
            break
        if elapsed > float(config.max_seconds):
            stopped = "time_budget"
            break

    if best["state"] is None:  # pragma: no cover - the first probe always improves
        raise Phase11BRawTrainError(f"{config.candidate_id} produced no checkpoint")
    model.load_state_dict(best["state"])
    model.eval()
    # The full shared metric block is computed once, on the weights that won.
    best_metrics = evaluate(
        predict_probabilities_raw(
            model,
            dev_observations,
            dev_data,
            device=config.device,
            batch_positions=config.eval_batch_positions,
        ),
        dev_data,
        baseline=dev_baseline,
        bootstrap_resamples=200,
    )
    # `curve` carries one row per *probe*, several per epoch, so the epoch
    # count is the number of epoch-boundary rows and not the curve's length.
    epochs_completed = sum(1 for row in curve if not row["sub_epoch"])
    return {
        "candidate_id": config.candidate_id,
        "config": config.to_dict(),
        "curve": curve,
        "epochs_run": epochs_completed,
        # Patience can fire on the last scheduled epoch, which is not an
        # early stop. The label reports what actually happened.
        "stopped_because": (
            "epochs_exhausted" if epochs_completed >= int(config.epochs) else stopped
        ),
        "best_epoch": best["epoch"],
        "best_step": best["step"],
        "best_epoch_fraction": round(best["step"] / steps_per_epoch, 4),
        "steps_per_epoch": steps_per_epoch,
        "evaluations": len(curve),
        "best_state": best["state"],
        "time_to_best_seconds": best["seconds"],
        "training_seconds": round(time.perf_counter() - started, 3),
        "train_positions": samples,
        "train_pieces": int(train_data["pieces"]),
        "dev_metrics": best_metrics,
        "observations_staged_on_device": staged,
    }


def inference_cost(
    model: nn.Module,
    observations: torch.Tensor,
    data: dict,
    *,
    device: str = "cpu",
    positions: int = 256,
    repeats: int = 10,
) -> dict:
    """Latency of the deployed path: one batch of positions to logit fields.

    Reported per decision *and* per hidden piece. Agent 1's heads are priced
    per piece because a head runs once per piece over a shared encoder pass;
    a convolution tower runs once per *position* and produces all 100
    squares at once, so per-decision is the honest primary number and
    per-piece is derived from the corpus's own pieces-per-decision.
    """
    target = torch.device(device)
    model = model.to(target).eval()
    rows = min(int(positions), int(data["samples"]))
    batch = observations[: rows].to(target)
    per_position = float(int(data["pieces"]) / max(int(data["samples"]), 1))
    with torch.no_grad():
        for _ in range(3):
            model(batch)
        _synchronize(target)
        started = time.perf_counter()
        for _ in range(int(repeats)):
            model(batch)
        _synchronize(target)
        elapsed = (time.perf_counter() - started) / int(repeats)
    with torch.no_grad():
        single = batch[:1]
        for _ in range(3):
            model(single)
        _synchronize(target)
        started = time.perf_counter()
        for _ in range(int(repeats)):
            model(single)
        _synchronize(target)
        single_elapsed = (time.perf_counter() - started) / int(repeats)
    return {
        "device": str(device),
        "batch_positions": rows,
        "batch_seconds": round(elapsed, 6),
        "milliseconds_per_decision_batched": round(elapsed / rows * 1e3, 4),
        "milliseconds_per_decision_single": round(single_elapsed * 1e3, 4),
        "microseconds_per_piece_batched": round(elapsed / rows / per_position * 1e6, 4),
        "hidden_pieces_per_decision": round(per_position, 3),
        "repeats": int(repeats),
    }


__all__ = [
    "RAW_TRAINER_VERSION",
    "Phase11BRawTrainError",
    "RawTrainConfig",
    "inference_cost",
    "predict_probabilities_raw",
    "stage_observations",
    "subset_split",
    "throughput_pilot",
    "train_raw_cnn",
]
