"""Phase 8 Agent 4: validation metrics against the frozen Agent 3 baselines.

Specification sources:

- `04_AGENT_4_TRAINER_AND_RESUME.md` ("Validation implementation")
- Agent 1's `warmstart_eval_v1`
  (:func:`stratego.training.warmstart_contract.evaluation_contract`)
- Agent 3's frozen baselines (:mod:`stratego.training.warmstart_baselines`
  and `reports/phase_8_data/agent_03_validation_baselines.json`)

What a validation pass is allowed to do
---------------------------------------
Read one held-out split in its frozen sequential order, run the model under
`no_grad`, and aggregate per game. It must not update the optimizer or
scheduler, must not touch the training data cursor (it plans from its own
fresh sequential cursor), must not leave the model in a different mode than it
found it, and must never open the test split before Agent 7 — enforced here by
routing every `split="test"` request through the frozen
:func:`stratego.training.warmstart_contract.check_test_corpus_access` gate.

Baselines are recomputed over the served population
---------------------------------------------------
Each metric pairs the model against its frozen baseline *on exactly the same
examples*: uniform-legal CE from the batch's own legal counts, the constant
train-fitted W/D/L prior, and the observable unresolved-inventory belief
marginal read off the observation planes. Over the full validation split these
reproduce Agent 3's frozen artifact numbers; over a cadence-sized prefix they
remain an exactly paired comparison, which is what the ratio needs to mean
anything. Per-game sufficient statistics are kept so confidence intervals can
bootstrap by game, never by decision.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..model.contract import ModelOutputs
from ..model.losses import masked_policy_log_probabilities
from .synthetic_corpus import repository_root
from .warmstart_baselines import (
    belief_marginal_statistics,
    unresolved_counts_from_observation,
)
from .warmstart_contract import (
    METRIC_LOG_EPSILON,
    WARMSTART_EVAL_VERSION,
    check_test_corpus_access,
)
from .warmstart_dataset import (
    DEFAULT_BATCH_SIZE,
    ORDER_SEQUENTIAL,
    DataCursor,
    WarmstartBatch,
    WarmstartDataset,
    batch_from_arrays,
    plan_batch,
)

#: The validation-metric implementation version.
WARMSTART_METRICS_VERSION = "warmstart_metrics_v1"

#: Artifact holding Agent 3's frozen train-fitted value prior.
_AGENT3_BASELINES_ARTIFACT = "reports/phase_8_data/agent_03_validation_baselines.json"


class WarmstartMetricsError(RuntimeError):
    """A validation pass violated its contract or met non-finite outputs."""


def frozen_train_value_prior() -> tuple:
    """The frozen constant W/D/L prior Agent 3 fitted on train and froze.

    Read from the accepted artifact rather than re-fitted, so every later
    agent evaluates against literally the same three numbers.
    """
    path = repository_root() / _AGENT3_BASELINES_ARTIFACT
    try:
        payload = json.loads(path.read_text())
        prior = payload["value_prior"]["prior_win_draw_loss"]
    except (OSError, KeyError, ValueError) as error:
        raise WarmstartMetricsError(
            f"cannot read the frozen train value prior from {path}: {error}"
        ) from error
    if len(prior) != 3 or abs(sum(prior) - 1.0) > 1e-9:
        raise WarmstartMetricsError(f"frozen value prior {prior!r} is not a distribution")
    return tuple(float(entry) for entry in prior)


# ---------------------------------------------------------------------------
# Per-game sufficient statistics
# ---------------------------------------------------------------------------

#: One game's additive contributions to every validation metric. Plain floats
#: so a games-long dict serializes straight into an artifact.
_GAME_FIELDS = (
    "policy_weighted_ce",
    "policy_weighted_baseline_ce",
    "policy_weighted_top1",
    "policy_weighted_expected_top1",
    "policy_weight_sum",
    "policy_examples",
    "value_ce",
    "value_baseline_ce",
    "value_brier",
    "value_baseline_brier",
    "value_top1",
    "value_baseline_top1",
    "value_examples",
    "belief_ce",
    "belief_baseline_ce",
    "belief_top1",
    "belief_baseline_top1",
    "belief_pieces",
)


def _empty_game_stats() -> dict:
    return {name: 0.0 for name in _GAME_FIELDS}


def accumulate_batch_statistics(
    outputs: ModelOutputs,
    batch: WarmstartBatch,
    *,
    value_prior: tuple,
    per_game: dict,
) -> None:
    """Fold one batch's model-vs-baseline statistics into `per_game`.

    Everything is computed on detached CPU float32 copies in float64
    arithmetic; ties break toward the lowest index (`numpy.argmax` takes the
    first maximum), exactly the frozen `warmstart_eval_v1` tie-break.
    """
    detached = outputs.detached_cpu()
    if not detached.all_finite():
        raise WarmstartMetricsError(
            "the model produced a non-finite output on a validation batch"
        )
    targets = batch.targets
    batch_size = batch.batch_size

    legal_mask = targets.legal_mask.to(torch.bool)
    log_probabilities = (
        masked_policy_log_probabilities(detached.policy_logits, legal_mask)
        .to(torch.float64)
        .numpy()
    )
    legal = legal_mask.numpy()
    legal_counts = legal.sum(axis=1).astype(np.float64)
    actions = targets.policy_action_model.to(torch.int64).numpy()
    weights = targets.policy_weight.to(torch.float64).numpy()
    # -inf on illegal entries for the argmax so an illegal index can never win
    # a tie against a legal logit that happens to sit at the fill value.
    policy_scores = detached.policy_logits.to(torch.float64).numpy().copy()
    policy_scores[~legal] = -np.inf
    policy_top1 = policy_scores.argmax(axis=1)

    value_log_probabilities = (
        torch.log_softmax(detached.value_logits.to(torch.float64), dim=1).numpy()
    )
    value_probabilities = np.exp(value_log_probabilities)
    value_targets = targets.value_target.to(torch.int64).numpy()
    prior = np.asarray(value_prior, dtype=np.float64)
    prior_log = np.log(np.maximum(prior, METRIC_LOG_EPSILON))
    prior_top1 = int(np.argmax(prior))
    prior_brier_by_class = ((prior[None, :] - np.eye(prior.size)) ** 2).sum(axis=1)

    belief_log_probabilities = (
        torch.log_softmax(detached.belief_logits.to(torch.float64), dim=2).numpy()
    )
    belief_targets = targets.belief_target.to(torch.int64).numpy()
    belief_mask = targets.belief_mask.to(torch.bool).numpy()
    observations = batch.observations.numpy()

    for row in range(batch_size):
        game_id = batch.keys[row][0]
        stats = per_game.get(game_id)
        if stats is None:
            stats = _empty_game_stats()
            per_game[game_id] = stats

        weight = float(weights[row])
        if weight > 0.0:
            action = int(actions[row])
            count = float(legal_counts[row])
            stats["policy_weighted_ce"] += weight * float(-log_probabilities[row, action])
            stats["policy_weighted_baseline_ce"] += weight * float(np.log(count))
            stats["policy_weighted_top1"] += weight * float(policy_top1[row] == action)
            stats["policy_weighted_expected_top1"] += weight / count
            stats["policy_weight_sum"] += weight
            stats["policy_examples"] += 1.0

        target = int(value_targets[row])
        stats["value_ce"] += float(-value_log_probabilities[row, target])
        stats["value_baseline_ce"] += float(-prior_log[target])
        one_hot = np.zeros(prior.size)
        one_hot[target] = 1.0
        stats["value_brier"] += float(((value_probabilities[row] - one_hot) ** 2).sum())
        stats["value_baseline_brier"] += float(prior_brier_by_class[target])
        stats["value_top1"] += float(int(np.argmax(value_probabilities[row])) == target)
        stats["value_baseline_top1"] += float(prior_top1 == target)
        stats["value_examples"] += 1.0

        supervised = np.flatnonzero(belief_mask[row])
        if supervised.size:
            labels = belief_targets[row, supervised]
            stats["belief_ce"] += float(
                -belief_log_probabilities[row, supervised, labels].sum()
            )
            predicted = belief_log_probabilities[row, supervised].argmax(axis=1)
            stats["belief_top1"] += float((predicted == labels).sum())
            marginal = belief_marginal_statistics(
                unresolved_counts_from_observation(observations[row]), labels
            )
            stats["belief_baseline_ce"] += marginal["cross_entropy_sum"]
            stats["belief_baseline_top1"] += float(marginal["top1_hits"])
            stats["belief_pieces"] += float(supervised.size)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """One validation pass: totals, frozen-baseline ratios, per-game support."""

    split: str
    examples: int
    games: int
    batches: int
    policy: dict
    value: dict
    belief: dict
    selection_score: "float | None"
    per_game: dict = field(repr=False)
    seconds: float = 0.0
    eval_version: str = WARMSTART_EVAL_VERSION
    metrics_version: str = WARMSTART_METRICS_VERSION

    def to_dict(self, *, include_per_game: bool = False) -> dict:
        payload = {
            "split": self.split,
            "examples": self.examples,
            "games": self.games,
            "batches": self.batches,
            "policy": dict(self.policy),
            "value": dict(self.value),
            "belief": dict(self.belief),
            "selection_score": self.selection_score,
            "seconds": self.seconds,
            "eval_version": self.eval_version,
            "metrics_version": self.metrics_version,
        }
        if include_per_game:
            payload["per_game"] = {key: dict(value) for key, value in self.per_game.items()}
        return payload


def _ratio(model: float, baseline: float) -> "float | None":
    return model / baseline if baseline > 0 else None


def summarize_games(per_game: dict, *, split: str, batches: int, seconds: float) -> ValidationResult:
    """Fold per-game sufficient statistics into one `ValidationResult`."""
    totals = _empty_game_stats()
    for stats in per_game.values():
        for name in _GAME_FIELDS:
            totals[name] += stats[name]

    weight_sum = totals["policy_weight_sum"]
    policy = {
        "population": "policy-supervised examples (weight > 0)",
        "examples": int(totals["policy_examples"]),
        "weight_sum": weight_sum,
        "model_ce": totals["policy_weighted_ce"] / weight_sum if weight_sum else None,
        "baseline_ce": (
            totals["policy_weighted_baseline_ce"] / weight_sum if weight_sum else None
        ),
        "model_top1": totals["policy_weighted_top1"] / weight_sum if weight_sum else None,
        "baseline_expected_top1": (
            totals["policy_weighted_expected_top1"] / weight_sum if weight_sum else None
        ),
    }
    policy["ce_ratio"] = (
        _ratio(policy["model_ce"], policy["baseline_ce"]) if weight_sum else None
    )

    value_examples = totals["value_examples"]
    value = {
        "population": "every selected decision",
        "examples": int(value_examples),
        "model_ce": totals["value_ce"] / value_examples if value_examples else None,
        "baseline_ce": totals["value_baseline_ce"] / value_examples if value_examples else None,
        "model_brier": totals["value_brier"] / value_examples if value_examples else None,
        "baseline_brier": (
            totals["value_baseline_brier"] / value_examples if value_examples else None
        ),
        "model_accuracy": totals["value_top1"] / value_examples if value_examples else None,
        "baseline_accuracy": (
            totals["value_baseline_top1"] / value_examples if value_examples else None
        ),
    }
    value["ce_ratio"] = (
        _ratio(value["model_ce"], value["baseline_ce"]) if value_examples else None
    )

    pieces = totals["belief_pieces"]
    belief = {
        "population": "supervised hidden opponent pieces",
        "pieces": int(pieces),
        "model_ce": totals["belief_ce"] / pieces if pieces else None,
        "baseline_ce": totals["belief_baseline_ce"] / pieces if pieces else None,
        "model_top1": totals["belief_top1"] / pieces if pieces else None,
        "baseline_top1": totals["belief_baseline_top1"] / pieces if pieces else None,
    }
    belief["ce_ratio"] = _ratio(belief["model_ce"], belief["baseline_ce"]) if pieces else None

    ratios = [policy["ce_ratio"], value["ce_ratio"], belief["ce_ratio"]]
    selection_score = (
        sum(ratios) / len(ratios) if all(entry is not None for entry in ratios) else None
    )
    return ValidationResult(
        split=split,
        examples=int(value_examples),
        games=len(per_game),
        batches=batches,
        policy=policy,
        value=value,
        belief=belief,
        selection_score=selection_score,
        per_game=per_game,
        seconds=seconds,
    )


# ---------------------------------------------------------------------------
# The validation pass
# ---------------------------------------------------------------------------


def spread_batch_positions(
    universe_size: int, batch_size: int, batches: int
) -> tuple:
    """`batches` full-batch cursor positions spread evenly over one split.

    A pure function of `(universe_size, batch_size, batches)`: position `k` is
    `floor(k * M / N) * batch_size` over the `M = floor(size/B)` full-batch
    grid, deduplicated and ascending. A cadence-sized validation served from
    these positions sees every region of the frozen sequential order — the
    schedule is cell-major, so a plain prefix would sample only the first
    matchup cells (which are policy-unsupervised random-vs-random games).
    """
    size = int(universe_size)
    stride = int(batch_size)
    wanted = max(1, int(batches))
    full_batches = max(1, size // stride) if size else 1
    positions = dict.fromkeys(
        (index * full_batches) // wanted * stride for index in range(wanted)
    )
    return tuple(positions)


def run_validation(
    model,
    dataset: WarmstartDataset,
    *,
    split: str = "validation",
    value_prior: tuple,
    batches: "int | None" = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: "torch.device | str" = "cpu",
    phase8_agent: int = 4,
    spread: bool = False,
) -> ValidationResult:
    """One deterministic held-out pass under the frozen access rules.

    Serves the split's frozen sequential order, stopping after `batches`
    batches (`None` = the whole split, one epoch, from position zero).
    `spread=True` draws the `batches` batch positions evenly across the split
    (`spread_batch_positions`) instead of taking the prefix — the cadence-
    validation mode, still a pure function of the frozen order. The model's
    train/eval mode is restored afterwards; nothing here can reach an
    optimizer, a scheduler, or the training cursor, because none are passed
    in. Requests against the sealed test split go through the frozen
    `check_test_corpus_access` gate and raise before Agent 7.
    """
    if split == "test":
        check_test_corpus_access("final_evaluation", phase8_agent=phase8_agent)
    started = time.perf_counter()
    universe = dataset.universe(split)
    per_game: dict = {}
    served = 0
    was_training = bool(getattr(model, "training", False))
    model.eval()

    def serve(cursor: DataCursor) -> DataCursor:
        nonlocal served
        keys, cursor_after = plan_batch(universe, cursor)
        arrays, metadata, _stats = dataset.batch_arrays(keys)
        batch = batch_from_arrays(arrays, metadata)
        outputs = model.forward_observation(batch.model_input().to(torch.device(device)))
        accumulate_batch_statistics(
            outputs, batch, value_prior=value_prior, per_game=per_game
        )
        served += 1
        return cursor_after

    try:
        with torch.no_grad():
            if spread and batches is not None:
                for position in spread_batch_positions(
                    len(universe), int(batch_size), int(batches)
                ):
                    serve(
                        DataCursor(
                            split=split,
                            batch_size=int(batch_size),
                            position=int(position),
                            order=ORDER_SEQUENTIAL,
                        )
                    )
            else:
                cursor = DataCursor(
                    split=split, batch_size=int(batch_size), order=ORDER_SEQUENTIAL
                )
                while cursor.epoch == 0 and (batches is None or served < int(batches)):
                    cursor = serve(cursor)
    finally:
        model.train(was_training)
    return summarize_games(
        per_game,
        split=split,
        batches=served,
        seconds=time.perf_counter() - started,
    )


__all__ = [
    "WARMSTART_METRICS_VERSION",
    "ValidationResult",
    "WarmstartMetricsError",
    "accumulate_batch_statistics",
    "frozen_train_value_prior",
    "run_validation",
    "spread_batch_positions",
    "summarize_games",
]
