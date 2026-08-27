"""Phase 16 Agent 3: the damped schedules.

Specification source: `03_AGENT_3_TRAINING_LOOP_V2.md` section 2.3.

Why a schedule at all
---------------------
Phase 14 ran a constant 7.5e-5 learning rate and its *terminal* 0.001 entropy
floor from step 0, and gained nothing measurable after hour six. Two of the
divergences from the paper live here and are cheap to close:

```text
lr(n)   = clamp(lr_max * (n/n_ref)**-1.1, lr_min, lr_max)
c_H(n)  = max(0.001, 0.005 * n**-0.3)
```

The reference iteration (amendment, 2026-08-26)
-----------------------------------------------
`n_ref` is not in the brief; `n_ref = 1` reproduces the brief exactly and is
the default. It exists because the brief's exponents were transcribed from a
run of ~43,000 iterations and this one is ~313: at `n_ref = 1` the power law
reaches its 1.5e-5 floor at **n = 9**, so a six-hour arm would spend ~97% of
itself at a rate five times below the control's constant 7.5e-5, and the
shootout would measure a starved learning rate rather than a damped schedule.
Setting `n_ref = ceil(0.125 * N)` holds `lr_max` for the first ~12% of the
run, decays smoothly through the middle, and reaches the floor as the run
ends -- the same *shape*, mapped onto this horizon.

The entropy anneal is deliberately **not** re-horizoned: at `n^-0.3` it decays
from 0.005 and reaches the 0.001 terminal floor at n = 213, which is ~68% of a
313-iteration run. That is already a smooth decay across most of the run
followed by the terminal value, which is what section 2.3 asks for; the failure
it names is Phase 14 running *the terminal floor from step 0*, not a floor
arriving before the last iteration.

`n` is the **1-based iteration**, never the optimizer step. That distinction
matters on resume: an iteration is the unit the window collector and the
checkpoint both count, so a resumed run recomputes exactly the same value from
its restored iteration number, with no dependence on how many minibatches the
crashed process happened to complete.

Both are pure functions with no state, so nothing about them can drift across
a restart, and the constant variants exist as the control arm's flags rather
than as a separate code path.
"""

from __future__ import annotations

import math

from .contract import (
    ENTROPY_ANNEALED,
    ENTROPY_CONSTANT,
    LR_CONSTANT,
    LR_POWER_LAW,
    ArmConfig,
    Phase16TrainingError,
)


def _require_iteration(iteration: int) -> int:
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 1:
        raise Phase16TrainingError(
            f"the schedule index is the 1-based iteration, got {iteration!r}"
        )
    return int(iteration)


def power_law_learning_rate(
    iteration: int,
    *,
    lr_max: float,
    lr_min: float,
    exponent: float,
    reference: int = 1,
) -> float:
    """`clamp(lr_max * (n/n_ref)**-exponent, lr_min, lr_max)`.

    `reference = 1` is the brief's formula unchanged.
    """
    n = _require_iteration(iteration)
    if lr_min > lr_max:
        raise Phase16TrainingError(f"lr_min {lr_min} exceeds lr_max {lr_max}")
    if not isinstance(reference, int) or isinstance(reference, bool) or reference < 1:
        raise Phase16TrainingError(
            f"the reference iteration must be an int >= 1, got {reference!r}"
        )
    value = float(lr_max) * (float(n) / float(reference)) ** (-float(exponent))
    return float(min(max(value, float(lr_min)), float(lr_max)))


def annealed_entropy(
    iteration: int, *, start: float, floor: float, exponent: float
) -> float:
    """`max(floor, start * n**-exponent)`."""
    n = _require_iteration(iteration)
    value = float(start) * float(n) ** (-float(exponent))
    return float(max(value, float(floor)))


def learning_rate_for(config: ArmConfig, iteration: int) -> float:
    """The arm's learning rate at one iteration."""
    if config.lr_schedule == LR_CONSTANT:
        _require_iteration(iteration)
        return float(config.lr_constant)
    if config.lr_schedule == LR_POWER_LAW:
        return power_law_learning_rate(
            iteration,
            lr_max=config.lr_max,
            lr_min=config.lr_min,
            exponent=config.lr_exponent,
            reference=config.lr_reference,
        )
    raise Phase16TrainingError(f"unknown lr schedule: {config.lr_schedule!r}")


def entropy_coefficient_for(config: ArmConfig, iteration: int) -> float:
    """The arm's entropy coefficient at one iteration."""
    if config.entropy_schedule == ENTROPY_CONSTANT:
        _require_iteration(iteration)
        return float(config.entropy_constant)
    if config.entropy_schedule == ENTROPY_ANNEALED:
        return annealed_entropy(
            iteration,
            start=config.entropy_start,
            floor=config.entropy_floor,
            exponent=config.entropy_exponent,
        )
    raise Phase16TrainingError(f"unknown entropy schedule: {config.entropy_schedule!r}")


def schedule_row(config: ArmConfig, iteration: int) -> dict:
    """Both scheduled values at one iteration, for the telemetry row."""
    return {
        "iteration": _require_iteration(iteration),
        "learning_rate": learning_rate_for(config, iteration),
        "entropy_coefficient": entropy_coefficient_for(config, iteration),
        "lr_schedule": config.lr_schedule,
        "entropy_schedule": config.entropy_schedule,
    }


def schedule_curve(config: ArmConfig, iterations: int) -> list:
    """The first `iterations` scheduled rows, for the run config document."""
    if iterations < 1:
        raise Phase16TrainingError("a schedule curve covers at least one iteration")
    return [schedule_row(config, n) for n in range(1, int(iterations) + 1)]


def floor_iteration(config: ArmConfig) -> "int | None":
    """The first iteration at which the LR sits on its floor, if it ever does.

    Reported rather than inferred: "when does this schedule stop decaying" is
    the whole question the reference iteration exists to answer, and a run
    config that stated the constants but not their consequence would leave the
    reader to solve for it.
    """
    if config.lr_schedule != LR_POWER_LAW:
        return None
    ratio = float(config.lr_min) / float(config.lr_max)
    if ratio >= 1.0:  # pragma: no cover - the config validator refuses this
        return 1
    return int(math.ceil(config.lr_reference * ratio ** (-1.0 / config.lr_exponent)))


def schedule_semantics(config: ArmConfig) -> dict:
    return {
        "index": "the 1-based iteration; never the optimizer step",
        "lr": {
            "schedule": config.lr_schedule,
            "formula": (
                f"clamp({config.lr_max} * (n/{config.lr_reference})**-"
                f"{config.lr_exponent}, {config.lr_min}, {config.lr_max})"
                if config.lr_schedule == LR_POWER_LAW
                else f"{config.lr_constant} at every iteration"
            ),
            "reference_iteration": config.lr_reference,
            "planned_iterations": config.planned_iterations,
            "floor_reached_at": floor_iteration(config),
        },
        "entropy": {
            "schedule": config.entropy_schedule,
            "formula": (
                f"max({config.entropy_floor}, {config.entropy_start} * "
                f"n**-{config.entropy_exponent})"
                if config.entropy_schedule == ENTROPY_ANNEALED
                else f"{config.entropy_constant} at every iteration"
            ),
        },
    }


__all__ = [
    "annealed_entropy",
    "floor_iteration",
    "entropy_coefficient_for",
    "learning_rate_for",
    "power_law_learning_rate",
    "schedule_curve",
    "schedule_row",
    "schedule_semantics",
]
