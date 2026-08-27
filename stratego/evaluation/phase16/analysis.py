"""Phase 16 Agent 1: reading the packs.

EWR is the mean of the accepted engine's own `effective_score_for` — draws
count half, nothing modelled, nothing adjusted. Every table carries its own
game count, standard errors are reported, and **no significance claim is
made anywhere**: these are engineering packs read against predeclared
margins.

The baseline reading is predeclared in the brief and restated here so the
code cannot drift from it:

```text
drop = EWR(arm 1 control) - EWR(arm 2 adversarial opponent)
drop >= 0.10  confirms the distribution hypothesis
drop <  0.05  weakens it — and must be stated plainly either way
```
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .contract import (
    ADVERSARIAL_CONFIRM_DROP,
    ADVERSARIAL_WEAKEN_DROP,
    ARM_ADVERSARIAL_BOTH,
    ARM_ADVERSARIAL_OPPONENT,
    ARM_CONTROL,
    Phase16MeasurementError,
)

ANALYSIS_VERSION = "phase16_measurement_analysis_v1"


def _ewr(rows) -> "float | None":
    if not rows:
        return None
    return float(np.mean([float(row["effective_score"]) for row in rows]))


def _wdl(rows) -> dict:
    counts = {"win": 0, "draw": 0, "loss": 0}
    for row in rows:
        counts[row["outcome"]] += 1
    return counts


def paired_delta_by(rows, reference, key: str = "ordinal") -> dict:
    """Mean over shared units of `rows - reference`, with its standard error."""
    left = {row[key]: float(row["effective_score"]) for row in rows}
    right = {row[key]: float(row["effective_score"]) for row in reference}
    shared = sorted(set(left) & set(right))
    if not shared:
        return {"pairs": 0, "delta": None, "standard_error": None}
    differences = np.asarray([left[unit] - right[unit] for unit in shared])
    standard_error = (
        float(differences.std(ddof=1) / math.sqrt(len(shared)))
        if len(shared) > 1
        else None
    )
    return {
        "pairs": len(shared),
        "delta": round(float(differences.mean()), 5),
        "standard_error": round(standard_error, 5) if standard_error is not None else None,
        "wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "losses": int((differences < 0).sum()),
    }


def predeclared_reading(drop: "float | None") -> str:
    if drop is None:
        return "not_measured"
    if drop >= ADVERSARIAL_CONFIRM_DROP:
        return "confirms_distribution_hypothesis"
    if drop < ADVERSARIAL_WEAKEN_DROP:
        return "weakens_distribution_hypothesis"
    return "between_predeclared_thresholds"


def analyse_baseline(rows: "list[dict]") -> dict:
    """Overall and per-family readings of one preset's three-arm pack.

    `rows` are game rows of a single preset; the arm is the row's
    `setup_source` and the adversarial family is its `requested_family`.
    """
    if not rows:
        raise Phase16MeasurementError("no rows to analyse")
    presets = {row["preset_id"] for row in rows}
    if len(presets) != 1:
        raise Phase16MeasurementError(
            f"analyse one preset at a time; got {sorted(presets)}"
        )
    by_arm: dict[str, list] = defaultdict(list)
    for row in rows:
        by_arm[row["setup_source"]].append(row)

    arms = {}
    for arm, arm_rows in sorted(by_arm.items()):
        by_family: dict[str, list] = defaultdict(list)
        by_opponent: dict[str, list] = defaultdict(list)
        for row in arm_rows:
            by_family[row["requested_family"]].append(row)
            by_opponent[row["opponent"]].append(row)
        arms[arm] = {
            "games": len(arm_rows),
            "ewr": round(_ewr(arm_rows), 5),
            **_wdl(arm_rows),
            "ewr_by_family": {
                family: {"games": len(entries), "ewr": round(_ewr(entries), 5)}
                for family, entries in sorted(by_family.items())
            },
            "ewr_by_opponent": {
                opponent: {"games": len(entries), "ewr": round(_ewr(entries), 5)}
                for opponent, entries in sorted(by_opponent.items())
            },
        }

    control = by_arm.get(ARM_CONTROL, [])
    comparisons = {}
    for arm in (ARM_ADVERSARIAL_OPPONENT, ARM_ADVERSARIAL_BOTH):
        arm_rows = by_arm.get(arm, [])
        if not arm_rows or not control:
            continue
        overall = paired_delta_by(arm_rows, control)
        families = {}
        control_by_family: dict[str, list] = defaultdict(list)
        arm_by_family: dict[str, list] = defaultdict(list)
        for row in control:
            control_by_family[row["requested_family"]].append(row)
        for row in arm_rows:
            arm_by_family[row["requested_family"]].append(row)
        for family in sorted(set(control_by_family) | set(arm_by_family)):
            families[family] = paired_delta_by(
                arm_by_family.get(family, []), control_by_family.get(family, [])
            )
        drop = None if overall["delta"] is None else round(-overall["delta"], 5)
        comparisons[f"{arm}_minus_control"] = {
            "overall": overall,
            "drop": drop,
            "per_family": families,
        }

    primary = comparisons.get(f"{ARM_ADVERSARIAL_OPPONENT}_minus_control", {})
    return {
        "analysis_version": ANALYSIS_VERSION,
        "preset": rows[0]["preset_id"],
        "games": len(rows),
        "arms": arms,
        "paired": comparisons,
        "predeclared_thresholds": {
            "confirm_drop": ADVERSARIAL_CONFIRM_DROP,
            "weaken_drop": ADVERSARIAL_WEAKEN_DROP,
        },
        "reading": predeclared_reading(primary.get("drop")),
    }


__all__ = [
    "ANALYSIS_VERSION",
    "analyse_baseline",
    "paired_delta_by",
    "predeclared_reading",
]
