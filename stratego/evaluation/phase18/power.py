"""Power and sample size for the Phase 18 Gate G1 non-inferiority confirmation.

P18-D002 failed one margin for a reason that was not the model: at 1,024 paired
setups the two-sided 95% paired interval is about +/-0.0116 wide, and the margin
it had to certify was 0.010. The reviewing chat's audit made the point precisely
- the difficulty is that an *approximately equal* model has poor power against a
tight non-inferiority margin, not that passing is impossible, since a
sufficiently positive true delta clears a fixed-width interval.

This module is the arithmetic behind that. It is deliberately tiny and is used
to size the confirmation *before* any outcome is opened; nothing here may be
re-run on observed data to justify a sample after the fact.

The test the project actually applies is: read the lower endpoint of the
two-sided `confidence` interval and pass when it exceeds `-margin`. Treating the
paired mean as approximately normal with standard error `sd / sqrt(n)`, the
probability of passing at a true difference `delta` is

```text
power = Phi( (delta + margin) / se  -  z_(1 - alpha/2) )
```

which inverts to

```text
n = ceil( ( (z_(1 - alpha/2) + z_power) * sd / (delta + margin) )^2 )
```

At `delta = 0`, `sd = 0.189374` and `margin = 0.010` this gives 2,815 pairs for
80% power and 3,769 for 90% - the two figures the audit quotes. The confirmation
is frozen at 4,096.

The normal quantile is imported from `stratego.evaluation.statistics` rather than
reimplemented, so this module and the interval it reasons about cannot drift onto
two different definitions of the same number.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from stratego.evaluation.statistics import _normal_quantile

#: The planning standard deviation of the paired per-case difference, measured by
#: Agent 2 on the original 1,024-pair random gate. A planning input, never
#: re-estimated from the confirmation's own outcomes.
PLANNING_SD = 0.189374

DEFAULT_CONFIDENCE = 0.95
DEFAULT_TARGET_POWER = 0.90


class PowerError(ValueError):
    """A malformed power calculation. Always raised, never silently repaired."""


@dataclass(frozen=True)
class PowerPlan:
    """One sizing calculation, with every input it depended on."""

    planning_sd: float
    margin: float
    confidence: float
    target_power: float
    true_delta: float
    z_confidence: float
    z_power: float
    minimum_n: int
    frozen_n: int
    power_at_frozen_n: float
    standard_error_at_frozen_n: float
    half_width_at_frozen_n: float
    formula: str

    def to_dict(self) -> dict:
        return asdict(self)


def normal_cdf(value: float) -> float:
    """Standard normal CDF, the inverse of `statistics._normal_quantile`."""
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _validate(sd: float, margin: float, confidence: float) -> None:
    if sd <= 0.0:
        raise PowerError(f"planning sd must be positive, got {sd}")
    if margin <= 0.0:
        raise PowerError(f"margin must be a positive magnitude, got {margin}")
    if not 0.0 < confidence < 1.0:
        raise PowerError(f"confidence must be in (0, 1), got {confidence}")


def noninferiority_sample_size(
    sd: float = PLANNING_SD,
    margin: float = 0.010,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    target_power: float = DEFAULT_TARGET_POWER,
    true_delta: float = 0.0,
) -> int:
    """Paired cases needed for `target_power` against `margin` at `true_delta`."""
    _validate(sd, margin, confidence)
    if not 0.0 < target_power < 1.0:
        raise PowerError(f"target power must be in (0, 1), got {target_power}")
    slack = true_delta + margin
    if slack <= 0.0:
        raise PowerError(
            f"a true delta of {true_delta} is not inside the margin {margin}; no "
            "sample size gives the test power against it"
        )
    z_confidence = _normal_quantile(1.0 - (1.0 - confidence) / 2.0)
    z_power = _normal_quantile(target_power)
    return math.ceil(((z_confidence + z_power) * sd / slack) ** 2)


def noninferiority_power(
    n: int,
    sd: float = PLANNING_SD,
    margin: float = 0.010,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    true_delta: float = 0.0,
) -> float:
    """Probability the lower endpoint clears `-margin` with `n` paired cases."""
    _validate(sd, margin, confidence)
    if n < 1:
        raise PowerError(f"n must be at least 1, got {n}")
    z_confidence = _normal_quantile(1.0 - (1.0 - confidence) / 2.0)
    standard_error = sd / math.sqrt(n)
    return normal_cdf((true_delta + margin) / standard_error - z_confidence)


def plan(
    frozen_n: int,
    sd: float = PLANNING_SD,
    margin: float = 0.010,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    target_power: float = DEFAULT_TARGET_POWER,
    true_delta: float = 0.0,
) -> PowerPlan:
    """The complete sizing record frozen into the confirmation contract."""
    _validate(sd, margin, confidence)
    z_confidence = _normal_quantile(1.0 - (1.0 - confidence) / 2.0)
    standard_error = sd / math.sqrt(frozen_n)
    return PowerPlan(
        planning_sd=float(sd),
        margin=float(margin),
        confidence=float(confidence),
        target_power=float(target_power),
        true_delta=float(true_delta),
        z_confidence=z_confidence,
        z_power=_normal_quantile(target_power),
        minimum_n=noninferiority_sample_size(
            sd, margin, confidence=confidence, target_power=target_power,
            true_delta=true_delta,
        ),
        frozen_n=int(frozen_n),
        power_at_frozen_n=noninferiority_power(
            frozen_n, sd, margin, confidence=confidence, true_delta=true_delta
        ),
        standard_error_at_frozen_n=standard_error,
        half_width_at_frozen_n=z_confidence * standard_error,
        formula=(
            "n = ceil( ((z_(1-alpha/2) + z_power) * sd / (true_delta + margin))^2 ); "
            "power = Phi( (true_delta + margin)/(sd/sqrt(n)) - z_(1-alpha/2) )"
        ),
    )
