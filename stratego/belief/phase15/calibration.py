"""Phase 15 Agent 1 section 10: one temperature per specialist.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` section 10.

Search needs probabilities it can trust
---------------------------------------
The constrained-world sampler weights each candidate rank by
`learned_probability * remaining_count`. A model that is confidently wrong
therefore does not merely score badly — it steers the sampler. One positive
scalar temperature, fitted by minimizing negative log-likelihood on the
**calibration split only**, is the smallest correction that can help, and
the only one that provably cannot change which rank a piece is assigned as
its top-1.

Why top-1 cannot move
---------------------
`softmax(z / T)` with `T > 0` is a strictly increasing reparameterisation
of each row's ordering: `z_i > z_j` iff `z_i/T > z_j/T`. The argmax is
therefore identical for every positive `T`, and :func:`fit_temperature`
asserts it on the fitted value rather than trusting the algebra.

Kept only if it helps
---------------------
Section 10: keep the calibrated version only if it improves development
NLL *and* calibration error. :func:`decide` applies that rule to the two
development metric blocks and returns the decision with the numbers that
produced it, so a report never has to restate the comparison.
"""

from __future__ import annotations

import numpy as np

from .contract import Phase15Error

#: The calibration procedure's identity.
CALIBRATION_VERSION = "phase15_temperature_scaling_v1"

#: The search interval for `log T`. `T` in `[e^-3, e^3]` = `[0.05, 20]`,
#: far wider than any temperature a working model needs.
LOG_TEMPERATURE_BOUNDS = (-3.0, 3.0)

#: Golden-section iterations. 60 brackets `log T` to under 1e-6.
SEARCH_ITERATIONS = 60


class Phase15CalibrationError(Phase15Error):
    """A temperature could not be fitted or validated."""


def scaled_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    """`softmax(logits / T)` in float64, numerically stabilised."""
    temperature = float(temperature)
    if not temperature > 0.0:
        raise Phase15CalibrationError(f"temperature must be positive, got {temperature}")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def negative_log_likelihood(
    logits: np.ndarray, true_rank: np.ndarray, temperature: float
) -> float:
    """Mean NLL of the temperature-scaled logits."""
    probabilities = scaled_probabilities(logits, temperature)
    rows = np.arange(probabilities.shape[0])
    mass = np.maximum(
        probabilities[rows, np.asarray(true_rank, dtype=np.int64)],
        np.finfo(np.float64).tiny,
    )
    return float(-np.log(mass).mean())


def fit_temperature(logits: np.ndarray, true_rank: np.ndarray) -> dict:
    """Fit one positive scalar temperature by golden-section on `log T`.

    NLL as a function of `log T` is smooth and unimodal for a fixed set of
    logits, so a derivative-free bracket search is both sufficient and
    exactly reproducible — no optimizer state, no learning rate, no seed.
    """
    logits = np.asarray(logits, dtype=np.float64)
    true_rank = np.asarray(true_rank, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[0] != true_rank.shape[0]:
        raise Phase15CalibrationError(
            f"logits {logits.shape} do not match {true_rank.shape[0]} labels"
        )
    if not logits.size:
        raise Phase15CalibrationError("no calibration pieces")

    golden = (np.sqrt(5.0) - 1.0) / 2.0
    lower, upper = LOG_TEMPERATURE_BOUNDS
    left = upper - golden * (upper - lower)
    right = lower + golden * (upper - lower)
    f_left = negative_log_likelihood(logits, true_rank, float(np.exp(left)))
    f_right = negative_log_likelihood(logits, true_rank, float(np.exp(right)))
    for _ in range(SEARCH_ITERATIONS):
        if f_left < f_right:
            upper, right, f_right = right, left, f_left
            left = upper - golden * (upper - lower)
            f_left = negative_log_likelihood(logits, true_rank, float(np.exp(left)))
        else:
            lower, left, f_left = left, right, f_right
            right = lower + golden * (upper - lower)
            f_right = negative_log_likelihood(logits, true_rank, float(np.exp(right)))
    temperature = float(np.exp(0.5 * (lower + upper)))

    raw_argmax = logits.argmax(axis=1)
    scaled_argmax = scaled_probabilities(logits, temperature).argmax(axis=1)
    moved = int((raw_argmax != scaled_argmax).sum())
    if moved:  # pragma: no cover - impossible for positive T
        raise Phase15CalibrationError(
            f"temperature scaling moved {moved} top-1 labels; T={temperature}"
        )
    return {
        "calibration_version": CALIBRATION_VERSION,
        "temperature": temperature,
        "log_temperature_bounds": list(LOG_TEMPERATURE_BOUNDS),
        "iterations": SEARCH_ITERATIONS,
        "calibration_pieces": int(logits.shape[0]),
        "calibration_nll_raw": negative_log_likelihood(logits, true_rank, 1.0),
        "calibration_nll_fitted": negative_log_likelihood(
            logits, true_rank, temperature
        ),
        "top1_labels_changed": moved,
    }


def decide(raw_metrics: dict, calibrated_metrics: dict) -> dict:
    """Section 10's keep-or-drop rule, applied to two development blocks.

    Keep the calibrated version only if it improves development NLL **and**
    calibration error, and only if it left legality and interface behaviour
    untouched — which for a positive scalar temperature means the top-1
    labels and therefore the accuracy are identical.
    """
    nll_improved = calibrated_metrics["nll"] < raw_metrics["nll"]
    ece_improved = (
        calibrated_metrics["expected_calibration_error"]
        < raw_metrics["expected_calibration_error"]
    )
    top1_unchanged = abs(calibrated_metrics["top1"] - raw_metrics["top1"]) < 1e-12
    keep = bool(nll_improved and ece_improved and top1_unchanged)
    return {
        "keep_calibrated": keep,
        "development_nll_raw": raw_metrics["nll"],
        "development_nll_calibrated": calibrated_metrics["nll"],
        "development_nll_improved": bool(nll_improved),
        "development_ece_raw": raw_metrics["expected_calibration_error"],
        "development_ece_calibrated": calibrated_metrics["expected_calibration_error"],
        "development_ece_improved": bool(ece_improved),
        "development_mce_raw": raw_metrics["maximum_calibration_error"],
        "development_mce_calibrated": calibrated_metrics["maximum_calibration_error"],
        "top1_unchanged": bool(top1_unchanged),
        "rule": (
            "keep the calibrated version only if it improves development NLL and "
            "calibration error without changing legality or interface behaviour"
        ),
    }


__all__ = [
    "CALIBRATION_VERSION",
    "LOG_TEMPERATURE_BOUNDS",
    "SEARCH_ITERATIONS",
    "Phase15CalibrationError",
    "decide",
    "fit_temperature",
    "negative_log_likelihood",
    "scaled_probabilities",
]
