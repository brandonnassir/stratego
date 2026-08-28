"""Phase 17 Agent 3: diversity and entropy measurement for sampled setups.

Specification sources:

- `03_AGENT_3_AUTOREGRESSIVE_SETUP_NETWORK.md` section 7 ("Meaningful diversity")
- `00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md` sections 12 and 13
- `reports/phase17/phase17_contract_handoff_v1.json` ->
  `gates.calibration_dependencies`

Why the accepted metrics are imported but the accepted thresholds are not
------------------------------------------------------------------------
`stratego/setups/diversity.py` is the accepted measurement standard and its
primitives are reused here unchanged. Its *numbers* are not: they were frozen
against an 8,000-board library of eight authored families, and Agent 1's
handoff states plainly that frozen-library family thresholds must not be
borrowed. Phase 17's setup distribution starts from a randomly initialised
masked model, which is a different object entirely -- a threshold that a
random model passes trivially and a converged one fails is worse than no
threshold. Every alarm here is calibrated against the initial masked model
and the soak, and the calibration is reported rather than assumed.

Raw uniqueness is not evidence
------------------------------
Two setups that are mirror images are different tuples and identical
strategies, so a generator that draws a board and its reflection scores 100%
"unique". Every uniqueness figure here is therefore reported twice: exact,
and by reflection class. The class figure is the one that means something.

Units
-----
Per-square entropy follows the accepted module and is in **bits**. Prefix and
sequence entropies are in **nats**, matching the setup advantage's `I`. Both
are named in their field names so the two can never be silently compared.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ...engine.constants import BOMB, FLAG, NUM_PIECE_TYPES
from ...setups.diversity import (
    LibraryEntry,
    class_distance,
    folded_support,
    hamming_distance,
    per_square_entropy_bits,
)
from ...setups.identity import (
    CANONICAL_CELLS,
    class_fingerprint,
    content_fingerprint,
    reflect_canonical,
)
from .setup_contract import (
    PROVISIONAL_FLAG_EFFECTIVE_SUPPORT_FLOOR,
    PROVISIONAL_PREFIX_ENTROPY_FLOOR_CONSECUTIVE_CHECKS,
    PROVISIONAL_PREFIX_ENTROPY_FLOOR_FRACTION,
    SETUP_PREFIXES,
    Phase17SetupError,
)

#: The label carried by the throwaway `LibraryEntry` wrappers below. The
#: accepted per-square-entropy and folded-support helpers read only
#: `.canonical`, and `test_setup_metrics` pins that by recomputing both
#: independently -- so this label is inert, not a claimed family membership.
_WRAPPER_FAMILY = "phase17_sampled"
_WRAPPER_SPLIT = "phase17"


def _entries(setups: "list[tuple]") -> "list[LibraryEntry]":
    return [
        LibraryEntry(family_id=_WRAPPER_FAMILY, split=_WRAPPER_SPLIT, canonical=tuple(setup))
        for setup in setups
    ]


def _matrix(setups: "list[tuple]") -> np.ndarray:
    matrix = np.array([tuple(setup) for setup in setups], dtype=np.uint8)
    if matrix.ndim != 2 or matrix.shape[1] != CANONICAL_CELLS:
        raise Phase17SetupError("diversity metrics need 40-entry canonical setups")
    return matrix


def shannon_entropy_nats(probabilities: np.ndarray) -> float:
    """Entropy in nats of a probability vector, zeros excluded."""
    values = np.asarray(probabilities, dtype=np.float64)
    values = values[values > 0.0]
    return float(-(values * np.log(values)).sum())


def effective_support(probabilities: np.ndarray) -> float:
    """`exp(H)` -- how many outcomes the distribution behaves like it has.

    The production stop condition is stated in these units (`flag effective
    support below four`), so it is computed as a perplexity rather than as a
    raw count of distinct values: 100 boards that put the flag on square 3
    once and square 7 ninety-nine times have a support of 2 and an effective
    support of 1.06.
    """
    return float(np.exp(shannon_entropy_nats(probabilities)))


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------


def uniqueness_metrics(setups: "list[tuple]") -> dict:
    """Exact and reflection-class uniqueness, collisions, and the top class."""
    if not setups:
        raise Phase17SetupError("uniqueness needs at least one setup")
    exact = Counter(tuple(setup) for setup in setups)
    classes = Counter(class_fingerprint(tuple(setup)) for setup in setups)
    top_class, top_count = classes.most_common(1)[0]
    total = len(setups)
    return {
        "sample_count": total,
        "exact_unique": len(exact),
        "exact_unique_fraction": len(exact) / total,
        "reflection_class_unique": len(classes),
        "reflection_class_unique_fraction": len(classes) / total,
        "exact_collision_rate": 1.0 - len(exact) / total,
        "reflection_class_collision_rate": 1.0 - len(classes) / total,
        "most_frequent_reflection_class": top_class,
        "most_frequent_reflection_class_count": int(top_count),
        "mirrored_pairs_present": len(exact) - len(classes),
    }


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def empirical_entropy_metrics(setups: "list[tuple]") -> dict:
    """Per-square entropy in bits, over the empirical sample."""
    per_square = per_square_entropy_bits(_entries(setups))
    return {
        "per_square_entropy_bits": [float(value) for value in per_square],
        "mean_per_square_entropy_bits": float(np.mean(per_square)),
        "min_per_square_entropy_bits": float(np.min(per_square)),
        "max_per_square_entropy_bits": float(np.max(per_square)),
    }


def prefix_entropy_metrics(behavior_probabilities: np.ndarray) -> dict:
    """Model-side prefix entropy in nats -- the stop policy's own quantity.

    `behavior_probabilities` is `[N, 40, 12]`, the masked distributions the
    sampler actually drew from. Averaging over prefixes is what section 13's
    "setup mean prefix entropy" names; the per-prefix curve is kept because a
    collapse at the early prefixes and a collapse at the late ones are very
    different failures and the mean hides which happened.
    """
    array = np.asarray(behavior_probabilities, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] != SETUP_PREFIXES or array.shape[2] != NUM_PIECE_TYPES:
        raise Phase17SetupError(
            f"expected [N, {SETUP_PREFIXES}, {NUM_PIECE_TYPES}] probabilities, "
            f"got {array.shape}"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(array > 0.0, array * np.log(array), 0.0)
    per_prefix = -terms.sum(axis=2)
    return {
        "mean_prefix_entropy_nats": float(per_prefix.mean()),
        "per_prefix_entropy_nats": [float(value) for value in per_prefix.mean(axis=0)],
        "first_prefix_entropy_nats": float(per_prefix[:, 0].mean()),
        "mean_sequence_entropy_nats": float(per_prefix.sum(axis=1).mean()),
    }


def information_metrics(suffix_information: np.ndarray) -> dict:
    """The realized sequence-information distribution, `I_0` in nats.

    `I_0` is the information content of the whole sampled setup and is the
    quantity the advantage's entropy term is built from, so its spread -- not
    only its mean -- is what tells whether the model is still exploring.
    """
    array = np.asarray(suffix_information, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != SETUP_PREFIXES:
        raise Phase17SetupError(f"expected [N, {SETUP_PREFIXES}] information, got {array.shape}")
    sequence = array[:, 0]
    return {
        "sequence_information_mean_nats": float(sequence.mean()),
        "sequence_information_std_nats": float(sequence.std()),
        "sequence_information_quantiles_nats": {
            str(q): float(np.quantile(sequence, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "sequence_information_min_nats": float(sequence.min()),
        "sequence_information_max_nats": float(sequence.max()),
    }


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------


def _pairwise(setups: "list[tuple]", sample_cap: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    """Vectorized Hamming and class distances over a capped subsample.

    The full matrix is O(N^2); a 5,000-sample gate would build 12.5M pairs per
    metric. A fixed-seed subsample of at most `sample_cap` rows is used
    instead, and the cap is reported so the number is never mistaken for the
    exhaustive one.
    """
    matrix = _matrix(setups)
    if matrix.shape[0] > sample_cap:
        chosen = np.random.RandomState(seed).choice(matrix.shape[0], sample_cap, replace=False)
        matrix = matrix[np.sort(chosen)]
    mirrored = np.array(
        [reflect_canonical(tuple(int(v) for v in row)) for row in matrix], dtype=np.uint8
    )
    count = matrix.shape[0]
    upper = np.triu_indices(count, k=1)
    plain = np.empty(upper[0].size, dtype=np.int16)
    folded = np.empty(upper[0].size, dtype=np.int16)
    cursor = 0
    for row in range(count - 1):
        span = count - row - 1
        direct = (matrix[row][None, :] != matrix[row + 1 :]).sum(axis=1)
        flipped = (matrix[row][None, :] != mirrored[row + 1 :]).sum(axis=1)
        plain[cursor : cursor + span] = direct
        folded[cursor : cursor + span] = np.minimum(direct, flipped)
        cursor += span
    return plain, folded


def distance_metrics(setups: "list[tuple]", *, sample_cap: int = 600, seed: int = 17) -> dict:
    """Mean and quantile pairwise Hamming and reflection-class distance."""
    if len(setups) < 2:
        raise Phase17SetupError("distance metrics need at least two setups")
    plain, folded = _pairwise(setups, sample_cap, seed)
    quantiles = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95)
    return {
        "distance_sample_cap": sample_cap,
        "distance_pairs": int(plain.size),
        "mean_hamming": float(plain.mean()),
        "min_hamming": int(plain.min()),
        "hamming_quantiles": {str(q): float(np.quantile(plain, q)) for q in quantiles},
        "mean_class_distance": float(folded.mean()),
        "min_class_distance": int(folded.min()),
        "class_distance_quantiles": {str(q): float(np.quantile(folded, q)) for q in quantiles},
        "near_duplicate_pair_fraction": float((folded < 10).mean()),
    }


# ---------------------------------------------------------------------------
# Piece placement support
# ---------------------------------------------------------------------------


def _square_distribution(matrix: np.ndarray, piece_type: int) -> np.ndarray:
    """Fraction of placements of `piece_type` landing on each canonical square."""
    counts = (matrix == piece_type).sum(axis=0).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        raise Phase17SetupError(f"no placements of piece type {piece_type} in the sample")
    return counts / total


def placement_metrics(setups: "list[tuple]") -> dict:
    """Flag and bomb support, effective support, and token concentration."""
    matrix = _matrix(setups)
    entries = _entries(setups)

    flag = _square_distribution(matrix, FLAG)
    bomb = _square_distribution(matrix, BOMB)

    bomb_patterns = Counter(
        tuple(sorted(int(cell) for cell in np.nonzero(np.asarray(setup) == BOMB)[0]))
        for setup in setups
    )
    top_bomb_pattern, top_bomb_count = bomb_patterns.most_common(1)[0]

    # Per-square modal share, averaged: 1.0 means every board agrees on every
    # square, which is total collapse; 1/12 is the unconstrained floor.
    modal_share = np.array(
        [
            np.bincount(matrix[:, cell], minlength=NUM_PIECE_TYPES).max() / matrix.shape[0]
            for cell in range(CANONICAL_CELLS)
        ],
        dtype=np.float64,
    )

    return {
        "flag_square_support": int((flag > 0).sum()),
        "flag_effective_support": effective_support(flag),
        "flag_folded_support": folded_support(entries, (FLAG,)),
        "flag_top_square_share": float(flag.max()),
        "flag_square_distribution": [float(value) for value in flag],
        "bomb_square_support": int((bomb > 0).sum()),
        "bomb_effective_support": effective_support(bomb),
        "bomb_folded_support": folded_support(entries, (BOMB,)),
        "bomb_pattern_unique": len(bomb_patterns),
        "bomb_pattern_top_count": int(top_bomb_count),
        "bomb_pattern_top": list(top_bomb_pattern),
        "mean_top_token_concentration": float(modal_share.mean()),
        "max_top_token_concentration": float(modal_share.max()),
    }


# ---------------------------------------------------------------------------
# The full profile and its calibrated thresholds
# ---------------------------------------------------------------------------


def diversity_profile(
    setups: "list[tuple]",
    *,
    behavior_probabilities: np.ndarray | None = None,
    suffix_information: np.ndarray | None = None,
    label: str = "sample",
    distance_sample_cap: int = 600,
) -> dict:
    """Every section 7 diversity measurement over one sample."""
    profile = {
        "label": label,
        **uniqueness_metrics(setups),
        **empirical_entropy_metrics(setups),
        **distance_metrics(setups, sample_cap=distance_sample_cap),
        **placement_metrics(setups),
    }
    if behavior_probabilities is not None:
        profile.update(prefix_entropy_metrics(behavior_probabilities))
    if suffix_information is not None:
        profile.update(information_metrics(suffix_information))
    return profile


@dataclass(frozen=True)
class DiversityAlarms:
    """Production warning/hard thresholds, calibrated from a baseline profile.

    The hard floors are the common contract's provisional ones expressed
    against *this run's* measured baseline, never against a library standard.
    The warning band sits at 80% so a drift is visible before the stop
    condition fires.
    """

    baseline_label: str
    baseline_mean_prefix_entropy_nats: float
    baseline_flag_effective_support: float
    baseline_mean_class_distance: float
    baseline_reflection_class_unique_fraction: float
    hard_mean_prefix_entropy_nats: float
    warn_mean_prefix_entropy_nats: float
    hard_flag_effective_support: float
    warn_flag_effective_support: float
    consecutive_checks: int

    @classmethod
    def from_baseline(cls, profile: dict) -> "DiversityAlarms":
        entropy = float(profile["mean_prefix_entropy_nats"])
        flag = float(profile["flag_effective_support"])
        return cls(
            baseline_label=profile.get("label", "baseline"),
            baseline_mean_prefix_entropy_nats=entropy,
            baseline_flag_effective_support=flag,
            baseline_mean_class_distance=float(profile["mean_class_distance"]),
            baseline_reflection_class_unique_fraction=float(
                profile["reflection_class_unique_fraction"]
            ),
            hard_mean_prefix_entropy_nats=entropy * PROVISIONAL_PREFIX_ENTROPY_FLOOR_FRACTION,
            warn_mean_prefix_entropy_nats=entropy * 0.80,
            hard_flag_effective_support=PROVISIONAL_FLAG_EFFECTIVE_SUPPORT_FLOOR,
            warn_flag_effective_support=PROVISIONAL_FLAG_EFFECTIVE_SUPPORT_FLOOR * 1.5,
            consecutive_checks=PROVISIONAL_PREFIX_ENTROPY_FLOOR_CONSECUTIVE_CHECKS,
        )

    def evaluate(self, profile: dict) -> dict:
        """Classify one later profile as ok / warning / hard."""
        entropy = float(profile["mean_prefix_entropy_nats"])
        flag = float(profile["flag_effective_support"])
        checks = {
            "mean_prefix_entropy_nats": {
                "observed": entropy,
                "warn_below": self.warn_mean_prefix_entropy_nats,
                "hard_below": self.hard_mean_prefix_entropy_nats,
                "status": "hard"
                if entropy < self.hard_mean_prefix_entropy_nats
                else "warning"
                if entropy < self.warn_mean_prefix_entropy_nats
                else "ok",
            },
            "flag_effective_support": {
                "observed": flag,
                "warn_below": self.warn_flag_effective_support,
                "hard_below": self.hard_flag_effective_support,
                "status": "hard"
                if flag < self.hard_flag_effective_support
                else "warning"
                if flag < self.warn_flag_effective_support
                else "ok",
            },
        }
        statuses = [check["status"] for check in checks.values()]
        return {
            "label": profile.get("label", "sample"),
            "checks": checks,
            "status": "hard" if "hard" in statuses else "warning" if "warning" in statuses else "ok",
            "consecutive_checks_required": self.consecutive_checks,
        }

    def document(self) -> dict:
        return {
            "calibration_source": "initial masked model and the Agent 3 soak",
            "must_not_borrow": "frozen-library family thresholds",
            **self.__dict__,
        }


__all__ = [
    "DiversityAlarms",
    "class_distance",
    "distance_metrics",
    "diversity_profile",
    "effective_support",
    "empirical_entropy_metrics",
    "hamming_distance",
    "information_metrics",
    "placement_metrics",
    "prefix_entropy_metrics",
    "shannon_entropy_nats",
    "uniqueness_metrics",
]
