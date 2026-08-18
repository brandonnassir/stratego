"""Phase 10 Agent 3: the independent audit of the two fitted utility models.

Specification sources:

- `03_AGENT_3_UTILITY_MODELS_AND_AUDIT.md` ("Feature reconstruction",
  "Independent audit", "Deterministic refit", "Negative controls")

Independence, concretely
------------------------
Everything this module verifies it recomputes through its own code path:

- trait vectors are rebuilt from each base's stored *placement*
  (`canonical_setup`) via the frozen `compute_trait_vector`, never trusted
  from the library's stored `trait_vector` field (the two are compared);
- the 47-scalar flattening is re-derived here from `TRAIT_SCHEMA` and its
  name order compared against the frozen feature-name list;
- the standardizer is recomputed from the reconstructed matrix of **all
  6,400 train bases** with plain numpy `mean`/`std(ddof=0)` and compared to
  the frozen literals elementwise;
- targets are rebuilt from the stored W/D/L token through this module's own
  copy of the frozen mapping; orientation is re-derived from the game id;
- logits, sigmoid probabilities, BCE, L2, the full objective, the family
  centering and the analytic gradient are all computed in numpy from the
  exported coefficients, without calling
  :mod:`stratego.training.phase10_utility_fit` for any verified quantity.

Frozen tolerances
-----------------
Every comparison tolerance is frozen here, before any comparison runs, per
the instruction to never invent a tolerance after seeing a difference:

- reconstruction, scaler moments, targets, family indices: **exact**;
- audit-vs-reported objective and per-record logits: ``1e-10`` — two
  mathematically identical float64 expressions summed in different orders
  over 16,384 terms have a naive worst-case error bound near ``4e-12``
  (``N * eps * |term|``); ``1e-10`` gives an order of headroom while
  sitting five orders below the smallest difference any negative control
  produces;
- analytic-vs-central-difference gradient: ``1e-6 + 1e-6 * |value|`` with
  step ``1e-6`` — the central-difference truncation plus float64 rounding
  error for an objective of magnitude ~0.7 is below ``1e-9``;
- stationarity of the fitted point: gradient max-abs ``<= 1e-6`` — the
  frozen optimizer stops at gradient ``1e-10`` or change ``1e-12``, so a
  genuinely fitted point sits far inside this;
- deterministic refits: **bit-exact equality** of the canonical coefficient
  document, justified because the fit is single-threaded CPU float64 with a
  frozen reduction order, which this repository has already measured to be
  bit-reproducible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from ..setups.contracts import SPLIT_TRAIN, TRAIN_TOTAL
from ..setups.families import FAMILY_IDS
from ..setups.traits import TRAIT_SCHEMA, TRAIT_SCHEMA_VERSION, compute_trait_vector
from .phase10_schedule import (
    GAMES_PER_ORDERED_PAIR,
    ORDERED_FAMILY_PAIRS,
    TOTAL_CORPUS_GAMES,
    rebuild_game,
)
from .phase10_seed import parse_phase10_game_id

#: Frozen comparison tolerances (see the module docstring for justification).
LOGIT_AGREEMENT_TOLERANCE = 1e-10
OBJECTIVE_AGREEMENT_TOLERANCE = 1e-10
GRADIENT_STATIONARITY_TOLERANCE = 1e-6
GRADIENT_FD_STEP = 1e-6
GRADIENT_FD_ABS_TOLERANCE = 1e-6
GRADIENT_FD_REL_TOLERANCE = 1e-6
CENTERING_TOLERANCE = 1e-8

#: The audit's own copy of the frozen Red-perspective target mapping. Kept
#: separate from the production constants on purpose: if either drifts, the
#: record audit fails instead of both moving together.
AUDIT_RESULT_TARGETS = {"red_win": 1.0, "draw": 0.5, "red_loss": 0.0}
AUDIT_RESULT_WINNER = {"red_win": "red", "draw": None, "red_loss": "blue"}

#: The frozen L2 coefficient, restated for the same reason.
AUDIT_L2_LAMBDA = 1e-3


class Phase10UtilityAuditError(RuntimeError):
    """An audit precondition failed (not a finding — a broken audit input)."""


def _canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Independent feature reconstruction
# ---------------------------------------------------------------------------


def independent_feature_names() -> "tuple[str, ...]":
    """The 47 feature names re-derived from `TRAIT_SCHEMA`, independently."""
    names: list[str] = []
    for field in TRAIT_SCHEMA:
        if field.kind == "int_list4":
            names.extend(f"{field.name}[{position}]" for position in range(4))
        else:
            names.append(field.name)
    return tuple(names)


def independent_flatten(trait_vector: dict) -> "tuple[float, ...]":
    """This module's own 35-field -> 47-scalar flattening."""
    values: list[float] = []
    for field in TRAIT_SCHEMA:
        raw = trait_vector[field.name]
        if field.kind == "int_list4":
            if len(raw) != 4:
                raise Phase10UtilityAuditError(
                    f"{field.name}: histogram has {len(raw)} components"
                )
            values.extend(float(component) for component in raw)
        else:
            values.append(float(raw))
    return tuple(values)


@dataclass(frozen=True)
class ReconstructedLibrary:
    """Every base's independently rebuilt traits, plus split matrices."""

    feature_names: "tuple[str, ...]"
    train_base_ids: "tuple[str, ...]"
    train_families: "tuple[str, ...]"
    train_matrix: np.ndarray  # (6400, 47) float64, canonical library order
    base_trait_digest: dict  # base_setup_id -> sha256 of recomputed trait vector
    base_family: dict  # base_setup_id -> family_id
    base_split: dict  # base_setup_id -> split
    base_features: dict  # base_setup_id -> raw (unstandardized) 47-tuple
    stored_trait_mismatches: "tuple[str, ...]"


def reconstruct_library(entries) -> ReconstructedLibrary:
    """Rebuild every entry's trait vector from its placement and index it.

    ``entries`` is the full library (all splits): corpus verification needs
    train bases, and the held-out-scaler negative control needs validation
    bases, so both are reconstructed once here. The stored `trait_vector`
    field is compared against the recomputation and never used further.
    """
    order = {family_id: position for position, family_id in enumerate(FAMILY_IDS)}
    names = independent_feature_names()

    mismatches: list[str] = []
    digests: dict[str, str] = {}
    families: dict[str, str] = {}
    splits: dict[str, str] = {}
    features: dict[str, tuple] = {}

    train_entries = []
    for entry in entries:
        recomputed = compute_trait_vector(entry.canonical_setup)
        if dict(entry.trait_vector) != recomputed:
            mismatches.append(entry.base_setup_id)
        digests[entry.base_setup_id] = hashlib.sha256(
            _canonical_json(recomputed).encode()
        ).hexdigest()
        families[entry.base_setup_id] = entry.family_id
        splits[entry.base_setup_id] = entry.split
        features[entry.base_setup_id] = independent_flatten(recomputed)
        if entry.split == SPLIT_TRAIN:
            train_entries.append(entry)

    if len(train_entries) != TRAIN_TOTAL:
        raise Phase10UtilityAuditError(
            f"library holds {len(train_entries)} train bases, expected {TRAIN_TOTAL}"
        )
    train_entries.sort(key=lambda entry: (order[entry.family_id], entry.base_index))
    matrix = np.array(
        [features[entry.base_setup_id] for entry in train_entries], dtype=np.float64
    )
    return ReconstructedLibrary(
        feature_names=names,
        train_base_ids=tuple(entry.base_setup_id for entry in train_entries),
        train_families=tuple(entry.family_id for entry in train_entries),
        train_matrix=matrix,
        base_trait_digest=digests,
        base_family=families,
        base_split=splits,
        base_features=features,
        stored_trait_mismatches=tuple(mismatches),
    )


def independent_scaler_moments(matrix: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Population mean/std (ddof=0) of a feature matrix, in numpy."""
    return matrix.mean(axis=0), matrix.std(axis=0, ddof=0)


def standardize(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """The frozen zero-std-to-constant-0 standardization, re-implemented."""
    divisor = np.where(std == 0.0, 1.0, std)
    standardized = (np.asarray(matrix, dtype=np.float64) - mean) / divisor
    return np.where(std == 0.0, 0.0, standardized)


# ---------------------------------------------------------------------------
# Independent record audit and design reconstruction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditDesign:
    """The audit's own design matrices, rebuilt without the production helper."""

    game_ids: "tuple[str, ...]"
    targets: np.ndarray
    red_family_index: np.ndarray
    blue_family_index: np.ndarray
    red_features: np.ndarray  # (N, 47) standardized
    blue_features: np.ndarray


def audit_corpus_records(
    records,
    library: ReconstructedLibrary,
    *,
    expected_digests: dict,
    frozen_mean: np.ndarray,
    frozen_std: np.ndarray,
) -> "tuple[dict, AuditDesign]":
    """Every record re-derived and cross-checked; the audit design built.

    Verifies, for all 16,384 records: game-id parse and schedule rebuild
    (families, ordinal, match seed, split), result-token / winner /
    red_score / target consistency (the frozen draw handling included),
    Red/Blue orientation (the design row is +Red, -Blue by construction
    here and compared to production elsewhere), base family/split via the
    reconstructed library, recorded trait identity digests against the
    recomputed ones, and the four identity digests every record carries.

    ``expected_digests`` maps record fields (`library_content_digest`,
    `corpus_contract_digest`, `outcome_schedule_digest`,
    `contract_bundle_digest`) to their accepted values.
    """
    family_position = {family_id: index for index, family_id in enumerate(FAMILY_IDS)}
    violations: list[str] = []
    pair_counts: dict = {}
    result_counts = {"red_win": 0, "draw": 0, "red_loss": 0}

    game_ids: list[str] = []
    targets: list[float] = []
    red_index: list[int] = []
    blue_index: list[int] = []
    red_rows: list[tuple] = []
    blue_rows: list[tuple] = []

    def flag(condition: bool, message: str) -> None:
        if not condition and len(violations) < 64:
            violations.append(message)

    previous_id = ""
    total = 0
    for record in records:
        total += 1
        game_id = record["game_id"]
        flag(game_id > previous_id, f"{game_id}: not in canonical order")
        previous_id = game_id

        parsed = parse_phase10_game_id(game_id)
        game = rebuild_game(game_id)
        flag(record["red_family"] == parsed["red_family"], f"{game_id}: red family != id")
        flag(record["blue_family"] == parsed["blue_family"], f"{game_id}: blue family != id")
        flag(record["ordinal"] == parsed["ordinal"], f"{game_id}: ordinal != id")
        flag(record["match_seed"] == game.match_seed, f"{game_id}: match seed drifted")
        flag(record["split"] == SPLIT_TRAIN, f"{game_id}: split {record['split']!r}")

        result = record["result"]
        flag(result in AUDIT_RESULT_TARGETS, f"{game_id}: unknown result {result!r}")
        if result in AUDIT_RESULT_TARGETS:
            result_counts[result] += 1
            target = AUDIT_RESULT_TARGETS[result]
            flag(
                float(record["red_score"]) == target,
                f"{game_id}: red_score {record['red_score']} != frozen target {target}",
            )
            flag(
                record["winner"] == AUDIT_RESULT_WINNER[result],
                f"{game_id}: winner {record['winner']!r} inconsistent with {result!r}",
            )
        else:  # pragma: no cover - flagged above, keep the row well-typed
            target = 0.5

        for name, expected in expected_digests.items():
            flag(record[name] == expected, f"{game_id}: {name} drifted")
        flag(
            record["trait_schema_version"] == TRAIT_SCHEMA_VERSION,
            f"{game_id}: trait schema {record['trait_schema_version']!r}",
        )

        pair = (record["red_family"], record["blue_family"])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

        row: dict = {}
        for color in ("red", "blue"):
            base_id = record[f"{color}_base_setup_id"]
            flag(
                library.base_split.get(base_id) == SPLIT_TRAIN,
                f"{game_id}: {color} base {base_id} split "
                f"{library.base_split.get(base_id)!r}",
            )
            flag(
                library.base_family.get(base_id) == record[f"{color}_family"],
                f"{game_id}: {color} base {base_id} family disagrees with record",
            )
            identity = record[f"{color}_trait_identity"]
            flag(
                identity["base_trait_digest"] == library.base_trait_digest.get(base_id),
                f"{game_id}: {color} recorded base trait digest != recomputed",
            )
            flag(
                identity["trait_schema_version"] == TRAIT_SCHEMA_VERSION,
                f"{game_id}: {color} trait identity schema version",
            )
            row[color] = library.base_features[base_id]

        game_ids.append(game_id)
        targets.append(target)
        red_index.append(family_position[record["red_family"]])
        blue_index.append(family_position[record["blue_family"]])
        red_rows.append(row["red"])
        blue_rows.append(row["blue"])

    checks = {
        "total_records_exact": total == TOTAL_CORPUS_GAMES,
        "ordered_pairs_complete": len(pair_counts) == ORDERED_FAMILY_PAIRS,
        "games_per_pair_exact": all(
            count == GAMES_PER_ORDERED_PAIR for count in pair_counts.values()
        ),
        "no_violations": not violations,
        "result_counts_total": sum(result_counts.values()) == total,
    }
    summary = {
        "records_audited": total,
        "ordered_pairs": len(pair_counts),
        "result_counts": dict(result_counts),
        "violations": violations,
        "checks": checks,
        "all_pass": all(checks.values()),
    }
    design = AuditDesign(
        game_ids=tuple(game_ids),
        targets=np.asarray(targets, dtype=np.float64),
        red_family_index=np.asarray(red_index, dtype=np.int64),
        blue_family_index=np.asarray(blue_index, dtype=np.int64),
        red_features=standardize(np.array(red_rows, dtype=np.float64), frozen_mean, frozen_std),
        blue_features=standardize(np.array(blue_rows, dtype=np.float64), frozen_mean, frozen_std),
    )
    return summary, design


# ---------------------------------------------------------------------------
# Independent objective, gradient and model audit
# ---------------------------------------------------------------------------


def _coefficients(fitted: dict) -> dict:
    """Numpy views of one exported model's coefficients (audit's parsing)."""
    return {
        "model_id": fitted["model_id"],
        "intercept": float(fitted["red_first_intercept"]),
        "offsets_raw": np.asarray(fitted["family_offsets_raw"], dtype=np.float64),
        "weights": (
            None
            if fitted["trait_weights"] is None
            else np.asarray(fitted["trait_weights"], dtype=np.float64)
        ),
    }


def audit_logits(design: AuditDesign, fitted: dict) -> np.ndarray:
    """`eta` recomputed from the exported coefficients, in numpy.

    Centering is re-derived here: `b_eff[c] = b_raw[c] - mean(b_raw[c])`,
    and the game logit is `intercept + u(red setup, red) - u(blue setup,
    blue)` — the frozen sign and order.
    """
    coefficients = _coefficients(fitted)
    offsets = coefficients["offsets_raw"]
    effective = offsets - offsets.mean(axis=1, keepdims=True)
    eta = (
        coefficients["intercept"]
        + effective[0][design.red_family_index]
        - effective[1][design.blue_family_index]
    )
    if coefficients["weights"] is not None:
        eta = (
            eta
            + design.red_features @ coefficients["weights"][0]
            - design.blue_features @ coefficients["weights"][1]
        )
    return eta


def audit_objective(design: AuditDesign, fitted: dict) -> dict:
    """BCE, L2 and the full objective recomputed independently."""
    eta = audit_logits(design, fitted)
    y = design.targets
    bce = float(np.mean(y * np.logaddexp(0.0, -eta) + (1.0 - y) * np.logaddexp(0.0, eta)))
    coefficients = _coefficients(fitted)
    penalty = float(np.sum(coefficients["offsets_raw"] ** 2))
    if coefficients["weights"] is not None:
        penalty += float(np.sum(coefficients["weights"] ** 2))
    l2 = AUDIT_L2_LAMBDA * penalty
    probabilities = 1.0 / (1.0 + np.exp(-eta))
    return {
        "bce": bce,
        "l2_penalty": l2,
        "objective": bce + l2,
        "logits_finite": bool(np.isfinite(eta).all()),
        "probabilities_finite": bool(np.isfinite(probabilities).all()),
        "probabilities_in_unit_interval": bool(
            (probabilities > 0.0).all() and (probabilities < 1.0).all()
        ),
        "logit_max_abs": float(np.abs(eta).max()),
    }


def audit_gradient(design: AuditDesign, fitted: dict) -> dict:
    """The analytic objective gradient at the fitted point, in numpy.

    Derivation: with `p = sigmoid(eta)` and soft target `y`,
    `dBCE/deta_i = (p_i - y_i) / N`. The chain rule through the centering
    map subtracts each colour row's mean contribution, and the L2 term adds
    `2 * lambda * theta` on raw offsets and weights (never the intercept).
    """
    coefficients = _coefficients(fitted)
    eta = audit_logits(design, fitted)
    n = float(len(design.targets))
    residual = (1.0 / (1.0 + np.exp(-eta)) - design.targets) / n

    gradient_intercept = float(residual.sum())

    family_count = len(FAMILY_IDS)
    red_counts = np.zeros(family_count)
    blue_counts = np.zeros(family_count)
    np.add.at(red_counts, design.red_family_index, residual)
    np.add.at(blue_counts, design.blue_family_index, residual)
    # d eta / d b_raw[0, f] = [red_family == f] - 1/16 (centering); the
    # -1/16 term sums residuals over every game, once per family.
    gradient_offsets = np.zeros((2, family_count))
    gradient_offsets[0] = red_counts - residual.sum() / family_count
    gradient_offsets[1] = -(blue_counts - residual.sum() / family_count)
    gradient_offsets += 2.0 * AUDIT_L2_LAMBDA * coefficients["offsets_raw"]

    pieces = [np.array([gradient_intercept]), gradient_offsets.ravel()]
    if coefficients["weights"] is not None:
        gradient_weights = np.vstack(
            [
                design.red_features.T @ residual,
                -(design.blue_features.T @ residual),
            ]
        )
        gradient_weights += 2.0 * AUDIT_L2_LAMBDA * coefficients["weights"]
        pieces.append(gradient_weights.ravel())
    flat = np.concatenate(pieces)
    return {
        "max_abs": float(np.abs(flat).max()),
        "l2_norm": float(np.linalg.norm(flat)),
        "finite": bool(np.isfinite(flat).all()),
        "stationary": bool(np.abs(flat).max() <= GRADIENT_STATIONARITY_TOLERANCE),
        "flat": flat,
    }


def _perturbed_objective(design: AuditDesign, fitted: dict, coordinate: str, index, delta: float) -> float:
    """Objective with one named coefficient nudged — for finite differences."""
    perturbed = json.loads(json.dumps(fitted))
    if coordinate == "intercept":
        perturbed["red_first_intercept"] = float(perturbed["red_first_intercept"]) + delta
    elif coordinate == "offset":
        colour, family = index
        perturbed["family_offsets_raw"][colour][family] += delta
    elif coordinate == "weight":
        colour, feature = index
        perturbed["trait_weights"][colour][feature] += delta
    else:  # pragma: no cover - internal misuse
        raise Phase10UtilityAuditError(f"unknown coordinate kind {coordinate!r}")
    return audit_objective(design, perturbed)["objective"]


def finite_difference_spot_checks(design: AuditDesign, fitted: dict, gradient: dict) -> dict:
    """Central-difference checks of the analytic gradient at sampled coords."""
    family_count = len(FAMILY_IDS)
    coordinates = [("intercept", None, 0)]
    for colour, family in ((0, 0), (0, 7), (1, 3), (1, 15)):
        coordinates.append(("offset", (colour, family), 1 + colour * family_count + family))
    if fitted["trait_weights"] is not None:
        offset_block = 1 + 2 * family_count
        feature_count = len(fitted["trait_weights"][0])
        for colour, feature in ((0, 0), (0, 21), (1, 5), (1, 46)):
            coordinates.append(
                ("weight", (colour, feature), offset_block + colour * feature_count + feature)
            )

    flat = gradient["flat"]
    checks = []
    worst = 0.0
    for kind, index, position in coordinates:
        plus = _perturbed_objective(design, fitted, kind, index, +GRADIENT_FD_STEP)
        minus = _perturbed_objective(design, fitted, kind, index, -GRADIENT_FD_STEP)
        estimate = (plus - minus) / (2.0 * GRADIENT_FD_STEP)
        analytic = float(flat[position])
        difference = abs(analytic - estimate)
        allowed = GRADIENT_FD_ABS_TOLERANCE + GRADIENT_FD_REL_TOLERANCE * abs(estimate)
        worst = max(worst, difference)
        checks.append(
            {
                "coordinate": f"{kind}{'' if index is None else list(index)}",
                "analytic": analytic,
                "finite_difference": estimate,
                "abs_difference": difference,
                "within_tolerance": bool(difference <= allowed),
            }
        )
    return {
        "step": GRADIENT_FD_STEP,
        "checks": checks,
        "worst_abs_difference": worst,
        "all_within_tolerance": all(entry["within_tolerance"] for entry in checks),
    }


def audit_fitted_model(design: AuditDesign, fitted: dict) -> dict:
    """The complete independent audit of one exported model."""
    coefficients = _coefficients(fitted)
    finite = (
        np.isfinite(coefficients["intercept"])
        and bool(np.isfinite(coefficients["offsets_raw"]).all())
        and (
            coefficients["weights"] is None
            or bool(np.isfinite(coefficients["weights"]).all())
        )
    )
    objective = audit_objective(design, fitted)
    gradient = audit_gradient(design, fitted)
    spot = finite_difference_spot_checks(design, fitted, gradient)

    reported = fitted["diagnostics"]
    objective_difference = abs(objective["objective"] - reported["objective"])
    bce_difference = abs(objective["bce"] - reported["bce"])
    l2_difference = abs(objective["l2_penalty"] - reported["l2_penalty"])

    raw_means = coefficients["offsets_raw"].mean(axis=1)
    effective = np.asarray(fitted["family_offsets_effective"], dtype=np.float64)
    expected_effective = coefficients["offsets_raw"] - coefficients["offsets_raw"].mean(
        axis=1, keepdims=True
    )

    checks = {
        "coefficients_finite": bool(finite),
        "objective_finite": bool(np.isfinite(objective["objective"])),
        "logits_finite": objective["logits_finite"],
        "probabilities_finite": objective["probabilities_finite"],
        "probabilities_in_unit_interval": objective["probabilities_in_unit_interval"],
        "objective_agrees": bool(objective_difference <= OBJECTIVE_AGREEMENT_TOLERANCE),
        "bce_agrees": bool(bce_difference <= OBJECTIVE_AGREEMENT_TOLERANCE),
        "l2_agrees": bool(l2_difference <= OBJECTIVE_AGREEMENT_TOLERANCE),
        "raw_offsets_centered": bool(np.abs(raw_means).max() <= CENTERING_TOLERANCE),
        "effective_offsets_match_centering": bool(
            np.abs(effective - expected_effective).max() <= 1e-15
        ),
        "gradient_finite": gradient["finite"],
        "gradient_stationary": gradient["stationary"],
        "finite_differences_agree": spot["all_within_tolerance"],
    }
    return {
        "model_id": fitted["model_id"],
        "independent_objective": {key: objective[key] for key in ("bce", "l2_penalty", "objective")},
        "reported_objective": {
            "bce": reported["bce"],
            "l2_penalty": reported["l2_penalty"],
            "objective": reported["objective"],
        },
        "objective_abs_difference": objective_difference,
        "logit_max_abs": objective["logit_max_abs"],
        "raw_offset_means": [float(value) for value in raw_means],
        "gradient": {key: gradient[key] for key in ("max_abs", "l2_norm", "finite", "stationary")},
        "finite_difference": spot,
        "tolerances": {
            "objective": OBJECTIVE_AGREEMENT_TOLERANCE,
            "logit": LOGIT_AGREEMENT_TOLERANCE,
            "stationarity": GRADIENT_STATIONARITY_TOLERANCE,
            "centering": CENTERING_TOLERANCE,
        },
        "checks": checks,
        "all_pass": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# Deterministic refit comparison
# ---------------------------------------------------------------------------


def compare_refits(coefficient_documents: "list[dict]") -> dict:
    """Bit-exact agreement across independent fits of one model.

    The frozen criterion is exact equality of the canonical coefficient
    JSON. The maximum absolute coefficient difference is reported as
    evidence (0.0 when the criterion holds), never used to loosen it.
    """
    if len(coefficient_documents) < 2:
        raise Phase10UtilityAuditError("refit comparison needs at least two fits")
    canonical = [_canonical_json(document) for document in coefficient_documents]
    identical = all(text == canonical[0] for text in canonical[1:])

    def flatten(document: dict) -> np.ndarray:
        values = [float(document["red_first_intercept"])]
        for row in document["family_offsets_raw"]:
            values.extend(float(value) for value in row)
        if document["trait_weights"] is not None:
            for row in document["trait_weights"]:
                values.extend(float(value) for value in row)
        return np.asarray(values, dtype=np.float64)

    first = flatten(coefficient_documents[0])
    max_difference = 0.0
    for document in coefficient_documents[1:]:
        other = flatten(document)
        if other.shape != first.shape:
            return {"identical": False, "max_abs_difference": float("inf"), "fits": len(canonical)}
        max_difference = max(max_difference, float(np.abs(other - first).max()))
    return {
        "identical": bool(identical),
        "max_abs_difference": max_difference,
        "fits": len(canonical),
        "criterion": "bit-exact canonical coefficient JSON equality, frozen before comparison",
    }


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


def _detected_objective_shift(design: AuditDesign, fitted: dict, corrupted: AuditDesign) -> dict:
    """Whether the audit's agreement checks catch a corrupted design."""
    true_eta = audit_logits(design, fitted)
    wrong_eta = audit_logits(corrupted, fitted)
    logit_shift = float(np.abs(true_eta - wrong_eta).max())
    wrong_objective = audit_objective(corrupted, fitted)["objective"]
    reported = fitted["diagnostics"]["objective"]
    objective_shift = abs(wrong_objective - reported)
    return {
        "logit_max_shift": logit_shift,
        "objective_abs_difference": objective_shift,
        "detected": bool(
            objective_shift > OBJECTIVE_AGREEMENT_TOLERANCE
            and logit_shift > LOGIT_AGREEMENT_TOLERANCE
        ),
    }


def control_orientation_swap(design: AuditDesign, fitted: dict) -> dict:
    """Reverse the Red/Blue pair orientation; the agreement audit must fail."""
    swapped = AuditDesign(
        game_ids=design.game_ids,
        targets=design.targets,
        red_family_index=design.blue_family_index,
        blue_family_index=design.red_family_index,
        red_features=design.blue_features,
        blue_features=design.red_features,
    )
    outcome = _detected_objective_shift(design, fitted, swapped)
    return {"control": "orientation_swap", **outcome}


def control_wrong_draw_target(design: AuditDesign, fitted: dict, records) -> dict:
    """Score draws 0.0 instead of the frozen 0.5; both audits must catch it."""
    wrong_targets = np.where(design.targets == 0.5, 0.0, design.targets)
    draws = int((design.targets == 0.5).sum())
    mismatches = 0
    for record, wrong in zip(records, wrong_targets):
        if AUDIT_RESULT_TARGETS[record["result"]] != wrong:
            mismatches += 1
    corrupted = AuditDesign(
        game_ids=design.game_ids,
        targets=wrong_targets,
        red_family_index=design.red_family_index,
        blue_family_index=design.blue_family_index,
        red_features=design.red_features,
        blue_features=design.blue_features,
    )
    wrong_objective = audit_objective(corrupted, fitted)["objective"]
    objective_shift = abs(wrong_objective - fitted["diagnostics"]["objective"])
    return {
        "control": "wrong_draw_target",
        "draws_in_corpus": draws,
        "target_mismatches_detected": mismatches,
        "objective_abs_difference": objective_shift,
        "detected": bool(
            draws > 0
            and mismatches == draws
            and objective_shift > OBJECTIVE_AGREEMENT_TOLERANCE
        ),
    }


def control_held_out_scaler(
    library: ReconstructedLibrary,
    entries,
    *,
    frozen_mean: np.ndarray,
    frozen_std: np.ndarray,
    frozen_digest: str,
) -> dict:
    """Standardize with validation-split statistics; the identity must fail.

    Reads held-out bases' *structural* trait vectors only — no outcome, no
    strength signal — exactly like the frozen bank construction already
    does. The wrong scaler is built, detected and discarded.
    """
    from .phase10_utility import TraitScaler

    validation_rows = [
        library.base_features[entry.base_setup_id]
        for entry in entries
        if entry.split == "validation"
    ]
    matrix = np.asarray(validation_rows, dtype=np.float64)
    mean, std = independent_scaler_moments(matrix)
    wrong_digest = TraitScaler(mean, std, base_count=matrix.shape[0], split="validation").digest()
    return {
        "control": "held_out_scaler",
        "validation_bases": int(matrix.shape[0]),
        "mean_max_abs_difference": float(np.abs(mean - frozen_mean).max()),
        "std_max_abs_difference": float(np.abs(std - frozen_std).max()),
        "wrong_digest": wrong_digest,
        "detected": bool(
            wrong_digest != frozen_digest
            and float(np.abs(mean - frozen_mean).max()) > 0.0
        ),
    }


def control_permuted_trait_column(design: AuditDesign, fitted: dict) -> dict:
    """Swap two feature columns; order-sensitive checks must fire."""
    if fitted["trait_weights"] is None:
        raise Phase10UtilityAuditError("the permutation control targets model_T")
    permutation = np.arange(design.red_features.shape[1])
    permutation[[0, 1]] = permutation[[1, 0]]
    corrupted = AuditDesign(
        game_ids=design.game_ids,
        targets=design.targets,
        red_family_index=design.red_family_index,
        blue_family_index=design.blue_family_index,
        red_features=design.red_features[:, permutation],
        blue_features=design.blue_features[:, permutation],
    )
    outcome = _detected_objective_shift(design, fitted, corrupted)
    names = list(independent_feature_names())
    names[0], names[1] = names[1], names[0]
    outcome["feature_order_mismatch_detected"] = tuple(names) != independent_feature_names()
    outcome["detected"] = bool(outcome["detected"] and outcome["feature_order_mismatch_detected"])
    return {"control": "permuted_trait_column", **outcome}


def control_altered_family_id(
    records,
    library: ReconstructedLibrary,
    *,
    expected_digests: dict,
    frozen_mean: np.ndarray,
    frozen_std: np.ndarray,
) -> dict:
    """Alter one record's family id; the record audit must report it."""
    corrupted = [dict(record) for record in records]
    original = corrupted[0]["red_family"]
    position = FAMILY_IDS.index(original)
    corrupted[0] = dict(corrupted[0])
    corrupted[0]["red_family"] = FAMILY_IDS[(position + 1) % len(FAMILY_IDS)]
    summary, _design = audit_corpus_records(
        corrupted,
        library,
        expected_digests=expected_digests,
        frozen_mean=frozen_mean,
        frozen_std=frozen_std,
    )
    return {
        "control": "altered_family_id",
        "violations_reported": len(summary["violations"]),
        "detected": bool(not summary["all_pass"] and summary["violations"]),
    }


def control_altered_coefficient(design: AuditDesign, fitted: dict) -> dict:
    """Nudge one coefficient; digest and objective agreement must both fail."""
    from .phase10_utility import document_digest

    tampered = json.loads(json.dumps(fitted))
    if tampered["trait_weights"] is not None:
        tampered["trait_weights"][0][0] += 1e-3
    else:
        tampered["family_offsets_raw"][0][0] += 1e-3

    def digest_of(model: dict) -> str:
        return document_digest(
            {
                "utility_version": model.get("utility_version"),
                "model_id": model["model_id"],
                "colour_order": model["colour_order"],
                "family_order": model["family_order"],
                "feature_order": model["feature_order"],
                "red_first_intercept": model["red_first_intercept"],
                "family_offsets_raw": model["family_offsets_raw"],
                "trait_weights": model["trait_weights"],
            }
        )

    digest_changed = digest_of(tampered) != digest_of(fitted)
    objective_shift = abs(
        audit_objective(design, tampered)["objective"] - fitted["diagnostics"]["objective"]
    )
    return {
        "control": "altered_coefficient",
        "digest_changed": bool(digest_changed),
        "objective_abs_difference": objective_shift,
        "detected": bool(digest_changed and objective_shift > OBJECTIVE_AGREEMENT_TOLERANCE),
    }


__all__ = [
    "AUDIT_L2_LAMBDA",
    "AUDIT_RESULT_TARGETS",
    "AUDIT_RESULT_WINNER",
    "CENTERING_TOLERANCE",
    "GRADIENT_FD_ABS_TOLERANCE",
    "GRADIENT_FD_REL_TOLERANCE",
    "GRADIENT_FD_STEP",
    "GRADIENT_STATIONARITY_TOLERANCE",
    "LOGIT_AGREEMENT_TOLERANCE",
    "OBJECTIVE_AGREEMENT_TOLERANCE",
    "AuditDesign",
    "Phase10UtilityAuditError",
    "ReconstructedLibrary",
    "audit_corpus_records",
    "audit_fitted_model",
    "audit_gradient",
    "audit_logits",
    "audit_objective",
    "compare_refits",
    "control_altered_coefficient",
    "control_altered_family_id",
    "control_held_out_scaler",
    "control_orientation_swap",
    "control_permuted_trait_column",
    "control_wrong_draw_target",
    "finite_difference_spot_checks",
    "independent_feature_names",
    "independent_flatten",
    "independent_scaler_moments",
    "reconstruct_library",
    "standardize",
]
