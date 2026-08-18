"""Phase 10 Agent 1: frozen setup-utility features, standardizer, fit protocol.

Specification sources:

- `00_PHASE_10_SEQUENCE_AND_COMMON_CONTRACT.md` ("Utility models", "Exactly
  six candidates")
- `01_AGENT_1_CONTRACT_SEEDS_BANKS_ACCEPTANCE.md` ("Freeze utility fitting")

What this module freezes — and what it deliberately does not do
---------------------------------------------------------------
Agent 1 fits nothing. This module contains the complete *definition* of the
two utility models so that Agent 3 has no learning-design decision left to
make: the flattened trait feature order, the train-only standardizer (which
depends on the frozen Phase 7 library and on no game outcome whatsoever),
the parameter layout, the identifiability rule, the exact objective, and
the exact deterministic optimizer settings. The one thing it never contains
is a fitted parameter: `fit_utility_model` does not exist here, and the
completion gate `no_utility_fit` is a statement about this module too.

The utility domain is the base, not the played setup
----------------------------------------------------
`u(s, c)` is a function of a *Phase 7 base setup* — its primary family and
its stored trait vector — never of the final played arrangement. That is
forced by the selector contract: a selector chooses a base and only then
hands it to the frozen reflection/perturbation path, so any feature the
selector could not see at choice time would be unusable. Fitting on base
identity and selecting on base identity is therefore the only self-consistent
choice, and it keeps the six legal selector inputs exactly legal.

`phase10_trait_feature_v1`
--------------------------
`setup_trait_vector_v1` has 35 frozen fields, four of which are per-rank
histograms rather than scalars. The feature vector is the **lossless**
flattening of those 35 fields in `TRAIT_SCHEMA` order: an `int` or `float6`
field contributes one scalar; an `int_list4` field contributes its four rank
components as `name[0]..name[3]`. No field is dropped, no field is invented,
and no transformation other than the frozen standardizer is applied. The
result is 47 float64 scalars.

The flattening surfaces exact linear dependencies the schema already
contained. Each of the four histograms sums to a fixed inventory count, and
the schema separately carries that histogram's `front2`, `back2` and (for
Bombs, Scouts and Miners) `front_rank` aggregates, which are sums of the very
components now present; `movable_front_rank_count` and
`front_rank_immovable_count` likewise sum to ten. That is 16 exact relations,
so the 47-column standardized train matrix has rank 31. The frozen L2 penalty
of 1e-3 on the trait weights makes the minimizer unique regardless, which is
why the lossless flattening is safe to freeze rather than something to prune
by hand. :func:`train_feature_rank` records the observed rank as a frozen
diagnostic.

One caveat is recorded rather than papered over: `flag_file` is the single
field of the 35 that is not reflection-invariant. It is a deterministic
property of the base's stored class representative, so it is a legal,
reproducible selector input; it simply describes the stored orientation
rather than the reflection class. It is kept because the contract freezes
the 35-field vector, and its presence is noted in the utility contract.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from ..setups.contracts import (
    LIBRARY_JSONL_PATH,
    SPLIT_TRAIN,
    TRAIN_TOTAL,
)
from ..setups.families import FAMILY_IDS
from ..setups.library import read_library_jsonl
from ..setups.traits import TRAIT_SCHEMA, TRAIT_SCHEMA_VERSION

#: Version of the flattened feature definition. A change to the field order,
#: to the histogram expansion, or to the source trait schema is a new
#: version, never a silent edit.
TRAIT_FEATURE_VERSION = "phase10_trait_feature_v1"

#: Version of the train-only standardizer.
TRAIT_SCALER_VERSION = "phase10_trait_scaler_v1"

#: Version of the two utility models and their frozen fit protocol.
SETUP_UTILITY_VERSION = "phase10_setup_utility_v1"

#: The two model ids. There is no third model.
MODEL_FAMILY_ONLY = "model_F"
MODEL_FAMILY_TRAITS = "model_T"
UTILITY_MODEL_IDS = (MODEL_FAMILY_ONLY, MODEL_FAMILY_TRAITS)

#: Number of rank components an `int_list4` trait field expands into.
HISTOGRAM_WIDTH = 4


class Phase10UtilityError(ValueError):
    """Raised when a Phase 10 utility definition is used incorrectly."""


def _feature_names() -> "tuple[str, ...]":
    names: list[str] = []
    for field in TRAIT_SCHEMA:
        if field.kind == "int_list4":
            names.extend(f"{field.name}[{index}]" for index in range(HISTOGRAM_WIDTH))
        elif field.kind in ("int", "float6"):
            names.append(field.name)
        else:  # pragma: no cover - a new trait kind must be frozen explicitly
            raise Phase10UtilityError(
                f"trait field {field.name!r} has kind {field.kind!r}, which "
                f"{TRAIT_FEATURE_VERSION} does not know how to flatten"
            )
    return tuple(names)


#: The frozen feature order: 47 float64 scalars derived losslessly from the
#: 35 frozen trait fields.
TRAIT_FEATURE_NAMES = _feature_names()
TRAIT_FEATURE_COUNT = len(TRAIT_FEATURE_NAMES)
assert TRAIT_FEATURE_COUNT == 47, TRAIT_FEATURE_COUNT

#: The single trait field that is not reflection-invariant, recorded so the
#: caveat travels with the contract instead of living in a comment.
NON_REFLECTION_INVARIANT_FEATURES = tuple(
    field.name for field in TRAIT_SCHEMA if not field.reflection_invariant
)
assert NON_REFLECTION_INVARIANT_FEATURES == ("flag_file",)


def trait_feature_vector(trait_vector: dict) -> "tuple[float, ...]":
    """The 47-scalar float64 feature vector of one base's trait vector.

    Pure and total: it reads exactly the 35 frozen fields, expands the four
    histograms in rank order, and never consults a game outcome, a model
    score or any strength signal.
    """
    values: list[float] = []
    for field in TRAIT_SCHEMA:
        try:
            raw = trait_vector[field.name]
        except KeyError:
            raise Phase10UtilityError(
                f"trait vector is missing frozen field {field.name!r}"
            ) from None
        if field.kind == "int_list4":
            if len(raw) != HISTOGRAM_WIDTH:
                raise Phase10UtilityError(
                    f"{field.name}: expected {HISTOGRAM_WIDTH} rank components, "
                    f"got {len(raw)}"
                )
            values.extend(float(component) for component in raw)
        else:
            values.append(float(raw))
    return tuple(values)


# ---------------------------------------------------------------------------
# The train-only standardizer
# ---------------------------------------------------------------------------


class TraitScaler:
    """Population mean/std over **all 6,400 train bases**, and nothing else.

    Held-out bases never enter the statistics: the standardizer is part of
    the model, and fitting it on validation or test bases would be exactly
    the leak the phase's stop conditions forbid. `ddof=0` is the frozen
    population convention, and a zero-std field standardizes to a constant
    0.0 rather than dividing by zero — recorded in
    :attr:`zero_std_features` so the degeneracy is visible instead of
    silently absorbed.
    """

    __slots__ = ("mean", "std", "zero_std_features", "base_count", "split")

    def __init__(self, mean, std, base_count: int, split: str = SPLIT_TRAIN) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        if self.mean.shape != (TRAIT_FEATURE_COUNT,) or self.std.shape != (
            TRAIT_FEATURE_COUNT,
        ):
            raise Phase10UtilityError(
                f"scaler expects {TRAIT_FEATURE_COUNT} features, got "
                f"mean{self.mean.shape} std{self.std.shape}"
            )
        self.zero_std_features = tuple(
            name
            for name, value in zip(TRAIT_FEATURE_NAMES, self.std)
            if value == 0.0
        )
        self.base_count = int(base_count)
        self.split = str(split)

    def transform(self, features) -> np.ndarray:
        """Standardize one feature vector or a stack of them, in float64."""
        array = np.asarray(features, dtype=np.float64)
        centered = array - self.mean
        divisor = np.where(self.std == 0.0, 1.0, self.std)
        standardized = centered / divisor
        return np.where(self.std == 0.0, 0.0, standardized)

    def to_dict(self) -> dict:
        return {
            "scaler_version": TRAIT_SCALER_VERSION,
            "feature_version": TRAIT_FEATURE_VERSION,
            "trait_schema_version": TRAIT_SCHEMA_VERSION,
            "split": self.split,
            "base_count": self.base_count,
            "ddof": 0,
            "feature_names": list(TRAIT_FEATURE_NAMES),
            "mean": [float(value) for value in self.mean],
            "std": [float(value) for value in self.std],
            "zero_std_features": list(self.zero_std_features),
            "zero_std_rule": "a zero-std feature standardizes to the constant 0.0",
        }

    def digest(self) -> str:
        """SHA-256 over the scaler's canonical JSON — its stable identity."""
        return document_digest(self.to_dict())


def load_train_features(path: str = LIBRARY_JSONL_PATH):
    """`(base_setup_ids, families, features)` for all 6,400 train bases.

    Order is the library's frozen enumeration order restricted to the train
    split — ascending `(family_index, base_index)` — so the standardizer and
    every later selector share one deterministic base ordering.
    """
    entries = [entry for entry in read_library_jsonl(path) if entry.split == SPLIT_TRAIN]
    if len(entries) != TRAIN_TOTAL:
        raise Phase10UtilityError(
            f"expected {TRAIN_TOTAL} train bases, got {len(entries)}"
        )
    order = {family_id: index for index, family_id in enumerate(FAMILY_IDS)}
    entries.sort(key=lambda entry: (order[entry.family_id], entry.base_index))
    base_ids = tuple(entry.base_setup_id for entry in entries)
    families = tuple(entry.family_id for entry in entries)
    features = np.array(
        [trait_feature_vector(entry.trait_vector) for entry in entries],
        dtype=np.float64,
    )
    return base_ids, families, features


def fit_trait_scaler(path: str = LIBRARY_JSONL_PATH) -> TraitScaler:
    """The frozen standardizer, computed from the train split alone.

    "Fit" here means two moments of the frozen library — no outcome, no
    game, no model. It is deterministic, so recomputing it in any process
    yields the identical digest.
    """
    _, _, features = load_train_features(path)
    return TraitScaler(
        mean=features.mean(axis=0),
        std=features.std(axis=0, ddof=0),
        base_count=features.shape[0],
        split=SPLIT_TRAIN,
    )


def train_feature_rank(scaler: "TraitScaler | None" = None, path: str = LIBRARY_JSONL_PATH) -> dict:
    """Numerical rank of the standardized train feature matrix.

    A frozen diagnostic, recorded because the lossless histogram expansion
    is knowingly collinear: the rank shortfall is expected and bounded, and
    the L2 penalty is what makes the minimizer unique anyway.
    """
    scaler = fit_trait_scaler(path) if scaler is None else scaler
    _, _, features = load_train_features(path)
    standardized = scaler.transform(features)
    singular = np.linalg.svd(standardized, compute_uv=False)
    tolerance = float(
        max(standardized.shape) * np.finfo(np.float64).eps * float(singular[0])
    )
    rank = int((singular > tolerance).sum())
    return {
        "rows": int(standardized.shape[0]),
        "columns": int(standardized.shape[1]),
        "rank": rank,
        "rank_deficiency": int(standardized.shape[1] - rank),
        "tolerance": tolerance,
        "largest_singular_value": float(singular[0]),
        "smallest_singular_value": float(singular[-1]),
        "note": (
            "collinearity is expected and exactly accounted for: each of the four "
            "histograms sums to a fixed inventory count (4 relations), the schema's "
            "front2 and back2 aggregates are sums of those components (8), the "
            "Bomb/Scout/Miner front_rank aggregates repeat component 3 (3), and "
            "movable_front_rank_count + front_rank_immovable_count = 10 (1) -- 16 "
            "relations, hence rank 31 of 47; the frozen L2 penalty makes the "
            "minimizer unique regardless"
        ),
    }


# ---------------------------------------------------------------------------
# Frozen fit protocol
# ---------------------------------------------------------------------------

#: Outcome target from the Red perspective, with a draw exactly halfway.
OUTCOME_TARGETS = {"red_win": 1.0, "draw": 0.5, "red_loss": 0.0}

#: L2 coefficient on the family offsets and trait weights. The red-first
#: intercept is never penalized.
L2_LAMBDA = 1e-3

#: The frozen deterministic optimizer settings. `strong_wolfe` is available
#: in this environment (torch 2.13), so no deterministic-equivalent
#: authorization is required; Agent 1 verified it from live bytes before
#: freezing, and a future environment lacking it must stop rather than
#: substitute.
FIT_PROTOCOL = {
    "device": "cpu",
    "precision": "float64",
    "objective": "full_batch_bce_plus_l2",
    "bce_reduction": "mean over the N scheduled games",
    "l2_lambda": L2_LAMBDA,
    "l2_applies_to": ["family_offsets", "trait_weights"],
    "intercept_penalty": "none",
    "optimizer": "deterministic full-batch L-BFGS",
    "lr": 1.0,
    "max_iterations": 500,
    "history_size": 50,
    "tolerance_grad": 1e-10,
    "tolerance_change": 1e-12,
    "line_search_fn": "strong_wolfe",
    "initialisation": "exact all-zero parameter vector",
    "hyperparameter_search": "forbidden",
    "batching": "single full-batch closure; no shuffling, no minibatches",
}

#: The frozen identifiability rule. The penalty is applied to the *raw*
#: family parameters while the logit uses their per-colour centered values,
#: so the unique minimizer is automatically centered and the raw parameters
#: carry no flat direction for the optimizer to wander along.
FAMILY_CENTERING_RULE = (
    "effective offset b_eff[c, f] = b_raw[c, f] - mean_f b_raw[c, :]; the logit "
    "uses b_eff and the L2 penalty uses b_raw, so the unique minimizer satisfies "
    "mean_f b_raw[c, :] = 0 for each colour"
)


def parameter_layout(model_id: str) -> dict:
    """The frozen parameter layout of one utility model."""
    if model_id not in UTILITY_MODEL_IDS:
        raise Phase10UtilityError(
            f"unknown utility model {model_id!r}; expected one of {list(UTILITY_MODEL_IDS)}"
        )
    family_parameters = 2 * len(FAMILY_IDS)
    trait_parameters = 2 * TRAIT_FEATURE_COUNT if model_id == MODEL_FAMILY_TRAITS else 0
    return {
        "model_id": model_id,
        "utility": (
            "u_F(s, c) = b_eff[c, family(s)]"
            if model_id == MODEL_FAMILY_ONLY
            else "u_T(s, c) = b_eff[c, family(s)] + w[c] . x(s)"
        ),
        "utility_domain": "phase 7 base setup (its primary family and stored trait vector)",
        "family_offsets": {"shape": [2, len(FAMILY_IDS)], "count": family_parameters},
        "trait_weights": (
            {"shape": [2, TRAIT_FEATURE_COUNT], "count": trait_parameters}
            if trait_parameters
            else None
        ),
        "red_first_intercept": {"shape": [], "count": 1},
        "total_parameters": family_parameters + trait_parameters + 1,
        "colour_order": ["red", "blue"],
        "family_order": list(FAMILY_IDS),
        "feature_order": list(TRAIT_FEATURE_NAMES) if trait_parameters else [],
    }


def game_logit(
    intercept: float,
    red_utility: float,
    blue_utility: float,
) -> float:
    """`eta = red_first_intercept + u(red_setup, red) - u(blue_setup, blue)`."""
    return float(intercept) + float(red_utility) - float(blue_utility)


def objective_value(
    logits,
    targets,
    penalized_parameters,
    *,
    l2_lambda: float = L2_LAMBDA,
) -> float:
    """The frozen objective, recomputed independently in float64 numpy.

    Agent 3 fits with L-BFGS in torch; this is the *independent formula* the
    fit must agree with, written once so an auditor never has to infer the
    reduction or the penalty scope from optimizer code:

    ```text
    L = mean_i [ -( y_i log p_i + (1 - y_i) log(1 - p_i) ) ]  +  lambda * sum(theta_pen^2)
    p_i = sigmoid(eta_i)
    ```

    Targets are the Red-perspective outcomes 1.0 / 0.5 / 0.0, so a draw
    contributes the soft-label cross entropy rather than being dropped.
    Computed through `logaddexp` so a large-magnitude logit cannot overflow.
    """
    eta = np.asarray(logits, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if eta.shape != y.shape:
        raise Phase10UtilityError(
            f"logits{eta.shape} and targets{y.shape} must have the same shape"
        )
    # -log(sigmoid(eta)) = log(1 + exp(-eta)); -log(1 - sigmoid(eta)) = log(1 + exp(eta))
    negative_log_p = np.logaddexp(0.0, -eta)
    negative_log_1mp = np.logaddexp(0.0, eta)
    bce = float(np.mean(y * negative_log_p + (1.0 - y) * negative_log_1mp))
    theta = np.asarray(penalized_parameters, dtype=np.float64).ravel()
    return bce + float(l2_lambda) * float(np.dot(theta, theta))


def document_digest(document) -> str:
    """SHA-256 over a document's canonical JSON — the frozen convention."""
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def utility_contract_document(scaler: "TraitScaler | None" = None) -> dict:
    """`phase10_setup_utility_v1` — the complete frozen utility contract."""
    scaler = fit_trait_scaler() if scaler is None else scaler
    return {
        "utility_version": SETUP_UTILITY_VERSION,
        "models": [parameter_layout(model_id) for model_id in UTILITY_MODEL_IDS],
        "model_count": len(UTILITY_MODEL_IDS),
        "candidate_specific_refitting": "forbidden; the two models are fit once",
        "feature": {
            "feature_version": TRAIT_FEATURE_VERSION,
            "trait_schema_version": TRAIT_SCHEMA_VERSION,
            "source_fields": len(TRAIT_SCHEMA),
            "feature_count": TRAIT_FEATURE_COUNT,
            "feature_names": list(TRAIT_FEATURE_NAMES),
            "flattening": (
                "TRAIT_SCHEMA order; int and float6 fields contribute one scalar; "
                "int_list4 fields contribute their four rank components as "
                "name[0]..name[3]; nothing is dropped and nothing is invented"
            ),
            "non_reflection_invariant_features": list(NON_REFLECTION_INVARIANT_FEATURES),
            "non_reflection_invariant_note": (
                "flag_file describes the stored class representative's orientation; "
                "it is a deterministic property of the base and therefore a legal "
                "selector input, but it is not a property of the reflection class"
            ),
        },
        "scaler": scaler.to_dict(),
        "scaler_digest": scaler.digest(),
        "targets": dict(OUTCOME_TARGETS),
        "target_orientation": "red perspective",
        "game_logit": "eta = red_first_intercept + u(red_setup, red) - u(blue_setup, blue)",
        "intercept_role": "fit diagnostic only; never used to rank setups",
        "family_centering": FAMILY_CENTERING_RULE,
        "objective": (
            "L = mean_i BCE(sigmoid(eta_i), y_i) + lambda * sum(theta_pen^2), "
            "theta_pen = raw family offsets and trait weights, intercept excluded"
        ),
        "fit_protocol": dict(FIT_PROTOCOL),
        "fit_seed": {
            "root": "utility_fit_seed 2026081804",
            "consumed_in_v1": False,
            "reason": "the frozen protocol starts from an exact all-zero vector",
        },
        "train_feature_rank": train_feature_rank(scaler),
        "held_out_use": (
            "the standardizer and both models see train-split bases only; a "
            "validation or test base entering either is a BLOCKED leak"
        ),
    }


__all__ = [
    "FAMILY_CENTERING_RULE",
    "FIT_PROTOCOL",
    "HISTOGRAM_WIDTH",
    "L2_LAMBDA",
    "MODEL_FAMILY_ONLY",
    "MODEL_FAMILY_TRAITS",
    "NON_REFLECTION_INVARIANT_FEATURES",
    "OUTCOME_TARGETS",
    "SETUP_UTILITY_VERSION",
    "TRAIT_FEATURE_COUNT",
    "TRAIT_FEATURE_NAMES",
    "TRAIT_FEATURE_VERSION",
    "TRAIT_SCALER_VERSION",
    "UTILITY_MODEL_IDS",
    "Phase10UtilityError",
    "TraitScaler",
    "document_digest",
    "fit_trait_scaler",
    "game_logit",
    "load_train_features",
    "objective_value",
    "parameter_layout",
    "train_feature_rank",
    "utility_contract_document",
]
