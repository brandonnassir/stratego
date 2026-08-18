"""Phase 10 Agent 3: fit the two frozen setup-utility models from the sealed corpus.

Specification sources:

- `03_AGENT_3_UTILITY_MODELS_AND_AUDIT.md` ("Fit Model F", "Fit Model T",
  "Production-input safety", "No model selection")
- `00_PHASE_10_SEQUENCE_AND_COMMON_CONTRACT.md` ("Utility models")
- `stratego/training/phase10_utility.py` — Agent 1's frozen definitions,
  which this module executes and adds nothing to

What fitting is allowed to see
------------------------------
The sealed corpus record has 37 stored fields; the 37-field record is
storage and provenance, **not a feature set**. Fitting reads records only
through :class:`AllowlistedRecord`, which raises on any field outside the
model's frozen allowlist:

```text
Model F: game_id (row identity), red_family, blue_family, result
Model T: the same, plus red_base_setup_id / blue_base_setup_id
```

`game_id` orders the rows and `result` rebuilds the target; neither is a
feature. The base ids exist so Model T can resolve each side's **base**
through `setup_library_v1` and derive the frozen `phase10_trait_feature_v1`
47-scalar representation through the frozen train-only scaler. No final
played fingerprint, reflection/perturbation realization, setup seed,
terminal reason, game length, digest, or sampler provenance can enter a
design matrix: the guard makes reading one an error, not a code review
finding.

The outcome target is reconstructed from the stored W/D/L `result` token
through the frozen mapping (red win 1.0, draw 0.5, red loss 0.0). The stored
`red_score` field is deliberately *not* read here — the independent audit
checks it agrees, but fitting derives its own target.

One fit, frozen protocol
------------------------
Each model is fit exactly once, from the exact all-zero parameter vector,
with the frozen CPU float64 full-batch L-BFGS settings in
:data:`stratego.training.phase10_utility.FIT_PROTOCOL`. There is no
hyperparameter argument on :func:`fit_utility_model` to search over. The
fit runs single-threaded so the reduction order — and therefore every bit
of every coefficient — is identical across processes; the deterministic
refit gate compares coefficients for exact equality, not closeness.

The logit uses centered offsets `b_eff[c] = b_raw[c] - mean(b_raw[c])`
while the L2 penalty uses the raw offsets, so the unique minimizer is
automatically centered (the frozen identifiability rule).

Production scoring is own-side only
-----------------------------------
The training objective sees both completed-game setups because a game has
two sides; the exported scorer does not. :class:`SetupUtilityScorer` scores
`u(s, c)` from exactly (own colour, own family, own base trait vector) —
there is no opponent argument to pass, no opponent-conditioned table in the
artifact for one to index, and the red-first intercept is stored as a fit
diagnostic that no scoring path reads.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..setups.contracts import SPLIT_TRAIN
from ..setups.families import FAMILY_IDS
from .phase10_schedule import RESULT_TARGETS
from .phase10_utility import (
    FIT_PROTOCOL,
    L2_LAMBDA,
    MODEL_FAMILY_ONLY,
    MODEL_FAMILY_TRAITS,
    OUTCOME_TARGETS,
    SETUP_UTILITY_VERSION,
    TRAIT_FEATURE_COUNT,
    TRAIT_FEATURE_NAMES,
    TRAIT_FEATURE_VERSION,
    UTILITY_MODEL_IDS,
    TraitScaler,
    document_digest,
    fit_trait_scaler,
    trait_feature_vector,
)

#: Identity of the *fitted* artifact, distinct from the frozen definition
#: `phase10_setup_utility_v1` it satisfies. The common contract names this
#: identity for the accepted utility Agent 4's selector consumes.
FITTED_UTILITY_VERSION = "setup_utility_v1"

#: The accepted train-only standardizer digest, pinned where fitting happens
#: so a drifted scaler cannot silently reach a design matrix.
ACCEPTED_TRAIT_SCALER_DIGEST = (
    "fa6eb1c112defc4c1034831b84db8848181e1f674f8439c9c265916d89e8b7f9"
)

#: Where the fitted production artifact lives, relative to the repository
#: root — the Phase 10 model hierarchy, beside `checkpoints/phase9/...`.
FITTED_UTILITY_RELATIVE_PATH = "checkpoints/phase10/setup_utility_v1.json"

# The two frozen mappings must agree or the phase is incoherent; assert the
# identity once, at import, rather than trusting two modules to stay in step.
assert RESULT_TARGETS == OUTCOME_TARGETS, (RESULT_TARGETS, OUTCOME_TARGETS)

#: The frozen fitting-input allowlist of each model. `game_id` is row
#: identity, `result` is the target source; features are the family fields
#: and, for Model T only, the base ids that resolve to library trait
#: vectors. There is no third allowlist and no way to widen one per call.
FIT_INPUT_ALLOWLIST = {
    MODEL_FAMILY_ONLY: ("game_id", "red_family", "blue_family", "result"),
    MODEL_FAMILY_TRAITS: (
        "game_id",
        "red_family",
        "blue_family",
        "result",
        "red_base_setup_id",
        "blue_base_setup_id",
    ),
}

#: Every stored/derived record field fitting must never read, computed as
#: the complement of the widest allowlist so a new stored field is forbidden
#: by default rather than allowed by omission.
from .phase10_outcome_store import ASSEMBLED_RECORD_FIELDS  # noqa: E402

FORBIDDEN_FIT_FIELDS = tuple(
    sorted(set(ASSEMBLED_RECORD_FIELDS) - set(FIT_INPUT_ALLOWLIST[MODEL_FAMILY_TRAITS]))
)
assert "red_final_fingerprint" in FORBIDDEN_FIT_FIELDS
assert "terminal_reason" in FORBIDDEN_FIT_FIELDS
assert "plies" in FORBIDDEN_FIT_FIELDS
assert "red_score" in FORBIDDEN_FIT_FIELDS


class Phase10UtilityFitError(RuntimeError):
    """A fitting precondition, allowlist or protocol condition failed."""


class AllowlistedRecord(Mapping):
    """A corpus record restricted to one model's frozen fitting allowlist.

    Reading any other field raises: the 37-field record is storage and
    provenance, and the only way fitting code can consume a forbidden field
    is to not be behind this guard — which the tests and the independent
    audit check it is. Accessed keys are recorded so the audit can publish
    exactly which fields fitting consumed.
    """

    __slots__ = ("_record", "_allowlist", "accessed")

    def __init__(self, record: dict, allowlist: "tuple[str, ...]") -> None:
        self._record = record
        self._allowlist = frozenset(allowlist)
        self.accessed: set[str] = set()

    def __getitem__(self, key: str):
        if key not in self._allowlist:
            raise Phase10UtilityFitError(
                f"field {key!r} is not on the fitting-input allowlist; the stored "
                "record is provenance, not a feature set"
            )
        self.accessed.add(key)
        return self._record[key]

    def __iter__(self):
        return iter(sorted(self._allowlist & set(self._record)))

    def __len__(self) -> int:
        return len(self._allowlist & set(self._record))


@dataclass(frozen=True)
class FitData:
    """The complete design of one utility-model fit, in canonical row order."""

    model_id: str
    game_ids: "tuple[str, ...]"
    targets: np.ndarray  # (N,) float64 in {1.0, 0.5, 0.0}
    red_family_index: np.ndarray  # (N,) int64 into FAMILY_IDS
    blue_family_index: np.ndarray  # (N,) int64 into FAMILY_IDS
    red_features: "np.ndarray | None"  # (N, 47) float64, standardized
    blue_features: "np.ndarray | None"
    scaler_digest: "str | None"
    accessed_fields: "tuple[str, ...]"

    @property
    def game_count(self) -> int:
        return len(self.game_ids)


def build_fit_data(
    records,
    model_id: str,
    *,
    index=None,
    scaler: "TraitScaler | None" = None,
) -> FitData:
    """The design matrices of one model, read through the allowlist guard.

    ``records`` is an iterable of assembled corpus records (normally
    ``OutcomeReader.iter_records()``), which the reader yields in canonical
    ``sorted(game_id)`` order; that order is verified, not assumed. For
    Model T every side's base id is resolved through the live
    `setup_library_v1` index: a base outside the train split stops the fit
    (a held-out base reaching a design matrix is the phase's hard leak), and
    a base whose library family disagrees with the record's scheduled family
    stops it too.
    """
    if model_id not in UTILITY_MODEL_IDS:
        raise Phase10UtilityFitError(f"unknown utility model {model_id!r}")
    allowlist = FIT_INPUT_ALLOWLIST[model_id]
    wants_traits = model_id == MODEL_FAMILY_TRAITS

    if wants_traits:
        if index is None:
            from ..setups.sampler import load_library_index

            index = load_library_index()
        scaler = fit_trait_scaler() if scaler is None else scaler
        observed_scaler = scaler.digest()
        if observed_scaler != ACCEPTED_TRAIT_SCALER_DIGEST:
            raise Phase10UtilityFitError(
                f"trait scaler digest {observed_scaler} is not the accepted "
                f"{ACCEPTED_TRAIT_SCALER_DIGEST}; refusing to standardize features "
                "with an unfrozen scaler"
            )

    family_index = {family_id: position for position, family_id in enumerate(FAMILY_IDS)}
    feature_cache: dict[str, np.ndarray] = {}

    def side_features(base_setup_id: str, recorded_family: str) -> np.ndarray:
        cached = feature_cache.get(base_setup_id)
        if cached is None:
            entry = index.base(base_setup_id)
            if entry.split != SPLIT_TRAIN:
                raise Phase10UtilityFitError(
                    f"base {base_setup_id} is in split {entry.split!r}; a held-out "
                    "base entering utility fitting is a hard leak (BLOCKED)"
                )
            cached = np.asarray(
                scaler.transform(trait_feature_vector(entry.trait_vector)),
                dtype=np.float64,
            )
            feature_cache[base_setup_id] = cached
        entry = index.base(base_setup_id)
        if entry.family_id != recorded_family:
            raise Phase10UtilityFitError(
                f"base {base_setup_id} belongs to family {entry.family_id}, but the "
                f"record schedules family {recorded_family}"
            )
        return cached

    game_ids: list[str] = []
    targets: list[float] = []
    red_families: list[int] = []
    blue_families: list[int] = []
    red_rows: list[np.ndarray] = []
    blue_rows: list[np.ndarray] = []
    accessed: set[str] = set()

    previous_id = ""
    for raw in records:
        record = AllowlistedRecord(raw, allowlist)
        game_id = str(record["game_id"])
        if game_id <= previous_id:
            raise Phase10UtilityFitError(
                f"records are not in strict canonical game-id order at {game_id!r}"
            )
        previous_id = game_id

        result = record["result"]
        if result not in RESULT_TARGETS:
            raise Phase10UtilityFitError(f"{game_id}: unknown result token {result!r}")
        red_family = record["red_family"]
        blue_family = record["blue_family"]
        if red_family not in family_index or blue_family not in family_index:
            raise Phase10UtilityFitError(
                f"{game_id}: unknown family ({red_family!r}, {blue_family!r})"
            )

        game_ids.append(game_id)
        targets.append(RESULT_TARGETS[result])
        red_families.append(family_index[red_family])
        blue_families.append(family_index[blue_family])
        if wants_traits:
            red_rows.append(side_features(record["red_base_setup_id"], red_family))
            blue_rows.append(side_features(record["blue_base_setup_id"], blue_family))
        accessed |= record.accessed

    if not game_ids:
        raise Phase10UtilityFitError("no corpus records supplied")

    return FitData(
        model_id=model_id,
        game_ids=tuple(game_ids),
        targets=np.asarray(targets, dtype=np.float64),
        red_family_index=np.asarray(red_families, dtype=np.int64),
        blue_family_index=np.asarray(blue_families, dtype=np.int64),
        red_features=np.vstack(red_rows) if wants_traits else None,
        blue_features=np.vstack(blue_rows) if wants_traits else None,
        scaler_digest=scaler.digest() if wants_traits else None,
        accessed_fields=tuple(sorted(accessed)),
    )


# ---------------------------------------------------------------------------
# The frozen fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FittedUtilityModel:
    """One fitted utility model: coefficients, layout and fit diagnostics."""

    model_id: str
    red_first_intercept: float
    family_offsets_raw: "tuple[tuple[float, ...], ...]"  # (2, 16) red then blue
    trait_weights: "tuple[tuple[float, ...], ...] | None"  # (2, 47) or None
    diagnostics: dict

    @property
    def family_offsets_effective(self) -> "tuple[tuple[float, ...], ...]":
        """The centered offsets the logit and every scorer actually use."""
        centered = []
        for row in self.family_offsets_raw:
            mean = sum(row) / len(row)
            centered.append(tuple(value - mean for value in row))
        return tuple(centered)

    def coefficient_document(self) -> dict:
        """The parameters and their frozen layout — identity, no diagnostics."""
        return {
            "utility_version": SETUP_UTILITY_VERSION,
            "model_id": self.model_id,
            "colour_order": ["red", "blue"],
            "family_order": list(FAMILY_IDS),
            "feature_order": (
                list(TRAIT_FEATURE_NAMES) if self.trait_weights is not None else []
            ),
            "red_first_intercept": self.red_first_intercept,
            "family_offsets_raw": [list(row) for row in self.family_offsets_raw],
            "trait_weights": (
                None
                if self.trait_weights is None
                else [list(row) for row in self.trait_weights]
            ),
        }

    def coefficient_digest(self) -> str:
        return document_digest(self.coefficient_document())

    def to_dict(self) -> dict:
        return {
            **self.coefficient_document(),
            "family_offsets_effective": [
                list(row) for row in self.family_offsets_effective
            ],
            "coefficient_digest": self.coefficient_digest(),
            "diagnostics": dict(self.diagnostics),
        }


def fit_utility_model(fit_data: FitData) -> FittedUtilityModel:
    """One deterministic execution of the frozen fit protocol.

    No argument chooses a hyperparameter: device, precision, objective,
    penalty, optimizer settings and initialisation all come from the frozen
    `phase10_setup_utility_v1` contract. The fit is single-threaded so two
    executions of this function — in this process or any other — produce
    bit-identical coefficients, which the deterministic-refit gate then
    checks by exact equality.
    """
    import torch

    wants_traits = fit_data.model_id == MODEL_FAMILY_TRAITS
    if wants_traits and (fit_data.red_features is None or fit_data.blue_features is None):
        raise Phase10UtilityFitError("model_T fit data carries no feature matrices")
    if not wants_traits and (
        fit_data.red_features is not None or fit_data.blue_features is not None
    ):
        raise Phase10UtilityFitError("model_F fit data must not carry feature matrices")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        targets = torch.from_numpy(np.ascontiguousarray(fit_data.targets))
        red_index = torch.from_numpy(np.ascontiguousarray(fit_data.red_family_index))
        blue_index = torch.from_numpy(np.ascontiguousarray(fit_data.blue_family_index))
        red_features = blue_features = None
        if wants_traits:
            red_features = torch.from_numpy(np.ascontiguousarray(fit_data.red_features))
            blue_features = torch.from_numpy(np.ascontiguousarray(fit_data.blue_features))

        # The exact all-zero initialisation the contract freezes.
        intercept = torch.zeros((), dtype=torch.float64, requires_grad=True)
        family_offsets = torch.zeros(
            (2, len(FAMILY_IDS)), dtype=torch.float64, requires_grad=True
        )
        parameters = [intercept, family_offsets]
        trait_weights = None
        if wants_traits:
            trait_weights = torch.zeros(
                (2, TRAIT_FEATURE_COUNT), dtype=torch.float64, requires_grad=True
            )
            parameters.append(trait_weights)

        def logits() -> torch.Tensor:
            effective = family_offsets - family_offsets.mean(dim=1, keepdim=True)
            eta = (
                intercept
                + effective[0].index_select(0, red_index)
                - effective[1].index_select(0, blue_index)
            )
            if wants_traits:
                eta = eta + red_features.mv(trait_weights[0]) - blue_features.mv(
                    trait_weights[1]
                )
            return eta

        def objective() -> torch.Tensor:
            eta = logits()
            # The frozen all-zero start makes every logit exactly 0.0, where a
            # hand-composed stable BCE (clamp/abs/log1p pieces) autodiffs to the
            # WRONG subgradient (-y instead of sigmoid(0) - y), and L-BFGS then
            # line-searches a non-descent direction and never moves. The
            # library primitive computes the same stable soft-label forward and
            # its backward is the analytic sigmoid(eta) - y, exact at 0.
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                eta, targets, reduction="mean"
            )
            penalty = family_offsets.pow(2).sum()
            if wants_traits:
                penalty = penalty + trait_weights.pow(2).sum()
            return bce + L2_LAMBDA * penalty

        optimizer = torch.optim.LBFGS(
            parameters,
            lr=FIT_PROTOCOL["lr"],
            max_iter=FIT_PROTOCOL["max_iterations"],
            history_size=FIT_PROTOCOL["history_size"],
            tolerance_grad=FIT_PROTOCOL["tolerance_grad"],
            tolerance_change=FIT_PROTOCOL["tolerance_change"],
            line_search_fn=FIT_PROTOCOL["line_search_fn"],
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            loss.backward()
            return loss

        optimizer.step(closure)
        state = optimizer.state[parameters[0]]
        iterations = int(state.get("n_iter", 0))
        function_evaluations = int(state.get("func_evals", 0))

        # Final diagnostics at the returned parameters, outside the optimizer.
        for parameter in parameters:
            parameter.grad = None
        final_loss = objective()
        final_loss.backward()
        gradient = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
        with torch.no_grad():
            eta = logits()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                eta, targets, reduction="mean"
            )
            penalty = family_offsets.pow(2).sum()
            if wants_traits:
                penalty = penalty + trait_weights.pow(2).sum()

        diagnostics = {
            "objective": float(final_loss.detach()),
            "bce": float(bce),
            "l2_penalty": float(L2_LAMBDA * penalty),
            "l2_lambda": L2_LAMBDA,
            "iterations": iterations,
            "function_evaluations": function_evaluations,
            "final_grad_max_abs": float(gradient.abs().max()),
            "final_grad_l2": float(gradient.norm()),
            "converged_by_tolerance": iterations < FIT_PROTOCOL["max_iterations"],
            "hit_max_iterations": iterations >= FIT_PROTOCOL["max_iterations"],
            "raw_offset_means": [
                float(family_offsets.detach()[0].mean()),
                float(family_offsets.detach()[1].mean()),
            ],
            "games": fit_data.game_count,
            "device": FIT_PROTOCOL["device"],
            "precision": FIT_PROTOCOL["precision"],
            "threads": 1,
            "initialisation": FIT_PROTOCOL["initialisation"],
            "line_search_fn": FIT_PROTOCOL["line_search_fn"],
            "accessed_fields": list(fit_data.accessed_fields),
            "scaler_digest": fit_data.scaler_digest,
        }

        return FittedUtilityModel(
            model_id=fit_data.model_id,
            red_first_intercept=float(intercept.detach()),
            family_offsets_raw=tuple(
                tuple(float(value) for value in row)
                for row in family_offsets.detach().numpy()
            ),
            trait_weights=(
                None
                if trait_weights is None
                else tuple(
                    tuple(float(value) for value in row)
                    for row in trait_weights.detach().numpy()
                )
            ),
            diagnostics=diagnostics,
        )
    finally:
        torch.set_num_threads(previous_threads)


# ---------------------------------------------------------------------------
# The exported production artifact and its own-side scorer
# ---------------------------------------------------------------------------


def utility_models_artifact(
    models: "dict[str, FittedUtilityModel]",
    scaler: TraitScaler,
    *,
    corpus_content_digest: str,
    corpus_games: int,
) -> dict:
    """The complete fitted-utility artifact, ready to serialize.

    Everything a selector needs to score a base — and nothing else. The
    corpus identity names the evidence the coefficients came from; the
    scaler travels inside the artifact so production scoring never re-reads
    the library to standardize a trait vector.
    """
    if set(models) != set(UTILITY_MODEL_IDS):
        raise Phase10UtilityFitError(
            f"expected exactly models {list(UTILITY_MODEL_IDS)}, got {sorted(models)}"
        )
    scaler_digest = scaler.digest()
    if scaler_digest != ACCEPTED_TRAIT_SCALER_DIGEST:
        raise Phase10UtilityFitError(
            f"refusing to export with scaler digest {scaler_digest}"
        )
    return {
        "artifact_version": FITTED_UTILITY_VERSION,
        "utility_version": SETUP_UTILITY_VERSION,
        "feature_version": TRAIT_FEATURE_VERSION,
        "fit_protocol": dict(FIT_PROTOCOL),
        "training_corpus": {
            "corpus_version": "phase10_setup_outcome_corpus_v1",
            "content_digest": corpus_content_digest,
            "games": int(corpus_games),
            "split": SPLIT_TRAIN,
        },
        "fitting_input_allowlist": {
            model_id: list(fields) for model_id, fields in FIT_INPUT_ALLOWLIST.items()
        },
        "forbidden_fitting_fields": list(FORBIDDEN_FIT_FIELDS),
        "scaler": scaler.to_dict(),
        "scaler_digest": scaler_digest,
        "models": {model_id: models[model_id].to_dict() for model_id in UTILITY_MODEL_IDS},
        "production_scoring_rule": (
            "u(s, c) = family_offsets_effective[c][family_index(s)]"
            " + trait_weights[c] . standardized_features(s) for model_T;"
            " the red_first_intercept is a fit diagnostic and is never used to"
            " rank or score setups"
        ),
        "own_side_only": {
            "inputs": ["own colour", "own family", "own base trait vector"],
            "excluded": [
                "opponent family",
                "opponent base",
                "opponent trait vector",
                "opponent policy identity",
                "matchup matrix",
                "game outcome",
                "red_first_intercept",
            ],
        },
    }


def write_utility_models_artifact(artifact: dict, path: "str | Path") -> str:
    """Serialize the artifact and return its SHA-256 file digest."""
    import hashlib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


#: The exact key set of one serialized model entry, and of the artifact
#: root. The own-side proof checks the artifact against these closed sets so
#: an opponent-conditioned table cannot ride in under a new name.
MODEL_ENTRY_KEYS = frozenset(
    {
        "utility_version",
        "model_id",
        "colour_order",
        "family_order",
        "feature_order",
        "red_first_intercept",
        "family_offsets_raw",
        "family_offsets_effective",
        "trait_weights",
        "coefficient_digest",
        "diagnostics",
    }
)

ARTIFACT_ROOT_KEYS = frozenset(
    {
        "artifact_version",
        "utility_version",
        "feature_version",
        "fit_protocol",
        "training_corpus",
        "fitting_input_allowlist",
        "forbidden_fitting_fields",
        "scaler",
        "scaler_digest",
        "models",
        "production_scoring_rule",
        "own_side_only",
    }
)


def own_side_only_findings(artifact: dict) -> list:
    """Every way the artifact fails to be a pure own-side scorer.

    Mechanical, not narrative: closed key sets at the root and per model,
    exact per-colour parameter shapes (2 x 16 offsets, 2 x 47 weights), and
    no parameter object indexed by anything but own colour, own family and
    own feature. An empty list is the production-input safety proof the
    completion gate records.
    """
    findings: list[str] = []
    extra_root = set(artifact) - ARTIFACT_ROOT_KEYS
    missing_root = ARTIFACT_ROOT_KEYS - set(artifact)
    if extra_root:
        findings.append(f"unexpected artifact keys: {sorted(extra_root)}")
    if missing_root:
        findings.append(f"missing artifact keys: {sorted(missing_root)}")

    models = artifact.get("models", {})
    if set(models) != set(UTILITY_MODEL_IDS):
        findings.append(f"artifact models are {sorted(models)}")
    for model_id, entry in models.items():
        extra = set(entry) - MODEL_ENTRY_KEYS
        if extra:
            findings.append(f"{model_id}: unexpected model keys {sorted(extra)}")
        offsets = entry.get("family_offsets_raw", [])
        if len(offsets) != 2 or any(len(row) != len(FAMILY_IDS) for row in offsets):
            findings.append(f"{model_id}: family offsets are not exactly 2 x 16")
        weights = entry.get("trait_weights")
        if model_id == MODEL_FAMILY_ONLY and weights is not None:
            findings.append("model_F carries trait weights")
        if model_id == MODEL_FAMILY_TRAITS and (
            weights is None
            or len(weights) != 2
            or any(len(row) != TRAIT_FEATURE_COUNT for row in weights)
        ):
            findings.append("model_T trait weights are not exactly 2 x 47")
        if entry.get("colour_order") != ["red", "blue"]:
            findings.append(f"{model_id}: colour order is not ['red', 'blue']")
        if entry.get("family_order") != list(FAMILY_IDS):
            findings.append(f"{model_id}: family order drifted")
    return findings


class SetupUtilityScorer:
    """Own-side production scoring over a fitted-utility artifact.

    ``utility(model_id, color, family_id, trait_vector)`` is the entire
    surface: there is no opponent parameter, no game parameter and no
    outcome parameter, and the intercept is not reachable from any scoring
    path. Model F ignores ``trait_vector``; Model T requires the base's
    frozen 35-field trait vector and standardizes it with the artifact's
    own frozen scaler copy.
    """

    def __init__(self, artifact: dict) -> None:
        findings = own_side_only_findings(artifact)
        if findings:
            raise Phase10UtilityFitError(
                f"artifact is not a pure own-side scorer: {findings}"
            )
        self.artifact_version = artifact["artifact_version"]
        self._family_index = {
            family_id: position for position, family_id in enumerate(FAMILY_IDS)
        }
        scaler = artifact["scaler"]
        self._mean = np.asarray(scaler["mean"], dtype=np.float64)
        self._std = np.asarray(scaler["std"], dtype=np.float64)
        self._models = {}
        for model_id, entry in artifact["models"].items():
            self._models[model_id] = {
                "offsets": {
                    "red": tuple(entry["family_offsets_effective"][0]),
                    "blue": tuple(entry["family_offsets_effective"][1]),
                },
                "weights": (
                    None
                    if entry["trait_weights"] is None
                    else {
                        "red": np.asarray(entry["trait_weights"][0], dtype=np.float64),
                        "blue": np.asarray(entry["trait_weights"][1], dtype=np.float64),
                    }
                ),
            }

    @staticmethod
    def from_path(path: "str | Path") -> "SetupUtilityScorer":
        return SetupUtilityScorer(json.loads(Path(path).read_text()))

    def standardized_features(self, trait_vector: dict) -> np.ndarray:
        features = np.asarray(trait_feature_vector(trait_vector), dtype=np.float64)
        divisor = np.where(self._std == 0.0, 1.0, self._std)
        standardized = (features - self._mean) / divisor
        return np.where(self._std == 0.0, 0.0, standardized)

    def utility(
        self,
        model_id: str,
        color: str,
        family_id: str,
        trait_vector: "dict | None" = None,
    ) -> float:
        """`u(s, c)` from own colour, own family and own trait vector only."""
        model = self._models.get(model_id)
        if model is None:
            raise Phase10UtilityFitError(f"unknown utility model {model_id!r}")
        if color not in ("red", "blue"):
            raise Phase10UtilityFitError(f"colour must be 'red' or 'blue', got {color!r}")
        position = self._family_index.get(family_id)
        if position is None:
            raise Phase10UtilityFitError(f"unknown family id {family_id!r}")
        value = model["offsets"][color][position]
        if model["weights"] is not None:
            if trait_vector is None:
                raise Phase10UtilityFitError(
                    "model_T scores a base from its trait vector; none was supplied"
                )
            value += float(np.dot(model["weights"][color], self.standardized_features(trait_vector)))
        return float(value)


__all__ = [
    "ACCEPTED_TRAIT_SCALER_DIGEST",
    "ARTIFACT_ROOT_KEYS",
    "FIT_INPUT_ALLOWLIST",
    "FITTED_UTILITY_RELATIVE_PATH",
    "FITTED_UTILITY_VERSION",
    "FORBIDDEN_FIT_FIELDS",
    "MODEL_ENTRY_KEYS",
    "AllowlistedRecord",
    "FitData",
    "FittedUtilityModel",
    "Phase10UtilityFitError",
    "SetupUtilityScorer",
    "build_fit_data",
    "fit_utility_model",
    "own_side_only_findings",
    "utility_models_artifact",
    "write_utility_models_artifact",
]
