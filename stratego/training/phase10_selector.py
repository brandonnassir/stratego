"""Phase 10 Agent 4: `setup_selector_v1` and `learned_setup_source_v1`.

Specification sources:

- `04_AGENT_4_SELECTOR_AND_PRODUCTION_SOURCE.md` (distribution construction,
  sampling identity, Phase 7 preservation, diversity audit, permitted-input
  boundary)
- `00_PHASE_10_SEQUENCE_AND_COMMON_CONTRACT.md` ("Selector semantics",
  "Exactly six candidates", "Diversity contract")

What this module is
-------------------
A **setup source**, and nothing else. It turns Agent 3's frozen own-side
utility into a distribution over the Phase 7 bases of one split, mixes that
distribution with `neutral_v1` at the frozen 0.35 / 0.65 weights, samples a
base from a seeded stream, and then hands the base to the accepted Phase 7
reflection/perturbation path unchanged. It fits nothing, evaluates no
strength, plays no game and selects no candidate — Agent 5 owns selection and
Agent 7 owns acceptance.

The six legal inputs
--------------------
A selector call may read only:

```text
own colour            requested Phase 7 split       selector identity
selector seed         the candidate base's own family and own trait vector
```

Nothing about the opponent — setup, family, base id, fingerprint, seed,
policy, or any outcome — reaches any scoring or sampling path here.
:class:`SelectorRequest` is the only way in, and its constructor rejects any
field outside that closed set, so an injection attempt raises rather than
being silently ignored. Utility is consumed exclusively through Agent 3's
accepted own-side scorer, whose entire surface is
``utility(model_id, colour, family_id, trait_vector)``: there is no opponent
argument to pass, no centering to re-derive by hand, and the fitted
Red-first intercept is not reachable from any path in this module.

Streams
-------
Six decisions, six domain-separated streams, no mutable global RNG cursor:

```text
mixture branch      phase10_seed.selector_branch_uniform   (selector_branch)
learned base draw   phase10_seed.selector_base_uniform     (selector_base)
neutral base draw   the accepted setup_sampler_v1 family/base streams
reflection          the accepted setup_sampler_v1 orientation stream
perturbation coin   the accepted setup_sampler_v1 perturbation stream
perturbation seed   the accepted setup_sampler_v1 intensity + seed streams
```

The first two are Phase 10 identities under the `strat-s10` tag; the last
four are the accepted Phase 7 streams, re-derived here through the public
:func:`stratego.setups.identity.derive_stream_seed` under the accepted
`neutral_v1` profile so that Phase 7 bytes stay untouched. That re-derivation
is an *adapter*, and the assignment requires an adapter to prove identical
output: :func:`neutral_branch_matches_accepted_sampler` compares a
neutral-branch draw field for field against
``sample_setup(split, seed, 'neutral_v1')``, and Agent 4's audit runs that
comparison on every neutral-branch draw it makes.

Because every stream is a pure hash of the draw's logical identity, worker
count, shard boundaries, call order and process restarts cannot move a single
draw, and resume is exact set subtraction by draw id.

What the learned branch changes, and what it cannot
---------------------------------------------------
Exactly one thing: *which base* is chosen. Reflection, the perturbation coin,
the swap count and the perturbation seed all come from the accepted Phase 7
streams, which depend on `(profile, split, seed)` — and, for the perturbation
seed, on the chosen base id, exactly as the accepted sampler does. So the
frozen post-selection path keeps its accepted marginals (reflection 0.5,
perturbation 0.5, swap count uniform over 1..6, Hamming window 2..12, frozen
retry rules) on both branches, and `neutral_v1` is consumed, never redefined.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..setups.contracts import SPLITS
from ..setups.families import FAMILY_IDS
from ..setups.identity import SetupLibraryError, derive_stream_seed
from ..setups.perturbation import encode_perturbation_seed
from ..setups.sampler import (
    NEUTRAL_PROFILE,
    SAMPLER_VERSION,
    SampledSetup,
    SetupLibraryIndex,
    build_descendant,
    load_library_index,
    rebuild_from_provenance,
    sample_setup,
)
from .phase10_contract import (
    CANDIDATE_MATRIX,
    DIVERSITY_THRESHOLDS,
    LEARNED_MIXTURE_WEIGHT,
    LEARNED_SETUP_SOURCE_VERSION,
    NEUTRAL_MIXTURE_WEIGHT,
    NEUTRAL_PROFILE_NAME,
    POST_SELECTION_PATH,
    SELECTOR_AUDIT_DRAWS,
    SELECTOR_BASE_ORDER,
    SETUP_SELECTOR_VERSION,
)
from .phase10_seed import (
    COLORS,
    parse_selector_audit_draw_id,
    selector_audit_draw_id,
    selector_audit_seed,
    selector_base_uniform,
    selector_branch_uniform,
)
from .phase10_utility_fit import FITTED_UTILITY_RELATIVE_PATH, SetupUtilityScorer

#: The canonical digest domain of a published probability vector. A change to
#: the digest payload is a new domain, never a silent re-hash.
DISTRIBUTION_DIGEST_DOMAIN = "phase10_selector_distribution_v1"

#: The two mixture branches, named so a record never carries a bare boolean.
BRANCH_NEUTRAL = "neutral"
BRANCH_LEARNED = "learned"
BRANCHES = (BRANCH_NEUTRAL, BRANCH_LEARNED)

#: The closed set of fields a selector call may carry. Anything else is an
#: information-safety failure, and :class:`SelectorRequest` raises on it.
ALLOWED_REQUEST_FIELDS = frozenset({"split", "color", "selector_seed"})

#: Field-name fragments that name opponent-private or outcome information.
#: A request carrying one is rejected by name as well as by the closed set,
#: so a positive control gets a message that says what it tried to inject.
FORBIDDEN_REQUEST_TOKENS = (
    "opponent",
    "enemy",
    "their",
    "outcome",
    "result",
    "winner",
    "win",
    "loss",
    "score",
    "reward",
    "value",
    "elo",
    "policy",
    "checkpoint",
    "matchup",
    "game_id",
    "fingerprint",
    "hidden",
    "truth",
    "path",
)

#: The audit's zero-tolerance counters, one per frozen
#: `SELECTOR_AUDIT_ZERO_TOLERANCE` entry. A missing counter is a failure, not
#: a pass, so the tuple is the single source both the auditor and the
#: acceptance gate read.
AUDIT_COUNTERS = (
    "illegal_setups",
    "inventory_errors",
    "stranded_sampled_setups",
    "split_violations",
    "provenance_mismatches",
    "determinism_mismatches",
    "non_finite_selector_values",
)


class Phase10SelectorError(ValueError):
    """Raised when a selector identity, request or distribution is malformed."""


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectorCandidate:
    """One of the exactly six frozen candidates. There is no seventh."""

    candidate_id: str
    utility_model: str
    temperature: float

    @property
    def selector_identity(self) -> str:
        """The candidate's `selector identity` — one of the six legal inputs.

        Self-describing on purpose: it names the production source version,
        the candidate, its utility model and its temperature, so re-pairing a
        candidate with a different model or temperature is a *different*
        identity and therefore a different stream, not a silent reuse of the
        old one.
        """
        return (
            f"{LEARNED_SETUP_SOURCE_VERSION}|k={self.candidate_id}"
            f"|m={self.utility_model}|T={self.temperature:.2f}"
        )

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "utility_model": self.utility_model,
            "temperature": self.temperature,
            "selector_identity": self.selector_identity,
        }


CANDIDATES = tuple(
    SelectorCandidate(
        candidate_id=entry["candidate_id"],
        utility_model=entry["utility_model"],
        temperature=float(entry["temperature"]),
    )
    for entry in CANDIDATE_MATRIX
)
CANDIDATE_BY_ID = {candidate.candidate_id: candidate for candidate in CANDIDATES}
assert len(CANDIDATES) == 6
assert len({candidate.selector_identity for candidate in CANDIDATES}) == 6


def candidate(candidate_id: str) -> SelectorCandidate:
    try:
        return CANDIDATE_BY_ID[candidate_id]
    except KeyError as error:
        raise Phase10SelectorError(
            f"unknown candidate {candidate_id!r}; the frozen six are "
            f"{sorted(CANDIDATE_BY_ID)}"
        ) from error


# ---------------------------------------------------------------------------
# The permitted-input boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectorRequest:
    """The complete input of one selector call, and nothing else.

    Three fields, because the other three legal inputs are properties of the
    selector rather than of the call: the selector identity is the source's
    own, and a candidate base's family and trait vector are read from the
    frozen library at scoring time. There is no field for an opponent, a
    matchup, a game, an outcome or a storage path, and :meth:`from_payload`
    refuses to build a request from a mapping that carries one.
    """

    split: str
    color: str
    selector_seed: int

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise Phase10SelectorError(
                f"split must be one of {sorted(SPLITS)}, got {self.split!r}"
            )
        if self.color not in COLORS:
            raise Phase10SelectorError(
                f"colour must be one of {list(COLORS)}, got {self.color!r}"
            )
        if (
            not isinstance(self.selector_seed, int)
            or isinstance(self.selector_seed, bool)
            or self.selector_seed < 0
        ):
            raise Phase10SelectorError(
                f"selector seed must be a non-negative int, got {self.selector_seed!r}"
            )

    @staticmethod
    def from_payload(payload: dict) -> "SelectorRequest":
        """Build a request from a mapping, rejecting every illegal field.

        The audited entry point, and the thing Agent 4's positive controls
        attack: a payload naming an opponent family, an opponent base id, a
        setup fingerprint, a policy id or a game outcome raises here, so an
        injected field can never be quietly dropped on the floor.
        """
        if not isinstance(payload, dict):
            raise Phase10SelectorError(
                f"a selector request is a mapping, got {type(payload).__name__}"
            )
        extra = sorted(set(payload) - ALLOWED_REQUEST_FIELDS)
        if extra:
            forbidden = sorted(
                name
                for name in extra
                if any(token in name.lower() for token in FORBIDDEN_REQUEST_TOKENS)
            )
            detail = (
                f"; these name opponent-private or outcome information: {forbidden}"
                if forbidden
                else ""
            )
            raise Phase10SelectorError(
                f"selector request carries fields outside the frozen allowlist "
                f"{sorted(ALLOWED_REQUEST_FIELDS)}: {extra}{detail}"
            )
        missing = sorted(ALLOWED_REQUEST_FIELDS - set(payload))
        if missing:
            raise Phase10SelectorError(f"selector request is missing fields: {missing}")
        return SelectorRequest(
            split=payload["split"],
            color=payload["color"],
            selector_seed=payload["selector_seed"],
        )

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "color": self.color,
            "selector_seed": self.selector_seed,
        }


# ---------------------------------------------------------------------------
# The exact distribution
# ---------------------------------------------------------------------------


def split_base_entries(split: str, index: "SetupLibraryIndex | None" = None) -> tuple:
    """Every base of one split, in the frozen selector base order.

    Ascending `(family_index, base_index)`: the frozen library enumeration
    order restricted to the split. This ordering is what the softmax vector,
    its digest and the inverse-CDF walk are all stated over, so it is derived
    in one place and never re-implemented at a call site.
    """
    if split not in SPLITS:
        raise Phase10SelectorError(f"unknown split: {split!r}")
    library = load_library_index() if index is None else index
    entries: list = []
    for family_id in FAMILY_IDS:
        entries.extend(library.eligible_bases(family_id, split))
    return tuple(entries)


@dataclass(frozen=True)
class SelectorDistribution:
    """The exact `p_neutral`, `p_learned` and `p_phase10` of one cell.

    A cell is one `(candidate, colour, split)`. Everything here is exact
    arithmetic over the whole split — never an empirical frequency — so the
    diversity contract is evaluated on the distribution itself and the
    large-sample audit only has to agree with it.
    """

    candidate_id: str
    utility_model: str
    temperature: float
    color: str
    split: str
    base_ids: "tuple[str, ...]"
    family_ids: "tuple[str, ...]"
    utilities: np.ndarray
    p_neutral: np.ndarray
    p_learned: np.ndarray
    p_mixed: np.ndarray
    #: The learned branch's inverse-CDF ladder. This is the cumulative
    #: **softmax** mass — `cumsum(p_learned)` — and deliberately *not*
    #: `cumsum(p_mixed)`: the branch coin has already applied the 0.35 neutral
    #: weight before this ladder is ever walked, so walking the mixed vector
    #: here would apply that weight a second time and realize
    #: `0.5775*neutral + 0.4225*learned` instead of the frozen 0.35 / 0.65.
    cumulative_learned: np.ndarray

    @property
    def base_count(self) -> int:
        return len(self.base_ids)

    @property
    def bases_per_family(self) -> int:
        return self.base_count // len(FAMILY_IDS)

    def finiteness(self) -> dict:
        """Every finiteness and normalization fact the gates read."""
        vectors = {
            "utilities": self.utilities,
            "p_neutral": self.p_neutral,
            "p_learned": self.p_learned,
            "p_mixed": self.p_mixed,
            "cumulative_learned": self.cumulative_learned,
        }
        return {
            "all_finite": all(bool(np.isfinite(v).all()) for v in vectors.values()),
            "non_finite_counts": {
                name: int((~np.isfinite(vector)).sum()) for name, vector in vectors.items()
            },
            "all_non_negative": all(
                bool((vector >= 0.0).all()) for vector in (self.p_neutral, self.p_learned, self.p_mixed)
            ),
            "sums": {
                "p_neutral": float(self.p_neutral.sum()),
                "p_learned": float(self.p_learned.sum()),
                "p_mixed": float(self.p_mixed.sum()),
            },
            "sum_deviations": {
                name: abs(float(vector.sum()) - 1.0)
                for name, vector in (
                    ("p_neutral", self.p_neutral),
                    ("p_learned", self.p_learned),
                    ("p_mixed", self.p_mixed),
                )
            },
            "cumulative_final": float(self.cumulative_learned[-1]),
            "cumulative_monotone": bool(np.all(np.diff(self.cumulative_learned) >= 0.0)),
        }

    def mixture_is_exact(self) -> bool:
        """Whether `p_mixed` is exactly `0.35*p_neutral + 0.65*p_learned`.

        Recomputed from the two components rather than trusted, and compared
        for bit equality: the mixture weights are frozen, so there is no
        tolerance to argue about.
        """
        recomputed = (
            NEUTRAL_MIXTURE_WEIGHT * self.p_neutral + LEARNED_MIXTURE_WEIGHT * self.p_learned
        )
        return bool(np.array_equal(recomputed, self.p_mixed))

    def probability_vector_digest(self) -> str:
        """The canonical, path-independent digest of `p_phase10`.

        Probabilities enter as `float.hex()`, an exact binary rendering, so
        the digest pins the actual float64 vector rather than a rounded
        decimal view of it, and two machines agree or visibly disagree.
        """
        return self._digest(self.p_mixed, "p_phase10")

    def component_digests(self) -> dict:
        return {
            "p_neutral": self._digest(self.p_neutral, "p_neutral"),
            "p_learned": self._digest(self.p_learned, "p_learned"),
            "p_phase10": self._digest(self.p_mixed, "p_phase10"),
            "utilities": self._digest(self.utilities, "utilities"),
        }

    def _digest(self, vector: np.ndarray, label: str) -> str:
        payload = "\n".join(
            [
                DISTRIBUTION_DIGEST_DOMAIN,
                label,
                self.candidate_id,
                self.utility_model,
                f"T={self.temperature:.2f}",
                self.color,
                self.split,
                f"n={self.base_count}",
                *[
                    f"{base_id}:{float(value).hex()}"
                    for base_id, value in zip(self.base_ids, vector)
                ],
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def family_probabilities(self) -> np.ndarray:
        """`P(family)` under `p_phase10`, in the frozen family order."""
        return self.p_mixed.reshape(len(FAMILY_IDS), self.bases_per_family).sum(axis=1)

    def diversity(self) -> dict:
        """Every diversity quantity the common contract names, exactly."""
        return diversity_metrics(self.p_mixed, self.split)

    def base_index_for_uniform(self, uniform: float) -> int:
        """The frozen learned-branch walk over cumulative **softmax** mass.

        Called only on the learned branch, so the ladder is `p_learned` and
        never `p_mixed`: the mixture's 0.35 neutral weight is applied by the
        branch coin upstream, and applying it again here would realize
        `0.5775*neutral + 0.4225*learned` rather than the frozen 0.35 / 0.65.

        `searchsorted(..., side='right')` returns the count of cumulative
        entries `<= u`, which is precisely the index the accepted linear walk
        would stop at; the final `min` is the float tail guard the contract
        names, for the case where rounding leaves the last cumulative entry a
        hair below 1.0.
        """
        position = int(
            np.searchsorted(self.cumulative_learned, float(uniform), side="right")
        )
        return min(position, self.base_count - 1)

    def to_dict(self) -> dict:
        diversity = self.diversity()
        return {
            "candidate_id": self.candidate_id,
            "utility_model": self.utility_model,
            "temperature": self.temperature,
            "color": self.color,
            "split": self.split,
            "base_count": self.base_count,
            "bases_per_family": self.bases_per_family,
            "base_order": SELECTOR_BASE_ORDER,
            "mixture": {
                "neutral_weight": NEUTRAL_MIXTURE_WEIGHT,
                "learned_weight": LEARNED_MIXTURE_WEIGHT,
                "exact": self.mixture_is_exact(),
            },
            "digests": self.component_digests(),
            "finiteness": self.finiteness(),
            "diversity": diversity,
            "utility_summary": {
                "min": float(self.utilities.min()),
                "max": float(self.utilities.max()),
                "mean": float(self.utilities.mean()),
            },
        }


def build_distribution(
    selector_candidate: SelectorCandidate,
    color: str,
    split: str,
    scorer: SetupUtilityScorer,
    index: "SetupLibraryIndex | None" = None,
) -> SelectorDistribution:
    """The exact distribution of one `(candidate, colour, split)` cell.

    Enumerates every base of the split in the frozen order, scores each one
    through Agent 3's accepted own-side scorer — own colour, own family, own
    trait vector, and nothing else — divides by the candidate's temperature,
    takes a finite softmax, and mixes with `neutral_v1` at the frozen
    weights. No opponent quantity is available to this function, and the
    fitted Red-first intercept is never read.
    """
    if color not in COLORS:
        raise Phase10SelectorError(f"colour must be one of {list(COLORS)}, got {color!r}")
    if selector_candidate.temperature <= 0.0:
        raise Phase10SelectorError(
            f"temperature must be positive, got {selector_candidate.temperature!r}"
        )
    entries = split_base_entries(split, index)
    base_ids = tuple(entry.base_setup_id for entry in entries)
    family_ids = tuple(entry.family_id for entry in entries)

    utilities = np.array(
        [
            scorer.utility(
                selector_candidate.utility_model,
                color,
                entry.family_id,
                entry.trait_vector,
            )
            for entry in entries
        ],
        dtype=np.float64,
    )
    if not np.isfinite(utilities).all():
        raise Phase10SelectorError(
            f"{selector_candidate.candidate_id}/{color}/{split}: the utility model "
            "produced a non-finite score"
        )

    # Shift by the maximum before exponentiating: the softmax is invariant
    # under it and it is what keeps exp() finite at the low temperatures.
    scaled = utilities / selector_candidate.temperature
    exponentiated = np.exp(scaled - scaled.max())
    p_learned = exponentiated / exponentiated.sum()

    # neutral_v1's base choice: uniform over the 16 families and uniform over
    # that family's bases inside the split. Every family contributes equally
    # many bases to a split, so the product is uniform over the split.
    count = len(entries)
    p_neutral = np.full(count, 1.0 / count, dtype=np.float64)

    p_mixed = NEUTRAL_MIXTURE_WEIGHT * p_neutral + LEARNED_MIXTURE_WEIGHT * p_learned
    # The learned branch's ladder is the softmax mass alone. See
    # SelectorDistribution.cumulative_learned for why it must not be p_mixed.
    cumulative_learned = np.cumsum(p_learned)

    return SelectorDistribution(
        candidate_id=selector_candidate.candidate_id,
        utility_model=selector_candidate.utility_model,
        temperature=selector_candidate.temperature,
        color=color,
        split=split,
        base_ids=base_ids,
        family_ids=family_ids,
        utilities=utilities,
        p_neutral=p_neutral,
        p_learned=p_learned,
        p_mixed=p_mixed,
        cumulative_learned=cumulative_learned,
    )


# ---------------------------------------------------------------------------
# Diversity, over the final mixed distribution
# ---------------------------------------------------------------------------


def _entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy in nats, with the frozen `0 log 0 = 0` convention."""
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def diversity_metrics(probabilities: np.ndarray, split: str) -> dict:
    """Every diversity quantity the common contract states, exactly.

    Stated over the **final mixed distribution**, per the contract's scope:
    family entropy and its normalization, the effective family count, the
    extreme family probabilities, and — per family — the normalized entropy
    of the conditional base distribution and its maximum.
    """
    vector = np.asarray(probabilities, dtype=np.float64)
    families = len(FAMILY_IDS)
    if vector.size % families:
        raise Phase10SelectorError(
            f"a split distribution must cover all {families} families evenly, "
            f"got {vector.size} entries"
        )
    per_family = vector.size // families
    grid = vector.reshape(families, per_family)
    family_mass = grid.sum(axis=1)

    family_entropy = _entropy(family_mass)
    normalized_family_entropy = family_entropy / math.log(families)
    effective_families = math.exp(family_entropy)

    conditional = grid / family_mass[:, None]
    within = []
    for row in conditional:
        within.append(_entropy(row) / math.log(per_family))
    within_array = np.array(within, dtype=np.float64)

    return {
        "split": split,
        "families": families,
        "bases_per_family": per_family,
        "family_entropy_nats": family_entropy,
        "normalized_family_entropy": normalized_family_entropy,
        "effective_families": effective_families,
        "min_family_probability": float(family_mass.min()),
        "max_family_probability": float(family_mass.max()),
        "family_probabilities": {
            family_id: float(mass) for family_id, mass in zip(FAMILY_IDS, family_mass)
        },
        "min_within_family_normalized_base_entropy": float(within_array.min()),
        "max_within_family_normalized_base_entropy": float(within_array.max()),
        "within_family_normalized_base_entropy": {
            family_id: float(value) for family_id, value in zip(FAMILY_IDS, within_array)
        },
        "max_conditional_base_probability": float(conditional.max()),
        "max_conditional_base_probability_per_family": {
            family_id: float(row.max()) for family_id, row in zip(FAMILY_IDS, conditional)
        },
        "effective_base_diversity": math.exp(_entropy(vector)),
    }


def evaluate_diversity(metrics: dict) -> dict:
    """The frozen thresholds applied to one cell's metrics.

    Every threshold in the common contract is non-strict; the two ceilings
    are written as `observed <= limit` rather than negated floors so a
    boundary value passes on both sides of the contract's wording.
    """
    thresholds = DIVERSITY_THRESHOLDS
    checks = {
        "normalized_family_entropy": metrics["normalized_family_entropy"]
        >= thresholds["normalized_family_entropy_min"],
        "effective_families": metrics["effective_families"]
        >= thresholds["effective_families_min"],
        "family_probability_min": metrics["min_family_probability"]
        >= thresholds["family_probability_min"],
        "family_probability_max": metrics["max_family_probability"]
        <= thresholds["family_probability_max"],
        "within_family_base_entropy": metrics["min_within_family_normalized_base_entropy"]
        >= thresholds["within_family_normalized_base_entropy_min"],
        "conditional_base_probability_max": metrics["max_conditional_base_probability"]
        <= thresholds["max_conditional_base_probability"],
    }
    return {
        "checks": {name: bool(value) for name, value in checks.items()},
        "thresholds": dict(thresholds),
        "all_pass": all(checks.values()),
        "failed": sorted(name for name, value in checks.items() if not value),
    }


# ---------------------------------------------------------------------------
# The accepted Phase 7 decision streams, re-derived
# ---------------------------------------------------------------------------


def _phase7_stream(field_name: str, split: str, seed: int) -> random.Random:
    """One accepted `setup_sampler_v1` decision stream under `neutral_v1`.

    Byte-for-byte the derivation the accepted sampler uses, expressed through
    the public :func:`derive_stream_seed` so no Phase 7 module is touched and
    no constant is restated: the profile name, the reflection and
    perturbation probabilities and the intensity weights are all read off the
    accepted :data:`NEUTRAL_PROFILE` object.
    """
    return random.Random(
        derive_stream_seed(
            f"{SAMPLER_VERSION}:{field_name}", NEUTRAL_PROFILE.name, split, int(seed)
        )
    )


def _uniform_index(rng: random.Random, count: int) -> int:
    return min(int(rng.random() * count), count - 1)


def _swap_count(rng: random.Random) -> int:
    """The accepted intensity draw, over `neutral_v1`'s own weights."""
    counts = NEUTRAL_PROFILE.swap_counts
    weights = NEUTRAL_PROFILE.intensity_weights
    target = rng.random() * sum(weights)
    accumulated = 0.0
    for count, weight in zip(counts, weights):
        accumulated += weight
        if accumulated >= target:
            return count
    return counts[-1]  # pragma: no cover - float tail guard


@dataclass(frozen=True)
class PostSelectionDecisions:
    """The frozen Phase 7 decisions that do not depend on the chosen base."""

    reflection_applied: bool
    perturbation_requested: bool
    swap_count: "int | None"

    def to_dict(self) -> dict:
        return {
            "reflection_applied": self.reflection_applied,
            "perturbation_requested": self.perturbation_requested,
            "swap_count": self.swap_count,
        }


def post_selection_decisions(split: str, seed: int) -> PostSelectionDecisions:
    """Reflection, the perturbation coin and the intensity, for one draw.

    Base-independent by construction, exactly as in the accepted sampler,
    which is why a learned base can be substituted for the neutral one
    without disturbing the frozen post-selection marginals.
    """
    reflection = (
        _phase7_stream("orientation", split, seed).random()
        < NEUTRAL_PROFILE.reflection_probability
    )
    requested = (
        _phase7_stream("perturbation", split, seed).random()
        < NEUTRAL_PROFILE.perturbation_probability
    )
    swap_count = _swap_count(_phase7_stream("intensity", split, seed)) if requested else None
    return PostSelectionDecisions(
        reflection_applied=bool(reflection),
        perturbation_requested=bool(requested),
        swap_count=swap_count,
    )


def perturbation_seed_for(split: str, seed: int, base_setup_id: str, swap_count: int) -> int:
    """The accepted composite perturbation seed of one `(draw, base)` pair."""
    raw_seed = derive_stream_seed(
        f"{SAMPLER_VERSION}:perturbation_seed",
        NEUTRAL_PROFILE.name,
        split,
        int(seed),
        base_setup_id,
        swap_count,
    )
    return encode_perturbation_seed(swap_count, raw_seed)


def neutral_branch_base_id(split: str, seed: int, index: "SetupLibraryIndex | None" = None) -> str:
    """The base `neutral_v1` would have chosen for this draw identity."""
    library = load_library_index() if index is None else index
    family_id = FAMILY_IDS[
        _uniform_index(_phase7_stream("family", split, seed), len(FAMILY_IDS))
    ]
    eligible = library.eligible_bases(family_id, split)
    return eligible[
        _uniform_index(_phase7_stream("base", split, seed), len(eligible))
    ].base_setup_id


# ---------------------------------------------------------------------------
# The production setup source
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectorDraw:
    """One complete `learned_setup_source_v1` output.

    The selector's own record sits at the top level — branch, candidate,
    identity, seed, split, colour and the two uniforms — and the accepted
    Phase 7 record is nested unchanged underneath as `setup_provenance`. A
    consumer therefore never has to infer which branch produced a draw from
    a Phase 7 field, which matters because `setup_provenance.sampler_profile`
    names the frozen *post-selection* profile (`neutral_v1`: reflection 0.5,
    perturbation 0.5, uniform 1..6) on **both** branches — it is a true
    statement about reflection and perturbation, and says nothing about how
    the base was chosen.
    """

    candidate_id: str
    selector_identity: str
    split: str
    color: str
    selector_seed: int
    branch: str
    branch_uniform: float
    base_uniform: "float | None"
    base_setup_id: str
    family_id: str
    setup: SampledSetup

    @property
    def final_setup_fingerprint(self) -> str:
        return self.setup.provenance["final_setup_fingerprint"]

    @property
    def setup_provenance(self) -> dict:
        return self.setup.provenance

    def oriented(self, player: int) -> "tuple[int, ...]":
        return self.setup.oriented(player)

    def selector_provenance(self) -> dict:
        """The Phase 10 half of the record: selection, not construction."""
        return {
            "selector_version": SETUP_SELECTOR_VERSION,
            "production_source_version": LEARNED_SETUP_SOURCE_VERSION,
            "candidate_id": self.candidate_id,
            "selector_identity": self.selector_identity,
            "split": self.split,
            "color": self.color,
            "selector_seed": self.selector_seed,
            "branch": self.branch,
            "branch_uniform": self.branch_uniform,
            "base_uniform": self.base_uniform,
            "base_setup_id": self.base_setup_id,
            "primary_family_id": self.family_id,
            "final_setup_fingerprint": self.final_setup_fingerprint,
        }

    def to_dict(self) -> dict:
        return {
            "selector": self.selector_provenance(),
            "setup_provenance": dict(self.setup.provenance),
        }


class LearnedSetupSource:
    """`learned_setup_source_v1` — the production setup source of one candidate.

    Construction is pure: give it a candidate, Agent 3's accepted scorer and
    the frozen library, and every distribution and every draw follows. The
    object holds no mutable RNG cursor and no per-call state, so two
    processes that build it independently produce identical draws for
    identical draw identities.
    """

    def __init__(
        self,
        selector_candidate: SelectorCandidate,
        scorer: SetupUtilityScorer,
        index: "SetupLibraryIndex | None" = None,
    ) -> None:
        self.candidate = selector_candidate
        self.scorer = scorer
        self.index = load_library_index() if index is None else index
        self._distributions: dict = {}
        self._entries: dict = {}

    @property
    def selector_identity(self) -> str:
        return self.candidate.selector_identity

    def distribution(self, color: str, split: str) -> SelectorDistribution:
        """The cached exact distribution of one cell."""
        key = (color, split)
        cached = self._distributions.get(key)
        if cached is None:
            cached = build_distribution(
                self.candidate, color, split, self.scorer, self.index
            )
            self._distributions[key] = cached
            self._entries[split] = split_base_entries(split, self.index)
        return cached

    def _entry(self, split: str, position: int):
        entries = self._entries.get(split)
        if entries is None:
            entries = split_base_entries(split, self.index)
            self._entries[split] = entries
        return entries[position]

    def draw(self, request: SelectorRequest) -> SelectorDraw:
        """One complete selector draw: branch, base, then the Phase 7 path.

        The whole procedure, in the frozen order: a branch coin from the
        `selector_branch` stream; a base from either the accepted `neutral_v1`
        family/base streams or the `selector_base` inverse-CDF walk; then the
        accepted Phase 7 reflection, perturbation coin, intensity, composite
        perturbation seed and the complete final-output validation stack,
        delegated to :func:`build_descendant` — the accepted sampler's single
        construction path, shared with `sample_setup` and
        `rebuild_from_provenance`.
        """
        if not isinstance(request, SelectorRequest):
            raise Phase10SelectorError(
                "a selector draw takes a SelectorRequest; build one with "
                "SelectorRequest.from_payload so illegal fields are rejected"
            )
        split, color, seed = request.split, request.color, request.selector_seed

        branch_uniform = selector_branch_uniform(
            self.selector_identity, split, color, seed
        )
        if not math.isfinite(branch_uniform):
            raise Phase10SelectorError("the mixture branch uniform is not finite")

        base_uniform: "float | None" = None
        if branch_uniform < NEUTRAL_MIXTURE_WEIGHT:
            branch = BRANCH_NEUTRAL
            base_id = neutral_branch_base_id(split, seed, self.index)
            base_entry = self.index.base(base_id)
        else:
            branch = BRANCH_LEARNED
            distribution = self.distribution(color, split)
            base_uniform = selector_base_uniform(
                self.selector_identity, split, color, seed
            )
            if not math.isfinite(base_uniform):
                raise Phase10SelectorError("the learned base uniform is not finite")
            base_entry = self._entry(split, distribution.base_index_for_uniform(base_uniform))

        decisions = post_selection_decisions(split, seed)
        perturbation_seed = (
            perturbation_seed_for(
                split, seed, base_entry.base_setup_id, decisions.swap_count
            )
            if decisions.perturbation_requested
            else None
        )
        setup = build_descendant(
            base_entry,
            reflection_applied=decisions.reflection_applied,
            perturbation_requested=decisions.perturbation_requested,
            perturbation_seed=perturbation_seed,
            profile_name=NEUTRAL_PROFILE.name,
            draw_seed=int(seed),
        )
        return SelectorDraw(
            candidate_id=self.candidate.candidate_id,
            selector_identity=self.selector_identity,
            split=split,
            color=color,
            selector_seed=int(seed),
            branch=branch,
            branch_uniform=branch_uniform,
            base_uniform=base_uniform,
            base_setup_id=base_entry.base_setup_id,
            family_id=base_entry.family_id,
            setup=setup,
        )

    def draw_from_payload(self, payload: dict) -> SelectorDraw:
        """Draw from a raw mapping, rejecting every illegal input field."""
        return self.draw(SelectorRequest.from_payload(payload))

    def audit_draw(self, draw_ordinal: int, color: str, split: str) -> "tuple[str, SelectorDraw]":
        """The draw addressed by one selector-audit draw id.

        The audit's unit of work. The id and its seed come from Agent 1's
        frozen `selector_audit` domain, so a draw is addressable, resume is
        exact set subtraction by draw id, and re-sharding cannot move it.
        """
        draw_id = selector_audit_draw_id(
            self.candidate.candidate_id, split, color, draw_ordinal
        )
        seed = selector_audit_seed(
            self.candidate.candidate_id, split, color, draw_ordinal
        )
        return draw_id, self.draw(
            SelectorRequest(split=split, color=color, selector_seed=seed)
        )


def neutral_baseline_draw(
    split: str, seed: int, index: "SetupLibraryIndex | None" = None
) -> SampledSetup:
    """The `neutral_v1` baseline draw, straight through the accepted sampler.

    Phase 10 consumes `neutral_v1`; it never redefines it. This is a one-line
    delegation on purpose — the baseline arm of every paired comparison is
    the accepted Phase 7 API called with the accepted profile, with no
    Phase 10 code in the path at all.
    """
    return sample_setup(split, seed, NEUTRAL_PROFILE_NAME, index)


def neutral_branch_matches_accepted_sampler(
    draw: SelectorDraw, index: "SetupLibraryIndex | None" = None
) -> "list[str]":
    """Every field on which a neutral-branch draw differs from the baseline.

    The adapter proof the assignment requires. A neutral-branch draw must be
    bit-identical to `sample_setup(split, seed, 'neutral_v1')` — same base,
    same reflection, same perturbation identity, same final fingerprint, same
    complete provenance record. An empty list is the proof; anything else
    means the re-derived Phase 7 streams drifted from the accepted ones.
    """
    if draw.branch != BRANCH_NEUTRAL:
        raise Phase10SelectorError(
            "the accepted-sampler comparison applies to neutral-branch draws"
        )
    baseline = neutral_baseline_draw(draw.split, draw.selector_seed, index)
    findings: list[str] = []
    if baseline.canonical != draw.setup.canonical:
        findings.append("canonical setup differs from the accepted sampler's")
    for name in sorted(set(baseline.provenance) | set(draw.setup_provenance)):
        if baseline.provenance.get(name) != draw.setup_provenance.get(name):
            findings.append(
                f"provenance {name}: accepted {baseline.provenance.get(name)!r} vs "
                f"selector {draw.setup_provenance.get(name)!r}"
            )
    return findings


def learned_branch_shares_phase7_decisions(
    draw: SelectorDraw, index: "SetupLibraryIndex | None" = None
) -> "list[str]":
    """Every base-independent decision on which a learned draw drifted.

    A learned-branch draw may differ from the `neutral_v1` baseline in
    exactly one thing — the base — so reflection, the perturbation coin and
    the swap count must still be the accepted sampler's for this draw
    identity. This is what keeps the frozen post-selection marginals intact
    on the learned branch.
    """
    if draw.branch != BRANCH_LEARNED:
        raise Phase10SelectorError(
            "the shared-decision comparison applies to learned-branch draws"
        )
    baseline = neutral_baseline_draw(draw.split, draw.selector_seed, index)
    provenance = draw.setup_provenance
    findings: list[str] = []
    for name in ("reflection_applied", "perturbation_requested", "perturbation_swap_count"):
        if baseline.provenance.get(name) != provenance.get(name):
            findings.append(
                f"{name}: accepted {baseline.provenance.get(name)!r} vs selector "
                f"{provenance.get(name)!r}"
            )
    if baseline.provenance.get("sampler_profile") != provenance.get("sampler_profile"):
        findings.append("sampler_profile is not the accepted neutral_v1")
    return findings


# ---------------------------------------------------------------------------
# Per-draw verification
# ---------------------------------------------------------------------------


def classify_construction_failure(message: str) -> str:
    """Which zero-tolerance counter a failed construction belongs to.

    The accepted `validate_sampled_setup` prefixes every failure with its
    category, so the classification reads that prefix rather than guessing;
    anything unrecognised counts as an illegal setup, which is the
    conservative bucket.
    """
    lowered = message.lower()
    if "inventory" in lowered:
        return "inventory_errors"
    if "stranded" in lowered:
        return "stranded_sampled_setups"
    if "split" in lowered:
        return "split_violations"
    return "illegal_setups"


def verify_draw(
    source: LearnedSetupSource,
    draw: SelectorDraw,
    draw_id: str,
    *,
    cross_check_accepted_sampler: bool = True,
) -> dict:
    """Everything one audited draw must satisfy, recomputed from the draw.

    Legality is already established by construction — `build_descendant` runs
    the complete final-output validation stack and raises rather than
    returning an invalid setup — so what remains is what construction does
    not check: that the draw landed in the requested split, that its base is
    one of that split's bases, that no selector value is non-finite, that the
    recorded provenance rebuilds to the identical setup, and that the branch
    agrees with the accepted sampler where the contract says it must.
    """
    counters = {name: 0 for name in AUDIT_COUNTERS}
    findings: list[str] = []
    identity = parse_selector_audit_draw_id(draw_id)

    if identity["candidate_id"] != draw.candidate_id:
        findings.append("draw id names a different candidate")
    if identity["split"] != draw.split or identity["color"] != draw.color:
        findings.append("draw id names a different colour or split")

    provenance = draw.setup_provenance
    if draw.setup.split != draw.split or provenance["split"] != draw.split:
        counters["split_violations"] += 1
        findings.append(
            f"split mismatch: requested {draw.split!r}, produced {draw.setup.split!r}"
        )
    if source.index.base(draw.base_setup_id).split != draw.split:
        counters["split_violations"] += 1
        findings.append(f"base {draw.base_setup_id} does not belong to {draw.split!r}")
    if draw.family_id not in FAMILY_IDS:
        findings.append(f"unknown family {draw.family_id!r}")

    values = [draw.branch_uniform] + ([] if draw.base_uniform is None else [draw.base_uniform])
    if not all(math.isfinite(value) for value in values):
        counters["non_finite_selector_values"] += 1
        findings.append("a selector uniform is not finite")
    if not 0.0 <= draw.branch_uniform < 1.0:
        counters["non_finite_selector_values"] += 1
        findings.append(f"branch uniform {draw.branch_uniform} is outside [0, 1)")
    if draw.base_uniform is not None and not 0.0 <= draw.base_uniform < 1.0:
        counters["non_finite_selector_values"] += 1
        findings.append(f"base uniform {draw.base_uniform} is outside [0, 1)")

    expected_branch = (
        BRANCH_NEUTRAL if draw.branch_uniform < NEUTRAL_MIXTURE_WEIGHT else BRANCH_LEARNED
    )
    if draw.branch != expected_branch:
        findings.append(f"branch {draw.branch!r} contradicts the mixture coin")

    try:
        rebuilt = rebuild_from_provenance(provenance, source.index)
        if rebuilt.provenance != provenance or rebuilt.canonical != draw.setup.canonical:
            counters["provenance_mismatches"] += 1
            findings.append("provenance does not rebuild to the identical setup")
    except SetupLibraryError as error:
        counters["provenance_mismatches"] += 1
        findings.append(f"provenance rebuild failed: {error}")

    if cross_check_accepted_sampler:
        drift = (
            neutral_branch_matches_accepted_sampler(draw, source.index)
            if draw.branch == BRANCH_NEUTRAL
            else learned_branch_shares_phase7_decisions(draw, source.index)
        )
        if drift:
            counters["provenance_mismatches"] += 1
            findings.extend(drift)

    return {"counters": counters, "findings": findings, "ok": not findings}


# ---------------------------------------------------------------------------
# The frozen contract document
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def load_scorer(path: str = FITTED_UTILITY_RELATIVE_PATH) -> SetupUtilityScorer:
    """Agent 3's accepted own-side scorer, cached per path."""
    return SetupUtilityScorer.from_path(path)


def selector_contract_document(distribution_digests: "dict | None" = None) -> dict:
    """The machine-readable `setup_selector_v1` / `learned_setup_source_v1` record."""
    document = {
        "selector_version": SETUP_SELECTOR_VERSION,
        "production_source_version": LEARNED_SETUP_SOURCE_VERSION,
        "baseline_profile": NEUTRAL_PROFILE_NAME,
        "baseline_is_not_a_candidate": True,
        "candidates": [entry.to_dict() for entry in CANDIDATES],
        "candidate_count": len(CANDIDATES),
        "allowed_request_fields": sorted(ALLOWED_REQUEST_FIELDS),
        "selector_inputs": [
            "own colour",
            "requested Phase 7 split",
            "selector identity",
            "selector seed",
            "the candidate base's own family",
            "the candidate base's own trait vector",
        ],
        "mixture": {
            "formula": "p_phase10 = 0.35 * p_neutral_v1 + 0.65 * p_learned",
            "neutral_weight": NEUTRAL_MIXTURE_WEIGHT,
            "learned_weight": LEARNED_MIXTURE_WEIGHT,
        },
        "learned_distribution": "p_learned(s | c, split) = softmax(u(s, c) / T)",
        "base_order": SELECTOR_BASE_ORDER,
        "neutral_branch_definition": (
            "the base the accepted setup_sampler_v1 would have taken for "
            "(split, selector_seed, profile='neutral_v1'), so a neutral-branch "
            "draw is bit-identical to the neutral baseline's draw"
        ),
        "streams": {
            "mixture_branch": "phase10_seed.selector_branch_uniform (selector_branch)",
            "learned_base_draw": "phase10_seed.selector_base_uniform (selector_base)",
            "neutral_base_draw": "setup_sampler_v1 family/base streams under neutral_v1",
            "reflection": "setup_sampler_v1 orientation stream under neutral_v1",
            "perturbation_decision": "setup_sampler_v1 perturbation stream under neutral_v1",
            "perturbation_seed": (
                "setup_sampler_v1 intensity + perturbation_seed streams under "
                "neutral_v1, folded into the seed_encoding_v1 composite"
            ),
            "independence": (
                "every stream is a pure hash of the draw's logical identity; no "
                "mutable global RNG cursor exists, so worker count, shard "
                "boundaries, call order and process restarts cannot move a draw"
            ),
        },
        "post_selection_path": dict(POST_SELECTION_PATH),
        "post_selection_delegation": (
            "build_descendant, the accepted sampler's single construction path, "
            "shared with sample_setup and rebuild_from_provenance"
        ),
        "adapter_proof": (
            "neutral_branch_matches_accepted_sampler compares a neutral-branch "
            "draw field for field against sample_setup(split, seed, 'neutral_v1'); "
            "learned_branch_shares_phase7_decisions checks that a learned draw "
            "differs from that baseline in the base alone"
        ),
        "sampler_profile_reading": (
            "setup_provenance.sampler_profile is 'neutral_v1' on both branches "
            "because it names the frozen post-selection profile that was used "
            "(reflection 0.5, perturbation 0.5, uniform 1..6); the branch and the "
            "learned identity live in the Phase 10 selector provenance beside it"
        ),
        "audit_draw_identity": (
            "phase10_selector_audit_v1|ms=<master>|k=<candidate>|s=<split>"
            "|c=<colour>|n=<ordinal:05d>, seeded by selector_audit_seed"
        ),
        "audit_draws_per_candidate_colour_split": SELECTOR_AUDIT_DRAWS,
        "audit_counters": list(AUDIT_COUNTERS),
        "diversity": {
            "scope": (
                "the final mixed distribution, for every candidate, colour and split"
            ),
            "thresholds": dict(DIVERSITY_THRESHOLDS),
        },
        "distribution_digest": {
            "domain": DISTRIBUTION_DIGEST_DOMAIN,
            "payload": (
                "domain, label, candidate id, utility model, T, colour, split, "
                "base count, then '<base id>:<float.hex()>' per base in the frozen "
                "base order; SHA-256 over the newline-joined text"
            ),
        },
        "no_strength_selection": (
            "this module fits nothing, plays no game, reads no outcome and "
            "selects no candidate; Agent 5 owns selection and Agent 7 acceptance"
        ),
    }
    if distribution_digests is not None:
        document["distribution_digests"] = distribution_digests
    return document


def selector_contract_digest(distribution_digests: "dict | None" = None) -> str:
    """SHA-256 over the contract's canonical JSON — the frozen convention."""
    canonical = json.dumps(
        selector_contract_document(distribution_digests),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


__all__ = [
    "ALLOWED_REQUEST_FIELDS",
    "AUDIT_COUNTERS",
    "BRANCHES",
    "BRANCH_LEARNED",
    "BRANCH_NEUTRAL",
    "CANDIDATES",
    "CANDIDATE_BY_ID",
    "DISTRIBUTION_DIGEST_DOMAIN",
    "FORBIDDEN_REQUEST_TOKENS",
    "LearnedSetupSource",
    "Phase10SelectorError",
    "PostSelectionDecisions",
    "SelectorCandidate",
    "SelectorDistribution",
    "SelectorDraw",
    "SelectorRequest",
    "build_distribution",
    "candidate",
    "classify_construction_failure",
    "diversity_metrics",
    "evaluate_diversity",
    "learned_branch_shares_phase7_decisions",
    "load_scorer",
    "neutral_baseline_draw",
    "neutral_branch_base_id",
    "neutral_branch_matches_accepted_sampler",
    "perturbation_seed_for",
    "post_selection_decisions",
    "selector_contract_digest",
    "selector_contract_document",
    "split_base_entries",
    "verify_draw",
]
