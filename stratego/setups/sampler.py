"""The deterministic runtime setup sampler: `setup_sampler_v1`.

Specification sources:

- `04_AGENT_4_REFLECTION_AND_PERTURBATION.md` (sampling contract, family/base
  sampling, perturbation identity, final-output validation, split isolation)
- `00_PHASE_7_SEQUENCE_AND_COMMON_CONTRACT.md` (reflection rule, split
  permanence, observer safety, expected Phase 8 output)

The sampling pipeline
---------------------
```text
choose split      caller supplies it; it is never re-chosen
choose family     uniform over the 16 primary families
choose base       uniform over that family's bases inside the split
optional perturbation   constrained, family-preserving, frozen-validated
choose orientation      seeded 50/50 left-right reflection
final validation        frozen engine + frozen contracts, from scratch
return setup + provenance
```

Every descendant inherits its base's `split`, `primary_family_id` and
`base_setup_id` verbatim. The sampler never writes to the base library, never
reassigns a split, and never consults a game outcome, a model score or any
strength signal — the only inputs are the caller's split, the caller's seed,
and the frozen profile.

Uniformity
----------
Family selection is uniform over all 16 families and base selection is uniform
over the 500-per-family / 400-50-50-per-split index ranges, so no family gains
mass because its generator needed more candidates, and every base inside a
split carries equal weight. Reflection is an independent seeded coin.

Determinism and rebuild
-----------------------
Each decision draws from its own domain-separated `derive_stream_seed` stream
keyed by `(sampler_version, profile, split, seed)`, so decisions are
independent and no mutable global RNG state is consumed. Because every
decision is recorded, :func:`rebuild_from_provenance` reconstructs a sampled
descendant — setup, perturbation attempts, rejections and fingerprints alike —
from provenance alone, without replaying the draw that produced it.

Observer safety
---------------
Provenance is training/debug metadata. It names the base, the family, the
split and the seeds, so it must never reach a move-policy input. Nothing in
this module touches `observation_v2_1_127ch`, the model contract or
`trajectory_v1`; Agent 5 owns proving the boundary holds in the pipeline.
"""

import json
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..engine.setup import SetupError, deserialize_setup, serialize_setup, validate_setup
from .contracts import (
    BASES_PER_FAMILY,
    LIBRARY_JSONL_PATH,
    SETUP_FAMILY_VERSION,
    SETUP_GENERATOR_CONTRACT_VERSION,
    SETUP_LIBRARY_VERSION,
    SETUP_TRAIT_VECTOR_VERSION,
    SPLITS,
    TEST_PER_FAMILY,
    TRAIN_PER_FAMILY,
    VALIDATION_PER_FAMILY,
    split_for_base_index,
)
from .families import FAMILY_BY_ID, FAMILY_IDS, family_contract
from .identity import (
    CANONICAL_CELLS,
    SetupLibraryError,
    class_fingerprint,
    content_fingerprint,
    derive_stream_seed,
    orient_setup,
    reflect_canonical,
)
from .library import (
    FORBIDDEN_ENTRY_FIELD_TOKENS,
    library_content_digest,
    read_library_jsonl,
)
from .mobility import setup_has_initial_mobility
from . import perturbation as perturbation_v1
from .perturbation import (
    MAX_SWAP_COUNT,
    MIN_SWAP_COUNT,
    PERTURBATION_VERSION,
    PerturbationResult,
    decode_perturbation_seed,
    encode_perturbation_seed,
    operator_mix_document,
    perturb_setup,
)
from .traits import compute_trait_vector

#: Version identifier of this sampler. A semantic change to the decision
#: order, the stream derivation, the provenance schema or the validation stack
#: is a new identifier, never a silent reinterpretation.
SAMPLER_VERSION = "setup_sampler_v1"

#: Per-split base-index ranges inside every family, restated from Agent 1's
#: frozen split rule so base selection is split-restricted by construction and
#: not by a post-hoc filter.
SPLIT_BASE_RANGES = {
    "train": (0, TRAIN_PER_FAMILY),
    "validation": (TRAIN_PER_FAMILY, TRAIN_PER_FAMILY + VALIDATION_PER_FAMILY),
    "test": (
        TRAIN_PER_FAMILY + VALIDATION_PER_FAMILY,
        TRAIN_PER_FAMILY + VALIDATION_PER_FAMILY + TEST_PER_FAMILY,
    ),
}
assert set(SPLIT_BASE_RANGES) == set(SPLITS)
assert SPLIT_BASE_RANGES["test"][1] == BASES_PER_FAMILY

#: Provenance keys the Agent 4 assignment requires at minimum.
REQUIRED_PROVENANCE_FIELDS = (
    "setup_library_version",
    "sampler_version",
    "split",
    "primary_family_id",
    "base_setup_id",
    "reflection_applied",
    "perturbation_applied",
    "perturbation_version",
    "perturbation_id",
    "perturbation_seed",
    "final_setup_fingerprint",
)

#: The complete provenance schema, in the order :func:`_provenance` emits it.
#: `tests/setups/test_sampler.py` pins this tuple to the record builder, so the
#: schema can be published in the contract artifact without probing the
#: builder with a fake entry.
PROVENANCE_FIELDS = (
    "setup_library_version",
    "contract_version",
    "family_contract_version",
    "trait_schema_version",
    "sampler_version",
    "sampler_profile",
    "perturbation_version",
    "split",
    "primary_family_id",
    "family_key",
    "base_setup_id",
    "base_index",
    "base_fingerprint",
    "reflection_applied",
    "perturbation_requested",
    "perturbation_applied",
    "perturbation_exhausted",
    "perturbation_swap_count",
    "perturbation_seed",
    "perturbation_id",
    "perturbation_max_attempts",
    "perturbation_attempts",
    "perturbation_accepted_attempt_index",
    "perturbation_hamming_from_base",
    "draw_seed",
    "final_setup",
    "final_setup_fingerprint",
    "final_setup_class_fingerprint",
)


# ---------------------------------------------------------------------------
# Sampler profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplerProfile:
    """A named, versioned set of sampler mixing parameters.

    A profile controls *how often* a branch is taken; it can never control
    whether a descendant is legal, which family it belongs to or which split
    it came from. Agent 6 owns the final Phase 8 profile freeze; the profiles
    here are candidates plus the two single-branch instruments.
    """

    name: str
    #: Probability that a sampled output is perturbed rather than
    #: reflection-only.
    perturbation_probability: float
    #: Relative weights over swap counts `MIN_SWAP_COUNT..MAX_SWAP_COUNT`.
    #: The chosen count is *encoded into* the perturbation seed; once the
    #: seed is emitted, perturbation execution is profile-independent.
    intensity_weights: "tuple[float, ...]"
    #: Probability of applying left-right reflection.
    reflection_probability: float = 0.5
    rationale: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.perturbation_probability <= 1.0:
            raise SetupLibraryError(
                f"perturbation_probability must be in [0, 1], got "
                f"{self.perturbation_probability}"
            )
        if not 0.0 <= self.reflection_probability <= 1.0:
            raise SetupLibraryError(
                f"reflection_probability must be in [0, 1], got "
                f"{self.reflection_probability}"
            )
        expected = MAX_SWAP_COUNT - MIN_SWAP_COUNT + 1
        if len(self.intensity_weights) != expected:
            raise SetupLibraryError(
                f"intensity_weights must cover swap counts "
                f"{MIN_SWAP_COUNT}..{MAX_SWAP_COUNT} ({expected} entries), got "
                f"{len(self.intensity_weights)}"
            )
        if any(weight < 0.0 for weight in self.intensity_weights):
            raise SetupLibraryError("intensity weights must be non-negative")
        if sum(self.intensity_weights) <= 0.0:
            raise SetupLibraryError("intensity weights must not be all zero")

    @property
    def swap_counts(self) -> "tuple[int, ...]":
        return tuple(range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sampler_version": SAMPLER_VERSION,
            "perturbation_version": PERTURBATION_VERSION,
            "family_selection": "uniform over the 16 primary families",
            "base_selection": (
                "uniform over the family's base indices inside the requested "
                "split"
            ),
            "reflection_probability": self.reflection_probability,
            "perturbation_probability": self.perturbation_probability,
            "swap_counts": list(self.swap_counts),
            "intensity_weights": list(self.intensity_weights),
            "perturbation_execution": (
                "profile-independent: the profile chooses whether to perturb "
                "and encodes the intensity into the perturbation seed; the "
                "descendant is then a pure function of (base_setup_id, "
                "sampler_version, perturbation_seed)"
            ),
            "rationale": self.rationale,
        }


#: The neutral candidate training profile. Family and base selection are
#: uniform; reflection is a fair coin; perturbation is a fair coin; and the
#: intensity mix is **uniform** over the whole frozen swap window, so no
#: structural claim is smuggled into the default. Agent 6 makes the final
#: freeze decision.
NEUTRAL_PROFILE = SamplerProfile(
    name="neutral_v1",
    perturbation_probability=0.5,
    intensity_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    reflection_probability=0.5,
    rationale=(
        "uniform family, uniform base within split, fair reflection coin, "
        "fair perturbation coin and a uniform intensity mix over the frozen "
        "Hamming window; no branch is weighted by any structural or strength "
        "argument, so the default asserts nothing Agent 6 has not frozen"
    ),
)

#: Reflection-only instrument: the static library plus both orientations.
REFLECTION_ONLY_PROFILE = SamplerProfile(
    name="reflection_only_v1",
    perturbation_probability=0.0,
    intensity_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    rationale="acceptance instrument: exercises the unperturbed path only",
)

#: Perturbation-only instrument: every output carries a perturbation.
PERTURBATION_ONLY_PROFILE = SamplerProfile(
    name="perturbation_only_v1",
    perturbation_probability=1.0,
    intensity_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    rationale="acceptance instrument: exercises the perturbed path only",
)

PROFILES = {
    profile.name: profile
    for profile in (NEUTRAL_PROFILE, REFLECTION_ONLY_PROFILE, PERTURBATION_ONLY_PROFILE)
}

DEFAULT_PROFILE = NEUTRAL_PROFILE


def sampler_profile(name: str) -> SamplerProfile:
    try:
        return PROFILES[name]
    except KeyError as error:
        raise SetupLibraryError(f"unknown sampler profile: {name!r}") from error


# ---------------------------------------------------------------------------
# Library index
# ---------------------------------------------------------------------------


class SetupLibraryIndex:
    """The materialized base library, indexed for split-restricted sampling.

    Built once from the production JSONL and then read-only. Construction
    re-checks the frozen counts and re-derives every split from the base
    index, so a tampered or truncated library cannot be sampled from silently.
    """

    __slots__ = ("entries", "content_digest", "_by_family_split", "_by_id", "_path")

    def __init__(self, entries, path: "str | None" = None) -> None:
        materialized = tuple(entries)
        if len(materialized) != len(FAMILY_IDS) * BASES_PER_FAMILY:
            raise SetupLibraryError(
                f"expected {len(FAMILY_IDS) * BASES_PER_FAMILY} base entries, "
                f"got {len(materialized)}"
            )
        by_family_split: dict = {
            (family_id, split): [] for family_id in FAMILY_IDS for split in SPLITS
        }
        by_id: dict = {}
        for entry in materialized:
            if entry.family_id not in FAMILY_BY_ID:
                raise SetupLibraryError(f"unknown family id in library: {entry.family_id!r}")
            expected_split = split_for_base_index(entry.base_index)
            if entry.split != expected_split:
                raise SetupLibraryError(
                    f"{entry.base_setup_id}: stored split {entry.split!r} "
                    f"contradicts the frozen split rule ({expected_split!r})"
                )
            by_family_split[(entry.family_id, entry.split)].append(entry)
            if entry.base_setup_id in by_id:
                raise SetupLibraryError(f"duplicate base id: {entry.base_setup_id}")
            by_id[entry.base_setup_id] = entry

        expected_counts = {
            "train": TRAIN_PER_FAMILY,
            "validation": VALIDATION_PER_FAMILY,
            "test": TEST_PER_FAMILY,
        }
        for (family_id, split), members in by_family_split.items():
            if len(members) != expected_counts[split]:
                raise SetupLibraryError(
                    f"{family_id}/{split}: expected {expected_counts[split]} "
                    f"bases, got {len(members)}"
                )
            members.sort(key=lambda entry: entry.base_index)

        self.entries = materialized
        self._by_family_split = {
            key: tuple(members) for key, members in by_family_split.items()
        }
        self._by_id = by_id
        self._path = path
        self.content_digest = library_content_digest(materialized)

    def eligible_bases(self, family_id: str, split: str) -> tuple:
        """The family's base entries inside `split`, base-index ascending."""
        if family_id not in FAMILY_BY_ID:
            raise SetupLibraryError(f"unknown family id: {family_id!r}")
        if split not in SPLITS:
            raise SetupLibraryError(f"unknown split: {split!r}")
        return self._by_family_split[(family_id, split)]

    def base(self, base_setup_id: str):
        try:
            return self._by_id[base_setup_id]
        except KeyError as error:
            raise SetupLibraryError(f"unknown base_setup_id: {base_setup_id!r}") from error

    def to_dict(self) -> dict:
        return {
            "library_version": SETUP_LIBRARY_VERSION,
            "library_jsonl_path": self._path,
            "entry_count": len(self.entries),
            "library_content_digest": self.content_digest,
        }


@lru_cache(maxsize=4)
def load_library_index(path: str = LIBRARY_JSONL_PATH) -> SetupLibraryIndex:
    """Load and cache the materialized base library for sampling.

    Read-only: the sampler never writes to `path`. The cache exists so Phase 8
    can call :func:`sample_setup` in a hot loop without re-reading 8,000
    entries per draw.
    """
    resolved = str(Path(path))
    return SetupLibraryIndex(read_library_jsonl(resolved), path=resolved)


# ---------------------------------------------------------------------------
# Final-output validation
# ---------------------------------------------------------------------------


def validate_sampled_setup(
    canonical: "tuple[int, ...]",
    base_entry,
    split: str,
    family_id: str,
) -> "list[str]":
    """Every final-output validation failure, recomputed from scratch.

    Runs the complete Agent 4 acceptance list on a finished sampler output —
    exact inventory, engine setup validation, initial-mobility quality, base
    split unchanged, base primary family unchanged, family required
    predicates, and serialization/fingerprint round-trip — without trusting
    any decision the sampler made along the way. An empty list is the only
    acceptable result; :func:`build_descendant` raises otherwise.
    """
    failures: list[str] = []

    if len(canonical) != CANONICAL_CELLS:
        failures.append(
            f"length: expected {CANONICAL_CELLS} canonical cells, got {len(canonical)}"
        )
        return failures

    try:
        validate_setup(canonical, 0)
    except SetupError as error:
        failures.append(f"inventory/legality: {error}")
        return failures

    if base_entry.split != split:
        failures.append(
            f"split migration: base {base_entry.base_setup_id} is "
            f"{base_entry.split!r} but the output claims {split!r}"
        )
    if split_for_base_index(base_entry.base_index) != split:
        failures.append(
            f"split rule: base index {base_entry.base_index} belongs to "
            f"{split_for_base_index(base_entry.base_index)!r}, not {split!r}"
        )
    if base_entry.family_id != family_id:
        failures.append(
            f"family migration: base {base_entry.base_setup_id} is "
            f"{base_entry.family_id!r} but the output claims {family_id!r}"
        )

    satisfied, violations = family_contract(family_id).evaluate(
        compute_trait_vector(canonical)
    )
    if not satisfied:
        failures.extend(f"family {family_id} clause violated: {name}" for name in violations)

    if not setup_has_initial_mobility(canonical):
        failures.append("stranded: no initial legal move for the owner")

    serialized = serialize_setup(canonical)
    if deserialize_setup(serialized) != canonical:
        failures.append("serialization: round-trip did not reproduce the setup")
    if reflect_canonical(reflect_canonical(canonical)) != canonical:
        failures.append("reflection: the involution did not round-trip")
    if class_fingerprint(canonical) != class_fingerprint(reflect_canonical(canonical)):
        failures.append(
            "reflection: the class fingerprint differs between the two orientations"
        )

    return failures


# ---------------------------------------------------------------------------
# The sampled output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampledSetup:
    """One sampler output: the final setup plus its complete provenance."""

    #: The final canonical own-orientation 40-tuple, after any perturbation
    #: and after the sampled reflection.
    canonical: "tuple[int, ...]"
    #: The base library entry's stored canonical representative.
    base_canonical: "tuple[int, ...]"
    #: The descendant before reflection was applied.
    perturbed_canonical: "tuple[int, ...]"
    split: str
    family_id: str
    base_setup_id: str
    reflection_applied: bool
    perturbation_applied: bool
    perturbation: "PerturbationResult | None"
    provenance: dict = field(default_factory=dict)

    def oriented(self, player: int) -> "tuple[int, ...]":
        """The engine-ready setup tuple for `player`.

        The frozen engine owns the canonical-to-board mapping; this is only a
        convenience so callers never re-derive the orientation convention.
        """
        return orient_setup(self.canonical, player)

    def serialized(self) -> str:
        return serialize_setup(self.canonical)


def _provenance(
    *,
    base_entry,
    split: str,
    canonical: "tuple[int, ...]",
    reflection_applied: bool,
    perturbation_requested: bool,
    perturbation: "PerturbationResult | None",
    perturbation_seed: "int | None",
    profile_name: str,
    draw_seed: "int | None",
) -> dict:
    """The provenance record of one sampled descendant.

    Training/debug metadata only: it names the library, the sampler, the
    perturbation and the base identity, and it deliberately carries no game
    outcome, model score or strength signal of any kind.

    `perturbation_seed` is the composite `seed_encoding_v1` value — the
    complete perturbation identity. `perturbation_swap_count` is **derived**
    from it by decoding, and `perturbation_max_attempts` restates the
    `setup_perturbation_v1` version constant; both are descriptive/integrity
    metadata, and :func:`rebuild_from_provenance` rejects a record whose
    values disagree rather than honouring them.
    """
    applied = bool(perturbation is not None and perturbation.accepted)
    return {
        "setup_library_version": SETUP_LIBRARY_VERSION,
        "contract_version": SETUP_GENERATOR_CONTRACT_VERSION,
        "family_contract_version": SETUP_FAMILY_VERSION,
        "trait_schema_version": SETUP_TRAIT_VECTOR_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "sampler_profile": profile_name,
        "perturbation_version": PERTURBATION_VERSION,
        "split": split,
        "primary_family_id": base_entry.family_id,
        "family_key": base_entry.family_key,
        "base_setup_id": base_entry.base_setup_id,
        "base_index": base_entry.base_index,
        "base_fingerprint": base_entry.fingerprint,
        "reflection_applied": reflection_applied,
        "perturbation_requested": perturbation_requested,
        "perturbation_applied": applied,
        "perturbation_exhausted": bool(perturbation_requested and not applied),
        "perturbation_swap_count": (
            perturbation.swap_count if perturbation_requested else None
        ),
        "perturbation_seed": perturbation_seed if perturbation_requested else None,
        "perturbation_id": perturbation.perturbation_id if perturbation else None,
        "perturbation_max_attempts": (
            perturbation_v1.MAX_PERTURBATION_ATTEMPTS
            if perturbation_requested
            else None
        ),
        "perturbation_attempts": perturbation.attempts if perturbation else 0,
        "perturbation_accepted_attempt_index": (
            perturbation.accepted_attempt_index if perturbation else None
        ),
        "perturbation_hamming_from_base": (
            perturbation.hamming_from_base if perturbation else 0
        ),
        "draw_seed": draw_seed,
        "final_setup": serialize_setup(canonical),
        "final_setup_fingerprint": content_fingerprint(canonical),
        "final_setup_class_fingerprint": class_fingerprint(canonical),
    }


def build_descendant(
    base_entry,
    *,
    reflection_applied: bool,
    perturbation_requested: bool,
    perturbation_seed: "int | None" = None,
    profile_name: str = DEFAULT_PROFILE.name,
    draw_seed: "int | None" = None,
) -> SampledSetup:
    """Build one descendant from an explicit decision set.

    The single construction path of this module: :func:`sample_setup` draws the
    decisions from seeded streams and calls it, and
    :func:`rebuild_from_provenance` reads the decisions from provenance and
    calls it. Both therefore run identical code, which is what makes rebuild
    exactness a property of the design rather than of a duplicated routine.

    `perturbation_seed` is the complete perturbation identity: the composite
    `seed_encoding_v1` value carrying both the swap count and the raw
    randomness. `profile_name` and `draw_seed` are recorded as sampler
    metadata and cannot change the descendant — the same
    `(base_setup_id, sampler_version, perturbation_seed)` yields the same
    canonical descendant from any caller context.

    Perturbation is applied to the base's canonical representative and
    reflection is applied last. Family membership, mobility and legality are
    all reflection-invariant under the frozen contracts, so the order cannot
    change a verdict; it merely keeps the perturbation identity independent of
    the orientation bit.

    Raises :class:`SetupLibraryError` if the finished output fails any
    final-output check. An invalid setup is never returned.
    """
    base = base_entry.canonical_setup
    perturbation: "PerturbationResult | None" = None

    if perturbation_requested:
        if perturbation_seed is None:
            raise SetupLibraryError(
                "a requested perturbation needs a perturbation_seed"
            )
        perturbation = perturb_setup(base, base_entry.family_id, perturbation_seed)
        descendant = perturbation.canonical
    else:
        descendant = base

    canonical = reflect_canonical(descendant) if reflection_applied else descendant

    failures = validate_sampled_setup(
        canonical, base_entry, base_entry.split, base_entry.family_id
    )
    if failures:
        raise SetupLibraryError(
            f"{base_entry.base_setup_id}: sampler output failed final validation: "
            + "; ".join(failures)
        )

    provenance = _provenance(
        base_entry=base_entry,
        split=base_entry.split,
        canonical=canonical,
        reflection_applied=reflection_applied,
        perturbation_requested=perturbation_requested,
        perturbation=perturbation,
        perturbation_seed=perturbation_seed,
        profile_name=profile_name,
        draw_seed=draw_seed,
    )

    return SampledSetup(
        canonical=canonical,
        base_canonical=base,
        perturbed_canonical=descendant,
        split=base_entry.split,
        family_id=base_entry.family_id,
        base_setup_id=base_entry.base_setup_id,
        reflection_applied=reflection_applied,
        perturbation_applied=bool(perturbation is not None and perturbation.accepted),
        perturbation=perturbation,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# The public sampling API
# ---------------------------------------------------------------------------


def _stream(field_name: str, profile_name: str, split: str, seed: int) -> random.Random:
    """A private RNG stream for one sampler decision.

    Domain-separated per decision, so family, base, orientation, perturbation
    and intensity draws are independent and none of them can be perturbed by
    adding a decision elsewhere. No global RNG state is ever consumed.
    """
    return random.Random(
        derive_stream_seed(
            f"{SAMPLER_VERSION}:{field_name}", profile_name, split, int(seed)
        )
    )


def _uniform_index(rng: random.Random, count: int) -> int:
    return min(int(rng.random() * count), count - 1)


def sample_setup(
    split: str,
    seed: int,
    profile: "SamplerProfile | str" = DEFAULT_PROFILE,
    index: "SetupLibraryIndex | None" = None,
) -> SampledSetup:
    """Sample one setup and its provenance deterministically.

    The Phase 8 entry point. `split` selects the eligible base population and
    is never re-chosen; `seed` is the caller's draw identity, so the same
    `(split, seed, profile)` always yields the same setup, and different seeds
    explore the procedural support.

    Family selection is uniform over the 16 families, base selection is
    uniform over that family's bases *within the requested split*, and
    orientation is a fair seeded coin. The returned setup has already passed
    the complete final-output validation stack.
    """
    if split not in SPLITS:
        raise SetupLibraryError(f"unknown split: {split!r}")
    if isinstance(profile, str):
        profile = sampler_profile(profile)
    library = load_library_index() if index is None else index

    family_id = FAMILY_IDS[
        _uniform_index(_stream("family", profile.name, split, seed), len(FAMILY_IDS))
    ]
    eligible = library.eligible_bases(family_id, split)
    base_entry = eligible[
        _uniform_index(_stream("base", profile.name, split, seed), len(eligible))
    ]

    reflection_applied = (
        _stream("orientation", profile.name, split, seed).random()
        < profile.reflection_probability
    )
    perturbation_requested = (
        _stream("perturbation", profile.name, split, seed).random()
        < profile.perturbation_probability
    )

    perturbation_seed: "int | None" = None
    if perturbation_requested:
        # The profile chooses the intensity and the raw randomness, then both
        # are folded into ONE effective perturbation seed. From here on the
        # perturbation result is profile-independent: the descendant is a pure
        # function of (base_setup_id, sampler_version, perturbation_seed).
        intensity_rng = _stream("intensity", profile.name, split, seed)
        swap_count = _weighted_swap_count(intensity_rng, profile)
        raw_seed = derive_stream_seed(
            f"{SAMPLER_VERSION}:perturbation_seed",
            profile.name,
            split,
            int(seed),
            base_entry.base_setup_id,
            swap_count,
        )
        perturbation_seed = encode_perturbation_seed(swap_count, raw_seed)

    return build_descendant(
        base_entry,
        reflection_applied=reflection_applied,
        perturbation_requested=perturbation_requested,
        perturbation_seed=perturbation_seed,
        profile_name=profile.name,
        draw_seed=int(seed),
    )


def _weighted_swap_count(rng: random.Random, profile: SamplerProfile) -> int:
    counts = profile.swap_counts
    weights = profile.intensity_weights
    target = rng.random() * sum(weights)
    accumulated = 0.0
    for count, weight in zip(counts, weights):
        accumulated += weight
        if accumulated >= target:
            return count
    return counts[-1]  # pragma: no cover - float tail guard


def rebuild_from_provenance(
    provenance: dict,
    index: "SetupLibraryIndex | None" = None,
) -> SampledSetup:
    """Reconstruct a sampled descendant from its provenance alone.

    The lower-level deterministic rebuild API required by the sampling
    contract. It replays the recorded decisions through the same
    :func:`build_descendant` path — including a requested-but-exhausted
    perturbation, which is why `perturbation_requested` is recorded alongside
    `perturbation_applied` — and then verifies that the rebuilt setup carries
    the recorded fingerprint. A mismatch raises rather than returning a
    plausible-looking setup.

    The perturbation identity is `perturbation_seed` alone. The derived
    metadata fields are treated as integrity checks, not inputs: a record
    whose `perturbation_swap_count` disagrees with the seed's encoding, or
    whose `perturbation_max_attempts` disagrees with the
    `setup_perturbation_v1` version constant, is rejected rather than
    honoured. The recorded profile name is sampler metadata and cannot alter
    reconstruction.
    """
    missing = [key for key in REQUIRED_PROVENANCE_FIELDS if key not in provenance]
    if missing:
        raise SetupLibraryError(f"provenance missing required fields: {missing}")

    library = load_library_index() if index is None else index
    base_entry = library.base(str(provenance["base_setup_id"]))

    if base_entry.split != provenance["split"]:
        raise SetupLibraryError(
            f"provenance split {provenance['split']!r} contradicts base "
            f"{base_entry.base_setup_id} ({base_entry.split!r})"
        )
    if base_entry.family_id != provenance["primary_family_id"]:
        raise SetupLibraryError(
            f"provenance family {provenance['primary_family_id']!r} contradicts "
            f"base {base_entry.base_setup_id} ({base_entry.family_id!r})"
        )

    requested = bool(provenance.get("perturbation_requested", provenance["perturbation_applied"]))
    profile_name = str(provenance.get("sampler_profile", DEFAULT_PROFILE.name))

    perturbation_seed = None
    if requested:
        if provenance.get("perturbation_seed") is None:
            raise SetupLibraryError(
                "provenance requests a perturbation but carries no perturbation_seed"
            )
        perturbation_seed = int(provenance["perturbation_seed"])
        expected_swap_count, _raw_seed = decode_perturbation_seed(perturbation_seed)
        recorded_swap_count = provenance.get("perturbation_swap_count")
        if recorded_swap_count is not None and int(recorded_swap_count) != expected_swap_count:
            raise SetupLibraryError(
                f"tampered provenance: perturbation_swap_count "
                f"{recorded_swap_count} contradicts the perturbation_seed "
                f"encoding (decodes to {expected_swap_count}); swap_count is "
                "derived metadata, not an input"
            )
        recorded_max_attempts = provenance.get("perturbation_max_attempts")
        if (
            recorded_max_attempts is not None
            and int(recorded_max_attempts) != perturbation_v1.MAX_PERTURBATION_ATTEMPTS
        ):
            raise SetupLibraryError(
                f"tampered provenance: perturbation_max_attempts "
                f"{recorded_max_attempts} is not the setup_perturbation_v1 "
                f"version constant {perturbation_v1.MAX_PERTURBATION_ATTEMPTS}; "
                "the budget is not an input"
            )

    rebuilt = build_descendant(
        base_entry,
        reflection_applied=bool(provenance["reflection_applied"]),
        perturbation_requested=requested,
        perturbation_seed=perturbation_seed,
        profile_name=profile_name,
        draw_seed=provenance.get("draw_seed"),
    )

    if rebuilt.provenance["final_setup_fingerprint"] != provenance["final_setup_fingerprint"]:
        raise SetupLibraryError(
            f"{base_entry.base_setup_id}: rebuilt descendant fingerprint "
            f"{rebuilt.provenance['final_setup_fingerprint']} does not match "
            f"the recorded {provenance['final_setup_fingerprint']}"
        )
    return rebuilt


def provenance_is_observer_safe(provenance: dict) -> "list[str]":
    """Field names in `provenance` that would betray an outcome or strength signal.

    Phase 7 provenance is training/debug metadata. It legitimately names
    hidden setup truth, so Agent 5 must keep it away from move-model inputs;
    what it must never contain is a game outcome, win rate, Elo, value or
    policy signal. This reuses the library's frozen forbidden-token list so
    there is one such list in the package, not two.
    """
    return [
        name
        for name in sorted(provenance)
        if any(token in name.lower() for token in FORBIDDEN_ENTRY_FIELD_TOKENS)
    ]


def provenance_round_trips(provenance: dict) -> bool:
    """Whether the provenance survives a canonical JSON round-trip unchanged."""
    return json.loads(json.dumps(provenance, sort_keys=True)) == provenance


# ---------------------------------------------------------------------------
# Acceptance instrument: the deterministic balanced stress corpus
# ---------------------------------------------------------------------------

#: Version identifier of the stress corpus enumeration. This is Agent 4's
#: acceptance instrument, deliberately balanced across every branch; it is
#: **not** the Phase 8 default sampling profile, which Agent 6 freezes.
STRESS_CORPUS_VERSION = "setup_stress_corpus_v1"

#: Outputs per family per split: the library's own 400/50/50 ratio scaled by
#: 12.5, giving 6,250 per family and 100,000 in total.
STRESS_SPLIT_OUTPUTS = {"train": 5000, "validation": 625, "test": 625}
STRESS_OUTPUTS_PER_FAMILY = sum(STRESS_SPLIT_OUTPUTS.values())
STRESS_TOTAL_OUTPUTS = STRESS_OUTPUTS_PER_FAMILY * len(FAMILY_IDS)
assert STRESS_OUTPUTS_PER_FAMILY == 6250
assert STRESS_TOTAL_OUTPUTS == 100000


@dataclass(frozen=True)
class StressDraw:
    """One planned stress-corpus output, decided before anything is built."""

    family_id: str
    split: str
    position: int
    base_index: int
    base_setup_id: str
    reflection_applied: bool
    perturbation_requested: bool
    #: Derived plan metadata; always equal to the seed's decoded swap count.
    swap_count: int
    #: The composite `seed_encoding_v1` perturbation identity.
    perturbation_seed: int


def stress_corpus_plan(
    split_outputs: "dict[str, int]" = None,
    family_ids: "tuple[str, ...]" = FAMILY_IDS,
):
    """Enumerate the deterministic balanced acceptance corpus.

    Every decision is a pure function of `(family_id, split, position)`, so the
    corpus is reproducible without running it and an auditor can regenerate any
    single output in isolation:

    ```text
    lap                     position // len(eligible)
    base index              eligible[position % len(eligible)]      round robin
    ordinal                 position + lap                          de-aliased
    reflection              ordinal % 2 == 1                        50/50
    perturbation requested  (ordinal // 2) % 2 == 1                 50/50
    swap count              1 + (ordinal // 4) % 6                  whole window
    raw seed                derive_stream_seed(corpus, family, split,
                                               position, base id, swap count)
    perturbation seed       encode_perturbation_seed(swap count, raw seed)
    ```

    The reflection and perturbation bits use different bit positions of the
    same counter, so all four branch combinations — reflection-only,
    perturbation-only, both, neither — receive close to a quarter of the
    corpus each.

    The `lap` offset is what keeps the instrument honest. The base round robin
    has period `len(eligible)`, which is 400 for train and 50 for validation
    and test — every one of them a multiple of 4, the period of the branch
    counter. Keying the branch bits on `position` alone would therefore alias
    them onto the base index: every output of a given base would carry the
    *same* orientation and the *same* branch, so no base would ever appear in
    both orientations and mirror-image duplicates between two descendants of
    one base could not arise at all. Adding `lap` makes the stride odd
    (`len(eligible) + 1`), which is coprime with 4, so each base cycles through
    all four branch combinations across its own laps while the corpus-wide
    balance is preserved.

    Split counts of 625 are odd, so those segments cannot split exactly
    evenly; the run reports the achieved counts rather than assuming the
    nominal ones.
    """
    from .contracts import base_setup_id as make_base_setup_id

    outputs = STRESS_SPLIT_OUTPUTS if split_outputs is None else split_outputs
    for family_id in family_ids:
        for split in SPLITS:
            count = outputs.get(split, 0)
            start, stop = SPLIT_BASE_RANGES[split]
            eligible = tuple(range(start, stop))
            for position in range(count):
                base_index = eligible[position % len(eligible)]
                ordinal = position + position // len(eligible)
                identifier = make_base_setup_id(family_id, base_index)
                swap_count = MIN_SWAP_COUNT + (ordinal // 4) % (
                    MAX_SWAP_COUNT - MIN_SWAP_COUNT + 1
                )
                yield StressDraw(
                    family_id=family_id,
                    split=split,
                    position=position,
                    base_index=base_index,
                    base_setup_id=identifier,
                    reflection_applied=ordinal % 2 == 1,
                    perturbation_requested=(ordinal // 2) % 2 == 1,
                    swap_count=swap_count,
                    perturbation_seed=encode_perturbation_seed(
                        swap_count,
                        derive_stream_seed(
                            STRESS_CORPUS_VERSION,
                            family_id,
                            split,
                            position,
                            identifier,
                            swap_count,
                        ),
                    ),
                )


def build_stress_output(
    draw: StressDraw,
    index: "SetupLibraryIndex | None" = None,
) -> SampledSetup:
    """Build one planned stress output through the ordinary sampler path."""
    library = load_library_index() if index is None else index
    base_entry = library.base(draw.base_setup_id)
    return build_descendant(
        base_entry,
        reflection_applied=draw.reflection_applied,
        perturbation_requested=draw.perturbation_requested,
        perturbation_seed=draw.perturbation_seed,
        profile_name=STRESS_CORPUS_VERSION,
        draw_seed=draw.position,
    )


# ---------------------------------------------------------------------------
# Contract document
# ---------------------------------------------------------------------------


def sampler_contract_document() -> dict:
    """The machine-readable `setup_sampler_v1` contract, for the artifact."""
    return {
        "sampler_version": SAMPLER_VERSION,
        "library_version": SETUP_LIBRARY_VERSION,
        "contract_version": SETUP_GENERATOR_CONTRACT_VERSION,
        "family_contract_version": SETUP_FAMILY_VERSION,
        "trait_schema_version": SETUP_TRAIT_VECTOR_VERSION,
        "public_api": {
            "sample_setup": "sample_setup(split, seed, profile=DEFAULT_PROFILE) -> SampledSetup",
            "rebuild": "rebuild_from_provenance(provenance) -> SampledSetup",
            "build_descendant": (
                "build_descendant(base_entry, reflection_applied, "
                "perturbation_requested, perturbation_seed, ...) "
                "-> SampledSetup"
            ),
            "engine_handoff": "SampledSetup.oriented(player) -> engine setup tuple",
        },
        "sampling_order": [
            "split (caller-supplied, never re-chosen)",
            "family (uniform over 16)",
            "base (uniform over the family's bases inside the split)",
            "perturbation requested (profile coin)",
            "perturbation intensity (profile weights over swap counts 1..6)",
            "intensity + raw randomness encoded into one perturbation seed",
            "constrained perturbation with deterministic rejection/retry",
            "reflection (profile coin, applied last)",
            "final-output validation from scratch",
        ],
        "stream_derivation": (
            "random.Random(derive_stream_seed('setup_sampler_v1:<field>', "
            "profile, split, seed)); one domain-separated stream per decision; "
            "no mutable global RNG state is consumed"
        ),
        "perturbation_identity": (
            "the descendant is a pure function of (base_setup_id, "
            "sampler_version, perturbation_seed); the seed is the "
            "seed_encoding_v1 composite carrying swap count and raw "
            "randomness, the retry budget is the setup_perturbation_v1 "
            "version constant, and no profile or caller input can change the "
            "result once the seed is emitted"
        ),
        "inheritance": (
            "every descendant inherits base_setup_id, primary_family_id and "
            "split verbatim; a train-derived setup can never become a "
            "validation or test setup"
        ),
        "split_base_ranges": {
            split: list(bounds) for split, bounds in SPLIT_BASE_RANGES.items()
        },
        "provenance_schema": {
            "required_fields": list(REQUIRED_PROVENANCE_FIELDS),
            "full_fields": list(PROVENANCE_FIELDS),
            "derived_fields": {
                "perturbation_swap_count": (
                    "decoded from perturbation_seed; rebuild rejects a "
                    "record whose value disagrees with the encoding"
                ),
                "perturbation_max_attempts": (
                    "the setup_perturbation_v1 version constant; rebuild "
                    "rejects a record whose value disagrees with it"
                ),
                "sampler_profile": (
                    "sampler decision metadata only; cannot alter "
                    "reconstruction once the perturbation seed is fixed"
                ),
            },
            "prohibited": (
                "no game outcome, win rate, Elo, value, policy or strength "
                "field may appear; provenance is training/debug metadata and "
                "must not cross the observer-safe model boundary"
            ),
        },
        "final_output_validation": [
            "exact inventory (engine validate_setup)",
            "engine setup validation",
            "initial-mobility quality check",
            "base split unchanged",
            "base primary family unchanged",
            "family required predicates",
            "serialization/fingerprint round-trip",
        ],
        "perturbation": operator_mix_document(),
        "profiles": {name: profile.to_dict() for name, profile in PROFILES.items()},
        "default_profile": DEFAULT_PROFILE.name,
        "profile_freeze_owner": "Agent 6 makes the final sampler-profile freeze",
        "stress_corpus": {
            "corpus_version": STRESS_CORPUS_VERSION,
            "outputs_per_family": STRESS_OUTPUTS_PER_FAMILY,
            "split_outputs_per_family": dict(STRESS_SPLIT_OUTPUTS),
            "total_outputs": STRESS_TOTAL_OUTPUTS,
            "note": (
                "an acceptance instrument that deliberately balances every "
                "branch; not the Phase 8 default profile"
            ),
        },
    }


__all__ = [
    "SAMPLER_VERSION",
    "SPLIT_BASE_RANGES",
    "REQUIRED_PROVENANCE_FIELDS",
    "PROVENANCE_FIELDS",
    "SamplerProfile",
    "PROFILES",
    "DEFAULT_PROFILE",
    "NEUTRAL_PROFILE",
    "REFLECTION_ONLY_PROFILE",
    "PERTURBATION_ONLY_PROFILE",
    "sampler_profile",
    "SetupLibraryIndex",
    "load_library_index",
    "SampledSetup",
    "build_descendant",
    "sample_setup",
    "rebuild_from_provenance",
    "validate_sampled_setup",
    "provenance_is_observer_safe",
    "provenance_round_trips",
    "STRESS_CORPUS_VERSION",
    "STRESS_SPLIT_OUTPUTS",
    "STRESS_OUTPUTS_PER_FAMILY",
    "STRESS_TOTAL_OUTPUTS",
    "StressDraw",
    "stress_corpus_plan",
    "build_stress_output",
    "sampler_contract_document",
]
