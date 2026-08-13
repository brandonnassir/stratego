"""Constrained family-preserving perturbation: `setup_perturbation_v1`.

Specification sources:

- `04_AGENT_4_REFLECTION_AND_PERTURBATION.md` (constrained perturbation,
  perturbation identity, final-output validation)
- Agent 1's frozen invariants in `contracts.py`
  (:data:`PERTURBATION_INVARIANTS`, :func:`validate_perturbation`)

What a perturbation is
----------------------
A perturbation is a sequence of `k` **disjoint piece swaps** applied to a base
setup's canonical 40-tuple. Each swap exchanges two cells holding *different*
piece types, so the exact engine inventory is preserved by construction rather
than by repair, and each swap contributes exactly 2 to the canonical Hamming
distance. Swaps never reuse a cell, so a `k`-swap descendant sits at Hamming
distance exactly `2k` from its base and `k` in `1..6` covers Agent 1's frozen
window `[PERTURBATION_MIN_HAMMING, PERTURBATION_MAX_HAMMING] = [2, 12]`
exactly. The Flag cell is excluded from every operator, so the Flag can never
move.

Operators are construction rules, never acceptance rules
--------------------------------------------------------
The seven operators below are *proposal* distributions, in the same spirit as
Agent 2's sixteen family plans: they bias proposals toward transformations
that tend to preserve a family, but they decide nothing. Acceptance is always
Agent 1's frozen :func:`validate_perturbation` applied to the finished
candidate — engine inventory/legality, the Flag cell, the `[2, 12]` Hamming
window, every required and forbidden family clause, and initial mobility.
An operator that drifts from a family contract therefore cannot smuggle a
non-conforming descendant out of this module; it can only waste attempts.

Nothing here re-derives a family predicate, a distance bound or a mobility
rule. `validate_perturbation` is imported and called; it is not reimplemented.

Identity
--------
Agent 1's frozen invariant makes a descendant a pure function of
`(base_setup_id, sampler_version, perturbation_seed)`. This module therefore
gives `perturbation_seed` no companions: the seed is a **versioned composite**
that itself encodes every result-affecting perturbation decision.

```text
perturbation_seed = (raw_seed << 3) | (swap_count - 1)      seed_encoding_v1
```

The low three bits carry the swap count (values 0..5 encode counts 1..6;
6 and 7 are invalid encodings and are rejected), and the remaining bits carry
the raw randomness. :func:`encode_perturbation_seed` and
:func:`decode_perturbation_seed` are the exact, bijective, tested mapping.
`swap_count` is thus *derived metadata*, never an independent semantic input;
the operator mix and the retry budget are frozen constants of
`setup_perturbation_v1`; and no caller — sampler profile included — can
change the descendant once the seed is fixed.

Determinism
-----------
:func:`perturb_setup` is a pure function of
`(base_canonical, family_id, perturbation_seed)` under the frozen operator
mix and budget. Candidate attempts `0, 1, 2, ...` are drawn from
domain-separated `derive_stream_seed` streams keyed by the decoded
`(raw_seed, swap_count)`, and the first candidate accepted by the frozen
validator wins, so rejection and retry are themselves reproducible. No
mutable global RNG state is consumed anywhere: every random decision comes
from a `random.Random` seeded by a hash of the perturbation identity.

Exhaustion is not a fallback to an invalid setup
------------------------------------------------
If the attempt budget is exhausted the result reports `accepted=False` and
carries the **unmodified base setup**, which is legal, family-correct, mobile
and split-correct by Agent 3's audit. The caller records the descendant as
unperturbed. No invalid arrangement is ever returned, and the exhaustion is
counted rather than hidden.
"""

import random
from dataclasses import dataclass, field

from ..engine.constants import BOMB, FLAG, MINER, SCOUT
from .contracts import (
    PERTURBATION_MAX_HAMMING,
    PERTURBATION_MIN_HAMMING,
    validate_perturbation,
)
from .families import FAMILY_BY_ID
from .identity import (
    CANONICAL_CELLS,
    CANONICAL_FILES,
    SetupLibraryError,
    canonical_rank_file,
    derive_stream_seed,
)
from .traits import DECOY_MIN_FLAG_DISTANCE, HIGH_RANK_TYPES

#: Version identifier of this perturbation contract. A semantic change to the
#: operator set, the swap-count semantics, the attempt ordering or the
#: acceptance stack is a new identifier, never a silent reinterpretation.
PERTURBATION_VERSION = "setup_perturbation_v1"

#: Each swap moves exactly two cells, so the swap count and the canonical
#: Hamming distance from the base are locked to each other.
HAMMING_PER_SWAP = 2

MIN_SWAP_COUNT = PERTURBATION_MIN_HAMMING // HAMMING_PER_SWAP
MAX_SWAP_COUNT = PERTURBATION_MAX_HAMMING // HAMMING_PER_SWAP
assert (MIN_SWAP_COUNT, MAX_SWAP_COUNT) == (1, 6)

#: Candidate attempts allowed for one perturbation identity before the
#: perturbation is abandoned and the base is returned unperturbed. A **version
#: constant** of `setup_perturbation_v1`, not a configurable input: the
#: accepted candidate is the first one the budget reaches, so a different
#: budget is a different perturbation semantics and therefore a different
#: version. Tests may force exhaustion through the private diagnostic entry
#: point; production identity is defined solely by :func:`perturb_setup`.
MAX_PERTURBATION_ATTEMPTS = 64

# ---------------------------------------------------------------------------
# The versioned composite perturbation-seed encoding: `seed_encoding_v1`
# ---------------------------------------------------------------------------

#: Name of the composite-seed encoding, frozen with the perturbation version.
PERTURBATION_SEED_ENCODING = "seed_encoding_v1"

_SEED_SWAP_BITS = 3
_SEED_SWAP_MASK = (1 << _SEED_SWAP_BITS) - 1


def encode_perturbation_seed(swap_count: int, raw_seed: int) -> int:
    """Pack `(swap_count, raw_seed)` into one composite perturbation seed.

    `seed_encoding_v1`: the low three bits hold `swap_count - 1` (0..5), the
    remaining bits hold the raw seed. The mapping is a bijection between
    valid `(swap_count, raw_seed)` pairs and valid composite seeds, so the
    swap count has no semantic freedom independent of the seed.
    """
    if isinstance(swap_count, bool) or not isinstance(swap_count, int):
        raise SetupLibraryError(f"swap_count must be an int, got {swap_count!r}")
    if not MIN_SWAP_COUNT <= swap_count <= MAX_SWAP_COUNT:
        raise SetupLibraryError(
            f"swap_count must be in {MIN_SWAP_COUNT}..{MAX_SWAP_COUNT}, got {swap_count}"
        )
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
        raise SetupLibraryError(f"raw_seed must be an int, got {raw_seed!r}")
    if raw_seed < 0:
        raise SetupLibraryError(f"raw_seed must be non-negative, got {raw_seed}")
    return (raw_seed << _SEED_SWAP_BITS) | (swap_count - MIN_SWAP_COUNT)


def decode_perturbation_seed(perturbation_seed: int) -> "tuple[int, int]":
    """Inverse of :func:`encode_perturbation_seed`: `(swap_count, raw_seed)`.

    Rejects any integer whose low bits do not name a swap count in the frozen
    window, so an arbitrary or truncated value cannot silently decode to a
    plausible identity.
    """
    if isinstance(perturbation_seed, bool) or not isinstance(perturbation_seed, int):
        raise SetupLibraryError(
            f"perturbation_seed must be an int, got {perturbation_seed!r}"
        )
    if perturbation_seed < 0:
        raise SetupLibraryError(
            f"perturbation_seed must be non-negative, got {perturbation_seed}"
        )
    swap_count = MIN_SWAP_COUNT + (perturbation_seed & _SEED_SWAP_MASK)
    if swap_count > MAX_SWAP_COUNT:
        raise SetupLibraryError(
            f"invalid {PERTURBATION_SEED_ENCODING} perturbation seed: low bits "
            f"{perturbation_seed & _SEED_SWAP_MASK} do not encode a swap count "
            f"in {MIN_SWAP_COUNT}..{MAX_SWAP_COUNT}"
        )
    return swap_count, perturbation_seed >> _SEED_SWAP_BITS

# Rejection reasons. Every rejected candidate is counted under exactly one.
REJECTION_CONSTRUCTION = "construction_infeasible"
REJECTION_ENGINE_INVALID = "engine_invalid"
REJECTION_HAMMING_WINDOW = "hamming_window"
REJECTION_FLAG_MOVED = "flag_moved"
REJECTION_FAMILY_PREDICATE = "family_predicate"
REJECTION_STRANDED = "stranded"
REJECTION_REASONS = (
    REJECTION_CONSTRUCTION,
    REJECTION_ENGINE_INVALID,
    REJECTION_HAMMING_WINDOW,
    REJECTION_FLAG_MOVED,
    REJECTION_FAMILY_PREDICATE,
    REJECTION_STRANDED,
)

#: Domain tag of the per-attempt stream. Distinct from every other Phase 7
#: stream purpose, so perturbation draws can never alias generation draws.
_ATTEMPT_STREAM = "setup_perturbation_v1:attempt"


class _ProposalFailure(Exception):
    """An operator found no eligible cell pair for this attempt.

    Internal control flow only: the attempt is counted as a
    `construction_infeasible` rejection and the next attempt stream is drawn.
    Never propagates out of :func:`perturb_setup`.
    """


# ---------------------------------------------------------------------------
# Deterministic drawing helpers
# ---------------------------------------------------------------------------


def _weighted_choice(rng: random.Random, items: "tuple", weights: "tuple[float, ...]"):
    """One item drawn from an ascending-ordered sequence by positive weight.

    Mirrors the generator's draw helper exactly, so the whole package shares
    one deterministic selection convention: items are always supplied in a
    fixed order, never from set or dict iteration.
    """
    total = sum(weights)
    if not items or total <= 0.0:  # pragma: no cover - defensive
        raise _ProposalFailure("no eligible option remains")
    target = rng.random() * total
    accumulated = 0.0
    for item, weight in zip(items, weights):
        accumulated += weight
        if accumulated >= target:
            return item
    return items[-1]  # pragma: no cover - float tail guard


def _uniform_pair(rng: random.Random, pairs: "list[tuple[int, int]]") -> "tuple[int, int]":
    """One cell pair drawn uniformly from an ascending pair list."""
    if not pairs:
        raise _ProposalFailure("no eligible cell pair remains")
    index = int(rng.random() * len(pairs))
    return pairs[min(index, len(pairs) - 1)]


# ---------------------------------------------------------------------------
# Geometry helpers (canonical frame)
# ---------------------------------------------------------------------------


def _rank_of(cell: int) -> int:
    return cell // CANONICAL_FILES


def _manhattan(cell_a: int, cell_b: int) -> int:
    rank_a, file_a = canonical_rank_file(cell_a)
    rank_b, file_b = canonical_rank_file(cell_b)
    return abs(rank_a - rank_b) + abs(file_a - file_b)


def _chebyshev(cell_a: int, cell_b: int) -> int:
    rank_a, file_a = canonical_rank_file(cell_a)
    rank_b, file_b = canonical_rank_file(cell_b)
    return max(abs(rank_a - rank_b), abs(file_a - file_b))


#: Every unordered cell pair, ascending. Enumerated once so eligibility
#: filtering is a pure list comprehension over a fixed order.
_ALL_PAIRS: "tuple[tuple[int, int], ...]" = tuple(
    (left, right)
    for left in range(CANONICAL_CELLS)
    for right in range(left + 1, CANONICAL_CELLS)
)

_SAME_RANK_PAIRS = tuple(
    pair for pair in _ALL_PAIRS if _rank_of(pair[0]) == _rank_of(pair[1])
)
_CROSS_RANK_PAIRS = tuple(
    pair for pair in _ALL_PAIRS if _rank_of(pair[0]) != _rank_of(pair[1])
)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwapOperator:
    """One family-preserving *proposal* rule over unordered cell pairs.

    `eligible` returns the ascending list of cell pairs this operator may
    swap, given the arrangement, the Flag cell and the cells already consumed
    by earlier swaps in the same candidate. The two piece types are always
    different and neither cell is the Flag, which is what makes a swap worth
    exactly `HAMMING_PER_SWAP` and keeps the Flag pinned.
    """

    name: str
    description: str
    technique: str
    #: Relative proposal weight in the frozen default mix.
    weight: float

    def eligible(
        self,
        canonical: "tuple[int, ...]",
        flag_cell: int,
        used: "frozenset[int]",
    ) -> "list[tuple[int, int]]":  # pragma: no cover - overridden per operator
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "technique": self.technique,
            "default_weight": self.weight,
        }


def _swappable(
    canonical: "tuple[int, ...]",
    flag_cell: int,
    used: "frozenset[int]",
    pair: "tuple[int, int]",
) -> bool:
    """The universal swap precondition shared by every operator."""
    left, right = pair
    if left == flag_cell or right == flag_cell:
        return False
    if left in used or right in used:
        return False
    return canonical[left] != canonical[right]


@dataclass(frozen=True)
class _PositionalSwap(SwapOperator):
    """Bounded swap restricted to a fixed positional pair set."""

    pairs: "tuple[tuple[int, int], ...]" = ()

    def eligible(self, canonical, flag_cell, used):
        return [
            pair
            for pair in self.pairs
            if _swappable(canonical, flag_cell, used, pair)
        ]


@dataclass(frozen=True)
class _RoleSwap(SwapOperator):
    """Role-compatible swap: exactly one cell holds a piece of `role_types`.

    Relocating one role member into a cell held by a non-member is the
    "controlled Scout/Miner relocation" and "role-compatible piece swap"
    technique: the role's rank profile changes by exactly one piece, which is
    the smallest move that can shift a family's rank-count clause.
    """

    role_types: "tuple[int, ...]" = ()
    #: Optional extra geometric restriction on the pair, evaluated against the
    #: Flag cell (used by the fortress and decoy variants).
    locality: "str | None" = None

    def _locality_ok(self, pair: "tuple[int, int]", flag_cell: int) -> bool:
        if self.locality is None:
            return True
        left, right = pair
        if self.locality == "flag_zone":
            return (
                _chebyshev(left, flag_cell) <= 2 or _chebyshev(right, flag_cell) <= 2
            )
        if self.locality == "away_from_flag":
            return (
                _manhattan(left, flag_cell) >= DECOY_MIN_FLAG_DISTANCE
                and _manhattan(right, flag_cell) >= DECOY_MIN_FLAG_DISTANCE
            )
        raise SetupLibraryError(  # pragma: no cover - defensive
            f"unknown locality rule: {self.locality!r}"
        )

    def eligible(self, canonical, flag_cell, used):
        return [
            pair
            for pair in _ALL_PAIRS
            if _swappable(canonical, flag_cell, used, pair)
            and (
                (canonical[pair[0]] in self.role_types)
                != (canonical[pair[1]] in self.role_types)
            )
            and self._locality_ok(pair, flag_cell)
        ]


#: The frozen operator set. Every entry realizes one technique named by the
#: Agent 4 assignment; together they cover role-compatible piece swaps,
#: bounded within-rank and cross-rank swaps, local fortress variation,
#: controlled Scout/Miner relocation and controlled decoy variation.
#:
#: The default weights are **structural coverage** choices, never strength
#: choices. Within-rank swaps preserve every per-rank trait a family clause
#: can reference (rank histograms, front/back counts, Marshal/General/Spy
#: rank, front-rank mobility), so they are the highest-acceptance proposal and
#: carry the largest weight; cross-rank swaps are the only proposals that move
#: mass between ranks, so they carry the second largest weight because they are
#: the ones that expand support in the rank dimension. The five targeted
#: operators exist so Bomb structure, Scout placement and Miner placement —
#: the traits Agent 1's family contracts most often pin — are varied directly
#: rather than only as a by-product of positional draws.
OPERATORS: "tuple[SwapOperator, ...]" = (
    _PositionalSwap(
        name="within_rank_swap",
        description=(
            "exchange two different piece types inside one canonical rank; "
            "every per-rank trait is preserved exactly, so only adjacency, "
            "file-spread and Flag-zone traits can move"
        ),
        technique="bounded within-rank swaps",
        weight=0.30,
        pairs=_SAME_RANK_PAIRS,
    ),
    _PositionalSwap(
        name="cross_rank_swap",
        description=(
            "exchange two different piece types across canonical ranks; the "
            "only proposal that moves a piece between ranks and therefore the "
            "one that expands rank-dimension support"
        ),
        technique="bounded cross-rank swaps",
        weight=0.25,
        pairs=_CROSS_RANK_PAIRS,
    ),
    _RoleSwap(
        name="fortress_variation",
        description=(
            "relocate a Bomb into or out of the Flag's Chebyshev-2 zone, "
            "varying the fortress wall without moving the Flag"
        ),
        technique="local fortress variation",
        weight=0.12,
        role_types=(BOMB,),
        locality="flag_zone",
    ),
    _RoleSwap(
        name="decoy_variation",
        description=(
            "relocate a Bomb between cells at Manhattan distance >= "
            f"{DECOY_MIN_FLAG_DISTANCE} from the Flag, varying a false "
            "fortress or lane block far from the real Flag"
        ),
        technique="controlled decoy variation",
        weight=0.10,
        role_types=(BOMB,),
        locality="away_from_flag",
    ),
    _RoleSwap(
        name="scout_relocation",
        description="exchange a Scout with a non-Scout piece",
        technique="controlled Scout relocation",
        weight=0.10,
        role_types=(SCOUT,),
    ),
    _RoleSwap(
        name="miner_relocation",
        description="exchange a Miner with a non-Miner piece",
        technique="controlled Miner relocation",
        weight=0.08,
        role_types=(MINER,),
    ),
    _RoleSwap(
        name="high_rank_relocation",
        description=(
            "exchange a combat-rank >= 7 piece with a piece below that "
            "threshold, varying where the heavies stand"
        ),
        technique="role-compatible piece swaps",
        weight=0.05,
        role_types=HIGH_RANK_TYPES,
    ),
)

OPERATOR_NAMES = tuple(operator.name for operator in OPERATORS)
OPERATOR_BY_NAME = {operator.name: operator for operator in OPERATORS}

#: The frozen default proposal mix, in `OPERATORS` order.
DEFAULT_OPERATOR_WEIGHTS = tuple(operator.weight for operator in OPERATORS)
assert abs(sum(DEFAULT_OPERATOR_WEIGHTS) - 1.0) < 1e-9


def operator_mix_document() -> dict:
    """The machine-readable operator mix, for the sampler contract artifact."""
    return {
        "perturbation_version": PERTURBATION_VERSION,
        "seed_encoding": PERTURBATION_SEED_ENCODING,
        "seed_encoding_rule": (
            "perturbation_seed = (raw_seed << 3) | (swap_count - 1); low bits "
            "0..5 encode swap counts 1..6, low bits 6 and 7 are invalid; "
            "swap_count is derived by decoding, never an independent input"
        ),
        "identity": (
            "the descendant is a pure function of (base_setup_id, "
            "sampler_version, perturbation_seed); the operator mix and the "
            "retry budget are frozen constants of this version, and no "
            "caller-configurable input can change the result"
        ),
        "hamming_per_swap": HAMMING_PER_SWAP,
        "min_swap_count": MIN_SWAP_COUNT,
        "max_swap_count": MAX_SWAP_COUNT,
        "min_hamming": PERTURBATION_MIN_HAMMING,
        "max_hamming": PERTURBATION_MAX_HAMMING,
        "max_attempts": MAX_PERTURBATION_ATTEMPTS,
        "max_attempts_status": "version constant of setup_perturbation_v1",
        "rejection_reasons": list(REJECTION_REASONS),
        "operators": [
            {**operator.to_dict(), "weight": operator.weight}
            for operator in OPERATORS
        ],
        "acceptance_stack": (
            "stratego.setups.contracts.validate_perturbation — engine "
            "inventory/legality, Flag cell fixed, canonical Hamming in "
            f"[{PERTURBATION_MIN_HAMMING}, {PERTURBATION_MAX_HAMMING}], every "
            "required family clause satisfied and every forbidden clause "
            "failed, initial mobility present"
        ),
        "construction_note": (
            "operators are proposal rules only; they are never consulted "
            "during acceptance, and a descendant is accepted solely by the "
            "frozen validator"
        ),
    }


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def apply_swaps(
    base_canonical: "list[int] | tuple[int, ...]",
    swaps: "list[tuple[int, int]] | tuple[tuple[int, int], ...]",
) -> "tuple[int, ...]":
    """Apply disjoint cell swaps to a canonical arrangement.

    Exposed so an auditor can replay a recorded swap list without re-entering
    the proposal machinery.
    """
    cells = list(base_canonical)
    for left, right in swaps:
        cells[left], cells[right] = cells[right], cells[left]
    return tuple(cells)


def _propose_candidate(
    base_canonical: "tuple[int, ...]",
    swap_count: int,
    rng: random.Random,
    weights: "tuple[float, ...]",
) -> "tuple[tuple[int, ...], tuple[str, ...], tuple[tuple[int, int], ...]]":
    """Draw one `swap_count`-swap candidate from `rng`.

    Swaps are applied sequentially to the working arrangement, and every cell
    touched is retired from later draws, so the swaps are disjoint and the
    candidate sits at canonical Hamming distance exactly
    `HAMMING_PER_SWAP * swap_count` from the base.
    """
    working = base_canonical
    flag_cell = base_canonical.index(FLAG)
    used: frozenset[int] = frozenset()
    applied: list[str] = []
    swaps: list[tuple[int, int]] = []

    for _ in range(swap_count):
        operator = _weighted_choice(rng, OPERATORS, weights)
        pairs = operator.eligible(working, flag_cell, used)
        left, right = _uniform_pair(rng, pairs)
        working = apply_swaps(working, ((left, right),))
        used = used | {left, right}
        applied.append(operator.name)
        swaps.append((left, right))

    return working, tuple(applied), tuple(swaps)


def _classify(violations: "list[str]") -> str:
    """Map the frozen validator's violation strings onto a rejection reason.

    The validator is the authority; this only buckets its verdict for
    reporting. The first violation decides the bucket, in the validator's own
    reporting order.
    """
    for violation in violations:
        if violation.startswith("inventory/legality"):
            return REJECTION_ENGINE_INVALID
        if violation.startswith("hamming distance"):
            return REJECTION_HAMMING_WINDOW
        if violation.startswith("the Flag moved"):
            return REJECTION_FLAG_MOVED
        if violation.startswith("family "):
            return REJECTION_FAMILY_PREDICATE
        if violation.startswith("stranded"):
            return REJECTION_STRANDED
    raise SetupLibraryError(  # pragma: no cover - defensive
        f"unclassifiable perturbation violation: {violations!r}"
    )


# ---------------------------------------------------------------------------
# The perturbation API
# ---------------------------------------------------------------------------


def perturbation_id(perturbation_seed: int) -> str:
    """The stable identifier of one perturbation.

    `setup_perturbation_v1:k3:0a1b...` — version, decoded swap count, composite
    seed. The swap count in the string is derived by decoding the seed, never
    supplied independently. Together with the base setup id this names the
    descendant exactly, which is the identity Agent 1's invariants require
    provenance to carry.
    """
    swap_count, _raw_seed = decode_perturbation_seed(perturbation_seed)
    return f"{PERTURBATION_VERSION}:k{swap_count}:{int(perturbation_seed):016x}"


@dataclass(frozen=True)
class PerturbationResult:
    """The outcome of one perturbation identity."""

    #: Whether a candidate satisfied the frozen validator within the budget.
    accepted: bool
    #: The descendant when accepted; the unmodified base when not.
    canonical: "tuple[int, ...]"
    base_canonical: "tuple[int, ...]"
    family_id: str
    #: Derived by decoding `perturbation_seed`; never an independent input.
    swap_count: int
    #: The composite `seed_encoding_v1` seed — the complete perturbation
    #: identity alongside the base and the sampler version.
    perturbation_seed: int
    perturbation_id: str
    #: Candidates drawn, including the accepted one.
    attempts: int
    accepted_attempt_index: "int | None"
    #: Operator names of the accepted candidate, in application order.
    operators_applied: "tuple[str, ...]" = ()
    #: The accepted candidate's disjoint cell swaps, in application order.
    swaps: "tuple[tuple[int, int], ...]" = ()
    #: `reason -> count` over the rejected candidates of this identity.
    rejections: dict = field(default_factory=dict)

    @property
    def hamming_from_base(self) -> int:
        return sum(
            1 for a, b in zip(self.base_canonical, self.canonical) if a != b
        )

    def to_dict(self) -> dict:
        return {
            "perturbation_version": PERTURBATION_VERSION,
            "accepted": self.accepted,
            "family_id": self.family_id,
            "swap_count": self.swap_count,
            "perturbation_seed": self.perturbation_seed,
            "perturbation_id": self.perturbation_id,
            "attempts": self.attempts,
            "accepted_attempt_index": self.accepted_attempt_index,
            "operators_applied": list(self.operators_applied),
            "swaps": [list(swap) for swap in self.swaps],
            "rejections": dict(sorted(self.rejections.items())),
            "hamming_from_base": self.hamming_from_base,
        }


def perturb_setup(
    base_canonical: "list[int] | tuple[int, ...]",
    family_id: str,
    perturbation_seed: int,
) -> PerturbationResult:
    """Perturb one base setup deterministically, or report exhaustion.

    The production `setup_perturbation_v1` identity function, per Agent 1's
    frozen invariant: the descendant is a pure function of
    `(base_canonical, family_id, perturbation_seed)`. There are no other
    inputs. The swap count is decoded from the composite seed
    (`seed_encoding_v1`), the operator mix is the frozen
    :data:`DEFAULT_OPERATOR_WEIGHTS`, and the retry budget is the frozen
    :data:`MAX_PERTURBATION_ATTEMPTS` version constant.

    Attempts `0, 1, 2, ...` are drawn from `derive_stream_seed` streams keyed
    by the decoded `(raw_seed, swap_count)`, and the first candidate for
    which the frozen
    :func:`~stratego.setups.contracts.validate_perturbation` reports no
    violation is accepted. Every rejection is classified and counted, so the
    retry process is as reproducible as the accepted result.

    The base is *not* re-validated here: it is an audited library entry. Only
    the candidate is validated, and only by the frozen validator.
    """
    swap_count, raw_seed = decode_perturbation_seed(perturbation_seed)
    return _perturb(
        base_canonical,
        family_id,
        perturbation_seed,
        swap_count,
        raw_seed,
        DEFAULT_OPERATOR_WEIGHTS,
        MAX_PERTURBATION_ATTEMPTS,
    )


def _perturb_setup_diagnostic(
    base_canonical: "list[int] | tuple[int, ...]",
    family_id: str,
    raw_seed: int,
    swap_count: int,
    weights: "tuple[float, ...]" = DEFAULT_OPERATOR_WEIGHTS,
    max_attempts: int = MAX_PERTURBATION_ATTEMPTS,
) -> PerturbationResult:
    """Test-only entry point that exposes the internal knobs.

    Exists so tests can force rarely-taken branches — exhaustion under a
    truncated budget, construction infeasibility under a pinned operator mix —
    without those knobs existing in the production identity. It does **not**
    define a second production semantics: with the default arguments it is
    exactly :func:`perturb_setup` on `encode(swap_count, raw_seed)`, and
    nothing outside `tests/` may call it.
    """
    return _perturb(
        base_canonical,
        family_id,
        encode_perturbation_seed(swap_count, raw_seed),
        swap_count,
        raw_seed,
        weights,
        max_attempts,
    )


def _perturb(
    base_canonical: "list[int] | tuple[int, ...]",
    family_id: str,
    perturbation_seed: int,
    swap_count: int,
    raw_seed: int,
    weights: "tuple[float, ...]",
    max_attempts: int,
) -> PerturbationResult:
    """The shared perturbation engine behind the public and diagnostic entries."""
    base = tuple(base_canonical)
    if len(base) != CANONICAL_CELLS:
        raise SetupLibraryError(
            f"expected {CANONICAL_CELLS} canonical entries, got {len(base)}"
        )
    if family_id not in FAMILY_BY_ID:
        raise SetupLibraryError(f"unknown family id: {family_id!r}")
    if max_attempts < 1:
        raise SetupLibraryError(f"max_attempts must be positive, got {max_attempts}")
    if len(weights) != len(OPERATORS):
        raise SetupLibraryError(
            f"expected {len(OPERATORS)} operator weights, got {len(weights)}"
        )

    identifier = perturbation_id(perturbation_seed)
    rejections: dict = {}

    for attempt in range(max_attempts):
        rng = random.Random(
            derive_stream_seed(_ATTEMPT_STREAM, raw_seed, swap_count, attempt)
        )
        try:
            candidate, operators_applied, swaps = _propose_candidate(
                base, swap_count, rng, weights
            )
        except _ProposalFailure:
            rejections[REJECTION_CONSTRUCTION] = (
                rejections.get(REJECTION_CONSTRUCTION, 0) + 1
            )
            continue

        violations = validate_perturbation(base, candidate, family_id)
        if violations:
            reason = _classify(violations)
            rejections[reason] = rejections.get(reason, 0) + 1
            continue

        return PerturbationResult(
            accepted=True,
            canonical=candidate,
            base_canonical=base,
            family_id=family_id,
            swap_count=swap_count,
            perturbation_seed=perturbation_seed,
            perturbation_id=identifier,
            attempts=attempt + 1,
            accepted_attempt_index=attempt,
            operators_applied=operators_applied,
            swaps=swaps,
            rejections=rejections,
        )

    return PerturbationResult(
        accepted=False,
        canonical=base,
        base_canonical=base,
        family_id=family_id,
        swap_count=swap_count,
        perturbation_seed=perturbation_seed,
        perturbation_id=identifier,
        attempts=max_attempts,
        accepted_attempt_index=None,
        rejections=rejections,
    )


__all__ = [
    "PERTURBATION_VERSION",
    "PERTURBATION_SEED_ENCODING",
    "HAMMING_PER_SWAP",
    "MIN_SWAP_COUNT",
    "MAX_SWAP_COUNT",
    "MAX_PERTURBATION_ATTEMPTS",
    "REJECTION_REASONS",
    "OPERATORS",
    "OPERATOR_NAMES",
    "OPERATOR_BY_NAME",
    "DEFAULT_OPERATOR_WEIGHTS",
    "SwapOperator",
    "PerturbationResult",
    "apply_swaps",
    "encode_perturbation_seed",
    "decode_perturbation_seed",
    "operator_mix_document",
    "perturb_setup",
    "perturbation_id",
]
