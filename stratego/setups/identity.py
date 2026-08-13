"""Canonical setup frame, reflection, canonicalization, fingerprints, seeds.

Specification sources:

- `00_PHASE_7_SEQUENCE_AND_COMMON_CONTRACT.md` (canonical setup orientation,
  reflection rule, determinism and identity)
- `01_AGENT_1_SETUP_CONTRACT_AND_TAXONOMY.md` (canonical representation,
  reflection and canonicalization)
- `03_game_engine_spec.md` section 7 (setup representation)

Canonical own-orientation frame
-------------------------------
Identical to the convention established by the accepted Phase 4 evaluation
setup bank (`stratego/evaluation/setup_bank.py`):

```text
rank 0 = own back row   (furthest from the lakes)
rank 3 = own front row  (nearest the lakes)
file 0..9 = absolute board column, left to right
canonical index = rank * 10 + file
```

The helpers here are deliberately self-contained rather than imported from the
evaluation module, so the production setup library never depends on the frozen
Phase 4 evaluation fixture. `tests/setups/test_identity.py` pins the two
implementations to each other exhaustively, which keeps them one convention
rather than two.

Reflection and canonical class representative
---------------------------------------------
Left-right reflection maps file `f` to `9 - f` inside every rank, which is
exactly the frozen engine's `reflect_setup` applied to the canonical 40-tuple.
Reflection is an involution, and no legal setup can equal its own reflection
(the single Flag would have to occupy two squares), so every reflection
equivalence class contains exactly two distinct arrangements. The canonical
class representative is the lexicographically smaller of the two piece-type
tuples under Python tuple comparison. That rule is total, deterministic,
content-only, and independently recomputable by any auditor.

Identity
--------
Content identity is a SHA-256 fingerprint over a domain-prefixed serialization
of the class representative, so a setup and its reflection always share one
fingerprint. Nothing here uses Python's process-randomized built-in hash.
"""

import hashlib

from ..engine.constants import PIECES_PER_PLAYER, PLAYERS, RED
from ..engine.setup import reflect_setup, serialize_setup, validate_setup

CANONICAL_RANKS = 4
CANONICAL_FILES = 10
CANONICAL_CELLS = CANONICAL_RANKS * CANONICAL_FILES
assert CANONICAL_CELLS == PIECES_PER_PLAYER

BACK_RANK = 0
FRONT_RANK = CANONICAL_RANKS - 1

#: Domain prefix of the reflection-class content fingerprint. Versioned so a
#: future change to fingerprint semantics is a new identifier, never a silent
#: reinterpretation.
CLASS_FINGERPRINT_DOMAIN = "stratego_setup_class_v1"

#: Domain prefix of the orientation-specific content fingerprint.
CONTENT_FINGERPRINT_DOMAIN = "stratego_setup_content_v1"

#: blake2b personalization tags for the Phase 7 seed streams. Distinct from the
#: Phase 4 bank tags (`strat-bnk`, `strat-sid`), so library streams can never
#: collide with evaluation-bank streams.
_BASE_SEED_PERSON = b"strat-lb7"
_ATTEMPT_SEED_PERSON = b"strat-at7"
_STREAM_SEED_PERSON = b"strat-st7"


class SetupLibraryError(ValueError):
    """Raised when a Phase 7 setup-library contract condition is violated."""


# ---------------------------------------------------------------------------
# Canonical frame helpers
# ---------------------------------------------------------------------------


def canonical_index(rank: int, file: int) -> int:
    """Canonical own-orientation cell index for `(rank, file)`."""
    if not 0 <= rank < CANONICAL_RANKS:
        raise SetupLibraryError(f"rank out of range: {rank}")
    if not 0 <= file < CANONICAL_FILES:
        raise SetupLibraryError(f"file out of range: {file}")
    return rank * CANONICAL_FILES + file


def canonical_rank_file(index: int) -> tuple[int, int]:
    """Inverse of :func:`canonical_index`."""
    if not 0 <= index < CANONICAL_CELLS:
        raise SetupLibraryError(f"canonical index out of range: {index}")
    return divmod(index, CANONICAL_FILES)


def canonical_neighbours(index: int) -> tuple[int, ...]:
    """Orthogonally adjacent canonical cells, ascending.

    Adjacency inside the 4x10 canonical grid is genuine board adjacency for
    both players because each setup area is a contiguous 4x10 block and
    :func:`orient_setup` only permutes whole rank rows.
    """
    rank, file = canonical_rank_file(index)
    neighbours = []
    if rank > 0:
        neighbours.append(canonical_index(rank - 1, file))
    if rank < CANONICAL_RANKS - 1:
        neighbours.append(canonical_index(rank + 1, file))
    if file > 0:
        neighbours.append(canonical_index(rank, file - 1))
    if file < CANONICAL_FILES - 1:
        neighbours.append(canonical_index(rank, file + 1))
    return tuple(sorted(neighbours))


def edge_file_distance(file: int) -> int:
    """Distance from the nearer board edge: `min(file, 9 - file)`, 0..4.

    Reflection-invariant by construction, so family predicates stated over it
    hold for a setup exactly when they hold for its reflection.
    """
    if not 0 <= file < CANONICAL_FILES:
        raise SetupLibraryError(f"file out of range: {file}")
    return min(file, CANONICAL_FILES - 1 - file)


def orient_setup(canonical: "list[int] | tuple[int, ...]", player: int) -> tuple[int, ...]:
    """Map a canonical own-orientation arrangement onto `player`'s setup order.

    Red's engine setup order already runs back-to-front, so the map is the
    identity. Blue's runs front-to-back, so the four rank rows are reversed
    while every file keeps its absolute column. The map equals the accepted
    Phase 4 convention (`stratego/evaluation/setup_bank.py`).
    """
    if player not in PLAYERS:
        raise SetupLibraryError(f"unknown player: {player!r}")
    entries = tuple(canonical)
    if len(entries) != CANONICAL_CELLS:
        raise SetupLibraryError(
            f"expected {CANONICAL_CELLS} canonical entries, got {len(entries)}"
        )
    if player == RED:
        return entries
    oriented: list[int | None] = [None] * CANONICAL_CELLS
    for index, piece_type in enumerate(entries):
        rank, file = canonical_rank_file(index)
        oriented[canonical_index(CANONICAL_RANKS - 1 - rank, file)] = piece_type
    return tuple(oriented)  # type: ignore[arg-type]


def deorient_setup(setup: "list[int] | tuple[int, ...]", player: int) -> tuple[int, ...]:
    """Inverse of :func:`orient_setup`. The blue map is its own inverse."""
    return orient_setup(setup, player)


# ---------------------------------------------------------------------------
# Reflection and the canonical class representative
# ---------------------------------------------------------------------------


def reflect_canonical(canonical: "list[int] | tuple[int, ...]") -> tuple[int, ...]:
    """Left-right reflection in the canonical frame: file `f` -> `9 - f`.

    Delegates to the frozen engine's :func:`reflect_setup`, whose per-row
    reversal over the row-major 40-tuple is exactly the canonical-frame
    reflection, because canonical rows are rank rows.
    """
    entries = tuple(canonical)
    if len(entries) != CANONICAL_CELLS:
        raise SetupLibraryError(
            f"expected {CANONICAL_CELLS} canonical entries, got {len(entries)}"
        )
    return reflect_setup(entries)


def canonical_class_representative(
    canonical: "list[int] | tuple[int, ...]",
) -> tuple[int, ...]:
    """The canonical representative of `{setup, reflect(setup)}`.

    The representative is the lexicographically smaller of the two piece-type
    tuples. Both class members map to the same representative, which is what
    makes base-library uniqueness a statement about reflection classes.
    """
    entries = tuple(canonical)
    reflected = reflect_canonical(entries)
    return entries if entries <= reflected else reflected


def is_canonical_representative(canonical: "list[int] | tuple[int, ...]") -> bool:
    """Whether `canonical` is its own reflection-class representative."""
    entries = tuple(canonical)
    return entries == canonical_class_representative(entries)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def content_fingerprint(canonical: "list[int] | tuple[int, ...]") -> str:
    """SHA-256 fingerprint of one specific orientation of a setup.

    Validates inventory first, so a fingerprint can never name an illegal
    arrangement. Orientation-specific: a setup and its reflection have
    different content fingerprints.
    """
    validated = validate_setup(tuple(canonical), RED)
    payload = f"{CONTENT_FINGERPRINT_DOMAIN}:{serialize_setup(validated)}".encode()
    return hashlib.sha256(payload).hexdigest()


def class_fingerprint(canonical: "list[int] | tuple[int, ...]") -> str:
    """SHA-256 fingerprint of a setup's reflection equivalence class.

    Computed over the canonical class representative, so `fingerprint(A) ==
    fingerprint(reflect(A))` always holds. This is the base-library identity
    fingerprint: exact-duplicate and reflection-duplicate detection are both
    equality of this value.
    """
    representative = canonical_class_representative(tuple(canonical))
    validated = validate_setup(representative, RED)
    payload = f"{CLASS_FINGERPRINT_DOMAIN}:{serialize_setup(validated)}".encode()
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Deterministic seed derivation
# ---------------------------------------------------------------------------


def _derive_seed(person: bytes, payload: str) -> int:
    """A 63-bit deterministic seed from a domain-separated blake2b digest."""
    digest = hashlib.blake2b(payload.encode(), digest_size=8, person=person).digest()
    return int.from_bytes(digest, "big") >> 1


def derive_base_seed(
    contract_version: str,
    library_version: str,
    master_seed: int,
    family_id: str,
    base_index: int,
) -> int:
    """The deterministic generation seed of one base setup.

    Pure function of the five identity inputs, so any process can regenerate
    any single base setup in isolation, without generating any preceding
    setup. Hashed rather than arithmetic, so consecutive base indices receive
    unrelated streams.
    """
    if base_index < 0:
        raise SetupLibraryError(f"base_index must be non-negative, got {base_index}")
    payload = (
        f"{contract_version}:{library_version}:{int(master_seed)}:"
        f"{family_id}:{int(base_index)}"
    )
    return _derive_seed(_BASE_SEED_PERSON, payload)


def derive_attempt_seed(base_seed: int, attempt: int) -> int:
    """The seed of rejection-sampling attempt `attempt` for one base setup.

    Generation for a base setup must draw candidate `attempt = 0, 1, 2, ...`
    from these streams and accept the first candidate that satisfies the
    contract. Rejection is therefore local to the base identity: no accepted
    setup ever depends on any other base setup's outcome, which is what keeps
    isolated regeneration exact.
    """
    if attempt < 0:
        raise SetupLibraryError(f"attempt must be non-negative, got {attempt}")
    return _derive_seed(_ATTEMPT_SEED_PERSON, f"{int(base_seed)}:{int(attempt)}")


def derive_stream_seed(purpose: str, *parts: "int | str") -> int:
    """A general domain-separated stream seed for later Phase 7 agents.

    `purpose` names the stream (for example `"perturbation"`); `parts` are the
    identity inputs. Provided so Agents 2-4 never invent ad-hoc seed mixing.
    """
    if not purpose:
        raise SetupLibraryError("purpose must be a non-empty string")
    payload = ":".join([purpose, *[str(part) for part in parts]])
    return _derive_seed(_STREAM_SEED_PERSON, payload)
