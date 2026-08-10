"""Fixed, deterministic, versioned evaluation-only setup bank.

Specification sources:

- `02_project_ruleset.md` section 6 (setup legality)
- `03_game_engine_spec.md` section 7 (required setup operations)
- Phase 4 Agent 1 instructions ("Fixed evaluation setup bank")

Scope
-----
This bank exists so that every Phase 4 evaluation match starts from a starting
position that is reproducible from an identifier alone. It is **not** the Phase 7
training setup generator: nothing here learns, adapts or is tuned against
results, and no policy may read the bank to infer an opponent's arrangement.

Canonical orientation frame
---------------------------
A setup is 40 piece types in the player's own `SETUP_SQUARES[player]` order,
which is ascending absolute square index. That ordering is *not* symmetric
between the colours:

- red's setup index 0..9 is board row 0, the row **furthest** from the centre;
- blue's setup index 0..9 is board row 6, the row **nearest** the centre.

So the same 40-tuple is a physically different arrangement depending on who
plays it. To make the structural rules below mean the same thing for both
players, arrangements are generated in a canonical own-orientation frame:

```text
rank 0 = own back row   (furthest from the lakes)
rank 3 = own front row  (nearest the lakes)
file 0..9 = absolute board column, left to right
canonical index = rank * 10 + file
```

:func:`orient_setup` maps that frame onto a player's setup indices. For red the
map is the identity; for blue it reverses the four rank blocks while leaving
files alone (a mirror across the horizontal centre line). Adjacency inside the
4x10 canonical grid is genuine board adjacency for both players, which is what
lets the flag/bomb rule below be stated once.

Structural generation
---------------------
Uniformly shuffled setups are legal but frequently degenerate -- a flag on the
front rank, a bare flag, bombs stacked at the back. Those distort a strength
ladder, because a baseline then wins by punishing a setup blunder rather than by
playing better. Generation therefore applies a handful of fixed, hand-coded
structural constraints (see :data:`GENERATION_FAMILY`). None of them is learned,
scored or tuned, and everything not covered by a constraint is drawn from a
seeded stream.

Determinism
-----------
Every pair is generated from `derive_pair_seed(root_seed, setup_pair_id)` alone,
so any process can rebuild any single pair without generating its neighbours,
and the whole bank is reproducible from `root_seed`. :func:`bank_digest` gives a
stable content hash for byte-identical regeneration checks.
"""

import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..engine.constants import (
    BLUE,
    BOMB,
    FLAG,
    GENERAL,
    MARSHAL,
    MINER,
    PIECE_COUNTS,
    PIECES_PER_PLAYER,
    PLAYERS,
    RED,
    SCOUT,
    SPY,
)
from ..engine.setup import (
    SetupError,
    deserialize_setup,
    serialize_setup,
    setup_to_placements,
    validate_setup,
    validate_setup_placement,
)

SETUP_BANK_VERSION = "evaluation_setup_bank_v1"
GENERATION_FAMILY = "structured_v1"

#: Root seed of the canonical Phase 4 bank. Changing it changes every setup, so
#: it is frozen alongside the bank version.
DEFAULT_BANK_ROOT_SEED = 20260101

#: Preferred bank size from the Phase 4 instructions (512 is the floor).
DEFAULT_BANK_SIZE = 1024
MINIMUM_BANK_SIZE = 512

CANONICAL_RANKS = 4
CANONICAL_FILES = 10
CANONICAL_CELLS = CANONICAL_RANKS * CANONICAL_FILES
assert CANONICAL_CELLS == PIECES_PER_PLAYER

FRONT_RANK = CANONICAL_RANKS - 1

# Rank preferences used by the structural rules. Higher is more likely; the
# weights are deliberately mild so the bank keeps broad variety.
_BOMB_RANK_WEIGHTS = (4.0, 3.0, 2.0, 1.0)
_SCOUT_RANK_WEIGHTS = (1.0, 2.0, 3.0, 4.0)
_SPY_RANK_WEIGHTS = (3.0, 3.0, 2.0, 1.0)
_MINER_RANK_WEIGHTS = (2.0, 2.0, 2.0, 1.0)

#: Flag is confined to the two rows furthest from the centre, favouring the back.
_FLAG_RANK_WEIGHTS = (3.0, 1.0, 0.0, 0.0)

#: Number of bombs placed orthogonally adjacent to the flag.
_FLAG_GUARD_BOMB_CHOICES = (2, 3)


class SetupBankError(ValueError):
    """Raised when a bank is malformed, inconsistent or fails validation."""


# ---------------------------------------------------------------------------
# Canonical frame helpers
# ---------------------------------------------------------------------------


def canonical_index(rank: int, file: int) -> int:
    """Canonical own-orientation cell index for `(rank, file)`."""
    if not 0 <= rank < CANONICAL_RANKS:
        raise SetupBankError(f"rank out of range: {rank}")
    if not 0 <= file < CANONICAL_FILES:
        raise SetupBankError(f"file out of range: {file}")
    return rank * CANONICAL_FILES + file


def canonical_rank_file(index: int) -> tuple[int, int]:
    """Inverse of :func:`canonical_index`."""
    if not 0 <= index < CANONICAL_CELLS:
        raise SetupBankError(f"canonical index out of range: {index}")
    return divmod(index, CANONICAL_FILES)


def canonical_neighbours(index: int) -> tuple[int, ...]:
    """Orthogonally adjacent canonical cells, ascending.

    Adjacency inside the 4x10 canonical grid is real board adjacency for both
    players, because each setup area is a contiguous 4x10 block of squares and
    :func:`orient_setup` only permutes whole rank blocks.
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


def orient_setup(canonical: "Sequence[int]", player: int) -> tuple[int, ...]:
    """Map a canonical own-orientation arrangement onto `player`'s setup order.

    Red's setup order already runs back-to-front, so the map is the identity.
    Blue's runs front-to-back, so the four rank blocks are reversed while each
    file keeps its absolute column.
    """
    if player not in PLAYERS:
        raise SetupBankError(f"unknown player: {player!r}")
    entries = tuple(canonical)
    if len(entries) != CANONICAL_CELLS:
        raise SetupBankError(f"expected {CANONICAL_CELLS} canonical entries, got {len(entries)}")
    if player == RED:
        return entries
    oriented: list[int | None] = [None] * CANONICAL_CELLS
    for index, piece_type in enumerate(entries):
        rank, file = canonical_rank_file(index)
        oriented[canonical_index(CANONICAL_RANKS - 1 - rank, file)] = piece_type
    return tuple(oriented)  # type: ignore[arg-type]


def deorient_setup(setup: "Sequence[int]", player: int) -> tuple[int, ...]:
    """Inverse of :func:`orient_setup`. The blue map is its own inverse."""
    return orient_setup(setup, player)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def derive_pair_seed(root_seed: int, setup_pair_id: int) -> int:
    """Deterministic per-pair seed.

    A hash rather than arithmetic mixing, so consecutive pair identifiers get
    unrelated arrangements instead of correlated ones, and so any pair can be
    rebuilt in isolation.
    """
    payload = f"{int(root_seed)}:{int(setup_pair_id)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"strat-bnk").digest()
    return int.from_bytes(digest, "big") >> 1


def _side_rng(pair_seed: int, player: int) -> random.Random:
    """Independent stream per side, so red and blue never mirror each other."""
    payload = f"{int(pair_seed)}:{int(player)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"strat-sid").digest()
    return random.Random(int.from_bytes(digest, "big") >> 1)


# ---------------------------------------------------------------------------
# Structured arrangement generation
# ---------------------------------------------------------------------------


def _weighted_take(
    rng: random.Random,
    pool: list[int],
    weights: "tuple[float, ...]",
    count: int,
) -> list[int]:
    """Draw `count` cells from `pool` without replacement, weighted by rank.

    `weights` is indexed by canonical rank. A cell whose rank weight is zero is
    never drawn; if that would make the draw impossible the caller has a bug, so
    it raises rather than silently relaxing the constraint.
    """
    if count < 0:
        raise SetupBankError("count must be non-negative")
    eligible = [cell for cell in pool if weights[canonical_rank_file(cell)[0]] > 0.0]
    if count > len(eligible):
        raise SetupBankError(
            f"cannot draw {count} cells from {len(eligible)} eligible candidates"
        )

    taken: list[int] = []
    for _ in range(count):
        candidates = [cell for cell in pool if weights[canonical_rank_file(cell)[0]] > 0.0]
        cell_weights = [weights[canonical_rank_file(cell)[0]] for cell in candidates]
        total = sum(cell_weights)
        target = rng.random() * total
        accumulated = 0.0
        chosen = candidates[-1]
        for cell, weight in zip(candidates, cell_weights):
            accumulated += weight
            if accumulated >= target:
                chosen = cell
                break
        pool.remove(chosen)
        taken.append(chosen)
    return taken


def generate_canonical_arrangement(rng: random.Random) -> tuple[int, ...]:
    """One structured, legal arrangement in the canonical own-orientation frame.

    The structural rules, in application order:

    1. the flag sits on rank 0 or rank 1, never within reach of an opening rush;
    2. two or three bombs are placed orthogonally adjacent to the flag;
    3. the remaining bombs favour the back ranks;
    4. scouts favour the front ranks, where their ray movement is worth having;
    5. the marshal and general are kept off the front rank;
    6. the spy and miners favour the back ranks;
    7. every remaining piece is dealt uniformly to the remaining cells.

    Rules 1-6 constrain roughly half the pieces. Everything else -- which file
    the flag takes, which bombs guard it, where the mid ranks go -- comes from
    `rng`, so the bank is varied rather than one arrangement permuted.
    """
    assignment: list[int | None] = [None] * CANONICAL_CELLS
    free = list(range(CANONICAL_CELLS))

    # 1. Flag.
    flag_cell = _weighted_take(rng, free, _FLAG_RANK_WEIGHTS, 1)[0]
    assignment[flag_cell] = FLAG

    # 2. Bombs adjacent to the flag.
    guards = [cell for cell in canonical_neighbours(flag_cell) if assignment[cell] is None]
    guard_count = min(rng.choice(_FLAG_GUARD_BOMB_CHOICES), len(guards))
    for cell in rng.sample(guards, guard_count):
        assignment[cell] = BOMB
        free.remove(cell)

    # 3. Remaining bombs.
    for cell in _weighted_take(rng, free, _BOMB_RANK_WEIGHTS, PIECE_COUNTS[BOMB] - guard_count):
        assignment[cell] = BOMB

    # 4. Scouts.
    for cell in _weighted_take(rng, free, _SCOUT_RANK_WEIGHTS, PIECE_COUNTS[SCOUT]):
        assignment[cell] = SCOUT

    # 5. Marshal and general, kept off the front rank.
    back_only = tuple(
        0.0 if rank == FRONT_RANK else 1.0 for rank in range(CANONICAL_RANKS)
    )
    for piece_type in (MARSHAL, GENERAL):
        cell = _weighted_take(rng, free, back_only, 1)[0]
        assignment[cell] = piece_type

    # 6. Spy and miners.
    for cell in _weighted_take(rng, free, _SPY_RANK_WEIGHTS, PIECE_COUNTS[SPY]):
        assignment[cell] = SPY
    for cell in _weighted_take(rng, free, _MINER_RANK_WEIGHTS, PIECE_COUNTS[MINER]):
        assignment[cell] = MINER

    # 7. Everything the structural rules do not mention.
    remaining: list[int] = []
    placed = {FLAG, BOMB, SCOUT, MARSHAL, GENERAL, SPY, MINER}
    for piece_type, count in sorted(PIECE_COUNTS.items()):
        if piece_type in placed:
            continue
        remaining.extend([piece_type] * count)
    if len(remaining) != len(free):
        raise SetupBankError(  # pragma: no cover - guarded by inventory arithmetic
            f"{len(remaining)} pieces left for {len(free)} free cells"
        )
    rng.shuffle(remaining)
    for cell, piece_type in zip(sorted(free), remaining):
        assignment[cell] = piece_type

    if any(entry is None for entry in assignment):  # pragma: no cover - defensive
        raise SetupBankError("structured generation left a cell empty")
    return tuple(assignment)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Bank entries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetupPair:
    """One reproducible starting position: a red setup and a blue setup.

    Both setups are stored in their own player's `SETUP_SQUARES` order, so they
    can be handed straight to :func:`stratego.engine.state.create_game`.
    """

    setup_pair_id: int
    red_setup: tuple[int, ...]
    blue_setup: tuple[int, ...]
    generation_seed: int
    bank_version: str = SETUP_BANK_VERSION
    generation_family: str = GENERATION_FAMILY

    def setup_for(self, player: int) -> tuple[int, ...]:
        if player == RED:
            return self.red_setup
        if player == BLUE:
            return self.blue_setup
        raise SetupBankError(f"unknown player: {player!r}")

    def to_dict(self) -> dict:
        return {
            "setup_pair_id": self.setup_pair_id,
            "red_setup": serialize_setup(self.red_setup),
            "blue_setup": serialize_setup(self.blue_setup),
            "generation_seed": self.generation_seed,
            "bank_version": self.bank_version,
            "generation_family": self.generation_family,
        }

    @staticmethod
    def from_dict(payload: dict) -> "SetupPair":
        return SetupPair(
            setup_pair_id=int(payload["setup_pair_id"]),
            red_setup=deserialize_setup(payload["red_setup"]),
            blue_setup=deserialize_setup(payload["blue_setup"]),
            generation_seed=int(payload["generation_seed"]),
            bank_version=str(payload["bank_version"]),
            generation_family=str(payload["generation_family"]),
        )


def generate_setup_pair(
    setup_pair_id: int,
    root_seed: int = DEFAULT_BANK_ROOT_SEED,
    bank_version: str = SETUP_BANK_VERSION,
) -> SetupPair:
    """Build one bank entry from its identifier alone."""
    if setup_pair_id < 0:
        raise SetupBankError(f"setup_pair_id must be non-negative, got {setup_pair_id}")
    pair_seed = derive_pair_seed(root_seed, setup_pair_id)
    red = orient_setup(generate_canonical_arrangement(_side_rng(pair_seed, RED)), RED)
    blue = orient_setup(generate_canonical_arrangement(_side_rng(pair_seed, BLUE)), BLUE)
    return SetupPair(
        setup_pair_id=setup_pair_id,
        red_setup=validate_setup(red, RED),
        blue_setup=validate_setup(blue, BLUE),
        generation_seed=pair_seed,
        bank_version=bank_version,
    )


# ---------------------------------------------------------------------------
# The bank
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetupBank:
    """An immutable, versioned collection of setup pairs."""

    bank_version: str
    root_seed: int
    generation_family: str
    pairs: tuple[SetupPair, ...]

    def __post_init__(self) -> None:
        identifiers = [pair.setup_pair_id for pair in self.pairs]
        if len(set(identifiers)) != len(identifiers):
            raise SetupBankError("setup bank contains duplicate setup_pair_id values")

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)

    @property
    def pair_ids(self) -> tuple[int, ...]:
        return tuple(pair.setup_pair_id for pair in self.pairs)

    def pair(self, setup_pair_id: int) -> SetupPair:
        for entry in self.pairs:
            if entry.setup_pair_id == setup_pair_id:
                return entry
        raise SetupBankError(f"setup_pair_id {setup_pair_id} is not in this bank")

    # -- construction ------------------------------------------------------

    @staticmethod
    def generate(
        size: int = DEFAULT_BANK_SIZE,
        root_seed: int = DEFAULT_BANK_ROOT_SEED,
        bank_version: str = SETUP_BANK_VERSION,
    ) -> "SetupBank":
        """Generate `size` pairs with identifiers `0..size-1`."""
        if size <= 0:
            raise SetupBankError(f"bank size must be positive, got {size}")
        pairs = tuple(
            generate_setup_pair(index, root_seed=root_seed, bank_version=bank_version)
            for index in range(size)
        )
        return SetupBank(
            bank_version=bank_version,
            root_seed=root_seed,
            generation_family=GENERATION_FAMILY,
            pairs=pairs,
        )

    def subset(self, pair_ids: "Iterable[int]") -> "SetupBank":
        """A bank restricted to `pair_ids`, preserving the given order."""
        selected = tuple(self.pair(int(pair_id)) for pair_id in pair_ids)
        return SetupBank(
            bank_version=self.bank_version,
            root_seed=self.root_seed,
            generation_family=self.generation_family,
            pairs=selected,
        )

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "bank_version": self.bank_version,
            "root_seed": self.root_seed,
            "generation_family": self.generation_family,
            "pair_count": len(self.pairs),
            "pairs": [pair.to_dict() for pair in self.pairs],
        }

    def to_json(self) -> str:
        """Canonical JSON. Key order is fixed, so the text is a stable digest input."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_dict(payload: dict) -> "SetupBank":
        bank = SetupBank(
            bank_version=str(payload["bank_version"]),
            root_seed=int(payload["root_seed"]),
            generation_family=str(payload["generation_family"]),
            pairs=tuple(SetupPair.from_dict(entry) for entry in payload["pairs"]),
        )
        declared = int(payload.get("pair_count", len(bank.pairs)))
        if declared != len(bank.pairs):
            raise SetupBankError(
                f"pair_count {declared} disagrees with {len(bank.pairs)} stored pairs"
            )
        return bank

    @staticmethod
    def from_json(text: str) -> "SetupBank":
        return SetupBank.from_dict(json.loads(text))

    def digest(self) -> str:
        return bank_digest(self)


def bank_digest(bank: SetupBank) -> str:
    """SHA-256 over the bank's canonical JSON form."""
    return hashlib.sha256(bank.to_json().encode()).hexdigest()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_setup_pair(pair: SetupPair) -> list[str]:
    """Every legality failure found in one pair, as human-readable strings.

    Checks the four properties the Phase 4 instructions name -- exact inventory,
    legal setup rows, no overlap or lakes, and a constructible game -- by
    delegating to the frozen engine rather than reimplementing legality.
    """
    failures: list[str] = []
    for player, setup, label in ((RED, pair.red_setup, "red"), (BLUE, pair.blue_setup, "blue")):
        try:
            validate_setup(setup, player)
        except SetupError as error:
            failures.append(f"pair {pair.setup_pair_id} {label} inventory: {error}")
            continue
        try:
            # Square-oriented validation is what can catch a lake square, a
            # square outside the setup area, or an unfilled setup square.
            validate_setup_placement(setup_to_placements(setup, player), player)
        except SetupError as error:
            failures.append(f"pair {pair.setup_pair_id} {label} placement: {error}")

    if failures:
        return failures

    # A constructible game proves the two setups do not overlap and that the
    # board the engine builds is exactly 80 occupied non-lake squares.
    from ..engine.state import create_game

    try:
        state = create_game(pair.red_setup, pair.blue_setup, game_id=f"bank-{pair.setup_pair_id}")
    except ValueError as error:
        failures.append(f"pair {pair.setup_pair_id} game construction: {error}")
        return failures

    occupied = sum(1 for entry in state.board if entry is not None)
    if occupied != 2 * PIECES_PER_PLAYER:
        failures.append(
            f"pair {pair.setup_pair_id} occupies {occupied} squares, expected "
            f"{2 * PIECES_PER_PLAYER}"
        )
    return failures


def validate_bank(bank: SetupBank) -> dict:
    """Validate an entire bank and return a machine-readable summary."""
    failures: list[str] = []
    for pair in bank.pairs:
        if pair.bank_version != bank.bank_version:
            failures.append(
                f"pair {pair.setup_pair_id} carries bank_version {pair.bank_version!r}, "
                f"bank declares {bank.bank_version!r}"
            )
        failures.extend(validate_setup_pair(pair))

    identifiers = [pair.setup_pair_id for pair in bank.pairs]
    duplicate_ids = sorted(
        {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
    )

    red_setups = {pair.red_setup for pair in bank.pairs}
    blue_setups = {pair.blue_setup for pair in bank.pairs}
    positions = {(pair.red_setup, pair.blue_setup) for pair in bank.pairs}

    return {
        "bank_version": bank.bank_version,
        "root_seed": bank.root_seed,
        "generation_family": bank.generation_family,
        "pair_count": len(bank.pairs),
        "validation_failures": failures,
        "validation_failure_count": len(failures),
        "duplicate_setup_pair_ids": duplicate_ids,
        "distinct_red_setups": len(red_setups),
        "distinct_blue_setups": len(blue_setups),
        "distinct_positions": len(positions),
        "digest": bank_digest(bank),
    }


def bank_diversity(bank: SetupBank) -> dict:
    """Coarse variation metrics, so 'enough variation' is measured, not asserted.

    Every metric is computed in the canonical own-orientation frame so red and
    blue are directly comparable.
    """
    flag_cells: set[int] = set()
    flag_files: set[int] = set()
    flag_ranks: dict[int, int] = {}
    front_rows: set[tuple[int, ...]] = set()
    back_rows: set[tuple[int, ...]] = set()
    guard_counts: dict[int, int] = {}

    for pair in bank.pairs:
        for player in PLAYERS:
            canonical = deorient_setup(pair.setup_for(player), player)
            flag_cell = canonical.index(FLAG)
            rank, file = canonical_rank_file(flag_cell)
            flag_cells.add(flag_cell)
            flag_files.add(file)
            flag_ranks[rank] = flag_ranks.get(rank, 0) + 1
            front_rows.add(canonical[FRONT_RANK * CANONICAL_FILES :])
            back_rows.add(canonical[:CANONICAL_FILES])
            guards = sum(
                1 for cell in canonical_neighbours(flag_cell) if canonical[cell] == BOMB
            )
            guard_counts[guards] = guard_counts.get(guards, 0) + 1

    return {
        "arrangements": 2 * len(bank.pairs),
        "distinct_flag_cells": len(flag_cells),
        "distinct_flag_files": len(flag_files),
        "flag_rank_histogram": {str(key): value for key, value in sorted(flag_ranks.items())},
        "distinct_front_rows": len(front_rows),
        "distinct_back_rows": len(back_rows),
        "flag_guard_bomb_histogram": {
            str(key): value for key, value in sorted(guard_counts.items())
        },
    }


def structural_violations(bank: SetupBank) -> list[str]:
    """Entries that break the documented structural rules of `structured_v1`.

    Kept separate from :func:`validate_bank` because these are bank-design
    invariants, not rules of Stratego: a violation means the generator drifted,
    not that a setup is illegal.
    """
    violations: list[str] = []
    for pair in bank.pairs:
        for player, label in ((RED, "red"), (BLUE, "blue")):
            canonical = deorient_setup(pair.setup_for(player), player)
            flag_cell = canonical.index(FLAG)
            flag_rank, _ = canonical_rank_file(flag_cell)
            if flag_rank > 1:
                violations.append(
                    f"pair {pair.setup_pair_id} {label}: flag on rank {flag_rank}"
                )
            guards = sum(
                1 for cell in canonical_neighbours(flag_cell) if canonical[cell] == BOMB
            )
            if guards < min(_FLAG_GUARD_BOMB_CHOICES):
                violations.append(
                    f"pair {pair.setup_pair_id} {label}: flag guarded by {guards} bombs"
                )
            for piece_type, name in ((MARSHAL, "marshal"), (GENERAL, "general")):
                cell = canonical.index(piece_type)
                if canonical_rank_file(cell)[0] == FRONT_RANK:
                    violations.append(
                        f"pair {pair.setup_pair_id} {label}: {name} on the front rank"
                    )
    return violations
