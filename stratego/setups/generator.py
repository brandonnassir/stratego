"""One deterministic base-setup generator: `setup_base_generator_v1`.

Specification sources:

- `02_AGENT_2_BASE_LIBRARY_GENERATOR.md` (one generator framework, master seed
  and isolated regeneration, generation requirements, entry metadata)
- Agent 1's frozen contracts in this package (`families`, `traits`,
  `identity`, `mobility`, `contracts`)

One framework, sixteen plans
----------------------------
There is exactly one construction procedure here. Families differ only by a
declarative :class:`FamilyPlan` — where the Flag may stand, how many Bombs
guard it, which regions each piece group must occupy — and every plan clause
exists to realize a clause of Agent 1's frozen family contract. Identity, seed
derivation, engine validation, family evaluation, mobility, canonicalization,
fingerprinting, metadata and serialization are shared infrastructure; nothing
is hand-authored per setup and no family has its own generator program.

The plans are *construction* rules, never acceptance rules. Acceptance is
always Agent 1's frozen predicate stack applied to the finished arrangement:

```text
engine validate_setup   ->  official inventory, legal representation
family predicate        ->  the primary-family contract
initial mobility        ->  the curated library-quality rule
```

A plan that drifts from its family contract therefore cannot smuggle an
invalid setup into the library; it can only waste attempts.

Rejection sampling and isolated rebuild
---------------------------------------
For a base identity, candidate attempts `0, 1, 2, ...` are drawn from the
frozen attempt-seed streams and the first candidate passing the predicate
stack is accepted, with every rejection counted by reason. Because the streams
depend only on `(contract, library, master_seed, family_id, base_index,
attempt)`, :func:`rebuild_base_setup` reproduces any accepted entry — setup and
metadata alike — without generating a single other base.

No cross-base state exists in this module: no accepted-fingerprint set, no
enumeration counter, no global RNG. Global uniqueness is an acceptance gate
owned by `library.py`, exactly as Agent 1's contract requires; a collision
there is a BLOCKED finding, never a licence to reroll.
"""

import random
from dataclasses import dataclass, field, replace

from ..engine.constants import (
    BOMB,
    COLONEL,
    FLAG,
    GENERAL,
    MAJOR,
    MARSHAL,
    MINER,
    PIECE_COUNTS,
    SCOUT,
    SPY,
)
from ..engine.setup import SetupError, deserialize_setup, serialize_setup, validate_setup
from .contracts import (
    BASES_PER_FAMILY,
    SETUP_FAMILY_VERSION,
    SETUP_GENERATOR_CONTRACT_VERSION,
    SETUP_LIBRARY_VERSION,
    SETUP_TRAIT_VECTOR_VERSION,
    base_setup_id,
    split_for_base_index,
)
from .families import FAMILY_BY_ID, FAMILY_IDS, family_contract
from .identity import (
    CANONICAL_CELLS,
    CANONICAL_FILES,
    CANONICAL_RANKS,
    FRONT_RANK,
    SetupLibraryError,
    canonical_class_representative,
    canonical_neighbours,
    canonical_rank_file,
    class_fingerprint,
    content_fingerprint,
    edge_file_distance,
    reflect_canonical,
)
from .mobility import setup_has_initial_mobility
from .seed import DEFAULT_SEED_CONTEXT, LibrarySeedContext
from .traits import DECOY_MIN_FLAG_DISTANCE, compute_trait_vector

#: Version identifier of this generator. A semantic change to construction,
#: attempt ordering or entry metadata is a new identifier.
GENERATOR_VERSION = "setup_base_generator_v1"

#: Attempts allowed per base identity before generation is declared
#: unachievable under the frozen contract. Observed acceptance needs a handful
#: of attempts at worst, so exhausting this budget means the family contract
#: cannot be met by this plan — a BLOCKED finding, not a reason to weaken a
#: contract or reroll a seed.
MAX_ATTEMPTS_PER_BASE = 256

# Rejection reasons. Every rejected candidate is counted under exactly one.
REJECTION_CONSTRUCTION = "construction_infeasible"
REJECTION_ENGINE_INVALID = "engine_invalid"
REJECTION_FAMILY_PREDICATE = "family_predicate"
REJECTION_STRANDED = "stranded"
REJECTION_REASONS = (
    REJECTION_CONSTRUCTION,
    REJECTION_ENGINE_INVALID,
    REJECTION_FAMILY_PREDICATE,
    REJECTION_STRANDED,
)


class _ConstructionFailure(Exception):
    """A plan step ran out of legal cells for this attempt.

    Internal control flow only: the attempt is counted as a
    `construction_infeasible` rejection and the next attempt stream is drawn.
    Never propagates out of :func:`generate_base_setup`.
    """


# ---------------------------------------------------------------------------
# Weighted, deterministic cell drawing
# ---------------------------------------------------------------------------

#: Rank weights meaning "no preference"; used wherever a family contract says
#: nothing about a piece group.
UNIFORM_RANKS = (1.0, 1.0, 1.0, 1.0)


def _rank_of(cell: int) -> int:
    return cell // CANONICAL_FILES


def _file_of(cell: int) -> int:
    return cell % CANONICAL_FILES


def _manhattan(cell_a: int, cell_b: int) -> int:
    rank_a, file_a = canonical_rank_file(cell_a)
    rank_b, file_b = canonical_rank_file(cell_b)
    return abs(rank_a - rank_b) + abs(file_a - file_b)


def _chebyshev(cell_a: int, cell_b: int) -> int:
    rank_a, file_a = canonical_rank_file(cell_a)
    rank_b, file_b = canonical_rank_file(cell_b)
    return max(abs(rank_a - rank_b), abs(file_a - file_b))


def _diagonal_neighbours(cell: int) -> tuple[int, ...]:
    rank, file = canonical_rank_file(cell)
    neighbours = []
    for delta_rank, delta_file in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        neighbour_rank, neighbour_file = rank + delta_rank, file + delta_file
        if 0 <= neighbour_rank < CANONICAL_RANKS and 0 <= neighbour_file < CANONICAL_FILES:
            neighbours.append(neighbour_rank * CANONICAL_FILES + neighbour_file)
    return tuple(sorted(neighbours))


def _weighted_choice(rng: random.Random, candidates: "list[int]", weights: "list[float]") -> int:
    """One cell drawn from `candidates` with the given positive weights.

    Candidates are always supplied as an ascending list, so the draw is a pure
    function of the stream and the candidate set, independent of any set or
    dict iteration order.
    """
    total = sum(weights)
    if not candidates or total <= 0.0:
        raise _ConstructionFailure("no eligible cell remains")
    target = rng.random() * total
    accumulated = 0.0
    for cell, weight in zip(candidates, weights):
        accumulated += weight
        if accumulated >= target:
            return cell
    return candidates[-1]  # pragma: no cover - float tail guard


class _Canvas:
    """The 40 canonical cells under construction."""

    __slots__ = ("cells", "bomb_forbidden")

    def __init__(self) -> None:
        self.cells: list[int | None] = [None] * CANONICAL_CELLS
        #: Cells reserved for a movable piece by a plan step (the F05 decoy),
        #: so a later Bomb draw cannot overwrite the structure it depends on.
        self.bomb_forbidden: set[int] = set()

    def free_cells(self, predicate=None) -> "list[int]":
        return [
            cell
            for cell, entry in enumerate(self.cells)
            if entry is None and (predicate is None or predicate(cell))
        ]

    def cells_of(self, piece_type: int) -> "list[int]":
        return [cell for cell, entry in enumerate(self.cells) if entry == piece_type]

    def place(self, cell: int, piece_type: int) -> None:
        if self.cells[cell] is not None:  # pragma: no cover - defensive
            raise _ConstructionFailure(f"cell {cell} is already occupied")
        self.cells[cell] = piece_type

    def draw(
        self,
        rng: random.Random,
        count: int,
        rank_weights: "tuple[float, ...]",
        predicate=None,
    ) -> "list[int]":
        """Draw `count` free cells without replacement, weighted by rank."""
        if count < 0:  # pragma: no cover - defensive
            raise _ConstructionFailure(f"negative draw count {count}")
        taken: list[int] = []
        for _ in range(count):
            candidates = [
                cell
                for cell in self.free_cells(predicate)
                if rank_weights[_rank_of(cell)] > 0.0
            ]
            weights = [rank_weights[_rank_of(cell)] for cell in candidates]
            taken.append(_weighted_choice(rng, candidates, weights))
            self.cells[taken[-1]] = -1  # provisional reservation
        for cell in taken:
            self.cells[cell] = None
        return taken

    def to_setup(self) -> "tuple[int, ...]":
        if any(entry is None for entry in self.cells):  # pragma: no cover - defensive
            raise _ConstructionFailure("construction left a cell empty")
        return tuple(self.cells)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Declarative family plans
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlagPlan:
    """Where the Flag may stand, and with what rank preference."""

    #: Relative weight per canonical rank; a zero forbids the rank outright.
    rank_weights: "tuple[float, float, float, float]"
    #: Permitted edge distances `min(file, 9 - file)`; reflection-invariant, so
    #: the rule reads the same for a setup and its mirror.
    edge_distances: "tuple[int, ...]" = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class BombPlan:
    """How the six Bombs are distributed."""

    #: Permitted counts of Bombs placed on the Flag's orthogonal neighbours.
    #: `None` means the plan takes no position and Bombs fall where the free
    #: draw puts them (F06, F07, F15).
    guard_choices: "tuple[int, ...] | None" = None
    #: Ceiling on the Flag's orthogonal Bomb guards after the free draw.
    #: `None` leaves the trait unconstrained.
    max_flag_orth_guards: "int | None" = None
    #: Forbid Bombs on the Flag's diagonal neighbours (F04).
    forbid_flag_diagonal: bool = False
    #: Ceiling on Bombs within Chebyshev distance 2 of the Flag (F03, F04).
    flag_zone_cap: "int | None" = None
    #: Permitted counts of Bombs in the front half, ranks 2-3 (F07).
    front2_choices: "tuple[int, ...] | None" = None
    #: Ceiling on Bombs standing on the front rank (F14).
    front_rank_cap: "int | None" = None
    #: Forbid orthogonally adjacent Bomb pairs (F06).
    dispersed: bool = False
    #: Floor on the number of distinct files Bombs occupy (F06).
    min_distinct_files: "int | None" = None
    #: Permitted counts of Bombs surrounding a movable decoy piece in the back
    #: half, at Manhattan distance >= 4 from the Flag (F05).
    decoy_guard_choices: "tuple[int, ...] | None" = None
    #: Rank preference of every Bomb the structural steps do not pin.
    rank_weights: "tuple[float, float, float, float]" = (4.0, 3.0, 2.0, 1.0)


@dataclass(frozen=True)
class GroupPlan:
    """Region quotas for one piece group.

    `front2_choices` and `front_rank_choices` are permitted *counts* of group
    members in ranks 2-3 and on rank 3; `None` means unconstrained, in which
    case only `rank_weights` shapes the placement. Every quota here mirrors a
    frozen family clause — `high_front2_count >= 5`, `scout_front_rank_count
    == 0`, and so on.
    """

    name: str
    pieces: "tuple[int, ...]"
    front2_choices: "tuple[int, ...] | None" = None
    front_rank_choices: "tuple[int, ...] | None" = None
    rank_weights: "tuple[float, float, float, float]" = UNIFORM_RANKS


@dataclass(frozen=True)
class FamilyPlan:
    """The complete construction rule for one primary family."""

    family_id: str
    flag: FlagPlan
    bombs: BombPlan
    groups: "tuple[GroupPlan, ...]"
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.family_id not in FAMILY_BY_ID:
            raise SetupLibraryError(f"unknown family id in plan: {self.family_id!r}")
        if sum(self.flag.rank_weights) <= 0.0:
            raise SetupLibraryError(f"{self.family_id}: Flag plan forbids every rank")
        if self.bombs.front2_choices is not None and (
            self.bombs.guard_choices or self.bombs.decoy_guard_choices
        ):
            raise SetupLibraryError(
                f"{self.family_id}: a Bomb front-half quota cannot be combined "
                "with pinned guard Bombs"
            )
        names = [group.name for group in self.groups]
        if len(set(names)) != len(names):
            raise SetupLibraryError(f"{self.family_id}: duplicate group names {names}")

    def to_dict(self) -> dict:
        return {
            "family_id": self.family_id,
            "rationale": self.rationale,
            "flag": {
                "rank_weights": list(self.flag.rank_weights),
                "edge_distances": list(self.flag.edge_distances),
            },
            "bombs": {
                "guard_choices": list(self.bombs.guard_choices)
                if self.bombs.guard_choices is not None
                else None,
                "max_flag_orth_guards": self.bombs.max_flag_orth_guards,
                "forbid_flag_diagonal": self.bombs.forbid_flag_diagonal,
                "flag_zone_cap": self.bombs.flag_zone_cap,
                "front2_choices": list(self.bombs.front2_choices)
                if self.bombs.front2_choices is not None
                else None,
                "front_rank_cap": self.bombs.front_rank_cap,
                "dispersed": self.bombs.dispersed,
                "min_distinct_files": self.bombs.min_distinct_files,
                "decoy_guard_choices": list(self.bombs.decoy_guard_choices)
                if self.bombs.decoy_guard_choices is not None
                else None,
                "rank_weights": list(self.bombs.rank_weights),
            },
            "groups": [
                {
                    "name": group.name,
                    "pieces": list(group.pieces),
                    "front2_choices": list(group.front2_choices)
                    if group.front2_choices is not None
                    else None,
                    "front_rank_choices": list(group.front_rank_choices)
                    if group.front_rank_choices is not None
                    else None,
                    "rank_weights": list(group.rank_weights),
                }
                for group in self.groups
            ],
        }


# ---------------------------------------------------------------------------
# The shared default groups
# ---------------------------------------------------------------------------

#: Mild rank preferences for groups no family clause pins. They follow the
#: accepted Phase 4 bank's structural precedent (Scouts forward, Spy and
#: Miners rearward, heavies off the very front), so families differ from one
#: another only in the dimensions their contracts actually name.
_DEFAULT_SCOUT_WEIGHTS = (1.0, 2.0, 3.0, 4.0)
_DEFAULT_MINER_WEIGHTS = (2.0, 2.0, 2.0, 1.0)
_DEFAULT_HEAVY_WEIGHTS = (2.0, 2.0, 2.0, 1.0)
_DEFAULT_SPY_WEIGHTS = (3.0, 3.0, 2.0, 1.0)

_HIGH_OTHER_PIECES = (COLONEL,) * PIECE_COUNTS[COLONEL] + (MAJOR,) * PIECE_COUNTS[MAJOR]


def _default_groups(
    *,
    scouts: "GroupPlan | None" = None,
    miners: "GroupPlan | None" = None,
    marshal: "GroupPlan | None" = None,
    general: "GroupPlan | None" = None,
    high_others: "GroupPlan | None" = None,
    spy: "GroupPlan | None" = None,
) -> "tuple[GroupPlan, ...]":
    """The six structured groups, with per-family overrides.

    Placement order is fixed — Scouts, Miners, Marshal, General, remaining
    high ranks, Spy — so a plan is a pure function of its attempt stream. Every
    piece no group names (Sergeants, Lieutenants, Captains) is dealt uniformly
    in the fill step.
    """
    return (
        scouts
        or GroupPlan("scouts", (SCOUT,) * PIECE_COUNTS[SCOUT], rank_weights=_DEFAULT_SCOUT_WEIGHTS),
        miners
        or GroupPlan("miners", (MINER,) * PIECE_COUNTS[MINER], rank_weights=_DEFAULT_MINER_WEIGHTS),
        marshal or GroupPlan("marshal", (MARSHAL,), rank_weights=_DEFAULT_HEAVY_WEIGHTS),
        general or GroupPlan("general", (GENERAL,), rank_weights=_DEFAULT_HEAVY_WEIGHTS),
        high_others
        or GroupPlan("high_others", _HIGH_OTHER_PIECES, rank_weights=_DEFAULT_HEAVY_WEIGHTS),
        spy or GroupPlan("spy", (SPY,), rank_weights=_DEFAULT_SPY_WEIGHTS),
    )


_BACK_TWO_FLAG = FlagPlan(rank_weights=(3.0, 1.0, 0.0, 0.0))
_BACK_RANK_FLAG = FlagPlan(rank_weights=(1.0, 0.0, 0.0, 0.0))
_UNIFORM_FLAG = FlagPlan(rank_weights=UNIFORM_RANKS)


# ---------------------------------------------------------------------------
# The sixteen plans
# ---------------------------------------------------------------------------

FAMILY_PLANS: "tuple[FamilyPlan, ...]" = (
    FamilyPlan(
        family_id="F00",
        flag=replace(_BACK_RANK_FLAG, edge_distances=(0,)),
        bombs=BombPlan(guard_choices=(2,)),
        groups=_default_groups(),
        rationale=(
            "Flag pinned to a back-rank corner, whose two orthogonal "
            "neighbours are both Bombs, sealing the corner exactly as "
            "flag_orth_bomb_guards == 2 demands"
        ),
    ),
    FamilyPlan(
        family_id="F01",
        flag=replace(_BACK_RANK_FLAG, edge_distances=(1, 2)),
        bombs=BombPlan(guard_choices=(2, 3)),
        groups=_default_groups(),
        rationale=(
            "Flag one or two files off a back-rank corner with two or three "
            "of its three orthogonal neighbours bombed"
        ),
    ),
    FamilyPlan(
        family_id="F02",
        flag=replace(_BACK_RANK_FLAG, edge_distances=(3, 4)),
        bombs=BombPlan(guard_choices=(2, 3)),
        groups=_default_groups(),
        rationale="central back-rank Flag with the same two-or-three guard wall",
    ),
    FamilyPlan(
        family_id="F03",
        flag=_BACK_TWO_FLAG,
        bombs=BombPlan(
            guard_choices=(1,),
            max_flag_orth_guards=1,
            flag_zone_cap=3,
        ),
        groups=_default_groups(),
        rationale=(
            "exactly one orthogonal guard, and the free Bombs are capped at "
            "three inside the Flag's Chebyshev-2 zone so the defense stays "
            "partial (the forbidden clause trips at four)"
        ),
    ),
    FamilyPlan(
        family_id="F04",
        flag=_BACK_TWO_FLAG,
        bombs=BombPlan(
            guard_choices=(0,),
            max_flag_orth_guards=0,
            forbid_flag_diagonal=True,
            flag_zone_cap=2,
        ),
        groups=_default_groups(),
        rationale=(
            "no Bomb touches the Flag orthogonally or diagonally and at most "
            "two Bombs sit in its Chebyshev-2 zone, so the Flag square "
            "advertises nothing"
        ),
    ),
    FamilyPlan(
        family_id="F05",
        flag=_BACK_TWO_FLAG,
        bombs=BombPlan(
            guard_choices=(0, 1),
            max_flag_orth_guards=1,
            decoy_guard_choices=(2, 3),
        ),
        groups=_default_groups(),
        rationale=(
            "a two- or three-Bomb pocket is built around a reserved movable "
            "cell in the back half at Manhattan distance >= 4 from the Flag, "
            "while the Flag itself keeps at most one guard"
        ),
    ),
    FamilyPlan(
        family_id="F06",
        flag=_BACK_TWO_FLAG,
        bombs=BombPlan(
            max_flag_orth_guards=1,
            dispersed=True,
            min_distinct_files=5,
            rank_weights=UNIFORM_RANKS,
        ),
        groups=_default_groups(),
        rationale=(
            "all six Bombs drawn as mutually non-adjacent lane blockers over "
            "at least five distinct files, with no concentrated Flag fortress"
        ),
    ),
    FamilyPlan(
        family_id="F07",
        flag=_BACK_TWO_FLAG,
        bombs=BombPlan(front2_choices=(4, 5, 6), rank_weights=UNIFORM_RANKS),
        groups=_default_groups(),
        rationale="four to six Bombs pushed into ranks 2-3 as forward lane blockers",
    ),
    FamilyPlan(
        family_id="F08",
        flag=_BACK_TWO_FLAG,
        bombs=BombPlan(),
        groups=_default_groups(
            marshal=GroupPlan("marshal", (MARSHAL,), front2_choices=(1,)),
            general=GroupPlan("general", (GENERAL,), front2_choices=(1,)),
            high_others=GroupPlan(
                "high_others", _HIGH_OTHER_PIECES, front2_choices=(3, 4, 5)
            ),
        ),
        rationale=(
            "Marshal and General forward plus three to five of the remaining "
            "five heavies, giving high_front2_count 5..7"
        ),
    ),
    FamilyPlan(
        family_id="F09",
        flag=_BACK_TWO_FLAG,
        bombs=BombPlan(),
        groups=_default_groups(
            marshal=GroupPlan("marshal", (MARSHAL,), front2_choices=(0,)),
            general=GroupPlan("general", (GENERAL,), front2_choices=(0,)),
            high_others=GroupPlan(
                "high_others", _HIGH_OTHER_PIECES, front2_choices=(0, 1, 2)
            ),
        ),
        rationale=(
            "the mirror of F08: Marshal and General held back with three to "
            "five of the remaining heavies, giving high_back2_count 5..7"
        ),
    ),
    FamilyPlan(
        family_id="F10",
        flag=_BACK_TWO_FLAG,
        bombs=BombPlan(),
        groups=_default_groups(
            scouts=GroupPlan(
                "scouts",
                (SCOUT,) * PIECE_COUNTS[SCOUT],
                front2_choices=(6, 7, 8),
                front_rank_choices=(3, 4, 5),
            )
        ),
        rationale="six to eight Scouts in ranks 2-3, three to five of them on the front rank",
    ),
    FamilyPlan(
        family_id="F11",
        flag=_BACK_TWO_FLAG,
        bombs=BombPlan(),
        groups=_default_groups(
            scouts=GroupPlan(
                "scouts",
                (SCOUT,) * PIECE_COUNTS[SCOUT],
                front2_choices=(0, 1, 2, 3),
                front_rank_choices=(0,),
                rank_weights=(3.0, 3.0, 1.0, 0.0),
            )
        ),
        rationale=(
            "no Scout on the front rank and at most three on rank 2, leaving "
            "five to eight preserved in the back half"
        ),
    ),
    FamilyPlan(
        family_id="F12",
        flag=_BACK_TWO_FLAG,
        bombs=BombPlan(),
        groups=_default_groups(
            miners=GroupPlan(
                "miners", (MINER,) * PIECE_COUNTS[MINER], front2_choices=(3, 4, 5)
            )
        ),
        rationale="three to five of the five Miners in ranks 2-3, ready to clear forward lanes",
    ),
    FamilyPlan(
        family_id="F13",
        flag=_BACK_TWO_FLAG,
        bombs=BombPlan(),
        groups=_default_groups(
            miners=GroupPlan(
                "miners",
                (MINER,) * PIECE_COUNTS[MINER],
                front2_choices=(0, 1),
                front_rank_choices=(0,),
                rank_weights=(3.0, 3.0, 1.0, 0.0),
            )
        ),
        rationale="no Miner on the front rank and at least four held in ranks 0-1",
    ),
    FamilyPlan(
        family_id="F14",
        flag=_BACK_RANK_FLAG,
        bombs=BombPlan(guard_choices=(2, 3), front_rank_cap=2),
        groups=_default_groups(
            scouts=GroupPlan(
                "scouts",
                (SCOUT,) * PIECE_COUNTS[SCOUT],
                front2_choices=(3, 4, 5, 6),
                rank_weights=_DEFAULT_SCOUT_WEIGHTS,
            ),
            miners=GroupPlan(
                "miners",
                (MINER,) * PIECE_COUNTS[MINER],
                front2_choices=(0, 1, 2, 3),
                rank_weights=_DEFAULT_MINER_WEIGHTS,
            ),
            marshal=GroupPlan(
                "marshal", (MARSHAL,), front_rank_choices=(0,), rank_weights=_DEFAULT_HEAVY_WEIGHTS
            ),
            general=GroupPlan(
                "general", (GENERAL,), front_rank_choices=(0,), rank_weights=_DEFAULT_HEAVY_WEIGHTS
            ),
        ),
        rationale=(
            "the textbook shape: guarded back-rank Flag, at most two "
            "front-rank Bombs (which alone secures >= 8 movable front-rank "
            "pieces), Marshal and General off the front rank, three to six "
            "forward Scouts, at least two reserved Miners"
        ),
    ),
    FamilyPlan(
        family_id="F15",
        flag=_UNIFORM_FLAG,
        bombs=BombPlan(rank_weights=UNIFORM_RANKS),
        groups=_default_groups(
            scouts=GroupPlan("scouts", (SCOUT,) * PIECE_COUNTS[SCOUT]),
            miners=GroupPlan("miners", (MINER,) * PIECE_COUNTS[MINER]),
            marshal=GroupPlan("marshal", (MARSHAL,)),
            general=GroupPlan("general", (GENERAL,)),
            high_others=GroupPlan("high_others", _HIGH_OTHER_PIECES),
            spy=GroupPlan("spy", (SPY,)),
        ),
        rationale=(
            "deliberately unstructured: every group uniform over the whole "
            "zone including the Flag, so the unconventional-feature clause is "
            "met by high-entropy draws rather than by construction; the "
            "conventional fortress signature is filtered by the frozen "
            "forbidden clause"
        ),
    ),
)

PLAN_BY_FAMILY = {plan.family_id: plan for plan in FAMILY_PLANS}

assert tuple(PLAN_BY_FAMILY) == FAMILY_IDS, "one plan per frozen family, in order"


def family_plan(family_id: str) -> FamilyPlan:
    """Look up the construction plan of one family."""
    try:
        return PLAN_BY_FAMILY[family_id]
    except KeyError as error:
        raise SetupLibraryError(f"unknown family id: {family_id!r}") from error


def plans_document() -> dict:
    """The machine-readable plan table, for the generation artifact."""
    return {
        "generator_version": GENERATOR_VERSION,
        "max_attempts_per_base": MAX_ATTEMPTS_PER_BASE,
        "rejection_reasons": list(REJECTION_REASONS),
        "construction_order": [
            "flag",
            "flag guard bombs",
            "decoy pocket bombs",
            "free bombs",
            "scouts",
            "miners",
            "marshal",
            "general",
            "high_others",
            "spy",
            "uniform fill of the remaining pieces",
        ],
        "acceptance_order": ["engine_validate_setup", "family_predicate", "initial_mobility"],
        "plans": [plan.to_dict() for plan in FAMILY_PLANS],
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _place_flag(rng: random.Random, canvas: _Canvas, plan: FamilyPlan) -> int:
    allowed = set(plan.flag.edge_distances)
    candidates = [
        cell
        for cell in range(CANONICAL_CELLS)
        if plan.flag.rank_weights[_rank_of(cell)] > 0.0
        and edge_file_distance(_file_of(cell)) in allowed
    ]
    weights = [plan.flag.rank_weights[_rank_of(cell)] for cell in candidates]
    flag_cell = _weighted_choice(rng, candidates, weights)
    canvas.place(flag_cell, FLAG)
    return flag_cell


def _place_guard_bombs(
    rng: random.Random, canvas: _Canvas, plan: FamilyPlan, flag_cell: int
) -> int:
    """Bombs pinned to the Flag's orthogonal neighbours. Returns the count."""
    if plan.bombs.guard_choices is None:
        return 0
    guard_count = rng.choice(plan.bombs.guard_choices)
    neighbours = [cell for cell in canonical_neighbours(flag_cell) if canvas.cells[cell] is None]
    if guard_count > len(neighbours):
        raise _ConstructionFailure(
            f"{plan.family_id}: {guard_count} guards requested, {len(neighbours)} cells free"
        )
    for cell in rng.sample(neighbours, guard_count):
        canvas.place(cell, BOMB)
    return guard_count


def _place_decoy_pocket(
    rng: random.Random, canvas: _Canvas, plan: FamilyPlan, flag_cell: int
) -> int:
    """The F05 false fortress: Bombs around a reserved movable back-half cell."""
    if plan.bombs.decoy_guard_choices is None:
        return 0
    pocket_count = rng.choice(plan.bombs.decoy_guard_choices)
    candidates = [
        cell
        for cell in canvas.free_cells()
        if _rank_of(cell) <= 1
        and _manhattan(cell, flag_cell) >= DECOY_MIN_FLAG_DISTANCE
        and sum(1 for n in canonical_neighbours(cell) if canvas.cells[n] is None) >= pocket_count
    ]
    if not candidates:
        raise _ConstructionFailure(f"{plan.family_id}: no eligible decoy cell")
    decoy_cell = candidates[rng.randrange(len(candidates))]
    free_neighbours = [
        cell for cell in canonical_neighbours(decoy_cell) if canvas.cells[cell] is None
    ]
    for cell in rng.sample(free_neighbours, pocket_count):
        canvas.place(cell, BOMB)
    # The decoy piece itself must stay movable: only the Flag and the Bombs are
    # immovable, and both are fully placed before the fill step, so reserving
    # the cell against later Bomb draws is sufficient.
    canvas.bomb_forbidden.add(decoy_cell)
    return pocket_count


def _place_free_bombs(
    rng: random.Random, canvas: _Canvas, plan: FamilyPlan, flag_cell: int, placed: int
) -> None:
    """The Bombs no structural step pinned, drawn under the plan's filters."""
    remaining = PIECE_COUNTS[BOMB] - placed
    if remaining < 0:  # pragma: no cover - guarded by plan validation
        raise _ConstructionFailure("more Bombs pinned than the inventory allows")

    if plan.bombs.front2_choices is not None:
        front_half = rng.choice(plan.bombs.front2_choices)
        regions = ["front2"] * front_half + ["back2"] * (remaining - front_half)
    else:
        regions = ["any"] * remaining

    flag_orthogonals = set(canonical_neighbours(flag_cell))
    flag_diagonals = set(_diagonal_neighbours(flag_cell))

    for region in regions:
        bomb_cells = canvas.cells_of(BOMB)
        orth_guards = sum(1 for cell in bomb_cells if cell in flag_orthogonals)
        zone_bombs = sum(1 for cell in bomb_cells if _chebyshev(cell, flag_cell) <= 2)
        front_rank_bombs = sum(1 for cell in bomb_cells if _rank_of(cell) == FRONT_RANK)
        used_files = {_file_of(cell) for cell in bomb_cells}
        still_to_place = len(regions) - (len(bomb_cells) - placed)
        files_needed = (
            max(0, plan.bombs.min_distinct_files - len(used_files))
            if plan.bombs.min_distinct_files is not None
            else 0
        )
        adjacent_to_bomb = {
            neighbour for cell in bomb_cells for neighbour in canonical_neighbours(cell)
        }

        def eligible(cell: int) -> bool:
            if cell in canvas.bomb_forbidden:
                return False
            if region == "front2" and _rank_of(cell) < 2:
                return False
            if region == "back2" and _rank_of(cell) > 1:
                return False
            if (
                plan.bombs.max_flag_orth_guards is not None
                and cell in flag_orthogonals
                and orth_guards >= plan.bombs.max_flag_orth_guards
            ):
                return False
            if plan.bombs.forbid_flag_diagonal and cell in flag_diagonals:
                return False
            if (
                plan.bombs.flag_zone_cap is not None
                and _chebyshev(cell, flag_cell) <= 2
                and zone_bombs >= plan.bombs.flag_zone_cap
            ):
                return False
            if (
                plan.bombs.front_rank_cap is not None
                and _rank_of(cell) == FRONT_RANK
                and front_rank_bombs >= plan.bombs.front_rank_cap
            ):
                return False
            if plan.bombs.dispersed and cell in adjacent_to_bomb:
                return False
            if still_to_place <= files_needed and _file_of(cell) in used_files:
                return False
            return True

        chosen = canvas.draw(rng, 1, plan.bombs.rank_weights, eligible)[0]
        canvas.place(chosen, BOMB)


def _place_group(rng: random.Random, canvas: _Canvas, group: GroupPlan) -> None:
    """Place one piece group under its region quotas."""
    total = len(group.pieces)
    front2 = rng.choice(group.front2_choices) if group.front2_choices is not None else None
    front_rank = None
    if group.front_rank_choices is not None:
        allowed = [
            value
            for value in group.front_rank_choices
            if value <= total and (front2 is None or value <= front2)
        ]
        if not allowed:
            raise _ConstructionFailure(
                f"{group.name}: no front-rank quota fits front-half quota {front2}"
            )
        front_rank = rng.choice(allowed)

    cells: list[int] = []
    if front_rank is not None:
        cells += canvas.draw(rng, front_rank, UNIFORM_RANKS, lambda c: _rank_of(c) == FRONT_RANK)
        if front2 is not None:
            cells += canvas.draw(rng, front2 - front_rank, UNIFORM_RANKS, lambda c: _rank_of(c) == 2)
            cells += canvas.draw(rng, total - front2, group.rank_weights, lambda c: _rank_of(c) <= 1)
        else:
            cells += canvas.draw(
                rng, total - front_rank, group.rank_weights, lambda c: _rank_of(c) < FRONT_RANK
            )
    elif front2 is not None:
        cells += canvas.draw(rng, front2, group.rank_weights, lambda c: _rank_of(c) >= 2)
        cells += canvas.draw(rng, total - front2, group.rank_weights, lambda c: _rank_of(c) <= 1)
    else:
        cells += canvas.draw(rng, total, group.rank_weights)

    pieces = list(group.pieces)
    rng.shuffle(pieces)
    for cell, piece_type in zip(sorted(cells), pieces):
        canvas.place(cell, piece_type)


def _fill_remaining(rng: random.Random, canvas: _Canvas) -> None:
    """Deal every piece no plan step named into the cells left free."""
    remaining: list[int] = []
    for piece_type, required in sorted(PIECE_COUNTS.items()):
        placed = len(canvas.cells_of(piece_type))
        if placed > required:  # pragma: no cover - defensive
            raise _ConstructionFailure(f"piece type {piece_type} over-placed")
        remaining.extend([piece_type] * (required - placed))
    free = canvas.free_cells()
    if len(remaining) != len(free):  # pragma: no cover - defensive
        raise _ConstructionFailure(f"{len(remaining)} pieces left for {len(free)} cells")
    rng.shuffle(remaining)
    for cell, piece_type in zip(free, remaining):
        canvas.place(cell, piece_type)


def construct_candidate(plan: FamilyPlan, rng: random.Random) -> "tuple[int, ...]":
    """One candidate arrangement for `plan`, drawn entirely from `rng`.

    The result is a canonical own-orientation 40-tuple. It is a *candidate*:
    legality, family membership and initial mobility are decided afterwards by
    Agent 1's frozen predicates, never by this function.
    """
    canvas = _Canvas()
    flag_cell = _place_flag(rng, canvas, plan)
    pinned = _place_guard_bombs(rng, canvas, plan, flag_cell)
    pinned += _place_decoy_pocket(rng, canvas, plan, flag_cell)
    _place_free_bombs(rng, canvas, plan, flag_cell, pinned)
    for group in plan.groups:
        _place_group(rng, canvas, group)
    _fill_remaining(rng, canvas)
    return canvas.to_setup()


# ---------------------------------------------------------------------------
# Accepted entries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseSetupEntry:
    """One accepted base-library entry: content plus full provenance.

    Every field is derived from the base identity and the frozen contracts.
    Nothing here records a game outcome, a model score or any strength signal;
    `library.py` asserts that statically against the serialized field names.
    """

    base_setup_id: str
    library_version: str
    contract_version: str
    family_contract_version: str
    trait_schema_version: str
    generator_version: str
    family_id: str
    family_key: str
    base_index: int
    split: str
    canonical_setup: "tuple[int, ...]"
    fingerprint: str
    content_fingerprint: str
    reflected_content_fingerprint: str
    master_seed: int
    generation_seed: int
    accepted_attempt_index: int
    accepted_attempt_seed: int
    generation_attempts: int
    trait_vector: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """The serialized entry, in the frozen Agent 1 entry format."""
        return {
            "base_setup_id": self.base_setup_id,
            "library_version": self.library_version,
            "contract_version": self.contract_version,
            "family_contract_version": self.family_contract_version,
            "trait_schema_version": self.trait_schema_version,
            "generator_version": self.generator_version,
            "family_id": self.family_id,
            "family_key": self.family_key,
            "base_index": self.base_index,
            "split": self.split,
            "canonical_setup": serialize_setup(self.canonical_setup),
            "fingerprint": self.fingerprint,
            "content_fingerprint": self.content_fingerprint,
            "reflected_content_fingerprint": self.reflected_content_fingerprint,
            "master_seed": self.master_seed,
            "generation_seed": self.generation_seed,
            "accepted_attempt_index": self.accepted_attempt_index,
            "accepted_attempt_seed": self.accepted_attempt_seed,
            "generation_attempts": self.generation_attempts,
            "trait_vector": dict(self.trait_vector),
        }

    @staticmethod
    def from_dict(payload: dict) -> "BaseSetupEntry":
        """Rebuild an entry from its serialized form, for independent audit."""
        return BaseSetupEntry(
            base_setup_id=str(payload["base_setup_id"]),
            library_version=str(payload["library_version"]),
            contract_version=str(payload["contract_version"]),
            family_contract_version=str(payload["family_contract_version"]),
            trait_schema_version=str(payload["trait_schema_version"]),
            generator_version=str(payload["generator_version"]),
            family_id=str(payload["family_id"]),
            family_key=str(payload["family_key"]),
            base_index=int(payload["base_index"]),
            split=str(payload["split"]),
            canonical_setup=deserialize_setup(payload["canonical_setup"]),
            fingerprint=str(payload["fingerprint"]),
            content_fingerprint=str(payload["content_fingerprint"]),
            reflected_content_fingerprint=str(payload["reflected_content_fingerprint"]),
            master_seed=int(payload["master_seed"]),
            generation_seed=int(payload["generation_seed"]),
            accepted_attempt_index=int(payload["accepted_attempt_index"]),
            accepted_attempt_seed=int(payload["accepted_attempt_seed"]),
            generation_attempts=int(payload["generation_attempts"]),
            trait_vector=dict(payload["trait_vector"]),
        )


@dataclass(frozen=True)
class BaseGenerationRecord:
    """An accepted entry together with the attempts it cost."""

    entry: BaseSetupEntry
    #: `reason -> count` over the rejected attempts of this base identity.
    rejections: dict


def _reject_reason(
    candidate: "tuple[int, ...]", family_id: str
) -> "tuple[str | None, list[str]]":
    """Apply Agent 1's frozen acceptance stack. `(None, [])` means accepted."""
    try:
        validate_setup(candidate, 0)
    except SetupError:
        return REJECTION_ENGINE_INVALID, []
    satisfied, violations = family_contract(family_id).evaluate(
        compute_trait_vector(candidate)
    )
    if not satisfied:
        return REJECTION_FAMILY_PREDICATE, violations
    if not setup_has_initial_mobility(candidate):
        return REJECTION_STRANDED, []
    return None, []


def generate_base_setup(
    family_id: str,
    base_index: int,
    seed_context: LibrarySeedContext = DEFAULT_SEED_CONTEXT,
) -> BaseGenerationRecord:
    """Generate one base-library entry from its identity alone.

    Draws attempts `0, 1, 2, ...` from the frozen attempt streams and accepts
    the first candidate that passes engine validation, the primary-family
    predicate and the initial-mobility rule. The accepted arrangement is then
    canonicalized to its reflection-class representative — legal, family
    membership and mobility are all reflection-invariant under the frozen
    contract, so canonicalizing after acceptance cannot change the verdict.

    Raises :class:`SetupLibraryError` if the attempt budget is exhausted: that
    is the BLOCKED condition of the frozen contract, never a licence to reroll
    the seed or weaken the family.
    """
    if family_id not in FAMILY_BY_ID:
        raise SetupLibraryError(f"unknown family id: {family_id!r}")
    if not 0 <= base_index < BASES_PER_FAMILY:
        raise SetupLibraryError(
            f"base_index must be in 0..{BASES_PER_FAMILY - 1}, got {base_index}"
        )

    plan = family_plan(family_id)
    contract = FAMILY_BY_ID[family_id]
    base_seed = seed_context.base_seed(family_id, base_index)
    rejections: dict = {}

    for attempt in range(MAX_ATTEMPTS_PER_BASE):
        attempt_seed = seed_context.attempt_seed(family_id, base_index, attempt)
        rng = random.Random(attempt_seed)
        try:
            candidate = construct_candidate(plan, rng)
        except _ConstructionFailure:
            rejections[REJECTION_CONSTRUCTION] = rejections.get(REJECTION_CONSTRUCTION, 0) + 1
            continue

        reason, _violations = _reject_reason(candidate, family_id)
        if reason is not None:
            rejections[reason] = rejections.get(reason, 0) + 1
            continue

        representative = canonical_class_representative(candidate)
        entry = BaseSetupEntry(
            base_setup_id=base_setup_id(family_id, base_index),
            library_version=seed_context.library_version,
            contract_version=seed_context.contract_version,
            family_contract_version=SETUP_FAMILY_VERSION,
            trait_schema_version=SETUP_TRAIT_VECTOR_VERSION,
            generator_version=GENERATOR_VERSION,
            family_id=family_id,
            family_key=contract.key,
            base_index=base_index,
            split=split_for_base_index(base_index),
            canonical_setup=representative,
            fingerprint=class_fingerprint(representative),
            content_fingerprint=content_fingerprint(representative),
            reflected_content_fingerprint=content_fingerprint(
                reflect_canonical(representative)
            ),
            master_seed=seed_context.master_seed,
            generation_seed=base_seed,
            accepted_attempt_index=attempt,
            accepted_attempt_seed=attempt_seed,
            generation_attempts=attempt + 1,
            trait_vector=compute_trait_vector(representative),
        )
        return BaseGenerationRecord(entry=entry, rejections=rejections)

    raise SetupLibraryError(
        f"{family_id}:{base_index:03d} exhausted {MAX_ATTEMPTS_PER_BASE} attempts under "
        f"the frozen contract (rejections: {rejections}); this is a BLOCKED "
        "condition, not a licence to weaken the family contract"
    )


def rebuild_base_setup(
    family_id: str,
    base_index: int,
    seed_context: LibrarySeedContext = DEFAULT_SEED_CONTEXT,
) -> BaseSetupEntry:
    """Rebuild one accepted entry in isolation, without any other base.

    The required isolated-regeneration API of the Phase 7 contract: it is a
    pure function of `(contract, library, master seed, family, index)` and is
    byte-identical to the entry the full library run materialized.
    """
    return generate_base_setup(family_id, base_index, seed_context).entry


assert (
    SETUP_LIBRARY_VERSION == DEFAULT_SEED_CONTEXT.library_version
    and SETUP_GENERATOR_CONTRACT_VERSION == DEFAULT_SEED_CONTEXT.contract_version
), "the default seed context must carry the frozen contract identifiers"
