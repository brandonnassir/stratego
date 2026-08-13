"""The 16 primary setup-family contracts: `setup_family_v1`.

Specification sources:

- `00_PHASE_7_SEQUENCE_AND_COMMON_CONTRACT.md` (fixed Phase 7 family list,
  family-contract rule)
- `01_AGENT_1_SETUP_CONTRACT_AND_TAXONOMY.md` (family-contract requirements)

Every family is a measurable structural contract over the deterministic trait
vector (`setup_trait_vector_v1`), never "whatever generator branch produced
it": membership is decidable from the setup alone by any independent auditor.
A setup satisfies a family exactly when every required clause evaluates true
and every forbidden clause evaluates false on its trait vector.

Reflection invariance
---------------------
Family membership is a property of the reflection equivalence class. Every
clause therefore references only reflection-invariant trait fields (enforced
by `tests/setups/test_families.py` against the trait schema), so a setup
satisfies a family exactly when its reflection does.

Overlap
-------
Families are not mathematically disjoint. Every library setup declares exactly
one primary family, whose contract it must satisfy (hard requirement);
satisfying another family's clauses as a side effect is expected and is
reported by Agent 3 as an overlap/confusion matrix, not treated as a failure.

Freezing
--------
Once Agent 1 reports PASS these contracts are frozen. A semantic change to any
clause requires a new family-contract version identifier, never a silent
reinterpretation of `setup_family_v1`.
"""

from dataclasses import dataclass, field

from .identity import SetupLibraryError
from .traits import TRAIT_NAMES, compute_trait_vector

FAMILY_CONTRACT_VERSION = "setup_family_v1"

_VALID_OPS = ("==", "!=", ">=", "<=", ">", "<")


# ---------------------------------------------------------------------------
# Clause algebra
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One comparison against a named scalar trait, e.g. `flag_rank == 0`."""

    trait: str
    op: str
    value: int

    def __post_init__(self) -> None:
        if self.trait not in TRAIT_NAMES:
            raise SetupLibraryError(f"unknown trait in condition: {self.trait!r}")
        if self.op not in _VALID_OPS:
            raise SetupLibraryError(f"unknown comparison operator: {self.op!r}")

    def evaluate(self, traits: dict) -> bool:
        actual = traits[self.trait]
        if self.op == "==":
            return actual == self.value
        if self.op == "!=":
            return actual != self.value
        if self.op == ">=":
            return actual >= self.value
        if self.op == "<=":
            return actual <= self.value
        if self.op == ">":
            return actual > self.value
        return actual < self.value

    def referenced_traits(self) -> tuple[str, ...]:
        return (self.trait,)

    def to_dict(self) -> dict:
        return {"trait": self.trait, "op": self.op, "value": self.value}

    def describe(self) -> str:
        return f"{self.trait} {self.op} {self.value}"


@dataclass(frozen=True)
class AllOf:
    """A conjunction of conditions, used where one clause needs several terms."""

    conditions: tuple[Condition, ...]

    def evaluate(self, traits: dict) -> bool:
        return all(condition.evaluate(traits) for condition in self.conditions)

    def referenced_traits(self) -> tuple[str, ...]:
        return tuple(
            trait
            for condition in self.conditions
            for trait in condition.referenced_traits()
        )

    def to_dict(self) -> dict:
        return {"all_of": [condition.to_dict() for condition in self.conditions]}

    def describe(self) -> str:
        return " and ".join(condition.describe() for condition in self.conditions)


@dataclass(frozen=True)
class Clause:
    """A named, documented predicate over the trait vector."""

    name: str
    description: str
    expression: "Condition | AllOf"

    def evaluate(self, traits: dict) -> bool:
        return self.expression.evaluate(traits)

    def referenced_traits(self) -> tuple[str, ...]:
        return self.expression.referenced_traits()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "expression": self.expression.to_dict(),
            "formula": self.expression.describe(),
        }


# ---------------------------------------------------------------------------
# Family contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyContract:
    """One measurable primary-family definition."""

    family_id: str
    key: str
    display_name: str
    purpose: str
    required: tuple[Clause, ...]
    forbidden: tuple[Clause, ...] = ()
    #: Effective numeric envelopes implied by the clauses, as documentation:
    #: `trait -> [minimum, maximum]` inclusive.
    allowed_ranges: dict = field(default_factory=dict)
    #: Trait names an auditor should inspect first for this family.
    primary_diagnostics: tuple[str, ...] = ()
    #: Non-binding expectations, reported but never enforced.
    secondary_expectations: tuple[str, ...] = ()

    #: Every family shares the same reflection rule and perturbation rule; the
    #: strings restate them so each contract is self-describing in artifacts.
    reflection_invariance_rule: str = (
        "every clause references only reflection-invariant traits, so the "
        "family holds for a setup exactly when it holds for its reflection"
    )
    perturbation_invariants: str = (
        "a perturbed descendant must keep the Flag on its base cell, "
        "re-satisfy every required clause, avoid every forbidden clause, and "
        "obey the library-wide perturbation invariants in "
        "stratego.setups.contracts"
    )

    def evaluate(self, traits: dict) -> tuple[bool, list[str]]:
        """Evaluate the contract; returns `(satisfied, violated_clause_names)`.

        A required clause that is false and a forbidden clause that is true
        are both violations; forbidden violations are prefixed `forbidden:`.
        """
        violations = [
            clause.name for clause in self.required if not clause.evaluate(traits)
        ]
        violations.extend(
            f"forbidden:{clause.name}"
            for clause in self.forbidden
            if clause.evaluate(traits)
        )
        return (not violations, violations)

    def satisfied_by_setup(self, canonical: "list[int] | tuple[int, ...]") -> bool:
        return self.evaluate(compute_trait_vector(canonical))[0]

    def referenced_traits(self) -> tuple[str, ...]:
        seen: list[str] = []
        for clause in (*self.required, *self.forbidden):
            for trait in clause.referenced_traits():
                if trait not in seen:
                    seen.append(trait)
        return tuple(seen)

    def to_dict(self) -> dict:
        return {
            "family_id": self.family_id,
            "key": self.key,
            "display_name": self.display_name,
            "purpose": self.purpose,
            "required": [clause.to_dict() for clause in self.required],
            "forbidden": [clause.to_dict() for clause in self.forbidden],
            "allowed_ranges": {
                trait: list(bounds) for trait, bounds in self.allowed_ranges.items()
            },
            "primary_diagnostics": list(self.primary_diagnostics),
            "secondary_expectations": list(self.secondary_expectations),
            "reflection_invariance_rule": self.reflection_invariance_rule,
            "perturbation_invariants": self.perturbation_invariants,
        }


def _condition(trait: str, op: str, value: int) -> Condition:
    return Condition(trait=trait, op=op, value=value)


def _clause(name: str, description: str, trait: str, op: str, value: int) -> Clause:
    return Clause(name=name, description=description, expression=_condition(trait, op, value))


# The clause shared by families F03-F14: a curated non-irregular setup keeps
# its Flag in the back two ranks. F00-F02 pin the Flag to the back rank; F15
# deliberately frees it.
_FLAG_BACK_TWO = _clause(
    "flag_in_back_two_ranks",
    "the Flag sits in the back two ranks of the setup zone",
    "flag_rank",
    "<=",
    1,
)


FAMILY_CONTRACTS: tuple[FamilyContract, ...] = (
    FamilyContract(
        family_id="F00",
        key="corner_flag_fortress",
        display_name="Corner Flag fortress",
        purpose=(
            "the Flag sits in a board corner of the back rank, sealed behind "
            "a complete Bomb wall on both of its orthogonal neighbours"
        ),
        required=(
            _clause("flag_on_back_rank", "the Flag sits on the back rank", "flag_rank", "==", 0),
            _clause("flag_in_corner", "the Flag occupies a corner file", "flag_edge_distance", "==", 0),
            _clause(
                "corner_fully_sealed",
                "both orthogonal neighbours of the corner Flag are Bombs",
                "flag_orth_bomb_guards",
                "==",
                2,
            ),
        ),
        allowed_ranges={
            "flag_rank": (0, 0),
            "flag_edge_distance": (0, 0),
            "flag_orth_bomb_guards": (2, 2),
        },
        primary_diagnostics=("flag_rank", "flag_edge_distance", "flag_orth_bomb_guards"),
        secondary_expectations=(
            "remaining Bombs tend toward the Flag's corner and back ranks",
            "the diagonal cell behind the guard wall is often a third Bomb",
        ),
    ),
    FamilyContract(
        family_id="F01",
        key="near_corner_flag_fortress",
        display_name="Near-corner Flag fortress",
        purpose=(
            "the Flag sits on the back rank one or two files off a corner, "
            "guarded by at least two orthogonally adjacent Bombs"
        ),
        required=(
            _clause("flag_on_back_rank", "the Flag sits on the back rank", "flag_rank", "==", 0),
            _clause("flag_off_corner", "the Flag is at least one file off the corner", "flag_edge_distance", ">=", 1),
            _clause("flag_near_edge", "the Flag is within two files of the edge", "flag_edge_distance", "<=", 2),
            _clause("fortress_guards", "at least two orthogonal neighbours of the Flag are Bombs", "flag_orth_bomb_guards", ">=", 2),
        ),
        allowed_ranges={
            "flag_rank": (0, 0),
            "flag_edge_distance": (1, 2),
            "flag_orth_bomb_guards": (2, 3),
        },
        primary_diagnostics=("flag_rank", "flag_edge_distance", "flag_orth_bomb_guards"),
        secondary_expectations=("the corner cell beside the Flag often holds a Bomb or a sacrifice piece",),
    ),
    FamilyContract(
        family_id="F02",
        key="central_back_flag_fortress",
        display_name="Central/back-row Flag fortress",
        purpose=(
            "the Flag hides in the central files of the back rank, guarded by "
            "at least two orthogonally adjacent Bombs"
        ),
        required=(
            _clause("flag_on_back_rank", "the Flag sits on the back rank", "flag_rank", "==", 0),
            _clause("flag_central", "the Flag is at least three files from the nearer edge", "flag_edge_distance", ">=", 3),
            _clause("fortress_guards", "at least two orthogonal neighbours of the Flag are Bombs", "flag_orth_bomb_guards", ">=", 2),
        ),
        allowed_ranges={
            "flag_rank": (0, 0),
            "flag_edge_distance": (3, 4),
            "flag_orth_bomb_guards": (2, 3),
        },
        primary_diagnostics=("flag_rank", "flag_edge_distance", "flag_orth_bomb_guards"),
        secondary_expectations=("central fortresses often keep a spare Bomb within Chebyshev distance 2",),
    ),
    FamilyContract(
        family_id="F03",
        key="partially_bombed_flag",
        display_name="Partially bombed Flag",
        purpose=(
            "the Flag has exactly one orthogonal Bomb guard: real protection "
            "exists but the wall is deliberately incomplete"
        ),
        required=(
            _FLAG_BACK_TWO,
            _clause("single_orthogonal_guard", "exactly one orthogonal neighbour of the Flag is a Bomb", "flag_orth_bomb_guards", "==", 1),
        ),
        forbidden=(
            _clause(
                "dense_bomb_field_near_flag",
                "four or more Bombs within Chebyshev distance 2 of the Flag would make the defense heavy, not partial",
                "flag_zone_bomb_count_r2",
                ">=",
                4,
            ),
        ),
        allowed_ranges={
            "flag_rank": (0, 1),
            "flag_orth_bomb_guards": (1, 1),
            "flag_zone_bomb_count_r2": (1, 3),
        },
        primary_diagnostics=("flag_orth_bomb_guards", "flag_zone_bomb_count_r2", "flag_rank"),
        secondary_expectations=("the unguarded Flag sides are often covered by movable mid ranks",),
    ),
    FamilyContract(
        family_id="F04",
        key="lightly_defended_deceptive_flag",
        display_name="Lightly defended / deceptive Flag",
        purpose=(
            "no Bomb touches the Flag orthogonally or diagonally, so the Flag "
            "square looks like an ordinary piece rather than a fortress"
        ),
        required=(
            _FLAG_BACK_TWO,
            _clause("no_orthogonal_guards", "no orthogonal neighbour of the Flag is a Bomb", "flag_orth_bomb_guards", "==", 0),
            _clause("no_diagonal_guards", "no diagonal neighbour of the Flag is a Bomb", "flag_diag_bomb_guards", "==", 0),
        ),
        forbidden=(
            _clause(
                "local_bomb_presence",
                "three or more Bombs within Chebyshev distance 2 of the Flag would still advertise the location",
                "flag_zone_bomb_count_r2",
                ">=",
                3,
            ),
        ),
        allowed_ranges={
            "flag_rank": (0, 1),
            "flag_orth_bomb_guards": (0, 0),
            "flag_diag_bomb_guards": (0, 0),
            "flag_zone_bomb_count_r2": (0, 2),
        },
        primary_diagnostics=("flag_orth_bomb_guards", "flag_diag_bomb_guards", "flag_zone_bomb_count_r2"),
        secondary_expectations=("Bombs are typically spent elsewhere as lane blockers or decoys",),
    ),
    FamilyContract(
        family_id="F05",
        key="false_fortress_bomb_decoy",
        display_name="False fortress / Bomb decoy",
        purpose=(
            "a convincing Bomb pocket surrounds a movable decoy piece far from "
            "the Flag, while the Flag itself is at most lightly guarded"
        ),
        required=(
            _FLAG_BACK_TWO,
            _clause("flag_lightly_guarded", "at most one orthogonal neighbour of the Flag is a Bomb", "flag_orth_bomb_guards", "<=", 1),
            _clause(
                "decoy_pocket_exists",
                "some movable piece in the back two ranks, at Manhattan distance >= 4 from the Flag, has at least two orthogonal Bomb neighbours",
                "decoy_pocket_bombs",
                ">=",
                2,
            ),
        ),
        allowed_ranges={
            "flag_rank": (0, 1),
            "flag_orth_bomb_guards": (0, 1),
            "decoy_pocket_bombs": (2, 4),
        },
        primary_diagnostics=("decoy_pocket_bombs", "flag_orth_bomb_guards", "flag_rank"),
        secondary_expectations=("the decoy pocket often sits near the opposite corner from the Flag",),
    ),
    FamilyContract(
        family_id="F06",
        key="distributed_bomb_defense",
        display_name="Distributed Bomb defense",
        purpose=(
            "Bombs are spread across the board as independent lane blockers: "
            "no two Bombs touch and no concentrated Flag fortress exists"
        ),
        required=(
            _FLAG_BACK_TWO,
            _clause("bombs_spread_across_files", "Bombs occupy at least five distinct files", "bomb_distinct_files", ">=", 5),
            _clause("no_adjacent_bomb_pairs", "no two Bombs are orthogonally adjacent", "bomb_adjacent_pairs", "==", 0),
        ),
        forbidden=(
            _clause(
                "concentrated_flag_fortress",
                "two or more orthogonal Bomb guards on the Flag would concentrate the defense this family distributes",
                "flag_orth_bomb_guards",
                ">=",
                2,
            ),
        ),
        allowed_ranges={
            "flag_rank": (0, 1),
            "bomb_distinct_files": (5, 6),
            "bomb_adjacent_pairs": (0, 0),
            "flag_orth_bomb_guards": (0, 1),
        },
        primary_diagnostics=("bomb_distinct_files", "bomb_adjacent_pairs", "flag_orth_bomb_guards"),
        secondary_expectations=("mean pairwise Bomb Manhattan distance is high for this family",),
    ),
    FamilyContract(
        family_id="F07",
        key="high_bomb_placement",
        display_name="High Bomb placement",
        purpose=(
            "most Bombs are pushed into the front half of the setup zone as "
            "forward lane blockers and early-attack punishment"
        ),
        required=(
            _FLAG_BACK_TWO,
            _clause("bombs_pushed_forward", "at least four Bombs sit in the front two ranks", "bomb_front2_count", ">=", 4),
        ),
        allowed_ranges={
            "flag_rank": (0, 1),
            "bomb_front2_count": (4, 6),
        },
        primary_diagnostics=("bomb_front2_count", "bomb_front_rank_count", "flag_zone_bomb_count_r2"),
        secondary_expectations=("the Flag zone is often thinly bombed as a consequence",),
    ),
    FamilyContract(
        family_id="F08",
        key="aggressive_high_rank_front",
        display_name="Aggressive high-rank front",
        purpose=(
            "the heavy pieces, including both the Marshal and the General, "
            "start in the front half ready for immediate contact"
        ),
        required=(
            _FLAG_BACK_TWO,
            _clause("high_ranks_forward", "at least five of the seven rank>=7 pieces sit in the front two ranks", "high_front2_count", ">=", 5),
            _clause("marshal_forward", "the Marshal sits in the front two ranks", "marshal_rank", ">=", 2),
            _clause("general_forward", "the General sits in the front two ranks", "general_rank", ">=", 2),
        ),
        allowed_ranges={
            "flag_rank": (0, 1),
            "high_front2_count": (5, 7),
            "marshal_rank": (2, 3),
            "general_rank": (2, 3),
        },
        primary_diagnostics=("high_front2_count", "marshal_rank", "general_rank"),
        secondary_expectations=("front-rank mobility stays high because Bombs stay back",),
    ),
    FamilyContract(
        family_id="F09",
        key="conservative_high_rank_rear",
        display_name="Conservative high-rank rear",
        purpose=(
            "the heavy pieces, including both the Marshal and the General, "
            "are held in the back half as a late-game reserve"
        ),
        required=(
            _FLAG_BACK_TWO,
            _clause("high_ranks_rear", "at least five of the seven rank>=7 pieces sit in the back two ranks", "high_back2_count", ">=", 5),
            _clause("marshal_back", "the Marshal sits in the back two ranks", "marshal_rank", "<=", 1),
            _clause("general_back", "the General sits in the back two ranks", "general_rank", "<=", 1),
        ),
        allowed_ranges={
            "flag_rank": (0, 1),
            "high_back2_count": (5, 7),
            "marshal_rank": (0, 1),
            "general_rank": (0, 1),
        },
        primary_diagnostics=("high_back2_count", "marshal_rank", "general_rank"),
        secondary_expectations=("the front ranks are manned by expendable mid and low ranks",),
    ),
    FamilyContract(
        family_id="F10",
        key="scout_forward_information",
        display_name="Scout-forward information",
        purpose=(
            "Scouts mass in the front half, including a strong front-rank "
            "presence, to buy early information with cheap probes"
        ),
        required=(
            _FLAG_BACK_TWO,
            _clause("scouts_forward", "at least six of the eight Scouts sit in the front two ranks", "scout_front2_count", ">=", 6),
            _clause("front_rank_scout_presence", "at least three Scouts sit on the front rank", "scout_front_rank_count", ">=", 3),
        ),
        allowed_ranges={
            "flag_rank": (0, 1),
            "scout_front2_count": (6, 8),
            "scout_front_rank_count": (3, 8),
        },
        primary_diagnostics=("scout_front2_count", "scout_front_rank_count"),
        secondary_expectations=("initial mobility is very high because Scouts man open files",),
    ),
    FamilyContract(
        family_id="F11",
        key="scout_preservation",
        display_name="Scout-preservation",
        purpose=(
            "Scouts are held in the back half with none exposed on the front "
            "rank, preserving late-game information and chase power"
        ),
        required=(
            _FLAG_BACK_TWO,
            _clause("scouts_held_back", "at least five of the eight Scouts sit in the back two ranks", "scout_back2_count", ">=", 5),
            _clause("no_front_rank_scouts", "no Scout sits on the front rank", "scout_front_rank_count", "==", 0),
        ),
        allowed_ranges={
            "flag_rank": (0, 1),
            "scout_back2_count": (5, 8),
            "scout_front_rank_count": (0, 0),
        },
        primary_diagnostics=("scout_back2_count", "scout_front_rank_count"),
        secondary_expectations=("the front rank is manned by mid ranks and Miners instead",),
    ),
    FamilyContract(
        family_id="F12",
        key="miner_forward",
        display_name="Miner-forward",
        purpose=(
            "a majority of Miners start in the front half, ready to clear "
            "forward Bomb lanes early"
        ),
        required=(
            _FLAG_BACK_TWO,
            _clause("miners_forward", "at least three of the five Miners sit in the front two ranks", "miner_front2_count", ">=", 3),
        ),
        allowed_ranges={
            "flag_rank": (0, 1),
            "miner_front2_count": (3, 5),
        },
        primary_diagnostics=("miner_front2_count", "miner_front_rank_count"),
        secondary_expectations=("early Miner losses are accepted in exchange for tempo",),
    ),
    FamilyContract(
        family_id="F13",
        key="miner_preservation",
        display_name="Miner-preservation",
        purpose=(
            "Miners are protected in the back half with none exposed on the "
            "front rank, guaranteeing endgame Bomb clearance"
        ),
        required=(
            _FLAG_BACK_TWO,
            _clause("miners_held_back", "at least four of the five Miners sit in the back two ranks", "miner_back2_count", ">=", 4),
            _clause("no_front_rank_miners", "no Miner sits on the front rank", "miner_front_rank_count", "==", 0),
        ),
        allowed_ranges={
            "flag_rank": (0, 1),
            "miner_back2_count": (4, 5),
            "miner_front_rank_count": (0, 0),
        },
        primary_diagnostics=("miner_back2_count", "miner_front_rank_count"),
        secondary_expectations=("Scouts and mid ranks absorb the early exchanges instead",),
    ),
    FamilyContract(
        family_id="F14",
        key="balanced_conventional",
        display_name="Balanced conventional",
        purpose=(
            "a textbook tournament shape: back-rank guarded Flag, heavies out "
            "of the first line, forward Scouts, reserved Miners, mobile front"
        ),
        required=(
            _clause("flag_on_back_rank", "the Flag sits on the back rank", "flag_rank", "==", 0),
            _clause("fortress_guards", "at least two orthogonal neighbours of the Flag are Bombs", "flag_orth_bomb_guards", ">=", 2),
            _clause("marshal_not_front", "the Marshal stays off the front rank", "marshal_rank", "<=", 2),
            _clause("general_not_front", "the General stays off the front rank", "general_rank", "<=", 2),
            _clause("some_forward_scouts", "at least three Scouts sit in the front two ranks", "scout_front2_count", ">=", 3),
            _clause("limited_front_rank_bombs", "at most two Bombs sit on the front rank", "bomb_front_rank_count", "<=", 2),
            _clause("miners_reserved", "at least two Miners sit in the back two ranks", "miner_back2_count", ">=", 2),
            _clause("mobile_front_rank", "at least eight of the ten front-rank pieces are movable", "movable_front_rank_count", ">=", 8),
        ),
        allowed_ranges={
            "flag_rank": (0, 0),
            "flag_orth_bomb_guards": (2, 3),
            "marshal_rank": (0, 2),
            "general_rank": (0, 2),
            "scout_front2_count": (3, 8),
            "bomb_front_rank_count": (0, 2),
            "miner_back2_count": (2, 5),
            "movable_front_rank_count": (8, 10),
        },
        primary_diagnostics=(
            "flag_orth_bomb_guards",
            "marshal_rank",
            "scout_front2_count",
            "movable_front_rank_count",
        ),
        secondary_expectations=("this family most closely resembles the Phase 4 structured_v1 profile",),
    ),
    FamilyContract(
        family_id="F15",
        key="irregular_high_entropy",
        display_name="Deliberately irregular / high-entropy",
        purpose=(
            "deliberately unconventional structure drawn from a high-entropy "
            "distribution, so the move learner also sees setups that break "
            "textbook assumptions"
        ),
        required=(
            _clause(
                "unconventional_structure",
                "at least two of the eight fixed unconventional-structure features hold (see setup_trait_vector_v1 UNCONVENTIONAL_FEATURES)",
                "unconventional_feature_count",
                ">=",
                2,
            ),
        ),
        forbidden=(
            Clause(
                name="conventional_fortress_signature",
                description=(
                    "a back-rank Flag with a two-Bomb orthogonal guard wall is "
                    "the conventional signature this family must not carry"
                ),
                expression=AllOf(
                    (
                        _condition("flag_rank", "==", 0),
                        _condition("flag_orth_bomb_guards", ">=", 2),
                    )
                ),
            ),
        ),
        allowed_ranges={
            "flag_rank": (0, 3),
            "unconventional_feature_count": (2, 8),
        },
        primary_diagnostics=("unconventional_feature_count", "flag_rank", "flag_orth_bomb_guards"),
        secondary_expectations=(
            "per-square entropy for this family is the highest in the library",
            "the Flag may legally appear on any rank, including the front",
        ),
    ),
)

FAMILY_IDS = tuple(contract.family_id for contract in FAMILY_CONTRACTS)
FAMILY_KEYS = tuple(contract.key for contract in FAMILY_CONTRACTS)
FAMILY_BY_ID = {contract.family_id: contract for contract in FAMILY_CONTRACTS}
FAMILY_BY_KEY = {contract.key: contract for contract in FAMILY_CONTRACTS}

assert len(FAMILY_CONTRACTS) == 16
assert FAMILY_IDS == tuple(f"F{index:02d}" for index in range(16))


def family_contract(family_id: str) -> FamilyContract:
    """Look up a family contract by `F00`-style identifier."""
    try:
        return FAMILY_BY_ID[family_id]
    except KeyError as error:
        raise SetupLibraryError(f"unknown family id: {family_id!r}") from error


def evaluate_family(
    family_id: str, canonical: "list[int] | tuple[int, ...]"
) -> tuple[bool, list[str]]:
    """Evaluate one family contract against a canonical arrangement."""
    return family_contract(family_id).evaluate(compute_trait_vector(canonical))


def families_document() -> dict:
    """The machine-readable 16-family table for the Agent 1 contract artifact."""
    return {
        "family_contract_version": FAMILY_CONTRACT_VERSION,
        "family_count": len(FAMILY_CONTRACTS),
        "families": [contract.to_dict() for contract in FAMILY_CONTRACTS],
    }
