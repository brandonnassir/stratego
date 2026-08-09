"""Exhaustive combat resolution.

Specification sources:

- `01_official_rules.md` sections 6, 7 (resolution and special cases)
- `02_project_ruleset.md` section 1
- `04_engine_validation_plan.md` section 7 (combat matrix gate)

The resolver is a pure function of the two true types. It performs no state
mutation, which lets the combat matrix be tested exhaustively in isolation.
"""

from .constants import (
    BOMB,
    FLAG,
    IMMOVABLE_TYPES,
    MARSHAL,
    MINER,
    PIECE_RANKS,
    PIECE_TYPE_NAMES,
    SPY,
)

ATTACKER_WINS = "attacker_survives"
DEFENDER_WINS = "defender_survives"
BOTH_REMOVED = "both_removed"

COMBAT_OUTCOMES = (ATTACKER_WINS, DEFENDER_WINS, BOTH_REMOVED)


class CombatError(ValueError):
    """Raised when combat is requested for a pair that cannot legally occur."""


def resolve_combat(attacker_type: int, defender_type: int) -> str:
    """Resolve one attack and return one of the three outcome labels.

    Post-combat occupancy follows `01_official_rules.md` section 6: the surviving
    piece holds the destination square, and on a tie the destination is emptied.
    That bookkeeping happens in `transition.py`; this function only decides who
    survives.
    """
    if attacker_type in IMMOVABLE_TYPES:
        # Flag and Bomb never move, so they can never be the attacker. Reaching
        # here means legal-move generation is broken.
        raise CombatError(
            f"{PIECE_TYPE_NAMES[attacker_type]} cannot attack; it is immovable"
        )

    # Flag loses to every attacker and its capture ends the game.
    if defender_type == FLAG:
        return ATTACKER_WINS

    # A Bomb destroys every attacker except a Miner.
    if defender_type == BOMB:
        return ATTACKER_WINS if attacker_type == MINER else DEFENDER_WINS

    # Spy attacking Marshal is the one rank inversion in the game. The reverse
    # direction (Marshal attacking Spy) is ordinary rank comparison and is
    # handled by the general case below.
    if attacker_type == SPY and defender_type == MARSHAL:
        return ATTACKER_WINS

    attacker_rank = PIECE_RANKS[attacker_type]
    defender_rank = PIECE_RANKS[defender_type]
    if attacker_rank is None or defender_rank is None:  # pragma: no cover - defensive
        raise CombatError(
            f"unhandled combat pair: {PIECE_TYPE_NAMES[attacker_type]} vs "
            f"{PIECE_TYPE_NAMES[defender_type]}"
        )

    if attacker_rank > defender_rank:
        return ATTACKER_WINS
    if attacker_rank < defender_rank:
        return DEFENDER_WINS
    return BOTH_REMOVED
