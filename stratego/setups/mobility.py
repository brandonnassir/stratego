"""Initial-mobility library-quality check, delegated to the frozen engine.

Specification sources:

- `00_PHASE_7_SEQUENCE_AND_COMMON_CONTRACT.md` (curated initial-mobility
  quality rule)
- `01_AGENT_1_SETUP_CONTRACT_AND_TAXONOMY.md` (initial-mobility quality rule)

The rule
--------
`setup_library_v1` is a curated strategic training library, so every accepted
base setup and every generated descendant must have at least one legal move
available for its owner in the corresponding initial board geometry. This is a
library acceptance criterion, not an engine legality rule: the frozen engine
still accepts arbitrary legal setups, including ones that are terminal at
creation (`phase2_1_reference_1.2.0` handles those correctly), and nothing here
rerolls or redefines engine setups globally. A stranded candidate is simply
rejected from the library, and the rejection is counted by the caller.

Engine authority
----------------
The check constructs a real initial game through the frozen engine's
`create_game` and asks the engine's own `has_legal_action` — the exact
function the engine's mobility-termination rule consults. No competing
movement implementation exists in this module; `tests/setups/test_mobility.py`
asserts that statically.

Opponent independence
---------------------
A player's initial mobility does not depend on the opponent's arrangement. At
ply 0 the two armies occupy their own filled 4x10 setup blocks, so a player's
moves are advances into the empty central rows or Scout rays; a ray that
reaches an opposing piece is a legal attack regardless of the defender's
identity (attack legality depends only on ownership). The probe therefore
plays the same canonical arrangement mirrored for both sides, which keeps the
check self-contained.
"""

from ..engine.constants import BLUE, RED, TRAINING_RULES
from ..engine.legal_moves import has_legal_action
from ..engine.state import create_game
from .identity import orient_setup

#: game_id used by the probe. Purely cosmetic; recorded for log clarity.
_PROBE_GAME_ID = "phase7-mobility-probe"


def setup_has_initial_mobility(canonical: "list[int] | tuple[int, ...]") -> bool:
    """Whether the setup's owner has at least one legal move at game start.

    `canonical` is a canonical own-orientation 40-tuple. The verdict is the
    frozen engine's: the arrangement is oriented for red, the same arrangement
    is mirrored for blue, `create_game` builds the initial state (validating
    both setups), and `has_legal_action(state, RED)` answers for the owner.

    By the rank-flip symmetry of `orient_setup` and the lake geometry, the
    same canonical arrangement is mobile for red exactly when it is mobile for
    blue, so one probe answers for both colours. Reflection preserves the
    verdict too, because the lake-facing file set is symmetric under
    `f -> 9 - f`; the library therefore never needs to re-check a reflected
    descendant of an accepted arrangement, though auditors may.
    """
    state = create_game(
        orient_setup(canonical, RED),
        orient_setup(canonical, BLUE),
        rules=TRAINING_RULES,
        game_id=_PROBE_GAME_ID,
    )
    return has_legal_action(state, RED)
