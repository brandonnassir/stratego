"""Phase 11B constants: corpus shape, strata, status markers, paths.

Specification sources:

- `00_PHASE_11B_OVERVIEW.md` ("Namespace", "Common Phase 11B Dataset",
  "Shared Metrics", "Shared Belief Interface")
- `01_AGENT_1_ATTACHED_BELIEF_HEAD.md` ("Part 0", "Evaluation")

Why the strata are imported and not re-declared
-----------------------------------------------
The four Phase 11B behaviour strata are four of the eight accepted Phase 11
strata, named by the *same* tokens and played by the *same* accepted
policies. Re-spelling `"scout_rush"` here would let a typo silently create a
fifth stratum whose games nobody could join to a Phase 11 slice, so the
tokens and the policy bindings are read from the accepted contract and
filtered. Phase 11B adds no behaviour of its own.

The rank vocabulary is imported for the same reason: `R_CE`'s denominator is
the accepted `remaining_count_belief_v1` arithmetic, and that arithmetic is
only reproducible against the accepted rank order and initial inventory.
"""

from __future__ import annotations

from ...training.phase11_contract import (
    IMMOVABLE_RANK_INDICES,
    MOVABLE_RANK_INDICES,
    Phase11ContractError,
    RANK_COUNT,
    RANK_INITIAL_COUNTS,
    RANK_NAMES,
    STRATUM_BINDINGS,
)
from ...training.phase11_seed import (
    STRATUM_PHASE9,
    STRATUM_SCOUT_RUSH,
    STRATUM_STRATEGIC,
    STRATUM_TACTICAL,
)

#: The sprint identity, carried by every Phase 11B artifact.
PHASE11B_VERSION = "phase11b_engineering_v1"

#: The common-corpus identity. Agents 2-5 reuse the bytes this names.
CORPUS_VERSION = "phase11b_common_corpus_v1"

#: The four status markers `00_PHASE_11B_OVERVIEW.md` requires on every
#: Phase 11B checkpoint and report. They are a constant, not a computed
#: value: no Phase 11B result can flip one.
PHASE11B_STATUS_MARKERS = {
    "phase": "phase11b",
    "status": "engineering_prototype",
    "phase11_fail_unchanged": True,
    "phase11_test_bank_used": False,
    "phase12_authorized_by_this_artifact": False,
}

#: The Phase 11 facts Phase 11B restates but never changes.
PHASE11_FACTS = {
    "phase11_final_classification": "FAIL",
    "phase11_test_bank_spent": True,
    "scientific_claim": "none",
}


class Phase11BError(Phase11ContractError):
    """A Phase 11B contract, corpus or training invariant was violated."""


# ---------------------------------------------------------------------------
# Behaviour strata
# ---------------------------------------------------------------------------

#: The four Phase 11B strata, in the frozen corpus order. Same tokens and
#: same accepted opponents as the Phase 11 strata of the same names.
CORPUS_STRATA = (
    STRATUM_PHASE9,
    STRATUM_STRATEGIC,
    STRATUM_TACTICAL,
    STRATUM_SCOUT_RUSH,
)

#: `stratum -> accepted evaluation-registry policy id`, None for the neural
#: seat. Filtered from the accepted Phase 11 bindings so the opponent of a
#: Phase 11B game is literally the opponent of a Phase 11 game.
STRATUM_POLICY_IDS = {
    entry["stratum"]: entry["opponent_policy_id"]
    for entry in STRATUM_BINDINGS
    if entry["stratum"] in CORPUS_STRATA
}
if tuple(sorted(STRATUM_POLICY_IDS)) != tuple(sorted(CORPUS_STRATA)):  # pragma: no cover
    raise Phase11BError("a Phase 11B stratum is not an accepted Phase 11 stratum")

#: Setup sources, in the frozen corpus order. Same tokens as Phase 11's.
SOURCE_P10D = "p10d"
SOURCE_NEUTRAL = "neutral"
CORPUS_SOURCES = (SOURCE_P10D, SOURCE_NEUTRAL)

#: Observer colours, in the frozen corpus order.
COLOR_RED = "red"
COLOR_BLUE = "blue"
CORPUS_COLORS = (COLOR_RED, COLOR_BLUE)

# ---------------------------------------------------------------------------
# Corpus shape
# ---------------------------------------------------------------------------

#: `split -> (games, games per stratum, decisions sampled per game, setup
#: library split)`. The two splits draw from **disjoint setup-library
#: splits** as well as from disjoint seed streams, so a development game
#: cannot share a base arrangement with a training game: non-overlap is a
#: property of the id space rather than of a post-hoc check.
CORPUS_SPLITS = {
    "train": {
        "games": 2048,
        "games_per_stratum": 512,
        "decisions_per_game": 16,
        "library_split": "train",
    },
    "dev": {
        "games": 512,
        "games_per_stratum": 128,
        "decisions_per_game": 4,
        "library_split": "validation",
    },
}

#: Every corpus game is one cell of (stratum x source x observer colour),
#: cell-major, so balance over all three is a property of the id space.
CELLS = tuple(
    (stratum, source, color)
    for stratum in CORPUS_STRATA
    for source in CORPUS_SOURCES
    for color in CORPUS_COLORS
)

for _name, _spec in CORPUS_SPLITS.items():
    if _spec["games"] % len(CELLS):  # pragma: no cover - arithmetic guard
        raise Phase11BError(f"{_name} games {_spec['games']} do not fill {len(CELLS)} cells")
    if _spec["games_per_stratum"] * len(CORPUS_STRATA) != _spec["games"]:  # pragma: no cover
        raise Phase11BError(f"{_name} stratum counts do not sum to its game count")

#: Relative roots. Bytes live under `data/`, weights under `checkpoints/`,
#: compact artifacts under `reports/` — the accepted repository convention.
CORPUS_ROOT = "data/phase11b/common_corpus_v1"
CHECKPOINT_ROOT = "checkpoints/phase11b"
REPORT_ROOT = "reports/phase11b"

#: The two halves of a stored split. The names are load-bearing: the
#: privileged directory holds true ranks and nothing that reaches a model
#: input, and `corpus.load_split` refuses to hand both to one caller
#: unless the caller asks for labels by name.
PUBLIC_DIRECTORY = "public"
PRIVILEGED_DIRECTORY = "privileged"

#: The accepted observation shape, restated so a stored array can be
#: checked without importing the engine.
OBSERVATION_SHAPE = (127, 10, 10)
NUM_SQUARES = 100

#: The C1 shared-encoder width — the per-piece belief feature every Agent 1
#: candidate consumes.
C1_FEATURE_WIDTH = 128

__all__ = [
    "C1_FEATURE_WIDTH",
    "CELLS",
    "CHECKPOINT_ROOT",
    "COLOR_BLUE",
    "COLOR_RED",
    "CORPUS_COLORS",
    "CORPUS_ROOT",
    "CORPUS_SOURCES",
    "CORPUS_SPLITS",
    "CORPUS_STRATA",
    "CORPUS_VERSION",
    "IMMOVABLE_RANK_INDICES",
    "MOVABLE_RANK_INDICES",
    "NUM_SQUARES",
    "OBSERVATION_SHAPE",
    "PHASE11B_STATUS_MARKERS",
    "PHASE11B_VERSION",
    "PHASE11_FACTS",
    "PRIVILEGED_DIRECTORY",
    "PUBLIC_DIRECTORY",
    "Phase11BError",
    "RANK_COUNT",
    "RANK_INITIAL_COUNTS",
    "RANK_NAMES",
    "REPORT_ROOT",
    "SOURCE_NEUTRAL",
    "SOURCE_P10D",
    "STRATUM_POLICY_IDS",
]
