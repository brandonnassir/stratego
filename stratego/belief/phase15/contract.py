"""Phase 15 Agent 1: the frozen identities, shapes and mixtures.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` sections 3-9.

What this module is for
-----------------------
Every number Phase 15 Agent 1 was *given* lives here once, so a reader can
check the implementation against the instruction by reading one file, and
so a later refactor cannot quietly move a target. Nothing here is derived
and nothing here is searched.

The two specialists
-------------------
```text
P18 -> B18      Phase 14 candidate hour_018, policy/value, immutable
P24 -> B24      Phase 14 candidate hour_024, policy/value, immutable
```

`P18` and `P24` are never trained, never re-saved and never rotated. `B18`
and `B24` are separate belief-only checkpoints that carry a copy of their
source's final encoder block, a copy of its encoder norm, a fresh belief
MLP and a calibration temperature — and no policy or value parameter at
all.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

#: Any change to a constant in this module is a new contract version.
PHASE15_CONTRACT_VERSION = "phase15_agent01_contract_v1"

#: The corpus identity. Distinct from `phase11b_common_corpus_v1`, whose
#: Blue setups are mis-oriented and which this phase may not reuse.
CORPUS_VERSION = "phase15_belief_corpus_v1"

#: The specialists, and the Phase 14 candidate hours they are bound to.
SPECIALIST_B18 = "b18"
SPECIALIST_B24 = "b24"
SPECIALISTS = (SPECIALIST_B18, SPECIALIST_B24)

SOURCE_P18 = "p18"
SOURCE_P24 = "p24"
POLICY_SOURCES = (SOURCE_P18, SOURCE_P24)

#: `p18`/`p24` -> the Phase 14 candidate hour each resolves from.
SOURCE_HOURS = {SOURCE_P18: 18, SOURCE_P24: 24}

#: `b18`/`b24` -> the policy source it is fine-tuned from.
SPECIALIST_SOURCE = {SPECIALIST_B18: SOURCE_P18, SPECIALIST_B24: SOURCE_P24}

# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

#: The frozen 127-channel observation, unchanged since Phase 2.
OBSERVATION_SHAPE = (127, 10, 10)
NUM_SQUARES = 100
RANK_COUNT = 12
C1_FEATURE_WIDTH = 128

#: The belief MLP of section 8: `128 -> 512 -> 512 -> 12`, GELU.
MLP_HIDDEN_WIDTHS = (512, 512)
MLP_ACTIVATION = "gelu"

# ---------------------------------------------------------------------------
# Splits and the position budget (section 5)
# ---------------------------------------------------------------------------

SPLIT_TRAIN = "train"
SPLIT_CALIBRATION = "calibration"
SPLIT_DEVELOPMENT = "development"
CORPUS_SPLITS = (SPLIT_TRAIN, SPLIT_CALIBRATION, SPLIT_DEVELOPMENT)

#: The initial engineering target, in eligible observer positions.
POSITION_TARGET = {
    SPLIT_TRAIN: 120_000,
    SPLIT_CALIBRATION: 15_000,
    SPLIT_DEVELOPMENT: 20_000,
}

#: The floor a throughput pilot may fall back to, and no further. A run
#: that lands between the floor and the target must preserve the pilot
#: evidence that justified it.
POSITION_FLOOR = {
    SPLIT_TRAIN: 80_000,
    SPLIT_CALIBRATION: 10_000,
    SPLIT_DEVELOPMENT: 10_000,
}

#: Setup-library split each corpus split draws its base setups from.
#: Training draws from `train`; calibration and development draw from
#: `validation`, so no library identity is shared across that boundary.
LIBRARY_SPLIT = {
    SPLIT_TRAIN: "train",
    SPLIT_CALIBRATION: "validation",
    SPLIT_DEVELOPMENT: "validation",
}

#: Section 6: "non-overlapping validation identities for calibration/
#: development". Sharing the `validation` split between them is not enough —
#: two games whose observer drew the same base setup reach the same opening
#: public state, so the two splits would share positions. The validation
#: population is therefore partitioned in half, per family, and each split
#: draws only from its own half.
PARTITION_CALIBRATION = "a"
PARTITION_DEVELOPMENT = "b"
LIBRARY_PARTITIONS = (PARTITION_CALIBRATION, PARTITION_DEVELOPMENT)
LIBRARY_PARTITION = {
    SPLIT_TRAIN: None,
    SPLIT_CALIBRATION: PARTITION_CALIBRATION,
    SPLIT_DEVELOPMENT: PARTITION_DEVELOPMENT,
}

#: The bounded number of evenly spaced eligible decisions taken from one
#: trajectory. A cap, not a quota: a short game contributes fewer.
DECISIONS_PER_GAME = 16

# ---------------------------------------------------------------------------
# The corpus mixture (section 6)
# ---------------------------------------------------------------------------

#: Observer source. Exactly balanced.
OBSERVER_MIXTURE = {SOURCE_P18: 0.50, SOURCE_P24: 0.50}

#: Opponent mixture *within each observer half*.
OPPONENT_P18 = "p18"
OPPONENT_P24 = "p24"
OPPONENT_STRATEGIC = "strategic_rule_based"
OPPONENT_TACTICAL = "tactical_rule_based"
OPPONENT_SCOUT_RUSH = "stress_scout_rush"
OPPONENT_MINER_RUSH = "stress_miner_rush"
OPPONENT_INFORMATION_MISER = "stress_information_miser"

OPPONENT_MIXTURE = {
    OPPONENT_P18: 0.25,
    OPPONENT_P24: 0.25,
    OPPONENT_STRATEGIC: 0.10,
    OPPONENT_TACTICAL: 0.10,
    OPPONENT_SCOUT_RUSH: 0.10,
    OPPONENT_MINER_RUSH: 0.10,
    OPPONENT_INFORMATION_MISER: 0.10,
}
OPPONENTS = tuple(OPPONENT_MIXTURE)

#: The two neural opponents are the same frozen policy objects the observer
#: halves use; the other five are accepted catalogue policies, by id.
NEURAL_OPPONENTS = (OPPONENT_P18, OPPONENT_P24)
RULE_OPPONENT_POLICY_IDS = {
    OPPONENT_STRATEGIC: "strategic_rule_based",
    OPPONENT_TACTICAL: "tactical_rule_based",
    OPPONENT_SCOUT_RUSH: "stress_scout_rush",
    OPPONENT_MINER_RUSH: "stress_miner_rush",
    OPPONENT_INFORMATION_MISER: "stress_information_miser",
}
OPPONENT_CLASS = {
    OPPONENT_P18: "neural",
    OPPONENT_P24: "neural",
    OPPONENT_STRATEGIC: "rule",
    OPPONENT_TACTICAL: "rule",
    OPPONENT_SCOUT_RUSH: "stress",
    OPPONENT_MINER_RUSH: "stress",
    OPPONENT_INFORMATION_MISER: "stress",
}

#: Observer colour. Balanced inside every major cell.
COLOR_RED = "red"
COLOR_BLUE = "blue"
CORPUS_COLORS = (COLOR_RED, COLOR_BLUE)

# ---------------------------------------------------------------------------
# The setup mixture (section 6)
# ---------------------------------------------------------------------------

SETUP_NEUTRAL = "neutral_v1"
SETUP_LEARNED = "phase14_learned"
SETUP_TARGETED = "targeted_family"
SETUP_SOURCES = (SETUP_NEUTRAL, SETUP_LEARNED, SETUP_TARGETED)

SETUP_MIXTURE = {
    SETUP_NEUTRAL: 0.35,
    SETUP_LEARNED: 0.45,
    SETUP_TARGETED: 0.20,
}

#: The targeted families, by their accepted setup-library keys. Named
#: exactly as section 6 names them; `corner_flag_fortress` and
#: `near_corner_flag_fortress` are the "or" pair and both are drawn.
TARGETED_FAMILY_KEYS = (
    "high_bomb_placement",
    "aggressive_high_rank_front",
    "distributed_bomb_defense",
    "corner_flag_fortress",
    "near_corner_flag_fortress",
    "scout_forward_information",
    "miner_forward",
    "irregular_high_entropy",
)

# ---------------------------------------------------------------------------
# The training recipe (section 9)
# ---------------------------------------------------------------------------

#: One declared recipe for both specialists. Not a sweep.
RECIPE = {
    "loss": "hidden_piece_cross_entropy",
    "optimizer": "adamw",
    "head_learning_rate": 1.0e-3,
    "final_block_learning_rate": 1.0e-4,
    "weight_decay": 1.0e-4,
    "schedule": "cosine",
    "batch_size": 256,
    "max_epochs": 12,
    "early_stop_patience": 3,
    "selection": "best_development_cross_entropy",
}

#: The game band boundaries used for the early/middle/late breakdown, in
#: plies. Declared here so every report cuts the corpus the same way.
GAME_BAND_EARLY_MAX_PLY = 40
GAME_BAND_MIDDLE_MAX_PLY = 120
GAME_BANDS = ("early", "middle", "late")

# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------

PUBLIC_DIRECTORY = "public"
PRIVILEGED_DIRECTORY = "privileged"

#: Where a report says this work sits, so no reader mistakes it for a claim
#: about final playing strength.
PHASE15_STATUS_MARKERS = {
    "phase": "phase_15",
    "agent": "agent_01",
    "status": "engineering_deliverable_not_a_strength_claim",
    "search_implemented": False,
}


class Phase15Error(RuntimeError):
    """Base class for every Phase 15 Agent 1 refusal."""


def specialist_of(source: str) -> str:
    """`p18` -> `b18`. The inverse of :data:`SPECIALIST_SOURCE`."""
    for specialist, bound in SPECIALIST_SOURCE.items():
        if bound == source:
            return specialist
    raise Phase15Error(f"unknown policy source {source!r}")


def game_band(total_moves: int) -> str:
    """The early/middle/late band of a decision, by ply."""
    ply = int(total_moves)
    if ply < 0:
        raise Phase15Error(f"ply must be non-negative, got {ply}")
    if ply <= GAME_BAND_EARLY_MAX_PLY:
        return "early"
    if ply <= GAME_BAND_MIDDLE_MAX_PLY:
        return "middle"
    return "late"


__all__ = [
    "CORPUS_COLORS",
    "CORPUS_SPLITS",
    "CORPUS_VERSION",
    "C1_FEATURE_WIDTH",
    "DECISIONS_PER_GAME",
    "GAME_BANDS",
    "LIBRARY_PARTITION",
    "LIBRARY_PARTITIONS",
    "LIBRARY_SPLIT",
    "MLP_ACTIVATION",
    "MLP_HIDDEN_WIDTHS",
    "NEURAL_OPPONENTS",
    "NUM_SQUARES",
    "OBSERVATION_SHAPE",
    "OBSERVER_MIXTURE",
    "OPPONENTS",
    "OPPONENT_CLASS",
    "OPPONENT_MIXTURE",
    "PHASE15_CONTRACT_VERSION",
    "PHASE15_STATUS_MARKERS",
    "POLICY_SOURCES",
    "POSITION_FLOOR",
    "POSITION_TARGET",
    "PRIVILEGED_DIRECTORY",
    "PUBLIC_DIRECTORY",
    "RANK_COUNT",
    "RECIPE",
    "RULE_OPPONENT_POLICY_IDS",
    "SETUP_MIXTURE",
    "SETUP_SOURCES",
    "SOURCE_HOURS",
    "SPECIALISTS",
    "SPECIALIST_SOURCE",
    "SPLIT_CALIBRATION",
    "SPLIT_DEVELOPMENT",
    "SPLIT_TRAIN",
    "TARGETED_FAMILY_KEYS",
    "Phase15Error",
    "game_band",
    "specialist_of",
]
