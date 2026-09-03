"""Phase 18 Stage 6B: the frozen implementation contract of the matched
two-lineage joint-training pilot (Gate G3).

Design source: `reports/phase18/g3_design/phase18_g3_stage6a_joint_design_v2.md`
at commit `7a37cde59f3d94dec3f2fbb66c47accc618f6001` (approved 2026-09-02), and
the reviewer's seven Stage 6B decisions recorded with it.

The two lineages
----------------
```text
candidate   C1 (policy / value / belief) and the setup model start from the
            recorded fresh initialisations and train together: every period
            ends with K supervised C1 updates, one setup update and one EMA
            update; the next pool is sampled by the updated raw setup model
control     C1 starts from the SAME initialisation and receives the SAME K
            supervised C1 updates per period of the same type; the setup
            model is FROZEN at the recorded initial version: no setup update,
            no EMA update, the setup optimizer never steps; every pool is
            sampled by that frozen model under the same pool seeds
matched     both lineages derive every collector, pool, pairing, cell and
            C1 live-draw seed from (namespace, seed_index, period, slot,
            ordinal); the lineage id enters no seed
```

Everything numeric here is the reviewer's frozen default or the design's
derived constant; nothing is tuned. The one deliberate split of the design's
"retention" word: the live-example retention window (32 periods, a reviewer
default) and the setup buffer's storage duration (21 periods, the S21 rule
`ceil(4000 / 202) + 1` that keeps an outcome attributable for the whole
absolute move limit) are different quantities and are carried separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ...engine.constants import EVALUATION_RULES, TRAINING_RULES
from ...setups.contracts import (
    FAMILY_COUNT,
    TEST_PER_FAMILY,
    TRAIN_PER_FAMILY,
    VALIDATION_PER_FAMILY,
)
from ..warmstart_contract import CORPUS_RULES
from ..warmstart_trainer import WarmstartTrainConfig
from .setup_contract import (
    SETUP_POOL_SIZE,
    Phase18SetupError,
    SetupTrainingConfig,
    json_document_digest,
    model_seed,
    stream_seed,
)

# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------

G3_HARNESS_VERSION = "phase18_g3_two_lineage_harness_v1"
G3_BUNDLE_VERSION = "phase18_g3_joint_bundle_v1"
G3_LIVE_STORE_VERSION = "phase18_g3_live_record_store_v1"
G3_EVALUATION_VERSION = "phase18_g3_bundle_evaluation_v1"
G3_COLLECTION_POLICY_VERSION = "phase18_g3_teacher_schedule_collector_v1"

#: The approved Stage 6A design commit every Stage 6B artifact names.
G3_DESIGN_COMMIT = "7a37cde59f3d94dec3f2fbb66c47accc618f6001"

LINEAGE_CANDIDATE = "candidate"
LINEAGE_CONTROL = "control"
LINEAGES = (LINEAGE_CANDIDATE, LINEAGE_CONTROL)


class Phase18G3Error(Phase18SetupError):
    """Any refusal raised by the G3 harness. Always fatal."""


class Phase18G3LineageError(Phase18G3Error):
    """A component of one lineage was about to be paired with another's."""


class Phase18G3AccountingError(Phase18G3Error):
    """A period's accounting identity did not hold."""


# ---------------------------------------------------------------------------
# Frozen defaults (reviewer decisions 2 and 5; design sections 2.3 and 3.2)
# ---------------------------------------------------------------------------

#: K supervised C1 updates per period, per lineage.
C1_UPDATES_PER_PERIOD = 64
#: The Phase 8 batch is 256; the canonical:live mixture is 1:1.
C1_BATCH_SIZE = 256
CANONICAL_PER_BATCH = 128
LIVE_PER_BATCH = 128
#: Live examples enter the C1 training split for this many periods.
LIVE_RETENTION_PERIODS = 32
#: Bounded pilot length and the joint-bundle cadence.
PILOT_PERIODS = 256
BUNDLE_CADENCE_PERIODS = 32
#: The published period: every slot advances 2 x train_every_per_player plies.
PLIES_PER_PERIOD = 202
#: Provisional collector width (design 3.2; recorded in the launch record).
COLLECTOR_SLOTS = 2560
#: One immutable pool per period (S20), 512 per lane.
POOL_SIZE = SETUP_POOL_SIZE
#: S21: the setup buffer keeps a row for the whole absolute move limit.
BUFFER_STORAGE_PERIODS = math.ceil(TRAINING_RULES.absolute_move_limit / PLIES_PER_PERIOD) + 1
assert BUFFER_STORAGE_PERIODS == 21
#: The frozen 100-cell teacher schedule is taken in cyclic order.
SCHEDULE_CELLS = 100
#: Trajectory snapshot cadence of the accepted corpus.
SNAPSHOT_INTERVAL = 32

#: The collector plays every game under the Phase 8 training rules
#: (battleless 100); play evaluation runs under EVALUATION_RULES (P18-A001).
COLLECTOR_RULES = CORPUS_RULES
assert COLLECTOR_RULES == TRAINING_RULES
PLAY_EVALUATION_RULES = EVALUATION_RULES

# ---------------------------------------------------------------------------
# Evaluation cases (design section 4.1; reviewer decisions 3 and 5)
# ---------------------------------------------------------------------------

#: The eight frozen handcrafted code opponents, by policy id and in schedule
#: order. `random_legal` and `stress_draw_seeker` are deliberately absent.
HANDCRAFTED_OPPONENTS = (
    "basic_heuristic",
    "strategic_rule_based",
    "tactical_rule_based",
    "stress_scout_rush",
    "stress_miner_rush",
    "stress_berserker",
    "stress_information_miser",
    "stress_chaos",
)
#: Opponent formations: library validation bases 400..409, ten per family.
EVALUATION_BASE_INDICES = tuple(range(TRAIN_PER_FAMILY, TRAIN_PER_FAMILY + 10))
#: Reserved for a later independent confirmation; never opened by this harness.
RESERVED_BASE_INDICES = tuple(range(TRAIN_PER_FAMILY + 10, TRAIN_PER_FAMILY + VALIDATION_PER_FAMILY))
#: The sealed test bases; never opened by any Phase 18 training-side code.
TEST_BASE_INDICES = tuple(
    range(TRAIN_PER_FAMILY + VALIDATION_PER_FAMILY, TRAIN_PER_FAMILY + VALIDATION_PER_FAMILY + TEST_PER_FAMILY)
)
assert RESERVED_BASE_INDICES == tuple(range(410, 450))
EVALUATION_BASES_PER_FAMILY = len(EVALUATION_BASE_INDICES)
EVALUATION_BASES = FAMILY_COUNT * EVALUATION_BASES_PER_FAMILY
EVALUATION_COLOURS = 2
EVALUATION_CASES_PER_BASE = len(HANDCRAFTED_OPPONENTS) * EVALUATION_COLOURS
EVALUATION_CASES_PER_ARM = EVALUATION_BASES * EVALUATION_CASES_PER_BASE
assert (EVALUATION_BASES, EVALUATION_CASES_PER_BASE, EVALUATION_CASES_PER_ARM) == (160, 16, 2560)

#: The frozen practical-margin rule of the primary contrast.
PRIMARY_MARGIN = 0.05
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95


def assert_base_index_is_evaluable(base_index: int) -> int:
    """Refuse every base outside 400..409: the reserved and test bases stay closed."""
    index = int(base_index)
    if index in RESERVED_BASE_INDICES:
        raise Phase18G3Error(
            f"base index {index} is reserved (410..449) for a later independent "
            "confirmation and may not be opened by the G3 pilot"
        )
    if index in TEST_BASE_INDICES:
        raise Phase18G3Error(f"base index {index} is a sealed test base (450..499)")
    if index not in EVALUATION_BASE_INDICES:
        raise Phase18G3Error(
            f"base index {index} is not an evaluation base; the pilot evaluates "
            f"exactly {EVALUATION_BASE_INDICES[0]}..{EVALUATION_BASE_INDICES[-1]}"
        )
    return index


# ---------------------------------------------------------------------------
# Seeds: every stream derives through `derive_stream_seed`
# ---------------------------------------------------------------------------


def collector_policy_seed(namespace: str, seed_index: int, colour: str, period: int, slot: int, ordinal: int) -> int:
    """The match-level seed of one side's teacher in one collector game."""
    if colour not in ("red", "blue"):
        raise Phase18G3Error(f"unknown colour {colour!r}")
    return stream_seed(namespace, "collector_policy", int(seed_index), colour, int(period), int(slot), int(ordinal))


def pairing_seed(namespace: str, seed_index: int, period: int) -> int:
    """The per-period permutation that pairs blue pool rows with red ones."""
    return stream_seed(namespace, "pool_pairing", int(seed_index), int(period))


def live_draw_seed(namespace: str, seed_index: int, period: int, update: int) -> int:
    """The seed of one C1 update's draw from the retained live examples."""
    return stream_seed(namespace, "c1_live_draw", int(seed_index), int(period), int(update))


def evaluation_bootstrap_seed(namespace: str) -> int:
    return stream_seed(namespace, "evaluation_bootstrap")


def evaluation_schedule_seed(namespace: str) -> int:
    return stream_seed(namespace, "evaluation_schedule_root")


def setup_model_init_seed(namespace: str, seed_index: int) -> int:
    """The one fresh setup initialisation both lineages start from."""
    return model_seed(namespace, seed_index)


# ---------------------------------------------------------------------------
# The pilot configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PilotConfig:
    """One lineage's complete configuration.

    Every field except `lineage` must agree between the two lineages of a
    pilot; :func:`matched_document` is what the matching check compares.
    """

    run_id: str
    namespace: str
    seed_index: int
    lineage: str
    c1_train_config: WarmstartTrainConfig
    periods: int = PILOT_PERIODS
    c1_updates_per_period: int = C1_UPDATES_PER_PERIOD
    canonical_per_batch: int = CANONICAL_PER_BATCH
    live_per_batch: int = LIVE_PER_BATCH
    live_retention_periods: int = LIVE_RETENTION_PERIODS
    buffer_storage_periods: int = BUFFER_STORAGE_PERIODS
    bundle_cadence_periods: int = BUNDLE_CADENCE_PERIODS
    plies_per_period: int = PLIES_PER_PERIOD
    slots: int = COLLECTOR_SLOTS
    pool_size: int = POOL_SIZE
    snapshot_interval: int = SNAPSHOT_INTERVAL
    setup_device: str = "cpu"
    threads: int = 4
    loader_workers: int = 1
    loader_prefetch: int = 2
    record_cache_size: int = 512
    #: For the tiny smoke configurations only; production keeps every cell.
    schedule_cells: int = SCHEDULE_CELLS
    #: Smoke configurations may name the exact cells (cyclic order); None means
    #: the first `schedule_cells` cells of the frozen schedule.
    cell_indices: "tuple | None" = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id or not self.namespace:
            raise Phase18G3Error("run_id and namespace must be non-empty")
        if self.lineage not in LINEAGES:
            raise Phase18G3Error(f"lineage must be one of {LINEAGES}, got {self.lineage!r}")
        if int(self.seed_index) < 1:
            raise Phase18G3Error("seed_index is one-based")
        if self.canonical_per_batch + self.live_per_batch != self.c1_train_config.batch_size:
            raise Phase18G3Error(
                f"canonical {self.canonical_per_batch} + live {self.live_per_batch} must equal "
                f"the C1 batch size {self.c1_train_config.batch_size}"
            )
        if self.canonical_per_batch < 1 or self.live_per_batch < 0:
            raise Phase18G3Error("the canonical half must be positive and the live half non-negative")
        for name in (
            "periods",
            "c1_updates_per_period",
            "live_retention_periods",
            "buffer_storage_periods",
            "bundle_cadence_periods",
            "plies_per_period",
            "slots",
            "pool_size",
            "snapshot_interval",
            "threads",
            "loader_workers",
            "loader_prefetch",
            "record_cache_size",
            "schedule_cells",
        ):
            if int(getattr(self, name)) < 1:
                raise Phase18G3Error(f"{name} must be positive")
        if self.pool_size % 2:
            raise Phase18G3Error("pool_size must be even: one lane per colour")
        if not 1 <= self.schedule_cells <= SCHEDULE_CELLS:
            raise Phase18G3Error(f"schedule_cells must be in 1..{SCHEDULE_CELLS}")
        if self.cell_indices is not None:
            cells = tuple(int(c) for c in self.cell_indices)
            if not cells or len(set(cells)) != len(cells) or any(not 0 <= c < SCHEDULE_CELLS for c in cells):
                raise Phase18G3Error("cell_indices must be distinct indices in 0..99")
            if len(cells) != self.schedule_cells:
                raise Phase18G3Error("schedule_cells must equal the number of named cell_indices")
            object.__setattr__(self, "cell_indices", cells)
        if self.buffer_storage_periods < math.ceil(COLLECTOR_RULES.absolute_move_limit / self.plies_per_period) + 1:
            raise Phase18G3Error(
                "buffer_storage_periods is shorter than the absolute move limit in periods; "
                "an outcome could fail attribution for age (S21)"
            )

    # -- derived -------------------------------------------------------------

    @property
    def setup_updates_enabled(self) -> bool:
        """The lineage switch. True only for the candidate."""
        return self.lineage == LINEAGE_CANDIDATE

    @property
    def pool_per_lane(self) -> int:
        return self.pool_size // 2

    def setup_config(self) -> SetupTrainingConfig:
        return SetupTrainingConfig(run_id=self.run_id, device=self.setup_device, pool_size=self.pool_size)

    def setup_init_seed(self) -> int:
        return setup_model_init_seed(self.namespace, self.seed_index)

    def is_production_scale(self) -> bool:
        return (
            self.periods == PILOT_PERIODS
            and self.c1_updates_per_period == C1_UPDATES_PER_PERIOD
            and self.slots == COLLECTOR_SLOTS
            and self.pool_size == POOL_SIZE
            and self.plies_per_period == PLIES_PER_PERIOD
            and self.schedule_cells == SCHEDULE_CELLS
            and self.cell_indices is None
        )

    def with_lineage(self, lineage: str) -> "PilotConfig":
        from dataclasses import replace

        return replace(self, lineage=lineage)

    # -- documents ------------------------------------------------------------

    def matched_document(self) -> dict:
        """Every field the two lineages must share; the lineage id is absent."""
        return {
            "harness_version": G3_HARNESS_VERSION,
            "design_commit": G3_DESIGN_COMMIT,
            "run_id": self.run_id,
            "namespace": self.namespace,
            "seed_index": int(self.seed_index),
            "c1_train_config_digest": self.c1_train_config.digest(),
            "c1_train_config": self.c1_train_config.identity(),
            "setup_config_digest": self.setup_config().config_digest(),
            "setup_init_seed": self.setup_init_seed(),
            "periods": int(self.periods),
            "c1_updates_per_period": int(self.c1_updates_per_period),
            "c1_batch_size": int(self.c1_train_config.batch_size),
            "canonical_per_batch": int(self.canonical_per_batch),
            "live_per_batch": int(self.live_per_batch),
            "live_retention_periods": int(self.live_retention_periods),
            "buffer_storage_periods": int(self.buffer_storage_periods),
            "bundle_cadence_periods": int(self.bundle_cadence_periods),
            "plies_per_period": int(self.plies_per_period),
            "slots": int(self.slots),
            "pool_size": int(self.pool_size),
            "schedule_cells": int(self.schedule_cells),
            "cell_indices": list(self.cell_indices) if self.cell_indices is not None else None,
            "snapshot_interval": int(self.snapshot_interval),
            "collector_rules": COLLECTOR_RULES.rules_version + f"|battleless={COLLECTOR_RULES.battleless_move_limit}",
            "collection_policy_version": G3_COLLECTION_POLICY_VERSION,
            "setup_device": self.setup_device,
            "threads": int(self.threads),
            "seed_function": "stratego.setups.identity.derive_stream_seed",
            "extra": dict(self.extra),
        }

    def document(self) -> dict:
        return {
            **self.matched_document(),
            "lineage": self.lineage,
            "setup_updates_enabled": self.setup_updates_enabled,
            "loader": {
                "workers": int(self.loader_workers),
                "prefetch": int(self.loader_prefetch),
                "record_cache_size": int(self.record_cache_size),
            },
        }

    def matched_digest(self) -> str:
        return json_document_digest(self.matched_document())

    def config_digest(self) -> str:
        return json_document_digest(self.document())


__all__ = [
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_REPLICATES",
    "BUFFER_STORAGE_PERIODS",
    "BUNDLE_CADENCE_PERIODS",
    "C1_BATCH_SIZE",
    "C1_UPDATES_PER_PERIOD",
    "CANONICAL_PER_BATCH",
    "COLLECTOR_RULES",
    "COLLECTOR_SLOTS",
    "EVALUATION_BASES",
    "EVALUATION_BASES_PER_FAMILY",
    "EVALUATION_BASE_INDICES",
    "EVALUATION_CASES_PER_ARM",
    "EVALUATION_CASES_PER_BASE",
    "EVALUATION_COLOURS",
    "G3_BUNDLE_VERSION",
    "G3_COLLECTION_POLICY_VERSION",
    "G3_DESIGN_COMMIT",
    "G3_EVALUATION_VERSION",
    "G3_HARNESS_VERSION",
    "G3_LIVE_STORE_VERSION",
    "HANDCRAFTED_OPPONENTS",
    "LINEAGES",
    "LINEAGE_CANDIDATE",
    "LINEAGE_CONTROL",
    "LIVE_PER_BATCH",
    "LIVE_RETENTION_PERIODS",
    "PILOT_PERIODS",
    "PLAY_EVALUATION_RULES",
    "PLIES_PER_PERIOD",
    "POOL_SIZE",
    "PRIMARY_MARGIN",
    "Phase18G3AccountingError",
    "Phase18G3Error",
    "Phase18G3LineageError",
    "PilotConfig",
    "RESERVED_BASE_INDICES",
    "SCHEDULE_CELLS",
    "SNAPSHOT_INTERVAL",
    "TEST_BASE_INDICES",
    "assert_base_index_is_evaluable",
    "collector_policy_seed",
    "evaluation_bootstrap_seed",
    "evaluation_schedule_seed",
    "live_draw_seed",
    "pairing_seed",
    "setup_model_init_seed",
]
