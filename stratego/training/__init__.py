"""Phase Three training infrastructure built on top of the frozen engine.

Nothing in this package may change reference behaviour. Every rule decision,
observation channel and event still comes from :mod:`stratego.engine`; the
modules here only decide *which* states exist, *when* they advance and *how*
they are transported, stored and rebuilt.

- batch semantics: :mod:`batch_simulation`
- cross-process transport: :mod:`shared_buffers`, :mod:`worker_pool`
- compact storage: :mod:`serialization`, :mod:`trajectory`
- historical positions: :mod:`reconstruction`

:mod:`representative_model`, :mod:`mps_benchmark`, :mod:`coordinator`,
:mod:`end_to_end_benchmark` and :mod:`phase6_pipeline_benchmark` are deliberately
*not* re-exported here. They require PyTorch, which is a benchmark dependency
(`requirements-training.txt`) rather than an engine dependency, and importing
this package must keep working with `requirements.txt` alone. Import those
modules directly.
"""

from .batch_simulation import (
    BatchIllegalActionError,
    BatchSimulationError,
    BatchSimulator,
    BatchStepResult,
    BatchTerminalStateError,
    EnvironmentSlot,
    SlotOutcome,
    UnknownEnvironmentError,
    derive_slot_seed,
    slot_game_id,
)
from .reconstruction import (
    RECONSTRUCTION_VERSION,
    DecisionDigest,
    ReconstructedDecision,
    compare_digests,
    digest_live_decision,
    digest_reconstructed_decision,
    iter_reconstructed_decisions,
    public_knowledge_view,
    reconstruct_decision,
    reconstruct_state,
    verify_decision,
)
from .trajectory import (
    DEFAULT_SNAPSHOT_INTERVAL,
    SUPPORTED_SNAPSHOT_INTERVALS,
    TRAJECTORY_VERSION,
    DecisionRecord,
    GameContext,
    GameRecord,
    GameTrajectoryBuilder,
    SnapshotEntry,
    TrajectoryError,
    builder_for_slot,
    collect_games,
    decode_game_record,
    decode_game_record_compressed,
    encode_game_record,
    encode_game_record_compressed,
    validate_decision_record,
    validate_game_record,
)

__all__ = [
    "DEFAULT_SNAPSHOT_INTERVAL",
    "RECONSTRUCTION_VERSION",
    "SUPPORTED_SNAPSHOT_INTERVALS",
    "TRAJECTORY_VERSION",
    "BatchIllegalActionError",
    "BatchSimulationError",
    "BatchSimulator",
    "BatchStepResult",
    "BatchTerminalStateError",
    "DecisionDigest",
    "DecisionRecord",
    "EnvironmentSlot",
    "GameContext",
    "GameRecord",
    "GameTrajectoryBuilder",
    "ReconstructedDecision",
    "SlotOutcome",
    "SnapshotEntry",
    "TrajectoryError",
    "UnknownEnvironmentError",
    "builder_for_slot",
    "collect_games",
    "compare_digests",
    "decode_game_record",
    "decode_game_record_compressed",
    "derive_slot_seed",
    "digest_live_decision",
    "digest_reconstructed_decision",
    "encode_game_record",
    "encode_game_record_compressed",
    "iter_reconstructed_decisions",
    "public_knowledge_view",
    "reconstruct_decision",
    "reconstruct_state",
    "slot_game_id",
    "validate_decision_record",
    "validate_game_record",
    "verify_decision",
]
