"""Phase 16 Agent 3: training loop v2.

Additive namespace. Nothing here edits, wraps in place, or overwrites an
accepted Phase 2-15 module: the objective, the KL controller, the engine, the
trajectory builder and the setup selector are all imported unmodified, and
what differs -- the schedule and the data distribution -- is rebuilt here.
"""

from .contract import (  # noqa: F401
    ARM_A,
    ARM_B,
    ARM_C,
    SHOOTOUT_ARMS,
    ArmConfig,
    PHASE16_TRAINING_VERSION,
    Phase16TrainingError,
    arm,
    contract_digest,
    contract_document,
)

__all__ = [
    "ARM_A",
    "ARM_B",
    "ARM_C",
    "PHASE16_TRAINING_VERSION",
    "SHOOTOUT_ARMS",
    "ArmConfig",
    "Phase16TrainingError",
    "arm",
    "contract_digest",
    "contract_document",
]
