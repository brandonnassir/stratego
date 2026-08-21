"""Phase 14: the integrated configuration document and its digest.

Specification source: `02_AGENT_2_FINAL_TRAINING_INTEGRATION.md` section 17.

One digest over the whole runner
--------------------------------
Section 17 asks for a deterministic digest binding every input the Phase 14
runner has: the starting checkpoint, the objective, both learning rates, the
transition time, both opponent mixtures, the pool algorithm, the setup source,
the checkpoint cadences, the candidate pack, the selection rule, the storage
policy and the deadline semantics. This module builds that document from the
*live* modules — the contract's own digest, the population digest, the seed
contract digest, the frozen document's file hash — so the digest moves if any
of them moves, and Agent 4's launch manifest can bind one value instead of
fourteen.

Deliberately not in the digest
------------------------------
Anything that is an operational choice rather than an identity: the device,
loader worker count, games-in-flight, the storage *root* (as opposed to the
retention policy), the clock, log paths. Those change what a run costs, not
what it is, and folding them in would make two identical experiments on
different machines look like different experiments.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .phase14_checkpoint import PHASE14_CHECKPOINT_VERSION
from .phase14_collector import PHASE14_COLLECTOR_VERSION
from .phase14_contract import (
    ANCHOR_SHA256,
    ARCHIVE_CADENCE_SECONDS,
    CANDIDATE_CADENCE_SECONDS,
    CANDIDATE_HOURS,
    DEADLINE_RULE,
    DEADLINE_SECONDS,
    ENTROPY_COEFFICIENT,
    FROZEN_CONTRACT_RELATIVE_PATH,
    FULL_RAW_RETENTION,
    HOT_CHECKPOINT_RETAIN,
    HOT_CHECKPOINT_SECONDS,
    LATE_LEARNING_RATE,
    MAIN_LEARNING_RATE,
    NO_DELETION_RULE,
    PHASE14_CONTRACT_VERSION,
    PHASE14_POOL_VERSION,
    POOL_CATEGORY_WEIGHTS,
    POOL_SIZE,
    ROLLING_DELETION_RULE,
    SEARCH_PROHIBITION,
    SEGMENT_BUCKET_COUNTS,
    SELECTION_PACK_DIGEST,
    SELECTION_RULE,
    SETUP_SELECTOR_CONFIG_SHA256,
    SETUP_SOURCE_IDENTITY,
    STARTING_CHECKPOINT,
    STARTING_CHECKPOINT_SHA256,
    STARTING_MODEL_STATE_DIGEST,
    STORAGE_RESERVE_GIB,
    TRANSITION_SECONDS,
    contract_digest,
    file_sha256,
    inherited_phase9_values,
    repository_root,
)
from .phase14_pool import pool_semantics
from .phase14_runner import PHASE14_RUNNER_VERSION
from .phase14_schedule import population_digest
from .phase14_seed import seed_contract_digest
from .phase14_setup_source import PHASE14_SETUP_SOURCE_VERSION
from .phase14_telemetry import PHASE14_TELEMETRY_VERSION
from .phase14_trainer import PHASE14_TRAINER_VERSION

INTEGRATED_CONFIG_ARTIFACT = "phase13_integrated_training_config_v1"


def integrated_config_document() -> dict:
    """Every Phase 14 runner input that decides what the run *is*."""
    frozen = repository_root() / FROZEN_CONTRACT_RELATIVE_PATH
    return {
        "artifact": INTEGRATED_CONFIG_ARTIFACT,
        "phase": 13,
        "agent": 2,
        "purpose": (
            "the deterministic identity of the integrated Phase 14 runner; an input "
            "to Agent 4's immutable launch manifest. Building this document starts "
            "nothing."
        ),
        "frozen_contract": {
            "path": FROZEN_CONTRACT_RELATIVE_PATH,
            "sha256": file_sha256(frozen),
        },
        "implementation": {
            "contract_version": PHASE14_CONTRACT_VERSION,
            "contract_digest": contract_digest(),
            "seed_contract_digest": seed_contract_digest(),
            "population_digest": population_digest(),
            "modules": {
                "collector": PHASE14_COLLECTOR_VERSION,
                "trainer": PHASE14_TRAINER_VERSION,
                "checkpoint": PHASE14_CHECKPOINT_VERSION,
                "pool": PHASE14_POOL_VERSION,
                "setup_source": PHASE14_SETUP_SOURCE_VERSION,
                "telemetry": PHASE14_TELEMETRY_VERSION,
                "runner": PHASE14_RUNNER_VERSION,
            },
        },
        "starting_checkpoint": {
            "path": STARTING_CHECKPOINT,
            "sha256": STARTING_CHECKPOINT_SHA256,
            "model_state_digest": STARTING_MODEL_STATE_DIGEST,
        },
        "training_objective": inherited_phase9_values(),
        "entropy_coefficient": ENTROPY_COEFFICIENT,
        "learning_rate": {
            "main": MAIN_LEARNING_RATE,
            "late": LATE_LEARNING_RATE,
            "schedule": "constant within a segment",
        },
        "transition": {
            "transition_seconds": TRANSITION_SECONDS,
            "rule": "elapsed against the original run start; downtime counts",
        },
        "opponent_mixture": {
            segment: dict(counts) for segment, counts in SEGMENT_BUCKET_COUNTS.items()
        },
        "historical_pool": {
            **pool_semantics(),
            "anchors": dict(ANCHOR_SHA256),
            "size": POOL_SIZE,
            "weights": dict(POOL_CATEGORY_WEIGHTS),
        },
        "setup_source": {
            "identity": SETUP_SOURCE_IDENTITY,
            "selector_config_sha256": SETUP_SELECTOR_CONFIG_SHA256,
            "implementation": PHASE14_SETUP_SOURCE_VERSION,
        },
        "checkpoint_cadences": {
            "hot_seconds": HOT_CHECKPOINT_SECONDS,
            "hot_retain": HOT_CHECKPOINT_RETAIN,
            "archive_seconds": ARCHIVE_CADENCE_SECONDS,
            "candidate_seconds": CANDIDATE_CADENCE_SECONDS,
            "candidate_hours": list(CANDIDATE_HOURS),
        },
        "candidate_evaluation": {
            "pack_digest": SELECTION_PACK_DIGEST,
            "selection_rule": SELECTION_RULE,
        },
        "storage_policy": {
            "full_raw_retention": FULL_RAW_RETENTION,
            "reserve_gib": STORAGE_RESERVE_GIB,
            "contingency": ROLLING_DELETION_RULE,
            "no_deletion_rule": NO_DELETION_RULE,
        },
        "deadline_semantics": {
            "deadline_seconds": DEADLINE_SECONDS,
            "rule": DEADLINE_RULE,
            "restart": "a restart reuses the persisted window; it never creates a new one",
        },
        "search": SEARCH_PROHIBITION,
    }


def integrated_config_digest() -> str:
    """The deterministic digest Agent 4's launch manifest binds."""
    document = integrated_config_document()
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_integrated_config(path) -> dict:
    """Write the document with its own digest attached."""
    document = integrated_config_document()
    document["integrated_config_digest"] = integrated_config_digest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return document
