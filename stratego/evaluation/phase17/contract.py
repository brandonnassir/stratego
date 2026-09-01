"""Phase 17 Agent 5: identities, refusals and seed domains for external evaluation.

Specification sources: common contract sections 10-11, Agent 1's frozen
composite-pack semantics, Agent 5 instruction sections 3 and 5.

Why a separate contract module
------------------------------
The remote MacBook runs a *subset* of this repository. Everything the two
machines must agree on -- pack identity, lane names, seed derivation, digest
convention, the tolerance a float comparison is allowed to use -- lives here,
so "the two machines agree" is a property of one small file that is itself
digest-bound rather than of a matching pair of assumptions.

Packaging may change; semantics may not
---------------------------------------
Agent 5 instruction section 5 permits portability work to change *packaging*
only. Concretely: the composite pack ships the 120 boards' setup **bytes**
instead of shipping the 13 MB accepted setup library plus the Phase 15 draw
machinery. That is legitimate precisely because
:func:`stratego.evaluation.phase16.benchmark.materialize_benchmark` proves,
on the machine that owns the library, that the stored bytes are byte-identical
to what the library redraws. The proof is recorded in the pack. Anything that
changed a case, a seed, a rule, the inference math or the scoring would need a
new pack version instead.
"""

from __future__ import annotations

import hashlib
import json

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

#: The composite pack Agent 1 froze the semantics of and Agent 5 materializes.
COMPOSITE_PACK_ID = "phase17_composite_benchmark_v1"

#: The two lanes, in report order.
LANE_MOVE_ONLY = "move_only"
LANE_JOINT = "joint_move_setup"
LANES = (LANE_MOVE_ONLY, LANE_JOINT)

#: The joint lane's own identity. It is ALSO the fixed `run_id` the joint
#: lane's setup seeds are derived under -- see `joint_setup_seeds`.
JOINT_LANE_ID = "phase17_joint_setup_lane_v1"

#: The accepted move-only base and its immutable digest (contract section 11).
MOVE_ONLY_BASE = "phase16_benchmark_v1"
MOVE_ONLY_BASE_DIGEST = (
    "ebd130198ea500248b32df990bee876583a10d53546f38a6346ec522407320c2"
)
MOVE_ONLY_BASE_FILE = "data/phase16/phase16_benchmark_v1.json"

#: The evaluator's own version. A change to cases, rules, seeds, inference
#: math or scoring bumps this AND the pack; packaging alone bumps neither.
EVALUATOR_VERSION = "phase17_remote_evaluator_v1"

#: The bundle Agent 4 froze at c2c0365. The evaluator refuses anything else.
EXPECTED_EXPORT_SCHEMA = "phase17_paired_export_v1"

#: Receipt and transport schemas.
RECEIPT_SCHEMA_VERSION = "phase17_eval_receipt_v1"
TRANSPORT_PROTOCOL_VERSION = "phase17_transport_v1"

#: Both machines evaluate on CPU with a pinned thread count, deliberately.
#: CPU float32 is bit-exact run to run; MPS is not (Agent 4 measured a
#: 9.83e-07 spread between two *identical* MPS runs). Determinism is what
#: makes a cross-machine identity claim checkable at all.
EVALUATION_DEVICE = "cpu"
EVALUATION_DTYPE = "float32"
WORKER_TORCH_THREADS = 1

#: The candidate move seat decides greedily, like the accepted Phase 16
#: `p24_direct` arm it is compared against. Training samples and never uses
#: argmax; evaluation is the opposite on purpose, because a lane with RNG in
#: the *candidate* seat could not be reproduced across machines from weights
#: alone.
CANDIDATE_DECISION_MODE = "greedy"

#: Scoring: effective win rate, draw = 0.5 (contract section 11).
DRAW_SCORE = 0.5

#: The float agreement the cross-machine fixture demands before the pack may
#: be trusted on two machines. Contract section 5 requires this to be
#: MEASURED, not assumed; this is the ceiling a measurement must come in
#: under, not a claim about what will be observed.
CROSS_MACHINE_PROBABILITY_TOLERANCE = 1e-6

#: A decision disagreement is never within tolerance. Argmax is discrete: one
#: flipped decision changes an entire game, so the fixture compares selected
#: actions exactly and probabilities numerically.
CROSS_MACHINE_DECISIONS_MUST_MATCH_EXACTLY = True


class Phase17EvaluationError(RuntimeError):
    """An external evaluation could not be performed as specified."""


class Phase17BundleError(Phase17EvaluationError):
    """A candidate bundle was absent, partial, or bound to the wrong identity."""


class Phase17PackError(Phase17EvaluationError):
    """The composite benchmark pack failed its own identity check."""


class Phase17TransportError(Phase17EvaluationError):
    """A transfer, publication or receipt could not be completed safely."""


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def json_digest(payload) -> str:
    """The convention Agent 1 froze: sorted, separator-tight, sha256.

    Identical to `stratego.evaluation.phase16.benchmark.manifest_digest` and
    to `stratego.training.phase17.checkpoint.json_digest`, restated here so
    the remote subset does not have to import a training module to check a
    pack.
    """
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path, *, chunk: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def state_mapping_digest(state) -> str:
    """The accepted `state_dict_digest` walk, over a plain mapping.

    Byte-for-byte the function Agent 4's `export._state_mapping_digest` uses,
    which is in turn the `phase9_behavior.state_dict_digest` walk over a live
    module. The remote side recomputes with this and must reproduce the
    manifest's claim before a single game is played.
    """
    import torch

    hasher = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        hasher.update(name.encode())
        array = (
            torch.as_tensor(tensor)
            .detach()
            .to("cpu", torch.float32)
            .contiguous()
            .numpy()
        )
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# The joint lane's fixed setup seeds
# ---------------------------------------------------------------------------


def joint_setup_seeds(board_id: str, color: int) -> dict:
    """The frozen `(root, per-token)` seeds one joint case draws under.

    Derived under the *lane* id, never under the run id. That is the whole
    point of "fixed per-case setup RNG seeds": case 7's setup must be drawn
    from the same random stream at hour 0 and at hour 12, and in a rerun of
    an entirely different run, so that the only thing that moves between two
    candidates is the setup network's weights.
    """
    from ...training.phase17.setup_contract import (
        SETUP_PREFIXES,
        setup_root_seed,
        setup_token_seed,
    )

    root = setup_root_seed(JOINT_LANE_ID, board_id, int(color))
    return {
        "setup_root_seed": int(root),
        "per_token_seeds": [
            int(setup_token_seed(root, prefix)) for prefix in range(SETUP_PREFIXES)
        ],
    }


__all__ = [
    "CANDIDATE_DECISION_MODE",
    "COMPOSITE_PACK_ID",
    "CROSS_MACHINE_DECISIONS_MUST_MATCH_EXACTLY",
    "CROSS_MACHINE_PROBABILITY_TOLERANCE",
    "DRAW_SCORE",
    "EVALUATION_DEVICE",
    "EVALUATION_DTYPE",
    "EVALUATOR_VERSION",
    "EXPECTED_EXPORT_SCHEMA",
    "JOINT_LANE_ID",
    "LANES",
    "LANE_JOINT",
    "LANE_MOVE_ONLY",
    "MOVE_ONLY_BASE",
    "MOVE_ONLY_BASE_DIGEST",
    "MOVE_ONLY_BASE_FILE",
    "Phase17BundleError",
    "Phase17EvaluationError",
    "Phase17PackError",
    "Phase17TransportError",
    "RECEIPT_SCHEMA_VERSION",
    "TRANSPORT_PROTOCOL_VERSION",
    "WORKER_TORCH_THREADS",
    "file_sha256",
    "joint_setup_seeds",
    "json_digest",
    "state_mapping_digest",
]
