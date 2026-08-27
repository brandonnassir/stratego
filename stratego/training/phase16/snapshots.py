"""Phase 16 Agent 3: behavior snapshots and the historical pool binding.

A window collector needs the same guarantee Phase 9 built its store around:
every current-policy decision in one window came from *one* immutable set of
weights, and the stored distribution is reproducible from those weights. The
learner's parameters move between windows, so the snapshot is taken by copying
them, not by aliasing them -- an aliased "snapshot" would be mutated by the
next optimizer step and the PPO ratio's denominator would stop meaning what it
says.

Copies, not files
-----------------
Phase 9 and 14 bound a snapshot to a checkpoint *path* because their opponents
were archived files. A Phase 16 arm's behavior snapshot is its own live model
one moment earlier, so it is bound to a state-dict digest instead and the
`checkpoint_path` field records that provenance in words. The accepted
`assert_frozen` check still runs and still means what it meant: no trainable
parameter, not in training mode, and the digest unchanged since the copy.
"""

from __future__ import annotations

import copy

import torch

from ..phase9_behavior import BehaviorSnapshot, state_dict_digest
from ..phase9_collector import IterationParticipants, RulePolicyCache
from .contract import INFERENCE_BATCH_SHAPE, Phase16TrainingError
from .population import HistoricalPool

PHASE16_SNAPSHOT_VERSION = "phase16_behavior_snapshot_v1"

BEHAVIOR_TOKEN_PREFIX = "phase16_behavior_v1"


def behavior_identity(iteration: int) -> str:
    """`W0012` is "the learner frozen at the start of window 12"."""
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise Phase16TrainingError(f"iteration must be an int >= 0, got {iteration!r}")
    return f"W{iteration:04d}"


def behavior_token(identity: str) -> str:
    return f"{BEHAVIOR_TOKEN_PREFIX}|{identity}"


def freeze_model(model, *, device: str):
    """A detached, frozen, eval-mode copy of one live model on `device`."""
    frozen = copy.deepcopy(model).to(torch.device(device))
    frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen


def snapshot_from_model(
    model,
    *,
    identity: str,
    device: str,
    inference_batch_shape: int = INFERENCE_BATCH_SHAPE,
    provenance: str = "live learner weights, copied",
) -> BehaviorSnapshot:
    """One immutable behavior snapshot from the live learner."""
    frozen = freeze_model(model, device=device)
    digest = state_dict_digest(frozen)
    snapshot = BehaviorSnapshot(
        logical_identity=str(identity),
        policy_token=behavior_token(identity),
        checkpoint_path=f"<{PHASE16_SNAPSHOT_VERSION}: {provenance}>",
        checkpoint_sha256=digest,
        device=str(device),
        inference_batch_shape=int(inference_batch_shape),
        model=frozen,
        loaded_state_dict_digest=digest,
    )
    snapshot.assert_frozen()
    return snapshot


def participants_for(
    model,
    *,
    identity: str,
    device: str,
    pool: "HistoricalPool | None" = None,
    historical: "dict | None" = None,
    inference_batch_shape: int = INFERENCE_BATCH_SHAPE,
) -> IterationParticipants:
    """Everything one window needs to play its games.

    `historical` maps a pool identity to a frozen snapshot. An identity with no
    entry cannot be collected against -- the accepted guard against fabricating
    a snapshot to satisfy a draw -- so an arm that never rotates its pool
    carries exactly one entry: the weights it started from.
    """
    behavior = snapshot_from_model(
        model,
        identity=identity,
        device=device,
        inference_batch_shape=inference_batch_shape,
    )
    return IterationParticipants(
        behavior=behavior,
        historical=dict(historical or {}),
        rules=RulePolicyCache(),
    )


def bind_anchor(model, *, identity: str, device: str, inference_batch_shape: int = INFERENCE_BATCH_SHAPE) -> dict:
    """The pool's starting member: the weights every arm begins from."""
    return {
        identity: snapshot_from_model(
            model,
            identity=identity,
            device=device,
            inference_batch_shape=inference_batch_shape,
            provenance="the read-only P24 copy this arm started from",
        )
    }


def snapshot_semantics() -> dict:
    return {
        "snapshot_version": PHASE16_SNAPSHOT_VERSION,
        "binding": "state-dict digest of a frozen deep copy, not a file path",
        "identity": "W<window>, the learner frozen at the start of that window",
        "guarantee": (
            "every current-policy decision in one window came from one "
            "immutable set of weights; assert_frozen is the accepted check"
        ),
    }


__all__ = [
    "BEHAVIOR_TOKEN_PREFIX",
    "PHASE16_SNAPSHOT_VERSION",
    "behavior_identity",
    "behavior_token",
    "bind_anchor",
    "freeze_model",
    "participants_for",
    "snapshot_from_model",
    "snapshot_semantics",
]
