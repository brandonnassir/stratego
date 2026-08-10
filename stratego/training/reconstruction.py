"""Rebuild historical positions from compact trajectory records.

Specification sources:

- `03_game_engine_spec.md` sections 12, 13 (replay, snapshots)
- `06_observation_v2_127ch.md` section 15 (privileged belief targets)
- `08_internal_state_spec.md` sections 15, 16

The reconstruction path is:

```text
game record + nearest snapshot at or before p + subsequent actions
    -> frozen reference GameState
    -> observation_v2_1_127ch
    -> legal actions / dense mask
    -> privileged belief target
```

Every step is a call into `stratego.engine`. This module chooses *which* state
to build and *what* to compare; it never decides a rule, a channel value or a
legality.

Belief separation
-----------------
:func:`reconstruct_decision` returns the observation and the belief target in
two different fields, and the observation is produced by `build_observation`,
which has no code path to `belief_target`. A consumer that feeds
:attr:`ReconstructedDecision.observation` to a network cannot reach the
privileged labels by accident; a consumer that wants the training target must
name :attr:`ReconstructedDecision.belief_target` explicitly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..engine.events import public_board_view, public_setup_view
from ..engine.legal_moves import legal_action_mask, legal_actions
from ..engine.observation import belief_target, build_observation
from ..engine.snapshot import clone_state, restore_snapshot
from ..engine.state import GameState, state_fingerprint
from ..engine.transition import apply_action
from .trajectory import DecisionRecord, GameRecord, TrajectoryError, decode_snapshot

RECONSTRUCTION_VERSION = "reconstruction_v1"

_DIGEST_SIZE = 16


# ---------------------------------------------------------------------------
# State reconstruction
# ---------------------------------------------------------------------------


def restore_snapshot_entry(record: GameRecord, index: int) -> GameState:
    """Decode and restore one of a record's snapshots into a live state."""
    if not 0 <= index < len(record.snapshots):
        raise TrajectoryError(
            f"snapshot index {index} is outside game {record.game_id} "
            f"({len(record.snapshots)} snapshots)"
        )
    entry = record.snapshots[index]
    return restore_snapshot(decode_snapshot(entry.payload, record.context()))


def reconstruct_state(record: GameRecord, ply: int) -> tuple[GameState, int]:
    """Rebuild the position before ply `ply`.

    Returns the state and the number of actions that had to be replayed on top
    of the snapshot, which is the quantity the snapshot-interval benchmark
    reports.
    """
    index = record.snapshot_index_for_ply(ply)
    state = restore_snapshot_entry(record, index)
    start = record.snapshots[index].ply
    for action_id in record.actions[start:ply]:
        apply_action(state, action_id)
    return state, ply - start


# ---------------------------------------------------------------------------
# Reconstructed decisions
# ---------------------------------------------------------------------------


@dataclass
class ReconstructedDecision:
    """Everything a training consumer needs for one historical decision.

    `observation` is the model input. `belief_target` is a *training target*
    and is deliberately a separate field: it is never mixed into the tensor.
    """

    game_id: str
    environment_id: int
    generation: int
    ply: int
    acting_player: int
    state: GameState
    observation: np.ndarray
    legal_action_ids: tuple[int, ...]
    legal_mask: np.ndarray | None
    belief_target: list[dict]
    public_knowledge: dict | None
    replayed_actions: int


def reconstruct_decision(
    record: GameRecord,
    ply: int,
    *,
    dense_mask: bool = False,
    include_public_knowledge: bool = True,
    state: GameState | None = None,
    replayed_actions: int | None = None,
) -> ReconstructedDecision:
    """Rebuild one historical decision from the record.

    `state` lets a caller that already advanced a state to `ply` skip the
    snapshot restore; it is how :func:`iter_reconstructed_decisions` walks a
    whole game cheaply. Random-access callers leave it alone.
    """
    if state is None:
        state, replayed = reconstruct_state(record, ply)
    else:
        replayed = 0 if replayed_actions is None else replayed_actions
        if state.total_moves != ply:
            raise TrajectoryError(
                f"supplied state is at ply {state.total_moves}, not {ply}"
            )

    legal = tuple(legal_actions(state))
    observation = build_observation(state, state.acting_player)
    mask = legal_action_mask(state, list(legal)) if dense_mask else None
    public = None
    if include_public_knowledge:
        public = public_knowledge_view(state)

    return ReconstructedDecision(
        game_id=record.game_id,
        environment_id=record.environment_id,
        generation=record.generation,
        ply=ply,
        acting_player=state.acting_player,
        state=state,
        observation=observation,
        legal_action_ids=legal,
        legal_mask=mask,
        belief_target=belief_target(state, state.acting_player),
        public_knowledge=public,
        replayed_actions=replayed,
    )


def iter_reconstructed_decisions(
    record: GameRecord,
    plies: "list[int] | tuple[int, ...] | None" = None,
    *,
    dense_mask: bool = False,
    include_public_knowledge: bool = True,
    copy_state: bool = True,
):
    """Reconstruct many plies of one game in ascending order.

    Sequential access advances a single state instead of restoring a snapshot
    per position. The result is identical to independent reconstruction; the
    snapshot-interval benchmark deliberately does *not* use this path, because
    the quantity it measures is random-access cost.

    The observation, legal set, mask and belief target are built at yield time
    and are therefore always correct for their ply. The underlying `state` is
    the one being advanced, so `copy_state=True` (the default) hands out an
    independent copy and a materialised list of results stays valid. Pass
    `copy_state=False` only when each result is consumed before the next is
    pulled; the shared state will already have moved on afterwards.
    """
    requested = sorted(set(range(len(record.actions)) if plies is None else plies))
    if not requested:
        return
    if requested[0] < 0 or requested[-1] > len(record.actions):
        raise TrajectoryError(f"ply out of range for game {record.game_id}")

    state = None
    replayed = 0
    for ply in requested:
        if state is None or state.total_moves > ply:
            state, replayed = reconstruct_state(record, ply)
        elif state.total_moves < ply:
            for action_id in record.actions[state.total_moves : ply]:
                apply_action(state, action_id)
                replayed += 1
        rebuilt = reconstruct_decision(
            record,
            ply,
            dense_mask=dense_mask,
            include_public_knowledge=include_public_knowledge,
            state=state,
            replayed_actions=replayed,
        )
        if copy_state:
            rebuilt.state = clone_state(state)
        yield rebuilt
        replayed = 0


# ---------------------------------------------------------------------------
# Public knowledge
# ---------------------------------------------------------------------------


def public_knowledge_view(state: GameState) -> dict:
    """Both observers' browser-safe views of the position.

    This is the knowledge half of the public schema: the board and setup views
    from `09_public_event_and_replay_schema.md` section 12, which are derived
    from the per-piece knowledge flags and therefore reconstruct exactly from a
    snapshot. The event *stream* additionally needs the game's derived event
    log, which a compact snapshot deliberately omits; the acceptance harness
    checks that separately by replaying whole games from ply 0.
    """
    return {
        "red": {
            "board": public_board_view(state, 0),
            "setup": public_setup_view(state, 0),
        },
        "blue": {
            "board": public_board_view(state, 1),
            "setup": public_setup_view(state, 1),
        },
    }


# ---------------------------------------------------------------------------
# Digests: comparing live collection against later reconstruction
# ---------------------------------------------------------------------------


def _digest(*parts: bytes) -> bytes:
    hasher = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return hasher.digest()


def observation_digest(observation: np.ndarray) -> bytes:
    """Digest of the exact `float32` bytes of an observation tensor."""
    array = np.ascontiguousarray(observation, dtype=np.float32)
    return _digest(array.tobytes())


def legal_actions_digest(legal_action_ids: "tuple[int, ...] | list[int]") -> bytes:
    return _digest(",".join(str(action) for action in legal_action_ids).encode())


def legal_mask_digest(mask: np.ndarray) -> bytes:
    return _digest(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())


def belief_target_digest(targets: "list[dict]") -> bytes:
    payload = ";".join(
        f"{item['piece_id']}:{item['square']}:{item['true_type']}" for item in targets
    )
    return _digest(payload.encode())


def _canonical_bytes(value) -> bytes:
    """Deterministic byte form of a nested structure of engine primitives.

    `repr` is used rather than a sorting serialiser because it is orders of
    magnitude faster at a million comparisons, and because the structures being
    hashed are built by a single engine function from an equivalent state on
    both sides: key order is a property of that function, not of the caller. A
    difference in ordering would itself be a real difference worth reporting.
    """
    return repr(value).encode()


def public_knowledge_digest(public: dict) -> bytes:
    return _digest(_canonical_bytes(public))


def state_fingerprint_digest(state: GameState) -> bytes:
    """Digest of the full privileged fingerprint, history excluded.

    History is excluded because a snapshot legitimately carries no event log or
    action history (`08_internal_state_spec.md` section 15); the action history
    is verified separately by the record's own action list.
    """
    return _digest(_canonical_bytes(state_fingerprint(state, include_history=False)))


@dataclass(frozen=True)
class DecisionDigest:
    """Compact comparison surface for one decision.

    Captured live during collection and again after reconstruction. Digests
    rather than tensors: a million live observations would be 50 GB, and the
    comparison only needs to detect difference.
    """

    game_id: str
    environment_id: int
    generation: int
    ply: int
    acting_player: int
    selected_action_id: int
    state_fingerprint: bytes
    observation: bytes
    legal_actions: bytes
    legal_action_count: int
    belief_target: bytes
    public_knowledge: bytes
    legal_mask: bytes | None


def digest_live_decision(
    state: GameState,
    decision: DecisionRecord,
    *,
    environment_id: int,
    generation: int,
    dense_mask: bool = False,
    legal_action_ids: "tuple[int, ...] | list[int] | None" = None,
) -> DecisionDigest:
    """Digest a decision from the live game, before its action is applied."""
    legal = tuple(legal_actions(state)) if legal_action_ids is None else tuple(legal_action_ids)
    return DecisionDigest(
        game_id=state.game_id,
        environment_id=environment_id,
        generation=generation,
        ply=state.total_moves,
        acting_player=state.acting_player,
        selected_action_id=decision.selected_action_id,
        state_fingerprint=state_fingerprint_digest(state),
        observation=observation_digest(build_observation(state, state.acting_player)),
        legal_actions=legal_actions_digest(legal),
        legal_action_count=len(legal),
        belief_target=belief_target_digest(belief_target(state, state.acting_player)),
        public_knowledge=public_knowledge_digest(public_knowledge_view(state)),
        legal_mask=(
            legal_mask_digest(legal_action_mask(state, list(legal))) if dense_mask else None
        ),
    )


def digest_reconstructed_decision(
    reconstructed: ReconstructedDecision, decision: DecisionRecord
) -> DecisionDigest:
    """Digest a reconstructed decision using the same surface."""
    if reconstructed.public_knowledge is None:
        raise TrajectoryError(
            "reconstruct with include_public_knowledge=True to compare public knowledge"
        )
    return DecisionDigest(
        game_id=reconstructed.game_id,
        environment_id=reconstructed.environment_id,
        generation=reconstructed.generation,
        ply=reconstructed.ply,
        acting_player=reconstructed.acting_player,
        selected_action_id=decision.selected_action_id,
        state_fingerprint=state_fingerprint_digest(reconstructed.state),
        observation=observation_digest(reconstructed.observation),
        legal_actions=legal_actions_digest(reconstructed.legal_action_ids),
        legal_action_count=len(reconstructed.legal_action_ids),
        belief_target=belief_target_digest(reconstructed.belief_target),
        public_knowledge=public_knowledge_digest(reconstructed.public_knowledge),
        legal_mask=(
            legal_mask_digest(reconstructed.legal_mask)
            if reconstructed.legal_mask is not None
            else None
        ),
    )


# The comparison surface, in the order the acceptance report lists it. Each
# entry is `(mismatch category, attribute)`.
COMPARISON_FIELDS = (
    ("identity_generation", "game_id"),
    ("identity_generation", "environment_id"),
    ("identity_generation", "generation"),
    ("identity_generation", "ply"),
    ("acting_player", "acting_player"),
    ("selected_action", "selected_action_id"),
    ("state", "state_fingerprint"),
    ("observation", "observation"),
    ("legal_list", "legal_actions"),
    ("belief_target", "belief_target"),
    ("public_knowledge", "public_knowledge"),
    ("legal_mask", "legal_mask"),
)


def compare_digests(live: DecisionDigest, rebuilt: DecisionDigest) -> list[tuple[str, str]]:
    """Mismatched `(category, field)` pairs between a live and a rebuilt digest.

    A `legal_mask` of `None` on either side means the dense mask was not part of
    this comparison; that is a skip, not a mismatch.
    """
    mismatches: list[tuple[str, str]] = []
    for category, attribute in COMPARISON_FIELDS:
        left = getattr(live, attribute)
        right = getattr(rebuilt, attribute)
        if attribute == "legal_mask" and (left is None or right is None):
            continue
        if left != right:
            mismatches.append((category, attribute))
    return mismatches


def verify_decision(
    record: GameRecord,
    decision: DecisionRecord,
    live: DecisionDigest,
    *,
    dense_mask: bool = False,
) -> tuple[list[tuple[str, str]], int]:
    """Reconstruct one stored decision and compare it against its live digest.

    Returns the mismatch list and the number of actions replayed on top of the
    snapshot.
    """
    rebuilt = reconstruct_decision(record, decision.ply, dense_mask=dense_mask)
    digest = digest_reconstructed_decision(rebuilt, decision)
    return compare_digests(live, digest), rebuilt.replayed_actions


__all__ = [
    "COMPARISON_FIELDS",
    "RECONSTRUCTION_VERSION",
    "DecisionDigest",
    "ReconstructedDecision",
    "belief_target_digest",
    "compare_digests",
    "digest_live_decision",
    "digest_reconstructed_decision",
    "iter_reconstructed_decisions",
    "legal_actions_digest",
    "legal_mask_digest",
    "observation_digest",
    "public_knowledge_digest",
    "public_knowledge_view",
    "reconstruct_decision",
    "reconstruct_state",
    "restore_snapshot_entry",
    "state_fingerprint_digest",
]
