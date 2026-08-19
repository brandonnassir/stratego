"""Phase 11 Agent 2: the frozen observer-legal public-state document.

Specification sources:

- `02_AGENT_2_BELIEF_EVALUATOR_BASELINES_VALIDATION.md` ("Strict
  public/privileged separation", "Prediction recorder")
- Agent 1's `phase11_belief_contract_v1`, section `public_state_document`

Why the builder takes a `PublicView` and not a `GameState`
----------------------------------------------------------
`public_state_identity` is the anchor of every Phase 11 purity claim: a
sampled world, a belief vector and a baseline vector are all supposed to be
pure functions of it. That claim is only as strong as the document's
inability to see hidden truth, so the builder is given a type that *cannot*
carry it. :class:`~stratego.evaluation.policy.PublicView` is the accepted
Phase 4 projection whose every field "is invariant under a valid hidden
identity permutation"; this module adds nothing to it except one constant
table.

The one thing `PublicView` does not carry is `starting_square`, which the
frozen document requires. It is not privileged information and it is not
recovered from the engine's piece records: a piece's setup slot *is* its
index into the fixed :data:`~stratego.engine.constants.SETUP_SQUARES` table
for its owner, so the starting square of `(owner, slot)` is a compile-time
constant of the rules, identical in every game ever played. `PUBLIC_START_SQUARES`
is that table, and `tests/evaluation/test_phase11_public_state.py` pins it
against `public_setup_view`'s opponent occupancy, which is the engine's own
statement of what the observer may legally see about the start.

Identity
--------
```text
observation_sha256      sha256 over the (127, 10, 10) float32 C-order bytes
public_state_identity   sha256 over the document's canonical JSON
```

The document *embeds* `observation_sha256`, so its identity covers the
complete model input as well as the observer-legal facts — the purity claim
Agent 3's sampler and Agent 4's safety attack both rest on.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from ..engine.constants import (
    NUM_PIECE_TYPES,
    NUM_SQUARES,
    PLAYER_NAMES,
    SETUP_SQUARES,
)
from ..engine.observation import OBSERVATION_VERSION
from ..engine.pieces import piece_owner, piece_setup_slot
from ..training.phase11_contract import (
    IMMOVABLE_RANK_INDICES,
    PUBLIC_PIECE_FIELDS,
    PUBLIC_STATE_DOCUMENT_FIELDS,
    PUBLIC_STATE_DOCUMENT_VERSION,
    Phase11ContractError,
    RANK_COUNT,
    progress_bucket,
)

#: The engine's frozen rules and reference-engine identities, recorded in
#: every document so a stored identity can never be read under other rules.
RULES_VERSION = "stratego_project_v1"
ENGINE_VERSION = "phase2_1_reference_1.2.0"

#: `PUBLIC_START_SQUARES[owner][slot]` — the square a piece started on. A
#: constant of the rules: `create_piece_records` assigns setup slot `i` to
#: `SETUP_SQUARES[owner][i]`, so this depends on nothing about the game.
PUBLIC_START_SQUARES: dict[int, tuple[int, ...]] = {
    owner: tuple(squares) for owner, squares in SETUP_SQUARES.items()
}

#: The observation the document commits to, as a shape and a dtype.
OBSERVATION_DIGEST_RECIPE = (
    "sha256 over the observation's float32 C-order bytes, shape (127, 10, 10)"
)

#: The domain the canonical-JSON identity is taken over.
PUBLIC_STATE_IDENTITY_RECIPE = "sha256 over the document's canonical JSON"


class Phase11PublicStateError(Phase11ContractError):
    """A public-state document could not be built or verified."""


def canonical_json(payload) -> str:
    """The accepted canonical form: sorted keys, no incidental whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def observation_digest(observation) -> str:
    """`sha256` over the model input exactly as the model consumes it."""
    array = np.asarray(observation)
    if array.shape != (127, 10, 10):
        raise Phase11PublicStateError(
            f"observation has shape {array.shape}, expected (127, 10, 10)"
        )
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise Phase11PublicStateError("observation carries a non-finite value")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def legal_rank_mask(has_moved: bool) -> tuple[int, ...]:
    """The frozen 12-entry public legal-rank mask of one hidden piece.

    Movement impossibility *only*: a piece the opponent has publicly seen
    move cannot be a Flag or a Bomb. The mask never consults counts — the
    baseline and the sampler multiply it by the remaining inventory, and
    folding counts in here would double-count them.
    """
    if has_moved:
        return tuple(
            0 if rank in IMMOVABLE_RANK_INDICES else 1 for rank in range(RANK_COUNT)
        )
    return (1,) * RANK_COUNT


def build_public_state_document(view, observation) -> dict:
    """The frozen `phase11_public_state_v1` document for one decision.

    `view` is an accepted :class:`PublicView` built for the observer and
    `observation` is the acting player's 127-channel input. Nothing else is
    read, so the document cannot depend on hidden truth even by accident.
    """
    observer = int(view.observer)
    if observer not in PUBLIC_START_SQUARES:
        raise Phase11PublicStateError(f"unknown observer {view.observer!r}")

    pieces = []
    for piece in view.pieces:
        owner = piece_owner(piece.piece_id)
        slot = piece_setup_slot(piece.piece_id)
        known = bool(piece.known)
        entry = {
            "piece_slot": int(slot),
            "owner_color": PLAYER_NAMES[owner],
            "alive": bool(piece.alive),
            "current_square": None if piece.square is None else int(piece.square),
            "has_moved": bool(piece.has_moved),
            "known_to_observer": known,
            "known_rank_index": int(piece.piece_type) if known else None,
            "starting_square": int(PUBLIC_START_SQUARES[owner][slot]),
        }
        if tuple(entry) != PUBLIC_PIECE_FIELDS:
            raise Phase11PublicStateError(
                "public piece fields drifted from the frozen schema"
            )
        pieces.append(entry)
    if len(pieces) != 2 * len(PUBLIC_START_SQUARES[observer]):
        raise Phase11PublicStateError(f"expected 80 pieces, got {len(pieces)}")

    recent = [
        {
            "ply": int(move.ply),
            "player_color": PLAYER_NAMES[int(move.player)],
            "piece_slot": int(piece_setup_slot(move.piece_id)),
            "piece_owner_color": PLAYER_NAMES[piece_owner(move.piece_id)],
            "source": int(move.source),
            "destination": int(move.destination),
            "was_attack": bool(move.was_attack),
            "target_piece_slot": (
                None
                if move.target_piece_id is None
                else int(piece_setup_slot(move.target_piece_id))
            ),
            "target_owner_color": (
                None
                if move.target_piece_id is None
                else PLAYER_NAMES[piece_owner(move.target_piece_id)]
            ),
        }
        for move in view.recent_moves
    ]

    document = {
        "document_version": PUBLIC_STATE_DOCUMENT_VERSION,
        "observer_color": PLAYER_NAMES[observer],
        "acting_player_color": PLAYER_NAMES[int(view.acting_player)],
        "total_moves": int(view.ply),
        "battleless_moves": int(view.battleless_moves),
        "rules_version": RULES_VERSION,
        "engine_version": ENGINE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "pieces": pieces,
        "recent_moves": recent,
        "observation_sha256": observation_digest(observation),
    }
    if tuple(document) != PUBLIC_STATE_DOCUMENT_FIELDS:
        raise Phase11PublicStateError(
            "public-state document fields drifted from the frozen schema"
        )
    return document


def public_state_identity(document: dict) -> str:
    """`sha256` over the document's canonical JSON — the frozen identity."""
    if document.get("document_version") != PUBLIC_STATE_DOCUMENT_VERSION:
        raise Phase11PublicStateError(
            f"document version {document.get('document_version')!r} is not "
            f"{PUBLIC_STATE_DOCUMENT_VERSION!r}"
        )
    return hashlib.sha256(canonical_json(document).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Reading a document back
# ---------------------------------------------------------------------------


def hidden_opponent_pieces(document: dict) -> "list[dict]":
    """Live opponent pieces whose exact rank the observer may not know.

    Exactly the frozen prediction-event set, read back off a stored
    document: the same live/opponent/unknown filter the engine's
    `belief_target` applies, in setup-slot order.
    """
    observer = document["observer_color"]
    hidden = [
        piece
        for piece in document["pieces"]
        if piece["owner_color"] != observer
        and piece["alive"]
        and not piece["known_to_observer"]
    ]
    return sorted(hidden, key=lambda piece: piece["piece_slot"])


def document_progress_bucket(document: dict) -> str:
    """The frozen progress bucket of the decision this document describes."""
    return progress_bucket(int(document["total_moves"]))


def document_summary(document: dict) -> dict:
    """A compact, report-safe description of one document."""
    hidden = hidden_opponent_pieces(document)
    return {
        "document_version": document["document_version"],
        "observer_color": document["observer_color"],
        "total_moves": document["total_moves"],
        "progress_bucket": document_progress_bucket(document),
        "hidden_opponent_pieces": len(hidden),
        "moved_hidden_pieces": sum(1 for piece in hidden if piece["has_moved"]),
        "observation_sha256": document["observation_sha256"],
        "public_state_identity": public_state_identity(document),
    }


__all__ = [
    "ENGINE_VERSION",
    "OBSERVATION_DIGEST_RECIPE",
    "PUBLIC_START_SQUARES",
    "PUBLIC_STATE_IDENTITY_RECIPE",
    "Phase11PublicStateError",
    "RULES_VERSION",
    "build_public_state_document",
    "canonical_json",
    "document_progress_bucket",
    "document_summary",
    "hidden_opponent_pieces",
    "legal_rank_mask",
    "observation_digest",
    "public_state_identity",
]
