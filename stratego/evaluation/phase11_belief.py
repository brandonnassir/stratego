"""Phase 11 Agent 2: the production belief-inference path and its boundary.

Specification sources:

- `02_AGENT_2_BELIEF_EVALUATOR_BASELINES_VALIDATION.md` section 1 ("Strict
  public/privileged separation")
- Agent 1's `phase11_belief_contract_v1`, section `production_request`

The boundary is a type, not a convention
----------------------------------------
:class:`Phase11BeliefRequest` carries exactly the five frozen fields and
:meth:`Phase11BeliefRequest.from_payload` **raises** — never drops — on
anything else, following the accepted `SelectorRequest.from_payload`
pattern. A payload carrying a `true_rank`, a `target`, a private piece
table or a storage path cannot become a request, so the production path
cannot be handed hidden truth even by a caller that wants to. The rejection
covers both the frozen forbidden tokens and any unknown field, because the
next leak will have a name nobody predicted.

One forward serves the decision and the belief
-----------------------------------------------
Agent 1 froze prediction events to "the same forward that chooses the
observer's move". :class:`Phase11BeliefOwner` therefore subclasses the
accepted :class:`~stratego.evaluation.neural_worker.InferenceOwner` and adds
a single method that runs the accepted `_forward` once and then uses the
accepted `_select` for the action and the same outputs' belief logits for
the marginals. The move rule is not reimplemented anywhere in Phase 11 —
it is literally the accepted call.

The learned 12-vector is the raw float64 softmax of the head's logits at
the piece's perspective-normalized square: no masking, no epsilon, full
simplex. That is how the head was trained and how the sampler consumes it,
so it is how it must be measured.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields

import numpy as np

from ..engine.constants import PLAYER_NAMES, PLAYERS
from ..engine.coordinates import to_perspective
from ..training.phase11_contract import (
    ALLOWED_BELIEF_REQUEST_FIELDS,
    BELIEF_REQUEST_VERSION,
    EVALUATOR_VERSION,
    FORBIDDEN_BELIEF_REQUEST_TOKENS,
    Phase11ContractError,
    RANK_COUNT,
)
from .neural_worker import InferenceOwner
from .phase11_public_state import (
    PUBLIC_STATE_DOCUMENT_VERSION,
    hidden_opponent_pieces,
    observation_digest,
    public_state_identity,
)

#: `PLAYER_NAMES` inverted: the colour token a request names -> the player id.
PLAYER_BY_COLOR = {PLAYER_NAMES[player]: player for player in PLAYERS}


class Phase11BeliefError(Phase11ContractError):
    """A belief request was refused, or a belief output failed its checks."""


def _forbidden_token_in(name: str) -> "str | None":
    lowered = str(name).lower()
    for token in FORBIDDEN_BELIEF_REQUEST_TOKENS:
        if token in lowered:
            return token
    return None


@dataclass(frozen=True)
class Phase11BeliefRequest:
    """One production belief request. Public information, structurally.

    Constructed directly by the runner from observer-legal products, or
    from an untrusted mapping through :meth:`from_payload`, which is the
    boundary the negative controls attack.
    """

    request_version: str
    request_id: str
    observer_color: str
    public_state_document: dict
    observation: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if tuple(item.name for item in fields(self)) != ALLOWED_BELIEF_REQUEST_FIELDS:
            raise Phase11BeliefError(
                "Phase11BeliefRequest fields drifted from the frozen allowlist"
            )
        if self.request_version != BELIEF_REQUEST_VERSION:
            raise Phase11BeliefError(
                f"request version {self.request_version!r} is not "
                f"{BELIEF_REQUEST_VERSION!r}"
            )
        if not isinstance(self.request_id, str) or not self.request_id:
            raise Phase11BeliefError("the request carries no request_id")
        if self.observer_color not in PLAYER_BY_COLOR:
            raise Phase11BeliefError(
                f"observer_color {self.observer_color!r} is not a colour"
            )
        document = self.public_state_document
        if not isinstance(document, dict):
            raise Phase11BeliefError("public_state_document must be a mapping")
        if document.get("document_version") != PUBLIC_STATE_DOCUMENT_VERSION:
            raise Phase11BeliefError(
                "public_state_document is not a "
                f"{PUBLIC_STATE_DOCUMENT_VERSION!r} document"
            )
        if document.get("observer_color") != self.observer_color:
            raise Phase11BeliefError(
                "the request and its document disagree about the observer"
            )
        digest = observation_digest(self.observation)
        if document.get("observation_sha256") != digest:
            raise Phase11BeliefError(
                "the observation does not match the document's observation_sha256"
            )

    @property
    def observer(self) -> int:
        return PLAYER_BY_COLOR[self.observer_color]

    @property
    def public_state_identity(self) -> str:
        return public_state_identity(self.public_state_document)

    @classmethod
    def from_payload(cls, payload) -> "Phase11BeliefRequest":
        """Build from an untrusted mapping, refusing anything off-allowlist.

        Raises on an unknown field, on a field whose name carries a frozen
        forbidden token, and on a missing field. Nothing is silently
        dropped: a dropped field is a leak that succeeded quietly.
        """
        if not isinstance(payload, dict):
            raise Phase11BeliefError(
                f"a belief request payload must be a mapping, got "
                f"{type(payload).__name__}"
            )
        unknown = [key for key in payload if key not in ALLOWED_BELIEF_REQUEST_FIELDS]
        if unknown:
            offending = ", ".join(sorted(str(key) for key in unknown))
            raise Phase11BeliefError(
                f"belief request carries fields outside the frozen allowlist: "
                f"{offending}"
            )
        for key in payload:
            token = _forbidden_token_in(key)
            if token is not None:  # pragma: no cover - unreachable via allowlist
                raise Phase11BeliefError(
                    f"belief request field {key!r} carries the forbidden token "
                    f"{token!r}"
                )
        missing = [
            key for key in ALLOWED_BELIEF_REQUEST_FIELDS if key not in payload
        ]
        if missing:
            raise Phase11BeliefError(
                f"belief request is missing {', '.join(missing)}"
            )
        document = payload["public_state_document"]
        if isinstance(document, dict):
            for key in document:
                token = _forbidden_token_in(key)
                if token is not None:
                    raise Phase11BeliefError(
                        f"the public-state document carries a forbidden field "
                        f"{key!r} (token {token!r})"
                    )
        return cls(
            request_version=payload["request_version"],
            request_id=payload["request_id"],
            observer_color=payload["observer_color"],
            public_state_document=document,
            observation=payload["observation"],
        )

    def digest(self) -> str:
        """Content digest of the request, for the boundary audit."""
        hasher = hashlib.sha256()
        hasher.update(self.request_version.encode())
        hasher.update(self.request_id.encode())
        hasher.update(self.observer_color.encode())
        hasher.update(self.public_state_identity.encode())
        return hasher.hexdigest()


@dataclass(frozen=True)
class Phase11BeliefPrediction:
    """Learned marginals for one decision. Public information, structurally.

    Carries no true rank and no field that could hold one; the privileged
    evaluator attaches truth to a *copy* of the primitive rows afterwards.
    """

    request_id: str
    observer_color: str
    public_state_identity: str
    observation_sha256: str
    total_moves: int
    #: Public setup slot -> the head's raw float32 12-logit row.
    belief_logits: dict
    #: Public setup slot -> the piece's perspective-normalized square.
    perspective_squares: dict

    def probabilities(self) -> "dict[int, np.ndarray]":
        """The frozen learned 12-vectors: float64 softmax, no masking."""
        return {
            slot: softmax_float64(row) for slot, row in self.belief_logits.items()
        }


def softmax_float64(logits) -> np.ndarray:
    """The frozen probability extraction: subtract max, exponentiate, normalize."""
    row = np.asarray(logits, dtype=np.float64)
    if row.shape != (RANK_COUNT,):
        raise Phase11BeliefError(
            f"a belief logit row has shape {row.shape}, expected ({RANK_COUNT},)"
        )
    if not np.isfinite(row).all():
        raise Phase11BeliefError("a belief logit row carries a non-finite value")
    shifted = np.exp(row - row.max())
    total = float(shifted.sum())
    if not np.isfinite(total) or total <= 0.0:  # pragma: no cover - defensive
        raise Phase11BeliefError("a belief softmax produced no mass")
    return shifted / total


class Phase11BeliefOwner(InferenceOwner):
    """The accepted inference owner, serving one belief forward per decision.

    Adds exactly one method. The move still comes from the accepted
    `_forward` + `_select` pair, so the observer's play in Phase 11 is the
    accepted Phase 9 greedy policy and nothing else.
    """

    evaluator_version = EVALUATOR_VERSION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.belief_forwards = 0
        self.belief_rows = 0
        self.hidden_input_accesses = 0
        self.belief_seconds = 0.0

    def serve_decision(self, payload, belief_request: Phase11BeliefRequest):
        """`(response, prediction)` from one forward pass.

        `payload` is the accepted :class:`InferenceRequest` that drives the
        move; `belief_request` is the Phase 11 production belief request.
        The two are cross-checked on the observation digest before the
        forward, so a decision can never pair one position's move with
        another position's beliefs.
        """
        if not isinstance(belief_request, Phase11BeliefRequest):
            raise Phase11BeliefError(
                "the belief path accepts only a Phase11BeliefRequest, got "
                f"{type(belief_request).__name__}"
            )
        document = belief_request.public_state_document
        if observation_digest(payload.observation) != document["observation_sha256"]:
            raise Phase11BeliefError(
                "the move request and the belief request describe different "
                "positions"
            )
        observer = belief_request.observer
        if int(payload.acting_player) != observer:
            raise Phase11BeliefError(
                "belief events are recorded only where the observer is acting"
            )
        if observer not in PLAYERS:  # pragma: no cover - defensive
            raise Phase11BeliefError(f"unknown observer {observer!r}")

        self._validate_request(payload)
        legality = self._adapter.prepare_legality(
            payload.legal_actions, payload.legal_action_mask, payload.acting_player
        )
        outputs, elapsed = self._forward([payload])
        response = self._select(outputs, 0, payload, legality, 1)

        rows = outputs.belief_logits[0].detach().to("cpu", self._torch.float32).numpy()
        logits: dict[int, np.ndarray] = {}
        squares: dict[int, int] = {}
        for piece in hidden_opponent_pieces(document):
            slot = int(piece["piece_slot"])
            square = piece["current_square"]
            if square is None:  # pragma: no cover - a live piece has a square
                raise Phase11BeliefError(f"live hidden slot {slot} has no square")
            normalized = to_perspective(int(square), observer)
            squares[slot] = int(normalized)
            logits[slot] = np.array(rows[normalized], dtype=np.float32, copy=True)

        self.belief_forwards += 1
        self.belief_rows += len(logits)
        self.belief_seconds += elapsed
        prediction = Phase11BeliefPrediction(
            request_id=belief_request.request_id,
            observer_color=belief_request.observer_color,
            public_state_identity=belief_request.public_state_identity,
            observation_sha256=document["observation_sha256"],
            total_moves=int(document["total_moves"]),
            belief_logits=logits,
            perspective_squares=squares,
        )
        return response, prediction, elapsed

    def boundary_report(self) -> dict:
        """What the production path saw, for the boundary gate."""
        return {
            "evaluator_version": self.evaluator_version,
            "belief_request_version": BELIEF_REQUEST_VERSION,
            "allowed_request_fields": list(ALLOWED_BELIEF_REQUEST_FIELDS),
            "forbidden_tokens": list(FORBIDDEN_BELIEF_REQUEST_TOKENS),
            "belief_forwards": self.belief_forwards,
            "belief_rows": self.belief_rows,
            "hidden_input_accesses": self.hidden_input_accesses,
            "request_type_rejects_truth": True,
        }


__all__ = [
    "PLAYER_BY_COLOR",
    "Phase11BeliefError",
    "Phase11BeliefOwner",
    "Phase11BeliefPrediction",
    "Phase11BeliefRequest",
    "softmax_float64",
]
