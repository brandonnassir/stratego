"""The observer-safe policy contract shared by every Phase 4 evaluation policy.

Specification sources:

- `06_observation_v2_127ch.md` (the observation a policy may read)
- `09_public_event_and_replay_schema.md` sections 11, 12, 16 (what is public)
- Phase 4 Agent 1 instructions ("Policy interface")

One interface, three consumers
------------------------------
The same :class:`Policy` interface is meant to carry Phase 4 baselines, future
neural checkpoints and later decision-time search policies. Nothing here knows
what a baseline is; Agent 2 owns strategy.

Why the input holds data instead of the state
---------------------------------------------
A policy must never be able to read a hidden identity. The obvious way to make
the 127-channel observation lazy is a closure over the `GameState`, but a
closure *is* a live reference to the privileged state, so it fails the
instruction's "no engine object that permits reading hidden types" requirement.

Instead a policy **declares** what it needs (:class:`PolicyRequirements`) and
:func:`build_policy_input` materialises exactly that much. The resulting
:class:`PolicyInput` is a frozen dataclass of plain scalars, tuples and NumPy
arrays with no reference of any kind to `GameState`, `PieceRecord`, a belief
target or a privileged replay. Anything a policy does not ask for arrives as
`None`, so an unused observation is never built and an unused observation can
never leak.

Everything on :class:`PublicView` is invariant under
:func:`stratego.engine.permutation.permute_hidden_identities`, which is what
Agent 4's audit tests directly.

Decision seeds
--------------
`policy_seed` is fixed per (match, role) by :mod:`stratego.evaluation.match_spec`.
`decision_seed = derive_decision_seed(policy_seed, ply)` gives each ply its own
independent stream, so a stochastic policy is reproducible ply by ply without
having to replay the game to rebuild its generator state.
"""

import hashlib
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..engine.constants import (
    NUM_PIECE_TYPES,
    NUM_SQUARES,
    PIECE_COUNTS,
    PLAYERS,
    RulesConfig,
    opponent_of,
)
from ..engine.events import filter_events_for_observer, public_setup_view
from ..engine.legal_moves import legal_action_mask, legal_actions
from ..engine.observation import build_observation
from ..engine.state import GameState

POLICY_INTERFACE_VERSION = "policy_interface_v1"


class PolicyContractError(RuntimeError):
    """Raised when a policy or a policy input violates the contract.

    Deliberately loud: the Phase 4 instructions forbid papering over a policy
    failure with a substituted legal move.
    """


# ---------------------------------------------------------------------------
# Identity and requirements
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class PolicyRef:
    """A policy's identity and version, as it appears in a match identifier."""

    policy_id: str
    policy_version: str

    @property
    def token(self) -> str:
        """Canonical `id@version` text used inside identity hashes."""
        return f"{self.policy_id}@{self.policy_version}"

    def to_dict(self) -> dict:
        return {"policy_id": self.policy_id, "policy_version": self.policy_version}

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "PolicyRef":
        return PolicyRef(str(payload["policy_id"]), str(payload["policy_version"]))

    @staticmethod
    def from_token(token: str) -> "PolicyRef":
        policy_id, separator, policy_version = token.rpartition("@")
        if not separator:
            raise PolicyContractError(f"malformed policy token: {token!r}")
        return PolicyRef(policy_id, policy_version)


@dataclass(frozen=True)
class PolicyRequirements:
    """Which observer-safe products a policy wants materialised.

    Defaults suit a rule-based baseline: the decoded public view, no tensor.
    A neural checkpoint sets `observation=True` and usually
    `legal_action_mask=True`.
    """

    observation: bool = False
    legal_action_mask: bool = False
    public_view: bool = True
    public_events: bool = False
    public_setup: bool = False

    def to_dict(self) -> dict:
        return {
            "observation": self.observation,
            "legal_action_mask": self.legal_action_mask,
            "public_view": self.public_view,
            "public_events": self.public_events,
            "public_setup": self.public_setup,
        }


# ---------------------------------------------------------------------------
# The decoded observer-safe view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicPiece:
    """One physical piece as the observer is legally allowed to see it.

    `piece_type` is `None` exactly when the observer may not know the type. The
    stable `piece_id` is always present because a human observer can follow a
    concealed piece as it moves; it encodes owner and starting slot only, never
    type.
    """

    piece_id: int
    owner: int
    square: int | None
    piece_type: int | None
    known: bool
    alive: bool
    has_moved: bool
    capture_ply: int | None

    @property
    def hidden(self) -> bool:
        return not self.known


@dataclass(frozen=True)
class PublicMove:
    """One entry of the public 16-ply move window."""

    ply: int
    player: int
    piece_id: int
    source: int
    destination: int
    was_attack: bool
    target_piece_id: int | None


@dataclass(frozen=True)
class PublicView:
    """Everything about the position that `observer` may legally know.

    Every field is derived from public facts or from the observer's own pieces,
    and every field is invariant under a valid hidden-identity permutation.
    """

    observer: int
    acting_player: int
    ply: int
    battleless_moves: int
    battleless_move_limit: int
    absolute_move_limit: int
    terminal: bool
    #: `square -> piece_id or None`. Occupancy and piece identity are public.
    occupancy: tuple[int | None, ...]
    #: All 80 pieces, indexed by `piece_id`, with unknown types masked out.
    pieces: tuple[PublicPiece, ...]
    own_piece_ids: tuple[int, ...]
    opponent_piece_ids: tuple[int, ...]
    #: Live opponent pieces whose exact type the observer cannot know.
    unresolved_opponent_piece_ids: tuple[int, ...]
    #: Per type, how many of the opponent's copies remain unaccounted for.
    unresolved_opponent_counts: tuple[int, ...]
    #: Own pieces the opponent has legally learned; useful for exposure control.
    own_piece_ids_known_to_opponent: tuple[int, ...]
    recent_moves: tuple[PublicMove, ...]

    @property
    def opponent(self) -> int:
        return opponent_of(self.observer)

    def piece(self, piece_id: int) -> PublicPiece:
        return self.pieces[piece_id]

    def piece_at(self, square: int) -> PublicPiece | None:
        piece_id = self.occupancy[square]
        return None if piece_id is None else self.pieces[piece_id]

    @property
    def moves_until_battleless_draw(self) -> int:
        return self.battleless_move_limit - self.battleless_moves


def build_public_view(state: GameState, observer: int) -> PublicView:
    """Decode `state` into the observer-legal facts a policy may reason over.

    This is the same information the browser-safe `public_board_view` carries,
    in a compact form that a rule-based policy can use without decoding 127
    observation planes. No field consults `true_type` for an opponent piece the
    observer does not legally know.
    """
    if observer not in PLAYERS:
        raise PolicyContractError(f"unknown observer: {observer!r}")
    opponent = opponent_of(observer)

    pieces: list[PublicPiece] = []
    own_ids: list[int] = []
    opponent_ids: list[int] = []
    unresolved_ids: list[int] = []
    exposed_own_ids: list[int] = []
    resolved_opponent_counts = [0] * NUM_PIECE_TYPES

    for record in state.pieces:
        known = record.known_to(observer)
        pieces.append(
            PublicPiece(
                piece_id=record.piece_id,
                owner=record.owner,
                square=record.current_square if record.alive else None,
                piece_type=record.true_type if known else None,
                known=known,
                alive=record.alive,
                has_moved=record.has_moved,
                capture_ply=record.capture_ply,
            )
        )
        if record.owner == observer:
            if record.alive:
                own_ids.append(record.piece_id)
            if record.known_to(opponent):
                exposed_own_ids.append(record.piece_id)
            continue
        if known:
            # Legally revealed, so counting it against the public inventory is
            # a deduction the observer is entitled to make.
            resolved_opponent_counts[record.true_type] += 1
        if record.alive:
            opponent_ids.append(record.piece_id)
            if not known:
                unresolved_ids.append(record.piece_id)

    unresolved_counts = tuple(
        PIECE_COUNTS[piece_type] - resolved_opponent_counts[piece_type]
        for piece_type in range(NUM_PIECE_TYPES)
    )

    recent = tuple(
        PublicMove(
            ply=move.ply,
            player=move.player,
            piece_id=move.piece_id,
            source=move.source,
            destination=move.destination,
            was_attack=move.destination_had_opponent,
            target_piece_id=move.target_piece_id,
        )
        for move in state.recent_moves
    )

    occupancy = tuple(state.board)
    if len(occupancy) != NUM_SQUARES:  # pragma: no cover - defensive
        raise PolicyContractError(f"board has {len(occupancy)} squares")

    return PublicView(
        observer=observer,
        acting_player=state.acting_player,
        ply=state.total_moves,
        battleless_moves=state.battleless_moves,
        battleless_move_limit=state.rules.battleless_move_limit,
        absolute_move_limit=state.rules.absolute_move_limit,
        terminal=state.terminal,
        occupancy=occupancy,
        pieces=tuple(pieces),
        own_piece_ids=tuple(own_ids),
        opponent_piece_ids=tuple(opponent_ids),
        unresolved_opponent_piece_ids=tuple(unresolved_ids),
        unresolved_opponent_counts=unresolved_counts,
        own_piece_ids_known_to_opponent=tuple(exposed_own_ids),
        recent_moves=recent,
    )


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def derive_decision_seed(policy_seed: int, ply: int) -> int:
    """Per-ply seed derived from the policy's match-level seed.

    Hashing rather than mixing keeps consecutive plies uncorrelated and lets a
    single decision be reproduced without replaying the game.
    """
    payload = f"{int(policy_seed)}:{int(ply)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"strat-dec").digest()
    return int.from_bytes(digest, "big") >> 1


# ---------------------------------------------------------------------------
# The decision request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyInput:
    """Everything a policy is allowed to see when choosing one action.

    Contains no `GameState`, no `PieceRecord`, no belief target and no
    privileged replay -- only plain data derived for `acting_player`. Products
    the policy did not request are `None`.
    """

    policy: PolicyRef
    suite_version: str
    match_id: str
    paired_unit_id: str
    game_id: str
    ply: int
    acting_player: int
    legal_actions: tuple[int, ...]
    policy_seed: int
    decision_seed: int
    rules: RulesConfig
    observation: np.ndarray | None = None
    legal_action_mask: np.ndarray | None = None
    public_view: PublicView | None = None
    public_events: tuple[dict, ...] | None = None
    public_setup: Mapping[str, Any] | None = None

    def random_stream(self) -> random.Random:
        """A fresh generator seeded from this decision alone.

        Fresh each call, so a policy that draws twice from `random_stream()`
        gets the same numbers twice rather than depending on call order.
        """
        return random.Random(self.decision_seed)

    def require_observation(self) -> np.ndarray:
        if self.observation is None:
            raise PolicyContractError(
                f"policy {self.policy.token} read the observation without declaring "
                "PolicyRequirements(observation=True)"
            )
        return self.observation

    def require_legal_action_mask(self) -> np.ndarray:
        if self.legal_action_mask is None:
            raise PolicyContractError(
                f"policy {self.policy.token} read the legality mask without declaring "
                "PolicyRequirements(legal_action_mask=True)"
            )
        return self.legal_action_mask

    def require_public_view(self) -> PublicView:
        if self.public_view is None:
            raise PolicyContractError(
                f"policy {self.policy.token} read the public view without declaring "
                "PolicyRequirements(public_view=True)"
            )
        return self.public_view

    def identity(self) -> dict:
        """Serialisable identity of this decision, for diagnostics and replay."""
        return {
            "suite_version": self.suite_version,
            "match_id": self.match_id,
            "paired_unit_id": self.paired_unit_id,
            "game_id": self.game_id,
            "ply": self.ply,
            "acting_player": self.acting_player,
            "policy": self.policy.to_dict(),
            "policy_seed": self.policy_seed,
            "decision_seed": self.decision_seed,
        }


def build_policy_input(
    state: GameState,
    *,
    policy: PolicyRef,
    policy_seed: int,
    requirements: PolicyRequirements = PolicyRequirements(),
    suite_version: str = "",
    match_id: str = "",
    paired_unit_id: str = "",
    game_id: str | None = None,
    legal: "Sequence[int] | None" = None,
) -> PolicyInput:
    """Project `state` down to what the acting player may legally see.

    The returned object keeps no reference to `state`. `legal` may be passed in
    when the caller already generated the legal-action list, which is the normal
    case inside a match runner.
    """
    if state.terminal:
        raise PolicyContractError("a policy decision was requested for a terminal state")

    observer = state.acting_player
    actions = tuple(legal_actions(state) if legal is None else legal)
    if not actions:
        raise PolicyContractError(
            f"non-terminal state {state.game_id!r} presented no legal actions at ply "
            f"{state.total_moves}"
        )

    observation = build_observation(state, observer) if requirements.observation else None
    if observation is not None:
        # The policy must not be able to write back into engine-owned memory.
        observation.setflags(write=False)

    mask = None
    if requirements.legal_action_mask:
        mask = legal_action_mask(state, list(actions))
        mask.setflags(write=False)

    events = None
    if requirements.public_events:
        events = tuple(filter_events_for_observer(state.events, observer))

    setup_view = None
    if requirements.public_setup:
        setup_view = public_setup_view(state, observer)

    return PolicyInput(
        policy=policy,
        suite_version=suite_version,
        match_id=match_id,
        paired_unit_id=paired_unit_id,
        game_id=state.game_id if game_id is None else game_id,
        ply=state.total_moves,
        acting_player=observer,
        legal_actions=actions,
        policy_seed=policy_seed,
        decision_seed=derive_decision_seed(policy_seed, state.total_moves),
        rules=state.rules,
        observation=observation,
        legal_action_mask=mask,
        public_view=build_public_view(state, observer) if requirements.public_view else None,
        public_events=events,
        public_setup=setup_view,
    )


# ---------------------------------------------------------------------------
# The decision result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyResult:
    """One policy decision, with enough metadata to reproduce it."""

    selected_action_id: int
    policy: PolicyRef
    decision_seed: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def policy_id(self) -> str:
        return self.policy.policy_id

    @property
    def policy_version(self) -> str:
        return self.policy.policy_version

    def to_dict(self) -> dict:
        return {
            "selected_action_id": self.selected_action_id,
            "policy": self.policy.to_dict(),
            "decision_seed": self.decision_seed,
            "diagnostics": dict(self.diagnostics),
        }


def validate_policy_result(result: PolicyResult, request: PolicyInput) -> PolicyResult:
    """Check a result against the contract, raising on any violation.

    The engine stays the final legality authority; this only catches a broken
    policy earlier and with a clearer message than an `IllegalActionError` from
    deep inside a transition.
    """
    if not isinstance(result, PolicyResult):
        raise PolicyContractError(
            f"policy {request.policy.token} returned {type(result).__name__}, "
            "expected PolicyResult"
        )
    if result.policy != request.policy:
        raise PolicyContractError(
            f"policy {request.policy.token} returned a result claiming to be "
            f"{result.policy.token}"
        )
    if result.decision_seed != request.decision_seed:
        raise PolicyContractError(
            f"policy {request.policy.token} returned decision seed "
            f"{result.decision_seed}, expected {request.decision_seed}"
        )
    if result.selected_action_id not in request.legal_actions:
        raise PolicyContractError(
            f"policy {request.policy.token} selected illegal action "
            f"{result.selected_action_id} at ply {request.ply} of match "
            f"{request.match_id!r}"
        )
    return result


# ---------------------------------------------------------------------------
# The policy interface
# ---------------------------------------------------------------------------


class Policy(ABC):
    """Base class for every Phase 4 and later evaluation policy.

    Subclasses declare identity and requirements as class attributes and
    implement :meth:`decide`. They must not import or accept a `GameState`.
    """

    policy_id: ClassVar[str]
    policy_version: ClassVar[str]
    requirements: ClassVar[PolicyRequirements] = PolicyRequirements()
    #: Whether the policy consumes `PolicyInput.random_stream()`.
    stochastic: ClassVar[bool] = False
    #: Free-text note carried into reports; useful for stress-policy tables.
    description: ClassVar[str] = ""

    @property
    def ref(self) -> PolicyRef:
        return PolicyRef(self.policy_id, self.policy_version)

    @abstractmethod
    def decide(self, request: PolicyInput) -> PolicyResult:
        """Choose one legal action. Must not mutate `request`."""

    def result(
        self,
        request: PolicyInput,
        selected_action_id: int,
        diagnostics: "Mapping[str, Any] | None" = None,
    ) -> PolicyResult:
        """Build a well-formed :class:`PolicyResult` for this request."""
        return PolicyResult(
            selected_action_id=int(selected_action_id),
            policy=request.policy,
            decision_seed=request.decision_seed,
            diagnostics=dict(diagnostics or {}),
        )

    def decide_checked(self, request: PolicyInput) -> PolicyResult:
        """`decide` plus contract validation. Match runners should use this."""
        if request.policy != self.ref:
            raise PolicyContractError(
                f"request addressed to {request.policy.token} was handed to "
                f"{self.ref.token}"
            )
        return validate_policy_result(self.decide(request), request)

    def describe(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "interface_version": POLICY_INTERFACE_VERSION,
            "requirements": self.requirements.to_dict(),
            "stochastic": self.stochastic,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Minimal policies for contract testing
# ---------------------------------------------------------------------------
#
# Agent 2 owns the real baseline suite. The three policies below exist only so
# the contract, the seeding and the requirement plumbing can be tested without
# any strategy at all, and their identifiers are prefixed `contract_` so they
# can never be mistaken for a ladder opponent.


class FirstLegalActionPolicy(Policy):
    """Deterministic: always the lowest legal action identifier."""

    policy_id = "contract_first_legal"
    policy_version = "1.0.0"
    requirements = PolicyRequirements(public_view=False)
    description = "Contract fixture: lowest legal action identifier."

    def decide(self, request: PolicyInput) -> PolicyResult:
        return self.result(request, min(request.legal_actions), {"rule": "min_legal"})


class SeededUniformPolicy(Policy):
    """Stochastic: uniform over the legal actions, seeded per decision."""

    policy_id = "contract_uniform_random"
    policy_version = "1.0.0"
    requirements = PolicyRequirements(public_view=False)
    stochastic = True
    description = "Contract fixture: uniform legal choice from the decision seed."

    def decide(self, request: PolicyInput) -> PolicyResult:
        rng = request.random_stream()
        index = rng.randrange(len(request.legal_actions))
        return self.result(
            request,
            request.legal_actions[index],
            {"rule": "uniform_legal", "candidate_count": len(request.legal_actions)},
        )


class ObservationProbePolicy(Policy):
    """Deterministic function of the 127-channel observation.

    Exists to prove two things at once: that requirement-declared materialisation
    reaches a policy, and that a decision derived from the observation is
    invariant under hidden-identity permutation.
    """

    policy_id = "contract_observation_probe"
    policy_version = "1.0.0"
    requirements = PolicyRequirements(observation=True, legal_action_mask=True)
    description = "Contract fixture: selects from a checksum of the observation."

    def decide(self, request: PolicyInput) -> PolicyResult:
        observation = request.require_observation()
        mask = request.require_legal_action_mask()
        checksum = int(np.rint(float(np.sum(observation)) * 1000.0))
        index = checksum % len(request.legal_actions)
        return self.result(
            request,
            request.legal_actions[index],
            {"rule": "observation_checksum", "legal_mask_total": int(mask.sum())},
        )
