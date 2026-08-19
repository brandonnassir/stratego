"""Phase 11 Agent 2: the validation predictive run.

Specification sources:

- `02_AGENT_2_BELIEF_EVALUATOR_BASELINES_VALIDATION.md` section 5
  ("Validation run")
- Agent 1's `phase11_belief_bank_v1` (stratum bindings, eval move behaviour)

Games are played by the accepted machinery
------------------------------------------
Phase 11 does not own a game loop. Each of the 1,024 validation games is
played by `stratego.evaluation.match_runner.play_match` under a
:class:`MatchSpec` whose `root_seed` is the frozen Phase 11 match seed, and
the opponent is wrapped in the accepted Phase 10 `FrozenSeedPolicy` so its
randomness is a pure function of `(case, game index, ply)`. That is the
accepted Phase 10 Agent 5 pattern, reused rather than re-derived.

The observer seat is the one new object: a policy that asks the belief
owner for the accepted greedy move *and* the belief marginals from one
forward pass, then records the public prediction rows. It reads a
`PolicyInput` and nothing else, so it is structurally incapable of seeing a
hidden rank.

Truth arrives afterwards, from somewhere else
---------------------------------------------
:func:`privileged_truth_pass` runs after the game is over and every learned
and baseline vector already exists. It replays the recorded action history
through the engine, re-derives the public-state identity of every recorded
decision from scratch — a full independent recomputation, not a spot check
— and only then reads `record.true_type`. It cannot feed anything back:
its single output is an `int8` array written to a separate shard.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..engine.constants import BLUE, PLAYER_NAMES, RED
from ..engine.legal_moves import legal_actions
from ..engine.observation import build_observation
from ..engine.state import create_game
from ..engine.transition import apply_action
from ..training.phase11_contract import (
    ACCEPTED_ANCHOR_EXPORT_PATH,
    BELIEF_REQUEST_VERSION,
    EVAL_MOVE_BEHAVIOR,
    Phase11ContractError,
    STRATUM_BINDINGS,
    STRATUM_PHASE8_ANCHOR,
    STRATUM_PHASE9,
)
from ..training.phase11_seed import CASE_GAME_INDICES
from .match_runner import ON_POLICY_ERROR_RAISE, play_match
from .match_spec import EVALUATION_RULES, MatchSpec
from .neural_worker import (
    DECISION_MODE_GREEDY,
    InferenceRequest,
    LocalInferenceChannel,
    RemoteNeuralPolicy,
)
from .phase10_validation import FrozenSeedPolicy
from .phase11_baselines import remaining_counts
from .phase11_belief import Phase11BeliefRequest
from .phase11_public_state import (
    build_public_state_document,
    hidden_opponent_pieces,
    legal_rank_mask,
    public_state_identity,
)
from .phase11_records import Phase11GameRecorder
from .policy import Policy, PolicyRequirements, PolicyRef, build_public_view
from .registry import build_policy
from .setup_bank import SetupBank, SetupPair

#: The suite identity every Phase 11 validation game is played under.
PHASE11_RUN_VERSION = "phase11_belief_validation_v1"

#: The observer seat's policy identity. Distinct from the Phase 9 opponent
#: seat's even in the self-play stratum, so `play_match` resolves two
#: objects and only the observer records predictions.
OBSERVER_POLICY_ID = "phase11_belief_observer_v1"
PHASE9_OPPONENT_POLICY_ID = "phase11_phase9_opponent_v1"
ANCHOR_OPPONENT_POLICY_ID = "phase11_phase8_anchor_opponent_v1"

#: `stratum -> accepted evaluation-registry policy id`, for the six
#: non-neural strata. The two neural strata are bound by checkpoint.
STRATUM_POLICY_IDS = {
    entry["stratum"]: entry["opponent_policy_id"]
    for entry in STRATUM_BINDINGS
    if entry["opponent_policy_id"] is not None
}

_PLAYER_OF = {"red": RED, "blue": BLUE}


class Phase11RunError(Phase11ContractError):
    """A validation game could not be played, recorded or verified."""


def _policy_version(token: str) -> str:
    return f"{PHASE11_RUN_VERSION}+{token}"


def observer_ref() -> PolicyRef:
    return PolicyRef(
        policy_id=OBSERVER_POLICY_ID,
        policy_version=_policy_version(EVAL_MOVE_BEHAVIOR["dtype"]),
    )


def neural_opponent_ref(stratum: str) -> PolicyRef:
    policy_id = (
        PHASE9_OPPONENT_POLICY_ID
        if stratum == STRATUM_PHASE9
        else ANCHOR_OPPONENT_POLICY_ID
    )
    return PolicyRef(
        policy_id=policy_id, policy_version=_policy_version(EVAL_MOVE_BEHAVIOR["dtype"])
    )


# ---------------------------------------------------------------------------
# The observer seat
# ---------------------------------------------------------------------------


class Phase11ObserverPolicy(Policy):
    """The accepted Phase 9 greedy seat, recording its belief marginals.

    One forward per decision serves both the move and the marginals, which
    is what Agent 1 froze. The policy sees a `PolicyInput` and builds its
    public-state document from the `PublicView` alone, so no code path here
    can reach a hidden rank.
    """

    requirements = PolicyRequirements(
        observation=True, legal_action_mask=True, public_view=True
    )
    description = (
        "Phase 11 belief observer: the accepted Phase 9 greedy decision plus "
        "the same forward's belief marginals, recorded from public products."
    )

    def __init__(self, ref: PolicyRef, owner, recorder: Phase11GameRecorder) -> None:
        self.policy_id = ref.policy_id
        self.policy_version = ref.policy_version
        self.stochastic = False
        self.owner = owner
        self.recorder = recorder
        self.decisions = 0
        self.events = 0
        self.forward_seconds = 0.0
        self.request_digests: list[str] = []

    @property
    def ref(self) -> PolicyRef:
        return PolicyRef(policy_id=self.policy_id, policy_version=self.policy_version)

    def decide(self, request):
        view = request.require_public_view()
        observation = request.require_observation()
        document = build_public_state_document(view, observation)
        belief_request = Phase11BeliefRequest(
            request_version=BELIEF_REQUEST_VERSION,
            request_id=f"{request.match_id}#{int(request.ply)}",
            observer_color=PLAYER_NAMES[int(view.observer)],
            public_state_document=document,
            observation=observation,
        )
        payload = InferenceRequest.from_policy_input(request)
        response, prediction, elapsed = self.owner.serve_decision(
            payload, belief_request
        )

        counts = remaining_counts(document)
        events = []
        for piece in hidden_opponent_pieces(document):
            slot = int(piece["piece_slot"])
            events.append(
                {
                    "piece_slot": slot,
                    "piece_square": int(piece["current_square"]),
                    "perspective_square": int(prediction.perspective_squares[slot]),
                    "piece_moved": bool(piece["has_moved"]),
                    "legal_rank_mask": legal_rank_mask(bool(piece["has_moved"])),
                    "belief_logits": prediction.belief_logits[slot],
                }
            )
        self.recorder.record_decision(
            decision_index=int(request.ply),
            public_state_identity=prediction.public_state_identity,
            observation_sha256=document["observation_sha256"],
            remaining_counts=counts,
            events=events,
        )
        self.decisions += 1
        self.events += len(events)
        self.forward_seconds += elapsed
        self.request_digests.append(belief_request.digest())
        return self.result(
            request,
            response.absolute_action_id,
            {
                "phase11_hidden_targets": len(events),
                "phase11_public_state_identity": prediction.public_state_identity,
            },
        )

    def describe(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "decision_mode": DECISION_MODE_GREEDY,
            "holds_model_weights": False,
            "records_predictions": True,
        }


# ---------------------------------------------------------------------------
# One validation game
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase11GamePlan:
    """Everything one validation game needs, resolved from the frozen bank."""

    case_id: str
    case_index: int
    game_id: str
    game_index: int
    stratum: str
    setup_source: str
    observer_color: str
    opponent_color: str
    match_seed: int
    red_setup: tuple
    blue_setup: tuple


def game_plan(case: dict, game_index: int) -> Phase11GamePlan:
    """Resolve one frozen bank case/game into a playable plan."""
    if int(game_index) not in CASE_GAME_INDICES:
        raise Phase11RunError(f"unknown game index {game_index!r}")
    game = case["games"][str(int(game_index))]
    observer_color = game["observer_color"]
    observer_setup = tuple(game["observer"]["setup"])
    opponent_setup = tuple(game["opponent"]["setup"])
    red, blue = (
        (observer_setup, opponent_setup)
        if observer_color == "red"
        else (opponent_setup, observer_setup)
    )
    return Phase11GamePlan(
        case_id=case["case_id"],
        case_index=int(case["case_index"]),
        game_id=game["game_id"],
        game_index=int(game_index),
        stratum=case["stratum"],
        setup_source=case["setup_source"],
        observer_color=observer_color,
        opponent_color=game["opponent_color"],
        match_seed=int(game["match_seed"]),
        red_setup=red,
        blue_setup=blue,
    )


def build_spec(plan: Phase11GamePlan, opponent: PolicyRef) -> MatchSpec:
    """The completely determined specification of one validation game."""
    return MatchSpec(
        candidate=observer_ref(),
        opponent=opponent,
        setup_pair_id=plan.case_index,
        candidate_color=_PLAYER_OF[plan.observer_color],
        replicate=plan.game_index,
        root_seed=plan.match_seed,
        suite_version=PHASE11_RUN_VERSION,
        setup_bank_version=f"{PHASE11_RUN_VERSION}|st={plan.stratum}|src={plan.setup_source}",
        rules=EVALUATION_RULES,
    )


def single_game_bank(spec: MatchSpec, plan: Phase11GamePlan) -> SetupBank:
    """A one-pair bank holding exactly this game's frozen position."""
    pair = SetupPair(
        setup_pair_id=spec.setup_pair_id,
        red_setup=plan.red_setup,
        blue_setup=plan.blue_setup,
        generation_seed=spec.root_seed,
        bank_version=spec.setup_bank_version,
        generation_family=PHASE11_RUN_VERSION,
    )
    return SetupBank(
        bank_version=spec.setup_bank_version,
        root_seed=spec.root_seed,
        generation_family=PHASE11_RUN_VERSION,
        pairs=(pair,),
    )


def opponent_seat(plan: Phase11GamePlan, owners: dict):
    """`(ref, policy)` for the opponent stratum of this game."""
    if plan.stratum == STRATUM_PHASE9:
        ref = neural_opponent_ref(plan.stratum)
        return ref, RemoteNeuralPolicy(
            ref, LocalInferenceChannel(owners["phase9"]), decision_mode=DECISION_MODE_GREEDY
        )
    if plan.stratum == STRATUM_PHASE8_ANCHOR:
        ref = neural_opponent_ref(plan.stratum)
        return ref, RemoteNeuralPolicy(
            ref, LocalInferenceChannel(owners["anchor"]), decision_mode=DECISION_MODE_GREEDY
        )
    policy_id = STRATUM_POLICY_IDS.get(plan.stratum)
    if policy_id is None:
        raise Phase11RunError(f"unknown opponent stratum {plan.stratum!r}")
    policy = build_policy(policy_id)
    return policy.ref, policy


def play_validation_game(case: dict, game_index: int, owners: dict, bank_version: str):
    """Play one frozen validation game; return `(plan, result, recorder)`.

    W/D/L is carried on the result and is report-only: no Phase 11 gate
    reads a game outcome.
    """
    plan = game_plan(case, game_index)
    opponent_ref, opponent_policy = opponent_seat(plan, owners)
    spec = build_spec(plan, opponent_ref)
    recorder = Phase11GameRecorder(
        {
            "bank_version": bank_version,
            "case_id": plan.case_id,
            "case_index": plan.case_index,
            "game_id": plan.game_id,
            "game_index": plan.game_index,
            "observer_color": plan.observer_color,
            "opponent_stratum": plan.stratum,
            "opponent_setup_source": plan.setup_source,
            "match_seed": plan.match_seed,
            "match_id": spec.match_id,
        }
    )
    observer = Phase11ObserverPolicy(observer_ref(), owners["phase9"], recorder)
    policies = {observer_ref().token: observer}
    if opponent_ref.token != observer_ref().token:
        policies[opponent_ref.token] = FrozenSeedPolicy(opponent_policy, plan.match_seed)
    result = play_match(
        spec,
        bank=single_game_bank(spec, plan),
        policies=policies,
        record_actions=True,
        on_policy_error=ON_POLICY_ERROR_RAISE,
    )
    if result.errored:  # pragma: no cover - raises above under RAISE
        raise Phase11RunError(f"{plan.game_id} errored: {result.policy_error}")
    recorder.action_history = tuple(int(action) for action in (result.action_history or ()))
    recorder.meta.update(
        {
            "plies": int(result.plies),
            "decisions": int(result.decisions),
            "terminal_reason": result.terminal_reason,
            "observer_result": result.candidate_result,
            "replay_digest": result.replay_digest,
            "observer_decisions": observer.decisions,
            "forward_seconds": round(observer.forward_seconds, 6),
        }
    )
    return plan, result, recorder, observer


# ---------------------------------------------------------------------------
# The privileged pass
# ---------------------------------------------------------------------------


def privileged_truth_pass(plan: Phase11GamePlan, result, arrays: dict) -> dict:
    """True ranks for one game's recorded events, from a fresh replay.

    Runs only after every learned and baseline vector exists. Re-derives
    each recorded decision's public-state identity, hidden-target set,
    remaining inventory and per-piece mask from scratch, so the stored
    primitives are independently reconstructed at 100% coverage and the
    truth can only ever be attached to the position it belongs to.
    """
    observer = _PLAYER_OF[plan.observer_color]
    history = result.action_history
    if history is None:
        raise Phase11RunError(f"{plan.game_id} recorded no action history")

    decision_index = arrays["decision_index"]
    offsets = arrays["event_offset"]
    slots = arrays["piece_slot"]
    squares = arrays["piece_square"]
    moved = arrays["piece_moved"]
    identities = arrays["public_state_identity"]

    stored_counts = arrays["remaining_counts"]
    stored_masks = arrays["legal_rank_mask"]

    truth = np.full(int(slots.size), -1, dtype=np.int8)
    identity_mismatches = 0
    alignment_mismatches = 0
    count_mismatches = 0
    mask_mismatches = 0
    verified_decisions = 0

    state = create_game(
        plan.red_setup, plan.blue_setup, rules=EVALUATION_RULES, game_id=plan.game_id
    )
    wanted = {int(value): position for position, value in enumerate(decision_index)}
    for action in history:
        if state.terminal:  # pragma: no cover - the history stops at terminal
            break
        ply = int(state.total_moves)
        if state.acting_player == observer and ply in wanted:
            position = wanted.pop(ply)
            view = build_public_view(state, observer)
            document = build_public_state_document(
                view, build_observation(state, observer)
            )
            rebuilt = public_state_identity(document)
            if rebuilt != bytes(identities[position]).hex():
                identity_mismatches += 1
            verified_decisions += 1
            start, stop = int(offsets[position]), int(offsets[position + 1])
            hidden = {
                int(piece["piece_slot"]): piece
                for piece in hidden_opponent_pieces(document)
            }
            if sorted(hidden) != [int(value) for value in slots[start:stop]]:
                alignment_mismatches += 1
            if tuple(remaining_counts(document)) != tuple(
                int(value) for value in stored_counts[position]
            ):
                count_mismatches += 1
            for cursor in range(start, stop):
                slot = int(slots[cursor])
                piece = hidden.get(slot)
                if piece is None:
                    alignment_mismatches += 1
                    continue
                if int(piece["current_square"]) != int(squares[cursor]) or bool(
                    piece["has_moved"]
                ) != bool(moved[cursor]):
                    alignment_mismatches += 1
                if legal_rank_mask(bool(piece["has_moved"])) != tuple(
                    int(value) for value in stored_masks[cursor]
                ):
                    mask_mismatches += 1
                record = state.pieces[_piece_id_of(observer, slot)]
                truth[cursor] = int(record.true_type)
        apply_action(state, int(action))

    if wanted:
        alignment_mismatches += len(wanted)
    return {
        "game_id": plan.game_id,
        "true_rank_index": truth,
        "verified_decisions": verified_decisions,
        "recorded_decisions": int(decision_index.size),
        "identity_mismatches": identity_mismatches,
        "alignment_mismatches": alignment_mismatches,
        "count_mismatches": count_mismatches,
        "mask_mismatches": mask_mismatches,
        "unlabelled_events": int((truth < 0).sum()),
        "replay_digest": result.replay_digest,
    }


def _piece_id_of(observer: int, slot: int) -> int:
    """The engine piece id of the *opponent's* piece in `slot`."""
    from ..engine.constants import PIECES_PER_PLAYER
    from ..engine.pieces import make_piece_id

    opponent = BLUE if observer == RED else RED
    if not 0 <= slot < PIECES_PER_PLAYER:  # pragma: no cover - defensive
        raise Phase11RunError(f"piece slot {slot} is outside 0..39")
    return make_piece_id(opponent, slot)


def run_summary(started: float, games: int, decisions: int, events: int) -> dict:
    elapsed = time.perf_counter() - started
    return {
        "run_version": PHASE11_RUN_VERSION,
        "games": games,
        "observer_decisions": decisions,
        "prediction_events": events,
        "wall_clock_seconds": round(elapsed, 3),
        "games_per_second": round(games / elapsed, 3) if elapsed > 0 else None,
    }


__all__ = [
    "ANCHOR_OPPONENT_POLICY_ID",
    "OBSERVER_POLICY_ID",
    "PHASE11_RUN_VERSION",
    "PHASE9_OPPONENT_POLICY_ID",
    "Phase11GamePlan",
    "Phase11ObserverPolicy",
    "Phase11RunError",
    "STRATUM_POLICY_IDS",
    "build_spec",
    "game_plan",
    "neural_opponent_ref",
    "observer_ref",
    "opponent_seat",
    "play_validation_game",
    "privileged_truth_pass",
    "run_summary",
    "single_game_bank",
]
