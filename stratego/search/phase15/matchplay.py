"""Phase 15 Agent 2 section 12: playing the arms on the fresh pack.

Specification source: `02_AGENT_2_SEARCH_IMPLEMENTATION.md` sections 10, 12.

Paired by construction
----------------------
Every arm plays the *same* board list, with the same setups, the same
opponent and the same match seeds. The board id determines every stream a
game consumes, so "P18+B18 minus P18 direct on board X" is a difference
between two plays of one position rather than between two samples of a
distribution — which is what makes a compact pack able to say anything at
all.

Two kinds of seat, one interface
--------------------------------
```text
DirectSeat   the accepted RemoteNeuralPolicy over P18 or P24, one forward
SearchSeat   the accepted Phase 12 engine over one (move model, provider)
```

`DirectSeat` decides through `build_policy_input` + `decide_checked` — the
same two calls `play_match` itself makes — so the direct arm is the accepted
direct player rather than a re-implementation of it. That is also what makes
"search changed the move" mean "changed it relative to the direct arm", and
the probe checks it decision by decision.

The probe, and its positive control
-----------------------------------
`SeatProbe` re-decides sampled positions on a state whose hidden opponent
identities have been permuted by the accepted `permute_hidden_identities`.
A production seat reads only public information, so its answer must not
change. The oracle arm is the positive control: it reads the true world by
design, so `expects_hidden_truth=True` inverts the reading and a changed
answer is counted as sensitivity. A probe that never fired on the oracle
would be a probe with no power, and the report says so either way.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from ...engine.constants import BLUE, RED
from ...engine.legal_moves import legal_actions
from ...engine.permutation import permute_hidden_identities
from ...engine.state import create_game
from ...engine.transition import apply_action
from ...evaluation.match_spec import EVALUATION_RULES, MatchSpec
from ...evaluation.neural_worker import (
    DECISION_MODE_GREEDY,
    InferenceOwner,
    LocalInferenceChannel,
    RemoteNeuralPolicy,
)
from ...evaluation.phase10_validation import FrozenSeedPolicy
from ...evaluation.policy import PolicyRef, build_policy_input
from ...evaluation.registry import build_policy
from ...evaluation.setup_bank import SetupBank, SetupPair
from ..phase12.contract import Phase12SearchError
from ..phase12.engine import Phase12SearchEngine
from .boards import Phase15BoardPlan
from .contract import (
    DOMAIN_PROBE,
    MATCH_VERSION,
    NEURAL_OPPONENTS,
    OPPONENT_CLASS,
    RULE_OPPONENT_POLICY_IDS,
    Pairing,
    Phase15SearchError,
    derive_search_seed,
    pairing as pairing_of,
    search_seed_for,
)

#: The player's arm-independent seat identity. One identity for every arm,
#: so a stored row is distinguished by its `arm_id` column and not by a
#: policy token that would also change the match id and therefore the board.
PLAYER_POLICY_ID = "phase15_match_player_v1"
OPPONENT_POLICY_ID = "phase15_match_opponent_v1"

_COLOR_OF = {RED: "red", BLUE: "blue"}


class Phase15MatchError(Phase15SearchError):
    """A Phase 15 match game could not be set up or played."""


# ---------------------------------------------------------------------------
# Owners
# ---------------------------------------------------------------------------


def build_owners(models, *, device: str = "cpu") -> dict:
    """One long-lived inference owner per neural opponent.

    The owners exist for the *opponent* seats and for the direct arms; the
    search engine holds the loaded model objects directly. Both read the same
    verified files, and the loader has already bound those files to the
    handoff digests.
    """
    from ...model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config

    paths = {
        name: loaded.identity["checkpoint_path"]
        for name, loaded in models.move_models.items()
    }
    if models.anchor_identity:
        paths["phase9_anchor"] = models.anchor_identity["export_path"]
    return {
        name: InferenceOwner(
            path,
            decision_mode=DECISION_MODE_GREEDY,
            device=device,
            dtype="float32",
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name=f"phase15_match_{name}",
        )
        for name, path in paths.items()
    }


# ---------------------------------------------------------------------------
# Spec, bank, opponent
# ---------------------------------------------------------------------------


def player_ref() -> PolicyRef:
    return PolicyRef(policy_id=PLAYER_POLICY_ID, policy_version=MATCH_VERSION)


def build_spec(plan: Phase15BoardPlan, opponent: PolicyRef) -> MatchSpec:
    return MatchSpec(
        candidate=player_ref(),
        opponent=opponent,
        setup_pair_id=plan.cell_index,
        candidate_color=plan.player,
        replicate=plan.ordinal,
        root_seed=plan.match_seed,
        suite_version=MATCH_VERSION,
        setup_bank_version=(
            f"{MATCH_VERSION}|opp={plan.opponent}|src={plan.setup_source}"
            f"|fam={plan.requested_family}"
        ),
        rules=EVALUATION_RULES,
    )


def single_game_bank(spec: MatchSpec, plan: Phase15BoardPlan) -> SetupBank:
    """The one-pair bank this board resolves through.

    The setups could go straight to `create_game`; routing them through the
    accepted bank keeps them inside the accepted identity apparatus, so a
    stored row names the board it was played on. The setups placed here are
    the *engine-ready* tuples the orientation gate already accepted.
    """
    pair = SetupPair(
        setup_pair_id=spec.setup_pair_id,
        red_setup=plan.red_setup,
        blue_setup=plan.blue_setup,
        generation_seed=spec.root_seed,
        bank_version=spec.setup_bank_version,
        generation_family=MATCH_VERSION,
    )
    return SetupBank(
        bank_version=spec.setup_bank_version,
        root_seed=spec.root_seed,
        generation_family=MATCH_VERSION,
        pairs=(pair,),
    )


def opponent_seat(plan: Phase15BoardPlan, owners: dict):
    """`(ref, policy)` for this board's opponent."""
    if plan.opponent in NEURAL_OPPONENTS:
        if plan.opponent not in owners:
            raise Phase15MatchError(
                f"board {plan.board_id} needs a {plan.opponent} owner; loaded owners "
                f"are {sorted(owners)}"
            )
        ref = PolicyRef(
            policy_id=f"{OPPONENT_POLICY_ID}|{plan.opponent}",
            policy_version=MATCH_VERSION,
        )
        return ref, RemoteNeuralPolicy(
            ref,
            LocalInferenceChannel(owners[plan.opponent]),
            decision_mode=DECISION_MODE_GREEDY,
        )
    policy = build_policy(RULE_OPPONENT_POLICY_IDS[plan.opponent])
    return policy.ref, FrozenSeedPolicy(policy, plan.match_seed)


# ---------------------------------------------------------------------------
# Seats
# ---------------------------------------------------------------------------


class DirectSeat:
    """A direct arm: P18 or P24 greedy, one forward per decision."""

    kind = "direct"

    def __init__(self, pairing: Pairing, owners: dict) -> None:
        if pairing.kind != "direct":
            raise Phase15MatchError(f"{pairing.pairing_id!r} is not a direct pairing")
        if pairing.move_model not in owners:
            raise Phase15MatchError(
                f"{pairing.pairing_id} needs a {pairing.move_model} owner"
            )
        self.pairing = pairing
        self.arm_id = pairing.pairing_id
        self.ref = player_ref()
        self.policy = RemoteNeuralPolicy(
            self.ref,
            LocalInferenceChannel(owners[pairing.move_model]),
            decision_mode=DECISION_MODE_GREEDY,
        )

    def decide(self, state, legal, spec: MatchSpec, plan: Phase15BoardPlan):
        request = build_policy_input(
            state,
            policy=self.ref,
            policy_seed=spec.policy_seed_for(state.acting_player),
            requirements=self.policy.requirements,
            suite_version=spec.suite_version,
            match_id=spec.match_id,
            paired_unit_id=spec.paired_unit_id,
            legal=legal,
        )
        started = time.perf_counter()
        result = self.policy.decide_checked(request)
        seconds = time.perf_counter() - started
        return int(result.selected_action_id), {
            "ply": int(state.total_moves),
            "seconds": seconds,
            "legal_actions": len(legal),
            "move_changed": None,
            "c1_forwards": 1,
            "unique_worlds": None,
            "candidates": None,
            "fallback": None,
            "direct_action_id": int(result.selected_action_id),
        }

    def describe(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "pairing": self.pairing.describe(),
            "seat": f"{self.pairing.move_model.upper()} greedy, one forward per decision",
        }


class SearchSeat:
    """A search arm: the accepted Phase 12 engine over one pairing.

    Falls back to the engine's own direct action on any search failure, so a
    game can never be forfeited because search broke — the same rule the
    working player applies, exercised here on every match decision.
    """

    kind = "search"

    def __init__(
        self,
        pairing: Pairing,
        engine: Phase12SearchEngine,
        *,
        owners: "dict | None" = None,
        time_cap: "float | None" = None,
    ) -> None:
        if pairing.kind == "direct":
            raise Phase15MatchError(f"{pairing.pairing_id!r} is not a search pairing")
        if engine.provider.provider_id != pairing.provider:
            raise Phase15MatchError(
                f"arm {pairing.pairing_id!r} wants provider {pairing.provider!r} but "
                f"the engine carries {engine.provider.provider_id!r}"
            )
        self.pairing = pairing
        self.arm_id = pairing.pairing_id
        self.engine = engine
        self.time_cap = None if time_cap is None else float(time_cap)
        self.fallbacks: dict[str, int] = {}
        self.direct_policy = None
        if owners is not None and pairing.move_model in owners:
            self.direct_policy = RemoteNeuralPolicy(
                player_ref(),
                LocalInferenceChannel(owners[pairing.move_model]),
                decision_mode=DECISION_MODE_GREEDY,
            )

    def _fallback(self, state, legal, spec, plan, reason: str, started: float):
        """Play the selected move model's direct legal move."""
        self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1
        if self.direct_policy is not None:
            request = build_policy_input(
                state,
                policy=self.direct_policy.ref,
                policy_seed=spec.policy_seed_for(state.acting_player),
                requirements=self.direct_policy.requirements,
                suite_version=spec.suite_version,
                match_id=spec.match_id,
                paired_unit_id=spec.paired_unit_id,
                legal=legal,
            )
            action = int(self.direct_policy.decide_checked(request).selected_action_id)
        else:  # pragma: no cover - only when no owner was supplied
            action = int(min(legal))
        return action, {
            "ply": int(state.total_moves),
            "seconds": time.perf_counter() - started,
            "legal_actions": len(legal),
            "move_changed": False,
            "c1_forwards": 1,
            "unique_worlds": None,
            "candidates": None,
            "fallback": reason,
            "direct_action_id": action,
        }

    def decide(self, state, legal, spec: MatchSpec, plan: Phase15BoardPlan):
        seed = search_seed_for(plan.board_id, int(state.total_moves))
        started = time.perf_counter()
        deadline = None if self.time_cap is None else started + self.time_cap
        try:
            decision = self.engine.choose_action(state, seed=seed, deadline=deadline)
        except Phase12SearchError as error:
            reason = (
                "timeout" if type(error).__name__ == "Phase12SearchTimeout" else "search_error"
            )
            return self._fallback(state, legal, spec, plan, reason, started)
        selected = int(decision.selected_action_id)
        if selected not in legal:  # pragma: no cover - the engine checks first
            return self._fallback(state, legal, spec, plan, "illegal_action", started)
        return selected, {
            "ply": int(state.total_moves),
            "seconds": float(decision.seconds),
            "legal_actions": int(decision.legal_action_count),
            "move_changed": bool(decision.move_changed),
            "c1_forwards": int(decision.c1_forwards),
            "unique_worlds": int(decision.unique_worlds),
            "candidates": len(decision.candidates),
            "forward_seconds": float(decision.forward_seconds),
            "fallback": None,
            "direct_action_id": int(decision.direct_action_id),
            "score_margin": _score_margin(decision),
        }

    def describe(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "pairing": self.pairing.describe(),
            "time_cap_seconds": self.time_cap,
            "fallbacks": dict(self.fallbacks),
            "seat": self.engine.describe(),
        }


def _score_margin(decision) -> "float | None":
    """Chosen score minus runner-up score, or `None` with one candidate."""
    scores = sorted((candidate.score for candidate in decision.candidates), reverse=True)
    if len(scores) < 2:
        return None
    return float(scores[0] - scores[1])


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


@dataclass
class SeatProbe:
    """Sampled permutation-invariance and direct-agreement checks."""

    reference: object = None
    interval: int = 20
    budget: int = 24
    expects_hidden_truth: bool = False
    permutation_checks: int = 0
    permutation_changed: int = 0
    permutation_sensitive: int = 0
    direct_checks: int = 0
    failures: list = field(default_factory=list)

    def due(self, decision_index: int) -> bool:
        return (
            self.budget > 0
            and self.interval > 0
            and decision_index > 0
            and decision_index % self.interval == 0
        )

    def run(self, seat, state, legal, spec, plan, action: int, record: dict) -> None:
        self.budget -= 1
        player = state.acting_player
        rng = random.Random(
            derive_search_seed(DOMAIN_PROBE, plan.board_id, int(state.total_moves))
        )
        permuted, info = permute_hidden_identities(state, player, rng)
        permuted_action, _ = seat.decide(permuted, legal_actions(permuted), spec, plan)
        self.permutation_checks += 1
        if info.get("changed"):
            self.permutation_changed += 1
        if permuted_action != action:
            if self.expects_hidden_truth:
                self.permutation_sensitive += 1
            else:
                self.failures.append(
                    {
                        "check": "permutation_invariance",
                        "board_id": plan.board_id,
                        "arm_id": seat.arm_id,
                        "ply": int(state.total_moves),
                        "action": int(action),
                        "permuted_action": int(permuted_action),
                        "hidden_pieces": int(info.get("hidden_pieces", 0)),
                    }
                )
        if self.reference is not None and record.get("direct_action_id") is not None:
            request = build_policy_input(
                state,
                policy=self.reference.ref,
                policy_seed=spec.policy_seed_for(player),
                requirements=self.reference.requirements,
                suite_version=spec.suite_version,
                match_id=spec.match_id,
                paired_unit_id=spec.paired_unit_id,
                legal=legal,
            )
            accepted = int(self.reference.decide_checked(request).selected_action_id)
            self.direct_checks += 1
            if accepted != int(record["direct_action_id"]) and not record.get("fallback"):
                self.failures.append(
                    {
                        "check": "direct_action_agreement",
                        "board_id": plan.board_id,
                        "arm_id": seat.arm_id,
                        "ply": int(state.total_moves),
                        "engine_direct": int(record["direct_action_id"]),
                        "accepted_direct": accepted,
                    }
                )

    def summary(self) -> dict:
        return {
            "expects_hidden_truth": self.expects_hidden_truth,
            "permutation_checks": self.permutation_checks,
            "permutation_assignments_changed": self.permutation_changed,
            "permutation_sensitive": self.permutation_sensitive,
            "direct_agreement_checks": self.direct_checks,
            "failures": list(self.failures),
            "passed": not self.failures,
        }


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameRecord:
    """The public result of one arm playing one board."""

    board_id: str
    arm_id: str
    move_model: str
    provider: "str | None"
    preset_id: str
    opponent: str
    opponent_class: str
    setup_source: str
    requested_family: str
    player_family_key: str
    player_color: str
    ordinal: int
    match_id: str
    outcome: str
    effective_score: float
    winner: "str | None"
    terminal_reason: str
    plies: int
    player_decisions: int
    seconds: float
    player_seconds: float
    move_changes: int
    fallbacks: int
    c1_forwards: int
    move_seconds: tuple = field(repr=False, default=())
    moves: tuple = field(repr=False, default=())

    @property
    def move_change_rate(self) -> "float | None":
        if self.player_decisions == 0:
            return None
        changes = [row["move_changed"] for row in self.moves]
        if any(value is None for value in changes):
            return None
        return self.move_changes / self.player_decisions

    def row(self) -> dict:
        return {
            "board_id": self.board_id,
            "arm_id": self.arm_id,
            "move_model": self.move_model,
            "provider": self.provider,
            "preset_id": self.preset_id,
            "opponent": self.opponent,
            "opponent_class": self.opponent_class,
            "setup_source": self.setup_source,
            "requested_family": self.requested_family,
            "player_family_key": self.player_family_key,
            "player_color": self.player_color,
            "ordinal": self.ordinal,
            "match_id": self.match_id,
            "outcome": self.outcome,
            "effective_score": self.effective_score,
            "winner": self.winner,
            "terminal_reason": self.terminal_reason,
            "plies": self.plies,
            "player_decisions": self.player_decisions,
            "seconds": round(self.seconds, 4),
            "player_seconds": round(self.player_seconds, 4),
            "seconds_per_player_move": (
                round(self.player_seconds / self.player_decisions, 5)
                if self.player_decisions
                else None
            ),
            "move_changes": self.move_changes,
            "move_change_rate": (
                round(self.move_change_rate, 5)
                if self.move_change_rate is not None
                else None
            ),
            "fallbacks": self.fallbacks,
            "c1_forwards": self.c1_forwards,
        }


def outcome_of(score: float) -> str:
    if score > 0.5:
        return "win"
    if score < 0.5:
        return "loss"
    return "draw"


def play_board(
    plan: Phase15BoardPlan,
    seat,
    owners: dict,
    *,
    probe: "SeatProbe | None" = None,
    keep_moves: bool = False,
    preset_id: str = "-",
) -> GameRecord:
    """Play one board with one arm in the player seat."""
    opponent_reference, opponent_policy = opponent_seat(plan, owners)
    spec = build_spec(plan, opponent_reference)
    bank = single_game_bank(spec, plan)
    red_setup, blue_setup = spec.resolve_setups(bank)
    state = create_game(red_setup, blue_setup, rules=spec.rules, game_id=spec.game_id)

    player = plan.player
    moves: list = []
    move_seconds: list = []
    player_seconds = 0.0
    move_changes = 0
    fallbacks = 0
    c1_forwards = 0
    started = time.perf_counter()

    while not state.terminal:
        actor = state.acting_player
        legal = legal_actions(state)
        if actor == player:
            action, record = seat.decide(state, legal, spec, plan)
            if action not in legal:
                raise Phase15MatchError(
                    f"{seat.arm_id} selected illegal action {action} at ply "
                    f"{state.total_moves} of {plan.board_id}"
                )
            player_seconds += float(record["seconds"])
            move_seconds.append(float(record["seconds"]))
            c1_forwards += int(record["c1_forwards"] or 0)
            if record["move_changed"]:
                move_changes += 1
            if record.get("fallback"):
                fallbacks += 1
            if probe is not None and probe.due(len(moves)):
                probe.run(seat, state, legal, spec, plan, action, record)
            record["action_id"] = int(action)
            moves.append(record if keep_moves else {"move_changed": record["move_changed"]})
        else:
            request = build_policy_input(
                state,
                policy=opponent_reference,
                policy_seed=spec.policy_seed_for(actor),
                requirements=opponent_policy.requirements,
                suite_version=spec.suite_version,
                match_id=spec.match_id,
                paired_unit_id=spec.paired_unit_id,
                legal=legal,
            )
            action = int(opponent_policy.decide_checked(request).selected_action_id)
        apply_action(state, action, legal=legal)

    seconds = time.perf_counter() - started
    score = float(state.effective_score_for(player))
    return GameRecord(
        board_id=plan.board_id,
        arm_id=seat.arm_id,
        move_model=seat.pairing.move_model,
        provider=seat.pairing.provider,
        preset_id=preset_id,
        opponent=plan.opponent,
        opponent_class=OPPONENT_CLASS[plan.opponent],
        setup_source=plan.setup_source,
        requested_family=plan.requested_family,
        player_family_key=plan.player_family_key,
        player_color=plan.color,
        ordinal=plan.ordinal,
        match_id=spec.match_id,
        outcome=outcome_of(score),
        effective_score=score,
        winner=None if state.winner is None else _COLOR_OF[state.winner],
        terminal_reason=str(state.terminal_reason),
        plies=int(state.total_moves),
        player_decisions=len(moves),
        seconds=seconds,
        player_seconds=player_seconds,
        move_changes=move_changes,
        fallbacks=fallbacks,
        c1_forwards=c1_forwards,
        move_seconds=tuple(move_seconds),
        moves=tuple(moves),
    )


__all__ = [
    "DirectSeat",
    "GameRecord",
    "OPPONENT_POLICY_ID",
    "PLAYER_POLICY_ID",
    "Phase15MatchError",
    "SearchSeat",
    "SeatProbe",
    "build_owners",
    "build_spec",
    "opponent_seat",
    "outcome_of",
    "play_board",
    "player_ref",
    "single_game_bank",
]
