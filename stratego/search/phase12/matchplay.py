"""The Phase 12 first search match test: whole games, one seat per arm.

Specification source: `04_PHASE_12_AGENT_3_FIRST_MATCH_TEST.md` — four
arms (direct accepted Phase 9 C1, and search over each of the three
production belief providers), the four accepted opponent behaviours, eight
balanced-colour games per opponent per arm, no larger budget than SMALL.

Why this module owns a play loop
--------------------------------
Every other Phase 12 stage plays its games through the accepted
`match_runner.play_match`, and this one cannot. `play_match` hands a policy
a :class:`~stratego.evaluation.policy.PolicyInput` — plain arrays with no
reference to the engine state — which is exactly the property that makes a
policy structurally unable to read a hidden rank. The search engine needs a
`GameState`: it must clone the root, overwrite the hidden opponent ranks
with a sampled world, and roll that world out.

So the driver here holds the true state and hands it to
:meth:`Phase12SearchEngine.choose_action`, while the opponent seat keeps the
accepted boundary untouched — it is built by `build_policy_input` and
decided by `decide_checked`, byte for byte the accepted path. Everything
else is imported rather than restated: the setups come from the accepted
sources through the Phase 11B wrapper, the opponents are the accepted Phase
11 strata, the rules are `EVALUATION_RULES`, the seeds and match identity
come from the accepted `MatchSpec`, and the terminal facts are the engine's
own.

What keeps the search seat honest
---------------------------------
Holding the state is not the same as reading it. The engine reaches the
root through `build_observation(state, root)` (the observer's legal view)
and `legal_actions(state)` (its own moves), and reaches the opponent only
through worlds whose hidden ranks the provider sampled — every hidden rank
is overwritten before a single rollout ply. Agent 1 froze that boundary and
gates it per decision. This module adds a second, independent check at
match time: :class:`SeatProbe` re-decides a sampled position on a state
whose hidden identities have been permuted by the accepted
`permute_hidden_identities`, and requires the identical action. A seat that
had read a hidden rank would answer differently.

The same probe also pins the engine's internal "direct Phase 9 action" to
the accepted `RemoteNeuralPolicy` decision on the same position, which is
what makes the reported move-change rate a comparison against arm A rather
than against a private notion of directness.

Common random numbers
---------------------
Eight games per opponent is a small sample, so the arms play the *same*
games: the board identity, the two setups, the opponent's frozen seed and
the per-ply search seed are all derived without reference to the arm. The
`MatchSpec` names an arm-independent player, so `match_id` — and therefore
the opponent's policy seed — is identical in every arm. Arms diverge only
where their decisions diverge, which is the comparison the instruction asks
for and not a setup lottery.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field

from ...belief.phase11b.contract import (
    CORPUS_COLORS,
    CORPUS_SOURCES,
    CORPUS_STRATA,
    STRATUM_POLICY_IDS,
)
from ...belief.phase11b.corpus import Phase11BSetupSources
from ...engine.constants import BLUE, RED
from ...engine.legal_moves import legal_actions
from ...engine.permutation import permute_hidden_identities
from ...engine.state import GameState, create_game
from ...engine.transition import apply_action
from ...evaluation.match_spec import EVALUATION_RULES, MatchSpec
from ...evaluation.neural_worker import (
    DECISION_MODE_GREEDY,
    LocalInferenceChannel,
    RemoteNeuralPolicy,
)
from ...evaluation.phase10_validation import FrozenSeedPolicy
from ...evaluation.policy import PolicyRef, build_policy_input
from ...evaluation.registry import build_policy
from ...evaluation.setup_bank import SetupBank, SetupPair
from .contract import (
    PROVIDER_AGENT1C,
    PROVIDER_ORACLE,
    PROVIDER_ORIGINAL_PHASE11,
    PROVIDER_REMAINING_COUNT,
    Phase12SearchError,
    derive_phase12_seed,
)
from .engine import Phase12SearchEngine

#: The identity of this match set. Any change to the cells, the game count,
#: the seed derivation or the arm definitions is a new version.
MATCH_VERSION = "phase12_match_test_v1"

#: The Phase 12 match master seed. Distinct from Agent 2's diagnostic
#: master (2026082002), so no match game can coincide with a diagnostic
#: game even though both draw from the same accepted setup library.
MATCH_MASTER_SEED = 2026082003

#: Neither the pool the spent Phase 11 sealed bank drew from (`test`) nor
#: the pool Agent 1C trained on (`train`). Same choice, same reason, as the
#: Agent 2 diagnostic set.
MATCH_LIBRARY_SPLIT = "validation"

#: The four instructed opponents, imported rather than re-spelled: a match
#: opponent is literally an accepted Phase 11 stratum.
MATCH_STRATA = tuple(CORPUS_STRATA)
MATCH_SOURCES = tuple(CORPUS_SOURCES)
MATCH_COLORS = tuple(CORPUS_COLORS)

#: (setup source x player colour). Four boards per opponent per ordinal, so
#: eight games per opponent are balanced over both by construction rather
#: than by a post-hoc count.
BOARD_CELLS = tuple(
    (source, color) for source in MATCH_SOURCES for color in MATCH_COLORS
)

#: The instructed engineering target: 8 balanced-colour games per opponent
#: per arm, i.e. 32 games per arm over the four opponents.
GAMES_PER_OPPONENT = 8

#: Identities every match game carries.
PLAYER_POLICY_ID = "phase12_match_player_v1"
PHASE9_OPPONENT_POLICY_ID = "phase12_match_phase9_opponent_v1"

#: Seed domains. Four independent streams: the player's setup draw, the
#: opponent's setup draw, the match root seed and the per-ply search seed.
DOMAIN_PLAYER_SETUP = "match_player_setup"
DOMAIN_OPPONENT_SETUP = "match_opponent_setup"
DOMAIN_MATCH = "match_root"
DOMAIN_SEARCH = "match_search"
DOMAIN_PROBE = "match_probe"

MAX_ORDINAL_FORMAT = 99

_PLAYER_OF = {"red": RED, "blue": BLUE}
_COLOR_OF = {RED: "red", BLUE: "blue"}

_BOARD_ID_PATTERN = re.compile(
    rf"^phase12_match_v1\|ms=(?P<master>[0-9]+)"
    rf"\|st=(?P<stratum>{'|'.join(MATCH_STRATA)})"
    rf"\|src=(?P<source>{'|'.join(MATCH_SOURCES)})"
    rf"\|pl=(?P<color>{'|'.join(MATCH_COLORS)})"
    rf"\|g=(?P<ordinal>[0-9]{{2}})$"
)


class Phase12MatchError(Phase12SearchError):
    """A match game could not be planned, played or probed."""


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchArm:
    """One player under test.

    `kind` is `direct` for the accepted Phase 9 C1 seat with no search, and
    `search` for the engine over one belief provider. `diagnostic_only`
    marks the oracle arm, which may never stand in a production
    configuration and is reported as an upper bound rather than a peer.
    """

    arm_id: str
    kind: str
    label: str
    provider_id: "str | None" = None
    diagnostic_only: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("direct", "search"):
            raise Phase12MatchError(f"unknown arm kind {self.kind!r}")
        if (self.kind == "search") != (self.provider_id is not None):
            raise Phase12MatchError(
                f"arm {self.arm_id!r}: a search arm needs a provider and a direct "
                "arm must not name one"
            )

    def describe(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "kind": self.kind,
            "label": self.label,
            "provider_id": self.provider_id,
            "diagnostic_only": self.diagnostic_only,
        }


ARM_DIRECT = MatchArm("direct_c1", "direct", "direct accepted Phase 9 C1")
ARM_COUNT = MatchArm(
    "search_remaining_count", "search", "search + remaining_count", PROVIDER_REMAINING_COUNT
)
ARM_ORIGINAL = MatchArm(
    "search_original_phase11",
    "search",
    "search + original_phase11",
    PROVIDER_ORIGINAL_PHASE11,
)
ARM_AGENT1C = MatchArm(
    "search_agent1c", "search", "search + agent1c", PROVIDER_AGENT1C
)
ARM_ORACLE = MatchArm(
    "search_oracle",
    "search",
    "search + oracle (diagnostic)",
    PROVIDER_ORACLE,
    diagnostic_only=True,
)

#: Report order, and the order the run plays them in: the instruction's
#: A/B/C/D, then the optional oracle arm last.
PRODUCTION_ARMS = (ARM_DIRECT, ARM_COUNT, ARM_ORIGINAL, ARM_AGENT1C)
ALL_ARMS = PRODUCTION_ARMS + (ARM_ORACLE,)
ARMS_BY_ID = {arm.arm_id: arm for arm in ALL_ARMS}


# ---------------------------------------------------------------------------
# Identity and seeds
# ---------------------------------------------------------------------------


def board_id(stratum: str, source: str, player_color: str, ordinal: int) -> str:
    """The stable identifier of one *board*, shared by every arm.

    ```text
    phase12_match_v1|ms=2026082003|st=tactical_rule|src=p10d|pl=red|g=03
    ```

    It names the opponent, the setups and the seeds — everything the arms
    hold in common — and deliberately not the arm.
    """
    if stratum not in MATCH_STRATA:
        raise Phase12MatchError(f"unknown stratum {stratum!r}")
    if source not in MATCH_SOURCES:
        raise Phase12MatchError(f"unknown setup source {source!r}")
    if player_color not in MATCH_COLORS:
        raise Phase12MatchError(f"unknown player colour {player_color!r}")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise Phase12MatchError(f"ordinal must be an int, got {ordinal!r}")
    if not 0 <= ordinal <= MAX_ORDINAL_FORMAT:
        raise Phase12MatchError(f"ordinal {ordinal} outside 0..{MAX_ORDINAL_FORMAT}")
    identifier = (
        f"phase12_match_v1|ms={MATCH_MASTER_SEED}|st={stratum}|src={source}"
        f"|pl={player_color}|g={ordinal:02d}"
    )
    if _BOARD_ID_PATTERN.match(identifier) is None:  # pragma: no cover - defensive
        raise Phase12MatchError(f"constructed a malformed board id: {identifier!r}")
    return identifier


def parse_board_id(identifier: str) -> dict:
    """The identity fields of a board id, validated."""
    match = _BOARD_ID_PATTERN.match(identifier)
    if match is None:
        raise Phase12MatchError(f"malformed Phase 12 match board id: {identifier!r}")
    fields = match.groupdict()
    if int(fields["master"]) != MATCH_MASTER_SEED:
        raise Phase12MatchError(
            f"board id names master seed {fields['master']}, expected {MATCH_MASTER_SEED}"
        )
    return {
        "master_seed": int(fields["master"]),
        "stratum": fields["stratum"],
        "setup_source": fields["source"],
        "player_color": fields["color"],
        "ordinal": int(fields["ordinal"]),
    }


def match_seed_value(domain: str, identifier: str, ordinal: int = 0) -> int:
    """A 63-bit Phase 12 match stream value.

    The right shift matches the accepted phases' convention of handing
    policies and samplers a non-negative 63-bit seed.
    """
    return derive_phase12_seed(domain, identifier, int(ordinal)) >> 1


def search_seed_for(identifier: str, ply: int) -> int:
    """The world-sampling seed for one decision.

    A pure function of the board and the ply, so two arms that reach the
    same ply of the same board sample worlds under the same seed. The arm
    is not an input: a seed policy that varied by arm would make an arm's
    result partly a draw of the seed stream.
    """
    return match_seed_value(DOMAIN_SEARCH, identifier, ply)


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchGamePlan:
    """One board: opponent, setups and seeds, resolved from its identity."""

    board_id: str
    stratum: str
    setup_source: str
    player_color: str
    opponent_color: str
    ordinal: int
    cell_index: int
    match_seed: int
    red_setup: tuple
    blue_setup: tuple

    @property
    def player(self) -> int:
        return _PLAYER_OF[self.player_color]

    @property
    def opponent(self) -> int:
        return _PLAYER_OF[self.opponent_color]

    def describe(self) -> dict:
        return {
            "board_id": self.board_id,
            "stratum": self.stratum,
            "setup_source": self.setup_source,
            "player_color": self.player_color,
            "ordinal": self.ordinal,
            "match_seed": self.match_seed,
        }


def match_plan(
    stratum: str,
    source: str,
    player_color: str,
    ordinal: int,
    sources: Phase11BSetupSources,
) -> MatchGamePlan:
    """Resolve one board's setups and seeds from its identity alone.

    The player always draws from the accepted P10-D production source — the
    accepted phases' own observer-setup convention — and the opponent draws
    from the cell's source, so the setup source varies on the side whose
    behaviour the cell is meant to vary.
    """
    identifier = board_id(stratum, source, player_color, ordinal)
    opponent_color = "blue" if player_color == "red" else "red"
    player_setup = sources.draw(
        "p10d",
        MATCH_LIBRARY_SPLIT,
        player_color,
        match_seed_value(DOMAIN_PLAYER_SETUP, identifier),
    )
    opponent_setup = sources.draw(
        source,
        MATCH_LIBRARY_SPLIT,
        opponent_color,
        match_seed_value(DOMAIN_OPPONENT_SETUP, identifier),
    )
    red, blue = (
        (player_setup, opponent_setup)
        if player_color == "red"
        else (opponent_setup, player_setup)
    )
    return MatchGamePlan(
        board_id=identifier,
        stratum=stratum,
        setup_source=source,
        player_color=player_color,
        opponent_color=opponent_color,
        ordinal=ordinal,
        cell_index=BOARD_CELLS.index((source, player_color)),
        match_seed=match_seed_value(DOMAIN_MATCH, identifier),
        red_setup=tuple(red),
        blue_setup=tuple(blue),
    )


def match_plans(
    sources: "Phase11BSetupSources | None" = None,
    *,
    games_per_opponent: int = GAMES_PER_OPPONENT,
) -> "list[MatchGamePlan]":
    """Every board of the match set, ordinal-major.

    Ordinal-major means a run cut short after any whole ordinal is still
    balanced over opponents, setup sources and colours — a scaled-down
    match set rather than a biased corner of one.
    """
    if games_per_opponent < 1:
        raise Phase12MatchError("games_per_opponent must be at least 1")
    if games_per_opponent % len(BOARD_CELLS):
        raise Phase12MatchError(
            f"games_per_opponent must be a multiple of {len(BOARD_CELLS)} so the "
            "colour and setup-source balance is exact"
        )
    if sources is None:
        sources = Phase11BSetupSources()
    ordinals = games_per_opponent // len(BOARD_CELLS)
    plans = []
    for ordinal in range(ordinals):
        for stratum in MATCH_STRATA:
            for source, color in BOARD_CELLS:
                plans.append(match_plan(stratum, source, color, ordinal, sources))
    return plans


# ---------------------------------------------------------------------------
# Seats
# ---------------------------------------------------------------------------


def player_ref() -> PolicyRef:
    """The arm-independent player identity. See the module docstring."""
    return PolicyRef(policy_id=PLAYER_POLICY_ID, policy_version=MATCH_VERSION)


def build_spec(plan: MatchGamePlan, opponent: PolicyRef) -> MatchSpec:
    return MatchSpec(
        candidate=player_ref(),
        opponent=opponent,
        setup_pair_id=plan.cell_index,
        candidate_color=plan.player,
        replicate=plan.ordinal,
        root_seed=plan.match_seed,
        suite_version=MATCH_VERSION,
        setup_bank_version=(
            f"{MATCH_VERSION}|st={plan.stratum}|src={plan.setup_source}"
        ),
        rules=EVALUATION_RULES,
    )


def single_game_bank(spec: MatchSpec, plan: MatchGamePlan) -> SetupBank:
    """The one-pair bank this board resolves through.

    The driver could pass the setups straight to `create_game`; routing
    them through the accepted bank keeps the setups inside the accepted
    identity apparatus, so a stored row names the board it was played on.
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


def opponent_seat(plan: MatchGamePlan, owners: dict):
    """`(ref, policy)` for this board's opponent behaviour group."""
    policy_id = STRATUM_POLICY_IDS[plan.stratum]
    if policy_id is None:
        ref = PolicyRef(
            policy_id=PHASE9_OPPONENT_POLICY_ID, policy_version=MATCH_VERSION
        )
        return ref, RemoteNeuralPolicy(
            ref,
            LocalInferenceChannel(owners["phase9"]),
            decision_mode=DECISION_MODE_GREEDY,
        )
    policy = build_policy(policy_id)
    return policy.ref, FrozenSeedPolicy(policy, plan.match_seed)


class DirectSeat:
    """Arm A: the accepted Phase 9 greedy seat, no search.

    Decides through `build_policy_input` and `decide_checked` — the same
    two calls `play_match` makes — so arm A is the accepted direct player
    and not a re-implementation of it.
    """

    kind = "direct"

    def __init__(self, arm: MatchArm, owners: dict) -> None:
        if arm.kind != "direct":  # pragma: no cover - guarded by the caller
            raise Phase12MatchError(f"{arm.arm_id!r} is not a direct arm")
        self.arm = arm
        self.ref = player_ref()
        self.policy = RemoteNeuralPolicy(
            self.ref,
            LocalInferenceChannel(owners["phase9"]),
            decision_mode=DECISION_MODE_GREEDY,
        )

    def decide(self, state: GameState, legal, spec: MatchSpec, plan: MatchGamePlan):
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
        }

    def describe(self) -> dict:
        return {
            "arm": self.arm.describe(),
            "seat": "accepted Phase 9 C1, greedy, one forward per decision",
            "policy": self.policy.describe(),
        }


class SearchSeat:
    """Arms B-E: :class:`Phase12SearchEngine` over one belief provider."""

    kind = "search"

    def __init__(self, arm: MatchArm, engine: Phase12SearchEngine) -> None:
        if arm.kind != "search":  # pragma: no cover - guarded by the caller
            raise Phase12MatchError(f"{arm.arm_id!r} is not a search arm")
        if engine.provider.provider_id != arm.provider_id:
            raise Phase12MatchError(
                f"arm {arm.arm_id!r} wants provider {arm.provider_id!r} but the "
                f"engine carries {engine.provider.provider_id!r}"
            )
        self.arm = arm
        self.engine = engine

    def decide(self, state: GameState, legal, spec: MatchSpec, plan: MatchGamePlan):
        seed = search_seed_for(plan.board_id, int(state.total_moves))
        decision = self.engine.choose_action(state, seed=seed)
        return int(decision.selected_action_id), {
            "ply": int(state.total_moves),
            "seconds": float(decision.seconds),
            "legal_actions": int(decision.legal_action_count),
            "move_changed": bool(decision.move_changed),
            "c1_forwards": int(decision.c1_forwards),
            "unique_worlds": int(decision.unique_worlds),
            "candidates": len(decision.candidates),
            "forward_seconds": float(decision.forward_seconds),
            "direct_action_id": int(decision.direct_action_id),
        }

    def describe(self) -> dict:
        return {"arm": self.arm.describe(), "seat": self.engine.describe()}


# ---------------------------------------------------------------------------
# The match-time boundary probe
# ---------------------------------------------------------------------------


@dataclass
class SeatProbe:
    """Two independent checks on a sample of the seat's own decisions.

    `permutation`: re-decide the position on a state whose hidden opponent
    identities have been permuted by the accepted
    `permute_hidden_identities`. The public surface is invariant under that
    transformation, so a seat that reads only public information must
    answer identically; one that read a hidden rank would not.

    The oracle arm is the positive control for exactly that check. It reads
    the true world by design, so `expects_hidden_truth=True` inverts the
    reading: a changed answer is the expected outcome and is counted as
    sensitivity rather than recorded as a failure. A probe that never fired
    on the oracle would be a probe with no power, and the report says so
    either way.

    `direct agreement`: require the engine's internal direct Phase 9 action
    to equal the accepted `RemoteNeuralPolicy` decision on the same
    position, which is what makes "search changed the move" mean "changed
    it relative to arm A".

    Both are sampled, not exhaustive: each costs a whole extra decision,
    and the invariant they check is structural, not statistical.
    """

    reference: "RemoteNeuralPolicy | None" = None
    interval: int = 24
    budget: int = 32
    expects_hidden_truth: bool = False
    permutation_checks: int = 0
    permutation_changed: int = 0
    permutation_sensitive: int = 0
    direct_checks: int = 0
    failures: list = field(default_factory=list)

    def due(self, decision_index: int) -> bool:
        """Whether to spend a probe on this decision.

        Never decision 0: the opening position is the same in every game of
        a colour, and a check that only ever ran there would be one check
        repeated, not a sample of the seat's real decisions.
        """
        return (
            self.budget > 0
            and self.interval > 0
            and decision_index > 0
            and decision_index % self.interval == 0
        )

    def run(self, seat, state: GameState, legal, spec, plan, action: int, record: dict) -> None:
        self.budget -= 1
        player = state.acting_player

        rng = random.Random(
            match_seed_value(DOMAIN_PROBE, plan.board_id, int(state.total_moves))
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
                        "arm_id": seat.arm.arm_id,
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
            if accepted != int(record["direct_action_id"]):
                self.failures.append(
                    {
                        "check": "direct_action_agreement",
                        "board_id": plan.board_id,
                        "arm_id": seat.arm.arm_id,
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
    stratum: str
    setup_source: str
    player_color: str
    ordinal: int
    match_id: str
    opponent_policy: str
    outcome: str
    effective_score: float
    winner: "str | None"
    terminal_reason: str
    plies: int
    player_decisions: int
    seconds: float
    player_seconds: float
    move_changes: int
    c1_forwards: int
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
        """The flat record a results table stores."""
        return {
            "board_id": self.board_id,
            "arm_id": self.arm_id,
            "stratum": self.stratum,
            "setup_source": self.setup_source,
            "player_color": self.player_color,
            "ordinal": self.ordinal,
            "match_id": self.match_id,
            "opponent_policy": self.opponent_policy,
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
            "c1_forwards": self.c1_forwards,
        }


def outcome_of(score: float) -> str:
    if score > 0.5:
        return "win"
    if score < 0.5:
        return "loss"
    return "draw"


def play_arm_game(
    plan: MatchGamePlan,
    seat,
    owners: dict,
    *,
    probe: "SeatProbe | None" = None,
    keep_moves: bool = True,
) -> GameRecord:
    """Play one board with one arm in the player seat.

    The opponent decides through the accepted observer-safe path; the
    player seat decides through its own interface (a `PolicyInput` for arm
    A, the search engine for the rest). Every terminal fact returned is the
    accepted engine's own.
    """
    opponent_reference, opponent_policy = opponent_seat(plan, owners)
    spec = build_spec(plan, opponent_reference)
    bank = single_game_bank(spec, plan)
    red_setup, blue_setup = spec.resolve_setups(bank)
    state = create_game(
        red_setup, blue_setup, rules=spec.rules, game_id=spec.game_id
    )

    player = plan.player
    moves: list = []
    player_seconds = 0.0
    move_changes = 0
    c1_forwards = 0
    started = time.perf_counter()

    while not state.terminal:
        actor = state.acting_player
        legal = legal_actions(state)
        if actor == player:
            action, record = seat.decide(state, legal, spec, plan)
            if action not in legal:
                raise Phase12MatchError(
                    f"{seat.arm.arm_id} selected illegal action {action} at ply "
                    f"{state.total_moves} of {plan.board_id}"
                )
            player_seconds += float(record["seconds"])
            c1_forwards += int(record["c1_forwards"] or 0)
            if record["move_changed"]:
                move_changes += 1
            if probe is not None and probe.due(len(moves)):
                probe.run(seat, state, legal, spec, plan, action, record)
            if keep_moves:
                moves.append(record)
            else:
                moves.append({"move_changed": record["move_changed"]})
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
    winner = None if state.winner is None else _COLOR_OF[state.winner]
    return GameRecord(
        board_id=plan.board_id,
        arm_id=seat.arm.arm_id,
        stratum=plan.stratum,
        setup_source=plan.setup_source,
        player_color=plan.player_color,
        ordinal=plan.ordinal,
        match_id=spec.match_id,
        opponent_policy=opponent_reference.token,
        outcome=outcome_of(score),
        effective_score=score,
        winner=winner,
        terminal_reason=str(state.terminal_reason),
        plies=int(state.total_moves),
        player_decisions=len(moves),
        seconds=seconds,
        player_seconds=player_seconds,
        move_changes=move_changes,
        c1_forwards=c1_forwards,
        moves=tuple(moves),
    )


__all__ = [
    "ALL_ARMS",
    "ARMS_BY_ID",
    "ARM_AGENT1C",
    "ARM_COUNT",
    "ARM_DIRECT",
    "ARM_ORACLE",
    "ARM_ORIGINAL",
    "BOARD_CELLS",
    "DirectSeat",
    "GAMES_PER_OPPONENT",
    "GameRecord",
    "MATCH_LIBRARY_SPLIT",
    "MATCH_MASTER_SEED",
    "MATCH_VERSION",
    "MatchArm",
    "MatchGamePlan",
    "PRODUCTION_ARMS",
    "Phase12MatchError",
    "SearchSeat",
    "SeatProbe",
    "board_id",
    "build_spec",
    "match_plan",
    "match_plans",
    "match_seed_value",
    "opponent_seat",
    "outcome_of",
    "parse_board_id",
    "play_arm_game",
    "player_ref",
    "search_seed_for",
    "single_game_bank",
]
