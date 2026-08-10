"""Deterministic match execution and reproducible parallel evaluation.

Specification sources:

- `03_game_engine_spec.md` sections 11-12 (transitions, replay)
- `09_public_event_and_replay_schema.md` sections 3, 13 (replay record)
- Phase 4 Agent 3 instructions ("Match execution", "Parallel execution",
  "Raw match result")

What this module is
-------------------
The evaluation engine: it turns a :class:`~stratego.evaluation.match_spec.MatchSpec`
into a played game and a raw :class:`MatchResult`, and runs a whole schedule
across worker processes without letting the worker count touch a single game.

It adds no rules. `legal_actions` decides what may be played, `apply_action`
applies it, and the engine's terminal precedence decides how the game ends. The
runner's only judgements are about *process*: whose turn it is, what the policy
is allowed to see, and whether a policy honoured its contract.

Why the worker count cannot change a result
-------------------------------------------
Every input to a game is fixed by :class:`MatchSpec` before dispatch:

- the setups come from `setup_pair_id` in a versioned bank;
- the colour assignment is part of `match_id`;
- both policy seeds are derived from `match_id` alone;
- each per-ply decision seed is derived from the policy seed and the ply.

Nothing in that list can see a worker index, a shard boundary, a submission
order or a clock. So parallelism is a scheduling concern only, and the runner
deliberately keeps it that way: workers receive already-built match identities,
never the parameters from which identities are computed.

Reproducibility and wall-clock timing
-------------------------------------
`wall_clock_seconds` genuinely differs between runs, which makes a naive
whole-row comparison useless for reproducibility checks. :meth:`MatchResult.comparable`
is therefore the canonical form for equality: everything the game determines,
with timing and the sidecar replay reference dropped. :func:`results_digest` is
built from it and is order-independent, so two runs at different worker counts
can be compared with one string.

Policy failures
---------------
A failing policy is never papered over with a substitute legal move. The default
`on_policy_error="raise"` aborts the run at the first violation. The opt-in
`"quarantine"` mode instead records the failure on the row, marks the match
`error` with **no** win/draw/loss contribution, and lets a long league finish;
:mod:`stratego.evaluation.statistics` then refuses to summarise a result set
containing an errored match unless the caller explicitly acknowledges it. Either
way the failure is loud and the match produces no score.
"""

import hashlib
import json
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from ..engine.constants import (
    BLUE,
    EVENT_SCHEMA_VERSION,
    OBSERVATION_VERSION,
    PLAYER_NAMES,
    RED,
    REPLAY_VERSION,
    RulesConfig,
)
from ..engine.invariants import check_invariants
from ..engine.legal_moves import legal_actions
from ..engine.replay import ReplayRecord, build_replay_record
from ..engine.setup import deserialize_setup, serialize_setup
from ..engine.state import create_game
from ..engine.transition import apply_action
from .match_spec import (
    MatchSpec,
    MatchSpecError,
    rules_token,
    schedule_digest,
    shard_schedule,
)
from .policy import (
    Policy,
    PolicyContractError,
    PolicyRef,
    PolicyResult,
    build_policy_input,
    validate_policy_result,
)
from .registry import build_policy
from .setup_bank import SetupBank

MATCH_RESULT_SCHEMA_VERSION = "match_result_v1"
MATCH_RUNNER_VERSION = "match_runner_v1"

#: Candidate outcome labels. `error` is not an outcome of play -- it means the
#: game was abandoned because a policy broke its contract, and it carries no
#: score.
RESULT_WIN = "win"
RESULT_LOSS = "loss"
RESULT_DRAW = "draw"
RESULT_ERROR = "error"
RESULT_LABELS = (RESULT_WIN, RESULT_LOSS, RESULT_DRAW, RESULT_ERROR)

#: Terminal reason recorded for a quarantined match. Deliberately not one of the
#: engine's `TERMINAL_REASONS`, so an errored row can never be mistaken for a
#: game that actually finished.
TERMINAL_POLICY_ERROR = "policy_error"

ON_POLICY_ERROR_RAISE = "raise"
ON_POLICY_ERROR_QUARANTINE = "quarantine"
ON_POLICY_ERROR_MODES = (ON_POLICY_ERROR_RAISE, ON_POLICY_ERROR_QUARANTINE)

#: Failure categories, so `illegal_policy_actions` can be counted separately
#: from other contract breaches rather than inferred from a message.
ERROR_ILLEGAL_ACTION = "illegal_action"
ERROR_CONTRACT_VIOLATION = "contract_violation"
ERROR_POLICY_EXCEPTION = "policy_exception"
ERROR_ENGINE_REJECTED = "engine_rejected_action"
ERROR_CATEGORIES = (
    ERROR_ILLEGAL_ACTION,
    ERROR_CONTRACT_VIOLATION,
    ERROR_POLICY_EXCEPTION,
    ERROR_ENGINE_REJECTED,
)


class MatchRunnerError(RuntimeError):
    """Raised when a match cannot be played or a stored result is inconsistent."""


class PolicyFailure(RuntimeError):
    """A policy broke its contract during a match.

    Carries the classification the raw result needs. The original exception is
    always chained, so `raise` mode loses nothing.
    """

    def __init__(
        self,
        message: str,
        *,
        category: str,
        policy: PolicyRef,
        role: str,
        ply: int,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.policy = policy
        self.role = role
        self.ply = ply


# ---------------------------------------------------------------------------
# Raw match result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchResult:
    """One played game, in the form the statistics and reports consume.

    The row is deliberately self-sufficient: it carries both setups, so a game
    can be rebuilt and replayed from the row alone, without the setup bank that
    produced it. That costs 80 characters per row and removes the only external
    dependency a stored result would otherwise have.

    `action_history` is optional because a large league would otherwise store
    tens of millions of integers inline. `replay_digest` is always present, so a
    divergence is always detectable even when the histories live in a sidecar
    file.
    """

    # -- identity ----------------------------------------------------------
    match_id: str
    paired_unit_id: str
    suite_version: str
    pairing_mode: str
    candidate_policy_id: str
    candidate_policy_version: str
    opponent_policy_id: str
    opponent_policy_version: str
    candidate_color: int
    setup_pair_id: int
    setup_bank_version: str
    replicate: int
    root_seed: int
    candidate_seed: int
    opponent_seed: int
    rules: str
    rules_payload: Mapping[str, Any]

    # -- position ----------------------------------------------------------
    red_setup: str
    blue_setup: str
    first_player: int

    # -- outcome -----------------------------------------------------------
    winner: int | None
    draw: bool
    candidate_result: str
    candidate_score: float | None
    terminal_reason: str
    plies: int
    decisions: int

    # -- replay ------------------------------------------------------------
    replay_digest: str
    action_history: tuple[int, ...] | None = None
    replay_reference: str | None = None

    # -- diagnostics -------------------------------------------------------
    wall_clock_seconds: float = 0.0
    policy_error: str | None = None
    policy_error_category: str | None = None
    policy_error_role: str | None = None
    policy_error_policy: str | None = None
    policy_error_ply: int | None = None

    schema_version: str = MATCH_RESULT_SCHEMA_VERSION
    runner_version: str = MATCH_RUNNER_VERSION

    def __post_init__(self) -> None:
        if self.candidate_result not in RESULT_LABELS:
            raise MatchRunnerError(f"unknown candidate_result {self.candidate_result!r}")
        if self.errored:
            if self.candidate_score is not None:
                raise MatchRunnerError("an errored match must not carry a score")
            if self.policy_error is None:
                raise MatchRunnerError("an errored match must carry a policy_error")
        elif self.candidate_score is None:
            raise MatchRunnerError(f"match {self.match_id} has no score and no error")

    # -- derived -----------------------------------------------------------

    @property
    def errored(self) -> bool:
        return self.candidate_result == RESULT_ERROR

    @property
    def scored(self) -> bool:
        """Whether this row contributes to win/draw/loss statistics."""
        return not self.errored

    @property
    def opponent_color(self) -> int:
        return BLUE if self.candidate_color == RED else RED

    @property
    def candidate_color_name(self) -> str:
        return PLAYER_NAMES[self.candidate_color]

    @property
    def winner_name(self) -> str | None:
        return None if self.winner is None else PLAYER_NAMES[self.winner]

    @property
    def candidate(self) -> PolicyRef:
        return PolicyRef(self.candidate_policy_id, self.candidate_policy_version)

    @property
    def opponent(self) -> PolicyRef:
        return PolicyRef(self.opponent_policy_id, self.opponent_policy_version)

    @property
    def matchup(self) -> str:
        """`candidate@version vs opponent@version`, for grouping summaries."""
        return f"{self.candidate.token} vs {self.opponent.token}"

    def rules_config(self) -> RulesConfig:
        """Rebuild the exact rules this game was played under.

        The reconstructed configuration is checked against the stored token, so a
        tampered or truncated payload fails here rather than silently replaying
        the game under different limits.
        """
        config = RulesConfig(**dict(self.rules_payload))
        if rules_token(config) != self.rules:
            raise MatchRunnerError(
                f"match {self.match_id}: rules_payload rebuilds to {rules_token(config)!r} "
                f"but the row stores {self.rules!r}"
            )
        return config

    def spec(self) -> MatchSpec:
        """The specification this result came from, with its identity re-verified."""
        spec = MatchSpec(
            candidate=self.candidate,
            opponent=self.opponent,
            setup_pair_id=self.setup_pair_id,
            candidate_color=self.candidate_color,
            replicate=self.replicate,
            root_seed=self.root_seed,
            suite_version=self.suite_version,
            setup_bank_version=self.setup_bank_version,
            pairing_mode=self.pairing_mode,
            rules=self.rules_config(),
        )
        if spec.match_id != self.match_id:
            raise MatchRunnerError(
                f"stored match_id {self.match_id!r} does not match the specification "
                f"rebuilt from this row ({spec.match_id!r})"
            )
        return spec

    # -- comparison --------------------------------------------------------

    def comparable(self) -> dict:
        """Everything the game determines, with timing and file references removed.

        This is the canonical form for reproducibility comparison. `action_history`
        is included when present but is not required, because `replay_digest`
        already covers it: a run that stored histories and a run that did not
        remain comparable on every field either has.
        """
        payload = {
            "match_id": self.match_id,
            "paired_unit_id": self.paired_unit_id,
            "suite_version": self.suite_version,
            "pairing_mode": self.pairing_mode,
            "candidate": self.candidate.token,
            "opponent": self.opponent.token,
            "candidate_color": self.candidate_color,
            "setup_pair_id": self.setup_pair_id,
            "setup_bank_version": self.setup_bank_version,
            "replicate": self.replicate,
            "root_seed": self.root_seed,
            "candidate_seed": self.candidate_seed,
            "opponent_seed": self.opponent_seed,
            "rules": self.rules,
            "red_setup": self.red_setup,
            "blue_setup": self.blue_setup,
            "first_player": self.first_player,
            "winner": self.winner,
            "draw": self.draw,
            "candidate_result": self.candidate_result,
            "candidate_score": self.candidate_score,
            "terminal_reason": self.terminal_reason,
            "plies": self.plies,
            "decisions": self.decisions,
            "replay_digest": self.replay_digest,
            "policy_error_category": self.policy_error_category,
            "policy_error_role": self.policy_error_role,
            "policy_error_ply": self.policy_error_ply,
        }
        if self.action_history is not None:
            payload["action_history"] = list(self.action_history)
        return payload

    def comparable_digest(self) -> str:
        """Digest of the reproducible content, excluding the inline action history.

        The history is dropped deliberately: `replay_digest` is part of the
        payload and already covers every action, so excluding it makes the digest
        -- and therefore :func:`results_digest` -- identical whether a run stored
        histories inline or left them in a sidecar. Two runs at different worker
        counts are then comparable on one string regardless of that choice.
        """
        payload = self.comparable()
        payload.pop("action_history", None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        payload = {
            "schema_version": self.schema_version,
            "runner_version": self.runner_version,
            "match_id": self.match_id,
            "paired_unit_id": self.paired_unit_id,
            "suite_version": self.suite_version,
            "pairing_mode": self.pairing_mode,
            "candidate_policy_id": self.candidate_policy_id,
            "candidate_policy_version": self.candidate_policy_version,
            "opponent_policy_id": self.opponent_policy_id,
            "opponent_policy_version": self.opponent_policy_version,
            "candidate_color": self.candidate_color,
            "candidate_color_name": self.candidate_color_name,
            "setup_pair_id": self.setup_pair_id,
            "setup_bank_version": self.setup_bank_version,
            "replicate": self.replicate,
            "root_seed": self.root_seed,
            "candidate_seed": self.candidate_seed,
            "opponent_seed": self.opponent_seed,
            "rules": self.rules,
            "rules_payload": dict(self.rules_payload),
            "red_setup": self.red_setup,
            "blue_setup": self.blue_setup,
            "first_player": self.first_player,
            "winner": self.winner,
            "winner_name": self.winner_name,
            "draw": self.draw,
            "candidate_result": self.candidate_result,
            "candidate_score": self.candidate_score,
            "terminal_reason": self.terminal_reason,
            "plies": self.plies,
            "decisions": self.decisions,
            "replay_digest": self.replay_digest,
            "action_history": None if self.action_history is None else list(self.action_history),
            "replay_reference": self.replay_reference,
            "wall_clock_seconds": self.wall_clock_seconds,
            "policy_error": self.policy_error,
            "policy_error_category": self.policy_error_category,
            "policy_error_role": self.policy_error_role,
            "policy_error_policy": self.policy_error_policy,
            "policy_error_ply": self.policy_error_ply,
        }
        return payload

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "MatchResult":
        history = payload.get("action_history")
        return MatchResult(
            match_id=str(payload["match_id"]),
            paired_unit_id=str(payload["paired_unit_id"]),
            suite_version=str(payload["suite_version"]),
            pairing_mode=str(payload["pairing_mode"]),
            candidate_policy_id=str(payload["candidate_policy_id"]),
            candidate_policy_version=str(payload["candidate_policy_version"]),
            opponent_policy_id=str(payload["opponent_policy_id"]),
            opponent_policy_version=str(payload["opponent_policy_version"]),
            candidate_color=int(payload["candidate_color"]),
            setup_pair_id=int(payload["setup_pair_id"]),
            setup_bank_version=str(payload["setup_bank_version"]),
            replicate=int(payload["replicate"]),
            root_seed=int(payload["root_seed"]),
            candidate_seed=int(payload["candidate_seed"]),
            opponent_seed=int(payload["opponent_seed"]),
            rules=str(payload["rules"]),
            rules_payload=dict(payload["rules_payload"]),
            red_setup=str(payload["red_setup"]),
            blue_setup=str(payload["blue_setup"]),
            first_player=int(payload["first_player"]),
            winner=None if payload["winner"] is None else int(payload["winner"]),
            draw=bool(payload["draw"]),
            candidate_result=str(payload["candidate_result"]),
            candidate_score=(
                None if payload["candidate_score"] is None else float(payload["candidate_score"])
            ),
            terminal_reason=str(payload["terminal_reason"]),
            plies=int(payload["plies"]),
            decisions=int(payload["decisions"]),
            replay_digest=str(payload["replay_digest"]),
            action_history=None if history is None else tuple(int(a) for a in history),
            replay_reference=payload.get("replay_reference"),
            wall_clock_seconds=float(payload.get("wall_clock_seconds", 0.0)),
            policy_error=payload.get("policy_error"),
            policy_error_category=payload.get("policy_error_category"),
            policy_error_role=payload.get("policy_error_role"),
            policy_error_policy=payload.get("policy_error_policy"),
            policy_error_ply=(
                None
                if payload.get("policy_error_ply") is None
                else int(payload["policy_error_ply"])
            ),
            schema_version=str(payload.get("schema_version", MATCH_RESULT_SCHEMA_VERSION)),
            runner_version=str(payload.get("runner_version", MATCH_RUNNER_VERSION)),
        )

    # -- replay ------------------------------------------------------------

    def replay_record(self) -> ReplayRecord:
        """The engine replay record for this game, built from the row alone.

        Requires `action_history`; a digest-only row must fetch its history from
        the sidecar file first.
        """
        if self.action_history is None:
            raise MatchRunnerError(
                f"match {self.match_id} stores no action history; load it from "
                f"{self.replay_reference or 'the replay sidecar'} before replaying"
            )
        config = self.rules_config()
        return ReplayRecord(
            replay_version=REPLAY_VERSION,
            rules_version=config.rules_version,
            observation_version=OBSERVATION_VERSION,
            event_schema_version=EVENT_SCHEMA_VERSION,
            game_id=self.match_id,
            red_setup=self.red_setup,
            blue_setup=self.blue_setup,
            first_player=PLAYER_NAMES[config.first_player],
            battleless_move_limit=config.battleless_move_limit,
            absolute_move_limit=config.absolute_move_limit,
            rules_context=config.context,
            actions=list(self.action_history),
            terminal_result=_terminal_result_from_row(self),
            terminal_reason=self.terminal_reason,
            total_moves=self.plies,
            seeds={
                "candidate_seed": self.candidate_seed,
                "opponent_seed": self.opponent_seed,
                "root_seed": self.root_seed,
            },
        )

    def with_replay_reference(self, reference: str | None) -> "MatchResult":
        return replace(self, replay_reference=reference)

    def without_action_history(self) -> "MatchResult":
        """The same row with its inline history dropped. The digest is unaffected."""
        return replace(self, action_history=None)


def _terminal_result_from_row(result: MatchResult) -> str:
    if result.errored:
        return "unfinished"
    if result.winner is None:
        return "draw"
    return f"{PLAYER_NAMES[result.winner]}_win"


def replay_digest(record: ReplayRecord) -> str:
    """SHA-256 over a replay record's canonical JSON.

    The record already contains both setups, the full rules configuration, the
    ordered action list and the terminal outcome, so this single string covers
    everything the reproducibility gate requires: setups, action history,
    result, terminal reason and ply count.
    """
    return hashlib.sha256(record.to_json().encode()).hexdigest()


# ---------------------------------------------------------------------------
# One decision
# ---------------------------------------------------------------------------


def _decide(policy: Policy, request, role: str) -> PolicyResult:
    """One classified, contract-checked decision.

    Mirrors :meth:`Policy.decide_checked` but distinguishes *how* a policy
    failed, so `illegal_policy_actions` can be counted directly instead of
    inferred from an exception message. `validate_policy_result` still runs last
    and remains authoritative: a violation this function fails to classify is
    raised by Agent 1's validator, not swallowed.
    """
    ref = request.policy
    if ref != policy.ref:
        raise PolicyFailure(
            f"request addressed to {ref.token} was handed to {policy.ref.token}",
            category=ERROR_CONTRACT_VIOLATION,
            policy=ref,
            role=role,
            ply=request.ply,
        )

    try:
        raw = policy.decide(request)
    except Exception as error:  # noqa: BLE001 -- classified and re-raised below
        raise PolicyFailure(
            f"policy {ref.token} raised {type(error).__name__} at ply {request.ply}: {error}",
            category=ERROR_POLICY_EXCEPTION,
            policy=ref,
            role=role,
            ply=request.ply,
        ) from error

    if not isinstance(raw, PolicyResult):
        raise PolicyFailure(
            f"policy {ref.token} returned {type(raw).__name__}, expected PolicyResult",
            category=ERROR_CONTRACT_VIOLATION,
            policy=ref,
            role=role,
            ply=request.ply,
        )
    if raw.selected_action_id not in request.legal_actions:
        raise PolicyFailure(
            f"policy {ref.token} selected illegal action {raw.selected_action_id} at ply "
            f"{request.ply} of match {request.match_id!r}",
            category=ERROR_ILLEGAL_ACTION,
            policy=ref,
            role=role,
            ply=request.ply,
        )

    try:
        return validate_policy_result(raw, request)
    except PolicyContractError as error:
        raise PolicyFailure(
            str(error),
            category=ERROR_CONTRACT_VIOLATION,
            policy=ref,
            role=role,
            ply=request.ply,
        ) from error


# ---------------------------------------------------------------------------
# One match
# ---------------------------------------------------------------------------


def resolve_policies(
    spec: MatchSpec, policies: "Mapping[str, Policy] | None" = None
) -> dict[str, Policy]:
    """The two policy objects this match needs, keyed by `id@version` token.

    Falls back to the Phase 4 catalogue, and checks the catalogue's version
    against the version named in the match identity: a silently re-versioned
    policy would otherwise play games recorded under the old identifier.
    """
    resolved: dict[str, Policy] = {}
    for ref in (spec.candidate, spec.opponent):
        if policies is not None and ref.token in policies:
            resolved[ref.token] = policies[ref.token]
            continue
        policy = build_policy(ref.policy_id)
        if policy.ref != ref:
            raise MatchRunnerError(
                f"match {spec.match_id} names {ref.token} but the catalogue provides "
                f"{policy.ref.token}; a stored schedule cannot be replayed against a "
                "different policy version"
            )
        resolved[ref.token] = policy
    return resolved


def play_match(
    spec: MatchSpec,
    *,
    bank: "SetupBank | None" = None,
    setups: "tuple[Sequence[int], Sequence[int]] | None" = None,
    policies: "Mapping[str, Policy] | None" = None,
    record_actions: bool = True,
    on_policy_error: str = ON_POLICY_ERROR_RAISE,
    verify_invariants: bool = False,
) -> MatchResult:
    """Play one fully determined game and return its raw result.

    Exactly one of `bank` or `setups` supplies the position. `setups` exists so a
    stored row can be re-played without rebuilding a 1,024-pair bank.

    The policies see only :func:`build_policy_input` products, and only the ones
    they declared in `PolicyRequirements` -- an undeclared observation is never
    built, which is most of why a league is affordable.
    """
    if on_policy_error not in ON_POLICY_ERROR_MODES:
        raise MatchRunnerError(
            f"unknown on_policy_error mode {on_policy_error!r}; expected one of "
            f"{', '.join(ON_POLICY_ERROR_MODES)}"
        )
    if (bank is None) == (setups is None):
        raise MatchRunnerError("play_match needs exactly one of `bank` or `setups`")

    if bank is not None:
        red_setup, blue_setup = spec.resolve_setups(bank)
    else:
        red_setup, blue_setup = (tuple(setups[0]), tuple(setups[1]))

    resolved = resolve_policies(spec, policies)
    started = time.perf_counter()

    state = create_game(red_setup, blue_setup, rules=spec.rules, game_id=spec.game_id)
    failure: PolicyFailure | None = None
    decisions = 0

    while not state.terminal:
        actor = state.acting_player
        ref = spec.policy_ref_for(actor)
        policy = resolved[ref.token]
        legal = legal_actions(state)
        request = build_policy_input(
            state,
            policy=ref,
            policy_seed=spec.policy_seed_for(actor),
            requirements=policy.requirements,
            suite_version=spec.suite_version,
            match_id=spec.match_id,
            paired_unit_id=spec.paired_unit_id,
            legal=legal,
        )
        try:
            result = _decide(policy, request, spec.role_for(actor))
        except PolicyFailure as error:
            if on_policy_error == ON_POLICY_ERROR_RAISE:
                raise
            failure = error
            break

        decisions += 1
        try:
            # The engine is the final legality authority; `_decide` only catches a
            # broken policy earlier and with a better message.
            apply_action(state, result.selected_action_id, legal=legal)
        except Exception as error:  # noqa: BLE001 -- classified and re-raised
            wrapped = PolicyFailure(
                f"engine rejected action {result.selected_action_id} from {ref.token} at "
                f"ply {request.ply}: {error}",
                category=ERROR_ENGINE_REJECTED,
                policy=ref,
                role=spec.role_for(actor),
                ply=request.ply,
            )
            if on_policy_error == ON_POLICY_ERROR_RAISE:
                raise wrapped from error
            failure = wrapped
            break

        if verify_invariants:
            check_invariants(state)

    elapsed = time.perf_counter() - started
    return _build_result(
        spec,
        state,
        red_setup=red_setup,
        blue_setup=blue_setup,
        decisions=decisions,
        elapsed=elapsed,
        record_actions=record_actions,
        failure=failure,
    )


def _build_result(
    spec: MatchSpec,
    state,
    *,
    red_setup: "Sequence[int]",
    blue_setup: "Sequence[int]",
    decisions: int,
    elapsed: float,
    record_actions: bool,
    failure: "PolicyFailure | None",
) -> MatchResult:
    history = tuple(state.action_history)
    record = build_replay_record(
        state,
        tuple(red_setup),
        tuple(blue_setup),
        seeds={
            "candidate_seed": spec.candidate_seed,
            "opponent_seed": spec.opponent_seed,
            "root_seed": spec.root_seed,
        },
    )
    if failure is not None:
        # A quarantined game is unfinished, so the record must say so rather than
        # inherit `not_terminal` semantics from a state that never ended.
        record = replace(
            record, terminal_result="unfinished", terminal_reason=TERMINAL_POLICY_ERROR
        )

    if failure is not None:
        candidate_result = RESULT_ERROR
        candidate_score = None
        winner = None
        draw = False
        terminal_reason = TERMINAL_POLICY_ERROR
    else:
        if not state.terminal:  # pragma: no cover -- the loop only exits on terminal
            raise MatchRunnerError(
                f"match {spec.match_id} left the play loop in a non-terminal state"
            )
        winner = state.winner
        draw = bool(state.is_draw or state.winner is None)
        candidate_score = state.effective_score_for(spec.candidate_color)
        candidate_result = (
            RESULT_DRAW
            if candidate_score == 0.5
            else (RESULT_WIN if candidate_score == 1.0 else RESULT_LOSS)
        )
        terminal_reason = state.terminal_reason

    return MatchResult(
        match_id=spec.match_id,
        paired_unit_id=spec.paired_unit_id,
        suite_version=spec.suite_version,
        pairing_mode=spec.pairing_mode,
        candidate_policy_id=spec.candidate.policy_id,
        candidate_policy_version=spec.candidate.policy_version,
        opponent_policy_id=spec.opponent.policy_id,
        opponent_policy_version=spec.opponent.policy_version,
        candidate_color=spec.candidate_color,
        setup_pair_id=spec.setup_pair_id,
        setup_bank_version=spec.setup_bank_version,
        replicate=spec.replicate,
        root_seed=spec.root_seed,
        candidate_seed=spec.candidate_seed,
        opponent_seed=spec.opponent_seed,
        rules=rules_token(spec.rules),
        rules_payload=_rules_payload(spec.rules),
        red_setup=serialize_setup(tuple(red_setup)),
        blue_setup=serialize_setup(tuple(blue_setup)),
        first_player=spec.first_player,
        winner=winner,
        draw=draw,
        candidate_result=candidate_result,
        candidate_score=candidate_score,
        terminal_reason=terminal_reason,
        plies=state.total_moves,
        decisions=decisions,
        replay_digest=replay_digest(record),
        action_history=history if record_actions else None,
        wall_clock_seconds=elapsed,
        policy_error=None if failure is None else str(failure),
        policy_error_category=None if failure is None else failure.category,
        policy_error_role=None if failure is None else failure.role,
        policy_error_policy=None if failure is None else failure.policy.token,
        policy_error_ply=None if failure is None else failure.ply,
    )


def _rules_payload(rules: RulesConfig) -> dict:
    """RulesConfig as a plain dict, in a fixed key order."""
    return {
        "rules_version": rules.rules_version,
        "board_geometry_version": rules.board_geometry_version,
        "first_player": rules.first_player,
        "battleless_move_limit": rules.battleless_move_limit,
        "absolute_move_limit": rules.absolute_move_limit,
        "two_square_rule_enabled": rules.two_square_rule_enabled,
        "continuous_chasing_rule_enabled": rules.continuous_chasing_rule_enabled,
        "context": rules.context,
    }


# ---------------------------------------------------------------------------
# Reproducing and verifying a stored row
# ---------------------------------------------------------------------------


def replay_stored_match(result: MatchResult) -> list[str]:
    """Replay a row's stored action history through the engine.

    Pure engine replay -- no policies are consulted -- so it checks that the
    recorded actions really produce the recorded outcome. Returns human-readable
    problems; an empty list means the row is internally consistent.
    """
    if result.action_history is None:
        return [f"match {result.match_id}: no stored action history to replay"]
    if result.errored:
        return [f"match {result.match_id}: errored matches are unfinished and not replayable"]

    problems: list[str] = []
    config = result.rules_config()
    state = create_game(
        deserialize_setup(result.red_setup),
        deserialize_setup(result.blue_setup),
        rules=config,
        game_id=result.match_id,
    )
    for index, action_id in enumerate(result.action_history):
        if state.terminal:
            problems.append(
                f"match {result.match_id}: game ended at ply {index} but the history "
                f"holds {len(result.action_history)} actions"
            )
            return problems
        try:
            apply_action(state, action_id)
        except Exception as error:  # noqa: BLE001 -- reported, not raised
            problems.append(f"match {result.match_id}: action {index} rejected: {error}")
            return problems

    if not state.terminal:
        problems.append(f"match {result.match_id}: replay did not reach a terminal state")
        return problems
    if state.terminal_reason != result.terminal_reason:
        problems.append(
            f"match {result.match_id}: replay ended {state.terminal_reason!r}, row says "
            f"{result.terminal_reason!r}"
        )
    if state.winner != result.winner:
        problems.append(
            f"match {result.match_id}: replay winner {state.winner!r}, row says {result.winner!r}"
        )
    if state.total_moves != result.plies:
        problems.append(
            f"match {result.match_id}: replay took {state.total_moves} plies, row says "
            f"{result.plies}"
        )
    rebuilt = replay_digest(
        build_replay_record(
            state,
            deserialize_setup(result.red_setup),
            deserialize_setup(result.blue_setup),
            seeds={
                "candidate_seed": result.candidate_seed,
                "opponent_seed": result.opponent_seed,
                "root_seed": result.root_seed,
            },
        )
    )
    if rebuilt != result.replay_digest:
        problems.append(
            f"match {result.match_id}: rebuilt replay digest {rebuilt[:12]} != stored "
            f"{result.replay_digest[:12]}"
        )
    return problems


def reproduce_match(
    result: MatchResult,
    *,
    bank: "SetupBank | None" = None,
    policies: "Mapping[str, Policy] | None" = None,
    on_policy_error: str = ON_POLICY_ERROR_RAISE,
) -> MatchResult:
    """Re-play a stored row from its identity and return the fresh result.

    Uses the row's own setups when no bank is supplied, which is what makes a
    stored result reproducible on a machine that never generated the bank.
    """
    spec = result.spec()
    if bank is not None:
        return play_match(
            spec, bank=bank, policies=policies, on_policy_error=on_policy_error
        )
    return play_match(
        spec,
        setups=(deserialize_setup(result.red_setup), deserialize_setup(result.blue_setup)),
        policies=policies,
        on_policy_error=on_policy_error,
    )


def compare_results(
    first: "Iterable[MatchResult]", second: "Iterable[MatchResult]"
) -> list[str]:
    """Field-level differences between two runs of the same schedule.

    Both sides are keyed by `match_id`, so the order rows arrive in is
    irrelevant -- which is exactly the guarantee the parallel gate needs.

    Only fields present on **both** sides are compared. In practice that means
    one thing: a run that stored action histories can be compared against a run
    that stored only digests, and the comparison is still complete, because
    `replay_digest` covers the history and is always present. Without this rule a
    digest-only row would differ from a full row on every match purely because
    one of them declined to store 300 integers.
    """
    left = {row.match_id: row for row in first}
    right = {row.match_id: row for row in second}
    problems: list[str] = []

    for match_id in sorted(set(left) - set(right)):
        problems.append(f"match {match_id} present in the first run only")
    for match_id in sorted(set(right) - set(left)):
        problems.append(f"match {match_id} present in the second run only")

    for match_id in sorted(set(left) & set(right)):
        a = left[match_id].comparable()
        b = right[match_id].comparable()
        for key in sorted(set(a) & set(b)):
            if a[key] != b[key]:
                problems.append(f"match {match_id}: {key} differs ({a[key]!r} != {b[key]!r})")
    return problems


def results_digest(results: "Iterable[MatchResult]") -> str:
    """Order-independent digest over the reproducible content of a result set."""
    digests = sorted(row.comparable_digest() for row in results)
    return hashlib.sha256("\n".join(digests).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Running a schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSummary:
    """What one execution of a schedule produced.

    `results` is sorted by `match_id`, so downstream code sees a stable order
    whatever order the workers finished in.
    """

    results: tuple[MatchResult, ...]
    schedule_digest: str
    results_digest: str
    worker_count: int
    chunk_count: int
    matches_run: int
    paired_units_run: int
    policy_errors: int
    illegal_policy_actions: int
    wall_clock_seconds: float
    runner_version: str = MATCH_RUNNER_VERSION
    schema_version: str = MATCH_RESULT_SCHEMA_VERSION

    @property
    def errored(self) -> tuple[MatchResult, ...]:
        return tuple(row for row in self.results if row.errored)

    @property
    def plies(self) -> int:
        return sum(row.plies for row in self.results)

    def summary_dict(self) -> dict:
        """Run metadata without the rows, for a data file or manifest."""
        return {
            "runner_version": self.runner_version,
            "match_result_schema_version": self.schema_version,
            "schedule_digest": self.schedule_digest,
            "results_digest": self.results_digest,
            "worker_count": self.worker_count,
            "chunk_count": self.chunk_count,
            "matches_run": self.matches_run,
            "paired_units_run": self.paired_units_run,
            "policy_errors": self.policy_errors,
            "illegal_policy_actions": self.illegal_policy_actions,
            "total_plies": self.plies,
            "wall_clock_seconds": self.wall_clock_seconds,
        }


#: Per-process state for the parallel path. A worker rebuilds the bank and the
#: policies once and reuses them for every chunk it is handed; Agent 2 proved
#: policy instances carry no state between decisions.
_WORKER_STATE: dict[str, Any] = {}


def _worker_init(bank_payload: dict, options: dict) -> None:
    _WORKER_STATE.clear()
    _WORKER_STATE["bank"] = SetupBank.from_dict(bank_payload)
    _WORKER_STATE["options"] = options
    _WORKER_STATE["policies"] = {}


def _worker_policies(spec: MatchSpec) -> dict[str, Policy]:
    """The two policies this match needs, built at most once per worker process."""
    cache: dict[str, Policy] = _WORKER_STATE["policies"]
    needed = (spec.candidate, spec.opponent)
    if any(ref.token not in cache for ref in needed):
        cache.update(resolve_policies(spec))
    return {ref.token: cache[ref.token] for ref in needed}


def _run_chunk(payload: list[dict]) -> list[dict]:
    """Play one chunk of matches inside a worker process.

    Takes and returns plain dictionaries: `MatchSpec` is rebuilt from its stored
    identity (which re-verifies `match_id`), and results travel back as dicts, so
    nothing about the transport can influence a game.
    """
    options = _WORKER_STATE["options"]
    bank = _WORKER_STATE["bank"]
    rules = RulesConfig(**options["rules_payload"])
    rows: list[dict] = []
    for entry in payload:
        spec = MatchSpec.from_dict(entry, rules=rules)
        rows.append(
            play_match(
                spec,
                bank=bank,
                policies=_worker_policies(spec),
                record_actions=options["record_actions"],
                on_policy_error=options["on_policy_error"],
                verify_invariants=options["verify_invariants"],
            ).to_dict()
        )
    return rows


def run_schedule(
    matches: "Sequence[MatchSpec]",
    bank: SetupBank,
    *,
    policies: "Mapping[str, Policy] | None" = None,
    worker_count: int = 1,
    chunks_per_worker: int = 4,
    record_actions: bool = True,
    on_policy_error: str = ON_POLICY_ERROR_RAISE,
    verify_invariants: bool = False,
) -> RunSummary:
    """Run a schedule, optionally across worker processes.

    `worker_count=1` runs in this process with no pool at all, which is both the
    fast path for small runs and the serial reference the parallel gate compares
    against.

    Chunking exists for load balance, not for identity. Matches are dealt to
    chunks round-robin by :func:`shard_schedule`, so a matchup whose games are
    much longer than average (Agent 2 measured 866 mean plies for
    `stress_information_miser` against ~300 for the ladder) spreads across
    workers instead of landing on one. `chunks_per_worker` above 1 lets a worker
    that drew short games pick up more work.
    """
    if worker_count < 1:
        raise MatchRunnerError(f"worker_count must be at least 1, got {worker_count}")
    if chunks_per_worker < 1:
        raise MatchRunnerError(f"chunks_per_worker must be at least 1, got {chunks_per_worker}")
    if on_policy_error not in ON_POLICY_ERROR_MODES:
        raise MatchRunnerError(f"unknown on_policy_error mode {on_policy_error!r}")

    specs = tuple(matches)
    if not specs:
        raise MatchRunnerError("run_schedule was given an empty schedule")

    rules_set = {rules_token(spec.rules) for spec in specs}
    if len(rules_set) > 1:
        raise MatchRunnerError(
            "a single run must use one rules configuration; this schedule mixes "
            f"{len(rules_set)}. Split it into one run per configuration."
        )

    started = time.perf_counter()
    if worker_count == 1:
        # No pool at all: this is both the fast path for small runs and the
        # serial reference the parallel gate is compared against.
        results = [
            play_match(
                spec,
                bank=bank,
                policies=policies,
                record_actions=record_actions,
                on_policy_error=on_policy_error,
                verify_invariants=verify_invariants,
            )
            for spec in specs
        ]
        chunk_count = 1
    else:
        results, chunk_count = _run_parallel(
            specs,
            bank=bank,
            worker_count=worker_count,
            chunks_per_worker=chunks_per_worker,
            record_actions=record_actions,
            on_policy_error=on_policy_error,
            verify_invariants=verify_invariants,
        )
    elapsed = time.perf_counter() - started

    ordered = tuple(sorted(results, key=lambda row: row.match_id))
    return RunSummary(
        results=ordered,
        schedule_digest=schedule_digest(specs),
        results_digest=results_digest(ordered),
        worker_count=worker_count,
        chunk_count=chunk_count,
        matches_run=len(ordered),
        paired_units_run=len({row.paired_unit_id for row in ordered}),
        policy_errors=sum(1 for row in ordered if row.errored),
        illegal_policy_actions=sum(
            1 for row in ordered if row.policy_error_category == ERROR_ILLEGAL_ACTION
        ),
        wall_clock_seconds=elapsed,
    )


def _run_parallel(
    specs: "Sequence[MatchSpec]",
    *,
    bank: SetupBank,
    worker_count: int,
    chunks_per_worker: int,
    record_actions: bool,
    on_policy_error: str,
    verify_invariants: bool,
) -> tuple[list[MatchResult], int]:
    chunk_count = min(len(specs), max(worker_count * chunks_per_worker, worker_count))
    chunks = [chunk for chunk in shard_schedule(specs, chunk_count) if chunk]
    payloads = [[spec.to_dict() for spec in chunk] for chunk in chunks]
    options = {
        "record_actions": record_actions,
        "on_policy_error": on_policy_error,
        "verify_invariants": verify_invariants,
        "rules_payload": _rules_payload(specs[0].rules),
    }

    results: list[MatchResult] = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_worker_init,
        initargs=(bank.to_dict(), options),
    ) as pool:
        for rows in pool.map(_run_chunk, payloads):
            results.extend(MatchResult.from_dict(row) for row in rows)
    return results, len(chunks)


def suggested_worker_count(maximum: int = 8) -> int:
    """A sensible default worker count for this machine.

    Leaves two cores for the parent process and the operating system; a fully
    saturated machine makes wall-clock measurements noisy without finishing
    sooner.
    """
    cores = os.cpu_count() or 1
    return max(1, min(maximum, cores - 2))


__all__ = [
    "ERROR_CATEGORIES",
    "ERROR_CONTRACT_VIOLATION",
    "ERROR_ENGINE_REJECTED",
    "ERROR_ILLEGAL_ACTION",
    "ERROR_POLICY_EXCEPTION",
    "MATCH_RESULT_SCHEMA_VERSION",
    "MATCH_RUNNER_VERSION",
    "ON_POLICY_ERROR_MODES",
    "ON_POLICY_ERROR_QUARANTINE",
    "ON_POLICY_ERROR_RAISE",
    "RESULT_DRAW",
    "RESULT_ERROR",
    "RESULT_LABELS",
    "RESULT_LOSS",
    "RESULT_WIN",
    "TERMINAL_POLICY_ERROR",
    "MatchResult",
    "MatchRunnerError",
    "PolicyFailure",
    "RunSummary",
    "compare_results",
    "play_match",
    "replay_digest",
    "replay_stored_match",
    "reproduce_match",
    "resolve_policies",
    "results_digest",
    "run_schedule",
    "suggested_worker_count",
]
