"""Phase 12 Agent 5: the working search-enhanced player.

Specification sources:

- `06_PHASE_12_AGENT_5_WORKING_SEARCH_PLAYER.md`
- `00_PHASE_12_SEQUENCE_AND_COMMON_CONTRACT.md` section 14

This module turns the accepted TINY + Agent 1C configuration into the one
player the project actually uses from here: machine seats, human play, and
whatever training or evaluation comes next. It is engineering integration
over the frozen Agent 1 engine — no search behaviour is defined here, and
no budget or score constant is restated rather than imported.

The production stack, structurally
----------------------------------
:class:`Phase12SearchPlayer` accepts exactly one belief provider identity:
`agent1c`, over the accepted Phase 9 C1 move model. The oracle cannot enter
production three separate ways, none of them a UI convention:

- this constructor refuses any provider with `uses_hidden_truth` (and any
  provider that is not `agent1c` at all);
- every engine is built with `production=True`, whose constructor refuses
  hidden-truth providers independently;
- the provider factory refuses to *build* an oracle under
  `production=True` in the first place.

There is no parameter, mode name or configuration field through which an
oracle could reach a `Phase12SearchPlayer`.

Time cap and fallback
---------------------
Every search decision runs under a per-move wall-clock cap (see
`MODE_TIME_CAP_SECONDS`; the production TINY cap is 0.5 s against an
observed 0.138 s p95 and 0.193 s max — comfortable headroom, not the p95
itself). If search runs past the cap, raises, returns a non-finite score,
or produces anything but a legal action, the player falls back to the
direct accepted Phase 9 C1 action — the same accepted greedy-adapter rule
the engine's own `direct_action_id` is pinned to. The player never
forfeits and never emits an illegal action because search failed; every
fallback is counted by reason and logged.

Determinism
-----------
Search decisions are the engine's: seed-deterministic. The time cap is the
one deliberately wall-clock-dependent element — a decision that runs past
its cap falls back, and whether a marginal decision does so depends on the
machine. At the production cap that margin is ~3.6x above the observed
p95, so in practice the player replays the Agent 4 games bit-identically
(the runner checks exactly that).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field

import torch

from ...engine.legal_moves import legal_action_mask, legal_actions
from ...engine.observation import build_observation
from ...engine.state import GameState
from ...model.base import StrategoModel
from ...model.policy_adapter import (
    DECISION_MODE_GREEDY,
    prepare_legality,
    select_action,
)
from ...model.tokenization import observation_batch_from_numpy, observation_to_tokens
from .contract import (
    PROVIDER_AGENT1C,
    SCORE_DEFINITION,
    SEARCH_PRESETS,
    SEARCH_VERSION,
    Phase12SearchError,
    Phase12SearchTimeout,
    derive_phase12_seed,
)
from .engine import Phase12SearchDecision, Phase12SearchEngine
from .matchplay import MatchArm, search_seed_for
from .providers import Phase12BeliefProvider

logger = logging.getLogger("stratego.phase12.player")

#: The working player's identity. Any change to the mode set, the caps, the
#: fallback rule or the seat bookkeeping is a new version.
PLAYER_VERSION = "phase12_search_player_v1"

#: The engineering candidate this player freezes (contract section 14).
CANDIDATE_ARTIFACT = "phase12_search_candidate_v1"

#: The explicit modes the instruction requires. `direct` is the accepted
#: Phase 9 greedy player with no search; the three search modes are the
#: instructed presets by name. TINY is the selected production default;
#: SMALL and MEDIUM remain optional engineering/debug modes.
MODE_DIRECT = "direct"
MODE_TINY = "tiny"
MODE_SMALL = "small"
MODE_MEDIUM = "medium"
PLAYER_MODES = (MODE_DIRECT, MODE_TINY, MODE_SMALL, MODE_MEDIUM)
SEARCH_MODES = (MODE_TINY, MODE_SMALL, MODE_MEDIUM)
DEFAULT_MODE = MODE_TINY

#: Mode -> instructed preset name in `SEARCH_PRESETS`.
MODE_PRESETS = {MODE_TINY: "TINY", MODE_SMALL: "SMALL", MODE_MEDIUM: "MEDIUM"}

#: By project direction, MEDIUM is additionally designated the *current
#: maximum-strength candidate*: the strongest rung Agent 4 observed
#: (EWR 0.6875 against TINY's 0.6406 on the same pack) at 0.846 s/move
#: median. The 0.0469 EWR lead sits inside the 0.10 engineering margin, so
#: this names the strongest observed configuration — for callers who want
#: maximum strength and accept the latency — without disturbing TINY as
#: the selected production default.
MAX_STRENGTH_MODE = MODE_MEDIUM

#: Per-move search time caps in seconds. Chosen from the Agent 4 latency
#: profile with comfortable headroom above the observed p95 rather than the
#: p95 itself: TINY observed 0.126 s median / 0.138 s p95 / 0.193 s max ->
#: 0.5 s cap (3.6x p95, 2.6x max). The debug rungs scale the same way:
#: SMALL 0.382 s p95 -> 1.5 s, MEDIUM 0.916 s p95 -> 3.5 s. The cap absorbs
#: scheduler jitter and thermal throttling without ever letting a human
#: opponent notice a stall; it is not a tuning knob on search behaviour.
MODE_TIME_CAP_SECONDS = {MODE_TINY: 0.5, MODE_SMALL: 1.5, MODE_MEDIUM: 3.5}

#: The frozen fallback rule, stated once for reports.
FALLBACK_POLICY = "direct accepted Phase 9 C1"

#: Why a decision fell back, in the order the checks run.
FALLBACK_TIMEOUT = "timeout"
FALLBACK_SEARCH_ERROR = "search_error"
FALLBACK_UNEXPECTED_ERROR = "unexpected_error"
FALLBACK_NON_FINITE = "non_finite_score"
FALLBACK_ILLEGAL_ACTION = "illegal_action"
#: The defensive last resort: the direct model call itself failed, so the
#: lowest legal action id is played. This is not a search failure mode; it
#: exists so no failure of any kind can make the player forfeit.
FALLBACK_DIRECT_ERROR = "direct_error"
FALLBACK_REASONS = (
    FALLBACK_TIMEOUT,
    FALLBACK_SEARCH_ERROR,
    FALLBACK_UNEXPECTED_ERROR,
    FALLBACK_NON_FINITE,
    FALLBACK_ILLEGAL_ACTION,
    FALLBACK_DIRECT_ERROR,
)

#: The oracle's production availability, as a module-level structural fact
#: the artifact and the tests both read. See the module docstring for the
#: three independent refusals that make it true.
ORACLE_AVAILABLE_IN_PRODUCTION = False

#: Seed domain for standalone play (human games, ad-hoc drivers), where no
#: match board id exists. Deterministic in `(game_id, ply)`, so a replayed
#: game re-derives the same worlds. Match seats use the match stream via
#: :func:`stratego.search.phase12.matchplay.search_seed_for` instead.
DOMAIN_PLAYER_SEARCH = "player_search"


def player_seed_for(game_id: str, ply: int) -> int:
    """The default world-sampling seed for one standalone decision."""
    return derive_phase12_seed(DOMAIN_PLAYER_SEARCH, str(game_id), int(ply)) >> 1


def check_mode(mode: str) -> str:
    """`mode`, or a refusal naming the legal modes. Oracle is not a mode."""
    if mode not in PLAYER_MODES:
        raise Phase12SearchError(
            f"unknown player mode {mode!r}; modes are {PLAYER_MODES}. There is "
            "no oracle mode: the oracle provider is diagnostic-only and is "
            "structurally unavailable in production"
        )
    return mode


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase12PlayerDecision:
    """One working-player decision, with its provenance.

    `action_id` is always legal in the state it was asked about — that is
    the player's contract, not an aspiration. `used_search` is true only
    when the returned action is the search engine's; on any fallback it is
    false and `fallback_reason` says why. `search` carries the full engine
    decision when one completed, for logs and diagnostics.
    """

    player_version: str
    mode: str
    preset_id: "str | None"
    action_id: int
    used_search: bool
    fallback_reason: "str | None"
    direct_action_id: "int | None"
    move_changed: "bool | None"
    legal_action_count: int
    seed: "int | None"
    time_cap_seconds: "float | None"
    seconds: float
    search: "Phase12SearchDecision | None" = field(repr=False, default=None)

    def summary(self) -> dict:
        return {
            "player_version": self.player_version,
            "mode": self.mode,
            "preset_id": self.preset_id,
            "action_id": self.action_id,
            "used_search": self.used_search,
            "fallback_reason": self.fallback_reason,
            "direct_action_id": self.direct_action_id,
            "move_changed": self.move_changed,
            "legal_action_count": self.legal_action_count,
            "seed": self.seed,
            "time_cap_seconds": self.time_cap_seconds,
            "seconds": self.seconds,
            "c1_forwards": self.search.c1_forwards if self.search else 1,
        }


# ---------------------------------------------------------------------------
# The player
# ---------------------------------------------------------------------------


class Phase12SearchPlayer:
    """The stable production player: accepted Phase 9 C1 + Agent 1C search.

    One instance owns the move model, the one production belief provider
    and one engine per search preset; `decide` answers for whichever mode
    is active (or explicitly requested), under the mode's time cap, with
    the direct accepted Phase 9 action as the fallback for every failure.
    """

    def __init__(
        self,
        model: StrategoModel,
        provider: Phase12BeliefProvider,
        *,
        mode: str = DEFAULT_MODE,
        device: "torch.device | str" = "cpu",
        model_identity: "dict | None" = None,
        time_caps: "dict | None" = None,
    ) -> None:
        if not isinstance(provider, Phase12BeliefProvider):
            raise Phase12SearchError(
                f"expected a Phase12BeliefProvider, got {type(provider).__name__}"
            )
        if provider.uses_hidden_truth:
            raise Phase12SearchError(
                "the working player is a production interface and structurally "
                "excludes any provider that reads hidden truth; the oracle is "
                "diagnostic-only"
            )
        if provider.provider_id != PROVIDER_AGENT1C:
            raise Phase12SearchError(
                "the production stack is frozen as accepted Phase 9 C1 + agent1c "
                f"beliefs; provider {provider.provider_id!r} is not the working "
                "player's to use"
            )
        self.mode = check_mode(mode)
        self.provider = provider
        self.model_identity = dict(model_identity or {})
        self.time_caps = dict(MODE_TIME_CAP_SECONDS)
        for name, cap in (time_caps or {}).items():
            if name not in SEARCH_MODES:
                raise Phase12SearchError(
                    f"time cap for unknown search mode {name!r}; search modes "
                    f"are {SEARCH_MODES}"
                )
            if not (float(cap) > 0.0):
                raise Phase12SearchError(f"time cap for {name!r} must be positive")
            self.time_caps[name] = float(cap)

        # One engine per instructed preset, production configuration —
        # `production=True` re-refuses hidden-truth providers independently
        # of the check above. The engines share the model and provider.
        self.engines = {
            name: Phase12SearchEngine(
                model,
                provider,
                SEARCH_PRESETS[MODE_PRESETS[name]],
                device=device,
                model_identity=self.model_identity,
            )
            for name in SEARCH_MODES
        }
        # The engine moved the model to its device; decide direct moves on
        # the same instance so every path sees identical weights.
        reference = self.engines[MODE_TINY]
        self.model = reference.model
        self.device = reference.device
        self.dtype = reference.dtype

        self.decision_counts = {name: 0 for name in PLAYER_MODES}
        self.fallback_counts = {reason: 0 for reason in FALLBACK_REASONS}
        self.fallback_events: list = []
        logger.info(
            "phase12 player ready: mode=%s budget=%s cap=%ss provider=%s",
            self.mode,
            self.budget_text(self.mode),
            self.time_caps.get(self.mode),
            provider.provider_id,
        )

    # -- modes ---------------------------------------------------------------

    def set_mode(self, mode: str) -> str:
        """Switch the active mode; returns the previous one."""
        previous, self.mode = self.mode, check_mode(mode)
        logger.info(
            "phase12 player mode: %s -> %s (budget %s, cap %ss)",
            previous,
            self.mode,
            self.budget_text(self.mode),
            self.time_caps.get(self.mode),
        )
        return previous

    def budget_text(self, mode: str) -> str:
        if mode == MODE_DIRECT:
            return "no search, one forward per decision"
        config = self.engines[mode].config
        return (
            f"{config.preset_id}: {config.worlds} worlds, depth "
            f"{config.rollout_depth}, <= {config.max_root_candidates} candidates"
        )

    # -- the direct accepted action -------------------------------------------

    def _direct_action(self, state: GameState, legal) -> int:
        """The accepted Phase 9 greedy action: one forward, the adapter rule.

        This is the same decision rule the engine pins its own
        `direct_action_id` to, and that the Agent 3/4 match-time probes
        held equal to the accepted `RemoteNeuralPolicy` seat.
        """
        mask = legal_action_mask(state, legal)
        legality = prepare_legality(legal, mask, state.acting_player)
        observation = build_observation(state, state.acting_player)
        batch = observation_batch_from_numpy(
            [observation], dtype=self.dtype, device=self.device
        )
        with torch.no_grad():
            outputs = self.model(observation_to_tokens(batch))
        if self.device.type == "mps":
            torch.mps.synchronize()
        row = outputs.policy_logits.detach().to("cpu", torch.float32)[0]
        chosen = select_action(row, legality, decision_mode=DECISION_MODE_GREEDY)
        return int(chosen.absolute_action_id)

    # -- deciding --------------------------------------------------------------

    def decide(
        self,
        state: GameState,
        *,
        seed: "int | None" = None,
        mode: "str | None" = None,
    ) -> Phase12PlayerDecision:
        """One legal action for the acting player of `state`.

        `mode` overrides the active mode for this decision only. `seed`
        defaults to the standalone stream `player_seed_for(game_id, ply)`;
        match drivers pass the match stream's seed instead.
        """
        started = time.perf_counter()
        mode = self.mode if mode is None else check_mode(mode)
        if state.terminal:
            raise Phase12SearchError("a decision was requested for a terminal state")
        legal = legal_actions(state)
        if not legal:
            raise Phase12SearchError("a non-terminal state presented no legal actions")
        legal_set = set(legal)
        self.decision_counts[mode] += 1

        if mode == MODE_DIRECT:
            action, reason = self._decide_direct(state, legal, mode)
            return self._record(
                state, mode, None, action, False, reason, action if reason is None else None,
                None, len(legal), None, None, started, None,
            )

        cap = self.time_caps[mode]
        deadline = started + cap
        if seed is None:
            seed = player_seed_for(state.game_id, int(state.total_moves))
        engine = self.engines[mode]
        preset_id = engine.config.preset_id

        decision = None
        reason = None
        try:
            decision = engine.choose_action(state, seed=int(seed), deadline=deadline)
        except Phase12SearchTimeout:
            reason = FALLBACK_TIMEOUT
        except Phase12SearchError:
            logger.exception("search failed at ply %s of %s", state.total_moves, state.game_id)
            reason = FALLBACK_SEARCH_ERROR
        except Exception:
            logger.exception(
                "unexpected search failure at ply %s of %s", state.total_moves, state.game_id
            )
            reason = FALLBACK_UNEXPECTED_ERROR
        else:
            if time.perf_counter() > deadline:
                # Completed, but past the cap: the cap is a promise about
                # move latency, so a late answer is still a timeout.
                reason = FALLBACK_TIMEOUT
            elif not all(math.isfinite(c.score) for c in decision.candidates):
                reason = FALLBACK_NON_FINITE
            elif int(decision.selected_action_id) not in legal_set:
                reason = FALLBACK_ILLEGAL_ACTION

        if reason is None:
            return self._record(
                state, mode, preset_id, int(decision.selected_action_id), True, None,
                int(decision.direct_action_id), bool(decision.move_changed),
                len(legal), int(seed), cap, started, decision,
            )

        action, direct_reason = self._decide_direct(state, legal, mode)
        reason = reason if direct_reason is None else direct_reason
        self._count_fallback(state, mode, reason, started)
        return self._record(
            state, mode, preset_id, action, False, reason,
            action if direct_reason is None else None, False,
            len(legal), int(seed), cap, started, decision,
        )

    def _decide_direct(self, state: GameState, legal, mode: str):
        """`(action, reason)`: the direct action, or the last resort.

        The last resort — the lowest legal action id — exists only so that
        no conceivable failure makes the player forfeit or move illegally.
        `reason` is `None` on the normal path.
        """
        try:
            action = self._direct_action(state, legal)
            if action in set(legal):
                return action, None
            logger.error("direct action %s is not legal; playing the last resort", action)
        except Exception:
            logger.exception(
                "direct decision failed at ply %s of %s", state.total_moves, state.game_id
            )
        if mode == MODE_DIRECT:
            self._count_fallback(state, mode, FALLBACK_DIRECT_ERROR, time.perf_counter())
        return int(min(legal)), FALLBACK_DIRECT_ERROR

    def _count_fallback(self, state, mode, reason, started) -> None:
        self.fallback_counts[reason] += 1
        event = {
            "game_id": state.game_id,
            "ply": int(state.total_moves),
            "mode": mode,
            "reason": reason,
            "seconds": round(time.perf_counter() - started, 5),
        }
        self.fallback_events.append(event)
        del self.fallback_events[:-32]
        logger.warning(
            "fallback to %s: %s at ply %s of %s", FALLBACK_POLICY, reason,
            event["ply"], event["game_id"],
        )

    def _record(
        self, state, mode, preset_id, action, used_search, reason, direct_id,
        move_changed, legal_count, seed, cap, started, decision,
    ) -> Phase12PlayerDecision:
        return Phase12PlayerDecision(
            player_version=PLAYER_VERSION,
            mode=mode,
            preset_id=preset_id,
            action_id=int(action),
            used_search=used_search,
            fallback_reason=reason,
            direct_action_id=direct_id,
            move_changed=move_changed,
            legal_action_count=legal_count,
            seed=seed,
            time_cap_seconds=cap,
            seconds=time.perf_counter() - started,
            search=decision,
        )

    # -- reporting -------------------------------------------------------------

    def status(self) -> dict:
        """The live view a log line or UI shows: mode, budget, cap, fallbacks."""
        return {
            "player_version": PLAYER_VERSION,
            "mode": self.mode,
            "max_strength_mode": MAX_STRENGTH_MODE,
            "budget": self.budget_text(self.mode),
            "time_cap_seconds": self.time_caps.get(self.mode),
            "time_caps": dict(self.time_caps),
            "belief_provider": self.provider.provider_id,
            "decisions": dict(self.decision_counts),
            "fallback_total": sum(self.fallback_counts.values()),
            "fallbacks": dict(self.fallback_counts),
            "recent_fallbacks": list(self.fallback_events),
            "fallback_policy": FALLBACK_POLICY,
            "oracle_available_in_production": ORACLE_AVAILABLE_IN_PRODUCTION,
        }

    def describe(self) -> dict:
        """The static identity: versions, presets, caps, models, provider."""
        return {
            "player_version": PLAYER_VERSION,
            "search_version": SEARCH_VERSION,
            "score_definition": SCORE_DEFINITION,
            "modes": list(PLAYER_MODES),
            "default_mode": DEFAULT_MODE,
            "max_strength_mode": MAX_STRENGTH_MODE,
            "mode": self.mode,
            "presets": {
                name: self.engines[name].config.describe() for name in SEARCH_MODES
            },
            "time_caps": dict(self.time_caps),
            "fallback_policy": FALLBACK_POLICY,
            "fallback_reasons": list(FALLBACK_REASONS),
            "belief_provider": self.provider.describe(),
            "model_identity": dict(self.model_identity),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "oracle_available_in_production": ORACLE_AVAILABLE_IN_PRODUCTION,
        }


# ---------------------------------------------------------------------------
# The match seat
# ---------------------------------------------------------------------------


#: One arm identity per mode, for the accepted Phase 12 match driver.
PLAYER_SEAT_ARMS = {
    MODE_DIRECT: MatchArm("player_direct", "direct", "working player, direct C1"),
    MODE_TINY: MatchArm(
        "player_search_tiny", "search",
        "working player, search + agent1c @ TINY", PROVIDER_AGENT1C,
    ),
    MODE_SMALL: MatchArm(
        "player_search_small", "search",
        "working player, search + agent1c @ SMALL", PROVIDER_AGENT1C,
    ),
    MODE_MEDIUM: MatchArm(
        "player_search_medium", "search",
        "working player, search + agent1c @ MEDIUM", PROVIDER_AGENT1C,
    ),
}


class Phase12PlayerSeat:
    """The working player behind the existing match-driver seat interface.

    This is how machine-vs-machine games select `direct C1` or
    `search + Agent1C`: one seat per mode over a shared player, playable by
    the accepted :func:`stratego.search.phase12.matchplay.play_arm_game`
    loop and probeable by its `SeatProbe`. Search seats draw their per-ply
    world seeds from the match stream (`search_seed_for`), so a seat with a
    quiet fallback log replays the Agent 3/4 games move for move.
    """

    def __init__(self, player: Phase12SearchPlayer, mode: "str | None" = None) -> None:
        self.player = player
        self.mode = player.mode if mode is None else check_mode(mode)
        self.arm = PLAYER_SEAT_ARMS[self.mode]
        self.kind = self.arm.kind

    def decide(self, state: GameState, legal, spec, plan):
        seed = (
            None
            if self.mode == MODE_DIRECT
            else search_seed_for(plan.board_id, int(state.total_moves))
        )
        decision = self.player.decide(state, seed=seed, mode=self.mode)
        search = decision.search if decision.used_search else None
        return int(decision.action_id), {
            "ply": int(state.total_moves),
            "seconds": float(decision.seconds),
            "legal_actions": int(decision.legal_action_count),
            "move_changed": decision.move_changed,
            "c1_forwards": int(search.c1_forwards) if search else 1,
            "unique_worlds": int(search.unique_worlds) if search else None,
            "candidates": len(search.candidates) if search else None,
            "forward_seconds": float(search.forward_seconds) if search else None,
            "direct_action_id": decision.direct_action_id,
            "fallback_reason": decision.fallback_reason,
            "used_search": decision.used_search,
        }

    def describe(self) -> dict:
        return {
            "arm": self.arm.describe(),
            "mode": self.mode,
            "time_cap_seconds": self.player.time_caps.get(self.mode),
            "seat": self.player.describe(),
        }


# ---------------------------------------------------------------------------
# Loading the production player
# ---------------------------------------------------------------------------


def load_search_player(
    repository_root,
    *,
    mode: str = DEFAULT_MODE,
    device: str = "cpu",
    time_caps: "dict | None" = None,
    handoff_path=None,
):
    """`(player, identities)`: the production player from the frozen bytes.

    Loads the accepted Phase 9 C1 through its read-only export and the
    Agent 1C belief head from its surviving checkpoint, digest-checking
    both against the Phase 11B -> 12 handoff record. This is the one
    production entry point: it can only ever produce the frozen stack.
    """
    from pathlib import Path

    from ...belief.phase11b.features import load_frozen_c1

    check_mode(mode)
    root = Path(repository_root)
    handoff_path = (
        root / "reports" / "phase11b" / "phase12_handoff.json"
        if handoff_path is None
        else Path(handoff_path)
    )
    handoff = json.loads(handoff_path.read_text())
    if handoff.get("artifact") != "phase11b_phase12_handoff_v1":
        raise Phase12SearchError(f"{handoff_path} is not the Phase 12 handoff")

    model, identity = load_frozen_c1(
        root, root / "checkpoints" / "phase12" / "phase9_c1_readonly_copy.pt",
        device=device,
    )
    expected = handoff["accepted_phase9_checkpoint"]
    for key in ("model_state_digest", "belief_head_digest"):
        if identity[key] != expected[key]:
            raise Phase12SearchError(
                f"loaded Phase 9 {key} {identity[key]} != handoff {expected[key]}"
            )

    from .providers import build_belief_provider

    record = handoff["agent1c_checkpoint"]
    provider = build_belief_provider(
        PROVIDER_AGENT1C,
        encoder=model,
        agent1c_checkpoint=root / record["path"],
        expected_agent1c_sha256=record["sha256"],
        expected_agent1c_state_digest=record["state_dict_digest"],
        production=True,
        device=device,
    )
    player = Phase12SearchPlayer(
        model,
        provider,
        mode=mode,
        device=device,
        model_identity=identity,
        time_caps=time_caps,
    )
    identities = {
        "handoff_path": str(handoff_path),
        "move_model_identity": dict(identity),
        "belief_model_identity": dict(
            provider.identity if hasattr(provider, "identity") else {}
        ),
        "handoff": handoff,
    }
    return player, identities


# ---------------------------------------------------------------------------
# The engineering candidate record
# ---------------------------------------------------------------------------


def build_candidate_record(
    *,
    move_model_identity: dict,
    belief_model_identity: dict,
    agent4: dict,
    generated_utc: str,
    environment: dict,
    agent4_medium: "dict | None" = None,
    quick_checks: "dict | None" = None,
    known_limitations: "list | None" = None,
) -> dict:
    """The `phase12_search_candidate_v1` artifact, contract section 14.

    Human-readable names *and* the exact digests: the record binds the
    frozen stack to bytes, not labels. `agent4` carries the exact Agent 4
    operating-point numbers the headline strings are rounded from;
    `agent4_medium` the exact numbers behind the MEDIUM maximum-strength
    designation.
    """
    tiny = SEARCH_PRESETS["TINY"]
    medium = SEARCH_PRESETS[MODE_PRESETS[MAX_STRENGTH_MODE]]
    core = {
        "move_model": "accepted Phase 9 C1",
        "belief_model": "Agent1C",
        "search_version": SEARCH_VERSION,
        "selected_preset": tiny.preset_id,
        "worlds": tiny.worlds,
        "root_candidates": f"<= {tiny.max_root_candidates}",
        "depth": tiny.rollout_depth,
        "beta": tiny.beta,
        "epsilon": tiny.epsilon,
        "score_definition": SCORE_DEFINITION,
        "belief_provider": PROVIDER_AGENT1C,
        "move_model_identity": dict(move_model_identity),
        "belief_model_identity": dict(belief_model_identity),
        "player_version": PLAYER_VERSION,
        "modes": list(PLAYER_MODES),
        "default_mode": DEFAULT_MODE,
        "max_strength_mode": MAX_STRENGTH_MODE,
        "max_strength_preset": medium.preset_id,
        "time_cap_seconds": MODE_TIME_CAP_SECONDS[DEFAULT_MODE],
        "time_caps_by_mode": dict(MODE_TIME_CAP_SECONDS),
        "fallback_policy": FALLBACK_POLICY,
        "fallback_reasons": list(FALLBACK_REASONS),
    }
    record = {
        "artifact": CANDIDATE_ARTIFACT,
        "phase": "phase12",
        "agent": 5,
        "generated_utc": generated_utc,
        **core,
        "expected_latency_median": f"{agent4['move_seconds_median']:.3f} s/move",
        "expected_latency_p95": f"{agent4['move_seconds_p95']:.3f} s/move",
        "Agent4_quick_EWR": round(float(agent4["ewr"]), 4),
        "Agent4_direct_EWR": round(float(agent4["direct_ewr"]), 4),
        "agent4_operating_point_exact": dict(agent4),
        "quick_checks": dict(quick_checks or {}),
        "known_limitations": list(known_limitations or []),
        "environment": dict(environment),
        "oracle_available_in_production": ORACLE_AVAILABLE_IN_PRODUCTION,
        "phase11_final_classification": "FAIL",
        "phase11b_selection": "Agent1C",
        "scientific_validation_status": "not performed",
    }
    if agent4_medium is not None:
        record["maximum_strength_candidate"] = {
            "designation": (
                "current maximum-strength candidate, by project direction; "
                "the production default remains the selected TINY preset"
            ),
            "mode": MAX_STRENGTH_MODE,
            "preset": medium.preset_id,
            "worlds": medium.worlds,
            "root_candidates": f"<= {medium.max_root_candidates}",
            "depth": medium.rollout_depth,
            "time_cap_seconds": MODE_TIME_CAP_SECONDS[MAX_STRENGTH_MODE],
            "expected_latency_median": (
                f"{agent4_medium['move_seconds_median']:.3f} s/move"
            ),
            "expected_latency_p95": f"{agent4_medium['move_seconds_p95']:.3f} s/move",
            "Agent4_quick_EWR": round(float(agent4_medium["ewr"]), 4),
            "ewr_lead_over_selected": round(
                float(agent4_medium["ewr"]) - float(agent4["ewr"]), 4
            ),
            "caveat": (
                "strongest rung observed on the Agent 4 pack; its lead over TINY "
                "is inside the 0.10 engineering margin, so this is a designation "
                "of the strongest observed configuration, not a validated ordering"
            ),
            "agent4_exact": dict(agent4_medium),
        }
    payload = json.dumps({**core, "artifact": CANDIDATE_ARTIFACT}, sort_keys=True)
    record["candidate_config_digest"] = hashlib.sha256(payload.encode()).hexdigest()
    return record


__all__ = [
    "CANDIDATE_ARTIFACT",
    "DEFAULT_MODE",
    "DOMAIN_PLAYER_SEARCH",
    "FALLBACK_DIRECT_ERROR",
    "FALLBACK_ILLEGAL_ACTION",
    "FALLBACK_NON_FINITE",
    "FALLBACK_POLICY",
    "FALLBACK_REASONS",
    "FALLBACK_SEARCH_ERROR",
    "FALLBACK_TIMEOUT",
    "FALLBACK_UNEXPECTED_ERROR",
    "MAX_STRENGTH_MODE",
    "MODE_DIRECT",
    "MODE_MEDIUM",
    "MODE_PRESETS",
    "MODE_SMALL",
    "MODE_TIME_CAP_SECONDS",
    "MODE_TINY",
    "ORACLE_AVAILABLE_IN_PRODUCTION",
    "PLAYER_MODES",
    "PLAYER_SEAT_ARMS",
    "PLAYER_VERSION",
    "Phase12PlayerDecision",
    "Phase12PlayerSeat",
    "Phase12SearchPlayer",
    "SEARCH_MODES",
    "build_candidate_record",
    "check_mode",
    "load_search_player",
    "player_seed_for",
]
