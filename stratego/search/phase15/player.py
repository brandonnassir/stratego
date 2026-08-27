"""Phase 15 Agent 2 section 15: the working player.

Specification source: `02_AGENT_2_SEARCH_IMPLEMENTATION.md` sections 14, 15, 16.

The modes
---------
```text
p18_direct          P18 greedy, no search
p24_direct          P24 greedy, no search
selected_search     the selected complete system at the selected budget
maximum_strength    the strongest observed system inside the 5 s ceiling
```

plus the production pairing ids as *diagnostic* mode names, so a machine
evaluation can ask for `p24_b18` by name. `oracle` is not among them and
cannot be reached: :meth:`Phase15SearchPlayer.check_mode` refuses any name
that is not in the mode table, the table is built from
`PRODUCTION_PAIRING_IDS`, and every engine the player holds was built under
a production configuration that the accepted Phase 12 engine already refuses
an oracle provider for.

Never forfeit
-------------
Section 15's rule is absolute: on timeout, search error, non-finite score or
an illegal result, play the direct legal move of *the same* move model the
mode selects. :meth:`decide` implements that as a chain of refusals with a
counted reason each, ending in a last-resort legal move that exists so no
failure of any kind — including a failure of the direct model itself — can
lose a game by forfeit. The reason is recorded, logged and reported; a silent
fallback would make the strength numbers unreadable.

Time caps
---------
A cap is a *latency guarantee*, not a search parameter: the engine's deadline
checks are pure control flow, so a decision that completes is bit-identical
whether or not a cap was set. Caps are set from measured p95 with headroom,
the same way Phase 12 set its own, and they are recorded on the frozen
candidate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..phase12.contract import Phase12SearchError, Phase12SearchTimeout
from .contract import (
    ACCEPTABLE_MOVE_SECONDS,
    INTEGRATION_VERSION,
    MOVE_MODELS,
    ORACLE_AVAILABLE_IN_PRODUCTION,
    PHASE15_SCORE_DEFINITION,
    PHASE15_SEARCH_VERSION,
    PRODUCTION_PAIRING_IDS,
    PROVIDER_ORACLE,
    Phase15SearchError,
    pairing as pairing_of,
)
from .systems import SystemBundle

#: The working player's identity. Any change to the mode set, the caps, the
#: fallback rule or the seat bookkeeping is a new version.
PLAYER_VERSION = "phase15_search_player_v1"

#: The frozen engineering candidate this player packages.
CANDIDATE_ARTIFACT = "phase15_search_candidate_v1"

MODE_P18_DIRECT = "p18_direct"
MODE_P24_DIRECT = "p24_direct"
MODE_SELECTED = "selected_search"
MODE_MAX_STRENGTH = "maximum_strength"

#: The four required modes, plus the production pairing ids as diagnostic
#: names for machine evaluation. `oracle` appears in neither list.
REQUIRED_MODES = (MODE_P18_DIRECT, MODE_P24_DIRECT, MODE_SELECTED, MODE_MAX_STRENGTH)
DIAGNOSTIC_MODES = tuple(
    name for name in PRODUCTION_PAIRING_IDS if name not in (MODE_P18_DIRECT, MODE_P24_DIRECT)
)
PLAYER_MODES = REQUIRED_MODES + DIAGNOSTIC_MODES

#: Why a decision fell back, in the order the checks run.
FALLBACK_TIMEOUT = "timeout"
FALLBACK_SEARCH_ERROR = "search_error"
FALLBACK_UNEXPECTED_ERROR = "unexpected_error"
FALLBACK_NON_FINITE = "non_finite_score"
FALLBACK_ILLEGAL_ACTION = "illegal_action"
#: The defensive last resort: the direct model call itself failed, so the
#: lowest legal action is played. Not a search failure mode — it exists so no
#: failure of any kind can make the player forfeit.
FALLBACK_DIRECT_ERROR = "direct_error"
FALLBACK_REASONS = (
    FALLBACK_TIMEOUT,
    FALLBACK_SEARCH_ERROR,
    FALLBACK_UNEXPECTED_ERROR,
    FALLBACK_NON_FINITE,
    FALLBACK_ILLEGAL_ACTION,
    FALLBACK_DIRECT_ERROR,
)

#: Seed domain for standalone play, where no match board id exists.
DOMAIN_PLAYER_SEARCH = "player_search"


class Phase15PlayerError(Phase15SearchError):
    """A working-player request was refused."""


def player_seed_for(game_id: str, ply: int) -> int:
    """Deterministic in `(game_id, ply)`, so a replay re-derives its worlds."""
    from .contract import DOMAIN_WORLDS, derive_search_seed

    return derive_search_seed(DOMAIN_WORLDS, DOMAIN_PLAYER_SEARCH, str(game_id), int(ply))


@dataclass(frozen=True)
class Phase15PlayerDecision:
    """One decision, with everything a log line and a report both need."""

    mode: str
    move_model: str
    provider: "str | None"
    preset_id: str
    action_id: int
    direct_action_id: int
    move_changed: bool
    searched: bool
    fallback_reason: "str | None"
    seconds: float
    time_cap: "float | None"
    seed: "int | None"
    ply: int
    legal_actions: int
    c1_forwards: int
    unique_worlds: "int | None"
    score_margin: "float | None" = None

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "move_model": self.move_model,
            "provider": self.provider,
            "preset_id": self.preset_id,
            "action_id": self.action_id,
            "direct_action_id": self.direct_action_id,
            "move_changed": self.move_changed,
            "searched": self.searched,
            "fallback_reason": self.fallback_reason,
            "seconds": round(self.seconds, 5),
            "time_cap_seconds": self.time_cap,
            "seed": self.seed,
            "ply": self.ply,
            "legal_actions": self.legal_actions,
            "c1_forwards": self.c1_forwards,
            "unique_worlds": self.unique_worlds,
            "score_margin": self.score_margin,
        }

    def log_line(self) -> str:
        """The visible mode/budget/latency/fallback line section 15 requires."""
        budget = self.preset_id if self.searched else "direct"
        tail = f" FALLBACK={self.fallback_reason}" if self.fallback_reason else ""
        return (
            f"[phase15 {self.mode}] ply={self.ply} budget={budget} "
            f"action={self.action_id} changed={int(self.move_changed)} "
            f"{self.seconds:.3f}s/cap={self.time_cap}{tail}"
        )


class Phase15SearchPlayer:
    """The packaged Phase 15 player: explicit modes, capped, never forfeits."""

    version = PLAYER_VERSION

    def __init__(
        self,
        systems,
        models,
        *,
        mode: str = MODE_SELECTED,
        time_caps: "dict | None" = None,
        device: str = "cpu",
    ) -> None:
        if isinstance(systems, SystemBundle):
            systems = {MODE_SELECTED: systems, MODE_MAX_STRENGTH: systems}
        if not isinstance(systems, dict) or not systems:
            raise Phase15PlayerError(
                "the player needs a mode -> SystemBundle mapping (or one bundle)"
            )
        for name, bundle in systems.items():
            if not isinstance(bundle, SystemBundle):
                raise Phase15PlayerError(f"{name!r} is not a SystemBundle")
            if bundle.pairing.kind == "diagnostic":
                raise Phase15PlayerError(
                    f"{name!r} carries the oracle; the working player has no "
                    "diagnostic mode"
                )
            if bundle.provider is not None and getattr(
                bundle.provider, "uses_hidden_truth", False
            ):
                raise Phase15PlayerError(
                    f"{name!r} carries a provider that reads hidden truth"
                )
        self.systems = dict(systems)
        self.models = models
        self.device = device
        self.time_caps = dict(time_caps or {})
        self.fallback_counts: dict[str, int] = {}
        self.decisions = 0
        self.searched_decisions = 0
        self.move_changes = 0
        self.total_seconds = 0.0
        self.mode = self.check_mode(mode)
        if self.mode not in self.systems and not self._is_direct(self.mode):
            raise Phase15PlayerError(
                f"mode {mode!r} has no system; the player holds {sorted(self.systems)}"
            )

    # -- modes -------------------------------------------------------------

    @staticmethod
    def check_mode(mode: str) -> str:
        """Refuse any name that is not a Phase 15 player mode.

        The oracle is refused by name here as well as by absence from the
        table, so the refusal reads as intentional rather than incidental.
        """
        if mode == PROVIDER_ORACLE or (isinstance(mode, str) and mode.endswith("_oracle")):
            raise Phase15PlayerError(
                "the oracle is an offline diagnostic and is not a player mode; "
                f"oracle_available_in_production={ORACLE_AVAILABLE_IN_PRODUCTION}"
            )
        if mode not in PLAYER_MODES:
            raise Phase15PlayerError(
                f"unknown mode {mode!r}; Phase 15 player modes are {list(PLAYER_MODES)}"
            )
        return mode

    @staticmethod
    def _is_direct(mode: str) -> bool:
        return mode in (MODE_P18_DIRECT, MODE_P24_DIRECT)

    def set_mode(self, mode: str) -> str:
        mode = self.check_mode(mode)
        if not self._is_direct(mode) and mode not in self.systems:
            raise Phase15PlayerError(
                f"mode {mode!r} has no system; the player holds {sorted(self.systems)}"
            )
        self.mode = mode
        return mode

    def mode_move_model(self, mode: str) -> str:
        if mode == MODE_P18_DIRECT:
            return "p18"
        if mode == MODE_P24_DIRECT:
            return "p24"
        return self.systems[mode].pairing.move_model

    def budget_text(self, mode: str) -> str:
        if self._is_direct(mode):
            return "direct, no search"
        config = self.systems[mode].config
        return (
            f"{config.preset_id}: {config.worlds} worlds, "
            f"<= {config.max_root_candidates} candidates, depth {config.rollout_depth}"
        )

    # -- the direct move ---------------------------------------------------

    def direct_action(self, state, legal=None, *, move_model: "str | None" = None) -> int:
        """The greedy action of one move model, through the accepted adapter."""
        import torch

        from ...engine.legal_moves import legal_action_mask, legal_actions
        from ...engine.observation import build_observation
        from ...model.policy_adapter import (
            DECISION_MODE_GREEDY,
            prepare_legality,
            select_action,
        )
        from ...model.tokenization import (
            observation_batch_from_numpy,
            observation_to_tokens,
        )

        legal = list(legal_actions(state)) if legal is None else list(legal)
        name = move_model or self.mode_move_model(self.mode)
        model = self.models.move_models[name].model
        actor = state.acting_player
        mask = legal_action_mask(state, legal)
        observation = build_observation(state, actor)
        batch = observation_batch_from_numpy([observation], device=self.device)
        with torch.no_grad():
            outputs = model(observation_to_tokens(batch))
        row = outputs.policy_logits.detach().to("cpu", torch.float32)[0]
        legality = prepare_legality(legal, mask, actor)
        return int(
            select_action(row, legality, decision_mode=DECISION_MODE_GREEDY).absolute_action_id
        )

    # -- one decision ------------------------------------------------------

    def decide(
        self,
        state,
        *,
        legal=None,
        mode: "str | None" = None,
        seed: "int | None" = None,
        deadline_override: "float | None" = None,
        force_error: bool = False,
    ) -> Phase15PlayerDecision:
        """One capped decision that always returns a legal move."""
        from ...engine.legal_moves import legal_actions

        mode = self.mode if mode is None else self.check_mode(mode)
        legal = list(legal_actions(state)) if legal is None else list(legal)
        if not legal:
            raise Phase15PlayerError("a decision was requested with no legal action")
        move_model = self.mode_move_model(mode)
        started = time.perf_counter()

        if self._is_direct(mode):
            action, reason = self._direct_or_last_resort(state, legal, move_model)
            return self._record(
                mode=mode,
                move_model=move_model,
                provider=None,
                preset_id="direct",
                action=action,
                direct=action,
                searched=False,
                reason=reason,
                started=started,
                cap=None,
                seed=None,
                state=state,
                legal=legal,
                forwards=1,
                unique_worlds=None,
                margin=None,
            )

        bundle = self.systems[mode]
        cap = self.time_caps.get(mode)
        if seed is None:
            seed = player_seed_for(getattr(state, "game_id", "adhoc"), int(state.total_moves))
        deadline = deadline_override
        if deadline is None and cap is not None:
            deadline = started + float(cap)

        reason = None
        decision = None
        if force_error:
            reason = FALLBACK_SEARCH_ERROR
        else:
            try:
                decision = bundle.engine.choose_action(state, seed=int(seed), deadline=deadline)
            except Phase12SearchTimeout:
                reason = FALLBACK_TIMEOUT
            except Phase12SearchError:
                reason = FALLBACK_SEARCH_ERROR
            except Exception:  # noqa: BLE001 - never forfeit
                reason = FALLBACK_UNEXPECTED_ERROR

        if decision is not None:
            scores = [candidate.score for candidate in decision.candidates]
            if not all(score == score and abs(score) != float("inf") for score in scores):
                reason = FALLBACK_NON_FINITE
            elif int(decision.selected_action_id) not in set(legal):
                reason = FALLBACK_ILLEGAL_ACTION

        if reason is None:
            margin = None
            ordered = sorted(
                (candidate.score for candidate in decision.candidates), reverse=True
            )
            if len(ordered) >= 2:
                margin = float(ordered[0] - ordered[1])
            return self._record(
                mode=mode,
                move_model=move_model,
                provider=bundle.pairing.provider,
                preset_id=bundle.config.preset_id,
                action=int(decision.selected_action_id),
                direct=int(decision.direct_action_id),
                searched=True,
                reason=None,
                started=started,
                cap=cap,
                seed=int(seed),
                state=state,
                legal=legal,
                forwards=int(decision.c1_forwards),
                unique_worlds=int(decision.unique_worlds),
                margin=margin,
            )

        action, direct_reason = self._direct_or_last_resort(state, legal, move_model)
        return self._record(
            mode=mode,
            move_model=move_model,
            provider=bundle.pairing.provider,
            preset_id=bundle.config.preset_id,
            action=action,
            direct=action,
            searched=False,
            reason=direct_reason or reason,
            started=started,
            cap=cap,
            seed=int(seed),
            state=state,
            legal=legal,
            forwards=1,
            unique_worlds=None,
            margin=None,
        )

    def _direct_or_last_resort(self, state, legal, move_model: str):
        try:
            action = self.direct_action(state, legal, move_model=move_model)
        except Exception:  # noqa: BLE001 - the last resort exists for exactly this
            return int(min(legal)), FALLBACK_DIRECT_ERROR
        if action not in set(legal):  # pragma: no cover - the adapter checks first
            return int(min(legal)), FALLBACK_DIRECT_ERROR
        return int(action), None

    def _record(
        self,
        *,
        mode,
        move_model,
        provider,
        preset_id,
        action,
        direct,
        searched,
        reason,
        started,
        cap,
        seed,
        state,
        legal,
        forwards,
        unique_worlds,
        margin,
    ) -> Phase15PlayerDecision:
        seconds = time.perf_counter() - started
        self.decisions += 1
        self.total_seconds += seconds
        if searched:
            self.searched_decisions += 1
        if reason:
            self.fallback_counts[reason] = self.fallback_counts.get(reason, 0) + 1
        changed = int(action) != int(direct)
        if changed:
            self.move_changes += 1
        return Phase15PlayerDecision(
            mode=mode,
            move_model=move_model,
            provider=provider,
            preset_id=preset_id,
            action_id=int(action),
            direct_action_id=int(direct),
            move_changed=changed,
            searched=searched,
            fallback_reason=reason,
            seconds=seconds,
            time_cap=cap,
            seed=seed,
            ply=int(state.total_moves),
            legal_actions=len(legal),
            c1_forwards=int(forwards),
            unique_worlds=unique_worlds,
            score_margin=margin,
        )

    # -- description -------------------------------------------------------

    def status(self) -> dict:
        return {
            "player_version": PLAYER_VERSION,
            "mode": self.mode,
            "decisions": self.decisions,
            "searched_decisions": self.searched_decisions,
            "move_changes": self.move_changes,
            "move_change_rate": (
                round(self.move_changes / self.decisions, 5) if self.decisions else None
            ),
            "fallbacks": dict(self.fallback_counts),
            "fallback_rate": (
                round(sum(self.fallback_counts.values()) / self.decisions, 5)
                if self.decisions
                else None
            ),
            "mean_seconds": (
                round(self.total_seconds / self.decisions, 5) if self.decisions else None
            ),
        }

    def describe(self) -> dict:
        modes = {}
        for name in PLAYER_MODES:
            if self._is_direct(name):
                modes[name] = {
                    "kind": "direct",
                    "move_model": self.mode_move_model(name),
                    "budget": "direct, no search",
                    "time_cap_seconds": self.time_caps.get(name),
                    "available": True,
                }
            elif name in self.systems:
                bundle = self.systems[name]
                modes[name] = {
                    "kind": "search",
                    "move_model": bundle.pairing.move_model,
                    "provider": bundle.pairing.provider,
                    "budget": self.budget_text(name),
                    "time_cap_seconds": self.time_caps.get(name),
                    "available": True,
                    "identities": {
                        "move_model": bundle.identities.get("move_model", {}),
                        "belief_model": bundle.identities.get("belief_model", {}),
                    },
                }
            else:
                modes[name] = {"available": False, "reason": "not built in this player"}
        return {
            "player_version": PLAYER_VERSION,
            "integration_version": INTEGRATION_VERSION,
            "search_version": PHASE15_SEARCH_VERSION,
            "score_definition": PHASE15_SCORE_DEFINITION,
            "modes": modes,
            "fallback_policy": "the same mode's direct P18/P24 legal move",
            "fallback_reasons": list(FALLBACK_REASONS),
            "oracle_available_in_production": ORACLE_AVAILABLE_IN_PRODUCTION,
            "latency_ceiling_seconds": ACCEPTABLE_MOVE_SECONDS,
            "status": self.status(),
        }


# ---------------------------------------------------------------------------
# Match seat
# ---------------------------------------------------------------------------


class Phase15PlayerSeat:
    """The working player as a match seat, for machine-vs-machine play."""

    kind = "player"

    def __init__(self, player: Phase15SearchPlayer, mode: "str | None" = None) -> None:
        self.player = player
        self.mode = player.check_mode(mode) if mode else player.mode
        self.pairing = (
            pairing_of(f"{player.mode_move_model(self.mode)}_direct")
            if player._is_direct(self.mode)
            else player.systems[self.mode].pairing
        )
        self.arm_id = f"player|{self.mode}"

    def decide(self, state, legal, spec, plan):
        from .contract import search_seed_for

        seed = search_seed_for(plan.board_id, int(state.total_moves))
        decision = self.player.decide(state, legal=legal, mode=self.mode, seed=seed)
        return decision.action_id, {
            "ply": decision.ply,
            "seconds": decision.seconds,
            "legal_actions": decision.legal_actions,
            "move_changed": decision.move_changed if decision.searched else None,
            "c1_forwards": decision.c1_forwards,
            "unique_worlds": decision.unique_worlds,
            "candidates": None,
            "fallback": decision.fallback_reason,
            "direct_action_id": decision.direct_action_id,
        }

    def describe(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "mode": self.mode,
            "pairing": self.pairing.describe(),
            "player": self.player.describe(),
        }


__all__ = [
    "CANDIDATE_ARTIFACT",
    "DIAGNOSTIC_MODES",
    "FALLBACK_REASONS",
    "MODE_MAX_STRENGTH",
    "MODE_P18_DIRECT",
    "MODE_P24_DIRECT",
    "MODE_SELECTED",
    "PLAYER_MODES",
    "PLAYER_VERSION",
    "Phase15PlayerDecision",
    "Phase15PlayerError",
    "Phase15PlayerSeat",
    "Phase15SearchPlayer",
    "REQUIRED_MODES",
    "player_seed_for",
]
