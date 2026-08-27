"""Phase 16 Agent 2: the two sampling knobs as deployable objects.

Specification source: `02_AGENT_2_STOCHASTIC_SEARCH.md` sections 2, 4, 6.

The layering
------------
```text
StochasticArm            (tau, tau_r, top_p) — one named grid point
build_stochastic_bundle  the frozen Phase 15 system, or the sampled-rollout
                         engine, under one accepted preset
sample_move              a ~ softmax(S(a)/tau) over the engine's candidates
StochasticSeat           the match seat: accepted seed streams, accepted
                         fallback chain, one extra draw per decision
Phase16VariedPlayer      the two working modes for play_phase16.py
```

Bit-identity at zero temperature
--------------------------------
`tau_r = 0` builds the *frozen* Phase 15 bundle through the accepted
:func:`~stratego.search.phase15.systems.build_engine` — not a copy of it —
and `tau = 0` returns the engine decision's own `selected_action_id`
untouched. The zero-temperature arm therefore *is* the accepted player's
decision path, and the regression test proves it replays the frozen Phase 15
Stage A decisions bit-identically.

Oracle refusals, preserved
--------------------------
Every constructor here goes through the accepted production builders, which
refuse the oracle by name, by provider factory, by engine construction and
by the pairing table. This module adds a fifth refusal: `StochasticArm`
bundles are built only for pairings of kind `search`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from ..phase12.contract import Phase12SearchError, Phase12SearchTimeout
from ..phase15.contract import (
    Pairing,
    pairing as pairing_of,
    preset as preset_of,
    search_seed_for,
)
from ..phase15.systems import SystemBundle, build_engine
from .contract import (
    ACCEPTABLE_MOVE_SECONDS,
    MODE_VARIED_STRENGTH,
    ROLLOUT_TOP_P,
    STOCHASTIC_PAIRING,
    STOCHASTIC_VERSION,
    VARIED_MODE_PRESETS,
    VARIED_MODES,
    Phase16StochasticError,
    arm_name,
    move_sample_seed,
    rollout_sample_seed,
)
from .engine import Phase16StochasticEngine


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StochasticArm:
    """One (tau, tau_r) grid point over one production pairing."""

    tau: float
    tau_r: float
    top_p: float = ROLLOUT_TOP_P
    pairing_id: str = STOCHASTIC_PAIRING

    def __post_init__(self) -> None:
        if not float(self.tau) >= 0.0:
            raise Phase16StochasticError(f"tau must be >= 0, got {self.tau!r}")
        if not float(self.tau_r) >= 0.0:
            raise Phase16StochasticError(f"tau_r must be >= 0, got {self.tau_r!r}")
        if not 0.0 < float(self.top_p) <= 1.0:
            raise Phase16StochasticError(f"top_p must be in (0, 1], got {self.top_p!r}")
        target = pairing_of(self.pairing_id)
        if target.kind != "search":
            raise Phase16StochasticError(
                f"a stochastic arm runs over a production search pairing; "
                f"{self.pairing_id!r} is kind {target.kind!r}"
            )

    @property
    def arm_id(self) -> str:
        return arm_name(self.tau, self.tau_r)

    @property
    def is_control(self) -> bool:
        return float(self.tau) == 0.0 and float(self.tau_r) == 0.0

    @property
    def samples_rollouts(self) -> bool:
        return float(self.tau_r) > 0.0

    def describe(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "stochastic_version": STOCHASTIC_VERSION,
            "pairing_id": self.pairing_id,
            "tau": float(self.tau),
            "tau_r": float(self.tau_r),
            "top_p": float(self.top_p),
            "move_sampling": (
                "argmax S (frozen)" if self.tau == 0 else f"a ~ softmax(S/{self.tau})"
            ),
            "rollout_sampling": (
                "greedy (frozen)"
                if self.tau_r == 0
                else f"model distribution at tau_r={self.tau_r}, top_p={self.top_p}"
            ),
        }


def build_stochastic_bundle(
    models,
    arm: StochasticArm,
    preset: str,
    *,
    device: str = "cpu",
) -> SystemBundle:
    """The complete system one arm decides with, under one accepted preset.

    `tau_r = 0`: the accepted Phase 15 builder, unchanged — the frozen
    engine object itself. `tau_r > 0`: the same provider, model, identities
    and configuration, driving :class:`Phase16StochasticEngine`.
    """
    if not arm.samples_rollouts:
        return build_engine(arm.pairing_id, models, preset, device=device)
    from ..phase15.providers import build_phase15_provider

    target: Pairing = pairing_of(arm.pairing_id)
    config = replace(preset_of(preset), production=True)
    provider = build_phase15_provider(
        target.provider, models, production=True, device=device
    )
    move = models.move_models[target.move_model]
    engine = Phase16StochasticEngine(
        move.model,
        provider,
        config,
        rollout_temperature=arm.tau_r,
        rollout_top_p=arm.top_p,
        device=device,
        model_identity=dict(move.identity),
    )
    belief_identity = dict(models.specialists[target.provider].identity)
    return SystemBundle(
        pairing=target,
        config=config,
        engine=engine,
        provider=provider,
        identities={
            "move_model": dict(move.identity),
            "belief_model": belief_identity,
        },
    )


# ---------------------------------------------------------------------------
# The root move draw
# ---------------------------------------------------------------------------


def move_distribution(decision, tau: float) -> "tuple[list[int], np.ndarray]":
    """`(absolute action ids, probabilities)` of the softmax(S/tau) draw.

    Candidate order is the engine's own (descending prior, the accepted
    tie-break), so the cumulative-sum draw is deterministic given the rng.
    """
    actions = [int(candidate.absolute_action_id) for candidate in decision.candidates]
    scores = np.asarray(
        [float(candidate.score) for candidate in decision.candidates], dtype=np.float64
    )
    if not np.isfinite(scores).all():
        raise Phase12SearchError(
            "non-finite candidate scores reached the move sampler; the caller "
            "should have fallen back before sampling"
        )
    if float(tau) == 0.0:
        probabilities = np.zeros(len(actions), dtype=np.float64)
        probabilities[int(np.argmax(scores == scores.max()))] = 1.0
        return actions, probabilities
    scaled = scores / float(tau)
    shifted = np.exp(scaled - scaled.max())
    return actions, shifted / shifted.sum()


def sample_move(decision, tau: float, rng: "np.random.Generator | None") -> "tuple[int, dict]":
    """The played action under move-sampling knob 1, plus its record.

    `tau = 0` returns the frozen decision's own selected action and consumes
    no randomness — the argmax control is the accepted engine's choice
    object, untouched. `tau > 0` draws once from `softmax(S/tau)` over the
    candidate set.
    """
    argmax_action = int(decision.selected_action_id)
    if float(tau) == 0.0:
        return argmax_action, {
            "tau": 0.0,
            "sampled": False,
            "argmax_action_id": argmax_action,
            "changed_from_argmax": False,
            "candidates": len(decision.candidates),
            "modal_probability": 1.0,
            "entropy_nats": 0.0,
        }
    if rng is None:
        raise Phase16StochasticError("tau > 0 needs a seeded generator")
    actions, probabilities = move_distribution(decision, tau)
    cumulative = np.cumsum(probabilities)
    draw = float(rng.random())
    index = int(np.searchsorted(cumulative, draw, side="right"))
    index = min(index, len(actions) - 1)
    chosen = int(actions[index])
    entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
    return chosen, {
        "tau": float(tau),
        "sampled": True,
        "argmax_action_id": argmax_action,
        "changed_from_argmax": chosen != argmax_action,
        "candidates": len(actions),
        "chosen_probability": float(probabilities[index]),
        "modal_probability": float(probabilities.max()),
        "entropy_nats": entropy,
    }


def move_rng(arm: StochasticArm, identifier: str, ply: int, replay: int = 0) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64(move_sample_seed(arm.tau, arm.tau_r, identifier, ply, replay))
    )


def rollout_seed_for_arm(arm: StochasticArm, identifier: str, ply: int, replay: int = 0) -> int:
    return rollout_sample_seed(arm.tau_r, arm.top_p, identifier, ply, replay)


# ---------------------------------------------------------------------------
# The match seat
# ---------------------------------------------------------------------------


class StochasticSeat:
    """One stochastic arm as a Phase 15 match seat.

    The accepted seed streams (worlds from `search_seed_for(board_id, ply)`),
    the accepted fallback chain (timeout / search error / non-finite score /
    illegal result -> the same move model's direct legal move; never forfeit),
    plus one seeded draw per decision. At `tau = tau_r = 0` this seat replays
    the accepted `SearchSeat` decision for decision — a test pins it.
    """

    kind = "stochastic_search"

    def __init__(
        self,
        arm: StochasticArm,
        bundle: SystemBundle,
        *,
        owners: "dict | None" = None,
        time_cap: "float | None" = None,
    ) -> None:
        from ...evaluation.neural_worker import (
            DECISION_MODE_GREEDY,
            LocalInferenceChannel,
            RemoteNeuralPolicy,
        )
        from ..phase15.matchplay import player_ref

        if bundle.pairing.pairing_id != arm.pairing_id:
            raise Phase16StochasticError(
                f"arm {arm.arm_id} wants pairing {arm.pairing_id!r} but the bundle "
                f"carries {bundle.pairing.pairing_id!r}"
            )
        if bundle.pairing.kind != "search":  # pragma: no cover - arm refuses first
            raise Phase16StochasticError("a stochastic seat needs a search pairing")
        self.arm = arm
        self.bundle = bundle
        self.pairing = bundle.pairing
        self.arm_id = arm.arm_id
        self.engine = bundle.engine
        self.time_cap = None if time_cap is None else float(time_cap)
        self.fallbacks: dict[str, int] = {}
        self.sampled_changes = 0
        self.decisions = 0
        self.direct_policy = None
        if owners is not None and bundle.pairing.move_model in owners:
            self.direct_policy = RemoteNeuralPolicy(
                player_ref(),
                LocalInferenceChannel(owners[bundle.pairing.move_model]),
                decision_mode=DECISION_MODE_GREEDY,
            )

    def _fallback(self, state, legal, spec, plan, reason: str, started: float):
        from ...evaluation.policy import build_policy_input

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
            "sampled_move_changed": None,
        }

    def _choose(self, state, seed: int, rollout_seed: int, deadline):
        if isinstance(self.engine, Phase16StochasticEngine):
            return self.engine.choose_action(
                state, seed=seed, deadline=deadline, rollout_seed=rollout_seed
            )
        return self.engine.choose_action(state, seed=seed, deadline=deadline)

    def decide(self, state, legal, spec, plan):
        seed = search_seed_for(plan.board_id, int(state.total_moves))
        started = time.perf_counter()
        deadline = None if self.time_cap is None else started + self.time_cap
        rollout_seed = rollout_seed_for_arm(
            self.arm, plan.board_id, int(state.total_moves)
        )
        try:
            decision = self._choose(state, seed, rollout_seed, deadline)
        except Phase12SearchError as error:
            reason = (
                "timeout"
                if isinstance(error, Phase12SearchTimeout)
                else "search_error"
            )
            return self._fallback(state, legal, spec, plan, reason, started)
        scores = [candidate.score for candidate in decision.candidates]
        if not all(score == score and abs(score) != float("inf") for score in scores):
            return self._fallback(state, legal, spec, plan, "non_finite_score", started)
        rng = move_rng(self.arm, plan.board_id, int(state.total_moves))
        selected, sample_record = sample_move(decision, self.arm.tau, rng)
        if selected not in legal:
            return self._fallback(state, legal, spec, plan, "illegal_action", started)
        self.decisions += 1
        if sample_record["changed_from_argmax"]:
            self.sampled_changes += 1
        return selected, {
            "ply": int(state.total_moves),
            "seconds": time.perf_counter() - started,
            "legal_actions": int(decision.legal_action_count),
            "move_changed": int(selected) != int(decision.direct_action_id),
            "c1_forwards": int(decision.c1_forwards),
            "unique_worlds": int(decision.unique_worlds),
            "candidates": len(decision.candidates),
            "forward_seconds": float(decision.forward_seconds),
            "fallback": None,
            "direct_action_id": int(decision.direct_action_id),
            "argmax_action_id": int(decision.selected_action_id),
            "sampled_move_changed": bool(sample_record["changed_from_argmax"]),
            "score_margin": _score_margin(decision),
        }

    def describe(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "arm": self.arm.describe(),
            "pairing": self.pairing.describe(),
            "time_cap_seconds": self.time_cap,
            "fallbacks": dict(self.fallbacks),
            "sampled_move_changes": self.sampled_changes,
            "decisions": self.decisions,
            "seat": self.engine.describe(),
        }


def benchmark_seat_factory(
    *,
    models,
    owners,
    preset: str,
    device: str = "cpu",
    tau: float,
    tau_r: float,
    top_p: float = ROLLOUT_TOP_P,
    pairing_id: str = STOCHASTIC_PAIRING,
) -> StochasticSeat:
    """One stochastic arm as a seat for Agent 1's benchmark runner.

    Matches the factory signature `phase16` scoring runners call
    (`models, owners, preset, device, **kwargs`) and returns an object with
    the Phase 15 decision-seat interface (`decide`, `pairing`, `arm_id`).
    The oracle stays unreachable: `StochasticArm` refuses non-search
    pairings and the bundle builders run production-only.
    """
    arm = StochasticArm(float(tau), float(tau_r), top_p=float(top_p), pairing_id=pairing_id)
    bundle = build_stochastic_bundle(models, arm, preset, device=device)
    return StochasticSeat(arm, bundle, owners=owners)


def _score_margin(decision) -> "float | None":
    scores = sorted((candidate.score for candidate in decision.candidates), reverse=True)
    if len(scores) < 2:
        return None
    return float(scores[0] - scores[1])


# ---------------------------------------------------------------------------
# The working player (brief section 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase16PlayerDecision:
    """One varied-mode decision, with everything a log line needs."""

    mode: str
    arm_id: str
    move_model: str
    provider: "str | None"
    preset_id: str
    action_id: int
    direct_action_id: int
    argmax_action_id: "int | None"
    move_changed: bool
    sampled_move_changed: "bool | None"
    searched: bool
    fallback_reason: "str | None"
    seconds: float
    time_cap: "float | None"
    seed: "int | None"
    move_seed: "int | None"
    rollout_seed: "int | None"
    ply: int
    legal_actions: int
    c1_forwards: int
    unique_worlds: "int | None"
    entropy_nats: "float | None" = None

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "arm_id": self.arm_id,
            "move_model": self.move_model,
            "provider": self.provider,
            "preset_id": self.preset_id,
            "action_id": self.action_id,
            "direct_action_id": self.direct_action_id,
            "argmax_action_id": self.argmax_action_id,
            "move_changed": self.move_changed,
            "sampled_move_changed": self.sampled_move_changed,
            "searched": self.searched,
            "fallback_reason": self.fallback_reason,
            "seconds": round(self.seconds, 5),
            "time_cap_seconds": self.time_cap,
            "seed": self.seed,
            "move_seed": self.move_seed,
            "rollout_seed": self.rollout_seed,
            "ply": self.ply,
            "legal_actions": self.legal_actions,
            "c1_forwards": self.c1_forwards,
            "unique_worlds": self.unique_worlds,
            "entropy_nats": self.entropy_nats,
        }

    def log_line(self) -> str:
        tail = f" FALLBACK={self.fallback_reason}" if self.fallback_reason else ""
        varied = ""
        if self.sampled_move_changed:
            varied = " [sampled away from argmax]"
        return (
            f"[phase16 {self.mode}] ply={self.ply} budget={self.preset_id} "
            f"action={self.action_id} changed={int(self.move_changed)} "
            f"{self.seconds:.3f}s/cap={self.time_cap}{tail}{varied}"
        )


class Phase16VariedPlayer:
    """The two varied modes over one selected stochastic arm.

    Same information boundary as the Phase 15 player: legal knowledge only,
    the oracle refused by name here and structurally absent from every
    bundle this class can hold. Same never-forfeit rule, with the same
    counted reasons.
    """

    version = STOCHASTIC_VERSION

    def __init__(
        self,
        arm: StochasticArm,
        bundles: "dict[str, SystemBundle]",
        models,
        *,
        time_caps: "dict | None" = None,
        device: str = "cpu",
    ) -> None:
        missing = [mode for mode in VARIED_MODES if mode not in bundles]
        if missing:
            raise Phase16StochasticError(
                f"the varied player needs bundles for {list(VARIED_MODES)}; missing "
                f"{missing}"
            )
        for mode, bundle in bundles.items():
            if bundle.pairing.kind == "diagnostic":
                raise Phase16StochasticError(
                    f"{mode!r} carries the oracle; the varied player has no "
                    "diagnostic mode"
                )
            if bundle.provider is not None and getattr(
                bundle.provider, "uses_hidden_truth", False
            ):
                raise Phase16StochasticError(
                    f"{mode!r} carries a provider that reads hidden truth"
                )
        self.arm = arm
        self.bundles = dict(bundles)
        self.models = models
        self.device = device
        self.time_caps = dict(time_caps or {})
        self.fallback_counts: dict[str, int] = {}
        self.decisions = 0
        self.move_changes = 0
        self.sampled_changes = 0

    @staticmethod
    def check_mode(mode: str) -> str:
        if mode == "oracle" or (isinstance(mode, str) and mode.endswith("_oracle")):
            raise Phase16StochasticError(
                "the oracle is an offline diagnostic and is not a player mode"
            )
        if mode not in VARIED_MODES:
            raise Phase16StochasticError(
                f"unknown varied mode {mode!r}; Phase 16 varied modes are "
                f"{list(VARIED_MODES)}"
            )
        return mode

    def budget_text(self, mode: str) -> str:
        config = self.bundles[self.check_mode(mode)].config
        return (
            f"{config.preset_id}: {config.worlds} worlds, "
            f"<= {config.max_root_candidates} candidates, depth {config.rollout_depth}"
        )

    def _direct_or_last_resort(self, state, legal):
        from ..phase15.player import Phase15SearchPlayer

        try:
            action = Phase15SearchPlayer.direct_action(
                self, state, legal, move_model=pairing_of(self.arm.pairing_id).move_model
            )
        except Exception:  # noqa: BLE001 - the last resort exists for exactly this
            return int(min(legal)), "direct_error"
        if action not in set(legal):  # pragma: no cover - the adapter checks first
            return int(min(legal)), "direct_error"
        return int(action), None

    @property
    def mode(self):  # for Phase15SearchPlayer.direct_action reuse
        return MODE_VARIED_STRENGTH

    def mode_move_model(self, mode: str) -> str:
        return pairing_of(self.arm.pairing_id).move_model

    def decide(
        self,
        state,
        *,
        legal=None,
        mode: str = MODE_VARIED_STRENGTH,
        game_id: "str | None" = None,
    ) -> Phase16PlayerDecision:
        from ...engine.legal_moves import legal_actions

        mode = self.check_mode(mode)
        legal = list(legal_actions(state)) if legal is None else list(legal)
        if not legal:
            raise Phase16StochasticError("a decision was requested with no legal action")
        bundle = self.bundles[mode]
        identifier = str(game_id or getattr(state, "game_id", "adhoc"))
        ply = int(state.total_moves)
        cap = self.time_caps.get(mode)
        started = time.perf_counter()
        deadline = None if cap is None else started + float(cap)
        seed = search_seed_for(identifier, ply)
        move_seed = move_sample_seed(self.arm.tau, self.arm.tau_r, identifier, ply)
        rollout_seed = rollout_seed_for_arm(self.arm, identifier, ply)

        reason = None
        decision = None
        try:
            if isinstance(bundle.engine, Phase16StochasticEngine):
                decision = bundle.engine.choose_action(
                    state, seed=seed, deadline=deadline, rollout_seed=rollout_seed
                )
            else:
                decision = bundle.engine.choose_action(state, seed=seed, deadline=deadline)
        except Phase12SearchTimeout:
            reason = "timeout"
        except Phase12SearchError:
            reason = "search_error"
        except Exception:  # noqa: BLE001 - never forfeit
            reason = "unexpected_error"

        selected = None
        sample_record: dict = {}
        if decision is not None:
            scores = [candidate.score for candidate in decision.candidates]
            if not all(score == score and abs(score) != float("inf") for score in scores):
                reason = "non_finite_score"
            else:
                rng = np.random.Generator(np.random.PCG64(int(move_seed)))
                selected, sample_record = sample_move(decision, self.arm.tau, rng)
                if selected not in set(legal):
                    reason = "illegal_action"
                    selected = None

        if reason is None and selected is not None:
            self.decisions += 1
            changed = int(selected) != int(decision.direct_action_id)
            if changed:
                self.move_changes += 1
            if sample_record.get("changed_from_argmax"):
                self.sampled_changes += 1
            return Phase16PlayerDecision(
                mode=mode,
                arm_id=self.arm.arm_id,
                move_model=bundle.pairing.move_model,
                provider=bundle.pairing.provider,
                preset_id=bundle.config.preset_id,
                action_id=int(selected),
                direct_action_id=int(decision.direct_action_id),
                argmax_action_id=int(decision.selected_action_id),
                move_changed=changed,
                sampled_move_changed=bool(sample_record.get("changed_from_argmax")),
                searched=True,
                fallback_reason=None,
                seconds=time.perf_counter() - started,
                time_cap=cap,
                seed=int(seed),
                move_seed=int(move_seed),
                rollout_seed=int(rollout_seed) if self.arm.samples_rollouts else None,
                ply=ply,
                legal_actions=len(legal),
                c1_forwards=int(decision.c1_forwards),
                unique_worlds=int(decision.unique_worlds),
                entropy_nats=sample_record.get("entropy_nats"),
            )

        action, direct_reason = self._direct_or_last_resort(state, legal)
        final_reason = direct_reason or reason
        self.decisions += 1
        self.fallback_counts[final_reason] = self.fallback_counts.get(final_reason, 0) + 1
        return Phase16PlayerDecision(
            mode=mode,
            arm_id=self.arm.arm_id,
            move_model=bundle.pairing.move_model,
            provider=bundle.pairing.provider,
            preset_id=bundle.config.preset_id,
            action_id=int(action),
            direct_action_id=int(action),
            argmax_action_id=None,
            move_changed=False,
            sampled_move_changed=None,
            searched=False,
            fallback_reason=final_reason,
            seconds=time.perf_counter() - started,
            time_cap=cap,
            seed=int(seed),
            move_seed=int(move_seed),
            rollout_seed=int(rollout_seed) if self.arm.samples_rollouts else None,
            ply=ply,
            legal_actions=len(legal),
            c1_forwards=1,
            unique_worlds=None,
        )

    def status(self) -> dict:
        return {
            "player_version": STOCHASTIC_VERSION,
            "arm_id": self.arm.arm_id,
            "decisions": self.decisions,
            "move_changes": self.move_changes,
            "sampled_move_changes": self.sampled_changes,
            "fallbacks": dict(self.fallback_counts),
        }

    def describe(self) -> dict:
        modes = {}
        for mode in VARIED_MODES:
            bundle = self.bundles[mode]
            modes[mode] = {
                "kind": "stochastic_search",
                "arm": self.arm.describe(),
                "move_model": bundle.pairing.move_model,
                "provider": bundle.pairing.provider,
                "budget": self.budget_text(mode),
                "time_cap_seconds": self.time_caps.get(mode),
                "identities": {
                    "move_model": bundle.identities.get("move_model", {}),
                    "belief_model": bundle.identities.get("belief_model", {}),
                },
            }
        return {
            "player_version": STOCHASTIC_VERSION,
            "arm": self.arm.describe(),
            "modes": modes,
            "mode_presets": dict(VARIED_MODE_PRESETS),
            "fallback_policy": "the same move model's direct legal move; never forfeit",
            "oracle_available_in_production": False,
            "latency_ceiling_seconds": ACCEPTABLE_MOVE_SECONDS,
            "status": self.status(),
        }


__all__ = [
    "Phase16PlayerDecision",
    "Phase16VariedPlayer",
    "StochasticArm",
    "StochasticSeat",
    "benchmark_seat_factory",
    "build_stochastic_bundle",
    "move_distribution",
    "move_rng",
    "rollout_seed_for_arm",
    "sample_move",
]
