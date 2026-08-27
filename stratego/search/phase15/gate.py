"""Phase 15 Agent 2 section 9: the correctness gate.

Specification source: `02_AGENT_2_SEARCH_IMPLEMENTATION.md` sections 8, 9.

No match pack may start until every check here passes. The checks are written
to be *mechanical* — each one either observes the property on real fresh
positions or raises — rather than documentary, because the whole value of the
gate is that a wiring mistake between two nearly identical models cannot get
past it.

The role checks are the interesting ones
----------------------------------------
"P18 pairings use P18 for policy/value/rollouts/fallback" is easy to assert
and hard to prove. Two independent observations prove it here:

*positive*  the engine's own direct action and root value equal what the P18
            model alone produces on the same state, forward for forward;
*negative*  on the same positions P24 disagrees somewhere, so the positive
            check has power and is not passing because the two models happen
            to answer identically everywhere.

"B18/B24 are used only for beliefs" gets the same treatment: for one move
model, the engine's direct action is required to be *identical* across all
three belief providers — a belief model that touched the policy could not
leave it unchanged — while the sampled worlds are required to differ between
`remaining_count` and the learned providers, so the belief model is
demonstrably being consulted at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ...engine.legal_moves import legal_actions
from ...engine.permutation import permute_hidden_identities
from ..phase12.contract import Phase12SearchTimeout
from .contract import (
    COMBINED_PAIRING_IDS,
    LEARNED_PROVIDERS,
    MOVE_MODELS,
    PRODUCTION_PROVIDERS,
    PROVIDER_ORACLE,
    PROVIDER_REMAINING_COUNT,
    Phase15SearchError,
    pairing as pairing_of,
)
from .systems import build_engine

#: The gate identity a report and the frozen candidate record.
GATE_VERSION = "phase15_correctness_gate_v1"


class Phase15GateError(Phase15SearchError):
    """A section 9 correctness check failed."""


@dataclass
class GateResult:
    """The gate's findings. `passed` is true only if every check passed."""

    checks: dict
    seconds: float

    @property
    def passed(self) -> bool:
        return all(entry.get("passed") for entry in self.checks.values())

    def summary(self) -> dict:
        return {
            "gate_version": GATE_VERSION,
            "passed": self.passed,
            "checks_run": len(self.checks),
            "checks_passed": sum(
                1 for entry in self.checks.values() if entry.get("passed")
            ),
            "failed": sorted(
                name for name, entry in self.checks.items() if not entry.get("passed")
            ),
            "seconds": round(self.seconds, 3),
            "checks": self.checks,
        }


# ---------------------------------------------------------------------------
# Fresh gate positions
# ---------------------------------------------------------------------------


def gate_positions(owners, sources, *, games: int = 4, per_game: int = 3) -> list:
    """Fresh, orientation-gated mid-game states for the checks to run on."""
    from ...engine.state import create_game
    from ...engine.transition import apply_action
    from ...evaluation.match_spec import EVALUATION_RULES
    from .boards import board_plan
    from .contract import MATCH_OPPONENTS, MATCH_SETUP_SOURCES
    from .positions import POSITION_ORDINAL_BASE, play_for_positions

    states = []
    for index in range(int(games)):
        opponent = MATCH_OPPONENTS[index % len(MATCH_OPPONENTS)]
        source = MATCH_SETUP_SOURCES[index % len(MATCH_SETUP_SOURCES)]
        color = ("red", "blue")[index % 2]
        plan = board_plan(
            opponent, source, color, POSITION_ORDINAL_BASE + 500 + index, sources
        )
        observer = MOVE_MODELS[index % len(MOVE_MODELS)]
        found = play_for_positions(plan, observer, owners, per_game=int(per_game))
        for position in found:
            state = create_game(
                plan.red_setup,
                plan.blue_setup,
                rules=EVALUATION_RULES,
                game_id=position.position_id,
            )
            for action in position.action_prefix:
                legal = legal_actions(state)
                apply_action(state, int(action), legal=legal)
            states.append((position, state, plan))
    if not states:  # pragma: no cover - the eligibility floors are generous
        raise Phase15GateError("no eligible gate position was produced")
    return states


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------



def _public_state(state):
    """The public view of a state, in the accepted provider input type."""
    from ...belief.phase11b.interface import Phase11BPublicState
    from ...engine.observation import build_observation
    from ...evaluation.phase11_public_state import build_public_state_document
    from ...evaluation.policy import build_public_view

    observer = state.acting_player
    observation = build_observation(state, observer)
    document = build_public_state_document(build_public_view(state, observer), observation)
    return Phase11BPublicState(document, observation)


def check_identities(models) -> dict:
    """All four pairings load, with the exact handoff digests."""
    handoff = models.handoff
    findings = []
    for name in MOVE_MODELS:
        identity = models.move_models[name].identity
        record = handoff["policy_models"][name]
        if identity["checkpoint_sha256"] != record["checkpoint_sha256"]:
            findings.append(f"{name}: checkpoint sha256 mismatch")
        if identity["model_state_digest"] != record["model_state_digest"]:
            findings.append(f"{name}: model state digest mismatch")
    for name in LEARNED_PROVIDERS:
        identity = models.specialists[name].identity
        record = handoff["belief_models"][name]
        if identity["checkpoint_sha256"] != record["checkpoint_sha256"]:
            findings.append(f"{name}: checkpoint sha256 mismatch")
        if identity["state_digest"] != record["state_digest"]:
            findings.append(f"{name}: state digest mismatch")
        if identity["holds_policy_parameters"] or identity["holds_value_parameters"]:
            findings.append(f"{name}: claims policy or value parameters")
    combinations = {}
    for pairing_id in COMBINED_PAIRING_IDS:
        bundle = build_engine(pairing_id, models, "TINY")
        combinations[pairing_id] = {
            "move_model_state_digest": bundle.identities["move_model"]["model_state_digest"],
            "belief_state_digest": bundle.identities["belief_model"]["state_digest"],
            "belief_prefix_backbone": bundle.identities["belief_model"]["prefix_backbone"],
            "applied_temperature": bundle.identities["belief_model"]["applied_temperature"],
        }
    return {
        "passed": not findings,
        "findings": findings,
        "pairings": combinations,
        "corpus_digest": (models.handoff.get("corpus") or {}).get("corpus_digest"),
    }


def check_decisions(models, states, *, preset: str = "TINY") -> dict:
    """Legality, seed reproducibility, shared worlds and the direct candidate."""
    findings = []
    observed = {
        "decisions": 0,
        "repeat_decisions": 0,
        "worlds_checked": 0,
        "candidates_checked": 0,
    }
    for pairing_id in COMBINED_PAIRING_IDS:
        bundle = build_engine(pairing_id, models, preset)
        for _position, state, plan in states:
            legal = set(legal_actions(state))
            first = bundle.engine.choose_action(state, seed=4242)
            again = bundle.engine.choose_action(state, seed=4242)
            observed["decisions"] += 1
            observed["repeat_decisions"] += 1
            if first.selected_action_id not in legal:
                findings.append(f"{pairing_id}: selected an illegal action")
            if first.selected_action_id != again.selected_action_id:
                findings.append(f"{pairing_id}: the same seed chose two actions")
            if first.world_weights != again.world_weights:
                findings.append(f"{pairing_id}: the same seed sampled different worlds")
            if sum(first.world_weights) != first.worlds_requested:
                findings.append(f"{pairing_id}: world weights do not sum to the budget")
            if not any(candidate.is_direct for candidate in first.candidates):
                findings.append(f"{pairing_id}: the direct action was not a candidate")
            widths = {len(candidate.world_values) for candidate in first.candidates}
            if widths != {first.unique_worlds}:
                findings.append(
                    f"{pairing_id}: candidates were not all evaluated on the same "
                    f"{first.unique_worlds} root worlds (widths {sorted(widths)})"
                )
            observed["worlds_checked"] += first.unique_worlds
            observed["candidates_checked"] += len(first.candidates)
    return {"passed": not findings, "findings": findings, **observed}


def check_worlds_legal(models, states, *, worlds: int = 8) -> dict:
    """Every sampled world passes Agent 1's accepted section 12 stack."""
    from ...belief.phase15.interface_checks import check_provider
    from ...evaluation.phase11_public_state import build_public_state_document
    from ...engine.observation import build_observation
    from ...evaluation.policy import build_public_view
    from ...belief.phase11b.interface import Phase11BPublicState
    from .providers import build_phase15_provider

    prepared = []
    for _position, state, plan in states:
        observer = state.acting_player
        observation = build_observation(state, observer)
        view = build_public_view(state, observer)
        document = build_public_state_document(view, observation)
        truth = {}
        prepared.append(
            (Phase11BPublicState(document, observation), truth, state.game_id, state.total_moves)
        )
    reports = {}
    findings = []
    for provider_id in LEARNED_PROVIDERS:
        provider = build_phase15_provider(provider_id, models)
        try:
            reports[provider_id] = check_provider(
                provider.belief_provider, prepared, worlds=int(worlds)
            )
        except Exception as error:  # noqa: BLE001 - the check reports its own refusals
            findings.append(f"{provider_id}: {error}")
    return {"passed": not findings, "findings": findings, "providers": reports}


def check_model_roles(models, states, *, preset: str = "TINY") -> dict:
    """P18 pairings run on P18; P24 pairings run on P24; beliefs touch neither."""
    import torch

    from ...engine.observation import build_observation
    from ...model.contract import expected_value
    from ...model.policy_adapter import (
        DECISION_MODE_GREEDY,
        prepare_legality,
        select_action,
    )
    from ...engine.legal_moves import legal_action_mask
    from ...model.tokenization import observation_batch_from_numpy, observation_to_tokens

    def direct_of(move_model: str, state):
        model = models.move_models[move_model].model
        legal = legal_actions(state)
        mask = legal_action_mask(state, legal)
        observation = build_observation(state, state.acting_player)
        batch = observation_batch_from_numpy([observation], device="cpu")
        with torch.no_grad():
            outputs = model(observation_to_tokens(batch))
        row = outputs.policy_logits.detach().to("cpu", torch.float32)[0]
        value = float(expected_value(outputs.value_logits).detach().to("cpu")[0])
        legality = prepare_legality(legal, mask, state.acting_player)
        chosen = select_action(row, legality, decision_mode=DECISION_MODE_GREEDY)
        return int(chosen.absolute_action_id), value

    findings = []
    matched = {name: 0 for name in MOVE_MODELS}
    discriminating = 0
    provider_invariant = 0
    provider_varied_worlds = 0
    for _position, state, plan in states:
        references = {name: direct_of(name, state) for name in MOVE_MODELS}
        if references[MOVE_MODELS[0]][0] != references[MOVE_MODELS[1]][0]:
            discriminating += 1
        for move_model in MOVE_MODELS:
            expected_action, expected_value_ = references[move_model]
            directs = set()
            world_signatures = {}
            for provider_id in PRODUCTION_PROVIDERS:
                bundle = build_engine(f"{move_model}_{provider_id}", models, preset)
                decision = bundle.engine.choose_action(state, seed=99)
                if decision.direct_action_id != expected_action:
                    findings.append(
                        f"{move_model}_{provider_id}: the engine's direct action "
                        f"{decision.direct_action_id} != {move_model.upper()}'s own "
                        f"{expected_action}"
                    )
                if abs(decision.root_direct_value - expected_value_) > 1e-5:
                    findings.append(
                        f"{move_model}_{provider_id}: the engine's root value "
                        f"{decision.root_direct_value} != {move_model.upper()}'s own "
                        f"{expected_value_}"
                    )
                directs.add(decision.direct_action_id)
                # The worlds themselves, not a proxy: two providers that
                # sampled the same armies would produce the same signature
                # even if their marginals differed, and that is exactly the
                # thing this check has to be able to see.
                world_signatures[provider_id] = tuple(
                    tuple(sorted(assignment.items()))
                    for assignment in bundle.provider.sample_assignments(
                        _public_state(state), bundle.config.worlds, 99
                    )
                )
            matched[move_model] += 1
            if len(directs) == 1:
                provider_invariant += 1
            else:
                findings.append(
                    f"{move_model}: the direct action changed with the belief "
                    "provider; a belief model reached a policy decision"
                )
            learned = {
                key: value
                for key, value in world_signatures.items()
                if key in LEARNED_PROVIDERS
            }
            if any(
                value != world_signatures[PROVIDER_REMAINING_COUNT]
                for value in learned.values()
            ):
                provider_varied_worlds += 1
    if provider_varied_worlds == 0:
        findings.append(
            "no position showed a difference between the count baseline and a "
            "learned provider; the belief model may not be consulted at all"
        )
    if discriminating == 0:
        findings.append(
            "P18 and P24 agreed on every gate position, so the positive role "
            "check has no power here"
        )
    return {
        "passed": not findings,
        "findings": findings,
        "positions": len(states),
        "matched_decisions": matched,
        "positions_where_p18_and_p24_differ": discriminating,
        "direct_action_provider_invariant": provider_invariant,
        "positions_where_search_differed_by_provider": provider_varied_worlds,
    }


def check_permutation_invariance(models, states, *, preset: str = "TINY") -> dict:
    """Section 8: the production answer must not change; oracle is the control."""
    import random

    from .contract import DOMAIN_PROBE, derive_search_seed

    findings = []
    checked = 0
    changed_assignments = 0
    oracle_sensitive = 0
    oracle_checks = 0
    for pairing_id in COMBINED_PAIRING_IDS:
        bundle = build_engine(pairing_id, models, preset)
        for position, state, plan in states:
            rng = random.Random(
                derive_search_seed(DOMAIN_PROBE, "gate", position.position_id)
            )
            permuted, info = permute_hidden_identities(state, state.acting_player, rng)
            base = bundle.engine.choose_action(state, seed=777)
            other = bundle.engine.choose_action(permuted, seed=777)
            checked += 1
            if info.get("changed"):
                changed_assignments += 1
            if base.selected_action_id != other.selected_action_id:
                findings.append(
                    f"{pairing_id}: the answer changed under a hidden-identity "
                    f"permutation at {position.position_id}"
                )
    for move_model in MOVE_MODELS:
        bundle = build_engine(
            f"{move_model}_{PROVIDER_ORACLE}", models, preset, production=False
        )
        for position, state, plan in states:
            rng = random.Random(
                derive_search_seed(DOMAIN_PROBE, "gate", position.position_id)
            )
            permuted, info = permute_hidden_identities(state, state.acting_player, rng)
            base = bundle.engine.choose_action(state, seed=777)
            other = bundle.engine.choose_action(permuted, seed=777)
            oracle_checks += 1
            if base.selected_action_id != other.selected_action_id:
                oracle_sensitive += 1
    if oracle_checks and oracle_sensitive == 0:
        findings.append(
            "the oracle positive control never changed its answer under a "
            "permutation; the test has no power"
        )
    return {
        "passed": not findings,
        "findings": findings,
        "production_checks": checked,
        "permutations_that_changed_assignments": changed_assignments,
        "oracle_checks": oracle_checks,
        "oracle_sensitive": oracle_sensitive,
    }


def check_oracle_refusals(models) -> dict:
    """The oracle is refused everywhere a production caller could reach it."""
    from .contract import check_production_provider
    from .providers import build_phase15_provider
    from .systems import build_systems

    refusals = {}
    findings = []

    def refuses(name, call):
        try:
            call()
        except Exception as error:  # noqa: BLE001 - a refusal is the pass condition
            refusals[name] = str(error).split(";")[0][:120]
            return True
        findings.append(f"{name}: the oracle was NOT refused")
        return False

    refuses("contract.check_production_provider", lambda: check_production_provider(PROVIDER_ORACLE))
    refuses(
        "providers.build_phase15_provider",
        lambda: build_phase15_provider(PROVIDER_ORACLE, models),
    )
    refuses("systems.build_engine", lambda: build_engine("p18_oracle", models, "TINY"))
    refuses(
        "systems.build_systems",
        lambda: build_systems(["p18_oracle"], models, "TINY"),
    )

    from ...search.phase12.contract import search_preset
    from ...search.phase12.engine import Phase12SearchEngine

    oracle = build_phase15_provider(
        PROVIDER_ORACLE, models, production=False, offline_diagnostic=True
    )
    refuses(
        "phase12 engine under production=True",
        lambda: Phase12SearchEngine(
            models.move("p18"), oracle, search_preset("TINY", production=True)
        ),
    )

    from .player import Phase15SearchPlayer

    refuses(
        "player.set_mode(oracle)",
        lambda: Phase15SearchPlayer.check_mode(PROVIDER_ORACLE),
    )
    return {"passed": not findings, "findings": findings, "refusals": refusals}


def check_fallback(models, states, *, preset: str = "TINY") -> dict:
    """A deadline and a forced error both return the right direct move."""
    from .matchplay import DirectSeat
    from .player import Phase15SearchPlayer

    findings = []
    observed = {"timeout_fallbacks": 0, "error_fallbacks": 0}
    for move_model in MOVE_MODELS:
        for provider_id in LEARNED_PROVIDERS:
            bundle = build_engine(f"{move_model}_{provider_id}", models, preset)
            player = Phase15SearchPlayer(
                bundle, models, mode="selected_search", time_caps={"selected_search": 60.0}
            )
            for _position, state, plan in states:
                legal = legal_actions(state)
                reference = player.direct_action(state, legal)

                # 1. an already-expired deadline
                timed = player.decide(state, legal=legal, deadline_override=time.perf_counter() - 1.0)
                if timed.fallback_reason != "timeout":
                    findings.append(
                        f"{move_model}_{provider_id}: an expired deadline did not "
                        f"fall back (reason {timed.fallback_reason!r})"
                    )
                elif timed.action_id != reference:
                    findings.append(
                        f"{move_model}_{provider_id}: the timeout fallback played "
                        f"{timed.action_id}, {move_model.upper()}'s direct move is "
                        f"{reference}"
                    )
                else:
                    observed["timeout_fallbacks"] += 1

                # 2. a forced search error
                broken = player.decide(state, legal=legal, force_error=True)
                if broken.fallback_reason != "search_error":
                    findings.append(
                        f"{move_model}_{provider_id}: a forced search error did not "
                        f"fall back (reason {broken.fallback_reason!r})"
                    )
                elif broken.action_id != reference:
                    findings.append(
                        f"{move_model}_{provider_id}: the error fallback played "
                        f"{broken.action_id}, {move_model.upper()}'s direct move is "
                        f"{reference}"
                    )
                else:
                    observed["error_fallbacks"] += 1
                if broken.action_id not in legal:
                    findings.append(
                        f"{move_model}_{provider_id}: a fallback played an illegal move"
                    )
    return {"passed": not findings, "findings": findings, **observed}


def latency_probe(models, states, presets=("TINY", "SMALL")) -> dict:
    """Section 9's small TINY/SMALL probes on all four pairings."""
    report = {}
    for preset in presets:
        for pairing_id in COMBINED_PAIRING_IDS:
            bundle = build_engine(pairing_id, models, preset)
            timings = []
            forwards = []
            unique = []
            for _position, state, plan in states:
                started = time.perf_counter()
                decision = bundle.engine.choose_action(state, seed=31337)
                timings.append(time.perf_counter() - started)
                forwards.append(decision.c1_forwards)
                unique.append(decision.unique_worlds)
            array = np.asarray(timings, dtype=np.float64)
            report[f"{pairing_id}|{preset}"] = {
                "decisions": len(timings),
                "mean_seconds": round(float(array.mean()), 4),
                "median_seconds": round(float(np.median(array)), 4),
                "p95_seconds": round(float(np.percentile(array, 95)), 4),
                "max_seconds": round(float(array.max()), 4),
                "mean_c1_forwards": round(float(np.mean(forwards)), 1),
                "mean_unique_worlds": round(float(np.mean(unique)), 2),
            }
    return {"passed": True, "probes": report}


def run_gate(
    models,
    owners,
    sources,
    *,
    games: int = 4,
    per_game: int = 3,
    preset: str = "TINY",
) -> GateResult:
    """Every section 9 check, on fresh positions, in one call."""
    started = time.perf_counter()
    states = gate_positions(owners, sources, games=games, per_game=per_game)
    checks = {
        "identities": check_identities(models),
        "decisions": check_decisions(models, states, preset=preset),
        "worlds_legal": check_worlds_legal(models, states),
        "model_roles": check_model_roles(models, states, preset=preset),
        "permutation_invariance": check_permutation_invariance(models, states, preset=preset),
        "oracle_refusals": check_oracle_refusals(models),
        "fallback": check_fallback(models, states, preset=preset),
        "latency_probe": latency_probe(models, states),
    }
    checks["gate_positions"] = {
        "passed": True,
        "positions": len(states),
        "plies": [int(position.ply) for position, _state, _plan in states],
        "unresolved": [int(position.unresolved) for position, _state, _plan in states],
    }
    return GateResult(checks=checks, seconds=time.perf_counter() - started)


__all__ = [
    "GATE_VERSION",
    "GateResult",
    "Phase15GateError",
    "check_decisions",
    "check_fallback",
    "check_identities",
    "check_model_roles",
    "check_oracle_refusals",
    "check_permutation_invariance",
    "check_worlds_legal",
    "gate_positions",
    "latency_probe",
    "run_gate",
]
