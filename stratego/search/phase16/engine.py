"""Phase 16 Agent 2: sampled rollouts over the frozen root-world search.

Specification source: `02_AGENT_2_STOCHASTIC_SEARCH.md` section 2, knob 2.

What changes and what cannot
----------------------------
:class:`Phase16StochasticEngine` subclasses the accepted
:class:`~stratego.search.phase12.engine.Phase12SearchEngine` and differs in
exactly one behaviour: when ``rollout_temperature > 0``, each rollout action
(both sides) is *drawn* from the move model's legal-action distribution at
temperature ``tau_r``, restricted to the smallest set covering ``top_p``
probability mass, instead of taken greedily. Root candidates, priors, the
direct action, world sampling, world deduplication, the deadline checks, the
score definition and the final argmax are the accepted code, byte for byte:

- ``rollout_temperature == 0`` does not re-implement anything — it
  **delegates to the accepted engine's own** ``choose_action``, so the
  control is bit-identical by construction, not merely by test;
- ``rollout_temperature > 0`` runs a faithful transcription of the accepted
  body in which the single line
  ``chosen_model = _greedy_model_action(policy_np[row], sim_model)`` becomes
  ``chosen_model = _sampled_model_action(rng, policy_np[row], sim_model, ...)``.
  A regression test pins the transcription to the accepted body at
  ``tau_r = 0`` semantics (same worlds, same candidates, same priors) so the
  copy cannot drift silently.

Determinism
-----------
The sampled path consumes randomness from exactly one
``numpy.random.Generator`` seeded by the caller-supplied ``rollout_seed``
(default: derived from the world seed under the Phase 16 stream). The
rollout loop iterates simulations in a deterministic order, so the same
``(seed, rollout_seed)`` reproduces the same decision — worlds, Q values and
chosen action — exactly.

World deduplication under sampled rollouts
------------------------------------------
The accepted engine evaluates duplicate sampled worlds once and weights by
multiplicity, which is *exact* for deterministic greedy rollouts. With
sampled rollouts it means duplicated worlds share one sampled rollout rather
than drawing independent ones. The brief holds deduplication byte-identical,
so this sharing is kept and stated rather than "fixed": it changes the
variance of the Q estimate, never its support, and `tau_r = 0` is unaffected.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ...belief.phase11b.interface import Phase11BPublicState
from ...engine.legal_moves import (
    generate_actions_for_player,
    legal_action_mask,
    legal_actions,
)
from ...engine.observation import build_observation
from ...engine.snapshot import create_snapshot, restore_snapshot
from ...engine.state import GameState
from ...engine.transition import apply_action
from ...evaluation.phase11_public_state import build_public_state_document
from ...evaluation.policy import build_public_view
from ...model.action_frame import (
    absolute_legal_actions_to_model,
    model_action_to_absolute,
)
from ...model.contract import expected_value
from ...model.policy_adapter import (
    DECISION_MODE_GREEDY,
    prepare_legality,
    select_action,
)
from ..phase12.contract import Phase12SearchError, Phase12SearchTimeout
from ..phase12.engine import (
    Phase12CandidateResult,
    Phase12SearchDecision,
    Phase12SearchEngine,
    _Simulation,
    apply_assignment_in_place,
)
from .contract import (
    DOMAIN_ROLLOUT_SAMPLE,
    ROLLOUT_SEARCH_VERSION,
    ROLLOUT_TOP_P,
    Phase16StochasticError,
    derive_stochastic_seed,
)


def _sampled_model_action(
    rng: np.random.Generator,
    row: np.ndarray,
    legal_model: np.ndarray,
    temperature: float,
    top_p: float,
) -> int:
    """One rollout action drawn from the model's legal distribution.

    ``softmax(row[legal]/temperature)`` restricted to the smallest set
    covering ``top_p`` probability mass (ties in the nucleus ordering break
    to the lowest normalized action id, the accepted adapter convention),
    renormalized, then sampled with one uniform draw. Raises the same
    refusal as the accepted greedy chooser on a non-finite legal logit.
    `legal_model` must be ascending, as in the accepted engine.
    """
    values = row[legal_model].astype(np.float64)
    if not np.isfinite(values).all():
        raise Phase12SearchError(
            "the model produced a non-finite logit on a legal action during a "
            "rollout; refusing to choose a move from an untrustworthy distribution"
        )
    scaled = values / float(temperature)
    shifted = np.exp(scaled - scaled.max())
    probabilities = shifted / shifted.sum()
    # Nucleus: descending probability, ties to the lowest normalized id
    # (np.lexsort's last key is primary).
    order = np.lexsort((legal_model, -probabilities))
    cumulative = np.cumsum(probabilities[order])
    keep = int(np.searchsorted(cumulative, float(top_p))) + 1
    keep = min(keep, len(order))
    kept = order[:keep]
    kept_probabilities = probabilities[kept]
    kept_probabilities = kept_probabilities / kept_probabilities.sum()
    draw = float(rng.random())
    index = int(np.searchsorted(np.cumsum(kept_probabilities), draw, side="right"))
    index = min(index, keep - 1)
    return int(legal_model[kept[index]])


class Phase16StochasticEngine(Phase12SearchEngine):
    """The accepted engine with sampled rollouts as its only new behaviour."""

    def __init__(
        self,
        model,
        provider,
        config,
        *,
        rollout_temperature: float,
        rollout_top_p: float = ROLLOUT_TOP_P,
        device: "torch.device | str" = "cpu",
        dtype: torch.dtype = torch.float32,
        model_identity: "dict | None" = None,
    ) -> None:
        super().__init__(
            model,
            provider,
            config,
            device=device,
            dtype=dtype,
            model_identity=model_identity,
        )
        temperature = float(rollout_temperature)
        top_p = float(rollout_top_p)
        if not temperature >= 0.0:
            raise Phase16StochasticError(
                f"rollout_temperature must be >= 0, got {rollout_temperature!r}"
            )
        if not 0.0 < top_p <= 1.0:
            raise Phase16StochasticError(
                f"rollout_top_p must be in (0, 1], got {rollout_top_p!r}"
            )
        self.rollout_temperature = temperature
        self.rollout_top_p = top_p

    # -- the decision --------------------------------------------------------

    def choose_action(
        self,
        state: GameState,
        *,
        seed: int,
        deadline: "float | None" = None,
        rollout_seed: "int | None" = None,
    ) -> Phase12SearchDecision:
        """One search decision.

        ``rollout_temperature == 0`` delegates to the accepted engine and is
        bit-identical to it (the extra `rollout_seed`, if any, is unused —
        greedy rollouts consume no randomness). Otherwise the transcribed
        body below runs with one `numpy` generator seeded by `rollout_seed`
        (default: derived from `seed` under the Phase 16 rollout stream).
        """
        if self.rollout_temperature == 0.0:
            return super().choose_action(state, seed=seed, deadline=deadline)
        if rollout_seed is None:
            rollout_seed = derive_stochastic_seed(
                DOMAIN_ROLLOUT_SAMPLE, "engine_default", int(seed)
            )
        rng = np.random.Generator(np.random.PCG64(int(rollout_seed)))

        # ---- from here the body is the accepted Phase 12 `choose_action`,
        # transcribed verbatim except where marked `PHASE16`. -----------------
        started = time.perf_counter()
        if state.terminal:
            raise Phase12SearchError("a search decision was requested for a terminal state")
        config = self.config
        root = state.acting_player
        observation_seconds = 0.0
        forward_seconds = 0.0
        forward_batches: list[int] = []

        def check_deadline() -> None:
            if deadline is not None and time.perf_counter() > deadline:
                raise Phase12SearchTimeout(
                    f"search ran past its deadline at ply {state.total_moves} "
                    f"({time.perf_counter() - started:.3f}s elapsed)"
                )

        def timed_observation(target: GameState, player: int) -> np.ndarray:
            nonlocal observation_seconds
            tick = time.perf_counter()
            built = build_observation(target, player)
            observation_seconds += time.perf_counter() - tick
            return built

        # ---- root products -------------------------------------------------
        legal_abs = legal_actions(state)
        if not legal_abs:
            raise Phase12SearchError("a non-terminal state presented no legal actions")
        mask = legal_action_mask(state, legal_abs)
        observation = timed_observation(state, root)
        legality = prepare_legality(legal_abs, mask, root)

        outputs, elapsed = self._forward([observation])
        forward_seconds += elapsed
        forward_batches.append(1)
        check_deadline()
        policy_row = outputs.policy_logits.detach().to("cpu", torch.float32)[0]
        root_direct_value = float(
            expected_value(outputs.value_logits).detach().to("cpu", torch.float32)[0]
        )

        # The direct Phase 9 action, through the accepted adapter rule.
        direct = select_action(
            policy_row, legality, decision_mode=DECISION_MODE_GREEDY
        )

        # ---- priors and candidates -----------------------------------------
        legal_model = np.sort(np.asarray(legality.model, dtype=np.int64))
        row_np = policy_row.numpy()
        legal_logits = row_np[legal_model].astype(np.float64)
        shifted = np.exp(legal_logits - legal_logits.max())
        priors = shifted / shifted.sum()

        order = np.lexsort((legal_model, -legal_logits))
        candidate_count = min(config.max_root_candidates, len(legal_model))
        chosen_positions = order[:candidate_count]
        if int(legal_model[chosen_positions[0]]) != direct.model_action_id:
            raise Phase12SearchError(
                "the top-ranked candidate does not equal the direct Phase 9 "
                "action; candidate selection is broken"
            )

        candidate_model_ids = [int(legal_model[p]) for p in chosen_positions]
        candidate_abs_ids = [
            model_action_to_absolute(model_id, root) for model_id in candidate_model_ids
        ]
        legal_abs_set = set(legal_abs)
        if any(action not in legal_abs_set for action in candidate_abs_ids):
            raise Phase12SearchError("a candidate converted to an illegal engine action")
        candidate_priors = [float(priors[p]) for p in chosen_positions]

        # ---- worlds ---------------------------------------------------------
        view = build_public_view(state, root)
        document = build_public_state_document(view, observation)
        public = Phase11BPublicState(document, observation)

        if self.provider.uses_hidden_truth:
            if config.production:  # pragma: no cover - refused at construction
                raise Phase12SearchError("oracle sampling attempted in production")
            assignments = self.provider.sample_assignments_privileged(
                state, public, config.worlds, seed
            )
        else:
            assignments = self.provider.sample_assignments(public, config.worlds, seed)
        if len(assignments) != config.worlds:
            raise Phase12SearchError(
                f"the provider returned {len(assignments)} worlds, expected "
                f"{config.worlds}"
            )
        check_deadline()

        if config.deduplicate_worlds:
            buckets: dict[tuple, list] = {}
            for assignment in assignments:
                key = tuple(sorted(assignment.items()))
                entry = buckets.get(key)
                if entry is None:
                    buckets[key] = [assignment, 1]
                else:
                    entry[1] += 1
            unique_assignments = [entry[0] for entry in buckets.values()]
            weights = np.array(
                [entry[1] for entry in buckets.values()], dtype=np.float64
            )
        else:
            unique_assignments = list(assignments)
            weights = np.ones(len(assignments), dtype=np.float64)

        base_snapshot = create_snapshot(state, include_history=False)
        world_snapshots = []
        for assignment in unique_assignments:
            check_deadline()
            world = restore_snapshot(base_snapshot)
            apply_assignment_in_place(world, root, assignment)
            if config.verify_world_public_surface:
                world_observation = timed_observation(world, root)
                if not np.array_equal(world_observation, observation):
                    raise Phase12SearchError(
                        "a materialized world changed the root player's "
                        "observation; the hidden-identity invariance was violated"
                    )
                if generate_actions_for_player(world, root) != legal_abs:
                    raise Phase12SearchError(
                        "a materialized world changed the root player's legal "
                        "actions; the hidden-identity invariance was violated"
                    )
            world_snapshots.append(create_snapshot(world, include_history=False))

        # ---- candidate application ------------------------------------------
        unique_count = len(world_snapshots)
        results = np.zeros((candidate_count, unique_count), dtype=np.float64)
        terminal_leaves = 0
        value_leaves = 0
        rollout_plies_total = 0
        active: list[_Simulation] = []
        for candidate_index, absolute_action in enumerate(candidate_abs_ids):
            for world_index, world_snapshot in enumerate(world_snapshots):
                sim_state = restore_snapshot(world_snapshot)
                apply_action(sim_state, absolute_action, legal=legal_abs)
                if sim_state.terminal:
                    results[candidate_index, world_index] = sim_state.result_for(root)
                    terminal_leaves += 1
                else:
                    active.append(
                        _Simulation(candidate_index, world_index, sim_state)
                    )

        # ---- lockstep rollouts ----------------------------------------------
        rollout_iterations = 0
        while active:
            check_deadline()
            rollout_iterations += 1
            batch = np.stack(
                [
                    timed_observation(sim.state, sim.state.acting_player)
                    for sim in active
                ]
            )
            outputs, elapsed = self._forward(batch)
            forward_seconds += elapsed
            forward_batches.append(len(active))
            policy_np = (
                outputs.policy_logits.detach().to("cpu", torch.float32).numpy()
            )
            values_np = (
                expected_value(outputs.value_logits)
                .detach()
                .to("cpu", torch.float32)
                .numpy()
            )

            still_active: list[_Simulation] = []
            for row, sim in enumerate(active):
                actor = sim.state.acting_player
                if sim.plies >= config.rollout_depth:
                    leaf = float(values_np[row])
                    results[sim.candidate_index, sim.world_index] = (
                        leaf if actor == root else -leaf
                    )
                    value_leaves += 1
                    continue
                sim_legal = legal_actions(sim.state)
                sim_model = np.sort(
                    np.asarray(
                        absolute_legal_actions_to_model(sim_legal, actor),
                        dtype=np.int64,
                    )
                )
                # PHASE16: the one behavioural change — the rollout action is
                # drawn from the move model's distribution instead of argmax.
                chosen_model = _sampled_model_action(
                    rng,
                    policy_np[row],
                    sim_model,
                    self.rollout_temperature,
                    self.rollout_top_p,
                )
                chosen_abs = model_action_to_absolute(chosen_model, actor)
                apply_action(sim.state, chosen_abs, legal=sim_legal)
                sim.plies += 1
                rollout_plies_total += 1
                if sim.state.terminal:
                    results[sim.candidate_index, sim.world_index] = (
                        sim.state.result_for(root)
                    )
                    terminal_leaves += 1
                else:
                    still_active.append(sim)
            active = still_active

        # ---- scores -----------------------------------------------------------
        total_weight = float(weights.sum())
        if total_weight != float(config.worlds):  # pragma: no cover - defensive
            raise Phase12SearchError("world weights do not sum to the world count")
        q_values = results @ weights / total_weight

        candidates = []
        best_score = None
        best_model_id = None
        for index in range(candidate_count):
            prior = candidate_priors[index]
            log_prior_term = config.beta * float(np.log(prior + config.epsilon))
            score = float(q_values[index]) + log_prior_term
            candidates.append(
                Phase12CandidateResult(
                    absolute_action_id=int(candidate_abs_ids[index]),
                    model_action_id=int(candidate_model_ids[index]),
                    prior=prior,
                    q_value=float(q_values[index]),
                    log_prior_term=log_prior_term,
                    score=score,
                    is_direct=(
                        int(candidate_abs_ids[index]) == direct.absolute_action_id
                    ),
                    world_values=tuple(float(v) for v in results[index]),
                )
            )
            better = best_score is None or score > best_score
            same = best_score is not None and score == best_score
            if better or (same and candidate_model_ids[index] < best_model_id):
                best_score = score
                best_model_id = candidate_model_ids[index]

        selected_abs = model_action_to_absolute(int(best_model_id), root)
        if selected_abs not in legal_abs_set:  # pragma: no cover - bijection
            raise Phase12SearchError("the selected action is not legal")

        return Phase12SearchDecision(
            # PHASE16: an honest identity — these rollouts were sampled.
            search_version=ROLLOUT_SEARCH_VERSION,
            provider_id=self.provider.provider_id,
            preset_id=config.preset_id,
            seed=int(seed),
            game_id=state.game_id,
            root_ply=int(state.total_moves),
            acting_player=int(root),
            selected_action_id=int(selected_abs),
            direct_action_id=int(direct.absolute_action_id),
            move_changed=int(selected_abs) != int(direct.absolute_action_id),
            root_direct_value=root_direct_value,
            candidates=tuple(candidates),
            worlds_requested=int(config.worlds),
            unique_worlds=int(unique_count),
            world_weights=tuple(int(w) for w in weights),
            legal_action_count=len(legal_abs),
            c1_forwards=int(sum(forward_batches)),
            forward_batch_sizes=tuple(forward_batches),
            rollout_iterations=int(rollout_iterations),
            rollout_plies_total=int(rollout_plies_total),
            terminal_leaves=int(terminal_leaves),
            value_leaves=int(value_leaves),
            seconds=time.perf_counter() - started,
            forward_seconds=forward_seconds,
            observation_seconds=observation_seconds,
        )

    # -- description ----------------------------------------------------------

    def describe(self) -> dict:
        report = super().describe()
        if self.rollout_temperature > 0.0:
            report["search_version"] = ROLLOUT_SEARCH_VERSION
            report["rollout_policy"] = (
                "sampled from the move model's legal distribution at "
                f"tau_r={self.rollout_temperature}, top_p={self.rollout_top_p} "
                "(both sides); tau_r=0 is the accepted greedy rollout"
            )
        report["rollout_temperature"] = self.rollout_temperature
        report["rollout_top_p"] = self.rollout_top_p
        return report


__all__ = [
    "Phase16StochasticEngine",
    "_sampled_model_action",
]
