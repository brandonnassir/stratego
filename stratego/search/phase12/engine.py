"""The Phase 12 minimal search engine: root-sampled worlds, greedy rollouts.

Specification sources:

- `00_PHASE_12_SEQUENCE_AND_COMMON_CONTRACT.md` sections 6-9
- `02_PHASE_12_AGENT_1_SEARCH_CORE.md` sections 4-9

The algorithm, exactly as instructed
------------------------------------
.. code-block:: text

    public state
      -> belief provider -> K complete hidden worlds (sampled once, at the root)
      -> Phase 9 root policy -> candidate root moves (all legal if <= 8, else top 8)
      -> evaluate every candidate on the same K worlds
      -> short Phase 9 greedy-policy rollouts (both sides)
      -> exact terminal result, else Phase 9 leaf value V = P(win) - P(loss)
      -> average across worlds -> Q(a)
      -> S(a) = Q(a) + beta * log(pi(a) + epsilon) -> chosen move

Worlds and the information boundary
-----------------------------------
A sampled world is the real root state with the hidden opponent ranks
replaced by one sampled assignment — exactly the transformation of the
accepted anti-leak permutation gate
(:mod:`stratego.engine.permutation`), which proves the observer's public
surface is invariant under it. The engine can therefore reuse the root's
legal-action list inside every world, and (while
`verify_world_public_surface` is on) it re-derives the world's observation
and legal actions and requires them identical to the real root's, turning
the gate's guarantee into a per-decision runtime check.

During rollouts each simulated player receives only
`build_observation(world, that_player)` — its legal view of the world. The
sampled truth reaches the real root player only as per-candidate value
aggregates; no assignment, rank or marginal appears in a decision.

Determinism
-----------
The engine consumes no randomness at all: candidate selection, rollouts and
the final argmax are all deterministic with defined tie-breaks (lowest
normalized action id, the accepted adapter convention). The only random
input is the provider `seed`, so a fixed seed reproduces the same decision
whenever the model's forward passes are themselves deterministic on the
chosen device.

Costs
-----
One C1 forward at the root, then one batched forward per rollout ply over
every live (candidate, world) simulation, plus one leaf-value forward per
simulation that reaches the depth limit. Duplicate sampled worlds are
evaluated once and weighted by multiplicity, which is exact here because
greedy rollouts are deterministic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from ...belief.phase11b.interface import Phase11BPublicState
from ...engine.constants import IMMOVABLE_TYPES, NUM_PIECE_TYPES
from ...engine.legal_moves import (
    generate_actions_for_player,
    legal_action_mask,
    legal_actions,
)
from ...engine.observation import build_observation
from ...engine.permutation import hidden_opponent_piece_ids
from ...engine.pieces import piece_setup_slot
from ...engine.snapshot import clone_state, create_snapshot, restore_snapshot
from ...engine.state import GameState
from ...engine.transition import apply_action
from ...evaluation.phase11_public_state import build_public_state_document
from ...evaluation.policy import build_public_view
from ...model.action_frame import (
    absolute_legal_actions_to_model,
    model_action_to_absolute,
)
from ...model.base import StrategoModel
from ...model.contract import expected_value
from ...model.policy_adapter import (
    DECISION_MODE_GREEDY,
    prepare_legality,
    select_action,
)
from ...model.tokenization import observation_batch_from_numpy, observation_to_tokens
from .contract import (
    Phase12SearchConfig,
    Phase12SearchError,
    Phase12SearchTimeout,
    SEARCH_VERSION,
)
from .providers import Phase12BeliefProvider


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------


def apply_assignment_in_place(
    state: GameState, observer: int, assignment: "dict[int, int]"
) -> None:
    """Overwrite the hidden opponent ranks of `state` with `assignment`.

    The in-place half of world materialization. `assignment` must cover
    exactly the live opponent pieces whose rank `observer` may not legally
    know, and must respect the public moved/immovable constraint — both are
    checked here even though every provider validates upstream, because a
    world that violates them would silently poison every rollout on it.
    """
    hidden_ids = hidden_opponent_piece_ids(state, observer)
    hidden_slots = {piece_setup_slot(piece_id) for piece_id in hidden_ids}
    if set(assignment) != hidden_slots:
        raise Phase12SearchError(
            "a world assignment does not cover exactly the unresolved opponent "
            f"pieces (assigned {sorted(assignment)}, unresolved {sorted(hidden_slots)})"
        )
    for piece_id in hidden_ids:
        record = state.pieces[piece_id]
        rank = int(assignment[piece_setup_slot(piece_id)])
        if not 0 <= rank < NUM_PIECE_TYPES:
            raise Phase12SearchError(f"assignment rank {rank} is not a piece type")
        if record.has_moved and rank in IMMOVABLE_TYPES:
            raise Phase12SearchError(
                f"assignment gives moved piece {record.piece_id} an immovable rank"
            )
        record.true_type = rank


def materialize_world(
    root_state: GameState, observer: int, assignment: "dict[int, int]"
) -> GameState:
    """An independent playable copy of `root_state` under one assignment."""
    world = clone_state(root_state)
    apply_assignment_in_place(world, observer, assignment)
    return world


# ---------------------------------------------------------------------------
# Decision records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase12CandidateResult:
    """One evaluated root candidate."""

    absolute_action_id: int
    model_action_id: int
    prior: float
    q_value: float
    log_prior_term: float
    score: float
    is_direct: bool
    #: Root-player value of this candidate in each unique world, aligned
    #: with the decision's `world_weights`.
    world_values: tuple


@dataclass(frozen=True)
class Phase12SearchDecision:
    """One search decision with enough diagnostics to study it."""

    search_version: str
    provider_id: str
    preset_id: str
    seed: int
    game_id: str
    root_ply: int
    acting_player: int
    selected_action_id: int
    direct_action_id: int
    move_changed: bool
    root_direct_value: float
    candidates: tuple
    worlds_requested: int
    unique_worlds: int
    world_weights: tuple
    legal_action_count: int
    c1_forwards: int
    forward_batch_sizes: tuple
    rollout_iterations: int
    rollout_plies_total: int
    terminal_leaves: int
    value_leaves: int
    seconds: float
    forward_seconds: float
    observation_seconds: float

    def summary(self) -> dict:
        return {
            "search_version": self.search_version,
            "provider_id": self.provider_id,
            "preset_id": self.preset_id,
            "seed": self.seed,
            "game_id": self.game_id,
            "root_ply": self.root_ply,
            "acting_player": self.acting_player,
            "selected_action_id": self.selected_action_id,
            "direct_action_id": self.direct_action_id,
            "move_changed": self.move_changed,
            "root_direct_value": self.root_direct_value,
            "worlds_requested": self.worlds_requested,
            "unique_worlds": self.unique_worlds,
            "legal_action_count": self.legal_action_count,
            "candidate_count": len(self.candidates),
            "c1_forwards": self.c1_forwards,
            "max_forward_batch": max(self.forward_batch_sizes),
            "rollout_iterations": self.rollout_iterations,
            "rollout_plies_total": self.rollout_plies_total,
            "terminal_leaves": self.terminal_leaves,
            "value_leaves": self.value_leaves,
            "seconds": self.seconds,
            "forward_seconds": self.forward_seconds,
            "observation_seconds": self.observation_seconds,
        }


class _Simulation:
    """One live (candidate, world) rollout."""

    __slots__ = ("candidate_index", "world_index", "state", "plies")

    def __init__(self, candidate_index: int, world_index: int, state: GameState):
        self.candidate_index = candidate_index
        self.world_index = world_index
        self.state = state
        self.plies = 0


def _greedy_model_action(row: np.ndarray, legal_model: np.ndarray) -> int:
    """Highest legal logit, ties to the lowest normalized action id.

    The NumPy fast path of the accepted
    :func:`stratego.model.policy_adapter.greedy_action` rule; a test pins
    the two to identical choices. `legal_model` must be ascending.
    """
    values = row[legal_model]
    if not np.isfinite(values).all():
        raise Phase12SearchError(
            "the model produced a non-finite logit on a legal action during a "
            "rollout; refusing to choose a move from an untrustworthy distribution"
        )
    best = values.max()
    return int(legal_model[int(np.argmax(values == best))])


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class Phase12SearchEngine:
    """Root-world search over one accepted Phase 9 move model.

    The model provides the root policy, the rollout policy for both sides
    and the leaf value; the provider supplies hidden worlds and nothing
    else. Construction refuses an oracle provider whenever the
    configuration says `production=True`.
    """

    def __init__(
        self,
        model: StrategoModel,
        provider: Phase12BeliefProvider,
        config: Phase12SearchConfig,
        *,
        device: "torch.device | str" = "cpu",
        dtype: torch.dtype = torch.float32,
        model_identity: "dict | None" = None,
    ) -> None:
        if not isinstance(model, StrategoModel):
            raise Phase12SearchError(
                f"expected a StrategoModel, got {type(model).__name__}"
            )
        if not isinstance(provider, Phase12BeliefProvider):
            raise Phase12SearchError(
                f"expected a Phase12BeliefProvider, got {type(provider).__name__}"
            )
        if not isinstance(config, Phase12SearchConfig):
            raise Phase12SearchError(
                f"expected a Phase12SearchConfig, got {type(config).__name__}"
            )
        if config.production and provider.uses_hidden_truth:
            raise Phase12SearchError(
                "a production search configuration structurally excludes the "
                "oracle provider; build the engine with production=False for "
                "offline diagnostics"
            )
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = model.to(device=self.device, dtype=dtype)
        self.model.eval()
        self.provider = provider
        self.config = config
        self.model_identity = dict(model_identity or {})

    # -- inference ----------------------------------------------------------

    def _forward(self, observations) -> tuple:
        """One batched forward. Returns `(outputs, seconds)`."""
        batch = observation_batch_from_numpy(
            observations, dtype=self.dtype, device=self.device
        )
        started = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(observation_to_tokens(batch))
        if self.device.type == "mps":
            torch.mps.synchronize()
        return outputs, time.perf_counter() - started

    # -- the decision --------------------------------------------------------

    def choose_action(
        self,
        state: GameState,
        *,
        seed: int,
        deadline: "float | None" = None,
    ) -> Phase12SearchDecision:
        """One search decision; see the module docstring for the algorithm.

        `deadline` is an optional absolute `time.perf_counter()` timestamp.
        When set, the engine checks it cooperatively at its loop boundaries
        (after the root forward, after world sampling, per materialized
        world, and before each batched rollout forward) and raises
        :class:`Phase12SearchTimeout` once it has passed. The checks are
        pure control flow: with the same seed, a decision that completes is
        bit-identical whether or not a deadline was supplied. The default
        `None` is exactly the Agent 1-4 behaviour.
        """
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

        # Descending logit, ties to the lowest normalized id (np.lexsort's
        # last key is primary). The first entry is the direct action by the
        # accepted greedy rule's own definition, asserted below.
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
                # The root's legal actions are world-invariant (verified
                # above when the check is on, and guaranteed by the accepted
                # permutation gate), so the root list revalidates the move.
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
                    # Depth reached: this forward is the leaf evaluation.
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
                chosen_model = _greedy_model_action(policy_np[row], sim_model)
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
            search_version=SEARCH_VERSION,
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
        return {
            "search_version": SEARCH_VERSION,
            "config": self.config.describe(),
            "provider": self.provider.describe(),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "model_architecture_id": self.model.architecture_id,
            "model_identity": dict(self.model_identity),
            "rollout_policy": "accepted Phase 9 greedy (both sides)",
            "leaf_value": "V = P(win) - P(loss), terminal results override",
        }


__all__ = [
    "Phase12CandidateResult",
    "Phase12SearchDecision",
    "Phase12SearchEngine",
    "apply_assignment_in_place",
    "materialize_world",
]
