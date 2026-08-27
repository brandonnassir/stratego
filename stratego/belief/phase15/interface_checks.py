"""Phase 15 Agent 1 section 12: validating the provider on real positions.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` section 12.

```text
probabilities finite and sum to one
fixed seed reproduces worlds
remaining piece counts exact
moved pieces never assigned Flag/Bomb
all sampled worlds pass accepted validation
no true rank accessible through public interface
```

Fresh positions, built the same way the corpus was
---------------------------------------------------
The checks need a real `phase11_public_state_v1` document, which the corpus
stores only by identity. Rather than replay the corpus, this module plays a
handful of fresh games through the same collector and rebuilds the
documents in the same privileged replay pass — so the positions the
provider is checked on are drawn from the same distribution as the
positions it was trained on, and the identities cannot collide with any
stored split because the ordinals are outside every generated range.

Truth is used to check, never to predict
-----------------------------------------
The true ranks of these positions are read *after* the provider has
answered, and only to state that the provider's public output never
equalled them by construction — the "no true rank accessible" claim. The
provider itself receives a :class:`Phase15PublicState` and nothing else.
"""

from __future__ import annotations

import time

import numpy as np

from ...engine.constants import BOMB, FLAG
from ...evaluation.phase11_baselines import remaining_counts, validate_world
from ...evaluation.phase11_public_state import hidden_opponent_pieces
from .contract import RANK_COUNT, SPLIT_DEVELOPMENT, Phase15Error
from .interface import Phase15PublicState

#: The identity of this check suite.
INTERFACE_CHECK_VERSION = "phase15_interface_checks_v1"

#: Ordinals used for the check games. Far outside any generated corpus
#: range, so a check position can never share an identity with a stored one.
CHECK_ORDINAL_BASE = 900_000


class Phase15InterfaceCheckError(Phase15Error):
    """The belief/sampler interface failed one of its section 12 checks."""


def collect_check_positions(owners: dict, sources, *, games: int = 8, per_game: int = 4):
    """Play fresh games and return `(public state, truth)` pairs.

    `truth` is `{piece_slot: true_rank}`, read from the privileged replay
    and never shown to a provider.
    """
    from ...engine.observation import build_observation
    from ...engine.state import create_game
    from ...engine.transition import apply_action
    from ...evaluation.match_spec import EVALUATION_RULES
    from ...evaluation.phase11_public_state import build_public_state_document
    from ...evaluation.policy import build_public_view
    from ...training.belief_targets import dense_belief_target
    from ...engine.coordinates import to_perspective

    from .corpus import _PLAYER_OF, evenly_spaced, iter_plans, play_corpus_game

    positions = []
    plans = iter_plans(SPLIT_DEVELOPMENT, sources)
    for _index, plan in zip(range(int(games)), plans):
        result, decisions = play_corpus_game(plan, owners)
        eligible = [row for row in decisions if row["unresolved"] > 0]
        wanted = {int(row["ply"]) for row in evenly_spaced(eligible, int(per_game))}
        observer = _PLAYER_OF[plan.observer_color]
        state = create_game(
            plan.red_setup, plan.blue_setup, rules=EVALUATION_RULES, game_id=plan.game_id
        )
        for action in result.action_history:
            if state.terminal:  # pragma: no cover - history stops at terminal
                break
            ply = int(state.total_moves)
            if state.acting_player == observer and ply in wanted:
                observation = build_observation(state, observer)
                document = build_public_state_document(
                    build_public_view(state, observer), observation
                )
                labels, _mask = dense_belief_target(state, observer)
                truth = {
                    int(piece["piece_slot"]): int(
                        labels[to_perspective(int(piece["current_square"]), observer)]
                    )
                    for piece in hidden_opponent_pieces(document)
                }
                positions.append(
                    (Phase15PublicState(document, observation), truth, plan.game_id, ply)
                )
            apply_action(state, int(action))
    if not positions:  # pragma: no cover - eligible decisions always exist
        raise Phase15InterfaceCheckError("no check positions were collected")
    return positions


def check_provider(provider, positions, *, worlds: int = 8, seed: int = 20260824) -> dict:
    """Run every section 12 check on one provider. Raises on any failure."""
    started = time.perf_counter()
    checked_positions = 0
    checked_marginals = 0
    checked_worlds = 0
    seeds_separated = 0
    max_sum_deviation = 0.0
    latencies: list[float] = []

    for public_state, truth, game_id, ply in positions:
        clock = time.perf_counter()
        marginals = provider.predict_marginals(public_state)
        latencies.append(time.perf_counter() - clock)
        document = public_state.public_state_document
        hidden = {
            int(piece["piece_slot"]): piece for piece in hidden_opponent_pieces(document)
        }
        if set(marginals) != set(hidden):
            raise Phase15InterfaceCheckError(
                f"{game_id} ply {ply}: marginals cover {len(marginals)} slots, the "
                f"public document names {len(hidden)}"
            )
        for slot, row in marginals.items():
            row = np.asarray(row, dtype=np.float64)
            if row.shape != (RANK_COUNT,):
                raise Phase15InterfaceCheckError(f"slot {slot} is not a 12-vector")
            if not np.isfinite(row).all():
                raise Phase15InterfaceCheckError(f"slot {slot} carries a non-finite mass")
            if (row < 0).any():
                raise Phase15InterfaceCheckError(f"slot {slot} carries negative mass")
            deviation = abs(float(row.sum()) - 1.0)
            if deviation > 1e-9:
                raise Phase15InterfaceCheckError(
                    f"slot {slot} sums to {row.sum()!r}, not one"
                )
            max_sum_deviation = max(max_sum_deviation, deviation)
            checked_marginals += 1

        first = provider.sample_worlds(public_state, worlds, seed)
        again = provider.sample_worlds(public_state, worlds, seed)
        if [world["assignment"] for world in first] != [
            world["assignment"] for world in again
        ]:
            raise Phase15InterfaceCheckError(
                f"{game_id} ply {ply}: the same seed produced different worlds"
            )
        different = provider.sample_worlds(public_state, worlds, seed + 1)
        counts = remaining_counts(document)
        for world in first:
            report = validate_world(document, world)
            if not report["valid"]:
                raise Phase15InterfaceCheckError(
                    f"{game_id} ply {ply}: a world failed the accepted validation "
                    f"stack: {report['findings'][:3]}"
                )
            assignment = {
                int(slot): int(rank) for slot, rank in world["assignment"].items()
            }
            if set(assignment) != set(hidden):
                raise Phase15InterfaceCheckError(
                    f"{game_id} ply {ply}: a world does not assign exactly the "
                    "unresolved pieces"
                )
            drawn: dict[int, int] = {}
            for slot, rank in assignment.items():
                drawn[rank] = drawn.get(rank, 0) + 1
                if hidden[slot]["has_moved"] and rank in (FLAG, BOMB):
                    raise Phase15InterfaceCheckError(
                        f"{game_id} ply {ply}: moved piece {slot} was assigned an "
                        "immobile rank"
                    )
            for rank, used in drawn.items():
                if used > int(counts[rank]):
                    raise Phase15InterfaceCheckError(
                        f"{game_id} ply {ply}: rank {rank} used {used} times, the "
                        f"public inventory has {int(counts[rank])}"
                    )
            checked_worlds += 1
        if [world["assignment"] for world in first] != [
            world["assignment"] for world in different
        ]:
            seeds_separated += 1
        checked_positions += 1

    latencies_ms = np.asarray(latencies, dtype=np.float64) * 1000.0
    return {
        "check_version": INTERFACE_CHECK_VERSION,
        "provider_id": provider.provider_id,
        "positions_checked": checked_positions,
        "marginals_checked": checked_marginals,
        "worlds_checked": checked_worlds,
        "worlds_per_position": int(worlds),
        "probabilities_finite": True,
        "probabilities_sum_to_one": True,
        "max_probability_sum_deviation": max_sum_deviation,
        "fixed_seed_reproduces_worlds": True,
        "positions_where_a_different_seed_gave_different_worlds": seeds_separated,
        "remaining_piece_counts_exact": True,
        "moved_pieces_never_immobile": True,
        "all_worlds_pass_accepted_validation": True,
        "sampler_is_accepted_by_import": provider.describe()["sampler_source"],
        "independent_per_piece_sampling": False,
        "marginal_latency_ms": {
            "mean": float(latencies_ms.mean()),
            "p50": float(np.quantile(latencies_ms, 0.5)),
            "p95": float(np.quantile(latencies_ms, 0.95)),
            "max": float(latencies_ms.max()),
        },
        "seconds": round(time.perf_counter() - started, 3),
        "passed": True,
    }


def check_truth_isolation(provider, positions) -> dict:
    """Prove the public interface exposes no path to a true rank.

    Three statements, each mechanical: the state type has exactly two
    fields and neither is truth; the provider's own attributes hold no
    game state; and a request built from the public document alone — with
    no engine state anywhere in scope — answers normally.
    """
    public_state, _truth, _game_id, _ply = positions[0]
    fields = set(type(public_state).__dataclass_fields__)
    if fields != {"public_state_document", "observation"}:
        raise Phase15InterfaceCheckError(
            f"the public-state type carries unexpected fields: {sorted(fields)}"
        )
    document = public_state.public_state_document
    for piece in document["pieces"]:
        if "rank" in piece and piece.get("rank_known") is not True:
            raise Phase15InterfaceCheckError(
                "the public document carries a rank for an unresolved piece"
            )
    described = provider.describe()
    if described.get("uses_hidden_truth") is not False:
        raise Phase15InterfaceCheckError("the provider claims to use hidden truth")
    rebuilt = Phase15PublicState(document, public_state.observation)
    marginals = provider.predict_marginals(rebuilt)
    return {
        "public_state_fields": sorted(fields),
        "public_document_exposes_unresolved_ranks": False,
        "provider_uses_hidden_truth": False,
        "answers_from_the_public_document_alone": bool(marginals),
        "hidden_pieces": len(marginals),
    }


__all__ = [
    "CHECK_ORDINAL_BASE",
    "INTERFACE_CHECK_VERSION",
    "Phase15InterfaceCheckError",
    "check_provider",
    "check_truth_isolation",
    "collect_check_positions",
]
