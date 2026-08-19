"""Phase 11 Agent 2: the two frozen nonlearned belief baselines.

Specification sources:

- `02_AGENT_2_BELIEF_EVALUATOR_BASELINES_VALIDATION.md` sections 3 and 4
- Agent 1's `phase11_belief_baseline_v1` and `phase11_belief_sampler_v1`

`remaining_count_belief_v1`
---------------------------
The primary predictive baseline, and the denominator of `R_CE`. For one
hidden piece:

```text
c[r] = initial_counts[r] - known[r]           public remaining inventory
q[r] = c[r] * mask[r] / sum_r' c[r'] * mask[r']
```

`known[r]` counts opponent pieces of rank `r` the observer legally knows,
**alive or captured** — every captured piece was revealed by the combat that
killed it, so counting the dead is a deduction the observer is entitled to.
That is also why `sum_r c[r]` equals the number of unresolved opponent
pieces exactly, which this module checks as a correctness invariant rather
than assuming.

The denominator is never zero: the piece's own true rank has a positive
remaining count and passes its own mask, so `q[true_rank] > 0` always and
the baseline's CE is always finite. Nothing here reads truth — that
statement is a consequence of the input type, which is the public document.

`count_uniform_world_sampler_v1`
--------------------------------
The structural fallback: the frozen sampler skeleton — same piece order,
same categorical walk, same completion-feasibility rule, same validation
stack — with step 7's weight reduced to `remaining_count` alone. It is a
*search fallback* baseline and feeds no predictive gate.

Agent 2 deliberately stops here. The learned `belief_sampler_v1` is
Agent 3's, and this module contains no learned weighting anywhere.
"""

from __future__ import annotations

import numpy as np

from ..training.phase11_contract import (
    IMMOVABLE_RANK_INDICES,
    MOVABLE_RANK_INDICES,
    Phase11ContractError,
    RANK_COUNT,
    RANK_INITIAL_COUNTS,
    RANK_NAMES,
    WORLD_BASELINE_VERSION,
)
from ..training.phase11_seed import (
    phase11_sample_token,
    world_categorical_uniform,
    world_order_key,
)
from .phase11_public_state import (
    hidden_opponent_pieces,
    legal_rank_mask,
    public_state_identity,
)

#: The frozen predictive-baseline identity.
REMAINING_COUNT_BASELINE_VERSION = "remaining_count_belief_v1"

#: The frozen structural world-baseline identity.
COUNT_UNIFORM_WORLD_SAMPLER_VERSION = WORLD_BASELINE_VERSION

_IMMOVABLE = frozenset(IMMOVABLE_RANK_INDICES)


class Phase11BaselineError(Phase11ContractError):
    """A baseline could not be computed, or violated one of its invariants."""


# ---------------------------------------------------------------------------
# Public inventory and masks
# ---------------------------------------------------------------------------


def remaining_counts(document: dict) -> tuple[int, ...]:
    """`c[r]` — the publicly inferable remaining opponent inventory.

    Counts every opponent piece whose exact rank the observer legally
    knows, alive or captured, and subtracts it from the initial army.
    """
    observer = document["observer_color"]
    known = [0] * RANK_COUNT
    for piece in document["pieces"]:
        if piece["owner_color"] == observer:
            continue
        if not piece["known_to_observer"]:
            continue
        rank = piece["known_rank_index"]
        if rank is None or not 0 <= int(rank) < RANK_COUNT:
            raise Phase11BaselineError(
                f"a known opponent piece carries rank index {rank!r}"
            )
        known[int(rank)] += 1
    counts = tuple(
        int(RANK_INITIAL_COUNTS[rank] - known[rank]) for rank in range(RANK_COUNT)
    )
    if any(count < 0 for count in counts):
        raise Phase11BaselineError(
            f"the public inventory went negative: {dict(zip(RANK_NAMES, counts))}"
        )
    return counts


def check_count_conservation(document: dict, counts=None) -> dict:
    """`sum_r c[r]` must equal the number of unresolved opponent pieces."""
    counts = remaining_counts(document) if counts is None else counts
    hidden = hidden_opponent_pieces(document)
    total = int(sum(counts))
    return {
        "remaining_total": total,
        "unresolved_pieces": len(hidden),
        "conserved": total == len(hidden),
    }


def piece_masks(document: dict) -> "dict[int, tuple[int, ...]]":
    """The public legal-rank mask of every hidden opponent piece."""
    return {
        int(piece["piece_slot"]): legal_rank_mask(bool(piece["has_moved"]))
        for piece in hidden_opponent_pieces(document)
    }


# ---------------------------------------------------------------------------
# remaining_count_belief_v1
# ---------------------------------------------------------------------------


def remaining_count_distribution(counts, mask) -> np.ndarray:
    """`q[r] = c[r] * mask[r] / sum(c * mask)`, float64."""
    weights = np.asarray(counts, dtype=np.float64) * np.asarray(mask, dtype=np.float64)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise Phase11BaselineError(
            "the remaining-count baseline has no legal mass; the public state "
            f"is inconsistent (counts={list(counts)}, mask={list(mask)})"
        )
    return weights / total


def remaining_count_belief(document: dict) -> "dict[int, np.ndarray]":
    """`remaining_count_belief_v1` for every hidden piece of one document.

    Keyed by public setup slot, which is the observer's public tracker for
    a concealed piece.
    """
    counts = remaining_counts(document)
    conservation = check_count_conservation(document, counts)
    if not conservation["conserved"]:
        raise Phase11BaselineError(
            "count conservation failed: "
            f"{conservation['remaining_total']} remaining vs "
            f"{conservation['unresolved_pieces']} unresolved pieces"
        )
    return {
        int(piece["piece_slot"]): remaining_count_distribution(
            counts, legal_rank_mask(bool(piece["has_moved"]))
        )
        for piece in hidden_opponent_pieces(document)
    }


# ---------------------------------------------------------------------------
# The frozen completion-feasibility rule (shared skeleton)
# ---------------------------------------------------------------------------


def feasible_ranks(
    legal: "tuple[int, ...]",
    counts: "list[int]",
    *,
    piece_has_moved: bool,
    moved_unresolved_remaining: int,
) -> "tuple[int, ...]":
    """Step 6's legal set, under Agent 1's frozen feasibility guard.

    An *unmoved* piece may take a movable rank only when taking it still
    leaves one movable rank for each publicly-moved piece that has not been
    assigned yet. A moved piece needs no guard beyond the public mask: its
    mask already excludes the immovable ranks.
    """
    available = tuple(rank for rank in legal if counts[rank] > 0)
    if piece_has_moved:
        return available
    movable_remaining = sum(counts[rank] for rank in MOVABLE_RANK_INDICES)
    kept = []
    for rank in available:
        if rank in _IMMOVABLE:
            kept.append(rank)
        elif movable_remaining - 1 >= moved_unresolved_remaining:
            kept.append(rank)
    return tuple(kept)


def inverse_cdf_choice(ranks, weights, uniform: float) -> int:
    """Step 8's walk: float64 cumulative mass, last legal rank as tail guard."""
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise Phase11BaselineError("inverse-CDF walk over zero total mass")
    cumulative = 0.0
    target = float(uniform) * total
    for rank, weight in zip(ranks, weights):
        cumulative += float(weight)
        if target < cumulative:
            return int(rank)
    return int(ranks[-1])


def sample_world(
    document: dict,
    sample_ordinal: int,
    *,
    sampler_version: str = COUNT_UNIFORM_WORLD_SAMPLER_VERSION,
    learned_probabilities: "dict[int, np.ndarray] | None" = None,
) -> dict:
    """One complete legal hidden world under the count-only baseline.

    `learned_probabilities` exists only so a caller can prove the skeleton
    is shared: the *baseline* passes `None`, which is the frozen
    "weight = remaining_count alone". Agent 2 never passes anything else,
    and the learned sampler is Agent 3's to build.
    """
    if learned_probabilities is not None and (
        sampler_version == COUNT_UNIFORM_WORLD_SAMPLER_VERSION
    ):
        raise Phase11BaselineError(
            "count_uniform_world_sampler_v1 takes no learned weighting"
        )
    identity = public_state_identity(document)
    token = phase11_sample_token(sampler_version, identity, int(sample_ordinal))
    counts = list(remaining_counts(document))
    hidden = hidden_opponent_pieces(document)

    order = sorted(
        hidden,
        key=lambda piece: (
            world_order_key(token, int(piece["piece_slot"])),
            int(piece["piece_slot"]),
        ),
    )
    moved_unresolved_remaining = sum(1 for piece in order if piece["has_moved"])

    assignment: dict[int, int] = {}
    fallback_steps: list[int] = []
    dead_end_events = 0
    for step_index, piece in enumerate(order):
        slot = int(piece["piece_slot"])
        has_moved = bool(piece["has_moved"])
        if has_moved:
            moved_unresolved_remaining -= 1
        mask = legal_rank_mask(has_moved)
        legal = tuple(rank for rank in range(RANK_COUNT) if mask[rank])
        ranks = feasible_ranks(
            legal,
            counts,
            piece_has_moved=has_moved,
            moved_unresolved_remaining=moved_unresolved_remaining,
        )
        if not ranks:
            dead_end_events += 1
            raise Phase11BaselineError(
                f"the sampler dead-ended at step {step_index} on slot {slot}; "
                "the feasibility rule proves this cannot happen on a publicly "
                "consistent state"
            )
        weights = np.array([float(counts[rank]) for rank in ranks], dtype=np.float64)
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise Phase11BaselineError("a non-finite or negative weight row")
        if float(weights.sum()) <= 0.0:  # pragma: no cover - counts are positive here
            fallback_steps.append(step_index)
            weights = np.ones(len(ranks), dtype=np.float64)
        uniform = world_categorical_uniform(token, step_index)
        chosen = inverse_cdf_choice(ranks, weights, uniform)
        assignment[slot] = chosen
        counts[chosen] -= 1

    return {
        "sample_token": token,
        "sampler_version": sampler_version,
        "public_state_identity": identity,
        "belief_model_label": "selfplay_c1_v1",
        "sample_ordinal": int(sample_ordinal),
        "piece_order": [int(piece["piece_slot"]) for piece in order],
        "fallback_steps": fallback_steps,
        "assignment": {int(slot): int(rank) for slot, rank in sorted(assignment.items())},
        "dead_end_events": dead_end_events,
    }


# ---------------------------------------------------------------------------
# The frozen validation stack
# ---------------------------------------------------------------------------

#: The zero-tolerance counters a world is checked against.
WORLD_COUNTER_NAMES = (
    "inventory_errors",
    "public_knowledge_violations",
    "known_rank_violations",
    "immobility_violations",
    "impossible_assignments",
    "nonfinite_probability_rows",
    "provenance_mismatches",
    "hidden_input_accesses",
    "dead_end_events",
)


def validate_world(document: dict, world: dict) -> dict:
    """Check one sampled world against the frozen validation stack."""
    counters = {name: 0 for name in WORLD_COUNTER_NAMES}
    findings: list[str] = []
    assignment = {int(slot): int(rank) for slot, rank in world["assignment"].items()}
    hidden = {int(piece["piece_slot"]): piece for piece in hidden_opponent_pieces(document)}
    observer = document["observer_color"]

    if set(assignment) != set(hidden):
        counters["impossible_assignments"] += 1
        findings.append("the world does not assign exactly the unresolved pieces")

    # Setup slots repeat across owners, so the assignment key space is the
    # *opponent's* slots alone: an own piece is never addressable here, and
    # checking one against the assignment would compare two different pieces.
    own_slots = {
        int(piece["piece_slot"])
        for piece in document["pieces"]
        if piece["owner_color"] == observer
    }
    if not own_slots >= set():  # pragma: no cover - defensive
        raise Phase11BaselineError("the observer owns no pieces")
    for piece in document["pieces"]:
        if piece["owner_color"] == observer:
            continue
        slot = int(piece["piece_slot"])
        if piece["known_to_observer"] and slot in assignment:
            counters["known_rank_violations"] += 1
            findings.append(f"slot {slot} has a known rank but was assigned")
        if not piece["alive"] and slot in assignment:
            counters["impossible_assignments"] += 1
            findings.append(f"slot {slot} is captured but was assigned")
    stray = set(assignment) - {
        int(piece["piece_slot"])
        for piece in document["pieces"]
        if piece["owner_color"] != observer
    }
    if stray:
        counters["public_knowledge_violations"] += len(stray)
        findings.append(f"the world assigned non-opponent slots {sorted(stray)}")

    observed = [0] * RANK_COUNT
    for slot, rank in assignment.items():
        if not 0 <= rank < RANK_COUNT:
            counters["impossible_assignments"] += 1
            findings.append(f"slot {slot} took rank index {rank}")
            continue
        observed[rank] += 1
        piece = hidden.get(slot)
        if piece is None:
            continue
        if piece["has_moved"] and rank in _IMMOVABLE:
            counters["immobility_violations"] += 1
            findings.append(f"moved slot {slot} was assigned {RANK_NAMES[rank]}")

    expected = remaining_counts(document)
    if tuple(observed) != tuple(expected):
        counters["inventory_errors"] += 1
        findings.append(
            f"inventory mismatch: assigned {observed} vs remaining {list(expected)}"
        )

    if world.get("public_state_identity") != public_state_identity(document):
        counters["provenance_mismatches"] += 1
        findings.append("the world's public-state identity does not match")
    expected_token = phase11_sample_token(
        world["sampler_version"],
        world["public_state_identity"],
        int(world["sample_ordinal"]),
    )
    if world.get("sample_token") != expected_token:
        counters["provenance_mismatches"] += 1
        findings.append("the world's sample token does not re-derive")
    counters["dead_end_events"] += int(world.get("dead_end_events", 0))

    return {
        "counters": counters,
        "findings": findings,
        "valid": not findings and all(value == 0 for value in counters.values()),
    }


__all__ = [
    "COUNT_UNIFORM_WORLD_SAMPLER_VERSION",
    "Phase11BaselineError",
    "REMAINING_COUNT_BASELINE_VERSION",
    "WORLD_COUNTER_NAMES",
    "check_count_conservation",
    "feasible_ranks",
    "inverse_cdf_choice",
    "piece_masks",
    "remaining_count_belief",
    "remaining_count_distribution",
    "remaining_counts",
    "sample_world",
    "validate_world",
]
