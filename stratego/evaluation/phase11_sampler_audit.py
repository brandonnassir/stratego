"""Phase 11 Agent 3: the independent sampler-verification path.

Specification sources:

- `03_AGENT_3_CONSTRAINED_BELIEF_SAMPLER.md` ("Independent audit")

An audit that shares an implementation audits nothing
-----------------------------------------------------
This module rebuilds every constraint the sampler claims to satisfy from
first principles, and deliberately imports **none** of the Phase 11 modules
the primary path is made of — not `phase11_seed`, not `phase11_baselines`,
not `phase11_public_state`, not `phase11_sampler`, not `phase11_contract`.
Its authorities are:

- the accepted **engine constants** (`PIECE_TYPE_NAMES`, `PIECE_COUNTS`,
  `FLAG`, `BOMB`), written long before Phase 11 existed — the same
  authority Agent 1's contract names for the rank order and the army;
- the **frozen contract text** itself: the seed derivation
  (`blake2b(person='strat-b11', digest_size=8)` over
  `'phase11_identity_v1:domain:domain_root:part:...'`, big-endian, right
  shift one), the sample-token format, and the twelve algorithm steps are
  restated here as local constants and local arithmetic, so agreement with
  the primary path is evidence about the *frozen mathematics*, not about a
  shared helper.

Arithmetic is scalar Python (`math.fsum`, sequential cumulative sums) where
the primary path is NumPy. The two float paths agree exactly unless a
categorical draw lands within a few ulps of a bin boundary; such knife-edge
steps are counted and reported (`knife_edge_events`) so a disagreement can
never hide behind tolerance. On the frozen audit data the expected count is
zero, and any non-zero count is a recorded finding.

What `verify_world_independently` re-derives, per world:

1. the public-state identity (its own canonical JSON + SHA-256);
2. the sample token from `(sampler_version, identity, ordinal)`;
3. the remaining inventory from the raw document (its own piece walk);
4. the public legal-rank masks from the raw `has_moved` flags;
5. the unresolved-piece order from raw `blake2b` order keys;
6. the completion-feasibility guard, from public inputs only, with the
   invariant `movable_remaining >= moved_unresolved_remaining` asserted
   after every assignment;
7. the categorical draws from raw `blake2b` seeds and exact `seed / 2**63`
   uniforms, walked with its own scalar arithmetic;
8. the zero-mass fallback rule (counts over the same legal set);
9. the final world against every public fact: exact multiset, locked
   knowns, dead pieces excluded, ownership, immobility, provenance fields.
"""

from __future__ import annotations

import hashlib
import json
import math

from ..engine.constants import BOMB, FLAG, PIECE_COUNTS, PIECE_TYPE_NAMES

# ---------------------------------------------------------------------------
# Frozen constants, restated from the Agent 1 contract document
# ---------------------------------------------------------------------------

#: The Phase 11 seed personalization and identity version, from the frozen
#: derivation record (`phase11_belief_contract_v1.seeds.derivation`).
_PERSON = b"strat-b11"
_IDENTITY_VERSION = "phase11_identity_v1"

#: The frozen world-sampling root and master seed (common contract, "Root
#: seeds").
_WORLD_SAMPLING_ROOT = 2026081904
_MASTER_SEED = 2026081901

#: The frozen model label a sample token carries.
_MODEL_LABEL = "selfplay_c1_v1"
_SAMPLE_VERSION = "phase11_world_sample_v1"

#: The rank space, from the engine authority directly.
RANKS = len(PIECE_TYPE_NAMES)
INITIAL_COUNTS = tuple(int(PIECE_COUNTS[rank]) for rank in range(RANKS))
IMMOVABLE = frozenset({FLAG, BOMB})
MOVABLE = tuple(rank for rank in range(RANKS) if rank not in IMMOVABLE)

#: Two draws closer than this many ulps to a bin boundary are knife-edge
#: events: the only place scalar and NumPy float paths could ever disagree.
KNIFE_EDGE_ULPS = 8.0


class Phase11IndependentAuditError(RuntimeError):
    """The independent path could not audit a world."""


# ---------------------------------------------------------------------------
# Seed derivation, from raw blake2b
# ---------------------------------------------------------------------------


def independent_seed(domain: str, *parts) -> int:
    """The frozen 63-bit stream seed, from the published derivation alone."""
    for part in parts:
        if isinstance(part, str) and ":" in part:
            raise Phase11IndependentAuditError(
                f"string identity parts may not contain ':' (got {part!r})"
            )
    payload = ":".join(
        [_IDENTITY_VERSION, domain, str(_WORLD_SAMPLING_ROOT)]
        + [str(part) for part in parts]
    )
    digest = hashlib.blake2b(payload.encode(), digest_size=8, person=_PERSON).digest()
    return int.from_bytes(digest, "big") >> 1


def independent_unit_uniform(seed: int) -> float:
    """`seed / 2**63`, the exact binary division of the frozen contract."""
    return seed / float(1 << 63)


def independent_sample_token(
    sampler_version: str, public_state_identity: str, sample_ordinal: int
) -> str:
    """The frozen sample-token format, rebuilt from its published shape."""
    return (
        f"{_SAMPLE_VERSION}|ms={_MASTER_SEED}|model={_MODEL_LABEL}"
        f"|smp={sampler_version}|ps={public_state_identity}|n={sample_ordinal:05d}"
    )


# ---------------------------------------------------------------------------
# Public facts, from the raw document
# ---------------------------------------------------------------------------


def independent_identity(document: dict) -> str:
    """SHA-256 over the document's canonical JSON — its own dumps call."""
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def independent_hidden_pieces(document: dict) -> "list[dict]":
    """Live opponent pieces with no observer-known rank, in slot order."""
    observer = document["observer_color"]
    hidden = [
        piece
        for piece in document["pieces"]
        if piece["owner_color"] != observer
        and bool(piece["alive"])
        and not bool(piece["known_to_observer"])
    ]
    return sorted(hidden, key=lambda piece: int(piece["piece_slot"]))


def independent_inventory(document: dict) -> "list[int]":
    """`c[r] = initial[r] - known[r]`, from its own walk of the pieces."""
    observer = document["observer_color"]
    known = [0] * RANKS
    for piece in document["pieces"]:
        if piece["owner_color"] == observer:
            continue
        if not bool(piece["known_to_observer"]):
            continue
        rank = piece["known_rank_index"]
        if rank is None or not 0 <= int(rank) < RANKS:
            raise Phase11IndependentAuditError(
                f"a known opponent piece carries rank index {rank!r}"
            )
        known[int(rank)] += 1
    counts = [INITIAL_COUNTS[rank] - known[rank] for rank in range(RANKS)]
    if any(count < 0 for count in counts):
        raise Phase11IndependentAuditError(
            f"the public inventory went negative: {counts}"
        )
    if sum(counts) != len(independent_hidden_pieces(document)):
        raise Phase11IndependentAuditError(
            "count conservation failed on the raw document"
        )
    return counts


def independent_mask(has_moved: bool) -> "tuple[int, ...]":
    """Movement impossibility only: moved pieces exclude flag and bomb."""
    if has_moved:
        return tuple(0 if rank in IMMOVABLE else 1 for rank in range(RANKS))
    return (1,) * RANKS


def independent_softmax(logits) -> "list[float]":
    """Scalar max-shifted softmax, for the probability-row cross-check."""
    values = [float(value) for value in logits]
    top = max(values)
    shifted = [math.exp(value - top) for value in values]
    total = math.fsum(shifted)
    return [value / total for value in shifted]


# ---------------------------------------------------------------------------
# The independent walk
# ---------------------------------------------------------------------------


def independent_walk(
    document: dict,
    probability_rows: "dict[int, list[float]] | None",
    sampler_version: str,
    sample_ordinal: int,
) -> dict:
    """Re-run the twelve frozen steps with local arithmetic only.

    `probability_rows` maps piece slot to the frozen float64 learned
    12-vector; `None` selects the count-only weighting (the
    `count_uniform_world_sampler_v1` skeleton instantiation).
    """
    identity = independent_identity(document)
    token = independent_sample_token(sampler_version, identity, int(sample_ordinal))
    counts = independent_inventory(document)
    hidden = independent_hidden_pieces(document)

    order = sorted(
        hidden,
        key=lambda piece: (
            independent_seed("world_order", token, int(piece["piece_slot"])),
            int(piece["piece_slot"]),
        ),
    )
    moved_unresolved = sum(1 for piece in order if bool(piece["has_moved"]))

    assignment: dict[int, int] = {}
    fallback_steps: list[int] = []
    guard_pruned_steps = 0
    knife_edge_events = 0
    invariant_violations = 0

    for step_index, piece in enumerate(order):
        slot = int(piece["piece_slot"])
        has_moved = bool(piece["has_moved"])
        if has_moved:
            moved_unresolved -= 1
        mask = independent_mask(has_moved)
        available = [
            rank for rank in range(RANKS) if mask[rank] and counts[rank] > 0
        ]
        if has_moved:
            legal = available
        else:
            movable_remaining = sum(counts[rank] for rank in MOVABLE)
            legal = [
                rank
                for rank in available
                if rank in IMMOVABLE or movable_remaining - 1 >= moved_unresolved
            ]
            if len(legal) < len(available):
                guard_pruned_steps += 1
        if not legal:
            raise Phase11IndependentAuditError(
                f"the independent walk dead-ended at step {step_index}"
            )

        if probability_rows is None:
            weights = [float(counts[rank]) for rank in legal]
        else:
            row = probability_rows[slot]
            weights = [float(row[rank]) * float(counts[rank]) for rank in legal]
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise Phase11IndependentAuditError(
                f"a non-finite or negative weight at step {step_index}"
            )
        total = math.fsum(weights)
        if total <= 0.0:
            fallback_steps.append(step_index)
            weights = [float(counts[rank]) for rank in legal]
            total = math.fsum(weights)

        seed = independent_seed("world_categorical", token, int(step_index))
        uniform = independent_unit_uniform(seed)
        target = uniform * total
        chosen = int(legal[-1])
        cumulative = 0.0
        edge = KNIFE_EDGE_ULPS * math.ulp(total if total > 0.0 else 1.0)
        for rank, weight in zip(legal, weights):
            cumulative += float(weight)
            if abs(target - cumulative) < edge:
                knife_edge_events += 1
            if target < cumulative:
                chosen = int(rank)
                break
        assignment[slot] = chosen
        counts[chosen] -= 1

        movable_after = sum(counts[rank] for rank in MOVABLE)
        if movable_after < moved_unresolved:
            invariant_violations += 1

    return {
        "sample_token": token,
        "public_state_identity": identity,
        "piece_order": [int(piece["piece_slot"]) for piece in order],
        "assignment": dict(sorted(assignment.items())),
        "fallback_steps": fallback_steps,
        "guard_pruned_steps": guard_pruned_steps,
        "knife_edge_events": knife_edge_events,
        "invariant_violations": invariant_violations,
        "steps": len(order),
    }


# ---------------------------------------------------------------------------
# Verifying a primary world
# ---------------------------------------------------------------------------


def verify_world_independently(
    document: dict,
    probability_rows: "dict[int, list[float]] | None",
    world: dict,
    *,
    logits_rows: "dict[int, list[float]] | None" = None,
    softmax_tolerance: float = 1e-12,
) -> dict:
    """One primary world, fully re-derived and re-checked from scratch.

    Returns findings (empty means exact agreement plus every public fact
    holding) and the walk's report-only counters.
    """
    findings: list[str] = []

    if logits_rows is not None and probability_rows is not None:
        for slot, logits in logits_rows.items():
            local = independent_softmax(logits)
            row = probability_rows[int(slot)]
            deviation = max(
                abs(float(row[rank]) - local[rank]) for rank in range(RANKS)
            )
            if deviation > softmax_tolerance:
                findings.append(
                    f"softmax row of slot {slot} deviates by {deviation:.3e}"
                )

    walk = independent_walk(
        document,
        probability_rows,
        world["sampler_version"],
        int(world["sample_ordinal"]),
    )

    if walk["public_state_identity"] != world["public_state_identity"]:
        findings.append("the public-state identity does not re-derive")
    if walk["sample_token"] != world["sample_token"]:
        findings.append("the sample token does not re-derive")
    if walk["piece_order"] != [int(slot) for slot in world["piece_order"]]:
        findings.append("the unresolved-piece order does not re-derive")
    if walk["fallback_steps"] != [int(step) for step in world["fallback_steps"]]:
        findings.append("the zero-mass fallback steps do not re-derive")
    primary_assignment = {
        int(slot): int(rank) for slot, rank in world["assignment"].items()
    }
    if walk["assignment"] != primary_assignment:
        findings.append("the complete assignment does not re-derive")
    if walk["invariant_violations"]:
        findings.append(
            f"{walk['invariant_violations']} feasibility-invariant violations"
        )

    # Every public fact, re-checked on the primary assignment itself.
    observer = document["observer_color"]
    hidden = {
        int(piece["piece_slot"]): piece
        for piece in independent_hidden_pieces(document)
    }
    if set(primary_assignment) != set(hidden):
        findings.append("the world does not assign exactly the unresolved pieces")
    observed = [0] * RANKS
    for slot, rank in primary_assignment.items():
        if not 0 <= rank < RANKS:
            findings.append(f"slot {slot} took rank index {rank}")
            continue
        observed[rank] += 1
        piece = hidden.get(slot)
        if piece is None:
            continue
        if bool(piece["has_moved"]) and rank in IMMOVABLE:
            findings.append(
                f"moved slot {slot} was assigned {PIECE_TYPE_NAMES[rank]}"
            )
    if observed != independent_inventory(document):
        findings.append("the assigned multiset does not match the raw inventory")
    for piece in document["pieces"]:
        if piece["owner_color"] == observer:
            continue
        slot = int(piece["piece_slot"])
        if bool(piece["known_to_observer"]) and slot in primary_assignment:
            findings.append(f"known slot {slot} was assigned")
        if not bool(piece["alive"]) and slot in primary_assignment:
            findings.append(f"dead slot {slot} was assigned")

    return {
        "agrees": not findings,
        "findings": findings,
        "steps": walk["steps"],
        "guard_pruned_steps": walk["guard_pruned_steps"],
        "knife_edge_events": walk["knife_edge_events"],
        "fallback_steps": len(walk["fallback_steps"]),
    }


__all__ = [
    "IMMOVABLE",
    "INITIAL_COUNTS",
    "KNIFE_EDGE_ULPS",
    "MOVABLE",
    "Phase11IndependentAuditError",
    "RANKS",
    "independent_hidden_pieces",
    "independent_identity",
    "independent_inventory",
    "independent_mask",
    "independent_sample_token",
    "independent_seed",
    "independent_softmax",
    "independent_unit_uniform",
    "independent_walk",
    "verify_world_independently",
]
