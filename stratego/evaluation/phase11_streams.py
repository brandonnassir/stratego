"""Phase 11 Agent 4: the materialized random-stream identity universe.

Specification sources:

- `00_PHASE_11_SEQUENCE_AND_COMMON_CONTRACT.md` ("Root seeds": "All random
  needs use named domain-separated derivations")
- Agent 1's `phase11_belief_contract_v1` seed record and
  `stratego.training.phase11_seed.stream_collision_audit`
- Agent 3's recorded reading `world_sample_root_seed_derived_for_the_
  collision_audit`, whose scope this module extends to Agent 4

What is being proved
--------------------
Every random draw in Phase 11 is a pure function of a *logical identity* —
a domain plus its frozen identity parts — hashed through
:func:`~stratego.training.phase11_seed.derive_phase11_seed`. Two different
logical identities sharing a derived seed would silently couple two
independent draws, so the obligation is: over every identity Phase 11 has
actually materialized, the identity-to-seed map is injective.

Agent 3 discharged that for its own world streams plus Agent 1's enumerable
universe. Agent 4 materializes new identities that neither covered:

- world-sample tokens at sample ordinals Agent 3 never used (Agent 3 used
  0..17 learned and 0..3 baseline; Agent 4 uses 0..63), on states Agent 3
  never sampled, and their `world_order` / `world_categorical` children;
- `safety_trial` draws beyond ordinal 0 — Agent 1's enumeration pins draw 0
  of each purpose, while the attack's shuffle-retry loop and its
  no-alternative walk materialize draws 1, 2, ... as well.

Intentional reuse is not a collision
------------------------------------
The contract *requires* certain identities to repeat: the original and
permuted sides of one safety trial must issue the same sampler identity
(that is the comparison), and all eight topology legs must reissue identical
request and sample identities (that is the reproducibility claim). This
module therefore deduplicates by logical identity **first** — a global token
set, and per-token enumeration of each piece slot and step index exactly
once — so what remains is one entry per distinct identity. Only then are
seeds compared, and any duplicate that survives is an accidental collision
between two genuinely different identities.

Why the derivation is called directly
--------------------------------------
:func:`~stratego.training.phase11_seed.world_order_key` and its siblings
re-parse the sample token on every call, which is right for a sampler
issuing tens of draws and wrong for an audit issuing tens of millions.
:func:`world_stream_seeds` therefore calls
:func:`~stratego.training.phase11_seed.derive_phase11_seed` directly, and
:func:`verify_fast_path` re-derives a frozen sample through the public
helpers and requires exact agreement — so the shortcut is checked, not
assumed.
"""

from __future__ import annotations

import numpy as np

from ..training.phase11_seed import (
    DOMAIN_SAFETY_TRIAL,
    DOMAIN_WORLD_CATEGORICAL,
    DOMAIN_WORLD_ORDER,
    DOMAIN_WORLD_SAMPLE,
    SAFETY_PURPOSES,
    derive_phase11_seed,
    phase11_sample_token,
    safety_trial_seed,
    world_categorical_uniform,
    world_order_key,
    world_sample_seed,
)

#: The stream domains Agent 4 can materialize. `repro_schedule` and
#: `benchmark` are deliberately absent: both frozen selection rules are
#: hash-order rules over the recorded store that consume no randomness, so
#: the implementation instantiates neither domain. Their Agent 1 enumerable
#: entries are still carried into the combined check.
AGENT4_MATERIALIZED_DOMAINS = (
    DOMAIN_WORLD_SAMPLE,
    DOMAIN_WORLD_ORDER,
    DOMAIN_WORLD_CATEGORICAL,
    DOMAIN_SAFETY_TRIAL,
)


class Phase11StreamAuditError(RuntimeError):
    """The stream universe could not be enumerated or checked."""


# ---------------------------------------------------------------------------
# The state index, read from the Agent 2 prediction store
# ---------------------------------------------------------------------------


def store_state_index(root, manifest: dict) -> dict:
    """`public_state_identity -> unresolved piece slots`, from the store.

    A sample token names a public state, and the state fixes both the piece
    slots `world_order` is keyed by and the number of steps
    `world_categorical` is keyed by. Both come from the recorded public
    shards, so the enumeration needs no replay and no privileged read.

    A public-state identity that appears in more than one game must carry
    the same hidden-slot set in each — the identity is a hash of the whole
    document, so a disagreement would mean the document did not determine
    it. Disagreements are collected and returned rather than swallowed.
    """
    from .phase11_records import read_public_shard

    slots_by_identity: dict[str, tuple] = {}
    identity_by_decision: dict[tuple, str] = {}
    disagreements: list[dict] = []
    duplicate_identities = 0
    for entry in sorted(manifest["games_index"], key=lambda item: item["game_id"]):
        game_id = entry["game_id"]
        arrays = read_public_shard(root, game_id)
        offsets = np.asarray(arrays["event_offset"], dtype=np.int64)
        decisions = np.asarray(arrays["decision_index"], dtype=np.int64)
        piece_slots = np.asarray(arrays["piece_slot"], dtype=np.int64)
        identities = arrays["public_state_identity"]
        for position in range(len(decisions)):
            identity = bytes(identities[position]).hex()
            start = int(offsets[position])
            end = int(offsets[position + 1])
            slots = tuple(int(value) for value in piece_slots[start:end])
            identity_by_decision[(game_id, int(decisions[position]))] = identity
            existing = slots_by_identity.get(identity)
            if existing is None:
                slots_by_identity[identity] = slots
                continue
            duplicate_identities += 1
            if existing != slots:
                disagreements.append(
                    {
                        "public_state_identity": identity,
                        "game_id": game_id,
                        "decision_index": int(decisions[position]),
                    }
                )
    return {
        "slots_by_identity": slots_by_identity,
        "identity_by_decision": identity_by_decision,
        "distinct_identities": len(slots_by_identity),
        "repeated_identity_occurrences": duplicate_identities,
        "slot_set_disagreements": disagreements,
    }


# ---------------------------------------------------------------------------
# Token universes
# ---------------------------------------------------------------------------


def tokens_for(identities, ordinals, sampler_version: str) -> "set[str]":
    """Every `(identity, ordinal)` sample token under one sampler version."""
    return {
        phase11_sample_token(sampler_version, identity, int(ordinal))
        for identity in identities
        for ordinal in ordinals
    }


def token_identity(token: str) -> str:
    """The public-state identity a sample token names."""
    marker = "|ps="
    start = token.index(marker) + len(marker)
    end = token.index("|", start)
    return token[start:end]


# ---------------------------------------------------------------------------
# Seed enumeration
# ---------------------------------------------------------------------------


def world_stream_seeds(tokens, slots_by_identity: dict) -> dict:
    """The three world-stream seed arrays over a distinct token set.

    One `world_sample` seed per token; one `world_order` seed per
    `(token, piece_slot)`; one `world_categorical` seed per
    `(token, step_index)` for `step_index` in `0..n-1`. Every entry is a
    distinct logical identity by construction: the token set is a set, and
    a token's slots and steps are each enumerated once.
    """
    ordered = sorted(tokens)
    widths = []
    for token in ordered:
        identity = token_identity(token)
        slots = slots_by_identity.get(identity)
        if slots is None:
            raise Phase11StreamAuditError(
                f"no recorded public state for identity {identity[:16]}..."
            )
        if len(set(slots)) != len(slots):
            raise Phase11StreamAuditError(
                f"identity {identity[:16]}... repeats a piece slot"
            )
        widths.append(len(slots))
    total_children = int(sum(widths))

    sample = np.empty(len(ordered), dtype=np.uint64)
    order = np.empty(total_children, dtype=np.uint64)
    categorical = np.empty(total_children, dtype=np.uint64)

    cursor = 0
    for index, token in enumerate(ordered):
        slots = slots_by_identity[token_identity(token)]
        sample[index] = derive_phase11_seed(DOMAIN_WORLD_SAMPLE, token)
        for step, slot in enumerate(slots):
            order[cursor] = derive_phase11_seed(DOMAIN_WORLD_ORDER, token, int(slot))
            categorical[cursor] = derive_phase11_seed(
                DOMAIN_WORLD_CATEGORICAL, token, int(step)
            )
            cursor += 1
    if cursor != total_children:
        raise Phase11StreamAuditError("world-stream enumeration lost entries")
    return {
        DOMAIN_WORLD_SAMPLE: sample,
        DOMAIN_WORLD_ORDER: order,
        DOMAIN_WORLD_CATEGORICAL: categorical,
    }


def safety_trial_seeds(draws_by_trial: dict) -> dict:
    """The `safety_trial` seeds actually materialized, by purpose.

    `draws_by_trial[trial_id][purpose]` is the count of draw ordinals the
    attack consumed on that purpose, so ordinals `0..count-1` are
    enumerated. Agent 1's enumerable universe pins ordinal 0 of every
    purpose for all 50,000 trials; because every trial consumes at least
    one draw of every purpose, that enumeration is a subset of this one and
    is not added again — which is deduplication of an intentional shared
    identity, not a dropped check.
    """
    arrays: dict[str, np.ndarray] = {}
    for purpose in SAFETY_PURPOSES:
        total = sum(int(entry[purpose]) for entry in draws_by_trial.values())
        seeds = np.empty(total, dtype=np.uint64)
        cursor = 0
        for trial_id in sorted(draws_by_trial):
            for ordinal in range(int(draws_by_trial[trial_id][purpose])):
                seeds[cursor] = safety_trial_seed(trial_id, purpose, ordinal)
                cursor += 1
        if cursor != total:
            raise Phase11StreamAuditError(
                f"safety enumeration lost entries on {purpose}"
            )
        arrays[f"{DOMAIN_SAFETY_TRIAL}:{purpose}"] = seeds
    return arrays


def verify_fast_path(tokens, slots_by_identity: dict, draws_by_trial: dict, *, limit: int = 256) -> dict:
    """Re-derive a frozen sample through the accepted public helpers.

    The bulk enumeration calls `derive_phase11_seed` directly to avoid
    tens of millions of token re-parses. This re-derives the first `limit`
    tokens (and safety draws) through `world_sample_seed`,
    `world_order_key`, `world_categorical_uniform` and `safety_trial_seed`
    and requires exact agreement, so the shortcut is evidence rather than
    an assumption.
    """
    from ..training.phase11_seed import unit_uniform

    checked = 0
    mismatches = 0
    for token in sorted(tokens)[:limit]:
        if world_sample_seed(token) != derive_phase11_seed(DOMAIN_WORLD_SAMPLE, token):
            mismatches += 1
        checked += 1
        slots = slots_by_identity[token_identity(token)]
        for step, slot in enumerate(slots):
            if world_order_key(token, int(slot)) != derive_phase11_seed(
                DOMAIN_WORLD_ORDER, token, int(slot)
            ):
                mismatches += 1
            if world_categorical_uniform(token, int(step)) != unit_uniform(
                derive_phase11_seed(DOMAIN_WORLD_CATEGORICAL, token, int(step))
            ):
                mismatches += 1
            checked += 2
    for trial_id in sorted(draws_by_trial)[:limit]:
        for purpose in SAFETY_PURPOSES:
            for ordinal in range(int(draws_by_trial[trial_id][purpose])):
                if safety_trial_seed(trial_id, purpose, ordinal) != derive_phase11_seed(
                    DOMAIN_SAFETY_TRIAL, trial_id, purpose, ordinal
                ):
                    mismatches += 1
                checked += 1
    return {
        "derivations_checked": checked,
        "mismatches": mismatches,
        "exact": mismatches == 0,
    }


# ---------------------------------------------------------------------------
# The combined injectivity check
# ---------------------------------------------------------------------------


def combined_collision_audit(domain_arrays: dict) -> dict:
    """Require the identity-to-seed map to be injective across every domain.

    Each input array holds one entry per *distinct logical identity* of its
    domain, so any repeated seed — inside a domain or across two — is an
    accidental collision between two different identities. The check sorts
    the concatenation rather than building a 30-million-entry dictionary;
    when a duplicate does appear, the slower labelled pass runs to name the
    domains and seeds involved.
    """
    names = sorted(domain_arrays)
    if not names:
        raise Phase11StreamAuditError("no streams to audit")
    per_domain = {}
    for name in names:
        seeds = np.asarray(domain_arrays[name], dtype=np.uint64)
        distinct = int(np.unique(seeds).size) if seeds.size else 0
        per_domain[name] = {
            "identities": int(seeds.size),
            "distinct_seeds": distinct,
            "internal_duplicates": int(seeds.size) - distinct,
        }
    combined = np.concatenate(
        [np.asarray(domain_arrays[name], dtype=np.uint64) for name in names]
    )
    total = int(combined.size)
    ordered = np.sort(combined, kind="stable")
    duplicate_mask = ordered[1:] == ordered[:-1] if total > 1 else np.zeros(0, bool)
    collisions = int(np.count_nonzero(duplicate_mask))
    distinct_total = total - collisions

    findings: list[dict] = []
    if collisions:
        labels = np.concatenate(
            [
                np.full(int(np.asarray(domain_arrays[name]).size), index, dtype=np.int32)
                for index, name in enumerate(names)
            ]
        )
        permutation = np.argsort(combined, kind="stable")
        sorted_seeds = combined[permutation]
        sorted_labels = labels[permutation]
        positions = np.nonzero(sorted_seeds[1:] == sorted_seeds[:-1])[0]
        for position in positions[:32]:
            findings.append(
                {
                    "seed": int(sorted_seeds[position]),
                    "domains": sorted(
                        {
                            names[int(sorted_labels[position])],
                            names[int(sorted_labels[position + 1])],
                        }
                    ),
                }
            )
    return {
        "per_domain": per_domain,
        "domains": names,
        "total_identities": total,
        "distinct_seeds": distinct_total,
        "accidental_collisions": collisions,
        "no_collisions": collisions == 0,
        "findings": findings,
        "bit_width": 63,
        "expected_random_collisions": round(
            (total * (total - 1) / 2.0) / float(1 << 63), 6
        ),
    }


__all__ = [
    "AGENT4_MATERIALIZED_DOMAINS",
    "Phase11StreamAuditError",
    "combined_collision_audit",
    "safety_trial_seeds",
    "store_state_index",
    "token_identity",
    "tokens_for",
    "verify_fast_path",
    "world_stream_seeds",
]
