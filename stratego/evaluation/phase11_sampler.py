"""Phase 11 Agent 3: the learned `belief_sampler_v1` and its request boundary.

Specification sources:

- `03_AGENT_3_CONSTRAINED_BELIEF_SAMPLER.md` ("Implement exact frozen
  algorithm", "API boundary")
- Agent 1's `phase11_belief_sampler_v1` (the twelve frozen steps, the
  completion-feasibility rule, the provenance fields, the validation stack)

What Agent 3 adds and what it reuses
------------------------------------
Agent 1 froze one sampler skeleton with two weightings: the learned
`belief_sampler_v1` (step 7: `learned_probability * remaining_count`) and
the nonlearned `count_uniform_world_sampler_v1` (the same skeleton with the
weight reduced to `remaining_count` alone). Agent 2 built the count-only
baseline; this module builds the learned sampler **on the same frozen
primitives** — :func:`~stratego.evaluation.phase11_baselines.feasible_ranks`,
:func:`~stratego.evaluation.phase11_baselines.inverse_cdf_choice` and
:func:`~stratego.evaluation.phase11_baselines.validate_world` — because the
contract's "same piece order, same categorical walk, same feasibility rule,
same validation stack" is a statement about *identity*, and two copies of a
frozen rule are two chances for it to mean two things. The Agent 3 harness
then re-verifies every constraint through a second implementation path
(:mod:`stratego.evaluation.phase11_sampler_audit`) that shares none of these
primitives, so the reuse here is checked, not trusted.

The completion-feasibility guard reads public inputs only
---------------------------------------------------------
Step 6's guard admits a movable rank for an *unmoved* piece only when
`movable_remaining - 1 >= moved_unresolved_remaining`. Every quantity in
that inequality is derived from the public-state document alone:

- `movable_remaining` sums the public remaining inventory `c[r]` (initial
  army minus publicly known ranks) over the ten movable rank indices;
- `moved_unresolved_remaining` counts public `has_moved` flags over the
  not-yet-assigned pieces after the current one;
- the piece's own `has_moved` flag and 12-entry legal-rank mask are public.

No hidden rank appears in any of these, and none *can*: the request type
below has no field that could carry one, and
:meth:`Phase11SamplerRequest.from_payload` raises on any field outside the
frozen four-name allowlist. The guard is therefore a pure function of
public constraints — the proof the Agent 3 instruction requires — and the
harness's independent path recomputes it from the raw document on every
audited step.

Purity
------
A sampled world is a pure function of `(public_state_identity,
belief-model identity, sampler_version, sample_ordinal)`. The only
randomness consumed is the frozen `world_order` / `world_categorical`
streams keyed by the sample token; no mutable RNG cursor exists anywhere in
this module, and nothing here reads worker count, call order, process id,
path or wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from ..training.phase11_contract import (
    ALLOWED_SAMPLER_REQUEST_FIELDS,
    BELIEF_SAMPLER_VERSION,
    FORBIDDEN_SAMPLER_REQUEST_TOKENS,
    PUBLIC_PIECE_FIELDS,
    PUBLIC_STATE_DOCUMENT_FIELDS,
    PUBLIC_STATE_DOCUMENT_VERSION,
    Phase11ContractError,
    RANK_COUNT,
    SAMPLER_PROVENANCE_FIELDS,
)
from ..training.phase11_seed import (
    BELIEF_MODEL_LABEL,
    MAX_SAMPLE_ORDINAL_FORMAT,
    phase11_sample_token,
    world_categorical_uniform,
    world_order_key,
)
from .phase11_baselines import (
    feasible_ranks,
    inverse_cdf_choice,
    remaining_counts,
    validate_world,
)
from .phase11_public_state import (
    hidden_opponent_pieces,
    legal_rank_mask,
    public_state_identity,
)

#: The frozen field sets a request's document must carry — exactly. Set
#: comparison, not tuple comparison, because a canonical-JSON round trip
#: sorts keys and the boundary is about *what* arrives, not its order.
_DOCUMENT_FIELD_SET = frozenset(PUBLIC_STATE_DOCUMENT_FIELDS)
_PIECE_FIELD_SET = frozenset(PUBLIC_PIECE_FIELDS)

#: The frozen fields of one `recent_moves` entry, exactly as the accepted
#: document builder emits them. Two of these names carry the substring
#: "target" — they name the *publicly attacked piece*, which both players
#: watch, so the boundary for nested frozen content is exact schema
#: equality, never a token scan of the frozen names themselves.
_MOVE_FIELD_SET = frozenset(
    {
        "ply",
        "player_color",
        "piece_slot",
        "piece_owner_color",
        "source",
        "destination",
        "was_attack",
        "target_piece_slot",
        "target_owner_color",
    }
)

_VALID_COLORS = ("red", "blue")


class Phase11SamplerError(Phase11ContractError):
    """A sampler request was refused, or a sampled world failed a check."""


class Phase11SamplerDeadEndError(Phase11SamplerError):
    """The walk found no legal rank — provably impossible under the guard.

    Raised so a dead end can never be silently absorbed; the audits count
    every raise into the zero-tolerance `dead_end_events` counter.
    """


def _forbidden_token_in(name) -> "str | None":
    lowered = str(name).lower()
    for token in FORBIDDEN_SAMPLER_REQUEST_TOKENS:
        if token in lowered:
            return token
    return None


def _refuse_schema_drift(kind: str, keys, frozen: frozenset) -> None:
    """Refuse any container whose fields are not exactly the frozen schema.

    The refusal names the unexpected fields, and additionally names any
    frozen forbidden token they carry, so a smuggling attempt is called
    what it is.
    """
    keys = set(keys)
    if keys == frozen:
        return
    unexpected = sorted(keys - frozen)
    missing = sorted(frozen - keys)
    tokens = sorted(
        {
            token
            for token in (_forbidden_token_in(key) for key in unexpected)
            if token is not None
        }
    )
    suffix = f"; forbidden tokens: {', '.join(tokens)}" if tokens else ""
    raise Phase11SamplerError(
        f"the document's {kind} fields are not exactly the frozen schema "
        f"(unexpected {unexpected}, missing {missing}{suffix})"
    )


@dataclass(frozen=True)
class Phase11SamplerRequest:
    """One complete-world sampling request. Public information, structurally.

    Exactly the four frozen fields, in the frozen order. There is no field
    a true rank, a private piece table, a setup truth, a result or a
    storage path could arrive in, and :meth:`from_payload` raises — never
    drops — on anything outside the allowlist. That structural absence is
    the sampler's `hidden_input_accesses = 0` claim.
    """

    sampler_version: str
    public_state_document: dict
    learned_probabilities: dict
    sample_ordinal: int

    def __post_init__(self) -> None:
        if tuple(item.name for item in fields(self)) != ALLOWED_SAMPLER_REQUEST_FIELDS:
            raise Phase11SamplerError(
                "Phase11SamplerRequest fields drifted from the frozen allowlist"
            )
        if self.sampler_version != BELIEF_SAMPLER_VERSION:
            raise Phase11SamplerError(
                f"sampler version {self.sampler_version!r} is not "
                f"{BELIEF_SAMPLER_VERSION!r}; the count-uniform baseline has its "
                "own accepted path in phase11_baselines"
            )
        if (
            not isinstance(self.sample_ordinal, int)
            or isinstance(self.sample_ordinal, bool)
            or not 0 <= self.sample_ordinal <= MAX_SAMPLE_ORDINAL_FORMAT
        ):
            raise Phase11SamplerError(
                f"sample_ordinal must be an int in 0..{MAX_SAMPLE_ORDINAL_FORMAT}, "
                f"got {self.sample_ordinal!r}"
            )
        document = self.public_state_document
        if not isinstance(document, dict):
            raise Phase11SamplerError("public_state_document must be a mapping")
        if document.get("document_version") != PUBLIC_STATE_DOCUMENT_VERSION:
            raise Phase11SamplerError(
                "public_state_document is not a "
                f"{PUBLIC_STATE_DOCUMENT_VERSION!r} document"
            )
        _refuse_schema_drift("top-level", document, _DOCUMENT_FIELD_SET)
        if document.get("observer_color") not in _VALID_COLORS:
            raise Phase11SamplerError(
                f"observer_color {document.get('observer_color')!r} is not a colour"
            )
        for piece in document["pieces"]:
            if not isinstance(piece, dict):
                raise Phase11SamplerError("a document piece is not a mapping")
            _refuse_schema_drift("piece", piece, _PIECE_FIELD_SET)
        for move in document["recent_moves"]:
            if not isinstance(move, dict):
                raise Phase11SamplerError("a document move is not a mapping")
            _refuse_schema_drift("recent-move", move, _MOVE_FIELD_SET)

        hidden_slots = {
            int(piece["piece_slot"]) for piece in hidden_opponent_pieces(document)
        }
        probabilities = self.learned_probabilities
        if not isinstance(probabilities, dict):
            raise Phase11SamplerError("learned_probabilities must be a mapping")
        keys = set()
        for key in probabilities:
            if not isinstance(key, int) or isinstance(key, bool):
                raise Phase11SamplerError(
                    f"learned_probabilities keys must be int piece slots, got {key!r}"
                )
            keys.add(int(key))
        if keys != hidden_slots:
            raise Phase11SamplerError(
                "learned_probabilities must cover exactly the unresolved opponent "
                f"pieces (got slots {sorted(keys)}, unresolved {sorted(hidden_slots)})"
            )
        if hidden_slots:
            rows = np.stack(
                [
                    np.asarray(probabilities[slot], dtype=np.float64)
                    for slot in sorted(hidden_slots)
                ]
            )
            if rows.shape != (len(hidden_slots), RANK_COUNT):
                raise Phase11SamplerError(
                    f"learned probability rows have shape {rows.shape}, expected "
                    f"({len(hidden_slots)}, {RANK_COUNT})"
                )
            if not np.isfinite(rows).all():
                raise Phase11SamplerError(
                    "a learned probability row carries a non-finite value"
                )
            if (rows < 0.0).any():
                raise Phase11SamplerError(
                    "a learned probability row carries a negative value"
                )

    @property
    def public_state_identity(self) -> str:
        return public_state_identity(self.public_state_document)

    @classmethod
    def from_payload(cls, payload) -> "Phase11SamplerRequest":
        """Build from an untrusted mapping, refusing anything off-allowlist.

        Raises on an unknown field, on a field whose name carries a frozen
        forbidden token, and on a missing field. Nothing is silently
        dropped: a dropped field is a leak that succeeded quietly. This is
        the boundary the negative controls attack with `true_rank`,
        `private_piece_table`, `opponent_setup`, `winner`, `storage_path`
        and friends.
        """
        if not isinstance(payload, dict):
            raise Phase11SamplerError(
                f"a sampler request payload must be a mapping, got "
                f"{type(payload).__name__}"
            )
        unknown = [key for key in payload if key not in ALLOWED_SAMPLER_REQUEST_FIELDS]
        if unknown:
            offending = ", ".join(sorted(str(key) for key in unknown))
            tokens = sorted(
                {
                    token
                    for token in (_forbidden_token_in(key) for key in unknown)
                    if token is not None
                }
            )
            suffix = f" (forbidden tokens: {', '.join(tokens)})" if tokens else ""
            raise Phase11SamplerError(
                f"sampler request carries fields outside the frozen allowlist: "
                f"{offending}{suffix}"
            )
        missing = [key for key in ALLOWED_SAMPLER_REQUEST_FIELDS if key not in payload]
        if missing:
            raise Phase11SamplerError(
                f"sampler request is missing {', '.join(missing)}"
            )
        probabilities = payload["learned_probabilities"]
        if isinstance(probabilities, dict):
            coerced = {}
            for key, row in probabilities.items():
                token = _forbidden_token_in(key)
                if token is not None:
                    raise Phase11SamplerError(
                        f"learned_probabilities carries a forbidden key {key!r} "
                        f"(token {token!r})"
                    )
                try:
                    slot = int(key)
                except (TypeError, ValueError):
                    raise Phase11SamplerError(
                        f"learned_probabilities key {key!r} is not a piece slot"
                    ) from None
                coerced[slot] = row
            probabilities = coerced
        return cls(
            sampler_version=payload["sampler_version"],
            public_state_document=payload["public_state_document"],
            learned_probabilities=probabilities,
            sample_ordinal=payload["sample_ordinal"],
        )


def sampler_boundary_report() -> dict:
    """The sampler-side boundary, as data, for the audit artifact."""
    return {
        "sampler_version": BELIEF_SAMPLER_VERSION,
        "allowed_request_fields": list(ALLOWED_SAMPLER_REQUEST_FIELDS),
        "forbidden_tokens": list(FORBIDDEN_SAMPLER_REQUEST_TOKENS),
        "request_type_rejects_truth": True,
        "request_type_field_for_truth_exists": False,
        "guard_inputs": [
            "public remaining inventory c[r] (initial army minus publicly "
            "known ranks)",
            "public has_moved flags of the not-yet-assigned unresolved pieces",
            "the current piece's public has_moved flag and legal-rank mask",
        ],
    }


def sample_belief_world(
    request: Phase11SamplerRequest, *, run_validation_stack: bool = True
) -> dict:
    """One complete legal hidden world under the frozen `belief_sampler_v1`.

    The twelve frozen steps, exactly:

    1.  the request carries the public document and learned marginals only;
    2.  publicly known ranks are locked (they are never sampled);
    3.  `remaining_counts` computes the exact public inventory;
    4.  `legal_rank_mask` applies the public impossibility masks;
    5.  the unresolved-piece order is ascending
        `(world_order_key(sample_token, piece_slot), piece_slot)`;
    6.  legal ranks have remaining count > 0, pass the mask, and satisfy
        the frozen completion-feasibility guard (`feasible_ranks`);
    7.  `weight = learned_probability * remaining_count`;
    8.  `inverse_cdf_choice` walks the float64 cumulative mass at
        `world_categorical_uniform(sample_token, step_index)`, last legal
        rank as the tail guard;
    9.  the chosen rank's remaining count decrements;
    10. zero learned mass on the legal set falls back to the remaining
        counts over the same legal set (recorded in `fallback_steps`);
    11. the walk continues until every unresolved piece is assigned;
    12. the complete world is verified against the frozen validation stack
        (`validate_world`), and any finding raises.
    """
    if not isinstance(request, Phase11SamplerRequest):
        raise Phase11SamplerError(
            "the learned sampler accepts only a Phase11SamplerRequest, got "
            f"{type(request).__name__}"
        )
    document = request.public_state_document
    identity = public_state_identity(document)
    token = phase11_sample_token(
        request.sampler_version, identity, int(request.sample_ordinal)
    )
    counts = list(remaining_counts(document))
    hidden = hidden_opponent_pieces(document)
    probabilities = {
        int(slot): np.asarray(row, dtype=np.float64)
        for slot, row in request.learned_probabilities.items()
    }

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
            raise Phase11SamplerDeadEndError(
                f"the sampler dead-ended at step {step_index} on slot {slot}; "
                "the feasibility guard proves this cannot happen on a publicly "
                "consistent state"
            )
        row = probabilities[slot]
        weights = np.array(
            [float(row[rank]) * float(counts[rank]) for rank in ranks],
            dtype=np.float64,
        )
        if not np.isfinite(weights).all() or (weights < 0.0).any():
            raise Phase11SamplerError(
                f"a non-finite or negative weight row at step {step_index} "
                f"(slot {slot})"
            )
        if float(weights.sum()) <= 0.0:
            fallback_steps.append(step_index)
            weights = np.array(
                [float(counts[rank]) for rank in ranks], dtype=np.float64
            )
        uniform = world_categorical_uniform(token, step_index)
        chosen = inverse_cdf_choice(ranks, weights, uniform)
        assignment[slot] = chosen
        counts[chosen] -= 1

    world = {
        "sample_token": token,
        "sampler_version": request.sampler_version,
        "public_state_identity": identity,
        "belief_model_label": BELIEF_MODEL_LABEL,
        "sample_ordinal": int(request.sample_ordinal),
        "piece_order": [int(piece["piece_slot"]) for piece in order],
        "fallback_steps": fallback_steps,
        "assignment": {
            int(slot): int(rank) for slot, rank in sorted(assignment.items())
        },
        "dead_end_events": 0,
    }
    if tuple(world)[: len(SAMPLER_PROVENANCE_FIELDS)] != SAMPLER_PROVENANCE_FIELDS:
        raise Phase11SamplerError(
            "the world's provenance fields drifted from the frozen schema"
        )

    if run_validation_stack:
        check = validate_world(document, world)
        if not check["valid"]:
            raise Phase11SamplerError(
                "the sampled world failed the frozen validation stack: "
                f"{check['findings'][:3]}"
            )
    return world


__all__ = [
    "Phase11SamplerDeadEndError",
    "Phase11SamplerError",
    "Phase11SamplerRequest",
    "sample_belief_world",
    "sampler_boundary_report",
]
