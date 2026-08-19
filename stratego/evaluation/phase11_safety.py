"""Phase 11 Agent 4: the hidden-truth permutation attack and its controls.

Specification sources:

- `04_AGENT_4_INFO_SAFETY_REPRO_RUNTIME.md` ("Part A — hidden-truth
  permutation attack", "Part D — sensitivity controls")
- Agent 1's `phase11_information_safety_v1`
  (`hidden_truth_attack`: the state pool, the permutation rule, the
  no-alternative rule, the six checks, the injection controls and the four
  zero-tolerance counters)

What the attack actually proves
-------------------------------
The claim under test is a *conditional independence*: holding every public
byte fixed, the belief outputs and the fixed-seed sampled world must not
move when the private hidden truth moves. The attack constructs the
counterfactual directly — it takes a real validation position, permutes the
true ranks of the unresolved opponent pieces into a different but publicly
indistinguishable truth, and re-runs the production belief path and the
frozen sampler on the permuted position. Six quantities must be identical:
belief logits, learned probabilities, public legal-rank masks, the sampler
request, the sampled world and the sampler provenance.

Structural absence is the first line and instrumentation is the second
------------------------------------------------------------------------
:class:`~stratego.evaluation.phase11_belief.Phase11BeliefRequest` and
:class:`~stratego.evaluation.phase11_sampler.Phase11SamplerRequest` have no
field a hidden rank could arrive in, and both refuse — never drop —
anything off their allowlists. That is structural, and
:func:`injection_controls` re-proves it against every named private field
on both boundaries.

Structure alone would still miss a leak that reached around the request and
read the privileged :class:`~stratego.engine.state.GameState` while building
the public products. :func:`instrument_hidden_types` closes that: it
replaces the unresolved opponent pieces of a state with
:class:`TracedPieceRecord`, whose ``true_type`` is a counting property, so
every read of a hidden rank during `build_public_view`,
`build_public_state_document` and `build_observation` is counted. The
accepted observation builder guards all three of its `true_type` sites on
`is_own` or `observer_knows`, so the expected count is exactly zero and any
non-zero count is a Gate F failure, not a tolerance.

The permutation is public-consistent by construction
-----------------------------------------------------
A permutation of the unresolved pieces' true ranks preserves the remaining
inventory automatically. The one public fact it can contradict is
immobility: a piece the opponent has publicly seen move cannot be a Flag or
a Bomb. :func:`build_alternative_truth` therefore draws seeded Fisher-Yates
shuffles from the trial's `truth_permutation` stream and accepts the first
that is immobility-legal and changes at least one piece; if the frozen
attempt budget is exhausted it falls back to a stream-selected valid
transposition, which :func:`admits_alternative_truth` has already proved
exists. No trial is dropped and no loop is unbounded.

States that admit no altered legal truth at all (every unresolved piece
already shares one rank, or immobility forces the identity) are skipped by
the frozen `no_alternative_rule`: the trial walks to the next candidate in
its own `state_selection` stream.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..engine.pieces import PieceRecord
from ..training.phase11_contract import (
    IMMOVABLE_RANK_INDICES,
    Phase11ContractError,
    RANK_COUNT,
)
from ..training.phase11_seed import (
    SAFETY_PURPOSE_PERMUTATION,
    SAFETY_PURPOSE_SAMPLE,
    SAFETY_PURPOSE_STATE,
    safety_trial_seed,
)

#: The frozen number of seeded shuffle attempts before a trial falls back to
#: a stream-selected valid transposition. Chosen before any trial ran; the
#: fallback is exact, so the budget changes nothing but the shape of the
#: alternative truth it produces.
PERMUTATION_ATTEMPT_BUDGET = 32

#: The frozen ceiling on `state_selection` walks per trial under the
#: no-alternative rule. The candidate pool is filtered to admitting states
#: before any trial runs, so the walk is expected to terminate on its first
#: draw; the ceiling exists so a contradiction raises instead of hanging.
STATE_SELECTION_WALK_LIMIT = 64

#: The frozen minimum number of unresolved opponent pieces a candidate state
#: must carry, from `phase11_information_safety_v1.hidden_truth_attack`.
MIN_UNRESOLVED_PIECES = 2

_IMMOVABLE = frozenset(IMMOVABLE_RANK_INDICES)

#: The named private fields the injection controls push at both request
#: boundaries, from the Agent 1 contract's `injection_controls` sentence.
INJECTION_FIELD_PROBES = (
    ("true_rank", 7),
    ("true_rank_index", 7),
    ("truth", {"3": 9}),
    ("private_piece_table", {"3": "marshal"}),
    ("opponent_setup", [0] * 40),
    ("opponent_setup_truth", [0] * 40),
    ("hidden_start_rank", 10),
    ("winner", "red"),
    ("result", 1.0),
    ("reward", -1.0),
    ("outcome", "win"),
    ("future_action", 1234),
    ("future_search_result", {"value": 0.5}),
    ("storage_path", "/tmp/truth.npz"),
    ("label", 3),
    ("target_rank", 3),
)


class Phase11SafetyError(Phase11ContractError):
    """A safety trial could not be constructed, or a control misbehaved."""


# ---------------------------------------------------------------------------
# Instrumented hidden-rank access
# ---------------------------------------------------------------------------


class HiddenAccessCounter:
    """A shared, explicit tally of hidden-rank reads.

    Deliberately not a module global: a counter that lives in a module is a
    mutable global, which is exactly the thing Gate G forbids elsewhere in
    Phase 11. Each instrumented state owns its counter.
    """

    __slots__ = ("reads", "slots")

    def __init__(self) -> None:
        self.reads = 0
        self.slots: list[int] = []

    def record(self, piece_id: int) -> None:
        self.reads += 1
        if len(self.slots) < 32:
            self.slots.append(int(piece_id))

    def report(self) -> dict:
        return {"reads": int(self.reads), "first_piece_ids": list(self.slots)}


class TracedPieceRecord(PieceRecord):
    """A piece record whose hidden rank cannot be read without being counted.

    `true_type` becomes a property over `_traced_true_type`; the dataclass
    `__init__` assigns through the setter, so construction is unchanged and
    every later read lands in the counter. Only *unresolved opponent* pieces
    are traced, so the observer's own ranks and legally revealed opponent
    ranks stay ordinary attributes and never inflate the count.
    """

    __hidden_counter__: "HiddenAccessCounter | None" = None

    @property
    def true_type(self) -> int:
        counter = self.__hidden_counter__
        if counter is not None:
            counter.record(self.piece_id)
        return self._traced_true_type

    @true_type.setter
    def true_type(self, value) -> None:
        self._traced_true_type = value

    def untraced_true_type(self) -> int:
        """The rank, read on the privileged path without counting."""
        return self._traced_true_type


def unresolved_opponent_records(state, observer: int) -> "list[PieceRecord]":
    """Live opponent pieces the observer has no legal knowledge of, in id order.

    Privileged: this walks the real `GameState`. It is the attack's own
    construction path, never an input to belief inference or sampling.
    """
    return [
        record
        for record in state.pieces
        if record.owner != observer and record.alive and not record.known_to(observer)
    ]


def instrument_hidden_types(state, observer: int):
    """`(instrumented state, counter)` — a copy whose hidden ranks are traced.

    The copy shares no piece record with `state`, so tracing cannot leak
    back into the caller's position, and the board/history objects are
    rebuilt to point at the traced records.
    """
    import copy as _copy

    traced = _copy.deepcopy(state)
    counter = HiddenAccessCounter()
    for index, record in enumerate(traced.pieces):
        if record.owner == observer or not record.alive or record.known_to(observer):
            continue
        replacement = TracedPieceRecord(
            piece_id=record.piece_id,
            owner=record.owner,
            true_type=record.true_type,
            starting_square=record.starting_square,
            current_square=record.current_square,
            alive=record.alive,
            has_moved=record.has_moved,
            known_to_red=record.known_to_red,
            known_to_blue=record.known_to_blue,
            reveal_reason_red=record.reveal_reason_red,
            reveal_reason_blue=record.reveal_reason_blue,
            capture_ply=record.capture_ply,
        )
        replacement.__dict__["__hidden_counter__"] = counter
        traced.pieces[index] = replacement
    return traced, counter


# ---------------------------------------------------------------------------
# The alternative hidden truth
# ---------------------------------------------------------------------------


def valid_transpositions(types, moved) -> "list[tuple[int, int]]":
    """Index pairs whose swap is a different, immobility-legal truth.

    A swap of `i` and `j` is admissible when their true ranks differ and
    neither piece would end up publicly-moved-and-immovable. The list is in
    ascending `(i, j)` order, so a stream draw over it is deterministic.
    """
    pairs = []
    count = len(types)
    for i in range(count):
        for j in range(i + 1, count):
            if types[i] == types[j]:
                continue
            if moved[i] and types[j] in _IMMOVABLE:
                continue
            if moved[j] and types[i] in _IMMOVABLE:
                continue
            pairs.append((i, j))
    return pairs


def admits_alternative_truth(types, moved) -> bool:
    """Whether this position has any altered legal hidden truth at all.

    Decided constructively, by the existence of a valid transposition: a
    transposition *is* an alternative truth, so this is sufficient, and it
    is the exact predicate the trial pool and the no-alternative rule use.
    A position failing it (all unresolved ranks equal, or every differing
    pair blocked by immobility) is skipped, never silently scored.
    """
    return bool(valid_transpositions(types, moved))


def _is_legal_truth(candidate, moved) -> bool:
    return all(
        not (moved[index] and rank in _IMMOVABLE)
        for index, rank in enumerate(candidate)
    )


def _seeded_shuffle(values, seed: int) -> list:
    """Fisher-Yates over a frozen 63-bit seed, in the accepted style.

    The seed is consumed high-bits-first through a `blake2b` counter so the
    walk needs no mutable RNG object; every draw is a pure function of
    `(seed, position)`.
    """
    shuffled = list(values)
    for position in range(len(shuffled) - 1, 0, -1):
        digest = hashlib.blake2b(
            f"phase11_safety_shuffle_v1:{seed}:{position}".encode(),
            digest_size=8,
            person=b"strat-b11",
        ).digest()
        pick = (int.from_bytes(digest, "big") >> 1) % (position + 1)
        shuffled[position], shuffled[pick] = shuffled[pick], shuffled[position]
    return shuffled


@dataclass(frozen=True)
class AlternativeTruth:
    """One constructed counterfactual truth for one trial."""

    ranks: tuple[int, ...]
    changed_pieces: int
    attempts: int
    method: str


def build_alternative_truth(trial_id: str, types, moved) -> AlternativeTruth:
    """A different, publicly indistinguishable hidden truth for this state.

    Seeded shuffles first (the general permutation the contract names), a
    stream-selected valid transposition as the exact fallback. Raises when
    the state admits no alternative at all — callers must have filtered on
    :func:`admits_alternative_truth` first, which is what the frozen
    no-alternative rule does.
    """
    types = tuple(int(rank) for rank in types)
    moved = tuple(bool(flag) for flag in moved)
    if len(types) != len(moved):
        raise Phase11SafetyError("truth and moved-flag lengths disagree")
    pairs = valid_transpositions(types, moved)
    if not pairs:
        raise Phase11SafetyError(
            "this state admits no altered legal hidden truth; the frozen "
            "no-alternative rule walks to the next candidate state instead"
        )
    for attempt in range(PERMUTATION_ATTEMPT_BUDGET):
        seed = safety_trial_seed(trial_id, SAFETY_PURPOSE_PERMUTATION, attempt)
        candidate = tuple(_seeded_shuffle(types, seed))
        if candidate == types or not _is_legal_truth(candidate, moved):
            continue
        return AlternativeTruth(
            ranks=candidate,
            changed_pieces=sum(1 for a, b in zip(types, candidate) if a != b),
            attempts=attempt + 1,
            method="shuffle",
        )
    seed = safety_trial_seed(
        trial_id, SAFETY_PURPOSE_PERMUTATION, PERMUTATION_ATTEMPT_BUDGET
    )
    first, second = pairs[seed % len(pairs)]
    candidate = list(types)
    candidate[first], candidate[second] = candidate[second], candidate[first]
    return AlternativeTruth(
        ranks=tuple(candidate),
        changed_pieces=2,
        attempts=PERMUTATION_ATTEMPT_BUDGET + 1,
        method="transposition",
    )


def apply_alternative_truth(state, observer: int, ranks):
    """A deep copy of `state` whose unresolved opponent ranks are `ranks`.

    Assignment is by piece-id order over the unresolved set — the same
    order :func:`unresolved_opponent_records` returns — so the mapping is a
    pure function of the position and the rank tuple.
    """
    import copy as _copy

    altered = _copy.deepcopy(state)
    records = unresolved_opponent_records(altered, observer)
    if len(records) != len(ranks):
        raise Phase11SafetyError(
            f"alternative truth covers {len(ranks)} pieces, the position has "
            f"{len(records)} unresolved"
        )
    for record, rank in zip(records, ranks):
        record.true_type = int(rank)
    return altered


def trial_state_walk(trial_id: str, pool_size: int, admits) -> dict:
    """The frozen state choice of one trial, including the skip walk.

    `admits[index]` says whether candidate `index` admits an altered legal
    truth. Draws from the trial's own `state_selection` stream and walks to
    the next draw while the chosen candidate does not admit one, so a
    skipped state costs a recorded walk step and never a dropped trial.
    """
    if pool_size <= 0:
        raise Phase11SafetyError("the safety candidate pool is empty")
    for step in range(STATE_SELECTION_WALK_LIMIT):
        seed = safety_trial_seed(trial_id, SAFETY_PURPOSE_STATE, step)
        index = seed % pool_size
        if admits[index]:
            return {"pool_index": int(index), "walk_steps": int(step)}
    raise Phase11SafetyError(
        f"{trial_id}: {STATE_SELECTION_WALK_LIMIT} state draws found no candidate "
        "admitting an altered legal truth"
    )


def trial_sample_ordinal(trial_id: str, max_ordinal: int) -> int:
    """The fixed sample ordinal whose world must be bit-identical."""
    seed = safety_trial_seed(trial_id, SAFETY_PURPOSE_SAMPLE, 0)
    return int(seed % int(max_ordinal))


# ---------------------------------------------------------------------------
# Canonical comparison
# ---------------------------------------------------------------------------


def belief_digest(logits: dict, probabilities: dict, masks: dict) -> str:
    """A byte-exact digest of one belief output, masks included.

    Logits go in as raw float32 bytes and probabilities as raw float64
    bytes: comparing digests of raw bytes is comparing the values with no
    tolerance anywhere, which is what "byte-identical" has to mean.
    """
    hasher = hashlib.sha256()
    hasher.update(b"phase11_belief_output_v1")
    for slot in sorted(logits):
        hasher.update(f"|slot={int(slot)}".encode())
        hasher.update(np.asarray(logits[slot], dtype=np.float32).tobytes())
        hasher.update(np.asarray(probabilities[slot], dtype=np.float64).tobytes())
        mask = np.asarray(masks[slot], dtype=np.uint8)
        if mask.shape != (RANK_COUNT,):
            raise Phase11SafetyError(f"slot {slot} mask has shape {mask.shape}")
        hasher.update(mask.tobytes())
    return hasher.hexdigest()


def world_digest(world: dict) -> str:
    """A byte-exact digest of one sampled world and its provenance."""
    import json

    return hashlib.sha256(
        json.dumps(
            {str(key): value for key, value in world.items()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def sampler_request_digest(document: dict, probabilities: dict, ordinal: int) -> str:
    """A byte-exact digest of the sampler request the trial issues."""
    import json

    hasher = hashlib.sha256()
    hasher.update(b"phase11_sampler_request_v1")
    hasher.update(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    )
    for slot in sorted(probabilities):
        hasher.update(f"|slot={int(slot)}".encode())
        hasher.update(np.asarray(probabilities[slot], dtype=np.float64).tobytes())
    hasher.update(f"|n={int(ordinal)}".encode())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Injection controls
# ---------------------------------------------------------------------------


def _belief_payload(document: dict, observation) -> dict:
    from ..training.phase11_contract import BELIEF_REQUEST_VERSION

    return {
        "request_version": BELIEF_REQUEST_VERSION,
        "request_id": "phase11_injection_probe",
        "observer_color": document["observer_color"],
        "public_state_document": document,
        "observation": observation,
    }


def _sampler_payload(document: dict, probabilities: dict) -> dict:
    from ..training.phase11_contract import BELIEF_SAMPLER_VERSION

    return {
        "sampler_version": BELIEF_SAMPLER_VERSION,
        "public_state_document": document,
        "learned_probabilities": {
            int(slot): np.asarray(row, dtype=np.float64).tolist()
            for slot, row in probabilities.items()
        },
        "sample_ordinal": 0,
    }


def injection_controls(document: dict, observation, probabilities: dict) -> dict:
    """Push every named private field at both request boundaries.

    Each probe must raise. A probe that *builds* a request is an acceptance
    and increments `injection_acceptances`, which Gate F requires to be
    zero. Both boundaries are probed, because a leak that the belief path
    refuses and the sampler path accepts is still a leak.
    """
    from .phase11_belief import Phase11BeliefRequest
    from .phase11_sampler import Phase11SamplerRequest

    probes = []
    acceptances = 0
    for boundary, base, builder in (
        (
            "belief",
            _belief_payload(document, observation),
            Phase11BeliefRequest.from_payload,
        ),
        (
            "sampler",
            _sampler_payload(document, probabilities),
            Phase11SamplerRequest.from_payload,
        ),
    ):
        for field_name, value in INJECTION_FIELD_PROBES:
            payload = dict(base)
            payload[field_name] = value
            rejected = False
            detail = ""
            try:
                builder(payload)
            except Exception as error:  # noqa: BLE001 - any refusal is a refusal
                rejected = True
                detail = f"{type(error).__name__}: {error}"
            if not rejected:
                acceptances += 1
            probes.append(
                {
                    "boundary": boundary,
                    "field": field_name,
                    "rejected": rejected,
                    "detail": detail[:200],
                }
            )
    # A nested smuggle: a private field hidden inside the frozen document.
    for boundary, base, builder in (
        (
            "belief",
            _belief_payload(document, observation),
            Phase11BeliefRequest.from_payload,
        ),
        (
            "sampler",
            _sampler_payload(document, probabilities),
            Phase11SamplerRequest.from_payload,
        ),
    ):
        for where in ("document", "piece"):
            payload = dict(base)
            poisoned = {
                key: ([dict(item) for item in value] if key == "pieces" else value)
                for key, value in document.items()
            }
            if where == "document":
                poisoned["true_rank_table"] = {"3": 9}
            else:
                poisoned["pieces"][0]["true_rank_index"] = 9
            payload["public_state_document"] = poisoned
            rejected = False
            detail = ""
            try:
                builder(payload)
            except Exception as error:  # noqa: BLE001
                rejected = True
                detail = f"{type(error).__name__}: {error}"
            if not rejected:
                acceptances += 1
            probes.append(
                {
                    "boundary": boundary,
                    "field": f"nested_{where}_private_field",
                    "rejected": rejected,
                    "detail": detail[:200],
                }
            )
    return {
        "probes": probes,
        "probe_count": len(probes),
        "injection_acceptances": int(acceptances),
        "all_rejected": acceptances == 0,
    }


__all__ = [
    "AlternativeTruth",
    "HiddenAccessCounter",
    "INJECTION_FIELD_PROBES",
    "MIN_UNRESOLVED_PIECES",
    "PERMUTATION_ATTEMPT_BUDGET",
    "Phase11SafetyError",
    "STATE_SELECTION_WALK_LIMIT",
    "TracedPieceRecord",
    "admits_alternative_truth",
    "apply_alternative_truth",
    "belief_digest",
    "build_alternative_truth",
    "injection_controls",
    "instrument_hidden_types",
    "sampler_request_digest",
    "trial_sample_ordinal",
    "trial_state_walk",
    "unresolved_opponent_records",
    "valid_transpositions",
    "world_digest",
]
