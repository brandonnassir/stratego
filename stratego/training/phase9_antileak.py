"""Phase 9 Agent 4: the information-security boundary of an RL example.

Specification sources:

- `04_AGENT_4_RL_TARGETS_ADVANTAGES_AND_ANTILEAK.md` ("Anti-leak trials",
  "Belief target", "Dataset/example contract")
- `00_PHASE_9_SEQUENCE_AND_COMMON_CONTRACT.md` (observer safety as a hard
  Phase 9 gate)
- The accepted Phase 8 machinery this continues:
  `stratego.engine.permutation.permute_hidden_identities` (Phase 2) and
  `stratego.training.warmstart_examples.hidden_permutation_trial` (Phase 8
  Agent 3), whose paired-trial shape is reused rather than reinvented.

What a trial proves
-------------------
Take one learner decision. Reassign the true identities of every opponent
piece the learner may not legally know, keeping the assignment consistent with
everything already public. Rebuild the *whole example* — not a hand-picked
subset of it — from the counterfactual state through the same production
builder. Then:

```text
must be bitwise identical   observation, legal actions, model action mapping,
                            legal mask, learner designation, every
                            public/behavior-derived PPO input, belief mask
must change exactly when    belief labels, privileged hidden truth
the assignment changed
```

The first list is everything a network or a policy loss can see. If any entry
of it moved, a hidden identity reached it. The second list is the supervision
signal, which is *supposed* to depend on the truth — a belief label that did
not move when the truth did would mean the labels are not labels.

Why the whole example is rebuilt
--------------------------------
A trial that compares only the observation proves only that the observation is
clean. Phase 9 adds advantages, standardized advantages, PPO eligibility,
W/D/L targets and behavior probabilities to the object a trainer consumes, and
each is a new place privileged truth could enter. Rebuilding through
:func:`stratego.training.phase9_targets.build_example` means every field is
covered by construction, including any field added later.

Positive controls
-----------------
An audit that cannot fail is not evidence. :func:`positive_controls` plants
each of the five failures the assignment names — a privileged identity in the
observation, privileged metadata on the model input, a wrong action frame, a
wrong value perspective, a wrong learner-control side — and requires the audit
to report each one.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np

from ..engine.constants import NUM_PIECE_TYPES, NUM_SQUARES, opponent_of
from ..engine.legal_moves import legal_action_mask, legal_actions
from ..engine.observation import CH_HIDDEN_OPPONENT_OCCUPANCY, build_observation
from ..engine.permutation import permute_hidden_identities
from ..model.action_frame import absolute_action_to_model
from ..model.contract import OBSERVATION_SHAPE
from .phase9_targets import (
    MODEL_INPUT_FIELDS,
    TARGET_ABS_TOLERANCE,
    Phase9RLExample,
    audit_example,
    build_example,
    terminal_outcome,
)

#: The Phase 9 anti-leak audit version. Recorded in the artifact so a later
#: agent can tell which comparison surface produced a mismatch count.
PHASE9_ANTILEAK_VERSION = "phase9_antileak_v1"

#: Example fields that must be bitwise invariant under any valid reassignment
#: of unresolved opponent identities. Everything a backbone, a mask, a PPO
#: ratio, a KL term or a learner designation can read.
INVARIANT_FIELDS = (
    "observation",
    "legal_mask",
    "sampled_action_abs",
    "sampled_action_model",
    "behavior_action_probability",
    "behavior_action_logprob",
    "behavior_legal_actions",
    "behavior_legal_probabilities",
    "advantage",
    "standardized_advantage",
    "ppo_eligible",
    "wdl_target",
    "belief_mask",
    "game_id",
    "decision_index",
    "learner_side",
    "behavior_checkpoint_sha256",
    "rollout_id",
)

#: Example fields that carry privileged truth and are therefore *expected* to
#: move with the hidden assignment.
PRIVILEGED_FIELDS = ("belief_target",)

#: The five failures the assignment requires the audit to detect.
POSITIVE_CONTROL_NAMES = (
    "privileged_identity_in_observation",
    "privileged_metadata_on_model_input",
    "wrong_action_frame",
    "wrong_value_perspective",
    "wrong_learner_control_side",
)


class Phase9AntileakError(RuntimeError):
    """An anti-leak trial could not be run as contracted."""


@dataclass
class PermutedDecision:
    """A counterfactual decision, shaped like a `ReconstructedDecision`.

    Only the attributes :func:`build_example` reads are carried, and they are
    rebuilt from the permuted state through the frozen engine rather than
    copied from the original — a copied legal set would silently pass the very
    check the trial exists to make.
    """

    ply: int
    acting_player: int
    state: object
    observation: np.ndarray
    legal_action_ids: tuple
    legal_mask: np.ndarray


def rebuild_from_state(state, ply: int) -> PermutedDecision:
    """Everything one example needs, re-derived from one privileged state."""
    actor = int(state.acting_player)
    legal = tuple(legal_actions(state))
    return PermutedDecision(
        ply=int(ply),
        acting_player=actor,
        state=state,
        observation=build_observation(state, actor),
        legal_action_ids=legal,
        legal_mask=legal_action_mask(state, list(legal)),
    )


def hidden_truth(state, observer: int) -> tuple:
    """The unresolved opponent types, in piece order. Privileged; audit only."""
    return tuple(
        piece.true_type
        for piece in state.pieces
        if piece.owner == opponent_of(int(observer))
        and piece.alive
        and not piece.known_to(int(observer))
    )


def _field_differs(left, right) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return not np.array_equal(np.asarray(left), np.asarray(right))
    return left != right


def hidden_permutation_trial(
    record,
    metadata: dict,
    rebuilt,
    sequence,
    statistics,
    rng,
) -> dict:
    """One paired anti-leak trial at one reconstructed learner decision.

    Returns the trial's bookkeeping and its mismatches. `valid` reports whether
    the engine produced a consistent counterfactual assignment at all — a
    position with fewer than two unresolved opponent pieces has nothing to
    permute, and such trials are counted separately rather than passed off as
    evidence.
    """
    base = build_example(record, metadata, rebuilt, sequence, statistics)
    actor = int(rebuilt.acting_player)
    before = hidden_truth(rebuilt.state, actor)

    permuted_state, info = permute_hidden_identities(rebuilt.state, actor, rng)
    after = hidden_truth(permuted_state, actor)
    mismatches: list[str] = []
    try:
        counterfactual = build_example(
            record,
            metadata,
            rebuild_from_state(permuted_state, rebuilt.ply),
            sequence,
            statistics,
        )
    except Exception as error:  # noqa: BLE001 - a refused rebuild is a finding
        # The builder cross-checks legality and the acting side against the
        # stored record. Neither may depend on a hidden identity, so a refusal
        # here is exactly the leak the trial is looking for — reported, not
        # raised, so one bad decision cannot abort a 25,000-trial run.
        return {
            "game_id": record.game_id,
            "ply": int(rebuilt.ply),
            "learner_side": actor,
            "valid": bool(info["valid"]),
            "changed": bool(info["changed"]),
            "hidden_pieces": int(info["hidden_pieces"]),
            "labels_differ": False,
            "truth_differs": after != before,
            "control_ok": False,
            "mismatches": [
                f"the example could not be rebuilt under a hidden permutation: {error}"
            ],
        }

    for field in INVARIANT_FIELDS:
        if _field_differs(getattr(base, field), getattr(counterfactual, field)):
            mismatches.append(f"{field} changed under a hidden-identity permutation")

    labels_differ = _field_differs(base.belief_target, counterfactual.belief_target)
    truth_differs = after != before
    control_ok = True
    if info["changed"] and not (labels_differ and truth_differs):
        control_ok = False
        mismatches.append(
            "the hidden assignment changed but the privileged belief labels did not"
        )
    if not info["changed"] and (labels_differ or truth_differs):
        control_ok = False
        mismatches.append(
            "the hidden assignment did not change but privileged truth did"
        )
    return {
        "game_id": record.game_id,
        "ply": int(rebuilt.ply),
        "learner_side": actor,
        "valid": bool(info["valid"]),
        "changed": bool(info["changed"]),
        "hidden_pieces": int(info["hidden_pieces"]),
        "labels_differ": bool(labels_differ),
        "truth_differs": bool(truth_differs),
        "control_ok": bool(control_ok),
        "mismatches": mismatches,
    }


# ---------------------------------------------------------------------------
# The model-input boundary
# ---------------------------------------------------------------------------


def audit_model_input(payload: dict) -> list:
    """Every way a model-input mapping could carry more than an observation.

    Agent 5 hands the backbone whatever `model_input_fields_only` (or a batch's
    `model_input`) returns, so this is the last place before a tensor where a
    privileged field can be caught.
    """
    problems: list[str] = []
    extra = sorted(set(payload) - set(MODEL_INPUT_FIELDS))
    if extra:
        problems.append(f"model input carries non-observation fields: {extra}")
    missing = sorted(set(MODEL_INPUT_FIELDS) - set(payload))
    if missing:
        problems.append(f"model input is missing {missing}")
    observation = payload.get("observation")
    if observation is not None:
        array = np.asarray(observation)
        if array.shape not in (OBSERVATION_SHAPE, (array.shape[0],) + OBSERVATION_SHAPE):
            problems.append(f"model input observation has shape {array.shape}")
        if array.dtype != np.float32:
            problems.append(f"model input observation has dtype {array.dtype}")
    return problems


def audit_example_object_graph(example: Phase9RLExample) -> list:
    """Refuse an example that grew a privileged field.

    The belief label is privileged by design and lives in a named target field;
    anything *else* that could name a true type, a setup or a terminal result
    would be a new leak surface. Walking the field names rather than the values
    keeps this honest as the schema evolves.
    """
    from .phase9_targets import FORBIDDEN_EXAMPLE_FIELDS

    problems: list[str] = []
    present = set(vars(example))
    forbidden = sorted(present & set(FORBIDDEN_EXAMPLE_FIELDS))
    if forbidden:
        problems.append(f"example carries privileged fields: {forbidden}")
    return problems


# ---------------------------------------------------------------------------
# Positive controls
# ---------------------------------------------------------------------------


def _replace(example: Phase9RLExample, **changes) -> Phase9RLExample:
    """A copy of one example with fields replaced. Test/control use only."""
    return replace(example, **changes)


def plant_identity_in_observation(example: Phase9RLExample, state) -> Phase9RLExample:
    """Control 1: write privileged hidden types into the model input.

    The unresolved opponent types are written onto the hidden-occupancy plane,
    which is exactly the leak a permutation trial is built to notice: the
    channel already marks *where* the hidden pieces are, so encoding *what*
    they are there is the most plausible accidental version of this bug.

    A position with no unresolved opponent piece has nothing to plant, and a
    control that plants nothing must never be counted as fired — so it raises
    instead of returning an unchanged example.
    """
    observation = np.array(example.observation, dtype=np.float32, copy=True)
    planes = observation.reshape(observation.shape[0], NUM_SQUARES)
    actor = int(example.learner_side)
    from ..engine.coordinates import to_perspective

    for piece in state.pieces:
        if piece.owner == actor or not piece.alive or piece.known_to(actor):
            continue
        square = to_perspective(int(piece.current_square), actor)
        planes[CH_HIDDEN_OPPONENT_OCCUPANCY][square] = 1.0 + (
            int(piece.true_type) + 1
        ) / (NUM_PIECE_TYPES + 1)
    if np.array_equal(observation, example.observation):
        raise Phase9AntileakError(
            "no unresolved opponent identity exists at this decision, so the "
            "planted-observation control would be vacuous"
        )
    return _replace(example, observation=observation)


def attach_privileged_metadata_to_model_input(example: Phase9RLExample) -> dict:
    """Control 2: a model-input mapping that smuggles the belief labels in."""
    payload = {field: getattr(example, field) for field in MODEL_INPUT_FIELDS}
    payload["belief_target"] = example.belief_target
    return payload


def use_wrong_action_frame(example: Phase9RLExample) -> Phase9RLExample:
    """Control 3: convert the sampled action in the opponent's frame.

    A centrally symmetric action can map to itself under the opposite
    perspective. That is a real property of the frame, not a leak, and such a
    decision simply cannot host this control — so it raises rather than plant a
    change that is no change.
    """
    wrong = absolute_action_to_model(
        int(example.sampled_action_abs), opponent_of(int(example.learner_side))
    )
    if int(wrong) == int(example.sampled_action_model):
        raise Phase9AntileakError(
            "the opposite perspective maps this action to the same model action, "
            "so the wrong-frame control would be vacuous"
        )
    return _replace(example, sampled_action_model=int(wrong))


def use_wrong_value_perspective(example: Phase9RLExample, record) -> Phase9RLExample:
    """Control 4: build the W/D/L target from the opponent's outcome.

    The subtlety this catches is the one that would never crash: a reversed
    perspective still produces a valid simplex, still sums to one, and is still
    finite. Only a recomputation from the learner's own final perspective
    notices.
    """
    outcome = terminal_outcome(record.terminal_result, opponent_of(int(example.learner_side)))
    one_hot = {"win": (1.0, 0.0, 0.0), "draw": (0.0, 1.0, 0.0), "loss": (0.0, 0.0, 1.0)}
    planted = one_hot[outcome]
    if all(
        abs(float(left) - float(right)) <= TARGET_ABS_TOLERANCE
        for left, right in zip(planted, example.wdl_target)
    ):
        raise Phase9AntileakError(
            "the opponent-perspective target coincides with this decision's own "
            "target (a drawn terminal decision), so the control would be vacuous"
        )
    return _replace(example, wdl_target=planted)


def use_wrong_learner_control_side(example: Phase9RLExample) -> Phase9RLExample:
    """Control 5: attribute the decision to the side that did not act."""
    return _replace(example, learner_side=opponent_of(int(example.learner_side)))


def positive_controls(record, metadata: dict, rebuilt, sequence, statistics) -> list:
    """Plant each named failure and require the audit to report it.

    Every control is applied to a *real* example of a real sealed decision, so
    a control that fires proves the audit works on the same objects the
    exhaustive audit consumed — not on a synthetic fixture that happens to be
    easier to detect.
    """
    example = build_example(record, metadata, rebuilt, sequence, statistics)
    clean = audit_example(example, record, metadata, rebuilt, sequence, statistics)
    if clean:
        raise Phase9AntileakError(
            f"the control decision {record.game_id} ply {rebuilt.ply} does not audit "
            f"clean before any control is planted: {clean[:3]}"
        )
    if audit_model_input({field: getattr(example, field) for field in MODEL_INPUT_FIELDS}):
        raise Phase9AntileakError("a clean model input failed the boundary audit")

    results: list[dict] = []

    planted = plant_identity_in_observation(example, rebuilt.state)
    results.append(
        {
            "control": "privileged_identity_in_observation",
            "problems": audit_example(planted, record, metadata, rebuilt, sequence, statistics),
        }
    )

    results.append(
        {
            "control": "privileged_metadata_on_model_input",
            "problems": audit_model_input(
                attach_privileged_metadata_to_model_input(example)
            ),
        }
    )

    framed = use_wrong_action_frame(example)
    results.append(
        {
            "control": "wrong_action_frame",
            "problems": audit_example(framed, record, metadata, rebuilt, sequence, statistics),
        }
    )

    valued = use_wrong_value_perspective(example, record)
    results.append(
        {
            "control": "wrong_value_perspective",
            "problems": audit_example(valued, record, metadata, rebuilt, sequence, statistics),
        }
    )

    sided = use_wrong_learner_control_side(example)
    results.append(
        {
            "control": "wrong_learner_control_side",
            "problems": audit_example(sided, record, metadata, rebuilt, sequence, statistics),
        }
    )

    for result in results:
        result["fired"] = bool(result["problems"])
        result["problems"] = result["problems"][:4]
    return results


def leaking_observation_builder(state, observer: int) -> np.ndarray:
    """A deliberately leaking observation builder, for the collector probe.

    Kept here beside the controls it belongs with: the Phase 9 observer probe
    accepts a builder precisely so an audit can be pointed at one that leaks
    and shown to fail.
    """
    observation = np.array(build_observation(state, observer), dtype=np.float32, copy=True)
    planes = observation.reshape(observation.shape[0], NUM_SQUARES)
    counterfactual = copy.deepcopy(state)
    from ..engine.coordinates import to_perspective

    for piece in counterfactual.pieces:
        if piece.owner == int(observer) or not piece.alive or piece.known_to(int(observer)):
            continue
        planes[CH_HIDDEN_OPPONENT_OCCUPANCY][
            to_perspective(int(piece.current_square), int(observer))
        ] = float(piece.true_type) + 2.0
    return observation


__all__ = [
    "INVARIANT_FIELDS",
    "PHASE9_ANTILEAK_VERSION",
    "POSITIVE_CONTROL_NAMES",
    "PRIVILEGED_FIELDS",
    "PermutedDecision",
    "Phase9AntileakError",
    "attach_privileged_metadata_to_model_input",
    "audit_example_object_graph",
    "audit_model_input",
    "hidden_permutation_trial",
    "hidden_truth",
    "leaking_observation_builder",
    "plant_identity_in_observation",
    "positive_controls",
    "rebuild_from_state",
    "use_wrong_action_frame",
    "use_wrong_learner_control_side",
    "use_wrong_value_perspective",
]
