"""Phase 8 Agent 3: `warmstart_example_v1` reconstruction and target audits.

Specification sources:

- `03_AGENT_3_TRAINING_EXAMPLES_AND_TARGETS.md` (replay reconstruction, target
  audits, teacher reproduction, hidden-permutation trials)
- `00_PHASE_8_SEQUENCE_AND_COMMON_CONTRACT.md` sections 14-16 (decision
  sampler, example schema, target definitions)

What lives here
---------------
The conversion of one committed corpus decision into one training example, and
the independent audits that prove the conversion right. Construction trusts
nothing it can re-derive: the engine replays the position, the engine
regenerates legality, the frozen `stratego.model.action_frame` tables convert
frames, and the frozen `dense_belief_target` builds belief labels. The stored
decision record contributes exactly two facts a replay cannot invent — which
action the teacher chose and which policy was acting — and both are checked
against the replayed state before an example is emitted.

Privilege boundary
------------------
An example carries the observation (the only model input), the loss inputs,
and identifying metadata. The privileged belief labels ride in their own
fields, produced *after* the public observation exists, exactly as
`ReconstructedDecision` already separates them. Nothing here ever writes a
label, a true type, an outcome, or an identity into the observation tensor;
:mod:`tests.information_security.test_warmstart_target_boundary` walks the
object graph to prove the separation holds at the batch boundary too.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ..engine.constants import (
    BLUE,
    NUM_PIECE_TYPES,
    NUM_SQUARES,
    PLAYER_NAMES,
    RED,
    opponent_of,
)
from ..engine.coordinates import from_perspective
from ..engine.legal_moves import legal_action_mask, legal_actions
from ..engine.observation import (
    CH_HIDDEN_OPPONENT_OCCUPANCY,
    CH_KNOWN_OPPONENT_IDENTITY,
    CH_LAKE_MASK,
    CH_OWN_IDENTITY,
    build_observation,
)
from ..engine.permutation import permute_hidden_identities
from ..engine.state import create_game
from ..engine.transition import apply_action
from ..evaluation.policy import build_policy_input, derive_decision_seed
from ..model.action_frame import (
    absolute_action_to_model,
    absolute_legal_actions_to_model,
    absolute_legal_mask_to_model,
    model_action_to_absolute,
    model_legal_mask_to_absolute,
)
from ..model.contract import (
    BELIEF_IGNORE_INDEX,
    OBSERVATION_SHAPE,
    VALUE_DRAW_INDEX,
    VALUE_LOSS_INDEX,
    VALUE_WIN_INDEX,
)
from .belief_targets import dense_belief_target
from .reconstruction import ReconstructedDecision, iter_reconstructed_decisions
from .rule_population import TeacherCache, teacher_by_token
from .trajectory import GameRecord
from .warmstart_contract import WARMSTART_EXAMPLE_VERSION, policy_weight
from .warmstart_seed import (
    MAX_DECISIONS_PER_GAME,
    parse_synthetic_game_id,
    selected_decision_indices,
)

#: Number of contiguous game-progress buckets used by the universe summaries:
#: quartiles of `decision_index / final_ply`, giving opening / early-middle /
#: late-middle / endgame coverage counts.
PROGRESS_BUCKETS = 4


class WarmstartExampleError(RuntimeError):
    """A training example could not be built or audited as contracted."""


#: `absolute square -> normalized square` per observer, built by inverting the
#: frozen `from_perspective` direction. The example builder goes through
#: `to_perspective` (inside `dense_belief_target`), so audits that use this
#: table exercise the opposite direction of the same frozen mapping.
_NORMALIZED_BY_ABSOLUTE = {
    observer: {
        from_perspective(normalized, observer): normalized
        for normalized in range(NUM_SQUARES)
    }
    for observer in (RED, BLUE)
}


# ---------------------------------------------------------------------------
# Target primitives
# ---------------------------------------------------------------------------


def value_target_index(terminal_result: str, acting_player: int) -> int:
    """WIN/DRAW/LOSS class index of one decision, acting-player perspective."""
    if terminal_result == "draw":
        return VALUE_DRAW_INDEX
    if terminal_result == "red_win":
        winner = RED
    elif terminal_result == "blue_win":
        winner = BLUE
    else:
        raise WarmstartExampleError(f"unknown terminal result {terminal_result!r}")
    return VALUE_WIN_INDEX if winner == acting_player else VALUE_LOSS_INDEX


def acting_policy_token(metadata: dict, acting_player: int) -> str:
    """`id@version` of whichever frozen teacher acted at one ply."""
    if acting_player == RED:
        return f"{metadata['red_policy_id']}@{metadata['red_policy_version']}"
    if acting_player == BLUE:
        return f"{metadata['blue_policy_id']}@{metadata['blue_policy_version']}"
    raise WarmstartExampleError(f"unknown acting player {acting_player!r}")


def acting_policy_id(metadata: dict, acting_player: int) -> str:
    return acting_policy_token(metadata, acting_player).split("@", 1)[0]


def acting_setup_family(metadata: dict, acting_player: int) -> str:
    """The acting side's primary setup family, from `setup_provenance_v1`."""
    side = PLAYER_NAMES[acting_player]
    return str(metadata["setup_provenance"][side]["primary_family_id"])


def progress_bucket(decision_index: int, total_decisions: int) -> int:
    """Which of the `PROGRESS_BUCKETS` quartiles one decision falls into."""
    if total_decisions <= 0:
        raise WarmstartExampleError("a game with no decisions has no progress")
    bucket = (decision_index * PROGRESS_BUCKETS) // total_decisions
    return min(bucket, PROGRESS_BUCKETS - 1)


# ---------------------------------------------------------------------------
# The training example
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarmstartExample:
    """One `warmstart_example_v1` training example.

    `observation` is the only model input. Every other field is a loss input
    or metadata, and the batch boundary in
    :mod:`stratego.training.warmstart_dataset` keeps them out of the tensor
    the model forward call receives.
    """

    observation: np.ndarray  # float32 [127, 10, 10]
    legal_mask: np.ndarray  # bool [10000], model frame
    acting_player: int
    policy_action_abs: int
    policy_action_model: int
    policy_weight: float
    value_target: int
    belief_target: np.ndarray  # int64 [100]
    belief_mask: np.ndarray  # bool [100]
    game_id: str
    decision_index: int
    source_policy_id: str
    corpus_split: str

    @property
    def key(self) -> tuple:
        return (self.game_id, self.decision_index)

    @property
    def policy_supervised(self) -> bool:
        return self.policy_weight > 0.0

    @property
    def supervised_belief_pieces(self) -> int:
        return int(self.belief_mask.sum())


def build_example(
    record: GameRecord, metadata: dict, rebuilt: ReconstructedDecision
) -> WarmstartExample:
    """One example from one reconstructed decision, cross-checked as it is built.

    The engine-replayed legal set must equal the stored one and must contain
    the stored action; a disagreement means the corpus and the replay no longer
    describe the same game, which is a stop condition rather than a repair.
    """
    ply = rebuilt.ply
    decision = record.decisions[ply]
    if decision.ply != ply:
        raise WarmstartExampleError(
            f"{record.game_id}: decision record at index {ply} names ply {decision.ply}"
        )
    if tuple(rebuilt.legal_action_ids) != tuple(decision.legal_action_ids):
        raise WarmstartExampleError(
            f"{record.game_id} ply {ply}: replayed legal actions differ from the record"
        )
    if rebuilt.acting_player != decision.acting_player:
        raise WarmstartExampleError(
            f"{record.game_id} ply {ply}: replayed acting player differs from the record"
        )
    action = int(decision.selected_action_id)
    if action not in rebuilt.legal_action_ids:
        raise WarmstartExampleError(
            f"{record.game_id} ply {ply}: the recorded action is not legal on replay"
        )
    if rebuilt.legal_mask is None:
        raise WarmstartExampleError(
            "build_example needs dense_mask=True reconstruction; the model-frame "
            "mask is converted from the engine's dense mask"
        )

    actor = int(rebuilt.acting_player)
    labels, mask = dense_belief_target(rebuilt.state, actor)
    observation = np.ascontiguousarray(rebuilt.observation, dtype=np.float32)
    model_mask = absolute_legal_mask_to_model(rebuilt.legal_mask, actor).astype(bool)

    return WarmstartExample(
        observation=observation,
        legal_mask=model_mask,
        acting_player=actor,
        policy_action_abs=action,
        policy_action_model=absolute_action_to_model(action, actor),
        policy_weight=float(
            policy_weight(acting_policy_id(metadata, actor))
        ),
        value_target=value_target_index(record.terminal_result, actor),
        belief_target=labels,
        belief_mask=mask,
        game_id=record.game_id,
        decision_index=ply,
        source_policy_id=acting_policy_id(metadata, actor),
        corpus_split=str(metadata["corpus_split"]),
    )


def examples_for_game(
    record: GameRecord,
    metadata: dict,
    indices: "tuple[int, ...] | None" = None,
):
    """Yield the examples of one game's selected decisions, ascending by ply.

    `indices` defaults to the frozen `warmstart_decision_sampler_v1` selection.
    Sequential reconstruction advances one state instead of restoring a
    snapshot per position; the result is identical to independent random
    access, which `tests/training/test_warmstart_examples.py` asserts.
    """
    if metadata["synthetic_game_id"] != record.game_id:
        raise WarmstartExampleError(
            f"metadata for {metadata['synthetic_game_id']!r} was paired with "
            f"record {record.game_id!r}"
        )
    if indices is None:
        indices = selected_decision_indices(record.game_id, len(record.decisions))
    for rebuilt in iter_reconstructed_decisions(
        record,
        indices,
        dense_mask=True,
        include_public_knowledge=False,
        copy_state=False,
    ):
        # copy_state=False shares the advancing state, so the example must be
        # fully built (belief labels included) before the next ply is pulled.
        yield build_example(record, metadata, rebuilt)


# ---------------------------------------------------------------------------
# Static (decode-only) audit of one game's selected decisions
# ---------------------------------------------------------------------------


def audit_game_static(record: GameRecord, metadata: dict, commit_total_decisions: int) -> dict:
    """Audit every selected decision of one game without replaying it.

    Everything checked here is a pure function of the stored record, the
    stored metadata and the frozen contracts: the sampler contract, the value
    mapping, the frozen supervision weights, action membership in the stored
    (already engine-verified) legal set, and the exactness of the frozen
    action-frame conversion. Replay-based checks live in
    :func:`audit_example`.
    """
    problems: list[str] = []
    game_id = record.game_id
    total = len(record.decisions)

    from_commit = selected_decision_indices(game_id, commit_total_decisions)
    indices = selected_decision_indices(game_id, total)
    if from_commit != indices:
        problems.append("commit index and decoded record disagree on the selection")
    if total > MAX_DECISIONS_PER_GAME:
        if len(indices) != MAX_DECISIONS_PER_GAME:
            problems.append(f"{len(indices)} selected decisions for a long game")
    elif indices != tuple(range(total)):
        problems.append("a short game did not select every decision")
    if any(b <= a for a, b in zip(indices, indices[1:])):
        problems.append("selected indices are not strictly increasing")

    checked = 0
    value_counts = [0, 0, 0]
    weight_by_policy: dict[str, float] = {}
    for index in indices:
        decision = record.decisions[index]
        if decision.ply != index:
            problems.append(f"ply {index}: decision record names ply {decision.ply}")
            continue
        actor = int(decision.acting_player)
        action = int(decision.selected_action_id)
        legal = decision.legal_action_ids
        if action not in legal:
            problems.append(f"ply {index}: recorded action is outside the stored legal set")
        model_action = absolute_action_to_model(action, actor)
        if model_action_to_absolute(model_action, actor) != action:
            problems.append(f"ply {index}: model-frame conversion does not invert")
        model_legal = absolute_legal_actions_to_model(legal, actor)
        if len(model_legal) != len(legal):
            problems.append(f"ply {index}: converted legal set changed size")
        if model_action not in model_legal:
            problems.append(f"ply {index}: converted action left the converted legal set")

        token = acting_policy_token(metadata, actor)
        teacher = teacher_by_token(token)
        weight = policy_weight(teacher.policy_id)
        if weight != teacher.policy_weight:
            problems.append(f"ply {index}: contract and roster disagree on the weight")
        expected_weight = float(
            metadata["red_policy_weight" if actor == RED else "blue_policy_weight"]
        )
        if weight != expected_weight:
            problems.append(f"ply {index}: metadata weight {expected_weight} is not frozen")
        if teacher.role in ("tier_random", "stress") and weight != 0.0:
            problems.append(f"ply {index}: {teacher.role} decision carries policy weight")
        if teacher.role == "tier_strategic" and weight != 1.0:
            problems.append(f"ply {index}: strategic weight is not 1.0")
        if teacher.role == "tier_tactical" and weight != 1.0:
            problems.append(f"ply {index}: tactical weight is not 1.0")
        if teacher.role == "tier_basic" and weight != 0.5:
            problems.append(f"ply {index}: basic weight is not 0.5")
        weight_by_policy[teacher.policy_id] = weight

        value_counts[value_target_index(record.terminal_result, actor)] += 1
        checked += 1

    return {
        "game_id": game_id,
        "selected": len(indices),
        "checked": checked,
        "value_counts": value_counts,
        "weight_by_policy": weight_by_policy,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Replay-based audit of one built example
# ---------------------------------------------------------------------------


def audit_example(
    example: WarmstartExample,
    record: GameRecord,
    metadata: dict,
    rebuilt: ReconstructedDecision,
) -> list:
    """Every disagreement between one example and its independently re-derived
    ground truth.

    The recomputation deliberately avoids the code paths
    :func:`build_example` used wherever an independent route exists: the
    model-frame legal set is rebuilt through the *list* conversion and
    compared against the example's *mask* conversion, the mask is inverted
    back into the engine frame, and the belief labels are re-derived square by
    square from the privileged piece records rather than through
    `dense_belief_target`.
    """
    problems: list[str] = []
    state = rebuilt.state
    actor = int(rebuilt.acting_player)
    ply = rebuilt.ply
    prefix = f"{record.game_id} ply {ply}"

    # -- identity ----------------------------------------------------------
    identity = parse_synthetic_game_id(record.game_id)
    if example.corpus_split != identity["split"]:
        problems.append(f"{prefix}: example split {example.corpus_split!r}")
    if example.game_id != record.game_id or example.decision_index != ply:
        problems.append(f"{prefix}: example identity fields disagree")
    if example.acting_player != state.acting_player:
        problems.append(f"{prefix}: acting player disagrees with the replayed state")
    if state.total_moves != ply:
        problems.append(f"{prefix}: replayed state is at ply {state.total_moves}")

    # -- policy target and frames -----------------------------------------
    decision = record.decisions[ply]
    live_legal = tuple(legal_actions(state))
    if live_legal != tuple(decision.legal_action_ids):
        problems.append(f"{prefix}: engine legal set differs from the stored one")
    if example.policy_action_abs != decision.selected_action_id:
        problems.append(f"{prefix}: example does not carry the recorded action")
    if example.policy_action_abs not in live_legal:
        problems.append(f"{prefix}: recorded action is illegal in the replayed state")
    if model_action_to_absolute(example.policy_action_model, actor) != (
        example.policy_action_abs
    ):
        problems.append(f"{prefix}: model-frame action does not invert to the absolute one")

    model_legal = absolute_legal_actions_to_model(live_legal, actor)
    mask_indices = tuple(int(i) for i in np.flatnonzero(example.legal_mask))
    if mask_indices != model_legal:
        problems.append(f"{prefix}: model mask disagrees with the converted legal list")
    if int(example.legal_mask.sum()) != len(live_legal):
        problems.append(f"{prefix}: model mask population differs from the legal count")
    if not example.legal_mask[example.policy_action_model]:
        problems.append(f"{prefix}: model mask excludes the target action")
    engine_mask = legal_action_mask(state, list(live_legal))
    round_trip = model_legal_mask_to_absolute(
        example.legal_mask.astype(np.uint8), actor
    )
    if not np.array_equal(round_trip, engine_mask):
        problems.append(f"{prefix}: model mask does not invert to the engine mask")

    # -- supervision weight ------------------------------------------------
    teacher = teacher_by_token(acting_policy_token(metadata, actor))
    if example.source_policy_id != teacher.policy_id:
        problems.append(f"{prefix}: source policy id disagrees with the metadata")
    if example.policy_weight != teacher.policy_weight:
        problems.append(f"{prefix}: example weight is not the frozen teacher weight")

    # -- value -------------------------------------------------------------
    if example.value_target != value_target_index(record.terminal_result, actor):
        problems.append(f"{prefix}: value target disagrees with the recomputed mapping")

    # -- belief, square by square from privileged piece records ------------
    # The builder normalizes squares with `to_perspective`; the audit inverts
    # `from_perspective` instead, so the two directions of the frozen mapping
    # must agree for the comparison to pass.
    inverse = _NORMALIZED_BY_ABSOLUTE[actor]
    expected: dict[int, int] = {}
    for piece in state.pieces:
        if piece.owner == actor or not piece.alive or piece.known_to(actor):
            continue
        expected[inverse[piece.current_square]] = piece.true_type
    mask_squares = {int(square) for square in np.flatnonzero(example.belief_mask)}
    if mask_squares != set(expected):
        problems.append(f"{prefix}: belief mask does not cover exactly the hidden pieces")
    for square in range(NUM_SQUARES):
        label = int(example.belief_target[square])
        if square in expected:
            if label != expected[square]:
                problems.append(f"{prefix}: belief label at square {square} is wrong")
            if not 0 <= label < NUM_PIECE_TYPES:
                problems.append(f"{prefix}: belief label out of range at {square}")
        elif label != BELIEF_IGNORE_INDEX:
            problems.append(f"{prefix}: unsupervised square {square} carries a label")
    if not np.array_equal(example.belief_mask, example.belief_target != BELIEF_IGNORE_INDEX):
        problems.append(f"{prefix}: belief mask and labels disagree")

    # -- observation cross-checks ------------------------------------------
    observation = example.observation
    if observation.shape != OBSERVATION_SHAPE or observation.dtype != np.float32:
        problems.append(f"{prefix}: observation shape/dtype violates the contract")
    planes = observation.reshape(observation.shape[0], NUM_SQUARES)
    hidden_plane = planes[CH_HIDDEN_OPPONENT_OCCUPANCY] > 0.5
    if not np.array_equal(hidden_plane, example.belief_mask):
        problems.append(f"{prefix}: hidden-occupancy channel disagrees with the belief mask")
    known_planes = planes[
        CH_KNOWN_OPPONENT_IDENTITY : CH_KNOWN_OPPONENT_IDENTITY + NUM_PIECE_TYPES
    ].sum(axis=0)
    if np.any(known_planes[example.belief_mask] > 0.0):
        problems.append(f"{prefix}: a supervised square holds a known opponent piece")
    own_planes = planes[CH_OWN_IDENTITY : CH_OWN_IDENTITY + NUM_PIECE_TYPES].sum(axis=0)
    if np.any(own_planes[example.belief_mask] > 0.0):
        problems.append(f"{prefix}: a supervised square holds an own piece")
    if np.any(planes[CH_LAKE_MASK][example.belief_mask] > 0.0):
        problems.append(f"{prefix}: a supervised square is a lake")

    return problems


# ---------------------------------------------------------------------------
# Teacher-decision reproduction
# ---------------------------------------------------------------------------


def reproduce_teacher_decisions(
    record: GameRecord,
    metadata: dict,
    plies: "tuple[int, ...]",
    teachers: "TeacherCache | None" = None,
) -> dict:
    """Re-invoke the recorded rule policy at the given plies and compare.

    The game is replayed from its stored setups and actions — the same frozen
    path the generator used — and at each requested ply the acting side's
    live policy receives exactly the request `play_corpus_game` built: its
    recorded match seed from the metadata, its declared requirements, and the
    engine's regenerated legal set. The policy must reproduce the recorded
    action; anything else is a mismatch, never a substitution.
    """
    cache = TeacherCache() if teachers is None else teachers
    wanted = sorted(set(int(ply) for ply in plies))
    seeds = {
        RED: int(metadata["red_policy_seed"]),
        BLUE: int(metadata["blue_policy_seed"]),
    }
    tokens = {
        RED: acting_policy_token(metadata, RED),
        BLUE: acting_policy_token(metadata, BLUE),
    }
    state = create_game(
        record.red_setup, record.blue_setup, rules=record.rules(), game_id=record.game_id
    )
    reproduced = 0
    mismatches: list[str] = []
    cursor = 0
    for ply, action_id in enumerate(record.actions):
        if cursor < len(wanted) and wanted[cursor] == ply:
            cursor += 1
            actor = state.acting_player
            decision = record.decisions[ply]
            if decision.acting_player != actor:
                mismatches.append(f"ply {ply}: replayed acting player differs")
            policy = cache.get(tokens[actor])
            legal = legal_actions(state)
            request = build_policy_input(
                state,
                policy=policy.ref,
                policy_seed=seeds[actor],
                requirements=policy.requirements,
                game_id=record.game_id,
                legal=legal,
            )
            if request.decision_seed != derive_decision_seed(seeds[actor], ply):
                mismatches.append(f"ply {ply}: decision seed derivation drifted")
            result = policy.decide_checked(request)
            if int(result.selected_action_id) != int(decision.selected_action_id):
                mismatches.append(
                    f"ply {ply}: {tokens[actor]} chose {result.selected_action_id}, "
                    f"the record stores {decision.selected_action_id}"
                )
            reproduced += 1
        if cursor >= len(wanted):
            break
        apply_action(state, action_id)
    if cursor < len(wanted):
        mismatches.append(f"{len(wanted) - cursor} requested plies were never reached")
    return {
        "game_id": record.game_id,
        "requested": len(wanted),
        "reproduced": reproduced,
        "mismatches": mismatches,
    }


# ---------------------------------------------------------------------------
# Hidden-permutation paired trials
# ---------------------------------------------------------------------------


def hidden_permutation_trial(
    record: GameRecord,
    metadata: dict,
    rebuilt: ReconstructedDecision,
    rng: random.Random,
) -> dict:
    """One paired anti-leak trial at one reconstructed decision.

    The privileged state is cloned with its unresolved opponent identities
    permuted (`stratego.engine.permutation`, the frozen Phase 2 machinery) and
    a second example is built from the clone. Everything the model or the
    policy loss can see must be identical; the privileged belief labels must
    change exactly when the permutation changed an identity.
    """
    base = build_example(record, metadata, rebuilt)
    actor = int(rebuilt.acting_player)
    state = rebuilt.state
    hidden_before = [
        piece.true_type
        for piece in state.pieces
        if piece.owner == opponent_of(actor) and piece.alive and not piece.known_to(actor)
    ]

    permuted, info = permute_hidden_identities(state, actor, rng)
    observation = np.ascontiguousarray(build_observation(permuted, actor), dtype=np.float32)
    legal = tuple(legal_actions(permuted))
    model_mask = absolute_legal_mask_to_model(
        legal_action_mask(permuted, list(legal)), actor
    ).astype(bool)
    labels, mask = dense_belief_target(permuted, actor)
    hidden_after = [
        piece.true_type
        for piece in permuted.pieces
        if piece.owner == opponent_of(actor) and piece.alive and not piece.known_to(actor)
    ]

    mismatches: list[str] = []
    if not np.array_equal(observation, base.observation):
        mismatches.append("observation changed under a hidden permutation")
    if legal != tuple(rebuilt.legal_action_ids):
        mismatches.append("legal actions changed under a hidden permutation")
    if not np.array_equal(model_mask, base.legal_mask):
        mismatches.append("model legal mask changed under a hidden permutation")
    if absolute_action_to_model(base.policy_action_abs, actor) != base.policy_action_model:
        mismatches.append("policy action conversion changed")
    if not np.array_equal(mask, base.belief_mask):
        mismatches.append("belief mask changed under a hidden permutation")

    labels_differ = not np.array_equal(labels, base.belief_target)
    truth_differs = hidden_after != hidden_before
    control_ok = True
    if info["changed"] and not (labels_differ and truth_differs):
        control_ok = False
    if not info["changed"] and (labels_differ or truth_differs):
        control_ok = False

    return {
        "valid": bool(info["valid"]),
        "changed": bool(info["changed"]),
        "hidden_pieces": int(info["hidden_pieces"]),
        "labels_differ": labels_differ,
        "truth_differs": truth_differs,
        "control_ok": control_ok,
        "mismatches": mismatches,
    }


__all__ = [
    "PROGRESS_BUCKETS",
    "WARMSTART_EXAMPLE_VERSION",
    "WarmstartExample",
    "WarmstartExampleError",
    "acting_policy_id",
    "acting_policy_token",
    "acting_setup_family",
    "audit_example",
    "audit_game_static",
    "build_example",
    "examples_for_game",
    "hidden_permutation_trial",
    "progress_bucket",
    "reproduce_teacher_decisions",
    "value_target_index",
]
