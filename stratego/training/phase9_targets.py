"""Phase 9 Agent 4: RL example semantics — same-player targets and the dataset.

Specification sources:

- `04_AGENT_4_RL_TARGETS_ADVANTAGES_AND_ANTILEAK.md` (same-player extraction,
  scalar behavior value, advantage, WDL lambda target, advantage filter,
  belief target, dataset/example contract, handoff to Agent 5)
- `00_PHASE_9_SEQUENCE_AND_COMMON_CONTRACT.md` ("Same-player temporal
  targets", "Advantage filtering", "Learner-control semantics", the
  `phase9_train_order_v1` shuffle)
- Agent 1's `phase9_contract`: `gamma`, `lambda_A`, `lambda_V`, the filter
  quantile/floor, the standardization rule and the train-order definition are
  all frozen there. This module *consumes* those constants; it never restates
  one, so a tuned value would have to be tuned in the frozen contract where the
  Agent 1 digest would catch it.

What lives here
---------------
The conversion of one sealed rollout into trainable Phase 9 examples, and
nothing else. No optimizer, no loss, no gradient, no PPO objective: Agent 5
implements optimization and receives the deterministic iterator, the train
order, the cursor, and the targets built here.

Two passes, and why
-------------------
`tau = max(Q_0.75(|A|), 0.01)` is a *per sealed iteration* statistic, so PPO
eligibility of one decision depends on every other learner decision in the same
rollout. That forces two passes, and the split is the useful one:

```text
pass 1   decisions + metadata only          -> sequences, advantages, tau,
         (no engine replay, no observation)    standardization, eligibility
pass 2   replay + observation + belief      -> the examples themselves
```

Pass 1 is cheap because the same-player targets are pure functions of what the
collector already stored: the learner's own W/D/L predictions and the terminal
result. Pass 2 is the expensive one and is streamed, so an iteration is never
held in memory as tensors.

The privilege boundary
----------------------
:class:`Phase9RLExample` carries exactly one model input — `observation` — and
:data:`MODEL_INPUT_FIELDS` names it. The belief labels are targets and ride in
their own fields, produced *after* the observation exists, exactly as
`ReconstructedDecision` and Phase 8's `WarmstartExample` already separate them.
The legal mask is a masking input, never a backbone channel.
:func:`model_input_fields_only` and
:mod:`stratego.training.phase9_antileak` prove the separation rather than
assert it.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

import numpy as np

from ..engine.constants import BLUE, NUM_SQUARES, PLAYER_NAMES, RED
from ..engine.legal_moves import legal_action_mask, legal_actions
from ..model.action_frame import (
    absolute_action_to_model,
    absolute_legal_mask_to_model,
)
from ..model.contract import BELIEF_IGNORE_INDEX, OBSERVATION_SHAPE
from .belief_targets import dense_belief_target
from .phase9_contract import (
    ADVANTAGE_FILTER_FLOOR,
    ADVANTAGE_FILTER_QUANTILE,
    ADVANTAGE_STANDARDIZATION_EPSILON,
    BEHAVIOR_LOG_EPSILON,
    GAMMA,
    LAMBDA_ADVANTAGE,
    LAMBDA_VALUE,
    MINIBATCH_SIZE,
    PHASE9_ADVANTAGE_VERSION,
    PHASE9_TRAIN_ORDER_VERSION,
    advantage_filter_threshold,
    advantages as contract_advantages,
    behavior_value_scalar,
    temporal_deltas,
    terminal_z,
    wdl_lambda_targets,
)
from .phase9_seed import parse_phase9_game_id, train_order_seed
from .reconstruction import iter_reconstructed_decisions
from .trajectory import PROBABILITY_SUM_TOLERANCE, GameRecord

#: The Phase 9 trainable-example schema. An Agent 4 addition, not one of
#: Agent 1's nine frozen identities: the learning-design constants it consumes
#: stay in `phase9_contract`, and this version names only the *shape* of the
#: object Agent 5 receives. A field change is a new version, never a silent
#: edit.
PHASE9_EXAMPLE_VERSION = "phase9_example_v1"

#: Numeric slack allowed when a rebuilt target is compared against a stored or
#: independently recomputed one. The targets are float64 blends of stored
#: float32 probabilities, so the only differences that may appear are
#: summation-order ulps.
TARGET_ABS_TOLERANCE = 1e-9

#: Sum-to-one / non-negativity slack for a W/D/L simplex check. Inherited from
#: `trajectory_v1`, which is the tolerance the stored behavior predictions were
#: themselves validated under.
SIMPLEX_TOLERANCE = PROBABILITY_SUM_TOLERANCE

#: The only field of an example that may reach the neural backbone.
MODEL_INPUT_FIELDS = ("observation",)

#: Masking input. Used to zero illegal logits and nothing else — it is not a
#: backbone channel and carries no privileged information (legality is public).
MASKING_FIELDS = ("legal_mask",)

#: Fields that exist to compute a loss term.
LOSS_INPUT_FIELDS = (
    "sampled_action_model",
    "behavior_action_probability",
    "behavior_action_logprob",
    "behavior_legal_probabilities",
    "advantage",
    "standardized_advantage",
    "ppo_eligible",
    "wdl_target",
    "belief_target",
    "belief_mask",
)

#: Fields that identify an example. Diagnostic and provenance only.
IDENTITY_FIELDS = (
    "game_id",
    "decision_index",
    "learner_side",
    "sampled_action_abs",
    "behavior_checkpoint_sha256",
    "rollout_id",
    "sealed_rollout_digest",
)

#: Privileged truth an example must never carry outside its belief fields.
#: Named so the boundary audit has something concrete to look for.
FORBIDDEN_EXAMPLE_FIELDS = (
    "true_types",
    "hidden_identities",
    "opponent_setup",
    "red_setup",
    "blue_setup",
    "terminal_result",
    "state",
    "pieces",
)

LEARNER_PLAYERS_BY_CONTROL = {
    "red": (RED,),
    "blue": (BLUE,),
    "both": (RED, BLUE),
}


class Phase9TargetError(RuntimeError):
    """A Phase 9 example or target could not be built as contracted."""


# ---------------------------------------------------------------------------
# Learner designation
# ---------------------------------------------------------------------------


def learner_players(metadata: dict) -> tuple:
    """The colours this game trains, from `learner_control`.

    `current vs current` trains both colours; every asymmetric matchup trains
    only the current-policy side. The opponent's decisions stay in the
    trajectory for state reconstruction and receive no Phase 9 loss, which is
    the whole content of the learner-control semantics.
    """
    control = metadata.get("learner_control")
    if control not in LEARNER_PLAYERS_BY_CONTROL:
        raise Phase9TargetError(f"unknown learner_control {control!r}")
    players = LEARNER_PLAYERS_BY_CONTROL[control]
    colour = metadata.get("learner_color")
    if control == "both":
        if colour is not None:
            raise Phase9TargetError(
                "a both-sided game names a single learner_color; the two fields "
                "disagree about who is being trained"
            )
    elif colour != control:
        raise Phase9TargetError(
            f"learner_color {colour!r} disagrees with learner_control {control!r}"
        )
    return players


def learner_side_name(player: int) -> str:
    return PLAYER_NAMES[int(player)]


def is_learner_decision(metadata: dict, acting_player: int) -> bool:
    return int(acting_player) in learner_players(metadata)


def terminal_outcome(terminal_result: str, player: int) -> str:
    """`win` / `draw` / `loss` from one player's own final perspective."""
    if terminal_result == "draw":
        return "draw"
    if terminal_result == "red_win":
        winner = RED
    elif terminal_result == "blue_win":
        winner = BLUE
    else:
        raise Phase9TargetError(f"unknown terminal result {terminal_result!r}")
    return "win" if int(player) == winner else "loss"


def learner_decision_plies(record: GameRecord, metadata: dict) -> dict:
    """`{player: (ply, ...)}` — each learner's own decisions, in game order.

    Game order is preserved and the opponent's plies simply do not appear;
    nothing is inserted, re-sorted or interleaved. That is what makes the
    sequence "same-player" in the sense the contract means: consecutive entries
    are consecutive *turns of one player*, with an opponent move (or the
    terminal) in between.
    """
    wanted = learner_players(metadata)
    plies = {player: [] for player in wanted}
    for index, decision in enumerate(record.decisions):
        if int(decision.ply) != index:
            raise Phase9TargetError(
                f"{record.game_id}: decision {index} names ply {decision.ply}"
            )
        actor = int(decision.acting_player)
        if actor in plies:
            plies[actor].append(index)
    return {player: tuple(values) for player, values in plies.items()}


# ---------------------------------------------------------------------------
# One learner colour's temporal sequence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LearnerSequence:
    """One learner colour's own decisions and their same-player targets.

    Every tuple is indexed by position *within this player's sequence*, not by
    ply; `plies[i]` is the game ply of entry `i`. One fixed perspective holds
    for the whole sequence, which is why no sign flips appear anywhere: `v`,
    `z`, the deltas, the advantages and the W/D/L targets are all this
    player's.
    """

    game_id: str
    player: int
    outcome: str
    z: int
    plies: tuple
    predictions: tuple
    values: tuple
    deltas: tuple
    advantages: tuple
    wdl_targets: tuple

    def __len__(self) -> int:
        return len(self.plies)

    def index_of_ply(self, ply: int) -> int:
        try:
            return self.plies.index(int(ply))
        except ValueError as error:  # pragma: no cover - callers hold the ply
            raise Phase9TargetError(
                f"{self.game_id}: ply {ply} is not a learner decision of "
                f"{learner_side_name(self.player)}"
            ) from error


def validate_behavior_wdl(wdl, *, where: str) -> None:
    """Agent 1's storage contract on one stored behavior prediction."""
    values = [float(value) for value in wdl]
    if len(values) != 3:
        raise Phase9TargetError(f"{where}: W/D/L prediction has {len(values)} entries")
    if not all(np.isfinite(values)):
        raise Phase9TargetError(f"{where}: W/D/L prediction is not finite: {values}")
    if any(value < -SIMPLEX_TOLERANCE for value in values):
        raise Phase9TargetError(f"{where}: W/D/L prediction has a negative entry: {values}")
    if abs(sum(values) - 1.0) > SIMPLEX_TOLERANCE:
        raise Phase9TargetError(
            f"{where}: W/D/L prediction sums to {sum(values)!r}, not 1 within "
            f"{SIMPLEX_TOLERANCE}"
        )


def validate_wdl_target(target, *, where: str) -> None:
    """The four requirements the assignment places on every W/D/L target."""
    values = [float(value) for value in target]
    if len(values) != 3:
        raise Phase9TargetError(f"{where}: W/D/L target has {len(values)} entries")
    if not all(np.isfinite(values)):
        raise Phase9TargetError(f"{where}: W/D/L target is not finite: {values}")
    if any(value < -SIMPLEX_TOLERANCE for value in values):
        raise Phase9TargetError(f"{where}: W/D/L target has a negative entry: {values}")
    if abs(sum(values) - 1.0) > SIMPLEX_TOLERANCE:
        raise Phase9TargetError(
            f"{where}: W/D/L target sums to {sum(values)!r}, not 1 within "
            f"{SIMPLEX_TOLERANCE}"
        )


def build_sequence(record: GameRecord, metadata: dict, player: int) -> LearnerSequence:
    """One learner colour's sequence, with every frozen target already applied."""
    plies = learner_decision_plies(record, metadata)[int(player)]
    outcome = terminal_outcome(record.terminal_result, player)
    z = terminal_z(outcome)
    predictions = []
    for ply in plies:
        prediction = tuple(
            float(value) for value in record.decisions[ply].win_draw_loss_prediction
        )
        validate_behavior_wdl(prediction, where=f"{record.game_id} ply {ply}")
        predictions.append(prediction)
    values = tuple(behavior_value_scalar(prediction) for prediction in predictions)
    deltas = tuple(temporal_deltas(list(values), z))
    advantages = tuple(contract_advantages(list(values), z))
    targets = tuple(wdl_lambda_targets(list(predictions), outcome))
    for ply, target in zip(plies, targets):
        validate_wdl_target(target, where=f"{record.game_id} ply {ply}")
    return LearnerSequence(
        game_id=record.game_id,
        player=int(player),
        outcome=outcome,
        z=z,
        plies=tuple(plies),
        predictions=tuple(predictions),
        values=values,
        deltas=deltas,
        advantages=advantages,
        wdl_targets=targets,
    )


def build_sequences(record: GameRecord, metadata: dict) -> dict:
    """`{player: LearnerSequence}` for every colour this game trains."""
    return {
        player: build_sequence(record, metadata, player)
        for player in learner_players(metadata)
    }


def sequence_decision_count(sequences: dict) -> int:
    return sum(len(sequence) for sequence in sequences.values())


def verify_learner_decision_count(record: GameRecord, metadata: dict, sequences: dict) -> list:
    """The stored collection record's learner quantity, re-derived from payload.

    `learner_decision_count` is the collector's own bookkeeping. It is checked
    against a count taken from the payload, so a sidecar that over- or
    under-claims trainable decisions is a finding rather than an assumption.
    """
    problems: list[str] = []
    rebuilt = sequence_decision_count(sequences)
    stored = int(metadata["learner_decision_count"])
    if rebuilt != stored:
        problems.append(
            f"{record.game_id}: payload holds {rebuilt} learner decisions, the "
            f"collection record claims {stored}"
        )
    if int(metadata["total_decisions"]) != len(record.decisions):
        problems.append(
            f"{record.game_id}: metadata claims {metadata['total_decisions']} "
            f"decisions, the payload holds {len(record.decisions)}"
        )
    return problems


# ---------------------------------------------------------------------------
# Per-iteration advantage filter and standardization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IterationTargetStatistics:
    """The `phase9_advantage_v1` state of one sealed iteration.

    Computed once over every learner decision of the rollout and then applied
    to each example, because both the threshold and the standardization
    moments are iteration-level quantities. `zero_variance` and `no_eligible`
    record the two degenerate cases explicitly rather than letting a division
    decide what happens.
    """

    namespace: str
    iteration: int
    sealed_rollout_digest: str
    games: int
    learner_decisions: int
    threshold: float
    eligible: int
    retention_fraction: float
    mean_eligible: float
    std_eligible: float
    standardization_epsilon: float
    zero_variance: bool
    no_eligible: bool
    advantage_min: float
    advantage_max: float
    advantage_mean: float

    def is_eligible(self, advantage: float) -> bool:
        """`|A_t| >= tau`, the only PPO eligibility rule."""
        return abs(float(advantage)) >= self.threshold

    def standardize(self, advantage: float) -> float:
        """`A_hat = (A - mean_selected) / (std_selected + 1e-8)`.

        Both degenerate cases fall out of the frozen formula rather than out of
        a special case: with no eligible decisions the moments are `0/0` by
        convention and every standardized advantage is `0`; with zero variance
        the numerator is `0` for every eligible decision, so the quotient is
        again `0` and the PPO term contributes no gradient. An ineligible
        decision is standardized with the same moments — its `ppo_eligible`
        flag, not its value, is what keeps it out of the policy loss.
        """
        if self.no_eligible:
            return 0.0
        return (float(advantage) - self.mean_eligible) / (
            self.std_eligible + self.standardization_epsilon
        )

    def to_dict(self) -> dict:
        return {
            "advantage_version": PHASE9_ADVANTAGE_VERSION,
            "namespace": self.namespace,
            "iteration": int(self.iteration),
            "sealed_rollout_digest": self.sealed_rollout_digest,
            "games": int(self.games),
            "learner_decisions": int(self.learner_decisions),
            "quantile": ADVANTAGE_FILTER_QUANTILE,
            "floor": ADVANTAGE_FILTER_FLOOR,
            "threshold": self.threshold,
            "eligible": int(self.eligible),
            "retention_fraction": self.retention_fraction,
            "mean_eligible": self.mean_eligible,
            "std_eligible": self.std_eligible,
            "standardization_epsilon": self.standardization_epsilon,
            "zero_variance": bool(self.zero_variance),
            "no_eligible": bool(self.no_eligible),
            "advantage_min": self.advantage_min,
            "advantage_max": self.advantage_max,
            "advantage_mean": self.advantage_mean,
        }


def iteration_statistics(
    advantages_by_key: dict,
    *,
    namespace: str,
    iteration: int,
    sealed_rollout_digest: str,
    games: int,
) -> IterationTargetStatistics:
    """The filter threshold and PPO-subset moments of one sealed iteration.

    `advantages_by_key` maps `(game_id, ply)` to that decision's advantage; the
    keys are unused here beyond counting, but taking the mapping rather than a
    bare list keeps the caller honest about auditing every learner decision
    exactly once.
    """
    values = [float(value) for value in advantages_by_key.values()]
    if not values:
        raise Phase9TargetError(
            f"{namespace} iteration {iteration}: no learner decisions to filter"
        )
    threshold = advantage_filter_threshold(values)
    eligible = [value for value in values if abs(value) >= threshold]
    if eligible:
        array = np.asarray(eligible, dtype=np.float64)
        mean = float(array.mean())
        std = float(array.std())  # population std, ddof=0, as frozen
    else:
        mean = 0.0
        std = 0.0
    return IterationTargetStatistics(
        namespace=str(namespace),
        iteration=int(iteration),
        sealed_rollout_digest=str(sealed_rollout_digest),
        games=int(games),
        learner_decisions=len(values),
        threshold=float(threshold),
        eligible=len(eligible),
        retention_fraction=len(eligible) / len(values),
        mean_eligible=mean,
        std_eligible=std,
        standardization_epsilon=ADVANTAGE_STANDARDIZATION_EPSILON,
        zero_variance=bool(eligible) and std == 0.0,
        no_eligible=not eligible,
        advantage_min=min(values),
        advantage_max=max(values),
        advantage_mean=float(np.asarray(values, dtype=np.float64).mean()),
    )


def collect_iteration_advantages(reader, *, limit: "int | None" = None) -> tuple:
    """Pass 1: every learner decision's advantage, without replaying a game.

    Returns `(advantages_by_key, sequences_by_game, problems)`. Nothing here
    touches the engine: the same-player targets are functions of the stored
    W/D/L predictions and the terminal result alone, which is what makes the
    iteration-level filter affordable.
    """
    advantages_by_key: dict = {}
    sequences_by_game: dict = {}
    problems: list[str] = []
    for count, game_id in enumerate(reader.game_ids):
        if limit is not None and count >= limit:
            break
        record, metadata = reader.read_game(game_id)
        sequences = build_sequences(record, metadata)
        problems.extend(verify_learner_decision_count(record, metadata, sequences))
        sequences_by_game[game_id] = sequences
        for sequence in sequences.values():
            for ply, advantage in zip(sequence.plies, sequence.advantages):
                key = (game_id, int(ply))
                if key in advantages_by_key:  # pragma: no cover - one actor per ply
                    problems.append(f"{game_id} ply {ply}: two learner sequences claim it")
                advantages_by_key[key] = float(advantage)
    return advantages_by_key, sequences_by_game, problems


# ---------------------------------------------------------------------------
# The trainable example
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase9RLExample:
    """One `phase9_example_v1` trainable decision.

    `observation` is the only model input. `legal_mask` masks logits.
    Everything else is a loss input or identity. The belief label and its mask
    are targets: they are built from privileged truth *after* the public
    observation exists and never enter the tensor the backbone receives.
    """

    observation: np.ndarray  # float32 [127, 10, 10]
    legal_mask: np.ndarray  # bool [10000], model frame
    sampled_action_abs: int
    sampled_action_model: int
    behavior_action_probability: float
    behavior_action_logprob: float
    behavior_legal_actions: tuple  # ascending absolute, as stored
    behavior_legal_probabilities: tuple  # aligned with the above
    advantage: float
    standardized_advantage: float
    ppo_eligible: bool
    wdl_target: tuple  # (W, D, L), learner perspective
    belief_target: np.ndarray  # int64 [100]
    belief_mask: np.ndarray  # bool [100]
    game_id: str
    decision_index: int
    learner_side: int
    behavior_checkpoint_sha256: str
    rollout_id: str
    sealed_rollout_digest: str

    @property
    def key(self) -> tuple:
        return (self.game_id, self.decision_index)

    @property
    def learner_side_name(self) -> str:
        return learner_side_name(self.learner_side)

    @property
    def supervised_belief_squares(self) -> int:
        return int(self.belief_mask.sum())


def rollout_identity(namespace: str, iteration: int) -> str:
    """The logical name of one sealed rollout. Never a path."""
    return f"phase9_rollout_v1|ns={namespace}|it={int(iteration):03d}"


def behavior_action_probability(decision) -> float:
    """`pi_b(a_t|s_t)`: the stored float32 entry of the realized action.

    The storage is the authority (Agent 1's `training_time_probability`), so
    the PPO ratio's denominator is exact and independent of any device
    recomputation. The index is found by looking the action up in the stored
    ascending legal list rather than by trusting a position.
    """
    legal = tuple(int(action) for action in decision.legal_action_ids)
    action = int(decision.selected_action_id)
    if action not in legal:
        raise Phase9TargetError(
            f"{decision.game_id} ply {decision.ply}: the realized action is not in "
            "the stored legal set"
        )
    probability = float(decision.old_probabilities[legal.index(action)])
    if not np.isfinite(probability) or probability < 0.0:
        raise Phase9TargetError(
            f"{decision.game_id} ply {decision.ply}: behavior probability "
            f"{probability!r} is not a usable probability"
        )
    return probability


def behavior_action_logprob(probability: float) -> float:
    """`log pi_b = ln(max(p, 1e-12))`, the frozen floor from Agent 1."""
    return float(np.log(max(float(probability), BEHAVIOR_LOG_EPSILON)))


def build_example(
    record: GameRecord,
    metadata: dict,
    rebuilt,
    sequence: LearnerSequence,
    statistics: "IterationTargetStatistics | None",
) -> Phase9RLExample:
    """One example from one reconstructed learner decision.

    Cross-checks as it builds, exactly as Phase 8's builder does: the replayed
    legal set must equal the stored one, the replayed actor must be the stored
    actor and must be the learner colour this sequence belongs to, and the
    realized action must be legal in the replayed position. A disagreement
    means the payload and the replay no longer describe the same game, which is
    a stop condition rather than something to repair.
    """
    ply = int(rebuilt.ply)
    decision = record.decisions[ply]
    if int(decision.ply) != ply:
        raise Phase9TargetError(
            f"{record.game_id}: decision record at index {ply} names ply {decision.ply}"
        )
    actor = int(rebuilt.acting_player)
    if actor != int(decision.acting_player):
        raise Phase9TargetError(
            f"{record.game_id} ply {ply}: replayed acting player differs from the record"
        )
    if actor != int(sequence.player):
        raise Phase9TargetError(
            f"{record.game_id} ply {ply}: decision belongs to "
            f"{learner_side_name(actor)}, not to this "
            f"{learner_side_name(sequence.player)} sequence"
        )
    if not is_learner_decision(metadata, actor):
        raise Phase9TargetError(
            f"{record.game_id} ply {ply}: {learner_side_name(actor)} is not "
            f"learner-controlled under learner_control={metadata['learner_control']!r}"
        )
    if tuple(rebuilt.legal_action_ids) != tuple(decision.legal_action_ids):
        raise Phase9TargetError(
            f"{record.game_id} ply {ply}: replayed legal actions differ from the record"
        )
    if rebuilt.legal_mask is None:
        raise Phase9TargetError(
            "build_example needs dense_mask=True reconstruction; the model-frame "
            "mask is converted from the engine's dense mask"
        )
    action = int(decision.selected_action_id)
    if action not in rebuilt.legal_action_ids:
        raise Phase9TargetError(
            f"{record.game_id} ply {ply}: the recorded action is not legal on replay"
        )

    index = sequence.index_of_ply(ply)
    advantage = float(sequence.advantages[index])
    if statistics is None:
        eligible = False
        standardized = 0.0
    else:
        eligible = statistics.is_eligible(advantage)
        standardized = statistics.standardize(advantage)

    labels, mask = dense_belief_target(rebuilt.state, actor)
    probability = behavior_action_probability(decision)
    return Phase9RLExample(
        observation=np.ascontiguousarray(rebuilt.observation, dtype=np.float32),
        legal_mask=absolute_legal_mask_to_model(rebuilt.legal_mask, actor).astype(bool),
        sampled_action_abs=action,
        sampled_action_model=absolute_action_to_model(action, actor),
        behavior_action_probability=probability,
        behavior_action_logprob=behavior_action_logprob(probability),
        behavior_legal_actions=tuple(int(item) for item in decision.legal_action_ids),
        behavior_legal_probabilities=tuple(
            float(item) for item in decision.old_probabilities
        ),
        advantage=advantage,
        standardized_advantage=float(standardized),
        ppo_eligible=bool(eligible),
        wdl_target=tuple(float(value) for value in sequence.wdl_targets[index]),
        belief_target=labels,
        belief_mask=mask,
        game_id=record.game_id,
        decision_index=ply,
        learner_side=actor,
        behavior_checkpoint_sha256=str(metadata["behavior_checkpoint_sha256"]),
        rollout_id=rollout_identity(metadata["namespace"], metadata["iteration"]),
        sealed_rollout_digest=str(statistics.sealed_rollout_digest) if statistics else "",
    )


def examples_for_game(
    record: GameRecord,
    metadata: dict,
    statistics: "IterationTargetStatistics | None" = None,
    sequences: "dict | None" = None,
):
    """Yield one game's learner examples in ascending ply order.

    Both learner colours of a `both` game are yielded from the same replay, in
    game order: the sequences are per colour, the replay is not. Sequential
    reconstruction advances one state instead of restoring a snapshot per
    position, so `copy_state=False` is safe only because each example is fully
    built — belief labels included — before the next ply is pulled.
    """
    if metadata["game_id"] != record.game_id:
        raise Phase9TargetError(
            f"metadata for {metadata['game_id']!r} was paired with record "
            f"{record.game_id!r}"
        )
    if sequences is None:
        sequences = build_sequences(record, metadata)
    by_ply = {
        int(ply): sequence
        for sequence in sequences.values()
        for ply in sequence.plies
    }
    if not by_ply:
        return
    for rebuilt in iter_reconstructed_decisions(
        record,
        sorted(by_ply),
        dense_mask=True,
        include_public_knowledge=False,
        copy_state=False,
    ):
        yield build_example(record, metadata, rebuilt, by_ply[int(rebuilt.ply)], statistics)


def iter_rollout_examples(
    reader,
    statistics: "IterationTargetStatistics | None" = None,
    *,
    sequences_by_game: "dict | None" = None,
    game_ids=None,
):
    """The deterministic rollout-to-example iterator handed to Agent 5.

    Games are walked in the reader's own ascending `game_id` order and each
    game's decisions in ascending ply order, so the example stream of a sealed
    rollout is a pure function of the rollout — no worker count, arrival order
    or resume boundary appears anywhere in it. Every read is digest-checked by
    the store.
    """
    for game_id in reader.game_ids if game_ids is None else game_ids:
        record, metadata = reader.read_game(game_id)
        sequences = None if sequences_by_game is None else sequences_by_game.get(game_id)
        yield from examples_for_game(record, metadata, statistics, sequences)


# ---------------------------------------------------------------------------
# Train order, minibatches and the resumable cursor
# ---------------------------------------------------------------------------


def train_order_keys(reader, sequences_by_game: "dict | None" = None) -> tuple:
    """The universe: every learner decision of the sealed iteration, sorted.

    Sorted by `(game_id, ply)` — the order `phase9_train_order_v1` shuffles.
    Sorting first is what makes the shuffle reproducible from the rollout
    alone: the store's own iteration order could in principle change with a
    reconciliation, the sorted key order cannot.
    """
    keys: list = []
    for game_id in reader.game_ids:
        if sequences_by_game is not None and game_id in sequences_by_game:
            sequences = sequences_by_game[game_id]
        else:
            record, metadata = reader.read_game(game_id)
            sequences = build_sequences(record, metadata)
        for sequence in sequences.values():
            keys.extend((game_id, int(ply)) for ply in sequence.plies)
    return tuple(sorted(keys))


def epoch_order(keys, namespace: str, iteration: int, epoch: int) -> tuple:
    """Positions into `keys` for one optimizer epoch, in consumption order.

    `random.Random(train_order_seed(namespace, iteration, epoch)).shuffle` over
    the index list, exactly as frozen. Returning indices rather than keys keeps
    the shuffle independent of what an example carries.
    """
    order = list(range(len(keys)))
    random.Random(train_order_seed(namespace, int(iteration), int(epoch))).shuffle(order)
    return tuple(order)


def minibatch_slices(total: int, size: int = MINIBATCH_SIZE) -> tuple:
    """Contiguous `(start, stop)` slices; the final partial batch is consumed."""
    if total < 0 or size < 1:
        raise Phase9TargetError(f"cannot slice {total} examples into batches of {size}")
    return tuple(
        (start, min(start + size, total)) for start in range(0, total, size)
    )


@dataclass(frozen=True)
class Phase9MinibatchCursor:
    """Where a training pass is, in logical terms only.

    Carries no tensors, no file offsets and no worker identity: an epoch index,
    a minibatch index within that epoch, and the running examples-consumed
    counter `phase9_checkpoint_v1` requires. A resumed trainer rebuilds the
    same shuffled order from `(namespace, iteration, epoch)` and skips
    `minibatch_index` batches — which is exact because the order is a pure
    function of those three values and the sealed rollout's key list.
    """

    namespace: str
    iteration: int
    sealed_rollout_digest: str
    epoch: int
    minibatch_index: int
    examples_consumed: int
    total_examples: int
    minibatch_size: int = MINIBATCH_SIZE
    epochs: int = 0

    @property
    def minibatches_per_epoch(self) -> int:
        return len(minibatch_slices(self.total_examples, self.minibatch_size))

    @property
    def finished(self) -> bool:
        return self.epochs > 0 and self.epoch >= self.epochs

    def advance(self, consumed: int) -> "Phase9MinibatchCursor":
        """The cursor after one minibatch of `consumed` examples."""
        index = self.minibatch_index + 1
        epoch = self.epoch
        if index >= self.minibatches_per_epoch:
            index = 0
            epoch += 1
        return Phase9MinibatchCursor(
            namespace=self.namespace,
            iteration=self.iteration,
            sealed_rollout_digest=self.sealed_rollout_digest,
            epoch=epoch,
            minibatch_index=index,
            examples_consumed=self.examples_consumed + int(consumed),
            total_examples=self.total_examples,
            minibatch_size=self.minibatch_size,
            epochs=self.epochs,
        )

    def to_dict(self) -> dict:
        return {
            "train_order_version": PHASE9_TRAIN_ORDER_VERSION,
            "namespace": self.namespace,
            "iteration": int(self.iteration),
            "sealed_rollout_digest": self.sealed_rollout_digest,
            "epoch": int(self.epoch),
            "minibatch_index": int(self.minibatch_index),
            "examples_consumed": int(self.examples_consumed),
            "total_examples": int(self.total_examples),
            "minibatch_size": int(self.minibatch_size),
            "epochs": int(self.epochs),
        }

    @staticmethod
    def start(
        *,
        namespace: str,
        iteration: int,
        sealed_rollout_digest: str,
        total_examples: int,
        epochs: int,
        minibatch_size: int = MINIBATCH_SIZE,
    ) -> "Phase9MinibatchCursor":
        return Phase9MinibatchCursor(
            namespace=str(namespace),
            iteration=int(iteration),
            sealed_rollout_digest=str(sealed_rollout_digest),
            epoch=0,
            minibatch_index=0,
            examples_consumed=0,
            total_examples=int(total_examples),
            minibatch_size=int(minibatch_size),
            epochs=int(epochs),
        )


def minibatch_keys(keys, namespace: str, iteration: int, epoch: int, cursor_index: int,
                   size: int = MINIBATCH_SIZE) -> tuple:
    """The exact `(game_id, ply)` keys of one minibatch of one epoch."""
    order = epoch_order(keys, namespace, iteration, epoch)
    slices = minibatch_slices(len(keys), size)
    if not 0 <= cursor_index < len(slices):
        raise Phase9TargetError(
            f"minibatch {cursor_index} is outside 0..{len(slices) - 1}"
        )
    start, stop = slices[cursor_index]
    return tuple(keys[position] for position in order[start:stop])


# ---------------------------------------------------------------------------
# The batch boundary
# ---------------------------------------------------------------------------


def model_input_fields_only(example: Phase9RLExample) -> dict:
    """Exactly what may reach the backbone, and nothing else.

    Agent 5 calls this rather than reaching into an example, so the boundary is
    a function with a name instead of a convention someone has to remember.
    """
    return {field: getattr(example, field) for field in MODEL_INPUT_FIELDS}


def build_batch(examples) -> dict:
    """Collate examples into arrays, keeping the privilege boundary visible.

    The returned mapping separates `model_input`, `masking`, `loss_inputs` and
    `identity`; a caller passing `batch["model_input"]` to a network cannot
    reach a belief label, an advantage or a game id by accident.
    """
    items = list(examples)
    if not items:
        raise Phase9TargetError("cannot build a batch from no examples")
    return {
        "example_version": PHASE9_EXAMPLE_VERSION,
        "size": len(items),
        "model_input": {
            "observation": np.stack(
                [np.ascontiguousarray(item.observation, dtype=np.float32) for item in items]
            )
        },
        "masking": {
            "legal_mask": np.stack([item.legal_mask for item in items]),
        },
        "loss_inputs": {
            "sampled_action_model": np.asarray(
                [item.sampled_action_model for item in items], dtype=np.int64
            ),
            "behavior_action_probability": np.asarray(
                [item.behavior_action_probability for item in items], dtype=np.float32
            ),
            "behavior_action_logprob": np.asarray(
                [item.behavior_action_logprob for item in items], dtype=np.float32
            ),
            "advantage": np.asarray([item.advantage for item in items], dtype=np.float32),
            "standardized_advantage": np.asarray(
                [item.standardized_advantage for item in items], dtype=np.float32
            ),
            "ppo_eligible": np.asarray([item.ppo_eligible for item in items], dtype=bool),
            "wdl_target": np.asarray([item.wdl_target for item in items], dtype=np.float32),
            "belief_target": np.stack([item.belief_target for item in items]),
            "belief_mask": np.stack([item.belief_mask for item in items]),
        },
        "identity": {
            "game_id": tuple(item.game_id for item in items),
            "decision_index": tuple(int(item.decision_index) for item in items),
            "learner_side": tuple(int(item.learner_side) for item in items),
            "behavior_checkpoint_sha256": tuple(
                item.behavior_checkpoint_sha256 for item in items
            ),
            "rollout_id": tuple(item.rollout_id for item in items),
        },
        "behavior_legal": tuple(
            (item.behavior_legal_actions, item.behavior_legal_probabilities)
            for item in items
        ),
    }


# ---------------------------------------------------------------------------
# The independent example audit
# ---------------------------------------------------------------------------


#: `absolute square -> normalized square` per observer, built by inverting the
#: frozen `from_perspective` direction — the audit walks the opposite direction
#: of the mapping `dense_belief_target` used, so the two must agree.
def _normalized_by_absolute():
    from ..engine.coordinates import from_perspective

    return {
        observer: {
            from_perspective(normalized, observer): normalized
            for normalized in range(NUM_SQUARES)
        }
        for observer in (RED, BLUE)
    }


_NORMALIZED_BY_ABSOLUTE = _normalized_by_absolute()


def audit_example(
    example: Phase9RLExample,
    record: GameRecord,
    metadata: dict,
    rebuilt,
    sequence: LearnerSequence,
    statistics: "IterationTargetStatistics | None" = None,
) -> list:
    """Every disagreement between one example and independently rebuilt truth.

    Deliberately avoids the routes :func:`build_example` took wherever a second
    one exists: the legal mask is inverted back through the engine frame and
    compared against the engine's own dense mask, the belief labels are
    re-derived square by square from the privileged piece records instead of
    through `dense_belief_target`, the learner designation is recomputed from
    `learner_control`, and the advantage / W/D/L target are recomputed from the
    sequence's own arithmetic. What is left over — the observation itself — is
    compared against a fresh engine build.
    """
    from ..engine.constants import NUM_PIECE_TYPES, opponent_of
    from ..engine.observation import (
        CH_HIDDEN_OPPONENT_OCCUPANCY,
        CH_KNOWN_OPPONENT_IDENTITY,
        CH_LAKE_MASK,
        CH_OWN_IDENTITY,
        build_observation,
    )
    from ..model.action_frame import (
        absolute_legal_actions_to_model,
        model_action_to_absolute,
        model_legal_mask_to_absolute,
    )

    problems: list[str] = []
    state = rebuilt.state
    actor = int(rebuilt.acting_player)
    ply = int(rebuilt.ply)
    prefix = f"{record.game_id} ply {ply}"
    decision = record.decisions[ply]

    # -- identity and learner designation ----------------------------------
    if example.game_id != record.game_id or example.decision_index != ply:
        problems.append(f"{prefix}: example identity fields disagree")
    if example.learner_side != actor:
        problems.append(f"{prefix}: example learner side is not the acting player")
    if actor != int(state.acting_player):
        problems.append(f"{prefix}: acting player disagrees with the replayed state")
    if int(state.total_moves) != ply:
        problems.append(f"{prefix}: replayed state is at ply {state.total_moves}")
    if actor not in learner_players(metadata):
        problems.append(
            f"{prefix}: {learner_side_name(actor)} is not learner-controlled under "
            f"learner_control={metadata['learner_control']!r}"
        )
    if example.behavior_checkpoint_sha256 != metadata["behavior_checkpoint_sha256"]:
        problems.append(f"{prefix}: example names a foreign behavior checkpoint")
    if example.rollout_id != rollout_identity(metadata["namespace"], metadata["iteration"]):
        problems.append(f"{prefix}: example names a foreign rollout")

    # -- action frames ------------------------------------------------------
    live_legal = tuple(legal_actions(state))
    if live_legal != tuple(decision.legal_action_ids):
        problems.append(f"{prefix}: engine legal set differs from the stored one")
    if example.sampled_action_abs != int(decision.selected_action_id):
        problems.append(f"{prefix}: example does not carry the recorded action")
    if example.sampled_action_abs not in live_legal:
        problems.append(f"{prefix}: recorded action is illegal in the replayed state")
    if model_action_to_absolute(example.sampled_action_model, actor) != (
        example.sampled_action_abs
    ):
        problems.append(f"{prefix}: model-frame action does not invert to the absolute one")
    model_legal = absolute_legal_actions_to_model(live_legal, actor)
    mask_indices = tuple(int(index) for index in np.flatnonzero(example.legal_mask))
    if mask_indices != model_legal:
        problems.append(f"{prefix}: model mask disagrees with the converted legal list")
    if not example.legal_mask[example.sampled_action_model]:
        problems.append(f"{prefix}: model mask excludes the sampled action")
    engine_mask = legal_action_mask(state, list(live_legal))
    if not np.array_equal(
        model_legal_mask_to_absolute(example.legal_mask.astype(np.uint8), actor), engine_mask
    ):
        problems.append(f"{prefix}: model mask does not invert to the engine mask")

    # -- behavior quantity --------------------------------------------------
    if tuple(example.behavior_legal_actions) != tuple(decision.legal_action_ids):
        problems.append(f"{prefix}: example legal list is not the stored one")
    if tuple(example.behavior_legal_probabilities) != tuple(decision.old_probabilities):
        problems.append(f"{prefix}: example behavior distribution is not the stored one")
    expected_probability = behavior_action_probability(decision)
    if example.behavior_action_probability != expected_probability:
        problems.append(f"{prefix}: pi_b(a_t|s_t) is not the stored realized probability")
    if abs(
        example.behavior_action_logprob
        - float(np.log(max(expected_probability, BEHAVIOR_LOG_EPSILON)))
    ) > TARGET_ABS_TOLERANCE:
        problems.append(f"{prefix}: behavior log-probability disagrees with its floor rule")

    # -- same-player targets ------------------------------------------------
    index = sequence.index_of_ply(ply)
    if abs(example.advantage - float(sequence.advantages[index])) > TARGET_ABS_TOLERANCE:
        problems.append(f"{prefix}: advantage disagrees with the sequence arithmetic")
    for component, (value, expected) in enumerate(
        zip(example.wdl_target, sequence.wdl_targets[index])
    ):
        if abs(float(value) - float(expected)) > TARGET_ABS_TOLERANCE:
            problems.append(f"{prefix}: W/D/L target component {component} disagrees")
    validate_wdl_target(example.wdl_target, where=prefix)
    if sequence.outcome != terminal_outcome(record.terminal_result, actor):
        problems.append(f"{prefix}: sequence outcome is not this player's perspective")
    if statistics is not None:
        if example.ppo_eligible != statistics.is_eligible(example.advantage):
            problems.append(f"{prefix}: PPO eligibility disagrees with |A| >= tau")
        if abs(
            example.standardized_advantage - statistics.standardize(example.advantage)
        ) > TARGET_ABS_TOLERANCE:
            problems.append(f"{prefix}: standardized advantage disagrees with the moments")
        if example.sealed_rollout_digest != statistics.sealed_rollout_digest:
            problems.append(f"{prefix}: example names a foreign sealed rollout digest")

    # -- belief labels, square by square from the privileged piece records --
    inverse = _NORMALIZED_BY_ABSOLUTE[actor]
    expected_labels: dict = {}
    for piece in state.pieces:
        if piece.owner == actor or not piece.alive or piece.known_to(actor):
            continue
        expected_labels[inverse[piece.current_square]] = piece.true_type
    mask_squares = {int(square) for square in np.flatnonzero(example.belief_mask)}
    if mask_squares != set(expected_labels):
        problems.append(f"{prefix}: belief mask does not cover exactly the hidden pieces")
    for square in range(NUM_SQUARES):
        label = int(example.belief_target[square])
        if square in expected_labels:
            if label != expected_labels[square]:
                problems.append(f"{prefix}: belief label at square {square} is wrong")
            if not 0 <= label < NUM_PIECE_TYPES:
                problems.append(f"{prefix}: belief label out of range at {square}")
        elif label != BELIEF_IGNORE_INDEX:
            problems.append(f"{prefix}: unsupervised square {square} carries a label")
    if not np.array_equal(
        example.belief_mask, example.belief_target != BELIEF_IGNORE_INDEX
    ):
        problems.append(f"{prefix}: belief mask and labels disagree")

    # -- the model input ----------------------------------------------------
    observation = example.observation
    if observation.shape != OBSERVATION_SHAPE or observation.dtype != np.float32:
        problems.append(f"{prefix}: observation shape/dtype violates the contract")
    elif not np.array_equal(observation, build_observation(state, actor)):
        problems.append(f"{prefix}: observation is not the observer-safe engine build")
    planes = observation.reshape(observation.shape[0], NUM_SQUARES)
    if not np.array_equal(planes[CH_HIDDEN_OPPONENT_OCCUPANCY] > 0.5, example.belief_mask):
        problems.append(f"{prefix}: hidden-occupancy channel disagrees with the belief mask")
    known = planes[
        CH_KNOWN_OPPONENT_IDENTITY : CH_KNOWN_OPPONENT_IDENTITY + NUM_PIECE_TYPES
    ].sum(axis=0)
    if np.any(known[example.belief_mask] > 0.0):
        problems.append(f"{prefix}: a supervised square holds a known opponent piece")
    own = planes[CH_OWN_IDENTITY : CH_OWN_IDENTITY + NUM_PIECE_TYPES].sum(axis=0)
    if np.any(own[example.belief_mask] > 0.0):
        problems.append(f"{prefix}: a supervised square holds an own piece")
    if np.any(planes[CH_LAKE_MASK][example.belief_mask] > 0.0):
        problems.append(f"{prefix}: a supervised square is a lake")
    if opponent_of(actor) == actor:  # pragma: no cover - defensive, keeps the import honest
        problems.append(f"{prefix}: opponent resolution collapsed")
    return problems


# ---------------------------------------------------------------------------
# The serializable example contract
# ---------------------------------------------------------------------------


def example_contract() -> dict:
    """The `phase9_example_v1` dataset/example contract, as published.

    Field roles are part of the contract, not documentation: `model_input`
    lists the only array a backbone may receive, and every privileged quantity
    is a target. The learning constants are quoted from `phase9_contract` so a
    reader can see them, but this document is not their source.
    """
    return {
        "example_version": PHASE9_EXAMPLE_VERSION,
        "advantage_version": PHASE9_ADVANTAGE_VERSION,
        "train_order_version": PHASE9_TRAIN_ORDER_VERSION,
        "constants": {
            "gamma": GAMMA,
            "lambda_A": LAMBDA_ADVANTAGE,
            "lambda_V": LAMBDA_VALUE,
            "filter_quantile": ADVANTAGE_FILTER_QUANTILE,
            "filter_floor": ADVANTAGE_FILTER_FLOOR,
            "standardization_epsilon": ADVANTAGE_STANDARDIZATION_EPSILON,
            "behavior_log_epsilon": BEHAVIOR_LOG_EPSILON,
            "minibatch_size": MINIBATCH_SIZE,
        },
        "fields": {
            "observation": {
                "role": "model_input",
                "dtype": "float32",
                "shape": list(OBSERVATION_SHAPE),
                "note": "observation_v2_1_127ch; the only backbone input",
            },
            "legal_mask": {
                "role": "masking",
                "dtype": "bool",
                "shape": [10000],
                "note": "model frame (perspective_normalized_squares); masks logits only",
            },
            "sampled_action_abs": {"role": "identity", "dtype": "int", "note": "engine frame"},
            "sampled_action_model": {
                "role": "loss_input",
                "dtype": "int",
                "note": "model frame; the PPO ratio's numerator index",
            },
            "behavior_action_probability": {
                "role": "loss_input",
                "dtype": "float32",
                "note": "pi_b(a_t|s_t), the stored float32 entry of the realized action",
            },
            "behavior_action_logprob": {
                "role": "loss_input",
                "dtype": "float64",
                "note": "ln(max(pi_b, 1e-12))",
            },
            "behavior_legal_probabilities": {
                "role": "loss_input",
                "dtype": "float32[]",
                "note": (
                    "the frozen behavior distribution over the legal set, ascending "
                    "absolute order; the KL term's pi_b"
                ),
            },
            "behavior_legal_actions": {
                "role": "loss_input",
                "dtype": "int[]",
                "note": "ascending absolute legal actions, aligned with the above",
            },
            "advantage": {"role": "loss_input", "dtype": "float64", "note": "A_t"},
            "standardized_advantage": {
                "role": "loss_input",
                "dtype": "float64",
                "note": "A_hat over the PPO subset's moments",
            },
            "ppo_eligible": {
                "role": "loss_input",
                "dtype": "bool",
                "note": "|A_t| >= tau; policy loss only",
            },
            "wdl_target": {
                "role": "loss_input",
                "dtype": "float64[3]",
                "note": "soft W/D/L lambda target, learner perspective",
            },
            "belief_target": {
                "role": "loss_input",
                "dtype": "int64",
                "shape": [100],
                "note": (
                    "privileged hidden-only label per normalized square, "
                    f"{BELIEF_IGNORE_INDEX} elsewhere; never a model input"
                ),
            },
            "belief_mask": {
                "role": "loss_input",
                "dtype": "bool",
                "shape": [100],
                "note": "true exactly on unresolved opponent pieces",
            },
            "game_id": {"role": "identity", "dtype": "str"},
            "decision_index": {"role": "identity", "dtype": "int", "note": "game ply"},
            "learner_side": {"role": "identity", "dtype": "int", "note": "0 red, 1 blue"},
            "behavior_checkpoint_sha256": {"role": "identity", "dtype": "str"},
            "rollout_id": {"role": "identity", "dtype": "str"},
            "sealed_rollout_digest": {"role": "identity", "dtype": "str"},
        },
        "model_input_fields": list(MODEL_INPUT_FIELDS),
        "masking_fields": list(MASKING_FIELDS),
        "loss_input_fields": list(LOSS_INPUT_FIELDS),
        "identity_fields": list(IDENTITY_FIELDS),
        "forbidden_fields": list(FORBIDDEN_EXAMPLE_FIELDS),
        "populations": {
            "policy": "learner decisions with ppo_eligible = True",
            "value": "every learner decision",
            "belief": "every learner decision",
            "kl": "every learner decision",
        },
        "sequence": {
            "definition": (
                "per game and per learner-controlled colour, that player's own "
                "decisions in ascending ply order; opponent decisions are never "
                "inserted as learner steps"
            ),
            "perspective": (
                "one fixed player perspective for the whole sequence, so no sign "
                "flip is applied at an opponent turn"
            ),
            "terminal": (
                "the final entry uses delta_t = z - v_t, which covers the case "
                "where the game ends before that player's next turn"
            ),
        },
        "belief_semantics": {
            "source": "stratego.training.belief_targets.dense_belief_target",
            "scope": "unresolved opponent hidden pieces only",
            "frame": "perspective_normalized_squares",
            "ignored": "own, revealed, empty, lake, captured",
            "types": 12,
            "privilege": (
                "built from the privileged state after the public observation "
                "exists; no belief label enters model input"
            ),
        },
        "train_order": {
            "universe": "the sealed iteration's learner decisions",
            "key": "(game_id, ply), sorted",
            "shuffle": (
                "random.Random(train_order_seed(namespace, iteration, epoch))"
                ".shuffle over indices into the sorted key list"
            ),
            "minibatches": (
                f"contiguous {MINIBATCH_SIZE}-example slices; the final partial "
                "minibatch is consumed, never dropped"
            ),
            "cursor": (
                "(epoch, minibatch_index, examples_consumed); a resume rebuilds "
                "the epoch order from (namespace, iteration, epoch) and skips "
                "minibatch_index batches"
            ),
        },
        "iterator": (
            "iter_rollout_examples(reader, statistics): games in ascending "
            "game_id order, decisions in ascending ply order, every read "
            "digest-checked by phase9_rollout_store"
        ),
    }


def example_contract_digest() -> str:
    """SHA-256 over the canonical `phase9_example_v1` document."""
    return hashlib.sha256(
        json.dumps(example_contract(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "FORBIDDEN_EXAMPLE_FIELDS",
    "IDENTITY_FIELDS",
    "LEARNER_PLAYERS_BY_CONTROL",
    "LOSS_INPUT_FIELDS",
    "MASKING_FIELDS",
    "MODEL_INPUT_FIELDS",
    "PHASE9_EXAMPLE_VERSION",
    "SIMPLEX_TOLERANCE",
    "TARGET_ABS_TOLERANCE",
    "IterationTargetStatistics",
    "LearnerSequence",
    "Phase9MinibatchCursor",
    "Phase9RLExample",
    "Phase9TargetError",
    "audit_example",
    "behavior_action_logprob",
    "behavior_action_probability",
    "build_batch",
    "build_example",
    "build_sequence",
    "build_sequences",
    "collect_iteration_advantages",
    "epoch_order",
    "example_contract",
    "example_contract_digest",
    "examples_for_game",
    "is_learner_decision",
    "iter_rollout_examples",
    "iteration_statistics",
    "learner_decision_plies",
    "learner_players",
    "learner_side_name",
    "minibatch_keys",
    "minibatch_slices",
    "model_input_fields_only",
    "rollout_identity",
    "sequence_decision_count",
    "terminal_outcome",
    "train_order_keys",
    "validate_behavior_wdl",
    "validate_wdl_target",
    "verify_learner_decision_count",
]
