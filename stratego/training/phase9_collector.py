"""Phase 9 Agent 3: the production self-play collector.

Specification sources:

- `03_AGENT_3_SELFPLAY_COLLECTOR_AND_ROLLOUT_STORE.md` ("Neural action
  selection", "Rule/stress/historical actions", "Collection soak",
  "Observer-safety boundary")
- Agent 2's `phase9_schedule`: which logical games exist, who plays them, and
  which colours the learner controls
- Agent 1's `phase9_contract` / `phase9_behavior`: how a neural decision is
  made and stored

The collection boundary
-----------------------
This module's whole job is to make a later importance ratio trustworthy. That
reduces to three promises:

1. every current-policy decision in an iteration came from *one* immutable
   behavior snapshot;
2. every neural decision's stored distribution is the one that actually chose
   the move, reproducible from the acting side's own checkpoint;
3. no game becomes visible until its payload, metadata and commit all verify.

Nothing here optimizes anything. There is no optimizer, no loss, no gradient
and no PPO in this module or anything it imports for collection.

Per-side identity, not per-game identity
----------------------------------------
A historical matchup has two neural sides that are *different networks*. The
iteration's current snapshot goes in `GameRecord.collection_checkpoint_id`;
the archive member's real SHA-256 goes in the sidecar's
`opponent_checkpoint_sha256`; and each decision stores the acting side's own
policy token. An auditor that verified the whole game against the game-level
digest would be checking half the moves against the wrong network, so
:func:`acting_snapshot_for` is the only place a side's identity is resolved
and every consumer goes through it.

Throughput
----------
Games are run in lockstep so neural decisions from many games batch into one
forward pass. That is a topology choice, measured, and it changes no logical
identity: the action a decision takes is a pure function of `(game_id, ply)`
and the acting checkpoint, the forward pass is padded to a frozen batch shape
so a row's outputs never depend on its neighbours, and rule-side randomness
stays on the frozen per-decision Phase 4 stream.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..engine.constants import PLAYER_NAMES, PLAYERS
from ..engine.legal_moves import legal_action_mask, legal_actions
from ..engine.observation import build_observation
from ..engine.state import create_game
from ..engine.transition import apply_action
from ..evaluation.policy import (
    Policy,
    PolicyContractError,
    build_policy_input,
)
from ..evaluation.registry import POLICY_INDEX, build_policy, policy_ref
from ..model.policy_adapter import prepare_legality
from .phase9_behavior import (
    DEFAULT_INFERENCE_BATCH_SHAPE,
    BehaviorSnapshot,
    build_decision,
    evaluate_observations,
    load_behavior_snapshot,
)
from .phase9_rollout_store import (
    Phase9RolloutWriter,
    build_rollout_metadata,
    next_worker_id,
    pending_game_ids,
    read_iteration_state,
    reconcile_iteration,
    seal_iteration,
    write_iteration_state,
)
from .phase9_schedule import (
    SETUP_ENVIRONMENT_ID,
    SETUP_GENERATION,
    rebuild_scheduled_game,
)
from .rule_population import NEUTRAL_VALUE_PREDICTION
from .setup_source import training_setup_source
from .trajectory import (
    DEFAULT_SNAPSHOT_INTERVAL,
    GameTrajectoryBuilder,
    validate_game_record,
)
from .warmstart_contract import CORPUS_RULES, EXPECTED_SETUP_PROFILE

#: The collector implementation version. Not a contract identity — Agent 1
#: froze those — but recorded in the rollout manifest so a topology change is
#: visible in the artifacts rather than inferred.
PHASE9_COLLECTOR_VERSION = "phase9_collector_v1"

#: How many games are kept in flight so their neural decisions batch together.
#: Above the frozen inference batch shape, because at any instant some games
#: are on a rule-side move or have just finished.
DEFAULT_GAMES_IN_FLIGHT = 96


class Phase9CollectorError(RuntimeError):
    """Raised when a scheduled game cannot be collected as specified."""


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------


class SnapshotResolver:
    """Loads behavior snapshots once and shares identical weights.

    `H000` and a fresh run's `B001` are the same Phase 8 file, so they share
    one loaded model — but they stay two :class:`BehaviorSnapshot` objects
    with two logical identities, because that distinction is exactly what the
    store has to preserve.
    """

    def __init__(
        self,
        device: str = "cpu",
        inference_batch_shape: int = DEFAULT_INFERENCE_BATCH_SHAPE,
    ) -> None:
        self.device = device
        self.inference_batch_shape = int(inference_batch_shape)
        self._models: dict[str, object] = {}
        self._digests: dict[str, str] = {}
        self.load_count = 0

    def resolve(
        self,
        checkpoint_path,
        *,
        logical_identity: str,
        policy_token: str,
        expected_sha256: "str | None" = None,
    ) -> BehaviorSnapshot:
        key = str(Path(checkpoint_path).resolve())
        snapshot = load_behavior_snapshot(
            checkpoint_path,
            logical_identity=logical_identity,
            policy_token=policy_token,
            device=self.device,
            inference_batch_shape=self.inference_batch_shape,
            expected_sha256=expected_sha256,
            model=self._models.get(key),
            state_dict_digest_hint=self._digests.get(key),
        )
        if key not in self._models:
            self._models[key] = snapshot.model
            self._digests[key] = snapshot.loaded_state_dict_digest
            self.load_count += 1
        return snapshot


class RulePolicyCache:
    """One instance per frozen Phase 4 policy id, built on first use."""

    def __init__(self) -> None:
        self._policies: dict[str, Policy] = {}

    def get(self, policy_id: str) -> Policy:
        if policy_id not in POLICY_INDEX:
            raise Phase9CollectorError(f"unknown frozen policy id: {policy_id!r}")
        if policy_id not in self._policies:
            self._policies[policy_id] = build_policy(policy_id)
        return self._policies[policy_id]


@dataclass(frozen=True)
class IterationParticipants:
    """Everything one iteration needs to actually play its scheduled games.

    `behavior` is the iteration's single current-policy snapshot; `historical`
    maps an active archive identity to the real immutable checkpoint bound to
    it. An identity with no entry here cannot be collected — which is the
    guard against fabricating a future checkpoint to satisfy a schedule.
    """

    behavior: BehaviorSnapshot
    historical: dict
    rules: RulePolicyCache = field(default_factory=RulePolicyCache)

    def historical_snapshot(self, identity: str) -> BehaviorSnapshot:
        snapshot = self.historical.get(identity)
        if snapshot is None:
            raise Phase9CollectorError(
                f"historical identity {identity!r} has no bound checkpoint; a "
                "scheduled archive member must be a real immutable checkpoint "
                "before its games can be collected"
            )
        return snapshot


def acting_snapshot_for(
    scheduled, participants: IterationParticipants, acting_player: int
) -> "BehaviorSnapshot | None":
    """The snapshot that owns one side's decisions, or `None` for rule/stress.

    The single resolution point for "whose network is this move?". Everything
    that stores, verifies or reproduces a decision asks here, so no consumer
    can accidentally attribute a historical opponent's move to the current
    learner.
    """
    colour = PLAYER_NAMES[acting_player]
    learner_side = scheduled.learner_color is None or scheduled.learner_color == colour
    if learner_side:
        return participants.behavior
    if scheduled.opponent_kind == "historical_snapshot":
        return participants.historical_snapshot(scheduled.historical_snapshot_identity)
    if scheduled.opponent_kind == "current_policy":  # pragma: no cover - colour is None
        return participants.behavior
    return None


def _opponent_policy_id(scheduled) -> str:
    """The frozen Phase 4 policy id behind a rule/stress opponent token."""
    token = scheduled.opponent_identity
    policy_id = token.split("@", 1)[0]
    if policy_id not in POLICY_INDEX:
        raise Phase9CollectorError(f"opponent token {token!r} names no frozen policy")
    return policy_id


# ---------------------------------------------------------------------------
# Observer safety
# ---------------------------------------------------------------------------


def observer_safety_probe(state, observer: int, observation, *, builder=None) -> dict:
    """Prove one model input carries no privileged information.

    Two genuinely independent checks, because there are two ways privileged
    truth reaches a network:

    - *the array is not what it claims to be* — it is compared against what
      `builder` produces for this observer, and must own its own memory rather
      than aliasing engine state. A caller that hand-edited a channel fails
      here.
    - *the builder itself leaks* — permuting the true types of every opponent
      piece the observer may not legally know must leave the built observation
      bitwise unchanged. An observation function that encoded a hidden identity
      anywhere, in a channel or a count or a derived feature, moves under that
      permutation.

    `builder` defaults to the frozen engine observation and exists so the
    second check can be pointed at a deliberately leaking builder: an audit
    that cannot be made to fail is not evidence of anything.
    """
    builder = build_observation if builder is None else builder
    array = np.asarray(observation)
    problems: list[str] = []
    rebuilt = builder(state, observer)
    if array.shape != rebuilt.shape or not np.array_equal(array, rebuilt):
        problems.append("the model input is not the observer-safe observation")
    if array.base is not None:
        problems.append("the model input aliases another array")

    counterfactual = copy.deepcopy(state)
    hidden = [
        record
        for record in counterfactual.pieces
        if record.owner != observer and not record.known_to(observer)
    ]
    types = [record.true_type for record in hidden]
    permuted = types[1:] + types[:1] if len(set(types)) > 1 else types
    for record, true_type in zip(hidden, permuted):
        record.true_type = true_type
    shuffled = builder(counterfactual, observer)
    leaked = int(np.count_nonzero(shuffled != rebuilt))
    if leaked:
        problems.append(
            f"{leaked} observation entries changed when hidden opponent types were "
            "permuted; the input encodes privileged truth"
        )
    return {
        "observer": int(observer),
        "hidden_opponent_pieces": len(hidden),
        "permutation_applied": len(set(types)) > 1,
        "entries_sensitive_to_hidden_truth": leaked,
        "safe": not problems,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# One game in flight
# ---------------------------------------------------------------------------


@dataclass
class NeuralRequest:
    """One pending neural decision, waiting for its batch."""

    runner: "GameRunner"
    snapshot: BehaviorSnapshot
    observation: np.ndarray
    legality: object
    ply: int


class GameRunner:
    """One scheduled logical game, played incrementally.

    :meth:`advance` plays every decision that needs no forward pass and then
    stops, either finished or holding one pending neural request. Running many
    of these in lockstep is what lets neural decisions batch without changing
    a single logical identity.
    """

    def __init__(
        self,
        scheduled,
        participants: IterationParticipants,
        *,
        setup_source,
        behavior_checkpoint_sha256: str,
        snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
        observer_probe_plies: int = 0,
    ) -> None:
        self.scheduled = scheduled
        self.participants = participants
        self.game_id = scheduled.phase9_game_id
        self.observer_probe_plies = int(observer_probe_plies)
        self.observer_probes: list[dict] = []

        self.assignment = setup_source.assign(
            root_seed=scheduled.setup_root_seed,
            environment_id=SETUP_ENVIRONMENT_ID,
            generation=SETUP_GENERATION,
            game_id=self.game_id,
        )
        if self.assignment.provenance is None:  # pragma: no cover - library always emits
            raise Phase9CollectorError(f"{self.game_id}: the setup source emitted no provenance")

        self.state = create_game(
            self.assignment.red_setup,
            self.assignment.blue_setup,
            rules=CORPUS_RULES,
            game_id=self.game_id,
        )
        self.builder = GameTrajectoryBuilder(
            game_id=self.game_id,
            environment_id=SETUP_ENVIRONMENT_ID,
            generation=SETUP_GENERATION,
            red_setup=self.assignment.red_setup,
            blue_setup=self.assignment.blue_setup,
            rules=CORPUS_RULES,
            root_seed=int(scheduled.setup_root_seed),
            slot_seed=0,
            snapshot_interval=snapshot_interval,
            collection_policy_version=scheduled.behavior_snapshot_identity,
            collection_checkpoint_id=behavior_checkpoint_sha256,
            setup_family=setup_source.setup_family,
        )
        self.learner_decision_count = 0
        self.neural_decision_count = 0
        self.learner_neural_decision_count = 0
        self.record = None
        self.pending: "NeuralRequest | None" = None

    # -- helpers -----------------------------------------------------------

    @property
    def finished(self) -> bool:
        return self.record is not None

    def _side_token(self, acting_player: int) -> str:
        if acting_player == PLAYERS[0]:
            return self.scheduled.red_policy_identity
        return self.scheduled.blue_policy_identity

    def _is_learner(self, acting_player: int) -> bool:
        return PLAYER_NAMES[acting_player] in self.scheduled.learner_sides

    def _rule_decision(self, legal, mask) -> None:
        """One frozen Phase 4 rule/stress decision, stored as the accepted one-hot."""
        actor = self.state.acting_player
        policy = self.participants.rules.get(_opponent_policy_id(self.scheduled))
        seed = (
            self.scheduled.red_policy_seed
            if actor == PLAYERS[0]
            else self.scheduled.blue_policy_seed
        )
        if seed is None:
            raise Phase9CollectorError(
                f"{self.game_id}: the rule/stress side has no match-level policy seed"
            )
        request = build_policy_input(
            self.state,
            policy=policy_ref(policy.policy_id),
            policy_seed=int(seed),
            requirements=policy.requirements,
            game_id=self.game_id,
            legal=list(legal),
        )
        try:
            result = policy.decide_checked(request)
        except PolicyContractError as error:
            raise Phase9CollectorError(
                f"{self.game_id}: policy {policy.policy_id} violated its contract at "
                f"ply {request.ply}: {error}"
            ) from error
        selected = int(result.selected_action_id)
        self.builder.record_decision(
            self.state,
            legal_action_ids=legal,
            probabilities=tuple(1.0 if action == selected else 0.0 for action in legal),
            win_draw_loss_prediction=NEUTRAL_VALUE_PREDICTION,
            selected_action_id=selected,
            collection_policy_version=self._side_token(actor),
        )
        if self._is_learner(actor):  # pragma: no cover - a rule side is never the learner
            self.learner_decision_count += 1
        apply_action(self.state, selected, legal=list(legal))

    def _finish(self) -> None:
        record = self.builder.finish(self.state)
        problems = validate_game_record(record)
        if problems:
            raise Phase9CollectorError(f"{self.game_id}: sealed trajectory invalid: {problems}")
        self.record = record

    # -- the loop ----------------------------------------------------------

    def advance(self) -> "NeuralRequest | None":
        """Play up to the next neural decision. Returns it, or `None` if done."""
        if self.record is not None:
            return None
        if self.pending is not None:
            return self.pending
        while not self.state.terminal:
            legal = legal_actions(self.state)
            if not legal:  # pragma: no cover - the engine terminates such a state
                raise Phase9CollectorError(f"{self.game_id}: no legal actions in a live state")
            actor = self.state.acting_player
            snapshot = acting_snapshot_for(self.scheduled, self.participants, actor)
            mask = legal_action_mask(self.state, legal)
            if snapshot is None:
                self._rule_decision(legal, mask)
                continue
            legality = prepare_legality(legal, mask, actor)
            observation = build_observation(self.state, actor)
            if len(self.observer_probes) < self.observer_probe_plies:
                self.observer_probes.append(
                    observer_safety_probe(self.state, actor, observation)
                )
            self.pending = NeuralRequest(
                runner=self,
                snapshot=snapshot,
                observation=observation,
                legality=legality,
                ply=self.state.total_moves,
            )
            return self.pending
        self._finish()
        return None

    def apply_neural(self, policy_logits_row, wdl_row) -> None:
        """Store and play the pending neural decision from its batch row."""
        request = self.pending
        if request is None:  # pragma: no cover - the caller pairs these
            raise Phase9CollectorError(f"{self.game_id}: no pending neural decision")
        decision = build_decision(
            request.snapshot,
            game_id=self.game_id,
            ply=request.ply,
            legality=request.legality,
            policy_logits_row=policy_logits_row,
            wdl_row=wdl_row,
        )
        actor = request.legality.acting_player
        expected_token = self._side_token(actor)
        if decision.policy_token != expected_token:
            raise Phase9CollectorError(
                f"{self.game_id} ply {request.ply}: acting snapshot token "
                f"{decision.policy_token!r} is not the scheduled {expected_token!r}"
            )
        self.builder.record_decision(
            self.state,
            legal_action_ids=decision.legal_action_ids,
            probabilities=decision.probabilities,
            win_draw_loss_prediction=decision.win_draw_loss,
            selected_action_id=decision.selected_action_id,
            collection_policy_version=expected_token,
        )
        self.neural_decision_count += 1
        if self._is_learner(actor):
            self.learner_decision_count += 1
            self.learner_neural_decision_count += 1
        self.pending = None
        apply_action(self.state, decision.selected_action_id, legal=list(decision.legal_action_ids))


# ---------------------------------------------------------------------------
# Playing a whole iteration
# ---------------------------------------------------------------------------


def play_game(
    scheduled,
    participants: IterationParticipants,
    *,
    setup_source=None,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    observer_probe_plies: int = 0,
) -> GameRunner:
    """Play exactly one scheduled game, serially. The reference implementation.

    The batched runner below must agree with this move for move; the tests
    assert that rather than assume it.
    """
    source = training_setup_source(EXPECTED_SETUP_PROFILE) if setup_source is None else setup_source
    runner = GameRunner(
        scheduled,
        participants,
        setup_source=source,
        behavior_checkpoint_sha256=participants.behavior.checkpoint_sha256,
        snapshot_interval=snapshot_interval,
        observer_probe_plies=observer_probe_plies,
    )
    while True:
        request = runner.advance()
        if request is None:
            break
        policy_logits, wdl = evaluate_observations(
            request.snapshot, request.observation[None, ...]
        )
        runner.apply_neural(policy_logits[0], wdl[0])
    return runner


def _drain_batches(pending: list) -> list:
    """Group pending neural requests into forward passes by acting checkpoint.

    Two snapshots that share a checkpoint digest share their weights, so their
    decisions belong in the same batch; two that do not, never can.
    """
    grouped: dict[str, list] = {}
    for request in pending:
        grouped.setdefault(request.snapshot.checkpoint_sha256, []).append(request)
    batches = []
    for requests in grouped.values():
        shape = requests[0].snapshot.inference_batch_shape
        for start in range(0, len(requests), shape):
            batches.append(requests[start : start + shape])
    return batches


def collect_games(
    game_ids,
    participants: IterationParticipants,
    *,
    setup_source=None,
    games_in_flight: int = DEFAULT_GAMES_IN_FLIGHT,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    observer_probe_plies: int = 0,
    history=None,
    on_game=None,
):
    """Play a sequence of scheduled games, yielding finished runners in order.

    Games are started in schedule order and finished games are yielded as soon
    as they complete, so the caller commits them as they arrive rather than
    holding an iteration in memory.
    """
    source = training_setup_source(EXPECTED_SETUP_PROFILE) if setup_source is None else setup_source
    queue = list(game_ids)
    active: list[GameRunner] = []
    cursor = 0
    while cursor < len(queue) or active:
        while len(active) < games_in_flight and cursor < len(queue):
            scheduled = rebuild_scheduled_game(queue[cursor], history=history)
            active.append(
                GameRunner(
                    scheduled,
                    participants,
                    setup_source=source,
                    behavior_checkpoint_sha256=participants.behavior.checkpoint_sha256,
                    snapshot_interval=snapshot_interval,
                    observer_probe_plies=observer_probe_plies,
                )
            )
            cursor += 1

        pending = []
        finished = []
        for runner in active:
            request = runner.advance()
            if request is None:
                finished.append(runner)
            else:
                pending.append(request)

        for batch in _drain_batches(pending):
            observations = np.stack([request.observation for request in batch])
            policy_logits, wdl = evaluate_observations(batch[0].snapshot, observations)
            for row, request in enumerate(batch):
                request.runner.apply_neural(policy_logits[row], wdl[row])

        if finished:
            active = [runner for runner in active if not runner.finished]
            for runner in finished:
                if on_game is not None:
                    on_game(runner)
                yield runner


# ---------------------------------------------------------------------------
# The iteration driver
# ---------------------------------------------------------------------------


def _sealed_summary(root, namespace, iteration, state, participants) -> dict:
    """The summary of an iteration that is already past COLLECTING.

    Read back from the sealed bytes rather than reconstructed from memory, so
    the counts are the store's own and a re-run reports what is actually there.
    """
    from .phase9_rollout_store import Phase9RolloutReader

    reader = Phase9RolloutReader(root, namespace, iteration)
    buckets: dict[str, int] = {}
    results: dict[str, int] = {}
    decisions = 0
    learner = 0
    plies = 0
    for game_id, commit in reader.commits.items():
        metadata = reader.metadata.get(game_id, {})
        bucket = metadata.get("bucket", "unknown")
        buckets[bucket] = buckets.get(bucket, 0) + 1
        result = metadata.get("terminal_result", "unknown")
        results[result] = results.get(result, 0) + 1
        decisions += commit.total_decisions
        learner += commit.learner_decision_count
        plies += commit.final_ply
    return {
        "namespace": namespace,
        "iteration": int(iteration),
        "state": state["state"],
        "already_sealed": True,
        "sealed": state["state"] != "COLLECTING",
        "sealed_rollout_digest": state.get("sealed_rollout_digest"),
        "collector_version": state.get("collector_version", PHASE9_COLLECTOR_VERSION),
        "behavior_snapshot_id": state.get("behavior_snapshot_id"),
        "behavior_checkpoint_sha256": state.get("behavior_checkpoint_sha256"),
        "inference_device": state.get("inference_device", participants.behavior.device),
        "inference_batch_shape": state.get(
            "inference_batch_shape", participants.behavior.inference_batch_shape
        ),
        "games_already_committed": len(reader),
        "games_outstanding": 0,
        "games_collected": 0,
        "bucket_counts": buckets,
        "terminal_results": results,
        "total_decisions": decisions,
        "neural_decisions": 0,
        "learner_decisions": learner,
        "total_plies": plies,
        "seconds": 0.0,
        "games_per_second": 0.0,
        "decisions_per_second": 0.0,
        "positions_per_second": 0.0,
        "bytes_discarded_on_resume": 0,
        "writer_stats": None,
        "observer_probes": 0,
        "observer_probe_failures": 0,
        "seal": {
            "namespace": namespace,
            "iteration": int(iteration),
            "sealed": state["state"] != "COLLECTING",
            "sealed_rollout_digest": state.get("sealed_rollout_digest"),
            "scheduled_games": len(reader),
            "committed_games": len(reader),
            "missing_games": 0,
            "unscheduled_games": 0,
            "duplicate_game_ids": 0,
            "orphan_records": 0,
            "behavior_snapshot_identities": [state["behavior_snapshot_id"]]
            if state.get("behavior_snapshot_id")
            else [],
            "behavior_checkpoint_digests": [state["behavior_checkpoint_sha256"]]
            if state.get("behavior_checkpoint_sha256")
            else [],
            "total_decisions": decisions,
            "learner_decision_count": learner,
            "problems": [],
        },
    }


def collect_iteration(
    root,
    namespace: str,
    iteration: int,
    participants: IterationParticipants,
    *,
    population_version: str,
    schedule_version: str,
    contract_digest: str,
    games_in_flight: int = DEFAULT_GAMES_IN_FLIGHT,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    observer_probe_plies: int = 0,
    target_bytes: int = None,
    fsync_on_commit: bool = False,
    crash_hook=None,
    history=None,
    limit: "int | None" = None,
    seal: bool = True,
    progress=None,
) -> dict:
    """Collect one iteration into the rollout store, resuming if bytes exist.

    Reconciles first (so only committed games survive), subtracts them from the
    schedule, regenerates exactly what is missing, and — unless `limit` cut the
    work short — seals. Every call is safe to repeat: a fully collected
    iteration re-seals to the same digest and plays nothing.
    """
    root = Path(root)
    participants.behavior.assert_frozen()
    state = read_iteration_state(root, namespace, iteration)
    if state is not None and state["state"] != "COLLECTING":
        # Already past COLLECTING: play nothing, and report the same shape a
        # fresh collection does so a caller aggregating several iterations does
        # not have to special-case the resumed one.
        return _sealed_summary(root, namespace, iteration, state, participants)

    # A resume must reproduce bytes, not merely equivalent games. Two devices
    # and two batch shapes agree to 1e-4 but not to the last float32 bit, so a
    # partially collected iteration may only be finished under the conditions
    # that wrote its committed games.
    if state is not None:
        for key, current in (
            ("behavior_checkpoint_sha256", participants.behavior.checkpoint_sha256),
            ("behavior_snapshot_id", participants.behavior.logical_identity),
            ("inference_device", participants.behavior.device),
            ("inference_batch_shape", participants.behavior.inference_batch_shape),
        ):
            recorded = state.get(key)
            if recorded is not None and recorded != current:
                raise Phase9CollectorError(
                    f"{namespace} iteration {iteration} was collecting with "
                    f"{key}={recorded!r}; resuming with {current!r} would not "
                    "converge to the same sealed rollout digest"
                )

    write_iteration_state(
        root,
        namespace,
        iteration,
        "COLLECTING",
        behavior_snapshot_id=participants.behavior.logical_identity,
        behavior_checkpoint_sha256=participants.behavior.checkpoint_sha256,
        inference_device=participants.behavior.device,
        inference_batch_shape=participants.behavior.inference_batch_shape,
        collector_version=PHASE9_COLLECTOR_VERSION,
    )
    reconciled = reconcile_iteration(root, namespace, iteration)
    outstanding = pending_game_ids(root, namespace, iteration)
    if limit is not None:
        outstanding = outstanding[: int(limit)]

    started = time.perf_counter()
    writer = None
    collected = 0
    decisions = 0
    neural_decisions = 0
    learner_decisions = 0
    plies = 0
    observer_probes: list[dict] = []
    buckets: dict[str, int] = {}
    results: dict[str, int] = {}
    if outstanding:
        writer = Phase9RolloutWriter(
            root,
            namespace=namespace,
            iteration=iteration,
            worker_id=next_worker_id(root, namespace, iteration),
            fsync_on_commit=fsync_on_commit,
            crash_hook=crash_hook,
            **({} if target_bytes is None else {"target_bytes": target_bytes}),
        )
        try:
            for runner in collect_games(
                outstanding,
                participants,
                games_in_flight=games_in_flight,
                snapshot_interval=snapshot_interval,
                observer_probe_plies=observer_probe_plies,
                history=history,
            ):
                scheduled = runner.scheduled
                opponent_digest = None
                if scheduled.opponent_kind == "historical_snapshot":
                    opponent_digest = participants.historical_snapshot(
                        scheduled.historical_snapshot_identity
                    ).checkpoint_sha256
                metadata = build_rollout_metadata(
                    scheduled,
                    runner.record,
                    setup_provenance=runner.assignment.provenance,
                    behavior_checkpoint_sha256=participants.behavior.checkpoint_sha256,
                    opponent_checkpoint_sha256=opponent_digest,
                    learner_decision_count=runner.learner_decision_count,
                    population_version=population_version,
                    schedule_version=schedule_version,
                    contract_digest=contract_digest,
                )
                writer.write_game(runner.record, metadata)
                collected += 1
                decisions += len(runner.record.decisions)
                neural_decisions += runner.neural_decision_count
                learner_decisions += runner.learner_decision_count
                plies += int(runner.record.final_ply)
                buckets[scheduled.bucket] = buckets.get(scheduled.bucket, 0) + 1
                results[runner.record.terminal_result] = (
                    results.get(runner.record.terminal_result, 0) + 1
                )
                observer_probes.extend(runner.observer_probes)
                if progress is not None and collected % max(1, len(outstanding) // 20) == 0:
                    progress(collected, len(outstanding))
        finally:
            writer.close()

    participants.behavior.assert_frozen()
    elapsed = time.perf_counter() - started
    summary = {
        "namespace": namespace,
        "iteration": int(iteration),
        "collector_version": PHASE9_COLLECTOR_VERSION,
        "behavior_snapshot_id": participants.behavior.logical_identity,
        "behavior_checkpoint_sha256": participants.behavior.checkpoint_sha256,
        "inference_device": participants.behavior.device,
        "games_in_flight": int(games_in_flight),
        "inference_batch_shape": participants.behavior.inference_batch_shape,
        "games_already_committed": len(reconciled["committed_game_ids"]),
        "games_outstanding": len(outstanding),
        "games_collected": collected,
        "bucket_counts": buckets,
        "terminal_results": results,
        "total_decisions": decisions,
        "neural_decisions": neural_decisions,
        "learner_decisions": learner_decisions,
        "total_plies": plies,
        "seconds": elapsed,
        "games_per_second": collected / elapsed if elapsed else 0.0,
        "decisions_per_second": decisions / elapsed if elapsed else 0.0,
        "positions_per_second": plies / elapsed if elapsed else 0.0,
        "bytes_discarded_on_resume": reconciled["bytes_discarded"],
        "writer_stats": writer.stats() if writer is not None else None,
        "observer_probes": len(observer_probes),
        "observer_probe_failures": sum(
            0 if probe["safe"] else 1 for probe in observer_probes
        ),
    }
    if seal:
        summary["seal"] = seal_iteration(
            root,
            namespace,
            iteration,
            expected_behavior_checkpoint=participants.behavior.checkpoint_sha256,
            manifest_extra={
                "collector_version": PHASE9_COLLECTOR_VERSION,
                "inference_device": participants.behavior.device,
                "inference_batch_shape": participants.behavior.inference_batch_shape,
                "behavior_snapshot_id": participants.behavior.logical_identity,
                "population_version": population_version,
                "schedule_version": schedule_version,
                "contract_digest": contract_digest,
            },
        )
        summary["sealed_rollout_digest"] = summary["seal"]["sealed_rollout_digest"]
        summary["sealed"] = summary["seal"]["sealed"]
    return summary


__all__ = [
    "DEFAULT_GAMES_IN_FLIGHT",
    "PHASE9_COLLECTOR_VERSION",
    "GameRunner",
    "IterationParticipants",
    "NeuralRequest",
    "Phase9CollectorError",
    "RulePolicyCache",
    "SnapshotResolver",
    "acting_snapshot_for",
    "collect_games",
    "collect_iteration",
    "observer_safety_probe",
    "play_game",
]
