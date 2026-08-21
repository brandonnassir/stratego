"""Phase 14: the population self-play collector.

Specification source: `02_AGENT_2_FINAL_TRAINING_INTEGRATION.md` sections 3 and
6, over the frozen `opponent_mixture` and `setup_source` blocks.

Reuse, not reimplementation
---------------------------
The game loop, the batching topology, the observer-safety boundary, the
trajectory builder and the crash-safe append-only store are the **accepted
Phase 9** ones, imported and subclassed rather than copied. Exactly two things
are Phase 14's own, and both are required:

1. the action-sampling stream, which descends from the Phase 14 roots rather
   than from Phase 9's;
2. the rollout-id scheme, so a Phase 14 game can never be mistaken for — or
   silently mixed into — an accepted Phase 9 or Phase 10B rollout.

Nothing here optimizes anything: there is no optimizer, no loss, no gradient
and no PPO in this module or anything it imports for collection. There is also
no search — not as an option, not behind a flag; the frozen contract puts it
outside Phase 14 training entirely.

Segment and pool are inputs
---------------------------
A collection unit is launched *in* a segment and *against* a pool. Both are
passed in and both are recorded in the iteration's state document, so a resume
regenerates the missing games of a crashed iteration under the composition that
iteration actually had, not under the composition the resume moment implies.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..engine.transition import apply_action
from .phase14_contract import (
    PHASE14_COLLECTOR_VERSION,
    PHASE14_NAMESPACE,
    PHASE14_POPULATION_VERSION,
    PHASE14_ROLLOUT_VERSION,
    PHASE14_SCHEDULE_VERSION,
    PRODUCTION_POPULATION,
    Population,
    contract_digest,
)
from .phase14_pool import ActivePool
from .phase14_schedule import iteration_game_ids, rebuild_scheduled_game
from .phase14_seed import action_sampling_uniform, parse_game_id
from .phase14_setup_source import validate_assignment_provenance
from .phase9_behavior import (
    DEFAULT_INFERENCE_BATCH_SHAPE,
    BehaviorDecision,
    Phase9BehaviorError,
    behavior_distribution,
    evaluate_observations,
)
from .phase9_collector import (
    DEFAULT_GAMES_IN_FLIGHT,
    GameRunner,
    IterationParticipants,
)
from .phase9_rollout_store import (
    Phase9RolloutReader,
    Phase9RolloutWriter,
    build_rollout_metadata,
    next_worker_id,
    read_iteration_state,
    reconcile_iteration,
    seal_iteration,
    validate_rollout_metadata,
    write_iteration_state,
)
from .serialization import to_float32
from .trajectory import DEFAULT_SNAPSHOT_INTERVAL


class Phase14CollectorError(RuntimeError):
    """Raised when a scheduled Phase 14 game cannot be collected as specified."""


# ---------------------------------------------------------------------------
# The Phase 14 behavior decision
# ---------------------------------------------------------------------------


def select_action(probabilities, legal_absolute, game_id: str, ply: int) -> int:
    """The frozen cumulative-walk draw over the stored distribution.

    Identical in rule to the accepted Phase 9 sampler — walk ascending,
    accumulate, take the first action whose cumulative mass reaches the
    uniform, and let a float32 tail shortfall take the last legal action — with
    the uniform drawn from the Phase 14 `action_sampling` domain.
    """
    actions = tuple(int(action) for action in legal_absolute)
    if len(actions) != len(probabilities):
        raise Phase9BehaviorError(
            f"{len(probabilities)} probabilities for {len(actions)} legal actions"
        )
    if list(actions) != sorted(actions):
        raise Phase9BehaviorError("the legal action list is not ascending")
    uniform = action_sampling_uniform(game_id, ply)
    cumulative = 0.0
    for action, probability in zip(actions, probabilities):
        cumulative += float(probability)
        if cumulative >= uniform:
            return int(action)
    return int(actions[-1])


def build_decision(
    snapshot, *, game_id: str, ply: int, legality, policy_logits_row, wdl_row
) -> BehaviorDecision:
    """Assemble one stored Phase 14 neural decision from one batch row."""
    probabilities = behavior_distribution(policy_logits_row, legality)
    selected = select_action(probabilities, legality.absolute, game_id, ply)
    if selected not in legality.absolute:  # pragma: no cover - selection is an index
        raise Phase9BehaviorError(
            f"{game_id} ply {ply}: selected action {selected} is not legal"
        )
    wdl = tuple(to_float32(float(value)) for value in np.asarray(wdl_row).reshape(3))
    return BehaviorDecision(
        game_id=str(game_id),
        ply=int(ply),
        acting_player=int(legality.acting_player),
        legal_action_ids=tuple(legality.absolute),
        probabilities=probabilities,
        win_draw_loss=wdl,
        selected_action_id=int(selected),
        policy_token=snapshot.policy_token,
        checkpoint_sha256=snapshot.checkpoint_sha256,
        snapshot_identity=snapshot.logical_identity,
    )


class Phase14GameRunner(GameRunner):
    """The accepted Phase 9 runner on the Phase 14 action-sampling stream.

    Everything about how a game is played, stored and validated is inherited.
    The single override is where the realized action comes from, which is the
    one place a Phase 9 seed would otherwise reach into a Phase 14 rollout.
    """

    def apply_neural(self, policy_logits_row, wdl_row) -> None:
        request = self.pending
        if request is None:  # pragma: no cover - the caller pairs these
            raise Phase14CollectorError(f"{self.game_id}: no pending neural decision")
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
            raise Phase14CollectorError(
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
        apply_action(
            self.state,
            decision.selected_action_id,
            legal=list(decision.legal_action_ids),
        )


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------


def resolve_pool_participants(
    pool: ActivePool,
    *,
    behavior,
    device: str = "cpu",
    inference_batch_shape: int = DEFAULT_INFERENCE_BATCH_SHAPE,
    resolver: "SnapshotResolver | None" = None,
) -> IterationParticipants:
    """Bind every active pool member to real, digest-checked weights.

    A logical identity is only allowed to name a checkpoint whose actual
    SHA-256 matches what the pool records, which is the guard against
    fabricating a snapshot to satisfy a schedule. Members that share a file
    share one loaded model — at iteration 1 the P9 anchor and the learner's own
    behavior snapshot are the same bytes — while staying two identities,
    because that distinction is what the store preserves.
    """
    from .phase14_checkpoint import Phase14SnapshotResolver
    from .phase14_contract import repository_root
    from .phase14_pool import historical_policy_token

    resolver = resolver or Phase14SnapshotResolver(
        device=device, inference_batch_shape=inference_batch_shape
    )
    historical: dict = {}
    for identity in pool.members():
        binding = pool.checkpoint_for(identity)
        path = Path(binding["path"])
        if not path.is_absolute():
            path = repository_root() / path
        historical[identity] = resolver.bind(
            path,
            logical_identity=identity,
            policy_token=historical_policy_token(identity),
            expected_sha256=binding["sha256"],
        )
    return IterationParticipants(behavior=behavior, historical=historical)


# ---------------------------------------------------------------------------
# Store integration
# ---------------------------------------------------------------------------


def validate_metadata(metadata: dict, record=None) -> list:
    """The accepted sidecar verification, on Phase 14 rollout ids."""
    problems = validate_rollout_metadata(metadata, record, id_parser=parse_game_id)
    provenance = metadata.get("setup_provenance")
    if isinstance(provenance, dict):
        problems.extend(validate_assignment_provenance(provenance))
    return problems


def build_metadata(
    scheduled,
    record,
    *,
    setup_provenance: dict,
    behavior_checkpoint_sha256: str,
    opponent_checkpoint_sha256: "str | None",
    learner_decision_count: int,
) -> dict:
    """The Phase 14 sidecar of one collected game.

    The accepted builder owns the field set and the emission order; the one
    field it hard-codes for Phase 9 is the rollout version, which is rewritten
    here in place so the sidecar names the experiment that actually produced the
    game. Duplicating the field list to change one string would be the worse
    trade.
    """
    metadata = build_rollout_metadata(
        scheduled,
        record,
        setup_provenance=setup_provenance,
        behavior_checkpoint_sha256=behavior_checkpoint_sha256,
        opponent_checkpoint_sha256=opponent_checkpoint_sha256,
        learner_decision_count=learner_decision_count,
        population_version=PHASE14_POPULATION_VERSION,
        schedule_version=PHASE14_SCHEDULE_VERSION,
        contract_digest=contract_digest(),
    )
    metadata["rollout_version"] = PHASE14_ROLLOUT_VERSION
    return metadata


# ---------------------------------------------------------------------------
# Playing games
# ---------------------------------------------------------------------------


def collect_games(
    game_ids,
    participants: IterationParticipants,
    *,
    setup_source,
    segment: str,
    pool: ActivePool,
    population: Population = PRODUCTION_POPULATION,
    games_in_flight: int = DEFAULT_GAMES_IN_FLIGHT,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    observer_probe_plies: int = 0,
    should_continue=None,
):
    """Play a sequence of scheduled games, yielding finished runners in order.

    Games are started in schedule order and finished games are yielded as soon
    as they complete, so the caller commits them as they arrive rather than
    holding an iteration in memory. Lockstep batching is a topology choice and
    changes no logical identity: a decision is a pure function of
    `(game_id, ply)` and the acting checkpoint.

    `should_continue` is the emergency-stop / deadline hook. It can only stop
    *starting* new games; the ones already in flight finish, because a
    half-played game is not a thing the store can commit.
    """
    queue = list(game_ids)
    active: list = []
    cursor = 0
    while cursor < len(queue) or active:
        admitting = should_continue is None or bool(should_continue())
        while admitting and len(active) < games_in_flight and cursor < len(queue):
            scheduled = rebuild_scheduled_game(
                queue[cursor],
                segment=segment,
                pool=pool,
                setup_source=setup_source,
                population=population,
            )
            active.append(
                Phase14GameRunner(
                    scheduled,
                    participants,
                    setup_source=setup_source,
                    behavior_checkpoint_sha256=participants.behavior.checkpoint_sha256,
                    snapshot_interval=snapshot_interval,
                    observer_probe_plies=observer_probe_plies,
                )
            )
            cursor += 1
        if not active:
            break

        pending = []
        finished = []
        for runner in active:
            request = runner.advance()
            if request is None:
                finished.append(runner)
            else:
                pending.append(request)

        from .phase9_collector import _drain_batches

        for batch in _drain_batches(pending):
            observations = np.stack([request.observation for request in batch])
            policy_logits, wdl = evaluate_observations(batch[0].snapshot, observations)
            for row, request in enumerate(batch):
                request.runner.apply_neural(policy_logits[row], wdl[row])

        if finished:
            active = [runner for runner in active if not runner.finished]
            for runner in finished:
                yield runner


def _opponent_digest(scheduled, participants: IterationParticipants) -> "str | None":
    if scheduled.opponent_kind != "historical_snapshot":
        return None
    snapshot = participants.historical_snapshot(scheduled.historical_snapshot_identity)
    recorded = scheduled.opponent_checkpoint_digest
    if recorded is not None and recorded != snapshot.checkpoint_sha256:
        raise Phase14CollectorError(
            f"{scheduled.rollout_game_id}: the schedule binds "
            f"{scheduled.historical_snapshot_identity!r} to {recorded}, but the "
            f"resolved snapshot is {snapshot.checkpoint_sha256}"
        )
    return snapshot.checkpoint_sha256


def collect_iteration(
    root,
    iteration: int,
    participants: IterationParticipants,
    *,
    setup_source,
    segment: str,
    pool: ActivePool,
    population: Population = PRODUCTION_POPULATION,
    games_in_flight: int = DEFAULT_GAMES_IN_FLIGHT,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    observer_probe_plies: int = 0,
    fsync_on_commit: bool = False,
    limit: "int | None" = None,
    seal: bool = True,
    progress=None,
    should_continue=None,
) -> dict:
    """Collect one iteration into the rollout store, resuming if bytes exist.

    Reconciles first so only committed games survive, subtracts them from the
    schedule, regenerates exactly what is missing, and — unless `limit` or the
    stop hook cut the work short — seals. Every call is safe to repeat.
    """
    root = Path(root)
    namespace = PHASE14_NAMESPACE
    participants.behavior.assert_frozen()
    pool.validate()

    state = read_iteration_state(root, namespace, iteration)
    if state is not None and state["state"] != "COLLECTING":
        reader = Phase9RolloutReader(root, namespace, iteration)
        return {
            "iteration": int(iteration),
            "segment": str(state.get("segment", segment)),
            "state": state["state"],
            "already_sealed": True,
            "sealed_rollout_digest": state.get("sealed_rollout_digest"),
            "games_already_committed": len(reader),
            "games_outstanding": 0,
            "games_collected": 0,
            "seconds": 0.0,
        }

    # A resume must reproduce bytes, not merely equivalent games: two devices
    # and two batch shapes agree to 1e-4 but not to the last float32 bit. The
    # segment and pool are on this list for the same reason — they decide who
    # played, and a mismatch would silently complete an iteration with a
    # different composition than the one it started with.
    if state is not None:
        for key, current in (
            ("behavior_checkpoint_sha256", participants.behavior.checkpoint_sha256),
            ("behavior_snapshot_id", participants.behavior.logical_identity),
            ("inference_device", participants.behavior.device),
            ("inference_batch_shape", participants.behavior.inference_batch_shape),
            ("segment", segment),
            ("active_pool_digest", pool.digest()),
            ("population_divisor", population.divisor),
        ):
            recorded = state.get(key)
            if recorded is not None and recorded != current:
                raise Phase14CollectorError(
                    f"iteration {iteration} was collecting with {key}={recorded!r}; "
                    f"resuming with {current!r} would not converge to the same "
                    "sealed rollout digest"
                )

    write_iteration_state(
        root,
        namespace,
        iteration,
        "COLLECTING",
        collector_version=PHASE14_COLLECTOR_VERSION,
        behavior_snapshot_id=participants.behavior.logical_identity,
        behavior_checkpoint_sha256=participants.behavior.checkpoint_sha256,
        inference_device=participants.behavior.device,
        inference_batch_shape=participants.behavior.inference_batch_shape,
        segment=segment,
        active_pool_digest=pool.digest(),
        active_pool_k=pool.k,
        population_divisor=population.divisor,
    )
    reconciled = reconcile_iteration(root, namespace, iteration)
    committed = set(Phase9RolloutReader(root, namespace, iteration).commits)
    scheduled_ids = iteration_game_ids(iteration, segment, population)
    outstanding = [
        identifier for identifier in scheduled_ids if identifier not in committed
    ]
    if limit is not None:
        outstanding = outstanding[: int(limit)]

    writer = None
    collected = 0
    decisions = 0
    learner_decisions = 0
    plies = 0
    buckets: dict = {}
    results: dict = {}
    members: dict = {}
    probe_failures = 0
    probes = 0
    started = time.perf_counter()
    if outstanding:
        writer = Phase9RolloutWriter(
            root,
            namespace=namespace,
            iteration=iteration,
            worker_id=next_worker_id(root, namespace, iteration),
            fsync_on_commit=fsync_on_commit,
            metadata_validator=validate_metadata,
        )
        try:
            for runner in collect_games(
                outstanding,
                participants,
                setup_source=setup_source,
                segment=segment,
                pool=pool,
                population=population,
                games_in_flight=games_in_flight,
                snapshot_interval=snapshot_interval,
                observer_probe_plies=observer_probe_plies,
                should_continue=should_continue,
            ):
                scheduled = runner.scheduled
                provenance = runner.assignment.provenance
                problems = validate_assignment_provenance(provenance)
                if problems:
                    raise Phase14CollectorError(
                        f"{scheduled.rollout_game_id}: setup provenance {problems[:3]}"
                    )
                metadata = build_metadata(
                    scheduled,
                    runner.record,
                    setup_provenance=provenance,
                    behavior_checkpoint_sha256=participants.behavior.checkpoint_sha256,
                    opponent_checkpoint_sha256=_opponent_digest(scheduled, participants),
                    learner_decision_count=runner.learner_decision_count,
                )
                writer.write_game(runner.record, metadata)
                collected += 1
                decisions += len(runner.record.decisions)
                learner_decisions += runner.learner_decision_count
                plies += int(runner.record.final_ply)
                buckets[scheduled.bucket] = buckets.get(scheduled.bucket, 0) + 1
                results[runner.record.terminal_result] = (
                    results.get(runner.record.terminal_result, 0) + 1
                )
                if scheduled.historical_snapshot_identity is not None:
                    identity = scheduled.historical_snapshot_identity
                    members[identity] = members.get(identity, 0) + 1
                probes += len(runner.observer_probes)
                probe_failures += sum(
                    1 for probe in runner.observer_probes if not probe.get("safe", True)
                )
                if progress is not None and collected % 128 == 0:
                    progress(collected, len(outstanding))
        finally:
            writer.close()

    seconds = time.perf_counter() - started
    complete = len(committed) + collected == len(scheduled_ids)
    summary = {
        "iteration": int(iteration),
        "segment": segment,
        "collector_version": PHASE14_COLLECTOR_VERSION,
        "behavior_snapshot_id": participants.behavior.logical_identity,
        "behavior_checkpoint_sha256": participants.behavior.checkpoint_sha256,
        "inference_device": participants.behavior.device,
        "inference_batch_shape": participants.behavior.inference_batch_shape,
        "active_pool_digest": pool.digest(),
        "active_pool_k": pool.k,
        "games_scheduled": len(scheduled_ids),
        "games_already_committed": len(committed),
        "games_outstanding": len(outstanding),
        "games_collected": collected,
        "bucket_counts": buckets,
        "terminal_results": results,
        "historical_member_games": members,
        "total_decisions": decisions,
        "learner_decisions": learner_decisions,
        "total_plies": plies,
        "observer_probes": probes,
        "observer_probe_failures": probe_failures,
        "reconciled": reconciled,
        "seconds": seconds,
        "games_per_second": (collected / seconds) if seconds > 0 else 0.0,
        "complete": complete,
        "writer_stats": None if writer is None else writer.stats(),
    }
    if seal and complete:
        from .phase14_schedule import population_digest

        summary["seal"] = seal_iteration(
            root,
            namespace,
            iteration,
            expected_behavior_checkpoint=participants.behavior.checkpoint_sha256,
            scheduled_game_ids=scheduled_ids,
            metadata_validator=validate_metadata,
            manifest_extra={
                "collector_version": PHASE14_COLLECTOR_VERSION,
                "population_version": PHASE14_POPULATION_VERSION,
                "schedule_version": PHASE14_SCHEDULE_VERSION,
                "population_digest": population_digest(),
                "contract_digest": contract_digest(),
                "segment": segment,
                "active_pool_digest": pool.digest(),
                "active_pool_k": pool.k,
            },
        )
        summary["sealed_rollout_digest"] = summary["seal"].get("sealed_rollout_digest")
        summary["sealed"] = bool(summary["seal"].get("sealed"))
    else:
        summary["sealed"] = False
    return summary


def collector_semantics() -> dict:
    return {
        "collector_version": PHASE14_COLLECTOR_VERSION,
        "namespace": PHASE14_NAMESPACE,
        "runner": "stratego.training.phase9_collector.GameRunner, subclassed",
        "distribution": "stratego.training.phase9_behavior.behavior_distribution",
        "store": "stratego.training.phase9_rollout_store, accepted, Phase 14 parser",
        "phase14_own": [
            "the action-sampling stream",
            "the rollout id scheme",
            "the segment/pool binding recorded in the iteration state",
        ],
        "search": "absent; no module under stratego.search is imported",
    }
