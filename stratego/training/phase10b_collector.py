"""Optional Phase 10B: the P10-D-conditioned self-play collector.

Specification source: `OPTIONAL_PHASE_10B_SETUP_CONDITIONED_FINE_TUNING_AGENT.md`
sections 5-8 and 24.

Reuse, not reimplementation
---------------------------
The game loop, the batching topology, the observer-safety boundary, the
trajectory builder and the crash-safe append-only store are the **accepted
Phase 9** ones, imported and subclassed rather than copied. Exactly two things
are Phase 10B's own, and both are required by the plan:

1. the action-sampling stream, which descends from the Phase 10B roots
   through the plan's `action_sampling` domain rather than from Phase 9's;
2. the rollout-id scheme, so a Phase 10B game can never be mistaken for — or
   silently mixed into — an accepted Phase 9 rollout.

Nothing here optimizes anything: there is no optimizer, no loss, no gradient
and no PPO in this module or anything it imports for collection. There is also
no search, which section 4 of the plan forbids outright.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .phase10b_contract import (
    BUCKET_CURRENT,
    PHASE10B_NAMESPACE,
    contract_digest as phase10b_contract_digest,
)
from .phase10b_schedule import (
    ActiveArchiveManifest,
    Phase10BScheduleError,
    iteration_game_ids,
    population_digest,
    rebuild_scheduled_game,
)
from .phase10b_contract import (
    PHASE10B_POPULATION_VERSION,
    PHASE10B_ROLLOUT_VERSION,
    PHASE10B_SCHEDULE_VERSION,
)
from .phase10b_seed import action_sampling_uniform, parse_game_id
from .phase10b_setup_source import validate_assignment_provenance
from .phase9_behavior import (
    BehaviorDecision,
    Phase9BehaviorError,
    behavior_distribution,
    evaluate_observations,
)
from .phase9_collector import (
    DEFAULT_GAMES_IN_FLIGHT,
    GameRunner,
    IterationParticipants,
    Phase9CollectorError,
    _drain_batches,
    acting_snapshot_for,
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
from .trajectory import DEFAULT_SNAPSHOT_INTERVAL
from .serialization import to_float32

PHASE10B_COLLECTOR_VERSION = "phase10b_collector_v1"


class Phase10BCollectorError(RuntimeError):
    """Raised when a scheduled Phase 10B game cannot be collected as specified."""


# ---------------------------------------------------------------------------
# The Phase 10B behavior decision
# ---------------------------------------------------------------------------


def select_action(probabilities, legal_absolute, game_id: str, ply: int) -> int:
    """The frozen cumulative-walk draw over the stored distribution.

    Identical in rule to the accepted Phase 9 sampler — walk ascending,
    accumulate, take the first action whose cumulative mass reaches the
    uniform, and let a float32 tail shortfall take the last legal action —
    with the uniform drawn from the Phase 10B `action_sampling` domain.
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
    """Assemble one stored Phase 10B neural decision from one batch row."""
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


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


class Phase10BGameRunner(GameRunner):
    """The accepted Phase 9 runner on the Phase 10B action-sampling stream.

    Everything about how a game is played, stored and validated is inherited.
    The single override is where the realized action comes from, which is the
    one place a Phase 9 seed would otherwise reach into a Phase 10B rollout.
    """

    def apply_neural(self, policy_logits_row, wdl_row) -> None:
        request = self.pending
        if request is None:  # pragma: no cover - the caller pairs these
            raise Phase10BCollectorError(f"{self.game_id}: no pending neural decision")
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
            raise Phase10BCollectorError(
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
        from ..engine.transition import apply_action

        apply_action(
            self.state,
            decision.selected_action_id,
            legal=list(decision.legal_action_ids),
        )


# ---------------------------------------------------------------------------
# Store integration
# ---------------------------------------------------------------------------


def validate_metadata(metadata: dict, record=None) -> list:
    """The accepted sidecar verification, on Phase 10B rollout ids."""
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
    """The Phase 10B sidecar of one collected game.

    The accepted builder owns the field set and the emission order; the one
    field it hard-codes for Phase 9 is the rollout version, which is rewritten
    here in place so the sidecar names the experiment that actually produced
    the game. Duplicating the field list to change one string would be the
    worse trade.
    """
    metadata = build_rollout_metadata(
        scheduled,
        record,
        setup_provenance=setup_provenance,
        behavior_checkpoint_sha256=behavior_checkpoint_sha256,
        opponent_checkpoint_sha256=opponent_checkpoint_sha256,
        learner_decision_count=learner_decision_count,
        population_version=PHASE10B_POPULATION_VERSION,
        schedule_version=PHASE10B_SCHEDULE_VERSION,
        contract_digest=phase10b_contract_digest(),
    )
    metadata["rollout_version"] = PHASE10B_ROLLOUT_VERSION
    return metadata


# ---------------------------------------------------------------------------
# Playing games
# ---------------------------------------------------------------------------


def collect_games(
    game_ids,
    participants: IterationParticipants,
    *,
    setup_source,
    history: ActiveArchiveManifest,
    games_in_flight: int = DEFAULT_GAMES_IN_FLIGHT,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    observer_probe_plies: int = 0,
):
    """Play a sequence of scheduled games, yielding finished runners in order.

    Games are started in schedule order and finished games are yielded as soon
    as they complete, so the caller commits them as they arrive rather than
    holding an iteration in memory. Lockstep batching is a topology choice and
    changes no logical identity: a decision is a pure function of
    `(game_id, ply)` and the acting checkpoint.
    """
    queue = list(game_ids)
    active: list = []
    cursor = 0
    while cursor < len(queue) or active:
        while len(active) < games_in_flight and cursor < len(queue):
            scheduled = rebuild_scheduled_game(
                queue[cursor], setup_source=setup_source, history=history
            )
            active.append(
                Phase10BGameRunner(
                    scheduled,
                    participants,
                    setup_source=setup_source,
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
                yield runner


def _opponent_digest(scheduled, participants: IterationParticipants) -> "str | None":
    if scheduled.opponent_kind != "historical_snapshot":
        return None
    snapshot = participants.historical_snapshot(scheduled.historical_snapshot_identity)
    recorded = scheduled.opponent_checkpoint_digest
    if recorded is not None and recorded != snapshot.checkpoint_sha256:
        raise Phase10BCollectorError(
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
    history: ActiveArchiveManifest,
    games_in_flight: int = DEFAULT_GAMES_IN_FLIGHT,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    observer_probe_plies: int = 0,
    fsync_on_commit: bool = False,
    limit: "int | None" = None,
    seal: bool = True,
    progress=None,
) -> dict:
    """Collect one iteration into the rollout store, resuming if bytes exist.

    Reconciles first so only committed games survive, subtracts them from the
    schedule, regenerates exactly what is missing, and — unless `limit` cut the
    work short — seals. Every call is safe to repeat.
    """
    root = Path(root)
    namespace = PHASE10B_NAMESPACE
    participants.behavior.assert_frozen()
    history.validate()

    state = read_iteration_state(root, namespace, iteration)
    if state is not None and state["state"] != "COLLECTING":
        reader = Phase9RolloutReader(root, namespace, iteration)
        return {
            "iteration": int(iteration),
            "state": state["state"],
            "already_sealed": True,
            "sealed_rollout_digest": state.get("sealed_rollout_digest"),
            "games_already_committed": len(reader),
            "games_collected": 0,
            "seconds": 0.0,
        }

    if state is not None:
        for key, current in (
            ("behavior_checkpoint_sha256", participants.behavior.checkpoint_sha256),
            ("behavior_snapshot_id", participants.behavior.logical_identity),
            ("inference_device", participants.behavior.device),
            ("inference_batch_shape", participants.behavior.inference_batch_shape),
        ):
            recorded = state.get(key)
            if recorded is not None and recorded != current:
                raise Phase10BCollectorError(
                    f"iteration {iteration} was collecting with {key}={recorded!r}; "
                    f"resuming with {current!r} would not converge to the same "
                    "sealed rollout digest"
                )

    write_iteration_state(
        root,
        namespace,
        iteration,
        "COLLECTING",
        collector_version=PHASE10B_COLLECTOR_VERSION,
        behavior_snapshot_id=participants.behavior.logical_identity,
        behavior_checkpoint_sha256=participants.behavior.checkpoint_sha256,
        inference_device=participants.behavior.device,
        inference_batch_shape=participants.behavior.inference_batch_shape,
    )
    reconciled = reconcile_iteration(root, namespace, iteration)
    committed = set(Phase9RolloutReader(root, namespace, iteration).commits)
    outstanding = [
        identifier
        for identifier in iteration_game_ids(iteration)
        if identifier not in committed
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
                history=history,
                games_in_flight=games_in_flight,
                snapshot_interval=snapshot_interval,
                observer_probe_plies=observer_probe_plies,
            ):
                scheduled = runner.scheduled
                provenance = runner.assignment.provenance
                problems = validate_assignment_provenance(provenance)
                if problems:
                    raise Phase10BCollectorError(
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
                probes += len(runner.observer_probes)
                probe_failures += sum(
                    1 for probe in runner.observer_probes if not probe.get("safe", True)
                )
                if progress is not None and collected % 128 == 0:
                    progress(collected, len(outstanding))
        finally:
            writer.close()

    seconds = time.perf_counter() - started
    summary = {
        "iteration": int(iteration),
        "collector_version": PHASE10B_COLLECTOR_VERSION,
        "behavior_snapshot_id": participants.behavior.logical_identity,
        "behavior_checkpoint_sha256": participants.behavior.checkpoint_sha256,
        "inference_device": participants.behavior.device,
        "inference_batch_shape": participants.behavior.inference_batch_shape,
        "games_already_committed": len(committed),
        "games_outstanding": len(outstanding),
        "games_collected": collected,
        "bucket_counts": buckets,
        "terminal_results": results,
        "total_decisions": decisions,
        "learner_decisions": learner_decisions,
        "total_plies": plies,
        "observer_probes": probes,
        "observer_probe_failures": probe_failures,
        "bytes_discarded_on_resume": reconciled.get("bytes_discarded", 0),
        "seconds": seconds,
        "games_per_second": collected / seconds if seconds > 0 else 0.0,
        "decisions_per_second": decisions / seconds if seconds > 0 else 0.0,
        "already_sealed": False,
    }
    if not seal or limit is not None:
        summary["state"] = "COLLECTING"
        summary["seal"] = None
        return summary

    sealed = seal_iteration(
        root,
        namespace,
        iteration,
        expected_behavior_checkpoint=participants.behavior.checkpoint_sha256,
        scheduled_game_ids=iteration_game_ids(iteration),
        metadata_validator=validate_metadata,
        manifest_extra={
            "collector_version": PHASE10B_COLLECTOR_VERSION,
            "population_version": PHASE10B_POPULATION_VERSION,
            "schedule_version": PHASE10B_SCHEDULE_VERSION,
            "population_digest": population_digest(),
            "contract_digest": phase10b_contract_digest(),
            "setup_source": setup_source.describe(),
            "active_archive": history.to_dict(),
        },
    )
    summary["seal"] = sealed
    summary["state"] = "SEALED" if sealed["sealed"] else "COLLECTING"
    summary["sealed_rollout_digest"] = sealed["sealed_rollout_digest"]
    if not sealed["sealed"]:
        raise Phase10BCollectorError(
            f"iteration {iteration} did not seal: {sealed['problems'][:3]}"
        )
    return summary


__all__ = [
    "PHASE10B_COLLECTOR_VERSION",
    "Phase10BCollectorError",
    "Phase10BGameRunner",
    "build_decision",
    "build_metadata",
    "collect_games",
    "collect_iteration",
    "select_action",
    "validate_metadata",
]
