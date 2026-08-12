"""Phase 6 Agent 4: real candidates inside the accepted Phase 3 pipeline.

Specification source: `04_AGENT_4_INTEGRATED_PIPELINE.md`.

What this module measures
-------------------------
Agent 3 measured each candidate alone on Metal, with no simulator, no shared
memory and no trajectory. That is the hardware frontier, not the pipeline's. This
module puts the shortlisted `stratego_transformer_v1` candidates into the
bulk-synchronous coordinator that Phase 3 accepted and measures what the whole
thing sustains:

1. an end-to-end correctness gate for `model_contract_v2` -- absolute legality
   in, normalized legality, normalized selection, absolute action out, engine
   validation -- run against independently rebuilt reference games;
2. collection-only throughput, with trajectory persistence off;
3. production-recording throughput, on the real compact `trajectory_v1` path at
   snapshot interval 32;
4. reconstruction of stored decisions, including the two normalized products the
   record deliberately does not store;
5. the trajectory storage rate, and the 24/168-hour projections that follow;
6. a per-candidate bottleneck ratio `R`, with both sides measured here.

What it does not do
-------------------
It does not choose the primary model, it does not train anything, and it never
uses playing strength as evidence. Candidate weights are the family's fixed
initialization: they are random, and a random network's results are used only as
a *cost* measurement.

Where the frame conversion lives
--------------------------------
Not here. `stratego.training.coordinator.NormalizedActionFrame` performs it, on
the device, inside the same `_run_chunk` a production step calls, and it builds
its tables from `stratego.model.action_frame`. This module drives that path and
checks it against the frozen engine; it does not re-implement it, so a bug in
the conversion cannot be hidden by a matching bug in the checker.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..engine.constants import ACTION_SPACE_SIZE, PLAYERS, RulesConfig, TRAINING_RULES
from ..engine.legal_moves import legal_action_mask, legal_actions
from ..engine.observation import build_observation
from ..model.action_frame import (
    absolute_action_to_model,
    absolute_legal_actions_to_model,
    absolute_legal_mask_to_model,
    model_action_to_absolute,
    model_legal_actions_to_absolute,
)
from ..model.architecture_configs import (
    ARCHITECTURE_FAMILY,
    ARCHITECTURE_FAMILY_VERSION,
    FAMILY_INITIALIZATION_SEED,
)
from ..model.contract import MODEL_CONTRACT_VERSION, ModelContractError
from ..model.production_model import ProductionModel, build_candidate_model
from .coordinator import (
    ACTION_FRAME_NORMALIZED,
    ActionFrameMismatchError,
    CoordinatorConfig,
    CoordinatorError,
    RunTotals,
    SelfPlayCoordinator,
    compact_legality_from_masks,
    mps_memory_bytes,
)
from .end_to_end_benchmark import (
    ReferenceMirror,
    compute_ratio,
    measure_simulation_pipeline,
    swap_bytes,
)
from .mps_benchmark import peak_memory_bytes
from .reconstruction import iter_reconstructed_decisions
from .shared_buffers import POLICY_CAPACITY, STATUS_ACTIVE
from .trajectory import (
    TRAJECTORY_VERSION,
    decode_game_record,
    encode_game_record,
    validate_game_record,
)
from .worker_pool import WorkerPoolError

BENCHMARK_VERSION = "agent_04_integrated_pipeline_0.1.0"

#: Stamped into every decision this agent records. Distinct from Agent 3's
#: `synthetic_hash_policy_v1` and from Phase 3's
#: `end_to_end_representative_probe_v1`, because these decisions come from a real
#: `stratego_transformer_v1` candidate in the normalized frame and a training
#: consumer must be able to tell the three corpora apart. The candidate id and
#: precision are appended, so two candidates never share a corpus label.
COLLECTION_POLICY_PREFIX = "phase6_candidate"

#: The Phase 3 accepted starting topology, which this agent begins from.
STARTING_WORKERS = 10
STARTING_ENVIRONMENTS = 1536
STARTING_SNAPSHOT_INTERVAL = 32
STARTING_PRECISION = "float16"
STARTING_LEGALITY = "dense"

#: The environment/inference-batch points the instructions name.
SWEEP_POINTS = (512, 1024, 1536, 2048)

MODE_COLLECTION = "collection_only"
MODE_RECORDING = "production_recording"

BYTES_PER_GIB = 1024**3

#: Free space the storage analysis compares a 168-hour run against. Stated as
#: constants rather than probed, because the comparison the instructions ask for
#: is against the user's declared capacity, not against today's free space.
INTERNAL_FREE_BYTES = 150 * 1000**3
EXTERNAL_FREE_BYTES = 1000 * 1000**3

#: Re-evaluating a stored decision through the same candidate cannot be
#: bit-exact: the record was produced inside a batch of up to 2,048 rows and the
#: check runs a batch of one, and float16 reductions are not associative. This is
#: the deviation above which the recorded distribution is not the candidate's.
POLICY_REEVALUATION_TOLERANCE = 0.02


def collection_policy_version(candidate_id: str, precision: str) -> str:
    """The `collection_policy_version` stamped into this candidate's records."""
    return f"{COLLECTION_POLICY_PREFIX}_{candidate_id.lower()}_{precision}_v1"


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def build_pipeline_candidate(
    candidate_id: str, *, seed: int = FAMILY_INITIALIZATION_SEED
) -> ProductionModel:
    """Reconstruct one of Agent 3's advancing candidates, exactly.

    Built on CPU in float32 from `(candidate id, family seed)` alone, which is
    what Agent 2 made sufficient; the coordinator moves and casts it. Nothing
    here may adjust an architecture -- Agent 4 is forbidden from modifying the
    family, and a candidate that differed from the one Agent 3 benchmarked would
    make the two sets of measurements incomparable.
    """
    return build_candidate_model(candidate_id, seed=seed, device="cpu", dtype=torch.float32)


def candidate_identity(model: ProductionModel) -> dict:
    """The identity fields every row and artifact carries."""
    config = model.config
    return {
        "candidate_id": config.candidate_id,
        "config_digest": config.digest(),
        "architecture_family": ARCHITECTURE_FAMILY,
        "architecture_family_version": ARCHITECTURE_FAMILY_VERSION,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "parameters": model.parameter_count(),
        "initialisation_seed": model.initialisation_seed,
        "width": config.width,
        "blocks": config.blocks,
        "heads": config.heads,
        "feed_forward_width": config.feed_forward_width,
    }


def candidate_configuration(
    candidate_id: str,
    *,
    workers: int = STARTING_WORKERS,
    environments: int = STARTING_ENVIRONMENTS,
    inference_batch_size: int = 1024,
    precision: str = STARTING_PRECISION,
    legality: str = STARTING_LEGALITY,
    record_trajectories: bool = False,
    snapshot_interval: int = STARTING_SNAPSHOT_INTERVAL,
    detailed_timing: bool = True,
    root_seed: int = 60_004,
    verify_target_decisions: int = 0,
    max_concurrent_verifications: int = 2,
    retain_games: int = 0,
    rules: RulesConfig = TRAINING_RULES,
) -> CoordinatorConfig:
    """A `CoordinatorConfig` in the normalized frame, for one candidate."""
    return CoordinatorConfig(
        num_environments=environments,
        num_workers=workers,
        inference_batch_size=inference_batch_size,
        precision=precision,
        legality=legality,
        action_frame=ACTION_FRAME_NORMALIZED,
        root_seed=root_seed,
        record_trajectories=record_trajectories,
        snapshot_interval=snapshot_interval,
        collection_policy_version=collection_policy_version(candidate_id, precision),
        collection_checkpoint_id=None,
        verify_target_decisions=verify_target_decisions,
        max_concurrent_verifications=max_concurrent_verifications,
        retain_games=retain_games,
        detailed_timing=detailed_timing,
    )


def open_candidate_coordinator(
    candidate_id: str,
    config: CoordinatorConfig,
    *,
    device=None,
    seed: int = FAMILY_INITIALIZATION_SEED,
) -> SelfPlayCoordinator:
    """A coordinator owning one real candidate. Not started."""
    if config.action_frame != ACTION_FRAME_NORMALIZED:
        raise CoordinatorError(
            f"a {MODEL_CONTRACT_VERSION} candidate emits normalized actions; "
            f"the coordinator was configured for {config.action_frame!r}"
        )
    return SelfPlayCoordinator(
        config,
        device=device,
        model=build_pipeline_candidate(candidate_id, seed=seed),
        model_label=candidate_id,
    )


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

FAILURE_WORKER = "worker_errors"
FAILURE_MODEL = "model_errors"
FAILURE_NONFINITE = "nonfinite_outputs"
FAILURE_ILLEGAL = "illegal_actions"
FAILURE_FRAME = "action_frame_errors"
FAILURE_OTHER = "other_errors"

FAILURE_CATEGORIES = (
    FAILURE_WORKER,
    FAILURE_MODEL,
    FAILURE_NONFINITE,
    FAILURE_ILLEGAL,
    FAILURE_FRAME,
    FAILURE_OTHER,
)


def classify_failure(error: BaseException) -> str:
    """Which required counter a raised failure belongs to.

    The pipeline is deliberately loud: an illegal selection, a frame failure and
    a non-finite head all raise rather than being repaired. That makes every
    failure an exception, so the benchmark's per-row counters are produced by
    classifying exceptions instead of by tolerating faults and counting them.
    """
    message = str(error)
    if isinstance(error, ActionFrameMismatchError):
        return FAILURE_FRAME
    if isinstance(error, WorkerPoolError):
        return FAILURE_WORKER
    if isinstance(error, ModelContractError):
        return FAILURE_NONFINITE if "non-finite" in message else FAILURE_MODEL
    if isinstance(error, CoordinatorError):
        if "legality mask forbids" in message:
            return FAILURE_ILLEGAL
        return FAILURE_OTHER
    return FAILURE_OTHER


def empty_failure_counts() -> dict:
    return {category: 0 for category in FAILURE_CATEGORIES}


# ---------------------------------------------------------------------------
# The v2 correctness gate
# ---------------------------------------------------------------------------


@dataclass
class FrameGateReport:
    """Result of driving the whole `model_contract_v2` chain against the engine."""

    global_steps: int = 0
    environment_steps: int = 0
    rows_checked: int = 0
    normalized_mask_comparisons: int = 0
    normalized_legal_list_comparisons: int = 0
    round_trip_checks: int = 0
    red_decisions: int = 0
    blue_decisions: int = 0
    games_completed: int = 0
    resets_observed: int = 0
    illegal_selections: int = 0
    action_frame_mismatches: int = 0
    model_errors: int = 0
    nonfinite_outputs: int = 0
    state_replay_mismatches: int = 0
    mismatch_details: list = field(default_factory=list)
    terminal_reason_counts: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def note(self, category: str, detail: str) -> None:
        if category == "selected_action_legality":
            self.illegal_selections += 1
        elif category in ("normalized_legality", "action_round_trip", "acting_player"):
            self.action_frame_mismatches += 1
        elif category == "model_output":
            self.model_errors += 1
        else:
            self.state_replay_mismatches += 1
        if len(self.mismatch_details) < 40:
            self.mismatch_details.append({"category": category, "detail": detail})

    @property
    def total_mismatches(self) -> int:
        return (
            self.illegal_selections
            + self.action_frame_mismatches
            + self.model_errors
            + self.nonfinite_outputs
            + self.state_replay_mismatches
        )

    @property
    def both_colors_exercised(self) -> bool:
        return self.red_decisions > 0 and self.blue_decisions > 0

    def as_dict(self) -> dict:
        return {
            "global_steps": self.global_steps,
            "environment_steps": self.environment_steps,
            "rows_checked": self.rows_checked,
            "normalized_mask_comparisons": self.normalized_mask_comparisons,
            "normalized_legal_list_comparisons": self.normalized_legal_list_comparisons,
            "round_trip_checks": self.round_trip_checks,
            "red_decisions": self.red_decisions,
            "blue_decisions": self.blue_decisions,
            "both_colors_exercised": self.both_colors_exercised,
            "games_completed": self.games_completed,
            "resets_observed": self.resets_observed,
            "illegal_selections": self.illegal_selections,
            "action_frame_mismatches": self.action_frame_mismatches,
            "model_errors": self.model_errors,
            "nonfinite_outputs": self.nonfinite_outputs,
            "state_replay_mismatches": self.state_replay_mismatches,
            "total_mismatches": self.total_mismatches,
            "mismatch_details": list(self.mismatch_details),
            "terminal_reason_counts": dict(self.terminal_reason_counts),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def _derive_normalized_products(
    *, buffers, slot: int, reference, report: FrameGateReport
) -> tuple[int, tuple[int, ...]]:
    """Re-derive one slot's normalized legality straight from the reference game.

    Deliberately built from the *reference* game rather than from the published
    row: the point is to prove that what the model was shown is what the frozen
    engine's legal set means in the model's frame, and re-deriving it from the
    pipeline's own mask would only prove the mask equals itself.

    Called before the step, while the published row and the reference are still
    the same position.
    """
    player = int(reference.acting_player)
    absolute_legal = tuple(legal_actions(reference))
    normalized_legal = absolute_legal_actions_to_model(absolute_legal, player)
    report.normalized_legal_list_comparisons += 1

    # 1. The normalized set is the absolute set, no larger and no smaller, and it
    #    inverts exactly -- a non-bijective frame would show up as either.
    if len(normalized_legal) != len(absolute_legal):
        report.note(
            "normalized_legality",
            f"slot {slot}: {len(absolute_legal)} absolute legal actions became "
            f"{len(normalized_legal)} normalized ones",
        )
    if model_legal_actions_to_absolute(normalized_legal, player) != tuple(
        sorted(absolute_legal)
    ):
        report.note(
            "normalized_legality",
            f"slot {slot}: the normalized legal set does not invert to the engine's",
        )

    # 2. The dense normalized mask agrees with the normalized list, and the mask
    #    the pipeline published is the engine's.
    absolute_mask = legal_action_mask(reference, list(absolute_legal))
    normalized_mask = absolute_legal_mask_to_model(absolute_mask, player)
    report.normalized_mask_comparisons += 1
    from_mask = tuple(int(action) for action in np.flatnonzero(normalized_mask))
    if from_mask != normalized_legal:
        report.note(
            "normalized_legality",
            f"slot {slot}: the normalized dense mask and the normalized legal list disagree",
        )
    if not np.array_equal(buffers.legal_mask[slot].astype(bool), absolute_mask.astype(bool)):
        report.note(
            "legal_mask",
            f"slot {slot}: the published absolute mask differs from the reference",
        )
    return player, normalized_legal


def _check_selection_round_trip(
    *,
    coordinator: SelfPlayCoordinator,
    slot: int,
    player: int,
    normalized_legal: tuple[int, ...],
    report: FrameGateReport,
) -> None:
    """The selected normalized action must invert to the action the engine got.

    Called after the step, while the reference has not yet been advanced, so the
    normalized legal set derived before the step is still the set the selection
    was drawn from.
    """
    model_action = int(coordinator.last_model_actions[slot])
    absolute_action = int(coordinator.last_actions[slot])
    report.round_trip_checks += 1
    if model_action < 0:
        report.note(
            "action_round_trip",
            f"slot {slot}: the coordinator recorded no normalized selection",
        )
        return
    if model_action not in normalized_legal:
        report.note(
            "normalized_legality",
            f"slot {slot}: normalized selection {model_action} is not in the "
            "normalized legal set",
        )
    if model_action_to_absolute(model_action, player) != absolute_action:
        report.note(
            "action_round_trip",
            f"slot {slot}: normalized {model_action} inverts to "
            f"{model_action_to_absolute(model_action, player)}, not the applied "
            f"{absolute_action}",
        )
    if absolute_action_to_model(absolute_action, player) != model_action:
        report.note(
            "action_round_trip",
            f"slot {slot}: absolute {absolute_action} converts to "
            f"{absolute_action_to_model(absolute_action, player)}, not the selected "
            f"{model_action}",
        )
    if player == PLAYERS[0]:
        report.red_decisions += 1
    else:
        report.blue_decisions += 1


def run_frame_correctness_gate(
    candidate_id: str,
    *,
    num_environments: int = 64,
    num_workers: int = 4,
    inference_batch_size: int = 64,
    target_environment_steps: int = 6_000,
    precision: str = STARTING_PRECISION,
    legality: str = STARTING_LEGALITY,
    root_seed: int = 60_005,
    rows_per_step: int = 8,
    device=None,
    max_global_steps: int | None = None,
    seed: int = FAMILY_INITIALIZATION_SEED,
    progress=None,
) -> FrameGateReport:
    """Prove the whole v2 path before anything is timed.

    An independent set of engine games is advanced in lockstep with the live
    pipeline, exactly as Phase 3's integrated gate does, and every published row,
    every sampled action and every terminal outcome is compared against it. On
    top of that, each step re-derives the normalized legality and the normalized
    selection for a sample of slots directly from the reference game, so the
    chain

        engine legal set -> normalized legality -> candidate -> normalized
        selection -> absolute action -> engine

    is checked at both ends rather than only at the end the engine can see.
    """
    report = FrameGateReport()
    config = candidate_configuration(
        candidate_id,
        workers=num_workers,
        environments=num_environments,
        inference_batch_size=inference_batch_size,
        precision=precision,
        legality=legality,
        record_trajectories=False,
        detailed_timing=False,
        root_seed=root_seed,
    )
    mirror = ReferenceMirror(num_environments, root_seed=root_seed)
    started = time.perf_counter()
    coordinator = open_candidate_coordinator(candidate_id, config, device=device, seed=seed)
    coordinator.start()
    try:
        buffers = coordinator.pool.buffers
        limit = max_global_steps or (target_environment_steps // num_environments + 8)
        cursor = 0
        while report.environment_steps < target_environment_steps:
            if report.global_steps >= limit:
                break

            # 1. Everything the workers published matches the reference exactly.
            #    Reuses Phase 3's comparison so the accepted checks are not
            #    weakened by being restated.
            phase_report = _MirrorAdapter(mirror, report)
            phase_report.check_published(buffers)

            # 2. The normalized legality a sample of rows implies, derived from
            #    the reference games while they are still at the published ply.
            active = np.flatnonzero(buffers.status == STATUS_ACTIVE)
            sampled: list[tuple[int, int, tuple[int, ...]]] = []
            if active.size and rows_per_step:
                slots = [
                    int(active[(cursor + offset) % active.size])
                    for offset in range(min(rows_per_step, int(active.size)))
                ]
                cursor = (cursor + len(slots)) % max(int(active.size), 1)
                for slot in slots:
                    reference = mirror.states[slot]
                    if reference.terminal:
                        continue
                    report.rows_checked += 1
                    player, normalized_legal = _derive_normalized_products(
                        buffers=buffers, slot=slot, reference=reference, report=report
                    )
                    sampled.append((slot, player, normalized_legal))

            # 3. The pipeline takes its step.
            coordinator.step()
            actions = coordinator.last_actions.copy()
            report.global_steps += 1

            # 4. The selection the model made inverts to the action the engine
            #    was handed, for each sampled row.
            for slot, player, normalized_legal in sampled:
                _check_selection_round_trip(
                    coordinator=coordinator,
                    slot=slot,
                    player=player,
                    normalized_legal=normalized_legal,
                    report=report,
                )

            # 5. Every sampled action must be legal in the reference, which then
            #    advances by it.
            newly_terminal = phase_report.check_and_apply(actions)

            for slot in newly_terminal:
                phase_report.check_terminal(buffers, slot)
                mirror.reset(slot, phase_report)
                report.resets_observed += 1

            if progress is not None:
                progress(report)
    finally:
        coordinator.shutdown()
    report.elapsed_seconds = time.perf_counter() - started
    return report


class _MirrorAdapter:
    """Lets Phase 3's `ReferenceMirror` report into a `FrameGateReport`.

    `ReferenceMirror` is the accepted Phase 3 comparison and is reused verbatim;
    it reports through a `GateReport`-shaped object. Rather than copy its checks
    into this module -- where they could drift -- this adapter presents the
    interface it expects and routes each finding into the v2 report's categories.
    """

    def __init__(self, mirror: ReferenceMirror, report: FrameGateReport) -> None:
        self._mirror = mirror
        self._report = report
        self.mismatches = 0
        self.mismatch_categories: dict[str, int] = {}
        self.mismatch_details: list = []
        self.row_comparisons = 0
        self.action_legality_checks = 0
        self.environment_steps = 0
        self.games_completed = 0
        self.resets_observed = 0
        self.terminal_reason_counts = report.terminal_reason_counts

    def note(self, category: str, detail: str) -> None:
        self.mismatches += 1
        self._report.note(category, detail)

    def check_published(self, buffers) -> None:
        self._mirror.check_published(buffers, self)
        self._report.rows_checked += 0  # published rows are counted by the mirror

    def check_and_apply(self, actions: np.ndarray) -> list[int]:
        newly_terminal = self._mirror.check_and_apply(actions, self)
        self._report.environment_steps += self.environment_steps
        self.environment_steps = 0
        return newly_terminal

    def check_terminal(self, buffers, slot: int) -> None:
        self._mirror.check_terminal(buffers, slot, self)
        self._report.games_completed += self.games_completed
        self.games_completed = 0


# ---------------------------------------------------------------------------
# Non-finite output probe
# ---------------------------------------------------------------------------


def probe_head_finiteness(
    candidate_id: str,
    captured: dict,
    *,
    batch: int,
    precision: str = STARTING_PRECISION,
    device=None,
    seed: int = FAMILY_INITIALIZATION_SEED,
) -> dict:
    """Check all three heads on real published observations, at pipeline precision.

    A timed run cannot afford to scan 10,000 logits per row, and it does not have
    to: `ModelOutputs.validated` already refuses a non-finite value or belief head
    inside every forward pass, so a timed run that completed had finite heads
    there. The policy head is deliberately *not* finiteness-checked by the
    contract -- a model may score an illegal index arbitrarily -- so this probe
    covers it explicitly, once, outside any timed region.
    """
    from ..model.contract import validate_policy_logits

    rows = min(batch, captured["rows"])
    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[
        precision
    ]
    resolved = device if device is not None else torch.device("mps")
    model = build_pipeline_candidate(candidate_id, seed=seed).to(device=resolved, dtype=dtype)
    model.eval()
    observation_rows = captured["observation_rows"][:rows]
    tokens = torch.from_numpy(np.ascontiguousarray(observation_rows)).to(resolved)
    tokens = tokens.transpose(1, 2).contiguous().to(dtype)
    with torch.no_grad():
        outputs = model(tokens)
        validate_policy_logits(outputs.policy_logits, batch=rows, require_finite=True)
        nonfinite = {
            "policy": int((~torch.isfinite(outputs.policy_logits)).sum().item()),
            "value": int((~torch.isfinite(outputs.value_logits)).sum().item()),
            "belief": int((~torch.isfinite(outputs.belief_logits)).sum().item()),
        }
    del model, tokens, outputs
    return {
        "candidate_id": candidate_id,
        "precision": precision,
        "rows_checked": rows,
        "logits_checked": rows * (ACTION_SPACE_SIZE + 3 + 100 * 12),
        "nonfinite_by_head": nonfinite,
        "nonfinite_outputs": sum(nonfinite.values()),
        "all_finite": sum(nonfinite.values()) == 0,
    }


# ---------------------------------------------------------------------------
# Throughput measurement
# ---------------------------------------------------------------------------


def _latency_summary(latencies: list[float]) -> dict:
    if not latencies:
        return {"mean_step_ms": 0.0, "p50_step_ms": 0.0, "p95_step_ms": 0.0, "max_step_ms": 0.0}
    ordered = sorted(latencies)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean_step_ms": 1000 * statistics.fmean(latencies),
        "p50_step_ms": 1000 * statistics.median(ordered),
        "p95_step_ms": 1000 * ordered[index],
        "max_step_ms": 1000 * ordered[-1],
    }


def measure_candidate_configuration(
    candidate_id: str,
    config: CoordinatorConfig,
    *,
    seconds: float,
    mode: str,
    warmup_steps: int = 6,
    min_steps: int = 10,
    device=None,
    seed: int = FAMILY_INITIALIZATION_SEED,
) -> dict:
    """Run one candidate at one topology and report throughput and where time went.

    Warm-up steps are discarded by resetting the totals afterwards, so the first
    Metal compilation, the pool's first publish and the allocator's growth do not
    land inside the measured window.
    """
    model = build_pipeline_candidate(candidate_id, seed=seed)
    identity = candidate_identity(model)
    coordinator = SelfPlayCoordinator(config, device=device, model=model, model_label=candidate_id)
    row: dict = {
        **identity,
        "mode": mode,
        "precision": config.precision,
        "legality": config.legality,
        "action_frame": config.action_frame,
        "workers": config.num_workers,
        "environment_count": config.num_environments,
        "inference_batch_size": config.inference_batch_size,
        "snapshot_interval": config.snapshot_interval if config.record_trajectories else 0,
        "detailed_timing": config.detailed_timing,
        "collection_policy_version": config.collection_policy_version,
        "trajectory_version": TRAJECTORY_VERSION,
        **empty_failure_counts(),
    }
    swap_start = swap_bytes()
    pool_totals: dict = {}
    try:
        coordinator.start()
        for _ in range(warmup_steps):
            coordinator.step()
        coordinator.totals = RunTotals()
        warm_games = coordinator.games_finished
        warm_moves = coordinator.total_game_moves
        warm_reasons = dict(coordinator.terminal_reason_counts)

        started = time.perf_counter()
        steps = 0
        while time.perf_counter() - started < seconds or steps < min_steps:
            coordinator.step()
            steps += 1
        elapsed = time.perf_counter() - started

        totals = coordinator.totals
        games = coordinator.games_finished - warm_games
        moves = coordinator.total_game_moves - warm_moves
        reasons = {
            reason: count - warm_reasons.get(reason, 0)
            for reason, count in coordinator.terminal_reason_counts.items()
            if count - warm_reasons.get(reason, 0) > 0
        }
        worker_wall = max(elapsed * config.num_workers, 1e-9)
        row.update(
            {
                "status": "ok",
                "duration_seconds": elapsed,
                "global_steps": totals.steps,
                "positions": totals.positions,
                "transitions": totals.transitions,
                "games": games,
                "resets": totals.resets,
                "positions_per_second": totals.positions / elapsed,
                "transitions_per_second": totals.transitions / elapsed,
                "games_per_second": games / elapsed,
                "mean_game_length": (moves / games) if games else 0.0,
                "terminal_reason_counts": reasons,
                "chunks_per_step": totals.chunks / max(totals.steps, 1),
                # Fractions of wall time. `mps_inference_fraction` is the forward
                # pass alone; the frame conversion and the categorical draw are
                # reported together because both are device work performed on the
                # normalized legality product.
                "mps_inference_fraction": totals.inference_seconds / elapsed,
                "host_to_device_fraction": totals.transfer_seconds / elapsed,
                "normalized_legality_sampling_fraction": (
                    totals.frame_seconds + totals.sampling_seconds
                )
                / elapsed,
                "frame_conversion_fraction": totals.frame_seconds / elapsed,
                "compact_legality_fraction": totals.legality_seconds / elapsed,
                "writeback_fraction": totals.writeback_seconds / elapsed,
                "coordinator_active_fraction": totals.coordinator_seconds / elapsed,
                "coordinator_wait_fraction": totals.worker_seconds / elapsed,
                "worker_active_fraction": totals.worker_busy_seconds / worker_wall,
                "worker_wait_fraction": 1.0 - (totals.worker_busy_seconds / worker_wall),
                "barrier_fraction": totals.barrier_seconds / elapsed,
                "straggler_fraction": totals.straggler_seconds / elapsed,
                "process_rss_bytes": peak_memory_bytes(),
                "shared_memory_bytes": coordinator.pool.buffers.nbytes,
                "metal_memory_bytes": mps_memory_bytes().get("driver_allocated_bytes", 0),
                "metal_allocated_bytes": mps_memory_bytes().get("current_allocated_bytes", 0),
                **_latency_summary(totals.step_latencies),
            }
        )
    except BaseException as error:  # noqa: BLE001 - a failed point is a result
        category = classify_failure(error)
        row.update(
            {
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "error_category": category,
            }
        )
        row[category] = row.get(category, 0) + 1
    finally:
        try:
            pool_totals = coordinator.shutdown()
        except Exception as error:  # noqa: BLE001 - shutdown failure is a result
            pool_totals = {}
            row.setdefault("shutdown_error", f"{type(error).__name__}: {error}")

    swap_end = swap_bytes()
    row["swap_bytes"] = int(swap_end.get("swap_used_bytes", 0))
    row["swap_total_bytes"] = int(swap_end.get("swap_total_bytes", 0))
    row["swap_used_bytes_start"] = int(swap_start.get("swap_used_bytes", 0))
    row["worker_max_rss_bytes"] = int(pool_totals.get("worker_max_rss_bytes", 0))
    row["worker_cpu_seconds"] = float(pool_totals.get("worker_cpu_seconds", 0.0))

    if row.get("status") == "ok" and config.record_trajectories:
        elapsed = row["duration_seconds"]
        record_bytes = int(pool_totals.get("total_record_bytes", 0))
        decisions = int(pool_totals.get("total_decisions_recorded", 0))
        recorded_games = int(pool_totals.get("total_games_recorded", 0))
        recording_seconds = float(pool_totals.get("recording_seconds", 0.0))
        row.update(
            {
                "trajectory_bytes": record_bytes,
                "trajectory_records": recorded_games,
                "trajectory_decisions": decisions,
                "snapshot_bytes": int(pool_totals.get("total_snapshot_bytes", 0)),
                "snapshot_count": int(pool_totals.get("total_snapshot_count", 0)),
                "bytes_per_decision": (record_bytes / decisions) if decisions else 0.0,
                "bytes_per_game": (record_bytes / recorded_games) if recorded_games else 0.0,
                "gib_per_hour": (record_bytes / BYTES_PER_GIB) * (3600.0 / elapsed),
                "games_joined_late": int(pool_totals.get("total_games_joined_late", 0)),
                "verified_games": int(pool_totals.get("total_verified_games", 0)),
                "verified_decisions": int(pool_totals.get("total_verified_decisions", 0)),
                "reconstruction_mismatches": int(
                    pool_totals.get("total_reconstruction_mismatches", 0)
                ),
                # A whole-worker fraction: recording happens inside the worker
                # phase, so the honest denominator is worker wall time.
                "trajectory_write_fraction": recording_seconds
                / max(elapsed * config.num_workers, 1e-9),
                "recording_seconds": recording_seconds,
            }
        )
        row["retained_records"] = tuple(pool_totals.get("retained_records", ()))
        row["mismatch_details"] = list(pool_totals.get("mismatch_details", ()))
    elif row.get("status") == "ok":
        row.update(
            {
                "trajectory_bytes": 0,
                "trajectory_records": 0,
                "trajectory_decisions": 0,
                "snapshot_bytes": 0,
                "snapshot_count": 0,
                "bytes_per_decision": 0.0,
                "bytes_per_game": 0.0,
                "gib_per_hour": 0.0,
                "trajectory_write_fraction": 0.0,
                "recording_seconds": 0.0,
            }
        )
    return row


# ---------------------------------------------------------------------------
# The two sides of the bottleneck ratio
# ---------------------------------------------------------------------------


def capture_published_batch(
    config: CoordinatorConfig,
    *,
    steps: int = 6,
    device=None,
    candidate_id: str | None = None,
    seed: int = FAMILY_INITIALIZATION_SEED,
) -> dict:
    """Real published rows, copied out of shared memory and kept.

    The inference-capacity measurement needs a batch representative of what the
    pipeline actually produces -- real observations at real plies, with real
    legal-set sizes and a real mix of acting colours. Synthetic rows would change
    the legality density, and therefore the sampling cost.

    Why two consecutive steps
    -------------------------
    The pool is bulk-synchronous and every game starts at ply 0 with red to move,
    so at any single step almost every slot has the *same* player to move. A
    single snapshot is therefore about 99 percent one colour, and since the frame
    permutation is the identity for red and real work for blue, timing against
    one snapshot would either halve or double the conversion cost depending on
    which step it was taken at.

    A real run pays the full permutation on blue steps and nothing on red steps,
    averaging half -- which is exactly what a batch of half blue rows costs. So
    the capture takes two consecutive steps and keeps half its rows from each.
    """
    coordinator = open_candidate_coordinator(
        candidate_id or "C0", config, device=device, seed=seed
    )
    coordinator.start()
    halves: list[dict] = []
    try:
        for _ in range(steps):
            coordinator.step()
        for phase in range(2):
            buffers = coordinator.pool.buffers
            active = np.flatnonzero(buffers.status == STATUS_ACTIVE)
            observations = buffers.observations
            flat = observations.reshape(
                config.num_environments, observations.shape[1], -1
            )
            # The first half of the slots from the first step, the second half
            # from the next, so no slot contributes two different plies of the
            # same game and the union is still one row per slot.
            half = active.size // 2
            chosen = active[:half] if phase == 0 else active[half:]
            halves.append(
                {
                    "observation_rows": np.ascontiguousarray(flat[chosen]),
                    "mask_rows": np.ascontiguousarray(buffers.legal_mask[chosen]),
                    "acting_rows": np.ascontiguousarray(buffers.acting_player[chosen]),
                }
            )
            if phase == 0:
                coordinator.step()
    finally:
        coordinator.shutdown()

    # Interleaved rather than concatenated, so a measurement that only takes the
    # first `batch` rows still gets a balanced mix of colours.
    order = np.argsort(
        np.concatenate(
            [
                np.arange(halves[0]["acting_rows"].size) * 2,
                np.arange(halves[1]["acting_rows"].size) * 2 + 1,
            ]
        ),
        kind="stable",
    )
    captured = {
        key: np.ascontiguousarray(
            np.concatenate([halves[0][key], halves[1][key]], axis=0)[order]
        )
        for key in ("observation_rows", "mask_rows", "acting_rows")
    }
    captured["rows"] = int(captured["acting_rows"].size)
    counts = captured["mask_rows"].sum(axis=1)
    captured["mean_legal_actions"] = float(counts.mean())
    captured["red_rows"] = int((captured["acting_rows"] == PLAYERS[0]).sum())
    captured["blue_rows"] = int((captured["acting_rows"] == PLAYERS[1]).sum())
    captured["capture_note"] = (
        "half the rows from one global step and half from the next, interleaved, "
        "so the acting-colour mix matches the average a bulk-synchronous run pays"
    )
    return captured


def measure_inference_capacity(
    candidate_id: str,
    *,
    batch: int,
    captured: dict,
    seconds: float = 6.0,
    warmup: int = 5,
    precision: str = STARTING_PRECISION,
    legality: str = STARTING_LEGALITY,
    record_probabilities: bool = False,
    device=None,
    seed: int = FAMILY_INITIALIZATION_SEED,
) -> dict:
    """The R denominator: this pipeline's inference stage, with no simulator.

    Measured by calling the coordinator's own `infer_batch`, which is the very
    `_run_chunk` a global step calls -- host-to-device transfer, the absolute ->
    normalized legality permutation, the candidate forward pass, masked sampling,
    the normalized -> absolute inverse and the readback. The worker pool is shut
    down first, so no simulation competes for a core while it runs.
    """
    rows = captured["rows"]
    if batch > rows:
        batch = rows
    config = candidate_configuration(
        candidate_id,
        workers=1,
        environments=max(batch, 2),
        inference_batch_size=batch,
        precision=precision,
        legality=legality,
        record_trajectories=False,
        detailed_timing=False,
    )
    model = build_pipeline_candidate(candidate_id, seed=seed)
    coordinator = SelfPlayCoordinator(config, device=device, model=model, model_label=candidate_id)

    observation_rows = captured["observation_rows"][:batch]
    mask_rows = captured["mask_rows"][:batch]
    acting_rows = captured["acting_rows"][:batch]
    compact_ids = compact_valid = None
    if record_probabilities or legality == "compact":
        compact_ids, compact_valid, _ = compact_legality_from_masks(
            mask_rows, capacity=POLICY_CAPACITY
        )

    for _ in range(warmup):
        coordinator.infer_batch(
            observation_rows,
            mask_rows,
            acting_rows,
            compact_ids=compact_ids,
            compact_valid=compact_valid,
            record_probabilities=record_probabilities,
        )

    iterations = 0
    started = time.perf_counter()
    while time.perf_counter() - started < seconds or iterations < 3:
        coordinator.infer_batch(
            observation_rows,
            mask_rows,
            acting_rows,
            compact_ids=compact_ids,
            compact_valid=compact_valid,
            record_probabilities=record_probabilities,
        )
        iterations += 1
    elapsed = time.perf_counter() - started

    return {
        "candidate_id": candidate_id,
        "precision": precision,
        "legality": legality,
        "batch": batch,
        "iterations": iterations,
        "measured_seconds": elapsed,
        "positions": iterations * batch,
        "positions_per_second": (iterations * batch) / elapsed,
        "records_probabilities": record_probabilities,
        "mean_legal_actions": captured["mean_legal_actions"],
        "boundary": (
            "coordinator._run_chunk: host->device transfer, absolute->normalized dense "
            "legality permutation, candidate forward, masked Gumbel sampling, "
            "normalized->absolute inverse, action/value readback"
        ),
    }


def measure_simulation_capacity(
    *,
    workers: int,
    environments: int,
    seconds: float,
    record_trajectories: bool = False,
    snapshot_interval: int = STARTING_SNAPSHOT_INTERVAL,
    root_seed: int = 60_011,
) -> dict:
    """The R numerator: the same worker pool with the model removed.

    Phase 3's measurement, reused unchanged: observation building, legality
    generation, the engine transition, the shared-memory transport, worker
    synchronisation and independent reset, driven by the deterministic benchmark
    policy at the same worker and environment count as the integrated run. It is
    candidate-independent by construction, which is what makes it a fair
    numerator for every candidate's ratio.
    """
    return measure_simulation_pipeline(
        num_environments=environments,
        num_workers=workers,
        seconds=seconds,
        root_seed=root_seed,
        record_trajectories=record_trajectories,
        snapshot_interval=snapshot_interval,
    )


def candidate_bottleneck_ratio(
    *,
    candidate_id: str,
    simulation: dict,
    inference: dict,
) -> dict:
    """`R = sustainable simulation capacity / sustainable candidate inference`."""
    ratio = compute_ratio(
        simulation["positions_per_second"], inference["positions_per_second"]
    )
    ratio.update(
        {
            "candidate_id": candidate_id,
            "candidate_inference_positions_per_second": inference["positions_per_second"],
            "numerator_measurement": (
                f"stratego.training.end_to_end_benchmark.measure_simulation_pipeline at "
                f"{simulation['num_workers']} workers x {simulation['num_environments']} "
                f"environments for {simulation['measured_seconds']:.1f}s, model removed and "
                "replaced by the deterministic benchmark policy"
            ),
            "denominator_measurement": inference["boundary"]
            + f", batch {inference['batch']}, {inference['precision']}, "
            f"{inference['measured_seconds']:.1f}s, worker pool shut down",
            "numerator_positions_per_second": simulation["positions_per_second"],
            "serial_composition_ceiling": 1.0
            / (
                1.0 / simulation["positions_per_second"]
                + 1.0 / inference["positions_per_second"]
            ),
        }
    )
    # `compute_ratio` names the representative model, which is not what ran.
    ratio["candidate_inference_positions_per_second"] = inference["positions_per_second"]
    ratio.pop("representative_model_inference_positions_per_second", None)
    return ratio


# ---------------------------------------------------------------------------
# Reconstruction of stored decisions, including the normalized products
# ---------------------------------------------------------------------------


@dataclass
class ReconstructionReport:
    """Result of rebuilding stored games from their accepted compact bytes."""

    games_sampled: int = 0
    decisions_sampled: int = 0
    codec_round_trips: int = 0
    state_mismatches: int = 0
    observation_mismatches: int = 0
    absolute_legal_mismatches: int = 0
    normalized_legal_mismatches: int = 0
    normalized_selection_mismatches: int = 0
    absolute_selection_mismatches: int = 0
    policy_field_mismatches: int = 0
    value_field_mismatches: int = 0
    schema_problems: int = 0
    belief_targets_found_in_record: int = 0
    policy_reevaluations: int = 0
    policy_reevaluation_max_deviation: float = 0.0
    policy_reevaluation_over_tolerance: int = 0
    details: list = field(default_factory=list)

    def note(self, field_name: str, detail: str) -> None:
        setattr(self, field_name, getattr(self, field_name) + 1)
        if len(self.details) < 40:
            self.details.append({"field": field_name, "detail": detail})

    @property
    def total_mismatches(self) -> int:
        return (
            self.state_mismatches
            + self.observation_mismatches
            + self.absolute_legal_mismatches
            + self.normalized_legal_mismatches
            + self.normalized_selection_mismatches
            + self.absolute_selection_mismatches
            + self.policy_field_mismatches
            + self.value_field_mismatches
            + self.schema_problems
            + self.belief_targets_found_in_record
        )

    def as_dict(self) -> dict:
        return {
            "games_sampled": self.games_sampled,
            "decisions_sampled": self.decisions_sampled,
            "codec_round_trips": self.codec_round_trips,
            "state_mismatches": self.state_mismatches,
            "observation_mismatches": self.observation_mismatches,
            "absolute_legal_mismatches": self.absolute_legal_mismatches,
            "normalized_legal_mismatches": self.normalized_legal_mismatches,
            "normalized_selection_mismatches": self.normalized_selection_mismatches,
            "absolute_selection_mismatches": self.absolute_selection_mismatches,
            "policy_field_mismatches": self.policy_field_mismatches,
            "value_field_mismatches": self.value_field_mismatches,
            "schema_problems": self.schema_problems,
            "belief_targets_found_in_record": self.belief_targets_found_in_record,
            "total_mismatches": self.total_mismatches,
            "policy_reevaluations": self.policy_reevaluations,
            "policy_reevaluation_max_deviation": round(
                self.policy_reevaluation_max_deviation, 6
            ),
            "policy_reevaluation_tolerance": POLICY_REEVALUATION_TOLERANCE,
            "policy_reevaluation_over_tolerance": self.policy_reevaluation_over_tolerance,
            "details": list(self.details),
        }


def _record_mentions_belief(payload: bytes) -> bool:
    """Whether an encoded record carries anything belief-shaped.

    The trajectory schema has no belief field and must not grow one: a privileged
    target inside ordinary collected data would make the corpus unsafe to hand to
    a model. Checked against the *bytes*, because that is what would ship.
    """
    return b"belief" in payload


def reconstruct_stored_games(
    payloads,
    *,
    max_games: int | None = None,
    max_decisions_per_game: int | None = None,
    model: ProductionModel | None = None,
    device=None,
    dtype: torch.dtype = torch.float16,
) -> ReconstructionReport:
    """Rebuild stored games from their bytes and check every required product.

    The instruction's list, in order:

    ``state``
        rebuilt from the nearest snapshot plus replayed actions, then compared to
        the ply and acting player the record independently claims.
    ``observation``
        rebuilt by the frozen engine from that state; its shape and finiteness
        are checked, and it is what the re-evaluation below is fed.
    ``absolute legal actions``
        the engine's set at the rebuilt state, compared to the stored set.
    ``normalized model legal actions``
        derived from the rebuilt absolute set through Agent 1's converter and
        required to invert back to it.
    ``selected normalized action``
        derived from the stored absolute action; required to be in the normalized
        legal set and to invert to the stored action.
    ``selected absolute engine action``
        required to equal the record's action-list entry for that ply.
    ``policy/value decision fields``
        one probability per legal action in ascending absolute order, summing to
        one, with the selected action present; the value triple normalized.

    `model` is optional. When given, each sampled decision is re-evaluated
    through the candidate and the recorded distribution is compared to a freshly
    computed one. That cannot be bit-exact -- the record came out of a batch of
    up to 2,048 rows in float16 -- so it is reported as a deviation against
    :data:`POLICY_REEVALUATION_TOLERANCE` rather than counted as a mismatch.
    """
    report = ReconstructionReport()
    for payload in payloads:
        if max_games is not None and report.games_sampled >= max_games:
            break
        record = decode_game_record(payload)
        report.games_sampled += 1

        # The bytes decode to a record that re-encodes to the same bytes: exact
        # reconstruction of the stored form, not merely a readable one.
        if encode_game_record(record) != payload:
            report.note("schema_problems", f"{record.game_id}: codec is not byte-exact")
        else:
            report.codec_round_trips += 1
        for problem in validate_game_record(record):
            report.note("schema_problems", f"{record.game_id}: {problem}")
        if _record_mentions_belief(payload):
            report.note(
                "belief_targets_found_in_record",
                f"{record.game_id}: the encoded record mentions a belief field",
            )

        plies = list(range(len(record.actions)))
        if max_decisions_per_game is not None:
            plies = plies[:max_decisions_per_game]
        if not plies:
            continue

        for rebuilt in iter_reconstructed_decisions(
            record, plies, dense_mask=True, copy_state=False
        ):
            decision = record.decision_at(rebuilt.ply)
            report.decisions_sampled += 1
            player = int(rebuilt.acting_player)

            # -- state ---------------------------------------------------------
            if rebuilt.state.total_moves != decision.ply:
                report.note(
                    "state_mismatches",
                    f"{record.game_id} ply {decision.ply}: rebuilt state is at ply "
                    f"{rebuilt.state.total_moves}",
                )
            if player != decision.acting_player:
                report.note(
                    "state_mismatches",
                    f"{record.game_id} ply {decision.ply}: acting player "
                    f"{player} != stored {decision.acting_player}",
                )

            # -- observation ---------------------------------------------------
            observation = rebuilt.observation
            expected = build_observation(rebuilt.state, player)
            if not np.array_equal(observation, expected) or not np.isfinite(observation).all():
                report.note(
                    "observation_mismatches",
                    f"{record.game_id} ply {decision.ply}: rebuilt observation is not "
                    "the engine's observation for the rebuilt state",
                )

            # -- absolute legal actions ---------------------------------------
            absolute_legal = tuple(rebuilt.legal_action_ids)
            if absolute_legal != tuple(decision.legal_action_ids):
                report.note(
                    "absolute_legal_mismatches",
                    f"{record.game_id} ply {decision.ply}: {len(absolute_legal)} rebuilt "
                    f"legal actions vs {len(decision.legal_action_ids)} stored",
                )
            if rebuilt.legal_mask is not None:
                from_mask = tuple(int(a) for a in np.flatnonzero(rebuilt.legal_mask))
                if from_mask != absolute_legal:
                    report.note(
                        "absolute_legal_mismatches",
                        f"{record.game_id} ply {decision.ply}: dense mask and legal list "
                        "disagree",
                    )

            # -- normalized model legal actions --------------------------------
            normalized_legal = absolute_legal_actions_to_model(absolute_legal, player)
            if len(normalized_legal) != len(absolute_legal):
                report.note(
                    "normalized_legal_mismatches",
                    f"{record.game_id} ply {decision.ply}: normalized set has "
                    f"{len(normalized_legal)} of {len(absolute_legal)} actions",
                )
            elif model_legal_actions_to_absolute(normalized_legal, player) != tuple(
                sorted(absolute_legal)
            ):
                report.note(
                    "normalized_legal_mismatches",
                    f"{record.game_id} ply {decision.ply}: normalized set does not invert",
                )

            # -- selected normalized / absolute action -------------------------
            stored_action = int(decision.selected_action_id)
            normalized_action = absolute_action_to_model(stored_action, player)
            if normalized_action not in normalized_legal:
                report.note(
                    "normalized_selection_mismatches",
                    f"{record.game_id} ply {decision.ply}: normalized selection "
                    f"{normalized_action} is not in the normalized legal set",
                )
            if model_action_to_absolute(normalized_action, player) != stored_action:
                report.note(
                    "normalized_selection_mismatches",
                    f"{record.game_id} ply {decision.ply}: normalized selection does not "
                    "invert to the stored action",
                )
            if stored_action not in absolute_legal:
                report.note(
                    "absolute_selection_mismatches",
                    f"{record.game_id} ply {decision.ply}: stored action {stored_action} "
                    "is not legal in the rebuilt state",
                )
            if stored_action != int(record.actions[decision.ply]):
                report.note(
                    "absolute_selection_mismatches",
                    f"{record.game_id} ply {decision.ply}: decision action disagrees with "
                    "the record's action list",
                )

            # -- policy / value decision fields --------------------------------
            probabilities = np.asarray(decision.old_probabilities, dtype=np.float64)
            if probabilities.size != len(absolute_legal):
                report.note(
                    "policy_field_mismatches",
                    f"{record.game_id} ply {decision.ply}: {probabilities.size} "
                    f"probabilities for {len(absolute_legal)} legal actions",
                )
            elif (
                not np.isfinite(probabilities).all()
                or probabilities.min() < 0.0
                or abs(probabilities.sum() - 1.0) > 1e-3
            ):
                report.note(
                    "policy_field_mismatches",
                    f"{record.game_id} ply {decision.ply}: stored distribution sums to "
                    f"{probabilities.sum():.6f}",
                )
            value = np.asarray(decision.win_draw_loss_prediction, dtype=np.float64)
            if value.size != 3 or not np.isfinite(value).all() or abs(value.sum() - 1.0) > 1e-3:
                report.note(
                    "value_field_mismatches",
                    f"{record.game_id} ply {decision.ply}: value triple {value.tolist()}",
                )

            # -- optional re-evaluation through the candidate -------------------
            if model is not None and probabilities.size == len(absolute_legal):
                deviation = _reevaluate_policy(
                    model=model,
                    observation=observation,
                    absolute_legal=absolute_legal,
                    normalized_legal=normalized_legal,
                    player=player,
                    stored=probabilities,
                    device=device,
                    dtype=dtype,
                )
                report.policy_reevaluations += 1
                report.policy_reevaluation_max_deviation = max(
                    report.policy_reevaluation_max_deviation, deviation
                )
                if deviation > POLICY_REEVALUATION_TOLERANCE:
                    report.policy_reevaluation_over_tolerance += 1
    return report


def _reevaluate_policy(
    *,
    model: ProductionModel,
    observation: np.ndarray,
    absolute_legal: tuple[int, ...],
    normalized_legal: tuple[int, ...],
    player: int,
    stored: np.ndarray,
    device,
    dtype: torch.dtype,
) -> float:
    """Largest absolute difference between the stored and a fresh distribution.

    Runs the candidate on the rebuilt observation, masks with the *normalized*
    legal set, and gathers one probability per legal action in ascending
    *absolute* order -- the order the record stores. A frame error would move
    mass to entirely different moves and show up here as a deviation near one,
    which is why this is worth computing even though it cannot be exact.
    """
    resolved = device if device is not None else next(model.parameters()).device
    tokens = torch.from_numpy(
        np.ascontiguousarray(observation.reshape(1, observation.shape[0], -1))
    ).to(resolved)
    tokens = tokens.transpose(1, 2).contiguous().to(dtype)
    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.policy_logits.to(torch.float32)
        mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool, device=resolved)
        mask[torch.tensor(normalized_legal, dtype=torch.long, device=resolved)] = True
        masked = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
        dense = torch.softmax(masked, dim=1)
        order = torch.tensor(
            [absolute_action_to_model(action, player) for action in absolute_legal],
            dtype=torch.long,
            device=resolved,
        )
        fresh = dense[0].index_select(0, order).to("cpu").numpy().astype(np.float64)
    return float(np.abs(fresh - stored).max())


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

#: How far a recording run must be driven before its byte rate is read.
#:
#: A pool starts with every environment at ply 0, and a trajectory is written
#: only when a game is *sealed*, so early on the decisions of a thousand
#: unfinished games sit in the denominator while none of their bytes are in the
#: numerator. Measured convergence on this machine: C0's windowed rate climbs
#: from 0.63 GiB/hour over the first 20 seconds to a flat 7.0-7.2 GiB/hour from
#: about 120 seconds -- roughly 1,000 global steps -- onward. The transient
#: understates the steady state by an order of magnitude, so a short recording
#: row's byte rate must never be projected to a week.
#:
#: The threshold is counted in **global steps**, not seconds, because what has to
#: happen is that the slots' plies spread out, and that takes about two mean game
#: lengths of *simulated* time. A candidate half as fast needs twice the wall
#: clock to reach the same place, and a seconds-based warmup would silently give
#: the slowest candidate the least settled measurement -- exactly backwards.
STORAGE_WARMUP_STEPS = 1_100
STORAGE_MEASURE_STEPS = 900
STORAGE_SAMPLE_STEPS = 150

#: Largest relative spread across the measured windows that still counts as
#: settled. Above this the run is reported as unsettled rather than averaged.
STORAGE_SETTLED_SPREAD = 0.35


def measure_storage_rate(
    candidate_id: str,
    config: CoordinatorConfig,
    *,
    warmup_steps: int = STORAGE_WARMUP_STEPS,
    measure_steps: int = STORAGE_MEASURE_STEPS,
    sample_steps: int = STORAGE_SAMPLE_STEPS,
    device=None,
    seed: int = FAMILY_INITIALIZATION_SEED,
) -> dict:
    """Sustained trajectory byte rate, sampled over time rather than divided.

    Runs the real production-recording path and reads the cumulative recording
    counters every `sample_steps` global steps. The reported rate covers only the
    steps after `warmup_steps`, by which point the environments have
    desynchronized and games are sealing at their steady rate. Dividing total
    bytes by total elapsed time -- which is what a short benchmark row does --
    would instead average the cold-start transient into the answer.

    Every sample is returned, so the convergence is visible in the artifact and a
    reader can check that the measured window really is flat rather than taking
    the claim on trust.
    """
    if not config.record_trajectories:
        raise ValueError("a storage rate needs a recording configuration")
    coordinator = SelfPlayCoordinator(
        config,
        device=device,
        model=build_pipeline_candidate(candidate_id, seed=seed),
        model_label=candidate_id,
    )
    samples: list[dict] = []
    coordinator.start()
    started = time.perf_counter()
    previous = {
        "elapsed": 0.0,
        "step": 0,
        "bytes": 0,
        "decisions": 0,
        "games": 0,
        "positions": 0,
    }
    try:
        for step in range(1, warmup_steps + measure_steps + 1):
            coordinator.step()
            if step % sample_steps and step != warmup_steps + measure_steps:
                continue
            totals = coordinator.pool.recording_totals()
            current = {
                "elapsed": time.perf_counter() - started,
                "step": step,
                "bytes": int(totals["total_record_bytes"]),
                "decisions": int(totals["total_decisions_recorded"]),
                "games": int(totals["total_games_recorded"]),
                "snapshots": int(totals["total_snapshot_count"]),
                "positions": int(coordinator.totals.positions),
            }
            window = current["elapsed"] - previous["elapsed"]
            window_bytes = current["bytes"] - previous["bytes"]
            window_decisions = current["decisions"] - previous["decisions"]
            samples.append(
                {
                    **current,
                    "window_seconds": window,
                    "window_steps": current["step"] - previous["step"],
                    "window_bytes_per_second": window_bytes / window,
                    "window_gib_per_hour": (window_bytes / window) * 3600 / BYTES_PER_GIB,
                    "window_games_per_second": (current["games"] - previous["games"]) / window,
                    "window_positions_per_second": (
                        current["positions"] - previous["positions"]
                    )
                    / window,
                    "window_bytes_per_decision": (
                        window_bytes / window_decisions if window_decisions else 0.0
                    ),
                    "in_measured_window": current["step"] > warmup_steps,
                }
            )
            previous = current
    finally:
        pool_totals = coordinator.shutdown()

    measured = [sample for sample in samples if sample["in_measured_window"]]
    if not measured:
        raise CoordinatorError(
            "the storage run produced no sample after the warmup; lengthen it"
        )
    last = measured[-1]
    # Differenced from the raw cumulative counters at the two ends of the steady
    # window, so the headline rate does not depend on how the per-window derived
    # values were rounded or on the sampling cadence.
    baseline_index = samples.index(measured[0]) - 1
    baseline = (
        samples[baseline_index]
        if baseline_index >= 0
        else {"elapsed": 0.0, "step": 0, "bytes": 0, "decisions": 0, "games": 0,
              "positions": 0}
    )
    steady_seconds = last["elapsed"] - baseline["elapsed"]
    steady_bytes = last["bytes"] - baseline["bytes"]
    steady_decisions = last["decisions"] - baseline["decisions"]
    steady_games = last["games"] - baseline["games"]
    steady_positions = last["positions"] - baseline["positions"]

    rates = [sample["window_gib_per_hour"] for sample in measured]
    return {
        "candidate_id": candidate_id,
        "configuration": config.as_dict(),
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "warmup_seconds": baseline["elapsed"],
        "measured_seconds": steady_seconds,
        "measured_steps": last["step"] - baseline["step"],
        "total_run_seconds": last["elapsed"],
        "total_run_steps": last["step"],
        "samples": samples,
        "steady_state_record_bytes": steady_bytes,
        "steady_state_decisions": steady_decisions,
        "steady_state_games": steady_games,
        "steady_state_positions": steady_positions,
        "steady_state_bytes_per_second": steady_bytes / steady_seconds,
        "steady_state_gib_per_hour": (steady_bytes / steady_seconds) * 3600 / BYTES_PER_GIB,
        "steady_state_positions_per_second": steady_positions / steady_seconds,
        "steady_state_games_per_second": steady_games / steady_seconds,
        "steady_state_bytes_per_decision": (
            steady_bytes / steady_decisions if steady_decisions else 0.0
        ),
        "steady_state_bytes_per_game": steady_bytes / steady_games if steady_games else 0.0,
        "steady_state_mean_game_length": (
            steady_positions / steady_games if steady_games else 0.0
        ),
        "window_rate_spread": (max(rates) - min(rates)) / max(statistics.fmean(rates), 1e-9),
        "window_rates_gib_per_hour": rates,
        "cumulative_gib_per_hour_if_naively_divided": (
            (last["bytes"] / last["elapsed"]) * 3600 / BYTES_PER_GIB
        ),
        "total_record_bytes": int(pool_totals.get("total_record_bytes", 0)),
        "total_games_recorded": int(pool_totals.get("total_games_recorded", 0)),
        "total_snapshot_count": int(pool_totals.get("total_snapshot_count", 0)),
        "reconstruction_mismatches": int(
            pool_totals.get("total_reconstruction_mismatches", 0)
        ),
    }


def storage_projection(*, record_bytes: int, seconds: float, label: str = "") -> dict:
    """Measured bytes over measured time, projected out to a production week."""
    if seconds <= 0:
        raise ValueError("a storage rate needs a positive measured duration")
    per_second = record_bytes / seconds
    per_hour = per_second * 3600.0
    gib_per_hour = per_hour / BYTES_PER_GIB
    bytes_168 = per_hour * 168.0
    return {
        "label": label,
        "measured_record_bytes": int(record_bytes),
        "measured_seconds": round(seconds, 3),
        "bytes_per_second": per_second,
        "gib_per_hour": gib_per_hour,
        "gib_per_24_hours": gib_per_hour * 24.0,
        "gib_per_168_hours": gib_per_hour * 168.0,
        "bytes_per_168_hours": bytes_168,
        "internal_free_bytes": INTERNAL_FREE_BYTES,
        "external_free_bytes": EXTERNAL_FREE_BYTES,
        "fraction_of_internal_free": bytes_168 / INTERNAL_FREE_BYTES,
        "fraction_of_external_free": bytes_168 / EXTERNAL_FREE_BYTES,
        "fits_internal_uncompressed": bytes_168 <= INTERNAL_FREE_BYTES,
        "fits_external_uncompressed": bytes_168 <= EXTERNAL_FREE_BYTES,
    }


# ---------------------------------------------------------------------------
# Finalist selection
# ---------------------------------------------------------------------------

#: Every input the finalist rule may read. Playing strength, win rate, match
#: results and Elo are absent by construction, and a test asserts that no field
#: name here matches a strength-shaped substring.
FINALIST_INPUT_KEYS = (
    "candidate_id",
    "parameters",
    "standalone_inference_positions_per_second",
    "standalone_training_examples_per_second",
    "collection_positions_per_second",
    "recording_positions_per_second",
    "gib_per_hour",
    "process_rss_bytes",
    "metal_memory_bytes",
    "numerically_stable_float16",
    "bottleneck_ratio",
)

FORBIDDEN_INPUT_SUBSTRINGS = (
    "win",
    "loss_rate",
    "draw",
    "elo",
    "strength",
    "score",
    "result",
    "match",
    "gauntlet",
)


def finalist_inputs(summary) -> dict:
    """The only fields the finalist rule sees."""
    return {key: summary.get(key) for key in FINALIST_INPUT_KEYS}


def recommend_finalists(summaries, *, minimum: int = 2, maximum: int = 3) -> dict:
    """Two or three finalists, under a rule stated before the numbers existed.

    The rule is a capacity/throughput frontier, not a ranking:

    1. a candidate is **excluded** when it is numerically unstable in the real
       pipeline, or when its production-recording throughput is zero (it could
       not sustain the recording path at all);
    2. among the rest, a candidate is **dominated** when another has at least as
       many parameters, at least as much recording throughput, and no worse
       memory, with at least one strict improvement;
    3. the surviving frontier is ordered by parameter count and the largest, the
       smallest and -- when three fit -- the best-throughput middle are taken, so
       the handoff spans the capacity range rather than clustering.

    Capacity proxy is parameter count, exactly as Agents 2 and 3 used it. No
    playing-strength quantity is reachable from here.
    """
    rows = [dict(summary) for summary in summaries]
    verdicts: dict[str, str] = {}
    reasons: dict[str, str] = {}

    viable = []
    for row in rows:
        candidate = row["candidate_id"]
        if not row.get("numerically_stable_float16", True):
            verdicts[candidate] = "EXCLUDED"
            reasons[candidate] = "not numerically stable in the integrated pipeline"
            continue
        if not row.get("recording_positions_per_second"):
            verdicts[candidate] = "EXCLUDED"
            reasons[candidate] = "no sustained production-recording throughput"
            continue
        viable.append(row)

    frontier = []
    for row in viable:
        candidate = row["candidate_id"]
        dominated_by = None
        for other in viable:
            if other["candidate_id"] == candidate:
                continue
            at_least = (
                other["parameters"] >= row["parameters"]
                and other["recording_positions_per_second"]
                >= row["recording_positions_per_second"]
                and other.get("process_rss_bytes", 0) <= row.get("process_rss_bytes", 0)
            )
            strictly = (
                other["parameters"] > row["parameters"]
                or other["recording_positions_per_second"]
                > row["recording_positions_per_second"]
                or other.get("process_rss_bytes", 0) < row.get("process_rss_bytes", 0)
            )
            if at_least and strictly:
                dominated_by = other["candidate_id"]
                break
        if dominated_by is not None:
            verdicts[candidate] = "DOMINATED"
            reasons[candidate] = (
                f"{dominated_by} has at least as much capacity and recording throughput "
                "with no more memory"
            )
        else:
            frontier.append(row)

    frontier.sort(key=lambda row: row["parameters"])
    chosen: list[dict] = []
    if frontier:
        chosen.append(frontier[0])
        if len(frontier) > 1:
            chosen.append(frontier[-1])
        if len(frontier) > 2 and maximum >= 3:
            middle = max(
                frontier[1:-1], key=lambda row: row["recording_positions_per_second"]
            )
            chosen.insert(1, middle)
    chosen = chosen[:maximum]

    for row in frontier:
        candidate = row["candidate_id"]
        if any(pick["candidate_id"] == candidate for pick in chosen):
            verdicts[candidate] = "FINALIST"
            reasons[candidate] = (
                f"on the integrated frontier: {row['parameters']:,} parameters, "
                f"{row['collection_positions_per_second']:,.0f} positions/s collection-only, "
                f"{row['recording_positions_per_second']:,.0f} positions/s recording, "
                f"{row['gib_per_hour']:.2f} GiB/hour"
            )
        else:
            verdicts[candidate] = "FRONTIER_NOT_SELECTED"
            reasons[candidate] = (
                "on the frontier but not one of the spanning picks; kept as evidence"
            )

    return {
        "finalist_ids": [row["candidate_id"] for row in chosen],
        "finalist_reasons": {
            row["candidate_id"]: reasons[row["candidate_id"]] for row in chosen
        },
        "rejected_shortlist_ids": [
            row["candidate_id"]
            for row in rows
            if verdicts.get(row["candidate_id"]) != "FINALIST"
        ],
        "verdicts": verdicts,
        "reasons": reasons,
        "frontier_ids": [row["candidate_id"] for row in frontier],
        "rule": {
            "minimum_finalists": minimum,
            "maximum_finalists": maximum,
            "capacity_proxy": "parameter_count",
            "inputs": list(FINALIST_INPUT_KEYS),
            "forbidden_input_substrings": list(FORBIDDEN_INPUT_SUBSTRINGS),
            "strength_is_not_an_input": (
                "no playing-strength, win-rate, Elo or match-result field is reachable "
                "from recommend_finalists; see finalist_inputs()"
            ),
            "excluded_if_any": [
                "not numerically stable in the integrated pipeline",
                "no sustained production-recording throughput",
            ],
            "dominated_if": (
                "another viable candidate has >= parameters, >= recording positions/s and "
                "<= process RSS, with at least one strict improvement"
            ),
            "selection": (
                "smallest and largest of the surviving frontier, plus the "
                "highest-recording-throughput middle candidate when three fit"
            ),
        },
    }


__all__ = [
    "BENCHMARK_VERSION",
    "BYTES_PER_GIB",
    "COLLECTION_POLICY_PREFIX",
    "EXTERNAL_FREE_BYTES",
    "FAILURE_CATEGORIES",
    "FINALIST_INPUT_KEYS",
    "FORBIDDEN_INPUT_SUBSTRINGS",
    "INTERNAL_FREE_BYTES",
    "MODE_COLLECTION",
    "MODE_RECORDING",
    "POLICY_REEVALUATION_TOLERANCE",
    "STARTING_ENVIRONMENTS",
    "STARTING_LEGALITY",
    "STARTING_PRECISION",
    "STARTING_SNAPSHOT_INTERVAL",
    "STARTING_WORKERS",
    "STORAGE_MEASURE_STEPS",
    "STORAGE_SAMPLE_STEPS",
    "STORAGE_SETTLED_SPREAD",
    "STORAGE_WARMUP_STEPS",
    "SWEEP_POINTS",
    "FrameGateReport",
    "ReconstructionReport",
    "build_pipeline_candidate",
    "candidate_bottleneck_ratio",
    "candidate_configuration",
    "candidate_identity",
    "capture_published_batch",
    "classify_failure",
    "collection_policy_version",
    "empty_failure_counts",
    "finalist_inputs",
    "measure_candidate_configuration",
    "measure_inference_capacity",
    "measure_simulation_capacity",
    "measure_storage_rate",
    "open_candidate_coordinator",
    "probe_head_finiteness",
    "recommend_finalists",
    "reconstruct_stored_games",
    "run_frame_correctness_gate",
    "storage_projection",
]
