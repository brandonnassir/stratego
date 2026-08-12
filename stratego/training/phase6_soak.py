"""Phase 6 Agent 6: the one-hour production soak and the decision that follows.

Specification source: `06_AGENT_6_SOAK_AND_DECISION.md`.

What this module does
---------------------
1. Runs one finalist candidate continuously for approximately one hour through
   Agent 4's accepted bulk-synchronous production-recording pipeline, sampling a
   time series of throughput, correctness and memory counters as it goes.
2. Turns that time series into the four things the decision needs: a sustained
   recording-inclusive throughput, a sustained trajectory byte rate, a drift
   figure and a memory-growth verdict.
3. Projects the user's exact 168-hour final run from the sustained figures.
4. Analyses the projection against the user's declared storage capacity.
5. Selects one exact primary and one exact fallback architecture from the
   measured capacity/compute frontier.

What it deliberately does not do
--------------------------------
It does not train, it does not tune, it does not touch the engine, and no
playing-strength quantity is reachable from any selection rule here -- the same
guarantee Agent 4's `recommend_finalists` carries, enforced the same way.

Why the soak is not `run_neural_schedule`
-----------------------------------------
Agent 5's ~449 positions/s is *evaluation* throughput: whole games played to
termination through `play_match`, one forward pass per decision, batch 1. The
production collection pipeline is a different machine -- 1,536 environments
advanced in lockstep, one dispatch of up to 2,048 rows per global step -- and it
is the machine the 168-hour run will actually use. The soak therefore drives
Agent 4's pipeline, and Agent 5's number is never mixed into a collection or
training projection.

Steady state versus the whole hour
----------------------------------
Two windows, reported separately and never averaged together:

- **The whole hour** is the stability window. Every correctness counter --
  illegal actions, frame errors, worker and model failures, non-finite outputs,
  reconstruction mismatches -- covers every global step from the first.
- **The post-warmup window** is the measurement window. A pool starts with all
  1,536 environments at ply 0, so until the slots desynchronise the run is
  sealing almost no games while recording almost all of their decisions. Agent 4
  measured that this understates the sustained byte rate by roughly an order of
  magnitude. Sustained throughput, games/s and GiB/hour therefore come from the
  steps after `SOAK_WARMUP_STEPS`, and the 168-hour projection is built from
  those alone.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import numpy as np
import psutil
import torch

from ..engine.constants import ACTION_SPACE_SIZE
from ..model.architecture_configs import (
    ARCHITECTURE_FAMILY,
    ARCHITECTURE_FAMILY_VERSION,
    BELIEF_HEAD,
    FAMILY_CONSTANTS,
    FAMILY_INITIALIZATION_SEED,
    POLICY_HEAD,
    VALUE_HEAD,
    candidate_config,
)
from ..model.contract import MODEL_CONTRACT_VERSION, validate_policy_logits
from .coordinator import (
    ACTION_FRAME_NORMALIZED,
    CoordinatorConfig,
    CoordinatorError,
    SelfPlayCoordinator,
    mps_memory_bytes,
)
from .end_to_end_benchmark import swap_bytes
from .shared_buffers import STATUS_ACTIVE
from .phase6_pipeline_benchmark import (
    BYTES_PER_GIB,
    EXTERNAL_FREE_BYTES,
    FORBIDDEN_INPUT_SUBSTRINGS,
    INTERNAL_FREE_BYTES,
    build_pipeline_candidate,
    candidate_configuration,
    classify_failure,
    empty_failure_counts,
)
from .trajectory import TRAJECTORY_VERSION
from .worker_pool import WorkerPoolError

SOAK_VERSION = "agent_06_soak_0.1.0"

#: Agent 4's best defensible production topology, adopted unchanged. Batch 2,048
#: rather than Agent 3's 1,024 because Agent 4 measured the chunking interaction
#: a standalone curve cannot see: at 2,048 the whole ready set goes to the device
#: in one dispatch and the frame conversion is applied to one contiguous block.
SOAK_WORKERS = 10
SOAK_ENVIRONMENTS = 1536
SOAK_INFERENCE_BATCH = 2048
SOAK_PRECISION = "float16"
SOAK_LEGALITY = "dense"
SOAK_SNAPSHOT_INTERVAL = 32

#: One continuous hour.
SOAK_SECONDS = 3600.0

#: How often the time series is sampled. Sixty samples an hour is enough to see
#: drift and a memory trend without the sampling itself becoming a cost.
SOAK_SAMPLE_SECONDS = 60.0

#: Steps discarded from the *measurement* window and from that window alone.
#:
#: Counted in steps rather than seconds because desynchronising 1,536 slots takes
#: a number of mean game lengths of simulated time, which is a step count, not a
#: wall-clock duration -- a seconds-based warmup would give the slowest candidate
#: the least settled measurement.
#:
#: Agent 4 used 1,100 steps, about two mean game lengths, which is enough for the
#: trajectory *byte* rate. This is higher because the soak also has to answer a
#: question Agent 4 never asked: whether resident memory is still growing. A pool
#: starts with every slot at ply 0 and they grow their trajectory builders in
#: lockstep, so a cold run's resident set climbs to the envelope of a fully
#: synchronised population before the slots spread out. Agent 6's calibration
#: pilot measured that climb still converging at step ~1,300, decaying with a
#: time constant near two mean game lengths. Six mean game lengths puts the
#: measurement window past it, and still leaves well over 3,000 seconds of the
#: hour inside the window.
SOAK_WARMUP_STEPS = 3_000

#: Rows fed to the periodic non-finite probe. The value and belief heads are
#: already refused by `ModelOutputs.validated` inside every forward pass a timed
#: step makes, so those two are covered continuously and for free. The policy
#: head is deliberately *not* finiteness-checked by the contract -- a model may
#: score an illegal index arbitrarily -- so this probe exists to cover it, on
#: real published positions, at pipeline precision, throughout the run.
SOAK_FINITENESS_PROBE_ROWS = 512

#: Per-worker live-verification budget. Set high enough that verification never
#: exhausts inside the hour: the point of a soak is that the correctness layer is
#: running at minute 59 as well as minute 1.
SOAK_VERIFY_TARGET_DECISIONS = 1_000_000

#: One digested game per worker at a time. Digesting costs roughly 35x plain
#: recording and a verified game replays its whole history when it seals, so
#: concurrency is what decides both the steady cost and the size of the latency
#: spike a seal produces. One is the smallest value that still keeps a
#: verification live in every worker for the whole hour.
SOAK_MAX_CONCURRENT_VERIFICATIONS = 1

#: Declared before the soak ran. A soak cannot prove the absence of a leak; it
#: can show that any trend over an hour is inside sampling scatter. The rule is
#: the second half of the measurement window against the first half, because that
#: is robust to the cold-start ramp in a way a whole-window regression is not.
MEMORY_GROWTH_TOLERANCE = 0.02

#: Throughput drift is expected to be small and is allowed to be non-zero, but it
#: has to be reported and it has to be explainable. This is the magnitude above
#: which the soak stops calling it small.
THROUGHPUT_DRIFT_TOLERANCE = 0.10

#: The user's official final run.
FINAL_RUN_HOURS = 168.0
FINAL_RUN_SECONDS = 604800.0

#: How far below the best rung a step of the capacity ladder may score and still
#: be taken. Declared before any Agent 6 measurement was read. At 0.5 a step must
#: buy at least half the capacity-per-throughput that the best available step
#: buys; below that the ladder is charging disproportionately for its next
#: increment, which is what the instruction's "knee" means.
KNEE_EFFICIENCY_FLOOR = 0.5


class SoakError(CoordinatorError):
    """The soak could not run, or lost an invariant while running."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def soak_configuration(
    candidate_id: str,
    *,
    workers: int = SOAK_WORKERS,
    environments: int = SOAK_ENVIRONMENTS,
    inference_batch_size: int = SOAK_INFERENCE_BATCH,
    root_seed: int = 60_006,
    verify_target_decisions: int = SOAK_VERIFY_TARGET_DECISIONS,
    max_concurrent_verifications: int = SOAK_MAX_CONCURRENT_VERIFICATIONS,
) -> CoordinatorConfig:
    """Agent 4's production topology, with recording and verification both on.

    `detailed_timing` is off: a soak measures what production sustains, and the
    per-stage synchronisations exist to attribute time, not to produce it. The
    stage fractions are still emitted so the row shape matches Agent 4's, but
    they are flagged as unattributed and no conclusion here rests on them.
    """
    config = candidate_configuration(
        candidate_id,
        workers=workers,
        environments=environments,
        inference_batch_size=inference_batch_size,
        precision=SOAK_PRECISION,
        legality=SOAK_LEGALITY,
        record_trajectories=True,
        snapshot_interval=SOAK_SNAPSHOT_INTERVAL,
        detailed_timing=False,
        root_seed=root_seed,
        verify_target_decisions=verify_target_decisions,
        max_concurrent_verifications=max_concurrent_verifications,
        retain_games=0,
    )
    if config.action_frame != ACTION_FRAME_NORMALIZED:
        raise SoakError(
            f"the soak must run {MODEL_CONTRACT_VERSION} normalized model "
            f"actions; got {config.action_frame!r}"
        )
    if not config.verify_sampled_legality:
        raise SoakError("the soak requires per-step sampled-legality verification")
    return config


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def process_memory() -> dict:
    """Current -- not peak -- resident memory for the coordinator and its workers.

    `resource.getrusage` reports a high-water mark, which can only ever rise and
    so cannot distinguish a leak from a single early allocation. A growth gate
    needs the instantaneous value, and it needs the workers separately: the
    coordinator holds Metal and the model, the workers hold the games.
    """
    process = psutil.Process()
    coordinator_rss = int(process.memory_info().rss)
    worker_rss = 0
    worker_count = 0
    for child in process.children(recursive=True):
        try:
            worker_rss += int(child.memory_info().rss)
            worker_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):  # pragma: no cover
            continue
    return {
        "coordinator_rss_bytes": coordinator_rss,
        "worker_rss_bytes": worker_rss,
        "worker_processes": worker_count,
        "total_rss_bytes": coordinator_rss + worker_rss,
    }


def probe_live_finiteness(coordinator: SelfPlayCoordinator, *, rows: int) -> dict:
    """All three heads on positions the pool published, through the live model.

    Uses the coordinator's own model instance rather than a freshly built one, so
    what is checked is the weights the soak is actually running -- a second copy
    would prove nothing about the first.
    """
    buffers = coordinator.pool.buffers
    config = coordinator.config
    active = np.flatnonzero(buffers.status == STATUS_ACTIVE)
    if active.size == 0:
        return {"rows_checked": 0, "logits_checked": 0, "nonfinite_outputs": 0}
    chosen = active[: min(rows, active.size)]
    flat = buffers.observations.reshape(
        config.num_environments, buffers.observations.shape[1], -1
    )
    tokens = torch.from_numpy(np.ascontiguousarray(flat[chosen])).to(coordinator.device)
    tokens = tokens.transpose(1, 2).contiguous().to(coordinator.dtype)
    with torch.no_grad():
        outputs = coordinator.model(tokens)
        # Counted before the contract validator runs, so a non-finite row is
        # recorded as a number rather than only as the exception the validator
        # raises. Both end the soak; only one of them says how many.
        nonfinite = {
            "policy": int((~torch.isfinite(outputs.policy_logits)).sum().item()),
            "value": int((~torch.isfinite(outputs.value_logits)).sum().item()),
            "belief": int((~torch.isfinite(outputs.belief_logits)).sum().item()),
        }
        validate_policy_logits(
            outputs.policy_logits, batch=int(chosen.size), require_finite=True
        )
    del tokens, outputs
    return {
        "rows_checked": int(chosen.size),
        "logits_checked": int(chosen.size) * (ACTION_SPACE_SIZE + 3 + 100 * 12),
        "nonfinite_by_head": nonfinite,
        "nonfinite_outputs": sum(nonfinite.values()),
    }


def _ordinary_least_squares(x: list[float], y: list[float]) -> dict:
    """Slope, intercept and R^2. Returns a zero slope for a degenerate input."""
    n = len(x)
    if n < 2:
        return {"slope": 0.0, "intercept": y[0] if y else 0.0, "r_squared": 0.0, "n": n}
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    sxx = sum((value - mean_x) ** 2 for value in x)
    if sxx <= 0:
        return {"slope": 0.0, "intercept": mean_y, "r_squared": 0.0, "n": n}
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    syy = sum((value - mean_y) ** 2 for value in y)
    residual = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y))
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": (1.0 - residual / syy) if syy > 0 else 0.0,
        "n": n,
    }


def half_over_half(values: list[float]) -> dict:
    """Second half against first half. The declared memory-growth statistic."""
    if len(values) < 2:
        return {
            "first_half_mean": values[0] if values else 0.0,
            "second_half_mean": values[0] if values else 0.0,
            "relative_change": 0.0,
            "samples": len(values),
        }
    split = len(values) // 2
    first = statistics.fmean(values[:split])
    second = statistics.fmean(values[split:])
    return {
        "first_half_mean": first,
        "second_half_mean": second,
        "absolute_change": second - first,
        "relative_change": (second - first) / first if first else 0.0,
        "samples": len(values),
    }


def growth_report(label: str, elapsed: list[float], values: list[float]) -> dict:
    """One quantity's trend over the measurement window, against the declared rule."""
    halves = half_over_half(values)
    fit = _ordinary_least_squares(elapsed, values)
    mean = statistics.fmean(values) if values else 0.0
    return {
        "quantity": label,
        "samples": len(values),
        "minimum": min(values) if values else 0.0,
        "maximum": max(values) if values else 0.0,
        "mean": mean,
        "first_sample": values[0] if values else 0.0,
        "last_sample": values[-1] if values else 0.0,
        **halves,
        "slope_per_second": fit["slope"],
        "slope_per_hour": fit["slope"] * 3600.0,
        "relative_slope_per_hour": (fit["slope"] * 3600.0 / mean) if mean else 0.0,
        "r_squared": fit["r_squared"],
        "tolerance": MEMORY_GROWTH_TOLERANCE,
        "within_tolerance": abs(halves["relative_change"]) <= MEMORY_GROWTH_TOLERANCE,
    }


# ---------------------------------------------------------------------------
# The soak
# ---------------------------------------------------------------------------


@dataclass
class _CumulativeSample:
    """The raw cumulative counters at one sampling instant."""

    elapsed: float
    step: int
    positions: int
    games: int
    record_bytes: int
    decisions: int
    verified_decisions: int


def run_soak(
    candidate_id: str,
    config: CoordinatorConfig,
    *,
    seconds: float = SOAK_SECONDS,
    sample_seconds: float = SOAK_SAMPLE_SECONDS,
    warmup_steps: int = SOAK_WARMUP_STEPS,
    probe_rows: int = SOAK_FINITENESS_PROBE_ROWS,
    device=None,
    seed: int = FAMILY_INITIALIZATION_SEED,
    progress=None,
) -> dict:
    """Run one candidate continuously and return the samples plus the summary.

    The loop never swallows a failure. Every hard gate this soak reports on --
    an illegal selection, a frame conversion that did not invert, a non-finite
    policy row, a dead worker, a reconstruction mismatch -- is raised by the
    pipeline rather than tolerated, so the counters are produced by classifying
    an exception that ended the run, not by counting faults it survived. A soak
    that reaches the end of its hour is a soak in which none of them happened.
    """
    if not config.record_trajectories:
        raise SoakError("the soak must run the production-recording path")
    model = build_pipeline_candidate(candidate_id, seed=seed)
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    coordinator = SelfPlayCoordinator(
        config, device=device, model=model, model_label=candidate_id
    )

    samples: list[dict] = []
    failures = empty_failure_counts()
    status = "ok"
    error_text = None
    error_category = None
    probe_seconds = 0.0
    probe_logits = 0
    probe_rows_checked = 0
    worker_liveness_checks = 0
    swap_start = swap_bytes()
    metal_start = mps_memory_bytes()

    coordinator.start()
    started = time.perf_counter()
    previous = _CumulativeSample(0.0, 0, 0, 0, 0, 0, 0)
    next_sample_at = sample_seconds
    warmup_marker: _CumulativeSample | None = None

    try:
        while True:
            coordinator.step()
            elapsed = time.perf_counter() - started

            # The warmup boundary is a step count, and it is recorded as its own
            # baseline so the measurement window can be differenced from the raw
            # cumulative counters rather than reassembled from per-window values.
            if warmup_marker is None and coordinator.step_index >= warmup_steps:
                recording = coordinator.pool.recording_totals()
                warmup_marker = _CumulativeSample(
                    elapsed=elapsed,
                    step=coordinator.step_index,
                    positions=int(coordinator.totals.positions),
                    games=int(coordinator.games_finished),
                    record_bytes=int(recording["total_record_bytes"]),
                    decisions=int(recording["total_decisions_recorded"]),
                    verified_decisions=int(recording["total_verified_decisions"]),
                )

            if elapsed < next_sample_at and elapsed < seconds:
                continue

            probe_started = time.perf_counter()
            probe = probe_live_finiteness(coordinator, rows=probe_rows)
            probe_seconds += time.perf_counter() - probe_started
            probe_logits += probe["logits_checked"]
            probe_rows_checked += probe["rows_checked"]
            failures["nonfinite_outputs"] += probe["nonfinite_outputs"]

            liveness = coordinator.pool.worker_liveness()
            worker_liveness_checks += len(liveness)
            dead = [index for index, alive in enumerate(liveness) if not alive]
            if dead:
                # Raised as a pool error rather than a soak error so that
                # `classify_failure` books it against `worker_errors`, which is
                # the counter the hard gate reads.
                raise WorkerPoolError(f"worker(s) {dead} are no longer running")

            recording = coordinator.pool.recording_totals()
            memory = process_memory()
            metal = mps_memory_bytes()
            swap = swap_bytes()
            current = _CumulativeSample(
                elapsed=elapsed,
                step=coordinator.step_index,
                positions=int(coordinator.totals.positions),
                games=int(coordinator.games_finished),
                record_bytes=int(recording["total_record_bytes"]),
                decisions=int(recording["total_decisions_recorded"]),
                verified_decisions=int(recording["total_verified_decisions"]),
            )
            window = max(current.elapsed - previous.elapsed, 1e-9)
            mismatches = int(recording["total_reconstruction_mismatches"])
            if mismatches:
                raise SoakError(
                    f"{mismatches} trajectory reconstruction mismatch(es) reported "
                    f"by the worker pool"
                )

            row = {
                "candidate_id": candidate_id,
                "sample_index": len(samples),
                "elapsed_seconds": current.elapsed,
                "global_step": current.step,
                "in_measured_window": current.step > warmup_steps,
                "positions": current.positions,
                "window_positions": current.positions - previous.positions,
                "positions_per_second": (current.positions - previous.positions) / window,
                "cumulative_positions_per_second": current.positions / current.elapsed,
                "games": current.games,
                "window_games": current.games - previous.games,
                "games_per_second": (current.games - previous.games) / window,
                "mean_game_length": (
                    (current.positions - previous.positions)
                    / max(current.games - previous.games, 1)
                ),
                "terminal_reason_counts": dict(coordinator.terminal_reason_counts),
                "worker_failures": failures["worker_errors"],
                "model_failures": failures["model_errors"],
                "nonfinite_outputs": failures["nonfinite_outputs"],
                "illegal_actions": failures["illegal_actions"],
                "action_frame_errors": failures["action_frame_errors"],
                "workers_alive": int(sum(liveness)),
                "sampled_legality_checks": current.positions,
                "verified_decisions": current.verified_decisions,
                "verified_games": int(recording["total_verified_games"]),
                "reconstruction_mismatches": mismatches,
                "decisions_recorded": current.decisions,
                "trajectory_bytes": current.record_bytes,
                "window_trajectory_bytes": current.record_bytes - previous.record_bytes,
                "gib_per_hour": (
                    (current.record_bytes - previous.record_bytes) / window
                )
                * 3600.0
                / BYTES_PER_GIB,
                "bytes_per_decision": (
                    (current.record_bytes - previous.record_bytes)
                    / max(current.decisions - previous.decisions, 1)
                ),
                "snapshot_count": int(recording["total_snapshot_count"]),
                "coordinator_rss_bytes": memory["coordinator_rss_bytes"],
                "worker_rss_bytes": memory["worker_rss_bytes"],
                "total_rss_bytes": memory["total_rss_bytes"],
                "worker_processes": memory["worker_processes"],
                "shared_memory_bytes": int(coordinator.pool.buffers.nbytes),
                "metal_current_allocated_bytes": int(
                    metal.get("current_allocated_bytes", 0)
                ),
                "metal_driver_allocated_bytes": int(
                    metal.get("driver_allocated_bytes", 0)
                ),
                "swap_used_bytes": int(swap.get("swap_used_bytes", 0)),
                "probe_rows_checked": probe["rows_checked"],
                "probe_logits_checked": probe["logits_checked"],
                "cumulative_probe_seconds": probe_seconds,
            }
            samples.append(row)
            if progress is not None:
                progress(row)
            previous = current
            next_sample_at += sample_seconds
            if elapsed >= seconds:
                break
    except BaseException as error:  # noqa: BLE001 - a failed soak is a result
        status = "error"
        error_text = f"{type(error).__name__}: {error}"
        error_category = classify_failure(error)
        failures[error_category] = failures.get(error_category, 0) + 1
    finally:
        try:
            pool_totals = coordinator.pool.recording_totals()
        except Exception:  # noqa: BLE001 - best effort before shutdown
            pool_totals = {}
        try:
            shutdown_totals = coordinator.shutdown()
        except Exception as error:  # noqa: BLE001 - shutdown failure is a result
            shutdown_totals = {}
            status = "error" if status == "ok" else status
            error_text = error_text or f"shutdown: {type(error).__name__}: {error}"

    total_seconds = time.perf_counter() - started
    swap_end = swap_bytes()
    metal_end = mps_memory_bytes()
    totals = {**pool_totals, **shutdown_totals}

    summary = summarize_soak(
        candidate_id=candidate_id,
        config=config,
        samples=samples,
        warmup_marker=warmup_marker,
        warmup_steps=warmup_steps,
        total_seconds=total_seconds,
        totals=totals,
        failures=failures,
        parameters=parameters,
        status=status,
        error_text=error_text,
        error_category=error_category,
        probe_seconds=probe_seconds,
        probe_logits=probe_logits,
        probe_rows_checked=probe_rows_checked,
        worker_liveness_checks=worker_liveness_checks,
        swap_start=swap_start,
        swap_end=swap_end,
        metal_start=metal_start,
        metal_end=metal_end,
        requested_seconds=seconds,
    )
    summary["samples"] = samples
    return summary


def summarize_soak(
    *,
    candidate_id: str,
    config: CoordinatorConfig,
    samples: list[dict],
    warmup_marker,
    warmup_steps: int,
    total_seconds: float,
    totals: dict,
    failures: dict,
    parameters: int,
    status: str,
    error_text,
    error_category,
    probe_seconds: float,
    probe_logits: int,
    probe_rows_checked: int,
    worker_liveness_checks: int,
    swap_start: dict,
    swap_end: dict,
    metal_start: dict,
    metal_end: dict,
    requested_seconds: float,
) -> dict:
    """Turn the raw samples into the headline figures and the gate verdicts."""
    measured = [sample for sample in samples if sample["in_measured_window"]]
    last = samples[-1] if samples else None

    steady: dict = {}
    if measured and warmup_marker is not None and last is not None:
        window_seconds = last["elapsed_seconds"] - warmup_marker.elapsed
        if window_seconds > 0:
            steady_positions = last["positions"] - warmup_marker.positions
            steady_games = last["games"] - warmup_marker.games
            steady_bytes = last["trajectory_bytes"] - warmup_marker.record_bytes
            steady_decisions = last["decisions_recorded"] - warmup_marker.decisions
            steady = {
                "window_seconds": window_seconds,
                "window_steps": last["global_step"] - warmup_marker.step,
                "warmup_seconds": warmup_marker.elapsed,
                "warmup_steps": warmup_marker.step,
                "positions": steady_positions,
                "games": steady_games,
                "decisions": steady_decisions,
                "record_bytes": steady_bytes,
                "positions_per_second": steady_positions / window_seconds,
                "games_per_second": steady_games / window_seconds,
                "mean_game_length": (
                    steady_positions / steady_games if steady_games else 0.0
                ),
                "bytes_per_second": steady_bytes / window_seconds,
                "gib_per_hour": (steady_bytes / window_seconds) * 3600.0 / BYTES_PER_GIB,
                "bytes_per_decision": (
                    steady_bytes / steady_decisions if steady_decisions else 0.0
                ),
                "samples_in_window": len(measured),
            }

    elapsed = [sample["elapsed_seconds"] for sample in measured]
    drift = {}
    if len(measured) >= 2:
        rates = [sample["positions_per_second"] for sample in measured]
        mean_rate = statistics.fmean(rates)
        fit = _ordinary_least_squares(elapsed, rates)
        halves = half_over_half(rates)
        drift = {
            "mean_positions_per_second": mean_rate,
            "minimum_positions_per_second": min(rates),
            "maximum_positions_per_second": max(rates),
            "stdev_positions_per_second": statistics.pstdev(rates),
            "coefficient_of_variation": (
                statistics.pstdev(rates) / mean_rate if mean_rate else 0.0
            ),
            "slope_positions_per_second_per_hour": fit["slope"] * 3600.0,
            "relative_drift_per_hour": (
                fit["slope"] * 3600.0 / mean_rate if mean_rate else 0.0
            ),
            "r_squared": fit["r_squared"],
            **{f"half_over_half_{key}": value for key, value in halves.items()},
            "tolerance": THROUGHPUT_DRIFT_TOLERANCE,
            "small_and_stable": abs(
                fit["slope"] * 3600.0 / mean_rate if mean_rate else 0.0
            )
            <= THROUGHPUT_DRIFT_TOLERANCE,
        }

    memory_growth = {}
    if len(measured) >= 2:
        for label, key in (
            ("coordinator_rss_bytes", "coordinator_rss_bytes"),
            ("worker_rss_bytes", "worker_rss_bytes"),
            ("total_rss_bytes", "total_rss_bytes"),
            ("metal_driver_allocated_bytes", "metal_driver_allocated_bytes"),
            ("metal_current_allocated_bytes", "metal_current_allocated_bytes"),
            ("shared_memory_bytes", "shared_memory_bytes"),
        ):
            memory_growth[label] = growth_report(
                label, elapsed, [sample[key] for sample in measured]
            )

    swap_used = max(
        int(swap_end.get("swap_used_bytes", 0)),
        max((sample["swap_used_bytes"] for sample in samples), default=0),
    )
    reconstruction_mismatches = int(totals.get("total_reconstruction_mismatches", 0))
    verified_decisions = int(totals.get("total_verified_decisions", 0))
    verified_games = int(totals.get("total_verified_games", 0))

    completed = (
        status == "ok"
        and last is not None
        and last["elapsed_seconds"] >= requested_seconds * 0.98
    )

    gates = {
        "soak_completed_continuously": completed,
        "illegal_actions_zero": failures["illegal_actions"] == 0,
        "action_frame_mismatches_zero": failures["action_frame_errors"] == 0,
        "reconstruction_mismatches_zero": reconstruction_mismatches == 0,
        "worker_failures_zero": failures["worker_errors"] == 0,
        "model_mps_failures_zero": failures["model_errors"] == 0,
        "nonfinite_production_outputs_zero": failures["nonfinite_outputs"] == 0,
        "other_failures_zero": failures["other_errors"] == 0,
        "swap_zero": swap_used == 0,
        "no_unexplained_memory_growth": bool(memory_growth)
        and all(report["within_tolerance"] for report in memory_growth.values()),
        "reconstruction_ran_throughout": verified_decisions > 0
        and bool(measured)
        and measured[-1]["verified_decisions"] > measured[0]["verified_decisions"],
        "steady_state_window_measured": bool(steady),
    }

    return {
        "soak_version": SOAK_VERSION,
        "candidate_id": candidate_id,
        "parameters": parameters,
        "status": status,
        "error": error_text,
        "error_category": error_category,
        "configuration": config.as_dict(),
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "architecture_family": ARCHITECTURE_FAMILY,
        "architecture_family_version": ARCHITECTURE_FAMILY_VERSION,
        "trajectory_version": TRAJECTORY_VERSION,
        "requested_seconds": requested_seconds,
        "total_seconds": total_seconds,
        "sample_count": len(samples),
        "whole_run": {
            "positions": last["positions"] if last else 0,
            "games": last["games"] if last else 0,
            "global_steps": last["global_step"] if last else 0,
            "decisions_recorded": last["decisions_recorded"] if last else 0,
            "trajectory_bytes": last["trajectory_bytes"] if last else 0,
            "snapshot_count": last["snapshot_count"] if last else 0,
            "positions_per_second": (
                last["positions"] / last["elapsed_seconds"] if last else 0.0
            ),
            "terminal_reason_counts": last["terminal_reason_counts"] if last else {},
        },
        "steady_state": steady,
        "throughput_drift": drift,
        "memory_growth": memory_growth,
        "failures": failures,
        "correctness": {
            "sampled_legality_checks": last["positions"] if last else 0,
            "worker_liveness_checks": worker_liveness_checks,
            "verified_games": verified_games,
            "verified_decisions": verified_decisions,
            "reconstruction_mismatches": reconstruction_mismatches,
            "mismatch_details": list(totals.get("mismatch_details", ()) or ()),
            "finiteness_probe_rows": probe_rows_checked,
            "finiteness_probe_logits": probe_logits,
            "finiteness_probe_seconds": probe_seconds,
            "finiteness_probe_fraction_of_wall": (
                probe_seconds / total_seconds if total_seconds else 0.0
            ),
            "games_joined_late": int(totals.get("total_games_joined_late", 0)),
        },
        "memory": {
            "swap_used_bytes_start": int(swap_start.get("swap_used_bytes", 0)),
            "swap_used_bytes_end": int(swap_end.get("swap_used_bytes", 0)),
            "swap_used_bytes_max": swap_used,
            "swap_total_bytes": int(swap_end.get("swap_total_bytes", 0)),
            "metal_start": metal_start,
            "metal_end": metal_end,
            "shared_memory_bytes": (
                samples[-1]["shared_memory_bytes"] if samples else 0
            ),
        },
        "recording_totals": {
            key: value
            for key, value in totals.items()
            if isinstance(value, (int, float))
        },
        "completion_gates": gates,
        "gates_true": sum(1 for value in gates.values() if value),
        "gates_total": len(gates),
        "passed": all(gates.values()),
    }


# ---------------------------------------------------------------------------
# The 168-hour projection
# ---------------------------------------------------------------------------


def weekly_projection(
    *,
    candidate_id: str,
    positions_per_second: float,
    games_per_second: float,
    bytes_per_second: float,
    bytes_per_decision: float,
    checkpoint_bytes: int,
    training_examples_per_second: float,
    measurement_source: str,
    measured_seconds: float,
) -> dict:
    """Extrapolate one measured sustained rate to exactly 168 hours.

    Measured inputs and extrapolated outputs are kept in separate blocks on
    purpose. Nothing here is a claim about learning: it is arithmetic on a
    throughput, and a trained network will produce different game lengths and
    therefore different totals.
    """
    if positions_per_second <= 0:
        raise ValueError("a projection needs a positive measured throughput")
    positions = positions_per_second * FINAL_RUN_SECONDS
    games = games_per_second * FINAL_RUN_SECONDS
    trajectory_bytes = bytes_per_second * FINAL_RUN_SECONDS
    gib_per_hour = bytes_per_second * 3600.0 / BYTES_PER_GIB

    checkpoint_schedules = {}
    for label, hours in (
        ("hourly", 1.0),
        ("every_4_hours", 4.0),
        ("every_12_hours", 12.0),
        ("daily", 24.0),
    ):
        count = int(FINAL_RUN_HOURS / hours)
        checkpoint_schedules[label] = {
            "interval_hours": hours,
            "checkpoints_retained": count,
            "bytes": count * checkpoint_bytes,
            "gib": count * checkpoint_bytes / BYTES_PER_GIB,
        }

    return {
        "candidate_id": candidate_id,
        "final_run_hours": FINAL_RUN_HOURS,
        "final_run_seconds": FINAL_RUN_SECONDS,
        "measured": {
            "source": measurement_source,
            "measured_seconds": measured_seconds,
            "recording_inclusive_positions_per_second": positions_per_second,
            "games_per_second": games_per_second,
            "trajectory_bytes_per_second": bytes_per_second,
            "trajectory_gib_per_hour": gib_per_hour,
            "bytes_per_decision": bytes_per_decision,
            "checkpoint_bytes": int(checkpoint_bytes),
            "training_step_examples_per_second": training_examples_per_second,
        },
        "extrapolated": {
            "positions": positions,
            "games": games,
            "decisions": positions,
            "trajectory_bytes": trajectory_bytes,
            "trajectory_gib": trajectory_bytes / BYTES_PER_GIB,
            "trajectory_gib_per_24_hours": gib_per_hour * 24.0,
            "positions_per_24_hours": positions_per_second * 86400.0,
            "games_per_24_hours": games_per_second * 86400.0,
            "checkpoint_storage": checkpoint_schedules,
            "training_step_opportunities": {
                "note": (
                    "Agent 3 measured this backward-pass rate standalone, with no "
                    "simulator running. Collection and training contend for the "
                    "same single Metal device, so these are opportunity ceilings "
                    "for a training process that had the device to itself, not a "
                    "concurrent-throughput prediction."
                ),
                "examples_per_second": training_examples_per_second,
                "examples_if_training_ran_for_the_whole_week": (
                    training_examples_per_second * FINAL_RUN_SECONDS
                ),
                "epochs_over_the_projected_corpus": (
                    (training_examples_per_second * FINAL_RUN_SECONDS) / positions
                    if positions
                    else 0.0
                ),
            },
        },
        "extrapolation_is_not_a_learning_claim": (
            "These totals follow from a measured cost rate on random weights. "
            "They say how much data the machine can produce, not how strong the "
            "resulting network will be, and a trained network's game lengths will "
            "differ from the random-weight lengths these rates were measured at."
        ),
    }


def storage_analysis(
    *,
    candidate_id: str,
    trajectory_bytes_168h: float,
    checkpoint_bytes: int,
    measured_bytes_per_decision: float,
    compression_ratio: float | None = None,
    compression_source: str = "",
) -> dict:
    """The 168-hour trajectory production against the user's declared capacity."""
    internal = INTERNAL_FREE_BYTES
    external = EXTERNAL_FREE_BYTES
    analysis = {
        "candidate_id": candidate_id,
        "declared_capacity": {
            "internal_free_bytes": internal,
            "internal_free_gb": internal / 1000**3,
            "external_free_bytes": external,
            "external_free_gb": external / 1000**3,
            "preference": "preserve most games externally when practical",
        },
        "measured_production": {
            "trajectory_bytes_168h": trajectory_bytes_168h,
            "trajectory_gib_168h": trajectory_bytes_168h / BYTES_PER_GIB,
            "trajectory_gb_168h": trajectory_bytes_168h / 1000**3,
            "bytes_per_decision": measured_bytes_per_decision,
            "checkpoint_bytes": int(checkpoint_bytes),
        },
        "uncompressed": {
            "fraction_of_internal": trajectory_bytes_168h / internal,
            "fraction_of_external": trajectory_bytes_168h / external,
            "fits_internal": trajectory_bytes_168h <= internal,
            "fits_external": trajectory_bytes_168h <= external,
        },
    }
    if compression_ratio is not None:
        compressed = trajectory_bytes_168h * compression_ratio
        analysis["compressed"] = {
            "ratio": compression_ratio,
            "source": compression_source,
            "is_measured_not_assumed": True,
            "trajectory_bytes_168h": compressed,
            "trajectory_gib_168h": compressed / BYTES_PER_GIB,
            "trajectory_gb_168h": compressed / 1000**3,
            "fraction_of_external": compressed / external,
            "fits_external": compressed <= external,
            "fits_internal": compressed <= internal,
        }
    return analysis


# ---------------------------------------------------------------------------
# The capacity/compute frontier and the architecture decision
# ---------------------------------------------------------------------------

#: Every field the selection rule may read. Identical in spirit to Agent 4's
#: `FINALIST_INPUT_KEYS`: capacity proxy and measured cost, nothing else. A test
#: asserts no key here matches a strength-shaped substring, and a second test
#: adds a win-rate field to every summary and asserts the selection is unchanged.
SELECTION_INPUT_KEYS = (
    "candidate_id",
    "parameters",
    "standalone_float32_positions_per_second",
    "standalone_float16_positions_per_second",
    "training_examples_per_second",
    "collection_positions_per_second",
    "recording_positions_per_second",
    "gib_per_hour",
    "process_rss_bytes",
    "metal_memory_bytes",
    "checkpoint_bytes",
    "numerically_stable_float16",
    "bottleneck_ratio",
)


def selection_inputs(summary) -> dict:
    """The only fields the selection rule sees."""
    return {key: summary.get(key) for key in SELECTION_INPUT_KEYS}


def _relative_change(new: float, old: float) -> float:
    return (new - old) / old if old else 0.0


def neighbor_tradeoffs(summaries) -> list[dict]:
    """Percentage changes between neighbouring candidates, ordered by capacity."""
    ordered = sorted(summaries, key=lambda summary: summary["parameters"])
    rows: list[dict] = []
    for smaller, larger in zip(ordered, ordered[1:]):
        parameter_gain = _relative_change(larger["parameters"], smaller["parameters"])
        recording_loss = _relative_change(
            larger["recording_positions_per_second"],
            smaller["recording_positions_per_second"],
        )
        # Capacity bought per unit of throughput given up, on a log scale so the
        # comparison is symmetric in ratios rather than dominated by whichever
        # end of the ladder happens to be larger.
        efficiency = 0.0
        if recording_loss < 0 and parameter_gain > 0:
            efficiency = np.log1p(parameter_gain) / -np.log1p(recording_loss)
        rows.append(
            {
                "from": smaller["candidate_id"],
                "to": larger["candidate_id"],
                "parameters_change": parameter_gain,
                "parameter_ratio": larger["parameters"] / smaller["parameters"],
                "standalone_float16_inference_change": _relative_change(
                    larger["standalone_float16_positions_per_second"],
                    smaller["standalone_float16_positions_per_second"],
                ),
                "standalone_float32_inference_change": _relative_change(
                    larger["standalone_float32_positions_per_second"],
                    smaller["standalone_float32_positions_per_second"],
                ),
                "training_step_change": _relative_change(
                    larger["training_examples_per_second"],
                    smaller["training_examples_per_second"],
                ),
                "collection_change": _relative_change(
                    larger["collection_positions_per_second"],
                    smaller["collection_positions_per_second"],
                ),
                "recording_change": recording_loss,
                "memory_change": _relative_change(
                    larger["process_rss_bytes"], smaller["process_rss_bytes"]
                ),
                "metal_memory_change": _relative_change(
                    larger["metal_memory_bytes"], smaller["metal_memory_bytes"]
                ),
                "storage_rate_change": _relative_change(
                    larger["gib_per_hour"], smaller["gib_per_hour"]
                ),
                "capacity_per_recording_throughput_given_up": float(efficiency),
            }
        )
    return rows


def select_architectures(summaries, *, minimum_recording_positions: float = 4000.0) -> dict:
    """The capacity/compute knee, from a rule stated before the numbers existed.

    The knee is the last rung of the ladder whose capacity gain is *not*
    disproportionate to what it costs. Formally: walking up the candidates in
    parameter order, each step is scored by

        log(1 + parameter gain) / -log(1 + recording throughput change)

    -- capacity bought per unit of sustained production throughput given up --
    and the knee is the top of the last step whose score is at least
    `KNEE_EFFICIENCY_FLOOR` times the best step's score. Steps beyond it buy
    materially less capacity per unit of throughput than the ladder has already
    demonstrated is available, which is exactly what "disproportionate" means
    here.

    The fallback is then the next candidate *below* the primary that is still
    fully correct, numerically stable, and materially cheaper.

    Neither branch can see a playing-strength quantity: `SELECTION_INPUT_KEYS` is
    the complete list of fields either reads.
    """
    inputs = [selection_inputs(summary) for summary in summaries]
    viable = [
        row
        for row in inputs
        if row["numerically_stable_float16"]
        and (row["recording_positions_per_second"] or 0.0) >= minimum_recording_positions
    ]
    if len(viable) < 2:
        raise SoakError(
            "the frontier needs at least two viable candidates to choose a "
            "primary and a fallback"
        )
    ordered = sorted(viable, key=lambda row: row["parameters"])
    steps = neighbor_tradeoffs(ordered)
    if not steps:
        raise SoakError("no neighbouring pair to score")

    scores = [step["capacity_per_recording_throughput_given_up"] for step in steps]
    best = max(scores)
    floor = KNEE_EFFICIENCY_FLOOR * best
    knee_index = 0
    for index, score in enumerate(steps):
        if score["capacity_per_recording_throughput_given_up"] >= floor:
            knee_index = index + 1
        else:
            break

    primary = ordered[knee_index]
    fallback = ordered[knee_index - 1] if knee_index > 0 else ordered[0]
    if fallback["candidate_id"] == primary["candidate_id"]:
        raise SoakError("primary and fallback resolved to the same candidate")

    return {
        "rule": {
            "capacity_proxy": "parameter_count",
            "cost_axis": "sustained recording-inclusive collection positions/s",
            "step_score": (
                "log(1 + parameter gain) / -log(1 + recording throughput change)"
            ),
            "knee": (
                "the top of the last step scoring at least "
                f"{KNEE_EFFICIENCY_FLOOR} x the best step"
            ),
            "efficiency_floor": KNEE_EFFICIENCY_FLOOR,
            "minimum_recording_positions_per_second": minimum_recording_positions,
            "fallback": "the next candidate below the primary on the same frontier",
            "inputs": list(SELECTION_INPUT_KEYS),
            "strength_is_not_an_input": (
                "no playing-strength, win-rate, Elo or match-result field is "
                "reachable from select_architectures; see selection_inputs()"
            ),
            "forbidden_input_substrings": list(FORBIDDEN_INPUT_SUBSTRINGS),
        },
        "ordered_candidate_ids": [row["candidate_id"] for row in ordered],
        "excluded_candidate_ids": [
            row["candidate_id"]
            for row in inputs
            if row["candidate_id"] not in {entry["candidate_id"] for entry in ordered}
        ],
        "step_scores": steps,
        "best_step_score": best,
        "efficiency_floor_value": floor,
        "knee_index": knee_index,
        "primary_id": primary["candidate_id"],
        "fallback_id": fallback["candidate_id"],
    }


def architecture_record(candidate_id: str, *, extra: dict | None = None) -> dict:
    """The exact frozen configuration of one selected architecture."""
    config = candidate_config(candidate_id)
    record = {
        "candidate_id": candidate_id,
        "architecture_family": ARCHITECTURE_FAMILY,
        "architecture_family_version": ARCHITECTURE_FAMILY_VERSION,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "initialization_seed": FAMILY_INITIALIZATION_SEED,
        "width": config.width,
        "blocks": config.blocks,
        "heads": config.heads,
        "feed_forward_width": config.feed_forward_width,
        "position_encoding": config.position_encoding,
        "normalization": config.normalization,
        "policy_head": POLICY_HEAD,
        "value_head": VALUE_HEAD,
        "belief_head": BELIEF_HEAD,
        "input_channels": config.input_channels,
        "board_tokens": config.board_tokens,
        "policy_size": config.policy_size,
        "value_classes": config.value_classes,
        "belief_classes": config.belief_classes,
        "dropout": config.dropout,
        "head_dimension": config.head_dimension,
        "architecture_id": config.architecture_id,
        "configuration": config.to_dict(),
        "configuration_digest": config.digest(),
        "family_constants": dict(FAMILY_CONSTANTS),
        "describe": config.describe(),
    }
    if extra:
        record.update(extra)
    return record


__all__ = [
    "FINAL_RUN_HOURS",
    "FINAL_RUN_SECONDS",
    "KNEE_EFFICIENCY_FLOOR",
    "MEMORY_GROWTH_TOLERANCE",
    "SELECTION_INPUT_KEYS",
    "SOAK_ENVIRONMENTS",
    "SOAK_INFERENCE_BATCH",
    "SOAK_LEGALITY",
    "SOAK_PRECISION",
    "SOAK_SAMPLE_SECONDS",
    "SOAK_SECONDS",
    "SOAK_SNAPSHOT_INTERVAL",
    "SOAK_VERSION",
    "SOAK_WARMUP_STEPS",
    "SOAK_WORKERS",
    "THROUGHPUT_DRIFT_TOLERANCE",
    "SoakError",
    "architecture_record",
    "growth_report",
    "half_over_half",
    "neighbor_tradeoffs",
    "probe_live_finiteness",
    "process_memory",
    "run_soak",
    "select_architectures",
    "selection_inputs",
    "soak_configuration",
    "storage_analysis",
    "summarize_soak",
    "weekly_projection",
]
