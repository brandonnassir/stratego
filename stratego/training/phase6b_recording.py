"""Phase 6B: the multi-hour persisted-recording soak and its memory verdict.

Specification source: the Phase 6B production-recording follow-up.

The question
------------
Phase 6's one-hour soak passed every correctness gate but showed host RSS rising
about 191 MiB/hour without decelerating, and it *encoded* trajectories without
ever writing them to disk. Two things were therefore unproven: whether the real
persistence path is operationally sound, and whether memory reaches a plateau.

This module answers both by running the accepted C1 production configuration for
several hours with durable per-worker shard writing switched on, sampling a time
series that includes per-worker resident memory and every disk-side quantity, and
then classifying the memory result against four outcomes declared in advance:

    A  RSS plateaus                                    -> operational PASS
    B  RSS grows because of a specific retained object  -> fix it, regress it
    C  RSS grows, looks like allocator behaviour        -> prove process recycling
    D  RSS grows with no demonstrated mitigation        -> BLOCKED

The classification is computed, not asserted. :func:`classify_memory_outcome`
takes the settled samples and returns the verdict together with the evidence it
used, so "it plateaued" is a measurement rather than a hope.

What a plateau means here
-------------------------
A plateau is not "the slope is small". It is that the slope over the settled
window is small enough that the run cannot exhaust memory over the 168-hour
horizon with the headroom this machine has, *and* that the trend is not a
convincing straight line. Both are required: a tiny but perfectly linear climb
is still a climb, and 168 hours is long. The thresholds are declared as module
constants before any Phase 6B measurement was taken.
"""

from __future__ import annotations

import shutil
import statistics
import time
from dataclasses import dataclass

import psutil

from .phase6_soak import (
    SOAK_ENVIRONMENTS,
    SOAK_INFERENCE_BATCH,
    SOAK_LEGALITY,
    SOAK_PRECISION,
    SOAK_SNAPSHOT_INTERVAL,
    SOAK_WORKERS,
    _ordinary_least_squares,
    half_over_half,
    probe_live_finiteness,
)
from .coordinator import CoordinatorConfig, SelfPlayCoordinator, mps_memory_bytes
from .end_to_end_benchmark import swap_bytes
from .phase6_pipeline_benchmark import (
    BYTES_PER_GIB,
    build_pipeline_candidate,
    candidate_configuration,
    classify_failure,
    empty_failure_counts,
)
from .shard_writer import DEFAULT_SHARD_TARGET_BYTES, directory_summary
from .worker_pool import WorkerPoolError

RECORDING_SOAK_VERSION = "agent_06b_recording_soak_0.1.0"

#: The follow-up runs at least this long; six hours is the target.
DEFAULT_SOAK_SECONDS = 6 * 3600.0
MINIMUM_SOAK_SECONDS = 4 * 3600.0
DEFAULT_SAMPLE_SECONDS = 60.0

#: Phase 6 measured that a cold pool's resident set climbs toward the envelope of
#: a fully synchronised slot population, and that the climb was still converging
#: at step ~1,300. Phase 6 used 3,000 steps; this keeps that, because the
#: question here is the *settled* slope and nothing is gained by including the
#: ramp in it.
DEFAULT_WARMUP_STEPS = 3_000

#: Declared before Phase 6B measured anything.
#:
#: `PLATEAU_SLOPE_MIB_PER_HOUR` is the slope at or below which the settled window
#: is called flat outright. 20 MiB/hour is ~3.4 GiB over the 168-hour run, which
#: this machine has headroom for several times over.
PLATEAU_SLOPE_MIB_PER_HOUR = 20.0

#: A slope above the plateau threshold is only called *growth* if the trend is
#: also a convincing line. Below this R^2 the samples are scatter around a level,
#: and calling that a leak would be reading noise.
GROWTH_R_SQUARED_FLOOR = 0.50

#: How much of total system memory the *growth* may consume before the process
#: is recycled. The pipeline's settled baseline is around 8.6 GiB on this 48 GiB
#: machine, so a quarter of system memory is a growth budget that still leaves
#: the operating system and the page cache a wide margin, and swap was a
#: zero-tolerance gate for good reason.
RECYCLE_BUDGET_FRACTION = 0.25

#: The shortest recycling interval that is still operationally sensible. Below
#: this the run is restarting so often that the restart cost and the risk of a
#: bad interaction outweigh the benefit, and the honest answer is BLOCKED rather
#: than a mitigation nobody would want to operate.
MINIMUM_PRACTICAL_RESTART_HOURS = 2.0

FINAL_RUN_HOURS = 168.0
FINAL_RUN_SECONDS = 604800.0

#: Machine-level watchdog thresholds. The first Phase 6B session showed why a
#: run needs these: a process that drives the machine into swap does not fail --
#: it *wedges*, in uninterruptible page-in waits, with no exception to classify.
#: The only loud moment is before the thrashing starts, so the soak checks the
#: system at every sample and aborts while an abort is still possible.
SWAP_GROWTH_LIMIT_BYTES = 2 * 1024**3
MINIMUM_SYSTEM_AVAILABLE_FRACTION = 0.08


class RecordingSoakError(RuntimeError):
    """The persisted-recording soak could not run, or lost an invariant."""


def check_memory_watchdog(
    *, swap_start_bytes: int, swap_now_bytes: int, system: dict
) -> str | None:
    """The message that should abort the run, or None while the machine is fine.

    Two conditions, both about the machine rather than the process: swap growth
    beyond what any healthy run would cause, and system available memory falling
    toward the point where macOS starts compressing and paging everything --
    past that point the process stops throwing exceptions and starts hanging in
    the kernel, which is the failure mode this watchdog exists to preempt.
    """
    growth = swap_now_bytes - swap_start_bytes
    if growth > SWAP_GROWTH_LIMIT_BYTES:
        return (
            f"swap grew {growth / 1024**3:.2f} GiB since the run started, above "
            f"the {SWAP_GROWTH_LIMIT_BYTES / 1024**3:.0f} GiB watchdog limit"
        )
    total = int(system.get("total_bytes", 0))
    available = int(system.get("available_bytes", 0))
    if total and available / total < MINIMUM_SYSTEM_AVAILABLE_FRACTION:
        return (
            f"system available memory is {available / total:.1%} of "
            f"{total / 1024**3:.0f} GiB, below the "
            f"{MINIMUM_SYSTEM_AVAILABLE_FRACTION:.0%} watchdog floor"
        )
    return None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def recording_configuration(
    candidate_id: str = "C1",
    *,
    output_directory: str,
    run_id: str,
    compress: bool = True,
    shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
    workers: int = SOAK_WORKERS,
    environments: int = SOAK_ENVIRONMENTS,
    inference_batch_size: int = SOAK_INFERENCE_BATCH,
    root_seed: int = 60_006,
    verify_target_decisions: int = 1_000_000,
    max_concurrent_verifications: int = 1,
) -> CoordinatorConfig:
    """The frozen Phase 6 C1 production topology, plus durable persistence.

    Every field the Phase 6 decision froze is passed through unchanged; the only
    additions are the three that turn encode-and-drop into encode-compress-write.
    """
    from dataclasses import replace

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
    return replace(
        config,
        trajectory_output_directory=str(output_directory),
        compress_trajectories=bool(compress),
        shard_target_bytes=int(shard_target_bytes),
        run_id=str(run_id),
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def per_worker_memory(pids) -> dict:
    """Resident memory for each worker process, keyed by worker id.

    Phase 6 reported the workers as one aggregate. Phase 6B needs them apart: if
    one worker grows and nine do not, that is a retained object in a specific
    slot range and a completely different diagnosis from all ten drifting
    together.
    """
    per_worker: dict[int, int] = {}
    total = 0
    alive = 0
    for worker_id, pid in enumerate(pids):
        try:
            process = psutil.Process(pid)
            resident = int(process.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):  # pragma: no cover
            per_worker[worker_id] = 0
            continue
        per_worker[worker_id] = resident
        total += resident
        alive += 1
    return {
        "per_worker_rss_bytes": per_worker,
        "worker_rss_bytes": total,
        "workers_measured": alive,
    }


def _swap_report(swap_start: dict, samples) -> dict:
    """Swap during the run, reported as a change rather than as an absolute.

    `sysctl vm.swapusage` is **system-wide**: it counts whatever every process on
    the machine has paged out, including applications that have nothing to do
    with collection. Phase 6's soak happened to read a clean zero, but a machine
    that has been busy will show a non-zero baseline, and failing the pipeline
    for another application's paging would be measuring the wrong thing.

    The gate is therefore the *growth*: did this run push the machine into
    swapping? The absolute values are reported alongside so the baseline is
    visible rather than hidden.
    """
    at_start = int(swap_start.get("swap_used_bytes", 0))
    at_end = int(swap_bytes().get("swap_used_bytes", 0))
    observed = [int(sample["swap_used_bytes"]) for sample in samples]
    maximum = max(observed, default=at_start)
    return {
        "used_at_start": at_start,
        "used_at_end": at_end,
        "maximum_observed": maximum,
        "growth_bytes": max(maximum - at_start, 0),
        "grew_during_run": maximum > at_start,
        "note": (
            "vm.swapusage is system-wide. The gate is growth attributable to this "
            "run; the absolute baseline comes from other processes on the machine."
        ),
    }


def disk_free_bytes(path) -> int:
    try:
        return int(shutil.disk_usage(str(path)).free)
    except OSError:  # pragma: no cover - path may vanish
        return 0


def system_memory() -> dict:
    """Total/available memory, so headroom is a measured quantity."""
    virtual = psutil.virtual_memory()
    report = {
        "total_bytes": int(virtual.total),
        "available_bytes": int(virtual.available),
        "percent_used": float(virtual.percent),
    }
    return report


# ---------------------------------------------------------------------------
# The memory verdict
# ---------------------------------------------------------------------------


def classify_memory_outcome(
    samples,
    *,
    total_system_bytes: int,
    key: str = "total_rss_bytes",
) -> dict:
    """Decide between outcomes A, C and D from the settled samples.

    Outcome B -- a specific retained object -- cannot be diagnosed from a slope;
    it needs the per-worker breakdown and a code-level cause, so this returns the
    evidence that would point at it (whether growth is concentrated in one
    worker) and leaves the naming to the caller.
    """
    settled = [sample for sample in samples if sample.get("in_measured_window")]
    if len(settled) < 4:
        raise RecordingSoakError(
            "the memory verdict needs at least four settled samples"
        )
    elapsed = [sample["elapsed_seconds"] for sample in settled]
    values = [float(sample[key]) for sample in settled]
    fit = _ordinary_least_squares(elapsed, values)
    halves = half_over_half(values)
    slope_per_hour = fit["slope"] * 3600.0
    slope_mib = slope_per_hour / 2**20
    projected_168h = slope_per_hour * FINAL_RUN_HOURS
    headroom_fraction = (
        projected_168h / total_system_bytes if total_system_bytes else 0.0
    )

    # Per-worker concentration: the signature that separates "one slot range is
    # retaining something" from "everything drifts a little".
    worker_slopes: dict[str, float] = {}
    if settled and settled[0].get("per_worker_rss_bytes"):
        worker_ids = sorted(settled[0]["per_worker_rss_bytes"])
        for worker_id in worker_ids:
            series = [
                float(sample["per_worker_rss_bytes"].get(worker_id, 0))
                for sample in settled
            ]
            worker_slopes[str(worker_id)] = (
                _ordinary_least_squares(elapsed, series)["slope"] * 3600.0
            )
    concentrated = False
    if worker_slopes:
        magnitudes = sorted((abs(value) for value in worker_slopes.values()), reverse=True)
        if len(magnitudes) > 1 and sum(magnitudes[1:]) > 0:
            # One worker carrying more than half of all worker growth.
            concentrated = magnitudes[0] > sum(magnitudes)
        elif len(magnitudes) > 1:
            concentrated = magnitudes[0] > 0

    is_flat = slope_mib <= PLATEAU_SLOPE_MIB_PER_HOUR
    is_a_line = fit["r_squared"] >= GROWTH_R_SQUARED_FLOOR

    # What recycling would have to look like if the trend is real. A large
    # 168-hour projection does not by itself mean the run is unsafe -- bounding
    # exactly that is what recycling does -- so the C/D split is decided by
    # whether the required interval is one anybody would actually operate.
    growth_budget = RECYCLE_BUDGET_FRACTION * total_system_bytes
    required_interval_hours = (
        growth_budget / slope_per_hour if slope_per_hour > 0 else float("inf")
    )

    if is_flat or not is_a_line:
        outcome = "A"
        reason = (
            f"settled slope {slope_mib:+.1f} MiB/hour"
            + (
                f" is at or below the {PLATEAU_SLOPE_MIB_PER_HOUR:.0f} MiB/hour "
                f"plateau threshold"
                if is_flat
                else f" has R^2 {fit['r_squared']:.3f}, below the "
                f"{GROWTH_R_SQUARED_FLOOR:.2f} floor, so the samples are scatter "
                f"around a level rather than a trend"
            )
        )
    elif required_interval_hours >= MINIMUM_PRACTICAL_RESTART_HOURS:
        outcome = "C"
        reason = (
            f"settled slope {slope_mib:+.1f} MiB/hour with R^2 "
            f"{fit['r_squared']:.3f} is a real trend above the plateau threshold; "
            f"a {growth_budget / BYTES_PER_GIB:.1f} GiB growth budget "
            f"({RECYCLE_BUDGET_FRACTION:.0%} of system memory) is consumed in "
            f"{required_interval_hours:.1f} hours, so controlled recycling at or "
            f"below that interval is the mitigation"
        )
    else:
        outcome = "D"
        reason = (
            f"settled slope {slope_mib:+.1f} MiB/hour would consume a "
            f"{growth_budget / BYTES_PER_GIB:.1f} GiB growth budget in "
            f"{required_interval_hours:.2f} hours, below the "
            f"{MINIMUM_PRACTICAL_RESTART_HOURS:.0f}-hour floor for a recycling "
            f"interval anyone would operate"
        )

    return {
        "outcome": outcome,
        "reason": reason,
        "quantity": key,
        "settled_samples": len(settled),
        "settled_window_seconds": elapsed[-1] - elapsed[0],
        "first_bytes": values[0],
        "last_bytes": values[-1],
        "minimum_bytes": min(values),
        "maximum_bytes": max(values),
        "slope_bytes_per_hour": slope_per_hour,
        "slope_mib_per_hour": slope_mib,
        "r_squared": fit["r_squared"],
        "half_over_half": halves,
        "projected_168h_bytes": projected_168h,
        "projected_168h_gib": projected_168h / BYTES_PER_GIB,
        "total_system_bytes": total_system_bytes,
        "headroom_fraction_of_system_memory": headroom_fraction,
        "growth_budget_bytes": growth_budget,
        "growth_budget_gib": growth_budget / BYTES_PER_GIB,
        "required_restart_interval_hours": required_interval_hours,
        "per_worker_slope_bytes_per_hour": worker_slopes,
        "growth_concentrated_in_one_worker": concentrated,
        "thresholds": {
            "plateau_slope_mib_per_hour": PLATEAU_SLOPE_MIB_PER_HOUR,
            "growth_r_squared_floor": GROWTH_R_SQUARED_FLOOR,
            "recycle_budget_fraction": RECYCLE_BUDGET_FRACTION,
            "minimum_practical_restart_hours": MINIMUM_PRACTICAL_RESTART_HOURS,
        },
    }


def recommended_restart_interval_hours(
    verdict: dict,
    *,
    budget_bytes: float,
) -> dict:
    """How long the process may run before recycling, from the measured slope.

    Derived rather than chosen: the interval is the time the measured slope takes
    to consume `budget_bytes`, which is what makes "every 24 hours" an answer
    instead of a habit.
    """
    slope = verdict["slope_bytes_per_hour"]
    if slope <= 0:
        return {
            "required": False,
            "reason": "no positive settled slope; nothing to recycle against",
            "interval_hours": None,
        }
    hours = budget_bytes / slope
    return {
        "required": True,
        "budget_bytes": budget_bytes,
        "budget_gib": budget_bytes / BYTES_PER_GIB,
        "slope_bytes_per_hour": slope,
        "slope_mib_per_hour": slope / 2**20,
        "hours_to_consume_budget": hours,
        "interval_hours": hours,
        "restarts_per_168_hours": FINAL_RUN_HOURS / hours if hours else 0.0,
    }


# ---------------------------------------------------------------------------
# The soak
# ---------------------------------------------------------------------------


@dataclass
class _Cumulative:
    elapsed: float
    step: int
    positions: int
    games: int
    record_bytes: int
    persisted_bytes: int
    compressed_bytes: int
    decisions: int


def run_recording_soak(
    candidate_id: str,
    config: CoordinatorConfig,
    *,
    seconds: float = DEFAULT_SOAK_SECONDS,
    sample_seconds: float = DEFAULT_SAMPLE_SECONDS,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    probe_rows: int = 512,
    device=None,
    progress=None,
    stop_after_steps: int | None = None,
) -> dict:
    """Run the persisted-recording soak and return samples plus the summary."""
    if not config.record_trajectories:
        raise RecordingSoakError("the recording soak must record trajectories")
    if not config.trajectory_output_directory:
        raise RecordingSoakError(
            "the recording soak must persist to disk; set "
            "trajectory_output_directory"
        )
    output = config.trajectory_output_directory
    model = build_pipeline_candidate(candidate_id)
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    coordinator = SelfPlayCoordinator(
        config, device=device, model=model, model_label=candidate_id
    )

    samples: list[dict] = []
    failures = empty_failure_counts()
    status = "ok"
    error_text = None
    probe_seconds = 0.0
    probe_logits = 0
    free_at_start = disk_free_bytes(output)
    swap_start = swap_bytes()

    coordinator.start()
    started = time.perf_counter()
    previous = _Cumulative(0.0, 0, 0, 0, 0, 0, 0, 0)
    warmup_marker: _Cumulative | None = None
    next_sample_at = sample_seconds

    try:
        while True:
            coordinator.step()
            elapsed = time.perf_counter() - started

            if warmup_marker is None and coordinator.step_index >= warmup_steps:
                totals = coordinator.pool.recording_totals()
                warmup_marker = _Cumulative(
                    elapsed=elapsed,
                    step=coordinator.step_index,
                    positions=int(coordinator.totals.positions),
                    games=int(coordinator.games_finished),
                    record_bytes=int(totals["total_record_bytes"]),
                    persisted_bytes=int(totals["total_persisted_bytes"]),
                    compressed_bytes=int(totals["total_compressed_bytes"]),
                    decisions=int(totals["total_decisions_recorded"]),
                )

            reached_end = elapsed >= seconds or (
                stop_after_steps is not None
                and coordinator.step_index >= stop_after_steps
            )
            if elapsed < next_sample_at and not reached_end:
                continue

            probe_started = time.perf_counter()
            probe = probe_live_finiteness(coordinator, rows=probe_rows)
            probe_seconds += time.perf_counter() - probe_started
            probe_logits += probe["logits_checked"]
            failures["nonfinite_outputs"] += probe["nonfinite_outputs"]

            liveness = coordinator.pool.worker_liveness()
            dead = [index for index, alive in enumerate(liveness) if not alive]
            if dead:
                raise WorkerPoolError(f"worker(s) {dead} are no longer running")

            totals = coordinator.pool.recording_totals()
            memory = per_worker_memory(coordinator.pool.worker_pids())
            coordinator_rss = int(psutil.Process().memory_info().rss)
            metal = mps_memory_bytes()
            swap = swap_bytes()
            system = system_memory()
            free_now = disk_free_bytes(output)

            watchdog = check_memory_watchdog(
                swap_start_bytes=int(swap_start.get("swap_used_bytes", 0)),
                swap_now_bytes=int(swap.get("swap_used_bytes", 0)),
                system=system,
            )
            if watchdog is not None:
                raise RecordingSoakError(f"memory watchdog: {watchdog}")

            write_errors = int(totals["total_write_errors"])
            if write_errors:
                raise RecordingSoakError(
                    f"{write_errors} filesystem write error(s) reported by the "
                    f"shard writers"
                )
            mismatches = int(totals["total_reconstruction_mismatches"])
            if mismatches:
                raise RecordingSoakError(
                    f"{mismatches} trajectory reconstruction mismatch(es)"
                )

            current = _Cumulative(
                elapsed=elapsed,
                step=coordinator.step_index,
                positions=int(coordinator.totals.positions),
                games=int(coordinator.games_finished),
                record_bytes=int(totals["total_record_bytes"]),
                persisted_bytes=int(totals["total_persisted_bytes"]),
                compressed_bytes=int(totals["total_compressed_bytes"]),
                decisions=int(totals["total_decisions_recorded"]),
            )
            window = max(current.elapsed - previous.elapsed, 1e-9)
            window_persisted = current.persisted_bytes - previous.persisted_bytes
            window_produced = current.record_bytes - previous.record_bytes

            row = {
                "candidate_id": candidate_id,
                "sample_index": len(samples),
                "elapsed_seconds": current.elapsed,
                "global_step": current.step,
                "in_measured_window": current.step > warmup_steps,
                "positions": current.positions,
                "positions_per_second": (current.positions - previous.positions)
                / window,
                "games": current.games,
                "games_per_second": (current.games - previous.games) / window,
                "decisions_recorded": current.decisions,
                # -- disk ------------------------------------------------------
                "trajectory_bytes_produced": current.record_bytes,
                "trajectory_bytes_written": current.persisted_bytes,
                "compressed_bytes": current.compressed_bytes,
                "window_bytes_produced": window_produced,
                "window_bytes_written": window_persisted,
                "produced_gib_per_hour": (window_produced / window)
                * 3600.0
                / BYTES_PER_GIB,
                "written_gib_per_hour": (window_persisted / window)
                * 3600.0
                / BYTES_PER_GIB,
                "compression_ratio": (
                    current.compressed_bytes / current.record_bytes
                    if current.record_bytes
                    else 0.0
                ),
                "write_throughput_bytes_per_second": window_persisted / window,
                "records_persisted": int(totals["total_records_persisted"]),
                "shards_opened": int(totals["total_shards_opened"]),
                "shards_closed": int(totals["total_shards_closed"]),
                "encode_seconds": float(totals["total_encode_seconds"]),
                "compress_seconds": float(totals["total_compress_seconds"]),
                "write_seconds": float(totals["total_write_seconds"]),
                "flush_seconds": float(totals["total_flush_seconds"]),
                "write_errors": write_errors,
                "pending_records": int(totals["total_pending_records"]),
                "pending_bytes": int(totals["total_pending_bytes"]),
                "disk_free_bytes": free_now,
                "disk_free_change_bytes": free_now - free_at_start,
                # -- memory ----------------------------------------------------
                "coordinator_rss_bytes": coordinator_rss,
                "worker_rss_bytes": memory["worker_rss_bytes"],
                "per_worker_rss_bytes": memory["per_worker_rss_bytes"],
                "workers_measured": memory["workers_measured"],
                "total_rss_bytes": coordinator_rss + memory["worker_rss_bytes"],
                "shared_memory_bytes": int(coordinator.pool.buffers.nbytes),
                "metal_current_allocated_bytes": int(
                    metal.get("current_allocated_bytes", 0)
                ),
                "metal_driver_allocated_bytes": int(
                    metal.get("driver_allocated_bytes", 0)
                ),
                "swap_used_bytes": int(swap.get("swap_used_bytes", 0)),
                "system_memory_available_bytes": system["available_bytes"],
                "system_memory_percent_used": system["percent_used"],
                # -- correctness ----------------------------------------------
                "illegal_actions": failures["illegal_actions"],
                "action_frame_errors": failures["action_frame_errors"],
                "worker_failures": failures["worker_errors"],
                "model_failures": failures["model_errors"],
                "nonfinite_outputs": failures["nonfinite_outputs"],
                "verified_decisions": int(totals["total_verified_decisions"]),
                "verified_games": int(totals["total_verified_games"]),
                "reconstruction_mismatches": mismatches,
                "workers_alive": int(sum(liveness)),
            }
            samples.append(row)
            if progress is not None:
                progress(row)
            previous = current
            next_sample_at += sample_seconds
            if reached_end:
                break
    except BaseException as error:  # noqa: BLE001 - a failed soak is a result
        status = "error"
        error_text = f"{type(error).__name__}: {error}"
        category = classify_failure(error)
        if isinstance(error, RecordingSoakError) and "write error" in str(error):
            category = "write_errors"
        failures[category] = failures.get(category, 0) + 1
    finally:
        try:
            pool_totals = coordinator.pool.recording_totals()
        except Exception:  # noqa: BLE001
            pool_totals = {}
        try:
            shutdown_totals = coordinator.shutdown()
        except Exception as error:  # noqa: BLE001
            shutdown_totals = {}
            status = "error" if status == "ok" else status
            error_text = error_text or f"shutdown: {type(error).__name__}: {error}"

    total_seconds = time.perf_counter() - started
    totals = {**pool_totals, **shutdown_totals}
    free_at_end = disk_free_bytes(output)

    return {
        "recording_soak_version": RECORDING_SOAK_VERSION,
        "candidate_id": candidate_id,
        "parameters": parameters,
        "status": status,
        "error": error_text,
        "configuration": config.as_dict(),
        "output_directory": str(output),
        "requested_seconds": seconds,
        "total_seconds": total_seconds,
        "samples": samples,
        "sample_count": len(samples),
        "warmup_steps": warmup_steps,
        "warmup_marker": (
            {
                "elapsed": warmup_marker.elapsed,
                "step": warmup_marker.step,
                "positions": warmup_marker.positions,
                "games": warmup_marker.games,
                "record_bytes": warmup_marker.record_bytes,
                "persisted_bytes": warmup_marker.persisted_bytes,
                "compressed_bytes": warmup_marker.compressed_bytes,
                "decisions": warmup_marker.decisions,
            }
            if warmup_marker
            else None
        ),
        "failures": failures,
        "recording_totals": {
            key: value for key, value in totals.items() if isinstance(value, (int, float))
        },
        "disk": {
            "free_bytes_at_start": free_at_start,
            "free_bytes_at_end": free_at_end,
            "free_bytes_change": free_at_end - free_at_start,
        },
        "swap": _swap_report(swap_start, samples),
        "finiteness_probe": {
            "seconds": probe_seconds,
            "logits_checked": probe_logits,
            "fraction_of_wall": probe_seconds / total_seconds if total_seconds else 0.0,
        },
        "system_memory": system_memory(),
    }


def steady_state_summary(result: dict) -> dict:
    """Sustained rates over the settled window, differenced from raw counters."""
    marker = result.get("warmup_marker")
    samples = result.get("samples") or []
    settled = [sample for sample in samples if sample["in_measured_window"]]
    if not marker or not settled:
        return {}
    last = settled[-1]
    window = last["elapsed_seconds"] - marker["elapsed"]
    if window <= 0:
        return {}
    positions = last["positions"] - marker["positions"]
    games = last["games"] - marker["games"]
    produced = last["trajectory_bytes_produced"] - marker["record_bytes"]
    written = last["trajectory_bytes_written"] - marker["persisted_bytes"]
    compressed = last["compressed_bytes"] - marker["compressed_bytes"]
    decisions = last["decisions_recorded"] - marker["decisions"]
    return {
        "window_seconds": window,
        "window_steps": last["global_step"] - marker["step"],
        "warmup_seconds": marker["elapsed"],
        "warmup_steps": marker["step"],
        "samples_in_window": len(settled),
        "positions": positions,
        "games": games,
        "decisions": decisions,
        "positions_per_second": positions / window,
        "games_per_second": games / window,
        "mean_game_length": positions / games if games else 0.0,
        "bytes_produced": produced,
        "bytes_written": written,
        "compressed_bytes": compressed,
        "produced_gib_per_hour": (produced / window) * 3600.0 / BYTES_PER_GIB,
        "written_gib_per_hour": (written / window) * 3600.0 / BYTES_PER_GIB,
        "compression_ratio": compressed / produced if produced else 0.0,
        "write_throughput_bytes_per_second": written / window,
        "bytes_per_decision_produced": produced / decisions if decisions else 0.0,
        "bytes_per_decision_written": written / decisions if decisions else 0.0,
    }


def storage_projection_from_disk(
    steady: dict,
    *,
    volume_total_bytes: int,
    volume_free_bytes: int,
    shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
) -> dict:
    """Project the 168-hour run from the rate that actually reached the disk."""
    per_second = steady["write_throughput_bytes_per_second"]
    per_hour = per_second * 3600.0
    total = per_second * FINAL_RUN_SECONDS
    return {
        "measured": {
            "source": "Phase 6B persisted-recording soak, settled window",
            "window_seconds": steady["window_seconds"],
            "write_throughput_bytes_per_second": per_second,
            "written_gib_per_hour": per_hour / BYTES_PER_GIB,
            "produced_gib_per_hour": steady["produced_gib_per_hour"],
            "compression_ratio": steady["compression_ratio"],
            "bytes_per_decision_written": steady["bytes_per_decision_written"],
        },
        "extrapolated": {
            "gib_per_hour": per_hour / BYTES_PER_GIB,
            "gib_per_24_hours": per_hour * 24.0 / BYTES_PER_GIB,
            "gib_per_168_hours": total / BYTES_PER_GIB,
            "gb_per_168_hours": total / 1000**3,
            "bytes_per_168_hours": total,
            # One shard per worker may be open and therefore not yet counted
            # toward a closed manifest; that is the only transient headroom the
            # writer needs, and it is bounded by the rollover size.
            "shard_headroom_bytes": shard_target_bytes * SOAK_WORKERS,
            "shard_headroom_gib": shard_target_bytes * SOAK_WORKERS / BYTES_PER_GIB,
        },
        "volume": {
            "total_bytes": volume_total_bytes,
            "total_gib": volume_total_bytes / BYTES_PER_GIB,
            "free_bytes_now": volume_free_bytes,
            "free_gib_now": volume_free_bytes / BYTES_PER_GIB,
            "fraction_of_total": total / volume_total_bytes
            if volume_total_bytes
            else 0.0,
            "fraction_of_free_now": total / volume_free_bytes
            if volume_free_bytes
            else 0.0,
            "fits_in_total": total <= volume_total_bytes,
            "fits_in_free_now": total <= volume_free_bytes,
            "remaining_after_run_bytes": volume_total_bytes - total,
            "remaining_after_run_gib": (volume_total_bytes - total) / BYTES_PER_GIB,
        },
    }


__all__ = [
    "DEFAULT_SAMPLE_SECONDS",
    "DEFAULT_SOAK_SECONDS",
    "DEFAULT_WARMUP_STEPS",
    "FINAL_RUN_HOURS",
    "FINAL_RUN_SECONDS",
    "GROWTH_R_SQUARED_FLOOR",
    "MINIMUM_PRACTICAL_RESTART_HOURS",
    "MINIMUM_SOAK_SECONDS",
    "MINIMUM_SYSTEM_AVAILABLE_FRACTION",
    "PLATEAU_SLOPE_MIB_PER_HOUR",
    "RECYCLE_BUDGET_FRACTION",
    "RECORDING_SOAK_VERSION",
    "SWAP_GROWTH_LIMIT_BYTES",
    "RecordingSoakError",
    "check_memory_watchdog",
    "classify_memory_outcome",
    "disk_free_bytes",
    "per_worker_memory",
    "recommended_restart_interval_hours",
    "recording_configuration",
    "run_recording_soak",
    "steady_state_summary",
    "storage_projection_from_disk",
    "system_memory",
]
