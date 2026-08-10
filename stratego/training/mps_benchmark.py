"""Apple Metal (MPS) inference benchmark for the Phase 3 representative probe.

Specification sources:

- `05_project_plan.md` Phase 3 benchmark matrix (inference batches 64, 128,
  256, 512, 1,024, 1,536, 2,048; dense legality first, then a sparse
  comparison; supported reduced precision compared against float32)
- `03_game_engine_spec.md` section 18 (only the coordinator owns the Metal
  device; simulation workers stay on the central processing unit)

What is measured
----------------
The full coordinator-side model step, not just a matrix multiply:

```text
host observations -> device transfer -> encoder forward
    -> legality application -> action sampling -> chosen action ids back to host
```

Every timed region is bracketed by an explicit device synchronisation, because
Metal dispatch is asynchronous and an unsynchronised timer measures queue
submission rather than work.

Two passes are run per configuration:

- a **phased** pass that synchronises between stages, which is what makes the
  model-only and legality+sampling figures separable but adds synchronisation
  overhead;
- an **end-to-end** pass with a single synchronisation, which is the honest
  sustainable rate.

The network under test is the throw-away probe in
:mod:`stratego.training.representative_model`, not a frozen design.

Position source
---------------
Batches are drawn from *real* frozen-engine positions produced by
:class:`~stratego.training.batch_simulation.BatchSimulator`, sampled across
game phases, so legality densities match real play instead of a guess.
"""

from __future__ import annotations

import gc
import platform
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..engine.constants import ACTION_SPACE_SIZE, IMPLEMENTATION_VERSION, OBSERVATION_VERSION
from .batch_simulation import BatchSimulator
from .representative_model import (
    REPRESENTATIVE_MODEL_VERSION,
    CompactLegality,
    RepresentativeConfig,
    build_compact_legality,
    build_representative_model,
    compact_gathered_logits,
    compact_legal_probabilities,
    dense_legal_probabilities,
    observation_to_tokens,
    sample_compact,
    sample_dense,
)

BENCHMARK_VERSION = "agent_04_mps_benchmark_0.1.0"

#: The Phase 3 planning matrix.
DEFAULT_BATCH_SIZES = (64, 128, 256, 512, 1024, 1536, 2048)

#: float32 is the baseline; the rest are probed and kept only if they work.
DEFAULT_PRECISIONS = ("float32", "float16", "bfloat16")

DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

LEGALITY_MODES = ("dense", "compact")

#: Padding capacity is rounded up to a multiple of this so the compact tensor
#: shape is stable across pools.
CAPACITY_GRANULARITY = 8

#: A compact legality path only earns its extra machinery if it beats dense by
#: more than this on the end-to-end model step. Below it, dense wins on
#: simplicity: the engine already produces the dense mask, and the compact form
#: adds a fixed padding capacity and therefore a new way to fail.
COMPACT_ADOPTION_MARGIN = 0.05

#: Reduced precision is only recommended if it is both faster than float32 by
#: more than this margin *and* measured stable on the complete path.
PRECISION_ADOPTION_MARGIN = 0.05

#: Two reduced precisions within this much of each other are treated as equally
#: fast, and the one that stays closer to the float32 distribution wins.
PRECISION_SPEED_TIE = 0.02

#: The padding capacity a production compact path would have to use. A capacity
#: fitted to one sample of positions is not a bound on the legal-action count:
#: Agent 3 observed up to 62 legal actions across 1,000,162 decisions, so 128
#: leaves roughly twice that headroom while `build_compact_legality` still fails
#: loudly above it. The compact-versus-dense verdict is taken at this capacity,
#: because this is what would actually run.
PRODUCTION_COMPACT_CAPACITY = 128


# ---------------------------------------------------------------------------
# Platform and device detection
# ---------------------------------------------------------------------------


def _sysctl(name: str) -> str | None:
    try:
        value = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if value.returncode != 0:
        return None
    return value.stdout.strip() or None


def peak_memory_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes, Linux reports kilobytes.
    return usage if sys.platform == "darwin" else usage * 1024


def mps_memory_bytes() -> dict:
    """Best-effort Metal allocator counters; `None` where unsupported."""
    report: dict[str, int | None] = {
        "current_allocated_bytes": None,
        "driver_allocated_bytes": None,
        "recommended_max_bytes": None,
    }
    if not torch.backends.mps.is_available():
        return report
    for key, function in (
        ("current_allocated_bytes", getattr(torch.mps, "current_allocated_memory", None)),
        ("driver_allocated_bytes", getattr(torch.mps, "driver_allocated_memory", None)),
        ("recommended_max_bytes", getattr(torch.mps, "recommended_max_memory", None)),
    ):
        if function is None:
            continue
        try:
            report[key] = int(function())
        except (RuntimeError, TypeError, ValueError):
            report[key] = None
    return report


def detect_device_report() -> dict:
    """Everything the report needs to know about where the numbers came from."""
    available = bool(torch.backends.mps.is_available())
    report = {
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "platform_full": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": available,
        "mps_device_info": None,
    }
    if available:
        report["mps_device_info"] = {
            "chip": _sysctl("machdep.cpu.brand_string"),
            "physical_cores": _sysctl("hw.physicalcpu"),
            "logical_cores": _sysctl("hw.logicalcpu"),
            "unified_memory_bytes": int(_sysctl("hw.memsize") or 0) or None,
            "backend": "torch.backends.mps",
            **mps_memory_bytes(),
        }
    return report


def synchronize(device: torch.device) -> None:
    """Block until `device` has finished every queued kernel."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":  # pragma: no cover - not a target platform
        torch.cuda.synchronize()


def _empty_device_cache(device: torch.device) -> None:
    if device.type == "mps":
        empty = getattr(torch.mps, "empty_cache", None)
        if empty is not None:
            empty()


def make_generator(device: torch.device, seed: int) -> torch.Generator | None:
    """Seeded generator on `device`, or `None` when the backend has none."""
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        return None
    generator.manual_seed(seed)
    return generator


# ---------------------------------------------------------------------------
# Real-position pool
# ---------------------------------------------------------------------------


@dataclass
class PositionPool:
    """Real frozen-engine positions held on the host, ready to batch.

    `tokens` is already in the `(P, 100, 127)` token layout because the
    coordinator would choose that layout in shared memory; the reshape is a
    pure view change and is reported separately rather than hidden inside a
    timed region.
    """

    tokens: np.ndarray  # (P, 100, 127) float32
    dense_mask: np.ndarray  # (P, 10000) uint8
    legal_lists: list[list[int]]
    capacity: int
    plies: list[int]
    build_seconds: float
    root_seed: int

    @property
    def size(self) -> int:
        return int(self.tokens.shape[0])

    def stats(self) -> dict:
        counts = [len(row) for row in self.legal_lists]
        return {
            "positions": self.size,
            "root_seed": self.root_seed,
            "build_seconds": round(self.build_seconds, 3),
            "compact_capacity": self.capacity,
            "legal_actions_min": min(counts),
            "legal_actions_max": max(counts),
            "legal_actions_mean": round(statistics.fmean(counts), 3),
            "legal_actions_median": statistics.median(counts),
            "legal_density_mean": round(statistics.fmean(counts) / ACTION_SPACE_SIZE, 8),
            "ply_min": min(self.plies),
            "ply_max": max(self.plies),
            "ply_mean": round(statistics.fmean(self.plies), 2),
            "token_bytes": int(self.tokens.nbytes),
            "dense_mask_bytes": int(self.dense_mask.nbytes),
        }


def build_position_pool(
    *,
    target_positions: int = 2048,
    num_environments: int = 128,
    collection_stride: int = 32,
    root_seed: int = 20260809,
) -> PositionPool:
    """Collect `target_positions` real acting-player positions across game phases.

    Slots take uniformly random legal actions and are sampled every
    `collection_stride` plies, so the pool spans openings through late
    middlegame rather than only ply 0.
    """
    started = time.perf_counter()
    simulator = BatchSimulator(num_environments=num_environments, root_seed=root_seed)
    rng = np.random.default_rng(root_seed)

    token_blocks: list[np.ndarray] = []
    mask_blocks: list[np.ndarray] = []
    legal_lists: list[list[int]] = []
    plies: list[int] = []
    collected = 0
    step_index = 0

    while collected < target_positions:
        if step_index % collection_stride == 0:
            active = simulator.active_slots()
            if active:
                observations = simulator.observations(active)
                masks = simulator.legal_action_masks(active)
                lists = simulator.legal_action_lists(active)
                keep = min(len(active), target_positions - collected)
                token_blocks.append(observation_to_tokens(observations[:keep]))
                mask_blocks.append(masks[:keep])
                legal_lists.extend([list(row) for row in lists[:keep]])
                plies.extend(
                    int(simulator.game_state(slot).ply) for slot in active[:keep]
                )
                collected += keep
                if collected >= target_positions:
                    break

        active = simulator.active_slots()
        if not active:
            simulator.reset_finished()
            step_index += 1
            continue
        actions = {}
        for slot in active:
            legal = simulator.legal_actions(slot)
            actions[slot] = int(legal[rng.integers(len(legal))])
        simulator.step(actions)
        simulator.reset_finished()
        step_index += 1

    tokens = np.concatenate(token_blocks, axis=0)
    dense_mask = np.concatenate(mask_blocks, axis=0)
    longest = max(len(row) for row in legal_lists)
    capacity = (
        (longest + CAPACITY_GRANULARITY - 1) // CAPACITY_GRANULARITY
    ) * CAPACITY_GRANULARITY
    return PositionPool(
        tokens=np.ascontiguousarray(tokens, dtype=np.float32),
        dense_mask=np.ascontiguousarray(dense_mask, dtype=np.uint8),
        legal_lists=legal_lists,
        capacity=max(capacity, CAPACITY_GRANULARITY),
        plies=plies,
        build_seconds=time.perf_counter() - started,
        root_seed=root_seed,
    )


@dataclass
class HostBatch:
    """One fixed host-side batch, reused across every timed iteration."""

    tokens: torch.Tensor  # (B, 100, 127) float32, CPU
    dense_mask: torch.Tensor  # (B, 10000) bool, CPU
    compact: CompactLegality  # CPU
    legal_lists: list[list[int]]

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])


def make_host_batch(
    pool: PositionPool, batch_size: int, *, offset: int = 0, capacity: int | None = None
) -> HostBatch:
    """Contiguous wrap-around slice of `pool`, mirroring a shared-memory read.

    `capacity` overrides the pool's observed padding capacity, which is what
    the capacity-sensitivity probe uses: production cannot size the padding
    from one sample of positions and must pick a safe upper bound.
    """
    indices = (offset + np.arange(batch_size)) % pool.size
    tokens = torch.from_numpy(np.ascontiguousarray(pool.tokens[indices]))
    mask = torch.from_numpy(np.ascontiguousarray(pool.dense_mask[indices])).to(torch.bool)
    lists = [pool.legal_lists[index] for index in indices]
    compact = build_compact_legality(lists, capacity=capacity or pool.capacity)
    return HostBatch(tokens=tokens, dense_mask=mask, compact=compact, legal_lists=lists)


# ---------------------------------------------------------------------------
# One timed configuration
# ---------------------------------------------------------------------------


@dataclass
class StageTimings:
    transfer: list[float] = field(default_factory=list)
    model: list[float] = field(default_factory=list)
    legality: list[float] = field(default_factory=list)
    readback: list[float] = field(default_factory=list)
    end_to_end: list[float] = field(default_factory=list)


def _summarise(samples: list[float]) -> dict:
    if not samples:
        return {"count": 0}
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "mean_ms": round(1000 * statistics.fmean(samples), 4),
        "median_ms": round(1000 * statistics.median(samples), 4),
        "min_ms": round(1000 * ordered[0], 4),
        "max_ms": round(1000 * ordered[-1], 4),
        "p95_ms": round(1000 * ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
        "stdev_ms": round(1000 * (statistics.stdev(samples) if len(samples) > 1 else 0.0), 4),
    }


def _model_step(
    model,
    tokens_device: torch.Tensor,
    dense_mask_device: torch.Tensor | None,
    compact_device: CompactLegality | None,
    legality_mode: str,
    generator: torch.Generator | None,
) -> torch.Tensor:
    outputs = model(tokens_device)
    if legality_mode == "dense":
        return sample_dense(outputs.policy_logits, dense_mask_device, generator=generator)
    return sample_compact(outputs.policy_logits, compact_device, generator=generator)


def benchmark_configuration(
    *,
    model,
    device: torch.device,
    dtype: torch.dtype,
    host: HostBatch,
    legality_mode: str,
    warmup_iterations: int = 5,
    target_pass_seconds: float = 1.5,
    min_iterations: int = 8,
    max_iterations: int = 80,
    trials: int = 3,
    seed: int = 4,
) -> dict:
    """Time one (batch size, precision, legality mode) point.

    Returns a result dictionary; on out-of-memory or an unsupported operation
    the dictionary carries `status: "FAILED"` and the error text instead of
    timings, and the sweep continues.
    """
    batch_size = host.batch_size
    result: dict = {
        "batch_size": batch_size,
        "precision": {v: k for k, v in DTYPE_BY_NAME.items()}[dtype],
        "legality": legality_mode,
        "status": "OK",
        "device": device.type,
    }

    generator = make_generator(device, seed)
    result["seeded_generator"] = generator is not None

    try:
        tokens_device = host.tokens.to(device=device, dtype=dtype)
        dense_mask_device = host.dense_mask.to(device) if legality_mode == "dense" else None
        compact_device = host.compact.to(device) if legality_mode == "compact" else None

        # -- warm-up: shader compilation, allocator growth, lazy init --------
        warmup_started = time.perf_counter()
        for _ in range(warmup_iterations):
            with torch.inference_mode():
                actions = _model_step(
                    model,
                    tokens_device,
                    dense_mask_device,
                    compact_device,
                    legality_mode,
                    generator,
                )
            _ = actions.to("cpu")
        synchronize(device)
        result["warmup_iterations"] = warmup_iterations
        result["warmup_seconds"] = round(time.perf_counter() - warmup_started, 4)

        # -- pilot: how many iterations reach `target_pass_seconds`? ---------
        pilot_started = time.perf_counter()
        for _ in range(3):
            with torch.inference_mode():
                actions = _model_step(
                    model,
                    tokens_device,
                    dense_mask_device,
                    compact_device,
                    legality_mode,
                    generator,
                )
            _ = actions.to("cpu")
        synchronize(device)
        per_iteration = max((time.perf_counter() - pilot_started) / 3, 1e-6)
        iterations = int(min(max_iterations, max(min_iterations, target_pass_seconds / per_iteration)))
        result["iterations_per_pass"] = iterations
        result["trials"] = trials

        timings = StageTimings()
        sampled_actions: torch.Tensor | None = None

        for _trial in range(trials):
            # -- phased pass: a synchronisation between every stage ----------
            for _ in range(iterations):
                start = time.perf_counter()
                staged_tokens = host.tokens.to(device=device, dtype=dtype)
                if legality_mode == "dense":
                    staged_legality = host.dense_mask.to(device)
                else:
                    staged_legality = host.compact.to(device)
                synchronize(device)
                after_transfer = time.perf_counter()

                with torch.inference_mode():
                    outputs = model(staged_tokens)
                synchronize(device)
                after_model = time.perf_counter()

                with torch.inference_mode():
                    if legality_mode == "dense":
                        actions = sample_dense(
                            outputs.policy_logits, staged_legality, generator=generator
                        )
                    else:
                        actions = sample_compact(
                            outputs.policy_logits, staged_legality, generator=generator
                        )
                synchronize(device)
                after_legality = time.perf_counter()

                host_actions = actions.to("cpu")
                synchronize(device)
                after_readback = time.perf_counter()

                timings.transfer.append(after_transfer - start)
                timings.model.append(after_model - after_transfer)
                timings.legality.append(after_legality - after_model)
                timings.readback.append(after_readback - after_legality)
                sampled_actions = host_actions

            # -- end-to-end pass: one synchronisation per step ---------------
            for _ in range(iterations):
                start = time.perf_counter()
                staged_tokens = host.tokens.to(device=device, dtype=dtype)
                if legality_mode == "dense":
                    staged_legality = host.dense_mask.to(device)
                else:
                    staged_legality = host.compact.to(device)
                with torch.inference_mode():
                    outputs = model(staged_tokens)
                    if legality_mode == "dense":
                        actions = sample_dense(
                            outputs.policy_logits, staged_legality, generator=generator
                        )
                    else:
                        actions = sample_compact(
                            outputs.policy_logits, staged_legality, generator=generator
                        )
                host_actions = actions.to("cpu")
                synchronize(device)
                timings.end_to_end.append(time.perf_counter() - start)
                sampled_actions = host_actions

        # -- legality of every sampled action, as a correctness guard --------
        illegal = 0
        if sampled_actions is not None:
            for row, action in enumerate(sampled_actions.tolist()):
                if action not in host.legal_lists[row]:
                    illegal += 1
        result["sampled_actions_checked"] = int(batch_size)
        result["illegal_samples"] = illegal

        # -- finiteness of the last forward ----------------------------------
        with torch.inference_mode():
            outputs = model(tokens_device)
            result["outputs_finite"] = bool(outputs.all_finite())
            result["policy_shape"] = list(outputs.policy_logits.shape)
            result["value_shape"] = list(outputs.value_logits.shape)
            result["belief_shape"] = list(outputs.belief_logits.shape)
        synchronize(device)

        end_to_end_mean = statistics.fmean(timings.end_to_end)
        model_mean = statistics.fmean(timings.model)
        legality_mean = statistics.fmean(timings.legality)
        transfer_mean = statistics.fmean(timings.transfer)
        readback_mean = statistics.fmean(timings.readback)

        result["latency"] = {
            "end_to_end": _summarise(timings.end_to_end),
            "transfer": _summarise(timings.transfer),
            "model_only": _summarise(timings.model),
            "legality_and_sampling": _summarise(timings.legality),
            "readback": _summarise(timings.readback),
        }
        result["positions_per_second"] = round(batch_size / end_to_end_mean, 2)
        result["model_only_positions_per_second"] = round(batch_size / model_mean, 2)
        result["legality_sampling_positions_per_second"] = round(
            batch_size / legality_mean, 2
        )
        result["transfer_positions_per_second"] = round(batch_size / transfer_mean, 2)
        result["readback_positions_per_second"] = round(batch_size / readback_mean, 2)
        result["phased_sum_positions_per_second"] = round(
            batch_size / (transfer_mean + model_mean + legality_mean + readback_mean), 2
        )
        result["legality_share_of_end_to_end"] = round(legality_mean / end_to_end_mean, 4)
        result["model_share_of_end_to_end"] = round(model_mean / end_to_end_mean, 4)

        # -- memory ----------------------------------------------------------
        legality_bytes = (
            host.dense_mask.numel() * host.dense_mask.element_size()
            if legality_mode == "dense"
            else host.compact.nbytes()
        )
        result["legality_host_bytes"] = int(legality_bytes)
        result["legality_bytes_per_position"] = round(legality_bytes / batch_size, 2)
        result["peak_process_memory_bytes"] = peak_memory_bytes()
        result["mps_memory_bytes"] = mps_memory_bytes()

    except RuntimeError as error:
        message = str(error)
        result["status"] = "FAILED"
        result["error"] = message
        result["out_of_memory"] = (
            "out of memory" in message.lower() or "insufficient memory" in message.lower()
        )
        _empty_device_cache(device)
        gc.collect()
        return result

    _empty_device_cache(device)
    return result


# ---------------------------------------------------------------------------
# Correctness / stability probes
# ---------------------------------------------------------------------------


def legality_equivalence_check(
    *, model, device: torch.device, dtype: torch.dtype, host: HostBatch
) -> dict:
    """Dense and compact must agree on normalised probabilities over the legal set."""
    tokens = host.tokens.to(device=device, dtype=dtype)
    dense_mask = host.dense_mask.to(device)
    compact = host.compact.to(device)
    with torch.inference_mode():
        outputs = model(tokens)
        dense_probabilities = dense_legal_probabilities(outputs.policy_logits, dense_mask)
        compact_probabilities = compact_legal_probabilities(outputs.policy_logits, compact)
        gathered_dense = dense_probabilities.gather(1, compact.action_ids)
        gathered_dense = gathered_dense.masked_fill(~compact.valid, 0.0)
        difference = (gathered_dense - compact_probabilities).abs()
        max_difference = float(difference.max())
        dense_mass = float(dense_probabilities.sum(dim=1).min())
        illegal_mass = float(
            dense_probabilities.masked_fill(dense_mask, 0.0).abs().max()
        )
        compact_mass = float(compact_probabilities.sum(dim=1).min())
    synchronize(device)
    return {
        "batch_size": host.batch_size,
        "max_absolute_probability_difference": max_difference,
        "max_illegal_probability_mass": illegal_mass,
        "min_dense_probability_sum": dense_mass,
        "min_compact_probability_sum": compact_mass,
        "equivalent": max_difference < 1e-5 and illegal_mass == 0.0,
    }


def determinism_check(
    *, model, device: torch.device, dtype: torch.dtype, host: HostBatch, repeats: int = 5
) -> dict:
    """Same input, same weights, same device/precision -> stable output."""
    tokens = host.tokens.to(device=device, dtype=dtype)
    with torch.inference_mode():
        reference = model(tokens).policy_logits.detach().clone()
        worst = 0.0
        for _ in range(repeats - 1):
            again = model(tokens).policy_logits
            worst = max(worst, float((again.float() - reference.float()).abs().max()))
    synchronize(device)
    return {
        "repeats": repeats,
        "max_absolute_logit_difference": worst,
        "bitwise_identical": worst == 0.0,
    }


def precision_agreement_check(
    *,
    reference_model,
    candidate_model,
    device: torch.device,
    candidate_dtype: torch.dtype,
    host: HostBatch,
) -> dict:
    """Compare a reduced-precision path against the float32 baseline.

    Agreement is measured on the *complete* path: legal-set probabilities and
    the greedy action, not on one operation.
    """
    tokens32 = host.tokens.to(device=device, dtype=torch.float32)
    tokens_candidate = host.tokens.to(device=device, dtype=candidate_dtype)
    compact = host.compact.to(device)
    with torch.inference_mode():
        reference = reference_model(tokens32)
        candidate = candidate_model(tokens_candidate)
        reference_probabilities = compact_legal_probabilities(
            reference.policy_logits, compact
        )
        candidate_probabilities = compact_legal_probabilities(
            candidate.policy_logits, compact
        )
        finite = bool(candidate.all_finite())
        max_probability_difference = float(
            (reference_probabilities - candidate_probabilities).abs().max()
        )
        total_variation = float(
            0.5 * (reference_probabilities - candidate_probabilities).abs().sum(dim=1).max()
        )
        reference_top = compact_gathered_logits(reference.policy_logits, compact).argmax(1)
        candidate_top = compact_gathered_logits(candidate.policy_logits, compact).argmax(1)
        top1_agreement = float((reference_top == candidate_top).float().mean())
        value_difference = float(
            (
                torch.softmax(reference.value_logits.float(), dim=1)
                - torch.softmax(candidate.value_logits.float(), dim=1)
            )
            .abs()
            .max()
        )
    synchronize(device)
    return {
        "batch_size": host.batch_size,
        "finite_outputs": finite,
        "max_legal_probability_difference": max_probability_difference,
        "max_total_variation_distance": total_variation,
        "greedy_action_agreement": round(top1_agreement, 6),
        "max_value_probability_difference": value_difference,
        # "Stable" here means: finite, and the legal-set distribution has not
        # moved enough to change which actions are plausibly sampled.
        "stable": finite and max_probability_difference < 5e-2 and total_variation < 5e-2,
    }


def legality_ab_repeatability(
    *,
    model,
    device: torch.device,
    dtype: torch.dtype,
    pool: PositionPool,
    batch_size: int,
    capacity: int,
    pairs: int = 5,
    target_pass_seconds: float = 2.0,
    seed: int = 4,
) -> dict:
    """Interleaved dense/compact A/B at one configuration, repeated `pairs` times.

    A single dense-versus-compact comparison sits close enough to the adoption
    margin that it can land on either side from run to run. Measuring the two
    paths back to back, several times, turns "which one won today" into a
    distribution that can be reported and reasoned about. Interleaving also
    cancels slow drift such as thermal throttling, which a block of dense
    followed by a block of compact would absorb into the difference.
    """
    host_dense = make_host_batch(pool, batch_size)
    host_compact = make_host_batch(pool, batch_size, capacity=capacity)
    dense_rates: list[float] = []
    compact_rates: list[float] = []
    gains: list[float] = []
    for index in range(pairs):
        dense = benchmark_configuration(
            model=model,
            device=device,
            dtype=dtype,
            host=host_dense,
            legality_mode="dense",
            trials=1,
            target_pass_seconds=target_pass_seconds,
            seed=seed + index,
        )
        compact = benchmark_configuration(
            model=model,
            device=device,
            dtype=dtype,
            host=host_compact,
            legality_mode="compact",
            trials=1,
            target_pass_seconds=target_pass_seconds,
            seed=seed + index,
        )
        if dense["status"] != "OK" or compact["status"] != "OK":
            continue
        dense_rates.append(dense["positions_per_second"])
        compact_rates.append(compact["positions_per_second"])
        gains.append(compact["positions_per_second"] / dense["positions_per_second"] - 1.0)

    if not gains:
        return {"pairs": 0, "usable": False}
    return {
        "pairs": len(gains),
        "usable": True,
        "batch_size": batch_size,
        "precision": {v: k for k, v in DTYPE_BY_NAME.items()}[dtype],
        "compact_capacity": capacity,
        "dense_positions_per_second": [round(rate, 2) for rate in dense_rates],
        "compact_positions_per_second": [round(rate, 2) for rate in compact_rates],
        "compact_gain_per_pair": [round(gain, 5) for gain in gains],
        "mean_gain": round(statistics.fmean(gains), 5),
        "median_gain": round(statistics.median(gains), 5),
        "min_gain": round(min(gains), 5),
        "max_gain": round(max(gains), 5),
        "stdev_gain": round(statistics.stdev(gains) if len(gains) > 1 else 0.0, 5),
        "compact_won_every_pair": all(gain > 0 for gain in gains),
    }


def sustained_throughput(
    *,
    model,
    device: torch.device,
    dtype: torch.dtype,
    host: HostBatch,
    legality_mode: str,
    seconds: float = 5.0,
    seed: int = 11,
) -> dict:
    """Run the winning configuration continuously to get a sustainable rate."""
    generator = make_generator(device, seed)
    tokens = host.tokens.to(device=device, dtype=dtype)
    dense_mask = host.dense_mask.to(device) if legality_mode == "dense" else None
    compact = host.compact.to(device) if legality_mode == "compact" else None

    for _ in range(5):
        with torch.inference_mode():
            _model_step(model, tokens, dense_mask, compact, legality_mode, generator)
    synchronize(device)

    steps = 0
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        with torch.inference_mode():
            actions = _model_step(
                model, tokens, dense_mask, compact, legality_mode, generator
            )
        _ = actions.to("cpu")
        steps += 1
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "batch_size": host.batch_size,
        "precision": {v: k for k, v in DTYPE_BY_NAME.items()}[dtype],
        "legality": legality_mode,
        "steps": steps,
        "seconds": round(elapsed, 4),
        "positions_per_second": round(steps * host.batch_size / elapsed, 2),
        "peak_process_memory_bytes": peak_memory_bytes(),
        "mps_memory_bytes": mps_memory_bytes(),
    }


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------


def _best(results: list[dict], legality_mode: str) -> dict | None:
    candidates = [
        entry
        for entry in results
        if entry["status"] == "OK"
        and entry["legality"] == legality_mode
        and entry.get("illegal_samples", 1) == 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda entry: entry["positions_per_second"])


def recommend_legality(report: dict, precision: str | None = None) -> tuple[str, str]:
    """Dense or compact, decided on the measured end-to-end model step.

    A faster legality *stage* is not the question: the question is whether the
    complete step gets faster, because that is what the coordinator pays.

    `precision` pins the comparison to the recommended precision. Without it the
    verdict would ride on whichever precision happened to win the dense race,
    and float16 and bfloat16 are within noise of each other here.
    """
    candidates = [
        entry
        for entry in report["results"]
        if entry["status"] == "OK"
        and entry.get("illegal_samples", 1) == 0
        and (precision is None or entry["precision"] == precision)
    ]

    def _pick(mode: str) -> dict | None:
        rows = [entry for entry in candidates if entry["legality"] == mode]
        return max(rows, key=lambda entry: entry["positions_per_second"]) if rows else None

    dense = _pick("dense") or report.get("best_dense_configuration")
    compact = _pick("compact") or report.get("best_compact_configuration")
    if dense is None and compact is None:
        return "undetermined", "No legality configuration completed successfully."
    if compact is None:
        return "dense", "The compact path produced no successful configuration."
    if dense is None:
        return "compact", "The dense path produced no successful configuration."

    headline_gain = (
        compact["positions_per_second"] / dense["positions_per_second"] - 1.0
        if dense["positions_per_second"]
        else 0.0
    )

    # Compare the two paths at one matched configuration as well, so the verdict
    # is not an artefact of two different winning batch sizes or precisions.
    by_mode = {
        entry["legality"]: entry
        for entry in report["results"]
        if entry["status"] == "OK"
        and entry["batch_size"] == dense["batch_size"]
        and entry["precision"] == dense["precision"]
    }
    matched_note = ""
    decision_gain = headline_gain
    if {"dense", "compact"} <= set(by_mode):
        decision_gain = (
            by_mode["compact"]["positions_per_second"]
            / by_mode["dense"]["positions_per_second"]
            - 1.0
        )
        stage_ratio = (
            by_mode["compact"]["legality_sampling_positions_per_second"]
            / by_mode["dense"]["legality_sampling_positions_per_second"]
        )
        matched_note = (
            f" At the matched configuration (batch {dense['batch_size']}, "
            f"{dense['precision']}) compact is {decision_gain * 100:+.1f} percent on the "
            f"end-to-end step while its legality+sampling stage alone is "
            f"{stage_ratio:.1f}x faster: the encoder dominates the step, so the stage "
            "speed-up does not carry through."
        )

    # The sweep measures compact at the pool's observed capacity, which no
    # production run could rely on. Re-take the verdict at the capacity a real
    # coordinator would have to pad to.
    conservative_note = ""
    matched_dense = by_mode.get("dense")
    production = [
        entry
        for entry in report.get("compact_capacity_sensitivity", [])
        if entry["status"] == "OK"
        and entry["precision"] == dense["precision"]
        and entry["capacity"] == PRODUCTION_COMPACT_CAPACITY
    ]
    # Only comparable when both sides were measured at the same batch size.
    if production and matched_dense and matched_dense["positions_per_second"]:
        production = [
            entry
            for entry in production
            if entry["batch_size"] == matched_dense["batch_size"]
        ]
    if production and matched_dense and matched_dense["positions_per_second"]:
        conservative_gain = (
            production[0]["positions_per_second"] / matched_dense["positions_per_second"] - 1.0
        )
        conservative_note = (
            f" Priced at the production padding capacity of "
            f"{PRODUCTION_COMPACT_CAPACITY} rather than the pool's observed "
            f"{report['position_pool']['compact_capacity']}, compact is "
            f"{conservative_gain * 100:+.1f} percent against the same dense "
            "configuration."
        )
        decision_gain = conservative_gain

    # A single comparison lands close enough to the margin to flip between runs,
    # so the verdict is taken on the repeated interleaved A/B when one exists:
    # compact must beat dense on average *and* in every pair. A representation
    # that only wins on some runs is not a benefit worth imposing on Agent 5.
    repeatability = report.get("legality_ab_repeatability") or {}
    robust_note = ""
    robust = None
    if repeatability.get("usable"):
        decision_gain = repeatability["mean_gain"]
        robust = (
            decision_gain > COMPACT_ADOPTION_MARGIN
            and repeatability["compact_won_every_pair"]
        )
        robust_note = (
            f" Repeatability, {repeatability['pairs']} interleaved dense/compact pairs at "
            f"batch {repeatability['batch_size']}, {repeatability['precision']}, capacity "
            f"{repeatability['compact_capacity']}: mean {repeatability['mean_gain'] * 100:+.1f} "
            f"percent, median {repeatability['median_gain'] * 100:+.1f} percent, range "
            f"{repeatability['min_gain'] * 100:+.1f} to {repeatability['max_gain'] * 100:+.1f} "
            f"percent, standard deviation {repeatability['stdev_gain'] * 100:.1f} percentage "
            f"points, compact ahead in every pair: {repeatability['compact_won_every_pair']}."
        )

    if robust if robust is not None else decision_gain > COMPACT_ADOPTION_MARGIN:
        choice = "compact"
        verdict = (
            f"Compact legality is recommended: {decision_gain * 100:.1f} percent faster "
            f"end to end, above the {COMPACT_ADOPTION_MARGIN * 100:.0f} percent adoption "
            "margin and ahead of dense in every repeated pair."
        )
    else:
        choice = "dense"
        verdict = (
            f"Dense legality is recommended: compact is only {decision_gain * 100:+.1f} "
            f"percent end to end, which does not clear the "
            f"{COMPACT_ADOPTION_MARGIN * 100:.0f} percent adoption margin robustly, so its "
            "padded fixed-capacity representation and the capacity-overflow failure mode "
            "it introduces are not justified. Sparse legality should not be forced on "
            "later agents."
        )
    verdict += (
        f" Legality transport costs {dense['legality_bytes_per_position']:.0f} "
        f"bytes/position dense against {compact['legality_bytes_per_position']:.0f} "
        f"bytes/position compact at capacity {report['position_pool']['compact_capacity']}."
    )

    # A capacity chosen from one sample of positions is not a safe production
    # bound, so say what happens when it is sized conservatively instead.
    sensitivity = [
        entry
        for entry in report.get("compact_capacity_sensitivity", [])
        if entry["status"] == "OK" and entry["precision"] == dense["precision"]
    ]
    if len(sensitivity) > 1:
        smallest = min(sensitivity, key=lambda entry: entry["capacity"])
        largest = max(sensitivity, key=lambda entry: entry["capacity"])
        change = (
            largest["positions_per_second"] / smallest["positions_per_second"] - 1.0
            if smallest["positions_per_second"]
            else 0.0
        )
        verdict += (
            f" Widening the padding capacity from {smallest['capacity']} to "
            f"{largest['capacity']} changes the compact end-to-end rate by "
            f"{change * 100:+.1f} percent."
        )
    return choice, verdict + matched_note + conservative_note + robust_note


def recommend_precision(report: dict) -> tuple[str, str]:
    """float32 unless a reduced mode is both meaningfully faster and stable."""
    successes = [entry for entry in report["results"] if entry["status"] == "OK"]
    if not successes:
        return "undetermined", "No configuration completed successfully."

    best_by_precision: dict[str, dict] = {}
    for entry in successes:
        current = best_by_precision.get(entry["precision"])
        if current is None or entry["positions_per_second"] > current["positions_per_second"]:
            best_by_precision[entry["precision"]] = entry

    baseline = best_by_precision.get("float32")
    if baseline is None:
        return "undetermined", "The float32 baseline did not complete."

    notes = []
    eligible: list[tuple[str, float, float]] = []
    for name, entry in sorted(best_by_precision.items()):
        if name == "float32":
            continue
        stability = report["precision_stability"].get(name, {})
        gain = entry["positions_per_second"] / baseline["positions_per_second"] - 1.0
        agreement = stability.get("greedy_action_agreement", 0.0)
        notes.append(
            f"{name}: {gain * 100:+.1f} percent against float32, stable="
            f"{stability.get('stable')}, max legal-probability difference "
            f"{stability.get('max_legal_probability_difference', float('nan')):.2e}, "
            f"greedy agreement {agreement:.3f}"
        )
        if stability.get("stable") and gain > PRECISION_ADOPTION_MARGIN:
            eligible.append((name, gain, agreement))

    if not eligible:
        choice = "float32"
        verdict = (
            "float32 is recommended. No reduced precision cleared the "
            f"{PRECISION_ADOPTION_MARGIN * 100:.0f} percent margin on the complete "
            "inference + legality + sampling path while remaining stable."
        )
    else:
        fastest_gain = max(gain for _, gain, _ in eligible)
        # Within the speed tie band, keep the mode that stays closest to the
        # float32 distribution rather than the one that is nominally fastest.
        contenders = [
            record for record in eligible if fastest_gain - record[1] <= PRECISION_SPEED_TIE
        ]
        choice, chosen_gain, chosen_agreement = max(
            contenders, key=lambda record: (record[2], record[1])
        )
        verdict = (
            f"{choice} is recommended: {chosen_gain * 100:.1f} percent faster than float32 "
            "on the complete inference + legality + sampling path, with greedy-action "
            f"agreement {chosen_agreement:.3f} and the legal-set distribution unchanged "
            "within tolerance."
        )
        if len(contenders) > 1:
            verdict += (
                f" Ties within {PRECISION_SPEED_TIE * 100:.0f} percent on speed were broken "
                "on agreement with the float32 distribution."
            )
    unsupported = sorted(
        name
        for name, support in report["precision_support"].items()
        if not support.get("supported")
    )
    if unsupported:
        verdict += f" Unsupported or unstable on this device: {', '.join(unsupported)}."
    return choice, verdict + " Measured: " + "; ".join(notes) + "."


def run_full_benchmark(
    *,
    pool: PositionPool,
    batch_sizes=DEFAULT_BATCH_SIZES,
    precisions=DEFAULT_PRECISIONS,
    legality_modes=LEGALITY_MODES,
    model_config: RepresentativeConfig | None = None,
    trials: int = 3,
    legality_ab_pairs: int = 5,
    target_pass_seconds: float = 1.5,
    sustained_seconds: float = 5.0,
    seed: int = 4,
    progress=None,
) -> dict:
    """Benchmark every feasible (batch size, precision, legality) point on MPS."""
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "Metal Performance Shaders are not available; Agent 4 requires MPS "
            "and must not substitute central-processing-unit results"
        )
    device = torch.device("mps")
    config = model_config or RepresentativeConfig()

    report: dict = {
        "benchmark_version": BENCHMARK_VERSION,
        "representative_model_version": REPRESENTATIVE_MODEL_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "device": detect_device_report(),
        "position_pool": pool.stats(),
        "batch_sizes_requested": list(batch_sizes),
        "precision_modes_requested": list(precisions),
        "legality_modes": list(legality_modes),
        "trials_per_configuration": trials,
        "results": [],
        "precision_support": {},
        "precision_stability": {},
        "legality_equivalence": [],
        "determinism": {},
        "failures": [],
    }

    models: dict[str, object] = {}
    for name in precisions:
        dtype = DTYPE_BY_NAME[name]
        try:
            model = build_representative_model(config, seed=seed, device=device, dtype=dtype)
            probe = make_host_batch(pool, 64)
            with torch.inference_mode():
                outputs = model(probe.tokens.to(device=device, dtype=dtype))
                finite = bool(outputs.all_finite())
            synchronize(device)
            if not finite:
                raise RuntimeError("forward pass produced non-finite values")
            models[name] = model
            report["precision_support"][name] = {"supported": True}
        except (RuntimeError, TypeError) as error:
            report["precision_support"][name] = {"supported": False, "error": str(error)}
            report["failures"].append(
                {"stage": "precision_probe", "precision": name, "error": str(error)}
            )

    if "float32" not in models:
        raise RuntimeError("float32 baseline is required but could not be built on MPS")

    report["representative_model_parameter_count"] = models["float32"].parameter_count()
    report["architecture_summary"] = models["float32"].architecture_summary()

    # -- correctness probes at a single mid-sized batch ----------------------
    probe_batch = make_host_batch(pool, 256, offset=13)
    for name, model in models.items():
        dtype = DTYPE_BY_NAME[name]
        equivalence = legality_equivalence_check(
            model=model, device=device, dtype=dtype, host=probe_batch
        )
        equivalence["precision"] = name
        report["legality_equivalence"].append(equivalence)
        report["determinism"][name] = determinism_check(
            model=model, device=device, dtype=dtype, host=probe_batch
        )
        if name != "float32":
            report["precision_stability"][name] = precision_agreement_check(
                reference_model=models["float32"],
                candidate_model=model,
                device=device,
                candidate_dtype=dtype,
                host=probe_batch,
            )

    # -- the sweep ------------------------------------------------------------
    for batch_size in batch_sizes:
        host = make_host_batch(pool, batch_size)
        for name, model in models.items():
            for legality_mode in legality_modes:
                if progress is not None:
                    progress(f"batch={batch_size} precision={name} legality={legality_mode}")
                entry = benchmark_configuration(
                    model=model,
                    device=device,
                    dtype=DTYPE_BY_NAME[name],
                    host=host,
                    legality_mode=legality_mode,
                    trials=trials,
                    target_pass_seconds=target_pass_seconds,
                    seed=seed,
                )
                report["results"].append(entry)
                if entry["status"] != "OK":
                    report["failures"].append(
                        {
                            "stage": "benchmark",
                            "batch_size": batch_size,
                            "precision": name,
                            "legality": legality_mode,
                            "error": entry.get("error"),
                            "out_of_memory": entry.get("out_of_memory"),
                        }
                    )
                elif entry.get("illegal_samples"):
                    report["failures"].append(
                        {
                            "stage": "legality",
                            "batch_size": batch_size,
                            "precision": name,
                            "legality": legality_mode,
                            "illegal_samples": entry["illegal_samples"],
                        }
                    )
        del host
        gc.collect()

    report["batch_sizes_completed"] = sorted(
        {entry["batch_size"] for entry in report["results"] if entry["status"] == "OK"}
    )
    report["dense_legality_results"] = [
        entry for entry in report["results"] if entry["legality"] == "dense"
    ]
    report["compact_legality_results"] = [
        entry for entry in report["results"] if entry["legality"] == "compact"
    ]
    report["best_dense_configuration"] = _best(report["results"], "dense")
    report["best_compact_configuration"] = _best(report["results"], "compact")

    # -- capacity sensitivity of the compact path ------------------------------
    #
    # The pool's observed maximum is not a safe production capacity: a single
    # sample of positions cannot bound the legal-action count. Measure whether
    # the compact path still wins once the padding is sized conservatively.
    #
    # The probe runs at the batch size the legality verdict will be taken at,
    # so the production-capacity number is directly comparable to the dense
    # measurement it is judged against. That means the precision recommendation
    # has to be settled first; it depends only on the sweep.
    precision_choice, precision_rationale = recommend_precision(report)
    completed = set(report["batch_sizes_completed"])
    dense_at_precision = [
        entry
        for entry in report["results"]
        if entry["status"] == "OK"
        and entry["legality"] == "dense"
        and entry["precision"] == precision_choice
        and entry.get("illegal_samples", 1) == 0
    ]
    probe_batch_size = (
        max(dense_at_precision, key=lambda entry: entry["positions_per_second"])["batch_size"]
        if dense_at_precision
        else max(size for size in batch_sizes if size in completed)
    )

    report["compact_capacity_sensitivity"] = []
    report["compact_capacity_sensitivity_batch_size"] = probe_batch_size
    largest = probe_batch_size
    for capacity in sorted({pool.capacity, 64, PRODUCTION_COMPACT_CAPACITY, 256}):
        for name in models:
            host = make_host_batch(pool, largest, capacity=capacity)
            entry = benchmark_configuration(
                model=models[name],
                device=device,
                dtype=DTYPE_BY_NAME[name],
                host=host,
                legality_mode="compact",
                # Same trial count as the sweep: the compact-versus-dense verdict
                # is taken off these numbers, so they cannot be noisier than the
                # numbers they are compared against.
                trials=trials,
                target_pass_seconds=target_pass_seconds,
                seed=seed,
            )
            report["compact_capacity_sensitivity"].append(
                {
                    "capacity": capacity,
                    "precision": name,
                    "batch_size": largest,
                    "status": entry["status"],
                    "positions_per_second": entry.get("positions_per_second"),
                    "legality_sampling_positions_per_second": entry.get(
                        "legality_sampling_positions_per_second"
                    ),
                    "legality_bytes_per_position": entry.get("legality_bytes_per_position"),
                    "illegal_samples": entry.get("illegal_samples"),
                }
            )
            del host
        gc.collect()

    # -- repeatability of the dense-versus-compact difference ------------------
    if precision_choice in models:
        report["legality_ab_repeatability"] = legality_ab_repeatability(
            model=models[precision_choice],
            device=device,
            dtype=DTYPE_BY_NAME[precision_choice],
            pool=pool,
            batch_size=probe_batch_size,
            capacity=PRODUCTION_COMPACT_CAPACITY,
            pairs=legality_ab_pairs,
            target_pass_seconds=target_pass_seconds,
            seed=seed,
        )
    else:  # pragma: no cover - float32 is required, so this cannot normally happen
        report["legality_ab_repeatability"] = {"pairs": 0, "usable": False}

    # -- recommendations ------------------------------------------------------
    # `precision_choice` was settled above, before the capacity probe. The
    # legality verdict is taken at that precision rather than at whichever
    # precision happened to win the dense race.
    legality_choice, legality_rationale = recommend_legality(report, precision_choice)
    report["recommended_legality_representation"] = legality_choice
    report["recommended_legality_rationale"] = legality_rationale
    report["recommended_precision"] = precision_choice
    report["recommended_precision_rationale"] = precision_rationale

    # -- sustained rates ------------------------------------------------------
    #
    # Three sustained runs, because they answer three different questions:
    #
    # - the fastest configuration measured, whatever it is;
    # - the configuration actually recommended for Phase 4, which may be slower
    #   than the fastest once robustness is priced in;
    # - the conservative float32 + dense baseline.
    def _run_sustained(entry: dict | None) -> dict | None:
        if entry is None or entry["precision"] not in models:
            return None
        host = make_host_batch(pool, entry["batch_size"], offset=29)
        return sustained_throughput(
            model=models[entry["precision"]],
            device=device,
            dtype=DTYPE_BY_NAME[entry["precision"]],
            host=host,
            legality_mode=entry["legality"],
            seconds=sustained_seconds,
        )

    successes = [entry for entry in report["results"] if entry["status"] == "OK"]
    fastest = (
        max(successes, key=lambda entry: entry["positions_per_second"])
        if successes
        else None
    )
    recommended_candidates = [
        entry
        for entry in successes
        if entry["precision"] == precision_choice and entry["legality"] == legality_choice
    ]
    recommended = (
        max(recommended_candidates, key=lambda entry: entry["positions_per_second"])
        if recommended_candidates
        else None
    )
    baseline_candidates = [
        entry
        for entry in successes
        if entry["precision"] == "float32" and entry["legality"] == "dense"
    ]
    baseline = (
        max(baseline_candidates, key=lambda entry: entry["positions_per_second"])
        if baseline_candidates
        else None
    )

    report["sustained_throughput"] = _run_sustained(fastest)
    report["sustained_throughput_recommended"] = (
        report["sustained_throughput"]
        if recommended is not None
        and fastest is not None
        and (
            recommended["batch_size"],
            recommended["precision"],
            recommended["legality"],
        )
        == (fastest["batch_size"], fastest["precision"], fastest["legality"])
        else _run_sustained(recommended)
    )
    report["sustained_throughput_float32_dense"] = (
        report["sustained_throughput"]
        if baseline is not None
        and fastest is not None
        and (baseline["batch_size"], baseline["precision"], baseline["legality"])
        == (fastest["batch_size"], fastest["precision"], fastest["legality"])
        else _run_sustained(baseline)
    )

    report["peak_memory_bytes"] = peak_memory_bytes()
    report["mps_memory_bytes"] = mps_memory_bytes()
    return report
