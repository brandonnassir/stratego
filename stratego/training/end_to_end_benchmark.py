"""Integrated correctness, scaling and soak measurement for the Phase 3 decision.

Specification source: `05_AGENT_5_END_TO_END_DECISION.md`.

Three things live here:

1. **The integrated correctness gate.** An independent set of engine games is
   advanced in lockstep with the live pipeline and compared against what the
   pipeline actually published and did. This is stronger than Agent 1's or
   Agent 2's differential runs because the action being applied is the one the
   *model* sampled, so the comparison covers the whole chain -- worker publish,
   coordinator inference, legality, sampling, write-back, worker step, republish
   -- rather than the simulation layer alone.

2. **Scaling measurement.** Short screening runs across the required
   worker/environment/inference-batch dimensions, then longer sustained runs on
   the finalists.

3. **The soak.** The chosen configuration run continuously, sampling memory,
   swap, throughput, terminal reasons and worker liveness at a fixed interval so
   a growth trend or a throughput collapse is visible rather than inferred.

The decision ratio
------------------
`R = sustainable simulation-pipeline positions per second / sustainable
representative-model inference positions per second`.

The numerator is measured here, by running the *same* worker pool at the same
worker and environment count with the model removed and Agent 2's deterministic
benchmark policy in its place. It is deliberately not the Phase 2 single-core
rate multiplied by a core count -- Agent 2 showed that extrapolation overstates
the real figure by roughly a factor of two.
"""

from __future__ import annotations

import platform
import re
import statistics
import subprocess
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from ..engine.constants import PLAYERS, RulesConfig, TRAINING_RULES
from ..engine.legal_moves import legal_action_mask, legal_actions
from ..engine.observation import build_observation
from ..engine.random_play import make_random_setups
from ..engine.state import create_game
from ..engine.transition import apply_action
from .batch_simulation import derive_slot_seed, slot_game_id
from .coordinator import (
    CoordinatorConfig,
    CoordinatorError,
    RunTotals,
    SelfPlayCoordinator,
    mps_memory_bytes,
)
from .mps_benchmark import peak_memory_bytes
from .shared_buffers import (
    NO_ACTING_PLAYER,
    STATUS_ACTIVE,
    STATUS_TERMINAL,
    terminal_reason_name,
)
from .worker_pool import WorkerPool, select_actions

BENCHMARK_VERSION = "agent_05_end_to_end_0.1.0"

#: Required screening dimensions.
WORKER_COUNTS = (4, 6, 8, 10, 12)
ENVIRONMENT_COUNTS = (256, 512, 1024, 1536, 2048)
INFERENCE_BATCH_SIZES = (64, 128, 256, 512, 1024, 1536, 2048)

#: Decision-rule thresholds.
KEEP_PYTHON_RATIO = 2.0
OPTIONAL_OPTIMISATION_RATIO = 1.25

DECISION_KEEP_PYTHON = "KEEP_PYTHON"
DECISION_KEEP_PYTHON_OPTIONAL = "KEEP_PYTHON_OPTIMIZATION_OPTIONAL"
DECISION_BUILD_BACKEND = "BUILD_OPTIMIZED_BACKEND"


# ---------------------------------------------------------------------------
# System sampling
# ---------------------------------------------------------------------------


def swap_bytes() -> dict:
    """System swap usage, from `sysctl vm.swapusage`."""
    try:
        output = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except Exception:  # pragma: no cover - platform dependent
        return {}
    values = {}
    for name, raw, unit in re.findall(r"(\w+)\s*=\s*([\d.]+)([KMG])", output):
        scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[unit]
        values[f"swap_{name}_bytes"] = int(float(raw) * scale)
    return values


def platform_report() -> dict:
    return {
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


# ---------------------------------------------------------------------------
# Integrated correctness gate
# ---------------------------------------------------------------------------


@dataclass
class GateReport:
    """Aggregate result of one integrated differential run."""

    environment_steps: int = 0
    global_steps: int = 0
    row_comparisons: int = 0
    action_legality_checks: int = 0
    games_completed: int = 0
    resets_observed: int = 0
    distinct_trajectory_keys: int = 0
    mismatches: int = 0
    mismatch_categories: dict = field(default_factory=dict)
    mismatch_details: list = field(default_factory=list)
    terminal_reason_counts: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def note(self, category: str, detail: str) -> None:
        self.mismatches += 1
        self.mismatch_categories[category] = self.mismatch_categories.get(category, 0) + 1
        if len(self.mismatch_details) < 40:
            self.mismatch_details.append({"category": category, "detail": detail})

    def as_dict(self) -> dict:
        return {
            "environment_steps": self.environment_steps,
            "global_steps": self.global_steps,
            "row_comparisons": self.row_comparisons,
            "action_legality_checks": self.action_legality_checks,
            "games_completed": self.games_completed,
            "resets_observed": self.resets_observed,
            "distinct_trajectory_keys": self.distinct_trajectory_keys,
            "mismatches": self.mismatches,
            "mismatch_categories": dict(self.mismatch_categories),
            "mismatch_details": list(self.mismatch_details),
            "terminal_reason_counts": dict(self.terminal_reason_counts),
            "elapsed_seconds": self.elapsed_seconds,
        }


def reference_game(
    root_seed: int,
    environment_id: int,
    generation: int,
    rules: RulesConfig = TRAINING_RULES,
):
    """The game a slot must contain, built straight from the frozen engine.

    The same construction Agent 1's and Agent 2's differential harnesses use.
    It is repeated here rather than imported from `tests/` so the measurement
    library does not depend on the test package.
    """
    seed = derive_slot_seed(root_seed, environment_id, generation)
    red_setup, blue_setup = make_random_setups(seed)
    return create_game(
        red_setup,
        blue_setup,
        rules=rules,
        game_id=slot_game_id(root_seed, environment_id, generation),
    )


class ReferenceMirror:
    """Independently built engine games, one per slot, advanced in lockstep.

    The mirror never reads a worker's simulator. It rebuilds each slot's game
    from `(root_seed, environment_id, generation)` -- the identity triple Agent 1
    made sufficient -- and advances it with the action the coordinator actually
    sampled. Everything the pipeline publishes is then checked against it.
    """

    def __init__(
        self,
        num_environments: int,
        *,
        root_seed: int,
        rules: RulesConfig = TRAINING_RULES,
    ) -> None:
        self.num_environments = num_environments
        self.root_seed = int(root_seed)
        self.rules = rules
        self.generations = np.zeros(num_environments, dtype=np.int64)
        self.states = [
            reference_game(self.root_seed, slot, 0, rules)
            for slot in range(num_environments)
        ]
        self.trajectory_keys: set[tuple[int, int]] = {
            (slot, 0) for slot in range(num_environments)
        }

    # -- comparison ---------------------------------------------------------

    def check_published(self, buffers, report: GateReport) -> None:
        """Compare every published row against the reference for that slot."""
        for slot in range(self.num_environments):
            reference = self.states[slot]
            report.row_comparisons += 1

            generation = int(buffers.generation[slot])
            if generation != int(self.generations[slot]):
                report.note(
                    "environment_generation",
                    f"slot {slot}: published generation {generation} != "
                    f"reference {int(self.generations[slot])}",
                )
                continue
            if int(buffers.environment_id[slot]) != slot:
                report.note(
                    "trajectory_identifier",
                    f"slot {slot}: published environment_id "
                    f"{int(buffers.environment_id[slot])}",
                )
            expected_game_id = slot_game_id(self.root_seed, slot, generation)
            if reference.game_id != expected_game_id:
                report.note(
                    "trajectory_identifier",
                    f"slot {slot}: reference game_id {reference.game_id} != "
                    f"{expected_game_id}",
                )

            if int(buffers.ply[slot]) != reference.total_moves:
                report.note(
                    "resulting_state",
                    f"slot {slot}: published ply {int(buffers.ply[slot])} != "
                    f"reference {reference.total_moves}",
                )
            if int(buffers.battleless_moves[slot]) != reference.battleless_moves:
                report.note(
                    "resulting_state",
                    f"slot {slot}: battleless counter differs",
                )

            # A published slot is always non-terminal: the pool resets a
            # finished game inside the same phase it finished.
            if reference.terminal:
                report.note(
                    "resulting_state",
                    f"slot {slot}: reference is terminal but the slot was published",
                )
                continue
            if int(buffers.status[slot]) != STATUS_ACTIVE:
                report.note(
                    "resulting_state",
                    f"slot {slot}: status {int(buffers.status[slot])} is not active",
                )
            if int(buffers.acting_player[slot]) != reference.acting_player:
                report.note(
                    "resulting_state",
                    f"slot {slot}: acting player "
                    f"{int(buffers.acting_player[slot])} != {reference.acting_player}",
                )

            reference_legal = legal_actions(reference)
            if int(buffers.legal_count[slot]) != len(reference_legal):
                report.note(
                    "legal_actions",
                    f"slot {slot}: legal_count {int(buffers.legal_count[slot])} != "
                    f"{len(reference_legal)}",
                )
            reference_mask = legal_action_mask(reference, reference_legal)
            if not np.array_equal(buffers.legal_mask[slot], reference_mask):
                report.note("legal_mask", f"slot {slot}: dense legal mask differs")

            expected_observation = build_observation(reference, reference.acting_player)
            if not np.array_equal(buffers.observations[slot], expected_observation):
                report.note(
                    "observation",
                    f"slot {slot}: published observation differs from the reference",
                )

    def check_and_apply(self, actions: np.ndarray, report: GateReport) -> list[int]:
        """Verify each sampled action is legal, then advance the references.

        Returns the slots whose reference game became terminal.
        """
        newly_terminal: list[int] = []
        for slot in range(self.num_environments):
            action = int(actions[slot])
            reference = self.states[slot]
            report.action_legality_checks += 1
            if action < 0:
                report.note(
                    "selected_action_legality",
                    f"slot {slot}: coordinator skipped an active slot",
                )
                continue
            if action not in legal_actions(reference):
                report.note(
                    "selected_action_legality",
                    f"slot {slot}: sampled action {action} is not legal in the "
                    f"reference game at ply {reference.total_moves}",
                )
                continue
            apply_action(reference, action)
            report.environment_steps += 1
            if reference.terminal:
                newly_terminal.append(slot)
        return newly_terminal

    def check_terminal(self, buffers, slot: int, report: GateReport) -> None:
        """Compare a finished game's published outcome against the reference."""
        reference = self.states[slot]
        report.games_completed += 1
        reason = reference.terminal_reason
        report.terminal_reason_counts[reason] = (
            report.terminal_reason_counts.get(reason, 0) + 1
        )
        if terminal_reason_name(int(buffers.last_terminal_reason[slot])) != reason:
            report.note(
                "terminal_reason",
                f"slot {slot}: published reason "
                f"{terminal_reason_name(int(buffers.last_terminal_reason[slot]))} "
                f"!= reference {reason}",
            )
        published_winner = int(buffers.last_winner[slot])
        expected_winner = -1 if reference.winner is None else reference.winner
        if published_winner != expected_winner:
            report.note(
                "terminal_result",
                f"slot {slot}: published winner {published_winner} != "
                f"reference {expected_winner}",
            )
        if bool(buffers.last_is_draw[slot]) != bool(reference.is_draw):
            report.note("terminal_result", f"slot {slot}: draw flag differs")
        if int(buffers.last_total_moves[slot]) != reference.total_moves:
            report.note("terminal_result", f"slot {slot}: final ply differs")
        if float(buffers.last_result_red[slot]) != reference.result_for(PLAYERS[0]):
            report.note("terminal_result", f"slot {slot}: red result differs")
        if float(buffers.last_result_blue[slot]) != reference.result_for(PLAYERS[1]):
            report.note("terminal_result", f"slot {slot}: blue result differs")

    def reset(self, slot: int, report: GateReport) -> None:
        """Rebuild a slot's reference for its next generation."""
        self.generations[slot] += 1
        generation = int(self.generations[slot])
        self.states[slot] = reference_game(
            self.root_seed, slot, generation, self.rules
        )
        key = (slot, generation)
        if key in self.trajectory_keys:
            report.note(
                "trajectory_identifier",
                f"slot {slot}: trajectory key {key} was reused",
            )
        self.trajectory_keys.add(key)
        report.resets_observed += 1


def run_integrated_gate(
    *,
    num_environments: int = 64,
    num_workers: int = 4,
    inference_batch_size: int = 256,
    target_environment_steps: int = 10_000,
    precision: str = "float16",
    legality: str = "dense",
    root_seed: int = 50_005,
    device=None,
    max_global_steps: int | None = None,
    progress=None,
) -> GateReport:
    """Drive the real pipeline and an independent reference set in lockstep."""
    report = GateReport()
    config = CoordinatorConfig(
        num_environments=num_environments,
        num_workers=num_workers,
        inference_batch_size=inference_batch_size,
        precision=precision,
        legality=legality,
        root_seed=root_seed,
        record_trajectories=False,
        detailed_timing=False,
    )
    mirror = ReferenceMirror(num_environments, root_seed=root_seed)
    started = time.perf_counter()
    coordinator = SelfPlayCoordinator(config, device=device)
    coordinator.start()
    try:
        buffers = coordinator.pool.buffers
        limit = max_global_steps or (target_environment_steps // num_environments + 8)
        while report.environment_steps < target_environment_steps:
            if report.global_steps >= limit:
                break
            # 1. What the workers published must match the reference exactly.
            mirror.check_published(buffers, report)

            # 2. The pipeline takes its step; the actions it wrote are the ones
            #    the reference will be advanced by.
            coordinator.step()
            actions = coordinator.last_actions.copy()
            report.global_steps += 1

            # 3. Every sampled action must be legal, and the reference advances.
            newly_terminal = mirror.check_and_apply(actions, report)

            # 4. Finished games: compare the published outcome, then reset the
            #    reference the same way the pool reset the slot.
            for slot in newly_terminal:
                mirror.check_terminal(buffers, slot, report)
                mirror.reset(slot, report)

            if progress is not None:
                progress(report)
    finally:
        coordinator.shutdown()
    report.distinct_trajectory_keys = len(mirror.trajectory_keys)
    report.elapsed_seconds = time.perf_counter() - started
    return report


def run_reconstruction_gate(
    *,
    num_environments: int = 256,
    num_workers: int = 6,
    inference_batch_size: int = 256,
    target_decisions: int = 10_000,
    precision: str = "float16",
    legality: str = "dense",
    snapshot_interval: int = 32,
    root_seed: int = 50_007,
    max_global_steps: int = 20_000,
    max_concurrent_verifications: int = 8,
    device=None,
    progress=None,
) -> dict:
    """Record real pipeline decisions and reconstruct them through Agent 3.

    Verification round-trips each selected game through the trajectory codec and
    rebuilds every one of its decisions, comparing observation, legal set, dense
    mask, state fingerprint, belief target, public knowledge, acting player and
    selected action against digests captured live at decision time.
    """
    config = CoordinatorConfig(
        num_environments=num_environments,
        num_workers=num_workers,
        inference_batch_size=inference_batch_size,
        precision=precision,
        legality=legality,
        root_seed=root_seed,
        record_trajectories=True,
        snapshot_interval=snapshot_interval,
        verify_target_decisions=target_decisions // num_workers + 1,
        max_concurrent_verifications=max_concurrent_verifications,
        retain_games=1,
        detailed_timing=False,
    )
    started = time.perf_counter()
    coordinator = SelfPlayCoordinator(config, device=device)
    coordinator.start()
    verified = 0
    steps = 0
    try:
        while verified < target_decisions and steps < max_global_steps:
            coordinator.step()
            steps += 1
            if steps % 25 == 0:
                totals = coordinator.pool.recording_totals()
                verified = int(totals["total_verified_decisions"])
                if progress is not None:
                    progress(steps, verified)
    finally:
        totals = coordinator.shutdown()

    return {
        "global_steps": steps,
        "elapsed_seconds": time.perf_counter() - started,
        "decisions_recorded": int(totals.get("total_decisions_recorded", 0)),
        "games_recorded": int(totals.get("total_games_recorded", 0)),
        "decisions_reconstructed": int(totals.get("total_verified_decisions", 0)),
        "games_reconstructed": int(totals.get("total_verified_games", 0)),
        "reconstruction_mismatches": int(
            totals.get("total_reconstruction_mismatches", 0)
        ),
        "games_joined_late": int(totals.get("total_games_joined_late", 0)),
        "record_bytes": int(totals.get("total_record_bytes", 0)),
        "snapshot_bytes": int(totals.get("total_snapshot_bytes", 0)),
        "mismatch_details": list(totals.get("mismatch_details", ())),
        "retained_record_bytes": [len(r) for r in totals.get("retained_records", ())],
        "snapshot_interval": snapshot_interval,
        "collection_policy_version": config.collection_policy_version,
        "configuration": config.as_dict(),
    }


# ---------------------------------------------------------------------------
# Throughput measurement
# ---------------------------------------------------------------------------


def _summarise_latencies(latencies: list[float]) -> dict:
    if not latencies:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(latencies)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean_ms": 1000 * statistics.fmean(latencies),
        "p50_ms": 1000 * statistics.median(ordered),
        "p95_ms": 1000 * ordered[index],
        "max_ms": 1000 * ordered[-1],
    }


def measure_configuration(
    config: CoordinatorConfig,
    *,
    seconds: float,
    warmup_steps: int = 5,
    device=None,
    min_steps: int = 8,
) -> dict:
    """Run one configuration and report throughput and where the time went."""
    coordinator = SelfPlayCoordinator(config, device=device)
    result: dict = {"configuration": config.as_dict()}
    swap_start = swap_bytes()
    try:
        coordinator.start()
        for _ in range(warmup_steps):
            coordinator.step()
        # Reset so warm-up allocation and the first Metal compile do not count.
        coordinator.totals = RunTotals()
        started = time.perf_counter()
        steps = 0
        while time.perf_counter() - started < seconds or steps < min_steps:
            coordinator.step()
            steps += 1
        elapsed = time.perf_counter() - started
        totals = coordinator.totals

        worker_wall = totals.worker_seconds
        result.update(
            {
                "status": "ok",
                "measured_seconds": elapsed,
                "global_steps": totals.steps,
                "positions": totals.positions,
                "transitions": totals.transitions,
                "games": coordinator.games_finished,
                "resets": totals.resets,
                "chunks_per_step": totals.chunks / max(totals.steps, 1),
                "positions_per_second": totals.positions / elapsed,
                "transitions_per_second": totals.transitions / elapsed,
                "games_per_second": coordinator.games_finished / elapsed,
                "resets_per_second": totals.resets / elapsed,
                "observation_seconds": totals.observation_seconds,
                "legality_seconds": totals.legality_seconds,
                "transfer_seconds": totals.transfer_seconds,
                "inference_seconds": totals.inference_seconds,
                "sampling_seconds": totals.sampling_seconds,
                "writeback_seconds": totals.writeback_seconds,
                "worker_seconds": worker_wall,
                "barrier_seconds": totals.barrier_seconds,
                "straggler_seconds": totals.straggler_seconds,
                # The coordinator is busy whenever it is not inside `pool.step`,
                # and every worker is blocked for exactly that time in a strictly
                # bulk-synchronous loop.
                "coordinator_active_fraction": totals.coordinator_seconds / elapsed,
                "coordinator_wait_fraction": worker_wall / elapsed,
                "mps_active_fraction": totals.inference_seconds / elapsed,
                "worker_active_fraction": totals.worker_busy_seconds
                / max(elapsed * config.num_workers, 1e-9),
                "worker_barrier_wait_fraction": 1.0
                - (
                    totals.worker_busy_seconds
                    / max(elapsed * config.num_workers, 1e-9)
                ),
                "terminal_reason_counts": dict(coordinator.terminal_reason_counts),
                "process_memory_bytes": peak_memory_bytes(),
                "shared_memory_bytes": coordinator.pool.buffers.nbytes,
                "mps_memory": mps_memory_bytes(),
                **_summarise_latencies(totals.step_latencies),
            }
        )
    except Exception as error:  # noqa: BLE001 - a failed point is a result
        result.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
    finally:
        try:
            pool_totals = coordinator.shutdown()
        except Exception:  # pragma: no cover - defensive
            pool_totals = {}
    if result.get("status") == "ok":
        result["worker_max_rss_bytes"] = int(pool_totals.get("worker_max_rss_bytes", 0))
        result["worker_cpu_seconds"] = float(pool_totals.get("worker_cpu_seconds", 0.0))
        result["recording"] = {
            key: pool_totals.get(key, 0)
            for key in (
                "total_decisions_recorded",
                "total_games_recorded",
                "total_record_bytes",
                "total_verified_decisions",
                "total_reconstruction_mismatches",
                "recording_seconds",
            )
        }
        result["swap_start"] = swap_start
        result["swap_end"] = swap_bytes()
    return result


def measure_simulation_pipeline(
    *,
    num_environments: int,
    num_workers: int,
    seconds: float,
    root_seed: int = 50_011,
    rules: RulesConfig = TRAINING_RULES,
    warmup_steps: int = 5,
    record_trajectories: bool = False,
    snapshot_interval: int = 32,
) -> dict:
    """The R numerator: the simulation pipeline with the model removed.

    Observation building, legality generation, the engine transition, the
    shared-memory transport, worker synchronisation and independent reset -- all
    of it, at the same worker and environment count the end-to-end finalist uses,
    driven by Agent 2's deterministic benchmark policy instead of the network.
    """
    from .worker_pool import RecordingConfig

    recording = RecordingConfig(
        enabled=record_trajectories,
        snapshot_interval=snapshot_interval,
    )
    pool = WorkerPool(
        num_environments,
        num_workers,
        root_seed=root_seed,
        rules=rules,
        recording=recording,
    )
    pool.start()
    latencies: list[float] = []
    try:
        actions = np.full(num_environments, -1, dtype=np.int32)
        for _ in range(warmup_steps):
            select_actions(pool.buffers, pool.root_seed, actions)
            pool.set_actions(actions)
            pool.step()

        positions = 0
        transitions = 0
        steps = 0
        started = time.perf_counter()
        while time.perf_counter() - started < seconds:
            step_started = time.perf_counter()
            active = int(np.count_nonzero(pool.buffers.status == STATUS_ACTIVE))
            select_actions(pool.buffers, pool.root_seed, actions)
            if record_trajectories:
                # Recording needs a decision to store; a flat distribution over
                # the legal set is enough to exercise the storage path at cost.
                _write_uniform_decisions(pool)
            pool.set_actions(actions)
            report = pool.step()
            positions += active
            transitions += report.stepped
            steps += 1
            latencies.append(time.perf_counter() - step_started)
        elapsed = time.perf_counter() - started
    finally:
        totals = pool.shutdown()

    return {
        "num_workers": num_workers,
        "num_environments": num_environments,
        "record_trajectories": record_trajectories,
        "measured_seconds": elapsed,
        "global_steps": steps,
        "positions": positions,
        "transitions": transitions,
        "positions_per_second": positions / elapsed,
        "transitions_per_second": transitions / elapsed,
        "worker_cpu_seconds": float(totals.get("worker_cpu_seconds", 0.0)),
        "worker_max_rss_bytes": int(totals.get("worker_max_rss_bytes", 0)),
        "decisions_recorded": int(totals.get("total_decisions_recorded", 0)),
        "record_bytes": int(totals.get("total_record_bytes", 0)),
        **_summarise_latencies(latencies),
    }


def _write_uniform_decisions(pool: WorkerPool) -> None:
    """Uniform probabilities over each active slot's legal set."""
    buffers = pool.buffers
    pool.clear_decisions()
    active = np.flatnonzero(buffers.status == STATUS_ACTIVE)
    counts = buffers.legal_count[active].astype(np.int64)
    buffers.value_prediction[active] = np.float32(1.0 / 3.0)
    width = buffers.policy_probabilities.shape[1]
    columns = np.arange(width)[None, :]
    weights = (columns < counts[:, None]).astype(np.float32)
    buffers.policy_probabilities[active] = weights / np.maximum(counts, 1)[:, None]
    buffers.decision_valid[active] = 1


# ---------------------------------------------------------------------------
# Configuration screening
# ---------------------------------------------------------------------------


def build_screening_plan(
    *,
    anchor_workers: int = 8,
    anchor_environments: int = 2048,
    anchor_batch: int = 2048,
    precision: str = "float16",
    legality: str = "dense",
) -> list[dict]:
    """The screened subset of the required 5 x 5 x 7 grid.

    The full Cartesian product is 175 points and Agents 2 and 4 already rule
    most of it out: Agent 2 measured throughput moving under 3 percent across a
    factor of eight in environments, and Agent 4 measured batch sizes above
    1,024 buying about 2 percent. Screening every point would spend most of an
    hour re-measuring known-flat axes.

    What is kept:

    - a full sweep of each required axis through the anchor point, so every
      listed worker count, environment count and inference batch size is
      measured at least once;
    - deliberate off-diagonal points where the CPU generation and Metal
      consumption sides are mismatched -- many workers with a small batch, few
      workers with a large batch -- which is where an interaction would show and
      where a per-axis sweep would miss it;
    - the float32 dense baseline and the compact-legality variant at the anchor.

    Each entry is a keyword dictionary for :class:`CoordinatorConfig`.
    """
    plan: list[dict] = []
    seen: set[tuple] = set()

    def add(workers: int, environments: int, batch: int, group: str, **overrides):
        if environments < workers:
            return
        key = (
            workers,
            environments,
            batch,
            overrides.get("precision", precision),
            overrides.get("legality", legality),
        )
        if key in seen:
            return
        seen.add(key)
        plan.append(
            {
                "num_workers": workers,
                "num_environments": environments,
                "inference_batch_size": batch,
                "precision": overrides.get("precision", precision),
                "legality": overrides.get("legality", legality),
                "group": group,
            }
        )

    # Axis sweeps through the anchor.
    for batch in INFERENCE_BATCH_SIZES:
        add(anchor_workers, anchor_environments, batch, "batch_sweep")
    for workers in WORKER_COUNTS:
        add(workers, anchor_environments, anchor_batch, "worker_sweep")
    for environments in ENVIRONMENT_COUNTS:
        add(anchor_workers, environments, min(anchor_batch, environments), "environment_sweep")

    # Off-diagonal interaction probes: a starved consumer against a starved
    # producer, and the small-environment / large-batch corner where the batch
    # cannot be filled.
    add(12, 2048, 64, "interaction")
    add(12, 512, 512, "interaction")
    add(4, 2048, 2048, "interaction")
    add(4, 256, 256, "interaction")
    add(10, 1024, 256, "interaction")
    add(10, 1536, 1536, "interaction")
    add(6, 1024, 1024, "interaction")
    add(6, 256, 1024, "interaction")

    # Baselines required for comparison.
    add(anchor_workers, anchor_environments, anchor_batch, "baseline", precision="float32")
    add(anchor_workers, anchor_environments, anchor_batch, "baseline", legality="compact")
    return plan


# ---------------------------------------------------------------------------
# Decision ratio
# ---------------------------------------------------------------------------


def decide_backend(ratio: float) -> tuple[str, bool]:
    """Apply the pre-registered decision rule to a measured ratio."""
    if ratio >= KEEP_PYTHON_RATIO:
        return DECISION_KEEP_PYTHON, False
    if ratio >= OPTIONAL_OPTIMISATION_RATIO:
        return DECISION_KEEP_PYTHON_OPTIONAL, False
    return DECISION_BUILD_BACKEND, True


def compute_ratio(
    simulation_positions_per_second: float,
    inference_positions_per_second: float,
) -> dict:
    if inference_positions_per_second <= 0:
        raise CoordinatorError("inference rate must be positive to form R")
    ratio = simulation_positions_per_second / inference_positions_per_second
    decision, required = decide_backend(ratio)
    return {
        "simulation_pipeline_positions_per_second": simulation_positions_per_second,
        "representative_model_inference_positions_per_second": (
            inference_positions_per_second
        ),
        "R": ratio,
        "backend_decision": decision,
        "optimized_backend_required": required,
        "thresholds": {
            "keep_python_at_or_above": KEEP_PYTHON_RATIO,
            "optional_optimisation_at_or_above": OPTIONAL_OPTIMISATION_RATIO,
        },
    }


# ---------------------------------------------------------------------------
# Soak
# ---------------------------------------------------------------------------


def run_soak(
    config: CoordinatorConfig,
    *,
    duration_seconds: float,
    sample_interval_seconds: float = 60.0,
    device=None,
    on_sample=None,
) -> dict:
    """Run one configuration continuously, sampling health at a fixed interval."""
    coordinator = SelfPlayCoordinator(config, device=device)
    samples: list[dict] = []
    errors: list[dict] = []
    swap_start = swap_bytes()
    started = time.perf_counter()
    coordinator.start()

    last_sample = started
    last_positions = 0
    last_games = 0
    last_sample_time = started
    steps = 0
    try:
        while True:
            now = time.perf_counter()
            if now - started >= duration_seconds:
                break
            try:
                coordinator.step()
                steps += 1
            except Exception as error:  # noqa: BLE001 - a soak failure is a result
                # The soak stops here, as it must: a correctness or worker
                # failure invalidates continuing. The samples collected so far
                # are still returned, because the trend leading up to a failure
                # is the most useful thing the soak has to say about it.
                errors.append(
                    {
                        "elapsed_seconds": time.perf_counter() - started,
                        "step": steps,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
                break

            now = time.perf_counter()
            if now - last_sample >= sample_interval_seconds:
                totals = coordinator.totals
                window_seconds = now - last_sample_time
                window_positions = totals.positions - last_positions
                window_games = coordinator.games_finished - last_games
                alive = coordinator.pool.worker_liveness()
                sample = {
                    "elapsed_seconds": now - started,
                    "global_steps": totals.steps,
                    "positions": totals.positions,
                    "games": coordinator.games_finished,
                    "resets": totals.resets,
                    "window_seconds": window_seconds,
                    "window_positions_per_second": window_positions / window_seconds,
                    "window_games_per_second": window_games / window_seconds,
                    "coordinator_rss_bytes": peak_memory_bytes(),
                    "shared_memory_bytes": coordinator.pool.buffers.nbytes,
                    "workers_alive": int(sum(alive)),
                    "workers_expected": config.num_workers,
                    "terminal_reason_counts": dict(coordinator.terminal_reason_counts),
                    **{f"mps_{k}": v for k, v in mps_memory_bytes().items()},
                    **swap_bytes(),
                }
                samples.append(sample)
                if on_sample is not None:
                    on_sample(sample)
                last_sample = now
                last_sample_time = now
                last_positions = totals.positions
                last_games = coordinator.games_finished
    finally:
        elapsed = time.perf_counter() - started
        totals = coordinator.totals
        games = coordinator.games_finished
        reasons = dict(coordinator.terminal_reason_counts)
        pool_totals = coordinator.shutdown()

    memory_growth, throughput_change = _soak_trends(samples)
    return {
        "configuration": config.as_dict(),
        "duration_seconds": elapsed,
        "global_steps": totals.steps,
        "positions": totals.positions,
        "games": games,
        "resets": totals.resets,
        "positions_per_second": totals.positions / max(elapsed, 1e-9),
        "games_per_second": games / max(elapsed, 1e-9),
        "terminal_reason_counts": reasons,
        "samples": samples,
        "errors": errors,
        "swap_start": swap_start,
        "swap_end": swap_bytes(),
        "memory_growth_bytes": memory_growth,
        "throughput_change_fraction": throughput_change,
        "peak_process_memory_bytes": peak_memory_bytes(),
        "worker_max_rss_bytes": int(pool_totals.get("worker_max_rss_bytes", 0)),
        "recording": {
            key: pool_totals.get(key, 0)
            for key in (
                "total_decisions_recorded",
                "total_games_recorded",
                "total_record_bytes",
                "total_verified_decisions",
                "total_reconstruction_mismatches",
                "total_games_joined_late",
            )
        },
        "reconstruction_mismatch_details": list(
            pool_totals.get("mismatch_details", ())
        ),
        **_summarise_latencies(totals.step_latencies),
    }


def _soak_trends(samples: list[dict]) -> tuple[int, float]:
    """Memory growth and throughput change between the first and last quarter.

    Comparing quarters rather than single samples keeps one slow interval from
    reading as a trend.
    """
    if len(samples) < 4:
        return 0, 0.0
    quarter = max(1, len(samples) // 4)
    head = samples[:quarter]
    tail = samples[-quarter:]

    def mean(rows, key):
        return statistics.fmean(row[key] for row in rows)

    memory_growth = int(
        mean(tail, "coordinator_rss_bytes") - mean(head, "coordinator_rss_bytes")
    )
    head_rate = mean(head, "window_positions_per_second")
    tail_rate = mean(tail, "window_positions_per_second")
    change = (tail_rate - head_rate) / head_rate if head_rate else 0.0
    return memory_growth, change


__all__ = [
    "BENCHMARK_VERSION",
    "DECISION_BUILD_BACKEND",
    "DECISION_KEEP_PYTHON",
    "DECISION_KEEP_PYTHON_OPTIONAL",
    "ENVIRONMENT_COUNTS",
    "INFERENCE_BATCH_SIZES",
    "WORKER_COUNTS",
    "GateReport",
    "ReferenceMirror",
    "compute_ratio",
    "decide_backend",
    "measure_configuration",
    "measure_simulation_pipeline",
    "platform_report",
    "run_integrated_gate",
    "run_reconstruction_gate",
    "run_soak",
    "swap_bytes",
]
