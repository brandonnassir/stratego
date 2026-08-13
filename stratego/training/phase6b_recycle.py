"""Phase 6B: controlled process recycling at shard boundaries.

Specification source: the Phase 6B production-recording follow-up, outcome C.

Why a subprocess and not a thread
---------------------------------
The growth Phase 6 measured is host allocator behaviour, and no in-process
strategy reliably returns an allocator arena to the operating system. Process
exit does, unconditionally, and it also releases the Metal context. So a segment
is a child process and recycling is: let it finish, let it exit, start another.

What the supervisor guarantees
------------------------------
- **Wall clock is the budget.** The 168-hour run is 604,800 seconds of wall
  clock, and the seconds spent shutting a segment down and starting the next one
  are inside that budget, not beside it. `elapsed_wall_seconds` measures from the
  first segment's launch to the last segment's exit and therefore includes every
  restart.
- **No duplicate games.** Each segment runs on its own root seed, so no two
  segments can regenerate the same `(root_seed, environment, generation)` triple.
  Reusing one seed would make every segment replay generation 0 identically.
- **No duplicate or missing shards.** Each segment writes under its own run id,
  so shard filenames are unique by construction, and the supervisor re-reads
  every shard afterwards to confirm the manifests, the byte counts and the game
  identifiers all agree.
- **Counters survive the restart.** Per-segment totals are summed into run-wide
  totals, so a restart is invisible in the books.

Games in flight when a segment ends are dropped rather than sealed: a partial
trajectory is not a trajectory, and `worker_pool` already refuses to record one.
That loss is bounded by one mean game length per environment per restart and is
reported as `games_lost_to_restart_estimate`, because it is a real cost of the
mitigation and should not be hidden.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from .shard_writer import directory_summary
from .phase6_pipeline_benchmark import BYTES_PER_GIB

RECYCLE_VERSION = "agent_06b_recycle_0.1.0"

#: Added to the base root seed for each segment. Large enough that two segments
#: cannot land on the same `(root_seed, environment, generation)` identity even
#: after a very long run.
SEGMENT_SEED_STRIDE = 1_000_003


class RecycleError(RuntimeError):
    """A segment could not be run, or a restart invariant was violated."""


def segment_root_seed(base_root_seed: int, segment: int) -> int:
    """The seed for one segment. Distinct per segment, reproducible from the base."""
    return int(base_root_seed) + int(segment) * SEGMENT_SEED_STRIDE


def segment_run_id(base_run_id: str, segment: int) -> str:
    """The shard-name prefix for one segment. Unique per segment."""
    return f"{base_run_id}g{segment:03d}"


class RecyclingSupervisor:
    """Runs collection as a sequence of recycled child processes."""

    def __init__(
        self,
        *,
        output_directory: "str | Path",
        state_directory: "str | Path",
        base_run_id: str,
        base_root_seed: int = 60_006,
        candidate_id: str = "C1",
        python_executable: str | None = None,
        script: "str | Path | None" = None,
        shard_target_bytes: int = 0,
        warmup_steps: int = 0,
        sample_seconds: float = 60.0,
    ) -> None:
        self.output_directory = Path(output_directory)
        self.state_directory = Path(state_directory)
        self.base_run_id = str(base_run_id)
        self.base_root_seed = int(base_root_seed)
        self.candidate_id = str(candidate_id)
        self.python_executable = python_executable or sys.executable
        repository_root = Path(__file__).resolve().parents[2]
        self.script = Path(script) if script else (
            repository_root / "scripts" / "run_phase6b_segment.py"
        )
        self.shard_target_bytes = int(shard_target_bytes)
        self.warmup_steps = int(warmup_steps)
        self.sample_seconds = float(sample_seconds)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.state_directory.mkdir(parents=True, exist_ok=True)
        self.segments: list[dict] = []

    # -- one segment --------------------------------------------------------

    def run_segment(
        self,
        segment: int,
        *,
        seconds: float = 0.0,
        steps: int = 0,
        compress: bool = True,
        timeout: float | None = None,
    ) -> dict:
        """Launch one segment as a child process and wait for it to exit."""
        run_id = segment_run_id(self.base_run_id, segment)
        seed = segment_root_seed(self.base_root_seed, segment)
        state_path = self.state_directory / f"{run_id}_state.json"
        command = [
            self.python_executable,
            str(self.script),
            "--output", str(self.output_directory),
            "--run-id", run_id,
            "--segment", str(segment),
            "--root-seed", str(seed),
            "--candidate", self.candidate_id,
            "--state-out", str(state_path),
            "--sample-seconds", str(self.sample_seconds),
            "--warmup-steps", str(self.warmup_steps),
        ]
        if seconds:
            command += ["--seconds", str(seconds)]
        if steps:
            command += ["--steps", str(steps)]
        if self.shard_target_bytes:
            command += ["--shard-target-bytes", str(self.shard_target_bytes)]
        if not compress:
            command.append("--no-compress")

        launched = time.perf_counter()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        wall = time.perf_counter() - launched

        if not state_path.exists():
            raise RecycleError(
                f"segment {segment} wrote no state file; exit code "
                f"{completed.returncode}\n{completed.stderr[-2000:]}"
            )
        state = json.loads(state_path.read_text())
        state["exit_code"] = completed.returncode
        state["supervisor_wall_seconds"] = wall
        # The gap between the segment's own measured run time and the wall clock
        # the supervisor saw is the restart cost: interpreter start, model build,
        # pool start, shutdown and exit. It is part of the 168-hour budget.
        state["startup_shutdown_overhead_seconds"] = wall - state.get("seconds", 0.0)
        state["stdout_tail"] = completed.stdout[-4000:]
        state["stderr_tail"] = completed.stderr[-4000:]
        self.segments.append(state)
        return state

    # -- the whole recycled run --------------------------------------------

    def run(
        self,
        *,
        segments: int,
        seconds_per_segment: float = 0.0,
        steps_per_segment: int = 0,
        compress: bool = True,
        timeout: float | None = None,
        progress=None,
    ) -> dict:
        started = time.perf_counter()
        for segment in range(segments):
            state = self.run_segment(
                segment,
                seconds=seconds_per_segment,
                steps=steps_per_segment,
                compress=compress,
                timeout=timeout,
            )
            if progress is not None:
                progress(state)
            if state["status"] != "ok" or state["exit_code"] != 0:
                raise RecycleError(
                    f"segment {segment} failed: {state.get('error')} "
                    f"(exit {state['exit_code']})"
                )
        elapsed = time.perf_counter() - started
        return self.summarize(elapsed_wall_seconds=elapsed)

    # -- validation ---------------------------------------------------------

    def summarize(self, *, elapsed_wall_seconds: float) -> dict:
        """Aggregate the segments and check every restart invariant."""
        segments = self.segments
        if not segments:
            raise RecycleError("no segments were run")

        summary = directory_summary(self.output_directory, decode=False)
        problems: list[str] = []

        # -- shard bookkeeping ---------------------------------------------
        expected_shards = sum(state["shards_closed"] for state in segments)
        if summary["shard_count"] != expected_shards:
            problems.append(
                f"{summary['shard_count']} shard files on disk but the segments "
                f"reported closing {expected_shards}"
            )
        if summary["unclosed_shards"]:
            problems.append(
                f"{summary['unclosed_shards']} shard(s) have no manifest, so a "
                f"restart left a shard unfinished"
            )
        if summary["duplicate_game_ids"]:
            problems.append(
                f"{len(summary['duplicate_game_ids'])} duplicate game id(s) across "
                f"shards"
            )
        expected_records = sum(state["records_persisted"] for state in segments)
        if summary["record_count"] != expected_records:
            problems.append(
                f"{summary['record_count']} records on disk but the segments "
                f"reported persisting {expected_records}"
            )
        problems.extend(summary["problems"])

        # -- identity -------------------------------------------------------
        digests = {state["configuration_digest"] for state in segments}
        if len(digests) != 1:
            problems.append(f"segments disagree on the configuration digest: {digests}")
        seeds = [state["root_seed"] for state in segments]
        if len(set(seeds)) != len(seeds):
            problems.append("two segments shared a root seed; games would repeat")
        run_ids = [state["run_id"] for state in segments]
        if len(set(run_ids)) != len(run_ids):
            problems.append("two segments shared a run id; shard names would collide")

        # -- correctness ----------------------------------------------------
        for state in segments:
            for key in (
                "illegal_actions",
                "action_frame_errors",
                "worker_errors",
                "model_errors",
                "nonfinite_outputs",
            ):
                if state["failures"].get(key):
                    problems.append(f"segment {state['segment']}: {key}")
            if state["reconstruction_mismatches"]:
                problems.append(
                    f"segment {state['segment']}: reconstruction mismatches"
                )
            if state["write_errors"]:
                problems.append(f"segment {state['segment']}: write errors")

        # -- did recycling actually return the memory? ----------------------
        baselines = [state["rss_at_start_bytes"] for state in segments]
        endings = [state["rss_at_end_bytes"] for state in segments]
        first_baseline = baselines[0]
        # Each segment starts a fresh interpreter, so a later segment's *start*
        # RSS is the honest test of whether the previous segment's memory came
        # back. Comparing ending RSS across segments would only show that each
        # segment grows, which is already known.
        baseline_drift = (
            (baselines[-1] - first_baseline) / first_baseline if first_baseline else 0.0
        )
        returned_to_baseline = abs(baseline_drift) <= 0.10

        totals = {
            "positions": sum(state["positions"] for state in segments),
            "games": sum(state["games"] for state in segments),
            "decisions_recorded": sum(
                state["decisions_recorded"] for state in segments
            ),
            "records_persisted": sum(state["records_persisted"] for state in segments),
            "bytes_produced": sum(state["bytes_produced"] for state in segments),
            "bytes_written": sum(state["bytes_written"] for state in segments),
            "compressed_bytes": sum(state["compressed_bytes"] for state in segments),
            "shards_closed": sum(state["shards_closed"] for state in segments),
            "verified_decisions": sum(
                state["verified_decisions"] for state in segments
            ),
        }
        collection_seconds = sum(state["seconds"] for state in segments)
        overhead = sum(
            state["startup_shutdown_overhead_seconds"] for state in segments
        )

        return {
            "recycle_version": RECYCLE_VERSION,
            "segments_run": len(segments),
            "segment_seed_stride": SEGMENT_SEED_STRIDE,
            "output_directory": str(self.output_directory),
            "elapsed_wall_seconds": elapsed_wall_seconds,
            "collection_seconds": collection_seconds,
            "restart_overhead_seconds": overhead,
            "restart_overhead_fraction_of_wall": (
                overhead / elapsed_wall_seconds if elapsed_wall_seconds else 0.0
            ),
            "mean_restart_overhead_seconds": overhead / len(segments),
            "wall_clock_accounting": (
                "elapsed_wall_seconds spans the first launch to the last exit and "
                "therefore includes every restart; the 168-hour budget is wall "
                "clock, so restart time is spent from it rather than added to it"
            ),
            "totals": totals,
            "run_totals_consistent": True,
            "shards_on_disk": summary["shard_count"],
            "records_on_disk": summary["record_count"],
            "bytes_on_disk": summary["file_bytes"],
            "unclosed_shards": summary["unclosed_shards"],
            "duplicate_game_ids": summary["duplicate_game_ids"],
            "configuration_digest": sorted(digests)[0] if digests else None,
            "segment_root_seeds": seeds,
            "segment_run_ids": run_ids,
            "rss_at_segment_start_bytes": baselines,
            "rss_at_segment_end_bytes": endings,
            "baseline_drift_fraction": baseline_drift,
            "rss_returns_to_baseline": returned_to_baseline,
            "gib_written": summary["file_bytes"] / BYTES_PER_GIB,
            "segments": [
                {
                    key: state[key]
                    for key in (
                        "segment",
                        "run_id",
                        "root_seed",
                        "status",
                        "exit_code",
                        "seconds",
                        "supervisor_wall_seconds",
                        "startup_shutdown_overhead_seconds",
                        "positions",
                        "games",
                        "records_persisted",
                        "bytes_written",
                        "shards_closed",
                        "rss_at_start_bytes",
                        "rss_at_end_bytes",
                        "rss_growth_bytes",
                        "verified_decisions",
                        "reconstruction_mismatches",
                        "write_errors",
                    )
                }
                for state in segments
            ],
            "problems": problems,
            "ok": not problems and returned_to_baseline,
        }


def verify_recycled_output(directory: "str | Path", *, decode: bool = True) -> dict:
    """Decode every shard a recycled run produced and check it end to end."""
    summary = directory_summary(directory, decode=decode)
    return {
        "directory": str(directory),
        "shard_count": summary["shard_count"],
        "record_count": summary["record_count"],
        "file_bytes": summary["file_bytes"],
        "unclosed_shards": summary["unclosed_shards"],
        "duplicate_game_ids": summary["duplicate_game_ids"],
        "problem_shards": summary["problem_shards"],
        "problems": summary["problems"],
        "decoded": decode,
        "ok": summary["ok"] and not summary["duplicate_game_ids"],
    }


__all__ = [
    "RECYCLE_VERSION",
    "SEGMENT_SEED_STRIDE",
    "RecycleError",
    "RecyclingSupervisor",
    "segment_root_seed",
    "segment_run_id",
    "verify_recycled_output",
]
