#!/usr/bin/env python3
"""Phase 6B: run one collection *segment* and exit.

A segment is the unit of process recycling. The supervisor
(:mod:`stratego.training.phase6b_recycle`) runs one of these per segment as a
child process, so when the segment ends the operating system reclaims every byte
the collection process held -- host heap, allocator arenas and the Metal context
alike. That is the whole point: nothing short of process exit reliably returns an
allocator arena, and Phase 6 measured the arena as the thing that grows.

Contract with the supervisor
----------------------------
- Each segment gets its own `--root-seed`, so the games it plays are *different*
  games. Reusing the base seed would make every segment replay generation 0 of
  the same environments and produce duplicate `game_id`s, which is precisely the
  "no duplicated games caused by restart" failure the follow-up gates on.
- Each segment gets its own `--run-id`, so shard filenames cannot collide across
  segments and a shard can always be traced to the segment that wrote it.
- The segment writes a JSON state file on the way out. The supervisor reads it
  to accumulate run-wide counters, so restarting does not reset the run's books.
- Shards are closed before the state file is written, so a state file implies
  every shard it counted has a manifest.

This script deliberately does no aggregation and makes no decisions. It runs,
persists, reports and exits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="shard output directory")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--segment", type=int, required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--candidate", default="C1")
    parser.add_argument("--state-out", required=True)
    parser.add_argument("--shard-target-bytes", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--sample-seconds", type=float, default=60.0)
    parser.add_argument("--no-compress", action="store_true")
    arguments = parser.parse_args()

    import psutil
    import torch

    from stratego.model.architecture_configs import candidate_config
    from stratego.training import phase6b_recording as p6b
    from stratego.training.shard_writer import DEFAULT_SHARD_TARGET_BYTES

    process = psutil.Process()
    rss_at_start = int(process.memory_info().rss)
    started_unix = time.time()

    device = torch.device("mps") if torch.backends.mps.is_available() else None
    if device is None:
        print("MPS unavailable", file=sys.stderr)
        return 1

    config = p6b.recording_configuration(
        arguments.candidate,
        output_directory=arguments.output,
        run_id=arguments.run_id,
        compress=not arguments.no_compress,
        shard_target_bytes=(
            arguments.shard_target_bytes or DEFAULT_SHARD_TARGET_BYTES
        ),
        root_seed=arguments.root_seed,
    )

    # The identity check the supervisor cannot do for us: this process is the one
    # that builds the weights, so it is the one that must confirm they are the
    # architecture the run was frozen on.
    configuration = candidate_config(arguments.candidate)

    def progress(row: dict) -> None:
        print(
            f"[segment {arguments.segment}] t={row['elapsed_seconds']:7.1f}s "
            f"step={row['global_step']:6d} pos/s={row['positions_per_second']:9.1f} "
            f"written={row['written_gib_per_hour']:5.2f} GiB/h "
            f"ratio={row['compression_ratio']:.4f} "
            f"rss={row['total_rss_bytes'] / 2**30:5.2f}G "
            f"swap={row['swap_used_bytes']}",
            flush=True,
        )

    result = p6b.run_recording_soak(
        arguments.candidate,
        config,
        seconds=arguments.seconds if arguments.seconds > 0 else 10**9,
        sample_seconds=arguments.sample_seconds,
        warmup_steps=arguments.warmup_steps,
        device=device,
        progress=progress,
        stop_after_steps=arguments.steps or None,
    )

    totals = result["recording_totals"]
    rss_at_end = int(process.memory_info().rss)
    state = {
        "segment": arguments.segment,
        "run_id": arguments.run_id,
        "candidate_id": arguments.candidate,
        "configuration_digest": configuration.digest(),
        "root_seed": arguments.root_seed,
        "pid": os.getpid(),
        "status": result["status"],
        "error": result["error"],
        "started_unix": started_unix,
        "ended_unix": time.time(),
        "seconds": result["total_seconds"],
        "samples": result["samples"],
        "sample_count": result["sample_count"],
        "failures": result["failures"],
        "rss_at_start_bytes": rss_at_start,
        "rss_at_end_bytes": rss_at_end,
        "rss_growth_bytes": rss_at_end - rss_at_start,
        "positions": int(result["samples"][-1]["positions"]) if result["samples"] else 0,
        "games": int(result["samples"][-1]["games"]) if result["samples"] else 0,
        "records_persisted": int(totals.get("total_records_persisted", 0)),
        "bytes_produced": int(totals.get("total_record_bytes", 0)),
        "bytes_written": int(totals.get("total_persisted_bytes", 0)),
        "compressed_bytes": int(totals.get("total_compressed_bytes", 0)),
        "shards_opened": int(totals.get("total_shards_opened", 0)),
        "shards_closed": int(totals.get("total_shards_closed", 0)),
        "write_errors": int(totals.get("total_write_errors", 0)),
        "verified_decisions": int(totals.get("total_verified_decisions", 0)),
        "reconstruction_mismatches": int(
            totals.get("total_reconstruction_mismatches", 0)
        ),
        "decisions_recorded": int(totals.get("total_decisions_recorded", 0)),
        # Games terminal at creation (`phase2_1_reference_1.2.0`): counted so
        # the supervisor can reconcile them against the persisted
        # zero-decision records instead of letting them vanish silently.
        "stillborn_games": int(totals.get("total_stillborn_games", 0)),
    }
    Path(arguments.state_out).write_text(json.dumps(state, indent=1, default=str) + "\n")
    print(
        f"[segment {arguments.segment}] done: status={state['status']} "
        f"positions={state['positions']:,} games={state['games']:,} "
        f"shards={state['shards_closed']} written={state['bytes_written'] / 2**30:.3f} GiB",
        flush=True,
    )
    return 0 if state["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
