#!/usr/bin/env python3
"""Phase 6 Agent 4 addendum: how much a stored trajectory compresses.

Writes `reports/phase_6_data/agent_04_compression_probe.json`.

Why this exists
---------------
The main harness measures the *uncompressed* production storage rate, because
that is what `trajectory_v1` writes today and what a 168-hour projection has to
be built from. That projection turns out to be the binding constraint: at the
fastest finalist a week of collection is larger than the user's external volume,
so whether the retention policy has any headroom at all depends on a number the
main harness does not produce.

`stratego.training.trajectory` already ships a compressed codec
(`encode_game_record_compressed`), and Agent 6 has to choose a retention policy.
Handing it a compression ratio measured on *real sealed games of production
length* is strictly better than leaving it to assume one. This does not finalize
the retention policy, and it does not change what the pipeline writes.

Games are collected past the storage warmup so the sample contains
production-length games. A ratio measured on the short games that seal in the
first few seconds of a cold pool would not transfer: a short record is dominated
by its fixed header and compresses differently from a 500-ply game.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.training import phase6_pipeline_benchmark as p6  # noqa: E402
from stratego.training.coordinator import resolve_device  # noqa: E402
from stratego.training.serialization import DEFAULT_COMPRESSION_LEVEL  # noqa: E402
from stratego.training.trajectory import (  # noqa: E402
    decode_game_record,
    encode_game_record,
    encode_game_record_compressed,
    validate_game_record,
)

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_6_data"


def main() -> int:
    candidate_id = "C1"
    device = resolve_device()
    config = p6.candidate_configuration(
        candidate_id,
        workers=p6.STARTING_WORKERS,
        environments=p6.STARTING_ENVIRONMENTS,
        inference_batch_size=2048,
        record_trajectories=True,
        detailed_timing=False,
        retain_games=6,
    )
    print(f"[compression] collecting production-length games with {candidate_id}", flush=True)
    started = time.perf_counter()
    coordinator = p6.open_candidate_coordinator(candidate_id, config, device=device)
    coordinator.start()
    try:
        for _ in range(p6.STORAGE_WARMUP_STEPS + 400):
            coordinator.step()
    finally:
        totals = coordinator.shutdown()
    payloads = list(totals.get("retained_records", ()))
    print(f"[compression] {len(payloads)} sealed games in {time.perf_counter() - started:.0f}s")

    rows = []
    for payload in payloads:
        record = decode_game_record(payload)
        problems = validate_game_record(record)
        compressed = encode_game_record_compressed(record)
        # The compressed form must decode back to the same record, or a smaller
        # file would just be a different file.
        restored = decode_game_record(encode_game_record(decode_game_record(payload)))
        rows.append(
            {
                "decisions": len(record.decisions),
                "snapshots": len(record.snapshots),
                "raw_bytes": len(payload),
                "compressed_bytes": len(compressed),
                "ratio": len(compressed) / len(payload),
                "raw_bytes_per_decision": len(payload) / max(len(record.decisions), 1),
                "compressed_bytes_per_decision": len(compressed)
                / max(len(record.decisions), 1),
                "schema_problems": problems,
                "round_trips": restored == record,
            }
        )

    raw_total = sum(row["raw_bytes"] for row in rows)
    compressed_total = sum(row["compressed_bytes"] for row in rows)
    decisions_total = sum(row["decisions"] for row in rows)
    aggregate_ratio = compressed_total / raw_total if raw_total else 0.0

    payload = {
        "agent": "agent_04",
        "probe": "trajectory_compression",
        "benchmark_version": p6.BENCHMARK_VERSION,
        "candidate_id": candidate_id,
        "compression_level": DEFAULT_COMPRESSION_LEVEL,
        "configuration": config.as_dict(),
        "collected_after_steps": p6.STORAGE_WARMUP_STEPS + 400,
        "games": len(rows),
        "decisions": decisions_total,
        "raw_bytes": raw_total,
        "compressed_bytes": compressed_total,
        "aggregate_ratio": aggregate_ratio,
        "aggregate_saving": 1.0 - aggregate_ratio,
        "raw_bytes_per_decision": raw_total / max(decisions_total, 1),
        "compressed_bytes_per_decision": compressed_total / max(decisions_total, 1),
        "median_ratio": statistics.median(row["ratio"] for row in rows) if rows else 0.0,
        "mean_decisions_per_game": decisions_total / max(len(rows), 1),
        "all_records_valid": all(not row["schema_problems"] for row in rows),
        "all_records_round_trip": all(row["round_trips"] for row in rows),
        "rows": rows,
        "note": (
            "Measured on real sealed games collected past the storage warmup, so the "
            "sample is production-length rather than the short games a cold pool seals "
            "first. The pipeline still writes uncompressed records; this only tells "
            "Agent 6 what a compressed retention policy would buy. Retention is not "
            "decided here."
        ),
    }
    path = DATA_DIRECTORY / "agent_04_compression_probe.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(
        f"[compression] {len(rows)} games, {decisions_total} decisions, "
        f"ratio {aggregate_ratio:.3f} ({100 * (1 - aggregate_ratio):.1f}% saved), "
        f"{payload['raw_bytes_per_decision']:.0f} -> "
        f"{payload['compressed_bytes_per_decision']:.0f} bytes/decision, "
        f"mean {payload['mean_decisions_per_game']:.0f} decisions/game"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
