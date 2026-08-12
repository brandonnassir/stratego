#!/usr/bin/env python3
"""Phase 6 Agent 6: localize the soak's resident-memory growth.

Writes

    reports/phase_6_data/agent_06_memory_localization.json

Why this exists
---------------
The one-hour soak passed its declared memory gate (+0.96% second half over first,
tolerance 2%) but showed a real, monotone host-RSS trend of roughly 190 MiB/hour
that did not decelerate within the hour. Device memory was exactly flat, so the
growth is host-side. One hour cannot separate "slow approach to a bounded
allocator envelope" from "slow leak", and a 168-hour run is long enough for the
difference to matter.

This runs the *same* topology with production recording switched off, which is
the one variable that plausibly explains it, and compares the settled slope
against the soak's. The recording-on arm is read from `agent_06_soak.json` rather
than re-measured, so this costs one short run instead of a second hour.

It changes nothing and gates nothing. It is a diagnostic that tells Phase 7 where
to look.

Usage::

    python scripts/run_phase6_agent06_memory.py
    python scripts/run_phase6_agent06_memory.py --seconds 600
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.training import phase6_soak as soak  # noqa: E402
from stratego.training.coordinator import SelfPlayCoordinator  # noqa: E402
from stratego.training.phase6_pipeline_benchmark import (  # noqa: E402
    build_pipeline_candidate,
    candidate_configuration,
)

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_6_data"

#: Samples before this are discarded: the collection path has its own cold-start
#: ramp, and the question here is the *settled* slope, not the ramp.
SETTLED_AFTER_SECONDS = 600.0


def log(message: str) -> None:
    print(f"[agent-06-memory] {message}", flush=True)


def slope_per_hour(elapsed: list[float], values: list[float]) -> float:
    if len(elapsed) < 2:
        return 0.0
    mean_x = statistics.fmean(elapsed)
    mean_y = statistics.fmean(values)
    sxx = sum((value - mean_x) ** 2 for value in elapsed)
    if sxx <= 0:
        return 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(elapsed, values))
    return (sxy / sxx) * 3600.0


def measure_without_recording(candidate_id: str, *, seconds: float, device) -> dict:
    """The soak topology with `record_trajectories=False`, sampled the same way."""
    config = candidate_configuration(
        candidate_id,
        workers=soak.SOAK_WORKERS,
        environments=soak.SOAK_ENVIRONMENTS,
        inference_batch_size=soak.SOAK_INFERENCE_BATCH,
        precision=soak.SOAK_PRECISION,
        legality=soak.SOAK_LEGALITY,
        record_trajectories=False,
        detailed_timing=False,
        root_seed=60_006,
    )
    coordinator = SelfPlayCoordinator(
        config,
        device=device,
        model=build_pipeline_candidate(candidate_id),
        model_label=candidate_id,
    )
    samples: list[dict] = []
    coordinator.start()
    started = time.perf_counter()
    next_sample_at = 60.0
    try:
        while True:
            coordinator.step()
            elapsed = time.perf_counter() - started
            if elapsed < next_sample_at and elapsed < seconds:
                continue
            memory = soak.process_memory()
            row = {
                "elapsed_seconds": elapsed,
                "global_step": coordinator.step_index,
                "positions_per_second": coordinator.totals.positions / elapsed,
                **memory,
            }
            samples.append(row)
            log(
                f"  t={elapsed:6.1f}s step={coordinator.step_index:6d} "
                f"coord={memory['coordinator_rss_bytes'] / 2**30:5.3f}G "
                f"wkr={memory['worker_rss_bytes'] / 2**30:5.3f}G "
                f"tot={memory['total_rss_bytes'] / 2**30:5.3f}G"
            )
            next_sample_at += 60.0
            if elapsed >= seconds:
                break
    finally:
        coordinator.shutdown()

    settled = [row for row in samples if row["elapsed_seconds"] >= SETTLED_AFTER_SECONDS]
    if len(settled) < 3:
        settled = samples[len(samples) // 2 :]
    elapsed = [row["elapsed_seconds"] for row in settled]
    return {
        "configuration": config.as_dict(),
        "samples": samples,
        "settled_after_seconds": SETTLED_AFTER_SECONDS,
        "settled_samples": len(settled),
        "settled_window_seconds": elapsed[-1] - elapsed[0],
        "slopes_bytes_per_hour": {
            key: slope_per_hour(elapsed, [row[key] for row in settled])
            for key in (
                "coordinator_rss_bytes",
                "worker_rss_bytes",
                "total_rss_bytes",
            )
        },
        "first_settled": {
            key: settled[0][key]
            for key in ("coordinator_rss_bytes", "worker_rss_bytes", "total_rss_bytes")
        },
        "last_settled": {
            key: settled[-1][key]
            for key in ("coordinator_rss_bytes", "worker_rss_bytes", "total_rss_bytes")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=1200.0)
    parser.add_argument("--candidate", default=None)
    arguments = parser.parse_args()

    if not torch.backends.mps.is_available():
        log("MPS is not available")
        return 1
    device = torch.device("mps")

    soak_payload = json.loads((DATA_DIRECTORY / "agent_06_soak.json").read_text())
    recorded = soak_payload["soak"]
    candidate_id = arguments.candidate or recorded["candidate_id"]

    log(f"recording-on arm: read from agent_06_soak.json ({candidate_id})")
    with_recording = {
        "source": "agent_06_soak.json, the one-hour production soak",
        "measured_seconds": recorded["steady_state"]["window_seconds"],
        "slopes_bytes_per_hour": {
            key: report["slope_per_hour"]
            for key, report in recorded["memory_growth"].items()
            if key in ("coordinator_rss_bytes", "worker_rss_bytes", "total_rss_bytes")
        },
        "half_over_half": {
            key: report["relative_change"]
            for key, report in recorded["memory_growth"].items()
        },
    }

    log(f"recording-off arm: measuring {arguments.seconds:.0f}s")
    without_recording = measure_without_recording(
        candidate_id, seconds=arguments.seconds, device=device
    )

    on = with_recording["slopes_bytes_per_hour"]
    off = without_recording["slopes_bytes_per_hour"]
    attribution = {
        key: {
            "with_recording_mib_per_hour": on.get(key, 0.0) / 2**20,
            "without_recording_mib_per_hour": off.get(key, 0.0) / 2**20,
            "attributable_to_recording_mib_per_hour": (
                on.get(key, 0.0) - off.get(key, 0.0)
            )
            / 2**20,
        }
        for key in ("coordinator_rss_bytes", "worker_rss_bytes", "total_rss_bytes")
    }

    payload = {
        "agent": "agent_06",
        "phase": "phase_6",
        "probe": "memory_growth_localization",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "candidate_id": candidate_id,
        "question": (
            "The one-hour soak passed its memory gate but showed a monotone host "
            "RSS trend that did not decelerate. Device memory was exactly flat. "
            "Which part of the pipeline does the growth belong to?"
        ),
        "method": (
            "Same topology, same candidate, same seed, with production recording "
            "switched off, sampled the same way; the settled slope is compared "
            "against the soak's. The recording-on arm is read from the soak "
            "artifact rather than re-measured."
        ),
        "with_recording": with_recording,
        "without_recording": without_recording,
        "attribution_mib_per_hour": attribution,
        "gates_nothing": True,
    }
    path = DATA_DIRECTORY / "agent_06_memory_localization.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")

    log("settled slopes (MiB/hour):")
    for key, row in attribution.items():
        log(
            f"  {key:24s} recording on {row['with_recording_mib_per_hour']:+7.1f}  "
            f"off {row['without_recording_mib_per_hour']:+7.1f}  "
            f"attributable {row['attributable_to_recording_mib_per_hour']:+7.1f}"
        )
    log(f"wrote {path.relative_to(REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
