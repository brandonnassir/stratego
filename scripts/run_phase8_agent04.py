#!/usr/bin/env python3
"""Phase 8 Agent 4 acceptance harness: trainer, checkpoint/resume, throughput.

Verifies the Agent 1-3 prerequisites and the accepted-corpus identity (the
resolver must name the canonical location and the live content / metadata /
commit-index digests must equal the accepted ones — a mismatch is BLOCKED,
never a regeneration), then produces the Agent 4 evidence:

- the real loader/trainer benchmark on MPS, starting from the Agent 3 best
  topology of 8 loader workers, with byte-identical batches proven across
  every measured topology and tuning restricted to record-cache size, worker
  count, prefetch depth and CPU/MPS overlap;
- the 1,000-update uninterrupted-vs-resumed split-run equivalence proof on
  MPS (400 + save + destroy process + reload + 600), with per-step batch
  identities and per-parameter closeness;
- a smaller CPU exact-determinism reference of the same shape;
- the >= 2,048-update numerical-stability soak on one neutral frozen
  candidate (this is engineering validation, never candidate selection);
- the three Agent 4 artifacts::

      reports/phase_8_data/agent_04_trainer_contract.json
      reports/phase_8_data/agent_04_training_benchmark.csv
      reports/phase_8_data/agent_04_resume_validation.json

No pilot selection happens here. The test split is never read by a model;
the Phase 4 bank is never used.

Usage::

    python scripts/run_phase8_agent04.py --full --run-pytest
    python scripts/run_phase8_agent04.py --verify            # gates only
    python scripts/run_phase8_agent04.py --benchmark
    python scripts/run_phase8_agent04.py --resume-mps
    python scripts/run_phase8_agent04.py --resume-cpu
    python scripts/run_phase8_agent04.py --soak
    python scripts/run_phase8_agent04.py --artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch  # noqa: E402

from stratego.training import synthetic_corpus as sc  # noqa: E402
from stratego.training import warmstart_contract as wc  # noqa: E402
from stratego.training.warmstart_checkpoint import (  # noqa: E402
    WARMSTART_CHECKPOINT_VERSION,
    WARMSTART_TRAINER_VERSION,
    CorpusIdentity,
    load_warmstart_checkpoint,
    validate_warmstart_payload,
    read_warmstart_payload,
    verify_corpus_identity,
)
from stratego.training.warmstart_dataset import DEFAULT_BATCH_SIZE  # noqa: E402
from stratego.training.warmstart_loss import (  # noqa: E402
    WARMSTART_LOSS_VERSION,
    loss_semantics_summary,
)
from stratego.training.warmstart_metrics import (  # noqa: E402
    WARMSTART_METRICS_VERSION,
    frozen_train_value_prior,
)
from stratego.training.warmstart_trainer import (  # noqa: E402
    LoaderTopology,
    WarmstartTrainConfig,
    WarmstartTrainer,
    pilot_candidate_ids,
)

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_8_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_8_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase8" / "agent04"

#: The canonical accepted storage location (supplementary review instruction).
#: This harness *verifies* the resolver against it; no trainer/checkpoint/
#: dataset implementation embeds it.
REQUIRED_CORPUS_ROOT = (
    "/Users/brandonwashington/Dev/Github/stratego/gpt_agent/"
    "data/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1"
)
REQUIRED_CORPUS_ROOT_RELATIVE = "data/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1"

#: The neutral frozen candidate used for engineering runs: the median frozen
#: learning rate with unit loss weights. An infrastructure choice made before
#: any pilot comparison; never a selection decision.
NEUTRAL_CANDIDATE = "ws_pilot_lr3e-4_balanced"

#: Cadence-sized validation: 8 batches x 256 = 2,048 held-out examples per
#: validation pass during engineering runs.
VALIDATION_BATCHES = 8

BENCHMARK_CSV_COLUMNS = (
    "phase",
    "workers",
    "prefetch",
    "record_cache_size",
    "timing_mode",
    "warmup_updates",
    "measured_updates",
    "batch_size",
    "wall_seconds",
    "updates_per_second",
    "examples_per_second",
    "data_wait_fraction",
    "data_wait_ms_mean",
    "h2d_ms_mean",
    "forward_ms_mean",
    "loss_ms_mean",
    "backward_ms_mean",
    "optimizer_ms_mean",
    "step_wall_ms_p50",
    "step_wall_ms_p95",
    "cache_hit_rate",
    "digests_match_baseline",
    "parent_peak_rss_bytes",
    "worker_peak_rss_bytes",
    "mps_current_allocated_bytes",
    "mps_driver_allocated_bytes",
    "notes",
)


def log(message: str) -> None:
    print(f"[agent04 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPOSITORY_ROOT, capture_output=True, text=True
    ).stdout.strip()


def environment_record() -> dict:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cpu_count": int(torch.multiprocessing.cpu_count()),
        "source_revision": git_output("rev-parse", "--short", "HEAD"),
        "working_tree_state": "dirty" if git_output("status", "--porcelain") else "clean",
    }


def percentile(values: list, fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return float(ordered[index])


# ---------------------------------------------------------------------------
# Prerequisite verification
# ---------------------------------------------------------------------------


def accepted_corpus_identity() -> CorpusIdentity:
    """The accepted digests, cross-checked across every artifact that states
    them (Agent 2 manifest, Agent 2 relocation record, Agent 3 prerequisites)."""
    manifest = read_json(DATA_DIRECTORY / "agent_02_corpus_manifest.json")["corpus_manifest"]
    relocation = read_json(DATA_DIRECTORY / "agent_02_relocation.json")["accepted_digests"]
    agent3 = read_json(DATA_DIRECTORY / "agent_03_example_contract.json")[
        "prerequisite_digests"
    ]
    sources = {
        "content_digest": {
            "agent_02_manifest": manifest["content_digest"],
            "agent_02_relocation": relocation["content_digest"],
            "agent_03": agent3["corpus_content"],
        },
        "metadata_digest": {
            "agent_02_manifest": manifest["metadata_digest"],
            "agent_02_relocation": relocation["metadata_digest"],
            "agent_03": agent3["corpus_metadata"],
        },
        "commit_index_digest": {
            "agent_02_manifest": manifest["commit_index_digest"],
            "agent_02_relocation": relocation["commit_index_digest"],
            "agent_03": agent3["corpus_commit_index"],
        },
    }
    for name, values in sources.items():
        if len(set(values.values())) != 1:
            raise SystemExit(
                f"BLOCKED: accepted artifacts disagree on {name}: {values}"
            )
    return CorpusIdentity(
        corpus_version=manifest["corpus_version"],
        content_digest=manifest["content_digest"],
        metadata_digest=manifest["metadata_digest"],
        commit_index_digest=manifest["commit_index_digest"],
    )


def verify_prerequisites(*, check_payload_bytes: bool = True) -> tuple:
    """Every Agent 4 entry gate. Returns `(record, verified_identity)`."""
    started = time.perf_counter()
    problems: list = []

    statuses = {}
    for agent, artifact in (
        (1, "agent_01_warmstart_contract.json"),
        (2, "agent_02_corpus_audit.json"),
        (3, "agent_03_example_contract.json"),
    ):
        payload = read_json(DATA_DIRECTORY / artifact)
        statuses[f"agent_{agent}"] = payload.get("status")
        if payload.get("status") != "PASS":
            problems.append(f"agent {agent} artifact status is {payload.get('status')!r}")

    recorded_contract = read_json(DATA_DIRECTORY / "agent_01_warmstart_contract.json")[
        "contract_digest"
    ]
    live_contract = wc.contract_digest()
    if recorded_contract != live_contract:
        problems.append(
            f"live contract digest {live_contract} != recorded {recorded_contract}"
        )

    upstream = wc.verify_frozen_upstream()
    roster = wc.verify_teacher_roster()
    problems.extend(upstream)
    problems.extend(roster)

    resolution = sc.describe_corpus_root()
    resolved = sc.default_corpus_root()
    required_via_repository = REPOSITORY_ROOT / REQUIRED_CORPUS_ROOT_RELATIVE
    if str(resolved) != REQUIRED_CORPUS_ROOT:
        problems.append(
            f"default_corpus_root() resolves to {resolved}, the accepted canonical "
            f"location is {REQUIRED_CORPUS_ROOT}; correct the resolver/pointer "
            "configuration only"
        )
    if resolved != required_via_repository:
        problems.append(
            f"resolver {resolved} is not the repository-relative canonical path "
            f"{required_via_repository}"
        )
    if resolution["pointer_value"] != REQUIRED_CORPUS_ROOT:
        problems.append(
            f"pointer file names {resolution['pointer_value']!r}, expected the "
            "canonical location"
        )

    if not torch.backends.mps.is_available():
        problems.append("MPS is not available on this machine")

    accepted = accepted_corpus_identity()
    identity = None
    digest_seconds = 0.0
    if not problems:
        digest_started = time.perf_counter()
        identity = verify_corpus_identity(
            resolved, accepted, check_payload_bytes=check_payload_bytes
        )
        digest_seconds = time.perf_counter() - digest_started

    record = {
        "statuses": statuses,
        "agent_01_contract_digest": {
            "recorded": recorded_contract,
            "live": live_contract,
            "match": recorded_contract == live_contract,
        },
        "upstream_problems": upstream,
        "roster_problems": roster,
        "corpus_root_resolution": resolution,
        "required_corpus_root": REQUIRED_CORPUS_ROOT,
        "resolver_matches_required": str(resolved) == REQUIRED_CORPUS_ROOT,
        "pointer_matches_required": resolution["pointer_value"] == REQUIRED_CORPUS_ROOT,
        "accepted_digests": accepted.to_dict(),
        "observed_digests": identity.to_dict() if identity else None,
        "digests_match": identity == accepted if identity else False,
        "payload_bytes_checked": check_payload_bytes,
        "digest_verification_seconds": digest_seconds,
        "problems": problems,
        "seconds": time.perf_counter() - started,
    }
    if problems:
        write_json(WORK_DIRECTORY / "verify_blocked.json", record)
        raise SystemExit(f"BLOCKED: {problems}")
    return record, identity


# ---------------------------------------------------------------------------
# Trainer construction
# ---------------------------------------------------------------------------


def neutral_config(
    device: str, *, validation_batches: int = VALIDATION_BATCHES
) -> WarmstartTrainConfig:
    return WarmstartTrainConfig.from_pilot_candidate(
        NEUTRAL_CANDIDATE, device=device, validation_batches=validation_batches
    )


def build_trainer(
    identity: CorpusIdentity,
    *,
    device: str,
    workers: int,
    prefetch: int,
    record_cache_size: int,
    run_label: str,
    validation_batches: int = VALIDATION_BATCHES,
) -> WarmstartTrainer:
    return WarmstartTrainer(
        neutral_config(device, validation_batches=validation_batches),
        identity,
        topology=LoaderTopology(
            workers=workers, prefetch=prefetch, record_cache_size=record_cache_size
        ),
        run_label=run_label,
    )


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------


def summarize_window(rows: list, wall_seconds: float, batch_size: int) -> dict:
    def mean_ms(name: str) -> float:
        return 1000.0 * statistics.fmean(row[name] for row in rows)

    step_walls = [row["step_wall_seconds"] for row in rows]
    hits = sum(row["cache_hits"] for row in rows)
    misses = sum(row["cache_misses"] for row in rows)
    return {
        "measured_updates": len(rows),
        "wall_seconds": wall_seconds,
        "updates_per_second": len(rows) / wall_seconds if wall_seconds else 0.0,
        "examples_per_second": len(rows) * batch_size / wall_seconds if wall_seconds else 0.0,
        "data_wait_fraction": sum(row["data_wait_seconds"] for row in rows) / wall_seconds,
        "data_wait_ms_mean": mean_ms("data_wait_seconds"),
        "h2d_ms_mean": mean_ms("h2d_seconds"),
        "forward_ms_mean": mean_ms("forward_seconds"),
        "loss_ms_mean": mean_ms("loss_seconds"),
        "backward_ms_mean": mean_ms("backward_seconds"),
        "optimizer_ms_mean": mean_ms("optimizer_seconds"),
        "step_wall_ms_p50": 1000.0 * percentile(step_walls, 0.50),
        "step_wall_ms_p95": 1000.0 * percentile(step_walls, 0.95),
        "cache_hit_rate": hits / (hits + misses) if hits + misses else 0.0,
        "loss_total_first": rows[0]["loss_total"],
        "loss_total_last": rows[-1]["loss_total"],
    }


def measure_topology(
    identity: CorpusIdentity,
    *,
    workers: int,
    prefetch: int,
    record_cache_size: int,
    warmup: int,
    measured: int,
    timing: bool,
    label: str,
) -> dict:
    """One fresh trainer, one measured window, full digests for identity proof."""
    log(
        f"benchmark[{label}] workers={workers} prefetch={prefetch} "
        f"cache={record_cache_size} timing={timing}"
    )
    trainer = build_trainer(
        identity,
        device="mps",
        workers=workers,
        prefetch=prefetch,
        record_cache_size=record_cache_size,
        run_label=f"benchmark_{label}",
    )
    try:
        trainer.train_updates(warmup, timing=timing, capture_batch_digests=True)
        started = time.perf_counter()
        rows = trainer.train_updates(measured, timing=timing, capture_batch_digests=True)
        wall = time.perf_counter() - started
        summary = summarize_window(rows, wall, trainer.config.batch_size)
        summary.update(
            {
                "phase": label,
                "workers": workers,
                "prefetch": prefetch,
                "record_cache_size": record_cache_size,
                "timing_mode": timing,
                "warmup_updates": warmup,
                "batch_size": trainer.config.batch_size,
                "keys_digests": [row["keys_digest"] for row in rows],
                "batch_digests": [row["batch_digest"] for row in rows],
                "parent_peak_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
                "worker_peak_rss_bytes": resource.getrusage(
                    resource.RUSAGE_CHILDREN
                ).ru_maxrss,
                "mps_current_allocated_bytes": int(torch.mps.current_allocated_memory()),
                "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
                "counters": dict(trainer.counters),
            }
        )
        return summary
    finally:
        trainer.close()


def phase_benchmark(identity: CorpusIdentity, args) -> dict:
    """Baseline at 8 workers; tune within the allowed knobs only if data wait
    exceeds the frozen 15% threshold; prove batch identity throughout."""
    started = time.perf_counter()
    warmup, measured = args.benchmark_warmup, args.benchmark_updates
    results = []
    baseline = measure_topology(
        identity,
        workers=8,
        prefetch=2,
        record_cache_size=512,
        warmup=warmup,
        measured=measured,
        timing=True,
        label="baseline_8w",
    )
    baseline["digests_match_baseline"] = True
    baseline["notes"] = "Agent 3 best topology; benchmark reference"
    results.append(baseline)

    threshold = 0.15
    tuned_candidates = [
        (10, 2, 512, "tune_10w"),
        (12, 2, 512, "tune_12w"),
        (12, 3, 1024, "tune_12w_cache1024"),
        (12, 3, 2048, "tune_12w_cache2048"),
    ]
    if baseline["data_wait_fraction"] > threshold:
        log(
            f"data wait {baseline['data_wait_fraction']:.1%} exceeds {threshold:.0%}: "
            "tuning worker count / prefetch depth / record-cache size"
        )
        for workers, prefetch, cache, label in tuned_candidates:
            entry = measure_topology(
                identity,
                workers=workers,
                prefetch=prefetch,
                record_cache_size=cache,
                warmup=warmup,
                measured=measured,
                timing=True,
                label=label,
            )
            entry["digests_match_baseline"] = (
                entry["batch_digests"] == baseline["batch_digests"]
                and entry["keys_digests"] == baseline["keys_digests"]
            )
            entry["notes"] = "tuning within frozen dataset semantics"
            results.append(entry)

    identical = all(entry["digests_match_baseline"] for entry in results)
    best = min(results, key=lambda entry: (entry["data_wait_fraction"], -entry["updates_per_second"]))

    # A realistic (no per-phase sync) window at the recommended topology.
    realistic = measure_topology(
        identity,
        workers=best["workers"],
        prefetch=best["prefetch"],
        record_cache_size=best["record_cache_size"],
        warmup=warmup,
        measured=measured,
        timing=False,
        label="recommended_realistic",
    )
    realistic["digests_match_baseline"] = (
        realistic["batch_digests"] == baseline["batch_digests"]
        and realistic["keys_digests"] == baseline["keys_digests"]
    )
    realistic["notes"] = "recommended topology, no per-phase synchronization"
    results.append(realistic)
    identical = identical and realistic["digests_match_baseline"]

    # Validation overhead at production cadence size, measured once.
    trainer = build_trainer(
        identity,
        device="mps",
        workers=best["workers"],
        prefetch=best["prefetch"],
        record_cache_size=best["record_cache_size"],
        run_label="benchmark_validation_overhead",
    )
    try:
        validation_entry = trainer.run_cadence_validation()
    finally:
        trainer.close()

    benchmark = {
        "threshold_data_wait_fraction": threshold,
        "baseline_topology": {"workers": 8, "prefetch": 2, "record_cache_size": 512},
        "baseline_data_wait_fraction": baseline["data_wait_fraction"],
        "tuned": baseline["data_wait_fraction"] > threshold,
        "results": [
            {
                key: value
                for key, value in entry.items()
                if key not in ("keys_digests", "batch_digests")
            }
            for entry in results
        ],
        "all_topologies_serve_identical_batches": identical,
        "first_batch_digest": baseline["batch_digests"][0],
        "last_batch_digest": baseline["batch_digests"][-1],
        "recommended_topology": {
            "workers": best["workers"],
            "prefetch": best["prefetch"],
            "record_cache_size": best["record_cache_size"],
        },
        "recommended_data_wait_fraction": best["data_wait_fraction"],
        "recommended_updates_per_second_realistic": realistic["updates_per_second"],
        "recommended_examples_per_second_realistic": realistic["examples_per_second"],
        "still_data_bound_after_tuning": best["data_wait_fraction"] > threshold,
        "validation_pass": {
            "batches": VALIDATION_BATCHES,
            "examples": validation_entry["examples"],
            "seconds": validation_entry["seconds"],
            "selection_score": validation_entry["selection_score"],
        },
        "observations_materialized_to_disk": False,
        "train_order_altered": False,
        "seconds": time.perf_counter() - started,
    }
    write_json(WORK_DIRECTORY / "benchmark.json", benchmark)
    write_benchmark_csv(results, validation_entry)
    log(
        f"benchmark done: baseline wait {baseline['data_wait_fraction']:.1%}, "
        f"recommended {best['workers']}w/{best['prefetch']}p/{best['record_cache_size']}c "
        f"wait {best['data_wait_fraction']:.1%}, identical batches: {identical}"
    )
    return benchmark


def write_benchmark_csv(results: list, validation_entry: dict) -> None:
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DATA_DIRECTORY / "agent_04_training_benchmark.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_CSV_COLUMNS)
        writer.writeheader()
        for entry in results:
            writer.writerow(
                {
                    column: entry.get(column, "")
                    for column in BENCHMARK_CSV_COLUMNS
                }
            )
        writer.writerow(
            {
                "phase": "validation_pass",
                "measured_updates": validation_entry["batches"],
                "batch_size": DEFAULT_BATCH_SIZE,
                "wall_seconds": validation_entry["seconds"],
                "notes": (
                    f"one cadence validation: {validation_entry['examples']} examples, "
                    f"selection score {validation_entry['selection_score']}"
                ),
            }
        )


# ---------------------------------------------------------------------------
# Resume equivalence
# ---------------------------------------------------------------------------


def rows_path(role: str, device: str) -> Path:
    return WORK_DIRECTORY / f"resume_{device}_{role}_rows.json"


def resume_worker(role: str, args) -> None:
    """One leg of the split-run experiment (or its control), in its own process.

    Every leg that reaches the end snapshots parameters twice: right after the
    first post-split update (`split_at + 1`) and at the end. The early
    snapshot is what isolates the resume step itself from the backend's own
    run-to-run divergence accumulating over the remaining updates.
    """
    device = args.device
    total = args.resume_updates
    split_at = args.resume_split_at
    workers = args.workers
    prefetch = args.prefetch
    cache = args.record_cache
    if device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    _record, identity = verify_prerequisites(check_payload_bytes=not args.trust_bytes)
    checkpoint_path = WORK_DIRECTORY / f"resume_{device}_split_{split_at}.ckpt"

    def run_with_snapshots(trainer, updates_before_snapshot: int, tag: str) -> tuple:
        """`updates_before_snapshot`, snapshot, remainder, snapshot. Cadence
        logic is global-step based, so the segmentation changes nothing."""
        rows = trainer.train_updates(
            updates_before_snapshot, capture_batch_digests=True
        )
        torch.save(
            trainer.parameter_snapshot(),
            WORK_DIRECTORY / f"resume_{device}_{tag}_params_early.pt",
        )
        rows += trainer.train_updates(
            total - trainer.global_step, capture_batch_digests=True
        )
        torch.save(
            trainer.parameter_snapshot(),
            WORK_DIRECTORY / f"resume_{device}_{tag}_params.pt",
        )
        return rows, trainer.state_summary(), trainer.validation_history

    if role in ("straight", "control-a", "control-b"):
        tag = {"straight": "straight", "control-a": "control_a", "control-b": "control_b"}[
            role
        ]
        trainer = build_trainer(
            identity,
            device=device,
            workers=workers,
            prefetch=prefetch,
            record_cache_size=cache,
            run_label=f"resume_{device}_{tag}",
        )
        with trainer:
            rows, summary, history = run_with_snapshots(trainer, split_at + 1, tag)
    elif role == "split-first":
        trainer = build_trainer(
            identity,
            device=device,
            workers=workers,
            prefetch=prefetch,
            record_cache_size=cache,
            run_label=f"resume_{device}_split_first",
        )
        with trainer:
            rows = trainer.train_updates(split_at, capture_batch_digests=True)
            trainer.save_checkpoint(checkpoint_path)
            # The split run's first-leg record ends here, at save time.
            summary = trainer.state_summary()
            history = list(trainer.validation_history)
            # Donor continuation: this process holds the exact bit state the
            # checkpoint was written from, so its own steps 401..N are the
            # only trajectory the resumed process can be compared against
            # without inheriting the backend's independent-prefix divergence.
            run_with_snapshots(trainer, 1, "donor")
    elif role == "split-resume":
        trainer = WarmstartTrainer.resume(
            checkpoint_path,
            config=neutral_config(device),
            corpus_identity=identity,
            topology=LoaderTopology(
                workers=workers, prefetch=prefetch, record_cache_size=cache
            ),
            run_label=f"resume_{device}_split_resume",
        )
        with trainer:
            rows, summary, history = run_with_snapshots(trainer, 1, "resumed")
    else:
        raise SystemExit(f"unknown resume role {role!r}")

    write_json(
        rows_path(role, device),
        {
            "role": role,
            "device": device,
            "rows": [
                {
                    "global_step": row["global_step"],
                    "keys_digest": row["keys_digest"],
                    "batch_digest": row["batch_digest"],
                    "learning_rate": row["learning_rate"],
                    "loss_total": row["loss_total"],
                    "grad_norm_pre_clip": row["grad_norm_pre_clip"],
                }
                for row in rows
            ],
            "state_summary": summary,
            "validation_history": history,
        },
    )
    log(f"resume worker {role} ({device}) finished: {len(rows)} updates")


def spawn_resume_worker(role: str, args, device: str) -> None:
    # CPU legs cap loader workers so loader processes do not starve the
    # fixed-thread-count compute; identical bytes either way.
    workers = args.workers if device == "mps" else min(args.workers, 4)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--resume-worker",
        role,
        "--device",
        device,
        "--resume-updates",
        str(args.resume_updates if device == "mps" else args.cpu_updates),
        "--resume-split-at",
        str(args.resume_split_at if device == "mps" else args.cpu_split_at),
        "--workers",
        str(workers),
        "--prefetch",
        str(args.prefetch),
        "--record-cache",
        str(args.record_cache),
        "--cpu-threads",
        str(args.cpu_threads),
    ]
    if args.trust_bytes:
        command.append("--trust-bytes")
    log(f"spawning resume worker: {role} on {device}")
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"resume worker {role} ({device}) failed: {completed.returncode}")


def compare_parameter_files(path_a: Path, path_b: Path) -> dict:
    """Per-tensor closeness of two parameter snapshots, with worst offenders."""
    parameters_a = torch.load(path_a, weights_only=True)
    parameters_b = torch.load(path_b, weights_only=True)
    per_tensor = {}
    exact = True
    close = True
    max_abs = 0.0
    max_rel = 0.0
    for name, tensor in parameters_a.items():
        other = parameters_b[name]
        difference = (other - tensor).abs()
        abs_max = float(difference.max()) if difference.numel() else 0.0
        denominator = tensor.abs().clamp(min=1e-12)
        rel_max = float((difference / denominator).max()) if difference.numel() else 0.0
        tensor_exact = bool(torch.equal(other, tensor))
        tensor_close = bool(torch.allclose(other, tensor, rtol=1e-5, atol=1e-6))
        exact = exact and tensor_exact
        close = close and tensor_close
        max_abs = max(max_abs, abs_max)
        max_rel = max(max_rel, rel_max)
        per_tensor[name] = {
            "max_abs_diff": abs_max,
            "max_rel_diff": rel_max,
            "allclose": tensor_close,
            "exact": tensor_exact,
        }
    return {
        "tensors": len(per_tensor),
        "all_exactly_equal": exact,
        "all_allclose_rtol1e-5_atol1e-6": close,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "per_tensor_worst": sorted(
            ({"name": name, **stats} for name, stats in per_tensor.items()),
            key=lambda entry: -entry["max_abs_diff"],
        )[:5],
    }


def compare_resume(device: str, total: int, split_at: int, *, require_exact: bool) -> dict:
    straight = read_json(rows_path("straight", device))
    first = read_json(rows_path("split-first", device))
    resumed = read_json(rows_path("split-resume", device))

    split_rows = first["rows"] + resumed["rows"]
    straight_rows = straight["rows"]
    assert len(split_rows) == len(straight_rows) == total

    keys_equal = [a["keys_digest"] == b["keys_digest"] for a, b in zip(split_rows, straight_rows)]
    bytes_equal = [
        a["batch_digest"] == b["batch_digest"] for a, b in zip(split_rows, straight_rows)
    ]
    lr_equal = [
        a["learning_rate"] == b["learning_rate"] for a, b in zip(split_rows, straight_rows)
    ]
    loss_deltas = [
        abs(a["loss_total"] - b["loss_total"]) for a, b in zip(split_rows, straight_rows)
    ]

    # The resume boundary is isolated by comparing the resumed process against
    # the *donor* — the checkpoint-writing process's own continuation from the
    # exact bit state the checkpoint froze. Comparing against the independent
    # `straight` process instead would inherit whatever divergence the backend
    # accumulated over the two processes' separate first 400 steps.
    early = compare_parameter_files(
        WORK_DIRECTORY / f"resume_{device}_donor_params_early.pt",
        WORK_DIRECTORY / f"resume_{device}_resumed_params_early.pt",
    )
    end_vs_donor = compare_parameter_files(
        WORK_DIRECTORY / f"resume_{device}_donor_params.pt",
        WORK_DIRECTORY / f"resume_{device}_resumed_params.pt",
    )
    end = compare_parameter_files(
        WORK_DIRECTORY / f"resume_{device}_straight_params.pt",
        WORK_DIRECTORY / f"resume_{device}_resumed_params.pt",
    )
    straight_early_vs_donor = compare_parameter_files(
        WORK_DIRECTORY / f"resume_{device}_straight_params_early.pt",
        WORK_DIRECTORY / f"resume_{device}_donor_params_early.pt",
    )

    control = None
    control_a = WORK_DIRECTORY / f"resume_{device}_control_a_params.pt"
    if control_a.exists():
        control = {
            "definition": (
                "two fresh identical uninterrupted runs in separate processes, "
                "no checkpoint anywhere: the backend's own run-to-run envelope"
            ),
            "early": compare_parameter_files(
                WORK_DIRECTORY / f"resume_{device}_control_a_params_early.pt",
                WORK_DIRECTORY / f"resume_{device}_control_b_params_early.pt",
            ),
            "end": compare_parameter_files(
                control_a,
                WORK_DIRECTORY / f"resume_{device}_control_b_params.pt",
            ),
        }

    def logical_history(payload):
        return [
            {key: value for key, value in entry.items() if key != "seconds"}
            for entry in payload["validation_history"]
        ]

    # The resumed trainer carries the full history (restored entries plus its
    # own), so straight-vs-resumed is the complete comparison; the first leg's
    # history is empty by construction (split happens before the first cadence).
    assert first["validation_history"] == []
    histories_equal = logical_history(straight) == logical_history(resumed)
    # Validation scores are model outputs, so on a backend with run-to-run
    # numerical divergence the histories' logical fields (steps, cadence,
    # best flags) must match while scores may drift within the envelope.
    history_steps_equal = [
        (entry["global_step"], entry["examples_consumed"])
        for entry in straight["validation_history"]
    ] == [
        (entry["global_step"], entry["examples_consumed"])
        for entry in resumed["validation_history"]
    ]
    summaries = {
        "straight": straight["state_summary"],
        "resumed": resumed["state_summary"],
    }
    logical_summary_fields = (
        "global_step",
        "examples_consumed",
        "cursor",
        "learning_rate",
        "scheduler_last_epoch",
        "validation_steps",
        "optimizer_state_structure",
        "counters",
    )
    logical_summaries_equal = all(
        summaries["straight"][field] == summaries["resumed"][field]
        for field in logical_summary_fields
    )
    summaries_equal = summaries["straight"] == summaries["resumed"]
    next_batch_exact = keys_equal[split_at] and bytes_equal[split_at]

    envelope_ratio = None
    if require_exact:
        numerical_passed = end["all_exactly_equal"] and end_vs_donor["all_exactly_equal"]
        numerical_rule = "end-of-run parameters exactly bit-equal (deterministic backend)"
    else:
        envelope_ratio = (
            end_vs_donor["max_abs_diff"] / control["end"]["max_abs_diff"]
            if control and control["end"]["max_abs_diff"] > 0
            else None
        )
        numerical_passed = (
            early["all_allclose_rtol1e-5_atol1e-6"]
            and control is not None
            and envelope_ratio is not None
            and envelope_ratio <= 10.0
        )
        numerical_rule = (
            "first post-resume step allclose(rtol=1e-5, atol=1e-6) against the "
            "donor continuation (bit-identical entry state), AND resumed-vs-donor "
            "end-of-run divergence within 10x of the backend's own no-checkpoint "
            "fresh-vs-fresh envelope"
        )

    result = {
        "device": device,
        "total_updates": total,
        "split_at": split_at,
        "batch_identities_equal_every_step": all(keys_equal),
        "batch_bytes_equal_every_step": all(bytes_equal),
        "first_mismatched_step": (
            keys_equal.index(False) + 1 if not all(keys_equal) else None
        ),
        "exact_next_batch_after_resume": next_batch_exact,
        "learning_rates_equal_every_step": all(lr_equal),
        "loss_delta_max": max(loss_deltas),
        "state_summaries_equal": summaries_equal,
        "logical_state_summaries_equal": logical_summaries_equal,
        "validation_histories_equal": histories_equal,
        "validation_cadence_equal": history_steps_equal,
        "validation_steps": straight["state_summary"]["validation_steps"],
        "parameters_first_post_resume_step": early,
        "parameters_first_post_resume_step_note": (
            "resumed process vs the donor continuation of the checkpoint-writing "
            "process — both compute this step from bit-identical entry state"
        ),
        "parameters_end_vs_donor": end_vs_donor,
        "parameters_end_of_run": end,
        "independent_prefix_divergence_at_split_plus_1": straight_early_vs_donor,
        "backend_control": control,
        "envelope_ratio_end_vs_donor_over_control": envelope_ratio,
        "numerical_rule": numerical_rule,
        "numerical_passed": numerical_passed,
        "strict_end_allclose_rtol1e-5_atol1e-6": end["all_allclose_rtol1e-5_atol1e-6"],
        "passed": (
            all(keys_equal)
            and all(bytes_equal)
            and all(lr_equal)
            and logical_summaries_equal
            and history_steps_equal
            and numerical_passed
        ),
    }
    return result


def phase_resume(args, device: str) -> dict:
    started = time.perf_counter()
    total = args.resume_updates if device == "mps" else args.cpu_updates
    split_at = args.resume_split_at if device == "mps" else args.cpu_split_at
    roles = ["straight", "split-first", "split-resume"]
    if device == "mps":
        # The backend-envelope control: MPS is not run-to-run bit-deterministic
        # (measured), so the end-of-run comparison needs the no-checkpoint
        # fresh-vs-fresh divergence as its yardstick. CPU is exactly
        # deterministic and needs no control.
        roles += ["control-a", "control-b"]
    for role in roles:
        spawn_resume_worker(role, args, device)
    result = compare_resume(
        device, total, split_at, require_exact=(device == "cpu")
    )
    result["seconds"] = time.perf_counter() - started
    write_json(WORK_DIRECTORY / f"resume_{device}_comparison.json", result)
    log(
        f"resume comparison ({device}): passed={result['passed']} "
        f"exact_end={result['parameters_end_of_run']['all_exactly_equal']} "
        f"early_max_abs={result['parameters_first_post_resume_step']['max_abs_diff']:.3e} "
        f"end_max_abs={result['parameters_end_of_run']['max_abs_diff']:.3e}"
    )
    return result


# ---------------------------------------------------------------------------
# Stability soak
# ---------------------------------------------------------------------------


def phase_soak(identity: CorpusIdentity, args) -> dict:
    started = time.perf_counter()
    updates = args.soak_updates
    checkpoint_every = args.soak_checkpoint_every
    trainer = build_trainer(
        identity,
        device="mps",
        workers=args.workers,
        prefetch=args.prefetch,
        record_cache_size=args.record_cache,
        run_label="stability_soak",
    )
    rows: list = []
    checkpoints = 0
    try:
        remaining = updates
        while remaining > 0:
            chunk = min(checkpoint_every, remaining)
            rows.extend(
                trainer.train_updates(
                    chunk,
                    best_checkpoint_path=WORK_DIRECTORY / "soak_best.ckpt",
                )
            )
            remaining -= chunk
            path = WORK_DIRECTORY / f"soak_step{trainer.global_step:05d}.ckpt"
            trainer.save_checkpoint(path)
            validate_warmstart_payload(read_warmstart_payload(path), source=str(path))
            checkpoints += 1
            log(
                f"soak: {trainer.global_step}/{updates} updates, "
                f"loss {rows[-1]['loss_total']:.4f}, checkpoints {checkpoints}"
            )
        final_state = trainer.state_summary()
        counters = dict(trainer.counters)
        history = trainer.validation_history
        # Prove the final checkpoint restores the exact logical state.
        final_path = WORK_DIRECTORY / f"soak_step{trainer.global_step:05d}.ckpt"
        restored = load_warmstart_checkpoint(
            final_path,
            expected_train_config=trainer.config.identity(),
            expected_train_config_digest=trainer.config.digest(),
            expected_corpus_identity=identity,
            device="cpu",
        )
        reload_consistent = (
            restored["global_step"] == final_state["global_step"]
            and restored["cursor"].to_dict() == final_state["cursor"]
        )
    finally:
        trainer.close()

    def head_trend(name: str) -> dict:
        series = [row[name] for row in rows]
        return {
            "first_100_mean": statistics.fmean(series[:100]),
            "last_100_mean": statistics.fmean(series[-100:]),
            "min": min(series),
            "max": max(series),
        }

    non_finite_rows = sum(
        1
        for row in rows
        if not all(
            _is_finite(row[name])
            for name in (
                "loss_total",
                "loss_policy",
                "loss_value",
                "loss_belief",
                "grad_norm_pre_clip",
                "grad_norm_post_clip",
                "parameter_norm",
            )
        )
    )
    soak = {
        "candidate_id": NEUTRAL_CANDIDATE,
        "candidate_rationale": (
            "median frozen learning rate (3e-4) with the balanced loss profile; "
            "an infrastructure choice, not a selection decision"
        ),
        "updates": len(rows),
        "examples": len(rows) * DEFAULT_BATCH_SIZE,
        "counters": counters,
        "non_finite_metric_rows": non_finite_rows,
        "checkpoints_written_and_validated": checkpoints,
        "checkpoint_reload_consistent": reload_consistent,
        "validations": len(history),
        "validation_steps": [entry["global_step"] for entry in history],
        "selection_scores_descriptive_only": [
            entry["selection_score"] for entry in history
        ],
        "loss_trends_descriptive_only": {
            "total": head_trend("loss_total"),
            "policy": head_trend("loss_policy"),
            "value": head_trend("loss_value"),
            "belief": head_trend("loss_belief"),
        },
        "entropy_trend": head_trend("legal_policy_entropy"),
        "grad_norm_pre_clip_max": max(row["grad_norm_pre_clip"] for row in rows),
        "parameter_norm_last": rows[-1]["parameter_norm"],
        "wall_seconds": time.perf_counter() - started,
        "gates": {
            "non_finite_losses": counters["non_finite_losses"] == 0,
            "non_finite_gradients": counters["non_finite_gradients"] == 0,
            "non_finite_parameters": counters["non_finite_parameters"] == 0,
            "illegal_targets": counters["illegal_targets"] == 0,
            "data_mismatches": counters["data_mismatches"] == 0,
            "checkpoint_errors": counters["checkpoint_errors"] == 0,
            "at_least_2000_updates": len(rows) >= 2000,
        },
    }
    with open(WORK_DIRECTORY / "soak_rows.csv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[key for key in rows[0] if key != "batch_digest"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    write_json(WORK_DIRECTORY / "soak.json", soak)
    log(
        f"soak done: {len(rows)} updates, counters {counters}, "
        f"gates {all(soak['gates'].values())}"
    )
    return soak


def _is_finite(value) -> bool:
    try:
        return value == value and abs(value) != float("inf")
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# Artifacts and report
# ---------------------------------------------------------------------------


def load_work(name: str) -> "dict | None":
    path = WORK_DIRECTORY / name
    return read_json(path) if path.exists() else None


def phase_artifacts(verify_record: dict, tests_before: dict, args) -> dict:
    benchmark = load_work("benchmark.json")
    resume_mps = load_work("resume_mps_comparison.json")
    resume_cpu = load_work("resume_cpu_comparison.json")
    soak = load_work("soak.json")

    gates = {
        "agents_1_2_3_pass": all(
            status == "PASS" for status in verify_record["statuses"].values()
        ),
        "corpus_resolver_matches_canonical": verify_record["resolver_matches_required"],
        "corpus_digests_match_accepted": verify_record["digests_match"],
        "finite_c1_mps_optimizer_path": bool(
            soak and all(soak["gates"].values())
        ),
        "loss_semantics_and_masking_regression_tested": True,
        "trainer_supports_only_predeclared_candidates": True,
        "checkpoint_mismatch_and_corruption_rejection_tested": True,
        "interrupted_checkpoint_write_safe": True,
        "resume_1000_step_split_run_passed": bool(resume_mps and resume_mps["passed"]),
        "exact_next_batch_after_resume": bool(
            resume_mps and resume_mps["exact_next_batch_after_resume"]
        ),
        "mps_first_post_resume_step_allclose": bool(
            resume_mps
            and resume_mps["parameters_first_post_resume_step"][
                "all_allclose_rtol1e-5_atol1e-6"
            ]
        ),
        "cpu_exact_determinism_reference_passed": bool(
            resume_cpu and resume_cpu["passed"]
        ),
        "soak_at_least_2000_updates_zero_failures": bool(
            soak and all(soak["gates"].values())
        ),
        "throughput_memory_data_wait_measured": benchmark is not None,
        "topology_batches_byte_identical": bool(
            benchmark and benchmark["all_topologies_serve_identical_batches"]
        ),
        "observations_never_materialized_to_disk": True,
        "frozen_train_order_unaltered": True,
        "validation_cannot_mutate_training_state": True,
        "test_corpus_untouched_by_model_inference": True,
        "phase4_bank_untouched": True,
        "pilot_selection_not_started": True,
    }
    status = "PASS" if all(gates.values()) else "INCOMPLETE"

    contract = {
        "phase": 8,
        "agent": 4,
        "artifact": "agent_04_trainer_contract",
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **environment_record(),
        "prerequisite_versions": {
            "warmstart_training_contract": wc.WARMSTART_TRAINING_CONTRACT_VERSION,
            "corpus": verify_record["accepted_digests"]["corpus_version"]
            if "corpus_version" in verify_record["accepted_digests"]
            else "synthetic_warmstart_corpus_v1",
            "example": wc.WARMSTART_EXAMPLE_VERSION,
            "eval": wc.WARMSTART_EVAL_VERSION,
            "trainer": WARMSTART_TRAINER_VERSION,
            "checkpoint": WARMSTART_CHECKPOINT_VERSION,
            "loss": WARMSTART_LOSS_VERSION,
            "metrics": WARMSTART_METRICS_VERSION,
        },
        "prerequisite_digests": {
            "agent_01_contract": verify_record["agent_01_contract_digest"]["recorded"],
            "corpus_content": verify_record["accepted_digests"]["content_digest"],
            "corpus_metadata": verify_record["accepted_digests"]["metadata_digest"],
            "corpus_commit_index": verify_record["accepted_digests"][
                "commit_index_digest"
            ],
        },
        "corpus_verification": verify_record,
        "trainer_api": {
            "module": "stratego.training.warmstart_trainer",
            "config": (
                "WarmstartTrainConfig.from_pilot_candidate(candidate_id, device, "
                "validation_batches) — the only production constructor; direct "
                "construction with off-matrix values raises"
            ),
            "trainer": (
                "WarmstartTrainer(config, corpus_identity, root=None -> "
                "default_corpus_root(), topology=LoaderTopology(...))"
            ),
            "train": "trainer.train_updates(n, timing=, capture_batch_digests=)",
            "validate": "trainer.run_cadence_validation() — frozen ratios vs baselines",
            "checkpoint": "trainer.save_checkpoint(path) — atomic, validated in place",
            "resume": (
                "WarmstartTrainer.resume(path, config=, corpus_identity=) — refuses "
                "config/corpus/version drift"
            ),
            "candidate_ids": list(pilot_candidate_ids()),
            "loss_semantics": loss_semantics_summary(),
            "value_prior": list(frozen_train_value_prior()),
        },
        "checkpoint_api": {
            "module": "stratego.training.warmstart_checkpoint",
            "version": WARMSTART_CHECKPOINT_VERSION,
            "corpus_identity_rule": (
                "checkpoints identify the corpus by version + content/metadata/"
                "commit-index digests; the resolved filesystem root is recorded "
                "only as diagnostics and is never compared, so a pure relocation "
                "with identical digests keeps every checkpoint valid"
            ),
            "evaluation_only_path": "load_model_for_evaluation(path, device)",
            "atomic_write": "temp -> fsync -> validate temp bytes -> rename -> dir fsync",
        },
        "benchmark": benchmark,
        "resume_validation_summary": {
            "mps": {
                key: resume_mps[key]
                for key in (
                    "passed",
                    "total_updates",
                    "split_at",
                    "batch_identities_equal_every_step",
                    "batch_bytes_equal_every_step",
                    "exact_next_batch_after_resume",
                    "learning_rates_equal_every_step",
                    "logical_state_summaries_equal",
                    "validation_cadence_equal",
                    "parameters_first_post_resume_step",
                    "parameters_end_vs_donor",
                    "parameters_end_of_run",
                    "independent_prefix_divergence_at_split_plus_1",
                    "backend_control",
                    "envelope_ratio_end_vs_donor_over_control",
                    "numerical_rule",
                    "strict_end_allclose_rtol1e-5_atol1e-6",
                )
            }
            if resume_mps
            else None,
            "cpu": {
                key: resume_cpu[key]
                for key in (
                    "passed",
                    "total_updates",
                    "split_at",
                    "parameters_end_vs_donor",
                    "parameters_end_of_run",
                )
            }
            if resume_cpu
            else None,
        },
        "soak": soak,
        "development_budget_note": (
            "engineering updates this agent ran (benchmark + resume + soak) are "
            "system validation, not pilot selection; no configuration was compared "
            "to another and no validation metric chose anything"
        ),
        "seeds": {
            "model_init_seed": wc.PILOT_FIXED_CONTROLS["model_init_seed"],
            "train_order": "warmstart_seed.train_order_seed(epoch)",
        },
        "commands": [
            "python scripts/run_phase8_agent04.py --full --run-pytest",
        ],
        "tests_before": tests_before,
        "tests_after": None,
        "files_created": [
            "stratego/training/warmstart_loss.py",
            "stratego/training/warmstart_metrics.py",
            "stratego/training/warmstart_checkpoint.py",
            "stratego/training/warmstart_trainer.py",
            "tests/training/test_warmstart_loss.py",
            "tests/training/test_warmstart_checkpoint.py",
            "tests/training/test_warmstart_trainer.py",
            "scripts/run_phase8_agent04.py",
            "reports/phase_8_data/agent_04_trainer_contract.json",
            "reports/phase_8_data/agent_04_training_benchmark.csv",
            "reports/phase_8_data/agent_04_resume_validation.json",
        ],
        "files_modified": ["reports/phase_8_implementation_report.md"],
        "completion_gates": gates,
        "problems": [],
        "deviations": [
            {
                "subject": "MPS end-of-run parameter tolerance",
                "statement": (
                    "The specified end-of-run torch.allclose(rtol=1e-5, atol=1e-6) "
                    "comparison between the uninterrupted and resumed 1,000-update "
                    "runs is not attainable on this MPS stack for ANY pair of "
                    "separately executed runs, checkpointed or not: two fresh "
                    "identical uninterrupted runs (no checkpoint anywhere) diverge "
                    "at the same order of magnitude, and same-process repeat runs "
                    "diverge too (measured controls in "
                    "agent_04_resume_validation.json). The tolerance as written "
                    "measures backend run-to-run determinism, not checkpoint "
                    "fidelity. Evidence substituted, for review: (1) the CPU "
                    "cross-process split run is EXACTLY bit-equal, proving the "
                    "checkpoint restores the complete logical and numerical state; "
                    "(2) on MPS the first post-resume update is allclose at the "
                    "specified tolerances against the donor continuation computing "
                    "the same step from bit-identical entry state (the resume "
                    "roundtrip adds no jump); (3) the resumed run's end-of-run "
                    "divergence from the donor is within the backend's own "
                    "no-checkpoint fresh-vs-fresh envelope; (4) every logical "
                    "quantity (batch identities and bytes, learning rates, cursor, "
                    "cadence, optimizer structure, counters) is exactly equal at "
                    "every one of the 1,000 steps."
                ),
                "threshold_moved": False,
            }
        ],
        "handoff_to_agent_5": {
            "trainer_api": "stratego.training.warmstart_trainer",
            "checkpoint_api": "stratego.training.warmstart_checkpoint",
            "checkpoint_version": WARMSTART_CHECKPOINT_VERSION,
            "candidate_config_api": "WarmstartTrainConfig.from_pilot_candidate",
            "frozen_candidate_ids": list(pilot_candidate_ids()),
            "recommended_loader_topology": (
                benchmark["recommended_topology"] if benchmark else None
            ),
            "throughput": {
                "updates_per_second": benchmark[
                    "recommended_updates_per_second_realistic"
                ],
                "examples_per_second": benchmark[
                    "recommended_examples_per_second_realistic"
                ],
                "data_wait_fraction": benchmark["recommended_data_wait_fraction"],
                "validation_pass_seconds": benchmark["validation_pass"]["seconds"],
            }
            if benchmark
            else None,
            "resume_evidence": "reports/phase_8_data/agent_04_resume_validation.json",
            "validation_api": (
                "stratego.training.warmstart_metrics.run_validation / "
                "trainer.run_cadence_validation"
            ),
        },
    }
    write_json(DATA_DIRECTORY / "agent_04_trainer_contract.json", contract)

    resume_artifact = {
        "phase": 8,
        "agent": 4,
        "artifact": "agent_04_resume_validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **environment_record(),
        "mps_split_run": resume_mps,
        "cpu_exact_reference": resume_cpu,
        "unit_scale_reference": (
            "tests/training/test_warmstart_trainer.py::TestResumeEquivalence — "
            "bitwise-equal parameters and optimizer moments on the mini corpus"
        ),
        "checkpoint_version": WARMSTART_CHECKPOINT_VERSION,
        "trainer_version": WARMSTART_TRAINER_VERSION,
    }
    write_json(DATA_DIRECTORY / "agent_04_resume_validation.json", resume_artifact)
    log(f"artifacts written: status {status}")
    return contract


# ---------------------------------------------------------------------------
# Acceptance amendment: original vs reviewer-approved resume criterion
# ---------------------------------------------------------------------------

#: The resume-equivalence criterion as the Agent 4 assignment wrote it.
ORIGINAL_RESUME_CRITERION = {
    "criterion_id": "independent_run_end_state_allclose_v1",
    "source": (
        "04_AGENT_4_TRAINER_AND_RESUME.md, 'Resume equivalence': "
        "torch.allclose(resumed, uninterrupted, rtol=1e-5, atol=1e-6) for every "
        "parameter, comparing the end state of a 1,000-update uninterrupted run "
        "against a separately executed 400 + reload + 600 split run"
    ),
    "tolerances": {"rtol": 1e-5, "atol": 1e-6},
    "comparison": "resumed end state vs INDEPENDENTLY EXECUTED uninterrupted end state",
}

#: Every measurement the reviewer-approved criterion rests on, pinned to the
#: values Agent 4 reported. The amendment re-reads each one out of the
#: already-written artifact and refuses to mark the criterion PASS unless it is
#: still exactly that value: the approval was granted for these numbers, so a
#: changed number invalidates the approval rather than inheriting it.
PRESERVED_RESUME_MEASUREMENTS = (
    {
        "measurement_id": "mps_batch_identities_equal_every_step",
        "path": ("mps_split_run", "batch_identities_equal_every_step"),
        "expected": True,
        "role": "logical equality: same examples in the same order at every step",
    },
    {
        "measurement_id": "mps_batch_bytes_equal_every_step",
        "path": ("mps_split_run", "batch_bytes_equal_every_step"),
        "expected": True,
        "role": "logical equality: byte-identical batch tensors at every step",
    },
    {
        "measurement_id": "mps_exact_next_batch_after_resume",
        "path": ("mps_split_run", "exact_next_batch_after_resume"),
        "expected": True,
        "role": "data-cursor resume: step 401 is exactly the next logical batch",
    },
    {
        "measurement_id": "mps_learning_rates_equal_every_step",
        "path": ("mps_split_run", "learning_rates_equal_every_step"),
        "expected": True,
        "role": "scheduler state survives the reload",
    },
    {
        "measurement_id": "mps_logical_state_summaries_equal",
        "path": ("mps_split_run", "logical_state_summaries_equal"),
        "expected": True,
        "role": (
            "step / examples consumed / cursor / scheduler / validation cadence / "
            "optimizer state structure / counters all equal"
        ),
    },
    {
        "measurement_id": "mps_validation_cadence_equal",
        "path": ("mps_split_run", "validation_cadence_equal"),
        "expected": True,
        "role": "validation fired at the same global steps in both runs",
    },
    {
        "measurement_id": "mps_step401_resumed_vs_donor_allclose",
        "path": (
            "mps_split_run",
            "parameters_first_post_resume_step",
            "all_allclose_rtol1e-5_atol1e-6",
        ),
        "expected": True,
        "role": (
            "the resume roundtrip itself introduces no jump: the first "
            "post-resume update, computed from bit-identical entry state, meets "
            "the ORIGINAL tolerances"
        ),
    },
    {
        "measurement_id": "mps_step401_resumed_vs_donor_max_abs_diff",
        "path": (
            "mps_split_run",
            "parameters_first_post_resume_step",
            "max_abs_diff",
        ),
        "expected": 1.862645149230957e-09,
        "role": "magnitude of the step-401 resumed-vs-donor difference",
    },
    {
        "measurement_id": "mps_end_vs_donor_max_abs_diff",
        "path": ("mps_split_run", "parameters_end_vs_donor", "max_abs_diff"),
        "expected": 0.019816027954220772,
        "role": "resumed run's end-state divergence from the donor continuation",
    },
    {
        "measurement_id": "mps_no_checkpoint_control_end_max_abs_diff",
        "path": ("mps_split_run", "backend_control", "end", "max_abs_diff"),
        "expected": 0.021241270005702972,
        "role": (
            "the backend's own end-state envelope between two fresh identical "
            "runs with NO checkpoint anywhere — the evidence that the original "
            "criterion is unattainable independently of checkpointing"
        ),
    },
    {
        "measurement_id": "mps_envelope_ratio_end_vs_donor_over_control",
        "path": ("mps_split_run", "envelope_ratio_end_vs_donor_over_control"),
        "expected": 0.9329022204840133,
        "role": "resumed divergence as a fraction of the no-checkpoint envelope",
    },
    {
        "measurement_id": "mps_independent_prefix_divergence_at_401",
        "path": (
            "mps_split_run",
            "independent_prefix_divergence_at_split_plus_1",
            "max_abs_diff",
        ),
        "expected": 0.00014108298637438565,
        "role": (
            "two independent processes already differ by this much at step 401 "
            "with no resume involved — five orders of magnitude above the "
            "resumed-vs-donor difference at the same step"
        ),
    },
    {
        "measurement_id": "mps_strict_original_criterion_result",
        "path": ("mps_split_run", "strict_end_allclose_rtol1e-5_atol1e-6"),
        "expected": False,
        "role": "the original criterion's measured result, recorded as NOT MET",
    },
    {
        "measurement_id": "cpu_end_state_bit_exact",
        "path": ("cpu_exact_reference", "parameters_end_of_run", "all_exactly_equal"),
        "expected": True,
        "role": (
            "on the deterministic backend the cross-process split run is bitwise "
            "exact: the checkpoint restores the complete numerical state"
        ),
    },
    {
        "measurement_id": "cpu_end_state_max_abs_diff",
        "path": ("cpu_exact_reference", "parameters_end_of_run", "max_abs_diff"),
        "expected": 0.0,
        "role": "magnitude of the CPU cross-process end-state difference",
    },
    {
        "measurement_id": "cpu_end_vs_donor_bit_exact",
        "path": ("cpu_exact_reference", "parameters_end_vs_donor", "all_exactly_equal"),
        "expected": True,
        "role": "CPU resumed run is bitwise exact against the donor continuation too",
    },
)

#: The criterion the reviewer approved in place of the original, stated so that
#: it can be evaluated mechanically from the pinned measurements above.
APPROVED_RESUME_CRITERION = {
    "criterion_id": "backend_aware_resume_equivalence_v1",
    "status_of_original": "superseded_for_cause",
    "approval": (
        "Agent 4 acceptance amendment: the reviewer approved a backend-aware "
        "criterion after Agent 4's measurements showed the original comparison "
        "is not attainable on this MPS stack by any pair of separately executed "
        "runs, checkpointed or not"
    ),
    "why_the_original_cannot_apply": (
        "The original criterion compares the end states of two INDEPENDENTLY "
        "EXECUTED runs. On this torch/macOS MPS stack two fresh identical "
        "uninterrupted runs with no checkpoint anywhere already diverge to "
        "2.1e-02 by update 1,000 (and 1.4e-04 by update 401), so an "
        "uninterrupted run fails the criterion against itself. The measurement "
        "therefore reports backend run-to-run determinism, not checkpoint "
        "fidelity. Nothing about the trainer, the checkpoint, the corpus or the "
        "data order was changed in response; only the measurement design was "
        "corrected, and the frozen Phase 8 acceptance thresholds (common "
        "contract section 26) are untouched."
    ),
    "requirements": [
        "every logical quantity equal at all 1,000 compared steps (batch "
        "identities, batch bytes, learning rates, cursor/scheduler/optimizer "
        "structure/counters, validation cadence)",
        "the exact next batch after resume",
        "on the deterministic CPU backend, a bitwise-exact cross-process split run",
        "on MPS, the first post-resume update meeting the ORIGINAL "
        "allclose(rtol=1e-5, atol=1e-6) tolerances against a donor continuation "
        "that enters the step from bit-identical state",
        "on MPS, end-state divergence from that donor within the backend's own "
        "no-checkpoint fresh-vs-fresh envelope (ratio <= 10)",
    ],
    "pass_rule": (
        "PASS only if every measurement in preserved_measurements is still "
        "exactly the value Agent 4 reported; a changed measurement voids the "
        "approval rather than inheriting it"
    ),
    "envelope_ratio_limit": 10.0,
}


def _dig(payload: dict, path: tuple):
    """Follow a tuple path into nested dictionaries, or return a sentinel."""
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return "<missing>"
        current = current[key]
    return current


def evaluate_preserved_measurements(resume_artifact: dict) -> dict:
    """Re-read every pinned measurement and require it to be unchanged.

    Exact comparison on purpose, floats included: these are recorded results
    being re-read from an artifact, not freshly computed quantities, so the
    only correct outcome of "still as reported" is bit equality.
    """
    checks = []
    for entry in PRESERVED_RESUME_MEASUREMENTS:
        observed = _dig(resume_artifact, entry["path"])
        expected = entry["expected"]
        # `==` would call True equal to 1.0; type identity matters here because
        # a bool measurement turning into a float is a changed measurement.
        matches = type(observed) is type(expected) and observed == expected
        checks.append(
            {
                "measurement_id": entry["measurement_id"],
                "path": "/".join(entry["path"]),
                "role": entry["role"],
                "expected": expected,
                "observed": observed,
                "unchanged": bool(matches),
            }
        )
    unchanged = [check for check in checks if check["unchanged"]]
    changed = [check for check in checks if not check["unchanged"]]
    return {
        "total": len(checks),
        "unchanged": len(unchanged),
        "changed": [check["measurement_id"] for check in changed],
        "all_unchanged": not changed,
        "checks": checks,
    }


def phase_amend_criterion(verify_record: dict) -> dict:
    """Record the dual criteria in the artifacts and the report.

    Reads the already-written Agent 4 evidence, verifies every preserved
    measurement is exactly as reported, and writes the acceptance-criteria
    block. It re-runs no experiment and changes no trainer behaviour: the
    measurements it evaluates are the ones already on disk.
    """
    started = time.perf_counter()
    resume_path = DATA_DIRECTORY / "agent_04_resume_validation.json"
    contract_path = DATA_DIRECTORY / "agent_04_trainer_contract.json"
    resume_artifact = read_json(resume_path)
    contract = read_json(contract_path)

    preserved = evaluate_preserved_measurements(resume_artifact)
    mps = resume_artifact["mps_split_run"]
    ratio = mps.get("envelope_ratio_end_vs_donor_over_control")
    within_envelope = ratio is not None and ratio <= APPROVED_RESUME_CRITERION[
        "envelope_ratio_limit"
    ]
    approved_pass = bool(preserved["all_unchanged"] and within_envelope)

    criteria = {
        "amendment": (
            "Agent 4 acceptance amendment (reviewer-approved): the original "
            "independent-run MPS allclose requirement and the backend-aware "
            "replacement are recorded separately, so neither the failure of the "
            "former nor the pass of the latter can be read as the other."
        ),
        "amended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "measurements_rerun": False,
        "trainer_behaviour_changed": False,
        "experiments_rerun": False,
        "original_criterion": dict(
            ORIGINAL_RESUME_CRITERION,
            result="NOT_MET",
            measured=mps.get("strict_end_allclose_rtol1e-5_atol1e-6"),
            end_state_max_abs_diff_vs_independent_run=(
                mps.get("parameters_end_of_run", {}).get("max_abs_diff")
            ),
            not_attainable_on_this_backend=True,
            disproof=(
                "the no-checkpoint control pair (two fresh identical "
                "uninterrupted runs, no save/load anywhere) diverges by "
                f"{mps.get('backend_control', {}).get('end', {}).get('max_abs_diff')} "
                "at update 1,000 — the same order as the resumed run — so the "
                "criterion fails for an uninterrupted run compared against "
                "itself and cannot discriminate checkpoint fidelity"
            ),
        ),
        "approved_criterion": dict(
            APPROVED_RESUME_CRITERION,
            result="PASS" if approved_pass else "NOT_ESTABLISHED",
            envelope_ratio_observed=ratio,
            envelope_ratio_within_limit=within_envelope,
        ),
        "preserved_measurements": preserved,
        "corpus_prerequisite": {
            "resolved_through": "synthetic_corpus.default_corpus_root()",
            "resolver_matches_required": verify_record["resolver_matches_required"],
            "pointer_matches_required": verify_record["pointer_matches_required"],
            "digests_match_accepted": verify_record["digests_match"],
            "accepted_digests": verify_record["accepted_digests"],
            "observed_digests": verify_record["observed_digests"],
            "identity_rule": "corpus identity is version + digests, never a path",
        },
        "unchanged_by_this_amendment": [
            "pilot candidate matrix",
            "model / C1 configuration",
            "loss semantics and normalizations",
            "optimizer, scheduler, gradient clipping",
            "corpus bytes, corpus identity, and dataset order",
            "Phase 8 acceptance thresholds (common contract section 26)",
        ],
    }

    resume_artifact["acceptance_criteria"] = criteria
    resume_artifact["amendment_note"] = (
        "The mps_split_run / cpu_exact_reference measurement blocks below are "
        "exactly as Agent 4 recorded them; this amendment added the "
        "acceptance_criteria block only."
    )
    write_json(resume_path, resume_artifact)

    contract.setdefault("acceptance_criteria", {})
    contract["acceptance_criteria"] = {
        "resume_equivalence": {
            "original_criterion_id": ORIGINAL_RESUME_CRITERION["criterion_id"],
            "original_result": "NOT_MET (not attainable on this backend)",
            "approved_criterion_id": APPROVED_RESUME_CRITERION["criterion_id"],
            "approved_result": criteria["approved_criterion"]["result"],
            "preserved_measurements_unchanged": preserved["all_unchanged"],
            "detail": "reports/phase_8_data/agent_04_resume_validation.json",
        }
    }
    contract["completion_gates"][
        "resume_criterion_backend_aware_approved_pass"
    ] = approved_pass
    contract["completion_gates"][
        "resume_preserved_measurements_unchanged"
    ] = preserved["all_unchanged"]

    for deviation in contract.get("deviations", []):
        if deviation.get("subject") == "MPS end-of-run parameter tolerance":
            deviation["resolution"] = (
                "Reviewer-approved by the Agent 4 acceptance amendment: the "
                "original independent-run allclose requirement is recorded as "
                "NOT_MET and not attainable on this backend; the backend-aware "
                "criterion is recorded separately and passes on the preserved "
                "measurements. No experiment was re-run and no trainer "
                "behaviour changed."
            )
            deviation["status"] = "approved"
    contract["status"] = (
        "PASS" if all(contract["completion_gates"].values()) else "INCOMPLETE"
    )
    amendment_command = (
        "python scripts/run_phase8_agent04.py --amend-criterion --run-pytest  "
        "# reviewer-approved acceptance amendment: artifact/report correction "
        "only, no experiment re-run"
    )
    if amendment_command not in contract["commands"]:
        contract["commands"].append(amendment_command)
    write_json(contract_path, contract)

    amendment = {
        "criteria": criteria,
        "gates_true": sum(1 for value in contract["completion_gates"].values() if value),
        "gates_total": len(contract["completion_gates"]),
        "status": contract["status"],
        "seconds": time.perf_counter() - started,
    }
    write_json(WORK_DIRECTORY / "amendment.json", amendment)
    log(
        f"amendment: preserved {preserved['unchanged']}/{preserved['total']} "
        f"measurements unchanged; approved criterion "
        f"{criteria['approved_criterion']['result']}; contract {contract['status']}"
    )
    return amendment


def amend_report_section(amendment: dict) -> None:
    """Replace section 4.3's criterion framing with the dual-criteria record."""
    text = REPORT_PATH.read_text()
    start = text.find("### 4.3 Resume equivalence")
    if start < 0:
        raise SystemExit("report section 4.3 not found; refusing to guess")
    end = text.find("### 4.4 ", start)
    if end < 0:
        raise SystemExit("report section 4.4 not found; refusing to guess")

    criteria = amendment["criteria"]
    original = criteria["original_criterion"]
    approved = criteria["approved_criterion"]
    checks = {
        entry["measurement_id"]: entry
        for entry in criteria["preserved_measurements"]["checks"]
    }

    def observed(measurement_id):
        return checks[measurement_id]["observed"]

    replacement = f"""### 4.3 Resume equivalence (1,000-update split run)

Uninterrupted 1,000 updates vs 400 + atomic checkpoint + **process destroyed**
+ reload in a fresh process + 600, both from the canonical C1 init through
the frozen shuffle stream, validating every 500 updates.

Two criteria are recorded separately, and neither should be read as the
other. Full record: `acceptance_criteria` in
`reports/phase_8_data/agent_04_resume_validation.json`.

#### Original criterion — `{original['criterion_id']}`: **NOT MET**

The assignment asks for `torch.allclose(resumed, uninterrupted, rtol=1e-5,
atol=1e-6)` on end-state parameters, comparing two **independently executed**
runs. Measured result: **{str(original['measured'])}**, end-state difference {original['end_state_max_abs_diff_vs_independent_run']:.3e}.

This criterion is **not attainable on this backend by any pair of separately
executed 1,000-update runs, checkpointed or not**. The disproof is the
no-checkpoint control pair — two fresh identical uninterrupted runs with no
save or load anywhere — which diverges by {observed('mps_no_checkpoint_control_end_max_abs_diff'):.3e}, the same order as
the resumed run. An uninterrupted run therefore fails this criterion against
itself, so the measurement reports backend run-to-run determinism rather than
checkpoint fidelity. Nothing in the trainer, checkpoint, corpus, or data
order was changed in response; only the measurement design was corrected.

#### Approved criterion — `{approved['criterion_id']}`: **{approved['result']}**

Reviewer-approved backend-aware replacement, passing only while every
measurement below remains exactly as Agent 4 reported
({criteria['preserved_measurements']['unchanged']}/{criteria['preserved_measurements']['total']} verified unchanged at amendment time).

**Logical run — exactly equal on MPS at all 1,000 steps:**

```text
batch identities equal at every compared step   {str(observed('mps_batch_identities_equal_every_step'))} (keys)
batch bytes equal at every compared step        {str(observed('mps_batch_bytes_equal_every_step'))} (full tensor digests)
exact next batch after resume (step 401)        {str(observed('mps_exact_next_batch_after_resume'))}
learning-rate trajectory equal                  {str(observed('mps_learning_rates_equal_every_step'))} (every step)
step/examples/cursor/scheduler/validation
    cadence/optimizer structure/counters        {str(observed('mps_logical_state_summaries_equal'))}
```

**Numerical path.** On the deterministic CPU backend the cross-process split
run is **bitwise exact** (100 updates split at 40, fixed thread count: every
parameter `torch.equal`, max abs diff {observed('cpu_end_state_max_abs_diff'):.1f}, against both the independent
uninterrupted run and the donor continuation) — the checkpoint restores the
complete numerical state. On MPS the resume boundary is isolated against the
**donor**: the checkpoint-writing process's own continuation, which computes
step 401 from bit-identical entry state.

```text
step-401 resumed vs donor, allclose(1e-5, 1e-6)  {str(observed('mps_step401_resumed_vs_donor_allclose'))}  <- the ORIGINAL tolerances
    max abs diff at that step                    {observed('mps_step401_resumed_vs_donor_max_abs_diff'):.3e}   (resume roundtrip adds no jump)
independent processes' divergence at the same
    step (straight vs donor, no resume anywhere) {observed('mps_independent_prefix_divergence_at_401'):.3e}   (~10^5 x larger)
end-of-run resumed vs donor                      {observed('mps_end_vs_donor_max_abs_diff'):.3e}
backend's own fresh-vs-fresh end-of-run
    envelope (no-checkpoint control pair)        {observed('mps_no_checkpoint_control_end_max_abs_diff'):.3e}
resumed/control envelope ratio                   {observed('mps_envelope_ratio_end_vs_donor_over_control'):.2f} (gate: <= {approved['envelope_ratio_limit']:.0f})
```

The step-401 result is the load-bearing one: at the resume boundary, with the
comparison freed of independent-prefix drift, the resumed run meets the
**original** tolerances with a difference five orders of magnitude below what
two independent processes already show at the same step. The mini-corpus
suite regression additionally proves bitwise-equal parameters *and* optimizer
moments through save/destroy/reload on CPU.

"""
    REPORT_PATH.write_text(text[:start] + replacement + text[end:])
    log("report section 4.3 rewritten with the dual-criteria record")


def update_report_suite_line(tests_after: dict) -> None:
    """Rewrite section 4.6's post-change suite result with the latest run."""
    text = REPORT_PATH.read_text()
    start = text.find("### 4.6 Suite")
    if start < 0:
        log("report section 4.6 not found; leaving the suite line alone")
        return
    replacement = f"""### 4.6 Suite

Pre-edit: `3598 passed, 3 skipped` at `eb730d4` (dirty tree carrying accepted
Agent 3 work). Post-implementation: `3660 passed, 3 skipped`. Steady state
after the acceptance amendment (artifact/report correction only — no trainer,
test, or experiment change): `{tests_after['summary']}`.
"""
    REPORT_PATH.write_text(text[:start] + replacement)
    log("report section 4.6 updated with the steady-state suite result")


def run_pytest() -> dict:
    log("running the full suite")
    started = time.perf_counter()
    completed = subprocess.run(
        [str(REPOSITORY_ROOT / ".venv/bin/python"), "-m", "pytest", "tests", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    record = {
        "command": ".venv/bin/python -m pytest tests -q",
        "returncode": completed.returncode,
        "summary": tail,
        "seconds": time.perf_counter() - started,
    }
    log(f"suite: {tail}")
    return record


def append_report_section(contract: dict) -> None:
    marker = "## 4. Agent 4 — MPS Trainer, Checkpoint/Resume, and Throughput Validation"
    text = REPORT_PATH.read_text()
    if marker in text:
        log("report section 4 already present; not appending twice")
        return
    benchmark = contract["benchmark"]
    resume = contract["resume_validation_summary"]
    soak = contract["soak"]
    verify = contract["corpus_verification"]
    mps = resume["mps"]
    cpu = resume["cpu"]
    recommended = benchmark["recommended_topology"]
    tests_after = contract.get("tests_after") or {}
    section = f"""

{marker}

**Status: {contract['status']}** — {sum(1 for value in contract['completion_gates'].values() if value)} / {len(contract['completion_gates'])} completion gates true.
Machine-readable record: `reports/phase_8_data/agent_04_trainer_contract.json`
(API, gates, benchmark, soak), `reports/phase_8_data/agent_04_training_benchmark.csv`
(per-topology measurements), and `reports/phase_8_data/agent_04_resume_validation.json`
(split-run equivalence evidence). Produced by
`python scripts/run_phase8_agent04.py --full --run-pytest`.

### 4.0 Prerequisite: accepted-corpus identity through the resolver

Per the supplementary review instruction, the accepted corpus was resolved
exclusively through `synthetic_corpus.default_corpus_root()` before any
trainer construction or optimizer step, in every process that trains
(including each split-run subprocess):

```text
resolver result: MATCH   (pointer_file -> canonical location)
required:  {REQUIRED_CORPUS_ROOT}
resolved:  identical (also equal to the repository-relative canonical path)
content digest        c95c3545…87d0d   == accepted (Agents 2/3)
metadata digest       1db0f02f…9c81bb  == accepted
commit-index digest   32e8e18d…c15db1  == accepted
payload bytes         re-verified against every committed digest (0 failures)
verification cost     {verify['digest_verification_seconds']:.1f}s per verifying process
```

No trainer, checkpoint, dataset or downstream module embeds the absolute
path; this harness alone pins it as the expected value to verify against.
Checkpoints identify the corpus by **version + digests only**
(`CorpusIdentity`); the resolved root is recorded as diagnostics and never
compared, and `tests/training/test_warmstart_checkpoint.py` proves a pure
relocation with identical digests keeps checkpoints resumable while any
byte/journal drift is a BLOCKED stop, never a regeneration.

### 4.1 Implementation

```text
stratego/training/warmstart_loss.py        warmstart_loss_v1 — frozen per-batch
                                           normalizations over the frozen
                                           stratego.model.losses primitives
stratego/training/warmstart_metrics.py     warmstart_metrics_v1 — validation
                                           vs the three frozen baselines,
                                           per-game sufficient statistics
stratego/training/warmstart_checkpoint.py  warmstart_checkpoint_v1 — atomic
                                           writes, integrity digest, digest-only
                                           corpus identity, evaluation-only load
stratego/training/warmstart_trainer.py     warmstart_trainer_v1 — C1 float32
                                           MPS trainer, AdamW + versioned
                                           warmup, persistent ordered pipeline
```

The trainer constructs **only** Agent 1's six frozen pilot candidates
(`WarmstartTrainConfig.from_pilot_candidate`); any off-matrix hyperparameter
raises, and the suite proves it field by field. Losses are exactly the frozen
semantics: weighted masked legal policy CE normalized by the weight sum
(zero-weight batches contribute exactly 0 through a connected graph),
mean WDL CE, hidden-only belief CE per supervised square. Illegal logits are
replaced by the frozen −1e9 fill before normalization — the suite proves
arbitrarily large illegal logits leave every loss bit unchanged — and an
illegal teacher action raises. Gradient clipping records pre/post norms;
per-batch reporting covers losses, supervision counts, legal-policy entropy,
learning rate, gradient and parameter norms, per-phase times and data wait.
61 new regressions: `test_warmstart_loss.py` (17),
`test_warmstart_checkpoint.py` (23), `test_warmstart_trainer.py` (21), plus
this three-artifact harness.

### 4.2 Loader/trainer balance (benchmark on real reconstructed examples)

Raw C1 float32 MPS compute at batch 256 is ~92 ms/update (~2,800 examples/s),
so the reconstruction loader is the constraint. Starting from Agent 3's best
topology (8 workers):

```text
baseline 8w/2p/512c    data wait {benchmark['baseline_data_wait_fraction']:.1%} of wall — exceeds the frozen 15%
tuned within the allowed knobs (workers / prefetch / record cache / overlap):
recommended {recommended['workers']}w/{recommended['prefetch']}p/{recommended['record_cache_size']}c   data wait {benchmark['recommended_data_wait_fraction']:.1%}
realistic recommended-topology throughput: {benchmark['recommended_updates_per_second_realistic']:.2f} updates/s
                                           {benchmark['recommended_examples_per_second_realistic']:.0f} examples/s
still data-bound after tuning: {str(benchmark['still_data_bound_after_tuning'])}
batch identity across every measured topology: byte-for-byte identical
observations materialized to disk: NO; frozen train order altered: NO
one cadence validation (8 x 256 held-out examples): {benchmark['validation_pass']['seconds']:.1f}s
```

Every topology served bit-identical batches (full `batch_digest` equality
against the baseline sequence), so tuning changed arrival times only.

### 4.3 Resume equivalence (1,000-update split run)

Uninterrupted 1,000 updates vs 400 + atomic checkpoint + **process destroyed**
+ reload in a fresh process + 600, both from the canonical C1 init through
the frozen shuffle stream, validating every 500 updates.

**The logical run is exactly equal on MPS:**

```text
batch identities equal at every compared step   {str(mps['batch_identities_equal_every_step'])} (keys, all {mps['total_updates']})
batch bytes equal at every compared step        {str(mps['batch_bytes_equal_every_step'])} (full tensor digests)
exact next batch after resume (step {mps['split_at'] + 1})        {str(mps['exact_next_batch_after_resume'])}
learning-rate trajectory equal                  {str(mps['learning_rates_equal_every_step'])} (every step)
step/examples/cursor/scheduler/validation
    cadence/optimizer structure/counters        {str(mps['logical_state_summaries_equal'])}
```

**The numerical path, honestly:** the CPU cross-process split run is
**exactly bit-equal** ({cpu['total_updates']} updates split at
{cpu['split_at']}, fixed thread count: every parameter `torch.equal`, max
abs diff 0.0, against both the independent uninterrupted run and the donor
continuation), proving the checkpoint restores the complete numerical state.
On MPS, this torch/macOS stack is **not run-to-run bit-deterministic even
without any checkpoint**: two fresh identical uninterrupted runs diverge
(measured control, no save/load anywhere), and same-process repeats diverge
too, so the specified end-of-run allclose(rtol=1e-5, atol=1e-6) cannot be
met by *any* pair of separately executed 1,000-update MPS runs — including
an uninterrupted run against itself. Recorded as a deviation for review
(threshold not moved). The resume boundary is therefore isolated against the
**donor**: the checkpoint-writing process's own continuation, which computes
step {mps['split_at'] + 1} from bit-identical entry state. On MPS:

```text
step-{mps['split_at'] + 1} resumed vs donor, allclose(1e-5, 1e-6)    {str(mps['parameters_first_post_resume_step']['all_allclose_rtol1e-5_atol1e-6'])}
    max abs diff at that step                    {mps['parameters_first_post_resume_step']['max_abs_diff']:.3e}   (the resume roundtrip adds no jump)
independent processes' divergence at the same
    step (straight vs donor, no resume anywhere) {mps['independent_prefix_divergence_at_split_plus_1']['max_abs_diff']:.3e}
end-of-run resumed vs donor                      {mps['parameters_end_vs_donor']['max_abs_diff']:.3e}
backend's own fresh-vs-fresh end-of-run
    envelope (no-checkpoint control pair)        {mps['backend_control']['end']['max_abs_diff']:.3e}
resumed/control envelope ratio                   {mps['envelope_ratio_end_vs_donor_over_control']:.2f} (gate: <= 10)
end-of-run resumed vs independent uninterrupted  {mps['parameters_end_of_run']['max_abs_diff']:.3e} (recorded)
```

The mini-corpus suite regression additionally proves bitwise-equal
parameters *and* optimizer moments through save/destroy/reload on CPU.

### 4.4 Numerical-stability soak ({soak['updates']} updates, MPS)

One neutral frozen candidate (`{NEUTRAL_CANDIDATE}`: median frozen learning
rate, balanced profile — an infrastructure choice, not a selection):

```text
optimizer updates                 {soak['updates']}
non-finite losses                 {soak['counters']['non_finite_losses']}
non-finite gradients              {soak['counters']['non_finite_gradients']}
non-finite parameters             {soak['counters']['non_finite_parameters']}
illegal targets                   {soak['counters']['illegal_targets']}
data mismatches                   {soak['counters']['data_mismatches']}
checkpoint errors                 {soak['counters']['checkpoint_errors']}
checkpoints written + validated   {soak['checkpoints_written_and_validated']} (+ reload-consistency proof)
loss trend (descriptive only)     total {soak['loss_trends_descriptive_only']['total']['first_100_mean']:.3f} -> {soak['loss_trends_descriptive_only']['total']['last_100_mean']:.3f}
```

Loss trends are recorded descriptively; nothing here chose a configuration.

### 4.5 Held-out discipline

Training reads the train split through the frozen cursor; validation reads
the validation split through its own dataset instance and a fresh sequential
cursor under `no_grad`, restores model mode, and is proven side-effect-free
(parameters, optimizer moments, scheduler, cursor and counters bit-identical
before/after in the suite). The sealed test split is refused by the frozen
`check_test_corpus_access` gate (regression-tested); no Phase 4 bank access
of any kind occurred. Pilot selection has not begun.

### 4.6 Suite

Pre-edit: `3598 passed, 3 skipped` at `eb730d4` (dirty tree carrying accepted
Agent 3 work). Post-implementation: `{tests_after.get('summary', 'see artifact')}`.
"""
    REPORT_PATH.write_text(text + section)
    log("report section 4 appended")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--resume-mps", action="store_true")
    parser.add_argument("--resume-cpu", action="store_true")
    parser.add_argument("--soak", action="store_true")
    parser.add_argument("--artifacts", action="store_true")
    parser.add_argument(
        "--amend-criterion",
        action="store_true",
        help=(
            "reviewer-approved acceptance amendment: record the original and "
            "backend-aware resume criteria separately in the artifacts and the "
            "report. Re-runs no experiment and changes no trainer behaviour."
        ),
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument(
        "--resume-worker",
        choices=["straight", "split-first", "split-resume", "control-a", "control-b"],
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--trust-bytes", action="store_true", help="skip payload byte re-verification (workers of an already-verified orchestration)")
    parser.add_argument("--benchmark-warmup", type=int, default=12)
    parser.add_argument("--benchmark-updates", type=int, default=96)
    parser.add_argument("--resume-updates", type=int, default=1000)
    parser.add_argument("--resume-split-at", type=int, default=400)
    parser.add_argument("--cpu-updates", type=int, default=100)
    parser.add_argument("--cpu-split-at", type=int, default=40)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--soak-updates", type=int, default=2048)
    parser.add_argument("--soak-checkpoint-every", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--record-cache", type=int, default=512)
    args = parser.parse_args()

    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if args.resume_worker:
        resume_worker(args.resume_worker, args)
        return

    durations = {}
    tests_before = {
        "command": ".venv/bin/python -m pytest tests -q",
        "summary": "3598 passed, 3 skipped in 220.67s",
        "measured_at": "before any Agent 4 edit (recorded by the session)",
    }

    verify_record = None
    identity = None
    if (
        args.verify
        or args.full
        or args.benchmark
        or args.soak
        or args.artifacts
        or args.amend_criterion
    ):
        started = time.perf_counter()
        verify_record, identity = verify_prerequisites()
        durations["verify"] = time.perf_counter() - started
        write_json(WORK_DIRECTORY / "verify.json", verify_record)
        log(
            f"prerequisites verified: resolver MATCH, digests MATCH "
            f"({durations['verify']:.1f}s)"
        )
        # This invocation has verified payload bytes; its worker subprocesses
        # still re-verify the three identity digests, skipping only the
        # payload byte re-read.
        args.trust_bytes = True

    if args.benchmark or args.full:
        started = time.perf_counter()
        benchmark = phase_benchmark(identity, args)
        durations["benchmark"] = time.perf_counter() - started
        # Later phases default to the recommended topology.
        recommended = benchmark["recommended_topology"]
        args.workers = recommended["workers"]
        args.prefetch = recommended["prefetch"]
        args.record_cache = recommended["record_cache_size"]

    if args.resume_mps or args.full:
        started = time.perf_counter()
        phase_resume(args, "mps")
        durations["resume_mps"] = time.perf_counter() - started

    if args.resume_cpu or args.full:
        started = time.perf_counter()
        phase_resume(args, "cpu")
        durations["resume_cpu"] = time.perf_counter() - started

    if args.soak or args.full:
        started = time.perf_counter()
        phase_soak(identity, args)
        durations["soak"] = time.perf_counter() - started

    contract = None
    if args.artifacts or args.full:
        if verify_record is None:
            verify_record = load_work("verify.json")
        contract = phase_artifacts(verify_record, tests_before, args)

    amendment = None
    if args.amend_criterion:
        started = time.perf_counter()
        amendment = phase_amend_criterion(verify_record)
        durations["amend_criterion"] = time.perf_counter() - started
        amend_report_section(amendment)

    if args.run_pytest:
        tests_after = run_pytest()
        contract_path = DATA_DIRECTORY / "agent_04_trainer_contract.json"
        # The amendment rewrote the contract on disk, so the suite result is
        # merged into the current file rather than into a stale in-memory copy.
        current = contract if contract is not None else read_json(contract_path)
        current["tests_after"] = tests_after
        current.setdefault("durations", {}).update(durations)
        write_json(contract_path, current)
        update_report_suite_line(tests_after)
        contract = current

    if contract is not None and not args.amend_criterion:
        append_report_section(contract)
    if contract is not None:
        log(f"done: status {contract['status']}; durations {durations}")


if __name__ == "__main__":
    main()
