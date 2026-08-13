#!/usr/bin/env python3
"""Phase 7 Agent 5 acceptance harness: production pipeline integration.

Connects the accepted `setup_library_v1` + `setup_sampler_v1` sampler to the
frozen Phase 6 collection/persistence pipeline and proves the connection with
a real campaign rather than an argument:

    1. Agents 1-4 PASS and the library/sampler digests match the handoff.
    2. The Phase 4 `evaluation_setup_bank_v1` identity is captured *before*
       anything runs, and again afterwards.
    3. A deterministic integration campaign runs through the real
       MPS-coordinator / CPU-worker / compressed-shard path with the train
       split as the setup source, until it has enough completed games and
       enough coverage of the 256 ordered family pairs.
    4. Every persisted trajectory is decoded and checked against its
       provenance sidecar: split, family, base identity, fingerprint and the
       exact setup bytes the engine received.
    5. A meaningful decision sample is reconstructed from the persisted
       records.
    6. Validation and test access are exercised as separate explicit requests
       that never touch the training campaign.
    7. Setup assignment is shown to be independent of worker count, schedule
       and a recycle boundary.
    8. The sampler's cost is measured.

Writes:

    reports/phase_7_data/agent_05_pipeline_integration.json
    reports/phase_7_data/agent_05_setup_provenance.csv

What this script is and is not
------------------------------
It is an acceptance instrument. It performs no optimizer step, no learning
update and no playing-strength interpretation: the C1 weights are the frozen
deterministic Phase 6 construction and are never written to. It does not
modify the base library, the family contracts, the sampler semantics, the
frozen engine, `trajectory_v1`, `observation_v2_1_127ch` or the Phase 4
evaluation bank.

Usage::

    python scripts/run_phase7_agent05.py                     # full campaign
    python scripts/run_phase7_agent05.py --run-pytest        # also run the suite
    python scripts/run_phase7_agent05.py --smoke             # fast shape check
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

from stratego.engine.constants import (  # noqa: E402
    IMPLEMENTATION_VERSION,
    OBSERVATION_CHANNELS,
    OBSERVATION_VERSION,
    PLAYERS,
    RULES_VERSION,
)
from stratego.engine.setup import deserialize_setup  # noqa: E402
from stratego.evaluation.setup_bank import (  # noqa: E402
    DEFAULT_BANK_ROOT_SEED,
    DEFAULT_BANK_SIZE,
    SETUP_BANK_VERSION,
    SetupBank,
    bank_digest,
    validate_bank,
)
from stratego.setups import (  # noqa: E402
    FAMILY_IDS,
    LIBRARY_JSONL_PATH,
    LIBRARY_MANIFEST_PATH,
    SETUP_FAMILY_VERSION,
    SETUP_GENERATOR_CONTRACT_VERSION,
    SETUP_LIBRARY_VERSION,
    SETUP_TRAIT_VECTOR_VERSION,
    read_library_jsonl,
    read_manifest,
)
from stratego.setups.contracts import TRAIN_PER_FAMILY  # noqa: E402
from stratego.setups.library import library_content_digest, manifest_digest  # noqa: E402
from stratego.setups.sampler import (  # noqa: E402
    SAMPLER_VERSION,
    load_library_index,
    rebuild_from_provenance,
)
from stratego.model.contract import MODEL_CONTRACT_VERSION  # noqa: E402
from stratego.training.batch_simulation import BatchSimulator  # noqa: E402
from stratego.training.coordinator import (  # noqa: E402
    SelfPlayCoordinator,
    mps_memory_bytes,
)
from stratego.training.end_to_end_benchmark import swap_bytes  # noqa: E402
from stratego.training.phase6_pipeline_benchmark import (  # noqa: E402
    BYTES_PER_GIB,
    build_pipeline_candidate,
    classify_failure,
    empty_failure_counts,
)
from stratego.training.phase6b_recording import (  # noqa: E402
    recording_configuration,
    system_memory,
)
from stratego.training.phase6b_recycle import segment_root_seed  # noqa: E402
from stratego.training.phase6_soak import (  # noqa: E402
    SOAK_ENVIRONMENTS,
    SOAK_INFERENCE_BATCH,
    SOAK_WORKERS,
    probe_live_finiteness,
)
from stratego.training.reconstruction import (  # noqa: E402
    iter_reconstructed_decisions,
)
from stratego.training.serialization import decompress  # noqa: E402
from stratego.training.setup_source import (  # noqa: E402
    PROVENANCE_PLAYER_EXTRA_FIELDS,
    PROVENANCE_RECORD_FIELDS,
    PROVENANCE_SCHEMA_VERSION,
    REQUIRED_PLAYER_PROVENANCE_FIELDS,
    SETUP_SOURCE_VERSION,
    TRAINING_SPLIT,
    audit_setup_source,
    describe_setup_source,
    family_pair,
    iter_provenance_records,
    read_provenance_index,
    training_setup_source,
    validate_provenance_record,
    verify_provenance_against_setups,
    verify_provenance_split,
)
from stratego.training.shard_writer import (  # noqa: E402
    directory_summary,
    iter_shard_payloads,
    read_shard_header,
    shard_paths,
)
from stratego.training.trajectory import (  # noqa: E402
    TRAJECTORY_VERSION,
    decode_game_record,
    validate_game_record,
)
from stratego.training.worker_pool import RecordingConfig, WorkerPool  # noqa: E402

AGENT = 5
PHASE = 7
INTEGRATION_VERSION = "phase7_agent05_integration_0.1.0"

REPORT_DATA = REPOSITORY_ROOT / "reports" / "phase_7_data"
AGENT_01_CONTRACT = REPORT_DATA / "agent_01_setup_contract.json"
AGENT_02_MANIFEST = REPORT_DATA / "agent_02_base_library_manifest.json"
AGENT_03_AUDIT = REPORT_DATA / "agent_03_library_audit.json"
AGENT_04_SAMPLER = REPORT_DATA / "agent_04_sampler_contract.json"
AGENT_04_STRESS = REPORT_DATA / "agent_04_procedural_stress.json"

INTEGRATION_JSON = REPORT_DATA / "agent_05_pipeline_integration.json"
PROVENANCE_CSV = REPORT_DATA / "agent_05_setup_provenance.csv"

LIBRARY_PATH = REPOSITORY_ROOT / LIBRARY_JSONL_PATH
MANIFEST_PATH = REPOSITORY_ROOT / LIBRARY_MANIFEST_PATH

#: The frozen Phase 6 production topology. Not retuned for Phase 7.
CAMPAIGN_WORKERS = SOAK_WORKERS
CAMPAIGN_ENVIRONMENTS = SOAK_ENVIRONMENTS
CAMPAIGN_INFERENCE_BATCH = SOAK_INFERENCE_BATCH
CAMPAIGN_CANDIDATE = "C1"
CAMPAIGN_ROOT_SEED = 70_005
CAMPAIGN_RUN_ID = "p7a05"

#: Assignment targets.
MINIMUM_GAMES = 4_096
ORDERED_FAMILY_PAIRS = len(FAMILY_IDS) * len(FAMILY_IDS)
MINIMUM_GAMES_PER_FAMILY_PAIR = 16
MINIMUM_RECONSTRUCTED_DECISIONS = 10_000

#: Per-worker streaming-verification budget for the campaign. Large enough to
#: exercise the live-digest path throughout the run, small enough that the
#: campaign is not dominated by it; the offline reconstruction pass below is
#: the independent check on what actually reached the disk.
CAMPAIGN_VERIFY_DECISIONS = 100_000

#: Safety rails for the campaign loop.
CAMPAIGN_MAX_STEPS = 60_000
CAMPAIGN_MAX_SECONDS = 3 * 3600.0
COVERAGE_CHECK_EVERY_STEPS = 250

#: The split smoke requests. Deliberately tiny, in their own directories, and
#: never merged into the training campaign.
SMOKE_ENVIRONMENTS = 8
SMOKE_WORKERS = 2
SMOKE_STEPS = 1_500
SMOKE_ROOT_SEED = 70_105

#: Determinism probe. Small enough to run three times, long enough that games
#: complete and provenance exists to compare.
DETERMINISM_ENVIRONMENTS = 32
DETERMINISM_STEPS = 1_400
DETERMINISM_ROOT_SEED = 70_205


# ---------------------------------------------------------------------------
# Environment and prerequisites
# ---------------------------------------------------------------------------


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _environment() -> dict:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "numpy_version": np.__version__,
    }


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _verify_prerequisites() -> dict:
    """Agents 1-4 PASS plus the exact library/sampler digests, before any work."""
    problems: list[str] = []
    statuses: dict = {}
    for name, path in (
        ("agent_01_setup_contract", AGENT_01_CONTRACT),
        ("agent_02_base_library_manifest", AGENT_02_MANIFEST),
        ("agent_03_library_audit", AGENT_03_AUDIT),
        ("agent_04_sampler_contract", AGENT_04_SAMPLER),
        ("agent_04_procedural_stress", AGENT_04_STRESS),
    ):
        if not path.exists():
            problems.append(f"missing prerequisite artifact: {path.name}")
            statuses[name] = None
            continue
        payload = json.loads(path.read_text())
        statuses[name] = payload.get("status")
        if payload.get("status") != "PASS":
            problems.append(f"{name} status is {payload.get('status')!r}, not PASS")

    audit = json.loads(AGENT_03_AUDIT.read_text()) if AGENT_03_AUDIT.exists() else {}
    sampler = json.loads(AGENT_04_SAMPLER.read_text()) if AGENT_04_SAMPLER.exists() else {}
    entries = read_library_jsonl(LIBRARY_PATH)
    manifest = read_manifest(MANIFEST_PATH)
    observed = {
        "library_digest": library_content_digest(entries),
        "manifest_digest": manifest_digest(manifest),
        "entry_count": len(entries),
        "sampler_version": SAMPLER_VERSION,
    }
    expected = {
        "library_digest": audit.get("library_digest"),
        "manifest_digest": audit.get("manifest_digest"),
        "entry_count": audit.get("setup_count"),
        "sampler_version": (sampler.get("sampler_contract") or {}).get(
            "sampler_version", sampler.get("sampler_version")
        ),
    }
    for key, value in expected.items():
        if value is not None and observed[key] != value:
            problems.append(
                f"{key} changed since handoff: expected {value!r}, observed "
                f"{observed[key]!r}"
            )
    if sampler.get("library_digest") not in (None, observed["library_digest"]):
        problems.append("the sampler contract names a different library digest")

    return {
        "statuses": statuses,
        "expected": expected,
        "observed": observed,
        "handoff_unchanged": not problems,
        "problems": problems,
    }


def _phase_4_bank_identity() -> dict:
    """`evaluation_setup_bank_v1` identity, reproduced from its own generator."""
    bank = SetupBank.generate(size=DEFAULT_BANK_SIZE, root_seed=DEFAULT_BANK_ROOT_SEED)
    validation = validate_bank(bank)
    return {
        "bank_version": bank.bank_version,
        "declared_version": SETUP_BANK_VERSION,
        "count": len(bank),
        "root_seed": DEFAULT_BANK_ROOT_SEED,
        "digest": bank_digest(bank),
        "generation_family": validation["generation_family"],
        "pair_count": int(validation["pair_count"]),
        "distinct_positions": int(validation["distinct_positions"]),
        "validation_failure_count": int(validation["validation_failure_count"]),
        "duplicate_setup_pair_ids": list(validation["duplicate_setup_pair_ids"]),
    }


# ---------------------------------------------------------------------------
# The setup-source API document
# ---------------------------------------------------------------------------


def _setup_source_api(source) -> dict:
    return {
        "setup_source_version": SETUP_SOURCE_VERSION,
        "module": "stratego/training/setup_source.py",
        "injection_point": (
            "BatchSimulator(setup_source=...) -> consulted once per created "
            "game in _build_slot; nothing after game creation reads it"
        ),
        "plumbing": [
            "CoordinatorConfig.setup_source",
            "SelfPlayCoordinator -> WorkerPool(setup_source=...)",
            "WorkerPool.start -> _worker_main(..., setup_source)",
            "_WorkerRuntime -> BatchSimulator(setup_source=...)",
        ],
        "entry_points": {
            "training": "training_setup_source(profile) -> LibrarySetupSource(split='train')",
            "audit": "audit_setup_source(split, justification, profile)",
            "default": "None / UniformRandomSetupSource() -> frozen Phase 6 behaviour",
        },
        "interface": (
            "assign(root_seed, environment_id, generation, slot_seed, game_id) "
            "-> SetupAssignment(red_setup, blue_setup, provenance)"
        ),
        "training_source": describe_setup_source(source),
        "split_rule": (
            "purpose='training' is locked to split='train'; a validation or "
            "test source requires purpose='evaluation_audit' plus a non-empty "
            "access_justification, checked at construction"
        ),
        "provenance_sidecar": {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "path": "<output_directory>/<run_id>_w<NN>_setup_provenance.jsonl",
            "record_fields": list(PROVENANCE_RECORD_FIELDS),
            "player_integration_fields": list(PROVENANCE_PLAYER_EXTRA_FIELDS),
            "player_required_fields": list(REQUIRED_PLAYER_PROVENANCE_FIELDS),
            "player_sampler_fields": (
                "the complete frozen setup_sampler_v1 provenance record, stored "
                "verbatim so rebuild_from_provenance reconstructs the sample"
            ),
            "trajectory_v1_changed": False,
            "note": (
                "a sibling file of the shards, never part of one; a Phase 6 "
                "shard remains readable by everything that could read it before"
            ),
        },
        "frozen_versions_untouched": {
            "trajectory_version": TRAJECTORY_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "observation_channels": OBSERVATION_CHANNELS,
            "model_contract_version": MODEL_CONTRACT_VERSION,
            "rules_version": RULES_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
        },
    }


# ---------------------------------------------------------------------------
# Sidecar coverage
# ---------------------------------------------------------------------------


def _coverage(directory) -> dict:
    """Family-pair coverage of everything sealed so far."""
    pairs: Counter = Counter()
    games = 0
    for record in iter_provenance_records(directory):
        pairs[family_pair(record)] += 1
        games += 1
    return {"games": games, "pairs": pairs}


def _coverage_summary(pairs: Counter) -> dict:
    counts = [pairs.get((red, blue), 0) for red in FAMILY_IDS for blue in FAMILY_IDS]
    return {
        "ordered_pairs_possible": ORDERED_FAMILY_PAIRS,
        "ordered_pairs_seen": sum(1 for count in counts if count),
        "ordered_pairs_missing": sum(1 for count in counts if not count),
        "minimum_games_per_pair": min(counts) if counts else 0,
        "maximum_games_per_pair": max(counts) if counts else 0,
        "mean_games_per_pair": (sum(counts) / len(counts)) if counts else 0.0,
        "pairs_below_target": sum(
            1 for count in counts if count < MINIMUM_GAMES_PER_FAMILY_PAIR
        ),
        "target_games_per_pair": MINIMUM_GAMES_PER_FAMILY_PAIR,
    }


# ---------------------------------------------------------------------------
# The integration campaign
# ---------------------------------------------------------------------------


def run_campaign(
    directory: Path,
    *,
    minimum_games: int,
    minimum_pairs: int,
    minimum_per_pair: int,
    max_steps: int,
    max_seconds: float,
    verify_target_decisions: int,
    environments: int,
    workers: int,
    inference_batch: int,
    progress=None,
) -> dict:
    """Drive the real collection path until the coverage targets are met."""
    source = training_setup_source()
    config = recording_configuration(
        CAMPAIGN_CANDIDATE,
        output_directory=str(directory),
        run_id=CAMPAIGN_RUN_ID,
        compress=True,
        workers=workers,
        environments=environments,
        inference_batch_size=inference_batch,
        root_seed=CAMPAIGN_ROOT_SEED,
        verify_target_decisions=verify_target_decisions,
        max_concurrent_verifications=1,
    )
    config = replace(config, setup_source=source, detailed_timing=False)

    model = build_pipeline_candidate(CAMPAIGN_CANDIDATE)
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    coordinator = SelfPlayCoordinator(
        config, device=None, model=model, model_label=CAMPAIGN_CANDIDATE
    )
    # Taken after the coordinator has moved and cast the model, so the two
    # checksums are comparable and any weight change during the run shows up.
    checksum_before = parameter_checksum(coordinator.model)
    gradients_enabled_before = any(
        parameter.requires_grad for parameter in coordinator.model.parameters()
    )

    failures = empty_failure_counts()
    status = "ok"
    error_text = None
    swap_start = swap_bytes()
    disk_free_start = int(shutil.disk_usage(str(directory)).free)
    samples: list[dict] = []
    coverage = {"games": 0, "pairs": Counter()}
    stop_reason = "targets met"
    probe_logits = 0

    coordinator.start()
    started = time.perf_counter()
    try:
        while True:
            coordinator.step()
            elapsed = time.perf_counter() - started
            step = coordinator.step_index

            if step % COVERAGE_CHECK_EVERY_STEPS:
                if elapsed < max_seconds and step < max_steps:
                    continue

            probe = probe_live_finiteness(coordinator, rows=512)
            probe_logits += probe["logits_checked"]
            failures["nonfinite_outputs"] += probe["nonfinite_outputs"]

            totals = coordinator.pool.recording_totals()
            if int(totals["total_write_errors"]):
                raise RuntimeError("shard write error reported by a worker")
            if int(totals["total_reconstruction_mismatches"]):
                raise RuntimeError("streaming verification reported a mismatch")
            if int(totals.get("total_provenance_write_errors", 0)):
                raise RuntimeError("a provenance sidecar write failed")

            coverage = _coverage(directory)
            summary = _coverage_summary(coverage["pairs"])
            sample = {
                "step": step,
                "elapsed_seconds": elapsed,
                "positions": int(coordinator.totals.positions),
                "games_finished": int(coordinator.games_finished),
                "games_sealed": int(totals["total_games_recorded"]),
                "provenance_records": int(totals.get("total_provenance_records", 0)),
                "decisions_recorded": int(totals["total_decisions_recorded"]),
                "verified_decisions": int(totals["total_verified_decisions"]),
                "bytes_written": int(totals["total_persisted_bytes"]),
                "ordered_pairs_seen": summary["ordered_pairs_seen"],
                "minimum_games_per_pair": summary["minimum_games_per_pair"],
                "positions_per_second": int(coordinator.totals.positions) / elapsed,
                "games_per_second": coverage["games"] / elapsed,
            }
            samples.append(sample)
            if progress is not None:
                progress(sample)

            enough_games = coverage["games"] >= minimum_games
            enough_coverage = (
                summary["ordered_pairs_seen"] >= minimum_pairs
                and summary["minimum_games_per_pair"] >= minimum_per_pair
            )
            if enough_games and enough_coverage:
                break
            if step >= max_steps:
                stop_reason = "step limit reached"
                break
            if elapsed >= max_seconds:
                stop_reason = "time limit reached"
                break
    except BaseException as error:  # noqa: BLE001 - a failed campaign is a result
        status = "error"
        error_text = f"{type(error).__name__}: {error}"
        failures[classify_failure(error)] = failures.get(classify_failure(error), 0) + 1
        stop_reason = "aborted"
    finally:
        try:
            pool_totals = coordinator.pool.recording_totals()
        except Exception:  # noqa: BLE001
            pool_totals = {}
        checksum_after = parameter_checksum(coordinator.model)
        shutdown_totals = coordinator.shutdown()

    wall_seconds = time.perf_counter() - started
    totals = {**pool_totals, **shutdown_totals}
    latencies = [
        micros / 1e6 for micros in totals.get("setup_source_latency_micros", ())
    ]
    disk_free_end = int(shutil.disk_usage(str(directory)).free)

    return {
        "status": status,
        "error": error_text,
        "stop_reason": stop_reason,
        "configuration": config.as_dict(),
        "candidate": {
            "candidate_id": CAMPAIGN_CANDIDATE,
            "parameters": parameters,
            "optimizer_steps": 0,
            "loss_evaluations": 0,
            "gradients_enabled": gradients_enabled_before,
            "parameter_checksum_before": checksum_before,
            "parameter_checksum_after": checksum_after,
            "weights_mutated": checksum_before != checksum_after,
        },
        "root_seed": CAMPAIGN_ROOT_SEED,
        "run_id": CAMPAIGN_RUN_ID,
        "directory": str(directory),
        "wall_seconds": wall_seconds,
        "steps": int(coordinator.step_index),
        "samples": samples,
        "failures": failures,
        "totals": {
            key: value for key, value in totals.items() if isinstance(value, (int, float))
        },
        "setup_latencies_seconds": latencies,
        "finiteness_probe_logits": probe_logits,
        "terminal_reason_counts": dict(coordinator.terminal_reason_counts),
        "swap": {
            "used_at_start": int(swap_start.get("swap_used_bytes", 0)),
            "used_at_end": int(swap_bytes().get("swap_used_bytes", 0)),
        },
        "disk": {
            "free_bytes_at_start": disk_free_start,
            "free_bytes_at_end": disk_free_end,
            "consumed_bytes": disk_free_start - disk_free_end,
        },
        "memory": {
            "coordinator_rss_bytes": int(psutil.Process().memory_info().rss),
            "peak_rss_bytes": _peak_rss_bytes(),
            "worker_max_rss_bytes": int(totals.get("worker_max_rss_bytes", 0)),
            "metal": mps_memory_bytes(),
            "system": system_memory(),
        },
    }


def parameter_checksum(model) -> float:
    """A cheap, order-stable checksum over the model's parameters.

    Recorded before and after the campaign so "the weights were never written
    to" is a measurement rather than a claim.
    """
    with torch.no_grad():
        return float(
            sum(
                float(parameter.detach().float().sum().item())
                for parameter in model.parameters()
            )
        )


# ---------------------------------------------------------------------------
# Offline verification of what reached the disk
# ---------------------------------------------------------------------------


def verify_persisted_corpus(
    directory: Path,
    *,
    expected_split: str,
    reconstruct_target: int,
    progress=None,
) -> dict:
    """Decode every persisted record and check it against its provenance.

    Streaming: a record is decoded, verified and dropped, so a corpus larger
    than memory can be checked. Everything reported here comes from the bytes
    on disk, not from the run's own counters.
    """
    index = read_provenance_index(directory)
    library = load_library_index()

    counters = {
        "records_decoded": 0,
        "zero_decision_records": 0,
        "decode_failures": 0,
        "record_validation_failures": 0,
        "provenance_missing": 0,
        "provenance_schema_failures": 0,
        "provenance_mismatches": 0,
        "split_violations": 0,
        "family_identity_mismatches": 0,
        "base_identity_mismatches": 0,
        "fingerprint_mismatches": 0,
        "duplicate_game_ids": 0,
        "setup_engine_validation_failures": 0,
        "decisions_total": 0,
        "reflection_true": 0,
        "reflection_false": 0,
        "perturbation_true": 0,
        "perturbation_false": 0,
    }
    problems: list[str] = []
    seen_game_ids: set[str] = set()
    pairs: Counter = Counter()
    families: Counter = Counter()
    bases: set[str] = set()
    splits: Counter = Counter()
    rows: list[dict] = []
    game_lengths: list[int] = []
    reconstruct_candidates: list[tuple[str, Path, int]] = []

    for path in shard_paths(directory):
        header = read_shard_header(path)
        compressed = bool(header.get("compressed", False))
        for offset, payload in enumerate(iter_shard_payloads(path)):
            try:
                record = decode_game_record(
                    decompress(payload) if compressed else payload
                )
            except Exception as error:  # noqa: BLE001 - a bad record is a finding
                counters["decode_failures"] += 1
                problems.append(f"{path.name}#{offset}: {type(error).__name__}: {error}")
                continue
            counters["records_decoded"] += 1
            if validate_game_record(record):
                counters["record_validation_failures"] += 1
                problems.append(f"{record.game_id}: record validation failed")
            if record.game_id in seen_game_ids:
                counters["duplicate_game_ids"] += 1
                problems.append(f"{record.game_id}: duplicate game id")
            seen_game_ids.add(record.game_id)
            game_lengths.append(len(record.actions))
            counters["decisions_total"] += len(record.decisions)
            if not record.decisions:
                counters["zero_decision_records"] += 1

            provenance = index.get(record.game_id)
            if provenance is None:
                counters["provenance_missing"] += 1
                problems.append(f"{record.game_id}: no provenance record")
                continue

            schema = validate_provenance_record(provenance)
            if schema:
                counters["provenance_schema_failures"] += 1
                problems.append(f"{record.game_id}: {schema[0]}")
            split = verify_provenance_split(provenance, expected_split)
            if split:
                counters["split_violations"] += 1
                problems.append(f"{record.game_id}: {split[0]}")

            mismatches = verify_provenance_against_setups(
                provenance,
                red_setup=record.red_setup,
                blue_setup=record.blue_setup,
            )
            if mismatches:
                counters["provenance_mismatches"] += 1
                problems.append(f"{record.game_id}: {mismatches[0]}")

            for side, player, stored in (
                ("red", PLAYERS[0], record.red_setup),
                ("blue", PLAYERS[1], record.blue_setup),
            ):
                entry = provenance[side]
                base = library.base(entry["base_setup_id"])
                if base.family_id != entry["primary_family_id"]:
                    counters["family_identity_mismatches"] += 1
                if base.split != entry["split"]:
                    counters["base_identity_mismatches"] += 1
                if entry["base_index"] >= TRAIN_PER_FAMILY and (
                    expected_split == TRAINING_SPLIT
                ):
                    counters["split_violations"] += 1
                rebuilt = rebuild_from_provenance(entry, index=library)
                if rebuilt.oriented(player) != stored:
                    counters["fingerprint_mismatches"] += 1
                if deserialize_setup(entry["engine_setup"]) != stored:
                    counters["setup_engine_validation_failures"] += 1
                families[entry["primary_family_id"]] += 1
                bases.add(entry["base_setup_id"])
                splits[entry["split"]] += 1
                counters[
                    "reflection_true" if entry["reflection_applied"] else "reflection_false"
                ] += 1
                counters[
                    "perturbation_true"
                    if entry["perturbation_applied"]
                    else "perturbation_false"
                ] += 1

            pairs[family_pair(provenance)] += 1
            rows.append(_provenance_row(record, provenance))
            reconstruct_candidates.append((record.game_id, path, offset))
            if progress is not None and counters["records_decoded"] % 2000 == 0:
                progress(counters["records_decoded"])

    orphans = set(index) - seen_game_ids
    if orphans:
        problems.append(f"{len(orphans)} provenance records name no persisted game")

    reconstruction = _reconstruct_sample(
        directory, reconstruct_candidates, index, target=reconstruct_target
    )

    return {
        "counters": counters,
        "problems": problems[:50],
        "problem_count": len(problems),
        "orphan_provenance_records": len(orphans),
        "distinct_games": len(seen_game_ids),
        "distinct_bases_used": len(bases),
        "family_counts": dict(sorted(families.items())),
        "split_counts": dict(splits),
        "coverage": _coverage_summary(pairs),
        "family_pair_counts": {
            f"{red}->{blue}": pairs.get((red, blue), 0)
            for red in FAMILY_IDS
            for blue in FAMILY_IDS
        },
        "game_length": {
            "mean": statistics.fmean(game_lengths) if game_lengths else 0.0,
            "minimum": min(game_lengths) if game_lengths else 0,
            "maximum": max(game_lengths) if game_lengths else 0,
        },
        "reconstruction": reconstruction,
        "rows": rows,
    }


def _provenance_row(record, provenance: dict) -> dict:
    """One CSV row: the game plus both players' provenance."""
    row = {
        "game_id": record.game_id,
        "environment_id": record.environment_id,
        "generation": record.generation,
        "run_id": provenance["run_id"],
        "worker_id": provenance["worker_id"],
        "split": provenance["split"],
        "sampler_profile": provenance["sampler_profile"],
        "setup_library_version": provenance["setup_library_version"],
        "sampler_version": provenance["sampler_version"],
        "final_ply": record.final_ply,
        "terminal_reason": record.terminal_reason,
        "decisions": len(record.decisions),
        "trajectory_setup_id": record.setup_id,
        "setup_family_label": record.setup_family,
    }
    for side in ("red", "blue"):
        entry = provenance[side]
        row.update(
            {
                f"{side}_primary_family_id": entry["primary_family_id"],
                f"{side}_base_setup_id": entry["base_setup_id"],
                f"{side}_base_index": entry["base_index"],
                f"{side}_split": entry["split"],
                f"{side}_reflection_applied": int(bool(entry["reflection_applied"])),
                f"{side}_perturbation_applied": int(bool(entry["perturbation_applied"])),
                f"{side}_perturbation_seed": entry["perturbation_seed"],
                f"{side}_perturbation_id": entry["perturbation_id"],
                f"{side}_final_setup_fingerprint": entry["final_setup_fingerprint"],
                f"{side}_side_seed": entry["side_seed"],
            }
        )
    return row


def _reconstruct_sample(directory, candidates, index, *, target: int) -> dict:
    """Reconstruct decisions from persisted records until `target` is reached.

    Games are taken evenly across the corpus rather than from its head, so the
    sample is not one worker's shard. Every sampled game's setup/provenance
    agreement is re-verified here as well, which is what the assignment asks
    for on the reconstructed sample specifically.
    """
    if not candidates:
        return {"games": 0, "decisions": 0, "mismatches": 0, "setup_mismatches": 0}
    by_path: dict = {}
    stride = max(1, len(candidates) // 64)
    chosen = candidates[::stride]

    decisions = 0
    games = 0
    mismatches = 0
    setup_mismatches = 0
    zero_decision_games = 0
    details: list[str] = []
    started = time.perf_counter()

    for game_id, path, offset in chosen:
        if decisions >= target:
            break
        payloads = by_path.get(path)
        if payloads is None:
            header = read_shard_header(path)
            payloads = (bool(header.get("compressed", False)), list(iter_shard_payloads(path)))
            by_path = {path: payloads}
        compressed, blobs = payloads
        record = decode_game_record(
            decompress(blobs[offset]) if compressed else blobs[offset]
        )
        provenance = index[record.game_id]
        if verify_provenance_against_setups(
            provenance, red_setup=record.red_setup, blue_setup=record.blue_setup
        ):
            setup_mismatches += 1
            details.append(f"{record.game_id}: setup/provenance mismatch")
        games += 1
        if not record.decisions:
            zero_decision_games += 1
            continue
        for rebuilt in iter_reconstructed_decisions(
            record, dense_mask=False, include_public_knowledge=False, copy_state=False
        ):
            stored = record.decision_at(rebuilt.ply)
            if tuple(rebuilt.legal_action_ids) != tuple(stored.legal_action_ids):
                mismatches += 1
                details.append(f"{record.game_id}#{rebuilt.ply}: legal set differs")
            if rebuilt.acting_player != stored.acting_player:
                mismatches += 1
                details.append(f"{record.game_id}#{rebuilt.ply}: acting player differs")
            if stored.selected_action_id not in rebuilt.legal_action_ids:
                mismatches += 1
                details.append(f"{record.game_id}#{rebuilt.ply}: stored action illegal")
            decisions += 1

    return {
        "games": games,
        "decisions": decisions,
        "zero_decision_games_included": zero_decision_games,
        "mismatches": mismatches,
        "setup_mismatches": setup_mismatches,
        "details": details[:20],
        "seconds": time.perf_counter() - started,
        "target": target,
    }


# ---------------------------------------------------------------------------
# Small pool runs: split smoke and determinism
# ---------------------------------------------------------------------------


def _drive_pool(pool, steps: int, *, reset_every: int | None = None) -> None:
    """Drive `steps` phases, optionally deferring resets to a slower cadence.

    `reset_every=None` is the ordinary production cadence: a finished slot is
    reset in the same phase it finished in. A value defers resets, so a slot
    sits terminal for several phases before starting its next generation --
    the same games in a different scheduling order, which is what the
    determinism probe varies.
    """
    from stratego.training.shared_buffers import STATUS_ACTIVE, STATUS_TERMINAL

    for step in range(steps):
        actions = pool.select_actions()
        pool.buffers.decision_valid[:] = (pool.buffers.status == STATUS_ACTIVE).astype(
            pool.buffers.decision_valid.dtype
        )
        pool.buffers.actions[:] = actions
        if reset_every is None:
            pool.step()
            continue
        if step % reset_every == 0:
            finished = np.flatnonzero(pool.buffers.status == STATUS_TERMINAL)
            if finished.size:
                pool.request_reset(finished.tolist())
        pool.step(apply_actions=True, auto_reset=False)


def _pool_run(
    directory: Path,
    *,
    source,
    root_seed: int,
    environments: int,
    workers: int,
    steps: int,
    run_id: str,
    reset_every: int | None = None,
) -> dict:
    """One recorded worker-pool run through the real reset/record/persist path."""
    directory.mkdir(parents=True, exist_ok=True)
    pool = WorkerPool(
        environments,
        workers,
        root_seed=root_seed,
        recording=RecordingConfig(
            enabled=True,
            snapshot_interval=32,
            output_directory=str(directory),
            compress_records=True,
            run_id=run_id,
            encode_records=True,
        ),
        setup_source=source,
    )
    started = time.perf_counter()
    pool.start()
    try:
        _drive_pool(pool, steps, reset_every=reset_every)
    finally:
        totals = pool.shutdown()
    provenance = read_provenance_index(directory)
    return {
        "directory": str(directory),
        "run_id": run_id,
        "root_seed": root_seed,
        "environments": environments,
        "workers": workers,
        "steps": steps,
        "reset_cadence": "immediate" if reset_every is None else f"every {reset_every} phases",
        "seconds": time.perf_counter() - started,
        "games_sealed": int(totals["total_games_recorded"]),
        "provenance_records": int(totals.get("total_provenance_records", 0)),
        "provenance_missing": int(totals.get("total_provenance_missing", 0)),
        "setup_source_calls": int(totals.get("total_setup_source_calls", 0)),
        "records": provenance,
    }


def run_split_smoke(root: Path) -> dict:
    """Explicit validation/test access, in its own directories."""
    results = {}
    for split in ("validation", "test"):
        justification = (
            f"Phase 7 Agent 5 explicit {split}-split access smoke request; "
            f"separate directory, never merged into the training campaign"
        )
        source = audit_setup_source(split, justification)
        directory = root / f"smoke_{split}"
        run = _pool_run(
            directory,
            source=source,
            root_seed=SMOKE_ROOT_SEED,
            environments=SMOKE_ENVIRONMENTS,
            workers=SMOKE_WORKERS,
            steps=SMOKE_STEPS,
            run_id=f"smoke{split[:3]}",
        )
        records = run.pop("records")
        violations = 0
        bases = set()
        for record in records.values():
            violations += len(verify_provenance_split(record, split))
            for side in ("red", "blue"):
                bases.add(record[side]["base_setup_id"])
                if record[side]["base_index"] < TRAIN_PER_FAMILY:
                    violations += 1
        results[split] = {
            **run,
            "requested_split": split,
            "justification": justification,
            "split_violations": violations,
            "distinct_bases": len(bases),
            "purpose": source.purpose,
        }
    return results


def run_determinism(root: Path) -> dict:
    """Setup assignment under changed worker count, schedule and recycle boundary.

    The comparison design, stated exactly:

    ```text
    baseline   4 workers, 32 environments, root seed D, 1,400 phases, a
               finished slot reset in the phase it finished in
    variant A  8 workers -- a different slot-to-worker partitioning and a
               different arrival interleaving; everything else identical
    variant B  4 workers, resets deferred to every 7th phase, so a slot sits
               terminal for several phases before its next generation starts:
               the same games created in a different scheduling order
    variant C  the recycle boundary. A recycled segment restarts the process
               under `segment_root_seed(base, segment)`; the check is that a
               cold replay of that segment, at a different worker count,
               reproduces its assignment exactly
    ```

    Each run writes its own provenance sidecar. The gate is that for every
    logical game identity `(environment_id, generation)` present in two runs,
    the two runs assigned the *same* setups, fingerprints and base ids. Model
    action histories are not compared: this gate is about setup assignment.
    """
    source = training_setup_source()
    runs = {}
    runs["baseline"] = _pool_run(
        root / "det_baseline",
        source=source,
        root_seed=DETERMINISM_ROOT_SEED,
        environments=DETERMINISM_ENVIRONMENTS,
        workers=4,
        steps=DETERMINISM_STEPS,
        run_id="detbase",
    )
    runs["worker_count"] = _pool_run(
        root / "det_workers",
        source=source,
        root_seed=DETERMINISM_ROOT_SEED,
        environments=DETERMINISM_ENVIRONMENTS,
        workers=8,
        steps=DETERMINISM_STEPS,
        run_id="detwork",
    )

    runs["schedule"] = _pool_run(
        root / "det_schedule",
        source=source,
        root_seed=DETERMINISM_ROOT_SEED,
        environments=DETERMINISM_ENVIRONMENTS,
        workers=4,
        steps=DETERMINISM_STEPS,
        run_id="detsch",
        reset_every=7,
    )

    # The recycle boundary. A recycled segment restarts the process with
    # `segment_root_seed(base, segment)`, so the check is that replaying a
    # segment's identity from a cold process reproduces its assignment exactly.
    segment_seed = segment_root_seed(DETERMINISM_ROOT_SEED, 1)
    runs["recycle_segment_first"] = _pool_run(
        root / "det_recycle_a",
        source=source,
        root_seed=segment_seed,
        environments=DETERMINISM_ENVIRONMENTS,
        workers=4,
        steps=DETERMINISM_STEPS,
        run_id="detrec1",
    )
    runs["recycle_segment_replay"] = _pool_run(
        root / "det_recycle_b",
        source=source,
        root_seed=segment_seed,
        environments=DETERMINISM_ENVIRONMENTS,
        workers=2,
        steps=DETERMINISM_STEPS,
        run_id="detrec2",
    )

    def keyed(run):
        return {
            (record["environment_id"], record["generation"]): record
            for record in run["records"].values()
        }

    comparisons = {}
    baseline = keyed(runs["baseline"])
    recycle_reference = keyed(runs["recycle_segment_first"])
    for name, reference in (
        ("worker_count", baseline),
        ("schedule", baseline),
        ("recycle_segment_replay", recycle_reference),
    ):
        other = keyed(runs[name])
        shared = sorted(set(reference) & set(other))
        mismatches = 0
        for key in shared:
            for side in ("red", "blue"):
                left, right = reference[key][side], other[key][side]
                if (
                    left["engine_setup"] != right["engine_setup"]
                    or left["final_setup_fingerprint"] != right["final_setup_fingerprint"]
                    or left["base_setup_id"] != right["base_setup_id"]
                    or bool(left["reflection_applied"]) != bool(right["reflection_applied"])
                    or left["perturbation_seed"] != right["perturbation_seed"]
                ):
                    mismatches += 1
        comparisons[name] = {
            "compared_game_identities": len(shared),
            "setup_assignment_mismatches": mismatches,
        }

    # Isolated regeneration: generation `g` of a slot rebuilt from a cold
    # simulator, without replaying the generations before it.
    isolated_mismatches = 0
    for environment_id in (0, 7, 31):
        for generation in (0, 1, 3):
            simulator = BatchSimulator(
                1,
                root_seed=DETERMINISM_ROOT_SEED,
                first_environment_id=environment_id,
                setup_source=source,
            )
            while simulator.generation(0) < generation:
                simulator.reset_slots([0])
            expected = source.assign(
                root_seed=DETERMINISM_ROOT_SEED,
                environment_id=environment_id,
                generation=generation,
            )
            if simulator.setups(0) != (expected.red_setup, expected.blue_setup):
                isolated_mismatches += 1

    return {
        "design": {
            "baseline": "4 workers, 32 environments, 1,400 pool steps",
            "worker_count": "8 workers, everything else identical",
            "schedule": "4 workers, the same phases driven as two 700-step pools",
            "recycle_segment": (
                "segment_root_seed(base, 1); one run at 4 workers and a cold "
                "replay at 2 workers"
            ),
            "gate": (
                "for every logical game identity (environment_id, generation) "
                "present in both runs, identical engine setups, fingerprints, "
                "base ids, reflection bits and perturbation seeds"
            ),
            "not_compared": (
                "model action histories; float16 batch-shape effects are out of "
                "scope for a setup-assignment gate"
            ),
        },
        "runs": {
            name: {key: value for key, value in run.items() if key != "records"}
            for name, run in runs.items()
        },
        "comparisons": comparisons,
        "isolated_regeneration_mismatches": isolated_mismatches,
        "total_setup_assignment_mismatches": sum(
            entry["setup_assignment_mismatches"] for entry in comparisons.values()
        ),
    }


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def measure_sampler_cost(samples: int = 2_000) -> dict:
    """In-process sampler cost, separated from the one-time library load.

    The 8,000-entry library is parsed once per process, so the first call in a
    worker is far more expensive than every call after it. Both are measured:
    the cold load is a fixed per-worker startup cost, and the steady-state
    latency is what a collection run actually pays per game.
    """
    source = training_setup_source()
    load_library_index.cache_clear()
    started = time.perf_counter()
    load_library_index()
    load_seconds = time.perf_counter() - started

    latencies: list[float] = []
    for index in range(samples):
        call_started = time.perf_counter()
        source.assign(root_seed=1, environment_id=index, generation=0)
        latencies.append(time.perf_counter() - call_started)
    return _latency_report(
        latencies,
        extra={
            "cold_library_load_seconds": load_seconds,
            "cold_library_load_note": (
                "once per worker process at startup; excluded from the "
                "per-call figures above"
            ),
        },
    )


def _latency_report(latencies, extra: dict | None = None) -> dict:
    ordered = sorted(latencies)
    total = sum(ordered)
    report = {
        "calls": len(ordered),
        "total_seconds": total,
        "calls_per_second": (len(ordered) / total) if total else 0.0,
        "mean_seconds": statistics.fmean(ordered) if ordered else 0.0,
        "median_seconds": statistics.median(ordered) if ordered else 0.0,
        "p95_seconds": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        if ordered
        else 0.0,
        "p99_seconds": ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]
        if ordered
        else 0.0,
        "max_seconds": ordered[-1] if ordered else 0.0,
    }
    report.update(extra or {})
    return report


def performance_report(campaign: dict, corpus: dict, microbenchmark: dict) -> dict:
    totals = campaign["totals"]
    wall = campaign["wall_seconds"]
    games = corpus["distinct_games"]
    positions = int(totals.get("total_transitions", 0))
    setup_seconds = float(totals.get("total_setup_source_seconds", 0.0))
    setup_calls = int(totals.get("total_setup_source_calls", 0))
    written = int(totals.get("total_persisted_bytes", 0))
    worker_seconds = float(totals.get("worker_cpu_seconds", 0.0))

    in_run = _latency_report(campaign["setup_latencies_seconds"])
    return {
        "wall_seconds": wall,
        "steps": campaign["steps"],
        "positions": positions,
        "positions_per_second": positions / wall if wall else 0.0,
        "games": games,
        "games_per_second": games / wall if wall else 0.0,
        "decisions_recorded": int(totals.get("total_decisions_recorded", 0)),
        "compressed_bytes_written": written,
        "compressed_gib_per_hour": (written / wall) * 3600.0 / BYTES_PER_GIB
        if wall
        else 0.0,
        "compression_ratio": (
            written / int(totals["total_record_bytes"])
            if totals.get("total_record_bytes")
            else 0.0
        ),
        "setup_sampling": {
            "calls": setup_calls,
            "calls_per_game": setup_calls / games if games else 0.0,
            "total_seconds_across_workers": setup_seconds,
            "calls_per_second_across_workers": setup_calls / setup_seconds
            if setup_seconds
            else 0.0,
            "fraction_of_wall_time": setup_seconds / wall if wall else 0.0,
            "fraction_of_worker_cpu_time": setup_seconds / worker_seconds
            if worker_seconds
            else 0.0,
            "in_run_latency": in_run,
            "microbenchmark": microbenchmark,
            "provenance_seconds": float(totals.get("total_provenance_seconds", 0.0)),
            "provenance_bytes": int(totals.get("total_provenance_bytes", 0)),
            "provenance_fraction_of_wall_time": (
                float(totals.get("total_provenance_seconds", 0.0)) / wall
                if wall
                else 0.0
            ),
        },
        "memory": campaign["memory"],
        "swap": campaign["swap"],
        "disk": campaign["disk"],
    }


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def write_provenance_csv(rows: list[dict], path: Path) -> dict:
    if not rows:
        path.write_text("")
        return {"rows": 0, "path": str(path)}
    fieldnames = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "columns": fieldnames, "path": str(path)}


def completion_gates(
    prerequisites: dict,
    campaign: dict,
    corpus: dict,
    shards: dict,
    smoke: dict,
    determinism: dict,
    bank_before: dict,
    bank_after: dict,
    performance: dict,
    tests_after: dict | None,
) -> dict:
    counters = corpus["counters"]
    coverage = corpus["coverage"]
    failures = campaign["failures"]
    reconstruction = corpus["reconstruction"]
    gates = {
        "agents_1_4_pass_verified": prerequisites["handoff_unchanged"],
        "library_and_sampler_digests_unchanged": prerequisites["handoff_unchanged"],
        "sampler_integrated_into_the_real_reset_path": (
            campaign["configuration"]["setup_source"]["kind"] == "setup_library"
        ),
        "train_split_is_the_default": (
            campaign["configuration"]["setup_source"]["split"] == TRAINING_SPLIT
        ),
        "held_out_splits_need_an_explicit_request": all(
            entry["purpose"] == "evaluation_audit" for entry in smoke.values()
        ),
        "campaign_games_at_least_4096": corpus["distinct_games"] >= MINIMUM_GAMES,
        "all_256_ordered_family_pairs_present": (
            coverage["ordered_pairs_seen"] == ORDERED_FAMILY_PAIRS
        ),
        "at_least_16_games_per_ordered_pair": (
            coverage["minimum_games_per_pair"] >= MINIMUM_GAMES_PER_FAMILY_PAIR
        ),
        "both_reflection_branches_exercised": (
            counters["reflection_true"] > 0 and counters["reflection_false"] > 0
        ),
        "both_perturbation_branches_exercised": (
            counters["perturbation_true"] > 0 and counters["perturbation_false"] > 0
        ),
        "zero_setup_engine_validation_failures": (
            counters["setup_engine_validation_failures"] == 0
        ),
        "zero_provenance_mismatches": counters["provenance_mismatches"] == 0,
        "zero_provenance_missing": counters["provenance_missing"] == 0,
        "zero_wrong_split_samples": counters["split_violations"] == 0,
        "zero_family_identity_mismatches": counters["family_identity_mismatches"] == 0,
        "zero_base_identity_mismatches": counters["base_identity_mismatches"] == 0,
        "zero_fingerprint_mismatches": counters["fingerprint_mismatches"] == 0,
        "zero_illegal_actions": failures.get("illegal_actions", 0) == 0,
        "zero_action_frame_mismatches": failures.get("action_frame_errors", 0) == 0,
        "zero_model_failures": failures.get("model_errors", 0) == 0,
        "zero_nonfinite_outputs": failures.get("nonfinite_outputs", 0) == 0,
        "zero_worker_failures": failures.get("worker_errors", 0) == 0,
        "zero_trajectory_decode_failures": counters["decode_failures"] == 0,
        "zero_record_validation_failures": counters["record_validation_failures"] == 0,
        "zero_persisted_record_corruption": shards["ok"],
        "zero_duplicate_game_ids": counters["duplicate_game_ids"] == 0,
        "streaming_verification_clean": (
            int(campaign["totals"].get("total_verified_decisions", 0)) > 0
            and int(campaign["totals"].get("total_reconstruction_mismatches", 0)) == 0
        ),
        "reconstructed_decisions_at_least_10000": (
            reconstruction["decisions"] >= MINIMUM_RECONSTRUCTED_DECISIONS
        ),
        "zero_reconstruction_mismatches": (
            reconstruction["mismatches"] == 0
            and reconstruction["setup_mismatches"] == 0
        ),
        "setup_assignment_deterministic": (
            determinism["total_setup_assignment_mismatches"] == 0
            and determinism["isolated_regeneration_mismatches"] == 0
        ),
        "phase_4_bank_unchanged": (
            bank_before["digest"] == bank_after["digest"]
            and bank_before["count"] == bank_after["count"]
            and bank_before["bank_version"] == bank_after["bank_version"]
        ),
        "trajectory_v1_semantics_unchanged": TRAJECTORY_VERSION == "trajectory_v1",
        "observation_contract_unchanged": (
            OBSERVATION_VERSION == "observation_v2_1_127ch"
            and OBSERVATION_CHANNELS == 127
        ),
        "no_meaningful_training": (
            campaign["candidate"]["optimizer_steps"] == 0
            and not campaign["candidate"]["weights_mutated"]
        ),
        "performance_measured": performance["setup_sampling"]["calls"] > 0,
        "campaign_completed_without_error": campaign["status"] == "ok",
    }
    if tests_after is not None:
        gates["full_repository_suite_green"] = tests_after.get("failed", 1) == 0
    gates["passed"] = sum(1 for value in gates.values() if value is True)
    gates["total"] = len(gates) - 1
    return gates


def run_pytest() -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout else ""
    import re

    def _count(label: str) -> int:
        match = re.search(rf"(\d+) {label}", tail)
        return int(match.group(1)) if match else 0

    return {
        "command": "python -m pytest -q",
        "returncode": completed.returncode,
        "summary": tail,
        "passed": _count("passed"),
        "skipped": _count("skipped"),
        "failed": _count("failed") + _count("error"),
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
        help="where the campaign's shards and sidecars are written",
    )
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="a fast shape check: a tiny campaign and relaxed targets",
    )
    parser.add_argument("--keep-output", action="store_true")
    arguments = parser.parse_args()

    output_root = Path(
        arguments.output
        or (REPOSITORY_ROOT / ".phase7_agent05_output")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    campaign_directory = output_root / "campaign"
    if campaign_directory.exists():
        shutil.rmtree(campaign_directory)
    campaign_directory.mkdir(parents=True)

    print(f"Phase 7 Agent 5 — production pipeline integration ({INTEGRATION_VERSION})")
    started_at = time.time()
    overall_started = time.perf_counter()

    prerequisites = _verify_prerequisites()
    print(f"  prerequisites: {'OK' if prerequisites['handoff_unchanged'] else 'FAILED'}")
    if not prerequisites["handoff_unchanged"]:
        for problem in prerequisites["problems"]:
            print(f"    - {problem}")
        return 2

    bank_before = _phase_4_bank_identity()
    print(f"  Phase 4 bank before: {bank_before['digest'][:16]}… ({bank_before['count']})")

    source = training_setup_source()
    api = _setup_source_api(source)

    minimum_games = 64 if arguments.smoke else MINIMUM_GAMES
    minimum_pairs = 0 if arguments.smoke else ORDERED_FAMILY_PAIRS
    minimum_per_pair = 0 if arguments.smoke else MINIMUM_GAMES_PER_FAMILY_PAIR
    environments = 64 if arguments.smoke else CAMPAIGN_ENVIRONMENTS
    workers = 4 if arguments.smoke else CAMPAIGN_WORKERS
    inference_batch = 64 if arguments.smoke else CAMPAIGN_INFERENCE_BATCH
    verify_decisions = 2_000 if arguments.smoke else CAMPAIGN_VERIFY_DECISIONS
    reconstruct_target = (
        1_000 if arguments.smoke else MINIMUM_RECONSTRUCTED_DECISIONS
    )

    def progress(sample: dict) -> None:
        print(
            f"    step {sample['step']:6d}  "
            f"games {sample['games_sealed']:6d}  "
            f"pairs {sample['ordered_pairs_seen']:3d}/256  "
            f"min/pair {sample['minimum_games_per_pair']:3d}  "
            f"{sample['positions_per_second']:8.0f} pos/s  "
            f"{sample['elapsed_seconds']:7.0f}s"
        )

    print("  campaign:")
    campaign = run_campaign(
        campaign_directory,
        minimum_games=minimum_games,
        minimum_pairs=minimum_pairs,
        minimum_per_pair=minimum_per_pair,
        max_steps=CAMPAIGN_MAX_STEPS,
        max_seconds=CAMPAIGN_MAX_SECONDS,
        verify_target_decisions=verify_decisions,
        environments=environments,
        workers=workers,
        inference_batch=inference_batch,
        progress=progress,
    )
    if campaign["status"] != "ok":
        print(f"  campaign FAILED: {campaign['error']}")

    print("  verifying persisted shards …")
    shards = directory_summary(campaign_directory, decode=False)
    print(f"    {shards['shard_count']} shards, {shards['record_count']} records")

    print("  verifying every record against its provenance …")
    corpus = verify_persisted_corpus(
        campaign_directory,
        expected_split=TRAINING_SPLIT,
        reconstruct_target=reconstruct_target,
    )
    rows = corpus.pop("rows")
    print(
        f"    {corpus['distinct_games']} games, "
        f"{corpus['coverage']['ordered_pairs_seen']}/256 pairs, "
        f"min/pair {corpus['coverage']['minimum_games_per_pair']}, "
        f"{corpus['reconstruction']['decisions']} decisions reconstructed"
    )

    print("  explicit validation/test split smoke requests …")
    smoke = run_split_smoke(output_root)

    print("  determinism probe …")
    determinism = run_determinism(output_root)
    print(
        f"    setup assignment mismatches: "
        f"{determinism['total_setup_assignment_mismatches']}"
    )

    print("  sampler microbenchmark …")
    microbenchmark = measure_sampler_cost(200 if arguments.smoke else 2_000)
    performance = performance_report(campaign, corpus, microbenchmark)
    print(
        f"    {performance['setup_sampling']['calls_per_second_across_workers']:.0f} "
        f"setup calls/s, "
        f"{performance['setup_sampling']['fraction_of_wall_time'] * 100:.4f}% of wall"
    )

    bank_after = _phase_4_bank_identity()
    print(f"  Phase 4 bank after: {bank_after['digest'][:16]}…")

    tests_after = run_pytest() if arguments.run_pytest else None
    if tests_after is not None:
        print(f"  suite: {tests_after['summary']}")

    gates = completion_gates(
        prerequisites,
        campaign,
        corpus,
        shards,
        smoke,
        determinism,
        bank_before,
        bank_after,
        performance,
        tests_after,
    )
    status = "PASS" if gates["passed"] == gates["total"] else "FAIL"

    REPORT_DATA.mkdir(parents=True, exist_ok=True)
    csv_report = write_provenance_csv(rows, PROVENANCE_CSV)

    payload = {
        "agent": AGENT,
        "phase": PHASE,
        "status": status,
        "integration_version": INTEGRATION_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_at)),
        "commit": _git("rev-parse", "HEAD"),
        "working_tree_state": "dirty" if _git("status", "--porcelain") else "clean",
        **_environment(),
        "prerequisite_status": prerequisites,
        "frozen_versions": {
            "reference_engine": IMPLEMENTATION_VERSION,
            "rules_version": RULES_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "model_contract_version": MODEL_CONTRACT_VERSION,
            "trajectory_version": TRAJECTORY_VERSION,
            "setup_library_version": SETUP_LIBRARY_VERSION,
            "setup_contract_version": SETUP_GENERATOR_CONTRACT_VERSION,
            "family_contract_version": SETUP_FAMILY_VERSION,
            "trait_schema_version": SETUP_TRAIT_VECTOR_VERSION,
            "sampler_version": SAMPLER_VERSION,
            "setup_source_version": SETUP_SOURCE_VERSION,
            "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
            "backend": "KEEP_PYTHON",
            "primary_model": CAMPAIGN_CANDIDATE,
        },
        "library_digest": prerequisites["observed"]["library_digest"],
        "manifest_digest": prerequisites["observed"]["manifest_digest"],
        "setup_source_api": api,
        "seeds": {
            "campaign_root_seed": CAMPAIGN_ROOT_SEED,
            "smoke_root_seed": SMOKE_ROOT_SEED,
            "determinism_root_seed": DETERMINISM_ROOT_SEED,
            "recycle_segment_root_seed": segment_root_seed(DETERMINISM_ROOT_SEED, 1),
            "side_seed_derivation": api["training_source"]["side_seed_derivation"],
        },
        "campaign": campaign,
        "persisted_shards": {
            key: value for key, value in shards.items() if key != "game_ids"
        },
        "corpus_verification": corpus,
        "split_access": smoke,
        "determinism": determinism,
        "performance": performance,
        "phase_4_evaluation_bank": {
            "before": bank_before,
            "after": bank_after,
            "unchanged": bank_before["digest"] == bank_after["digest"],
            "replaced_by_setup_library_v1": False,
        },
        "observer_safety": {
            "regression": "tests/information_security/test_setup_provenance_boundary.py",
            "claims": [
                "the shared transport carries no setup or provenance field",
                "the published observation equals build_observation(state, mover)",
                "the model is called with exactly one observation tensor",
                "no live provenance value is reachable from the model inputs",
                "no live provenance value is reachable from the coordinator",
                "observation_v2_1_127ch gained no channel",
                "trajectory_v1 gained no provenance field",
            ],
            "positive_control": (
                "the same reachability walk finds every provenance value when one "
                "record is deliberately attached"
            ),
        },
        "tests_after": tests_after,
        "artifacts": {
            "integration_json": str(INTEGRATION_JSON.relative_to(REPOSITORY_ROOT)),
            "provenance_csv": str(PROVENANCE_CSV.relative_to(REPOSITORY_ROOT)),
            "provenance_csv_rows": csv_report["rows"],
        },
        "completion_gates": gates,
        "problems": corpus["problems"],
        "deviations": [],
        "total_seconds": time.perf_counter() - overall_started,
    }
    INTEGRATION_JSON.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")

    if not arguments.keep_output:
        shutil.rmtree(output_root, ignore_errors=True)

    print(f"\n  gates: {gates['passed']}/{gates['total']}")
    print(f"  status: {status}")
    print(f"  wrote {INTEGRATION_JSON.relative_to(REPOSITORY_ROOT)}")
    print(f"  wrote {PROVENANCE_CSV.relative_to(REPOSITORY_ROOT)} ({csv_report['rows']} rows)")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
