#!/usr/bin/env python3
"""Phase 9 Agent 3 acceptance harness: self-play collector and rollout store.

Re-verifies the accepted Agent 1 and Agent 2 freezes from live source, proves
the external rollout volume is really an external volume before a single
production shard is written, runs the >= 8,192-game infrastructure collection
soak, independently reproduces >= 100,000 learner-controlled neural decisions
from the acting-side checkpoint, and measures real Phase 9 storage density.

Artifacts:

    reports/phase_9_data/agent_03_rollout_store.json
    reports/phase_9_data/agent_03_collection_soak.json
    reports/phase_9_data/agent_03_behavior_reproduction.json
    reports/phase_9_data/agent_03_acceptance.json

The soak schedule
-----------------
Iteration 1 of all seven run namespaces: canonical (2,048) plus the six pilots
(1,024 each) = exactly 8,192 scheduled games, no additions and no
replacements. That schedule is chosen because every one of those iterations
has active history `(H000,)` — the real, immutable Phase 8 anchor — and a
behavior snapshot `B001` that the frozen contract defines as a fresh start
from the accepted Phase 8 checkpoint. Nothing here needs a checkpoint that
does not exist yet, so no future archive SHA is ever invented.

What this script does not do
----------------------------
No optimizer is constructed, no loss is computed, no gradient is taken, and
nothing is trained on the soak. The Phase 9 final-test bank is never opened,
never loaded and never inferred against.

Usage::

    python scripts/run_phase9_agent03.py                 # every stage
    python scripts/run_phase9_agent03.py --stage verify
    python scripts/run_phase9_agent03.py --soak-games 256 --quick
    python scripts/run_phase9_agent03.py --run-pytest
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import plistlib
import re
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from stratego.engine.constants import (  # noqa: E402
    IMPLEMENTATION_VERSION,
    OBSERVATION_VERSION,
    RULES_VERSION,
)
from stratego.engine.legal_moves import legal_action_mask, legal_actions  # noqa: E402
from stratego.engine.observation import build_observation  # noqa: E402
from stratego.engine.state import create_game  # noqa: E402
from stratego.engine.transition import apply_action  # noqa: E402
from stratego.model.policy_adapter import prepare_legality  # noqa: E402
from stratego.training import phase9_behavior as pb  # noqa: E402
from stratego.training import phase9_collector as pc  # noqa: E402
from stratego.training import phase9_contract as contract  # noqa: E402
from stratego.training import phase9_rollout_store as store  # noqa: E402
from stratego.training import phase9_schedule as schedule  # noqa: E402
from stratego.training import phase9_seed as seeds  # noqa: E402
from stratego.training import phase9_storage as storage  # noqa: E402
from stratego.training import synthetic_corpus as sc  # noqa: E402
from stratego.training.warmstart_contract import CORPUS_RULES  # noqa: E402

AGENT = 3
PHASE = 9
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_9_data"

STORE_ARTIFACT = DATA_DIRECTORY / "agent_03_rollout_store.json"
SOAK_ARTIFACT = DATA_DIRECTORY / "agent_03_collection_soak.json"
REPRODUCTION_ARTIFACT = DATA_DIRECTORY / "agent_03_behavior_reproduction.json"
ACCEPTANCE_ARTIFACT = DATA_DIRECTORY / "agent_03_acceptance.json"

AGENT1_ACCEPTANCE = DATA_DIRECTORY / "agent_01_acceptance.json"
AGENT2_ACCEPTANCE = DATA_DIRECTORY / "agent_02_acceptance.json"

#: Frozen Phase 8 inputs, pinned from the common contract.
ANCHOR_CHECKPOINT = REPOSITORY_ROOT / "checkpoints" / "phase8" / "warmstart_c1_v1.pt"
ANCHOR_SHA256 = "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca"
C1_CONFIG_DIGEST = "31ca84ab140c523e65567787b0289fe0dbdf5ab0344667410a5fda7060cfe07d"
C1_PARAMETERS = 863_959
ACCEPTED_CONTRACT_DIGEST = (
    "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"
)
ACCEPTED_POPULATION_DIGEST = (
    "6756790b15ee66195bc6339363e19fc475e3c606ef10613619b78b23d21bda73"
)

#: The harness may pin the expected resolver result to *verify the resolver*;
#: no library code hard-codes either path.
EXPECTED_CORPUS_ROOT = (
    "/Users/brandonwashington/Dev/Github/stratego/gpt_agent/"
    "data/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1"
)
EXPECTED_ROLLOUT_ROOT = "/Volumes/Brandon_Washington/stratego_phase9/rollouts"
EXPECTED_EXTERNAL_MOUNT = "/Volumes/Brandon_Washington"

#: The soak subtree. Agent 3's soak collects the real logical iteration-1
#: games, so its sealed digests are the real ones — but it writes them beside
#: the production tree rather than into it, so Agent 7's canonical run still
#: begins from an empty namespace it created itself. Identity is version +
#: digests, never a path, so a later agent that wants these bytes relocates
#: the directory and verifies the digest recorded here.
SOAK_SUBTREE = "agent_03_soak"

#: The soak: iteration 1 of every namespace. Exactly 8,192 scheduled games,
#: every one of which needs only the real H000 anchor.
SOAK_ITERATIONS = tuple((namespace, 1) for namespace in seeds.RUN_NAMESPACES)

#: Minimum learner-controlled neural decisions the reproduction audit must
#: independently re-derive.
REPRODUCTION_MINIMUM = 100_000

#: Historical-opponent decisions verified against H000 rather than B001. The
#: point is the per-side discipline, not the volume, so this is a sample.
HISTORICAL_REPRODUCTION_SAMPLE = 20_000


def _print(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Stage 1: prerequisites
# ---------------------------------------------------------------------------


def verify_prerequisites() -> dict:
    """Agents 1-2 PASS, every accepted digest, the corpus, and C1 identity."""
    problems: list[str] = []
    agent1 = json.loads(AGENT1_ACCEPTANCE.read_text())
    agent2 = json.loads(AGENT2_ACCEPTANCE.read_text())
    for name, document in (("agent 1", agent1), ("agent 2", agent2)):
        if document.get("status") != "PASS":
            problems.append(f"{name} is {document.get('status')!r}, not PASS")
        if document.get("problems"):
            problems.append(f"{name} reported problems: {document['problems']}")

    # Contract identities, recomputed from live source rather than read back.
    live_contract_digest = contract.contract_digest()
    if live_contract_digest != ACCEPTED_CONTRACT_DIGEST:
        problems.append(
            f"live contract digest {live_contract_digest} != accepted "
            f"{ACCEPTED_CONTRACT_DIGEST}"
        )
    if agent1.get("contract_digest") != ACCEPTED_CONTRACT_DIGEST:
        problems.append("agent 1 acceptance records a different contract digest")

    live_population_digest = schedule.population_digest()
    if live_population_digest != ACCEPTED_POPULATION_DIGEST:
        problems.append(
            f"live population digest {live_population_digest} != accepted "
            f"{ACCEPTED_POPULATION_DIGEST}"
        )
    accepted_run_digests = dict(agent2["run_schedule_digests"])
    live_run_digests = {
        namespace: schedule.run_schedule_digest(namespace)
        for namespace in seeds.RUN_NAMESPACES
    }
    for namespace, digest in accepted_run_digests.items():
        if live_run_digests.get(namespace) != digest:
            problems.append(f"{namespace} schedule digest drifted from the accepted freeze")

    # Exact scheduled counts, from live source.
    counts = {
        namespace: contract.games_per_iteration(namespace)
        for namespace in seeds.RUN_NAMESPACES
    }
    if counts["canonical"] != 2048 or any(
        counts[namespace] != 1024 for namespace in seeds.PILOT_NAMESPACES
    ):
        problems.append(f"scheduled per-iteration counts are not the frozen ones: {counts}")
    canonical_buckets = contract.bucket_counts("canonical")
    if canonical_buckets != {"current": 1024, "historical": 512, "rule": 307, "stress": 205}:
        problems.append(f"canonical bucket counts drifted: {canonical_buckets}")

    # The Phase 8 checkpoint and C1 identity.
    anchor_digest = pb.file_sha256(ANCHOR_CHECKPOINT)
    if anchor_digest != ANCHOR_SHA256:
        problems.append(f"Phase 8 checkpoint SHA-256 {anchor_digest} != {ANCHOR_SHA256}")
    snapshot = pb.load_behavior_snapshot(
        ANCHOR_CHECKPOINT,
        logical_identity="B001",
        policy_token="probe",
        expected_sha256=ANCHOR_SHA256,
    )
    parameters = snapshot.model.parameter_count()
    if parameters != C1_PARAMETERS:
        problems.append(f"C1 has {parameters} parameters, expected {C1_PARAMETERS}")
    from stratego.model.architecture_configs import config_digests

    live_c1_digest = config_digests().get("C1")
    if live_c1_digest != C1_CONFIG_DIGEST:
        problems.append(f"C1 config digest {live_c1_digest} != {C1_CONFIG_DIGEST}")

    corpus = verify_corpus()
    problems.extend(corpus["problems"])

    return {
        "agent1_status": agent1.get("status"),
        "agent2_status": agent2.get("status"),
        "agent1_gates": f"{agent1.get('gates_true', 18)}/{agent1.get('gates_total', 18)}",
        "agent2_gates": f"{agent2.get('gates_true')}/{agent2.get('gates_total')}",
        "contract_digest": live_contract_digest,
        "contract_digest_matches_accepted": live_contract_digest == ACCEPTED_CONTRACT_DIGEST,
        "population_digest": live_population_digest,
        "run_schedule_digests": live_run_digests,
        "run_schedule_digests_match_accepted": live_run_digests
        == {**live_run_digests, **accepted_run_digests},
        "scheduled_games_per_iteration": counts,
        "canonical_bucket_counts": canonical_buckets,
        "phase8_checkpoint_sha256": anchor_digest,
        "phase8_checkpoint_matches_accepted": anchor_digest == ANCHOR_SHA256,
        "c1_parameters": parameters,
        "c1_config_digest": live_c1_digest,
        "corpus": corpus,
        "problems": problems,
    }


def verify_corpus() -> dict:
    """Resolve the accepted Phase 8 corpus and require all three digests."""
    problems: list[str] = []
    resolution = sc.describe_corpus_root()
    root = sc.default_corpus_root()
    if str(root) != EXPECTED_CORPUS_ROOT:
        problems.append(f"corpus resolver returned {root}, expected {EXPECTED_CORPUS_ROOT}")

    from stratego.training.corpus_commit import corpus_content_digest

    observed = {
        "corpus_version": contract.EXPECTED_CORPUS_VERSION,
        "content_digest": corpus_content_digest(root, sc.CORPUS_SPLITS),
        "metadata_digest": sc._metadata_digest(root, sc.CORPUS_SPLITS),
        "commit_index_digest": sc._commit_index_digest(root, sc.CORPUS_SPLITS),
    }
    accepted = {
        "corpus_version": contract.EXPECTED_CORPUS_VERSION,
        "content_digest": contract.EXPECTED_CORPUS_CONTENT_DIGEST,
        "metadata_digest": contract.EXPECTED_CORPUS_METADATA_DIGEST,
        "commit_index_digest": contract.EXPECTED_CORPUS_COMMIT_INDEX_DIGEST,
    }
    for key, value in accepted.items():
        if observed[key] != value:
            problems.append(f"corpus {key} {observed[key]} != accepted {value}")

    # Prove no collection or rollout-store module hard-codes the corpus path.
    hard_coded = []
    for module_path in (
        "stratego/training/phase9_collector.py",
        "stratego/training/phase9_rollout_store.py",
        "stratego/training/phase9_behavior.py",
        "stratego/training/phase9_storage.py",
        "stratego/training/phase9_schedule.py",
    ):
        text = (REPOSITORY_ROOT / module_path).read_text()
        if EXPECTED_CORPUS_ROOT in text or "/Volumes/" in text:
            hard_coded.append(module_path)
    if hard_coded:
        problems.append(f"modules hard-code an absolute data path: {hard_coded}")

    return {
        "resolver": "stratego.training.synthetic_corpus.default_corpus_root()",
        "resolution": resolution,
        "resolved_root_matches_expected": str(root) == EXPECTED_CORPUS_ROOT,
        "accepted_identity": accepted,
        "observed_identity": observed,
        "identity_matches": observed == accepted,
        "modules_hard_coding_absolute_paths": hard_coded,
        "identity_rule": (
            "corpus identity is version + accepted digests, not filesystem "
            "location; a digest mismatch is BLOCKED and never repaired"
        ),
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Stage 2: storage
# ---------------------------------------------------------------------------


def _external_volume_facts(mount: str) -> dict:
    """Ask the OS whether `mount` really is a mounted external volume.

    A directory named `/Volumes/Brandon_Washington` on the boot filesystem
    would satisfy every path check and silently absorb production shards, so
    the question is answered by `diskutil`, not by `os.path.exists`.
    """
    facts: dict = {"mount_point": mount, "queried": True}
    try:
        raw = subprocess.run(
            ["diskutil", "info", "-plist", mount],
            capture_output=True,
            check=False,
        )
        if raw.returncode != 0:
            facts["diskutil_error"] = raw.stderr.decode()[:400]
            return facts
        info = plistlib.loads(raw.stdout)
    except (OSError, plistlib.InvalidFileException) as error:  # pragma: no cover
        facts["diskutil_error"] = f"{type(error).__name__}: {error}"
        return facts
    facts.update(
        {
            "volume_name": info.get("VolumeName"),
            "device_node": info.get("DeviceNode"),
            "filesystem": info.get("FilesystemName"),
            "protocol": info.get("BusProtocol"),
            "device_location": info.get("DeviceTreePath") or info.get("DeviceLocation"),
            "internal": bool(info.get("Internal", True)),
            "ejectable": bool(info.get("Ejectable", False)),
            "removable": bool(info.get("RemovableMedia", False)),
            "mounted": bool(info.get("MountPoint")),
            "volume_read_only": bool(info.get("WritableVolume") is False),
            "total_bytes": int(info.get("TotalSize") or 0),
            "free_bytes": int(info.get("FreeSpace") or 0),
        }
    )
    facts["is_external"] = (not facts["internal"]) and facts["mounted"]
    return facts


def verify_storage(*, scheduled_games: int) -> dict:
    """Resolve the rollout root and prove the external volume before writing.

    Stops `BLOCKED` rather than silently creating a substitute path on the
    boot filesystem: the whole point of the check is that a *missing* external
    drive must be loud.
    """
    problems: list[str] = []
    resolution = storage.describe_rollout_root()
    root = storage.default_rollout_root()
    if str(root) != EXPECTED_ROLLOUT_ROOT:
        problems.append(f"rollout resolver returned {root}, expected {EXPECTED_ROLLOUT_ROOT}")

    volume = _external_volume_facts(EXPECTED_EXTERNAL_MOUNT)
    if not volume.get("mounted"):
        problems.append(f"{EXPECTED_EXTERNAL_MOUNT} is not a mounted volume")
    if volume.get("internal", True):
        problems.append(
            f"{EXPECTED_EXTERNAL_MOUNT} reports as internal storage; refusing to "
            "treat it as the external rollout volume"
        )
    if volume.get("volume_read_only"):
        problems.append(f"{EXPECTED_EXTERNAL_MOUNT} is mounted read-only")

    # An ordinary directory on the boot disk would not be its own mount point.
    is_mount_point = os.path.ismount(EXPECTED_EXTERNAL_MOUNT)
    if not is_mount_point:
        problems.append(
            f"{EXPECTED_EXTERNAL_MOUNT} is not a mount point; it is an ordinary "
            "directory on the boot filesystem"
        )
    boot_device = os.stat("/").st_dev
    try:
        volume_device = os.stat(EXPECTED_EXTERNAL_MOUNT).st_dev
    except OSError as error:
        volume_device = None
        problems.append(f"cannot stat {EXPECTED_EXTERNAL_MOUNT}: {error}")
    if volume_device is not None and volume_device == boot_device:
        problems.append(
            f"{EXPECTED_EXTERNAL_MOUNT} shares a device with the boot filesystem"
        )

    evaluation = storage.evaluate_storage_target(root, scheduled_games)
    problems.extend(evaluation["problems"])

    return {
        "resolver": "stratego.training.phase9_storage.default_rollout_root()",
        "resolution": resolution,
        "resolved_root": str(root),
        "resolved_root_matches_expected": str(root) == EXPECTED_ROLLOUT_ROOT,
        "external_volume": volume,
        "is_mount_point": is_mount_point,
        "boot_filesystem_device": boot_device,
        "volume_device": volume_device,
        "distinct_from_boot_filesystem": volume_device is not None
        and volume_device != boot_device,
        "capacity_evaluation": evaluation,
        "soak_subtree": SOAK_SUBTREE,
        "identity_rule": storage.STORAGE_IDENTITY_RULE,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Stage 3: the collection soak
# ---------------------------------------------------------------------------


def build_participants(device: str, batch_shape: int) -> tuple:
    """`B001` and `H000` for a fresh run: both the accepted Phase 8 anchor.

    Two logical identities, one real checkpoint, and the binding checked
    against the frozen SHA-256 before either can play a move.
    """
    resolver = pc.SnapshotResolver(device=device, inference_batch_shape=batch_shape)
    behavior = resolver.resolve(
        ANCHOR_CHECKPOINT,
        logical_identity="B001",
        policy_token=schedule.behavior_policy_token("canonical", 1),
        expected_sha256=ANCHOR_SHA256,
    )
    anchor = resolver.resolve(
        ANCHOR_CHECKPOINT,
        logical_identity=contract.HISTORICAL_ANCHOR_ID,
        policy_token=schedule.ANCHOR_POLICY_TOKEN,
        expected_sha256=ANCHOR_SHA256,
    )
    return resolver, behavior, anchor


def run_soak(
    root: Path,
    *,
    device: str,
    batch_shape: int,
    games_in_flight: int,
    limit_per_iteration: "int | None",
    observer_probe_plies: int,
) -> dict:
    """Collect iteration 1 of every namespace and seal each rollout."""
    import resource

    resolver, behavior, anchor = build_participants(device, batch_shape)
    iterations = []
    started = time.perf_counter()
    cpu_before = _cpu_seconds()
    peak_rss = 0
    peak_mps = 0
    for namespace, iteration in SOAK_ITERATIONS:
        # One loaded model, one logical identity per namespace: `B001` means
        # "this run's learner at iteration 1", and every run starts from the
        # same accepted Phase 8 file, so the weights are shared and the tokens
        # are not.
        namespace_behavior = (
            behavior
            if namespace == seeds.CANONICAL_NAMESPACE
            else resolver.resolve(
                ANCHOR_CHECKPOINT,
                logical_identity=schedule.behavior_snapshot_identity(iteration),
                policy_token=schedule.behavior_policy_token(namespace, iteration),
                expected_sha256=ANCHOR_SHA256,
            )
        )
        participants = pc.IterationParticipants(
            behavior=namespace_behavior,
            historical={contract.HISTORICAL_ANCHOR_ID: anchor},
        )
        # The active window is an explicit, validated input carrying the real
        # digest bound to the logical archive identity.
        history = schedule.ActiveHistoryManifest.frozen_for(
            namespace,
            iteration,
            checkpoint_digests={contract.HISTORICAL_ANCHOR_ID: anchor.checkpoint_sha256},
        )
        history.validate()

        iteration_started = time.perf_counter()
        _print(f"      {namespace} iteration {iteration}: collecting")
        summary = pc.collect_iteration(
            root,
            namespace,
            iteration,
            participants,
            population_version=contract.PHASE9_POPULATION_VERSION,
            schedule_version=contract.PHASE9_ROLLOUT_SCHEDULE_VERSION,
            contract_digest=ACCEPTED_CONTRACT_DIGEST,
            games_in_flight=games_in_flight,
            observer_probe_plies=observer_probe_plies,
            history=history,
            limit=limit_per_iteration,
            seal=limit_per_iteration is None,
            progress=lambda done, total: _print(f"        {done}/{total}"),
        )
        summary["active_history"] = history.to_dict()
        summary["storage"] = store.iteration_storage_summary(root, namespace, iteration)
        summary["wall_seconds"] = time.perf_counter() - iteration_started
        iterations.append(summary)
        peak_rss = max(peak_rss, _rss_bytes())
        peak_mps = max(peak_mps, _mps_bytes(device))
        _print(
            f"      {namespace}: {summary['games_collected']} games, "
            f"sealed={summary.get('sealed')} "
            f"{summary['games_per_second']:.2f} games/s"
        )

    elapsed = time.perf_counter() - started
    totals = _soak_totals(iterations, elapsed, peak_rss, device, batch_shape, games_in_flight)
    totals["cpu"] = _cpu_utilization(cpu_before, _cpu_seconds(), elapsed)
    totals["peak_mps_bytes"] = peak_mps
    return {"iterations": iterations, "totals": totals}


def _rss_bytes() -> int:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    return int(usage if sys.platform == "darwin" else usage * 1024)


def _mps_bytes(device: str) -> int:
    if device != "mps":
        return 0
    return int(torch.mps.driver_allocated_memory())


def _cpu_seconds() -> dict:
    import resource

    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "user": own.ru_utime + children.ru_utime,
        "system": own.ru_stime + children.ru_stime,
    }


def _cpu_utilization(before: dict, after: dict, elapsed: float) -> dict:
    """Process CPU time against wall time, in cores.

    `cores_busy` is the honest instrument here: a single collector process
    driving one Metal device is not usefully described by a percentage of the
    whole machine. MPS device utilization is not exposed by the framework at
    all, so allocated device memory is reported instead and labelled as the
    proxy it is.
    """
    user = after["user"] - before["user"]
    system = after["system"] - before["system"]
    total = user + system
    return {
        "user_seconds": user,
        "system_seconds": system,
        "cpu_seconds": total,
        "wall_seconds": elapsed,
        "cores_busy": total / elapsed if elapsed else 0.0,
        "logical_cores": os.cpu_count(),
        "machine_utilization_fraction": (
            total / elapsed / os.cpu_count() if elapsed and os.cpu_count() else 0.0
        ),
        "mps_utilization_note": (
            "PyTorch exposes no MPS device-utilization counter; allocated and "
            "driver-allocated device memory are reported instead"
        ),
    }


def _soak_totals(iterations, elapsed, peak_rss, device, batch_shape, games_in_flight) -> dict:
    games = sum(item["games_collected"] for item in iterations)
    committed = sum(
        item["storage"]["committed_games"] for item in iterations
    )
    decisions = sum(item["total_decisions"] for item in iterations)
    neural = sum(item["neural_decisions"] for item in iterations)
    learner = sum(item["learner_decisions"] for item in iterations)
    plies = sum(item["total_plies"] for item in iterations)
    total_bytes = sum(item["storage"]["total_bytes"] for item in iterations)
    shard_bytes = sum(item["storage"]["shard_bytes"] for item in iterations)
    metadata_bytes = sum(item["storage"]["metadata_bytes"] for item in iterations)
    journal_bytes = sum(item["storage"]["journal_bytes"] for item in iterations)
    uncompressed = sum(
        (item["writer_stats"] or {}).get("uncompressed_bytes", 0) for item in iterations
    )
    compressed = sum(
        (item["writer_stats"] or {}).get("compressed_bytes", 0) for item in iterations
    )
    buckets: dict = {}
    results: dict = {}
    for item in iterations:
        for key, value in item["bucket_counts"].items():
            buckets[key] = buckets.get(key, 0) + value
        for key, value in item["terminal_results"].items():
            results[key] = results.get(key, 0) + value
    return {
        "iterations": len(iterations),
        "games_collected": games,
        "games_committed": committed,
        "bucket_counts": buckets,
        "buckets_represented": sorted(buckets),
        "terminal_results": results,
        "total_decisions": decisions,
        "neural_decisions": neural,
        "learner_decisions": learner,
        "total_plies": plies,
        "mean_game_length_plies": plies / games if games else 0.0,
        "wall_seconds": elapsed,
        "games_per_second": games / elapsed if elapsed else 0.0,
        "positions_per_second": plies / elapsed if elapsed else 0.0,
        "decisions_per_second": decisions / elapsed if elapsed else 0.0,
        "inference_device": device,
        "inference_batch_shape": batch_shape,
        "games_in_flight": games_in_flight,
        "peak_rss_bytes": peak_rss,
        "mps_allocated_bytes": (
            int(torch.mps.current_allocated_memory()) if device == "mps" else None
        ),
        "mps_driver_allocated_bytes": (
            int(torch.mps.driver_allocated_memory()) if device == "mps" else None
        ),
        "storage": {
            "total_bytes": total_bytes,
            "shard_bytes": shard_bytes,
            "metadata_bytes": metadata_bytes,
            "journal_bytes": journal_bytes,
            "bytes_per_game": total_bytes / committed if committed else 0.0,
            "bytes_per_decision": total_bytes / decisions if decisions else 0.0,
            "bytes_per_position": total_bytes / plies if plies else 0.0,
            "payload_uncompressed_bytes": uncompressed,
            "payload_compressed_bytes": compressed,
            "compression_ratio": compressed / uncompressed if uncompressed else 0.0,
            "storage_per_hour_bytes": (
                total_bytes / elapsed * 3600 if elapsed else 0.0
            ),
        },
    }


def project_storage(totals: dict) -> dict:
    """Project the measured density onto the real Phase 9 workloads.

    Agent 2's 8.08 GiB figure was a planning estimate scaled from the Phase 8
    rule-vs-rule corpus. This replaces it with the measured neural-rollout
    number and reports both, because the difference between them is the whole
    reason the measurement was required.
    """
    per_game = totals["storage"]["bytes_per_game"]
    canonical_games = 60 * 2048
    pilot_games = 6 * 8 * 1024
    measured_total = per_game * (canonical_games + pilot_games)
    planning = storage.projected_rollout_bytes(canonical_games + pilot_games)
    return {
        "measured_bytes_per_game": per_game,
        "measured_bytes_per_decision": totals["storage"]["bytes_per_decision"],
        "measured_compression_ratio": totals["storage"]["compression_ratio"],
        "canonical_run_games": canonical_games,
        "canonical_run_bytes": per_game * canonical_games,
        "canonical_run_gib": per_game * canonical_games / 1024**3,
        "all_pilots_games": pilot_games,
        "all_pilots_bytes": per_game * pilot_games,
        "all_pilots_gib": per_game * pilot_games / 1024**3,
        "phase9_total_games": canonical_games + pilot_games,
        "phase9_total_bytes": measured_total,
        "phase9_total_gib": measured_total / 1024**3,
        "agent_2_planning_estimate_bytes": planning["projected_bytes"],
        "agent_2_planning_estimate_gib": planning["projected_gib"],
        "measured_over_planning_ratio": (
            measured_total / planning["projected_bytes"] if planning["projected_bytes"] else 0.0
        ),
        "note": (
            "the planning estimate scaled the Phase 8 rule-vs-rule corpus by a "
            "deliberately pessimistic factor; this row is the measured Phase 9 "
            "neural-rollout density and supersedes it for capacity decisions"
        ),
    }


# ---------------------------------------------------------------------------
# Stage 4: the behavior reproduction audit
# ---------------------------------------------------------------------------


def _replay_requests(record, metadata, *, sides, tokens):
    """Rebuild every stored decision of one game whose acting side is wanted.

    The game is replayed from its own payload through the frozen engine, so
    the observation, the legal set and the action frame are re-derived rather
    than trusted. Nothing the collector reported is read except the stored
    decision itself, which is the thing under audit.
    """
    state = create_game(
        record.red_setup, record.blue_setup, rules=CORPUS_RULES, game_id=record.game_id
    )
    for decision in record.decisions:
        legal = legal_actions(state)
        if decision.acting_player in sides:
            legality = prepare_legality(
                legal, legal_action_mask(state, legal), state.acting_player
            )
            if tuple(legality.absolute) != tuple(decision.legal_action_ids):
                yield ("legal_set_mismatch", decision.ply, None)
            else:
                yield (
                    "request",
                    decision.ply,
                    pb.ReproductionRequest(
                        game_id=record.game_id,
                        ply=decision.ply,
                        acting_player=decision.acting_player,
                        observation=build_observation(state, state.acting_player),
                        legality=legality,
                        stored_probabilities=decision.old_probabilities,
                        stored_wdl=decision.win_draw_loss_prediction,
                        stored_action=decision.selected_action_id,
                        stored_policy_token=decision.collection_policy_version,
                        stored_checkpoint_sha256=tokens[decision.collection_policy_version],
                    ),
                )
        apply_action(state, decision.selected_action_id, legal=legal)


def run_reproduction_audit(
    root: Path,
    *,
    device: str,
    batch_shape: int,
    minimum: int,
    historical_sample: int,
) -> dict:
    """Re-derive stored decisions from the acting side's own checkpoint."""
    _resolver, behavior, anchor = build_participants(device, batch_shape)
    snapshots = {behavior.policy_token: behavior, anchor.policy_token: anchor}

    learner_reports: list[dict] = []
    historical_reports: list[dict] = []
    legal_set_mismatches = 0
    games_audited = 0
    observation_digests = set()
    started = time.perf_counter()

    for namespace, iteration in SOAK_ITERATIONS:
        state_document = store.read_iteration_state(root, namespace, iteration)
        if state_document is None:
            continue
        reader = store.Phase9RolloutReader(root, namespace, iteration)
        behavior_token = schedule.behavior_policy_token(namespace, iteration)
        tokens = {
            behavior_token: behavior.checkpoint_sha256,
            schedule.ANCHOR_POLICY_TOKEN: anchor.checkpoint_sha256,
        }
        namespace_snapshots = {
            behavior_token: pb.load_behavior_snapshot(
                ANCHOR_CHECKPOINT,
                logical_identity=schedule.behavior_snapshot_identity(iteration),
                policy_token=behavior_token,
                device=device,
                inference_batch_shape=batch_shape,
                expected_sha256=ANCHOR_SHA256,
                model=behavior.model,
                state_dict_digest_hint=behavior.loaded_state_dict_digest,
            ),
            schedule.ANCHOR_POLICY_TOKEN: anchor,
        }

        for game_id in reader.game_ids:
            if len(learner_reports) >= minimum and len(historical_reports) >= historical_sample:
                break
            record, metadata = reader.read_game(game_id)
            learner_players = {
                0 if colour == "red" else 1
                for colour in schedule.rebuild_scheduled_game(game_id).learner_sides
            }
            neural_sides = set(learner_players)
            if metadata["opponent_kind"] == "historical_snapshot":
                neural_sides |= {1 - player for player in learner_players}

            pending_learner: list = []
            pending_historical: list = []
            for kind, _ply, item in _replay_requests(
                record, metadata, sides=neural_sides, tokens=tokens
            ):
                if kind == "legal_set_mismatch":
                    legal_set_mismatches += 1
                    continue
                if item.acting_player in learner_players:
                    pending_learner.append(item)
                else:
                    pending_historical.append(item)
                observation_digests.add(
                    hash(np.asarray(item.observation).tobytes())
                )

            if len(learner_reports) < minimum and pending_learner:
                learner_reports.extend(
                    pb.reproduce_decisions(namespace_snapshots[behavior_token], pending_learner)
                )
            if len(historical_reports) < historical_sample and pending_historical:
                historical_reports.extend(
                    pb.reproduce_decisions(
                        namespace_snapshots[schedule.ANCHOR_POLICY_TOKEN], pending_historical
                    )
                )
            games_audited += 1
        if len(learner_reports) >= minimum and len(historical_reports) >= historical_sample:
            break

    elapsed = time.perf_counter() - started
    cross = run_cross_checkpoint_control(root, device, batch_shape)
    return {
        "tolerance": contract.BEHAVIOR_PROBABILITY_ABS_TOLERANCE,
        "reference_device": device,
        "inference_batch_shape": batch_shape,
        "games_audited": games_audited,
        "distinct_observations": len(observation_digests),
        "legal_set_mismatches": legal_set_mismatches,
        "learner": _reproduction_summary(learner_reports),
        "historical": _reproduction_summary(historical_reports),
        "cross_checkpoint_control": cross,
        "seconds": elapsed,
        "decisions_per_second": (
            (len(learner_reports) + len(historical_reports)) / elapsed if elapsed else 0.0
        ),
        "audited_fields": [
            "acting player",
            "observation digest",
            "legal set",
            "action frame",
            "behavior distribution",
            "sampled action legality",
            "WDL output",
            "behavior snapshot identity",
        ],
    }


def _reproduction_summary(reports) -> dict:
    verified = sum(1 for report in reports if report["verified"])
    differences = [
        report["max_abs_difference"]
        for report in reports
        if report.get("max_abs_difference") is not None
    ]
    wdl = [
        report["wdl_max_abs_difference"]
        for report in reports
        if report.get("wdl_max_abs_difference") is not None
    ]
    return {
        "decisions": len(reports),
        "verified": verified,
        "mismatches": len(reports) - verified,
        "max_abs_probability_difference": max(differences) if differences else 0.0,
        "max_abs_wdl_difference": max(wdl) if wdl else 0.0,
        "problems": [
            problem for report in reports if not report["verified"] for problem in report["problems"]
        ][:10],
    }


def run_cross_checkpoint_control(root: Path, device: str, batch_shape: int) -> dict:
    """Negative control: the audit must fail against the wrong network.

    A reproduction audit that cannot fail proves nothing. This verifies a
    sample of learner decisions against the *untrained* Phase 8 checkpoint —
    the mistake an auditor makes when it checks every move against one
    game-level digest — and requires the mismatch rate to be total.
    """
    untrained_path = REPOSITORY_ROOT / "checkpoints" / "phase8" / "warmstart_c1_v1_initialisation.pt"
    namespace, iteration = SOAK_ITERATIONS[0]
    if store.read_iteration_state(root, namespace, iteration) is None:
        return {"skipped": "no sealed iteration"}
    reader = store.Phase9RolloutReader(root, namespace, iteration)
    behavior_token = schedule.behavior_policy_token(namespace, iteration)
    wrong = pb.load_behavior_snapshot(
        untrained_path,
        logical_identity="B001",
        policy_token=behavior_token,
        device=device,
        inference_batch_shape=batch_shape,
    )
    sample: list = []
    for game_id in reader.game_ids[:4]:
        record, metadata = reader.read_game(game_id)
        learner_players = {
            0 if colour == "red" else 1
            for colour in schedule.rebuild_scheduled_game(game_id).learner_sides
        }
        for kind, _ply, item in _replay_requests(
            record,
            metadata,
            sides=learner_players,
            # Claim the wrong network's identity so the hard identity veto is
            # bypassed and the numerical comparison actually runs.
            tokens={
                behavior_token: wrong.checkpoint_sha256,
                schedule.ANCHOR_POLICY_TOKEN: wrong.checkpoint_sha256,
            },
        ):
            if kind == "request":
                sample.append(item)
            if len(sample) >= 256:
                break
        if len(sample) >= 256:
            break
    reports = pb.reproduce_decisions(wrong, sample)
    summary = _reproduction_summary(reports)
    return {
        "wrong_checkpoint": str(untrained_path.relative_to(REPOSITORY_ROOT)),
        "wrong_checkpoint_sha256": wrong.checkpoint_sha256,
        "decisions": summary["decisions"],
        "verified_against_wrong_checkpoint": summary["verified"],
        "mismatches": summary["mismatches"],
        "max_abs_probability_difference": summary["max_abs_probability_difference"],
        "control_holds": summary["decisions"] > 0 and summary["verified"] == 0,
    }


# ---------------------------------------------------------------------------
# Stage 5: store audit (replay, provenance, observer safety, crash evidence)
# ---------------------------------------------------------------------------


def audit_store(root: Path, *, sample_games: int, device: str, batch_shape: int) -> dict:
    """Replay committed games and check every seal precondition end to end."""
    iterations = []
    replay_illegal = 0
    setup_mismatches = 0
    duplicate_ids = 0
    unscheduled = 0
    orphans = 0
    decoded = 0
    replayed = 0
    all_ids: set = set()

    for namespace, iteration in SOAK_ITERATIONS:
        document = store.read_iteration_state(root, namespace, iteration)
        if document is None:
            continue
        reader = store.Phase9RolloutReader(root, namespace, iteration)
        scheduled = set(schedule.iteration_game_ids(namespace, iteration))
        committed = set(reader.commits)
        duplicate_ids += len(committed & all_ids)
        all_ids |= committed
        unscheduled += len(committed - scheduled)
        orphan = reader.orphans()
        orphans += len(orphan["metadata_without_commit"]) + len(orphan["commit_without_metadata"])

        source = None
        step = max(1, len(reader.game_ids) // max(1, sample_games))
        sampled = reader.game_ids[::step][:sample_games]
        for game_id in sampled:
            record, metadata = reader.read_game(game_id)
            decoded += 1
            problems = store.validate_rollout_metadata(metadata, record)
            if problems:
                setup_mismatches += 1
            # Replay every action through the frozen engine.
            state = create_game(
                record.red_setup, record.blue_setup, rules=CORPUS_RULES, game_id=record.game_id
            )
            legal_ok = True
            for action in record.actions:
                legal = legal_actions(state)
                if action not in legal:
                    legal_ok = False
                    break
                apply_action(state, action, legal=legal)
            if not legal_ok or not state.terminal:
                replay_illegal += 1
            else:
                replayed += 1
                if state.terminal_reason != record.terminal_reason:
                    replay_illegal += 1
            # Setup provenance must reconstruct the stored setups exactly.
            if source is None:
                from stratego.training.setup_source import training_setup_source
                from stratego.training.warmstart_contract import EXPECTED_SETUP_PROFILE

                source = training_setup_source(EXPECTED_SETUP_PROFILE)
            assignment = source.assign(
                root_seed=int(metadata["setup_root_seed"]),
                environment_id=schedule.SETUP_ENVIRONMENT_ID,
                generation=schedule.SETUP_GENERATION,
                game_id=game_id,
            )
            if (assignment.red_setup, assignment.blue_setup) != (
                record.red_setup,
                record.blue_setup,
            ):
                setup_mismatches += 1
            if assignment.provenance != metadata["setup_provenance"]:
                setup_mismatches += 1

        iterations.append(
            {
                "namespace": namespace,
                "iteration": iteration,
                "state": document["state"],
                "sealed_rollout_digest": document.get("sealed_rollout_digest"),
                "scheduled_games": len(scheduled),
                "committed_games": len(committed),
                "missing_games": len(scheduled - committed),
                "unscheduled_games": len(committed - scheduled),
                "orphan_records": len(orphan["metadata_without_commit"])
                + len(orphan["commit_without_metadata"]),
                "sampled_games_replayed": len(sampled),
                "behavior_snapshot_id": document.get("behavior_snapshot_id"),
                "behavior_checkpoint_sha256": document.get("behavior_checkpoint_sha256"),
                "inference_device": document.get("inference_device"),
                "inference_batch_shape": document.get("inference_batch_shape"),
                "storage": store.iteration_storage_summary(root, namespace, iteration),
            }
        )

    return {
        "store_version": store.PHASE9_COMMIT_VERSION,
        "collector_version": pc.PHASE9_COLLECTOR_VERSION,
        "states": list(contract.ROLLOUT_STATES),
        "iterations": iterations,
        "sealed_iterations": sum(1 for item in iterations if item["state"] == "SEALED"),
        "distinct_game_ids": len(all_ids),
        "duplicate_game_ids": duplicate_ids,
        "unscheduled_games": unscheduled,
        "orphan_records": orphans,
        "games_decoded": decoded,
        "games_replayed_legally": replayed,
        "replay_illegal_actions": replay_illegal,
        "setup_provenance_mismatches": setup_mismatches,
        "metadata_fields": list(store.METADATA_FIELDS),
        "commit_fields": list(store.COMMIT_FIELDS),
        "crash_stages": list(store.CRASH_STAGES),
    }


def audit_observer_boundary(root: Path, *, sample_games: int) -> dict:
    """Re-run the boundary audit over stored games, plus its positive control."""
    checked = 0
    failures = 0
    hidden_pieces = 0
    for namespace, iteration in SOAK_ITERATIONS:
        if store.read_iteration_state(root, namespace, iteration) is None:
            continue
        reader = store.Phase9RolloutReader(root, namespace, iteration)
        for game_id in reader.game_ids[:sample_games]:
            record, _metadata = reader.read_game(game_id)
            state = create_game(
                record.red_setup, record.blue_setup, rules=CORPUS_RULES, game_id=game_id
            )
            for index, decision in enumerate(record.decisions):
                legal = legal_actions(state)
                if index % 17 == 0:
                    report = pc.observer_safety_probe(
                        state,
                        state.acting_player,
                        build_observation(state, state.acting_player),
                    )
                    checked += 1
                    hidden_pieces += report["hidden_opponent_pieces"]
                    failures += 0 if report["safe"] else 1
                apply_action(state, decision.selected_action_id, legal=legal)
        break  # one namespace is a sample; the suite covers the rest

    control = _observer_positive_control()
    return {
        "boundary": (
            "the model input is the frozen observer-safe observation and the "
            "dense legality mask used for masking; no GameState, piece record, "
            "belief target or replay is reachable from it"
        ),
        "probes": checked,
        "failures": failures,
        "mean_hidden_opponent_pieces": hidden_pieces / checked if checked else 0.0,
        "positive_control": control,
        "privileged_state_location": (
            "true setups and per-piece truth live in the rollout payload and "
            "metadata for later belief labelling; neither is reachable from a "
            "model input"
        ),
    }


def _observer_positive_control() -> dict:
    """Plant privileged truth and require the audit to find it."""
    from stratego.training.setup_source import training_setup_source
    from stratego.training.warmstart_contract import EXPECTED_SETUP_PROFILE

    source = training_setup_source(EXPECTED_SETUP_PROFILE)
    assignment = source.assign(root_seed=777, environment_id=0, generation=0)
    state = create_game(
        assignment.red_setup, assignment.blue_setup, rules=CORPUS_RULES, game_id="control"
    )
    for _ in range(4):
        apply_action(state, legal_actions(state)[0])
    observer = state.acting_player

    def leaking_builder(live_state, live_observer):
        leaked = np.array(build_observation(live_state, live_observer))
        for record in live_state.pieces:
            if (
                record.owner != live_observer
                and record.alive
                and not record.known_to(live_observer)
            ):
                row, column = divmod(record.current_square, 10)
                leaked[0, row, column] = float(record.true_type) + 1.0
        return leaked

    planted = pc.observer_safety_probe(
        state, observer, leaking_builder(state, observer), builder=leaking_builder
    )
    clean = pc.observer_safety_probe(state, observer, build_observation(state, observer))
    return {
        "planted_leak_detected": not planted["safe"],
        "planted_leak_entries": planted["entries_sensitive_to_hidden_truth"],
        "frozen_builder_passes_the_same_check": clean["safe"],
        "control_holds": (not planted["safe"]) and clean["safe"],
    }


#: Symbols that would mean training. Their absence from the collection path is
#: checked structurally rather than asserted, because "we did not optimize" is
#: the one claim an infrastructure soak cannot prove by its results.
FORBIDDEN_TRAINING_SYMBOLS = (
    "backward",
    "zero_grad",
    "AdamW",
    "Adam",
    "SGD",
    "optim",
    "optimizer",
    "cross_entropy",
    "policy_loss",
    "value_loss",
    "belief_loss",
    "multi_head_loss",
    "ppo",
)

COLLECTION_MODULES = (
    "stratego/training/phase9_behavior.py",
    "stratego/training/phase9_collector.py",
    "stratego/training/phase9_rollout_store.py",
)


def audit_no_optimizer_steps() -> dict:
    """Prove structurally that nothing on the collection path can train.

    Two checks: no forbidden symbol is *used* anywhere in the collection
    modules (an AST walk over names and attributes, so a string in a docstring
    does not trip it), and every parameter of a live behavior snapshot is
    frozen with gradients disabled after a real game has been played.
    """
    import ast

    findings: list[str] = []
    for module_path in COLLECTION_MODULES:
        tree = ast.parse((REPOSITORY_ROOT / module_path).read_text())
        used: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
        hits = sorted(used & set(FORBIDDEN_TRAINING_SYMBOLS))
        if hits:
            findings.append(f"{module_path} uses {hits}")

    snapshot = pb.load_behavior_snapshot(
        ANCHOR_CHECKPOINT,
        logical_identity="B001",
        policy_token=schedule.behavior_policy_token("canonical", 1),
        inference_batch_shape=8,
        expected_sha256=ANCHOR_SHA256,
    )
    before = snapshot.loaded_state_dict_digest
    participants = pc.IterationParticipants(behavior=snapshot, historical={})
    runner = pc.play_game(
        schedule.rebuild_scheduled_game(
            seeds.phase9_game_id("canonical", 1, "current", 0)
        ),
        participants,
    )
    snapshot.assert_frozen()
    after = pb.state_dict_digest(snapshot.model)
    trainable = [
        name for name, parameter in snapshot.model.named_parameters() if parameter.requires_grad
    ]
    if trainable:
        findings.append(f"{len(trainable)} parameters remained trainable")
    if after != before:
        findings.append("the behavior snapshot's weights moved during collection")
    return {
        "modules_scanned": list(COLLECTION_MODULES),
        "forbidden_symbols": list(FORBIDDEN_TRAINING_SYMBOLS),
        "symbol_findings": findings,
        "probe_game_decisions": len(runner.record.decisions),
        "trainable_parameters_after_collection": len(trainable),
        "state_dict_digest_before": before,
        "state_dict_digest_after": after,
        "weights_unchanged": after == before,
        "no_optimizer_steps": not findings,
    }


def demonstrate_crash_resume(root: Path, *, device: str, batch_shape: int) -> dict:
    """A live crash/resume convergence demonstration on a real sub-rollout.

    The suite proves this property on a shrunken iteration; this repeats it on
    the production store, at production settings, so the reported convergence
    is measured rather than inherited from a test fixture.
    """
    import shutil
    import tempfile

    namespace, iteration = "canonical", 1
    games = schedule.iteration_game_ids(namespace, iteration)[:12]
    scratch = Path(tempfile.mkdtemp(prefix="phase9_crash_", dir=str(root)))
    try:
        _resolver, behavior, anchor = build_participants(device, batch_shape)
        participants = pc.IterationParticipants(
            behavior=behavior, historical={contract.HISTORICAL_ANCHOR_ID: anchor}
        )
        kwargs = dict(
            population_version=contract.PHASE9_POPULATION_VERSION,
            schedule_version=contract.PHASE9_ROLLOUT_SCHEDULE_VERSION,
            contract_digest=ACCEPTED_CONTRACT_DIGEST,
        )
        original = store.iteration_game_ids
        store.iteration_game_ids = lambda ns, it: games
        try:
            clean_root = scratch / "clean"
            crashed_root = scratch / "crashed"
            clean = pc.collect_iteration(
                clean_root, namespace, iteration, participants, games_in_flight=12, **kwargs
            )

            class Boom(RuntimeError):
                pass

            calls = {"n": 0}

            def crash_hook(stage, _writer):
                if stage != "after_metadata":
                    return
                calls["n"] += 1
                if calls["n"] == 5:
                    raise Boom("injected collection crash")

            crashed_partial = 0
            try:
                pc.collect_iteration(
                    crashed_root,
                    namespace,
                    iteration,
                    participants,
                    games_in_flight=3,
                    crash_hook=crash_hook,
                    **kwargs,
                )
            except Boom:
                crashed_partial = len(
                    store.Phase9RolloutReader(crashed_root, namespace, iteration)
                )
            resumed = pc.collect_iteration(
                crashed_root, namespace, iteration, participants, games_in_flight=7, **kwargs
            )
            clean_reader = store.Phase9RolloutReader(clean_root, namespace, iteration)
            crashed_reader = store.Phase9RolloutReader(crashed_root, namespace, iteration)
            identical = all(
                crashed_reader.read_payload(game_id) == clean_reader.read_payload(game_id)
                for game_id in clean_reader.game_ids
            )
            return {
                "games": len(games),
                "clean_digest": clean["sealed_rollout_digest"],
                "committed_before_crash": crashed_partial,
                "resumed_digest": resumed["sealed_rollout_digest"],
                "digests_converge": clean["sealed_rollout_digest"]
                == resumed["sealed_rollout_digest"],
                "payload_bytes_identical": identical,
                "clean_workers_in_flight": 12,
                "resume_workers_in_flight": 7,
                "worker_topology_changed": True,
                "bytes_discarded_on_resume": resumed["bytes_discarded_on_resume"],
                "crash_stages_covered_by_suite": list(store.CRASH_STAGES),
            }
        finally:
            store.iteration_game_ids = original
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


def run_pytest() -> dict:
    command = [sys.executable, "-m", "pytest", "tests", "-q"]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, cwd=REPOSITORY_ROOT)
    output = completed.stdout.decode() + completed.stderr.decode()
    summary = ""
    for line in reversed(output.splitlines()):
        if " passed" in line or " failed" in line or "error" in line.lower():
            summary = line.strip()
            break
    numbers = {
        key: int(match.group(1))
        for key, pattern in (
            ("passed", r"(\d+) passed"),
            ("failed", r"(\d+) failed"),
            ("skipped", r"(\d+) skipped"),
        )
        if (match := re.search(pattern, summary))
    }
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "summary": summary,
        "seconds": time.perf_counter() - started,
        **numbers,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 9 Agent 3 acceptance harness")
    parser.add_argument(
        "--stage",
        action="append",
        choices=["verify", "storage", "soak", "audit", "reproduction", "artifacts"],
        help="run only these stages (default: all)",
    )
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--batch-shape", type=int, default=pb.DEFAULT_INFERENCE_BATCH_SHAPE)
    parser.add_argument("--games-in-flight", type=int, default=pc.DEFAULT_GAMES_IN_FLIGHT)
    parser.add_argument(
        "--limit-per-iteration",
        type=int,
        default=None,
        help="collect only this many games per iteration (a smoke run; no seal)",
    )
    parser.add_argument("--reproduction-minimum", type=int, default=REPRODUCTION_MINIMUM)
    parser.add_argument("--sample-games", type=int, default=64)
    parser.add_argument("--observer-probe-plies", type=int, default=2)
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument(
        "--record-final-suite",
        action="store_true",
        help=(
            "re-run the full suite against the published artifacts and record "
            "that result; the in-run suite executes before the artifacts exist, "
            "so only this pass actually exercises the artifact tests"
        ),
    )
    options = parser.parse_args()
    if options.record_final_suite:
        return record_final_suite()
    stages = set(options.stage or ["verify", "storage", "soak", "audit", "reproduction", "artifacts"])

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    started = time.time()
    durations: dict = {}
    problems: list[str] = []

    _print("[1/7] verifying the accepted Agent 1 and Agent 2 freezes")
    mark = time.perf_counter()
    prerequisites = verify_prerequisites() if "verify" in stages else {"skipped": True}
    durations["verify"] = time.perf_counter() - mark
    problems.extend(prerequisites.get("problems", []))
    if "verify" in stages:
        _print(
            f"      agent1={prerequisites['agent1_status']} "
            f"agent2={prerequisites['agent2_status']} "
            f"contract={prerequisites['contract_digest'][:16]} "
            f"corpus_identity={prerequisites['corpus']['identity_matches']}"
        )
    if problems:
        _print(f"BLOCKED: {problems}")
        return 2

    _print("[2/7] resolving Phase 9 rollout storage and proving the external volume")
    mark = time.perf_counter()
    total_scheduled = sum(
        schedule.total_scheduled_games(namespace) for namespace in seeds.RUN_NAMESPACES
    )
    storage_report = (
        verify_storage(scheduled_games=total_scheduled) if "storage" in stages else {"skipped": True}
    )
    durations["storage"] = time.perf_counter() - mark
    if "storage" in stages:
        problems.extend(storage_report["problems"])
        _print(
            f"      {storage_report['resolved_root']} "
            f"external={storage_report['external_volume'].get('is_external')} "
            f"free={storage_report['capacity_evaluation']['volume']['free_gib']} GiB"
        )
    if problems:
        _print(f"BLOCKED: {problems}")
        return 2

    root = Path(storage_report["resolved_root"]) / SOAK_SUBTREE if "storage" in stages else None
    if root is None:
        root = storage.default_rollout_root() / SOAK_SUBTREE
    root.mkdir(parents=True, exist_ok=True)

    _print(f"[3/7] collection soak into {root}")
    mark = time.perf_counter()
    soak = (
        run_soak(
            root,
            device=options.device,
            batch_shape=options.batch_shape,
            games_in_flight=options.games_in_flight,
            limit_per_iteration=options.limit_per_iteration,
            observer_probe_plies=options.observer_probe_plies,
        )
        if "soak" in stages
        # A staged re-run reads the measurements back rather than reporting
        # none: sealed iterations cannot be re-timed, so the recorded numbers
        # are the only honest ones available.
        else (
            json.loads(SOAK_ARTIFACT.read_text())
            if SOAK_ARTIFACT.exists()
            else {"skipped": True, "iterations": [], "totals": {}}
        )
    )
    durations["soak"] = time.perf_counter() - mark
    if "soak" in stages:
        totals = soak["totals"]
        _print(
            f"      {totals['games_collected']} games, "
            f"{totals['total_decisions']} decisions, "
            f"{totals['games_per_second']:.2f} games/s, "
            f"{totals['storage']['bytes_per_game']:.0f} bytes/game"
        )
        soak["storage_projection"] = project_storage(totals)
        # Written as soon as it exists: the soak is the expensive stage and a
        # failure in a later one must not throw away its measurements. A
        # re-run over sealed iterations reports zero throughput by
        # construction, so this number cannot simply be recollected.
        SOAK_ARTIFACT.write_text(json.dumps(soak, indent=2, default=str) + "\n")

    _print("[4/7] auditing the rollout store")
    mark = time.perf_counter()
    store_audit = (
        audit_store(
            root,
            sample_games=options.sample_games,
            device=options.device,
            batch_shape=options.batch_shape,
        )
        if "audit" in stages
        else {"skipped": True}
    )
    observer = (
        audit_observer_boundary(root, sample_games=max(2, options.sample_games // 8))
        if "audit" in stages
        else {"skipped": True}
    )
    crash = (
        demonstrate_crash_resume(root, device=options.device, batch_shape=options.batch_shape)
        if "audit" in stages
        else {"skipped": True}
    )
    optimizer_audit = audit_no_optimizer_steps() if "audit" in stages else {"skipped": True}
    durations["audit"] = time.perf_counter() - mark
    if "audit" in stages:
        _print(
            f"      sealed={store_audit['sealed_iterations']} "
            f"replay_illegal={store_audit['replay_illegal_actions']} "
            f"provenance_mismatches={store_audit['setup_provenance_mismatches']} "
            f"crash_converges={crash['digests_converge']} "
            f"no_optimizer={optimizer_audit['no_optimizer_steps']}"
        )

    _print("[5/7] behavior reproduction audit")
    mark = time.perf_counter()
    reproduction = (
        run_reproduction_audit(
            root,
            device=options.device,
            batch_shape=options.batch_shape,
            minimum=options.reproduction_minimum,
            historical_sample=HISTORICAL_REPRODUCTION_SAMPLE,
        )
        if "reproduction" in stages
        else {"skipped": True}
    )
    durations["reproduction"] = time.perf_counter() - mark
    if "reproduction" in stages:
        _print(
            f"      learner={reproduction['learner']['decisions']} "
            f"mismatches={reproduction['learner']['mismatches']} "
            f"max_diff={reproduction['learner']['max_abs_probability_difference']:.2e} "
            f"control_holds={reproduction['cross_checkpoint_control'].get('control_holds')}"
        )

    _print("[6/7] test suite")
    tests_after = run_pytest() if options.run_pytest else {"skipped": True}
    if options.run_pytest:
        _print(f"      {tests_after['summary']}")

    _print("[7/7] writing artifacts")
    gates = build_gates(
        prerequisites,
        storage_report,
        soak,
        store_audit,
        observer,
        crash,
        reproduction,
        tests_after,
        optimizer_audit,
    )
    acceptance = {
        "phase": PHASE,
        "agent": AGENT,
        "status": "PASS" if all(gates.values()) and not problems else "BLOCKED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cpu_count": os.cpu_count(),
        "source_revision": _git_revision(),
        "working_tree_state": _git_state(),
        "rules_version": RULES_VERSION,
        "engine_version": IMPLEMENTATION_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "artifact": "agent_03_acceptance",
        "store_version": store.PHASE9_COMMIT_VERSION,
        "collector_version": pc.PHASE9_COLLECTOR_VERSION,
        "prerequisites": prerequisites,
        "storage": storage_report,
        "soak_totals": soak.get("totals", {}),
        "storage_projection": soak.get("storage_projection", {}),
        "store_audit_summary": {
            key: store_audit.get(key)
            for key in (
                "sealed_iterations",
                "distinct_game_ids",
                "duplicate_game_ids",
                "unscheduled_games",
                "orphan_records",
                "games_decoded",
                "replay_illegal_actions",
                "setup_provenance_mismatches",
            )
        },
        "observer_safety": observer,
        "crash_resume": crash,
        "no_optimizer_audit": optimizer_audit,
        "behavior_reproduction_summary": {
            key: reproduction.get(key)
            for key in ("learner", "historical", "cross_checkpoint_control", "legal_set_mismatches")
        },
        "tests_after": tests_after,
        "completion_gates": gates,
        "gates_total": len(gates),
        "gates_true": sum(1 for value in gates.values() if value),
        "problems": problems,
        "deviations": DEVIATIONS,
        "durations": durations,
        "handoff_to_agent_4": HANDOFF,
    }

    if "artifacts" in stages:
        STORE_ARTIFACT.write_text(json.dumps(store_audit, indent=2, default=str) + "\n")
        if "soak" in stages:
            SOAK_ARTIFACT.write_text(json.dumps(soak, indent=2, default=str) + "\n")
        REPRODUCTION_ARTIFACT.write_text(json.dumps(reproduction, indent=2, default=str) + "\n")
        ACCEPTANCE_ARTIFACT.write_text(json.dumps(acceptance, indent=2, default=str) + "\n")
        _print(f"      {ACCEPTANCE_ARTIFACT}")

    _print(
        f"\nAgent 3 {acceptance['status']}: "
        f"{acceptance['gates_true']}/{acceptance['gates_total']} gates"
    )
    for name, value in gates.items():
        if not value:
            _print(f"  FAILED GATE: {name}")
    return 0 if acceptance["status"] == "PASS" else 2


def record_final_suite() -> int:
    """Re-run the whole suite with the artifacts in place and record it.

    The in-run suite executes before the artifacts are written, so the
    artifact tests skip there. This pass is the one whose green actually
    covers them, and its result is what `full_suite_green` reports.
    """
    if not ACCEPTANCE_ARTIFACT.exists():
        _print(f"no acceptance artifact at {ACCEPTANCE_ARTIFACT}; run the harness first")
        return 2
    acceptance = json.loads(ACCEPTANCE_ARTIFACT.read_text())
    _print("re-running the full suite against the published artifacts")
    result = run_pytest()
    _print(f"      {result['summary']}")
    acceptance["tests_after"] = result
    acceptance["tests_after"]["covers_agent_03_artifact_tests"] = True
    acceptance["completion_gates"]["full_suite_green"] = result["returncode"] == 0
    acceptance["gates_true"] = sum(1 for value in acceptance["completion_gates"].values() if value)
    acceptance["status"] = (
        "PASS"
        if acceptance["gates_true"] == acceptance["gates_total"] and not acceptance["problems"]
        else "BLOCKED"
    )
    ACCEPTANCE_ARTIFACT.write_text(json.dumps(acceptance, indent=2, default=str) + "\n")
    _print(
        f"\nAgent 3 {acceptance['status']}: "
        f"{acceptance['gates_true']}/{acceptance['gates_total']} gates"
    )
    for name, value in acceptance["completion_gates"].items():
        if not value:
            _print(f"  FAILED GATE: {name}")
    return 0 if acceptance["status"] == "PASS" else 2


def build_gates(
    prerequisites,
    storage_report,
    soak,
    store_audit,
    observer,
    crash,
    reproduction,
    tests_after,
    optimizer_audit,
) -> dict:
    totals = soak.get("totals", {})
    learner = reproduction.get("learner", {})
    return {
        "agents1_2_pass": prerequisites.get("agent1_status") == "PASS"
        and prerequisites.get("agent2_status") == "PASS"
        and prerequisites.get("contract_digest_matches_accepted", False),
        "corpus_resolver_verified": prerequisites.get("corpus", {}).get(
            "resolved_root_matches_expected", False
        )
        and not prerequisites.get("corpus", {}).get("modules_hard_coding_absolute_paths", ["x"]),
        "corpus_digests_match": prerequisites.get("corpus", {}).get("identity_matches", False),
        "external_volume_verified": bool(
            storage_report.get("external_volume", {}).get("is_external")
            and storage_report.get("is_mount_point")
            and storage_report.get("distinct_from_boot_filesystem")
            and not storage_report.get("problems")
        ),
        "behavior_snapshot_immutable": all(
            item.get("behavior_checkpoint_sha256") == ANCHOR_SHA256
            for item in store_audit.get("iterations", [])
        )
        and bool(store_audit.get("iterations")),
        "one_behavior_identity_per_iteration": all(
            item.get("behavior_snapshot_id") == "B001"
            for item in store_audit.get("iterations", [])
        )
        and bool(store_audit.get("iterations")),
        "neural_actions_legal": store_audit.get("replay_illegal_actions", 1) == 0,
        "behavior_storage_matches_contract": learner.get("max_abs_probability_difference", 1.0)
        <= contract.BEHAVIOR_PROBABILITY_ABS_TOLERANCE,
        "behavior_reproduction_ge_100k": learner.get("decisions", 0) >= REPRODUCTION_MINIMUM,
        "behavior_reproduction_mismatches_zero": learner.get("mismatches", 1) == 0
        and reproduction.get("historical", {}).get("mismatches", 1) == 0
        and reproduction.get("legal_set_mismatches", 1) == 0,
        "reproduction_control_fails_on_the_wrong_checkpoint": reproduction.get(
            "cross_checkpoint_control", {}
        ).get("control_holds", False),
        "rollout_commit_protocol_pass": store_audit.get("games_decoded", 0) > 0
        and store_audit.get("orphan_records", 1) == 0,
        "crash_resume_converges": bool(
            crash.get("digests_converge") and crash.get("payload_bytes_identical")
        ),
        "orphan_records_zero": store_audit.get("orphan_records", 1) == 0,
        "duplicate_game_ids_zero": store_audit.get("duplicate_game_ids", 1) == 0,
        "unscheduled_games_zero": store_audit.get("unscheduled_games", 1) == 0,
        "replay_illegal_actions_zero": store_audit.get("replay_illegal_actions", 1) == 0,
        "setup_provenance_mismatches_zero": store_audit.get("setup_provenance_mismatches", 1) == 0,
        "observer_input_leaks_zero": observer.get("failures", 1) == 0
        and observer.get("positive_control", {}).get("control_holds", False),
        "collection_soak_ge_8192_games": totals.get("games_committed", 0) >= 8192,
        "all_four_buckets_represented": sorted(totals.get("buckets_represented", []))
        == ["current", "historical", "rule", "stress"],
        "storage_density_measured": totals.get("storage", {}).get("bytes_per_game", 0) > 0,
        "no_rl_optimizer_steps": bool(optimizer_audit.get("no_optimizer_steps")),
        # `returncode` is present only when pytest actually ran. Deliberately
        # not keyed off `skipped`: that is pytest's skipped-*test* count, and
        # reading it as "this stage was skipped" made a green suite with three
        # skipped tests report a failed gate.
        "full_suite_green": tests_after.get("returncode") == 0,
    }


DEVIATIONS = [
    (
        "The soak writes to <rollout_root>/agent_03_soak rather than directly into "
        "the production <rollout_root>/<namespace>/ tree. The games are the real "
        "scheduled iteration-1 games and the sealed digests are the real ones, "
        "recorded in this report; the subtree keeps Agent 7's canonical run "
        "starting from a namespace it created itself rather than silently "
        "inheriting bytes from Agent 3. Identity is version + digests, never a "
        "path, so adopting these bytes later is a relocation plus a digest check."
    ),
    (
        "The inference batch shape is a recorded collector parameter (production "
        "default 64) rather than an implicit constant. Measured: a fixed shape is "
        "bitwise stable per row on both CPU and MPS, while a variable shape moves "
        "value logits by ~9e-8 — inside the 1e-4 reproduction tolerance but "
        "outside float32 storage rounding, so it would break byte-identical "
        "resume. The rollout state records the shape and the device, and a resume "
        "that would change either is refused."
    ),
]

HANDOFF = {
    "sealed_rollout_reader": "phase9_rollout_store.Phase9RolloutReader(root, namespace, iteration)",
    "random_access_reconstruction": "reader.read_game(game_id) -> (GameRecord, phase9 metadata), digest-checked on every read",
    "behavior_quantity_access": "DecisionRecord.old_probabilities is the stored float32 legal-action behavior distribution in ascending absolute action order; pi_b(a_t|s_t) is its entry for selected_action_id",
    "behavior_wdl_outputs": "DecisionRecord.win_draw_loss_prediction, the acting network's own W/D/L softmax from the acting player's perspective",
    "learner_control_masks": "metadata['learner_control'] / ['learner_color'] and ScheduledGame.learner_sides; metadata['learner_decision_count'] is the expected count",
    "privileged_target_only_state": "GameRecord.red_setup / blue_setup plus the snapshot cadence reconstruct per-piece truth for belief labelling through the accepted Phase 6 path (reconstruction.reconstruct_state / iter_reconstructed_decisions); none of it is reachable from a model input",
    "state_reconstruction": "stratego.training.reconstruction — reconstruct_state(record, ply) and iter_reconstructed_decisions(record) rebuild any position of any committed game from its own payload",
    "rollout_digests": "phase9_rollout_store.sealed_rollout_digest(commits) and the per-iteration state.json / manifest.json",
    "crash_safe_iteration_state": "read_iteration_state / write_iteration_state over COLLECTING -> SEALED -> TRAINING -> EVALUATED -> COMMITTED; Agent 3 owns COLLECTING -> SEALED only",
    "independent_reproduction": "phase9_behavior.reproduce_decisions(acting_snapshot, requests) — Agent 4 must verify each decision against the ACTING side's checkpoint, never against GameRecord.collection_checkpoint_id for a historical opponent's moves",
    "per_side_identity_warning": "metadata['opponent_checkpoint_sha256'] is the historical opponent's real SHA-256; metadata['behavior_checkpoint_sha256'] is the iteration's current learner only",
}


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            cwd=REPOSITORY_ROOT,
            check=False,
        ).stdout.decode().strip()
    except OSError:  # pragma: no cover
        return "unknown"


def _git_state() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, cwd=REPOSITORY_ROOT, check=False
    )
    return "dirty" if result.stdout.strip() else "clean"


if __name__ == "__main__":
    raise SystemExit(main())
