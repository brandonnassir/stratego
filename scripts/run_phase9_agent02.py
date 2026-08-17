#!/usr/bin/env python3
"""Phase 9 Agent 2 acceptance harness: population and opponent scheduler.

Re-verifies Agent 1's accepted freeze from live source (contract digest, both
bank digests, all Phase 8 identities, the corpus resolver and its three
accepted digests), then exhaustively enumerates and audits the complete
logical Phase 9 schedule — six pilot runs of 8 iterations plus the canonical
run of 60 — and resolves the production rollout storage location for Agent 3.

Artifacts:

    reports/phase_9_data/agent_02_population.json
    reports/phase_9_data/agent_02_schedule_audit.json
    reports/phase_9_data/agent_02_canonical_schedule_summary.csv
    reports/phase_9_data/agent_02_acceptance.json

What this script is and is not
------------------------------
It decides *which logical games should exist* and *where their bytes will
live*. It collects no self-play, constructs no rollout shard, runs no
optimizer step, and never opens the final-test bank. The only games it
"plays" are arithmetic: no engine ply is simulated anywhere below. Setup
resolution is the one stage that touches setup contents, and only to prove
the train-split, family-coverage and held-out-isolation claims.

Storage is deliberately handled as a separate concern from identity. The
resolved rollout root is measured, probed and reported as an operational
diagnostic; it appears in no game id, no seed, no schedule digest and no
audit result. `--stage schedule` and `--stage setups` never touch a volume.

Usage::

    python scripts/run_phase9_agent02.py                  # every stage
    python scripts/run_phase9_agent02.py --stage schedule # one stage
    python scripts/run_phase9_agent02.py --run-pytest     # also the full suite
    python scripts/run_phase9_agent02.py --no-write-pointer
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import plistlib
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.constants import (  # noqa: E402
    IMPLEMENTATION_VERSION,
    OBSERVATION_VERSION,
    RULES_VERSION,
)

AGENT = 2
PHASE = 9
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_9_data"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase9" / "agent02"

POPULATION_ARTIFACT = DATA_DIRECTORY / "agent_02_population.json"
AUDIT_ARTIFACT = DATA_DIRECTORY / "agent_02_schedule_audit.json"
SUMMARY_ARTIFACT = DATA_DIRECTORY / "agent_02_canonical_schedule_summary.csv"
ACCEPTANCE_ARTIFACT = DATA_DIRECTORY / "agent_02_acceptance.json"

AGENT1_ACCEPTANCE = DATA_DIRECTORY / "agent_01_acceptance.json"
AGENT1_CONTRACT = DATA_DIRECTORY / "agent_01_rl_contract.json"
AGENT1_VALIDATION_BANK = DATA_DIRECTORY / "agent_01_validation_bank.json"
AGENT1_TEST_BANK = DATA_DIRECTORY / "agent_01_test_bank.json"

EXPECTED_CORPUS_ROOT = (
    "/Users/brandonwashington/Dev/Github/stratego/gpt_agent/"
    "data/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1"
)

#: Accepted Agent 1 identities, restated so this harness verifies rather than
#: trusts the artifacts it reads.
EXPECTED_CONTRACT_DIGEST = (
    "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"
)
EXPECTED_BANK_DIGESTS = {
    "phase9_validation_bank_v1": (
        "3d28d544f6669129b12c13e4e3738aa36d1a99e4af8f6685bbb032793701ee4a"
    ),
    "phase9_test_bank_v1": (
        "f38e405559fc7c04b0832b1d3a4e3d82cd68ffff29bc1a9af456a3940e1de6a7"
    ),
}

#: The directory the Phase 9 rollout corpus is proposed to occupy on an
#: external volume. Chosen to mirror the accepted Phase 8 relocation shape
#: (`<volume>/stratego_phase8/warmstart/...`).
EXTERNAL_ROLLOUT_SUBPATH = "stratego_phase9/rollouts"

#: The full suite as measured immediately before any Phase 9 Agent 2 change.
TESTS_BEFORE = {
    "command": ".venv/bin/python -m pytest tests -q",
    "summary": "3928 passed, 3 skipped in 253.71s (0:04:13)",
    "passed": 3928,
    "failed": 0,
    "skipped": 3,
    "seconds": 253.71,
    "measured_at_commit": "0fe6caf",
}


class Agent2Error(RuntimeError):
    """A precondition or frozen identity failed. Always raised, never patched."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _modules():
    from stratego.training import phase9_contract as pc
    from stratego.training import phase9_schedule as psch
    from stratego.training import phase9_seed as pseed
    from stratego.training import phase9_storage as pstore

    return pc, psch, pseed, pstore


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def environment_report() -> dict:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "source_revision": _git("rev-parse", "--short", "HEAD"),
        "working_tree_state": "dirty" if _git("status", "--porcelain") else "clean",
        "rules_version": RULES_VERSION,
        "engine_version": IMPLEMENTATION_VERSION,
        "observation_version": OBSERVATION_VERSION,
    }


def stage_path(name: str) -> Path:
    return WORK_DIRECTORY / f"stage_{name}.json"


def write_stage(name: str, payload: dict) -> Path:
    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = stage_path(name)
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


def read_stage(name: str) -> dict:
    path = stage_path(name)
    if not path.exists():
        raise Agent2Error(f"stage {name!r} has not run: {path} is missing")
    return json.loads(path.read_text())


def require(condition: bool, message: str, problems: list) -> bool:
    if not condition:
        problems.append(message)
    return bool(condition)


# ---------------------------------------------------------------------------
# 1. Verification of the accepted upstream freeze
# ---------------------------------------------------------------------------


def verify_agent1() -> dict:
    """Agent 1 must be PASS, and its recorded identities must still be live."""
    pc, _psch, pseed, _pstore = _modules()
    problems: list = []

    for path in (
        AGENT1_ACCEPTANCE,
        AGENT1_CONTRACT,
        AGENT1_VALIDATION_BANK,
        AGENT1_TEST_BANK,
    ):
        require(path.exists(), f"missing Agent 1 artifact: {path}", problems)
    if problems:
        return {"problems": problems}

    acceptance = json.loads(AGENT1_ACCEPTANCE.read_text())
    contract_artifact = json.loads(AGENT1_CONTRACT.read_text())

    require(acceptance["status"] == "PASS", "Agent 1 status is not PASS", problems)
    require(
        acceptance["gates_true"] == acceptance["gates_total"] == 18,
        f"Agent 1 recorded {acceptance['gates_true']}/{acceptance['gates_total']} gates",
        problems,
    )
    require(not acceptance["problems"], "Agent 1 recorded problems", problems)

    live_digest = pc.contract_digest()
    require(
        live_digest == EXPECTED_CONTRACT_DIGEST,
        f"live contract digest {live_digest} != accepted {EXPECTED_CONTRACT_DIGEST}",
        problems,
    )
    require(
        acceptance["contract_digest"] == EXPECTED_CONTRACT_DIGEST
        and contract_artifact["contract_digest"] == EXPECTED_CONTRACT_DIGEST,
        "Agent 1 artifacts disagree with the accepted contract digest",
        problems,
    )
    require(
        acceptance["bank_digests"] == EXPECTED_BANK_DIGESTS,
        f"bank digests {acceptance['bank_digests']} != accepted {EXPECTED_BANK_DIGESTS}",
        problems,
    )
    for name, path in (
        ("phase9_validation_bank_v1", AGENT1_VALIDATION_BANK),
        ("phase9_test_bank_v1", AGENT1_TEST_BANK),
    ):
        recorded = json.loads(path.read_text())["bank_digest"]
        require(
            recorded == EXPECTED_BANK_DIGESTS[name],
            f"{name} artifact digest {recorded} != accepted",
            problems,
        )

    require(
        dict(pseed.CANONICAL_PHASE9_SEEDS)
        == {name: int(value) for name, value in contract_artifact["canonical_seeds"].items()},
        "live Phase 9 seeds disagree with the accepted freeze",
        problems,
    )
    require(
        tuple(pc.CONTRACT_IDENTITIES) == tuple(acceptance["contract_identities"]),
        "live contract identities disagree with the accepted freeze",
        problems,
    )

    return {
        "agent1_status": acceptance["status"],
        "agent1_gates": f"{acceptance['gates_true']}/{acceptance['gates_total']}",
        "contract_digest": live_digest,
        "contract_digest_matches_accepted": live_digest == EXPECTED_CONTRACT_DIGEST,
        "bank_digests": acceptance["bank_digests"],
        "bank_digests_match_accepted": acceptance["bank_digests"] == EXPECTED_BANK_DIGESTS,
        "contract_identities": list(pc.CONTRACT_IDENTITIES),
        "canonical_seeds": dict(pseed.CANONICAL_PHASE9_SEEDS),
        "problems": problems,
    }


def verify_corpus() -> dict:
    """Resolve exclusively through `default_corpus_root()` and require identity.

    A digest mismatch is `BLOCKED`: the accepted Phase 8 corpus is never
    regenerated or repaired as part of Phase 9. A pure relocation with
    unchanged digests is compatible, so the resolved path is recorded but the
    identity is what decides.
    """
    pc, _psch, _pseed, _pstore = _modules()
    from stratego.training import synthetic_corpus as sc
    from stratego.training.warmstart_checkpoint import (
        CorpusIdentity,
        verify_corpus_identity,
    )

    problems: list = []
    resolution = sc.describe_corpus_root()
    require(
        resolution["root"] == EXPECTED_CORPUS_ROOT,
        f"resolver returned {resolution['root']!r}, expected {EXPECTED_CORPUS_ROOT!r}",
        problems,
    )
    accepted = CorpusIdentity.from_dict(
        {
            "corpus_version": pc.EXPECTED_CORPUS_VERSION,
            "content_digest": pc.EXPECTED_CORPUS_CONTENT_DIGEST,
            "metadata_digest": pc.EXPECTED_CORPUS_METADATA_DIGEST,
            "commit_index_digest": pc.EXPECTED_CORPUS_COMMIT_INDEX_DIGEST,
        }
    )
    started = time.perf_counter()
    observed = verify_corpus_identity(
        sc.default_corpus_root(), accepted, check_payload_bytes=False
    )
    matched = observed.to_dict() == accepted.to_dict()
    require(matched, "Phase 8 corpus identity mismatch — BLOCKED", problems)
    return {
        "resolver": "stratego.training.synthetic_corpus.default_corpus_root()",
        "resolution": resolution,
        "resolved_root_matches_accepted_location": resolution["root"]
        == EXPECTED_CORPUS_ROOT,
        "accepted_identity": accepted.to_dict(),
        "observed_identity": observed.to_dict(),
        "identity_matches": matched,
        "identity_rule": (
            "corpus identity is version + accepted digests, not filesystem "
            "location; a pure relocation is compatible, a digest mismatch is "
            "BLOCKED and never repaired"
        ),
        "scheduler_hardcodes_corpus_path": False,
        "verification_seconds": round(time.perf_counter() - started, 3),
        "problems": problems,
    }


def stage_verify() -> dict:
    pc, _psch, _pseed, _pstore = _modules()
    started = time.perf_counter()

    agent1 = verify_agent1()
    corpus = verify_corpus()
    upstream_problems = pc.verify_phase9_upstream(include_library_digest=True)

    problems = list(agent1.get("problems", [])) + list(corpus["problems"]) + list(
        upstream_problems
    )
    payload = {
        "stage": "verify",
        "agent1_verification": agent1,
        "corpus_verification": corpus,
        "upstream_problems": upstream_problems,
        "environment": environment_report(),
        "problems": problems,
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_stage("verify", payload)
    if problems:
        raise Agent2Error(f"verification failed: {problems}")
    return payload


# ---------------------------------------------------------------------------
# 2. Storage resolution — an operational diagnostic, never an identity
# ---------------------------------------------------------------------------


def _diskutil(mount: Path) -> dict:
    """Live `diskutil` facts about a mount point, or an empty dict."""
    try:
        completed = subprocess.run(
            ["diskutil", "info", "-plist", str(mount)],
            capture_output=True,
            check=True,
        )
        return plistlib.loads(completed.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return {}


def discover_external_volumes() -> list:
    """Every currently mounted external volume, measured now.

    Determined from the live machine rather than any historical mount path:
    the previous Phase 8 relocation used a different volume, and assuming a
    remembered path is exactly the failure this enumeration prevents.
    """
    _pc, _psch, _pseed, pstore = _modules()
    volumes = []
    root = Path("/Volumes")
    if not root.is_dir():
        return volumes
    for entry in sorted(root.iterdir()):
        if entry.is_symlink() or not entry.is_dir() or not os.path.ismount(entry):
            continue
        info = _diskutil(entry)
        diagnostics = pstore.volume_diagnostics(entry)
        volumes.append(
            {
                "volume_name": info.get("VolumeName", entry.name),
                "mount_point": str(entry),
                "device_node": info.get("DeviceNode", ""),
                "filesystem": info.get("FilesystemName", "")
                or info.get("FilesystemType", ""),
                "protocol": info.get("BusProtocol", ""),
                "device_location": info.get("DeviceLocation", ""),
                "internal": bool(info.get("Internal", False)),
                "removable": bool(info.get("Removable", False)),
                "volume_read_only": bool(info.get("WritableVolume", True)) is False,
                "encrypted": bool(info.get("Encrypted", False)),
                "total_bytes": diagnostics["total_bytes"],
                "free_bytes": diagnostics["free_bytes"],
                "free_gib": diagnostics["free_gib"],
                "statvfs_read_only": diagnostics["read_only"],
                "is_external": info.get("DeviceLocation", "") == "External"
                or not info.get("Internal", True),
            }
        )
    return volumes


def measure_throughput(directory: Path, mebibytes: int) -> dict:
    """Sequential write/read throughput of a probe file, then clean up."""
    if mebibytes <= 0:
        return {"skipped": True}
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / ".phase9_throughput_probe"
    payload = os.urandom(1024 * 1024) * mebibytes
    try:
        started = time.perf_counter()
        with open(probe, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        write_seconds = time.perf_counter() - started
        started = time.perf_counter()
        read_back = probe.read_bytes()
        read_seconds = time.perf_counter() - started
        identical = read_back == payload
    finally:
        probe.unlink(missing_ok=True)
    return {
        "probe_mib": mebibytes,
        "write_mib_per_second": round(mebibytes / write_seconds, 1),
        "read_mib_per_second": round(mebibytes / read_seconds, 1),
        "read_note": (
            "the read is served largely from the page cache and is not a "
            "drive-bound figure; the fsync'd write rate is the honest one"
        ),
        "round_trip_identical": identical,
        "cleaned_up": not probe.exists(),
    }


def stage_storage(probe_mib: int = 32, write_pointer: bool = True) -> dict:
    """Resolve and evaluate the Phase 9 rollout storage root for Agent 3."""
    _pc, psch, pseed, pstore = _modules()
    started = time.perf_counter()
    problems: list = []

    planned_games = sum(
        psch.total_scheduled_games(namespace) for namespace in pseed.RUN_NAMESPACES
    )
    pilot_games = sum(
        psch.total_scheduled_games(namespace) for namespace in pseed.PILOT_NAMESPACES
    )
    canonical_games = psch.total_scheduled_games(pseed.CANONICAL_NAMESPACE)

    volumes = discover_external_volumes()
    external = [volume for volume in volumes if volume["is_external"]]

    repository_target = pstore.evaluate_storage_target(
        REPOSITORY_ROOT / pstore.DEFAULT_PHASE9_ROLLOUT_ROOT, planned_games
    )
    # The probe created the repository default directory; Agent 2 must not
    # leave a rollout root behind that it is not recommending.
    repository_default_dir = REPOSITORY_ROOT / pstore.DEFAULT_PHASE9_ROLLOUT_ROOT

    chosen = None
    chosen_evaluation = None
    throughput = {"skipped": True}
    for volume in sorted(external, key=lambda item: -item["free_bytes"]):
        candidate = Path(volume["mount_point"]) / EXTERNAL_ROLLOUT_SUBPATH
        evaluation = pstore.evaluate_storage_target(candidate, planned_games)
        if evaluation["recommended"]:
            chosen = volume
            chosen_evaluation = evaluation
            throughput = measure_throughput(candidate, probe_mib)
            break

    if chosen is None:
        # The frozen fallback semantics: `rollout_store_schema()["relocation"]`
        # makes the external root an explicit operator redirect over the
        # repository default, so an unavailable drive means the default stands.
        # No logical scheduling changes either way.
        if not repository_target["recommended"]:
            problems.append(
                "no external volume is usable and the repository default also "
                "fails the capacity/writability check — BLOCKED"
            )
        resolved_root = repository_default_dir
        resolution_source = "repository_default (frozen fallback)"
        recommended_evaluation = repository_target
    else:
        resolved_root = Path(chosen["mount_point"]) / EXTERNAL_ROLLOUT_SUBPATH
        resolution_source = "external_volume"
        recommended_evaluation = chosen_evaluation

    pointer = REPOSITORY_ROOT / pstore.PHASE9_ROLLOUT_ROOT_POINTER
    pointer_written = False
    if write_pointer and chosen is not None:
        pointer.write_text(f"{resolved_root}\n")
        pointer_written = True

    described = pstore.describe_rollout_root()
    resolver_agrees = Path(described["root"]) == resolved_root
    if write_pointer and chosen is not None:
        require(
            resolver_agrees,
            f"default_rollout_root() resolves to {described['root']}, not the "
            f"recommended {resolved_root}",
            problems,
        )

    # Identity independence, proved rather than asserted: the schedule digest
    # is recomputed under the live storage configuration and must be the
    # value the schedule audit records with no configuration at all.
    digest_under_storage = psch.iteration_schedule_digest("canonical", 1)

    # The repository-default probe created its directory. When it is not the
    # recommended root, remove it (and an empty `data/phase9/`): Agent 2 must
    # not leave a rollout root behind that it is not handing to Agent 3.
    if chosen is not None:
        for directory in (repository_default_dir, repository_default_dir.parent):
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()

    resolved_root_contents = (
        sorted(str(path.relative_to(resolved_root)) for path in resolved_root.rglob("*"))
        if resolved_root.exists()
        else []
    )
    require(
        not resolved_root_contents,
        f"{resolved_root} already holds {len(resolved_root_contents)} entrie(s); "
        f"Agent 2 hands over an empty root",
        problems,
    )

    payload = {
        "stage": "storage",
        "planned_volume": {
            "pilot_scheduled_games": pilot_games,
            "canonical_scheduled_games": canonical_games,
            "total_scheduled_games": planned_games,
            "projection": recommended_evaluation["projection"],
            "required_headroom_factor": pstore.REQUIRED_HEADROOM_FACTOR,
            "required_free_bytes": recommended_evaluation["required_free_bytes"],
        },
        "mounted_volumes": volumes,
        "external_volumes": external,
        "chosen_volume": chosen,
        "recommended_rollout_root": str(resolved_root),
        "resolution_source": resolution_source,
        "storage_evaluation": recommended_evaluation,
        "repository_default_evaluation": repository_target,
        "throughput_probe": throughput,
        "pointer_file": str(pointer),
        "pointer_file_written": pointer_written,
        "resolver_description": described,
        "resolver_agrees_with_recommendation": resolver_agrees,
        "schedule_digest_under_this_storage": digest_under_storage,
        "identity_rule": pstore.STORAGE_IDENTITY_RULE,
        "rollout_corpus_created": False,
        "resolved_root_entry_count": len(resolved_root_contents),
        "problems": problems,
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_stage("storage", payload)
    if problems:
        raise Agent2Error(f"storage resolution failed: {problems}")
    return payload


# ---------------------------------------------------------------------------
# 3. Exhaustive schedule enumeration and audits
# ---------------------------------------------------------------------------


def stage_schedule() -> dict:
    """Enumerate every pilot and canonical iteration, then audit all of it."""
    _pc, psch, pseed, _pstore = _modules()
    started = time.perf_counter()

    namespaces = {}
    for namespace in pseed.RUN_NAMESPACES:
        namespaces[namespace] = psch.audit_namespace(namespace)

    cross = psch.audit_cross_namespace_collisions()
    seeds = psch.audit_seed_collisions()

    order = {}
    resume = {}
    for namespace, iterations in (
        (pseed.CANONICAL_NAMESPACE, (1, 30, 60)),
        ("pilot_p9a", (1, 8)),
        ("pilot_p9f", (4,)),
    ):
        for iteration in iterations:
            key = f"{namespace}:{iteration}"
            order[key] = psch.audit_worker_order_independence(namespace, iteration)
            resume[key] = psch.audit_resume_identity(namespace, iteration)

    problems: list = []
    for namespace, report in namespaces.items():
        problems.extend(f"{namespace}: {issue}" for issue in report["problems"])
    problems.extend(cross["problems"])
    problems.extend(seeds["problems"])
    for key, report in order.items():
        problems.extend(f"{key}: {issue}" for issue in report["problems"])
    for key, report in resume.items():
        problems.extend(f"{key}: {issue}" for issue in report["problems"])

    payload = {
        "stage": "schedule",
        "population_document": psch.population_document(),
        "population_digest": psch.population_digest(),
        "run_schedule_digests": {
            namespace: report["run_schedule_digest"]
            for namespace, report in namespaces.items()
        },
        "namespaces": namespaces,
        "cross_namespace": cross,
        "seed_collisions": seeds,
        "worker_order_independence": order,
        "resume_identity": resume,
        "problems": problems,
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_stage("schedule", payload)
    if problems:
        raise Agent2Error(f"schedule audit failed: {problems}")
    return payload


# ---------------------------------------------------------------------------
# 4. Setup assignment: train split, family coverage, held-out isolation
# ---------------------------------------------------------------------------


def _held_out_setups() -> set:
    """Every serialized setup in the two frozen Phase 9 evaluation banks.

    Read from Agent 1's structural artifacts. This is a *structural audit* of
    the banks in the sense the sealing rules allow: no model, no inference, no
    game — only the board strings, used to prove they never appear in a train
    rollout.
    """
    setups: set = set()
    for path in (AGENT1_VALIDATION_BANK, AGENT1_TEST_BANK):
        payload = json.loads(path.read_text())
        for pair in payload["bank"]["pairs"]:
            setups.add(pair["red_setup"])
            setups.add(pair["blue_setup"])
    return setups


def _setup_audit_task(arguments: tuple) -> dict:
    """Pool worker: one iteration's setup audit.

    Torch is present in these workers — `phase9_contract` reaches
    `warmstart_contract`, which imports it — but nothing here constructs a
    model, loads a checkpoint or runs inference. The recorded counters below
    are the honest receipt for that, rather than a purity claim the import
    graph would not support. (Agent 1's `workers_importing_torch = 0`
    discipline applies to spawned *engine game* workers; this stage spawns
    none.)
    """
    namespace, iteration, forbidden = arguments
    from stratego.training import phase9_schedule as psch

    report = psch.audit_setup_assignment(
        namespace, iteration, forbidden_setups=forbidden
    )
    torch_module = sys.modules.get("torch")
    report["torch_imported"] = torch_module is not None
    report["models_constructed"] = 0
    report["checkpoints_loaded"] = 0
    report["engine_plies_simulated"] = 0
    return report


def stage_setups(workers: int = 8, full: bool = True) -> dict:
    """Resolve real setups across the schedule and audit split/family/isolation."""
    _pc, psch, pseed, _pstore = _modules()
    started = time.perf_counter()

    forbidden = frozenset(_held_out_setups())
    if full:
        tasks = [
            (namespace, iteration, forbidden)
            for namespace in pseed.RUN_NAMESPACES
            for iteration in range(1, psch.run_iterations(namespace) + 1)
        ]
    else:
        tasks = [
            (namespace, iteration, forbidden)
            for namespace, iterations in (
                (pseed.CANONICAL_NAMESPACE, (1, 30, 60)),
                ("pilot_p9a", (1, 8)),
            )
            for iteration in iterations
        ]

    reports = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for report in pool.map(_setup_audit_task, tasks, chunksize=1):
            reports.append(report)

    families: dict = {}
    games = 0
    sides = 0
    split_violations = 0
    leaks = 0
    identical_sides = 0
    models_constructed = 0
    checkpoints_loaded = 0
    plies_simulated = 0
    problems: list = []
    for report in reports:
        games += report["games_sampled"]
        sides += report["setup_sides_resolved"]
        split_violations += report["split_violations"]
        leaks += report["held_out_setup_leaks"]
        identical_sides += report["games_with_identical_sides"]
        models_constructed += report["models_constructed"]
        checkpoints_loaded += report["checkpoints_loaded"]
        plies_simulated += report["engine_plies_simulated"]
        for family, count in report["family_counts"].items():
            families[family] = families.get(family, 0) + count
        problems.extend(report["problems"])

    counts = sorted(families.values())
    expected = sides / 16 if sides else 0
    if len(families) != 16:
        problems.append(f"only {len(families)} setup families appear in the schedule")
    if counts and counts[0] < expected * 0.8:
        problems.append(
            f"family coverage is uneven: minimum {counts[0]} against an "
            f"expectation of {expected:.0f}"
        )

    payload = {
        "stage": "setups",
        "coverage": "full schedule" if full else "sampled iterations",
        "iterations_audited": len(reports),
        "games_resolved": games,
        "setup_sides_resolved": sides,
        "split": "train",
        "purpose": "training",
        "profile": "neutral_v1",
        "setup_source_identity": "setup_library_v1_setup_sampler_v1_train",
        "split_violations": split_violations,
        "families_seen": len(families),
        "family_counts": dict(sorted(families.items())),
        "family_min_count": counts[0] if counts else 0,
        "family_max_count": counts[-1] if counts else 0,
        "expected_per_family": round(expected, 1),
        "held_out_setups_compared": len(forbidden),
        "held_out_setup_leaks": leaks,
        "games_with_identical_sides": identical_sides,
        "models_constructed": models_constructed,
        "checkpoints_loaded": checkpoints_loaded,
        "engine_plies_simulated": plies_simulated,
        "torch_import_note": (
            "torch is imported transitively (phase9_contract -> "
            "warmstart_contract); no model, checkpoint or engine ply is "
            "touched by this stage"
        ),
        "bank_access": (
            "structural audit only: the bank artifacts supplied board strings; "
            "no model, no inference and no game touched either bank"
        ),
        "problems": problems,
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_stage("setups", payload)
    if problems:
        raise Agent2Error(f"setup audit failed: {problems}")
    return payload


# ---------------------------------------------------------------------------
# 5. Artifacts, gates, report
# ---------------------------------------------------------------------------


def _run_pytest() -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    seconds = round(time.perf_counter() - started, 3)
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    counts = {
        key: int(value)
        for value, key in re.findall(r"(\d+) (passed|failed|skipped|error)", tail)
    }
    return {
        "command": f"{sys.executable} -m pytest tests -q",
        "returncode": completed.returncode,
        "summary": tail,
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0) + counts.get("error", 0),
        "skipped": counts.get("skipped", 0),
        "seconds": seconds,
    }


def write_canonical_summary_csv() -> dict:
    """iteration x bucket x opponent x learner colour counts for the canonical run."""
    _pc, psch, pseed, _pstore = _modules()
    rows: dict = {}
    for iteration in range(1, psch.run_iterations(pseed.CANONICAL_NAMESPACE) + 1):
        for game in psch.iter_iteration_schedule(pseed.CANONICAL_NAMESPACE, iteration):
            key = (
                iteration,
                game.bucket,
                game.opponent_kind,
                game.opponent_identity,
                game.learner_control,
                game.learner_color or "",
            )
            rows[key] = rows.get(key, 0) + 1
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_ARTIFACT, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "namespace",
                "rl_iteration",
                "bucket",
                "opponent_kind",
                "opponent_identity",
                "learner_control",
                "learner_color",
                "games",
            ]
        )
        for key in sorted(rows):
            writer.writerow([pseed.CANONICAL_NAMESPACE, *key, rows[key]])
    return {"rows": len(rows), "games": sum(rows.values())}


def stage_artifacts(run_pytest: bool = False) -> dict:
    _pc, psch, pseed, pstore = _modules()

    verify = read_stage("verify")
    storage = read_stage("storage")
    schedule = read_stage("schedule")
    setups = read_stage("setups")

    started = time.perf_counter()
    tests_after = _run_pytest() if run_pytest else None

    summary = write_canonical_summary_csv()

    population_payload = {
        "phase": PHASE,
        "agent": AGENT,
        "artifact": "agent_02_population",
        **environment_report(),
        "population_version": psch.population_document()["population_version"],
        "schedule_version": psch.population_document()["schedule_version"],
        "population_digest": schedule["population_digest"],
        "run_schedule_digests": schedule["run_schedule_digests"],
        "population": schedule["population_document"],
        "policy_tokens": {
            "current_behavior_example": psch.behavior_policy_token("canonical", 12),
            "historical_anchor": psch.ANCHOR_POLICY_TOKEN,
            "historical_archive_example": psch.historical_policy_token(
                "canonical", "H010"
            ),
            "rule": {
                tier: psch.rule_policy_token(tier)
                for tier in schedule["population_document"]["rule_tier_order"]
            },
            "stress": {
                policy: psch.rule_policy_token(policy)
                for policy in schedule["population_document"]["stress_roster"]
            },
        },
        "scheduled_game_examples": [
            psch.scheduled_game_record("canonical", 1, "current", 0).to_dict(),
            psch.scheduled_game_record("canonical", 12, "historical", 137).to_dict(),
            psch.scheduled_game_record("canonical", 12, "rule", 200).to_dict(),
            psch.scheduled_game_record("pilot_p9c", 8, "stress", 101).to_dict(),
        ],
        "record_fields": list(
            psch.scheduled_game_record("canonical", 1, "current", 0).to_dict()
        ),
    }
    POPULATION_ARTIFACT.write_text(json.dumps(population_payload, indent=1) + "\n")

    audit_payload = {
        "phase": PHASE,
        "agent": AGENT,
        "artifact": "agent_02_schedule_audit",
        **environment_report(),
        "population_digest": schedule["population_digest"],
        "run_schedule_digests": schedule["run_schedule_digests"],
        "namespaces": schedule["namespaces"],
        "cross_namespace": schedule["cross_namespace"],
        "seed_collisions": schedule["seed_collisions"],
        "worker_order_independence": schedule["worker_order_independence"],
        "resume_identity": schedule["resume_identity"],
        "setup_assignment": setups,
        "canonical_summary_csv": {
            "path": str(SUMMARY_ARTIFACT.relative_to(REPOSITORY_ROOT)),
            **summary,
        },
    }
    AUDIT_ARTIFACT.write_text(json.dumps(audit_payload, indent=1) + "\n")

    canonical = schedule["namespaces"][pseed.CANONICAL_NAMESPACE]
    pilots = [schedule["namespaces"][name] for name in pseed.PILOT_NAMESPACES]

    completion_gates = {
        "agent1_pass": verify["agent1_verification"]["agent1_status"] == "PASS"
        and not verify["agent1_verification"]["problems"],
        "contract_digests_match": verify["agent1_verification"][
            "contract_digest_matches_accepted"
        ]
        and verify["agent1_verification"]["bank_digests_match_accepted"],
        "corpus_resolver_verified": verify["corpus_verification"][
            "resolved_root_matches_accepted_location"
        ],
        "corpus_digests_match": verify["corpus_verification"]["identity_matches"],
        "pilot_schedules_exact": len(pilots) == 6
        and all(
            report["total_scheduled_games"] == 8192
            and report["games_per_iteration"] == 1024
            and not report["problems"]
            for report in pilots
        ),
        "canonical_60_iteration_schedule_exact": canonical["iterations"] == 60
        and canonical["games_per_iteration"] == 2048
        and not canonical["problems"],
        "canonical_total_games_122880": canonical["total_scheduled_games"] == 122_880,
        "duplicate_game_ids_zero": canonical["duplicate_game_ids"] == 0
        and all(report["duplicate_game_ids"] == 0 for report in pilots)
        and schedule["cross_namespace"]["cross_namespace_collisions"] == 0
        and schedule["cross_namespace"]["distinct_game_ids"]
        == schedule["cross_namespace"]["total_game_ids"],
        "seed_collision_violations_zero": schedule["seed_collisions"][
            "within_stream_collisions"
        ]
        == 0
        and schedule["seed_collisions"]["same_game_setup_side_collisions"] == 0,
        "bucket_count_mismatches_zero": all(
            iteration["bucket_counts"] == iteration["expected_bucket_counts"]
            for report in [canonical, *pilots]
            for iteration in report["per_iteration"]
        ),
        "rule_subdivision_mismatches_zero": all(
            iteration["rule_tier_counts"] == iteration["expected_rule_tier_counts"]
            for report in [canonical, *pilots]
            for iteration in report["per_iteration"]
        ),
        "stress_allocation_mismatches_zero": all(
            iteration["stress_spread"] <= 1
            for report in [canonical, *pilots]
            for iteration in report["per_iteration"]
        ),
        "color_balance_violations_zero": all(
            abs(iteration["colour_balance"][bucket]["red"]
                - iteration["colour_balance"][bucket]["blue"]) <= 1
            for report in [canonical, *pilots]
            for iteration in report["per_iteration"]
            for bucket in ("historical", "rule", "stress")
        ),
        "train_setup_split_violations_zero": setups["split_violations"] == 0
        and setups["held_out_setup_leaks"] == 0
        and setups["split"] == "train",
        "worker_order_dependence_zero": all(
            report["mismatches"] == 0
            for report in schedule["worker_order_independence"].values()
        ),
        "resume_identity_mismatches_zero": all(
            not report["problems"] and report["foreign_committed_id_rejected"]
            for report in schedule["resume_identity"].values()
        ),
        "setup_family_coverage_complete": setups["families_seen"] == 16,
        "storage_root_resolved": bool(storage["recommended_rollout_root"])
        and storage["storage_evaluation"]["recommended"],
        "no_neural_training": (
            storage["resolved_root_entry_count"] == 0
            and not storage["rollout_corpus_created"]
            and setups["models_constructed"] == 0
            and setups["checkpoints_loaded"] == 0
            and setups["engine_plies_simulated"] == 0
            and sorted(
                str(path.relative_to(REPOSITORY_ROOT))
                for path in (REPOSITORY_ROOT / "checkpoints" / "phase9").rglob("*.pt")
            )
            == ["checkpoints/phase9/agent01/anchor_eval.pt"]
        ),
        "full_suite_green": (
            tests_after["returncode"] == 0 and not tests_after["failed"]
            if tests_after is not None
            else TESTS_BEFORE["failed"] == 0
        ),
    }

    problems = [name for name, value in completion_gates.items() if not value]
    status = "PASS" if not problems else "FAIL"

    acceptance_payload = {
        "phase": PHASE,
        "agent": AGENT,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **environment_report(),
        "tests_before": TESTS_BEFORE,
        "tests_after": tests_after,
        "commands": [
            f"{sys.executable} scripts/run_phase9_agent02.py"
            + (" --run-pytest" if run_pytest else "")
        ],
        "artifact": "agent_02_acceptance",
        "population_version": population_payload["population_version"],
        "schedule_version": population_payload["schedule_version"],
        "population_digest": schedule["population_digest"],
        "run_schedule_digests": schedule["run_schedule_digests"],
        "agent1_verification": verify["agent1_verification"],
        "corpus_verification": verify["corpus_verification"],
        "schedule_totals": {
            "canonical_iterations": canonical["iterations"],
            "canonical_games_per_iteration": canonical["games_per_iteration"],
            "canonical_total_games": canonical["total_scheduled_games"],
            "pilot_runs": len(pilots),
            "pilot_iterations_each": pilots[0]["iterations"],
            "pilot_games_per_iteration": pilots[0]["games_per_iteration"],
            "pilot_total_games_each": pilots[0]["total_scheduled_games"],
            "all_namespaces_total_games": schedule["cross_namespace"]["total_game_ids"],
            "distinct_game_ids": schedule["cross_namespace"]["distinct_game_ids"],
        },
        "seed_collisions": {
            "within_stream_collisions": schedule["seed_collisions"][
                "within_stream_collisions"
            ],
            "same_game_setup_side_collisions": schedule["seed_collisions"][
                "same_game_setup_side_collisions"
            ],
            "per_stream": {
                name: report["collisions"]
                for name, report in schedule["seed_collisions"]["per_stream"].items()
            },
            "seeds_derived": sum(
                report["values_derived"]
                for report in schedule["seed_collisions"]["per_stream"].values()
            ),
        },
        "setup_assignment": {
            key: setups[key]
            for key in (
                "coverage",
                "games_resolved",
                "setup_sides_resolved",
                "split",
                "purpose",
                "profile",
                "split_violations",
                "families_seen",
                "family_min_count",
                "family_max_count",
                "expected_per_family",
                "held_out_setups_compared",
                "held_out_setup_leaks",
                "games_with_identical_sides",
                "models_constructed",
                "checkpoints_loaded",
                "engine_plies_simulated",
            )
        },
        "storage_handoff": {
            "logical_schedule_identity": {
                "population_version": population_payload["population_version"],
                "schedule_version": population_payload["schedule_version"],
                "rollout_version": schedule["population_document"]["rollout_version"],
                "population_digest": schedule["population_digest"],
                "run_schedule_digests": schedule["run_schedule_digests"],
            },
            "resolved_rollout_root": storage["recommended_rollout_root"],
            "external_volume": storage["chosen_volume"],
            "free_space_measurement": {
                "measured_at": storage["storage_evaluation"]["volume"]["mount_point"],
                "free_bytes": storage["storage_evaluation"]["volume"]["free_bytes"],
                "free_gib": storage["storage_evaluation"]["volume"]["free_gib"],
                "total_bytes": storage["storage_evaluation"]["volume"]["total_bytes"],
                "projected_requirement_bytes": storage["storage_evaluation"][
                    "projection"
                ]["projected_bytes"],
                "observed_headroom_factor": storage["storage_evaluation"][
                    "observed_headroom_factor"
                ],
            },
            "storage_resolution_source": storage["resolution_source"],
            "resolver": "stratego.training.phase9_storage.default_rollout_root()",
            "pointer_file": storage["pointer_file"],
            "pointer_file_written": storage["pointer_file_written"],
            "throughput_probe": storage["throughput_probe"],
            "path_is_diagnostic_not_identity": pstore.STORAGE_IDENTITY_RULE,
            "rollout_corpus_created": storage["rollout_corpus_created"],
        },
        "handoff_to_agent_3": {
            "schedule_enumeration_api": [
                "phase9_schedule.iteration_schedule(namespace, iteration, history=...)",
                "phase9_schedule.iter_iteration_schedule(...)",
                "phase9_schedule.iteration_game_ids(namespace, iteration)",
                "phase9_schedule.iter_run_schedule(namespace)",
            ],
            "game_id_parser_rebuilder": (
                "phase9_schedule.rebuild_scheduled_game(game_id) — pure, "
                "reproduces the full record from the identifier alone"
            ),
            "active_history_manifest_interface": (
                "phase9_schedule.ActiveHistoryManifest.frozen_for(namespace, "
                "iteration, checkpoint_digests) — explicit immutable input, "
                "validated against the frozen window"
            ),
            "learner_control_field": "ScheduledGame.learner_control (red|blue|both)",
            "policy_identities": (
                "ScheduledGame.red_policy_identity / blue_policy_identity; "
                "behavior_snapshot_identity names the iteration's frozen "
                "learner, historical_snapshot_identity the archive member"
            ),
            "setup_identity_derivation": (
                "training_setup_source('neutral_v1').assign(root_seed="
                "setup_root_seed(game_id), environment_id=0, generation=0)"
            ),
            "resume_subtraction": (
                "phase9_schedule.pending_game_ids(namespace, iteration, "
                "committed) — scheduled minus committed; a foreign committed "
                "id raises rather than being ignored"
            ),
            "per_side_checkpoint_identity_warning": (
                "GameRecord.collection_checkpoint_id is the *current* behavior "
                "snapshot's SHA-256; a historical opponent's decisions were "
                "produced by a different checkpoint, whose identity travels in "
                "the phase9_rollout_store_v1 sidecar "
                "(opponent_checkpoint_sha256 / ScheduledGame."
                "opponent_checkpoint_digest). Agent 3 must verify each neural "
                "decision against the acting side's own checkpoint identity, "
                "never against the game-level behavior digest"
            ),
            "no_new_learning_design_decisions": True,
        },
        "carry_forward_notes": [
            (
                "Agent 1's active_historical_window() applies the frozen "
                "5-iteration archive cadence in every run namespace, so each "
                "8-iteration pilot schedules H005 opponents from iteration 6 "
                f"onward (~{schedule['namespaces']['pilot_p9a']['historical_totals'].get('H005', 0)} "
                "games per pilot). The common contract states the cadence "
                "under the canonical run only; the scheduler follows Agent 1's "
                "namespace-independent function rather than making a new "
                "decision. Agents 5/6 must therefore archive an immutable "
                "pilot snapshot after pilot iteration 5, or those scheduled "
                "games have no opponent weights to load"
            ),
            (
                "H000 receives the largest share of historical games "
                f"({schedule['namespaces']['canonical']['historical_totals']['H000']} of "
                f"{sum(schedule['namespaces']['canonical']['historical_totals'].values())} "
                "canonical historical games) because the Phase 8 anchor never "
                "leaves the active window while every later snapshot enters "
                "and exits it. This is the frozen window rule, not a "
                "preference, and must not be 'corrected' downstream"
            ),
            (
                "the recommended rollout root sits on a USB external volume; "
                "if it is unmounted at collection time, "
                "phase9_storage.default_rollout_root() still resolves to it "
                "through the pointer file and the write will fail loudly "
                "rather than silently landing on the boot disk. Logical "
                "scheduling is unaffected either way"
            ),
        ],
        "completion_gates": completion_gates,
        "gates_total": len(completion_gates),
        "gates_true": sum(bool(value) for value in completion_gates.values()),
        "problems": problems,
        "deviations": [],
        "durations": {
            "verify": verify["seconds"],
            "storage": storage["seconds"],
            "schedule": schedule["seconds"],
            "setups": setups["seconds"],
            "artifacts": round(time.perf_counter() - started, 3),
        },
    }
    ACCEPTANCE_ARTIFACT.write_text(json.dumps(acceptance_payload, indent=1) + "\n")

    print(f"status: {status}")
    print(
        f"gates: {sum(bool(v) for v in completion_gates.values())} / "
        f"{len(completion_gates)} true"
    )
    print(f"population digest: {schedule['population_digest']}")
    print(f"rollout root:      {storage['recommended_rollout_root']}")
    for path in (
        POPULATION_ARTIFACT,
        AUDIT_ARTIFACT,
        SUMMARY_ARTIFACT,
        ACCEPTANCE_ARTIFACT,
    ):
        print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")
    return acceptance_payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STAGES = ("verify", "storage", "schedule", "setups", "artifacts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, help="run a single stage")
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--probe-mib", type=int, default=32)
    parser.add_argument(
        "--no-write-pointer",
        action="store_true",
        help="resolve and report the rollout root without recording the redirect",
    )
    parser.add_argument(
        "--sampled-setups",
        action="store_true",
        help="audit setups on a few iterations instead of the whole schedule",
    )
    arguments = parser.parse_args()
    write_pointer = not arguments.no_write_pointer
    full_setups = not arguments.sampled_setups

    if arguments.stage == "verify":
        stage_verify()
        return 0
    if arguments.stage == "storage":
        stage_storage(probe_mib=arguments.probe_mib, write_pointer=write_pointer)
        return 0
    if arguments.stage == "schedule":
        stage_schedule()
        return 0
    if arguments.stage == "setups":
        stage_setups(workers=arguments.workers, full=full_setups)
        return 0
    if arguments.stage == "artifacts":
        payload = stage_artifacts(run_pytest=arguments.run_pytest)
        return 0 if payload["status"] == "PASS" else 1

    stage_verify()
    stage_storage(probe_mib=arguments.probe_mib, write_pointer=write_pointer)
    stage_schedule()
    stage_setups(workers=arguments.workers, full=full_setups)
    payload = stage_artifacts(run_pytest=arguments.run_pytest)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
