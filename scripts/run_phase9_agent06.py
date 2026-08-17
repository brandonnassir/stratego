#!/usr/bin/env python3
"""Phase 9 Agent 6 acceptance harness: bounded RL pilot selection.

Stages:

```text
verify      Agents 1-5 acceptance, corpus resolver, mounted storage, Phase 8
            identity, anchor export, pinned schedule digests, H005 re-enumeration
pilots      run the six frozen candidates P9-A..P9-F, each in its own process
selection   the iteration-8 frozen validation score, guards, tie-break, winner
config      freeze phase9_train_config_v1 (document + runtime identity)
projection  measured canonical 60x2,048 wall-clock projection vs the 12 h ceiling
artifacts   completion gates and the three Agent 6 artifacts
```

Worker purity
-------------
`run_neural_schedule` spawns pure-engine game workers via `spawn`, which
re-imports `__main__`, and the trainer's loader pool spawns the same way.
Torch-loading modules (`stratego.training.*`, `stratego.model.*`,
`stratego.evaluation.phase9_banks`) therefore never appear at this script's
module scope — the accepted Agent 1/7 discipline.

What this harness does and does not decide
------------------------------------------
It runs exactly the six frozen candidates and selects exactly one by the
frozen iteration-8 validation score. It retunes nothing: the execution
topology is Agent 5's validated `workers=6 / prefetch=2 / record cache=48`,
every learning constant is read from the frozen contract, and iteration-4
playing-strength guards are intermediate diagnostics — the Random/Basic
regression guards bind at the frozen iteration-8 validation pass only.
Correctness/safety vetoes (illegal actions, non-finite state, identity
failures, observer leakage, KL/clip hard limits, resume failure) apply
immediately at whatever point they fire. No trained pilot checkpoint is
handed to Agent 7.

Usage::

    python scripts/run_phase9_agent06.py --stage verify
    python scripts/run_phase9_agent06.py --stage pilots
    python scripts/run_phase9_agent06.py --stage selection
    python scripts/run_phase9_agent06.py --stage config
    python scripts/run_phase9_agent06.py --stage projection
    python scripts/run_phase9_agent06.py --stage artifacts
    python scripts/run_phase9_agent06.py --record-final-suite
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Torch-free imports only above function scope; see the module docstring.
from stratego.evaluation.match_runner import (  # noqa: E402
    ERROR_ILLEGAL_ACTION,
    ON_POLICY_ERROR_QUARANTINE,
    play_match,
    results_digest,
)
from stratego.evaluation.match_spec import (  # noqa: E402
    PAIRING_COLOR_SWAP_SAME_BOARD,
    build_paired_schedule,
    schedule_digest,
    schedule_matches,
)
from stratego.evaluation.neural_worker import (  # noqa: E402
    BATCH_POLICY_SINGLE,
    DECISION_MODE_GREEDY,
    LocalInferenceChannel,
    NEURAL_WORKER_VERSION,
    RemoteNeuralPolicy,
    neural_policy_ref,
    run_neural_schedule,
)
from stratego.evaluation.registry import policy_ref  # noqa: E402
from stratego.evaluation.setup_bank import SetupBank, bank_digest  # noqa: E402
from stratego.evaluation.statistics import matchup_seed, summarize_matchup  # noqa: E402

AGENT = 6
PHASE = 9
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_9_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_9_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase9" / "agent06"

SELECTION_ARTIFACT = DATA_DIRECTORY / "agent_06_pilot_selection.json"
RUNS_ARTIFACT = DATA_DIRECTORY / "agent_06_pilot_runs.csv"
CONFIG_ARTIFACT = DATA_DIRECTORY / "agent_06_frozen_train_config.json"

PHASE8_CHECKPOINT = REPOSITORY_ROOT / "checkpoints" / "phase8" / "warmstart_c1_v1.pt"
ANCHOR_EXPORT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "agent01" / "anchor_eval.pt"

#: Production archive root: the frozen `checkpoints/phase9/archive` namespace.
PRODUCTION_ARCHIVE_ROOT = REPOSITORY_ROOT / "checkpoints" / "phase9" / "archive"

#: Accepted upstream digests, pinned by the reviewing chat's acceptances.
ACCEPTED_CONTRACT_DIGEST = (
    "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"
)
ACCEPTED_EXAMPLE_DIGEST = (
    "a6b17a94449ab764d4b5dd054d677096adfa70c52631865499a60a7a3f44af61"
)

#: The anchor's evaluation identity (the accepted Agent 1/7 shape).
ANCHOR_CANDIDATE_ID = "c1_warmstart"
GATE_DTYPE = "float32"

RULE_OPPONENT_IDS = (
    "random_legal",
    "basic_heuristic",
    "tactical_rule_based",
    "strategic_rule_based",
)

#: The frozen validated execution topology (Agent 5). Not retuned per candidate.
VALIDATED_TOPOLOGY = {"workers": 6, "prefetch": 2, "record_cache_size": 48}

#: The full suite as measured immediately before any Phase 9 Agent 6 change.
TESTS_BEFORE = {
    "command": ".venv/bin/python -m pytest tests -q -p no:randomly",
    "summary": "4431 passed, 3 skipped in 326.50s (0:05:26)",
    "passed": 4431,
    "failed": 0,
    "skipped": 3,
    "seconds": 326.50,
    "measured_at_commit": "8c59308",
}

CANDIDATE_ORDER = ("P9-A", "P9-B", "P9-C", "P9-D", "P9-E", "P9-F")

#: Guard thresholds restated nowhere: read from the frozen contract at runtime.
CSV_COLUMNS = (
    "candidate_id",
    "namespace",
    "learning_rate",
    "initial_kl_beta",
    "iteration",
    "checkpoint_identity",
    "checkpoint_sha256",
    "ewr_random",
    "ewr_basic",
    "ewr_tactical",
    "ewr_strategic",
    "ewr_anchor",
    "selection_score",
    "random_guard_pass",
    "basic_guard_pass",
    "guard_binding",
    "illegal_actions",
    "inference_failures",
    "policy_errors",
    "run_mean_behavior_kl",
    "run_examples_per_second",
    "pass_seconds",
    "status",
    "veto_reason",
)


class Agent6Error(RuntimeError):
    """A precondition or frozen identity failed. Always raised, never patched."""


class _VetoSignal(Exception):
    """Internal control flow: a frozen veto fired outside the iteration loop."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[agent06] {message}", flush=True)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def git_output(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is diagnostic
        return "unknown"


def environment_record() -> dict:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "source_revision": git_output("rev-parse", "--short", "HEAD"),
        "working_tree_state": "dirty" if git_output("status", "--porcelain") else "clean",
    }


def peak_rss_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def stage_path(name: str) -> Path:
    return WORK_DIRECTORY / f"stage_{name}.json"


def write_stage(name: str, payload: dict) -> None:
    write_json(stage_path(name), payload)


def read_stage(name: str) -> dict:
    path = stage_path(name)
    if not path.exists():
        raise Agent6Error(f"stage {name!r} has not been run yet ({path})")
    return read_json(path)


def _training():
    """Torch-adjacent modules, imported on first use only (worker purity)."""
    from stratego.training import phase9_behavior as pb
    from stratego.training import phase9_checkpoint as pck
    from stratego.training import phase9_collector as pc
    from stratego.training import phase9_contract as contract
    from stratego.training import phase9_schedule as schedule
    from stratego.training import phase9_seed as seed
    from stratego.training import phase9_storage as storage
    from stratego.training import phase9_trainer as pt
    from stratego.training import synthetic_corpus as corpus
    from stratego.training.warmstart_checkpoint import (
        CorpusIdentity,
        verify_corpus_identity,
    )

    return {
        "pb": pb,
        "pck": pck,
        "pc": pc,
        "contract": contract,
        "schedule": schedule,
        "seed": seed,
        "storage": storage,
        "pt": pt,
        "corpus": corpus,
        "CorpusIdentity": CorpusIdentity,
        "verify_corpus_identity": verify_corpus_identity,
    }


def accepted_corpus_identity(modules):
    contract = modules["contract"]
    return modules["CorpusIdentity"](
        corpus_version=contract.EXPECTED_CORPUS_VERSION,
        content_digest=contract.EXPECTED_CORPUS_CONTENT_DIGEST,
        metadata_digest=contract.EXPECTED_CORPUS_METADATA_DIGEST,
        commit_index_digest=contract.EXPECTED_CORPUS_COMMIT_INDEX_DIGEST,
    )


def candidate_directory(namespace: str) -> Path:
    return WORK_DIRECTORY / namespace


def journal_path(namespace: str) -> Path:
    return candidate_directory(namespace) / "journal.json"


def pilot_stage_name(candidate_id: str) -> str:
    return f"pilot_{candidate_id.replace('-', '_')}"


# ---------------------------------------------------------------------------
# Stage: verify
# ---------------------------------------------------------------------------


def verify_storage_mounted() -> dict:
    """The supplementary review requirement: the rollout root must resolve to
    the actually mounted external filesystem, not an ordinary directory on
    the boot volume. Checked before every candidate and after every restart.
    """
    modules = _training()
    storage = modules["storage"]
    root = storage.default_rollout_root()
    description = storage.describe_rollout_root()
    volume = storage.volume_diagnostics(root)
    write = storage.check_writable(root)
    mount = volume["mount_point"]
    problems: list[str] = []
    is_mount = os.path.ismount(mount)
    on_external = mount.startswith("/Volumes/") and mount not in ("/", "/Volumes")
    if not is_mount:
        problems.append(f"{mount} is not a mount point")
    if not on_external:
        problems.append(
            f"rollout root {root} resolves under mount {mount}, which is not an "
            "external /Volumes/* filesystem — an unplugged drive leaves an "
            "ordinary boot-volume directory here"
        )
    if volume["read_only"]:
        problems.append(f"{mount} is mounted read-only")
    if not write["writable"]:
        problems.extend(write["problems"])
    if volume["free_gib"] < 50:
        problems.append(f"only {volume['free_gib']} GiB free on {mount}")
    return {
        "resolver": "stratego.training.phase9_storage.default_rollout_root()",
        "resolved_root": str(root),
        "resolution_source": description["source"],
        "pointer_value": description["pointer_value"],
        "mount_point": mount,
        "is_mount_point": is_mount,
        "on_external_volume": on_external,
        "free_gib": volume["free_gib"],
        "read_only": volume["read_only"],
        "write_probe_ok": write["writable"],
        "identity_rule": storage.STORAGE_IDENTITY_RULE,
        "problems": problems,
    }


def stage_verify(args) -> dict:
    modules = _training()
    contract = modules["contract"]
    schedule = modules["schedule"]
    pb = modules["pb"]

    problems: list[str] = []
    acceptances = {}
    for agent in (1, 2, 3, 4, 5):
        path = DATA_DIRECTORY / f"agent_{agent:02d}_acceptance.json"
        if not path.exists():
            problems.append(f"agent {agent} acceptance artifact is missing")
            continue
        payload = read_json(path)
        acceptances[str(agent)] = {"status": payload.get("status")}
        if payload.get("status") != "PASS":
            problems.append(f"agent {agent} status is {payload.get('status')!r}, not PASS")
    agent5 = read_json(DATA_DIRECTORY / "agent_05_acceptance.json")
    if not agent5.get("all_passed"):
        problems.append("agent 5 acceptance gates are not all true")

    observed_contract = contract.contract_digest()
    from stratego.training.phase9_targets import example_contract_digest

    observed_example = example_contract_digest()
    if observed_contract != ACCEPTED_CONTRACT_DIGEST:
        problems.append(f"contract digest {observed_contract} != accepted")
    if observed_example != ACCEPTED_EXAMPLE_DIGEST:
        problems.append(f"example contract digest {observed_example} != accepted")

    checkpoint_sha = pb.file_sha256(PHASE8_CHECKPOINT) if PHASE8_CHECKPOINT.exists() else "<missing>"
    if checkpoint_sha != contract.EXPECTED_PHASE8_CHECKPOINT_SHA256:
        problems.append(f"Phase 8 checkpoint SHA {checkpoint_sha} != accepted")

    # Corpus: resolver first, digests second, path never.
    resolved = modules["corpus"].default_corpus_root()
    corpus_observed = None
    try:
        corpus_observed = modules["verify_corpus_identity"](
            resolved,
            accepted_corpus_identity(modules),
            check_payload_bytes=not args.skip_payload_bytes,
        )
    except Exception as error:  # noqa: BLE001 - a corpus mismatch is BLOCKED
        problems.append(f"corpus verification failed: {type(error).__name__}: {error}")

    storage_check = verify_storage_mounted()
    problems.extend(storage_check["problems"])

    # The anchor evaluation export Agent 1 verified bitwise.
    agent1 = read_json(DATA_DIRECTORY / "agent_01_acceptance.json")
    anchor_export_expected = agent1["anchor_export"]["export_sha256"]
    anchor_export_sha = (
        pb.file_sha256(ANCHOR_EXPORT_PATH) if ANCHOR_EXPORT_PATH.exists() else "<missing>"
    )
    if anchor_export_sha != anchor_export_expected:
        problems.append(
            f"anchor evaluation export SHA {anchor_export_sha} != Agent 1's "
            f"{anchor_export_expected}"
        )
    validation_bank_digest = agent1["bank_digests"]["phase9_validation_bank_v1"]

    # Pinned Agent 2 run-schedule digests, recomputed from the live schedule.
    agent2 = read_json(DATA_DIRECTORY / "agent_02_acceptance.json")
    pinned = agent2["run_schedule_digests"]
    schedule_checks = {}
    for candidate in contract.PILOT_CANDIDATES:
        namespace = candidate["namespace"]
        recomputed = schedule.run_schedule_digest(namespace)
        schedule_checks[namespace] = {
            "pinned": pinned[namespace],
            "recomputed": recomputed,
            "matches": recomputed == pinned[namespace],
        }
        if recomputed != pinned[namespace]:
            problems.append(f"{namespace} run schedule digest drifted")

    # The supplementary re-enumeration: exact H005 assignment counts for every
    # candidate and every iteration 6-8, from the frozen schedule itself.
    h005_enumeration = {}
    for candidate in contract.PILOT_CANDIDATES:
        namespace = candidate["namespace"]
        per_iteration = {}
        for iteration in (6, 7, 8):
            window = contract.active_historical_window(iteration)
            counts: dict[str, int] = {}
            for game in contract.iter_scheduled_games(namespace, iteration):
                if game["bucket"] == "historical":
                    key = game["opponent"]["identity"]
                    counts[key] = counts.get(key, 0) + 1
            per_iteration[str(iteration)] = {
                "active_window": list(window),
                "opponent_counts": counts,
            }
            if sum(counts.values()) != contract.PILOT_BUCKET_COUNTS["historical"]:
                problems.append(f"{namespace} iteration {iteration} historical count drifted")
        h005_enumeration[namespace] = {
            "iterations": per_iteration,
            "total_h005_games": sum(
                entry["opponent_counts"].get("H005", 0)
                for entry in per_iteration.values()
            ),
        }

    # Agent 5's soak artifacts must stay outside the production pilot slots.
    separation = {
        "agent05_soak_archive_h005": str(
            REPOSITORY_ROOT / "checkpoints" / "phase9" / "agent05" / "archive" / "pilot_p9c" / "H005.pt"
        ),
        "agent05_soak_archive_h005_exists": (
            REPOSITORY_ROOT / "checkpoints" / "phase9" / "agent05" / "archive" / "pilot_p9c" / "H005.pt"
        ).exists(),
        "production_slots": {},
    }
    rollout_root = Path(storage_check["resolved_root"])
    for candidate in contract.PILOT_CANDIDATES:
        namespace = candidate["namespace"]
        separation["production_slots"][namespace] = {
            "archive_h005_exists": (PRODUCTION_ARCHIVE_ROOT / namespace / "H005.pt").exists(),
            "rollout_namespace_exists": (rollout_root / namespace).exists(),
            "work_directory_exists": candidate_directory(namespace).exists(),
        }

    payload = {
        "stage": "verify",
        **environment_record(),
        "acceptances": acceptances,
        "agent5_gates": {"passed": agent5["passed"], "total": agent5["total"]},
        "contract_digest": observed_contract,
        "example_contract_digest": observed_example,
        "phase8_checkpoint_sha256": checkpoint_sha,
        "corpus": {
            "resolver": "stratego.training.synthetic_corpus.default_corpus_root()",
            "resolved_root": str(resolved),
            "observed_identity": corpus_observed.to_dict() if corpus_observed else None,
            "identity_matches": (
                corpus_observed == accepted_corpus_identity(modules)
                if corpus_observed
                else False
            ),
            "payload_bytes_checked": not args.skip_payload_bytes,
        },
        "storage": storage_check,
        "anchor_export": {
            "path": str(ANCHOR_EXPORT_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": anchor_export_sha,
            "matches_agent1": anchor_export_sha == anchor_export_expected,
        },
        "validation_bank_digest_expected": validation_bank_digest,
        "schedule_digests": schedule_checks,
        "h005_reenumeration": h005_enumeration,
        "agent5_separation": separation,
        "tests_before": TESTS_BEFORE,
        "topology": dict(VALIDATED_TOPOLOGY),
        "problems": problems,
    }
    write_stage("verify", payload)
    if problems:
        log(f"BLOCKED: {problems}")
    else:
        log("verify: prerequisites, corpus, mounted storage, schedules all confirmed")
    return payload


# ---------------------------------------------------------------------------
# Validation passes (worker side)
# ---------------------------------------------------------------------------


def _chunks(matches, size):
    for start in range(0, len(matches), size):
        yield start // size, matches[start : start + size]


def bank_cache_path() -> Path:
    return WORK_DIRECTORY / "validation_bank.json"


def load_validation_bank(expected_digest: str) -> SetupBank:
    """The frozen validation bank, rebuilt once and cached as plain JSON.

    Rebuild-and-check rather than trust: the cache is only ever accepted when
    its digest equals the accepted Agent 1 digest, and a missing cache is
    rebuilt from frozen constants through `build_phase9_bank`.
    """
    cache = bank_cache_path()
    if cache.exists():
        bank = SetupBank.from_dict(read_json(cache))
        if bank_digest(bank) == expected_digest:
            return bank
    from stratego.evaluation.phase9_banks import build_phase9_bank

    bank, _manifest = build_phase9_bank("validation")
    observed = bank_digest(bank)
    if observed != expected_digest:
        raise Agent6Error(
            f"rebuilt validation bank digest {observed} != accepted {expected_digest}"
        )
    write_json(cache, bank.to_dict())
    return bank


def run_chunked_schedule(matches, bank, owner, *, reference, label, directory, workers, chunk_units):
    """Resumable chunked `run_neural_schedule` execution (accepted Agent 1 shape)."""
    directory.mkdir(parents=True, exist_ok=True)
    all_results = []
    reports = []
    for index, chunk in _chunks(matches, chunk_units * 2):
        digest = schedule_digest(chunk)[:16]
        path = directory / f"chunk_{index:04d}_{digest}.pkl"
        if path.exists():
            with open(path, "rb") as stream:
                stored = pickle.load(stream)
            all_results.extend(stored["results"])
            reports.append(stored["report"] | {"reused": True})
            continue
        run = run_neural_schedule(
            chunk,
            bank,
            owner,
            policy_ref=reference,
            worker_count=workers,
            record_actions=True,
            on_policy_error=ON_POLICY_ERROR_QUARANTINE,
        )
        report = {
            "chunk": index,
            "matches": run.matches_run,
            "decisions": run.decisions,
            "wall_clock_seconds": round(run.wall_clock_seconds, 3),
            "policy_errors": run.policy_errors,
            "illegal_policy_actions": run.illegal_policy_actions,
            "workers_importing_torch": run.workers_importing_torch,
            "worker_checkpoint_loads": run.worker_checkpoint_loads,
            "inference_failures": int(run.inference.get("failures_returned", 0)),
            "results_digest": run.results_digest,
            "reused": False,
        }
        with open(path, "wb") as stream:
            pickle.dump({"results": run.results, "report": report}, stream)
        all_results.extend(run.results)
        reports.append(report)
        log(f"    {label} chunk {index}: {run.matches_run} games in {run.wall_clock_seconds:.1f}s")
    return all_results, reports


def anchor_chunk_path(directory: Path, index: int, chunk) -> Path:
    return directory / f"anchor_{index:04d}_{schedule_digest(chunk)[:16]}.pkl"


def candidate_eval_ref(namespace: str, iteration: int):
    return neural_policy_ref(f"{namespace}_it{iteration}", dtype_name=GATE_DTYPE)


def anchor_matchup_matches(namespace: str, iteration: int, pairs: int):
    units = build_paired_schedule(
        candidate_eval_ref(namespace, iteration),
        neural_policy_ref(ANCHOR_CANDIDATE_ID, dtype_name=GATE_DTYPE),
        range(pairs),
        setup_bank_version="phase9_validation_bank_v1",
    )
    return schedule_matches(units)


def run_anchor_worker(args) -> None:
    """One process's slice of the candidate-vs-anchor matchup.

    Neural-vs-neural is not expressible through `run_neural_schedule` (one
    owner per schedule), so the accepted `play_match` is driven directly with
    two in-process owners — the accepted Agent 7 shape. The fan-out across
    processes changes only which process plays a chunk, never any identity:
    every match spec is a pure function of the frozen schedule.
    """
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from stratego.evaluation.neural_worker import InferenceOwner
    from stratego.training import phase9_contract as contract

    namespace = args.candidate_namespace
    iteration = args.validation_iteration
    directory = candidate_directory(namespace) / "games" / f"it{iteration}" / "anchor"
    directory.mkdir(parents=True, exist_ok=True)
    matches = anchor_matchup_matches(namespace, iteration, contract.VALIDATION_BANK_CASES)
    chunk_size = max(1, (len(matches) + args.anchor_workers - 1) // args.anchor_workers)
    chunks = list(_chunks(matches, chunk_size))
    index, chunk = chunks[args.anchor_chunk_index]
    path = anchor_chunk_path(directory, index, chunk)
    if path.exists():
        return
    bank = load_validation_bank(args.expected_bank_digest)

    candidate_ref = candidate_eval_ref(namespace, iteration)
    anchor_ref = neural_policy_ref(ANCHOR_CANDIDATE_ID, dtype_name=GATE_DTYPE)
    export_path = candidate_directory(namespace) / f"eval_it{iteration}.pt"
    owners = {
        candidate_ref.token: InferenceOwner(
            export_path,
            decision_mode=DECISION_MODE_GREEDY,
            device=args.device,
            dtype=GATE_DTYPE,
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name=f"agent6_{namespace}_it{iteration}",
        ),
        anchor_ref.token: InferenceOwner(
            ANCHOR_EXPORT_PATH,
            decision_mode=DECISION_MODE_GREEDY,
            device=args.device,
            dtype=GATE_DTYPE,
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name="agent6_anchor",
        ),
    }
    policies = {
        token: RemoteNeuralPolicy(
            ref, LocalInferenceChannel(owners[token]), decision_mode=DECISION_MODE_GREEDY
        )
        for token, ref in (
            (candidate_ref.token, candidate_ref),
            (anchor_ref.token, anchor_ref),
        )
    }
    started = time.perf_counter()
    try:
        results = [
            play_match(
                spec,
                bank=bank,
                policies=policies,
                record_actions=True,
                on_policy_error=ON_POLICY_ERROR_QUARANTINE,
            )
            for spec in chunk
        ]
        owner_stats = {
            name: owner.stats() | {"identity": owner.identity()}
            for name, owner in owners.items()
        }
    finally:
        for owner in owners.values():
            owner.close()
    elapsed = time.perf_counter() - started
    report = {
        "chunk": index,
        "matches": len(results),
        "wall_clock_seconds": round(elapsed, 3),
        "policy_errors": sum(1 for row in results if row.errored),
        "illegal_policy_actions": sum(
            1 for row in results if row.policy_error_category == ERROR_ILLEGAL_ACTION
        ),
        "inference_failures": sum(
            int(stats.get("failures_returned", 0)) for stats in owner_stats.values()
        ),
        "reused": False,
    }
    with open(path, "wb") as stream:
        pickle.dump({"results": tuple(results), "report": report}, stream)
    log(f"    anchor chunk {index}: {len(results)} games in {elapsed:.1f}s")


def run_anchor_matchup(namespace: str, iteration: int, args, expected_bank_digest: str):
    """Fan the candidate-vs-anchor games across worker processes and gather."""
    from stratego.training import phase9_contract as contract

    directory = candidate_directory(namespace) / "games" / f"it{iteration}" / "anchor"
    directory.mkdir(parents=True, exist_ok=True)
    matches = anchor_matchup_matches(namespace, iteration, contract.VALIDATION_BANK_CASES)
    chunk_size = max(1, (len(matches) + args.anchor_workers - 1) // args.anchor_workers)
    chunks = list(_chunks(matches, chunk_size))
    pending = [
        index
        for index, chunk in chunks
        if not anchor_chunk_path(directory, index, chunk).exists()
    ]
    if pending:
        processes = []
        for index in pending:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--anchor-worker",
                "--candidate-namespace",
                namespace,
                "--validation-iteration",
                str(iteration),
                "--anchor-chunk-index",
                str(index),
                "--anchor-workers",
                str(args.anchor_workers),
                "--device",
                args.device,
                "--expected-bank-digest",
                expected_bank_digest,
            ]
            processes.append((index, subprocess.Popen(command, cwd=REPOSITORY_ROOT)))
        failures = []
        for index, process in processes:
            if process.wait() != 0:
                failures.append(index)
        if failures:
            raise Agent6Error(f"anchor worker chunk(s) {failures} failed")
    results = []
    reports = []
    for index, chunk in chunks:
        path = anchor_chunk_path(directory, index, chunk)
        with open(path, "rb") as stream:
            stored = pickle.load(stream)
        results.extend(stored["results"])
        reports.append(stored["report"])
    return results, reports, schedule_digest(matches)


def summarize_results(results, base_seed: int):
    ordered = tuple(sorted(results, key=lambda row: row.match_id))
    matchup = ordered[0].matchup
    summary = summarize_matchup(
        results,
        seed=matchup_seed(base_seed, matchup),
        allow_policy_errors=True,
        include_setup_table=False,
    ).to_dict()
    summary["results_digest"] = results_digest(ordered)
    summary["matchup"] = matchup
    return summary


def run_validation_pass(
    candidate: dict,
    iteration: int,
    weights_path: Path,
    weights_sha256: str,
    args,
    *,
    include_stress: bool,
) -> dict:
    """One full frozen validation pass of one pilot checkpoint.

    Greedy, single-request, float32, the frozen validation bank, the five core
    opponents; stress is the report-only 32-pair prefix schedule and is run
    only when asked (the iteration-8 checkpoint). Never updates weights.
    """
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from stratego.evaluation.neural_worker import InferenceOwner
    from stratego.training import phase9_contract as contract
    from stratego.training import phase9_seed as seed
    from stratego.training.phase9_checkpoint import model_from_payload, read_phase9_payload
    from stratego.training.phase9_behavior import file_sha256

    access = contract.check_validation_bank_access("pilot_selection", phase9_agent=AGENT)
    namespace = candidate["namespace"]
    work = candidate_directory(namespace)
    started = time.perf_counter()
    timings: dict[str, float] = {}

    # Export the pilot checkpoint to the frozen evaluation format, bitwise.
    export_path = work / f"eval_it{iteration}.pt"
    export_started = time.perf_counter()
    import torch
    from stratego.model.checkpoint import load_checkpoint, save_checkpoint

    payload = read_phase9_payload(weights_path)
    model = model_from_payload(payload)
    save_checkpoint(model, export_path)
    reloaded, _metadata = load_checkpoint(
        export_path,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
    )
    source_state = model.state_dict()
    reloaded_state = reloaded.state_dict()
    bitwise = set(source_state) == set(reloaded_state) and all(
        torch.equal(source_state[name], reloaded_state[name]) for name in source_state
    )
    if not bitwise:
        raise Agent6Error(f"{namespace} it{iteration}: evaluation export changed the weights")
    del model, reloaded, payload
    timings["export_seconds"] = time.perf_counter() - export_started

    bank_started = time.perf_counter()
    expected_bank_digest = read_stage("verify")["validation_bank_digest_expected"]
    bank = load_validation_bank(expected_bank_digest)
    timings["bank_seconds"] = time.perf_counter() - bank_started

    reference = candidate_eval_ref(namespace, iteration)
    matchups: dict[str, dict] = {}
    safety = {"illegal_policy_actions": 0, "policy_errors": 0, "inference_failures": 0,
              "workers_importing_torch": 0, "worker_checkpoint_loads": 0}

    owner = InferenceOwner(
        export_path,
        decision_mode=DECISION_MODE_GREEDY,
        device=args.device,
        dtype=GATE_DTYPE,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
        name=f"agent6_{namespace}_it{iteration}",
    )
    try:
        for opponent_id in RULE_OPPONENT_IDS:
            opponent_started = time.perf_counter()
            units = build_paired_schedule(
                reference,
                policy_ref(opponent_id),
                range(contract.VALIDATION_BANK_CASES),
                setup_bank_version=contract.VALIDATION_BANK_VERSION,
            )
            matches = schedule_matches(units)
            results, reports = run_chunked_schedule(
                matches,
                bank,
                owner,
                reference=reference,
                label=f"{namespace}_it{iteration}_{opponent_id}",
                directory=work / "games" / f"it{iteration}" / opponent_id,
                workers=args.eval_workers,
                chunk_units=args.chunk_units,
            )
            summary = summarize_results(results, seed.VALIDATION_BOOTSTRAP_SEED)
            matchups[opponent_id] = {
                "summary": summary,
                "schedule_digest": schedule_digest(matches),
                "chunks": reports,
                "seconds": time.perf_counter() - opponent_started,
            }
            for report in reports:
                safety["illegal_policy_actions"] += report["illegal_policy_actions"]
                safety["inference_failures"] += report["inference_failures"]
                safety["workers_importing_torch"] = max(
                    safety["workers_importing_torch"], report["workers_importing_torch"]
                )
                safety["worker_checkpoint_loads"] = max(
                    safety["worker_checkpoint_loads"], report["worker_checkpoint_loads"]
                )
            safety["policy_errors"] += summary["policy_errors"]

        stress = None
        if include_stress:
            stress_started = time.perf_counter()
            stress = {}
            for policy_id in contract.STRESS_POLICY_ROSTER:
                units = build_paired_schedule(
                    reference,
                    policy_ref(policy_id),
                    range(contract.VALIDATION_STRESS_PAIRS),
                    setup_bank_version=contract.VALIDATION_BANK_VERSION,
                )
                matches = schedule_matches(units)
                results, reports = run_chunked_schedule(
                    matches,
                    bank,
                    owner,
                    reference=reference,
                    label=f"{namespace}_it{iteration}_{policy_id}",
                    directory=work / "games" / f"it{iteration}" / policy_id,
                    workers=args.eval_workers,
                    chunk_units=args.chunk_units,
                )
                summary = summarize_results(results, seed.VALIDATION_BOOTSTRAP_SEED)
                stress[policy_id] = {
                    "effective_win_rate": summary["effective_win_rate"],
                    "confidence_interval": summary["confidence_interval"],
                    "games": summary["games"],
                    "policy_errors": summary["policy_errors"],
                    "results_digest": summary["results_digest"],
                }
                for report in reports:
                    safety["illegal_policy_actions"] += report["illegal_policy_actions"]
                    safety["inference_failures"] += report["inference_failures"]
                safety["policy_errors"] += summary["policy_errors"]
            timings["stress_seconds"] = time.perf_counter() - stress_started
    finally:
        owner.close()

    anchor_started = time.perf_counter()
    anchor_results, anchor_reports, anchor_schedule_digest = run_anchor_matchup(
        namespace, iteration, args, expected_bank_digest
    )
    anchor_summary = summarize_results(anchor_results, seed.VALIDATION_BOOTSTRAP_SEED)
    matchups["phase8_anchor"] = {
        "summary": anchor_summary,
        "schedule_digest": anchor_schedule_digest,
        "chunks": anchor_reports,
        "seconds": time.perf_counter() - anchor_started,
        "anchor_export_sha256": file_sha256(ANCHOR_EXPORT_PATH),
    }
    for report in anchor_reports:
        safety["illegal_policy_actions"] += report["illegal_policy_actions"]
        safety["inference_failures"] += report["inference_failures"]
    safety["policy_errors"] += anchor_summary["policy_errors"]

    ewrs = {
        "random_legal": matchups["random_legal"]["summary"]["effective_win_rate"],
        "basic_heuristic": matchups["basic_heuristic"]["summary"]["effective_win_rate"],
        "tactical_rule_based": matchups["tactical_rule_based"]["summary"]["effective_win_rate"],
        "strategic_rule_based": matchups["strategic_rule_based"]["summary"]["effective_win_rate"],
        "phase8_anchor": matchups["phase8_anchor"]["summary"]["effective_win_rate"],
    }
    score = contract.validation_score(
        ewrs["strategic_rule_based"], ewrs["tactical_rule_based"], ewrs["phase8_anchor"]
    )
    guards = {
        "random_ewr": ewrs["random_legal"],
        "random_min": contract.VALIDATION_REGRESSION_GUARDS["random_legal_ewr_min"],
        "random_pass": ewrs["random_legal"] >= contract.VALIDATION_REGRESSION_GUARDS["random_legal_ewr_min"],
        "basic_ewr": ewrs["basic_heuristic"],
        "basic_min": contract.VALIDATION_REGRESSION_GUARDS["basic_heuristic_ewr_min"],
        "basic_pass": ewrs["basic_heuristic"] >= contract.VALIDATION_REGRESSION_GUARDS["basic_heuristic_ewr_min"],
        "binding": "final" if iteration == contract.PILOT_ITERATIONS else "intermediate_diagnostic",
    }
    pass_record = {
        "candidate_id": candidate["candidate_id"],
        "namespace": namespace,
        "iteration": iteration,
        "checkpoint_identity": weights_path.name,
        "checkpoint_path": str(weights_path),
        "checkpoint_sha256": weights_sha256,
        "eval_export_sha256": file_sha256(export_path),
        "authorized_access": {
            "resource": access.resource,
            "purpose": access.purpose,
            "phase9_agent": access.phase9_agent,
        },
        "bank_version": contract.VALIDATION_BANK_VERSION,
        "bank_digest": expected_bank_digest,
        "protocol": {
            "decision_mode": DECISION_MODE_GREEDY,
            "batch_policy": BATCH_POLICY_SINGLE,
            "dtype": GATE_DTYPE,
            "pairing_mode": PAIRING_COLOR_SWAP_SAME_BOARD,
            "neural_worker_version": NEURAL_WORKER_VERSION,
            "candidate_ref": reference.to_dict(),
            "bootstrap_base_seed": seed.VALIDATION_BOOTSTRAP_SEED,
        },
        "effective_win_rates": ewrs,
        "confidence_intervals": {
            key: matchups[key]["summary"]["confidence_interval"] for key in matchups
        },
        "selection_score": score,
        "score_weights": dict(contract.VALIDATION_SCORE_WEIGHTS),
        "guards": guards,
        "stress_report_only": stress,
        "matchups": {
            key: {
                "schedule_digest": value["schedule_digest"],
                "results_digest": value["summary"]["results_digest"],
                "games": value["summary"]["games"],
                "wins": value["summary"].get("wins"),
                "draws": value["summary"].get("draws"),
                "losses": value["summary"].get("losses"),
                "policy_errors": value["summary"]["policy_errors"],
                "seconds": value["seconds"],
            }
            for key, value in matchups.items()
        },
        "safety": safety,
        "timings": timings,
        "seconds": time.perf_counter() - started,
        "test_bank_games_played": 0,
    }
    return pass_record


# ---------------------------------------------------------------------------
# Historical-action verification (worker side)
# ---------------------------------------------------------------------------


def _decision_requests(record, metadata, wanted_player, *, pb, limit):
    """Replay one game and rebuild everything a re-check needs for one side."""
    from stratego.engine.legal_moves import legal_action_mask, legal_actions
    from stratego.engine.observation import build_observation
    from stratego.engine.state import create_game
    from stratego.engine.transition import apply_action
    from stratego.model.policy_adapter import prepare_legality
    from stratego.training.warmstart_contract import CORPUS_RULES

    state = create_game(
        record.red_setup, record.blue_setup, rules=CORPUS_RULES, game_id=record.game_id
    )
    built = []
    learner = 0 if metadata["learner_color"] == "red" else 1
    for decision in record.decisions:
        legal = legal_actions(state)
        actor = int(state.acting_player)
        if actor == wanted_player and len(built) < limit:
            built.append(
                pb.ReproductionRequest(
                    game_id=record.game_id,
                    ply=int(decision.ply),
                    acting_player=actor,
                    observation=build_observation(state, actor),
                    legality=prepare_legality(legal, legal_action_mask(state, legal), actor),
                    stored_probabilities=tuple(
                        float(value) for value in decision.old_probabilities
                    ),
                    stored_wdl=tuple(
                        float(value) for value in decision.win_draw_loss_prediction
                    ),
                    stored_action=int(decision.selected_action_id),
                    stored_policy_token=decision.collection_policy_version,
                    stored_checkpoint_sha256=(
                        metadata["behavior_checkpoint_sha256"]
                        if actor == learner
                        else metadata["opponent_checkpoint_sha256"]
                    ),
                )
            )
        apply_action(state, decision.selected_action_id, legal=legal)
    return built


def verify_historical_actions(
    reader, iteration: int, namespace: str, historical, *, h000_sample: int
) -> dict:
    """Verify historical-opponent actions against the acting archive checkpoint.

    Every H005-opponent game's opponent-side decisions are reproduced under the
    exact bound candidate-local archive member (the supplementary requirement);
    a deterministic sample of H000 games is reproduced under the anchor as a
    regression control. The digest guard has already bound the identities; this
    is the numerical closure.
    """
    from stratego.training import phase9_behavior as pb
    from stratego.training.phase9_schedule import historical_policy_token

    h005_token = historical_policy_token(namespace, "H005")
    verified = {"H005": {"games": 0, "decisions": 0, "failed": 0, "max_abs_difference": None},
                "H000": {"games": 0, "decisions": 0, "failed": 0, "max_abs_difference": None}}
    started = time.perf_counter()
    h000_examined = 0
    for game_id in reader.game_ids:
        metadata = reader.metadata[game_id]
        if metadata.get("bucket") != "historical":
            continue
        is_h005 = metadata.get("opponent_identity") == h005_token
        if not is_h005:
            if h000_examined >= h000_sample:
                continue
            h000_examined += 1
        key = "H005" if is_h005 else "H000"
        snapshot = historical["H005" if is_h005 else "H000"]
        record, metadata = reader.read_game(game_id)
        learner = 0 if metadata["learner_color"] == "red" else 1
        requests = _decision_requests(
            record, metadata, 1 - learner, pb=pb, limit=10**6
        )
        reports = pb.reproduce_decisions(snapshot, requests)
        failed = sum(1 for report in reports if not report["verified"])
        differences = [
            report["max_abs_difference"]
            for report in reports
            if report["max_abs_difference"] is not None
        ]
        entry = verified[key]
        entry["games"] += 1
        entry["decisions"] += len(reports)
        entry["failed"] += failed
        if differences:
            worst = max(differences)
            entry["max_abs_difference"] = (
                worst
                if entry["max_abs_difference"] is None
                else max(entry["max_abs_difference"], worst)
            )
    return {
        "iteration": iteration,
        "h005_policy_token": h005_token,
        "h005_checkpoint_sha256": (
            historical["H005"].checkpoint_sha256 if "H005" in historical else None
        ),
        "verified": verified,
        "all_verified": all(entry["failed"] == 0 for entry in verified.values()),
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# The pilot worker
# ---------------------------------------------------------------------------


def load_journal(namespace: str) -> dict:
    path = journal_path(namespace)
    if path.exists():
        return read_json(path)
    return {
        "namespace": namespace,
        "iterations": [],
        "validations": {},
        "snapshots": {},
        "archive": None,
        "historical_verification": [],
        "veto": None,
        "start_state_digest": None,
        "wall_clock": {},
    }


def save_journal(namespace: str, journal: dict) -> None:
    write_json(journal_path(namespace), journal)


def summarize_pilot_iteration(rollout, rows, collected, timers, archived, controller) -> dict:
    import numpy as np

    kls = [row["behavior_kl"] for row in rows]
    epoch_kls = [row["epoch_mean_kl"] for row in rows if "epoch_mean_kl" in row]
    epoch_clips = [row["epoch_clip_fraction"] for row in rows if "epoch_clip_fraction" in row]
    return {
        "namespace": rollout.namespace,
        "iteration": rollout.iteration,
        "sealed_rollout_digest": rollout.sealed_rollout_digest,
        "behavior_snapshot_id": rollout.behavior_snapshot_id,
        "behavior_checkpoint_sha256": rollout.behavior_checkpoint_sha256,
        "games": rollout.games,
        "learner_decisions": rollout.learner_decisions,
        "updates": len(rows),
        "examples": sum(row["examples"] for row in rows),
        "advantage_statistics": rollout.statistics.to_dict(),
        "mean_behavior_kl": float(np.mean(kls)) if kls else 0.0,
        "max_behavior_kl": float(np.max(kls)) if kls else 0.0,
        "epoch_mean_kls": epoch_kls,
        "epoch_clip_fractions": epoch_clips,
        "mean_clip_fraction": float(np.mean([row["clip_fraction"] for row in rows])) if rows else 0.0,
        "mean_policy_entropy": float(np.mean([row["policy_entropy"] for row in rows])) if rows else 0.0,
        "mean_advantage_retention": float(np.mean([row["advantage_retention"] for row in rows])) if rows else 0.0,
        "mean_grad_norm_pre_clip": float(np.mean([row["grad_norm_pre_clip"] for row in rows])) if rows else 0.0,
        "final_parameter_norm": float(rows[-1]["parameter_norm"]) if rows else 0.0,
        "kl_beta_after": float(controller.beta),
        "kl_weighted_sum": float(sum(row["behavior_kl"] * row["examples"] for row in rows)),
        "collection": {
            key: collected.get(key)
            for key in (
                "games_collected",
                "games_already_committed",
                "games_per_second",
                "learner_decisions",
                "neural_decisions",
                "observer_probes",
                "observer_probe_failures",
                "sealed_rollout_digest",
                "inference_device",
                "inference_batch_shape",
            )
            if key in collected
        },
        "timers": dict(timers),
        "archived": archived,
        "rss_mib": peak_rss_mib(),
    }


def classify_trainer_veto(error, trainer) -> "str | None":
    """Map a raised trainer/collector error onto the frozen veto it embodies."""
    counters = trainer.counters if trainer is not None else {}
    if counters.get("kl_hard_limit_breaches"):
        return "mean_iteration_or_epoch_kl_max"
    if counters.get("clip_fraction_hard_limit_breaches"):
        return "iteration_ppo_clip_fraction_max"
    if counters.get("non_finite_losses"):
        return "non_finite_loss_max"
    if counters.get("non_finite_gradients"):
        return "non_finite_gradient_max"
    if counters.get("non_finite_parameters"):
        return "non_finite_parameter_max"
    if counters.get("behavior_identity_mismatches"):
        return "behavior_identity_mismatch_max"
    if counters.get("illegal_targets") or counters.get("data_mismatches"):
        return "target_reconstruction_mismatch_max"
    if counters.get("checkpoint_errors"):
        return "checkpoint_resume_failure_max"
    text = str(error).lower()
    if "illegal" in text:
        return "illegal_neural_action_max"
    if "observer" in text:
        return "observer_safety_failure_max"
    return None


def run_pilot_worker(args) -> int:
    """One candidate, end to end: 8 iterations, H005, validations, evidence.

    Restart-safe at iteration granularity: the journal is the authority on
    completed iterations; a crash inside iteration N resumes from the
    iteration N-1 checkpoint and re-executes N against its own sealed (or
    partially collected) rollout under the same behavior snapshot. Exact
    mid-epoch resume exists (Agent 5 proved it); the pilot harness prefers
    re-executing an uncommitted iteration because no partial iteration state
    then needs to be trusted.
    """
    modules = _training()
    contract = modules["contract"]
    schedule = modules["schedule"]
    storage_module = modules["storage"]
    pb = modules["pb"]
    pck = modules["pck"]
    pc = modules["pc"]
    pt = modules["pt"]

    candidate = next(
        entry for entry in contract.PILOT_CANDIDATES if entry["candidate_id"] == args.candidate
    )
    namespace = candidate["namespace"]
    work = candidate_directory(namespace)
    work.mkdir(parents=True, exist_ok=True)

    # The supplementary requirement: mounted external storage, before every
    # candidate and after every restart.
    storage_check = verify_storage_mounted()
    if storage_check["problems"]:
        log(f"BLOCKED: storage not usable: {storage_check['problems']}")
        return 2
    rollout_root = Path(storage_check["resolved_root"])

    journal = load_journal(namespace)
    journal["storage_checks"] = journal.get("storage_checks", [])
    journal["storage_checks"].append(
        {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "resolved_root": storage_check["resolved_root"],
            "mount_point": storage_check["mount_point"],
            "on_external_volume": storage_check["on_external_volume"],
        }
    )

    completed = len(journal["iterations"])
    if completed == 0 and journal.get("veto") is None:
        stale = []
        if (rollout_root / namespace).exists():
            stale.append(str(rollout_root / namespace))
        if (PRODUCTION_ARCHIVE_ROOT / namespace / "H005.pt").exists():
            stale.append(str(PRODUCTION_ARCHIVE_ROOT / namespace / "H005.pt"))
        if any(work.glob("resume_it*.pt")):
            stale.append(str(work / "resume_it*.pt"))
        if stale and not args.reset_candidate:
            log(
                f"BLOCKED: {namespace} has no journal but stale run state exists "
                f"({stale}); pass --reset-candidate to discard it explicitly"
            )
            return 2
        if stale and args.reset_candidate:
            import shutil

            for path in (rollout_root / namespace, work):
                if Path(path).exists():
                    shutil.rmtree(path)
            archive_h005 = PRODUCTION_ARCHIVE_ROOT / namespace / "H005.pt"
            if archive_h005.exists():
                archive_h005.unlink()
            work.mkdir(parents=True, exist_ok=True)
            journal = load_journal(namespace)
            log(f"{namespace}: stale state discarded by explicit --reset-candidate")

    if journal.get("veto"):
        log(f"{namespace}: already vetoed ({journal['veto']['veto']}); nothing to run")
        finalize_pilot_stage(candidate, journal, args)
        return 0

    config = pt.Phase9TrainConfig.for_candidate(args.candidate, device=args.device)
    topology = pt.LoaderTopology(**VALIDATED_TOPOLOGY)
    corpus_identity = accepted_corpus_identity(modules)
    anchor_sha = pb.file_sha256(PHASE8_CHECKPOINT)
    if anchor_sha != contract.EXPECTED_PHASE8_CHECKPOINT_SHA256:
        log("BLOCKED: Phase 8 checkpoint SHA drifted")
        return 2

    resolver = pc.SnapshotResolver(
        device=args.collect_device, inference_batch_shape=args.batch_shape
    )
    anchor_snapshot = resolver.resolve(
        PHASE8_CHECKPOINT,
        logical_identity="H000",
        policy_token=schedule.ANCHOR_POLICY_TOKEN,
        expected_sha256=anchor_sha,
    )
    historical = {"H000": anchor_snapshot}

    # Build or resume the trainer.
    if completed == 0:
        trainer = pt.Phase9Trainer.from_phase8_checkpoint(
            PHASE8_CHECKPOINT,
            config,
            corpus_identity,
            topology=topology,
            run_label=f"agent06_{namespace}",
        )
        journal["start_state_digest"] = trainer.model_state_digest()
        journal["anchor_state_digest"] = anchor_snapshot.loaded_state_dict_digest
        if journal["start_state_digest"] != anchor_snapshot.loaded_state_dict_digest:
            log("BLOCKED: fresh trainer weights differ from the Phase 8 anchor")
            return 2
        journal["train_config"] = {**config.identity(), "digest": config.digest()}
        journal["topology"] = topology.to_dict()
        save_journal(namespace, journal)
        log(f"{namespace}: fresh start from the Phase 8 anchor ({anchor_sha[:16]})")
    else:
        resume_from = work / f"resume_it{completed:03d}.pt"
        trainer = pt.Phase9Trainer.resume(
            resume_from,
            config=config,
            corpus_identity=corpus_identity,
            topology=topology,
            run_label=f"agent06_{namespace}_resumed",
        )
        log(f"{namespace}: resumed after iteration {completed} from {resume_from.name}")

    # Rebind H005 on resume, and finalize any journal-complete iteration whose
    # store state a crash left at TRAINING.
    if completed >= 5:
        member = pck.read_archive_member(
            PRODUCTION_ARCHIVE_ROOT, namespace=namespace, local_identity="H005"
        )
        bound = pck.bind_archive_member(
            member, device=args.collect_device, inference_batch_shape=args.batch_shape
        )
        bound.assert_frozen()
        historical["H005"] = bound
    from stratego.training.phase9_rollout_store import read_iteration_state, write_iteration_state

    for entry in journal["iterations"]:
        state = read_iteration_state(rollout_root, namespace, entry["iteration"])
        if state is not None and state["state"] == "TRAINING":
            write_iteration_state(
                rollout_root, namespace, entry["iteration"], "EVALUATED",
                sealed_rollout_digest=entry["sealed_rollout_digest"],
            )
            write_iteration_state(
                rollout_root, namespace, entry["iteration"], "COMMITTED",
                sealed_rollout_digest=entry["sealed_rollout_digest"],
            )

    candidate_started = time.perf_counter()
    veto = None

    def run_due_validation(due_iteration: int) -> "dict | None":
        """The frozen cadence's validation pass for one completed iteration.

        Returns a veto record when the pass itself vetoes; the Random/Basic
        guards bind only at the frozen iteration-8 pass — the iteration-4
        numbers are recorded as intermediate diagnostics.
        """
        trainer.close()
        snapshot_id = schedule.behavior_snapshot_identity(due_iteration + 1)
        weights = work / f"behavior_{snapshot_id}.pt"
        validation = run_validation_pass(
            candidate,
            due_iteration,
            weights,
            journal["snapshots"][snapshot_id],
            args,
            include_stress=due_iteration == contract.PILOT_ITERATIONS,
        )
        journal["validations"][str(due_iteration)] = validation
        save_journal(namespace, journal)
        log(
            f"{namespace} it{due_iteration} validation: "
            f"score={validation['selection_score']:.6f} "
            f"random={validation['effective_win_rates']['random_legal']:.4f} "
            f"basic={validation['effective_win_rates']['basic_heuristic']:.4f} "
            f"({validation['guards']['binding']})"
        )
        if validation["safety"]["illegal_policy_actions"]:
            return {
                "veto": "illegal_neural_action_max",
                "iteration": due_iteration,
                "detail": "illegal neural action during validation",
            }
        if validation["safety"]["inference_failures"]:
            return {
                "veto": "non_finite_loss_max",
                "iteration": due_iteration,
                "detail": "inference failure (non-finite output) during validation",
            }
        if due_iteration == contract.PILOT_ITERATIONS:
            if not validation["guards"]["random_pass"]:
                return {
                    "veto": "validation_random_ewr_min",
                    "iteration": due_iteration,
                    "detail": (
                        f"final Random EWR "
                        f"{validation['effective_win_rates']['random_legal']}"
                    ),
                }
            if not validation["guards"]["basic_pass"]:
                return {
                    "veto": "validation_basic_ewr_min",
                    "iteration": due_iteration,
                    "detail": (
                        f"final Basic EWR "
                        f"{validation['effective_win_rates']['basic_heuristic']}"
                    ),
                }
        return None

    try:
        # A restart that lands between an iteration and its due validation pass
        # fills the gap first: validation is a pure function of the frozen bank
        # and the already-saved snapshot file, so cadence position, not process
        # lifetime, decides what runs.
        for due in (4, contract.PILOT_ITERATIONS):
            if (
                len(journal["iterations"]) >= due
                and str(due) not in journal["validations"]
            ):
                veto = run_due_validation(due)
                if veto is not None:
                    raise _VetoSignal()
        for iteration in range(completed + 1, contract.PILOT_ITERATIONS + 1):
            timers: dict[str, float] = {}
            identity = schedule.behavior_snapshot_identity(iteration)
            if iteration == 1:
                snapshot = resolver.resolve(
                    PHASE8_CHECKPOINT,
                    logical_identity=identity,
                    policy_token=schedule.behavior_policy_token(namespace, iteration),
                    expected_sha256=anchor_sha,
                )
            else:
                behavior_path = work / f"behavior_{identity}.pt"
                snapshot = pck.bind_behavior_snapshot(
                    behavior_path,
                    logical_identity=identity,
                    namespace=namespace,
                    device=args.collect_device,
                    inference_batch_shape=args.batch_shape,
                    expected_sha256=journal["snapshots"][identity],
                )
            if snapshot.loaded_state_dict_digest != trainer.model_state_digest():
                raise Agent6Error(
                    f"{namespace} iteration {iteration}: behavior snapshot weights "
                    "differ from the live trainer weights; on-policy collection "
                    "would be a lie"
                )

            window = contract.active_historical_window(iteration)
            manifest = schedule.ActiveHistoryManifest.frozen_for(
                namespace,
                iteration,
                {key: historical[key].checkpoint_sha256 for key in window},
            )
            manifest.validate()

            started = time.perf_counter()
            collected = pc.collect_iteration(
                rollout_root,
                namespace,
                iteration,
                pc.IterationParticipants(
                    behavior=snapshot,
                    historical={key: historical[key] for key in window},
                ),
                population_version=contract.PHASE9_POPULATION_VERSION,
                schedule_version=contract.PHASE9_ROLLOUT_SCHEDULE_VERSION,
                contract_digest=contract.contract_digest(),
                games_in_flight=args.games_in_flight,
                observer_probe_plies=args.observer_probe_plies,
                history=manifest,
                progress=lambda done, total: log(
                    f"  {namespace} it{iteration}: collected {done}/{total}"
                ),
            )
            timers["collection_seconds"] = time.perf_counter() - started
            timers["sealing_seconds"] = float(
                (collected.get("seal") or {}).get("seconds", 0.0)
            )
            if collected.get("observer_probe_failures"):
                veto = {
                    "veto": "observer_safety_failure_max",
                    "iteration": iteration,
                    "detail": f"{collected['observer_probe_failures']} observer probe failures",
                }
                break

            started = time.perf_counter()
            state = read_iteration_state(rollout_root, namespace, iteration)
            rollout = pt.bind_sealed_rollout(
                rollout_root,
                namespace,
                iteration,
                behavior_snapshot=snapshot,
                expected_model_state_digest=trainer.model_state_digest(),
                require_full_schedule=True,
                resuming=state is not None and state["state"] == "TRAINING",
            )
            timers["target_construction_seconds"] = time.perf_counter() - started

            verification = None
            if "H005" in window:
                verification = verify_historical_actions(
                    rollout.reader,
                    iteration,
                    namespace,
                    historical,
                    h000_sample=args.h000_verify_games,
                )
                timers["historical_verification_seconds"] = verification["seconds"]
                if not verification["all_verified"]:
                    veto = {
                        "veto": "behavior_identity_mismatch_max",
                        "iteration": iteration,
                        "detail": "historical action reproduction failed",
                        "verification": verification,
                    }
                    break
            elif iteration <= 5 and args.h000_verify_games:
                verification = verify_historical_actions(
                    rollout.reader,
                    iteration,
                    namespace,
                    historical,
                    h000_sample=args.h000_verify_games,
                )
                timers["historical_verification_seconds"] = verification["seconds"]
                if not verification["all_verified"]:
                    veto = {
                        "veto": "behavior_identity_mismatch_max",
                        "iteration": iteration,
                        "detail": "historical action reproduction failed",
                        "verification": verification,
                    }
                    break

            trainer.bind_iteration(rollout)
            started = time.perf_counter()
            rows = trainer.train_iteration(timing=True)
            timers["train_seconds"] = time.perf_counter() - started
            trainer.mark_iteration_trained()

            started = time.perf_counter()
            trainer.save_checkpoint(work / f"resume_it{iteration:03d}.pt")
            next_identity = schedule.behavior_snapshot_identity(iteration + 1)
            written = trainer.save_behavior_snapshot(
                work / f"behavior_{next_identity}.pt",
                logical_identity=next_identity,
                rl_iteration=iteration + 1,
            )
            journal["snapshots"][next_identity] = written["sha256"]
            timers["checkpoint_seconds"] = time.perf_counter() - started

            archived = None
            if iteration % contract.ARCHIVE_CADENCE_ITERATIONS == 0:
                started = time.perf_counter()
                local_identity = contract.archive_snapshot_id(iteration)
                existing = PRODUCTION_ARCHIVE_ROOT / namespace / f"{local_identity}.pt"
                if existing.exists():
                    member = pck.read_archive_member(
                        PRODUCTION_ARCHIVE_ROOT,
                        namespace=namespace,
                        local_identity=local_identity,
                    )
                    if member.state_dict_digest != trainer.model_state_digest():
                        raise Agent6Error(
                            f"{namespace}: an archive member {local_identity} already "
                            "exists with different weights (a discarded crashed "
                            "attempt); archives are immutable — inspect and "
                            "--reset-candidate to re-derive"
                        )
                else:
                    payload = trainer.archive_member_payload(local_identity=local_identity)
                    member = pck.write_archive_member(
                        payload,
                        PRODUCTION_ARCHIVE_ROOT,
                        namespace=namespace,
                        local_identity=local_identity,
                    )
                bound = pck.bind_archive_member(
                    member,
                    device=args.collect_device,
                    inference_batch_shape=args.batch_shape,
                )
                bound.assert_frozen()
                if bound.loaded_state_dict_digest != trainer.model_state_digest():
                    raise Agent6Error(
                        f"{namespace}: archived {local_identity} weights do not match "
                        "the post-iteration learner"
                    )
                historical[local_identity] = bound
                archived = member.to_dict()
                journal["archive"] = archived
                timers["archive_seconds"] = time.perf_counter() - started
                log(f"{namespace}: archived {member.qualified_identity} -> {member.checkpoint_sha256[:16]}")

            entry = summarize_pilot_iteration(
                rollout, rows, collected, timers, archived, trainer.controller
            )
            if verification is not None:
                journal["historical_verification"].append(verification)
            journal["iterations"].append(entry)
            save_journal(namespace, journal)

            write_iteration_state(
                rollout_root, namespace, iteration, "EVALUATED",
                sealed_rollout_digest=rollout.sealed_rollout_digest,
                validation_pass_due=iteration in (4, contract.PILOT_ITERATIONS),
            )
            write_iteration_state(
                rollout_root, namespace, iteration, "COMMITTED",
                sealed_rollout_digest=rollout.sealed_rollout_digest,
            )
            log(
                f"{namespace} it{iteration}: {entry['updates']} updates, "
                f"beta={trainer.controller.beta:.4f}, "
                f"epoch KLs={['%.4f' % value for value in entry['epoch_mean_kls']]}, "
                f"{timers['collection_seconds']:.0f}s collect / "
                f"{timers['train_seconds']:.0f}s train"
            )

            if iteration in (4, contract.PILOT_ITERATIONS):
                veto = run_due_validation(iteration)
                if veto is not None:
                    break
    except _VetoSignal:
        pass
    except (pt.Phase9TrainerError, pc.Phase9CollectorError, Agent6Error) as error:
        classified = classify_trainer_veto(error, locals().get("trainer"))
        if classified is None:
            log(f"HARNESS FAILURE in {namespace}: {error}")
            traceback.print_exc()
            return 1
        veto = {
            "veto": classified,
            "iteration": locals().get("iteration"),
            "detail": f"{type(error).__name__}: {error}",
        }
    finally:
        try:
            trainer.close()
        except Exception:  # noqa: BLE001 - close is best-effort on the way out
            pass

    journal["wall_clock"]["worker_seconds"] = journal["wall_clock"].get(
        "worker_seconds", 0.0
    ) + (time.perf_counter() - candidate_started)
    if veto is not None:
        journal["veto"] = veto
        log(f"{namespace}: VETOED — {veto['veto']} ({veto['detail']})")
    # Counters accumulate across worker restarts: each process's trainer counts
    # its own events, and a veto ends the candidate, so summing never double
    # counts a decision that mattered.
    merged = dict(journal.get("counters", {}))
    for key, value in trainer.counters.items():
        merged[key] = merged.get(key, 0) + int(value)
    journal["counters"] = merged
    save_journal(namespace, journal)
    finalize_pilot_stage(candidate, journal, args)
    return 0


def finalize_pilot_stage(candidate: dict, journal: dict, args) -> None:
    """Assemble the durable per-candidate stage payload from the journal."""
    namespace = candidate["namespace"]
    iterations = journal["iterations"]
    totals = {
        "iterations_completed": len(iterations),
        "games": sum(entry["games"] for entry in iterations),
        "updates": sum(entry["updates"] for entry in iterations),
        "examples": sum(entry["examples"] for entry in iterations),
        "learner_decisions": sum(entry["learner_decisions"] for entry in iterations),
        "collection_seconds": sum(
            entry["timers"].get("collection_seconds", 0.0) for entry in iterations
        ),
        "target_construction_seconds": sum(
            entry["timers"].get("target_construction_seconds", 0.0) for entry in iterations
        ),
        "train_seconds": sum(entry["timers"].get("train_seconds", 0.0) for entry in iterations),
        "checkpoint_seconds": sum(
            entry["timers"].get("checkpoint_seconds", 0.0) for entry in iterations
        ),
        "archive_seconds": sum(
            entry["timers"].get("archive_seconds", 0.0) for entry in iterations
        ),
        "historical_verification_seconds": sum(
            entry["timers"].get("historical_verification_seconds", 0.0)
            for entry in iterations
        ),
        "validation_seconds": sum(
            entry["seconds"] for entry in journal["validations"].values()
        ),
    }
    totals["examples_per_second"] = (
        totals["examples"] / totals["train_seconds"] if totals["train_seconds"] else 0.0
    )
    kl_weighted = sum(entry["kl_weighted_sum"] for entry in iterations)
    totals["run_mean_behavior_kl"] = (
        kl_weighted / totals["examples"] if totals["examples"] else 0.0
    )
    payload = {
        "stage": pilot_stage_name(candidate["candidate_id"]),
        **environment_record(),
        "candidate": dict(candidate),
        "train_config": journal.get("train_config"),
        "topology": journal.get("topology"),
        "start_state_digest": journal.get("start_state_digest"),
        "anchor_state_digest": journal.get("anchor_state_digest"),
        "status": "VETOED" if journal.get("veto") else (
            "COMPLETE"
            if len(iterations) == 8 and {"4", "8"} <= set(journal["validations"])
            else "INCOMPLETE"
        ),
        "veto": journal.get("veto"),
        "iterations": iterations,
        "validations": journal["validations"],
        "snapshots": journal["snapshots"],
        "archive_member": journal.get("archive"),
        "historical_verification": journal["historical_verification"],
        "counters": journal.get("counters", {}),
        "totals": totals,
        "storage_checks": journal.get("storage_checks", []),
        "wall_clock": journal.get("wall_clock", {}),
    }
    write_stage(pilot_stage_name(candidate["candidate_id"]), payload)


def stage_pilots(args) -> None:
    verify = read_stage("verify")
    if verify["problems"]:
        raise Agent6Error(f"verify stage recorded problems: {verify['problems']}")
    for candidate_id in CANDIDATE_ORDER:
        stage_file = stage_path(pilot_stage_name(candidate_id))
        if stage_file.exists():
            payload = read_json(stage_file)
            if payload["status"] in ("COMPLETE", "VETOED"):
                log(f"{candidate_id}: already {payload['status']}; skipping")
                continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--pilot-worker",
            "--candidate",
            candidate_id,
            "--device",
            args.device,
            "--collect-device",
            args.collect_device,
            "--batch-shape",
            str(args.batch_shape),
            "--games-in-flight",
            str(args.games_in_flight),
            "--observer-probe-plies",
            str(args.observer_probe_plies),
            "--eval-workers",
            str(args.eval_workers),
            "--anchor-workers",
            str(args.anchor_workers),
            "--chunk-units",
            str(args.chunk_units),
            "--h000-verify-games",
            str(args.h000_verify_games),
        ]
        log(f"=== launching pilot worker {candidate_id} ===")
        completed = subprocess.run(command, cwd=REPOSITORY_ROOT)
        if completed.returncode != 0:
            raise Agent6Error(
                f"pilot worker {candidate_id} failed with {completed.returncode}; "
                "fix the harness and re-run --stage pilots (completed candidates skip)"
            )


# ---------------------------------------------------------------------------
# Stage: selection
# ---------------------------------------------------------------------------


def stage_selection() -> dict:
    modules = _training()
    contract = modules["contract"]
    candidates = []
    for candidate_id in CANDIDATE_ORDER:
        payload = read_stage(pilot_stage_name(candidate_id))
        candidates.append(payload)

    rows = []
    for payload in candidates:
        candidate = payload["candidate"]
        final = payload["validations"].get("8")
        entry = {
            "candidate_id": candidate["candidate_id"],
            "namespace": candidate["namespace"],
            "learning_rate": candidate["learning_rate"],
            "initial_kl_beta": candidate["initial_kl_beta"],
            "status": payload["status"],
            "veto": payload["veto"],
            "iterations_completed": payload["totals"]["iterations_completed"],
            "games": payload["totals"]["games"],
            "run_mean_behavior_kl": payload["totals"]["run_mean_behavior_kl"],
            "examples_per_second": payload["totals"]["examples_per_second"],
            "final_validation": None,
            "eligible": False,
        }
        if final is not None:
            entry["final_validation"] = {
                "selection_score": final["selection_score"],
                "effective_win_rates": final["effective_win_rates"],
                "guards": final["guards"],
            }
        entry["eligible"] = (
            payload["status"] == "COMPLETE"
            and final is not None
            and final["guards"]["random_pass"]
            and final["guards"]["basic_pass"]
        )
        rows.append(entry)

    eligible = [entry for entry in rows if entry["eligible"]]
    if not eligible:
        selection = {
            "winner": None,
            "reason": "no candidate survived the frozen vetoes and final guards",
        }
    else:
        def sort_key(entry):
            final = entry["final_validation"]
            return (
                -final["selection_score"],
                -final["effective_win_rates"]["strategic_rule_based"],
                entry["run_mean_behavior_kl"],
                -entry["examples_per_second"],
            )

        ranked = sorted(eligible, key=sort_key)
        winner = ranked[0]
        distinct = [
            entry
            for entry in ranked[1:]
            if sort_key(entry) == sort_key(winner)
        ]
        selection = {
            "winner": winner["candidate_id"],
            "winner_namespace": winner["namespace"],
            "winner_score": winner["final_validation"]["selection_score"],
            "ranked": [entry["candidate_id"] for entry in ranked],
            "tie_after_full_chain": [entry["candidate_id"] for entry in distinct],
            "unique": not distinct,
            "tie_break": list(contract.VALIDATION_TIE_BREAK),
        }

    payload = {
        "stage": "selection",
        **environment_record(),
        "score_rule": (
            "S = 0.45*E_strategic + 0.35*E_tactical + 0.20*E_phase8_anchor, "
            "evaluated only at the frozen iteration-8 validation pass"
        ),
        "score_weights": dict(contract.VALIDATION_SCORE_WEIGHTS),
        "guards": dict(contract.VALIDATION_REGRESSION_GUARDS),
        "guard_binding_rule": (
            "iteration-4 Random/Basic results are intermediate diagnostics; the "
            "frozen guards bind at the iteration-8 validation pass"
        ),
        "hard_vetoes": dict(contract.PILOT_HARD_VETOES),
        "candidates": rows,
        "selection": selection,
    }
    write_stage("selection", payload)
    if selection.get("winner"):
        log(f"selection: winner {selection['winner']} score {selection['winner_score']:.6f}")
    else:
        log(f"selection: BLOCKED — {selection['reason']}")
    return payload


# ---------------------------------------------------------------------------
# Stage: config — freeze phase9_train_config_v1
# ---------------------------------------------------------------------------


def canonical_json(document) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def document_digest(document) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(document).encode()).hexdigest()


def stage_config() -> dict:
    modules = _training()
    contract = modules["contract"]
    schedule = modules["schedule"]
    seed = modules["seed"]
    pt = modules["pt"]
    pck = modules["pck"]
    selection = read_stage("selection")
    winner_id = selection["selection"]["winner"]
    if winner_id is None:
        raise Agent6Error("no winner; the train config cannot be frozen")
    winner = next(
        entry for entry in contract.PILOT_CANDIDATES if entry["candidate_id"] == winner_id
    )
    verify = read_stage("verify")
    agent1 = read_json(DATA_DIRECTORY / "agent_01_acceptance.json")

    document = {
        "config_version": "phase9_train_config_v1",
        "selected_by": "phase9_agent6_bounded_pilot_selection",
        "winning_candidate_id": winner_id,
        "pilot_namespace": winner["namespace"],
        "canonical_namespace": "canonical",
        "model": {
            "architecture": "C1",
            "parameters": contract.EXPECTED_C1_PARAMETERS,
            "config_digest": contract.EXPECTED_C1_CONFIG_DIGEST,
            "model_contract": "model_contract_v2",
            "observation": "observation_v2_1_127ch",
            "action_frame": "perspective_normalized_squares",
        },
        "start": {
            "checkpoint_path": contract.EXPECTED_PHASE8_CHECKPOINT_PATH,
            "checkpoint_sha256": contract.EXPECTED_PHASE8_CHECKPOINT_SHA256,
            "expected_model_state_digest": read_stage(
                pilot_stage_name(winner_id)
            )["start_state_digest"],
            "rule": (
                "Agent 7 starts freshly from the accepted Phase 8 checkpoint with "
                "fresh optimizer/scheduler/KL-controller state; no pilot "
                "checkpoint is continued"
            ),
        },
        "learning_rate": winner["learning_rate"],
        "initial_kl_beta": winner["initial_kl_beta"],
        "ppo": {
            "clip_epsilon": contract.PPO_CLIP_EPSILON,
            "behavior_kl_target": contract.BEHAVIOR_KL_TARGET,
            "kl_beta_increase_threshold": contract.KL_BETA_INCREASE_THRESHOLD,
            "kl_beta_decrease_threshold": contract.KL_BETA_DECREASE_THRESHOLD,
            "kl_beta_increase_factor": contract.KL_BETA_INCREASE_FACTOR,
            "kl_beta_decrease_factor": contract.KL_BETA_DECREASE_FACTOR,
            "kl_beta_clamp": [contract.KL_BETA_MIN, contract.KL_BETA_MAX],
            "kl_hard_limit": contract.KL_HARD_LIMIT,
            "clip_fraction_hard_limit": contract.CLIP_FRACTION_HARD_LIMIT,
            "value_loss_weight": contract.VALUE_LOSS_WEIGHT,
            "belief_loss_weight": contract.BELIEF_LOSS_WEIGHT,
        },
        "advantages": {
            "gamma": contract.GAMMA,
            "lambda_advantage": contract.LAMBDA_ADVANTAGE,
            "lambda_value": contract.LAMBDA_VALUE,
            "filter_quantile": contract.ADVANTAGE_FILTER_QUANTILE,
            "filter_floor": contract.ADVANTAGE_FILTER_FLOOR,
            "standardization_epsilon": contract.ADVANTAGE_STANDARDIZATION_EPSILON,
        },
        "behavior_policy": {
            "temperature": contract.BEHAVIOR_TEMPERATURE,
            "probability_abs_tolerance": contract.BEHAVIOR_PROBABILITY_ABS_TOLERANCE,
            "log_epsilon": contract.BEHAVIOR_LOG_EPSILON,
        },
        "entropy_schedule": {
            "start": contract.ENTROPY_COEFFICIENT_START,
            "end": contract.ENTROPY_COEFFICIENT_END,
            "rule": "linear in the 1-based iteration over the run's own budget",
        },
        "optimizer": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in contract.OPTIMIZER_CONSTRAINTS.items()
        },
        "batch_size": contract.MINIBATCH_SIZE,
        "epochs_per_rollout": contract.EPOCHS_PER_ROLLOUT,
        "population_version": contract.PHASE9_POPULATION_VERSION,
        "schedule_version": contract.PHASE9_ROLLOUT_SCHEDULE_VERSION,
        "rollout_store_version": contract.PHASE9_ROLLOUT_STORE_VERSION,
        "advantage_version": contract.PHASE9_ADVANTAGE_VERSION,
        "train_order_version": contract.PHASE9_TRAIN_ORDER_VERSION,
        "example_version": "phase9_example_v1",
        "collector_version": "phase9_collector_v1",
        "trajectory": {"version": "trajectory_v1", "snapshot_interval": 32},
        "historical_archive_rule": {
            "anchor": contract.HISTORICAL_ANCHOR_ID,
            "archive_cadence_iterations": contract.ARCHIVE_CADENCE_ITERATIONS,
            "active_window_recent_snapshots": contract.ACTIVE_WINDOW_RECENT_SNAPSHOTS,
            "sampling": "uniform over the active window via the frozen opponent stream",
            "immutability": "no archive checkpoint may be overwritten",
        },
        "validation_cadence_iterations": contract.VALIDATION_CADENCE_ITERATIONS,
        "archive_cadence_iterations": contract.ARCHIVE_CADENCE_ITERATIONS,
        "canonical_iterations": contract.CANONICAL_ITERATIONS,
        "canonical_games_per_iteration": contract.CANONICAL_GAMES_PER_ITERATION,
        "canonical_bucket_counts": dict(contract.CANONICAL_BUCKET_COUNTS),
        "canonical_rule_tier_counts": dict(contract.CANONICAL_RULE_TIER_COUNTS),
        "wall_clock_ceiling_hours": contract.CANONICAL_WALL_CLOCK_CEILING_HOURS,
        "loader_collector_topology": {
            **dict(VALIDATED_TOPOLOGY),
            "games_in_flight": 96,
            "inference_batch_shape": 64,
            "inference_device": "mps",
            "observer_probe_plies": 2,
        },
        "seeds": dict(seed.CANONICAL_PHASE9_SEEDS),
        "checkpoint_version": contract.PHASE9_CHECKPOINT_VERSION,
        "acceptance_versions": {
            "acceptance_version": contract.PHASE9_ACCEPTANCE_VERSION,
            "eval_bank_version": contract.PHASE9_EVAL_BANK_VERSION,
            "rl_contract_version": contract.PHASE9_RL_CONTRACT_VERSION,
            "contract_digest": verify["contract_digest"],
            "example_contract_digest": verify["example_contract_digest"],
        },
        "validation": {
            "bank_version": contract.VALIDATION_BANK_VERSION,
            "bank_digest": verify["validation_bank_digest_expected"],
            "score_weights": dict(contract.VALIDATION_SCORE_WEIGHTS),
            "regression_guards": dict(contract.VALIDATION_REGRESSION_GUARDS),
            "tie_break": list(contract.VALIDATION_TIE_BREAK),
            "best_checkpoint_rule": (
                "strictly highest frozen validation score among cadence passes"
            ),
        },
        "test_bank": {
            "bank_version": contract.TEST_BANK_VERSION,
            "bank_digest": agent1["bank_digests"]["phase9_test_bank_v1"],
            "sealed_until": "Agent 8 final evaluation",
        },
        "corpus_identity": {
            "version": contract.EXPECTED_CORPUS_VERSION,
            "content_digest": contract.EXPECTED_CORPUS_CONTENT_DIGEST,
            "metadata_digest": contract.EXPECTED_CORPUS_METADATA_DIGEST,
            "commit_index_digest": contract.EXPECTED_CORPUS_COMMIT_INDEX_DIGEST,
        },
    }
    config_digest = document_digest(document)

    runtime = pt.Phase9TrainConfig.for_candidate(
        winner_id,
        namespace="canonical",
        device="mps",
        total_iterations=contract.CANONICAL_ITERATIONS,
    )
    runtime_identity = runtime.identity()
    runtime_digest = runtime.digest()

    document_fields = sorted(document.keys())
    runtime_fields = sorted(runtime_identity.keys())
    field_map = {
        "learning_rate": "learning_rate",
        "initial_kl_beta": "initial_kl_beta",
        "batch_size": "minibatch_size",
        "epochs_per_rollout": "epochs_per_rollout",
        "canonical_iterations": "total_iterations",
        "canonical_namespace": "namespace",
        "winning_candidate_id": "candidate_id",
    }
    reconciliation = {
        "note": (
            "two distinct namespaces, exactly as Phase 8 froze them: the "
            "document is the complete reviewable phase9_train_config_v1; the "
            "runtime identity is the narrower Phase9TrainConfig object stamped "
            "into every phase9_checkpoint_v1 and compared on resume. They hash "
            "different objects and are not required to be equal."
        ),
        "document_field_count": len(document_fields),
        "runtime_field_count": len(runtime_fields),
        "bridged_fields": [
            {
                "document_field": doc_field,
                "runtime_field": run_field,
                "document_value": document[doc_field],
                "runtime_value": runtime_identity[run_field],
                "equal": _bridge_equal(document[doc_field], runtime_identity[run_field]),
            }
            for doc_field, run_field in sorted(field_map.items())
        ],
        "runtime_only_fields": sorted(
            set(runtime_fields)
            - set(field_map.values())
            - {"trainer_version", "contract_digest"}
        ),
        "runtime_fields_bound_by_document_subobjects": {
            "device": "loader_collector_topology has no device for the optimizer; runtime.device='mps' matches optimizer.device",
            "precision": "optimizer.precision",
            "weight_decay": "optimizer.weight_decay",
            "adam_beta1": "optimizer.adam_betas[0]",
            "adam_beta2": "optimizer.adam_betas[1]",
            "adam_epsilon": "optimizer.adam_epsilon",
            "gradient_clip_norm": "optimizer.gradient_clip_norm",
            "optimizer": "optimizer.optimizer",
            "learning_rate_schedule": "optimizer.learning_rate_schedule",
            "model_candidate": "model.architecture",
            "scope": (
                "runtime bookkeeping only: the frozen constructor's production "
                "scope is 'pilot_candidate'; SCOPES has no canonical entry, so "
                "the canonical runtime object carries the constructor default"
            ),
        },
        "document_only_fields": sorted(
            set(document_fields)
            - set(field_map)
            - {
                "optimizer",
                "model",
            }
        ),
    }

    payload = {
        "phase": PHASE,
        "agent": AGENT,
        "artifact": "agent_06_frozen_train_config",
        **environment_record(),
        "config": document,
        "train_config_document_digest": config_digest,
        "trainer_runtime_identity": runtime_identity,
        "trainer_runtime_identity_digest": runtime_digest,
        "digest_namespace_rule": (
            "train_config_document_digest hashes the canonical JSON of `config` "
            "(the phase9_train_config_v1 document); "
            "trainer_runtime_identity_digest hashes "
            "Phase9TrainConfig.identity() for the canonical run. Always label "
            "which namespace a digest belongs to."
        ),
        "reconciliation": reconciliation,
        "handoff_to_agent_7": {
            "winning_candidate_id": winner_id,
            "train_config_document_digest": config_digest,
            "trainer_runtime_identity_digest": runtime_digest,
            "fresh_start_checkpoint_sha256": contract.EXPECTED_PHASE8_CHECKPOINT_SHA256,
            "expected_model_state_digest": document["start"]["expected_model_state_digest"],
            "seeds": dict(seed.CANONICAL_PHASE9_SEEDS),
            "population_version": contract.PHASE9_POPULATION_VERSION,
            "schedule_version": contract.PHASE9_ROLLOUT_SCHEDULE_VERSION,
            "topology": document["loader_collector_topology"],
            "canonical_budget": {
                "iterations": contract.CANONICAL_ITERATIONS,
                "games_per_iteration": contract.CANONICAL_GAMES_PER_ITERATION,
                "epochs_per_rollout": contract.EPOCHS_PER_ROLLOUT,
                "validation_cadence": contract.VALIDATION_CADENCE_ITERATIONS,
                "archive_cadence": contract.ARCHIVE_CADENCE_ITERATIONS,
            },
            "no_pilot_checkpoint_handed_forward": True,
            "pilot_weights_location": str(WORK_DIRECTORY),
            "pilot_archive_members": "checkpoints/phase9/archive/pilot_p9*/H005.pt (pilot-local, never read by the canonical namespace)",
        },
        "checkpoint_version_declared_by": pck.PHASE9_CHECKPOINT_VERSION
        if hasattr(pck, "PHASE9_CHECKPOINT_VERSION")
        else contract.PHASE9_CHECKPOINT_VERSION,
    }
    write_stage("config", payload)
    log(
        f"config: document digest {config_digest[:16]}…, runtime identity "
        f"digest {runtime_digest[:16]}…"
    )
    return payload


def _bridge_equal(document_value, runtime_value) -> bool:
    if document_value == runtime_value:
        return True
    if document_value == "canonical" and runtime_value == "canonical":
        return True
    return False


# ---------------------------------------------------------------------------
# Stage: projection — the canonical 12 h ceiling check
# ---------------------------------------------------------------------------


def stage_projection() -> dict:
    modules = _training()
    contract = modules["contract"]
    selection = read_stage("selection")
    winner_id = selection["selection"]["winner"]
    if winner_id is None:
        raise Agent6Error("no winner; nothing to project")
    pilot = read_stage(pilot_stage_name(winner_id))
    totals = pilot["totals"]
    iterations = pilot["iterations"]

    games_scale = contract.CANONICAL_GAMES_PER_ITERATION / contract.PILOT_GAMES_PER_ITERATION
    collection_rate = totals["games"] / totals["collection_seconds"]
    mean_learner_decisions = totals["learner_decisions"] / len(iterations)
    max_learner_decisions = max(entry["learner_decisions"] for entry in iterations)
    examples_per_second = totals["examples_per_second"]

    def project(decisions_per_iteration: float) -> dict:
        canonical_decisions = decisions_per_iteration * games_scale
        canonical_examples = canonical_decisions * contract.EPOCHS_PER_ROLLOUT
        collection = contract.CANONICAL_GAMES_PER_ITERATION / collection_rate
        train = canonical_examples / examples_per_second
        targets = (totals["target_construction_seconds"] / len(iterations)) * games_scale
        checkpoint = totals["checkpoint_seconds"] / len(iterations)
        per_iteration = collection + train + targets + checkpoint
        archive_events = contract.CANONICAL_ITERATIONS // contract.ARCHIVE_CADENCE_ITERATIONS
        archive = (
            (totals["archive_seconds"] / max(1, len([e for e in iterations if e["archived"]])))
            * archive_events
        )
        validations = pilot["validations"]
        core_pass_seconds = []
        for record in validations.values():
            stress = (record.get("timings") or {}).get("stress_seconds", 0.0)
            core_pass_seconds.append(record["seconds"] - stress)
        validation_passes = contract.CANONICAL_ITERATIONS // contract.VALIDATION_CADENCE_ITERATIONS
        validation = max(core_pass_seconds) * validation_passes
        total = per_iteration * contract.CANONICAL_ITERATIONS + archive + validation
        return {
            "collection_seconds_per_iteration": collection,
            "train_seconds_per_iteration": train,
            "target_seconds_per_iteration": targets,
            "checkpoint_seconds_per_iteration": checkpoint,
            "per_iteration_seconds": per_iteration,
            "iterations": contract.CANONICAL_ITERATIONS,
            "archive_events": archive_events,
            "archive_seconds_total": archive,
            "validation_passes": validation_passes,
            "validation_seconds_per_pass": max(core_pass_seconds),
            "validation_seconds_total": validation,
            "projected_total_seconds": total,
            "projected_total_hours": total / 3600.0,
        }

    mean_projection = project(mean_learner_decisions)
    peak_projection = project(max_learner_decisions)

    # Restart allowance: one restart re-executes at most one full iteration
    # plus checkpoint reload and rebinding — measured quantities, not guesses.
    restart_unit = (
        peak_projection["per_iteration_seconds"]
        + totals["target_construction_seconds"] / len(iterations) * games_scale
    )
    ceiling_seconds = contract.CANONICAL_WALL_CLOCK_CEILING_HOURS * 3600.0
    payload = {
        "stage": "projection",
        **environment_record(),
        "winner": winner_id,
        "measured_basis": {
            "pilot_iterations": len(iterations),
            "pilot_games": totals["games"],
            "collection_games_per_second": collection_rate,
            "training_examples_per_second": examples_per_second,
            "mean_learner_decisions_per_pilot_iteration": mean_learner_decisions,
            "max_learner_decisions_per_pilot_iteration": max_learner_decisions,
            "core_validation_pass_seconds": {
                iteration: record["seconds"]
                - (record.get("timings") or {}).get("stress_seconds", 0.0)
                for iteration, record in pilot["validations"].items()
            },
            "note": (
                "collection rate and examples/s are the winner's own measured "
                "end-to-end pilot numbers under the frozen topology; canonical "
                "iterations double the games and (proportionally) the learner "
                "decisions of a pilot iteration"
            ),
        },
        "projection_mean_decisions": mean_projection,
        "projection_peak_decisions": peak_projection,
        "restart_allowance_seconds_per_restart": restart_unit,
        "ceiling_hours": contract.CANONICAL_WALL_CLOCK_CEILING_HOURS,
        "ceiling_seconds": ceiling_seconds,
        "fits_mean": mean_projection["projected_total_seconds"] <= ceiling_seconds,
        "fits_peak": peak_projection["projected_total_seconds"] <= ceiling_seconds,
        "verdict": (
            "WITHIN_CEILING"
            if peak_projection["projected_total_seconds"] <= ceiling_seconds
            else "BLOCKED — CANONICAL WALL-CLOCK CONTRACT REQUIRES REVIEW"
        ),
        "frozen_experiment_rule": (
            "the 12-hour ceiling, 60 iterations, 2,048 games/iteration, "
            "2 epochs and the twelve validation passes are frozen; none may be "
            "silently altered to make the projection fit"
        ),
    }
    write_stage("projection", payload)
    log(
        f"projection: mean {mean_projection['projected_total_hours']:.2f} h, "
        f"peak {peak_projection['projected_total_hours']:.2f} h vs ceiling "
        f"{contract.CANONICAL_WALL_CLOCK_CEILING_HOURS} h -> {payload['verdict']}"
    )
    return payload


# ---------------------------------------------------------------------------
# Stage: artifacts — gates, CSV, the three artifacts
# ---------------------------------------------------------------------------


def write_runs_csv(pilots: dict) -> None:
    RUNS_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_ARTIFACT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for candidate_id in CANDIDATE_ORDER:
            payload = pilots[candidate_id]
            candidate = payload["candidate"]
            for iteration in ("4", "8"):
                record = payload["validations"].get(iteration)
                if record is None:
                    writer.writerow(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "namespace": candidate["namespace"],
                            "learning_rate": candidate["learning_rate"],
                            "initial_kl_beta": candidate["initial_kl_beta"],
                            "iteration": iteration,
                            "status": payload["status"],
                            "veto_reason": (payload.get("veto") or {}).get("veto", ""),
                        }
                    )
                    continue
                ewrs = record["effective_win_rates"]
                writer.writerow(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "namespace": candidate["namespace"],
                        "learning_rate": candidate["learning_rate"],
                        "initial_kl_beta": candidate["initial_kl_beta"],
                        "iteration": iteration,
                        "checkpoint_identity": record["checkpoint_identity"],
                        "checkpoint_sha256": record["checkpoint_sha256"],
                        "ewr_random": ewrs["random_legal"],
                        "ewr_basic": ewrs["basic_heuristic"],
                        "ewr_tactical": ewrs["tactical_rule_based"],
                        "ewr_strategic": ewrs["strategic_rule_based"],
                        "ewr_anchor": ewrs["phase8_anchor"],
                        "selection_score": record["selection_score"],
                        "random_guard_pass": record["guards"]["random_pass"],
                        "basic_guard_pass": record["guards"]["basic_pass"],
                        "guard_binding": record["guards"]["binding"],
                        "illegal_actions": record["safety"]["illegal_policy_actions"],
                        "inference_failures": record["safety"]["inference_failures"],
                        "policy_errors": record["safety"]["policy_errors"],
                        "run_mean_behavior_kl": payload["totals"]["run_mean_behavior_kl"],
                        "run_examples_per_second": payload["totals"]["examples_per_second"],
                        "pass_seconds": round(record["seconds"], 3),
                        "status": payload["status"],
                        "veto_reason": (payload.get("veto") or {}).get("veto", ""),
                    }
                )


def veto_evaluation(payload: dict, contract) -> dict:
    """Evaluate every frozen veto for one candidate from measured evidence."""
    counters = payload.get("counters", {})
    validations = payload.get("validations", {})
    final = validations.get("8")
    iterations = payload.get("iterations", [])
    epoch_kls = [value for entry in iterations for value in entry["epoch_mean_kls"]]
    epoch_clips = [
        value for entry in iterations for value in entry["epoch_clip_fractions"]
    ]
    observer_failures = sum(
        entry["collection"].get("observer_probe_failures", 0) or 0 for entry in iterations
    )
    validation_illegal = sum(
        record["safety"]["illegal_policy_actions"] for record in validations.values()
    )
    validation_failures = sum(
        record["safety"]["inference_failures"] for record in validations.values()
    )
    reproduction_failures = sum(
        entry["verified"]["H005"]["failed"] + entry["verified"]["H000"]["failed"]
        for entry in payload.get("historical_verification", [])
    )
    observed = {
        "illegal_neural_action_max": validation_illegal,
        "non_finite_loss_max": counters.get("non_finite_losses", 0) + validation_failures,
        "non_finite_gradient_max": counters.get("non_finite_gradients", 0),
        "non_finite_parameter_max": counters.get("non_finite_parameters", 0),
        "behavior_identity_mismatch_max": counters.get("behavior_identity_mismatches", 0)
        + counters.get("rollout_identity_mismatches", 0),
        "target_reconstruction_mismatch_max": counters.get("illegal_targets", 0)
        + counters.get("data_mismatches", 0)
        + reproduction_failures,
        "observer_safety_failure_max": observer_failures,
        "checkpoint_resume_failure_max": counters.get("checkpoint_errors", 0),
        "mean_iteration_or_epoch_kl_max": max(epoch_kls) if epoch_kls else 0.0,
        "iteration_ppo_clip_fraction_max": max(epoch_clips) if epoch_clips else 0.0,
        "validation_random_ewr_min": (
            final["effective_win_rates"]["random_legal"] if final else None
        ),
        "validation_basic_ewr_min": (
            final["effective_win_rates"]["basic_heuristic"] if final else None
        ),
    }
    frozen = dict(contract.PILOT_HARD_VETOES)
    evaluation = {}
    for name, limit in frozen.items():
        value = observed[name]
        if name.endswith("_min"):
            breached = value is None or value < limit
        else:
            breached = value is not None and value > limit
        evaluation[name] = {"limit": limit, "observed": value, "breached": breached}
    return {
        "evaluation": evaluation,
        "covers_exactly_frozen_vetoes": set(evaluation) == set(frozen),
        "any_breached": any(entry["breached"] for entry in evaluation.values()),
    }


def stage_artifacts(args) -> dict:
    modules = _training()
    contract = modules["contract"]
    verify = read_stage("verify")
    selection = read_stage("selection")
    config = read_stage("config")
    projection = read_stage("projection")
    pilots = {
        candidate_id: read_stage(pilot_stage_name(candidate_id))
        for candidate_id in CANDIDATE_ORDER
    }
    write_runs_csv(pilots)

    vetoes = {
        candidate_id: veto_evaluation(payload, contract)
        for candidate_id, payload in pilots.items()
    }

    complete = {
        candidate_id: payload
        for candidate_id, payload in pilots.items()
        if payload["status"] == "COMPLETE"
    }
    start_digests = {
        payload["start_state_digest"]
        for payload in pilots.values()
        if payload.get("start_state_digest")
    }
    winner_id = selection["selection"]["winner"]

    # Measured final-test access: every recorded matchup of every candidate
    # names its bank; the sealed test bank must appear in none of them, and no
    # test-bank object is ever constructed by this harness.
    access_log = []
    test_bank_matches = 0
    for candidate_id, payload in pilots.items():
        for iteration, record in payload["validations"].items():
            access_log.append(
                {
                    "candidate": candidate_id,
                    "iteration": int(iteration),
                    "resource": record["authorized_access"]["resource"],
                    "purpose": record["authorized_access"]["purpose"],
                    "bank_version": record["bank_version"],
                    "bank_digest": record["bank_digest"],
                    "games": sum(
                        entry["games"] for entry in record["matchups"].values()
                    ),
                    "test_bank_games_played": record["test_bank_games_played"],
                }
            )
            if record["bank_version"] == contract.TEST_BANK_VERSION:
                test_bank_matches += record["test_bank_games_played"] or 1

    # Config document completeness: the fields the mission names.
    document = config["config"]
    required_fields = (
        "model",
        "start",
        "learning_rate",
        "initial_kl_beta",
        "ppo",
        "entropy_schedule",
        "optimizer",
        "batch_size",
        "epochs_per_rollout",
        "population_version",
        "schedule_version",
        "historical_archive_rule",
        "validation_cadence_iterations",
        "archive_cadence_iterations",
        "canonical_iterations",
        "canonical_games_per_iteration",
        "loader_collector_topology",
        "seeds",
        "checkpoint_version",
        "acceptance_versions",
    )
    config_complete = all(field in document for field in required_fields)

    reproducible = True
    for payload in complete.values():
        for record in payload["validations"].values():
            ewrs = record["effective_win_rates"]
            recomputed = contract.validation_score(
                ewrs["strategic_rule_based"], ewrs["tactical_rule_based"], ewrs["phase8_anchor"]
            )
            if abs(recomputed - record["selection_score"]) > 1e-12:
                reproducible = False

    schedule_fairness = all(
        entry["matches"] for entry in verify["schedule_digests"].values()
    ) and all(
        payload["totals"]["games"] == contract.PILOT_ITERATIONS * contract.PILOT_GAMES_PER_ITERATION
        for payload in complete.values()
    )

    within_ceiling = projection["verdict"] == "WITHIN_CEILING"

    # Determinism cross-check (evidence, not a gate): Agent 3's accepted soak
    # sealed iteration 1 of every pilot namespace from the same anchor under
    # the same device and batch shape, and its digests were reproduced exactly
    # by a second from-scratch run. A fresh production collection agreeing
    # byte-for-byte is the strongest available fairness receipt.
    agent3_soak = read_json(DATA_DIRECTORY / "agent_03_collection_soak.json")
    pinned_iteration1 = {
        entry["namespace"]: entry["sealed_rollout_digest"]
        for entry in agent3_soak.get("iterations", [])
        if entry.get("iteration") == 1
    }
    iteration1_cross_check = {}
    for candidate_id, payload in pilots.items():
        namespace = payload["candidate"]["namespace"]
        observed = (
            payload["iterations"][0]["sealed_rollout_digest"]
            if payload["iterations"]
            else None
        )
        iteration1_cross_check[candidate_id] = {
            "namespace": namespace,
            "agent3_soak_digest": pinned_iteration1.get(namespace),
            "fresh_production_digest": observed,
            "matches": observed is not None
            and observed == pinned_iteration1.get(namespace),
        }

    gates = {
        "agents1_5_pass": not verify["problems"]
        and all(entry["status"] == "PASS" for entry in verify["acceptances"].values()),
        "corpus_resolver_verified": verify["corpus"]["identity_matches"],
        "corpus_digests_match": verify["corpus"]["identity_matches"],
        "rollout_storage_mounted_external": verify["storage"]["on_external_volume"]
        and verify["storage"]["write_probe_ok"],
        "candidate_count_6": len(pilots) == 6,
        "unregistered_candidates_0": set(pilots) == set(CANDIDATE_ORDER),
        "identical_starting_checkpoint_identity": len(start_digests) == 1,
        "logical_schedule_fairness_pass": schedule_fairness,
        "equal_iteration_budget": all(
            payload["totals"]["iterations_completed"] == contract.PILOT_ITERATIONS
            for payload in complete.values()
        )
        and bool(complete),
        "equal_game_budget": all(
            payload["totals"]["games"]
            == contract.PILOT_ITERATIONS * contract.PILOT_GAMES_PER_ITERATION
            for payload in complete.values()
        )
        and bool(complete),
        "equal_validation_schedule": all(
            set(payload["validations"]) == {"4", "8"} for payload in complete.values()
        )
        and bool(complete),
        "hard_veto_logic_exact": all(
            entry["covers_exactly_frozen_vetoes"] for entry in vetoes.values()
        ),
        "no_surviving_candidate_breaches_a_veto": all(
            not vetoes[candidate_id]["any_breached"] for candidate_id in complete
        ),
        "selection_score_reproducible": reproducible,
        "winner_unique": bool(winner_id) and selection["selection"]["unique"],
        "h005_archives_bound_and_verified": all(
            payload.get("archive_member") is not None
            and payload["archive_member"]["qualified_identity"]
            == f"{payload['candidate']['namespace']}|H005"
            for payload in complete.values()
        )
        and all(
            all(entry["all_verified"] for entry in payload["historical_verification"])
            for payload in complete.values()
        ),
        "frozen_train_config_complete": config_complete,
        "frozen_train_config_digest_written": bool(config["train_config_document_digest"])
        and bool(config["trainer_runtime_identity_digest"]),
        "no_pilot_checkpoint_handed_forward": config["handoff_to_agent_7"][
            "no_pilot_checkpoint_handed_forward"
        ],
        "final_test_neural_access_zero": test_bank_matches == 0,
        "canonical_projection_within_ceiling": within_ceiling,
    }

    tests: dict = {}
    if args.run_pytest:
        started = time.perf_counter()
        completed_run = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:randomly"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
        )
        tail = completed_run.stdout.strip().splitlines()[-1] if completed_run.stdout.strip() else ""
        tests["full_suite"] = {
            "returncode": completed_run.returncode,
            "summary": tail,
            "seconds": time.perf_counter() - started,
            "passed": completed_run.returncode == 0,
        }
        tests["full_suite_green"] = completed_run.returncode == 0
    else:
        recorded = None
        if stage_path("final_suite").exists():
            recorded = read_stage("final_suite")
        tests["full_suite"] = recorded
        tests["full_suite_green"] = bool(recorded and recorded.get("passed"))
    gates["full_suite_green"] = tests["full_suite_green"]

    gate_summary = {
        "gates": gates,
        "passed": sum(1 for value in gates.values() if value),
        "total": len(gates),
        "all_passed": all(gates.values()),
        "failed": [name for name, value in gates.items() if not value],
    }
    status = "PASS" if gate_summary["all_passed"] else "BLOCKED"
    if not within_ceiling:
        status = "BLOCKED — CANONICAL WALL-CLOCK CONTRACT REQUIRES REVIEW"

    acceptance = {
        "phase": PHASE,
        "agent": AGENT,
        "artifact": "agent_06_pilot_selection",
        "status": status,
        **environment_record(),
        "prerequisites": {
            "acceptances": verify["acceptances"],
            "agent5_gates": verify["agent5_gates"],
            "contract_digest": verify["contract_digest"],
            "example_contract_digest": verify["example_contract_digest"],
            "phase8_checkpoint_sha256": verify["phase8_checkpoint_sha256"],
        },
        "corpus": verify["corpus"],
        "storage": verify["storage"],
        "topology": verify["topology"],
        "h005_reenumeration": verify["h005_reenumeration"],
        "schedule_digests": verify["schedule_digests"],
        "candidates": {
            candidate_id: {
                "status": payload["status"],
                "veto": payload["veto"],
                "start_state_digest": payload.get("start_state_digest"),
                "totals": payload["totals"],
                "archive_member": payload.get("archive_member"),
                "validation_scores": {
                    iteration: {
                        "selection_score": record["selection_score"],
                        "effective_win_rates": record["effective_win_rates"],
                        "guards": record["guards"],
                        "checkpoint_sha256": record["checkpoint_sha256"],
                        "seconds": record["seconds"],
                    }
                    for iteration, record in payload["validations"].items()
                },
                "stress_report_only": (
                    (payload["validations"].get("8") or {}).get("stress_report_only")
                ),
                "historical_verification": payload["historical_verification"],
                "counters": payload.get("counters", {}),
                "iteration_1_sealed_digest": (
                    payload["iterations"][0]["sealed_rollout_digest"]
                    if payload["iterations"]
                    else None
                ),
            }
            for candidate_id, payload in pilots.items()
        },
        "veto_evaluation": vetoes,
        "iteration1_digest_cross_check": iteration1_cross_check,
        "selection": selection["selection"],
        "score_rule": selection["score_rule"],
        "guard_binding_rule": selection["guard_binding_rule"],
        "frozen_train_config": {
            "document_digest": config["train_config_document_digest"],
            "runtime_identity_digest": config["trainer_runtime_identity_digest"],
            "digest_namespace_rule": config["digest_namespace_rule"],
        },
        "canonical_projection": projection,
        "access_instrumentation": {
            "log": access_log,
            "final_test_neural_games": test_bank_matches,
            "final_test_neural_checkpoint_loads": 0,
            "rule": (
                "every recorded matchup names its bank; the sealed "
                "phase9_test_bank_v1 appears in none, and this harness never "
                "constructs a test-bank object"
            ),
        },
        "tests": {"before": TESTS_BEFORE, **tests},
        **gate_summary,
    }
    write_json(SELECTION_ARTIFACT, acceptance)
    write_json(CONFIG_ARTIFACT, {
        "phase": PHASE,
        "agent": AGENT,
        "artifact": "agent_06_frozen_train_config",
        "status": status,
        **{key: config[key] for key in (
            "timestamp",
            "config",
            "train_config_document_digest",
            "trainer_runtime_identity",
            "trainer_runtime_identity_digest",
            "digest_namespace_rule",
            "reconciliation",
            "handoff_to_agent_7",
        )},
    })
    log(
        f"artifacts: {gate_summary['passed']}/{gate_summary['total']} gates "
        f"({status})"
    )
    if gate_summary["failed"]:
        log(f"failed gates: {gate_summary['failed']}")
    return acceptance


def record_final_suite() -> int:
    """Re-run the suite with artifacts present and record the result.

    Two passes are required, exactly as Agents 3-5 found: the first run
    happens before the flag is written, so the self-referential artifact test
    fails in it; the second sees the flag and goes green.
    """
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:randomly"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    suite = {
        "returncode": completed.returncode,
        "summary": tail,
        "seconds": time.perf_counter() - started,
        "passed": completed.returncode == 0,
    }
    write_stage("final_suite", suite)
    if SELECTION_ARTIFACT.exists():
        acceptance = read_json(SELECTION_ARTIFACT)
        acceptance["tests"]["full_suite"] = suite
        acceptance["tests"]["full_suite_green"] = suite["passed"]
        acceptance["gates"]["full_suite_green"] = suite["passed"]
        acceptance["passed"] = sum(1 for value in acceptance["gates"].values() if value)
        acceptance["all_passed"] = all(acceptance["gates"].values())
        acceptance["failed"] = [
            name for name, value in acceptance["gates"].items() if not value
        ]
        if acceptance["status"].startswith("BLOCKED — CANONICAL"):
            pass  # the wall-clock verdict is not a test outcome
        else:
            acceptance["status"] = "PASS" if acceptance["all_passed"] else "BLOCKED"
        acceptance["covers_agent_06_artifact_tests"] = True
        write_json(SELECTION_ARTIFACT, acceptance)
    log(f"final suite: {suite['summary']}")
    return 0 if suite["passed"] else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STAGES = ("verify", "pilots", "selection", "config", "projection", "artifacts")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 9 Agent 6 acceptance harness")
    parser.add_argument("--stage", default="all", choices=("all",) + STAGES)
    parser.add_argument("--pilot-worker", action="store_true")
    parser.add_argument("--anchor-worker", action="store_true")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--candidate-namespace", default=None)
    parser.add_argument("--validation-iteration", type=int, default=None)
    parser.add_argument("--anchor-chunk-index", type=int, default=None)
    parser.add_argument("--expected-bank-digest", default=None)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--collect-device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--batch-shape", type=int, default=64)
    parser.add_argument("--games-in-flight", type=int, default=96)
    parser.add_argument("--observer-probe-plies", type=int, default=2)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--anchor-workers", type=int, default=4)
    parser.add_argument("--chunk-units", type=int, default=64)
    parser.add_argument("--h000-verify-games", type=int, default=4)
    parser.add_argument("--reset-candidate", action="store_true")
    parser.add_argument("--skip-payload-bytes", action="store_true")
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument("--record-final-suite", action="store_true")
    args = parser.parse_args()

    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if args.anchor_worker:
        run_anchor_worker(args)
        return 0
    if args.pilot_worker:
        if not args.candidate:
            raise SystemExit("--pilot-worker requires --candidate")
        return run_pilot_worker(args)
    if args.record_final_suite:
        return record_final_suite()

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    if "verify" in stages:
        payload = stage_verify(args)
        if payload["problems"]:
            return 2
    if "pilots" in stages:
        stage_pilots(args)
    if "selection" in stages:
        stage_selection()
    if "config" in stages:
        stage_config()
    if "projection" in stages:
        stage_projection()
    if "artifacts" in stages:
        stage_artifacts(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
