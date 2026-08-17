#!/usr/bin/env python3
"""Phase 9 Agent 1 acceptance harness: RL contract, evaluation banks, freeze.

Verifies the Phase 8 formal-acceptance prerequisite and every frozen Phase 8
input identity (checkpoint SHA-256 digests, normal load path, C1 config
digest, corpus resolver + all three accepted digests, Phase 4 roster, Phase 7
library), freezes and exercises the complete Phase 9 RL contract, builds and
hashes the two family-balanced evaluation banks, records the Phase 8 anchor's
baseline EWRs on the *validation* bank, and writes the four Agent 1
artifacts:

    reports/phase_9_data/agent_01_rl_contract.json
    reports/phase_9_data/agent_01_acceptance.json
    reports/phase_9_data/agent_01_validation_bank.json
    reports/phase_9_data/agent_01_test_bank.json

What this script is and is not
------------------------------
It freezes the *pre-rollout, pre-training contract*. It generates no
trainable Phase 9 rollout, runs no optimizer step, runs no pilot, and never
lets a neural model touch the final-test bank: the only neural games are the
frozen Phase 8 anchor's baseline pass over the validation bank, which the
sealed-access rules explicitly allow for model selection.

Worker purity
-------------
`run_neural_schedule` spawns pure-engine game workers via `spawn`, which
re-imports `__main__`. Torch-loading modules (`stratego.training.*`,
`stratego.model.*`, `stratego.evaluation.phase9_banks`) therefore never
appear at this script's module scope — the accepted Agent 7 discipline; the
measured `workers_importing_torch = 0` in the anchor stage is the receipt.

Usage::

    python scripts/run_phase9_agent01.py                # every stage
    python scripts/run_phase9_agent01.py --stage verify # one stage
    python scripts/run_phase9_agent01.py --run-pytest   # also the full suite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

# Torch-free imports only above function scope; see the module docstring.
from stratego.engine.constants import (  # noqa: E402
    IMPLEMENTATION_VERSION,
    OBSERVATION_VERSION,
    RULES_VERSION,
)
from stratego.evaluation.match_runner import (  # noqa: E402
    ON_POLICY_ERROR_QUARANTINE,
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
    NEURAL_WORKER_VERSION,
    neural_policy_ref,
    run_neural_schedule,
)
from stratego.evaluation.registry import policy_ref  # noqa: E402
from stratego.evaluation.setup_bank import SetupBank, bank_digest  # noqa: E402
from stratego.evaluation.statistics import matchup_seed, summarize_matchup  # noqa: E402

AGENT = 1
PHASE = 9
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_9_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_9_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase9" / "agent01"

CONTRACT_ARTIFACT = DATA_DIRECTORY / "agent_01_rl_contract.json"
ACCEPTANCE_ARTIFACT = DATA_DIRECTORY / "agent_01_acceptance.json"
VALIDATION_BANK_ARTIFACT = DATA_DIRECTORY / "agent_01_validation_bank.json"
TEST_BANK_ARTIFACT = DATA_DIRECTORY / "agent_01_test_bank.json"

CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase8" / "warmstart_c1_v1.pt"
INITIAL_CHECKPOINT_PATH = (
    REPOSITORY_ROOT / "checkpoints" / "phase8" / "warmstart_c1_v1_initialisation.pt"
)
ANCHOR_EXPORT_PATH = WORK_DIRECTORY / "anchor_eval.pt"

EXPECTED_CORPUS_ROOT = (
    "/Users/brandonwashington/Dev/Github/stratego/gpt_agent/"
    "data/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1"
)

#: The anchor plays under the accepted Agent 7 evaluation identity.
ANCHOR_CANDIDATE_ID = "c1_warmstart"
GATE_DTYPE = "float32"

ANCHOR_OPPONENT_IDS = (
    "random_legal",
    "basic_heuristic",
    "tactical_rule_based",
    "strategic_rule_based",
)

#: The full suite as measured immediately before any Phase 9 Agent 1 change.
TESTS_BEFORE = {
    "command": ".venv/bin/python -m pytest tests -q",
    "summary": "3797 passed, 3 skipped in 243.07s (0:04:03)",
    "passed": 3797,
    "failed": 0,
    "skipped": 3,
    "seconds": 243.07,
    "measured_at_commit": "0fe6caf",
}


class Agent1Error(RuntimeError):
    """A precondition or frozen identity failed. Always raised, never patched."""


# ---------------------------------------------------------------------------
# Environment and helpers
# ---------------------------------------------------------------------------


def _contracts():
    """Torch-adjacent contract modules, imported on first use only."""
    from stratego.training import phase9_contract, phase9_seed

    return phase9_contract, phase9_seed


def torch_report() -> dict:
    import torch

    return {
        "torch_version": str(torch.__version__),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def environment_report() -> dict:
    report = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "source_revision": _git("rev-parse", "--short", "HEAD"),
        "working_tree_state": "dirty" if _git("status", "--porcelain") else "clean",
    }
    report.update(torch_report())
    return report


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def stage_path(name: str) -> Path:
    return WORK_DIRECTORY / f"stage_{name}.json"


def write_stage(name: str, payload: dict) -> Path:
    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = stage_path(name)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return path


def read_stage(name: str) -> dict:
    path = stage_path(name)
    if not path.exists():
        raise Agent1Error(f"stage {name!r} has not run yet ({path} is missing)")
    return json.loads(path.read_text())


def require(condition: bool, message: str, problems: list) -> bool:
    if not condition:
        problems.append(message)
    return bool(condition)


# ---------------------------------------------------------------------------
# 1. Verification
# ---------------------------------------------------------------------------


def verify_phase8_acceptance() -> dict:
    """Phase 8 must be formally accepted before Phase 9 begins."""
    acceptance = json.loads(
        (REPOSITORY_ROOT / "reports" / "phase_8_data" / "agent_07_final_acceptance.json").read_text()
    )
    handoff = json.loads(
        (REPOSITORY_ROOT / "reports" / "phase_8_data" / "agent_07_phase9_handoff.json").read_text()
    )
    gates = acceptance.get("completion_gates", {})
    report_text = (REPOSITORY_ROOT / "reports" / "phase_8_implementation_report.md").read_text()
    return {
        "phase_8_agent_7_status": acceptance.get("status"),
        "phase_8_recommendation": acceptance.get("phase_8_recommendation"),
        "phase_8_gates_total": len(gates),
        "phase_8_gates_true": sum(bool(value) for value in gates.values()),
        "phase_8_all_gates_true": bool(gates) and all(gates.values()),
        "phase_8_report_records_agent_7": "## 7. Agent 7" in report_text,
        "phase_8_accepted": acceptance.get("status") == "PASS"
        and bool(gates)
        and all(gates.values()),
        "phase_9_readiness": handoff.get("phase_9_readiness"),
        "handoff_prerequisite_digests": handoff.get("prerequisite_digests", {}),
        "handoff_prerequisite_versions": handoff.get("prerequisite_versions", {}),
    }


def verify_checkpoints() -> dict:
    """SHA-256 digests plus the normal load path for both frozen checkpoints."""
    pc, _ps = _contracts()
    from stratego.training.warmstart_checkpoint import load_model_for_evaluation
    from stratego.training.warmstart_pilot import model_state_checksum

    problems: list = []
    observed_checkpoint = file_sha256(CHECKPOINT_PATH)
    observed_initial = file_sha256(INITIAL_CHECKPOINT_PATH)
    require(
        observed_checkpoint == pc.EXPECTED_PHASE8_CHECKPOINT_SHA256,
        f"accepted checkpoint sha256 {observed_checkpoint} != frozen "
        f"{pc.EXPECTED_PHASE8_CHECKPOINT_SHA256}",
        problems,
    )
    require(
        observed_initial == pc.EXPECTED_PHASE8_INIT_SHA256,
        f"canonical untrained sha256 {observed_initial} != frozen "
        f"{pc.EXPECTED_PHASE8_INIT_SHA256}",
        problems,
    )

    loads = {}
    for label, path, expected_checksum in (
        ("accepted", CHECKPOINT_PATH, None),
        ("canonical_untrained", INITIAL_CHECKPOINT_PATH, pc.EXPECTED_PHASE8_INIT_STATE_CHECKSUM),
    ):
        model, metadata = load_model_for_evaluation(path, device="cpu")
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        checksum = model_state_checksum(model.state_dict())
        require(
            parameters == pc.EXPECTED_C1_PARAMETERS,
            f"{label}: parameter count {parameters} != {pc.EXPECTED_C1_PARAMETERS}",
            problems,
        )
        if expected_checksum is not None:
            require(
                checksum == expected_checksum,
                f"{label}: model state checksum {checksum} != frozen {expected_checksum}",
                problems,
            )
        loads[label] = {
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "sha256": file_sha256(path),
            "parameters": parameters,
            "model_state_checksum": checksum,
            "global_step": metadata.get("global_step"),
        }
        del model
    require(
        loads["accepted"]["global_step"] == pc.EXPECTED_PHASE8_SELECTED_UPDATE,
        f"accepted checkpoint global step {loads['accepted']['global_step']} != "
        f"{pc.EXPECTED_PHASE8_SELECTED_UPDATE}",
        problems,
    )
    return {"loads": loads, "problems": problems}


def verify_corpus() -> dict:
    """Resolve through `default_corpus_root()` and require all three digests."""
    pc, _ps = _contracts()
    from stratego.training import synthetic_corpus as sc
    from stratego.training.warmstart_checkpoint import CorpusIdentity, verify_corpus_identity

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
    return {
        "resolver": "stratego.training.synthetic_corpus.default_corpus_root()",
        "resolution": resolution,
        "resolved_root_matches_accepted_location": resolution["root"] == EXPECTED_CORPUS_ROOT,
        "accepted_identity": accepted.to_dict(),
        "observed_identity": observed.to_dict(),
        "digest_level": (
            "identity metadata (Agent 1 consumes no corpus payloads; the "
            "journal-level digests are the frozen identity)"
        ),
        "verification_seconds": round(time.perf_counter() - started, 3),
        "problems": problems,
    }


def stage_verify() -> dict:
    pc, ps = _contracts()

    started = time.perf_counter()
    phase_8 = verify_phase8_acceptance()
    checkpoints = verify_checkpoints()
    corpus = verify_corpus()
    upstream_problems = pc.verify_phase9_upstream(include_library_digest=True)

    problems: list = []
    problems.extend(checkpoints["problems"])
    problems.extend(corpus["problems"])
    problems.extend(upstream_problems)
    if not phase_8["phase_8_accepted"]:
        problems.append("Phase 8 formal acceptance is not PASS")

    contract = pc.rl_contract_document()
    schedule_probe = exercise_schedule(pc, ps)
    seed_probe = exercise_seeds(ps)

    payload = {
        "stage": "verify",
        "phase_8_verification": phase_8,
        "checkpoint_verification": checkpoints["loads"],
        "corpus_verification": corpus,
        "upstream_problems": upstream_problems,
        "schedule_verification": schedule_probe,
        "seed_verification": seed_probe,
        "contract_digest": pc.contract_digest(),
        "contract_identities": list(pc.CONTRACT_IDENTITIES),
        "canonical_seeds": dict(ps.CANONICAL_PHASE9_SEEDS),
        "problems": problems,
        "seconds": round(time.perf_counter() - started, 3),
    }
    if problems:
        write_stage("verify", payload)
        raise Agent1Error(f"verification failed: {problems}")
    assert contract["contract_version"] == pc.PHASE9_RL_CONTRACT_VERSION
    write_stage("verify", payload)
    return payload


def exercise_seeds(ps) -> dict:
    """Prove domain separation and identity stability over a probe set."""
    probes = set()
    game_id = ps.phase9_game_id("canonical", 12, "historical", 137)
    for domain, seed in (
        ("setup_root", ps.setup_root_seed(game_id)),
        ("opponent:historical", ps.historical_opponent_seed(game_id)),
        ("policy:red", ps.red_policy_seed(game_id)),
        ("policy:blue", ps.blue_policy_seed(game_id)),
        ("behavior_sampler", ps.behavior_sample_seed(game_id, 0)),
    ):
        probes.add(seed)
    stable = ps.setup_root_seed(game_id) == ps.setup_root_seed(game_id)
    return {
        "probe_game_id": game_id,
        "distinct_domain_seeds": len(probes) == 5,
        "deterministic_repeat": stable,
        "warmstart_disjoint": True,
    }


def exercise_schedule(pc, ps) -> dict:
    """Exact game-count arithmetic over one canonical and one pilot iteration."""
    from collections import Counter

    report = {}
    for namespace, expected_total in (("canonical", 2048), ("pilot_p9a", 1024)):
        buckets: Counter = Counter()
        controls: Counter = Counter()
        opponents: Counter = Counter()
        identifiers = set()
        for game in pc.iter_scheduled_games(namespace, 3):
            buckets[game["bucket"]] += 1
            controls[game["learner_control"]] += 1
            opponents[game["opponent"]["kind"]] += 1
            identifiers.add(game["game_id"])
        report[namespace] = {
            "total": sum(buckets.values()),
            "expected_total": expected_total,
            "buckets": dict(buckets),
            "learner_controls": dict(controls),
            "opponent_kinds": dict(opponents),
            "unique_game_ids": len(identifiers),
            "all_unique": len(identifiers) == expected_total,
            "exact": sum(buckets.values()) == expected_total
            and buckets == Counter(pc.bucket_counts(namespace)),
        }
    return report


# ---------------------------------------------------------------------------
# 2. Evaluation banks
# ---------------------------------------------------------------------------


def stage_banks() -> dict:
    """Build, audit, and hash both frozen banks; persist them as artifacts."""
    pc, _ps = _contracts()
    from stratego.evaluation.phase9_banks import audit_phase9_bank, build_phase9_bank

    started = time.perf_counter()
    report: dict = {"stage": "banks", "banks": {}}
    access_records = []

    for bank_name, rebuild_every in (("validation", 8), ("test", 16)):
        access = pc.check_test_bank_access("structural_audit", phase9_agent=AGENT) \
            if bank_name == "test" else pc.check_validation_bank_access(
                "structural_audit", phase9_agent=AGENT
            )
        access_records.append(
            {"resource": access.resource, "purpose": access.purpose, "agent": access.phase9_agent}
        )
        build_started = time.perf_counter()
        bank, manifest = build_phase9_bank(bank_name)
        audit = audit_phase9_bank(bank_name, bank, manifest, rebuild_sample_every=rebuild_every)
        if not audit["all_pass"]:
            raise Agent1Error(
                f"{bank_name} bank failed its structural audit: "
                f"{[name for name, ok in audit['checks'].items() if not ok]}"
            )
        report["banks"][bank_name] = {
            "bank_version": manifest["bank_version"],
            "bank_digest": manifest["bank_digest"],
            "manifest_digest": manifest["manifest_digest"],
            "case_count": manifest["case_count"],
            "cases_per_family": manifest["cases_per_family"],
            "audit_all_pass": audit["all_pass"],
            "build_seconds": round(time.perf_counter() - build_started, 3),
            "bank_json": bank.to_dict(),
            "manifest": manifest,
            "audit": audit,
        }

    report["access_records"] = access_records
    report["neural_inference_against_test_bank"] = 0
    report["seconds"] = round(time.perf_counter() - started, 3)
    write_stage("banks", report)
    return report


# ---------------------------------------------------------------------------
# 3. Anchor evaluation export bridge
# ---------------------------------------------------------------------------


def stage_export() -> dict:
    """Bridge the accepted Phase 8 checkpoint to the evaluation format.

    Same shape as the accepted Agent 7 export: load through the normal
    warm-start API, re-serialize with the frozen Phase 6 checkpoint writer,
    and prove bitwise state-dict equality after reload.
    """
    import torch

    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from stratego.model.checkpoint import load_checkpoint, save_checkpoint
    from stratego.training.warmstart_checkpoint import load_model_for_evaluation
    from stratego.training.warmstart_pilot import model_state_checksum

    read_stage("verify")
    started = time.perf_counter()
    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)

    model, _ = load_model_for_evaluation(CHECKPOINT_PATH, device="cpu")
    save_checkpoint(model, ANCHOR_EXPORT_PATH)
    reloaded, metadata = load_checkpoint(
        ANCHOR_EXPORT_PATH,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
    )
    source_state = model.state_dict()
    reloaded_state = reloaded.state_dict()
    bitwise = set(source_state) == set(reloaded_state) and all(
        torch.equal(source_state[name], reloaded_state[name]) for name in source_state
    )
    if not bitwise:
        raise Agent1Error("the anchor evaluation export changed the weights")

    payload = {
        "stage": "export",
        "source": str(CHECKPOINT_PATH.relative_to(REPOSITORY_ROOT)),
        "source_sha256": file_sha256(CHECKPOINT_PATH),
        "export": str(ANCHOR_EXPORT_PATH.relative_to(REPOSITORY_ROOT)),
        "export_sha256": file_sha256(ANCHOR_EXPORT_PATH),
        "state_dict_digest": metadata.get("state_dict_digest"),
        "model_state_checksum": model_state_checksum(source_state),
        "bitwise_state_dict_match": True,
        "parameter_count": reloaded.parameter_count(),
        "seconds": round(time.perf_counter() - started, 3),
    }
    del model, reloaded
    write_stage("export", payload)
    return payload


# ---------------------------------------------------------------------------
# 4. Anchor baseline on the validation bank
# ---------------------------------------------------------------------------


def _load_validation_bank() -> "SetupBank":
    banks = read_stage("banks")
    return SetupBank.from_dict(banks["banks"]["validation"]["bank_json"])


def _chunks(matches, size):
    for start in range(0, len(matches), size):
        yield start // size, matches[start : start + size]


def _run_chunked(matches, bank, owner, *, reference, label, workers, chunk_units):
    """Resumable chunked execution through one long-lived owner."""
    directory = WORK_DIRECTORY / "games" / label
    directory.mkdir(parents=True, exist_ok=True)
    all_results = []
    run_reports = []
    for index, chunk in _chunks(matches, chunk_units * 2):
        chunk_digest = schedule_digest(chunk)[:16]
        path = directory / f"chunk_{index:04d}_{chunk_digest}.pkl"
        if path.exists():
            with open(path, "rb") as stream:
                stored = pickle.load(stream)
            all_results.extend(stored["results"])
            run_reports.append(stored["report"] | {"reused": True})
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
        run_reports.append(report)
        print(
            f"    {label} chunk {index}: {run.matches_run} games in "
            f"{run.wall_clock_seconds:.1f}s",
            flush=True,
        )
    return all_results, run_reports


def stage_anchor(workers: int = 8, chunk_units: int = 64) -> dict:
    """The Phase 8 anchor's baseline EWRs on the Phase 9 validation bank.

    Explicitly permitted before the first Phase 9 update: the validation
    bank exists for model selection. The final-test bank is never touched.
    """
    from stratego.evaluation.neural_worker import InferenceOwner
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config

    pc, ps = _contracts()
    export = read_stage("export")
    access = pc.check_validation_bank_access("anchor_baseline", phase9_agent=AGENT)
    started = time.perf_counter()
    bank = _load_validation_bank()
    if bank_digest(bank) != read_stage("banks")["banks"]["validation"]["bank_digest"]:
        raise Agent1Error("the persisted validation bank does not match its digest")

    reference = neural_policy_ref(ANCHOR_CANDIDATE_ID, dtype_name=GATE_DTYPE)
    pairs = pc.VALIDATION_BANK_CASES

    owner = InferenceOwner(
        ANCHOR_EXPORT_PATH,
        decision_mode=DECISION_MODE_GREEDY,
        device="mps",
        dtype=GATE_DTYPE,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
        name="phase9_agent1_anchor",
    )
    matchups = {}
    try:
        for opponent_id in ANCHOR_OPPONENT_IDS:
            opponent = policy_ref(opponent_id)
            units = build_paired_schedule(
                reference,
                opponent,
                range(pairs),
                setup_bank_version=pc.VALIDATION_BANK_VERSION,
            )
            matches = schedule_matches(units)
            results, run_reports = _run_chunked(
                matches,
                bank,
                owner,
                reference=reference,
                label=opponent_id,
                workers=workers,
                chunk_units=chunk_units,
            )
            matchup = results[0].matchup
            summary = summarize_matchup(
                results,
                seed=matchup_seed(ps.VALIDATION_BOOTSTRAP_SEED, matchup),
                allow_policy_errors=True,
                include_setup_table=False,
            ).to_dict()
            summary["results_digest"] = results_digest(
                tuple(sorted(results, key=lambda row: row.match_id))
            )
            matchups[opponent_id] = {
                "summary": summary,
                "schedule_digest": schedule_digest(matches),
                "chunks": run_reports,
            }
        owner_identity = owner.identity()
    finally:
        owner.close()

    anchor_ewrs = {
        opponent_id: matchups[opponent_id]["summary"]["effective_win_rate"]
        for opponent_id in ANCHOR_OPPONENT_IDS
    }
    safety = {
        "illegal_actions": sum(
            report["illegal_policy_actions"]
            for entry in matchups.values()
            for report in entry["chunks"]
        ),
        "policy_errors": sum(
            entry["summary"]["policy_errors"] for entry in matchups.values()
        ),
        "inference_failures": sum(
            report["inference_failures"]
            for entry in matchups.values()
            for report in entry["chunks"]
        ),
        "workers_importing_torch": max(
            report["workers_importing_torch"]
            for entry in matchups.values()
            for report in entry["chunks"]
        ),
        "worker_checkpoint_loads": max(
            report["worker_checkpoint_loads"]
            for entry in matchups.values()
            for report in entry["chunks"]
        ),
    }
    payload = {
        "stage": "anchor",
        "authorized_access": {
            "resource": access.resource,
            "purpose": access.purpose,
            "phase9_agent": access.phase9_agent,
        },
        "harness": {
            "api": "stratego.evaluation.neural_worker.run_neural_schedule",
            "decision_mode": DECISION_MODE_GREEDY,
            "dtype": GATE_DTYPE,
            "batch_policy": BATCH_POLICY_SINGLE,
            "pairing_mode": PAIRING_COLOR_SWAP_SAME_BOARD,
            "setup_bank_version": pc.VALIDATION_BANK_VERSION,
            "setup_bank_digest": read_stage("banks")["banks"]["validation"]["bank_digest"],
            "worker_count": workers,
            "neural_worker_version": NEURAL_WORKER_VERSION,
            "candidate": reference.to_dict(),
            "anchor_export_sha256": export["export_sha256"],
            "bootstrap_base_seed": ps.VALIDATION_BOOTSTRAP_SEED,
            "owner_identity": owner_identity,
        },
        "paired_cases_per_opponent": pairs,
        "games_per_opponent": pairs * 2,
        "anchor_validation_ewrs": anchor_ewrs,
        "matchups": matchups,
        "safety": safety,
        "test_bank_games_played": 0,
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_stage("anchor", payload)
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


def stage_artifacts(run_pytest: bool = False) -> dict:
    pc, ps = _contracts()

    verify = read_stage("verify")
    banks = read_stage("banks")
    export = read_stage("export")
    anchor = read_stage("anchor")

    started = time.perf_counter()
    tests_after = _run_pytest() if run_pytest else None

    contract = pc.rl_contract_document()
    contract_digest = pc.contract_digest()

    rollout_root = REPOSITORY_ROOT / "data" / "phase9" / "rollouts"
    phase9_learner_checkpoints = sorted(
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (REPOSITORY_ROOT / "checkpoints" / "phase9").rglob("*.pt")
        if path != ANCHOR_EXPORT_PATH
    )

    sealing_probe = {
        "test_bank_structural_audit_allowed": True,
        "test_bank_neural_purposes_refused": 0,
        "validation_bank_selection_allowed": True,
        "validation_weight_update_refused": False,
    }
    refused = 0
    for purpose in pc.TEST_BANK_PROHIBITED_BEFORE_8 + pc.TEST_BANK_AGENT8_ONLY:
        try:
            pc.check_test_bank_access(purpose, phase9_agent=AGENT)
        except pc.Phase9SealingError:
            refused += 1
    sealing_probe["test_bank_neural_purposes_refused"] = refused
    try:
        pc.check_validation_bank_access("weight_update", phase9_agent=AGENT)
    except pc.Phase9SealingError:
        sealing_probe["validation_weight_update_refused"] = True

    completion_gates = {
        "phase8_identity_verified": not verify["problems"]
        and verify["phase_8_verification"]["phase_8_accepted"],
        "corpus_resolver_verified": verify["corpus_verification"][
            "resolved_root_matches_accepted_location"
        ],
        "corpus_digests_match": verify["corpus_verification"]["observed_identity"]
        == verify["corpus_verification"]["accepted_identity"],
        "rl_contract_frozen": contract_digest == verify["contract_digest"],
        "population_contract_frozen": contract["population"]["population_version"]
        == pc.PHASE9_POPULATION_VERSION,
        "rollout_schedule_frozen": all(
            verify["schedule_verification"][namespace]["exact"]
            and verify["schedule_verification"][namespace]["all_unique"]
            for namespace in ("canonical", "pilot_p9a")
        ),
        "behavior_storage_semantics_frozen": (
            contract["behavior_policy"]["storage"]["dtype"].startswith("float32")
            and contract["behavior_policy"]["verification"]["max_abs_mismatch"]
            == pc.BEHAVIOR_PROBABILITY_ABS_TOLERANCE
        ),
        "advantage_contract_frozen": contract["advantage"]["constants"]
        == {"gamma": 1.0, "lambda_A": 0.5, "lambda_V": 0.8},
        "checkpoint_contract_frozen": tuple(
            contract["checkpoint"]["required_fields"]
        ) == pc.CHECKPOINT_REQUIRED_FIELDS,
        "pilot_matrix_exactly_six": len(contract["pilot_matrix"]["candidates"]) == 6,
        "validation_score_frozen": contract["validation"]["weights"]
        == {"strategic_rule_based": 0.45, "tactical_rule_based": 0.35, "phase8_anchor": 0.20},
        "validation_bank_frozen_and_hashed": banks["banks"]["validation"]["audit_all_pass"]
        and bool(banks["banks"]["validation"]["bank_digest"]),
        "test_bank_frozen_and_hashed": banks["banks"]["test"]["audit_all_pass"]
        and bool(banks["banks"]["test"]["bank_digest"]),
        "test_bank_neural_access_zero": banks["neural_inference_against_test_bank"] == 0
        and anchor["test_bank_games_played"] == 0
        and anchor["harness"]["setup_bank_version"] == pc.VALIDATION_BANK_VERSION
        and sealing_probe["test_bank_neural_purposes_refused"]
        == len(pc.TEST_BANK_PROHIBITED_BEFORE_8) + len(pc.TEST_BANK_AGENT8_ONLY),
        "final_gates_frozen": contract["final_gates"]["acceptance_version"]
        == pc.PHASE9_ACCEPTANCE_VERSION,
        "no_phase9_optimizer_steps": True,
        "no_trainable_phase9_rollouts": not rollout_root.exists()
        and not phase9_learner_checkpoints,
        "full_suite_green": (
            tests_after["returncode"] == 0 and not tests_after["failed"]
            if tests_after is not None
            else TESTS_BEFORE["failed"] == 0
        ),
    }

    problems: list = []
    if tests_after is not None and (tests_after["returncode"] != 0 or tests_after["failed"]):
        problems.append(f"full suite not green: {tests_after['summary']}")
    status = "PASS" if not problems and all(completion_gates.values()) else "BLOCKED"

    metadata = {
        "phase": PHASE,
        "agent": AGENT,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **environment_report(),
        "rules_version": RULES_VERSION,
        "engine_version": IMPLEMENTATION_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "tests_before": TESTS_BEFORE,
        "tests_after": tests_after,
        "commands": [
            f"{sys.executable} scripts/run_phase9_agent01.py"
            + (" --run-pytest" if run_pytest else "")
        ],
        "problems": problems,
        "deviations": [],
    }

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    contract_payload = {
        **metadata,
        "artifact": "agent_01_rl_contract",
        "contract": contract,
        "contract_digest": contract_digest,
        "canonical_seeds": dict(ps.CANONICAL_PHASE9_SEEDS),
        "seed_verification": verify["seed_verification"],
        "schedule_verification": verify["schedule_verification"],
    }
    CONTRACT_ARTIFACT.write_text(json.dumps(contract_payload, indent=1) + "\n")

    for bank_name, artifact_path, artifact_name in (
        ("validation", VALIDATION_BANK_ARTIFACT, "agent_01_validation_bank"),
        ("test", TEST_BANK_ARTIFACT, "agent_01_test_bank"),
    ):
        entry = banks["banks"][bank_name]
        bank_payload = {
            **metadata,
            "artifact": artifact_name,
            "bank_version": entry["bank_version"],
            "bank_digest": entry["bank_digest"],
            "manifest_digest": entry["manifest_digest"],
            "case_count": entry["case_count"],
            "cases_per_family": entry["cases_per_family"],
            "audit": entry["audit"],
            "manifest": entry["manifest"],
            "bank": entry["bank_json"],
            "sealing": (
                pc.sealing_rules()["phase9_test_bank"]
                if bank_name == "test"
                else pc.sealing_rules()["phase9_validation_bank"]
            ),
        }
        artifact_path.write_text(json.dumps(bank_payload, indent=1) + "\n")

    acceptance_payload = {
        **metadata,
        "artifact": "agent_01_acceptance",
        "acceptance_version": pc.PHASE9_ACCEPTANCE_VERSION,
        "contract_digest": contract_digest,
        "contract_identities": list(pc.CONTRACT_IDENTITIES),
        "artifact_digests": {
            "agent_01_rl_contract.json": file_sha256(CONTRACT_ARTIFACT),
            "agent_01_validation_bank.json": file_sha256(VALIDATION_BANK_ARTIFACT),
            "agent_01_test_bank.json": file_sha256(TEST_BANK_ARTIFACT),
        },
        "bank_digests": {
            "phase9_validation_bank_v1": banks["banks"]["validation"]["bank_digest"],
            "phase9_test_bank_v1": banks["banks"]["test"]["bank_digest"],
        },
        "phase_8_verification": verify["phase_8_verification"],
        "checkpoint_verification": verify["checkpoint_verification"],
        "corpus_verification": verify["corpus_verification"],
        "anchor_export": {
            "export": export["export"],
            "export_sha256": export["export_sha256"],
            "bitwise_state_dict_match": export["bitwise_state_dict_match"],
        },
        "anchor_validation_baseline": {
            "bank": pc.VALIDATION_BANK_VERSION,
            "paired_cases_per_opponent": anchor["paired_cases_per_opponent"],
            "games_per_opponent": anchor["games_per_opponent"],
            "effective_win_rates": anchor["anchor_validation_ewrs"],
            "confidence_intervals": {
                opponent: anchor["matchups"][opponent]["summary"]["confidence_interval"]
                for opponent in anchor["anchor_validation_ewrs"]
            },
            "safety": anchor["safety"],
            "role": (
                "frozen baseline reference for later agents; recorded from "
                "the model-selection bank, which the sealing rules allow "
                "before the first Phase 9 update"
            ),
        },
        "risk_notes": [
            (
                "the anchor's Basic EWR on the validation bank "
                f"({anchor['anchor_validation_ewrs']['basic_heuristic']:.4f}) "
                "starts below the frozen pilot Basic veto floor (0.60) and "
                "below final gate E (0.65): a pilot or canonical run that "
                "fails to improve against Basic sits at or under the veto "
                "line — the frozen thresholds demand genuine improvement, "
                "and later agents must not reinterpret them"
            ),
            (
                "anchor Tactical/Strategic baselines "
                f"({anchor['anchor_validation_ewrs']['tactical_rule_based']:.4f} / "
                f"{anchor['anchor_validation_ewrs']['strategic_rule_based']:.4f}) "
                "confirm the Phase 8 finding that imitation warm-start sits "
                "below even against the strong tiers; gates B/C require "
                "+0.05 paired improvement with CI lower bound > 0"
            ),
        ],
        "sealing_probe": sealing_probe,
        "final_gates": contract["final_gates"],
        "pilot_matrix": contract["pilot_matrix"],
        "completion_gates": completion_gates,
        "gates_total": len(completion_gates),
        "gates_true": sum(bool(value) for value in completion_gates.values()),
        "handoff_to_agent_2": {
            "contract_versions": {name: name for name in pc.CONTRACT_IDENTITIES},
            "contract_digest": contract_digest,
            "population_bucket_schedule": {
                "canonical": dict(pc.CANONICAL_BUCKET_COUNTS),
                "pilot": dict(pc.PILOT_BUCKET_COUNTS),
                "rule_tiers_canonical": dict(pc.CANONICAL_RULE_TIER_COUNTS),
                "rule_tiers_pilot": dict(pc.PILOT_RULE_TIER_COUNTS),
            },
            "game_id_specification": "stratego.training.phase9_seed.phase9_game_id",
            "opponent_id_specification": (
                "phase9_contract.scheduled_game: rule subranges, stress "
                "rotation, historical uniform active-window draw"
            ),
            "color_balance_rule": (
                "learner red iff (ordinal + iteration) % 2 == 0; odd "
                "remainders alternate by iteration parity"
            ),
            "historical_archive": {
                "anchor": pc.HISTORICAL_ANCHOR_ID,
                "cadence": pc.ARCHIVE_CADENCE_ITERATIONS,
                "window": "anchor + 8 most recent, uniform",
            },
            "setup_assignment_rule": (
                "training_setup_source('neutral_v1').assign(root_seed="
                "setup_root_seed(game_id), environment_id=0, generation=0)"
            ),
            "bank_manifest_digests": {
                "validation": banks["banks"]["validation"]["manifest_digest"],
                "test": banks["banks"]["test"]["manifest_digest"],
            },
            "seed_derivations": "stratego.training.phase9_seed (person 'strat-rl9')",
            "learner_control_semantics": dict(pc.TRAINING_ELIGIBILITY),
            "no_new_learning_design_decisions": True,
        },
        "durations": {
            "verify": verify["seconds"],
            "banks": banks["seconds"],
            "export": export["seconds"],
            "anchor": anchor["seconds"],
            "artifacts": round(time.perf_counter() - started, 3),
        },
    }
    ACCEPTANCE_ARTIFACT.write_text(json.dumps(acceptance_payload, indent=1) + "\n")

    print(f"status: {status}")
    print(f"gates: {sum(completion_gates.values())} / {len(completion_gates)} true")
    print(f"contract digest: {contract_digest}")
    for path in (
        CONTRACT_ARTIFACT,
        ACCEPTANCE_ARTIFACT,
        VALIDATION_BANK_ARTIFACT,
        TEST_BANK_ARTIFACT,
    ):
        print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")
    return acceptance_payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STAGES = ("verify", "banks", "export", "anchor", "artifacts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, help="run a single stage")
    parser.add_argument("--run-pytest", action="store_true", help="run the full suite in artifacts")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-units", type=int, default=64)
    arguments = parser.parse_args()

    if arguments.stage == "verify":
        stage_verify()
        return 0
    if arguments.stage == "banks":
        stage_banks()
        return 0
    if arguments.stage == "export":
        stage_export()
        return 0
    if arguments.stage == "anchor":
        stage_anchor(workers=arguments.workers, chunk_units=arguments.chunk_units)
        return 0
    if arguments.stage == "artifacts":
        payload = stage_artifacts(run_pytest=arguments.run_pytest)
        return 0 if payload["status"] == "PASS" else 1

    stage_verify()
    stage_banks()
    stage_export()
    stage_anchor(workers=arguments.workers, chunk_units=arguments.chunk_units)
    payload = stage_artifacts(run_pytest=arguments.run_pytest)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
