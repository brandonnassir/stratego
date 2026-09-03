#!/usr/bin/env python3
"""Phase 18 Gate G3: the bounded matched two-lineage joint-training pilot.

Stages, in order; each must finish before the next may start:

* `--freeze`            write the frozen pilot contract: both lineage configurations
                        (identical except the lineage switch), every seed, the
                        evaluation case set and schedule digest, the decision rule.
                        No game, no model, no pool.
* `--verify`            run the Stage 6B test files with a JUnit record and the
                        design-section-6 restart check; write the verification record.
* `--launch-manifest`   from the clean execution worktree: bind the source commit, its
                        tree, every harness source and test digest, the contract digest,
                        and the runtime root. Refuses a non-empty runtime root.
* `--run --lineage L`   run one lineage to the bounded horizon (256 periods), writing a
                        joint bundle every 32 periods; `--resume` continues from the
                        lineage's last bundle. Nothing continues past the horizon.
* `--check-matching`    the period-1 lineage-identity check and the equal-budget check.
* `--evaluate --arm A`  play the frozen evaluation schedule with one final bundle on
                        the established G1 harness; one immutable receipt per game.
* `--analyse`           the candidate-versus-control contrast, the stratified cluster
                        bootstrap, the frozen rule, the ten gates.
* `--restart-check`     the section-6 restart test on the smoke configuration, on demand.

Frozen by the reviewer (2026-09-02): fresh initialisation of both lineages;
K = 64 C1 updates per period; canonical:live 1:1; live retention 32 periods;
256 periods; bundle cadence 32; the period-128 bundle is SAVED but not
evaluated unless a diagnosis needs it (`--diagnostic`); validation bases
410..449 stay reserved; a conditional second seed is never pooled to rescue a
failed primary; nothing here authorises the pilot itself.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from stratego.training.phase18 import PHASE18_SETUP_PACKAGE_VERSION  # noqa: E402
from stratego.training.phase18.g3_contract import (  # noqa: E402
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_REPLICATES,
    EVALUATION_BASE_INDICES,
    G3_DESIGN_COMMIT,
    G3_EVALUATION_VERSION,
    G3_HARNESS_VERSION,
    HANDCRAFTED_OPPONENTS,
    LINEAGES,
    LINEAGE_CANDIDATE,
    LINEAGE_CONTROL,
    PILOT_PERIODS,
    PRIMARY_MARGIN,
    RESERVED_BASE_INDICES,
    Phase18G3Error,
    PilotConfig,
    evaluation_bootstrap_seed,
    evaluation_schedule_seed,
)
from stratego.training.phase18.setup_contract import file_sha256, json_document_digest  # noqa: E402
from stratego.training.warmstart_trainer import WarmstartTrainConfig  # noqa: E402

RUN_ID = "G3-PILOT-2026-A"
GATE = "G3"
AGENT = "phase_18_agent_6"
WORK_PACKAGE = "phase18_setup_integrated_warmstart"
#: The seed namespace binds every stream to the approved design commit (G2 lesson).
NAMESPACE = f"phase18_g3_pilot_v1:{G3_DESIGN_COMMIT}"
WINNING_CANDIDATE_ID = "ws_pilot_lr1e-3_balanced"

#: Storage policy (reports/phase18/phase18_execution_storage_policy_v1.md).
CANONICAL_ROOT = Path("/Users/brandonwashington/Dev/Github/stratego/gpt_agent")
RUNTIME_RELATIVE = Path("output/phase18/runtime/g3_pilot_v1")
RUNTIME_ROOT = CANONICAL_ROOT / RUNTIME_RELATIVE
REPORTS_RELATIVE = Path("reports/phase18/g3_pilot")

CONTRACT_NAME = "phase18_g3_pilot_contract_v1.json"
VERIFICATION_NAME = "phase18_g3_stage6b_verification_v1.json"
LAUNCH_NAME = "phase18_g3_pilot_launch_manifest_v1.json"
MATCHING_NAME = "phase18_g3_pilot_matching_v1.json"
RESULTS_NAME = "phase18_g3_pilot_results_v1.json"

AUTHORIZATION_FILES = (
    "reports/phase18/g3_design/phase18_g3_stage6a_joint_design_v2.md",
    "reports/phase18/g3_design/phase18_g3_joint_design_tables_v2.json",
    "reports/phase18/amendments/P18-A001_EVALUATION_RULE_IDENTITY.json",
    "reports/phase18/phase18_execution_storage_policy_v1.md",
)

HARNESS_SOURCES = (
    "scripts/phase18_g3_pilot.py",
    "stratego/training/phase18/__init__.py",
    "stratego/training/phase18/g3_contract.py",
    "stratego/training/phase18/g3_buffer_state.py",
    "stratego/training/phase18/g3_live_store.py",
    "stratego/training/phase18/g3_collector.py",
    "stratego/training/phase18/g3_c1.py",
    "stratego/training/phase18/g3_bundle.py",
    "stratego/training/phase18/g3_pilot.py",
    "stratego/training/phase18/g3_smoke.py",
    "stratego/training/phase18/g3_evaluation.py",
    "stratego/training/phase18/setup_contract.py",
    "stratego/training/phase18/setup_model.py",
    "stratego/training/phase18/setup_sampling.py",
    "stratego/training/phase18/setup_buffer.py",
    "stratego/training/phase18/setup_learning.py",
    "stratego/training/warmstart_trainer.py",
    "stratego/training/warmstart_dataset.py",
    "stratego/training/warmstart_examples.py",
    "stratego/training/warmstart_checkpoint.py",
    "stratego/training/rule_population.py",
    "stratego/training/trajectory.py",
    "stratego/training/phase17/checkpoint.py",
    "stratego/evaluation/match_runner.py",
    "stratego/evaluation/match_spec.py",
    "stratego/evaluation/neural_worker.py",
    "stratego/evaluation/setup_bank.py",
    "stratego/setups/identity.py",
    "stratego/engine/constants.py",
)

TEST_FILES = (
    "tests/training/phase18/test_g3_buffer_state.py",
    "tests/training/phase18/test_g3_live_store.py",
    "tests/training/phase18/test_g3_collector.py",
    "tests/training/phase18/test_g3_c1_mixture.py",
    "tests/training/phase18/test_g3_pilot.py",
    "tests/training/phase18/test_g3_driver.py",
    "tests/evaluation/phase18/test_g3_evaluation.py",
)

TEST_TARGETS = TEST_FILES + ("tests/training/phase18/test_setup_learning.py", "tests/training/phase18/test_setup_buffer.py")


class G3Error(RuntimeError):
    """A frozen identity, accounting or sealing precondition failed."""


def log(message: str) -> None:
    print(f"[g3 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    return path


def git_output(*arguments: str, cwd: Path = REPOSITORY_ROOT) -> str:
    return subprocess.run(["git", *arguments], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def environment() -> dict:
    import torch

    porcelain = git_output("status", "--porcelain")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "executing_tree": str(REPOSITORY_ROOT),
        "source_commit": git_output("rev-parse", "HEAD"),
        "source_tree": git_output("rev-parse", "HEAD^{tree}"),
        "working_tree_state": "clean" if not porcelain else f"dirty ({len(porcelain.splitlines())} paths)",
        "dirty_paths": porcelain.splitlines()[:40],
    }


def digests_of(names, root: Path = REPOSITORY_ROOT) -> dict:
    record = {}
    for name in names:
        path = root / name
        if not path.exists():
            raise G3Error(f"BLOCKED: {name} is missing from {root}")
        record[name] = file_sha256(path)
    return record


# ---------------------------------------------------------------------------
# The production configuration
# ---------------------------------------------------------------------------


def production_config(lineage: str, *, c1_device: str = "mps", threads: int = 4, loader_workers: int = 12, loader_prefetch: int = 2) -> PilotConfig:
    """The frozen defaults; only the device and loader topology are launch options."""
    c1 = WarmstartTrainConfig.from_pilot_candidate(WINNING_CANDIDATE_ID, device=c1_device, validation_batches=64)
    return PilotConfig(
        run_id=RUN_ID,
        namespace=NAMESPACE,
        seed_index=1,
        lineage=lineage,
        c1_train_config=c1,
        threads=int(threads),
        loader_workers=int(loader_workers),
        loader_prefetch=int(loader_prefetch),
    )


# ---------------------------------------------------------------------------
# Stage 1: freeze
# ---------------------------------------------------------------------------


def stage_freeze(reports: Path, *, c1_device: str) -> dict:
    from stratego.training.phase18.g3_evaluation import build_cases, build_schedule, load_evaluation_bases, schedule_record

    started = time.perf_counter()
    configs = {lineage: production_config(lineage, c1_device=c1_device) for lineage in LINEAGES}
    if configs[LINEAGE_CANDIDATE].matched_digest() != configs[LINEAGE_CONTROL].matched_digest():
        raise G3Error("BLOCKED: the two lineage configurations are not matched")
    if not configs[LINEAGE_CANDIDATE].is_production_scale():
        raise G3Error("BLOCKED: the frozen configuration is not at production scale")
    bases = load_evaluation_bases()
    cases = build_cases(bases)
    matches = build_schedule(cases, namespace=NAMESPACE)
    schedule = schedule_record(cases, matches, namespace=NAMESPACE)
    candidate = configs[LINEAGE_CANDIDATE]
    contract = {
        "artifact": "phase18_g3_pilot_contract_v1",
        "work_package": WORK_PACKAGE,
        "agent": AGENT,
        "gate": GATE,
        "run_id": RUN_ID,
        "harness_version": G3_HARNESS_VERSION,
        "evaluation_version": G3_EVALUATION_VERSION,
        "package_version": PHASE18_SETUP_PACKAGE_VERSION,
        "design_commit": G3_DESIGN_COMMIT,
        "timestamp_utc": utc_now(),
        "authorization": digests_of(AUTHORIZATION_FILES),
        "status": "CONTRACT FROZEN; the pilot itself is NOT authorised by this file",
        "question": {
            "text": (
                "Does enabling setup learning together with the policy co-adaptation it causes improve play "
                "against the eight handcrafted opponents, relative to identical policy training on the frozen "
                "initial setup source?"
            ),
            "estimand": "the total benefit of enabling setup learning + co-adaptation; the setup network's isolated causal contribution is not identified",
            "primary_contrast": "EWR(candidate_final) - EWR(control_final), paired by case, stratified cluster bootstrap over bases within families",
            "decision_rule": f"PROCEED requires the {int(BOOTSTRAP_CONFIDENCE * 100)}% lower bound above zero AND the point estimate at least {PRIMARY_MARGIN}",
            "near_boundary_rule": "a second seed is a conditional follow-up when the interval contains the margin or the run is operationally irregular; never pooled after the fact to rescue a failed primary",
            "diagnostics": "candidate_final vs candidate_128 / candidate_0 are progress diagnostics only (require --diagnostic); the period-128 bundle is saved, not evaluated, unless a diagnosis needs it",
        },
        "lineages": {
            lineage: {"config_digest": config.config_digest(), "setup_updates_enabled": config.setup_updates_enabled}
            for lineage, config in configs.items()
        },
        "matched_configuration": candidate.matched_document(),
        "matched_digest": candidate.matched_digest(),
        "frozen_defaults": {
            "initialisation": "both lineages fresh: C1 from the canonical Phase 8 init seed, the setup model from derive_stream_seed(namespace, 'model_init', 1)",
            "c1_updates_per_period": candidate.c1_updates_per_period,
            "canonical_live_mixture": f"{candidate.canonical_per_batch}:{candidate.live_per_batch} of batch {candidate.c1_train_config.batch_size}",
            "live_retention_periods": candidate.live_retention_periods,
            "buffer_storage_periods": candidate.buffer_storage_periods,
            "periods": candidate.periods,
            "bundle_cadence_periods": candidate.bundle_cadence_periods,
            "plies_per_period": candidate.plies_per_period,
            "slots": candidate.slots,
            "pool_size": candidate.pool_size,
            "collector_rules": "TRAINING_RULES (battleless 100)",
            "evaluation_rules": "EVALUATION_RULES (battleless 200, P18-A001)",
            "seeds_one": 1,
            "c1_device": c1_device,
            "setup_device": candidate.setup_device,
        },
        "seeds": {
            "namespace": NAMESPACE,
            "seed_function": "stratego.setups.identity.derive_stream_seed",
            "setup_init_seed": candidate.setup_init_seed(),
            "c1_init_seed": int(candidate.c1_train_config.model_init_seed),
            "evaluation_schedule_root": evaluation_schedule_seed(NAMESPACE),
            "evaluation_bootstrap_seed": evaluation_bootstrap_seed(NAMESPACE),
            "streams": "collector_policy(colour, period, slot, ordinal), pool_pairing(period), c1_live_draw(period, update), pool(snapshot, index), reflection(snapshot, index), shuffle(update, epoch)",
        },
        "evaluation": {
            "schedule": schedule,
            "bases": {"count": len(bases), "indices": list(EVALUATION_BASE_INDICES), "per_family": 10, "reserved_untouched": list(RESERVED_BASE_INDICES)},
            "opponents": list(HANDCRAFTED_OPPONENTS),
            "cases_per_arm": len(cases),
            "arms_primary": ["candidate_final", "control_final"],
            "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "confidence": BOOTSTRAP_CONFIDENCE, "unit": "base within family, all 16 cases carried", "rescaling": "sqrt(n_f / (n_f - 1))"},
            "margin": PRIMARY_MARGIN,
        },
        "gates": {
            "G1": "legal setup generation: legality failures = 0 in every pool and evaluation sample",
            "G2": "orientation / reflection: orientation failures = 0",
            "G3": "exact setup-to-outcome attribution: attribution failures = 0",
            "G4": "completed-game accounting: started = completed + in-flight + failed per period; planned = completed + failed + missing per arm",
            "G5": "exact joint-bundle identity; no cross-checkpoint or cross-lineage pairing",
            "G6": "minimal checkpoint / resume equivalence (the restart check)",
            "G7": "paired evaluation: identical schedule digest, opponents, formations, colours and seeds across arms",
            "G8": "duplicate / diversity collapse: distinct reflection classes >= 922 of 1,024 in every pool",
            "G9": "finite losses, valid files: non-finite events = 0; every bundle reloads and verifies",
            "G10": "clean committed deliverable: no protected or sealed artifact modified",
        },
        "runtime": {"root_absolute": str(RUNTIME_ROOT), "root_relative": str(RUNTIME_RELATIVE)},
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_json(reports / CONTRACT_NAME, contract)
    log(f"contract frozen under {reports}: {len(cases)} cases per arm, schedule {schedule['digest'][:16]}")
    return contract


def load_contract(reports: Path) -> tuple:
    path = reports / CONTRACT_NAME
    if not path.exists():
        raise G3Error(f"BLOCKED: no frozen contract at {path}; run --freeze and commit first")
    return json.loads(path.read_text()), file_sha256(path)


def verify_frozen_identity(contract: dict) -> dict:
    """Rebuild the configurations and the schedule digest; refuse any drift."""
    from stratego.training.phase18.g3_evaluation import build_cases, build_schedule, load_evaluation_bases, schedule_record

    c1_device = contract["frozen_defaults"]["c1_device"]
    configs = {lineage: production_config(lineage, c1_device=c1_device) for lineage in LINEAGES}
    if configs[LINEAGE_CANDIDATE].matched_digest() != contract["matched_digest"]:
        raise G3Error("BLOCKED: the matched configuration does not re-derive from the code")
    for lineage, config in configs.items():
        if config.config_digest() != contract["lineages"][lineage]["config_digest"]:
            raise G3Error(f"BLOCKED: the {lineage} configuration does not re-derive")
    if contract["seeds"]["namespace"] != NAMESPACE or contract["run_id"] != RUN_ID:
        raise G3Error("BLOCKED: run id or namespace drift")
    bases = load_evaluation_bases()
    cases = build_cases(bases)
    matches = build_schedule(cases, namespace=NAMESPACE)
    schedule = schedule_record(cases, matches, namespace=NAMESPACE)
    if schedule["digest"] != contract["evaluation"]["schedule"]["digest"] or schedule["cases_digest"] != contract["evaluation"]["schedule"]["cases_digest"]:
        raise G3Error("BLOCKED: the evaluation schedule does not re-derive")
    return {"configs": configs, "cases": cases, "matches": matches, "schedule": schedule}


# ---------------------------------------------------------------------------
# Stage 2: verify (tests + the restart check)
# ---------------------------------------------------------------------------


def run_pytest(targets, junit: Path) -> dict:
    junit.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", *targets, "-q", "--no-header", "-p", "no:cacheprovider", f"--junitxml={junit}"]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True)
    import xml.etree.ElementTree as ET

    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for case in ET.parse(junit).getroot().iter("testcase"):
        status = "passed"
        for child in case:
            if child.tag == "failure":
                status = "failed"
            elif child.tag == "error":
                status = "error"
            elif child.tag == "skipped":
                status = "skipped"
        counts[status] += 1
    return {
        "targets": list(targets),
        "command": " ".join(command),
        "return_code": completed.returncode,
        "seconds": round(time.perf_counter() - started, 3),
        "summary_line": completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "",
        "counts": counts,
        "junit": str(junit),
        "junit_sha256": file_sha256(junit),
    }


def stage_verify(reports: Path, *, corpus_root: "Path | None") -> dict:
    from stratego.training.phase18.g3_smoke import restart_check

    contract, contract_sha = load_contract(reports)
    verify_frozen_identity(contract)
    work = reports / "g3_pilot_verification"
    log("running the Stage 6B test files")
    tests = run_pytest(TEST_TARGETS, work / "junit_stage6b.xml")
    log(f"tests: {tests['summary_line']}")
    log("running the restart check (design section 6)")
    import tempfile

    with tempfile.TemporaryDirectory(prefix="g3_restart_") as scratch:
        # The throwaway mini corpus lives in scratch too: reproducible from its
        # frozen ids, never committed under reports/.
        root = corpus_root if corpus_root is not None else _mini_corpus(Path(scratch) / "mini_corpus")
        restart = restart_check(root=Path(scratch), corpus_root=root, namespace=f"{NAMESPACE}:restart_check")
    log(f"restart check: {'PASS' if restart['passed'] else 'FAIL'} ({restart['unfinished_games_at_save']} games unfinished at the save point)")
    record = {
        "artifact": "phase18_g3_stage6b_verification_v1",
        "work_package": WORK_PACKAGE,
        "agent": AGENT,
        "gate": GATE,
        "contract_sha256": contract_sha,
        "tests": tests,
        "restart_check": restart,
        "source_digests": digests_of(HARNESS_SOURCES),
        "test_digests": digests_of(TEST_FILES),
        "environment": environment(),
        "timestamp_utc": utc_now(),
    }
    write_json(reports / VERIFICATION_NAME, record)
    return record


def _mini_corpus(root: Path) -> Path:
    """The suite's six-game mini corpus, generated once for the restart check."""
    from stratego.training import synthetic_corpus as sc
    from stratego.training.warmstart_seed import synthetic_game_id

    ids = (
        synthetic_game_id("train", "strategic_rule_based@1.1.0", "random_legal@1.0.0", 0),
        synthetic_game_id("train", "tactical_rule_based@1.0.0", "basic_heuristic@1.0.0", 0),
        synthetic_game_id("train", "random_legal@1.0.0", "random_legal@1.0.0", 0),
        synthetic_game_id("validation", "basic_heuristic@1.0.0", "stress_chaos@1.0.0", 0),
    )
    if not (root / "train").exists():
        sc.generate_corpus(root, worker_count=1, chunks_per_worker=1, game_ids=ids)
    return root


# ---------------------------------------------------------------------------
# Stage 3: launch manifest (from the clean execution worktree)
# ---------------------------------------------------------------------------


def stage_launch_manifest(reports: Path, *, source_commit: str) -> dict:
    porcelain = git_output("status", "--porcelain")
    if porcelain:
        raise G3Error(f"BLOCKED: the execution worktree is not clean:\n{porcelain}")
    head = git_output("rev-parse", "HEAD")
    if head != source_commit:
        raise G3Error(f"BLOCKED: HEAD {head} is not the source commit {source_commit}")
    contract, contract_sha = load_contract(reports)
    frozen = verify_frozen_identity(contract)
    verification_path = reports / VERIFICATION_NAME
    if not verification_path.exists():
        raise G3Error("BLOCKED: no verification record; run --verify first")
    verification = json.loads(verification_path.read_text())
    if verification["tests"]["return_code"] != 0 or not verification["restart_check"]["passed"]:
        raise G3Error("BLOCKED: the verification record is not green")
    ignored = subprocess.run(["git", "check-ignore", "-q", str(RUNTIME_RELATIVE / "probe")], cwd=CANONICAL_ROOT, capture_output=True).returncode == 0
    if not ignored:
        raise G3Error(f"BLOCKED: {RUNTIME_RELATIVE} is not git-ignored in the canonical tree")
    if RUNTIME_ROOT.exists() and any(RUNTIME_ROOT.iterdir()):
        raise G3Error(f"BLOCKED: the runtime root {RUNTIME_ROOT} already exists and is not empty")
    manifest = {
        "artifact": "phase18_g3_pilot_launch_manifest_v1",
        "work_package": WORK_PACKAGE,
        "agent": AGENT,
        "gate": GATE,
        "run_id": RUN_ID,
        "timestamp_utc": utc_now(),
        "authorization": digests_of(AUTHORIZATION_FILES),
        "source": {
            "g3_source_commit": source_commit,
            "g3_source_tree": git_output("rev-parse", f"{source_commit}^{{tree}}"),
            "design_commit": G3_DESIGN_COMMIT,
            "execution_worktree": str(REPOSITORY_ROOT),
            "worktree_porcelain_empty": True,
            "canonical_tree": str(CANONICAL_ROOT),
        },
        "runtime": {"root_absolute": str(RUNTIME_ROOT), "root_relative": str(RUNTIME_RELATIVE), "git_ignored": ignored},
        "contract_sha256": contract_sha,
        "verification_sha256": file_sha256(verification_path),
        "matched_digest": contract["matched_digest"],
        "schedule_digest": frozen["schedule"]["digest"],
        "source_digests": digests_of(HARNESS_SOURCES),
        "test_digests": digests_of(TEST_FILES),
        "environment": environment(),
        "budget": contract["frozen_defaults"],
        "authorises_pilot": False,
        "note": "a launch manifest binds identities; the pilot starts only on an explicit written instruction after review",
    }
    write_json(reports / LAUNCH_NAME, manifest)
    log(f"launch manifest bound to {source_commit[:12]} under {reports}")
    return manifest


# ---------------------------------------------------------------------------
# Stage 4: run one lineage
# ---------------------------------------------------------------------------


def accepted_corpus() -> tuple:
    """The accepted Phase 8 corpus through the resolver, identity cross-checked."""
    import run_phase8_agent06 as a6
    from stratego.training import synthetic_corpus as sc

    root = sc.default_corpus_root()
    if str(root) != a6.REQUIRED_CORPUS_ROOT:
        raise G3Error(f"BLOCKED: default_corpus_root() resolves to {root}, expected {a6.REQUIRED_CORPUS_ROOT}")
    return root, a6.accepted_corpus_identity()


def stage_run(reports: Path, *, lineage: str, runtime: Path, resume: bool, periods: "int | None", skip_payload_bytes: bool) -> dict:
    from stratego.training.phase18.g3_pilot import BUNDLES_DIRECTORY, LineageRunner, bundle_name
    from stratego.training.warmstart_checkpoint import verify_corpus_identity

    contract, contract_sha = load_contract(reports)
    frozen = verify_frozen_identity(contract)
    launch_path = reports / LAUNCH_NAME
    if not launch_path.exists():
        raise G3Error("BLOCKED: no launch manifest; run --launch-manifest from the clean worktree first")
    launch = json.loads(launch_path.read_text())
    if launch["contract_sha256"] != contract_sha:
        raise G3Error("BLOCKED: the launch manifest binds a different contract")
    config = frozen["configs"][lineage]
    root, accepted = accepted_corpus()
    log("verifying the accepted corpus identity" + (" (digests only)" if skip_payload_bytes else " and every payload"))
    identity = verify_corpus_identity(root, accepted, check_payload_bytes=not skip_payload_bytes)
    keywords = dict(run_root=runtime, corpus_root=root, corpus_identity=identity, log=log)
    started = time.perf_counter()
    if resume:
        state_path = runtime / lineage / "run_state.json"
        if not state_path.exists():
            raise G3Error(f"BLOCKED: nothing to resume under {runtime / lineage}")
        state = json.loads(state_path.read_text())
        bundle = runtime / lineage / BUNDLES_DIRECTORY / bundle_name(int(state["period"]) // config.bundle_cadence_periods * config.bundle_cadence_periods)
        if not bundle.exists():
            raise G3Error(f"BLOCKED: the last bundle {bundle} is missing")
        runner = LineageRunner.resume(config, bundle_directory=bundle, **keywords)
        log(f"{lineage}: resumed from {bundle.name} at period {runner.period}")
    else:
        runner = LineageRunner.fresh(config, **keywords)
        log(f"{lineage}: fresh lineage, bundle_0 written")
    try:
        records = runner.run(periods=periods)
    finally:
        runner.close()
    summary = {
        "artifact": f"phase18_g3_pilot_run_{lineage}_v1",
        "run_id": RUN_ID,
        "lineage": lineage,
        "g3_source_commit": launch["source"]["g3_source_commit"],
        "contract_sha256": contract_sha,
        "periods_run_this_process": len(records),
        "period_reached": runner.period,
        "horizon": config.periods,
        "complete": runner.period == config.periods,
        "last_bundle_id": runner.last_bundle_id,
        "integrity": runner.integrity,
        "setup_skips": runner.setup_skips,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "environment": environment(),
        "timestamp_utc": utc_now(),
    }
    write_json(runtime / lineage / "run_summary.json", summary)
    log(f"{lineage}: period {runner.period}/{config.periods} in {summary['wall_seconds']:.0f} s; nothing continues past the horizon")
    return summary


# ---------------------------------------------------------------------------
# Stage 5: matching check
# ---------------------------------------------------------------------------


def stage_check_matching(reports: Path, *, runtime: Path) -> dict:
    from stratego.training.phase18.g3_pilot import matching_check

    contract, contract_sha = load_contract(reports)
    report = matching_check(runtime, c1_device=contract["frozen_defaults"]["c1_device"])
    record = {"artifact": "phase18_g3_pilot_matching_v1", "run_id": RUN_ID, "contract_sha256": contract_sha, "timestamp_utc": utc_now(), **report}
    write_json(reports / MATCHING_NAME, record)
    log(f"matching: {'OK' if report['matched'] else 'PROBLEMS ' + str(report['problems'])}")
    return record


# ---------------------------------------------------------------------------
# Stage 6: evaluate one arm
# ---------------------------------------------------------------------------

ARMS = {
    "candidate_final": (LINEAGE_CANDIDATE, PILOT_PERIODS, False),
    "control_final": (LINEAGE_CONTROL, PILOT_PERIODS, False),
    "candidate_128": (LINEAGE_CANDIDATE, 128, True),
    "candidate_0": (LINEAGE_CANDIDATE, 0, True),
    "control_128": (LINEAGE_CONTROL, 128, True),
}


def stage_evaluate(reports: Path, *, runtime: Path, arm: str, device: str, workers: int, chunk_units: int, diagnostic: bool) -> dict:
    from stratego.training.phase18.g3_evaluation import evaluate_bundle
    from stratego.training.phase18.g3_pilot import BUNDLES_DIRECTORY, bundle_name

    contract, contract_sha = load_contract(reports)
    frozen = verify_frozen_identity(contract)
    if arm not in ARMS:
        raise G3Error(f"BLOCKED: unknown arm {arm!r}; known: {sorted(ARMS)}")
    lineage, period, is_diagnostic = ARMS[arm]
    if is_diagnostic and not diagnostic:
        raise G3Error(f"BLOCKED: {arm} is a progress diagnostic; pass --diagnostic to run it (reviewer decision 4)")
    config = frozen["configs"][lineage]
    bundle = runtime / lineage / BUNDLES_DIRECTORY / bundle_name(period)
    work = runtime / "evaluation" / arm
    record, _rows = evaluate_bundle(bundle, config=config, lineage=lineage, label=arm, cases=frozen["cases"], work=work, device=device, workers=workers, chunk_units=chunk_units, log=log)
    record = record | {"artifact": f"phase18_g3_pilot_arm_{arm}_v1", "contract_sha256": contract_sha, "diagnostic": is_diagnostic, "environment": environment(), "timestamp_utc": utc_now()}
    write_json(work / "arm_record.json", record)
    log(f"{arm}: {record['accounting']['completed']}/{record['accounting']['planned']} games, reconciles={record['accounting']['reconciles']}")
    return record


# ---------------------------------------------------------------------------
# Stage 7: analyse
# ---------------------------------------------------------------------------


def stage_analyse(reports: Path, *, runtime: Path) -> dict:
    from stratego.training.phase18.g3_evaluation import paired_analysis, prove_arm_identity, read_receipt_rows, reconcile
    from stratego.training.phase18.g3_pilot import read_period_records

    contract, contract_sha = load_contract(reports)
    frozen = verify_frozen_identity(contract)
    cases, matches = frozen["cases"], frozen["matches"]
    arms = {}
    for arm in ("candidate_final", "control_final"):
        work = runtime / "evaluation" / arm
        record_path = work / "arm_record.json"
        if not record_path.exists():
            raise G3Error(f"BLOCKED: arm {arm} has not been evaluated")
        record = json.loads(record_path.read_text())
        rows = read_receipt_rows(record["receipts"]["path"])
        if file_sha256(record["receipts"]["path"]) != record["receipts"]["sha256"]:
            raise G3Error(f"BLOCKED: the {arm} receipts digest moved")
        accounting = reconcile(matches, rows)
        if not accounting["complete_for_primary"]:
            raise G3Error(f"BLOCKED: the {arm} arm is incomplete: {accounting}")
        if record["schedule"]["digest"] != frozen["schedule"]["digest"]:
            raise G3Error(f"BLOCKED: the {arm} arm played a different schedule")
        arms[arm] = {"record": record, "rows": rows, "accounting": accounting}
    identity = prove_arm_identity({arm: arms[arm]["rows"] for arm in arms}, cases)
    if identity["problems"]:
        raise G3Error(f"BLOCKED: arm identity failed: {identity['problems']}")
    analysis = paired_analysis(arms["candidate_final"]["rows"], arms["control_final"]["rows"], cases, namespace=NAMESPACE)
    matching_path = reports / MATCHING_NAME
    matching = json.loads(matching_path.read_text()) if matching_path.exists() else None
    periods = {lineage: read_period_records(runtime / lineage) for lineage in LINEAGES}
    integrity = {lineage: (periods[lineage][-1]["integrity"] if periods[lineage] else None) for lineage in LINEAGES}
    diversity = {lineage: min((r["pool"]["distinct_reflection_classes"] for r in periods[lineage]), default=None) for lineage in LINEAGES}
    gates = {
        "G1_legality": all(i and i["legality_failures"] == 0 for i in integrity.values()) and all(arms[a]["record"]["own_setups"]["legality_failures"] == 0 for a in arms),
        "G2_orientation": all(i and i["orientation_failures"] == 0 for i in integrity.values()) and all(arms[a]["record"]["own_setups"]["orientation_failures"] == 0 for a in arms),
        "G3_attribution": all(i and i["attribution_failures"] == 0 for i in integrity.values()),
        "G4_accounting": all(arms[a]["accounting"]["complete_for_primary"] for a in arms) and all(p and all(r["collection"]["failed"] == 0 for r in p) for p in periods.values()),
        "G5_bundle_identity": all(arms[a]["record"]["bundle_id"] for a in arms) and arms["candidate_final"]["record"]["lineage"] == "candidate" and arms["control_final"]["record"]["lineage"] == "control",
        "G6_restart": bool(json.loads((reports / VERIFICATION_NAME).read_text())["restart_check"]["passed"]) if (reports / VERIFICATION_NAME).exists() else False,
        "G7_paired": not identity["problems"] and arms["candidate_final"]["record"]["schedule"]["digest"] == arms["control_final"]["record"]["schedule"]["digest"],
        "G8_diversity": all(d is not None and d >= 922 for d in diversity.values()),
        "G9_finite_and_valid": all(i and i["non_finite_events"] == 0 for i in integrity.values()),
        "G10_clean_deliverable": None,
    }
    decision = "PROCEED" if analysis["passes"] and all(v for k, v in gates.items() if k != "G10_clean_deliverable") else ("NEAR_BOUNDARY" if analysis["near_boundary"] else "FAIL")
    results = {
        "artifact": "phase18_g3_pilot_results_v1",
        "work_package": WORK_PACKAGE,
        "agent": AGENT,
        "gate": GATE,
        "run_id": RUN_ID,
        "contract_sha256": contract_sha,
        "schedule_digest": frozen["schedule"]["digest"],
        "arms": {arm: {k: arms[arm]["record"][k] for k in ("bundle_id", "bundle_period", "lineage", "c1_state_digest", "setup_model_digest", "accounting", "receipts")} for arm in arms},
        "arm_identity_proof": identity,
        "primary": analysis,
        "gates": gates,
        "matching": matching,
        "pool_diversity_minimum": diversity,
        "decision_input": {"decision": decision, "rule": contract["question"]["decision_rule"], "near_boundary_rule": contract["question"]["near_boundary_rule"], "note": "G10 is judged at review from the committed tree, not computed here"},
        "environment": environment(),
        "timestamp_utc": utc_now(),
    }
    write_json(reports / RESULTS_NAME, results)
    log(f"primary: point {analysis['point']:+.4f}, 95% [{analysis['lower']:+.4f}, {analysis['upper']:+.4f}] -> {decision}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--freeze", action="store_true")
    stage.add_argument("--verify", action="store_true")
    stage.add_argument("--launch-manifest", action="store_true")
    stage.add_argument("--run", action="store_true")
    stage.add_argument("--check-matching", action="store_true")
    stage.add_argument("--evaluate", action="store_true")
    stage.add_argument("--analyse", action="store_true")
    stage.add_argument("--restart-check", action="store_true")
    parser.add_argument("--reports", type=Path, default=CANONICAL_ROOT / REPORTS_RELATIVE)
    parser.add_argument("--runtime", type=Path, default=RUNTIME_ROOT)
    parser.add_argument("--lineage", choices=LINEAGES)
    parser.add_argument("--arm", choices=sorted(ARMS))
    parser.add_argument("--source-commit")
    parser.add_argument("--c1-device", default="mps")
    parser.add_argument("--device", default="mps", help="inference device for --evaluate")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-units", type=int, default=64)
    parser.add_argument("--periods", type=int, default=None, help="--run: stop after this many more periods (never beyond the horizon)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--skip-payload-bytes", action="store_true")
    parser.add_argument("--corpus-root", type=Path, default=None, help="--verify / --restart-check: corpus for the smoke restart check (default: a generated mini corpus)")
    arguments = parser.parse_args(argv)
    try:
        if arguments.freeze:
            stage_freeze(arguments.reports, c1_device=arguments.c1_device)
        elif arguments.verify:
            stage_verify(arguments.reports, corpus_root=arguments.corpus_root)
        elif arguments.launch_manifest:
            if not arguments.source_commit:
                parser.error("--launch-manifest needs --source-commit")
            stage_launch_manifest(arguments.reports, source_commit=arguments.source_commit)
        elif arguments.run:
            if not arguments.lineage:
                parser.error("--run needs --lineage")
            stage_run(arguments.reports, lineage=arguments.lineage, runtime=arguments.runtime, resume=arguments.resume, periods=arguments.periods, skip_payload_bytes=arguments.skip_payload_bytes)
        elif arguments.check_matching:
            stage_check_matching(arguments.reports, runtime=arguments.runtime)
        elif arguments.evaluate:
            if not arguments.arm:
                parser.error("--evaluate needs --arm")
            stage_evaluate(arguments.reports, runtime=arguments.runtime, arm=arguments.arm, device=arguments.device, workers=arguments.workers, chunk_units=arguments.chunk_units, diagnostic=arguments.diagnostic)
        elif arguments.analyse:
            stage_analyse(arguments.reports, runtime=arguments.runtime)
        elif arguments.restart_check:
            from stratego.training.phase18.g3_smoke import restart_check
            import tempfile

            with tempfile.TemporaryDirectory(prefix="g3_restart_") as scratch:
                root = arguments.corpus_root if arguments.corpus_root is not None else _mini_corpus(Path(scratch) / "mini_corpus")
                report = restart_check(root=Path(scratch), corpus_root=root, namespace=f"{NAMESPACE}:restart_check")
            print(json.dumps(report, indent=1, sort_keys=True))
            return 0 if report["passed"] else 1
    except (G3Error, Phase18G3Error) as error:
        log(str(error))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
