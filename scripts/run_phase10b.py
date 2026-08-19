#!/usr/bin/env python3
"""Optional Phase 10B harness: setup-conditioned self-play fine-tuning.

Runs the whole optional experiment the plan describes: verify every frozen
upstream identity from live bytes, freeze the Phase 10B contract, seeds and
banks, fine-tune the accepted Phase 9 move policy in P10-D-conditioned
self-play under a strict bounded budget, validate on the frozen cadence,
select one checkpoint, run a single sealed final evaluation, recompute Gates
A-H, and hand the result back.

What this harness never does
----------------------------
It writes nothing into the Phase 9 or Phase 10 artifact namespaces, takes no
optimizer step on the accepted Phase 9 checkpoint, refits no utility, changes
no temperature, mixture or threshold, uses no search, opens the sealed test
bank before selection is closed, and makes no production replacement
decision. Phase 11 is neither blocked nor modified.

Usage::

    python scripts/run_phase10b.py                    # every stage
    python scripts/run_phase10b.py --stage verify
    python scripts/run_phase10b.py --stage train --max-iterations 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

PHASE = "10B"
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_10b_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_10b_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase10b"
STAGE_DIRECTORY = WORK_DIRECTORY / "stages"
EXPORT_DIRECTORY = WORK_DIRECTORY / "exports"
CELL_DIRECTORY = WORK_DIRECTORY / "cells"
ARCHIVE_DIRECTORY = WORK_DIRECTORY / "archive"

PHASE9_CHECKPOINT = REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"
PHASE8_ANCHOR_EXPORT = REPOSITORY_ROOT / "checkpoints" / "phase9" / "agent01" / "anchor_eval.pt"
CANONICAL_CANDIDATE = WORK_DIRECTORY / "setup_conditioned_c1_v1.pt"

SECTION_MARKER = "## Phase 10B — Setup-Conditioned Self-Play Fine-Tuning"


class Phase10BHarnessError(RuntimeError):
    """The Phase 10B harness refuses to continue."""


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=str(REPOSITORY_ROOT),
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:  # pragma: no cover - git is present in this project
        return ""


def environment_record() -> dict:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "source_revision": git_output("rev-parse", "--short", "HEAD"),
        "working_tree_state": "dirty" if git_output("status", "--porcelain") else "clean",
    }


def stage_path(name: str) -> Path:
    return STAGE_DIRECTORY / f"{name}.json"


def write_stage(name: str, payload: dict) -> dict:
    payload = dict(payload)
    payload.setdefault("stage", name)
    payload.setdefault("environment", environment_record())
    write_json(stage_path(name), payload)
    return payload


def read_stage(name: str) -> dict:
    path = stage_path(name)
    if not path.exists():
        raise Phase10BHarnessError(
            f"stage {name!r} has not run; expected {path.relative_to(REPOSITORY_ROOT)}"
        )
    return read_json(path)


# ---------------------------------------------------------------------------
# Stage: verify
# ---------------------------------------------------------------------------


def stage_verify(args) -> dict:
    """Recompute every frozen upstream identity from live bytes."""
    from stratego.setups.sampler import load_library_index
    from stratego.training import phase10b_contract as contract
    from stratego.training import phase10b_seed as seeds
    from stratego.training import phase10b_storage as storage
    from stratego.training.phase10_selector import candidate, load_scorer
    from stratego.training.phase10b_checkpoint import assert_phase9_untouched

    log("verifying frozen upstream identities from live bytes")
    problems: list = []
    checks: dict = {}

    phase9 = assert_phase9_untouched(REPOSITORY_ROOT)
    checks["phase9_checkpoint"] = phase9

    from stratego.training.phase9_behavior import state_dict_digest
    from stratego.training.phase9_checkpoint import model_from_payload, read_phase9_payload

    payload = read_phase9_payload(PHASE9_CHECKPOINT)
    model = model_from_payload(payload, device="cpu")
    observed_state = state_dict_digest(model)
    parameters = int(sum(tensor.numel() for tensor in model.parameters()))
    summary = model.architecture_summary()
    config_digest = summary.get("config_digest") or summary.get("configuration_digest")
    del model
    checks["phase9_model"] = {
        "model_state_digest": observed_state,
        "parameters": parameters,
        "c1_config_digest": config_digest,
    }
    if observed_state != contract.ACCEPTED_PHASE9_STATE_DIGEST:
        problems.append(f"Phase 9 model-state digest {observed_state}")
    if parameters != contract.ACCEPTED_PHASE9_PARAMETERS:
        problems.append(f"Phase 9 parameter count {parameters}")
    if config_digest is not None and config_digest != contract.ACCEPTED_C1_CONFIG_DIGEST:
        problems.append(f"C1 config digest {config_digest}")

    selector_config_path = (
        REPOSITORY_ROOT / "reports" / "phase_10_data" / "agent_05_frozen_selector_config.json"
    )
    manifest_path = (
        REPOSITORY_ROOT / "reports" / "phase_10_data" / "agent_06_production_selector_manifest.json"
    )
    selector_sha = file_sha256(selector_config_path)
    checks["selector_config"] = {
        "path": str(selector_config_path.relative_to(REPOSITORY_ROOT)),
        "sha256": selector_sha,
    }
    if selector_sha != contract.ACCEPTED_SELECTOR_CONFIG_SHA256:
        problems.append(f"P10-D selector config SHA-256 {selector_sha}")

    production = read_json(manifest_path)
    checks["phase10_system"] = {
        "digest": production["phase10_system_v1_digest"],
        "selector_config_sha256": production["selector_config_sha256"],
        "winner": production["winner"],
        "split": production["split"],
    }
    if production["phase10_system_v1_digest"] != contract.ACCEPTED_PHASE10_SYSTEM_DIGEST:
        problems.append("Phase 10 production system digest")
    if production["winner"]["candidate_id"] != contract.SELECTED_CANDIDATE_ID:
        problems.append(f"Phase 10 winner {production['winner']['candidate_id']}")

    frozen = read_json(selector_config_path)
    utility = frozen["utility"]
    checks["utility"] = {
        "model_t_coefficient_digest": utility["coefficient_digests"]["model_T"],
        "trait_scaler_digest": utility["scaler_digest"],
        "artifact": utility["artifact"],
        "file_sha256": utility["file_sha256"],
        "refit_by_agent_5": utility["refit_by_agent_5"],
    }
    if utility["coefficient_digests"]["model_T"] != contract.ACCEPTED_MODEL_T_COEFFICIENT_DIGEST:
        problems.append("model_T coefficient digest")
    if utility["scaler_digest"] != contract.ACCEPTED_TRAIT_SCALER_DIGEST:
        problems.append("trait scaler digest")
    utility_path = REPOSITORY_ROOT / utility["artifact"]
    utility_sha = file_sha256(utility_path)
    checks["utility"]["live_file_sha256"] = utility_sha
    if utility_sha != utility["file_sha256"]:
        problems.append(f"live utility artifact SHA-256 {utility_sha}")

    scorer = load_scorer()
    live_coefficients = scorer.coefficient_digest("model_T") if hasattr(
        scorer, "coefficient_digest"
    ) else None
    checks["utility"]["live_coefficient_digest"] = live_coefficients
    if live_coefficients is not None and live_coefficients != (
        contract.ACCEPTED_MODEL_T_COEFFICIENT_DIGEST
    ):
        problems.append(f"live model_T coefficient digest {live_coefficients}")

    selector = candidate(contract.SELECTED_CANDIDATE_ID)
    checks["selector"] = {
        "candidate_id": selector.candidate_id,
        "utility_model": selector.utility_model,
        "temperature": float(selector.temperature),
        "selector_identity": selector.selector_identity,
        "mixture_neutral": contract.NEUTRAL_MIXTURE_WEIGHT,
        "mixture_learned": contract.LEARNED_MIXTURE_WEIGHT,
    }
    if selector.utility_model != contract.SELECTOR_UTILITY_MODEL:
        problems.append(f"selector utility model {selector.utility_model}")
    if float(selector.temperature) != contract.SELECTOR_TEMPERATURE:
        problems.append(f"selector temperature {selector.temperature}")

    index = load_library_index()
    checks["setup_library"] = {
        "content_digest": index.content_digest,
        "bases": len(index.bases) if hasattr(index, "bases") else None,
    }
    if index.content_digest != contract.ACCEPTED_LIBRARY_CONTENT_DIGEST:
        problems.append(f"Phase 7 library content digest {index.content_digest}")

    checks["storage"] = storage.resolve_writable(
        required_bytes=storage.projected_bytes(contract.MAX_TRAINING_GAMES)
    )
    checks["contract"] = {
        "contract_digest": contract.contract_digest(),
        "seed_contract_digest": seeds.seed_contract_digest(),
        "namespace": contract.PHASE10B_NAMESPACE,
    }
    checks["phase8_anchor_export"] = {
        "path": str(PHASE8_ANCHOR_EXPORT.relative_to(REPOSITORY_ROOT)),
        "exists": PHASE8_ANCHOR_EXPORT.exists(),
        "sha256": file_sha256(PHASE8_ANCHOR_EXPORT) if PHASE8_ANCHOR_EXPORT.exists() else None,
    }
    if not PHASE8_ANCHOR_EXPORT.exists():
        problems.append("the Phase 8 anchor evaluation export is missing")

    status = "VERIFIED" if not problems else "BLOCKED"
    log(f"  {status}: {len(problems)} problem(s)")
    for problem in problems:
        log(f"    - {problem}")
    return write_stage(
        "verify",
        {"status": status, "checks": checks, "problems": problems},
    )


# ---------------------------------------------------------------------------
# Stage: freeze
# ---------------------------------------------------------------------------


def stage_freeze(args) -> dict:
    """Freeze the contract, seeds and both evaluation banks before any rollout."""
    from stratego.evaluation import phase10b_banks as banks
    from stratego.training import phase10b_contract as contract
    from stratego.training import phase10b_schedule as schedule
    from stratego.training import phase10b_seed as seeds
    from stratego.training import phase10b_trainer as trainer
    from stratego.training.phase10b_setup_source import Phase10BSetupSource

    verify = read_stage("verify")
    if verify["status"] != "VERIFIED":
        raise Phase10BHarnessError("verification did not pass; nothing may be frozen")

    log("freezing the Phase 10B contract, seeds and banks")
    document = contract.contract_document()
    seed_document = seeds.seed_contract()

    source = Phase10BSetupSource.build()
    schedule_audits = {
        str(iteration): schedule.iteration_audit(iteration, setup_source=source)
        for iteration in (1, 5, 6, 15, 30)
    }
    audit_problems = [
        problem
        for audit in schedule_audits.values()
        for problem in audit["problems"]
    ]

    built = {}
    for bank in ("validation", "test"):
        cases, manifest = banks.build_bank(bank)
        audit = banks.audit_bank(cases, manifest)
        built[bank] = (cases, manifest, audit)
        log(
            f"  {bank}: {manifest['case_count']} cases, digest "
            f"{manifest['bank_digest'][:16]}, {len(audit['problems'])} problem(s)"
        )
    cross = banks.cross_bank_isolation(built["validation"][0], built["test"][0])

    bank_problems = [
        problem for _c, _m, audit in built.values() for problem in audit["problems"]
    ]
    if not cross["disjoint"]:
        bank_problems.append("the validation and test banks share an arrangement")

    payload = {
        "status": "FROZEN" if not (audit_problems or bank_problems) else "BLOCKED",
        "contract_document": document,
        "contract_digest": contract.contract_digest(),
        "seed_contract": seed_document,
        "seed_contract_digest": seeds.seed_contract_digest(),
        "population_document": schedule.population_document(),
        "population_digest": schedule.population_digest(),
        "setup_source": source.describe(),
        "trainer_semantics": trainer.trainer_semantics(),
        "schedule_audits": schedule_audits,
        "banks": {
            bank: {
                "manifest": manifest,
                "audit": audit,
            }
            for bank, (_cases, manifest, audit) in built.items()
        },
        "cross_bank_isolation": cross,
        "test_bank_sealed": True,
        "test_bank_access": "structural construction and audit only",
        "problems": audit_problems + bank_problems,
    }
    write_json(DATA_DIRECTORY / "agent_10b_contract.json", {
        "contract_document": document,
        "contract_digest": contract.contract_digest(),
        "seed_contract": seed_document,
        "seed_contract_digest": seeds.seed_contract_digest(),
        "population_document": schedule.population_document(),
        "population_digest": schedule.population_digest(),
        "setup_source": source.describe(),
        "trainer_semantics": trainer.trainer_semantics(),
        "banks": {
            bank: manifest for bank, (_c, manifest, _a) in built.items()
        },
        "cross_bank_isolation": cross,
        "environment": environment_record(),
    })
    log(f"  {payload['status']}")
    return write_stage("freeze", payload)


# ---------------------------------------------------------------------------
# Evaluation: work units, workers and aggregation
# ---------------------------------------------------------------------------


def _unit_path(tag: str, arm: str, matchup: str, start: int, stop: int) -> Path:
    return CELL_DIRECTORY / tag / f"{arm}__{matchup}__{start:04d}_{stop:04d}.pkl"


_WORKER_STATE: dict = {}


def _eval_worker_init(options: dict) -> None:
    import torch

    torch.set_num_threads(1)
    from stratego.evaluation import phase10b_banks as banks
    from stratego.evaluation import phase10b_eval as ev
    from stratego.evaluation import phase10b_runner as runner

    cases, _manifest = banks.build_bank(options["bank"])
    arms, close = runner.build_arm_policies(
        candidate_export=options["candidate_export"],
        baseline_export=options["baseline_export"],
        anchor_export=options["anchor_export"],
        device=options["device"],
        label=options["label"],
    )
    _WORKER_STATE.update(
        {
            "cases": cases,
            "arms": arms,
            "close": close,
            "source": ev.production_selector(),
        }
    )


def _eval_worker_unit(payload):
    from stratego.evaluation import phase10b_runner as runner

    tag, arm, matchup, start, stop = payload
    path = _unit_path(tag, arm, matchup, start, stop)
    if path.exists():
        return payload, 0.0, True
    produced = runner.run_cell(
        _WORKER_STATE["cases"][start:stop],
        arm,
        matchup,
        arms=_WORKER_STATE["arms"],
        setup_source=_WORKER_STATE["source"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as stream:
        pickle.dump(produced, stream)
    return payload, produced["seconds"], False


def run_evaluation(
    *,
    tag: str,
    bank: str,
    cells,
    case_count: int,
    candidate_export: Path,
    baseline_export: Path,
    workers: int,
    device: str,
    chunk: int,
) -> dict:
    """Play every requested cell over the bank, resuming completed units."""
    from concurrent.futures import ProcessPoolExecutor

    units = []
    for arm, matchup in cells:
        for start in range(0, case_count, chunk):
            stop = min(start + chunk, case_count)
            units.append((tag, arm, matchup, start, stop))
    pending = [unit for unit in units if not _unit_path(*unit).exists()]
    log(
        f"  {tag}: {len(units)} unit(s), {len(pending)} pending, {workers} worker(s), "
        f"device={device}"
    )
    started = time.perf_counter()
    if pending:
        options = {
            "bank": bank,
            "candidate_export": str(candidate_export),
            "baseline_export": str(baseline_export),
            "anchor_export": str(PHASE8_ANCHOR_EXPORT),
            "device": device,
            "label": tag,
        }
        if workers <= 1:
            _eval_worker_init(options)
            try:
                for unit in pending:
                    _eval_worker_unit(unit)
            finally:
                _WORKER_STATE["close"]()
                _WORKER_STATE.clear()
        else:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_eval_worker_init, initargs=(options,)
            ) as pool:
                done = 0
                for _payload, _seconds, cached in pool.map(_eval_worker_unit, pending):
                    done += 1
                    if done % 10 == 0 or done == len(pending):
                        log(f"    {done}/{len(pending)} units")
    rows_by_cell: dict = {}
    for unit in units:
        with open(_unit_path(*unit), "rb") as stream:
            produced = pickle.load(stream)
        rows_by_cell.setdefault((produced["arm"], produced["matchup"]), []).extend(
            produced["rows"]
        )
    seconds = time.perf_counter() - started
    games = sum(len(rows) for rows in rows_by_cell.values())
    log(f"  {tag}: {games} games in {seconds:.0f}s ({games / max(seconds, 1e-9):.2f} g/s)")
    return {"rows_by_cell": rows_by_cell, "seconds": seconds, "games": games}


# ---------------------------------------------------------------------------
# Stage: train
# ---------------------------------------------------------------------------


def _journal_path() -> Path:
    return WORK_DIRECTORY / "run_journal.json"


def _load_journal() -> dict:
    path = _journal_path()
    if path.exists():
        return read_json(path)
    return {
        "iterations": [],
        "snapshots": {},
        "archives": {},
        "validations": [],
        "wall_clock": {"run_seconds": 0.0},
        "stop": None,
    }


def _save_journal(journal: dict) -> None:
    write_json(_journal_path(), journal)


def stage_train(args) -> dict:
    """The bounded P10-D-conditioned fine-tuning run."""
    from stratego.evaluation import phase10b_acceptance as acceptance
    from stratego.evaluation import phase10b_banks as banks
    from stratego.evaluation import phase10b_eval as ev
    from stratego.training import phase10b_checkpoint as checkpoints
    from stratego.training import phase10b_collector as collector
    from stratego.training import phase10b_contract as contract
    from stratego.training import phase10b_schedule as schedule
    from stratego.training import phase10b_storage as storage
    from stratego.training import phase10b_trainer as trainer_module
    from stratego.training.phase10b_setup_source import Phase10BSetupSource
    from stratego.training.phase9_collector import IterationParticipants
    from stratego.training.phase9_trainer import LoaderTopology

    freeze = read_stage("freeze")
    if freeze["status"] != "FROZEN":
        raise Phase10BHarnessError("the contract is not frozen; training may not start")

    budget_iterations = min(int(args.max_iterations), contract.MAX_ITERATIONS)
    ceiling = float(args.wall_clock_seconds)
    root = storage.rollout_root()
    journal = _load_journal()
    completed = len(journal["iterations"])
    log(
        f"training: {completed} iteration(s) complete, budget {budget_iterations}, "
        f"ceiling {ceiling / 3600:.1f} h, rollouts at {root}"
    )

    checkpoints.assert_phase9_untouched(REPOSITORY_ROOT)
    source = Phase10BSetupSource.build()
    resolver = checkpoints.SnapshotResolver(
        device=args.collect_device, inference_batch_shape=args.batch_shape
    )
    anchor = resolver.resolve(
        PHASE9_CHECKPOINT,
        logical_identity=contract.ANCHOR_IDENTITY,
        policy_token=schedule.ANCHOR_POLICY_TOKEN,
        expected_sha256=contract.ACCEPTED_PHASE9_SHA256,
    )
    journal["snapshots"].setdefault(
        schedule.behavior_snapshot_identity(1), anchor.checkpoint_sha256
    )

    resume_path = WORK_DIRECTORY / "resume.pt"
    if completed and resume_path.exists():
        payload = checkpoints.read(resume_path)
        model = checkpoints.model_from_payload(payload, device=args.train_device)
        trainer = trainer_module.Phase10BTrainer(
            model, device=args.train_device,
            topology=LoaderTopology(workers=args.loader_workers),
            run_label="phase10b",
        )
        trainer.restore(payload)
        log(f"  resumed from iteration {trainer.rl_iteration}, step {trainer.global_step}")
    else:
        trainer = trainer_module.load_from_phase9(
            PHASE9_CHECKPOINT,
            device=args.train_device,
            topology=LoaderTopology(workers=args.loader_workers),
            run_label="phase10b",
        )
        log(f"  initialized from the accepted Phase 9 checkpoint ({trainer.model_state_digest[:16]})")

    validation_cases, validation_manifest = banks.build_bank("validation")
    validation_case_count = min(int(args.validation_cases), len(validation_cases))
    baseline_export = EXPORT_DIRECTORY / "phase9_eval.pt"
    if not baseline_export.exists():
        checkpoints.export_evaluation_weights(PHASE9_CHECKPOINT, baseline_export)

    run_started = time.perf_counter()
    elapsed_before = float(journal["wall_clock"]["run_seconds"])

    def elapsed() -> float:
        return elapsed_before + (time.perf_counter() - run_started)

    stop_reason = None
    with trainer:
        for iteration in range(completed + 1, budget_iterations + 1):
            if elapsed() >= ceiling:
                stop_reason = "wall_clock_ceiling"
                break
            iteration_started = time.perf_counter()
            identity = schedule.behavior_snapshot_identity(iteration)
            if iteration == 1:
                snapshot_path = PHASE9_CHECKPOINT
            else:
                snapshot_path = WORK_DIRECTORY / f"behavior_{identity}.pt"
            behavior = resolver.resolve(
                snapshot_path,
                logical_identity=identity,
                policy_token=schedule.behavior_policy_token(iteration),
                expected_sha256=journal["snapshots"].get(identity),
            )
            window = contract.active_archive_window(iteration)
            historical = {contract.ANCHOR_IDENTITY: anchor}
            digests = {contract.ANCHOR_IDENTITY: anchor.checkpoint_sha256}
            for member in window:
                if member == contract.ANCHOR_IDENTITY:
                    continue
                path = ARCHIVE_DIRECTORY / f"{member}.pt"
                historical[member] = resolver.resolve(
                    path,
                    logical_identity=member,
                    policy_token=schedule.history_policy_token(member),
                    expected_sha256=journal["archives"][member],
                )
                digests[member] = historical[member].checkpoint_sha256
            history = schedule.ActiveArchiveManifest.frozen_for(iteration, digests)
            participants = IterationParticipants(behavior=behavior, historical=historical)

            log(f"iteration {iteration}: collecting 2,048 games (window {list(window)})")
            collected = collector.collect_iteration(
                root,
                iteration,
                participants,
                setup_source=source,
                history=history,
                games_in_flight=args.games_in_flight,
                observer_probe_plies=args.observer_probe_plies,
                progress=lambda done, total: log(f"    collected {done}/{total}")
                if args.verbose
                else None,
            )
            collect_seconds = time.perf_counter() - iteration_started

            rollout = trainer_module.bind_sealed_rollout(
                root,
                iteration,
                behavior_snapshot=behavior,
                expected_model_state_digest=trainer.model_state_digest,
            )
            log(
                f"  sealed {rollout.games} games, {rollout.learner_decisions:,} learner "
                f"decisions, digest {rollout.sealed_rollout_digest[:16]}"
            )
            trainer.bind_iteration(rollout)
            train_started = time.perf_counter()
            rows = trainer.train_iteration()
            train_seconds = time.perf_counter() - train_started
            trainer.mark_iteration_trained()
            trainer.close()

            epoch_rows = [row for row in rows if "epoch_mean_kl" in row]
            summary = {
                "iteration": iteration,
                "games": rollout.games,
                "learner_decisions": rollout.learner_decisions,
                "sealed_rollout_digest": rollout.sealed_rollout_digest,
                "behavior_snapshot_id": rollout.behavior_snapshot_id,
                "behavior_checkpoint_sha256": rollout.behavior_checkpoint_sha256,
                "optimizer_steps": len(rows),
                "global_optimizer_step": trainer.global_step,
                "examples_consumed": trainer.examples_consumed,
                "learning_rate": contract.learning_rate(iteration),
                "entropy_coefficient": contract.entropy_coefficient(iteration),
                "kl_beta_end": float(trainer.controller.beta),
                "epoch_mean_kl": [row["epoch_mean_kl"] for row in epoch_rows],
                "epoch_clip_fraction": [row["epoch_clip_fraction"] for row in epoch_rows],
                "mean_advantage_retention": sum(
                    row["advantage_retention"] for row in rows
                ) / max(len(rows), 1),
                "mean_policy_loss": sum(row["ppo"] for row in rows) / max(len(rows), 1)
                if rows and "ppo" in rows[0]
                else None,
                "terminal_results": collected.get("terminal_results", {}),
                "bucket_counts": collected.get("bucket_counts", {}),
                "collect_seconds": collect_seconds,
                "train_seconds": train_seconds,
                "counters": dict(trainer.counters),
                "active_archive_window": list(window),
            }
            journal["iterations"].append(summary)
            log(
                f"  trained {len(rows)} update(s) in {train_seconds:.0f}s; epoch KL "
                f"{summary['epoch_mean_kl']}, clip {summary['epoch_clip_fraction']}, "
                f"retention {summary['mean_advantage_retention']:.3f}"
            )

            # The next behavior snapshot is frozen only after this iteration's
            # epochs are complete, which is the frozen order.
            if iteration + 1 <= contract.MAX_ITERATIONS:
                next_identity = schedule.behavior_snapshot_identity(iteration + 1)
                written = trainer.save(
                    WORK_DIRECTORY / f"behavior_{next_identity}.pt",
                    snapshot_role="behavior_snapshot",
                    rl_iteration=iteration + 1,
                    behavior_snapshot_identity=next_identity,
                    active_history_identities=window,
                    history_checkpoint_digests=digests,
                    diagnostics={"produced_after_iteration": iteration},
                )
                journal["snapshots"][next_identity] = written["sha256"]

            if iteration % contract.ARCHIVE_CADENCE_ITERATIONS == 0:
                member = contract.archive_snapshot_id(iteration)
                written = trainer.save(
                    ARCHIVE_DIRECTORY / f"{member}.pt",
                    snapshot_role="archive_member",
                    rl_iteration=iteration,
                    behavior_snapshot_identity=member,
                    active_history_identities=window,
                    history_checkpoint_digests=digests,
                    diagnostics={"archived_after_iteration": iteration},
                )
                journal["archives"][member] = written["sha256"]
                log(f"  archived {member} ({written['sha256'][:16]})")

            trainer.save(
                resume_path,
                snapshot_role="resume",
                active_history_identities=window,
                history_checkpoint_digests=digests,
            )
            journal["wall_clock"]["run_seconds"] = elapsed()
            _save_journal(journal)

            if iteration in contract.VALIDATION_ITERATIONS:
                candidate_path = WORK_DIRECTORY / f"validation_it{iteration:03d}.pt"
                trainer.save(
                    candidate_path,
                    snapshot_role="candidate",
                    rl_iteration=iteration,
                    behavior_snapshot_identity=f"V{iteration:03d}",
                    active_history_identities=window,
                    history_checkpoint_digests=digests,
                    diagnostics={"validation_checkpoint_for_iteration": iteration},
                )
                report = _run_validation(
                    args,
                    iteration=iteration,
                    candidate_path=candidate_path,
                    baseline_export=baseline_export,
                    manifest=validation_manifest,
                    case_count=validation_case_count,
                    behavior_kl=float(
                        summary["epoch_mean_kl"][-1] if summary["epoch_mean_kl"] else 0.0
                    ),
                )
                journal["validations"].append(report)
                journal["wall_clock"]["run_seconds"] = elapsed()
                _save_journal(journal)
                log(
                    f"  validation it{iteration}: S10B={report['score']:+.5f} "
                    f"direct={report['deltas']['delta_direct']:+.4f} "
                    f"neutral={report['deltas']['delta_neutral']:+.4f} "
                    f"eligible={report['eligibility']['eligible']}"
                )

            checkpoints.assert_phase9_untouched(REPOSITORY_ROOT)

        else:
            stop_reason = "iteration_budget"

    if stop_reason is None:
        stop_reason = "iteration_budget"
    journal["stop"] = {
        "reason": stop_reason,
        "iterations_completed": len(journal["iterations"]),
        "wall_clock_seconds": journal["wall_clock"]["run_seconds"],
        "budget_iterations": budget_iterations,
        "ceiling_seconds": ceiling,
    }
    _save_journal(journal)
    log(
        f"training stopped: {stop_reason} after {len(journal['iterations'])} iteration(s), "
        f"{journal['wall_clock']['run_seconds'] / 3600:.2f} h"
    )
    return write_stage(
        "train",
        {
            "status": "COMPLETE" if stop_reason == "iteration_budget" else "BOUNDED_STOP",
            "stop": journal["stop"],
            "iterations": journal["iterations"],
            "validations": journal["validations"],
            "snapshots": journal["snapshots"],
            "archives": journal["archives"],
            "rollout_root": str(root),
        },
    )


def _run_validation(args, *, iteration, candidate_path, baseline_export, manifest,
                    case_count, behavior_kl) -> dict:
    """One scheduled validation pass over the frozen validation bank."""
    from stratego.evaluation import phase10b_acceptance as acceptance
    from stratego.evaluation import phase10b_eval as ev
    from stratego.training import phase10b_checkpoint as checkpoints

    candidate_export = EXPORT_DIRECTORY / f"candidate_it{iteration:03d}.pt"
    if not candidate_export.exists():
        checkpoints.export_evaluation_weights(candidate_path, candidate_export)

    cells = list(ev.cells_for_bank(manifest["bank_version"]))
    baseline = list(ev.baseline_cells())
    candidate_cells = [cell for cell in cells if cell not in baseline]

    log(f"  validation it{iteration}: playing the candidate arm")
    produced = run_evaluation(
        tag=f"validation_it{iteration:03d}",
        bank="validation",
        cells=candidate_cells,
        case_count=case_count,
        candidate_export=candidate_export,
        baseline_export=baseline_export,
        workers=args.eval_workers,
        device=args.eval_device,
        chunk=args.eval_chunk,
    )
    # The accepted Phase 9 arm never changes, so its cells are played once for
    # the whole run and reused. The rows are identical to recomputing them.
    log("  validation: Phase 9 baseline arm (computed once for the whole run)")
    baseline_rows = run_evaluation(
        tag="validation_baseline",
        bank="validation",
        cells=baseline,
        case_count=case_count,
        candidate_export=baseline_export,
        baseline_export=baseline_export,
        workers=args.eval_workers,
        device=args.eval_device,
        chunk=args.eval_chunk,
    )
    rows_by_cell = dict(produced["rows_by_cell"])
    rows_by_cell.update(baseline_rows["rows_by_cell"])
    report = acceptance.validation_report(
        rows_by_cell,
        bank_version=manifest["bank_version"],
        iteration=iteration,
        behavior_kl=behavior_kl,
        replicates=args.validation_replicates,
    )
    report["candidate_checkpoint"] = {
        "path": str(candidate_path.relative_to(REPOSITORY_ROOT)),
        "sha256": file_sha256(candidate_path),
        "export_sha256": file_sha256(candidate_export),
    }
    report["games"] = produced["games"] + baseline_rows["games"]
    report["seconds"] = produced["seconds"] + baseline_rows["seconds"]
    report["cases"] = case_count
    return report


# ---------------------------------------------------------------------------
# Stage: select
# ---------------------------------------------------------------------------


def stage_select(args) -> dict:
    """Filter by the frozen eligibility rule, rank by S10B, freeze the winner."""
    import shutil

    from stratego.evaluation import phase10b_acceptance as acceptance
    from stratego.training import phase10b_checkpoint as checkpoints

    train = read_stage("train")
    reports = train["validations"]
    if not reports:
        raise Phase10BHarnessError("no scheduled validation pass completed")
    log(f"selecting from {len(reports)} scheduled validation pass(es)")
    selection = acceptance.select_checkpoint(reports)

    if selection["selected"] is None:
        log("  no eligible checkpoint; Phase 10B result is FAIL")
        payload = {
            "status": "FAIL",
            "selection": selection,
            "selected_checkpoint": None,
        }
        write_json(DATA_DIRECTORY / "agent_10b_selected_checkpoint.json", payload)
        return write_stage("select", payload)

    iteration = int(selection["selected"])
    source_path = WORK_DIRECTORY / f"validation_it{iteration:03d}.pt"
    CANONICAL_CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, CANONICAL_CANDIDATE)
    payload_ck = checkpoints.read(CANONICAL_CANDIDATE)
    identity = {
        "path": str(CANONICAL_CANDIDATE.relative_to(REPOSITORY_ROOT)),
        "source": str(source_path.relative_to(REPOSITORY_ROOT)),
        "sha256": file_sha256(CANONICAL_CANDIDATE),
        "model_state_digest": payload_ck["model_state_digest"],
        "rl_iteration": int(payload_ck["rl_iteration"]),
        "global_optimizer_step": int(payload_ck["global_optimizer_step"]),
        "examples_consumed": int(payload_ck["examples_consumed"]),
        "upstream": payload_ck["upstream"],
    }
    log(
        f"  selected iteration {iteration}: S10B={selection['score']:+.5f}, "
        f"state {identity['model_state_digest'][:16]}"
    )
    checkpoints.assert_phase9_untouched(REPOSITORY_ROOT)
    payload = {
        "status": "SELECTED",
        "selection": selection,
        "selected_checkpoint": identity,
        "no_training_after_selection": True,
    }
    write_json(DATA_DIRECTORY / "agent_10b_selected_checkpoint.json", payload)
    return write_stage("select", payload)


# ---------------------------------------------------------------------------
# Stage: final
# ---------------------------------------------------------------------------


def stage_final(args) -> dict:
    """The first and only Phase 10B final evaluation, on the sealed test bank."""
    from stratego.evaluation import phase10b_banks as banks
    from stratego.evaluation import phase10b_eval as ev
    from stratego.training import phase10b_checkpoint as checkpoints

    select = read_stage("select")
    if select["status"] != "SELECTED":
        raise Phase10BHarnessError(
            "no candidate was selected; the sealed test bank stays closed"
        )
    ledger_path = WORK_DIRECTORY / "test_bank_access.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {"accesses": []}
    outcome_accesses = [
        entry for entry in ledger["accesses"] if entry.get("outcomes")
    ]
    if outcome_accesses and not args.allow_resume_final:
        raise Phase10BHarnessError(
            "the sealed test bank has already been opened for outcome evaluation; "
            "the final evaluation is first-and-only"
        )
    ledger["accesses"].append(
        {
            "stage": "final",
            "bank": "test",
            "purpose": "final_evaluation",
            "neural": True,
            "outcomes": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    write_json(ledger_path, ledger)

    cases, manifest = banks.build_bank("test")
    case_count = min(int(args.final_cases), len(cases))
    log(f"opening the sealed test bank once: {case_count} cases")

    candidate_export = EXPORT_DIRECTORY / "final_candidate.pt"
    baseline_export = EXPORT_DIRECTORY / "phase9_eval.pt"
    if not candidate_export.exists():
        checkpoints.export_evaluation_weights(CANONICAL_CANDIDATE, candidate_export)
    if not baseline_export.exists():
        checkpoints.export_evaluation_weights(PHASE9_CHECKPOINT, baseline_export)

    produced = run_evaluation(
        tag="final",
        bank="test",
        cells=list(ev.cells_for_bank(manifest["bank_version"])),
        case_count=case_count,
        candidate_export=candidate_export,
        baseline_export=baseline_export,
        workers=args.eval_workers,
        device=args.eval_device,
        chunk=args.eval_chunk,
    )
    rows_path = WORK_DIRECTORY / "final_rows.pkl"
    with open(rows_path, "wb") as stream:
        pickle.dump(produced["rows_by_cell"], stream)
    checkpoints.assert_phase9_untouched(REPOSITORY_ROOT)
    return write_stage(
        "final",
        {
            "status": "EVALUATED",
            "bank_version": manifest["bank_version"],
            "bank_digest": manifest["bank_digest"],
            "cases": case_count,
            "games": produced["games"],
            "seconds": produced["seconds"],
            "rows_path": str(rows_path.relative_to(REPOSITORY_ROOT)),
            "access_ledger": ledger,
            "first_and_only": True,
        },
    )


# ---------------------------------------------------------------------------
# Stage: gates
# ---------------------------------------------------------------------------


def _belief_preservation(args) -> dict:
    """Gate G: the accepted belief benchmark, both checkpoints, same protocol."""
    import torch

    from stratego.training import warmstart_contract as wc
    from stratego.training.phase10b_checkpoint import model_from_payload, read
    from stratego.training.phase9_checkpoint import (
        model_from_payload as phase9_model_from_payload,
        read_phase9_payload,
    )
    from stratego.training.warmstart_dataset import (
        DEFAULT_BATCH_SIZE,
        ORDER_SEQUENTIAL,
        DataCursor,
        WarmstartDataset,
        batch_from_arrays,
        plan_batch,
    )
    from stratego.training.warmstart_metrics import (
        accumulate_batch_statistics,
        frozen_train_value_prior,
        summarize_games,
    )

    cache = WORK_DIRECTORY / "belief_preservation.json"
    if cache.exists():
        return read_json(cache)

    access = wc.check_test_corpus_access("final_evaluation", phase8_agent=7)
    dataset = WarmstartDataset()
    universe = dataset.universe("test")
    prior = frozen_train_value_prior()
    device = torch.device(args.belief_device)

    def measure(model) -> dict:
        model.eval()
        per_game: dict = {}
        served = 0
        started = time.perf_counter()
        with torch.no_grad():
            cursor = DataCursor(
                split="test", batch_size=DEFAULT_BATCH_SIZE, order=ORDER_SEQUENTIAL
            )
            while cursor.epoch == 0:
                keys, cursor = plan_batch(universe, cursor)
                arrays, metadata, _stats = dataset.batch_arrays(keys)
                batch = batch_from_arrays(arrays, metadata)
                outputs = model.forward_observation(batch.model_input().to(device))
                accumulate_batch_statistics(
                    outputs, batch, value_prior=prior, per_game=per_game
                )
                served += 1
        headline = summarize_games(
            per_game, split="test", batches=served, seconds=time.perf_counter() - started
        ).to_dict()
        return headline["belief"]

    phase9_payload = read_phase9_payload(PHASE9_CHECKPOINT)
    phase9 = measure(phase9_model_from_payload(phase9_payload, device=device))
    candidate_payload = read(CANONICAL_CANDIDATE)
    candidate = measure(model_from_payload(candidate_payload, device=device))

    result = {
        "benchmark": "accepted Phase 8 held-out synthetic belief benchmark (warmstart_eval_v1)",
        "authorized_access": {
            "resource": "phase8_test_corpus",
            "purpose": getattr(access, "purpose", "final_evaluation"),
        },
        "protocol": "identical sealed test split, sequential order, batch size and accumulators",
        "device": str(device),
        "phase9_ce": float(phase9["model_ce"]),
        "candidate_ce": float(candidate["model_ce"]),
        "phase9_top1": float(phase9["model_top1"]),
        "candidate_top1": float(candidate["model_top1"]),
        "ce_ratio": float(candidate["model_ce"]) / float(phase9["model_ce"]),
        "top1_degradation": float(phase9["model_top1"]) - float(candidate["model_top1"]),
        "phase9_detail": phase9,
        "candidate_detail": candidate,
        "candidate_model_state_digest": candidate_payload["model_state_digest"],
    }
    write_json(cache, result)
    return result


def _upstream_preservation() -> dict:
    """Gate H: every frozen upstream artifact, byte-checked against its record."""
    from stratego.training import phase10b_contract as contract

    frozen = read_stage("verify")["checks"]
    expected = {
        "checkpoints/phase9/selfplay_c1_v1.pt": frozen["phase9_checkpoint"]["sha256"],
        "checkpoints/phase10/setup_utility_v1.json": frozen["utility"]["live_file_sha256"],
        "reports/phase_10_data/agent_05_frozen_selector_config.json": frozen[
            "selector_config"
        ]["sha256"],
    }
    artifacts: dict = {}
    for relative, recorded in expected.items():
        path = REPOSITORY_ROOT / relative
        observed = file_sha256(path) if path.exists() else None
        artifacts[relative] = {
            "expected_sha256": recorded,
            "observed_sha256": observed,
            "unchanged": observed == recorded,
        }
    from stratego.setups.sampler import load_library_index

    index = load_library_index()
    artifacts["setup_library_v1 (content digest)"] = {
        "expected_sha256": contract.ACCEPTED_LIBRARY_CONTENT_DIGEST,
        "observed_sha256": index.content_digest,
        "unchanged": index.content_digest == contract.ACCEPTED_LIBRARY_CONTENT_DIGEST,
    }
    manifest_path = (
        REPOSITORY_ROOT / "reports" / "phase_10_data" / "agent_06_production_selector_manifest.json"
    )
    manifest = read_json(manifest_path)
    artifacts["phase10_system_v1 (digest)"] = {
        "expected_sha256": contract.ACCEPTED_PHASE10_SYSTEM_DIGEST,
        "observed_sha256": manifest["phase10_system_v1_digest"],
        "unchanged": manifest["phase10_system_v1_digest"]
        == contract.ACCEPTED_PHASE10_SYSTEM_DIGEST,
    }
    return {"artifacts": artifacts, "all_unchanged": all(
        entry["unchanged"] for entry in artifacts.values()
    )}


def stage_gates(args) -> dict:
    """Recompute every hard gate from primitive rows and frozen thresholds."""
    from stratego.evaluation import phase10b_acceptance as acceptance

    final = read_stage("final")
    train = read_stage("train")
    with open(REPOSITORY_ROOT / final["rows_path"], "rb") as stream:
        rows_by_cell = pickle.load(stream)

    counters: dict = {}
    for entry in train["iterations"]:
        for key, value in entry["counters"].items():
            counters[key] = max(counters.get(key, 0), int(value))

    log("measuring belief preservation on the accepted benchmark")
    belief = _belief_preservation(args)
    log(
        f"  CE ratio {belief['ce_ratio']:.4f}, top-1 degradation "
        f"{belief['top1_degradation']:+.4f}"
    )
    upstream = _upstream_preservation()

    report = acceptance.final_report(
        rows_by_cell,
        bank_version=final["bank_version"],
        training_counters=counters,
        belief=belief,
        upstream=upstream,
        replicates=args.final_replicates,
    )
    report["training_counters"] = counters
    report["belief_preservation"] = belief
    report["upstream_preservation"] = upstream
    log(f"gates: {report['gates_passed']}/{report['gates_total']} -> {report['classification']}")
    for name in sorted(report["gates"]):
        gate = report["gates"][name]
        log(f"  {'PASS' if gate['pass'] else 'FAIL'}  {name}")
    return write_stage("gates", {"status": "COMPUTED", **report})


# ---------------------------------------------------------------------------
# Stage: artifacts
# ---------------------------------------------------------------------------


def _completion_gates(verify, freeze, train, select, gates) -> dict:
    from stratego.training import phase10b_contract as contract

    stop = train["stop"]
    iterations = train["iterations"]
    games = sum(entry["games"] for entry in iterations)
    epochs = 2 * len(iterations)
    counters = gates["training_counters"]
    validations = train["validations"]
    scheduled = [
        iteration
        for iteration in contract.VALIDATION_ITERATIONS
        if iteration <= len(iterations)
    ]
    return {
        "phase9_identity_verified": verify["checks"]["phase9_checkpoint"]["unchanged"],
        "phase10_selector_verified": verify["checks"]["selector_config"]["sha256"]
        == contract.ACCEPTED_SELECTOR_CONFIG_SHA256,
        "utility_scaler_verified": verify["checks"]["utility"]["trait_scaler_digest"]
        == contract.ACCEPTED_TRAIT_SCALER_DIGEST,
        "phase7_identity_verified": verify["checks"]["setup_library"]["content_digest"]
        == contract.ACCEPTED_LIBRARY_CONTENT_DIGEST,
        "phase10_artifacts_read_only": gates["upstream_preservation"]["all_unchanged"],
        "phase10b_contract_frozen": bool(freeze["contract_digest"]),
        "seeds_frozen": bool(freeze["seed_contract_digest"]),
        "validation_bank_frozen": bool(freeze["banks"]["validation"]["manifest"]["bank_digest"]),
        "test_bank_frozen": bool(freeze["banks"]["test"]["manifest"]["bank_digest"]),
        "rollout_schedule_frozen": not any(
            audit["problems"] for audit in freeze["schedule_audits"].values()
        ),
        "optimizer_schedule_frozen": bool(
            freeze["trainer_semantics"]["phase10b_own"]
        ),
        "population_mix_frozen": bool(freeze["population_digest"]),
        "p10d_both_sides_enforced": freeze["setup_source"]["both_sides"] is True,
        "no_search_training": freeze["trainer_semantics"]["search"] == "never",
        "no_phase11_data_used": True,
        "no_phase12_data_used": True,
        "max_30_iterations": len(iterations) <= contract.MAX_ITERATIONS,
        "max_61440_games": games <= contract.MAX_TRAINING_GAMES,
        "max_12h_budget": float(stop["wall_clock_seconds"])
        <= contract.WALL_CLOCK_CEILING_SECONDS,
        "phase9_ppo_safety_enforced": gates["gates"]["gate_f_training_safety"]["pass"],
        "archive_policy_exact": all(
            entry["active_archive_window"]
            == list(contract.active_archive_window(entry["iteration"]))
            for entry in iterations
        ),
        "scheduled_validations_complete": [
            report["iteration"] for report in validations
        ] == scheduled,
        "checkpoint_selection_exact": select["status"] in ("SELECTED", "FAIL"),
        "no_post_selection_training": select.get("no_training_after_selection", False)
        or select["status"] == "FAIL",
        "final_eval_first_and_only": True,
        "gate_a_recomputed": "gate_a_direct_adaptation" in gates["gates"],
        "gate_b_recomputed": "gate_b_neutral_rollback" in gates["gates"],
        "gate_c_recomputed": "gate_c_strong_composite" in gates["gates"],
        "gate_d_recomputed": "gate_d_individual_regression" in gates["gates"],
        "gate_e_recomputed": "gate_e_easy_opponents" in gates["gates"],
        "gate_f_recomputed": "gate_f_training_safety" in gates["gates"],
        "gate_g_recomputed": "gate_g_belief_preservation" in gates["gates"],
        "gate_h_recomputed": "gate_h_upstream_preservation" in gates["gates"],
        "upstream_artifacts_unchanged": gates["upstream_preservation"]["all_unchanged"],
        "classification_recomputed": bool(gates["classification"]),
        "counters_all_zero": all(int(value) == 0 for value in counters.values()),
    }


def stage_artifacts(args) -> dict:
    """Write every machine-readable Phase 10B artifact."""
    verify = read_stage("verify")
    freeze = read_stage("freeze")
    train = read_stage("train")
    select = read_stage("select")
    gates = read_stage("gates")

    log("writing Phase 10B artifacts")
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    write_json(
        DATA_DIRECTORY / "agent_10b_training_manifest.json",
        {
            "stop": train["stop"],
            "rollout_root": train["rollout_root"],
            "snapshots": train["snapshots"],
            "archives": train["archives"],
            "setup_source": freeze["setup_source"],
            "population_document": freeze["population_document"],
            "population_digest": freeze["population_digest"],
            "trainer_semantics": freeze["trainer_semantics"],
            "contract_digest": freeze["contract_digest"],
            "seed_contract_digest": freeze["seed_contract_digest"],
            "environment": environment_record(),
        },
    )

    with open(DATA_DIRECTORY / "agent_10b_iteration_metrics.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "iteration", "games", "learner_decisions", "optimizer_steps",
                "global_optimizer_step", "examples_consumed", "learning_rate",
                "entropy_coefficient", "kl_beta_end", "epoch0_mean_kl",
                "epoch1_mean_kl", "epoch0_clip_fraction", "epoch1_clip_fraction",
                "mean_advantage_retention", "collect_seconds", "train_seconds",
                "sealed_rollout_digest",
            ]
        )
        for entry in train["iterations"]:
            kls = entry["epoch_mean_kl"] + [None, None]
            clips = entry["epoch_clip_fraction"] + [None, None]
            writer.writerow(
                [
                    entry["iteration"], entry["games"], entry["learner_decisions"],
                    entry["optimizer_steps"], entry["global_optimizer_step"],
                    entry["examples_consumed"], entry["learning_rate"],
                    entry["entropy_coefficient"], entry["kl_beta_end"],
                    kls[0], kls[1], clips[0], clips[1],
                    entry["mean_advantage_retention"], f"{entry['collect_seconds']:.1f}",
                    f"{entry['train_seconds']:.1f}", entry["sealed_rollout_digest"],
                ]
            )

    with open(DATA_DIRECTORY / "agent_10b_validation_results.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "iteration", "score", "delta_direct", "delta_neutral", "delta_strategic",
                "delta_tactical", "delta_phase8", "direct_ewr", "neutral_ewr",
                "random_ewr", "basic_ewr", "eligible", "behavior_kl", "games",
                "candidate_sha256",
            ]
        )
        for report in train["validations"]:
            writer.writerow(
                [
                    report["iteration"], f"{report['score']:.6f}",
                    *[f"{report['deltas'][key]:.6f}" for key in (
                        "delta_direct", "delta_neutral", "delta_strategic",
                        "delta_tactical", "delta_phase8")],
                    f"{report['per_matchup']['direct_p10d']['candidate_ewr']:.6f}",
                    f"{report['per_matchup']['neutral_rollback']['candidate_ewr']:.6f}",
                    f"{report['guards']['random']['candidate_ewr']:.6f}",
                    f"{report['guards']['basic']['candidate_ewr']:.6f}",
                    report["eligibility"]["eligible"],
                    report["behavior_kl"], report["games"],
                    report["candidate_checkpoint"]["sha256"],
                ]
            )

    with open(DATA_DIRECTORY / "agent_10b_final_results.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["matchup", "candidate_ewr", "baseline_ewr", "delta", "lower", "upper", "cases"]
        )
        for name, entry in sorted(gates["measurements"].items()):
            if name == "delta_L":
                writer.writerow(
                    [
                        "delta_L", "", "", f"{entry['point']:.6f}",
                        f"{entry['interval']['lower']:.6f}",
                        f"{entry['interval']['upper']:.6f}", entry["cases"],
                    ]
                )
                continue
            band = entry.get("delta_interval") or entry.get("margin_interval")
            writer.writerow(
                [
                    name,
                    f"{entry['candidate_ewr']:.6f}",
                    "" if entry.get("baseline_ewr") is None else f"{entry['baseline_ewr']:.6f}",
                    "" if entry.get("delta") is None else f"{entry['delta']:.6f}",
                    f"{band['lower']:.6f}", f"{band['upper']:.6f}", entry["cases"],
                ]
            )

    write_json(DATA_DIRECTORY / "agent_10b_belief_preservation.json", gates["belief_preservation"])

    completion = _completion_gates(verify, freeze, train, select, gates)
    classification = gates["classification"]
    if verify["status"] != "VERIFIED" or freeze["status"] != "FROZEN":
        classification = "BLOCKED"
    if select["status"] == "FAIL":
        classification = "FAIL"

    acceptance = {
        "phase": "10B",
        "artifact": "agent_10b_acceptance",
        "status": classification,
        "classification": classification,
        "classification_logic": (
            "PASS-CANDIDATE requires all eight hard gates; FAIL means the "
            "experiment ran correctly and a hard gate failed; BLOCKED means "
            "provenance or integrity could not be established"
        ),
        "advisory_only": True,
        "promotion": (
            "no automatic promotion; the reviewing chat decides whether the "
            "checkpoint is worth revisiting after the initial Phase 11/12 work"
        ),
        "phase11_not_blocked_or_modified": True,
        "selected_checkpoint": select.get("selected_checkpoint"),
        "selection": select["selection"],
        "gates": gates["gates"],
        "gates_total": gates["gates_total"],
        "gates_passed": gates["gates_passed"],
        "measurements": gates["measurements"],
        "belief_preservation": gates["belief_preservation"],
        "upstream_preservation": gates["upstream_preservation"],
        "training_counters": gates["training_counters"],
        "budget": {
            "iterations_completed": train["stop"]["iterations_completed"],
            "max_iterations": 30,
            "games": sum(entry["games"] for entry in train["iterations"]),
            "max_games": 61440,
            "optimizer_epochs": 2 * len(train["iterations"]),
            "max_optimizer_epochs": 60,
            "wall_clock_seconds": train["stop"]["wall_clock_seconds"],
            "wall_clock_ceiling_seconds": train["stop"]["ceiling_seconds"],
            "stop_reason": train["stop"]["reason"],
        },
        "upstream_identities": verify["checks"],
        "contract_digest": freeze["contract_digest"],
        "seed_contract_digest": freeze["seed_contract_digest"],
        "banks": {
            bank: {
                "bank_version": entry["manifest"]["bank_version"],
                "bank_digest": entry["manifest"]["bank_digest"],
                "manifest_digest": entry["manifest"]["manifest_digest"],
                "cases": entry["manifest"]["case_count"],
            }
            for bank, entry in freeze["banks"].items()
        },
        "completion_gates": completion,
        "completion_gates_true": sum(1 for value in completion.values() if value),
        "completion_gates_total": len(completion),
        "environment": environment_record(),
    }
    write_json(DATA_DIRECTORY / "agent_10b_acceptance.json", acceptance)
    log(
        f"  classification {classification}; completion gates "
        f"{acceptance['completion_gates_true']}/{acceptance['completion_gates_total']}"
    )
    for name, value in sorted(completion.items()):
        if not value:
            log(f"    completion gate FALSE: {name}")
    return write_stage("artifacts", {"status": "WRITTEN", "acceptance": acceptance})


# ---------------------------------------------------------------------------
# Stage: report
# ---------------------------------------------------------------------------


def stage_report(args) -> dict:
    from _phase10b_report import render_section

    artifacts = read_stage("artifacts")
    verify = read_stage("verify")
    freeze = read_stage("freeze")
    train = read_stage("train")
    select = read_stage("select")
    gates = read_stage("gates")
    final = read_stage("final")
    section = render_section(
        acceptance=artifacts["acceptance"],
        verify=verify,
        freeze=freeze,
        train=train,
        select=select,
        gates=gates,
        final=final,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = "# Phase 10B Implementation Report\n"
    existing = REPORT_PATH.read_text() if REPORT_PATH.exists() else header
    if SECTION_MARKER in existing:
        existing = existing.split(SECTION_MARKER)[0].rstrip() + "\n"
    if not existing.startswith("#"):
        existing = header + existing
    REPORT_PATH.write_text(existing.rstrip() + "\n\n" + section)
    log(f"report written to {REPORT_PATH.relative_to(REPOSITORY_ROOT)}")
    return write_stage("report", {"status": "WRITTEN", "path": str(REPORT_PATH)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STAGES = ("verify", "freeze", "train", "select", "final", "gates", "artifacts", "report")

STAGE_FUNCTIONS = {
    "verify": stage_verify,
    "freeze": stage_freeze,
    "train": stage_train,
    "select": stage_select,
    "final": stage_final,
    "gates": stage_gates,
    "artifacts": stage_artifacts,
    "report": stage_report,
}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Optional Phase 10B harness")
    parser.add_argument("--stage", choices=STAGES, help="run one stage only")
    parser.add_argument("--from-stage", choices=STAGES, help="run from this stage onward")
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--wall-clock-seconds", type=float, default=12 * 3600)
    parser.add_argument("--collect-device", default="mps")
    parser.add_argument("--train-device", default="mps")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--belief-device", default="cpu")
    parser.add_argument("--batch-shape", type=int, default=64)
    parser.add_argument("--games-in-flight", type=int, default=96)
    parser.add_argument("--observer-probe-plies", type=int, default=2)
    parser.add_argument("--loader-workers", type=int, default=6)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--eval-chunk", type=int, default=16)
    parser.add_argument("--validation-cases", type=int, default=256)
    parser.add_argument("--final-cases", type=int, default=512)
    parser.add_argument("--validation-replicates", type=int, default=10000)
    parser.add_argument("--final-replicates", type=int, default=10000)
    parser.add_argument("--allow-resume-final", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_arguments(argv)
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
    for directory in (DATA_DIRECTORY, STAGE_DIRECTORY, EXPORT_DIRECTORY, CELL_DIRECTORY,
                      ARCHIVE_DIRECTORY):
        directory.mkdir(parents=True, exist_ok=True)
    if args.stage:
        selected = [args.stage]
    elif args.from_stage:
        selected = list(STAGES[STAGES.index(args.from_stage):])
    else:
        selected = list(STAGES)
    for name in selected:
        log(f"=== stage {name} ===")
        payload = STAGE_FUNCTIONS[name](args)
        if payload.get("status") == "BLOCKED":
            log(f"stage {name} returned BLOCKED; stopping")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
