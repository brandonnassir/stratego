#!/usr/bin/env python3
"""Phase 10 Agent 3 harness: fit the two utility models and audit them independently.

Verifies every Agent 1/Agent 2 prerequisite from live bytes (both PASS with
no false gate, all eight contract digests plus the bundle, the frozen
utility contract, the accepted Phase 9 checkpoint, the Phase 7 library, and
the SEALED corpus at its accepted content digest), reconstructs every trait
vector and the train-only standardizer independently, fits exactly Model F
and Model T once each under the frozen CPU float64 L-BFGS protocol, proves
deterministic refitting in independent processes, runs the independent
audit with its six negative controls, and writes:

    checkpoints/phase10/setup_utility_v1.json      (production coefficients + scaler)
    reports/phase_10_data/agent_03_utility_models.json
    reports/phase_10_data/agent_03_utility_audit.json
    reports/phase_10_data/agent_03_acceptance.json

What this script is and is not
------------------------------
It fits two models and audits the fit. It selects no candidate, compares no
model against another by any strength signal, evaluates no selector, plays
zero games, reads zero validation or test outcomes (neither bank stores an
outcome to read), and takes zero optimizer steps on the Phase 9 checkpoint,
which is hashed before fitting and again after everything else.

Usage::

    python scripts/run_phase10_agent03.py                   # every stage
    python scripts/run_phase10_agent03.py --stage verify    # one stage
    python scripts/run_phase10_agent03.py --run-pytest      # also the full suite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

AGENT = 3
PHASE = 10
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_10_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_10_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase10" / "agent03"
STAGE_DIRECTORY = WORK_DIRECTORY / "stages"
REFIT_DIRECTORY = WORK_DIRECTORY / "refits"

MODELS_ARTIFACT = DATA_DIRECTORY / "agent_03_utility_models.json"
AUDIT_ARTIFACT = DATA_DIRECTORY / "agent_03_utility_audit.json"
ACCEPTANCE_ARTIFACT = DATA_DIRECTORY / "agent_03_acceptance.json"

CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"

#: The report heading Agent 3 owns. Rewritten in place on every run.
SECTION_MARKER = "## 3. Agent 3 — Utility Models and Independent Fit Audit"

#: The accepted corpus identity Agent 2 sealed. A different digest is a
#: different corpus and a hard stop.
ACCEPTED_CORPUS_CONTENT_DIGEST = (
    "1977bb6f5e2611b0498c7976f6129718fdfe7f6f44216f3b3f1932c8192b3c50"
)

#: The full suite as measured immediately before any Phase 10 Agent 3 change.
TESTS_BEFORE = {
    "command": ".venv/bin/python -m pytest tests -q",
    "summary": "4964 passed, 3 skipped in 301.74s (0:05:01)",
    "passed": 4964,
    "failed": 0,
    "skipped": 3,
    "seconds": 301.74,
    "measured_at_commit": "6977584",
}

#: Every access this script makes to either sealed evaluation bank, with its
#: purpose. Agent 3 needs neither; the only entries are the structural
#: digest checks that prove the banks did not move. Neither bank stores a
#: game outcome, and this script plays no game on either.
BANK_ACCESS_LOG = (
    {
        "stage": "verify",
        "bank": "phase10_validation_bank_v1",
        "purpose": "digest_computation",
        "neural": False,
        "outcomes": False,
    },
    {
        "stage": "verify",
        "bank": "phase10_test_bank_v1",
        "purpose": "digest_computation",
        "neural": False,
        "outcomes": False,
    },
)

#: Number of refit processes per model, beyond the canonical in-process fit.
REFIT_PROCESSES = 2


class Agent3Error(RuntimeError):
    """A precondition or frozen identity failed. Always raised, never patched."""


# ---------------------------------------------------------------------------
# Environment and helpers
# ---------------------------------------------------------------------------


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
        return "unknown"


def torch_report() -> dict:
    import torch

    return {
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
    }


def environment_report() -> dict:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "source_revision": _git("rev-parse", "--short", "HEAD"),
        "working_tree_state": "dirty" if _git("status", "--porcelain") else "clean",
        **torch_report(),
    }


def file_sha256(path: Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stage_path(name: str) -> Path:
    return STAGE_DIRECTORY / f"{name}.json"


def save_stage(name: str, payload: dict) -> dict:
    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stage_path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def load_stage(name: str) -> dict:
    path = stage_path(name)
    if not path.exists():
        raise Agent3Error(
            f"stage {name!r} has not run; run `--stage {name}` first or run all stages"
        )
    return json.loads(path.read_text())


def require(condition: bool, message: str, problems: list) -> bool:
    if not condition:
        problems.append(message)
    return bool(condition)


def log(message: str) -> None:
    print(f"[agent3] {message}", flush=True)


def corpus_root() -> Path:
    from stratego.training import phase10_storage as storage

    check = storage.check_corpus_root()
    if not check["usable"]:
        raise Agent3Error(f"corpus root is unusable (BLOCKED): {check['blocked']}")
    return Path(check["resolved_root"])


def read_corpus_records():
    """`(reader, records)` for the sealed corpus, in canonical order."""
    from stratego.training import phase10_outcome_store as store

    root = corpus_root()
    if store.read_state(root) != store.STATE_SEALED:
        raise Agent3Error(f"{root} is not SEALED; Agent 3 fits only a sealed corpus")
    reader = store.OutcomeReader(root)
    records = list(reader.iter_records())
    return reader, records


# ---------------------------------------------------------------------------
# Stage: verify
# ---------------------------------------------------------------------------


def verify_agent(agent: int, problems: list) -> dict:
    """One upstream acceptance artifact: PASS, with no false completion gate."""
    path = DATA_DIRECTORY / f"agent_0{agent}_acceptance.json"
    if not path.exists():
        raise Agent3Error(f"{path} is missing; Agent 3 cannot start (BLOCKED)")
    payload = json.loads(path.read_text())
    gates = payload.get("completion_gates", {})
    false_gates = sorted(name for name, value in gates.items() if not value)
    require(payload.get("status") == "PASS", f"Agent {agent} status is {payload.get('status')!r}", problems)
    require(not false_gates, f"Agent {agent} has false completion gates: {false_gates}", problems)
    return {
        "artifact": str(path.relative_to(REPOSITORY_ROOT)),
        "status": payload.get("status"),
        "gates_total": payload.get("gates_total"),
        "gates_true": payload.get("gates_true"),
        "false_gates": false_gates,
    }


def verify_agent2_handoff(problems: list) -> dict:
    payload = json.loads((DATA_DIRECTORY / "agent_02_acceptance.json").read_text())
    handoff = payload.get("handoff_to_agent_3", {})
    require(
        handoff.get("corpus_content_digest") == ACCEPTED_CORPUS_CONTENT_DIGEST,
        "Agent 2 handoff names a different corpus content digest",
        problems,
    )
    require(handoff.get("corpus_state") == "SEALED", "Agent 2 handoff is not SEALED", problems)
    leak = handoff.get("proof_no_leak", {})
    require(
        int(leak.get("validation_bank_outcome_access", -1)) == 0
        and int(leak.get("test_bank_outcome_access", -1)) == 0,
        "Agent 2 records bank outcome access",
        problems,
    )
    return {
        "corpus_content_digest": handoff.get("corpus_content_digest"),
        "corpus_state": handoff.get("corpus_state"),
        "replay_evidence": handoff.get("replay_evidence"),
        "deviations_recorded": len(payload.get("deviations", [])),
    }


def verify_contract_digests(problems: list) -> dict:
    """Every Phase 10 contract, bank and schedule digest, recomputed live."""
    from stratego.evaluation import phase10_banks as banks
    from stratego.training import phase10_contract as contract
    from stratego.training.phase10_schedule import schedule_digest
    from tests.training import phase10_frozen_digests as pin

    observed = contract.contract_digests()
    mismatched = sorted(
        name for name, value in pin.CONTRACT_DIGESTS.items() if observed.get(name) != value
    )
    require(not mismatched, f"Phase 10 contract digests moved: {mismatched}", problems)
    bundle = contract.contract_bundle_digest()
    require(bundle == pin.CONTRACT_BUNDLE_DIGEST, "contract bundle digest moved", problems)
    schedule = schedule_digest()
    require(schedule == pin.OUTCOME_SCHEDULE_DIGEST, "outcome schedule digest moved", problems)

    isolation, isolation_manifest = banks.phase9_isolation_set()
    require(
        isolation_manifest["set_digest"] == pin.PHASE9_ISOLATION_SET_DIGEST,
        "Phase 9 isolation set digest moved",
        problems,
    )
    bank_digests = {}
    for split in ("validation", "test"):
        cases, manifest = banks.build_phase10_bank(split, isolation, isolation_manifest)
        observed_bank = banks.bank_digest(cases)
        observed_manifest = banks.manifest_digest(manifest)
        bank_digests[split] = {
            "bank_digest": observed_bank,
            "manifest_digest": observed_manifest,
            "cases": len(cases),
        }
        require(observed_bank == pin.BANK_DIGESTS[split], f"{split} bank digest moved", problems)
        require(
            observed_manifest == pin.BANK_MANIFEST_DIGESTS[split],
            f"{split} bank manifest digest moved",
            problems,
        )
    return {
        "contract_digests": observed,
        "contract_bundle_digest": bundle,
        "outcome_schedule_digest": schedule,
        "banks": bank_digests,
        "bank_access_log": [dict(entry) for entry in BANK_ACCESS_LOG],
        "bank_neural_outcome_access": 0,
    }


def verify_utility_contract(problems: list) -> dict:
    """The live utility contract equals Agent 1's frozen artifact copy."""
    from stratego.training.phase10_utility import fit_trait_scaler, utility_contract_document
    from stratego.training.phase10_utility_fit import ACCEPTED_TRAIT_SCALER_DIGEST
    from tests.training import phase10_frozen_digests as pin

    live = utility_contract_document()
    frozen_bundle = json.loads(
        (DATA_DIRECTORY / "agent_01_setup_selection_contract.json").read_text()
    )
    frozen = frozen_bundle["contracts"]["phase10_setup_utility_v1"]
    require(live == frozen, "live utility contract differs from Agent 1's frozen copy", problems)
    scaler_digest = fit_trait_scaler().digest()
    require(scaler_digest == pin.TRAIT_SCALER_DIGEST, "trait scaler digest moved", problems)
    require(
        ACCEPTED_TRAIT_SCALER_DIGEST == pin.TRAIT_SCALER_DIGEST,
        "the fit module pins a different scaler digest than the freeze",
        problems,
    )
    return {
        "utility_contract_matches_agent1": live == frozen,
        "fit_protocol": dict(live["fit_protocol"]),
        "scaler_digest": scaler_digest,
        "feature_count": live["feature"]["feature_count"],
        "models": [entry["model_id"] for entry in live["models"]],
    }


def verify_phase9_checkpoint(problems: list, *, label: str) -> dict:
    """File SHA, model-state digest, parameter count and finiteness, live."""
    import torch

    from stratego.model.architecture_configs import config_digests
    from stratego.training import phase10_contract as pc
    from stratego.training import phase9_behavior, phase9_checkpoint

    observed_sha = file_sha256(CHECKPOINT_PATH)
    payload = phase9_checkpoint.read_phase9_payload(CHECKPOINT_PATH)
    model = phase9_checkpoint.model_from_payload(payload)
    state_digest = phase9_behavior.state_dict_digest(model)
    parameters = sum(tensor.numel() for tensor in model.parameters())
    finite = all(bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values())
    c1_digest = config_digests()["C1"]

    require(observed_sha == pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256, f"[{label}] Phase 9 SHA moved", problems)
    require(
        state_digest == pc.ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
        f"[{label}] Phase 9 model-state digest moved",
        problems,
    )
    require(parameters == pc.ACCEPTED_PHASE9_PARAMETERS, f"[{label}] Phase 9 parameter count moved", problems)
    require(finite, f"[{label}] Phase 9 model carries a non-finite parameter", problems)
    require(c1_digest == pc.ACCEPTED_C1_CONFIG_DIGEST, f"[{label}] C1 config digest moved", problems)
    del model, payload
    return {
        "label": label,
        "path": str(CHECKPOINT_PATH.relative_to(REPOSITORY_ROOT)),
        "sha256": observed_sha,
        "model_state_digest": state_digest,
        "parameters": int(parameters),
        "all_parameters_finite": bool(finite),
        "c1_config_digest": c1_digest,
        "c1_optimizer_steps": 0,
    }


def verify_phase7_library(problems: list) -> dict:
    from collections import Counter

    from stratego.setups.sampler import load_library_index
    from stratego.training import phase10_contract as pc

    index = load_library_index()
    counts = Counter(entry.split for entry in index.entries)
    require(index.content_digest == pc.PHASE7_LIBRARY_CONTENT_DIGEST, "Phase 7 library digest moved", problems)
    require(
        (counts.get("train"), counts.get("validation"), counts.get("test")) == (6400, 800, 800),
        f"library splits are {dict(counts)}",
        problems,
    )
    return {"content_digest": index.content_digest, "splits": dict(counts)}


def verify_corpus(problems: list) -> dict:
    """The sealed corpus: mounted, SEALED, digest-exact, structurally exact."""
    from stratego.training import phase10_outcome_store as store
    from stratego.training import phase10_storage as storage

    check = storage.check_corpus_root()
    require(check["usable"], f"corpus root unusable: {check['blocked']}", problems)
    root = Path(check["resolved_root"])
    state = store.read_state(root)
    require(state == store.STATE_SEALED, f"corpus state is {state!r}, not SEALED", problems)
    seal = store.verify_seal(root)
    require(seal["all_pass"], f"corpus seal verification failed: {seal['checks']}", problems)
    require(
        seal["observed_content_digest"] == ACCEPTED_CORPUS_CONTENT_DIGEST,
        f"corpus content digest {seal['observed_content_digest']} is not the accepted one",
        problems,
    )
    require(
        seal["observed_committed_games"] == 16384,
        f"corpus holds {seal['observed_committed_games']} games",
        problems,
    )
    integrity = store.audit_store_integrity(root)
    require(integrity["all_pass"], f"corpus integrity audit failed: {integrity['checks']}", problems)
    return {
        "resolved_root": str(root),
        "storage": {key: check[key] for key in ("external_volume", "external_volume_mounted")},
        "state": state,
        "content_digest": seal["observed_content_digest"],
        "committed_games": seal["observed_committed_games"],
        "integrity_checks": integrity["checks"],
    }


def stage_verify(_args) -> dict:
    problems: list = []
    log("verifying Agents 1-2, contracts, utility freeze, checkpoint, library and corpus")
    payload = {
        "stage": "verify",
        "environment": environment_report(),
        "agent1": verify_agent(1, problems),
        "agent2": verify_agent(2, problems),
        "agent2_handoff": verify_agent2_handoff(problems),
        "contracts": verify_contract_digests(problems),
        "utility_contract": verify_utility_contract(problems),
        "phase9_before": verify_phase9_checkpoint(problems, label="before"),
        "phase7_library": verify_phase7_library(problems),
        "corpus": verify_corpus(problems),
        "problems": problems,
        "all_pass": not problems,
    }
    if problems:
        save_stage("verify", payload)
        raise Agent3Error(f"verification failed (BLOCKED): {problems}")
    log("verify: all prerequisites hold")
    return save_stage("verify", payload)


# ---------------------------------------------------------------------------
# Stage: features — independent reconstruction and the standardizer
# ---------------------------------------------------------------------------


def frozen_scaler_literals() -> dict:
    """The frozen scaler's literal mean/std, read from Agent 1's artifact bytes."""
    frozen_bundle = json.loads(
        (DATA_DIRECTORY / "agent_01_setup_selection_contract.json").read_text()
    )
    return frozen_bundle["contracts"]["phase10_setup_utility_v1"]["scaler"]


def stage_features(_args) -> dict:
    import numpy as np

    from stratego.setups.contracts import LIBRARY_JSONL_PATH
    from stratego.setups.library import read_library_jsonl
    from stratego.training import phase10_utility_audit as audit
    from stratego.training.phase10_utility import (
        TRAIT_FEATURE_NAMES,
        fit_trait_scaler,
    )
    from tests.training import phase10_frozen_digests as pin

    problems: list = []
    log("reconstructing every trait vector from placements and re-deriving the scaler")

    entries = read_library_jsonl(LIBRARY_JSONL_PATH)
    library = audit.reconstruct_library(entries)
    require(
        library.stored_trait_mismatches == (),
        f"stored trait vectors disagree with recomputation: "
        f"{library.stored_trait_mismatches[:8]}",
        problems,
    )
    require(
        library.feature_names == TRAIT_FEATURE_NAMES,
        "independent flattening order differs from the frozen feature names",
        problems,
    )
    require(library.train_matrix.shape == (6400, 47), "train matrix shape drifted", problems)

    frozen = frozen_scaler_literals()
    frozen_mean = np.asarray(frozen["mean"], dtype=np.float64)
    frozen_std = np.asarray(frozen["std"], dtype=np.float64)
    mean, std = audit.independent_scaler_moments(library.train_matrix)
    mean_exact = bool(np.array_equal(mean, frozen_mean))
    std_exact = bool(np.array_equal(std, frozen_std))
    require(mean_exact, "independent train mean differs from the frozen literals", problems)
    require(std_exact, "independent train std differs from the frozen literals", problems)
    require(frozen["zero_std_features"] == [], "frozen scaler records zero-std features", problems)

    production_digest = fit_trait_scaler().digest()
    require(production_digest == pin.TRAIT_SCALER_DIGEST, "production scaler digest moved", problems)

    log("auditing all 16,384 corpus records against the reconstruction")
    _reader, records = read_corpus_records()
    from stratego.training import phase10_contract as contract
    from stratego.training.phase10_schedule import corpus_contract_document, schedule_digest

    expected_digests = {
        "library_content_digest": contract.PHASE7_LIBRARY_CONTENT_DIGEST,
        "corpus_contract_digest": contract.document_digest(corpus_contract_document()),
        "outcome_schedule_digest": schedule_digest(),
        "contract_bundle_digest": contract.contract_bundle_digest(),
    }
    record_audit, _design = audit.audit_corpus_records(
        records,
        library,
        expected_digests=expected_digests,
        frozen_mean=frozen_mean,
        frozen_std=frozen_std,
    )
    require(record_audit["all_pass"], f"record audit failed: {record_audit['checks']}", problems)

    unique_bases = sorted(
        {record["red_base_setup_id"] for record in records}
        | {record["blue_base_setup_id"] for record in records}
    )
    base_splits = {library.base_split[base_id] for base_id in unique_bases}
    require(base_splits == {"train"}, f"corpus bases span splits {sorted(base_splits)}", problems)

    payload = {
        "stage": "features",
        "library_entries_reconstructed": len(entries),
        "stored_trait_mismatches": 0 if not library.stored_trait_mismatches else len(library.stored_trait_mismatches),
        "feature_names_match_frozen": library.feature_names == TRAIT_FEATURE_NAMES,
        "train_matrix_shape": list(library.train_matrix.shape),
        "independent_scaler": {
            "mean_matches_frozen_exactly": mean_exact,
            "std_matches_frozen_exactly": std_exact,
            "ddof": 0,
            "base_count": int(library.train_matrix.shape[0]),
            "split": "train",
        },
        "production_scaler_digest": production_digest,
        "record_audit": record_audit,
        "unique_corpus_bases": len(unique_bases),
        "unique_corpus_base_splits": sorted(base_splits),
        "trait_identity_digests_verified": record_audit["records_audited"] * 2,
        "problems": problems,
        "all_pass": not problems,
    }
    if problems:
        save_stage("features", payload)
        raise Agent3Error(f"feature reconstruction failed: {problems}")
    log(
        f"features: {len(entries)} entries reconstructed, "
        f"{len(unique_bases)} unique corpus bases, scaler exact"
    )
    return save_stage("features", payload)


# ---------------------------------------------------------------------------
# Stage: fit — the two canonical fits and the exported artifact
# ---------------------------------------------------------------------------


def canonical_fits() -> dict:
    """Both models fit once, in this process, under the frozen protocol."""
    from stratego.training import phase10_utility_fit as fit
    from stratego.training.phase10_utility import fit_trait_scaler

    _reader, records = read_corpus_records()
    scaler = fit_trait_scaler()
    results = {}
    for model_id in ("model_F", "model_T"):
        log(f"fitting {model_id} from the all-zero initialisation")
        started = time.perf_counter()
        data = fit.build_fit_data(records, model_id, scaler=scaler)
        fitted = fit.fit_utility_model(data)
        results[model_id] = {
            "fitted": fitted,
            "seconds": time.perf_counter() - started,
            "accessed_fields": list(data.accessed_fields),
            "games": data.game_count,
        }
        log(
            f"{model_id}: objective {fitted.diagnostics['objective']:.6f} "
            f"({fitted.diagnostics['iterations']} iterations, "
            f"grad max {fitted.diagnostics['final_grad_max_abs']:.2e})"
        )
    return {"records": records, "scaler": scaler, "results": results}


def stage_fit(_args) -> dict:
    from stratego.training import phase10_utility_fit as fit

    outcome = canonical_fits()
    scaler = outcome["scaler"]
    models = {model_id: entry["fitted"] for model_id, entry in outcome["results"].items()}

    artifact = fit.utility_models_artifact(
        models,
        scaler,
        corpus_content_digest=ACCEPTED_CORPUS_CONTENT_DIGEST,
        corpus_games=16384,
    )
    findings = fit.own_side_only_findings(artifact)
    if findings:
        raise Agent3Error(f"exported artifact is not own-side only: {findings}")
    artifact_path = REPOSITORY_ROOT / fit.FITTED_UTILITY_RELATIVE_PATH
    artifact_sha = fit.write_utility_models_artifact(artifact, artifact_path)
    log(f"wrote {artifact_path.relative_to(REPOSITORY_ROOT)} ({artifact_sha[:12]}...)")

    payload = {
        "stage": "fit",
        "artifact_path": str(artifact_path.relative_to(REPOSITORY_ROOT)),
        "artifact_sha256": artifact_sha,
        "scaler_digest": scaler.digest(),
        "models": {
            model_id: {
                **models[model_id].to_dict(),
                "fit_seconds": outcome["results"][model_id]["seconds"],
            }
            for model_id in models
        },
        "fitting_input_allowlist": {
            model_id: list(fields) for model_id, fields in fit.FIT_INPUT_ALLOWLIST.items()
        },
        "forbidden_fitting_fields": list(fit.FORBIDDEN_FIT_FIELDS),
        "accessed_fields": {
            model_id: outcome["results"][model_id]["accessed_fields"] for model_id in models
        },
    }
    return save_stage("fit", payload)


# ---------------------------------------------------------------------------
# Stage: refit — deterministic refits in independent processes
# ---------------------------------------------------------------------------


def stage_refit_worker(args) -> dict:
    """One independent-process refit: fit one model, write its coefficients."""
    from stratego.training import phase10_utility_fit as fit
    from stratego.training.phase10_utility import fit_trait_scaler

    _reader, records = read_corpus_records()
    data = fit.build_fit_data(records, args.model, scaler=fit_trait_scaler())
    fitted = fit.fit_utility_model(data)
    payload = {
        "model_id": args.model,
        "process_id": os.getpid(),
        "coefficient_document": fitted.coefficient_document(),
        "coefficient_digest": fitted.coefficient_digest(),
        "diagnostics": fitted.diagnostics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def stage_refit(_args) -> dict:
    from stratego.training import phase10_utility_audit as audit

    fit_stage = load_stage("fit")
    REFIT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    comparisons = {}
    for model_id in ("model_F", "model_T"):
        documents = [
            {
                key: fit_stage["models"][model_id][key]
                for key in (
                    "utility_version",
                    "model_id",
                    "colour_order",
                    "family_order",
                    "feature_order",
                    "red_first_intercept",
                    "family_offsets_raw",
                    "trait_weights",
                )
            }
        ]
        digests = [fit_stage["models"][model_id]["coefficient_digest"]]
        objectives = [fit_stage["models"][model_id]["diagnostics"]["objective"]]
        worker_pids = []
        for replicate in range(REFIT_PROCESSES):
            output = REFIT_DIRECTORY / f"{model_id}_refit_{replicate}.json"
            if output.exists():
                output.unlink()
            log(f"refitting {model_id} in an independent process ({replicate + 1}/{REFIT_PROCESSES})")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--stage",
                    "refit-worker",
                    "--model",
                    model_id,
                    "--output",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise Agent3Error(
                    f"refit worker failed for {model_id}: {completed.stderr[-2000:]}"
                )
            worker = json.loads(output.read_text())
            documents.append(worker["coefficient_document"])
            digests.append(worker["coefficient_digest"])
            objectives.append(worker["diagnostics"]["objective"])
            worker_pids.append(worker["process_id"])

        comparison = audit.compare_refits(documents)
        if not comparison["identical"]:
            raise Agent3Error(
                f"{model_id}: independent refits are not bit-identical "
                f"(max abs difference {comparison['max_abs_difference']})"
            )
        objective_spread = max(objectives) - min(objectives)
        comparisons[model_id] = {
            **comparison,
            "coefficient_digests": digests,
            "digests_identical": len(set(digests)) == 1,
            "objectives": objectives,
            "objective_spread": objective_spread,
            "worker_process_ids": worker_pids,
            "canonical_process_id": os.getpid(),
        }
        log(f"{model_id}: {comparison['fits']} fits bit-identical, objective spread {objective_spread}")

    payload = {
        "stage": "refit",
        "processes_per_model": 1 + REFIT_PROCESSES,
        "independent_processes": REFIT_PROCESSES,
        "comparisons": comparisons,
        "all_identical": all(entry["identical"] and entry["digests_identical"] for entry in comparisons.values()),
    }
    if not payload["all_identical"]:
        raise Agent3Error("deterministic refit failed")
    return save_stage("refit", payload)


# ---------------------------------------------------------------------------
# Stage: audit — the independent audit and its negative controls
# ---------------------------------------------------------------------------


def stage_audit(_args) -> dict:
    import numpy as np

    from stratego.setups.contracts import LIBRARY_JSONL_PATH
    from stratego.setups.library import read_library_jsonl
    from stratego.training import phase10_contract as contract
    from stratego.training import phase10_utility_audit as audit
    from stratego.training import phase10_utility_fit as fit
    from stratego.training.phase10_schedule import corpus_contract_document, schedule_digest
    from stratego.training.phase10_utility import fit_trait_scaler
    from tests.training import phase10_frozen_digests as pin

    problems: list = []
    fit_stage = load_stage("fit")
    artifact_path = REPOSITORY_ROOT / fit_stage["artifact_path"]
    artifact = json.loads(artifact_path.read_text())
    require(
        file_sha256(artifact_path) == fit_stage["artifact_sha256"],
        "exported artifact bytes moved between fit and audit",
        problems,
    )

    log("rebuilding the audit design independently of the production fit helper")
    entries = read_library_jsonl(LIBRARY_JSONL_PATH)
    library = audit.reconstruct_library(entries)
    frozen = frozen_scaler_literals()
    frozen_mean = np.asarray(frozen["mean"], dtype=np.float64)
    frozen_std = np.asarray(frozen["std"], dtype=np.float64)

    _reader, records = read_corpus_records()
    expected_digests = {
        "library_content_digest": contract.PHASE7_LIBRARY_CONTENT_DIGEST,
        "corpus_contract_digest": contract.document_digest(corpus_contract_document()),
        "outcome_schedule_digest": schedule_digest(),
        "contract_bundle_digest": contract.contract_bundle_digest(),
    }
    record_audit, design = audit.audit_corpus_records(
        records,
        library,
        expected_digests=expected_digests,
        frozen_mean=frozen_mean,
        frozen_std=frozen_std,
    )
    require(record_audit["all_pass"], f"record audit failed: {record_audit['checks']}", problems)

    log("comparing the audit design against the production design")
    scaler = fit_trait_scaler()
    design_agreement = {}
    for model_id in ("model_F", "model_T"):
        data = fit.build_fit_data(records, model_id, scaler=scaler)
        agreement = {
            "targets_exact": bool(np.array_equal(data.targets, design.targets)),
            "red_families_exact": bool(np.array_equal(data.red_family_index, design.red_family_index)),
            "blue_families_exact": bool(np.array_equal(data.blue_family_index, design.blue_family_index)),
            "game_ids_exact": data.game_ids == design.game_ids,
        }
        if model_id == "model_T":
            agreement["red_features_max_abs_difference"] = float(
                np.abs(data.red_features - design.red_features).max()
            )
            agreement["blue_features_max_abs_difference"] = float(
                np.abs(data.blue_features - design.blue_features).max()
            )
            agreement["features_exact"] = bool(
                agreement["red_features_max_abs_difference"] == 0.0
                and agreement["blue_features_max_abs_difference"] == 0.0
            )
        design_agreement[model_id] = agreement
        require(
            all(value for key, value in agreement.items() if key.endswith("_exact")),
            f"{model_id}: audit design disagrees with the production design",
            problems,
        )

    log("running the independent model audits")
    model_audits = {}
    for model_id in ("model_F", "model_T"):
        fitted = artifact["models"][model_id]
        outcome = audit.audit_fitted_model(design, fitted)
        model_audits[model_id] = outcome
        require(outcome["all_pass"], f"{model_id}: independent audit failed: {outcome['checks']}", problems)

    log("firing the six negative controls")
    fitted_t = artifact["models"]["model_T"]
    fitted_f = artifact["models"]["model_F"]
    controls = [
        audit.control_orientation_swap(design, fitted_t),
        {
            **audit.control_orientation_swap(design, fitted_f),
            "control": "orientation_swap_model_F",
        },
        audit.control_wrong_draw_target(design, fitted_t, records),
        audit.control_held_out_scaler(
            library,
            entries,
            frozen_mean=frozen_mean,
            frozen_std=frozen_std,
            frozen_digest=pin.TRAIT_SCALER_DIGEST,
        ),
        audit.control_permuted_trait_column(design, fitted_t),
        audit.control_altered_family_id(
            records,
            library,
            expected_digests=expected_digests,
            frozen_mean=frozen_mean,
            frozen_std=frozen_std,
        ),
        audit.control_altered_coefficient(design, fitted_t),
    ]
    for control in controls:
        require(control["detected"], f"negative control did not fire: {control['control']}", problems)

    log("proving the production scorer is own-side only and decomposes the logit")
    findings = fit.own_side_only_findings(artifact)
    require(not findings, f"artifact own-side findings: {findings}", problems)
    from stratego.setups.sampler import load_library_index

    index = load_library_index()
    scorer = fit.SetupUtilityScorer(artifact)
    sample_positions = list(range(0, len(records), 1024))
    decomposition_max_abs = 0.0
    for model_id in ("model_F", "model_T"):
        fitted = artifact["models"][model_id]
        eta = audit.audit_logits(design, fitted)
        intercept = float(fitted["red_first_intercept"])
        for position in sample_positions:
            record = records[position]
            red_trait = blue_trait = None
            if model_id == "model_T":
                red_trait = index.base(record["red_base_setup_id"]).trait_vector
                blue_trait = index.base(record["blue_base_setup_id"]).trait_vector
            red_utility = scorer.utility(model_id, "red", record["red_family"], red_trait)
            blue_utility = scorer.utility(model_id, "blue", record["blue_family"], blue_trait)
            decomposition_max_abs = max(
                decomposition_max_abs,
                abs((intercept + red_utility - blue_utility) - float(eta[position])),
            )
    require(
        decomposition_max_abs <= audit.LOGIT_AGREEMENT_TOLERANCE,
        f"scorer decomposition disagrees with audit logits by {decomposition_max_abs}",
        problems,
    )

    payload = {
        "stage": "audit",
        "record_audit": record_audit,
        "design_agreement": design_agreement,
        "model_audits": model_audits,
        "negative_controls": controls,
        "negative_controls_fired": sum(1 for control in controls if control["detected"]),
        "own_side_findings": findings,
        "scorer_decomposition_max_abs_difference": decomposition_max_abs,
        "scorer_decomposition_samples": len(sample_positions) * 2,
        "problems": problems,
        "all_pass": not problems,
    }
    if problems:
        save_stage("audit", payload)
        raise Agent3Error(f"independent audit failed: {problems}")
    log(f"audit: all checks pass, {payload['negative_controls_fired']}/{len(controls)} controls fired")
    return save_stage("audit", payload)


# ---------------------------------------------------------------------------
# Stage: artifacts — acceptance, artifacts, report
# ---------------------------------------------------------------------------


def run_pytest() -> dict:
    log("running the full suite")
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    seconds = time.perf_counter() - started
    tail = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    summary = tail[-1] if tail else "no output"
    import re

    def count(token: str) -> int:
        match = re.search(rf"(\d+) {token}", summary)
        return int(match.group(1)) if match else 0

    return {
        "command": ".venv/bin/python -m pytest tests -q",
        "returncode": completed.returncode,
        "summary": summary,
        "passed": count("passed"),
        "failed": count("failed"),
        "skipped": count("skipped"),
        "seconds": round(seconds, 2),
    }


def completion_gates(verify, features, fit_stage, refit, audit_stage, after, suite) -> dict:
    """The 19 Agent 3 completion gates, every one recomputed from evidence."""
    model_audits = audit_stage["model_audits"]
    controls = audit_stage["negative_controls"]
    corpus_after = after["corpus_after"]

    def diagnostics(model_id: str) -> dict:
        return fit_stage["models"][model_id]["diagnostics"]

    return {
        "agents1_2_pass": bool(
            verify["agent1"]["status"] == "PASS"
            and verify["agent2"]["status"] == "PASS"
            and not verify["agent1"]["false_gates"]
            and not verify["agent2"]["false_gates"]
        ),
        "corpus_digest_verified": bool(
            verify["corpus"]["content_digest"] == ACCEPTED_CORPUS_CONTENT_DIGEST
            and corpus_after["content_digest"] == ACCEPTED_CORPUS_CONTENT_DIGEST
            and corpus_after["state"] == "SEALED"
        ),
        "corpus_train_only": bool(
            features["unique_corpus_base_splits"] == ["train"]
            and features["record_audit"]["all_pass"]
        ),
        "trait_vectors_reconstructed": bool(
            features["stored_trait_mismatches"] == 0
            and features["feature_names_match_frozen"]
            and features["record_audit"]["all_pass"]
        ),
        "standardizer_train_only": bool(
            features["independent_scaler"]["mean_matches_frozen_exactly"]
            and features["independent_scaler"]["std_matches_frozen_exactly"]
            and features["independent_scaler"]["base_count"] == 6400
            and features["independent_scaler"]["split"] == "train"
            and features["production_scaler_digest"] == verify["utility_contract"]["scaler_digest"]
        ),
        "model_f_fit_complete": bool(fit_stage["models"]["model_F"]["coefficient_digest"]),
        "model_t_fit_complete": bool(fit_stage["models"]["model_T"]["coefficient_digest"]),
        "coefficients_finite": bool(
            model_audits["model_F"]["checks"]["coefficients_finite"]
            and model_audits["model_T"]["checks"]["coefficients_finite"]
        ),
        "objectives_finite": bool(
            model_audits["model_F"]["checks"]["objective_finite"]
            and model_audits["model_T"]["checks"]["objective_finite"]
        ),
        "independent_objective_audit_pass": bool(
            model_audits["model_F"]["all_pass"] and model_audits["model_T"]["all_pass"]
        ),
        "red_blue_orientation_audit_pass": bool(
            audit_stage["record_audit"]["all_pass"]
            and next(
                control
                for control in controls
                if control["control"] == "orientation_swap"
            )["detected"]
        ),
        "deterministic_refit_pass": bool(refit["all_identical"]),
        "negative_controls_fire": bool(
            audit_stage["negative_controls_fired"] == len(controls)
        ),
        "production_scorer_own_side_only": bool(
            not audit_stage["own_side_findings"]
            and audit_stage["scorer_decomposition_max_abs_difference"] <= 1e-10
        ),
        "no_validation_outcome_access": bool(
            all(
                not entry["outcomes"] and not entry["neural"]
                for entry in BANK_ACCESS_LOG
                if entry["bank"] == "phase10_validation_bank_v1"
            )
        ),
        "no_test_outcome_access": bool(
            all(
                not entry["outcomes"] and not entry["neural"]
                for entry in BANK_ACCESS_LOG
                if entry["bank"] == "phase10_test_bank_v1"
            )
        ),
        "no_candidate_selection": bool(
            set(fit_stage["models"]) == {"model_F", "model_T"}
            and not any(
                "selected" in key or "winner" in key for key in fit_stage["models"]
            )
        ),
        "phase9_checkpoint_unchanged": bool(
            verify["phase9_before"]["sha256"] == after["phase9_after"]["sha256"]
            and verify["phase9_before"]["model_state_digest"]
            == after["phase9_after"]["model_state_digest"]
        ),
        "full_suite_green": bool(suite["returncode"] == 0 and suite["failed"] == 0),
    }


def stage_artifacts(args) -> dict:
    verify = load_stage("verify")
    features = load_stage("features")
    fit_stage = load_stage("fit")
    refit = load_stage("refit")
    audit_stage = load_stage("audit")

    problems: list = []
    log("re-verifying the Phase 9 checkpoint and the sealed corpus after all work")
    phase9_after = verify_phase9_checkpoint(problems, label="after")
    corpus_after = verify_corpus(problems)
    if problems:
        raise Agent3Error(f"post-work preservation checks failed: {problems}")
    after = {"phase9_after": phase9_after, "corpus_after": corpus_after}

    # Without `--run-pytest` the suite is recorded as not run, which fails the
    # `full_suite_green` gate: a gate must never be claimable without evidence.
    suite = run_pytest() if args.run_pytest else {
        "command": ".venv/bin/python -m pytest tests -q",
        "summary": "not run in this invocation",
        "returncode": -1,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "seconds": 0.0,
    }

    gates = completion_gates(verify, features, fit_stage, refit, audit_stage, after, suite)
    false_gates = sorted(name for name, value in gates.items() if not value)
    status = "PASS" if not false_gates else "FAIL"

    from stratego.training import phase10_contract as contract
    from stratego.training import phase10_utility_fit as fit

    selector_document = contract.selector_document()

    models_payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_03_utility_models",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status_source": "reports/phase_10_data/agent_03_acceptance.json",
        "upstream": {
            "corpus_version": "phase10_setup_outcome_corpus_v1",
            "corpus_content_digest": ACCEPTED_CORPUS_CONTENT_DIGEST,
            "corpus_games": 16384,
            "contract_bundle_digest": verify["contracts"]["contract_bundle_digest"],
            "utility_contract_digest": verify["contracts"]["contract_digests"][
                "phase10_setup_utility_v1"
            ],
            "scaler_digest": fit_stage["scaler_digest"],
            "phase7_library_content_digest": verify["phase7_library"]["content_digest"],
            "phase9_checkpoint_sha256": verify["phase9_before"]["sha256"],
        },
        "fitted_artifact": {
            "path": fit_stage["artifact_path"],
            "sha256": fit_stage["artifact_sha256"],
            "artifact_version": fit.FITTED_UTILITY_VERSION,
        },
        "fitting_input_allowlist": fit_stage["fitting_input_allowlist"],
        "forbidden_fitting_fields": fit_stage["forbidden_fitting_fields"],
        "accessed_fields": fit_stage["accessed_fields"],
        "models": fit_stage["models"],
        "refit": refit,
        "no_model_selection": (
            "both models go forward to Agent 4; training objective values are "
            "diagnostics and rank nothing"
        ),
    }
    MODELS_ARTIFACT.write_text(json.dumps(models_payload, indent=2, sort_keys=True) + "\n")

    audit_payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_03_utility_audit",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "independence": (
            "trait vectors rebuilt from placements; flattening re-derived from "
            "TRAIT_SCHEMA; scaler recomputed with numpy from all 6,400 train "
            "bases and compared to the frozen literals; targets, orientation, "
            "logits, sigmoid probabilities, BCE, L2, objective, centering and "
            "gradients recomputed in numpy from the exported coefficients "
            "without calling the production fit helper"
        ),
        "feature_reconstruction": features,
        "record_audit": audit_stage["record_audit"],
        "design_agreement": audit_stage["design_agreement"],
        "model_audits": audit_stage["model_audits"],
        "negative_controls": audit_stage["negative_controls"],
        "scorer_decomposition": {
            "max_abs_difference": audit_stage["scorer_decomposition_max_abs_difference"],
            "samples": audit_stage["scorer_decomposition_samples"],
        },
        "deterministic_refit": refit,
    }
    AUDIT_ARTIFACT.write_text(json.dumps(audit_payload, indent=2, sort_keys=True) + "\n")

    acceptance = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_03_acceptance",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status,
        "completion_gates": gates,
        "gates_total": len(gates),
        "gates_true": sum(1 for value in gates.values() if value),
        "false_gates": false_gates,
        "environment": verify["environment"],
        "frozen_inputs": {
            "corpus_content_digest": ACCEPTED_CORPUS_CONTENT_DIGEST,
            "contract_bundle_digest": verify["contracts"]["contract_bundle_digest"],
            "utility_contract_digest": verify["contracts"]["contract_digests"][
                "phase10_setup_utility_v1"
            ],
            "outcome_schedule_digest": verify["contracts"]["outcome_schedule_digest"],
            "scaler_digest": fit_stage["scaler_digest"],
            "phase7_library_content_digest": verify["phase7_library"]["content_digest"],
            "phase9_checkpoint_sha256": verify["phase9_before"]["sha256"],
            "phase9_model_state_digest": verify["phase9_before"]["model_state_digest"],
        },
        "new_digests": {
            "setup_utility_v1_file_sha256": fit_stage["artifact_sha256"],
            "model_F_coefficient_digest": fit_stage["models"]["model_F"]["coefficient_digest"],
            "model_T_coefficient_digest": fit_stage["models"]["model_T"]["coefficient_digest"],
        },
        "discipline": {
            "c1_optimizer_steps": 0,
            "candidates_selected": 0,
            "held_out_bases_in_fitting": 0,
            "human_games_used": 0,
            "neural_inference_on_either_bank": 0,
            "games_played": 0,
            "test_bank_outcome_access": 0,
            "validation_bank_outcome_access": 0,
            "utility_models_fit": 2,
            "fits_per_model_canonical": 1,
            "refit_replays_per_model": REFIT_PROCESSES,
            "hyperparameter_search_runs": 0,
        },
        "bank_access_log": [dict(entry) for entry in BANK_ACCESS_LOG],
        "phase9_preservation": {
            "before": {
                "sha256": verify["phase9_before"]["sha256"],
                "model_state_digest": verify["phase9_before"]["model_state_digest"],
            },
            "after": {
                "sha256": phase9_after["sha256"],
                "model_state_digest": phase9_after["model_state_digest"],
            },
            "unchanged": gates["phase9_checkpoint_unchanged"],
        },
        "corpus_preservation": {
            "before": verify["corpus"]["content_digest"],
            "after": corpus_after["content_digest"],
            "state_after": corpus_after["state"],
            "byte_identical_content": corpus_after["content_digest"]
            == verify["corpus"]["content_digest"],
        },
        "suite_before": TESTS_BEFORE,
        "suite": suite,
        "deviations": [
            {
                "contract_text": "Fit each model exactly once from the all-zero initialization",
                "reading": (
                    "one canonical fit per model produced the accepted coefficients; "
                    "the instruction's deterministic-refit section separately requires "
                    "fitting each model at least twice in independent processes, so two "
                    "subprocess refits per model re-ran the identical frozen protocol "
                    "and were compared bit-for-bit and discarded. No refit informed any "
                    "coefficient, threshold or decision; the accepted artifact is the "
                    "canonical fit's output alone."
                ),
            },
            {
                "contract_text": "deterministic full-batch L-BFGS (device cpu, float64)",
                "reading": (
                    "the fit runs under torch.set_num_threads(1) so the reduction order "
                    "is fixed and coefficients are bit-reproducible across processes; "
                    "thread count is an execution detail the frozen protocol does not "
                    "name, recorded here because the refit gate compares for exact "
                    "equality"
                ),
            },
            {
                "contract_text": "negative control: use held-out statistics for standardization",
                "reading": (
                    "the control computes a validation-split scaler from held-out bases' "
                    "structural trait vectors only (the same bytes the frozen bank "
                    "construction already reads), proves its digest and moments are "
                    "detected as wrong, and discards it; no held-out outcome exists or "
                    "was read, and no held-out statistic touched any fit"
                ),
            },
            {
                "contract_text": "Add gradient or finite-difference spot checks",
                "reading": (
                    "the audit gates stationarity at gradient max-abs <= 1e-6 and "
                    "analytic-vs-central-difference agreement at 1e-6 + 1e-6*|value| "
                    "with step 1e-6; both tolerances were frozen in "
                    "stratego/training/phase10_utility_audit.py before any comparison "
                    "ran, with the justification recorded beside them"
                ),
            },
            {
                "contract_text": "objective: full-batch BCE + L2; initialisation: exact all-zero parameter vector",
                "reading": (
                    "BCE is computed through torch.nn.functional."
                    "binary_cross_entropy_with_logits: at the frozen all-zero start "
                    "every logit is exactly 0.0, where a hand-composed stable BCE "
                    "(clamp/abs/log1p pieces) autodiffs to a wrong subgradient (-y "
                    "instead of sigmoid(0) - y) and L-BFGS line-searches a "
                    "non-descent direction and never moves; the library primitive's "
                    "backward is the analytic sigmoid(eta) - y, exact at 0, and the "
                    "independent audit's numpy objective agrees with the reported "
                    "values to ~1e-16 while its finite-difference checks confirm the "
                    "gradient at the fitted point"
                ),
            },
        ],
        "carried_forward_obligations": [
            {
                "for_agent": 4,
                "obligation": (
                    "exhaustively collision-check the materialized selector_audit seed "
                    "universe when the millions of per-candidate x colour x split draw "
                    "ids actually exist; Agent 1's 58,792-seed audit does not cover them"
                ),
            },
            {
                "for_agent": 4,
                "obligation": (
                    "Agent 2's CPU-vs-MPS probe was 32 games and is evidence about those "
                    "games only, not exhaustive backend identity; the sealed corpus is "
                    "authoritative as pure-CPU float32 evidence and later agents must "
                    "not cite the probe as proving backend equivalence"
                ),
            },
        ],
        "handoff_to_agent_4": handoff_document(verify, features, fit_stage, refit, audit_stage, selector_document),
    }
    ACCEPTANCE_ARTIFACT.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n")

    write_report(models_payload, audit_payload, acceptance)
    log(f"status {status}; gates {acceptance['gates_true']}/{acceptance['gates_total']}")
    return save_stage(
        "artifacts",
        {
            "stage": "artifacts",
            "status": status,
            "gates_true": acceptance["gates_true"],
            "gates_total": acceptance["gates_total"],
            "false_gates": false_gates,
            "artifacts": [
                str(MODELS_ARTIFACT.relative_to(REPOSITORY_ROOT)),
                str(AUDIT_ARTIFACT.relative_to(REPOSITORY_ROOT)),
                str(ACCEPTANCE_ARTIFACT.relative_to(REPOSITORY_ROOT)),
            ],
        },
    )


def handoff_document(verify, features, fit_stage, refit, audit_stage, selector_document) -> dict:
    return {
        "for_agent": 4,
        "mission": "implement the selector and production setup source; sampling and diversity only",
        "fitted_utility": {
            "artifact_version": "setup_utility_v1",
            "path": fit_stage["artifact_path"],
            "file_sha256": fit_stage["artifact_sha256"],
            "model_F_coefficient_digest": fit_stage["models"]["model_F"]["coefficient_digest"],
            "model_T_coefficient_digest": fit_stage["models"]["model_T"]["coefficient_digest"],
            "scaler_digest": fit_stage["scaler_digest"],
            "loader": "stratego.training.phase10_utility_fit.SetupUtilityScorer.from_path",
        },
        "scoring_contract": (
            "u(s, c) = family_offsets_effective[c][family(s)] for model_F, plus "
            "trait_weights[c] . standardized_features(s) for model_T; inputs are own "
            "colour, own family and own base trait vector only; the red_first_intercept "
            "is a fit diagnostic no scoring path reads"
        ),
        "six_candidates": selector_document["candidates"],
        "mixture": selector_document["mixture"],
        "no_refitting": "the two models are fit once; candidate-specific refitting is forbidden",
        "proof_no_held_out_outcomes": {
            "validation_bank_outcome_access": 0,
            "test_bank_outcome_access": 0,
            "corpus_split": "train",
            "unique_corpus_base_splits": features["unique_corpus_base_splits"],
        },
        "proof_phase9_unchanged": {
            "sha256": verify["phase9_before"]["sha256"],
            "model_state_digest": verify["phase9_before"]["model_state_digest"],
        },
        "deterministic_refit": {
            model_id: {
                "fits": refit["comparisons"][model_id]["fits"],
                "identical": refit["comparisons"][model_id]["identical"],
            }
            for model_id in refit["comparisons"]
        },
        "diagnostics_only_note": (
            "fit objectives and corpus result counts are diagnostics; they rank "
            "nothing and select nothing"
        ),
    }


# ---------------------------------------------------------------------------
# The report section
# ---------------------------------------------------------------------------


def _f(value: float, places: int = 6) -> str:
    return f"{value:.{places}f}"


def write_report(models_payload: dict, audit_payload: dict, acceptance: dict) -> None:
    fit_f = models_payload["models"]["model_F"]
    fit_t = models_payload["models"]["model_T"]
    audit_f = audit_payload["model_audits"]["model_F"]
    audit_t = audit_payload["model_audits"]["model_T"]
    refit = models_payload["refit"]
    features = audit_payload["feature_reconstruction"]
    record_audit = audit_payload["record_audit"]
    controls = audit_payload["negative_controls"]
    gates = acceptance["completion_gates"]

    controls_lines = "\n".join(
        f"| `{control['control']}` | {'fired' if control['detected'] else 'DID NOT FIRE'} |"
        for control in controls
    )
    gates_lines = "\n".join(
        f"| `{name}` | {'true' if value else 'FALSE'} |" for name, value in sorted(gates.items())
    )

    section = f"""{SECTION_MARKER}

Status: **{acceptance['status']}** — {acceptance['gates_true']}/{acceptance['gates_total']} completion gates true.
Agent 3 fit exactly the two frozen utility models from the sealed
`phase10_setup_outcome_corpus_v1` (content digest `{ACCEPTED_CORPUS_CONTENT_DIGEST[:12]}...`),
audited the fit through an independent numpy path, and selected nothing:
both models go forward to Agent 4, and no validation or test outcome was
touched (neither bank stores one, and this agent played zero games).

### 3.1 Verified prerequisites

Agents 1 and 2 are PASS with no false gate. All eight contract digests, the
bundle (`257f140d...`), the outcome-schedule digest, both bank digests and
manifests, the Phase 9 isolation set, the Phase 7 library
(`7b8a6660...`, 6,400/800/800), and the accepted Phase 9 checkpoint (file
SHA `dfd698e5...`, model state `f1df694d...`, 863,959 parameters, all
finite) were recomputed from live bytes. The live
`phase10_setup_utility_v1` document equals Agent 1's frozen artifact copy
byte for byte, so no learning-design decision was left to make here. The
corpus was verified SEALED at its accepted content digest before fitting
and re-verified byte-identical after all work.

### 3.2 The fitting-input allowlist

The 37-field record is storage and provenance, not a feature set. Fitting
reads records only through `AllowlistedRecord`
(`stratego/training/phase10_utility_fit.py`), which raises on any field
outside the model's frozen allowlist:

```text
model_F: game_id, red_family, blue_family, result
model_T: + red_base_setup_id, blue_base_setup_id
```

`game_id` orders rows, `result` rebuilds the target through the frozen
mapping (red win 1.0, draw 0.5, red loss 0.0) — the stored `red_score` is
never read by fitting — and the base ids resolve each side's *base* through
`setup_library_v1` into the frozen `phase10_trait_feature_v1` 47-scalar
representation through the frozen train-only scaler. The other
{len(models_payload['forbidden_fitting_fields'])} stored fields (final fingerprints, provenance,
seeds, attempts, terminal reason, plies, decisions, digests, policy
identity, match seed, ordinal, winner, red_score, ...) are forbidden by
complement: accessing one is an exception, and the fields actually accessed
are recorded in the artifact (`accessed_fields`).

### 3.3 Feature reconstruction and the standardizer

Every one of the 8,000 library entries had its trait vector rebuilt from
its stored placement via `compute_trait_vector` and compared to the stored
vector: {features['stored_trait_mismatches']} mismatches. The 35-field to 47-scalar flattening was
re-derived independently from `TRAIT_SCHEMA` and its name order equals the
frozen feature names. The standardizer was recomputed with plain numpy over
**all 6,400 train bases** (`ddof=0`): mean and std match Agent 1's frozen
literals exactly, the production scaler digest is the frozen
`fa6eb1c1...`, and there are no zero-std fields. The corpus touches
{features['unique_corpus_bases']} unique bases, all train-split; both recorded per-side trait
identity digests were re-derived and matched for every record
({features['trait_identity_digests_verified']} digests).

### 3.4 The two fits

Both models were fit exactly once, in canonical corpus order, from the
exact all-zero parameter vector, under the frozen protocol (CPU float64,
full-batch BCE + L2 1e-3 on raw family offsets and trait weights, intercept
unpenalized, L-BFGS lr 1.0, max 500 iterations, history 50, tolerance_grad
1e-10, tolerance_change 1e-12, strong Wolfe, single-threaded):

```text
model_F  objective {_f(fit_f['diagnostics']['objective'])}  bce {_f(fit_f['diagnostics']['bce'])}  l2 {_f(fit_f['diagnostics']['l2_penalty'], 8)}
         iterations {fit_f['diagnostics']['iterations']}  evaluations {fit_f['diagnostics']['function_evaluations']}  grad max {fit_f['diagnostics']['final_grad_max_abs']:.2e}
model_T  objective {_f(fit_t['diagnostics']['objective'])}  bce {_f(fit_t['diagnostics']['bce'])}  l2 {_f(fit_t['diagnostics']['l2_penalty'], 8)}
         iterations {fit_t['diagnostics']['iterations']}  evaluations {fit_t['diagnostics']['function_evaluations']}  grad max {fit_t['diagnostics']['final_grad_max_abs']:.2e}
```

The logit uses centered offsets while the penalty uses raw ones, so the
minimizer self-centers: observed raw-offset means are at the
{max(abs(value) for value in fit_f['diagnostics']['raw_offset_means'] + fit_t['diagnostics']['raw_offset_means']):.1e} level. Objective values are diagnostics; they rank nothing.

Production coefficients and the scaler live in
`{models_payload['fitted_artifact']['path']}`
(SHA-256 `{models_payload['fitted_artifact']['sha256'][:16]}...`), referenced by digest from the
artifacts. Coefficient digests: model_F
`{fit_f['coefficient_digest'][:16]}...`, model_T `{fit_t['coefficient_digest'][:16]}...`.

### 3.5 Deterministic refit

Each model was refit {refit['independent_processes']} more times in independent processes from the same
all-zero initialisation. The frozen criterion — bit-exact equality of the
canonical coefficient JSON — held for every fit of both models
(max abs coefficient difference 0.0, objective spread 0.0, digests
identical across {refit['processes_per_model']} processes per model).

### 3.6 The independent audit

The audit (`stratego/training/phase10_utility_audit.py`) rebuilt the design
without the production fit helper: placements -> trait vectors -> its own
flattening -> the frozen scaler literals -> per-record standardized
features, targets from the stored W/D/L token through its own mapping, and
Red/Blue orientation re-derived from each game id (families, ordinal, match
seed, winner/`red_score` consistency — all {record_audit['records_audited']} records, zero
violations). Production and audit designs agree exactly (targets, family
indices, game ids, and features to 0.0 max abs difference).

From the exported coefficients alone it recomputed logits, sigmoid
probabilities, BCE, L2, the full objective, the centering, and the analytic
gradient, all finite:

```text
model_F  |L_audit - L_reported| = {audit_f['objective_abs_difference']:.2e}   grad max {audit_f['gradient']['max_abs']:.2e}   FD worst {audit_f['finite_difference']['worst_abs_difference']:.2e}
model_T  |L_audit - L_reported| = {audit_t['objective_abs_difference']:.2e}   grad max {audit_t['gradient']['max_abs']:.2e}   FD worst {audit_t['finite_difference']['worst_abs_difference']:.2e}
```

All tolerances (objective/logit agreement 1e-10, stationarity 1e-6,
finite-difference 1e-6 + 1e-6|value|, centering 1e-8, refit exact) were
frozen in the audit module before any comparison ran.

### 3.7 Negative controls

{controls_lines and '| control | outcome |'}
| --- | --- |
{controls_lines}

Each control corrupts one thing the audit is supposed to catch — reversed
pair orientation (both models), draws scored 0.0, a validation-split
standardizer, a swapped trait column, a tampered family id, a tampered
coefficient — and every one was detected by the same checks that pass on
the true inputs.

### 3.8 Production-input safety

The exported artifact decomposes to own-side `u(s, c)`: closed key sets at
the root and per model, exactly 2 x 16 offsets and 2 x 47 weights indexed by
own colour / own family / own feature, no opponent-conditioned table, no
matchup matrix, no outcome-conditioned production feature. The scorer's
entire surface is `utility(model_id, colour, family_id, trait_vector)` —
there is no opponent argument to pass — and the recorded game logits equal
`intercept + u(red) - u(blue)` on sampled records to
{audit_payload['scorer_decomposition']['max_abs_difference']:.1e}. The red-first intercept is stored as a diagnostic and no
scoring path reads it.

### 3.9 No model selection

Model F and Model T were not compared by any strength signal. Both go
forward to Agent 4 with the six frozen candidate definitions. Corpus result
counts ({record_audit['result_counts']['red_win']} red wins / {record_audit['result_counts']['draw']} draws / {record_audit['result_counts']['red_loss']} red losses) remain diagnostics.

### 3.10 Phase 9 and corpus preservation

The accepted Phase 9 checkpoint hashed identical before and after all Agent
3 work (file SHA and model-state digest; zero optimizer steps). The sealed
corpus re-verified at its accepted content digest with the seal intact:
Agent 3 opened no writer and reconciled nothing.

### 3.11 Recorded readings

Four readings are recorded in the acceptance artifact rather than decided
silently:

- **"fit exactly once" vs the deterministic-refit requirement** — one
  canonical fit per model produced the accepted coefficients; the two
  subprocess refits per model are byte-identical replays run only for the
  determinism gate, informed nothing, and were discarded.
- **single-threaded execution** — the frozen protocol names device and
  precision but not thread count; `torch.set_num_threads(1)` fixes the
  reduction order so the exact-equality refit criterion is meaningful, and
  is recorded in every fit's diagnostics.
- **the held-out-scaler control's inputs** — the control reads
  validation-split bases' *structural* trait vectors only (bytes the frozen
  bank construction already reads), proves the wrong scaler is detected,
  and discards it; no outcome exists or was read.
- **audit-internal gradient tolerances** — stationarity 1e-6 and
  finite-difference 1e-6 + 1e-6|value| were frozen in the audit module
  before any comparison ran.

One implementation finding is worth the reviewing chat's attention: the
frozen all-zero initialisation makes every logit exactly 0.0 at step 0,
where a hand-composed stable BCE (`clamp`/`abs`/`log1p` pieces) autodiffs
to a wrong subgradient (`-y` instead of `sigmoid(0) - y`), and L-BFGS then
line-searches a non-descent direction and terminates without moving. The
fit therefore computes BCE through
`torch.nn.functional.binary_cross_entropy_with_logits`, whose backward is
the analytic `sigmoid(eta) - y`, exact at 0; the unit suite pins the
analytic gradient at the zero start and the audit's finite-difference
checks confirm it at the fitted point.

### 3.12 Evidence

```text
tests before   {acceptance['suite_before']['summary']}
tests after    {acceptance['suite']['summary']}
```

Machine-readable: `reports/phase_10_data/agent_03_utility_models.json`,
`reports/phase_10_data/agent_03_utility_audit.json`,
`reports/phase_10_data/agent_03_acceptance.json`.

| gate | value |
| --- | --- |
{gates_lines}

### 3.13 Handoff to Agent 4

The fitted `setup_utility_v1` artifact (path + SHA above), both coefficient
digests, the scaler digest `fa6eb1c1...`, the pure own-side scoring
contract, the six frozen candidates (P10-A..F over the 0.35/0.65 mixture),
proof that no held-out outcome was used (the corpus is train-only; neither
bank stores an outcome; zero games played), and proof Phase 9 is unchanged.
Agent 4 implements sampling and diversity only — the utility models are
frozen from here.

Two obligations carry forward: (1) Agent 4 must exhaustively
collision-check the materialized `selector_audit` seed universe when its
millions of draw ids exist — Agent 1's 58,792-seed audit does not cover
them; (2) Agent 2's 32-game CPU-vs-MPS probe is evidence about those games
only, never to be cited as exhaustive backend identity — the corpus is
authoritative as pure-CPU float32 evidence.
"""

    text = REPORT_PATH.read_text()
    if SECTION_MARKER in text:
        head, _, tail = text.partition(SECTION_MARKER)
        # The section runs to the next same-level heading or EOF.
        remainder = tail.partition("\n## ")
        text = head + section + ("\n## " + remainder[2] if remainder[1] else "")
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + section
    REPORT_PATH.write_text(text)
    log(f"report section written to {REPORT_PATH.relative_to(REPOSITORY_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STAGES = {
    "verify": stage_verify,
    "features": stage_features,
    "fit": stage_fit,
    "refit": stage_refit,
    "audit": stage_audit,
    "artifacts": stage_artifacts,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=[*STAGES, "refit-worker", "all"], default="all")
    parser.add_argument("--model", choices=["model_F", "model_T"], help="refit-worker only")
    parser.add_argument("--output", help="refit-worker only")
    parser.add_argument("--run-pytest", action="store_true")
    args = parser.parse_args()

    if args.stage == "refit-worker":
        if not args.model or not args.output:
            raise SystemExit("refit-worker needs --model and --output")
        stage_refit_worker(args)
        return 0

    started = time.perf_counter()
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for name in stages:
        STAGES[name](args)
    log(f"done in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
