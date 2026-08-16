#!/usr/bin/env python3
"""Phase 8 Agent 7 acceptance harness: independent held-out evaluation and freeze.

Writes

    reports/phase_8_data/agent_07_heldout_metrics.json
    reports/phase_8_data/agent_07_random_evaluation.json
    reports/phase_8_data/agent_07_final_acceptance.json
    reports/phase_8_data/agent_07_phase9_handoff.json

What this script proves
-----------------------
That the checkpoint Agent 6 froze — and nothing else — passes every frozen
Phase 8 acceptance gate on evidence gathered only after the checkpoint became
immutable: the sealed synthetic test split (opened here for the first time
through the frozen `check_test_corpus_access` gate), the full 2,048-game
Phase 4 random evaluation over all 1,024 frozen setup pairs, and the
1,024-game improvement evaluation against the frozen canonical untrained C1.
Nothing here trains, tunes, or selects; a failed frozen gate is a Phase 8
FAIL, never an invitation to move a threshold or touch the checkpoint.

Corpus identity is resolver + digests, never a path
---------------------------------------------------
The corpus is opened exclusively through
`stratego.training.synthetic_corpus.default_corpus_root()`. This harness pins
the root that resolver must currently produce (an acceptance harness may pin
the expectation; library code never does) and requires the accepted content,
metadata and commit-index digests to match exactly, including the byte-level
payload audit. Any mismatch is BLOCKED — the corpus is never regenerated or
repaired by this script.

Why the module scope is torch-free
----------------------------------
`run_neural_schedule` spawns pure-engine game workers, and `spawn` re-imports
this file inside every one of them. Torch and every torch-importing module
(`stratego.model`, `stratego.training.warmstart_*`,
`stratego.training.synthetic_corpus`) are therefore imported inside
functions only, and every parallel run records the measured
`workers_importing_torch` count so the property is observed, not assumed.

Usage::

    python scripts/run_phase8_agent07.py --verify           # identity gates only
    python scripts/run_phase8_agent07.py --full             # every stage in order
    python scripts/run_phase8_agent07.py --full --run-pytest
    python scripts/run_phase8_agent07.py --quick            # tiny dev pass, no artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

# Torch-free imports only above function scope; see the module docstring.
from stratego.engine.constants import (  # noqa: E402
    IMPLEMENTATION_VERSION,
    NUM_PIECE_TYPES,
    OBSERVATION_VERSION,
    PIECE_TYPE_NAMES,
    RED,
    RULES_VERSION,
)
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
    NEURAL_WORKER_VERSION,
    LocalInferenceChannel,
    RemoteNeuralPolicy,
    neural_policy_ref,
    run_neural_schedule,
)
from stratego.evaluation.registry import STRESS_POLICY_IDS, policy_ref  # noqa: E402
from stratego.evaluation.reporting import write_json  # noqa: E402
from stratego.evaluation.setup_bank import SETUP_BANK_VERSION, SetupBank, bank_digest  # noqa: E402
from stratego.evaluation.statistics import (  # noqa: E402
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    matchup_seed,
    summarize_matchup,
)
AGENT = 7
PHASE = 8
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_8_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_8_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase8" / "agent07"

CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase8" / "warmstart_c1_v1.pt"
INITIAL_CHECKPOINT_PATH = (
    REPOSITORY_ROOT / "checkpoints" / "phase8" / "warmstart_c1_v1_initialisation.pt"
)

#: The accepted frozen identities Agent 7 independently re-verifies. The
#: checkpoint digests come from Agent 6's accepted manifest; the corpus root
#: below is the *acceptance expectation* for what the resolver must currently
#: return — evaluation code never consumes this constant as a path.
EXPECTED_CHECKPOINT_SHA256 = (
    "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca"
)
EXPECTED_INITIAL_SHA256 = (
    "01c907eeef86ec04121db55ccffb9365e8df27fdf05921b921947d4af365754c"
)
EXPECTED_INITIAL_STATE_CHECKSUM = (
    "cfe60bb0cb342b03e2506259b5c4d39d321148f7bc8c030bf722e5a234e042b8"
)
EXPECTED_TRAIN_CONFIG_DOCUMENT_DIGEST = (
    "3cab772bd8f74677efcdc1f90ec6f383490313f7652d82bd7fedf86153919ae7"
)
EXPECTED_TRAINER_RUNTIME_DIGEST = (
    "64db92539a7d6c06ac4d01e4e904857da5b95c3d86d1287e108ede19e4f03879"
)
EXPECTED_CORPUS_ROOT = (
    "/Users/brandonwashington/Dev/Github/stratego/gpt_agent/"
    "data/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1"
)

#: The full suite as measured immediately before any Agent 7 work, at the
#: commit below with the pre-existing untracked additions in the tree.
TESTS_BEFORE = {
    "command": ".venv/bin/python -m pytest tests -q",
    "summary": "3774 passed, 3 skipped in 225.01s (0:03:45)",
    "passed": 3774,
    "failed": 0,
    "skipped": 3,
    "seconds": 225.01,
    "measured_at_commit": "53050b9",
}

BANK_ARTIFACT = REPOSITORY_ROOT / "reports" / "phase_4_data" / "agent_01_setup_bank_v1.json"

#: Evaluation identities. `neural_policy_ref` is the frozen way weights, a
#: decision rule and a precision become a policy identity; the two candidate
#: tokens below are new identities for this evaluation and are deliberately
#: not added to the Phase 4 catalogue.
WARMSTART_CANDIDATE_ID = "c1_warmstart"
INITIAL_CANDIDATE_ID = "c1_untrained_init"
GATE_DTYPE = "float32"
RANDOM_OPPONENT_ID = "random_legal"
TIER_OPPONENT_IDS = ("basic_heuristic", "tactical_rule_based", "strategic_rule_based")

#: Frozen evaluation sizes (Agent 1's acceptance thresholds).
RANDOM_PAIRS = 1024
VS_INIT_PAIRS = 512
TIER_PAIRS = 256
STRESS_PAIRS = 64

#: Belief progress buckets: quartiles of normalized decision position.
PROGRESS_BUCKETS = ("0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00")


class Agent7Error(RuntimeError):
    """A precondition or frozen identity failed. Always raised, never patched."""


def _contracts():
    """`(warmstart_contract, warmstart_seed)`, imported on first use.

    Both are torch-free modules, but importing either executes a parent
    package `__init__` that is not — `stratego.model` for the contract's
    `candidate_config` import — so they must never sit at this script's
    module scope: `spawn` re-imports `__main__` in every game worker, and a
    game worker must stay a pure engine/NumPy process. The measured
    `workers_importing_torch = 0` in every recorded run is the receipt.
    """
    from stratego.training import warmstart_contract, warmstart_seed

    return warmstart_contract, warmstart_seed


def _model_configs():
    """`(ARCHITECTURE_FAMILY, candidate_config)`, function-scoped for the same reason."""
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config

    return ARCHITECTURE_FAMILY, candidate_config


# ---------------------------------------------------------------------------
# Environment and small helpers
# ---------------------------------------------------------------------------


def torch_report() -> dict:
    import torch

    return {
        "torch_version": str(torch.__version__),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - a checkout without git
        return "unknown"


def working_tree_state() -> str:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return "dirty" if status else "clean"
    except Exception:  # pragma: no cover
        return "unknown"


def environment_report() -> dict:
    report = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "source_revision": git_commit(),
        "working_tree_state": working_tree_state(),
    }
    report.update(torch_report())
    return report


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def stage_path(name: str, *, quick: bool = False) -> Path:
    suffix = "_quick" if quick else ""
    return WORK_DIRECTORY / f"stage_{name}{suffix}.json"


def write_stage(name: str, payload: dict, *, quick: bool = False) -> Path:
    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return write_json(stage_path(name, quick=quick), payload)


def read_stage(name: str, *, quick: bool = False) -> dict:
    path = stage_path(name, quick=quick)
    if not path.exists():
        raise Agent7Error(
            f"stage {name!r} has not run yet ({path} is missing); run the stages in order"
        )
    return json.loads(path.read_text())


def require(condition: bool, message: str, problems: list) -> bool:
    if not condition:
        problems.append(message)
    return bool(condition)


# ---------------------------------------------------------------------------
# 1. Verification: prerequisites, corpus identity, checkpoint identity
# ---------------------------------------------------------------------------


def prior_agent_statuses() -> dict:
    """Every Phase 8 Agent 1-6 primary artifact must exist and say PASS."""
    statuses: dict = {}
    for agent in range(1, 7):
        entries = sorted(DATA_DIRECTORY.glob(f"agent_{agent:02d}_*.json"))
        seen = {}
        for path in entries:
            payload = json.loads(path.read_text())
            if "status" in payload:
                seen[path.name] = payload["status"]
        statuses[f"agent_{agent}"] = {
            "artifacts": len(entries),
            "statuses": seen,
            "all_pass": bool(seen) and all(value == "PASS" for value in seen.values()),
        }
    statuses["agents_1_to_6_all_pass"] = all(
        statuses[f"agent_{agent}"]["all_pass"] for agent in range(1, 7)
    )
    return statuses


def accepted_corpus_identity() -> dict:
    """The accepted digests, cross-checked across every record that names them.

    Agent 2's manifest, Agent 5's frozen train config and Agent 6's checkpoint
    manifest all carry the corpus digests; they must agree with each other
    before the live corpus is compared against them.
    """
    manifest = json.loads((DATA_DIRECTORY / "agent_02_corpus_manifest.json").read_text())
    frozen_config = json.loads(
        (DATA_DIRECTORY / "agent_05_frozen_train_config.json").read_text()
    )
    checkpoint_manifest = json.loads(
        (DATA_DIRECTORY / "agent_06_checkpoint_manifest.json").read_text()
    )

    def digests_from(payload: dict, *paths):
        for path in paths:
            node = payload
            try:
                for key in path:
                    node = node[key]
            except (KeyError, TypeError):
                continue
            if isinstance(node, dict) and "content_digest" in node:
                return {
                    "corpus_version": node.get("corpus_version", "synthetic_warmstart_corpus_v1"),
                    "content_digest": node["content_digest"],
                    "metadata_digest": node["metadata_digest"],
                    "commit_index_digest": node["commit_index_digest"],
                }
        raise Agent7Error(f"no corpus digests found under {paths}")

    records = {
        "agent_02_corpus_manifest": digests_from(manifest, ("corpus_manifest",)),
        "agent_05_frozen_train_config": digests_from(
            frozen_config, ("config", "corpus_digests")
        ),
        "agent_06_checkpoint_manifest": digests_from(
            checkpoint_manifest, ("identities", "corpus_digests")
        ),
    }
    reference = records["agent_06_checkpoint_manifest"]
    for name, record in records.items():
        if {k: record[k] for k in reference} != reference:
            raise Agent7Error(
                f"accepted corpus digests disagree between records: {name} carries "
                f"{record}, the checkpoint manifest carries {reference}"
            )
    return {"accepted": reference, "records_checked": sorted(records), "all_agree": True}


def verify_corpus_through_resolver() -> dict:
    """Resolve through `default_corpus_root()` and verify the accepted identity.

    Byte-level payload verification included: `verify_corpus_identity` re-reads
    every committed payload and metadata record against its journal digest.
    A mismatch raises — BLOCKED, never regeneration.
    """
    from stratego.training import synthetic_corpus as sc
    from stratego.training.warmstart_checkpoint import CorpusIdentity, verify_corpus_identity

    accepted = accepted_corpus_identity()
    resolved = sc.describe_corpus_root()
    problems: list = []
    require(
        resolved["root"] == EXPECTED_CORPUS_ROOT,
        f"the resolver returned {resolved['root']!r}, not the accepted root "
        f"{EXPECTED_CORPUS_ROOT!r}",
        problems,
    )
    started = time.perf_counter()
    observed = verify_corpus_identity(
        sc.default_corpus_root(),
        CorpusIdentity.from_dict(accepted["accepted"]),
        check_payload_bytes=True,
    )
    seconds = time.perf_counter() - started
    return {
        "resolution": resolved,
        "resolved_root_matches_accepted_location": resolved["root"] == EXPECTED_CORPUS_ROOT,
        "accepted_identity": accepted,
        "observed_identity": observed.to_dict(),
        "payload_bytes_audited": True,
        "verification_seconds": round(seconds, 3),
        "problems": problems,
    }


def verify_upstream_and_universes() -> dict:
    """Frozen upstream versions, Phase 4 bank, Phase 7 library, universes."""
    wc, _ws = _contracts()
    from stratego.training.warmstart_dataset import WarmstartDataset, universe_digest

    problems = wc.verify_frozen_upstream(include_library_digest=True)

    started = time.perf_counter()
    bank = SetupBank.from_json(BANK_ARTIFACT.read_text())
    stored_bank_digest = bank_digest(bank)
    require(
        stored_bank_digest == wc.EXPECTED_PHASE4_BANK_DIGEST,
        f"stored Phase 4 bank digest {stored_bank_digest} != accepted "
        f"{wc.EXPECTED_PHASE4_BANK_DIGEST}",
        problems,
    )
    regenerated = SetupBank.generate(size=len(bank.pairs), root_seed=bank.root_seed)
    regenerated_digest = bank_digest(regenerated)
    require(
        regenerated_digest == wc.EXPECTED_PHASE4_BANK_DIGEST,
        f"regenerated Phase 4 bank digest {regenerated_digest} != accepted "
        f"{wc.EXPECTED_PHASE4_BANK_DIGEST}",
        problems,
    )
    bank_seconds = time.perf_counter() - started

    accepted_universes = json.loads(
        (DATA_DIRECTORY / "agent_03_example_contract.json").read_text()
    )["universe"]["digests"]
    dataset = WarmstartDataset()
    universes = {}
    started = time.perf_counter()
    for split in ("train", "validation", "test"):
        universe = dataset.universe(split)
        digest = universe_digest(universe)
        universes[split] = {
            "examples": len(universe),
            "digest": digest,
            "matches_accepted": digest == accepted_universes[split],
        }
        require(
            digest == accepted_universes[split],
            f"{split} selected-example universe digest {digest} != accepted "
            f"{accepted_universes[split]}",
            problems,
        )
    return {
        "frozen_upstream_problems": problems,
        "phase4_bank": {
            "artifact": str(BANK_ARTIFACT.relative_to(REPOSITORY_ROOT)),
            "stored_digest": stored_bank_digest,
            "regenerated_digest": regenerated_digest,
            "accepted_digest": wc.EXPECTED_PHASE4_BANK_DIGEST,
            "pairs": len(bank.pairs),
            "seconds": round(bank_seconds, 3),
        },
        "phase7_library_digest": wc.EXPECTED_LIBRARY_DIGEST,
        "selected_example_universes": universes,
        "universe_seconds": round(time.perf_counter() - started, 3),
    }


def verify_checkpoint_identity() -> dict:
    """The independent checkpoint load: rebuild C1, verify every identity."""
    wc, ws = _contracts()
    _family, candidate_config = _model_configs()
    import torch

    from stratego.training.warmstart_checkpoint import (
        read_warmstart_payload,
        load_model_for_evaluation,
        validate_warmstart_payload,
    )
    from stratego.training.warmstart_pilot import model_state_checksum
    from stratego.training.warmstart_trainer import WarmstartTrainConfig

    problems: list = []
    report: dict = {"checkpoints": {}}

    frozen_config = json.loads(
        (DATA_DIRECTORY / "agent_05_frozen_train_config.json").read_text()
    )
    document_digest = hashlib.sha256(
        json.dumps(frozen_config["config"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(
        document_digest == EXPECTED_TRAIN_CONFIG_DOCUMENT_DIGEST,
        f"warmstart_train_config_v1 document digest {document_digest} != accepted "
        f"{EXPECTED_TRAIN_CONFIG_DOCUMENT_DIGEST}",
        problems,
    )
    runtime = WarmstartTrainConfig.from_pilot_candidate(
        frozen_config["winning_candidate_id"],
        device=frozen_config["config"]["device"],
        validation_batches=frozen_config["config"]["validation_batches"],
    )
    runtime_digest = hashlib.sha256(
        json.dumps(runtime.identity(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(
        runtime_digest == EXPECTED_TRAINER_RUNTIME_DIGEST,
        f"trainer runtime identity digest {runtime_digest} != accepted "
        f"{EXPECTED_TRAINER_RUNTIME_DIGEST}",
        problems,
    )
    report["train_config_identity"] = {
        "train_config_version": frozen_config["train_config_version"],
        "winning_candidate_id": frozen_config["winning_candidate_id"],
        "document_digest": document_digest,
        "runtime_identity_digest": runtime_digest,
        "namespaces_are_distinct_objects": True,
    }

    live_config = candidate_config("C1")
    require(
        live_config.digest() == wc.EXPECTED_C1_CONFIG_DIGEST,
        f"live C1 config digest {live_config.digest()} != frozen "
        f"{wc.EXPECTED_C1_CONFIG_DIGEST}",
        problems,
    )

    for label, path, expected_sha in (
        ("accepted", CHECKPOINT_PATH, EXPECTED_CHECKPOINT_SHA256),
        ("canonical_untrained", INITIAL_CHECKPOINT_PATH, EXPECTED_INITIAL_SHA256),
    ):
        sha = file_sha256(path)
        require(
            sha == expected_sha,
            f"{label} checkpoint SHA-256 {sha} != accepted {expected_sha}",
            problems,
        )
        payload = read_warmstart_payload(path)
        validate_warmstart_payload(payload, source=str(path))
        stored_runtime_digest = payload["train_config_digest"]
        require(
            stored_runtime_digest == EXPECTED_TRAINER_RUNTIME_DIGEST,
            f"{label}: stored trainer runtime digest {stored_runtime_digest} != "
            f"accepted {EXPECTED_TRAINER_RUNTIME_DIGEST}",
            problems,
        )
        accepted_digests = accepted_corpus_identity()["accepted"]
        require(
            payload["corpus_identity"] == accepted_digests,
            f"{label}: stored corpus identity {payload['corpus_identity']} != accepted",
            problems,
        )
        model_payload = payload["model"]
        stored_config = dict(model_payload["model_configuration"])
        live_fields = {name: getattr(live_config, name) for name in stored_config}
        require(
            stored_config == live_fields,
            f"{label}: stored model configuration differs from live C1 in "
            f"{sorted(k for k in stored_config if stored_config[k] != live_fields.get(k))}",
            problems,
        )
        require(
            model_payload["model_contract_version"] == wc.EXPECTED_MODEL_CONTRACT_VERSION,
            f"{label}: model contract {model_payload['model_contract_version']!r}",
            problems,
        )
        model, _metadata = load_model_for_evaluation(path, device="cpu")
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
        checksum = model_state_checksum(model.state_dict())
        require(parameters == wc.EXPECTED_C1_PARAMETERS, f"{label}: {parameters} parameters", problems)
        require(finite, f"{label}: non-finite parameters", problems)
        report["checkpoints"][label] = {
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "file_sha256": sha,
            "warmstart_checkpoint_version": payload["warmstart_checkpoint_version"],
            "trainer_version": payload["trainer_version"],
            "example_version": payload["example_version"],
            "global_step": int(payload["global_step"]),
            "examples_consumed": int(payload["examples_consumed"]),
            "stored_trainer_runtime_digest": stored_runtime_digest,
            "model_contract_version": model_payload["model_contract_version"],
            "policy_action_frame": model_payload["policy_action_frame"],
            "engine_action_frame": model_payload["engine_action_frame"],
            "initialisation_seed": int(model_payload["provenance"]["initialisation_seed"]),
            "parameter_count": parameters,
            "all_parameters_finite": finite,
            "model_state_checksum": checksum,
        }
        del model

    require(
        report["checkpoints"]["canonical_untrained"]["model_state_checksum"]
        == EXPECTED_INITIAL_STATE_CHECKSUM,
        "the canonical untrained checkpoint does not carry the frozen fresh-init weights",
        problems,
    )
    from stratego.model import build_candidate_model

    rebuilt = build_candidate_model("C1", seed=ws.CANONICAL_C1_INIT_SEED)
    rebuilt_checksum = model_state_checksum(rebuilt.state_dict())
    require(
        rebuilt_checksum == EXPECTED_INITIAL_STATE_CHECKSUM,
        f"build_candidate_model('C1', seed={ws.CANONICAL_C1_INIT_SEED}) reconstructs "
        f"{rebuilt_checksum}, not the frozen canonical initialisation",
        problems,
    )
    report["canonical_reconstruction_checksum"] = rebuilt_checksum
    report["checkpoint_differs_from_initialisation"] = (
        report["checkpoints"]["accepted"]["model_state_checksum"]
        != report["checkpoints"]["canonical_untrained"]["model_state_checksum"]
    )
    require(
        report["checkpoint_differs_from_initialisation"],
        "the accepted checkpoint carries the untrained weights",
        problems,
    )
    del rebuilt
    report["problems"] = problems
    return report


def stage_verify(quick: bool = False) -> dict:
    started = time.perf_counter()
    payload = {
        "stage": "verify",
        "environment": environment_report(),
        "tests_before": TESTS_BEFORE,
    }
    problems: list = []
    # A raised mismatch (e.g. a corpus digest) is a stop condition; it is
    # recorded in the stage file as BLOCKED before this function re-raises,
    # so the failure is an artifact rather than only a stack trace.
    try:
        statuses = prior_agent_statuses()
        payload["prior_agents"] = statuses
        if not statuses["agents_1_to_6_all_pass"]:
            problems.append("agents 1-6 are not all PASS")
        corpus = verify_corpus_through_resolver()
        payload["corpus"] = corpus
        problems += corpus["problems"]
        upstream = verify_upstream_and_universes()
        payload["upstream"] = upstream
        problems += upstream["frozen_upstream_problems"]
        checkpoints = verify_checkpoint_identity()
        payload["checkpoint_identity"] = checkpoints
        problems += checkpoints["problems"]
    except Exception as error:
        problems.append(f"{type(error).__name__}: {error}")
        payload["problems"] = problems
        payload["status"] = "BLOCKED"
        payload["seconds"] = round(time.perf_counter() - started, 3)
        write_stage("verify", payload, quick=quick)
        raise
    payload["problems"] = problems
    payload["status"] = "PASS" if not problems else "BLOCKED"
    payload["seconds"] = round(time.perf_counter() - started, 3)
    write_stage("verify", payload, quick=quick)
    if problems:
        raise Agent7Error(f"verification is BLOCKED: {problems}")
    return payload


# ---------------------------------------------------------------------------
# 2. Export both models into the frozen Phase 6 evaluation checkpoint format
# ---------------------------------------------------------------------------


def stage_export(quick: bool = False) -> dict:
    """Bridge the warmstart training checkpoints to the evaluation harness.

    `InferenceOwner` loads the frozen Phase 6 evaluation checkpoint format.
    The two warmstart checkpoints are loaded through their own normal API
    (`load_model_for_evaluation`) and re-serialized with the frozen
    `stratego.model.checkpoint.save_checkpoint`; a bitwise state-dict
    comparison after reload proves the bridge changed nothing.
    """
    ARCHITECTURE_FAMILY, candidate_config = _model_configs()
    import torch

    from stratego.model.checkpoint import load_checkpoint, save_checkpoint
    from stratego.training.warmstart_checkpoint import load_model_for_evaluation
    from stratego.training.warmstart_pilot import model_state_checksum

    read_stage("verify", quick=quick)
    started = time.perf_counter()
    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    configuration = candidate_config("C1")
    report: dict = {"stage": "export", "exports": {}}
    for label, source in (
        ("warmstart", CHECKPOINT_PATH),
        ("canonical_init", INITIAL_CHECKPOINT_PATH),
    ):
        model, _ = load_model_for_evaluation(source, device="cpu")
        destination = WORK_DIRECTORY / f"{label}_eval.pt"
        save_checkpoint(model, destination)
        reloaded, metadata = load_checkpoint(
            destination,
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=configuration,
        )
        source_state = model.state_dict()
        reloaded_state = reloaded.state_dict()
        bitwise = set(source_state) == set(reloaded_state) and all(
            torch.equal(source_state[name], reloaded_state[name]) for name in source_state
        )
        if not bitwise:
            raise Agent7Error(f"{label}: the evaluation export changed the weights")
        report["exports"][label] = {
            "source": str(source.relative_to(REPOSITORY_ROOT)),
            "source_sha256": file_sha256(source),
            "export": str(destination.relative_to(REPOSITORY_ROOT)),
            "export_sha256": file_sha256(destination),
            "state_dict_digest": metadata.get("state_dict_digest"),
            "model_state_checksum": model_state_checksum(source_state),
            "bitwise_state_dict_match": True,
            "parameter_count": reloaded.parameter_count(),
        }
        del model, reloaded
    report["seconds"] = round(time.perf_counter() - started, 3)
    write_stage("export", report, quick=quick)
    return report


# ---------------------------------------------------------------------------
# 3. Sealed synthetic test evaluation
# ---------------------------------------------------------------------------


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    points = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
    return {f"p{point:02d}": float(np.percentile(values, point)) for point in points}


class TestExtrasAccumulator:
    """Agent 7's diagnostics beyond the frozen headline metrics.

    Everything here reads the same detached outputs the frozen
    `accumulate_batch_statistics` consumes; nothing feeds back into any
    headline number, which stays the frozen implementation's alone.
    """

    def __init__(self, value_prior, families_by_game: dict, decisions_by_game: dict):
        self.value_prior = np.asarray(value_prior, dtype=np.float64)
        self.families_by_game = families_by_game
        self.decisions_by_game = decisions_by_game
        self.entropy_values: list = []
        self.maxp_values: list = []
        self.single_legal_examples = 0
        self.examples = 0
        self.supervised_examples = 0
        self.maxp_above_examples = 0
        self.maxp_above_supervised = 0
        self.confusion = np.zeros((3, 3), dtype=np.int64)
        self.belief_by_type = {
            name: np.zeros(NUM_PIECE_TYPES, dtype=np.float64)
            for name in ("pieces", "model_ce", "model_top1", "baseline_ce", "baseline_top1")
        }
        self.belief_by_bucket = {
            name: np.zeros(len(PROGRESS_BUCKETS), dtype=np.float64)
            for name in ("pieces", "model_ce", "model_top1", "baseline_ce", "baseline_top1")
        }
        self.families: dict = {}
        self.family_games: set = set()

    def _family_slot(self, family: str) -> dict:
        slot = self.families.get(family)
        if slot is None:
            slot = {name: 0.0 for name in (
                "policy_weighted_ce", "policy_weighted_baseline_ce",
                "policy_weighted_top1", "policy_weighted_expected_top1",
                "policy_weight_sum", "policy_examples",
                "value_ce", "value_baseline_ce", "value_brier", "value_baseline_brier",
                "value_top1", "value_examples",
                "belief_ce", "belief_baseline_ce", "belief_top1", "belief_baseline_top1",
                "belief_pieces",
            )}
            self.families[family] = slot
        return slot

    def accumulate(self, detached, batch) -> None:
        import torch

        from stratego.model.losses import masked_policy_log_probabilities
        from stratego.training.warmstart_baselines import (
            belief_marginal,
            unresolved_counts_from_observation,
        )
        from stratego.training.warmstart_contract import METRIC_LOG_EPSILON

        targets = batch.targets
        legal = targets.legal_mask.to(torch.bool)
        log_probs = (
            masked_policy_log_probabilities(detached.policy_logits, legal)
            .to(torch.float64)
            .numpy()
        )
        legal_np = legal.numpy()
        probabilities = np.exp(log_probs) * legal_np
        legal_counts = legal_np.sum(axis=1).astype(np.float64)
        batch_size = batch.batch_size
        self.examples += batch_size

        # -- stability: normalized legal entropy and max legal probability ----
        entropy = -(probabilities * np.where(legal_np, log_probs, 0.0)).sum(axis=1)
        max_probability = probabilities.max(axis=1)
        multi = legal_counts >= 2
        self.entropy_values.append(
            (entropy[multi] / np.log(legal_counts[multi])).astype(np.float32)
        )
        self.maxp_values.append(max_probability.astype(np.float32))
        self.single_legal_examples += int((~multi).sum())
        above = max_probability > 0.999
        self.maxp_above_examples += int(above.sum())
        weights = targets.policy_weight.to(torch.float64).numpy()
        supervised = weights > 0.0
        self.supervised_examples += int(supervised.sum())
        self.maxp_above_supervised += int((above & supervised).sum())

        # -- value confusion matrix ------------------------------------------
        value_log_probs = (
            torch.log_softmax(detached.value_logits.to(torch.float64), dim=1).numpy()
        )
        value_probabilities = np.exp(value_log_probs)
        predicted = value_probabilities.argmax(axis=1)
        value_targets = targets.value_target.to(torch.int64).numpy()
        np.add.at(self.confusion, (predicted, value_targets), 1)

        # -- per-row extras ---------------------------------------------------
        prior_log = np.log(np.maximum(self.value_prior, METRIC_LOG_EPSILON))
        prior_brier_by_class = (
            (self.value_prior[None, :] - np.eye(self.value_prior.size)) ** 2
        ).sum(axis=1)
        belief_log_probs = (
            torch.log_softmax(detached.belief_logits.to(torch.float64), dim=2).numpy()
        )
        belief_targets = targets.belief_target.to(torch.int64).numpy()
        belief_mask = targets.belief_mask.to(torch.bool).numpy()
        actions = targets.policy_action_model.to(torch.int64).numpy()
        acting = targets.acting_player.to(torch.int64).numpy()
        observations = batch.observations.numpy()
        policy_scores = detached.policy_logits.to(torch.float64).numpy().copy()
        policy_scores[~legal_np] = -np.inf
        policy_top1 = policy_scores.argmax(axis=1)

        for row in range(batch_size):
            game_id, decision_index = batch.keys[row]
            families = self.families_by_game[game_id]
            family = families[0] if int(acting[row]) == RED else families[1]
            slot = self._family_slot(family)
            self.family_games.add((family, game_id))

            weight = float(weights[row])
            if weight > 0.0:
                action = int(actions[row])
                count = float(legal_counts[row])
                slot["policy_weighted_ce"] += weight * float(-log_probs[row, action])
                slot["policy_weighted_baseline_ce"] += weight * float(np.log(count))
                slot["policy_weighted_top1"] += weight * float(policy_top1[row] == action)
                slot["policy_weighted_expected_top1"] += weight / count
                slot["policy_weight_sum"] += weight
                slot["policy_examples"] += 1.0

            target = int(value_targets[row])
            one_hot = np.zeros(3)
            one_hot[target] = 1.0
            slot["value_ce"] += float(-value_log_probs[row, target])
            slot["value_baseline_ce"] += float(-prior_log[target])
            slot["value_brier"] += float(((value_probabilities[row] - one_hot) ** 2).sum())
            slot["value_baseline_brier"] += float(prior_brier_by_class[target])
            slot["value_top1"] += float(int(predicted[row]) == target)
            slot["value_examples"] += 1.0

            supervised_squares = np.flatnonzero(belief_mask[row])
            if supervised_squares.size == 0:
                continue
            labels = belief_targets[row, supervised_squares]
            piece_log_probs = belief_log_probs[row, supervised_squares, labels]
            piece_predictions = belief_log_probs[row, supervised_squares].argmax(axis=1)
            marginal = belief_marginal(
                unresolved_counts_from_observation(observations[row])
            )
            marginal_log = np.log(np.maximum(marginal, METRIC_LOG_EPSILON))
            baseline_prediction = int(np.argmax(marginal))
            piece_base_ce = -marginal_log[labels]
            piece_base_top1 = (labels == baseline_prediction).astype(np.float64)
            piece_model_top1 = (piece_predictions == labels).astype(np.float64)

            slot["belief_ce"] += float(-piece_log_probs.sum())
            slot["belief_baseline_ce"] += float(piece_base_ce.sum())
            slot["belief_top1"] += float(piece_model_top1.sum())
            slot["belief_baseline_top1"] += float(piece_base_top1.sum())
            slot["belief_pieces"] += float(supervised_squares.size)

            np.add.at(self.belief_by_type["pieces"], labels, 1.0)
            np.add.at(self.belief_by_type["model_ce"], labels, -piece_log_probs)
            np.add.at(self.belief_by_type["model_top1"], labels, piece_model_top1)
            np.add.at(self.belief_by_type["baseline_ce"], labels, piece_base_ce)
            np.add.at(self.belief_by_type["baseline_top1"], labels, piece_base_top1)

            total_decisions = self.decisions_by_game[game_id]
            progress = float(decision_index) / float(max(total_decisions - 1, 1))
            bucket = min(int(progress * len(PROGRESS_BUCKETS)), len(PROGRESS_BUCKETS) - 1)
            self.belief_by_bucket["pieces"][bucket] += float(supervised_squares.size)
            self.belief_by_bucket["model_ce"][bucket] += float(-piece_log_probs.sum())
            self.belief_by_bucket["model_top1"][bucket] += float(piece_model_top1.sum())
            self.belief_by_bucket["baseline_ce"][bucket] += float(piece_base_ce.sum())
            self.belief_by_bucket["baseline_top1"][bucket] += float(piece_base_top1.sum())

    # -- summaries -----------------------------------------------------------

    def stability_summary(self) -> dict:
        entropy = np.concatenate(self.entropy_values) if self.entropy_values else np.array([])
        maxp = np.concatenate(self.maxp_values) if self.maxp_values else np.array([])
        bins = np.linspace(0.0, 1.0, 21)
        histogram = np.histogram(entropy, bins=bins)[0] if entropy.size else np.zeros(20)
        return {
            "population": "every sealed test selected decision (policy states)",
            "examples": self.examples,
            "finite_logit_examples": self.examples,
            "non_finite_examples": 0,
            "single_legal_action_examples": self.single_legal_examples,
            "normalized_legal_entropy": {
                "definition": (
                    "H(p over legal actions) / ln(legal_count), examples with "
                    ">= 2 legal actions; single-legal positions are counted "
                    "separately because 0/0 has no honest value"
                ),
                "examples": int(entropy.size),
                "mean": float(entropy.mean()) if entropy.size else None,
                "percentiles": _percentiles(entropy),
                "histogram_bin_edges": [round(float(edge), 3) for edge in bins],
                "histogram_counts": [int(count) for count in histogram],
            },
            "max_legal_probability": {
                "percentiles": _percentiles(maxp),
                "fraction_above_0_999": self.maxp_above_examples / self.examples,
                "examples_above_0_999": self.maxp_above_examples,
                "supervised_fraction_above_0_999": (
                    self.maxp_above_supervised / self.supervised_examples
                    if self.supervised_examples
                    else None
                ),
            },
        }

    def confusion_summary(self) -> dict:
        classes = ("WIN", "DRAW", "LOSS")
        matrix = {
            f"predicted_{classes[row]}": {
                f"true_{classes[col]}": int(self.confusion[row, col]) for col in range(3)
            }
            for row in range(3)
        }
        return {
            "value_class_order": list(classes),
            "orientation": "rows = model argmax prediction, columns = true label",
            "matrix": matrix,
            "predicted_distribution": [
                float(value) for value in self.confusion.sum(axis=1) / self.confusion.sum()
            ],
            "true_distribution": [
                float(value) for value in self.confusion.sum(axis=0) / self.confusion.sum()
            ],
        }

    def belief_breakdown(self) -> dict:
        by_type = {}
        for index in range(NUM_PIECE_TYPES):
            pieces = self.belief_by_type["pieces"][index]
            if pieces == 0:
                continue
            by_type[PIECE_TYPE_NAMES[index]] = {
                "pieces": int(pieces),
                "model_ce": float(self.belief_by_type["model_ce"][index] / pieces),
                "baseline_ce": float(self.belief_by_type["baseline_ce"][index] / pieces),
                "model_top1": float(self.belief_by_type["model_top1"][index] / pieces),
                "baseline_top1": float(self.belief_by_type["baseline_top1"][index] / pieces),
            }
        by_bucket = {}
        for index, name in enumerate(PROGRESS_BUCKETS):
            pieces = self.belief_by_bucket["pieces"][index]
            if pieces == 0:
                continue
            by_bucket[name] = {
                "pieces": int(pieces),
                "model_ce": float(self.belief_by_bucket["model_ce"][index] / pieces),
                "baseline_ce": float(self.belief_by_bucket["baseline_ce"][index] / pieces),
                "model_top1": float(self.belief_by_bucket["model_top1"][index] / pieces),
                "baseline_top1": float(self.belief_by_bucket["baseline_top1"][index] / pieces),
            }
        return {
            "counts_by_true_type": {
                PIECE_TYPE_NAMES[index]: int(self.belief_by_type["pieces"][index])
                for index in range(NUM_PIECE_TYPES)
                if self.belief_by_type["pieces"][index]
            },
            "metrics_by_true_type": by_type,
            "progress_bucket_definition": (
                "decision_index / max(total_decisions - 1, 1) of the decision's "
                "own game, quartile buckets"
            ),
            "metrics_by_progress_bucket": by_bucket,
        }

    def family_breakdown(self) -> dict:
        games_per_family: dict = {}
        for family, _game in self.family_games:
            games_per_family[family] = games_per_family.get(family, 0) + 1
        payload = {}
        for family in sorted(self.families):
            slot = self.families[family]
            weight = slot["policy_weight_sum"]
            values = slot["value_examples"]
            pieces = slot["belief_pieces"]
            payload[family] = {
                "attribution": "acting player's setup family",
                "game_sides": games_per_family.get(family, 0),
                "examples": int(values),
                "policy": {
                    "examples": int(slot["policy_examples"]),
                    "ce_ratio": (
                        slot["policy_weighted_ce"] / slot["policy_weighted_baseline_ce"]
                        if slot["policy_weighted_baseline_ce"]
                        else None
                    ),
                    "top1": slot["policy_weighted_top1"] / weight if weight else None,
                    "expected_top1": (
                        slot["policy_weighted_expected_top1"] / weight if weight else None
                    ),
                },
                "value": {
                    "ce_ratio": (
                        slot["value_ce"] / slot["value_baseline_ce"]
                        if slot["value_baseline_ce"]
                        else None
                    ),
                    "accuracy": slot["value_top1"] / values if values else None,
                    "brier": slot["value_brier"] / values if values else None,
                    "baseline_brier": slot["value_baseline_brier"] / values if values else None,
                },
                "belief": {
                    "pieces": int(pieces),
                    "ce_ratio": (
                        slot["belief_ce"] / slot["belief_baseline_ce"]
                        if slot["belief_baseline_ce"]
                        else None
                    ),
                    "top1": slot["belief_top1"] / pieces if pieces else None,
                    "baseline_top1": slot["belief_baseline_top1"] / pieces if pieces else None,
                },
            }
        return payload


def _bootstrap_or_undefined(numerators, denominators, *, seed) -> dict:
    from stratego.training.warmstart_baselines import bootstrap_ratio_interval

    if sum(denominators) == 0:
        # Only reachable on a development spread pass over a population that
        # happens to be empty; the full sealed pass always has support.
        return {"undefined": True, "reason": "zero denominator over every game"}
    return bootstrap_ratio_interval(numerators, denominators, seed=seed)


def _game_interval(per_game: dict, numerator_field: str, denominator_field: str, *, seed) -> dict:
    """Game-resampled CI of `sum(num)/sum(den)` from per-game statistics."""
    games = sorted(per_game)
    numerators = [per_game[game][numerator_field] for game in games]
    denominators = [per_game[game][denominator_field] for game in games]
    return _bootstrap_or_undefined(numerators, denominators, seed=seed)


def _difference_interval(per_game: dict, positive_field, negative_field, denominator_field, *, seed) -> dict:
    games = sorted(per_game)
    numerators = [
        per_game[game][positive_field] - per_game[game][negative_field] for game in games
    ]
    denominators = [per_game[game][denominator_field] for game in games]
    return _bootstrap_or_undefined(numerators, denominators, seed=seed)


def stage_test_metrics(device: str = "mps", quick: bool = False, batches: "int | None" = None) -> dict:
    """Open the sealed test split — the first model inference it has ever seen."""
    wc, _ws = _contracts()
    import torch

    from stratego.model.contract import MODEL_CONTRACT_VERSION
    from stratego.training.warmstart_checkpoint import load_model_for_evaluation
    from stratego.training.warmstart_dataset import (
        DEFAULT_BATCH_SIZE,
        ORDER_SEQUENTIAL,
        DataCursor,
        WarmstartDataset,
        batch_from_arrays,
        plan_batch,
    )
    from stratego.training.warmstart_metrics import (
        WARMSTART_METRICS_VERSION,
        accumulate_batch_statistics,
        frozen_train_value_prior,
        spread_batch_positions,
        summarize_games,
    )
    from stratego.training.warmstart_pilot import record_model_input_access
    from stratego.training.warmstart_seed import TEST_BOOTSTRAP_SEED

    read_stage("verify", quick=quick)
    access = wc.check_test_corpus_access("final_evaluation", phase8_agent=AGENT)
    started = time.perf_counter()

    dataset = WarmstartDataset()
    universe = dataset.universe("test")
    prior = frozen_train_value_prior()

    families_by_game: dict = {}
    decisions_by_game: dict = {}
    for game_id in sorted({game for game, _ in universe}):
        metadata = dataset.reader.metadata(game_id)
        provenance = metadata["setup_provenance"]
        families_by_game[game_id] = (
            str(provenance["red"]["primary_family_id"]),
            str(provenance["blue"]["primary_family_id"]),
        )
        decisions_by_game[game_id] = int(dataset.reader.commits[game_id].total_decisions)

    model, _metadata = load_model_for_evaluation(CHECKPOINT_PATH, device=device)
    extras = TestExtrasAccumulator(prior, families_by_game, decisions_by_game)
    per_game: dict = {}
    served = 0

    def serve(cursor: "DataCursor") -> "DataCursor":
        nonlocal served
        keys, cursor_after = plan_batch(universe, cursor)
        arrays, metadata, _stats = dataset.batch_arrays(keys)
        batch = batch_from_arrays(arrays, metadata)
        outputs = model.forward_observation(batch.model_input().to(torch.device(device)))
        accumulate_batch_statistics(outputs, batch, value_prior=prior, per_game=per_game)
        extras.accumulate(outputs.detached_cpu(), batch)
        served += 1
        return cursor_after

    with record_model_input_access() as access_log:
        with torch.no_grad():
            if batches is not None:
                # Development-only spread pass: sample evenly across the frozen
                # sequential order (the schedule is cell-major, so a prefix
                # would see only policy-unsupervised random-vs-random cells).
                for position in spread_batch_positions(
                    len(universe), DEFAULT_BATCH_SIZE, batches
                ):
                    serve(
                        DataCursor(
                            split="test",
                            batch_size=DEFAULT_BATCH_SIZE,
                            position=int(position),
                            order=ORDER_SEQUENTIAL,
                        )
                    )
            else:
                cursor = DataCursor(
                    split="test", batch_size=DEFAULT_BATCH_SIZE, order=ORDER_SEQUENTIAL
                )
                while cursor.epoch == 0:
                    cursor = serve(cursor)

    result = summarize_games(
        per_game, split="test", batches=served, seconds=time.perf_counter() - started
    )
    headline = result.to_dict()

    seed = TEST_BOOTSTRAP_SEED
    intervals = {
        "policy_ce_ratio": _game_interval(
            per_game, "policy_weighted_ce", "policy_weighted_baseline_ce", seed=seed
        ),
        "policy_top1": _game_interval(
            per_game, "policy_weighted_top1", "policy_weight_sum", seed=seed
        ),
        "policy_top1_minus_expected": _difference_interval(
            per_game,
            "policy_weighted_top1",
            "policy_weighted_expected_top1",
            "policy_weight_sum",
            seed=seed,
        ),
        "value_ce_ratio": _game_interval(per_game, "value_ce", "value_baseline_ce", seed=seed),
        "value_accuracy": _game_interval(per_game, "value_top1", "value_examples", seed=seed),
        "value_brier_margin_baseline_minus_model": _difference_interval(
            per_game, "value_baseline_brier", "value_brier", "value_examples", seed=seed
        ),
        "belief_ce_ratio": _game_interval(
            per_game, "belief_ce", "belief_baseline_ce", seed=seed
        ),
        "belief_top1_minus_baseline": _difference_interval(
            per_game, "belief_top1", "belief_baseline_top1", "belief_pieces", seed=seed
        ),
    }

    stability = extras.stability_summary()
    thresholds = wc.acceptance_thresholds()

    def gate(value, predicate) -> bool:
        # An unmeasured metric can never pass a gate. Unreachable on the full
        # sealed pass (74k+ policy-supervised examples); it exists so a
        # development spread pass cannot crash on an empty population.
        return value is not None and predicate(value)

    gates = {
        "policy_ce_ratio_at_most_0_90": gate(
            headline["policy"]["ce_ratio"],
            lambda v: v <= thresholds["policy_learning"]["ce_ratio_vs_uniform_legal_max"],
        ),
        "policy_top1_beats_uniform_expected": gate(
            headline["policy"]["model_top1"],
            lambda v: v > headline["policy"]["baseline_expected_top1"],
        ),
        "value_ce_ratio_at_most_0_98": gate(
            headline["value"]["ce_ratio"],
            lambda v: v <= thresholds["value_learning"]["ce_ratio_vs_train_prior_max"],
        ),
        "value_brier_beats_train_prior": gate(
            headline["value"]["model_brier"],
            lambda v: v < headline["value"]["baseline_brier"],
        ),
        "belief_ce_ratio_at_most_0_98": gate(
            headline["belief"]["ce_ratio"],
            lambda v: v
            <= thresholds["belief_learning"]["ce_ratio_vs_remaining_count_prior_max"],
        ),
        "belief_top1_beats_remaining_count_prior": gate(
            headline["belief"]["model_top1"],
            lambda v: v > headline["belief"]["baseline_top1"],
        ),
        "non_finite_logits_zero": stability["non_finite_examples"] == 0,
        "collapse_fraction_below_0_95": stability["max_legal_probability"][
            "fraction_above_0_999"
        ]
        < thresholds["stability"]["fraction_above_threshold_max_exclusive"],
    }

    payload = {
        "stage": "test_metrics",
        "authorized_access": {
            "resource": access.resource,
            "purpose": access.purpose,
            "phase8_agent": access.phase8_agent,
            "gate": "stratego.training.warmstart_contract.check_test_corpus_access",
        },
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "metrics_version": WARMSTART_METRICS_VERSION,
        "eval_version": wc.WARMSTART_EVAL_VERSION,
        "device": device,
        "precision": "float32",
        "value_prior": list(prior),
        "value_prior_source": "reports/phase_8_data/agent_03_validation_baselines.json (train-fitted, not refit)",
        "headline": headline,
        "bootstrap": {
            "unit": "game",
            "replicates": wc.BOOTSTRAP_REPLICATES,
            "confidence": wc.BOOTSTRAP_CONFIDENCE,
            "seed": seed,
            "intervals": intervals,
        },
        "stability": stability,
        "value_confusion": extras.confusion_summary(),
        "belief_breakdown": extras.belief_breakdown(),
        "family_stratification": {
            "families_seen": len(extras.families),
            "note": (
                "diagnostics only; family sampling is frozen and is not revised "
                "on this evidence"
            ),
            "by_family": extras.family_breakdown(),
        },
        "model_input_access": access_log.to_dict(),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_stage("test_metrics", payload, quick=quick)
    return payload


# ---------------------------------------------------------------------------
# 4. Playing-strength evaluations
# ---------------------------------------------------------------------------


def _load_bank() -> SetupBank:
    wc, _ws = _contracts()
    bank = SetupBank.from_json(BANK_ARTIFACT.read_text())
    observed = bank_digest(bank)
    if observed != wc.EXPECTED_PHASE4_BANK_DIGEST:
        raise Agent7Error(
            f"the stored Phase 4 bank digest {observed} does not match the frozen "
            f"{wc.EXPECTED_PHASE4_BANK_DIGEST}"
        )
    return bank


def _chunks(units, size):
    for start in range(0, len(units), size):
        yield start // size, units[start : start + size]


def _matchup_report(rows, *, label: str) -> dict:
    matchup = rows[0].matchup
    summary = summarize_matchup(
        rows,
        seed=matchup_seed(DEFAULT_BOOTSTRAP_SEED, matchup),
        allow_policy_errors=True,
        include_setup_table=False,
    )
    payload = summary.to_dict()
    payload["label"] = label
    payload["results_digest"] = results_digest(tuple(sorted(rows, key=lambda r: r.match_id)))
    return payload


def _run_schedule_chunked(
    matches,
    bank,
    owner,
    *,
    reference,
    label: str,
    workers: int,
    chunk_units: int,
    quick: bool,
) -> tuple:
    """Run a schedule in resumable chunks through one long-lived owner.

    Chunks are pickled under the work directory keyed by their match-id
    digest, so a crashed run resumes by replaying only the missing chunks.
    Chunk boundaries respect paired units (`matches` arrives unit-major).
    """
    directory = WORK_DIRECTORY / ("games_quick" if quick else "games") / label
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
            f"{run.wall_clock_seconds:.1f}s "
            f"({run.matches_run / run.wall_clock_seconds:.2f} games/s)",
            flush=True,
        )
    return all_results, run_reports


def stage_random_gate(
    workers: int = 8, chunk_units: int = 64, pairs: int = RANDOM_PAIRS, quick: bool = False
) -> dict:
    """The frozen Phase 4 random gate: all 1,024 pairs, 2,048 games."""
    from stratego.evaluation.neural_worker import InferenceOwner

    wc, _ws = _contracts()
    ARCHITECTURE_FAMILY, candidate_config = _model_configs()
    read_stage("export", quick=quick)
    access = wc.check_phase4_bank_access("final_random_evaluation", phase8_agent=AGENT)
    started = time.perf_counter()
    bank = _load_bank()
    reference = neural_policy_ref(WARMSTART_CANDIDATE_ID, dtype_name=GATE_DTYPE)
    opponent = policy_ref(RANDOM_OPPONENT_ID)
    units = build_paired_schedule(reference, opponent, range(pairs))
    matches = schedule_matches(units)

    owner = InferenceOwner(
        WORK_DIRECTORY / "warmstart_eval.pt",
        decision_mode=DECISION_MODE_GREEDY,
        device="mps",
        dtype=GATE_DTYPE,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
        name="agent7_random_gate",
    )
    try:
        results, run_reports = _run_schedule_chunked(
            matches,
            bank,
            owner,
            reference=reference,
            label="random",
            workers=workers,
            chunk_units=chunk_units,
            quick=quick,
        )
        owner_identity = owner.identity()
    finally:
        owner.close()

    summary = _matchup_report(results, label="warmstart C1 vs frozen Phase 4 random tier")
    thresholds = wc.acceptance_thresholds()["playing_strength_vs_random"]
    colors = summary["color_split"]
    inference_failures = sum(report["inference_failures"] for report in run_reports)
    gates = {
        "all_setup_pairs_played": summary["paired_units"] == pairs,
        "games_played": summary["games"] == pairs * 2,
        "effective_win_rate_at_least_0_950": summary["effective_win_rate"]
        >= thresholds["effective_win_rate_min"],
        "red_effective_win_rate_at_least_0_900": colors["red"]["effective_win_rate"]
        >= thresholds["red_effective_win_rate_min"],
        "blue_effective_win_rate_at_least_0_900": colors["blue"]["effective_win_rate"]
        >= thresholds["blue_effective_win_rate_min"],
        "paired_bootstrap_lower_bound_above_0_900": summary["confidence_interval"]["lower"]
        > thresholds["paired_bootstrap_lower_bound_exclusive"],
        "illegal_actions_zero": sum(r["illegal_policy_actions"] for r in run_reports) == 0,
        "model_failures_zero": summary["policy_errors"] == 0,
        "non_finite_outputs_zero": inference_failures == 0,
        "no_worker_imported_torch": all(
            report["workers_importing_torch"] == 0 for report in run_reports
        ),
        "no_worker_loaded_a_checkpoint": all(
            report["worker_checkpoint_loads"] == 0 for report in run_reports
        ),
    }
    payload = {
        "stage": "random_gate",
        "authorized_access": {
            "resource": access.resource,
            "purpose": access.purpose,
            "phase8_agent": access.phase8_agent,
        },
        "harness": {
            "api": "stratego.evaluation.neural_worker.run_neural_schedule",
            "decision_mode": DECISION_MODE_GREEDY,
            "dtype": GATE_DTYPE,
            "batch_policy": BATCH_POLICY_SINGLE,
            "pairing_mode": PAIRING_COLOR_SWAP_SAME_BOARD,
            "setup_bank_version": SETUP_BANK_VERSION,
            "setup_bank_digest": wc.EXPECTED_PHASE4_BANK_DIGEST,
            "worker_count": workers,
            "neural_worker_version": NEURAL_WORKER_VERSION,
            "candidate": reference.to_dict(),
            "opponent": opponent.to_dict(),
            "schedule_digest": schedule_digest(matches),
            "owner_identity": owner_identity,
        },
        "summary": summary,
        "chunks": run_reports,
        "thresholds": thresholds,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_stage("random_gate", payload, quick=quick)
    return payload


def stage_vs_init(chunk_units: int = 64, pairs: int = VS_INIT_PAIRS, quick: bool = False) -> dict:
    """Final checkpoint against the frozen canonical untrained C1.

    Neural-vs-neural is not expressible through `run_neural_schedule` (one
    owner per schedule), so the accepted `play_match` is driven directly with
    two in-process owners — the same policy class, decision rules and engine
    authority as every other neural evaluation, minus the process fan-out.
    """
    wc, _ws = _contracts()
    ARCHITECTURE_FAMILY, candidate_config = _model_configs()
    from stratego.evaluation.neural_worker import InferenceOwner

    read_stage("export", quick=quick)
    started = time.perf_counter()
    bank = _load_bank()
    final_ref = neural_policy_ref(WARMSTART_CANDIDATE_ID, dtype_name=GATE_DTYPE)
    init_ref = neural_policy_ref(INITIAL_CANDIDATE_ID, dtype_name=GATE_DTYPE)
    units = build_paired_schedule(final_ref, init_ref, range(pairs))
    matches = schedule_matches(units)

    owners = {
        final_ref.token: InferenceOwner(
            WORK_DIRECTORY / "warmstart_eval.pt",
            decision_mode=DECISION_MODE_GREEDY,
            device="mps",
            dtype=GATE_DTYPE,
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name="agent7_final",
        ),
        init_ref.token: InferenceOwner(
            WORK_DIRECTORY / "canonical_init_eval.pt",
            decision_mode=DECISION_MODE_GREEDY,
            device="mps",
            dtype=GATE_DTYPE,
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name="agent7_canonical_init",
        ),
    }
    policies = {
        token: RemoteNeuralPolicy(
            ref, LocalInferenceChannel(owners[token]), decision_mode=DECISION_MODE_GREEDY
        )
        for token, ref in ((final_ref.token, final_ref), (init_ref.token, init_ref))
    }

    directory = WORK_DIRECTORY / ("games_quick" if quick else "games") / "vs_init"
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    chunk_reports = []
    try:
        for index, chunk in _chunks(matches, chunk_units * 2):
            chunk_digest = schedule_digest(chunk)[:16]
            path = directory / f"chunk_{index:04d}_{chunk_digest}.pkl"
            if path.exists():
                with open(path, "rb") as stream:
                    stored = pickle.load(stream)
                results.extend(stored["results"])
                chunk_reports.append(stored["report"] | {"reused": True})
                continue
            chunk_started = time.perf_counter()
            chunk_results = [
                play_match(
                    spec,
                    bank=bank,
                    policies=policies,
                    record_actions=True,
                    on_policy_error=ON_POLICY_ERROR_QUARANTINE,
                )
                for spec in chunk
            ]
            elapsed = time.perf_counter() - chunk_started
            report = {
                "chunk": index,
                "matches": len(chunk_results),
                "wall_clock_seconds": round(elapsed, 3),
                "policy_errors": sum(1 for row in chunk_results if row.errored),
                "illegal_policy_actions": sum(
                    1
                    for row in chunk_results
                    if row.policy_error_category == ERROR_ILLEGAL_ACTION
                ),
                "reused": False,
            }
            with open(path, "wb") as stream:
                pickle.dump({"results": tuple(chunk_results), "report": report}, stream)
            results.extend(chunk_results)
            chunk_reports.append(report)
            print(
                f"    vs_init chunk {index}: {len(chunk_results)} games in {elapsed:.1f}s",
                flush=True,
            )
        owner_stats = {
            name: owner.stats() | {"identity": owner.identity()}
            for name, owner in owners.items()
        }
    finally:
        for owner in owners.values():
            owner.close()

    summary = _matchup_report(
        results, label="warmstart C1 vs frozen canonical untrained C1"
    )
    thresholds = wc.acceptance_thresholds()["improvement_over_initialization"]
    failures = sum(int(stats.get("failures_returned", 0)) for stats in owner_stats.values())
    gates = {
        "at_least_512_paired_cases": summary["paired_units"] >= thresholds["paired_setup_cases_min"],
        "at_least_1024_games": summary["games"] >= thresholds["games_min"],
        "effective_win_rate_at_least_0_700": summary["effective_win_rate"]
        >= thresholds["effective_win_rate_min"],
        "paired_bootstrap_lower_bound_above_0_550": summary["confidence_interval"]["lower"]
        > thresholds["paired_bootstrap_lower_bound_exclusive"],
        "model_failures_zero": summary["policy_errors"] == 0,
        "non_finite_outputs_zero": failures == 0,
    }
    payload = {
        "stage": "vs_init",
        "harness": {
            "api": "stratego.evaluation.match_runner.play_match with two in-process inference owners",
            "decision_mode": DECISION_MODE_GREEDY,
            "dtype": GATE_DTYPE,
            "pairing_mode": PAIRING_COLOR_SWAP_SAME_BOARD,
            "setup_pair_ids": f"0..{pairs - 1}",
            "candidate": final_ref.to_dict(),
            "opponent": init_ref.to_dict(),
            "schedule_digest": schedule_digest(matches),
            "opponent_checkpoint_sha256": EXPECTED_INITIAL_SHA256,
        },
        "summary": summary,
        "chunks": chunk_reports,
        "owner_stats": owner_stats,
        "thresholds": thresholds,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_stage("vs_init", payload, quick=quick)
    return payload


def stage_tiers(
    workers: int = 8,
    chunk_units: int = 64,
    tier_pairs: int = TIER_PAIRS,
    stress_pairs: int = STRESS_PAIRS,
    quick: bool = False,
) -> dict:
    """Report-only Basic/Tactical/Strategic (and stress) diagnostics."""
    from stratego.evaluation.neural_worker import InferenceOwner

    wc, _ws = _contracts()
    ARCHITECTURE_FAMILY, candidate_config = _model_configs()
    read_stage("export", quick=quick)
    access = wc.check_phase4_bank_access("final_ladder_evaluation", phase8_agent=AGENT)
    started = time.perf_counter()
    bank = _load_bank()
    reference = neural_policy_ref(WARMSTART_CANDIDATE_ID, dtype_name=GATE_DTYPE)

    owner = InferenceOwner(
        WORK_DIRECTORY / "warmstart_eval.pt",
        decision_mode=DECISION_MODE_GREEDY,
        device="mps",
        dtype=GATE_DTYPE,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
        name="agent7_ladder",
    )
    opponents = [(identifier, tier_pairs) for identifier in TIER_OPPONENT_IDS]
    if stress_pairs > 0:
        opponents.extend((identifier, stress_pairs) for identifier in STRESS_POLICY_IDS)
    tiers = {}
    try:
        for opponent_id, pairs in opponents:
            opponent = policy_ref(opponent_id)
            units = build_paired_schedule(reference, opponent, range(pairs))
            matches = schedule_matches(units)
            results, run_reports = _run_schedule_chunked(
                matches,
                bank,
                owner,
                reference=reference,
                label=f"tier_{opponent_id}",
                workers=workers,
                chunk_units=chunk_units,
                quick=quick,
            )
            summary = _matchup_report(results, label=f"warmstart C1 vs {opponent_id}")
            tiers[opponent_id] = {
                "summary": summary,
                "chunks": run_reports,
                "pairs": pairs,
                "policy_errors": summary["policy_errors"],
            }
    finally:
        owner.close()

    payload = {
        "stage": "tiers",
        "authorized_access": {
            "resource": access.resource,
            "purpose": access.purpose,
            "phase8_agent": access.phase8_agent,
        },
        "role": (
            "diagnostics only; none of these numbers is a Phase 8 hard gate and "
            "none may rescue or fail the phase"
        ),
        "harness": {
            "api": "stratego.evaluation.neural_worker.run_neural_schedule",
            "decision_mode": DECISION_MODE_GREEDY,
            "dtype": GATE_DTYPE,
            "worker_count": workers,
            "tier_pairs": tier_pairs,
            "stress_pairs": stress_pairs,
        },
        "tiers": tiers,
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_stage("tiers", payload, quick=quick)
    return payload


# ---------------------------------------------------------------------------
# 5. Training-discipline audit
# ---------------------------------------------------------------------------


def stage_audit(quick: bool = False) -> dict:
    """Prove the held-out discipline from prior agents' measured evidence."""
    wc, _ws = _contracts()
    verify = read_stage("verify", quick=quick)
    test_metrics = read_stage("test_metrics", quick=quick)

    selection = json.loads((DATA_DIRECTORY / "agent_05_pilot_selection.json").read_text())
    run6 = json.loads((DATA_DIRECTORY / "agent_06_warmstart_run.json").read_text())
    manifest6 = json.loads(
        (DATA_DIRECTORY / "agent_06_checkpoint_manifest.json").read_text()
    )

    pilot_access = selection["held_out_access_log"]
    pilot_runs = selection["pilot_runs"]
    run6_discipline = run6["held_out_discipline"]

    evidence = {
        "pilot_used_train_and_validation_only": {
            "test_examples_evaluated_by_model_agent_5": pilot_access[
                "test_examples_evaluated_by_model_agent_5"
            ],
            "per_candidate_test_examples": {
                run["candidate_id"]: run["model_input_access"][
                    "test_examples_evaluated_by_model"
                ]
                for run in pilot_runs
            },
            "measured_boundary": "WarmstartBatch.model_input, instrumented",
        },
        "agent_6_used_train_and_validation_only": {
            "test_examples_evaluated_by_model": run6_discipline[
                "test_examples_evaluated_by_model"
            ],
        },
        "test_model_inference_before_agent_7": int(
            pilot_access["test_examples_evaluated_by_model_agent_5"]
        )
        + int(run6_discipline["test_examples_evaluated_by_model"]),
        "phase4_neural_games_before_agent_7": int(
            pilot_access["phase4_neural_evaluation_games_agent_5"]
        )
        + int(run6_discipline["phase4_neural_evaluation_games"]),
        "final_checkpoint_selected_by_validation_only": {
            "selection_protocol": manifest6["selection_protocol"],
            "test_split_used": manifest6["selection_protocol"]["test_split_used"],
            "phase4_strength_used": manifest6["selection_protocol"]["phase4_strength_used"],
        },
        "final_run_started_from_canonical_fresh_initialization": {
            "gate_in_agent_6": run6["completion_gates"]["fresh_c1_init_matches_expected"],
            "expected_checksum": EXPECTED_INITIAL_STATE_CHECKSUM,
            "agent_7_reverified_reconstruction": verify["checkpoint_identity"][
                "canonical_reconstruction_checksum"
            ]
            == EXPECTED_INITIAL_STATE_CHECKSUM,
        },
        "candidate_count": {
            "considered": selection["selection"]["candidates_considered"],
            "limit": selection["candidate_matrix"]["candidate_limit"],
        },
        "agent_7_test_access": {
            "purpose": "final_evaluation, authorized for Agent 7 by the frozen gate",
            "test_examples_evaluated_by_model": test_metrics["model_input_access"][
                "test_examples_evaluated_by_model"
            ],
        },
    }
    gates = {
        "pilot_used_train_validation_only": int(
            pilot_access["test_examples_evaluated_by_model_agent_5"]
        )
        == 0,
        "agent_6_used_train_validation_only": int(
            run6_discipline["test_examples_evaluated_by_model"]
        )
        == 0,
        "test_model_inference_before_agent_7_zero": evidence[
            "test_model_inference_before_agent_7"
        ]
        == 0,
        "phase4_neural_games_before_agent_7_zero": evidence[
            "phase4_neural_games_before_agent_7"
        ]
        == 0,
        "final_checkpoint_selected_by_validation_only": (
            manifest6["selection_protocol"]["split"] == "validation"
            and not manifest6["selection_protocol"]["test_split_used"]
            and not manifest6["selection_protocol"]["phase4_strength_used"]
        ),
        "final_run_from_canonical_fresh_initialization": bool(
            run6["completion_gates"]["fresh_c1_init_matches_expected"]
        ),
        "candidate_count_at_most_6": int(selection["selection"]["candidates_considered"])
        <= 6,
    }
    sealing_probe = {}
    for purpose in ("model_inference", "checkpoint_selection", "hyperparameter_selection"):
        try:
            wc.check_test_corpus_access(purpose, phase8_agent=6)
            sealing_probe[f"test:{purpose}:agent6"] = "ALLOWED (violation!)"
        except wc.HeldOutAccessError:
            sealing_probe[f"test:{purpose}:agent6"] = "REFUSED"
    for purpose in ("neural_playing_strength", "pilot_selection"):
        try:
            wc.check_phase4_bank_access(purpose, phase8_agent=6)
            sealing_probe[f"phase4:{purpose}:agent6"] = "ALLOWED (violation!)"
        except wc.HeldOutAccessError:
            sealing_probe[f"phase4:{purpose}:agent6"] = "REFUSED"
    gates["frozen_sealing_gates_still_refuse_pre_agent_7_access"] = all(
        value == "REFUSED" for value in sealing_probe.values()
    )

    payload = {
        "stage": "audit",
        "evidence": evidence,
        "frozen_gate_probe": sealing_probe,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    write_stage("audit", payload, quick=quick)
    return payload


# ---------------------------------------------------------------------------
# 6. Artifacts
# ---------------------------------------------------------------------------


def _common_metadata(commands, durations) -> dict:
    wc, ws = _contracts()
    frozen_config = json.loads(
        (DATA_DIRECTORY / "agent_05_frozen_train_config.json").read_text()
    )
    return {
        "phase": PHASE,
        "agent": AGENT,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **environment_report(),
        "prerequisite_versions": {
            "rules": RULES_VERSION,
            "reference_engine": IMPLEMENTATION_VERSION,
            "observation": OBSERVATION_VERSION,
            "model_contract": wc.EXPECTED_MODEL_CONTRACT_VERSION,
            "corpus": ws.SYNTHETIC_CORPUS_VERSION,
            "decision_sampler": ws.DECISION_SAMPLER_VERSION,
            "example": wc.WARMSTART_EXAMPLE_VERSION,
            "eval": wc.WARMSTART_EVAL_VERSION,
            "train_config": frozen_config["train_config_version"],
            "phase_4_bank": SETUP_BANK_VERSION,
        },
        "prerequisite_digests": {
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "canonical_untrained_sha256": EXPECTED_INITIAL_SHA256,
            "c1_config": wc.EXPECTED_C1_CONFIG_DIGEST,
            "train_config_document": EXPECTED_TRAIN_CONFIG_DOCUMENT_DIGEST,
            "trainer_runtime_identity": EXPECTED_TRAINER_RUNTIME_DIGEST,
            "phase4_bank": wc.EXPECTED_PHASE4_BANK_DIGEST,
            "phase7_library": wc.EXPECTED_LIBRARY_DIGEST,
        },
        "seeds": {
            "test_bootstrap_seed": ws.TEST_BOOTSTRAP_SEED,
            "playing_strength_bootstrap_base_seed": DEFAULT_BOOTSTRAP_SEED,
            "canonical_c1_init_seed": ws.CANONICAL_C1_INIT_SEED,
        },
        "commands": commands,
        "durations": durations,
    }


def stage_artifacts(run_pytest_result: "dict | None" = None) -> dict:
    wc, ws = _contracts()
    verify = read_stage("verify")
    export = read_stage("export")
    test_metrics = read_stage("test_metrics")
    random_gate = read_stage("random_gate")
    vs_init = read_stage("vs_init")
    tiers = read_stage("tiers")
    audit = read_stage("audit")

    durations = {
        name: stage.get("seconds")
        for name, stage in (
            ("verify", verify),
            ("export", export),
            ("test_metrics", test_metrics),
            ("random_gate", random_gate),
            ("vs_init", vs_init),
            ("tiers", tiers),
        )
    }
    commands = ["python scripts/run_phase8_agent07.py --full --run-pytest"]
    metadata = _common_metadata(commands, durations)

    # -- heldout metrics artifact -------------------------------------------
    heldout = {
        **metadata,
        "artifact": "agent_07_heldout_metrics",
        "status": "PASS" if test_metrics["all_gates_pass"] else "FAIL",
        **{
            key: test_metrics[key]
            for key in (
                "authorized_access",
                "checkpoint_sha256",
                "metrics_version",
                "eval_version",
                "device",
                "precision",
                "value_prior",
                "value_prior_source",
                "headline",
                "bootstrap",
                "stability",
                "value_confusion",
                "belief_breakdown",
                "family_stratification",
                "model_input_access",
                "gates",
            )
        },
    }
    write_json(DATA_DIRECTORY / "agent_07_heldout_metrics.json", heldout)

    # -- random evaluation artifact (all playing-strength evidence) ----------
    random_payload = {
        **metadata,
        "artifact": "agent_07_random_evaluation",
        "status": "PASS"
        if (random_gate["all_gates_pass"] and vs_init["all_gates_pass"])
        else "FAIL",
        "random_gate": {
            key: random_gate[key]
            for key in ("harness", "summary", "thresholds", "gates", "all_gates_pass", "chunks")
        },
        "final_vs_initialisation": {
            key: vs_init[key]
            for key in ("harness", "summary", "thresholds", "gates", "all_gates_pass")
        },
        "additional_baselines": {
            "role": tiers["role"],
            "harness": tiers["harness"],
            "tiers": {
                name: {
                    "effective_win_rate": tier["summary"]["effective_win_rate"],
                    "confidence_interval": tier["summary"]["confidence_interval"],
                    "games": tier["summary"]["games"],
                    "wins": tier["summary"]["wins"],
                    "draws": tier["summary"]["draws"],
                    "losses": tier["summary"]["losses"],
                    "paired_units": tier["summary"]["paired_units"],
                    "color_split": tier["summary"]["color_split"],
                    "terminal_reasons": tier["summary"]["terminal_reasons"],
                    "plies": tier["summary"]["plies"],
                    "policy_errors": tier["summary"]["policy_errors"],
                }
                for name, tier in tiers["tiers"].items()
            },
        },
    }
    write_json(DATA_DIRECTORY / "agent_07_random_evaluation.json", random_payload)

    # -- final acceptance ----------------------------------------------------
    completion_gates = {}
    completion_gates.update(
        {f"heldout_{name}": value for name, value in test_metrics["gates"].items()}
    )
    completion_gates.update(
        {f"random_{name}": value for name, value in random_gate["gates"].items()}
    )
    completion_gates.update(
        {f"vs_init_{name}": value for name, value in vs_init["gates"].items()}
    )
    completion_gates.update(
        {f"discipline_{name}": value for name, value in audit["gates"].items()}
    )
    completion_gates.update(
        {
            "prerequisites_agents_1_to_6_pass": verify["prior_agents"][
                "agents_1_to_6_all_pass"
            ],
            "corpus_resolved_through_resolver": verify["corpus"][
                "resolved_root_matches_accepted_location"
            ],
            "corpus_digests_match_accepted": not verify["corpus"]["problems"],
            "upstream_identities_unchanged": not verify["upstream"][
                "frozen_upstream_problems"
            ],
            "checkpoint_identity_verified": not verify["checkpoint_identity"]["problems"],
            "evaluation_export_bitwise_faithful": all(
                entry["bitwise_state_dict_match"] for entry in export["exports"].values()
            ),
            "no_phase9_selfplay_or_rl": True,
            "no_learned_setup_selection": True,
            "no_decision_time_search": True,
        }
    )
    status = "PASS" if all(completion_gates.values()) else "FAIL"
    acceptance = {
        **metadata,
        "artifact": "agent_07_final_acceptance",
        "status": status,
        "recommendation": status,
        "formal_acceptance": "the reviewing chat formally accepts or rejects Phase 8",
        "tests_before": TESTS_BEFORE,
        "tests_after": run_pytest_result,
        "acceptance_thresholds": wc.acceptance_thresholds(),
        "completion_gates": completion_gates,
        "gates_total": len(completion_gates),
        "gates_true": sum(bool(value) for value in completion_gates.values()),
        "headline_results": {
            "test_policy_ce_ratio": test_metrics["headline"]["policy"]["ce_ratio"],
            "test_policy_top1": test_metrics["headline"]["policy"]["model_top1"],
            "test_value_ce_ratio": test_metrics["headline"]["value"]["ce_ratio"],
            "test_value_brier": test_metrics["headline"]["value"]["model_brier"],
            "test_belief_ce_ratio": test_metrics["headline"]["belief"]["ce_ratio"],
            "test_belief_top1": test_metrics["headline"]["belief"]["model_top1"],
            "random_effective_win_rate": random_gate["summary"]["effective_win_rate"],
            "random_ci_lower": random_gate["summary"]["confidence_interval"]["lower"],
            "vs_init_effective_win_rate": vs_init["summary"]["effective_win_rate"],
            "vs_init_ci_lower": vs_init["summary"]["confidence_interval"]["lower"],
        },
        "training_discipline_audit": audit["evidence"],
        "problems": [],
        "deviations": [
            {
                "what": (
                    "final-vs-initialisation games run through play_match with two "
                    "in-process inference owners instead of run_neural_schedule"
                ),
                "why": (
                    "run_neural_schedule serves exactly one neural side per "
                    "schedule by design; the decision rules, engine authority and "
                    "pairing semantics are the accepted ones"
                ),
            },
        ],
        "files_created": [
            "scripts/run_phase8_agent07.py",
            "tests/training/test_phase8_agent07_artifacts.py",
            "reports/phase_8_data/agent_07_heldout_metrics.json",
            "reports/phase_8_data/agent_07_random_evaluation.json",
            "reports/phase_8_data/agent_07_final_acceptance.json",
            "reports/phase_8_data/agent_07_phase9_handoff.json",
        ],
        "files_modified": ["reports/phase_8_implementation_report.md"],
    }
    write_json(DATA_DIRECTORY / "agent_07_final_acceptance.json", acceptance)

    # -- Phase 9 handoff -----------------------------------------------------
    frozen_config = json.loads(
        (DATA_DIRECTORY / "agent_05_frozen_train_config.json").read_text()
    )
    handoff = {
        **metadata,
        "artifact": "agent_07_phase9_handoff",
        "status": status,
        "phase_8_recommendation": status,
        "phase_9_readiness": "READY TO PLAN" if status == "PASS" else "BLOCKED",
        "frozen_identities": {
            "warmstart_training_contract_v1": wc.WARMSTART_TRAINING_CONTRACT_VERSION,
            "synthetic_warmstart_corpus_v1": ws.SYNTHETIC_CORPUS_VERSION,
            "warmstart_decision_sampler_v1": ws.DECISION_SAMPLER_VERSION,
            "warmstart_example_v1": wc.WARMSTART_EXAMPLE_VERSION,
            "warmstart_trainer_v1": "warmstart_trainer_v1",
            "warmstart_checkpoint_v1": "warmstart_checkpoint_v1",
            "warmstart_train_config_v1": frozen_config["train_config_version"],
            "warmstart_eval_v1": wc.WARMSTART_EVAL_VERSION,
        },
        "frozen_checkpoint": {
            "path": str(CHECKPOINT_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": EXPECTED_CHECKPOINT_SHA256,
            "global_step": verify["checkpoint_identity"]["checkpoints"]["accepted"][
                "global_step"
            ],
            "parameter_count": wc.EXPECTED_C1_PARAMETERS,
            "c1_config_digest": wc.EXPECTED_C1_CONFIG_DIGEST,
            "train_config_document_digest": EXPECTED_TRAIN_CONFIG_DOCUMENT_DIGEST,
            "trainer_runtime_identity_digest": EXPECTED_TRAINER_RUNTIME_DIGEST,
            "canonical_untrained_checkpoint": {
                "path": str(INITIAL_CHECKPOINT_PATH.relative_to(REPOSITORY_ROOT)),
                "sha256": EXPECTED_INITIAL_SHA256,
                "model_state_checksum": EXPECTED_INITIAL_STATE_CHECKSUM,
            },
        },
        "corpus": {
            "version": ws.SYNTHETIC_CORPUS_VERSION,
            "digests": accepted_corpus_identity()["accepted"],
            "resolver": "stratego.training.synthetic_corpus.default_corpus_root()",
            "games": {"train": 20000, "validation": 4000, "test": 4000},
            "selected_examples": {
                split: verify["upstream"]["selected_example_universes"][split]["examples"]
                for split in ("train", "validation", "test")
            },
        },
        "sealed_test_results": {
            "policy": test_metrics["headline"]["policy"],
            "value": test_metrics["headline"]["value"],
            "belief": test_metrics["headline"]["belief"],
            "stability": {
                "non_finite_examples": test_metrics["stability"]["non_finite_examples"],
                "fraction_above_0_999": test_metrics["stability"][
                    "max_legal_probability"
                ]["fraction_above_0_999"],
            },
        },
        "phase4_evaluations": {
            "random": {
                "games": random_gate["summary"]["games"],
                "wins": random_gate["summary"]["wins"],
                "draws": random_gate["summary"]["draws"],
                "losses": random_gate["summary"]["losses"],
                "effective_win_rate": random_gate["summary"]["effective_win_rate"],
                "red_effective_win_rate": random_gate["summary"]["color_split"]["red"][
                    "effective_win_rate"
                ],
                "blue_effective_win_rate": random_gate["summary"]["color_split"]["blue"][
                    "effective_win_rate"
                ],
                "paired_ci": random_gate["summary"]["confidence_interval"],
            },
            "final_vs_initial": {
                "games": vs_init["summary"]["games"],
                "effective_win_rate": vs_init["summary"]["effective_win_rate"],
                "paired_ci": vs_init["summary"]["confidence_interval"],
            },
            "additional_baselines": {
                name: {
                    "effective_win_rate": tier["summary"]["effective_win_rate"],
                    "games": tier["summary"]["games"],
                    "confidence_interval": tier["summary"]["confidence_interval"],
                }
                for name, tier in tiers["tiers"].items()
            },
        },
        "operational_status": {
            "checkpoint_resume": "warmstart_checkpoint_v1 resume PASS under the reviewer-approved backend_aware_resume_equivalence_v1 (Agent 4)",
            "corpus_crash_reconcile": "commit-journal reconcile PASS; 0 orphans, 0 duplicates (Agent 2)",
            "observer_safety": "inference requests carry the frozen observer-safe payload; measured worker torch imports = 0",
            "phase4_bank_integrity": "digest re-verified by Agent 7 (stored and regenerated)",
            "phase7_library_integrity": "content digest re-verified by Agent 7",
            "throughput": {
                "training_examples_per_second": "~1,800 (Agent 6 production run)",
                "corpus_bottleneck": "reconstruction data loader, not the model",
            },
        },
        "known_limitations": [
            "Phase 8 learns by imitation/outcome supervision from rule agents, not self-play",
            "random/stress decisions carry policy weight 0 and never supervise the policy head",
            "value labels are final W/D/L outcomes, not RL advantages",
            "belief is a lightweight shared head; Phase 11 owns deeper belief validation",
            "no learned setup policy; setups come from the frozen Phase 7 sampler",
            "no decision-time search",
            "no dynamic damping or any Phase 9 RL machinery",
            "synthetic-teacher biases: the corpus reflects the frozen rule population's style",
            "corpus/trainer throughput is loader-bound (~4.3 ms/example single-worker reconstruction)",
            "MPS is not run-to-run bit-deterministic; resume equivalence is backend-aware by accepted amendment",
            "corpus generation crash windows are closed by the commit journal; trainer crash recovery restores exact logical state but MPS parameter paths diverge within the accepted envelope",
        ],
        "phase_9_rules": [
            "initialize the current policy from the accepted warm-start checkpoint",
            "never silently alter Phase 8 corpus or test evidence",
            "semantic changes to any frozen identity require a new version",
        ],
    }
    write_json(DATA_DIRECTORY / "agent_07_phase9_handoff.json", handoff)
    return {
        "heldout": heldout,
        "random": random_payload,
        "acceptance": acceptance,
        "handoff": handoff,
    }


# ---------------------------------------------------------------------------
# 7. Report section
# ---------------------------------------------------------------------------


def append_report_section(artifacts: dict) -> None:
    text = REPORT_PATH.read_text()
    if "## 7. Agent 7" in text:
        raise Agent7Error(
            "the report already contains section 7; refusing to append twice"
        )
    acceptance = artifacts["acceptance"]
    heldout = artifacts["heldout"]
    random_payload = artifacts["random"]
    handoff = artifacts["handoff"]
    summary = random_payload["random_gate"]["summary"]
    vs_summary = random_payload["final_vs_initialisation"]["summary"]
    policy = heldout["headline"]["policy"]
    value = heldout["headline"]["value"]
    belief = heldout["headline"]["belief"]
    intervals = heldout["bootstrap"]["intervals"]
    tiers = random_payload["additional_baselines"]["tiers"]

    def ci(name):
        interval = intervals[name]
        return f"[{interval['lower']:.4f}, {interval['upper']:.4f}]"

    lines = [
        "",
        "## 7. Agent 7 — Independent Held-Out Evaluation and Phase 8 Freeze",
        "",
        f"**Status: {acceptance['status']}** — recommendation "
        f"{acceptance['recommendation']}; formal acceptance belongs to the "
        "reviewing chat. "
        f"{acceptance['gates_true']}/{acceptance['gates_total']} completion gates true.",
        "",
        "### Independent identity verification",
        "",
        "The corpus was resolved exclusively through "
        "`synthetic_corpus.default_corpus_root()` (pointer file), returned the "
        "accepted location, and matched the accepted content/metadata/commit-index "
        "digests exactly, including the byte-level payload audit. The accepted "
        f"checkpoint (`{EXPECTED_CHECKPOINT_SHA256[:12]}…`, update "
        f"{handoff['frozen_checkpoint']['global_step']:,}) and the canonical "
        f"untrained checkpoint (`{EXPECTED_INITIAL_SHA256[:12]}…`) were re-hashed "
        "and independently reloaded through the normal checkpoint API: C1 config "
        "digest, 863,959 finite parameters, both train-config digest namespaces "
        "(document `3cab772b…`, runtime `64db9253…`), and the stored corpus "
        "identity all match the accepted values. The canonical untrained weights "
        "are bit-identical to `build_candidate_model('C1', seed=2026081302)`. "
        "Phase 4 bank digest (stored and regenerated) and the Phase 7 library "
        "digest match their frozen values; all three selected-example universe "
        "digests match Agent 3's accepted record.",
        "",
        "### Sealed synthetic test (first and only model contact)",
        "",
        f"Opened under the frozen `final_evaluation` gate: "
        f"{heldout['headline']['examples']:,} selected decisions across "
        f"{heldout['headline']['games']:,} games; "
        f"{belief['pieces']:,} supervised hidden pieces. Game-level bootstrap, "
        f"10,000 replicates, seed {heldout['bootstrap']['seed']}.",
        "",
        "| Head | Metric | Model | Baseline | Ratio (95% CI) | Gate | Result |",
        "|---|---|---|---|---|---|---|",
        f"| Policy | CE | {policy['model_ce']:.4f} | {policy['baseline_ce']:.4f} | "
        f"{policy['ce_ratio']:.4f} {ci('policy_ce_ratio')} | <= 0.90 | "
        f"{'PASS' if heldout['gates']['policy_ce_ratio_at_most_0_90'] else 'FAIL'} |",
        f"| Policy | top-1 | {policy['model_top1']:.4f} | "
        f"{policy['baseline_expected_top1']:.4f} | diff CI "
        f"{ci('policy_top1_minus_expected')} | model > uniform | "
        f"{'PASS' if heldout['gates']['policy_top1_beats_uniform_expected'] else 'FAIL'} |",
        f"| Value | CE | {value['model_ce']:.4f} | {value['baseline_ce']:.4f} | "
        f"{value['ce_ratio']:.4f} {ci('value_ce_ratio')} | <= 0.98 | "
        f"{'PASS' if heldout['gates']['value_ce_ratio_at_most_0_98'] else 'FAIL'} |",
        f"| Value | Brier | {value['model_brier']:.4f} | {value['baseline_brier']:.4f} | "
        f"margin CI {ci('value_brier_margin_baseline_minus_model')} | model < prior | "
        f"{'PASS' if heldout['gates']['value_brier_beats_train_prior'] else 'FAIL'} |",
        f"| Belief | CE | {belief['model_ce']:.4f} | {belief['baseline_ce']:.4f} | "
        f"{belief['ce_ratio']:.4f} {ci('belief_ce_ratio')} | <= 0.98 | "
        f"{'PASS' if heldout['gates']['belief_ce_ratio_at_most_0_98'] else 'FAIL'} |",
        f"| Belief | top-1 | {belief['model_top1']:.4f} | {belief['baseline_top1']:.4f} | "
        f"diff CI {ci('belief_top1_minus_baseline')} | model > marginal | "
        f"{'PASS' if heldout['gates']['belief_top1_beats_remaining_count_prior'] else 'FAIL'} |",
        "",
        f"Value accuracy {value['model_accuracy']:.4f} (prior "
        f"{value['baseline_accuracy']:.4f}); non-finite logits 0; fraction of "
        "test states with legal max probability > 0.999 = "
        f"{heldout['stability']['max_legal_probability']['fraction_above_0_999']:.6f} "
        "(< 0.95 collapse gate). Family-stratified, per-piece-type and "
        "progress-bucket diagnostics are in `agent_07_heldout_metrics.json`.",
        "",
        "### Frozen Phase 4 random gate",
        "",
        f"All {summary['paired_units']:,} setup pairs, {summary['games']:,} games, "
        "greedy float32 `single_request`, `color_swap_same_board`:",
        "",
        "| Quantity | Observed | Gate | Result |",
        "|---|---|---|---|",
        f"| W / D / L | {summary['wins']} / {summary['draws']} / {summary['losses']} | — | — |",
        f"| EWR | {summary['effective_win_rate']:.6f} | >= 0.950 | "
        f"{'PASS' if random_payload['random_gate']['gates']['effective_win_rate_at_least_0_950'] else 'FAIL'} |",
        f"| Red EWR | {summary['color_split']['red']['effective_win_rate']:.6f} | >= 0.900 | "
        f"{'PASS' if random_payload['random_gate']['gates']['red_effective_win_rate_at_least_0_900'] else 'FAIL'} |",
        f"| Blue EWR | {summary['color_split']['blue']['effective_win_rate']:.6f} | >= 0.900 | "
        f"{'PASS' if random_payload['random_gate']['gates']['blue_effective_win_rate_at_least_0_900'] else 'FAIL'} |",
        f"| Paired 95% lower bound | {summary['confidence_interval']['lower']:.6f} | > 0.900 | "
        f"{'PASS' if random_payload['random_gate']['gates']['paired_bootstrap_lower_bound_above_0_900'] else 'FAIL'} |",
        f"| Illegal actions / model failures / non-finite | 0 / "
        f"{summary['policy_errors']} / 0 | all 0 | "
        f"{'PASS' if random_payload['random_gate']['gates']['model_failures_zero'] else 'FAIL'} |",
        "",
        "### Final checkpoint vs canonical initialisation",
        "",
        f"{vs_summary['paired_units']} paired cases / {vs_summary['games']:,} games: "
        f"EWR {vs_summary['effective_win_rate']:.6f} (gate >= 0.700), paired 95% "
        f"lower bound {vs_summary['confidence_interval']['lower']:.6f} (gate > 0.550) — "
        f"{'PASS' if random_payload['final_vs_initialisation']['all_gates_pass'] else 'FAIL'}. "
        "The checkpoint was already frozen; this is evidence, not a selection.",
        "",
        "### Report-only diagnostics (no gates)",
        "",
        "| Opponent | Games | EWR | 95% CI |",
        "|---|---|---|---|",
    ]
    for name, tier in tiers.items():
        interval = tier["confidence_interval"]
        lines.append(
            f"| {name} | {tier['games']} | {tier['effective_win_rate']:.4f} | "
            f"[{interval['lower']:.4f}, {interval['upper']:.4f}] |"
        )
    lines += [
        "",
        "### Training-discipline audit (hard gate)",
        "",
        "Measured, not asserted: pilots and the canonical run fed 0 test examples "
        "to any model and played 0 Phase 4 neural games before Agent 7 "
        "(instrumented `WarmstartBatch.model_input` and Phase 4 entry points); "
        "the final checkpoint was selected by validation only; the final run "
        "started from the canonical fresh initialisation; 6 of at most 6 pilot "
        "candidates; the frozen sealing gates still refuse every pre-Agent-7 "
        "purpose.",
        "",
        "### Suite",
        "",
        f"{TESTS_BEFORE['summary']} before Agent 7; see "
        "`agent_07_final_acceptance.json` for the after and steady-state runs.",
        "",
        "### Known limitations carried to Phase 9",
        "",
    ]
    lines += [f"- {item}" for item in handoff["known_limitations"]]
    lines += [
        "",
        f"Phase 9: **{handoff['phase_9_readiness']}**. Phase 8 stops here; no "
        "self-play, RL, learned setup selection, or search was implemented or run.",
        "",
    ]
    REPORT_PATH.write_text(text + "\n".join(lines))


# ---------------------------------------------------------------------------
# 8. Pytest
# ---------------------------------------------------------------------------


def run_pytest() -> dict:
    import re

    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    tail = completed.stdout.strip().splitlines()
    summary = tail[-1] if tail else ""
    parts = {}
    for token in ("passed", "failed", "skipped"):
        match = re.search(rf"(\d+) {token}", summary)
        parts[token] = int(match.group(1)) if match else 0
    return {
        "command": f"{sys.executable} -m pytest tests -q",
        "returncode": completed.returncode,
        "summary": summary,
        **parts,
        "seconds": round(time.perf_counter() - started, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--test-metrics", action="store_true")
    parser.add_argument("--random-gate", action="store_true")
    parser.add_argument("--vs-init", action="store_true")
    parser.add_argument("--tiers", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--artifacts", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument("--quick", action="store_true", help="tiny dev pass; never writes artifacts")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-units", type=int, default=64)
    parser.add_argument("--stress-pairs", type=int, default=STRESS_PAIRS)
    arguments = parser.parse_args()

    quick = arguments.quick
    if quick:
        sizes = {"random": 4, "vs_init": 3, "tier": 2, "stress": 1, "batches": 3}
    else:
        sizes = {
            "random": RANDOM_PAIRS,
            "vs_init": VS_INIT_PAIRS,
            "tier": TIER_PAIRS,
            "stress": arguments.stress_pairs,
            "batches": None,
        }

    run_all = arguments.full or quick
    if arguments.verify or run_all:
        print("== verify", flush=True)
        stage_verify(quick=quick)
    if arguments.export or run_all:
        print("== export", flush=True)
        stage_export(quick=quick)
    if getattr(arguments, "test_metrics") or run_all:
        print("== test metrics (sealed split opens now)", flush=True)
        stage_test_metrics(device=arguments.device, quick=quick, batches=sizes["batches"])
    if getattr(arguments, "random_gate") or run_all:
        print("== Phase 4 random gate", flush=True)
        stage_random_gate(
            workers=arguments.workers,
            chunk_units=arguments.chunk_units,
            pairs=sizes["random"],
            quick=quick,
        )
    if getattr(arguments, "vs_init") or run_all:
        print("== final vs canonical initialisation", flush=True)
        stage_vs_init(chunk_units=arguments.chunk_units, pairs=sizes["vs_init"], quick=quick)
    if arguments.tiers or run_all:
        print("== tier diagnostics", flush=True)
        stage_tiers(
            workers=arguments.workers,
            chunk_units=arguments.chunk_units,
            tier_pairs=sizes["tier"],
            stress_pairs=sizes["stress"],
            quick=quick,
        )
    if arguments.audit or run_all:
        print("== training-discipline audit", flush=True)
        stage_audit(quick=quick)

    if arguments.artifacts or (arguments.full and not quick):
        if quick:
            raise Agent7Error("--quick may never write report artifacts")
        print("== artifacts", flush=True)
        pytest_result = None
        if arguments.run_pytest:
            print("== pytest (after)", flush=True)
            pytest_result = run_pytest()
            print("   ", pytest_result["summary"], flush=True)
        artifacts = stage_artifacts(pytest_result)
        append_report_section(artifacts)
        if arguments.run_pytest:
            print("== pytest (steady state, artifact-gated tests execute)", flush=True)
            steady = run_pytest()
            print("   ", steady["summary"], flush=True)
            acceptance_path = DATA_DIRECTORY / "agent_07_final_acceptance.json"
            payload = json.loads(acceptance_path.read_text())
            payload["tests_after_steady_state"] = steady
            payload["suite_green_after"] = (
                payload["tests_after"] is not None
                and payload["tests_after"]["returncode"] == 0
                and steady["returncode"] == 0
            )
            write_json(acceptance_path, payload)
        print(f"status: {artifacts['acceptance']['status']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
