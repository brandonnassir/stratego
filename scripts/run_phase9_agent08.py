#!/usr/bin/env python3
"""Phase 9 Agent 8 acceptance harness: independent final acceptance and freeze.

Stages:

```text
verify      Agents 1-7 acceptance, the administrative working-tree freeze,
            every frozen identity from live bytes (contract, example,
            amendment chain 12h -> 15h -> 24h, three train-config documents,
            trainer runtime identity, Phase 8 anchor, Phase 9 frozen
            checkpoint, B041/B040 lineage, banks, corpus, seeds, library),
            and checkpoint finiteness
discipline  training-discipline evidence: fresh Phase 8 start, exactly six
            pilots, validation-only candidate and checkpoint selection, zero
            final-test neural access before Agent 8, no post-selection
            training, and the iteration-30 observer-safety reconciliation
            from durable rollout/journal/checkpoint evidence
final       the one sealed final evaluation: gates A-H, the frozen stress
            schedule, the collapse/observer replay audit, the belief
            retention benchmark, and the league matrix
artifacts   completion gates and the three Agent 8 artifacts
```

Sealing
-------
`stage_final` refuses to run until `verify` and `discipline` are green, and
its first act is `check_test_bank_access("final_evaluation", phase9_agent=8)`
— the frozen sealing rule that makes Agent 8 the first legitimate neural
reader of `phase9_test_bank_v1`. Nothing here trains, tunes, or selects: the
checkpoint under evaluation is `checkpoints/phase9/selfplay_c1_v1.pt` exactly
as Agent 7 froze it, and every threshold is read from the frozen contract.

Worker purity
-------------
`run_neural_schedule` spawns pure-engine game workers via `spawn`, which
re-imports `__main__`. Torch-loading modules therefore never appear at this
script's module scope — the accepted Agent 1/6/7 discipline.

Usage::

    python scripts/run_phase9_agent08.py --stage verify
    python scripts/run_phase9_agent08.py --stage discipline
    python scripts/run_phase9_agent08.py --stage final
    python scripts/run_phase9_agent08.py --stage artifacts
    python scripts/run_phase9_agent08.py --record-final-suite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
import subprocess
import sys
import time
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
    DEFAULT_ROOT_SEED,
    PAIRING_COLOR_SWAP_SAME_BOARD,
    build_paired_schedule,
    schedule_digest,
    schedule_matches,
)
from stratego.evaluation.neural_worker import (  # noqa: E402
    BATCH_POLICY_SINGLE,
    DECISION_MODE_GREEDY,
    InferenceRequest,
    LocalInferenceChannel,
    NEURAL_WORKER_VERSION,
    RemoteNeuralPolicy,
    neural_policy_ref,
    run_neural_schedule,
)
from stratego.evaluation.registry import policy_ref  # noqa: E402
from stratego.evaluation.setup_bank import SetupBank, bank_digest  # noqa: E402
from stratego.evaluation.statistics import (  # noqa: E402
    build_paired_units,
    bootstrap_interval,
    matchup_seed,
    paired_bootstrap_interval,
    summarize_matchup,
)

AGENT = 8
PHASE = 9
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_9_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_9_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase9" / "agent08"

ACCEPTANCE_ARTIFACT = DATA_DIRECTORY / "agent_08_final_acceptance.json"
STRENGTH_ARTIFACT = DATA_DIRECTORY / "agent_08_strength_results.csv"
LEAGUE_ARTIFACT = DATA_DIRECTORY / "agent_08_league_matrix.csv"

PHASE8_CHECKPOINT = REPOSITORY_ROOT / "checkpoints" / "phase8" / "warmstart_c1_v1.pt"
ANCHOR_EXPORT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "agent01" / "anchor_eval.pt"
FROZEN_CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"
AGENT7_WORK = REPOSITORY_ROOT / "checkpoints" / "phase9" / "agent07"
AGENT7_JOURNAL = AGENT7_WORK / "canonical" / "journal.json"
B041_PATH = AGENT7_WORK / "canonical" / "behavior_B041.pt"
B040_PATH = AGENT7_WORK / "canonical" / "behavior_B040.pt"
H040_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "archive" / "canonical" / "H040.pt"

NAMESPACE = "canonical"
CANDIDATE_ID = "P9-C"

#: The frozen Phase 9 checkpoint under evaluation, exactly as Agent 7 froze it.
ACCEPTED_PHASE9_CHECKPOINT_SHA256 = (
    "dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea"
)
ACCEPTED_PHASE9_MODEL_STATE_DIGEST = (
    "f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd"
)
#: The frozen selection: iteration 40, sourced from the post-iteration-40
#: behavior snapshot `behavior_B041.pt` (never B040, which collected it 40).
ACCEPTED_SELECTED_ITERATION = 40
ACCEPTED_SOURCE_SNAPSHOT = "behavior_B041.pt"

#: Accepted upstream digests, pinned by the reviewing chat's acceptances.
ACCEPTED_CONTRACT_DIGEST = (
    "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"
)
ACCEPTED_EXAMPLE_DIGEST = (
    "a6b17a94449ab764d4b5dd054d677096adfa70c52631865499a60a7a3f44af61"
)
ACCEPTED_AMENDMENT_DIGEST = (
    "ee4b05078c676128f78c8e5c31bd10ce4f0841e34a57c4c7c3fca6616e083ac4"
)
ACCEPTED_AMENDMENT_V2_DIGEST = (
    "92ad4f67fb07a14551ef555335b71000d6369cd817dad59c839d793888de9e71"
)
ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_12H = (
    "9284fbc6b0962937450372d5552f690b2262911275ae5b4000f55da764fba1ba"
)
ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_AMENDED = (
    "22ac552da90989dd4f5cb70371c6579f7168d4daefb5dd9b467a241feda379d9"
)
ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_AMENDED_V2 = (
    "f3b1efdb7b7f34a761b1b5de2c16634ae62b2f562a176411bfdb6b0dda741dc6"
)
ACCEPTED_TRAINER_RUNTIME_IDENTITY_DIGEST = (
    "77af4d45dd8b64e7bf87a82499bc6e54e808320cb214e9b6c58545aa6617b036"
)
ACCEPTED_VALIDATION_BANK_DIGEST = (
    "3d28d544f6669129b12c13e4e3738aa36d1a99e4af8f6685bbb032793701ee4a"
)
ACCEPTED_TEST_BANK_DIGEST = (
    "f38e405559fc7c04b0832b1d3a4e3d82cd68ffff29bc1a9af456a3940e1de6a7"
)
ACCEPTED_PHASE7_LIBRARY_DIGEST = (
    "7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777"
)
ACCEPTED_PHASE8_MODEL_STATE_DIGEST = (
    "f2ec4fc24d72ca170341c2a176aec32c7bf7e75d3315bb39d365835a29d9dd8c"
)

#: The frozen ceiling chain: original contract, then two layered operational
#: amendments. All three identities preserved unedited; only wall-clock moved.
ACCEPTED_CEILING_CHAIN = ((12, 43_200), (15, 54_000), (24, 86_400))

#: The candidate's accepted evaluation identity: the exact `neural_policy_ref`
#: Agent 7's selection and freeze-reload passes evaluated this checkpoint
#: under. Reused, not reinvented — the token participates in match identity.
CANDIDATE_EVAL_ID = "canonical_it40"
ANCHOR_CANDIDATE_ID = "c1_warmstart"
GATE_DTYPE = "float32"

RULE_OPPONENT_IDS = (
    "random_legal",
    "basic_heuristic",
    "tactical_rule_based",
    "strategic_rule_based",
)

#: The frozen observer-probe density of the canonical run (probes per game,
#: taken at the first neural-actor plies). Used for the iteration-30
#: reconciliation and re-applied to the final-test games.
OBSERVER_PROBE_PLIES = 2

#: The full suite as measured immediately before any Phase 9 Agent 8 change.
TESTS_BEFORE = {
    "command": ".venv/bin/python -m pytest tests -q",
    "summary": "4582 passed, 3 skipped in 316.87s (0:05:16)",
    "passed": 4582,
    "failed": 0,
    "skipped": 3,
    "seconds": 316.87,
    "measured_at_commit": "87fd903",
}

STAGES = ("verify", "discipline", "final", "artifacts")


class Agent8Error(RuntimeError):
    """A prerequisite, identity, or sealing condition failed."""


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[agent08 {stamp}] {message}", flush=True)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=1, sort_keys=True)
        stream.write("\n")


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def environment_record() -> dict:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "source_revision": git_output("rev-parse", "--short", "HEAD"),
        "working_tree_state": "dirty" if git_output("status", "--porcelain") else "clean",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def stage_path(name: str) -> Path:
    return WORK_DIRECTORY / f"stage_{name}.json"


def write_stage(name: str, payload: dict) -> None:
    write_json(stage_path(name), payload)


def read_stage(name: str) -> dict:
    path = stage_path(name)
    if not path.exists():
        raise Agent8Error(
            f"stage {name!r} has not run; execute --stage {name} first"
        )
    return read_json(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def document_digest(document) -> str:
    """SHA-256 over a document's canonical JSON — the frozen digest convention."""
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _training():
    """Torch-loading modules, imported at function scope only."""
    from stratego.training import phase9_amendment as amendment
    from stratego.training import phase9_amendment_v2 as amendment_v2
    from stratego.training import phase9_behavior as pb
    from stratego.training import phase9_checkpoint as pck
    from stratego.training import phase9_contract as contract
    from stratego.training import phase9_schedule as schedule
    from stratego.training import phase9_seed as seed
    from stratego.training import phase9_storage as storage
    from stratego.training import synthetic_corpus as corpus
    from stratego.training.warmstart_checkpoint import (
        CorpusIdentity,
        verify_corpus_identity,
    )

    return {
        "amendment": amendment,
        "amendment_v2": amendment_v2,
        "pb": pb,
        "pck": pck,
        "contract": contract,
        "schedule": schedule,
        "seed": seed,
        "storage": storage,
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


def rollout_root() -> Path:
    pointer = REPOSITORY_ROOT / "data" / "phase9_rollout_root.txt"
    return Path(pointer.read_text(encoding="utf-8").strip())


# ---------------------------------------------------------------------------
# Working-tree freeze
# ---------------------------------------------------------------------------


def working_tree_freeze() -> dict:
    """The administrative freeze: tracked tree byte-identical to HEAD.

    Agent 7 was accepted subject only to freezing the exact reviewed working
    tree into a stable commit before final-test access. The freeze is
    verified, never assumed: any tracked modification, deletion, or staged
    change breaks it. Untracked files are listed for the record — the frozen
    identity is the tracked tree.
    """
    head = git_output("rev-parse", "HEAD")
    porcelain = git_output("status", "--porcelain")
    tracked_drift = []
    untracked = []
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        if line.startswith("??"):
            untracked.append(line[3:].strip())
        else:
            tracked_drift.append(line.strip())
    agent7_artifacts_in_head = git_output(
        "ls-tree", "-r", "--name-only", "HEAD", "reports/phase_9_data"
    ).splitlines()
    agent7_present = sorted(
        name for name in agent7_artifacts_in_head if "agent_07" in name
    )
    return {
        "head_commit": head,
        "tracked_drift": tracked_drift,
        "tracked_tree_clean": not tracked_drift,
        "untracked_files": untracked,
        "agent7_artifacts_in_head": agent7_present,
        "rule": (
            "Agent 7 accepted subject only to freezing the reviewed working "
            "tree into a stable commit before final-test access; the tracked "
            "tree must be byte-identical to HEAD, and HEAD must carry the "
            "Agent 7 artifacts"
        ),
    }


def require_frozen_tree(problems: list) -> dict:
    freeze = working_tree_freeze()
    if not freeze["tracked_tree_clean"]:
        problems.append(
            f"tracked working tree has drifted from HEAD: {freeze['tracked_drift']}"
        )
    if len(freeze["agent7_artifacts_in_head"]) < 4:
        problems.append(
            "HEAD does not carry the four Agent 7 artifacts; the administrative "
            "freeze commit is incomplete"
        )
    return freeze


# ---------------------------------------------------------------------------
# stage: verify
# ---------------------------------------------------------------------------


def prior_agent_records(problems: list) -> dict:
    """Agents 1-7 acceptance from their published artifacts."""
    acceptance_artifacts = {
        agent: DATA_DIRECTORY / f"agent_{agent:02d}_acceptance.json"
        for agent in (1, 2, 3, 4, 5)
    }
    acceptance_artifacts[6] = DATA_DIRECTORY / "agent_06_pilot_selection.json"
    acceptance_artifacts[7] = DATA_DIRECTORY / "agent_07_canonical_run.json"
    records = {}
    for agent, path in sorted(acceptance_artifacts.items()):
        if not path.exists():
            problems.append(f"agent {agent} acceptance artifact is missing ({path.name})")
            continue
        payload = read_json(path)
        gates = payload.get("completion_gates") or payload.get("gates") or {}
        false_gates = sorted(key for key, value in gates.items() if value is not True)
        records[str(agent)] = {
            "artifact": path.name,
            "status": payload.get("status"),
            "gates_total": len(gates),
            "gates_true": len(gates) - len(false_gates),
            "false_gates": false_gates,
        }
        if payload.get("status") != "PASS":
            problems.append(f"agent {agent} status is {payload.get('status')!r}, not PASS")
        if false_gates:
            problems.append(f"agent {agent} has false completion gates: {false_gates}")
    return records


def checkpoint_identity(modules, path: Path) -> dict:
    """`(file SHA, model-state digest, payload facts)` from live bytes."""
    pb = modules["pb"]
    pck = modules["pck"]
    payload = pck.read_phase9_payload(path)
    model = pck.model_from_payload(payload)
    digest = pb.state_dict_digest(model)
    parameters = sum(p.numel() for p in model.parameters())
    facts = {
        "path": str(path.relative_to(REPOSITORY_ROOT)) if path.is_relative_to(REPOSITORY_ROOT) else str(path),
        "sha256": file_sha256(path),
        "model_state_digest": digest,
        "parameters": int(parameters),
        "behavior_snapshot_identity": payload.get("behavior_snapshot_identity"),
        "rl_iteration": payload.get("rl_iteration"),
        "snapshot_role": payload.get("snapshot_role"),
        "produced_after_iteration": (payload.get("diagnostics") or {}).get(
            "produced_after_iteration"
        ),
        "global_optimizer_step": payload.get("global_optimizer_step"),
        "train_config_digest": payload.get("train_config_digest"),
        "contract_digest": (payload.get("train_config") or {}).get("contract_digest"),
        "candidate_id": (payload.get("train_config") or {}).get("candidate_id"),
        "namespace": (payload.get("train_config") or {}).get("namespace"),
        "sealed_rollout_digest": payload.get("sealed_rollout_digest"),
        "rollout_iteration_identity": payload.get("rollout_iteration_identity"),
        "behavior_checkpoint_sha256": payload.get("behavior_checkpoint_sha256"),
    }
    del model, payload
    return facts


def finiteness_probe(modules, device: str) -> dict:
    """All parameters finite, and one legal forward pass with finite outputs.

    The probe observation is built from a training-split sampler draw — never
    from either sealed bank — so the finiteness check itself opens nothing.
    """
    import numpy as np
    import torch

    from stratego.engine.legal_moves import legal_action_mask, legal_actions
    from stratego.engine.observation import build_observation
    from stratego.engine.state import create_game
    from stratego.setups.contracts import SPLIT_TRAIN
    from stratego.setups.sampler import load_library_index, sample_setup
    from stratego.training.warmstart_contract import CORPUS_RULES, EXPECTED_SETUP_PROFILE

    pck = modules["pck"]
    payload = pck.read_phase9_payload(FROZEN_CHECKPOINT_PATH)
    model = pck.model_from_payload(payload, device=device)
    model.eval()

    non_finite_parameters = 0
    tensor_count = 0
    for _name, tensor in sorted(model.state_dict().items()):
        tensor_count += 1
        if not torch.isfinite(tensor).all():
            non_finite_parameters += 1

    library = load_library_index()
    # A diagnostic-only train-split draw seed. Deliberately NOT a frozen
    # Phase 9 stream domain: the probe informs no selection and touches no
    # sealed bank, so it must not mint a contract seed.
    probe_seed = 2026081801
    red = sample_setup(SPLIT_TRAIN, probe_seed, profile=EXPECTED_SETUP_PROFILE, index=library)
    blue = sample_setup(
        SPLIT_TRAIN, probe_seed + 1, profile=EXPECTED_SETUP_PROFILE, index=library
    )
    state = create_game(
        red.oriented(0), blue.oriented(1), rules=CORPUS_RULES, game_id="agent8_finiteness_probe"
    )
    legal = legal_actions(state)
    observation = np.array(build_observation(state, 0), dtype=np.float32, copy=True)
    batch = torch.from_numpy(observation).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model.forward_observation(batch)
    finite = {
        "policy_logits_finite": bool(torch.isfinite(outputs.policy_logits).all()),
        "value_logits_finite": bool(torch.isfinite(outputs.value_logits).all()),
        "belief_logits_finite": bool(
            torch.isfinite(outputs.belief_logits).all()
            if getattr(outputs, "belief_logits", None) is not None
            else True
        ),
    }
    result = {
        "parameter_tensors": tensor_count,
        "non_finite_parameter_tensors": non_finite_parameters,
        "probe_split": SPLIT_TRAIN,
        "probe_legal_actions": len(legal),
        **finite,
        "all_finite": non_finite_parameters == 0 and all(finite.values()),
    }
    del model, payload, outputs
    return result


def stage_verify(args) -> dict:
    modules = _training()
    contract = modules["contract"]
    amendment = modules["amendment"]
    amendment_v2 = modules["amendment_v2"]
    pb = modules["pb"]

    problems: list[str] = []
    started = time.perf_counter()

    log("verify: prior agents")
    prior = prior_agent_records(problems)

    log("verify: administrative working-tree freeze")
    freeze = require_frozen_tree(problems)

    log("verify: frozen upstream roster")
    upstream = contract.verify_phase9_upstream()
    problems.extend(upstream)

    log("verify: contract, example, and amendment-chain identities")
    observed_contract = contract.contract_digest()
    if observed_contract != ACCEPTED_CONTRACT_DIGEST:
        problems.append(f"contract digest {observed_contract} != accepted")
    from stratego.training.phase9_targets import example_contract_digest

    observed_example = example_contract_digest()
    if observed_example != ACCEPTED_EXAMPLE_DIGEST:
        problems.append(f"example contract digest {observed_example} != accepted")

    chain_problems = amendment_v2.verify_chain_untouched()
    problems.extend(chain_problems)
    observed_v1 = amendment.amendment_digest()
    observed_v2 = amendment_v2.amendment_digest()
    if observed_v1 != ACCEPTED_AMENDMENT_DIGEST:
        problems.append(f"amendment v1 digest {observed_v1} != accepted")
    if observed_v2 != ACCEPTED_AMENDMENT_V2_DIGEST:
        problems.append(f"amendment v2 digest {observed_v2} != accepted")
    ceiling_history = amendment_v2.ceiling_history()
    observed_chain = tuple(
        (entry["hours"], entry["seconds"]) for entry in ceiling_history
    )
    if observed_chain != ACCEPTED_CEILING_CHAIN:
        problems.append(
            f"ceiling chain {observed_chain} != accepted {ACCEPTED_CEILING_CHAIN}"
        )

    log("verify: the three train-config documents and the runtime identity")
    frozen = read_json(DATA_DIRECTORY / "agent_06_frozen_train_config.json")
    digest_12h = document_digest(frozen["config"])
    digest_15h = document_digest(frozen["config_amended"])
    document_v2 = amendment_v2.apply_to_train_config_document(frozen["config_amended"])
    digest_24h = document_digest(document_v2)
    for label, observed, expected in (
        ("12h", digest_12h, ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_12H),
        ("15h", digest_15h, ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_AMENDED),
        ("24h", digest_24h, ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_AMENDED_V2),
    ):
        if observed != expected:
            problems.append(f"train-config document ({label}) digest {observed} != accepted")
    reconciliation_v1 = amendment.reconcile_documents(frozen["config"], frozen["config_amended"])
    reconciliation_v2 = amendment_v2.reconcile_documents(frozen["config_amended"], document_v2)
    if not reconciliation_v1["only_the_wall_clock_ceiling_changed"]:
        problems.append(
            "v1 amendment changed more than the ceiling: "
            f"{reconciliation_v1['changed_fields']}"
        )
    if not reconciliation_v2["only_the_wall_clock_ceiling_changed"]:
        problems.append(
            "v2 amendment changed more than the ceiling: "
            f"{reconciliation_v2['changed_fields']}"
        )

    import torch  # noqa: F401  (device availability check below)

    device = args.device
    runtime_identity_problem = None
    from stratego.training import phase9_trainer as pt

    config = pt.Phase9TrainConfig.for_candidate(
        CANDIDATE_ID,
        namespace=NAMESPACE,
        device=device,
        total_iterations=contract.CANONICAL_ITERATIONS,
        scope="pilot_candidate",
    )
    observed_runtime = config.digest()
    if observed_runtime != ACCEPTED_TRAINER_RUNTIME_IDENTITY_DIGEST:
        runtime_identity_problem = (
            f"trainer runtime identity {observed_runtime} != accepted"
        )
        problems.append(runtime_identity_problem)
    identity_effect = amendment_v2.runtime_identity_is_unaffected(
        frozen["trainer_runtime_identity"], config.identity()
    )
    if not identity_effect["unchanged"]:
        problems.append(
            f"the runtime identity moved: {identity_effect['differing_fields']}"
        )

    log("verify: corpus resolver and identity")
    resolved_root = modules["corpus"].default_corpus_root()
    corpus_report = None
    try:
        corpus_report = modules["verify_corpus_identity"](
            resolved_root,
            accepted_corpus_identity(modules),
            check_payload_bytes=False,
        )
    except Exception as error:  # noqa: BLE001 — a corpus mismatch is BLOCKED
        problems.append(f"corpus verification failed: {type(error).__name__}: {error}")

    log("verify: Phase 8 anchor from live bytes")
    phase8_sha = file_sha256(PHASE8_CHECKPOINT)
    if phase8_sha != contract.EXPECTED_PHASE8_CHECKPOINT_SHA256:
        problems.append(f"Phase 8 checkpoint SHA {phase8_sha} != accepted")
    from stratego.training.warmstart_checkpoint import load_model_for_evaluation

    anchor_model, _anchor_metadata = load_model_for_evaluation(
        PHASE8_CHECKPOINT, device="cpu"
    )
    anchor_state_digest = pb.state_dict_digest(anchor_model)
    anchor_parameters = sum(p.numel() for p in anchor_model.parameters())
    if anchor_state_digest != ACCEPTED_PHASE8_MODEL_STATE_DIGEST:
        problems.append(
            f"Phase 8 anchor model-state digest {anchor_state_digest} != accepted"
        )
    if anchor_parameters != 863_959:
        problems.append(f"anchor parameter count {anchor_parameters} != 863,959")
    del anchor_model

    anchor_export_sha = file_sha256(ANCHOR_EXPORT_PATH)
    agent1 = read_json(DATA_DIRECTORY / "agent_01_acceptance.json")
    if anchor_export_sha != agent1["anchor_export"]["export_sha256"]:
        problems.append(
            f"anchor evaluation export SHA {anchor_export_sha} != Agent 1's record"
        )

    log("verify: Phase 9 frozen checkpoint and B041/B040 lineage")
    frozen_facts = checkpoint_identity(modules, FROZEN_CHECKPOINT_PATH)
    if frozen_facts["sha256"] != ACCEPTED_PHASE9_CHECKPOINT_SHA256:
        problems.append(f"frozen checkpoint SHA {frozen_facts['sha256']} != accepted")
    if frozen_facts["model_state_digest"] != ACCEPTED_PHASE9_MODEL_STATE_DIGEST:
        problems.append(
            f"frozen model-state digest {frozen_facts['model_state_digest']} != accepted"
        )
    if frozen_facts["parameters"] != 863_959:
        problems.append(f"frozen parameter count {frozen_facts['parameters']} != 863,959")
    if frozen_facts["behavior_snapshot_identity"] != "B041":
        problems.append(
            f"frozen payload names snapshot {frozen_facts['behavior_snapshot_identity']!r}, "
            "expected 'B041'"
        )
    if frozen_facts["produced_after_iteration"] != ACCEPTED_SELECTED_ITERATION:
        problems.append(
            f"frozen payload produced_after_iteration = "
            f"{frozen_facts['produced_after_iteration']}, expected 40"
        )
    if frozen_facts["candidate_id"] != CANDIDATE_ID:
        problems.append(f"frozen train config candidate {frozen_facts['candidate_id']!r} != P9-C")
    if frozen_facts["namespace"] != NAMESPACE:
        problems.append(f"frozen train config namespace {frozen_facts['namespace']!r} != canonical")
    if frozen_facts["contract_digest"] != ACCEPTED_CONTRACT_DIGEST:
        problems.append("frozen payload's contract digest != accepted")
    if frozen_facts["train_config_digest"] != ACCEPTED_TRAINER_RUNTIME_IDENTITY_DIGEST:
        problems.append("frozen payload's train-config digest != accepted runtime identity")

    b041_sha = file_sha256(B041_PATH)
    b040_sha = file_sha256(B040_PATH)
    b041_bytes_equal = b041_sha == frozen_facts["sha256"] and (
        B041_PATH.read_bytes() == FROZEN_CHECKPOINT_PATH.read_bytes()
    )
    if not b041_bytes_equal:
        problems.append(
            "behavior_B041.pt is not byte-identical to the frozen checkpoint"
        )
    if b040_sha == frozen_facts["sha256"]:
        problems.append(
            "behavior_B040.pt hashes equal to the frozen checkpoint; the freeze "
            "must source the post-iteration-40 snapshot B041, not B040"
        )
    if frozen_facts["behavior_checkpoint_sha256"] != b040_sha:
        problems.append(
            "the frozen payload's collecting-behavior SHA does not equal B040's "
            "file SHA; the iteration-40 lineage (B040 collected, training "
            "produced B041) does not reconstruct"
        )
    b040_facts = checkpoint_identity(modules, B040_PATH)
    if b040_facts["model_state_digest"] == frozen_facts["model_state_digest"]:
        problems.append(
            "B040's model state equals the frozen checkpoint's; B040 and B041 "
            "must be different weights (negative control failed)"
        )
    h040_facts = checkpoint_identity(modules, H040_PATH)
    if h040_facts["model_state_digest"] != frozen_facts["model_state_digest"]:
        problems.append(
            "archive member H040 (created after iteration 40) does not carry the "
            "frozen checkpoint's model state; the post-iteration-40 lineage fails"
        )

    log("verify: selection recompute from the twelve validation passes")
    manifest = read_json(DATA_DIRECTORY / "agent_07_checkpoint_manifest.json")
    validation_history = manifest["validation_history"]
    scores = {
        int(entry["iteration"]): float(entry["selection_score"])
        for entry in validation_history
    }
    expected_iterations = tuple(range(5, 61, 5))
    if tuple(sorted(scores)) != expected_iterations:
        problems.append(
            f"validation passes cover iterations {sorted(scores)}, expected "
            f"{list(expected_iterations)}"
        )
    best_iteration = max(scores, key=lambda iteration: (scores[iteration], -iteration))
    strictly_highest = sum(
        1 for value in scores.values() if value == scores[best_iteration]
    ) == 1
    if best_iteration != ACCEPTED_SELECTED_ITERATION:
        problems.append(
            f"recomputed best iteration {best_iteration} != frozen selection 40"
        )
    if not strictly_highest:
        problems.append("the selected iteration's validation score is not strictly highest")
    selected_record = next(
        entry for entry in validation_history if int(entry["iteration"]) == 40
    )
    if selected_record["checkpoint_identity"] != ACCEPTED_SOURCE_SNAPSHOT:
        problems.append(
            f"the iteration-40 validation pass evaluated "
            f"{selected_record['checkpoint_identity']!r}, expected 'behavior_B041.pt'"
        )
    if selected_record["checkpoint_sha256"] != ACCEPTED_PHASE9_CHECKPOINT_SHA256:
        problems.append(
            "the iteration-40 validation pass hashed a different checkpoint than "
            "the frozen one"
        )

    log("verify: bank digests from full deterministic rebuilds")
    from stratego.evaluation.phase9_banks import audit_phase9_bank, build_phase9_bank

    validation_bank, validation_manifest = build_phase9_bank("validation")
    observed_validation_digest = bank_digest(validation_bank)
    if observed_validation_digest != ACCEPTED_VALIDATION_BANK_DIGEST:
        problems.append(
            f"validation bank digest {observed_validation_digest} != accepted"
        )
    test_bank, test_manifest = build_phase9_bank("test")
    observed_test_digest = bank_digest(test_bank)
    if observed_test_digest != ACCEPTED_TEST_BANK_DIGEST:
        problems.append(f"test bank digest {observed_test_digest} != accepted")
    test_audit = audit_phase9_bank(
        "test", test_bank, test_manifest, rebuild_sample_every=args.test_bank_rebuild_every
    )
    if not test_audit["all_pass"]:
        failed = sorted(k for k, v in test_audit["checks"].items() if not v)
        problems.append(f"test bank structural audit failed: {failed}")
    library_digest = validation_manifest["library_content_digest"]
    if library_digest != ACCEPTED_PHASE7_LIBRARY_DIGEST:
        problems.append(f"Phase 7 library digest {library_digest} != accepted")
    bank_cache = WORK_DIRECTORY / "test_bank.json"
    write_json(bank_cache, test_bank.to_dict())

    log("verify: frozen seeds")
    seed = modules["seed"]
    seed_expectations = {
        "PHASE9_MASTER_SEED": 2026081601,
        "ROLLOUT_SCHEDULE_SEED": 2026081602,
        "OPPONENT_SCHEDULE_SEED": 2026081603,
        "TRAIN_ORDER_SEED": 2026081604,
        "PILOT_NAMESPACE_SEED": 2026081605,
        "CANONICAL_NAMESPACE_SEED": 2026081606,
        "VALIDATION_BOOTSTRAP_SEED": 2026081607,
        "TEST_BOOTSTRAP_SEED": 2026081608,
    }
    observed_seeds = {}
    for name, expected in seed_expectations.items():
        observed = getattr(seed, name, None)
        observed_seeds[name] = observed
        if observed != expected:
            problems.append(f"seed {name} is {observed}, expected {expected}")

    log("verify: checkpoint finiteness probe")
    finiteness = finiteness_probe(modules, device)
    if not finiteness["all_finite"]:
        problems.append("the frozen checkpoint has non-finite parameters or outputs")

    payload = {
        "stage": "verify",
        **environment_record(),
        "problems": problems,
        "prior_agents": prior,
        "working_tree_freeze": freeze,
        "upstream_problems": upstream,
        "identities": {
            "contract_digest": observed_contract,
            "example_contract_digest": observed_example,
            "amendment_v1_digest": observed_v1,
            "amendment_v2_digest": observed_v2,
            "ceiling_history": ceiling_history,
            "train_config_document_digests": {
                "accepted_12h": digest_12h,
                "amended_15h": digest_15h,
                "amended_24h_executed": digest_24h,
            },
            "train_config_reconciliation_v1": reconciliation_v1,
            "train_config_reconciliation_v2": reconciliation_v2,
            "trainer_runtime_identity_digest": observed_runtime,
            "runtime_identity_amendment_effect": identity_effect,
        },
        "corpus": {
            "resolved_root": str(resolved_root),
            "report": corpus_report if isinstance(corpus_report, dict) else str(corpus_report),
        },
        "phase8_anchor": {
            "checkpoint_sha256": phase8_sha,
            "model_state_digest": anchor_state_digest,
            "parameters": anchor_parameters,
            "evaluation_export_sha256": anchor_export_sha,
        },
        "phase9_checkpoint": frozen_facts,
        "lineage": {
            "b041_sha256": b041_sha,
            "b040_sha256": b040_sha,
            "b041_bytes_identical_to_frozen": b041_bytes_equal,
            "b040_differs_from_frozen": b040_sha != frozen_facts["sha256"],
            "frozen_payload_collecting_behavior_is_b040": (
                frozen_facts["behavior_checkpoint_sha256"] == b040_sha
            ),
            "h040_model_state_digest": h040_facts["model_state_digest"],
            "h040_matches_frozen_model_state": (
                h040_facts["model_state_digest"] == frozen_facts["model_state_digest"]
            ),
            "b040_model_state_digest": b040_facts["model_state_digest"],
        },
        "selection": {
            "scores_by_iteration": {str(k): v for k, v in sorted(scores.items())},
            "recomputed_best_iteration": best_iteration,
            "strictly_highest": strictly_highest,
            "final_iteration_is_best": best_iteration == 60,
            "iteration_40_checkpoint_identity": selected_record["checkpoint_identity"],
            "iteration_40_checkpoint_sha256": selected_record["checkpoint_sha256"],
        },
        "banks": {
            "validation_bank_digest": observed_validation_digest,
            "test_bank_digest": observed_test_digest,
            "test_bank_audit_checks": test_audit["checks"],
            "test_bank_rebuild_sample_every": test_audit["rebuild_sample_every"],
            "phase7_library_digest": library_digest,
        },
        "seeds": observed_seeds,
        "finiteness": finiteness,
        "seconds": time.perf_counter() - started,
    }
    write_stage("verify", payload)
    if problems:
        raise Agent8Error(f"verify stage found {len(problems)} problem(s); see stage_verify.json")
    log(f"verify: PASS in {payload['seconds']:.1f}s")
    return payload


# ---------------------------------------------------------------------------
# stage: discipline
# ---------------------------------------------------------------------------


def probes_for_game(opponent_kind: str, learner_color, final_ply: int) -> int:
    """The exact observer-probe count one committed game contributes.

    The collector probes an acting player's ply only when that actor is
    neural, and stops after `OBSERVER_PROBE_PLIES` probes per game. Current
    and historical games are neural on both sides, so every ply counts; rule
    and stress games are neural only on the learner's side, and red acts on
    plies 1, 3, 5, ... (red always moves first under the frozen rules).
    """
    final_ply = int(final_ply)
    if opponent_kind in ("current_policy", "historical_snapshot"):
        neural_plies = final_ply
    elif learner_color == "red":
        neural_plies = (final_ply + 1) // 2
    elif learner_color == "blue":
        neural_plies = final_ply // 2
    else:
        raise Agent8Error(
            f"asymmetric game with learner_color {learner_color!r}; cannot "
            "reconstruct its probe count"
        )
    return min(OBSERVER_PROBE_PLIES, neural_plies)


def iter_metadata_lines(root: Path, iteration: int):
    directory = root / NAMESPACE / f"iteration_{iteration:03d}" / "metadata"
    for path in sorted(directory.glob("*.meta.jsonl")):
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    yield path.stem.split(".")[0], json.loads(line)


def read_commit_journal(root: Path, iteration: int) -> dict:
    """`file_set -> [commit rows]` for one iteration, from the durable journal."""
    directory = root / NAMESPACE / f"iteration_{iteration:03d}" / "journal"
    journals = {}
    for path in sorted(directory.glob("*.commit.jsonl")):
        rows = []
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        journals[path.name.split(".")[0]] = rows
    return journals


def observer_reconciliation(args) -> dict:
    """Reconcile the iteration-30 observer under-count from durable evidence.

    Four independent legs:

    1. the 27 pre-restart games are byte-valid committed games (digest-checked
       decode of every one, and the sealed digest recomputes from the commit
       journals);
    2. the probe-count rule reconstructs every cleanly-collected iteration's
       recorded probe count exactly, so it is measured, not assumed;
    3. under that verified rule, the exact probe count the lost session
       contributed is computed from durable per-game facts alone;
    4. the probes themselves are re-executed for the 27 games from stored
       bytes, so "no violation was recorded" is upgraded to "no violation
       exists in the committed data".
    """
    from stratego.training.phase9_rollout_store import (
        Phase9RolloutReader,
        sealed_rollout_digest,
    )

    root = rollout_root()
    problems: list[str] = []

    journal = read_json(AGENT7_JOURNAL)
    iterations = {int(row["iteration"]): row for row in journal["iterations"]}

    # Leg 1 — the resumed iteration's durable store.
    it30_journals = read_commit_journal(root, 30)
    w00 = it30_journals.get("w00", [])
    w01 = it30_journals.get("w01", [])
    if len(w00) != 27 or len(w01) != 2021:
        problems.append(
            f"iteration 30 commit journals hold {len(w00)}+{len(w01)} games, "
            "expected 27+2021"
        )
    state = read_json(root / NAMESPACE / "iteration_030" / "state.json")
    manifest = read_json(root / NAMESPACE / "iteration_030" / "manifest.json")
    collecting_entries = [
        entry for entry in state["history"] if entry["state"] == "COLLECTING"
    ]
    if len(collecting_entries) != 2:
        problems.append(
            f"iteration 30 state history has {len(collecting_entries)} COLLECTING "
            "entries, expected 2 (initial + resumed)"
        )
    if state["state"] != "COMMITTED":
        problems.append(f"iteration 30 rollout state is {state['state']!r}, not COMMITTED")
    if manifest["committed_games"] != 2048 or manifest["missing_games"] != 0:
        problems.append("iteration 30 manifest does not verify 2,048/0-missing games")
    if manifest["problems"]:
        problems.append(f"iteration 30 manifest records problems: {manifest['problems']}")
    if len(set(manifest["behavior_snapshot_identities"])) != 1:
        problems.append("iteration 30 mixes behavior snapshot identities")

    reader = Phase9RolloutReader(root, NAMESPACE, 30)
    triples = [
        (row["phase9_game_id"], row["payload_sha256"], row["metadata_sha256"])
        for row in w00 + w01
    ]

    class _Commit:
        __slots__ = ("phase9_game_id", "payload_sha256", "metadata_sha256")

        def __init__(self, triple):
            self.phase9_game_id, self.payload_sha256, self.metadata_sha256 = triple

    recomputed_sealed = sealed_rollout_digest([_Commit(t) for t in triples])
    if recomputed_sealed != state["sealed_rollout_digest"]:
        problems.append(
            "the sealed digest recomputed from the commit journals does not "
            "match the recorded sealed digest"
        )
    checkpoint_binding = iterations[30]["sealed_rollout_digest"]
    if checkpoint_binding != state["sealed_rollout_digest"]:
        problems.append(
            "the training journal bound a different sealed digest for iteration 30"
        )

    w00_ids = [row["phase9_game_id"] for row in w00]
    decode_failures = []
    for game_id in w00_ids:
        try:
            reader.read_game(game_id)  # digest-checked decode
        except Exception as error:  # noqa: BLE001 — a decode failure is a finding
            decode_failures.append(f"{game_id}: {type(error).__name__}: {error}")
    if decode_failures:
        problems.append(
            f"{len(decode_failures)} of the 27 pre-restart games fail digest-checked "
            f"decode: {decode_failures[:3]}"
        )

    # Leg 2 — measure the probe-count rule against every cleanly-collected
    # iteration (single commit journal), and iteration 30's resumed portion.
    log("discipline: validating the probe-count rule against all 60 iterations")
    rule_validation = {"iterations_checked": 0, "iterations_exact": 0, "mismatches": []}
    session_metadata = {}
    for iteration in range(1, 61):
        recorded = iterations[iteration]["collection"]["observer_probes"]
        journals = read_commit_journal(root, iteration)
        by_game_fileset = {}
        for file_set, rows in journals.items():
            for row in rows:
                by_game_fileset[row["phase9_game_id"]] = file_set
        expected_total = 0
        w01_total = 0
        w00_total = 0
        for _fs, record in iter_metadata_lines(root, iteration):
            probes = probes_for_game(
                record["opponent_kind"], record["learner_color"], record["final_ply"]
            )
            expected_total += probes
            file_set = by_game_fileset.get(record["game_id"])
            if file_set == "w01":
                w01_total += probes
            else:
                w00_total += probes
        if iteration == 30:
            session_metadata = {
                "w00_expected_probes": w00_total,
                "w01_expected_probes": w01_total,
            }
            comparison = w01_total  # the resumed session recorded only its own games
        else:
            comparison = expected_total
        rule_validation["iterations_checked"] += 1
        if comparison == recorded:
            rule_validation["iterations_exact"] += 1
        else:
            rule_validation["mismatches"].append(
                {
                    "iteration": iteration,
                    "recorded": recorded,
                    "reconstructed": comparison,
                }
            )
    if rule_validation["mismatches"]:
        problems.append(
            "the probe-count rule fails to reconstruct recorded probe counts: "
            f"{rule_validation['mismatches'][:5]}"
        )

    # Leg 3 — the exact reconciliation under the measured rule.
    recorded_total = sum(
        iterations[iteration]["collection"]["observer_probes"] for iteration in range(1, 61)
    )
    lost_session_probes = session_metadata.get("w00_expected_probes", 0)
    reconstructed_total = recorded_total + lost_session_probes

    # Leg 4 — re-execute the probes for the 27 pre-restart games.
    log("discipline: re-executing observer probes for the 27 pre-restart games")
    replay = replay_observer_probes(reader, w00_ids)
    if replay["failures"]:
        problems.append(
            f"re-executed observer probes found {len(replay['failures'])} unsafe "
            f"probes: {replay['failures'][:3]}"
        )
    if replay["probes_executed"] != lost_session_probes:
        problems.append(
            f"re-executed probe count {replay['probes_executed']} != reconstructed "
            f"{lost_session_probes}; the reconstruction rule failed on the exact "
            "games it exists for"
        )

    # No observer-safety hard stop anywhere in the run's durable journal.
    halt = journal.get("halt")
    harness_faults = journal.get("harness_faults", [])
    observer_fault = [
        fault for fault in harness_faults if "observer" in json.dumps(fault).lower()
    ]
    if halt:
        problems.append(f"the canonical run journal records an uncleared halt: {halt}")
    if observer_fault:
        problems.append(f"the journal records observer-related faults: {observer_fault}")
    recorded_failures = sum(
        int(iterations[iteration]["collection"].get("observer_probe_failures") or 0)
        for iteration in range(1, 61)
    )
    if recorded_failures:
        problems.append(f"{recorded_failures} observer probe failures recorded")

    return {
        "problems": problems,
        "iteration_30": {
            "w00_committed_games": len(w00),
            "w01_committed_games": len(w01),
            "state": state["state"],
            "collecting_entries": len(collecting_entries),
            "behavior_snapshot_identities": manifest["behavior_snapshot_identities"],
            "sealed_digest_recomputed_from_journals": recomputed_sealed,
            "sealed_digest_recorded": state["sealed_rollout_digest"],
            "sealed_digest_bound_by_training": checkpoint_binding,
            "pre_restart_games_decode_clean": not decode_failures,
        },
        "probe_rule": {
            "rule": (
                "probes(game) = min(2, neural-actor plies within final_ply); "
                "current/historical games are neural on every ply, rule/stress "
                "games only on the learner's plies (red acts first)"
            ),
            **rule_validation,
        },
        "reconciliation": {
            "recorded_session_total": recorded_total,
            "lost_session_probes_reconstructed": lost_session_probes,
            "corrected_full_run_total": reconstructed_total,
            "recorded_failures": recorded_failures,
            "reconstruction_is_exact": not rule_validation["mismatches"]
            and replay["probes_executed"] == lost_session_probes,
        },
        "probe_replay": replay,
    }


def replay_observer_probes(reader, game_ids) -> dict:
    """Re-execute the collection-time observer probes from stored bytes."""
    from stratego.engine.legal_moves import legal_actions
    from stratego.engine.observation import build_observation
    from stratego.engine.state import create_game
    from stratego.engine.transition import apply_action
    from stratego.training.phase9_collector import observer_safety_probe
    from stratego.training.warmstart_contract import CORPUS_RULES

    probes_executed = 0
    failures = []
    per_game = {}
    for game_id in game_ids:
        record, metadata = reader.read_game(game_id)
        kind = metadata["opponent_kind"]
        learner_color = metadata["learner_color"]
        neural_players = (
            (0, 1)
            if kind in ("current_policy", "historical_snapshot")
            else ((0,) if learner_color == "red" else (1,))
        )
        state = create_game(
            record.red_setup, record.blue_setup, rules=CORPUS_RULES, game_id=record.game_id
        )
        game_probes = 0
        for decision in record.decisions:
            legal = legal_actions(state)
            actor = int(state.acting_player)
            if actor in neural_players and game_probes < OBSERVER_PROBE_PLIES:
                probe = observer_safety_probe(
                    state, actor, build_observation(state, actor)
                )
                game_probes += 1
                probes_executed += 1
                if not probe["safe"]:
                    failures.append({"game_id": game_id, "ply": int(decision.ply), "probe": probe})
            apply_action(state, decision.selected_action_id, legal=legal)
        per_game[game_id] = game_probes
    return {
        "games_replayed": len(per_game),
        "probes_executed": probes_executed,
        "failures": failures,
        "per_game_min": min(per_game.values()) if per_game else None,
        "per_game_max": max(per_game.values()) if per_game else None,
    }


def stage_discipline(args) -> dict:
    read_stage("verify")
    modules = _training()
    contract = modules["contract"]

    problems: list[str] = []
    started = time.perf_counter()

    log("discipline: fresh Phase 8 start")
    journal = read_json(AGENT7_JOURNAL)
    fresh = journal["fresh_start"]
    if fresh["checkpoint_sha256"] != contract.EXPECTED_PHASE8_CHECKPOINT_SHA256:
        problems.append("the canonical run did not start from the accepted Phase 8 checkpoint")
    if fresh["model_state_digest"] != ACCEPTED_PHASE8_MODEL_STATE_DIGEST:
        problems.append("the canonical run's starting model state is not the anchor's")
    if fresh["global_optimizer_step"] != 0 or fresh.get("pilot_checkpoint_loaded"):
        problems.append("the canonical run did not start fresh (optimizer step or pilot state)")

    log("discipline: exactly six pilot candidates")
    pilot_names = sorted(
        path.name
        for path in rollout_root().iterdir()
        if path.is_dir() and path.name.startswith("pilot_")
    )
    expected_pilots = ["pilot_p9a", "pilot_p9b", "pilot_p9c", "pilot_p9d", "pilot_p9e", "pilot_p9f"]
    if pilot_names != expected_pilots:
        problems.append(f"pilot rollout namespaces {pilot_names} != the frozen six")
    if len(contract.PILOT_CANDIDATES) != 6:
        problems.append("the frozen pilot matrix does not hold exactly six candidates")
    selection6 = read_json(DATA_DIRECTORY / "agent_06_pilot_selection.json")
    if len(selection6["candidates"]) != 6:
        problems.append("Agent 6's artifact does not record exactly six candidates")
    if selection6["selection"]["winner"] != CANDIDATE_ID:
        problems.append(f"Agent 6's winner is {selection6['selection']['winner']!r}, not P9-C")
    if not selection6["selection"].get("unique", False):
        problems.append("Agent 6's winner is not unique")
    forbidden = selection6["selection"].get("forbidden_evidence_used")
    if forbidden not in (None, False, [], {}):
        problems.append(f"Agent 6 selection recorded forbidden evidence: {forbidden}")

    log("discipline: sealed final-test bank never touched before Agent 8")
    agent7 = read_json(DATA_DIRECTORY / "agent_07_canonical_run.json")
    bank_block = agent7["final_test_bank"]
    if bank_block["model_access_by_agent_7"] != 0:
        problems.append("Agent 7 records final-test model access")
    if bank_block.get("constructed_by_agent_7"):
        problems.append("Agent 7 constructed the final-test bank")
    agent1 = read_json(DATA_DIRECTORY / "agent_01_acceptance.json")
    sealing = agent1["sealing_probe"]
    if not sealing.get("test_bank_neural_purposes_refused"):
        problems.append("Agent 1's sealing probe did not demonstrate neural refusal")
    if agent1["completion_gates"].get("test_bank_neural_access_zero") is not True:
        problems.append("Agent 1 did not certify zero test-bank neural access")
    validation_records = journal.get("validations") or {}
    test_bank_games = [
        key
        for key, record in validation_records.items()
        if record.get("bank_version") == contract.TEST_BANK_VERSION
    ]
    if test_bank_games:
        problems.append("the canonical journal records validation passes on the test bank")
    off_bank = [
        key
        for key, record in validation_records.items()
        if record.get("bank_version") != contract.VALIDATION_BANK_VERSION
    ]
    if off_bank:
        problems.append(
            f"validation passes {off_bank} ran on a bank other than the frozen "
            "validation bank"
        )

    log("discipline: no post-selection training")
    frozen_sha_now = file_sha256(FROZEN_CHECKPOINT_PATH)
    b041_sha_now = file_sha256(B041_PATH)
    if frozen_sha_now != ACCEPTED_PHASE9_CHECKPOINT_SHA256:
        problems.append("the frozen checkpoint's bytes moved after the freeze")
    if b041_sha_now != ACCEPTED_PHASE9_CHECKPOINT_SHA256:
        problems.append("B041's bytes moved after the freeze")
    updates_through_40 = sum(
        int(row["updates"]) for row in journal["iterations"] if int(row["iteration"]) <= 40
    )
    total_updates = sum(int(row["updates"]) for row in journal["iterations"])
    manifest = read_json(DATA_DIRECTORY / "agent_07_checkpoint_manifest.json")
    frozen_step = manifest["frozen_phase9_checkpoint"]["payload_validation"][
        "global_optimizer_step"
    ]
    if frozen_step != updates_through_40:
        problems.append(
            f"the frozen payload's optimizer step {frozen_step} != the journal's "
            f"cumulative updates through iteration 40 ({updates_through_40})"
        )
    snapshots = manifest["behavior_snapshots"]
    if len(snapshots) != 60 or "B061" not in snapshots:
        problems.append("the behavior-snapshot ledger does not cover B002..B061")
    if len(set(snapshots.values())) != len(snapshots):
        problems.append("two behavior snapshots share bytes; training did not advance")

    log("discipline: observer-safety reconciliation for the resumed iteration 30")
    observer = observer_reconciliation(args)
    problems.extend(observer["problems"])

    hard_stops = journal["counters"]
    nonzero = {key: value for key, value in hard_stops.items() if value}
    if nonzero:
        problems.append(f"the canonical run recorded hard-stop counters: {nonzero}")

    payload = {
        "stage": "discipline",
        **environment_record(),
        "problems": problems,
        "fresh_start": fresh,
        "pilots": {
            "rollout_namespaces": pilot_names,
            "frozen_candidates": [c["candidate_id"] for c in contract.PILOT_CANDIDATES],
            "winner": selection6["selection"]["winner"],
            "winner_unique": selection6["selection"].get("unique"),
        },
        "final_test_bank_before_agent_8": {
            "agent_7_model_access": bank_block["model_access_by_agent_7"],
            "agent_7_constructed": bank_block.get("constructed_by_agent_7", False),
            "agent_1_neural_purposes_refused": sealing.get("test_bank_neural_purposes_refused"),
        },
        "no_post_selection_training": {
            "frozen_sha256_now": frozen_sha_now,
            "b041_sha256_now": b041_sha_now,
            "frozen_global_step": frozen_step,
            "journal_updates_through_iteration_40": updates_through_40,
            "journal_total_updates": total_updates,
            "behavior_snapshots": len(snapshots),
            "distinct_snapshot_hashes": len(set(snapshots.values())),
        },
        "hard_stop_counters": hard_stops,
        "observer_reconciliation": observer,
        "seconds": time.perf_counter() - started,
    }
    write_stage("discipline", payload)
    if problems:
        raise Agent8Error(
            f"discipline stage found {len(problems)} problem(s); see stage_discipline.json"
        )
    log(f"discipline: PASS in {payload['seconds']:.1f}s")
    return payload


# ---------------------------------------------------------------------------
# stage: final — the one sealed evaluation
# ---------------------------------------------------------------------------


def load_test_bank() -> "SetupBank":
    cache = WORK_DIRECTORY / "test_bank.json"
    if cache.exists():
        bank = SetupBank.from_dict(read_json(cache))
        if bank_digest(bank) == ACCEPTED_TEST_BANK_DIGEST:
            return bank
    from stratego.evaluation.phase9_banks import build_phase9_bank

    bank, _manifest = build_phase9_bank("test")
    observed = bank_digest(bank)
    if observed != ACCEPTED_TEST_BANK_DIGEST:
        raise Agent8Error(f"rebuilt test bank digest {observed} != accepted")
    write_json(cache, bank.to_dict())
    return bank


def candidate_eval_ref():
    return neural_policy_ref(CANDIDATE_EVAL_ID, dtype_name=GATE_DTYPE)


def anchor_eval_ref():
    return neural_policy_ref(ANCHOR_CANDIDATE_ID, dtype_name=GATE_DTYPE)


def candidate_export_path() -> Path:
    return WORK_DIRECTORY / "eval_final_it40.pt"


def export_evaluation_weights(source: Path, export_path: Path) -> str:
    """Export a `phase9_checkpoint_v1` file to the frozen evaluation format."""
    import torch
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from stratego.model.checkpoint import load_checkpoint, save_checkpoint
    from stratego.training.phase9_checkpoint import model_from_payload, read_phase9_payload

    payload = read_phase9_payload(source)
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
        raise Agent8Error(f"evaluation export of {source} changed the weights")
    del model, reloaded, payload
    return file_sha256(export_path)


def _chunks(sequence, size):
    for index in range(0, len(sequence), size):
        yield index // size, sequence[index : index + size]


def games_directory(label: str) -> Path:
    return WORK_DIRECTORY / "final_games" / label


def run_chunked_schedule(matches, bank, owner, *, reference, label, workers, chunk_units):
    """Resumable chunked `run_neural_schedule` execution (accepted shape)."""
    directory = games_directory(label)
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


def nvn_matchup_matches(pairs: int):
    units = build_paired_schedule(
        candidate_eval_ref(),
        anchor_eval_ref(),
        range(pairs),
        setup_bank_version="phase9_test_bank_v1",
    )
    return schedule_matches(units)


def nvn_chunk_path(directory: Path, index: int, chunk) -> Path:
    return directory / f"anchor_{index:04d}_{schedule_digest(chunk)[:16]}.pkl"


def run_nvn_worker(args) -> None:
    """One process's slice of the candidate-vs-anchor final matchup."""
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from stratego.evaluation.neural_worker import InferenceOwner
    from stratego.training import phase9_contract as contract

    directory = games_directory("candidate_vs_anchor")
    directory.mkdir(parents=True, exist_ok=True)
    matches = nvn_matchup_matches(contract.TEST_BANK_CASES)
    chunk_size = max(1, (len(matches) + args.nvn_workers - 1) // args.nvn_workers)
    chunks = list(_chunks(matches, chunk_size))
    index, chunk = chunks[args.nvn_chunk_index]
    path = nvn_chunk_path(directory, index, chunk)
    if path.exists():
        return
    bank = load_test_bank()

    candidate_ref = candidate_eval_ref()
    anchor_ref = anchor_eval_ref()
    owners = {
        candidate_ref.token: InferenceOwner(
            candidate_export_path(),
            decision_mode=DECISION_MODE_GREEDY,
            device=args.device,
            dtype=GATE_DTYPE,
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name="agent8_candidate",
        ),
        anchor_ref.token: InferenceOwner(
            ANCHOR_EXPORT_PATH,
            decision_mode=DECISION_MODE_GREEDY,
            device=args.device,
            dtype=GATE_DTYPE,
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name="agent8_anchor",
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
    log(f"    candidate_vs_anchor chunk {index}: {len(results)} games in {elapsed:.1f}s")


def run_nvn_matchup(args):
    """Fan the candidate-vs-anchor games across worker processes and gather."""
    from stratego.training import phase9_contract as contract

    directory = games_directory("candidate_vs_anchor")
    directory.mkdir(parents=True, exist_ok=True)
    matches = nvn_matchup_matches(contract.TEST_BANK_CASES)
    chunk_size = max(1, (len(matches) + args.nvn_workers - 1) // args.nvn_workers)
    chunks = list(_chunks(matches, chunk_size))
    pending = [
        index
        for index, chunk in chunks
        if not nvn_chunk_path(directory, index, chunk).exists()
    ]
    if pending:
        processes = []
        for index in pending:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--nvn-worker",
                "--nvn-chunk-index",
                str(index),
                "--nvn-workers",
                str(args.nvn_workers),
                "--device",
                args.device,
            ]
            processes.append((index, subprocess.Popen(command, cwd=REPOSITORY_ROOT)))
        failures = []
        for index, process in processes:
            if process.wait() != 0:
                failures.append(index)
        if failures:
            raise Agent8Error(f"candidate-vs-anchor worker chunk(s) {failures} failed")
    results = []
    reports = []
    for index, chunk in chunks:
        path = nvn_chunk_path(directory, index, chunk)
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
        include_setup_table=True,
    ).to_dict()
    summary["results_digest"] = results_digest(ordered)
    summary["matchup"] = matchup
    return summary


def diff_matchup_token(candidate_token: str, anchor_token: str, opponent_token: str) -> str:
    """The frozen paired-difference bootstrap token."""
    return f"diff|{candidate_token}|{anchor_token}|{opponent_token}"


def paired_difference(candidate_results, anchor_results, *, base_seed: int, opponent_token: str):
    """The frozen paired-improvement statistic for gates B and C.

    Per frozen rule: one observation per setup_pair_id — candidate unit score
    minus anchor unit score on the same pair — resampled with the frozen
    method, replicates, confidence, and the seed derived from
    `diff|candidate|anchor|opponent`.
    """
    candidate_units = {
        unit.setup_pair_id: unit.score for unit in build_paired_units(candidate_results)
    }
    anchor_units = {
        unit.setup_pair_id: unit.score for unit in build_paired_units(anchor_results)
    }
    if sorted(candidate_units) != sorted(anchor_units):
        raise Agent8Error(
            "candidate and anchor paired units cover different setup_pair_ids; "
            "the paired difference is undefined"
        )
    differences = [
        candidate_units[pair_id] - anchor_units[pair_id]
        for pair_id in sorted(candidate_units)
    ]
    candidate_token = candidate_results[0].matchup.split(" vs ")[0]
    anchor_token = anchor_results[0].matchup.split(" vs ")[0]
    token = diff_matchup_token(candidate_token, anchor_token, opponent_token)
    interval = bootstrap_interval(
        differences,
        resamples=10_000,
        seed=matchup_seed(base_seed, token),
        confidence=0.95,
        resampling_unit="paired_unit",
    )
    mean = sum(differences) / len(differences)
    return {
        "mean_improvement": mean,
        "units": len(differences),
        "token": token,
        "confidence_interval": interval.to_dict(),
    }


def independent_bootstrap_check(results, base_seed: int) -> dict:
    """`paired_bootstrap_exact`: reproduce one frozen CI with independent code.

    A from-scratch percentile bootstrap (NumPy PCG64, blocked draws exactly as
    frozen) over the gate-A paired-unit scores must equal the machinery's
    interval bit for bit.
    """
    import numpy as np

    units = build_paired_units(results)
    matchup = sorted(results, key=lambda row: row.match_id)[0].matchup
    seed = matchup_seed(base_seed, matchup)
    official = paired_bootstrap_interval(
        units, resamples=10_000, seed=seed, confidence=0.95
    )
    from stratego.evaluation.statistics import _BOOTSTRAP_BLOCK, quantile

    values = np.asarray([unit.score for unit in units], dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(10_000, dtype=np.float64)
    filled = 0
    block_size = _BOOTSTRAP_BLOCK
    while filled < 10_000:
        block = min(block_size, 10_000 - filled)
        indices = generator.integers(0, values.size, size=(block, values.size))
        means[filled : filled + block] = values[indices].mean(axis=1)
        filled += block
    means.sort()
    ordered = means.tolist()
    lower = quantile(ordered, 0.025)
    upper = quantile(ordered, 0.975)
    return {
        "seed": seed,
        "official_lower": official.lower,
        "official_upper": official.upper,
        "independent_lower": lower,
        "independent_upper": upper,
        "exact": lower == official.lower and upper == official.upper,
    }


# --- collapse / observer / reproduction replay ------------------------------


def replay_candidate_decisions(results, owner, *, label: str, probe_observer: bool = True,
                               value_subsample: int = 8):
    """Replay every candidate decision of `results` through the frozen model.

    For each decision: the policy-logit row at the game-time single-request
    shape, the legal softmax (max probability, entropy), the greedy action
    (must reproduce the recorded action), finiteness, and — at the frozen
    collection density — the observer-safety probe. Value diagnostics are
    sampled every `value_subsample`-th decision for the report-only
    calibration table.
    """
    import numpy as np
    import torch

    from stratego.engine.legal_moves import legal_action_mask, legal_actions
    from stratego.engine.observation import build_observation
    from stratego.engine.setup import deserialize_setup
    from stratego.engine.state import create_game
    from stratego.engine.transition import apply_action
    from stratego.model.policy_adapter import prepare_legality, select_action
    from stratego.training.phase9_collector import observer_safety_probe

    channel = LocalInferenceChannel(owner)
    totals = {
        "games": 0,
        "candidate_decisions": 0,
        "action_mismatches": 0,
        "non_finite_rows": 0,
        "above_threshold": 0,
        "observer_probes": 0,
        "observer_failures": 0,
        "entropy_sum": 0.0,
        "entropy_min": None,
        "maxp_histogram": {},
        "value_samples": [],
    }
    mismatch_examples = []
    for result in sorted(results, key=lambda row: row.match_id):
        if result.errored:
            continue
        rules = result.rules_config()
        red = deserialize_setup(result.red_setup)
        blue = deserialize_setup(result.blue_setup)
        candidate_player = int(result.candidate_color)
        state = create_game(red, blue, rules=rules, game_id=result.match_id)
        totals["games"] += 1
        probes_done = 0
        for ply_index, action in enumerate(result.action_history):
            legal = legal_actions(state)
            actor = int(state.acting_player)
            if actor == candidate_player:
                observation = np.array(
                    build_observation(state, actor), dtype=np.float32, copy=True
                )
                mask = np.array(legal_action_mask(state, legal), dtype=np.uint8, copy=True)
                request = InferenceRequest(
                    request_id=f"{result.match_id}#replay{ply_index}",
                    match_id=result.match_id,
                    paired_unit_id=result.paired_unit_id,
                    ply=ply_index,
                    acting_player=actor,
                    decision_seed=0,
                    observation=observation,
                    legal_actions=tuple(int(a) for a in legal),
                    legal_action_mask=mask,
                )
                row = owner.probe_policy_logits([request])[0]
                finite = bool(torch.isfinite(row).all())
                if not finite:
                    totals["non_finite_rows"] += 1
                legality = prepare_legality(request.legal_actions, mask, actor)
                selection = select_action(
                    row, legality, decision_mode=DECISION_MODE_GREEDY, rng=None
                )
                legal_logits = row[list(legality.model)].to(torch.float32)
                probabilities = torch.softmax(legal_logits, dim=0)
                maxp = float(probabilities.max())
                entropy = float(
                    -(probabilities * torch.log(probabilities.clamp_min(1e-45))).sum()
                )
                totals["candidate_decisions"] += 1
                totals["entropy_sum"] += entropy
                totals["entropy_min"] = (
                    entropy
                    if totals["entropy_min"] is None
                    else min(totals["entropy_min"], entropy)
                )
                bucket = min(int(maxp * 20), 19)
                totals["maxp_histogram"][str(bucket)] = (
                    totals["maxp_histogram"].get(str(bucket), 0) + 1
                )
                if maxp > 0.999:
                    totals["above_threshold"] += 1
                if selection.absolute_action_id != int(action):
                    totals["action_mismatches"] += 1
                    if len(mismatch_examples) < 5:
                        mismatch_examples.append(
                            {
                                "match_id": result.match_id,
                                "ply": ply_index,
                                "recorded": int(action),
                                "replayed": int(selection.absolute_action_id),
                            }
                        )
                if probe_observer and probes_done < OBSERVER_PROBE_PLIES:
                    probe = observer_safety_probe(
                        state, actor, build_observation(state, actor)
                    )
                    probes_done += 1
                    totals["observer_probes"] += 1
                    if not probe["safe"]:
                        totals["observer_failures"] += 1
                if (
                    value_subsample
                    and totals["candidate_decisions"] % value_subsample == 0
                ):
                    response = channel.infer(request)
                    diagnostics = getattr(response, "diagnostics", {}) or {}
                    wdl = {
                        key: diagnostics[key]
                        for key in diagnostics
                        if "win" in key or "draw" in key or "loss" in key or "value" in key
                    }
                    totals["value_samples"].append(
                        {
                            "match_id": result.match_id,
                            "ply": ply_index,
                            "candidate_score": result.candidate_score,
                            **{k: v for k, v in wdl.items() if isinstance(v, (int, float))},
                        }
                    )
            apply_action(state, action, legal=legal)
    decisions = totals["candidate_decisions"]
    return {
        "label": label,
        "games": totals["games"],
        "candidate_decisions": decisions,
        "action_mismatches": totals["action_mismatches"],
        "action_mismatch_examples": mismatch_examples,
        "non_finite_policy_rows": totals["non_finite_rows"],
        "fraction_above_0_999": (totals["above_threshold"] / decisions) if decisions else None,
        "decisions_above_0_999": totals["above_threshold"],
        "mean_legal_entropy": (totals["entropy_sum"] / decisions) if decisions else None,
        "min_legal_entropy": totals["entropy_min"],
        "maxp_histogram_20_buckets": totals["maxp_histogram"],
        "observer_probes": totals["observer_probes"],
        "observer_failures": totals["observer_failures"],
        "value_samples": totals["value_samples"],
    }


def league_matrix_rows() -> list:
    """The report-only league matrix, from durable committed training data.

    No new games: every row aggregates the sealed canonical rollouts by
    iteration and opponent identity, giving the current policy's measured
    performance against each league member, rule tier, and stress policy as
    the run progressed. Current-vs-current games are colour-split instead of
    given a learner EWR, since the learner controlled both sides.
    """
    root = rollout_root()
    rows = []
    for iteration in range(1, 61):
        aggregates = {}
        for _fs, record in iter_metadata_lines(root, iteration):
            key = (record["opponent_kind"], record["opponent_identity"])
            slot = aggregates.setdefault(
                key,
                {
                    "games": 0,
                    "learner_wins": 0,
                    "learner_draws": 0,
                    "learner_losses": 0,
                    "red_wins": 0,
                    "blue_wins": 0,
                    "draws": 0,
                },
            )
            slot["games"] += 1
            terminal = record["terminal_result"]
            if terminal == "draw":
                slot["draws"] += 1
                slot["learner_draws"] += 1
            elif terminal == "red_win":
                slot["red_wins"] += 1
            elif terminal == "blue_win":
                slot["blue_wins"] += 1
            learner_color = record["learner_color"]
            if learner_color in ("red", "blue") and terminal != "draw":
                won = (terminal == "red_win") == (learner_color == "red")
                slot["learner_wins" if won else "learner_losses"] += 1
        for (kind, identity), slot in sorted(aggregates.items()):
            decided = slot["learner_wins"] + slot["learner_losses"] + slot["learner_draws"]
            ewr = (
                (slot["learner_wins"] + 0.5 * slot["learner_draws"]) / decided
                if kind != "current_policy" and decided
                else ""
            )
            rows.append(
                {
                    "iteration": iteration,
                    "opponent_kind": kind,
                    "opponent_identity": identity,
                    "games": slot["games"],
                    "learner_wins": slot["learner_wins"] if kind != "current_policy" else "",
                    "learner_draws": slot["learner_draws"] if kind != "current_policy" else "",
                    "learner_losses": slot["learner_losses"] if kind != "current_policy" else "",
                    "learner_ewr": ewr,
                    "red_wins": slot["red_wins"],
                    "blue_wins": slot["blue_wins"],
                    "draws": slot["draws"],
                }
            )
    return rows


def stage_final(args) -> dict:
    verify = read_stage("verify")
    discipline = read_stage("discipline")
    if verify["problems"] or discipline["problems"]:
        raise Agent8Error("verify/discipline recorded problems; the sealed bank stays closed")
    problems: list[str] = []
    freeze = require_frozen_tree(problems)
    if problems:
        raise Agent8Error(f"the working tree drifted before final-test access: {problems}")

    modules = _training()
    contract = modules["contract"]
    seed = modules["seed"]
    started = time.perf_counter()

    # The one legitimate opening of the sealed final-test bank.
    access = contract.check_test_bank_access("final_evaluation", phase9_agent=AGENT)
    log(f"final: sealed bank opened under purpose {access.purpose!r} by agent {access.phase9_agent}")

    frozen_sha = file_sha256(FROZEN_CHECKPOINT_PATH)
    if frozen_sha != ACCEPTED_PHASE9_CHECKPOINT_SHA256:
        raise Agent8Error("the frozen checkpoint moved between verify and final")

    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from stratego.evaluation.neural_worker import InferenceOwner

    export_sha = export_evaluation_weights(FROZEN_CHECKPOINT_PATH, candidate_export_path())
    bank = load_test_bank()
    base_seed = seed.TEST_BOOTSTRAP_SEED
    candidate_ref = candidate_eval_ref()
    anchor_ref = anchor_eval_ref()

    matchups: dict = {}
    safety = {
        "illegal_policy_actions": 0,
        "policy_errors": 0,
        "inference_failures": 0,
        "workers_importing_torch": 0,
        "worker_checkpoint_loads": 0,
    }
    results_by_label: dict = {}

    def absorb(reports, summary):
        for report in reports:
            safety["illegal_policy_actions"] += report.get("illegal_policy_actions", 0)
            safety["inference_failures"] += report.get("inference_failures", 0)
            safety["workers_importing_torch"] = max(
                safety["workers_importing_torch"], report.get("workers_importing_torch", 0)
            )
            safety["worker_checkpoint_loads"] = max(
                safety["worker_checkpoint_loads"], report.get("worker_checkpoint_loads", 0)
            )
        safety["policy_errors"] += summary["policy_errors"]

    # Candidate vs the four rule opponents, then the frozen stress schedule.
    owner = InferenceOwner(
        candidate_export_path(),
        decision_mode=DECISION_MODE_GREEDY,
        device=args.device,
        dtype=GATE_DTYPE,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
        name="agent8_candidate",
    )
    try:
        for opponent_id in RULE_OPPONENT_IDS:
            log(f"final: candidate vs {opponent_id} ({contract.TEST_BANK_CASES} pairs)")
            units = build_paired_schedule(
                candidate_ref,
                policy_ref(opponent_id),
                range(contract.TEST_BANK_CASES),
                setup_bank_version=contract.TEST_BANK_VERSION,
            )
            matches = schedule_matches(units)
            results, reports = run_chunked_schedule(
                matches,
                bank,
                owner,
                reference=candidate_ref,
                label=f"candidate_{opponent_id}",
                workers=args.eval_workers,
                chunk_units=args.chunk_units,
            )
            summary = summarize_results(results, base_seed)
            matchups[f"candidate_vs_{opponent_id}"] = {
                "summary": summary,
                "schedule_digest": schedule_digest(matches),
                "chunks": reports,
            }
            results_by_label[f"candidate_vs_{opponent_id}"] = results
            absorb(reports, summary)

        stress = {}
        for policy_id in contract.STRESS_POLICY_ROSTER:
            log(f"final: candidate vs {policy_id} ({contract.TEST_STRESS_PAIRS} pairs, report-only)")
            units = build_paired_schedule(
                candidate_ref,
                policy_ref(policy_id),
                range(contract.TEST_STRESS_PAIRS),
                setup_bank_version=contract.TEST_BANK_VERSION,
            )
            matches = schedule_matches(units)
            results, reports = run_chunked_schedule(
                matches,
                bank,
                owner,
                reference=candidate_ref,
                label=f"candidate_{policy_id}",
                workers=args.eval_workers,
                chunk_units=args.chunk_units,
            )
            summary = summarize_results(results, base_seed)
            stress[policy_id] = {
                "effective_win_rate": summary["effective_win_rate"],
                "confidence_interval": summary["confidence_interval"],
                "games": summary["games"],
                "policy_errors": summary["policy_errors"],
                "results_digest": summary["results_digest"],
            }
            results_by_label[f"candidate_vs_{policy_id}"] = results
            matchups[f"candidate_vs_{policy_id}"] = {
                "summary": summary,
                "schedule_digest": schedule_digest(matches),
                "chunks": reports,
            }
            absorb(reports, summary)
    finally:
        owner.close()

    # The anchor on the exact same frozen Strategic and Tactical cases.
    anchor_owner = InferenceOwner(
        ANCHOR_EXPORT_PATH,
        decision_mode=DECISION_MODE_GREEDY,
        device=args.device,
        dtype=GATE_DTYPE,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
        name="agent8_anchor",
    )
    try:
        for opponent_id in ("strategic_rule_based", "tactical_rule_based"):
            log(f"final: anchor vs {opponent_id} ({contract.TEST_BANK_CASES} pairs)")
            units = build_paired_schedule(
                anchor_ref,
                policy_ref(opponent_id),
                range(contract.TEST_BANK_CASES),
                setup_bank_version=contract.TEST_BANK_VERSION,
            )
            matches = schedule_matches(units)
            results, reports = run_chunked_schedule(
                matches,
                bank,
                anchor_owner,
                reference=anchor_ref,
                label=f"anchor_{opponent_id}",
                workers=args.eval_workers,
                chunk_units=args.chunk_units,
            )
            summary = summarize_results(results, base_seed)
            matchups[f"anchor_vs_{opponent_id}"] = {
                "summary": summary,
                "schedule_digest": schedule_digest(matches),
                "chunks": reports,
            }
            results_by_label[f"anchor_vs_{opponent_id}"] = results
            absorb(reports, summary)
    finally:
        anchor_owner.close()

    log(f"final: candidate vs Phase 8 anchor ({contract.TEST_BANK_CASES} pairs, neural-vs-neural)")
    nvn_results, nvn_reports, nvn_digest = run_nvn_matchup(args)
    nvn_summary = summarize_results(nvn_results, base_seed)
    matchups["candidate_vs_phase8_anchor"] = {
        "summary": nvn_summary,
        "schedule_digest": nvn_digest,
        "chunks": nvn_reports,
    }
    results_by_label["candidate_vs_phase8_anchor"] = nvn_results
    absorb(nvn_reports, nvn_summary)

    # Gates A-E.
    gate_a_summary = nvn_summary
    gates = {}
    gates["gate_a"] = {
        "ewr": gate_a_summary["effective_win_rate"],
        "ewr_min": 0.58,
        "ci_lower": gate_a_summary["confidence_interval"]["lower"],
        "ci_lower_exclusive": 0.53,
        "games": gate_a_summary["games"],
        "passed": (
            gate_a_summary["effective_win_rate"] >= 0.58
            and gate_a_summary["confidence_interval"]["lower"] > 0.53
        ),
    }
    improvements = {}
    for gate_name, opponent_id in (("gate_b", "strategic_rule_based"), ("gate_c", "tactical_rule_based")):
        candidate_summary = matchups[f"candidate_vs_{opponent_id}"]["summary"]
        improvement = paired_difference(
            results_by_label[f"candidate_vs_{opponent_id}"],
            results_by_label[f"anchor_vs_{opponent_id}"],
            base_seed=base_seed,
            opponent_token=policy_ref(opponent_id).token,
        )
        improvements[opponent_id] = improvement
        gates[gate_name] = {
            "ewr": candidate_summary["effective_win_rate"],
            "ewr_min": 0.52,
            "anchor_ewr": matchups[f"anchor_vs_{opponent_id}"]["summary"]["effective_win_rate"],
            "paired_improvement": improvement["mean_improvement"],
            "paired_improvement_min": 0.05,
            "improvement_ci_lower": improvement["confidence_interval"]["lower"],
            "improvement_ci_lower_exclusive": 0.0,
            "stretch_0_55_report_only": candidate_summary["effective_win_rate"] >= 0.55,
            "passed": (
                candidate_summary["effective_win_rate"] >= 0.52
                and improvement["mean_improvement"] >= 0.05
                and improvement["confidence_interval"]["lower"] > 0.0
            ),
        }
    random_summary = matchups["candidate_vs_random_legal"]["summary"]
    random_colors = random_summary.get("color_split") or {}
    red_ewr = (random_colors.get("red") or {}).get("effective_win_rate")
    blue_ewr = (random_colors.get("blue") or {}).get("effective_win_rate")
    gates["gate_d"] = {
        "ewr": random_summary["effective_win_rate"],
        "overall_min": 0.94,
        "red_ewr": red_ewr,
        "blue_ewr": blue_ewr,
        "color_min": 0.90,
        "ci_lower": random_summary["confidence_interval"]["lower"],
        "ci_lower_exclusive": 0.92,
        "passed": (
            random_summary["effective_win_rate"] >= 0.94
            and red_ewr is not None
            and red_ewr >= 0.90
            and blue_ewr is not None
            and blue_ewr >= 0.90
            and random_summary["confidence_interval"]["lower"] > 0.92
        ),
    }
    basic_summary = matchups["candidate_vs_basic_heuristic"]["summary"]
    gates["gate_e"] = {
        "ewr": basic_summary["effective_win_rate"],
        "ewr_min": 0.65,
        "ci_lower": basic_summary["confidence_interval"]["lower"],
        "ci_lower_exclusive": 0.60,
        "passed": (
            basic_summary["effective_win_rate"] >= 0.65
            and basic_summary["confidence_interval"]["lower"] > 0.60
        ),
    }

    bootstrap_check = independent_bootstrap_check(nvn_results, base_seed)
    if not bootstrap_check["exact"]:
        problems.append("the independent bootstrap reproduction is not exact")

    # Gate G + F replay: every candidate decision across the final-test games.
    log("final: replaying every candidate decision (collapse, reproduction, observer)")
    replay_owner = InferenceOwner(
        candidate_export_path(),
        decision_mode=DECISION_MODE_GREEDY,
        device=args.device,
        dtype=GATE_DTYPE,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
        name="agent8_replay",
    )
    replays = {}
    try:
        for label in sorted(results_by_label):
            if label.startswith("anchor_vs_"):
                continue  # the anchor is not the final candidate
            cache = games_directory(label) / "replay.json"
            if cache.exists():
                replays[label] = read_json(cache)
                continue
            replays[label] = replay_candidate_decisions(
                results_by_label[label], replay_owner, label=label
            )
            write_json(cache, replays[label])
            log(
                f"    replay {label}: {replays[label]['candidate_decisions']} decisions, "
                f"fraction>{0.999}={replays[label]['fraction_above_0_999']:.4f}"
            )
    finally:
        replay_owner.close()

    total_decisions = sum(r["candidate_decisions"] for r in replays.values())
    total_above = sum(r["decisions_above_0_999"] for r in replays.values())
    collapse_fraction = total_above / total_decisions if total_decisions else None
    action_mismatches = sum(r["action_mismatches"] for r in replays.values())
    non_finite_rows = sum(r["non_finite_policy_rows"] for r in replays.values())
    replay_observer_probes = sum(r["observer_probes"] for r in replays.values())
    replay_observer_failures = sum(r["observer_failures"] for r in replays.values())

    gates["gate_g"] = {
        "population": "every final-candidate decision across the final-test games",
        "decisions": total_decisions,
        "decisions_above_0_999": total_above,
        "fraction_above_0_999": collapse_fraction,
        "fraction_max_exclusive": 0.25,
        "passed": collapse_fraction is not None and collapse_fraction < 0.25,
    }
    gates["gate_f"] = {
        "illegal_actions": safety["illegal_policy_actions"],
        "model_failures": safety["inference_failures"] + safety["policy_errors"],
        "non_finite_outputs": non_finite_rows,
        "observer_probes_on_final_games": replay_observer_probes,
        "observer_safety_failures": replay_observer_failures,
        "action_reproduction_mismatches": action_mismatches,
        "passed": (
            safety["illegal_policy_actions"] == 0
            and safety["inference_failures"] == 0
            and safety["policy_errors"] == 0
            and non_finite_rows == 0
            and replay_observer_failures == 0
            and action_mismatches == 0
        ),
    }

    # Gate H: the accepted Phase 8 held-out belief benchmark.
    log("final: belief retention on the sealed Phase 8 test split")
    belief = belief_retention_benchmark(args)
    gates["gate_h"] = {
        "belief_ce_ratio": belief["belief_ce_ratio"],
        "ratio_max": 0.98,
        "belief_top1": belief["belief_top1"],
        "remaining_count_top1": belief["baseline_top1"],
        "passed": (
            belief["belief_ce_ratio"] is not None
            and belief["belief_ce_ratio"] <= 0.98
            and belief["belief_top1"] > belief["baseline_top1"]
        ),
    }

    log("final: league matrix from the sealed canonical rollouts")
    league = league_matrix_rows()

    payload = {
        "stage": "final",
        **environment_record(),
        "problems": problems,
        "authorized_access": {
            "resource": access.resource,
            "purpose": access.purpose,
            "phase9_agent": access.phase9_agent,
        },
        "working_tree_freeze": freeze,
        "protocol": {
            "decision_mode": DECISION_MODE_GREEDY,
            "batch_policy": BATCH_POLICY_SINGLE,
            "dtype": GATE_DTYPE,
            "pairing_mode": PAIRING_COLOR_SWAP_SAME_BOARD,
            "neural_worker_version": NEURAL_WORKER_VERSION,
            "match_root_seed": DEFAULT_ROOT_SEED,
            "candidate_ref": candidate_ref.to_dict(),
            "anchor_ref": anchor_ref.to_dict(),
            "bootstrap_base_seed": base_seed,
            "bank_version": contract.TEST_BANK_VERSION,
            "bank_digest": ACCEPTED_TEST_BANK_DIGEST,
            "candidate_export_sha256": export_sha,
            "anchor_export_sha256": file_sha256(ANCHOR_EXPORT_PATH),
            "frozen_checkpoint_sha256": frozen_sha,
        },
        "matchups": {
            label: {
                "schedule_digest": value["schedule_digest"],
                "results_digest": value["summary"]["results_digest"],
                "games": value["summary"]["games"],
                "wins": value["summary"].get("wins"),
                "draws": value["summary"].get("draws"),
                "losses": value["summary"].get("losses"),
                "effective_win_rate": value["summary"]["effective_win_rate"],
                "confidence_interval": value["summary"]["confidence_interval"],
                "color_split": value["summary"].get("color_split"),
                "terminal_reasons": value["summary"].get("terminal_reasons"),
                "plies": value["summary"].get("plies"),
                "setup_family_stratification": value["summary"].get(
                    "setup_pair_stratification"
                ),
                "policy_errors": value["summary"]["policy_errors"],
            }
            for label, value in matchups.items()
        },
        "gates": gates,
        "improvements": improvements,
        "stress_report_only": stress,
        "safety": safety,
        "paired_bootstrap_exact": bootstrap_check,
        "replays": {
            label: {k: v for k, v in replay.items() if k != "value_samples"}
            for label, replay in replays.items()
        },
        "value_calibration_samples": sum(
            len(replay.get("value_samples", [])) for replay in replays.values()
        ),
        "belief_retention": belief,
        "league_rows": len(league),
        "seconds": time.perf_counter() - started,
    }
    write_json(WORK_DIRECTORY / "league_matrix.json", league)
    replay_samples = [
        sample
        for replay in replays.values()
        for sample in replay.get("value_samples", [])
    ]
    write_json(WORK_DIRECTORY / "value_calibration_samples.json", replay_samples)
    write_stage("final", payload)
    if problems:
        raise Agent8Error(f"final stage found {len(problems)} problem(s); see stage_final.json")
    log(f"final: complete in {payload['seconds']:.1f}s")
    return payload



# ---------------------------------------------------------------------------
# Gate H: the accepted Phase 8 belief benchmark, run on the Phase 9 model
# ---------------------------------------------------------------------------


def _bootstrap_or_undefined(numerators, denominators, *, seed) -> dict:
    from stratego.training.warmstart_baselines import bootstrap_ratio_interval

    if sum(denominators) == 0:
        return {"undefined": True, "reason": "zero denominator over every game"}
    return bootstrap_ratio_interval(numerators, denominators, seed=seed)


def _game_interval(per_game: dict, numerator_field: str, denominator_field: str, *, seed) -> dict:
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


def belief_retention_benchmark(args) -> dict:
    """Gate H: the accepted Phase 8 held-out synthetic belief benchmark.

    Exactly the accepted `warmstart_eval_v1` machinery — same sealed test
    split, same sequential order, same batch size, same accumulate functions,
    same frozen train-fitted value prior, same remaining-count belief
    baseline, same interval seeds — with the Phase 9 frozen checkpoint's
    weights in place of the Phase 8 ones. Nothing is refit on test. The
    Phase 8-style teacher policy imitation CE is computed and reported only.

    Access rides the Phase 8 sealing gate exactly as the Phase 9 contract
    extends it: `check_test_corpus_access('final_evaluation', phase8_agent=7)`
    semantics for the Phase 9 final evaluator (this agent).
    """
    cache = WORK_DIRECTORY / "belief_retention.json"
    if cache.exists():
        stored = read_json(cache)
        if stored.get("model_state_digest") == ACCEPTED_PHASE9_MODEL_STATE_DIGEST:
            return stored

    import torch

    from stratego.training import warmstart_contract as wc
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
    from stratego.training.warmstart_pilot import record_model_input_access
    from stratego.training.warmstart_seed import TEST_BOOTSTRAP_SEED as P8_TEST_SEED
    from stratego.training.phase9_behavior import state_dict_digest
    from stratego.training.phase9_checkpoint import model_from_payload, read_phase9_payload

    access = wc.check_test_corpus_access("final_evaluation", phase8_agent=7)
    started = time.perf_counter()

    payload = read_phase9_payload(FROZEN_CHECKPOINT_PATH)
    model = model_from_payload(payload, device=args.device)
    model.eval()
    observed_digest = state_dict_digest(model)
    if observed_digest != ACCEPTED_PHASE9_MODEL_STATE_DIGEST:
        raise Agent8Error(
            f"the benchmark model's state digest {observed_digest} != the frozen "
            "checkpoint's; refusing to benchmark different weights"
        )

    dataset = WarmstartDataset()
    universe = dataset.universe("test")
    prior = frozen_train_value_prior()
    per_game: dict = {}
    served = 0

    def serve(cursor):
        nonlocal served
        keys, cursor_after = plan_batch(universe, cursor)
        arrays, metadata, _stats = dataset.batch_arrays(keys)
        batch = batch_from_arrays(arrays, metadata)
        outputs = model.forward_observation(
            batch.model_input().to(torch.device(args.device))
        )
        accumulate_batch_statistics(outputs, batch, value_prior=prior, per_game=per_game)
        served += 1
        return cursor_after

    with record_model_input_access() as access_log:
        with torch.no_grad():
            cursor = DataCursor(
                split="test", batch_size=DEFAULT_BATCH_SIZE, order=ORDER_SEQUENTIAL
            )
            while cursor.epoch == 0:
                cursor = serve(cursor)

    headline = summarize_games(
        per_game, split="test", batches=served, seconds=time.perf_counter() - started
    ).to_dict()
    belief = headline["belief"]
    intervals = {
        "belief_ce_ratio": _game_interval(
            per_game, "belief_ce", "belief_baseline_ce", seed=P8_TEST_SEED
        ),
        "belief_top1_minus_baseline": _difference_interval(
            per_game, "belief_top1", "belief_baseline_top1", "belief_pieces", seed=P8_TEST_SEED
        ),
    }
    result = {
        "authorized_access": {
            "resource": "phase8_test_corpus",
            "purpose": access.purpose if hasattr(access, "purpose") else "final_evaluation",
            "rule": (
                "check_test_corpus_access('final_evaluation', phase8_agent=7) "
                "semantics extended to the Phase 9 final evaluator by "
                "phase9_contract.sealing_rules()"
            ),
        },
        "model_state_digest": observed_digest,
        "split": "test",
        "batches": served,
        "games": headline["games"],
        "examples": headline["examples"],
        "belief_pieces": belief["pieces"],
        "belief_ce": belief["model_ce"],
        "belief_baseline_ce": belief["baseline_ce"],
        "belief_ce_ratio": belief["ce_ratio"],
        "belief_top1": belief["model_top1"],
        "baseline_top1": belief["baseline_top1"],
        "intervals": intervals,
        "policy_imitation_report_only": headline["policy"],
        "value_report_only": headline["value"],
        "model_input_accesses": len(access_log) if hasattr(access_log, "__len__") else None,
        "baseline_rule": (
            "the original remaining-count belief baseline and the frozen "
            "train-fitted value prior; nothing refit on test"
        ),
        "seconds": time.perf_counter() - started,
    }
    del model, payload
    write_json(cache, result)
    return result


# ---------------------------------------------------------------------------
# stage: artifacts
# ---------------------------------------------------------------------------

HARD_GATE_ROWS = (
    ("gate_a_vs_phase8_anchor", "gate_a"),
    ("gate_b_strategic", "gate_b"),
    ("gate_c_tactical", "gate_c"),
    ("gate_d_random_guard", "gate_d"),
    ("gate_e_basic_guard", "gate_e"),
    ("gate_f_safety", "gate_f"),
    ("gate_g_policy_collapse", "gate_g"),
    ("gate_h_belief_retention", "gate_h"),
)


def hard_gate_table(final: dict) -> dict:
    """The machine-readable hard-gate table: observed, threshold, boolean."""
    gates = final["gates"]
    table = {}
    table["gate_a_vs_phase8_anchor"] = {
        "observed": {
            "effective_win_rate": gates["gate_a"]["ewr"],
            "paired_bootstrap_lower": gates["gate_a"]["ci_lower"],
        },
        "threshold": {
            "effective_win_rate_min": 0.58,
            "paired_bootstrap_lower_exclusive": 0.53,
        },
        "passed": gates["gate_a"]["passed"],
    }
    for name, key, opponent in (
        ("gate_b_strategic", "gate_b", "strategic_rule_based"),
        ("gate_c_tactical", "gate_c", "tactical_rule_based"),
    ):
        gate = gates[key]
        table[name] = {
            "observed": {
                "effective_win_rate": gate["ewr"],
                "anchor_effective_win_rate": gate["anchor_ewr"],
                "paired_improvement": gate["paired_improvement"],
                "improvement_ci_lower": gate["improvement_ci_lower"],
            },
            "threshold": {
                "effective_win_rate_min": 0.52,
                "paired_improvement_min": 0.05,
                "improvement_ci_lower_exclusive": 0.0,
            },
            "stretch_report_only": {
                "effective_win_rate": 0.55,
                "reached": gate["stretch_0_55_report_only"],
            },
            "passed": gate["passed"],
        }
    table["gate_d_random_guard"] = {
        "observed": {
            "effective_win_rate": gates["gate_d"]["ewr"],
            "red_effective_win_rate": gates["gate_d"]["red_ewr"],
            "blue_effective_win_rate": gates["gate_d"]["blue_ewr"],
            "paired_bootstrap_lower": gates["gate_d"]["ci_lower"],
        },
        "threshold": {
            "overall_ewr_min": 0.94,
            "red_ewr_min": 0.90,
            "blue_ewr_min": 0.90,
            "paired_bootstrap_lower_exclusive": 0.92,
        },
        "passed": gates["gate_d"]["passed"],
    }
    table["gate_e_basic_guard"] = {
        "observed": {
            "effective_win_rate": gates["gate_e"]["ewr"],
            "paired_bootstrap_lower": gates["gate_e"]["ci_lower"],
        },
        "threshold": {
            "ewr_min": 0.65,
            "paired_bootstrap_lower_exclusive": 0.60,
        },
        "passed": gates["gate_e"]["passed"],
    }
    table["gate_f_safety"] = {
        "observed": {
            "illegal_actions": gates["gate_f"]["illegal_actions"],
            "model_failures": gates["gate_f"]["model_failures"],
            "non_finite_outputs": gates["gate_f"]["non_finite_outputs"],
            "observer_safety_failures": gates["gate_f"]["observer_safety_failures"],
        },
        "threshold": {
            "illegal_actions_max": 0,
            "model_failures_max": 0,
            "non_finite_outputs_max": 0,
            "observer_safety_failures_max": 0,
        },
        "passed": gates["gate_f"]["passed"],
    }
    table["gate_g_policy_collapse"] = {
        "observed": {
            "fraction_above_0_999": gates["gate_g"]["fraction_above_0_999"],
            "decisions": gates["gate_g"]["decisions"],
        },
        "threshold": {"fraction_above_0_999_max_exclusive": 0.25},
        "passed": gates["gate_g"]["passed"],
    }
    table["gate_h_belief_retention"] = {
        "observed": {
            "belief_ce_ratio": gates["gate_h"]["belief_ce_ratio"],
            "belief_top1": gates["gate_h"]["belief_top1"],
            "remaining_count_top1": gates["gate_h"]["remaining_count_top1"],
        },
        "threshold": {
            "belief_ce_ratio_max": 0.98,
            "belief_top1_must_beat_remaining_count_top1": True,
        },
        "passed": gates["gate_h"]["passed"],
    }
    return table


def recompute_gate_booleans(table: dict) -> dict:
    """Recompute every hard-gate boolean from its own observed/threshold rows.

    Shared by the artifact writer and the artifact tests, so a table whose
    booleans disagree with its numbers can never be published or survive one.
    """
    recomputed = {}
    a = table["gate_a_vs_phase8_anchor"]
    recomputed["gate_a_vs_phase8_anchor"] = (
        a["observed"]["effective_win_rate"] >= a["threshold"]["effective_win_rate_min"]
        and a["observed"]["paired_bootstrap_lower"]
        > a["threshold"]["paired_bootstrap_lower_exclusive"]
    )
    for name in ("gate_b_strategic", "gate_c_tactical"):
        gate = table[name]
        recomputed[name] = (
            gate["observed"]["effective_win_rate"] >= gate["threshold"]["effective_win_rate_min"]
            and gate["observed"]["paired_improvement"]
            >= gate["threshold"]["paired_improvement_min"]
            and gate["observed"]["improvement_ci_lower"]
            > gate["threshold"]["improvement_ci_lower_exclusive"]
        )
    d = table["gate_d_random_guard"]
    recomputed["gate_d_random_guard"] = (
        d["observed"]["effective_win_rate"] >= d["threshold"]["overall_ewr_min"]
        and d["observed"]["red_effective_win_rate"] >= d["threshold"]["red_ewr_min"]
        and d["observed"]["blue_effective_win_rate"] >= d["threshold"]["blue_ewr_min"]
        and d["observed"]["paired_bootstrap_lower"]
        > d["threshold"]["paired_bootstrap_lower_exclusive"]
    )
    e = table["gate_e_basic_guard"]
    recomputed["gate_e_basic_guard"] = (
        e["observed"]["effective_win_rate"] >= e["threshold"]["ewr_min"]
        and e["observed"]["paired_bootstrap_lower"]
        > e["threshold"]["paired_bootstrap_lower_exclusive"]
    )
    f = table["gate_f_safety"]
    recomputed["gate_f_safety"] = all(
        f["observed"][key.replace("_max", "")] <= f["threshold"][key]
        for key in f["threshold"]
    )
    g = table["gate_g_policy_collapse"]
    recomputed["gate_g_policy_collapse"] = (
        g["observed"]["fraction_above_0_999"] is not None
        and g["observed"]["fraction_above_0_999"]
        < g["threshold"]["fraction_above_0_999_max_exclusive"]
    )
    h = table["gate_h_belief_retention"]
    recomputed["gate_h_belief_retention"] = (
        h["observed"]["belief_ce_ratio"] is not None
        and h["observed"]["belief_ce_ratio"] <= h["threshold"]["belief_ce_ratio_max"]
        and h["observed"]["belief_top1"] > h["observed"]["remaining_count_top1"]
    )
    return recomputed


def validate_acceptance_artifact(payload: dict) -> list:
    """Every way a published acceptance artifact contradicts itself."""
    problems = []
    table = payload.get("hard_gates") or {}
    if sorted(table) != sorted(name for name, _key in HARD_GATE_ROWS):
        problems.append(f"hard-gate table names {sorted(table)} are not the frozen eight")
        return problems
    recomputed = recompute_gate_booleans(table)
    for name, expected in recomputed.items():
        if bool(table[name]["passed"]) is not bool(expected):
            problems.append(
                f"{name}: published boolean {table[name]['passed']} disagrees with "
                f"its own observed/threshold rows ({expected})"
            )
    all_pass = all(recomputed.values())
    recommendation = payload.get("recommendation")
    gates_clean = all(
        value is True for value in (payload.get("completion_gates") or {}).values()
    )
    if recommendation == "PASS" and not (all_pass and gates_clean):
        problems.append("recommendation PASS but a hard or completion gate is false")
    if recommendation == "FAIL" and all_pass and gates_clean:
        problems.append("recommendation FAIL but every hard and completion gate is true")
    return problems


def stage_artifacts(args, run_pytest_result=None) -> dict:
    verify = read_stage("verify")
    discipline = read_stage("discipline")
    final = read_stage("final")
    started = time.perf_counter()

    table = hard_gate_table(final)
    recomputed = recompute_gate_booleans(table)
    hard_gates_pass = all(recomputed.values())

    observer = discipline["observer_reconciliation"]
    completion_gates = {
        "agents1_7_pass": not any(
            record.get("status") != "PASS" for record in verify["prior_agents"].values()
        ),
        "working_tree_frozen_at_stable_commit": verify["working_tree_freeze"][
            "tracked_tree_clean"
        ],
        "corpus_resolver_verified": bool(verify["corpus"]["resolved_root"]),
        "corpus_digests_match": not [
            problem for problem in verify["problems"] if "corpus" in problem
        ],
        "phase8_checkpoint_verified": (
            verify["phase8_anchor"]["checkpoint_sha256"]
            == "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca"
            and verify["phase8_anchor"]["model_state_digest"]
            == ACCEPTED_PHASE8_MODEL_STATE_DIGEST
        ),
        "phase9_checkpoint_verified": (
            verify["phase9_checkpoint"]["sha256"] == ACCEPTED_PHASE9_CHECKPOINT_SHA256
            and verify["phase9_checkpoint"]["model_state_digest"]
            == ACCEPTED_PHASE9_MODEL_STATE_DIGEST
        ),
        "phase9_selected_iteration_40_from_b041": (
            verify["lineage"]["b041_bytes_identical_to_frozen"]
            and verify["lineage"]["b040_differs_from_frozen"]
            and verify["selection"]["recomputed_best_iteration"] == 40
        ),
        "amendment_chain_verified_12h_15h_24h": (
            verify["identities"]["amendment_v1_digest"] == ACCEPTED_AMENDMENT_DIGEST
            and verify["identities"]["amendment_v2_digest"] == ACCEPTED_AMENDMENT_V2_DIGEST
            and verify["identities"]["train_config_reconciliation_v1"][
                "only_the_wall_clock_ceiling_changed"
            ]
            and verify["identities"]["train_config_reconciliation_v2"][
                "only_the_wall_clock_ceiling_changed"
            ]
        ),
        "phase9_config_verified": (
            verify["identities"]["trainer_runtime_identity_digest"]
            == ACCEPTED_TRAINER_RUNTIME_IDENTITY_DIGEST
        ),
        "final_bank_verified": (
            verify["banks"]["test_bank_digest"] == ACCEPTED_TEST_BANK_DIGEST
            and all(verify["banks"]["test_bank_audit_checks"].values())
        ),
        "pre_agent8_final_test_access_zero": (
            discipline["final_test_bank_before_agent_8"]["agent_7_model_access"] == 0
        ),
        "fresh_phase8_start": discipline["fresh_start"]["global_optimizer_step"] == 0,
        "pilot_count_exactly_six": len(discipline["pilots"]["rollout_namespaces"]) == 6,
        "validation_only_selection": (
            discipline["pilots"]["winner"] == "P9-C"
            and verify["selection"]["strictly_highest"]
        ),
        "no_post_selection_training": (
            discipline["no_post_selection_training"]["frozen_sha256_now"]
            == ACCEPTED_PHASE9_CHECKPOINT_SHA256
        ),
        "observer_reconciliation_exact": observer["reconciliation"][
            "reconstruction_is_exact"
        ],
        "phase9_vs_phase8_gate": recomputed["gate_a_vs_phase8_anchor"],
        "strategic_gate": recomputed["gate_b_strategic"],
        "tactical_gate": recomputed["gate_c_tactical"],
        "random_gate": recomputed["gate_d_random_guard"],
        "basic_gate": recomputed["gate_e_basic_guard"],
        "belief_retention_gate": recomputed["gate_h_belief_retention"],
        "collapse_gate": recomputed["gate_g_policy_collapse"],
        "illegal_actions_zero": final["gates"]["gate_f"]["illegal_actions"] == 0,
        "model_failures_zero": final["gates"]["gate_f"]["model_failures"] == 0,
        "nonfinite_outputs_zero": final["gates"]["gate_f"]["non_finite_outputs"] == 0,
        "observer_safety_zero": final["gates"]["gate_f"]["observer_safety_failures"] == 0,
        "paired_bootstrap_exact": final["paired_bootstrap_exact"]["exact"],
        "report_only_diagnostics_written": bool(final["stress_report_only"])
        and final["league_rows"] > 0,
    }

    if run_pytest_result is not None:
        completion_gates["full_suite_green"] = run_pytest_result.get("failed") == 0
    else:
        completion_gates["full_suite_green"] = False  # set by --record-final-suite

    all_completion = all(completion_gates.values())
    recommendation = "PASS" if (hard_gates_pass and all_completion) else "FAIL"
    if verify["problems"] or discipline["problems"] or final["problems"]:
        recommendation = "BLOCKED"

    acceptance = {
        "phase": PHASE,
        "agent": AGENT,
        "artifact": "agent_08_final_acceptance",
        **environment_record(),
        "status": recommendation,
        "recommendation": recommendation,
        "tests_before": TESTS_BEFORE,
        "frozen_inputs": {
            "phase9_checkpoint_sha256": ACCEPTED_PHASE9_CHECKPOINT_SHA256,
            "phase9_model_state_digest": ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
            "selected_iteration": ACCEPTED_SELECTED_ITERATION,
            "source_snapshot": ACCEPTED_SOURCE_SNAPSHOT,
            "phase8_checkpoint_sha256": (
                "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca"
            ),
            "contract_digest": ACCEPTED_CONTRACT_DIGEST,
            "amendment_v1_digest": ACCEPTED_AMENDMENT_DIGEST,
            "amendment_v2_digest": ACCEPTED_AMENDMENT_V2_DIGEST,
            "train_config_documents": {
                "accepted_12h": ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_12H,
                "amended_15h": ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_AMENDED,
                "amended_24h_executed": ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_AMENDED_V2,
            },
            "trainer_runtime_identity": ACCEPTED_TRAINER_RUNTIME_IDENTITY_DIGEST,
            "validation_bank_digest": ACCEPTED_VALIDATION_BANK_DIGEST,
            "test_bank_digest": ACCEPTED_TEST_BANK_DIGEST,
            "ceiling_chain_hours": [entry[0] for entry in ACCEPTED_CEILING_CHAIN],
        },
        "hard_gates": table,
        "hard_gates_all_pass": hard_gates_pass,
        "completion_gates": completion_gates,
        "gates_total": len(completion_gates),
        "gates_true": sum(1 for value in completion_gates.values() if value),
        "verify_summary": {
            "problems": verify["problems"],
            "selection": verify["selection"],
            "lineage": verify["lineage"],
            "identities": verify["identities"],
            "finiteness": verify["finiteness"],
        },
        "discipline_summary": {
            "problems": discipline["problems"],
            "observer_reconciliation": {
                "probe_rule": observer["probe_rule"],
                "reconciliation": observer["reconciliation"],
                "iteration_30": observer["iteration_30"],
                "probe_replay": {
                    key: value
                    for key, value in observer["probe_replay"].items()
                    if key != "failures"
                }
                | {"failures": len(observer["probe_replay"]["failures"])},
            },
            "no_post_selection_training": discipline["no_post_selection_training"],
        },
        "final_summary": {
            "problems": final["problems"],
            "authorized_access": final["authorized_access"],
            "protocol": final["protocol"],
            "matchups": final["matchups"],
            "stress_report_only": final["stress_report_only"],
            "safety": final["safety"],
            "paired_bootstrap_exact": final["paired_bootstrap_exact"],
            "replays": final["replays"],
            "belief_retention": {
                key: value
                for key, value in final["belief_retention"].items()
                if key not in ("policy_imitation_report_only", "value_report_only")
            },
            "report_only": {
                "policy_imitation": final["belief_retention"][
                    "policy_imitation_report_only"
                ],
                "value_benchmark": final["belief_retention"]["value_report_only"],
            },
        },
    }
    self_check = validate_acceptance_artifact(acceptance)
    if self_check:
        raise Agent8Error(f"the acceptance artifact contradicts itself: {self_check}")
    write_json(ACCEPTANCE_ARTIFACT, acceptance)

    # Strength CSV: one row per final matchup, plus the stress schedule.
    with open(STRENGTH_ARTIFACT, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "matchup",
                "candidate",
                "opponent",
                "games",
                "wins",
                "draws",
                "losses",
                "effective_win_rate",
                "ci_lower",
                "ci_upper",
                "red_ewr",
                "blue_ewr",
                "policy_errors",
                "schedule_digest",
                "results_digest",
            ]
        )
        for label in sorted(final["matchups"]):
            matchup = final["matchups"][label]
            colors = matchup.get("color_split") or {}
            candidate, _, opponent = label.partition("_vs_")
            writer.writerow(
                [
                    label,
                    candidate,
                    opponent,
                    matchup["games"],
                    matchup.get("wins"),
                    matchup.get("draws"),
                    matchup.get("losses"),
                    matchup["effective_win_rate"],
                    matchup["confidence_interval"]["lower"],
                    matchup["confidence_interval"]["upper"],
                    (colors.get("red") or {}).get("effective_win_rate"),
                    (colors.get("blue") or {}).get("effective_win_rate"),
                    matchup["policy_errors"],
                    matchup["schedule_digest"],
                    matchup["results_digest"],
                ]
            )

    # League CSV from the durable canonical rollouts.
    league = read_json(WORK_DIRECTORY / "league_matrix.json")
    with open(LEAGUE_ARTIFACT, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "iteration",
                "opponent_kind",
                "opponent_identity",
                "games",
                "learner_wins",
                "learner_draws",
                "learner_losses",
                "learner_ewr",
                "red_wins",
                "blue_wins",
                "draws",
            ]
        )
        for row in league:
            writer.writerow(
                [
                    row["iteration"],
                    row["opponent_kind"],
                    row["opponent_identity"],
                    row["games"],
                    row["learner_wins"],
                    row["learner_draws"],
                    row["learner_losses"],
                    row["learner_ewr"],
                    row["red_wins"],
                    row["blue_wins"],
                    row["draws"],
                ]
            )

    payload = {
        "stage": "artifacts",
        **environment_record(),
        "recommendation": recommendation,
        "hard_gates_all_pass": hard_gates_pass,
        "completion_gates": completion_gates,
        "artifacts": {
            "acceptance": str(ACCEPTANCE_ARTIFACT.relative_to(REPOSITORY_ROOT)),
            "strength": str(STRENGTH_ARTIFACT.relative_to(REPOSITORY_ROOT)),
            "league": str(LEAGUE_ARTIFACT.relative_to(REPOSITORY_ROOT)),
        },
        "seconds": time.perf_counter() - started,
    }
    write_stage("artifacts", payload)
    log(
        f"artifacts: recommendation {recommendation}; "
        f"{sum(1 for value in completion_gates.values() if value)}/"
        f"{len(completion_gates)} completion gates true"
    )
    return payload


def record_final_suite() -> int:
    """Re-run the full suite with artifacts present and record it in the artifact."""
    if not ACCEPTANCE_ARTIFACT.exists():
        raise Agent8Error("run --stage artifacts before --record-final-suite")
    command = [str(REPOSITORY_ROOT / ".venv" / "bin" / "python"), "-m", "pytest", "tests", "-q"]
    log("record-final-suite: running the complete repository suite")
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=REPOSITORY_ROOT, capture_output=True, text=True
    )
    tail = [line for line in completed.stdout.splitlines() if line.strip()][-1:]
    summary = tail[0] if tail else ""
    import re

    passed = int((re.search(r"(\d+) passed", summary) or [0, 0])[1])
    failed_match = re.search(r"(\d+) failed", summary)
    failed = int(failed_match[1]) if failed_match else 0
    skipped_match = re.search(r"(\d+) skipped", summary)
    skipped = int(skipped_match[1]) if skipped_match else 0
    record = {
        "command": " ".join(command[-4:]),
        "returncode": completed.returncode,
        "summary": summary,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "seconds": round(time.perf_counter() - started, 2),
    }
    acceptance = read_json(ACCEPTANCE_ARTIFACT)
    acceptance["tests_after"] = record
    acceptance["completion_gates"]["full_suite_green"] = (
        completed.returncode == 0 and failed == 0
    )
    acceptance["covers_agent_08_artifact_tests"] = True
    acceptance["gates_true"] = sum(
        1 for value in acceptance["completion_gates"].values() if value
    )
    all_completion = all(acceptance["completion_gates"].values())
    if acceptance["recommendation"] != "BLOCKED":
        acceptance["recommendation"] = (
            "PASS" if (acceptance["hard_gates_all_pass"] and all_completion) else "FAIL"
        )
        acceptance["status"] = acceptance["recommendation"]
    problems = validate_acceptance_artifact(acceptance)
    if problems:
        raise Agent8Error(f"the acceptance artifact contradicts itself: {problems}")
    write_json(ACCEPTANCE_ARTIFACT, acceptance)
    log(f"record-final-suite: {summary} (returncode {completed.returncode})")
    return 0 if completed.returncode == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=("all",) + STAGES)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--chunk-units", type=int, default=64)
    parser.add_argument("--nvn-workers", type=int, default=4)
    parser.add_argument("--nvn-worker", action="store_true")
    parser.add_argument("--nvn-chunk-index", type=int, default=None)
    parser.add_argument("--test-bank-rebuild-every", type=int, default=1)
    parser.add_argument("--record-final-suite", action="store_true")
    args = parser.parse_args()

    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if args.nvn_worker:
        run_nvn_worker(args)
        return 0
    if args.record_final_suite:
        return record_final_suite()

    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "verify":
            stage_verify(args)
        elif stage == "discipline":
            stage_discipline(args)
        elif stage == "final":
            stage_final(args)
        elif stage == "artifacts":
            stage_artifacts(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
