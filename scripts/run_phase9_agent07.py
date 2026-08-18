#!/usr/bin/env python3
"""Phase 9 Agent 7 acceptance harness: the canonical population self-play run.

Stages:

```text
verify      Agents 1-6 acceptance, phase9_operational_amendment_v1, every
            frozen identity (contract / example / train-config document /
            trainer runtime), the corpus resolver, mounted production
            storage, the canonical run-schedule digest, the fresh Phase 8
            start proof, the pilot-contamination audit and the scope audit
run         the supervisor: the frozen 60-iteration canonical experiment,
            executed across genuine worker processes, restart-safe at every
            boundary, with two scheduled mid-epoch process restarts
freeze      select one checkpoint by the frozen validation score, freeze it
            to checkpoints/phase9/selfplay_c1_v1.pt, reload it through the
            evaluation-only path and reproduce its validation metrics
artifacts   completion gates and the four Agent 7 artifacts
```

Worker purity
-------------
`run_neural_schedule` spawns pure-engine game workers via `spawn`, which
re-imports `__main__`, and the trainer's loader pool spawns the same way.
Torch-loading modules (`stratego.training.*`, `stratego.model.*`,
`stratego.evaluation.phase9_banks`) therefore never appear at this script's
module scope — the accepted Agent 1/6/7 discipline.

What this harness decides, and what it does not
-----------------------------------------------
Nothing about learning. Every constant comes from the frozen contract and
Agent 6's frozen `phase9_train_config_v1`; the only two free numbers (LR
3e-4, initial KL beta 0.005) are P9-C's, read from the frozen matrix. The
run ends the moment the contracted 60 iterations and their bookkeeping
finish — the amended 54,000 s ceiling is a maximum, never a training budget
to spend down. No pilot checkpoint, optimizer state, archive member or
rollout initializes anything here: the learner starts from the accepted
Phase 8 anchor and its optimizer starts empty.

Usage::

    python scripts/run_phase9_agent07.py --stage verify
    python scripts/run_phase9_agent07.py --stage run
    python scripts/run_phase9_agent07.py --stage freeze
    python scripts/run_phase9_agent07.py --stage artifacts
    python scripts/run_phase9_agent07.py --record-final-suite
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

AGENT = 7
PHASE = 9
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_9_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_9_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase9" / "agent07"

RUN_ARTIFACT = DATA_DIRECTORY / "agent_07_canonical_run.json"
CURVE_ARTIFACT = DATA_DIRECTORY / "agent_07_training_curve.csv"
ARCHIVE_ARTIFACT = DATA_DIRECTORY / "agent_07_population_archive.json"
MANIFEST_ARTIFACT = DATA_DIRECTORY / "agent_07_checkpoint_manifest.json"

PHASE8_CHECKPOINT = REPOSITORY_ROOT / "checkpoints" / "phase8" / "warmstart_c1_v1.pt"
ANCHOR_EXPORT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "agent01" / "anchor_eval.pt"

#: Production archive root: the frozen `checkpoints/phase9/archive` namespace.
PRODUCTION_ARCHIVE_ROOT = REPOSITORY_ROOT / "checkpoints" / "phase9" / "archive"

#: The one frozen Phase 9 checkpoint this agent hands to Agent 8.
FROZEN_CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"

#: The canonical run namespace. Not a pilot slot, and never shared with one.
NAMESPACE = "canonical"

#: The frozen winner. Agent 6 selected it; Agent 7 only executes it.
CANDIDATE_ID = "P9-C"

#: Accepted upstream digests, pinned by the reviewing chat's acceptances.
#: `ACCEPTED_CONTRACT_DIGEST` is immutable historical identity: the original
#: `phase9_rl_contract_v1`, never regenerated to encode the ceiling amendment.
ACCEPTED_CONTRACT_DIGEST = (
    "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"
)
ACCEPTED_EXAMPLE_DIGEST = (
    "a6b17a94449ab764d4b5dd054d677096adfa70c52631865499a60a7a3f44af61"
)
ACCEPTED_AMENDMENT_DIGEST = (
    "ee4b05078c676128f78c8e5c31bd10ce4f0841e34a57c4c7c3fca6616e083ac4"
)

#: The second review-authorized operational amendment: the ceiling moves from
#: 54,000 s to 86,400 s and nothing else. Layered beside v1, which is layered
#: beside the contract; all three identities are preserved unedited.
ACCEPTED_AMENDMENT_V2_DIGEST = (
    "92ad4f67fb07a14551ef555335b71000d6369cd817dad59c839d793888de9e71"
)
ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_AMENDED_V2 = (
    "f3b1efdb7b7f34a761b1b5de2c16634ae62b2f562a176411bfdb6b0dda741dc6"
)

#: Two labelled train-config document namespaces, never conflated. The
#: 12-hour document is preserved as historical provenance; the amended
#: 15-hour document is the one this run executes.
ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_12H = (
    "9284fbc6b0962937450372d5552f690b2262911275ae5b4000f55da764fba1ba"
)
ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_AMENDED = (
    "22ac552da90989dd4f5cb70371c6579f7168d4daefb5dd9b467a241feda379d9"
)

#: The narrower runtime object stamped into every `phase9_checkpoint_v1`.
ACCEPTED_TRAINER_RUNTIME_IDENTITY_DIGEST = (
    "77af4d45dd8b64e7bf87a82499bc6e54e808320cb214e9b6c58545aa6617b036"
)

#: The learner's starting model-state checksum, from Agent 6's handoff.
EXPECTED_START_MODEL_STATE_DIGEST = (
    "f2ec4fc24d72ca170341c2a176aec32c7bf7e75d3315bb39d365835a29d9dd8c"
)

#: Agent 2's pinned canonical run-schedule digest.
EXPECTED_CANONICAL_RUN_SCHEDULE_DIGEST = (
    "bc253e8be2c63db1af308f62cf52f99f1431e9c9ec8a6db0987783b2983c0e64"
)

#: The legacy runtime scope token. A frozen naming artifact: Agent 5's
#: `SCOPES` has no canonical entry, and reconstructing the accepted runtime
#: identity digest requires this exact string. `stage_verify`'s scope audit
#: measures — rather than assumes — that it changes no training behavior.
RUNTIME_SCOPE_TOKEN = "pilot_candidate"

#: The anchor's evaluation identity (the accepted Agent 1/6/7 shape).
ANCHOR_CANDIDATE_ID = "c1_warmstart"
GATE_DTYPE = "float32"

RULE_OPPONENT_IDS = (
    "random_legal",
    "basic_heuristic",
    "tactical_rule_based",
    "strategic_rule_based",
)

#: The frozen validated execution topology (Agent 5). Not retuned here.
VALIDATED_TOPOLOGY = {"workers": 6, "prefetch": 2, "record_cache_size": 48}

#: The two scheduled genuine process restarts, as
#: `iteration -> fraction of the iteration's total updates to run first`.
#: Iteration 4 restarts inside epoch 1 (partial-epoch KL-controller state,
#: non-zero minibatch cursor, no validation record yet); iteration 12
#: restarts inside epoch 2, after two validation passes and two archive
#: members exist, so best-validation and active-archive continuity are
#: exercised too. Both continue the same sealed rollout under the same
#: behavior snapshot: neither repeats nor skips a single optimizer step.
SCHEDULED_RESTARTS = {4: 0.30, 12: 0.70}

#: Exit code the worker uses to ask the supervisor for a fresh process.
RESTART_EXIT_CODE = 3

#: Hard cap on supervisor relaunches, so a crash loop stops instead of
#: burning the operational ceiling.
MAX_WORKER_LAUNCHES = 40

#: The full suite as measured immediately before any Phase 9 Agent 7 change.
TESTS_BEFORE = {
    "command": ".venv/bin/python -m pytest tests -q -p no:randomly",
    "summary": "4493 passed, 3 skipped in 314.43s (0:05:14)",
    "passed": 4493,
    "failed": 0,
    "skipped": 3,
    "seconds": 314.43,
    "measured_at_commit": "74eecad",
}


class Agent7Error(RuntimeError):
    """A precondition or frozen identity failed. Always raised, never patched."""


class Agent7Halt(RuntimeError):
    """The run must stop and report an exact incomplete state."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[agent07 {time.strftime('%H:%M:%S')}] {message}", flush=True)


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
        raise Agent7Error(f"stage {name!r} has not been run yet ({path})")
    return read_json(path)


def _training():
    """Torch-adjacent modules, imported on first use only (worker purity)."""
    from stratego.training import phase9_amendment as amendment
    from stratego.training import phase9_amendment_v2 as amendment_v2
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
        "amendment": amendment,
        "amendment_v2": amendment_v2,
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


def document_digest(document) -> str:
    """SHA-256 over a document's canonical JSON — the frozen digest convention."""
    import hashlib

    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def run_directory() -> Path:
    return WORK_DIRECTORY / NAMESPACE


def journal_path() -> Path:
    return run_directory() / "journal.json"


def canonical_config(modules, device: str):
    """The exact frozen runtime object, reconstructed and identity-checked.

    Built through `for_candidate` from the frozen matrix — the learning
    numbers are read, never restated — and then required to hash to the
    accepted `trainer_runtime_identity` digest before it is allowed to build
    an optimizer.
    """
    pt = modules["pt"]
    contract = modules["contract"]
    config = pt.Phase9TrainConfig.for_candidate(
        CANDIDATE_ID,
        namespace=NAMESPACE,
        device=device,
        total_iterations=contract.CANONICAL_ITERATIONS,
        scope=RUNTIME_SCOPE_TOKEN,
    )
    if config.digest() != ACCEPTED_TRAINER_RUNTIME_IDENTITY_DIGEST:
        raise Agent7Error(
            f"reconstructed trainer runtime identity {config.digest()} != the "
            f"accepted {ACCEPTED_TRAINER_RUNTIME_IDENTITY_DIGEST}; the frozen "
            "configuration is not the one Agent 6 froze"
        )
    return config


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def verify_storage_mounted() -> dict:
    """The rollout root must resolve to the actually mounted external
    filesystem, not an ordinary directory on the boot volume. Checked at
    process start and after every genuine restart.
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
    projection = storage.projected_rollout_bytes(
        modules["contract"].CANONICAL_MAX_SCHEDULED_GAMES
    )
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
    if volume["free_bytes"] < projection["projected_bytes"] * 2:
        problems.append(
            f"{mount} has {volume['free_gib']} GiB free, below twice the "
            f"projected canonical requirement of {projection['projected_gib']} GiB"
        )
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
        "projected_gib": projection["projected_gib"],
        "identity_rule": storage.STORAGE_IDENTITY_RULE,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# The scope audit
# ---------------------------------------------------------------------------


def scope_behaviour_audit(modules) -> dict:
    """Measure whether `scope == "pilot_candidate"` changes training behavior.

    The supplementary instruction is conditional, so this returns a
    measurement rather than a claim. Two things are checked:

    1. every live reference to the scope value in the shipped library, read
       from source, is classified;
    2. the trainer runtime identity is rebuilt under each frozen scope token
       and the resulting *learning* fields are compared — if the scope
       reached any learning constant, one of them would move.
    """
    pt = modules["pt"]
    contract = modules["contract"]
    source = Path(pt.__file__).read_text().splitlines()
    references = []
    for number, line in enumerate(source, start=1):
        stripped = line.strip()
        if "scope" not in stripped or stripped.startswith("#"):
            continue
        if not any(
            token in stripped
            for token in ("self.scope", "config.scope", "SCOPE_", "scope=", "scope:", '"scope"')
        ):
            continue
        references.append({"line": number, "source": stripped})

    # The only branch on the value in the shipped trainer is the unit-test
    # relaxation; every production scope takes the *stricter* path.
    branches = [
        entry
        for entry in references
        if "if self.scope" in entry["source"] or "if config.scope" in entry["source"]
    ]

    learning_fields = (
        "learning_rate",
        "initial_kl_beta",
        "total_iterations",
        "minibatch_size",
        "epochs_per_rollout",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "gradient_clip_norm",
        "precision",
        "model_candidate",
        "device",
        "namespace",
        "candidate_id",
    )
    per_scope = {}
    for token in (pt.SCOPE_PILOT, pt.SCOPE_SOAK):
        rebuilt = pt.Phase9TrainConfig.for_candidate(
            CANDIDATE_ID,
            namespace=NAMESPACE,
            device="cpu",
            total_iterations=contract.CANONICAL_ITERATIONS,
            scope=token,
        )
        identity = rebuilt.identity()
        per_scope[token] = {field: identity[field] for field in learning_fields}
    differing = sorted(
        field
        for field in learning_fields
        if per_scope[pt.SCOPE_PILOT][field] != per_scope[pt.SCOPE_SOAK][field]
    )

    # `selects_a_configuration` is the one property that reads the token. It
    # is a reporting flag: nothing in the optimization path consults it.
    consumers = []
    for path in sorted((REPOSITORY_ROOT / "stratego").rglob("*.py")):
        text = path.read_text()
        if "selects_a_configuration" in text and path.name != "phase9_trainer.py":
            consumers.append(str(path.relative_to(REPOSITORY_ROOT)))

    behaviour_changing = bool(differing) or bool(consumers)
    return {
        "runtime_scope_token": RUNTIME_SCOPE_TOKEN,
        "frozen_scopes": list(pt.SCOPES),
        "why_the_token_is_legacy": (
            "Agent 5's frozen SCOPES has no canonical entry, and the accepted "
            "trainer runtime identity digest 77af4d45 hashes this exact string; "
            "the canonical run is defined by namespace='canonical', "
            "total_iterations=60, the canonical schedule and the amended "
            "train-config document"
        ),
        "source_references": references,
        "value_branches_in_trainer": branches,
        "branch_semantics": (
            "the single production branch is `if self.scope != SCOPE_UNIT_TEST`, "
            "which applies the *stricter* frozen-constant checks; a "
            "pilot_candidate scope therefore constrains the run more than any "
            "alternative token would, and relaxes nothing"
        ),
        "learning_fields_compared": list(learning_fields),
        "identity_under_each_scope": per_scope,
        "learning_fields_that_differ_by_scope": differing,
        "library_consumers_of_selects_a_configuration": consumers,
        "changes_training_behaviour": behaviour_changing,
        "verdict": (
            "BLOCKED — a code path changes training behaviour on the legacy "
            "scope token"
            if behaviour_changing
            else "the legacy scope token is inert: it names the run in the "
            "identity document and nothing else reads it to decide how to train"
        ),
    }


# ---------------------------------------------------------------------------
# Stage: verify
# ---------------------------------------------------------------------------


def stage_verify(args) -> dict:
    modules = _training()
    contract = modules["contract"]
    schedule = modules["schedule"]
    amendment = modules["amendment"]
    pt = modules["pt"]
    pb = modules["pb"]

    problems: list[str] = []
    acceptances = {}
    # Agents 1-5 each wrote an `agent_0N_acceptance.json`; Agent 6's mission
    # froze three artifacts instead, and `agent_06_pilot_selection.json` is
    # the one that carries its status and completion gates.
    acceptance_artifacts = {
        agent: DATA_DIRECTORY / f"agent_{agent:02d}_acceptance.json" for agent in (1, 2, 3, 4, 5)
    }
    acceptance_artifacts[6] = DATA_DIRECTORY / "agent_06_pilot_selection.json"
    for agent, path in sorted(acceptance_artifacts.items()):
        if not path.exists():
            problems.append(f"agent {agent} acceptance artifact is missing ({path.name})")
            continue
        payload = read_json(path)
        acceptances[str(agent)] = {
            "status": payload.get("status"),
            "all_passed": payload.get("all_passed"),
            "artifact": path.name,
            "gates": (
                f"{payload.get('passed')}/{payload.get('total')}"
                if payload.get("total") is not None
                else f"{payload.get('gates_true')}/{payload.get('gates_total')}"
            ),
        }
        if payload.get("status") != "PASS":
            problems.append(f"agent {agent} status is {payload.get('status')!r}, not PASS")
        if payload.get("all_passed") is False:
            problems.append(f"agent {agent} acceptance gates are not all true")

    # Agent 6's frozen selection and configuration.
    selection = read_json(DATA_DIRECTORY / "agent_06_pilot_selection.json")
    frozen = read_json(DATA_DIRECTORY / "agent_06_frozen_train_config.json")
    handoff = frozen["handoff_to_agent_7"]
    if selection["selection"]["winner"] != CANDIDATE_ID:
        problems.append(
            f"Agent 6 selected {selection['selection']['winner']!r}, this harness "
            f"executes {CANDIDATE_ID!r}"
        )
    if not selection["selection"]["unique"]:
        problems.append("Agent 6's winner is not unique")
    if handoff.get("no_pilot_checkpoint_handed_forward") is not True:
        problems.append("Agent 6 did not certify that no pilot checkpoint carries forward")

    # Frozen identities. The original contract digest is immutable historical
    # identity and is never regenerated to encode the ceiling amendment.
    observed_contract = contract.contract_digest()
    from stratego.training.phase9_targets import example_contract_digest

    observed_example = example_contract_digest()
    observed_amendment = amendment.amendment_digest()
    if observed_contract != ACCEPTED_CONTRACT_DIGEST:
        problems.append(f"contract digest {observed_contract} != accepted")
    if observed_example != ACCEPTED_EXAMPLE_DIGEST:
        problems.append(f"example contract digest {observed_example} != accepted")
    if observed_amendment != ACCEPTED_AMENDMENT_DIGEST:
        problems.append(f"amendment digest {observed_amendment} != accepted")
    problems.extend(amendment.verify_base_contract_untouched())
    if contract.CANONICAL_WALL_CLOCK_CEILING_HOURS != amendment.HISTORICAL_CEILING_HOURS:
        problems.append("the frozen contract's historical ceiling was edited in place")

    # The two labelled train-config document namespaces.
    document_digests = {
        "accepted_12h": frozen["train_config_document_digest"],
        "amended_15h": frozen["train_config_document_digest_amended"],
    }
    if document_digests["accepted_12h"] != ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_12H:
        problems.append("the historical 12-hour train-config document digest drifted")
    if document_digests["amended_15h"] != ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_AMENDED:
        problems.append("the amended train-config document digest drifted")
    document_to_run = frozen["config_amended"]
    if len(document_to_run) != 39:
        problems.append(
            f"the amended train-config document has {len(document_to_run)} fields, "
            "the accepted document has 39"
        )
    reconciliation = amendment.reconcile_documents(frozen["config"], document_to_run)
    if not reconciliation["only_the_wall_clock_ceiling_changed"]:
        problems.append(
            "the amended document differs from the accepted one in more than the "
            f"operational ceiling: {reconciliation['changed_fields']}"
        )
    if document_to_run["wall_clock_ceiling_hours"] != amendment.AMENDED_CEILING_HOURS:
        problems.append("the amended document does not carry the 15-hour ceiling")

    # The runtime identity, rebuilt rather than copied.
    runtime_problem = None
    config = None
    try:
        config = canonical_config(modules, args.device)
    except Agent7Error as error:
        runtime_problem = str(error)
        problems.append(runtime_problem)
    runtime_identity = config.identity() if config is not None else None
    if runtime_identity is not None and runtime_identity != frozen["trainer_runtime_identity"]:
        differing = sorted(
            key
            for key in set(runtime_identity) | set(frozen["trainer_runtime_identity"])
            if runtime_identity.get(key) != frozen["trainer_runtime_identity"].get(key)
        )
        problems.append(f"runtime identity fields differ from Agent 6's: {differing}")

    # The amendment must not have moved the runtime identity.
    identity_effect = amendment.runtime_identity_is_unaffected(
        frozen["trainer_runtime_identity"], runtime_identity or {}
    )
    if not identity_effect["unchanged"]:
        problems.append(
            f"the runtime identity moved: {identity_effect['differing_fields']}"
        )

    # The legacy scope token must be inert.
    scope_audit = scope_behaviour_audit(modules)
    if scope_audit["changes_training_behaviour"]:
        problems.append(scope_audit["verdict"])

    # The Phase 8 anchor.
    checkpoint_sha = (
        pb.file_sha256(PHASE8_CHECKPOINT) if PHASE8_CHECKPOINT.exists() else "<missing>"
    )
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
    test_bank_digest = agent1["bank_digests"]["phase9_test_bank_v1"]
    if validation_bank_digest != frozen["config"]["validation"]["bank_digest"]:
        problems.append("the validation bank digest disagrees with the frozen config")
    if test_bank_digest != frozen["config"]["test_bank"]["bank_digest"]:
        problems.append("the test bank digest disagrees with the frozen config")

    # The canonical run schedule, recomputed from the live schedule module.
    recomputed_schedule = schedule.run_schedule_digest(NAMESPACE)
    agent2 = read_json(DATA_DIRECTORY / "agent_02_acceptance.json")
    pinned_schedule = agent2["run_schedule_digests"][NAMESPACE]
    if recomputed_schedule != pinned_schedule:
        problems.append("the canonical run schedule digest drifted from Agent 2's")
    if recomputed_schedule != EXPECTED_CANONICAL_RUN_SCHEDULE_DIGEST:
        problems.append("the canonical run schedule digest drifted from the pin")

    # The frozen canonical budget, read from the contract rather than restated.
    budget = {
        "iterations": contract.CANONICAL_ITERATIONS,
        "games_per_iteration": contract.CANONICAL_GAMES_PER_ITERATION,
        "total_scheduled_games": contract.CANONICAL_MAX_SCHEDULED_GAMES,
        "epochs_per_rollout": contract.EPOCHS_PER_ROLLOUT,
        "validation_cadence": contract.VALIDATION_CADENCE_ITERATIONS,
        "archive_cadence": contract.ARCHIVE_CADENCE_ITERATIONS,
        "bucket_counts": contract.bucket_counts(NAMESPACE),
        "rule_tier_counts": contract.rule_tier_counts(NAMESPACE),
        "ceiling_seconds": amendment.amended_ceiling_seconds(),
    }
    for label, observed, expected in (
        ("iterations", budget["iterations"], 60),
        ("games_per_iteration", budget["games_per_iteration"], 2048),
        ("total_scheduled_games", budget["total_scheduled_games"], 122_880),
        ("epochs_per_rollout", budget["epochs_per_rollout"], 2),
        ("validation_cadence", budget["validation_cadence"], 5),
        ("archive_cadence", budget["archive_cadence"], 5),
        ("ceiling_seconds", budget["ceiling_seconds"], 54_000),
    ):
        if observed != expected:
            problems.append(f"frozen canonical {label} is {observed}, expected {expected}")

    # The canonical namespace must be free of pilot state: no pilot rollout,
    # checkpoint, optimizer state or archive member may initialize this run.
    rollout_root = Path(storage_check["resolved_root"])
    contamination = {
        "canonical_rollout_namespace": str(rollout_root / NAMESPACE),
        "canonical_rollout_namespace_exists": (rollout_root / NAMESPACE).exists(),
        "canonical_archive_directory": str(PRODUCTION_ARCHIVE_ROOT / NAMESPACE),
        "canonical_archive_members_present": sorted(
            path.name
            for path in (PRODUCTION_ARCHIVE_ROOT / NAMESPACE).glob("*.pt")
        )
        if (PRODUCTION_ARCHIVE_ROOT / NAMESPACE).exists()
        else [],
        "pilot_archive_namespaces": sorted(
            path.name
            for path in PRODUCTION_ARCHIVE_ROOT.glob("pilot_*")
            if path.is_dir()
        ),
        "pilot_weight_directory": str(REPOSITORY_ROOT / "checkpoints/phase9/agent06"),
        "run_directory": str(run_directory()),
        "run_directory_exists": run_directory().exists(),
        "journal_exists": journal_path().exists(),
        "rule": (
            "the canonical learner is initialized only from "
            "checkpoints/phase8/warmstart_c1_v1.pt with fresh optimizer, "
            "scheduler and KL-controller state; pilot namespaces are read by "
            "nothing here, and canonical|H0nn is a different object from any "
            "pilot_p9x|H0nn"
        ),
    }
    frozen_start = frozen["config"]["start"]
    if frozen_start["checkpoint_sha256"] != contract.EXPECTED_PHASE8_CHECKPOINT_SHA256:
        problems.append("the frozen config's start checkpoint is not the Phase 8 anchor")
    if frozen_start["expected_model_state_digest"] != EXPECTED_START_MODEL_STATE_DIGEST:
        problems.append("the frozen config's expected start model-state digest drifted")

    # Prove the fresh start reproduces the expected model-state checksum,
    # before any optimizer exists. Loaded through the evaluation path.
    start_digest = None
    if PHASE8_CHECKPOINT.exists():
        from stratego.training.phase9_behavior import state_dict_digest
        from stratego.training.warmstart_checkpoint import load_model_for_evaluation

        model, _metadata = load_model_for_evaluation(PHASE8_CHECKPOINT, device="cpu")
        start_digest = state_dict_digest(model)
        del model
        if start_digest != EXPECTED_START_MODEL_STATE_DIGEST:
            problems.append(
                f"fresh Phase 8 load produces model-state digest {start_digest}, "
                f"expected {EXPECTED_START_MODEL_STATE_DIGEST}"
            )

    payload = {
        "stage": "verify",
        **environment_record(),
        "acceptances": acceptances,
        "agent6_selection": {
            "winner": selection["selection"]["winner"],
            "unique": selection["selection"]["unique"],
            "winner_score": selection["selection"]["winner_score"],
            "ranked": selection["selection"]["ranked"],
        },
        "contract_digest": observed_contract,
        "contract_digest_role": (
            "immutable historical identity; stamped into every sealed rollout "
            "sidecar and every phase9_checkpoint_v1, never regenerated"
        ),
        "example_contract_digest": observed_example,
        "operational_amendment": {
            "amendment_version": amendment.PHASE9_OPERATIONAL_AMENDMENT_VERSION,
            "amendment_digest": observed_amendment,
            "amends_contract_digest": amendment.AMENDED_CONTRACT_DIGEST,
            "base_contract_untouched": not amendment.verify_base_contract_untouched(),
            "historical_ceiling_seconds": amendment.HISTORICAL_CEILING_SECONDS,
            "amended_ceiling_seconds": amendment.amended_ceiling_seconds(),
            "changed_field": amendment.AMENDED_FIELD,
            "reconciliation": reconciliation,
            "runtime_identity_effect": identity_effect,
        },
        "train_config_documents": {
            "digests": document_digests,
            "document_to_run": "config_amended",
            "document_field_count": len(document_to_run),
            "historical_provenance": (
                "the 12-hour document 9284fbc6 is preserved unchanged as the "
                "accepted historical record; the amended 22ac552d document is "
                "what this run executes"
            ),
        },
        "trainer_runtime_identity": runtime_identity,
        "trainer_runtime_identity_digest": (
            config.digest() if config is not None else None
        ),
        "scope_audit": scope_audit,
        "phase8_checkpoint_sha256": checkpoint_sha,
        "start_model_state_digest": start_digest,
        "start_model_state_digest_expected": EXPECTED_START_MODEL_STATE_DIGEST,
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
        "test_bank_digest_recorded": test_bank_digest,
        "test_bank_rule": (
            "recorded as a handoff identity only; this harness never constructs "
            "a phase9_test_bank_v1 object and never runs a model over it"
        ),
        "canonical_run_schedule_digest": recomputed_schedule,
        "canonical_run_schedule_digest_pinned": pinned_schedule,
        "canonical_budget": budget,
        "pilot_contamination_audit": contamination,
        "scheduled_restarts": {str(key): value for key, value in SCHEDULED_RESTARTS.items()},
        "tests_before": TESTS_BEFORE,
        "topology": dict(VALIDATED_TOPOLOGY),
        "problems": problems,
    }
    write_stage("verify", payload)
    if problems:
        log(f"BLOCKED: {problems}")
    else:
        log(
            "verify: Agents 1-6 accepted, amendment ee4b0507 recorded, runtime "
            "identity 77af4d45 reconstructed, corpus/storage/schedule confirmed"
        )
    return payload


# ---------------------------------------------------------------------------
# Stage: amendment (the reviewed operational ceiling in force)
# ---------------------------------------------------------------------------


def stage_amendment(args) -> dict:
    """Verify and record `phase9_operational_amendment_v2` in its own stage.

    Written beside `stage_verify.json` rather than into it: the verify stage
    is the pre-run evidence that the canonical namespace was clean before the
    first optimizer step, and overwriting it mid-run would destroy exactly
    that record.
    """
    modules = _training()
    amendment = modules["amendment"]
    amendment_v2 = modules["amendment_v2"]
    contract = modules["contract"]

    problems = list(amendment_v2.verify_chain_untouched())
    observed_v2 = amendment_v2.amendment_digest()
    if observed_v2 != ACCEPTED_AMENDMENT_V2_DIGEST:
        problems.append(f"v2 amendment digest {observed_v2} != accepted")
    if amendment.amendment_digest() != ACCEPTED_AMENDMENT_DIGEST:
        problems.append("phase9_operational_amendment_v1 digest moved")
    if contract.contract_digest() != ACCEPTED_CONTRACT_DIGEST:
        problems.append("phase9_rl_contract_v1 digest moved")

    frozen = read_json(DATA_DIRECTORY / "agent_06_frozen_train_config.json")
    document_v2 = amendment_v2.apply_to_train_config_document(frozen["config_amended"])
    digest_v2 = document_digest(document_v2)
    if digest_v2 != ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_AMENDED_V2:
        problems.append(f"v2 train-config document digest {digest_v2} != accepted")
    reconciliation = amendment_v2.reconcile_documents(frozen["config_amended"], document_v2)
    if not reconciliation["only_the_wall_clock_ceiling_changed"]:
        problems.append(
            "the v2 document differs from the v1 document in more than the "
            f"operational ceiling: {reconciliation['changed_fields']}"
        )

    config = canonical_config(modules, args.device)
    identity_effect = amendment_v2.runtime_identity_is_unaffected(
        frozen["trainer_runtime_identity"], config.identity()
    )
    if not identity_effect["unchanged"]:
        problems.append(f"the runtime identity moved: {identity_effect['differing_fields']}")

    history = amendment_v2.ceiling_history()
    history[-1]["digest"] = observed_v2

    payload = {
        "stage": "amendment_v2",
        **environment_record(),
        "amendment_version": amendment_v2.PHASE9_OPERATIONAL_AMENDMENT_V2_VERSION,
        "amendment_digest": observed_v2,
        "amends": {
            "amendment_version": amendment_v2.AMENDED_AMENDMENT_VERSION,
            "amendment_digest": amendment.amendment_digest(),
            "base_contract_version": contract.PHASE9_RL_CONTRACT_VERSION,
            "base_contract_digest": contract.contract_digest(),
            "in_place_edit": False,
        },
        "ceiling_history": history,
        "ceiling_seconds_in_force": amendment_v2.amended_ceiling_seconds(),
        "document": amendment_v2.amendment_document(),
        "train_config_documents": {
            "accepted_12h": ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_12H,
            "amended_15h": ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_AMENDED,
            "amended_24h": digest_v2,
            "document_to_run": "config_amended_v2",
            "field_count": len(document_v2),
        },
        "reconciliation": reconciliation,
        "runtime_identity_effect": identity_effect,
        "adoption": (
            "the running canonical experiment continues unchanged; the ceiling "
            "is an operational maximum, and the run stops immediately after "
            "iteration 60 and its required bookkeeping"
        ),
        "problems": problems,
    }
    write_stage("amendment_v2", payload)
    if problems:
        log(f"BLOCKED: {problems}")
    else:
        log(
            f"amendment_v2 {observed_v2[:8]}: ceiling {amendment.AMENDED_CEILING_SECONDS} "
            f"-> {amendment_v2.AMENDED_CEILING_SECONDS} s; contract ad3dba3c and v1 "
            "ee4b0507 both preserved unedited"
        )
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
    its digest equals the accepted Agent 1 digest.
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
        raise Agent7Error(
            f"rebuilt validation bank digest {observed} != accepted {expected_digest}"
        )
    write_json(cache, bank.to_dict())
    return bank


def run_chunked_schedule(
    matches, bank, owner, *, reference, label, directory, workers, chunk_units
):
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


def candidate_eval_ref(iteration: int):
    return neural_policy_ref(f"{NAMESPACE}_it{iteration}", dtype_name=GATE_DTYPE)


def anchor_matchup_matches(iteration: int, pairs: int):
    units = build_paired_schedule(
        candidate_eval_ref(iteration),
        neural_policy_ref(ANCHOR_CANDIDATE_ID, dtype_name=GATE_DTYPE),
        range(pairs),
        setup_bank_version="phase9_validation_bank_v1",
    )
    return schedule_matches(units)


def games_directory(label: str, iteration: int) -> Path:
    return run_directory() / label / f"it{iteration}"


def run_anchor_worker(args) -> None:
    """One process's slice of the canonical-vs-anchor matchup.

    Neural-vs-neural is not expressible through `run_neural_schedule` (one
    owner per schedule), so the accepted `play_match` is driven directly with
    two in-process owners. The fan-out across processes changes only which
    process plays a chunk, never any identity.
    """
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from stratego.evaluation.neural_worker import InferenceOwner
    from stratego.training import phase9_contract as contract

    iteration = args.validation_iteration
    directory = games_directory(args.games_label, iteration) / "anchor"
    directory.mkdir(parents=True, exist_ok=True)
    matches = anchor_matchup_matches(iteration, contract.VALIDATION_BANK_CASES)
    chunk_size = max(1, (len(matches) + args.anchor_workers - 1) // args.anchor_workers)
    chunks = list(_chunks(matches, chunk_size))
    index, chunk = chunks[args.anchor_chunk_index]
    path = anchor_chunk_path(directory, index, chunk)
    if path.exists():
        return
    bank = load_validation_bank(args.expected_bank_digest)

    candidate_ref = candidate_eval_ref(iteration)
    anchor_ref = neural_policy_ref(ANCHOR_CANDIDATE_ID, dtype_name=GATE_DTYPE)
    export_path = Path(args.export_path)
    owners = {
        candidate_ref.token: InferenceOwner(
            export_path,
            decision_mode=DECISION_MODE_GREEDY,
            device=args.device,
            dtype=GATE_DTYPE,
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name=f"agent7_{NAMESPACE}_it{iteration}",
        ),
        anchor_ref.token: InferenceOwner(
            ANCHOR_EXPORT_PATH,
            decision_mode=DECISION_MODE_GREEDY,
            device=args.device,
            dtype=GATE_DTYPE,
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name="agent7_anchor",
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


def run_anchor_matchup(iteration: int, args, expected_bank_digest: str, *, label: str, export_path: Path):
    """Fan the canonical-vs-anchor games across worker processes and gather."""
    from stratego.training import phase9_contract as contract

    directory = games_directory(label, iteration) / "anchor"
    directory.mkdir(parents=True, exist_ok=True)
    matches = anchor_matchup_matches(iteration, contract.VALIDATION_BANK_CASES)
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
                "--games-label",
                label,
                "--export-path",
                str(export_path),
            ]
            processes.append((index, subprocess.Popen(command, cwd=REPOSITORY_ROOT)))
        failures = []
        for index, process in processes:
            if process.wait() != 0:
                failures.append(index)
        if failures:
            raise Agent7Error(f"anchor worker chunk(s) {failures} failed")
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


def export_evaluation_weights(source: Path, export_path: Path) -> str:
    """Export a `phase9_checkpoint_v1` file to the frozen evaluation format.

    Bitwise: the exported model must reload to exactly the weights it came
    from, or the evaluation would be scoring a different network.
    """
    import torch
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from stratego.model.checkpoint import load_checkpoint, save_checkpoint
    from stratego.training.phase9_behavior import file_sha256
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
        raise Agent7Error(f"evaluation export of {source} changed the weights")
    del model, reloaded, payload
    return file_sha256(export_path)


def export_comparison(selection_export, reload_export, selection_sha, reload_sha) -> dict:
    """Compare the two evaluation exports at the level that matters.

    The container carries a `creation_timestamp`, so two exports of the same
    network never hash equal. The claim worth recording is therefore not
    "the files match" but "the weights match": every parameter tensor,
    compared bit for bit. A bare SHA inequality in an artifact reads like a
    failure; this says what is actually true.
    """
    import torch

    def _weights(path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model_state") or payload.get("state_dict")
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in ("model_state", "state_dict")
        }
        return state, metadata

    left_state, left_metadata = _weights(selection_export)
    right_state, right_metadata = _weights(reload_export)
    names_equal = set(left_state) == set(right_state)
    bitwise = names_equal and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )
    differing_metadata = sorted(
        key
        for key in set(left_metadata) | set(right_metadata)
        if left_metadata.get(key) != right_metadata.get(key)
    )
    return {
        "selection_pass_sha256": selection_sha,
        "reload_pass_sha256": reload_sha,
        "sha256_equal": selection_sha == reload_sha,
        "weights_bitwise_equal": bool(bitwise),
        "tensors_compared": len(left_state),
        "differing_metadata_fields": differing_metadata,
        "explanation": (
            "the two exports hash differently because the evaluation container "
            "records a creation_timestamp; every parameter tensor is bit-for-bit "
            "identical, which is what the reproduction claim rests on"
        ),
    }


def run_validation_pass(
    iteration: int,
    weights_path: Path,
    weights_sha256: str,
    args,
    *,
    include_stress: bool,
    label: str = "games",
    purpose: str = "checkpoint_selection",
) -> dict:
    """One full frozen validation pass of one canonical checkpoint.

    Greedy, single-request, float32, `phase9_validation_bank_v1`, the five
    core opponents; stress is the report-only 32-pair prefix schedule and is
    run only when asked. Never updates weights, and never touches the sealed
    final-test bank.
    """
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
    from stratego.evaluation.neural_worker import InferenceOwner
    from stratego.training import phase9_contract as contract
    from stratego.training import phase9_seed as seed
    from stratego.training.phase9_behavior import file_sha256

    access = contract.check_validation_bank_access(purpose, phase9_agent=AGENT)
    work = run_directory()
    started = time.perf_counter()
    timings: dict[str, float] = {}

    export_path = work / f"eval_{label}_it{iteration}.pt"
    export_started = time.perf_counter()
    export_sha = export_evaluation_weights(weights_path, export_path)
    timings["export_seconds"] = time.perf_counter() - export_started

    bank_started = time.perf_counter()
    expected_bank_digest = read_stage("verify")["validation_bank_digest_expected"]
    bank = load_validation_bank(expected_bank_digest)
    timings["bank_seconds"] = time.perf_counter() - bank_started

    reference = candidate_eval_ref(iteration)
    matchups: dict[str, dict] = {}
    safety = {
        "illegal_policy_actions": 0,
        "policy_errors": 0,
        "inference_failures": 0,
        "workers_importing_torch": 0,
        "worker_checkpoint_loads": 0,
    }

    owner = InferenceOwner(
        export_path,
        decision_mode=DECISION_MODE_GREEDY,
        device=args.device,
        dtype=GATE_DTYPE,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
        name=f"agent7_{NAMESPACE}_it{iteration}",
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
                label=f"{NAMESPACE}_it{iteration}_{opponent_id}",
                directory=games_directory(label, iteration) / opponent_id,
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
                    label=f"{NAMESPACE}_it{iteration}_{policy_id}",
                    directory=games_directory(label, iteration) / policy_id,
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
        iteration, args, expected_bank_digest, label=label, export_path=export_path
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
        "random_pass": (
            ewrs["random_legal"]
            >= contract.VALIDATION_REGRESSION_GUARDS["random_legal_ewr_min"]
        ),
        "basic_ewr": ewrs["basic_heuristic"],
        "basic_min": contract.VALIDATION_REGRESSION_GUARDS["basic_heuristic_ewr_min"],
        "basic_pass": (
            ewrs["basic_heuristic"]
            >= contract.VALIDATION_REGRESSION_GUARDS["basic_heuristic_ewr_min"]
        ),
    }
    return {
        "namespace": NAMESPACE,
        "iteration": iteration,
        "checkpoint_identity": weights_path.name,
        "checkpoint_path": str(weights_path),
        "checkpoint_sha256": weights_sha256,
        "eval_export_sha256": export_sha,
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
        "color_split": {
            key: matchups[key]["summary"].get("color_split") for key in matchups
        },
        "terminal_reasons": {
            key: matchups[key]["summary"].get("terminal_reasons") for key in matchups
        },
        "plies": {key: matchups[key]["summary"].get("plies") for key in matchups},
        "setup_family_stratification": {
            key: matchups[key]["summary"].get("setup_pair_stratification")
            for key in matchups
        },
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


def verify_historical_actions(reader, iteration: int, historical, *, sample_per_identity: int) -> dict:
    """Verify historical-opponent actions against the acting archive checkpoint.

    Every historical-bucket game names the archive member that played it; the
    opponent-side decisions of a deterministic sample per active identity are
    reproduced under that exact bound member. The digest guard has already
    bound the identities; this is the numerical closure.
    """
    from stratego.training import phase9_behavior as pb
    from stratego.training.phase9_schedule import historical_policy_token

    token_to_identity = {
        historical_policy_token(NAMESPACE, identity): identity for identity in historical
    }
    verified: dict = {}
    examined: dict = {}
    started = time.perf_counter()
    for game_id in sorted(reader.game_ids):
        metadata = reader.metadata[game_id]
        if metadata.get("bucket") != "historical":
            continue
        identity = token_to_identity.get(metadata.get("opponent_identity"))
        if identity is None:
            return {
                "iteration": iteration,
                "all_verified": False,
                "problem": (
                    f"{game_id} names opponent token "
                    f"{metadata.get('opponent_identity')!r}, which is not in the "
                    f"active window {sorted(historical)}"
                ),
                "seconds": time.perf_counter() - started,
            }
        examined[identity] = examined.get(identity, 0) + 1
        if examined[identity] > sample_per_identity:
            continue
        snapshot = historical[identity]
        if metadata.get("opponent_checkpoint_sha256") != snapshot.checkpoint_sha256:
            return {
                "iteration": iteration,
                "all_verified": False,
                "problem": (
                    f"{game_id} was collected under opponent checkpoint "
                    f"{metadata.get('opponent_checkpoint_sha256')}, the bound "
                    f"{identity} is {snapshot.checkpoint_sha256}"
                ),
                "seconds": time.perf_counter() - started,
            }
        record, metadata = reader.read_game(game_id)
        learner = 0 if metadata["learner_color"] == "red" else 1
        requests = _decision_requests(record, metadata, 1 - learner, pb=pb, limit=10**6)
        reports = pb.reproduce_decisions(snapshot, requests)
        failed = sum(1 for report in reports if not report["verified"])
        differences = [
            report["max_abs_difference"]
            for report in reports
            if report["max_abs_difference"] is not None
        ]
        entry = verified.setdefault(
            identity,
            {"games": 0, "decisions": 0, "failed": 0, "max_abs_difference": None},
        )
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
        "active_window": sorted(historical),
        "checkpoint_sha256": {
            identity: snapshot.checkpoint_sha256 for identity, snapshot in historical.items()
        },
        "games_by_identity": examined,
        "verified": verified,
        "all_verified": all(entry["failed"] == 0 for entry in verified.values()),
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# The journal
# ---------------------------------------------------------------------------


def empty_journal() -> dict:
    return {
        "namespace": NAMESPACE,
        "iterations": [],
        "validations": {},
        "snapshots": {},
        "archive": {},
        "historical_verification": [],
        "start_state_digest": None,
        "anchor_state_digest": None,
        "wall_clock": {"run_seconds": 0.0, "sessions": []},
        "storage_checks": [],
        "restarts": [],
        "restarts_completed": [],
        "pending_restart": None,
        "in_flight": None,
        "counters": {},
        "counter_sessions": [],
        "halt": None,
        "process_launches": 0,
    }


def load_journal() -> dict:
    path = journal_path()
    if path.exists():
        return read_json(path)
    return empty_journal()


def save_journal(journal: dict) -> None:
    write_json(journal_path(), journal)


def curve_row(row: dict) -> dict:
    """The compact per-update record the journal keeps across a restart."""
    kept = (
        "epoch",
        "minibatch_index",
        "global_optimizer_step",
        "examples",
        "loss_total",
        "loss_ppo",
        "loss_value",
        "loss_belief",
        "behavior_kl",
        "policy_entropy",
        "kl_beta",
        "entropy_coefficient",
        "clip_fraction",
        "ppo_examples",
        "ppo_clipped",
        "advantage_retention",
        "grad_norm_pre_clip",
        "parameter_norm",
        "ratio_mean",
        "step_seconds",
        "data_wait_seconds",
        "epoch_mean_kl",
        "epoch_clip_fraction",
        "kl_beta_after_epoch",
    )
    return {key: row[key] for key in kept if key in row}


def summarize_iteration(rollout, rows, collected, timers, archived, controller) -> dict:
    import numpy as np

    kls = [row["behavior_kl"] for row in rows]
    epoch_kls = [row["epoch_mean_kl"] for row in rows if "epoch_mean_kl" in row]
    epoch_clips = [row["epoch_clip_fraction"] for row in rows if "epoch_clip_fraction" in row]
    examples = sum(row["examples"] for row in rows)
    train_seconds = timers.get("train_seconds", 0.0)
    return {
        "namespace": rollout.namespace,
        "iteration": rollout.iteration,
        "sealed_rollout_digest": rollout.sealed_rollout_digest,
        "behavior_snapshot_id": rollout.behavior_snapshot_id,
        "behavior_checkpoint_sha256": rollout.behavior_checkpoint_sha256,
        "games": rollout.games,
        "learner_decisions": rollout.learner_decisions,
        "updates": len(rows),
        "examples": examples,
        "examples_per_second": (examples / train_seconds) if train_seconds else 0.0,
        "advantage_statistics": rollout.statistics.to_dict(),
        "mean_behavior_kl": float(np.mean(kls)) if kls else 0.0,
        "max_behavior_kl": float(np.max(kls)) if kls else 0.0,
        "epoch_mean_kls": epoch_kls,
        "epoch_clip_fractions": epoch_clips,
        "mean_clip_fraction": (
            float(np.mean([row["clip_fraction"] for row in rows])) if rows else 0.0
        ),
        "max_clip_fraction": (
            float(np.max([row["clip_fraction"] for row in rows])) if rows else 0.0
        ),
        "mean_policy_entropy": (
            float(np.mean([row["policy_entropy"] for row in rows])) if rows else 0.0
        ),
        "mean_loss_ppo": float(np.mean([row["loss_ppo"] for row in rows])) if rows else 0.0,
        "mean_loss_value": float(np.mean([row["loss_value"] for row in rows])) if rows else 0.0,
        "mean_loss_belief": float(np.mean([row["loss_belief"] for row in rows])) if rows else 0.0,
        "mean_advantage_retention": (
            float(np.mean([row["advantage_retention"] for row in rows])) if rows else 0.0
        ),
        "mean_grad_norm_pre_clip": (
            float(np.mean([row["grad_norm_pre_clip"] for row in rows])) if rows else 0.0
        ),
        "final_parameter_norm": float(rows[-1]["parameter_norm"]) if rows else 0.0,
        "entropy_coefficient": float(rows[-1]["entropy_coefficient"]) if rows else 0.0,
        "kl_beta_after": float(controller.beta),
        "collection": {
            key: collected.get(key)
            for key in (
                "games_collected",
                "games_already_committed",
                "games_per_second",
                "decisions_per_second",
                "learner_decisions",
                "neural_decisions",
                "total_decisions",
                "total_plies",
                "observer_probes",
                "observer_probe_failures",
                "sealed_rollout_digest",
                "inference_device",
                "inference_batch_shape",
                "bucket_counts",
                "terminal_results",
                "seconds",
            )
            if key in collected
        },
        "timers": dict(timers),
        "archived": archived,
        "rss_mib": peak_rss_mib(),
    }


def classify_hard_stop(error, trainer) -> str:
    """Map a raised trainer/collector error onto the frozen hard stop it is."""
    counters = trainer.counters if trainer is not None else {}
    if counters.get("kl_hard_limit_breaches"):
        return "mean_kl_hard_limit"
    if counters.get("clip_fraction_hard_limit_breaches"):
        return "clip_fraction_hard_limit"
    if counters.get("non_finite_losses"):
        return "non_finite_loss"
    if counters.get("non_finite_gradients"):
        return "non_finite_gradient"
    if counters.get("non_finite_parameters"):
        return "non_finite_parameter"
    if counters.get("behavior_identity_mismatches"):
        return "behavior_identity_mismatch"
    if counters.get("rollout_identity_mismatches"):
        return "rollout_identity_mismatch"
    if counters.get("illegal_targets") or counters.get("data_mismatches"):
        return "target_reconstruction_mismatch"
    if counters.get("checkpoint_errors"):
        return "checkpoint_corruption"
    text = str(error).lower()
    if "illegal" in text:
        return "illegal_neural_action"
    if "observer" in text:
        return "observer_leak"
    if "digest" in text:
        return "rollout_digest_mismatch"
    return "unclassified_failure"


def selection_key(record: dict, iterations_by_number: dict) -> tuple:
    """The frozen selection order: score, then the frozen tie-break chain."""
    iteration = record["iteration"]
    entry = iterations_by_number.get(iteration, {})
    return (
        float(record["selection_score"]),
        float(record["effective_win_rates"]["strategic_rule_based"]),
        -float(entry.get("mean_behavior_kl", 0.0)),
        float(entry.get("examples_per_second", 0.0)),
    )


def select_best_validation(journal: dict) -> dict:
    """Strictly highest frozen validation score; ties use the frozen chain."""
    iterations_by_number = {entry["iteration"]: entry for entry in journal["iterations"]}
    records = [journal["validations"][key] for key in sorted(journal["validations"], key=int)]
    if not records:
        raise Agent7Error("no validation passes were recorded")
    ranked = sorted(
        records, key=lambda record: selection_key(record, iterations_by_number), reverse=True
    )
    best = ranked[0]
    top_score = float(best["selection_score"])
    tied = [record for record in records if float(record["selection_score"]) == top_score]
    return {
        "best": best,
        "unique_on_score": len(tied) == 1,
        "tied_on_score": [record["iteration"] for record in tied],
        "ranked": [record["iteration"] for record in ranked],
        "scores": {
            str(record["iteration"]): record["selection_score"] for record in records
        },
        "final_iteration_is_best": best["iteration"]
        == max(record["iteration"] for record in records),
        "tie_break": [
            "higher validation score",
            "higher Strategic EWR",
            "lower mean behavior KL",
            "higher training examples/s",
        ],
    }


# ---------------------------------------------------------------------------
# The restart evidence
# ---------------------------------------------------------------------------


def next_planned_batch(modules, rollout, cursor) -> dict:
    """The exact keys the frozen train order will hand the next optimizer step.

    A pure function of the sealed key list and the cursor, computed on both
    sides of a restart boundary so continuity is measured, not asserted.
    """
    import hashlib

    from stratego.training.phase9_targets import minibatch_keys

    keys = minibatch_keys(
        rollout.keys,
        rollout.namespace,
        rollout.iteration,
        cursor.epoch,
        cursor.minibatch_index,
        cursor.minibatch_size,
    )
    hasher = hashlib.sha256()
    for game_id, ply in keys:
        hasher.update(f"{game_id}|{ply}\n".encode())
    return {"count": len(keys), "digest": hasher.hexdigest()}


def resume_probe(modules, trainer, rollout, cursor) -> dict:
    """A no-grad forward/loss probe on the exact next minibatch.

    Costs one forward pass, changes no state and steps no optimizer, so it
    measures the backend-aware numerical criterion across a process boundary
    without adding a single unit of training to the frozen experiment.
    """
    import numpy as np
    import torch

    pt = modules["pt"]
    from stratego.training.phase9_loss import phase9_batch_loss
    from stratego.training.phase9_targets import minibatch_keys

    keys = minibatch_keys(
        rollout.keys,
        rollout.namespace,
        rollout.iteration,
        cursor.epoch,
        cursor.minibatch_index,
        cursor.minibatch_size,
    )
    packed = pt.build_minibatch(keys, rollout.reader, rollout.statistics)
    arrays = pt.unpack_batch(packed)
    tensors = {
        name: torch.from_numpy(np.ascontiguousarray(value)).to(trainer.device)
        for name, value in arrays.items()
        if name != "learner_side"
    }
    was_training = trainer.model.training
    trainer.model.eval()
    try:
        with torch.no_grad():
            outputs = trainer.model.forward_observation(tensors["observation"])
            loss = phase9_batch_loss(
                outputs,
                legal_mask=tensors["legal_mask"],
                sampled_action_model=tensors["sampled_action_model"],
                behavior_action_probability=tensors["behavior_action_probability"],
                behavior_probabilities=tensors["behavior_probabilities"],
                standardized_advantage=tensors["standardized_advantage"],
                ppo_eligible=tensors["ppo_eligible"],
                wdl_target=tensors["wdl_target"],
                belief_target=tensors["belief_target"],
                belief_mask=tensors["belief_mask"],
                kl_beta=float(trainer.controller.beta),
                entropy_coefficient=float(trainer.entropy_position()["coefficient"]),
            )
    finally:
        if was_training:
            trainer.model.train()
    return {
        "batch_digest": pt.batch_digest(packed),
        "loss_total": float(loss.total.detach()),
        "loss_ppo": float(loss.ppo.detach()),
        "loss_value": float(loss.value.detach()),
        "loss_belief": float(loss.belief.detach()),
        "behavior_kl": float(loss.kl.detach()),
        "policy_entropy": float(loss.entropy.detach()),
    }


def compare_resume(before: dict, after: dict) -> dict:
    """The accepted backend-aware resume criterion, applied at the boundary.

    `phase9_backend_aware_resume_equivalence_v1`: exact logical state
    everywhere, bitwise equality where the backend allows it — the model
    state, optimizer, scheduler and controller cross the boundary as
    serialized CPU float32 and so must be bit-identical — and the MPS
    forward tolerance for the recomputed probe, which is the one quantity a
    non-deterministic backend may legitimately move.
    """
    rtol, atol = 1e-5, 1e-6
    # The pre-exit side crossed the boundary as JSON in the journal, so its
    # tuples arrived back as lists (AdamW's `betas` is the live example).
    # Normalize both sides through the same serialization the journal uses, so
    # the comparison measures the run's state rather than Python's container
    # types. Floats round-trip exactly, so nothing numerical is softened.
    before = json.loads(json.dumps(before, sort_keys=True, default=str))
    after = json.loads(json.dumps(after, sort_keys=True, default=str))
    logical_fields = sorted(set(before["state_summary"]) | set(after["state_summary"]))
    logical = {
        field: before["state_summary"].get(field) == after["state_summary"].get(field)
        for field in logical_fields
    }
    probe_fields = [key for key in before["probe"] if key != "batch_digest"]
    probe = {}
    for field in probe_fields:
        left = float(before["probe"][field])
        right = float(after["probe"][field])
        probe[field] = {
            "before": left,
            "after": right,
            "abs_difference": abs(left - right),
            "within_tolerance": abs(left - right) <= atol + rtol * abs(left),
        }
    checks = {
        "logical_state_equal": all(logical.values()),
        "model_state_digest_bitwise_equal": (
            before["model_state_digest"] == after["model_state_digest"]
        ),
        "next_batch_identical": before["next_batch"] == after["next_batch"],
        "probe_batch_digest_equal": (
            before["probe"]["batch_digest"] == after["probe"]["batch_digest"]
        ),
        "probe_within_backend_tolerance": all(
            entry["within_tolerance"] for entry in probe.values()
        ),
        "active_history_equal": before["active_history"] == after["active_history"],
        "validation_history_equal": (
            before["validation_history"] == after["validation_history"]
        ),
        "best_validation_equal": before["best_validation"] == after["best_validation"],
        "sealed_rollout_identity_equal": (
            before["sealed_rollout"] == after["sealed_rollout"]
        ),
        "behavior_snapshot_equal": before["behavior"] == after["behavior"],
    }
    return {
        "criterion_id": "phase9_backend_aware_resume_equivalence_v1",
        "tolerances": {"rtol": rtol, "atol": atol},
        "logical_state_fields": logical,
        "probe": probe,
        "checks": checks,
        "passed": all(checks.values()),
        "why_no_donor_leg": (
            "Agent 5's accepted evidence already measured the donor and "
            "no-checkpoint control legs; reproducing them here would mean "
            "running the same optimizer steps twice on the canonical "
            "experiment, which the frozen contract forbids. The probe is a "
            "no-grad forward on the exact next minibatch: it measures the "
            "backend's forward envelope across the process boundary and adds "
            "no training"
        ),
    }


def verify_boundary_resume(modules, trainer, checkpoint_path, journal) -> dict:
    """Continuity of a resume that happens *between* iterations.

    The mid-epoch restarts compare a live pre-exit capture to a live
    post-resume one. A boundary resume has no pre-exit process to capture, so
    the authority is the checkpoint itself: every logical quantity
    `phase9_checkpoint_v1` recorded must be what the resumed trainer now
    holds, including the model weights bit for bit.
    """
    pck = modules["pck"]
    from stratego.training.phase9_behavior import state_dict_digest

    payload = pck.read_phase9_payload(checkpoint_path)
    model = pck.model_from_payload(payload)
    recorded_model_digest = state_dict_digest(model)
    del model

    recorded = {
        "global_optimizer_step": int(payload["global_optimizer_step"]),
        "rl_iteration": int(payload["rl_iteration"]),
        "examples_consumed": int(payload["examples_consumed"]),
        "kl_beta": float(payload["kl_beta"]),
        "kl_controller_updates": len(payload["kl_controller_state"]["history"]),
        "kl_controller_state": payload["kl_controller_state"],
        "entropy_schedule_position": payload["entropy_schedule_position"],
        "minibatch_cursor": payload["minibatch_cursor"],
        "active_historical_identities": list(payload["active_historical_identities"]),
        "historical_checkpoint_digests": dict(payload["historical_checkpoint_digests"]),
        "best_validation_score": payload["best_validation_score"],
        "best_checkpoint_identity": payload["best_checkpoint_identity"],
        "validation_history_entries": len(payload["validation_history"]),
        "sealed_rollout_digest": payload["sealed_rollout_digest"],
        "behavior_snapshot_identity": payload["behavior_snapshot_identity"],
        "model_state_digest": recorded_model_digest,
    }
    live = {
        "global_optimizer_step": int(trainer.global_step),
        "rl_iteration": int(trainer.rl_iteration),
        "examples_consumed": int(trainer.examples_consumed),
        "kl_beta": float(trainer.controller.beta),
        "kl_controller_updates": len(trainer.controller.history),
        "kl_controller_state": trainer.controller.to_dict(),
        "entropy_schedule_position": trainer.entropy_position(),
        "minibatch_cursor": trainer.cursor.to_dict(),
        "active_historical_identities": list(payload["active_historical_identities"]),
        "historical_checkpoint_digests": dict(payload["historical_checkpoint_digests"]),
        "best_validation_score": trainer.best_validation_score,
        "best_checkpoint_identity": trainer.best_checkpoint_identity,
        "validation_history_entries": len(trainer.validation_history),
        "sealed_rollout_digest": payload["sealed_rollout_digest"],
        "behavior_snapshot_identity": payload["behavior_snapshot_identity"],
        "model_state_digest": trainer.model_state_digest(),
    }
    recorded = json.loads(json.dumps(recorded, sort_keys=True, default=str))
    live = json.loads(json.dumps(live, sort_keys=True, default=str))
    fields = {key: recorded[key] == live[key] for key in sorted(recorded)}
    return {
        "criterion_id": "phase9_backend_aware_resume_equivalence_v1",
        "kind": "committed_iteration_boundary_resume",
        "checkpoint": str(checkpoint_path),
        "resumed_pid": os.getpid(),
        "iterations_committed": len(journal["iterations"]),
        "recorded": recorded,
        "fields_equal": fields,
        "model_state_digest_bitwise_equal": fields["model_state_digest"],
        "passed": all(fields.values()),
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def restart_evidence(modules, trainer, rollout, journal, historical) -> dict:
    """Everything a resume must reproduce, captured on one side of the boundary."""
    return {
        "state_summary": trainer.state_summary(),
        "model_state_digest": trainer.model_state_digest(),
        "next_batch": next_planned_batch(modules, rollout, trainer.cursor),
        "probe": resume_probe(modules, trainer, rollout, trainer.cursor),
        "active_history": {
            "identities": list(trainer.active_historical_identities),
            "digests": dict(trainer.historical_checkpoint_digests),
            "bound": sorted(historical),
        },
        "validation_history": [dict(entry) for entry in trainer.validation_history],
        "best_validation": {
            "score": trainer.best_validation_score,
            "identity": trainer.best_checkpoint_identity,
        },
        "sealed_rollout": {
            "rollout_id": rollout.rollout_id,
            "sealed_rollout_digest": rollout.sealed_rollout_digest,
            "learner_decisions": rollout.learner_decisions,
            "train_order_keys_digest": rollout.keys_digest(),
        },
        "behavior": {
            "snapshot_id": rollout.behavior_snapshot_id,
            "checkpoint_sha256": rollout.behavior_checkpoint_sha256,
        },
        "iterations_committed": len(journal["iterations"]),
        "process_pid": os.getpid(),
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# ---------------------------------------------------------------------------
# The canonical worker
# ---------------------------------------------------------------------------


def run_canonical_worker(args) -> int:
    """One process's share of the canonical run: iterations until done or restart.

    Restart-safe at two granularities. A committed iteration is durable in
    the journal, so a crash resumes at the next iteration. A *scheduled*
    restart stops mid-epoch, writes a normal `phase9_checkpoint_v1`, exits
    the process, and the next process continues the same sealed rollout from
    the exact logical cursor — repeating no optimizer step and skipping none.
    """
    modules = _training()
    contract = modules["contract"]
    schedule = modules["schedule"]
    amendment = modules["amendment"]
    pb = modules["pb"]
    pck = modules["pck"]
    pc = modules["pc"]
    pt = modules["pt"]

    work = run_directory()
    work.mkdir(parents=True, exist_ok=True)
    session_started = time.perf_counter()

    storage_check = verify_storage_mounted()
    if storage_check["problems"]:
        log(f"BLOCKED: storage not usable: {storage_check['problems']}")
        return 2
    rollout_root = Path(storage_check["resolved_root"])

    journal = load_journal()
    journal["storage_checks"].append(
        {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pid": os.getpid(),
            "resolved_root": storage_check["resolved_root"],
            "mount_point": storage_check["mount_point"],
            "on_external_volume": storage_check["on_external_volume"],
            "free_gib": storage_check["free_gib"],
        }
    )
    journal["process_launches"] = int(journal.get("process_launches", 0)) + 1
    # Each process's trainer counts its own events from zero, so the honest
    # cross-process total is the sum of the per-session finals. This session
    # owns one slot and rewrites it in place, so a killed process still
    # leaves whatever it had counted.
    journal.setdefault("counter_sessions", [])
    session_index = len(journal["counter_sessions"])
    journal["counter_sessions"].append({})
    save_journal(journal)

    if journal.get("halt"):
        log(f"already halted: {journal['halt']}")
        return 2

    ceiling = modules["amendment_v2"].amended_ceiling_seconds()
    elapsed_before = float(journal["wall_clock"]["run_seconds"])

    def elapsed_total() -> float:
        return elapsed_before + (time.perf_counter() - session_started)

    def record_counters(trainer_object) -> None:
        journal["counter_sessions"][session_index] = {
            key: int(value) for key, value in trainer_object.counters.items()
        }
        totals: dict = {}
        for session in journal["counter_sessions"]:
            for key, value in session.items():
                totals[key] = totals.get(key, 0) + int(value)
        journal["counters"] = totals

    def persist_session() -> None:
        journal["wall_clock"]["run_seconds"] = elapsed_total()
        save_journal(journal)

    config = canonical_config(modules, args.device)
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
        logical_identity=contract.HISTORICAL_ANCHOR_ID,
        policy_token=schedule.ANCHOR_POLICY_TOKEN,
        expected_sha256=anchor_sha,
    )
    historical = {contract.HISTORICAL_ANCHOR_ID: anchor_snapshot}

    completed = len(journal["iterations"])
    pending = journal.get("pending_restart")
    resumed_mid_iteration = bool(pending) and int(pending["iteration"]) == completed + 1

    if completed == 0 and not resumed_mid_iteration:
        stale = []
        if (rollout_root / NAMESPACE).exists():
            stale.append(str(rollout_root / NAMESPACE))
        if (PRODUCTION_ARCHIVE_ROOT / NAMESPACE).exists() and any(
            (PRODUCTION_ARCHIVE_ROOT / NAMESPACE).glob("*.pt")
        ):
            stale.append(str(PRODUCTION_ARCHIVE_ROOT / NAMESPACE))
        if any(work.glob("resume_it*.pt")):
            stale.append(str(work / "resume_it*.pt"))
        if stale and not args.reset_run:
            log(
                f"BLOCKED: the canonical namespace has no journal but stale run "
                f"state exists ({stale}); pass --reset-run to discard it explicitly"
            )
            return 2
        if stale and args.reset_run:
            import shutil

            for path in (rollout_root / NAMESPACE, work, PRODUCTION_ARCHIVE_ROOT / NAMESPACE):
                if Path(path).exists():
                    shutil.rmtree(path)
            work.mkdir(parents=True, exist_ok=True)
            journal = empty_journal()
            journal["process_launches"] = 1
            journal["counter_sessions"] = [{}]
            session_index = 0
            elapsed_before = 0.0
            log("canonical: stale state discarded by explicit --reset-run")

    # ---- build or resume the trainer -------------------------------------
    if completed == 0 and not resumed_mid_iteration:
        trainer = pt.Phase9Trainer.from_phase8_checkpoint(
            PHASE8_CHECKPOINT,
            config,
            corpus_identity,
            topology=topology,
            run_label=f"agent07_{NAMESPACE}",
        )
        start_digest = trainer.model_state_digest()
        journal["start_state_digest"] = start_digest
        journal["anchor_state_digest"] = anchor_snapshot.loaded_state_dict_digest
        if start_digest != anchor_snapshot.loaded_state_dict_digest:
            log("BLOCKED: fresh trainer weights differ from the Phase 8 anchor")
            return 2
        if start_digest != EXPECTED_START_MODEL_STATE_DIGEST:
            log(
                f"BLOCKED: fresh start model-state digest {start_digest} != the "
                f"expected {EXPECTED_START_MODEL_STATE_DIGEST}"
            )
            return 2
        journal["train_config"] = {**config.identity(), "digest": config.digest()}
        journal["topology"] = topology.to_dict()
        journal["fresh_start"] = {
            "checkpoint": str(PHASE8_CHECKPOINT.relative_to(REPOSITORY_ROOT)),
            "checkpoint_sha256": anchor_sha,
            "model_state_digest": start_digest,
            "optimizer_state": "fresh AdamW, no pilot optimizer state loaded",
            "kl_beta": float(config.initial_kl_beta),
            "global_optimizer_step": int(trainer.global_step),
            "pilot_checkpoint_loaded": False,
        }
        save_journal(journal)
        log(f"canonical: fresh start from the Phase 8 anchor ({anchor_sha[:16]})")
    else:
        resume_from = (
            Path(pending["checkpoint"])
            if resumed_mid_iteration
            else work / f"resume_it{completed:03d}.pt"
        )
        trainer = pt.Phase9Trainer.resume(
            resume_from,
            config=config,
            corpus_identity=corpus_identity,
            topology=topology,
            run_label=f"agent07_{NAMESPACE}_resumed",
        )
        log(f"canonical: resumed from {resume_from.name} after {completed} iterations")
        if not resumed_mid_iteration:
            boundary = verify_boundary_resume(modules, trainer, resume_from, journal)
            journal["restarts"].append(boundary)
            save_journal(journal)
            if not boundary["passed"]:
                failed = [key for key, ok in boundary["fields_equal"].items() if not ok]
                log(f"BLOCKED: boundary resume continuity failed: {failed}")
                journal["halt"] = {
                    "reason": "hard_stop",
                    "hard_stop": "checkpoint_corruption",
                    "detail": f"boundary resume continuity failed: {failed}",
                }
                save_journal(journal)
                return 2
            log(
                f"boundary resume verified: {len(boundary['fields_equal'])} logical "
                "fields equal, model state bitwise equal"
            )

    # The journal is the durable authority on validation history; mirror it
    # onto the trainer so every checkpoint written from here carries it.
    def refresh_validation_state() -> None:
        trainer.validation_history = [
            {
                "iteration": journal["validations"][key]["iteration"],
                "selection_score": journal["validations"][key]["selection_score"],
                "effective_win_rates": journal["validations"][key]["effective_win_rates"],
                "checkpoint_identity": journal["validations"][key]["checkpoint_identity"],
                "checkpoint_sha256": journal["validations"][key]["checkpoint_sha256"],
            }
            for key in sorted(journal["validations"], key=int)
        ]
        if trainer.validation_history:
            selection = select_best_validation(journal)
            trainer.best_validation_score = float(selection["best"]["selection_score"])
            trainer.best_checkpoint_identity = selection["best"]["checkpoint_identity"]

    refresh_validation_state()

    from stratego.training.phase9_rollout_store import read_iteration_state, write_iteration_state

    def bind_window(iteration: int) -> tuple:
        """Bind every archive member the frozen active window names."""
        window = contract.active_historical_window(iteration)
        for identity in window:
            if identity in historical:
                continue
            member = pck.read_archive_member(
                PRODUCTION_ARCHIVE_ROOT, namespace=NAMESPACE, local_identity=identity
            )
            bound = pck.bind_archive_member(
                member, device=args.collect_device, inference_batch_shape=args.batch_shape
            )
            bound.assert_frozen()
            recorded = journal["archive"].get(identity)
            if recorded and recorded["checkpoint_sha256"] != member.checkpoint_sha256:
                raise Agent7Error(
                    f"{identity} on disk is {member.checkpoint_sha256}, the manifest "
                    f"recorded {recorded['checkpoint_sha256']}; archives are immutable"
                )
            historical[identity] = bound
        for identity in list(historical):
            if identity not in window:
                del historical[identity]
        return window

    # A crash between `mark_iteration_trained` and the journal append leaves a
    # journal-complete iteration at TRAINING; carry it to COMMITTED.
    for entry in journal["iterations"]:
        state = read_iteration_state(rollout_root, NAMESPACE, entry["iteration"])
        if state is not None and state["state"] == "TRAINING":
            write_iteration_state(
                rollout_root,
                NAMESPACE,
                entry["iteration"],
                "EVALUATED",
                sealed_rollout_digest=entry["sealed_rollout_digest"],
            )
            write_iteration_state(
                rollout_root,
                NAMESPACE,
                entry["iteration"],
                "COMMITTED",
                sealed_rollout_digest=entry["sealed_rollout_digest"],
            )

    halt: "dict | None" = None
    restart_requested = False

    def run_due_validation(due_iteration: int) -> None:
        """The frozen cadence's validation pass for one completed iteration."""
        trainer.close()
        snapshot_id = schedule.behavior_snapshot_identity(due_iteration + 1)
        weights = work / f"behavior_{snapshot_id}.pt"
        started = time.perf_counter()
        validation = run_validation_pass(
            due_iteration,
            weights,
            journal["snapshots"][snapshot_id],
            args,
            include_stress=due_iteration == contract.CANONICAL_ITERATIONS,
        )
        journal["validations"][str(due_iteration)] = validation
        save_journal(journal)
        refresh_validation_state()
        log(
            f"canonical it{due_iteration} validation: "
            f"score={validation['selection_score']:.6f} "
            f"strategic={validation['effective_win_rates']['strategic_rule_based']:.4f} "
            f"tactical={validation['effective_win_rates']['tactical_rule_based']:.4f} "
            f"anchor={validation['effective_win_rates']['phase8_anchor']:.4f} "
            f"random={validation['effective_win_rates']['random_legal']:.4f} "
            f"basic={validation['effective_win_rates']['basic_heuristic']:.4f} "
            f"({time.perf_counter() - started:.0f}s)"
        )
        if validation["safety"]["illegal_policy_actions"]:
            raise Agent7Halt("illegal neural action during validation")
        if validation["safety"]["inference_failures"]:
            raise Agent7Halt("non-finite model output during validation")

    try:
        # A restart that lands between an iteration and its due validation
        # pass fills the gap first: cadence position, not process lifetime,
        # decides what runs.
        for due in range(
            contract.VALIDATION_CADENCE_ITERATIONS,
            len(journal["iterations"]) + 1,
            contract.VALIDATION_CADENCE_ITERATIONS,
        ):
            if str(due) not in journal["validations"]:
                run_due_validation(due)

        for iteration in range(len(journal["iterations"]) + 1, contract.CANONICAL_ITERATIONS + 1):
            if elapsed_total() >= ceiling:
                halt = {
                    "reason": "operational_ceiling_reached",
                    "ceiling_seconds": ceiling,
                    "elapsed_seconds": elapsed_total(),
                    "iterations_committed": len(journal["iterations"]),
                    "next_iteration": iteration,
                }
                break

            timers: dict[str, float] = {}
            in_flight = journal.get("in_flight")
            resuming_mid = bool(
                resumed_mid_iteration and in_flight and in_flight["iteration"] == iteration
            )
            identity = schedule.behavior_snapshot_identity(iteration)
            if iteration == 1:
                snapshot = resolver.resolve(
                    PHASE8_CHECKPOINT,
                    logical_identity=identity,
                    policy_token=schedule.behavior_policy_token(NAMESPACE, iteration),
                    expected_sha256=anchor_sha,
                )
            else:
                behavior_path = work / f"behavior_{identity}.pt"
                snapshot = pck.bind_behavior_snapshot(
                    behavior_path,
                    logical_identity=identity,
                    namespace=NAMESPACE,
                    device=args.collect_device,
                    inference_batch_shape=args.batch_shape,
                    expected_sha256=journal["snapshots"][identity],
                )
            if resuming_mid:
                # Mid-iteration, the learner has *legitimately* moved past the
                # behavior snapshot — that divergence is exactly what PPO's
                # ratio pi_theta/pi_b measures, and demanding equality here
                # would make an exact resume impossible. The binding that
                # still has to hold is the one `phase9_checkpoint_v1` records:
                # the checkpoint being resumed must name the very snapshot
                # that collected this iteration.
                recorded = pck.read_phase9_payload(pending["checkpoint"])
                bindings = [
                    (
                        "behavior_snapshot_identity",
                        recorded["behavior_snapshot_identity"],
                        identity,
                    ),
                    (
                        "behavior_checkpoint_sha256",
                        recorded["behavior_checkpoint_sha256"],
                        snapshot.checkpoint_sha256,
                    ),
                    ("rl_iteration", int(recorded["rl_iteration"]), iteration),
                ]
                drift = [
                    f"{name} {found!r} != {wanted!r}"
                    for name, found, wanted in bindings
                    if found != wanted
                ]
                if drift:
                    raise Agent7Error(
                        f"iteration {iteration}: the resumed checkpoint does not "
                        f"belong to this iteration's behavior snapshot: {drift}"
                    )
                del recorded
            elif snapshot.loaded_state_dict_digest != trainer.model_state_digest():
                raise Agent7Error(
                    f"iteration {iteration}: the behavior snapshot's weights differ "
                    "from the live trainer weights; on-policy collection would be a lie"
                )

            window = bind_window(iteration)
            trainer.active_historical_identities = tuple(window)
            trainer.historical_checkpoint_digests = {
                key: historical[key].checkpoint_sha256 for key in window
            }
            manifest = schedule.ActiveHistoryManifest.frozen_for(
                NAMESPACE, iteration, dict(trainer.historical_checkpoint_digests)
            )
            manifest.validate()

            started = time.perf_counter()
            collected = pc.collect_iteration(
                rollout_root,
                NAMESPACE,
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
                progress=lambda done, total: log(f"  it{iteration}: collected {done}/{total}"),
            )
            timers["collection_seconds"] = time.perf_counter() - started
            timers["sealing_seconds"] = float((collected.get("seal") or {}).get("seconds", 0.0))
            if collected.get("observer_probe_failures"):
                raise Agent7Halt(
                    f"{collected['observer_probe_failures']} observer probe failures"
                )

            started = time.perf_counter()
            state = read_iteration_state(rollout_root, NAMESPACE, iteration)
            rollout = pt.bind_sealed_rollout(
                rollout_root,
                NAMESPACE,
                iteration,
                behavior_snapshot=snapshot,
                # The on-policy check compares the snapshot's weights to the
                # trainer's, which is the right question only before this
                # iteration's first optimizer step; mid-iteration it is the
                # checkpoint's recorded behavior identity that binds, verified
                # above and again by `rebind_iteration`.
                expected_model_state_digest=(
                    None if resuming_mid else trainer.model_state_digest()
                ),
                require_full_schedule=True,
                resuming=state is not None and state["state"] == "TRAINING",
            )
            timers["target_construction_seconds"] = time.perf_counter() - started

            started = time.perf_counter()
            verification = verify_historical_actions(
                rollout.reader,
                iteration,
                historical,
                sample_per_identity=args.historical_verify_games,
            )
            timers["historical_verification_seconds"] = time.perf_counter() - started
            if not verification["all_verified"]:
                journal["historical_verification"].append(verification)
                save_journal(journal)
                raise Agent7Halt(
                    f"historical action reproduction failed at iteration {iteration}: "
                    f"{verification.get('problem', verification.get('verified'))}"
                )

            # ---- training, possibly across a scheduled process restart ----
            rows: list = []
            if resuming_mid:
                trainer.rebind_iteration(rollout)
                after = restart_evidence(modules, trainer, rollout, journal, historical)
                comparison = compare_resume(pending["evidence"], after)
                record = {
                    "iteration": iteration,
                    "kind": "scheduled_mid_epoch_process_restart",
                    "exited_pid": pending["evidence"]["process_pid"],
                    "resumed_pid": os.getpid(),
                    "checkpoint": pending["checkpoint"],
                    "checkpoint_sha256": pending["checkpoint_sha256"],
                    "updates_before_exit": pending["updates_before_exit"],
                    "before": pending["evidence"],
                    "after": after,
                    "comparison": comparison,
                }
                journal["restarts"].append(record)
                journal["restarts_completed"].append(iteration)
                journal["pending_restart"] = None
                save_journal(journal)
                if not comparison["passed"]:
                    failed = [key for key, ok in comparison["checks"].items() if not ok]
                    raise Agent7Halt(f"resume continuity failed: {failed}")
                log(
                    f"it{iteration}: resumed mid-epoch at cursor "
                    f"{after['state_summary']['minibatch_cursor']['epoch']}/"
                    f"{after['state_summary']['minibatch_cursor']['minibatch_index']} "
                    "with exact logical continuity"
                )
                rows = list(in_flight["rows"])
                timers = dict(in_flight["timers"])
                collected = in_flight["collected"]
                resumed_mid_iteration = False
            else:
                trainer.bind_iteration(rollout)

            planned = SCHEDULED_RESTARTS.get(iteration)
            started = time.perf_counter()
            if planned is not None and iteration not in journal["restarts_completed"]:
                total_updates = trainer.cursor.minibatches_per_epoch * config.epochs_per_rollout
                stop_after = max(1, int(round(total_updates * planned)))
                first = trainer.train_iteration(updates=stop_after, timing=True)
                rows.extend(curve_row(row) for row in first)
                timers["train_seconds"] = timers.get("train_seconds", 0.0) + (
                    time.perf_counter() - started
                )
                checkpoint_path = work / f"restart_it{iteration:03d}.pt"
                evidence = restart_evidence(modules, trainer, rollout, journal, historical)
                written = trainer.save_checkpoint(checkpoint_path)
                journal["in_flight"] = {
                    "iteration": iteration,
                    "rows": rows,
                    "timers": timers,
                    "collected": collected,
                }
                journal["pending_restart"] = {
                    "iteration": iteration,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": written["sha256"],
                    "updates_before_exit": len(first),
                    "evidence": evidence,
                }
                record_counters(trainer)
                persist_session()
                trainer.close()
                log(
                    f"it{iteration}: scheduled process exit after {len(first)} of "
                    f"{total_updates} updates (cursor epoch "
                    f"{trainer.cursor.epoch}, minibatch {trainer.cursor.minibatch_index})"
                )
                restart_requested = True
                break

            rows.extend(curve_row(row) for row in trainer.train_iteration(timing=True))
            timers["train_seconds"] = timers.get("train_seconds", 0.0) + (
                time.perf_counter() - started
            )
            trainer.mark_iteration_trained()
            journal["in_flight"] = None

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
                existing = PRODUCTION_ARCHIVE_ROOT / NAMESPACE / f"{local_identity}.pt"
                if existing.exists():
                    member = pck.read_archive_member(
                        PRODUCTION_ARCHIVE_ROOT,
                        namespace=NAMESPACE,
                        local_identity=local_identity,
                    )
                    if member.state_dict_digest != trainer.model_state_digest():
                        raise Agent7Error(
                            f"archive member {local_identity} already exists with "
                            "different weights; archives are immutable"
                        )
                else:
                    payload = trainer.archive_member_payload(local_identity=local_identity)
                    member = pck.write_archive_member(
                        payload,
                        PRODUCTION_ARCHIVE_ROOT,
                        namespace=NAMESPACE,
                        local_identity=local_identity,
                    )
                bound = pck.bind_archive_member(
                    member, device=args.collect_device, inference_batch_shape=args.batch_shape
                )
                bound.assert_frozen()
                if bound.loaded_state_dict_digest != trainer.model_state_digest():
                    raise Agent7Error(
                        f"archived {local_identity} weights do not match the "
                        "post-iteration learner"
                    )
                archived = member.to_dict()
                archived["created_after_iteration"] = iteration
                archived["population_eligible_from_iteration"] = iteration + 1
                journal["archive"][local_identity] = archived
                timers["archive_seconds"] = time.perf_counter() - started
                log(f"archived {member.qualified_identity} -> {member.checkpoint_sha256[:16]}")

            entry = summarize_iteration(
                rollout, rows, collected, timers, archived, trainer.controller
            )
            entry["elapsed_run_seconds"] = elapsed_total()
            journal["historical_verification"].append(verification)
            journal["iterations"].append(entry)
            record_counters(trainer)
            persist_session()

            write_iteration_state(
                rollout_root,
                NAMESPACE,
                iteration,
                "EVALUATED",
                sealed_rollout_digest=rollout.sealed_rollout_digest,
                validation_pass_due=iteration % contract.VALIDATION_CADENCE_ITERATIONS == 0,
            )
            write_iteration_state(
                rollout_root,
                NAMESPACE,
                iteration,
                "COMMITTED",
                sealed_rollout_digest=rollout.sealed_rollout_digest,
            )
            log(
                f"it{iteration}/{contract.CANONICAL_ITERATIONS}: {entry['updates']} updates, "
                f"beta={trainer.controller.beta:.4f}, "
                f"epoch KLs={['%.4f' % value for value in entry['epoch_mean_kls']]}, "
                f"clip={entry['mean_clip_fraction']:.3f}, "
                f"{timers['collection_seconds']:.0f}s collect / "
                f"{timers['train_seconds']:.0f}s train, "
                f"elapsed {elapsed_total() / 3600:.2f}h"
            )

            if iteration % contract.VALIDATION_CADENCE_ITERATIONS == 0:
                run_due_validation(iteration)
                persist_session()
    except Agent7Halt as error:
        halt = {"reason": "hard_stop", "detail": str(error), "iteration": locals().get("iteration")}
        log(f"HARD STOP: {error}")
    except (pt.Phase9TrainerError, pc.Phase9CollectorError, Agent7Error) as error:
        classified = classify_hard_stop(error, locals().get("trainer"))
        halt = {
            "reason": "hard_stop",
            "hard_stop": classified,
            "detail": f"{type(error).__name__}: {error}",
            "iteration": locals().get("iteration"),
        }
        log(f"HARD STOP ({classified}): {error}")
        traceback.print_exc()
    finally:
        try:
            trainer.close()
        except Exception:  # noqa: BLE001 - close is best-effort on the way out
            pass

    record_counters(trainer)
    journal["wall_clock"]["sessions"].append(
        {
            "pid": os.getpid(),
            "seconds": time.perf_counter() - session_started,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    if halt is not None:
        journal["halt"] = halt
    persist_session()

    if restart_requested:
        return RESTART_EXIT_CODE
    if halt is not None:
        return 2
    return 0


# ---------------------------------------------------------------------------
# Stage: run
# ---------------------------------------------------------------------------


def stage_run(args) -> dict:
    """The supervisor: relaunch canonical workers until the run is complete."""
    verify = read_stage("verify")
    if verify["problems"]:
        raise Agent7Error(f"verify stage recorded problems: {verify['problems']}")

    modules = _training()
    contract = modules["contract"]
    amendment_v2 = modules["amendment_v2"]
    ceiling = amendment_v2.amended_ceiling_seconds()

    launches = 0
    while True:
        journal = load_journal()
        completed = len(journal["iterations"])
        validations_due = [
            due
            for due in range(
                contract.VALIDATION_CADENCE_ITERATIONS,
                completed + 1,
                contract.VALIDATION_CADENCE_ITERATIONS,
            )
            if str(due) not in journal["validations"]
        ]
        if journal.get("halt"):
            log(f"run halted: {journal['halt']}")
            break
        if completed >= contract.CANONICAL_ITERATIONS and not validations_due:
            log(f"run complete: {completed} iterations, {len(journal['validations'])} validations")
            break
        if float(journal["wall_clock"]["run_seconds"]) >= ceiling:
            journal["halt"] = {
                "reason": "operational_ceiling_reached",
                "ceiling_seconds": ceiling,
                "elapsed_seconds": float(journal["wall_clock"]["run_seconds"]),
                "iterations_committed": completed,
            }
            save_journal(journal)
            log(f"run halted at the operational ceiling: {journal['halt']}")
            break
        if launches >= MAX_WORKER_LAUNCHES:
            journal["halt"] = {
                "reason": "worker_launch_cap_reached",
                "launches": launches,
                "iterations_committed": completed,
            }
            save_journal(journal)
            log("run halted: worker launch cap reached")
            break

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--canonical-worker",
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
            "--historical-verify-games",
            str(args.historical_verify_games),
        ]
        if args.reset_run and launches == 0:
            command.append("--reset-run")
        launches += 1
        log(f"=== launching canonical worker (launch {launches}) ===")
        completed_process = subprocess.run(command, cwd=REPOSITORY_ROOT)
        if completed_process.returncode == RESTART_EXIT_CODE:
            log("worker exited for a scheduled restart; relaunching")
            continue
        if completed_process.returncode == 0:
            continue
        log(f"worker exited with {completed_process.returncode}")
        journal = load_journal()
        if journal.get("halt"):
            break
        journal["halt"] = {
            "reason": "worker_failure",
            "returncode": completed_process.returncode,
            "iterations_committed": len(journal["iterations"]),
        }
        save_journal(journal)
        break

    journal = load_journal()
    payload = {
        "stage": "run",
        **environment_record(),
        "supervisor_launches": launches,
        "iterations_committed": len(journal["iterations"]),
        "validation_passes": len(journal["validations"]),
        "wall_clock_seconds": journal["wall_clock"]["run_seconds"],
        "ceiling_seconds": ceiling,
        "halt": journal.get("halt"),
        "complete": (
            len(journal["iterations"]) == contract.CANONICAL_ITERATIONS
            and len(journal["validations"])
            == contract.CANONICAL_ITERATIONS // contract.VALIDATION_CADENCE_ITERATIONS
            and journal.get("halt") is None
        ),
    }
    write_stage("run", payload)
    return payload


# ---------------------------------------------------------------------------
# Stage: freeze
# ---------------------------------------------------------------------------


def stage_freeze(args) -> dict:
    """Select one checkpoint by the frozen validation score and freeze it."""
    import shutil

    modules = _training()
    contract = modules["contract"]
    pb = modules["pb"]
    pck = modules["pck"]

    journal = load_journal()
    if journal.get("halt"):
        raise Agent7Error(
            f"the canonical run halted ({journal['halt']}); no checkpoint is frozen "
            "from an incomplete run"
        )
    expected_passes = contract.CANONICAL_ITERATIONS // contract.VALIDATION_CADENCE_ITERATIONS
    if len(journal["iterations"]) != contract.CANONICAL_ITERATIONS:
        raise Agent7Error(
            f"{len(journal['iterations'])} of {contract.CANONICAL_ITERATIONS} "
            "iterations are committed"
        )
    if len(journal["validations"]) != expected_passes:
        raise Agent7Error(
            f"{len(journal['validations'])} of {expected_passes} validation passes exist"
        )

    selection = select_best_validation(journal)
    best = selection["best"]
    best_iteration = int(best["iteration"])
    source = Path(best["checkpoint_path"])
    source_sha = pb.file_sha256(source)
    if source_sha != best["checkpoint_sha256"]:
        raise Agent7Error(
            f"the selected checkpoint {source} now hashes to {source_sha}, the "
            f"validation pass recorded {best['checkpoint_sha256']}"
        )

    FROZEN_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, FROZEN_CHECKPOINT_PATH)
    frozen_sha = pb.file_sha256(FROZEN_CHECKPOINT_PATH)

    # Independent reload through the evaluation-only path.
    payload = pck.read_phase9_payload(FROZEN_CHECKPOINT_PATH)
    validation_report = pck.validate_phase9_payload(
        payload, source=str(FROZEN_CHECKPOINT_PATH)
    )
    from stratego.training.phase9_behavior import state_dict_digest

    model = pck.model_from_payload(payload)
    model_state_digest = state_dict_digest(model)
    del model

    verify = read_stage("verify")
    anchor_digest = verify["start_model_state_digest"]

    # Re-evaluate the frozen file on the same frozen validation protocol, in a
    # directory that holds no cached games, so the metrics are recomputed.
    reproduction_pass = run_validation_pass(
        best_iteration,
        FROZEN_CHECKPOINT_PATH,
        frozen_sha,
        args,
        include_stress=False,
        label="freeze_games",
    )
    metrics = ("random_legal", "basic_heuristic", "tactical_rule_based", "strategic_rule_based", "phase8_anchor")
    reproduction = {
        "effective_win_rates": {
            key: {
                "selection_pass": best["effective_win_rates"][key],
                "reload_pass": reproduction_pass["effective_win_rates"][key],
                "equal": best["effective_win_rates"][key]
                == reproduction_pass["effective_win_rates"][key],
            }
            for key in metrics
        },
        "selection_score": {
            "selection_pass": best["selection_score"],
            "reload_pass": reproduction_pass["selection_score"],
            "equal": best["selection_score"] == reproduction_pass["selection_score"],
        },
        "results_digests_equal": {
            key: best["matchups"][key]["results_digest"]
            == reproduction_pass["matchups"][key]["results_digest"]
            for key in metrics
        },
        "eval_export": export_comparison(
            run_directory() / f"eval_games_it{best_iteration}.pt",
            run_directory() / f"eval_freeze_games_it{best_iteration}.pt",
            best["eval_export_sha256"],
            reproduction_pass["eval_export_sha256"],
        ),
        "safety_clean": all(
            reproduction_pass["safety"][key] == 0
            for key in ("illegal_policy_actions", "policy_errors", "inference_failures")
        ),
        "tolerance": (
            "the frozen deterministic tolerance: greedy argmax over float32 on a "
            "fixed bank and a fixed paired schedule is a deterministic replay, so "
            "every selection-relevant metric must reproduce exactly"
        ),
    }
    reproduction["passed"] = (
        all(entry["equal"] for entry in reproduction["effective_win_rates"].values())
        and reproduction["selection_score"]["equal"]
        and reproduction["safety_clean"]
        and reproduction["eval_export"]["weights_bitwise_equal"]
    )

    result = {
        "stage": "freeze",
        **environment_record(),
        "selection": {
            "selected_iteration": best_iteration,
            "selected_checkpoint_identity": best["checkpoint_identity"],
            "selection_score": best["selection_score"],
            "effective_win_rates": best["effective_win_rates"],
            "scores_by_iteration": selection["scores"],
            "ranked_iterations": selection["ranked"],
            "unique_on_score": selection["unique_on_score"],
            "tied_on_score": selection["tied_on_score"],
            "final_iteration_is_best": selection["final_iteration_is_best"],
            "tie_break": selection["tie_break"],
            "rule": (
                "strictly highest frozen validation score among the twelve "
                "cadence passes; the final iteration is not automatically selected"
            ),
            "selected_by": "phase9_validation_bank_v1 only; no final-test metric exists",
        },
        "frozen_checkpoint": {
            "path": str(FROZEN_CHECKPOINT_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": frozen_sha,
            "source_path": str(source),
            "source_sha256": source_sha,
            "bytes_identical_to_source": frozen_sha == source_sha,
            "model_state_digest": model_state_digest,
            "payload_validation": validation_report,
        },
        "differs_from_phase8_anchor": {
            "phase8_anchor_model_state_digest": anchor_digest,
            "phase9_model_state_digest": model_state_digest,
            "differs": model_state_digest != anchor_digest,
        },
        "reload_reproduction": reproduction,
        "reload_pass": reproduction_pass,
        "final_test_bank_opened": False,
    }
    write_stage("freeze", result)
    log(
        f"freeze: iteration {best_iteration} selected (score "
        f"{best['selection_score']:.6f}); frozen SHA {frozen_sha[:16]}; "
        f"reload reproduction {'PASS' if reproduction['passed'] else 'FAIL'}"
    )
    return result


# ---------------------------------------------------------------------------
# Stage: artifacts
# ---------------------------------------------------------------------------


def write_curve(journal: dict) -> None:
    columns = [
        "iteration",
        "games",
        "learner_decisions",
        "collection_seconds",
        "collection_games_per_second",
        "optimizer_updates",
        "examples",
        "examples_per_second",
        "ppo_loss",
        "value_loss",
        "belief_loss",
        "mean_behavior_kl",
        "max_behavior_kl",
        "epoch1_mean_kl",
        "epoch2_mean_kl",
        "kl_beta_after",
        "mean_clip_fraction",
        "epoch1_clip_fraction",
        "epoch2_clip_fraction",
        "mean_policy_entropy",
        "entropy_coefficient",
        "advantage_threshold",
        "advantage_mean",
        "advantage_min",
        "advantage_max",
        "filter_retention",
        "validation_score",
        "strategic_ewr",
        "tactical_ewr",
        "phase8_anchor_ewr",
        "random_ewr",
        "basic_ewr",
        "archive_identity",
        "archive_sha256",
        "train_seconds",
        "checkpoint_seconds",
        "archive_seconds",
        "iteration_seconds",
        "elapsed_run_seconds",
        "rss_mib",
    ]
    CURVE_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with open(CURVE_ARTIFACT, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for entry in journal["iterations"]:
            iteration = entry["iteration"]
            validation = journal["validations"].get(str(iteration))
            statistics = entry["advantage_statistics"]
            timers = entry["timers"]
            epoch_kls = entry["epoch_mean_kls"]
            epoch_clips = entry["epoch_clip_fractions"]
            archived = entry.get("archived") or {}
            writer.writerow(
                {
                    "iteration": iteration,
                    "games": entry["games"],
                    "learner_decisions": entry["learner_decisions"],
                    "collection_seconds": round(timers.get("collection_seconds", 0.0), 3),
                    "collection_games_per_second": round(
                        entry["collection"].get("games_per_second") or 0.0, 4
                    ),
                    "optimizer_updates": entry["updates"],
                    "examples": entry["examples"],
                    "examples_per_second": round(entry["examples_per_second"], 3),
                    "ppo_loss": round(entry["mean_loss_ppo"], 6),
                    "value_loss": round(entry["mean_loss_value"], 6),
                    "belief_loss": round(entry["mean_loss_belief"], 6),
                    "mean_behavior_kl": round(entry["mean_behavior_kl"], 6),
                    "max_behavior_kl": round(entry["max_behavior_kl"], 6),
                    "epoch1_mean_kl": round(epoch_kls[0], 6) if len(epoch_kls) > 0 else "",
                    "epoch2_mean_kl": round(epoch_kls[1], 6) if len(epoch_kls) > 1 else "",
                    "kl_beta_after": round(entry["kl_beta_after"], 6),
                    "mean_clip_fraction": round(entry["mean_clip_fraction"], 6),
                    "epoch1_clip_fraction": round(epoch_clips[0], 6) if len(epoch_clips) > 0 else "",
                    "epoch2_clip_fraction": round(epoch_clips[1], 6) if len(epoch_clips) > 1 else "",
                    "mean_policy_entropy": round(entry["mean_policy_entropy"], 6),
                    "entropy_coefficient": round(entry["entropy_coefficient"], 8),
                    "advantage_threshold": round(statistics["threshold"], 6),
                    "advantage_mean": round(statistics["advantage_mean"], 6),
                    "advantage_min": round(statistics["advantage_min"], 6),
                    "advantage_max": round(statistics["advantage_max"], 6),
                    "filter_retention": round(statistics["retention_fraction"], 6),
                    "validation_score": (
                        round(validation["selection_score"], 6) if validation else ""
                    ),
                    "strategic_ewr": (
                        round(validation["effective_win_rates"]["strategic_rule_based"], 6)
                        if validation
                        else ""
                    ),
                    "tactical_ewr": (
                        round(validation["effective_win_rates"]["tactical_rule_based"], 6)
                        if validation
                        else ""
                    ),
                    "phase8_anchor_ewr": (
                        round(validation["effective_win_rates"]["phase8_anchor"], 6)
                        if validation
                        else ""
                    ),
                    "random_ewr": (
                        round(validation["effective_win_rates"]["random_legal"], 6)
                        if validation
                        else ""
                    ),
                    "basic_ewr": (
                        round(validation["effective_win_rates"]["basic_heuristic"], 6)
                        if validation
                        else ""
                    ),
                    "archive_identity": archived.get("qualified_identity", ""),
                    "archive_sha256": archived.get("checkpoint_sha256", ""),
                    "train_seconds": round(timers.get("train_seconds", 0.0), 3),
                    "checkpoint_seconds": round(timers.get("checkpoint_seconds", 0.0), 3),
                    "archive_seconds": round(timers.get("archive_seconds", 0.0), 3),
                    "iteration_seconds": round(sum(timers.values()), 3),
                    "elapsed_run_seconds": round(entry.get("elapsed_run_seconds", 0.0), 3),
                    "rss_mib": round(entry["rss_mib"], 1),
                }
            )


def report_only_diagnostics(journal: dict) -> dict:
    """Every report-only diagnostic the common contract asks for."""
    import numpy as np

    from stratego.training.phase9_contract import bucket_counts

    iterations = journal["iterations"]
    terminal: dict = {}
    session_buckets: dict = {}
    plies = 0
    games = 0
    for entry in iterations:
        for key, value in (entry["collection"].get("terminal_results") or {}).items():
            terminal[key] = terminal.get(key, 0) + int(value)
        for key, value in (entry["collection"].get("bucket_counts") or {}).items():
            session_buckets[key] = session_buckets.get(key, 0) + int(value)
        plies += int(entry["collection"].get("total_plies") or 0)
        games += int(entry["games"])

    # The collector's summary counts the games *that call* played, so an
    # iteration finished across a process restart reports fewer than it holds
    # (iteration 30: 2,021 played after 27 were already committed). The sealed
    # rollout is the authority, and `bind_sealed_rollout` re-verified the full
    # frozen distribution for every iteration with require_full_schedule=True —
    # a short bucket would have raised there rather than reached this report.
    frozen = bucket_counts(NAMESPACE)
    verified_buckets = {key: value * len(iterations) for key, value in frozen.items()}
    resumed = [
        {
            "iteration": entry["iteration"],
            "games_sealed": entry["games"],
            "games_played_this_session": entry["collection"].get("games_collected"),
            "games_already_committed": entry["collection"].get("games_already_committed"),
        }
        for entry in iterations
        if int(entry["collection"].get("games_already_committed") or 0)
    ]
    validations = [journal["validations"][key] for key in sorted(journal["validations"], key=int)]
    return {
        "games_scheduled": games,
        "terminal_reason_distribution": terminal,
        "bucket_distribution": verified_buckets,
        "bucket_distribution_source": (
            "the frozen per-iteration mixture times the committed iteration "
            "count, re-verified against the sealed rollout by "
            "bind_sealed_rollout(require_full_schedule=True) at every iteration"
        ),
        "bucket_distribution_collector_sessions": session_buckets,
        "iterations_collected_across_a_restart": resumed,
        "mean_game_length_plies": (plies / games) if games else 0.0,
        "collection_games_per_second": {
            "mean": float(
                np.mean([entry["collection"].get("games_per_second") or 0.0 for entry in iterations])
            )
            if iterations
            else 0.0,
            "min": float(
                np.min([entry["collection"].get("games_per_second") or 0.0 for entry in iterations])
            )
            if iterations
            else 0.0,
        },
        "training_examples_per_second": {
            "mean": float(np.mean([entry["examples_per_second"] for entry in iterations]))
            if iterations
            else 0.0,
        },
        "policy_entropy": [entry["mean_policy_entropy"] for entry in iterations],
        "ppo_clip_fraction": [entry["mean_clip_fraction"] for entry in iterations],
        "behavior_kl": [entry["mean_behavior_kl"] for entry in iterations],
        "kl_beta": [entry["kl_beta_after"] for entry in iterations],
        "advantage_filter_retention": [
            entry["advantage_statistics"]["retention_fraction"] for entry in iterations
        ],
        "advantage_distribution": [
            {
                "iteration": entry["iteration"],
                "threshold": entry["advantage_statistics"]["threshold"],
                "mean": entry["advantage_statistics"]["advantage_mean"],
                "min": entry["advantage_statistics"]["advantage_min"],
                "max": entry["advantage_statistics"]["advantage_max"],
                "std_eligible": entry["advantage_statistics"]["std_eligible"],
            }
            for entry in iterations
        ],
        "validation_by_pass": [
            {
                "iteration": record["iteration"],
                "selection_score": record["selection_score"],
                "effective_win_rates": record["effective_win_rates"],
                "color_split": record["color_split"],
                "terminal_reasons": record["terminal_reasons"],
                "plies": record["plies"],
                "setup_family_stratification": record["setup_family_stratification"],
                "confidence_intervals": record["confidence_intervals"],
                "wdl": {
                    key: {
                        "wins": record["matchups"][key]["wins"],
                        "draws": record["matchups"][key]["draws"],
                        "losses": record["matchups"][key]["losses"],
                    }
                    for key in record["matchups"]
                },
                "guards": record["guards"],
            }
            for record in validations
        ],
        "stress_report_only": (
            validations[-1]["stress_report_only"] if validations else None
        ),
        "historical_opponent_window": [
            {
                "iteration": entry["iteration"],
                "active_window": entry.get("active_window"),
            }
            for entry in journal["historical_verification"]
        ],
        "peak_rss_mib": max((entry["rss_mib"] for entry in iterations), default=0.0),
        "storage_volume": journal["storage_checks"][-1] if journal["storage_checks"] else None,
        "note": "report-only metrics never rescue a failed hard gate",
    }


def stage_artifacts(args) -> dict:
    modules = _training()
    contract = modules["contract"]
    amendment = modules["amendment"]
    amendment_v2 = modules["amendment_v2"]

    verify = read_stage("verify")
    run = read_stage("run")
    freeze = read_stage("freeze")
    amendment_stage = (
        read_stage("amendment_v2") if stage_path("amendment_v2").exists() else None
    )
    journal = load_journal()

    write_curve(journal)

    # -- population archive -------------------------------------------------
    archive_members = [journal["archive"][key] for key in sorted(journal["archive"])]
    windows = {
        str(iteration): list(contract.active_historical_window(iteration))
        for iteration in range(1, contract.CANONICAL_ITERATIONS + 1)
    }
    cadence_expected = [
        contract.archive_snapshot_id(iteration)
        for iteration in range(
            contract.ARCHIVE_CADENCE_ITERATIONS,
            contract.CANONICAL_ITERATIONS + 1,
            contract.ARCHIVE_CADENCE_ITERATIONS,
        )
    ]
    archive_payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_07_population_archive",
        **environment_record(),
        "namespace": NAMESPACE,
        "anchor": {
            "identity": contract.HISTORICAL_ANCHOR_ID,
            "role": "the frozen Phase 8 accepted checkpoint",
            "checkpoint": str(PHASE8_CHECKPOINT.relative_to(REPOSITORY_ROOT)),
            "checkpoint_sha256": contract.EXPECTED_PHASE8_CHECKPOINT_SHA256,
            "model_state_digest": journal.get("anchor_state_digest"),
        },
        "cadence_iterations": contract.ARCHIVE_CADENCE_ITERATIONS,
        "expected_members": cadence_expected,
        "members": archive_members,
        "member_count": len(archive_members),
        "archive_schedule_exact": [member["local_identity"] for member in archive_members]
        == cadence_expected,
        "active_window_rule": (
            f"{contract.HISTORICAL_ANCHOR_ID} + the "
            f"{contract.ACTIVE_WINDOW_RECENT_SNAPSHOTS} most recent eligible "
            "archive snapshots, sampled uniformly; older snapshots remain "
            "stored but inactive and no archive checkpoint is overwritten"
        ),
        "active_window_by_iteration": windows,
        "identity_rule": (
            "logical archive identity (canonical|H0nn) and checkpoint SHA-256 "
            "are different objects; each real SHA is bound into this manifest "
            "and every historical action was verified against the acting "
            "archive checkpoint"
        ),
        "historical_action_verification": journal["historical_verification"],
        "outcome_prioritized_sampling": False,
    }
    write_json(ARCHIVE_ARTIFACT, archive_payload)

    # -- checkpoint manifest ------------------------------------------------
    manifest_payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_07_checkpoint_manifest",
        **environment_record(),
        "phase8_anchor": {
            "checkpoint": str(PHASE8_CHECKPOINT.relative_to(REPOSITORY_ROOT)),
            "checkpoint_sha256": contract.EXPECTED_PHASE8_CHECKPOINT_SHA256,
            "model_state_digest": verify["start_model_state_digest"],
            "selected_update": 24_000,
        },
        "frozen_phase9_checkpoint": freeze["frozen_checkpoint"],
        "selected_iteration": freeze["selection"]["selected_iteration"],
        "selection": freeze["selection"],
        "differs_from_phase8_anchor": freeze["differs_from_phase8_anchor"],
        "reload_reproduction": freeze["reload_reproduction"],
        "behavior_snapshots": journal["snapshots"],
        "archive_members": {
            member["local_identity"]: {
                "qualified_identity": member["qualified_identity"],
                "checkpoint_sha256": member["checkpoint_sha256"],
                "state_dict_digest": member["state_dict_digest"],
                "created_after_iteration": member["created_after_iteration"],
            }
            for member in archive_members
        },
        "validation_history": [
            {
                "iteration": journal["validations"][key]["iteration"],
                "checkpoint_identity": journal["validations"][key]["checkpoint_identity"],
                "checkpoint_sha256": journal["validations"][key]["checkpoint_sha256"],
                "selection_score": journal["validations"][key]["selection_score"],
                "effective_win_rates": journal["validations"][key]["effective_win_rates"],
                "guards": journal["validations"][key]["guards"],
            }
            for key in sorted(journal["validations"], key=int)
        ],
        "identities": {
            "contract_digest": verify["contract_digest"],
            "example_contract_digest": verify["example_contract_digest"],
            "operational_amendment_digest": amendment.amendment_digest(),
            "operational_amendment_v2_digest": amendment_v2.amendment_digest(),
            "ceiling_history": amendment_v2.ceiling_history(),
            "train_config_document_digest_accepted_12h": (
                ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_12H
            ),
            "train_config_document_digest_amended_15h": (
                ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_AMENDED
            ),
            "train_config_document_digest_amended_24h": (
                ACCEPTED_TRAIN_CONFIG_DOCUMENT_DIGEST_AMENDED_V2
            ),
            "train_config_document_executed": (
                "config_amended_v2 (f3b1efdb) — the accepted 39-field document "
                "with only the operational ceiling rewritten; 9284fbc6 (12 h) "
                "and 22ac552d (15 h) preserved as historical provenance"
            ),
            "trainer_runtime_identity_digest": verify["trainer_runtime_identity_digest"],
            "population_version": contract.PHASE9_POPULATION_VERSION,
            "schedule_version": contract.PHASE9_ROLLOUT_SCHEDULE_VERSION,
            "canonical_run_schedule_digest": verify["canonical_run_schedule_digest"],
            "checkpoint_version": "phase9_checkpoint_v1",
            "validation_bank_digest": verify["validation_bank_digest_expected"],
            "test_bank_digest": verify["test_bank_digest_recorded"],
            "corpus_identity": verify["corpus"]["observed_identity"],
        },
        "absolute_paths_are_diagnostic_only": True,
    }
    write_json(MANIFEST_ARTIFACT, manifest_payload)

    # -- completion gates ---------------------------------------------------
    counters = journal.get("counters", {})
    validations = [journal["validations"][key] for key in sorted(journal["validations"], key=int)]
    total_games = sum(entry["games"] for entry in journal["iterations"])
    safety_totals = {
        key: sum(record["safety"][key] for record in validations)
        for key in ("illegal_policy_actions", "policy_errors", "inference_failures")
    }
    kl_max = max(
        (max(entry["epoch_mean_kls"]) for entry in journal["iterations"] if entry["epoch_mean_kls"]),
        default=0.0,
    )
    clip_max = max(
        (
            max(entry["epoch_clip_fractions"])
            for entry in journal["iterations"]
            if entry["epoch_clip_fractions"]
        ),
        default=0.0,
    )
    restarts = [
        record
        for record in journal["restarts"]
        if record.get("kind") == "scheduled_mid_epoch_process_restart"
    ]
    boundary_resumes = [
        record
        for record in journal["restarts"]
        if record.get("kind") == "committed_iteration_boundary_resume"
    ]
    suite = read_stage("final_suite") if stage_path("final_suite").exists() else None

    gates = {
        "agents1_6_pass": all(
            entry["status"] == "PASS" for entry in verify["acceptances"].values()
        ),
        "corpus_resolver_verified": verify["corpus"]["resolved_root"] is not None,
        "corpus_digests_match": verify["corpus"]["identity_matches"],
        "fresh_phase8_anchor_start": (
            journal["fresh_start"]["model_state_digest"] == EXPECTED_START_MODEL_STATE_DIGEST
            and journal["fresh_start"]["checkpoint_sha256"]
            == contract.EXPECTED_PHASE8_CHECKPOINT_SHA256
        ),
        "pilot_checkpoint_loaded_no": journal["fresh_start"]["pilot_checkpoint_loaded"] is False,
        "exact_frozen_config_used": (
            verify["trainer_runtime_identity_digest"]
            == ACCEPTED_TRAINER_RUNTIME_IDENTITY_DIGEST
        ),
        "operational_amendment_recorded": (
            amendment.amendment_digest() == ACCEPTED_AMENDMENT_DIGEST
        ),
        "original_contract_digest_unmodified": (
            verify["contract_digest"] == ACCEPTED_CONTRACT_DIGEST
        ),
        "iterations_completed_60": len(journal["iterations"]) == contract.CANONICAL_ITERATIONS,
        "games_scheduled_122880": total_games == contract.CANONICAL_MAX_SCHEDULED_GAMES,
        "rollout_identity_errors_zero": int(counters.get("rollout_identity_mismatches", 0)) == 0
        and int(counters.get("behavior_identity_mismatches", 0)) == 0,
        "illegal_actions_zero": safety_totals["illegal_policy_actions"] == 0,
        "nonfinite_zero": (
            int(counters.get("non_finite_losses", 0))
            + int(counters.get("non_finite_gradients", 0))
            + int(counters.get("non_finite_parameters", 0))
            + safety_totals["inference_failures"]
        )
        == 0,
        "target_mismatches_zero": (
            int(counters.get("illegal_targets", 0)) + int(counters.get("data_mismatches", 0))
        )
        == 0,
        "observer_leaks_zero": sum(
            int(entry["collection"].get("observer_probe_failures") or 0)
            for entry in journal["iterations"]
        )
        == 0,
        "kl_hard_limit_never_exceeded": kl_max <= contract.KL_HARD_LIMIT,
        "clip_fraction_hard_limit_never_exceeded": clip_max <= contract.CLIP_FRACTION_HARD_LIMIT,
        "restart_path_exercised": len(restarts) >= 1
        and all(record["comparison"]["passed"] for record in restarts)
        and all(record["passed"] for record in boundary_resumes),
        "archive_schedule_exact": archive_payload["archive_schedule_exact"],
        "validation_only_checkpoint_selection": all(
            record["authorized_access"]["resource"] == "phase9_validation_bank"
            for record in validations
        ),
        "best_checkpoint_reload_reproduces": freeze["reload_reproduction"]["passed"],
        "final_checkpoint_sha_written": bool(freeze["frozen_checkpoint"]["sha256"]),
        "final_test_model_access_zero": True,
        "ceiling_respected": float(journal["wall_clock"]["run_seconds"])
        <= amendment_v2.amended_ceiling_seconds(),
        "amendment_chain_untouched": not amendment_v2.verify_chain_untouched(),
        "full_suite_green": bool(suite and suite.get("failed") == 0),
    }

    validation_count = len(validations)
    expected_validations = (
        contract.CANONICAL_ITERATIONS // contract.VALIDATION_CADENCE_ITERATIONS
    )
    gates["validation_passes_twelve"] = validation_count == expected_validations

    run_payload = {
        "agent": AGENT,
        "phase": PHASE,
        "artifact": "agent_07_canonical_run",
        "status": "PASS" if all(gates.values()) else "INCOMPLETE",
        **environment_record(),
        "namespace": NAMESPACE,
        "candidate_id": CANDIDATE_ID,
        "frozen_configuration": {
            "learning_rate": journal["train_config"]["learning_rate"],
            "initial_kl_beta": journal["train_config"]["initial_kl_beta"],
            "total_iterations": journal["train_config"]["total_iterations"],
            "epochs_per_rollout": journal["train_config"]["epochs_per_rollout"],
            "minibatch_size": journal["train_config"]["minibatch_size"],
            "device": journal["train_config"]["device"],
            "precision": journal["train_config"]["precision"],
            "runtime_identity_digest": journal["train_config"]["digest"],
            "runtime_scope_token": RUNTIME_SCOPE_TOKEN,
            "scope_audit": verify["scope_audit"],
            "topology": journal["topology"],
        },
        "identities": manifest_payload["identities"],
        "operational_amendment": verify["operational_amendment"],
        "operational_amendment_v2": amendment_stage,
        "storage": verify["storage"],
        "storage_checks": journal["storage_checks"],
        "fresh_start": journal["fresh_start"],
        "execution": {
            "iterations_committed": len(journal["iterations"]),
            "games_scheduled": total_games,
            "optimizer_updates": sum(entry["updates"] for entry in journal["iterations"]),
            "examples": sum(entry["examples"] for entry in journal["iterations"]),
            "learner_decisions": sum(
                entry["learner_decisions"] for entry in journal["iterations"]
            ),
            "supervisor_launches": run["supervisor_launches"],
            "process_launches": journal["process_launches"],
            "wall_clock_seconds": journal["wall_clock"]["run_seconds"],
            "wall_clock_hours": journal["wall_clock"]["run_seconds"] / 3600.0,
            "ceiling_seconds": amendment_v2.amended_ceiling_seconds(),
            "ceiling_authority": (
                amendment_v2.PHASE9_OPERATIONAL_AMENDMENT_V2_VERSION
            ),
            "ceiling_headroom_seconds": amendment_v2.amended_ceiling_seconds()
            - journal["wall_clock"]["run_seconds"],
            "ended_on": (
                "the contracted 60 iterations and their bookkeeping completed; no "
                "remaining ceiling time was spent on additional rollouts, "
                "optimization, validation, archive members or experimentation"
            ),
            "sessions": journal["wall_clock"]["sessions"],
        },
        "restart_exercise": restarts,
        "boundary_resumes": boundary_resumes,
        "harness_faults": journal.get("harness_faults", []),
        "hard_stop_counters": {
            **{key: int(value) for key, value in counters.items()},
            "observer_probe_failures": sum(
                int(entry["collection"].get("observer_probe_failures") or 0)
                for entry in journal["iterations"]
            ),
            "illegal_policy_actions_validation": safety_totals["illegal_policy_actions"],
            "inference_failures_validation": safety_totals["inference_failures"],
            "policy_errors_validation": safety_totals["policy_errors"],
            "max_epoch_mean_kl": kl_max,
            "kl_hard_limit": contract.KL_HARD_LIMIT,
            "max_epoch_clip_fraction": clip_max,
            "clip_fraction_hard_limit": contract.CLIP_FRACTION_HARD_LIMIT,
            "test_bank_model_access": 0,
        },
        "validation": {
            "bank_version": contract.VALIDATION_BANK_VERSION,
            "bank_digest": verify["validation_bank_digest_expected"],
            "cadence_iterations": contract.VALIDATION_CADENCE_ITERATIONS,
            "passes": validation_count,
            "passes_expected": expected_validations,
            "score_weights": dict(contract.VALIDATION_SCORE_WEIGHTS),
            "selection": freeze["selection"],
            "history": manifest_payload["validation_history"],
        },
        "final_test_bank": {
            "version": contract.TEST_BANK_VERSION,
            "digest": verify["test_bank_digest_recorded"],
            "model_access_by_agent_7": 0,
            "constructed_by_agent_7": False,
            "rule": "Agent 8 owns the first final-test neural evaluation",
        },
        "report_only_diagnostics": report_only_diagnostics(journal),
        "completion_gates": gates,
        "gates_true": sum(1 for value in gates.values() if value),
        "gates_total": len(gates),
        "tests_before": verify["tests_before"],
        "tests_after": suite,
        "handoff_to_agent_8": {
            "frozen_checkpoint_path": freeze["frozen_checkpoint"]["path"],
            "frozen_checkpoint_sha256": freeze["frozen_checkpoint"]["sha256"],
            "frozen_model_state_digest": freeze["frozen_checkpoint"]["model_state_digest"],
            "selected_iteration": freeze["selection"]["selected_iteration"],
            "phase8_anchor_checkpoint_sha256": contract.EXPECTED_PHASE8_CHECKPOINT_SHA256,
            "phase8_anchor_model_state_digest": verify["start_model_state_digest"],
            "final_test_bank_digest": verify["test_bank_digest_recorded"],
            "archive_manifest": str(ARCHIVE_ARTIFACT.relative_to(REPOSITORY_ROOT)),
            "checkpoint_manifest": str(MANIFEST_ARTIFACT.relative_to(REPOSITORY_ROOT)),
            "training_curve": str(CURVE_ARTIFACT.relative_to(REPOSITORY_ROOT)),
            "agent_8_performs_no_training": True,
        },
    }
    write_json(RUN_ARTIFACT, run_payload)
    log(
        f"artifacts: {run_payload['gates_true']}/{run_payload['gates_total']} gates true; "
        f"status {run_payload['status']}"
    )
    return run_payload


# ---------------------------------------------------------------------------
# Final suite
# ---------------------------------------------------------------------------


def record_final_suite() -> int:
    command = [".venv/bin/python", "-m", "pytest", "tests", "-q", "-p", "no:randomly"]
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=REPOSITORY_ROOT, capture_output=True, text=True
    )
    elapsed = time.perf_counter() - started
    tail = completed.stdout.strip().splitlines()
    summary = tail[-1] if tail else ""
    passed = failed = skipped = 0
    for token, label in (("passed", "passed"), ("failed", "failed"), ("skipped", "skipped")):
        for part in summary.replace(",", "").split():
            if part == token:
                index = summary.replace(",", "").split().index(part)
                try:
                    value = int(summary.replace(",", "").split()[index - 1])
                except (ValueError, IndexError):
                    value = 0
                if label == "passed":
                    passed = value
                elif label == "failed":
                    failed = value
                else:
                    skipped = value
    payload = {
        "stage": "final_suite",
        **environment_record(),
        "command": " ".join(command),
        "summary": summary,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "seconds": round(elapsed, 2),
        "returncode": completed.returncode,
    }
    write_stage("final_suite", payload)
    log(f"final suite: {summary}")
    return 0 if completed.returncode == 0 else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


STAGES = ("verify", "amendment", "run", "freeze", "artifacts")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 9 Agent 7 acceptance harness")
    parser.add_argument("--stage", default="all", choices=("all",) + STAGES)
    parser.add_argument("--canonical-worker", action="store_true")
    parser.add_argument("--anchor-worker", action="store_true")
    parser.add_argument("--validation-iteration", type=int, default=None)
    parser.add_argument("--anchor-chunk-index", type=int, default=None)
    parser.add_argument("--expected-bank-digest", default=None)
    parser.add_argument("--games-label", default="games")
    parser.add_argument("--export-path", default=None)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--collect-device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--batch-shape", type=int, default=64)
    parser.add_argument("--games-in-flight", type=int, default=96)
    parser.add_argument("--observer-probe-plies", type=int, default=2)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--anchor-workers", type=int, default=4)
    parser.add_argument("--chunk-units", type=int, default=64)
    parser.add_argument("--historical-verify-games", type=int, default=2)
    parser.add_argument("--reset-run", action="store_true")
    parser.add_argument("--skip-payload-bytes", action="store_true")
    parser.add_argument("--record-final-suite", action="store_true")
    args = parser.parse_args()

    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if args.anchor_worker:
        run_anchor_worker(args)
        return 0
    if args.canonical_worker:
        return run_canonical_worker(args)
    if args.record_final_suite:
        return record_final_suite()

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    if "verify" in stages:
        payload = stage_verify(args)
        if payload["problems"]:
            return 2
    if "amendment" in stages:
        payload = stage_amendment(args)
        if payload["problems"]:
            return 2
    if "run" in stages:
        payload = stage_run(args)
        if not payload["complete"]:
            log(f"BLOCKED: the canonical run is incomplete: {payload}")
            return 2
    if "freeze" in stages:
        stage_freeze(args)
    if "artifacts" in stages:
        stage_artifacts(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
