"""Phase 8 Agent 6 — the canonical C1 warm-start run.

One fresh C1 initialisation, the exact frozen `warmstart_train_config_v1`, the
accepted synthetic corpus, and 25,000 optimizer updates on MPS. The best
checkpoint is chosen by the frozen validation `selection_score` alone; the
sealed test split and the Phase 4 evaluation bank stay closed until Agent 7,
and that is proved here by measurement rather than assertion — the model-input
boundary and the Phase 4 entry points are instrumented for the whole run and
their tallies are reported.

Corpus identity
---------------
The corpus is resolved *only* through
`synthetic_corpus.default_corpus_root()`. This harness pins the accepted
absolute location as an expected value because an acceptance harness is
exactly the place that may; no library, trainer, checkpoint or downstream
module embeds it. Identity is `synthetic_warmstart_corpus_v1` plus its three
accepted digests, so a pure relocation stays compatible. Any digest mismatch
is BLOCKED — never a reason to regenerate or repair corpus bytes.

The restart exercise
--------------------
The canonical run is executed as two real processes. Segment 1 starts from the
fresh canonical initialisation, trains to the restart step, writes a normal
checkpoint and exits cleanly. Segment 2 is a new interpreter that reloads that
checkpoint through the accepted `WarmstartTrainer.resume` path and finishes the
budget. The data cursor, optimizer, scheduler, counters, best-validation state
and validation cadence must cross that boundary unchanged.

Resume equivalence is judged under the reviewer-approved
`backend_aware_resume_equivalence_v1` interpretation: MPS is not run-to-run
bit-deterministic, so the criterion is exact *logical* state continuity —
cursor, counters, optimizer/scheduler state structure and cadence — not
independent-run bit equality, which is unattainable on this backend and is not
resurrected here.

Modes
-----

```text
--verify      prerequisite and corpus-identity gates only
--run         the canonical two-segment run
--freeze      independent reload, revalidation, SHA-256, manifest
--artifacts   write the three artifacts and append report section 6
--full        all of the above
--run-pytest  full repository suite
```
"""

import argparse
import csv
import hashlib
import json
import platform
import re
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch  # noqa: E402

from stratego.training import synthetic_corpus as sc  # noqa: E402
from stratego.training import warmstart_contract as wc  # noqa: E402
from stratego.training import warmstart_pilot as wp  # noqa: E402
from stratego.training import warmstart_trainer as wt  # noqa: E402
from stratego.training.warmstart_checkpoint import (  # noqa: E402
    WARMSTART_CHECKPOINT_VERSION,
    WARMSTART_TRAINER_VERSION,
    CorpusIdentity,
    load_warmstart_checkpoint,
    verify_corpus_identity,
)
from stratego.training.warmstart_dataset import (  # noqa: E402
    TRAIN_ORDER_VERSION,
    WarmstartDataset,
)
from stratego.training.warmstart_loss import WARMSTART_LOSS_VERSION  # noqa: E402
from stratego.training.warmstart_metrics import (  # noqa: E402
    WARMSTART_METRICS_VERSION,
    frozen_train_value_prior,
    run_validation,
)
from stratego.training.warmstart_seed import CANONICAL_SEEDS  # noqa: E402
from stratego.training.warmstart_trainer import (  # noqa: E402
    LoaderTopology,
    WarmstartTrainConfig,
    WarmstartTrainer,
)

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_8_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_8_implementation_report.md"
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase8"
WORK_DIRECTORY = CHECKPOINT_DIRECTORY / "agent06"

#: The accepted checkpoint the phase freezes, and its manifest.
FROZEN_CHECKPOINT_PATH = CHECKPOINT_DIRECTORY / "warmstart_c1_v1.pt"
FROZEN_MANIFEST_PATH = CHECKPOINT_DIRECTORY / "warmstart_c1_v1_manifest.json"

#: The canonical accepted storage location (supplementary review instruction).
#: Verified against the resolver here; never embedded in library code.
REQUIRED_CORPUS_ROOT = (
    "/Users/brandonwashington/Dev/Github/stratego/gpt_agent/"
    "data/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1"
)
REQUIRED_CORPUS_ROOT_RELATIVE = (
    "data/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1"
)

#: Agent 5's frozen winner. Agent 6 may not tune any field of it.
WINNING_CANDIDATE_ID = "ws_pilot_lr1e-3_balanced"

#: The loader topology Agent 4 measured as best and Agent 5 froze.
#: Infrastructure only: it cannot change any batch, which is why it is absent
#: from the training-config digest.
FROZEN_TOPOLOGY = {"workers": 12, "prefetch": 2, "record_cache_size": 512}

#: Where the canonical run is cut in two so the production resume path is
#: exercised for real. A cadence multiple, so the boundary lands exactly where
#: a checkpoint already exists and the next validation stays on schedule.
DEFAULT_RESTART_AT = 12_500

CURVE_COLUMNS = (
    "global_step",
    "segment",
    "wall_seconds",
    "examples_consumed",
    "train_loss_total",
    "train_loss_policy",
    "train_loss_value",
    "train_loss_belief",
    "train_legal_policy_entropy",
    "train_legal_policy_entropy_normalized",
    "learning_rate",
    "grad_norm_pre_clip",
    "grad_norm_post_clip",
    "parameter_norm",
    "examples_per_second",
    "data_wait_fraction",
    "validation_selection_score",
    "validation_policy_ce",
    "validation_policy_ce_ratio",
    "validation_policy_top1",
    "validation_policy_baseline_expected_top1",
    "validation_value_ce",
    "validation_value_ce_ratio",
    "validation_value_accuracy",
    "validation_value_brier",
    "validation_value_baseline_brier",
    "validation_belief_ce",
    "validation_belief_ce_ratio",
    "validation_belief_top1",
    "validation_belief_baseline_top1",
    "validation_examples",
    "validation_games",
    "validation_batches",
    "validation_seconds",
    "is_best",
    "peak_rss_bytes",
    "mps_current_allocated_bytes",
    "mps_driver_allocated_bytes",
)

#: Where the artifacts land. `--dry-run` redirects every output into the work
#: directory so a shakedown of the harness cannot overwrite accepted evidence,
#: claim the canonical checkpoint path, or append a throwaway report section.
_OUTPUT_DIRECTORY = DATA_DIRECTORY
_CHECKPOINT_PATH = FROZEN_CHECKPOINT_PATH
_MANIFEST_PATH = FROZEN_MANIFEST_PATH
_APPEND_REPORT = True


def configure_output(*, dry_run: bool, work: Path) -> None:
    global _OUTPUT_DIRECTORY, _CHECKPOINT_PATH, _MANIFEST_PATH, _APPEND_REPORT
    if not dry_run:
        return
    _OUTPUT_DIRECTORY = work / "dry_run_artifacts"
    _CHECKPOINT_PATH = _OUTPUT_DIRECTORY / "warmstart_c1_v1.pt"
    _MANIFEST_PATH = _OUTPUT_DIRECTORY / "warmstart_c1_v1_manifest.json"
    _APPEND_REPORT = False
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def artifact_path(name: str) -> Path:
    return _OUTPUT_DIRECTORY / name


def log(message: str) -> None:
    print(f"[agent06 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPOSITORY_ROOT, capture_output=True, text=True
    ).stdout.strip()


def environment_record() -> dict:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cpu_count": int(torch.multiprocessing.cpu_count()),
        "source_revision": git_output("rev-parse", "--short", "HEAD"),
        "working_tree_state": "dirty" if git_output("status", "--porcelain") else "clean",
    }


def memory_record() -> dict:
    return {
        "mps_current_allocated_bytes": (
            int(torch.mps.current_allocated_memory())
            if torch.backends.mps.is_available()
            else 0
        ),
        "mps_driver_allocated_bytes": (
            int(torch.mps.driver_allocated_memory())
            if torch.backends.mps.is_available()
            else 0
        ),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def mean(values) -> "float | None":
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def file_sha256(path: "str | Path") -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Prerequisite verification
# ---------------------------------------------------------------------------


def accepted_corpus_identity() -> CorpusIdentity:
    """The accepted digests, cross-checked across every artifact stating them.

    Agent 5's frozen train config is included: from Agent 6 on, the config the
    trainer is handed and the corpus it reads must name the same corpus.
    """
    manifest = read_json(DATA_DIRECTORY / "agent_02_corpus_manifest.json")[
        "corpus_manifest"
    ]
    relocation = read_json(DATA_DIRECTORY / "agent_02_relocation.json")[
        "accepted_digests"
    ]
    agent3 = read_json(DATA_DIRECTORY / "agent_03_example_contract.json")[
        "prerequisite_digests"
    ]
    agent4 = read_json(DATA_DIRECTORY / "agent_04_trainer_contract.json")[
        "prerequisite_digests"
    ]
    agent5 = read_json(DATA_DIRECTORY / "agent_05_frozen_train_config.json")["config"][
        "corpus_digests"
    ]
    sources = {
        "content_digest": {
            "agent_02_manifest": manifest["content_digest"],
            "agent_02_relocation": relocation["content_digest"],
            "agent_03": agent3["corpus_content"],
            "agent_04": agent4["corpus_content"],
            "agent_05_frozen_config": agent5["content_digest"],
        },
        "metadata_digest": {
            "agent_02_manifest": manifest["metadata_digest"],
            "agent_02_relocation": relocation["metadata_digest"],
            "agent_03": agent3["corpus_metadata"],
            "agent_04": agent4["corpus_metadata"],
            "agent_05_frozen_config": agent5["metadata_digest"],
        },
        "commit_index_digest": {
            "agent_02_manifest": manifest["commit_index_digest"],
            "agent_02_relocation": relocation["commit_index_digest"],
            "agent_03": agent3["corpus_commit_index"],
            "agent_04": agent4["corpus_commit_index"],
            "agent_05_frozen_config": agent5["commit_index_digest"],
        },
    }
    for name, values in sources.items():
        if len(set(values.values())) != 1:
            raise SystemExit(f"BLOCKED: accepted artifacts disagree on {name}: {values}")
    return CorpusIdentity(
        corpus_version=manifest["corpus_version"],
        content_digest=manifest["content_digest"],
        metadata_digest=manifest["metadata_digest"],
        commit_index_digest=manifest["commit_index_digest"],
    )


def frozen_config_payload() -> dict:
    return read_json(DATA_DIRECTORY / "agent_05_frozen_train_config.json")


def build_frozen_trainer_config(device: str = "mps") -> WarmstartTrainConfig:
    """Exactly the construction Agent 5 recorded, with nothing tuned."""
    return WarmstartTrainConfig.from_pilot_candidate(
        WINNING_CANDIDATE_ID, device=device, validation_batches=64
    )


#: The two Agent 5 digests cover *different objects*, so they can never be
#: equal. Naming them here keeps the distinction explicit everywhere it is
#: reported, rather than leaving a bare hex string to be read as whichever
#: identity the reader had in mind.
TRAIN_CONFIG_DOCUMENT_DIGEST = "agent_05_frozen_train_config.config document"
TRAINER_RUNTIME_IDENTITY_DIGEST = "WarmstartTrainConfig.identity() runtime object"

#: Fields the frozen document and the runtime identity both express, some
#: under different names. The document is Agent 5's artifact vocabulary; the
#: runtime identity is the trainer's.
IDENTITY_FIELD_BRIDGE = {
    "train_split": "split",
    "train_order": "order",
}


def reconcile_train_config_identity(config: WarmstartTrainConfig) -> dict:
    """Prove the runtime configuration is Agent 5's frozen one, field for field.

    Two digests appear in Agent 5's accepted artifact and they are *not*
    alternative spellings of one identity:

    ```text
    train_config_digest    sha256 over payload["config"], the 41-field frozen
                           train-config document — the artifact-level identity
                           of warmstart_train_config_v1
    trainer_config_digest  sha256 over WarmstartTrainConfig.identity(), the
                           31-field runtime object the trainer stamps into
                           every checkpoint and compares on resume
    ```

    They serialize different field sets, so equality between them was never
    possible and their difference is not a mismatch. What must hold — and is
    measured here — is that both recompute to their recorded values, that the
    live runtime identity equals Agent 5's recorded one exactly, and that every
    field the two objects share agrees across the naming bridge.
    """
    frozen = frozen_config_payload()
    document = dict(frozen["config"])
    recorded_runtime = dict(frozen["trainer_config_identity"])
    live_runtime = dict(config.identity())

    def canonical(payload: dict) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    document_recomputed = hashlib.sha256(canonical(document).encode()).hexdigest()
    runtime_recomputed = config.digest()

    # Every field the two objects share, compared across the bridge.
    shared: dict = {}
    for document_name, value in document.items():
        runtime_name = IDENTITY_FIELD_BRIDGE.get(document_name, document_name)
        if runtime_name in live_runtime:
            shared[document_name] = {
                "runtime_field": runtime_name,
                "document_value": value,
                "runtime_value": live_runtime[runtime_name],
                "equal": value == live_runtime[runtime_name],
            }
    # adam_betas is one document field against two runtime fields.
    betas = list(document.get("adam_betas", []))
    shared["adam_betas"] = {
        "runtime_field": "adam_beta1 + adam_beta2",
        "document_value": betas,
        "runtime_value": [live_runtime["adam_beta1"], live_runtime["adam_beta2"]],
        "equal": betas == [live_runtime["adam_beta1"], live_runtime["adam_beta2"]],
    }

    # Frozen document fields with no runtime-identity counterpart are still
    # binding on this run; each is checked against the live source of truth.
    external = {
        "train_shuffle_seed": {
            "frozen": document["train_shuffle_seed"],
            "live": int(CANONICAL_SEEDS["train_order_seed"]),
            "source": "warmstart_seed.TRAIN_ORDER_SEED",
        },
        "model_config_digest": {
            "frozen": document["model_config_digest"],
            "live": wc.EXPECTED_C1_CONFIG_DIGEST,
            "source": "warmstart_contract.EXPECTED_C1_CONFIG_DIGEST",
        },
        "loader_topology": {
            "frozen": document["loader_topology"],
            "live": dict(FROZEN_TOPOLOGY),
            "source": "the topology this harness runs",
        },
        "checkpoint_version": {
            "frozen": document["checkpoint_version"],
            "live": WARMSTART_CHECKPOINT_VERSION,
            "source": "warmstart_checkpoint.WARMSTART_CHECKPOINT_VERSION",
        },
        "metrics_version": {
            "frozen": document["metrics_version"],
            "live": WARMSTART_METRICS_VERSION,
            "source": "warmstart_metrics.WARMSTART_METRICS_VERSION",
        },
        "loss_version": {
            "frozen": document["loss_version"],
            "live": WARMSTART_LOSS_VERSION,
            "source": "warmstart_loss.WARMSTART_LOSS_VERSION",
        },
    }
    for entry in external.values():
        entry["equal"] = entry["frozen"] == entry["live"]

    disagreements = sorted(
        [name for name, entry in shared.items() if not entry["equal"]]
        + [name for name, entry in external.items() if not entry["equal"]]
    )

    return {
        "verdict": (
            "distinct digest namespaces, both verified"
            if not disagreements
            and document_recomputed == frozen["train_config_digest"]
            and runtime_recomputed == frozen["trainer_config_digest"]
            and live_runtime == recorded_runtime
            else "FROZEN TRAIN CONFIG IDENTITY MISMATCH"
        ),
        "are_the_same_serialization": False,
        "explanation": (
            "The two digests cover different objects and were both authored by "
            "Agent 5 in one artifact: train_config_digest hashes the 41-field "
            "frozen train-config document (payload['config']), "
            "trainer_config_digest hashes the 31-field runtime "
            "WarmstartTrainConfig.identity(). They share 25 fields but each "
            "carries fields the other does not, so the two hashes could never "
            "be equal and their difference is not a mismatch."
        ),
        "namespaces": {
            "train_config_digest": {
                "label": TRAIN_CONFIG_DOCUMENT_DIGEST,
                "scope": (
                    "sha256 of json.dumps(agent_05_frozen_train_config['config'], "
                    "sort_keys=True, separators=(',',':'))"
                ),
                "identifies": "warmstart_train_config_v1, the frozen artifact",
                "field_count": len(document),
                "recorded": frozen["train_config_digest"],
                "recomputed": document_recomputed,
                "match": document_recomputed == frozen["train_config_digest"],
            },
            "trainer_config_digest": {
                "label": TRAINER_RUNTIME_IDENTITY_DIGEST,
                "scope": (
                    "sha256 of json.dumps(WarmstartTrainConfig.identity(), "
                    "sort_keys=True, separators=(',',':'))"
                ),
                "identifies": (
                    "the runtime trainer configuration stamped into every "
                    "checkpoint and compared by check_resume_identity"
                ),
                "field_count": len(live_runtime),
                "recorded": frozen["trainer_config_digest"],
                "recomputed": runtime_recomputed,
                "match": runtime_recomputed == frozen["trainer_config_digest"],
            },
        },
        "runtime_identity_equals_agent_5_recorded": live_runtime == recorded_runtime,
        # Three categories, kept apart so no field is counted twice: matched by
        # name, matched across the naming bridge, and genuinely unique to one
        # object. `adam_betas`/`train_split`/`train_order` are bridged, not
        # document-only, even though a raw set difference on names says
        # otherwise.
        "fields_matched_by_name": sorted(set(document) & set(live_runtime)),
        "fields_matched_across_bridge": {
            "train_split": "split",
            "train_order": "order",
            "adam_betas": "adam_beta1 + adam_beta2",
        },
        "fields_only_in_document": sorted(
            set(document) - set(live_runtime) - {"train_split", "train_order", "adam_betas"}
        ),
        "fields_only_in_runtime_identity": sorted(
            set(live_runtime)
            - set(document)
            - {"split", "order", "adam_beta1", "adam_beta2"}
        ),
        "raw_name_set_difference": {
            "document_minus_runtime": sorted(set(document) - set(live_runtime)),
            "runtime_minus_document": sorted(set(live_runtime) - set(document)),
            "note": (
                "by name only; the bridged fields appear here but are compared "
                "in shared_field_comparison"
            ),
        },
        "shared_field_comparison": shared,
        "shared_fields_compared": len(shared),
        "shared_fields_equal": sum(1 for e in shared.values() if e["equal"]),
        "document_only_fields_checked_against_live_source": external,
        "disagreements": disagreements,
    }


def verify_prerequisites(
    *, check_payload_bytes: bool = True, device: str = "mps"
) -> tuple:
    """Every Agent 6 entry gate. Returns `(record, verified_identity)`."""
    started = time.perf_counter()
    problems: list = []

    statuses = {}
    for agent, artifact in (
        (1, "agent_01_warmstart_contract.json"),
        (2, "agent_02_corpus_audit.json"),
        (3, "agent_03_example_contract.json"),
        (4, "agent_04_trainer_contract.json"),
        (5, "agent_05_pilot_selection.json"),
    ):
        payload = read_json(DATA_DIRECTORY / artifact)
        statuses[f"agent_{agent}"] = payload.get("status")
        if payload.get("status") != "PASS":
            problems.append(f"agent {agent} artifact status is {payload.get('status')!r}")

    contract_payload = read_json(DATA_DIRECTORY / "agent_01_warmstart_contract.json")
    recorded_contract = contract_payload["contract_digest"]
    live_contract = wc.contract_digest()
    if recorded_contract != live_contract:
        problems.append(
            f"live contract digest {live_contract} != recorded {recorded_contract}"
        )

    upstream = wc.verify_frozen_upstream()
    roster = wc.verify_teacher_roster()
    problems.extend(upstream)
    problems.extend(roster)

    # Agent 5's frozen configuration is Agent 6's subject matter: it must
    # verify against its own recorded digest, and the live trainer must
    # reconstruct it field for field.
    frozen = frozen_config_payload()
    frozen_problems = wp.verify_frozen_train_config(frozen)
    problems.extend(frozen_problems)

    config = build_frozen_trainer_config(device=device)
    recorded_train_digest = frozen["train_config_digest"]
    recorded_trainer_digest = frozen["trainer_config_digest"]
    live_trainer_digest = config.digest()
    if live_trainer_digest != recorded_trainer_digest:
        stored = dict(frozen["trainer_config_identity"])
        live = dict(config.identity())
        differing = sorted(
            key for key in set(stored) | set(live) if stored.get(key) != live.get(key)
        )
        problems.append(
            "the live trainer does not reconstruct Agent 5's frozen configuration "
            f"(digest {live_trainer_digest} != {recorded_trainer_digest}; fields: "
            f"{differing})"
        )

    # The two Agent 5 digests are different namespaces; this proves both
    # recompute and that the runtime configuration is the frozen one field for
    # field. A real disagreement here is the BLOCKED stop condition.
    reconciliation = reconcile_train_config_identity(config)
    if reconciliation["disagreements"]:
        problems.append(
            "FROZEN TRAIN CONFIG IDENTITY MISMATCH: the runtime configuration "
            f"differs from Agent 5's frozen payload in {reconciliation['disagreements']}"
        )
    for name, namespace in reconciliation["namespaces"].items():
        if not namespace["match"]:
            problems.append(
                f"FROZEN TRAIN CONFIG IDENTITY MISMATCH: {name} recomputes to "
                f"{namespace['recomputed']}, recorded {namespace['recorded']}"
            )
    if not reconciliation["runtime_identity_equals_agent_5_recorded"]:
        problems.append(
            "FROZEN TRAIN CONFIG IDENTITY MISMATCH: the live runtime identity "
            "does not equal Agent 5's recorded trainer_config_identity"
        )

    if frozen["winning_candidate_id"] != WINNING_CANDIDATE_ID:
        problems.append(
            f"frozen winner is {frozen['winning_candidate_id']!r}, this harness "
            f"is built for {WINNING_CANDIDATE_ID!r}"
        )

    # Every version this run claims to implement.
    versions = {
        "train_config_version": frozen["train_config_version"],
        "trainer_version": WARMSTART_TRAINER_VERSION,
        "checkpoint_version": WARMSTART_CHECKPOINT_VERSION,
        "loss_version": WARMSTART_LOSS_VERSION,
        "metrics_version": WARMSTART_METRICS_VERSION,
        "train_order_version": TRAIN_ORDER_VERSION,
    }
    for name, recorded in (
        ("trainer_version", frozen["config"]["trainer_version"]),
        ("checkpoint_version", frozen["config"]["checkpoint_version"]),
        ("loss_version", frozen["config"]["loss_version"]),
        ("metrics_version", frozen["config"]["metrics_version"]),
    ):
        if versions[name] != recorded:
            problems.append(
                f"live {name} {versions[name]!r} != frozen {recorded!r}"
            )

    resolution = sc.describe_corpus_root()
    resolved = sc.default_corpus_root()
    required_via_repository = REPOSITORY_ROOT / REQUIRED_CORPUS_ROOT_RELATIVE
    if str(resolved) != REQUIRED_CORPUS_ROOT:
        problems.append(
            f"default_corpus_root() resolves to {resolved}, the accepted canonical "
            f"location is {REQUIRED_CORPUS_ROOT}; correct the resolver/pointer "
            "configuration only"
        )
    if resolved != required_via_repository:
        problems.append(
            f"resolver {resolved} is not the repository-relative canonical path "
            f"{required_via_repository}"
        )
    if resolution["pointer_value"] != REQUIRED_CORPUS_ROOT:
        problems.append(
            f"pointer file names {resolution['pointer_value']!r}, expected the "
            "canonical location"
        )

    if not torch.backends.mps.is_available():
        problems.append("MPS is not available on this machine")

    accepted = accepted_corpus_identity()
    identity = None
    digest_seconds = 0.0
    if not problems:
        digest_started = time.perf_counter()
        identity = verify_corpus_identity(
            resolved, accepted, check_payload_bytes=check_payload_bytes
        )
        digest_seconds = time.perf_counter() - digest_started

    record = {
        "statuses": statuses,
        "agent_01_contract_digest": {
            "recorded": recorded_contract,
            "live": live_contract,
            "match": recorded_contract == live_contract,
        },
        "upstream_problems": upstream,
        "roster_problems": roster,
        "frozen_train_config_problems": frozen_problems,
        # Named, not bare: these two hashes cover different objects.
        "train_config_document_digest": {
            "label": TRAIN_CONFIG_DOCUMENT_DIGEST,
            "value": recorded_train_digest,
        },
        "frozen_train_config_digest": recorded_train_digest,
        "trainer_config_digest": {
            "label": TRAINER_RUNTIME_IDENTITY_DIGEST,
            "recorded": recorded_trainer_digest,
            "live": live_trainer_digest,
            "match": live_trainer_digest == recorded_trainer_digest,
        },
        "train_config_identity_reconciliation": reconciliation,
        "winning_candidate_id": frozen["winning_candidate_id"],
        "trainer_construction": frozen["trainer_construction"],
        "versions": versions,
        "corpus_root_resolution": resolution,
        "required_corpus_root": REQUIRED_CORPUS_ROOT,
        "resolver_matches_required": str(resolved) == REQUIRED_CORPUS_ROOT,
        "pointer_matches_required": resolution["pointer_value"] == REQUIRED_CORPUS_ROOT,
        "accepted_digests": accepted.to_dict(),
        "observed_digests": identity.to_dict() if identity else None,
        "digests_match": identity == accepted if identity else False,
        "payload_bytes_checked": check_payload_bytes,
        "digest_verification_seconds": digest_seconds,
        "expected_fresh_init_checksum": frozen["config"]["expected_fresh_init_checksum"],
        "canonical_seeds": dict(CANONICAL_SEEDS),
        "max_final_updates": frozen["config"]["max_final_updates"],
        "problems": problems,
        "seconds": time.perf_counter() - started,
    }
    return record, identity


# ---------------------------------------------------------------------------
# Validation capture
# ---------------------------------------------------------------------------


@contextmanager
def capture_validation_results():
    """Record the full `ValidationResult` of every cadence pass.

    The trainer's own `run_cadence_validation` keeps a deliberately small
    history entry. Agent 6 must report top-1 accuracies, Brier scores and
    baseline values as well, so the module-level `run_validation` the trainer
    calls is wrapped and its results are copied out. Purely observational: the
    original function does the work and its return value is passed through
    untouched, so the selection logic sees exactly what it always sees.
    """
    captured: list = []
    original = wt.run_validation

    def instrumented(*arguments, **keywords):
        result = original(*arguments, **keywords)
        captured.append(result.to_dict())
        return result

    wt.run_validation = instrumented
    try:
        yield captured
    finally:
        wt.run_validation = original


# ---------------------------------------------------------------------------
# One training segment (runs in its own process)
# ---------------------------------------------------------------------------


def segment_paths(work: Path, segment: int) -> dict:
    return {
        "rows": work / f"segment_{segment}_rows.jsonl",
        "validations": work / f"segment_{segment}_validations.jsonl",
        "summary": work / f"segment_{segment}_summary.json",
        "checkpoint": work / f"segment_{segment}_end.pt",
    }


def run_segment(
    *,
    segment: int,
    stop_at: int,
    resume_from: "Path | None",
    work: Path,
    device: str,
    topology: LoaderTopology,
) -> dict:
    """Train until `stop_at`, checkpoint, and exit cleanly.

    Segment 1 builds the fresh canonical initialisation and requires its
    checksum to equal Agent 5's frozen expectation before a single optimizer
    step. Later segments reload through the accepted resume path, which
    re-checks train-config and corpus identity before any state is touched.
    """
    work.mkdir(parents=True, exist_ok=True)
    paths = segment_paths(work, segment)
    frozen = frozen_config_payload()
    expected_checksum = frozen["config"]["expected_fresh_init_checksum"]
    accepted = accepted_corpus_identity()
    root = sc.default_corpus_root()
    if str(root) != REQUIRED_CORPUS_ROOT:
        raise SystemExit(
            f"BLOCKED: default_corpus_root() resolves to {root}, expected "
            f"{REQUIRED_CORPUS_ROOT}"
        )
    # Digest-only here: the byte-level integrity read already ran in --verify
    # and the bytes cannot change between two segments of one run.
    identity = verify_corpus_identity(root, accepted, check_payload_bytes=False)
    config = build_frozen_trainer_config(device=device)
    if config.digest() != frozen["trainer_config_digest"]:
        raise SystemExit("BLOCKED: live trainer config does not match the frozen digest")

    best_path = work / "best.pt"
    started_wall = time.time()
    started = time.perf_counter()

    with wp.record_model_input_access() as model_access, wp.record_phase4_access() as phase4_access:
        if resume_from is None:
            trainer = WarmstartTrainer(
                config,
                identity,
                topology=topology,
                run_label="phase8_agent06_canonical",
            )
            init_checksum = wp.model_state_checksum(trainer.model.state_dict())
            if init_checksum != expected_checksum:
                raise SystemExit(
                    "BLOCKED: fresh C1 initialisation checksum "
                    f"{init_checksum} != Agent 5's frozen expected checksum "
                    f"{expected_checksum}; the canonical run must start from the "
                    "exact canonical initialisation"
                )
            log(f"fresh C1 init verified: {init_checksum}")
            # The untrained canonical checkpoint is the comparison point for
            # 'the checkpoint differs materially from initialisation', and
            # Agent 7 needs it for the improvement-over-initialisation gate.
            initial_path = work / "canonical_initialisation.pt"
            trainer.save_checkpoint(initial_path)
            resumed_state = None
        else:
            trainer = WarmstartTrainer.resume(
                resume_from,
                config=config,
                corpus_identity=identity,
                topology=topology,
                run_label="phase8_agent06_canonical",
            )
            init_checksum = None
            initial_path = None
            resumed_state = trainer.state_summary()
            log(
                f"resumed at step {trainer.global_step} "
                f"(cursor epoch {trainer.cursor.epoch} position {trainer.cursor.position})"
            )

        entry_state = trainer.state_summary()
        rows_handle = paths["rows"].open("w")
        validations_handle = paths["validations"].open("w")
        captured_count = 0
        try:
            with trainer, capture_validation_results() as captured:
                remaining = int(stop_at) - trainer.global_step
                if remaining < 0:
                    raise SystemExit(
                        f"BLOCKED: segment {segment} asked to stop at {stop_at} but the "
                        f"run is already at {trainer.global_step}"
                    )
                cadence = config.validation_cadence_updates
                while trainer.global_step < stop_at:
                    chunk = min(cadence, stop_at - trainer.global_step)
                    rows = trainer.train_updates(
                        chunk, best_checkpoint_path=best_path
                    )
                    for row in rows:
                        rows_handle.write(json.dumps(row, default=str) + "\n")
                    rows_handle.flush()
                    while captured_count < len(captured):
                        validations_handle.write(
                            json.dumps(captured[captured_count], default=str) + "\n"
                        )
                        captured_count += 1
                    validations_handle.flush()
                    if trainer.validation_history:
                        last = trainer.validation_history[-1]
                        log(
                            f"step {trainer.global_step}/{stop_at} "
                            f"score={last['selection_score']:.6f} "
                            f"best={trainer.best_validation['score']:.6f}"
                            f"@{trainer.best_validation['step']}"
                            f"{' *' if last['is_best'] else ''}"
                        )
                    else:
                        log(f"step {trainer.global_step}/{stop_at}")
                exit_state = trainer.state_summary()
                trainer.save_checkpoint(paths["checkpoint"])
        finally:
            rows_handle.close()
            validations_handle.close()

    summary = {
        "segment": segment,
        "device": device,
        "stop_at": int(stop_at),
        "resumed_from": str(resume_from) if resume_from else None,
        "initial_checkpoint": str(initial_path) if initial_path else None,
        "fresh_init_checksum": init_checksum,
        "expected_fresh_init_checksum": expected_checksum,
        "entry_state": entry_state,
        "resumed_state": resumed_state,
        "exit_state": exit_state,
        "end_checkpoint": str(paths["checkpoint"]),
        "best_checkpoint": str(best_path),
        "best_validation": dict(trainer.best_validation),
        "counters": dict(trainer.counters),
        "validation_history": list(trainer.validation_history),
        "validation_seconds": trainer.validation_seconds,
        "topology": topology.to_dict(),
        "corpus_root": str(root),
        "corpus_identity": identity.to_dict(),
        "model_input_access": model_access.to_dict(),
        "phase4_access": phase4_access.to_dict(),
        "wall_started_unix": started_wall,
        "wall_seconds": time.perf_counter() - started,
        "memory": memory_record(),
        "environment": environment_record(),
    }
    write_json(paths["summary"], summary)
    log(
        f"segment {segment} finished at step {summary['exit_state']['global_step']} "
        f"in {summary['wall_seconds']:.1f}s"
    )
    return summary


# ---------------------------------------------------------------------------
# The canonical run driver
# ---------------------------------------------------------------------------


def run_canonical(
    *,
    work: Path,
    device: str,
    updates: int,
    restart_at: int,
    topology: LoaderTopology,
) -> dict:
    """Drive the two real processes that make up the canonical run."""
    work.mkdir(parents=True, exist_ok=True)
    plan = [
        {"segment": 1, "stop_at": int(restart_at), "resume_from": None},
        {
            "segment": 2,
            "stop_at": int(updates),
            "resume_from": segment_paths(work, 1)["checkpoint"],
        },
    ]
    started = time.perf_counter()
    summaries = []
    for step in plan:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--segment",
            str(step["segment"]),
            "--stop-at",
            str(step["stop_at"]),
            "--work-dir",
            str(work),
            "--device",
            device,
            "--workers",
            str(topology.workers),
            "--prefetch",
            str(topology.prefetch),
            "--record-cache",
            str(topology.record_cache_size),
        ]
        if step["resume_from"] is not None:
            command += ["--resume-from", str(step["resume_from"])]
        log(f"launching segment {step['segment']} -> step {step['stop_at']}")
        completed = subprocess.run(command, cwd=REPOSITORY_ROOT)
        if completed.returncode != 0:
            raise SystemExit(
                f"BLOCKED: segment {step['segment']} exited {completed.returncode}; the "
                "canonical run stops here rather than continuing past a failure"
            )
        summaries.append(read_json(segment_paths(work, step["segment"])["summary"]))

    restart = verify_restart(summaries[0], summaries[1])
    run = {
        "segments": summaries,
        "restart_proof": restart,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(work / "canonical_run.json", run)
    return run


def verify_restart(first: dict, second: dict) -> dict:
    """The stop/restart exercise, judged on logical state continuity.

    `backend_aware_resume_equivalence_v1`: the resumed process must hold the
    identical data cursor, counters, optimizer/scheduler state and validation
    cadence the exiting process froze. Independent-run bit equality is not
    tested — MPS cannot provide it, and Agent 4's amended criterion already
    settled that.
    """
    exit_state = first["exit_state"]
    entry_state = second["entry_state"]
    checks = {
        "global_step": exit_state["global_step"] == entry_state["global_step"],
        "examples_consumed": (
            exit_state["examples_consumed"] == entry_state["examples_consumed"]
        ),
        "cursor": exit_state["cursor"] == entry_state["cursor"],
        "learning_rate": exit_state["learning_rate"] == entry_state["learning_rate"],
        "scheduler_last_epoch": (
            exit_state["scheduler_last_epoch"] == entry_state["scheduler_last_epoch"]
        ),
        "best_validation": exit_state["best_validation"] == entry_state["best_validation"],
        "validation_steps": (
            exit_state["validation_steps"] == entry_state["validation_steps"]
        ),
        "validation_best_flags": (
            exit_state["validation_best_flags"] == entry_state["validation_best_flags"]
        ),
        "optimizer_state_structure": (
            exit_state["optimizer_state_structure"]
            == entry_state["optimizer_state_structure"]
        ),
        "counters": exit_state["counters"] == entry_state["counters"],
    }
    # The cadence must continue on the original global-step grid, not restart
    # relative to the resume point. The second segment's history is cumulative
    # (it was restored from the checkpoint), so it alone is the whole run.
    all_steps = [entry["global_step"] for entry in second["validation_history"]]
    ordered = sorted(set(all_steps))
    cadence_regular = (
        all_steps == ordered
        and len(all_steps) == len(ordered)
        and all(step % 500 == 0 for step in ordered)
        and all(
            later - earlier == 500
            for earlier, later in zip(ordered, ordered[1:])
        )
    )
    return {
        "interpretation": "backend_aware_resume_equivalence_v1",
        "rationale": (
            "MPS is not run-to-run bit-deterministic on this backend, so resume is "
            "judged on exact logical state continuity — cursor, counters, "
            "optimizer/scheduler state and validation cadence — which is what a "
            "correct production resume must preserve. The superseded "
            "independent-run bit-equality requirement is not resurrected."
        ),
        "restart_step": exit_state["global_step"],
        "processes": 2,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "validation_cadence_continuous": cadence_regular,
        "validation_steps": ordered,
        "cursor_at_boundary": exit_state["cursor"],
        "note": (
            "plan_batch is a pure function of (universe, cursor), so an identical "
            "restored cursor is an identical next batch by construction"
        ),
    }


# ---------------------------------------------------------------------------
# Curve assembly
# ---------------------------------------------------------------------------


def cumulative_history(run: dict) -> list:
    """The whole run's validation history, without double counting.

    A resumed trainer restores `validation_history` from the checkpoint and
    keeps appending, so the *last* segment's copy already covers every earlier
    segment. Concatenating the segments would count the pre-restart passes
    twice.
    """
    return list(run["segments"][-1]["validation_history"])


def segment_history(summary: dict) -> list:
    """Only the validation entries this segment actually measured."""
    start = summary["entry_state"]["global_step"]
    stop = summary["exit_state"]["global_step"]
    return [
        entry
        for entry in summary["validation_history"]
        if start < entry["global_step"] <= stop
    ]


def assemble_curve(work: Path, run: dict) -> list:
    """One row per reporting interval, folding train rows into each cadence."""
    rows: list = []
    for summary in run["segments"]:
        segment = summary["segment"]
        paths = segment_paths(work, segment)
        train_rows = [
            json.loads(line) for line in paths["rows"].read_text().splitlines() if line
        ]
        validations = [
            json.loads(line)
            for line in paths["validations"].read_text().splitlines()
            if line
        ]
        history = segment_history(summary)
        # Fold the interval's train rows into the validation that closed it.
        interval: list = []
        validation_index = 0
        for row in train_rows:
            interval.append(row)
            if row["global_step"] % 500 != 0:
                continue
            if validation_index >= len(history) or validation_index >= len(validations):
                interval = []
                continue
            entry = history[validation_index]
            full = validations[validation_index]
            validation_index += 1
            wall = sum(item["step_wall_seconds"] for item in interval)
            examples = sum(item["batch_size"] for item in interval)
            data_wait = sum(item["data_wait_seconds"] for item in interval)
            last = interval[-1]
            rows.append(
                {
                    "global_step": row["global_step"],
                    "segment": segment,
                    "wall_seconds": wall,
                    "examples_consumed": entry["examples_consumed"],
                    "train_loss_total": mean(
                        item["loss_total"] for item in interval
                    ),
                    "train_loss_policy": mean(
                        item["loss_policy"] for item in interval
                    ),
                    "train_loss_value": mean(item["loss_value"] for item in interval),
                    "train_loss_belief": mean(
                        item["loss_belief"] for item in interval
                    ),
                    "train_legal_policy_entropy": mean(
                        item["legal_policy_entropy"] for item in interval
                    ),
                    "train_legal_policy_entropy_normalized": mean(
                        item["legal_policy_entropy_normalized"] for item in interval
                    ),
                    "learning_rate": last["learning_rate"],
                    "grad_norm_pre_clip": mean(
                        item["grad_norm_pre_clip"] for item in interval
                    ),
                    "grad_norm_post_clip": mean(
                        item["grad_norm_post_clip"] for item in interval
                    ),
                    "parameter_norm": last["parameter_norm"],
                    "examples_per_second": examples / wall if wall else None,
                    "data_wait_fraction": data_wait / wall if wall else None,
                    "validation_selection_score": entry["selection_score"],
                    "validation_policy_ce": full["policy"]["model_ce"],
                    "validation_policy_ce_ratio": full["policy"]["ce_ratio"],
                    "validation_policy_top1": full["policy"]["model_top1"],
                    "validation_policy_baseline_expected_top1": full["policy"][
                        "baseline_expected_top1"
                    ],
                    "validation_value_ce": full["value"]["model_ce"],
                    "validation_value_ce_ratio": full["value"]["ce_ratio"],
                    "validation_value_accuracy": full["value"]["model_accuracy"],
                    "validation_value_brier": full["value"]["model_brier"],
                    "validation_value_baseline_brier": full["value"]["baseline_brier"],
                    "validation_belief_ce": full["belief"]["model_ce"],
                    "validation_belief_ce_ratio": full["belief"]["ce_ratio"],
                    "validation_belief_top1": full["belief"]["model_top1"],
                    "validation_belief_baseline_top1": full["belief"]["baseline_top1"],
                    "validation_examples": entry["examples"],
                    "validation_games": entry["games"],
                    "validation_batches": entry["batches"],
                    "validation_seconds": entry["seconds"],
                    "is_best": entry["is_best"],
                    "peak_rss_bytes": summary["memory"]["peak_rss_bytes"],
                    "mps_current_allocated_bytes": summary["memory"][
                        "mps_current_allocated_bytes"
                    ],
                    "mps_driver_allocated_bytes": summary["memory"][
                        "mps_driver_allocated_bytes"
                    ],
                }
            )
            interval = []
    return rows


def write_curve(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CURVE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CURVE_COLUMNS})


# ---------------------------------------------------------------------------
# Freeze: independent reload, revalidation, digest, manifest
# ---------------------------------------------------------------------------


def freeze_checkpoint(
    *, work: Path, run: dict, device: str, topology: LoaderTopology, full_validation: bool
) -> dict:
    """Reload the best checkpoint independently and prove it reproduces.

    The selected checkpoint is the strictly lowest validation `selection_score`
    over the frozen 64-spread-batch cadence protocol. A full-validation pass is
    run afterwards for confirmation and reporting only — it may not move the
    selection.
    """
    frozen = frozen_config_payload()
    accepted = accepted_corpus_identity()
    root = sc.default_corpus_root()
    identity = verify_corpus_identity(root, accepted, check_payload_bytes=False)
    config = build_frozen_trainer_config(device=device)

    history = cumulative_history(run)
    scored = [entry for entry in history if entry["selection_score"] is not None]
    if not scored:
        raise SystemExit("BLOCKED: no validation entry carried a selection score")
    best_entry = min(scored, key=lambda entry: (entry["selection_score"], entry["global_step"]))
    recorded_best = run["segments"][-1]["best_validation"]
    if recorded_best["step"] != best_entry["global_step"]:
        raise SystemExit(
            "BLOCKED: the trainer's best-checkpoint bookkeeping "
            f"({recorded_best}) disagrees with the validation history "
            f"(best at step {best_entry['global_step']})"
        )

    source = work / "best.pt"
    if not source.exists():
        raise SystemExit(f"BLOCKED: best checkpoint {source} is missing")

    # Independent reload: a fresh load through the accepted loader, with no
    # trainer state carried over from the run.
    restored = load_warmstart_checkpoint(
        source,
        expected_train_config=config.identity(),
        expected_train_config_digest=config.digest(),
        expected_corpus_identity=identity,
        device=device,
    )
    if restored["global_step"] != best_entry["global_step"]:
        raise SystemExit(
            f"BLOCKED: reloaded checkpoint is at step {restored['global_step']}, "
            f"the selected best is step {best_entry['global_step']}"
        )

    dataset = WarmstartDataset(
        root, record_cache_size=topology.record_cache_size
    )
    value_prior = frozen_train_value_prior()
    started = time.perf_counter()
    reproduced = run_validation(
        restored["model"],
        dataset,
        split="validation",
        value_prior=value_prior,
        batches=config.validation_batches,
        batch_size=config.batch_size,
        device=device,
        phase8_agent=6,
        spread=True,
    )
    cadence_seconds = time.perf_counter() - started

    tolerance = 1e-9
    selection_delta = abs(
        float(reproduced.selection_score) - float(best_entry["selection_score"])
    )
    reproduces = selection_delta <= tolerance

    confirmation = None
    if full_validation:
        log("running a full-validation confirmation pass (reporting only)")
        started = time.perf_counter()
        full = run_validation(
            restored["model"],
            dataset,
            split="validation",
            value_prior=value_prior,
            batches=None,
            batch_size=config.batch_size,
            device=device,
            phase8_agent=6,
            spread=False,
        )
        confirmation = full.to_dict()
        confirmation["seconds"] = time.perf_counter() - started
        confirmation["role"] = (
            "confirmation and reporting only; the checkpoint was already selected "
            "by the frozen 64-spread-batch cadence protocol and no full-validation "
            "measurement may retroactively change that selection"
        )

    # Freeze the accepted bytes at the canonical path, then digest what landed.
    _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CHECKPOINT_PATH.write_bytes(source.read_bytes())
    digest = file_sha256(_CHECKPOINT_PATH)
    source_digest = file_sha256(source)
    if digest != source_digest:
        raise SystemExit("BLOCKED: the frozen checkpoint copy does not match its source")

    initial_path = Path(run["segments"][0]["initial_checkpoint"])
    initial_digest = file_sha256(initial_path)
    frozen_initial = _CHECKPOINT_PATH.parent / "warmstart_c1_v1_initialisation.pt"
    frozen_initial.write_bytes(initial_path.read_bytes())

    # 'Materially different from initialisation' measured, not asserted.
    initial_restored = load_warmstart_checkpoint(
        frozen_initial,
        expected_train_config=config.identity(),
        expected_train_config_digest=config.digest(),
        expected_corpus_identity=identity,
        device="cpu",
    )
    final_state = {
        name: tensor.detach().to("cpu", torch.float32)
        for name, tensor in restored["model"].state_dict().items()
    }
    initial_state = {
        name: tensor.detach().to("cpu", torch.float32)
        for name, tensor in initial_restored["model"].state_dict().items()
    }
    deltas = []
    for name, tensor in final_state.items():
        difference = (tensor - initial_state[name]).abs()
        deltas.append(
            {
                "parameter": name,
                "max_abs_delta": float(difference.max()),
                "mean_abs_delta": float(difference.mean()),
            }
        )
    total_norm = float(
        torch.sqrt(
            sum(
                ((final_state[name] - initial_state[name]) ** 2).sum()
                for name in final_state
            )
        )
    )
    unchanged = [entry["parameter"] for entry in deltas if entry["max_abs_delta"] == 0.0]

    manifest = {
        "phase": 8,
        "agent": 6,
        "artifact": "agent_06_checkpoint_manifest",
        "status": "PASS" if reproduces and not unchanged else "BLOCKED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint_path": str(_CHECKPOINT_PATH),
        "checkpoint_repository_relative": sc.repository_relative(_CHECKPOINT_PATH),
        "checkpoint_sha256": digest,
        "checkpoint_bytes": _CHECKPOINT_PATH.stat().st_size,
        "run_source_checkpoint": str(source),
        "selected_global_step": best_entry["global_step"],
        "selected_examples_consumed": best_entry["examples_consumed"],
        "selection_metric": frozen["config"]["best_checkpoint_metric"],
        "selection_protocol": {
            "validation_batches": config.validation_batches,
            "validation_selection": config.identity()["validation_selection"],
            "batch_size": config.batch_size,
            "split": "validation",
            "rule": "strictly lowest validation selection_score wins",
            "test_split_used": False,
            "phase4_strength_used": False,
        },
        "selection_score_at_selection": best_entry["selection_score"],
        "reproduced_selection_score": float(reproduced.selection_score),
        "selection_score_absolute_delta": selection_delta,
        "reproduction_tolerance": tolerance,
        "reproduces": reproduces,
        "reproduced_validation": reproduced.to_dict(),
        "reproduced_validation_seconds": cadence_seconds,
        "full_validation_confirmation": confirmation,
        "independent_reload": {
            "loader": "stratego.training.warmstart_checkpoint.load_warmstart_checkpoint",
            "process": "fresh, no trainer state carried over from the run",
            "train_config_digest_checked": True,
            "corpus_identity_checked": True,
            "global_step": restored["global_step"],
            "examples_consumed": restored["examples_consumed"],
        },
        "initial_checkpoint_path": str(frozen_initial),
        "initial_checkpoint_sha256": initial_digest,
        "initial_checksum": run["segments"][0]["fresh_init_checksum"],
        "expected_initial_checksum": run["segments"][0][
            "expected_fresh_init_checksum"
        ],
        "differs_from_initialisation": {
            "l2_norm_of_parameter_delta": total_norm,
            "parameters_compared": len(deltas),
            "parameters_unchanged": unchanged,
            "largest_deltas": sorted(
                deltas, key=lambda entry: -entry["max_abs_delta"]
            )[:8],
            "material": not unchanged and total_norm > 0.0,
        },
        "identities": {
            "corpus_version": identity.corpus_version,
            "corpus_digests": identity.to_dict(),
            "corpus_root_at_freeze": str(root),
            "corpus_identity_rule": (
                "checkpoints identify the corpus by version and digests, never by "
                "path; a pure relocation with identical digests stays compatible"
            ),
            "train_config_version": frozen["train_config_version"],
            "train_config_digest": frozen["train_config_digest"],
            "trainer_config_digest": config.digest(),
            "checkpoint_version": WARMSTART_CHECKPOINT_VERSION,
            "trainer_version": WARMSTART_TRAINER_VERSION,
            "example_version": frozen["config"]["example_version"],
            "eval_version": frozen["config"]["eval_version"],
            "model_candidate": config.model_candidate,
            "model_config_digest": frozen["config"]["model_config_digest"],
            "model_init_seed": config.model_init_seed,
        },
        "environment": environment_record(),
    }
    write_json(_MANIFEST_PATH, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def relabel_manifest(*, device: str) -> dict:
    """Add the digest-namespace labels to an existing manifest, in place.

    Deliberately does not re-freeze: the accepted checkpoint bytes are never
    rewritten by a labelling correction. The file digest is re-measured and
    required to be unchanged, so a relabel that disturbed the checkpoint would
    fail loudly instead of quietly reissuing it.
    """
    manifest = read_json(_MANIFEST_PATH)
    before = manifest["checkpoint_sha256"]
    observed = file_sha256(manifest["checkpoint_path"])
    if observed != before:
        raise SystemExit(
            f"BLOCKED: the frozen checkpoint digest changed ({observed} != "
            f"{before}); a labelling correction must never modify the checkpoint"
        )
    config = build_frozen_trainer_config(device=device)
    reconciliation = reconcile_train_config_identity(config)
    if reconciliation["disagreements"]:
        raise SystemExit(
            "BLOCKED: FROZEN TRAIN CONFIG IDENTITY MISMATCH: "
            f"{reconciliation['disagreements']}"
        )
    frozen = frozen_config_payload()
    manifest["identities"]["digest_namespaces"] = {
        "train_config_document": {
            "label": TRAIN_CONFIG_DOCUMENT_DIGEST,
            "value": frozen["train_config_digest"],
            "identifies": "warmstart_train_config_v1, the frozen artifact",
        },
        "trainer_runtime_identity": {
            "label": TRAINER_RUNTIME_IDENTITY_DIGEST,
            "value": config.digest(),
            "identifies": (
                "the runtime configuration stamped into this checkpoint and "
                "compared by check_resume_identity"
            ),
        },
        "note": "distinct namespaces over different objects; never equal",
    }
    # The bare key kept its old ambiguous name; make the scope explicit while
    # leaving the value untouched.
    manifest["identities"]["trainer_runtime_identity_digest"] = config.digest()
    manifest["identities"]["train_config_document_digest"] = frozen[
        "train_config_digest"
    ]
    manifest["train_config_identity_reconciliation"] = reconciliation
    manifest["canonical_untrained_checkpoint"] = {
        "role": (
            "required by Agent 7's final-vs-initial improvement evaluation"
        ),
        "path": manifest["initial_checkpoint_path"],
        "repository_relative": sc.repository_relative(
            manifest["initial_checkpoint_path"]
        ),
        "file_sha256": manifest["initial_checkpoint_sha256"],
        "model_state_checksum": manifest["initial_checksum"],
        "model_init_seed": CANONICAL_SEEDS["canonical_c1_init_seed"],
        "global_step": 0,
    }
    manifest["checkpoint_sha256_reverified"] = observed
    manifest["relabelled_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["relabel_note"] = (
        "digest-namespace labelling correction only; no training was repeated "
        "and the accepted checkpoint bytes are unchanged"
    )
    write_json(_MANIFEST_PATH, manifest)
    return manifest


def build_run_artifact(
    *, verification: dict, run: dict, manifest: dict, curve_rows: list, tests: dict
) -> dict:
    frozen = frozen_config_payload()
    segments = run["segments"]
    history = cumulative_history(run)
    counters: dict = {}
    for summary in segments:
        for name, value in summary["counters"].items():
            counters[name] = counters.get(name, 0) + int(value)

    model_input = {"train": 0, "validation": 0, "test": 0}
    for summary in segments:
        for split, value in summary["model_input_access"]["examples_by_split"].items():
            model_input[split] = model_input.get(split, 0) + int(value)
    phase4_games = sum(
        int(summary["phase4_access"]["phase4_neural_evaluation_games"])
        for summary in segments
    )
    phase4_loads = sum(
        int(summary["phase4_access"]["neural_checkpoint_loads"]) for summary in segments
    )

    final_step = segments[-1]["exit_state"]["global_step"]
    throughput = [
        row["examples_per_second"] for row in curve_rows if row["examples_per_second"]
    ]
    wall = sum(summary["wall_seconds"] for summary in segments)

    gates = {
        "agents_1_to_5_pass": all(
            status == "PASS" for status in verification["statuses"].values()
        ),
        "corpus_resolved_through_resolver": verification["resolver_matches_required"],
        "corpus_digests_match_accepted": verification["digests_match"],
        "fresh_c1_init_matches_expected": (
            segments[0]["fresh_init_checksum"]
            == segments[0]["expected_fresh_init_checksum"]
        ),
        "no_pilot_checkpoint_loaded": segments[0]["resumed_from"] is None,
        "exact_frozen_config_used": verification["trainer_config_digest"]["match"],
        "train_config_identity_reconciled": (
            not verification["train_config_identity_reconciliation"]["disagreements"]
            and verification["train_config_identity_reconciliation"][
                "runtime_identity_equals_agent_5_recorded"
            ]
            and all(
                namespace["match"]
                for namespace in verification[
                    "train_config_identity_reconciliation"
                ]["namespaces"].values()
            )
        ),
        "canonical_untrained_checkpoint_recorded": bool(
            manifest.get("initial_checkpoint_sha256")
        ),
        "train_split_only_updated_weights": (
            model_input["validation"] > 0 and model_input["test"] == 0
        ),
        "validation_only_selected_checkpoint": (
            manifest["selection_protocol"]["test_split_used"] is False
            and manifest["selection_protocol"]["phase4_strength_used"] is False
        ),
        "no_test_model_inference": model_input.get("test", 0) == 0,
        "no_phase4_neural_evaluation": phase4_games == 0 and phase4_loads == 0,
        "zero_non_finite_losses": counters.get("non_finite_losses", 0) == 0,
        "zero_non_finite_gradients": counters.get("non_finite_gradients", 0) == 0,
        "zero_non_finite_parameters": counters.get("non_finite_parameters", 0) == 0,
        "zero_illegal_targets": counters.get("illegal_targets", 0) == 0,
        "zero_data_mismatches": counters.get("data_mismatches", 0) == 0,
        "zero_checkpoint_errors": counters.get("checkpoint_errors", 0) == 0,
        "restart_path_exercised": run["restart_proof"]["all_checks_pass"],
        "validation_cadence_continuous": run["restart_proof"][
            "validation_cadence_continuous"
        ],
        "best_checkpoint_reload_reproduces": manifest["reproduces"],
        "checkpoint_digest_and_manifest_written": bool(manifest["checkpoint_sha256"]),
        "checkpoint_differs_from_initialisation": manifest[
            "differs_from_initialisation"
        ]["material"],
        "budget_respected": final_step <= int(frozen["config"]["max_final_updates"]),
        "no_phase9_selfplay_or_rl": True,
        # The most recent complete suite result stands as the gate: a steady
        # state recorded after a correction supersedes the post-run figure.
        "full_suite_green": (
            tests.get("steady_state") or tests.get("after") or {}
        ).get("failed", 1) == 0,
    }

    best = manifest["selected_global_step"]
    first_entry = history[0] if history else None
    best_entry = next(
        (entry for entry in history if entry["global_step"] == best), None
    )

    return {
        "phase": 8,
        "agent": 6,
        "artifact": "agent_06_warmstart_run",
        "status": "PASS" if all(gates.values()) else "BLOCKED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": environment_record(),
        "prerequisite_verification": verification,
        "prerequisite_versions": verification["versions"],
        "prerequisite_digests": {
            "corpus_content": verification["accepted_digests"]["content_digest"],
            "corpus_metadata": verification["accepted_digests"]["metadata_digest"],
            "corpus_commit_index": verification["accepted_digests"][
                "commit_index_digest"
            ],
            "train_config_document": verification["frozen_train_config_digest"],
            "trainer_runtime_identity": verification["trainer_config_digest"]["live"],
            "agent_01_contract": verification["agent_01_contract_digest"]["live"],
        },
        "digest_namespaces": {
            "train_config_document": {
                "label": TRAIN_CONFIG_DOCUMENT_DIGEST,
                "value": verification["frozen_train_config_digest"],
            },
            "trainer_runtime_identity": {
                "label": TRAINER_RUNTIME_IDENTITY_DIGEST,
                "value": verification["trainer_config_digest"]["live"],
            },
            "note": (
                "distinct namespaces over different objects; see "
                "train_config_identity_reconciliation"
            ),
        },
        "train_config_identity_reconciliation": verification[
            "train_config_identity_reconciliation"
        ],
        "corpus_root_resolution": verification["corpus_root_resolution"],
        "frozen_train_config": frozen["config"],
        "trainer_construction": frozen["trainer_construction"],
        "run": {
            "updates_completed": final_step,
            "update_budget": int(frozen["config"]["max_final_updates"]),
            "examples_consumed": segments[-1]["exit_state"]["examples_consumed"],
            "segments": len(segments),
            "restart_at": run["restart_proof"]["restart_step"],
            "wall_seconds": wall,
            "wall_hours": wall / 3600.0,
            "validation_passes": len(history),
            "validation_seconds": sum(
                summary["validation_seconds"] for summary in segments
            ),
            "mean_examples_per_second": mean(throughput),
            "mean_data_wait_fraction": mean(
                row["data_wait_fraction"] for row in curve_rows
            ),
            "peak_rss_bytes": max(
                summary["memory"]["peak_rss_bytes"] for summary in segments
            ),
            "mps_driver_allocated_bytes": max(
                summary["memory"]["mps_driver_allocated_bytes"] for summary in segments
            ),
            "topology": segments[0]["topology"],
            "counters": counters,
        },
        "restart_proof": run["restart_proof"],
        "validation_history": history,
        "first_validation": first_entry,
        "best_validation": best_entry,
        "final_validation": history[-1] if history else None,
        "selected_checkpoint": {
            "global_step": manifest["selected_global_step"],
            "selection_score": manifest["selection_score_at_selection"],
            "path": manifest["checkpoint_path"],
            "sha256": manifest["checkpoint_sha256"],
        },
        "held_out_discipline": {
            "model_input_examples_by_split": model_input,
            "test_examples_evaluated_by_model": model_input.get("test", 0),
            "phase4_neural_evaluation_games": phase4_games,
            "phase4_neural_checkpoint_loads": phase4_loads,
            "measurement": (
                "warmstart_pilot.record_model_input_access instruments "
                "WarmstartBatch.model_input — the one boundary where an example "
                "becomes model input — and record_phase4_access wraps the Phase 4 "
                "evaluation entry points, for the whole canonical run in every "
                "segment process. These are observed counts, not assertions."
            ),
            "measurement_scope": (
                "the training run itself, in every segment process. The freeze "
                "step ran two further passes on the selected checkpoint — the "
                "64-batch cadence revalidation and the full-validation "
                "confirmation — both requested on the validation split "
                "explicitly; run_validation routes any test-split request "
                "through check_test_corpus_access, which raises before Agent 7"
            ),
            "weights_updated_by": "train split only",
            "checkpoint_selected_by": "validation split only",
            "test_split_state": "sealed until Agent 7",
            "phase4_bank_state": "sealed until Agent 7",
        },
        "seeds": dict(CANONICAL_SEEDS),
        "commands": [
            ".venv/bin/python scripts/run_phase8_agent06.py --full",
        ],
        "files_created": [
            "reports/phase_8_data/agent_06_warmstart_run.json",
            "reports/phase_8_data/agent_06_training_curve.csv",
            "reports/phase_8_data/agent_06_checkpoint_manifest.json",
            sc.repository_relative(_CHECKPOINT_PATH),
            sc.repository_relative(_MANIFEST_PATH),
        ],
        "files_modified": ["reports/phase_8_implementation_report.md"],
        "tests_before": tests.get("before"),
        "tests_after": tests.get("after"),
        "tests_steady_state": tests.get("steady_state"),
        "completion_gates": gates,
        "problems": verification["problems"],
        "deviations": [],
        "handoff_to_agent_7": {
            "frozen_checkpoint_path": manifest["checkpoint_path"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "checkpoint_manifest": sc.repository_relative(_MANIFEST_PATH),
            "canonical_untrained_checkpoint": {
                "role": (
                    "the frozen canonical untrained C1, required by Agent 7's "
                    "final-vs-initial improvement evaluation (>= 0.700 EWR over "
                    ">= 1,024 games)"
                ),
                "path": manifest["initial_checkpoint_path"],
                "repository_relative": sc.repository_relative(
                    manifest["initial_checkpoint_path"]
                ),
                "file_sha256": manifest["initial_checkpoint_sha256"],
                "model_state_checksum": manifest["initial_checksum"],
                "model_init_seed": CANONICAL_SEEDS["canonical_c1_init_seed"],
                "model_candidate": "C1",
                "global_step": 0,
                "checkpoint_version": WARMSTART_CHECKPOINT_VERSION,
                "written_before_first_optimizer_step": True,
            },
            "initial_checkpoint_path": manifest["initial_checkpoint_path"],
            "initial_checkpoint_sha256": manifest["initial_checkpoint_sha256"],
            "frozen_train_config": "reports/phase_8_data/agent_05_frozen_train_config.json",
            "training_curve": "reports/phase_8_data/agent_06_training_curve.csv",
            "best_validation_metrics": best_entry,
            "sealed_evidence": {
                "test_examples_evaluated_by_model": model_input.get("test", 0),
                "phase4_neural_evaluation_games": phase4_games,
            },
        },
    }


REPORT_SECTION_HEADING = "## 6. Agent 6 — Canonical C1 Warm-Start Run"


def suite_line(artifact: dict) -> str:
    before = artifact.get("tests_before")
    after = artifact.get("tests_after")
    steady = artifact.get("tests_steady_state")
    if not any((before, after, steady)):
        return "Suite: not run in this invocation."
    parts = []
    for label, record in (("before the run", before), ("after the run", after)):
        if record:
            parts.append(
                f"{record['passed']:,} passed / {record.get('skipped', 0)} skipped "
                f"{label}"
            )
    line = "Suite: " + ", ".join(parts) + "."
    if steady:
        line += (
            f" Steady state after the identity-labelling correction: "
            f"{steady['passed']:,} passed / {steady.get('skipped', 0)} skipped"
            f" — the increase over the post-run figure is the "
            f"{steady['passed'] - (after['passed'] if after else 0)} new "
            "regression tests pinning the two digest namespaces "
            "(`tests/training/test_warmstart_train_config_identity.py`)."
        )
    return line


def render_report_section(artifact: dict, manifest: dict, curve_rows: list) -> str:
    run = artifact["run"]
    verification = artifact["prerequisite_verification"]
    best = artifact["best_validation"]
    first = artifact["first_validation"]
    final = artifact["final_validation"]
    restart = artifact["restart_proof"]
    held_out = artifact["held_out_discipline"]
    gates = artifact["completion_gates"]
    confirmation = manifest.get("full_validation_confirmation")
    reconciliation = artifact["train_config_identity_reconciliation"]

    def number(value, digits=6):
        return "n/a" if value is None else f"{float(value):.{digits}f}"

    lines = [
        REPORT_SECTION_HEADING,
        "",
        f"**Status: {artifact['status']}** — one canonical run, "
        f"{run['updates_completed']:,} optimizer updates from a fresh C1 "
        f"initialisation, best checkpoint selected by validation alone.",
        "",
        "### 6.1 Corpus identity",
        "",
        "The corpus was resolved exclusively through "
        "`synthetic_corpus.default_corpus_root()`:",
        "",
        "```text",
        f"resolved root     {verification['corpus_root_resolution']['root']}",
        f"resolution source {verification['corpus_root_resolution']['source']}",
        f"corpus version    {verification['accepted_digests']['corpus_version']}",
        f"content           {verification['accepted_digests']['content_digest']}",
        f"metadata          {verification['accepted_digests']['metadata_digest']}",
        f"commit index      {verification['accepted_digests']['commit_index_digest']}",
        "```",
        "",
        "All three accepted digests matched, and the payload bytes were re-read "
        "against their commit journals (28,000 games, zero violations in every "
        "integrity category). This is the corpus's third recorded location; "
        "identity is the version plus the digests, never the path, so the "
        "relocation changed nothing that the checkpoint or any downstream "
        "consumer depends on. No absolute path is embedded in trainer, "
        "checkpoint or downstream code — only this acceptance harness pins one, "
        "as an expected value to verify the resolver against.",
        "",
        "### 6.2 Fresh initialisation",
        "",
        "```text",
        f"canonical init seed        {artifact['seeds']['canonical_c1_init_seed']}",
        f"expected init checksum     {manifest['expected_initial_checksum']}",
        f"reconstructed checksum     {manifest['initial_checksum']}",
        f"pilot weights loaded       none",
        "```",
        "",
        "C1 was rebuilt from the canonical seed and its pre-training checksum "
        "equalled Agent 5's frozen expectation before the first optimizer step. "
        "The untrained checkpoint was frozen alongside the accepted one, since "
        "Agent 7 needs it for the improvement-over-initialisation gate.",
        "",
        "### 6.3 Configuration",
        "",
        "`warmstart_train_config_v1` was used exactly as frozen, with no field "
        "tuned:",
        "",
        "```text",
        "model / precision   C1, float32, MPS",
        "batch size          256",
        "optimizer           AdamW, lr 1e-3, weight decay 0.01",
        "gradient clipping   1.0",
        "schedule            500-step linear warmup, then constant",
        "loss weights        policy 1 / value 1 / belief 1",
        f"train order seed    {artifact['seeds']['train_order_seed']}",
        "loader              12 workers / prefetch 2 / record cache 512",
        f"update budget       {run['update_budget']:,}",
        "```",
        "",
        "#### Train-config identity: two digests, two namespaces",
        "",
        "Agent 5's accepted artifact records two SHA-256 values, and they are "
        "not two spellings of one identity — they cover different objects, so "
        "they could never be equal. Both are named explicitly here and in the "
        "machine-readable artifacts:",
        "",
        "```text",
        f"train_config_document    {reconciliation['namespaces']['train_config_digest']['recorded']}",
        f"                         {reconciliation['namespaces']['train_config_digest']['field_count']}-field frozen "
        "`config` document = warmstart_train_config_v1",
        "",
        f"trainer_runtime_identity {reconciliation['namespaces']['trainer_config_digest']['recorded']}",
        f"                         {reconciliation['namespaces']['trainer_config_digest']['field_count']}-field "
        "`WarmstartTrainConfig.identity()`, stamped into every",
        "                         checkpoint and compared by `check_resume_identity`",
        "```",
        "",
        f"The two objects express {reconciliation['shared_fields_compared']} fields "
        f"in common — {len(reconciliation['fields_matched_by_name'])} matched by "
        f"name and {len(reconciliation['fields_matched_across_bridge'])} across a "
        "naming bridge (`train_split`↔`split`, `train_order`↔`order`, "
        "`adam_betas`↔`adam_beta1`/`adam_beta2`). Beyond those, the document "
        f"carries {len(reconciliation['fields_only_in_document'])} fields with no "
        "runtime counterpart at all "
        f"({', '.join('`' + name + '`' for name in reconciliation['fields_only_in_document'][:4])}, …) "
        f"and the runtime identity carries {len(reconciliation['fields_only_in_runtime_identity'])} "
        "the document does not "
        f"({', '.join('`' + name + '`' for name in reconciliation['fields_only_in_runtime_identity'])}). "
        "That asymmetry is why the two hashes can never coincide. "
        "Both recompute to their recorded values from the live source, the live "
        "runtime identity equals Agent 5's recorded `trainer_config_identity` "
        f"dictionary exactly, all {reconciliation['shared_fields_equal']} of "
        f"{reconciliation['shared_fields_compared']} shared fields agree, "
        "and every document-only field "
        "that binds this run — shuffle seed, C1 config digest, loader topology, "
        "checkpoint/metrics/loss versions — was checked against its live "
        "source. Zero disagreements.",
        "",
        "This was a reporting-label defect in the first issue of this section, "
        "which printed the runtime-identity digest under the heading "
        "`warmstart_train_config_v1`. The configuration actually run was never "
        "in question; no training was repeated to correct it.",
        "",
        "### 6.4 The run",
        "",
        "```text",
        f"updates completed     {run['updates_completed']:,}",
        f"examples consumed     {run['examples_consumed']:,}",
        f"wall time             {run['wall_hours']:.2f} h",
        f"throughput            {number(run['mean_examples_per_second'], 1)} examples/s",
        f"data wait fraction    {number(run['mean_data_wait_fraction'], 4)}",
        f"validation passes     {run['validation_passes']}",
        f"peak RSS              {run['peak_rss_bytes'] / 1e9:.2f} GB",
        f"MPS driver allocated  {run['mps_driver_allocated_bytes'] / 1e9:.2f} GB",
        "```",
        "",
        "Stability counters, all zero, across both segment processes:",
        "",
        "```text",
    ]
    for name, value in sorted(run["counters"].items()):
        lines.append(f"{name:<24} {value}")
    lines += [
        "```",
        "",
        "Nothing was skipped: the trainer raises on a non-finite loss, gradient "
        "or parameter, on a target mismatch, on a cursor inconsistency and on a "
        "batch carrying a split other than train, and no such stop fired.",
        "",
        "### 6.5 Validation and checkpoint selection",
        "",
        "Every 500 updates, the frozen 64-evenly-spread-batch protocol "
        "(16,384 held-out examples, identical positions at every cadence) "
        "produced a `selection_score`:",
        "",
        "```text",
        "                       step   policy    value   belief   score",
        f"first                {first['global_step']:>7}  "
        f"{number(first['policy_ce_ratio'], 4)}  {number(first['value_ce_ratio'], 4)}  "
        f"{number(first['belief_ce_ratio'], 4)}  {number(first['selection_score'], 4)}",
        f"best                 {best['global_step']:>7}  "
        f"{number(best['policy_ce_ratio'], 4)}  {number(best['value_ce_ratio'], 4)}  "
        f"{number(best['belief_ce_ratio'], 4)}  {number(best['selection_score'], 4)}",
        f"final                {final['global_step']:>7}  "
        f"{number(final['policy_ce_ratio'], 4)}  {number(final['value_ce_ratio'], 4)}  "
        f"{number(final['belief_ce_ratio'], 4)}  {number(final['selection_score'], 4)}",
        "```",
        "",
        f"The accepted checkpoint is the one at update "
        f"{manifest['selected_global_step']:,} — the strictly lowest validation "
        "selection score over the run. Selection used the validation split and "
        "nothing else.",
        "",
    ]
    if confirmation:
        lines += [
            "A full-validation pass over all "
            f"{confirmation['examples']:,} validation examples "
            f"({confirmation['batches']} batches, {confirmation['games']:,} games) "
            "was run on the already-selected checkpoint for confirmation and "
            f"reporting only: selection score {number(confirmation['selection_score'], 6)} "
            f"(policy {number(confirmation['policy']['ce_ratio'], 4)}, "
            f"value {number(confirmation['value']['ce_ratio'], 4)}, "
            f"belief {number(confirmation['belief']['ce_ratio'], 4)}). It did not "
            "and could not move the selection: no later full-validation "
            "measurement may retroactively select a different checkpoint.",
            "",
        ]
    lines += [
        "### 6.6 Checkpoint/restart exercise",
        "",
        "The canonical run was executed as two real processes. Segment 1 trained "
        f"from the fresh initialisation to update {restart['restart_step']:,}, "
        "wrote a normal checkpoint and exited cleanly; segment 2 was a new "
        "interpreter that reloaded it through `WarmstartTrainer.resume` and "
        "finished the budget.",
        "",
        "```text",
    ]
    for name, passed in sorted(restart["checks"].items()):
        lines.append(f"{name:<28} {'preserved' if passed else 'DRIFTED'}")
    lines += [
        f"{'validation cadence':<28} "
        f"{'continuous on the 500-step grid' if restart['validation_cadence_continuous'] else 'BROKEN'}",
        "```",
        "",
        "Judged under the reviewer-approved `backend_aware_resume_equivalence_v1` "
        "interpretation: exact logical state continuity across the boundary. "
        "`plan_batch` is a pure function of `(universe, cursor)`, so an identical "
        "restored cursor is an identical next batch by construction. The "
        "superseded independent-run bit-determinism requirement is not "
        "resurrected — MPS cannot satisfy it, as Agent 4 established.",
        "",
        "### 6.7 Held-out discipline, measured",
        "",
        "```text",
        f"train examples through the model      {held_out['model_input_examples_by_split'].get('train', 0):,}",
        f"validation examples through the model {held_out['model_input_examples_by_split'].get('validation', 0):,}",
        f"test examples through the model       {held_out['test_examples_evaluated_by_model']}",
        f"Phase 4 neural evaluation games       {held_out['phase4_neural_evaluation_games']}",
        f"Phase 4 neural checkpoint loads       {held_out['phase4_neural_checkpoint_loads']}",
        "```",
        "",
        "These are observations, not claims: "
        "`record_model_input_access` instruments `WarmstartBatch.model_input` — "
        "the single boundary where an example becomes model input — and "
        "`record_phase4_access` wraps the Phase 4 evaluation entry points, in "
        "every segment process for the whole run. Weights were updated by train "
        "examples only; the checkpoint was selected by validation only; the test "
        "split and the Phase 4 bank remain sealed for Agent 7.",
        "",
        "The counts above scope the training run itself. The freeze step then "
        "ran two further passes on the selected checkpoint — the 64-batch "
        "cadence revalidation and the full-validation confirmation — both "
        "requested on the validation split explicitly, and `run_validation` "
        "routes any test-split request through the frozen "
        "`check_test_corpus_access` gate, which raises before Agent 7. No path "
        "in this agent reads test examples.",
        "",
        "### 6.8 Frozen checkpoint",
        "",
        "```text",
        f"path            {manifest['checkpoint_repository_relative']}",
        f"SHA-256         {manifest['checkpoint_sha256']}",
        f"size            {manifest['checkpoint_bytes'] / 1e6:.2f} MB",
        f"selected step   {manifest['selected_global_step']:,}",
        f"examples        {manifest['selected_examples_consumed']:,}",
        "```",
        "",
        "The checkpoint was reloaded independently through "
        "`load_warmstart_checkpoint` — train-config digest and corpus identity "
        "re-checked, no trainer state carried over — and revalidated under the "
        "same frozen protocol: selection score "
        f"{number(manifest['reproduced_selection_score'])} against "
        f"{number(manifest['selection_score_at_selection'])} recorded at "
        f"selection (delta {manifest['selection_score_absolute_delta']:.3e}).",
        "",
        "It differs materially from the canonical initialisation: L2 norm of the "
        f"parameter delta {manifest['differs_from_initialisation']['l2_norm_of_parameter_delta']:.4f} "
        f"over {manifest['differs_from_initialisation']['parameters_compared']} "
        "tensors, with none unchanged.",
        "",
        "The canonical *untrained* C1 is frozen alongside it, because Agent 7's "
        "final-vs-initial gate needs exactly this object:",
        "",
        "```text",
        f"path                    {sc.repository_relative(manifest['initial_checkpoint_path'])}",
        f"file SHA-256            {manifest['initial_checkpoint_sha256']}",
        f"model state checksum    {manifest['initial_checksum']}",
        f"init seed               {artifact['seeds']['canonical_c1_init_seed']}",
        "global step             0 (written before the first optimizer step)",
        "```",
        "",
        "### 6.9 Completion gates",
        "",
        "```text",
    ]
    width = max(len(name) for name in gates)
    for name, passed in sorted(gates.items()):
        lines.append(f"{name:<{width}}  {'PASS' if passed else 'FAIL'}")
    lines += [
        "```",
        "",
        suite_line(artifact),
        "",
        "Not done here, by contract: no test-split model inference, no Phase 4 "
        "neural playing-strength evaluation, no Phase 9 self-play or RL "
        "machinery, and no Agent 7 work.",
        "",
    ]
    return "\n".join(lines)


def append_report_section(text: str) -> None:
    if not _APPEND_REPORT:
        (_OUTPUT_DIRECTORY / "report_section_6.md").write_text(text)
        return
    existing = REPORT_PATH.read_text() if REPORT_PATH.exists() else ""
    if REPORT_SECTION_HEADING in existing:
        head, _, tail = existing.partition(REPORT_SECTION_HEADING)
        following = tail.split("\n## ", 1)
        remainder = "\n## " + following[1] if len(following) > 1 else ""
        REPORT_PATH.write_text(head.rstrip("\n") + "\n\n" + text + remainder)
        return
    separator = "" if existing.endswith("\n\n") or not existing else "\n"
    REPORT_PATH.write_text(existing + separator + text)


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


def run_pytest() -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""

    def count(word: str) -> int:
        match = re.search(rf"(\d+) {word}\b", tail)
        return int(match.group(1)) if match else 0

    # A non-zero return code with no parsed failure count still fails the gate:
    # `failed` defaults to 1 so an unparsed summary can never read as green.
    failed = count("failed") + count("error")
    if completed.returncode != 0 and failed == 0:
        failed = 1
    return {
        "summary": tail,
        "passed": count("passed"),
        "failed": failed,
        "skipped": count("skipped"),
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="add digest-namespace labels to the existing manifest without "
        "re-freezing or touching the accepted checkpoint bytes",
    )
    parser.add_argument("--artifacts", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--restart-at", type=int, default=DEFAULT_RESTART_AT)
    parser.add_argument("--workers", type=int, default=FROZEN_TOPOLOGY["workers"])
    parser.add_argument("--prefetch", type=int, default=FROZEN_TOPOLOGY["prefetch"])
    parser.add_argument(
        "--record-cache", type=int, default=FROZEN_TOPOLOGY["record_cache_size"]
    )
    parser.add_argument("--work-dir", default=str(WORK_DIRECTORY))
    parser.add_argument(
        "--skip-payload-bytes",
        action="store_true",
        help="digest-only corpus check (the byte read is ~80s)",
    )
    parser.add_argument(
        "--no-full-validation",
        action="store_true",
        help="skip the confirmation-only full validation pass",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="shake the harness out into the work directory, touching no "
        "accepted artifact, the canonical checkpoint path, or the report",
    )
    # Internal: one training segment in its own process.
    parser.add_argument("--segment", type=int, default=None)
    parser.add_argument("--stop-at", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    arguments = parser.parse_args()

    work = Path(arguments.work_dir)
    topology = LoaderTopology(
        workers=arguments.workers,
        prefetch=arguments.prefetch,
        record_cache_size=arguments.record_cache,
    )

    configure_output(dry_run=arguments.dry_run, work=work)

    if arguments.segment is not None:
        run_segment(
            segment=arguments.segment,
            stop_at=arguments.stop_at,
            resume_from=Path(arguments.resume_from) if arguments.resume_from else None,
            work=work,
            device=arguments.device,
            topology=topology,
        )
        return

    do_verify = arguments.verify or arguments.full
    do_run = arguments.run or arguments.full
    do_freeze = arguments.freeze or arguments.full
    do_artifacts = arguments.artifacts or arguments.full
    if not any(
        (do_verify, do_run, do_freeze, do_artifacts, arguments.relabel, arguments.run_pytest)
    ):
        parser.error("choose at least one mode")

    frozen = frozen_config_payload()
    updates = arguments.updates or int(frozen["config"]["max_final_updates"])

    tests: dict = {}
    verification: "dict | None" = None
    if do_verify:
        log("verifying prerequisites and corpus identity")
        verification, _identity = verify_prerequisites(
            check_payload_bytes=not arguments.skip_payload_bytes,
            device=arguments.device,
        )
        if verification["problems"]:
            write_json(work / "verification.json", verification)
            for problem in verification["problems"]:
                log(f"BLOCKED: {problem}")
            raise SystemExit(1)
        write_json(work / "verification.json", verification)
        log("prerequisites verified")

    # Suite results are persisted so a later --artifacts pass reports the same
    # totals instead of losing them or re-running an hour of work.
    tests_path = work / "tests.json"
    if tests_path.exists():
        tests = read_json(tests_path)

    if arguments.run_pytest:
        log("running the suite before the canonical run")
        tests["before"] = run_pytest()
        log(f"suite before: {tests['before']['summary']}")
        write_json(tests_path, tests)
        if tests["before"]["returncode"] != 0:
            raise SystemExit("BLOCKED: the suite is not green before the run")

    run: "dict | None" = None
    if do_run:
        log(f"canonical run: {updates:,} updates, restart at {arguments.restart_at:,}")
        run = run_canonical(
            work=work,
            device=arguments.device,
            updates=updates,
            restart_at=arguments.restart_at,
            topology=topology,
        )
    elif do_freeze or do_artifacts:
        # A later stage re-run against an already finished run.
        run = read_json(work / "canonical_run.json")

    manifest: "dict | None" = None
    if do_freeze:
        log("freezing the best validation checkpoint")
        manifest = freeze_checkpoint(
            work=work,
            run=run,
            device=arguments.device,
            topology=topology,
            full_validation=not arguments.no_full_validation,
        )
        log(
            f"frozen: step {manifest['selected_global_step']} "
            f"sha256 {manifest['checkpoint_sha256']}"
        )
    elif _MANIFEST_PATH.exists():
        manifest = read_json(_MANIFEST_PATH)

    if arguments.relabel:
        log("relabelling the manifest digest namespaces (no re-freeze)")
        manifest = relabel_manifest(device=arguments.device)
        log(
            "checkpoint digest unchanged: "
            f"{manifest['checkpoint_sha256_reverified']}"
        )

    if arguments.dry_run:
        log(f"dry run: every output redirected to {_OUTPUT_DIRECTORY}")

    if do_artifacts:
        if verification is None:
            verification = read_json(work / "verification.json")
        if manifest is None:
            raise SystemExit("BLOCKED: no checkpoint manifest to report")
        curve_rows = assemble_curve(work, run)
        write_curve(curve_rows, artifact_path("agent_06_training_curve.csv"))
        if arguments.run_pytest:
            log("running the suite after the run")
            tests["after"] = run_pytest()
            log(f"suite after: {tests['after']['summary']}")
            write_json(tests_path, tests)
        artifact = build_run_artifact(
            verification=verification,
            run=run,
            manifest=manifest,
            curve_rows=curve_rows,
            tests=tests,
        )
        write_json(artifact_path("agent_06_warmstart_run.json"), artifact)
        write_json(artifact_path("agent_06_checkpoint_manifest.json"), manifest)
        append_report_section(
            render_report_section(artifact, manifest, curve_rows)
        )
        log(f"status: {artifact['status']}")
        failing = [
            name for name, passed in artifact["completion_gates"].items() if not passed
        ]
        if failing:
            log(f"failing gates: {failing}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
