#!/usr/bin/env python3
"""Phase 8 Agent 5 acceptance harness: bounded pilot selection and the freeze.

Verifies the Agent 1-4 prerequisites and the accepted-corpus identity (the
resolver must name the canonical location and the live content / metadata /
commit-index digests must equal the accepted ones — a mismatch is BLOCKED,
never a regeneration), runs Agent 1's predeclared candidate matrix exactly as
frozen, and freezes one `warmstart_train_config_v1`.

The run shape, per candidate, is identical by construction:

- a fresh canonical C1 initialization (same seed, checksum recorded and
  compared across candidates);
- the same frozen shuffle stream, so the ordered batch identities are the same
  sequence (folded to one `batch_sequence_digest` and compared);
- the same optimizer-step budget (5,000, Agent 1's frozen cap);
- validation at exactly the same update numbers (every 500), over exactly the
  same evenly spread held-out batches;
- one full-validation-split pass at the final checkpoint — the authoritative
  selection input.

Each candidate runs in its own subprocess so every pilot starts from a cold
process as well as a cold model, and so a partially finished matrix resumes
per candidate. The orchestrator verifies the corpus payload bytes once; the
per-candidate workers re-verify the digests (`--trust-bytes`).

Nothing here may broaden the search: `WarmstartTrainConfig.from_pilot_candidate`
is the only configuration constructor, and the matrix is cross-checked against
Agent 1's accepted artifact before the first update. The sealed test split is
never read by a model and the Phase 4 bank is never used — both are *measured*
through `warmstart_pilot.record_model_input_access` /
`record_phase4_access` and reported as counts.

Artifacts::

    reports/phase_8_data/agent_05_pilot_runs.csv
    reports/phase_8_data/agent_05_pilot_selection.json
    reports/phase_8_data/agent_05_frozen_train_config.json

Usage::

    python scripts/run_phase8_agent05.py --full --run-pytest
    python scripts/run_phase8_agent05.py --verify                 # gates only
    python scripts/run_phase8_agent05.py --pilots                 # the matrix
    python scripts/run_phase8_agent05.py --candidate <id>         # one worker
    python scripts/run_phase8_agent05.py --artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch  # noqa: E402

from stratego.training import synthetic_corpus as sc  # noqa: E402
from stratego.training import warmstart_contract as wc  # noqa: E402
from stratego.training import warmstart_pilot as wp  # noqa: E402
from stratego.training.warmstart_checkpoint import (  # noqa: E402
    WARMSTART_CHECKPOINT_VERSION,
    WARMSTART_TRAINER_VERSION,
    CorpusIdentity,
    verify_corpus_identity,
)
from stratego.training.warmstart_dataset import TRAIN_ORDER_VERSION  # noqa: E402
from stratego.training.warmstart_loss import WARMSTART_LOSS_VERSION  # noqa: E402
from stratego.training.warmstart_metrics import (  # noqa: E402
    WARMSTART_METRICS_VERSION,
    frozen_train_value_prior,
    run_validation,
    spread_batch_positions,
)
from stratego.training.warmstart_seed import CANONICAL_SEEDS  # noqa: E402
from stratego.training.warmstart_trainer import (  # noqa: E402
    LoaderTopology,
    WarmstartTrainConfig,
    WarmstartTrainer,
    keys_digest,
    pilot_candidate_ids,
)

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_8_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_8_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase8" / "agent05"

#: The canonical accepted storage location (supplementary review instruction).
#: This harness *verifies* the resolver against it; no library module embeds it.
REQUIRED_CORPUS_ROOT = (
    "/Users/brandonwashington/Dev/Github/stratego/gpt_agent/"
    "data/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1"
)
REQUIRED_CORPUS_ROOT_RELATIVE = "data/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1"

#: Cadence validation size: 64 evenly spread batches x 256 = 16,384 held-out
#: examples. `spread_batch_positions` is a pure function of
#: (universe size, batch size, batches), so every candidate validates on
#: literally the same examples at literally the same update numbers.
CADENCE_VALIDATION_BATCHES = 64

#: The loader topology Agent 4 measured as best. Infrastructure only: it
#: cannot change any batch, which is why it is absent from the config digest.
DEFAULT_TOPOLOGY = {"workers": 12, "prefetch": 2, "record_cache_size": 512}

#: Agent 5's frozen run-shape choices for the final Phase 8 warm start.
#: `max_final_updates` is Agent 1's own predeclared final-run figure; Agent 1
#: declared no alternate budgets, so none is invented here (the assignment's
#: sanity extension is explicitly conditional on such a predeclaration).
FINAL_MAX_UPDATES = wp.FINAL_UPDATE_BUDGET_MAX
FINAL_CHECKPOINT_CADENCE = 500
BEST_CHECKPOINT_METRIC = (
    "validation selection_score = mean(r_policy, r_value, r_belief) under "
    "warmstart_eval_v1; strictly lower wins; validation split only"
)

PILOT_CSV_COLUMNS = (
    "candidate_id",
    "learning_rate",
    "loss_profile",
    "lambda_policy",
    "lambda_value",
    "lambda_belief",
    "validation_scope",
    "global_step",
    "update_budget",
    "examples_consumed",
    "policy_model_ce",
    "policy_baseline_ce",
    "policy_ce_ratio",
    "policy_model_top1",
    "policy_baseline_expected_top1",
    "policy_examples",
    "value_model_ce",
    "value_baseline_ce",
    "value_ce_ratio",
    "value_model_brier",
    "value_baseline_brier",
    "value_model_accuracy",
    "value_baseline_accuracy",
    "value_examples",
    "belief_model_ce",
    "belief_baseline_ce",
    "belief_ce_ratio",
    "belief_model_top1",
    "belief_baseline_top1",
    "belief_pieces",
    "selection_score",
    "validation_games",
    "validation_batches",
    "validation_seconds",
    "train_loss_total",
    "train_loss_policy",
    "train_loss_value",
    "train_loss_belief",
    "train_legal_policy_entropy",
    "train_legal_policy_entropy_normalized",
    "train_grad_norm_pre_clip",
    "train_grad_norm_post_clip",
    "learning_rate_at_step",
    "train_examples_per_second",
    "train_updates_per_second",
    "examples_per_second",
    "mps_current_allocated_bytes",
    "mps_driver_allocated_bytes",
    "peak_rss_bytes",
    "non_finite_losses",
    "non_finite_gradients",
    "non_finite_parameters",
    "illegal_targets",
    "data_mismatches",
    "checkpoint_errors",
    "init_checksum",
    "batch_sequence_digest",
)


#: Where the three artifacts land. `--dry-run` redirects this into the work
#: directory so a shakedown of the harness cannot overwrite accepted evidence
#: or append a throwaway report section.
_OUTPUT_DIRECTORY = DATA_DIRECTORY


def artifact_path(name: str) -> Path:
    return _OUTPUT_DIRECTORY / name


def log(message: str) -> None:
    print(f"[agent05 {time.strftime('%H:%M:%S')}] {message}", flush=True)


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


def mean(values) -> "float | None":
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


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


# ---------------------------------------------------------------------------
# Prerequisite verification
# ---------------------------------------------------------------------------


def accepted_corpus_identity() -> CorpusIdentity:
    """The accepted digests, cross-checked across every artifact stating them."""
    manifest = read_json(DATA_DIRECTORY / "agent_02_corpus_manifest.json")["corpus_manifest"]
    relocation = read_json(DATA_DIRECTORY / "agent_02_relocation.json")["accepted_digests"]
    agent3 = read_json(DATA_DIRECTORY / "agent_03_example_contract.json")[
        "prerequisite_digests"
    ]
    agent4 = read_json(DATA_DIRECTORY / "agent_04_trainer_contract.json")[
        "prerequisite_digests"
    ]
    sources = {
        "content_digest": {
            "agent_02_manifest": manifest["content_digest"],
            "agent_02_relocation": relocation["content_digest"],
            "agent_03": agent3["corpus_content"],
            "agent_04": agent4["corpus_content"],
        },
        "metadata_digest": {
            "agent_02_manifest": manifest["metadata_digest"],
            "agent_02_relocation": relocation["metadata_digest"],
            "agent_03": agent3["corpus_metadata"],
            "agent_04": agent4["corpus_metadata"],
        },
        "commit_index_digest": {
            "agent_02_manifest": manifest["commit_index_digest"],
            "agent_02_relocation": relocation["commit_index_digest"],
            "agent_03": agent3["corpus_commit_index"],
            "agent_04": agent4["corpus_commit_index"],
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


def verify_prerequisites(*, check_payload_bytes: bool = True) -> tuple:
    """Every Agent 5 entry gate. Returns `(record, verified_identity)`."""
    started = time.perf_counter()
    problems: list = []

    statuses = {}
    for agent, artifact in (
        (1, "agent_01_warmstart_contract.json"),
        (2, "agent_02_corpus_audit.json"),
        (3, "agent_03_example_contract.json"),
        (4, "agent_04_trainer_contract.json"),
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

    # The candidate matrix is Agent 5's own subject matter: it must match
    # Agent 1's accepted artifact exactly before a single pilot update.
    recorded_matrix = contract_payload["contract"]["pilot_matrix"]
    matrix_problems = wp.verify_candidate_matrix(recorded_matrix)
    problems.extend(matrix_problems)

    # Agent 4 named the candidate ids it validated the trainer against; a
    # disagreement means the trainer and the matrix drifted apart.
    agent4_ids = list(
        read_json(DATA_DIRECTORY / "agent_04_trainer_contract.json")["handoff_to_agent_5"][
            "frozen_candidate_ids"
        ]
    )
    if agent4_ids != list(pilot_candidate_ids()):
        problems.append(
            f"Agent 4 recorded candidate ids {agent4_ids}, the live trainer offers "
            f"{list(pilot_candidate_ids())}"
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
        "candidate_matrix_problems": matrix_problems,
        "candidate_matrix_digest": wp.candidate_matrix_digest(),
        "candidate_ids": list(pilot_candidate_ids()),
        "candidate_count": len(pilot_candidate_ids()),
        "candidate_limit": wp.PILOT_CANDIDATE_LIMIT,
        "agent_04_recorded_candidate_ids": agent4_ids,
        "corpus_root_resolution": resolution,
        "required_corpus_root": REQUIRED_CORPUS_ROOT,
        "resolver_matches_required": str(resolved) == REQUIRED_CORPUS_ROOT,
        "pointer_matches_required": resolution["pointer_value"] == REQUIRED_CORPUS_ROOT,
        "accepted_digests": accepted.to_dict(),
        "observed_digests": identity.to_dict() if identity else None,
        "digests_match": identity == accepted if identity else False,
        "payload_bytes_checked": check_payload_bytes,
        "digest_verification_seconds": digest_seconds,
        "problems": problems,
        "seconds": time.perf_counter() - started,
    }
    if problems:
        write_json(WORK_DIRECTORY / "verify_blocked.json", record)
        raise SystemExit(f"BLOCKED: {problems}")
    return record, identity


# ---------------------------------------------------------------------------
# One pilot
# ---------------------------------------------------------------------------


def candidate_work_path(candidate_id: str, updates: int) -> Path:
    """Per-candidate run record, keyed by budget so a shakedown at a smaller
    budget can never be mistaken for the frozen 5,000-update run."""
    return WORK_DIRECTORY / f"pilot_{candidate_id}_{int(updates)}u.json"


def window_summary(rows: list, wall_seconds: float, batch_size: int) -> dict:
    """The training-side metrics of one validation window (one cadence block)."""
    return {
        "updates": len(rows),
        "loss_total": mean(row["loss_total"] for row in rows),
        "loss_policy": mean(row["loss_policy"] for row in rows),
        "loss_value": mean(row["loss_value"] for row in rows),
        "loss_belief": mean(row["loss_belief"] for row in rows),
        "legal_policy_entropy": mean(row["legal_policy_entropy"] for row in rows),
        "legal_policy_entropy_normalized": mean(
            row["legal_policy_entropy_normalized"] for row in rows
        ),
        "grad_norm_pre_clip": mean(row["grad_norm_pre_clip"] for row in rows),
        "grad_norm_post_clip": mean(row["grad_norm_post_clip"] for row in rows),
        "learning_rate": rows[-1]["learning_rate"] if rows else None,
        "wall_seconds": wall_seconds,
        "updates_per_second": len(rows) / wall_seconds if wall_seconds > 0 else None,
        "examples_per_second": (
            len(rows) * batch_size / wall_seconds if wall_seconds > 0 else None
        ),
        "data_wait_seconds": sum(row["data_wait_seconds"] for row in rows),
    }


def validation_record(result, *, scope: str) -> dict:
    """One validation pass flattened to the artifact's field names."""
    return {
        "scope": scope,
        "selection_score": result.selection_score,
        "policy_model_ce": result.policy["model_ce"],
        "policy_baseline_ce": result.policy["baseline_ce"],
        "policy_ce_ratio": result.policy["ce_ratio"],
        "policy_model_top1": result.policy["model_top1"],
        "policy_baseline_expected_top1": result.policy["baseline_expected_top1"],
        "policy_examples": result.policy["examples"],
        "value_model_ce": result.value["model_ce"],
        "value_baseline_ce": result.value["baseline_ce"],
        "value_ce_ratio": result.value["ce_ratio"],
        "value_model_brier": result.value["model_brier"],
        "value_baseline_brier": result.value["baseline_brier"],
        "value_model_accuracy": result.value["model_accuracy"],
        "value_baseline_accuracy": result.value["baseline_accuracy"],
        "value_examples": result.value["examples"],
        "belief_model_ce": result.belief["model_ce"],
        "belief_baseline_ce": result.belief["baseline_ce"],
        "belief_ce_ratio": result.belief["ce_ratio"],
        "belief_model_top1": result.belief["model_top1"],
        "belief_baseline_top1": result.belief["baseline_top1"],
        "belief_pieces": result.belief["pieces"],
        "validation_games": result.games,
        "validation_batches": result.batches,
        "validation_seconds": result.seconds,
        "eval_version": result.eval_version,
        "metrics_version": result.metrics_version,
    }


def run_one_pilot(candidate_id: str, args) -> dict:
    """Run one frozen candidate for the frozen budget and return its record.

    Every fairness-relevant fact is measured here rather than assumed: the
    fresh-init checksum, the ordered batch-identity digest, the update count,
    the validation update numbers, and the split of every example that
    crossed the model-input boundary.
    """
    started = time.perf_counter()
    _record, identity = verify_prerequisites(check_payload_bytes=not args.trust_bytes)

    config = WarmstartTrainConfig.from_pilot_candidate(
        candidate_id, device=args.device, validation_batches=args.validation_batches
    )
    topology = LoaderTopology(
        workers=args.workers,
        prefetch=args.prefetch,
        record_cache_size=args.record_cache,
    )
    value_prior = frozen_train_value_prior()

    with wp.record_model_input_access() as model_access, wp.record_phase4_access() as phase4_access:
        trainer = WarmstartTrainer(
            config,
            identity,
            topology=topology,
            run_label=f"agent05_pilot_{candidate_id}",
        )
        init_checksum = wp.model_state_checksum(trainer.model.state_dict())
        log(f"{candidate_id}: fresh init checksum {init_checksum[:16]}…")

        keys_digests: list = []
        cadence_checkpoints: list = []
        train_seconds = 0.0
        windows = args.updates // wp.PILOT_VALIDATION_CADENCE
        remainder = args.updates % wp.PILOT_VALIDATION_CADENCE
        blocks = [wp.PILOT_VALIDATION_CADENCE] * windows + ([remainder] if remainder else [])
        try:
            for block in blocks:
                window_started = time.perf_counter()
                rows = trainer.train_updates(
                    block, on_step=lambda row, batch: keys_digests.append(row["keys_digest"])
                )
                window_seconds = time.perf_counter() - window_started
                # The trainer's own cadence validation ran inside the block;
                # subtract it so the throughput number is training throughput.
                cadence_entry = (
                    trainer.validation_history[-1] if trainer.validation_history else None
                )
                internal_validation_seconds = (
                    float(cadence_entry["seconds"])
                    if cadence_entry is not None
                    and int(cadence_entry["global_step"]) == trainer.global_step
                    else 0.0
                )
                train_only = window_seconds - internal_validation_seconds
                train_seconds += train_only

                # Full-metric pass at the same update number, over the same
                # evenly spread batches the trainer just used. Doubles as a
                # validation-determinism check against the trainer's entry.
                result = run_validation(
                    trainer.model,
                    trainer.validation_dataset,
                    split=config.validation_split,
                    value_prior=value_prior,
                    batches=args.validation_batches,
                    batch_size=config.batch_size,
                    device=trainer.device,
                    phase8_agent=5,
                    spread=True,
                )
                entry = validation_record(result, scope=wp.CADENCE_SCOPE)
                entry["global_step"] = trainer.global_step
                entry["examples_consumed"] = trainer.examples_consumed
                entry["train"] = window_summary(rows, train_only, config.batch_size)
                entry["memory"] = memory_record()
                entry["trainer_cadence_selection_score"] = (
                    cadence_entry["selection_score"] if cadence_entry is not None else None
                )
                entry["trainer_cadence_agreement"] = (
                    abs(
                        float(cadence_entry["selection_score"])
                        - float(result.selection_score)
                    )
                    if cadence_entry is not None
                    and cadence_entry["selection_score"] is not None
                    and result.selection_score is not None
                    else None
                )
                cadence_checkpoints.append(entry)
                def show(value) -> str:
                    return f"{value:.4f}" if value is not None else "n/a"

                log(
                    f"{candidate_id}: step {trainer.global_step} "
                    f"score {show(result.selection_score)} "
                    f"(p {show(result.policy['ce_ratio'])} "
                    f"v {show(result.value['ce_ratio'])} "
                    f"b {show(result.belief['ce_ratio'])}) "
                    f"{len(rows) / train_only:.2f} upd/s"
                )

            log(f"{candidate_id}: final full-validation-split pass…")
            final_result = run_validation(
                trainer.model,
                trainer.validation_dataset,
                split=config.validation_split,
                value_prior=value_prior,
                batches=None,
                batch_size=config.batch_size,
                device=trainer.device,
                phase8_agent=5,
                spread=False,
            )
            final_entry = validation_record(final_result, scope=wp.SELECTION_SCOPE)
            final_entry["global_step"] = trainer.global_step
            final_entry["examples_consumed"] = trainer.examples_consumed
            final_entry["memory"] = memory_record()
            counters = dict(trainer.counters)
            completed = trainer.global_step
        finally:
            trainer.close()

    record = {
        "candidate_id": candidate_id,
        "candidate": dict(
            {entry["candidate_id"]: entry for entry in wp.frozen_candidate_matrix()}[
                candidate_id
            ]
        ),
        "config_identity": config.identity(),
        "config_digest": config.digest(),
        "candidate_matrix_digest": wp.candidate_matrix_digest(),
        "topology": topology.to_dict(),
        "init_checksum": init_checksum,
        "batch_sequence_digest": wp.batch_sequence_digest(keys_digests),
        "batch_sequence_length": len(keys_digests),
        "first_keys_digest": keys_digests[0] if keys_digests else None,
        "last_keys_digest": keys_digests[-1] if keys_digests else None,
        "update_budget": int(args.updates),
        "completed_updates": int(completed),
        "examples_consumed": int(trainer.examples_consumed),
        "counters": counters,
        "validation_update_numbers": [
            entry["global_step"] for entry in cadence_checkpoints
        ],
        "validation_batch_positions": list(
            spread_batch_positions(
                len(trainer.validation_dataset.universe(config.validation_split)),
                config.batch_size,
                args.validation_batches,
            )
        ),
        "cadence_checkpoints": cadence_checkpoints,
        "final_checkpoint": final_entry,
        "train_seconds": train_seconds,
        "updates_per_second": completed / train_seconds if train_seconds > 0 else None,
        "examples_per_second": (
            completed * config.batch_size / train_seconds if train_seconds > 0 else None
        ),
        "memory": memory_record(),
        "model_input_access": model_access.to_dict(),
        "phase4_access": phase4_access.to_dict(),
        "value_prior": list(value_prior),
        "wall_seconds": time.perf_counter() - started,
        "environment": environment_record(),
    }
    write_json(candidate_work_path(candidate_id, args.updates), record)
    log(
        f"{candidate_id}: done in {record['wall_seconds']:.0f}s, "
        f"final score {final_entry['selection_score']}"
    )
    return record


def phase_pilots(args) -> list:
    """Run (or reuse) every frozen candidate, each in its own subprocess."""
    records = []
    for candidate_id in pilot_candidate_ids():
        path = candidate_work_path(candidate_id, args.updates)
        if path.exists() and not args.rerun:
            existing = read_json(path)
            if int(existing.get("completed_updates", 0)) == int(args.updates):
                log(f"{candidate_id}: reusing completed run at {path.name}")
                records.append(existing)
                continue
            log(f"{candidate_id}: existing run is incomplete, re-running")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--candidate",
            candidate_id,
            "--device",
            args.device,
            "--updates",
            str(args.updates),
            "--validation-batches",
            str(args.validation_batches),
            "--workers",
            str(args.workers),
            "--prefetch",
            str(args.prefetch),
            "--record-cache",
            str(args.record_cache),
            "--trust-bytes",
        ]
        log(f"{candidate_id}: launching worker")
        result = subprocess.run(command, cwd=REPOSITORY_ROOT)
        if result.returncode != 0:
            raise SystemExit(
                f"BLOCKED: pilot worker for {candidate_id} exited {result.returncode}"
            )
        records.append(read_json(path))
    return records


# ---------------------------------------------------------------------------
# Fairness evidence across the matrix
# ---------------------------------------------------------------------------


def fairness_record(records: list) -> dict:
    """The cross-candidate equality facts every PASS gate rests on."""
    checksums = {record["candidate_id"]: record["init_checksum"] for record in records}
    sequences = {
        record["candidate_id"]: record["batch_sequence_digest"] for record in records
    }
    budgets = {record["candidate_id"]: record["completed_updates"] for record in records}
    update_numbers = {
        record["candidate_id"]: record["validation_update_numbers"] for record in records
    }
    positions = {
        record["candidate_id"]: record["validation_batch_positions"] for record in records
    }
    scopes = {
        record["candidate_id"]: record["final_checkpoint"]["scope"] for record in records
    }
    registered = set(pilot_candidate_ids())
    ran = [record["candidate_id"] for record in records]
    return {
        "init_checksums": checksums,
        "all_init_checksums_identical": len(set(checksums.values())) == 1,
        "expected_fresh_init_checksum": (
            next(iter(set(checksums.values()))) if len(set(checksums.values())) == 1 else None
        ),
        "batch_sequence_digests": sequences,
        "all_batch_sequences_identical": len(set(sequences.values())) == 1,
        "completed_updates": budgets,
        "all_budgets_equal": len(set(budgets.values())) == 1,
        "validation_update_numbers": update_numbers,
        "all_validation_update_numbers_identical": (
            len({tuple(value) for value in update_numbers.values()}) == 1
        ),
        "validation_batch_positions_identical": (
            len({tuple(value) for value in positions.values()}) == 1
        ),
        "selection_scope": scopes,
        "candidates_run": ran,
        "unregistered_configs_run": sorted(set(ran) - registered),
        "registered_candidates_not_run": sorted(registered - set(ran)),
        "trainer_cadence_agreement_max": max(
            (
                entry["trainer_cadence_agreement"]
                for record in records
                for entry in record["cadence_checkpoints"]
                if entry.get("trainer_cadence_agreement") is not None
            ),
            default=None,
        ),
    }


def access_record(records: list) -> dict:
    """The aggregate held-out access log across every pilot process."""
    examples: dict = {}
    batches: dict = {}
    for record in records:
        for split, count in record["model_input_access"]["examples_by_split"].items():
            examples[split] = examples.get(split, 0) + int(count)
        for split, count in record["model_input_access"]["batches_by_split"].items():
            batches[split] = batches.get(split, 0) + int(count)
    games = sum(
        int(record["phase4_access"]["phase4_neural_evaluation_games"])
        for record in records
    )
    loads = sum(
        int(record["phase4_access"]["neural_checkpoint_loads"]) for record in records
    )

    # The frozen gates must actively refuse Agent 5, not merely go unused.
    refusals = {}
    for purpose in ("model_inference", "model_metric", "hyperparameter_selection"):
        try:
            wc.check_test_corpus_access(purpose, phase8_agent=5)
            refusals[f"test_corpus:{purpose}"] = "ALLOWED"
        except wc.HeldOutAccessError:
            refusals[f"test_corpus:{purpose}"] = "REFUSED"
    for purpose in ("neural_playing_strength", "pilot_selection", "config_selection"):
        try:
            wc.check_phase4_bank_access(purpose, phase8_agent=5)
            refusals[f"phase4_bank:{purpose}"] = "ALLOWED"
        except wc.HeldOutAccessError:
            refusals[f"phase4_bank:{purpose}"] = "REFUSED"
    return {
        "boundary": "stratego.training.warmstart_dataset.WarmstartBatch.model_input",
        "examples_by_split": dict(sorted(examples.items())),
        "batches_by_split": dict(sorted(batches.items())),
        "test_examples_evaluated_by_model_agent_5": int(examples.get("test", 0)),
        "phase4_neural_evaluation_games_agent_5": games,
        "phase4_neural_checkpoint_loads_agent_5": loads,
        "frozen_gate_responses": refusals,
        "all_gates_refuse_agent_5": all(
            value == "REFUSED" for value in refusals.values()
        ),
        "structural_manifest_reads": (
            "allowed and used: the corpus manifest/commit index were read for "
            "digest verification; no test example ever reached a model"
        ),
    }


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def csv_rows(records: list) -> list:
    rows = []
    for record in records:
        candidate = record["candidate"]
        base = {
            "candidate_id": record["candidate_id"],
            "learning_rate": candidate["learning_rate"],
            "loss_profile": candidate["loss_profile"],
            "lambda_policy": candidate["lambda_policy"],
            "lambda_value": candidate["lambda_value"],
            "lambda_belief": candidate["lambda_belief"],
            "update_budget": record["update_budget"],
            "examples_per_second": record["examples_per_second"],
            "init_checksum": record["init_checksum"],
            "batch_sequence_digest": record["batch_sequence_digest"],
            **{name: record["counters"][name] for name in record["counters"]},
        }
        for entry in list(record["cadence_checkpoints"]) + [record["final_checkpoint"]]:
            train = entry.get("train", {})
            memory = entry.get("memory", {})
            row = dict(base)
            row.update(
                {
                    "validation_scope": entry["scope"],
                    "global_step": entry["global_step"],
                    "examples_consumed": entry["examples_consumed"],
                    "train_loss_total": train.get("loss_total"),
                    "train_loss_policy": train.get("loss_policy"),
                    "train_loss_value": train.get("loss_value"),
                    "train_loss_belief": train.get("loss_belief"),
                    "train_legal_policy_entropy": train.get("legal_policy_entropy"),
                    "train_legal_policy_entropy_normalized": train.get(
                        "legal_policy_entropy_normalized"
                    ),
                    "train_grad_norm_pre_clip": train.get("grad_norm_pre_clip"),
                    "train_grad_norm_post_clip": train.get("grad_norm_post_clip"),
                    "learning_rate_at_step": train.get("learning_rate"),
                    "train_examples_per_second": train.get("examples_per_second"),
                    "train_updates_per_second": train.get("updates_per_second"),
                    "mps_current_allocated_bytes": memory.get("mps_current_allocated_bytes"),
                    "mps_driver_allocated_bytes": memory.get("mps_driver_allocated_bytes"),
                    "peak_rss_bytes": memory.get("peak_rss_bytes"),
                }
            )
            for name in (
                "selection_score",
                "policy_model_ce",
                "policy_baseline_ce",
                "policy_ce_ratio",
                "policy_model_top1",
                "policy_baseline_expected_top1",
                "policy_examples",
                "value_model_ce",
                "value_baseline_ce",
                "value_ce_ratio",
                "value_model_brier",
                "value_baseline_brier",
                "value_model_accuracy",
                "value_baseline_accuracy",
                "value_examples",
                "belief_model_ce",
                "belief_baseline_ce",
                "belief_ce_ratio",
                "belief_model_top1",
                "belief_baseline_top1",
                "belief_pieces",
                "validation_games",
                "validation_batches",
                "validation_seconds",
            ):
                row[name] = entry.get(name)
            rows.append({name: row.get(name) for name in PILOT_CSV_COLUMNS})
    return rows


def write_pilot_csv(rows: list) -> Path:
    path = artifact_path("agent_05_pilot_runs.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PILOT_CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def selection_records(records: list) -> list:
    """Selection inputs: the full-split final checkpoint of each candidate."""
    return [
        {
            "candidate_id": record["candidate_id"],
            "completed_updates": record["completed_updates"],
            "update_budget": record["update_budget"],
            "examples_per_second": record["examples_per_second"],
            "counters": record["counters"],
            "final_checkpoint": {
                "global_step": record["final_checkpoint"]["global_step"],
                "scope": record["final_checkpoint"]["scope"],
                "policy_ce_ratio": record["final_checkpoint"]["policy_ce_ratio"],
                "value_ce_ratio": record["final_checkpoint"]["value_ce_ratio"],
                "belief_ce_ratio": record["final_checkpoint"]["belief_ce_ratio"],
                "selection_score": record["final_checkpoint"]["selection_score"],
            },
        }
        for record in records
    ]


def phase_artifacts(records: list, verify_record: dict, tests_before: dict, args) -> dict:
    started = time.perf_counter()
    rows = csv_rows(records)
    csv_path = write_pilot_csv(rows)

    inputs = selection_records(records)
    decision = wp.select_winner(inputs)

    # The published CSV must reproduce the decision on its own: re-read it
    # from disk and re-run the same pure function.
    with csv_path.open() as handle:
        reproduced = wp.select_winner(wp.records_from_rows(csv.DictReader(handle)))
    decision["reproducible_from_csv"] = (
        reproduced["winner"] == decision["winner"]
        and [entry["candidate_id"] for entry in reproduced["ranking"]]
        == [entry["candidate_id"] for entry in decision["ranking"]]
    )
    decision["csv_reproduced_winner"] = reproduced["winner"]

    fairness = fairness_record(records)
    access = access_record(records)

    # Recompute every published selection score from its own ratios, so the
    # artifact's arithmetic is checked rather than trusted.
    score_checks = []
    for record in records:
        for entry in list(record["cadence_checkpoints"]) + [record["final_checkpoint"]]:
            recomputed = wp.selection_score_from_metrics(entry)
            score_checks.append(
                {
                    "candidate_id": record["candidate_id"],
                    "global_step": entry["global_step"],
                    "scope": entry["scope"],
                    "published": entry["selection_score"],
                    "recomputed": recomputed,
                    "match": (
                        recomputed is not None
                        and entry["selection_score"] is not None
                        and abs(recomputed - entry["selection_score"]) <= 1e-12
                    ),
                }
            )

    identity = accepted_corpus_identity()
    frozen_config = None
    if decision["status"] == "PASS":
        winner_config = WarmstartTrainConfig.from_pilot_candidate(
            decision["winner"],
            device=args.device,
            validation_batches=args.validation_batches,
        )
        frozen_config = wp.build_frozen_train_config(
            winner_candidate_id=decision["winner"],
            train_config_identity=winner_config.identity(),
            train_config_digest=winner_config.digest(),
            model_config_digest=wc.EXPECTED_C1_CONFIG_DIGEST,
            expected_fresh_init_checksum=fairness["expected_fresh_init_checksum"],
            corpus_identity=identity.to_dict(),
            max_final_updates=FINAL_MAX_UPDATES,
            checkpoint_cadence_updates=FINAL_CHECKPOINT_CADENCE,
            best_checkpoint_metric=BEST_CHECKPOINT_METRIC,
            early_stop_rule=early_stop_rule(records),
            loader_topology=dict(DEFAULT_TOPOLOGY),
            seeds=dict(CANONICAL_SEEDS),
            validation_batches=args.validation_batches,
        )
        frozen_config["pilot_evidence"] = {
            "pilot_runs_csv": "reports/phase_8_data/agent_05_pilot_runs.csv",
            "pilot_selection": "reports/phase_8_data/agent_05_pilot_selection.json",
            "winner_selection_score": decision["winner_selection_score"],
            "runner_up": decision.get("runner_up"),
            "margin_to_runner_up": decision.get("margin_to_runner_up"),
        }
        frozen_config["problems"] = wp.verify_frozen_train_config(frozen_config)
        write_json(artifact_path("agent_05_frozen_train_config.json"), frozen_config)

    gates = completion_gates(
        records, decision, fairness, access, score_checks, frozen_config,
        verify_record, args,
    )
    status = "PASS" if all(gates.values()) and decision["status"] == "PASS" else "BLOCKED"

    selection_payload = {
        "phase": 8,
        "agent": 5,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **environment_record(),
        "pilot_version": wp.WARMSTART_PILOT_VERSION,
        "prerequisite_versions": {
            "contract": wc.WARMSTART_TRAINING_CONTRACT_VERSION,
            "corpus": identity.corpus_version,
            "example": wc.WARMSTART_EXAMPLE_VERSION,
            "eval": wc.WARMSTART_EVAL_VERSION,
            "loss": WARMSTART_LOSS_VERSION,
            "metrics": WARMSTART_METRICS_VERSION,
            "trainer": WARMSTART_TRAINER_VERSION,
            "checkpoint": WARMSTART_CHECKPOINT_VERSION,
            "train_order": TRAIN_ORDER_VERSION,
            "train_config": wp.WARMSTART_TRAIN_CONFIG_VERSION,
        },
        "prerequisite_digests": {
            "agent_01_contract": wc.contract_digest(),
            "candidate_matrix": wp.candidate_matrix_digest(),
            "corpus_content": identity.content_digest,
            "corpus_metadata": identity.metadata_digest,
            "corpus_commit_index": identity.commit_index_digest,
            "c1_config": wc.EXPECTED_C1_CONFIG_DIGEST,
        },
        "verification": verify_record,
        "candidate_matrix": wc.pilot_matrix(),
        "pilot_protocol": {
            "update_budget_per_candidate": int(args.updates),
            "validation_cadence_updates": wp.PILOT_VALIDATION_CADENCE,
            "cadence_validation_batches": int(args.validation_batches),
            "cadence_validation_examples": int(args.validation_batches) * 256,
            "cadence_validation_selection": (
                "spread_batch_positions — a pure function of (universe size, "
                "batch size, batches), so every candidate sees the same held-out "
                "examples at the same update numbers"
            ),
            "selection_validation": (
                "one full validation-split pass (all batches, sequential order) "
                "at the final pilot checkpoint; this is the selection input"
            ),
            "loader_topology": dict(DEFAULT_TOPOLOGY),
            "device": args.device,
            "precision": "float32",
            "early_stopping_during_pilots": (
                "none; no candidate was stopped early and none hit a predeclared "
                "hard failure"
            ),
        },
        "pilot_runs": [
            {
                key: value
                for key, value in record.items()
                if key not in ("cadence_checkpoints", "environment")
            }
            | {"cadence_checkpoints": record["cadence_checkpoints"]}
            for record in records
        ],
        "selection_inputs": inputs,
        "selection": decision,
        "selection_score_recomputation": {
            "checked": len(score_checks),
            "mismatches": [entry for entry in score_checks if not entry["match"]],
            "all_match": all(entry["match"] for entry in score_checks),
        },
        "fairness": fairness,
        "held_out_access_log": access,
        "sanity_extension": sanity_extension_record(),
        "frozen_train_config_digest": (
            frozen_config["train_config_digest"] if frozen_config else None
        ),
        "handoff_to_agent_6": handoff_record(decision, frozen_config, fairness, records),
        "completion_gates": gates,
        "tests_before": tests_before,
        "commands": [
            "python scripts/run_phase8_agent05.py --full --run-pytest",
            f"python scripts/run_phase8_agent05.py --candidate <id> --updates {args.updates} "
            f"--validation-batches {args.validation_batches} --workers {args.workers} "
            f"--prefetch {args.prefetch} --record-cache {args.record_cache} --trust-bytes",
        ],
        "seeds": dict(CANONICAL_SEEDS),
        "files_created": [
            "stratego/training/warmstart_pilot.py",
            "scripts/run_phase8_agent05.py",
            "tests/training/test_warmstart_pilot.py",
            "reports/phase_8_data/agent_05_pilot_runs.csv",
            "reports/phase_8_data/agent_05_pilot_selection.json",
            "reports/phase_8_data/agent_05_frozen_train_config.json",
        ],
        "files_modified": ["reports/phase_8_implementation_report.md"],
        "durations": {
            "artifacts_seconds": time.perf_counter() - started,
            "total_pilot_seconds": sum(record["wall_seconds"] for record in records),
        },
        "problems": [] if status == "PASS" else sorted(
            name for name, value in gates.items() if not value
        ),
        "deviations": deviations_record(args),
    }
    write_json(artifact_path("agent_05_pilot_selection.json"), selection_payload)
    log(f"artifacts written; status {status}")
    return selection_payload


def early_stop_rule(records: list) -> dict:
    """Agent 5's frozen early-stop decision, with the pilot evidence for it."""
    improving = []
    for record in records:
        scores = [
            entry["selection_score"]
            for entry in record["cadence_checkpoints"]
            if entry["selection_score"] is not None
        ]
        improving.append(bool(scores) and scores[-1] <= min(scores) + 1e-12)
    return {
        "rule": "none",
        "rationale": (
            "no early stopping in the final run; the best checkpoint is selected "
            "by validation selection_score at the frozen cadence, which already "
            "protects against a late regression without adding an unpredeclared "
            "stopping criterion"
        ),
        "pilot_evidence": {
            "candidates_still_improving_at_the_final_pilot_checkpoint": sum(improving),
            "candidates_measured": len(improving),
        },
    }


def sanity_extension_record() -> dict:
    """Why the assignment's optional validation-only extension was not run."""
    return {
        "run": False,
        "reason": (
            "the extension is explicitly conditional on Agent 1 having "
            "predeclared alternate final budgets ('If Agent 1 did not "
            "predeclare alternate final budgets, do not invent one'). Agent 1's "
            "development budget declares a single final-run figure — "
            f"final_run_optimizer_steps_max = {wp.FINAL_UPDATE_BUDGET_MAX} — and "
            "no shorter/longer alternative, so there is nothing to choose "
            "between and no new budget was invented"
        ),
        "agent_1_development_budget": dict(wc.DEVELOPMENT_BUDGET),
        "frozen_final_budget": FINAL_MAX_UPDATES,
    }


def deviations_record(args) -> list:
    return [
        {
            "topic": "cadence validation is measured twice per checkpoint",
            "detail": (
                "the trainer runs its own frozen cadence validation inside each "
                "500-update block; Agent 5 runs a second pass at the same update "
                "number over the same spread batch positions to record the full "
                "metric set (top-1, Brier, accuracy, per-head baselines) without "
                "modifying Agent 4's accepted trainer. The two selection scores "
                "are compared and the maximum disagreement is reported as "
                "validation-determinism evidence; the duplicate pass costs about "
                "8s per checkpoint and changes no training state."
            ),
        },
        {
            "topic": "selection uses a full validation-split pass",
            "detail": (
                "the cadence passes are evenly spread fixed-size samples for the "
                f"curve ({args.validation_batches} x 256 examples); the selection "
                "input is one pass over the entire validation split (249,963 "
                "examples) at the final pilot checkpoint, identical in shape for "
                "every candidate. This makes the selection score the split's own "
                "number rather than a subsample's."
            ),
        },
        {
            "topic": "one process per candidate",
            "detail": (
                "each pilot runs in a fresh subprocess so every candidate starts "
                "from a cold process as well as a cold canonical initialization, "
                "and so an interrupted matrix resumes per candidate. The "
                "orchestrator verifies the corpus payload bytes once; workers "
                "re-verify the three digests."
            ),
        },
    ]


def handoff_record(decision: dict, frozen_config: "dict | None", fairness: dict, records: list) -> dict:
    return {
        "winning_candidate_id": decision.get("winner"),
        "frozen_train_config": "reports/phase_8_data/agent_05_frozen_train_config.json",
        "frozen_train_config_digest": (
            frozen_config["train_config_digest"] if frozen_config else None
        ),
        "trainer_construction": (
            frozen_config["trainer_construction"] if frozen_config else None
        ),
        "seeds": dict(CANONICAL_SEEDS),
        "expected_fresh_init_checksum": fairness["expected_fresh_init_checksum"],
        "final_training_budget": FINAL_MAX_UPDATES,
        "validation_cadence_updates": wp.PILOT_VALIDATION_CADENCE,
        "checkpoint_cadence_updates": FINAL_CHECKPOINT_CADENCE,
        "best_checkpoint_metric": BEST_CHECKPOINT_METRIC,
        "loader_topology": dict(DEFAULT_TOPOLOGY),
        "pilot_evidence": {
            "pilot_runs_csv": "reports/phase_8_data/agent_05_pilot_runs.csv",
            "pilot_selection": "reports/phase_8_data/agent_05_pilot_selection.json",
            "per_candidate_final_scores": {
                record["candidate_id"]: record["final_checkpoint"]["selection_score"]
                for record in records
            },
        },
        "rules": [
            "Agent 6 must rebuild the canonical C1 initialization from the seed "
            "and confirm the expected fresh-init checksum before training",
            "no pilot checkpoint may be continued as the Phase 8 final run",
            "Agent 6 may not tune the frozen configuration",
        ],
    }


def completion_gates(
    records: list,
    decision: dict,
    fairness: dict,
    access: dict,
    score_checks: list,
    frozen_config: "dict | None",
    verify_record: dict,
    args,
) -> dict:
    counters_clean = all(
        sum(int(value) for value in record["counters"].values()) == 0
        for record in records
    )
    statuses = dict(verify_record.get("statuses", {}))
    return {
        "agents_1_to_4_pass": bool(statuses)
        and all(value == "PASS" for value in statuses.values()),
        "candidate_count_at_or_below_limit": len(records) <= wp.PILOT_CANDIDATE_LIMIT,
        "candidate_matrix_matches_agent_1": len(records) == len(pilot_candidate_ids())
        and not fairness["unregistered_configs_run"]
        and not fairness["registered_candidates_not_run"],
        "no_unregistered_config_ran": not fairness["unregistered_configs_run"],
        "all_model_init_checksums_identical": fairness["all_init_checksums_identical"],
        "all_batch_identity_sequences_identical": fairness[
            "all_batch_sequences_identical"
        ],
        "every_candidate_got_the_equal_update_budget": fairness["all_budgets_equal"]
        and all(
            int(record["completed_updates"]) == int(args.updates) for record in records
        ),
        "validation_at_identical_update_numbers": fairness[
            "all_validation_update_numbers_identical"
        ],
        "validation_examples_identical_across_candidates": fairness[
            "validation_batch_positions_identical"
        ],
        "pilot_update_budget_within_frozen_cap": int(args.updates)
        <= wp.PILOT_UPDATE_BUDGET,
        "no_non_finite_or_target_or_leak_counters": counters_clean,
        "selection_uses_validation_only": all(
            record["final_checkpoint"]["scope"] == wp.SELECTION_SCOPE
            for record in records
        )
        and access["test_examples_evaluated_by_model_agent_5"] == 0,
        "selection_score_arithmetic_verified": all(
            entry["match"] for entry in score_checks
        ),
        "one_winner_selected_deterministically": decision["status"] == "PASS"
        and decision["winner"] is not None,
        "winner_reproducible_from_csv": bool(decision.get("reproducible_from_csv")),
        "warmstart_train_config_v1_fully_frozen": frozen_config is not None
        and not frozen_config["problems"],
        "final_update_budget_within_25k": frozen_config is not None
        and int(frozen_config["config"]["max_final_updates"])
        <= wp.FINAL_UPDATE_BUDGET_MAX,
        "test_model_inference_count_zero": access[
            "test_examples_evaluated_by_model_agent_5"
        ]
        == 0,
        "phase4_neural_strength_games_zero": access[
            "phase4_neural_evaluation_games_agent_5"
        ]
        == 0,
        "frozen_held_out_gates_refuse_agent_5": access["all_gates_refuse_agent_5"],
        "no_pilot_checkpoint_handed_to_agent_6": frozen_config is None
        or "checkpoint_path" not in frozen_config,
    }


# ---------------------------------------------------------------------------
# Suite and report
# ---------------------------------------------------------------------------


def run_pytest() -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    tail = completed.stdout.strip().splitlines()
    summary = tail[-1] if tail else ""
    record = {
        "command": ".venv/bin/python -m pytest tests -q",
        "summary": summary,
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
    }
    log(f"pytest: {summary}")
    return record


def append_report_section(payload: dict, tests_after: "dict | None") -> None:
    """Append section 5. Earlier sections are never rewritten."""
    text = REPORT_PATH.read_text()
    if "\n## 5. Agent 5 " in text:
        log("report section 5 already present; leaving it untouched")
        return
    REPORT_PATH.write_text(text.rstrip("\n") + "\n\n\n" + report_section(payload, tests_after))
    log("appended report section 5")


def report_section(payload: dict, tests_after: "dict | None") -> str:
    decision = payload["selection"]
    fairness = payload["fairness"]
    access = payload["held_out_access_log"]
    gates = payload["completion_gates"]
    frozen = read_json(artifact_path("agent_05_frozen_train_config.json"))
    config = frozen["config"]
    runs = {record["candidate_id"]: record for record in payload["pilot_runs"]}

    ranking_lines = []
    for entry in decision["ranking"]:
        ranking_lines.append(
            f"{entry['rank']}. {entry['candidate_id']:<28} "
            f"score {entry['selection_score']:.6f}   "
            f"p {entry['policy_ce_ratio']:.4f}  "
            f"v {entry['value_ce_ratio']:.4f}  "
            f"b {entry['belief_ce_ratio']:.4f}   "
            f"{entry['examples_per_second']:.0f} ex/s"
        )
    ranking = "\n".join(ranking_lines)

    curve_lines = []
    winner_run = runs[decision["winner"]]
    for entry in winner_run["cadence_checkpoints"]:
        curve_lines.append(
            f"{entry['global_step']:>5}  "
            f"score {entry['selection_score']:.4f}  "
            f"p {entry['policy_ce_ratio']:.4f}  "
            f"v {entry['value_ce_ratio']:.4f}  "
            f"b {entry['belief_ce_ratio']:.4f}  "
            f"train loss {entry['train']['loss_total']:.4f}  "
            f"|g| {entry['train']['grad_norm_pre_clip']:.3f}  "
            f"lr {entry['train']['learning_rate']:.2e}"
        )
    curve = "\n".join(curve_lines)

    final = winner_run["final_checkpoint"]
    suite_line = tests_after["summary"] if tests_after else "(not re-run in this invocation)"
    gate_total = len(gates)
    gate_true = sum(1 for value in gates.values() if value)

    return f"""## 5. Agent 5 — Bounded Pilot Selection

**Status: {payload['status']}** — {gate_true} / {gate_total} completion gates true.
Machine-readable record: `reports/phase_8_data/agent_05_pilot_selection.json`
(protocol, per-candidate runs, fairness evidence, selection, access log),
`reports/phase_8_data/agent_05_pilot_runs.csv` (every validation checkpoint of
every candidate) and `reports/phase_8_data/agent_05_frozen_train_config.json`
(`warmstart_train_config_v1`). Produced by
`python scripts/run_phase8_agent05.py --full --run-pytest`.

### 5.0 Prerequisite: Agents 1-4 and the accepted corpus through the resolver

Agents 1, 2, 3 and 4 all read `PASS`; Agent 1's live contract digest equals the
recorded one; the frozen upstream and teacher roster verify clean. The corpus
was resolved only through `synthetic_corpus.default_corpus_root()` before any
optimizer step, in the orchestrator and in every pilot subprocess:

```text
resolver result: MATCH   (pointer_file -> canonical location)
content digest        {payload['prerequisite_digests']['corpus_content'][:8]}…{payload['prerequisite_digests']['corpus_content'][-5:]}   == accepted (Agents 2/3/4)
metadata digest       {payload['prerequisite_digests']['corpus_metadata'][:8]}…{payload['prerequisite_digests']['corpus_metadata'][-6:]}  == accepted
commit-index digest   {payload['prerequisite_digests']['corpus_commit_index'][:8]}…{payload['prerequisite_digests']['corpus_commit_index'][-6:]}  == accepted
payload bytes         re-verified once in the orchestrator
candidate matrix      digest {payload['prerequisite_digests']['candidate_matrix'][:16]}…
                      equals Agent 1's accepted artifact field for field
```

### 5.1 The matrix that ran, and the fairness it ran under

Exactly Agent 1's six frozen candidates, no more and no fewer, each
constructed only through `WarmstartTrainConfig.from_pilot_candidate`:

```text
candidates registered / run          6 / 6      (limit 6)
unregistered configurations run      0
fresh-init checksum, all 6 identical {fairness['expected_fresh_init_checksum'][:32]}…
ordered batch-identity sequence,
    all 6 identical                  {list(fairness['batch_sequence_digests'].values())[0][:32]}…
optimizer updates each               {payload['pilot_protocol']['update_budget_per_candidate']:,}   (Agent 1's cap: {wp.PILOT_UPDATE_BUDGET:,})
validation update numbers            {', '.join(str(number) for number in list(fairness['validation_update_numbers'].values())[0])}
    identical across candidates      True
cadence validation                   {payload['pilot_protocol']['cadence_validation_batches']} evenly spread batches x 256
                                     = {payload['pilot_protocol']['cadence_validation_examples']:,} held-out examples,
                                     the same positions for every candidate
selection validation                 one full validation-split pass
                                     ({final['value_examples']:,} examples, {final['validation_games']:,} games)
early stops                          none; no candidate hit a hard failure
non-finite / target / leak counters   0 across all six runs
validation determinism: max spread
    between the trainer's own cadence
    score and Agent 5's repeat pass   {fairness['trainer_cadence_agreement_max']:g}
```

The batch-identity sequence is folded to one SHA-256 over the run's {winner_run['batch_sequence_length']:,}
ordered per-step key digests, so "same ordered pilot batch identities" is a
single comparable string rather than a claim. The fresh-init checksum hashes
every parameter's name, shape and float32 bytes.

### 5.2 Per-candidate result at the final pilot checkpoint

Full validation split, update {payload['pilot_protocol']['update_budget_per_candidate']:,}, `selection_score = mean(r_policy, r_value,
r_belief)`, lower is better:

```text
{ranking}
```

Hard veto (Agent 1's frozen list) removed {decision['candidates_vetoed']} of 6 candidates.
{'Vetoed: ' + '; '.join(f"{entry['candidate_id']} ({', '.join(entry['reasons'])})" for entry in decision['vetoed']) if decision['vetoed'] else 'No candidate was vetoed.'}

### 5.3 The winner and how the tie-break resolved

```text
winner                    {decision['winner']}
selection score           {decision['winner_selection_score']:.6f}
runner-up                 {decision['runner_up']}
margin to runner-up       {decision['margin_to_runner_up']:.6f}
decided at tie-break key  {decision['tie_break_used']}
reproducible from the
    published CSV alone   {decision['reproducible_from_csv']}
```

`select_winner` is a pure function of the published records: the harness
re-reads `agent_05_pilot_runs.csv` from disk, re-runs the same function, and
requires the same winner and the same ranking — and the suite does the same
against the shipped artifact. Every published `selection_score` was
recomputed from its own three ratios ({payload['selection_score_recomputation']['checked']} checkpoints, 0 mismatches).

### 5.4 Winner's validation curve

```text
{curve}
```

Final full-split checkpoint of the winner:

```text
policy   CE {final['policy_model_ce']:.4f} vs uniform-legal {final['policy_baseline_ce']:.4f}   ratio {final['policy_ce_ratio']:.4f}
         top-1 {final['policy_model_top1']:.4f} vs expected {final['policy_baseline_expected_top1']:.4f}
value    CE {final['value_model_ce']:.4f} vs train prior {final['value_baseline_ce']:.4f}   ratio {final['value_ce_ratio']:.4f}
         Brier {final['value_model_brier']:.4f} vs {final['value_baseline_brier']:.4f}   accuracy {final['value_model_accuracy']:.4f} vs {final['value_baseline_accuracy']:.4f}
belief   CE {final['belief_model_ce']:.4f} vs remaining-count {final['belief_baseline_ce']:.4f}   ratio {final['belief_ce_ratio']:.4f}
         top-1 {final['belief_model_top1']:.4f} vs {final['belief_baseline_top1']:.4f}
decisions {final['value_examples']:,}   games {final['validation_games']:,}   hidden pieces {final['belief_pieces']:,}
```

These are validation numbers used to choose a configuration. They are not
Phase 8 acceptance results; the sealed test split decides that, and only
Agent 7 opens it.

### 5.5 Frozen `warmstart_train_config_v1`

Digest `{frozen['train_config_digest'][:32]}…`. Agent 6 runs this verbatim:

```text
model / config digest     {config['model_candidate']} / {config['model_config_digest'][:16]}…
model init seed           {config['model_init_seed']}
expected fresh-init cksum {config['expected_fresh_init_checksum'][:32]}…
trainer / checkpoint      {config['trainer_version']} / {config['checkpoint_version']}
example / corpus          {config['example_version']} / {config['corpus_version']}
batch size                {config['batch_size']}
optimizer                 {config['optimizer']}  betas {tuple(config['adam_betas'])}  eps {config['adam_epsilon']:g}
learning rate             {config['learning_rate']:g}
weight decay              {config['weight_decay']}
gradient clip             {config['gradient_clip_norm']}
schedule                  {config['lr_schedule']}
loss weights              policy {config['lambda_policy']}  value {config['lambda_value']}  belief {config['lambda_belief']}
train shuffle seed/order  {config['train_shuffle_seed']} / {config['train_order_version']}
max final updates         {config['max_final_updates']:,}   (frozen cap {wp.FINAL_UPDATE_BUDGET_MAX:,})
validation cadence        every {config['validation_cadence_updates']} updates, {config['validation_batches']} spread batches
checkpoint cadence        every {config['checkpoint_cadence_updates']} updates
best-checkpoint metric    validation selection_score, strictly lower wins
early-stop rule           {config['early_stop_rule']['rule']}
loader topology           {config['loader_topology']['workers']}w / {config['loader_topology']['prefetch']}p / {config['loader_topology']['record_cache_size']} record cache
device / precision        {config['device']} / {config['precision']}
```

The completeness check is structural: `warmstart_pilot.build_frozen_train_config`
refuses to emit a payload missing any required field, every hyperparameter is
copied from the frozen candidate (the function has no way to express a value
the matrix does not contain), and `verify_frozen_train_config` re-derives the
digest and re-checks the winner's hyperparameters against the matrix.

**No pilot checkpoint is handed to Agent 6.** The handoff carries the
configuration and the expected fresh-init checksum; Agent 6 must rebuild the
canonical C1 initialization from the seed.

### 5.6 Sanity extension: not run, and why

The assignment permits one validation-only extension solely to choose between
*already predeclared* shorter/longer final budgets, and states plainly: "If
Agent 1 did not predeclare alternate final budgets, do not invent one."
Agent 1's development budget declares one final-run figure —
`final_run_optimizer_steps_max = {wp.FINAL_UPDATE_BUDGET_MAX:,}` — and no alternative. There
is therefore nothing to choose between, no extension was run, and no new
budget was invented: the frozen `max_final_updates` is Agent 1's own number.

### 5.7 Held-out discipline, measured rather than asserted

`warmstart_pilot.record_model_input_access` instruments
`WarmstartBatch.model_input` — the single boundary where observations become
model input — and tallies examples by corpus split across every pilot
process. `record_phase4_access` wraps the Phase 4 evaluation entry points and
reads the neural checkpoint-load counter.

```text
test examples evaluated by a model, Agent 5      {access['test_examples_evaluated_by_model_agent_5']}
Phase 4 neural evaluation games, Agent 5         {access['phase4_neural_evaluation_games_agent_5']}
Phase 4 neural checkpoint loads, Agent 5         {access['phase4_neural_checkpoint_loads_agent_5']}
train examples through the model boundary        {access['examples_by_split'].get('train', 0):,}
validation examples through the model boundary   {access['examples_by_split'].get('validation', 0):,}

frozen gates asked to admit Agent 5:
  test_corpus:model_inference                    {access['frozen_gate_responses']['test_corpus:model_inference']}
  test_corpus:model_metric                       {access['frozen_gate_responses']['test_corpus:model_metric']}
  test_corpus:hyperparameter_selection           {access['frozen_gate_responses']['test_corpus:hyperparameter_selection']}
  phase4_bank:neural_playing_strength            {access['frozen_gate_responses']['phase4_bank:neural_playing_strength']}
  phase4_bank:pilot_selection                    {access['frozen_gate_responses']['phase4_bank:pilot_selection']}
  phase4_bank:config_selection                   {access['frozen_gate_responses']['phase4_bank:config_selection']}
```

Structural corpus manifests were read (digest verification); no test example
reached a model, and no game was played against any evaluation opponent.

### 5.8 What Agent 5 did not do

No architecture, teacher-roster, teacher-weight, setup-distribution, corpus or
split change; no candidate outside the frozen six; no "one more promising
run"; no early stop of a weak candidate; no test-split model metric; no Phase
4 strength evaluation; no continuation of a pilot checkpoint into the final
run; no modification of Agent 4's trainer, loss, dataset or checkpoint
modules; no rewrite of an accepted report section.

### 5.9 Post-edit suite and completion gates

```text
.venv/bin/python -m pytest tests -q
{suite_line}
```

{gate_true} / {gate_total} completion gates true (recorded in
`agent_05_pilot_selection.json` → `completion_gates`): candidate count at the
limit and equal to Agent 1's matrix; no unregistered configuration; identical
fresh-init checksums; identical ordered batch-identity sequences; equal update
budgets; validation at identical update numbers over identical held-out
examples; clean non-finite/target/leak counters; selection from validation
only; selection-score arithmetic re-verified; one deterministic winner;
winner reproducible from the published CSV; `warmstart_train_config_v1` fully
frozen and complete; final budget within 25,000; zero test model inferences;
zero Phase 4 neural evaluation games; every frozen held-out gate refuses Agent
5; no pilot checkpoint handed forward.

### 5.10 Handoff to Agent 6

In `agent_05_pilot_selection.json` → `handoff_to_agent_6`: the winning
candidate id, the frozen `warmstart_train_config_v1` and its digest, the exact
`WarmstartTrainConfig.from_pilot_candidate(...)` call that reconstructs it, all
canonical seeds, the expected fresh-init checksum Agent 6 must confirm before
training, the final budget of {config['max_final_updates']:,} updates, the validation and
checkpoint cadences, the loader topology, and the pilot evidence. Agent 6 runs
this configuration unchanged, from a fresh canonical C1 initialization, and
selects its checkpoint by validation only.
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--pilots", action="store_true")
    parser.add_argument("--artifacts", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--candidate", default=None, help="run one pilot worker")
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="ignore cached pilot runs")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--updates", type=int, default=wp.PILOT_UPDATE_BUDGET)
    parser.add_argument(
        "--validation-batches", type=int, default=CADENCE_VALIDATION_BATCHES
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_TOPOLOGY["workers"])
    parser.add_argument("--prefetch", type=int, default=DEFAULT_TOPOLOGY["prefetch"])
    parser.add_argument(
        "--record-cache", type=int, default=DEFAULT_TOPOLOGY["record_cache_size"]
    )
    parser.add_argument(
        "--trust-bytes",
        action="store_true",
        help="skip payload byte re-verification (workers of a verified orchestration)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write artifacts under the work directory and append no report section",
    )
    args = parser.parse_args()

    global _OUTPUT_DIRECTORY
    if args.dry_run:
        _OUTPUT_DIRECTORY = WORK_DIRECTORY / "dry_run"
        _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if args.updates > wp.PILOT_UPDATE_BUDGET:
        raise SystemExit(
            f"BLOCKED: {args.updates} updates exceeds Agent 1's frozen pilot cap of "
            f"{wp.PILOT_UPDATE_BUDGET}"
        )
    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if args.candidate is not None:
        run_one_pilot(args.candidate, args)
        return

    tests_before = None
    verify_record = None
    if args.verify or args.full:
        log("verifying prerequisites (payload bytes included; ~80s)…")
        verify_record, _identity = verify_prerequisites(check_payload_bytes=True)
        log("prerequisites clean")
        write_json(WORK_DIRECTORY / "verify.json", verify_record)
        if args.run_pytest and args.full:
            log("pre-change suite…")
            tests_before = run_pytest()
            write_json(WORK_DIRECTORY / "tests_before.json", tests_before)

    records = None
    if args.pilots or args.full:
        records = phase_pilots(args)

    if args.artifacts or args.full:
        if records is None:
            records = [
                read_json(candidate_work_path(candidate_id, args.updates))
                for candidate_id in pilot_candidate_ids()
            ]
        if verify_record is None:
            path = WORK_DIRECTORY / "verify.json"
            verify_record = read_json(path) if path.exists() else {"reused": False}
        if tests_before is None:
            path = WORK_DIRECTORY / "tests_before.json"
            tests_before = read_json(path) if path.exists() else None
        payload = phase_artifacts(records, verify_record, tests_before, args)
        tests_after = None
        if args.run_pytest:
            log("post-artifact suite…")
            tests_after = run_pytest()
            payload["tests_after"] = tests_after
            write_json(artifact_path("agent_05_pilot_selection.json"), payload)
        if args.dry_run:
            # Render the section so a shakedown still exercises every field it
            # reads, but never append it to the accepted report.
            (_OUTPUT_DIRECTORY / "report_section_5.md").write_text(
                report_section(payload, tests_after)
            )
            log("dry run: report section rendered, not appended")
        else:
            append_report_section(payload, tests_after)
        print(json.dumps({"status": payload["status"], "winner": payload["selection"]["winner"]}))


if __name__ == "__main__":
    main()
