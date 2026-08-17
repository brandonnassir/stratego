#!/usr/bin/env python3
"""Phase 9 Agent 5 acceptance harness.

Stages:

```text
verify       Agents 1-4 acceptance, corpus resolver, Phase 8 identity, storage
contract     the trainer/loss/checkpoint contract document
resume       CPU bit-exact + MPS backend-aware resume validation
soak         the >= 2,000-update non-selection infrastructure soak
binding      two genuinely different checkpoints in one historical matchup
archive      a real namespace-local immutable H005, produced and bound
throughput   complete iteration wall time, split by phase, plus topology
artifacts    gates, artifacts, report section
```

Nothing here selects a configuration. The soak runs the neutral middle
candidate's numbers under `SCOPE_SOAK`, opens no validation bank, computes no
score, and its weights are deliberately left where Agent 6 cannot inherit them.
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
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from stratego.model.policy_adapter import prepare_legality  # noqa: E402
from stratego.training import phase9_behavior as pb  # noqa: E402
from stratego.training import phase9_checkpoint as pck  # noqa: E402
from stratego.training import phase9_collector as pc  # noqa: E402
from stratego.training import phase9_loss as pl  # noqa: E402
from stratego.training import phase9_rollout_store as store  # noqa: E402
from stratego.training import phase9_storage as ps  # noqa: E402
from stratego.training import phase9_trainer as pt  # noqa: E402
from stratego.training import synthetic_corpus as sc  # noqa: E402
from stratego.training.phase9_contract import (  # noqa: E402
    ARCHIVE_CADENCE_ITERATIONS,
    CLIP_FRACTION_HARD_LIMIT,
    EXPECTED_CORPUS_COMMIT_INDEX_DIGEST,
    EXPECTED_CORPUS_CONTENT_DIGEST,
    EXPECTED_CORPUS_METADATA_DIGEST,
    EXPECTED_CORPUS_VERSION,
    EXPECTED_PHASE8_CHECKPOINT_PATH,
    EXPECTED_PHASE8_CHECKPOINT_SHA256,
    KL_HARD_LIMIT,
    PHASE9_POPULATION_VERSION,
    PHASE9_ROLLOUT_SCHEDULE_VERSION,
    PILOT_CANDIDATES,
    active_historical_window,
    archive_snapshot_id,
    contract_digest,
    iter_scheduled_games,
)
from stratego.training.phase9_schedule import (  # noqa: E402
    ActiveHistoryManifest,
    behavior_snapshot_identity,
    historical_policy_token,
    rebuild_scheduled_game,
)
from stratego.training.phase9_targets import example_contract_digest  # noqa: E402
from stratego.training.warmstart_checkpoint import (  # noqa: E402
    CorpusIdentity,
    verify_corpus_identity,
)

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_9_data"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase9" / "agent05"

#: The soak's archive lives under Agent 5's own work directory, not under the
#: production `checkpoints/phase9/archive/`. The archive member it writes is a
#: real immutable namespace-local `H005`, but it belongs to an infrastructure
#: soak — Agent 6's pilots must find their own `pilot_p9c/H005` slot empty.
ARCHIVE_ROOT = WORK_DIRECTORY / "archive"

#: The resume experiments read Agent 3's accepted sealed pilot rollout. It is
#: never written to: every bind in this harness passes `mark_training=False`
#: except the soak, which owns its own root.
RESUME_SOURCE_ROOT = "agent_03_soak"
RESUME_NAMESPACE = "pilot_p9c"
RESUME_ITERATION = 1

#: The soak's own root, beside Agent 3's and outside every production
#: namespace directory, so nothing Agent 6 or 7 collects can inherit it.
SOAK_ROOT_NAME = "agent_05_soak"
SOAK_NAMESPACE = "pilot_p9c"
SOAK_ITERATIONS = 5
SOAK_MINIMUM_UPDATES = 2000

ACCEPTED_CORPUS = CorpusIdentity(
    corpus_version=EXPECTED_CORPUS_VERSION,
    content_digest=EXPECTED_CORPUS_CONTENT_DIGEST,
    metadata_digest=EXPECTED_CORPUS_METADATA_DIGEST,
    commit_index_digest=EXPECTED_CORPUS_COMMIT_INDEX_DIGEST,
)

ACCEPTED_CONTRACT_DIGEST = (
    "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"
)
ACCEPTED_EXAMPLE_DIGEST = (
    "a6b17a94449ab764d4b5dd054d677096adfa70c52631865499a60a7a3f44af61"
)

#: The resume acceptance measurements, frozen before they are used as a gate.
RESUME_TOLERANCES = {
    "cpu_requires_bitwise_equality": True,
    "mps_first_post_resume_step_rtol": 1e-5,
    "mps_first_post_resume_step_atol": 1e-6,
    "mps_envelope_ratio_limit": 10.0,
}

BENCHMARK_COLUMNS = (
    "measurement",
    "namespace",
    "iteration",
    "workers",
    "prefetch",
    "updates",
    "games",
    "learner_decisions",
    "collection_seconds",
    "sealing_audit_seconds",
    "target_construction_seconds",
    "data_wait_seconds",
    "host_to_device_seconds",
    "forward_seconds",
    "loss_seconds",
    "backward_seconds",
    "optimizer_seconds",
    "checkpoint_seconds",
    "validation_infrastructure_seconds",
    "train_seconds",
    "iteration_seconds",
    "examples_per_second",
    "updates_per_second",
    "unsynchronized_examples_per_second",
    "peak_rss_mib",
    "peak_mps_mib",
)


def log(message: str) -> None:
    print(f"[agent05] {message}", flush=True)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


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
        "torch_version": str(torch.__version__),
        "mps_available": bool(torch.backends.mps.is_available()),
        "source_revision": git_output("rev-parse", "--short", "HEAD"),
        "working_tree_state": "dirty" if git_output("status", "--porcelain") else "clean",
    }


def peak_rss_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kibibytes.
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


#: Torch reports MPS driver allocation *now*, not a high-water mark, so the
#: maximum has to be accumulated by sampling. `getrusage` by contrast is a true
#: peak, which is why the two are reported by different mechanisms.
_MPS_HIGH_WATER = [0.0]


def peak_mps_mib() -> float:
    """The largest MPS driver allocation observed so far, in MiB."""
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            current = float(torch.mps.driver_allocated_memory()) / (1024 * 1024)
        except Exception:  # noqa: BLE001 - absent counters are not a failure
            return _MPS_HIGH_WATER[0]
        _MPS_HIGH_WATER[0] = max(_MPS_HIGH_WATER[0], current)
    return _MPS_HIGH_WATER[0]


def percentile(values, fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[index])


# ---------------------------------------------------------------------------
# Stage: verify
# ---------------------------------------------------------------------------


def verify_prerequisites() -> dict:
    """Agents 1-4 `PASS` and formal acceptance, plus every frozen digest."""
    problems: list[str] = []
    acceptances = {}
    for agent in (1, 2, 3, 4):
        path = DATA_DIRECTORY / f"agent_{agent:02d}_acceptance.json"
        if not path.exists():
            problems.append(f"agent {agent} acceptance artifact is missing")
            continue
        payload = read_json(path)
        acceptances[agent] = {
            "status": payload.get("status"),
            "gates_passed": payload.get("gates", {}).get("passed"),
            "gates_total": payload.get("gates", {}).get("total"),
        }
        if payload.get("status") != "PASS":
            problems.append(f"agent {agent} status is {payload.get('status')!r}, not PASS")

    observed_contract = contract_digest()
    observed_example = example_contract_digest()
    if observed_contract != ACCEPTED_CONTRACT_DIGEST:
        problems.append(
            f"contract digest {observed_contract} != accepted {ACCEPTED_CONTRACT_DIGEST}"
        )
    if observed_example != ACCEPTED_EXAMPLE_DIGEST:
        problems.append(
            f"example contract digest {observed_example} != accepted {ACCEPTED_EXAMPLE_DIGEST}"
        )

    checkpoint = REPOSITORY_ROOT / EXPECTED_PHASE8_CHECKPOINT_PATH
    observed_sha = pb.file_sha256(checkpoint) if checkpoint.exists() else "<missing>"
    if observed_sha != EXPECTED_PHASE8_CHECKPOINT_SHA256:
        problems.append(
            f"Phase 8 checkpoint SHA-256 {observed_sha} != accepted "
            f"{EXPECTED_PHASE8_CHECKPOINT_SHA256}"
        )
    return {
        "acceptances": acceptances,
        "contract_digest": observed_contract,
        "contract_digest_matches_accepted": observed_contract == ACCEPTED_CONTRACT_DIGEST,
        "example_contract_digest": observed_example,
        "example_contract_digest_matches_accepted": (
            observed_example == ACCEPTED_EXAMPLE_DIGEST
        ),
        "phase8_checkpoint_sha256": observed_sha,
        "phase8_checkpoint_matches_accepted": (
            observed_sha == EXPECTED_PHASE8_CHECKPOINT_SHA256
        ),
        "problems": problems,
    }


def verify_corpus(*, check_payload_bytes: bool) -> dict:
    """The mandatory resolver check: resolver first, digests second, path never."""
    resolved = sc.default_corpus_root()
    problems: list[str] = []
    observed = None
    try:
        observed = verify_corpus_identity(
            resolved, ACCEPTED_CORPUS, check_payload_bytes=check_payload_bytes
        )
    except Exception as error:  # noqa: BLE001 - a corpus mismatch is BLOCKED
        problems.append(f"{type(error).__name__}: {error}")

    modules = [
        "stratego/training/phase9_loss.py",
        "stratego/training/phase9_trainer.py",
        "stratego/training/phase9_checkpoint.py",
    ]
    hard_coded = []
    for relative in modules:
        text = (REPOSITORY_ROOT / relative).read_text()
        if "/Volumes/" in text or "data/stratego_phase8" in text:
            hard_coded.append(relative)
    if hard_coded:
        problems.append(f"modules hard-code an absolute data path: {hard_coded}")
    return {
        "resolver": "stratego.training.synthetic_corpus.default_corpus_root()",
        "resolved_root": str(resolved),
        "accepted_identity": ACCEPTED_CORPUS.to_dict(),
        "observed_identity": observed.to_dict() if observed else None,
        "identity_matches": observed == ACCEPTED_CORPUS if observed else False,
        "payload_bytes_checked": bool(check_payload_bytes),
        "modules_scanned": modules,
        "modules_hard_coding_absolute_paths": hard_coded,
        "identity_rule": (
            "corpus identity is version + accepted digests, not filesystem "
            "location; a digest mismatch is BLOCKED and never repaired"
        ),
        "problems": problems,
    }


def verify_storage() -> dict:
    """The Phase 9 rollout root, resolved rather than hard-coded."""
    root = ps.default_rollout_root()
    description = ps.describe_rollout_root()
    return {
        "resolver": "stratego.training.phase9_storage.default_rollout_root()",
        "resolved_root": str(root),
        "description": description,
        "resume_source": str(root / RESUME_SOURCE_ROOT / RESUME_NAMESPACE),
        "soak_root": str(root / SOAK_ROOT_NAME),
        "identity_rule": ps.STORAGE_IDENTITY_RULE,
    }


# ---------------------------------------------------------------------------
# Stage: contract
# ---------------------------------------------------------------------------


def build_contract(verification: dict, corpus: dict, storage: dict) -> dict:
    candidates = [
        {
            **dict(candidate),
            "train_config_digest": pt.Phase9TrainConfig.for_candidate(
                candidate["candidate_id"], device="mps"
            ).digest(),
        }
        for candidate in PILOT_CANDIDATES
    ]
    soak_config = pt.Phase9TrainConfig.for_soak(
        namespace=SOAK_NAMESPACE, device="mps", total_iterations=SOAK_ITERATIONS
    )
    return {
        "phase": 9,
        "agent": 5,
        "artifact": "agent_05_trainer_contract",
        **environment_record(),
        "prerequisites": verification,
        "corpus": corpus,
        "storage": storage,
        "trainer": pt.trainer_semantics(),
        "loss": pl.loss_semantics(),
        "checkpoint": pck.checkpoint_semantics(),
        "pilot_candidate_constructor": {
            "call": (
                "phase9_trainer.Phase9TrainConfig.for_candidate(candidate_id, "
                "device='mps', total_iterations=8)"
            ),
            "candidates": candidates,
            "rule": (
                "the learning rate and initial KL beta are read from Agent 1's "
                "frozen matrix; any other pair raises outside SCOPE_UNIT_TEST"
            ),
        },
        "trainer_constructor": {
            "fresh": (
                "phase9_trainer.Phase9Trainer.from_phase8_checkpoint(path, config, "
                "corpus_identity, topology=LoaderTopology(...))"
            ),
            "resume": (
                "phase9_trainer.Phase9Trainer.resume(path, config=config, "
                "corpus_identity=..., expected_sealed_rollout_digest=..., ...)"
            ),
            "sealed_rollout_consumption": (
                "phase9_trainer.bind_sealed_rollout(root, namespace, iteration, "
                "behavior_snapshot=..., expected_model_state_digest=...) then "
                "Phase9Trainer.bind_iteration(rollout)"
            ),
            "epochs": "Phase9Trainer.train_iteration() runs the frozen two epochs",
            "behavior_snapshot": (
                "Phase9Trainer.save_behavior_snapshot(path, "
                "logical_identity='B00N', rl_iteration=N)"
            ),
            "archive": (
                "phase9_checkpoint.write_archive_member("
                "Phase9Trainer.archive_member_payload(local_identity='H005'), "
                "root, namespace=..., local_identity='H005') then "
                "bind_archive_member(member)"
            ),
        },
        "soak_configuration": {
            **soak_config.identity(),
            "train_config_digest": soak_config.digest(),
            "selection_role": "none — SCOPE_SOAK is not a pilot candidate run",
        },
        "hard_veto_counters": {
            "raised_by_the_trainer": {
                "non_finite_loss": "counters['non_finite_losses'], and the update raises",
                "non_finite_gradient": "counters['non_finite_gradients']",
                "non_finite_parameter": "counters['non_finite_parameters']",
                "behavior_identity_mismatch": (
                    "counters['behavior_identity_mismatches'] — a minibatch whose "
                    "behavior checkpoint is not the bound iteration's"
                ),
                "rollout_identity_mismatch": "counters['rollout_identity_mismatches']",
                "target_reconstruction_mismatch": (
                    "counters['illegal_targets'] — every Phase9LossError, including "
                    "an illegal action, a behavior row off the simplex and a "
                    "dense/scalar disagreement"
                ),
                "checkpoint_resume_failure": "counters['checkpoint_errors']",
                "mean_iteration_or_epoch_kl": (
                    f"counters['kl_hard_limit_breaches']; the epoch raises above "
                    f"{KL_HARD_LIMIT}"
                ),
                "iteration_ppo_clip_fraction": (
                    f"counters['clip_fraction_hard_limit_breaches']; the epoch "
                    f"raises above {CLIP_FRACTION_HARD_LIMIT}"
                ),
            },
            "raised_elsewhere": {
                "illegal_neural_action": "Agent 3's collector, at decision time",
                "observer_safety_failure": "Agent 3's observer probe, at collection",
                "validation_random_ewr": "Agent 6's validation pass",
                "validation_basic_ewr": "Agent 6's validation pass",
            },
            "rule": (
                "every trainer-side veto is a raise, not a logged number: the "
                "counter records that it happened and the run stops"
            ),
        },
        "resume_acceptance_criterion": RESUME_TOLERANCES,
        "forbidden_here": [
            "the six-pilot matrix",
            "choosing a winner",
            "altering learning rates/betas beyond the frozen test fixture",
            "altering PPO clip, lambdas or loss weights",
            "using stale rollouts in later iterations",
            "opening the final-test bank",
            "continuing a soak checkpoint into Agent 6 or 7",
        ],
    }


# ---------------------------------------------------------------------------
# Resume experiment
# ---------------------------------------------------------------------------


def rows_path(role: str, device: str) -> Path:
    return WORK_DIRECTORY / f"resume_{device}_{role}_rows.json"


def parameters_path(device: str, tag: str, when: str) -> Path:
    return WORK_DIRECTORY / f"resume_{device}_{tag}_params_{when}.pt"


def resume_config(device: str) -> pt.Phase9TrainConfig:
    """The neutral configuration, used only as an experiment fixture."""
    return pt.Phase9TrainConfig.for_soak(
        namespace=RESUME_NAMESPACE, device=device, total_iterations=SOAK_ITERATIONS
    )


def resume_rollout():
    root = ps.default_rollout_root() / RESUME_SOURCE_ROOT
    return pt.bind_sealed_rollout(
        root, RESUME_NAMESPACE, RESUME_ITERATION, require_full_schedule=True
    )


def resume_worker(role: str, args) -> None:
    """One leg of the split-run experiment (or a control), in its own process.

    Every leg snapshots parameters twice: immediately after the first
    post-split update, and at the end. The early snapshot is what isolates the
    resume boundary itself from the backend's own run-to-run divergence
    accumulating over the remaining updates.
    """
    device = args.device
    total = args.resume_updates
    split_at = args.resume_split_at
    if device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    config = resume_config(device)
    rollout = resume_rollout()
    checkpoint_path = WORK_DIRECTORY / f"resume_{device}_split_{split_at}.ckpt"
    topology = pt.LoaderTopology(
        workers=args.workers, prefetch=args.prefetch, record_cache_size=args.record_cache
    )

    def fresh():
        trainer = pt.Phase9Trainer.from_phase8_checkpoint(
            REPOSITORY_ROOT / EXPECTED_PHASE8_CHECKPOINT_PATH,
            config,
            ACCEPTED_CORPUS,
            topology=topology,
            run_label=f"resume_{device}_{role}",
        )
        trainer.bind_iteration(rollout, mark_training=False)
        return trainer

    def run_with_snapshots(trainer, before_snapshot: int, tag: str):
        rows = trainer.train_iteration(
            updates=before_snapshot, capture_batch_digests=True
        )
        torch.save(trainer.parameter_snapshot(), parameters_path(device, tag, "early"))
        rows += trainer.train_iteration(
            updates=total - trainer.global_step, capture_batch_digests=True
        )
        torch.save(trainer.parameter_snapshot(), parameters_path(device, tag, "end"))
        return rows, trainer.state_summary()

    if role in ("straight", "control-a", "control-b"):
        tag = role.replace("-", "_")
        trainer = fresh()
        with trainer:
            rows, summary = run_with_snapshots(trainer, split_at + 1, tag)
    elif role == "split-first":
        trainer = fresh()
        with trainer:
            rows = trainer.train_iteration(
                updates=split_at, capture_batch_digests=True
            )
            trainer.save_checkpoint(checkpoint_path)
            summary = trainer.state_summary()
            # Donor continuation: this process holds the exact bit state the
            # checkpoint was written from, so its own steps split+1.. are the
            # only trajectory the resumed process can be compared against
            # without inheriting the backend's independent-prefix divergence.
            run_with_snapshots(trainer, 1, "donor")
    elif role == "split-resume":
        trainer = pt.Phase9Trainer.resume(
            checkpoint_path,
            config=config,
            corpus_identity=ACCEPTED_CORPUS,
            topology=topology,
            run_label=f"resume_{device}_split_resume",
            expected_sealed_rollout_digest=rollout.sealed_rollout_digest,
            expected_behavior_checkpoint_sha256=rollout.behavior_checkpoint_sha256,
        )
        trainer.rebind_iteration(rollout)
        with trainer:
            rows, summary = run_with_snapshots(trainer, 1, "resumed")
    else:
        raise SystemExit(f"unknown resume role {role!r}")

    write_json(
        rows_path(role, device),
        {
            "role": role,
            "device": device,
            "rows": [
                {
                    "global_optimizer_step": row["global_optimizer_step"],
                    "epoch": row["epoch"],
                    "minibatch_index": row["minibatch_index"],
                    "batch_digest": row["batch_digest"],
                    "learning_rate": row["learning_rate"],
                    "loss_total": row["loss_total"],
                    "behavior_kl": row["behavior_kl"],
                    "kl_beta": row["kl_beta"],
                    "grad_norm_pre_clip": row["grad_norm_pre_clip"],
                }
                for row in rows
            ],
            "state_summary": summary,
        },
    )
    log(f"resume worker {role} ({device}) finished: {len(rows)} updates")


def spawn_resume_worker(role: str, args, device: str) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--resume-worker",
        role,
        "--device",
        device,
        "--resume-updates",
        str(args.resume_updates if device == "mps" else args.cpu_updates),
        "--resume-split-at",
        str(args.resume_split_at if device == "mps" else args.cpu_split_at),
        "--workers",
        str(args.workers if device == "mps" else min(args.workers, 4)),
        "--prefetch",
        str(args.prefetch),
        "--record-cache",
        str(args.record_cache),
        "--cpu-threads",
        str(args.cpu_threads),
    ]
    log(f"spawning resume worker: {role} on {device}")
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"resume worker {role} ({device}) failed: {completed.returncode}")


def compare_parameters(path_a: Path, path_b: Path) -> dict:
    """Per-tensor closeness of two parameter snapshots, with worst offenders."""
    left = torch.load(path_a, weights_only=True)
    right = torch.load(path_b, weights_only=True)
    exact = True
    close = True
    max_abs = 0.0
    worst = []
    for name, tensor in left.items():
        other = right[name]
        difference = (other - tensor).abs()
        magnitude = float(difference.max()) if difference.numel() else 0.0
        max_abs = max(max_abs, magnitude)
        if not torch.equal(other, tensor):
            exact = False
        if not torch.allclose(other, tensor, rtol=1e-5, atol=1e-6):
            close = False
        worst.append((magnitude, name))
    worst.sort(reverse=True)
    return {
        "all_exactly_equal": exact,
        "all_allclose_rtol1e-5_atol1e-6": close,
        "max_abs_diff": max_abs,
        "worst_tensors": [
            {"name": name, "max_abs_diff": magnitude} for magnitude, name in worst[:3]
        ],
        "tensors_compared": len(left),
    }


def compare_resume(device: str, total: int, split_at: int, *, require_exact: bool) -> dict:
    """Assemble one device's complete resume evidence."""
    straight = read_json(rows_path("straight", device))
    first = read_json(rows_path("split-first", device))
    resumed = read_json(rows_path("split-resume", device))
    control_a = read_json(rows_path("control-a", device))
    control_b = read_json(rows_path("control-b", device))

    split_rows = first["rows"] + resumed["rows"]
    straight_rows = straight["rows"]
    digests_equal = [
        left["batch_digest"] == right["batch_digest"]
        for left, right in zip(straight_rows, split_rows)
    ]
    rates_equal = [
        left["learning_rate"] == right["learning_rate"]
        for left, right in zip(straight_rows, split_rows)
    ]
    steps_equal = [
        left["global_optimizer_step"] == right["global_optimizer_step"]
        for left, right in zip(straight_rows, split_rows)
    ]
    positions_equal = [
        (left["epoch"], left["minibatch_index"])
        == (right["epoch"], right["minibatch_index"])
        for left, right in zip(straight_rows, split_rows)
    ]
    exact_next = (
        len(resumed["rows"]) > 0
        and len(straight_rows) > split_at
        and resumed["rows"][0]["batch_digest"] == straight_rows[split_at]["batch_digest"]
        and resumed["rows"][0]["global_optimizer_step"] == split_at + 1
    )

    resumed_summary = dict(resumed["state_summary"])
    straight_summary = dict(straight["state_summary"])
    comparable = (
        "global_optimizer_step",
        "examples_consumed",
        "rl_iteration",
        "minibatch_cursor",
        "kl_beta",
        "kl_controller_updates",
        "kl_controller_history",
        "entropy_schedule_position",
        "learning_rate",
        "scheduler_last_epoch",
        "optimizer_state_structure",
        "counters",
        "sealed_rollout_digest",
        "train_config_digest",
    )
    logical_equal = {
        field: resumed_summary.get(field) == straight_summary.get(field)
        for field in comparable
    }

    evidence = {
        "device": device,
        "total_updates": total,
        "split_at": split_at,
        "compared_steps": len(digests_equal),
        "batch_identities_equal_every_step": all(digests_equal),
        "learning_rates_equal_every_step": all(rates_equal),
        "global_steps_equal_every_step": all(steps_equal),
        "cursor_positions_equal_every_step": all(positions_equal),
        "exact_next_batch_after_resume": bool(exact_next),
        "logical_state_summaries_equal": all(logical_equal.values()),
        "logical_state_fields": logical_equal,
        "parameters_first_post_resume_step": compare_parameters(
            parameters_path(device, "donor", "early"),
            parameters_path(device, "resumed", "early"),
        ),
        "parameters_end_vs_donor": compare_parameters(
            parameters_path(device, "donor", "end"),
            parameters_path(device, "resumed", "end"),
        ),
        "parameters_end_vs_independent_straight": compare_parameters(
            parameters_path(device, "straight", "end"),
            parameters_path(device, "resumed", "end"),
        ),
        "backend_control": {
            "early": compare_parameters(
                parameters_path(device, "control_a", "early"),
                parameters_path(device, "control_b", "early"),
            ),
            "end": compare_parameters(
                parameters_path(device, "control_a", "end"),
                parameters_path(device, "control_b", "end"),
            ),
            "role": (
                "two fresh identical runs with no checkpoint anywhere; the "
                "backend's own run-to-run envelope"
            ),
        },
    }
    control_end = evidence["backend_control"]["end"]["max_abs_diff"]
    donor_end = evidence["parameters_end_vs_donor"]["max_abs_diff"]
    evidence["envelope_ratio_end_vs_donor_over_control"] = (
        donor_end / control_end if control_end > 0 else (0.0 if donor_end == 0 else None)
    )
    evidence["backend_is_run_to_run_deterministic"] = bool(
        evidence["backend_control"]["end"]["all_exactly_equal"]
    )
    evidence["requires_bitwise_equality"] = bool(require_exact)
    if require_exact:
        evidence["passed"] = bool(
            evidence["batch_identities_equal_every_step"]
            and evidence["exact_next_batch_after_resume"]
            and evidence["logical_state_summaries_equal"]
            and evidence["parameters_end_vs_donor"]["all_exactly_equal"]
            and evidence["parameters_end_vs_independent_straight"]["all_exactly_equal"]
        )
    else:
        ratio = evidence["envelope_ratio_end_vs_donor_over_control"]
        evidence["passed"] = bool(
            evidence["batch_identities_equal_every_step"]
            and evidence["exact_next_batch_after_resume"]
            and evidence["logical_state_summaries_equal"]
            and evidence["parameters_first_post_resume_step"][
                "all_allclose_rtol1e-5_atol1e-6"
            ]
            and ratio is not None
            and ratio <= RESUME_TOLERANCES["mps_envelope_ratio_limit"]
        )
    return evidence


def run_resume_stage(args) -> dict:
    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    devices = []
    if not args.skip_cpu_resume:
        devices.append(("cpu", args.cpu_updates, args.cpu_split_at, True))
    if not args.skip_mps_resume:
        devices.append(("mps", args.resume_updates, args.resume_split_at, False))
    results = {}
    for device, total, split_at, require_exact in devices:
        for role in ("straight", "split-first", "split-resume", "control-a", "control-b"):
            spawn_resume_worker(role, args, device)
        results[device] = compare_resume(
            device, total, split_at, require_exact=require_exact
        )
        log(
            f"{device} resume: passed={results[device]['passed']} "
            f"envelope_ratio={results[device]['envelope_ratio_end_vs_donor_over_control']}"
        )
    rollout = resume_rollout()
    return {
        "phase": 9,
        "agent": 5,
        "artifact": "agent_05_resume_validation",
        **environment_record(),
        "criterion": {
            "criterion_id": "phase9_backend_aware_resume_equivalence_v1",
            "inherits": (
                "the Phase 8 Agent 4 reviewer-approved backend-aware principle: "
                "exact logical state everywhere, bitwise equality where the "
                "backend allows it, and an immediate resumed-vs-donor boundary "
                "check plus a measured no-checkpoint control envelope on MPS"
            ),
            "tolerances": RESUME_TOLERANCES,
            "why_not_independent_bit_determinism": (
                "two fresh identical MPS runs with no checkpoint anywhere "
                "already diverge, so an independent-run comparison measures "
                "backend determinism rather than checkpoint fidelity; the "
                "control legs measure that envelope directly"
            ),
        },
        "rollout": rollout.to_dict(),
        "devices": results,
        "cpu_resume_pass": results.get("cpu", {}).get("passed"),
        "mps_resume_pass": results.get("mps", {}).get("passed"),
    }


# ---------------------------------------------------------------------------
# Stage: soak
# ---------------------------------------------------------------------------


def soak_participants(behavior_snapshot, historical: dict) -> pc.IterationParticipants:
    return pc.IterationParticipants(behavior=behavior_snapshot, historical=historical)


def run_soak(args) -> dict:
    """The non-selection infrastructure soak: >= 2,000 updates, 5 iterations.

    Iteration 1 adopts Agent 3's accepted sealed `pilot_p9c` rollout, which was
    collected from the Phase 8 anchor and is therefore genuinely on-policy for
    a run that starts from that anchor. Iterations 2-5 are collected fresh from
    the snapshot frozen at the end of the previous iteration, so no iteration
    ever trains on a stale rollout.
    """
    import shutil

    root = ps.default_rollout_root() / SOAK_ROOT_NAME
    source = ps.default_rollout_root() / RESUME_SOURCE_ROOT
    if args.reset_soak:
        # An archive member is immutable, so a repeated soak has to say
        # explicitly that it is discarding the previous one rather than
        # silently overwriting it.
        for path in (root, ARCHIVE_ROOT):
            if path.exists():
                shutil.rmtree(path)
        for path in WORK_DIRECTORY.glob("soak_*.pt"):
            path.unlink()
        log("reset: previous soak rollouts, archive members and snapshots removed")
    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    iterations_planned = int(args.soak_iterations)
    config = pt.Phase9TrainConfig.for_soak(
        namespace=SOAK_NAMESPACE, device=args.device, total_iterations=iterations_planned
    )
    topology = pt.LoaderTopology(
        workers=args.workers, prefetch=args.prefetch, record_cache_size=args.record_cache
    )
    anchor_path = REPOSITORY_ROOT / EXPECTED_PHASE8_CHECKPOINT_PATH
    anchor_sha = pb.file_sha256(anchor_path)

    # Iteration 1 is a relocation of accepted bytes: same digests, same
    # rollout. Nothing is regenerated and nothing in Agent 3's tree is touched.
    adopted = adopt_iteration(source, root, SOAK_NAMESPACE, 1)

    trainer = pt.Phase9Trainer.from_phase8_checkpoint(
        anchor_path, config, ACCEPTED_CORPUS, topology=topology, run_label="agent05_soak"
    )
    resolver = pc.SnapshotResolver(
        device=args.collect_device, inference_batch_shape=args.batch_shape
    )
    anchor_snapshot = resolver.resolve(
        anchor_path,
        logical_identity="H000",
        policy_token="phase9_anchor_v1|H000",
        expected_sha256=anchor_sha,
    )
    historical = {"H000": anchor_snapshot}
    archive_members: list[dict] = []
    iterations: list[dict] = []
    all_rows: list[dict] = []
    behavior_path = anchor_path
    behavior_sha = anchor_sha

    total_updates = 0
    try:
        for iteration in range(1, iterations_planned + 1):
            identity = behavior_snapshot_identity(iteration)
            collection_seconds = 0.0
            sealing_seconds = 0.0
            # One snapshot object per iteration, used for both collection and
            # the on-policy binding check: the anchor is a Phase 8 container and
            # every later snapshot is a Phase 9 one, so the two loaders differ
            # while the identity they produce does not.
            if iteration == 1:
                snapshot = resolver.resolve(
                    behavior_path,
                    logical_identity=identity,
                    policy_token=f"phase9_behavior_v1|ns={SOAK_NAMESPACE}|{identity}",
                    expected_sha256=behavior_sha,
                )
            else:
                snapshot = pck.bind_behavior_snapshot(
                    behavior_path,
                    logical_identity=identity,
                    namespace=SOAK_NAMESPACE,
                    device=args.collect_device,
                    inference_batch_shape=args.batch_shape,
                    expected_sha256=behavior_sha,
                )
            if iteration == 1:
                collected = dict(adopted)
            else:
                window = active_historical_window(iteration)
                manifest = ActiveHistoryManifest.frozen_for(
                    SOAK_NAMESPACE,
                    iteration,
                    {key: historical[key].checkpoint_sha256 for key in window},
                )
                manifest.validate()
                started = time.perf_counter()
                collected = pc.collect_iteration(
                    root,
                    SOAK_NAMESPACE,
                    iteration,
                    soak_participants(
                        snapshot, {key: historical[key] for key in window}
                    ),
                    population_version=PHASE9_POPULATION_VERSION,
                    schedule_version=PHASE9_ROLLOUT_SCHEDULE_VERSION,
                    contract_digest=contract_digest(),
                    games_in_flight=args.games_in_flight,
                    history=manifest,
                    progress=lambda done, total: log(
                        f"  iteration {iteration}: collected {done}/{total} games"
                    ),
                )
                collection_seconds = time.perf_counter() - started
                sealing_seconds = float(collected.get("seal", {}).get("seconds", 0.0))
                log(
                    f"iteration {iteration}: collected {collected['games_collected']} "
                    f"games in {collection_seconds:.1f}s "
                    f"({collected.get('games_per_second', 0):.2f} games/s)"
                )

            started = time.perf_counter()
            rollout = pt.bind_sealed_rollout(
                root,
                SOAK_NAMESPACE,
                iteration,
                expected_model_state_digest=trainer.model_state_digest(),
                behavior_snapshot=snapshot,
                require_full_schedule=True,
            )
            target_seconds = time.perf_counter() - started

            trainer.bind_iteration(rollout)
            started = time.perf_counter()
            rows = trainer.train_iteration(timing=args.timing)
            train_seconds = time.perf_counter() - started
            trainer.mark_iteration_trained()
            total_updates += len(rows)
            for row in rows:
                all_rows.append(row)

            checkpoint_started = time.perf_counter()
            resume_path = WORK_DIRECTORY / f"soak_resume_it{iteration:03d}.pt"
            trainer.save_checkpoint(resume_path)
            next_identity = behavior_snapshot_identity(iteration + 1)
            behavior_path = WORK_DIRECTORY / f"soak_behavior_{next_identity}.pt"
            written = trainer.save_behavior_snapshot(
                behavior_path,
                logical_identity=next_identity,
                rl_iteration=iteration + 1,
            )
            behavior_sha = written["sha256"]
            checkpoint_seconds = time.perf_counter() - checkpoint_started

            archived = None
            if iteration % ARCHIVE_CADENCE_ITERATIONS == 0:
                # The frozen cadence applies to pilot namespaces too, which is
                # exactly why Agent 6's pilots need a real pilot-local H005
                # before their iterations 6-8 can be collected at all.
                local_archive = archive_snapshot_id(iteration)
                payload = trainer.archive_member_payload(local_identity=local_archive)
                member = pck.write_archive_member(
                    payload,
                    ARCHIVE_ROOT,
                    namespace=SOAK_NAMESPACE,
                    local_identity=local_archive,
                )
                bound = pck.bind_archive_member(
                    member,
                    device=args.collect_device,
                    inference_batch_shape=args.batch_shape,
                )
                bound.assert_frozen()
                historical[local_archive] = bound
                archived = member.to_dict()
                archive_members.append(archived)
                log(f"archived {member.qualified_identity} -> {member.checkpoint_sha256[:16]}")

            iterations.append(
                summarize_iteration(
                    rollout,
                    rows,
                    collected,
                    collection_seconds=collection_seconds,
                    sealing_seconds=sealing_seconds,
                    target_seconds=target_seconds,
                    train_seconds=train_seconds,
                    checkpoint_seconds=checkpoint_seconds,
                    archived=archived,
                    controller=trainer.controller,
                )
            )
            log(
                f"iteration {iteration}: {len(rows)} updates, total {total_updates}, "
                f"beta={trainer.controller.beta:.6f}, "
                f"mean KL={iterations[-1]['mean_behavior_kl']:.6f}, "
                f"clip={iterations[-1]['max_epoch_clip_fraction']:.4f}"
            )
        rehearsal = archive_rehearsal(
            root, historical, behavior_path, behavior_sha, args, iterations_planned
        )
    finally:
        trainer.close()

    payload = assemble_soak(
        trainer, config, iterations, all_rows, archive_members, root, total_updates, args
    )
    payload["archive_rehearsal"] = rehearsal
    return payload


def archive_rehearsal(
    root, historical, behavior_path, behavior_sha, args, iterations_completed
) -> dict:
    """Prove the pilot-local `H005` can actually answer an iteration-6 schedule.

    Agent 3's carry-forward: every 8-iteration pilot schedules `H005` opponents
    from iteration 6, so those games have no weights until a real pilot-local
    archive member exists. This plays a bounded slice of exactly those games
    against the member the soak just archived — the interface Agent 6 needs,
    exercised end to end rather than asserted.

    Nothing is persisted. The point is that the scheduled identity resolves to
    real immutable weights and the games run; writing a partial iteration 6
    into the store would leave a rollout no one intends to seal.
    """
    iteration = int(iterations_completed) + 1
    window = active_historical_window(iteration)
    if "H005" not in window:
        return {"skipped": f"iteration {iteration} window is {list(window)}"}
    snapshot = pck.bind_behavior_snapshot(
        behavior_path,
        logical_identity=behavior_snapshot_identity(iteration),
        namespace=SOAK_NAMESPACE,
        device=args.collect_device,
        inference_batch_shape=args.batch_shape,
        expected_sha256=behavior_sha,
    )
    manifest = ActiveHistoryManifest.frozen_for(
        SOAK_NAMESPACE,
        iteration,
        {key: historical[key].checkpoint_sha256 for key in window},
    )
    manifest.validate()
    participants = soak_participants(snapshot, {key: historical[key] for key in window})

    # Target the H005 games directly. A plain `limit` would take the first N
    # games in schedule order, which are all `current` bucket — it would
    # collect nothing that touches the archive member this rehearsal exists to
    # exercise.
    token = historical_policy_token(SOAK_NAMESPACE, "H005")
    wanted = [
        entry["game_id"]
        for entry in iter_scheduled_games(SOAK_NAMESPACE, iteration)
        if entry["bucket"] == "historical"
        and rebuild_scheduled_game(entry["game_id"]).opponent_identity == token
    ][: args.rehearsal_games]
    played = 0
    for runner in pc.collect_games(wanted, participants, games_in_flight=min(len(wanted), 32),
                                   history=manifest):
        scheduled = runner.scheduled
        digest = participants.historical_snapshot(
            scheduled.historical_snapshot_identity
        ).checkpoint_sha256
        store.build_rollout_metadata(
            scheduled,
            runner.record,
            setup_provenance=runner.assignment.provenance,
            behavior_checkpoint_sha256=snapshot.checkpoint_sha256,
            opponent_checkpoint_sha256=digest,
            learner_decision_count=runner.learner_decision_count,
            population_version=PHASE9_POPULATION_VERSION,
            schedule_version=PHASE9_ROLLOUT_SCHEDULE_VERSION,
            contract_digest=contract_digest(),
        )
        played += 1
    return {
        "iteration": iteration,
        "active_window": list(window),
        "scheduled_games_against_H005": len(wanted),
        "games_played": played,
        "sealed": False,
        "persisted": False,
        "games_against_H005": played,
        "H005_checkpoint_sha256": historical["H005"].checkpoint_sha256,
        "H005_policy_token": token,
        "opponent_digest_is_the_archived_member": played > 0,
        "closes_carry_forward": (
            "an 8-iteration pilot's iterations 6-8 schedule H005 opponents; this "
            "shows a real pilot-local H005 is produced by the frozen cadence and "
            "binds cleanly as a playable historical opponent"
        ),
    }


def adopt_iteration(source: Path, destination: Path, namespace: str, iteration: int) -> dict:
    """Relocate one sealed iteration and require the digest to be unchanged.

    A pure relocation with unchanged digests is the same rollout — the identity
    rule the common contract states for the corpus applies to rollouts for the
    same reason, and it is checked here rather than assumed.
    """
    import shutil

    source_directory = store.iteration_directory(source, namespace, iteration)
    target_directory = store.iteration_directory(destination, namespace, iteration)
    before = store.sealed_rollout_digest(
        store.Phase9RolloutReader(source, namespace, iteration).commits
    )
    if not target_directory.exists():
        target_directory.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_directory, target_directory)
    after = store.sealed_rollout_digest(
        store.Phase9RolloutReader(destination, namespace, iteration).commits
    )
    if before != after:
        raise SystemExit(
            f"relocating {namespace} iteration {iteration} changed its digest: "
            f"{before} -> {after}"
        )
    state = store.read_iteration_state(destination, namespace, iteration)
    if state["state"] != "SEALED":
        # A previous soak run may have advanced it; the bytes are what matter.
        store.write_iteration_state(
            destination,
            namespace,
            iteration,
            "SEALED",
            sealed_rollout_digest=after,
            behavior_snapshot_id=state.get("behavior_snapshot_id"),
            behavior_checkpoint_sha256=state.get("behavior_checkpoint_sha256"),
        )
    return {
        "adopted_from": str(source_directory),
        "adopted_to": str(target_directory),
        "sealed_rollout_digest": after,
        "digest_unchanged_by_relocation": before == after,
        "games_collected": 0,
        "games_per_second": 0.0,
    }


def summarize_iteration(
    rollout,
    rows,
    collected,
    *,
    collection_seconds,
    sealing_seconds,
    target_seconds,
    train_seconds,
    checkpoint_seconds,
    archived,
    controller,
) -> dict:
    kls = [row["behavior_kl"] for row in rows]
    clips = [row["epoch_clip_fraction"] for row in rows if "epoch_clip_fraction" in row]
    epoch_kls = [row["epoch_mean_kl"] for row in rows if "epoch_mean_kl" in row]
    return {
        "namespace": rollout.namespace,
        "iteration": rollout.iteration,
        "sealed_rollout_digest": rollout.sealed_rollout_digest,
        "behavior_snapshot_id": rollout.behavior_snapshot_id,
        "behavior_checkpoint_sha256": rollout.behavior_checkpoint_sha256,
        "games": rollout.games,
        "learner_decisions": rollout.learner_decisions,
        "advantage_statistics": rollout.statistics.to_dict(),
        "updates": len(rows),
        "collection_seconds": collection_seconds,
        "sealing_audit_seconds": sealing_seconds,
        "target_construction_seconds": target_seconds,
        "train_seconds": train_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "data_wait_seconds": sum(row["data_wait_seconds"] for row in rows),
        "forward_seconds": sum(row["forward_seconds"] for row in rows),
        "backward_seconds": sum(row["backward_seconds"] for row in rows),
        "optimizer_seconds": sum(row["optimizer_seconds"] for row in rows),
        "loss_seconds": sum(row["loss_seconds"] for row in rows),
        "host_to_device_seconds": sum(row["host_to_device_seconds"] for row in rows),
        "mean_behavior_kl": float(np.mean(kls)) if kls else 0.0,
        "max_behavior_kl": float(np.max(kls)) if kls else 0.0,
        "epoch_mean_kls": epoch_kls,
        "max_epoch_clip_fraction": float(max(clips)) if clips else 0.0,
        "mean_clip_fraction": float(np.mean([row["clip_fraction"] for row in rows])),
        "mean_policy_entropy": float(np.mean([row["policy_entropy"] for row in rows])),
        "mean_advantage_retention": float(
            np.mean([row["advantage_retention"] for row in rows])
        ),
        "mean_grad_norm_pre_clip": float(
            np.mean([row["grad_norm_pre_clip"] for row in rows])
        ),
        "final_parameter_norm": float(rows[-1]["parameter_norm"]) if rows else 0.0,
        "kl_beta_after": float(controller.beta),
        "mps_mib_after_iteration": peak_mps_mib(),
        "rss_mib_after_iteration": peak_rss_mib(),
        "collection": {
            key: collected.get(key)
            for key in (
                "games_collected",
                "games_per_second",
                "learner_decisions",
                "observer_probe_failures",
                "sealed_rollout_digest",
                "adopted_from",
                "digest_unchanged_by_relocation",
            )
            if key in collected
        },
        "archived": archived,
    }


def assemble_soak(
    trainer, config, iterations, rows, archive_members, root, total_updates, args
) -> dict:
    kls = [row["behavior_kl"] for row in rows]
    clips = [row["clip_fraction"] for row in rows]
    entropies = [row["policy_entropy"] for row in rows]
    retentions = [row["advantage_retention"] for row in rows]
    grads = [row["grad_norm_pre_clip"] for row in rows]
    epoch_kls = [row["epoch_mean_kl"] for row in rows if "epoch_mean_kl" in row]
    epoch_clips = [
        row["epoch_clip_fraction"] for row in rows if "epoch_clip_fraction" in row
    ]
    non_finite = sum(
        1
        for row in rows
        if not all(
            np.isfinite(row[key])
            for key in (
                "loss_total",
                "loss_ppo",
                "loss_value",
                "loss_belief",
                "behavior_kl",
                "policy_entropy",
                "grad_norm_pre_clip",
                "parameter_norm",
            )
        )
    )
    train_seconds = sum(entry["train_seconds"] for entry in iterations)
    examples = sum(row["examples"] for row in rows)
    return {
        "phase": 9,
        "agent": 5,
        "artifact": "agent_05_stability_soak",
        **environment_record(),
        "role": (
            "infrastructure stability soak; not configuration selection and not "
            "a seventh pilot"
        ),
        "selection_statement": {
            "scope": config.scope,
            "selects_a_configuration": config.selects_a_configuration,
            "validation_bank_opened": False,
            "final_test_bank_opened": False,
            "validation_score_computed": False,
            "weights_carried_into_agent_6": False,
            "weights_location": str(WORK_DIRECTORY),
            "rollout_root": str(root),
            "rationale": (
                "the neutral middle candidate's numbers were used solely so the "
                "optimizer path is exercised at a realistic scale; no candidate "
                "is compared with any other and no score exists to compare"
            ),
        },
        "train_config": {**config.identity(), "digest": config.digest()},
        "topology": trainer.topology.to_dict(),
        "iterations": iterations,
        "totals": {
            "rl_iterations": len(iterations),
            "optimizer_updates": total_updates,
            "examples_consumed": examples,
            "games": sum(entry["games"] for entry in iterations),
            "learner_decisions": sum(entry["learner_decisions"] for entry in iterations),
            "train_seconds": train_seconds,
            "collection_seconds": sum(entry["collection_seconds"] for entry in iterations),
            "target_construction_seconds": sum(
                entry["target_construction_seconds"] for entry in iterations
            ),
            "checkpoint_seconds": sum(entry["checkpoint_seconds"] for entry in iterations),
            "data_wait_seconds": sum(entry["data_wait_seconds"] for entry in iterations),
            "examples_per_second": examples / train_seconds if train_seconds else 0.0,
            "updates_per_second": total_updates / train_seconds if train_seconds else 0.0,
            "peak_rss_mib": peak_rss_mib(),
            "peak_mps_mib": peak_mps_mib(),
        },
        "required_zero_counters": dict(trainer.counters),
        "non_finite_metric_rows": non_finite,
        "stability": {
            "mean_behavior_kl": float(np.mean(kls)) if kls else 0.0,
            "max_behavior_kl": float(np.max(kls)) if kls else 0.0,
            "max_epoch_mean_kl": float(max(epoch_kls)) if epoch_kls else 0.0,
            "kl_hard_limit": KL_HARD_LIMIT,
            "kl_hard_limit_exceeded": bool(
                epoch_kls and max(epoch_kls) > KL_HARD_LIMIT
            ),
            "mean_clip_fraction": float(np.mean(clips)) if clips else 0.0,
            "max_batch_clip_fraction": float(np.max(clips)) if clips else 0.0,
            "max_epoch_clip_fraction": float(max(epoch_clips)) if epoch_clips else 0.0,
            "clip_fraction_hard_limit": CLIP_FRACTION_HARD_LIMIT,
            "clip_fraction_hard_limit_exceeded": bool(
                epoch_clips and max(epoch_clips) > CLIP_FRACTION_HARD_LIMIT
            ),
            "mean_policy_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "final_policy_entropy": float(entropies[-1]) if entropies else 0.0,
            "mean_advantage_retention": float(np.mean(retentions)) if retentions else 0.0,
            "mean_grad_norm_pre_clip": float(np.mean(grads)) if grads else 0.0,
            "max_grad_norm_pre_clip": float(np.max(grads)) if grads else 0.0,
            "grad_norm_p95": percentile(grads, 0.95),
            "kl_controller_history": [dict(e) for e in trainer.controller.history],
            "kl_beta_final": float(trainer.controller.beta),
        },
        "archive_members": archive_members,
        "meets_minimum_updates": total_updates >= SOAK_MINIMUM_UPDATES,
        "minimum_updates": SOAK_MINIMUM_UPDATES,
    }


# ---------------------------------------------------------------------------
# Stage: binding — two genuinely different checkpoints
# ---------------------------------------------------------------------------


def decision_requests(record, metadata, wanted_player, *, limit):
    """Replay one game and rebuild everything a re-check needs for one side."""
    from stratego.engine.legal_moves import legal_action_mask, legal_actions
    from stratego.engine.observation import build_observation
    from stratego.engine.state import create_game
    from stratego.engine.transition import apply_action
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


def run_binding_stage(args) -> dict:
    """Each side of a historical matchup, against its own real checkpoint.

    Agent 3's soak could not demonstrate this: every iteration it collected was
    iteration 1, where `B001` and `H000` are the same file. This stage uses a
    soak iteration >= 2, whose learner is a genuinely trained network and whose
    historical opponent is the Phase 8 anchor.
    """
    import dataclasses

    root = ps.default_rollout_root() / SOAK_ROOT_NAME
    iteration = args.binding_iteration
    reader = store.Phase9RolloutReader(root, SOAK_NAMESPACE, iteration)
    if not reader.game_ids:
        raise SystemExit(
            f"binding needs a collected soak iteration {iteration}; run --stage soak first"
        )
    state = store.read_iteration_state(root, SOAK_NAMESPACE, iteration)
    learner_path = WORK_DIRECTORY / f"soak_behavior_{behavior_snapshot_identity(iteration)}.pt"
    learner_snapshot = pck.bind_behavior_snapshot(
        learner_path,
        logical_identity=behavior_snapshot_identity(iteration),
        namespace=SOAK_NAMESPACE,
        device=args.collect_device,
        inference_batch_shape=args.batch_shape,
        expected_sha256=state["behavior_checkpoint_sha256"],
    )
    anchor_path = REPOSITORY_ROOT / EXPECTED_PHASE8_CHECKPOINT_PATH
    opponent_snapshot = pb.load_behavior_snapshot(
        anchor_path,
        logical_identity="H000",
        policy_token="phase9_anchor_v1|H000",
        device=args.collect_device,
        inference_batch_shape=args.batch_shape,
        expected_sha256=pb.file_sha256(anchor_path),
    )

    games = [
        game_id
        for game_id in reader.game_ids
        if reader.metadata[game_id]["bucket"] == "historical"
    ][: args.binding_games]
    learner_requests: list = []
    opponent_requests: list = []
    for game_id in games:
        record, metadata = reader.read_game(game_id)
        learner = 0 if metadata["learner_color"] == "red" else 1
        learner_requests.extend(
            decision_requests(record, metadata, learner, limit=args.binding_decisions)
        )
        opponent_requests.extend(
            decision_requests(record, metadata, 1 - learner, limit=args.binding_decisions)
        )

    def verdict(reports):
        verified = [report for report in reports if report["verified"]]
        differences = [
            report["max_abs_difference"]
            for report in reports
            if report["max_abs_difference"] is not None
        ]
        return {
            "decisions": len(reports),
            "verified": len(verified),
            "failed": len(reports) - len(verified),
            "max_abs_difference": float(max(differences)) if differences else None,
            "sample_problems": [
                report["problems"][:1] for report in reports if not report["verified"]
            ][:2],
        }

    def rebind(requests, digest):
        return [
            dataclasses.replace(request, stored_checkpoint_sha256=digest)
            for request in requests
        ]

    correct_learner = verdict(pb.reproduce_decisions(learner_snapshot, learner_requests))
    correct_opponent = verdict(
        pb.reproduce_decisions(opponent_snapshot, opponent_requests)
    )
    swapped_learner = verdict(
        pb.reproduce_decisions(
            opponent_snapshot,
            rebind(learner_requests, opponent_snapshot.checkpoint_sha256),
        )
    )
    swapped_opponent = verdict(
        pb.reproduce_decisions(
            learner_snapshot, rebind(opponent_requests, learner_snapshot.checkpoint_sha256)
        )
    )
    guard_only = verdict(pb.reproduce_decisions(opponent_snapshot, learner_requests))

    return {
        "namespace": SOAK_NAMESPACE,
        "iteration": iteration,
        "games_examined": len(games),
        "learner": {
            "logical_identity": learner_snapshot.logical_identity,
            "checkpoint_sha256": learner_snapshot.checkpoint_sha256,
            "state_dict_digest": learner_snapshot.loaded_state_dict_digest,
        },
        "historical_opponent": {
            "logical_identity": opponent_snapshot.logical_identity,
            "checkpoint_sha256": opponent_snapshot.checkpoint_sha256,
            "state_dict_digest": opponent_snapshot.loaded_state_dict_digest,
        },
        "checkpoints_are_genuinely_different": (
            learner_snapshot.checkpoint_sha256 != opponent_snapshot.checkpoint_sha256
            and learner_snapshot.loaded_state_dict_digest
            != opponent_snapshot.loaded_state_dict_digest
        ),
        "each_side_against_its_own_checkpoint": {
            "learner": correct_learner,
            "historical_opponent": correct_opponent,
            "all_verified": (
                correct_learner["failed"] == 0 and correct_opponent["failed"] == 0
            ),
        },
        "swapped_bindings": {
            "learner_decisions_against_opponent_checkpoint": swapped_learner,
            "opponent_decisions_against_learner_checkpoint": swapped_opponent,
            "all_failed": (
                swapped_learner["verified"] == 0 and swapped_opponent["verified"] == 0
            ),
            "note": (
                "the recorded checkpoint digest is rewritten in these cases so "
                "the numerical comparison actually runs; the digest guard alone "
                "is measured separately"
            ),
        },
        "digest_guard_alone": {
            **guard_only,
            "rejected_before_any_forward_pass": guard_only["max_abs_difference"] is None,
        },
        "closes_agent_3_limitation": (
            "Agent 3's soak only ever had B001 == H000, so a swapped binding "
            "would have passed; here the two sides are different networks and a "
            "swap fails by orders of magnitude"
        ),
    }


# ---------------------------------------------------------------------------
# Stage: throughput
# ---------------------------------------------------------------------------


def run_topology_probe(args) -> dict:
    """Prove that a topology change moves timing and nothing else."""
    rollout = resume_rollout()
    config = pt.Phase9TrainConfig.for_soak(
        namespace=RESUME_NAMESPACE, device=args.device, total_iterations=SOAK_ITERATIONS
    )
    results = []
    reference = None

    def measure(workers: int, *, timing: bool):
        trainer = pt.Phase9Trainer.from_phase8_checkpoint(
            REPOSITORY_ROOT / EXPECTED_PHASE8_CHECKPOINT_PATH,
            config,
            ACCEPTED_CORPUS,
            topology=pt.LoaderTopology(
                workers=workers,
                prefetch=args.prefetch,
                record_cache_size=args.record_cache,
            ),
            run_label=f"topology_w{workers}_{'timed' if timing else 'free'}",
        )
        with trainer:
            trainer.bind_iteration(rollout, mark_training=False)
            started = time.perf_counter()
            rows = trainer.train_iteration(
                updates=args.topology_updates,
                capture_batch_digests=True,
                timing=timing,
            )
            return rows, time.perf_counter() - started

    for workers in args.topology_workers:
        # Two passes: the timed one splits a step into its phases (the split
        # the assignment asks for), the untimed one measures what the pipeline
        # actually delivers. Device synchronization between phases is what
        # makes the split meaningful and is also what makes it slower, so
        # reporting only the timed rate would understate the real throughput.
        rows, elapsed = measure(workers, timing=True)
        free_rows, free_elapsed = measure(workers, timing=False)
        warm = rows[args.topology_warmup :]
        digests = [row["batch_digest"] for row in rows]
        if reference is None:
            reference = digests
        free_digests = [row["batch_digest"] for row in free_rows]
        results.append(
            {
                "workers": workers,
                "prefetch": args.prefetch,
                "updates": len(rows),
                "seconds": elapsed,
                "updates_per_second": len(rows) / elapsed if elapsed else 0.0,
                "examples_per_second": (
                    sum(row["examples"] for row in rows) / elapsed if elapsed else 0.0
                ),
                "unsynchronized_seconds": free_elapsed,
                "unsynchronized_examples_per_second": (
                    sum(row["examples"] for row in free_rows) / free_elapsed
                    if free_elapsed
                    else 0.0
                ),
                "unsynchronized_updates_per_second": (
                    len(free_rows) / free_elapsed if free_elapsed else 0.0
                ),
                "synchronization_cost_fraction": (
                    (elapsed - free_elapsed) / elapsed if elapsed else 0.0
                ),
                "batch_digests_match_untimed_pass": digests == free_digests,
                "mean_step_seconds": float(np.mean([row["step_seconds"] for row in warm])),
                "mean_data_wait_seconds": float(
                    np.mean([row["data_wait_seconds"] for row in warm])
                ),
                "mean_forward_seconds": float(
                    np.mean([row["forward_seconds"] for row in warm])
                ),
                "mean_backward_seconds": float(
                    np.mean([row["backward_seconds"] for row in warm])
                ),
                "mean_optimizer_seconds": float(
                    np.mean([row["optimizer_seconds"] for row in warm])
                ),
                "batch_digests_match_reference": digests == reference,
                "losses": [row["loss_total"] for row in rows],
            }
        )
        log(
            f"topology workers={workers}: {results[-1]['examples_per_second']:.0f} "
            f"examples/s, wait {results[-1]['mean_data_wait_seconds']*1000:.0f} ms"
        )
    losses_identical = all(
        entry["losses"] == results[0]["losses"] for entry in results
    )
    return {
        "measurements": results,
        "identical_logical_minibatch_identities": all(
            entry["batch_digests_match_reference"] for entry in results
        ),
        # Reported, not required. The train order's claim is about *which*
        # examples a minibatch holds and in what order — the batch digest.
        # Equal losses would additionally require the backend to be
        # run-to-run deterministic, which this MPS stack is not; on CPU they
        # do come out equal, which the trainer's own topology test asserts.
        "identical_losses_across_topologies": losses_identical,
        "loss_equality_is_a_backend_property": (
            "identical batch digests with differing losses is MPS "
            "non-determinism, not a train-order violation"
        ),
        "identical_across_synchronization": all(
            entry["batch_digests_match_untimed_pass"] for entry in results
        ),
        "rule": (
            "worker count, prefetch and device synchronization change arrival "
            "times only; the minibatch plan is a pure function of the cursor "
            "and every batch is verified against it"
        ),
        "synchronization_note": (
            "the timed pass synchronizes the device between phases so the "
            "per-phase split means what it says; the untimed pass measures what "
            "the pipeline actually delivers"
        ),
    }


def sealing_audit_probe(root, namespace: str, iteration: int) -> dict:
    """Time the seal's own work without writing a state transition.

    `collect_iteration` seals inside the same call it collects in, so the soak
    rows cannot separate the two. This repeats exactly what sealing verifies —
    read every journal, decode and validate every payload and sidecar, and
    recompute the sealed digest — and times it, so the phase the assignment
    asks for has a real number instead of an unmeasured zero.
    """
    started = time.perf_counter()
    reader = store.Phase9RolloutReader(root, namespace, iteration)
    open_seconds = time.perf_counter() - started

    started = time.perf_counter()
    decoded = 0
    problems = 0
    for game_id in reader.game_ids:
        record, metadata = reader.read_game(game_id)
        problems += len(store.validate_rollout_metadata(metadata, record))
        decoded += 1
    decode_seconds = time.perf_counter() - started

    started = time.perf_counter()
    digest = store.sealed_rollout_digest(reader.commits)
    digest_seconds = time.perf_counter() - started
    state = store.read_iteration_state(root, namespace, iteration)
    return {
        "namespace": namespace,
        "iteration": iteration,
        "games_decoded": decoded,
        "metadata_problems": problems,
        "journal_open_seconds": open_seconds,
        "decode_and_validate_seconds": decode_seconds,
        "digest_seconds": digest_seconds,
        "sealing_audit_seconds": open_seconds + decode_seconds + digest_seconds,
        "recomputed_digest": digest,
        "matches_state_record": digest == state.get("sealed_rollout_digest"),
        "note": (
            "measured without writing a state transition, so the soak's own "
            "state machine is untouched"
        ),
    }


def write_benchmark_csv(soak: dict, topology: dict) -> Path:
    path = DATA_DIRECTORY / "agent_05_training_benchmark.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_COLUMNS)
        writer.writeheader()
        # `collect_iteration` seals inside the call it collects in, so the
        # measured seal cost comes from the standalone probe rather than from a
        # column the soak could not separate.
        sealing = topology.get("sealing_audit_probe", {}).get("sealing_audit_seconds", 0.0)
        for entry in soak["iterations"]:
            iteration_seconds = (
                entry["collection_seconds"]
                + sealing
                + entry["target_construction_seconds"]
                + entry["train_seconds"]
                + entry["checkpoint_seconds"]
            )
            writer.writerow(
                {
                    "measurement": "soak_iteration",
                    "namespace": entry["namespace"],
                    "iteration": entry["iteration"],
                    "workers": soak["topology"]["workers"],
                    "prefetch": soak["topology"]["prefetch"],
                    "updates": entry["updates"],
                    "games": entry["games"],
                    "learner_decisions": entry["learner_decisions"],
                    "collection_seconds": round(entry["collection_seconds"], 3),
                    "sealing_audit_seconds": round(sealing, 3),
                    "target_construction_seconds": round(
                        entry["target_construction_seconds"], 3
                    ),
                    "data_wait_seconds": round(entry["data_wait_seconds"], 3),
                    "host_to_device_seconds": round(entry["host_to_device_seconds"], 3),
                    "forward_seconds": round(entry["forward_seconds"], 3),
                    "loss_seconds": round(entry["loss_seconds"], 3),
                    "backward_seconds": round(entry["backward_seconds"], 3),
                    "optimizer_seconds": round(entry["optimizer_seconds"], 3),
                    "checkpoint_seconds": round(entry["checkpoint_seconds"], 3),
                    "validation_infrastructure_seconds": 0.0,
                    "train_seconds": round(entry["train_seconds"], 3),
                    "iteration_seconds": round(iteration_seconds, 3),
                    "examples_per_second": round(
                        entry["updates"]
                        * soak["train_config"]["minibatch_size"]
                        / entry["train_seconds"],
                        1,
                    )
                    if entry["train_seconds"]
                    else 0.0,
                    "updates_per_second": round(
                        entry["updates"] / entry["train_seconds"], 3
                    )
                    if entry["train_seconds"]
                    else 0.0,
                    "peak_rss_mib": "",
                    "peak_mps_mib": "",
                }
            )
        for entry in topology["measurements"]:
            writer.writerow(
                {
                    "measurement": "topology_probe",
                    "namespace": RESUME_NAMESPACE,
                    "iteration": RESUME_ITERATION,
                    "workers": entry["workers"],
                    "prefetch": entry["prefetch"],
                    "updates": entry["updates"],
                    "games": "",
                    "learner_decisions": "",
                    "collection_seconds": "",
                    "sealing_audit_seconds": "",
                    "target_construction_seconds": "",
                    "data_wait_seconds": round(
                        entry["mean_data_wait_seconds"] * entry["updates"], 3
                    ),
                    "host_to_device_seconds": "",
                    "forward_seconds": round(
                        entry["mean_forward_seconds"] * entry["updates"], 3
                    ),
                    "loss_seconds": "",
                    "backward_seconds": round(
                        entry["mean_backward_seconds"] * entry["updates"], 3
                    ),
                    "optimizer_seconds": round(
                        entry["mean_optimizer_seconds"] * entry["updates"], 3
                    ),
                    "checkpoint_seconds": "",
                    "validation_infrastructure_seconds": "",
                    "train_seconds": round(entry["seconds"], 3),
                    "iteration_seconds": round(entry["seconds"], 3),
                    "examples_per_second": round(entry["examples_per_second"], 1),
                    "updates_per_second": round(entry["updates_per_second"], 3),
                    "unsynchronized_examples_per_second": round(
                        entry["unsynchronized_examples_per_second"], 1
                    ),
                    "peak_rss_mib": "",
                    "peak_mps_mib": "",
                }
            )
        totals = soak["totals"]
        writer.writerow(
            {
                "measurement": "soak_total",
                "namespace": SOAK_NAMESPACE,
                "iteration": "",
                "workers": soak["topology"]["workers"],
                "prefetch": soak["topology"]["prefetch"],
                "updates": totals["optimizer_updates"],
                "games": totals["games"],
                "learner_decisions": totals["learner_decisions"],
                "collection_seconds": round(totals["collection_seconds"], 3),
                "sealing_audit_seconds": "",
                "target_construction_seconds": round(
                    totals["target_construction_seconds"], 3
                ),
                "data_wait_seconds": round(totals["data_wait_seconds"], 3),
                "host_to_device_seconds": "",
                "forward_seconds": "",
                "loss_seconds": "",
                "backward_seconds": "",
                "optimizer_seconds": "",
                "checkpoint_seconds": round(totals["checkpoint_seconds"], 3),
                "validation_infrastructure_seconds": 0.0,
                "train_seconds": round(totals["train_seconds"], 3),
                "iteration_seconds": "",
                "examples_per_second": round(totals["examples_per_second"], 1),
                "updates_per_second": round(totals["updates_per_second"], 3),
                "peak_rss_mib": round(totals["peak_rss_mib"], 1),
                "peak_mps_mib": round(totals["peak_mps_mib"], 1),
            }
        )
    return path


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def build_gates(contract: dict, resume: dict, soak: dict, binding: dict,
                topology: dict, tests: dict) -> dict:
    prerequisites = contract["prerequisites"]
    corpus = contract["corpus"]
    stability = soak["stability"]
    counters = soak["required_zero_counters"]
    gates = {
        "agents1_4_pass": all(
            entry["status"] == "PASS" for entry in prerequisites["acceptances"].values()
        )
        and not prerequisites["problems"],
        "corpus_resolver_verified": corpus["resolved_root"] is not None
        and not corpus["modules_hard_coding_absolute_paths"],
        "corpus_digests_match": corpus["identity_matches"],
        "ppo_loss_matches_contract": tests.get("loss_tests_passed", False),
        "illegal_logit_masking_pass": tests.get("masking_tests_passed", False),
        "value_loss_matches_contract": tests.get("value_tests_passed", False),
        "belief_loss_matches_contract": tests.get("belief_tests_passed", False),
        "kl_direction_and_beta_controller_pass": tests.get("kl_tests_passed", False),
        "entropy_schedule_pass": tests.get("entropy_tests_passed", False),
        "opponent_only_gradients_zero": tests.get("opponent_tests_passed", False),
        "cpu_resume_pass": bool(resume.get("cpu_resume_pass")),
        "mps_backend_aware_resume_pass": bool(resume.get("mps_resume_pass")),
        "atomic_checkpoint_tests_pass": tests.get("checkpoint_tests_passed", False),
        "soak_updates_ge_2000": soak["totals"]["optimizer_updates"] >= SOAK_MINIMUM_UPDATES,
        "soak_several_sealed_iterations": soak["totals"]["rl_iterations"] >= 3,
        "nonfinite_zero": (
            soak["non_finite_metric_rows"] == 0
            and counters["non_finite_losses"] == 0
            and counters["non_finite_gradients"] == 0
            and counters["non_finite_parameters"] == 0
        ),
        "illegal_targets_zero": counters["illegal_targets"] == 0,
        "identity_mismatches_zero": (
            counters["behavior_identity_mismatches"] == 0
            and counters["rollout_identity_mismatches"] == 0
            and counters["data_mismatches"] == 0
            and counters["checkpoint_errors"] == 0
        ),
        "kl_hard_limit_not_exceeded": not stability["kl_hard_limit_exceeded"],
        "clip_fraction_hard_limit_not_exceeded": not stability[
            "clip_fraction_hard_limit_exceeded"
        ],
        "throughput_measured": bool(topology["measurements"])
        and topology["identical_logical_minibatch_identities"],
        "checkpoint_binding_fixture_pass": (
            binding["checkpoints_are_genuinely_different"]
            and binding["each_side_against_its_own_checkpoint"]["all_verified"]
            and binding["swapped_bindings"]["all_failed"]
        ),
        "namespace_qualified_archive_pass": bool(soak["archive_members"])
        and all(
            member["qualified_identity"].startswith(f"{SOAK_NAMESPACE}|")
            for member in soak["archive_members"]
        ),
        "no_pilot_selection": (
            not soak["selection_statement"]["selects_a_configuration"]
            and not soak["selection_statement"]["validation_score_computed"]
            and not soak["selection_statement"]["weights_carried_into_agent_6"]
        ),
        "no_final_test_access": not soak["selection_statement"]["final_test_bank_opened"],
        "full_suite_green": tests.get("full_suite_green", False),
    }
    return {
        "gates": gates,
        "passed": sum(1 for value in gates.values() if value),
        "total": len(gates),
        "all_passed": all(gates.values()),
        "failed": [name for name, value in gates.items() if not value],
    }


def run_pytest(selection=None, *, keyword: "str | None" = None) -> dict:
    command = [
        sys.executable,
        "-m",
        "pytest",
        *(selection or ["tests"]),
        "-q",
        "-p",
        "no:randomly",
    ]
    if keyword:
        command += ["-k", keyword]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True)
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "summary": tail,
        "seconds": time.perf_counter() - started,
        # Exit code 5 is "no tests collected", which must fail the gate: a
        # selection that matches nothing is not a selection that passed.
        "passed": completed.returncode == 0,
    }


LOSS_TESTS = "tests/training/test_phase9_loss.py"
TRAINER_TESTS = "tests/training/test_phase9_trainer.py"
CHECKPOINT_TESTS = "tests/training/test_phase9_checkpoint.py"
BINDING_TESTS = "tests/training/test_phase9_checkpoint_binding.py"

#: Each completion gate that rests on tests, mapped to the tests that actually
#: measure it. A gate claimed true because some larger module passed is a gate
#: nobody measured, so the selections are per-gate and an empty selection
#: (pytest exit code 5) fails rather than silently passing.
GATE_TEST_SELECTIONS = {
    "loss_tests_passed": ([LOSS_TESTS], "ppo or clip or total_is_the_frozen"),
    "masking_tests_passed": ([LOSS_TESTS], "illegal or masking or behavior_matrix"),
    "value_tests_passed": ([LOSS_TESTS], "value"),
    "belief_tests_passed": ([LOSS_TESTS], "belief"),
    "kl_tests_passed": ([LOSS_TESTS, TRAINER_TESTS], "kl or controller or damps"),
    "entropy_tests_passed": ([LOSS_TESTS, TRAINER_TESTS], "entropy"),
    "opponent_tests_passed": ([TRAINER_TESTS], "opponent"),
    "checkpoint_tests_passed": (
        [CHECKPOINT_TESTS, BINDING_TESTS, TRAINER_TESTS],
        "crash or atomic or archive or resume or reject or overwrite or swap",
    ),
}


def targeted_test_results() -> dict:
    """Run the Agent 5 test modules, and each gate's own tests separately."""
    modules = {
        "loss": LOSS_TESTS,
        "checkpoint": CHECKPOINT_TESTS,
        "trainer": TRAINER_TESTS,
        "binding": BINDING_TESTS,
        "artifacts": "tests/training/test_phase9_agent05_artifacts.py",
    }
    outcomes = {name: run_pytest([path]) for name, path in modules.items()}
    gates = {
        name: run_pytest(paths, keyword=keyword)
        for name, (paths, keyword) in GATE_TEST_SELECTIONS.items()
    }
    for name, result in gates.items():
        log(f"gate tests {name}: {result['summary']}")
    return {
        "modules": outcomes,
        "gate_selections": gates,
        **{name: result["passed"] for name, result in gates.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_stage(name: str):
    path = WORK_DIRECTORY / f"stage_{name}.json"
    return read_json(path) if path.exists() else None


def save_stage(name: str, payload) -> None:
    write_json(WORK_DIRECTORY / f"stage_{name}.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 9 Agent 5 acceptance harness")
    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "all",
            "verify",
            "contract",
            "resume",
            "soak",
            "binding",
            "throughput",
            "artifacts",
        ],
    )
    parser.add_argument("--resume-worker", default=None)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--collect-device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--batch-shape", type=int, default=pb.DEFAULT_INFERENCE_BATCH_SHAPE)
    parser.add_argument("--games-in-flight", type=int, default=pc.DEFAULT_GAMES_IN_FLIGHT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--record-cache", type=int, default=48)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--resume-updates", type=int, default=160)
    parser.add_argument("--resume-split-at", type=int, default=60)
    parser.add_argument("--cpu-updates", type=int, default=24)
    parser.add_argument("--cpu-split-at", type=int, default=10)
    parser.add_argument("--skip-cpu-resume", action="store_true")
    parser.add_argument("--skip-mps-resume", action="store_true")
    parser.add_argument("--reset-soak", action="store_true")
    parser.add_argument("--soak-iterations", type=int, default=SOAK_ITERATIONS)
    parser.add_argument("--rehearsal-games", type=int, default=48)
    parser.add_argument("--binding-iteration", type=int, default=2)
    parser.add_argument("--binding-games", type=int, default=16)
    parser.add_argument("--binding-decisions", type=int, default=12)
    parser.add_argument("--topology-workers", type=int, nargs="+", default=[1, 4, 6, 10])
    parser.add_argument("--topology-updates", type=int, default=10)
    parser.add_argument("--topology-warmup", type=int, default=2)
    parser.add_argument("--timing", action="store_true", default=True)
    parser.add_argument("--skip-payload-bytes", action="store_true")
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument("--record-final-suite", action="store_true")
    args = parser.parse_args()

    if args.resume_worker:
        resume_worker(args.resume_worker, args)
        return 0

    if args.record_final_suite:
        return record_final_suite()

    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stages = (
        ["verify", "contract", "resume", "soak", "binding", "throughput", "artifacts"]
        if args.stage == "all"
        else [args.stage]
    )

    if "verify" in stages:
        prerequisites = verify_prerequisites()
        corpus = verify_corpus(check_payload_bytes=not args.skip_payload_bytes)
        storage = verify_storage()
        if prerequisites["problems"] or corpus["problems"]:
            log(f"BLOCKED: {prerequisites['problems'] + corpus['problems']}")
            return 2
        save_stage("verify", {"prerequisites": prerequisites, "corpus": corpus,
                              "storage": storage})
        log("verify: prerequisites, corpus resolver and storage all confirmed")

    if "contract" in stages:
        verified = load_stage("verify")
        contract = build_contract(
            verified["prerequisites"], verified["corpus"], verified["storage"]
        )
        save_stage("contract", contract)
        write_json(DATA_DIRECTORY / "agent_05_trainer_contract.json", contract)
        log("contract: agent_05_trainer_contract.json written")

    if "resume" in stages:
        resume = run_resume_stage(args)
        save_stage("resume", resume)
        write_json(DATA_DIRECTORY / "agent_05_resume_validation.json", resume)
        log("resume: agent_05_resume_validation.json written")

    if "soak" in stages:
        soak = run_soak(args)
        save_stage("soak", soak)
        write_json(DATA_DIRECTORY / "agent_05_stability_soak.json", soak)
        log(
            f"soak: {soak['totals']['optimizer_updates']} updates over "
            f"{soak['totals']['rl_iterations']} iterations"
        )

    if "binding" in stages:
        binding = run_binding_stage(args)
        save_stage("binding", binding)
        log(
            "binding: swapped bindings failed = "
            f"{binding['swapped_bindings']['all_failed']}"
        )

    if "throughput" in stages:
        topology = run_topology_probe(args)
        topology["sealing_audit_probe"] = sealing_audit_probe(
            ps.default_rollout_root() / SOAK_ROOT_NAME,
            SOAK_NAMESPACE,
            args.soak_iterations,
        )
        log(
            "throughput: sealing/audit "
            f"{topology['sealing_audit_probe']['sealing_audit_seconds']:.2f}s for "
            f"{topology['sealing_audit_probe']['games_decoded']} games"
        )
        save_stage("throughput", topology)
        soak = load_stage("soak")
        if soak is not None:
            write_benchmark_csv(soak, topology)
            log("throughput: agent_05_training_benchmark.csv written")

    if "artifacts" in stages:
        contract = load_stage("contract")
        resume = load_stage("resume")
        soak = load_stage("soak")
        binding = load_stage("binding")
        topology = load_stage("throughput")
        missing = [
            name
            for name, payload in (
                ("contract", contract),
                ("resume", resume),
                ("soak", soak),
                ("binding", binding),
                ("throughput", topology),
            )
            if payload is None
        ]
        if missing:
            log(f"BLOCKED: these stages have not been run yet: {missing}")
            return 2
        tests = targeted_test_results()
        if args.run_pytest:
            suite = run_pytest()
            tests["full_suite"] = suite
            tests["full_suite_green"] = suite["passed"]
        else:
            recorded = load_stage("final_suite")
            tests["full_suite"] = recorded
            tests["full_suite_green"] = bool(recorded and recorded.get("passed"))
        gates = build_gates(contract, resume, soak, binding, topology, tests)
        soak["binding_fixture"] = binding
        soak["throughput"] = topology
        write_json(DATA_DIRECTORY / "agent_05_stability_soak.json", soak)
        acceptance = {
            "phase": 9,
            "agent": 5,
            "status": "PASS" if gates["all_passed"] else "BLOCKED",
            "artifact": "agent_05_acceptance",
            **environment_record(),
            "prerequisites": contract["prerequisites"],
            "corpus": contract["corpus"],
            "storage": contract["storage"],
            "trainer_version": pck.PHASE9_TRAINER_VERSION,
            "checkpoint_version": contract["checkpoint"]["checkpoint_version"],
            "resume": {
                "criterion": resume["criterion"],
                "cpu": resume["devices"].get("cpu"),
                "mps": resume["devices"].get("mps"),
            },
            "soak_totals": soak["totals"],
            "soak_stability": soak["stability"],
            "selection_statement": soak["selection_statement"],
            "archive_members": soak["archive_members"],
            "binding_fixture": binding,
            "throughput": topology,
            "tests": tests,
            **gates,
        }
        write_json(DATA_DIRECTORY / "agent_05_acceptance.json", acceptance)
        log(
            f"artifacts: {gates['passed']}/{gates['total']} gates "
            f"({'PASS' if gates['all_passed'] else 'BLOCKED'})"
        )
        if gates["failed"]:
            log(f"failed gates: {gates['failed']}")
    return 0


def record_final_suite() -> int:
    """Re-run the suite with artifacts present and record the result.

    Two passes are required, exactly as Agents 3 and 4 found: the first run
    happens before the flag is written, so the self-referential artifact test
    fails in it; the second sees the flag and goes green.
    """
    suite = run_pytest()
    save_stage("final_suite", suite)
    path = DATA_DIRECTORY / "agent_05_acceptance.json"
    if path.exists():
        acceptance = read_json(path)
        acceptance["tests"]["full_suite"] = suite
        acceptance["tests"]["full_suite_green"] = suite["passed"]
        acceptance["gates"]["full_suite_green"] = suite["passed"]
        acceptance["passed"] = sum(1 for value in acceptance["gates"].values() if value)
        acceptance["all_passed"] = all(acceptance["gates"].values())
        acceptance["failed"] = [
            name for name, value in acceptance["gates"].items() if not value
        ]
        acceptance["status"] = "PASS" if acceptance["all_passed"] else "BLOCKED"
        acceptance["covers_agent_05_artifact_tests"] = True
        write_json(path, acceptance)
    log(f"final suite: {suite['summary']}")
    return 0 if suite["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
