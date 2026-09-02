#!/usr/bin/env python3
"""Phase 18 Gate G1: the powered independent vs-random confirmation.

P18-D002 reproduced Phase 8 and passed all 42 original gates, but could not
certify one paired margin: at 1,024 setup pairs the two-sided 95% interval is
about +/-0.0116 wide and the margin is 0.010, which leaves an approximately equal
model only ~39% power. The reviewing chat authorised one measurement-only
revision - same margin, same rule, same two checkpoints, a larger independent
bank.

This script is that measurement, and nothing else. It does not train, does not
touch the sealed Phase 8 test split, and does not re-export or select weights: it
refuses outright unless both checkpoints hash to the digests frozen in the
instruction.

Three stages, in order, each of which must finish before the next may start:

* `--freeze` builds the 4,096-pair confirmation bank from its seed namespace,
  audits it against the original bank for canonical *and* reflection-class
  overlap, builds the paired schedule, and writes the contract. Everything the
  analysis will depend on is fixed here, before a single game is played.
* `--run` exports both checkpoints to the frozen evaluation format and plays the
  identical schedule with each, in the arm order frozen in the contract, writing
  one immutable receipt per game.
* `--analyse` averages each pair's two colour-swapped games within an arm, takes
  the candidate-minus-reference difference, and bootstraps those 4,096 paired
  differences with the frozen seed.

The arms share one schedule because they share a policy token: the match ids,
seeds, colours and boards are then identical by construction rather than by
convention, and the schedule digest is compared between arms as proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.evaluation.match_runner import (  # noqa: E402
    ERROR_ILLEGAL_ACTION,
    ON_POLICY_ERROR_QUARANTINE,
)
from stratego.evaluation.match_spec import (  # noqa: E402
    EVALUATION_RULES,
    PAIRING_COLOR_SWAP_SAME_BOARD,
    build_paired_schedule,
    rules_token,
    schedule_digest,
    schedule_matches,
    validate_schedule,
)
from stratego.evaluation.neural_worker import (  # noqa: E402
    BATCH_POLICY_SINGLE,
    DECISION_MODE_GREEDY,
    NEURAL_WORKER_VERSION,
    neural_policy_ref,
    run_neural_schedule,
)
from stratego.evaluation.phase18 import PHASE18_EVALUATION_VERSION  # noqa: E402
from stratego.evaluation.phase18.confirmation_bank import (  # noqa: E402
    CONFIRMATION_BANK_VERSION,
    CONFIRMATION_NAMESPACE,
    CONFIRMATION_PAIRS,
    bank_record,
    bank_root_seed,
    bootstrap_seed,
    build_confirmation_bank,
    pair_class_fingerprint,
    pair_content_fingerprint,
    schedule_root_seed,
    separation_audit,
)
from stratego.evaluation.phase18.noninferiority import (  # noqa: E402
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_REPLICATES,
    DIRECTION_DELTA_MIN,
    assess_margin,
    paired_unit_delta,
)
from stratego.evaluation.phase18.power import PLANNING_SD, plan  # noqa: E402
from stratego.evaluation.registry import policy_ref  # noqa: E402
from stratego.evaluation.setup_bank import SetupBank  # noqa: E402
from stratego.evaluation.statistics import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SEED,
    build_paired_units,
    matchup_seed,
    summarize_matchup,
)

RUN_ID = "G1-RANDOM-CONFIRMATION-2026-A"
WORK_PACKAGE = "phase18_setup_integrated_warmstart"

#: Frozen by the Agent 3 instruction. A mismatch is BLOCKED, never repaired.
ACCEPTED_CHECKPOINT = Path(
    "/Users/brandonwashington/Dev/Github/stratego/gpt_agent/checkpoints/phase8/warmstart_c1_v1.pt"
)
ACCEPTED_SHA256 = "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca"
CANDIDATE_CHECKPOINT = Path(
    "/Users/brandonwashington/Dev/stratego_phase18/g1_control_v1/dry_run_artifacts/warmstart_c1_v1.pt"
)
CANDIDATE_SHA256 = "460a246be32b821a6d6d7feb928b272a4be1014ff55053f329980e21e3be074c"

G1_SOURCE_COMMIT = "66b733ad92324751e30bd7e2a5e373129cbe87c3"
APPROVED_AGENT2_COMMIT = "18409f738613616e364f81ff14814d4648fc92d1"

#: The original bank the confirmation must be independent of.
REFERENCE_BANK_ARTIFACT = REPOSITORY_ROOT / "reports" / "phase_4_data" / "agent_01_setup_bank_v1.json"

#: Frozen statistical design. None of these may move after `--freeze`.
MARGIN = 0.010
SIGNED_MARGIN = -0.010
TARGET_POWER = 0.90

#: Frozen arm order. The reference is measured first; the first arm's result may
#: not alter or cancel the second.
ARM_ORDER = ("reference", "candidate")

CANDIDATE_TOKEN_ID = "c1_warmstart"
RANDOM_OPPONENT_ID = "random_legal"
GATE_DTYPE = "float32"


class ConfirmationError(RuntimeError):
    """A frozen identity, accounting or sealing precondition failed."""


def log(message: str) -> None:
    print(f"[g1-confirm {time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    return path


def git_output(*arguments: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def git_commit() -> str:
    return git_output("rev-parse", "HEAD")


def git_tree() -> str:
    return git_output("rev-parse", "HEAD^{tree}")


def git_porcelain() -> str:
    return git_output("status", "--porcelain")


def environment() -> dict:
    porcelain = git_porcelain()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executing_tree": str(REPOSITORY_ROOT),
        "source_commit": git_commit(),
        "source_tree": git_tree(),
        "working_tree_state": "clean" if not porcelain else f"dirty ({len(porcelain.splitlines())} paths)",
    }


#: The accepted installation the frozen checkpoint paths live in. Derived from
#: the frozen accepted-checkpoint path, not typed, so an execution worktree and
#: the main checkout agree on where the protected bytes are.
ACCEPTED_INSTALL_ROOT = ACCEPTED_CHECKPOINT.parents[2]

#: Re-hashed before and after every measuring stage; a change is BLOCKED.
PROTECTED = (
    "checkpoints/phase8/warmstart_c1_v1.pt",
    "checkpoints/phase8/warmstart_c1_v1_manifest.json",
    "checkpoints/phase8/warmstart_c1_v1_initialisation.pt",
    "reports/phase_8_implementation_report.md",
)


def protected_digests() -> dict:
    """The accepted Phase 8 artifacts plus the frozen G1 candidate, by digest."""
    digests = {}
    for name in PROTECTED:
        path = ACCEPTED_INSTALL_ROOT / name
        if not path.exists():
            raise ConfirmationError(f"BLOCKED: accepted Phase 8 artifact {path} is missing")
        digests[name] = sha256(path)
    if not CANDIDATE_CHECKPOINT.exists():
        raise ConfirmationError(
            f"BLOCKED: the G1 candidate checkpoint {CANDIDATE_CHECKPOINT} is missing"
        )
    digests["g1_candidate:warmstart_c1_v1.pt"] = sha256(CANDIDATE_CHECKPOINT)
    return digests


#: The decision trail this work package executes under. Hashed from the tree the
#: stage runs in, so the worktree proves it carries the same authorization the
#: main checkout was reviewed with.
AUTHORIZATION_FILES = (
    "reports/phase18/decisions/P18-D002.json",
    "reports/phase18/decisions/P18-D002.md",
    "reports/phase18/reviews/P18-D002_REVIEW.md",
    "instructions/phase_18_setup_integrated_warmstart/05_AGENT_3_G1_RANDOM_NONINFERIORITY_CONFIRMATION.md",
)


def authorization_digests() -> dict:
    digests = {}
    for name in AUTHORIZATION_FILES:
        path = REPOSITORY_ROOT / name
        if not path.exists():
            raise ConfirmationError(f"BLOCKED: authorization artifact {name} is missing")
        digests[name] = sha256(path)
    return digests


# ---------------------------------------------------------------------------
# Frozen identities
# ---------------------------------------------------------------------------


def verify_checkpoints() -> dict:
    """Both checkpoints, by digest. Any mismatch stops the work package."""
    record = {}
    for label, path, expected in (
        ("reference", ACCEPTED_CHECKPOINT, ACCEPTED_SHA256),
        ("candidate", CANDIDATE_CHECKPOINT, CANDIDATE_SHA256),
    ):
        if not path.exists():
            raise ConfirmationError(f"BLOCKED: {label} checkpoint is missing at {path}")
        observed = sha256(path)
        if observed != expected:
            raise ConfirmationError(
                f"BLOCKED: {label} checkpoint {path} hashes to {observed}, not the "
                f"frozen {expected}; do not repair, regenerate or substitute it"
            )
        record[label] = {
            "path": str(path),
            "sha256": observed,
            "matches_frozen_identity": True,
            "bytes": path.stat().st_size,
        }
    return record


def schedule_for(pairs: int) -> tuple:
    """The one schedule both arms play."""
    candidate = neural_policy_ref(CANDIDATE_TOKEN_ID, dtype_name=GATE_DTYPE)
    opponent = policy_ref(RANDOM_OPPONENT_ID)
    units = build_paired_schedule(
        candidate,
        opponent,
        range(pairs),
        root_seed=schedule_root_seed(),
        setup_bank_version=CONFIRMATION_BANK_VERSION,
        pairing_mode=PAIRING_COLOR_SWAP_SAME_BOARD,
        rules=EVALUATION_RULES,
    )
    return units, schedule_matches(units), candidate, opponent


# ---------------------------------------------------------------------------
# Stage 1: freeze
# ---------------------------------------------------------------------------


def stage_freeze(output: Path, pairs: int = CONFIRMATION_PAIRS) -> dict:
    """Build and audit the bank and schedule, and write the frozen contract."""
    started = time.perf_counter()
    checkpoints = verify_checkpoints()
    log("checkpoint digests match the frozen identities")

    bank = build_confirmation_bank(pairs)
    reference_bank = SetupBank.from_json(REFERENCE_BANK_ARTIFACT.read_text())
    audit = separation_audit(bank, reference_bank)
    log(
        f"bank {bank.digest()[:16]}: {audit.confirmation_pairs} pairs, "
        f"separated={audit.separated}"
    )
    if not audit.separated:
        raise ConfirmationError(
            f"BLOCKED: the confirmation bank is not independent: {audit.to_dict()}"
        )
    if audit.confirmation_pairs != pairs:
        raise ConfirmationError(
            f"BLOCKED: expected {pairs} pairs, built {audit.confirmation_pairs}"
        )

    units, matches, candidate, opponent = schedule_for(pairs)
    problems = validate_schedule(matches, bank)
    if problems:
        raise ConfirmationError(f"BLOCKED: schedule problems: {problems[:5]}")
    if len(matches) != pairs * 2:
        raise ConfirmationError(
            f"BLOCKED: expected {pairs * 2} matches, built {len(matches)}"
        )

    power = plan(pairs, PLANNING_SD, MARGIN, confidence=BOOTSTRAP_CONFIDENCE,
                 target_power=TARGET_POWER, true_delta=0.0)

    contract = {
        "artifact": "phase18_g1_random_confirmation_contract_v1",
        "work_package": WORK_PACKAGE,
        "agent": "phase_18_agent_3",
        "gate": "G1",
        "run_id": RUN_ID,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authorizing_decision": "P18-D002 accepted as REVISE",
        "primary_hypothesis": {
            "delta": "EWR(G1 candidate vs random) - EWR(accepted checkpoint vs random)",
            "null": "Delta <= -0.010",
            "alternative": "Delta > -0.010",
            "decision_rule": (
                "pass only if the lower endpoint of the two-sided 95% paired "
                "percentile-bootstrap interval is strictly greater than -0.010"
            ),
            "margin": SIGNED_MARGIN,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "replicates": BOOTSTRAP_REPLICATES,
            "resampling_unit": "paired setup case, both colours carried together",
            "score": "effective win rate, draw = 0.5",
            "primary_sample": "the confirmation bank only; the original 1,024 pairs are context and are never pooled",
            "normal_approximation": "diagnostic only; it may not decide the gate",
        },
        "power": power.to_dict(),
        "bank": {
            "bank_version": CONFIRMATION_BANK_VERSION,
            "namespace": CONFIRMATION_NAMESPACE,
            "seed_function": "stratego.setups.identity.derive_stream_seed",
            "root_seed": bank_root_seed(),
            "pair_count": pairs,
            "digest": bank.digest(),
            "separation_audit": audit.to_dict(),
            "reference_bank": {
                "artifact": str(REFERENCE_BANK_ARTIFACT.relative_to(REPOSITORY_ROOT)),
                "bank_version": reference_bank.bank_version,
                "pair_count": len(reference_bank.pairs),
                "sha256": sha256(REFERENCE_BANK_ARTIFACT),
            },
        },
        "schedule": {
            "digest": schedule_digest(matches),
            "paired_units": len(units),
            "matches": len(matches),
            "games_per_arm": len(matches),
            "total_games_both_arms": len(matches) * 2,
            "root_seed": schedule_root_seed(),
            "candidate": candidate.to_dict(),
            "candidate_token": candidate.token,
            "opponent": opponent.to_dict(),
            "opponent_token": opponent.token,
            "pairing_mode": PAIRING_COLOR_SWAP_SAME_BOARD,
            "rules": rules_token(EVALUATION_RULES),
            "decision_mode": DECISION_MODE_GREEDY,
            "dtype": GATE_DTYPE,
            "batch_policy": BATCH_POLICY_SINGLE,
            "neural_worker_version": NEURAL_WORKER_VERSION,
            "note": (
                "both arms play this identical schedule; they share a policy token, so "
                "match ids, seeds, colours and boards are identical by construction"
            ),
        },
        "arm_order": list(ARM_ORDER),
        "arm_order_rule": (
            "frozen before either arm runs; the first arm's result may not alter or "
            "cancel the second"
        ),
        "bootstrap_seed": bootstrap_seed(),
        "checkpoints": checkpoints,
        "source": {
            "g1_source_commit": G1_SOURCE_COMMIT,
            "approved_agent2_commit": APPROVED_AGENT2_COMMIT,
        },
        "authorization": authorization_digests(),
        "sealed_test_access": {
            "planned": 0,
            "rule": "this work package opens no Phase 8 sealed test data at all",
        },
        "accounting_rule": "planned = completed + failed + missing; failures and retries are never draws",
        "evaluator": PHASE18_EVALUATION_VERSION,
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output / "phase18_g1_random_confirmation_contract_v1.json", contract)
    write_json(output / "phase18_g1_random_confirmation_bank_v1.json", bank_record(bank))
    log(f"contract and bank frozen under {output}")
    return contract


# ---------------------------------------------------------------------------
# Stage 2: run the arms
# ---------------------------------------------------------------------------


def export_checkpoint(source: Path, destination: Path) -> dict:
    """Bridge a warmstart checkpoint into the frozen evaluation format."""
    import torch

    from stratego.model.checkpoint import load_checkpoint, save_checkpoint
    from stratego.training.warmstart_checkpoint import load_model_for_evaluation
    from stratego.training.warmstart_pilot import model_state_checksum
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config

    model, _ = load_model_for_evaluation(source, device="cpu")
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, destination)
    reloaded, metadata = load_checkpoint(
        destination,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
    )
    source_state, reloaded_state = model.state_dict(), reloaded.state_dict()
    bitwise = set(source_state) == set(reloaded_state) and all(
        torch.equal(source_state[name], reloaded_state[name]) for name in source_state
    )
    if not bitwise:
        raise ConfirmationError(f"BLOCKED: exporting {source} changed the weights")
    record = {
        "source": str(source),
        "source_sha256": sha256(source),
        "export": str(destination),
        "export_sha256": sha256(destination),
        "state_dict_digest": metadata.get("state_dict_digest"),
        "model_state_checksum": model_state_checksum(source_state),
        "bitwise_state_dict_match": True,
        "parameter_count": reloaded.parameter_count(),
    }
    del model, reloaded
    return record


def play_chunks(matches: tuple, directory: Path, runner, *, chunk_units: int, label: str) -> tuple:
    """Play a schedule in resumable chunks; the retry-safe path.

    A chunk file is keyed by its position and its own schedule digest, so a
    completed chunk is reused byte-for-byte on a rerun and a chunk written for
    any other schedule can never be picked up by name collision. A rerun after a
    crash therefore replays exactly the missing chunks: nothing completed is
    replayed, and no receipt or chunk file is ever deleted to make the retry
    possible. On reuse the stored rows are still checked against the chunk's
    match ids, so a corrupted or foreign file refuses instead of scoring.

    `runner(chunk)` returns `(results, report)`; injecting it is what makes this
    path testable without playing games.
    """
    directory.mkdir(parents=True, exist_ok=True)
    results, reports = [], []
    for index in range(0, len(matches), chunk_units * 2):
        chunk = matches[index : index + chunk_units * 2]
        number = index // (chunk_units * 2)
        digest = schedule_digest(chunk)[:16]
        path = directory / f"chunk_{number:04d}_{digest}.pkl"
        if path.exists():
            with open(path, "rb") as stream:
                stored = pickle.load(stream)
            expected_ids = sorted(spec.match_id for spec in chunk)
            stored_ids = sorted(row.match_id for row in stored["results"])
            if stored_ids != expected_ids:
                raise ConfirmationError(
                    f"BLOCKED: {label} chunk {number} at {path} holds different "
                    "match ids than the frozen schedule; refusing to reuse it"
                )
            results.extend(stored["results"])
            reports.append(stored["report"] | {"reused": True})
            continue
        chunk_results, report = runner(chunk)
        report = report | {"chunk": number, "reused": False}
        with open(path, "wb") as stream:
            pickle.dump({"results": chunk_results, "report": report}, stream)
        results.extend(chunk_results)
        reports.append(report)
        log(f"{label}: {len(results)}/{len(matches)} games (chunk {number})")
    return tuple(results), reports


def run_arm(
    *,
    label: str,
    checkpoint: Path,
    bank: SetupBank,
    matches: tuple,
    candidate_ref,
    work: Path,
    workers: int,
    chunk_units: int,
) -> dict:
    """Play the frozen schedule with one arm's weights."""
    from stratego.evaluation.neural_worker import InferenceOwner
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config

    work.mkdir(parents=True, exist_ok=True)
    export = export_checkpoint(checkpoint, work / "eval_weights.pt")
    log(f"{label}: exported weights {export['export_sha256'][:16]}")

    owner = InferenceOwner(
        work / "eval_weights.pt",
        decision_mode=DECISION_MODE_GREEDY,
        device="mps",
        dtype=GATE_DTYPE,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
        name=f"phase18_confirm_{label}",
    )
    started = time.perf_counter()

    def runner(chunk):
        run = run_neural_schedule(
            chunk,
            bank,
            owner,
            policy_ref=candidate_ref,
            worker_count=workers,
            record_actions=False,
            on_policy_error=ON_POLICY_ERROR_QUARANTINE,
        )
        report = {
            "matches": run.matches_run,
            "decisions": run.decisions,
            "wall_clock_seconds": round(run.wall_clock_seconds, 3),
            "policy_errors": run.policy_errors,
            "illegal_policy_actions": run.illegal_policy_actions,
            "workers_importing_torch": run.workers_importing_torch,
            "worker_checkpoint_loads": run.worker_checkpoint_loads,
            "inference_failures": int(run.inference.get("failures_returned", 0)),
            "results_digest": run.results_digest,
        }
        return run.results, report

    try:
        results, reports = play_chunks(
            matches, work / "games", runner, chunk_units=chunk_units, label=label
        )
        owner_identity = owner.identity()
        owner_stats = owner.stats()
    finally:
        owner.close()

    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "export": export,
        "owner_identity": owner_identity,
        "owner_stats": owner_stats,
        "chunks": reports,
        "results": tuple(results),
        "wall_seconds": round(time.perf_counter() - started, 3),
    }


def write_receipts(path: Path, arm: dict, bank: SetupBank) -> dict:
    """One immutable row per game, self-sufficient for audit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = {
        pair.setup_pair_id: (
            pair_content_fingerprint(pair),
            pair_class_fingerprint(pair),
        )
        for pair in bank.pairs
    }
    written = 0
    with open(path, "w") as stream:
        for row in sorted(arm["results"], key=lambda r: r.match_id):
            content, reflection = classes[row.setup_pair_id]
            stream.write(
                json.dumps(
                    {
                        "run_id": RUN_ID,
                        "g1_source_commit": G1_SOURCE_COMMIT,
                        "arm": arm["label"],
                        "checkpoint_sha256": arm["checkpoint_sha256"],
                        "match_id": row.match_id,
                        "paired_unit_id": row.paired_unit_id,
                        "setup_pair_id": row.setup_pair_id,
                        "pair_content_fingerprint": content,
                        "pair_reflection_class_fingerprint": reflection,
                        "setup_bank_version": row.setup_bank_version,
                        "red_setup": row.red_setup,
                        "blue_setup": row.blue_setup,
                        "candidate_color": row.candidate_color,
                        "first_player": row.first_player,
                        "candidate_seed": row.candidate_seed,
                        "opponent_seed": row.opponent_seed,
                        "root_seed": row.root_seed,
                        "candidate_policy": f"{row.candidate_policy_id}@{row.candidate_policy_version}",
                        "opponent_policy": f"{row.opponent_policy_id}@{row.opponent_policy_version}",
                        "rules": row.rules,
                        "winner": row.winner,
                        "draw": row.draw,
                        "candidate_result": row.candidate_result,
                        "candidate_score": row.candidate_score,
                        "terminal_reason": row.terminal_reason,
                        "plies": row.plies,
                        "decisions": row.decisions,
                        "replay_digest": row.replay_digest,
                        "errored": row.errored,
                        "policy_error": row.policy_error,
                        "policy_error_category": row.policy_error_category,
                        "illegal_action": row.policy_error_category == ERROR_ILLEGAL_ACTION,
                        "retry_lineage": [],
                        "wall_clock_seconds": row.wall_clock_seconds,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            written += 1
    return {"path": str(path), "rows": written, "sha256": sha256(path)}


# ---------------------------------------------------------------------------
# Accounting and identity proofs
# ---------------------------------------------------------------------------


def reconcile(matches: tuple, rows) -> dict:
    """`planned = completed + failed + missing`, with nothing unexplained.

    A failed or missing game is a failed or missing game. It never becomes a
    draw, never becomes a pass, and `complete_for_primary` is the only flag the
    analysis may read before opening the paired statistic.
    """
    planned_ids = [spec.match_id for spec in matches]
    planned_set = set(planned_ids)
    if len(planned_set) != len(planned_ids):
        raise ConfirmationError("BLOCKED: the frozen schedule repeats a match id")
    by_id: dict = {}
    duplicates = []
    for row in rows:
        if row.match_id in by_id:
            duplicates.append(row.match_id)
        else:
            by_id[row.match_id] = row
    unplanned = sorted(set(by_id) - planned_set)
    missing = sorted(planned_set - set(by_id))
    failed = sorted(match_id for match_id, row in by_id.items() if row.errored)
    completed = [
        match_id
        for match_id, row in by_id.items()
        if match_id in planned_set and not row.errored
    ]
    counts_reconcile = len(planned_ids) == len(completed) + len(failed) + len(missing)
    return {
        "planned": len(planned_ids),
        "completed": len(completed),
        "failed": len(failed),
        "missing": len(missing),
        "unplanned": len(unplanned),
        "duplicates": len(duplicates),
        "failed_ids": failed[:20],
        "missing_ids": missing[:20],
        "unplanned_ids": unplanned[:20],
        "duplicate_ids": sorted(duplicates)[:20],
        "reconciles": counts_reconcile and not unplanned and not duplicates,
        "complete_for_primary": (
            counts_reconcile
            and not failed
            and not missing
            and not unplanned
            and not duplicates
        ),
        "rule": "planned = completed + failed + missing; failures and retries are never draws and never passes",
    }


#: Per-game fields that must be identical between the arms for the same match
#: id. Everything the case is - board, seeds, colour, opponent, rules - and
#: nothing the model did.
ARM_INVARIANT_FIELDS = (
    "paired_unit_id",
    "setup_pair_id",
    "setup_bank_version",
    "red_setup",
    "blue_setup",
    "candidate_color",
    "first_player",
    "root_seed",
    "candidate_seed",
    "opponent_seed",
    "opponent_policy_id",
    "opponent_policy_version",
    "rules",
)


def prove_arm_identity(candidate_rows, reference_rows) -> dict:
    """The two arms played the same cases, or the comparison is not paired."""
    problems: list = []
    candidate_by_id = {row.match_id: row for row in candidate_rows}
    reference_by_id = {row.match_id: row for row in reference_rows}
    if sorted(candidate_by_id) != sorted(reference_by_id):
        problems.append(
            f"the arms played different match ids: {len(candidate_by_id)} vs "
            f"{len(reference_by_id)} rows"
        )
    mismatches: dict = {}
    if not problems:
        for match_id, candidate in candidate_by_id.items():
            reference = reference_by_id[match_id]
            for field in ARM_INVARIANT_FIELDS:
                if getattr(candidate, field) != getattr(reference, field):
                    mismatches[field] = mismatches.get(field, 0) + 1
    if mismatches:
        problems.append(f"case identity differs between arms: {mismatches}")
    return {
        "rows_per_arm": len(candidate_by_id),
        "identical_match_ids": sorted(candidate_by_id) == sorted(reference_by_id),
        "fields_checked": list(ARM_INVARIANT_FIELDS),
        "field_mismatches": mismatches,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Frozen-contract verification
# ---------------------------------------------------------------------------


CONTRACT_NAME = "phase18_g1_random_confirmation_contract_v1.json"
BANK_NAME = "phase18_g1_random_confirmation_bank_v1.json"
LAUNCH_NAME = "phase18_g1_random_confirmation_launch_v1.json"


def load_frozen_contract(frozen_dir: Path) -> tuple:
    path = frozen_dir / CONTRACT_NAME
    if not path.exists():
        raise ConfirmationError(
            f"BLOCKED: no frozen contract at {path}; run --freeze and commit first"
        )
    return json.loads(path.read_text()), sha256(path)


def verify_frozen_identity(contract: dict) -> tuple:
    """Rebuild every frozen identity and refuse on any drift.

    The run and the analysis both call this, so neither can proceed on a bank,
    schedule, seed or checkpoint that is not byte-for-byte the one the contract
    froze before outcomes existed.
    """
    checkpoints = verify_checkpoints()
    for label in ("reference", "candidate"):
        frozen = contract["checkpoints"][label]["sha256"]
        if checkpoints[label]["sha256"] != frozen:
            raise ConfirmationError(
                f"BLOCKED: {label} checkpoint digest moved from the frozen {frozen}"
            )
    pairs = int(contract["bank"]["pair_count"])
    if pairs != CONFIRMATION_PAIRS:
        raise ConfirmationError(
            f"BLOCKED: the contract froze {pairs} pairs, not {CONFIRMATION_PAIRS}"
        )
    if int(contract["bank"]["root_seed"]) != bank_root_seed():
        raise ConfirmationError("BLOCKED: the bank root seed does not re-derive")
    if int(contract["bootstrap_seed"]) != bootstrap_seed():
        raise ConfirmationError("BLOCKED: the bootstrap seed does not re-derive")
    bank = build_confirmation_bank(pairs)
    if bank.digest() != contract["bank"]["digest"]:
        raise ConfirmationError(
            f"BLOCKED: the rebuilt bank digests to {bank.digest()}, not the frozen "
            f"{contract['bank']['digest']}"
        )
    units, matches, candidate_ref, opponent_ref = schedule_for(pairs)
    observed = schedule_digest(matches)
    if observed != contract["schedule"]["digest"]:
        raise ConfirmationError(
            f"BLOCKED: the rebuilt schedule digests to {observed}, not the frozen "
            f"{contract['schedule']['digest']}"
        )
    if len(matches) != int(contract["schedule"]["matches"]):
        raise ConfirmationError("BLOCKED: the rebuilt schedule has the wrong size")
    problems = validate_schedule(matches, bank)
    if problems:
        raise ConfirmationError(f"BLOCKED: schedule problems: {problems[:5]}")
    return bank, units, matches, candidate_ref, opponent_ref, checkpoints


# ---------------------------------------------------------------------------
# Stage 2: run both arms over the frozen schedule
# ---------------------------------------------------------------------------


def stage_run(output: Path, frozen_dir: Path, *, workers: int, chunk_units: int) -> dict:
    contract, contract_digest = load_frozen_contract(frozen_dir)
    protected_before = protected_digests()
    bank, _units, matches, candidate_ref, _opponent_ref, checkpoints = (
        verify_frozen_identity(contract)
    )
    log(f"frozen identities verified against contract {contract_digest[:16]}")

    arm_order = tuple(contract["arm_order"])
    if arm_order != ARM_ORDER:
        raise ConfirmationError(
            f"BLOCKED: contract arm order {arm_order} is not the frozen {ARM_ORDER}"
        )
    sources = {"reference": ACCEPTED_CHECKPOINT, "candidate": CANDIDATE_CHECKPOINT}

    started = time.perf_counter()
    ledger: dict = {}
    for label in arm_order:
        log(f"running the {label} arm over {len(matches)} games")
        arm = run_arm(
            label=label,
            checkpoint=sources[label],
            bank=bank,
            matches=matches,
            candidate_ref=candidate_ref,
            work=output / "arms" / label,
            workers=workers,
            chunk_units=chunk_units,
        )
        receipts = write_receipts(
            output / "receipts" / f"{label}_receipts.jsonl", arm, bank
        )
        accounting = reconcile(matches, arm["results"])
        payload = {
            "artifact": f"phase18_g1_random_confirmation_arm_{label}",
            "run_id": RUN_ID,
            "work_package": WORK_PACKAGE,
            "gate": "G1",
            "arm": label,
            "g1_confirm_source_commit": git_commit(),
            "contract_sha256": contract_digest,
            "schedule_digest": contract["schedule"]["digest"],
            "bank_digest": contract["bank"]["digest"],
            "checkpoint": {k: v for k, v in arm.items() if k in ("checkpoint", "checkpoint_sha256")},
            "export": arm["export"],
            "owner_identity": arm["owner_identity"],
            "owner_stats": arm["owner_stats"],
            "chunks": arm["chunks"],
            "accounting": accounting,
            "receipts": receipts,
            "wall_seconds": arm["wall_seconds"],
            "environment": environment(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_json(output / f"arm_{label}_v1.json", payload)
        ledger[label] = payload
        if not accounting["reconciles"]:
            raise ConfirmationError(
                f"BLOCKED: the {label} arm does not reconcile: {accounting}"
            )

    protected_after = protected_digests()
    if protected_before != protected_after:
        raise ConfirmationError(
            "BLOCKED: a protected accepted artifact changed while the arms ran"
        )
    run_record = {
        "artifact": "phase18_g1_random_confirmation_run_v1",
        "run_id": RUN_ID,
        "work_package": WORK_PACKAGE,
        "g1_confirm_source_commit": git_commit(),
        "contract_sha256": contract_digest,
        "arm_order": list(arm_order),
        "arms": {label: ledger[label]["accounting"] for label in arm_order},
        "protected_artifacts": {
            "before": protected_before,
            "after": protected_after,
            "unchanged": True,
        },
        "sealed_test_access": {"examples_opened": 0},
        "seconds": round(time.perf_counter() - started, 3),
        "environment": environment(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(output / "run_v1.json", run_record)
    log("both arms complete and reconciled")
    return run_record


# ---------------------------------------------------------------------------
# Stage 3: analysis
# ---------------------------------------------------------------------------


def load_arm_rows(output: Path, label: str) -> tuple:
    directory = output / "arms" / label / "games"
    rows = []
    for path in sorted(directory.glob("chunk_*.pkl")):
        with open(path, "rb") as stream:
            rows.extend(pickle.load(stream)["results"])
    if not rows:
        raise ConfirmationError(f"BLOCKED: no persisted results under {directory}")
    return tuple(rows)


def strict_verdict(interval, signed_margin: float) -> dict:
    """The frozen decision: pass only if `lower` is STRICTLY above the margin.

    `assess_margin` implements the original G1 contract's `>=` reading; the
    Agent 3 instruction freezes `strictly greater than -0.010`, so the
    confirmation decides here and reports `assess_margin` beside it as the
    original dialect. On dyadic game scores the two cannot actually disagree,
    but the decision implements the instruction's words, not an argument about
    reachability.
    """
    return {
        "metric": "vs_random_ewr",
        "margin": float(signed_margin),
        "deciding_bound_name": "lower",
        "deciding_bound": float(interval.lower),
        "delta": float(interval.delta),
        "rule": (
            "pass only if the lower endpoint of the two-sided 95% paired "
            "percentile-bootstrap interval is strictly greater than -0.010"
        ),
        "non_inferior": bool(interval.lower > signed_margin),
    }


def recompute_from_receipts(path: Path) -> dict:
    """Rebuild each pair's colour scores from the immutable per-row receipts."""
    from stratego.engine.constants import BLUE, RED

    pairs: dict = {}
    rows = 0
    with open(path) as stream:
        for line in stream:
            row = json.loads(line)
            rows += 1
            if row["errored"] or row["candidate_score"] is None:
                raise ConfirmationError(
                    f"BLOCKED: receipt {row['match_id']} carries no score; the "
                    "primary analysis is invalid"
                )
            slot = pairs.setdefault(row["setup_pair_id"], {})
            color = int(row["candidate_color"])
            if color in slot:
                raise ConfirmationError(
                    f"BLOCKED: receipts repeat colour {color} for pair "
                    f"{row['setup_pair_id']}"
                )
            slot[color] = float(row["candidate_score"])
    scores = {}
    for pair_id, slot in pairs.items():
        if sorted(slot) != sorted((RED, BLUE)):
            raise ConfirmationError(
                f"BLOCKED: pair {pair_id} is missing a colour in the receipts"
            )
        scores[pair_id] = (slot[RED] + slot[BLUE]) / 2.0
    return {"rows": rows, "unit_scores": scores}


def stage_analyse(output: Path, frozen_dir: Path, reports_dir: Path) -> dict:
    from stratego.evaluation.statistics import _normal_quantile

    contract, contract_digest = load_frozen_contract(frozen_dir)
    protected_before = protected_digests()
    bank, _units, matches, _candidate_ref, _opponent_ref, checkpoints = (
        verify_frozen_identity(contract)
    )

    arms: dict = {}
    for label in ARM_ORDER:
        rows = load_arm_rows(output, label)
        accounting = reconcile(matches, rows)
        if not accounting["complete_for_primary"]:
            raise ConfirmationError(
                f"BLOCKED: the {label} arm is incomplete; the primary analysis is "
                f"invalid until every paired unit has both colours: {accounting}"
            )
        matchup = f"{rows[0].candidate.token} vs {rows[0].opponent.token}"
        summary = summarize_matchup(
            rows,
            seed=matchup_seed(DEFAULT_BOOTSTRAP_SEED, matchup),
            allow_policy_errors=False,
            include_setup_table=False,
        ).to_dict()
        units = build_paired_units(rows, allow_policy_errors=False)
        if len(units) != CONFIRMATION_PAIRS:
            raise ConfirmationError(
                f"BLOCKED: the {label} arm produced {len(units)} paired units, "
                f"not {CONFIRMATION_PAIRS}"
            )
        receipts_path = output / "receipts" / f"{label}_receipts.jsonl"
        receipts = recompute_from_receipts(receipts_path)
        by_pair = {unit.setup_pair_id: unit for unit in units}
        if receipts["unit_scores"] != {p: u.score for p, u in by_pair.items()}:
            raise ConfirmationError(
                f"BLOCKED: {label}: the receipts do not reproduce the paired unit "
                "scores exactly"
            )
        arms[label] = {
            "rows": rows,
            "accounting": accounting,
            "summary": summary,
            "units": by_pair,
            "receipts_rows": receipts["rows"],
            "receipts_sha256": sha256(receipts_path),
            "matchup": matchup,
        }

    identity = prove_arm_identity(arms["candidate"]["rows"], arms["reference"]["rows"])
    if identity["problems"]:
        raise ConfirmationError(f"BLOCKED: arm identity failed: {identity['problems']}")

    pair_ids = sorted(arms["candidate"]["units"])
    if pair_ids != sorted(arms["reference"]["units"]):
        raise ConfirmationError("BLOCKED: the arms carry different paired units")
    candidate_scores = [arms["candidate"]["units"][p].score for p in pair_ids]
    reference_scores = [arms["reference"]["units"][p].score for p in pair_ids]

    seed = int(contract["bootstrap_seed"])
    interval = paired_unit_delta(
        candidate_scores,
        reference_scores,
        seed=seed,
        replicates=BOOTSTRAP_REPLICATES,
        confidence=BOOTSTRAP_CONFIDENCE,
    )
    verdict = strict_verdict(interval, SIGNED_MARGIN)
    original_dialect = assess_margin(
        "vs_random_ewr", interval, margin=SIGNED_MARGIN, direction=DIRECTION_DELTA_MIN
    ).to_dict()

    differences = [c - r for c, r in zip(candidate_scores, reference_scores)]
    n = len(differences)
    mean = sum(differences) / n
    observed_sd = (sum((d - mean) ** 2 for d in differences) / (n - 1)) ** 0.5
    z = _normal_quantile(1.0 - (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0)
    standard_error = observed_sd / (n**0.5)
    normal_check = {
        "role": "diagnostic only; it may not decide the gate",
        "standard_error": standard_error,
        "half_width": z * standard_error,
        "lower": interval.delta - z * standard_error,
        "upper": interval.delta + z * standard_error,
    }

    original = json.loads((frozen_dir / "phase18_g1_noninferiority_v1.json").read_text())
    original_vs_random = original["paired_non_inferiority"]["comparisons"]["vs_random_ewr"]

    protected_after = protected_digests()
    if protected_before != protected_after:
        raise ConfirmationError("BLOCKED: a protected accepted artifact changed")

    identity_block = {
        "run_id": RUN_ID,
        "work_package": WORK_PACKAGE,
        "gate": "G1",
        "agent": "phase_18_agent_3",
        "g1_confirm_source_commit": git_commit(),
        "g1_source_commit": G1_SOURCE_COMMIT,
        "approved_agent2_commit": APPROVED_AGENT2_COMMIT,
        "contract_sha256": contract_digest,
        "bank_digest": contract["bank"]["digest"],
        "schedule_digest": contract["schedule"]["digest"],
        "checkpoints": {
            label: checkpoints[label]["sha256"] for label in ("reference", "candidate")
        },
        "evaluator": PHASE18_EVALUATION_VERSION,
    }

    for label in ARM_ORDER:
        arm = arms[label]
        write_json(
            reports_dir / f"phase18_g1_random_confirmation_{label}_v1.json",
            identity_block
            | {
                "artifact": f"phase18_g1_random_confirmation_{label}_v1",
                "arm": label,
                "matchup": arm["matchup"],
                "accounting": arm["accounting"],
                "summary": arm["summary"],
                "receipts_rows": arm["receipts_rows"],
                "receipts_sha256": arm["receipts_sha256"],
                "environment": environment(),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

    result = identity_block | {
        "artifact": "phase18_g1_random_confirmation_noninferiority_v1",
        "primary": {
            "hypothesis": contract["primary_hypothesis"],
            "interval": interval.to_dict(),
            "verdict": verdict,
            "original_gate_dialect_cross_check": original_dialect,
            "bootstrap_seed": seed,
            "bootstrap_seed_rule": (
                f"derive_stream_seed('{CONFIRMATION_NAMESPACE}', 'paired_bootstrap')"
            ),
        },
        "diagnostics": {
            "observed_paired_difference_sd": observed_sd,
            "planning_sd": PLANNING_SD,
            "pairs_on_which_arms_differ": sum(1 for d in differences if d != 0.0),
            "paired_units": n,
            "normal_approximation": normal_check,
            "color_split": {
                label: arms[label]["summary"]["color_split"] for label in ARM_ORDER
            },
            "terminal_reasons": {
                label: arms[label]["summary"]["terminal_reasons"] for label in ARM_ORDER
            },
            "arm_effective_win_rates": {
                label: arms[label]["summary"]["effective_win_rate"] for label in ARM_ORDER
            },
            "receipts_recompute_headline_exactly": True,
        },
        "arm_identity_proof": identity,
        "accounting": {label: arms[label]["accounting"] for label in ARM_ORDER},
        "original_1024_pair_context": {
            "role": (
                "historical evidence only; never pooled into the primary statistic "
                "and never used to alter the gate"
            ),
            "delta": original_vs_random["delta"],
            "lower": original_vs_random["lower"],
            "upper": original_vs_random["upper"],
            "sample_size": original_vs_random["sample_size"],
        },
        "power": contract["power"],
        "sealed_test_access": {
            "examples_opened": 0,
            "multiplicity_increment": 0,
            "rule": "this work package opens no Phase 8 sealed test data at all",
        },
        "protected_artifacts": {
            "before": protected_before,
            "after": protected_after,
            "unchanged": True,
        },
        "environment": environment(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(
        reports_dir / "phase18_g1_random_confirmation_noninferiority_v1.json", result
    )
    log(
        f"delta {interval.delta:+.9f}, 95% [{interval.lower:+.9f}, "
        f"{interval.upper:+.9f}], margin {SIGNED_MARGIN}: "
        f"{'PASS' if verdict['non_inferior'] else 'FAIL'}"
    )
    return result


# ---------------------------------------------------------------------------
# Launch manifest and binding ledger
# ---------------------------------------------------------------------------


#: Everything the measurement executes through, hashed into the launch manifest.
EVALUATOR_SOURCES = (
    "scripts/phase18_g1_random_confirmation.py",
    "stratego/evaluation/phase18/__init__.py",
    "stratego/evaluation/phase18/confirmation_bank.py",
    "stratego/evaluation/phase18/noninferiority.py",
    "stratego/evaluation/phase18/power.py",
    "stratego/evaluation/match_spec.py",
    "stratego/evaluation/match_runner.py",
    "stratego/evaluation/neural_worker.py",
    "stratego/evaluation/setup_bank.py",
    "stratego/evaluation/statistics.py",
    "stratego/setups/identity.py",
)

TEST_SOURCES = (
    "tests/evaluation/phase18/test_g1_harness.py",
    "tests/evaluation/phase18/test_noninferiority.py",
    "tests/evaluation/phase18/test_random_confirmation.py",
)


def stage_launch_manifest(frozen_dir: Path, reports_dir: Path) -> dict:
    """Bind the frozen measurement to one clean source commit, before any game.

    Run from the detached execution worktree. Its porcelain must be empty and
    the contract it binds must be the committed copy in the same tree.
    """
    porcelain = git_porcelain()
    if porcelain:
        raise ConfirmationError(
            f"BLOCKED: the execution worktree is not clean:\n{porcelain}"
        )
    contract, contract_digest = load_frozen_contract(frozen_dir)
    bank_path = frozen_dir / BANK_NAME
    if not bank_path.exists():
        raise ConfirmationError(f"BLOCKED: no frozen bank record at {bank_path}")
    checkpoints = verify_checkpoints()

    def tree_digests(names: tuple) -> dict:
        digests = {}
        for name in names:
            path = REPOSITORY_ROOT / name
            if not path.exists():
                raise ConfirmationError(f"BLOCKED: {name} is missing from the tree")
            digests[name] = sha256(path)
        return digests

    manifest = {
        "artifact": "phase18_g1_random_confirmation_launch_v1",
        "run_id": RUN_ID,
        "work_package": WORK_PACKAGE,
        "gate": "G1",
        "agent": "phase_18_agent_3",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authorization": authorization_digests(),
        "source": {
            "g1_confirm_source_commit": git_commit(),
            "g1_confirm_source_tree": git_tree(),
            "g1_source_commit": G1_SOURCE_COMMIT,
            "approved_agent2_commit": APPROVED_AGENT2_COMMIT,
            "execution_worktree": str(REPOSITORY_ROOT),
            "worktree_porcelain_empty": True,
        },
        "checkpoints": checkpoints,
        "contract_sha256": contract_digest,
        "bank_record_sha256": sha256(bank_path),
        "bank_digest": contract["bank"]["digest"],
        "schedule_digest": contract["schedule"]["digest"],
        "seed_function": "stratego.setups.identity.derive_stream_seed",
        "seed_namespace": CONFIRMATION_NAMESPACE,
        "bootstrap": {
            "seed": contract["bootstrap_seed"],
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "rule": contract["primary_hypothesis"]["decision_rule"],
        },
        "design": {
            "paired_units": CONFIRMATION_PAIRS,
            "games_per_arm": CONFIRMATION_PAIRS * 2,
            "total_games": CONFIRMATION_PAIRS * 4,
            "margin": SIGNED_MARGIN,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "power": contract["power"],
            "arm_order": list(ARM_ORDER),
        },
        "evaluator_digests": tree_digests(EVALUATOR_SOURCES),
        "test_digests": tree_digests(TEST_SOURCES),
        "evaluator": PHASE18_EVALUATION_VERSION,
        "environment": environment(),
    }
    write_json(reports_dir / LAUNCH_NAME, manifest)
    log(f"launch manifest bound to {manifest['source']['g1_confirm_source_commit'][:12]}")
    return manifest


#: The artifacts the binding ledger hashes, with the field each binds the
#: source commit through (None: bound by digest alone).
BINDING_ARTIFACTS = (
    (CONTRACT_NAME, None),
    (BANK_NAME, None),
    (LAUNCH_NAME, "source.g1_confirm_source_commit"),
    ("phase18_g1_random_confirmation_reference_v1.json", "g1_confirm_source_commit"),
    ("phase18_g1_random_confirmation_candidate_v1.json", "g1_confirm_source_commit"),
    ("phase18_g1_random_confirmation_noninferiority_v1.json", "g1_confirm_source_commit"),
)


def stage_bind(reports_dir: Path, expected_commit: str) -> dict:
    """One ledger that hashes every confirmation artifact where it lives."""
    artifacts: dict = {}
    mismatched = []
    for name, field in BINDING_ARTIFACTS:
        path = reports_dir / name
        if not path.exists():
            raise ConfirmationError(f"BLOCKED: binding target {path} is missing")
        record: dict = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if field is None:
            record["source_binding"] = {
                "form": "pre-commit artifact; bound by this ledger's digest and the launch manifest",
                "field": None,
                "value": None,
                "agrees": None,
            }
        else:
            payload = json.loads(path.read_text())
            value = payload
            for part in field.split("."):
                value = value[part]
            agrees = value == expected_commit
            record["source_binding"] = {
                "form": "full",
                "field": field,
                "value": value,
                "agrees": agrees,
            }
            if not agrees:
                mismatched.append(name)
        artifacts[name] = record
    ledger = {
        "artifact": "phase18_g1_random_confirmation_binding_v1",
        "run_id": RUN_ID,
        "work_package": WORK_PACKAGE,
        "gate": "G1",
        "agent": "phase_18_agent_3",
        "g1_confirm_source_commit": expected_commit,
        "artifacts": artifacts,
        "mismatched_artifacts": mismatched,
        "all_artifacts_bind_one_source_commit": not mismatched,
        "rule": (
            "every result artifact must repeat one source commit; any mismatch is "
            "fatal. The contract and bank predate that commit by construction and "
            "are bound through the launch manifest instead."
        ),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if mismatched:
        raise ConfirmationError(f"BLOCKED: artifacts bind different commits: {mismatched}")
    write_json(reports_dir / "phase18_g1_random_confirmation_binding_v1.json", ledger)
    return ledger


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--launch-manifest", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--analyse", action="store_true")
    parser.add_argument("--bind", action="store_true")
    parser.add_argument("--output", help="isolated output root for game data")
    parser.add_argument(
        "--reports-dir",
        default=str(REPOSITORY_ROOT / "reports" / "phase18"),
        help="where frozen and result artifacts are written",
    )
    parser.add_argument(
        "--frozen-dir",
        default=str(REPOSITORY_ROOT / "reports" / "phase18"),
        help="where the committed frozen contract is read from",
    )
    parser.add_argument("--expected-commit", help="required by --bind")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-units", type=int, default=64)
    arguments = parser.parse_args()

    reports_dir = Path(arguments.reports_dir).expanduser().resolve()
    frozen_dir = Path(arguments.frozen_dir).expanduser().resolve()
    stages = [
        name
        for name, active in (
            ("freeze", arguments.freeze),
            ("launch-manifest", arguments.launch_manifest),
            ("run", arguments.run),
            ("analyse", arguments.analyse),
            ("bind", arguments.bind),
        )
        if active
    ]
    if not stages:
        parser.error("choose at least one stage")
    if (arguments.run or arguments.analyse) and not arguments.output:
        parser.error("--run and --analyse need --output")
    output = Path(arguments.output).expanduser().resolve() if arguments.output else None

    for stage in stages:
        log(f"stage: {stage}")
        if stage == "freeze":
            stage_freeze(reports_dir)
        elif stage == "launch-manifest":
            stage_launch_manifest(frozen_dir, reports_dir)
        elif stage == "run":
            stage_run(output, frozen_dir, workers=arguments.workers,
                      chunk_units=arguments.chunk_units)
        elif stage == "analyse":
            stage_analyse(output, frozen_dir, reports_dir)
        elif stage == "bind":
            if not arguments.expected_commit:
                parser.error("--bind needs --expected-commit")
            stage_bind(reports_dir, arguments.expected_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
