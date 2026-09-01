"""Phase 17 Agent 5: the portable evaluator -- one bundle in, one receipt out.

Specification sources: Agent 5 instruction sections 3 and 4, common contract
section 11.

Order of operations, which is the safety property
--------------------------------------------------
```text
1  verify the bundle's bytes, digests, versions and parameter counts
2  verify the composite pack re-digests to the digest we were told to expect
3  verify every opponent file's sha256
4  build models
5  play
6  write the result atomically and return a receipt binding all of the above
```
Steps 1-3 happen before step 4. A candidate that fails any of them produces a
refusal receipt rather than a number, because contract section 13 makes
"evaluation result bound to the wrong candidate or benchmark" a stop condition
and a stop condition that reports an EWR is worse than useless.

Idempotent for a candidate identity
------------------------------------
`evaluate_candidate` writes `<candidate_id>.result.json` through a `.partial`
name. If a result for that candidate identity already exists and binds the same
bundle digest, the existing result is returned untouched -- a retried transfer
re-reads a result rather than re-playing 240 games. If a result exists and
binds a *different* bundle digest, that is a duplicate-conflicting candidate and
is refused, never overwritten.

Cross-machine determinism is a design choice, not an accident
--------------------------------------------------------------
CPU, float32, one torch thread per worker, greedy decisions, and setup
generation batched in a fixed order in the parent. Every one of those exists so
that the same bundle and the same pack produce the same games on two different
Macs. Section 5's fixture measures whether that held; it is not assumed here.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .candidate import (
    build_move_owner,
    build_setup_model,
    materialize_move_checkpoint,
    verify_bundle,
)
from .contract import (
    COMPOSITE_PACK_ID,
    EVALUATOR_VERSION,
    LANE_JOINT,
    LANE_MOVE_ONLY,
    RECEIPT_SCHEMA_VERSION,
    WORKER_TORCH_THREADS,
    Phase17EvaluationError,
    file_sha256,
    json_digest,
)
from .lanes import (
    Phase17CandidateSeat,
    generate_joint_setups,
    joint_plan,
    lane_row,
    score_lane,
)
from .opponents import verify_opponent_files
from .pack import DEFAULT_PACK_PATH, load_composite_pack, plan_from_row

_STATE: dict = {}

#: Keys inside a lane result that are wall-clock, not science. They are
#: reported but EXCLUDED from `result_digest`: a digest that moved because one
#: run generated setups 5 ms faster than another could never be compared across
#: two machines, and comparing results across machines is the entire point.
TIMING_KEYS = ("setup_generation_seconds",)


def scoring_projection(lane_results: dict) -> dict:
    """`lane_results` with timing removed -- the part a digest may bind."""
    return {
        lane: {key: value for key, value in result.items() if key not in TIMING_KEYS}
        for lane, result in lane_results.items()
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def evaluator_source_digest(*, root: "Path | str" = ".") -> str:
    """sha256 over the sorted (path, sha256) pairs of this package's sources."""
    package = Path(root) / "stratego" / "evaluation" / "phase17"
    pairs = sorted(
        (path.name, file_sha256(path)) for path in sorted(package.glob("*.py"))
    )
    return json_digest(pairs)


def host_identity() -> dict:
    import torch

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cpu_count": os.cpu_count(),
        "device": "cpu",
        "torch_threads_per_worker": WORKER_TORCH_THREADS,
    }


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


def _worker_init(root: str, pack_path: str, pack_digest: str, move_checkpoint: str,
                 candidate_id: str, generated: dict) -> None:
    import torch

    from .opponents import build_opponent_owners

    torch.set_num_threads(WORKER_TORCH_THREADS)
    pack = load_composite_pack(pack_path, expected_digest=pack_digest)
    _STATE["root"] = root
    _STATE["pack"] = pack
    _STATE["rows"] = {
        lane: {row["board_id"]: row for row in pack["lanes"][lane]["cases"]}
        for lane in (LANE_MOVE_ONLY, LANE_JOINT)
    }
    _STATE["owners"] = build_opponent_owners(root=root)
    _STATE["seat"] = Phase17CandidateSeat(
        build_move_owner(move_checkpoint, name=f"phase17_candidate_{candidate_id}"),
        candidate_id,
    )
    _STATE["generated"] = generated


def _play(task) -> dict:
    from ...search.phase15.matchplay import play_board

    lane, board_id = task
    row = _STATE["rows"][lane][board_id]
    plan = (
        plan_from_row(row)
        if lane == LANE_MOVE_ONLY
        else joint_plan(row, _STATE["generated"])
    )
    started = time.perf_counter()
    record = play_board(
        plan, _STATE["seat"], _STATE["owners"], probe=None, preset_id=lane
    )
    out = lane_row(record, lane=lane, board_id=board_id)
    out["wall_seconds"] = round(time.perf_counter() - started, 4)
    return out


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


def existing_result(results_directory: "Path | str", candidate_id: str) -> "dict | None":
    path = Path(results_directory) / f"{candidate_id}.result.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _write_atomic(path: "Path | str", payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".partial"
    )
    with os.fdopen(handle, "w") as sink:
        sink.write(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, target)
    return target


def evaluate_candidate(
    bundle_path: "Path | str",
    *,
    root: "Path | str" = ".",
    pack_path: "Path | str" = DEFAULT_PACK_PATH,
    expected_pack_digest: "str | None" = None,
    results_directory: "Path | str" = "results",
    workers: int = 4,
    expected_file_sha256: "str | None" = None,
    expected_run_id: "str | None" = None,
    expected_candidate_id: "str | None" = None,
    lanes: "tuple[str, ...]" = (LANE_MOVE_ONLY, LANE_JOINT),
    payload_digest: "str | None" = None,
) -> dict:
    """Verify, evaluate and receipt one candidate. Idempotent per identity."""
    started_utc = utc_now()
    started = time.perf_counter()
    root = Path(root)

    verified, payload = verify_bundle(
        bundle_path,
        expected_file_sha256=expected_file_sha256,
        expected_run_id=expected_run_id,
        expected_candidate_id=expected_candidate_id,
    )

    previous = existing_result(results_directory, verified.candidate_id)
    if previous is not None:
        if previous.get("bundle_digest") == verified.manifest_digest:
            previous["reused_existing_result"] = True
            return previous
        raise Phase17EvaluationError(
            f"{verified.candidate_id}: a result already exists binding bundle "
            f"digest {previous.get('bundle_digest')}, this bundle digests to "
            f"{verified.manifest_digest}; duplicate-conflicting candidate refused"
        )

    pack = load_composite_pack(pack_path, expected_digest=expected_pack_digest)
    if pack["pack_id"] != COMPOSITE_PACK_ID:
        raise Phase17EvaluationError(f"pack {pack['pack_id']!r} is not the composite pack")
    opponents = verify_opponent_files(root=root)

    lane_results: dict = {}
    with tempfile.TemporaryDirectory(prefix="phase17_eval_") as scratch:
        move_checkpoint = materialize_move_checkpoint(payload, scratch, verified=verified)

        generated: dict = {}
        setup_generation_seconds = 0.0
        if LANE_JOINT in lanes:
            setup_model = build_setup_model(payload, verified=verified)
            mark = time.perf_counter()
            generated = generate_joint_setups(
                setup_model,
                pack,
                setup_digest=verified.setup_ema_model_state_digest,
                iteration=verified.iteration,
            )
            setup_generation_seconds = time.perf_counter() - mark

        tasks = [
            (lane, row["board_id"])
            for lane in lanes
            for row in pack["lanes"][lane]["cases"]
        ]
        rows: dict = {lane: [] for lane in lanes}
        arguments = (
            str(root),
            str(pack_path),
            pack["pack_digest"],
            str(move_checkpoint),
            verified.candidate_id,
            generated,
        )
        if workers <= 1:
            _worker_init(*arguments)
            for task in tasks:
                rows[task[0]].append(_play(task))
        else:
            with ProcessPoolExecutor(
                max_workers=int(workers), initializer=_worker_init, initargs=arguments
            ) as pool:
                for result in pool.map(_play, tasks, chunksize=1):
                    rows[result["lane"]].append(result)

        for lane in lanes:
            lane_results[lane] = score_lane(
                sorted(rows[lane], key=lambda row: row["board_id"]),
                lane=lane,
                setup_used=bool(pack["lanes"][lane]["setup_used"]),
            )
            lane_results[lane]["setup_generation_seconds"] = (
                round(setup_generation_seconds, 4) if lane == LANE_JOINT else 0.0
            )

    finished_utc = utc_now()
    runtime = time.perf_counter() - started

    result_digest = json_digest(scoring_projection(lane_results))
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "work_package": "phase17",
        "candidate_id": verified.candidate_id,
        "bundle_id": verified.candidate_id,
        "bundle_id_note": (
            "Agent 1's schema sketched '<run_id>|h<HHMM>'; Agent 4's FROZEN export "
            "names candidates '<run_id>-cand-NNN'. The frozen export governs, and "
            "bundle_id echoes candidate_id so no reader has to guess."
        ),
        "candidate_index": verified.candidate_index,
        "run_id": verified.run_id,
        "iteration": verified.iteration,
        "elapsed_active_training_seconds": verified.elapsed_active_training_seconds,
        "bundle_digest": verified.manifest_digest,
        "bundle_file_sha256": verified.file_sha256,
        "move_ema_model_state_digest": verified.move_ema_model_state_digest,
        "setup_ema_model_state_digest": verified.setup_ema_model_state_digest,
        "config_digest": verified.config_digest,
        "source_digest": verified.source_digest,
        "benchmark_pack_id": pack["pack_id"],
        "benchmark_pack_digest": pack["pack_digest"],
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_source_digest": evaluator_source_digest(root=root),
        "payload_digest": payload_digest,
        "opponents": opponents,
        "host_identity": host_identity(),
        "workers": int(workers),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "runtime_seconds": round(runtime, 3),
        "lanes_run": list(lanes),
        "lane_results": lane_results,
        "result_digest": result_digest,
        "result_digest_covers": (
            "lane_results with wall-clock fields removed; two machines that "
            "played the same games reproduce this digest exactly"
        ),
        "status": "ok",
    }
    receipt["receipt_digest"] = json_digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    _write_atomic(
        Path(results_directory) / f"{verified.candidate_id}.result.json", receipt
    )
    return receipt


def refusal_receipt(
    *, candidate_id: str, reason: str, detail: str, root: "Path | str" = "."
) -> dict:
    """A failure is a receipt too. An absent result is never read as a pass."""
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "work_package": "phase17",
        "candidate_id": candidate_id,
        "bundle_id": candidate_id,
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_source_digest": evaluator_source_digest(root=root),
        "host_identity": host_identity(),
        "finished_utc": utc_now(),
        "status": "refused",
        "refusal_reason": reason,
        "refusal_detail": detail,
        "lane_results": {},
    }
    receipt["receipt_digest"] = json_digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    return receipt


def verify_receipt(receipt: dict, *, expected: "dict | None" = None) -> dict:
    """Re-verify a returned receipt on the training computer.

    Contract section 11: no result is eligible until its receipt has been
    re-verified at the source. Every identity in the receipt must equal the
    identity in the bundle; one mismatch is a stop condition.
    """
    claimed = receipt.get("receipt_digest")
    observed = json_digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    findings = {"receipt_digest_ok": claimed == observed, "mismatches": []}
    if receipt.get("status") == "ok":
        result = json_digest(scoring_projection(receipt.get("lane_results") or {}))
        findings["result_digest_ok"] = result == receipt.get("result_digest")
        if not findings["result_digest_ok"]:
            findings["mismatches"].append("result_digest")
    if not findings["receipt_digest_ok"]:
        findings["mismatches"].append("receipt_digest")
    for field, value in (expected or {}).items():
        if receipt.get(field) != value:
            findings["mismatches"].append(
                f"{field}: receipt {receipt.get(field)!r} != expected {value!r}"
            )
    findings["eligible"] = not findings["mismatches"]
    return findings


__all__ = [
    "TIMING_KEYS",
    "evaluate_candidate",
    "evaluator_source_digest",
    "existing_result",
    "host_identity",
    "refusal_receipt",
    "scoring_projection",
    "utc_now",
    "verify_receipt",
]
