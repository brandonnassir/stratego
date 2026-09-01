#!/usr/bin/env python3
"""Phase 17 Agent 5: capture the per-board rows behind each frozen receipt.

The receipts bind lane and stratum scores; they do not carry the 240 individual
game rows. Paired analysis needs them -- every candidate plays the *same* 120
cases against the same opponents, so a per-board difference is a far sharper
instrument than the 0.04 unpaired SE a 120-game lane affords.

This script does not re-score anything. It replays each candidate through the
evaluator's own `_worker_init`/`_play` -- the identical functions the receipts
were produced by, imported, not reimplemented -- and then PROVES the captured
rows are the scored games by recomputing `score_lane` over them and matching
the receipt's `result_digest` exactly. A candidate whose digest does not
reproduce is reported and its rows are discarded.

Nothing under `stratego/evaluation/phase17/` is touched, so
`evaluator_source_digest` is unchanged and the 25 receipts stay comparable.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stratego.evaluation.phase17.candidate import (  # noqa: E402
    build_setup_model,
    materialize_move_checkpoint,
    verify_bundle,
)
from stratego.evaluation.phase17.contract import LANE_JOINT, LANE_MOVE_ONLY  # noqa: E402
from stratego.evaluation.phase17.evaluator import (  # noqa: E402
    _play,
    _worker_init,
    json_digest,
    scoring_projection,
)
from stratego.evaluation.phase17.lanes import generate_joint_setups, score_lane  # noqa: E402
from stratego.evaluation.phase17.pack import load_composite_pack  # noqa: E402

PACK_PATH = ROOT / "data/phase17/phase17_composite_benchmark_v1.json"
PACK_DIGEST = "64450412dd8d03641ed667bc92e2112f7a6f4602047e1ebc8e2c35cc3d6de97f"
RESULTS = ROOT / "reports/phase17/local_eval/results"
ROWS = ROOT / "reports/phase17/local_eval/rows"
LANES = (LANE_MOVE_ONLY, LANE_JOINT)
WORKERS = 8


def capture(bundle_path: Path, receipt: dict) -> dict:
    verified, payload = verify_bundle(
        bundle_path,
        expected_run_id=receipt["run_id"],
        expected_candidate_id=receipt["candidate_id"],
        expected_file_sha256=receipt["bundle_file_sha256"],
    )
    pack = load_composite_pack(PACK_PATH, expected_digest=PACK_DIGEST)
    with tempfile.TemporaryDirectory(prefix="phase17_capture_") as scratch:
        move_checkpoint = materialize_move_checkpoint(payload, scratch, verified=verified)
        setup_model = build_setup_model(payload, verified=verified)
        mark = time.perf_counter()
        generated = generate_joint_setups(
            setup_model,
            pack,
            setup_digest=verified.setup_ema_model_state_digest,
            iteration=verified.iteration,
        )
        setup_seconds = time.perf_counter() - mark
        tasks = [
            (lane, row["board_id"])
            for lane in LANES
            for row in pack["lanes"][lane]["cases"]
        ]
        arguments = (
            str(ROOT), str(PACK_PATH), pack["pack_digest"],
            str(move_checkpoint), verified.candidate_id, generated,
        )
        rows: dict = {lane: [] for lane in LANES}
        with ProcessPoolExecutor(
            max_workers=WORKERS, initializer=_worker_init, initargs=arguments
        ) as pool:
            for row in pool.map(_play, tasks, chunksize=1):
                rows[row["lane"]].append(row)

    lane_results = {}
    for lane in LANES:
        rows[lane] = sorted(rows[lane], key=lambda r: r["board_id"])
        lane_results[lane] = score_lane(
            rows[lane], lane=lane, setup_used=bool(pack["lanes"][lane]["setup_used"])
        )
        lane_results[lane]["setup_generation_seconds"] = (
            round(setup_seconds, 4) if lane == LANE_JOINT else 0.0
        )
    observed = json_digest(scoring_projection(lane_results))
    return {
        "candidate_id": verified.candidate_id,
        "candidate_index": verified.candidate_index,
        "iteration": verified.iteration,
        "reproduced_result_digest": observed,
        "receipt_result_digest": receipt["result_digest"],
        "reproduces": observed == receipt["result_digest"],
        "generated_setups": {
            board: {
                "canonical_fingerprint": value["canonical_fingerprint"],
                "root_seed": value["root_seed"],
                "engine_setup": value["engine_setup"],
            }
            for board, value in generated.items()
        },
        "rows": rows,
    }


def main() -> int:
    ROWS.mkdir(parents=True, exist_ok=True)
    failures = []
    for path in sorted(RESULTS.glob("*.result.json")):
        receipt = json.loads(path.read_text())
        bundle = ROOT / (
            f"checkpoints/phase17/RUN-2026-B/exports/{receipt['candidate_id']}.pt"
        )
        captured = capture(bundle, receipt)
        status = "OK " if captured["reproduces"] else "FAIL"
        print(
            f"{status} {captured['candidate_id']} it={captured['iteration']:>3} "
            f"digest={captured['reproduced_result_digest'][:16]}",
            flush=True,
        )
        if not captured["reproduces"]:
            failures.append(captured["candidate_id"])
            continue
        (ROWS / f"{captured['candidate_id']}.rows.json").write_text(
            json.dumps(captured, indent=1, sort_keys=True) + "\n"
        )
    print(f"\nfailures: {failures or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
