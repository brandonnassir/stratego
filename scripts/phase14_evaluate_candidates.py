#!/usr/bin/env python3
"""Phase 14: the out-of-band candidate evaluator.

Every six-hour candidate must eventually receive the *same* frozen 128-game
direct-policy evaluation on pack `896a753b…`. This script is how that happens
without depending on anyone remembering: the supervisor launches it whenever
the ledger holds a pending mark, and it may also be run by hand.

It is deliberately a separate process. It imports no trainer, no scheduler and
no clock; it writes to the candidate ledger and nothing else; and no result it
produces can stop training, change a learning rate, change the opponent
mixture, change the setup source, change the historical pool or extend the
deadline — the accepted control surface refuses every one of those keys by
name.

**No search.** Candidates are evaluated as direct policies only.

A failed evaluation preserves the candidate, records the reason, and stays
re-runnable on the identical pack. Failing is not a training event.

Usage:

```text
python scripts/phase14_evaluate_candidates.py            # every pending candidate
python scripts/phase14_evaluate_candidates.py --hours 6 12
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def evaluate(storage, *, hours=None, device: str = "mps", limit=None) -> dict:
    """Evaluate pending candidates through the accepted evaluator."""
    from stratego.evaluation.phase14_candidates import (
        CandidateLedger,
        evaluate_candidate,
    )
    from stratego.training.phase14_checkpoint import export_evaluation_weights
    from stratego.training.phase14_contract import STARTING_CHECKPOINT, repository_root

    ledger = CandidateLedger.at(storage.evaluation_root)
    weights_root = Path(storage.evaluation_root) / "weights"
    weights_root.mkdir(parents=True, exist_ok=True)

    def weights_for(tag, source) -> Path:
        export = weights_root / f"{tag}.pt"
        if not export.exists():
            export_evaluation_weights(source, export)
        return export

    anchor = weights_for("anchor", repository_root() / STARTING_CHECKPOINT)
    wanted = None if not hours else {int(hour) for hour in hours}
    results = []
    for entry in ledger.pending():
        hour = int(entry["hour"])
        if wanted is not None and hour not in wanted:
            continue
        mark = entry.get("mark") or {}
        snapshot = mark.get("snapshot_path")
        if not snapshot or not Path(snapshot).exists():
            ledger.record_failure(hour, f"candidate bytes missing: {snapshot}")
            results.append({"hour": hour, "status": "failed", "error": "bytes missing"})
            continue
        try:
            result = evaluate_candidate(
                weights_for(f"hour_{hour:03d}", snapshot),
                anchor_weights=anchor,
                device=device,
                limit=limit,
            )
            ledger.record_result(hour, result)
            results.append(
                {
                    "hour": hour,
                    "status": "complete",
                    "mean_ewr": result["mean_ewr"],
                    "games": result["games_played"],
                    "complete": result["complete"],
                }
            )
        except Exception as error:  # noqa: BLE001 - evaluation never stops training
            ledger.record_failure(hour, f"{type(error).__name__}: {error}")
            results.append({"hour": hour, "status": "failed", "error": str(error)[:400]})
    return {
        "artifact": "phase14_candidate_evaluation_pass_v1",
        "results": results,
        "ledger": ledger.status_summary(),
        "search_used": False,
        "influences_training": False,
    }


def main(argv=None) -> int:
    from stratego.training.phase14_storage import Phase14Storage

    parser = argparse.ArgumentParser(description="Phase 14 out-of-band candidate evaluation")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--hot-root", default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--hours", type=int, nargs="*", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="games per candidate; omit for the full frozen 128-game pack",
    )
    args = parser.parse_args(argv)
    storage = (
        Phase14Storage.production()
        if args.external_root is None and args.hot_root is None
        else Phase14Storage.under(args.external_root, hot_root=args.hot_root)
    )
    report = evaluate(storage, hours=args.hours, device=args.device, limit=args.limit)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
