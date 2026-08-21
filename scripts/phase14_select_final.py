#!/usr/bin/env python3
"""Phase 14: apply the frozen post-run checkpoint-selection rule.

Run **after** training is closed, and only then. It refuses to select while any
marked candidate is still missing a complete 128-game result on the frozen
pack, because a candidate scored on 40 games is not comparable with one scored
on 128 — and because "hour 168 wins by default" is exactly the outcome the
frozen rule exists to prevent.

The rule, unchanged: highest equal-weight mean EWR across the four strata;
tie-break on highest minimum stratum EWR; then the later candidate hour.

Usage:

```text
python scripts/phase14_select_final.py            # refuses if anything is unevaluated
python scripts/phase14_select_final.py --check    # report readiness only
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


def main(argv=None) -> int:
    from stratego.evaluation.phase14_candidates import (
        CandidateLedger,
        select_final_candidate,
    )
    from stratego.training.phase14_storage import Phase14Storage
    from stratego.training.phase14_supervisor import (
        Phase14SupervisorError,
        assert_all_candidates_evaluated,
        run_manifest_state,
        unevaluated_candidates,
    )

    parser = argparse.ArgumentParser(description="Phase 14 final checkpoint selection")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--hot-root", default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    storage = (
        Phase14Storage.production()
        if args.external_root is None and args.hot_root is None
        else Phase14Storage.under(args.external_root, hot_root=args.hot_root)
    )
    manifest = run_manifest_state(storage.run_state_path)
    ledger = CandidateLedger.at(storage.evaluation_root)
    readiness = {
        "training_closed": manifest.get("closed", False),
        "close_reason": manifest.get("close_reason", ""),
        "ledger": ledger.status_summary(),
        "unevaluated_hours": unevaluated_candidates(storage.evaluation_root),
    }
    if args.check:
        print(json.dumps(readiness, indent=2, sort_keys=True, default=str))
        return 0
    if not readiness["training_closed"]:
        print(
            json.dumps(
                {
                    "selected": False,
                    "reason": "training is not closed; selection happens after the run",
                    **readiness,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return 2
    try:
        assert_all_candidates_evaluated(storage.evaluation_root)
    except Phase14SupervisorError as error:
        print(
            json.dumps(
                {"selected": False, "reason": str(error), **readiness},
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return 3
    selection = select_final_candidate(ledger.completed())
    print(json.dumps({"selected": True, **selection, **readiness}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
