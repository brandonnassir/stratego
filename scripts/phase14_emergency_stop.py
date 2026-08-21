#!/usr/bin/env python3
"""Phase 14: the emergency stop.

Writes (or clears) the durable stop file under the run's external directory.
The learner reads it at its next safe boundary and stops cleanly — finishing
the collection unit or optimizer step in flight and writing a hot checkpoint,
because a torn iteration is a thing the store then has to reconcile. The
supervisor reads the same file and does not relaunch over it.

What it does **not** do: change any frozen training value, extend or shorten
the deadline, or kill anything. Stopping does not stop the clock — downtime
counts against the 168 hours, so a stopped run is a run losing deadline.

Usage:

```text
python scripts/phase14_emergency_stop.py --reason "thermal event"
python scripts/phase14_emergency_stop.py --status
python scripts/phase14_emergency_stop.py --clear
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
    from stratego.training.phase14_launch import (
        clear_emergency_stop,
        emergency_stop_state,
        request_emergency_stop,
    )
    from stratego.training.phase14_storage import Phase14Storage

    parser = argparse.ArgumentParser(description="Phase 14 emergency stop")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--hot-root", default=None)
    parser.add_argument("--reason", default="operator request")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    storage = (
        Phase14Storage.production()
        if args.external_root is None and args.hot_root is None
        else Phase14Storage.under(args.external_root, hot_root=args.hot_root)
    )
    if args.status:
        record = emergency_stop_state(storage.external_root)
    elif args.clear:
        record = clear_emergency_stop(storage.external_root)
    else:
        record = request_emergency_stop(storage.external_root, reason=args.reason)
        print(
            "emergency stop requested; the learner stops at its next safe boundary "
            "and the supervisor will not relaunch it.\n"
            "The deadline is untouched: downtime counts against the 168 hours."
        )
    print(json.dumps(record, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
