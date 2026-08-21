#!/usr/bin/env python3
"""Phase 14: the local read-only monitoring dashboard.

An observational convenience. It reads what the run already writes — telemetry
rows, supervisor events, rollout-store manifests, checkpoint metadata, disk
state — and changes nothing. Training does not depend on it: launch it, kill
it, or never open it, and the run is identical.

Recovery and control stay in `PHASE_14_RUNBOOK.md` and the accepted supervisor.
This page has no button that does anything.

```text
python scripts/phase14_dashboard.py                 # http://127.0.0.1:8714
python scripts/phase14_dashboard.py --port 9000
python scripts/phase14_dashboard.py --once          # one JSON document, no server
python scripts/phase14_dashboard.py --external-root <dir> --hot-root <dir>
```

The process imports no part of `stratego` and therefore no part of torch: it is
a stdlib HTTP server reading JSON. That is deliberate — see
`monitoring/phase14_dashboard/contract.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from monitoring.phase14_dashboard.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
