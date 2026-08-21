"""Phase 14 local read-only monitoring dashboard.

An observational convenience, not a control plane. Nothing here writes to a
Phase 14 run, imports the model, touches MPS, or is required for recovery. If
this package is deleted mid-run, Phase 14 does not notice.

Launch it with `scripts/phase14_dashboard.py`; the recovery path stays
`PHASE_14_RUNBOOK.md`.
"""

from .contract import DASHBOARD_VERSION

__all__ = ["DASHBOARD_VERSION"]
