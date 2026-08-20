"""Phase 11B — the belief engineering sprint.

Specification source: `instructions/phase_11b_belief_engineering_sprint/`.

Phase 11B is an **engineering prototype branch**. It is not a retroactive
override of Phase 11, not a scientific repair phase and not authorization
for Phase 12. Every artifact this package writes carries the four Phase 11B
status markers from :mod:`stratego.belief.phase11b.contract`, and nothing
here reads, writes or reinterprets an accepted Phase 9/10/11 artifact:

- the accepted Phase 9 checkpoint is opened read-only and copied;
- the spent `phase11_test_bank_v1` is never touched;
- the accepted Phase 11 sampler is imported, never modified.
"""

from .contract import (
    PHASE11B_STATUS_MARKERS,
    PHASE11B_VERSION,
    Phase11BError,
)

__all__ = ["PHASE11B_STATUS_MARKERS", "PHASE11B_VERSION", "Phase11BError"]
