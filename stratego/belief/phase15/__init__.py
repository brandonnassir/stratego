"""Phase 15 — belief and search engineering, Agent 1.

Specification source: `instructions/phase_15_belief_search_engineering/`.

This package builds an orientation-safe hidden-piece corpus and two belief
specialists — `B18` and `B24` — from the frozen Phase 14 hour-18 and hour-24
policy candidates. It is an **engineering deliverable**, not a
playing-strength claim: no search is implemented here, and none is
evaluated.

Three boundaries hold across every module:

- `P18` and `P24` are opened read-only, copied, and never trained. A belief
  specialist holds deep copies of one trailing encoder block and the encoder
  norm; it owns no policy or value parameter at all.
- The mis-oriented Phase 11B corpus glue
  (`belief/phase11b/corpus.py`, `Phase11BSetupSources`, `corpus_plans`) is
  never imported. Every Phase 15 board leaves
  :func:`~stratego.belief.phase15.orientation.oriented_for`, which
  re-derives the placement from the engine's own `SETUP_SQUARES`.
- No module here can control a Phase 14 task: nothing creates a stop file,
  sends a signal, edits live run state, rotates a checkpoint or invokes a
  closeout command.
"""

from .contract import (
    CORPUS_VERSION,
    PHASE15_CONTRACT_VERSION,
    PHASE15_STATUS_MARKERS,
    SPECIALISTS,
    Phase15Error,
)

__all__ = [
    "CORPUS_VERSION",
    "PHASE15_CONTRACT_VERSION",
    "PHASE15_STATUS_MARKERS",
    "SPECIALISTS",
    "Phase15Error",
]
