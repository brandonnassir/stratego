"""Phase 15 Agent 2: P18/P24 belief-guided search integration.

The accepted Phase 12 search engine, run over the two frozen Phase 14 move
models and the two Phase 15 belief specialists. Nothing in this package
changes what `phase12_root_world_search_v1` does; it changes which models it
does it with, adds the fresh orientation-safe evidence packs Phase 15
requires, and packages the selected complete system as a working player.
"""

from .contract import (
    ALL_PAIRINGS,
    COMBINED_PAIRING_IDS,
    INTEGRATION_VERSION,
    ORACLE_AVAILABLE_IN_PRODUCTION,
    PRODUCTION_PROVIDERS,
    Pairing,
    Phase15SearchError,
    pairing,
)

__all__ = [
    "ALL_PAIRINGS",
    "COMBINED_PAIRING_IDS",
    "INTEGRATION_VERSION",
    "ORACLE_AVAILABLE_IN_PRODUCTION",
    "PRODUCTION_PROVIDERS",
    "Pairing",
    "Phase15SearchError",
    "pairing",
]
