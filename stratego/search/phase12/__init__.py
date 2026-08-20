"""Phase 12: the minimal belief-guided search core.

Public surface:

- :mod:`.contract` — the search version, presets and configuration;
- :mod:`.providers` — the four interchangeable belief providers;
- :mod:`.engine` — the root-world search engine.
"""

from .contract import (
    ALL_PROVIDERS,
    PRESET_MEDIUM,
    PRESET_SMALL,
    PRESET_TINY,
    PRODUCTION_PROVIDERS,
    PROVIDER_AGENT1C,
    PROVIDER_ORACLE,
    PROVIDER_ORIGINAL_PHASE11,
    PROVIDER_REMAINING_COUNT,
    SEARCH_PRESETS,
    SEARCH_VERSION,
    Phase12SearchConfig,
    Phase12SearchError,
    search_preset,
)
from .engine import (
    Phase12CandidateResult,
    Phase12SearchDecision,
    Phase12SearchEngine,
    materialize_world,
)
from .providers import (
    AdapterNeuralBeliefProvider,
    OracleBeliefProvider,
    Phase12BeliefProvider,
    RemainingCountBeliefProvider,
    build_belief_provider,
)

__all__ = [
    "ALL_PROVIDERS",
    "AdapterNeuralBeliefProvider",
    "OracleBeliefProvider",
    "PRESET_MEDIUM",
    "PRESET_SMALL",
    "PRESET_TINY",
    "PRODUCTION_PROVIDERS",
    "PROVIDER_AGENT1C",
    "PROVIDER_ORACLE",
    "PROVIDER_ORIGINAL_PHASE11",
    "PROVIDER_REMAINING_COUNT",
    "Phase12BeliefProvider",
    "Phase12CandidateResult",
    "Phase12SearchConfig",
    "Phase12SearchDecision",
    "Phase12SearchEngine",
    "Phase12SearchError",
    "RemainingCountBeliefProvider",
    "SEARCH_PRESETS",
    "SEARCH_VERSION",
    "build_belief_provider",
    "materialize_world",
    "search_preset",
]
