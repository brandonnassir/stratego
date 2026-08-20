"""Phase 12: the minimal belief-guided search core and the working player.

Public surface:

- :mod:`.contract` — the search version, presets and configuration;
- :mod:`.providers` — the four interchangeable belief providers;
- :mod:`.engine` — the root-world search engine;
- :mod:`.player` — the production working player (Agent 5).
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
    Phase12SearchTimeout,
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
from .player import (
    CANDIDATE_ARTIFACT,
    DEFAULT_MODE,
    MODE_TIME_CAP_SECONDS,
    ORACLE_AVAILABLE_IN_PRODUCTION,
    PLAYER_MODES,
    PLAYER_VERSION,
    Phase12PlayerDecision,
    Phase12PlayerSeat,
    Phase12SearchPlayer,
    build_candidate_record,
    load_search_player,
)

__all__ = [
    "CANDIDATE_ARTIFACT",
    "DEFAULT_MODE",
    "MODE_TIME_CAP_SECONDS",
    "ORACLE_AVAILABLE_IN_PRODUCTION",
    "PLAYER_MODES",
    "PLAYER_VERSION",
    "Phase12PlayerDecision",
    "Phase12PlayerSeat",
    "Phase12SearchPlayer",
    "Phase12SearchTimeout",
    "build_candidate_record",
    "load_search_player",
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
