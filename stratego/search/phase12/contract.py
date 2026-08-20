"""Phase 12 search: versions, provider names, presets and configuration.

Specification sources:

- `00_PHASE_12_SEQUENCE_AND_COMMON_CONTRACT.md` (sections 5-9)
- `02_PHASE_12_AGENT_1_SEARCH_CORE.md` (sections 3, 6, 7, 8)

Phase 12 is a rapid engineering phase. This module freezes only what the
other Phase 12 modules must agree on: the search version string, the four
belief-provider names, the score definition constants, and the three
instructed presets. It deliberately does not grow an acceptance apparatus.

The one seed derivation defined here exists for the `remaining_count`
provider, whose accepted sampler is keyed by `(sampler_version,
public_state_identity, sample_ordinal)` and takes no caller seed. The
neural providers reach the accepted sampler through the accepted Phase 11B
adapter, which already maps a caller seed onto a starting ordinal; this
derivation mirrors that pattern under a Phase 12 personalization so no
Phase 11/11B stream is touched or reused.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

#: The search algorithm identity. Root-sampled worlds, fixed candidate set,
#: greedy Phase 9 rollouts, policy-regularized scores. Any change to the
#: algorithm's decisions is a new version, never a silent edit.
SEARCH_VERSION = "phase12_root_world_search_v1"

#: The frozen score definition, stated once for reports:
#: `S(a) = Q(a) + beta * log(pi(a) + epsilon)` with `Q(a)` the world-average
#: root value and `pi` the Phase 9 policy softmax over the legal actions.
SCORE_DEFINITION = "S(a) = Q(a) + beta * log(pi(a) + epsilon)"

#: One fixed modest policy-regularization weight (contract section 8: no
#: grid search in the first agents). Chosen, not tuned: Q lives in [-1, 1]
#: and the top-8 candidates' log-priors typically span a few nats, so 0.1
#: makes the prior matter without letting it dominate a clear Q signal.
BETA_DEFAULT = 0.1
EPSILON_DEFAULT = 1e-6

#: The four belief-provider names of contract section 5.
PROVIDER_REMAINING_COUNT = "remaining_count"
PROVIDER_ORIGINAL_PHASE11 = "original_phase11"
PROVIDER_AGENT1C = "agent1c"
PROVIDER_ORACLE = "oracle"

#: What a production configuration may use. The oracle is diagnostic-only
#: and is structurally rejected by the provider factory and by the engine
#: whenever `production=True`.
PRODUCTION_PROVIDERS = (
    PROVIDER_REMAINING_COUNT,
    PROVIDER_ORIGINAL_PHASE11,
    PROVIDER_AGENT1C,
)
ALL_PROVIDERS = PRODUCTION_PROVIDERS + (PROVIDER_ORACLE,)


class Phase12SearchError(RuntimeError):
    """A Phase 12 search request was refused or an invariant was violated."""


# ---------------------------------------------------------------------------
# Phase 12 seed streams
# ---------------------------------------------------------------------------

#: blake2b personalization of every Phase 12 stream. Distinct from Phase
#: 11's ``strat-b11`` and Phase 11B's ``strat-b11b``, so a Phase 12 ordinal
#: walk can never coincide with an accepted phase's stream.
PHASE12_SEED_PERSON = b"strat-p12"

#: Domain of the count-uniform world-sampling ordinal walk.
DOMAIN_COUNT_WORLDS = "count_worlds"


def derive_phase12_seed(domain: str, label: str, value: int) -> int:
    """A 64-bit Phase 12 stream value for `(domain, label, value)`."""
    payload = f"{domain}:{label}:{int(value)}".encode()
    digest = hashlib.blake2b(
        payload, digest_size=8, person=PHASE12_SEED_PERSON
    ).digest()
    return int.from_bytes(digest, "big")


# ---------------------------------------------------------------------------
# Configuration and presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase12SearchConfig:
    """One complete search configuration.

    `rollout_depth` counts rollout plies *after* the candidate root action;
    a simulation therefore looks `1 + rollout_depth` plies ahead before the
    Phase 9 leaf value is read, unless an exact terminal ends it earlier.

    `production=True` (the default) structurally refuses the oracle: the
    engine raises at construction when handed a provider that reads hidden
    truth, and the provider factory refuses to build one.
    """

    preset_id: str
    worlds: int
    rollout_depth: int
    max_root_candidates: int = 8
    beta: float = BETA_DEFAULT
    epsilon: float = EPSILON_DEFAULT
    production: bool = True
    #: Re-derive the root observation and legal actions inside every
    #: materialized world and require them byte-identical to the real
    #: root's. This is the accepted anti-leak permutation gate, applied at
    #: run time; it is on by default while the implementation is young.
    verify_world_public_surface: bool = True
    #: Evaluate duplicate sampled worlds once and weight by multiplicity.
    #: Exact under this engine: rollouts are deterministic greedy, so two
    #: identical worlds produce identical values by construction.
    deduplicate_worlds: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.worlds, int) or self.worlds < 1:
            raise Phase12SearchError(f"worlds must be a positive int, got {self.worlds!r}")
        if not isinstance(self.rollout_depth, int) or self.rollout_depth < 0:
            raise Phase12SearchError(
                f"rollout_depth must be a non-negative int, got {self.rollout_depth!r}"
            )
        if not isinstance(self.max_root_candidates, int) or self.max_root_candidates < 1:
            raise Phase12SearchError(
                f"max_root_candidates must be a positive int, got {self.max_root_candidates!r}"
            )
        if not (self.beta >= 0.0):
            raise Phase12SearchError(f"beta must be non-negative, got {self.beta!r}")
        if not (self.epsilon > 0.0):
            raise Phase12SearchError(f"epsilon must be positive, got {self.epsilon!r}")

    def describe(self) -> dict:
        return {
            "search_version": SEARCH_VERSION,
            "score_definition": SCORE_DEFINITION,
            "preset_id": self.preset_id,
            "worlds": self.worlds,
            "rollout_depth": self.rollout_depth,
            "max_root_candidates": self.max_root_candidates,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "production": self.production,
            "verify_world_public_surface": self.verify_world_public_surface,
            "deduplicate_worlds": self.deduplicate_worlds,
        }


#: The three instructed presets (contract section 9). Rollouts per
#: action/world are fixed at 1 by the algorithm: rollouts are deterministic
#: greedy, so a second rollout of the same (action, world) is a repeat.
PRESET_TINY = Phase12SearchConfig("TINY", worlds=8, rollout_depth=4)
PRESET_SMALL = Phase12SearchConfig("SMALL", worlds=16, rollout_depth=6)
PRESET_MEDIUM = Phase12SearchConfig("MEDIUM", worlds=32, rollout_depth=8)

SEARCH_PRESETS = {
    config.preset_id: config for config in (PRESET_TINY, PRESET_SMALL, PRESET_MEDIUM)
}


def search_preset(name: str, **overrides) -> Phase12SearchConfig:
    """One of the instructed presets, optionally with non-budget overrides.

    Overriding `production` or the verification flags is normal; overriding
    a budget field produces a config whose `preset_id` no longer names an
    instructed preset, so the id is rewritten to say so.
    """
    if name not in SEARCH_PRESETS:
        known = ", ".join(sorted(SEARCH_PRESETS))
        raise Phase12SearchError(f"unknown search preset {name!r}; known presets: {known}")
    config = replace(SEARCH_PRESETS[name], **overrides)
    budget_fields = ("worlds", "rollout_depth", "max_root_candidates", "beta", "epsilon")
    if any(field in overrides for field in budget_fields):
        config = replace(config, preset_id=f"{name}_modified")
    return config


__all__ = [
    "ALL_PROVIDERS",
    "BETA_DEFAULT",
    "DOMAIN_COUNT_WORLDS",
    "EPSILON_DEFAULT",
    "PHASE12_SEED_PERSON",
    "PRESET_MEDIUM",
    "PRESET_SMALL",
    "PRESET_TINY",
    "PRODUCTION_PROVIDERS",
    "PROVIDER_AGENT1C",
    "PROVIDER_ORACLE",
    "PROVIDER_ORIGINAL_PHASE11",
    "PROVIDER_REMAINING_COUNT",
    "SCORE_DEFINITION",
    "SEARCH_PRESETS",
    "SEARCH_VERSION",
    "Phase12SearchConfig",
    "Phase12SearchError",
    "derive_phase12_seed",
    "search_preset",
]
