"""Phase 15 Agent 2: the frozen identities of belief-guided search.

Specification source:
`instructions/phase_15_belief_search_engineering/02_AGENT_2_SEARCH_IMPLEMENTATION.md`

What this module is, and what it deliberately is not
----------------------------------------------------
Section 5 is explicit: *reuse* `phase12_root_world_search_v1` and retain its
mechanics. So this module does not restate the algorithm, the score
definition, `beta`, the candidate rule or the three preset budgets — every
one of those is imported from :mod:`stratego.search.phase12.contract` and
re-exported under a Phase 15 name. What is genuinely new in Phase 15 is the
*integration*: which move model, which belief specialist, which pairings
exist, and which of them a production constructor may build.

```text
integration = (move model P18|P24) x (belief provider remaining_count|b18|b24)
              x (one Phase 12 preset)
```

The oracle, four times refused
------------------------------
`oracle` is a diagnostic provider name and never a deployable arm. It is
refused by :func:`check_production_provider` here, by
:func:`stratego.search.phase15.providers.build_phase15_provider` under a
production configuration, by the accepted Phase 12 engine whenever
`config.production` is true, and by the working player, whose mode table has
no entry that could reach it. `PRODUCTION_PROVIDERS` is the single list the
tests and the frozen candidate both read.

The seed streams
----------------
A new blake2b personalization, ``strat-p15s``, with new domain roots. No
Phase 12, Phase 14 or Phase 15-Agent-1 stream can coincide with a Phase 15
search stream even where two identity strings happen to spell the same text.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..phase12.contract import (
    BETA_DEFAULT,
    EPSILON_DEFAULT,
    PRESET_MEDIUM,
    PRESET_SMALL,
    PRESET_TINY,
    SCORE_DEFINITION,
    SEARCH_PRESETS as PHASE12_PRESETS,
    SEARCH_VERSION,
    Phase12SearchConfig,
    Phase12SearchError,
    Phase12SearchTimeout,
)

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

#: The Phase 15 integration identity. The *search algorithm* identity stays
#: :data:`SEARCH_VERSION` — `phase12_root_world_search_v1` — because Phase 15
#: changes which models the algorithm runs on, not what the algorithm does.
INTEGRATION_VERSION = "phase15_belief_search_integration_v1"

#: Re-exported so a Phase 15 reader never has to know these came from the
#: Phase 12 namespace, and so a Phase 15 report can quote them verbatim.
PHASE15_SEARCH_VERSION = SEARCH_VERSION
PHASE15_SCORE_DEFINITION = SCORE_DEFINITION

#: The Agent 1 handoff this agent binds to. Nothing is loaded that is not
#: named, with a digest, in this document.
HANDOFF_ARTIFACT = "phase15_search_handoff_v1"

#: Where a report says this work sits.
PHASE15_STATUS_MARKERS = {
    "phase": "phase_15",
    "agent": "agent_02",
    "status": "engineering_deliverable_not_a_strength_claim",
    "scientific_validation_status": "not performed",
}


class Phase15SearchError(Phase12SearchError):
    """A Phase 15 search request was refused, or an invariant was violated.

    Subclasses the accepted Phase 12 error so a caller that already handles
    `Phase12SearchError` — the working player's fallback path, for one —
    keeps working unchanged.
    """


# ---------------------------------------------------------------------------
# Move models and belief providers
# ---------------------------------------------------------------------------

MOVE_P18 = "p18"
MOVE_P24 = "p24"
MOVE_MODELS = (MOVE_P18, MOVE_P24)

PROVIDER_REMAINING_COUNT = "remaining_count"
PROVIDER_B18 = "b18"
PROVIDER_B24 = "b24"
PROVIDER_ORACLE = "oracle"

#: The learned belief specialists Agent 1 delivered.
LEARNED_PROVIDERS = (PROVIDER_B18, PROVIDER_B24)

#: What a production constructor may build. The oracle is absent by
#: construction rather than by a check a caller could forget to run.
PRODUCTION_PROVIDERS = (PROVIDER_REMAINING_COUNT, PROVIDER_B18, PROVIDER_B24)
ALL_PROVIDERS = PRODUCTION_PROVIDERS + (PROVIDER_ORACLE,)

#: Section 15's structural fact, as a module constant the frozen candidate
#: and the tests both read.
ORACLE_AVAILABLE_IN_PRODUCTION = False

#: `b18` is fine-tuned over P18's frozen prefix, `b24` over P24's. This maps
#: a provider to the backbone its checkpoint will *refuse* to load without —
#: it is not a claim about which move model it should be paired with.
PROVIDER_BACKBONE = {PROVIDER_B18: MOVE_P18, PROVIDER_B24: MOVE_P24}


def check_production_provider(name: str) -> str:
    """Refuse any provider name a production configuration may not use."""
    if name not in ALL_PROVIDERS:
        raise Phase15SearchError(
            f"unknown belief provider {name!r}; Phase 15 provider names are "
            f"{list(ALL_PROVIDERS)}"
        )
    if name not in PRODUCTION_PROVIDERS:
        raise Phase15SearchError(
            f"{name!r} is an offline diagnostic and is not available in a "
            f"production configuration; production providers are "
            f"{list(PRODUCTION_PROVIDERS)}"
        )
    return name


# ---------------------------------------------------------------------------
# Pairings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pairing:
    """One complete system: a move model, a belief provider, a role.

    `kind` is `direct` (no search at all), `search` (a production arm) or
    `diagnostic` (the oracle ceiling, offline only).
    """

    pairing_id: str
    move_model: str
    provider: str | None
    kind: str
    description: str

    def __post_init__(self) -> None:
        if self.move_model not in MOVE_MODELS:
            raise Phase15SearchError(
                f"{self.pairing_id}: move model must be one of {list(MOVE_MODELS)}, "
                f"got {self.move_model!r}"
            )
        if self.kind not in ("direct", "search", "diagnostic"):
            raise Phase15SearchError(f"{self.pairing_id}: unknown kind {self.kind!r}")
        if self.kind == "direct":
            if self.provider is not None:
                raise Phase15SearchError(
                    f"{self.pairing_id}: a direct pairing carries no provider"
                )
        elif self.provider not in ALL_PROVIDERS:
            raise Phase15SearchError(
                f"{self.pairing_id}: unknown provider {self.provider!r}"
            )
        if self.kind == "search" and self.provider == PROVIDER_ORACLE:
            raise Phase15SearchError(
                f"{self.pairing_id}: the oracle is never a production search arm"
            )
        if self.kind == "diagnostic" and self.provider != PROVIDER_ORACLE:
            raise Phase15SearchError(
                f"{self.pairing_id}: only the oracle is a diagnostic pairing"
            )

    @property
    def is_learned(self) -> bool:
        return self.provider in LEARNED_PROVIDERS

    def describe(self) -> dict:
        return {
            "pairing_id": self.pairing_id,
            "move_model": self.move_model,
            "provider": self.provider,
            "kind": self.kind,
            "is_learned_belief": self.is_learned,
            "description": self.description,
        }


def _pairings() -> "tuple[Pairing, ...]":
    built: list[Pairing] = []
    for move in MOVE_MODELS:
        built.append(
            Pairing(
                pairing_id=f"{move}_direct",
                move_model=move,
                provider=None,
                kind="direct",
                description=f"{move.upper()} greedy, no search",
            )
        )
    for move in MOVE_MODELS:
        for provider in PRODUCTION_PROVIDERS:
            built.append(
                Pairing(
                    pairing_id=f"{move}_{provider}",
                    move_model=move,
                    provider=provider,
                    kind="search",
                    description=f"{move.upper()} + {provider} search",
                )
            )
    for move in MOVE_MODELS:
        built.append(
            Pairing(
                pairing_id=f"{move}_oracle",
                move_model=move,
                provider=PROVIDER_ORACLE,
                kind="diagnostic",
                description=f"{move.upper()} + oracle search (offline diagnostic)",
            )
        )
    return tuple(built)


ALL_PAIRINGS = _pairings()
PAIRINGS_BY_ID = {pairing.pairing_id: pairing for pairing in ALL_PAIRINGS}

#: The four complete systems section 14's matrix is about.
COMBINED_PAIRING_IDS = ("p18_b18", "p18_b24", "p24_b18", "p24_b24")

#: Every arm that may be played in a production match pack.
PRODUCTION_PAIRING_IDS = tuple(
    pairing.pairing_id for pairing in ALL_PAIRINGS if pairing.kind != "diagnostic"
)
DIAGNOSTIC_PAIRING_IDS = tuple(
    pairing.pairing_id for pairing in ALL_PAIRINGS if pairing.kind == "diagnostic"
)


def pairing(pairing_id: str) -> Pairing:
    try:
        return PAIRINGS_BY_ID[pairing_id]
    except KeyError:
        raise Phase15SearchError(
            f"unknown pairing {pairing_id!r}; known pairings are "
            f"{sorted(PAIRINGS_BY_ID)}"
        ) from None


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

#: The three instructed presets, imported from the accepted Phase 12 table
#: rather than restated, cheapest first.
LADDER_PRESET_NAMES = ("TINY", "SMALL", "MEDIUM")

#: Section 7's gated fourth rung: 64 worlds, <= 12 candidates, depth 10-12.
#: The depth is chosen from a latency pilot; 10 is the cheap end of the
#: instructed range and the value :data:`PRESET_STRONG` is defined at. This
#: is not a default and is never reached except through the section 7 gate.
STRONG_PRESET_NAME = "STRONG"
STRONG_DEPTH_RANGE = (10, 12)
PRESET_STRONG = Phase12SearchConfig(
    STRONG_PRESET_NAME, worlds=64, rollout_depth=10, max_root_candidates=12
)

SEARCH_PRESETS = dict(PHASE12_PRESETS)
SEARCH_PRESETS[STRONG_PRESET_NAME] = PRESET_STRONG

# ---------------------------------------------------------------------------
# The deeper-search pilot (Agent 2 follow-up)
# ---------------------------------------------------------------------------

#: The identity of the deeper-search pilot. A narrow follow-up question:
#: does buying 2-4x more search compute make the *selected* P24+B24 system
#: meaningfully stronger than MEDIUM? Nothing else is varied.
DEEP_PILOT_VERSION = "phase15_deep_search_pilot_v1"

#: The two new rungs. Compute grows primarily through **worlds** — 2x then
#: 3x MEDIUM's 32 — with a modest depth increase on top, which is the shape
#: the pilot was asked for. Everything the pilot is told to hold fixed is
#: held fixed *by omission*: `max_root_candidates`, `beta` and `epsilon` are
#: not passed, so they inherit `Phase12SearchConfig`'s defaults, which are
#: exactly MEDIUM's values. A test pins that.
#:
#: These are deliberately **not** :data:`PRESET_STRONG`, which raises the
#: candidate count to 12: the pilot forbids changing candidate handling, so
#: reusing the section 7 STRONG rung would have silently violated its own
#: control.
PRESET_LARGE = Phase12SearchConfig("LARGE", worlds=64, rollout_depth=9)
PRESET_XLARGE = Phase12SearchConfig("XLARGE", worlds=96, rollout_depth=11)

SEARCH_PRESETS[PRESET_LARGE.preset_id] = PRESET_LARGE
SEARCH_PRESETS[PRESET_XLARGE.preset_id] = PRESET_XLARGE

#: The pilot's three rungs, cheapest first. MEDIUM is the incumbent and is
#: not re-run: it was already measured on this exact board list, with these
#: exact per-decision seeds, in Stage C.
DEEP_PILOT_PRESET_NAMES = ("MEDIUM", "LARGE", "XLARGE")

#: The only system the pilot touches.
DEEP_PILOT_PAIRING = "p24_b24"

#: Naive compute units of one rung, `worlds * rollout_depth`. A planning
#: figure only — duplicate sampled worlds are evaluated once and weighted, so
#: the *measured* forward count is the honest cost and is what the pilot
#: reports.
def naive_compute_units(config: "Phase12SearchConfig") -> int:
    return int(config.worlds) * int(config.rollout_depth)


#: The gain the pilot is willing to call meaningful, as a band. Below the
#: low end the answer is "keep MEDIUM"; at or above it a stronger rung is
#: recommended for a maximum-strength mode.
DEEP_MEANINGFUL_GAIN_LOW = 0.03
DEEP_MEANINGFUL_GAIN_HIGH = 0.05

#: Section 7's practical latency targets, in seconds per move.
PREFERRED_MOVE_SECONDS = 2.0
ACCEPTABLE_MOVE_SECONDS = 5.0

#: An EWR difference this engineering pack is willing to read as a real
#: gain. Kept identical to the accepted Phase 12 margin so the two phases'
#: verdicts are spoken in one language.
MEANINGFUL_EWR_GAIN = 0.10


def preset(name: str) -> Phase12SearchConfig:
    try:
        return SEARCH_PRESETS[name]
    except KeyError:
        raise Phase15SearchError(
            f"unknown preset {name!r}; Phase 15 presets are {sorted(SEARCH_PRESETS)}"
        ) from None


def strong_preset(depth: int) -> Phase12SearchConfig:
    """The STRONG configuration at one depth from the instructed range."""
    low, high = STRONG_DEPTH_RANGE
    if not isinstance(depth, int) or isinstance(depth, bool) or not low <= depth <= high:
        raise Phase15SearchError(
            f"STRONG depth must be an int in {low}-{high}, got {depth!r}"
        )
    from dataclasses import replace

    return replace(PRESET_STRONG, rollout_depth=int(depth))


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

#: blake2b personalization of every Phase 15 *search* stream. Distinct from
#: Phase 12's ``strat-p12`` and Agent 1's ``strat-p15``.
PHASE15_SEARCH_PERSON = b"strat-p15s"

#: Any change to the payload layout, the personalization or the roots is a
#: new identity version.
SEARCH_IDENTITY_VERSION = "phase15_search_identity_v1"

#: Master seed of this agent, folded into every stream below.
SEARCH_MASTER_SEED = 2026082420

DOMAIN_PLAYER_SETUP = "search_player_setup"
DOMAIN_OPPONENT_SETUP = "search_opponent_setup"
DOMAIN_MATCH = "search_match"
DOMAIN_WORLDS = "search_worlds"
DOMAIN_PROBE = "search_probe"
DOMAIN_POSITION = "search_position"

STREAM_DOMAINS = (
    DOMAIN_PLAYER_SETUP,
    DOMAIN_OPPONENT_SETUP,
    DOMAIN_MATCH,
    DOMAIN_WORLDS,
    DOMAIN_PROBE,
    DOMAIN_POSITION,
)

DOMAIN_ROOTS = {
    DOMAIN_PLAYER_SETUP: 2026082421,
    DOMAIN_OPPONENT_SETUP: 2026082422,
    DOMAIN_MATCH: 2026082423,
    DOMAIN_WORLDS: 2026082424,
    DOMAIN_PROBE: 2026082425,
    DOMAIN_POSITION: 2026082426,
}
assert set(DOMAIN_ROOTS) == set(STREAM_DOMAINS)


def derive_search_seed(domain: str, *parts: "int | str") -> int:
    """A 63-bit deterministic seed for one Phase 15 search stream."""
    if domain not in STREAM_DOMAINS:
        raise Phase15SearchError(f"unknown Phase 15 search domain: {domain!r}")
    for part in parts:
        if not isinstance(part, (int, str)) or isinstance(part, bool):
            raise Phase15SearchError(
                f"stream identity parts must be int or str, got {type(part).__name__}"
            )
        if isinstance(part, str) and ":" in part:
            raise Phase15SearchError(
                f"string identity parts may not contain ':' (got {part!r})"
            )
    payload = ":".join(
        [
            SEARCH_IDENTITY_VERSION,
            domain,
            str(DOMAIN_ROOTS[domain]),
            *[str(part) for part in parts],
        ]
    )
    digest = hashlib.blake2b(
        payload.encode(), digest_size=8, person=PHASE15_SEARCH_PERSON
    ).digest()
    return int.from_bytes(digest, "big") >> 1


# ---------------------------------------------------------------------------
# The match design (section 12)
# ---------------------------------------------------------------------------

OPPONENT_P18 = "p18"
OPPONENT_P24 = "p24"
OPPONENT_PHASE9_ANCHOR = "phase9_anchor"
OPPONENT_STRATEGIC = "strategic_rule_based"
OPPONENT_TACTICAL = "tactical_rule_based"
OPPONENT_SCOUT_RUSH = "stress_scout_rush"
OPPONENT_MINER_RUSH = "stress_miner_rush"
OPPONENT_BERSERKER = "stress_berserker"
OPPONENT_INFORMATION_MISER = "stress_information_miser"
OPPONENT_CHAOS = "stress_chaos"

#: The ten section 12 opponents, in report order.
MATCH_OPPONENTS = (
    OPPONENT_P18,
    OPPONENT_P24,
    OPPONENT_PHASE9_ANCHOR,
    OPPONENT_STRATEGIC,
    OPPONENT_TACTICAL,
    OPPONENT_SCOUT_RUSH,
    OPPONENT_MINER_RUSH,
    OPPONENT_BERSERKER,
    OPPONENT_INFORMATION_MISER,
    OPPONENT_CHAOS,
)

#: The three neural opponents are frozen model objects; the rest are
#: accepted catalogue policies, resolved by id.
NEURAL_OPPONENTS = (OPPONENT_P18, OPPONENT_P24, OPPONENT_PHASE9_ANCHOR)
RULE_OPPONENT_POLICY_IDS = {
    OPPONENT_STRATEGIC: "strategic_rule_based",
    OPPONENT_TACTICAL: "tactical_rule_based",
    OPPONENT_SCOUT_RUSH: "stress_scout_rush",
    OPPONENT_MINER_RUSH: "stress_miner_rush",
    OPPONENT_BERSERKER: "stress_berserker",
    OPPONENT_INFORMATION_MISER: "stress_information_miser",
    OPPONENT_CHAOS: "stress_chaos",
}
OPPONENT_CLASS = {
    OPPONENT_P18: "neural",
    OPPONENT_P24: "neural",
    OPPONENT_PHASE9_ANCHOR: "neural",
    OPPONENT_STRATEGIC: "rule",
    OPPONENT_TACTICAL: "rule",
    OPPONENT_SCOUT_RUSH: "stress",
    OPPONENT_MINER_RUSH: "stress",
    OPPONENT_BERSERKER: "stress",
    OPPONENT_INFORMATION_MISER: "stress",
    OPPONENT_CHAOS: "stress",
}

#: Section 12's setup families, by their accepted library keys. The
#: `corner_flag_fortress / near_corner_flag_fortress` pair is drawn as two
#: families, as Agent 1 also read it.
MATCH_FAMILY_KEYS = (
    "balanced_conventional",
    "high_bomb_placement",
    "aggressive_high_rank_front",
    "conservative_high_rank_rear",
    "corner_flag_fortress",
    "near_corner_flag_fortress",
    "distributed_bomb_defense",
    "scout_forward_information",
    "miner_forward",
    "irregular_high_entropy",
)

#: The three Agent 1 setup sources, reused by name. A board's source applies
#: to both seats, so a board is a fair fight between two draws of the same
#: distribution.
SETUP_NEUTRAL = "neutral_v1"
SETUP_LEARNED = "phase14_learned"
SETUP_TARGETED = "targeted_family"
MATCH_SETUP_SOURCES = (SETUP_NEUTRAL, SETUP_LEARNED, SETUP_TARGETED)

MATCH_COLORS = ("red", "blue")

#: Neither the pool the spent Phase 11 sealed bank drew from (`test`) nor the
#: pool B18/B24 were trained on (`train`).
MATCH_LIBRARY_SPLIT = "validation"

#: The match-pack identity. Any change to the cells, the seed derivation or
#: the setup rule is a new version, never a silent edit.
MATCH_VERSION = "phase15_match_pack_v1"

#: The decision-diagnostic pack identity (section 11).
POSITION_VERSION = "phase15_decision_positions_v1"

MAX_ORDINAL_FORMAT = 999

_BOARD_ID_PATTERN = re.compile(
    rf"^{re.escape(MATCH_VERSION)}\|ms={SEARCH_MASTER_SEED}"
    rf"\|opp=(?P<opponent>[a-z0-9_]+)"
    rf"\|src=(?P<source>[a-z0-9_]+)"
    rf"\|fam=(?P<family>[a-z0-9_]+)"
    rf"\|col=(?P<color>red|blue)"
    rf"\|g=(?P<ordinal>[0-9]{{3}})$"
)


def board_id(
    opponent: str, setup_source: str, family_key: str, color: str, ordinal: int
) -> str:
    """The stable identifier of one Phase 15 match board."""
    if opponent not in MATCH_OPPONENTS:
        raise Phase15SearchError(
            f"opponent must be one of {list(MATCH_OPPONENTS)}, got {opponent!r}"
        )
    if setup_source not in MATCH_SETUP_SOURCES:
        raise Phase15SearchError(
            f"setup source must be one of {list(MATCH_SETUP_SOURCES)}, got "
            f"{setup_source!r}"
        )
    if color not in MATCH_COLORS:
        raise Phase15SearchError(f"colour must be red or blue, got {color!r}")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise Phase15SearchError(f"ordinal must be an int, got {type(ordinal).__name__}")
    if not 0 <= ordinal <= MAX_ORDINAL_FORMAT:
        raise Phase15SearchError(f"ordinal {ordinal} is outside 0..{MAX_ORDINAL_FORMAT}")
    identifier = (
        f"{MATCH_VERSION}|ms={SEARCH_MASTER_SEED}|opp={opponent}|src={setup_source}"
        f"|fam={family_key}|col={color}|g={ordinal:03d}"
    )
    if _BOARD_ID_PATTERN.match(identifier) is None:
        raise Phase15SearchError(f"constructed a malformed board id: {identifier!r}")
    return identifier


def parse_board_id(identifier: str) -> dict:
    """The identity fields of a Phase 15 board id, validated."""
    match = _BOARD_ID_PATTERN.match(identifier)
    if match is None:
        raise Phase15SearchError(f"malformed Phase 15 board id: {identifier!r}")
    fields = match.groupdict()
    if fields["opponent"] not in MATCH_OPPONENTS:
        raise Phase15SearchError(f"board id names unknown opponent {fields['opponent']!r}")
    if fields["source"] not in MATCH_SETUP_SOURCES:
        raise Phase15SearchError(f"board id names unknown source {fields['source']!r}")
    return {
        "opponent": fields["opponent"],
        "setup_source": fields["source"],
        "family_key": fields["family"],
        "color": fields["color"],
        "ordinal": int(fields["ordinal"]),
    }


def search_seed_for(identifier: str, ply: int) -> int:
    """The world-sampling seed of one decision of one board."""
    return derive_search_seed(DOMAIN_WORLDS, identifier, int(ply))


__all__ = [
    "ACCEPTABLE_MOVE_SECONDS",
    "ALL_PAIRINGS",
    "ALL_PROVIDERS",
    "BETA_DEFAULT",
    "COMBINED_PAIRING_IDS",
    "DEEP_MEANINGFUL_GAIN_HIGH",
    "DEEP_MEANINGFUL_GAIN_LOW",
    "DEEP_PILOT_PAIRING",
    "DEEP_PILOT_PRESET_NAMES",
    "DEEP_PILOT_VERSION",
    "DIAGNOSTIC_PAIRING_IDS",
    "DOMAIN_MATCH",
    "DOMAIN_OPPONENT_SETUP",
    "DOMAIN_PLAYER_SETUP",
    "DOMAIN_POSITION",
    "DOMAIN_PROBE",
    "DOMAIN_WORLDS",
    "EPSILON_DEFAULT",
    "HANDOFF_ARTIFACT",
    "INTEGRATION_VERSION",
    "LADDER_PRESET_NAMES",
    "LEARNED_PROVIDERS",
    "MATCH_COLORS",
    "MATCH_FAMILY_KEYS",
    "MATCH_LIBRARY_SPLIT",
    "MATCH_OPPONENTS",
    "MATCH_SETUP_SOURCES",
    "MATCH_VERSION",
    "MEANINGFUL_EWR_GAIN",
    "MOVE_MODELS",
    "MOVE_P18",
    "MOVE_P24",
    "NEURAL_OPPONENTS",
    "OPPONENT_CLASS",
    "ORACLE_AVAILABLE_IN_PRODUCTION",
    "PAIRINGS_BY_ID",
    "PHASE15_SCORE_DEFINITION",
    "PHASE15_SEARCH_VERSION",
    "PHASE15_STATUS_MARKERS",
    "POSITION_VERSION",
    "PREFERRED_MOVE_SECONDS",
    "PRESET_MEDIUM",
    "PRESET_SMALL",
    "PRESET_LARGE",
    "PRESET_STRONG",
    "PRESET_XLARGE",
    "PRESET_TINY",
    "PRODUCTION_PAIRING_IDS",
    "PRODUCTION_PROVIDERS",
    "PROVIDER_B18",
    "PROVIDER_B24",
    "PROVIDER_BACKBONE",
    "PROVIDER_ORACLE",
    "PROVIDER_REMAINING_COUNT",
    "RULE_OPPONENT_POLICY_IDS",
    "SEARCH_MASTER_SEED",
    "SEARCH_PRESETS",
    "SETUP_LEARNED",
    "SETUP_NEUTRAL",
    "SETUP_TARGETED",
    "STRONG_DEPTH_RANGE",
    "STRONG_PRESET_NAME",
    "Pairing",
    "Phase15SearchError",
    "Phase12SearchConfig",
    "Phase12SearchTimeout",
    "board_id",
    "check_production_provider",
    "derive_search_seed",
    "naive_compute_units",
    "pairing",
    "parse_board_id",
    "preset",
    "search_seed_for",
    "strong_preset",
]
