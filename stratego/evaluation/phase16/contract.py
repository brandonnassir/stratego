"""Phase 16 Agent 1: the frozen identities of the measurement instruments.

Specification source: `01_AGENT_1_MEASUREMENT_AND_OPERATOR_EXAM.md`, bound by
`00_PHASE_16_OVERVIEW.md`.

What is identity here and what is reuse
---------------------------------------
The *instruments* — which boards, which opponents, which arms, which seeds —
are Phase 16 identities and live in this module. The *machinery* — how a
board is drawn, oriented, played and summarised — is Phase 15's, imported
unchanged. The opponent roster, the setup sources and the accepted-library
family keys are re-exported from `stratego.search.phase15.contract` rather
than restated, so the two phases can never silently disagree about them.

The seed streams
----------------
A new blake2b personalization, ``strat-p16m`` (m for measurement), with new
domain roots, per overview section 6 ("seeds derive from the phase
namespace"). Every payload begins ``phase16.agent1``. No Phase 12/14/15
stream can coincide with a Phase 16 measurement stream even where two
identity strings spell the same text.

One deliberate exception is documented rather than hidden: the *in-search
world-sampling* seeds of a Phase 16 game come through the imported Phase 15
seat (`search_seed_for(board_id, ply)`, personalization ``strat-p15s``),
keyed on the Phase 16 board id. Those seeds are a property of the accepted
seat being reused by import; they are disjoint from every Phase 15 stream
because no Phase 15 board id carries the ``phase16_`` prefix.
"""

from __future__ import annotations

import hashlib
import re

# Re-exported Phase 15 identities: the roster and setup vocabulary are shared
# by construction, not by coincidence.
from ...search.phase15.contract import (  # noqa: F401
    MATCH_COLORS,
    MATCH_FAMILY_KEYS,
    MATCH_LIBRARY_SPLIT,
    MATCH_OPPONENTS,
    MATCH_SETUP_SOURCES,
    NEURAL_OPPONENTS,
    OPPONENT_CLASS,
    SETUP_LEARNED,
    SETUP_NEUTRAL,
    SETUP_TARGETED,
    Phase15SearchError,
)

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

#: The canonical machine-opponent pack. This pack never changes; extensions
#: get a new version.
BENCHMARK_VERSION = "phase16_benchmark_v1"

#: The adversarial setup library.
ADVERSARIAL_VERSION = "phase16_adversarial_setups_v1"

#: The paired baseline measurement played on the adversarial library.
ADVERSARIAL_BASELINE_VERSION = "phase16_adversarial_baseline_v1"

#: The scoring runner identity the handoff records.
RUNNER_VERSION = "phase16_measurement_runner_v1"

#: The operator game log schema.
OPERATOR_LOG_SCHEMA = "phase16_operator_game_v1"

#: The Agent 1 handoff artifact name.
MEASUREMENT_HANDOFF_ARTIFACT = "phase16_measurement_handoff_v1"

#: Where a Phase 16 report says this work sits.
PHASE16_STATUS_MARKERS = {
    "phase": "phase_16",
    "agent": "agent_01",
    "status": "engineering_deliverable_not_a_strength_claim",
    "scientific_validation_status": "not performed",
}

#: Engineering margins, unchanged from Phase 15 (overview section 6).
SELECTION_MARGIN = 0.10
MEANINGFUL_BAND = (0.03, 0.05)

#: The predeclared reading of the adversarial baseline (brief section 5).
ADVERSARIAL_CONFIRM_DROP = 0.10
ADVERSARIAL_WEAKEN_DROP = 0.05


class Phase16MeasurementError(Phase15SearchError):
    """A Phase 16 measurement request was refused or an invariant violated.

    Subclasses the accepted Phase 15 error so callers that already handle
    the Phase 15 family keep working unchanged.
    """


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

#: blake2b personalization of every Phase 16 *measurement* stream.
PHASE16_MEASUREMENT_PERSON = b"strat-p16m"

#: Any change to the payload layout, the personalization or the roots is a
#: new identity version.
MEASUREMENT_IDENTITY_VERSION = "phase16_measurement_identity_v1"

#: The overview's namespace prefix, folded into every payload.
SEED_NAMESPACE = "phase16.agent1"

#: Master seed of this agent.
MEASUREMENT_MASTER_SEED = 2026082510

DOMAIN_PLAYER_SETUP = "measure_player_setup"
DOMAIN_OPPONENT_SETUP = "measure_opponent_setup"
DOMAIN_MATCH = "measure_match"
DOMAIN_ADVERSARIAL = "measure_adversarial_author"
DOMAIN_CAPTURE = "measure_capture"

STREAM_DOMAINS = (
    DOMAIN_PLAYER_SETUP,
    DOMAIN_OPPONENT_SETUP,
    DOMAIN_MATCH,
    DOMAIN_ADVERSARIAL,
    DOMAIN_CAPTURE,
)

DOMAIN_ROOTS = {
    DOMAIN_PLAYER_SETUP: 2026082511,
    DOMAIN_OPPONENT_SETUP: 2026082512,
    DOMAIN_MATCH: 2026082513,
    DOMAIN_ADVERSARIAL: 2026082514,
    DOMAIN_CAPTURE: 2026082515,
}
assert set(DOMAIN_ROOTS) == set(STREAM_DOMAINS)


def derive_measure_seed(domain: str, *parts: "int | str") -> int:
    """A 63-bit deterministic seed for one Phase 16 measurement stream."""
    if domain not in STREAM_DOMAINS:
        raise Phase16MeasurementError(f"unknown Phase 16 measurement domain: {domain!r}")
    for part in parts:
        if not isinstance(part, (int, str)) or isinstance(part, bool):
            raise Phase16MeasurementError(
                f"stream identity parts must be int or str, got {type(part).__name__}"
            )
        if isinstance(part, str) and ":" in part:
            raise Phase16MeasurementError(
                f"string identity parts may not contain ':' (got {part!r})"
            )
    payload = ":".join(
        [
            MEASUREMENT_IDENTITY_VERSION,
            SEED_NAMESPACE,
            domain,
            str(DOMAIN_ROOTS[domain]),
            *[str(part) for part in parts],
        ]
    )
    digest = hashlib.blake2b(
        payload.encode(), digest_size=8, person=PHASE16_MEASUREMENT_PERSON
    ).digest()
    return int.from_bytes(digest, "big") >> 1


# ---------------------------------------------------------------------------
# The benchmark pack design (brief section 4)
# ---------------------------------------------------------------------------

#: Boards per (opponent, setup source, colour) cell. 60 cells x 2 = the 120
#: paired boards the brief asks for.
BENCHMARK_BOARDS_PER_CELL = 2

#: Ordinal 0 of every cell is the predeclared 60-board quick subset for
#: training-run checkpoint scoring: balanced over opponent, source and colour
#: by construction, and a strict subset of the full pack.
QUICK_SUBSET_ORDINAL = 0
QUICK_SUBSET_NAME = "quick60"

#: The three Phase 16 baseline arms the brief names, with their presets.
BENCHMARK_BASELINES = (
    ("p24_direct", "direct"),
    ("p24_b24", "TINY"),
    ("p24_b24", "MEDIUM"),
)

MAX_ORDINAL_FORMAT = 999

_BENCH_BOARD_ID_PATTERN = re.compile(
    rf"^{re.escape(BENCHMARK_VERSION)}\|ms={MEASUREMENT_MASTER_SEED}"
    rf"\|opp=(?P<opponent>[a-z0-9_]+)"
    rf"\|src=(?P<source>[a-z0-9_]+)"
    rf"\|fam=(?P<family>[a-z0-9_]+)"
    rf"\|col=(?P<color>red|blue)"
    rf"\|g=(?P<ordinal>[0-9]{{3}})$"
)


def benchmark_board_id(
    opponent: str, setup_source: str, family_key: str, color: str, ordinal: int
) -> str:
    """The stable identifier of one Phase 16 benchmark board."""
    if opponent not in MATCH_OPPONENTS:
        raise Phase16MeasurementError(
            f"opponent must be one of {list(MATCH_OPPONENTS)}, got {opponent!r}"
        )
    if setup_source not in MATCH_SETUP_SOURCES:
        raise Phase16MeasurementError(
            f"setup source must be one of {list(MATCH_SETUP_SOURCES)}, got "
            f"{setup_source!r}"
        )
    if color not in MATCH_COLORS:
        raise Phase16MeasurementError(f"colour must be red or blue, got {color!r}")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise Phase16MeasurementError(
            f"ordinal must be an int, got {type(ordinal).__name__}"
        )
    if not 0 <= ordinal <= MAX_ORDINAL_FORMAT:
        raise Phase16MeasurementError(f"ordinal {ordinal} is outside 0..{MAX_ORDINAL_FORMAT}")
    identifier = (
        f"{BENCHMARK_VERSION}|ms={MEASUREMENT_MASTER_SEED}|opp={opponent}"
        f"|src={setup_source}|fam={family_key}|col={color}|g={ordinal:03d}"
    )
    if _BENCH_BOARD_ID_PATTERN.match(identifier) is None:
        raise Phase16MeasurementError(f"constructed a malformed board id: {identifier!r}")
    return identifier


def parse_benchmark_board_id(identifier: str) -> dict:
    match = _BENCH_BOARD_ID_PATTERN.match(identifier)
    if match is None:
        raise Phase16MeasurementError(f"malformed Phase 16 benchmark board id: {identifier!r}")
    fields = match.groupdict()
    if fields["opponent"] not in MATCH_OPPONENTS:
        raise Phase16MeasurementError(
            f"board id names unknown opponent {fields['opponent']!r}"
        )
    if fields["source"] not in MATCH_SETUP_SOURCES:
        raise Phase16MeasurementError(f"board id names unknown source {fields['source']!r}")
    return {
        "opponent": fields["opponent"],
        "setup_source": fields["source"],
        "family_key": fields["family"],
        "color": fields["color"],
        "ordinal": int(fields["ordinal"]),
    }


# ---------------------------------------------------------------------------
# The adversarial library design (brief section 5)
# ---------------------------------------------------------------------------

#: The required families, in report order. `operator_harvest` is present but
#: holds no authored setup: its entries come only from the capture tool and
#: the log harvester.
FAMILY_OPERATOR_HARVEST = "operator_harvest"
ADVERSARIAL_FAMILIES = (
    FAMILY_OPERATOR_HARVEST,
    "bombed_corner_flag",
    "bombed_center_flag",
    "scout_screen",
    "aggressive_marshal",
    "spy_shadow",
    "miner_wall",
    "decoy_flag_structure",
    "free_novelty",
)
AUTHORED_FAMILIES = tuple(
    family for family in ADVERSARIAL_FAMILIES if family != FAMILY_OPERATOR_HARVEST
)

#: Authored setups per family. 8 x 12 = 96, inside the required 96-128.
SETUPS_PER_FAMILY = 12

#: The three baseline arms (brief section 5).
ARM_CONTROL = "benchmark_control"
ARM_ADVERSARIAL_OPPONENT = "adversarial_opponent"
ARM_ADVERSARIAL_BOTH = "adversarial_both"
BASELINE_ARMS = (ARM_CONTROL, ARM_ADVERSARIAL_OPPONENT, ARM_ADVERSARIAL_BOTH)

#: The system under measurement and the two instructed presets.
BASELINE_PAIRING = "p24_b24"
BASELINE_PRESETS = ("TINY", "MEDIUM")

_ADV_BOARD_ID_PATTERN = re.compile(
    rf"^{re.escape(ADVERSARIAL_BASELINE_VERSION)}\|ms={MEASUREMENT_MASTER_SEED}"
    rf"\|arm=(?P<arm>[a-z0-9_]+)"
    rf"\|fam=(?P<family>[a-z0-9_]+)"
    rf"\|opp=(?P<opponent>[a-z0-9_]+)"
    rf"\|col=(?P<color>red|blue)"
    rf"\|pair=(?P<pair>[0-9]{{3}})$"
)


def adversarial_board_id(
    arm: str, family: str, opponent: str, color: str, pair_index: int
) -> str:
    """The stable identifier of one adversarial-baseline board."""
    if arm not in BASELINE_ARMS:
        raise Phase16MeasurementError(
            f"arm must be one of {list(BASELINE_ARMS)}, got {arm!r}"
        )
    if family not in ADVERSARIAL_FAMILIES:
        raise Phase16MeasurementError(
            f"family must be one of {list(ADVERSARIAL_FAMILIES)}, got {family!r}"
        )
    if opponent not in MATCH_OPPONENTS:
        raise Phase16MeasurementError(
            f"opponent must be one of {list(MATCH_OPPONENTS)}, got {opponent!r}"
        )
    if color not in MATCH_COLORS:
        raise Phase16MeasurementError(f"colour must be red or blue, got {color!r}")
    if not isinstance(pair_index, int) or isinstance(pair_index, bool):
        raise Phase16MeasurementError(
            f"pair index must be an int, got {type(pair_index).__name__}"
        )
    if not 0 <= pair_index <= MAX_ORDINAL_FORMAT:
        raise Phase16MeasurementError(
            f"pair index {pair_index} is outside 0..{MAX_ORDINAL_FORMAT}"
        )
    identifier = (
        f"{ADVERSARIAL_BASELINE_VERSION}|ms={MEASUREMENT_MASTER_SEED}|arm={arm}"
        f"|fam={family}|opp={opponent}|col={color}|pair={pair_index:03d}"
    )
    if _ADV_BOARD_ID_PATTERN.match(identifier) is None:
        raise Phase16MeasurementError(f"constructed a malformed board id: {identifier!r}")
    return identifier


def parse_adversarial_board_id(identifier: str) -> dict:
    match = _ADV_BOARD_ID_PATTERN.match(identifier)
    if match is None:
        raise Phase16MeasurementError(
            f"malformed Phase 16 adversarial board id: {identifier!r}"
        )
    fields = match.groupdict()
    return {
        "arm": fields["arm"],
        "family": fields["family"],
        "opponent": fields["opponent"],
        "color": fields["color"],
        "pair_index": int(fields["pair"]),
    }


__all__ = [
    "ADVERSARIAL_BASELINE_VERSION",
    "ADVERSARIAL_CONFIRM_DROP",
    "ADVERSARIAL_FAMILIES",
    "ADVERSARIAL_VERSION",
    "ADVERSARIAL_WEAKEN_DROP",
    "ARM_ADVERSARIAL_BOTH",
    "ARM_ADVERSARIAL_OPPONENT",
    "ARM_CONTROL",
    "AUTHORED_FAMILIES",
    "BASELINE_ARMS",
    "BASELINE_PAIRING",
    "BASELINE_PRESETS",
    "BENCHMARK_BASELINES",
    "BENCHMARK_BOARDS_PER_CELL",
    "BENCHMARK_VERSION",
    "DOMAIN_ADVERSARIAL",
    "DOMAIN_CAPTURE",
    "DOMAIN_MATCH",
    "DOMAIN_OPPONENT_SETUP",
    "DOMAIN_PLAYER_SETUP",
    "FAMILY_OPERATOR_HARVEST",
    "MATCH_COLORS",
    "MATCH_FAMILY_KEYS",
    "MATCH_LIBRARY_SPLIT",
    "MATCH_OPPONENTS",
    "MATCH_SETUP_SOURCES",
    "MEANINGFUL_BAND",
    "MEASUREMENT_HANDOFF_ARTIFACT",
    "MEASUREMENT_IDENTITY_VERSION",
    "MEASUREMENT_MASTER_SEED",
    "OPERATOR_LOG_SCHEMA",
    "PHASE16_STATUS_MARKERS",
    "Phase16MeasurementError",
    "QUICK_SUBSET_NAME",
    "QUICK_SUBSET_ORDINAL",
    "RUNNER_VERSION",
    "SELECTION_MARGIN",
    "SETUPS_PER_FAMILY",
    "adversarial_board_id",
    "benchmark_board_id",
    "derive_measure_seed",
    "parse_adversarial_board_id",
    "parse_benchmark_board_id",
]
