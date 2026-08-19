"""Phase 11 Agent 1: frozen belief-validation seeds and logical identities.

Specification sources:

- `00_PHASE_11_SEQUENCE_AND_COMMON_CONTRACT.md` ("Root seeds", "Phase 11
  banks", "Reproducibility", "Information-safety attack")
- `01_AGENT_1_CONTRACTS_SEEDS_BANKS_ACCEPTANCE.md` ("Freeze seeds and
  domains")

What lives here and why
-----------------------
Everything in this module is *identity*: the eight frozen Phase 11 root
seeds, the evaluation-case / game / prediction-event / world-sample /
safety-trial / reproducibility-request / benchmark-state / soak-request
identifiers, and the domain-separated stream seeds every Phase 11 consumer
draws from. Nothing here knows what a belief metric is, what an acceptance
gate says, or how a bank balances its strata —
:mod:`stratego.training.phase11_contract` layers the frozen experiment on
top of these identities, exactly as `phase10_contract` sat over
`phase10_seed`.

No derivation reads worker count, task arrival order, process id, wall
clock, or a physical storage path, so a bank case, a sampled world, a
safety trial or a soak request can always be rebuilt alone.

Seeds were chosen before any Phase 11 prediction, sampler output or test
outcome existed. They follow the repository's date-seed precedent
(`20260818xx` Phase 10): the Phase 11 block is the freeze date `20260819`
extended with a two-digit role suffix, fixed by the calendar rather than by
anything measured.

Derivation
----------
All streams come from :func:`derive_phase11_seed`, a domain-separated
``blake2b`` hash under the Phase 11 personalization tag ``strat-b11``
("belief, phase 11") — distinct from every earlier tag (``strat-s10``
Phase 10, ``strat-rl9`` Phase 9, ``strat-ws8`` Phase 8,
``strat-lb7``/``strat-at7``/``strat-st7`` Phase 7, ``strat-bnk``/
``strat-sid`` Phase 4 bank, ``strat-pls``/``strat-dec`` match/decision
seeds, ``strat-unt``/``strat-mch`` match identity), so no Phase 11 stream
can collide with any accepted upstream stream.

Root assignment
---------------
The common contract names eight roots. Their frozen consumers are:

```text
2026081901 master                 folded into every Phase 11 logical id
2026081902 bank/case schedule     bank setup draws; soak setup draws
2026081903 game/match randomness  bank match seeds; soak match seeds
2026081904 belief/world sampling  world-sample seeds, piece order, categorical
2026081905 information safety     hidden-truth permutation trial streams
2026081906 repro/runtime audit    topology replay + benchmark schedule streams
2026081907 validation bootstrap   validation-bank resampling streams
2026081908 final-test bootstrap   test-bank resampling streams
```

Two readings the common contract does not spell out letter by letter, both
recorded in the Agent 1 acceptance artifact:

- the contract froze no dedicated Agent 6 soak root (the Phase 10 soak had
  to invent its namespace after the fact — a recorded deviation this phase
  closes), so the soak's setup, match and sampling streams hang off the
  same three roots as their bank counterparts under *distinct domain
  tokens* (`soak_setup` vs `bank_observer_setup`/`bank_opponent_setup`,
  `soak_match` vs `bank_match`, and the shared `world_sample` domain keyed
  by soak public-state identity), which is exactly how the Phase 10 soak
  namespace was reconciled — except frozen now, in advance;
- both banks' streams hang off the single bank/case-schedule root
  `2026081902`, domain-separated by the two distinct bank-version tokens
  inside every case id (the Phase 10 `case_schedule_seed` reading, reused).

Neither reading adds a root, reuses a stream, or lets two derivations
collide; :func:`stream_collision_audit` proves the enumerable part
exhaustively and the contract carries the downstream obligation for the
million-scale world-sample space.
"""

from __future__ import annotations

import hashlib
import re

#: The Phase 11 logical-identity version. A change to an id format, to the
#: seed derivation, or to the domain tokens is a new version, never a silent
#: edit.
PHASE11_IDENTITY_VERSION = "phase11_identity_v1"

# ---------------------------------------------------------------------------
# Canonical Phase 11 root seeds — frozen by the common contract before any
# Phase 11 prediction, sampler output or test outcome existed.
# ---------------------------------------------------------------------------

#: Master seed of the whole phase. Folded into every Phase 11 logical id.
PHASE11_MASTER_SEED = 2026081901

#: Root of the bank/case schedule streams: every bank setup draw (observer
#: and opponent, both banks) and the soak's setup draws.
BANK_SCHEDULE_SEED = 2026081902

#: Root of game/match randomness: bank per-game match seeds and soak match
#: seeds. Rule-based opponents draw their per-decision randomness from the
#: accepted runner derivations rooted in these seeds; the observer plays the
#: accepted greedy decision mode and consumes none of it.
MATCH_RANDOMNESS_SEED = 2026081903

#: Root of belief/world sampling: world-sample seeds, the per-sample
#: unresolved-piece-order stream and the per-step categorical stream, for
#: the learned sampler and the count-uniform baseline sampler alike.
WORLD_SAMPLING_SEED = 2026081904

#: Root of the information-safety hidden-truth permutation trials.
INFORMATION_SAFETY_SEED = 2026081905

#: Root of the reproducibility/topology replay schedule and the runtime
#: benchmark schedule.
REPRO_RUNTIME_SEED = 2026081906

#: Frozen bootstrap roots of the two evaluation banks.
VALIDATION_BOOTSTRAP_SEED = 2026081907
TEST_BOOTSTRAP_SEED = 2026081908

CANONICAL_PHASE11_SEEDS = {
    "phase11_master_seed": PHASE11_MASTER_SEED,
    "bank_schedule_seed": BANK_SCHEDULE_SEED,
    "match_randomness_seed": MATCH_RANDOMNESS_SEED,
    "world_sampling_seed": WORLD_SAMPLING_SEED,
    "information_safety_seed": INFORMATION_SAFETY_SEED,
    "repro_runtime_seed": REPRO_RUNTIME_SEED,
    "validation_bootstrap_seed": VALIDATION_BOOTSTRAP_SEED,
    "test_bootstrap_seed": TEST_BOOTSTRAP_SEED,
}

# ---------------------------------------------------------------------------
# Stream domains
# ---------------------------------------------------------------------------

#: blake2b personalization of every Phase 11 stream.
_PHASE11_SEED_PERSON = b"strat-b11"

DOMAIN_BANK_OBSERVER_SETUP = "bank_observer_setup"
DOMAIN_BANK_OPPONENT_SETUP = "bank_opponent_setup"
DOMAIN_BANK_MATCH = "bank_match"
DOMAIN_WORLD_SAMPLE = "world_sample"
DOMAIN_WORLD_ORDER = "world_order"
DOMAIN_WORLD_CATEGORICAL = "world_categorical"
DOMAIN_SAFETY_TRIAL = "safety_trial"
DOMAIN_REPRO_SCHEDULE = "repro_schedule"
DOMAIN_BENCHMARK = "benchmark"
DOMAIN_BOOTSTRAP = "bootstrap"
DOMAIN_SOAK_SETUP = "soak_setup"
DOMAIN_SOAK_MATCH = "soak_match"

STREAM_DOMAINS = (
    DOMAIN_BANK_OBSERVER_SETUP,
    DOMAIN_BANK_OPPONENT_SETUP,
    DOMAIN_BANK_MATCH,
    DOMAIN_WORLD_SAMPLE,
    DOMAIN_WORLD_ORDER,
    DOMAIN_WORLD_CATEGORICAL,
    DOMAIN_SAFETY_TRIAL,
    DOMAIN_REPRO_SCHEDULE,
    DOMAIN_BENCHMARK,
    DOMAIN_BOOTSTRAP,
    DOMAIN_SOAK_SETUP,
    DOMAIN_SOAK_MATCH,
)

#: The root seed each domain hangs off, frozen here so the binding is data
#: rather than a convention repeated at every call site.
DOMAIN_ROOTS = {
    DOMAIN_BANK_OBSERVER_SETUP: BANK_SCHEDULE_SEED,
    DOMAIN_BANK_OPPONENT_SETUP: BANK_SCHEDULE_SEED,
    DOMAIN_BANK_MATCH: MATCH_RANDOMNESS_SEED,
    DOMAIN_WORLD_SAMPLE: WORLD_SAMPLING_SEED,
    DOMAIN_WORLD_ORDER: WORLD_SAMPLING_SEED,
    DOMAIN_WORLD_CATEGORICAL: WORLD_SAMPLING_SEED,
    DOMAIN_SAFETY_TRIAL: INFORMATION_SAFETY_SEED,
    DOMAIN_REPRO_SCHEDULE: REPRO_RUNTIME_SEED,
    DOMAIN_BENCHMARK: REPRO_RUNTIME_SEED,
    DOMAIN_BOOTSTRAP: PHASE11_MASTER_SEED,
    DOMAIN_SOAK_SETUP: BANK_SCHEDULE_SEED,
    DOMAIN_SOAK_MATCH: MATCH_RANDOMNESS_SEED,
}
assert set(DOMAIN_ROOTS) == set(STREAM_DOMAINS)

#: Colours, as text, in every stream identity — never engine integers.
COLOR_RED = "red"
COLOR_BLUE = "blue"
COLORS = (COLOR_RED, COLOR_BLUE)

#: Frozen colour pairing of an evaluation case: the observer (the accepted
#: Phase 9 policy + belief head) plays Red in game 0 and Blue in game 1.
#: The opponent stratum plays the other colour of each game.
CASE_GAME_INDICES = (0, 1)
CASE_GAME_OBSERVER_COLOR = {0: COLOR_RED, 1: COLOR_BLUE}
CASE_GAME_OPPONENT_COLOR = {0: COLOR_BLUE, 1: COLOR_RED}
assert tuple(sorted(CASE_GAME_OBSERVER_COLOR)) == CASE_GAME_INDICES

#: The two per-game seat roles a bank setup draw belongs to.
ROLE_OBSERVER = "observer"
ROLE_OPPONENT = "opponent"
SETUP_ROLES = (ROLE_OBSERVER, ROLE_OPPONENT)

#: The eight opponent-behaviour strata, in the common contract's order.
#: These tokens appear verbatim inside case ids; their policy bindings are
#: frozen in `phase11_contract`.
STRATUM_PHASE9 = "phase9_selfplay"
STRATUM_PHASE8_ANCHOR = "phase8_anchor"
STRATUM_STRATEGIC = "strategic_rule"
STRATUM_TACTICAL = "tactical_rule"
STRATUM_BASIC = "basic_rule"
STRATUM_INFORMATION_MISER = "information_miser"
STRATUM_SCOUT_RUSH = "scout_rush"
STRATUM_MINER_RUSH = "miner_rush"

OPPONENT_STRATA = (
    STRATUM_PHASE9,
    STRATUM_PHASE8_ANCHOR,
    STRATUM_STRATEGIC,
    STRATUM_TACTICAL,
    STRATUM_BASIC,
    STRATUM_INFORMATION_MISER,
    STRATUM_SCOUT_RUSH,
    STRATUM_MINER_RUSH,
)
assert len(set(OPPONENT_STRATA)) == 8

#: The two opponent setup-source strata, in the frozen enumeration order.
#: `p10d` is the accepted Phase 10 production source (P10-D); `neutral` is
#: the accepted `neutral_v1` baseline sampler.
SOURCE_P10D = "p10d"
SOURCE_NEUTRAL = "neutral"
SETUP_SOURCES = (SOURCE_P10D, SOURCE_NEUTRAL)


class Phase11SeedError(ValueError):
    """Raised when a Phase 11 identity or seed request is malformed."""


def derive_phase11_seed(domain: str, *parts: "int | str") -> int:
    """A 63-bit deterministic seed for one Phase 11 stream.

    ``domain`` must be one of :data:`STREAM_DOMAINS`; ``parts`` are the
    identity inputs of the stream. The payload is the colon-joined text of
    the identity version, the domain, the domain's frozen root seed and the
    parts, under the ``strat-b11`` personalization — so equal identities
    always agree, any change to any identity input yields an unrelated
    stream, and two domains sharing a root still cannot collide. String
    parts may not contain ``:`` — that keeps the colon-joined payload an
    injective encoding of the part tuple (Phase 11 ids use ``|`` and ``=``
    exclusively, so every frozen id is a legal single part).
    """
    if domain not in STREAM_DOMAINS:
        raise Phase11SeedError(f"unknown Phase 11 stream domain: {domain!r}")
    for part in parts:
        if not isinstance(part, (int, str)) or isinstance(part, bool):
            raise Phase11SeedError(
                f"stream identity parts must be int or str, got {type(part).__name__}"
            )
        if isinstance(part, str) and ":" in part:
            raise Phase11SeedError(
                f"string identity parts may not contain ':' (got {part!r}); the "
                "colon is the payload separator"
            )
    payload = ":".join(
        [
            PHASE11_IDENTITY_VERSION,
            domain,
            str(DOMAIN_ROOTS[domain]),
            *[str(part) for part in parts],
        ]
    )
    digest = hashlib.blake2b(
        payload.encode(), digest_size=8, person=_PHASE11_SEED_PERSON
    ).digest()
    return int.from_bytes(digest, "big") >> 1


def unit_uniform(seed: int) -> float:
    """A unit uniform from a 63-bit stream seed, exactly reproducible.

    Division by ``2**63`` is deterministic in binary floating point, so the
    value is identical on every platform and the frozen inverse-CDF walks
    cannot drift. The Phase 10 convention, re-frozen bit-for-bit for
    Phase 11 — including its one honest edge: the top ~2^9 seed values
    round to exactly ``1.0`` under float64, which is why every frozen
    inverse-CDF walk carries a last-element tail guard that absorbs
    ``u >= cumulative mass``.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise Phase11SeedError(f"stream seed must be a non-negative int, got {seed!r}")
    return seed / float(1 << 63)


# ---------------------------------------------------------------------------
# Evaluation-case and game identity
# ---------------------------------------------------------------------------

_STRATUM_ALTERNATION = "|".join(OPPONENT_STRATA)
_SOURCE_ALTERNATION = "|".join(SETUP_SOURCES)

_CASE_ID_PATTERN = re.compile(
    rf"^(?P<bank>phase11_[a-z]+_bank_v[0-9]+)\|ms=(?P<master>[0-9]+)"
    rf"\|st=(?P<stratum>{_STRATUM_ALTERNATION})\|src=(?P<source>{_SOURCE_ALTERNATION})"
    rf"\|c=(?P<ordinal>[0-9]{{3}})$"
)

MAX_CASE_ORDINAL_FORMAT = 999

_GAME_ID_PATTERN = re.compile(r"^(?P<case>.+)\|g=(?P<game>[01])$")


def phase11_case_id(
    bank_version: str, stratum: str, setup_source: str, case_ordinal: int
) -> str:
    """The stable identifier of one logical Phase 11 paired case.

    ```text
    phase11_validation_bank_v1|ms=2026081901|st=scout_rush|src=p10d|c=017
    ```

    The stratum is the case's opponent behaviour, the source is the case's
    opponent setup-source stratum, and the ordinal is the counter inside
    that (stratum, source) cell. Balance over strata, sources and colours is
    therefore a property of the id space, never of any draw. Both banks use
    the same format and differ only in their version token, which keeps
    every validation stream disjoint from every test stream while both hang
    off one frozen bank-schedule root.
    """
    if not isinstance(bank_version, str) or not bank_version:
        raise Phase11SeedError(
            f"bank_version must be a non-empty string, got {bank_version!r}"
        )
    if stratum not in OPPONENT_STRATA:
        raise Phase11SeedError(
            f"stratum must be one of {list(OPPONENT_STRATA)}, got {stratum!r}"
        )
    if setup_source not in SETUP_SOURCES:
        raise Phase11SeedError(
            f"setup_source must be one of {list(SETUP_SOURCES)}, got {setup_source!r}"
        )
    if not isinstance(case_ordinal, int) or isinstance(case_ordinal, bool):
        raise Phase11SeedError(
            f"case ordinal must be an int, got {type(case_ordinal).__name__}"
        )
    if not 0 <= case_ordinal <= MAX_CASE_ORDINAL_FORMAT:
        raise Phase11SeedError(
            f"case ordinal {case_ordinal} is outside 0..{MAX_CASE_ORDINAL_FORMAT}"
        )
    case_id = (
        f"{bank_version}|ms={PHASE11_MASTER_SEED}|st={stratum}"
        f"|src={setup_source}|c={case_ordinal:03d}"
    )
    if _CASE_ID_PATTERN.match(case_id) is None:
        raise Phase11SeedError(
            f"bank_version {bank_version!r} does not match the frozen Phase 11 "
            "bank naming rule 'phase11_<name>_bank_v<n>'"
        )
    return case_id


def parse_phase11_case_id(case_id: str) -> dict:
    """The identity fields of a Phase 11 case id, validated."""
    match = _CASE_ID_PATTERN.match(case_id)
    if match is None:
        raise Phase11SeedError(f"malformed Phase 11 case id: {case_id!r}")
    fields = match.groupdict()
    if int(fields["master"]) != PHASE11_MASTER_SEED:
        raise Phase11SeedError(
            f"case id names master seed {fields['master']}, expected {PHASE11_MASTER_SEED}"
        )
    return {
        "bank_version": fields["bank"],
        "phase11_master_seed": int(fields["master"]),
        "stratum": fields["stratum"],
        "setup_source": fields["source"],
        "case_ordinal": int(fields["ordinal"]),
    }


def phase11_game_id(case_id: str, game_index: int) -> str:
    """The stable identifier of one game of one case.

    Game 0 seats the observer as Red, game 1 as Blue — the frozen colour
    pairing — so the game id carries the colour assignment implicitly and
    exactly.
    """
    parse_phase11_case_id(case_id)
    if game_index not in CASE_GAME_OBSERVER_COLOR:
        raise Phase11SeedError(
            f"game_index must be one of {list(CASE_GAME_INDICES)}, got {game_index!r}"
        )
    return f"{case_id}|g={game_index}"


def parse_phase11_game_id(game_id: str) -> dict:
    """The identity fields of a Phase 11 game id, validated."""
    match = _GAME_ID_PATTERN.match(game_id)
    if match is None:
        raise Phase11SeedError(f"malformed Phase 11 game id: {game_id!r}")
    fields = parse_phase11_case_id(match.group("case"))
    game_index = int(match.group("game"))
    fields.update(
        {
            "case_id": match.group("case"),
            "game_index": game_index,
            "observer_color": CASE_GAME_OBSERVER_COLOR[game_index],
            "opponent_color": CASE_GAME_OPPONENT_COLOR[game_index],
        }
    )
    return fields


def case_setup_seed(case_id: str, game_index: int, role: str) -> int:
    """The frozen setup-draw seed of one seat of one game of one case.

    ``role`` picks the stream domain: the observer's draw always goes to
    the accepted P10-D production source, the opponent's to the case's
    setup-source stratum (P10-D or `neutral_v1`). The seat's colour is
    fixed by the frozen pairing (`observer` red in game 0, blue in game 1)
    and therefore already part of the identity through `game_index` —
    repeating it in the derivation could only ever disagree with the
    pairing table, so it is deliberately absent.
    """
    parse_phase11_case_id(case_id)
    if game_index not in CASE_GAME_OBSERVER_COLOR:
        raise Phase11SeedError(
            f"game_index must be one of {list(CASE_GAME_INDICES)}, got {game_index!r}"
        )
    if role == ROLE_OBSERVER:
        domain = DOMAIN_BANK_OBSERVER_SETUP
    elif role == ROLE_OPPONENT:
        domain = DOMAIN_BANK_OPPONENT_SETUP
    else:
        raise Phase11SeedError(
            f"role must be one of {list(SETUP_ROLES)}, got {role!r}"
        )
    return derive_phase11_seed(domain, case_id, int(game_index))


def game_match_seed(game_id: str) -> int:
    """The match-level randomness seed of one bank game.

    Independent of everything except the frozen game identity: rule-based
    opponents draw their per-decision randomness from the accepted runner
    derivations rooted here, and the neural seats play the accepted greedy
    decision mode and consume none of it.
    """
    fields = parse_phase11_game_id(game_id)
    return derive_phase11_seed(DOMAIN_BANK_MATCH, fields["case_id"], fields["game_index"])


# ---------------------------------------------------------------------------
# Prediction-event identity
# ---------------------------------------------------------------------------

_PREDICTION_ID_PATTERN = re.compile(
    r"^(?P<game>.+\|g=[01])\|d=(?P<decision>[0-9]{4})\|p=(?P<slot>[0-9]{2})$"
)

MAX_DECISION_INDEX_FORMAT = 9999
MAX_PIECE_SLOT = 39


def phase11_prediction_id(game_id: str, decision_index: int, piece_slot: int) -> str:
    """The stable identifier of one hidden-piece prediction event.

    `decision_index` is the pre-action `total_moves` of the observer
    decision that produced the forward; `piece_slot` is the opponent
    piece's public setup-slot index `0..39` (the slot component of the
    engine's stable public piece identifier). Deterministic, replayable,
    and free of worker/order/path information.
    """
    parse_phase11_game_id(game_id)
    if (
        not isinstance(decision_index, int)
        or isinstance(decision_index, bool)
        or not 0 <= decision_index <= MAX_DECISION_INDEX_FORMAT
    ):
        raise Phase11SeedError(
            f"decision_index must be an int in 0..{MAX_DECISION_INDEX_FORMAT}, "
            f"got {decision_index!r}"
        )
    if (
        not isinstance(piece_slot, int)
        or isinstance(piece_slot, bool)
        or not 0 <= piece_slot <= MAX_PIECE_SLOT
    ):
        raise Phase11SeedError(
            f"piece_slot must be an int in 0..{MAX_PIECE_SLOT}, got {piece_slot!r}"
        )
    return f"{game_id}|d={decision_index:04d}|p={piece_slot:02d}"


def parse_phase11_prediction_id(prediction_id: str) -> dict:
    """The identity fields of a prediction-event id, validated."""
    match = _PREDICTION_ID_PATTERN.match(prediction_id)
    if match is None:
        raise Phase11SeedError(f"malformed Phase 11 prediction id: {prediction_id!r}")
    fields = parse_phase11_game_id(match.group("game"))
    fields.update(
        {
            "game_id": match.group("game"),
            "decision_index": int(match.group("decision")),
            "piece_slot": int(match.group("slot")),
        }
    )
    return fields


# ---------------------------------------------------------------------------
# World-sample identity
# ---------------------------------------------------------------------------

PHASE11_SAMPLE_VERSION = "phase11_world_sample_v1"

#: The frozen model label a sample token carries. The label is bound to the
#: exact accepted digests (checkpoint SHA, model-state digest, belief-head
#: digest) by `phase11_contract`, so the token stays short while the
#: identity stays exact.
BELIEF_MODEL_LABEL = "selfplay_c1_v1"

_SAMPLE_TOKEN_PATTERN = re.compile(
    r"^phase11_world_sample_v1\|ms=(?P<master>[0-9]+)\|model=(?P<model>[a-z0-9_]+)"
    r"\|smp=(?P<sampler>[a-z0-9_]+)\|ps=(?P<state>[0-9a-f]{64})\|n=(?P<ordinal>[0-9]{5})$"
)

MAX_SAMPLE_ORDINAL_FORMAT = 99999


def phase11_sample_token(
    sampler_version: str, public_state_identity: str, sample_ordinal: int
) -> str:
    """The stable identifier of one complete-world sample.

    A sampled world is a pure function of exactly these inputs — the
    public-state identity, the (label-bound) belief-model identity, the
    sampler identity and the sample ordinal — as the common contract's
    reproducibility clause requires. Ordinals `0..63` are the production
    request's worlds; the audit spaces use the same format with larger
    ordinals.
    """
    if not isinstance(sampler_version, str) or not re.fullmatch(
        r"[a-z0-9_]+", sampler_version or ""
    ):
        raise Phase11SeedError(
            f"sampler_version must be a lowercase token, got {sampler_version!r}"
        )
    if not isinstance(public_state_identity, str) or not re.fullmatch(
        r"[0-9a-f]{64}", public_state_identity or ""
    ):
        raise Phase11SeedError(
            "public_state_identity must be a 64-hex-digit SHA-256, got "
            f"{public_state_identity!r}"
        )
    if (
        not isinstance(sample_ordinal, int)
        or isinstance(sample_ordinal, bool)
        or not 0 <= sample_ordinal <= MAX_SAMPLE_ORDINAL_FORMAT
    ):
        raise Phase11SeedError(
            f"sample ordinal must be an int in 0..{MAX_SAMPLE_ORDINAL_FORMAT}, "
            f"got {sample_ordinal!r}"
        )
    return (
        f"{PHASE11_SAMPLE_VERSION}|ms={PHASE11_MASTER_SEED}|model={BELIEF_MODEL_LABEL}"
        f"|smp={sampler_version}|ps={public_state_identity}|n={sample_ordinal:05d}"
    )


def parse_phase11_sample_token(sample_token: str) -> dict:
    """The identity fields of a world-sample token, validated."""
    match = _SAMPLE_TOKEN_PATTERN.match(sample_token)
    if match is None:
        raise Phase11SeedError(f"malformed Phase 11 sample token: {sample_token!r}")
    fields = match.groupdict()
    if int(fields["master"]) != PHASE11_MASTER_SEED:
        raise Phase11SeedError(
            f"sample token names master seed {fields['master']}, expected "
            f"{PHASE11_MASTER_SEED}"
        )
    if fields["model"] != BELIEF_MODEL_LABEL:
        raise Phase11SeedError(
            f"sample token names model {fields['model']!r}, expected "
            f"{BELIEF_MODEL_LABEL!r}"
        )
    return {
        "phase11_master_seed": int(fields["master"]),
        "belief_model_label": fields["model"],
        "sampler_version": fields["sampler"],
        "public_state_identity": fields["state"],
        "sample_ordinal": int(fields["ordinal"]),
    }


def world_sample_seed(sample_token: str) -> int:
    """The root stream seed of one complete-world sample."""
    parse_phase11_sample_token(sample_token)
    return derive_phase11_seed(DOMAIN_WORLD_SAMPLE, sample_token)


def world_order_key(sample_token: str, piece_slot: int) -> int:
    """The frozen ordering key of one unresolved piece inside one sample.

    The sample's deterministic random unresolved-piece order is ascending
    `(world_order_key, piece_slot)` over its unresolved pieces — the
    integer tie-break makes the order total even in the astronomically
    unlikely event of equal 63-bit keys.
    """
    parse_phase11_sample_token(sample_token)
    if (
        not isinstance(piece_slot, int)
        or isinstance(piece_slot, bool)
        or not 0 <= piece_slot <= MAX_PIECE_SLOT
    ):
        raise Phase11SeedError(
            f"piece_slot must be an int in 0..{MAX_PIECE_SLOT}, got {piece_slot!r}"
        )
    return derive_phase11_seed(DOMAIN_WORLD_ORDER, sample_token, int(piece_slot))


def world_categorical_uniform(sample_token: str, step_index: int) -> float:
    """The frozen `[0, 1)` uniform of one categorical draw inside one sample.

    `step_index` is the zero-based position in the sample's frozen piece
    order, so the draw of step `j` cannot move when an earlier step's
    outcome changes the remaining inventory.
    """
    parse_phase11_sample_token(sample_token)
    if (
        not isinstance(step_index, int)
        or isinstance(step_index, bool)
        or not 0 <= step_index <= MAX_PIECE_SLOT
    ):
        raise Phase11SeedError(
            f"step_index must be an int in 0..{MAX_PIECE_SLOT}, got {step_index!r}"
        )
    return unit_uniform(
        derive_phase11_seed(DOMAIN_WORLD_CATEGORICAL, sample_token, int(step_index))
    )


# ---------------------------------------------------------------------------
# Information-safety trial identity
# ---------------------------------------------------------------------------

PHASE11_SAFETY_TRIAL_VERSION = "phase11_safety_trial_v1"

_SAFETY_TRIAL_ID_PATTERN = re.compile(
    r"^phase11_safety_trial_v1\|ms=(?P<master>[0-9]+)\|n=(?P<ordinal>[0-9]{5})$"
)

#: Frozen trial volume — the common contract's minimum, exactly.
SAFETY_TRIAL_COUNT = 50_000

#: Frozen purposes of the safety-trial stream. `state_selection` picks the
#: trial's validation public state, `truth_permutation` drives the
#: alternative-hidden-truth construction, `sample_check` picks the fixed
#: sample ordinal whose world must be bit-identical.
SAFETY_PURPOSE_STATE = "state_selection"
SAFETY_PURPOSE_PERMUTATION = "truth_permutation"
SAFETY_PURPOSE_SAMPLE = "sample_check"
SAFETY_PURPOSES = (
    SAFETY_PURPOSE_STATE,
    SAFETY_PURPOSE_PERMUTATION,
    SAFETY_PURPOSE_SAMPLE,
)


def phase11_safety_trial_id(trial_ordinal: int) -> str:
    """The stable identifier of one hidden-truth permutation trial."""
    if (
        not isinstance(trial_ordinal, int)
        or isinstance(trial_ordinal, bool)
        or not 0 <= trial_ordinal < SAFETY_TRIAL_COUNT
    ):
        raise Phase11SeedError(
            f"trial ordinal must be an int in 0..{SAFETY_TRIAL_COUNT - 1}, "
            f"got {trial_ordinal!r}"
        )
    return f"{PHASE11_SAFETY_TRIAL_VERSION}|ms={PHASE11_MASTER_SEED}|n={trial_ordinal:05d}"


def parse_phase11_safety_trial_id(trial_id: str) -> dict:
    """The identity fields of a safety-trial id, validated."""
    match = _SAFETY_TRIAL_ID_PATTERN.match(trial_id)
    if match is None:
        raise Phase11SeedError(f"malformed Phase 11 safety-trial id: {trial_id!r}")
    fields = match.groupdict()
    if int(fields["master"]) != PHASE11_MASTER_SEED:
        raise Phase11SeedError(
            f"trial id names master seed {fields['master']}, expected "
            f"{PHASE11_MASTER_SEED}"
        )
    return {
        "phase11_master_seed": int(fields["master"]),
        "trial_ordinal": int(fields["ordinal"]),
    }


def safety_trial_seed(trial_id: str, purpose: str, draw_ordinal: int) -> int:
    """One stream seed inside one safety trial, by frozen purpose."""
    parse_phase11_safety_trial_id(trial_id)
    if purpose not in SAFETY_PURPOSES:
        raise Phase11SeedError(
            f"purpose must be one of {list(SAFETY_PURPOSES)}, got {purpose!r}"
        )
    if not isinstance(draw_ordinal, int) or isinstance(draw_ordinal, bool) or draw_ordinal < 0:
        raise Phase11SeedError(
            f"draw ordinal must be a non-negative int, got {draw_ordinal!r}"
        )
    return derive_phase11_seed(DOMAIN_SAFETY_TRIAL, trial_id, purpose, int(draw_ordinal))


# ---------------------------------------------------------------------------
# Reproducibility-request and benchmark-state identity
# ---------------------------------------------------------------------------

PHASE11_REPRO_REQUEST_VERSION = "phase11_repro_request_v1"
PHASE11_BENCHMARK_STATE_VERSION = "phase11_benchmark_state_v1"

#: Frozen topology/replay request volume: the deterministic request set that
#: must reproduce bit-identically under every required topology.
REPRO_REQUEST_COUNT = 2_048

#: Frozen benchmark schedule: states per (stratum x colour x progress
#: bucket) cell, over 8 x 2 x 3 = 48 cells.
BENCHMARK_STATES_PER_CELL = 10
BENCHMARK_CELL_COUNT = 48
BENCHMARK_STATE_COUNT = BENCHMARK_STATES_PER_CELL * BENCHMARK_CELL_COUNT

_REPRO_REQUEST_ID_PATTERN = re.compile(
    r"^phase11_repro_request_v1\|ms=(?P<master>[0-9]+)\|n=(?P<ordinal>[0-9]{5})$"
)
_BENCHMARK_STATE_ID_PATTERN = re.compile(
    r"^phase11_benchmark_state_v1\|ms=(?P<master>[0-9]+)\|n=(?P<ordinal>[0-9]{3})$"
)


def phase11_repro_request_id(request_ordinal: int) -> str:
    """The stable identifier of one topology/replay request."""
    if (
        not isinstance(request_ordinal, int)
        or isinstance(request_ordinal, bool)
        or not 0 <= request_ordinal < REPRO_REQUEST_COUNT
    ):
        raise Phase11SeedError(
            f"request ordinal must be an int in 0..{REPRO_REQUEST_COUNT - 1}, "
            f"got {request_ordinal!r}"
        )
    return (
        f"{PHASE11_REPRO_REQUEST_VERSION}|ms={PHASE11_MASTER_SEED}"
        f"|n={request_ordinal:05d}"
    )


def parse_phase11_repro_request_id(request_id: str) -> dict:
    """The identity fields of a reproducibility-request id, validated."""
    match = _REPRO_REQUEST_ID_PATTERN.match(request_id)
    if match is None:
        raise Phase11SeedError(f"malformed Phase 11 repro request id: {request_id!r}")
    fields = match.groupdict()
    if int(fields["master"]) != PHASE11_MASTER_SEED:
        raise Phase11SeedError(
            f"request id names master seed {fields['master']}, expected "
            f"{PHASE11_MASTER_SEED}"
        )
    return {
        "phase11_master_seed": int(fields["master"]),
        "request_ordinal": int(fields["ordinal"]),
    }


def phase11_benchmark_state_id(state_ordinal: int) -> str:
    """The stable identifier of one runtime-benchmark state slot."""
    if (
        not isinstance(state_ordinal, int)
        or isinstance(state_ordinal, bool)
        or not 0 <= state_ordinal < BENCHMARK_STATE_COUNT
    ):
        raise Phase11SeedError(
            f"state ordinal must be an int in 0..{BENCHMARK_STATE_COUNT - 1}, "
            f"got {state_ordinal!r}"
        )
    return (
        f"{PHASE11_BENCHMARK_STATE_VERSION}|ms={PHASE11_MASTER_SEED}"
        f"|n={state_ordinal:03d}"
    )


def repro_schedule_seed(purpose: str, ordinal: int) -> int:
    """One stream seed of the reproducibility/benchmark schedule domain.

    The frozen selection rules are hash-order deterministic and consume no
    randomness; this stream exists so any schedule step that *does* need a
    draw (and any future predeclared one) has a frozen, domain-separated
    source instead of an invented one.
    """
    if not isinstance(purpose, str) or not purpose or ":" in purpose:
        raise Phase11SeedError(f"purpose must be a colon-free non-empty string, got {purpose!r}")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
        raise Phase11SeedError(f"ordinal must be a non-negative int, got {ordinal!r}")
    return derive_phase11_seed(DOMAIN_REPRO_SCHEDULE, purpose, int(ordinal))


def benchmark_seed(purpose: str, ordinal: int) -> int:
    """One stream seed of the runtime-benchmark domain (same contract)."""
    if not isinstance(purpose, str) or not purpose or ":" in purpose:
        raise Phase11SeedError(f"purpose must be a colon-free non-empty string, got {purpose!r}")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
        raise Phase11SeedError(f"ordinal must be a non-negative int, got {ordinal!r}")
    return derive_phase11_seed(DOMAIN_BENCHMARK, purpose, int(ordinal))


# ---------------------------------------------------------------------------
# Bootstrap streams
# ---------------------------------------------------------------------------


def bootstrap_root(bank: str) -> int:
    """The frozen bootstrap root seed of one evaluation bank."""
    if bank == "validation":
        return VALIDATION_BOOTSTRAP_SEED
    if bank == "test":
        return TEST_BOOTSTRAP_SEED
    raise Phase11SeedError(
        f"bootstrap roots exist for the two evaluation banks only, not {bank!r}"
    )


def bootstrap_stream_seed(bank: str, metric_token: str) -> int:
    """The resampling stream of one metric (or one stratum slice) on one bank.

    Every bootstrapped quantity receives its own domain-separated token
    (overall metrics use the frozen metric tokens; stratum slices append
    `|st=<stratum>`), so two intervals are never resampled from the same
    stream and any interval can be recomputed in isolation from the
    primitive recorded rows.
    """
    root = bootstrap_root(bank)
    if not isinstance(metric_token, str) or not metric_token or ":" in metric_token:
        raise Phase11SeedError(
            f"metric_token must be a colon-free non-empty string, got {metric_token!r}"
        )
    return derive_phase11_seed(DOMAIN_BOOTSTRAP, root, bank, metric_token)


# ---------------------------------------------------------------------------
# Agent 6 soak identity — frozen now, closing the Phase 10 deviation
# ---------------------------------------------------------------------------

PHASE11_SOAK_VERSION = "phase11_soak_v1"

#: Frozen soak volume: 128 games per opponent stratum x 8 strata = 1,024
#: train-split games, each contributing exactly 8 scheduled production
#: belief requests = 8,192 requests, the common contract's minimum exactly.
SOAK_GAMES_PER_STRATUM = 128
SOAK_GAME_COUNT = SOAK_GAMES_PER_STRATUM * len(OPPONENT_STRATA)
SOAK_REQUESTS_PER_GAME = 8
SOAK_REQUEST_COUNT = SOAK_GAME_COUNT * SOAK_REQUESTS_PER_GAME
assert SOAK_REQUEST_COUNT == 8_192

_SOAK_GAME_ID_PATTERN = re.compile(
    rf"^phase11_soak_v1\|ms=(?P<master>[0-9]+)\|st=(?P<stratum>{_STRATUM_ALTERNATION})"
    rf"\|g=(?P<ordinal>[0-9]{{3}})$"
)
_SOAK_REQUEST_ID_PATTERN = re.compile(
    r"^(?P<game>.+\|g=[0-9]{3})\|r=(?P<request>[0-7])$"
)


def phase11_soak_game_id(stratum: str, game_ordinal: int) -> str:
    """The stable identifier of one soak game.

    The observer plays Red on even ordinals and Blue on odd ordinals — a
    frozen parity rule, so both colours are covered without a draw.
    """
    if stratum not in OPPONENT_STRATA:
        raise Phase11SeedError(
            f"stratum must be one of {list(OPPONENT_STRATA)}, got {stratum!r}"
        )
    if (
        not isinstance(game_ordinal, int)
        or isinstance(game_ordinal, bool)
        or not 0 <= game_ordinal < SOAK_GAMES_PER_STRATUM
    ):
        raise Phase11SeedError(
            f"soak game ordinal must be an int in 0..{SOAK_GAMES_PER_STRATUM - 1}, "
            f"got {game_ordinal!r}"
        )
    return (
        f"{PHASE11_SOAK_VERSION}|ms={PHASE11_MASTER_SEED}|st={stratum}"
        f"|g={game_ordinal:03d}"
    )


def parse_phase11_soak_game_id(game_id: str) -> dict:
    """The identity fields of a soak game id, validated."""
    match = _SOAK_GAME_ID_PATTERN.match(game_id)
    if match is None:
        raise Phase11SeedError(f"malformed Phase 11 soak game id: {game_id!r}")
    fields = match.groupdict()
    if int(fields["master"]) != PHASE11_MASTER_SEED:
        raise Phase11SeedError(
            f"soak game id names master seed {fields['master']}, expected "
            f"{PHASE11_MASTER_SEED}"
        )
    ordinal = int(fields["ordinal"])
    if ordinal >= SOAK_GAMES_PER_STRATUM:
        raise Phase11SeedError(
            f"soak game ordinal {ordinal} is outside 0..{SOAK_GAMES_PER_STRATUM - 1}"
        )
    return {
        "phase11_master_seed": int(fields["master"]),
        "stratum": fields["stratum"],
        "game_ordinal": ordinal,
        "observer_color": COLOR_RED if ordinal % 2 == 0 else COLOR_BLUE,
        "opponent_color": COLOR_BLUE if ordinal % 2 == 0 else COLOR_RED,
    }


def phase11_soak_request_id(soak_game_id: str, request_ordinal: int) -> str:
    """The stable identifier of one soak production belief request."""
    parse_phase11_soak_game_id(soak_game_id)
    if (
        not isinstance(request_ordinal, int)
        or isinstance(request_ordinal, bool)
        or not 0 <= request_ordinal < SOAK_REQUESTS_PER_GAME
    ):
        raise Phase11SeedError(
            f"soak request ordinal must be an int in 0..{SOAK_REQUESTS_PER_GAME - 1}, "
            f"got {request_ordinal!r}"
        )
    return f"{soak_game_id}|r={request_ordinal}"


def parse_phase11_soak_request_id(request_id: str) -> dict:
    """The identity fields of a soak request id, validated."""
    match = _SOAK_REQUEST_ID_PATTERN.match(request_id)
    if match is None:
        raise Phase11SeedError(f"malformed Phase 11 soak request id: {request_id!r}")
    fields = parse_phase11_soak_game_id(match.group("game"))
    fields.update(
        {
            "soak_game_id": match.group("game"),
            "request_ordinal": int(match.group("request")),
        }
    )
    return fields


def soak_setup_seed(soak_game_id: str, role: str) -> int:
    """The setup-draw seed of one seat of one soak game.

    Both soak seats draw from the accepted P10-D production source — the
    production-shaped exercise the soak exists for — with the seat colour
    fixed by the frozen ordinal-parity rule.
    """
    parse_phase11_soak_game_id(soak_game_id)
    if role not in SETUP_ROLES:
        raise Phase11SeedError(f"role must be one of {list(SETUP_ROLES)}, got {role!r}")
    return derive_phase11_seed(DOMAIN_SOAK_SETUP, soak_game_id, role)


def soak_match_seed(soak_game_id: str) -> int:
    """The match-level randomness seed of one soak game."""
    parse_phase11_soak_game_id(soak_game_id)
    return derive_phase11_seed(DOMAIN_SOAK_MATCH, soak_game_id)


# ---------------------------------------------------------------------------
# Collision audit and the derivation record
# ---------------------------------------------------------------------------


def stream_collision_audit(stream_seeds: "dict[str, list[int]]") -> dict:
    """Duplicate/collision findings over a set of named seed streams.

    Used by the Agent 1 harness to prove exhaustively that no two logical
    identities in the currently enumerable Phase 11 id space share a stream
    seed, within a stream and across streams alike. The Phase 10 audit
    routine, re-frozen verbatim for Phase 11.
    """
    findings: list[str] = []
    per_stream = {}
    combined: dict[int, str] = {}
    for name, seeds in sorted(stream_seeds.items()):
        materialized = list(seeds)
        distinct = len(set(materialized))
        per_stream[name] = {"count": len(materialized), "distinct": distinct}
        if distinct != len(materialized):
            findings.append(
                f"{name}: {len(materialized) - distinct} duplicate seeds inside the stream"
            )
        for seed in materialized:
            owner = combined.get(seed)
            if owner is not None and owner != name:
                findings.append(f"{name} collides with {owner} on seed {seed}")
            combined[seed] = name
    return {
        "streams": per_stream,
        "total_seeds": sum(entry["count"] for entry in per_stream.values()),
        "distinct_seeds": len(combined),
        "findings": findings,
        "no_collisions": not findings,
    }


#: The identity parts of every domain, published in the contract so an
#: auditor can re-derive any stream without reading this module's source.
_DOMAIN_PARTS = {
    DOMAIN_BANK_OBSERVER_SETUP: ["case_id", "game_index"],
    DOMAIN_BANK_OPPONENT_SETUP: ["case_id", "game_index"],
    DOMAIN_BANK_MATCH: ["case_id", "game_index"],
    DOMAIN_WORLD_SAMPLE: ["sample_token"],
    DOMAIN_WORLD_ORDER: ["sample_token", "piece_slot"],
    DOMAIN_WORLD_CATEGORICAL: ["sample_token", "step_index"],
    DOMAIN_SAFETY_TRIAL: ["trial_id", "purpose", "draw_ordinal"],
    DOMAIN_REPRO_SCHEDULE: ["purpose", "ordinal"],
    DOMAIN_BENCHMARK: ["purpose", "ordinal"],
    DOMAIN_BOOTSTRAP: ["bootstrap_root", "bank", "metric_token"],
    DOMAIN_SOAK_SETUP: ["soak_game_id", "role"],
    DOMAIN_SOAK_MATCH: ["soak_game_id"],
}
assert set(_DOMAIN_PARTS) == set(STREAM_DOMAINS)


def seed_derivation_document() -> dict:
    """The machine-readable derivation record for the Agent 1 contract."""
    return {
        "identity_version": PHASE11_IDENTITY_VERSION,
        "root_seeds": dict(CANONICAL_PHASE11_SEEDS),
        "personalization": _PHASE11_SEED_PERSON.decode(),
        "derivation": (
            "blake2b(person='strat-b11', digest_size=8) over "
            "'phase11_identity_v1:domain:domain_root:part:part:...', "
            "big-endian, right-shifted one bit; string parts may not contain "
            "':' so the payload encoding is injective"
        ),
        "domains": {
            domain: {
                "root_seed": DOMAIN_ROOTS[domain],
                "identity_parts": _DOMAIN_PARTS[domain],
            }
            for domain in STREAM_DOMAINS
        },
        "unit_uniform": (
            "seed / 2**63, an exact binary division, giving a [0, 1) uniform "
            "reproducible on every platform"
        ),
        "case_id_format": (
            "<bank_version>|ms=<master>|st=<opponent stratum>|src=<setup source>"
            "|c=<ordinal:03d>"
        ),
        "game_id_format": "<case_id>|g=<0|1>; observer red in game 0, blue in game 1",
        "prediction_id_format": (
            "<game_id>|d=<pre-action total_moves:04d>|p=<opponent piece slot:02d>"
        ),
        "sample_token_format": (
            "phase11_world_sample_v1|ms=<master>|model=selfplay_c1_v1"
            "|smp=<sampler version>|ps=<public-state sha256>|n=<ordinal:05d>"
        ),
        "safety_trial_id_format": "phase11_safety_trial_v1|ms=<master>|n=<ordinal:05d>",
        "repro_request_id_format": "phase11_repro_request_v1|ms=<master>|n=<ordinal:05d>",
        "benchmark_state_id_format": (
            "phase11_benchmark_state_v1|ms=<master>|n=<ordinal:03d>"
        ),
        "soak_game_id_format": (
            "phase11_soak_v1|ms=<master>|st=<stratum>|g=<ordinal:03d>; observer "
            "red on even ordinals, blue on odd"
        ),
        "soak_request_id_format": "<soak_game_id>|r=<0..7>",
        "independence": (
            "no derivation reads worker count, task arrival order, process id, "
            "wall clock, or a physical storage path, so any bank case, world "
            "sample, safety trial or soak request rebuilds alone and re-sharding "
            "cannot change a single draw"
        ),
        "root_reading_notes": [
            "the common contract froze no dedicated Agent 6 soak root, so the "
            "soak's setup/match streams hang off the bank-schedule and "
            "match-randomness roots under distinct domain tokens, frozen now "
            "(closing the Phase 10 soak-namespace deviation in advance)",
            "both banks' streams hang off the single bank/case-schedule root, "
            "domain-separated by the two distinct bank-version tokens inside "
            "every case id (the accepted Phase 10 reading, reused)",
        ],
        "downstream_collision_obligation": (
            "the world-sample id space is keyed by public-state identities that "
            "do not exist until games are played; every agent that realizes part "
            "of that space (Agents 3, 4, 6, 7) must run stream_collision_audit "
            "over every seed it actually derived and report zero collisions"
        ),
    }


__all__ = [
    "BANK_SCHEDULE_SEED",
    "BELIEF_MODEL_LABEL",
    "BENCHMARK_CELL_COUNT",
    "BENCHMARK_STATES_PER_CELL",
    "BENCHMARK_STATE_COUNT",
    "CANONICAL_PHASE11_SEEDS",
    "CASE_GAME_INDICES",
    "CASE_GAME_OBSERVER_COLOR",
    "CASE_GAME_OPPONENT_COLOR",
    "COLORS",
    "COLOR_BLUE",
    "COLOR_RED",
    "DOMAIN_BANK_MATCH",
    "DOMAIN_BANK_OBSERVER_SETUP",
    "DOMAIN_BANK_OPPONENT_SETUP",
    "DOMAIN_BENCHMARK",
    "DOMAIN_BOOTSTRAP",
    "DOMAIN_REPRO_SCHEDULE",
    "DOMAIN_ROOTS",
    "DOMAIN_SAFETY_TRIAL",
    "DOMAIN_SOAK_MATCH",
    "DOMAIN_SOAK_SETUP",
    "DOMAIN_WORLD_CATEGORICAL",
    "DOMAIN_WORLD_ORDER",
    "DOMAIN_WORLD_SAMPLE",
    "INFORMATION_SAFETY_SEED",
    "MATCH_RANDOMNESS_SEED",
    "MAX_CASE_ORDINAL_FORMAT",
    "MAX_DECISION_INDEX_FORMAT",
    "MAX_PIECE_SLOT",
    "MAX_SAMPLE_ORDINAL_FORMAT",
    "OPPONENT_STRATA",
    "PHASE11_BENCHMARK_STATE_VERSION",
    "PHASE11_IDENTITY_VERSION",
    "PHASE11_MASTER_SEED",
    "PHASE11_REPRO_REQUEST_VERSION",
    "PHASE11_SAFETY_TRIAL_VERSION",
    "PHASE11_SAMPLE_VERSION",
    "PHASE11_SOAK_VERSION",
    "REPRO_REQUEST_COUNT",
    "REPRO_RUNTIME_SEED",
    "ROLE_OBSERVER",
    "ROLE_OPPONENT",
    "SAFETY_PURPOSES",
    "SAFETY_PURPOSE_PERMUTATION",
    "SAFETY_PURPOSE_SAMPLE",
    "SAFETY_PURPOSE_STATE",
    "SAFETY_TRIAL_COUNT",
    "SETUP_ROLES",
    "SETUP_SOURCES",
    "SOAK_GAMES_PER_STRATUM",
    "SOAK_GAME_COUNT",
    "SOAK_REQUESTS_PER_GAME",
    "SOAK_REQUEST_COUNT",
    "SOURCE_NEUTRAL",
    "SOURCE_P10D",
    "STRATUM_BASIC",
    "STRATUM_INFORMATION_MISER",
    "STRATUM_MINER_RUSH",
    "STRATUM_PHASE8_ANCHOR",
    "STRATUM_PHASE9",
    "STRATUM_SCOUT_RUSH",
    "STRATUM_STRATEGIC",
    "STRATUM_TACTICAL",
    "STREAM_DOMAINS",
    "TEST_BOOTSTRAP_SEED",
    "VALIDATION_BOOTSTRAP_SEED",
    "WORLD_SAMPLING_SEED",
    "Phase11SeedError",
    "benchmark_seed",
    "bootstrap_root",
    "bootstrap_stream_seed",
    "case_setup_seed",
    "derive_phase11_seed",
    "game_match_seed",
    "parse_phase11_case_id",
    "parse_phase11_game_id",
    "parse_phase11_prediction_id",
    "parse_phase11_safety_trial_id",
    "parse_phase11_sample_token",
    "parse_phase11_soak_game_id",
    "parse_phase11_soak_request_id",
    "parse_phase11_repro_request_id",
    "phase11_benchmark_state_id",
    "phase11_case_id",
    "phase11_game_id",
    "phase11_prediction_id",
    "phase11_repro_request_id",
    "phase11_safety_trial_id",
    "phase11_sample_token",
    "phase11_soak_game_id",
    "phase11_soak_request_id",
    "repro_schedule_seed",
    "safety_trial_seed",
    "seed_derivation_document",
    "soak_match_seed",
    "soak_setup_seed",
    "stream_collision_audit",
    "unit_uniform",
    "world_categorical_uniform",
    "world_order_key",
    "world_sample_seed",
]
