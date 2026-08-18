"""Phase 10 Agent 1: frozen setup-selection seeds and logical identities.

Specification sources:

- `01_AGENT_1_CONTRACT_SEEDS_BANKS_ACCEPTANCE.md` ("Freeze seeds and
  derivations")
- `00_PHASE_10_SEQUENCE_AND_COMMON_CONTRACT.md` ("Phase 10 root seeds",
  "Controlled setup-outcome corpus", "Selector semantics", "Phase 10
  evaluation banks", "Statistics")

What lives here and why
-----------------------
Everything in this module is *identity*: the eight frozen Phase 10 root
seeds, the outcome-game identifier, the evaluation-case identifier, and the
domain-separated stream seeds every Phase 10 consumer draws from. Nothing
here knows what a utility model is, how many games a family pair gets, or
what an acceptance gate says — :mod:`stratego.training.phase10_contract`
layers the frozen experiment on top of these identities and is the intended
importer, exactly as `phase9_seed` sat under `phase9_contract`.

Keeping identity free of contract knowledge is what makes a seed unable to
depend on a measurement. No derivation reads worker count, task arrival
order, process id, wall clock, or a physical storage path, so a corpus game
or a bank case can always be rebuilt alone.

Seeds were chosen before any Phase 10 outcome game was played and before
either utility model was fit. They follow the repository's date-seed
precedent (`20260101` Phase 4 bank, `20260813` Phase 7 library, `20260816xx`
Phase 9): the Phase 10 block is the freeze date `20260818` extended with a
two-digit role suffix, fixed by the calendar rather than by anything
measured.

Derivation
----------
All streams come from :func:`derive_phase10_seed`, a domain-separated
``blake2b`` hash under the Phase 10 personalization tag ``strat-s10`` —
distinct from every earlier tag (``strat-rl9`` Phase 9, ``strat-ws8``
Phase 8, ``strat-lb7``/``strat-at7``/``strat-st7`` Phase 7,
``strat-bnk``/``strat-sid`` Phase 4 bank, ``strat-pls``/``strat-dec``
match/decision seeds, ``strat-unt``/``strat-mch`` match identity), so no
Phase 10 stream can collide with any accepted upstream stream.

Root assignment
---------------
The common contract names eight roots. Their frozen consumers are:

```text
2026081801 master            folded into every Phase 10 logical id
2026081802 outcome schedule  corpus match seeds
2026081803 setup draws       corpus side draws and bank opponent-setup draws
2026081804 utility fitting   deterministic fit initialisation
2026081805 selector draws    bank selector seeds and production selector draws
2026081806 validation cases  evaluation-case schedule (both banks)
2026081807 validation bootstrap  validation-bank resampling streams
2026081808 final-test bootstrap  test-bank resampling streams
```

Two of those bindings are readings the common contract does not spell out
letter by letter, and both are recorded as such in the Agent 1 acceptance
artifact:

- the contract lists a *validation* case root but no separate test-case
  root, so :data:`CASE_SCHEDULE_SEED` roots the case schedule of **both**
  banks, domain-separated by the two distinct bank versions;
- bank opponent-setup draws are Phase 10 setup draws and therefore hang off
  :data:`SETUP_DRAW_SEED`, while bank selector seeds are selector draws and
  hang off :data:`SELECTOR_DRAW_SEED`.

Neither reading adds a root, reuses a stream, or lets any two derivations
collide; :func:`stream_collision_audit` proves the second part exhaustively
over the frozen id space.
"""

from __future__ import annotations

import hashlib
import re

#: The Phase 10 logical-identity version. A change to an id format, to the
#: seed derivation, or to the domain tokens is a new version, never a silent
#: edit.
PHASE10_IDENTITY_VERSION = "phase10_identity_v1"

#: The outcome-corpus game-id version, carried inside every corpus game id.
PHASE10_OUTCOME_VERSION = "phase10_outcome_v1"

# ---------------------------------------------------------------------------
# Canonical Phase 10 root seeds — frozen by the common contract before any
# Phase 10 outcome game existed and before either utility model was fit.
# ---------------------------------------------------------------------------

#: Master seed of the whole phase. Folded into every Phase 10 logical id.
PHASE10_MASTER_SEED = 2026081801

#: Root of the outcome-corpus schedule streams (per-game match seeds). The
#: schedule's *counts* are pure arithmetic — 256 ordered family pairs times
#: 64 games — so this root never chooses how many games exist, only how the
#: games that the arithmetic already fixed are seeded.
OUTCOME_SCHEDULE_SEED = 2026081802

#: Root of every Phase 10 setup draw: the corpus's two per-game side draws
#: and the evaluation banks' fixed opponent-setup draws.
SETUP_DRAW_SEED = 2026081803

#: Root of deterministic utility-fit initialisation. The frozen protocol
#: starts L-BFGS from an all-zero parameter vector, so this root is frozen
#: and unconsumed in v1; a future protocol needing a draw must derive it
#: from here under a new utility version.
UTILITY_FIT_SEED = 2026081804

#: Root of every selector draw: the evaluation banks' per-case selector
#: seeds and the production selector's branch/base streams.
SELECTOR_DRAW_SEED = 2026081805

#: Root of the evaluation-case schedule of *both* banks (see the module
#: docstring). Consumed for per-case match seeds.
CASE_SCHEDULE_SEED = 2026081806

#: Frozen paired-bootstrap roots of the two evaluation banks.
VALIDATION_BOOTSTRAP_SEED = 2026081807
TEST_BOOTSTRAP_SEED = 2026081808

CANONICAL_PHASE10_SEEDS = {
    "phase10_master_seed": PHASE10_MASTER_SEED,
    "outcome_schedule_seed": OUTCOME_SCHEDULE_SEED,
    "setup_draw_seed": SETUP_DRAW_SEED,
    "utility_fit_seed": UTILITY_FIT_SEED,
    "selector_draw_seed": SELECTOR_DRAW_SEED,
    "case_schedule_seed": CASE_SCHEDULE_SEED,
    "validation_bootstrap_seed": VALIDATION_BOOTSTRAP_SEED,
    "test_bootstrap_seed": TEST_BOOTSTRAP_SEED,
}

# ---------------------------------------------------------------------------
# Stream domains
# ---------------------------------------------------------------------------

#: blake2b personalization of every Phase 10 stream.
_PHASE10_SEED_PERSON = b"strat-s10"

DOMAIN_CORPUS_SETUP = "corpus_setup"
DOMAIN_CORPUS_MATCH = "corpus_match"
DOMAIN_BANK_OPPONENT = "bank_opponent"
DOMAIN_BANK_SELECTOR = "bank_selector"
DOMAIN_BANK_MATCH = "bank_match"
DOMAIN_SELECTOR_BRANCH = "selector_branch"
DOMAIN_SELECTOR_BASE = "selector_base"
DOMAIN_UTILITY_FIT = "utility_fit"
DOMAIN_BOOTSTRAP = "bootstrap"

STREAM_DOMAINS = (
    DOMAIN_CORPUS_SETUP,
    DOMAIN_CORPUS_MATCH,
    DOMAIN_BANK_OPPONENT,
    DOMAIN_BANK_SELECTOR,
    DOMAIN_BANK_MATCH,
    DOMAIN_SELECTOR_BRANCH,
    DOMAIN_SELECTOR_BASE,
    DOMAIN_UTILITY_FIT,
    DOMAIN_BOOTSTRAP,
)

#: The root seed each domain hangs off, frozen here so the binding is data
#: rather than a convention repeated at every call site.
DOMAIN_ROOTS = {
    DOMAIN_CORPUS_SETUP: SETUP_DRAW_SEED,
    DOMAIN_CORPUS_MATCH: OUTCOME_SCHEDULE_SEED,
    DOMAIN_BANK_OPPONENT: SETUP_DRAW_SEED,
    DOMAIN_BANK_SELECTOR: SELECTOR_DRAW_SEED,
    DOMAIN_BANK_MATCH: CASE_SCHEDULE_SEED,
    DOMAIN_SELECTOR_BRANCH: SELECTOR_DRAW_SEED,
    DOMAIN_SELECTOR_BASE: SELECTOR_DRAW_SEED,
    DOMAIN_UTILITY_FIT: UTILITY_FIT_SEED,
    DOMAIN_BOOTSTRAP: PHASE10_MASTER_SEED,
}
assert set(DOMAIN_ROOTS) == set(STREAM_DOMAINS)

#: The two colours a selector can be asked to play. Frozen as text because
#: the selector's own colour is one of its six legal inputs and must appear
#: in stream identity verbatim, never as an engine integer whose meaning
#: could drift.
COLOR_RED = "red"
COLOR_BLUE = "blue"
COLORS = (COLOR_RED, COLOR_BLUE)

#: Frozen colour pairing of an evaluation case: the selector under test
#: plays Red in game 0 and Blue in game 1 against the same fixed opponent
#: setup. Deterministic, so a case's per-colour arms are fixed by identity.
CASE_GAME_INDICES = (0, 1)
CASE_GAME_COLOR = {0: COLOR_RED, 1: COLOR_BLUE}
assert tuple(sorted(CASE_GAME_COLOR)) == CASE_GAME_INDICES


class Phase10SeedError(ValueError):
    """Raised when a Phase 10 identity or seed request is malformed."""


def derive_phase10_seed(domain: str, *parts: "int | str") -> int:
    """A 63-bit deterministic seed for one Phase 10 stream.

    ``domain`` must be one of :data:`STREAM_DOMAINS`; ``parts`` are the
    identity inputs of the stream. The payload is the colon-joined text of
    the identity version, the domain, the domain's frozen root seed and the
    parts, under the ``strat-s10`` personalization — so equal identities
    always agree, any change to any identity input yields an unrelated
    stream, and two domains sharing a root still cannot collide.
    """
    if domain not in STREAM_DOMAINS:
        raise Phase10SeedError(f"unknown Phase 10 stream domain: {domain!r}")
    for part in parts:
        if not isinstance(part, (int, str)) or isinstance(part, bool):
            raise Phase10SeedError(
                f"stream identity parts must be int or str, got {type(part).__name__}"
            )
    payload = ":".join(
        [
            PHASE10_IDENTITY_VERSION,
            domain,
            str(DOMAIN_ROOTS[domain]),
            *[str(part) for part in parts],
        ]
    )
    digest = hashlib.blake2b(
        payload.encode(), digest_size=8, person=_PHASE10_SEED_PERSON
    ).digest()
    return int.from_bytes(digest, "big") >> 1


# ---------------------------------------------------------------------------
# Outcome-corpus game identity
# ---------------------------------------------------------------------------

_FAMILY_TOKEN = r"F[0-9]{2}"
_OUTCOME_ID_PATTERN = re.compile(
    rf"^(?P<version>[a-z0-9_]+)\|ms=(?P<master>[0-9]+)"
    rf"\|rf=(?P<red_family>{_FAMILY_TOKEN})\|bf=(?P<blue_family>{_FAMILY_TOKEN})"
    rf"\|g=(?P<ordinal>[0-9]{{2}})$"
)

#: The corpus schedule gives every ordered family pair the same number of
#: games; the identifier only has to carry the ordinal's frozen width.
MAX_CORPUS_ORDINAL_FORMAT = 99


def phase10_game_id(red_family: str, blue_family: str, ordinal: int) -> str:
    """The stable identifier of one logical Phase 10 outcome-corpus game.

    A pure function of exactly the identity fields the common contract
    requires — outcome version, frozen master seed, the *ordered* Red and
    Blue setup families, and the per-pair game ordinal — in a fixed,
    parseable ``key=value`` pipe format:

    ```text
    phase10_outcome_v1|ms=2026081801|rf=F03|bf=F11|g=07
    ```

    Ordering matters: `(F03, F11)` and `(F11, F03)` are two of the 256
    scheduled pairs, not one. Worker count, arrival order and resume
    boundaries appear nowhere, which is what makes crash-regeneration of a
    single missing game exact. How many ordinals a pair has is the
    contract's to enforce; the identifier accepts any ordinal its format can
    carry so that check stays in one place.
    """
    for name, family_id in (("red_family", red_family), ("blue_family", blue_family)):
        if not isinstance(family_id, str) or re.fullmatch(_FAMILY_TOKEN, family_id) is None:
            raise Phase10SeedError(f"{name} must look like 'F00'..'F15', got {family_id!r}")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise Phase10SeedError(f"game ordinal must be an int, got {type(ordinal).__name__}")
    if not 0 <= ordinal <= MAX_CORPUS_ORDINAL_FORMAT:
        raise Phase10SeedError(
            f"game ordinal {ordinal} is outside 0..{MAX_CORPUS_ORDINAL_FORMAT}"
        )
    return (
        f"{PHASE10_OUTCOME_VERSION}|ms={PHASE10_MASTER_SEED}"
        f"|rf={red_family}|bf={blue_family}|g={ordinal:02d}"
    )


def parse_phase10_game_id(game_id: str) -> dict:
    """The identity fields of a Phase 10 corpus game id, validated.

    Raises on anything that is not exactly a well-formed id of this outcome
    version under the frozen master seed, so a foreign or tampered
    identifier can never be mistaken for a Phase 10 corpus game.
    """
    match = _OUTCOME_ID_PATTERN.match(game_id)
    if match is None:
        raise Phase10SeedError(f"malformed Phase 10 outcome game id: {game_id!r}")
    fields = match.groupdict()
    if fields["version"] != PHASE10_OUTCOME_VERSION:
        raise Phase10SeedError(
            f"game id names outcome version {fields['version']!r}, expected "
            f"{PHASE10_OUTCOME_VERSION!r}"
        )
    if int(fields["master"]) != PHASE10_MASTER_SEED:
        raise Phase10SeedError(
            f"game id names master seed {fields['master']}, expected {PHASE10_MASTER_SEED}"
        )
    return {
        "outcome_version": fields["version"],
        "phase10_master_seed": int(fields["master"]),
        "red_family": fields["red_family"],
        "blue_family": fields["blue_family"],
        "ordinal": int(fields["ordinal"]),
    }


def corpus_setup_seed(game_id: str, color: str, attempt: int) -> int:
    """The sampler draw seed of one corpus side's family-rejection attempt.

    The corpus conditions each side on a scheduled family by walking
    ``attempt = 0, 1, 2, ...`` through the frozen `setup_sampler_v1` and
    accepting the first draw whose primary family matches — the accepted
    Phase 9 bank rule, reused rather than reinvented, so the conditional
    distribution is exactly `neutral_v1` given the family.
    """
    parse_phase10_game_id(game_id)
    _require_color(color)
    _require_attempt(attempt)
    return derive_phase10_seed(DOMAIN_CORPUS_SETUP, game_id, color, int(attempt))


def corpus_match_seed(game_id: str) -> int:
    """The match-level randomness seed of one corpus game.

    Both sides play the accepted Phase 9 checkpoint greedily, so no policy
    consumes this stream in v1; it is frozen anyway so the outcome record
    can name a match seed that a replay must reproduce, and so a later
    stochastic replay has a seed that was fixed before any outcome existed.
    """
    parse_phase10_game_id(game_id)
    return derive_phase10_seed(DOMAIN_CORPUS_MATCH, game_id)


# ---------------------------------------------------------------------------
# Evaluation-case identity
# ---------------------------------------------------------------------------

_CASE_ID_PATTERN = re.compile(
    rf"^(?P<bank>phase10_[a-z]+_bank_v[0-9]+)\|ms=(?P<master>[0-9]+)"
    rf"\|f=(?P<family>{_FAMILY_TOKEN})\|c=(?P<ordinal>[0-9]{{3}})$"
)

MAX_CASE_ORDINAL_FORMAT = 999


def phase10_case_id(bank_version: str, family_id: str, case_ordinal: int) -> str:
    """The stable identifier of one logical Phase 10 evaluation case.

    ```text
    phase10_validation_bank_v1|ms=2026081801|f=F03|c=005
    ```

    The family is the case's *opponent-setup* family — the axis the banks
    balance — and the ordinal is that family's case counter. Both banks use
    the same format and differ only in their version token, which is what
    keeps every validation stream disjoint from every test stream while
    both hang off one frozen case-schedule root.
    """
    if not isinstance(bank_version, str) or not bank_version:
        raise Phase10SeedError(f"bank_version must be a non-empty string, got {bank_version!r}")
    if not isinstance(family_id, str) or re.fullmatch(_FAMILY_TOKEN, family_id) is None:
        raise Phase10SeedError(f"family_id must look like 'F00'..'F15', got {family_id!r}")
    if not isinstance(case_ordinal, int) or isinstance(case_ordinal, bool):
        raise Phase10SeedError(
            f"case ordinal must be an int, got {type(case_ordinal).__name__}"
        )
    if not 0 <= case_ordinal <= MAX_CASE_ORDINAL_FORMAT:
        raise Phase10SeedError(
            f"case ordinal {case_ordinal} is outside 0..{MAX_CASE_ORDINAL_FORMAT}"
        )
    case_id = (
        f"{bank_version}|ms={PHASE10_MASTER_SEED}|f={family_id}|c={case_ordinal:03d}"
    )
    if _CASE_ID_PATTERN.match(case_id) is None:
        raise Phase10SeedError(
            f"bank_version {bank_version!r} does not match the frozen Phase 10 bank "
            "naming rule 'phase10_<name>_bank_v<n>'"
        )
    return case_id


def parse_phase10_case_id(case_id: str) -> dict:
    """The identity fields of a Phase 10 evaluation case id, validated."""
    match = _CASE_ID_PATTERN.match(case_id)
    if match is None:
        raise Phase10SeedError(f"malformed Phase 10 case id: {case_id!r}")
    fields = match.groupdict()
    if int(fields["master"]) != PHASE10_MASTER_SEED:
        raise Phase10SeedError(
            f"case id names master seed {fields['master']}, expected {PHASE10_MASTER_SEED}"
        )
    return {
        "bank_version": fields["bank"],
        "phase10_master_seed": int(fields["master"]),
        "family_id": fields["family"],
        "case_ordinal": int(fields["ordinal"]),
    }


def case_opponent_setup_seed(case_id: str, attempt: int) -> int:
    """The sampler draw seed of one case's fixed opponent-setup attempt.

    A case's opponent setup is held out, family-conditioned and frozen
    before any Phase 10 outcome exists: it is the same physical arrangement
    in every matchup and in both arms, which is what makes a learned-minus-
    neutral difference a paired quantity. Attempts are walked for family
    rejection and for the frozen bank-isolation rule.
    """
    parse_phase10_case_id(case_id)
    _require_attempt(attempt)
    return derive_phase10_seed(DOMAIN_BANK_OPPONENT, case_id, int(attempt))


def case_selector_seed(case_id: str, color: str, attempt: int = 0) -> int:
    """The selector draw seed handed to the selector under test.

    One accepted seed per `(case, colour)`, *independent of which selector is
    under test*: the learned candidate and the `neutral_v1` baseline receive
    the identical seed on the identical case, so their difference is measured
    against the same draw identity rather than against two unrelated draws.
    Colour is part of the identity because a selector's own colour is a legal
    selector input and the learned distribution differs by colour.

    ``attempt`` exists for the frozen bank-isolation walk: a case's seed is
    the first attempt whose `neutral_v1` own-side draw is fingerprint-clean
    and distinct from the case's other frozen setups. The walk is decided by
    the baseline draw alone — a quantity fixed before any selector exists —
    so it cannot depend on, or advantage, any candidate.
    """
    parse_phase10_case_id(case_id)
    _require_color(color)
    _require_attempt(attempt)
    return derive_phase10_seed(DOMAIN_BANK_SELECTOR, case_id, color, int(attempt))


def case_match_seed(case_id: str, game_index: int, matchup_token: str) -> int:
    """The match-level seed of one game of one case in one matchup.

    Deliberately independent of the arm (learned or neutral) and of the
    candidate id, so a rule-based opponent draws the identical randomness in
    both arms of a paired comparison and the difference is not polluted by
    the opponent's dice.
    """
    parse_phase10_case_id(case_id)
    if game_index not in CASE_GAME_COLOR:
        raise Phase10SeedError(
            f"game_index must be one of {list(CASE_GAME_INDICES)}, got {game_index!r}"
        )
    if not isinstance(matchup_token, str) or not matchup_token:
        raise Phase10SeedError(
            f"matchup_token must be a non-empty string, got {matchup_token!r}"
        )
    return derive_phase10_seed(
        DOMAIN_BANK_MATCH, case_id, int(game_index), matchup_token
    )


# ---------------------------------------------------------------------------
# Production selector streams
# ---------------------------------------------------------------------------


def selector_branch_uniform(
    selector_identity: str, split: str, color: str, selector_seed: int
) -> float:
    """The frozen uniform in ``[0, 1)`` deciding the mixture branch.

    `phase10_setup_selector_v1` freezes the rule: the draw takes the
    `neutral_v1` branch when this uniform is below the frozen neutral
    weight, and the learned branch otherwise. Keeping the branch coin in its
    own domain means adding or removing a learned base can never move the
    branch decision of any draw.
    """
    return _unit_uniform(
        derive_phase10_seed(
            DOMAIN_SELECTOR_BRANCH,
            _require_text(selector_identity, "selector_identity"),
            _require_text(split, "split"),
            _require_color(color),
            _require_seed(selector_seed),
        )
    )


def selector_base_uniform(
    selector_identity: str, split: str, color: str, selector_seed: int
) -> float:
    """The frozen uniform in ``[0, 1)`` selecting inside the learned branch.

    Consumed by the frozen inverse-CDF walk over the split's bases in
    ascending `(family_index, base_index)` order. One uniform per draw, in
    its own domain, so the learned branch cannot perturb the neutral branch's
    accepted sampler streams.
    """
    return _unit_uniform(
        derive_phase10_seed(
            DOMAIN_SELECTOR_BASE,
            _require_text(selector_identity, "selector_identity"),
            _require_text(split, "split"),
            _require_color(color),
            _require_seed(selector_seed),
        )
    )


# ---------------------------------------------------------------------------
# Fit and bootstrap streams
# ---------------------------------------------------------------------------


def utility_fit_seed(model_id: str) -> int:
    """The deterministic initialisation stream of one utility model.

    Frozen and, under `phase10_setup_utility_v1`, unconsumed: the protocol
    starts L-BFGS from an exact all-zero parameter vector, which is the only
    initialisation that needs no randomness at all. The seed exists so a
    future protocol cannot quietly invent one after seeing outcomes.
    """
    return derive_phase10_seed(DOMAIN_UTILITY_FIT, _require_text(model_id, "model_id"))


def bootstrap_root(bank: str) -> int:
    """The frozen paired-bootstrap root seed of one evaluation bank."""
    if bank == "validation":
        return VALIDATION_BOOTSTRAP_SEED
    if bank == "test":
        return TEST_BOOTSTRAP_SEED
    raise Phase10SeedError(
        f"bootstrap roots exist for the two evaluation banks only, not {bank!r}"
    )


def bootstrap_stream_seed(bank: str, matchup_token: str) -> int:
    """The resampling stream of one matchup or difference on one bank.

    Every matchup and every learned-minus-neutral difference receives its own
    domain-separated token, so two intervals are never resampled from the
    same stream and an interval can be recomputed in isolation from the
    primitive recorded outcomes.
    """
    root = bootstrap_root(bank)
    return derive_phase10_seed(
        DOMAIN_BOOTSTRAP, root, bank, _require_text(matchup_token, "matchup_token")
    )


# ---------------------------------------------------------------------------
# Shared validation and the collision audit
# ---------------------------------------------------------------------------


def _require_color(color: str) -> str:
    if color not in COLORS:
        raise Phase10SeedError(f"colour must be one of {list(COLORS)}, got {color!r}")
    return color


def _require_attempt(attempt: int) -> int:
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
        raise Phase10SeedError(f"attempt must be a non-negative int, got {attempt!r}")
    return attempt


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise Phase10SeedError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _require_seed(seed: int) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise Phase10SeedError(f"selector seed must be a non-negative int, got {seed!r}")
    return seed


def _unit_uniform(seed: int) -> float:
    """A ``[0, 1)`` uniform from a 63-bit stream seed, exactly reproducible.

    Division by ``2**63`` is exact in binary floating point, so the value is
    identical on every platform and the frozen inverse-CDF walk cannot drift.
    """
    return seed / float(1 << 63)


def stream_collision_audit(stream_seeds: "dict[str, list[int]]") -> dict:
    """Duplicate/collision findings over a set of named seed streams.

    Used by the Agent 1 harness to prove exhaustively that no two logical
    identities in the frozen Phase 10 id space share a stream seed, within a
    stream and across streams alike.
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


def seed_derivation_document() -> dict:
    """The machine-readable derivation record for the Agent 1 contract."""
    return {
        "identity_version": PHASE10_IDENTITY_VERSION,
        "outcome_version": PHASE10_OUTCOME_VERSION,
        "root_seeds": dict(CANONICAL_PHASE10_SEEDS),
        "personalization": _PHASE10_SEED_PERSON.decode(),
        "derivation": (
            "blake2b(person='strat-s10', digest_size=8) over "
            "'phase10_identity_v1:domain:domain_root:part:part:...', "
            "big-endian, right-shifted one bit"
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
        "outcome_game_id_format": (
            "phase10_outcome_v1|ms=<master>|rf=<red family>|bf=<blue family>|g=<ordinal:02d>"
        ),
        "case_id_format": (
            "<bank_version>|ms=<master>|f=<opponent-setup family>|c=<ordinal:03d>"
        ),
        "colour_pairing": {
            "rule": "the selector under test plays Red in game 0 and Blue in game 1",
            "game_indices": list(CASE_GAME_INDICES),
            "game_colour": {str(k): v for k, v in sorted(CASE_GAME_COLOR.items())},
        },
        "independence": (
            "no derivation reads worker count, task arrival order, process id, "
            "wall clock, or a physical storage path, so any corpus game or bank "
            "case rebuilds alone and re-sharding cannot change a single draw"
        ),
        "arm_independence": (
            "case selector seeds and case match seeds are independent of which "
            "selector is under test, so learned and neutral arms differ only in "
            "the selector and the paired difference is not polluted by the "
            "opponent's randomness"
        ),
        "root_reading_notes": [
            "the common contract names a validation-case root but no separate "
            "test-case root, so case_schedule_seed roots both banks' case "
            "schedules, domain-separated by the two distinct bank versions",
            "bank opponent-setup draws are Phase 10 setup draws (setup_draw_seed); "
            "bank selector seeds are selector draws (selector_draw_seed)",
        ],
    }


#: The identity parts of every domain, published in the contract so an
#: auditor can re-derive any stream without reading this module's source.
_DOMAIN_PARTS = {
    DOMAIN_CORPUS_SETUP: ["game_id", "color", "attempt"],
    DOMAIN_CORPUS_MATCH: ["game_id"],
    DOMAIN_BANK_OPPONENT: ["case_id", "attempt"],
    DOMAIN_BANK_SELECTOR: ["case_id", "color", "attempt"],
    DOMAIN_BANK_MATCH: ["case_id", "game_index", "matchup_token"],
    DOMAIN_SELECTOR_BRANCH: ["selector_identity", "split", "color", "selector_seed"],
    DOMAIN_SELECTOR_BASE: ["selector_identity", "split", "color", "selector_seed"],
    DOMAIN_UTILITY_FIT: ["model_id"],
    DOMAIN_BOOTSTRAP: ["bootstrap_root", "bank", "matchup_token"],
}
assert set(_DOMAIN_PARTS) == set(STREAM_DOMAINS)


__all__ = [
    "CANONICAL_PHASE10_SEEDS",
    "CASE_GAME_COLOR",
    "CASE_GAME_INDICES",
    "CASE_SCHEDULE_SEED",
    "COLORS",
    "COLOR_BLUE",
    "COLOR_RED",
    "DOMAIN_BANK_MATCH",
    "DOMAIN_BANK_OPPONENT",
    "DOMAIN_BANK_SELECTOR",
    "DOMAIN_BOOTSTRAP",
    "DOMAIN_CORPUS_MATCH",
    "DOMAIN_CORPUS_SETUP",
    "DOMAIN_ROOTS",
    "DOMAIN_SELECTOR_BASE",
    "DOMAIN_SELECTOR_BRANCH",
    "DOMAIN_UTILITY_FIT",
    "MAX_CASE_ORDINAL_FORMAT",
    "MAX_CORPUS_ORDINAL_FORMAT",
    "OUTCOME_SCHEDULE_SEED",
    "PHASE10_IDENTITY_VERSION",
    "PHASE10_MASTER_SEED",
    "PHASE10_OUTCOME_VERSION",
    "SELECTOR_DRAW_SEED",
    "SETUP_DRAW_SEED",
    "STREAM_DOMAINS",
    "TEST_BOOTSTRAP_SEED",
    "UTILITY_FIT_SEED",
    "VALIDATION_BOOTSTRAP_SEED",
    "Phase10SeedError",
    "bootstrap_root",
    "bootstrap_stream_seed",
    "case_match_seed",
    "case_opponent_setup_seed",
    "case_selector_seed",
    "corpus_match_seed",
    "corpus_setup_seed",
    "derive_phase10_seed",
    "parse_phase10_case_id",
    "parse_phase10_game_id",
    "phase10_case_id",
    "phase10_game_id",
    "seed_derivation_document",
    "selector_base_uniform",
    "selector_branch_uniform",
    "stream_collision_audit",
    "utility_fit_seed",
]
