"""Phase 11B stream identities and seeds.

Specification source: `00_PHASE_11B_OVERVIEW.md` ("Common Phase 11B
Dataset" — 2,048 fresh training games, 512 fresh development games).

A separate personalization, so "fresh" is structural
----------------------------------------------------
The Phase 11B games must be *fresh*: not the validation bank, not the spent
test bank, not the soak. Freshness is enforced by construction rather than
by comparison — the blake2b personalization is ``strat-b11b`` and the
domain roots are new integers, so no Phase 11B stream can coincide with a
Phase 11 stream even if two ids happened to spell the same text. Phase 11's
frozen `phase11_seed` module is imported for nothing at all; duplicating
the twelve lines of derivation is the price of not being able to touch it.

The corpus ids also carry their split, so a training game and a development
game can never share a setup draw or a match seed.
"""

from __future__ import annotations

import hashlib
import re

from .contract import (
    CORPUS_COLORS,
    CORPUS_SOURCES,
    CORPUS_SPLITS,
    CORPUS_STRATA,
    Phase11BError,
)

#: Any change to the payload layout, the personalization or the domain
#: tokens is a new identity version, never a silent edit.
PHASE11B_IDENTITY_VERSION = "phase11b_identity_v1"

#: blake2b personalization of every Phase 11B stream. Distinct from
#: Phase 11's ``strat-b11``, which is what makes the corpus fresh.
_PHASE11B_SEED_PERSON = b"strat-b11b"

#: Master seed of the sprint, folded into every Phase 11B logical id.
PHASE11B_MASTER_SEED = 2026081910

#: Root of the corpus setup-draw streams (both seats, both splits).
CORPUS_SETUP_SEED = 2026081911

#: Root of corpus per-game match randomness.
CORPUS_MATCH_SEED = 2026081912

#: Root of training-side randomness: parameter initialization and the
#: shuffling schedule of every Agent 1-5 candidate.
TRAINING_SEED = 2026081913

#: Root of the belief-interface world-sampling smoke checks.
INTERFACE_SEED = 2026081914

CANONICAL_PHASE11B_SEEDS = {
    "phase11b_master_seed": PHASE11B_MASTER_SEED,
    "corpus_setup_seed": CORPUS_SETUP_SEED,
    "corpus_match_seed": CORPUS_MATCH_SEED,
    "training_seed": TRAINING_SEED,
    "interface_seed": INTERFACE_SEED,
}

DOMAIN_OBSERVER_SETUP = "corpus_observer_setup"
DOMAIN_OPPONENT_SETUP = "corpus_opponent_setup"
DOMAIN_MATCH = "corpus_match"
DOMAIN_TRAINING = "training"
DOMAIN_INTERFACE = "interface"

STREAM_DOMAINS = (
    DOMAIN_OBSERVER_SETUP,
    DOMAIN_OPPONENT_SETUP,
    DOMAIN_MATCH,
    DOMAIN_TRAINING,
    DOMAIN_INTERFACE,
)

DOMAIN_ROOTS = {
    DOMAIN_OBSERVER_SETUP: CORPUS_SETUP_SEED,
    DOMAIN_OPPONENT_SETUP: CORPUS_SETUP_SEED,
    DOMAIN_MATCH: CORPUS_MATCH_SEED,
    DOMAIN_TRAINING: TRAINING_SEED,
    DOMAIN_INTERFACE: INTERFACE_SEED,
}
assert set(DOMAIN_ROOTS) == set(STREAM_DOMAINS)

ROLE_OBSERVER = "observer"
ROLE_OPPONENT = "opponent"
SETUP_ROLES = (ROLE_OBSERVER, ROLE_OPPONENT)

_ROLE_DOMAIN = {
    ROLE_OBSERVER: DOMAIN_OBSERVER_SETUP,
    ROLE_OPPONENT: DOMAIN_OPPONENT_SETUP,
}

_SPLIT_ALTERNATION = "|".join(CORPUS_SPLITS)
_STRATUM_ALTERNATION = "|".join(CORPUS_STRATA)
_SOURCE_ALTERNATION = "|".join(CORPUS_SOURCES)
_COLOR_ALTERNATION = "|".join(CORPUS_COLORS)

_GAME_ID_PATTERN = re.compile(
    rf"^phase11b_corpus_v1\|ms=(?P<master>[0-9]+)"
    rf"\|sp=(?P<split>{_SPLIT_ALTERNATION})"
    rf"\|st=(?P<stratum>{_STRATUM_ALTERNATION})"
    rf"\|src=(?P<source>{_SOURCE_ALTERNATION})"
    rf"\|obs=(?P<color>{_COLOR_ALTERNATION})"
    rf"\|g=(?P<ordinal>[0-9]{{4}})$"
)

MAX_GAME_ORDINAL_FORMAT = 9999


def derive_phase11b_seed(domain: str, *parts: "int | str") -> int:
    """A 63-bit deterministic seed for one Phase 11B stream.

    Same construction as the accepted Phase 11 derivation — colon-joined
    identity payload, blake2b-8, one right shift — under a different
    personalization and different roots, so equal identities always agree
    and no Phase 11B stream can collide with a Phase 11 one.
    """
    if domain not in STREAM_DOMAINS:
        raise Phase11BError(f"unknown Phase 11B stream domain: {domain!r}")
    for part in parts:
        if not isinstance(part, (int, str)) or isinstance(part, bool):
            raise Phase11BError(
                f"stream identity parts must be int or str, got {type(part).__name__}"
            )
        if isinstance(part, str) and ":" in part:
            raise Phase11BError(
                f"string identity parts may not contain ':' (got {part!r})"
            )
    payload = ":".join(
        [
            PHASE11B_IDENTITY_VERSION,
            domain,
            str(DOMAIN_ROOTS[domain]),
            *[str(part) for part in parts],
        ]
    )
    digest = hashlib.blake2b(
        payload.encode(), digest_size=8, person=_PHASE11B_SEED_PERSON
    ).digest()
    return int.from_bytes(digest, "big") >> 1


def corpus_game_id(
    split: str, stratum: str, source: str, observer_color: str, ordinal: int
) -> str:
    """The stable identifier of one Phase 11B corpus game.

    ```text
    phase11b_corpus_v1|ms=2026081910|sp=train|st=scout_rush|src=p10d|obs=red|g=0017
    ```

    The split, stratum, source and observer colour are all in the id, so
    every corpus slice is a pure string filter and every stream a game
    consumes is disjoint from every other cell's.
    """
    if split not in CORPUS_SPLITS:
        raise Phase11BError(f"split must be one of {list(CORPUS_SPLITS)}, got {split!r}")
    if stratum not in CORPUS_STRATA:
        raise Phase11BError(f"stratum must be one of {list(CORPUS_STRATA)}, got {stratum!r}")
    if source not in CORPUS_SOURCES:
        raise Phase11BError(f"source must be one of {list(CORPUS_SOURCES)}, got {source!r}")
    if observer_color not in CORPUS_COLORS:
        raise Phase11BError(
            f"observer colour must be one of {list(CORPUS_COLORS)}, got {observer_color!r}"
        )
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise Phase11BError(f"ordinal must be an int, got {type(ordinal).__name__}")
    if not 0 <= ordinal <= MAX_GAME_ORDINAL_FORMAT:
        raise Phase11BError(f"ordinal {ordinal} is outside 0..{MAX_GAME_ORDINAL_FORMAT}")
    game_id = (
        f"phase11b_corpus_v1|ms={PHASE11B_MASTER_SEED}|sp={split}|st={stratum}"
        f"|src={source}|obs={observer_color}|g={ordinal:04d}"
    )
    if _GAME_ID_PATTERN.match(game_id) is None:  # pragma: no cover - defensive
        raise Phase11BError(f"constructed a malformed Phase 11B game id: {game_id!r}")
    return game_id


def parse_corpus_game_id(game_id: str) -> dict:
    """The identity fields of a Phase 11B corpus game id, validated."""
    match = _GAME_ID_PATTERN.match(game_id)
    if match is None:
        raise Phase11BError(f"malformed Phase 11B game id: {game_id!r}")
    fields = match.groupdict()
    if int(fields["master"]) != PHASE11B_MASTER_SEED:
        raise Phase11BError(
            f"game id names master seed {fields['master']}, expected {PHASE11B_MASTER_SEED}"
        )
    return {
        "phase11b_master_seed": int(fields["master"]),
        "split": fields["split"],
        "stratum": fields["stratum"],
        "setup_source": fields["source"],
        "observer_color": fields["color"],
        "ordinal": int(fields["ordinal"]),
    }


def setup_seed(game_id: str, role: str) -> int:
    """The setup-draw seed of one seat of one corpus game."""
    parse_corpus_game_id(game_id)
    if role not in _ROLE_DOMAIN:
        raise Phase11BError(f"role must be one of {list(SETUP_ROLES)}, got {role!r}")
    return derive_phase11b_seed(_ROLE_DOMAIN[role], game_id)


def match_seed(game_id: str) -> int:
    """The match-randomness seed of one corpus game.

    The rule and stress opponents draw their per-decision randomness from
    the accepted `FrozenSeedPolicy` derivations rooted here; the neural
    seats play the accepted greedy mode and consume none of it.
    """
    parse_corpus_game_id(game_id)
    return derive_phase11b_seed(DOMAIN_MATCH, game_id)


def training_seed(candidate: str, purpose: str) -> int:
    """The seed of one training stream — `init`, `shuffle`, or similar."""
    return derive_phase11b_seed(DOMAIN_TRAINING, str(candidate), str(purpose))


def interface_seed(purpose: str, ordinal: int = 0) -> int:
    """The seed of one belief-interface smoke stream."""
    return derive_phase11b_seed(DOMAIN_INTERFACE, str(purpose), int(ordinal))


__all__ = [
    "CANONICAL_PHASE11B_SEEDS",
    "PHASE11B_IDENTITY_VERSION",
    "PHASE11B_MASTER_SEED",
    "ROLE_OBSERVER",
    "ROLE_OPPONENT",
    "STREAM_DOMAINS",
    "corpus_game_id",
    "derive_phase11b_seed",
    "interface_seed",
    "match_seed",
    "parse_corpus_game_id",
    "setup_seed",
    "training_seed",
    "training_seed",
]
