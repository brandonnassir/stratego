"""Phase 15 stream identities and seeds.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` sections 5-7.

Freshness is structural, not asserted
-------------------------------------
The blake2b personalization is ``strat-p15`` and the domain roots are new
integers, so no Phase 15 stream can coincide with a Phase 11, Phase 11B or
Phase 14 stream even if two ids happened to spell the same text. The
Phase 11B seed module is imported for nothing at all.

The game id carries its own identity
------------------------------------
A Phase 15 game id names its split, its observer model, its opponent, its
setup source and its observer colour. Every corpus slice is therefore a
pure string filter, every cell's streams are disjoint from every other
cell's, and the split is inside the id — so a training game and a
development game can never share a setup draw or a match seed.
"""

from __future__ import annotations

import hashlib
import re

from .contract import (
    CORPUS_COLORS,
    CORPUS_SPLITS,
    OPPONENTS,
    POLICY_SOURCES,
    SETUP_SOURCES,
    Phase15Error,
)

#: Any change to the payload layout, the personalization or the domain
#: tokens is a new identity version, never a silent edit.
PHASE15_IDENTITY_VERSION = "phase15_identity_v1"

#: blake2b personalization of every Phase 15 stream.
_PHASE15_SEED_PERSON = b"strat-p15"

#: Master seed of the phase, folded into every Phase 15 logical id.
PHASE15_MASTER_SEED = 2026082410

#: Root of the corpus setup-draw streams (both seats, all splits).
CORPUS_SETUP_SEED = 2026082411

#: Root of corpus per-game match randomness.
CORPUS_MATCH_SEED = 2026082412

#: Root of training-side randomness: initialization and shuffling.
TRAINING_SEED = 2026082413

#: Root of the belief-interface world-sampling checks.
INTERFACE_SEED = 2026082414

#: Root of the orientation gate's board-generation stream.
ORIENTATION_SEED = 2026082415

CANONICAL_PHASE15_SEEDS = {
    "phase15_master_seed": PHASE15_MASTER_SEED,
    "corpus_setup_seed": CORPUS_SETUP_SEED,
    "corpus_match_seed": CORPUS_MATCH_SEED,
    "training_seed": TRAINING_SEED,
    "interface_seed": INTERFACE_SEED,
    "orientation_seed": ORIENTATION_SEED,
}

DOMAIN_OBSERVER_SETUP = "corpus_observer_setup"
DOMAIN_OPPONENT_SETUP = "corpus_opponent_setup"
DOMAIN_MATCH = "corpus_match"
DOMAIN_TRAINING = "training"
DOMAIN_INTERFACE = "interface"
DOMAIN_ORIENTATION = "orientation"

STREAM_DOMAINS = (
    DOMAIN_OBSERVER_SETUP,
    DOMAIN_OPPONENT_SETUP,
    DOMAIN_MATCH,
    DOMAIN_TRAINING,
    DOMAIN_INTERFACE,
    DOMAIN_ORIENTATION,
)

DOMAIN_ROOTS = {
    DOMAIN_OBSERVER_SETUP: CORPUS_SETUP_SEED,
    DOMAIN_OPPONENT_SETUP: CORPUS_SETUP_SEED,
    DOMAIN_MATCH: CORPUS_MATCH_SEED,
    DOMAIN_TRAINING: TRAINING_SEED,
    DOMAIN_INTERFACE: INTERFACE_SEED,
    DOMAIN_ORIENTATION: ORIENTATION_SEED,
}
assert set(DOMAIN_ROOTS) == set(STREAM_DOMAINS)

ROLE_OBSERVER = "observer"
ROLE_OPPONENT = "opponent"
SETUP_ROLES = (ROLE_OBSERVER, ROLE_OPPONENT)

_ROLE_DOMAIN = {
    ROLE_OBSERVER: DOMAIN_OBSERVER_SETUP,
    ROLE_OPPONENT: DOMAIN_OPPONENT_SETUP,
}

MAX_GAME_ORDINAL_FORMAT = 999_999

_SPLIT_ALTERNATION = "|".join(CORPUS_SPLITS)
_OBSERVER_ALTERNATION = "|".join(POLICY_SOURCES)
_OPPONENT_ALTERNATION = "|".join(re.escape(name) for name in OPPONENTS)
_SOURCE_ALTERNATION = "|".join(re.escape(name) for name in SETUP_SOURCES)
_COLOR_ALTERNATION = "|".join(CORPUS_COLORS)

_GAME_ID_PATTERN = re.compile(
    rf"^{re.escape('phase15_belief_corpus_v1')}\|ms=(?P<master>[0-9]+)"
    rf"\|sp=(?P<split>{_SPLIT_ALTERNATION})"
    rf"\|obs=(?P<observer>{_OBSERVER_ALTERNATION})"
    rf"\|opp=(?P<opponent>{_OPPONENT_ALTERNATION})"
    rf"\|src=(?P<source>{_SOURCE_ALTERNATION})"
    rf"\|col=(?P<color>{_COLOR_ALTERNATION})"
    rf"\|g=(?P<ordinal>[0-9]{{6}})$"
)


def derive_phase15_seed(domain: str, *parts: "int | str") -> int:
    """A 63-bit deterministic seed for one Phase 15 stream.

    Same construction as the accepted Phase 11/11B derivations — colon
    joined identity payload, blake2b-8, one right shift — under a new
    personalization and new roots, so equal identities always agree and no
    Phase 15 stream can collide with an earlier phase's.
    """
    if domain not in STREAM_DOMAINS:
        raise Phase15Error(f"unknown Phase 15 stream domain: {domain!r}")
    for part in parts:
        if not isinstance(part, (int, str)) or isinstance(part, bool):
            raise Phase15Error(
                f"stream identity parts must be int or str, got {type(part).__name__}"
            )
        if isinstance(part, str) and ":" in part:
            raise Phase15Error(
                f"string identity parts may not contain ':' (got {part!r})"
            )
    payload = ":".join(
        [
            PHASE15_IDENTITY_VERSION,
            domain,
            str(DOMAIN_ROOTS[domain]),
            *[str(part) for part in parts],
        ]
    )
    digest = hashlib.blake2b(
        payload.encode(), digest_size=8, person=_PHASE15_SEED_PERSON
    ).digest()
    return int.from_bytes(digest, "big") >> 1


def corpus_game_id(
    split: str,
    observer_model: str,
    opponent: str,
    setup_source: str,
    observer_color: str,
    ordinal: int,
) -> str:
    """The stable identifier of one Phase 15 corpus game.

    ```text
    phase15_belief_corpus_v1|ms=2026082410|sp=train|obs=p18|opp=p24
        |src=neutral_v1|col=red|g=000017
    ```
    """
    if split not in CORPUS_SPLITS:
        raise Phase15Error(f"split must be one of {list(CORPUS_SPLITS)}, got {split!r}")
    if observer_model not in POLICY_SOURCES:
        raise Phase15Error(
            f"observer must be one of {list(POLICY_SOURCES)}, got {observer_model!r}"
        )
    if opponent not in OPPONENTS:
        raise Phase15Error(f"opponent must be one of {list(OPPONENTS)}, got {opponent!r}")
    if setup_source not in SETUP_SOURCES:
        raise Phase15Error(
            f"setup source must be one of {list(SETUP_SOURCES)}, got {setup_source!r}"
        )
    if observer_color not in CORPUS_COLORS:
        raise Phase15Error(
            f"observer colour must be one of {list(CORPUS_COLORS)}, got {observer_color!r}"
        )
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise Phase15Error(f"ordinal must be an int, got {type(ordinal).__name__}")
    if not 0 <= ordinal <= MAX_GAME_ORDINAL_FORMAT:
        raise Phase15Error(f"ordinal {ordinal} is outside 0..{MAX_GAME_ORDINAL_FORMAT}")
    game_id = (
        f"phase15_belief_corpus_v1|ms={PHASE15_MASTER_SEED}|sp={split}"
        f"|obs={observer_model}|opp={opponent}|src={setup_source}"
        f"|col={observer_color}|g={ordinal:06d}"
    )
    if _GAME_ID_PATTERN.match(game_id) is None:  # pragma: no cover - defensive
        raise Phase15Error(f"constructed a malformed Phase 15 game id: {game_id!r}")
    return game_id


def parse_corpus_game_id(game_id: str) -> dict:
    """The identity fields of a Phase 15 corpus game id, validated."""
    match = _GAME_ID_PATTERN.match(game_id)
    if match is None:
        raise Phase15Error(f"malformed Phase 15 game id: {game_id!r}")
    fields = match.groupdict()
    if int(fields["master"]) != PHASE15_MASTER_SEED:
        raise Phase15Error(
            f"game id names master seed {fields['master']}, expected {PHASE15_MASTER_SEED}"
        )
    return {
        "phase15_master_seed": int(fields["master"]),
        "split": fields["split"],
        "observer_model": fields["observer"],
        "opponent": fields["opponent"],
        "setup_source": fields["source"],
        "observer_color": fields["color"],
        "ordinal": int(fields["ordinal"]),
    }


def setup_seed(game_id: str, role: str) -> int:
    """The setup-draw seed of one seat of one corpus game."""
    parse_corpus_game_id(game_id)
    if role not in _ROLE_DOMAIN:
        raise Phase15Error(f"role must be one of {list(SETUP_ROLES)}, got {role!r}")
    return derive_phase15_seed(_ROLE_DOMAIN[role], game_id)


def match_seed(game_id: str) -> int:
    """The match-randomness seed of one corpus game."""
    parse_corpus_game_id(game_id)
    return derive_phase15_seed(DOMAIN_MATCH, game_id)


def training_seed(specialist: str, purpose: str) -> int:
    """The seed of one training stream — `init`, `shuffle`, or similar."""
    return derive_phase15_seed(DOMAIN_TRAINING, str(specialist), str(purpose))


def interface_seed(purpose: str, ordinal: int = 0) -> int:
    """The seed of one belief-interface check stream."""
    return derive_phase15_seed(DOMAIN_INTERFACE, str(purpose), int(ordinal))


def orientation_seed(purpose: str, ordinal: int = 0) -> int:
    """The seed of one orientation-gate board stream."""
    return derive_phase15_seed(DOMAIN_ORIENTATION, str(purpose), int(ordinal))


__all__ = [
    "CANONICAL_PHASE15_SEEDS",
    "PHASE15_IDENTITY_VERSION",
    "PHASE15_MASTER_SEED",
    "ROLE_OBSERVER",
    "ROLE_OPPONENT",
    "STREAM_DOMAINS",
    "corpus_game_id",
    "derive_phase15_seed",
    "interface_seed",
    "match_seed",
    "orientation_seed",
    "parse_corpus_game_id",
    "setup_seed",
    "training_seed",
]
