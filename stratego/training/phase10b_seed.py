"""Optional Phase 10B: root seeds, logical identities and derived streams.

Specification source: `OPTIONAL_PHASE_10B_SETUP_CONDITIONED_FINE_TUNING_AGENT.md`
sections 12 and 13.

Purity
------
Every function here is a pure function of the frozen roots and explicit
identity inputs. There is no filesystem access, no environment lookup, no
clock, no process-global RNG cursor and no path anywhere in this module.
That absence is the mechanical proof of the plan's rule that "no seed may
depend on worker count, process id, path, wall clock, or arrival order": a
Phase 10B rollout regenerated on another machine with a different worker
count is the same rollout.

Separation from Phase 9
-----------------------
Phase 10B derives its own streams from its own roots under its own
personalization. A Phase 10B stream can therefore never collide with a
Phase 9 one even where the identity inputs look similar, and the accepted
Phase 9 seed module is imported for nothing at all.
"""

from __future__ import annotations

import hashlib
import re

from .phase10b_contract import (
    ANCHOR_IDENTITY,
    MAX_ITERATIONS,
    PHASE10B_NAMESPACE,
    PHASE10B_ROLLOUT_VERSION,
    POPULATION_BUCKETS,
    Phase10BContractError,
    bucket_counts,
)


class Phase10BSeedError(RuntimeError):
    """Raised when a Phase 10B identity or stream request is invalid."""


# ---------------------------------------------------------------------------
# Frozen roots
# ---------------------------------------------------------------------------

MASTER_SEED = 20260819021
ROLLOUT_SCHEDULE_SEED = 20260819022
OPPONENT_SELECTION_SEED = 20260819023
SETUP_SELECTION_SEED = 20260819024
TRAINING_ORDER_SEED = 20260819025
VALIDATION_SCHEDULE_SEED = 20260819026
VALIDATION_BOOTSTRAP_SEED = 20260819027
FINAL_BOOTSTRAP_SEED = 20260819028

ROOT_SEEDS = {
    "master": MASTER_SEED,
    "rollout_schedule": ROLLOUT_SCHEDULE_SEED,
    "opponent_selection": OPPONENT_SELECTION_SEED,
    "setup_selection": SETUP_SELECTION_SEED,
    "training_order": TRAINING_ORDER_SEED,
    "validation_schedule": VALIDATION_SCHEDULE_SEED,
    "validation_bootstrap": VALIDATION_BOOTSTRAP_SEED,
    "final_bootstrap": FINAL_BOOTSTRAP_SEED,
}


# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------

DOMAIN_ROLLOUT_GAME = "rollout_game"
DOMAIN_OPPONENT_BUCKET = "opponent_bucket"
DOMAIN_OPPONENT_IDENTITY = "opponent_identity"
DOMAIN_RED_SETUP = "red_setup"
DOMAIN_BLUE_SETUP = "blue_setup"
DOMAIN_ACTION_SAMPLING = "action_sampling"
DOMAIN_TRAINING_ORDER = "training_order"
DOMAIN_ARCHIVE_SELECTION = "archive_selection"
DOMAIN_VALIDATION_CASE = "validation_case"
DOMAIN_BOOTSTRAP = "bootstrap"

#: Every domain the plan requires. A domain with no consumer is reserved
#: rather than removed, and `seed_contract` records why.
STREAM_DOMAINS = (
    DOMAIN_ROLLOUT_GAME,
    DOMAIN_OPPONENT_BUCKET,
    DOMAIN_OPPONENT_IDENTITY,
    DOMAIN_RED_SETUP,
    DOMAIN_BLUE_SETUP,
    DOMAIN_ACTION_SAMPLING,
    DOMAIN_TRAINING_ORDER,
    DOMAIN_ARCHIVE_SELECTION,
    DOMAIN_VALIDATION_CASE,
    DOMAIN_BOOTSTRAP,
)

#: The root each domain descends from.
DOMAIN_ROOTS = {
    DOMAIN_ROLLOUT_GAME: ROLLOUT_SCHEDULE_SEED,
    DOMAIN_OPPONENT_BUCKET: OPPONENT_SELECTION_SEED,
    DOMAIN_OPPONENT_IDENTITY: OPPONENT_SELECTION_SEED,
    DOMAIN_RED_SETUP: SETUP_SELECTION_SEED,
    DOMAIN_BLUE_SETUP: SETUP_SELECTION_SEED,
    DOMAIN_ACTION_SAMPLING: MASTER_SEED,
    DOMAIN_TRAINING_ORDER: TRAINING_ORDER_SEED,
    DOMAIN_ARCHIVE_SELECTION: OPPONENT_SELECTION_SEED,
    DOMAIN_VALIDATION_CASE: VALIDATION_SCHEDULE_SEED,
    DOMAIN_BOOTSTRAP: VALIDATION_BOOTSTRAP_SEED,
}

_PERSON = b"strat10b"


def derive_seed(domain: str, *parts: "int | str") -> int:
    """A 63-bit deterministic seed for one Phase 10B stream.

    The payload is the colon-joined text of the rollout version, the domain,
    the domain's frozen root and the identity parts, hashed under the Phase
    10B personalization. Equal identities always agree; any change to any
    identity input yields an unrelated stream.
    """
    if domain not in STREAM_DOMAINS:
        raise Phase10BSeedError(f"unknown Phase 10B stream domain: {domain!r}")
    for part in parts:
        if not isinstance(part, (int, str)) or isinstance(part, bool):
            raise Phase10BSeedError(
                f"stream identity parts must be int or str, got {type(part).__name__}"
            )
    payload = ":".join(
        [
            PHASE10B_ROLLOUT_VERSION,
            domain,
            str(DOMAIN_ROOTS[domain]),
            *[str(part) for part in parts],
        ]
    )
    digest = hashlib.blake2b(payload.encode(), digest_size=8, person=_PERSON).digest()
    return int.from_bytes(digest, "big") >> 1


# ---------------------------------------------------------------------------
# Logical game identity
# ---------------------------------------------------------------------------

_GAME_ID_PATTERN = re.compile(
    r"^(?P<version>[a-z0-9_]+)\|ms=(?P<master>[0-9]+)\|ns=(?P<namespace>[a-z0-9_]+)"
    r"\|it=(?P<iteration>[0-9]{3})\|b=(?P<bucket>[a-z]+)\|g=(?P<ordinal>[0-9]{4})$"
)

MIN_ITERATION = 1
MAX_ORDINAL_FORMAT = 9999


def game_id(iteration: int, bucket: str, ordinal: int) -> str:
    """The stable identifier of one logical Phase 10B rollout game.

    ```text
    phase10b_rollout_v1|ms=20260819021|ns=phase10b|it=012|b=archive|g=0137
    ```

    Worker count, process partitioning, arrival order and resume boundaries
    appear nowhere, which is what makes crash regeneration exact.
    """
    if bucket not in POPULATION_BUCKETS:
        raise Phase10BSeedError(
            f"unknown population bucket {bucket!r}; expected one of "
            f"{list(POPULATION_BUCKETS)}"
        )
    if not isinstance(iteration, int) or isinstance(iteration, bool):
        raise Phase10BSeedError(
            f"iteration must be an int, got {type(iteration).__name__}"
        )
    if not MIN_ITERATION <= iteration <= MAX_ITERATIONS:
        raise Phase10BSeedError(
            f"iteration {iteration} is outside the frozen "
            f"{MIN_ITERATION}..{MAX_ITERATIONS} budget"
        )
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise Phase10BSeedError(
            f"game ordinal must be an int, got {type(ordinal).__name__}"
        )
    limit = bucket_counts()[bucket]
    if not 0 <= ordinal < limit:
        raise Phase10BSeedError(
            f"ordinal {ordinal} is outside 0..{limit - 1} for bucket {bucket!r}"
        )
    return (
        f"{PHASE10B_ROLLOUT_VERSION}|ms={MASTER_SEED}|ns={PHASE10B_NAMESPACE}"
        f"|it={iteration:03d}|b={bucket}|g={ordinal:04d}"
    )


def parse_game_id(identifier: str) -> dict:
    """The identity fields of a Phase 10B rollout game id, fully validated.

    Refuses anything that is not exactly a well-formed id of this rollout
    version under the frozen Phase 10B master seed, so a Phase 9 identifier —
    or a tampered one — can never be mistaken for a Phase 10B rollout game.
    """
    match = _GAME_ID_PATTERN.match(identifier)
    if match is None:
        raise Phase10BSeedError(f"malformed Phase 10B game id: {identifier!r}")
    fields = match.groupdict()
    if fields["version"] != PHASE10B_ROLLOUT_VERSION:
        raise Phase10BSeedError(
            f"game id names rollout version {fields['version']!r}, expected "
            f"{PHASE10B_ROLLOUT_VERSION!r}"
        )
    if int(fields["master"]) != MASTER_SEED:
        raise Phase10BSeedError(
            f"game id names master seed {fields['master']}, expected {MASTER_SEED}"
        )
    if fields["namespace"] != PHASE10B_NAMESPACE:
        raise Phase10BSeedError(
            f"game id names namespace {fields['namespace']!r}, expected "
            f"{PHASE10B_NAMESPACE!r}"
        )
    if fields["bucket"] not in POPULATION_BUCKETS:
        raise Phase10BSeedError(f"game id names unknown bucket {fields['bucket']!r}")
    iteration = int(fields["iteration"])
    if not MIN_ITERATION <= iteration <= MAX_ITERATIONS:
        raise Phase10BSeedError(
            f"game id iteration {iteration} is outside 1..{MAX_ITERATIONS}"
        )
    ordinal = int(fields["ordinal"])
    limit = bucket_counts()[fields["bucket"]]
    if not 0 <= ordinal < limit:
        raise Phase10BSeedError(
            f"game id ordinal {ordinal} is outside 0..{limit - 1} for bucket "
            f"{fields['bucket']!r}"
        )
    return {
        "rollout_version": fields["version"],
        "master_seed": int(fields["master"]),
        "namespace": fields["namespace"],
        "iteration": iteration,
        "bucket": fields["bucket"],
        "ordinal": ordinal,
    }


# ---------------------------------------------------------------------------
# Per-game streams
# ---------------------------------------------------------------------------


def setup_root_seed(identifier: str) -> int:
    """The setup-source root seed of one logical rollout game."""
    parse_game_id(identifier)
    return derive_seed(DOMAIN_ROLLOUT_GAME, identifier)


def side_selector_seed(identifier: str, color: str) -> int:
    """The P10-D selector seed of one side of one logical rollout game.

    Red and Blue draw from two different domains, so the two sides of a game
    are independent draws through their own colour-specific distributions —
    exactly what section 6 of the plan requires — and neither can see the
    other's stream.
    """
    parse_game_id(identifier)
    if color == "red":
        domain = DOMAIN_RED_SETUP
    elif color == "blue":
        domain = DOMAIN_BLUE_SETUP
    else:
        raise Phase10BSeedError(f"unknown colour {color!r}; expected red or blue")
    return derive_seed(domain, identifier, color)


def opponent_policy_seed(identifier: str, color: str) -> int:
    """The match-level policy seed of a rule/stress side."""
    parse_game_id(identifier)
    if color not in ("red", "blue"):
        raise Phase10BSeedError(f"unknown colour {color!r}; expected red or blue")
    return derive_seed(DOMAIN_OPPONENT_IDENTITY, identifier, color)


def archive_selection_seed(identifier: str) -> int:
    """The archive-bucket draw stream of one logical rollout game."""
    fields = parse_game_id(identifier)
    if fields["bucket"] != "archive":
        raise Phase10BSeedError(
            f"{identifier} is a {fields['bucket']!r} game; only an archive game "
            "draws an archive member"
        )
    return derive_seed(DOMAIN_ARCHIVE_SELECTION, identifier)


def archive_member_for(identifier: str, window) -> str:
    """The frozen uniform active-window draw of one archive-bucket game.

    A pure function of the game identity and the frozen archive-selection
    root. No match result, win rate or league table enters here, because the
    plan forbids performance-based archive weighting.
    """
    members = tuple(window)
    if not members:
        raise Phase10BSeedError("the active archive window is empty")
    return members[archive_selection_seed(identifier) % len(members)]


def action_sampling_seed(identifier: str, ply: int) -> int:
    """The action-sampling stream of one neural decision.

    Exactly one player acts at each ply, so `(game_id, ply)` names one
    decision; both a learner side and a checkpoint-opponent side draw from
    this stream when they act.
    """
    parse_game_id(identifier)
    if not isinstance(ply, int) or isinstance(ply, bool) or ply < 0:
        raise Phase10BSeedError(f"ply must be a non-negative int, got {ply!r}")
    return derive_seed(DOMAIN_ACTION_SAMPLING, identifier, ply)


def action_sampling_uniform(identifier: str, ply: int) -> float:
    """The uniform in ``(0, 1]`` behind one behavior action draw.

    Same half-open orientation and same cumulative-walk contract as the
    accepted Phase 9 behavior sampler; only the stream identity is Phase
    10B's, so a probability-zero prefix stays unselectable.
    """
    return (action_sampling_seed(identifier, ply) + 1) / (2**63 + 1)


def train_order_seed(iteration: int, epoch: int) -> int:
    """The shuffle seed of one optimizer epoch over one sealed rollout."""
    if not isinstance(iteration, int) or isinstance(iteration, bool):
        raise Phase10BSeedError(f"iteration must be an int, got {iteration!r}")
    if not MIN_ITERATION <= iteration <= MAX_ITERATIONS:
        raise Phase10BSeedError(
            f"iteration {iteration} is outside 1..{MAX_ITERATIONS}"
        )
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise Phase10BSeedError(f"epoch must be a non-negative int, got {epoch!r}")
    return derive_seed(DOMAIN_TRAINING_ORDER, iteration, epoch)


# ---------------------------------------------------------------------------
# Evaluation streams
# ---------------------------------------------------------------------------


def case_selector_seed(bank_version: str, case_id: str, color: str) -> int:
    """The own-side selector seed of one evaluation case.

    Both arms of a paired comparison read this same seed, which is what makes
    the P10-D own-side draw identical in the Phase 10B arm and the Phase 9
    arm: the only difference between the two arms is the move policy.
    """
    if color not in ("red", "blue"):
        raise Phase10BSeedError(f"unknown colour {color!r}; expected red or blue")
    return derive_seed(DOMAIN_VALIDATION_CASE, bank_version, case_id, "selector", color)


def case_opponent_seed(bank_version: str, case_id: str, color: str) -> int:
    """The held-out opponent-setup draw seed of one evaluation case side."""
    if color not in ("red", "blue"):
        raise Phase10BSeedError(f"unknown colour {color!r}; expected red or blue")
    return derive_seed(DOMAIN_VALIDATION_CASE, bank_version, case_id, "opponent", color)


def case_match_seed(bank_version: str, case_id: str, game_index: int, matchup: str) -> int:
    """The frozen match seed of one evaluation game.

    Carries the matchup so a rule-based opponent's randomness is identical in
    both arms of the same comparison and independent across matchups.
    """
    if game_index not in (0, 1):
        raise Phase10BSeedError(f"unknown game index {game_index!r}; expected 0 or 1")
    return derive_seed(
        DOMAIN_VALIDATION_CASE, bank_version, case_id, "match", int(game_index), matchup
    )


def bootstrap_seed(bank_version: str, token: str) -> int:
    """The bootstrap stream of one measured quantity on one bank."""
    root = (
        FINAL_BOOTSTRAP_SEED
        if bank_version.endswith("test_bank_v1")
        else VALIDATION_BOOTSTRAP_SEED
    )
    return derive_seed(DOMAIN_BOOTSTRAP, root, bank_version, token)


# ---------------------------------------------------------------------------
# The seed contract document
# ---------------------------------------------------------------------------

#: Domains with a real draw behind them, and what that draw decides.
DOMAIN_CONSUMERS = {
    DOMAIN_ROLLOUT_GAME: "setup_root_seed of one logical rollout game",
    DOMAIN_RED_SETUP: "the Red P10-D selector seed of one rollout game",
    DOMAIN_BLUE_SETUP: "the Blue P10-D selector seed of one rollout game",
    DOMAIN_OPPONENT_IDENTITY: "the match-level policy seed of a rule/stress side",
    DOMAIN_ARCHIVE_SELECTION: "the uniform active-window draw of an archive game",
    DOMAIN_ACTION_SAMPLING: "one neural decision's cumulative-walk uniform",
    DOMAIN_TRAINING_ORDER: "the minibatch shuffle of one optimizer epoch",
    DOMAIN_VALIDATION_CASE: "evaluation case selector/opponent/match seeds",
    DOMAIN_BOOTSTRAP: "paired logical-case bootstrap resampling",
}

#: Domains reserved with no draw behind them, and why.
DOMAIN_RESERVED = {
    DOMAIN_OPPONENT_BUCKET: (
        "the population bucket and the opponent-bucket policy are assigned by "
        "contiguous ordinal subranges, so both are exact by construction and "
        "nothing is drawn; the domain is reserved so any future bucket-level "
        "draw has a separated home rather than borrowing another stream"
    ),
}


def seed_contract() -> dict:
    """The complete frozen seed contract, for the acceptance artifact."""
    reserved = set(DOMAIN_RESERVED)
    consumed = set(DOMAIN_CONSUMERS)
    if consumed | reserved != set(STREAM_DOMAINS):
        raise Phase10BContractError(
            "every declared domain must be either consumed or explicitly reserved"
        )
    return {
        "seed_version": "phase10b_seed_v1",
        "roots": dict(ROOT_SEEDS),
        "domains": list(STREAM_DOMAINS),
        "domain_roots": dict(DOMAIN_ROOTS),
        "consumers": dict(DOMAIN_CONSUMERS),
        "reserved": dict(DOMAIN_RESERVED),
        "derivation": (
            "blake2b(payload, digest_size=8, person='strat10b') >> 1 over "
            "'<rollout version>:<domain>:<domain root>:<identity parts>'"
        ),
        "independence": (
            "no seed reads worker count, process id, path, wall clock or "
            "arrival order; the module performs no I/O of any kind"
        ),
        "anchor_identity": ANCHOR_IDENTITY,
        "game_id_format": (
            f"{PHASE10B_ROLLOUT_VERSION}|ms={MASTER_SEED}|ns={PHASE10B_NAMESPACE}"
            "|it=NNN|b=<bucket>|g=NNNN"
        ),
    }


def seed_contract_digest() -> str:
    import json

    return hashlib.sha256(
        json.dumps(seed_contract(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [name for name in dir() if not name.startswith("_")]
