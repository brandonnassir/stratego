"""Phase 14: root seeds, logical identities and derived streams.

Specification source: the frozen `phase13_final_training_contract_v1`, via
`02_AGENT_2_FINAL_TRAINING_INTEGRATION.md` sections 3, 6 and 15.

Purity
------
Every function here is a pure function of the frozen roots and explicit
identity inputs. There is no filesystem access, no environment lookup, no
clock, no process-global RNG cursor and no path anywhere in this module. That
absence is what makes a Phase 14 rollout regenerate identically after a crash
on a machine with a different worker count: a decision is a function of
`(game_id, ply)`, never of arrival order.

Separation from Phase 9, 10B and the pilots
-------------------------------------------
Phase 14 derives its streams from its own roots under its own personalization
token, so a Phase 14 stream can never collide with a Phase 9 one even where the
identity inputs look similar. The accepted Phase 9 seed module is imported for
nothing at all, and a Phase 9 parser rejects a Phase 14 id on sight.

What has no stream here
-----------------------
Which historical member a game faces. The frozen pool contract partitions the
historical bucket into *exact per-member counts* rather than sampling it, so
that assignment is arithmetic over ordinals and deliberately owns no seed:
adding one would create a way for the opponent mix to drift from its frozen
percentages.
"""

from __future__ import annotations

import hashlib
import re

from .phase14_contract import (
    PHASE14_NAMESPACE,
    PHASE14_ROLLOUT_VERSION,
    POPULATION_BUCKETS,
    Phase14ContractError,
    bucket_counts,
)


class Phase14SeedError(RuntimeError):
    """Raised when a Phase 14 identity or stream request is invalid."""


# ---------------------------------------------------------------------------
# Frozen roots
# ---------------------------------------------------------------------------

MASTER_SEED = 20260820141
ROLLOUT_SCHEDULE_SEED = 20260820142
SETUP_SELECTION_SEED = 20260820143
POLICY_SEED = 20260820144
ACTION_SAMPLING_SEED = 20260820145
TRAINING_ORDER_SEED = 20260820146

ROOT_SEEDS = {
    "master": MASTER_SEED,
    "rollout_schedule": ROLLOUT_SCHEDULE_SEED,
    "setup_selection": SETUP_SELECTION_SEED,
    "policy": POLICY_SEED,
    "action_sampling": ACTION_SAMPLING_SEED,
    "training_order": TRAINING_ORDER_SEED,
}


# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------

DOMAIN_SETUP_ROOT = "setup_root"
DOMAIN_RED_SETUP = "red_setup"
DOMAIN_BLUE_SETUP = "blue_setup"
DOMAIN_RED_POLICY = "policy:red"
DOMAIN_BLUE_POLICY = "policy:blue"
DOMAIN_ACTION_SAMPLING = "action_sampling"
DOMAIN_TRAINING_ORDER = "training_order"

STREAM_DOMAINS = (
    DOMAIN_SETUP_ROOT,
    DOMAIN_RED_SETUP,
    DOMAIN_BLUE_SETUP,
    DOMAIN_RED_POLICY,
    DOMAIN_BLUE_POLICY,
    DOMAIN_ACTION_SAMPLING,
    DOMAIN_TRAINING_ORDER,
)

DOMAIN_ROOTS = {
    DOMAIN_SETUP_ROOT: SETUP_SELECTION_SEED,
    DOMAIN_RED_SETUP: SETUP_SELECTION_SEED,
    DOMAIN_BLUE_SETUP: SETUP_SELECTION_SEED,
    DOMAIN_RED_POLICY: POLICY_SEED,
    DOMAIN_BLUE_POLICY: POLICY_SEED,
    DOMAIN_ACTION_SAMPLING: ACTION_SAMPLING_SEED,
    DOMAIN_TRAINING_ORDER: TRAINING_ORDER_SEED,
}

#: The BLAKE2b personalization of every Phase 14 stream. Eight bytes, distinct
#: from every other phase's, which is the mechanical reason a Phase 14 stream
#: cannot equal a Phase 9 or Phase 10B one even under identical inputs.
_PERSON = b"strat14_"


def derive_seed(domain: str, *parts: "int | str") -> int:
    """One 63-bit stream seed from a domain and its identity inputs."""
    if domain not in STREAM_DOMAINS:
        raise Phase14SeedError(
            f"unknown Phase 14 stream domain {domain!r}; expected one of "
            f"{list(STREAM_DOMAINS)}"
        )
    material = "|".join(
        [str(MASTER_SEED), str(DOMAIN_ROOTS[domain]), domain, *(str(part) for part in parts)]
    )
    digest = hashlib.blake2b(material.encode(), digest_size=8, person=_PERSON).digest()
    return int.from_bytes(digest, "big") >> 1


# ---------------------------------------------------------------------------
# Logical game identity
# ---------------------------------------------------------------------------

#: Four iteration digits, not three: a 168-hour run at the measured Phase 9
#: loop rate is several hundred iterations, and a format that could overflow
#: mid-run would be a format that decides when the run ends.
_GAME_ID_PATTERN = re.compile(
    r"^(?P<version>[a-z0-9_]+)\|ms=(?P<master>[0-9]+)\|ns=(?P<namespace>[a-z0-9_]+)"
    r"\|it=(?P<iteration>[0-9]{4})\|b=(?P<bucket>[a-z]+)\|g=(?P<ordinal>[0-9]{4})$"
)

MIN_ITERATION = 1
MAX_ITERATION_FORMAT = 9999
MAX_ORDINAL_FORMAT = 9999


def game_id(iteration: int, bucket: str, ordinal: int) -> str:
    """The stable identifier of one logical Phase 14 rollout game.

    A pure function of the rollout version, the frozen master seed, the run
    namespace, the 1-based iteration, the population bucket and the per-bucket
    ordinal. Worker count, process partitioning, arrival order and resume
    boundaries appear nowhere, which is what makes crash regeneration exact.

    The *segment* is deliberately absent: it decides how many ordinals a bucket
    has, not what a given ordinal is called, and a segment-stamped id would
    make the same logical game unaddressable from a checkpoint written under
    the other segment.
    """
    if bucket not in POPULATION_BUCKETS:
        raise Phase14SeedError(
            f"unknown population bucket {bucket!r}; expected one of "
            f"{list(POPULATION_BUCKETS)}"
        )
    if not isinstance(iteration, int) or isinstance(iteration, bool):
        raise Phase14SeedError(f"iteration must be an int, got {type(iteration).__name__}")
    if not MIN_ITERATION <= iteration <= MAX_ITERATION_FORMAT:
        raise Phase14SeedError(
            f"iteration {iteration} is outside {MIN_ITERATION}..{MAX_ITERATION_FORMAT}"
        )
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise Phase14SeedError(f"game ordinal must be an int, got {type(ordinal).__name__}")
    if not 0 <= ordinal <= MAX_ORDINAL_FORMAT:
        raise Phase14SeedError(f"game ordinal {ordinal} is outside 0..{MAX_ORDINAL_FORMAT}")
    return (
        f"{PHASE14_ROLLOUT_VERSION}|ms={MASTER_SEED}|ns={PHASE14_NAMESPACE}"
        f"|it={iteration:04d}|b={bucket}|g={ordinal:04d}"
    )


def parse_game_id(identifier: str) -> dict:
    """The identity fields of a Phase 14 rollout id, validated.

    Raises on anything that is not exactly a well-formed id of this rollout
    version under the frozen master seed, so a foreign, Phase 9 or tampered
    identifier can never be mistaken for a Phase 14 rollout game.
    """
    match = _GAME_ID_PATTERN.match(str(identifier))
    if match is None:
        raise Phase14SeedError(f"malformed Phase 14 game id: {identifier!r}")
    fields = match.groupdict()
    if fields["version"] != PHASE14_ROLLOUT_VERSION:
        raise Phase14SeedError(
            f"game id names rollout version {fields['version']!r}, expected "
            f"{PHASE14_ROLLOUT_VERSION!r}"
        )
    if int(fields["master"]) != MASTER_SEED:
        raise Phase14SeedError(
            f"game id names master seed {fields['master']}, expected {MASTER_SEED}"
        )
    if fields["namespace"] != PHASE14_NAMESPACE:
        raise Phase14SeedError(f"game id names unknown namespace {fields['namespace']!r}")
    if fields["bucket"] not in POPULATION_BUCKETS:
        raise Phase14SeedError(f"game id names unknown bucket {fields['bucket']!r}")
    iteration = int(fields["iteration"])
    if iteration < MIN_ITERATION:
        raise Phase14SeedError(f"game id iteration {iteration} is below {MIN_ITERATION}")
    return {
        "rollout_version": fields["version"],
        "phase14_master_seed": int(fields["master"]),
        "namespace": fields["namespace"],
        "iteration": iteration,
        "bucket": fields["bucket"],
        "ordinal": int(fields["ordinal"]),
    }


def validate_ordinal(identifier: str, segment: str) -> dict:
    """Parse an id and refuse an ordinal outside its bucket in `segment`.

    The parser accepts any ordinal its format can carry so the format check
    stays in one place; this is where the *contract* check lives, and it needs
    the segment because the current/historical split is what changes at the
    frozen transition.
    """
    fields = parse_game_id(identifier)
    try:
        counts = bucket_counts(segment)
    except Phase14ContractError as error:
        raise Phase14SeedError(str(error)) from error
    limit = counts[fields["bucket"]]
    if not 0 <= fields["ordinal"] < limit:
        raise Phase14SeedError(
            f"{identifier}: ordinal {fields['ordinal']} is outside 0..{limit - 1} "
            f"for bucket {fields['bucket']!r} in the {segment} segment"
        )
    return fields


# ---------------------------------------------------------------------------
# Per-game domain-separated stream seeds
# ---------------------------------------------------------------------------


def setup_root_seed(identifier: str) -> int:
    """The `root_seed` recorded in one game's trajectory.

    Phase 14 draws each side through its own selector stream, so this value is
    an identity the accepted store expects rather than the source of the draw.
    """
    parse_game_id(identifier)
    return derive_seed(DOMAIN_SETUP_ROOT, identifier)


def side_selector_seed(identifier: str, color: str) -> int:
    """The setup-selector seed of one side of one logical game.

    Red and blue descend from two different frozen domains, so one side's
    arrangement is not a function of the other's.
    """
    parse_game_id(identifier)
    if color == "red":
        return derive_seed(DOMAIN_RED_SETUP, identifier)
    if color == "blue":
        return derive_seed(DOMAIN_BLUE_SETUP, identifier)
    raise Phase14SeedError(f"unknown colour {color!r}; expected 'red' or 'blue'")


def policy_seed(identifier: str, color: str) -> int:
    """The match-level seed of a handcrafted policy seated at one colour."""
    parse_game_id(identifier)
    if color == "red":
        return derive_seed(DOMAIN_RED_POLICY, identifier)
    if color == "blue":
        return derive_seed(DOMAIN_BLUE_POLICY, identifier)
    raise Phase14SeedError(f"unknown colour {color!r}; expected 'red' or 'blue'")


def action_sampling_seed(identifier: str, ply: int) -> int:
    """The action-sampling stream of one neural decision.

    Exactly one player acts at each ply, so `(game_id, ply)` names one
    decision. Current-policy and historical sides both draw from this stream
    when they act; the acting side is fixed by the game, so they cannot
    collide.
    """
    parse_game_id(identifier)
    if not isinstance(ply, int) or isinstance(ply, bool) or ply < 0:
        raise Phase14SeedError(f"ply must be a non-negative int, got {ply!r}")
    return derive_seed(DOMAIN_ACTION_SAMPLING, identifier, ply)


def action_sampling_uniform(identifier: str, ply: int) -> float:
    """The frozen uniform in ``(0, 1]`` behind one behavior action draw.

    The accepted Phase 9 orientation: half-open ``(0, 1]`` so a
    probability-zero prefix is unselectable.
    """
    return (action_sampling_seed(identifier, ply) + 1) / (2**63 + 1)


def train_order_seed(iteration: int, epoch: int) -> int:
    """The shuffle seed of one optimizer epoch over one sealed rollout."""
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 1:
        raise Phase14SeedError(f"iteration must be an int >= 1, got {iteration!r}")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise Phase14SeedError(f"epoch must be a non-negative int, got {epoch!r}")
    return derive_seed(DOMAIN_TRAINING_ORDER, iteration, epoch)


# ---------------------------------------------------------------------------
# The seed contract
# ---------------------------------------------------------------------------

DOMAIN_CONSUMERS = {
    DOMAIN_SETUP_ROOT: "the trajectory's recorded root seed",
    DOMAIN_RED_SETUP: "red's phase14_setup_source selector seed",
    DOMAIN_BLUE_SETUP: "blue's phase14_setup_source selector seed",
    DOMAIN_RED_POLICY: "a handcrafted policy seated as red",
    DOMAIN_BLUE_POLICY: "a handcrafted policy seated as blue",
    DOMAIN_ACTION_SAMPLING: "the behavior action draw of one neural decision",
    DOMAIN_TRAINING_ORDER: "the minibatch shuffle of one optimizer epoch",
}


def seed_contract() -> dict:
    return {
        "seed_contract_version": "phase14_seed_v1",
        "namespace": PHASE14_NAMESPACE,
        "rollout_version": PHASE14_ROLLOUT_VERSION,
        "roots": dict(ROOT_SEEDS),
        "personalization": _PERSON.decode(),
        "domains": {
            domain: {
                "root": DOMAIN_ROOTS[domain],
                "consumer": DOMAIN_CONSUMERS[domain],
            }
            for domain in STREAM_DOMAINS
        },
        "game_id_format": (
            f"{PHASE14_ROLLOUT_VERSION}|ms={MASTER_SEED}|ns={PHASE14_NAMESPACE}"
            "|it=NNNN|b=<bucket>|g=NNNN"
        ),
        "no_stream_for": (
            "historical-member assignment, which the frozen pool contract "
            "partitions exactly rather than sampling"
        ),
        "purity": (
            "no clock, no path, no environment, no worker count and no arrival "
            "order reaches any function in this module"
        ),
    }


def seed_contract_digest() -> str:
    import json

    return hashlib.sha256(
        json.dumps(seed_contract(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
