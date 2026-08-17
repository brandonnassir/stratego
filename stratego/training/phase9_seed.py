"""Phase 9 Agent 1: frozen RL seeds, rollout-game identity, and stream derivation.

Specification sources:

- `01_AGENT_1_RL_CONTRACT_AND_EVAL_BANKS.md` ("Required decisions to freeze",
  seed derivations for the Agent 2 handoff)
- `00_PHASE_9_SEQUENCE_AND_COMMON_CONTRACT.md` ("Canonical Phase 9 seeds",
  "Population mixture", rollout state machine)

What lives here and why
-----------------------
Everything in this module is *identity*: the eight frozen Phase 9 seeds, the
rollout-game identifier, and the domain-separated per-game / per-decision /
per-epoch stream seeds. None of it knows what a population bucket's size is,
what an opponent is, or what a loss looks like —
:mod:`stratego.training.phase9_contract` layers the frozen learning-design
contract on top of these identities and is the intended importer. Keeping
identity free of contract knowledge means a seed can never depend on a
measurement, exactly as `warmstart_seed` did for Phase 8.

Seeds were chosen before any Phase 9 rollout, pilot, or model result existed.
They follow the repository's date-seed precedent (`20260101` Phase 4 bank,
`20260813` Phase 7 library, `20260813xx` Phase 8): the Phase 9 block is the
freeze date `20260816` extended with a two-digit role suffix, fixed by the
calendar rather than by anything measured.

Derivation
----------
All streams come from :func:`derive_phase9_seed`, a domain-separated
``blake2b`` hash under the Phase 9 personalization tag ``strat-rl9`` —
distinct from every earlier tag (``strat-ws8`` Phase 8, ``strat-lb7``/
``strat-at7``/``strat-st7`` Phase 7, ``strat-bnk``/``strat-sid`` Phase 4
bank, ``strat-pls``/``strat-dec`` match/decision seeds, ``strat-unt``/
``strat-mch`` match identity), so no Phase 9 stream can collide with any
accepted upstream stream. There is no global RNG cursor anywhere: every
consumer receives a pure function of a logical identity and a frozen domain
string. Worker count, process partitioning, arrival order and resume
boundaries appear in no derivation, which is what makes crash-regeneration
of a missing game exact.

The frozen per-game domains are:

```text
setup_root              the setup-source root seed of one rollout game
opponent:historical     the active-window archive draw of one historical game
policy:red              the red rule/stress policy match-level seed
policy:blue             the blue rule/stress policy match-level seed
behavior_sampler        the per-decision behavior action-sampling stream
train_order             the per-(namespace, iteration, epoch) shuffle stream
eval_bank               the per-(bank, family, case, side, attempt) bank draws
```

Per-ply rule/stress-policy randomness stays on the frozen Phase 4 path: a
policy receives its match-level seed from here and each ply's stream is
``derive_decision_seed(policy_seed, ply)`` exactly as accepted in Phase 4.
"""

from __future__ import annotations

import hashlib
import re

#: The Phase 9 rollout-game identity version. A change to the game-id format,
#: the seed derivation, or the namespace tokens is a new version, never a
#: silent edit.
PHASE9_ROLLOUT_VERSION = "phase9_rollout_v1"

# ---------------------------------------------------------------------------
# Canonical Phase 9 seeds — frozen by the common contract before any
# trainable Phase 9 rollout exists, recorded here first.
# ---------------------------------------------------------------------------

#: Master seed of the whole phase. Folded into every rollout game id.
PHASE9_MASTER_SEED = 2026081601

#: Reserved root of schedule-level randomness. `phase9_rollout_schedule_v1`
#: is deliberately pure arithmetic (exact scheduled counts, subranges, and
#: parity rules — counts are scheduled, never sampled), so this seed is
#: frozen but unconsumed in v1; a future schedule that needs a draw must
#: derive it from here under a new schedule version.
ROLLOUT_SCHEDULE_SEED = 2026081602

#: Root of the historical-opponent selection streams.
OPPONENT_SCHEDULE_SEED = 2026081603

#: Root of the training-order/shuffle streams (one per namespace/iteration/epoch).
TRAIN_ORDER_SEED = 2026081604

#: Namespace roots keeping the six pilot runs' streams and the canonical
#: run's streams disjoint from each other and from everything above.
PILOT_NAMESPACE_SEED = 2026081605
CANONICAL_NAMESPACE_SEED = 2026081606

#: Frozen game-level bootstrap base seeds for the held-out evaluation banks.
#: A matchup's interval seed is `matchup_seed(bootstrap_seed, matchup_token)`
#: through the frozen Phase 4 statistics derivation.
VALIDATION_BOOTSTRAP_SEED = 2026081607
TEST_BOOTSTRAP_SEED = 2026081608

CANONICAL_PHASE9_SEEDS = {
    "phase9_master_seed": PHASE9_MASTER_SEED,
    "rollout_schedule_seed": ROLLOUT_SCHEDULE_SEED,
    "opponent_schedule_seed": OPPONENT_SCHEDULE_SEED,
    "train_order_seed": TRAIN_ORDER_SEED,
    "pilot_namespace_seed": PILOT_NAMESPACE_SEED,
    "canonical_namespace_seed": CANONICAL_NAMESPACE_SEED,
    "validation_bootstrap_seed": VALIDATION_BOOTSTRAP_SEED,
    "test_bootstrap_seed": TEST_BOOTSTRAP_SEED,
}

# ---------------------------------------------------------------------------
# Stream domains
# ---------------------------------------------------------------------------

#: blake2b personalization of every Phase 9 stream.
_PHASE9_SEED_PERSON = b"strat-rl9"

DOMAIN_SETUP_ROOT = "setup_root"
DOMAIN_HISTORICAL_OPPONENT = "opponent:historical"
DOMAIN_RED_POLICY = "policy:red"
DOMAIN_BLUE_POLICY = "policy:blue"
DOMAIN_BEHAVIOR_SAMPLER = "behavior_sampler"
DOMAIN_TRAIN_ORDER = "train_order"
DOMAIN_EVAL_BANK = "eval_bank"

STREAM_DOMAINS = (
    DOMAIN_SETUP_ROOT,
    DOMAIN_HISTORICAL_OPPONENT,
    DOMAIN_RED_POLICY,
    DOMAIN_BLUE_POLICY,
    DOMAIN_BEHAVIOR_SAMPLER,
    DOMAIN_TRAIN_ORDER,
    DOMAIN_EVAL_BANK,
)


class Phase9SeedError(ValueError):
    """Raised when a Phase 9 identity or seed request is malformed."""


def derive_phase9_seed(domain: str, *parts: "int | str") -> int:
    """A 63-bit deterministic seed for one Phase 9 stream.

    ``domain`` must be one of :data:`STREAM_DOMAINS`; ``parts`` are the
    identity inputs of the stream. The payload is the colon-joined text of
    the rollout version, domain and parts under the ``strat-rl9``
    personalization, so equal identities always agree and any change to any
    identity input yields an unrelated stream.
    """
    if domain not in STREAM_DOMAINS:
        raise Phase9SeedError(f"unknown Phase 9 stream domain: {domain!r}")
    for part in parts:
        if not isinstance(part, (int, str)) or isinstance(part, bool):
            raise Phase9SeedError(
                f"stream identity parts must be int or str, got {type(part).__name__}"
            )
    payload = ":".join([PHASE9_ROLLOUT_VERSION, domain, *[str(part) for part in parts]])
    digest = hashlib.blake2b(
        payload.encode(), digest_size=8, person=_PHASE9_SEED_PERSON
    ).digest()
    return int.from_bytes(digest, "big") >> 1


# ---------------------------------------------------------------------------
# Run namespaces
# ---------------------------------------------------------------------------

#: The canonical production run's namespace token.
CANONICAL_NAMESPACE = "canonical"

#: The six pilot namespace tokens, one per frozen pilot candidate.
PILOT_NAMESPACES = (
    "pilot_p9a",
    "pilot_p9b",
    "pilot_p9c",
    "pilot_p9d",
    "pilot_p9e",
    "pilot_p9f",
)

RUN_NAMESPACES = (CANONICAL_NAMESPACE,) + PILOT_NAMESPACES


def namespace_seed(namespace: str) -> int:
    """The frozen namespace root seed of one Phase 9 run namespace."""
    if namespace == CANONICAL_NAMESPACE:
        return CANONICAL_NAMESPACE_SEED
    if namespace in PILOT_NAMESPACES:
        return PILOT_NAMESPACE_SEED
    raise Phase9SeedError(
        f"unknown Phase 9 namespace {namespace!r}; expected one of {list(RUN_NAMESPACES)}"
    )


# ---------------------------------------------------------------------------
# Rollout-game identity
# ---------------------------------------------------------------------------

#: The four population buckets of the frozen mixture. Bucket sizes, opponent
#: maps and colour balance live in `phase9_contract`; the identifier only
#: names which bucket a logical game belongs to.
BUCKET_CURRENT = "current"
BUCKET_HISTORICAL = "historical"
BUCKET_RULE = "rule"
BUCKET_STRESS = "stress"
POPULATION_BUCKETS = (BUCKET_CURRENT, BUCKET_HISTORICAL, BUCKET_RULE, BUCKET_STRESS)

_GAME_ID_PATTERN = re.compile(
    r"^(?P<version>[a-z0-9_]+)\|ms=(?P<master>[0-9]+)\|ns=(?P<namespace>[a-z0-9_]+)"
    r"\|it=(?P<iteration>[0-9]{3})\|b=(?P<bucket>[a-z]+)\|g=(?P<ordinal>[0-9]{4})$"
)

#: Iterations are 1-based: iteration k trains on the k-th sealed rollout.
MIN_ITERATION = 1
MAX_ITERATION_FORMAT = 999
MAX_ORDINAL_FORMAT = 9999


def phase9_game_id(namespace: str, iteration: int, bucket: str, ordinal: int) -> str:
    """The stable identifier of one logical Phase 9 rollout game.

    A pure function of exactly the identity fields the common contract
    requires — rollout version, frozen master seed, run namespace, 1-based
    RL iteration, population bucket, and the per-bucket game ordinal — in a
    fixed, parseable ``key=value`` pipe format:

    ```text
    phase9_rollout_v1|ms=2026081601|ns=canonical|it=012|b=historical|g=0137
    ```

    Worker count, process partitioning, arrival order and resume boundaries
    appear nowhere, which is what makes the identity schedule-independent
    and crash-regeneration exact. Bucket sizes are the contract's to
    enforce; the identifier accepts any ordinal its format can carry so the
    contract check stays in one place.
    """
    if namespace not in RUN_NAMESPACES:
        raise Phase9SeedError(
            f"unknown Phase 9 namespace {namespace!r}; expected one of {list(RUN_NAMESPACES)}"
        )
    if bucket not in POPULATION_BUCKETS:
        raise Phase9SeedError(
            f"unknown population bucket {bucket!r}; expected one of {list(POPULATION_BUCKETS)}"
        )
    if not isinstance(iteration, int) or isinstance(iteration, bool):
        raise Phase9SeedError(f"iteration must be an int, got {type(iteration).__name__}")
    if not MIN_ITERATION <= iteration <= MAX_ITERATION_FORMAT:
        raise Phase9SeedError(
            f"iteration {iteration} is outside {MIN_ITERATION}..{MAX_ITERATION_FORMAT}"
        )
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise Phase9SeedError(f"game ordinal must be an int, got {type(ordinal).__name__}")
    if not 0 <= ordinal <= MAX_ORDINAL_FORMAT:
        raise Phase9SeedError(f"game ordinal {ordinal} is outside 0..{MAX_ORDINAL_FORMAT}")
    return (
        f"{PHASE9_ROLLOUT_VERSION}|ms={PHASE9_MASTER_SEED}|ns={namespace}"
        f"|it={iteration:03d}|b={bucket}|g={ordinal:04d}"
    )


def parse_phase9_game_id(game_id: str) -> dict:
    """The identity fields of a Phase 9 rollout game id, validated.

    Raises on anything that is not exactly a well-formed id of this rollout
    version under the frozen master seed, so a foreign or tampered
    identifier can never be mistaken for a Phase 9 rollout game.
    """
    match = _GAME_ID_PATTERN.match(game_id)
    if match is None:
        raise Phase9SeedError(f"malformed Phase 9 game id: {game_id!r}")
    fields = match.groupdict()
    if fields["version"] != PHASE9_ROLLOUT_VERSION:
        raise Phase9SeedError(
            f"game id names rollout version {fields['version']!r}, expected "
            f"{PHASE9_ROLLOUT_VERSION!r}"
        )
    if int(fields["master"]) != PHASE9_MASTER_SEED:
        raise Phase9SeedError(
            f"game id names master seed {fields['master']}, expected {PHASE9_MASTER_SEED}"
        )
    if fields["namespace"] not in RUN_NAMESPACES:
        raise Phase9SeedError(f"game id names unknown namespace {fields['namespace']!r}")
    if fields["bucket"] not in POPULATION_BUCKETS:
        raise Phase9SeedError(f"game id names unknown bucket {fields['bucket']!r}")
    iteration = int(fields["iteration"])
    if iteration < MIN_ITERATION:
        raise Phase9SeedError(f"game id iteration {iteration} is below {MIN_ITERATION}")
    return {
        "rollout_version": fields["version"],
        "phase9_master_seed": int(fields["master"]),
        "namespace": fields["namespace"],
        "iteration": iteration,
        "bucket": fields["bucket"],
        "ordinal": int(fields["ordinal"]),
    }


# ---------------------------------------------------------------------------
# Per-game domain-separated stream seeds
# ---------------------------------------------------------------------------


def setup_root_seed(game_id: str) -> int:
    """The setup-source root seed of one logical rollout game.

    Passed as ``root_seed`` to the frozen `setup_source_v1` ``assign`` call
    with the frozen constants ``environment_id=0, generation=0`` (exactly
    the accepted Phase 8 shape); the sampler's own accepted side derivation
    then gives red and blue their independent domain-separated draw streams.
    """
    parse_phase9_game_id(game_id)
    return derive_phase9_seed(DOMAIN_SETUP_ROOT, game_id)


def historical_opponent_seed(game_id: str) -> int:
    """The active-window archive draw stream of one historical-bucket game.

    Only defined for the historical bucket: every other bucket's opponent is
    fixed by schedule arithmetic, and refusing the call is what keeps an
    accidental seeded draw out of those buckets.
    """
    fields = parse_phase9_game_id(game_id)
    if fields["bucket"] != BUCKET_HISTORICAL:
        raise Phase9SeedError(
            f"historical opponent seeds exist for the {BUCKET_HISTORICAL!r} bucket "
            f"only, not {fields['bucket']!r}"
        )
    return derive_phase9_seed(
        DOMAIN_HISTORICAL_OPPONENT, OPPONENT_SCHEDULE_SEED, game_id
    )


def red_policy_seed(game_id: str) -> int:
    """The match-level seed of a red rule/stress policy in one rollout game."""
    parse_phase9_game_id(game_id)
    return derive_phase9_seed(DOMAIN_RED_POLICY, game_id)


def blue_policy_seed(game_id: str) -> int:
    """The match-level seed of a blue rule/stress policy in one rollout game."""
    parse_phase9_game_id(game_id)
    return derive_phase9_seed(DOMAIN_BLUE_POLICY, game_id)


def game_seeds(game_id: str) -> dict:
    """Every per-game stream seed, keyed by its contract name."""
    fields = parse_phase9_game_id(game_id)
    seeds = {
        "setup_root_seed": setup_root_seed(game_id),
        "red_policy_seed": red_policy_seed(game_id),
        "blue_policy_seed": blue_policy_seed(game_id),
    }
    if fields["bucket"] == BUCKET_HISTORICAL:
        seeds["historical_opponent_seed"] = historical_opponent_seed(game_id)
    return seeds


# ---------------------------------------------------------------------------
# Per-decision behavior sampling
# ---------------------------------------------------------------------------


def behavior_sample_seed(game_id: str, ply: int) -> int:
    """The action-sampling stream of one neural decision of one rollout game.

    Exactly one player acts at each ply, so `(game_id, ply)` names one
    decision. Both a current-policy side and a historical-snapshot side draw
    from this stream when they act; the acting side is fixed by the game, so
    the streams cannot collide.
    """
    parse_phase9_game_id(game_id)
    if not isinstance(ply, int) or isinstance(ply, bool) or ply < 0:
        raise Phase9SeedError(f"ply must be a non-negative int, got {ply!r}")
    return derive_phase9_seed(DOMAIN_BEHAVIOR_SAMPLER, game_id, ply)


def behavior_sample_uniform(game_id: str, ply: int) -> float:
    """The frozen uniform in ``(0, 1]`` behind one behavior action draw.

    `phase9_contract` freezes the selection rule: walk the legal actions in
    ascending action-id order accumulating their behavior probabilities and
    select the first action whose cumulative probability reaches the
    uniform; a float32 tail shortfall selects the last legal action. The
    half-open ``(0, 1]`` orientation makes probability-zero prefixes
    unselectable.
    """
    return (behavior_sample_seed(game_id, ply) + 1) / (2**63 + 1)


# ---------------------------------------------------------------------------
# Training-order and evaluation-bank streams
# ---------------------------------------------------------------------------


def train_order_seed(namespace: str, iteration: int, epoch: int) -> int:
    """The shuffle seed of one optimizer epoch over one sealed rollout.

    Domain-separated by namespace root, namespace token, 1-based iteration
    and 0-based epoch, so no two epochs anywhere in Phase 9 share an order
    stream.
    """
    root = namespace_seed(namespace)
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < MIN_ITERATION:
        raise Phase9SeedError(f"iteration must be an int >= {MIN_ITERATION}, got {iteration!r}")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise Phase9SeedError(f"epoch must be a non-negative int, got {epoch!r}")
    return derive_phase9_seed(
        DOMAIN_TRAIN_ORDER, TRAIN_ORDER_SEED, root, namespace, int(iteration), int(epoch)
    )


def eval_bank_draw_seed(
    bank_version: str, family_id: str, case_ordinal: int, side: str, attempt: int
) -> int:
    """The frozen sampler draw seed of one bank-case side attempt.

    `phase9_banks` resolves one family-balanced case side by walking
    ``attempt = 0, 1, 2, ...`` through the frozen `setup_sampler_v1` and
    accepting the first draw whose primary family matches the case family.
    Every input is part of the stream identity, so any case side of either
    bank can be rebuilt in isolation.
    """
    if side not in ("red", "blue"):
        raise Phase9SeedError(f"side must be 'red' or 'blue', got {side!r}")
    if not isinstance(case_ordinal, int) or isinstance(case_ordinal, bool) or case_ordinal < 0:
        raise Phase9SeedError(f"case ordinal must be a non-negative int, got {case_ordinal!r}")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
        raise Phase9SeedError(f"attempt must be a non-negative int, got {attempt!r}")
    return derive_phase9_seed(
        DOMAIN_EVAL_BANK, str(bank_version), str(family_id), case_ordinal, side, attempt
    )


def bootstrap_seed(bank: str) -> int:
    """The frozen game-level bootstrap base seed of one held-out bank."""
    if bank == "validation":
        return VALIDATION_BOOTSTRAP_SEED
    if bank == "test":
        return TEST_BOOTSTRAP_SEED
    raise Phase9SeedError(
        f"bootstrap seeds exist for the held-out banks only, not {bank!r}"
    )


__all__ = [
    "BUCKET_CURRENT",
    "BUCKET_HISTORICAL",
    "BUCKET_RULE",
    "BUCKET_STRESS",
    "CANONICAL_NAMESPACE",
    "CANONICAL_NAMESPACE_SEED",
    "CANONICAL_PHASE9_SEEDS",
    "DOMAIN_BEHAVIOR_SAMPLER",
    "DOMAIN_EVAL_BANK",
    "DOMAIN_HISTORICAL_OPPONENT",
    "DOMAIN_RED_POLICY",
    "DOMAIN_BLUE_POLICY",
    "DOMAIN_SETUP_ROOT",
    "DOMAIN_TRAIN_ORDER",
    "MAX_ITERATION_FORMAT",
    "MAX_ORDINAL_FORMAT",
    "MIN_ITERATION",
    "OPPONENT_SCHEDULE_SEED",
    "PHASE9_MASTER_SEED",
    "PHASE9_ROLLOUT_VERSION",
    "PILOT_NAMESPACES",
    "PILOT_NAMESPACE_SEED",
    "POPULATION_BUCKETS",
    "ROLLOUT_SCHEDULE_SEED",
    "RUN_NAMESPACES",
    "STREAM_DOMAINS",
    "TEST_BOOTSTRAP_SEED",
    "TRAIN_ORDER_SEED",
    "VALIDATION_BOOTSTRAP_SEED",
    "Phase9SeedError",
    "behavior_sample_seed",
    "behavior_sample_uniform",
    "blue_policy_seed",
    "bootstrap_seed",
    "derive_phase9_seed",
    "eval_bank_draw_seed",
    "game_seeds",
    "historical_opponent_seed",
    "namespace_seed",
    "parse_phase9_game_id",
    "phase9_game_id",
    "red_policy_seed",
    "setup_root_seed",
    "train_order_seed",
]
