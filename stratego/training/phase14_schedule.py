"""Phase 14: the population mixer and the per-iteration schedule.

Specification source: the frozen `opponent_mixture` and
`historical_archive_and_active_pool` blocks, via
`02_AGENT_2_FINAL_TRAINING_INTEGRATION.md` section 6.

What a Phase 14 iteration is
----------------------------
2,048 logical games whose composition is decided by two things and nothing
else: the frozen mixture of the segment the iteration was *launched* in, and
the active pool as of that launch. There is no adaptive reweighting because no
result reaches this module: a game's opponent is a function of its ordinal.

The mixture, spelled as ordinals
--------------------------------
Contiguous frozen subranges in the frozen order — current, historical,
strategic, tactical, scout-rush, miner-rush, information-miser — carried by two
store buckets (`rule` holds the two rule tiers, `stress` the three stress
policies) so the accepted rollout store sees the shapes it already audits.

Main segment: current 1,188 | historical 615 | rule 122 | stress 123.
Late segment: current 819 | historical 984 | rule 122 | stress 123.

Only the current/historical split moves at the frozen transition; the
handcrafted share is identical in both segments, which is what "the handcrafted
share remains exactly as frozen" means operationally.

Why `segment` and `pool` are arguments
--------------------------------------
Both are *run state*, not arithmetic: which segment an iteration belongs to
depends on when it was launched against the original clock, and the pool
depends on how many snapshots existed then. Passing them in — rather than
letting this module read a clock or an archive — is what makes a scheduled
record reproducible from a crashed run's checkpoint instead of from the time
the resume happens to occur.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from ..engine.constants import PLAYER_NAMES
from ..evaluation.registry import POLICY_INDEX
from .phase14_contract import (
    BUCKET_CURRENT,
    BUCKET_HISTORICAL,
    BUCKET_RULE,
    BUCKET_STRESS,
    HANDCRAFTED_COUNTS,
    HANDCRAFTED_FAMILY_ORDER,
    PHASE14_NAMESPACE,
    PHASE14_POPULATION_VERSION,
    PHASE14_SCHEDULE_VERSION,
    POPULATION_BUCKETS,
    PRODUCTION_POPULATION,
    SEGMENTS,
    Phase14ContractError,
    Population,
    bucket_counts,
    learner_color,
    learner_control_for,
    require_segment,
)
from .phase14_pool import (
    ActivePool,
    Phase14PoolError,
    historical_policy_token,
    member_for_ordinal,
    member_ordinal_ranges,
    realized_shares,
)
from .phase14_seed import game_id as phase14_game_id
from .phase14_seed import parse_game_id, policy_seed, setup_root_seed

BEHAVIOR_TOKEN_PREFIX = "phase14_behavior_v1"

LEARNER_CONTROL_BOTH = "both"


class Phase14ScheduleError(RuntimeError):
    """Raised when a Phase 14 schedule request is invalid."""


# ---------------------------------------------------------------------------
# Policy identity tokens
# ---------------------------------------------------------------------------


def behavior_snapshot_identity(iteration: int) -> str:
    """`B0012` is "the learner frozen at the start of iteration 12"."""
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 1:
        raise Phase14ScheduleError(f"iteration must be an int >= 1, got {iteration!r}")
    return f"B{iteration:04d}"


def behavior_policy_token(iteration: int) -> str:
    return f"{BEHAVIOR_TOKEN_PREFIX}|{behavior_snapshot_identity(iteration)}"


def handcrafted_policy_token(policy_id: str) -> str:
    """The frozen Phase 4 `id@version` token of a rule or stress policy."""
    if policy_id not in POLICY_INDEX:
        raise Phase14ScheduleError(f"unknown frozen policy id: {policy_id!r}")
    return f"{policy_id}@{POLICY_INDEX[policy_id].policy_version}"


def opponent_kind_for(bucket: str) -> str:
    if bucket == BUCKET_CURRENT:
        return "current_policy"
    if bucket == BUCKET_HISTORICAL:
        return "historical_snapshot"
    if bucket == BUCKET_RULE:
        return "rule_policy"
    if bucket == BUCKET_STRESS:
        return "stress_policy"
    raise Phase14ScheduleError(f"unknown bucket: {bucket!r}")


# ---------------------------------------------------------------------------
# One scheduled logical game
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase14ScheduledGame:
    """Everything `phase14_schedule_v1` fixes about one logical game.

    Carries no privileged information: no piece truth, no setup contents, no
    outcome. The setup *identity* is here; resolving it to boards is the
    collector's job through the frozen Phase 14 source.
    """

    rollout_game_id: str
    run_namespace: str
    rl_iteration: int
    game_ordinal: int
    bucket: str
    segment: str
    red_policy_identity: str
    blue_policy_identity: str
    learner_control: str
    learner_color: "str | None"
    behavior_snapshot_identity: str
    historical_snapshot_identity: "str | None"
    opponent_kind: str
    opponent_identity: str
    opponent_checkpoint_digest: "str | None"
    handcrafted_policy_id: "str | None"
    setup_root_seed: int
    red_setup_source_identity: str
    blue_setup_source_identity: str
    red_policy_seed: "int | None"
    blue_policy_seed: "int | None"

    @property
    def phase9_game_id(self) -> str:
        """The id under the attribute name the accepted collector reads.

        Phase 14 reuses the audited Phase 9 collection and store path rather
        than reimplementing it; this alias is the whole of that coupling.
        """
        return self.rollout_game_id

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def learner_sides(self) -> tuple:
        """Which colours receive Phase 14 policy/value/belief loss."""
        if self.learner_control == LEARNER_CONTROL_BOTH:
            return ("red", "blue")
        return (self.learner_control,)


def _player_index(color: str) -> int:
    for player, name in PLAYER_NAMES.items():
        if name == color:
            return player
    raise Phase14ScheduleError(f"unknown colour {color!r}")


def scheduled_game_record(
    iteration: int,
    bucket: str,
    ordinal: int,
    *,
    segment: str,
    pool: ActivePool,
    setup_source,
    population: Population = PRODUCTION_POPULATION,
) -> Phase14ScheduledGame:
    """The full scheduled record of one logical game.

    A pure function of `(iteration, bucket, ordinal, segment, pool, population)`:
    the same inputs rebuild the same record on any machine, with no enumeration
    state and no partial rollout in hand.
    """
    require_segment(segment)
    counts = population.bucket_counts(segment)
    if bucket not in counts:
        raise Phase14ScheduleError(f"unknown population bucket: {bucket!r}")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise Phase14ScheduleError(f"ordinal must be an int, got {ordinal!r}")
    if not 0 <= ordinal < counts[bucket]:
        raise Phase14ScheduleError(
            f"ordinal {ordinal} is outside 0..{counts[bucket] - 1} for bucket "
            f"{bucket!r} in the {segment} segment"
        )

    identifier = phase14_game_id(iteration, bucket, ordinal)
    control = learner_control_for(bucket, iteration, ordinal)
    color = learner_color(bucket, iteration, ordinal)
    learner_token = behavior_policy_token(iteration)

    historical_identity: "str | None" = None
    opponent_digest: "str | None" = None
    handcrafted_id: "str | None" = None
    red_seed: "int | None" = None
    blue_seed: "int | None" = None

    if bucket == BUCKET_CURRENT:
        opponent_token = learner_token
    elif bucket == BUCKET_HISTORICAL:
        try:
            historical_identity = member_for_ordinal(
                ordinal, counts[BUCKET_HISTORICAL], pool, iteration
            )
        except Phase14PoolError as error:
            raise Phase14ScheduleError(str(error)) from error
        opponent_token = historical_policy_token(historical_identity)
        opponent_digest = pool.checkpoint_for(historical_identity)["sha256"]
    else:
        try:
            handcrafted_id = population.handcrafted_policy_for_ordinal(bucket, ordinal)
        except Phase14ContractError as error:
            raise Phase14ScheduleError(str(error)) from error
        opponent_token = handcrafted_policy_token(handcrafted_id)

    if color is None:
        red_identity = blue_identity = learner_token
    elif color == "red":
        red_identity, blue_identity = learner_token, opponent_token
    else:
        red_identity, blue_identity = opponent_token, learner_token

    # Only a handcrafted side owns a match-level policy RNG stream; the frozen
    # sidecar stores null for every other side, so the schedule must not claim
    # one either.
    if bucket in (BUCKET_RULE, BUCKET_STRESS):
        if color == "red":
            blue_seed = policy_seed(identifier, "blue")
        else:
            red_seed = policy_seed(identifier, "red")

    return Phase14ScheduledGame(
        rollout_game_id=identifier,
        run_namespace=PHASE14_NAMESPACE,
        rl_iteration=int(iteration),
        game_ordinal=int(ordinal),
        bucket=bucket,
        segment=segment,
        red_policy_identity=red_identity,
        blue_policy_identity=blue_identity,
        learner_control=control,
        learner_color=color,
        behavior_snapshot_identity=behavior_snapshot_identity(iteration),
        historical_snapshot_identity=historical_identity,
        opponent_kind=opponent_kind_for(bucket),
        opponent_identity=opponent_token,
        opponent_checkpoint_digest=opponent_digest,
        handcrafted_policy_id=handcrafted_id,
        setup_root_seed=setup_root_seed(identifier),
        red_setup_source_identity=setup_source.side_identity(
            game_id=identifier, player=_player_index("red")
        ),
        blue_setup_source_identity=setup_source.side_identity(
            game_id=identifier, player=_player_index("blue")
        ),
        red_policy_seed=red_seed,
        blue_policy_seed=blue_seed,
    )


def rebuild_scheduled_game(
    identifier: str,
    *,
    segment: str,
    pool: ActivePool,
    setup_source,
    population: Population = PRODUCTION_POPULATION,
) -> Phase14ScheduledGame:
    """Rebuild a scheduled game from its identifier and its run state.

    The resume path: a committed id from a crashed run reconstructs its full
    record given the segment and pool the checkpoint recorded — never given the
    segment and pool that happen to be current when the resume runs.
    """
    fields = parse_game_id(identifier)
    return scheduled_game_record(
        fields["iteration"],
        fields["bucket"],
        fields["ordinal"],
        segment=segment,
        pool=pool,
        setup_source=setup_source,
        population=population,
    )


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def iteration_game_ids(
    iteration: int, segment: str, population: Population = PRODUCTION_POPULATION
) -> tuple:
    """Every logical game id of one iteration, bucket-major then ordinal."""
    counts = population.bucket_counts(segment)
    return tuple(
        phase14_game_id(iteration, bucket, ordinal)
        for bucket in POPULATION_BUCKETS
        for ordinal in range(counts[bucket])
    )


def iter_iteration_schedule(
    iteration: int,
    *,
    segment: str,
    pool: ActivePool,
    setup_source,
    population: Population = PRODUCTION_POPULATION,
):
    counts = population.bucket_counts(segment)
    for bucket in POPULATION_BUCKETS:
        for ordinal in range(counts[bucket]):
            yield scheduled_game_record(
                iteration,
                bucket,
                ordinal,
                segment=segment,
                pool=pool,
                setup_source=setup_source,
                population=population,
            )


# ---------------------------------------------------------------------------
# The mixture, as documents
# ---------------------------------------------------------------------------


def handcrafted_ordinal_ranges(bucket: str) -> tuple:
    """`((policy_id, start, stop), ...)` over one handcrafted bucket."""
    order = [
        name
        for name in HANDCRAFTED_FAMILY_ORDER
        if handcrafted_bucket_of(name) == bucket
    ]
    ranges = []
    cursor = 0
    for name in order:
        count = HANDCRAFTED_COUNTS[name]
        ranges.append((name, cursor, cursor + count))
        cursor += count
    return tuple(ranges)


def handcrafted_bucket_of(policy_id: str) -> str:
    if policy_id not in HANDCRAFTED_COUNTS:
        raise Phase14ScheduleError(f"{policy_id!r} is not a Phase 14 handcrafted family")
    return BUCKET_STRESS if policy_id.startswith("stress_") else BUCKET_RULE


def iteration_mixture(
    iteration: int,
    *,
    segment: str,
    pool: ActivePool,
    population: Population = PRODUCTION_POPULATION,
) -> dict:
    """The exact composition of one iteration, for telemetry and audit.

    Counts, never estimates: this is the same arithmetic the scheduler runs, so
    a telemetry row and the games actually played cannot disagree.
    """
    counts = population.bucket_counts(segment)
    historical = counts[BUCKET_HISTORICAL]
    total = sum(counts.values())
    return {
        "iteration": int(iteration),
        "segment": segment,
        "population": population.to_dict(),
        "games": total,
        "bucket_counts": dict(counts),
        "percentages": {
            "current": counts[BUCKET_CURRENT] / total,
            "historical": historical / total,
            "handcrafted": (counts[BUCKET_RULE] + counts[BUCKET_STRESS]) / total,
        },
        "handcrafted_counts": dict(HANDCRAFTED_COUNTS),
        "historical_members": [
            {"identity": identity, "games": stop - start}
            for identity, start, stop in member_ordinal_ranges(historical, pool, iteration)
        ],
        "historical_categories": realized_shares(historical, pool, iteration),
        "active_pool": pool.to_dict(),
    }


def population_document() -> dict:
    """The frozen population of Phase 14, in both segments."""
    return {
        "population_version": PHASE14_POPULATION_VERSION,
        "schedule_version": PHASE14_SCHEDULE_VERSION,
        "namespace": PHASE14_NAMESPACE,
        "segments": {
            segment: {
                "bucket_counts": bucket_counts(segment),
                "games": sum(bucket_counts(segment).values()),
            }
            for segment in SEGMENTS
        },
        "handcrafted_counts": dict(HANDCRAFTED_COUNTS),
        "ordinal_layout": list(HANDCRAFTED_FAMILY_ORDER),
        "handcrafted_ranges": {
            bucket: [
                {"policy_id": name, "start": start, "stop": stop}
                for name, start, stop in handcrafted_ordinal_ranges(bucket)
            ]
            for bucket in (BUCKET_RULE, BUCKET_STRESS)
        },
        "colour_balance": "the accepted Phase 9 parity rule, unchanged",
        "selection": "scheduled counts only; no sampling, no adaptive reweighting",
        "transition": "the current/historical split changes only at the frozen 132h mark",
    }


def population_digest() -> str:
    return hashlib.sha256(
        json.dumps(population_document(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
