"""Optional Phase 10B: the deterministic population and opponent schedule.

Specification source: `OPTIONAL_PHASE_10B_SETUP_CONDITIONED_FINE_TUNING_AGENT.md`
sections 7, 8 and 12.

What this layer decides
-----------------------
Which logical games exist, who plays them, which colour the learner holds and
which frozen streams each side draws from. It never decides how a neural move
is collected, how a target is built or how an optimizer steps.

Purity and outcome independence
-------------------------------
A scheduled game is a pure function of the rollout version, the frozen Phase
10B roots, the RL iteration, the bucket and the game ordinal, plus the
explicit immutable archive manifest handed in. There is no filesystem access,
no clock and no global RNG here, and no function reads a match result, a win
rate or a league table — the plan forbids performance-based archive weighting,
so outcome-prioritised matchmaking is not implemented at all.

Duck compatibility
------------------
:class:`Phase10BScheduledGame` exposes the field names the accepted Phase 9
collector and rollout store already consume, including the historical
`phase9_game_id` accessor. The alias is deliberate: reusing the accepted,
audited collection path is worth more than a tidier attribute name, and
:attr:`Phase10BScheduledGame.rollout_game_id` is the honest spelling that the
Phase 10B artifacts use.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from ..engine.constants import PLAYER_NAMES
from ..evaluation.registry import POLICY_INDEX
from .phase10b_contract import (
    ANCHOR_IDENTITY,
    BUCKET_ANCHOR,
    BUCKET_ARCHIVE,
    BUCKET_CURRENT,
    BUCKET_OPPONENT,
    LEARNER_CONTROL_BOTH,
    MAX_ITERATIONS,
    PHASE10B_NAMESPACE,
    PHASE10B_POPULATION_VERSION,
    PHASE10B_SCHEDULE_VERSION,
    POPULATION_BUCKETS,
    Phase10BContractError,
    active_archive_window,
    bucket_counts,
    learner_color,
    learner_control_for,
    opponent_counts,
    opponent_policy_for_ordinal,
)
from .phase10b_seed import (
    archive_member_for,
    game_id as phase10b_game_id,
    opponent_policy_seed,
    parse_game_id,
    setup_root_seed,
)

#: Frozen constants of the setup-source call every rollout game makes.
SETUP_ENVIRONMENT_ID = 0
SETUP_GENERATION = 0

BEHAVIOR_TOKEN_PREFIX = "phase10b_behavior"
ARCHIVE_TOKEN_PREFIX = "phase10b_archive"

#: The accepted Phase 9 checkpoint's stored policy token. It names the
#: accepted model, not a Phase 10B variant of it, because that is exactly what
#: the anchor is.
ANCHOR_POLICY_TOKEN = "phase9_selfplay_c1_v1"

#: Which opponent policy ids the plan calls "rule" and which "stress". The
#: split changes no game; it makes the stored `opponent_kind` say what kind of
#: frozen policy the other seat held.
RULE_POLICY_IDS = frozenset(
    {"strategic_rule_based", "tactical_rule_based", "basic_heuristic", "random_legal"}
)


class Phase10BScheduleError(RuntimeError):
    """Raised when a Phase 10B schedule request or audit input is invalid."""


# ---------------------------------------------------------------------------
# Policy tokens
# ---------------------------------------------------------------------------


def behavior_snapshot_identity(iteration: int) -> str:
    """`B012` is "the Phase 10B learner frozen at the start of iteration 12"."""
    if not isinstance(iteration, int) or isinstance(iteration, bool):
        raise Phase10BScheduleError(f"iteration must be an int, got {iteration!r}")
    if not 1 <= iteration <= MAX_ITERATIONS:
        raise Phase10BScheduleError(
            f"iteration {iteration} is outside 1..{MAX_ITERATIONS}"
        )
    return f"B{iteration:03d}"


def behavior_policy_token(iteration: int) -> str:
    return f"{BEHAVIOR_TOKEN_PREFIX}|ns={PHASE10B_NAMESPACE}|{behavior_snapshot_identity(iteration)}"


def history_policy_token(identity: str) -> str:
    """The stored policy token of a checkpoint-opponent side."""
    if identity == ANCHOR_IDENTITY:
        return ANCHOR_POLICY_TOKEN
    return f"{ARCHIVE_TOKEN_PREFIX}|ns={PHASE10B_NAMESPACE}|{identity}"


def opponent_policy_token(policy_id: str) -> str:
    """The frozen Phase 4 `id@version` token of a rule or stress policy."""
    if policy_id not in POLICY_INDEX:
        raise Phase10BScheduleError(f"unknown frozen policy id: {policy_id!r}")
    return f"{policy_id}@{POLICY_INDEX[policy_id].policy_version}"


def opponent_kind_for(policy_id: str) -> str:
    return "rule_policy" if policy_id in RULE_POLICY_IDS else "stress_policy"


# ---------------------------------------------------------------------------
# The active archive manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveArchiveManifest:
    """The immutable archive pool of one iteration, as an explicit input.

    The scheduler does not discover history; it is *given* history. A future
    checkpoint therefore cannot be fabricated to satisfy a schedule, because
    an identity with no entry here cannot be collected.
    """

    iteration: int
    identities: tuple
    checkpoint_digests: tuple = ()

    @classmethod
    def frozen_for(
        cls, iteration: int, checkpoint_digests: "dict | None" = None
    ) -> "ActiveArchiveManifest":
        digests = tuple(
            sorted(
                (str(key), str(value))
                for key, value in (checkpoint_digests or {}).items()
            )
        )
        return cls(
            iteration=int(iteration),
            identities=tuple(active_archive_window(iteration)),
            checkpoint_digests=digests,
        )

    @property
    def digest_map(self) -> dict:
        return dict(self.checkpoint_digests)

    def validate(self) -> None:
        expected = active_archive_window(self.iteration)
        if tuple(self.identities) != expected:
            raise Phase10BScheduleError(
                f"active archive manifest for iteration {self.iteration} is "
                f"{list(self.identities)}, but the frozen window is {list(expected)}"
            )
        unknown = sorted(set(self.digest_map) - set(expected))
        if unknown:
            raise Phase10BScheduleError(
                f"manifest carries digests for identities outside the active "
                f"window: {unknown}"
            )

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "identities": list(self.identities),
            "checkpoint_digests": self.digest_map,
            "window_rule": (
                f"{ANCHOR_IDENTITY} + up to 4 most recent Phase 10B snapshots "
                "archived every 5 committed iterations; the anchor is never "
                "evicted"
            ),
            "selection_rule": "uniform over the active window, outcome-independent",
        }


# ---------------------------------------------------------------------------
# One scheduled logical game
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase10BScheduledGame:
    """Everything `phase10b_rollout_schedule_v1` fixes about one logical game.

    Carries no privileged information: no piece truth, no setup contents, no
    outcome. The setup *identity* is here; resolving it to boards is the
    collector's job through the frozen P10-D source.
    """

    rollout_game_id: str
    run_namespace: str
    rl_iteration: int
    game_ordinal: int
    bucket: str
    red_policy_identity: str
    blue_policy_identity: str
    learner_control: str
    learner_color: "str | None"
    behavior_snapshot_identity: str
    historical_snapshot_identity: "str | None"
    opponent_kind: str
    opponent_identity: str
    opponent_checkpoint_digest: "str | None"
    setup_root_seed: int
    red_setup_source_identity: str
    blue_setup_source_identity: str
    red_policy_seed: "int | None"
    blue_policy_seed: "int | None"
    archive_selection_seed: "int | None"

    @property
    def phase9_game_id(self) -> str:
        """The id under the attribute name the accepted collector reads.

        Phase 10B reuses the audited Phase 9 collection and store path rather
        than reimplementing it; this alias is the whole of that coupling.
        """
        return self.rollout_game_id

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["rollout_game_id"] = self.rollout_game_id
        return payload

    @property
    def learner_sides(self) -> tuple:
        if self.learner_control == LEARNER_CONTROL_BOTH:
            return ("red", "blue")
        return (self.learner_control,)


def scheduled_game_record(
    iteration: int,
    bucket: str,
    ordinal: int,
    *,
    setup_source,
    history: "ActiveArchiveManifest | None" = None,
) -> Phase10BScheduledGame:
    """The full scheduled record of one logical game."""
    counts = bucket_counts()
    if bucket not in counts:
        raise Phase10BScheduleError(f"unknown population bucket: {bucket!r}")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise Phase10BScheduleError(f"ordinal must be an int, got {ordinal!r}")
    if not 0 <= ordinal < counts[bucket]:
        raise Phase10BScheduleError(
            f"ordinal {ordinal} is outside 0..{counts[bucket] - 1} for bucket "
            f"{bucket!r}"
        )
    if history is None:
        history = ActiveArchiveManifest.frozen_for(iteration)
    else:
        if history.iteration != iteration:
            raise Phase10BScheduleError(
                f"history manifest is for iteration {history.iteration}, not "
                f"{iteration}"
            )
        history.validate()

    identifier = phase10b_game_id(iteration, bucket, ordinal)
    control = learner_control_for(bucket, iteration, ordinal)
    color = learner_color(bucket, iteration, ordinal)
    learner_token = behavior_policy_token(iteration)

    historical_identity: "str | None" = None
    opponent_digest: "str | None" = None
    red_seed: "int | None" = None
    blue_seed: "int | None" = None
    archive_seed: "int | None" = None

    if bucket == BUCKET_CURRENT:
        opponent_kind = "current_policy"
        opponent_token = learner_token
    elif bucket == BUCKET_ANCHOR:
        opponent_kind = "historical_snapshot"
        historical_identity = ANCHOR_IDENTITY
        opponent_token = history_policy_token(ANCHOR_IDENTITY)
        opponent_digest = history.digest_map.get(ANCHOR_IDENTITY)
    elif bucket == BUCKET_ARCHIVE:
        opponent_kind = "historical_snapshot"
        historical_identity = archive_member_for(identifier, history.identities)
        opponent_token = history_policy_token(historical_identity)
        opponent_digest = history.digest_map.get(historical_identity)
        archive_seed = int(
            __import__(
                "stratego.training.phase10b_seed", fromlist=["archive_selection_seed"]
            ).archive_selection_seed(identifier)
        )
    else:
        policy_id = opponent_policy_for_ordinal(ordinal)
        opponent_kind = opponent_kind_for(policy_id)
        opponent_token = opponent_policy_token(policy_id)

    if color is None:
        red_identity = blue_identity = learner_token
    elif color == "red":
        red_identity, blue_identity = learner_token, opponent_token
    else:
        red_identity, blue_identity = opponent_token, learner_token

    # Only a rule or stress side owns a match-level policy RNG stream; every
    # other side stores null, so the schedule must not claim one either.
    if bucket == BUCKET_OPPONENT:
        if color == "red":
            blue_seed = opponent_policy_seed(identifier, "blue")
        else:
            red_seed = opponent_policy_seed(identifier, "red")

    return Phase10BScheduledGame(
        rollout_game_id=identifier,
        run_namespace=PHASE10B_NAMESPACE,
        rl_iteration=int(iteration),
        game_ordinal=int(ordinal),
        bucket=bucket,
        red_policy_identity=red_identity,
        blue_policy_identity=blue_identity,
        learner_control=control,
        learner_color=color,
        behavior_snapshot_identity=behavior_snapshot_identity(iteration),
        historical_snapshot_identity=historical_identity,
        opponent_kind=opponent_kind,
        opponent_identity=opponent_token,
        opponent_checkpoint_digest=opponent_digest,
        setup_root_seed=setup_root_seed(identifier),
        red_setup_source_identity=setup_source.side_identity(
            game_id=identifier, player=_player_index("red")
        ),
        blue_setup_source_identity=setup_source.side_identity(
            game_id=identifier, player=_player_index("blue")
        ),
        red_policy_seed=red_seed,
        blue_policy_seed=blue_seed,
        archive_selection_seed=archive_seed,
    )


def _player_index(color: str) -> int:
    for player, name in PLAYER_NAMES.items():
        if name == color:
            return player
    raise Phase10BScheduleError(f"unknown colour {color!r}")


def rebuild_scheduled_game(
    identifier: str,
    *,
    setup_source,
    history: "ActiveArchiveManifest | None" = None,
) -> Phase10BScheduledGame:
    """Rebuild a scheduled game from its identifier alone.

    The pure rebuilder resume uses: a committed id from a crashed run
    reconstructs its full schedule record with no enumeration state, no
    ordering and no partial rollout in hand.
    """
    fields = parse_game_id(identifier)
    return scheduled_game_record(
        fields["iteration"],
        fields["bucket"],
        fields["ordinal"],
        setup_source=setup_source,
        history=history,
    )


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def iteration_game_ids(iteration: int) -> tuple:
    """Every logical game id of one iteration, bucket-major then ordinal."""
    counts = bucket_counts()
    return tuple(
        phase10b_game_id(iteration, bucket, ordinal)
        for bucket in POPULATION_BUCKETS
        for ordinal in range(counts[bucket])
    )


def iter_iteration_schedule(
    iteration: int, *, setup_source, history: "ActiveArchiveManifest | None" = None
):
    if history is None:
        history = ActiveArchiveManifest.frozen_for(iteration)
    counts = bucket_counts()
    for bucket in POPULATION_BUCKETS:
        for ordinal in range(counts[bucket]):
            yield scheduled_game_record(
                iteration,
                bucket,
                ordinal,
                setup_source=setup_source,
                history=history,
            )


def pending_game_ids(iteration: int, committed) -> tuple:
    """`scheduled - committed`, in schedule order."""
    done = set(committed)
    return tuple(
        identifier
        for identifier in iteration_game_ids(iteration)
        if identifier not in done
    )


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def iteration_audit(iteration: int, *, setup_source) -> dict:
    """Everything about one iteration a reviewer should be able to check."""
    counts = bucket_counts()
    observed = {bucket: 0 for bucket in POPULATION_BUCKETS}
    colors = {bucket: {"red": 0, "blue": 0, "both": 0} for bucket in POPULATION_BUCKETS}
    opponents: dict = {}
    archive_draws: dict = {}
    identifiers = set()
    setup_identities = set()
    history = ActiveArchiveManifest.frozen_for(iteration)
    for record in iter_iteration_schedule(
        iteration, setup_source=setup_source, history=history
    ):
        observed[record.bucket] += 1
        colors[record.bucket][record.learner_control] += 1
        identifiers.add(record.rollout_game_id)
        setup_identities.add(record.red_setup_source_identity)
        setup_identities.add(record.blue_setup_source_identity)
        if record.bucket == BUCKET_OPPONENT:
            policy_id = record.opponent_identity.split("@", 1)[0]
            opponents[policy_id] = opponents.get(policy_id, 0) + 1
        if record.bucket == BUCKET_ARCHIVE:
            archive_draws[record.historical_snapshot_identity] = (
                archive_draws.get(record.historical_snapshot_identity, 0) + 1
            )
    total = sum(counts.values())
    problems: list = []
    if observed != counts:
        problems.append(f"bucket counts {observed} != frozen {counts}")
    if len(identifiers) != total:
        problems.append(f"{len(identifiers)} distinct ids for {total} scheduled games")
    if opponents != opponent_counts():
        problems.append(f"opponent counts {opponents} != frozen {opponent_counts()}")
    # Two setup identities per game, all distinct: no two sides anywhere in the
    # iteration share a selector stream.
    if len(setup_identities) != 2 * total:
        problems.append(
            f"{len(setup_identities)} distinct setup identities for {2 * total} sides"
        )
    return {
        "iteration": int(iteration),
        "scheduled_games": total,
        "bucket_counts": observed,
        "learner_control": colors,
        "opponent_counts": opponents,
        "archive_draws": archive_draws,
        "active_archive_window": list(history.identities),
        "distinct_game_ids": len(identifiers),
        "distinct_setup_identities": len(setup_identities),
        "problems": problems,
    }


def population_document() -> dict:
    return {
        "population_version": PHASE10B_POPULATION_VERSION,
        "schedule_version": PHASE10B_SCHEDULE_VERSION,
        "namespace": PHASE10B_NAMESPACE,
        "bucket_counts": bucket_counts(),
        "opponent_counts": opponent_counts(),
        "buckets": list(POPULATION_BUCKETS),
        "anchor_identity": ANCHOR_IDENTITY,
        "anchor_policy_token": ANCHOR_POLICY_TOKEN,
        "setup_environment_id": SETUP_ENVIRONMENT_ID,
        "setup_generation": SETUP_GENERATION,
        "iterations": MAX_ITERATIONS,
    }


def population_digest() -> str:
    return hashlib.sha256(
        json.dumps(population_document(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "ANCHOR_POLICY_TOKEN",
    "ActiveArchiveManifest",
    "Phase10BScheduleError",
    "Phase10BScheduledGame",
    "SETUP_ENVIRONMENT_ID",
    "SETUP_GENERATION",
    "behavior_policy_token",
    "behavior_snapshot_identity",
    "history_policy_token",
    "iter_iteration_schedule",
    "iteration_audit",
    "iteration_game_ids",
    "opponent_policy_token",
    "pending_game_ids",
    "population_digest",
    "population_document",
    "rebuild_scheduled_game",
    "scheduled_game_record",
]
