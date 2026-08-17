"""Phase 9 Agent 2: the deterministic population and opponent schedule.

Specification sources:

- `02_AGENT_2_POPULATION_AND_OPPONENT_SCHEDULER.md` (everything here)
- `00_PHASE_9_SEQUENCE_AND_COMMON_CONTRACT.md` (mixture, historical league,
  learner-control semantics, artifact namespaces)
- `phase9_contract` / `phase9_seed` (Agent 1, frozen and accepted)

What this layer decides, and what it does not
---------------------------------------------
This module answers exactly one question: **which logical games should
exist?** It never decides how a neural move is collected, how a target is
built, or how an optimizer steps. Every learning-design constant it uses —
bucket sizes, rule subdivisions, the stress rotation, the colour-balance
parity rule, the archive cadence and the active window — is imported from
Agent 1's frozen `phase9_contract`, never restated. Agent 1 froze the
arithmetic of one game; this module is the enumeration, identity,
verification and audit layer built on top of it, which is what Agent 3
consumes.

Purity
------
A scheduled game is a pure function of

```text
rollout version
frozen Phase 9 master / schedule / opponent seeds
run namespace (one of the six pilot candidates, or canonical)
RL iteration
logical game ordinal
the frozen active historical archive manifest
```

and of nothing else. There is no filesystem access, no environment lookup,
no clock, no global RNG and no path anywhere in this module — the resolved
storage root lives in :mod:`stratego.training.phase9_storage` and is never
imported here. That absence is the mechanical proof of two contract rules at
once: worker count, process partitioning, enumeration order, arrival order
and resume boundaries cannot influence logical identity, and a rollout
copied byte-for-byte to another volume is the same rollout.

Outcome independence
--------------------
Historical opponents are drawn uniformly from the active window by a
domain-separated hash of the game identity. No function here reads a match
result, a win rate or a league table; the archive manifest arrives as an
explicit immutable input. Outcome-prioritised matchmaking is not implemented
because `phase9_population_v1` forbids it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import gcd

from ..engine.constants import BLUE, PLAYER_NAMES, RED
from ..evaluation.registry import POLICY_INDEX
from .phase9_contract import (
    ACTIVE_WINDOW_RECENT_SNAPSHOTS,
    ARCHIVE_CADENCE_ITERATIONS,
    CANONICAL_ITERATIONS,
    HISTORICAL_ANCHOR_ID,
    LEARNER_CONTROL_BOTH,
    PHASE9_POPULATION_VERSION,
    PHASE9_ROLLOUT_SCHEDULE_VERSION,
    PILOT_ITERATIONS,
    POPULATION_PROPORTIONS,
    RULE_TIER_ORDER,
    STRESS_POLICY_ROSTER,
    TRAINING_ELIGIBILITY,
    Phase9ContractError,
    active_historical_window,
    bucket_counts,
    games_per_iteration,
    historical_opponent_for,
    learner_color,
    learner_control_for,
    rule_tier_counts,
    rule_tier_for_ordinal,
    stress_policy_for_ordinal,
)
from .phase9_seed import (
    BUCKET_CURRENT,
    BUCKET_HISTORICAL,
    BUCKET_RULE,
    BUCKET_STRESS,
    CANONICAL_NAMESPACE,
    PHASE9_MASTER_SEED,
    PHASE9_ROLLOUT_VERSION,
    PILOT_NAMESPACES,
    POPULATION_BUCKETS,
    RUN_NAMESPACES,
    blue_policy_seed,
    historical_opponent_seed,
    parse_phase9_game_id,
    phase9_game_id,
    red_policy_seed,
    setup_root_seed,
)
from .setup_source import (
    SETUP_SOURCE_VERSION,
    TRAINING_PURPOSE,
    TRAINING_SPLIT,
    training_setup_source,
)
from .warmstart_contract import EXPECTED_SETUP_PROFILE

#: Frozen constants of the `setup_source_v1` call every rollout game makes.
#: The accepted Phase 8 shape, restated here only so a reader of a scheduled
#: record can reproduce the assignment without opening the collector.
SETUP_ENVIRONMENT_ID = 0
SETUP_GENERATION = 0


class Phase9ScheduleError(RuntimeError):
    """Raised when a Phase 9 schedule request or audit input is invalid."""


# ---------------------------------------------------------------------------
# Policy identity tokens
# ---------------------------------------------------------------------------
#
# Agent 1 froze *which* policy plays each game and where each identity is
# stored (`DecisionRecord.collection_policy_version`, the sidecar's
# `red_policy_token` / `blue_policy_token`). It left the spelling of the two
# neural tokens open. These are that spelling and nothing more: no learning
# semantics hang off the strings, only unambiguous naming.

BEHAVIOR_TOKEN_PREFIX = "phase9_behavior_v1"
ARCHIVE_TOKEN_PREFIX = "phase9_archive_v1"

#: `H000` is the accepted Phase 8 checkpoint, bit-identical in every run
#: namespace, so its token is deliberately namespace-free: one checkpoint,
#: one name. Every later archive member is a *per-run* artifact — pilot
#: `H005` and canonical `H005` are different weights — so those tokens carry
#: their namespace and can never be confused across runs.
ANCHOR_POLICY_TOKEN = f"phase9_anchor_v1|{HISTORICAL_ANCHOR_ID}"


def behavior_snapshot_identity(iteration: int) -> str:
    """The logical identity of one iteration's immutable behavior snapshot.

    `B012` is "the learner frozen at the start of iteration 12". It mirrors
    the frozen `H0nn` archive spelling because the two label the same kind of
    object: an immutable set of weights that produced games.
    """
    if not isinstance(iteration, int) or isinstance(iteration, bool):
        raise Phase9ScheduleError(f"iteration must be an int, got {iteration!r}")
    if iteration < 1:
        raise Phase9ScheduleError(f"iteration must be >= 1, got {iteration}")
    return f"B{iteration:03d}"


def behavior_policy_token(namespace: str, iteration: int) -> str:
    """The stored policy token of the current-policy side of one game."""
    _require_namespace(namespace)
    return f"{BEHAVIOR_TOKEN_PREFIX}|ns={namespace}|{behavior_snapshot_identity(iteration)}"


def historical_policy_token(namespace: str, archive_identity: str) -> str:
    """The stored policy token of a historical-archive side of one game."""
    _require_namespace(namespace)
    if archive_identity == HISTORICAL_ANCHOR_ID:
        return ANCHOR_POLICY_TOKEN
    return f"{ARCHIVE_TOKEN_PREFIX}|ns={namespace}|{archive_identity}"


def rule_policy_token(policy_id: str) -> str:
    """The frozen Phase 4 `id@version` token of a rule or stress policy."""
    if policy_id not in POLICY_INDEX:
        raise Phase9ScheduleError(f"unknown frozen policy id: {policy_id!r}")
    return f"{policy_id}@{POLICY_INDEX[policy_id].policy_version}"


def _require_namespace(namespace: str) -> None:
    if namespace not in RUN_NAMESPACES:
        raise Phase9ScheduleError(
            f"unknown Phase 9 namespace {namespace!r}; expected one of "
            f"{list(RUN_NAMESPACES)}"
        )


def run_iterations(namespace: str) -> int:
    """The frozen iteration budget of one run namespace."""
    _require_namespace(namespace)
    return CANONICAL_ITERATIONS if namespace == CANONICAL_NAMESPACE else PILOT_ITERATIONS


# ---------------------------------------------------------------------------
# The active historical archive manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveHistoryManifest:
    """The immutable active archive of one iteration, as an explicit input.

    The scheduler does not discover history; it is *given* history. This
    object is the interface Agent 3 (and later the trainer) hands in, and
    :meth:`validate` refuses anything that disagrees with the frozen window
    rule — Phase 8 anchor plus the eight most recent eligible snapshots.
    Checkpoint digests are optional here because scheduling needs identities
    only; a collector that resolves real weights supplies them and gets the
    digest checked into every record it builds.
    """

    namespace: str
    iteration: int
    identities: tuple
    checkpoint_digests: tuple = ()

    @classmethod
    def frozen_for(
        cls, namespace: str, iteration: int, checkpoint_digests: "dict | None" = None
    ) -> "ActiveHistoryManifest":
        """The contract-derived manifest of one iteration."""
        _require_namespace(namespace)
        identities = active_historical_window(iteration)
        digests = tuple(
            sorted((str(key), str(value)) for key, value in (checkpoint_digests or {}).items())
        )
        return cls(
            namespace=namespace,
            iteration=int(iteration),
            identities=tuple(identities),
            checkpoint_digests=digests,
        )

    @property
    def digest_map(self) -> dict:
        return dict(self.checkpoint_digests)

    def validate(self) -> None:
        """Refuse a manifest that is not the frozen window of its iteration."""
        _require_namespace(self.namespace)
        expected = active_historical_window(self.iteration)
        if tuple(self.identities) != expected:
            raise Phase9ScheduleError(
                f"active history manifest for {self.namespace!r} iteration "
                f"{self.iteration} is {list(self.identities)}, but the frozen "
                f"window is {list(expected)}"
            )
        unknown = sorted(set(self.digest_map) - set(expected))
        if unknown:
            raise Phase9ScheduleError(
                f"manifest carries checkpoint digests for identities outside the "
                f"active window: {unknown}"
            )

    def to_dict(self) -> dict:
        return {
            "namespace": self.namespace,
            "iteration": self.iteration,
            "identities": list(self.identities),
            "checkpoint_digests": self.digest_map,
            "window_rule": (
                f"{HISTORICAL_ANCHOR_ID} + up to {ACTIVE_WINDOW_RECENT_SNAPSHOTS} "
                f"most recent snapshots archived every "
                f"{ARCHIVE_CADENCE_ITERATIONS} committed iterations"
            ),
            "selection_rule": "uniform over the active window, outcome-independent",
        }


# ---------------------------------------------------------------------------
# One scheduled logical game
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduledGame:
    """Everything `phase9_rollout_schedule_v1` fixes about one logical game.

    Deliberately carries no privileged information: no piece truth, no setup
    contents, no outcome. The setup *identity* is here (root seed and the two
    per-side source identities); resolving it to boards is the collector's
    job through the frozen train-split source.
    """

    phase9_game_id: str
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
    historical_opponent_seed: "int | None"

    def to_dict(self) -> dict:
        return asdict(self)

    def policy_rng_identities(self) -> dict:
        """The RNG streams this game's participants draw from, by name."""
        return {
            "setup_root_seed": self.setup_root_seed,
            "red_policy_seed": self.red_policy_seed,
            "blue_policy_seed": self.blue_policy_seed,
            "historical_opponent_seed": self.historical_opponent_seed,
            "behavior_sampler": (
                f"behavior_sample_seed({self.phase9_game_id!r}, ply) per neural decision"
            ),
            "rule_policy_ply_stream": (
                "derive_decision_seed(<side>_policy_seed, ply), the frozen Phase 4 path"
            ),
        }

    @property
    def learner_sides(self) -> tuple:
        """Which colours receive Phase 9 policy/value/belief loss."""
        if self.learner_control == LEARNER_CONTROL_BOTH:
            return ("red", "blue")
        return (self.learner_control,)


@lru_cache(maxsize=1)
def _setup_source():
    """The frozen production train-split setup source.

    Construction is cheap and library-free — the 8,000-entry index loads
    lazily inside the sampler — so a schedule may name setup identities
    without ever touching setup contents. Cached because a full-run
    enumeration asks for it once per game.
    """
    return training_setup_source(EXPECTED_SETUP_PROFILE)


def _setup_side_identity(source, root: int, player: int) -> str:
    """The per-side setup identity of one logical game.

    `source_id | player | side seed` is everything needed to rebuild that
    side's board through the frozen sampler and nothing more.
    """
    seed = source.side_seed(
        root_seed=root,
        environment_id=SETUP_ENVIRONMENT_ID,
        generation=SETUP_GENERATION,
        player=player,
    )
    return f"{source.setup_family}|player={PLAYER_NAMES[player]}|side_seed={seed}"


def scheduled_game_record(
    namespace: str,
    iteration: int,
    bucket: str,
    ordinal: int,
    *,
    history: "ActiveHistoryManifest | None" = None,
) -> ScheduledGame:
    """The full scheduled record of one logical game.

    Wraps Agent 1's frozen `scheduled_game` arithmetic and adds only the
    identities Agent 3 needs to actually run it: policy tokens per side, the
    behavior snapshot identity, the setup identity derivation and the per-
    stream seeds. `history` is the explicit immutable archive manifest; when
    omitted the frozen window of the iteration is used.
    """
    _require_namespace(namespace)
    counts = bucket_counts(namespace)
    if bucket not in counts:
        raise Phase9ScheduleError(f"unknown population bucket: {bucket!r}")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise Phase9ScheduleError(f"ordinal must be an int, got {ordinal!r}")
    if not 0 <= ordinal < counts[bucket]:
        raise Phase9ScheduleError(
            f"ordinal {ordinal} is outside 0..{counts[bucket] - 1} for bucket "
            f"{bucket!r} in {namespace!r}"
        )
    budget = run_iterations(namespace)
    if not isinstance(iteration, int) or isinstance(iteration, bool):
        raise Phase9ScheduleError(f"iteration must be an int, got {iteration!r}")
    if not 1 <= iteration <= budget:
        raise Phase9ScheduleError(
            f"iteration {iteration} is outside the frozen 1..{budget} budget of "
            f"{namespace!r}"
        )

    if history is None:
        history = ActiveHistoryManifest.frozen_for(namespace, iteration)
    else:
        if history.namespace != namespace or history.iteration != iteration:
            raise Phase9ScheduleError(
                f"history manifest is for {history.namespace!r} iteration "
                f"{history.iteration}, not {namespace!r} iteration {iteration}"
            )
        history.validate()

    game_id = phase9_game_id(namespace, iteration, bucket, ordinal)
    control = learner_control_for(bucket, iteration, ordinal)
    color = learner_color(bucket, iteration, ordinal)
    behavior_identity = behavior_snapshot_identity(iteration)
    learner_token = behavior_policy_token(namespace, iteration)

    historical_identity: "str | None" = None
    opponent_digest: "str | None" = None
    red_seed: "int | None" = None
    blue_seed: "int | None" = None
    history_seed: "int | None" = None

    if bucket == BUCKET_CURRENT:
        opponent_kind = "current_policy"
        opponent_token = learner_token
    elif bucket == BUCKET_HISTORICAL:
        opponent_kind = "historical_snapshot"
        historical_identity = historical_opponent_for(game_id)
        if historical_identity not in history.identities:
            raise Phase9ScheduleError(
                f"{game_id} drew historical identity {historical_identity!r}, which "
                f"is not in the supplied active window {list(history.identities)}"
            )
        opponent_token = historical_policy_token(namespace, historical_identity)
        opponent_digest = history.digest_map.get(historical_identity)
        history_seed = historical_opponent_seed(game_id)
    elif bucket == BUCKET_RULE:
        opponent_kind = "rule_policy"
        opponent_token = rule_policy_token(rule_tier_for_ordinal(namespace, ordinal))
    else:
        opponent_kind = "stress_policy"
        opponent_token = rule_policy_token(
            stress_policy_for_ordinal(iteration, ordinal, namespace=namespace)
        )

    if color is None:
        red_identity = blue_identity = learner_token
    elif color == RED:
        red_identity, blue_identity = learner_token, opponent_token
    else:
        red_identity, blue_identity = opponent_token, learner_token

    # Only a rule or stress side owns a match-level policy RNG stream; the
    # frozen sidecar contract stores null for every other side, so the
    # schedule must not claim one either.
    if bucket in (BUCKET_RULE, BUCKET_STRESS):
        if color == RED:
            blue_seed = blue_policy_seed(game_id)
        else:
            red_seed = red_policy_seed(game_id)

    root = setup_root_seed(game_id)
    source = _setup_source()
    return ScheduledGame(
        phase9_game_id=game_id,
        run_namespace=namespace,
        rl_iteration=int(iteration),
        game_ordinal=int(ordinal),
        bucket=bucket,
        red_policy_identity=red_identity,
        blue_policy_identity=blue_identity,
        learner_control=control,
        learner_color=None if color is None else PLAYER_NAMES[color],
        behavior_snapshot_identity=behavior_identity,
        historical_snapshot_identity=historical_identity,
        opponent_kind=opponent_kind,
        opponent_identity=opponent_token,
        opponent_checkpoint_digest=opponent_digest,
        setup_root_seed=root,
        red_setup_source_identity=_setup_side_identity(source, root, RED),
        blue_setup_source_identity=_setup_side_identity(source, root, BLUE),
        red_policy_seed=red_seed,
        blue_policy_seed=blue_seed,
        historical_opponent_seed=history_seed,
    )


def rebuild_scheduled_game(
    game_id: str, *, history: "ActiveHistoryManifest | None" = None
) -> ScheduledGame:
    """Rebuild a scheduled game from its identifier alone.

    The pure parser/rebuilder Agent 3 uses on resume: a committed id from a
    crashed run reconstructs its own full schedule record with no enumeration
    state, no ordering and no partial rollout in hand. Refuses an id whose
    ordinal is outside its bucket, which is what stops a malformed or foreign
    identifier from silently becoming a schedulable game.
    """
    fields = parse_phase9_game_id(game_id)
    return scheduled_game_record(
        fields["namespace"],
        fields["iteration"],
        fields["bucket"],
        fields["ordinal"],
        history=history,
    )


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def iter_iteration_schedule(
    namespace: str, iteration: int, *, history: "ActiveHistoryManifest | None" = None
):
    """Every logical game of one iteration, bucket-major then ordinal.

    A *schedule* order only. Collection may proceed in any order across any
    number of workers because every record is a pure function of its own
    identity — which :func:`audit_worker_order_independence` proves rather
    than assumes.
    """
    _require_namespace(namespace)
    if history is None:
        history = ActiveHistoryManifest.frozen_for(namespace, iteration)
    counts = bucket_counts(namespace)
    for bucket in POPULATION_BUCKETS:
        for ordinal in range(counts[bucket]):
            yield scheduled_game_record(
                namespace, iteration, bucket, ordinal, history=history
            )


def iteration_schedule(
    namespace: str, iteration: int, *, history: "ActiveHistoryManifest | None" = None
) -> tuple:
    return tuple(iter_iteration_schedule(namespace, iteration, history=history))


def iteration_game_ids(namespace: str, iteration: int) -> tuple:
    """Just the identifiers of one iteration — no policy or seed derivation.

    The cheap form the resume path needs: identity arithmetic only.
    """
    _require_namespace(namespace)
    budget = run_iterations(namespace)
    if not 1 <= iteration <= budget:
        raise Phase9ScheduleError(
            f"iteration {iteration} is outside the frozen 1..{budget} budget of "
            f"{namespace!r}"
        )
    counts = bucket_counts(namespace)
    return tuple(
        phase9_game_id(namespace, iteration, bucket, ordinal)
        for bucket in POPULATION_BUCKETS
        for ordinal in range(counts[bucket])
    )


def iter_run_schedule(namespace: str):
    """Every logical game of a whole run namespace, iteration-major."""
    for iteration in range(1, run_iterations(namespace) + 1):
        yield from iter_iteration_schedule(namespace, iteration)


def run_game_ids(namespace: str) -> tuple:
    return tuple(
        game_id
        for iteration in range(1, run_iterations(namespace) + 1)
        for game_id in iteration_game_ids(namespace, iteration)
    )


def total_scheduled_games(namespace: str) -> int:
    return run_iterations(namespace) * games_per_iteration(namespace)


# ---------------------------------------------------------------------------
# Resume subtraction
# ---------------------------------------------------------------------------


def pending_game_ids(namespace: str, iteration: int, committed) -> tuple:
    """Scheduled minus committed, in schedule order.

    The frozen crash rule is "deterministically regenerate only missing or
    uncommitted game ids", so resume is set subtraction over identities and
    nothing else — no cursor, no arrival order, no partial-file heuristics. A
    committed id that is not part of this iteration's schedule is an error
    rather than a no-op: it means the caller is holding another iteration's
    or another run's rollout, and silently ignoring that would let a resume
    seal an incomplete iteration.
    """
    scheduled = iteration_game_ids(namespace, iteration)
    scheduled_set = set(scheduled)
    committed_set = set(committed)
    foreign = sorted(committed_set - scheduled_set)
    if foreign:
        raise Phase9ScheduleError(
            f"{len(foreign)} committed game id(s) are not scheduled for "
            f"{namespace!r} iteration {iteration}, first: {foreign[0]!r}"
        )
    return tuple(game_id for game_id in scheduled if game_id not in committed_set)


def resume_plan(namespace: str, iteration: int, committed) -> dict:
    """The resume decision for one iteration, as a reportable record."""
    pending = pending_game_ids(namespace, iteration, committed)
    scheduled = iteration_game_ids(namespace, iteration)
    return {
        "namespace": namespace,
        "iteration": int(iteration),
        "scheduled": len(scheduled),
        "committed": len(set(committed)),
        "pending": len(pending),
        "complete": not pending,
        "pending_game_ids": list(pending),
        "rule": "pending = scheduled ids - committed ids; sealing requires pending == 0",
    }


# ---------------------------------------------------------------------------
# Schedule documents and digests
# ---------------------------------------------------------------------------


def _canonical_json(document: dict) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def _digest(document: dict) -> str:
    return hashlib.sha256(_canonical_json(document).encode()).hexdigest()


def population_document() -> dict:
    """The serializable `phase9_population_v1` document this layer implements."""
    return {
        "population_version": PHASE9_POPULATION_VERSION,
        "schedule_version": PHASE9_ROLLOUT_SCHEDULE_VERSION,
        "rollout_version": PHASE9_ROLLOUT_VERSION,
        "phase9_master_seed": PHASE9_MASTER_SEED,
        "proportions": dict(POPULATION_PROPORTIONS),
        "namespaces": {
            namespace: {
                "iterations": run_iterations(namespace),
                "games_per_iteration": games_per_iteration(namespace),
                "total_scheduled_games": total_scheduled_games(namespace),
                "bucket_counts": bucket_counts(namespace),
                "rule_tier_counts": rule_tier_counts(namespace),
            }
            for namespace in RUN_NAMESPACES
        },
        "rule_tier_order": list(RULE_TIER_ORDER),
        "rule_tier_tokens": {tier: rule_policy_token(tier) for tier in RULE_TIER_ORDER},
        "stress_roster": list(STRESS_POLICY_ROSTER),
        "stress_tokens": {
            policy: rule_policy_token(policy) for policy in STRESS_POLICY_ROSTER
        },
        "stress_rotation": "(ordinal + iteration) % 6 over the frozen roster",
        "colour_balance_rule": (
            "learner is red iff (ordinal + iteration) % 2 == 0; an odd-sized "
            "ordinal range's one-game remainder alternates with iteration parity"
        ),
        "learner_control": dict(TRAINING_ELIGIBILITY),
        "historical_league": {
            "anchor": HISTORICAL_ANCHOR_ID,
            "anchor_token": ANCHOR_POLICY_TOKEN,
            "archive_cadence_iterations": ARCHIVE_CADENCE_ITERATIONS,
            "active_window_recent_snapshots": ACTIVE_WINDOW_RECENT_SNAPSHOTS,
            "selection": "uniform over the active window under a domain-separated hash",
            "outcome_independence": (
                "no schedule function reads a match result, win rate or league "
                "table; outcome-prioritised matchmaking is not implemented"
            ),
        },
        "setup_assignment": {
            "setup_source_version": SETUP_SOURCE_VERSION,
            "split": TRAINING_SPLIT,
            "purpose": TRAINING_PURPOSE,
            "profile": EXPECTED_SETUP_PROFILE,
            "environment_id": SETUP_ENVIRONMENT_ID,
            "generation": SETUP_GENERATION,
            "root": "setup_root_seed(game_id)",
            "family_weighting": "none; families are never outcome-weighted",
        },
        "storage_independence": (
            "logical identity is independent of storage location; this module "
            "performs no filesystem or environment access"
        ),
    }


def population_digest() -> str:
    return _digest(population_document())


def iteration_schedule_document(
    namespace: str, iteration: int, *, history: "ActiveHistoryManifest | None" = None
) -> dict:
    """The full logical content of one iteration, ready to hash."""
    if history is None:
        history = ActiveHistoryManifest.frozen_for(namespace, iteration)
    games = iteration_schedule(namespace, iteration, history=history)
    return {
        "schedule_version": PHASE9_ROLLOUT_SCHEDULE_VERSION,
        "population_version": PHASE9_POPULATION_VERSION,
        "rollout_version": PHASE9_ROLLOUT_VERSION,
        "phase9_master_seed": PHASE9_MASTER_SEED,
        "namespace": namespace,
        "iteration": int(iteration),
        "active_history": history.to_dict(),
        "games": [game.to_dict() for game in games],
    }


def iteration_schedule_digest(
    namespace: str, iteration: int, *, history: "ActiveHistoryManifest | None" = None
) -> str:
    return _digest(iteration_schedule_document(namespace, iteration, history=history))


def run_schedule_document(namespace: str) -> dict:
    """Per-iteration digests of a whole run, plus its own digest inputs.

    Each iteration digest already covers every field of every game in that
    iteration, so hashing the list of digests is exactly as strong as hashing
    the concatenated schedules and far cheaper to carry in a report.
    """
    _require_namespace(namespace)
    iterations = run_iterations(namespace)
    return {
        "schedule_version": PHASE9_ROLLOUT_SCHEDULE_VERSION,
        "namespace": namespace,
        "iterations": iterations,
        "games_per_iteration": games_per_iteration(namespace),
        "total_scheduled_games": total_scheduled_games(namespace),
        "iteration_digests": {
            str(iteration): iteration_schedule_digest(namespace, iteration)
            for iteration in range(1, iterations + 1)
        },
    }


def run_schedule_digest(namespace: str) -> str:
    return _digest(run_schedule_document(namespace))


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def _colour_split(games) -> dict:
    red = sum(1 for game in games if game.learner_color == "red")
    blue = sum(1 for game in games if game.learner_color == "blue")
    none = sum(1 for game in games if game.learner_color is None)
    return {"red": red, "blue": blue, "unassigned": none}


def audit_iteration(
    namespace: str, iteration: int, *, history: "ActiveHistoryManifest | None" = None
) -> dict:
    """Every per-iteration structural claim, checked rather than assumed."""
    games = iteration_schedule(namespace, iteration, history=history)
    expected_counts = bucket_counts(namespace)
    expected_tiers = rule_tier_counts(namespace)
    window = active_historical_window(iteration)
    problems: list[str] = []

    observed_counts = {bucket: 0 for bucket in POPULATION_BUCKETS}
    tier_counts = {tier: 0 for tier in RULE_TIER_ORDER}
    stress_counts = {policy: 0 for policy in STRESS_POLICY_ROSTER}
    history_counts = {identity: 0 for identity in window}
    by_bucket: dict = {bucket: [] for bucket in POPULATION_BUCKETS}
    by_tier: dict = {tier: [] for tier in RULE_TIER_ORDER}
    seen_ids: set = set()
    duplicates = 0

    for game in games:
        if game.phase9_game_id in seen_ids:
            duplicates += 1
        seen_ids.add(game.phase9_game_id)
        observed_counts[game.bucket] += 1
        by_bucket[game.bucket].append(game)

        expected_control = TRAINING_ELIGIBILITY[game.bucket]
        if game.bucket == BUCKET_CURRENT:
            if game.learner_control != LEARNER_CONTROL_BOTH:
                problems.append(
                    f"{game.phase9_game_id}: current/current must be "
                    f"{LEARNER_CONTROL_BOTH!r}, got {game.learner_control!r} "
                    f"({expected_control})"
                )
            if game.red_policy_identity != game.blue_policy_identity:
                problems.append(
                    f"{game.phase9_game_id}: current/current sides must share one "
                    f"behavior snapshot identity"
                )
        else:
            if game.learner_control == LEARNER_CONTROL_BOTH:
                problems.append(
                    f"{game.phase9_game_id}: asymmetric bucket {game.bucket!r} must "
                    f"train the current-policy side only ({expected_control})"
                )
            if game.learner_color not in ("red", "blue"):
                problems.append(
                    f"{game.phase9_game_id}: asymmetric game has no learner colour"
                )

        if game.bucket == BUCKET_RULE:
            tier = rule_tier_for_ordinal(namespace, game.game_ordinal)
            tier_counts[tier] += 1
            by_tier[tier].append(game)
            if game.opponent_identity != rule_policy_token(tier):
                problems.append(
                    f"{game.phase9_game_id}: rule token {game.opponent_identity!r} "
                    f"disagrees with tier {tier!r}"
                )
        elif game.bucket == BUCKET_STRESS:
            policy = stress_policy_for_ordinal(
                iteration, game.game_ordinal, namespace=namespace
            )
            stress_counts[policy] += 1
            if game.opponent_identity != rule_policy_token(policy):
                problems.append(
                    f"{game.phase9_game_id}: stress token "
                    f"{game.opponent_identity!r} disagrees with {policy!r}"
                )
        elif game.bucket == BUCKET_HISTORICAL:
            identity = game.historical_snapshot_identity
            if identity not in history_counts:
                problems.append(
                    f"{game.phase9_game_id}: historical identity {identity!r} is "
                    f"outside the active window {list(window)}"
                )
            else:
                history_counts[identity] += 1

    if observed_counts != expected_counts:
        problems.append(
            f"bucket counts {observed_counts} != frozen {expected_counts}"
        )
    if tier_counts != expected_tiers:
        problems.append(f"rule tiers {tier_counts} != frozen {expected_tiers}")
    if len(games) != games_per_iteration(namespace):
        problems.append(
            f"{len(games)} games scheduled, frozen total is "
            f"{games_per_iteration(namespace)}"
        )
    if duplicates:
        problems.append(f"{duplicates} duplicate game id(s) within the iteration")

    stress_spread = max(stress_counts.values()) - min(stress_counts.values())
    if stress_spread > 1:
        problems.append(
            f"stress allocation spread {stress_spread} exceeds the frozen "
            f"one-game bound: {stress_counts}"
        )

    colour = {}
    for bucket in (BUCKET_HISTORICAL, BUCKET_RULE, BUCKET_STRESS):
        split = _colour_split(by_bucket[bucket])
        colour[bucket] = split
        if abs(split["red"] - split["blue"]) > 1:
            problems.append(f"{bucket}: colour imbalance {split}")
        if split["unassigned"]:
            problems.append(f"{bucket}: {split['unassigned']} game(s) without a colour")
    for tier in RULE_TIER_ORDER:
        split = _colour_split(by_tier[tier])
        colour[f"rule:{tier}"] = split
        if abs(split["red"] - split["blue"]) > 1:
            problems.append(f"rule tier {tier}: colour imbalance {split}")
        if len(by_tier[tier]) % 2 == 0 and split["red"] != split["blue"]:
            problems.append(
                f"rule tier {tier}: even-sized range must split exactly, got {split}"
            )
    current_split = _colour_split(by_bucket[BUCKET_CURRENT])
    colour[BUCKET_CURRENT] = current_split
    if current_split["unassigned"] != len(by_bucket[BUCKET_CURRENT]):
        problems.append(
            "current/current games must carry no learner colour (both sides train)"
        )

    return {
        "namespace": namespace,
        "iteration": int(iteration),
        "games": len(games),
        "bucket_counts": observed_counts,
        "expected_bucket_counts": expected_counts,
        "rule_tier_counts": tier_counts,
        "expected_rule_tier_counts": expected_tiers,
        "stress_counts": stress_counts,
        "stress_spread": stress_spread,
        "historical_counts": history_counts,
        "active_window": list(window),
        "colour_balance": colour,
        "duplicate_game_ids": duplicates,
        "schedule_digest": iteration_schedule_digest(
            namespace, iteration, history=history
        ),
        "problems": problems,
    }


def audit_namespace(namespace: str) -> dict:
    """Every whole-run claim: totals, uniqueness, long-run balance, windows."""
    iterations = run_iterations(namespace)
    problems: list[str] = []
    per_iteration = []
    all_ids: set = set()
    duplicates = 0
    stress_totals = {policy: 0 for policy in STRESS_POLICY_ROSTER}
    colour_totals = {"red": 0, "blue": 0, "unassigned": 0}
    history_totals: dict = {}
    total = 0

    for iteration in range(1, iterations + 1):
        report = audit_iteration(namespace, iteration)
        problems.extend(f"iteration {iteration}: {issue}" for issue in report["problems"])
        per_iteration.append(report)
        total += report["games"]
        for policy, count in report["stress_counts"].items():
            stress_totals[policy] += count
        for identity, count in report["historical_counts"].items():
            history_totals[identity] = history_totals.get(identity, 0) + count
        for bucket in (BUCKET_HISTORICAL, BUCKET_RULE, BUCKET_STRESS):
            split = report["colour_balance"][bucket]
            colour_totals["red"] += split["red"]
            colour_totals["blue"] += split["blue"]
        colour_totals["unassigned"] += report["colour_balance"][BUCKET_CURRENT][
            "unassigned"
        ]
        for game_id in iteration_game_ids(namespace, iteration):
            if game_id in all_ids:
                duplicates += 1
            all_ids.add(game_id)

    expected_total = total_scheduled_games(namespace)
    if total != expected_total:
        problems.append(f"{total} scheduled games, frozen total is {expected_total}")
    if len(all_ids) != expected_total:
        problems.append(
            f"{len(all_ids)} distinct game ids for {expected_total} scheduled games"
        )
    if duplicates:
        problems.append(f"{duplicates} duplicate game id(s) across the run")

    stress_spread = max(stress_totals.values()) - min(stress_totals.values())
    # Every iteration is allowed a one-game remainder; the rotation is meant to
    # move that remainder, so the long-run spread must be far tighter than the
    # per-iteration bound rather than merely equal to it.
    if stress_spread > iterations:
        problems.append(
            f"long-run stress spread {stress_spread} over {iterations} iterations: "
            f"{stress_totals}"
        )
    colour_gap = abs(colour_totals["red"] - colour_totals["blue"])
    if colour_gap > iterations:
        problems.append(
            f"long-run learner colour gap {colour_gap} over {iterations} iterations"
        )

    return {
        "namespace": namespace,
        "iterations": iterations,
        "games_per_iteration": games_per_iteration(namespace),
        "total_scheduled_games": total,
        "expected_total_scheduled_games": expected_total,
        "distinct_game_ids": len(all_ids),
        "duplicate_game_ids": duplicates,
        "stress_totals": stress_totals,
        "stress_long_run_spread": stress_spread,
        "learner_colour_totals": colour_totals,
        "learner_colour_gap": colour_gap,
        "historical_totals": history_totals,
        "final_active_window": list(active_historical_window(iterations)),
        "run_schedule_digest": run_schedule_digest(namespace),
        "per_iteration": per_iteration,
        "problems": problems,
    }


def audit_cross_namespace_collisions(namespaces=RUN_NAMESPACES) -> dict:
    """No game id may be shared by two run namespaces.

    The identifier carries `ns=`, so this is structurally impossible — which
    is exactly why it is worth measuring rather than asserting.
    """
    seen: dict = {}
    collisions: list = []
    per_namespace = {}
    for namespace in namespaces:
        ids = run_game_ids(namespace)
        per_namespace[namespace] = len(ids)
        for game_id in ids:
            owner = seen.get(game_id)
            if owner is not None and owner != namespace:
                collisions.append({"game_id": game_id, "namespaces": [owner, namespace]})
            seen[game_id] = namespace
    return {
        "namespaces": list(namespaces),
        "per_namespace_game_counts": per_namespace,
        "total_game_ids": sum(per_namespace.values()),
        "distinct_game_ids": len(seen),
        "cross_namespace_collisions": len(collisions),
        "examples": collisions[:5],
        "problems": (
            [f"{len(collisions)} cross-namespace game-id collision(s)"]
            if collisions
            else []
        ),
    }


def audit_seed_collisions(namespaces=RUN_NAMESPACES) -> dict:
    """Collision counts of every finite-width derived scheduling stream.

    Each stream is audited **separately**, because that is what the frozen
    uniqueness contract is about: two games must not share a setup root, a
    red policy stream, a blue policy stream or a historical draw stream.
    Cross-stream coincidences are reported too but are not violations — the
    streams are domain-separated and consumed by different code paths, so a
    setup root that happens to equal some policy seed changes nothing.
    """
    streams = {
        "setup_root": {},
        "setup_side_red": {},
        "setup_side_blue": {},
        "policy_red": {},
        "policy_blue": {},
        "historical_opponent": {},
    }
    within: dict = {name: [] for name in streams}
    source = _setup_source()
    # A game whose two setup sides shared a stream would deal both players the
    # identical board. The two sides live in different stream dictionaries, so
    # this needs its own explicit comparison rather than a dictionary lookup.
    same_game_side_collisions: list = []

    for namespace in namespaces:
        for iteration in range(1, run_iterations(namespace) + 1):
            for game_id in iteration_game_ids(namespace, iteration):
                fields = parse_phase9_game_id(game_id)
                root = setup_root_seed(game_id)
                values = {
                    "setup_root": root,
                    "setup_side_red": source.side_seed(
                        root_seed=root,
                        environment_id=SETUP_ENVIRONMENT_ID,
                        generation=SETUP_GENERATION,
                        player=RED,
                    ),
                    "setup_side_blue": source.side_seed(
                        root_seed=root,
                        environment_id=SETUP_ENVIRONMENT_ID,
                        generation=SETUP_GENERATION,
                        player=BLUE,
                    ),
                    "policy_red": red_policy_seed(game_id),
                    "policy_blue": blue_policy_seed(game_id),
                }
                if values["setup_side_red"] == values["setup_side_blue"]:
                    same_game_side_collisions.append(game_id)
                if fields["bucket"] == BUCKET_HISTORICAL:
                    values["historical_opponent"] = historical_opponent_seed(game_id)
                for name, value in values.items():
                    owner = streams[name].get(value)
                    if owner is not None:
                        within[name].append(
                            {"seed": value, "game_ids": [owner, game_id]}
                        )
                    else:
                        streams[name][value] = game_id

    per_stream = {
        name: {
            "distinct_seeds": len(streams[name]),
            "values_derived": len(streams[name]) + len(within[name]),
            "collisions": len(within[name]),
            "examples": within[name][:3],
        }
        for name in streams
    }

    # Report-only: the same integer appearing in two different domains.
    cross: list = []
    names = list(streams)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            shared = set(streams[first]) & set(streams[second])
            if shared:
                cross.append(
                    {"streams": [first, second], "shared_values": len(shared)}
                )

    total = sum(len(entries) for entries in within.values())
    problems = []
    if total:
        problems.append(
            f"{total} within-stream seed collision(s) violate the frozen "
            f"uniqueness contract"
        )
    if same_game_side_collisions:
        problems.append(
            f"{len(same_game_side_collisions)} game(s) draw both setup sides "
            f"from one stream, first: {same_game_side_collisions[0]!r}"
        )
    return {
        "namespaces": list(namespaces),
        "seed_width_bits": 63,
        "per_stream": per_stream,
        "within_stream_collisions": total,
        "same_game_setup_side_collisions": len(same_game_side_collisions),
        "same_game_setup_side_examples": same_game_side_collisions[:3],
        "cross_stream_shared_values": cross,
        "cross_stream_note": (
            "domain-separated streams are consumed by different code paths; a "
            "shared integer across two domains is not a contract violation"
        ),
        "problems": problems,
    }


def audit_worker_order_independence(
    namespace: str, iteration: int, worker_counts=(1, 3, 8, 13)
) -> dict:
    """Prove the schedule is identical under every partitioning of the work.

    Simulates each worker count with three different partitioning strategies
    (round-robin, contiguous blocks, and blocks walked backwards) and requires
    the reassembled set of records to be byte-identical to the reference
    enumeration. Since the records are built from identities alone, any
    difference would mean order had leaked into a derivation.
    """
    reference = {game.phase9_game_id: game.to_dict() for game in iteration_schedule(namespace, iteration)}
    ordered_ids = list(iteration_game_ids(namespace, iteration))
    mismatches: list = []
    strategies = ("round_robin", "blocks", "reversed_blocks")

    for workers in worker_counts:
        for strategy in strategies:
            partitions: list = [[] for _ in range(workers)]
            if strategy == "round_robin":
                for index, game_id in enumerate(ordered_ids):
                    partitions[index % workers].append(game_id)
            else:
                size = -(-len(ordered_ids) // workers)
                for index, game_id in enumerate(ordered_ids):
                    partitions[min(index // size, workers - 1)].append(game_id)
                if strategy == "reversed_blocks":
                    partitions = [list(reversed(part)) for part in partitions]
                    partitions.reverse()
            rebuilt: dict = {}
            for part in partitions:
                for game_id in part:
                    rebuilt[game_id] = rebuild_scheduled_game(game_id).to_dict()
            if set(rebuilt) != set(reference):
                mismatches.append(
                    {
                        "workers": workers,
                        "strategy": strategy,
                        "reason": "game-id set differs from the reference enumeration",
                    }
                )
                continue
            for game_id, record in rebuilt.items():
                if record != reference[game_id]:
                    mismatches.append(
                        {
                            "workers": workers,
                            "strategy": strategy,
                            "game_id": game_id,
                            "reason": "record differs from the reference enumeration",
                        }
                    )
                    break

    return {
        "namespace": namespace,
        "iteration": int(iteration),
        "worker_counts": list(worker_counts),
        "strategies": list(strategies),
        "games": len(reference),
        "partitionings_checked": len(worker_counts) * len(strategies),
        "mismatches": len(mismatches),
        "examples": mismatches[:3],
        "problems": (
            [f"{len(mismatches)} worker/order-dependent record(s)"] if mismatches else []
        ),
    }


def audit_resume_identity(namespace: str, iteration: int, fractions=(0.0, 0.37, 0.5, 0.999, 1.0)) -> dict:
    """Prove resume is exactly "scheduled minus committed", at several depths.

    The committed subsets are chosen by a deterministic stride rather than a
    random draw so the audit itself carries no RNG. Each pending id is
    rebuilt from its identifier alone and compared to the reference record,
    which is the property a crashed collector actually depends on.
    """
    reference = {
        game.phase9_game_id: game.to_dict()
        for game in iteration_schedule(namespace, iteration)
    }
    scheduled = list(iteration_game_ids(namespace, iteration))
    problems: list = []
    checks: list = []

    # A coprime stride walks the whole schedule, so every committed subset
    # interleaves the four buckets instead of being a tidy prefix of one.
    total = len(scheduled)
    step = max(1, total // 3) | 1
    while gcd(step, total) != 1:
        step += 2

    for fraction in fractions:
        take = int(round(total * fraction))
        committed = [scheduled[(index * step) % total] for index in range(take)]
        pending = pending_game_ids(namespace, iteration, committed)
        union = set(pending) | set(committed)
        record = {
            "fraction": fraction,
            "committed": len(committed),
            "pending": len(pending),
            "union_equals_scheduled": union == set(scheduled),
            "disjoint": not (set(pending) & set(committed)),
        }
        if not record["union_equals_scheduled"]:
            problems.append(f"fraction {fraction}: pending + committed != scheduled")
        if not record["disjoint"]:
            problems.append(f"fraction {fraction}: pending overlaps committed")
        rebuilt_mismatches = sum(
            1
            for game_id in pending
            if rebuild_scheduled_game(game_id).to_dict() != reference[game_id]
        )
        record["rebuild_mismatches"] = rebuilt_mismatches
        if rebuilt_mismatches:
            problems.append(
                f"fraction {fraction}: {rebuilt_mismatches} pending id(s) rebuilt "
                f"differently from the reference schedule"
            )
        checks.append(record)

    # An id from a neighbouring iteration is scheduled work, just not *this*
    # iteration's: accepting it would let a resume seal an incomplete rollout.
    other = iteration + 1 if iteration < run_iterations(namespace) else iteration - 1
    foreign_rejected = True
    try:
        pending_game_ids(
            namespace, iteration, [phase9_game_id(namespace, other, BUCKET_CURRENT, 0)]
        )
        foreign_rejected = False
    except Phase9ScheduleError:
        pass
    if not foreign_rejected:
        problems.append("a foreign committed game id was accepted by the resume path")

    return {
        "namespace": namespace,
        "iteration": int(iteration),
        "scheduled": len(scheduled),
        "checks": checks,
        "foreign_committed_id_rejected": foreign_rejected,
        "problems": problems,
    }


def audit_setup_assignment(
    namespace: str,
    iteration: int,
    *,
    limit: "int | None" = None,
    forbidden_setups=(),
) -> dict:
    """Resolve real setups and prove the split, the families and the isolation.

    This is the one audit that touches setup *contents*: it runs the frozen
    train-split source over scheduled games and checks that every side is a
    train-split draw, that all 16 families appear, and that no resolved board
    equals a held-out Phase 9 evaluation-bank board. `forbidden_setups` is the
    set of serialized bank setups; passing it empty checks the split only.
    """
    source = _setup_source()
    if source.split != TRAINING_SPLIT or source.purpose != TRAINING_PURPOSE:
        raise Phase9ScheduleError(
            f"the production setup source must be the {TRAINING_SPLIT!r} split for "
            f"{TRAINING_PURPOSE!r}, got {source.split!r}/{source.purpose!r}"
        )

    game_ids = list(iteration_game_ids(namespace, iteration))
    if limit is not None:
        game_ids = game_ids[:limit]
    forbidden = set(forbidden_setups)

    families: dict = {}
    split_violations = 0
    leaked: list = []
    identical_sides = 0
    problems: list = []

    for game_id in game_ids:
        root = setup_root_seed(game_id)
        assignment = source.assign(
            root_seed=root,
            environment_id=SETUP_ENVIRONMENT_ID,
            generation=SETUP_GENERATION,
            game_id=game_id,
        )
        provenance = assignment.provenance
        if provenance["split"] != TRAINING_SPLIT:
            split_violations += 1
        for side in ("red", "blue"):
            record = provenance[side]
            if record["split"] != TRAINING_SPLIT:
                split_violations += 1
            family = record["primary_family_id"]
            families[family] = families.get(family, 0) + 1
            serialized = record["engine_setup"]
            if serialized in forbidden:
                leaked.append({"game_id": game_id, "side": side})
        if provenance["red"]["engine_setup"] == provenance["blue"]["engine_setup"]:
            identical_sides += 1

    if split_violations:
        problems.append(
            f"{split_violations} setup record(s) outside the {TRAINING_SPLIT!r} split"
        )
    if leaked:
        problems.append(
            f"{len(leaked)} rollout setup(s) equal a held-out evaluation-bank setup"
        )
    missing = sorted({f"F{index:02d}" for index in range(16)} - set(families))
    if missing:
        problems.append(f"setup families absent from the sampled rollout: {missing}")

    sides = 2 * len(game_ids)
    counts = sorted(families.values())
    return {
        "namespace": namespace,
        "iteration": int(iteration),
        "games_sampled": len(game_ids),
        "setup_sides_resolved": sides,
        "split": source.split,
        "purpose": source.purpose,
        "profile": source.profile,
        "setup_source_identity": source.setup_family,
        "families_seen": len(families),
        "family_counts": dict(sorted(families.items())),
        "family_min_count": counts[0] if counts else 0,
        "family_max_count": counts[-1] if counts else 0,
        "expected_per_family": round(sides / 16, 1) if sides else 0,
        "split_violations": split_violations,
        "held_out_setup_leaks": len(leaked),
        "held_out_setups_compared": len(forbidden),
        "games_with_identical_sides": identical_sides,
        "problems": problems,
    }


__all__ = [
    "ANCHOR_POLICY_TOKEN",
    "ARCHIVE_TOKEN_PREFIX",
    "BEHAVIOR_TOKEN_PREFIX",
    "SETUP_ENVIRONMENT_ID",
    "SETUP_GENERATION",
    "ActiveHistoryManifest",
    "Phase9ScheduleError",
    "ScheduledGame",
    "audit_cross_namespace_collisions",
    "audit_iteration",
    "audit_namespace",
    "audit_resume_identity",
    "audit_seed_collisions",
    "audit_setup_assignment",
    "audit_worker_order_independence",
    "behavior_policy_token",
    "behavior_snapshot_identity",
    "historical_policy_token",
    "iter_iteration_schedule",
    "iter_run_schedule",
    "iteration_game_ids",
    "iteration_schedule",
    "iteration_schedule_digest",
    "iteration_schedule_document",
    "pending_game_ids",
    "population_digest",
    "population_document",
    "rebuild_scheduled_game",
    "resume_plan",
    "rule_policy_token",
    "run_game_ids",
    "run_iterations",
    "run_schedule_digest",
    "run_schedule_document",
    "scheduled_game_record",
    "total_scheduled_games",
]
