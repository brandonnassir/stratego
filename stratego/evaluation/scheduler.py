"""Named, reproducible evaluation schedules.

Specification source: Phase 4 Agent 3 instructions ("Parallel execution",
"Reproducibility acceptance").

Agent 1 supplies the identity primitives -- `build_paired_schedule`,
`build_round_robin_schedule`, `schedule_matches`, `schedule_digest`,
`validate_schedule`, `shard_schedule`. This module is the layer above them: it
binds a list of matches to the setup bank, rules configuration and policy
versions they were built for, gives the whole thing a name and a digest, and can
write it out and read it back with the identity re-verified.

Why a schedule is a first-class object
--------------------------------------
The reproducibility gate is not "the runner is deterministic" but "an evaluator
who has only the identifiers can rebuild the same work". That needs the *set of
matches* to be storable and checkable, not just each match. A stored
:class:`EvaluationSchedule` round-trips through JSON and re-verifies both every
`match_id` and the schedule digest on load, so a schedule that was edited, or
built against a different bank version or rules configuration, fails at load
rather than quietly running different games.

Ordering
--------
Enumeration order is a presentation detail and nothing more. `digest` sorts
identifiers before hashing, :meth:`EvaluationSchedule.shuffled` exists so tests
can prove order-invariance directly, and the runner sorts its results by
`match_id`. No identifier anywhere depends on position.
"""

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..engine.constants import EVALUATION_RULES, RulesConfig
from .match_spec import (
    DEFAULT_ROOT_SEED,
    EVALUATION_SUITE_VERSION,
    PAIRING_COLOR_SWAP_SAME_BOARD,
    MatchSpec,
    PairedUnit,
    build_paired_schedule,
    rules_token,
    schedule_digest,
    schedule_matches,
    shard_schedule,
    validate_schedule,
)
from .registry import ALL_POLICY_IDS, LADDER_POLICY_IDS, UnknownPolicyError, policy_ref
from .setup_bank import SETUP_BANK_VERSION, SetupBank

SCHEDULER_VERSION = "evaluation_scheduler_v1"


class ScheduleError(ValueError):
    """Raised when a schedule is malformed, inconsistent, or unrecoverable."""


def _normalise_pair_ids(pair_ids: "Sequence[int] | int") -> tuple[int, ...]:
    """Accept either an explicit list of setup pairs or a count meaning `range(n)`."""
    if isinstance(pair_ids, int):
        if pair_ids < 1:
            raise ScheduleError(f"setup-pair count must be at least 1, got {pair_ids}")
        return tuple(range(pair_ids))
    identifiers = tuple(int(value) for value in pair_ids)
    if not identifiers:
        raise ScheduleError("a schedule needs at least one setup pair")
    if len(set(identifiers)) != len(identifiers):
        raise ScheduleError("setup-pair identifiers contain a duplicate")
    return identifiers


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationSchedule:
    """A named set of matches, bound to the versions it was built for."""

    name: str
    matches: tuple[MatchSpec, ...]
    setup_bank_version: str = SETUP_BANK_VERSION
    rules: RulesConfig = EVALUATION_RULES
    suite_version: str = EVALUATION_SUITE_VERSION
    scheduler_version: str = SCHEDULER_VERSION

    def __post_init__(self) -> None:
        if not self.matches:
            raise ScheduleError(f"schedule {self.name!r} contains no matches")

    def __len__(self) -> int:
        return len(self.matches)

    def __iter__(self):
        return iter(self.matches)

    # -- identity ----------------------------------------------------------

    @property
    def digest(self) -> str:
        """Order-independent digest of the schedule's contents."""
        return schedule_digest(self.matches)

    @property
    def paired_unit_ids(self) -> tuple[str, ...]:
        """Unit identifiers in first-appearance order, without duplicates."""
        seen: dict[str, None] = {}
        for match in self.matches:
            seen.setdefault(match.paired_unit_id, None)
        return tuple(seen)

    @property
    def policy_tokens(self) -> tuple[str, ...]:
        tokens = {match.candidate.token for match in self.matches}
        tokens |= {match.opponent.token for match in self.matches}
        return tuple(sorted(tokens))

    @property
    def matchups(self) -> tuple[tuple[str, str], ...]:
        pairs = {(match.candidate.token, match.opponent.token) for match in self.matches}
        return tuple(sorted(pairs))

    @property
    def setup_pair_ids(self) -> tuple[int, ...]:
        return tuple(sorted({match.setup_pair_id for match in self.matches}))

    def units(self) -> tuple[PairedUnit, ...]:
        """The paired units this schedule covers, one per `paired_unit_id`."""
        units: dict[str, PairedUnit] = {}
        for match in self.matches:
            units.setdefault(match.paired_unit_id, PairedUnit.from_match(match))
        return tuple(units.values())

    # -- transformation ----------------------------------------------------

    def chunks(self, chunk_count: int) -> tuple[tuple[MatchSpec, ...], ...]:
        """Deal the matches into `chunk_count` chunks for parallel dispatch."""
        return shard_schedule(self.matches, chunk_count)

    def shuffled(self, seed: int) -> "EvaluationSchedule":
        """The same schedule in a different order.

        Exists so order-invariance can be tested as a property rather than
        assumed from the fact that no identifier reads a position.
        """
        ordered = list(self.matches)
        random.Random(seed).shuffle(ordered)
        return EvaluationSchedule(
            name=self.name,
            matches=tuple(ordered),
            setup_bank_version=self.setup_bank_version,
            rules=self.rules,
            suite_version=self.suite_version,
        )

    def limited(self, match_count: int) -> "EvaluationSchedule":
        """The first `match_count` matches, rounded down to whole paired units.

        Truncating mid-unit would leave a half-unit that the paired bootstrap
        cannot use, so the cut lands on a unit boundary.
        """
        if match_count < 2:
            raise ScheduleError("a limited schedule needs at least one whole paired unit")

        grouped: dict[str, list[MatchSpec]] = {}
        for match in self.matches:
            grouped.setdefault(match.paired_unit_id, []).append(match)

        selected: list[MatchSpec] = []
        for unit_matches in grouped.values():
            if len(selected) + len(unit_matches) > match_count:
                break
            selected.extend(unit_matches)
        if not selected:
            raise ScheduleError(
                f"limiting {self.name!r} to {match_count} matches left no complete paired unit"
            )
        return EvaluationSchedule(
            name=self.name,
            matches=tuple(selected),
            setup_bank_version=self.setup_bank_version,
            rules=self.rules,
            suite_version=self.suite_version,
        )

    # -- validation --------------------------------------------------------

    def validate(self, bank: "SetupBank | None" = None) -> list[str]:
        return validate_evaluation_schedule(self, bank)

    # -- serialisation -----------------------------------------------------

    def manifest(self, bank: "SetupBank | None" = None) -> dict:
        """Everything an evaluator needs to know what this run was, minus the rows."""
        payload = {
            "scheduler_version": self.scheduler_version,
            "name": self.name,
            "schedule_digest": self.digest,
            "suite_version": self.suite_version,
            "setup_bank_version": self.setup_bank_version,
            "rules": rules_token(self.rules),
            "match_count": len(self.matches),
            "paired_unit_count": len(self.paired_unit_ids),
            "setup_pair_count": len(self.setup_pair_ids),
            "policies": list(self.policy_tokens),
            "matchups": [f"{candidate} vs {opponent}" for candidate, opponent in self.matchups],
            "pairing_modes": sorted({match.pairing_mode for match in self.matches}),
        }
        if bank is not None:
            payload["setup_bank_digest"] = bank.digest()
            payload["setup_bank_pairs"] = len(bank)
        return payload

    def to_dict(self) -> dict:
        return {
            "scheduler_version": self.scheduler_version,
            "name": self.name,
            "schedule_digest": self.digest,
            "suite_version": self.suite_version,
            "setup_bank_version": self.setup_bank_version,
            "rules": rules_token(self.rules),
            "rules_payload": {
                "rules_version": self.rules.rules_version,
                "board_geometry_version": self.rules.board_geometry_version,
                "first_player": self.rules.first_player,
                "battleless_move_limit": self.rules.battleless_move_limit,
                "absolute_move_limit": self.rules.absolute_move_limit,
                "two_square_rule_enabled": self.rules.two_square_rule_enabled,
                "continuous_chasing_rule_enabled": self.rules.continuous_chasing_rule_enabled,
                "context": self.rules.context,
            },
            "matches": [match.to_dict() for match in self.matches],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "EvaluationSchedule":
        """Rebuild a schedule, re-verifying every match identity and the digest."""
        rules = RulesConfig(**dict(payload["rules_payload"]))
        if rules_token(rules) != payload["rules"]:
            raise ScheduleError(
                f"stored rules token {payload['rules']!r} disagrees with the rebuilt "
                f"configuration {rules_token(rules)!r}"
            )
        # `MatchSpec.from_dict` raises if a stored `match_id` disagrees with the
        # components beside it, so a tampered row cannot survive this loop.
        matches = tuple(MatchSpec.from_dict(entry, rules=rules) for entry in payload["matches"])
        schedule = EvaluationSchedule(
            name=str(payload["name"]),
            matches=matches,
            setup_bank_version=str(payload["setup_bank_version"]),
            rules=rules,
            suite_version=str(payload["suite_version"]),
        )
        stored = payload.get("schedule_digest")
        if stored is not None and stored != schedule.digest:
            raise ScheduleError(
                f"stored schedule digest {stored} does not match the rebuilt schedule "
                f"{schedule.digest}; a match was added, removed or altered"
            )
        return schedule

    @staticmethod
    def from_json(text: str) -> "EvaluationSchedule":
        return EvaluationSchedule.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_matchup_schedule(
    candidate_id: str,
    opponent_id: str,
    pair_ids: "Sequence[int] | int",
    *,
    name: str | None = None,
    replicates: int = 1,
    root_seed: int = DEFAULT_ROOT_SEED,
    suite_version: str = EVALUATION_SUITE_VERSION,
    setup_bank_version: str = SETUP_BANK_VERSION,
    pairing_mode: str = PAIRING_COLOR_SWAP_SAME_BOARD,
    rules: RulesConfig = EVALUATION_RULES,
) -> EvaluationSchedule:
    """One candidate against one opponent over `pair_ids`, both colours each.

    Policy identifiers are resolved to `id@version` through the catalogue, so a
    schedule always names the version it was built against rather than "whatever
    the registry happens to hold later".
    """
    identifiers = _normalise_pair_ids(pair_ids)
    units = build_paired_schedule(
        _ref(candidate_id),
        _ref(opponent_id),
        identifiers,
        replicates=replicates,
        root_seed=root_seed,
        suite_version=suite_version,
        setup_bank_version=setup_bank_version,
        pairing_mode=pairing_mode,
        rules=rules,
    )
    return EvaluationSchedule(
        name=name or f"{candidate_id}_vs_{opponent_id}",
        matches=schedule_matches(units),
        setup_bank_version=setup_bank_version,
        rules=rules,
        suite_version=suite_version,
    )


def build_gauntlet_schedule(
    candidate_id: str,
    opponent_ids: "Sequence[str]",
    pair_ids: "Sequence[int] | int",
    *,
    name: str | None = None,
    **kwargs: Any,
) -> EvaluationSchedule:
    """One candidate against many opponents.

    This is the shape a future checkpoint evaluation takes: the candidate is the
    thing being measured and the ladder is the fixed yardstick.
    """
    if candidate_id in opponent_ids:
        raise ScheduleError(f"candidate {candidate_id!r} also appears in the opponent list")
    matches: list[MatchSpec] = []
    for opponent_id in opponent_ids:
        matches.extend(
            build_matchup_schedule(candidate_id, opponent_id, pair_ids, **kwargs).matches
        )
    return EvaluationSchedule(
        name=name or f"{candidate_id}_gauntlet",
        matches=tuple(matches),
        setup_bank_version=kwargs.get("setup_bank_version", SETUP_BANK_VERSION),
        rules=kwargs.get("rules", EVALUATION_RULES),
        suite_version=kwargs.get("suite_version", EVALUATION_SUITE_VERSION),
    )


def build_league_schedule(
    policy_ids: "Sequence[str]",
    pair_ids: "Sequence[int] | int",
    *,
    name: str = "league",
    **kwargs: Any,
) -> EvaluationSchedule:
    """Every unordered pair of distinct policies, both colours each.

    Each unordered pair appears once: the colour swap inside a paired unit is
    what balances a matchup, so also scheduling the reversed ordering would only
    duplicate games under different candidate/opponent labels.
    """
    identifiers = list(policy_ids)
    if len(set(identifiers)) != len(identifiers):
        raise ScheduleError("the league policy list contains a duplicate")
    if len(identifiers) < 2:
        raise ScheduleError("a league needs at least two policies")

    matches: list[MatchSpec] = []
    for index, candidate_id in enumerate(identifiers):
        for opponent_id in identifiers[index + 1 :]:
            matches.extend(
                build_matchup_schedule(candidate_id, opponent_id, pair_ids, **kwargs).matches
            )
    return EvaluationSchedule(
        name=name,
        matches=tuple(matches),
        setup_bank_version=kwargs.get("setup_bank_version", SETUP_BANK_VERSION),
        rules=kwargs.get("rules", EVALUATION_RULES),
        suite_version=kwargs.get("suite_version", EVALUATION_SUITE_VERSION),
    )


def build_ladder_schedule(
    pair_ids: "Sequence[int] | int", **kwargs: Any
) -> EvaluationSchedule:
    """The four-tier core ladder as a round robin. Agent 4's screening shape."""
    return build_league_schedule(
        list(LADDER_POLICY_IDS), pair_ids, name=kwargs.pop("name", "core_ladder"), **kwargs
    )


def _ref(policy_id: str):
    try:
        return policy_ref(policy_id)
    except (KeyError, UnknownPolicyError) as error:
        raise ScheduleError(
            f"unknown policy_id {policy_id!r}; the Phase 4 catalogue holds "
            f"{', '.join(ALL_POLICY_IDS)}"
        ) from error


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_evaluation_schedule(
    schedule: EvaluationSchedule, bank: "SetupBank | None" = None
) -> list[str]:
    """Structural problems in a schedule, as human-readable strings.

    Extends Agent 1's `validate_schedule` -- which covers duplicate matches,
    incomplete paired units and setup pairs missing from the bank -- with the
    consistency checks that only matter once a schedule is a stored artefact:
    that every match agrees with the schedule's declared bank version, rules and
    suite version, and that every named policy exists in the catalogue at the
    version the schedule names.
    """
    problems: list[str] = list(validate_schedule(schedule.matches, bank))
    declared_rules = rules_token(schedule.rules)

    for match in schedule.matches:
        if match.setup_bank_version != schedule.setup_bank_version:
            problems.append(
                f"match {match.match_id} names bank {match.setup_bank_version!r} but the "
                f"schedule declares {schedule.setup_bank_version!r}"
            )
        if rules_token(match.rules) != declared_rules:
            problems.append(
                f"match {match.match_id} uses rules {rules_token(match.rules)!r} but the "
                f"schedule declares {declared_rules!r}"
            )
        if match.suite_version != schedule.suite_version:
            problems.append(
                f"match {match.match_id} names suite {match.suite_version!r} but the "
                f"schedule declares {schedule.suite_version!r}"
            )

    for token in schedule.policy_tokens:
        policy_id, _, version = token.partition("@")
        try:
            catalogued = policy_ref(policy_id)
        except (KeyError, UnknownPolicyError):
            problems.append(f"policy {token} is not in the Phase 4 catalogue")
            continue
        if catalogued.policy_version != version:
            problems.append(
                f"schedule names {token} but the catalogue holds {catalogued.token}; a "
                "stored schedule cannot be replayed against a re-versioned policy"
            )

    if bank is not None:
        if bank.bank_version != schedule.setup_bank_version:
            problems.append(
                f"bank {bank.bank_version!r} does not match the schedule's declared "
                f"{schedule.setup_bank_version!r}"
            )
        available = set(bank.pair_ids)
        missing = sorted(set(schedule.setup_pair_ids) - available)
        if missing:
            problems.append(
                f"{len(missing)} setup pairs are missing from the bank "
                f"(first: {missing[:5]})"
            )

    return problems


def require_valid_schedule(
    schedule: EvaluationSchedule, bank: "SetupBank | None" = None
) -> EvaluationSchedule:
    """Validate and raise. Call this before a long run rather than after it."""
    problems = schedule.validate(bank)
    if problems:
        listed = "\n  ".join(problems[:20])
        more = "" if len(problems) <= 20 else f"\n  ... and {len(problems) - 20} more"
        raise ScheduleError(f"schedule {schedule.name!r} is invalid:\n  {listed}{more}")
    return schedule


def schedule_fingerprint(schedule: EvaluationSchedule) -> str:
    """Digest over the schedule *and* the versions it is bound to.

    `EvaluationSchedule.digest` covers the match set. This also folds in the
    bank version, rules and suite version, so two schedules that contain
    identically-identified matches but declare different bindings are
    distinguishable in a report.
    """
    payload = json.dumps(
        {
            "digest": schedule.digest,
            "setup_bank_version": schedule.setup_bank_version,
            "rules": rules_token(schedule.rules),
            "suite_version": schedule.suite_version,
            "scheduler_version": schedule.scheduler_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def merge_schedules(
    schedules: "Iterable[EvaluationSchedule]", *, name: str
) -> EvaluationSchedule:
    """Combine schedules that share a bank version, rules and suite version.

    Duplicate matches are rejected rather than de-duplicated: a duplicate means
    the caller built the same work twice, and silently collapsing it would make
    the reported match count disagree with what was scheduled.
    """
    collected = list(schedules)
    if not collected:
        raise ScheduleError("merge_schedules was given nothing to merge")
    first = collected[0]
    matches: list[MatchSpec] = []
    seen: set[str] = set()
    for schedule in collected:
        if schedule.setup_bank_version != first.setup_bank_version:
            raise ScheduleError("cannot merge schedules built against different setup banks")
        if rules_token(schedule.rules) != rules_token(first.rules):
            raise ScheduleError("cannot merge schedules built under different rules")
        if schedule.suite_version != first.suite_version:
            raise ScheduleError("cannot merge schedules from different suite versions")
        for match in schedule.matches:
            if match.match_id in seen:
                raise ScheduleError(
                    f"match {match.match_id} appears in more than one schedule being merged"
                )
            seen.add(match.match_id)
            matches.append(match)
    return EvaluationSchedule(
        name=name,
        matches=tuple(matches),
        setup_bank_version=first.setup_bank_version,
        rules=first.rules,
        suite_version=first.suite_version,
    )


__all__ = [
    "SCHEDULER_VERSION",
    "EvaluationSchedule",
    "ScheduleError",
    "build_gauntlet_schedule",
    "build_ladder_schedule",
    "build_league_schedule",
    "build_matchup_schedule",
    "merge_schedules",
    "require_valid_schedule",
    "schedule_fingerprint",
    "validate_evaluation_schedule",
]
