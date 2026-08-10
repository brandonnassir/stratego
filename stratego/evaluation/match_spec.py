"""Deterministic match identity and paired match scheduling.

Specification sources:

- `08_internal_state_spec.md` section 3 (rules configuration)
- Phase 4 Agent 1 instructions ("Match identity contract", "Paired evaluation unit")

Match identity
--------------
A :class:`MatchSpec` names one game completely. Its `match_id` is a hash over
exactly the components that determine how the game is played:

```text
match_spec_version, suite_version, pairing_mode,
candidate policy id@version, opponent policy id@version,
setup bank version, setup_pair_id, candidate colour, replicate, root_seed,
the full rules configuration
```

Deliberately absent: worker index, position in a schedule, wall-clock time,
process identifier, and how many workers ran. So the same identifier always
implies the same setups, colours, policy seeds, first player and rules, and
re-sharding a schedule across a different number of workers cannot change a
single game.

Paired evaluation unit
----------------------
Pairing mode `color_swap_same_board` -- the only mode implemented, and the one
this project uses:

```text
Game A: red_setup = R, blue_setup = B, candidate plays RED
Game B: red_setup = R, blue_setup = B, candidate plays BLUE
```

Both games start from the *identical physical position*; only which policy
controls which colour flips. This matters because a setup is stored in each
player's own `SETUP_SQUARES` order, and those orders are not symmetric -- red's
index 0 is his back row while blue's is her front row -- so "give the candidate
the same setup on the other side" is a board transformation, not a relabelling.
Holding the board fixed avoids the transformation entirely and cancels two
confounders exactly:

- setup-quality asymmetry, since both policies play both arrangements;
- first-move advantage, since `first_player` is red and each policy is red once.

:mod:`stratego.evaluation.setup_bank` documents the orientation transform for
anyone who later wants the alternative pairing; it is not used here.
"""

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..engine.constants import BLUE, EVALUATION_RULES, PLAYER_NAMES, PLAYERS, RED, RulesConfig
from .policy import PolicyRef
from .setup_bank import SETUP_BANK_VERSION, SetupBank, SetupPair

MATCH_SPEC_VERSION = "match_spec_v1"
EVALUATION_SUITE_VERSION = "phase4_evaluation_suite_v1"

#: The only pairing mode implemented. See the module docstring.
PAIRING_COLOR_SWAP_SAME_BOARD = "color_swap_same_board"
PAIRING_MODES = (PAIRING_COLOR_SWAP_SAME_BOARD,)

ROLE_CANDIDATE = "candidate"
ROLE_OPPONENT = "opponent"
ROLES = (ROLE_CANDIDATE, ROLE_OPPONENT)

DEFAULT_ROOT_SEED = 20260401


class MatchSpecError(ValueError):
    """Raised when a match specification is malformed or inconsistent."""


# ---------------------------------------------------------------------------
# Canonical identity payloads
# ---------------------------------------------------------------------------


def rules_token(rules: RulesConfig) -> str:
    """Canonical text for a rules configuration, for use inside identity hashes.

    Every field is included, so a match identifier can never silently mean two
    different rule sets.
    """
    return "|".join(
        (
            rules.rules_version,
            rules.board_geometry_version,
            f"first_player={rules.first_player}",
            f"battleless_move_limit={rules.battleless_move_limit}",
            f"absolute_move_limit={rules.absolute_move_limit}",
            f"two_square_rule_enabled={int(rules.two_square_rule_enabled)}",
            f"continuous_chasing_rule_enabled={int(rules.continuous_chasing_rule_enabled)}",
            f"context={rules.context}",
        )
    )


def _hash(payload: str, person: bytes) -> str:
    return hashlib.blake2b(payload.encode(), digest_size=12, person=person).hexdigest()


def _derive_seed(payload: str, person: bytes) -> int:
    digest = hashlib.blake2b(payload.encode(), digest_size=8, person=person).digest()
    return int.from_bytes(digest, "big") >> 1


def derive_policy_seed(match_id: str, role: str) -> int:
    """The match-level seed handed to one side's policy.

    Derived from the match identifier, so it is fixed by identity alone and no
    scheduler, worker or clock can influence it.
    """
    if role not in ROLES:
        raise MatchSpecError(f"unknown role: {role!r}")
    return _derive_seed(f"{match_id}:{role}", b"strat-pls")


# ---------------------------------------------------------------------------
# Match specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchSpec:
    """One completely determined evaluation game.

    All derived values -- identifiers, seeds, colour assignment -- are
    properties rather than stored fields, so a specification cannot be
    constructed carrying an identifier that disagrees with its components.
    """

    candidate: PolicyRef
    opponent: PolicyRef
    setup_pair_id: int
    candidate_color: int
    replicate: int = 0
    root_seed: int = DEFAULT_ROOT_SEED
    suite_version: str = EVALUATION_SUITE_VERSION
    setup_bank_version: str = SETUP_BANK_VERSION
    pairing_mode: str = PAIRING_COLOR_SWAP_SAME_BOARD
    rules: RulesConfig = EVALUATION_RULES

    def __post_init__(self) -> None:
        if self.candidate_color not in PLAYERS:
            raise MatchSpecError(f"candidate_color must be RED or BLUE, got {self.candidate_color!r}")
        if self.pairing_mode not in PAIRING_MODES:
            raise MatchSpecError(f"unknown pairing mode: {self.pairing_mode!r}")
        if self.setup_pair_id < 0:
            raise MatchSpecError(f"setup_pair_id must be non-negative, got {self.setup_pair_id}")
        if self.replicate < 0:
            raise MatchSpecError(f"replicate must be non-negative, got {self.replicate}")

    # -- identity ----------------------------------------------------------

    @property
    def opponent_color(self) -> int:
        return BLUE if self.candidate_color == RED else RED

    @property
    def _unit_payload(self) -> str:
        """Identity of the paired unit: everything except the colour assignment."""
        return "|".join(
            (
                MATCH_SPEC_VERSION,
                self.suite_version,
                self.pairing_mode,
                self.candidate.token,
                self.opponent.token,
                self.setup_bank_version,
                f"setup_pair_id={self.setup_pair_id}",
                f"replicate={self.replicate}",
                f"root_seed={self.root_seed}",
                rules_token(self.rules),
            )
        )

    @property
    def paired_unit_id(self) -> str:
        return "u-" + _hash(self._unit_payload, b"strat-unt")

    @property
    def match_id(self) -> str:
        payload = f"{self._unit_payload}|candidate_color={self.candidate_color}"
        return "m-" + _hash(payload, b"strat-mch")

    @property
    def game_id(self) -> str:
        """Engine-visible game identifier; carried into replays and events."""
        return self.match_id

    # -- seeds -------------------------------------------------------------

    @property
    def candidate_seed(self) -> int:
        return derive_policy_seed(self.match_id, ROLE_CANDIDATE)

    @property
    def opponent_seed(self) -> int:
        return derive_policy_seed(self.match_id, ROLE_OPPONENT)

    def policy_seed_for(self, player: int) -> int:
        return self.candidate_seed if player == self.candidate_color else self.opponent_seed

    def policy_ref_for(self, player: int) -> PolicyRef:
        return self.candidate if player == self.candidate_color else self.opponent

    def role_for(self, player: int) -> str:
        return ROLE_CANDIDATE if player == self.candidate_color else ROLE_OPPONENT

    # -- setups ------------------------------------------------------------

    @property
    def first_player(self) -> int:
        return self.rules.first_player

    @property
    def candidate_moves_first(self) -> bool:
        return self.candidate_color == self.first_player

    def resolve_setups(self, bank: SetupBank) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """`(red_setup, blue_setup)` for this match.

        Under `color_swap_same_board` the board does not depend on the colour
        assignment at all: both games of a paired unit resolve to the identical
        pair of setups.
        """
        if bank.bank_version != self.setup_bank_version:
            raise MatchSpecError(
                f"match requires setup bank {self.setup_bank_version!r} but was given "
                f"{bank.bank_version!r}"
            )
        pair: SetupPair = bank.pair(self.setup_pair_id)
        return pair.red_setup, pair.blue_setup

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "match_spec_version": MATCH_SPEC_VERSION,
            "match_id": self.match_id,
            "paired_unit_id": self.paired_unit_id,
            "suite_version": self.suite_version,
            "pairing_mode": self.pairing_mode,
            "candidate": self.candidate.to_dict(),
            "opponent": self.opponent.to_dict(),
            "setup_bank_version": self.setup_bank_version,
            "setup_pair_id": self.setup_pair_id,
            "candidate_color": self.candidate_color,
            "candidate_color_name": PLAYER_NAMES[self.candidate_color],
            "replicate": self.replicate,
            "root_seed": self.root_seed,
            "rules": rules_token(self.rules),
            "candidate_seed": self.candidate_seed,
            "opponent_seed": self.opponent_seed,
            "first_player": self.first_player,
        }

    @staticmethod
    def from_dict(payload: "dict[str, Any]", rules: RulesConfig = EVALUATION_RULES) -> "MatchSpec":
        """Rebuild a specification. Raises if the stored identifiers disagree.

        `rules` must be supplied because a rules configuration is a frozen
        object, not a serialised blob; the token check below proves the caller
        supplied the right one.
        """
        spec = MatchSpec(
            candidate=PolicyRef.from_dict(payload["candidate"]),
            opponent=PolicyRef.from_dict(payload["opponent"]),
            setup_pair_id=int(payload["setup_pair_id"]),
            candidate_color=int(payload["candidate_color"]),
            replicate=int(payload["replicate"]),
            root_seed=int(payload["root_seed"]),
            suite_version=str(payload["suite_version"]),
            setup_bank_version=str(payload["setup_bank_version"]),
            pairing_mode=str(payload["pairing_mode"]),
            rules=rules,
        )
        stored = payload.get("match_id")
        if stored is not None and stored != spec.match_id:
            raise MatchSpecError(
                f"stored match_id {stored!r} does not match the rebuilt specification "
                f"{spec.match_id!r}; the rules configuration or a component differs"
            )
        return spec


# ---------------------------------------------------------------------------
# Paired evaluation unit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedUnit:
    """The two colour-swapped games that form one evaluation unit.

    This is the resampling unit for Agent 3's confidence intervals: the two
    games share a board and a pair of policies, so they are not independent.
    """

    candidate: PolicyRef
    opponent: PolicyRef
    setup_pair_id: int
    replicate: int = 0
    root_seed: int = DEFAULT_ROOT_SEED
    suite_version: str = EVALUATION_SUITE_VERSION
    setup_bank_version: str = SETUP_BANK_VERSION
    pairing_mode: str = PAIRING_COLOR_SWAP_SAME_BOARD
    rules: RulesConfig = EVALUATION_RULES

    def _match(self, candidate_color: int) -> MatchSpec:
        return MatchSpec(
            candidate=self.candidate,
            opponent=self.opponent,
            setup_pair_id=self.setup_pair_id,
            candidate_color=candidate_color,
            replicate=self.replicate,
            root_seed=self.root_seed,
            suite_version=self.suite_version,
            setup_bank_version=self.setup_bank_version,
            pairing_mode=self.pairing_mode,
            rules=self.rules,
        )

    @property
    def game_a(self) -> MatchSpec:
        """Candidate as red, opponent as blue. Candidate moves first."""
        return self._match(RED)

    @property
    def game_b(self) -> MatchSpec:
        """Candidate as blue, opponent as red. Opponent moves first."""
        return self._match(BLUE)

    @property
    def matches(self) -> tuple[MatchSpec, MatchSpec]:
        return (self.game_a, self.game_b)

    @property
    def paired_unit_id(self) -> str:
        return self.game_a.paired_unit_id

    def to_dict(self) -> dict:
        return {
            "paired_unit_id": self.paired_unit_id,
            "pairing_mode": self.pairing_mode,
            "match_ids": [match.match_id for match in self.matches],
            "candidate": self.candidate.to_dict(),
            "opponent": self.opponent.to_dict(),
            "setup_pair_id": self.setup_pair_id,
            "replicate": self.replicate,
            "root_seed": self.root_seed,
            "suite_version": self.suite_version,
            "setup_bank_version": self.setup_bank_version,
        }

    @staticmethod
    def from_match(spec: MatchSpec) -> "PairedUnit":
        """Recover the whole unit -- including the sibling game -- from one match."""
        return PairedUnit(
            candidate=spec.candidate,
            opponent=spec.opponent,
            setup_pair_id=spec.setup_pair_id,
            replicate=spec.replicate,
            root_seed=spec.root_seed,
            suite_version=spec.suite_version,
            setup_bank_version=spec.setup_bank_version,
            pairing_mode=spec.pairing_mode,
            rules=spec.rules,
        )


def sibling_match(spec: MatchSpec) -> MatchSpec:
    """The other game of `spec`'s paired unit."""
    unit = PairedUnit.from_match(spec)
    return unit.game_b if spec.candidate_color == RED else unit.game_a


# ---------------------------------------------------------------------------
# Scheduling primitives
# ---------------------------------------------------------------------------


def build_paired_schedule(
    candidate: PolicyRef,
    opponent: PolicyRef,
    setup_pair_ids: "Iterable[int]",
    *,
    replicates: int = 1,
    root_seed: int = DEFAULT_ROOT_SEED,
    suite_version: str = EVALUATION_SUITE_VERSION,
    setup_bank_version: str = SETUP_BANK_VERSION,
    pairing_mode: str = PAIRING_COLOR_SWAP_SAME_BOARD,
    rules: RulesConfig = EVALUATION_RULES,
) -> tuple[PairedUnit, ...]:
    """Every paired unit for one candidate/opponent matchup.

    Enumeration order is `(setup_pair_id, replicate)`, which only fixes the
    order the units are returned in; it is not part of any identifier.
    """
    if replicates < 1:
        raise MatchSpecError(f"replicates must be at least 1, got {replicates}")
    if candidate == opponent:
        raise MatchSpecError(
            f"candidate and opponent are the same policy ({candidate.token}); a "
            "mirror matchup needs two distinct policy versions"
        )
    units: list[PairedUnit] = []
    for setup_pair_id in setup_pair_ids:
        for replicate in range(replicates):
            units.append(
                PairedUnit(
                    candidate=candidate,
                    opponent=opponent,
                    setup_pair_id=int(setup_pair_id),
                    replicate=replicate,
                    root_seed=root_seed,
                    suite_version=suite_version,
                    setup_bank_version=setup_bank_version,
                    pairing_mode=pairing_mode,
                    rules=rules,
                )
            )
    return tuple(units)


def build_round_robin_schedule(
    policies: "Sequence[PolicyRef]",
    setup_pair_ids: "Iterable[int]",
    **kwargs: Any,
) -> tuple[PairedUnit, ...]:
    """Paired units for every unordered pair of distinct policies.

    Each unordered pair appears once, with the earlier policy as candidate. The
    colour swap inside each unit is what balances the matchup, so scheduling the
    reversed ordering as well would only duplicate games.
    """
    pair_ids = tuple(int(identifier) for identifier in setup_pair_ids)
    refs = list(policies)
    seen = {ref.token for ref in refs}
    if len(seen) != len(refs):
        raise MatchSpecError("round-robin policy list contains a duplicate policy")

    units: list[PairedUnit] = []
    for index, candidate in enumerate(refs):
        for challenger in refs[index + 1 :]:
            units.extend(build_paired_schedule(candidate, challenger, pair_ids, **kwargs))
    return tuple(units)


def schedule_matches(units: "Iterable[PairedUnit]") -> tuple[MatchSpec, ...]:
    """Flatten paired units into the match list a runner consumes."""
    matches: list[MatchSpec] = []
    for unit in units:
        matches.extend(unit.matches)
    return tuple(matches)


def schedule_digest(matches: "Iterable[MatchSpec]") -> str:
    """Order-independent digest of a schedule's *contents*.

    Two schedules with the same digest contain the same set of matches, whatever
    order they were produced or sharded in. Duplicates are significant, so the
    identifiers are sorted rather than de-duplicated.
    """
    identifiers = sorted(match.match_id for match in matches)
    return hashlib.sha256("\n".join(identifiers).encode()).hexdigest()


def validate_schedule(matches: "Sequence[MatchSpec]", bank: "SetupBank | None" = None) -> list[str]:
    """Structural problems in a schedule, as human-readable strings."""
    problems: list[str] = []

    seen: dict[str, MatchSpec] = {}
    for match in matches:
        if match.match_id in seen:
            problems.append(f"duplicate match_id {match.match_id}")
        seen[match.match_id] = match
        if bank is not None:
            try:
                match.resolve_setups(bank)
            except (MatchSpecError, ValueError) as error:
                problems.append(f"match {match.match_id}: {error}")

    units: dict[str, list[int]] = {}
    for match in matches:
        units.setdefault(match.paired_unit_id, []).append(match.candidate_color)
    for unit_id, colors in units.items():
        if sorted(colors) != [RED, BLUE] and sorted(colors) != [BLUE, RED]:
            problems.append(
                f"paired unit {unit_id} has colour assignments {sorted(colors)}, "
                "expected exactly one red and one blue"
            )
    return problems


def shard_schedule(
    matches: "Sequence[MatchSpec]", worker_count: int
) -> tuple[tuple[MatchSpec, ...], ...]:
    """Split a schedule across workers by position.

    Provided so that Agent 3 has one obvious place to shard. The shard index is
    not an input to any identifier, so any sharding produces the same games.
    """
    if worker_count < 1:
        raise MatchSpecError(f"worker_count must be at least 1, got {worker_count}")
    shards: list[list[MatchSpec]] = [[] for _ in range(worker_count)]
    for index, match in enumerate(matches):
        shards[index % worker_count].append(match)
    return tuple(tuple(shard) for shard in shards)


def match_identity_components(spec: MatchSpec) -> dict:
    """Exactly the values that enter `match_id`, for report tables and tests."""
    return {
        "match_spec_version": MATCH_SPEC_VERSION,
        "suite_version": spec.suite_version,
        "pairing_mode": spec.pairing_mode,
        "candidate": spec.candidate.token,
        "opponent": spec.opponent.token,
        "setup_bank_version": spec.setup_bank_version,
        "setup_pair_id": spec.setup_pair_id,
        "candidate_color": spec.candidate_color,
        "replicate": spec.replicate,
        "root_seed": spec.root_seed,
        "rules": rules_token(spec.rules),
    }
