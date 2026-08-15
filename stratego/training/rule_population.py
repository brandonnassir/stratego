"""Phase 8 Agent 2: playing one logical corpus game with the frozen rule agents.

Specification sources:

- `02_AGENT_2_SYNTHETIC_CORPUS.md` ("Rule-agent behavior", "Required per-game
  metadata", "Determinism")
- `00_PHASE_8_SEQUENCE_AND_COMMON_CONTRACT.md` sections 9-13 (frozen synthetic
  population, policy-supervision rule, split/schedule, game identity, compact
  storage)

What lives here
---------------
The bridge between three already-frozen things — the Phase 4 rule policies, the
Phase 7 setup source, and `trajectory_v1` — and one synthetic game id. Nothing
here decides anything: the roster, the weights, the schedule, the seeds and the
rules context all come from :mod:`stratego.training.warmstart_contract`, and
every rule of play comes from `stratego.engine`.

The one property this module exists to guarantee is that

```text
game content = f(synthetic_game_id)
```

exactly. :func:`play_corpus_game` takes an identifier and nothing else that can
vary: the setups come from the identity's `setup_root_seed`, each side's policy
takes its match seed from the identity's `policy:red` / `policy:blue` stream,
and each ply's randomness is the frozen Phase 4
`derive_decision_seed(policy_seed, ply)`. Worker count, process partitioning,
arrival order and resume boundaries are not inputs, so they cannot be outputs.

No neural model participates. The corpus is played entirely by the accepted
Phase 4 population, which is what makes Phase 8 a supervised warm start rather
than self-play.

What the stored decision distribution means
-------------------------------------------
`trajectory_v1` stores one probability per legal action. A rule policy publishes
no distribution, so the record stores the **realized** decision: 1.0 on the
action the policy actually chose and 0.0 elsewhere. That is the honest compact
form of "this policy selected this action at this ply", and it is exactly the
Phase 8 policy target (`00_...` section 16.1). It is deliberately *not* a claim
about the policy's behaviour distribution, and Phase 8 uses no importance ratio,
so nothing consumes it as one. The value slot carries the frozen neutral
placeholder for the same reason: Phase 8's value target is the game's final
outcome, never a stored prediction.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..engine.constants import PLAYER_NAMES, PLAYERS
from ..engine.legal_moves import legal_actions
from ..engine.setup import serialize_setup
from ..engine.state import create_game
from ..engine.transition import apply_action
from ..evaluation.policy import Policy, PolicyContractError, PolicyRef, build_policy_input
from ..evaluation.registry import POLICY_INDEX, build_policy
from .setup_source import LibrarySetupSource
from .trajectory import (
    DEFAULT_SNAPSHOT_INTERVAL,
    GameRecord,
    GameTrajectoryBuilder,
    validate_game_record,
)
from .warmstart_contract import (
    CORPUS_RULES,
    EXPECTED_TEACHER_ROSTER,
    SETUP_SOURCE_ENVIRONMENT_ID,
    SETUP_SOURCE_GENERATION,
    corpus_setup_source,
    ordered_matchup_cells,
    policy_weight,
)
from .warmstart_seed import (
    SYNTHETIC_CORPUS_VERSION,
    blue_policy_seed,
    parse_synthetic_game_id,
    red_policy_seed,
    setup_root_seed,
)

#: Version of the corpus *play* semantics: which policy acts, which seed it
#: receives, what a decision stores. A change here is a new corpus version.
RULE_POPULATION_VERSION = "warmstart_rule_population_v1"

#: The neutral value placeholder written into every stored decision. Phase 8
#: supervises value from the final outcome, so no stored prediction is a target;
#: a constant keeps the field finite, normalized and obviously non-informative.
NEUTRAL_VALUE_PREDICTION = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


class RulePopulationError(RuntimeError):
    """A corpus game could not be played under the frozen contracts."""


# ---------------------------------------------------------------------------
# The teacher population
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherPolicy:
    """One frozen teacher: its identity, its role and its supervision weight."""

    policy_id: str
    policy_version: str
    role: str
    policy_weight: float

    @property
    def token(self) -> str:
        return f"{self.policy_id}@{self.policy_version}"

    @property
    def ref(self) -> PolicyRef:
        return PolicyRef(self.policy_id, self.policy_version)


def teacher_by_token(token: str) -> TeacherPolicy:
    """The frozen teacher named by an `id@version` token."""
    for policy_id, version, role in EXPECTED_TEACHER_ROSTER:
        if f"{policy_id}@{version}" == token:
            return TeacherPolicy(policy_id, version, role, policy_weight(policy_id))
    raise RulePopulationError(f"{token!r} is not in the frozen Phase 8 teacher roster")


def build_teacher(token: str) -> Policy:
    """Instantiate the live Phase 4 policy a token names, version-checked.

    The version check is the point: a silently re-versioned policy would play
    games recorded under the old identifier, and the whole corpus identity is
    built on those tokens.
    """
    teacher = teacher_by_token(token)
    policy = build_policy(teacher.policy_id)
    if policy.policy_version != teacher.policy_version:
        raise RulePopulationError(
            f"the live catalogue provides {policy.policy_id}@{policy.policy_version}, "
            f"but the frozen roster names {teacher.token}; a corpus cannot be "
            "generated against a different policy version"
        )
    return policy


class TeacherCache:
    """Policy instances reused across the games of one process.

    Phase 4 Agent 2 established that a policy instance carries no state between
    decisions, so one instance per identifier per process is safe and saves
    rebuilding ten objects per game.
    """

    def __init__(self) -> None:
        self._policies: dict[str, Policy] = {}

    def get(self, token: str) -> Policy:
        policy = self._policies.get(token)
        if policy is None:
            policy = build_teacher(token)
            self._policies[token] = policy
        return policy

    def __len__(self) -> int:
        return len(self._policies)


def roster_digest() -> str:
    """SHA-256 over the frozen roster tokens, roles and supervision weights.

    Recorded in the corpus manifest so a later agent can prove the corpus was
    generated by exactly this population with exactly these weights.
    """
    payload = "|".join(
        f"{policy_id}@{version}:{role}:{policy_weight(policy_id):.3f}"
        for policy_id, version, role in EXPECTED_TEACHER_ROSTER
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_live_population() -> list:
    """Every disagreement between the frozen roster and the live Phase 4 code."""
    problems: list[str] = []
    for policy_id, version, role in EXPECTED_TEACHER_ROSTER:
        policy_class = POLICY_INDEX.get(policy_id)
        if policy_class is None:
            problems.append(f"{policy_id}: absent from the live Phase 4 registry")
            continue
        if policy_class.policy_version != version:
            problems.append(
                f"{policy_id}: live version {policy_class.policy_version!r} differs "
                f"from the frozen {version!r}"
            )
        if role == "stress" and not policy_id.startswith("stress_"):
            problems.append(f"{policy_id}: frozen stress role disagrees with the id")
    return problems


# ---------------------------------------------------------------------------
# One played game
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusGame:
    """One finished logical corpus game: the record and its metadata.

    `record` is the `trajectory_v1` payload and remains the game-history
    authority. `metadata` is the synthetic sidecar the corpus stores next to it
    — identity, seeds, provenance, outcome — and is never consulted for
    anything the record itself determines.
    """

    game_id: str
    record: GameRecord
    metadata: dict

    @property
    def total_decisions(self) -> int:
        return len(self.record.decisions)


def _cell_lookup() -> dict:
    return {
        (cell["red_token"], cell["blue_token"]): cell for cell in ordered_matchup_cells()
    }


_CELLS = None


def matchup_cell(red_token: str, blue_token: str) -> dict:
    """The frozen ordered-matchup cell of one (red, blue) pair."""
    global _CELLS
    if _CELLS is None:
        _CELLS = _cell_lookup()
    try:
        return _CELLS[(red_token, blue_token)]
    except KeyError:
        raise RulePopulationError(
            f"({red_token!r}, {blue_token!r}) is not one of the 100 frozen ordered "
            "matchup cells"
        ) from None


def ordered_matchup_id(red_token: str, blue_token: str) -> str:
    """Stable text identifier of one ordered cell, `red>blue`."""
    return f"{red_token}>{blue_token}"


def play_corpus_game(
    game_id: str,
    *,
    setup_source: "LibrarySetupSource | None" = None,
    teachers: "TeacherCache | None" = None,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
) -> CorpusGame:
    """Play the one logical game a synthetic game id names.

    Every input is derived from `game_id`: the split selects the frozen setup
    source, `setup_root_seed` draws both setups through the accepted Phase 7
    path, and `red_policy_seed` / `blue_policy_seed` seed the two rule policies.
    `setup_source` and `teachers` are caches, not configuration — passing a
    source for a different split raises rather than quietly changing the game.

    The engine is the sole legality and termination authority; a policy that
    returns an illegal action fails loudly here instead of being replaced by a
    substituted legal move.
    """
    identity = parse_synthetic_game_id(game_id)
    split = identity["split"]
    red_token = identity["red_token"]
    blue_token = identity["blue_token"]
    cell = matchup_cell(red_token, blue_token)

    source = corpus_setup_source(split) if setup_source is None else setup_source
    if source.split != split:
        raise RulePopulationError(
            f"game {game_id} belongs to split {split!r} but the supplied setup "
            f"source samples {source.split!r}"
        )

    cache = TeacherCache() if teachers is None else teachers
    red_teacher = teacher_by_token(red_token)
    blue_teacher = teacher_by_token(blue_token)
    policies = {PLAYERS[0]: cache.get(red_token), PLAYERS[1]: cache.get(blue_token)}
    teacher_by_player = {PLAYERS[0]: red_teacher, PLAYERS[1]: blue_teacher}

    root_seed = setup_root_seed(game_id)
    seeds = {PLAYERS[0]: red_policy_seed(game_id), PLAYERS[1]: blue_policy_seed(game_id)}

    assignment = source.assign(
        root_seed=root_seed,
        environment_id=SETUP_SOURCE_ENVIRONMENT_ID,
        generation=SETUP_SOURCE_GENERATION,
        game_id=game_id,
    )
    if assignment.provenance is None:  # pragma: no cover - library source always emits
        raise RulePopulationError(
            f"the setup source for split {split!r} produced no provenance; the "
            "corpus cannot record where a setup came from"
        )

    state = create_game(
        assignment.red_setup,
        assignment.blue_setup,
        rules=CORPUS_RULES,
        game_id=game_id,
    )
    builder = GameTrajectoryBuilder(
        game_id=game_id,
        environment_id=SETUP_SOURCE_ENVIRONMENT_ID,
        generation=SETUP_SOURCE_GENERATION,
        red_setup=assignment.red_setup,
        blue_setup=assignment.blue_setup,
        rules=CORPUS_RULES,
        root_seed=root_seed,
        slot_seed=0,
        snapshot_interval=snapshot_interval,
        collection_policy_version=RULE_POPULATION_VERSION,
        setup_family=source.setup_family,
    )

    while not state.terminal:
        actor = state.acting_player
        policy = policies[actor]
        legal = legal_actions(state)
        request = build_policy_input(
            state,
            policy=policy.ref,
            policy_seed=seeds[actor],
            requirements=policy.requirements,
            game_id=game_id,
            legal=legal,
        )
        try:
            result = policy.decide_checked(request)
        except PolicyContractError as error:
            raise RulePopulationError(
                f"game {game_id}: policy {policy.ref.token} violated its contract at "
                f"ply {request.ply}: {error}"
            ) from error
        selected = int(result.selected_action_id)
        builder.record_decision(
            state,
            legal_action_ids=legal,
            # The realized decision, not a behaviour distribution; see the module
            # docstring. `legal` is ascending, so the one-hot lands on the right
            # entry by construction.
            probabilities=tuple(
                1.0 if action == selected else 0.0 for action in legal
            ),
            win_draw_loss_prediction=NEUTRAL_VALUE_PREDICTION,
            selected_action_id=selected,
            collection_policy_version=policy.ref.token,
        )
        apply_action(state, selected, legal=legal)

    record = builder.finish(state)
    problems = validate_game_record(record)
    if problems:
        raise RulePopulationError(
            f"game {game_id}: the sealed trajectory is invalid: {problems}"
        )

    metadata = build_game_metadata(
        game_id=game_id,
        identity=identity,
        cell=cell,
        record=record,
        assignment_provenance=assignment.provenance,
        root_seed=root_seed,
        red_seed=seeds[PLAYERS[0]],
        blue_seed=seeds[PLAYERS[1]],
        red_teacher=teacher_by_player[PLAYERS[0]],
        blue_teacher=teacher_by_player[PLAYERS[1]],
    )
    return CorpusGame(game_id=game_id, record=record, metadata=metadata)


#: Every key a persisted metadata record must carry, in emission order. Checked
#: on write and again at finalization, so a schema drift cannot reach Agent 3.
METADATA_FIELDS = (
    "corpus_version",
    "corpus_split",
    "synthetic_game_id",
    "ordered_matchup_id",
    "cell_index",
    "matchup_ordinal",
    "red_policy_id",
    "red_policy_version",
    "red_policy_seed",
    "red_policy_weight",
    "blue_policy_id",
    "blue_policy_version",
    "blue_policy_seed",
    "blue_policy_weight",
    "setup_root_seed",
    "setup_environment_id",
    "setup_generation",
    "setup_provenance",
    "red_setup",
    "blue_setup",
    "setup_family",
    "setup_id",
    "trajectory_version",
    "snapshot_interval",
    "first_player",
    "rules_context",
    "terminal_result",
    "terminal_reason",
    "final_ply",
    "total_decisions",
    "rule_population_version",
)


def build_game_metadata(
    *,
    game_id: str,
    identity: dict,
    cell: dict,
    record: GameRecord,
    assignment_provenance: dict,
    root_seed: int,
    red_seed: int,
    blue_seed: int,
    red_teacher: TeacherPolicy,
    blue_teacher: TeacherPolicy,
) -> dict:
    """The synthetic sidecar of one game, in the frozen field order."""
    return {
        "corpus_version": SYNTHETIC_CORPUS_VERSION,
        "corpus_split": identity["split"],
        "synthetic_game_id": game_id,
        "ordered_matchup_id": ordered_matchup_id(
            identity["red_token"], identity["blue_token"]
        ),
        "cell_index": int(cell["cell_index"]),
        "matchup_ordinal": int(identity["ordinal"]),
        "red_policy_id": red_teacher.policy_id,
        "red_policy_version": red_teacher.policy_version,
        "red_policy_seed": int(red_seed),
        "red_policy_weight": float(red_teacher.policy_weight),
        "blue_policy_id": blue_teacher.policy_id,
        "blue_policy_version": blue_teacher.policy_version,
        "blue_policy_seed": int(blue_seed),
        "blue_policy_weight": float(blue_teacher.policy_weight),
        "setup_root_seed": int(root_seed),
        "setup_environment_id": SETUP_SOURCE_ENVIRONMENT_ID,
        "setup_generation": SETUP_SOURCE_GENERATION,
        "setup_provenance": assignment_provenance,
        "red_setup": serialize_setup(record.red_setup),
        "blue_setup": serialize_setup(record.blue_setup),
        "setup_family": record.setup_family,
        "setup_id": record.setup_id,
        "trajectory_version": record.trajectory_version,
        "snapshot_interval": int(record.snapshot_interval),
        "first_player": record.first_player,
        "rules_context": record.rules_context,
        "terminal_result": record.terminal_result,
        "terminal_reason": record.terminal_reason,
        "final_ply": int(record.final_ply),
        "total_decisions": len(record.decisions),
        "rule_population_version": RULE_POPULATION_VERSION,
    }


def validate_game_metadata(metadata: dict, record: "GameRecord | None" = None) -> list:
    """Every schema or agreement violation in one metadata record.

    With `record`, also checks that the sidecar agrees with the trajectory it
    accompanies — the identity, the setups, the outcome and the decision count.
    A sidecar that disagrees with its own record is exactly what the commit rule
    exists to keep out of the dataset.
    """
    problems: list[str] = []
    for key in METADATA_FIELDS:
        if key not in metadata:
            problems.append(f"missing metadata field {key!r}")
    if problems:
        return problems

    game_id = str(metadata["synthetic_game_id"])
    try:
        identity = parse_synthetic_game_id(game_id)
    except Exception as error:  # noqa: BLE001 - a malformed id is a finding
        return [f"{game_id!r}: {error}"]

    if metadata["corpus_version"] != SYNTHETIC_CORPUS_VERSION:
        problems.append(f"corpus version {metadata['corpus_version']!r}")
    if metadata["corpus_split"] != identity["split"]:
        problems.append(
            f"metadata split {metadata['corpus_split']!r} disagrees with the game id"
        )
    if metadata["matchup_ordinal"] != identity["ordinal"]:
        problems.append("matchup ordinal disagrees with the game id")

    red_token = f"{metadata['red_policy_id']}@{metadata['red_policy_version']}"
    blue_token = f"{metadata['blue_policy_id']}@{metadata['blue_policy_version']}"
    if red_token != identity["red_token"] or blue_token != identity["blue_token"]:
        problems.append("policy tokens disagree with the game id")
    else:
        cell = matchup_cell(red_token, blue_token)
        if metadata["cell_index"] != cell["cell_index"]:
            problems.append("cell index disagrees with the frozen schedule")
        if metadata["ordered_matchup_id"] != ordered_matchup_id(red_token, blue_token):
            problems.append("ordered matchup id disagrees with the policy tokens")
        for side, token in (("red", red_token), ("blue", blue_token)):
            expected = teacher_by_token(token).policy_weight
            if float(metadata[f"{side}_policy_weight"]) != expected:
                problems.append(
                    f"{side} policy weight {metadata[f'{side}_policy_weight']} is not "
                    f"the frozen {expected}"
                )

    for name, expected in (
        ("setup_root_seed", setup_root_seed(game_id)),
        ("red_policy_seed", red_policy_seed(game_id)),
        ("blue_policy_seed", blue_policy_seed(game_id)),
    ):
        if int(metadata[name]) != expected:
            problems.append(f"{name} is not the value derived from the game id")

    provenance = metadata.get("setup_provenance")
    if not isinstance(provenance, dict):
        problems.append("setup_provenance is missing or not an object")
    else:
        if provenance.get("split") != identity["split"]:
            problems.append(
                f"provenance split {provenance.get('split')!r} is not the game's split"
            )
        if provenance.get("game_id") not in ("", game_id):
            problems.append("provenance names a different game id")

    if record is not None:
        if record.game_id != game_id:
            problems.append("the record's game id is not the metadata's")
        if serialize_setup(record.red_setup) != metadata["red_setup"]:
            problems.append("red setup disagrees with the trajectory")
        if serialize_setup(record.blue_setup) != metadata["blue_setup"]:
            problems.append("blue setup disagrees with the trajectory")
        if record.terminal_result != metadata["terminal_result"]:
            problems.append("terminal result disagrees with the trajectory")
        if record.terminal_reason != metadata["terminal_reason"]:
            problems.append("terminal reason disagrees with the trajectory")
        if record.final_ply != metadata["final_ply"]:
            problems.append("final ply disagrees with the trajectory")
        if len(record.decisions) != metadata["total_decisions"]:
            problems.append("decision count disagrees with the trajectory")
        if record.setup_id != metadata["setup_id"]:
            problems.append("setup id disagrees with the trajectory")
        if record.setup_family != metadata["setup_family"]:
            problems.append("setup family disagrees with the trajectory")
        if record.snapshot_interval != metadata["snapshot_interval"]:
            problems.append("snapshot interval disagrees with the trajectory")
        if PLAYER_NAMES[CORPUS_RULES.first_player] != metadata["first_player"]:
            problems.append("first player disagrees with the frozen corpus rules")
    return problems


def acting_policy_weight(metadata: dict, acting_player: int) -> float:
    """The frozen supervision weight of whoever acted at one ply."""
    if acting_player == PLAYERS[0]:
        return float(metadata["red_policy_weight"])
    if acting_player == PLAYERS[1]:
        return float(metadata["blue_policy_weight"])
    raise RulePopulationError(f"unknown acting player: {acting_player!r}")


__all__ = [
    "METADATA_FIELDS",
    "NEUTRAL_VALUE_PREDICTION",
    "RULE_POPULATION_VERSION",
    "CorpusGame",
    "RulePopulationError",
    "TeacherCache",
    "TeacherPolicy",
    "acting_policy_weight",
    "build_game_metadata",
    "build_teacher",
    "matchup_cell",
    "ordered_matchup_id",
    "play_corpus_game",
    "roster_digest",
    "teacher_by_token",
    "validate_game_metadata",
    "verify_live_population",
]
