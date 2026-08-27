"""Phase 16 Agent 3: the persistent population and its opponent mixture.

Specification source: `03_AGENT_3_TRAINING_LOOP_V2.md` sections 2.1 and 2.6.

A population, not an iteration
------------------------------
Phase 14 scheduled 2,048 logical games per iteration and an iteration ended
when the last of them ended, which is exactly why its iterations grew from 24
to 138 minutes as games got longer. Phase 16 keeps a fixed number of games
*in flight* and ends an iteration on a decision budget instead. The identity
that survives that change is not "the n-th game of iteration k" but "the d-th
game played in slot s", so that is what :func:`draw_for_slot` keys on and what
`phase16_game_v1` ids record.

The mixture, as a per-draw coin
-------------------------------
Phase 14 spelled its mixture as contiguous ordinal subranges because it knew
the whole iteration in advance. A window collector draws a replacement game
the instant a slot frees up, so the composition is a per-draw function of the
slot and draw number through the `opponent_draw` domain: over a run the
realized shares converge on the declared ones, and every individual draw is
still reproducible from the run state alone.

```text
pure_current      100% current self-play          (the paper)
phase14_mixture   58% current / 30% historical / 12% handcrafted
```

Handcrafted bots are *evaluation* opponents regardless of this flag; they
appear in `phase14_mixture` only because reproducing Phase 14's recipe is what
the control arm is for. Overview section 7 forbids training *reward* shaped by
them, which is a different thing and is not done anywhere in this phase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ...engine.constants import PLAYER_NAMES, PLAYERS
from ...evaluation.registry import POLICY_INDEX
from ..phase14_contract import HANDCRAFTED_COUNTS, HANDCRAFTED_FAMILY_ORDER
from .contract import (
    DOMAIN_OPPONENT_DRAW,
    DOMAIN_POLICY,
    OPPONENTS_PHASE14_MIXTURE,
    OPPONENTS_PURE_CURRENT,
    PHASE14_MIXTURE_SHARES,
    ArmConfig,
    Phase16TrainingError,
    derive_train_seed,
    game_id as phase16_game_id,
    uniform_from_seed,
)

PHASE16_POPULATION_VERSION = "phase16_population_v1"

KIND_CURRENT = "current_policy"
KIND_HISTORICAL = "historical_snapshot"
KIND_HANDCRAFTED = "handcrafted_policy"

LEARNER_CONTROL_BOTH = "both"

#: The handcrafted roster and its relative weights, inherited from Phase 14's
#: frozen counts so the control arm reproduces that recipe rather than a new one.
HANDCRAFTED_ROSTER = tuple(HANDCRAFTED_FAMILY_ORDER)
_HANDCRAFTED_TOTAL = sum(HANDCRAFTED_COUNTS[name] for name in HANDCRAFTED_ROSTER)
HANDCRAFTED_WEIGHTS = tuple(
    HANDCRAFTED_COUNTS[name] / _HANDCRAFTED_TOTAL for name in HANDCRAFTED_ROSTER
)


class Phase16PopulationError(Phase16TrainingError):
    """A Phase 16 population draw is outside its contract."""


@dataclass(frozen=True)
class SlotDraw:
    """Everything the collector needs to start one game in one slot.

    Carries no privileged information: no piece truth, no setup contents, no
    outcome. The setups are resolved by the setup source from the game id.
    """

    game_id: str
    arm_id: str
    slot: int
    draw: int
    opponent_kind: str
    opponent_identity: str
    learner_control: str
    learner_color: "str | None"
    handcrafted_policy_id: "str | None"
    red_policy_seed: "int | None"
    blue_policy_seed: "int | None"

    @property
    def learner_sides(self) -> tuple:
        if self.learner_control == LEARNER_CONTROL_BOTH:
            return ("red", "blue")
        return (self.learner_control,)

    def to_dict(self) -> dict:
        return asdict(self)


def handcrafted_policy_token(policy_id: str) -> str:
    """The frozen Phase 4 `id@version` token of a rule or stress policy."""
    if policy_id not in POLICY_INDEX:
        raise Phase16PopulationError(f"unknown frozen policy id: {policy_id!r}")
    return f"{policy_id}@{POLICY_INDEX[policy_id].policy_version}"


def _pick_weighted(uniform: float, names, weights) -> str:
    cursor = 0.0
    for name, weight in zip(names, weights):
        cursor += float(weight)
        if uniform < cursor:
            return name
    return names[-1]


class HistoricalPool:
    """The arm's own snapshots, oldest first, with the start as the anchor.

    A 6-hour arm has no external archive to draw on: the historical opponents
    of a Phase 16 arm are the weights that arm passed through. Until the first
    snapshot is taken, "historical" resolves to the starting P24 copy, which is
    the honest answer -- the run has no other past.
    """

    def __init__(self, anchor_identity: str = "P24", anchor_path: "str | None" = None):
        self._members = [str(anchor_identity)]
        self._paths = {str(anchor_identity): anchor_path}

    def add(self, identity: str, path: "str | None" = None) -> None:
        identity = str(identity)
        if identity in self._paths:
            raise Phase16PopulationError(f"{identity} is already in the pool")
        self._members.append(identity)
        self._paths[identity] = path

    def members(self) -> tuple:
        return tuple(self._members)

    def path_for(self, identity: str) -> "str | None":
        if identity not in self._paths:
            raise Phase16PopulationError(f"{identity!r} is not a pool member")
        return self._paths[identity]

    def select(self, uniform: float) -> str:
        """One member, uniform over the pool as it stands at the draw."""
        index = int(float(uniform) * len(self._members))
        return self._members[min(index, len(self._members) - 1)]

    def to_dict(self) -> dict:
        return {"size": len(self._members), "members": list(self._members)}


def draw_for_slot(
    config: ArmConfig,
    *,
    slot: int,
    draw: int,
    pool: "HistoricalPool | None" = None,
) -> SlotDraw:
    """The full record of one logical game, from the run state alone.

    A pure function of `(arm, slot, draw, pool membership)`: the same inputs
    rebuild the same record on any machine, with no enumeration state and no
    partial game in hand.
    """
    identifier = phase16_game_id(config.arm_id, slot, draw)
    if config.opponents == OPPONENTS_PURE_CURRENT:
        return SlotDraw(
            game_id=identifier,
            arm_id=config.arm_id,
            slot=int(slot),
            draw=int(draw),
            opponent_kind=KIND_CURRENT,
            opponent_identity="current",
            learner_control=LEARNER_CONTROL_BOTH,
            learner_color=None,
            handcrafted_policy_id=None,
            red_policy_seed=None,
            blue_policy_seed=None,
        )
    if config.opponents != OPPONENTS_PHASE14_MIXTURE:
        raise Phase16PopulationError(f"unknown opponent mixture: {config.opponents!r}")

    bucket_uniform = uniform_from_seed(
        derive_train_seed(DOMAIN_OPPONENT_DRAW, config.arm_id, "bucket", slot, draw)
    )
    shares = PHASE14_MIXTURE_SHARES
    if bucket_uniform < shares["current"]:
        return SlotDraw(
            game_id=identifier,
            arm_id=config.arm_id,
            slot=int(slot),
            draw=int(draw),
            opponent_kind=KIND_CURRENT,
            opponent_identity="current",
            learner_control=LEARNER_CONTROL_BOTH,
            learner_color=None,
            handcrafted_policy_id=None,
            red_policy_seed=None,
            blue_policy_seed=None,
        )

    # The learner takes one colour against any non-current opponent, and the
    # colour alternates on the draw ordinal so a slot cannot become a colour.
    colour = "red" if (int(slot) + int(draw)) % 2 == 0 else "blue"
    opponent_colour = "blue" if colour == "red" else "red"

    if bucket_uniform < shares["current"] + shares["historical"]:
        member_uniform = uniform_from_seed(
            derive_train_seed(DOMAIN_OPPONENT_DRAW, config.arm_id, "member", slot, draw)
        )
        pool = pool or HistoricalPool()
        identity = pool.select(member_uniform)
        return SlotDraw(
            game_id=identifier,
            arm_id=config.arm_id,
            slot=int(slot),
            draw=int(draw),
            opponent_kind=KIND_HISTORICAL,
            opponent_identity=identity,
            learner_control=colour,
            learner_color=colour,
            handcrafted_policy_id=None,
            red_policy_seed=None,
            blue_policy_seed=None,
        )

    policy_uniform = uniform_from_seed(
        derive_train_seed(DOMAIN_OPPONENT_DRAW, config.arm_id, "policy", slot, draw)
    )
    policy_id = _pick_weighted(policy_uniform, HANDCRAFTED_ROSTER, HANDCRAFTED_WEIGHTS)
    seed = derive_train_seed(DOMAIN_POLICY, config.arm_id, opponent_colour, slot, draw)
    return SlotDraw(
        game_id=identifier,
        arm_id=config.arm_id,
        slot=int(slot),
        draw=int(draw),
        opponent_kind=KIND_HANDCRAFTED,
        opponent_identity=handcrafted_policy_token(policy_id),
        learner_control=colour,
        learner_color=colour,
        handcrafted_policy_id=policy_id,
        red_policy_seed=seed if opponent_colour == "red" else None,
        blue_policy_seed=seed if opponent_colour == "blue" else None,
    )


def realized_shares(draws) -> dict:
    """The composition actually drawn, for the telemetry row.

    Counts, never estimates: a telemetry row and the games actually played
    cannot disagree because this is computed from the draws themselves.
    """
    counts: dict = {}
    policies: dict = {}
    colours: dict = {}
    total = 0
    for record in draws:
        total += 1
        counts[record.opponent_kind] = counts.get(record.opponent_kind, 0) + 1
        if record.handcrafted_policy_id:
            policies[record.handcrafted_policy_id] = (
                policies.get(record.handcrafted_policy_id, 0) + 1
            )
        colours[record.learner_control] = colours.get(record.learner_control, 0) + 1
    return {
        "games": total,
        "kind_counts": dict(sorted(counts.items())),
        "kind_shares": {
            kind: value / total for kind, value in sorted(counts.items())
        }
        if total
        else {},
        "handcrafted_counts": dict(sorted(policies.items())),
        "learner_control_counts": dict(sorted(colours.items())),
    }


def player_index(color: str) -> int:
    for player, name in PLAYER_NAMES.items():
        if name == color:
            return player
    raise Phase16PopulationError(f"unknown colour {color!r}")


def population_semantics(config: ArmConfig) -> dict:
    return {
        "population_version": PHASE16_POPULATION_VERSION,
        "population": config.population,
        "window_decisions": config.window_decisions,
        "unit": "one window = a fixed budget of learner decisions",
        "identity": "the d-th game played in slot s",
        "mixture": config.opponents,
        "declared_shares": (
            {"current": 1.0}
            if config.opponents == OPPONENTS_PURE_CURRENT
            else dict(PHASE14_MIXTURE_SHARES)
        ),
        "handcrafted_roster": list(HANDCRAFTED_ROSTER),
        "handcrafted_weights": [round(w, 6) for w in HANDCRAFTED_WEIGHTS],
        "draw_rule": "per draw, from the phase16 opponent_draw domain",
        "replacement": "a game that ends mid-window is replaced by a fresh draw",
        "colour_rule": "the learner alternates colour on (slot + draw) parity",
        "search": "absent; no module under stratego.search is imported",
    }


__all__ = [
    "HANDCRAFTED_ROSTER",
    "HANDCRAFTED_WEIGHTS",
    "HistoricalPool",
    "KIND_CURRENT",
    "KIND_HANDCRAFTED",
    "KIND_HISTORICAL",
    "LEARNER_CONTROL_BOTH",
    "PHASE16_POPULATION_VERSION",
    "Phase16PopulationError",
    "SlotDraw",
    "draw_for_slot",
    "handcrafted_policy_token",
    "player_index",
    "population_semantics",
    "realized_shares",
]
