"""Phase 15 Agent 2 section 11: the Stage A decision-diagnostic positions.

Specification source: `02_AGENT_2_SEARCH_IMPLEMENTATION.md` sections 8, 11.

What a diagnostic position has to be
------------------------------------
Section 11 wants "one fresh, correctly oriented manifest of tactically
meaningful positions". Three things follow.

*Fresh*: the boards are drawn by :mod:`stratego.search.phase15.boards` at
ordinals that no Stage B match board uses, so the decision diagnostic and the
match comparison never share a position.

*Correctly oriented*: they come from the same orientation-gated construction
as every other Phase 15 board — a canonical Blue tuple cannot reach
`create_game` on this path either.

*Tactically meaningful*: a position is eligible only once the opening is over
(`ply >= 12`) and enough of the opponent's army is still unresolved
(`>= 6 hidden pieces`) that a belief model has something to say. A position
where nothing is hidden is one where every provider must answer identically,
and measuring on it would dilute the very difference the stage is for.

Replay, not storage
-------------------
A position is stored as its board id plus the action prefix that reaches it.
:func:`materialize_positions` replays the prefix through the accepted
`apply_action` and requires the rebuilt observation to hash to the digest the
generator recorded — so a stored manifest is an executable object rather than
a description of one, and a silent change anywhere in the engine, the setup
library or the orientation path turns into a refusal.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from ...engine.legal_moves import legal_actions
from ...engine.observation import build_observation
from ...engine.state import create_game
from ...engine.transition import apply_action
from ...evaluation.match_spec import EVALUATION_RULES
from ...evaluation.policy import build_public_view
from .boards import Phase15BoardPlan, Phase15MatchSetupSources, board_plan
from .contract import (
    MATCH_LIBRARY_SPLIT,
    MATCH_OPPONENTS,
    MATCH_SETUP_SOURCES,
    MOVE_MODELS,
    POSITION_VERSION,
    Phase15SearchError,
    parse_board_id,
)
from .matchplay import DirectSeat, build_spec, opponent_seat, single_game_bank
from .contract import pairing as pairing_of

#: The first board ordinal the diagnostic pack uses. Stage B counts up from
#: 0, so the two packs cannot collide however large either grows.
POSITION_ORDINAL_BASE = 100

#: Eligibility floors. See the module docstring.
MIN_PLY = 12
MIN_UNRESOLVED = 6

#: At most this many evenly spaced eligible decisions from one game.
POSITIONS_PER_GAME = 4


class Phase15PositionError(Phase15SearchError):
    """A diagnostic position could not be generated or replayed."""


def observation_digest(observation: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(observation, dtype=np.float32).tobytes()
    ).hexdigest()


def position_id(board_identifier: str, ply: int) -> str:
    return f"{board_identifier}|ply={int(ply):04d}"


@dataclass(frozen=True)
class DiagnosticPosition:
    """One replayable decision position."""

    position_id: str
    board_id: str
    observer_model: str
    opponent: str
    setup_source: str
    requested_family: str
    player_color: str
    ply: int
    unresolved: int
    legal_actions: int
    observation_sha256: str
    action_prefix: "tuple[int, ...]" = field(repr=False, default=())

    def describe(self) -> dict:
        return {
            "position_id": self.position_id,
            "board_id": self.board_id,
            "observer_model": self.observer_model,
            "opponent": self.opponent,
            "setup_source": self.setup_source,
            "requested_family": self.requested_family,
            "player_color": self.player_color,
            "ply": self.ply,
            "unresolved": self.unresolved,
            "legal_actions": self.legal_actions,
            "observation_sha256": self.observation_sha256,
            "action_prefix": list(self.action_prefix),
        }


def evenly_spaced(values: list, count: int) -> list:
    """At most `count` evenly spaced elements, both endpoints included."""
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    step = (len(values) - 1) / (count - 1) if count > 1 else 0.0
    return [values[int(round(index * step))] for index in range(count)]


def play_for_positions(
    plan: Phase15BoardPlan,
    observer_model: str,
    owners: dict,
    *,
    per_game: int = POSITIONS_PER_GAME,
) -> "list[DiagnosticPosition]":
    """Play one board with a direct observer and harvest its positions.

    The observer is the *direct* move model, never a search arm: a diagnostic
    position must not have been reached by the thing being diagnosed.
    """
    seat = DirectSeat(pairing_of(f"{observer_model}_direct"), owners)
    opponent_reference, opponent_policy = opponent_seat(plan, owners)
    spec = build_spec(plan, opponent_reference)
    bank = single_game_bank(spec, plan)
    red_setup, blue_setup = spec.resolve_setups(bank)
    state = create_game(red_setup, blue_setup, rules=spec.rules, game_id=spec.game_id)

    from ...evaluation.policy import build_policy_input

    player = plan.player
    history: list[int] = []
    eligible: list[dict] = []
    while not state.terminal:
        actor = state.acting_player
        legal = legal_actions(state)
        if actor == player:
            action, _record = seat.decide(state, legal, spec, plan)
            view = build_public_view(state, player)
            unresolved = len(view.unresolved_opponent_piece_ids)
            if state.total_moves >= MIN_PLY and unresolved >= MIN_UNRESOLVED:
                eligible.append(
                    {
                        "ply": int(state.total_moves),
                        "unresolved": int(unresolved),
                        "legal_actions": len(legal),
                        "prefix": tuple(history),
                        "observation_sha256": observation_digest(
                            build_observation(state, player)
                        ),
                    }
                )
        else:
            request = build_policy_input(
                state,
                policy=opponent_reference,
                policy_seed=spec.policy_seed_for(actor),
                requirements=opponent_policy.requirements,
                suite_version=spec.suite_version,
                match_id=spec.match_id,
                paired_unit_id=spec.paired_unit_id,
                legal=legal,
            )
            action = int(opponent_policy.decide_checked(request).selected_action_id)
        history.append(int(action))
        apply_action(state, action, legal=legal)

    return [
        DiagnosticPosition(
            position_id=position_id(plan.board_id, row["ply"]),
            board_id=plan.board_id,
            observer_model=observer_model,
            opponent=plan.opponent,
            setup_source=plan.setup_source,
            requested_family=plan.requested_family,
            player_color=plan.color,
            ply=row["ply"],
            unresolved=row["unresolved"],
            legal_actions=row["legal_actions"],
            observation_sha256=row["observation_sha256"],
            action_prefix=row["prefix"],
        )
        for row in evenly_spaced(eligible, int(per_game))
    ]


def position_cells(games_per_observer: int = 15) -> "list[tuple[str, str, str, str, int]]":
    """`(observer, opponent, setup source, colour, ordinal)` cells.

    Every observer meets every opponent; the setup source and the colour
    rotate with the opponent index so both stay balanced whatever the game
    count is.
    """
    cells = []
    for observer_index, observer in enumerate(MOVE_MODELS):
        for game in range(int(games_per_observer)):
            opponent = MATCH_OPPONENTS[game % len(MATCH_OPPONENTS)]
            source = MATCH_SETUP_SOURCES[
                (game + observer_index) % len(MATCH_SETUP_SOURCES)
            ]
            color = ("red", "blue")[(game + observer_index) % 2]
            cells.append(
                (observer, opponent, source, color, POSITION_ORDINAL_BASE + game)
            )
    return cells


def generate_positions(
    owners: dict,
    *,
    games_per_observer: int = 15,
    per_game: int = POSITIONS_PER_GAME,
    sources: "Phase15MatchSetupSources | None" = None,
    library_split: str = MATCH_LIBRARY_SPLIT,
    progress=None,
) -> "list[DiagnosticPosition]":
    """The whole diagnostic pack."""
    sources = Phase15MatchSetupSources() if sources is None else sources
    positions: list[DiagnosticPosition] = []
    cells = position_cells(games_per_observer)
    for index, (observer, opponent, source, color, ordinal) in enumerate(cells):
        plan = board_plan(
            opponent, source, color, ordinal, sources, library_split=library_split
        )
        found = play_for_positions(plan, observer, owners, per_game=per_game)
        positions.extend(found)
        if progress is not None:
            progress(index + 1, len(cells), len(positions))
    return positions


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest(
    positions: "list[DiagnosticPosition]", *, generated_utc: str, **extra
) -> dict:
    from ...belief.phase15.orientation import ORIENTATION_RULE, ORIENTATION_RULE_VERSION

    rows = [position.describe() for position in positions]
    by_observer: dict[str, int] = {}
    by_opponent: dict[str, int] = {}
    for position in positions:
        by_observer[position.observer_model] = by_observer.get(position.observer_model, 0) + 1
        by_opponent[position.opponent] = by_opponent.get(position.opponent, 0) + 1
    payload = {
        "artifact": POSITION_VERSION,
        "generated_utc": generated_utc,
        "orientation_rule_version": ORIENTATION_RULE_VERSION,
        "orientation_rule": ORIENTATION_RULE,
        "eligibility": {"min_ply": MIN_PLY, "min_unresolved": MIN_UNRESOLVED},
        "positions_per_game": POSITIONS_PER_GAME,
        "position_ordinal_base": POSITION_ORDINAL_BASE,
        "supersedes": (
            "the Phase 12 diagnostic positions, whose boards were built by the "
            "mis-orienting glue; no Phase 12 position is reused"
        ),
        "positions": rows,
        "position_count": len(rows),
        "balance": {
            "by_observer": dict(sorted(by_observer.items())),
            "by_opponent": dict(sorted(by_opponent.items())),
        },
        **extra,
    }
    import json

    payload["manifest_digest"] = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def materialize_positions(
    manifest: dict,
    *,
    sources: "Phase15MatchSetupSources | None" = None,
    verify: bool = True,
) -> "list[tuple[dict, object, Phase15BoardPlan]]":
    """Replay every stored position. Returns `(row, state, plan)` triples."""
    sources = Phase15MatchSetupSources() if sources is None else sources
    split = manifest.get("library_split", MATCH_LIBRARY_SPLIT)
    rebuilt = []
    cache: dict = {}
    for row in manifest["positions"]:
        fields = parse_board_id(row["board_id"])
        plan = cache.get(row["board_id"])
        if plan is None:
            plan = board_plan(
                fields["opponent"],
                fields["setup_source"],
                fields["color"],
                fields["ordinal"],
                sources,
                library_split=split,
            )
            cache[row["board_id"]] = plan
        state = create_game(
            plan.red_setup,
            plan.blue_setup,
            rules=EVALUATION_RULES,
            game_id=row["position_id"],
        )
        for action in row["action_prefix"]:
            legal = legal_actions(state)
            apply_action(state, int(action), legal=legal)
        if verify:
            if int(state.total_moves) != int(row["ply"]):
                raise Phase15PositionError(
                    f"{row['position_id']}: replay reached ply {state.total_moves}, "
                    f"the manifest records {row['ply']}"
                )
            if state.acting_player != plan.player:
                raise Phase15PositionError(
                    f"{row['position_id']}: replay hands the move to the wrong player"
                )
            digest = observation_digest(build_observation(state, plan.player))
            if digest != row["observation_sha256"]:
                raise Phase15PositionError(
                    f"{row['position_id']}: replayed observation hashes to {digest}, "
                    f"the manifest records {row['observation_sha256']}"
                )
        rebuilt.append((row, state, plan))
    return rebuilt


__all__ = [
    "DiagnosticPosition",
    "MIN_PLY",
    "MIN_UNRESOLVED",
    "POSITIONS_PER_GAME",
    "POSITION_ORDINAL_BASE",
    "Phase15PositionError",
    "build_manifest",
    "evenly_spaced",
    "generate_positions",
    "materialize_positions",
    "observation_digest",
    "play_for_positions",
    "position_cells",
    "position_id",
]
