#!/usr/bin/env python3
"""Generate the five human-readable inspection examples for the Phase Two report.

Each example prints a simplified board, the acting player, the action, the
expected result, the actual result and the relevant observation or event
changes, as required by section 24.16 of the Phase Two instructions.

    python scripts/manual_inspection_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.constants import (  # noqa: E402
    BLUE,
    RED,
    RulesConfig,
    SCOUT,
    SERGEANT,
)
from stratego.engine.coordinates import square_from_name  # noqa: E402
from stratego.engine.observation import (  # noqa: E402
    CH_HIDDEN_OPPONENT_OCCUPANCY,
    CH_KNOWN_OPPONENT_IDENTITY,
    CH_OWN_BEHAVIOR,
    CH_OWN_IDENTITY,
    CH_OWN_KNOWN_TO_OPPONENT,
    CH_OWN_MOVED,
    CH_RECENT_MOVES,
    CH_UNRESOLVED_INVENTORY,
    build_observation,
)
from stratego.engine.state import render_board  # noqa: E402
from tests.helpers import make_position, piece_at, play  # noqa: E402

LOOSE = RulesConfig(battleless_move_limit=10_000, absolute_move_limit=10_000)


def cell(observation, channel, name, observer=RED):
    from stratego.engine.coordinates import to_perspective

    row, column = divmod(to_perspective(square_from_name(name), observer), 10)
    return float(observation[channel, row, column])


def heading(number: int, title: str) -> None:
    print(f"\n### Example {number} — {title}\n")


def show_board(state, observer=None, caption="Board") -> None:
    label = "privileged view" if observer is None else (
        "red's view" if observer == RED else "blue's view"
    )
    print(f"{caption} ({label}):\n")
    print("```text")
    print(render_board(state, observer))
    print("```\n")


def example_one() -> None:
    heading(1, "Ordinary movement")
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e7": "sergeant", "j10": "flag"},
        acting_player=RED,
        rules=LOOSE,
    )
    show_board(state)
    before = build_observation(state, RED)
    print("Acting player: red")
    print("Action: `e3->e4` (one square, no combat)")
    print("Expected: captain occupies e4, moved flag set, battleless counter 1, no events beyond `move`.\n")

    events = play(state, "e3 e4")[0]
    after = build_observation(state, RED)

    print("Actual:")
    print(f"- captain now on e4: {piece_at(state, 'e4').type_name}")
    print(f"- total moves {state.total_moves}, battleless moves {state.battleless_moves}")
    print(f"- events: {[event['event_type'] for event in events]}")
    print("\nObservation changes (red's frame):")
    captain = 5  # captain type index
    print(
        f"- channel {CH_OWN_IDENTITY + captain} (own captain): e3 "
        f"{before[CH_OWN_IDENTITY + captain][2][4]:.0f} -> {after[CH_OWN_IDENTITY + captain][2][4]:.0f}, "
        f"e4 {before[CH_OWN_IDENTITY + captain][3][4]:.0f} -> {after[CH_OWN_IDENTITY + captain][3][4]:.0f}"
    )
    print(
        f"- channel {CH_OWN_MOVED} (own moved): e4 "
        f"{cell(before, CH_OWN_MOVED, 'e4'):.0f} -> {cell(after, CH_OWN_MOVED, 'e4'):.0f}"
    )
    print(
        f"- channel {CH_RECENT_MOVES} (last move): e3 {cell(after, CH_RECENT_MOVES, 'e3'):+.0f}, "
        f"e4 {cell(after, CH_RECENT_MOVES, 'e4'):+.0f}"
    )


def example_two() -> None:
    heading(2, "Scout revelation by a multi-square move")
    state = make_position(
        red={"a1": "scout", "j1": "flag"},
        blue={"a10": "sergeant", "j10": "flag"},
        acting_player=RED,
        rules=LOOSE,
    )
    show_board(state, BLUE, caption="Board before")
    scout = piece_at(state, "a1")
    print("Acting player: red")
    print("Action: `a1->a4` (four squares along the a-file)")
    print(
        "Expected: only a Scout can move more than one square, so blue legally "
        "learns the type; an `identity_reveal` event with reason "
        "`scout_multisquare` is emitted.\n"
    )

    events = play(state, "a1 a4")[0]
    blue_view = build_observation(state, BLUE)
    red_view = build_observation(state, RED)

    print("Actual:")
    print(f"- known to blue: {scout.known_to(BLUE)} (reason {scout.reveal_reason_blue!r})")
    print(f"- events: {[event['event_type'] for event in events]}")
    reveal = events[1]
    print(f"- reveal event: piece {reveal['piece_id']}, type {reveal['piece_type']}, newly known to {reveal['newly_known_to']}")
    show_board(state, BLUE, caption="Board after")
    print("Observation changes:")
    print(
        f"- blue channel {CH_KNOWN_OPPONENT_IDENTITY + SCOUT} (known opponent scout) at a4: "
        f"{cell(blue_view, CH_KNOWN_OPPONENT_IDENTITY + SCOUT, 'a4', BLUE):.0f}"
    )
    print(
        f"- blue channel {CH_HIDDEN_OPPONENT_OCCUPANCY} (hidden opponent) at a4: "
        f"{cell(blue_view, CH_HIDDEN_OPPONENT_OCCUPANCY, 'a4', BLUE):.0f}"
    )
    print(
        f"- red channel {CH_OWN_KNOWN_TO_OPPONENT} (own piece known to opponent) at a4: "
        f"{cell(red_view, CH_OWN_KNOWN_TO_OPPONENT, 'a4', RED):.0f}"
    )
    print(
        f"- blue unresolved scout inventory (channel {CH_UNRESOLVED_INVENTORY + SCOUT}): "
        f"{blue_view[CH_UNRESOLVED_INVENTORY + SCOUT][0][0]:.3f} -- in this two-piece "
        "fixture red's other 38 pieces are already captured and therefore revealed, "
        "so identifying the last scout resolves the whole type"
    )


def example_three() -> None:
    heading(3, "Combat")
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e4": "marshal", "j10": "flag"},
        acting_player=RED,
        rules=LOOSE,
        battleless_moves=42,
    )
    show_board(state, RED, caption="Board before")
    attacker = piece_at(state, "e3")
    defender = piece_at(state, "e4")
    print("Acting player: red")
    print("Action: `e3->e4` (captain attacks a hidden piece that turns out to be the marshal)")
    print(
        "Expected: the defender wins and keeps e4, the attacker is removed, both "
        "identities become public, and the battleless counter resets to 0.\n"
    )

    events = play(state, "e3 e4")[0]
    red_view = build_observation(state, RED)

    print("Actual:")
    print(f"- attacker alive: {attacker.alive}; defender alive: {defender.alive}")
    combat = next(event for event in events if event["event_type"] == "combat")
    print(
        f"- combat event: {combat['attacker_type']} vs {combat['defender_type']} -> "
        f"{combat['outcome']}"
    )
    print(f"- battleless moves: 42 -> {state.battleless_moves}")
    print(f"- events: {[event['event_type'] for event in events]}")
    show_board(state, RED, caption="Board after")
    marshal = 9
    print("Observation changes (red's frame):")
    print(
        f"- channel {CH_KNOWN_OPPONENT_IDENTITY + marshal} (known opponent marshal) at e4: "
        f"{cell(red_view, CH_KNOWN_OPPONENT_IDENTITY + marshal, 'e4'):.0f}"
    )
    print(
        f"- channel {CH_HIDDEN_OPPONENT_OCCUPANCY} at e4: "
        f"{cell(red_view, CH_HIDDEN_OPPONENT_OCCUPANCY, 'e4'):.0f}"
    )
    print(
        f"- unresolved marshal inventory (channel {CH_UNRESOLVED_INVENTORY + marshal}): "
        f"{red_view[CH_UNRESOLVED_INVENTORY + marshal][0][0]:.3f}"
    )


def example_four() -> None:
    heading(4, "Hidden-piece observation")
    state = make_position(
        red={"e3": "captain", "a1": "flag"},
        blue={"e6": "marshal", "f6": "spy", "j10": "flag"},
        acting_player=RED,
        rules=LOOSE,
    )
    show_board(state, None, caption="Board")
    show_board(state, RED, caption="Board")
    observation = build_observation(state, RED)
    print("Acting player: red")
    print("Action: none; this example inspects the observation of a static position.")
    print(
        "Expected: red sees generic occupancy on e6 and f6 and no type plane, even "
        "though the privileged state knows a marshal and a spy stand there.\n"
    )
    print("Actual:")
    for name in ("e6", "f6"):
        known = sum(
            cell(observation, CH_KNOWN_OPPONENT_IDENTITY + piece_type, name)
            for piece_type in range(12)
        )
        print(
            f"- {name}: hidden-occupancy channel {CH_HIDDEN_OPPONENT_OCCUPANCY} = "
            f"{cell(observation, CH_HIDDEN_OPPONENT_OCCUPANCY, name):.0f}, "
            f"sum of known-identity channels 12-23 = {known:.0f}"
        )
    from stratego.engine.observation import belief_target

    print(
        f"- privileged belief target (training only, never an observation input): "
        f"{belief_target(state, RED)}"
    )


def example_five() -> None:
    heading(5, "Behavioural event tracking")
    state = make_position(
        red={"e3": "captain", "c3": "miner", "a1": "flag"},
        blue={"e5": "sergeant", "j9": "scout", "j10": "flag"},
        acting_player=BLUE,
        rules=LOOSE,
        revealed={"e5"},
    )
    show_board(state, RED, caption="Board before")
    print("Acting player: blue, then red")
    print("Actions: `e5->e4` (blue threatens the red captain), then `c3->d3` (red's miner protects it)")
    print(
        "Expected: blue's sergeant records a `threat`; red's miner records a "
        "`protect` and the captain records a `was_protected`. The protect "
        "counterpart is the captain, and the threatener is kept as context.\n"
    )

    play(state, "e5 e4")
    threat_events = [
        event for event in state.events if event.get("behavior_type") == "threat"
    ]
    play(state, "c3 d3")

    miner = piece_at(state, "d3")
    captain = piece_at(state, "e3")
    protect = state.behavior_event(miner.piece_id, "protect")
    was_protected = state.behavior_event(captain.piece_id, "was_protected")

    show_board(state, RED, caption="Board after")
    print("Actual:")
    from stratego.engine.pieces import piece_id_name

    print(f"- threat events after blue's move: {threat_events}")
    print(
        f"- protect: actor {piece_id_name(protect.actor_piece_id)}, counterpart "
        f"{piece_id_name(protect.counterpart_piece_id)}, context (threatener) "
        f"{piece_id_name(protect.context_piece_id)}, ply {protect.event_ply}, "
        f"actor knew counterpart {protect.actor_knew_counterpart_type}"
    )
    print(
        f"- was_protected: actor {piece_id_name(was_protected.actor_piece_id)}, counterpart "
        f"{piece_id_name(was_protected.counterpart_piece_id)}, ply {was_protected.event_ply}"
    )

    observation = build_observation(state, RED)
    protect_base = CH_OWN_BEHAVIOR + 4 * 3
    was_protected_base = CH_OWN_BEHAVIOR + 4 * 4
    print("\nObservation changes (red's frame):")
    print(
        f"- protect block channels {protect_base}-{protect_base + 3} at d3: "
        f"recency {cell(observation, protect_base, 'd3'):.3f}, "
        f"rank {cell(observation, protect_base + 1, 'd3'):.3f}, "
        f"actor-knew {cell(observation, protect_base + 2, 'd3'):.0f}, "
        f"special {cell(observation, protect_base + 3, 'd3'):+.0f}"
    )
    print(
        f"- was-protected block channels {was_protected_base}-{was_protected_base + 3} at e3: "
        f"recency {cell(observation, was_protected_base, 'e3'):.3f}, "
        f"rank {cell(observation, was_protected_base + 1, 'e3'):.3f}, "
        f"actor-knew {cell(observation, was_protected_base + 2, 'e3'):.0f}"
    )
    opponent_threat_base = 88
    print(
        f"- opponent threat block channel {opponent_threat_base} at e4: "
        f"recency {cell(observation, opponent_threat_base, 'e4'):.3f} (one ply old), "
        f"rank {cell(observation, opponent_threat_base + 1, 'e4'):.3f}, "
        f"actor-knew {cell(observation, opponent_threat_base + 2, 'e4'):.0f}"
    )
    print(
        "  The counterpart of blue's threat is red's captain. Blue did not know "
        "that identity when the threat happened, so the actor-knew flag is 0 and "
        "the rank stays 0 even though red obviously knows its own captain: "
        "exposing rank needs both halves of the section 9.3 rule."
    )


def main() -> None:
    print("# Manual inspection examples\n")
    print(
        "Generated by `scripts/manual_inspection_examples.py`. Board codes: "
        "`r`/`b` prefix for owner, `1`-`9` for Marshal down to Scout, `S` Spy, "
        "`F` Flag, `B` Bomb, `?` unresolved, `~~` lake, `.` empty."
    )
    example_one()
    example_two()
    example_three()
    example_four()
    example_five()


if __name__ == "__main__":
    main()
