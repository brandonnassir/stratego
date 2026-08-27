#!/usr/bin/env python
"""Play against — or watch — the Phase 15 working search player.

The human/machine face of `stratego.search.phase15.player`: each seat is
`human` or one of the working player's modes (`p18_direct`, `p24_direct`,
`selected_search`, `maximum_strength`, plus the production pairing ids as
diagnostic names). Both machine seats share one player instance, so the
frozen stack — the selected Phase 14 move model, the selected Phase 15
belief specialist, the selected preset, the measured time cap and the direct
fallback — is loaded and digest-checked exactly once, from
`phase15_search_candidate_v1`.

Usage:

    python scripts/play_phase15.py                                  # human (red) vs the selection
    python scripts/play_phase15.py --red human --blue maximum_strength
    python scripts/play_phase15.py --red p18_direct --blue selected_search

Moves are typed as two squares, e.g. `b4 b5`. Commands: `moves`, `status`,
`board`, `resign`, `help`.

Orientation
-----------
Both setups are drawn through `stratego.belief.phase15.setups`, whose single
exit is the accepted `oriented_for` helper, and every board is then put
through Agent 1's whole section 4 gate before `create_game` sees it. This is
deliberately *not* the path `scripts/play_phase12.py` uses: that script draws
through `Phase11BSetupSources`, which returns canonical own-orientation
tuples, so its Blue army reaches the engine reversed. Phase 15 may not edit
Phase 12, so the difference is stated here rather than fixed there.

Information boundary
--------------------
The board is rendered from legal knowledge only. A human seat sees its own
army and only the opponent pieces legally revealed to it; a spectator sees a
rank only once the opposing player legally knows it. There is no flag, mode
or seat through which a hidden rank or the diagnostic oracle could be shown —
the player module refuses the oracle by name and by absence from its table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.actions import decode_action, encode_action  # noqa: E402
from stratego.engine.constants import (  # noqa: E402
    BLUE,
    BOARD_COLUMNS,
    BOARD_ROWS,
    LAKE_SQUARE_SET,
    PIECE_TYPE_CODES,
    RED,
)
from stratego.engine.coordinates import square_from_name, square_name  # noqa: E402
from stratego.engine.legal_moves import legal_actions  # noqa: E402
from stratego.engine.state import create_game  # noqa: E402
from stratego.engine.transition import apply_action  # noqa: E402
from stratego.evaluation.match_spec import EVALUATION_RULES  # noqa: E402
from stratego.search.phase15 import player as pl  # noqa: E402
from stratego.search.phase15.contract import (  # noqa: E402
    DOMAIN_PLAYER_SETUP,
    MATCH_LIBRARY_SPLIT,
    MATCH_SETUP_SOURCES,
    derive_search_seed,
)

#: What a seat may be. There is deliberately no oracle entry.
SEAT_CHOICES = ("human",) + pl.PLAYER_MODES

_COLOR_NAME = {RED: "red", BLUE: "blue"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play against, or watch, the Phase 15 working search player."
    )
    parser.add_argument(
        "--red", choices=SEAT_CHOICES, default="human", help="the red seat"
    )
    parser.add_argument(
        "--blue",
        choices=SEAT_CHOICES,
        default=pl.MODE_SELECTED,
        help=(
            f"the blue seat (default: {pl.MODE_SELECTED}; "
            f"'{pl.MODE_MAX_STRENGTH}' is the maximum-strength mode)"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--setup-seed", type=int, default=0,
        help="which accepted-library setups to draw (deterministic)",
    )
    parser.add_argument(
        "--setup-source", choices=MATCH_SETUP_SOURCES, default="phase14_learned",
        help="accepted setup source for both sides",
    )
    parser.add_argument(
        "--candidate", default=None, help="path to phase15_search_candidate_v1.json"
    )
    return parser


# ---------------------------------------------------------------------------
# Rendering (legal knowledge only)
# ---------------------------------------------------------------------------


def piece_glyph(record, viewer: "int | None") -> str:
    side = "R" if record.owner == RED else "B"
    if viewer is None:
        known = record.known_to(BLUE if record.owner == RED else RED)
    else:
        known = record.owner == viewer or record.known_to(viewer)
    return side + (PIECE_TYPE_CODES[record.true_type] if known else "?")


def render_board(state, viewer: "int | None") -> str:
    by_square = {
        record.current_square: record
        for record in state.pieces
        if record.alive and record.current_square is not None
    }
    header = "     " + "  ".join(chr(ord("a") + column) for column in range(BOARD_COLUMNS))
    lines = [header]
    for row in range(BOARD_ROWS - 1, -1, -1):
        cells = []
        for column in range(BOARD_COLUMNS):
            square = row * BOARD_COLUMNS + column
            if square in LAKE_SQUARE_SET:
                cells.append("~~")
            elif square in by_square:
                cells.append(piece_glyph(by_square[square], viewer))
            else:
                cells.append("··")
        lines.append(f"{row + 1:>3}  " + " ".join(cells))
    lines.append(header)
    return "\n".join(lines)


def move_text(action: int) -> str:
    source, destination = decode_action(action)
    return f"{square_name(source)} {square_name(destination)}"


# ---------------------------------------------------------------------------
# Seats
# ---------------------------------------------------------------------------


def human_turn(state, color_label: str) -> "int | None":
    """One human decision; `None` means resign."""
    legal = legal_actions(state)
    legal_set = set(legal)
    while True:
        try:
            text = input(f"[{color_label}] your move (e.g. b4 b5; 'help'): ").strip().lower()
        except EOFError:
            return None
        if text in ("resign", "quit", "exit"):
            return None
        if text in ("help", "h", "?"):
            print(
                "  <from> <to>  play a move        moves   list your legal moves\n"
                "  board        redraw the board   status  show the machine's status\n"
                "  resign       concede"
            )
            continue
        if text == "board":
            print(render_board(state, state.acting_player))
            continue
        if text == "moves":
            names = sorted(move_text(action) for action in legal)
            for start in range(0, len(names), 6):
                print("  " + "   ".join(names[start:start + 6]))
            continue
        if text == "status":
            print("  (status is shown for machine seats)")
            continue
        parts = text.split()
        if len(parts) != 2:
            print("  type two squares, like: b4 b5")
            continue
        try:
            action = encode_action(square_from_name(parts[0]), square_from_name(parts[1]))
        except Exception:
            print("  unreadable squares; columns a-j, rows 1-10")
            continue
        if action not in legal_set:
            print("  not a legal move ('moves' lists them)")
            continue
        return action


def machine_turn(player, state, mode: str, color_label: str) -> int:
    decision = player.decide(state, mode=mode)
    note = ""
    if decision.fallback_reason is not None:
        note = f"  [fallback: {decision.fallback_reason} -> direct {decision.move_model.upper()}]"
    elif decision.move_changed:
        note = "  [search changed the direct move]"
    budget = decision.preset_id or "direct"
    print(
        f"[{color_label}/{mode}] {move_text(decision.action_id)}  "
        f"({budget}, {decision.seconds:.3f}s){note}"
    )
    return decision.action_id


# ---------------------------------------------------------------------------
# The game
# ---------------------------------------------------------------------------


def draw_setups(arguments):
    """Two engine-ready setups, through the Phase 15 orientation gate."""
    from stratego.belief.phase15.orientation import check_board
    from stratego.search.phase15.boards import Phase15MatchSetupSources

    sources = Phase15MatchSetupSources()
    draws = {}
    for color in ("red", "blue"):
        seed = derive_search_seed(DOMAIN_PLAYER_SETUP, "cli", color, int(arguments.setup_seed))
        family = None
        if arguments.setup_source == "targeted_family":
            from stratego.search.phase15.contract import MATCH_FAMILY_KEYS

            family = MATCH_FAMILY_KEYS[int(arguments.setup_seed) % len(MATCH_FAMILY_KEYS)]
        draws[color] = sources.draw(
            arguments.setup_source, MATCH_LIBRARY_SPLIT, color, seed, family
        )
    # The same gate every Phase 15 board passes, on the human-play path too.
    check_board(draws["red"].canonical, draws["blue"].canonical)
    return draws["red"].engine, draws["blue"].engine


def main() -> int:
    arguments = build_parser().parse_args()
    seats = {RED: arguments.red, BLUE: arguments.blue}
    needs_player = any(seat != "human" for seat in seats.values())

    player = None
    if needs_player:
        from stratego.search.phase15.candidate import (
            DEFAULT_CANDIDATE_PATH,
            load_player_from_candidate,
        )

        path = Path(arguments.candidate or REPOSITORY_ROOT / DEFAULT_CANDIDATE_PATH)
        print("loading the frozen Phase 15 player (digest-checked)…")
        player, record = load_player_from_candidate(
            path, root=REPOSITORY_ROOT, device=arguments.device
        )
        print(
            f"  {pl.PLAYER_VERSION}: {record['selected_system']['move_model']} "
            f"{record['move_model']['model_state_digest'][:12]}… + "
            f"{record['selected_system']['belief_model']} "
            f"{record['belief_model']['state_digest'][:12]}…"
        )
        for color, seat in seats.items():
            if seat == "human":
                continue
            if seat not in player.systems and not player._is_direct(seat):
                raise SystemExit(
                    f"mode {seat!r} is not built into the frozen player; available "
                    f"modes are {sorted(player.systems) + list(pl.REQUIRED_MODES[:2])}"
                )
            cap = player.time_caps.get(seat)
            cap_text = f", cap {cap}s" if cap is not None else ""
            print(f"  {_COLOR_NAME[color]}: {seat} ({player.budget_text(seat)}{cap_text})")

    humans = [color for color, seat in seats.items() if seat == "human"]
    viewer = humans[0] if len(humans) == 1 else None

    red_setup, blue_setup = draw_setups(arguments)
    state = create_game(
        red_setup,
        blue_setup,
        rules=EVALUATION_RULES,
        game_id=f"phase15_cli|seed={arguments.setup_seed}",
    )
    print(
        f"\naccepted setups ({arguments.setup_source}/{MATCH_LIBRARY_SPLIT}, "
        f"seed {arguments.setup_seed}), orientation-gated; accepted evaluation rules"
    )
    if humans:
        print("you see your ranks and legally revealed opponent pieces; '?' is hidden\n")

    resigned = None
    while not state.terminal:
        actor = state.acting_player
        label = _COLOR_NAME[actor]
        seat = seats[actor]
        if seat == "human":
            print()
            print(render_board(state, actor))
            action = human_turn(state, label)
            if action is None:
                resigned = actor
                break
        else:
            action = machine_turn(player, state, seat, label)
        apply_action(state, action, legal=legal_actions(state))

    print()
    print(render_board(state, viewer))
    if resigned is not None:
        print(f"\n{_COLOR_NAME[resigned]} resigned.")
    else:
        winner = "draw" if state.winner is None else _COLOR_NAME[state.winner]
        print(f"\nresult: {winner} ({state.terminal_reason}) after {state.total_moves} plies")
    if player is not None:
        status = player.status()
        print(
            f"machine status: decisions {status['decisions']}, searched "
            f"{status['searched_decisions']}, move changes {status['move_changes']}, "
            f"fallbacks {status['fallbacks']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
