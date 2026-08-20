#!/usr/bin/env python
"""Play against — or watch — the Phase 12 working search player.

The human/machine face of `stratego.search.phase12.player`: each seat is
`human` or one of the working player's modes (`direct`, `tiny`, `small`,
`medium`; `tiny` is the production default). Both machine seats share one
player instance, so the frozen production stack — accepted Phase 9 C1
policy/value, Agent 1C beliefs, TINY search, 0.5 s cap, direct fallback —
is loaded and digest-checked exactly once.

Usage:

    python scripts/play_phase12.py                        # human (red) vs tiny
    python scripts/play_phase12.py --red human --blue small
    python scripts/play_phase12.py --red direct --blue tiny   # machine vs machine

Moves are typed as two squares, e.g. `b4 b5`. Commands: `moves`, `status`,
`board`, `resign`, `help`.

Information boundary: the board is rendered from legal knowledge only. A
human seat sees its own army and only the opponent pieces that have been
legally revealed to it; in machine-vs-machine games the spectator view
shows a piece's rank only once the opposing player legally knows it. There
is no flag, mode or seat through which hidden ranks or the diagnostic
oracle could be shown — the player module refuses the oracle structurally.
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
from stratego.search.phase12 import player as pl  # noqa: E402
from stratego.search.phase12.contract import derive_phase12_seed  # noqa: E402

#: What a seat may be. There is deliberately no oracle entry — the oracle is
#: an offline diagnostic and the player refuses it structurally anyway.
SEAT_CHOICES = ("human",) + pl.PLAYER_MODES

_COLOR_NAME = {RED: "red", BLUE: "blue"}
_SETUP_SPLIT = "validation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play against, or watch, the Phase 12 working search player."
    )
    parser.add_argument(
        "--red", choices=SEAT_CHOICES, default="human",
        help="the red seat (default: human)",
    )
    parser.add_argument(
        "--blue", choices=SEAT_CHOICES, default=pl.DEFAULT_MODE,
        help=(
            f"the blue seat (default: {pl.DEFAULT_MODE}, the production mode; "
            f"'{pl.MAX_STRENGTH_MODE}' is the current maximum-strength candidate)"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--setup-seed", type=int, default=0,
        help="which accepted-library setups to draw (deterministic)",
    )
    parser.add_argument(
        "--setup-source", choices=("p10d", "neutral"), default="p10d",
        help="accepted setup source for both sides",
    )
    return parser


# ---------------------------------------------------------------------------
# Rendering (legal knowledge only)
# ---------------------------------------------------------------------------


def piece_glyph(record, viewer: "int | None") -> str:
    """Two characters for one piece, from the viewer's legal knowledge.

    A human viewer sees its own ranks and any opponent rank legally
    revealed to it. A spectator (`viewer=None`, machine-vs-machine) sees a
    rank only once the *opposing player* legally knows it — the public
    reading, not the engine's hidden truth.
    """
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
    lines = []
    header = "     " + "  ".join(chr(ord("a") + c) for c in range(BOARD_COLUMNS))
    lines.append(header)
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
        note = f"  [fallback: {decision.fallback_reason} -> {pl.FALLBACK_POLICY}]"
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
    from stratego.belief.phase11b.corpus import Phase11BSetupSources

    sources = Phase11BSetupSources()
    seeds = {
        color: derive_phase12_seed("cli_setup", color, int(arguments.setup_seed)) >> 1
        for color in ("red", "blue")
    }
    return (
        sources.draw(arguments.setup_source, _SETUP_SPLIT, "red", seeds["red"]),
        sources.draw(arguments.setup_source, _SETUP_SPLIT, "blue", seeds["blue"]),
    )


def main() -> int:
    arguments = build_parser().parse_args()
    seats = {RED: arguments.red, BLUE: arguments.blue}
    needs_player = any(seat != "human" for seat in seats.values())

    player = None
    if needs_player:
        print("loading the production player (digest-checked)…")
        player, identities = pl.load_search_player(
            REPOSITORY_ROOT, device=arguments.device
        )
        move_digest = identities["move_model_identity"]["model_state_digest"]
        belief_digest = identities["belief_model_identity"]["checkpoint_sha256"]
        print(
            f"  {pl.PLAYER_VERSION}: accepted Phase 9 C1 {move_digest[:12]}… + "
            f"agent1c {belief_digest[:12]}…"
        )
        for color, seat in seats.items():
            if seat != "human":
                cap = player.time_caps.get(seat)
                budget = player.budget_text(seat)
                cap_text = f", cap {cap}s" if cap is not None else ""
                print(f"  {_COLOR_NAME[color]}: {seat} ({budget}{cap_text})")

    humans = [color for color, seat in seats.items() if seat == "human"]
    viewer = humans[0] if len(humans) == 1 else None

    red_setup, blue_setup = draw_setups(arguments)
    state = create_game(
        red_setup, blue_setup, rules=EVALUATION_RULES,
        game_id=f"phase12_cli|seed={arguments.setup_seed}",
    )
    print(
        f"\naccepted setups ({arguments.setup_source}/{_SETUP_SPLIT}, "
        f"seed {arguments.setup_seed}); rules: accepted evaluation rules"
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
            f"machine status: decisions {status['decisions']}, "
            f"fallbacks {status['fallback_total']} {status['fallbacks']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
