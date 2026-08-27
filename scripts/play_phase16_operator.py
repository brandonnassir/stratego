#!/usr/bin/env python
"""Play an operator series game against the frozen Phase 15 player, logged.

The Phase 16 operator-series face of `stratego.search.phase15.player`
(imported, never edited): the same digest-checked frozen stack as
`scripts/play_phase15.py`, plus the section-6 logging path — every finished
game appends one `phase16_operator_game_v1` JSON line to
`data/phase16/operator_games.jsonl`: seats, colours, both setups (canonical
tuples plus family when drawn), the full action history, the result and
per-move wall times. Until Agent 2's `play_phase16.py` lands, this script is
the operator's entry point.

Usage:

    .venv/bin/python scripts/play_phase16_operator.py \
        --series rebaseline_v1 --game-index 1 \
        --red human --blue maximum_strength

    # the operator brings their own setup (capture-tool grid format):
    .venv/bin/python scripts/play_phase16_operator.py \
        --series exam_v1 --game-index 7 --red human --blue maximum_strength \
        --red-setup-file my_setup.txt

Moves are typed as two squares, e.g. `b4 b5`. Commands: `moves`, `board`,
`resign`, `help`.

Orientation and information boundary
------------------------------------
Both setups pass the imported Phase 15 section-4 gate before `create_game`.
The board is rendered from legal knowledge only: a human seat sees its own
army and only legally revealed opponent pieces; there is no flag, mode or
seat through which a hidden rank or the diagnostic oracle could be shown.
"""

from __future__ import annotations

import argparse
import sys
import time
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
from stratego.evaluation.phase16.operator_log import (  # noqa: E402
    DEFAULT_LOG_PATH,
    OperatorGameLogger,
)
from stratego.search.phase15 import player as pl  # noqa: E402
from stratego.search.phase15.contract import (  # noqa: E402
    DOMAIN_PLAYER_SETUP,
    MATCH_LIBRARY_SPLIT,
    MATCH_SETUP_SOURCES,
    derive_search_seed,
)

SEAT_CHOICES = ("human",) + pl.PLAYER_MODES
_COLOR_NAME = {RED: "red", BLUE: "blue"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play a logged operator-series game against the Phase 15 player."
    )
    parser.add_argument("--red", choices=SEAT_CHOICES, default="human")
    parser.add_argument("--blue", choices=SEAT_CHOICES, default=pl.MODE_MAX_STRENGTH)
    parser.add_argument("--series", default=None, help="e.g. rebaseline_v1 or exam_v1")
    parser.add_argument("--game-index", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--setup-seed", type=int, default=0)
    parser.add_argument(
        "--setup-source", choices=MATCH_SETUP_SOURCES, default="phase14_learned",
        help="library source for seats without an explicit setup file",
    )
    parser.add_argument(
        "--red-setup-file", default=None,
        help="4x10 grid file for the red army (capture-tool format, back row first)",
    )
    parser.add_argument(
        "--blue-setup-file", default=None,
        help="4x10 grid file for the blue army (capture-tool format, back row first)",
    )
    parser.add_argument("--log", default=None, help="operator log path override")
    parser.add_argument("--no-log", action="store_true", help="do not append a log line")
    parser.add_argument("--notes", default=None)
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
                "  board        redraw the board   resign  concede"
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
    budget = decision.preset_id or "direct"
    print(
        f"[{color_label}/{mode}] {move_text(decision.action_id)}  "
        f"({budget}, {decision.seconds:.3f}s){note}"
    )
    return decision.action_id


# ---------------------------------------------------------------------------
# Setups
# ---------------------------------------------------------------------------


def resolve_setups(arguments, logger: "OperatorGameLogger | None"):
    """Engine-ready setups for both seats, each through the imported gate."""
    from stratego.belief.phase15.orientation import check_board, oriented_for
    from stratego.search.phase15.boards import Phase15MatchSetupSources

    grid_files = {"red": arguments.red_setup_file, "blue": arguments.blue_setup_file}
    canonicals: dict = {}
    metadata: dict = {}
    sources = None
    for color in ("red", "blue"):
        if grid_files[color]:
            sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
            from phase16_capture_setup import parse_grid

            canonicals[color] = parse_grid(Path(grid_files[color]).read_text())
            metadata[color] = {
                "source": "operator_entered",
                "family_key": None,
                "base_setup_id": None,
            }
        else:
            if sources is None:
                sources = Phase15MatchSetupSources()
            seed = derive_search_seed(
                DOMAIN_PLAYER_SETUP, "p16op", color, int(arguments.setup_seed)
            )
            family = None
            if arguments.setup_source == "targeted_family":
                from stratego.search.phase15.contract import MATCH_FAMILY_KEYS

                family = MATCH_FAMILY_KEYS[
                    int(arguments.setup_seed) % len(MATCH_FAMILY_KEYS)
                ]
            draw = sources.draw(
                arguments.setup_source, MATCH_LIBRARY_SPLIT, color, seed, family
            )
            canonicals[color] = tuple(draw.canonical)
            metadata[color] = {
                "source": arguments.setup_source,
                "family_key": draw.family_key,
                "base_setup_id": draw.base_setup_id,
            }
    check_board(canonicals["red"], canonicals["blue"])
    if logger is not None:
        for color in ("red", "blue"):
            logger.set_setup(color, canonical=canonicals[color], **metadata[color])
    return (
        oriented_for(canonicals["red"], RED),
        oriented_for(canonicals["blue"], BLUE),
    )


# ---------------------------------------------------------------------------
# The game
# ---------------------------------------------------------------------------


def main() -> int:
    arguments = build_parser().parse_args()
    seats = {RED: arguments.red, BLUE: arguments.blue}
    needs_player = any(seat != "human" for seat in seats.values())

    logger = None
    if not arguments.no_log:
        logger = OperatorGameLogger(
            seats={"red": arguments.red, "blue": arguments.blue},
            script="scripts/play_phase16_operator.py",
            series=arguments.series,
            game_index=arguments.game_index,
            rules="accepted evaluation rules",
            notes=arguments.notes,
        )

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
        for color, seat in seats.items():
            if seat == "human":
                continue
            if seat not in player.systems and not player._is_direct(seat):
                raise SystemExit(f"mode {seat!r} is not built into the frozen player")
            cap = player.time_caps.get(seat)
            cap_text = f", cap {cap}s" if cap is not None else ""
            print(f"  {_COLOR_NAME[color]}: {seat} ({player.budget_text(seat)}{cap_text})")

    humans = [color for color, seat in seats.items() if seat == "human"]
    viewer = humans[0] if len(humans) == 1 else None

    red_setup, blue_setup = resolve_setups(arguments, logger)
    state = create_game(
        red_setup,
        blue_setup,
        rules=EVALUATION_RULES,
        game_id=(
            f"phase16_operator|series={arguments.series}"
            f"|g={arguments.game_index}|seed={arguments.setup_seed}"
        ),
    )
    print("\nboth setups orientation-gated; accepted evaluation rules")
    if humans:
        print("you see your ranks and legally revealed opponent pieces; '?' is hidden\n")

    resigned = None
    while not state.terminal:
        actor = state.acting_player
        label = _COLOR_NAME[actor]
        seat = seats[actor]
        started = time.perf_counter()
        if seat == "human":
            print()
            print(render_board(state, actor))
            action = human_turn(state, label)
            if action is None:
                resigned = actor
                break
        else:
            action = machine_turn(player, state, seat, label)
        if logger is not None:
            logger.record_move(label, action, time.perf_counter() - started)
        apply_action(state, action, legal=legal_actions(state))

    print()
    print(render_board(state, viewer))
    if resigned is not None:
        result, winner = "resignation", _COLOR_NAME[BLUE if resigned == RED else RED]
        reason = f"{_COLOR_NAME[resigned]}_resigned"
        print(f"\n{_COLOR_NAME[resigned]} resigned.")
    else:
        winner = None if state.winner is None else _COLOR_NAME[state.winner]
        result = "draw" if winner is None else "win"
        reason = str(state.terminal_reason)
        print(f"\nresult: {winner or 'draw'} ({reason}) after {state.total_moves} plies")

    if logger is not None:
        logger.finish(
            result=result,
            winner=winner,
            terminal_reason=reason,
            plies=int(state.total_moves),
            machine_status=player.status() if player is not None else None,
        )
        destination = logger.append(
            arguments.log or DEFAULT_LOG_PATH, root=REPOSITORY_ROOT
        )
        print(f"logged -> {destination}")
    if player is not None:
        status = player.status()
        print(
            f"machine status: decisions {status['decisions']}, searched "
            f"{status['searched_decisions']}, fallbacks {status['fallbacks']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
