"""Shared fixtures for the Phase Two validation suite.

The engine only knows how to build a game from two complete legal setups, which
is right for real play but impractical for targeted rule tests. `make_position`
builds an arbitrary sparse position by creating the full 80-piece table and
marking every piece the test did not ask for as already captured.

Synthetic positions place pieces on squares that are not necessarily the setup
square implied by their slot, so tests that call `check_invariants` on them pass
`check_setup_slots=False`. Real games always satisfy the stronger check.
"""

from collections import Counter

from stratego.engine.constants import (
    BLUE,
    PHASE_PLAY,
    PIECE_COUNTS,
    PIECE_TYPE_BY_NAME,
    PLAYERS,
    RED,
    RulesConfig,
    TRAINING_RULES,
)
from stratego.engine.coordinates import square_from_name
from stratego.engine.pieces import PieceRecord, make_piece_id
from stratego.engine.setup import setup_squares
from stratego.engine.state import GameState
from stratego.engine.transition import apply_action_by_squares

# Convenience aliases so tests can read `T["marshal"]` instead of importing
# twelve module-level constants.
T = dict(PIECE_TYPE_BY_NAME)


def square(name: str) -> int:
    """`a1` -> 0. Thin alias that keeps test bodies short."""
    return square_from_name(name)


def squares(*names: str) -> list[int]:
    return [square_from_name(name) for name in names]


def make_position(
    red: "dict[str, str] | None" = None,
    blue: "dict[str, str] | None" = None,
    acting_player: int = RED,
    rules: RulesConfig = TRAINING_RULES,
    moved: "set[str] | None" = None,
    revealed: "set[str] | None" = None,
    battleless_moves: int = 0,
    total_moves: int = 0,
    game_id: str = "synthetic",
) -> GameState:
    """Build a synthetic position.

    `red` and `blue` map human square names to piece type names, for example
    `{"a4": "miner", "b4": "scout"}`. Every piece not placed is created as a
    captured piece so the 40-records-per-player invariant still holds.

    `moved` and `revealed` are sets of square names. A revealed piece is one the
    opponent legally knows; a moved piece is one that has moved at least once.
    """
    placements = {RED: red or {}, BLUE: blue or {}}
    moved_squares = {square_from_name(name) for name in (moved or set())}
    revealed_squares = {square_from_name(name) for name in (revealed or set())}

    all_squares = [
        square_from_name(name) for side in placements.values() for name in side
    ]
    duplicates = [item for item, count in Counter(all_squares).items() if count > 1]
    if duplicates:
        raise ValueError(f"two pieces placed on the same square: {duplicates}")

    records: list[PieceRecord] = []
    board: list[int | None] = [None] * 100

    for player in PLAYERS:
        requested = {
            square_from_name(name): PIECE_TYPE_BY_NAME[type_name]
            for name, type_name in placements[player].items()
        }
        remaining = Counter(
            {piece_type: count for piece_type, count in PIECE_COUNTS.items()}
        )
        for piece_type in requested.values():
            if remaining[piece_type] <= 0:
                raise ValueError(
                    f"position requests more pieces of type {piece_type} than exist"
                )
            remaining[piece_type] -= 1

        used = set(requested)
        filler_squares = [item for item in setup_squares(player) if item not in used]

        slot = 0
        for placed_square in sorted(requested):
            piece_type = requested[placed_square]
            record = PieceRecord(
                piece_id=make_piece_id(player, slot),
                owner=player,
                true_type=piece_type,
                # Synthetic positions treat the placed square as the origin, so
                # the "Flag and Bomb never leave their starting square"
                # invariant stays true.
                starting_square=placed_square,
                current_square=placed_square,
                has_moved=placed_square in moved_squares,
            )
            record.set_known_to(player, "own_piece")
            if placed_square in revealed_squares:
                record.set_known_to(1 - player, "combat")
            records.append(record)
            board[placed_square] = record.piece_id
            slot += 1

        # Fill the remaining inventory with already-captured pieces so counts,
        # setup-memory planes and unresolved-inventory maths stay well defined.
        for piece_type, count in sorted(remaining.items()):
            for _ in range(count):
                record = PieceRecord(
                    piece_id=make_piece_id(player, slot),
                    owner=player,
                    true_type=piece_type,
                    starting_square=filler_squares[slot - len(requested)],
                    current_square=None,
                    alive=False,
                    capture_ply=0,
                )
                record.set_known_to(player, "own_piece")
                record.set_known_to(1 - player, "combat")
                records.append(record)
                slot += 1

    return GameState(
        rules=rules,
        game_id=game_id,
        board=board,
        pieces=records,
        acting_player=acting_player,
        phase=PHASE_PLAY,
        total_moves=total_moves,
        battleless_moves=battleless_moves,
    )


def cell(observation, channel: int, square_name: str, observer: int = RED) -> float:
    """Read one observation cell by human square name, in `observer`'s frame."""
    from stratego.engine.coordinates import to_perspective

    normalized = to_perspective(square_from_name(square_name), observer)
    row, column = divmod(normalized, 10)
    return float(observation[channel, row, column])


def plane_sum(observation, channel: int) -> float:
    """Total of one observation plane, useful for `exactly N ones` assertions."""
    return float(observation[channel].sum())


def piece_at(state: GameState, name: str):
    """Piece record standing on the named square, or `None`."""
    return state.piece_at(square_from_name(name))


def play(state: GameState, *moves: str) -> list[list[dict]]:
    """Apply a sequence of moves written as `"a4 a5"` and return their events."""
    generated = []
    for move in moves:
        source_name, destination_name = move.split()
        generated.append(
            apply_action_by_squares(
                state, square_from_name(source_name), square_from_name(destination_name)
            )
        )
    return generated


def full_inventory_setup() -> tuple[int, ...]:
    """A deterministic legal setup listing the official inventory in type order."""
    pieces: list[int] = []
    for piece_type, count in sorted(PIECE_COUNTS.items()):
        pieces.extend([piece_type] * count)
    return tuple(pieces)


def nonterminal_state(target_ply: int, first_seed: int = 0, rules: RulesConfig = TRAINING_RULES):
    """A seeded random position at `target_ply` that is guaranteed still running.

    Random games sometimes end early, so the helper walks seeds until it finds
    one that survives to the requested ply. It stays deterministic because the
    seeds are tried in a fixed order.
    """
    from stratego.engine.random_play import play_random_game_to_ply

    for seed in range(first_seed, first_seed + 200):
        state = play_random_game_to_ply(seed, target_ply, rules=rules)
        if not state.terminal and state.total_moves == target_ply:
            return state
    raise RuntimeError(f"no seeded game survived to ply {target_ply}")


def known_good_game(rules: RulesConfig = TRAINING_RULES, game_id: str = "fixture"):
    """A real game built from two deterministic legal setups."""
    from stratego.engine.state import create_game

    red = full_inventory_setup()
    blue = tuple(reversed(full_inventory_setup()))
    return create_game(red, blue, rules=rules, game_id=game_id)


__all__ = [
    "BLUE",
    "RED",
    "T",
    "cell",
    "full_inventory_setup",
    "known_good_game",
    "make_position",
    "nonterminal_state",
    "piece_at",
    "plane_sum",
    "play",
    "square",
    "squares",
]
