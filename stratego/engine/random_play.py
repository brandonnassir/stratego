"""Uniform random legal agent and seeded game generation.

Specification sources:

- `03_game_engine_spec.md` section 18 (randomness must be seedable)
- `PHASE_2_IMPLEMENTATION_INSTRUCTIONS.md` section 19

The agent exists purely to drive validation. It has no playing strength and is
not a Phase Four baseline opponent.

All randomness flows through an explicit `random.Random` instance seeded from an
integer, so every generated game is reproducible from its seed alone.
"""

import random

from .constants import BLUE, RED, RulesConfig, TRAINING_RULES
from .legal_moves import legal_actions
from .replay import ReplayRecord, build_replay_record
from .setup import random_setup
from .state import GameState, create_game
from .transition import apply_action


def select_random_action(
    state: GameState, rng: random.Random, actions: "list[int] | None" = None
) -> int:
    """Choose uniformly among the legal actions of the acting player."""
    if actions is None:
        actions = legal_actions(state)
    if not actions:
        raise ValueError("no legal actions available")
    return actions[rng.randrange(len(actions))]


def make_random_setups(seed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Deterministically derive both setups from one integer seed."""
    rng = random.Random(seed)
    return random_setup(rng, RED), random_setup(rng, BLUE)


def play_random_game(
    seed: int,
    rules: RulesConfig = TRAINING_RULES,
    game_id: str | None = None,
    max_plies: int | None = None,
) -> tuple[GameState, ReplayRecord]:
    """Play one complete game with uniform random legal moves.

    `max_plies` is a harness-level stop for tests that need a partial game; the
    rules' own absolute move limit is what terminates a normal game.
    """
    red_setup, blue_setup = make_random_setups(seed)
    identifier = game_id if game_id is not None else f"random-{seed}"
    state = create_game(red_setup, blue_setup, rules=rules, game_id=identifier)

    rng = random.Random(seed + 1_000_003)
    plies = 0
    while not state.terminal:
        if max_plies is not None and plies >= max_plies:
            break
        actions = legal_actions(state)
        if not actions:
            # Terminal detection runs at the end of every transition, so an
            # empty list here would mean the terminal check missed a case.
            raise RuntimeError(
                f"non-terminal state with no legal actions in game {identifier}"
            )
        apply_action(state, select_random_action(state, rng, actions), legal=actions)
        plies += 1

    record = build_replay_record(
        state,
        red_setup,
        blue_setup,
        seeds={"game_seed": seed, "agent": "uniform_random_legal"},
    )
    return state, record


def play_random_game_to_ply(
    seed: int, target_ply: int, rules: RulesConfig = TRAINING_RULES
) -> GameState:
    """Play a seeded random game and stop at `target_ply` (or at termination)."""
    state, _ = play_random_game(seed, rules=rules, max_plies=target_ply)
    return state


def generate_random_games(
    count: int, base_seed: int = 0, rules: RulesConfig = TRAINING_RULES
):
    """Yield `(state, record)` for `count` independently seeded games."""
    for index in range(count):
        yield play_random_game(base_seed + index, rules=rules, game_id=f"random-{base_seed + index}")
