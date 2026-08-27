#!/usr/bin/env python
"""Play against — or watch — the Phase 16 varied player.

Supersedes `scripts/play_phase15.py`, which stays untouched: every Phase 15
mode is available here by import, and two new modes carry the selected
stochastic configuration from `phase16_stochastic_candidate_v1`:

```text
varied_strength   selected (tau, tau_r, top-p) at MEDIUM
varied_fast       the same configuration at TINY
```

Usage:

    python scripts/play_phase16.py                                   # human (red) vs varied_strength
    python scripts/play_phase16.py --red human --blue maximum_strength
    python scripts/play_phase16.py --red varied_fast --blue p24_direct

Operator logging
----------------
Every finished game appends one JSON line. When Agent 1's logging module
(`stratego.evaluation.phase16.operator_log`) is present it is used; until
then a local fallback writes the same schema to
`data/phase16/operator_games.jsonl`: timestamp, script + modes, seats,
colors, both setups (canonical tuples + family id), full action history,
result, ply count, per-move wall times.

Information boundary
--------------------
Identical to Phase 15: the board is rendered from legal knowledge only, and
there is no flag, mode or seat through which a hidden rank or the diagnostic
oracle could be shown — both player modules refuse the oracle by name and by
absence from their mode tables.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.constants import BLUE, RED  # noqa: E402
from stratego.engine.legal_moves import legal_actions  # noqa: E402
from stratego.engine.state import create_game  # noqa: E402
from stratego.engine.transition import apply_action  # noqa: E402
from stratego.evaluation.match_spec import EVALUATION_RULES  # noqa: E402
from stratego.search.phase15 import player as pl15  # noqa: E402
from stratego.search.phase15.contract import (  # noqa: E402
    DOMAIN_PLAYER_SETUP,
    MATCH_FAMILY_KEYS,
    MATCH_LIBRARY_SPLIT,
    MATCH_SETUP_SOURCES,
    derive_search_seed,
)
from stratego.search.phase16.contract import (  # noqa: E402
    MODE_VARIED_FAST,
    MODE_VARIED_STRENGTH,
    VARIED_MODES,
)

OPERATOR_LOG_PATH = REPOSITORY_ROOT / "data/phase16/operator_games.jsonl"
OPERATOR_LOG_SCHEMA = "phase16_operator_game_v1"


def _load_phase15_cli():
    """The accepted Phase 15 CLI as a library — rendering and turns reused."""
    spec = importlib.util.spec_from_file_location(
        "play_phase15_lib", REPOSITORY_ROOT / "scripts/play_phase15.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P15 = _load_phase15_cli()

SEAT_CHOICES = ("human",) + pl15.PLAYER_MODES + VARIED_MODES

_COLOR_NAME = {RED: "red", BLUE: "blue"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play against, or watch, the Phase 16 varied player."
    )
    parser.add_argument("--red", choices=SEAT_CHOICES, default="human")
    parser.add_argument(
        "--blue",
        choices=SEAT_CHOICES,
        default=MODE_VARIED_STRENGTH,
        help=(
            f"the blue seat (default: {MODE_VARIED_STRENGTH}; "
            f"'{MODE_VARIED_FAST}' is the fast varied mode; every Phase 15 "
            "mode is also accepted)"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--setup-seed", type=int, default=0)
    parser.add_argument(
        "--setup-source", choices=MATCH_SETUP_SOURCES, default="phase14_learned"
    )
    parser.add_argument(
        "--candidate", default=None, help="path to phase16_stochastic_candidate_v1.json"
    )
    parser.add_argument(
        "--no-log", action="store_true", help="do not append the operator game log"
    )
    return parser


# ---------------------------------------------------------------------------
# Setups (accepted sources, orientation-gated; canonical kept for the log)
# ---------------------------------------------------------------------------


def draw_setups(arguments):
    from stratego.belief.phase15.orientation import check_board
    from stratego.search.phase15.boards import Phase15MatchSetupSources

    sources = Phase15MatchSetupSources()
    draws = {}
    for color in ("red", "blue"):
        seed = derive_search_seed(
            DOMAIN_PLAYER_SETUP, "cli16", color, int(arguments.setup_seed)
        )
        family = None
        if arguments.setup_source == "targeted_family":
            family = MATCH_FAMILY_KEYS[int(arguments.setup_seed) % len(MATCH_FAMILY_KEYS)]
        draws[color] = sources.draw(
            arguments.setup_source, MATCH_LIBRARY_SPLIT, color, seed, family
        )
    check_board(draws["red"].canonical, draws["blue"].canonical)
    return draws


# ---------------------------------------------------------------------------
# Operator logging (Agent 1's module when present; same-schema fallback)
# ---------------------------------------------------------------------------


def append_operator_log(record: dict) -> str:
    """Returns a short description of the sink used."""
    try:
        from stratego.evaluation.phase16 import operator_log as agent1_log

        for name in ("append_game", "log_game", "append", "write_game"):
            hook = getattr(agent1_log, name, None)
            if callable(hook):
                hook(record)
                return f"stratego.evaluation.phase16.operator_log.{name}"
    except Exception:  # noqa: BLE001 - the fallback exists for exactly this
        pass
    OPERATOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OPERATOR_LOG_PATH, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return str(OPERATOR_LOG_PATH)


# ---------------------------------------------------------------------------
# Machine seats
# ---------------------------------------------------------------------------


def varied_turn(player16, state, mode: str, color_label: str, game_id: str) -> int:
    decision = player16.decide(state, mode=mode, game_id=game_id)
    note = ""
    if decision.fallback_reason is not None:
        note = f"  [fallback: {decision.fallback_reason} -> direct {decision.move_model.upper()}]"
    elif decision.sampled_move_changed:
        note = "  [sampled away from argmax]"
    elif decision.move_changed:
        note = "  [search changed the direct move]"
    print(
        f"[{color_label}/{mode}] {_P15.move_text(decision.action_id)}  "
        f"({decision.preset_id}, {decision.seconds:.3f}s){note}"
    )
    return decision.action_id


def main() -> int:
    arguments = build_parser().parse_args()
    seats = {RED: arguments.red, BLUE: arguments.blue}
    needs_p15 = any(seat in pl15.PLAYER_MODES for seat in seats.values())
    needs_p16 = any(seat in VARIED_MODES for seat in seats.values())

    player15 = None
    if needs_p15:
        from stratego.search.phase15.candidate import (
            DEFAULT_CANDIDATE_PATH,
            load_player_from_candidate,
        )

        print("loading the frozen Phase 15 player (digest-checked)…")
        player15, record15 = load_player_from_candidate(
            REPOSITORY_ROOT / DEFAULT_CANDIDATE_PATH,
            root=REPOSITORY_ROOT,
            device=arguments.device,
        )
        del record15

    player16 = None
    record16 = None
    if needs_p16:
        from stratego.search.phase16.candidate import (
            DEFAULT_CANDIDATE_PATH_16,
            load_varied_player,
        )

        path = Path(arguments.candidate or REPOSITORY_ROOT / DEFAULT_CANDIDATE_PATH_16)
        print("loading the frozen Phase 16 varied player (digest-checked)…")
        player16, record16 = load_varied_player(
            path, root=REPOSITORY_ROOT, device=arguments.device
        )
        configuration = record16["selected_configuration"]
        print(
            f"  {configuration['arm_id']}: tau={configuration['tau']} "
            f"tau_r={configuration['tau_r']} top_p={configuration['top_p']} over "
            f"{configuration['pairing_id']}"
        )
        for color, seat in seats.items():
            if seat in VARIED_MODES:
                cap = player16.time_caps.get(seat)
                print(
                    f"  {_COLOR_NAME[color]}: {seat} "
                    f"({player16.budget_text(seat)}, cap {cap}s)"
                )

    humans = [color for color, seat in seats.items() if seat == "human"]
    viewer = humans[0] if len(humans) == 1 else None

    draws = draw_setups(arguments)
    game_id = f"phase16_cli|seed={arguments.setup_seed}|{int(time.time())}"
    state = create_game(
        draws["red"].engine,
        draws["blue"].engine,
        rules=EVALUATION_RULES,
        game_id=game_id,
    )
    print(
        f"\naccepted setups ({arguments.setup_source}/{MATCH_LIBRARY_SPLIT}, "
        f"seed {arguments.setup_seed}), orientation-gated; accepted evaluation rules"
    )
    if humans:
        print("you see your ranks and legally revealed opponent pieces; '?' is hidden\n")

    action_history: list[int] = []
    move_walls: list[float] = []
    resigned = None
    while not state.terminal:
        actor = state.acting_player
        label = _COLOR_NAME[actor]
        seat = seats[actor]
        tick = time.perf_counter()
        if seat == "human":
            print()
            print(_P15.render_board(state, actor))
            action = _P15.human_turn(state, label)
            if action is None:
                resigned = actor
                break
        elif seat in VARIED_MODES:
            action = varied_turn(player16, state, seat, label, game_id)
        else:
            action = _P15.machine_turn(player15, state, seat, label)
        move_walls.append(round(time.perf_counter() - tick, 4))
        action_history.append(int(action))
        apply_action(state, action, legal=legal_actions(state))

    print()
    print(_P15.render_board(state, viewer))
    if resigned is not None:
        result = f"{_COLOR_NAME[resigned]}_resigned"
        print(f"\n{_COLOR_NAME[resigned]} resigned.")
    else:
        winner = "draw" if state.winner is None else _COLOR_NAME[state.winner]
        result = winner
        print(
            f"\nresult: {winner} ({state.terminal_reason}) after "
            f"{state.total_moves} plies"
        )

    if not arguments.no_log:
        record = {
            "schema": OPERATOR_LOG_SCHEMA,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script": "scripts/play_phase16.py",
            "game_id": game_id,
            "seats": {"red": seats[RED], "blue": seats[BLUE]},
            "human_seats": [_COLOR_NAME[color] for color in humans],
            "setup_source": arguments.setup_source,
            "setup_seed": int(arguments.setup_seed),
            "setups": {
                color: {
                    "canonical": list(draws[color].canonical),
                    "engine": list(draws[color].engine),
                    "family_key": draws[color].family_key,
                    "base_setup_id": draws[color].base_setup_id,
                }
                for color in ("red", "blue")
            },
            "action_history": action_history,
            "result": result,
            "terminal_reason": None if resigned is not None else str(state.terminal_reason),
            "plies": int(state.total_moves),
            "per_move_wall_seconds": move_walls,
            "varied_player": (
                None
                if record16 is None
                else {
                    "arm_id": record16["selected_configuration"]["arm_id"],
                    "artifact": record16["artifact"],
                }
            ),
        }
        sink = append_operator_log(record)
        print(f"logged game to {sink}")

    if player16 is not None:
        status = player16.status()
        print(
            f"varied player status: decisions {status['decisions']}, move changes "
            f"{status['move_changes']}, sampled changes "
            f"{status['sampled_move_changes']}, fallbacks {status['fallbacks']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
