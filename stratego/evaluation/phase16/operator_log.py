"""Phase 16 Agent 1 section 6: the operator game log and setup harvesting.

Every operator game appends **one JSON line** to
`data/phase16/operator_games.jsonl` (schema `phase16_operator_game_v1`):
timestamp, script and mode, seats, colours, both setups (canonical tuples
plus family id when drawn from a library), the full action history, the
result, ply count and per-move wall times.

The logger is a thin wrapper any `play_phase15.py`-style script can call:

```python
logger = OperatorGameLogger(seats={"red": "human", "blue": "maximum_strength"},
                            script="scripts/play_phase16_operator.py",
                            series="rebaseline_v1", game_index=3)
logger.set_setup("red", canonical=..., source="operator_entered")
logger.set_setup("blue", canonical=..., source="phase14_learned", family_key=...)
logger.record_move("red", action_id, seconds)
...
logger.finish(result="win", winner="blue", terminal_reason=..., plies=...)
logger.append(path)
```

`harvest_operator_setups` extracts every human seat's canonical setup from
the log into the adversarial library's `operator_harvest` family,
deduplicated by tuple, through the same validation gate as every other
entry.
"""

from __future__ import annotations

import datetime as _datetime
import json
from pathlib import Path

from .contract import OPERATOR_LOG_SCHEMA, Phase16MeasurementError

#: Where operator games are logged.
DEFAULT_LOG_PATH = Path("data/phase16/operator_games.jsonl")

#: Seat labels a human occupies. Anything else is a machine mode name.
HUMAN_SEATS = ("human", "operator")


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class OperatorGameLogger:
    """Collects one game's record and appends it as a single JSON line."""

    def __init__(
        self,
        *,
        seats: dict,
        script: str,
        series: "str | None" = None,
        game_index: "int | None" = None,
        rules: "str | None" = None,
        notes: "str | None" = None,
    ) -> None:
        if set(seats) != {"red", "blue"}:
            raise Phase16MeasurementError(
                f"seats must name exactly red and blue, got {sorted(seats)}"
            )
        self.record = {
            "schema": OPERATOR_LOG_SCHEMA,
            "started_utc": utc_now(),
            "script": str(script),
            "seats": {color: str(seat) for color, seat in seats.items()},
            "operator_color": next(
                (color for color, seat in seats.items() if seat in HUMAN_SEATS), None
            ),
            "series": series,
            "game_index": game_index,
            "rules": rules,
            "notes": notes,
            "setups": {},
            "actions": [],
            "move_seconds": {"red": [], "blue": []},
            "result": None,
        }

    def set_setup(
        self,
        color: str,
        *,
        canonical,
        source: str,
        family_key: "str | None" = None,
        base_setup_id: "str | None" = None,
    ) -> None:
        if color not in ("red", "blue"):
            raise Phase16MeasurementError(f"unknown colour {color!r}")
        self.record["setups"][color] = {
            "canonical": [int(piece) for piece in canonical],
            "source": source,
            "family_key": family_key,
            "base_setup_id": base_setup_id,
        }

    def record_move(self, color: str, action_id: int, seconds: float) -> None:
        if color not in ("red", "blue"):
            raise Phase16MeasurementError(f"unknown colour {color!r}")
        self.record["actions"].append(int(action_id))
        self.record["move_seconds"][color].append(round(float(seconds), 4))

    def finish(
        self,
        *,
        result: str,
        winner: "str | None",
        terminal_reason: str,
        plies: int,
        machine_status: "dict | None" = None,
    ) -> dict:
        self.record["result"] = {
            "result": result,
            "winner": winner,
            "terminal_reason": terminal_reason,
            "plies": int(plies),
        }
        self.record["ply_count"] = int(plies)
        self.record["finished_utc"] = utc_now()
        if machine_status is not None:
            self.record["machine_status"] = dict(machine_status)
        return self.record

    def append(self, path: "Path | str" = DEFAULT_LOG_PATH, *, root: "Path | str" = ".") -> Path:
        if self.record.get("result") is None:
            raise Phase16MeasurementError(
                "finish() must be called before the game is appended"
            )
        missing = [color for color in ("red", "blue") if color not in self.record["setups"]]
        if missing:
            raise Phase16MeasurementError(f"no setup recorded for {missing}")
        full = Path(root) / Path(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        with full.open("a") as handle:
            handle.write(json.dumps(self.record, sort_keys=True) + "\n")
        return full


def read_log(path: "Path | str" = DEFAULT_LOG_PATH, *, root: "Path | str" = ".") -> "list[dict]":
    full = Path(root) / Path(path)
    if not full.is_file():
        return []
    games = []
    with full.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            game = json.loads(line)
            if game.get("schema") != OPERATOR_LOG_SCHEMA:
                raise Phase16MeasurementError(
                    f"{full}:{line_number} is not a {OPERATOR_LOG_SCHEMA} line"
                )
            games.append(game)
    return games


def operator_series_summary(games: "list[dict]", series: "str | None" = None) -> dict:
    """EWR by game index for one series — the trend line the protocol tracks.

    The score is the *machine's* EWR (draws count half), because the exam's
    pass condition is stated for the model.
    """
    rows = []
    for game in games:
        if series is not None and game.get("series") != series:
            continue
        operator_color = game.get("operator_color")
        if operator_color is None:
            continue
        machine_color = "blue" if operator_color == "red" else "red"
        result = game.get("result") or {}
        winner = result.get("winner")
        score = 0.5 if winner is None else (1.0 if winner == machine_color else 0.0)
        rows.append(
            {
                "game_index": game.get("game_index"),
                "started_utc": game.get("started_utc"),
                "operator_color": operator_color,
                "machine_score": score,
                "plies": result.get("plies"),
            }
        )
    rows.sort(key=lambda row: (row["game_index"] is None, row["game_index"]))
    scores = [row["machine_score"] for row in rows]
    running = []
    total = 0.0
    for index, score in enumerate(scores, start=1):
        total += score
        running.append(round(total / index, 4))
    return {
        "series": series,
        "games": len(rows),
        "machine_ewr": round(total / len(scores), 4) if scores else None,
        "by_game": rows,
        "running_machine_ewr": running,
    }


def harvest_operator_setups(
    *,
    log_path: "Path | str" = DEFAULT_LOG_PATH,
    library_path: "Path | str | None" = None,
    root: "Path | str" = ".",
    write: bool = True,
) -> dict:
    """Extract operator setups from the log into `operator_harvest`.

    Deduplicated by canonical tuple — against the harvest family and within
    the log. Returns what was found and what was appended.
    """
    from .adversarial import (
        DEFAULT_LIBRARY_PATH,
        append_harvest_setup,
        load_library,
        save_library,
    )

    library_path = DEFAULT_LIBRARY_PATH if library_path is None else library_path
    games = read_log(log_path, root=root)
    document = load_library(library_path, root=root)
    found = 0
    appended = []
    for game in games:
        operator_color = game.get("operator_color")
        if operator_color is None:
            continue
        setup = (game.get("setups") or {}).get(operator_color)
        if not setup:
            continue
        found += 1
        entry = append_harvest_setup(
            document,
            tuple(setup["canonical"]),
            provenance={
                "log_path": str(log_path),
                "series": game.get("series"),
                "game_index": game.get("game_index"),
                "started_utc": game.get("started_utc"),
                "declared_source": setup.get("source"),
            },
            captured_utc=utc_now(),
        )
        if entry is not None:
            appended.append(entry["setup_id"])
    if write and appended:
        save_library(document, library_path, root=root)
    return {
        "games_scanned": len(games),
        "operator_setups_found": found,
        "appended": appended,
        "harvest_revision": document.get("harvest_revision"),
        "written": bool(write and appended),
    }


__all__ = [
    "DEFAULT_LOG_PATH",
    "HUMAN_SEATS",
    "OperatorGameLogger",
    "harvest_operator_setups",
    "operator_series_summary",
    "read_log",
    "utc_now",
]
