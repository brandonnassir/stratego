"""Phase 16 Agent 1 section 4: `phase16_benchmark_v1`, the canonical pack.

Design
------
```text
opponent        10   the Phase 15 Stage B roster, re-exported by the contract
setup source     3   neutral_v1 / phase14_learned / targeted_family
player colour    2   red / blue
boards per cell  2   -> 120 paired boards; ordinal 0 is the 60-board quick subset
```

Every board is drawn fresh from the accepted library's ``validation`` split
through Phase 15's match setup sources — the identical accepted draw path
Stage B used — under **Phase 16** seeds, and every setup leaves through the
imported orientation gate before it may describe a game. No Phase 15 board
is reused: the Phase 15 streams use personalization ``strat-p15s`` and the
Phase 16 streams use ``strat-p16m``, so the two packs cannot share a draw
even at equal ordinals.

Frozen, and checkably frozen
----------------------------
The manifest stores every board's identity *and* its setup bytes, plus a
digest over the board list. `materialize_benchmark` rebuilds each board from
its id alone and refuses the manifest if a single setup byte differs — the
same executable-manifest discipline as Phase 15. This pack never changes;
extensions are a new version.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ...engine.constants import BLUE, RED
from ...belief.phase15.orientation import (
    ORIENTATION_RULE,
    ORIENTATION_RULE_VERSION,
    assert_engine_orientation,
    check_board,
)
from ...search.phase15.boards import Phase15BoardPlan, Phase15MatchSetupSources
from .contract import (
    BENCHMARK_BOARDS_PER_CELL,
    BENCHMARK_VERSION,
    DOMAIN_MATCH,
    DOMAIN_OPPONENT_SETUP,
    DOMAIN_PLAYER_SETUP,
    MATCH_COLORS,
    MATCH_FAMILY_KEYS,
    MATCH_LIBRARY_SPLIT,
    MATCH_OPPONENTS,
    MATCH_SETUP_SOURCES,
    Phase16MeasurementError,
    QUICK_SUBSET_NAME,
    QUICK_SUBSET_ORDINAL,
    SETUP_TARGETED,
    benchmark_board_id,
    derive_measure_seed,
    parse_benchmark_board_id,
)

#: Where the frozen manifest lives.
DEFAULT_MANIFEST_PATH = Path("data/phase16/phase16_benchmark_v1.json")

#: The construction identity.
BOARD_CONSTRUCTION_VERSION = "phase16_board_construction_v1"

#: (opponent x setup source x player colour), cell-major — the same cell
#: order convention as Phase 15, restated here so the Phase 16 pack identity
#: does not silently track a Phase 15 constant.
BENCHMARK_CELLS = tuple(
    (opponent, source, color)
    for opponent in MATCH_OPPONENTS
    for source in MATCH_SETUP_SOURCES
    for color in MATCH_COLORS
)

FAMILY_ANY = "any"

_PLAYER_OF = {"red": RED, "blue": BLUE}


def requested_family(setup_source: str, cell_index: int, ordinal: int) -> str:
    """The accepted-library family a targeted cell's board requests."""
    if setup_source != SETUP_TARGETED:
        return FAMILY_ANY
    return MATCH_FAMILY_KEYS[(int(cell_index) + int(ordinal)) % len(MATCH_FAMILY_KEYS)]


def benchmark_board_plan(
    opponent: str,
    setup_source: str,
    color: str,
    ordinal: int,
    sources: Phase15MatchSetupSources,
    *,
    library_split: str = MATCH_LIBRARY_SPLIT,
    cell_index: "int | None" = None,
) -> Phase15BoardPlan:
    """One complete benchmark board, drawn and orientation-checked.

    Refusal, not repair: an ungated board never reaches `create_game`.
    """
    if cell_index is None:
        try:
            cell_index = BENCHMARK_CELLS.index((opponent, setup_source, color))
        except ValueError:
            raise Phase16MeasurementError(
                f"({opponent!r}, {setup_source!r}, {color!r}) is not a benchmark cell"
            ) from None
    family = requested_family(setup_source, cell_index, ordinal)
    identifier = benchmark_board_id(opponent, setup_source, family, color, int(ordinal))
    opponent_color = "blue" if color == "red" else "red"
    requested = None if setup_source != SETUP_TARGETED else family

    player_draw = sources.draw(
        setup_source,
        library_split,
        color,
        derive_measure_seed(DOMAIN_PLAYER_SETUP, identifier),
        requested,
    )
    opponent_draw = sources.draw(
        setup_source,
        library_split,
        opponent_color,
        derive_measure_seed(DOMAIN_OPPONENT_SETUP, identifier),
        requested,
    )
    red_draw, blue_draw = (
        (player_draw, opponent_draw) if color == "red" else (opponent_draw, player_draw)
    )

    # The production-path assertions, plus the whole imported board gate.
    assert_engine_orientation(red_draw.canonical, red_draw.engine, RED)
    assert_engine_orientation(blue_draw.canonical, blue_draw.engine, BLUE)
    orientation = check_board(red_draw.canonical, blue_draw.canonical)

    return Phase15BoardPlan(
        board_id=identifier,
        opponent=opponent,
        setup_source=setup_source,
        requested_family=family,
        color=color,
        ordinal=int(ordinal),
        cell_index=int(cell_index),
        match_seed=derive_measure_seed(DOMAIN_MATCH, identifier),
        red_setup=red_draw.engine,
        blue_setup=blue_draw.engine,
        player_family_key=player_draw.family_key,
        opponent_family_key=opponent_draw.family_key,
        player_base_setup_id=player_draw.base_setup_id,
        opponent_base_setup_id=opponent_draw.base_setup_id,
        player_setup_branch=player_draw.branch,
        opponent_setup_branch=opponent_draw.branch,
        orientation=orientation,
    )


def benchmark_plans(
    *,
    sources: "Phase15MatchSetupSources | None" = None,
    boards_per_cell: int = BENCHMARK_BOARDS_PER_CELL,
    library_split: str = MATCH_LIBRARY_SPLIT,
) -> "list[Phase15BoardPlan]":
    """The whole pack, cell-major: every cell contributes the same count."""
    if int(boards_per_cell) < 1:
        raise Phase16MeasurementError(
            f"boards_per_cell must be positive, got {boards_per_cell!r}"
        )
    sources = Phase15MatchSetupSources() if sources is None else sources
    plans: list[Phase15BoardPlan] = []
    for cell_index, (opponent, source, color) in enumerate(BENCHMARK_CELLS):
        for ordinal in range(int(boards_per_cell)):
            plans.append(
                benchmark_board_plan(
                    opponent,
                    source,
                    color,
                    ordinal,
                    sources,
                    library_split=library_split,
                    cell_index=cell_index,
                )
            )
    return plans


def quick_subset_ids(manifest: dict) -> "list[str]":
    """The predeclared quick subset: ordinal 0 of every cell."""
    return [
        row["board_id"]
        for row in manifest["boards"]
        if int(row["ordinal"]) == QUICK_SUBSET_ORDINAL
    ]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def manifest_digest(payload: dict) -> str:
    """sha256 over the canonical JSON of the manifest's board list."""
    body = json.dumps(payload["boards"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def build_benchmark_manifest(
    plans: "list[Phase15BoardPlan]",
    *,
    generated_utc: str,
    library_split: str = MATCH_LIBRARY_SPLIT,
    sources: "Phase15MatchSetupSources | None" = None,
    **extra,
) -> dict:
    boards = []
    for plan in plans:
        row = plan.describe()
        row["red_setup"] = list(plan.red_setup)
        row["blue_setup"] = list(plan.blue_setup)
        boards.append(row)
    opponents: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    colors: dict[str, int] = {}
    families: dict[str, int] = {}
    for plan in plans:
        opponents[plan.opponent] = opponents.get(plan.opponent, 0) + 1
        source_counts[plan.setup_source] = source_counts.get(plan.setup_source, 0) + 1
        colors[plan.color] = colors.get(plan.color, 0) + 1
        families[plan.player_family_key] = families.get(plan.player_family_key, 0) + 1
    payload = {
        "artifact": BENCHMARK_VERSION,
        "board_construction_version": BOARD_CONSTRUCTION_VERSION,
        "generated_utc": generated_utc,
        "library_split": library_split,
        "orientation_rule_version": ORIENTATION_RULE_VERSION,
        "orientation_rule": ORIENTATION_RULE,
        "orientation_gate": (
            "stratego.belief.phase15.orientation.check_board, imported, per board"
        ),
        "seed_identity": "phase16_measurement_identity_v1 (strat-p16m)",
        "frozen": "this pack never changes; extensions get a new version",
        "quick_subset": {
            "name": QUICK_SUBSET_NAME,
            "rule": f"ordinal == {QUICK_SUBSET_ORDINAL} of every cell",
            "boards": sum(1 for plan in plans if plan.ordinal == QUICK_SUBSET_ORDINAL),
        },
        "boards": boards,
        "board_count": len(boards),
        "balance": {
            "by_opponent": dict(sorted(opponents.items())),
            "by_setup_source": dict(sorted(source_counts.items())),
            "by_color": dict(sorted(colors.items())),
            "by_player_family": dict(sorted(families.items())),
        },
        **extra,
    }
    if sources is not None:
        payload["setup_sources"] = sources.describe()
    payload["manifest_digest"] = manifest_digest(payload)
    return payload


def load_benchmark_manifest(path: "Path | str" = DEFAULT_MANIFEST_PATH, *, root: "Path | str" = ".") -> dict:
    full = Path(root) / Path(path)
    if not full.is_file():
        raise Phase16MeasurementError(f"no benchmark manifest at {full}")
    manifest = json.loads(full.read_text())
    if manifest.get("artifact") != BENCHMARK_VERSION:
        raise Phase16MeasurementError(f"{full} is not a {BENCHMARK_VERSION} document")
    observed = manifest_digest(manifest)
    if observed != manifest.get("manifest_digest"):
        raise Phase16MeasurementError(
            f"{full}: stored digest {manifest.get('manifest_digest')} != recomputed "
            f"{observed}; refusing a tampered manifest"
        )
    return manifest


def materialize_benchmark(
    manifest: dict,
    *,
    sources: "Phase15MatchSetupSources | None" = None,
    verify: bool = True,
    subset: "set[str] | None" = None,
) -> "list[Phase15BoardPlan]":
    """Rebuild every board of the manifest from its id, and check the bytes."""
    sources = Phase15MatchSetupSources() if sources is None else sources
    split = manifest.get("library_split", MATCH_LIBRARY_SPLIT)
    rebuilt = []
    for row in manifest["boards"]:
        if subset is not None and row["board_id"] not in subset:
            continue
        fields = parse_benchmark_board_id(row["board_id"])
        plan = benchmark_board_plan(
            fields["opponent"],
            fields["setup_source"],
            fields["color"],
            fields["ordinal"],
            sources,
            library_split=split,
            cell_index=int(row["cell_index"]),
        )
        if verify:
            if list(plan.red_setup) != list(row["red_setup"]):
                raise Phase16MeasurementError(
                    f"{plan.board_id}: rebuilt Red setup differs from the manifest"
                )
            if list(plan.blue_setup) != list(row["blue_setup"]):
                raise Phase16MeasurementError(
                    f"{plan.board_id}: rebuilt Blue setup differs from the manifest"
                )
        rebuilt.append(plan)
    return rebuilt


__all__ = [
    "BENCHMARK_CELLS",
    "BOARD_CONSTRUCTION_VERSION",
    "DEFAULT_MANIFEST_PATH",
    "FAMILY_ANY",
    "benchmark_board_plan",
    "benchmark_plans",
    "build_benchmark_manifest",
    "load_benchmark_manifest",
    "manifest_digest",
    "materialize_benchmark",
    "quick_subset_ids",
    "requested_family",
]
