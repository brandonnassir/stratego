"""Phase 15 Agent 2 section 8: fresh, orientation-safe boards.

Specification source: `02_AGENT_2_SEARCH_IMPLEMENTATION.md` sections 8 and 12.

Why no Phase 12 board is reused
-------------------------------
The Phase 12 match packs were built by glue that handed a *canonical*
own-orientation tuple to `create_game()` for BLUE, which reversed Blue's
army and put a flag on the front row of 47 of 64 boards. Section 8 is
explicit that those boards may not be reused as new evidence. Every board in
this module is drawn afresh and leaves through
:func:`stratego.belief.phase15.orientation.oriented_for`, whose assertion
re-derives the placement from the engine's own `SETUP_SQUARES`; each board
then passes Agent 1's full :func:`check_board` gate — flag row, legal setup
rows, exact inventory, paired Red/Blue mirror — before it is allowed to
describe a game.

The design, and what each dimension is for
------------------------------------------
```text
opponent        10   section 12's list, from P18 to stress_chaos
setup source     3   neutral_v1 / phase14_learned / targeted_family
player colour    2   red / blue
```

which is 60 cells, balanced over colour and setup source by construction
rather than by a post-hoc count. The `targeted_family` cells carry section
12's family requirement: the family cycles with the cell index, so every one
of the ten named families appears an equal number of times.

Both seats of a board draw from the *same* source, so a board is a fair
fight between two draws of one distribution rather than a mismatch between
two.

Targeted families, through the accepted path
---------------------------------------------
Agent 1's targeted source picks its family from *its own* eight-family list.
Section 12 names ten. Rather than edit Agent 1's frozen list, this module
does its own base choice — uniformly inside the named family and library
split — and then runs the identical accepted post-selection path
(`post_selection_decisions`, `perturbation_seed_for`, `build_descendant`)
that the accepted selector and `sample_setup` both use. Only the base-choice
rule differs, and family membership is a property the accepted library
already carries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...engine.constants import BLUE, RED
from ...belief.phase15.orientation import assert_engine_orientation, check_board
from ...belief.phase15.setups import Phase15SetupDraw, Phase15SetupSources
from .contract import (
    DOMAIN_MATCH,
    DOMAIN_OPPONENT_SETUP,
    DOMAIN_PLAYER_SETUP,
    MATCH_COLORS,
    MATCH_FAMILY_KEYS,
    MATCH_LIBRARY_SPLIT,
    MATCH_OPPONENTS,
    MATCH_SETUP_SOURCES,
    MATCH_VERSION,
    SETUP_TARGETED,
    Phase15SearchError,
    board_id,
    derive_search_seed,
    parse_board_id,
)

#: The board-construction identity.
BOARD_VERSION = "phase15_board_construction_v1"

#: The family token a non-targeted board carries in its id. The *observed*
#: family of the draw is recorded on the plan either way.
FAMILY_ANY = "any"

_PLAYER_OF = {"red": RED, "blue": BLUE}
_COLOR_OF = {RED: "red", BLUE: "blue"}

#: (opponent x setup source x player colour), cell-major.
BOARD_CELLS = tuple(
    (opponent, source, color)
    for opponent in MATCH_OPPONENTS
    for source in MATCH_SETUP_SOURCES
    for color in MATCH_COLORS
)


class Phase15BoardError(Phase15SearchError):
    """A Phase 15 board could not be planned, drawn or oriented."""


# ---------------------------------------------------------------------------
# Setup sources
# ---------------------------------------------------------------------------


class Phase15MatchSetupSources:
    """The three setup sources, with a ten-family targeted draw.

    Wraps Agent 1's :class:`Phase15SetupSources` — used unchanged for
    `neutral_v1` and `phase14_learned` — and adds the family-specific
    targeted draw section 12 needs.
    """

    def __init__(self, sources: "Phase15SetupSources | None" = None) -> None:
        self.sources = Phase15SetupSources() if sources is None else sources
        self.index = self.sources.index

    def draw(
        self, source: str, library_split: str, color: str, seed: int, family_key: str = None
    ) -> Phase15SetupDraw:
        if source not in MATCH_SETUP_SOURCES:
            raise Phase15BoardError(
                f"setup source must be one of {list(MATCH_SETUP_SOURCES)}, got {source!r}"
            )
        if source != SETUP_TARGETED:
            if family_key is not None:
                raise Phase15BoardError(
                    f"{source!r} does not accept a requested family; only "
                    f"{SETUP_TARGETED!r} does"
                )
            return self.sources.draw(source, library_split, color, int(seed))
        if family_key not in MATCH_FAMILY_KEYS:
            raise Phase15BoardError(
                f"targeted family must be one of {list(MATCH_FAMILY_KEYS)}, got "
                f"{family_key!r}"
            )
        return self._targeted(library_split, color, int(seed), family_key)

    def _targeted(
        self, library_split: str, color: str, seed: int, family_key: str
    ) -> Phase15SetupDraw:
        from ...setups.families import FAMILY_BY_KEY
        from ...setups.sampler import build_descendant
        from ...training.phase10_contract import NEUTRAL_PROFILE_NAME
        from ...training.phase10_selector import (
            perturbation_seed_for,
            post_selection_decisions,
        )

        family_id = FAMILY_BY_KEY[family_key].family_id
        eligible = self.index.eligible_bases(family_id, library_split)
        if not eligible:
            raise Phase15BoardError(
                f"family {family_key} ({family_id}) has no base in library split "
                f"{library_split!r}"
            )
        entry = eligible[
            derive_search_seed(DOMAIN_PLAYER_SETUP, "targeted_base", family_key, int(seed))
            % len(eligible)
        ]
        decisions = post_selection_decisions(library_split, int(seed))
        perturbation_seed = (
            perturbation_seed_for(
                library_split, int(seed), entry.base_setup_id, decisions.swap_count
            )
            if decisions.perturbation_requested
            else None
        )
        setup = build_descendant(
            entry,
            reflection_applied=decisions.reflection_applied,
            perturbation_requested=decisions.perturbation_requested,
            perturbation_seed=perturbation_seed,
            profile_name=NEUTRAL_PROFILE_NAME,
            draw_seed=int(seed),
        )
        return self.sources._finish(
            SETUP_TARGETED,
            setup.canonical,
            entry,
            library_split,
            color,
            int(seed),
            branch=None,
            provenance=dict(setup.provenance),
        )

    def describe(self) -> dict:
        report = self.sources.describe()
        report.update(
            {
                "board_version": BOARD_VERSION,
                "match_families": list(MATCH_FAMILY_KEYS),
                "targeted_draw": (
                    "base chosen uniformly inside the named accepted family and "
                    "library split, then the accepted post_selection_decisions / "
                    "perturbation_seed_for / build_descendant path, unchanged"
                ),
            }
        )
        return report


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase15BoardPlan:
    """One fresh match board, resolved from its identity alone."""

    board_id: str
    opponent: str
    setup_source: str
    requested_family: str
    color: str
    ordinal: int
    cell_index: int
    match_seed: int
    red_setup: "tuple[int, ...]"
    blue_setup: "tuple[int, ...]"
    player_family_key: str
    opponent_family_key: str
    player_base_setup_id: str
    opponent_base_setup_id: str
    player_setup_branch: "str | None"
    opponent_setup_branch: "str | None"
    orientation: dict = field(default_factory=dict, repr=False)

    @property
    def player(self) -> int:
        return _PLAYER_OF[self.color]

    @property
    def opponent_player(self) -> int:
        return BLUE if self.player == RED else RED

    def describe(self) -> dict:
        return {
            "board_id": self.board_id,
            "opponent": self.opponent,
            "setup_source": self.setup_source,
            "requested_family": self.requested_family,
            "player_color": self.color,
            "ordinal": self.ordinal,
            "cell_index": self.cell_index,
            "match_seed": self.match_seed,
            "player_family_key": self.player_family_key,
            "opponent_family_key": self.opponent_family_key,
            "player_base_setup_id": self.player_base_setup_id,
            "opponent_base_setup_id": self.opponent_base_setup_id,
            "player_setup_branch": self.player_setup_branch,
            "opponent_setup_branch": self.opponent_setup_branch,
            "red_flag_row": self.orientation.get("red", {}).get("flag", {}).get("row"),
            "blue_flag_row": self.orientation.get("blue", {}).get("flag", {}).get("row"),
        }


def requested_family(setup_source: str, cell_index: int, ordinal: int) -> str:
    """The family a cell's board targets.

    Non-targeted sources request nothing. Targeted cells cycle the ten
    section 12 families with the cell index *and* the ordinal, so a pack of
    any size covers the families as evenly as its size allows.
    """
    if setup_source != SETUP_TARGETED:
        return FAMILY_ANY
    return MATCH_FAMILY_KEYS[(int(cell_index) + int(ordinal)) % len(MATCH_FAMILY_KEYS)]


def board_plan(
    opponent: str,
    setup_source: str,
    color: str,
    ordinal: int,
    sources: Phase15MatchSetupSources,
    *,
    library_split: str = MATCH_LIBRARY_SPLIT,
    cell_index: "int | None" = None,
) -> Phase15BoardPlan:
    """One complete board, drawn and orientation-checked.

    The two setups are drawn from independent streams, oriented through the
    accepted helper, and then put through Agent 1's whole section 4 gate.
    Refusal, not repair: an ungated board never reaches `create_game`.
    """
    if cell_index is None:
        try:
            cell_index = BOARD_CELLS.index((opponent, setup_source, color))
        except ValueError:
            raise Phase15BoardError(
                f"({opponent!r}, {setup_source!r}, {color!r}) is not a board cell"
            ) from None
    family = requested_family(setup_source, cell_index, ordinal)
    identifier = board_id(opponent, setup_source, family, color, int(ordinal))
    opponent_color = "blue" if color == "red" else "red"
    requested = None if setup_source != SETUP_TARGETED else family

    player_draw = sources.draw(
        setup_source,
        library_split,
        color,
        derive_search_seed(DOMAIN_PLAYER_SETUP, identifier),
        requested,
    )
    opponent_draw = sources.draw(
        setup_source,
        library_split,
        opponent_color,
        derive_search_seed(DOMAIN_OPPONENT_SETUP, identifier),
        requested,
    )
    red_draw, blue_draw = (
        (player_draw, opponent_draw) if color == "red" else (opponent_draw, player_draw)
    )

    # The production-path assertions. `oriented_for` already ran inside each
    # source; these are the explicit second check on the exact tuples that
    # reach `create_game`, plus Agent 1's whole board gate.
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
        match_seed=derive_search_seed(DOMAIN_MATCH, identifier),
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


def board_plans(
    boards_per_cell: int = 1,
    *,
    sources: "Phase15MatchSetupSources | None" = None,
    library_split: str = MATCH_LIBRARY_SPLIT,
    opponents=MATCH_OPPONENTS,
    cells=None,
) -> "list[Phase15BoardPlan]":
    """The whole pack, cell-major: every cell contributes the same count."""
    if int(boards_per_cell) < 1:
        raise Phase15BoardError(
            f"boards_per_cell must be positive, got {boards_per_cell!r}"
        )
    sources = Phase15MatchSetupSources() if sources is None else sources
    selected = (
        [cell for cell in BOARD_CELLS if cell[0] in set(opponents)]
        if cells is None
        else list(cells)
    )
    plans: list[Phase15BoardPlan] = []
    for cell in selected:
        opponent, setup_source, color = cell
        cell_index = BOARD_CELLS.index(cell)
        for ordinal in range(int(boards_per_cell)):
            plans.append(
                board_plan(
                    opponent,
                    setup_source,
                    color,
                    ordinal,
                    sources,
                    library_split=library_split,
                    cell_index=cell_index,
                )
            )
    return plans


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


def manifest_digest(payload: dict) -> str:
    """sha256 over the canonical JSON of a manifest's board list."""
    import hashlib
    import json

    body = json.dumps(payload["boards"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def build_manifest(
    plans: "list[Phase15BoardPlan]",
    *,
    generated_utc: str,
    library_split: str = MATCH_LIBRARY_SPLIT,
    sources: "Phase15MatchSetupSources | None" = None,
    **extra,
) -> dict:
    """The `agent_02_match_manifest.json` document."""
    from ...belief.phase15.orientation import ORIENTATION_RULE, ORIENTATION_RULE_VERSION

    boards = []
    for plan in plans:
        row = plan.describe()
        row["red_setup"] = list(plan.red_setup)
        row["blue_setup"] = list(plan.blue_setup)
        boards.append(row)
    opponents: dict[str, int] = {}
    families: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    colors: dict[str, int] = {}
    for plan in plans:
        opponents[plan.opponent] = opponents.get(plan.opponent, 0) + 1
        families[plan.player_family_key] = families.get(plan.player_family_key, 0) + 1
        source_counts[plan.setup_source] = source_counts.get(plan.setup_source, 0) + 1
        colors[plan.color] = colors.get(plan.color, 0) + 1
    payload = {
        "artifact": MATCH_VERSION,
        "board_version": BOARD_VERSION,
        "generated_utc": generated_utc,
        "library_split": library_split,
        "orientation_rule_version": ORIENTATION_RULE_VERSION,
        "orientation_rule": ORIENTATION_RULE,
        "orientation_gate": "stratego.belief.phase15.orientation.check_board, per board",
        "supersedes": (
            "the Phase 12 match packs, whose Blue setups reached create_game "
            "un-oriented; no Phase 12 board is reused as Phase 15 evidence"
        ),
        "boards": boards,
        "board_count": len(boards),
        "balance": {
            "by_opponent": dict(sorted(opponents.items())),
            "by_player_family": dict(sorted(families.items())),
            "by_setup_source": dict(sorted(source_counts.items())),
            "by_color": dict(sorted(colors.items())),
        },
        "requested_families": list(MATCH_FAMILY_KEYS),
        **extra,
    }
    if sources is not None:
        payload["setup_sources"] = sources.describe()
    payload["manifest_digest"] = manifest_digest(payload)
    return payload


def materialize_manifest(
    manifest: dict, *, sources: "Phase15MatchSetupSources | None" = None, verify: bool = True
) -> "list[Phase15BoardPlan]":
    """Rebuild every board of a manifest from its id, and check the bytes.

    A manifest stores the setups it played, but the plans are also a pure
    function of the board ids; rebuilding and comparing is what makes the
    stored pack reproducible rather than merely recorded.
    """
    sources = Phase15MatchSetupSources() if sources is None else sources
    rebuilt = []
    for row in manifest["boards"]:
        fields = parse_board_id(row["board_id"])
        plan = board_plan(
            fields["opponent"],
            fields["setup_source"],
            fields["color"],
            fields["ordinal"],
            sources,
            library_split=manifest.get("library_split", MATCH_LIBRARY_SPLIT),
            cell_index=int(row["cell_index"]),
        )
        if verify:
            if list(plan.red_setup) != list(row["red_setup"]):
                raise Phase15BoardError(
                    f"{plan.board_id}: rebuilt Red setup differs from the manifest"
                )
            if list(plan.blue_setup) != list(row["blue_setup"]):
                raise Phase15BoardError(
                    f"{plan.board_id}: rebuilt Blue setup differs from the manifest"
                )
        rebuilt.append(plan)
    return rebuilt


__all__ = [
    "BOARD_CELLS",
    "BOARD_VERSION",
    "FAMILY_ANY",
    "Phase15BoardError",
    "Phase15BoardPlan",
    "Phase15MatchSetupSources",
    "board_plan",
    "board_plans",
    "build_manifest",
    "manifest_digest",
    "materialize_manifest",
    "requested_family",
]
