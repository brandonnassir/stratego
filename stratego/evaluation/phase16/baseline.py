"""Phase 16 Agent 1 section 5: the adversarial baseline measurement.

Three arms on 96 paired board triples, the selected system (`p24_b24`) in
the player seat, at TINY and MEDIUM:

```text
arm 1  benchmark_control     opponent setup drawn from the accepted library
arm 2  adversarial_opponent  opponent setup drawn from phase16_adversarial_setups_v1
arm 3  adversarial_both      both setups drawn from the pack (secondary)
```

Paired by construction
----------------------
A *pair* is one triple of boards sharing everything the arms are not about:
the machine opponent, the player's colour, the player's own setup (arms 1
and 2), the match seed, and therefore every rule-opponent decision stream.
Arm 2 minus arm 1 on one pair is the effect of swapping the opponent's army
from the accepted distribution to the adversarial one, on one board.

The mapping from pair index to cell is fixed:

```text
family      = AUTHORED_FAMILIES[pair // 12]      (8 families x 12 = 96)
setup j     = pair % 12                          (the family's j-th entry)
opponent    = MATCH_OPPONENTS[(j + family_index) % 10]
player col  = red if j even else blue
player src  = MATCH_SETUP_SOURCES[pair % 3]
arm 3 player setup = the family's entry (j + 6) % 12  (never equal to j)
```

Every board — control and adversarial alike — passes the imported Phase 15
section-4 gate before it may describe a game.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ...engine.constants import BLUE, RED
from ...belief.phase15.orientation import (
    ORIENTATION_RULE,
    ORIENTATION_RULE_VERSION,
    check_board,
)
from ...search.phase15.boards import Phase15BoardPlan, Phase15MatchSetupSources
from .adversarial import library_entry, load_library
from .contract import (
    ADVERSARIAL_BASELINE_VERSION,
    ARM_ADVERSARIAL_BOTH,
    ARM_ADVERSARIAL_OPPONENT,
    ARM_CONTROL,
    AUTHORED_FAMILIES,
    BASELINE_ARMS,
    DOMAIN_MATCH,
    DOMAIN_OPPONENT_SETUP,
    DOMAIN_PLAYER_SETUP,
    MATCH_FAMILY_KEYS,
    MATCH_LIBRARY_SPLIT,
    MATCH_OPPONENTS,
    MATCH_SETUP_SOURCES,
    Phase16MeasurementError,
    SETUP_TARGETED,
    SETUPS_PER_FAMILY,
    adversarial_board_id,
    derive_measure_seed,
    parse_adversarial_board_id,
)

#: Where the frozen baseline manifest lives.
DEFAULT_MANIFEST_PATH = Path("data/phase16/phase16_adversarial_baseline_v1.json")

#: 8 authored families x 12 setups.
PAIR_COUNT = len(AUTHORED_FAMILIES) * SETUPS_PER_FAMILY

#: The arm-3 player setup is the same family's entry at this fixed offset,
#: which can never equal the opponent's entry because the offset is not 0
#: modulo `SETUPS_PER_FAMILY`.
ARM3_PLAYER_OFFSET = 6
assert ARM3_PLAYER_OFFSET % SETUPS_PER_FAMILY != 0

_PLAYER_OF = {"red": RED, "blue": BLUE}


def pair_cell(pair_index: int) -> dict:
    """The fixed identity of one pair."""
    if not 0 <= int(pair_index) < PAIR_COUNT:
        raise Phase16MeasurementError(
            f"pair index must be in 0..{PAIR_COUNT - 1}, got {pair_index!r}"
        )
    pair_index = int(pair_index)
    family_index, setup_ordinal = divmod(pair_index, SETUPS_PER_FAMILY)
    family = AUTHORED_FAMILIES[family_index]
    opponent = MATCH_OPPONENTS[(setup_ordinal + family_index) % len(MATCH_OPPONENTS)]
    color = "red" if setup_ordinal % 2 == 0 else "blue"
    source = MATCH_SETUP_SOURCES[pair_index % len(MATCH_SETUP_SOURCES)]
    targeted_family = (
        MATCH_FAMILY_KEYS[pair_index % len(MATCH_FAMILY_KEYS)]
        if source == SETUP_TARGETED
        else None
    )
    return {
        "pair_index": pair_index,
        "family": family,
        "family_index": family_index,
        "setup_ordinal": setup_ordinal,
        "arm3_player_ordinal": (setup_ordinal + ARM3_PLAYER_OFFSET) % SETUPS_PER_FAMILY,
        "opponent": opponent,
        "color": color,
        "library_source": source,
        "targeted_family": targeted_family,
    }


def _library_draw(sources: Phase15MatchSetupSources, cell: dict, color: str, domain: str, token: str):
    return sources.draw(
        cell["library_source"],
        MATCH_LIBRARY_SPLIT,
        color,
        derive_measure_seed(domain, token, cell["pair_index"]),
        cell["targeted_family"],
    )


def baseline_board_plan(
    arm: str,
    pair_index: int,
    library: dict,
    sources: Phase15MatchSetupSources,
) -> Phase15BoardPlan:
    """One arm's board of one pair, gated before it may describe a game."""
    if arm not in BASELINE_ARMS:
        raise Phase16MeasurementError(
            f"arm must be one of {list(BASELINE_ARMS)}, got {arm!r}"
        )
    cell = pair_cell(pair_index)
    color = cell["color"]
    opponent_color = "blue" if color == "red" else "red"
    identifier = adversarial_board_id(
        arm, cell["family"], cell["opponent"], color, cell["pair_index"]
    )

    # The player's own setup: the accepted library for arms 1-2, the same
    # adversarial family (different entry) for arm 3.
    if arm == ARM_ADVERSARIAL_BOTH:
        player_entry = library_entry(
            library, cell["family"], cell["arm3_player_ordinal"]
        )
        player_canonical = tuple(player_entry["canonical_setup"])
        player_family_key = cell["family"]
        player_base_id = player_entry["setup_id"]
        player_branch = "phase16_adversarial"
    else:
        player_draw = _library_draw(
            sources, cell, color, DOMAIN_PLAYER_SETUP, "advpair"
        )
        player_canonical = tuple(player_draw.canonical)
        player_family_key = player_draw.family_key
        player_base_id = player_draw.base_setup_id
        player_branch = player_draw.branch

    # The opponent's setup: the accepted library in the control arm, the
    # pair's adversarial entry in arms 2-3.
    if arm == ARM_CONTROL:
        opponent_draw = _library_draw(
            sources, cell, opponent_color, DOMAIN_OPPONENT_SETUP, "advpair_control"
        )
        opponent_canonical = tuple(opponent_draw.canonical)
        opponent_family_key = opponent_draw.family_key
        opponent_base_id = opponent_draw.base_setup_id
        opponent_branch = opponent_draw.branch
    else:
        opponent_entry = library_entry(library, cell["family"], cell["setup_ordinal"])
        opponent_canonical = tuple(opponent_entry["canonical_setup"])
        opponent_family_key = cell["family"]
        opponent_base_id = opponent_entry["setup_id"]
        opponent_branch = "phase16_adversarial"

    red_canonical, blue_canonical = (
        (player_canonical, opponent_canonical)
        if color == "red"
        else (opponent_canonical, player_canonical)
    )
    # The whole imported section-4 gate, which also produces the oriented
    # engine tuples through the accepted `oriented_for` path.
    orientation = check_board(red_canonical, blue_canonical)
    from ...belief.phase15.orientation import oriented_for

    red_setup = oriented_for(red_canonical, RED)
    blue_setup = oriented_for(blue_canonical, BLUE)

    return Phase15BoardPlan(
        board_id=identifier,
        opponent=cell["opponent"],
        setup_source=arm,
        requested_family=cell["family"],
        color=color,
        ordinal=cell["pair_index"],
        cell_index=cell["pair_index"],
        match_seed=derive_measure_seed(DOMAIN_MATCH, "advpair", cell["pair_index"]),
        red_setup=red_setup,
        blue_setup=blue_setup,
        player_family_key=player_family_key,
        opponent_family_key=opponent_family_key,
        player_base_setup_id=player_base_id,
        opponent_base_setup_id=opponent_base_id,
        player_setup_branch=player_branch,
        opponent_setup_branch=opponent_branch,
        orientation=orientation,
    )


def baseline_plans(
    *,
    arms=BASELINE_ARMS,
    library: "dict | None" = None,
    sources: "Phase15MatchSetupSources | None" = None,
    root: "Path | str" = ".",
) -> "list[Phase15BoardPlan]":
    """Every requested arm's board of every pair, pair-major."""
    library = load_library(root=root) if library is None else library
    sources = Phase15MatchSetupSources() if sources is None else sources
    plans = []
    for pair_index in range(PAIR_COUNT):
        for arm in arms:
            plans.append(baseline_board_plan(arm, pair_index, library, sources))
    return plans


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def manifest_digest(payload: dict) -> str:
    body = json.dumps(payload["boards"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def build_baseline_manifest(
    plans: "list[Phase15BoardPlan]",
    *,
    generated_utc: str,
    library_digest: str,
    **extra,
) -> dict:
    boards = []
    for plan in plans:
        row = plan.describe()
        row["red_setup"] = list(plan.red_setup)
        row["blue_setup"] = list(plan.blue_setup)
        boards.append(row)
    by_arm: dict[str, int] = {}
    by_family: dict[str, int] = {}
    by_opponent: dict[str, int] = {}
    by_color: dict[str, int] = {}
    for plan in plans:
        by_arm[plan.setup_source] = by_arm.get(plan.setup_source, 0) + 1
        by_family[plan.requested_family] = by_family.get(plan.requested_family, 0) + 1
        by_opponent[plan.opponent] = by_opponent.get(plan.opponent, 0) + 1
        by_color[plan.color] = by_color.get(plan.color, 0) + 1
    payload = {
        "artifact": ADVERSARIAL_BASELINE_VERSION,
        "generated_utc": generated_utc,
        "adversarial_library_digest": library_digest,
        "orientation_rule_version": ORIENTATION_RULE_VERSION,
        "orientation_rule": ORIENTATION_RULE,
        "pair_count": PAIR_COUNT,
        "arm3_player_offset": ARM3_PLAYER_OFFSET,
        "pairing": (
            "arms of one pair share the machine opponent, the player's colour, "
            "the match seed, and (arms 1-2) the player's own setup; only the "
            "army the arm is about differs"
        ),
        "boards": boards,
        "board_count": len(boards),
        "balance": {
            "by_arm": dict(sorted(by_arm.items())),
            "by_family": dict(sorted(by_family.items())),
            "by_opponent": dict(sorted(by_opponent.items())),
            "by_color": dict(sorted(by_color.items())),
        },
        **extra,
    }
    payload["manifest_digest"] = manifest_digest(payload)
    return payload


def load_baseline_manifest(
    path: "Path | str" = DEFAULT_MANIFEST_PATH, *, root: "Path | str" = "."
) -> dict:
    full = Path(root) / Path(path)
    if not full.is_file():
        raise Phase16MeasurementError(f"no baseline manifest at {full}")
    manifest = json.loads(full.read_text())
    if manifest.get("artifact") != ADVERSARIAL_BASELINE_VERSION:
        raise Phase16MeasurementError(
            f"{full} is not a {ADVERSARIAL_BASELINE_VERSION} document"
        )
    observed = manifest_digest(manifest)
    if observed != manifest.get("manifest_digest"):
        raise Phase16MeasurementError(
            f"{full}: stored digest {manifest.get('manifest_digest')} != recomputed "
            f"{observed}; refusing a tampered manifest"
        )
    return manifest


def materialize_baseline(
    manifest: dict,
    *,
    library: "dict | None" = None,
    sources: "Phase15MatchSetupSources | None" = None,
    root: "Path | str" = ".",
    verify: bool = True,
    subset: "set[str] | None" = None,
) -> "list[Phase15BoardPlan]":
    library = load_library(root=root) if library is None else library
    sources = Phase15MatchSetupSources() if sources is None else sources
    rebuilt = []
    for row in manifest["boards"]:
        if subset is not None and row["board_id"] not in subset:
            continue
        fields = parse_adversarial_board_id(row["board_id"])
        plan = baseline_board_plan(fields["arm"], fields["pair_index"], library, sources)
        if verify:
            if list(plan.red_setup) != list(row["red_setup"]) or list(
                plan.blue_setup
            ) != list(row["blue_setup"]):
                raise Phase16MeasurementError(
                    f"{plan.board_id}: rebuilt setups differ from the manifest"
                )
        rebuilt.append(plan)
    return rebuilt


__all__ = [
    "ARM3_PLAYER_OFFSET",
    "DEFAULT_MANIFEST_PATH",
    "PAIR_COUNT",
    "baseline_board_plan",
    "baseline_plans",
    "build_baseline_manifest",
    "load_baseline_manifest",
    "manifest_digest",
    "materialize_baseline",
    "pair_cell",
]
