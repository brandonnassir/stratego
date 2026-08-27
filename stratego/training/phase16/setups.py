"""Phase 16 Agent 3: the two setup mixtures.

Specification source: `03_AGENT_3_TRAINING_LOOP_V2.md` section 2.7.

```text
library    Phase 14's 35% neutral + 65% P10-D, train split
expanded   50% library + 50% drawn uniformly over Agent 1's
           phase16_adversarial_setups_v1 families
```

Why not the Phase 14 adapter itself
-----------------------------------
The accepted `Phase14SetupSource` *is* the library mixture, and the mixture
coin, the split and the orientation call all live inside the accepted objects
it wraps. Exactly one thing about it is wrong for Phase 16: its per-side
selector seeds descend from the Phase 14 roots, so a Phase 16 game would
replay the boards of the Phase 14 game with the same id. Overview section 6
puts Phase 16 seeds on `phase16.agent3`, so the accepted Phase 10B source is
subclassed here and the seed derivation -- and only the seed derivation -- is
Phase 16's own. The selector, the neutral/learned coin, the split and
`SelectorDraw.oriented(player)` are untouched.

Orientation
-----------
Every board this module hands to `create_game` has passed the accepted Phase 15
gate: the library half through the accepted `oriented(player)` path plus a
mechanical re-derivation, and the adversarial half through
`stratego.belief.phase15.orientation.oriented_for`, which is the same rule. The
concrete mistake being guarded against is `Phase11BSetupSources.draw` returning
*canonical* tuples: passing one of those to `create_game` for BLUE places the
army back-to-front, which is the root cause of Phase 12's "47/64 front-row
flags". That glue is not imported here and must never be.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ...belief.phase15.orientation import (
    ORIENTATION_RULE,
    ORIENTATION_RULE_VERSION,
    check_board,
    oriented_for,
)
from ...engine.constants import BLUE, PLAYER_NAMES, PLAYERS, RED
from ..phase10_selector import SelectorRequest
from ..phase10b_setup_source import Phase10BSetupSource
from ..setup_source import SetupAssignment, SetupSourceError
from .contract import (
    DOMAIN_SETUP_SIDE,
    EXPANDED_ADVERSARIAL_WEIGHT,
    SETUPS_EXPANDED,
    SETUPS_LIBRARY,
    Phase16TrainingError,
    derive_train_seed,
    uniform_from_seed,
)

PHASE16_SETUP_SOURCE_VERSION = "phase16_setup_source_v1"
PHASE16_SETUP_PROVENANCE_VERSION = "phase16_setup_provenance_v1"

#: Agent 1's authored pack, and the interim copy this agent builds if it is absent.
ADVERSARIAL_PACK_PATH = "data/phase16/phase16_adversarial_setups_v1.json"
INTERIM_PACK_ARTIFACT = "phase16_agent03_interim_families_v1"
INTERIM_PACK_PATH = "data/phase16/phase16_agent03_interim_families_v1.json"

#: `operator_harvest` is a live-growing family; the training draw uses the
#: authored families only, so an arm's distribution cannot change under it
#: mid-run because the operator played a game.
HARVEST_FAMILY = "operator_harvest"


class Phase16SetupError(Phase16TrainingError):
    """A Phase 16 setup draw is outside its contract."""


def side_seed(game_id: str, player: int) -> int:
    """The Phase 16 selector/draw seed of one side of one logical game."""
    if player not in PLAYERS:
        raise Phase16SetupError(f"unknown player: {player!r}")
    return derive_train_seed(DOMAIN_SETUP_SIDE, game_id, PLAYER_NAMES[player])


# ---------------------------------------------------------------------------
# The library half
# ---------------------------------------------------------------------------


@dataclass
class Phase16LibrarySetupSource(Phase10BSetupSource):
    """The accepted library mixture on the Phase 16 seed stream."""

    @classmethod
    def build(cls, *, index=None, scorer=None) -> "Phase16LibrarySetupSource":
        built = Phase10BSetupSource.build(index=index, scorer=scorer)
        return cls(source=built.source, split=built.split)

    @property
    def setup_family(self) -> str:
        return f"{PHASE16_SETUP_SOURCE_VERSION}_library"

    def side_seed(self, *, game_id: str, player: int) -> int:
        return side_seed(game_id, player)

    def assign(self, **kwargs) -> SetupAssignment:
        assignment = Phase10BSetupSource.assign(self, **kwargs)
        provenance = dict(assignment.provenance or {})
        provenance.update(
            {
                "provenance_schema_version": PHASE16_SETUP_PROVENANCE_VERSION,
                "setup_source_version": PHASE16_SETUP_SOURCE_VERSION,
                "phase16_mixture": SETUPS_LIBRARY,
                "phase16_side_family": {"red": SETUPS_LIBRARY, "blue": SETUPS_LIBRARY},
            }
        )
        return SetupAssignment(
            red_setup=assignment.red_setup,
            blue_setup=assignment.blue_setup,
            provenance=provenance,
        )


# ---------------------------------------------------------------------------
# The adversarial half
# ---------------------------------------------------------------------------


def load_adversarial_pack(root: "str | Path" = ".") -> dict:
    """Agent 1's authored pack, or this agent's interim copy, with its digest."""
    root = Path(root)
    for path, expected in (
        (root / ADVERSARIAL_PACK_PATH, "phase16_adversarial_setups_v1"),
        (root / INTERIM_PACK_PATH, INTERIM_PACK_ARTIFACT),
    ):
        if not path.is_file():
            continue
        document = json.loads(path.read_text())
        if document.get("artifact") != expected:
            raise Phase16SetupError(
                f"{path} carries artifact {document.get('artifact')!r}, expected "
                f"{expected!r}"
            )
        document["_path"] = str(path)
        document["_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return document
    raise Phase16SetupError(
        "no adversarial setup pack found; expected Agent 1's "
        f"{ADVERSARIAL_PACK_PATH} or this agent's interim {INTERIM_PACK_PATH}"
    )


@dataclass(frozen=True)
class AdversarialLibrary:
    """The authored adversarial boards, flattened into one ordered draw list."""

    artifact: str
    path: str
    file_sha256: str
    authored_digest: str
    library_digest: str
    entries: tuple = field(default_factory=tuple)
    families: tuple = field(default_factory=tuple)

    @classmethod
    def load(cls, root: "str | Path" = ".") -> "AdversarialLibrary":
        document = load_adversarial_pack(root)
        entries = []
        families = []
        for name in sorted(document.get("families", {})):
            if name == HARVEST_FAMILY:
                continue
            family = document["families"][name]
            setups = family.get("setups") or []
            if not setups:
                continue
            families.append(name)
            for setup in setups:
                canonical = tuple(int(value) for value in setup["canonical_setup"])
                entries.append(
                    {
                        "family": name,
                        "setup_id": str(setup["setup_id"]),
                        "ordinal": int(setup["ordinal"]),
                        "canonical": canonical,
                    }
                )
        if not entries:
            raise Phase16SetupError(f"{document['_path']} holds no authored setups")
        return cls(
            artifact=str(document["artifact"]),
            path=str(document["_path"]),
            file_sha256=str(document["_file_sha256"]),
            authored_digest=str(document.get("authored_digest", "")),
            library_digest=str(document.get("library_digest", "")),
            entries=tuple(entries),
            families=tuple(families),
        )

    def __len__(self) -> int:
        return len(self.entries)

    def draw(self, seed: int) -> dict:
        """One board, uniform over the authored entries, from a derived seed."""
        index = int(uniform_from_seed(seed) * len(self.entries))
        return self.entries[min(index, len(self.entries) - 1)]

    def describe(self) -> dict:
        counts: dict = {}
        for entry in self.entries:
            counts[entry["family"]] = counts.get(entry["family"], 0) + 1
        return {
            "artifact": self.artifact,
            "path": self.path,
            "file_sha256": self.file_sha256,
            "authored_digest": self.authored_digest,
            "library_digest": self.library_digest,
            "setups": len(self.entries),
            "families": dict(sorted(counts.items())),
            "excluded": [HARVEST_FAMILY],
            "draw": "uniform over the authored entries",
        }


# ---------------------------------------------------------------------------
# The expanded mixture
# ---------------------------------------------------------------------------


@dataclass
class Phase16ExpandedSetupSource:
    """50% library + 50% adversarial, drawn per side and gated per board."""

    library: Phase16LibrarySetupSource
    adversarial: AdversarialLibrary
    adversarial_weight: float = EXPANDED_ADVERSARIAL_WEIGHT

    @classmethod
    def build(
        cls, *, root: "str | Path" = ".", index=None, scorer=None
    ) -> "Phase16ExpandedSetupSource":
        return cls(
            library=Phase16LibrarySetupSource.build(index=index, scorer=scorer),
            adversarial=AdversarialLibrary.load(root),
        )

    @property
    def setup_family(self) -> str:
        return f"{PHASE16_SETUP_SOURCE_VERSION}_expanded"

    @property
    def split(self) -> str:
        return self.library.split

    def side_seed(self, *, game_id: str, player: int) -> int:
        return side_seed(game_id, player)

    def side_identity(self, *, game_id: str, player: int) -> str:
        return self.library.side_identity(game_id=game_id, player=player)

    def _side_choice(self, game_id: str, player: int) -> str:
        """Which half this side draws from. A pure function of (game, colour)."""
        seed = derive_train_seed(
            DOMAIN_SETUP_SIDE, "mixture", game_id, PLAYER_NAMES[player]
        )
        return (
            SETUPS_EXPANDED
            if uniform_from_seed(seed) < float(self.adversarial_weight)
            else SETUPS_LIBRARY
        )

    def draw_canonical(self, *, game_id: str, player: int) -> dict:
        """One side's canonical 40-tuple and where it came from."""
        choice = self._side_choice(game_id, player)
        if choice == SETUPS_LIBRARY:
            draw, seed = self.library.draw_for_player(game_id=game_id, player=player)
            return {
                "half": SETUPS_LIBRARY,
                "canonical": tuple(int(v) for v in draw.setup.canonical),
                "engine": tuple(int(v) for v in draw.oriented(player)),
                "selector_seed": int(seed),
                "setup_id": str(draw.base_setup_id),
                "family": str(draw.family_id),
                "branch": str(draw.branch),
                "fingerprint": str(draw.final_setup_fingerprint),
            }
        seed = derive_train_seed(
            DOMAIN_SETUP_SIDE, "adversarial", game_id, PLAYER_NAMES[player]
        )
        entry = self.adversarial.draw(seed)
        return {
            "half": SETUPS_EXPANDED,
            "canonical": entry["canonical"],
            "engine": tuple(int(v) for v in oriented_for(entry["canonical"], player)),
            "selector_seed": int(seed),
            "setup_id": entry["setup_id"],
            "family": entry["family"],
        }

    def assign(
        self,
        *,
        root_seed: int = 0,
        environment_id: int = 0,
        generation: int = 0,
        slot_seed: int = 0,
        game_id: str = "",
    ) -> SetupAssignment:
        if not game_id:
            raise SetupSourceError("the expanded setup source needs a game id")
        red = self.draw_canonical(game_id=game_id, player=RED)
        blue = self.draw_canonical(game_id=game_id, player=BLUE)
        # The accepted section-4 gate, on the pair, before either board can
        # reach `create_game`. Never re-derived here: `check_board` raises.
        gate = check_board(red["canonical"], blue["canonical"])
        provenance = {
            "provenance_schema_version": PHASE16_SETUP_PROVENANCE_VERSION,
            "setup_source_version": PHASE16_SETUP_SOURCE_VERSION,
            "phase16_mixture": SETUPS_EXPANDED,
            "adversarial_weight": float(self.adversarial_weight),
            "adversarial_library_digest": self.adversarial.library_digest,
            "adversarial_file_sha256": self.adversarial.file_sha256,
            "split": self.split,
            "game_id": str(game_id),
            "orientation_rule": ORIENTATION_RULE,
            "orientation_rule_version": ORIENTATION_RULE_VERSION,
            "orientation_gate": "stratego.belief.phase15.orientation.check_board",
            "gate_flags": {
                "red_flag_rank": gate.get("red", {}).get("flag_rank"),
                "blue_flag_rank": gate.get("blue", {}).get("flag_rank"),
            }
            if isinstance(gate, dict)
            else {},
            "phase16_side_family": {"red": red["half"], "blue": blue["half"]},
            "sides": {
                "red": {
                    "half": red["half"],
                    "setup_id": red["setup_id"],
                    "family": red["family"],
                    "selector_seed": red["selector_seed"],
                },
                "blue": {
                    "half": blue["half"],
                    "setup_id": blue["setup_id"],
                    "family": blue["family"],
                    "selector_seed": blue["selector_seed"],
                },
            },
        }
        return SetupAssignment(
            red_setup=red["engine"], blue_setup=blue["engine"], provenance=provenance
        )

    def describe(self) -> dict:
        return {
            "source_id": self.setup_family,
            "setup_source_version": PHASE16_SETUP_SOURCE_VERSION,
            "kind": "phase16_expanded_mixture",
            "mixture": {
                "library": 1.0 - float(self.adversarial_weight),
                "adversarial": float(self.adversarial_weight),
            },
            "drawn": "per side, independently",
            "library": self.library.describe(),
            "adversarial": self.adversarial.describe(),
            "orientation": ORIENTATION_RULE,
            "orientation_gate": "stratego.belief.phase15.orientation.check_board",
            "produces_provenance": True,
            "forbidden_glue": "stratego/belief/phase11b/corpus.py Phase11BSetupSources",
        }


# ---------------------------------------------------------------------------
# Selection and verification
# ---------------------------------------------------------------------------


def build_setup_source(mixture: str, *, root: "str | Path" = ".", index=None, scorer=None):
    """The setup source one arm's `setups` flag names."""
    if mixture == SETUPS_LIBRARY:
        return Phase16LibrarySetupSource.build(index=index, scorer=scorer)
    if mixture == SETUPS_EXPANDED:
        return Phase16ExpandedSetupSource.build(root=root, index=index, scorer=scorer)
    raise Phase16SetupError(f"unknown setup mixture: {mixture!r}")


def assert_orientation_path(source, game_id: str) -> dict:
    """Prove this source orients BLUE rather than emitting a canonical tuple.

    The Phase 14 check, on whichever mixture is in play: draw blue's board
    through `assign`, re-derive it by hand through the accepted helper, and
    require the engine setup to be the oriented tuple. Phase 12's
    mis-orientation expressed as something the production path runs.
    """
    assignment = source.assign(
        root_seed=0, environment_id=0, generation=0, game_id=game_id
    )
    engine = tuple(assignment.blue_setup)
    if isinstance(source, Phase16ExpandedSetupSource):
        drawn = source.draw_canonical(game_id=game_id, player=BLUE)
        canonical = drawn["canonical"]
        oriented = tuple(oriented_for(canonical, BLUE))
    else:
        seed = source.side_seed(game_id=game_id, player=BLUE)
        draw = source.source.draw(
            SelectorRequest(split=source.split, color="blue", selector_seed=seed)
        )
        canonical = tuple(draw.setup.canonical)
        oriented = tuple(draw.oriented(BLUE))
    if engine != oriented:
        raise Phase16SetupError(
            f"{game_id}: the blue engine setup is not the oriented board; this is "
            "the Phase 11B mis-orientation and may not reach Phase 16 training"
        )
    return {
        "game_id": game_id,
        "engine_is_oriented": True,
        "canonical_differs_from_oriented": canonical != oriented,
        "orientation_helper": "oriented(player) / oriented_for(canonical, player)",
    }


def orientation_census(source, game_ids) -> dict:
    """Run the accepted pair gate over many draws and report what was seen."""
    checked = 0
    halves: dict = {}
    for identifier in game_ids:
        assignment = source.assign(
            root_seed=0, environment_id=0, generation=0, game_id=identifier
        )
        provenance = assignment.provenance or {}
        for colour, half in (provenance.get("phase16_side_family") or {}).items():
            key = f"{colour}:{half}"
            halves[key] = halves.get(key, 0) + 1
        for setup in (assignment.red_setup, assignment.blue_setup):
            if len(setup) != 40:
                raise Phase16SetupError(
                    f"{identifier}: an engine setup has {len(setup)} entries, not 40"
                )
        checked += 1
    return {
        "boards_checked": checked,
        "side_halves": dict(sorted(halves.items())),
        "orientation_rule": ORIENTATION_RULE,
        "gate": "every expanded draw passed check_board inside assign()",
    }


def setup_semantics(source) -> dict:
    described = source.describe()
    described["phase16_setup_source_version"] = PHASE16_SETUP_SOURCE_VERSION
    described["side_seed_derivation"] = (
        "phase16.contract.derive_train_seed('setup_side', game_id, colour)"
    )
    return described


__all__ = [
    "ADVERSARIAL_PACK_PATH",
    "AdversarialLibrary",
    "INTERIM_PACK_ARTIFACT",
    "INTERIM_PACK_PATH",
    "PHASE16_SETUP_PROVENANCE_VERSION",
    "PHASE16_SETUP_SOURCE_VERSION",
    "Phase16ExpandedSetupSource",
    "Phase16LibrarySetupSource",
    "Phase16SetupError",
    "assert_orientation_path",
    "build_setup_source",
    "load_adversarial_pack",
    "orientation_census",
    "setup_semantics",
    "side_seed",
]
