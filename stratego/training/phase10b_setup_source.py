"""Optional Phase 10B: P10-D as the setup source of both sides of every game.

Specification source: `OPTIONAL_PHASE_10B_SETUP_CONDITIONED_FINE_TUNING_AGENT.md`
section 6.

What this adapter is
--------------------
The whole scientific object of Phase 10B is "the accepted move policy trained
while the environment's setup distribution is the frozen P10-D one". This
module is the join: it presents the accepted
:class:`~stratego.training.phase10_selector.LearnedSetupSource` through the
setup-source interface the accepted Phase 9 collector already consumes, so
neither the selector nor the collector needs a Phase 10B branch inside it.

Information safety
------------------
A draw sees exactly `own colour`, the requested split, the selector identity
and its own selector seed — the accepted `phase10_setup_selector_v1` input
set, unchanged. The two sides are drawn from two different frozen seed
domains, so Red's draw is not a function of Blue's and neither is a function
of the opponent's setup, the hidden rank truth, an outcome prediction, the
current checkpoint's strength or the matchup identity. The adapter never
receives those things, which is a stronger statement than not using them.

Read-only
---------
Nothing here writes a Phase 10 artifact. The selector, its utility, its
scaler, its temperature, its mixture and the Phase 7 library are loaded and
verified, never rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..engine.constants import PLAYER_NAMES, PLAYERS
from .phase10_selector import (
    LEARNED_SETUP_SOURCE_VERSION,
    SETUP_SELECTOR_VERSION,
    LearnedSetupSource,
    SelectorRequest,
    candidate,
    load_library_index,
    load_scorer,
)
from .phase10b_contract import (
    LEARNED_MIXTURE_WEIGHT,
    NEUTRAL_MIXTURE_WEIGHT,
    PHASE10B_CONTRACT_VERSION,
    SELECTED_CANDIDATE_ID,
    SELECTOR_TEMPERATURE,
    SELECTOR_UTILITY_MODEL,
)
from .phase10b_seed import Phase10BSeedError, side_selector_seed
from .setup_source import SetupAssignment, SetupSourceError

#: Phase 10B trains on the Phase 7 train split, exactly as Phase 9 did. The
#: split is not a parameter: a held-out base can never reach a training game.
TRAINING_SPLIT = "train"

PHASE10B_SETUP_SOURCE_VERSION = "phase10b_setup_source_v1"

#: The `trajectory_v1` `setup_family` label of a Phase 10B rollout game.
SETUP_FAMILY = f"{PHASE10B_SETUP_SOURCE_VERSION}_{SELECTED_CANDIDATE_ID}_{TRAINING_SPLIT}"


@dataclass
class Phase10BSetupSource:
    """`P10-D on both sides` as a Phase 9-compatible collection setup source.

    Holds the accepted selector and nothing mutable: two processes that build
    it independently produce identical assignments for identical game ids,
    because every draw is a pure function of the game id, the side and the
    frozen selector.
    """

    source: LearnedSetupSource
    split: str = TRAINING_SPLIT

    @classmethod
    def build(cls, *, index=None, scorer=None) -> "Phase10BSetupSource":
        """The production Phase 10B setup source: the frozen P10-D selector."""
        selector = candidate(SELECTED_CANDIDATE_ID)
        if selector.utility_model != SELECTOR_UTILITY_MODEL:
            raise SetupSourceError(
                f"{SELECTED_CANDIDATE_ID} names utility model "
                f"{selector.utility_model!r}, expected {SELECTOR_UTILITY_MODEL!r}"
            )
        if float(selector.temperature) != float(SELECTOR_TEMPERATURE):
            raise SetupSourceError(
                f"{SELECTED_CANDIDATE_ID} names temperature {selector.temperature}, "
                f"expected {SELECTOR_TEMPERATURE}"
            )
        return cls(
            source=LearnedSetupSource(
                selector,
                load_scorer() if scorer is None else scorer,
                load_library_index() if index is None else index,
            )
        )

    # -- description -------------------------------------------------------

    @property
    def setup_family(self) -> str:
        return SETUP_FAMILY

    @property
    def selector_identity(self) -> str:
        return self.source.selector_identity

    def describe(self) -> dict:
        return {
            "source_id": SETUP_FAMILY,
            "setup_source_version": PHASE10B_SETUP_SOURCE_VERSION,
            "kind": "phase10_learned_selector",
            "selector_version": SETUP_SELECTOR_VERSION,
            "production_source_version": LEARNED_SETUP_SOURCE_VERSION,
            "candidate_id": SELECTED_CANDIDATE_ID,
            "selector_identity": self.selector_identity,
            "utility_model": SELECTOR_UTILITY_MODEL,
            "temperature": SELECTOR_TEMPERATURE,
            "mixture": {
                "neutral": NEUTRAL_MIXTURE_WEIGHT,
                "learned": LEARNED_MIXTURE_WEIGHT,
            },
            "split": self.split,
            "both_sides": True,
            "produces_provenance": True,
            "contract_version": PHASE10B_CONTRACT_VERSION,
            "side_seed_derivation": (
                "phase10b_seed.side_selector_seed(game_id, colour); red and blue "
                "descend from two different frozen domains"
            ),
            "allowed_inputs": ["own colour", "requested split", "selector seed"],
            "forbidden_inputs": [
                "opponent setup",
                "hidden rank truth",
                "outcome prediction",
                "checkpoint strength",
                "matchup identity",
            ],
        }

    # -- drawing -----------------------------------------------------------

    def side_seed(self, *, game_id: str, player: int) -> int:
        """The selector seed of one side of one logical game."""
        if player not in PLAYERS:
            raise SetupSourceError(f"unknown player: {player!r}")
        return side_selector_seed(game_id, PLAYER_NAMES[player])

    def draw_for_player(self, *, game_id: str, player: int):
        """`(SelectorDraw, selector_seed)` for one side of one logical game."""
        seed = self.side_seed(game_id=game_id, player=player)
        draw = self.source.draw(
            SelectorRequest(
                split=self.split, color=PLAYER_NAMES[player], selector_seed=seed
            )
        )
        return draw, seed

    def side_identity(self, *, game_id: str, player: int) -> str:
        """The per-side setup identity recorded in the schedule."""
        seed = self.side_seed(game_id=game_id, player=player)
        return (
            f"{self.selector_identity}|split={self.split}"
            f"|player={PLAYER_NAMES[player]}|selector_seed={seed}"
        )

    def assign(
        self,
        *,
        root_seed: int,
        environment_id: int,
        generation: int,
        slot_seed: int = 0,
        game_id: str = "",
    ) -> SetupAssignment:
        """Both setups of one logical game, with full two-layer provenance.

        The signature is the accepted `setup_source_v1` one so the collector
        needs no Phase 10B branch, but the draw identity descends from the
        game id rather than the root seed: that is what lets Red and Blue sit
        on two separate frozen domains while the trajectory still records the
        root seed the accepted store expects.
        """
        if not game_id:
            raise SetupSourceError(
                "a Phase 10B setup assignment is addressed by its logical game "
                "id; an empty id would make the draw unaddressable"
            )
        try:
            engine_setups = {}
            sides = {}
            for player in PLAYERS:
                draw, seed = self.draw_for_player(game_id=game_id, player=player)
                engine_setups[player] = draw.oriented(player)
                sides[player] = {
                    "player": int(player),
                    "player_name": PLAYER_NAMES[player],
                    "selector_seed": int(seed),
                    "selector": draw.selector_provenance(),
                    "setup_provenance": dict(draw.setup_provenance),
                }
        except Phase10BSeedError as error:
            raise SetupSourceError(str(error)) from error

        provenance = {
            "provenance_schema_version": PHASE10B_SETUP_SOURCE_VERSION,
            "setup_source_version": PHASE10B_SETUP_SOURCE_VERSION,
            "selector_version": SETUP_SELECTOR_VERSION,
            "production_source_version": LEARNED_SETUP_SOURCE_VERSION,
            "candidate_id": SELECTED_CANDIDATE_ID,
            "selector_identity": self.selector_identity,
            "split": self.split,
            "game_id": game_id,
            "environment_id": int(environment_id),
            "generation": int(generation),
            "root_seed": int(root_seed),
            "red": sides[PLAYERS[0]],
            "blue": sides[PLAYERS[1]],
        }
        return SetupAssignment(
            red_setup=engine_setups[PLAYERS[0]],
            blue_setup=engine_setups[PLAYERS[1]],
            provenance=provenance,
        )


def validate_assignment_provenance(provenance: dict) -> list:
    """Every way one Phase 10B setup provenance record can be wrong."""
    problems: list = []
    if provenance is None:
        return ["the setup source emitted no provenance"]
    for field in (
        "provenance_schema_version",
        "selector_identity",
        "candidate_id",
        "split",
        "game_id",
        "red",
        "blue",
    ):
        if field not in provenance:
            problems.append(f"provenance is missing {field!r}")
    if problems:
        return problems
    if provenance["candidate_id"] != SELECTED_CANDIDATE_ID:
        problems.append(
            f"provenance names candidate {provenance['candidate_id']!r}, not the "
            f"frozen {SELECTED_CANDIDATE_ID!r}"
        )
    if provenance["split"] != TRAINING_SPLIT:
        problems.append(
            f"provenance names split {provenance['split']!r}, not {TRAINING_SPLIT!r}"
        )
    for color in ("red", "blue"):
        side = provenance[color]
        selector = side.get("selector", {})
        if selector.get("color") != color:
            problems.append(f"{color} side records colour {selector.get('color')!r}")
        if selector.get("split") != TRAINING_SPLIT:
            problems.append(f"{color} side records split {selector.get('split')!r}")
        if int(selector.get("selector_seed", -1)) != int(side.get("selector_seed", -2)):
            problems.append(f"{color} side selector seed disagrees with its draw")
        if not selector.get("final_setup_fingerprint"):
            problems.append(f"{color} side records no final setup fingerprint")
    if (
        provenance["red"].get("selector", {}).get("selector_seed")
        == provenance["blue"].get("selector", {}).get("selector_seed")
    ):
        problems.append("both sides drew from the same selector seed")
    return problems


__all__ = [
    "PHASE10B_SETUP_SOURCE_VERSION",
    "SETUP_FAMILY",
    "TRAINING_SPLIT",
    "Phase10BSetupSource",
    "validate_assignment_provenance",
]
