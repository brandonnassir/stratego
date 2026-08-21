"""Phase 14: the frozen `phase14_setup_source_v1` collection adapter.

Specification source: `reports/phase13/phase14_setup_source_v1.json` (FROZEN),
via `02_AGENT_2_FINAL_TRAINING_INTEGRATION.md` section 3.

What this adapter is
--------------------
The frozen source is "35% neutral_v1 + 65% the accepted P10-D learned selector,
accepted reflection/perturbation, train split, orientation through the accepted
oriented(player) path". That is *exactly* the accepted
:class:`~stratego.training.phase10b_setup_source.Phase10BSetupSource` object —
the mixture coin lives inside :class:`LearnedSetupSource` and the accepted
adapter already binds `SelectorDraw.oriented(player)` before `create_game`.
Phase 14 therefore subclasses it and changes exactly two things:

1. the per-side selector seeds descend from the Phase 14 roots, so a Phase 14
   game's setups are not the same boards a Phase 10B game with a similar id
   would have drawn;
2. the provenance and `setup_family` labels name Phase 14, so a stored game
   says which run produced it.

The orientation warning
-----------------------
Agent 1's document names the concrete mistake this module must not make:
`stratego/belief/phase11b/corpus.py`'s `Phase11BSetupSources.draw` returns
*canonical* tuples, and passing those to `create_game` for BLUE places the army
back-to-front — the root cause of Phase 12's "47/64 front-row flags"
observation. That glue is not imported here and must never be. The check is
mechanical rather than documentary: :func:`assert_orientation_path` re-derives
a draw through the accepted helper and refuses a source whose engine setup is
the canonical tuple for blue.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..engine.constants import BLUE, PLAYER_NAMES, PLAYERS
from .phase10_selector import (
    LEARNED_SETUP_SOURCE_VERSION,
    SETUP_SELECTOR_VERSION,
    SelectorRequest,
)
from .phase10b_setup_source import Phase10BSetupSource
from .phase14_contract import (
    PHASE14_CONTRACT_VERSION,
    SETUP_LEARNED_WEIGHT,
    SETUP_NEUTRAL_WEIGHT,
    SETUP_ORIENTATION_RULE,
    SETUP_SELECTOR_CANDIDATE,
    SETUP_SELECTOR_CONFIG_SHA256,
    SETUP_SOURCE_IDENTITY,
    SETUP_SPLIT,
)
from .phase14_seed import Phase14SeedError, side_selector_seed
from .setup_source import SetupSourceError

PHASE14_SETUP_SOURCE_VERSION = "phase14_setup_source_v1"

#: The `trajectory_v1` `setup_family` label of a Phase 14 rollout game.
SETUP_FAMILY = f"{PHASE14_SETUP_SOURCE_VERSION}_{SETUP_SELECTOR_CANDIDATE}_{SETUP_SPLIT}"


@dataclass
class Phase14SetupSource(Phase10BSetupSource):
    """`phase14_setup_source_v1` as a Phase 9-compatible collection source.

    Holds nothing mutable: two processes that build it independently produce
    identical assignments for identical game ids, because every draw is a pure
    function of the game id, the side and the frozen selector.
    """

    split: str = SETUP_SPLIT

    @classmethod
    def build(cls, *, index=None, scorer=None) -> "Phase14SetupSource":
        """The production Phase 14 setup source: the frozen P10-D selector."""
        built = Phase10BSetupSource.build(index=index, scorer=scorer)
        return cls(source=built.source, split=SETUP_SPLIT)

    # -- description -------------------------------------------------------

    @property
    def setup_family(self) -> str:
        return SETUP_FAMILY

    def describe(self) -> dict:
        described = dict(Phase10BSetupSource.describe(self))
        described.update(
            {
                "source_id": SETUP_FAMILY,
                "setup_source_version": PHASE14_SETUP_SOURCE_VERSION,
                "identity": SETUP_SOURCE_IDENTITY,
                "contract_version": PHASE14_CONTRACT_VERSION,
                "selector_config_sha256": SETUP_SELECTOR_CONFIG_SHA256,
                "mixture": {
                    "neutral": SETUP_NEUTRAL_WEIGHT,
                    "learned": SETUP_LEARNED_WEIGHT,
                },
                "orientation": SETUP_ORIENTATION_RULE,
                "side_seed_derivation": (
                    "phase14_seed.side_selector_seed(game_id, colour); red and blue "
                    "descend from two different frozen Phase 14 domains"
                ),
            }
        )
        return described

    # -- drawing -----------------------------------------------------------

    def side_seed(self, *, game_id: str, player: int) -> int:
        """The Phase 14 selector seed of one side of one logical game."""
        if player not in PLAYERS:
            raise SetupSourceError(f"unknown player: {player!r}")
        try:
            return side_selector_seed(game_id, PLAYER_NAMES[player])
        except Phase14SeedError as error:
            raise SetupSourceError(str(error)) from error

    def assign(self, **kwargs):
        """The accepted assignment, relabelled for Phase 14.

        The parent owns the draw, the orientation call and the provenance field
        set; the two version labels it hard-codes for Phase 10B are rewritten in
        place, exactly as the accepted Phase 10B collector rewrites the rollout
        version of a sidecar. Duplicating the whole adapter to change two
        strings would be the worse trade.
        """
        assignment = Phase10BSetupSource.assign(self, **kwargs)
        provenance = dict(assignment.provenance)
        provenance["provenance_schema_version"] = PHASE14_SETUP_SOURCE_VERSION
        provenance["setup_source_version"] = PHASE14_SETUP_SOURCE_VERSION
        provenance["setup_source_identity"] = SETUP_SOURCE_IDENTITY
        return type(assignment)(
            red_setup=assignment.red_setup,
            blue_setup=assignment.blue_setup,
            provenance=provenance,
        )


def validate_assignment_provenance(provenance: dict) -> list:
    """Every way one Phase 14 setup provenance record can be wrong.

    The accepted Phase 10B checks (candidate, split, per-side colour/seed/
    fingerprint agreement, two distinct seeds) plus the Phase 14 label check.
    """
    from .phase10b_setup_source import validate_assignment_provenance as accepted

    problems = list(accepted(provenance))
    if not provenance:
        return problems
    for field, expected in (
        ("provenance_schema_version", PHASE14_SETUP_SOURCE_VERSION),
        ("setup_source_version", PHASE14_SETUP_SOURCE_VERSION),
        ("setup_source_identity", SETUP_SOURCE_IDENTITY),
    ):
        if provenance.get(field) != expected:
            problems.append(
                f"provenance {field} is {provenance.get(field)!r}, not {expected!r}"
            )
    return problems


def assert_orientation_path(source: "Phase14SetupSource", game_id: str) -> dict:
    """Prove this source orients BLUE rather than emitting a canonical tuple.

    Draws blue's arrangement twice — once through the source's own `assign`,
    once through the accepted selector by hand — and requires the engine setup
    to be `draw.oriented(BLUE)` and *not* the canonical tuple. This is the
    Phase 11B defect expressed as a test the production path runs, so it cannot
    reappear silently in a later refactor.
    """
    assignment = source.assign(
        root_seed=0, environment_id=0, generation=0, game_id=game_id
    )
    seed = source.side_seed(game_id=game_id, player=BLUE)
    draw = source.source.draw(
        SelectorRequest(split=source.split, color="blue", selector_seed=seed)
    )
    canonical = tuple(draw.setup.canonical)
    oriented = tuple(draw.oriented(BLUE))
    engine = tuple(assignment.blue_setup)
    if engine != oriented:
        raise SetupSourceError(
            f"{game_id}: the blue engine setup is not SelectorDraw.oriented(BLUE); "
            "this is the Phase 11B mis-orientation and may not reach Phase 14"
        )
    return {
        "game_id": game_id,
        "selector_seed": int(seed),
        "engine_is_oriented": True,
        "canonical_differs_from_oriented": canonical != oriented,
        "orientation_helper": "SelectorDraw.oriented(player)",
    }


def setup_source_semantics() -> dict:
    return {
        "identity": SETUP_SOURCE_IDENTITY,
        "setup_source_version": PHASE14_SETUP_SOURCE_VERSION,
        "setup_family": SETUP_FAMILY,
        "selector_version": SETUP_SELECTOR_VERSION,
        "production_source_version": LEARNED_SETUP_SOURCE_VERSION,
        "candidate": SETUP_SELECTOR_CANDIDATE,
        "split": SETUP_SPLIT,
        "mixture": {"neutral": SETUP_NEUTRAL_WEIGHT, "learned": SETUP_LEARNED_WEIGHT},
        "selector_config_sha256": SETUP_SELECTOR_CONFIG_SHA256,
        "orientation": SETUP_ORIENTATION_RULE,
        "forbidden_glue": "stratego/belief/phase11b/corpus.py Phase11BSetupSources",
    }
