"""Phase 15 Agent 1 section 6: the corpus setup mixture.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` section 6.

Three named sources, one orientation path
-----------------------------------------
```text
35%   neutral_v1                    the accepted Phase 7 baseline draw
45%   phase14_learned               the accepted P10-D production selector
20%   targeted_family               accepted library bases, family-targeted
```

Every one of them returns a :class:`Phase15SetupDraw` whose `engine` tuple
came out of :func:`~.orientation.oriented_for`, which re-derives the
placement from the engine's own `SETUP_SQUARES` and refuses anything else.
There is no code path in this module on which a canonical tuple can reach
`create_game`, which is the whole point of section 4.

Why `phase14_learned` is the P10-D selector object
--------------------------------------------------
`phase14_setup_source_v1` is defined as "35% neutral_v1 + 65% the accepted
P10-D learned selector"; that mixture coin lives *inside*
:class:`LearnedSetupSource`, so the Phase 14 source and the accepted P10-D
production source are the same draw distribution under two names. Phase 15
calls the selector directly so its own 35% neutral share stays a separate,
countable label rather than being folded invisibly into the learned share.
Each learned draw records which internal branch it took, so a report can
state the true neutral fraction of the finished corpus rather than the
intended one.

Partitions, and why rejection rather than a fork
-------------------------------------------------
Section 6 asks for non-overlapping validation identities for calibration
and development. Both draw from the accepted `validation` library split, so
the separation is made by partitioning that split's base setups in half —
per family, so neither half loses a family — and conditioning each split's
draws on its own half. The conditioning is done by *rejection*: the draw
runs through the accepted sampler exactly as before, and a draw whose base
falls in the other half is retried under a derived successor seed. Every
returned board is therefore a genuine accepted draw, and no accepted
sampler was forked to produce it.

Targeted families invent nothing
--------------------------------
A targeted draw picks a base *from a named accepted family inside the
requested library split*, then runs the accepted post-selection path —
`post_selection_decisions`, `perturbation_seed_for`, `build_descendant` —
which is byte-for-byte the same construction the selector and
`sample_setup` use. Only the base-choice rule differs, and family
membership is a property the accepted library already carries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...engine.constants import PLAYER_NAMES, PLAYERS
from ...setups.contracts import SPLITS
from ...setups.families import FAMILY_BY_KEY
from ...setups.sampler import build_descendant, load_library_index, sample_setup
from .contract import (
    LIBRARY_PARTITIONS,
    SETUP_LEARNED,
    SETUP_NEUTRAL,
    SETUP_SOURCES,
    SETUP_TARGETED,
    TARGETED_FAMILY_KEYS,
    Phase15Error,
)
from .orientation import oriented_for
from .seeds import DOMAIN_OBSERVER_SETUP, derive_phase15_seed

#: The identity of this mixture.
SETUP_MIXTURE_VERSION = "phase15_setup_mixture_v1"

_PLAYER_BY_COLOR = {PLAYER_NAMES[player]: player for player in PLAYERS}


class Phase15SetupError(Phase15Error):
    """A Phase 15 setup could not be drawn or oriented."""


@dataclass(frozen=True)
class Phase15SetupDraw:
    """One drawn setup, canonical and engine-ready, with its provenance."""

    source: str
    family_id: str
    family_key: str
    base_setup_id: str
    library_split: str
    color: str
    seed: int
    canonical: "tuple[int, ...]"
    engine: "tuple[int, ...]"
    branch: "str | None" = None
    provenance: dict = field(default_factory=dict)

    @property
    def player(self) -> int:
        return _PLAYER_BY_COLOR[self.color]


class Phase15SetupSources:
    """The three Phase 15 setup sources, loaded once and reused.

    Holds the frozen library index and the accepted P10-D selector. Every
    draw is a pure function of `(source, library_split, color, seed)`, so
    two processes that build this object independently produce identical
    boards.
    """

    #: How many times a partitioned draw may be retried before giving up.
    #: A half-sized partition needs two draws on average; 64 is a refusal
    #: threshold, not a working limit.
    MAX_PARTITION_RETRIES = 64

    def __init__(self, index=None, scorer=None) -> None:
        from ...training.phase10_selector import LearnedSetupSource, candidate, load_scorer
        from ...training.phase11_contract import ACCEPTED_SELECTOR_CANDIDATE_ID

        self.index = load_library_index() if index is None else index
        self.scorer = load_scorer() if scorer is None else scorer
        self.selector_candidate_id = ACCEPTED_SELECTOR_CANDIDATE_ID
        self.learned = LearnedSetupSource(
            candidate(ACCEPTED_SELECTOR_CANDIDATE_ID), self.scorer, self.index
        )
        self._targeted_families = tuple(
            FAMILY_BY_KEY[key].family_id for key in TARGETED_FAMILY_KEYS
        )
        self._partitions = build_partitions(self.index)

    # -- the three sources -------------------------------------------------

    def _neutral(self, library_split: str, color: str, seed: int, partition=None) -> Phase15SetupDraw:
        from ...training.phase10_selector import neutral_baseline_draw

        drawn = neutral_baseline_draw(library_split, int(seed), self.index)
        entry = self.index.base(drawn.base_setup_id)
        return self._finish(
            SETUP_NEUTRAL,
            drawn.canonical,
            entry,
            library_split,
            color,
            seed,
            branch=None,
            provenance=drawn.provenance,
        )

    def _learned(self, library_split: str, color: str, seed: int, partition=None) -> Phase15SetupDraw:
        from ...training.phase10_selector import SelectorRequest

        drawn = self.learned.draw(
            SelectorRequest(split=library_split, color=color, selector_seed=int(seed))
        )
        entry = self.index.base(drawn.base_setup_id)
        return self._finish(
            SETUP_LEARNED,
            drawn.setup.canonical,
            entry,
            library_split,
            color,
            seed,
            branch=drawn.branch,
            provenance=drawn.to_dict(),
        )

    def _targeted(self, library_split: str, color: str, seed: int, partition=None) -> Phase15SetupDraw:
        from ...training.phase10_selector import (
            perturbation_seed_for,
            post_selection_decisions,
        )
        from ...training.phase10_contract import NEUTRAL_PROFILE_NAME

        # The family is chosen from the targeted list by the draw identity,
        # then the base uniformly inside that family and split. Both use the
        # Phase 15 stream, so the choice is reproducible from the game id.
        family_id = self._targeted_families[
            derive_phase15_seed(DOMAIN_OBSERVER_SETUP, "targeted_family", int(seed))
            % len(self._targeted_families)
        ]
        eligible = self.index.eligible_bases(family_id, library_split)
        if partition is not None:
            allowed = self._partitions[library_split][partition]
            eligible = tuple(
                entry for entry in eligible if entry.base_setup_id in allowed
            )
        if not eligible:
            raise Phase15SetupError(
                f"family {family_id} has no base in library split {library_split!r}"
                + (f" partition {partition!r}" if partition else "")
            )
        entry = eligible[
            derive_phase15_seed(DOMAIN_OBSERVER_SETUP, "targeted_base", int(seed))
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
        return self._finish(
            SETUP_TARGETED,
            setup.canonical,
            entry,
            library_split,
            color,
            seed,
            branch=None,
            provenance=setup.provenance,
        )

    # -- the shared tail ---------------------------------------------------

    def _finish(
        self,
        source: str,
        canonical,
        entry,
        library_split: str,
        color: str,
        seed: int,
        *,
        branch,
        provenance: dict,
    ) -> Phase15SetupDraw:
        """Orient the draw and package it. The single exit of every source."""
        canonical = tuple(canonical)
        player = _PLAYER_BY_COLOR[color]
        engine = oriented_for(canonical, player)
        return Phase15SetupDraw(
            source=source,
            family_id=entry.family_id,
            family_key=entry.family_key,
            base_setup_id=entry.base_setup_id,
            library_split=library_split,
            color=color,
            seed=int(seed),
            canonical=canonical,
            engine=engine,
            branch=branch,
            provenance=dict(provenance),
        )

    def draw(
        self,
        source: str,
        library_split: str,
        color: str,
        seed: int,
        partition: "str | None" = None,
    ) -> Phase15SetupDraw:
        """One engine-ready setup from a named Phase 15 source.

        `partition` restricts the eligible base population to one half of
        the library split — the mechanism behind section 6's non-overlapping
        calibration/development identities. The targeted source filters its
        own candidate list; the other two are conditioned by rejection,
        because their base choice lives inside an accepted sampler that
        Phase 15 may not fork.
        """
        if source not in SETUP_SOURCES:
            raise Phase15SetupError(
                f"setup source must be one of {list(SETUP_SOURCES)}, got {source!r}"
            )
        if library_split not in SPLITS:
            raise Phase15SetupError(f"unknown library split {library_split!r}")
        if color not in _PLAYER_BY_COLOR:
            raise Phase15SetupError(f"unknown colour {color!r}")
        if partition is not None and partition not in LIBRARY_PARTITIONS:
            raise Phase15SetupError(
                f"partition must be one of {list(LIBRARY_PARTITIONS)}, got {partition!r}"
            )
        if source == SETUP_TARGETED:
            return self._targeted(library_split, color, seed, partition)

        method = self._neutral if source == SETUP_NEUTRAL else self._learned
        if partition is None:
            return method(library_split, color, seed)
        allowed = self._partitions[library_split][partition]
        attempt = int(seed)
        for retry in range(self.MAX_PARTITION_RETRIES):
            drawn = method(library_split, color, attempt, partition)
            if drawn.base_setup_id in allowed:
                return drawn
            attempt = derive_phase15_seed(
                DOMAIN_OBSERVER_SETUP, "partition_retry", int(seed), retry
            )
        raise Phase15SetupError(
            f"no {source!r} draw landed in partition {partition!r} of "
            f"{library_split!r} within {self.MAX_PARTITION_RETRIES} attempts"
        )

    def describe(self) -> dict:
        from ...training.phase10_selector import (
            LEARNED_SETUP_SOURCE_VERSION,
            SETUP_SELECTOR_VERSION,
        )

        return {
            "setup_mixture_version": SETUP_MIXTURE_VERSION,
            "library_content_digest": self.index.content_digest,
            "selector_candidate": self.selector_candidate_id,
            "selector_version": SETUP_SELECTOR_VERSION,
            "production_source_version": LEARNED_SETUP_SOURCE_VERSION,
            "targeted_families": {
                key: FAMILY_BY_KEY[key].family_id for key in TARGETED_FAMILY_KEYS
            },
            "orientation": "every draw exits through orientation.oriented_for(player)",
            "forbidden_glue": (
                "stratego/belief/phase11b/corpus.py Phase11BSetupSources.draw — "
                "returns canonical tuples and is never imported here"
            ),
        }


def build_partitions(index) -> dict:
    """Split every library split's bases in half, per family.

    Per family rather than globally, so neither half of `validation` loses a
    family and the section 6 targeted-family requirement still holds inside
    each of calibration and development. Membership is a pure function of
    the base id and the family, so two processes build the same partition.
    """
    from ...setups.contracts import SPLITS as LIBRARY_SPLITS
    from ...setups.families import FAMILY_IDS

    partitions: dict = {}
    for split in LIBRARY_SPLITS:
        halves = {name: set() for name in LIBRARY_PARTITIONS}
        for family_id in FAMILY_IDS:
            eligible = index.eligible_bases(family_id, split)
            ordered = sorted(
                eligible,
                # Base ids carry colons, which the seed payload reserves as
                # its own separator; the dotted form is the same identity.
                key=lambda entry: derive_phase15_seed(
                    DOMAIN_OBSERVER_SETUP,
                    "partition",
                    entry.base_setup_id.replace(":", "."),
                ),
            )
            middle = len(ordered) // 2
            halves[LIBRARY_PARTITIONS[0]].update(
                entry.base_setup_id for entry in ordered[:middle]
            )
            halves[LIBRARY_PARTITIONS[1]].update(
                entry.base_setup_id for entry in ordered[middle:]
            )
        partitions[split] = halves
    return partitions


def targeted_family_coverage(
    sources: "Phase15SetupSources",
    library_split: str,
    draws: int = 512,
    partition: "str | None" = None,
) -> dict:
    """How the targeted source spreads over its families, for the manifest."""
    counts: dict[str, int] = {}
    for ordinal in range(int(draws)):
        drawn = sources.draw(SETUP_TARGETED, library_split, "red", ordinal, partition)
        counts[drawn.family_key] = counts.get(drawn.family_key, 0) + 1
    missing = [key for key in TARGETED_FAMILY_KEYS if key not in counts]
    return {
        "draws": int(draws),
        "library_split": library_split,
        "counts": dict(sorted(counts.items())),
        "families_covered": len(counts),
        "families_required": len(TARGETED_FAMILY_KEYS),
        "missing": missing,
        "covers_all_required": not missing,
    }


__all__ = [
    "SETUP_MIXTURE_VERSION",
    "build_partitions",
    "Phase15SetupDraw",
    "Phase15SetupError",
    "Phase15SetupSources",
    "targeted_family_coverage",
]
