"""Phase 15 Agent 1: stream identities and the section 6 setup mixture.

The corpus is reproducible only if every board is a pure function of its
game id, and the mixture is evidence only if the three named sources are
separable. These tests hold both, and hold the orientation guarantee at the
source's single exit.
"""

from __future__ import annotations

import collections

import pytest

from stratego.belief.phase15 import contract as C
from stratego.belief.phase15 import seeds as S
from stratego.belief.phase15 import setups as U
from stratego.belief.phase15.orientation import assert_engine_orientation
from stratego.engine.constants import BLUE, RED
from stratego.setups.families import FAMILY_BY_KEY


@pytest.fixture(scope="module")
def sources():
    return U.Phase15SetupSources()


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------


def test_game_id_round_trips_every_field():
    game_id = S.corpus_game_id(
        "calibration", "p24", "stress_miner_rush", "targeted_family", "blue", 4321
    )
    fields = S.parse_corpus_game_id(game_id)
    assert fields == {
        "phase15_master_seed": S.PHASE15_MASTER_SEED,
        "split": "calibration",
        "observer_model": "p24",
        "opponent": "stress_miner_rush",
        "setup_source": "targeted_family",
        "observer_color": "blue",
        "ordinal": 4321,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"split": "nope"},
        {"observer_model": "p99"},
        {"opponent": "not_a_policy"},
        {"setup_source": "invented"},
        {"observer_color": "green"},
        {"ordinal": -1},
        {"ordinal": S.MAX_GAME_ORDINAL_FORMAT + 1},
    ],
)
def test_malformed_game_ids_are_refused(kwargs):
    base = {
        "split": "train",
        "observer_model": "p18",
        "opponent": "p24",
        "setup_source": "neutral_v1",
        "observer_color": "red",
        "ordinal": 0,
    }
    with pytest.raises(C.Phase15Error):
        S.corpus_game_id(**{**base, **kwargs})


def test_seeds_are_deterministic_and_role_separated():
    game_id = S.corpus_game_id("train", "p18", "p24", "neutral_v1", "red", 7)
    assert S.setup_seed(game_id, S.ROLE_OBSERVER) == S.setup_seed(
        game_id, S.ROLE_OBSERVER
    )
    assert S.setup_seed(game_id, S.ROLE_OBSERVER) != S.setup_seed(
        game_id, S.ROLE_OPPONENT
    )
    assert S.match_seed(game_id) != S.setup_seed(game_id, S.ROLE_OBSERVER)


def test_the_split_is_inside_the_seed_so_splits_cannot_share_a_board():
    train = S.corpus_game_id("train", "p18", "p24", "neutral_v1", "red", 7)
    development = S.corpus_game_id("development", "p18", "p24", "neutral_v1", "red", 7)
    assert S.setup_seed(train, S.ROLE_OBSERVER) != S.setup_seed(
        development, S.ROLE_OBSERVER
    )


def test_phase15_streams_cannot_collide_with_phase11b_streams():
    from stratego.belief.phase11b.seeds import derive_phase11b_seed

    for ordinal in range(16):
        assert S.derive_phase15_seed(
            S.DOMAIN_MATCH, "x", ordinal
        ) != derive_phase11b_seed("corpus_match", "x", ordinal)


def test_seed_parts_refuse_a_colon_and_a_bool():
    with pytest.raises(C.Phase15Error):
        S.derive_phase15_seed(S.DOMAIN_MATCH, "a:b")
    with pytest.raises(C.Phase15Error):
        S.derive_phase15_seed(S.DOMAIN_MATCH, True)


# ---------------------------------------------------------------------------
# The setup sources
# ---------------------------------------------------------------------------


def test_every_source_returns_an_oriented_engine_setup(sources):
    for source in C.SETUP_SOURCES:
        for color, player in (("red", RED), ("blue", BLUE)):
            drawn = sources.draw(source, "train", color, 31)
            assert_engine_orientation(drawn.canonical, drawn.engine, player)


def test_blue_draws_are_not_their_own_canonical_tuple(sources):
    differing = 0
    for ordinal in range(32):
        drawn = sources.draw(C.SETUP_NEUTRAL, "train", "blue", ordinal)
        if drawn.engine != drawn.canonical:
            differing += 1
    # The old defect made these identical for every board. A handful of
    # rank-palindromic setups are genuinely their own orientation.
    assert differing >= 30


def test_draws_are_pure_functions_of_their_identity(sources):
    first = sources.draw(C.SETUP_LEARNED, "train", "red", 99)
    second = sources.draw(C.SETUP_LEARNED, "train", "red", 99)
    assert first.canonical == second.canonical
    assert first.engine == second.engine
    assert first.base_setup_id == second.base_setup_id


def test_the_learned_source_records_its_internal_branch(sources):
    branches = {
        sources.draw(C.SETUP_LEARNED, "train", "red", ordinal).branch
        for ordinal in range(64)
    }
    assert branches <= {"neutral", "learned"}
    assert len(branches) == 2, "the P10-D mixture coin should take both branches"


def test_targeted_draws_stay_inside_the_named_families(sources):
    for ordinal in range(96):
        drawn = sources.draw(C.SETUP_TARGETED, "train", "red", ordinal)
        assert drawn.family_key in C.TARGETED_FAMILY_KEYS


def test_every_required_targeted_family_is_reachable(sources):
    coverage = U.targeted_family_coverage(sources, "train", draws=400)
    assert coverage["covers_all_required"] is True
    assert coverage["missing"] == []
    assert set(coverage["counts"]) == set(C.TARGETED_FAMILY_KEYS)


def test_the_targeted_family_keys_are_accepted_library_families():
    for key in C.TARGETED_FAMILY_KEYS:
        assert key in FAMILY_BY_KEY


def test_targeted_draws_are_available_in_the_validation_split(sources):
    coverage = U.targeted_family_coverage(sources, "validation", draws=400)
    assert coverage["covers_all_required"] is True


def test_library_splits_do_not_share_base_setups(sources):
    train = {
        sources.draw(C.SETUP_NEUTRAL, "train", "red", ordinal).base_setup_id
        for ordinal in range(200)
    }
    validation = {
        sources.draw(C.SETUP_NEUTRAL, "validation", "red", ordinal).base_setup_id
        for ordinal in range(200)
    }
    assert train and validation
    assert not (train & validation)


def test_unknown_arguments_are_refused(sources):
    with pytest.raises(U.Phase15SetupError):
        sources.draw("invented_source", "train", "red", 0)
    with pytest.raises(U.Phase15SetupError):
        sources.draw(C.SETUP_NEUTRAL, "not_a_split", "red", 0)
    with pytest.raises(U.Phase15SetupError):
        sources.draw(C.SETUP_NEUTRAL, "train", "green", 0)


def test_describe_names_the_forbidden_glue(sources):
    described = sources.describe()
    assert "Phase11BSetupSources" in described["forbidden_glue"]
    assert described["selector_candidate"] == "P10-D"


# ---------------------------------------------------------------------------
# Section 6: non-overlapping validation identities
# ---------------------------------------------------------------------------


def test_the_validation_partitions_are_exactly_disjoint(sources):
    halves = sources._partitions["validation"]
    left = halves[C.PARTITION_CALIBRATION]
    right = halves[C.PARTITION_DEVELOPMENT]
    assert left and right
    assert not (left & right)
    assert len(left) + len(right) == 800


def test_each_partition_keeps_every_family(sources):
    from stratego.setups.families import FAMILY_IDS

    halves = sources._partitions["validation"]
    for name in C.LIBRARY_PARTITIONS:
        families = set()
        for family_id in FAMILY_IDS:
            eligible = sources.index.eligible_bases(family_id, "validation")
            if any(entry.base_setup_id in halves[name] for entry in eligible):
                families.add(family_id)
        assert families == set(FAMILY_IDS)


@pytest.mark.parametrize("source", list(C.SETUP_SOURCES))
def test_partitioned_draws_never_share_a_base_setup(sources, source):
    left = {
        sources.draw(source, "validation", "red", ordinal, C.PARTITION_CALIBRATION)
        .base_setup_id
        for ordinal in range(250)
    }
    right = {
        sources.draw(source, "validation", "red", ordinal, C.PARTITION_DEVELOPMENT)
        .base_setup_id
        for ordinal in range(250)
    }
    assert left and right
    assert not (left & right)


def test_a_partitioned_draw_is_still_deterministic(sources):
    first = sources.draw(
        C.SETUP_NEUTRAL, "validation", "blue", 77, C.PARTITION_DEVELOPMENT
    )
    second = sources.draw(
        C.SETUP_NEUTRAL, "validation", "blue", 77, C.PARTITION_DEVELOPMENT
    )
    assert first.canonical == second.canonical
    assert first.base_setup_id == second.base_setup_id


def test_a_partitioned_draw_is_still_oriented(sources):
    from stratego.belief.phase15.orientation import assert_engine_orientation

    for source in C.SETUP_SOURCES:
        drawn = sources.draw(source, "validation", "blue", 5, C.PARTITION_CALIBRATION)
        assert_engine_orientation(drawn.canonical, drawn.engine, BLUE)


def test_an_unknown_partition_is_refused(sources):
    with pytest.raises(U.Phase15SetupError):
        sources.draw(C.SETUP_NEUTRAL, "validation", "red", 0, "z")


def test_the_train_split_takes_no_partition():
    assert C.LIBRARY_PARTITION[C.SPLIT_TRAIN] is None
    assert C.LIBRARY_PARTITION[C.SPLIT_CALIBRATION] != C.LIBRARY_PARTITION[
        C.SPLIT_DEVELOPMENT
    ]


def test_each_partition_still_covers_every_targeted_family(sources):
    for name in C.LIBRARY_PARTITIONS:
        coverage = U.targeted_family_coverage(sources, "validation", 400, name)
        assert coverage["covers_all_required"] is True
