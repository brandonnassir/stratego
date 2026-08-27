"""The Phase 15 search contract and the fresh board construction."""

import pytest

from stratego.search.phase15 import contract
from stratego.search.phase15.boards import (
    BOARD_CELLS,
    FAMILY_ANY,
    Phase15BoardError,
    board_plans,
    build_manifest,
    materialize_manifest,
    requested_family,
)
from stratego.search.phase15.contract import (
    ALL_PROVIDERS,
    COMBINED_PAIRING_IDS,
    MATCH_FAMILY_KEYS,
    MATCH_OPPONENTS,
    MATCH_SETUP_SOURCES,
    PRODUCTION_PROVIDERS,
    Phase15SearchError,
    board_id,
    check_production_provider,
    derive_search_seed,
    pairing,
    parse_board_id,
    preset,
    strong_preset,
)


# -- identities -------------------------------------------------------------


def test_search_version_is_the_accepted_phase12_algorithm():
    from stratego.search.phase12.contract import SEARCH_VERSION

    assert contract.PHASE15_SEARCH_VERSION == SEARCH_VERSION == "phase12_root_world_search_v1"


def test_presets_are_the_accepted_phase12_objects():
    from stratego.search.phase12.contract import SEARCH_PRESETS as accepted

    for name, config in accepted.items():
        assert contract.SEARCH_PRESETS[name] is config


def test_score_constants_are_not_restated():
    from stratego.search.phase12.contract import BETA_DEFAULT, EPSILON_DEFAULT

    assert contract.BETA_DEFAULT is BETA_DEFAULT
    assert contract.EPSILON_DEFAULT is EPSILON_DEFAULT
    assert preset("TINY").beta == 0.1


# -- providers and pairings -------------------------------------------------


def test_oracle_is_not_a_production_provider():
    assert "oracle" in ALL_PROVIDERS
    assert "oracle" not in PRODUCTION_PROVIDERS
    assert contract.ORACLE_AVAILABLE_IN_PRODUCTION is False
    with pytest.raises(Phase15SearchError, match="offline diagnostic"):
        check_production_provider("oracle")


def test_every_production_provider_passes_the_check():
    for name in PRODUCTION_PROVIDERS:
        assert check_production_provider(name) == name


def test_unknown_provider_is_refused():
    with pytest.raises(Phase15SearchError, match="unknown belief provider"):
        check_production_provider("b99")


def test_the_four_combined_systems_exist_and_cross_pair():
    assert COMBINED_PAIRING_IDS == ("p18_b18", "p18_b24", "p24_b18", "p24_b24")
    for pairing_id in COMBINED_PAIRING_IDS:
        entry = pairing(pairing_id)
        assert entry.kind == "search"
        assert entry.is_learned
    assert pairing("p18_b24").move_model == "p18"
    assert pairing("p18_b24").provider == "b24"
    assert contract.PROVIDER_BACKBONE["b24"] == "p24"


def test_no_production_pairing_names_the_oracle():
    for pairing_id in contract.PRODUCTION_PAIRING_IDS:
        assert pairing(pairing_id).provider != "oracle"
    assert contract.DIAGNOSTIC_PAIRING_IDS == ("p18_oracle", "p24_oracle")
    for pairing_id in contract.DIAGNOSTIC_PAIRING_IDS:
        assert pairing(pairing_id).kind == "diagnostic"


def test_a_search_pairing_may_not_carry_the_oracle():
    with pytest.raises(Phase15SearchError, match="never a production search arm"):
        contract.Pairing("bad", "p18", "oracle", "search", "-")


def test_unknown_pairing_is_refused():
    with pytest.raises(Phase15SearchError, match="unknown pairing"):
        pairing("p18_b99")


# -- presets ----------------------------------------------------------------


def test_strong_preset_is_gated_to_the_instructed_depth_range():
    assert strong_preset(10).rollout_depth == 10
    assert strong_preset(12).rollout_depth == 12
    assert strong_preset(11).worlds == 64
    assert strong_preset(11).max_root_candidates == 12
    for bad in (9, 13, "10", True):
        with pytest.raises(Phase15SearchError, match="STRONG depth"):
            strong_preset(bad)


def test_unknown_preset_is_refused():
    with pytest.raises(Phase15SearchError, match="unknown preset"):
        preset("HUGE")


# -- seeds ------------------------------------------------------------------


def test_seed_streams_are_deterministic_and_separated():
    left = derive_search_seed(contract.DOMAIN_WORLDS, "board", 4)
    assert left == derive_search_seed(contract.DOMAIN_WORLDS, "board", 4)
    assert left != derive_search_seed(contract.DOMAIN_MATCH, "board", 4)
    assert left != derive_search_seed(contract.DOMAIN_WORLDS, "board", 5)
    assert 0 <= left < 2**63


def test_phase15_search_streams_never_equal_phase12_streams():
    from stratego.search.phase12.contract import derive_phase12_seed

    assert contract.PHASE15_SEARCH_PERSON != b"strat-p12"
    assert derive_search_seed(contract.DOMAIN_WORLDS, "x", 1) != derive_phase12_seed(
        "count_worlds", "x", 1
    )


def test_seed_parts_reject_the_reserved_separator():
    with pytest.raises(Phase15SearchError, match="may not contain"):
        derive_search_seed(contract.DOMAIN_MATCH, "a:b")


# -- board ids --------------------------------------------------------------


def test_board_ids_round_trip():
    identifier = board_id("p24", "targeted_family", "miner_forward", "blue", 7)
    fields = parse_board_id(identifier)
    assert fields == {
        "opponent": "p24",
        "setup_source": "targeted_family",
        "family_key": "miner_forward",
        "color": "blue",
        "ordinal": 7,
    }


@pytest.mark.parametrize(
    "args",
    [
        ("nobody", "neutral_v1", "any", "red", 0),
        ("p18", "nosuch", "any", "red", 0),
        ("p18", "neutral_v1", "any", "green", 0),
        ("p18", "neutral_v1", "any", "red", 1000),
    ],
)
def test_malformed_board_identities_are_refused(args):
    with pytest.raises(Phase15SearchError):
        board_id(*args)


def test_parse_refuses_a_foreign_identifier():
    with pytest.raises(Phase15SearchError, match="malformed"):
        parse_board_id("phase12_match_test_v1|st=x")


# -- the pack ---------------------------------------------------------------


def test_the_cell_grid_is_the_section_12_design():
    assert len(BOARD_CELLS) == len(MATCH_OPPONENTS) * len(MATCH_SETUP_SOURCES) * 2
    assert len(MATCH_OPPONENTS) == 10
    assert len(MATCH_FAMILY_KEYS) == 10


def test_requested_family_cycles_only_for_targeted_cells():
    assert requested_family("neutral_v1", 3, 0) == FAMILY_ANY
    assert requested_family("phase14_learned", 3, 1) == FAMILY_ANY
    seen = {requested_family("targeted_family", index, 0) for index in range(10)}
    assert seen == set(MATCH_FAMILY_KEYS)


def test_pack_is_balanced_and_covers_every_named_family(setup_sources):
    plans = board_plans(1, sources=setup_sources)
    assert len(plans) == len(BOARD_CELLS)
    colors = {}
    sources = {}
    requested = {}
    for plan in plans:
        colors[plan.color] = colors.get(plan.color, 0) + 1
        sources[plan.setup_source] = sources.get(plan.setup_source, 0) + 1
        if plan.requested_family != FAMILY_ANY:
            requested[plan.requested_family] = requested.get(plan.requested_family, 0) + 1
    assert colors == {"red": 30, "blue": 30}
    assert set(sources.values()) == {20}
    assert set(requested) == set(MATCH_FAMILY_KEYS)
    assert set(requested.values()) == {2}


def test_a_targeted_board_actually_lands_in_its_family(setup_sources):
    for plan in board_plans(1, sources=setup_sources):
        if plan.requested_family != FAMILY_ANY:
            assert plan.player_family_key == plan.requested_family
            assert plan.opponent_family_key == plan.requested_family


def test_boards_are_a_pure_function_of_their_identity(setup_sources):
    first = board_plans(1, sources=setup_sources)
    second = board_plans(1, sources=setup_sources)
    assert [plan.red_setup for plan in first] == [plan.red_setup for plan in second]
    assert [plan.blue_setup for plan in first] == [plan.blue_setup for plan in second]


def test_every_board_passes_the_orientation_gate(setup_sources):
    from stratego.engine.constants import BLUE, RED
    from stratego.belief.phase15.orientation import assert_engine_orientation

    plans = board_plans(1, sources=setup_sources)
    for plan in plans:
        assert plan.orientation["paired_mirror"] is True
        assert plan.orientation["red"]["inventory_exact"] is True
        assert plan.orientation["blue"]["inventory_exact"] is True


def test_blue_flags_sit_on_the_back_ranks_not_the_front(setup_sources):
    """The defect this whole construction exists to prevent.

    Under the Phase 11B glue 77% of Blue armies showed a front-row (engine
    row 6) flag. A correctly oriented pack shows one only when the canonical
    setup put the flag on the front canonical rank, which is rare.
    """
    plans = board_plans(1, sources=setup_sources)
    front = sum(1 for plan in plans if plan.orientation["blue"]["flag"]["row"] == 6)
    assert front / len(plans) < 0.15


def test_manifest_round_trips_through_materialize(setup_sources):
    plans = board_plans(1, sources=setup_sources)
    manifest = build_manifest(plans, generated_utc="2026-01-01T00:00:00Z", sources=setup_sources)
    assert manifest["board_count"] == len(plans)
    assert len(manifest["manifest_digest"]) == 64
    rebuilt = materialize_manifest(manifest, sources=setup_sources, verify=True)
    assert [plan.board_id for plan in rebuilt] == [plan.board_id for plan in plans]


def test_materialize_refuses_a_tampered_manifest(setup_sources):
    plans = board_plans(1, sources=setup_sources)
    manifest = build_manifest(plans, generated_utc="2026-01-01T00:00:00Z")
    manifest["boards"][0]["red_setup"] = list(reversed(manifest["boards"][0]["red_setup"]))
    with pytest.raises(Phase15BoardError, match="differs from the manifest"):
        materialize_manifest(manifest, sources=setup_sources, verify=True)


def test_a_non_targeted_source_refuses_a_requested_family(setup_sources):
    with pytest.raises(Phase15BoardError, match="does not accept a requested family"):
        setup_sources.draw("neutral_v1", "validation", "red", 1, "miner_forward")


def test_targeted_source_refuses_an_unnamed_family(setup_sources):
    with pytest.raises(Phase15BoardError, match="targeted family must be one of"):
        setup_sources.draw("targeted_family", "validation", "red", 1, "not_a_family")
