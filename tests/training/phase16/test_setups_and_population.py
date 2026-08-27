"""Phase 16 Agent 3: the two setup mixtures and the opponent population."""

import pytest

from stratego.engine.constants import BLUE, RED
from stratego.engine.state import create_game
from stratego.training.phase16 import contract as C
from stratego.training.phase16 import population as P
from stratego.training.phase16 import setups as S


def ids(arm="test_tiny", count=32):
    return [C.game_id(arm, index % 8, index) for index in range(count)]


# ---------------------------------------------------------------------------
# Setups
# ---------------------------------------------------------------------------


def test_the_library_mixture_is_the_accepted_source_on_a_phase16_seed(library_source):
    """The accepted Phase 10B object, with the seed derivation as the only change."""
    from stratego.training.phase10b_setup_source import Phase10BSetupSource

    assert isinstance(library_source, Phase10BSetupSource)
    identifier = C.game_id("test_tiny", 1, 1)
    ours = library_source.side_seed(game_id=identifier, player=RED)
    assert ours == S.side_seed(identifier, RED)
    # `assign` reaches the seed through `self.side_seed`, so the override is live:
    # the accepted parent would refuse this id outright.
    assert library_source.draw_for_player(game_id=identifier, player=RED)[1] == ours


def test_a_phase16_game_cannot_draw_a_phase14_board():
    """Structurally, not by convention: the id does not parse as a Phase 14 one."""
    from stratego.training.phase14_seed import (
        Phase14SeedError,
        side_selector_seed as phase14_side_seed,
    )

    identifier = C.game_id("test_tiny", 1, 1)
    with pytest.raises(Phase14SeedError):
        phase14_side_seed(identifier, "red")


def test_a_draw_is_a_pure_function_of_the_game_id(library_source):
    identifier = C.game_id("test_tiny", 2, 5)
    first = library_source.assign(root_seed=0, environment_id=0, generation=0, game_id=identifier)
    second = library_source.assign(root_seed=9, environment_id=3, generation=7, game_id=identifier)
    assert first.red_setup == second.red_setup
    assert first.blue_setup == second.blue_setup


def test_both_mixtures_orient_blue_rather_than_emitting_a_canonical_tuple(
    library_source, expanded_source
):
    for source in (library_source, expanded_source):
        for identifier in ids(count=6):
            report = S.assert_orientation_path(source, identifier)
            assert report["engine_is_oriented"]


def test_every_expanded_board_passes_the_accepted_pair_gate(expanded_source):
    census = S.orientation_census(expanded_source, ids(count=64))
    assert census["boards_checked"] == 64
    assert set(census["side_halves"]) <= {
        "red:library",
        "red:expanded",
        "blue:library",
        "blue:expanded",
    }


def test_the_expanded_mixture_is_about_half_adversarial_per_side(expanded_source):
    halves = {"library": 0, "expanded": 0}
    for identifier in ids(count=400):
        provenance = expanded_source.assign(
            root_seed=0, environment_id=0, generation=0, game_id=identifier
        ).provenance
        for half in provenance["phase16_side_family"].values():
            halves[half] += 1
    total = sum(halves.values())
    assert total == 800
    assert 0.42 < halves["expanded"] / total < 0.58


def test_the_adversarial_half_draws_only_authored_families(expanded_source):
    described = expanded_source.adversarial.describe()
    assert S.HARVEST_FAMILY in described["excluded"]
    assert S.HARVEST_FAMILY not in described["families"]
    assert described["setups"] == sum(described["families"].values())
    assert all(entry["family"] != S.HARVEST_FAMILY for entry in expanded_source.adversarial.entries)


def test_expanded_boards_reach_create_game(expanded_source):
    for identifier in ids(count=8):
        assignment = expanded_source.assign(
            root_seed=0, environment_id=0, generation=0, game_id=identifier
        )
        state = create_game(assignment.red_setup, assignment.blue_setup, game_id=identifier)
        assert not state.terminal


def test_the_expanded_provenance_records_what_it_drew(expanded_source):
    provenance = expanded_source.assign(
        root_seed=0, environment_id=0, generation=0, game_id=C.game_id("test_tiny", 0, 3)
    ).provenance
    assert provenance["provenance_schema_version"] == S.PHASE16_SETUP_PROVENANCE_VERSION
    assert provenance["phase16_mixture"] == C.SETUPS_EXPANDED
    assert provenance["adversarial_weight"] == C.EXPANDED_ADVERSARIAL_WEIGHT
    assert provenance["adversarial_library_digest"]
    assert set(provenance["sides"]) == {"red", "blue"}
    for side in provenance["sides"].values():
        assert side["half"] in (C.SETUPS_LIBRARY, C.SETUPS_EXPANDED)
        assert side["setup_id"]


def test_the_forbidden_phase11b_glue_is_not_imported():
    import stratego.training.phase16.setups as module

    source = module.__file__
    text = open(source).read()
    assert "Phase11BSetupSources" in text  # named, as the warning
    assert "from ...belief.phase11b" not in text
    assert "import phase11b" not in text


def test_build_setup_source_refuses_an_unknown_mixture():
    with pytest.raises(S.Phase16SetupError):
        S.build_setup_source("adversarial_only")


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def test_pure_current_puts_the_learner_on_both_colours():
    config = C.ARM_B
    for slot in range(8):
        draw = P.draw_for_slot(config, slot=slot, draw=slot * 3)
        assert draw.opponent_kind == P.KIND_CURRENT
        assert draw.learner_control == P.LEARNER_CONTROL_BOTH
        assert draw.learner_sides == ("red", "blue")
        assert draw.handcrafted_policy_id is None


def test_the_phase14_mixture_realizes_its_declared_shares():
    config = C.ARM_A
    pool = P.HistoricalPool("P24")
    draws = [
        P.draw_for_slot(config, slot=index % 96, draw=index // 96, pool=pool)
        for index in range(4000)
    ]
    shares = P.realized_shares(draws)["kind_shares"]
    assert shares[P.KIND_CURRENT] == pytest.approx(0.58, abs=0.03)
    assert shares[P.KIND_HISTORICAL] == pytest.approx(0.30, abs=0.03)
    assert shares[P.KIND_HANDCRAFTED] == pytest.approx(0.12, abs=0.02)


def test_a_draw_is_reproducible_from_the_run_state_alone():
    config = C.ARM_A
    pool = P.HistoricalPool("P24")
    first = P.draw_for_slot(config, slot=5, draw=11, pool=pool)
    second = P.draw_for_slot(config, slot=5, draw=11, pool=P.HistoricalPool("P24"))
    assert first == second


def test_a_non_current_opponent_takes_one_colour_and_owns_one_policy_seed():
    config = C.ARM_A
    pool = P.HistoricalPool("P24")
    handcrafted = [
        P.draw_for_slot(config, slot=index, draw=0, pool=pool)
        for index in range(400)
    ]
    for draw in handcrafted:
        if draw.opponent_kind == P.KIND_CURRENT:
            assert draw.red_policy_seed is None and draw.blue_policy_seed is None
            continue
        assert draw.learner_control in ("red", "blue")
        assert draw.learner_sides == (draw.learner_control,)
        if draw.opponent_kind == P.KIND_HANDCRAFTED:
            seeds = [draw.red_policy_seed, draw.blue_policy_seed]
            assert sum(seed is not None for seed in seeds) == 1
            # the seat with a seed is the one the learner does not hold
            held = draw.red_policy_seed if draw.learner_control == "blue" else draw.blue_policy_seed
            assert held is not None
        else:
            assert draw.red_policy_seed is None and draw.blue_policy_seed is None


def test_handcrafted_opponents_come_from_the_frozen_phase14_roster():
    from stratego.evaluation.registry import POLICY_INDEX

    assert set(P.HANDCRAFTED_ROSTER) <= set(POLICY_INDEX)
    assert sum(P.HANDCRAFTED_WEIGHTS) == pytest.approx(1.0)
    token = P.handcrafted_policy_token("strategic_rule_based")
    assert token.startswith("strategic_rule_based@")
    with pytest.raises(P.Phase16PopulationError):
        P.handcrafted_policy_token("no_such_policy")


def test_the_pool_starts_at_the_arms_own_past_and_refuses_a_stranger():
    pool = P.HistoricalPool("P24")
    assert pool.members() == ("P24",)
    assert pool.select(0.0) == "P24" and pool.select(0.999) == "P24"
    pool.add("W0010", path=None)
    assert pool.members() == ("P24", "W0010")
    assert pool.select(0.9) == "W0010"
    with pytest.raises(P.Phase16PopulationError):
        pool.add("W0010")
    with pytest.raises(P.Phase16PopulationError):
        pool.path_for("W9999")


def test_population_semantics_declares_the_window_unit():
    semantics = P.population_semantics(C.ARM_B)
    assert semantics["unit"].startswith("one window")
    assert semantics["declared_shares"] == {"current": 1.0}
    assert semantics["search"].startswith("absent")
    assert P.population_semantics(C.ARM_A)["declared_shares"] == C.PHASE14_MIXTURE_SHARES
