"""The adversarial baseline pack: pairing discipline and manifests."""

import json

import pytest

from stratego.evaluation.phase16.adversarial import library_entry
from stratego.evaluation.phase16.baseline import (
    ARM3_PLAYER_OFFSET,
    PAIR_COUNT,
    baseline_board_plan,
    build_baseline_manifest,
    load_baseline_manifest,
    materialize_baseline,
    pair_cell,
)
from stratego.evaluation.phase16.contract import (
    ARM_ADVERSARIAL_BOTH,
    ARM_ADVERSARIAL_OPPONENT,
    ARM_CONTROL,
    AUTHORED_FAMILIES,
    MATCH_OPPONENTS,
    Phase16MeasurementError,
    SETUPS_PER_FAMILY,
)


class TestPairCells:
    def test_pair_count(self):
        assert PAIR_COUNT == 96

    def test_family_coverage(self):
        counts: dict = {}
        for pair in range(PAIR_COUNT):
            cell = pair_cell(pair)
            counts[cell["family"]] = counts.get(cell["family"], 0) + 1
        assert counts == {family: SETUPS_PER_FAMILY for family in AUTHORED_FAMILIES}

    def test_opponent_coverage_within_family(self):
        for family_index in range(len(AUTHORED_FAMILIES)):
            opponents = {
                pair_cell(family_index * SETUPS_PER_FAMILY + j)["opponent"]
                for j in range(SETUPS_PER_FAMILY)
            }
            assert opponents == set(MATCH_OPPONENTS)

    def test_colors_balanced_within_family(self):
        for family_index in range(len(AUTHORED_FAMILIES)):
            colors = [
                pair_cell(family_index * SETUPS_PER_FAMILY + j)["color"]
                for j in range(SETUPS_PER_FAMILY)
            ]
            assert colors.count("red") == colors.count("blue") == 6

    def test_arm3_player_setup_never_the_opponents(self):
        for pair in range(PAIR_COUNT):
            cell = pair_cell(pair)
            assert cell["arm3_player_ordinal"] != cell["setup_ordinal"]
        assert ARM3_PLAYER_OFFSET % SETUPS_PER_FAMILY != 0

    def test_out_of_range_refused(self):
        with pytest.raises(Phase16MeasurementError):
            pair_cell(PAIR_COUNT)


@pytest.fixture(scope="module")
def triple(setup_sources, library_document):
    pair = 25
    return pair, {
        arm: baseline_board_plan(arm, pair, library_document, setup_sources)
        for arm in (ARM_CONTROL, ARM_ADVERSARIAL_OPPONENT, ARM_ADVERSARIAL_BOTH)
    }


class TestPairing:
    def test_shared_identity(self, triple):
        pair, plans = triple
        seeds = {plan.match_seed for plan in plans.values()}
        assert len(seeds) == 1
        opponents = {plan.opponent for plan in plans.values()}
        assert len(opponents) == 1
        colors = {plan.color for plan in plans.values()}
        assert len(colors) == 1
        assert {plan.ordinal for plan in plans.values()} == {pair}

    def test_player_setup_shared_by_arms_1_and_2(self, triple):
        _, plans = triple
        control, adversarial = plans[ARM_CONTROL], plans[ARM_ADVERSARIAL_OPPONENT]
        player = control.color
        if player == "red":
            assert control.red_setup == adversarial.red_setup
            assert control.blue_setup != adversarial.blue_setup
        else:
            assert control.blue_setup == adversarial.blue_setup
            assert control.red_setup != adversarial.red_setup

    def test_arm2_opponent_army_is_the_library_entry(
        self, triple, library_document
    ):
        pair, plans = triple
        cell = pair_cell(pair)
        from stratego.belief.phase15.orientation import oriented_for
        from stratego.engine.constants import BLUE, RED

        entry = library_entry(library_document, cell["family"], cell["setup_ordinal"])
        plan = plans[ARM_ADVERSARIAL_OPPONENT]
        opponent_player = BLUE if plan.color == "red" else RED
        expected = oriented_for(tuple(entry["canonical_setup"]), opponent_player)
        observed = plan.blue_setup if plan.color == "red" else plan.red_setup
        assert tuple(observed) == tuple(expected)

    def test_arm3_both_sides_from_the_family(self, triple):
        pair, plans = triple
        cell = pair_cell(pair)
        plan = plans[ARM_ADVERSARIAL_BOTH]
        assert plan.player_family_key == cell["family"]
        assert plan.opponent_family_key == cell["family"]
        assert plan.player_base_setup_id != plan.opponent_base_setup_id

    def test_all_boards_gated(self, triple):
        _, plans = triple
        for plan in plans.values():
            assert plan.orientation["paired_mirror"] is True


class TestManifest:
    def test_round_trip(self, tmp_path, setup_sources, library_document, triple):
        _, plans = triple
        manifest = build_baseline_manifest(
            list(plans.values()),
            generated_utc="t",
            library_digest=library_document["library_digest"],
        )
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(manifest, sort_keys=True))
        loaded = load_baseline_manifest(path)
        rebuilt = materialize_baseline(
            loaded, library=library_document, sources=setup_sources, verify=True
        )
        assert [plan.board_id for plan in rebuilt] == [
            plan.board_id for plan in plans.values()
        ]

    def test_tamper_refused(self, tmp_path, library_document, triple):
        _, plans = triple
        manifest = build_baseline_manifest(
            list(plans.values()),
            generated_utc="t",
            library_digest=library_document["library_digest"],
        )
        manifest["boards"][0]["blue_setup"][0] = 11
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(manifest, sort_keys=True))
        with pytest.raises(Phase16MeasurementError):
            load_baseline_manifest(path)


class TestFrozenArtifact:
    def test_delivered_manifest(self, repository_root):
        try:
            manifest = load_baseline_manifest(root=repository_root)
        except Phase16MeasurementError:
            pytest.skip("the frozen baseline manifest has not been generated yet")
        assert manifest["board_count"] == 3 * PAIR_COUNT
        assert set(manifest["balance"]["by_arm"].values()) == {PAIR_COUNT}
        assert set(manifest["balance"]["by_family"].values()) == {
            3 * SETUPS_PER_FAMILY
        }
