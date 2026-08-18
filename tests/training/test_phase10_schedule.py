"""Regression: the frozen 16,384-game Phase 10 setup-outcome schedule.

The schedule is arithmetic, not sampling. These tests pin the arithmetic,
the identity space, the train-only rule and the isolated-rebuild property
Agent 2 depends on for crash resume.
"""

import pytest

from stratego.setups.families import FAMILY_IDS
from stratego.training import phase10_schedule as sch
from stratego.training import phase10_seed as ps
from tests.training.phase10_frozen_digests import OUTCOME_SCHEDULE_DIGEST


@pytest.fixture(scope="module")
def schedule():
    return sch.enumerate_schedule()


class TestArithmetic:
    def test_the_frozen_counts(self):
        assert sch.FAMILY_COUNT == 16
        assert sch.ORDERED_FAMILY_PAIRS == 256
        assert sch.GAMES_PER_ORDERED_PAIR == 64
        assert sch.TOTAL_CORPUS_GAMES == 16_384

    def test_enumeration_is_exactly_the_scheduled_size(self, schedule):
        assert len(schedule) == 16_384

    def test_every_ordered_pair_appears_exactly_64_times(self, schedule):
        counts = {}
        for game in schedule:
            counts[(game.red_family, game.blue_family)] = (
                counts.get((game.red_family, game.blue_family), 0) + 1
            )
        assert len(counts) == 256
        assert set(counts.values()) == {64}
        assert set(counts) == {
            (red, blue) for red in FAMILY_IDS for blue in FAMILY_IDS
        }

    def test_ordering_is_a_real_distinction(self, schedule):
        pairs = {(game.red_family, game.blue_family) for game in schedule}
        assert ("F00", "F01") in pairs and ("F01", "F00") in pairs


class TestIdentity:
    def test_game_ids_and_match_seeds_are_unique(self, schedule):
        assert len({game.game_id for game in schedule}) == len(schedule)
        assert len({game.match_seed for game in schedule}) == len(schedule)

    def test_ids_carry_no_path_and_no_worker_information(self, schedule):
        for game in schedule[:64]:
            assert "/" not in game.game_id
            assert "worker" not in game.game_id

    def test_isolated_rebuild_is_exact(self, schedule):
        for game in schedule[::997]:
            assert sch.rebuild_game(game.game_id) == game

    def test_rebuild_refuses_an_out_of_range_ordinal(self):
        game_id = ps.phase10_game_id("F00", "F00", 90)
        with pytest.raises(sch.Phase10ScheduleError):
            sch.rebuild_game(game_id)

    def test_enumeration_is_byte_stable(self):
        assert sch.enumerate_schedule() == sch.enumerate_schedule()

    def test_schedule_digest_is_pinned(self):
        assert sch.schedule_digest() == OUTCOME_SCHEDULE_DIGEST


class TestAudit:
    def test_the_full_audit_passes(self):
        audit = sch.audit_schedule()
        assert audit["all_pass"], {k: v for k, v in audit["checks"].items() if not v}
        assert audit["total_games"] == 16_384
        assert audit["ordered_pair_count"] == 256
        assert audit["games_per_ordered_pair"] == [64]

    def test_setup_and_match_streams_are_disjoint(self):
        audit = sch.audit_schedule()
        assert audit["checks"]["seed_streams_disjoint"]


class TestSideDraws:
    def test_a_side_draw_matches_its_scheduled_family_and_split(self):
        from stratego.setups.sampler import load_library_index

        library = load_library_index()
        for game in sch.enumerate_schedule()[::4093]:
            for color in ("red", "blue"):
                sampled, attempt, seed = sch.resolve_side(
                    game.game_id, color, index=library
                )
                assert sampled.family_id == game.side_family(color)
                assert sampled.split == "train"
                assert seed == ps.corpus_setup_seed(game.game_id, color, attempt)

    def test_a_side_draw_is_deterministic(self):
        game_id = sch.enumerate_schedule()[0].game_id
        first = sch.resolve_side(game_id, "red")
        second = sch.resolve_side(game_id, "red")
        assert first[0].canonical == second[0].canonical
        assert first[1:] == second[1:]

    def test_corpus_never_reaches_a_held_out_base(self):
        assert sch.CORPUS_SPLIT == "train"
        assert sch.corpus_contract_document()["held_out_bases_used"] == 0


class TestOutcomeRecord:
    def test_schema_carries_everything_a_replay_needs(self):
        names = {entry["name"] for entry in sch.outcome_record_schema()["fields"]}
        for required in (
            "game_id",
            "red_base_setup_id",
            "blue_base_setup_id",
            "red_family",
            "blue_family",
            "trait_schema_version",
            "red_provenance",
            "blue_provenance",
            "result",
            "plies",
            "terminal_reason",
            "move_policy_identity",
            "red_setup_draw_seed",
            "blue_setup_draw_seed",
            "payload_digest",
            "metadata_digest",
            "commit_digest",
        ):
            assert required in names

    def test_result_targets_are_the_frozen_red_perspective_ones(self):
        assert sch.RESULT_TARGETS == {"red_win": 1.0, "draw": 0.5, "red_loss": 0.0}

    def test_move_behavior_is_greedy_float32_single_request_no_search(self):
        assert sch.CORPUS_MOVE_BEHAVIOR["decision_mode"] == "greedy"
        assert sch.CORPUS_MOVE_BEHAVIOR["dtype"] == "float32"
        assert sch.CORPUS_MOVE_BEHAVIOR["batch_policy"] == "single_request"
        assert sch.CORPUS_MOVE_BEHAVIOR["search"] == "none"
        assert sch.CORPUS_MOVE_BEHAVIOR["optimizer_steps"] == 0
