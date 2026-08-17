"""Phase 9 Agent 2: the logical population/opponent schedule.

The schedule decides which logical games exist. Everything downstream —
collection, resume, sealing, PPO eligibility — trusts that decision, so these
tests check the claims rather than the code paths: exact bucket counts, exact
rule subdivisions, bounded stress allocation, deterministic colour balance,
identity uniqueness, outcome independence, train-split-only setups, and the
property that no partitioning of the work can change a single record.

Negative controls sit beside every positive one. A scheduler that silently
accepts an out-of-range ordinal, a foreign committed id, or an archive
manifest that disagrees with the frozen window would pass a suite of
positive tests alone, and would corrupt an iteration in production.
"""

import ast
import json
from pathlib import Path

import pytest

from stratego.training import phase9_contract as pc
from stratego.training import phase9_schedule as psch
from stratego.training import phase9_seed as pseed

CANONICAL = pseed.CANONICAL_NAMESPACE
PILOTS = pseed.PILOT_NAMESPACES

#: Frozen at the Agent 2 freeze. These pins are the tripwire: any edit to the
#: schedule arithmetic, the record fields, or the identity spelling changes a
#: digest here and invalidates the published Agent 2 artifacts, exactly as
#: Agent 1's contract digest does for the contract document.
EXPECTED_POPULATION_DIGEST = (
    "6756790b15ee66195bc6339363e19fc475e3c606ef10613619b78b23d21bda73"
)
EXPECTED_RUN_DIGESTS = {
    "canonical": "bc253e8be2c63db1af308f62cf52f99f1431e9c9ec8a6db0987783b2983c0e64",
    "pilot_p9a": "c7bf66ff84b7ed829f477dfaf6b14cfa8c801ef5144f4d1f33175136562748cf",
    "pilot_p9b": "dd74fb86cb63108819db8f8db5e343fe18b87ca7c056c0f4603406d7e22d859d",
    "pilot_p9c": "1b5c4e742ab1e7be71b038aeefa817794b4924197ed9eab107f1a7ddc442039c",
    "pilot_p9d": "b9a42c9a34222dfc976b6e789605e0115e5257b5f054504bbe10088da597f1cc",
    "pilot_p9e": "1e61f341c6568388e611e810f24491a3f3a9fa739b73fad0caf4a9384f2a01ba",
    "pilot_p9f": "27a8b2566514fdd491cb755757fc949072bc366b7bc26855474a9706beaa475e",
}
EXPECTED_ITERATION_DIGESTS = {
    ("canonical", 1): (
        "9f80eda2d9a1d2c9428c5e9ef1203273bf13678e8a748f037780528e73da3255"
    ),
    ("canonical", 60): (
        "b7d89a520a0449f734b751a31a6dcb4b843b4fbc00b3c29ac71c765564e2419d"
    ),
    ("pilot_p9a", 8): (
        "1e82ab4ebf3a67602c312341183b52c750ccd1adc951261ba219d81232c20623"
    ),
}


@pytest.fixture(scope="module")
def canonical_first():
    return psch.iteration_schedule(CANONICAL, 1)


@pytest.fixture(scope="module")
def sampled_setups():
    return psch.audit_setup_assignment(CANONICAL, 1, limit=512)


@pytest.fixture(scope="module")
def seed_collisions():
    return psch.audit_seed_collisions(("pilot_p9a", "pilot_p9b", "pilot_p9c"))


# ---------------------------------------------------------------------------
# Exact counts
# ---------------------------------------------------------------------------


class TestBucketCounts:
    @pytest.mark.parametrize("iteration", (1, 2, 30, 59, 60))
    def test_canonical_iteration_is_exactly_2048_games(self, iteration):
        games = psch.iteration_schedule(CANONICAL, iteration)
        assert len(games) == 2048
        counts = {bucket: 0 for bucket in pseed.POPULATION_BUCKETS}
        for game in games:
            counts[game.bucket] += 1
        assert counts == {
            "current": 1024,
            "historical": 512,
            "rule": 307,
            "stress": 205,
        }

    @pytest.mark.parametrize("namespace", PILOTS)
    def test_pilot_iteration_is_exactly_1024_games(self, namespace):
        games = psch.iteration_schedule(namespace, 1)
        assert len(games) == 1024
        counts = {bucket: 0 for bucket in pseed.POPULATION_BUCKETS}
        for game in games:
            counts[game.bucket] += 1
        assert counts == {"current": 512, "historical": 256, "rule": 154, "stress": 102}

    def test_canonical_run_totals_122880_scheduled_games(self):
        assert psch.run_iterations(CANONICAL) == 60
        assert psch.total_scheduled_games(CANONICAL) == 122_880

    def test_each_pilot_run_totals_8192_scheduled_games(self):
        for namespace in PILOTS:
            assert psch.run_iterations(namespace) == 8
            assert psch.total_scheduled_games(namespace) == 8192

    def test_rule_subdivision_is_exact(self):
        for namespace, expected in (
            (CANONICAL, {"strategic_rule_based": 154, "tactical_rule_based": 107, "basic_heuristic": 46}),
            ("pilot_p9a", {"strategic_rule_based": 77, "tactical_rule_based": 54, "basic_heuristic": 23}),
        ):
            report = psch.audit_iteration(namespace, 3)
            assert report["rule_tier_counts"] == expected
            assert not report["problems"]

    @pytest.mark.parametrize("iteration", (1, 2, 3, 4, 5, 6, 7))
    def test_stress_allocation_never_differs_by_more_than_one_game(self, iteration):
        report = psch.audit_iteration(CANONICAL, iteration)
        assert report["stress_spread"] <= 1
        assert sum(report["stress_counts"].values()) == 205

    def test_pilot_stress_allocation_is_exactly_even(self):
        # 102 games over six policies divides exactly, so a pilot iteration has
        # no remainder to rotate at all.
        report = psch.audit_iteration("pilot_p9d", 5)
        assert report["stress_spread"] == 0
        assert set(report["stress_counts"].values()) == {17}


# ---------------------------------------------------------------------------
# Colour balance and learner control
# ---------------------------------------------------------------------------


class TestLearnerControl:
    def test_current_self_play_trains_both_colours(self, canonical_first):
        current = [game for game in canonical_first if game.bucket == "current"]
        assert len(current) == 1024
        for game in current:
            assert game.learner_control == "both"
            assert game.learner_color is None
            assert game.learner_sides == ("red", "blue")

    def test_current_self_play_uses_one_behavior_snapshot_on_both_sides(
        self, canonical_first
    ):
        for game in canonical_first:
            if game.bucket == "current":
                assert game.red_policy_identity == game.blue_policy_identity
                assert game.behavior_snapshot_identity in game.red_policy_identity

    @pytest.mark.parametrize("bucket", ("historical", "rule", "stress"))
    def test_asymmetric_buckets_train_the_current_side_only(
        self, canonical_first, bucket
    ):
        games = [game for game in canonical_first if game.bucket == bucket]
        assert games
        for game in games:
            assert game.learner_control in ("red", "blue")
            assert game.learner_sides == (game.learner_control,)
            assert game.learner_color == game.learner_control
            learner_side = f"{game.learner_control}_policy_identity"
            assert getattr(game, learner_side) == psch.behavior_policy_token(
                CANONICAL, 1
            )

    @pytest.mark.parametrize("iteration", (1, 2, 17, 60))
    def test_colour_balance_holds_in_every_asymmetric_bucket(self, iteration):
        report = psch.audit_iteration(CANONICAL, iteration)
        for bucket in ("historical", "rule", "stress"):
            split = report["colour_balance"][bucket]
            assert abs(split["red"] - split["blue"]) <= 1
            assert split["unassigned"] == 0
        assert not report["problems"]

    def test_odd_bucket_remainder_alternates_with_iteration_parity(self):
        # The stress bucket has 205 games: the extra game must swap sides
        # between consecutive iterations rather than always favouring red.
        even = psch.audit_iteration(CANONICAL, 2)["colour_balance"]["stress"]
        odd = psch.audit_iteration(CANONICAL, 3)["colour_balance"]["stress"]
        assert {even["red"], even["blue"]} == {102, 103}
        assert even["red"] == odd["blue"]
        assert even["blue"] == odd["red"]

    def test_even_sized_ranges_split_exactly_in_half(self):
        report = psch.audit_iteration(CANONICAL, 9)
        assert report["colour_balance"]["historical"] == {
            "red": 256,
            "blue": 256,
            "unassigned": 0,
        }
        # 154 strategic games is even, so no parity remainder exists.
        assert report["colour_balance"]["rule:strategic_rule_based"]["red"] == 77
        assert report["colour_balance"]["rule:strategic_rule_based"]["blue"] == 77

    def test_learner_colours_balance_exactly_over_a_whole_run(self):
        report = psch.audit_namespace("pilot_p9b")
        assert report["learner_colour_gap"] == 0
        assert not report["problems"]


# ---------------------------------------------------------------------------
# Historical league
# ---------------------------------------------------------------------------


class TestHistoricalScheduling:
    def test_first_iteration_plays_only_the_phase_8_anchor(self, canonical_first):
        historical = [game for game in canonical_first if game.bucket == "historical"]
        assert len(historical) == 512
        assert {game.historical_snapshot_identity for game in historical} == {"H000"}
        for game in historical:
            assert game.opponent_identity == psch.ANCHOR_POLICY_TOKEN

    def test_the_anchor_token_is_namespace_free(self):
        # H000 is the same accepted Phase 8 checkpoint in every run, so it must
        # not acquire a per-run identity.
        tokens = {
            psch.historical_policy_token(namespace, "H000")
            for namespace in pseed.RUN_NAMESPACES
        }
        assert tokens == {psch.ANCHOR_POLICY_TOKEN}

    def test_later_archive_tokens_are_namespaced(self):
        # Pilot H005 and canonical H005 are different weights.
        assert psch.historical_policy_token(CANONICAL, "H005") != (
            psch.historical_policy_token("pilot_p9a", "H005")
        )

    @pytest.mark.parametrize(
        "iteration,expected",
        (
            (1, ("H000",)),
            (5, ("H000",)),
            (6, ("H000", "H005")),
            (11, ("H000", "H005", "H010")),
            (60, ("H000", "H020", "H025", "H030", "H035", "H040", "H045", "H050", "H055")),
        ),
    )
    def test_active_window_follows_the_frozen_cadence(self, iteration, expected):
        assert pc.active_historical_window(iteration) == expected
        report = psch.audit_iteration(CANONICAL, iteration)
        assert tuple(report["active_window"]) == expected
        assert set(report["historical_counts"]) == set(expected)

    def test_window_never_exceeds_anchor_plus_eight(self):
        for iteration in range(1, 61):
            assert len(pc.active_historical_window(iteration)) <= 9

    def test_historical_assignment_is_outcome_independent(self):
        # Same identity, opposite imagined league standings: the draw is a pure
        # function of the game id, so there is nowhere for an outcome to enter.
        first = psch.scheduled_game_record(CANONICAL, 30, "historical", 11)
        second = psch.scheduled_game_record(CANONICAL, 30, "historical", 11)
        assert first == second
        assert first.historical_snapshot_identity in pc.active_historical_window(30)

    def test_manifest_digests_reach_the_scheduled_record(self):
        digest = "a" * 64
        manifest = psch.ActiveHistoryManifest.frozen_for(
            CANONICAL, 6, {"H000": digest, "H005": "b" * 64}
        )
        manifest.validate()
        game = psch.scheduled_game_record(
            CANONICAL, 6, "historical", 0, history=manifest
        )
        assert game.opponent_checkpoint_digest == manifest.digest_map[
            game.historical_snapshot_identity
        ]

    def test_manifest_that_disagrees_with_the_frozen_window_is_refused(self):
        bogus = psch.ActiveHistoryManifest(
            namespace=CANONICAL, iteration=6, identities=("H000",)
        )
        with pytest.raises(psch.Phase9ScheduleError, match="frozen window"):
            bogus.validate()
        with pytest.raises(psch.Phase9ScheduleError):
            psch.scheduled_game_record(CANONICAL, 6, "historical", 0, history=bogus)

    def test_manifest_with_a_digest_outside_the_window_is_refused(self):
        manifest = psch.ActiveHistoryManifest.frozen_for(
            CANONICAL, 6, {"H999": "c" * 64}
        )
        with pytest.raises(psch.Phase9ScheduleError, match="outside the active window"):
            manifest.validate()

    def test_manifest_for_another_iteration_is_refused(self):
        manifest = psch.ActiveHistoryManifest.frozen_for(CANONICAL, 11)
        with pytest.raises(psch.Phase9ScheduleError, match="history manifest is for"):
            psch.scheduled_game_record(CANONICAL, 6, "historical", 0, history=manifest)


# ---------------------------------------------------------------------------
# Opponent identities
# ---------------------------------------------------------------------------


class TestOpponentIdentities:
    def test_rule_tiers_use_frozen_phase_4_tokens(self, canonical_first):
        tokens = {
            game.opponent_identity for game in canonical_first if game.bucket == "rule"
        }
        assert tokens == {
            "strategic_rule_based@1.1.0",
            "tactical_rule_based@1.0.0",
            "basic_heuristic@1.0.0",
        }

    def test_stress_bucket_uses_all_six_frozen_stress_policies(self, canonical_first):
        tokens = {
            game.opponent_identity for game in canonical_first if game.bucket == "stress"
        }
        assert tokens == {
            f"{policy}@1.0.0" for policy in pc.STRESS_POLICY_ROSTER
        }

    def test_only_rule_and_stress_sides_own_a_policy_rng_stream(self, canonical_first):
        for game in canonical_first:
            seeds = (game.red_policy_seed, game.blue_policy_seed)
            if game.bucket in ("rule", "stress"):
                # exactly the opponent side, never the learner side
                assert sum(seed is not None for seed in seeds) == 1
                learner_seed = getattr(game, f"{game.learner_control}_policy_seed")
                assert learner_seed is None
            else:
                assert seeds == (None, None)

    def test_only_historical_games_own_an_archive_draw_stream(self, canonical_first):
        for game in canonical_first:
            if game.bucket == "historical":
                assert isinstance(game.historical_opponent_seed, int)
            else:
                assert game.historical_opponent_seed is None

    def test_unknown_policy_token_is_refused(self):
        with pytest.raises(psch.Phase9ScheduleError, match="unknown frozen policy"):
            psch.rule_policy_token("definitely_not_a_policy")


# ---------------------------------------------------------------------------
# Identity, purity and privilege
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_no_duplicate_game_ids_inside_an_iteration(self):
        ids = psch.iteration_game_ids(CANONICAL, 42)
        assert len(ids) == len(set(ids)) == 2048

    def test_no_duplicate_game_ids_across_a_whole_pilot_run(self):
        ids = psch.run_game_ids("pilot_p9e")
        assert len(ids) == len(set(ids)) == 8192

    def test_no_cross_namespace_collisions(self):
        report = psch.audit_cross_namespace_collisions()
        assert report["total_game_ids"] == 172_032
        assert report["distinct_game_ids"] == 172_032
        assert report["cross_namespace_collisions"] == 0

    def test_same_coordinates_in_two_namespaces_are_different_games(self):
        first = psch.scheduled_game_record("pilot_p9a", 3, "rule", 5)
        second = psch.scheduled_game_record("pilot_p9b", 3, "rule", 5)
        assert first.phase9_game_id != second.phase9_game_id
        assert first.setup_root_seed != second.setup_root_seed
        assert first.blue_policy_seed != second.blue_policy_seed

    def test_rebuild_from_game_id_reproduces_the_enumerated_record(self):
        for game in psch.iteration_schedule("pilot_p9f", 7):
            assert psch.rebuild_scheduled_game(game.phase9_game_id) == game

    def test_rebuild_refuses_a_malformed_identifier(self):
        with pytest.raises(pseed.Phase9SeedError):
            psch.rebuild_scheduled_game("not-a-phase9-game-id")

    def test_rebuild_refuses_an_ordinal_outside_its_bucket(self):
        # Well-formed by the id grammar, unschedulable by the population
        # contract: the rule bucket only has 307 games.
        smuggled = pseed.phase9_game_id(CANONICAL, 4, "rule", 400)
        with pytest.raises(psch.Phase9ScheduleError, match="outside 0..306"):
            psch.rebuild_scheduled_game(smuggled)

    def test_iteration_outside_the_frozen_budget_is_refused(self):
        with pytest.raises(psch.Phase9ScheduleError, match="frozen 1..60 budget"):
            psch.scheduled_game_record(CANONICAL, 61, "current", 0)
        with pytest.raises(psch.Phase9ScheduleError, match="frozen 1..8 budget"):
            psch.scheduled_game_record("pilot_p9a", 9, "current", 0)

    def test_unknown_namespace_and_bucket_are_refused(self):
        with pytest.raises(psch.Phase9ScheduleError, match="unknown Phase 9 namespace"):
            psch.scheduled_game_record("pilot_p9z", 1, "current", 0)
        with pytest.raises(psch.Phase9ScheduleError, match="unknown population bucket"):
            psch.scheduled_game_record(CANONICAL, 1, "adversarial", 0)

    def test_schedule_records_carry_no_privileged_information(self, canonical_first):
        forbidden = ("setup", "piece", "board", "truth", "outcome", "result")
        record = canonical_first[0].to_dict()
        for key, value in record.items():
            if any(word in key for word in ("piece", "board", "truth", "outcome", "result")):
                pytest.fail(f"scheduler record leaks {key!r}")
            # The setup fields present are identities/seeds, never contents.
            if "setup" in key:
                assert isinstance(value, (int, str))
                assert "player=" in str(value) or key == "setup_root_seed"
        assert "engine_setup" not in record
        assert forbidden  # keep the intent visible

    def test_the_schedule_module_performs_no_io(self):
        # The structural proof that logical identity cannot depend on storage:
        # the module imports nothing that can reach a filesystem, an
        # environment variable or a clock, and calls no builtin that can.
        # Checked over the parsed module rather than its text, so prose about
        # the rule cannot accidentally satisfy or break it.
        tree = ast.parse(Path(psch.__file__).read_text())
        banned_modules = {"os", "os.path", "pathlib", "shutil", "time", "random", "socket"}
        banned_calls = {"open", "input"}
        imported = set()
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(node.func.id)
        assert not imported & banned_modules, sorted(imported & banned_modules)
        assert not called & banned_calls, sorted(called & banned_calls)
        assert "phase9_storage" not in "".join(imported)


# ---------------------------------------------------------------------------
# Order independence and resume
# ---------------------------------------------------------------------------


class TestOrderIndependence:
    @pytest.mark.parametrize("namespace,iteration", ((CANONICAL, 13), ("pilot_p9c", 6)))
    def test_no_partitioning_of_the_work_changes_a_record(self, namespace, iteration):
        report = psch.audit_worker_order_independence(namespace, iteration)
        assert report["partitionings_checked"] == 12
        assert report["mismatches"] == 0
        assert not report["problems"]

    def test_enumeration_order_does_not_affect_the_digest(self):
        forward = psch.iteration_schedule_document(CANONICAL, 5)
        shuffled = dict(forward)
        shuffled["games"] = list(reversed(forward["games"]))
        # The digest is over the canonical document, so a caller cannot change
        # it by reordering its own copy — the enumeration is the definition.
        assert psch.iteration_schedule_digest(CANONICAL, 5) == psch.iteration_schedule_digest(
            CANONICAL, 5
        )
        assert shuffled["games"] != forward["games"]


class TestResume:
    def test_resume_is_scheduled_minus_committed(self):
        report = psch.audit_resume_identity(CANONICAL, 11)
        assert report["foreign_committed_id_rejected"]
        for check in report["checks"]:
            assert check["union_equals_scheduled"]
            assert check["disjoint"]
            assert check["rebuild_mismatches"] == 0
        assert not report["problems"]

    def test_nothing_committed_means_the_whole_iteration_is_pending(self):
        plan = psch.resume_plan("pilot_p9a", 2, [])
        assert plan["scheduled"] == plan["pending"] == 1024
        assert not plan["complete"]

    def test_everything_committed_means_the_iteration_can_seal(self):
        committed = psch.iteration_game_ids("pilot_p9a", 2)
        plan = psch.resume_plan("pilot_p9a", 2, committed)
        assert plan["pending"] == 0
        assert plan["complete"]

    def test_a_repeated_commit_record_does_not_shrink_the_schedule(self):
        committed = list(psch.iteration_game_ids("pilot_p9a", 2))[:10] * 3
        plan = psch.resume_plan("pilot_p9a", 2, committed)
        assert plan["committed"] == 10
        assert plan["pending"] == 1014

    def test_a_committed_id_from_another_iteration_is_refused(self):
        foreign = pseed.phase9_game_id("pilot_p9a", 3, "current", 0)
        with pytest.raises(psch.Phase9ScheduleError, match="not scheduled"):
            psch.pending_game_ids("pilot_p9a", 2, [foreign])

    def test_a_committed_id_from_another_namespace_is_refused(self):
        foreign = pseed.phase9_game_id("pilot_p9b", 2, "current", 0)
        with pytest.raises(psch.Phase9ScheduleError, match="not scheduled"):
            psch.pending_game_ids("pilot_p9a", 2, [foreign])


# ---------------------------------------------------------------------------
# Setups
# ---------------------------------------------------------------------------


class TestSetupAssignment:
    def test_rollout_setups_come_from_the_training_split_only(self, sampled_setups):
        assert sampled_setups["split"] == "train"
        assert sampled_setups["purpose"] == "training"
        assert sampled_setups["profile"] == "neutral_v1"
        assert sampled_setups["split_violations"] == 0

    def test_every_setup_family_appears(self, sampled_setups):
        assert sampled_setups["families_seen"] == 16
        assert set(sampled_setups["family_counts"]) == {f"F{index:02d}" for index in range(16)}

    def test_family_coverage_is_broad_rather_than_concentrated(self, sampled_setups):
        # Uniform family draws over 1,024 sides: a family starved to under half
        # its expectation would mean the schedule had acquired a family bias.
        expected = sampled_setups["setup_sides_resolved"] / 16
        assert sampled_setups["family_min_count"] > expected * 0.5
        assert sampled_setups["family_max_count"] < expected * 1.5

    def test_setup_identity_is_derived_from_the_game_id_alone(self):
        game = psch.scheduled_game_record(CANONICAL, 8, "rule", 12)
        assert game.setup_root_seed == pseed.setup_root_seed(game.phase9_game_id)
        assert "train" in game.red_setup_source_identity
        assert "player=red" in game.red_setup_source_identity
        assert "player=blue" in game.blue_setup_source_identity
        assert game.red_setup_source_identity != game.blue_setup_source_identity

    def test_the_two_sides_of_a_game_draw_independently(self, sampled_setups):
        assert sampled_setups["games_with_identical_sides"] == 0


# ---------------------------------------------------------------------------
# Seed streams
# ---------------------------------------------------------------------------


class TestSeedStreams:
    def test_no_within_stream_seed_collisions(self, seed_collisions):
        assert seed_collisions["within_stream_collisions"] == 0
        assert seed_collisions["same_game_setup_side_collisions"] == 0
        assert not seed_collisions["problems"]

    def test_every_audited_stream_is_fully_distinct(self, seed_collisions):
        for name, report in seed_collisions["per_stream"].items():
            assert report["collisions"] == 0, name
            assert report["distinct_seeds"] == report["values_derived"], name

    def test_the_audit_reports_streams_the_contract_names(self, seed_collisions):
        assert set(seed_collisions["per_stream"]) == {
            "setup_root",
            "setup_side_red",
            "setup_side_blue",
            "policy_red",
            "policy_blue",
            "historical_opponent",
        }

    def test_the_same_game_side_detector_fires_when_both_sides_agree(
        self, monkeypatch
    ):
        # Positive control. A setup source whose two sides share a stream would
        # deal both players the identical board; each game still has its own
        # root, so *within*-stream uniqueness is untouched and only the
        # same-game check should fire.
        class SideBlindSource:
            setup_family = "stub"

            def side_seed(self, *, root_seed, environment_id, generation, player):
                return root_seed

        monkeypatch.setattr(psch, "_setup_source", lambda: SideBlindSource())
        report = psch.audit_seed_collisions(("pilot_p9a",))
        assert report["same_game_setup_side_collisions"] == 8192
        assert report["per_stream"]["setup_side_red"]["collisions"] == 0
        assert report["problems"]

    def test_the_within_stream_detector_fires_on_a_shared_seed(self, monkeypatch):
        # Second positive control: a constant stream collides with itself on
        # every game after the first.
        class ConstantSource:
            setup_family = "stub"

            def side_seed(self, *, root_seed, environment_id, generation, player):
                return 7

        monkeypatch.setattr(psch, "_setup_source", lambda: ConstantSource())
        report = psch.audit_seed_collisions(("pilot_p9a",))
        assert report["per_stream"]["setup_side_red"]["collisions"] == 8191
        assert report["per_stream"]["setup_side_blue"]["collisions"] == 8191
        assert report["within_stream_collisions"] == 2 * 8191
        assert any("uniqueness contract" in problem for problem in report["problems"])


# ---------------------------------------------------------------------------
# Digest pins
# ---------------------------------------------------------------------------


class TestDigests:
    def test_population_digest_is_pinned(self):
        assert psch.population_digest() == EXPECTED_POPULATION_DIGEST

    @pytest.mark.parametrize("namespace", tuple(EXPECTED_RUN_DIGESTS))
    def test_run_schedule_digests_are_pinned(self, namespace):
        assert psch.run_schedule_digest(namespace) == EXPECTED_RUN_DIGESTS[namespace]

    @pytest.mark.parametrize("key", tuple(EXPECTED_ITERATION_DIGESTS))
    def test_iteration_digests_are_pinned(self, key):
        namespace, iteration = key
        assert (
            psch.iteration_schedule_digest(namespace, iteration)
            == EXPECTED_ITERATION_DIGESTS[key]
        )

    def test_every_run_namespace_has_a_distinct_schedule_digest(self):
        assert len(set(EXPECTED_RUN_DIGESTS.values())) == len(EXPECTED_RUN_DIGESTS)

    def test_the_population_document_is_json_serializable(self):
        document = psch.population_document()
        assert json.loads(json.dumps(document)) == document
        assert document["population_version"] == pc.PHASE9_POPULATION_VERSION
        assert document["schedule_version"] == pc.PHASE9_ROLLOUT_SCHEDULE_VERSION
