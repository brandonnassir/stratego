"""Phase 11 Agent 6: the soak schedule, request derivation and audit shape.

Unit-level tests over the Agent 6 harness's soak driver. Nothing here plays
a game or runs a forward: what is checked is the part that must be right
*before* the soak runs — that the schedule is Agent 1's frozen one, that the
attachment rule is the frozen arithmetic, that the ids live in the soak
namespace and nowhere near a bank, and that the set algebra a resume depends
on is exact.

The driver lives in `scripts/run_phase11_agent06.py` rather than under
`stratego/evaluation`, because every `phase11_*.py` module there is covered
by `phase11_pipeline.FROZEN_IMPLEMENTATION_MODULES` and adding one after
Agent 5's freeze would change the frozen implementation digest itself. The
harness is loaded the way the accepted Phase 9 harness tests load theirs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from stratego.training.phase11_contract import OPPONENT_STRATA, SOURCE_P10D
from stratego.training.phase11_seed import (
    SOAK_GAME_COUNT,
    SOAK_GAMES_PER_STRATUM,
    SOAK_REQUESTS_PER_GAME,
    SOAK_REQUEST_COUNT,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPOSITORY_ROOT / "scripts" / "run_phase11_agent06.py"


def _load_harness():
    specification = importlib.util.spec_from_file_location(
        "run_phase11_agent06_under_test", HARNESS_PATH
    )
    module = importlib.util.module_from_spec(specification)
    # Registered before execution so the module's dataclasses resolve.
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


soak = _load_harness()


def test_the_soak_driver_is_not_a_frozen_phase11_module():
    """Agent 6 adds no module to the Agent 5 implementation freeze."""
    from stratego.evaluation import phase11_pipeline as pipeline

    assert not (REPOSITORY_ROOT / "stratego" / "evaluation" / "phase11_soak.py").exists()
    live = {
        f"stratego/evaluation/{path.name}"
        for path in (REPOSITORY_ROOT / "stratego" / "evaluation").glob("phase11_*.py")
    } | {
        f"stratego/training/{path.name}"
        for path in (REPOSITORY_ROOT / "stratego" / "training").glob("phase11_*.py")
    }
    assert live == set(pipeline.FROZEN_IMPLEMENTATION_MODULES)


@pytest.fixture(scope="module")
def descriptors():
    return soak.soak_game_descriptors()


def test_schedule_is_the_frozen_volume(descriptors):
    assert len(descriptors) == SOAK_GAME_COUNT == 1024
    per_stratum = {}
    for descriptor in descriptors:
        per_stratum[descriptor["stratum"]] = per_stratum.get(descriptor["stratum"], 0) + 1
    assert set(per_stratum) == set(OPPONENT_STRATA)
    assert set(per_stratum.values()) == {SOAK_GAMES_PER_STRATUM}


def test_schedule_is_train_split_p10d_on_both_seats(descriptors):
    assert {descriptor["split"] for descriptor in descriptors} == {"train"}
    assert {descriptor["setup_source"] for descriptor in descriptors} == {SOURCE_P10D}
    assert soak.SOAK_SPLIT == "train"
    assert soak.SOAK_SETUP_SOURCE == SOURCE_P10D


def test_observer_colour_follows_the_frozen_ordinal_parity(descriptors):
    for descriptor in descriptors:
        expected = "red" if descriptor["game_ordinal"] % 2 == 0 else "blue"
        assert descriptor["observer_color"] == expected
        assert descriptor["opponent_color"] != descriptor["observer_color"]
    colours = {descriptor["observer_color"] for descriptor in descriptors}
    assert colours == {"red", "blue"}


def test_every_game_and_seed_identity_is_distinct(descriptors):
    ids = [descriptor["game_id"] for descriptor in descriptors]
    assert len(set(ids)) == len(ids)
    seeds = [descriptor["match_seed"] for descriptor in descriptors]
    assert len(set(seeds)) == len(seeds)
    setup_seeds = [
        seed
        for descriptor in descriptors
        for seed in (descriptor["observer_setup_seed"], descriptor["opponent_setup_seed"])
    ]
    assert len(set(setup_seeds)) == len(setup_seeds)


def test_schedule_is_a_pure_function_of_identity():
    assert soak.soak_game_descriptors() == soak.soak_game_descriptors()


def test_ids_live_in_the_soak_namespace(descriptors):
    for descriptor in descriptors:
        assert descriptor["game_id"].startswith("phase11_soak_v1|ms=2026081901|")


def test_attachment_rule_is_the_frozen_arithmetic():
    for total in (1, 2, 3, 7, 8, 9, 25, 47, 120):
        positions = soak.request_decision_positions(total)
        assert positions == [
            min((k * total) // SOAK_REQUESTS_PER_GAME, total - 1)
            for k in range(SOAK_REQUESTS_PER_GAME)
        ]
        assert len(positions) == SOAK_REQUESTS_PER_GAME
        assert positions == sorted(positions)
        assert 0 <= positions[0] and positions[-1] <= total - 1


def test_attachment_rule_shares_a_decision_only_on_short_games():
    assert len(set(soak.request_decision_positions(8))) == 8
    assert len(set(soak.request_decision_positions(20))) == 8
    assert len(set(soak.request_decision_positions(3))) == 3


def test_attachment_rule_refuses_a_game_with_no_decision():
    with pytest.raises(soak.Phase11SoakError):
        soak.request_decision_positions(0)


def _game(index: int, decisions: int) -> dict:
    descriptor = soak.soak_game_descriptors()[index]
    return {
        "game_index": index,
        "game_id": descriptor["game_id"],
        "stratum": descriptor["stratum"],
        "observer_color": descriptor["observer_color"],
        "plies": 2 * decisions,
        "terminal_reason": "flag_capture",
        "observer_result": "win",
        "observer_decisions": decisions,
        "observer_decision_indices": [2 * step for step in range(decisions)],
        "observer_state_identities": [f"identity{step:04d}" for step in range(decisions)],
        "observer_hidden_counts": [40] * decisions,
        "red_setup": list(range(40)),
        "blue_setup": list(range(40)),
        "action_history": [1, 2, 3, 4],
    }


def test_request_specs_carry_everything_the_request_path_needs():
    specs = soak.game_request_specs(_game(0, 24), 0)
    assert len(specs) == SOAK_REQUESTS_PER_GAME
    for ordinal, spec in enumerate(specs):
        assert spec["request_ordinal"] == ordinal
        assert spec["request_id"].endswith(f"|r={ordinal}")
        for field in (
            "game_id",
            "observer_color",
            "decision_index",
            "public_state_identity",
            "red_setup",
            "blue_setup",
            "action_history",
        ):
            assert field in spec


def test_request_schedule_is_the_frozen_count_and_has_no_duplicate_id():
    games = [_game(index, 24) for index in range(SOAK_GAME_COUNT)]
    specs = soak.build_request_schedule(games)
    assert len(specs) == SOAK_REQUEST_COUNT == 8192
    ids = [spec["request_id"] for spec in specs]
    assert len(set(ids)) == len(ids)
    assert [spec["request_ordinal"] for spec in specs] == list(range(len(specs)))
    findings = soak.schedule_findings(games, specs)
    assert findings["schedule_is_complete"] is True
    assert findings["zero_decision_games"] == 0


def test_request_schedule_never_exceeds_the_frozen_count():
    games = [_game(index, 24) for index in range(SOAK_GAME_COUNT)]
    games.append(_game(0, 24))
    with pytest.raises(soak.Phase11SoakError):
        soak.build_request_schedule(games)


def test_a_game_the_observer_never_moved_in_contributes_nothing():
    """The frozen attachment rule has no decision to attach to, so it does
    not attach one — and nothing is substituted for the missing requests."""
    assert soak.game_request_specs(_game(0, 0), 0) == []
    games = [_game(index, 24) for index in range(SOAK_GAME_COUNT)]
    games[5] = _game(5, 0)
    specs = soak.build_request_schedule(games)
    assert len(specs) == SOAK_REQUEST_COUNT - SOAK_REQUESTS_PER_GAME
    assert [spec["request_ordinal"] for spec in specs] == list(range(len(specs)))
    assert games[5]["game_id"] not in {spec["soak_game_id"] for spec in specs}


def test_schedule_findings_report_the_shortfall_exactly():
    games = [_game(index, 24) for index in range(SOAK_GAME_COUNT)]
    for index in (5, 9, 11):
        games[index] = _game(index, 0)
    specs = soak.build_request_schedule(games)
    findings = soak.schedule_findings(games, specs)
    assert findings["zero_decision_games"] == 3
    assert findings["realizable_request_count"] == len(specs)
    assert findings["unrealizable_requests"] == 3 * SOAK_REQUESTS_PER_GAME
    assert findings["schedule_is_complete"] is False
    assert findings["every_playable_game_gave_eight"] is True
    assert len(findings["zero_decision_game_ids"]) == 3


def test_schedule_digest_moves_with_content():
    descriptors = soak.soak_game_descriptors()
    ids = ["a", "b", "c"]
    first = soak.schedule_digest(descriptors, ids)
    assert first == soak.schedule_digest(descriptors, ids)
    assert first != soak.schedule_digest(descriptors, ids + ["d"])
    altered = [dict(descriptors[0]), *descriptors[1:]]
    altered[0]["match_seed"] = int(altered[0]["match_seed"]) + 1
    assert first != soak.schedule_digest(altered, ids)


def test_set_reconciliation_is_exact():
    scheduled = ["a", "b", "c"]
    clean = soak.set_reconciliation(scheduled, ["c", "a", "b"])
    assert clean["exactly_scheduled"]
    assert clean["missing_request_ids"] == []
    assert clean["duplicate_request_ids"] == []
    assert clean["unscheduled_request_ids"] == []

    missing = soak.set_reconciliation(scheduled, ["a", "b"])
    assert missing["missing_request_ids"] == ["c"]
    assert not missing["exactly_scheduled"]

    duplicate = soak.set_reconciliation(scheduled, ["a", "a", "b", "c"])
    assert duplicate["duplicate_request_ids"] == ["a"]
    assert not duplicate["exactly_scheduled"]

    unscheduled = soak.set_reconciliation(scheduled, ["a", "b", "c", "z"])
    assert unscheduled["unscheduled_request_ids"] == ["z"]
    assert not unscheduled["exactly_scheduled"]


def test_store_content_digest_is_content_only_and_order_free():
    rows = [
        {
            "request_id": "r1",
            "public_state_identity": "p1",
            "hidden_pieces": 40,
            "worlds": 64,
            "digest": "d1",
            "total_ns": 1,
        },
        {
            "request_id": "r0",
            "public_state_identity": "p0",
            "hidden_pieces": 39,
            "worlds": 64,
            "digest": "d0",
            "total_ns": 2,
        },
    ]
    first = soak.store_content_digest(rows)
    assert first == soak.store_content_digest(list(reversed(rows)))
    # A wall-clock field cannot move it: that is the whole point.
    timed = [dict(row, total_ns=row["total_ns"] * 1000) for row in rows]
    assert first == soak.store_content_digest(timed)
    # Content does move it.
    changed = [dict(rows[0], digest="other"), rows[1]]
    assert first != soak.store_content_digest(changed)


def test_progress_bucket_follows_the_frozen_boundaries():
    assert soak.progress_bucket(0) == "early"
    assert soak.progress_bucket(39) == "early"
    assert soak.progress_bucket(40) == "middle"
    assert soak.progress_bucket(119) == "middle"
    assert soak.progress_bucket(120) == "late"


def test_soak_world_count_is_the_production_request_shape():
    assert soak.SOAK_WORLD_ORDINALS == 64


def test_soak_run_version_is_not_the_bank_run_version():
    from stratego.evaluation.phase11_runner import PHASE11_RUN_VERSION

    assert soak.SOAK_RUN_VERSION != PHASE11_RUN_VERSION


# ---------------------------------------------------------------------------
# The reviewer-authorized supplement
# ---------------------------------------------------------------------------


def test_the_local_id_and_seed_rules_are_agent_1s_over_the_whole_frozen_range():
    """The supplement's whole legitimacy rests on this: the extended rules
    must BE Agent 1's rules, proven against the frozen helpers rather than
    asserted, before any supplemental game is drawn."""
    proof = soak.assert_extension_matches_frozen_rules()
    assert proof["rules_identical"] is True
    assert proof["mismatches"] == 0
    assert proof["frozen_games_covered"] == SOAK_GAME_COUNT
    # per game: the id, the observer colour, the match seed, both setup
    # seeds and all eight request ids.
    assert proof["comparisons"] == SOAK_GAME_COUNT * (5 + SOAK_REQUESTS_PER_GAME)
    assert proof["comparisons"] == 13_312


def test_the_frozen_module_is_not_edited_to_reach_the_supplement():
    """`phase11_seed.py` is inside the Agent 5 freeze; the supplement may
    not touch it, so the frozen helper still refuses the extended range."""
    from stratego.training.phase11_seed import Phase11SeedError, phase11_soak_game_id

    with pytest.raises(Phase11SeedError):
        phase11_soak_game_id(OPPONENT_STRATA[0], SOAK_GAMES_PER_STRATUM)
    # ... while the harness's own formatter extends it, in the same format.
    extended = soak.soak_game_id(OPPONENT_STRATA[0], SOAK_GAMES_PER_STRATUM)
    assert extended.startswith("phase11_soak_v1|ms=2026081901|st=")
    assert extended.endswith(f"|g={SOAK_GAMES_PER_STRATUM:03d}")


def test_supplemental_enumeration_is_ordinal_major_and_starts_after_the_range():
    order = soak.supplemental_candidate_order()
    first = [next(order) for _ in range(len(OPPONENT_STRATA) * 2)]
    assert soak.SUPPLEMENTAL_FIRST_ORDINAL == SOAK_GAMES_PER_STRATUM == 128
    assert [stratum for stratum, _ in first[: len(OPPONENT_STRATA)]] == list(
        OPPONENT_STRATA
    )
    assert {ordinal for _, ordinal in first[: len(OPPONENT_STRATA)]} == {128}
    assert {ordinal for _, ordinal in first[len(OPPONENT_STRATA) :]} == {129}


def test_supplemental_enumeration_is_a_pure_function_of_arithmetic():
    first = list(zip(soak.supplemental_candidate_order(), range(40)))
    second = list(zip(soak.supplemental_candidate_order(), range(40)))
    assert first == second


def test_supplemental_descriptors_keep_every_frozen_rule():
    for index, (stratum, ordinal) in zip(
        range(len(OPPONENT_STRATA) * 2), soak.supplemental_candidate_order()
    ):
        descriptor = soak.supplemental_descriptor(stratum, ordinal, 1024 + index)
        assert descriptor["split"] == "train"
        assert descriptor["setup_source"] == SOURCE_P10D
        assert descriptor["stratum"] == stratum
        assert descriptor["supplemental"] is True
        assert descriptor["observer_color"] == ("red" if ordinal % 2 == 0 else "blue")
        assert descriptor["opponent_color"] != descriptor["observer_color"]
        assert descriptor["game_id"] == soak.soak_game_id(stratum, ordinal)


def test_supplemental_seeds_never_collide_with_the_frozen_ones():
    frozen = set()
    for stratum in OPPONENT_STRATA:
        for ordinal in range(SOAK_GAMES_PER_STRATUM):
            seeds = soak.soak_stream_seeds(soak.soak_game_id(stratum, ordinal))
            frozen.update(seeds.values())
    supplemental = set()
    for index, (stratum, ordinal) in zip(
        range(len(OPPONENT_STRATA) * 8), soak.supplemental_candidate_order()
    ):
        seeds = soak.soak_stream_seeds(soak.soak_game_id(stratum, ordinal))
        supplemental.update(seeds.values())
    assert frozen & supplemental == set()
    assert len(supplemental) == len(OPPONENT_STRATA) * 8 * 3


def test_original_preservation_detects_any_disturbance():
    original = [
        {
            "request_id": f"r{index}",
            "request_ordinal": index,
            "soak_game_id": "g",
            "decision_index": index,
            "public_state_identity": f"p{index}",
        }
        for index in range(4)
    ]
    appended = original + [
        {
            "request_id": "r4",
            "request_ordinal": 4,
            "soak_game_id": "g2",
            "decision_index": 4,
            "public_state_identity": "p4",
        }
    ]
    clean = soak.original_preservation(original, appended)
    assert clean["original_requests_preserved_exactly"] is True
    assert clean["first_difference"] is None
    assert clean["supplemental_requests"] == 1

    disturbed = [dict(row) for row in appended]
    disturbed[2]["public_state_identity"] = "tampered"
    dirty = soak.original_preservation(original, disturbed)
    assert dirty["original_requests_preserved_exactly"] is False
    assert "public_state_identity" in dirty["first_difference"]

    truncated = soak.original_preservation(original, original[:2])
    assert truncated["original_requests_preserved_exactly"] is False


def test_the_supplement_targets_exactly_the_diagnosed_deficit():
    assert soak.SUPPLEMENTAL_PLAYABLE_TARGET == 29
    assert soak.SUPPLEMENTAL_PLAYABLE_TARGET * SOAK_REQUESTS_PER_GAME == 232
    assert soak.SUPPLEMENTAL_AUTHORIZED is True
