"""Phase 11 Agent 1: bank construction, balance, rebuild and the ledger."""

import json

import pytest

from stratego.evaluation import phase11_banks as pb
from stratego.setups.contracts import parse_base_setup_id, split_for_base_index
from stratego.setups.identity import content_fingerprint
from stratego.training.phase11_contract import (
    TEST_BANK_VERSION,
    VALIDATION_BANK_VERSION,
)
from stratego.training.phase11_seed import (
    CASE_GAME_OBSERVER_COLOR,
    CASE_GAME_OPPONENT_COLOR,
    OPPONENT_STRATA,
    ROLE_OBSERVER,
    ROLE_OPPONENT,
    SETUP_SOURCES,
    SOURCE_NEUTRAL,
    SOURCE_P10D,
    case_setup_seed,
    game_match_seed,
    parse_phase11_case_id,
)


@pytest.fixture(scope="module")
def sources():
    return pb.Phase11SetupSources()


@pytest.fixture(scope="module")
def sample_cases(sources):
    """One case from every (stratum, source) cell of the validation bank."""
    per_cell = pb.BANK_SPECIFICATIONS["validation"]["cases_per_cell"]
    indices = [
        ((stratum_index * 2) + source_index) * per_cell + 7
        for stratum_index in range(len(OPPONENT_STRATA))
        for source_index in range(len(SETUP_SOURCES))
    ]
    return tuple(pb.build_case("validation", index, sources) for index in indices)


def test_bank_specifications_match_the_contract():
    assert pb.BANK_SPECIFICATIONS["validation"]["case_count"] == 512
    assert pb.BANK_SPECIFICATIONS["validation"]["bank_version"] == VALIDATION_BANK_VERSION
    assert pb.BANK_SPECIFICATIONS["validation"]["split"] == "validation"
    assert pb.BANK_SPECIFICATIONS["test"]["case_count"] == 2_048
    assert pb.BANK_SPECIFICATIONS["test"]["bank_version"] == TEST_BANK_VERSION
    assert pb.BANK_SPECIFICATIONS["test"]["split"] == "test"
    with pytest.raises(pb.Phase11BankError):
        pb.bank_specification("train")


def test_case_cell_decomposition_is_cell_major():
    assert pb.case_cell(0, 32) == (OPPONENT_STRATA[0], SOURCE_P10D, 0)
    assert pb.case_cell(31, 32) == (OPPONENT_STRATA[0], SOURCE_P10D, 31)
    assert pb.case_cell(32, 32) == (OPPONENT_STRATA[0], SOURCE_NEUTRAL, 0)
    assert pb.case_cell(64, 32) == (OPPONENT_STRATA[1], SOURCE_P10D, 0)
    assert pb.case_cell(511, 32) == (OPPONENT_STRATA[7], SOURCE_NEUTRAL, 31)
    with pytest.raises(pb.Phase11BankError):
        pb.case_cell(512, 32)
    with pytest.raises(pb.Phase11BankError):
        pb.case_cell(-1, 32)


def test_every_cell_is_represented_and_ids_parse(sample_cases):
    cells = {(case.stratum, case.setup_source) for case in sample_cases}
    assert len(cells) == 16
    for case in sample_cases:
        fields = parse_phase11_case_id(case.case_id)
        assert fields["stratum"] == case.stratum
        assert fields["setup_source"] == case.setup_source
        assert fields["case_ordinal"] == case.case_ordinal == 7


def test_case_games_carry_the_frozen_colour_pairing_and_seeds(sample_cases):
    for case in sample_cases:
        for game_index in (0, 1):
            game = case.games[game_index]
            assert game["observer_color"] == CASE_GAME_OBSERVER_COLOR[game_index]
            assert game["opponent_color"] == CASE_GAME_OPPONENT_COLOR[game_index]
            assert game["match_seed"] == game_match_seed(game["game_id"])
            for role in (ROLE_OBSERVER, ROLE_OPPONENT):
                record = game[role]
                assert record["setup_seed"] == case_setup_seed(
                    case.case_id, game_index, role
                )
                assert record["color"] == (
                    game["observer_color"] if role == ROLE_OBSERVER else game["opponent_color"]
                )


def test_observer_is_always_p10d_and_opponent_follows_the_cell(sample_cases):
    for case in sample_cases:
        for game in case.games.values():
            assert game[ROLE_OBSERVER]["source"] == SOURCE_P10D
            assert game[ROLE_OBSERVER]["candidate_id"] == "P10-D"
            assert game[ROLE_OPPONENT]["source"] == case.setup_source
            if case.setup_source == SOURCE_NEUTRAL:
                assert "branch" not in game[ROLE_OPPONENT]
            else:
                assert game[ROLE_OPPONENT]["branch"] in ("neutral", "learned")


def test_setups_are_split_correct_and_fingerprints_match(sample_cases):
    for case in sample_cases:
        for game in case.games.values():
            for role in (ROLE_OBSERVER, ROLE_OPPONENT):
                record = game[role]
                assert record["split"] == "validation"
                _, _, base_index = parse_base_setup_id(record["base_setup_id"])
                assert split_for_base_index(base_index) == "validation"
                assert (
                    content_fingerprint(tuple(record["setup"]))
                    == record["final_setup_fingerprint"]
                )


def test_p10d_draws_differ_across_colours(sample_cases):
    """Each seat draws conditioned on its own colour — never mirrored."""
    differing = 0
    for case in sample_cases:
        red = case.games[0][ROLE_OBSERVER]
        blue = case.games[1][ROLE_OBSERVER]
        if red["setup"] != blue["setup"]:
            differing += 1
    assert differing == len(sample_cases)


def test_isolated_rebuild_is_exact(sources, sample_cases):
    for case in sample_cases[:4]:
        assert pb.build_case("validation", case.case_index, sources) == case


def test_bank_digest_is_content_sensitive(sample_cases):
    digest = pb.bank_digest(sample_cases)
    assert digest == pb.bank_digest(sample_cases)
    assert digest != pb.bank_digest(sample_cases[:-1])


def test_test_bank_cases_build_structurally(sources):
    case = pb.build_case("test", 2_047, sources)
    assert case.bank_version == TEST_BANK_VERSION
    assert case.split == "test"
    assert case.stratum == OPPONENT_STRATA[7]
    assert case.setup_source == SOURCE_NEUTRAL
    for game in case.games.values():
        for role in (ROLE_OBSERVER, ROLE_OPPONENT):
            _, _, base_index = parse_base_setup_id(game[role]["base_setup_id"])
            assert split_for_base_index(base_index) == "test"


def test_selector_artifact_verification_guards_construction(tmp_path):
    problems = pb.Phase11SetupSources.verify_selector_artifacts()
    assert problems == []
    assert pb.Phase11SetupSources.verify_selector_artifacts(tmp_path)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def test_ledger_entry_schema_and_round_trip(tmp_path):
    entry = pb.ledger_entry(
        1, "banks", VALIDATION_BANK_VERSION, "structural_build", structural_only=True
    )
    assert tuple(entry) == (
        "ledger_version",
        "agent",
        "stage",
        "bank_version",
        "purpose",
        "structural_only",
        "neural_inference_count",
        "scored_prediction_count",
        "privileged_truth_count",
        "outcome_count",
    )
    pb.append_ledger_entries([entry], tmp_path)
    pb.append_ledger_entries(
        [
            pb.ledger_entry(
                1, "banks", TEST_BANK_VERSION, "structural_audit", structural_only=True
            )
        ],
        tmp_path,
    )
    entries = pb.read_ledger(tmp_path)
    assert len(entries) == 2
    assert entries[0]["bank_version"] == VALIDATION_BANK_VERSION
    sealed = pb.verify_test_bank_sealed(entries)
    assert sealed["test_bank_structural_only"]
    assert sealed["test_bank_entries"] == 1


def test_ledger_rejects_drifted_entries(tmp_path):
    entry = pb.ledger_entry(
        1, "banks", TEST_BANK_VERSION, "structural_build", structural_only=True
    )
    entry["extra_field"] = 1
    with pytest.raises(pb.Phase11BankError):
        pb.append_ledger_entries([entry], tmp_path)


def test_ledger_detects_scored_test_access(tmp_path):
    scored = pb.ledger_entry(
        7,
        "final",
        TEST_BANK_VERSION,
        "scored_evaluation",
        structural_only=False,
        scored_prediction_count=100,
    )
    pb.append_ledger_entries([scored], tmp_path)
    sealed = pb.verify_test_bank_sealed(pb.read_ledger(tmp_path))
    assert not sealed["test_bank_structural_only"]
    assert sealed["scored_prediction_total"] == 100


def test_ledger_rejects_foreign_versions(tmp_path):
    path = pb.ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ledger_version": "phase10_ledger"}) + "\n")
    with pytest.raises(pb.Phase11BankError):
        pb.read_ledger(tmp_path)
