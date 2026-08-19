"""The Phase 10 final-test evaluation module, checked against the freeze.

What these tests protect: the final-test game-identity scheme (test bank
version in every cell token, frozen match seeds as root seeds), the closure
of selection (no candidate but the permanently selected P10-D can reach the
sealed bank), and the arm rules the direct comparison depends on. They play
no game and load no model, so they are safe to run before and after the
Agent 7 evaluation alike.
"""

from __future__ import annotations

import pytest

from stratego.evaluation.phase10_final import (
    FINAL_BANK,
    FINAL_NEUTRAL_ARM_MATCHUPS,
    PHASE10_FINAL_VERSION,
    Phase10FinalError,
    SELECTED_CANDIDATE_ID,
    build_final_spec,
    final_cell_token,
)
from stratego.evaluation.phase10_validation import (
    ARM_LEARNED,
    ARM_NEUTRAL,
    PHASE10_VALIDATION_VERSION,
)
from stratego.training.phase10_contract import (
    CANDIDATE_IDS,
    MATCHUP_LEARNED_VS_NEUTRAL,
    MATCHUP_TOKENS,
    TEST_BANK_VERSION,
    VALIDATION_BANK_VERSION,
)


def test_the_final_bank_is_the_test_bank():
    assert FINAL_BANK == "test"


def test_the_final_version_is_distinct_from_the_validation_version():
    assert PHASE10_FINAL_VERSION != PHASE10_VALIDATION_VERSION


def test_the_selected_candidate_is_the_frozen_winner():
    assert SELECTED_CANDIDATE_ID == "P10-D"
    assert SELECTED_CANDIDATE_ID in CANDIDATE_IDS


def test_cell_tokens_carry_the_test_bank_version():
    for matchup in MATCHUP_TOKENS:
        token = final_cell_token(ARM_LEARNED, SELECTED_CANDIDATE_ID, matchup)
        assert token.startswith(TEST_BANK_VERSION + "|")
        assert VALIDATION_BANK_VERSION not in token
    for matchup in FINAL_NEUTRAL_ARM_MATCHUPS:
        token = final_cell_token(ARM_NEUTRAL, None, matchup)
        assert token.startswith(TEST_BANK_VERSION + "|")
        assert token.endswith(f"|neutral_v1|{matchup}")


def test_no_other_candidate_reaches_the_sealed_bank():
    for candidate_id in CANDIDATE_IDS:
        if candidate_id == SELECTED_CANDIDATE_ID:
            continue
        with pytest.raises(Phase10FinalError):
            final_cell_token(ARM_LEARNED, candidate_id, MATCHUP_TOKENS[0])
    with pytest.raises(Phase10FinalError):
        final_cell_token(ARM_LEARNED, None, MATCHUP_TOKENS[0])


def test_the_direct_matchup_has_no_separate_neutral_arm():
    assert MATCHUP_LEARNED_VS_NEUTRAL not in FINAL_NEUTRAL_ARM_MATCHUPS
    assert set(FINAL_NEUTRAL_ARM_MATCHUPS) == set(MATCHUP_TOKENS) - {
        MATCHUP_LEARNED_VS_NEUTRAL
    }
    with pytest.raises(Phase10FinalError):
        final_cell_token(ARM_NEUTRAL, None, MATCHUP_LEARNED_VS_NEUTRAL)


def test_the_neutral_arm_carries_no_candidate_id():
    with pytest.raises(Phase10FinalError):
        final_cell_token(ARM_NEUTRAL, SELECTED_CANDIDATE_ID, FINAL_NEUTRAL_ARM_MATCHUPS[0])


def test_unknown_arms_and_matchups_are_rejected():
    with pytest.raises(Phase10FinalError):
        final_cell_token("stress", None, MATCHUP_TOKENS[0])
    with pytest.raises(Phase10FinalError):
        final_cell_token(ARM_LEARNED, SELECTED_CANDIDATE_ID, "vs_nobody")


class _CaseStub:
    """The minimal shape build_final_spec reads, with a steerable bank."""

    def __init__(self, bank_version):
        self.bank_version = bank_version
        self.case_id = f"{bank_version}|ms=2026081801|f=F00|c=000"
        self.case_index = 0


def test_final_specs_refuse_validation_cases():
    from stratego.evaluation.registry import policy_ref

    ref = policy_ref("random_legal")
    with pytest.raises(Phase10FinalError):
        build_final_spec(
            _CaseStub(VALIDATION_BANK_VERSION),
            0,
            MATCHUP_TOKENS[0],
            arm=ARM_LEARNED,
            candidate_id=SELECTED_CANDIDATE_ID,
            own_ref=ref,
            opponent_ref=ref,
        )


def test_final_specs_descend_from_the_frozen_case_match_seed():
    from stratego.evaluation.registry import policy_ref
    from stratego.training.phase10_seed import case_match_seed

    ref = policy_ref("random_legal")
    case = _CaseStub(TEST_BANK_VERSION)
    for matchup in MATCHUP_TOKENS:
        for game_index in (0, 1):
            spec = build_final_spec(
                case,
                game_index,
                matchup,
                arm=ARM_LEARNED,
                candidate_id=SELECTED_CANDIDATE_ID,
                own_ref=ref,
                opponent_ref=ref,
            )
            assert spec.root_seed == case_match_seed(case.case_id, game_index, matchup)
            assert spec.suite_version == PHASE10_FINAL_VERSION
            assert spec.setup_bank_version == final_cell_token(
                ARM_LEARNED, SELECTED_CANDIDATE_ID, matchup
            )
            assert spec.replicate == game_index


def test_colour_pairing_is_the_frozen_one():
    from stratego.engine.constants import BLUE, RED
    from stratego.evaluation.registry import policy_ref

    ref = policy_ref("random_legal")
    case = _CaseStub(TEST_BANK_VERSION)
    spec0 = build_final_spec(
        case, 0, MATCHUP_TOKENS[0], arm=ARM_LEARNED,
        candidate_id=SELECTED_CANDIDATE_ID, own_ref=ref, opponent_ref=ref,
    )
    spec1 = build_final_spec(
        case, 1, MATCHUP_TOKENS[0], arm=ARM_LEARNED,
        candidate_id=SELECTED_CANDIDATE_ID, own_ref=ref, opponent_ref=ref,
    )
    assert spec0.candidate_color == RED
    assert spec1.candidate_color == BLUE
