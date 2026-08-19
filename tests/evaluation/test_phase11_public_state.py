"""Phase 11 Agent 2: the observer-legal public-state document."""

import numpy as np
import pytest

from stratego.engine.constants import BLUE, PIECE_TYPE_NAMES, RED, SETUP_SQUARES
from stratego.engine.events import public_setup_view
from stratego.engine.observation import build_observation
from stratego.engine.pieces import piece_id_from_name
from stratego.engine.state import create_game
from stratego.engine.transition import apply_action
from stratego.engine.legal_moves import legal_actions
from stratego.evaluation.match_spec import EVALUATION_RULES
from stratego.evaluation.phase11_public_state import (
    PUBLIC_START_SQUARES,
    Phase11PublicStateError,
    build_public_state_document,
    document_progress_bucket,
    hidden_opponent_pieces,
    legal_rank_mask,
    observation_digest,
    public_state_identity,
)
from stratego.evaluation.policy import build_public_view
from stratego.setups.contracts import LIBRARY_JSONL_PATH
from stratego.setups.library import read_library_jsonl
from stratego.training.phase11_contract import (
    PUBLIC_PIECE_FIELDS,
    PUBLIC_STATE_DOCUMENT_FIELDS,
    PUBLIC_STATE_DOCUMENT_VERSION,
)


@pytest.fixture(scope="module")
def two_setups():
    entries = read_library_jsonl(LIBRARY_JSONL_PATH)
    return tuple(entries[0].canonical_setup), tuple(entries[1].canonical_setup)


@pytest.fixture(scope="module")
def opening(two_setups):
    red, blue = two_setups
    return create_game(red, blue, rules=EVALUATION_RULES, game_id="phase11-public-state")


def document_for(state, observer):
    return build_public_state_document(
        build_public_view(state, observer), build_observation(state, observer)
    )


def test_start_square_table_matches_the_engine_setup_squares():
    for owner in (RED, BLUE):
        assert PUBLIC_START_SQUARES[owner] == tuple(SETUP_SQUARES[owner])
        assert len(PUBLIC_START_SQUARES[owner]) == 40


def test_start_squares_agree_with_the_engine_public_setup_view(opening):
    """The engine's own statement of what the observer sees about the start."""
    for observer, opponent in ((RED, BLUE), (BLUE, RED)):
        occupancy = public_setup_view(opening, observer)["opponent_setup_occupancy"]
        for square_name, piece_name in occupancy.items():
            piece_id = piece_id_from_name(piece_name)
            slot = piece_id % 40
            record = opening.pieces[piece_id]
            assert record.owner == opponent
            assert PUBLIC_START_SQUARES[opponent][slot] == record.starting_square


def test_document_has_exactly_the_frozen_fields(opening):
    document = document_for(opening, RED)
    assert tuple(document) == PUBLIC_STATE_DOCUMENT_FIELDS
    assert document["document_version"] == PUBLIC_STATE_DOCUMENT_VERSION
    assert len(document["pieces"]) == 80
    for piece in document["pieces"]:
        assert tuple(piece) == PUBLIC_PIECE_FIELDS


def test_document_conceals_every_opponent_rank_at_the_opening(opening):
    document = document_for(opening, RED)
    opponents = [p for p in document["pieces"] if p["owner_color"] == "blue"]
    assert len(opponents) == 40
    assert all(piece["known_rank_index"] is None for piece in opponents)
    assert all(not piece["known_to_observer"] for piece in opponents)
    own = [p for p in document["pieces"] if p["owner_color"] == "red"]
    assert all(piece["known_rank_index"] is not None for piece in own)


def test_document_is_invariant_under_a_hidden_truth_permutation(two_setups):
    """The purity claim, tested where it matters: permute the hidden army."""
    red, blue = two_setups
    left = create_game(red, blue, rules=EVALUATION_RULES, game_id="g")
    baseline = document_for(left, RED)
    baseline_identity = public_state_identity(baseline)

    # Reassign the hidden (blue) true types among blue's pieces. Nothing the
    # observer may legally see changes, so neither may the document.
    right = create_game(red, blue, rules=EVALUATION_RULES, game_id="g")
    blue_records = [record for record in right.pieces if record.owner == BLUE]
    types = [record.true_type for record in blue_records]
    for record, piece_type in zip(blue_records, list(reversed(types))):
        record.true_type = piece_type
    assert document_for(right, RED) == baseline
    assert public_state_identity(document_for(right, RED)) == baseline_identity


def test_identity_changes_when_a_public_fact_changes(two_setups):
    red, blue = two_setups
    state = create_game(red, blue, rules=EVALUATION_RULES, game_id="g")
    before = public_state_identity(document_for(state, RED))
    apply_action(state, legal_actions(state)[0])
    after = public_state_identity(document_for(state, BLUE))
    assert before != after


def test_identity_covers_the_observation(opening):
    document = document_for(opening, RED)
    tampered = dict(document)
    tampered["observation_sha256"] = "0" * 64
    assert public_state_identity(tampered) != public_state_identity(document)


def test_observation_digest_rejects_a_wrong_shape():
    with pytest.raises(Phase11PublicStateError):
        observation_digest(np.zeros((3, 3), dtype=np.float32))


def test_observation_digest_rejects_non_finite():
    bad = np.zeros((127, 10, 10), dtype=np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(Phase11PublicStateError):
        observation_digest(bad)


def test_legal_rank_mask_records_movement_impossibility_only():
    assert legal_rank_mask(False) == (1,) * 12
    moved = legal_rank_mask(True)
    assert moved[PIECE_TYPE_NAMES.index("flag")] == 0
    assert moved[PIECE_TYPE_NAMES.index("bomb")] == 0
    assert sum(moved) == 10


def test_hidden_targets_are_live_unknown_opponent_pieces(opening):
    document = document_for(opening, RED)
    hidden = hidden_opponent_pieces(document)
    assert len(hidden) == 40
    assert [piece["piece_slot"] for piece in hidden] == sorted(
        piece["piece_slot"] for piece in hidden
    )
    assert all(piece["owner_color"] == "blue" for piece in hidden)


def test_progress_buckets_follow_the_frozen_thresholds(opening):
    document = document_for(opening, RED)
    assert document_progress_bucket({**document, "total_moves": 0}) == "early"
    assert document_progress_bucket({**document, "total_moves": 39}) == "early"
    assert document_progress_bucket({**document, "total_moves": 40}) == "middle"
    assert document_progress_bucket({**document, "total_moves": 119}) == "middle"
    assert document_progress_bucket({**document, "total_moves": 120}) == "late"


def test_public_state_identity_refuses_a_foreign_document():
    with pytest.raises(Phase11PublicStateError):
        public_state_identity({"document_version": "something_else"})
