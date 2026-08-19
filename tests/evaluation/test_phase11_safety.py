"""Phase 11 Agent 4: the hidden-truth permutation attack machinery.

The attack's whole claim rests on three properties of the code under test:
the alternative truth it builds is a *different* and *publicly legal*
truth, the trial-to-state walk is deterministic and skips nothing silently,
and the instrumented counter actually counts. Each is tested here directly,
on hand-built positions where the answer is known by construction, so a
regression fails in the suite instead of in a 50,000-trial run.
"""

import json

import numpy as np
import pytest

from stratego.engine.constants import BLUE, RED
from stratego.engine.legal_moves import legal_actions
from stratego.engine.observation import build_observation
from stratego.engine.pieces import PieceRecord
from stratego.engine.state import create_game
from stratego.engine.transition import apply_action
from stratego.evaluation.match_spec import EVALUATION_RULES
from stratego.evaluation.phase11_public_state import (
    build_public_state_document,
    public_state_identity,
)
from stratego.evaluation.phase11_safety import (
    INJECTION_FIELD_PROBES,
    MIN_UNRESOLVED_PIECES,
    PERMUTATION_ATTEMPT_BUDGET,
    STATE_SELECTION_WALK_LIMIT,
    HiddenAccessCounter,
    Phase11SafetyError,
    TracedPieceRecord,
    admits_alternative_truth,
    apply_alternative_truth,
    belief_digest,
    build_alternative_truth,
    injection_controls,
    instrument_hidden_types,
    sampler_request_digest,
    trial_sample_ordinal,
    trial_state_walk,
    unresolved_opponent_records,
    valid_transpositions,
    world_digest,
)
from stratego.evaluation.policy import build_public_view
from stratego.training.phase11_contract import (
    IMMOVABLE_RANK_INDICES,
    RANK_COUNT,
    RANK_INITIAL_COUNTS,
)
from stratego.training.phase11_seed import (
    SAFETY_TRIAL_COUNT,
    phase11_safety_trial_id,
)

FLAG_INDEX, BOMB_INDEX = IMMOVABLE_RANK_INDICES

#: One full army in rank-index order, the accepted test layout.
ARMY_LAYOUT = tuple(
    rank for rank in range(RANK_COUNT) for _ in range(RANK_INITIAL_COUNTS[rank])
)


def a_trial(ordinal=0):
    return phase11_safety_trial_id(ordinal)


# ---------------------------------------------------------------------------
# The alternative truth
# ---------------------------------------------------------------------------


def test_a_single_repeated_rank_admits_no_alternative():
    types = (5, 5, 5, 5)
    moved = (False,) * 4
    assert valid_transpositions(types, moved) == []
    assert admits_alternative_truth(types, moved) is False


def test_immobility_can_block_every_differing_pair():
    # One moved piece holding the only movable rank, one unmoved Flag: the
    # only differing pair would put the Flag on a piece that has moved.
    types = (3, FLAG_INDEX)
    moved = (True, False)
    assert valid_transpositions(types, moved) == []
    assert admits_alternative_truth(types, moved) is False


def test_a_differing_movable_pair_always_admits_an_alternative():
    types = (3, 7, FLAG_INDEX)
    moved = (True, True, False)
    assert (0, 1) in valid_transpositions(types, moved)
    assert admits_alternative_truth(types, moved) is True


def test_the_alternative_truth_changes_something_and_stays_legal():
    types = (0, 1, 2, 3, FLAG_INDEX, BOMB_INDEX)
    moved = (True, True, False, False, False, False)
    alternative = build_alternative_truth(a_trial(7), types, moved)
    assert sorted(alternative.ranks) == sorted(types)
    assert alternative.ranks != types
    assert alternative.changed_pieces >= 1
    for index, rank in enumerate(alternative.ranks):
        if moved[index]:
            assert rank not in IMMOVABLE_RANK_INDICES


def test_the_alternative_truth_is_a_pure_function_of_the_trial():
    types = (0, 1, 2, 3, 4, 5)
    moved = (False,) * 6
    first = build_alternative_truth(a_trial(11), types, moved)
    second = build_alternative_truth(a_trial(11), types, moved)
    assert first == second
    other = build_alternative_truth(a_trial(12), types, moved)
    assert other.ranks != first.ranks or other.attempts != first.attempts


def test_a_state_admitting_nothing_is_refused_rather_than_faked():
    with pytest.raises(Phase11SafetyError):
        build_alternative_truth(a_trial(3), (5, 5, 5), (False, False, False))


def test_the_transposition_fallback_is_exact_when_shuffles_cannot_win():
    # Every unmoved piece but one shares a rank, so the overwhelming
    # majority of shuffles are the identity on the ranks; the fallback must
    # still return a legal, different truth.
    types = (4,) * 12 + (9,)
    moved = (False,) * 13
    alternative = build_alternative_truth(a_trial(21), types, moved)
    assert sorted(alternative.ranks) == sorted(types)
    assert alternative.ranks != types
    assert alternative.attempts <= PERMUTATION_ATTEMPT_BUDGET + 1


def test_the_permutation_never_changes_the_inventory():
    types = tuple(ARMY_LAYOUT[:20])
    moved = tuple(rank not in IMMOVABLE_RANK_INDICES for rank in types)
    for ordinal in range(40):
        alternative = build_alternative_truth(a_trial(ordinal), types, moved)
        assert sorted(alternative.ranks) == sorted(types)


# ---------------------------------------------------------------------------
# The trial walk
# ---------------------------------------------------------------------------


def test_the_state_walk_is_deterministic_and_lands_on_an_admitting_state():
    admits = [index % 3 == 0 for index in range(30)]
    first = trial_state_walk(a_trial(5), len(admits), admits)
    assert first == trial_state_walk(a_trial(5), len(admits), admits)
    assert admits[first["pool_index"]] is True


def test_the_state_walk_reports_the_steps_it_had_to_take():
    # One admitting candidate in four: the walk is overwhelmingly likely to
    # need at least one skip somewhere in these ten trials, and must land
    # on the admitting index every time.
    admits = [index == 2 for index in range(4)]
    steps = []
    for ordinal in range(10):
        walk = trial_state_walk(a_trial(ordinal), len(admits), admits)
        assert walk["pool_index"] == 2
        steps.append(walk["walk_steps"])
    assert max(steps) >= 1


def test_the_state_walk_refuses_a_pool_that_admits_nothing():
    with pytest.raises(Phase11SafetyError):
        trial_state_walk(a_trial(1), 8, [False] * 8)
    with pytest.raises(Phase11SafetyError):
        trial_state_walk(a_trial(1), 0, [])


def test_the_walk_limit_is_a_ceiling_not_a_silent_drop():
    assert STATE_SELECTION_WALK_LIMIT > 0
    admits = [False] * 4
    with pytest.raises(Phase11SafetyError) as error:
        trial_state_walk(a_trial(2), len(admits), admits)
    assert str(STATE_SELECTION_WALK_LIMIT) in str(error.value)


def test_the_sample_ordinal_is_inside_the_frozen_world_range():
    for ordinal in range(64):
        assert 0 <= trial_sample_ordinal(a_trial(ordinal), 64) < 64


def test_every_frozen_trial_ordinal_has_an_identifier():
    assert SAFETY_TRIAL_COUNT == 50_000
    assert phase11_safety_trial_id(0).endswith("n=00000")
    assert phase11_safety_trial_id(SAFETY_TRIAL_COUNT - 1).endswith("n=49999")


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


def test_the_traced_record_counts_every_hidden_rank_read():
    counter = HiddenAccessCounter()
    record = TracedPieceRecord(
        piece_id=3,
        owner=BLUE,
        true_type=9,
        starting_square=91,
        current_square=91,
    )
    record.__dict__["__hidden_counter__"] = counter
    assert counter.reads == 0
    assert record.true_type == 9
    assert record.true_type == 9
    assert counter.reads == 2
    assert counter.report()["first_piece_ids"] == [3, 3]
    assert record.untraced_true_type() == 9
    assert counter.reads == 2


def test_an_untraced_record_is_an_ordinary_piece_record():
    record = TracedPieceRecord(
        piece_id=1, owner=RED, true_type=2, starting_square=0, current_square=0
    )
    assert isinstance(record, PieceRecord)
    assert record.true_type == 2
    assert record.is_movable_type is True


# ---------------------------------------------------------------------------
# The property under test, on a real position
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def a_played_position():
    """A real mid-game position with unresolved opponent pieces."""
    red = list(ARMY_LAYOUT)
    blue = list(reversed(ARMY_LAYOUT))
    state = create_game(red, blue, rules=EVALUATION_RULES, game_id="phase11-safety-test")
    for _ in range(24):
        if state.terminal:
            break
        apply_action(state, int(legal_actions(state)[0]))
    while not state.terminal and state.acting_player != RED:
        apply_action(state, int(legal_actions(state)[0]))
    return state


def public_bytes(state, observer):
    view = build_public_view(state, observer)
    observation = build_observation(state, observer)
    document = build_public_state_document(view, observation)
    return document, observation


def test_a_permuted_hidden_truth_leaves_every_public_byte_identical(a_played_position):
    state = a_played_position
    observer = RED
    document, observation = public_bytes(state, observer)
    records = unresolved_opponent_records(state, observer)
    assert len(records) >= MIN_UNRESOLVED_PIECES

    types = tuple(record.true_type for record in records)
    moved = tuple(record.has_moved for record in records)
    alternative = build_alternative_truth(a_trial(0), types, moved)
    permuted = apply_alternative_truth(state, observer, alternative.ranks)

    other_document, other_observation = public_bytes(permuted, observer)
    assert other_document == document
    assert public_state_identity(other_document) == public_state_identity(document)
    assert np.array_equal(other_observation, observation)
    assert alternative.changed_pieces >= 1


def test_building_the_public_products_reads_no_hidden_rank(a_played_position):
    state = a_played_position
    observer = RED
    document, _observation = public_bytes(state, observer)
    traced, counter = instrument_hidden_types(state, observer)
    traced_document, _ = public_bytes(traced, observer)
    assert traced_document == document
    assert counter.reads == 0


def test_instrumenting_does_not_disturb_the_original(a_played_position):
    state = a_played_position
    before = [record.true_type for record in state.pieces]
    traced, _counter = instrument_hidden_types(state, RED)
    for record in traced.pieces:
        record.true_type = 0
    assert [record.true_type for record in state.pieces] == before


def test_applying_an_alternative_truth_does_not_disturb_the_original(a_played_position):
    state = a_played_position
    before = [record.true_type for record in state.pieces]
    records = unresolved_opponent_records(state, RED)
    types = tuple(record.true_type for record in records)
    moved = tuple(record.has_moved for record in records)
    alternative = build_alternative_truth(a_trial(4), types, moved)
    apply_alternative_truth(state, RED, alternative.ranks)
    assert [record.true_type for record in state.pieces] == before


def test_the_alternative_truth_must_cover_exactly_the_unresolved_pieces(
    a_played_position,
):
    with pytest.raises(Phase11SafetyError):
        apply_alternative_truth(a_played_position, RED, (3, 4))


# ---------------------------------------------------------------------------
# Digests and injection controls
# ---------------------------------------------------------------------------


def test_the_belief_digest_moves_on_a_single_ulp():
    logits = {3: np.zeros(RANK_COUNT, dtype=np.float32)}
    probabilities = {3: np.full(RANK_COUNT, 1.0 / RANK_COUNT, dtype=np.float64)}
    masks = {3: tuple([1] * RANK_COUNT)}
    baseline = belief_digest(logits, probabilities, masks)
    moved = {3: probabilities[3].copy()}
    moved[3][0] = np.nextafter(moved[3][0], 1.0)
    assert belief_digest(logits, moved, masks) != baseline
    other_mask = {3: tuple([1] * 10 + [0, 0])}
    assert belief_digest(logits, probabilities, other_mask) != baseline


def test_the_world_digest_moves_on_a_single_provenance_field():
    world = {
        "sample_token": "t",
        "sampler_version": "belief_sampler_v1",
        "public_state_identity": "ab" * 32,
        "belief_model_label": "selfplay_c1_v1",
        "sample_ordinal": 0,
        "piece_order": [1, 2],
        "fallback_steps": [],
        "assignment": {1: 3, 2: 4},
    }
    baseline = world_digest(world)
    for field, value in (
        ("sample_ordinal", 1),
        ("piece_order", [2, 1]),
        ("assignment", {1: 4, 2: 3}),
    ):
        assert world_digest({**world, field: value}) != baseline


def test_the_sampler_request_digest_covers_document_probabilities_and_ordinal():
    document = {"document_version": "phase11_public_state_v1", "pieces": []}
    probabilities = {1: np.full(RANK_COUNT, 1.0 / RANK_COUNT)}
    baseline = sampler_request_digest(document, probabilities, 0)
    assert sampler_request_digest(document, probabilities, 1) != baseline
    other = {1: probabilities[1].copy()}
    other[1][0] = np.nextafter(other[1][0], 1.0)
    assert sampler_request_digest(document, other, 0) != baseline


def test_every_named_private_field_is_refused_at_both_boundaries(a_played_position):
    state = a_played_position
    observer = RED
    document, observation = public_bytes(state, observer)
    from stratego.evaluation.phase11_public_state import hidden_opponent_pieces

    probabilities = {
        int(piece["piece_slot"]): np.full(RANK_COUNT, 1.0 / RANK_COUNT)
        for piece in hidden_opponent_pieces(document)
    }
    report = injection_controls(document, observation, probabilities)
    assert report["injection_acceptances"] == 0
    assert report["all_rejected"] is True
    # Both boundaries, every named field, plus the two nested smuggles.
    assert report["probe_count"] == 2 * (len(INJECTION_FIELD_PROBES) + 2)
    boundaries = {probe["boundary"] for probe in report["probes"]}
    assert boundaries == {"belief", "sampler"}
    nested = [
        probe for probe in report["probes"] if probe["field"].startswith("nested_")
    ]
    assert len(nested) == 4
    assert all(probe["rejected"] for probe in nested)


def test_a_clean_request_still_builds_at_both_boundaries(a_played_position):
    """The controls must reject smuggling, not everything."""
    from stratego.evaluation.phase11_belief import Phase11BeliefRequest
    from stratego.evaluation.phase11_public_state import hidden_opponent_pieces
    from stratego.evaluation.phase11_sampler import Phase11SamplerRequest
    from stratego.training.phase11_contract import (
        BELIEF_REQUEST_VERSION,
        BELIEF_SAMPLER_VERSION,
    )

    document, observation = public_bytes(a_played_position, RED)
    probabilities = {
        int(piece["piece_slot"]): np.full(RANK_COUNT, 1.0 / RANK_COUNT).tolist()
        for piece in hidden_opponent_pieces(document)
    }
    belief = Phase11BeliefRequest.from_payload(
        {
            "request_version": BELIEF_REQUEST_VERSION,
            "request_id": "clean",
            "observer_color": "red",
            "public_state_document": document,
            "observation": observation,
        }
    )
    assert belief.public_state_identity == public_state_identity(document)
    sampler = Phase11SamplerRequest.from_payload(
        {
            "sampler_version": BELIEF_SAMPLER_VERSION,
            "public_state_document": document,
            "learned_probabilities": probabilities,
            "sample_ordinal": 0,
        }
    )
    assert sampler.sample_ordinal == 0


def test_the_belief_boundary_refuses_a_private_field_nested_in_a_piece(
    a_played_position,
):
    """The gap the Agent 4 injection controls found, kept closed."""
    from stratego.evaluation.phase11_belief import (
        Phase11BeliefError,
        Phase11BeliefRequest,
    )
    from stratego.training.phase11_contract import BELIEF_REQUEST_VERSION

    document, observation = public_bytes(a_played_position, RED)
    poisoned = json.loads(json.dumps(document))
    poisoned["pieces"][0]["true_rank_index"] = 9
    with pytest.raises(Phase11BeliefError):
        Phase11BeliefRequest.from_payload(
            {
                "request_version": BELIEF_REQUEST_VERSION,
                "request_id": "smuggle",
                "observer_color": "red",
                "public_state_document": poisoned,
                "observation": observation,
            }
        )
