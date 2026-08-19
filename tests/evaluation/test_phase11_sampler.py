"""Phase 11 Agent 3: the learned `belief_sampler_v1` and its boundary.

Documents here are built by hand, straight to the frozen
`phase11_public_state_v1` schema, so every test is independent of the
engine, the match runner and the recorded validation games. The identity of
a hand-built document is exactly what the sampler computes over it —
canonical JSON — which is the point: the sampler must behave as a pure
function of the document, wherever the document came from.
"""

import json

import numpy as np
import pytest

from stratego.engine.constants import IMPLEMENTATION_VERSION, OBSERVATION_VERSION, RULES_VERSION
from stratego.evaluation.phase11_baselines import (
    remaining_counts,
    sample_world,
    validate_world,
)
from stratego.evaluation.phase11_public_state import public_state_identity
from stratego.evaluation.phase11_sampler import (
    Phase11SamplerDeadEndError,
    Phase11SamplerError,
    Phase11SamplerRequest,
    sample_belief_world,
    sampler_boundary_report,
)
from stratego.training.phase11_contract import (
    ALLOWED_SAMPLER_REQUEST_FIELDS,
    BELIEF_SAMPLER_VERSION,
    RANK_COUNT,
    RANK_INITIAL_COUNTS,
    SAMPLER_PROVENANCE_FIELDS,
)
from stratego.training.phase11_seed import parse_phase11_sample_token

#: A fixed slot -> rank layout of one full army, in rank-index order.
ARMY_LAYOUT = tuple(
    rank for rank in range(RANK_COUNT) for _ in range(RANK_INITIAL_COUNTS[rank])
)
assert len(ARMY_LAYOUT) == 40

OBSERVATION_SHA = "ab" * 32


def make_document(
    *,
    observer="red",
    known_slots=(),
    dead_slots=(),
    moved_slots=(),
    total_moves=50,
):
    """A complete frozen-schema document with a hand-controlled opponent.

    Opponent slot `i` truly holds `ARMY_LAYOUT[i]`; `known_slots` are
    revealed at that rank, `dead_slots` are captured (and therefore known,
    as the engine's combat reveal guarantees), `moved_slots` carry the
    public has-moved flag.
    """
    opponent = "blue" if observer == "red" else "red"
    dead = set(dead_slots)
    known = set(known_slots) | dead
    moved = set(moved_slots)
    pieces = []
    for slot in range(40):
        pieces.append(
            {
                "piece_slot": slot,
                "owner_color": observer,
                "alive": True,
                "current_square": slot,
                "has_moved": False,
                "known_to_observer": True,
                "known_rank_index": int(ARMY_LAYOUT[slot]),
                "starting_square": slot,
            }
        )
    for slot in range(40):
        pieces.append(
            {
                "piece_slot": slot,
                "owner_color": opponent,
                "alive": slot not in dead,
                "current_square": None if slot in dead else 60 + slot % 40,
                "has_moved": slot in moved,
                "known_to_observer": slot in known,
                "known_rank_index": int(ARMY_LAYOUT[slot]) if slot in known else None,
                "starting_square": 60 + slot % 40,
            }
        )
    return {
        "document_version": "phase11_public_state_v1",
        "observer_color": observer,
        "acting_player_color": observer,
        "total_moves": int(total_moves),
        "battleless_moves": 3,
        "rules_version": RULES_VERSION,
        "engine_version": IMPLEMENTATION_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "pieces": pieces,
        "recent_moves": [],
        "observation_sha256": OBSERVATION_SHA,
    }


def uniform_probabilities(document):
    from stratego.evaluation.phase11_public_state import hidden_opponent_pieces

    return {
        int(piece["piece_slot"]): np.full(RANK_COUNT, 1.0 / RANK_COUNT)
        for piece in hidden_opponent_pieces(document)
    }


def make_request(document, ordinal=0, probabilities=None):
    return Phase11SamplerRequest(
        sampler_version=BELIEF_SAMPLER_VERSION,
        public_state_document=document,
        learned_probabilities=(
            uniform_probabilities(document) if probabilities is None else probabilities
        ),
        sample_ordinal=ordinal,
    )


# ---------------------------------------------------------------------------
# The request boundary
# ---------------------------------------------------------------------------


def test_the_request_fields_are_exactly_the_frozen_allowlist():
    document = make_document()
    request = make_request(document)
    assert tuple(request.__dataclass_fields__) == ALLOWED_SAMPLER_REQUEST_FIELDS
    report = sampler_boundary_report()
    assert report["allowed_request_fields"] == list(ALLOWED_SAMPLER_REQUEST_FIELDS)
    assert report["request_type_rejects_truth"]
    assert not report["request_type_field_for_truth_exists"]


@pytest.mark.parametrize(
    "field",
    [
        "true_rank",
        "true_rank_index",
        "private_piece_table",
        "opponent_setup",
        "hidden_start_rank",
        "winner",
        "match_result",
        "reward",
        "future_actions",
        "storage_path",
    ],
)
def test_every_rejected_input_is_refused_by_name(field):
    document = make_document()
    payload = {
        "sampler_version": BELIEF_SAMPLER_VERSION,
        "public_state_document": document,
        "learned_probabilities": uniform_probabilities(document),
        "sample_ordinal": 0,
        field: {"0": 5},
    }
    with pytest.raises(Phase11SamplerError, match="outside the frozen allowlist"):
        Phase11SamplerRequest.from_payload(payload)


def test_a_missing_field_is_refused_not_defaulted():
    document = make_document()
    payload = {
        "sampler_version": BELIEF_SAMPLER_VERSION,
        "public_state_document": document,
        "learned_probabilities": uniform_probabilities(document),
    }
    with pytest.raises(Phase11SamplerError, match="missing"):
        Phase11SamplerRequest.from_payload(payload)


def test_the_request_type_serves_the_learned_sampler_only():
    document = make_document()
    with pytest.raises(Phase11SamplerError, match="count-uniform"):
        Phase11SamplerRequest(
            sampler_version="count_uniform_world_sampler_v1",
            public_state_document=document,
            learned_probabilities=uniform_probabilities(document),
            sample_ordinal=0,
        )


def test_a_forbidden_token_inside_learned_probabilities_is_refused():
    document = make_document()
    probabilities = {str(key): value for key, value in uniform_probabilities(document).items()}
    probabilities["true_ranks"] = [1.0] * RANK_COUNT
    payload = {
        "sampler_version": BELIEF_SAMPLER_VERSION,
        "public_state_document": document,
        "learned_probabilities": probabilities,
        "sample_ordinal": 0,
    }
    with pytest.raises(Phase11SamplerError, match="forbidden key"):
        Phase11SamplerRequest.from_payload(payload)


def test_a_document_with_an_extra_field_is_refused():
    document = make_document()
    document["result_hint"] = "win"
    with pytest.raises(Phase11SamplerError, match="frozen schema"):
        make_request(document)


def test_a_document_piece_with_a_smuggled_field_is_refused():
    document = make_document()
    document["pieces"][41]["true_type"] = 9
    with pytest.raises(Phase11SamplerError, match="forbidden tokens: true"):
        make_request(document)


def frozen_move_entry():
    return {
        "ply": 4,
        "player_color": "blue",
        "piece_slot": 12,
        "piece_owner_color": "blue",
        "source": 61,
        "destination": 51,
        "was_attack": True,
        "target_piece_slot": 3,
        "target_owner_color": "red",
    }


def test_the_frozen_move_schema_is_accepted_including_target_fields():
    """`target_piece_slot`/`target_owner_color` are frozen public fields.

    They describe the publicly attacked piece, so the boundary must accept
    them even though their names contain the substring 'target'.
    """
    document = make_document(moved_slots=(12,))
    document["recent_moves"] = [frozen_move_entry()]
    world = sample_belief_world(make_request(document))
    assert validate_world(document, world)["valid"]


def test_a_move_entry_with_a_smuggled_field_is_refused():
    document = make_document(moved_slots=(12,))
    move = frozen_move_entry()
    move["attacker_true_rank"] = 9
    document["recent_moves"] = [move]
    with pytest.raises(Phase11SamplerError, match="recent-move fields"):
        make_request(document)


def test_probability_rows_must_cover_exactly_the_unresolved_pieces():
    document = make_document(known_slots=(0, 1))
    probabilities = uniform_probabilities(document)
    removed = probabilities.pop(sorted(probabilities)[0])
    with pytest.raises(Phase11SamplerError, match="exactly the unresolved"):
        make_request(document, probabilities=probabilities)
    probabilities[sorted(uniform_probabilities(document))[0]] = removed
    probabilities[0] = removed  # slot 0 is publicly known -> not unresolved
    with pytest.raises(Phase11SamplerError, match="exactly the unresolved"):
        make_request(document, probabilities=probabilities)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.25])
def test_a_nonfinite_or_negative_probability_row_is_refused(bad):
    document = make_document()
    probabilities = uniform_probabilities(document)
    slot = sorted(probabilities)[3]
    probabilities[slot] = np.array([bad] + [0.1] * (RANK_COUNT - 1))
    with pytest.raises(Phase11SamplerError, match="probability row"):
        make_request(document, probabilities=probabilities)


# ---------------------------------------------------------------------------
# The frozen walk
# ---------------------------------------------------------------------------


def test_a_sampled_world_is_complete_legal_and_validated():
    document = make_document(
        known_slots=(2, 3), dead_slots=(10, 11), moved_slots=(20, 21, 22)
    )
    world = sample_belief_world(make_request(document, ordinal=7))
    assert tuple(world)[: len(SAMPLER_PROVENANCE_FIELDS)] == SAMPLER_PROVENANCE_FIELDS
    check = validate_world(document, world)
    assert check["valid"], check["findings"]
    assert all(value == 0 for value in check["counters"].values())
    # Known and dead slots are never assigned; the multiset is exact.
    assert set(world["assignment"]).isdisjoint({2, 3, 10, 11})
    observed = [0] * RANK_COUNT
    for rank in world["assignment"].values():
        observed[rank] += 1
    assert tuple(observed) == remaining_counts(document)
    fields = parse_phase11_sample_token(world["sample_token"])
    assert fields["sampler_version"] == BELIEF_SAMPLER_VERSION
    assert fields["sample_ordinal"] == 7
    assert fields["public_state_identity"] == public_state_identity(document)


def test_moved_pieces_never_take_flag_or_bomb():
    document = make_document(moved_slots=tuple(range(1, 9)))
    for ordinal in range(24):
        world = sample_belief_world(make_request(document, ordinal=ordinal))
        for slot in range(1, 9):
            assert world["assignment"][slot] not in (10, 11)


def test_the_sampler_is_a_pure_function_of_its_identity_inputs():
    document = make_document(known_slots=(4,), moved_slots=(5, 6))
    first = sample_belief_world(make_request(document, ordinal=3))
    again = sample_belief_world(make_request(document, ordinal=3))
    assert first == again
    # A canonical-JSON round trip of the document changes nothing.
    round_tripped = json.loads(
        json.dumps(document, sort_keys=True, separators=(",", ":"))
    )
    rebuilt = sample_belief_world(make_request(round_tripped, ordinal=3))
    assert rebuilt == first


def test_different_ordinals_give_domain_separated_streams():
    document = make_document()
    first = sample_belief_world(make_request(document, ordinal=0))
    second = sample_belief_world(make_request(document, ordinal=1))
    assert first["sample_token"] != second["sample_token"]
    assert first["assignment"] != second["assignment"]


def test_learned_weighting_actually_weights():
    """A one-hot learned row forces its rank wherever it stays legal."""
    document = make_document(known_slots=tuple(range(8, 40)))
    # Unresolved slots 0..7: spy (slot 0) and seven scouts under ARMY_LAYOUT;
    # remaining inventory is {spy: 1, scout: 7}.
    probabilities = {}
    for slot in range(8):
        row = np.zeros(RANK_COUNT)
        row[1] = 1.0  # all mass on scout
        probabilities[slot] = row
    world = sample_belief_world(make_request(document, probabilities=probabilities))
    # Exactly one piece must still take the spy (inventory forces it), and
    # it does so through the zero-mass fallback at the walk's last step.
    ranks = sorted(world["assignment"].values())
    assert ranks == [0] + [1] * 7
    assert len(world["fallback_steps"]) == 1
    assert world["fallback_steps"][0] == 7


# ---------------------------------------------------------------------------
# The completion-feasibility guard
# ---------------------------------------------------------------------------


def scout_and_bombs_document():
    """Unresolved: two unmoved pieces and one moved; inventory scout + 2 bombs.

    The contract's dead-end scenario: the single movable rank must reach
    the moved piece, so the guard has to deny it to both unmoved pieces.
    ARMY_LAYOUT: slot 1 is a scout (moved), slots 35, 36 are bombs (unmoved,
    under layout ranks: slots 34..39 are bombs; slot 33 flag).
    """
    known = tuple(slot for slot in range(40) if slot not in (1, 35, 36))
    return make_document(known_slots=known, moved_slots=(1,))


def test_the_guard_routes_the_last_movable_rank_to_the_moved_piece():
    document = scout_and_bombs_document()
    counts = remaining_counts(document)
    assert counts[1] == 1 and counts[11] == 2 and sum(counts) == 3
    for ordinal in range(64):
        world = sample_belief_world(make_request(document, ordinal=ordinal))
        assert world["assignment"][1] == 1  # the moved piece takes the scout
        assert world["assignment"][35] == 11
        assert world["assignment"][36] == 11
        assert validate_world(document, world)["valid"]


def test_without_the_guard_the_same_state_can_dead_end():
    """The guard is load-bearing: an unguarded walk dies on this state."""
    document = scout_and_bombs_document()
    counts = list(remaining_counts(document))
    # Unguarded: an unmoved piece may take the scout. Then the moved piece
    # sees only bombs, which its mask forbids -> the legal set is empty.
    counts[1] -= 1  # an unmoved piece took the scout
    moved_available = [
        rank for rank in range(RANK_COUNT) if rank not in (10, 11) and counts[rank] > 0
    ]
    assert moved_available == []


def test_zero_mass_fallback_uses_counts_over_the_same_legal_set():
    document = scout_and_bombs_document()
    # All learned mass on the scout: for the unmoved pieces the guard makes
    # the legal set {bomb}, whose learned mass is zero -> step 10 falls back
    # to counts and still completes.
    probabilities = {}
    for slot in (1, 35, 36):
        row = np.zeros(RANK_COUNT)
        row[1] = 1.0
        probabilities[slot] = row
    for ordinal in range(16):
        world = sample_belief_world(
            make_request(document, ordinal=ordinal, probabilities=probabilities)
        )
        assert world["assignment"][1] == 1
        assert world["assignment"][35] == 11 and world["assignment"][36] == 11
        # The two unmoved steps had zero learned mass on their legal set.
        assert len(world["fallback_steps"]) == 2


def test_a_publicly_inconsistent_state_raises_instead_of_inventing():
    # One unresolved *moved* piece whose only remaining rank is a bomb: no
    # legal world exists, and the sampler must say so loudly.
    known = tuple(slot for slot in range(40) if slot not in (0, 35))
    document = make_document(known_slots=known, dead_slots=(0,), moved_slots=(35,))
    # dead spy keeps conservation exact: unresolved = {35}, counts = {bomb: 1}
    with pytest.raises(Phase11SamplerDeadEndError):
        sample_belief_world(make_request(document, ordinal=0))


# ---------------------------------------------------------------------------
# Provenance and the shared skeleton
# ---------------------------------------------------------------------------


def test_validate_world_catches_corrupted_provenance():
    document = make_document(moved_slots=(7,))
    world = sample_belief_world(make_request(document, ordinal=2))
    tampered = dict(world, sample_ordinal=3)
    check = validate_world(document, tampered)
    assert check["counters"]["provenance_mismatches"] >= 1

    other = make_document(moved_slots=(7, 8))
    check = validate_world(other, world)
    assert check["counters"]["provenance_mismatches"] >= 1


def test_the_learned_and_count_samplers_share_identity_streams_not_draws():
    document = make_document(moved_slots=(3,))
    learned = sample_belief_world(make_request(document, ordinal=0))
    baseline = sample_world(document, 0)
    assert learned["sample_token"] != baseline["sample_token"]
    assert parse_phase11_sample_token(baseline["sample_token"])["sampler_version"] == (
        "count_uniform_world_sampler_v1"
    )
    assert validate_world(document, baseline)["valid"]
