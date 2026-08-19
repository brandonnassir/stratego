"""Phase 11 Agent 2: `remaining_count_belief_v1` and the world baseline."""

import numpy as np
import pytest

from stratego.engine.constants import BLUE, PIECE_TYPE_NAMES, RED
from stratego.engine.observation import build_observation
from stratego.engine.state import create_game
from stratego.evaluation.match_spec import EVALUATION_RULES
from stratego.evaluation.phase11_baselines import (
    Phase11BaselineError,
    check_count_conservation,
    feasible_ranks,
    inverse_cdf_choice,
    piece_masks,
    remaining_count_belief,
    remaining_count_distribution,
    remaining_counts,
    sample_world,
    validate_world,
)
from stratego.evaluation.phase11_public_state import (
    PUBLIC_START_SQUARES,
    build_public_state_document,
    hidden_opponent_pieces,
    legal_rank_mask,
)
from stratego.evaluation.policy import build_public_view
from stratego.setups.contracts import LIBRARY_JSONL_PATH
from stratego.setups.library import read_library_jsonl
from stratego.training.phase11_contract import RANK_INITIAL_COUNTS, RANK_NAMES

FLAG = RANK_NAMES.index("flag")
BOMB = RANK_NAMES.index("bomb")
SCOUT = RANK_NAMES.index("scout")
MINER = RANK_NAMES.index("miner")


@pytest.fixture(scope="module")
def opening():
    entries = read_library_jsonl(LIBRARY_JSONL_PATH)
    return create_game(
        tuple(entries[0].canonical_setup),
        tuple(entries[1].canonical_setup),
        rules=EVALUATION_RULES,
        game_id="phase11-baselines",
    )


@pytest.fixture(scope="module")
def opening_document(opening):
    return build_public_state_document(
        build_public_view(opening, RED), build_observation(opening, RED)
    )


def synthetic_document(opponent_pieces, observer="red"):
    """A minimal but *schema-exact* document, for the constructed edge cases.

    `opponent_pieces` is a list of `(alive, has_moved, known_rank_index)`
    of length 40, in setup-slot order. The observer's own forty pieces are
    filled in as alive, unmoved and known, which is what they always are.
    """
    opponent = "blue" if observer == "red" else "red"
    owner_id = BLUE if opponent == "blue" else RED
    observer_id = RED if observer == "red" else BLUE
    pieces = []
    for slot in range(40):
        pieces.append(
            {
                "piece_slot": slot,
                "owner_color": observer,
                "alive": True,
                "current_square": PUBLIC_START_SQUARES[observer_id][slot],
                "has_moved": False,
                "known_to_observer": True,
                "known_rank_index": 0,
                "starting_square": PUBLIC_START_SQUARES[observer_id][slot],
            }
        )
    for slot, (alive, has_moved, known) in enumerate(opponent_pieces):
        pieces.append(
            {
                "piece_slot": slot,
                "owner_color": opponent,
                "alive": bool(alive),
                "current_square": (
                    PUBLIC_START_SQUARES[owner_id][slot] if alive else None
                ),
                "has_moved": bool(has_moved),
                "known_to_observer": known is not None,
                "known_rank_index": known,
                "starting_square": PUBLIC_START_SQUARES[owner_id][slot],
            }
        )
    return {
        "document_version": "phase11_public_state_v1",
        "observer_color": observer,
        "acting_player_color": observer,
        "total_moves": 60,
        "battleless_moves": 0,
        "rules_version": "stratego_project_v1",
        "engine_version": "phase2_1_reference_1.2.0",
        "observation_version": "observation_v2_1_127ch",
        "pieces": pieces,
        "recent_moves": [],
        "observation_sha256": "0" * 64,
    }


def army(hidden_ranks, *, moved=()):
    """40 opponent-piece descriptors: `hidden_ranks` stay hidden, rest known."""
    remaining = list(hidden_ranks)
    known_pool = []
    counts = list(RANK_INITIAL_COUNTS)
    for rank in remaining:
        counts[rank] -= 1
    for rank, count in enumerate(counts):
        known_pool.extend([rank] * count)
    pieces = []
    for slot in range(40):
        if slot < len(remaining):
            pieces.append((True, slot in moved, None))
        else:
            pieces.append((True, False, known_pool[slot - len(remaining)]))
    return pieces, remaining


# ---------------------------------------------------------------------------
# The inventory
# ---------------------------------------------------------------------------


def test_opening_inventory_is_the_full_army(opening_document):
    assert remaining_counts(opening_document) == RANK_INITIAL_COUNTS
    assert check_count_conservation(opening_document)["conserved"]


def test_inventory_matches_the_engines_own_public_view(opening):
    """A second, accepted implementation of the same public quantity."""
    for observer in (RED, BLUE):
        view = build_public_view(opening, observer)
        document = build_public_state_document(view, build_observation(opening, observer))
        assert remaining_counts(document) == tuple(view.unresolved_opponent_counts)


def test_revealed_ranks_leave_the_inventory(opening_document):
    document = dict(opening_document)
    pieces = [dict(piece) for piece in document["pieces"]]
    revealed = 0
    for piece in pieces:
        if piece["owner_color"] == "blue" and revealed < 3:
            piece["known_to_observer"] = True
            piece["known_rank_index"] = SCOUT
            revealed += 1
    document["pieces"] = pieces
    counts = remaining_counts(document)
    assert counts[SCOUT] == RANK_INITIAL_COUNTS[SCOUT] - 3
    assert check_count_conservation(document, counts)["conserved"]


def test_captured_pieces_leave_the_inventory_too(opening_document):
    """Every captured piece was revealed by the combat that killed it."""
    document = dict(opening_document)
    pieces = [dict(piece) for piece in document["pieces"]]
    killed = 0
    for piece in pieces:
        if piece["owner_color"] == "blue" and killed < 2:
            piece["alive"] = False
            piece["current_square"] = None
            piece["known_to_observer"] = True
            piece["known_rank_index"] = MINER
            killed += 1
    document["pieces"] = pieces
    counts = remaining_counts(document)
    assert counts[MINER] == RANK_INITIAL_COUNTS[MINER] - 2
    assert sum(counts) == 38 == len(hidden_opponent_pieces(document))
    assert check_count_conservation(document, counts)["conserved"]


def test_a_negative_inventory_is_refused():
    pieces, _ = army([SCOUT])
    pieces = [(True, False, SCOUT) for _ in range(40)]
    document = synthetic_document(pieces)
    with pytest.raises(Phase11BaselineError):
        remaining_counts(document)


# ---------------------------------------------------------------------------
# The distribution
# ---------------------------------------------------------------------------


def test_unmoved_piece_gets_the_full_count_proportional_distribution(opening_document):
    distributions = remaining_count_belief(opening_document)
    assert len(distributions) == 40
    row = next(iter(distributions.values()))
    expected = np.asarray(RANK_INITIAL_COUNTS, dtype=np.float64) / 40.0
    assert np.allclose(row, expected, atol=0, rtol=0)
    assert row.sum() == pytest.approx(1.0, abs=1e-15)


def test_moved_unknown_excludes_flag_and_bomb(opening_document):
    document = dict(opening_document)
    pieces = [dict(piece) for piece in document["pieces"]]
    for piece in pieces:
        if piece["owner_color"] == "blue" and piece["piece_slot"] == 0:
            piece["has_moved"] = True
    document["pieces"] = pieces
    row = remaining_count_belief(document)[0]
    assert row[FLAG] == 0.0
    assert row[BOMB] == 0.0
    assert row.sum() == pytest.approx(1.0, abs=1e-15)
    movable = 40 - RANK_INITIAL_COUNTS[FLAG] - RANK_INITIAL_COUNTS[BOMB]
    assert row[SCOUT] == pytest.approx(RANK_INITIAL_COUNTS[SCOUT] / movable, abs=1e-15)
    assert piece_masks(document)[0] == legal_rank_mask(True)


def test_near_endgame_exhaustion_zeroes_exhausted_ranks():
    pieces, hidden = army([SCOUT, MINER, FLAG])
    document = synthetic_document(pieces)
    counts = remaining_counts(document)
    assert sum(counts) == 3
    assert counts[SCOUT] == 1 and counts[MINER] == 1 and counts[FLAG] == 1
    row = remaining_count_belief(document)[0]
    assert row[SCOUT] == pytest.approx(1 / 3)
    assert sum(row[rank] for rank in range(12) if counts[rank] == 0) == 0.0


def test_one_legal_rank_case_is_a_point_mass():
    """A moved piece with only one movable rank left must be certain."""
    pieces, _ = army([SCOUT], moved={0})
    document = synthetic_document(pieces)
    counts = remaining_counts(document)
    assert sum(counts) == 1 and counts[SCOUT] == 1
    mask = legal_rank_mask(True)
    legal = [rank for rank in range(12) if mask[rank] and counts[rank] > 0]
    assert legal == [SCOUT]
    row = remaining_count_belief(document)[0]
    assert row[SCOUT] == 1.0
    assert row.sum() == 1.0


def test_the_true_rank_always_has_positive_baseline_mass(opening_document):
    """The well-definedness claim, over every rank a hidden piece could be."""
    distributions = remaining_count_belief(opening_document)
    counts = remaining_counts(opening_document)
    for row in distributions.values():
        for rank in range(12):
            if counts[rank] > 0:
                assert row[rank] > 0.0 or legal_rank_mask(True)[rank] == 0


def test_a_masked_out_impossible_state_is_refused():
    with pytest.raises(Phase11BaselineError):
        remaining_count_distribution((0,) * 12, (1,) * 12)


def test_public_scout_deduction_makes_a_piece_known():
    """The engine's `scout_multisquare` reveal is a known rank, not a mask."""
    from stratego.engine.legal_moves import legal_actions
    from stratego.engine.transition import apply_action
    from stratego.engine.actions import decode_action

    entries = read_library_jsonl(LIBRARY_JSONL_PATH)
    found = False
    for index in range(0, 24, 2):
        state = create_game(
            tuple(entries[index].canonical_setup),
            tuple(entries[index + 1].canonical_setup),
            rules=EVALUATION_RULES,
            game_id="scout",
        )
        for _ in range(24):
            if state.terminal:
                break
            actions = legal_actions(state)
            multi = [
                action
                for action in actions
                if _distance(decode_action(action)) >= 2
            ]
            apply_action(state, multi[0] if multi else actions[0])
            if multi:
                observer = state.acting_player
                document = build_public_state_document(
                    build_public_view(state, observer), build_observation(state, observer)
                )
                revealed = [
                    piece
                    for piece in document["pieces"]
                    if piece["owner_color"] != document["observer_color"]
                    and piece["known_to_observer"]
                ]
                if revealed:
                    assert all(
                        piece["known_rank_index"] == SCOUT for piece in revealed
                    )
                    counts = remaining_counts(document)
                    assert counts[SCOUT] == RANK_INITIAL_COUNTS[SCOUT] - len(revealed)
                    assert check_count_conservation(document, counts)["conserved"]
                    found = True
                    break
        if found:
            break
    assert found, "no public Scout deduction was reachable in the sampled games"


def _distance(decoded):
    source, destination = decoded
    return max(
        abs(source // 10 - destination // 10), abs(source % 10 - destination % 10)
    )


# ---------------------------------------------------------------------------
# count_uniform_world_sampler_v1
# ---------------------------------------------------------------------------


def test_the_world_baseline_produces_a_legal_complete_world(opening_document):
    world = sample_world(opening_document, 0)
    check = validate_world(opening_document, world)
    assert check["valid"], check["findings"]
    assert len(world["assignment"]) == 40
    assert sorted(world["assignment"].values()) == sorted(
        rank for rank, count in enumerate(RANK_INITIAL_COUNTS) for _ in range(count)
    )


def test_the_world_baseline_is_a_pure_function_of_its_identity(opening_document):
    first = sample_world(opening_document, 7)
    again = sample_world(opening_document, 7)
    assert first == again
    other = sample_world(opening_document, 8)
    assert other["sample_token"] != first["sample_token"]


def test_the_world_baseline_respects_immobility(opening_document):
    document = dict(opening_document)
    pieces = [dict(piece) for piece in document["pieces"]]
    for piece in pieces:
        if piece["owner_color"] == "blue" and piece["piece_slot"] < 30:
            piece["has_moved"] = True
    document["pieces"] = pieces
    for ordinal in range(16):
        world = sample_world(document, ordinal)
        check = validate_world(document, world)
        assert check["valid"], check["findings"]
        for slot in range(30):
            assert world["assignment"][slot] not in (FLAG, BOMB)


def test_the_world_baseline_takes_no_learned_weighting(opening_document):
    with pytest.raises(Phase11BaselineError):
        sample_world(opening_document, 0, learned_probabilities={0: np.ones(12) / 12})


def test_the_feasibility_rule_keeps_movable_ranks_for_moved_pieces():
    counts = [0] * 12
    counts[SCOUT] = 1
    counts[BOMB] = 1
    legal = (SCOUT, BOMB)
    # One movable rank left and one moved piece still to place: an unmoved
    # piece may not take the last movable rank.
    kept = feasible_ranks(legal, counts, piece_has_moved=False, moved_unresolved_remaining=1)
    assert kept == (BOMB,)
    # With no moved piece left to place, the guard permits it again.
    kept = feasible_ranks(legal, counts, piece_has_moved=False, moved_unresolved_remaining=0)
    assert kept == (SCOUT, BOMB)
    # A moved piece needs no guard beyond its mask.
    kept = feasible_ranks((SCOUT,), counts, piece_has_moved=True, moved_unresolved_remaining=0)
    assert kept == (SCOUT,)


def test_the_feasibility_rule_prevents_a_dead_end():
    """The exact instance Agent 1's reading was written for."""
    pieces, _ = army([SCOUT, BOMB], moved={0})
    document = synthetic_document(pieces)
    counts = remaining_counts(document)
    assert sum(counts) == 2 and counts[SCOUT] == 1 and counts[BOMB] == 1
    for ordinal in range(32):
        world = sample_world(document, ordinal)
        assert world["assignment"][0] == SCOUT
        assert world["assignment"][1] == BOMB
        assert world["dead_end_events"] == 0
        assert validate_world(document, world)["valid"]


def test_inverse_cdf_walk_uses_the_last_rank_as_the_tail_guard():
    assert inverse_cdf_choice((1, 2, 3), np.array([1.0, 1.0, 1.0]), 0.0) == 1
    assert inverse_cdf_choice((1, 2, 3), np.array([1.0, 1.0, 1.0]), 0.5) == 2
    assert inverse_cdf_choice((1, 2, 3), np.array([1.0, 1.0, 1.0]), 1.0) == 3
    with pytest.raises(Phase11BaselineError):
        inverse_cdf_choice((1,), np.array([0.0]), 0.5)


def test_validate_world_catches_an_inventory_error(opening_document):
    world = sample_world(opening_document, 0)
    tampered = dict(world)
    assignment = dict(world["assignment"])
    slot = next(iter(assignment))
    assignment[slot] = (assignment[slot] + 1) % 12
    tampered["assignment"] = assignment
    check = validate_world(opening_document, tampered)
    assert not check["valid"]
    assert check["counters"]["inventory_errors"] == 1


def test_validate_world_catches_a_provenance_mismatch(opening_document):
    world = dict(sample_world(opening_document, 0))
    world["sample_ordinal"] = 1
    check = validate_world(opening_document, world)
    assert not check["valid"]
    assert check["counters"]["provenance_mismatches"] >= 1
