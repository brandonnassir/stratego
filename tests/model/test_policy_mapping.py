"""Policy index 〈-〉 engine action, audited exhaustively.

Covers Phase 5 gates 9 (`all_10000_actions_round_trip`) and 10
(`policy_index_matches_engine_action`).

Two separate claims are checked here, and they are not the same claim:

1. **Encoding.** For every one of the 10,000 identifiers,
   `decode -> (source, destination) -> encode` returns the original identifier.
   That is pure arithmetic and is checked over the whole space.
2. **Selection.** When the model's unique maximum sits on a legal index, the
   adapter selects *that* engine action -- and the engine then accepts it as the
   move from that source to that destination. That is checked over a corpus of
   real positions covering every legal move family.
"""

from __future__ import annotations

import numpy as np
import torch

from stratego.engine.actions import (
    action_destination,
    action_source,
    decode_action,
    encode_action,
)
from stratego.engine.constants import ACTION_SPACE_SIZE, NUM_SQUARES, SCOUT
from stratego.engine.legal_moves import legal_action_mask, legal_actions
from stratego.engine.snapshot import clone_state
from stratego.engine.transition import apply_action
from stratego.evaluation.policy import PolicyRequirements, build_policy_input
from stratego.model.policy_adapter import greedy_action

from ..helpers import known_good_game, nonterminal_state
from .conftest import StubOutputPolicy, crafted_policy_logits


# ---------------------------------------------------------------------------
# 1. The exhaustive encoding audit
# ---------------------------------------------------------------------------


def test_every_action_identifier_round_trips():
    """All 10,000 identifiers, no sampling."""
    mismatches = []
    for action_id in range(ACTION_SPACE_SIZE):
        source, destination = decode_action(action_id)
        if encode_action(source, destination) != action_id:
            mismatches.append(action_id)
    assert mismatches == []


def test_every_action_identifier_decodes_to_the_documented_arithmetic():
    for action_id in range(ACTION_SPACE_SIZE):
        source, destination = decode_action(action_id)
        assert 0 <= source < NUM_SQUARES and 0 <= destination < NUM_SQUARES
        assert action_id == 100 * source + destination
        assert action_source(action_id) == source
        assert action_destination(action_id) == destination


def test_the_mapping_is_a_bijection_onto_every_square_pair():
    seen = {decode_action(action_id) for action_id in range(ACTION_SPACE_SIZE)}
    assert len(seen) == ACTION_SPACE_SIZE
    assert seen == {(s, d) for s in range(NUM_SQUARES) for d in range(NUM_SQUARES)}


# ---------------------------------------------------------------------------
# 2. Selection through the real adapter
# ---------------------------------------------------------------------------


def _position_corpus():
    """Positions spanning the opening and several mid-game plies.

    Different plies expose different move families: the opening position has
    only forward single steps available, while later positions add attacks,
    lateral moves and long Scout slides.
    """
    corpus = [known_good_game()]
    corpus.extend(nonterminal_state(ply) for ply in (10, 25, 50, 90, 140, 200, 260))
    # A second seed family, so the corpus is not one game sampled seven times.
    corpus.extend(nonterminal_state(ply, first_seed=37) for ply in (18, 60, 120, 240))
    return corpus


def _move_families(state, actions):
    """Classify a position's legal actions so coverage can be asserted."""
    families = set()
    for action in actions:
        source, destination = decode_action(action)
        piece_id = state.board[source]
        piece = state.pieces[piece_id]
        distance = max(
            abs(source // 10 - destination // 10), abs(source % 10 - destination % 10)
        )
        occupied = state.board[destination] is not None
        families.add("attack" if occupied else "quiet")
        families.add("long_scout" if distance > 1 else "single_step")
        if piece.true_type == SCOUT:
            families.add("scout_move")
        if source // 10 == destination // 10:
            families.add("lateral")
        else:
            families.add("vertical")
    return families


def test_the_corpus_covers_every_legal_move_family():
    families = set()
    for state in _position_corpus():
        families |= _move_families(state, legal_actions(state))
    assert {
        "attack",
        "quiet",
        "long_scout",
        "single_step",
        "scout_move",
        "lateral",
        "vertical",
    } <= families


def test_a_crafted_maximum_at_every_legal_index_selects_that_engine_action(model):
    """For every legal action in every corpus position, the adapter picks it.

    This is the claim that there is no remapping table: the logit index and the
    engine action identifier are the same number.
    """
    checked = 0
    for state in _position_corpus():
        actions = legal_actions(state)
        mask = legal_action_mask(state, actions)
        for action in actions:
            logits = crafted_policy_logits(action)
            selected = greedy_action(logits, actions)
            assert selected == action
            # ... and the engine agrees about what that identifier means.
            source, destination = decode_action(selected)
            assert mask[encode_action(source, destination)] == 1
            checked += 1
    assert checked > 250  # a real corpus, not two positions


def test_the_adapter_end_to_end_selects_the_crafted_action(model):
    """Same claim, but through `decide()` including the requirement plumbing."""
    state = nonterminal_state(50)
    actions = legal_actions(state)
    for action in actions[:: max(1, len(actions) // 25)]:
        policy = StubOutputPolicy(model, crafted_policy_logits(action), mode="greedy")
        request = build_policy_input(
            state,
            policy=policy.ref,
            policy_seed=7,
            requirements=PolicyRequirements(observation=True, legal_action_mask=True),
        )
        result = policy.decide_checked(request)
        assert result.selected_action_id == action
        assert result.diagnostics["source_square"] == action_source(action)
        assert result.diagnostics["destination_square"] == action_destination(action)


def test_the_engine_executes_the_move_the_index_named(model):
    """The selected identifier moves the piece the identifier's squares imply."""
    state = nonterminal_state(50)
    actions = legal_actions(state)
    for action in actions[:: max(1, len(actions) // 15)]:
        working = clone_state(state)
        source, destination = decode_action(action)
        moving_piece_id = working.board[source]
        policy = StubOutputPolicy(model, crafted_policy_logits(action), mode="greedy")
        request = build_policy_input(
            working,
            policy=policy.ref,
            policy_seed=11,
            requirements=PolicyRequirements(observation=True, legal_action_mask=True),
        )
        selected = policy.decide_checked(request).selected_action_id
        apply_action(working, selected)
        moved = working.pieces[moving_piece_id]
        # Either the piece now stands on the destination, or it died attacking it.
        assert (moved.current_square == destination) or (not moved.alive)
        assert working.board[source] is None


def test_the_dense_mask_and_the_legal_list_describe_the_same_actions():
    for state in _position_corpus():
        actions = legal_actions(state)
        mask = legal_action_mask(state, actions)
        assert np.flatnonzero(mask).tolist() == sorted(actions)


def test_a_perspective_normalized_index_is_not_what_the_adapter_uses():
    """Pin the frame decision: policy indices are absolute engine squares.

    Blue's observation is rotated 180 degrees, so if the adapter had silently
    adopted the normalized frame this test would fail -- which is exactly the
    ambiguity Phase 5 was asked to resolve explicitly.
    """
    from stratego.engine.actions import action_to_perspective
    from stratego.engine.constants import BLUE

    state = nonterminal_state(51)
    if state.acting_player != BLUE:
        state = nonterminal_state(50)
        assert state.acting_player is not None
    actions = legal_actions(state)
    action = actions[len(actions) // 2]
    normalized = action_to_perspective(action, state.acting_player)
    selected = greedy_action(crafted_policy_logits(action), actions)
    assert selected == action
    if normalized != action:
        # The normalized identifier is a different number, and it is not what a
        # peak at the absolute index selects.
        assert selected != normalized


def test_selection_uses_only_legal_indices_even_when_illegal_ones_score_higher():
    state = nonterminal_state(35)
    actions = legal_actions(state)
    illegal = next(action for action in range(ACTION_SPACE_SIZE) if action not in set(actions))
    logits = torch.full((ACTION_SPACE_SIZE,), -10.0)
    logits[illegal] = 1_000.0
    logits[actions[3]] = 5.0
    assert greedy_action(logits, actions) == actions[3]
