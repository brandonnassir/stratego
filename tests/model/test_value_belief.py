"""Value and belief semantics, including the privileged-target boundary.

Covers Phase 5 gates 7 (`value_output_contract_validated`) and 8
(`belief_output_and_mask_validated`).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stratego.engine.constants import BLUE, LAKE_SQUARE_SET, RED
from stratego.engine.coordinates import to_perspective
from stratego.engine.observation import belief_target, build_observation
from stratego.model.contract import (
    BELIEF_IGNORE_INDEX,
    VALUE_CLASS_ORDER,
    ModelContractError,
    expected_value,
    value_probabilities,
)
from stratego.model.losses import belief_loss
from stratego.model.tokenization import tokenize_numpy_observation
from stratego.training.belief_targets import (
    PIECE_TYPE_INDEX,
    belief_target_summary,
    dense_belief_target,
    dense_belief_target_batch,
)

from ..helpers import nonterminal_state
from ..observation.test_perspective import mirrored_games, play_mirrored
from .conftest import deterministic_observation


# ---------------------------------------------------------------------------
# Value: controlled logits
# ---------------------------------------------------------------------------


def test_the_three_classes_are_win_draw_loss_in_that_order():
    assert VALUE_CLASS_ORDER == ("WIN", "DRAW", "LOSS")
    # A logit vector that is maximal in slot k must give class k the most mass.
    for index in range(3):
        logits = torch.full((1, 3), -4.0)
        logits[0, index] = 4.0
        assert int(value_probabilities(logits).argmax()) == index


def test_probabilities_sum_to_one_over_many_controlled_rows():
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(256, 3, generator=generator) * 6.0
    probabilities = value_probabilities(logits)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(256), atol=1e-6)
    assert bool((probabilities >= 0).all())


def test_expected_value_equals_win_probability_minus_loss_probability():
    generator = torch.Generator().manual_seed(8)
    logits = torch.randn(128, 3, generator=generator) * 5.0
    probabilities = value_probabilities(logits)
    assert torch.allclose(
        expected_value(logits), probabilities[:, 0] - probabilities[:, 2], atol=1e-7
    )
    assert bool((expected_value(logits).abs() <= 1.0).all())


def test_a_scalar_value_head_would_be_rejected():
    with pytest.raises(ModelContractError):
        value_probabilities(torch.zeros(4, 1))


# ---------------------------------------------------------------------------
# Value: acting-player perspective, both colours
# ---------------------------------------------------------------------------


def test_the_value_head_speaks_for_whoever_is_acting_not_for_a_fixed_colour(model):
    """A position and its colour-swapped mirror give the same value vector.

    The observation is normalized to the acting player, so red-to-move in a game
    and blue-to-move in its mirror are the same input. If the value head meant
    "red's chances" rather than "the acting player's chances", these two would
    have to differ -- and they do not.
    """
    original, twin = mirrored_games(4)
    play_mirrored(original, twin, 24, 4)
    assert original.acting_player != twin.acting_player

    for state_a, state_b in ((original, twin),):
        for observer_a, observer_b in ((RED, BLUE), (BLUE, RED)):
            tokens_a = tokenize_numpy_observation(build_observation(state_a, observer_a))
            tokens_b = tokenize_numpy_observation(build_observation(state_b, observer_b))
            assert torch.equal(tokens_a, tokens_b)
            with torch.no_grad():
                out_a = model(tokens_a)
                out_b = model(tokens_b)
            assert torch.equal(out_a.value_logits, out_b.value_logits)
            assert torch.equal(expected_value(out_a.value_logits), expected_value(out_b.value_logits))


def test_the_value_head_is_finite_for_both_colours_in_a_real_game(model):
    state = nonterminal_state(60)
    for observer in (RED, BLUE):
        tokens = tokenize_numpy_observation(build_observation(state, observer))
        with torch.no_grad():
            outputs = model(tokens)
        assert bool(torch.isfinite(outputs.value_logits).all())
        assert abs(float(expected_value(outputs.value_logits)[0])) <= 1.0


# ---------------------------------------------------------------------------
# Belief: head shape
# ---------------------------------------------------------------------------


def test_the_belief_head_is_one_distribution_per_square(model):
    outputs = model.forward_observation(deterministic_observation(seed=2, batch=3))
    assert outputs.belief_logits.shape == (3, 100, 12)
    assert bool(torch.isfinite(outputs.belief_logits).all())


# ---------------------------------------------------------------------------
# Belief: targets and the loss mask
# ---------------------------------------------------------------------------


def test_target_and_mask_have_the_declared_shapes_and_dtypes():
    state = nonterminal_state(45)
    labels, mask = dense_belief_target(state, state.acting_player)
    assert labels.shape == (100,) and labels.dtype == np.int64
    assert mask.shape == (100,) and mask.dtype == bool
    assert np.array_equal(mask, labels != BELIEF_IGNORE_INDEX)


def test_the_mask_is_true_exactly_on_unresolved_opponent_squares():
    state = nonterminal_state(70)
    observer = state.acting_player
    labels, mask = dense_belief_target(state, observer)

    expected_squares = set()
    for entry in belief_target(state, observer):
        expected_squares.add(to_perspective(entry["square"], observer))
        # ... and the label really is that piece's true type.
        assert labels[to_perspective(entry["square"], observer)] == PIECE_TYPE_INDEX[
            entry["true_type"]
        ]
    assert set(np.flatnonzero(mask).tolist()) == expected_squares
    assert mask.sum() == len(belief_target(state, observer))


def test_own_pieces_empty_squares_lakes_and_revealed_pieces_are_all_excluded():
    state = nonterminal_state(80)
    observer = state.acting_player
    labels, mask = dense_belief_target(state, observer)

    supervised = set(np.flatnonzero(mask).tolist())

    # Lakes: never occupied, so never supervised.
    for lake in LAKE_SQUARE_SET:
        assert to_perspective(lake, observer) not in supervised
        assert labels[to_perspective(lake, observer)] == BELIEF_IGNORE_INDEX

    own_squares, empty_squares, revealed_squares = set(), set(), set()
    occupied = set()
    for record in state.pieces:
        if not record.alive:
            continue
        normalized = to_perspective(record.current_square, observer)
        occupied.add(normalized)
        if record.owner == observer:
            own_squares.add(normalized)
        elif record.known_to(observer):
            revealed_squares.add(normalized)
    empty_squares = set(range(100)) - occupied

    assert own_squares and empty_squares  # the test would be vacuous otherwise
    assert not (own_squares & supervised)
    assert not (empty_squares & supervised)
    assert not (revealed_squares & supervised)
    for square in own_squares | empty_squares | revealed_squares:
        assert labels[square] == BELIEF_IGNORE_INDEX


def test_a_revealed_opponent_piece_drops_out_of_the_mask():
    """Play until a capture reveals an opponent piece, then check it is excluded."""
    from stratego.engine.legal_moves import legal_actions
    from stratego.engine.transition import apply_action

    state = nonterminal_state(30)
    for _ in range(400):
        if state.terminal:
            break
        observer = state.acting_player
        revealed = [
            record
            for record in state.pieces
            if record.alive and record.owner != observer and record.known_to(observer)
        ]
        if revealed:
            _, mask = dense_belief_target(state, observer)
            for record in revealed:
                assert not mask[to_perspective(record.current_square, observer)]
            return
        actions = legal_actions(state)
        apply_action(state, actions[len(actions) // 2], legal=actions)
    pytest.skip("no opponent piece was revealed in this seeded game")


def test_targets_are_indexed_by_normalized_square_so_they_align_with_tokens():
    """Blue's targets must be rotated the same way blue's observation is."""
    state = nonterminal_state(55)
    for observer in (RED, BLUE):
        labels, mask = dense_belief_target(state, observer)
        for entry in belief_target(state, observer):
            absolute = entry["square"]
            normalized = to_perspective(absolute, observer)
            assert mask[normalized]
            if observer == BLUE and absolute != normalized:
                # The absolute index is a different square, and it is not the one
                # carrying the label -- unless another hidden piece happens to be
                # standing there.
                assert labels[normalized] == PIECE_TYPE_INDEX[entry["true_type"]]


def test_a_batch_of_targets_stacks_correctly():
    states = [nonterminal_state(ply) for ply in (20, 40, 60)]
    labels, mask = dense_belief_target_batch(states)
    assert labels.shape == (3, 100) and mask.shape == (3, 100)
    for index, state in enumerate(states):
        single_labels, single_mask = dense_belief_target(state, state.acting_player)
        assert np.array_equal(labels[index], single_labels)
        assert np.array_equal(mask[index], single_mask)


# ---------------------------------------------------------------------------
# Belief: labels are separated from inputs
# ---------------------------------------------------------------------------


def test_the_belief_target_is_not_derivable_from_the_observation():
    """Permuting hidden types changes the target and leaves the input identical."""
    import random

    from stratego.engine.permutation import permute_hidden_identities

    state = nonterminal_state(65)
    observer = state.acting_player
    twin, info = permute_hidden_identities(state, observer, random.Random(3))
    assert info["valid"] and info["changed"]

    assert np.array_equal(build_observation(state, observer), build_observation(twin, observer))
    original_labels, original_mask = dense_belief_target(state, observer)
    twin_labels, twin_mask = dense_belief_target(twin, observer)
    assert np.array_equal(original_mask, twin_mask)  # same squares are hidden
    assert not np.array_equal(original_labels, twin_labels)  # different truths


def test_the_model_package_never_imports_the_privileged_target_builder():
    """The separation is structural, not a convention: assert the import graph."""
    import importlib
    import pkgutil

    import stratego.model

    forbidden = ("belief_target", "GameState", "PieceRecord")
    for module_info in pkgutil.iter_modules(stratego.model.__path__):
        module = importlib.import_module(f"stratego.model.{module_info.name}")
        source = open(module.__file__, encoding="utf-8").read()
        for name in forbidden:
            assert f"import {name}" not in source
            assert f"from stratego.training.belief_targets" not in source


def test_the_summary_states_the_convention_for_the_report():
    summary = belief_target_summary()
    assert summary["square_frame"] == "perspective_normalized_squares"
    assert summary["privileged"] is True
    assert summary["reachable_from_model_input"] is False
    assert summary["ignore_index"] == BELIEF_IGNORE_INDEX


# ---------------------------------------------------------------------------
# The masked belief loss
# ---------------------------------------------------------------------------


def test_the_belief_loss_ignores_unsupervised_squares_entirely():
    logits = torch.zeros(1, 100, 12, requires_grad=True)
    labels = torch.full((1, 100), BELIEF_IGNORE_INDEX, dtype=torch.int64)
    mask = torch.zeros(1, 100, dtype=torch.bool)
    labels[0, 5] = 3
    mask[0, 5] = True

    loss = belief_loss(logits, labels, mask)
    loss.backward()
    # Only the supervised square received a gradient.
    gradient = logits.grad[0]
    assert bool((gradient[5] != 0).any())
    assert bool((gradient[torch.arange(100) != 5] == 0).all())


def test_changing_an_unsupervised_squares_logits_does_not_change_the_loss():
    labels = torch.full((1, 100), BELIEF_IGNORE_INDEX, dtype=torch.int64)
    mask = torch.zeros(1, 100, dtype=torch.bool)
    labels[0, 11] = 2
    mask[0, 11] = True

    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(1, 100, 12, generator=generator)
    before = float(belief_loss(logits, labels, mask))
    logits[0, 40] += 1000.0
    assert float(belief_loss(logits, labels, mask)) == before


def test_the_loss_is_normalised_per_supervised_square():
    labels = torch.full((2, 100), BELIEF_IGNORE_INDEX, dtype=torch.int64)
    mask = torch.zeros(2, 100, dtype=torch.bool)
    labels[0, :4] = 0
    mask[0, :4] = True
    labels[1, :1] = 0
    mask[1, :1] = True
    logits = torch.zeros(2, 100, 12)
    # Uniform logits: every supervised square costs log(12), so the mean does too.
    assert abs(float(belief_loss(logits, labels, mask)) - float(np.log(12))) < 1e-5


def test_a_mask_that_disagrees_with_the_labels_raises():
    labels = torch.full((1, 100), BELIEF_IGNORE_INDEX, dtype=torch.int64)
    mask = torch.zeros(1, 100, dtype=torch.bool)
    mask[0, 7] = True  # masked but unlabelled
    with pytest.raises(ModelContractError, match="disagree"):
        belief_loss(torch.zeros(1, 100, 12), labels, mask)


def test_a_batch_with_no_supervised_square_gives_a_real_zero():
    labels = torch.full((1, 100), BELIEF_IGNORE_INDEX, dtype=torch.int64)
    mask = torch.zeros(1, 100, dtype=torch.bool)
    logits = torch.zeros(1, 100, 12, requires_grad=True)
    loss = belief_loss(logits, labels, mask)
    assert float(loss.detach()) == 0.0
    loss.backward()  # still differentiable, so it composes into the total


def test_real_engine_targets_drive_the_loss(model):
    state = nonterminal_state(50)
    observer = state.acting_player
    labels, mask = dense_belief_target(state, observer)
    tokens = tokenize_numpy_observation(build_observation(state, observer))
    outputs = model(tokens)
    loss = belief_loss(
        outputs.belief_logits,
        torch.from_numpy(labels)[None, :],
        torch.from_numpy(mask)[None, :],
    )
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0
