"""Phase 8 Agent 3: the training-batch privilege boundary.

The model forward call receives one tensor — the observation batch — and
nothing else. These tests prove that claim structurally (an object-graph walk
from the model input reaches no privileged value, with positive controls
showing the walk would find one) and semantically (hidden-identity
permutations leave every model-visible product byte-identical while the
privileged labels move).

This is the training-pipeline anti-leak boundary, not a replacement for the
Phase 2 engine anti-leak validation, which remains authoritative for the
observation itself.
"""

from __future__ import annotations

import random
import types

import numpy as np
import pytest
import torch

from stratego.engine.constants import BLUE, RED
from stratego.training import warmstart_dataset as wd
from stratego.training import warmstart_examples as we
from stratego.training.reconstruction import iter_reconstructed_decisions
from stratego.training.rule_population import play_corpus_game
from stratego.training.warmstart_seed import synthetic_game_id

#: Values that must never be reachable from a model input: identities,
#: outcomes, provenance keys and teacher names of the fixture game.
_ATOMIC = (bytes, bytearray, int, float, bool, complex, type(None))

_NOT_FOLLOWED = (
    type,
    types.ModuleType,
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.MethodWrapperType,
    types.WrapperDescriptorType,
    types.MethodDescriptorType,
    types.GetSetDescriptorType,
    types.MemberDescriptorType,
    types.CodeType,
    types.FrameType,
    types.TracebackType,
    np.ndarray,
)


def reachable_strings(roots, *, limit: int = 400_000) -> set:
    """Every `str` an object holds, transitively, over data edges only.

    Unlike the Phase 7 provenance walker, tensors are *followed through their
    instance dictionaries*: a `torch.Tensor` is exactly the object handed to
    the model here, so an attribute smuggled onto it must be findable. The
    numeric storage itself holds no Python references and is not descended
    into.
    """
    seen: set[int] = set()
    found: set[str] = set()
    stack = list(roots)
    visited = 0
    while stack:
        item = stack.pop()
        identifier = id(item)
        if identifier in seen:
            continue
        seen.add(identifier)
        visited += 1
        if visited > limit:  # pragma: no cover - guard against a runaway walk
            raise AssertionError("reachability walk exceeded its node limit")
        if isinstance(item, str):
            found.add(item)
            continue
        if isinstance(item, torch.Tensor):
            attributes = getattr(item, "__dict__", None)
            if attributes:
                stack.append(attributes)
            continue
        if isinstance(item, _ATOMIC) or isinstance(item, _NOT_FOLLOWED):
            continue
        if isinstance(item, dict):
            for key, value in item.items():
                stack.append(key)
                stack.append(value)
            continue
        if isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)
            continue
        attributes = getattr(item, "__dict__", None)
        if isinstance(attributes, dict):
            stack.append(attributes)
        for name in getattr(type(item), "__slots__", ()) or ():
            if isinstance(name, str) and hasattr(item, name):
                stack.append(getattr(item, name))
    return found


@pytest.fixture(scope="module")
def played_game():
    """One deterministic corpus game, played in memory — no store needed."""
    game_id = synthetic_game_id(
        "train", "basic_heuristic@1.0.0", "random_legal@1.0.0", 3
    )
    game = play_corpus_game(game_id)
    return game.record, game.metadata


@pytest.fixture(scope="module")
def built_batch(played_game):
    record, metadata = played_game
    examples = list(we.examples_for_game(record, metadata))[:32]
    arrays, batch_metadata = wd.arrays_from_examples(examples)
    return examples, wd.batch_from_arrays(arrays, batch_metadata)


def _privileged_markers(record, metadata) -> set:
    markers = {
        record.game_id,
        metadata["synthetic_game_id"],
        "setup_provenance",
        "belief_target",
        "value_target",
        record.terminal_result,
        metadata["red_policy_id"],
        metadata["blue_policy_id"],
        str(metadata["setup_provenance"]["red"]["primary_family_id"]),
        str(metadata["setup_provenance"]["blue"]["primary_family_id"]),
    }
    return markers


# ---------------------------------------------------------------------------
# The object-graph boundary
# ---------------------------------------------------------------------------


def test_the_model_input_reaches_no_privileged_value(played_game, built_batch):
    record, metadata = played_game
    _examples, batch = built_batch
    model_input = batch.model_input()
    assert isinstance(model_input, torch.Tensor)
    strings = reachable_strings([model_input])
    assert strings == set()
    assert strings.isdisjoint(_privileged_markers(record, metadata))


def test_the_walk_would_find_the_privilege_if_it_were_there(played_game, built_batch):
    """Positive controls: the same walk, pointed at objects that do leak."""
    record, metadata = played_game
    examples, batch = built_batch
    markers = _privileged_markers(record, metadata)

    # Control one: the full batch object openly carries identities and, via
    # its targets, the supervision tensors. The walker must see them.
    from_batch = reachable_strings([batch])
    assert record.game_id in from_batch

    # Control two: privileged metadata smuggled onto the model-input tensor
    # itself must be detected.
    contaminated = batch.model_input().clone()
    contaminated.leaked = {"setup_provenance": metadata["setup_provenance"]}
    found = reachable_strings([contaminated])
    assert "setup_provenance" in found
    assert str(metadata["setup_provenance"]["red"]["primary_family_id"]) in found

    # Control three: an example object (pre-batch) carries its identity, so
    # a batch that handed the model an example instead of the bare tensor
    # would light the same walk up.
    example_strings = reachable_strings([examples[0]])
    assert record.game_id in example_strings
    assert markers & example_strings


def test_the_model_input_shares_no_storage_with_any_target(built_batch):
    examples, batch = built_batch
    observation = batch.model_input()
    targets = batch.targets
    for tensor in (
        targets.legal_mask,
        targets.policy_action_model,
        targets.policy_weight,
        targets.value_target,
        targets.belief_target,
        targets.belief_mask,
        targets.policy_action_abs,
        targets.acting_player,
    ):
        assert tensor.data_ptr() != observation.data_ptr()
        assert not np.shares_memory(observation.numpy(), tensor.numpy())
    # The batch copies example buffers rather than aliasing them.
    for example in examples[:4]:
        assert not np.shares_memory(observation.numpy(), example.observation)
        assert not np.shares_memory(observation.numpy(), example.belief_target)


def test_the_observation_never_encodes_the_belief_labels(built_batch):
    """No observation channel may correlate bytes with the label vector.

    The hidden-occupancy channel legitimately marks *where* unresolved pieces
    stand; the labels say *what* they are. Equal masks with independent label
    content is exactly the permutation invariance proven below; here the
    cheaper structural fact is pinned: the label vector's bytes appear nowhere
    inside the observation tensor.
    """
    examples, _batch = built_batch
    for example in examples[:8]:
        planes = example.observation.reshape(example.observation.shape[0], -1)
        supervised = np.flatnonzero(example.belief_mask)
        if supervised.size == 0:
            continue
        labels = example.belief_target[supervised].astype(np.float32)
        for channel in range(planes.shape[0]):
            row = planes[channel, supervised]
            if row.shape == labels.shape and np.allclose(row, labels):
                raise AssertionError(
                    f"channel {channel} reproduces the belief labels verbatim"
                )


# ---------------------------------------------------------------------------
# Hidden-permutation paired trials
# ---------------------------------------------------------------------------


def test_hidden_permutations_move_labels_but_never_the_model_view(played_game):
    record, metadata = played_game
    indices = tuple(
        index
        for index in we.selected_decision_indices(record.game_id, len(record.decisions))
    )
    rng = random.Random(20260815)
    valid = 0
    changed = 0
    for rebuilt in iter_reconstructed_decisions(
        record, indices, dense_mask=True, include_public_knowledge=False
    ):
        trial = we.hidden_permutation_trial(record, metadata, rebuilt, rng)
        assert trial["mismatches"] == [], trial
        assert trial["control_ok"], trial
        if trial["valid"] and trial["hidden_pieces"] >= 2:
            valid += 1
            changed += int(trial["changed"])
    assert valid >= 30
    assert changed >= 10  # the positive control fires, not vacuously


def test_the_paired_comparators_are_not_vacuously_true(played_game):
    """Positive control for the comparison surface itself.

    The permutation trials pass because the states genuinely agree, not
    because the comparators cannot fire: two *different* plies of the same
    game must disagree under exactly the comparisons the trial uses.
    """
    record, metadata = played_game
    indices = we.selected_decision_indices(record.game_id, len(record.decisions))
    first, second = (
        we.build_example(record, metadata, rebuilt)
        for rebuilt in iter_reconstructed_decisions(
            record, indices[4:6], dense_mask=True, include_public_knowledge=False
        )
    )
    assert not np.array_equal(first.observation, second.observation)
    assert first.decision_index != second.decision_index
    differs = (
        not np.array_equal(first.legal_mask, second.legal_mask)
        or not np.array_equal(first.belief_mask, second.belief_mask)
        or first.policy_action_model != second.policy_action_model
    )
    assert differs


def test_both_colors_are_exercised(played_game):
    record, metadata = played_game
    indices = we.selected_decision_indices(record.game_id, len(record.decisions))
    actors = set()
    for rebuilt in iter_reconstructed_decisions(
        record, indices[:8], dense_mask=True, include_public_knowledge=False
    ):
        actors.add(rebuilt.acting_player)
    assert actors == {RED, BLUE}
