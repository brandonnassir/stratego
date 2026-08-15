"""Phase 8 Agent 3: `warmstart_example_v1` construction and its audits.

Covers the deterministic selected-decision universe, the example schema, the
equivalence of sequential and random-access reconstruction, the action-frame
conversions, the frozen supervision weights, teacher-decision reproduction,
and the audits' ability to catch deliberately broken targets.
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest

from stratego.engine.constants import BLUE, RED
from stratego.model.action_frame import (
    absolute_action_to_model,
    model_action_to_absolute,
)
from stratego.model.contract import (
    BELIEF_IGNORE_INDEX,
    VALUE_DRAW_INDEX,
    VALUE_LOSS_INDEX,
    VALUE_WIN_INDEX,
)
from stratego.training.corpus_commit import CorpusReader
from stratego.training.reconstruction import (
    iter_reconstructed_decisions,
    reconstruct_decision,
)
from stratego.training import warmstart_examples as we
from stratego.training.warmstart_contract import POLICY_SUPERVISION_WEIGHTS
from stratego.training.warmstart_seed import (
    CORPUS_SPLITS,
    MAX_DECISIONS_PER_GAME,
    SYNTHETIC_CORPUS_VERSION,
    selected_decision_indices,
)


@pytest.fixture(scope="module")
def mini_reader(warmstart_mini_corpus):
    root, game_ids = warmstart_mini_corpus
    return CorpusReader(root, CORPUS_SPLITS), game_ids


@pytest.fixture(scope="module")
def built_examples(mini_reader):
    """Every selected example of every mini-corpus game, by game id."""
    reader, game_ids = mini_reader
    built = {}
    for game_id in game_ids:
        record, metadata = reader.game(game_id)
        built[game_id] = (record, metadata, list(we.examples_for_game(record, metadata)))
    return built


# ---------------------------------------------------------------------------
# The frozen decision selection, reproduced independently
# ---------------------------------------------------------------------------


def _independent_selection(game_id: str, total: int) -> tuple:
    """`warmstart_decision_sampler_v1` re-derived from its written spec.

    Bins are `[floor(b*T/64), floor((b+1)*T/64))`; each bin's draw is the
    domain-separated blake2b stream with the `strat-ws8` personalization,
    reduced modulo the bin width. Implemented here from scratch so a drift in
    the production helper cannot hide behind itself.
    """
    if total <= MAX_DECISIONS_PER_GAME:
        return tuple(range(total))
    selected = []
    for bin_index in range(MAX_DECISIONS_PER_GAME):
        low = (bin_index * total) // MAX_DECISIONS_PER_GAME
        high = ((bin_index + 1) * total) // MAX_DECISIONS_PER_GAME
        payload = ":".join(
            [SYNTHETIC_CORPUS_VERSION, "decision_sampler", game_id, str(bin_index)]
        )
        digest = hashlib.blake2b(
            payload.encode(), digest_size=8, person=b"strat-ws8"
        ).digest()
        draw = (int.from_bytes(digest, "big") >> 1) % (high - low)
        selected.append(low + draw)
    return tuple(selected)


def test_the_selection_matches_an_independent_reimplementation(mini_reader):
    reader, game_ids = mini_reader
    for game_id in game_ids:
        total = reader.commits[game_id].total_decisions
        assert selected_decision_indices(game_id, total) == _independent_selection(
            game_id, total
        )
    # And on synthetic long-game sizes that exercise the binned branch.
    for total in (65, 100, 259, 1024, 4001):
        assert selected_decision_indices(game_ids[0], total) == _independent_selection(
            game_ids[0], total
        )


def test_the_selection_respects_the_cap_and_short_game_rule(mini_reader):
    reader, game_ids = mini_reader
    for game_id in game_ids:
        total = reader.commits[game_id].total_decisions
        selected = selected_decision_indices(game_id, total)
        assert len(selected) == min(total, MAX_DECISIONS_PER_GAME)
        assert all(0 <= index < total for index in selected)
        assert list(selected) == sorted(set(selected))
        if total <= MAX_DECISIONS_PER_GAME:
            assert selected == tuple(range(total))


# ---------------------------------------------------------------------------
# Example schema and reconstruction equivalence
# ---------------------------------------------------------------------------


def test_examples_carry_the_contract_shapes_and_dtypes(built_examples):
    for _game_id, (_record, _metadata, examples) in built_examples.items():
        assert examples, "every mini-corpus game selects at least one decision"
        for example in examples:
            assert example.observation.shape == (127, 10, 10)
            assert example.observation.dtype == np.float32
            assert example.legal_mask.shape == (10000,)
            assert example.legal_mask.dtype == np.bool_
            assert example.belief_target.shape == (100,)
            assert example.belief_target.dtype == np.int64
            assert example.belief_mask.shape == (100,)
            assert example.belief_mask.dtype == np.bool_
            assert example.acting_player in (RED, BLUE)
            assert example.value_target in (
                VALUE_WIN_INDEX,
                VALUE_DRAW_INDEX,
                VALUE_LOSS_INDEX,
            )
            assert 0 <= example.policy_action_abs < 10000
            assert 0 <= example.policy_action_model < 10000
            assert example.corpus_split in CORPUS_SPLITS


def test_sequential_and_random_access_reconstruction_agree(built_examples):
    """The streaming path must equal independent per-ply snapshot access."""
    for game_id, (record, metadata, examples) in built_examples.items():
        for example in examples[:: max(1, len(examples) // 6)]:
            rebuilt = reconstruct_decision(
                record,
                example.decision_index,
                dense_mask=True,
                include_public_knowledge=False,
            )
            independent = we.build_example(record, metadata, rebuilt)
            assert np.array_equal(independent.observation, example.observation), game_id
            assert np.array_equal(independent.legal_mask, example.legal_mask)
            assert np.array_equal(independent.belief_target, example.belief_target)
            assert np.array_equal(independent.belief_mask, example.belief_mask)
            for field in (
                "acting_player",
                "policy_action_abs",
                "policy_action_model",
                "policy_weight",
                "value_target",
                "game_id",
                "decision_index",
                "source_policy_id",
                "corpus_split",
            ):
                assert getattr(independent, field) == getattr(example, field), field


def test_the_replay_audit_is_clean_on_every_built_example(built_examples):
    for _game_id, (record, metadata, examples) in built_examples.items():
        indices = tuple(example.decision_index for example in examples)
        rebuilt_stream = iter_reconstructed_decisions(
            record, indices, dense_mask=True, include_public_knowledge=False
        )
        for example, rebuilt in zip(examples, rebuilt_stream):
            assert we.audit_example(example, record, metadata, rebuilt) == []


def test_the_static_audit_is_clean_and_counts_every_selection(mini_reader):
    reader, game_ids = mini_reader
    for game_id in game_ids:
        record, metadata = reader.game(game_id)
        result = we.audit_game_static(
            record, metadata, reader.commits[game_id].total_decisions
        )
        assert result["problems"] == []
        assert result["checked"] == result["selected"] == len(
            selected_decision_indices(game_id, len(record.decisions))
        )
        assert sum(result["value_counts"]) == result["checked"]


# ---------------------------------------------------------------------------
# Action frames
# ---------------------------------------------------------------------------


def test_model_actions_invert_and_stay_inside_the_converted_legal_set(built_examples):
    for _game_id, (_record, _metadata, examples) in built_examples.items():
        for example in examples:
            assert (
                model_action_to_absolute(example.policy_action_model, example.acting_player)
                == example.policy_action_abs
            )
            assert example.legal_mask[example.policy_action_model]
            if example.acting_player == RED:
                # Red's normalization is the identity, so the frames agree.
                assert example.policy_action_model == example.policy_action_abs
            else:
                mapped = absolute_action_to_model(example.policy_action_abs, BLUE)
                assert example.policy_action_model == mapped


def test_blue_frames_are_rotated_not_copied(built_examples):
    """At least one blue decision must move under the frame conversion."""
    moved = 0
    for _game_id, (_record, _metadata, examples) in built_examples.items():
        for example in examples:
            if example.acting_player == BLUE and (
                example.policy_action_model != example.policy_action_abs
            ):
                moved += 1
    assert moved > 0


# ---------------------------------------------------------------------------
# Supervision weights
# ---------------------------------------------------------------------------


def test_policy_weights_match_the_frozen_contract(built_examples):
    observed = {}
    for _game_id, (_record, _metadata, examples) in built_examples.items():
        for example in examples:
            observed.setdefault(example.source_policy_id, set()).add(example.policy_weight)
    assert observed, "the mini corpus produced no examples"
    for policy_id, weights in observed.items():
        assert weights == {POLICY_SUPERVISION_WEIGHTS[policy_id]}, policy_id
    # Every weight class is represented in the fixture.
    seen = {POLICY_SUPERVISION_WEIGHTS[policy_id] for policy_id in observed}
    assert seen == {0.0, 0.5, 1.0}


def test_zero_weight_decisions_still_supervise_value_and_belief(built_examples):
    zero_weight = [
        example
        for _gid, (_r, _m, examples) in built_examples.items()
        for example in examples
        if example.policy_weight == 0.0
    ]
    assert zero_weight, "the fixture includes random/stress teachers"
    for example in zero_weight[:20]:
        assert example.value_target in (0, 1, 2)
        assert np.array_equal(
            example.belief_mask, example.belief_target != BELIEF_IGNORE_INDEX
        )


# ---------------------------------------------------------------------------
# Audit sensitivity: broken targets must be caught
# ---------------------------------------------------------------------------


def _first_example_with_state(built_examples):
    game_id = next(iter(built_examples))
    record, metadata, examples = built_examples[game_id]
    example = examples[0]
    rebuilt = reconstruct_decision(
        record, example.decision_index, dense_mask=True, include_public_knowledge=False
    )
    return record, metadata, example, rebuilt


def test_the_audit_catches_a_tampered_legal_mask(built_examples):
    record, metadata, example, rebuilt = _first_example_with_state(built_examples)
    mask = example.legal_mask.copy()
    flip = int(np.flatnonzero(~mask)[0])
    mask[flip] = True
    tampered = dataclasses.replace(example, legal_mask=mask)
    assert any(
        "mask" in problem for problem in we.audit_example(tampered, record, metadata, rebuilt)
    )


def test_the_audit_catches_a_wrong_value_perspective(built_examples):
    record, metadata, example, rebuilt = _first_example_with_state(built_examples)
    inverted = {
        VALUE_WIN_INDEX: VALUE_LOSS_INDEX,
        VALUE_LOSS_INDEX: VALUE_WIN_INDEX,
        VALUE_DRAW_INDEX: VALUE_WIN_INDEX,
    }[example.value_target]
    tampered = dataclasses.replace(example, value_target=inverted)
    assert any(
        "value" in problem for problem in we.audit_example(tampered, record, metadata, rebuilt)
    )


def test_the_audit_catches_a_corrupted_belief_label(built_examples):
    record, metadata, example, rebuilt = _first_example_with_state(built_examples)
    labels = example.belief_target.copy()
    supervised = np.flatnonzero(example.belief_mask)
    assert supervised.size > 0
    labels[supervised[0]] = (labels[supervised[0]] + 1) % 12
    tampered = dataclasses.replace(example, belief_target=labels)
    assert any(
        "belief" in problem for problem in we.audit_example(tampered, record, metadata, rebuilt)
    )


def test_the_audit_catches_a_belief_label_outside_the_hidden_set(built_examples):
    record, metadata, example, rebuilt = _first_example_with_state(built_examples)
    labels = example.belief_target.copy()
    mask = example.belief_mask.copy()
    empty = int(np.flatnonzero(~mask)[0])
    labels[empty] = 3
    mask[empty] = True
    tampered = dataclasses.replace(example, belief_target=labels, belief_mask=mask)
    assert we.audit_example(tampered, record, metadata, rebuilt)


def test_the_audit_catches_a_wrong_policy_weight(built_examples):
    record, metadata, example, rebuilt = _first_example_with_state(built_examples)
    tampered = dataclasses.replace(example, policy_weight=example.policy_weight + 0.25)
    assert any(
        "weight" in problem for problem in we.audit_example(tampered, record, metadata, rebuilt)
    )


# ---------------------------------------------------------------------------
# Value mapping
# ---------------------------------------------------------------------------


def test_value_mapping_is_exact_for_both_colors():
    assert we.value_target_index("red_win", RED) == VALUE_WIN_INDEX
    assert we.value_target_index("red_win", BLUE) == VALUE_LOSS_INDEX
    assert we.value_target_index("blue_win", BLUE) == VALUE_WIN_INDEX
    assert we.value_target_index("blue_win", RED) == VALUE_LOSS_INDEX
    assert we.value_target_index("draw", RED) == VALUE_DRAW_INDEX
    assert we.value_target_index("draw", BLUE) == VALUE_DRAW_INDEX
    with pytest.raises(we.WarmstartExampleError):
        we.value_target_index("red_wins", RED)


def test_value_targets_flip_between_the_two_sides_of_one_decisive_game(built_examples):
    for _game_id, (record, _metadata, examples) in built_examples.items():
        if record.terminal_result == "draw":
            continue
        red = {e.value_target for e in examples if e.acting_player == RED}
        blue = {e.value_target for e in examples if e.acting_player == BLUE}
        if red and blue:
            assert red.isdisjoint(blue)
            assert red | blue == {VALUE_WIN_INDEX, VALUE_LOSS_INDEX}


# ---------------------------------------------------------------------------
# Teacher-decision reproduction
# ---------------------------------------------------------------------------


def test_recorded_teacher_decisions_reproduce_exactly(built_examples):
    reproduced = 0
    for _game_id, (record, metadata, examples) in built_examples.items():
        plies = tuple(e.decision_index for e in examples if e.policy_weight > 0.0)[:12]
        if not plies:
            continue
        result = we.reproduce_teacher_decisions(record, metadata, plies)
        assert result["mismatches"] == []
        assert result["reproduced"] == len(plies)
        reproduced += result["reproduced"]
    assert reproduced >= 20


def test_reproduction_flags_a_tampered_recorded_action(built_examples):
    for _game_id, (record, metadata, examples) in built_examples.items():
        supervised = [e for e in examples if e.policy_weight > 0.0]
        if not supervised:
            continue
        ply = supervised[0].decision_index
        decision = record.decisions[ply]
        other = next(
            action
            for action in decision.legal_action_ids
            if action != decision.selected_action_id
        )
        tampered_decision = dataclasses.replace(decision, selected_action_id=other)
        decisions = list(record.decisions)
        decisions[ply] = tampered_decision
        tampered_record = dataclasses.replace(record, decisions=tuple(decisions))
        result = we.reproduce_teacher_decisions(tampered_record, metadata, (ply,))
        assert result["mismatches"]
        return
    pytest.fail("no policy-supervised example in the fixture")


# ---------------------------------------------------------------------------
# Progress buckets
# ---------------------------------------------------------------------------


def test_progress_buckets_partition_the_game():
    assert we.progress_bucket(0, 100) == 0
    assert we.progress_bucket(24, 100) == 0
    assert we.progress_bucket(25, 100) == 1
    assert we.progress_bucket(99, 100) == 3
    assert we.progress_bucket(0, 1) == 0
    with pytest.raises(we.WarmstartExampleError):
        we.progress_bucket(0, 0)
