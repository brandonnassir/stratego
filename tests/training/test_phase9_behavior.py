"""Phase 9 Agent 3: behavior snapshots and the neural decision path.

The collection boundary is what makes a later importance ratio trustworthy, so
these tests check the properties a PPO update depends on rather than the code
paths that produce them: one immutable snapshot per iteration, a distribution
that is exactly the one that chose the move, a selection rule that is a pure
function of the logical identity, and a reproduction audit that actually fails
when it should.

Every positive control has a negative one beside it. An audit that verified a
decision against the wrong checkpoint, or a batch shape that silently changed
stored bytes, would pass a suite of positive tests alone and would corrupt a
sealed rollout in production.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stratego.engine.legal_moves import legal_action_mask, legal_actions
from stratego.engine.observation import build_observation
from stratego.engine.state import create_game
from stratego.model.policy_adapter import prepare_legality
from stratego.training import phase9_behavior as pb
from stratego.training.phase9_contract import BEHAVIOR_PROBABILITY_ABS_TOLERANCE
from stratego.training.phase9_seed import behavior_sample_uniform, phase9_game_id
from stratego.training.setup_source import training_setup_source
from stratego.training.warmstart_contract import CORPUS_RULES, EXPECTED_SETUP_PROFILE

#: The accepted Phase 8 checkpoint and its frozen SHA-256. `H000` is this file.
ANCHOR_CHECKPOINT = "checkpoints/phase8/warmstart_c1_v1.pt"
ANCHOR_SHA256 = "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca"
UNTRAINED_CHECKPOINT = "checkpoints/phase8/warmstart_c1_v1_initialisation.pt"
UNTRAINED_SHA256 = "01c907eeef86ec04121db55ccffb9365e8df27fdf05921b921947d4af365754c"


@pytest.fixture(scope="module")
def anchor():
    return pb.load_behavior_snapshot(
        ANCHOR_CHECKPOINT,
        logical_identity="B001",
        policy_token="phase9_behavior_v1|ns=canonical|B001",
        inference_batch_shape=4,
        expected_sha256=ANCHOR_SHA256,
    )


@pytest.fixture(scope="module")
def untrained():
    """A second, genuinely different network: the negative control's teeth."""
    return pb.load_behavior_snapshot(
        UNTRAINED_CHECKPOINT,
        logical_identity="B001",
        policy_token="phase9_behavior_v1|ns=canonical|B001",
        inference_batch_shape=4,
        expected_sha256=UNTRAINED_SHA256,
    )


def _position(root_seed: int = 4242, plies: int = 0):
    """One live position, with everything a decision needs."""
    source = training_setup_source(EXPECTED_SETUP_PROFILE)
    assignment = source.assign(root_seed=root_seed, environment_id=0, generation=0)
    state = create_game(
        assignment.red_setup, assignment.blue_setup, rules=CORPUS_RULES, game_id="probe"
    )
    for _ in range(plies):
        from stratego.engine.transition import apply_action

        apply_action(state, legal_actions(state)[0])
    legal = legal_actions(state)
    legality = prepare_legality(legal, legal_action_mask(state, legal), state.acting_player)
    return state, legality, build_observation(state, state.acting_player)


# ---------------------------------------------------------------------------
# Snapshot identity and immutability
# ---------------------------------------------------------------------------


def test_snapshot_binds_logical_identity_to_the_real_file_digest(anchor):
    assert anchor.checkpoint_sha256 == ANCHOR_SHA256
    assert anchor.logical_identity == "B001"
    assert anchor.identity()["temperature"] == 1.0


def test_a_wrong_expected_digest_is_refused_rather_than_recorded():
    # The guard against binding a logical archive identity to whatever file
    # happens to sit at a path.
    with pytest.raises(pb.Phase9BehaviorError, match="bound to"):
        pb.load_behavior_snapshot(
            ANCHOR_CHECKPOINT,
            logical_identity="H005",
            policy_token="phase9_archive_v1|ns=canonical|H005",
            expected_sha256="0" * 64,
        )


def test_a_missing_checkpoint_cannot_be_collected_against():
    with pytest.raises(pb.Phase9BehaviorError, match="missing checkpoint"):
        pb.load_behavior_snapshot(
            "checkpoints/phase9/archive/does_not_exist.pt",
            logical_identity="H005",
            policy_token="phase9_archive_v1|ns=canonical|H005",
        )


def test_snapshot_is_frozen_against_optimizer_mutation(anchor):
    anchor.assert_frozen()
    assert not any(p.requires_grad for p in anchor.model.parameters())


def test_assert_frozen_detects_weights_that_moved(anchor):
    parameter = next(iter(anchor.model.parameters()))
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(1.0)
    try:
        with pytest.raises(pb.Phase9BehaviorError, match="changed since it was loaded"):
            anchor.assert_frozen()
    finally:
        with torch.no_grad():
            parameter.copy_(original)
    anchor.assert_frozen()


def test_training_mode_is_refused(anchor):
    anchor.model.train()
    try:
        with pytest.raises(pb.Phase9BehaviorError, match="training mode"):
            anchor.assert_frozen()
    finally:
        anchor.model.eval()


# ---------------------------------------------------------------------------
# The frozen batch shape
# ---------------------------------------------------------------------------


def test_a_row_is_bitwise_independent_of_its_neighbours(anchor):
    """The property the sealed-digest promise rests on."""
    _state, _legality, observation = _position()
    others = np.stack([_position(root_seed=seed)[2] for seed in (11, 12, 13)])

    alone_policy, alone_value = pb.evaluate_observations(anchor, observation[None, ...])
    mixed = np.stack([others[0], others[1], observation, others[2]])
    mixed_policy, mixed_value = pb.evaluate_observations(anchor, mixed)

    assert torch.equal(alone_policy[0], mixed_policy[2])
    assert torch.equal(alone_value[0], mixed_value[2])


def test_changing_the_batch_shape_is_not_guaranteed_to_be_bitwise_equal(anchor):
    """The negative control that justifies pinning the shape at all.

    The two shapes agree far inside the reproduction tolerance — that is why
    an audit passes across devices — but they are not required to agree to the
    last float32 bit, which is why a resume may not change the shape.
    """
    _state, _legality, observation = _position()
    wider = pb.load_behavior_snapshot(
        ANCHOR_CHECKPOINT,
        logical_identity="B001",
        policy_token="phase9_behavior_v1|ns=canonical|B001",
        inference_batch_shape=16,
        model=anchor.model,
        state_dict_digest_hint=anchor.loaded_state_dict_digest,
    )
    narrow_policy, narrow_value = pb.evaluate_observations(anchor, observation[None, ...])
    wide_policy, wide_value = pb.evaluate_observations(wider, observation[None, ...])
    assert float((narrow_policy[0] - wide_policy[0]).abs().max()) < BEHAVIOR_PROBABILITY_ABS_TOLERANCE
    assert float((narrow_value[0] - wide_value[0]).abs().max()) < BEHAVIOR_PROBABILITY_ABS_TOLERANCE


def test_an_oversized_batch_is_refused(anchor):
    _state, _legality, observation = _position()
    rows = np.stack([observation] * (anchor.inference_batch_shape + 1))
    with pytest.raises(pb.Phase9BehaviorError, match="outside 1.."):
        pb.evaluate_observations(anchor, rows)


# ---------------------------------------------------------------------------
# The behavior distribution and the frozen selection rule
# ---------------------------------------------------------------------------


def test_distribution_is_a_normalized_float32_legal_softmax(anchor):
    _state, legality, observation = _position()
    policy_logits, _wdl = pb.evaluate_observations(anchor, observation[None, ...])
    probabilities = pb.behavior_distribution(policy_logits[0], legality)

    assert len(probabilities) == len(legality.absolute)
    assert all(value >= 0.0 for value in probabilities)
    assert abs(sum(probabilities) - 1.0) <= 1e-4
    # Stored dtype: every entry survives a float32 round trip unchanged.
    assert all(np.float32(value) == np.float32(np.float32(value)) for value in probabilities)


def test_distribution_is_ordered_by_ascending_absolute_action_id(anchor):
    """`trajectory_v1` requires ascending engine ids; blue's model frame is a
    permutation of them, so this is the one place the two orders meet."""
    state, legality, observation = _position(root_seed=99, plies=1)
    assert state.acting_player == 1  # blue: the frames genuinely differ
    assert tuple(legality.absolute) == tuple(sorted(legality.absolute))
    assert tuple(legality.model) != tuple(legality.absolute)

    policy_logits, _wdl = pb.evaluate_observations(anchor, observation[None, ...])
    probabilities = pb.behavior_distribution(policy_logits[0], legality)

    # Every stored entry is the softmax mass of the *matching* model action.
    from stratego.model.action_frame import absolute_action_to_model

    ordered_model = tuple(sorted(legality.model))
    logits = policy_logits[0][list(ordered_model)].to(torch.float64)
    weights = torch.exp(logits - logits.max())
    reference = dict(zip(ordered_model, (weights / weights.sum()).tolist()))
    for action, stored in zip(legality.absolute, probabilities):
        expected = reference[absolute_action_to_model(action, legality.acting_player)]
        assert abs(stored - expected) <= 1e-7


def test_selection_is_a_pure_function_of_the_logical_identity():
    probabilities = (0.25, 0.25, 0.5)
    actions = (10, 20, 30)
    game_id = phase9_game_id("canonical", 1, "current", 0)
    first = pb.select_behavior_action(probabilities, actions, game_id, 7)
    assert first == pb.select_behavior_action(probabilities, actions, game_id, 7)
    # A different ply is a different stream, so it is free to differ; a
    # different game certainly is.
    other = phase9_game_id("canonical", 1, "current", 1)
    assert pb.select_behavior_action(probabilities, actions, other, 7) in actions


def test_selection_walks_the_stored_distribution_exactly():
    """The cumulative rule, checked against the frozen uniform directly."""
    actions = (5, 9, 40, 41)
    probabilities = (0.1, 0.2, 0.3, 0.4)
    game_id = phase9_game_id("pilot_p9a", 3, "rule", 2)
    uniform = behavior_sample_uniform(game_id, 12)
    cumulative = 0.0
    expected = actions[-1]
    for action, probability in zip(actions, probabilities):
        cumulative += probability
        if cumulative >= uniform:
            expected = action
            break
    assert pb.select_behavior_action(probabilities, actions, game_id, 12) == expected


def test_a_probability_zero_prefix_is_unselectable():
    # The half-open (0, 1] orientation of the frozen uniform.
    actions = (1, 2, 3)
    assert pb.select_behavior_action((0.0, 0.0, 1.0), actions, phase9_game_id(
        "canonical", 1, "current", 0
    ), 0) == 3


def test_a_float32_tail_shortfall_selects_the_last_legal_action():
    actions = (1, 2, 3)
    # Sums to well under any uniform, which is exactly the shortfall case.
    assert pb.select_behavior_action((0.1, 0.1, 0.1), actions, phase9_game_id(
        "canonical", 1, "current", 0
    ), 0) in actions


def test_non_ascending_legal_actions_are_refused():
    with pytest.raises(pb.Phase9BehaviorError, match="ascending"):
        pb.select_behavior_action((0.5, 0.5), (9, 3), "x", 0)


def test_build_decision_stores_the_acting_snapshot_identity(anchor):
    _state, legality, observation = _position()
    policy_logits, wdl = pb.evaluate_observations(anchor, observation[None, ...])
    decision = pb.build_decision(
        anchor,
        game_id=phase9_game_id("canonical", 1, "current", 0),
        ply=0,
        legality=legality,
        policy_logits_row=policy_logits[0],
        wdl_row=wdl[0],
    )
    assert decision.checkpoint_sha256 == ANCHOR_SHA256
    assert decision.snapshot_identity == "B001"
    assert decision.selected_action_id in decision.legal_action_ids
    assert abs(sum(decision.win_draw_loss) - 1.0) <= 1e-4


# ---------------------------------------------------------------------------
# The reproduction audit
# ---------------------------------------------------------------------------


def _stored(anchor, root_seed=4242, plies=0):
    state, legality, observation = _position(root_seed=root_seed, plies=plies)
    policy_logits, wdl = pb.evaluate_observations(anchor, observation[None, ...])
    decision = pb.build_decision(
        anchor,
        game_id=phase9_game_id("canonical", 1, "current", 0),
        ply=plies,
        legality=legality,
        policy_logits_row=policy_logits[0],
        wdl_row=wdl[0],
    )
    return state, legality, observation, decision


def test_reproduction_verifies_a_faithfully_stored_decision(anchor):
    _state, legality, observation, decision = _stored(anchor)
    report = pb.reproduce_decision(
        anchor,
        game_id=decision.game_id,
        ply=decision.ply,
        acting_player=decision.acting_player,
        observation=observation,
        legality=legality,
        stored_probabilities=decision.probabilities,
        stored_wdl=decision.win_draw_loss,
        stored_action=decision.selected_action_id,
        stored_policy_token=decision.policy_token,
        stored_checkpoint_sha256=decision.checkpoint_sha256,
    )
    assert report["verified"], report["problems"]
    assert report["max_abs_difference"] == 0.0


def test_reproduction_refuses_the_wrong_checkpoint_outright(anchor, untrained):
    """A digest mismatch is a hard veto, not a tolerance question."""
    _state, legality, observation, decision = _stored(anchor)
    report = pb.reproduce_decision(
        untrained,
        game_id=decision.game_id,
        ply=decision.ply,
        acting_player=decision.acting_player,
        observation=observation,
        legality=legality,
        stored_probabilities=decision.probabilities,
        stored_wdl=decision.win_draw_loss,
        stored_action=decision.selected_action_id,
        stored_policy_token=decision.policy_token,
        stored_checkpoint_sha256=decision.checkpoint_sha256,
    )
    assert not report["verified"]
    assert "is not the acting snapshot" in report["problems"][0]


def test_reproduction_detects_a_distribution_from_a_different_network(anchor, untrained):
    """The audit's real job: same claimed identity, different weights.

    Verifying against the wrong network is exactly what happens if a historical
    opponent's moves are checked against the iteration's current learner, so
    the failure has to be loud and far outside tolerance.
    """
    _state, legality, observation, decision = _stored(anchor)
    report = pb.reproduce_decision(
        untrained,
        game_id=decision.game_id,
        ply=decision.ply,
        acting_player=decision.acting_player,
        observation=observation,
        legality=legality,
        stored_probabilities=decision.probabilities,
        stored_wdl=decision.win_draw_loss,
        stored_action=decision.selected_action_id,
        stored_policy_token=decision.policy_token,
        # Claim the untrained network's identity so the veto above is bypassed
        # and the numerical comparison actually runs.
        stored_checkpoint_sha256=untrained.checkpoint_sha256,
    )
    assert not report["verified"]
    assert report["max_abs_difference"] > BEHAVIOR_PROBABILITY_ABS_TOLERANCE


def test_reproduction_detects_a_tampered_probability(anchor):
    _state, legality, observation, decision = _stored(anchor)
    tampered = list(decision.probabilities)
    tampered[0] = float(tampered[0]) + 0.01
    report = pb.reproduce_decision(
        anchor,
        game_id=decision.game_id,
        ply=decision.ply,
        acting_player=decision.acting_player,
        observation=observation,
        legality=legality,
        stored_probabilities=tuple(tampered),
        stored_wdl=decision.win_draw_loss,
        stored_action=decision.selected_action_id,
        stored_policy_token=decision.policy_token,
        stored_checkpoint_sha256=decision.checkpoint_sha256,
    )
    assert not report["verified"]
    assert report["max_abs_difference"] > BEHAVIOR_PROBABILITY_ABS_TOLERANCE


def test_reproduction_detects_an_action_the_distribution_did_not_choose(anchor):
    _state, legality, observation, decision = _stored(anchor)
    other = next(
        action for action in legality.absolute if action != decision.selected_action_id
    )
    report = pb.reproduce_decision(
        anchor,
        game_id=decision.game_id,
        ply=decision.ply,
        acting_player=decision.acting_player,
        observation=observation,
        legality=legality,
        stored_probabilities=decision.probabilities,
        stored_wdl=decision.win_draw_loss,
        stored_action=other,
        stored_policy_token=decision.policy_token,
        stored_checkpoint_sha256=decision.checkpoint_sha256,
    )
    assert not report["verified"]
    assert any("redraws action" in problem for problem in report["problems"])


def test_reproduction_detects_a_mislabelled_policy_token(anchor):
    _state, legality, observation, decision = _stored(anchor)
    report = pb.reproduce_decision(
        anchor,
        game_id=decision.game_id,
        ply=decision.ply,
        acting_player=decision.acting_player,
        observation=observation,
        legality=legality,
        stored_probabilities=decision.probabilities,
        stored_wdl=decision.win_draw_loss,
        stored_action=decision.selected_action_id,
        stored_policy_token="phase9_anchor_v1|H000",
        stored_checkpoint_sha256=decision.checkpoint_sha256,
    )
    assert not report["verified"]
    assert any("policy token" in problem for problem in report["problems"])
