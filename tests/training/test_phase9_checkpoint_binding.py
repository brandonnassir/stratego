"""Phase 9 Agent 5: two genuinely different checkpoints in one matchup.

Agent 3's acceptance soak could not demonstrate this. Every iteration it
collected was iteration 1, where the current learner `B001` and the historical
anchor `H000` are the *same* Phase 8 file — so "each side was verified against
its own checkpoint" and "both sides were verified against the same checkpoint"
produced identical evidence, and a swapped binding would have passed.

This fixture removes that ambiguity with the two real checkpoints the
repository already has: the accepted Phase 8 checkpoint and the canonical
untrained initialization. They are genuinely different weights with different
SHA-256s, so:

```text
each side against its own checkpoint      -> verified
learner decisions against the opponent    -> fails numerically
opponent decisions against the learner    -> fails numerically
```

The swapped cases deliberately rewrite the recorded checkpoint digest as well,
because otherwise the reproducer's digest guard rejects them before a single
forward pass runs — and a digest string comparison is not the claim being
tested. Both defenses are exercised: the guard, and the numbers behind it.
"""

from __future__ import annotations

import dataclasses

import pytest

from stratego.engine.legal_moves import legal_action_mask, legal_actions
from stratego.engine.observation import build_observation
from stratego.engine.state import create_game
from stratego.engine.transition import apply_action
from stratego.model.policy_adapter import prepare_legality
from stratego.training import phase9_behavior as pb
from stratego.training import phase9_collector as pc
from stratego.training import phase9_rollout_store as store
from stratego.training.phase9_contract import (
    PHASE9_POPULATION_VERSION,
    PHASE9_ROLLOUT_SCHEDULE_VERSION,
    contract_digest,
)
from stratego.training.phase9_schedule import rebuild_scheduled_game
from stratego.training.phase9_seed import phase9_game_id
from stratego.training.warmstart_contract import CORPUS_RULES

from .conftest import PHASE8_ANCHOR_PATH, PHASE8_ANCHOR_SHA256

#: The canonical untrained initialization: a real C1 checkpoint with entirely
#: different weights, which is exactly what a historical archive member is
#: relative to a later learner.
UNTRAINED_PATH = "checkpoints/phase8/warmstart_c1_v1_initialisation.pt"
UNTRAINED_SHA256 = "01c907eeef86ec04121db55ccffb9365e8df27fdf05921b921947d4af365754c"

BATCH_SHAPE = 4


@pytest.fixture(scope="module")
def matchup():
    """One historical game whose two sides really are two different networks.

    `learner` plays as the current policy; `opponent` is bound to the untrained
    checkpoint and stands in for a historical archive member.
    """
    resolver = pc.SnapshotResolver(device="cpu", inference_batch_shape=BATCH_SHAPE)
    learner = resolver.resolve(
        PHASE8_ANCHOR_PATH,
        logical_identity="B002",
        policy_token="phase9_behavior_v1|ns=canonical|B002",
        expected_sha256=PHASE8_ANCHOR_SHA256,
    )
    opponent = resolver.resolve(
        UNTRAINED_PATH,
        logical_identity="H000",
        policy_token="phase9_anchor_v1|H000",
        expected_sha256=UNTRAINED_SHA256,
    )
    assert learner.checkpoint_sha256 != opponent.checkpoint_sha256
    assert learner.loaded_state_dict_digest != opponent.loaded_state_dict_digest

    # Iteration 2, because that is the first iteration whose scheduled learner
    # token is not `B001`: the frozen schedule pins each side's token, so a
    # matchup with a genuinely different learner has to be scheduled as one.
    participants = pc.IterationParticipants(behavior=learner, historical={"H000": opponent})
    scheduled = rebuild_scheduled_game(phase9_game_id("canonical", 2, "historical", 0))
    runner = pc.play_game(scheduled, participants)
    metadata = store.build_rollout_metadata(
        scheduled,
        runner.record,
        setup_provenance=runner.assignment.provenance,
        behavior_checkpoint_sha256=learner.checkpoint_sha256,
        opponent_checkpoint_sha256=opponent.checkpoint_sha256,
        learner_decision_count=runner.learner_decision_count,
        population_version=PHASE9_POPULATION_VERSION,
        schedule_version=PHASE9_ROLLOUT_SCHEDULE_VERSION,
        contract_digest=contract_digest(),
    )
    return runner.record, metadata, learner, opponent, scheduled


def requests_for(record, metadata, wanted_player, *, limit=8):
    """Replay the game and rebuild what a re-check needs for one side."""
    state = create_game(
        record.red_setup, record.blue_setup, rules=CORPUS_RULES, game_id=record.game_id
    )
    built = []
    for decision in record.decisions:
        legal = legal_actions(state)
        actor = int(state.acting_player)
        if actor == wanted_player and len(built) < limit:
            legality = prepare_legality(legal, legal_action_mask(state, legal), actor)
            built.append(
                pb.ReproductionRequest(
                    game_id=record.game_id,
                    ply=int(decision.ply),
                    acting_player=actor,
                    observation=build_observation(state, actor),
                    legality=legality,
                    stored_probabilities=tuple(
                        float(value) for value in decision.old_probabilities
                    ),
                    stored_wdl=tuple(
                        float(value) for value in decision.win_draw_loss_prediction
                    ),
                    stored_action=int(decision.selected_action_id),
                    stored_policy_token=decision.collection_policy_version,
                    stored_checkpoint_sha256=(
                        metadata["behavior_checkpoint_sha256"]
                        if actor == _learner_player(metadata)
                        else metadata["opponent_checkpoint_sha256"]
                    ),
                )
            )
        apply_action(state, decision.selected_action_id, legal=legal)
    return built


def _learner_player(metadata):
    from stratego.engine.constants import BLUE, RED

    return RED if metadata["learner_color"] == "red" else BLUE


def _opponent_player(metadata):
    from stratego.engine.constants import BLUE, RED

    return BLUE if metadata["learner_color"] == "red" else RED


def rebind(requests, digest):
    """The same decisions, claiming a different checkpoint produced them."""
    return [
        dataclasses.replace(request, stored_checkpoint_sha256=digest)
        for request in requests
    ]


# ---------------------------------------------------------------------------
# The fixture really does hold two different networks
# ---------------------------------------------------------------------------


def test_the_matchup_binds_two_different_checkpoints(matchup):
    _record, metadata, learner, opponent, _scheduled = matchup
    assert metadata["behavior_checkpoint_sha256"] == learner.checkpoint_sha256
    assert metadata["opponent_checkpoint_sha256"] == opponent.checkpoint_sha256
    assert metadata["behavior_checkpoint_sha256"] != metadata["opponent_checkpoint_sha256"]
    assert metadata["learner_control"] in ("red", "blue")


def test_each_side_verifies_against_its_own_checkpoint(matchup):
    record, metadata, learner, opponent, _scheduled = matchup
    for player, snapshot in (
        (_learner_player(metadata), learner),
        (_opponent_player(metadata), opponent),
    ):
        requests = requests_for(record, metadata, player)
        assert requests, "the game produced no decisions for this side"
        reports = pb.reproduce_decisions(snapshot, requests)
        assert all(report["verified"] for report in reports), [
            report for report in reports if not report["verified"]
        ][:2]
        assert max(report["max_abs_difference"] for report in reports) < 1e-4


def test_swapping_the_bindings_makes_the_verification_fail(matchup):
    """The claim Agent 3's `B001 == H000` soak could not make."""
    record, metadata, learner, opponent, _scheduled = matchup
    learner_requests = requests_for(record, metadata, _learner_player(metadata))
    opponent_requests = requests_for(record, metadata, _opponent_player(metadata))

    # Learner decisions, evaluated against the historical opponent's network.
    swapped = pb.reproduce_decisions(
        opponent, rebind(learner_requests, opponent.checkpoint_sha256)
    )
    assert not any(report["verified"] for report in swapped)
    assert max(report["max_abs_difference"] for report in swapped) > 1e-2

    # ...and the mirror image.
    mirrored = pb.reproduce_decisions(
        learner, rebind(opponent_requests, learner.checkpoint_sha256)
    )
    assert not any(report["verified"] for report in mirrored)
    assert max(report["max_abs_difference"] for report in mirrored) > 1e-2


def test_the_digest_guard_rejects_a_swap_before_any_forward_pass(matchup):
    """The independent second line of defense, with digests left as recorded."""
    record, metadata, learner, opponent, _scheduled = matchup
    reports = pb.reproduce_decisions(
        opponent, requests_for(record, metadata, _learner_player(metadata))
    )
    assert not any(report["verified"] for report in reports)
    assert all(report["max_abs_difference"] is None for report in reports)
    assert all(
        "is not the acting snapshot" in " ".join(report["problems"])
        for report in reports
    )


def test_the_acting_snapshot_resolver_routes_each_side_correctly(matchup):
    """`acting_snapshot_for` is the single point that decides whose move it is."""
    _record, metadata, learner, opponent, scheduled = matchup
    participants = pc.IterationParticipants(behavior=learner, historical={"H000": opponent})
    learner_player = _learner_player(metadata)
    opponent_player = _opponent_player(metadata)
    assert (
        pc.acting_snapshot_for(scheduled, participants, learner_player).checkpoint_sha256
        == learner.checkpoint_sha256
    )
    assert (
        pc.acting_snapshot_for(scheduled, participants, opponent_player).checkpoint_sha256
        == opponent.checkpoint_sha256
    )
