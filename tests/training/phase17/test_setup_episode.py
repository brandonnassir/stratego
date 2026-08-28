"""Phase 17 Agent 3 sections 5 and 8: outcome binding and queue discipline."""

import numpy as np
import pytest

from stratego.engine.constants import BLUE, RED
from stratego.training.phase17.setup_contract import (
    SETUP_EPISODE_SCHEMA_VERSION,
    Phase17SetupError,
)
from stratego.training.phase17.setup_episode import (
    SetupEpisode,
    SetupEpisodeQueue,
    attach_setup_episodes,
    outcome_for,
    wdl_class,
)

RUN_ID = "RUN-TEST-A"


# -- outcome signs ----------------------------------------------------------


def test_red_win_blue_win_and_draw_signs_are_independent():
    """Section 5: test Red win, Blue win, and draw signs independently."""
    assert outcome_for("red_win", RED) == 1
    assert outcome_for("red_win", BLUE) == -1
    assert outcome_for("blue_win", BLUE) == 1
    assert outcome_for("blue_win", RED) == -1
    assert outcome_for("draw", RED) == 0
    assert outcome_for("draw", BLUE) == 0


def test_wdl_classes_are_owner_relative():
    assert (wdl_class(1), wdl_class(0), wdl_class(-1)) == (0, 1, 2)
    with pytest.raises(Phase17SetupError):
        wdl_class(2)


def test_an_unknown_result_is_refused():
    with pytest.raises(Phase17SetupError, match="unknown terminal result"):
        outcome_for("red_resigned", RED)


def test_both_sides_of_one_game_take_the_same_result_from_their_own_side(
    red_samples, blue_samples
):
    pair = attach_setup_episodes(
        red_samples[0], blue_samples[0], run_id=RUN_ID, game_id="game-0"
    )
    pair.complete("blue_win")
    assert pair.red.outcome == -1
    assert pair.blue.outcome == 1
    assert pair.red.terminal_result == pair.blue.terminal_result == "blue_win"


def test_attach_requires_red_then_blue(red_samples, blue_samples):
    with pytest.raises(Phase17SetupError, match="RED sample and a BLUE sample"):
        attach_setup_episodes(blue_samples[0], red_samples[0], run_id=RUN_ID, game_id="g")


def test_rebinding_a_different_outcome_is_fatal(red_samples, blue_samples):
    """Section 8: outcome rebinding must be impossible or fatal."""
    pair = attach_setup_episodes(red_samples[1], blue_samples[1], run_id=RUN_ID, game_id="g1")
    pair.complete("red_win")
    pair.red.complete("red_win")  # idempotent for the same result
    with pytest.raises(Phase17SetupError, match="refusing to rebind"):
        pair.red.complete("draw")


# -- schema -----------------------------------------------------------------


def test_the_episode_keeps_its_generating_snapshot(red_samples):
    episode = SetupEpisode.create(red_samples[0], run_id=RUN_ID, game_id="g")
    assert episode.setup_model_state_digest == red_samples[0].setup_model_state_digest
    assert episode.schema_version == SETUP_EPISODE_SCHEMA_VERSION
    assert episode.orientation_rule_version == "phase15_orientation_rule_v1"
    assert episode.state == "open"


def test_policy_age_and_compatibility_are_reported_not_enforced(red_samples):
    """Row S05: a setup is off-policy by design; age is telemetry."""
    episode = SetupEpisode.create(red_samples[0], run_id=RUN_ID, game_id="g")
    assert episode.policy_age(5) == 5
    assert episode.compatible_with(episode.setup_model_state_digest)
    assert not episode.compatible_with("something-else")


def test_owner_perspective_must_equal_colour(red_samples):
    episode = SetupEpisode.create(red_samples[0], run_id=RUN_ID, game_id="g")
    document = episode.to_document()
    document["owner_perspective"] = BLUE
    with pytest.raises(Phase17SetupError, match="owner_perspective must equal color"):
        SetupEpisode.from_document(document)


def test_a_document_round_trip_preserves_every_array(red_samples):
    episode = SetupEpisode.create(red_samples[2], run_id=RUN_ID, game_id="g2").complete("draw")
    restored = SetupEpisode.from_document(episode.to_document())
    assert restored.canonical_setup == episode.canonical_setup
    assert restored.engine_setup == episode.engine_setup
    assert restored.outcome == 0
    for name in (
        "tokens",
        "inventory_masks",
        "behavior_probabilities",
        "behavior_log_probabilities",
        "suffix_information_content",
        "prefix_wdl_predictions",
        "prefix_conditional_entropy_predictions",
    ):
        assert np.array_equal(getattr(restored, name), getattr(episode, name)), name
    assert restored.identity() == episode.identity()


def test_a_missing_required_field_fails_closed(red_samples):
    """Encoding rules: a required field that is absent fails closed."""
    document = SetupEpisode.create(red_samples[0], run_id=RUN_ID, game_id="g").to_document()
    del document["behavior_probabilities"]
    with pytest.raises(Phase17SetupError, match="missing 'behavior_probabilities'"):
        SetupEpisode.from_document(document)


def test_a_foreign_schema_version_is_refused(red_samples):
    document = SetupEpisode.create(red_samples[0], run_id=RUN_ID, game_id="g").to_document()
    document["schema_version"] = "phase17_setup_episode_v2"
    with pytest.raises(Phase17SetupError, match="episode schema"):
        SetupEpisode.from_document(document)


# -- the queue --------------------------------------------------------------


def test_an_open_episode_cannot_enter_the_queue(red_samples):
    queue = SetupEpisodeQueue(capacity=8, max_age_iterations=4)
    episode = SetupEpisode.create(red_samples[0], run_id=RUN_ID, game_id="g")
    assert queue.enqueue(episode) is False
    assert episode.state == "rejected"
    assert episode.rejected_reason == "episode has no terminal result"
    assert queue.rejected_count == 1


def test_a_duplicate_is_rejected_with_a_reason_never_dropped(completed_episodes):
    queue = SetupEpisodeQueue(capacity=64, max_age_iterations=4)
    assert queue.enqueue(completed_episodes[0]) is True
    assert queue.enqueue(completed_episodes[0]) is False
    assert queue.rejections[-1]["reason"] == "duplicate (run_id, game_id, color)"
    assert queue.enqueued_count == 1
    assert queue.rejected_count == 1


def test_capacity_overflow_raises_rather_than_evicting(completed_episodes):
    """Section 8: silent dropping is prohibited, so there is no eviction path."""
    queue = SetupEpisodeQueue(capacity=2, max_age_iterations=4)
    queue.enqueue(completed_episodes[0])
    queue.enqueue(completed_episodes[1])
    with pytest.raises(Phase17SetupError, match="refusing to evict"):
        queue.enqueue(completed_episodes[2])


def test_consumption_is_fifo_and_happens_exactly_once(completed_episodes):
    queue = SetupEpisodeQueue(capacity=64, max_age_iterations=4)
    for episode in completed_episodes[:6]:
        queue.enqueue(episode)
    taken = queue.consume(4, setup_iteration=3)
    assert [episode.game_id for episode in taken] == [
        episode.game_id for episode in completed_episodes[:4]
    ]
    assert all(episode.state == "consumed" for episode in taken)
    assert all(episode.consumed_in_setup_iteration == 3 for episode in taken)
    assert len(queue) == 2
    # A second pass sees only what is left; nothing is served twice.
    again = queue.consume(4, setup_iteration=4)
    assert len(again) == 2
    assert queue.consumed_count == 6


def test_a_short_queue_skips_explicitly_rather_than_shrinking_the_batch(completed_episodes):
    queue = SetupEpisodeQueue(capacity=64, max_age_iterations=4)
    for episode in completed_episodes[:3]:
        queue.enqueue(episode)
    assert queue.consume_exact(8, setup_iteration=1) == []
    assert queue.skip_count == 1
    assert len(queue) == 3  # nothing was consumed


def test_re_enqueueing_a_consumed_episode_is_rejected(completed_episodes):
    queue = SetupEpisodeQueue(capacity=64, max_age_iterations=4)
    queue.enqueue(completed_episodes[0])
    queue.consume(1, setup_iteration=1)
    second = SetupEpisodeQueue(capacity=64, max_age_iterations=4)
    assert second.enqueue(completed_episodes[0]) is False
    assert second.rejections[-1]["reason"] == "already consumed"


def test_queue_telemetry_reports_everything_section_8_names(completed_episodes):
    queue = SetupEpisodeQueue(capacity=64, max_age_iterations=4)
    for episode in completed_episodes[:5]:
        queue.enqueue(episode)
    telemetry = queue.telemetry(setup_iteration=6)
    assert telemetry.depth == 5
    assert telemetry.oldest_age == 6
    assert telemetry.mean_age == 6.0
    assert telemetry.enqueued_count == 5
    assert telemetry.consumed_count == 0
    assert queue.over_age(setup_iteration=6) == 5
    assert queue.over_age(setup_iteration=2) == 0


def test_the_queue_state_round_trips(completed_episodes):
    queue = SetupEpisodeQueue(capacity=64, max_age_iterations=4)
    for episode in completed_episodes[:5]:
        queue.enqueue(episode)
    queue.enqueue(completed_episodes[0])  # a recorded rejection
    restored = SetupEpisodeQueue.from_state_document(queue.state_document())
    assert len(restored) == len(queue)
    assert restored.rejected_count == queue.rejected_count
    assert restored.rejections == queue.rejections
    assert [episode.identity() for episode in restored._queue] == [
        episode.identity() for episode in queue._queue
    ]
    # And the restored queue still refuses the duplicates the original saw.
    assert restored.enqueue(completed_episodes[0]) is False
