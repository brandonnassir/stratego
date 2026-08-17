"""Shared Phase 8 training fixtures.

The warm-start example/dataset tests all need a small committed corpus with
every supervision-weight class represented. Generating it once per session
keeps the suite's cost at a handful of played games.
"""

from __future__ import annotations

import pytest

from stratego.training import synthetic_corpus as sc
from stratego.training.warmstart_seed import synthetic_game_id

#: The mini corpus: one game per interesting weight pairing, across all three
#: splits, kept short by choosing cheap teachers where the pairing allows it.
WARMSTART_MINI_GAME_IDS = (
    synthetic_game_id("train", "strategic_rule_based@1.1.0", "random_legal@1.0.0", 0),
    synthetic_game_id("train", "tactical_rule_based@1.0.0", "basic_heuristic@1.0.0", 0),
    synthetic_game_id("train", "random_legal@1.0.0", "random_legal@1.0.0", 0),
    synthetic_game_id("validation", "basic_heuristic@1.0.0", "stress_chaos@1.0.0", 0),
    synthetic_game_id("validation", "stress_draw_seeker@1.0.0", "strategic_rule_based@1.1.0", 1),
    synthetic_game_id("test", "random_legal@1.0.0", "stress_scout_rush@1.0.0", 0),
)


@pytest.fixture(scope="session")
def warmstart_mini_corpus(tmp_path_factory):
    """`(root, game_ids)` of a committed six-game corpus, generated once."""
    root = tmp_path_factory.mktemp("warmstart_mini_corpus")
    sc.generate_corpus(
        root, worker_count=1, chunks_per_worker=1, game_ids=WARMSTART_MINI_GAME_IDS
    )
    return root, WARMSTART_MINI_GAME_IDS


#: The Phase 9 mini rollout: one real game per learner-control mode, played
#: once per session. The Agent 5 trainer tests all need a *sealed* iteration,
#: and playing four games is far cheaper than the 2,048 a production seal
#: demands — so the fixture seals by hand and the trainer's own
#: `require_full_schedule` check is exercised separately, as a negative control.
PHASE9_MINI_GAMES = (("current", 0), ("historical", 0), ("historical", 1), ("rule", 0))

PHASE8_ANCHOR_PATH = "checkpoints/phase8/warmstart_c1_v1.pt"
PHASE8_ANCHOR_SHA256 = (
    "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca"
)


@pytest.fixture(scope="session")
def phase9_mini_rollout(tmp_path_factory):
    """`(root, namespace, iteration, behavior_snapshot)` of a sealed rollout."""
    from stratego.training import phase9_collector as pc
    from stratego.training import phase9_rollout_store as store
    from stratego.training.phase9_contract import (
        PHASE9_POPULATION_VERSION,
        PHASE9_ROLLOUT_SCHEDULE_VERSION,
        contract_digest,
    )
    from stratego.training.phase9_schedule import rebuild_scheduled_game
    from stratego.training.phase9_seed import phase9_game_id

    root = tmp_path_factory.mktemp("phase9_mini_rollout")
    resolver = pc.SnapshotResolver(device="cpu", inference_batch_shape=4)
    behavior = resolver.resolve(
        PHASE8_ANCHOR_PATH,
        logical_identity="B001",
        policy_token="phase9_behavior_v1|ns=canonical|B001",
        expected_sha256=PHASE8_ANCHOR_SHA256,
    )
    anchor = resolver.resolve(
        PHASE8_ANCHOR_PATH,
        logical_identity="H000",
        policy_token="phase9_anchor_v1|H000",
        expected_sha256=PHASE8_ANCHOR_SHA256,
    )
    participants = pc.IterationParticipants(behavior=behavior, historical={"H000": anchor})

    writer = store.Phase9RolloutWriter(root, namespace="canonical", iteration=1, worker_id=0)
    for bucket, ordinal in PHASE9_MINI_GAMES:
        scheduled = rebuild_scheduled_game(phase9_game_id("canonical", 1, bucket, ordinal))
        runner = pc.play_game(scheduled, participants)
        metadata = store.build_rollout_metadata(
            scheduled,
            runner.record,
            setup_provenance=runner.assignment.provenance,
            behavior_checkpoint_sha256=PHASE8_ANCHOR_SHA256,
            opponent_checkpoint_sha256=(
                PHASE8_ANCHOR_SHA256
                if scheduled.opponent_kind == "historical_snapshot"
                else None
            ),
            learner_decision_count=runner.learner_decision_count,
            population_version=PHASE9_POPULATION_VERSION,
            schedule_version=PHASE9_ROLLOUT_SCHEDULE_VERSION,
            contract_digest=contract_digest(),
        )
        writer.write_game(runner.record, metadata)
    writer.close()

    reader = store.Phase9RolloutReader(root, "canonical", 1)
    store.write_iteration_state(
        root,
        "canonical",
        1,
        "SEALED",
        sealed_rollout_digest=store.sealed_rollout_digest(reader.commits),
        behavior_snapshot_id="B001",
        behavior_checkpoint_sha256=PHASE8_ANCHOR_SHA256,
    )
    return root, "canonical", 1, behavior
