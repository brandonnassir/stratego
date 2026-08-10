"""Exact reconstruction tests (Phase 3 Agent 3).

Small deterministic scale, so these stay in the ordinary pytest run. The
1,000,000-decision gate and the snapshot-interval benchmark live in
`scripts/run_phase3_agent03.py`.

The property under test is exactness: a decision rebuilt from
`game record + nearest snapshot + subsequent actions` must be indistinguishable
from the live decision in state, acting player, observation, legal actions,
dense mask, public knowledge, belief target, stored action and identity.

The other property is separation: the privileged belief target must be reachable
only through its own field, never through the reconstructed observation.
"""

import numpy as np
import pytest

from stratego.engine.constants import OBSERVATION_SHAPE, TRAINING_RULES
from stratego.engine.legal_moves import legal_action_mask, legal_actions
from stratego.engine.observation import belief_target, build_observation
from stratego.engine.replay import replay_plies
from stratego.engine.state import state_fingerprint
from stratego.training.batch_simulation import BatchSimulator
from stratego.training.trajectory import (
    DEFAULT_SNAPSHOT_INTERVAL,
    SUPPORTED_SNAPSHOT_INTERVALS,
    TrajectoryError,
    collect_games,
    decode_game_record,
    encode_game_record,
)
from stratego.training.reconstruction import (
    COMPARISON_FIELDS,
    DecisionDigest,
    compare_digests,
    digest_live_decision,
    digest_reconstructed_decision,
    iter_reconstructed_decisions,
    observation_digest,
    public_knowledge_view,
    reconstruct_decision,
    reconstruct_state,
    restore_snapshot_entry,
    verify_decision,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def collect_with_digests(
    games: int = 2,
    *,
    root_seed: int = 6101,
    environments: int = 4,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    dense_mask: bool = True,
):
    """Play games while capturing a live digest of every decision.

    The digests are the ground truth: they are taken from the live game before
    each action is applied, so a later comparison is against what actually
    happened rather than against a second reconstruction.
    """
    simulator = BatchSimulator(environments, root_seed=root_seed, rules=TRAINING_RULES)
    live: "dict[str, list[DecisionDigest]]" = {}

    def on_decision(state, decision, builder):
        live.setdefault(state.game_id, []).append(
            digest_live_decision(
                state,
                decision,
                environment_id=builder.environment_id,
                generation=builder.generation,
                dense_mask=dense_mask,
            )
        )

    records = list(
        collect_games(
            simulator,
            games=games,
            snapshot_interval=snapshot_interval,
            on_decision=on_decision,
        )
    )
    return records, live


@pytest.fixture(scope="module")
def collected():
    return collect_with_digests(3)


@pytest.fixture(scope="module")
def records(collected):
    return collected[0]


@pytest.fixture(scope="module")
def live(collected):
    return collected[1]


@pytest.fixture(scope="module")
def record(records):
    return records[0]


# ---------------------------------------------------------------------------
# Exact reconstruction
# ---------------------------------------------------------------------------


def test_every_stored_decision_reconstructs_exactly(records, live):
    checked = 0
    for record in records:
        for decision in record.decisions:
            mismatches, _ = verify_decision(
                record, decision, live[record.game_id][decision.ply], dense_mask=True
            )
            assert mismatches == [], (record.game_id, decision.ply, mismatches)
            checked += 1
    assert checked > 100


def test_reconstruction_survives_a_serialisation_round_trip(records, live):
    for record in records:
        decoded = decode_game_record(encode_game_record(record))
        for decision in decoded.decisions:
            mismatches, _ = verify_decision(
                decoded, decision, live[record.game_id][decision.ply], dense_mask=True
            )
            assert mismatches == []


def test_reconstruction_matches_a_full_replay_from_the_start(record):
    """Snapshot-plus-delta agrees with replaying every action from ply 0."""
    for ply, state, _ in replay_plies(record.to_replay_record()):
        if ply >= record.final_ply:
            break
        expected = state_fingerprint(state, include_history=False)
        rebuilt, _ = reconstruct_state(record, ply)
        assert state_fingerprint(rebuilt, include_history=False) == expected
        assert rebuilt.acting_player == state.acting_player
        assert np.array_equal(build_observation(rebuilt), build_observation(state))


@pytest.mark.parametrize("interval", SUPPORTED_SNAPSHOT_INTERVALS)
def test_every_interval_reconstructs_exactly(interval):
    records, live = collect_with_digests(
        1, root_seed=7000 + interval, snapshot_interval=interval
    )
    record = records[0]
    assert record.snapshot_interval == interval
    for decision in record.decisions:
        mismatches, _ = verify_decision(
            record, decision, live[record.game_id][decision.ply], dense_mask=True
        )
        assert mismatches == []


def test_replayed_action_count_never_exceeds_the_interval(record):
    for decision in record.decisions:
        _, replayed = reconstruct_state(record, decision.ply)
        assert 0 <= replayed < record.snapshot_interval


def test_reconstruction_uses_the_nearest_snapshot(record):
    for decision in record.decisions:
        index = record.snapshot_index_for_ply(decision.ply)
        entry = record.snapshots[index]
        assert entry.ply <= decision.ply
        if index + 1 < len(record.snapshots):
            assert record.snapshots[index + 1].ply > decision.ply
        assert restore_snapshot_entry(record, index).total_moves == entry.ply


def test_sequential_and_random_access_agree(record):
    sequential = list(iter_reconstructed_decisions(record, dense_mask=True))
    assert len(sequential) == len(record.decisions)
    for rebuilt in sequential:
        independent = reconstruct_decision(record, rebuilt.ply, dense_mask=True)
        assert state_fingerprint(rebuilt.state, include_history=False) == state_fingerprint(
            independent.state, include_history=False
        )
        assert np.array_equal(rebuilt.observation, independent.observation)
        assert rebuilt.legal_action_ids == independent.legal_action_ids
        assert np.array_equal(rebuilt.legal_mask, independent.legal_mask)
        assert rebuilt.belief_target == independent.belief_target


# ---------------------------------------------------------------------------
# Field-by-field agreement with the frozen engine
# ---------------------------------------------------------------------------


def test_observation_is_the_frozen_engine_tensor(record):
    for decision in record.decisions[:40]:
        rebuilt = reconstruct_decision(record, decision.ply)
        assert rebuilt.observation.shape == OBSERVATION_SHAPE
        assert rebuilt.observation.dtype == np.float32
        assert np.array_equal(
            rebuilt.observation,
            build_observation(rebuilt.state, rebuilt.state.acting_player),
        )


def test_legal_list_and_dense_mask_agree(record):
    for decision in record.decisions[:40]:
        rebuilt = reconstruct_decision(record, decision.ply, dense_mask=True)
        assert list(rebuilt.legal_action_ids) == legal_actions(rebuilt.state)
        expected = legal_action_mask(rebuilt.state, list(rebuilt.legal_action_ids))
        assert np.array_equal(rebuilt.legal_mask, expected)
        assert int(rebuilt.legal_mask.sum()) == len(rebuilt.legal_action_ids)
        assert rebuilt.legal_mask[decision.selected_action_id] == 1


def test_dense_mask_is_only_built_on_request(record):
    assert reconstruct_decision(record, 0).legal_mask is None
    assert reconstruct_decision(record, 0, dense_mask=True).legal_mask is not None


def test_stored_legal_set_matches_the_reconstructed_one(record):
    for decision in record.decisions:
        rebuilt = reconstruct_decision(record, decision.ply, include_public_knowledge=False)
        assert rebuilt.legal_action_ids == decision.legal_action_ids
        assert decision.selected_action_id in rebuilt.legal_action_ids


def test_identity_and_generation_are_carried_through(record):
    for decision in record.decisions[:20]:
        rebuilt = reconstruct_decision(record, decision.ply, include_public_knowledge=False)
        assert rebuilt.game_id == record.game_id
        assert rebuilt.environment_id == record.environment_id
        assert rebuilt.generation == record.generation
        assert rebuilt.acting_player == decision.acting_player


def test_public_knowledge_reconstructs(record):
    for decision in record.decisions[:20]:
        rebuilt = reconstruct_decision(record, decision.ply)
        assert rebuilt.public_knowledge == public_knowledge_view(rebuilt.state)
        assert set(rebuilt.public_knowledge) == {"red", "blue"}


# ---------------------------------------------------------------------------
# Belief targets stay a training target
# ---------------------------------------------------------------------------


def test_belief_target_matches_the_privileged_engine_target(record):
    for decision in record.decisions[:40]:
        rebuilt = reconstruct_decision(record, decision.ply, include_public_knowledge=False)
        assert rebuilt.belief_target == belief_target(
            rebuilt.state, rebuilt.state.acting_player
        )


def test_belief_target_is_a_separate_field_from_the_observation(record):
    rebuilt = reconstruct_decision(record, record.final_ply // 2)
    assert isinstance(rebuilt.observation, np.ndarray)
    assert isinstance(rebuilt.belief_target, list)
    assert rebuilt.belief_target is not rebuilt.observation


def test_belief_target_never_enters_the_reconstructed_observation(record):
    """Hiding an opponent identity must not change the observation tensor.

    The belief target names the true type of every hidden opponent piece. If any
    of that reached the policy input, changing a hidden type would change the
    tensor. It does not.
    """
    from stratego.engine.permutation import permute_hidden_identities

    ply = record.final_ply // 2
    rebuilt = reconstruct_decision(record, ply, include_public_knowledge=False)
    observer = rebuilt.state.acting_player
    baseline = build_observation(rebuilt.state, observer)
    permuted_state, _ = permute_hidden_identities(
        rebuilt.state, observer, np.random.default_rng(17)
    )
    assert np.array_equal(build_observation(permuted_state, observer), baseline)


def test_belief_target_is_absent_from_the_stored_record(record):
    """Nothing privileged is serialised: the codec has no belief field."""
    payload = encode_game_record(record)
    for item in reconstruct_decision(record, 0, include_public_knowledge=False).belief_target:
        assert item["piece_id"].encode() not in payload


# ---------------------------------------------------------------------------
# Digest and comparison surface
# ---------------------------------------------------------------------------


def test_comparison_surface_covers_every_required_field():
    categories = {category for category, _ in COMPARISON_FIELDS}
    assert categories == {
        "identity_generation",
        "acting_player",
        "selected_action",
        "state",
        "observation",
        "legal_list",
        "belief_target",
        "public_knowledge",
        "legal_mask",
    }


def test_digest_comparison_detects_a_changed_observation(record, live):
    decision = record.decisions[3]
    rebuilt = reconstruct_decision(record, decision.ply, dense_mask=True)
    digest = digest_reconstructed_decision(rebuilt, decision)
    assert compare_digests(live[record.game_id][decision.ply], digest) == []

    tampered = rebuilt.observation.copy()
    tampered[0, 0, 0] = 1.0 - tampered[0, 0, 0]
    from dataclasses import replace as dataclass_replace

    broken = dataclass_replace(digest, observation=observation_digest(tampered))
    assert ("observation", "observation") in compare_digests(
        live[record.game_id][decision.ply], broken
    )


def test_digest_comparison_skips_an_absent_dense_mask(record, live):
    decision = record.decisions[2]
    rebuilt = reconstruct_decision(record, decision.ply, dense_mask=False)
    digest = digest_reconstructed_decision(rebuilt, decision)
    assert digest.legal_mask is None
    assert compare_digests(live[record.game_id][decision.ply], digest) == []


def test_digest_requires_public_knowledge(record):
    rebuilt = reconstruct_decision(record, 0, include_public_knowledge=False)
    with pytest.raises(TrajectoryError, match="include_public_knowledge"):
        digest_reconstructed_decision(rebuilt, record.decisions[0])


# ---------------------------------------------------------------------------
# Error surface
# ---------------------------------------------------------------------------


def test_reconstruction_rejects_an_out_of_range_ply(record):
    with pytest.raises(TrajectoryError, match="outside game"):
        reconstruct_state(record, record.final_ply + 1)


def test_restore_rejects_an_out_of_range_snapshot(record):
    with pytest.raises(TrajectoryError, match="outside game"):
        restore_snapshot_entry(record, len(record.snapshots))


def test_reconstruction_rejects_a_state_at_the_wrong_ply(record):
    state, _ = reconstruct_state(record, 4)
    with pytest.raises(TrajectoryError, match="not 5"):
        reconstruct_decision(record, 5, state=state)


def test_decision_lookup_rejects_a_missing_ply(record):
    with pytest.raises(TrajectoryError, match="stored no decision"):
        record.decision_at(record.final_ply + 10)
