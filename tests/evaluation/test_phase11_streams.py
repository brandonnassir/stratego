"""Phase 11 Agent 4: the materialized random-stream identity universe.

The audit's whole value rests on two properties of this module: it must
enumerate one entry per *distinct logical identity* (so intentional reuse
never inflates the count and never looks like a collision), and its
collision check must actually fire when two identities do share a seed.
Both are tested here on synthetic universes where the answer is known, so a
regression fails in the suite rather than in a thirty-million-identity run.
"""

import numpy as np
import pytest

from stratego.evaluation.phase11_streams import (
    AGENT4_MATERIALIZED_DOMAINS,
    Phase11StreamAuditError,
    combined_collision_audit,
    safety_trial_seeds,
    token_identity,
    tokens_for,
    verify_fast_path,
    world_stream_seeds,
)
from stratego.training.phase11_contract import BELIEF_SAMPLER_VERSION
from stratego.training.phase11_seed import (
    DOMAIN_SAFETY_TRIAL,
    DOMAIN_WORLD_CATEGORICAL,
    DOMAIN_WORLD_ORDER,
    DOMAIN_WORLD_SAMPLE,
    SAFETY_PURPOSES,
    derive_phase11_seed,
    phase11_safety_trial_id,
    phase11_sample_token,
    world_order_key,
    world_sample_seed,
)

IDENTITY_A = "aa" * 32
IDENTITY_B = "bb" * 32
SLOTS = {IDENTITY_A: (0, 3, 7, 11), IDENTITY_B: (1, 2)}


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def test_tokens_are_one_per_identity_and_ordinal():
    tokens = tokens_for([IDENTITY_A, IDENTITY_B], range(4), BELIEF_SAMPLER_VERSION)
    assert len(tokens) == 8
    assert all(token_identity(token) in SLOTS for token in tokens)


def test_reissuing_the_same_identity_and_ordinal_yields_one_token():
    """The eight topology legs reuse identities on purpose."""
    first = tokens_for([IDENTITY_A], range(64), BELIEF_SAMPLER_VERSION)
    again = tokens_for([IDENTITY_A], range(64), BELIEF_SAMPLER_VERSION)
    assert first == again
    assert len(first | again) == 64


def test_the_sampler_version_is_part_of_the_token():
    learned = tokens_for([IDENTITY_A], [0], BELIEF_SAMPLER_VERSION)
    baseline = tokens_for([IDENTITY_A], [0], "count_uniform_world_sampler_v1")
    assert learned != baseline
    assert len(learned | baseline) == 2


def test_token_identity_recovers_the_public_state():
    token = phase11_sample_token(BELIEF_SAMPLER_VERSION, IDENTITY_A, 5)
    assert token_identity(token) == IDENTITY_A


# ---------------------------------------------------------------------------
# World-stream enumeration
# ---------------------------------------------------------------------------


def test_world_streams_enumerate_one_entry_per_logical_identity():
    tokens = tokens_for([IDENTITY_A, IDENTITY_B], range(3), BELIEF_SAMPLER_VERSION)
    arrays = world_stream_seeds(tokens, SLOTS)
    assert arrays[DOMAIN_WORLD_SAMPLE].size == len(tokens)
    expected_children = 3 * (len(SLOTS[IDENTITY_A]) + len(SLOTS[IDENTITY_B]))
    assert arrays[DOMAIN_WORLD_ORDER].size == expected_children
    assert arrays[DOMAIN_WORLD_CATEGORICAL].size == expected_children


def test_world_stream_enumeration_is_order_independent():
    tokens = tokens_for([IDENTITY_A, IDENTITY_B], range(3), BELIEF_SAMPLER_VERSION)
    first = world_stream_seeds(tokens, SLOTS)
    second = world_stream_seeds(list(reversed(sorted(tokens))), SLOTS)
    for domain in first:
        assert np.array_equal(first[domain], second[domain])


def test_world_stream_seeds_match_the_accepted_public_helpers():
    tokens = sorted(tokens_for([IDENTITY_A], range(2), BELIEF_SAMPLER_VERSION))
    arrays = world_stream_seeds(tokens, SLOTS)
    assert int(arrays[DOMAIN_WORLD_SAMPLE][0]) == world_sample_seed(tokens[0])
    for offset, slot in enumerate(SLOTS[IDENTITY_A]):
        assert int(arrays[DOMAIN_WORLD_ORDER][offset]) == world_order_key(
            tokens[0], slot
        )


def test_an_unknown_public_state_is_refused_rather_than_skipped():
    tokens = tokens_for(["cc" * 32], [0], BELIEF_SAMPLER_VERSION)
    with pytest.raises(Phase11StreamAuditError):
        world_stream_seeds(tokens, SLOTS)


def test_a_repeated_piece_slot_is_refused():
    with pytest.raises(Phase11StreamAuditError):
        world_stream_seeds(
            tokens_for([IDENTITY_A], [0], BELIEF_SAMPLER_VERSION),
            {IDENTITY_A: (3, 3, 5)},
        )


# ---------------------------------------------------------------------------
# safety_trial enumeration
# ---------------------------------------------------------------------------


def draws(counts):
    return {
        phase11_safety_trial_id(ordinal): dict(zip(SAFETY_PURPOSES, entry))
        for ordinal, entry in enumerate(counts)
    }


def test_safety_enumeration_covers_every_consumed_draw_ordinal():
    arrays = safety_trial_seeds(draws([(1, 1, 1), (2, 33, 1)]))
    assert arrays[f"{DOMAIN_SAFETY_TRIAL}:state_selection"].size == 3
    assert arrays[f"{DOMAIN_SAFETY_TRIAL}:truth_permutation"].size == 34
    assert arrays[f"{DOMAIN_SAFETY_TRIAL}:sample_check"].size == 2


def test_safety_enumeration_subsumes_agent_1_draw_zero():
    """Agent 1 pins draw 0; the attack consumes draw 0 and beyond."""
    trial = phase11_safety_trial_id(0)
    arrays = safety_trial_seeds(draws([(3, 3, 1)]))
    for purpose in SAFETY_PURPOSES:
        seeds = arrays[f"{DOMAIN_SAFETY_TRIAL}:{purpose}"]
        expected = derive_phase11_seed(DOMAIN_SAFETY_TRIAL, trial, purpose, 0)
        assert expected in set(int(value) for value in seeds)


def test_safety_enumeration_has_no_internal_duplicates():
    arrays = safety_trial_seeds(draws([(4, 7, 1)] * 8))
    for seeds in arrays.values():
        assert np.unique(seeds).size == seeds.size


# ---------------------------------------------------------------------------
# The combined injectivity check
# ---------------------------------------------------------------------------


def test_a_clean_universe_reports_no_collisions():
    audit = combined_collision_audit(
        {
            "alpha": np.array([1, 2, 3], dtype=np.uint64),
            "beta": np.array([4, 5], dtype=np.uint64),
        }
    )
    assert audit["total_identities"] == 5
    assert audit["distinct_seeds"] == 5
    assert audit["accidental_collisions"] == 0
    assert audit["no_collisions"] is True
    assert audit["findings"] == []


def test_a_cross_domain_collision_is_detected_and_named():
    audit = combined_collision_audit(
        {
            "alpha": np.array([1, 2, 3], dtype=np.uint64),
            "beta": np.array([3, 9], dtype=np.uint64),
        }
    )
    assert audit["accidental_collisions"] == 1
    assert audit["no_collisions"] is False
    assert audit["findings"][0]["seed"] == 3
    assert audit["findings"][0]["domains"] == ["alpha", "beta"]


def test_an_intra_domain_duplicate_is_detected():
    audit = combined_collision_audit(
        {"alpha": np.array([7, 7, 8], dtype=np.uint64)}
    )
    assert audit["per_domain"]["alpha"]["internal_duplicates"] == 1
    assert audit["accidental_collisions"] == 1
    assert audit["no_collisions"] is False


def test_the_audit_refuses_an_empty_universe():
    with pytest.raises(Phase11StreamAuditError):
        combined_collision_audit({})


def test_the_audit_counts_are_self_consistent():
    audit = combined_collision_audit(
        {
            "alpha": np.arange(100, dtype=np.uint64),
            "beta": np.arange(50, 150, dtype=np.uint64),
        }
    )
    assert audit["total_identities"] == 200
    assert audit["accidental_collisions"] == 50
    assert audit["distinct_seeds"] == 150
    assert (
        audit["distinct_seeds"] + audit["accidental_collisions"]
        == audit["total_identities"]
    )


# ---------------------------------------------------------------------------
# The fast path
# ---------------------------------------------------------------------------


def test_the_fast_path_check_agrees_with_the_public_helpers():
    tokens = tokens_for([IDENTITY_A, IDENTITY_B], range(2), BELIEF_SAMPLER_VERSION)
    report = verify_fast_path(tokens, SLOTS, draws([(2, 3, 1)]))
    assert report["exact"] is True
    assert report["mismatches"] == 0
    assert report["derivations_checked"] > 0


def test_the_materialized_domain_list_excludes_the_uninstantiated_ones():
    assert set(AGENT4_MATERIALIZED_DOMAINS) == {
        DOMAIN_WORLD_SAMPLE,
        DOMAIN_WORLD_ORDER,
        DOMAIN_WORLD_CATEGORICAL,
        DOMAIN_SAFETY_TRIAL,
    }
    assert "repro_schedule" not in AGENT4_MATERIALIZED_DOMAINS
    assert "benchmark" not in AGENT4_MATERIALIZED_DOMAINS


def test_the_harness_instantiates_neither_uninstantiated_domain():
    """The claim is about the live source, so it is checked against it."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "scripts" / "run_phase11_agent04.py"
    ).read_text()
    body = source.split("def agent1_non_safety_universe", 1)
    assert len(body) == 2, "the Agent 1 enumeration helper moved"
    outside = body[0] + body[1].split("def stage_streams", 1)[1]
    assert "repro_schedule_seed(" not in outside
    assert "benchmark_seed(" not in outside
