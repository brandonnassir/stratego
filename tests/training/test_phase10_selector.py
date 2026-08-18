"""Unit coverage: Agent 4's selector, its input boundary and its Phase 7 adapter.

Everything here runs against the real frozen library and Agent 3's accepted
utility artifact, because the properties worth checking are properties of the
frozen objects rather than of a synthetic stand-in. What the suite pins:

- exactly six candidates, six distinct selector identities, no seventh;
- the input boundary is a wall: a request naming an opponent family, an
  opponent base id, a setup fingerprint, a policy id or a game outcome
  raises, and hidden opponent truth cannot change a draw;
- the distribution is exact — frozen base order, finite softmax, the frozen
  0.35 / 0.65 mixture reproduced bit-for-bit, and a canonical digest;
- the `searchsorted` inverse-CDF walk equals the accepted linear walk;
- a neutral-branch draw is bit-identical to
  `sample_setup(split, seed, 'neutral_v1')`, and a learned-branch draw
  differs from it in the base alone;
- `neutral_v1` itself is unchanged;
- the draw is a pure function of its logical identity, across independently
  constructed sources;
- `verify_draw` fires on a tampered draw rather than waving it through.
"""

import math
import random

import numpy as np
import pytest

from stratego.setups.families import FAMILY_IDS
from stratego.setups.identity import derive_stream_seed
from stratego.setups.sampler import (
    NEUTRAL_PROFILE,
    load_library_index,
    sample_setup,
)
from stratego.training import phase10_selector as selector
from stratego.training.phase10_contract import (
    CANDIDATE_MATRIX,
    DIVERSITY_THRESHOLDS,
    LEARNED_MIXTURE_WEIGHT,
    NEUTRAL_MIXTURE_WEIGHT,
)
from stratego.training.phase10_seed import (
    selector_audit_draw_id,
    selector_audit_seed,
    selector_branch_uniform,
)


@pytest.fixture(scope="module")
def index():
    return load_library_index()


@pytest.fixture(scope="module")
def scorer():
    return selector.load_scorer()


@pytest.fixture(scope="module")
def source_t(scorer, index):
    """P10-D: model_T at the lowest temperature, the sharpest candidate."""
    return selector.LearnedSetupSource(selector.candidate("P10-D"), scorer, index)


@pytest.fixture(scope="module")
def source_f(scorer, index):
    """P10-A: model_F at the lowest temperature."""
    return selector.LearnedSetupSource(selector.candidate("P10-A"), scorer, index)


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def test_exactly_six_candidates_matching_the_frozen_matrix():
    assert len(selector.CANDIDATES) == 6
    observed = [
        (entry.candidate_id, entry.utility_model, entry.temperature)
        for entry in selector.CANDIDATES
    ]
    expected = [
        (entry["candidate_id"], entry["utility_model"], float(entry["temperature"]))
        for entry in CANDIDATE_MATRIX
    ]
    assert observed == expected


def test_selector_identities_are_distinct_and_self_describing():
    identities = [entry.selector_identity for entry in selector.CANDIDATES]
    assert len(set(identities)) == 6
    for entry, identity in zip(selector.CANDIDATES, identities):
        assert entry.candidate_id in identity
        assert entry.utility_model in identity
        assert f"{entry.temperature:.2f}" in identity


def test_unknown_candidate_is_refused():
    with pytest.raises(selector.Phase10SelectorError):
        selector.candidate("P10-G")


# ---------------------------------------------------------------------------
# The permitted-input boundary
# ---------------------------------------------------------------------------


def _legal_payload():
    return {"split": "validation", "color": "red", "selector_seed": 12345}


@pytest.mark.parametrize(
    "injected",
    [
        {"opponent_family": "F03"},
        {"opponent_base_setup_id": "setup_library_v1:F03:410"},
        {"opponent_setup_fingerprint": "deadbeef"},
        {"opponent_policy_id": "strategic_rule_based"},
        {"game_outcome": "red_win"},
        {"result": 1.0},
        {"match_seed": 99},
        {"hidden_opponent_truth": [1, 2, 3]},
        {"storage_path": "/Volumes/Brandon_Washington/stratego_phase10"},
        {"red_score": 1.0},
    ],
)
def test_injected_opponent_or_outcome_fields_are_rejected(injected):
    payload = {**_legal_payload(), **injected}
    with pytest.raises(selector.Phase10SelectorError) as error:
        selector.SelectorRequest.from_payload(payload)
    assert sorted(injected)[0] in str(error.value)


def test_missing_and_malformed_request_fields_are_rejected():
    with pytest.raises(selector.Phase10SelectorError):
        selector.SelectorRequest.from_payload({"split": "validation", "color": "red"})
    with pytest.raises(selector.Phase10SelectorError):
        selector.SelectorRequest(split="holdout", color="red", selector_seed=1)
    with pytest.raises(selector.Phase10SelectorError):
        selector.SelectorRequest(split="validation", color="green", selector_seed=1)
    with pytest.raises(selector.Phase10SelectorError):
        selector.SelectorRequest(split="validation", color="red", selector_seed=-1)


def test_legal_payload_builds_a_request():
    request = selector.SelectorRequest.from_payload(_legal_payload())
    assert request.to_dict() == _legal_payload()


def test_a_draw_takes_a_request_not_a_raw_mapping(source_t):
    with pytest.raises(selector.Phase10SelectorError):
        source_t.draw(_legal_payload())


def test_hidden_opponent_truth_cannot_change_a_draw(source_t, index):
    """Changing everything about the opponent leaves the draw untouched.

    There is no channel for it to arrive through, which is the point: the
    control varies a whole opponent context around the call and requires the
    produced fingerprint to be identical, so the claim is demonstrated rather
    than asserted.
    """
    request = selector.SelectorRequest(split="validation", color="red", selector_seed=777)
    reference = source_t.draw(request)
    for opponent_seed in range(5):
        opponent = sample_setup("validation", 5_000 + opponent_seed, "neutral_v1", index)
        assert opponent.provenance["final_setup_fingerprint"]  # the truth exists
        again = source_t.draw(request)
        assert again.final_setup_fingerprint == reference.final_setup_fingerprint
        assert again.base_setup_id == reference.base_setup_id
        assert again.branch == reference.branch


# ---------------------------------------------------------------------------
# The exact distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("split,count", [("train", 6400), ("validation", 800), ("test", 800)])
def test_split_base_order_is_ascending_family_then_base_index(split, count, index):
    entries = selector.split_base_entries(split, index)
    assert len(entries) == count
    keys = [(FAMILY_IDS.index(entry.family_id), entry.base_index) for entry in entries]
    assert keys == sorted(keys)
    assert all(entry.split == split for entry in entries)


def test_distribution_is_finite_normalized_and_exactly_mixed(source_t):
    distribution = source_t.distribution("red", "validation")
    facts = distribution.finiteness()
    assert facts["all_finite"] and facts["all_non_negative"]
    assert facts["cumulative_monotone"]
    for deviation in facts["sum_deviations"].values():
        assert deviation <= 1e-12
    assert distribution.mixture_is_exact()
    recomputed = (
        NEUTRAL_MIXTURE_WEIGHT * distribution.p_neutral
        + LEARNED_MIXTURE_WEIGHT * distribution.p_learned
    )
    assert np.array_equal(recomputed, distribution.p_mixed)


def test_neutral_component_is_uniform_over_the_split(source_t):
    distribution = source_t.distribution("blue", "test")
    assert np.all(distribution.p_neutral == 1.0 / distribution.base_count)


def test_model_f_is_uniform_within_every_family(source_f):
    """Model F scores a family, so its learned mass must not vary inside one."""
    distribution = source_f.distribution("red", "validation")
    grid = distribution.p_learned.reshape(len(FAMILY_IDS), distribution.bases_per_family)
    for row in grid:
        assert np.allclose(row, row[0], rtol=0.0, atol=0.0)


def test_model_t_varies_within_a_family(source_t):
    distribution = source_t.distribution("red", "validation")
    grid = distribution.p_learned.reshape(len(FAMILY_IDS), distribution.bases_per_family)
    assert any(len(set(row.tolist())) > 1 for row in grid)


def test_distribution_digest_is_stable_and_discriminating(scorer, index):
    entry = selector.candidate("P10-E")
    first = selector.build_distribution(entry, "red", "validation", scorer, index)
    second = selector.build_distribution(entry, "red", "validation", scorer, index)
    assert first.probability_vector_digest() == second.probability_vector_digest()

    others = [
        selector.build_distribution(entry, "blue", "validation", scorer, index),
        selector.build_distribution(entry, "red", "test", scorer, index),
        selector.build_distribution(selector.candidate("P10-F"), "red", "validation", scorer, index),
    ]
    digests = {first.probability_vector_digest()} | {
        other.probability_vector_digest() for other in others
    }
    assert len(digests) == 4


def test_temperature_must_be_positive(scorer, index):
    broken = selector.SelectorCandidate("P10-X", "model_F", 0.0)
    with pytest.raises(selector.Phase10SelectorError):
        selector.build_distribution(broken, "red", "validation", scorer, index)


def test_the_learned_ladder_is_softmax_mass_not_the_mixture(source_t):
    """The regression test for the Agent 4 review-reconciliation defect.

    The learned branch is reached only after the branch coin has already
    applied the frozen 0.35 neutral weight, so its inverse-CDF ladder must be
    `cumsum(p_learned)`. Walking `cumsum(p_mixed)` applies that weight a
    second time and silently realizes `0.5775*neutral + 0.4225*learned`.

    The check is exact, and it is *discriminating*: it also requires the two
    ladders to disagree somewhere, so the assertion cannot be satisfied by a
    degenerate distribution where mixed and learned happen to coincide.
    """
    for split in ("validation", "train"):
        distribution = source_t.distribution("red", split)
        assert np.array_equal(distribution.cumulative_learned, np.cumsum(distribution.p_learned))
        mixed_ladder = np.cumsum(distribution.p_mixed)
        assert not np.array_equal(distribution.cumulative_learned, mixed_ladder)

        disagreements = sum(
            distribution.base_index_for_uniform(u)
            != min(
                int(np.searchsorted(mixed_ladder, u, side="right")),
                distribution.base_count - 1,
            )
            for u in np.linspace(0.001, 0.999, 400)
        )
        assert disagreements > 0


def test_the_realized_mixture_is_the_frozen_035_065(source_t):
    """Draws must actually realize the published distribution.

    The exact vectors being right is not the same claim as the sampler
    reproducing them, and the defect this pins broke only the second. Over
    24,000 frozen draw ids the empirical family distribution must sit far
    closer to `p_mixed` than to the `0.5775/0.4225` blend the double-applied
    neutral weight produces.
    """
    draws = 24_000
    distribution = source_t.distribution("blue", "validation")
    exact = distribution.family_probabilities()
    wrong = (
        0.5775 * distribution.p_neutral + 0.4225 * distribution.p_learned
    ).reshape(len(FAMILY_IDS), distribution.bases_per_family).sum(axis=1)

    counts = np.zeros(len(FAMILY_IDS), dtype=np.int64)
    for ordinal in range(draws):
        _, draw = source_t.audit_draw(ordinal, "blue", "validation")
        counts[FAMILY_IDS.index(draw.family_id)] += 1
    empirical = counts / draws

    distance_to_exact = 0.5 * np.abs(empirical - exact).sum()
    distance_to_wrong = 0.5 * np.abs(empirical - wrong).sum()
    noise = 0.5 * np.sqrt(2.0 / np.pi) * np.sqrt(exact * (1.0 - exact) / draws).sum()

    assert distance_to_exact < 4.0 * noise
    assert distance_to_exact < distance_to_wrong


def test_inverse_cdf_walk_equals_the_accepted_linear_walk(source_t):
    """`searchsorted` must agree with a plain accumulate-until-exceeded walk."""
    distribution = source_t.distribution("red", "validation")
    cumulative = distribution.cumulative_learned

    def linear(uniform: float) -> int:
        for position, mass in enumerate(cumulative):
            if mass > uniform:
                return position
        return distribution.base_count - 1

    rng = random.Random(20260818)
    probes = [0.0, 1e-18, 0.35, 0.5, 0.999999, math.nextafter(1.0, 0.0)]
    probes += [rng.random() for _ in range(2000)]
    probes += [float(value) for value in cumulative[:: max(1, distribution.base_count // 50)]]
    for uniform in probes:
        assert distribution.base_index_for_uniform(uniform) == linear(uniform)


def test_cumulative_mass_is_a_plain_sequential_running_sum(source_t, source_f):
    """The frozen wording is 'float64 cumulative softmax mass'.

    `np.cumsum` is used for the speed the 3.6M-draw audit needs, so the
    property that makes that substitution legitimate is pinned here: it is
    bit-for-bit the running sum a hand-written accumulation produces, on the
    largest split as well as the smallest.
    """
    for source in (source_t, source_f):
        for split in ("validation", "train"):
            distribution = source.distribution("red", split)
            accumulated = 0.0
            manual = []
            for value in distribution.p_learned.tolist():
                accumulated += value
                manual.append(accumulated)
            assert np.array_equal(
                np.array(manual, dtype=np.float64), distribution.cumulative_learned
            )


def test_the_tail_guard_keeps_the_index_in_range(source_t):
    distribution = source_t.distribution("red", "validation")
    assert distribution.base_index_for_uniform(1.0) == distribution.base_count - 1
    assert distribution.base_index_for_uniform(2.0) == distribution.base_count - 1


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------


def test_uniform_distribution_hits_the_analytic_diversity_values():
    count = 800
    metrics = selector.diversity_metrics(np.full(count, 1.0 / count), "validation")
    assert metrics["normalized_family_entropy"] == pytest.approx(1.0)
    assert metrics["effective_families"] == pytest.approx(16.0)
    assert metrics["min_within_family_normalized_base_entropy"] == pytest.approx(1.0)
    assert metrics["max_conditional_base_probability"] == pytest.approx(1.0 / 50)
    assert selector.evaluate_diversity(metrics)["all_pass"]


def test_a_degenerate_distribution_fails_the_thresholds():
    count = 800
    vector = np.full(count, 1e-9)
    vector[0] = 1.0 - 1e-9 * (count - 1)
    vector = vector / vector.sum()
    verdict = selector.evaluate_diversity(selector.diversity_metrics(vector, "validation"))
    assert not verdict["all_pass"]
    assert {"normalized_family_entropy", "effective_families", "family_probability_max"} <= set(
        verdict["failed"]
    )


def test_diversity_thresholds_are_the_frozen_ones():
    metrics = selector.diversity_metrics(np.full(800, 1.0 / 800), "validation")
    assert selector.evaluate_diversity(metrics)["thresholds"] == dict(DIVERSITY_THRESHOLDS)


def test_a_ragged_vector_is_refused():
    with pytest.raises(selector.Phase10SelectorError):
        selector.diversity_metrics(np.full(801, 1.0 / 801), "validation")


# ---------------------------------------------------------------------------
# Phase 7 preservation
# ---------------------------------------------------------------------------


def test_neutral_v1_is_unchanged():
    assert NEUTRAL_PROFILE.name == "neutral_v1"
    assert NEUTRAL_PROFILE.reflection_probability == 0.5
    assert NEUTRAL_PROFILE.perturbation_probability == 0.5
    assert NEUTRAL_PROFILE.intensity_weights == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert NEUTRAL_PROFILE.swap_counts == (1, 2, 3, 4, 5, 6)


def test_the_neutral_baseline_api_is_the_accepted_sampler(index):
    for seed in (1, 2, 3):
        theirs = sample_setup("validation", seed, "neutral_v1", index)
        ours = selector.neutral_baseline_draw("validation", seed, index)
        assert ours.provenance == theirs.provenance


def test_post_selection_streams_match_the_accepted_sampler(index):
    """The adapter re-derives the accepted streams, not new ones."""
    for seed in range(40):
        decisions = selector.post_selection_decisions("validation", seed)
        accepted = sample_setup("validation", seed, "neutral_v1", index).provenance
        assert decisions.reflection_applied == accepted["reflection_applied"]
        assert decisions.perturbation_requested == accepted["perturbation_requested"]
        assert decisions.swap_count == accepted["perturbation_swap_count"]
        assert selector.neutral_branch_base_id("validation", seed, index) == accepted[
            "base_setup_id"
        ]


def test_perturbation_seed_matches_the_accepted_composite(index):
    for seed in range(60):
        accepted = sample_setup("validation", seed, "neutral_v1", index).provenance
        if not accepted["perturbation_requested"]:
            continue
        assert selector.perturbation_seed_for(
            "validation",
            seed,
            accepted["base_setup_id"],
            accepted["perturbation_swap_count"],
        ) == accepted["perturbation_seed"]


def test_the_adapter_uses_the_accepted_derivation(index):
    """No private Phase 7 helper is reimplemented with a different key."""
    expected = derive_stream_seed("setup_sampler_v1:orientation", "neutral_v1", "validation", 11)
    assert selector._phase7_stream("orientation", "validation", 11).getstate() == random.Random(
        expected
    ).getstate()


def test_neutral_branch_draws_are_bit_identical_to_the_baseline(source_t, index):
    checked = 0
    for ordinal in range(150):
        _, draw = source_t.audit_draw(ordinal, "red", "validation")
        if draw.branch != selector.BRANCH_NEUTRAL:
            continue
        checked += 1
        assert selector.neutral_branch_matches_accepted_sampler(draw, index) == []
    assert checked >= 20


def test_learned_branch_differs_from_the_baseline_in_the_base_alone(source_t, index):
    checked = 0
    differing_bases = 0
    for ordinal in range(150):
        _, draw = source_t.audit_draw(ordinal, "red", "validation")
        if draw.branch != selector.BRANCH_LEARNED:
            continue
        checked += 1
        assert selector.learned_branch_shares_phase7_decisions(draw, index) == []
        baseline = sample_setup("validation", draw.selector_seed, "neutral_v1", index)
        if baseline.provenance["base_setup_id"] != draw.base_setup_id:
            differing_bases += 1
    assert checked >= 40
    assert differing_bases >= 1


def test_branch_helpers_refuse_the_wrong_branch(source_t, index):
    for ordinal in range(40):
        _, draw = source_t.audit_draw(ordinal, "red", "validation")
        if draw.branch == selector.BRANCH_NEUTRAL:
            with pytest.raises(selector.Phase10SelectorError):
                selector.learned_branch_shares_phase7_decisions(draw, index)
        else:
            with pytest.raises(selector.Phase10SelectorError):
                selector.neutral_branch_matches_accepted_sampler(draw, index)


# ---------------------------------------------------------------------------
# Determinism and identity
# ---------------------------------------------------------------------------


def test_a_draw_is_a_pure_function_of_its_logical_identity(scorer, index):
    first = selector.LearnedSetupSource(selector.candidate("P10-B"), scorer, index)
    second = selector.LearnedSetupSource(selector.candidate("P10-B"), scorer, load_library_index())
    for ordinal in (0, 1, 17, 999):
        for color in ("red", "blue"):
            left_id, left = first.audit_draw(ordinal, color, "test")
            right_id, right = second.audit_draw(ordinal, color, "test")
            assert left_id == right_id
            assert left.to_dict() == right.to_dict()


def test_call_order_does_not_move_a_draw(source_f):
    forward = [source_f.audit_draw(n, "red", "validation")[1] for n in range(30)]
    backward = [source_f.audit_draw(n, "red", "validation")[1] for n in reversed(range(30))]
    assert [draw.to_dict() for draw in forward] == [
        draw.to_dict() for draw in reversed(backward)
    ]


def test_audit_draw_identity_round_trips_and_matches_the_seed(source_t):
    draw_id, draw = source_t.audit_draw(42, "blue", "train")
    assert draw_id == selector_audit_draw_id("P10-D", "train", "blue", 42)
    assert draw.selector_seed == selector_audit_seed("P10-D", "train", "blue", 42)


def test_candidates_and_colours_produce_different_streams(scorer, index):
    identities = set()
    for entry in selector.CANDIDATES:
        for color in ("red", "blue"):
            identities.add(
                selector_branch_uniform(entry.selector_identity, "validation", color, 4242)
            )
    assert len(identities) == 12


def test_the_branch_coin_lands_near_the_frozen_weight(source_f):
    draws = [source_f.audit_draw(n, "red", "test")[1] for n in range(2000)]
    neutral = sum(1 for draw in draws if draw.branch == selector.BRANCH_NEUTRAL)
    assert abs(neutral / len(draws) - NEUTRAL_MIXTURE_WEIGHT) < 0.05


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_verify_draw_passes_a_clean_draw(source_t):
    for ordinal in range(25):
        draw_id, draw = source_t.audit_draw(ordinal, "red", "validation")
        result = selector.verify_draw(source_t, draw, draw_id)
        assert result["ok"], result["findings"]
        assert all(count == 0 for count in result["counters"].values())


def test_verify_draw_fires_on_a_tampered_split(source_t):
    import dataclasses

    draw_id, draw = source_t.audit_draw(3, "red", "validation")
    tampered = dataclasses.replace(draw, split="train")
    result = selector.verify_draw(source_t, tampered, draw_id)
    assert not result["ok"]
    assert result["counters"]["split_violations"] >= 1


def test_verify_draw_fires_on_a_tampered_branch(source_t):
    import dataclasses

    draw_id, draw = source_t.audit_draw(4, "red", "validation")
    other = (
        selector.BRANCH_LEARNED
        if draw.branch == selector.BRANCH_NEUTRAL
        else selector.BRANCH_NEUTRAL
    )
    result = selector.verify_draw(
        source_t, dataclasses.replace(draw, branch=other), draw_id, cross_check_accepted_sampler=False
    )
    assert not result["ok"]


def test_verify_draw_fires_on_a_tampered_uniform(source_t):
    import dataclasses

    draw_id, draw = source_t.audit_draw(5, "red", "validation")
    result = selector.verify_draw(
        source_t,
        dataclasses.replace(draw, branch_uniform=float("nan")),
        draw_id,
        cross_check_accepted_sampler=False,
    )
    assert not result["ok"]
    assert result["counters"]["non_finite_selector_values"] >= 1


def test_verify_draw_fires_on_a_foreign_draw_id(source_t):
    _, draw = source_t.audit_draw(6, "red", "validation")
    foreign = selector_audit_draw_id("P10-A", "validation", "red", 6)
    result = selector.verify_draw(source_t, draw, foreign, cross_check_accepted_sampler=False)
    assert not result["ok"]


def test_construction_failures_are_classified_conservatively():
    assert selector.classify_construction_failure("inventory/legality: bad") == "inventory_errors"
    assert selector.classify_construction_failure("stranded: no move") == "stranded_sampled_setups"
    assert selector.classify_construction_failure("split migration: x") == "split_violations"
    assert selector.classify_construction_failure("something else") == "illegal_setups"


def test_audit_counters_cover_the_frozen_zero_tolerance_list():
    from stratego.training.phase10_contract import SELECTOR_AUDIT_ZERO_TOLERANCE

    normalized = tuple(
        entry.replace(" ", "_").replace("-", "_") for entry in SELECTOR_AUDIT_ZERO_TOLERANCE
    )
    expected = {
        "illegal_setups",
        "inventory_errors",
        "stranded_sampled_setups",
        "split_violations",
        "provenance_mismatches",
        "determinism_mismatches",
        "non_finite_selector_values",
    }
    assert set(normalized) == expected == set(selector.AUDIT_COUNTERS)


# ---------------------------------------------------------------------------
# The contract document
# ---------------------------------------------------------------------------


def test_contract_document_is_stable_and_names_the_six_candidates():
    document = selector.selector_contract_document()
    assert document["candidate_count"] == 6
    assert [entry["candidate_id"] for entry in document["candidates"]] == [
        entry["candidate_id"] for entry in CANDIDATE_MATRIX
    ]
    assert document["mixture"]["neutral_weight"] == 0.35
    assert document["mixture"]["learned_weight"] == 0.65
    assert selector.selector_contract_digest() == selector.selector_contract_digest()


def test_contract_digest_moves_when_digests_are_bound():
    bare = selector.selector_contract_digest()
    bound = selector.selector_contract_digest({"P10-A": {"red": {"validation": "abc"}}})
    assert bare != bound
