"""Phase 17 Agent 3 sections 3 and 4: masking, orientation, and reproducibility."""

import numpy as np
import pytest
import torch

from stratego.belief.phase15.orientation import Phase15OrientationError
from stratego.engine.constants import BLUE, FLAG, PIECE_COUNTS, RED, SETUP_SQUARES
from stratego.engine.setup import validate_setup
from stratego.engine.state import create_game
from stratego.setups.identity import orient_setup
from stratego.training.phase17.setup_contract import (
    SETUP_PREFIXES,
    Phase17SetupError,
    Phase17SetupGenerationError,
    Phase17SetupOrientationError,
    setup_root_seed,
    setup_token_seed,
)
from stratego.training.phase17.setup_sampling import (
    batched_remaining,
    generate_setups,
    inventory_mask_from_prefix,
    masked_probabilities,
    remaining_counts,
    suffix_information,
    to_engine_setup,
)

RUN_ID = "RUN-TEST-A"


# -- inventory mask ---------------------------------------------------------


def test_remaining_counts_come_only_from_the_prefix():
    assert list(remaining_counts([])) == [PIECE_COUNTS[t] for t in range(12)]
    assert remaining_counts([FLAG])[FLAG] == 0
    assert inventory_mask_from_prefix([FLAG])[FLAG] is np.False_


def test_an_over_used_type_raises_rather_than_being_repaired():
    """Section 3: fail on an invalid prefix rather than repairing it."""
    with pytest.raises(Phase17SetupError, match="over-uses"):
        remaining_counts([FLAG, FLAG])


def test_an_unknown_type_in_the_prefix_is_refused():
    with pytest.raises(Phase17SetupError, match="unknown piece type"):
        remaining_counts([12])


def test_exhausted_types_receive_probability_exactly_zero():
    """Adversarial logits must not be able to sample an exhausted type."""
    logits = torch.full((1, 12), -50.0)
    logits[0, FLAG] = 1e6  # an adversary screaming for the exhausted type
    mask = torch.ones((1, 12), dtype=torch.bool)
    mask[0, FLAG] = False
    probabilities = masked_probabilities(logits, mask)
    assert float(probabilities[0, FLAG]) == 0.0
    assert pytest.approx(float(probabilities.sum()), abs=1e-6) == 1.0


def test_a_prefix_with_no_legal_type_is_fatal():
    logits = torch.zeros((1, 12))
    mask = torch.zeros((1, 12), dtype=torch.bool)
    with pytest.raises(Phase17SetupGenerationError, match="no legal next piece"):
        masked_probabilities(logits, mask)


def test_batched_remaining_reads_only_the_prefix_columns():
    from stratego.engine.constants import SCOUT

    tokens = torch.full((2, SETUP_PREFIXES), SCOUT, dtype=torch.long)
    tokens[:, 3:] = FLAG  # an uninitialised tail that must not leak in
    remaining = batched_remaining(tokens, 3)
    assert int(remaining[0, FLAG]) == PIECE_COUNTS[FLAG]
    assert int(remaining[0, SCOUT]) == PIECE_COUNTS[SCOUT] - 3


def test_batched_remaining_refuses_a_prefix_that_broke_the_inventory():
    tokens = torch.full((1, SETUP_PREFIXES), FLAG, dtype=torch.long)
    with pytest.raises(Phase17SetupGenerationError, match="inventory went negative"):
        batched_remaining(tokens, 2)


# -- orientation ------------------------------------------------------------


def test_red_is_the_identity_and_blue_reverses_the_ranks(red_samples, blue_samples):
    red = red_samples[0]
    blue = blue_samples[0]
    assert red.engine_setup == red.canonical_setup
    assert blue.engine_setup == orient_setup(blue.canonical_setup, BLUE)
    assert blue.engine_setup != blue.canonical_setup


def test_canonical_blue_handed_straight_to_the_engine_is_rejected(blue_samples):
    """The Phase 11B defect, as a live negative canary."""
    from stratego.belief.phase15.orientation import assert_engine_orientation

    canonical = blue_samples[0].canonical_setup
    with pytest.raises(Phase15OrientationError):
        assert_engine_orientation(canonical, canonical, BLUE)


def test_blue_flags_land_on_blue_back_rows_not_front_rows(blue_samples):
    """The measurable symptom the orientation rule exists to prevent.

    Canonical rank 0 is a player's own back rank. Under the correct rule Blue's
    canonical rank 0 lands on engine row 9; the old glue put it on row 6, which
    is the front row facing the lakes.
    """
    squares = SETUP_SQUARES[BLUE]
    for sample in blue_samples:
        canonical_rank = sample.canonical_setup.index(FLAG) // 10
        engine_row = squares[sample.engine_setup.index(FLAG)] // 10
        assert engine_row == 9 - canonical_rank


def test_an_illegal_inventory_never_reaches_the_engine():
    broken = tuple([FLAG] * 40)
    with pytest.raises(Phase17SetupGenerationError, match="inventory"):
        to_engine_setup(broken, RED)


def test_an_unknown_player_is_refused():
    with pytest.raises(Phase17SetupOrientationError, match="unknown player"):
        to_engine_setup(tuple(range(40)), 7)


# -- generation -------------------------------------------------------------


def test_every_generated_setup_is_a_legal_inventory(red_samples, blue_samples):
    for sample in list(red_samples) + list(blue_samples):
        validate_setup(sample.canonical_setup, sample.color)
        validate_setup(sample.engine_setup, sample.color)


def test_every_generated_board_round_trips_through_game_creation(red_samples, blue_samples):
    for red, blue in zip(red_samples, blue_samples):
        state = create_game(red.engine_setup, blue.engine_setup, game_id=f"{red.root_seed}")
        assert state.acting_player in (RED, BLUE)


def test_the_trace_replays_exactly_under_the_same_snapshot_and_seeds(setup_model, model_digest):
    first = generate_setups(
        setup_model, run_id=RUN_ID, game_ids=["repeat"], color=RED,
        model_state_digest=model_digest, snapshot_iteration=0,
    )[0]
    second = generate_setups(
        setup_model, run_id=RUN_ID, game_ids=["repeat"], color=RED,
        model_state_digest=model_digest, snapshot_iteration=0,
    )[0]
    assert first.canonical_setup == second.canonical_setup
    assert first.per_token_seeds == second.per_token_seeds
    assert np.array_equal(first.behavior_probabilities, second.behavior_probabilities)
    assert np.array_equal(first.behavior_log_probabilities, second.behavior_log_probabilities)


def test_a_changed_seed_domain_permits_a_different_legal_sample(setup_model, model_digest):
    first = generate_setups(
        setup_model, run_id=RUN_ID, game_ids=["seed-a"], color=RED,
        model_state_digest=model_digest, snapshot_iteration=0,
    )[0]
    second = generate_setups(
        setup_model, run_id=RUN_ID, game_ids=["seed-b"], color=RED,
        model_state_digest=model_digest, snapshot_iteration=0,
    )[0]
    assert first.canonical_setup != second.canonical_setup
    validate_setup(second.canonical_setup, RED)


def test_red_and_blue_of_one_game_are_independent_draws(setup_model, model_digest):
    red = generate_setups(
        setup_model, run_id=RUN_ID, game_ids=["same-game"], color=RED,
        model_state_digest=model_digest, snapshot_iteration=0,
    )[0]
    blue = generate_setups(
        setup_model, run_id=RUN_ID, game_ids=["same-game"], color=BLUE,
        model_state_digest=model_digest, snapshot_iteration=0,
    )[0]
    assert red.root_seed != blue.root_seed
    assert red.canonical_setup != blue.canonical_setup


def test_generation_is_independent_of_how_the_pool_was_batched(setup_model, model_digest):
    """A pool of 8 and a single chain must produce the same setup for a game id."""
    batched = generate_setups(
        setup_model, run_id=RUN_ID, game_ids=[f"batch-{i}" for i in range(8)], color=BLUE,
        model_state_digest=model_digest, snapshot_iteration=0,
    )
    alone = generate_setups(
        setup_model, run_id=RUN_ID, game_ids=["batch-5"], color=BLUE,
        model_state_digest=model_digest, snapshot_iteration=0,
    )[0]
    assert alone.canonical_setup == batched[5].canonical_setup


def test_behavior_probabilities_are_masked_and_normalised(red_samples):
    for sample in red_samples[:4]:
        probabilities = sample.behavior_probabilities
        masks = sample.inventory_masks
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        assert (probabilities[~masks] == 0.0).all()
        chosen = probabilities[np.arange(SETUP_PREFIXES), sample.tokens.astype(int)]
        assert (chosen > 0.0).all()


def test_recorded_masks_match_what_the_prefix_implies(red_samples):
    for sample in red_samples[:4]:
        for prefix in range(SETUP_PREFIXES):
            derived = inventory_mask_from_prefix(sample.tokens[:prefix])
            assert np.array_equal(derived, sample.inventory_masks[prefix])


def test_suffix_information_is_the_reverse_cumulative_surprisal():
    logs = np.array([-0.5, -1.0, -2.0], dtype=np.float32)
    assert np.allclose(suffix_information(logs), [3.5, 3.0, 2.0])


def test_recorded_suffix_information_matches_the_recorded_log_probabilities(red_samples):
    for sample in red_samples[:4]:
        expected = suffix_information(sample.behavior_log_probabilities)
        assert np.allclose(sample.suffix_information_content, expected, atol=1e-4)


def test_duplicate_game_ids_in_one_call_are_refused(setup_model, model_digest):
    with pytest.raises(Phase17SetupError, match="unique"):
        generate_setups(
            setup_model, run_id=RUN_ID, game_ids=["dup", "dup"], color=RED,
            model_state_digest=model_digest, snapshot_iteration=0,
        )


# -- pools ------------------------------------------------------------------


def test_a_rebound_pool_discards_entries_drawn_under_the_old_snapshot(setup_model, model_digest):
    """Section 4: never silently relabel old pool entries as current."""
    from stratego.training.phase17.setup_sampling import SetupPool

    pool = SetupPool(
        setup_model, run_id=RUN_ID, color=RED, model_state_digest=model_digest,
        snapshot_iteration=0, size=4,
    )
    pool.prefetch([f"pool-{index}" for index in range(4)])
    assert pool.unused_count == 4
    discarded = pool.rebind(setup_model, model_state_digest="new-digest", snapshot_iteration=1)
    assert discarded == 4
    assert pool.unused_count == 0
    taken = pool.take("pool-0")
    assert taken.setup_model_state_digest == "new-digest"
    assert taken.setup_snapshot_iteration == 1


def test_a_pool_refill_is_counted_not_hidden(setup_model, model_digest):
    from stratego.training.phase17.setup_sampling import SetupPool

    pool = SetupPool(
        setup_model, run_id=RUN_ID, color=RED, model_state_digest=model_digest,
        snapshot_iteration=0, size=2,
    )
    pool.prefetch(["a", "b"])
    pool.take("a")
    pool.take("unprefetched")
    telemetry = pool.telemetry()
    assert telemetry["refills"] == 1
    assert telemetry["consumed"] == 2
    assert telemetry["unused"] == 1


def test_seed_domains_are_separated():
    assert setup_root_seed("r", "g", RED) != setup_root_seed("r", "g", BLUE)
    assert setup_root_seed("r", "g", RED) != setup_root_seed("r2", "g", RED)
    root = setup_root_seed("r", "g", RED)
    assert setup_token_seed(root, 0) != setup_token_seed(root, 1)
    with pytest.raises(Phase17SetupError):
        setup_token_seed(root, SETUP_PREFIXES)


def test_the_token_trace_is_exact_across_batch_shapes_but_the_floats_are_not(
    setup_model, model_digest
):
    """The precise reproducibility claim, pinned so nobody over-reads it.

    The tokens, seeds and masks replay exactly at any batch shape, because
    each draw is an inverse-CDF lookup keyed by its own seed rather than a
    step of a shared generator. The recorded float32 probabilities are exact
    only at the same batch shape: a batched GEMM is not shape-invariant. This
    has no training consequence -- the PPO ratio's denominator is always the
    recorded probability and is never recomputed -- but a verifier that
    re-derives a setup at a different batch size must expect it.
    """
    ids = [f"shape-{index}" for index in range(200)]
    batched = generate_setups(
        setup_model, run_id=RUN_ID, game_ids=ids, color=RED,
        model_state_digest=model_digest, snapshot_iteration=0,
    )
    alone = generate_setups(
        setup_model, run_id=RUN_ID, game_ids=["shape-0"], color=RED,
        model_state_digest=model_digest, snapshot_iteration=0,
    )[0]

    assert alone.canonical_setup == batched[0].canonical_setup
    assert alone.per_token_seeds == batched[0].per_token_seeds
    assert np.array_equal(alone.tokens, batched[0].tokens)
    assert np.array_equal(alone.inventory_masks, batched[0].inventory_masks)

    delta = np.abs(alone.behavior_probabilities - batched[0].behavior_probabilities).max()
    assert delta < 1e-6

    repeated = generate_setups(
        setup_model, run_id=RUN_ID, game_ids=ids, color=RED,
        model_state_digest=model_digest, snapshot_iteration=0,
    )
    assert np.array_equal(
        repeated[0].behavior_probabilities, batched[0].behavior_probabilities
    )


# -- the inverse-CDF draw ---------------------------------------------------


def test_inverse_cdf_respects_the_mask_at_the_top_of_the_unit_interval():
    """A draw past the final cumulative value must not fall through to index 11.

    The regression this pins: forcing the last cumulative entry to 1.0 makes
    the LAST class the fallback for `u` near 1 whether or not it is legal. It
    fired on MPS first -- whose softmax rows sum to 1 + 1e-7 rather than
    1 - 1e-7 -- but the defect was never MPS-specific.
    """
    from stratego.training.phase17.setup_sampling import inverse_cdf_choice

    probabilities = np.zeros((1, 12), dtype=np.float64)
    probabilities[0, 0] = 0.5
    probabilities[0, 1] = 0.5
    mask = np.zeros((1, 12), dtype=bool)
    mask[0, 0] = mask[0, 1] = True

    for draw in (0.0, 0.4999, 0.5, 0.9, 1.0 - 1e-12, 0.99999999999999):
        chosen = inverse_cdf_choice(probabilities, mask, np.array([draw]))
        assert bool(mask[0, chosen[0]]), f"u={draw} chose the illegal index {chosen[0]}"


def test_inverse_cdf_survives_a_row_that_does_not_sum_to_one():
    from stratego.training.phase17.setup_sampling import inverse_cdf_choice

    probabilities = np.zeros((2, 12), dtype=np.float64)
    probabilities[:, 2] = 0.3
    probabilities[:, 5] = 0.7 - 1e-7  # sums just under 1
    mask = np.zeros((2, 12), dtype=bool)
    mask[:, 2] = mask[:, 5] = True
    chosen = inverse_cdf_choice(probabilities, mask, np.array([1.0 - 1e-15, 0.0]))
    assert chosen.tolist() == [5, 2]


def test_inverse_cdf_partitions_the_unit_interval_by_probability():
    from stratego.training.phase17.setup_sampling import inverse_cdf_choice

    probabilities = np.zeros((1, 12), dtype=np.float64)
    probabilities[0, 3] = 0.25
    probabilities[0, 7] = 0.75
    mask = np.zeros((1, 12), dtype=bool)
    mask[0, 3] = mask[0, 7] = True
    draws = np.linspace(0.0, 1.0, 10001, endpoint=False)
    picks = np.array(
        [inverse_cdf_choice(probabilities, mask, np.array([u]))[0] for u in draws]
    )
    assert (picks == 3).mean() == pytest.approx(0.25, abs=1e-3)
    assert (picks == 7).mean() == pytest.approx(0.75, abs=1e-3)


def test_inverse_cdf_refuses_a_row_with_no_legal_index():
    from stratego.training.phase17.setup_sampling import inverse_cdf_choice

    with pytest.raises(Phase17SetupGenerationError, match="no legal index"):
        inverse_cdf_choice(np.ones((1, 12)) / 12, np.zeros((1, 12), dtype=bool), np.array([0.5]))


def test_inverse_cdf_refuses_a_row_whose_legal_mass_is_zero():
    from stratego.training.phase17.setup_sampling import inverse_cdf_choice

    probabilities = np.zeros((1, 12), dtype=np.float64)
    probabilities[0, 0] = 1.0
    mask = np.zeros((1, 12), dtype=bool)
    mask[0, 5] = True  # the only legal index carries no mass
    with pytest.raises(Phase17SetupGenerationError, match="no probability mass"):
        inverse_cdf_choice(probabilities, mask, np.array([0.5]))


def test_generation_on_mps_produces_only_legal_setups(model_digest):
    """The device the defect surfaced on, exercised end to end."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this host")
    from stratego.training.phase17.setup_model import build_setup_model
    from stratego.engine.constants import BLUE as BLUE_PLAYER

    model = build_setup_model(device="mps", seed=3)
    from stratego.training.phase9_behavior import state_dict_digest

    samples = generate_setups(
        model, run_id=RUN_ID, game_ids=[f"mps-{index}" for index in range(64)],
        color=BLUE_PLAYER, model_state_digest=state_dict_digest(model), snapshot_iteration=0,
    )
    for sample in samples:
        validate_setup(sample.canonical_setup, BLUE_PLAYER)
        validate_setup(sample.engine_setup, BLUE_PLAYER)
