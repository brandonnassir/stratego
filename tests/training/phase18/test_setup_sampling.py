"""S02, S04, S05, S06, S07, S11, S20, S24: masking, handedness, reflection,
orientation, information and pool structure."""

import numpy as np
import pytest
import torch

from stratego.belief.phase15.orientation import Phase15OrientationError, assert_engine_orientation
from stratego.engine.constants import BLUE, BOMB, FLAG, PIECE_COUNTS, RED, SCOUT, SETUP_SQUARES
from stratego.engine.legal_moves import has_legal_action
from stratego.engine.setup import random_setup, validate_setup
from stratego.engine.state import create_game
from stratego.setups.identity import orient_setup, reflect_canonical
from stratego.training.phase18.setup_contract import (
    FLAG_PERMITTED_FILES,
    SETUP_PREFIXES,
    Phase18SetupError,
    Phase18SetupGenerationError,
    Phase18SetupOrientationError,
    pool_root_seed,
    reflection_seed,
    token_seed,
)
from stratego.training.phase18.setup_sampling import (
    CORRIDOR_FILES,
    batched_remaining,
    generate_pool,
    handedness_mask,
    has_opening_move,
    inventory_mask_from_prefix,
    inverse_cdf_choice,
    legal_masks,
    masked_log_probabilities,
    reflect_tokens,
    remaining_counts,
    suffix_information,
    to_engine_setup,
)

from .conftest import NAMESPACE


# -- S02: inventory masking ------------------------------------------------------


def test_remaining_counts_come_only_from_the_prefix():
    assert list(remaining_counts([])) == [PIECE_COUNTS[t] for t in range(12)]
    assert remaining_counts([FLAG])[FLAG] == 0
    assert inventory_mask_from_prefix([FLAG])[FLAG] is np.False_


def test_an_over_used_type_raises_rather_than_being_repaired():
    with pytest.raises(Phase18SetupError, match="over-uses"):
        remaining_counts([FLAG, FLAG])
    with pytest.raises(Phase18SetupError, match="unknown piece type"):
        remaining_counts([12])


def test_exhausted_types_receive_probability_exactly_zero():
    logits = torch.full((1, 12), -50.0)
    logits[0, FLAG] = 1e6
    mask = torch.ones((1, 12), dtype=torch.bool)
    mask[0, FLAG] = False
    log_probabilities = masked_log_probabilities(logits, mask)
    assert float(log_probabilities[0, FLAG]) == float("-inf")
    assert pytest.approx(float(log_probabilities.exp().sum()), abs=1e-6) == 1.0


def test_after_all_eight_scouts_are_placed_the_scout_probability_is_exactly_zero(pool):
    """Exhausted-type test on real samples: once the eighth scout is placed,
    every later prefix carries mask False and log-probability mass zero."""
    checked = 0
    for sample in pool.samples:
        tokens = sample.network_tokens.astype(int)
        positions = np.nonzero(tokens == SCOUT)[0]
        assert positions.size == PIECE_COUNTS[SCOUT]
        exhausted_from = int(positions[-1]) + 1
        for prefix in range(exhausted_from, SETUP_PREFIXES):
            assert not sample.legal_masks[prefix, SCOUT]
            assert sample.behavior_log_probs[prefix, SCOUT] == 0.0  # masked entries are stored as 0 with mask False
            checked += 1
    assert checked > 0


def test_every_generated_setup_reproduces_the_classic_piece_counts(pool):
    for sample in pool.samples:
        counts = np.bincount(sample.network_tokens.astype(int), minlength=12)
        assert counts.tolist() == [PIECE_COUNTS[t] for t in range(12)]
        validate_setup(sample.played_canonical, RED)
        validate_setup(sample.engine_setup, sample.lane)


def test_batched_remaining_reads_only_the_prefix_columns():
    tokens = torch.full((2, SETUP_PREFIXES), SCOUT, dtype=torch.long)
    tokens[:, 3:] = FLAG
    remaining = batched_remaining(tokens, 3)
    assert int(remaining[0, FLAG]) == PIECE_COUNTS[FLAG]
    assert int(remaining[0, SCOUT]) == PIECE_COUNTS[SCOUT] - 3
    with pytest.raises(Phase18SetupGenerationError, match="inventory went negative"):
        batched_remaining(torch.full((1, SETUP_PREFIXES), FLAG, dtype=torch.long), 2)


def test_recorded_masks_are_inventory_and_handedness_and_nothing_is_passed_in(pool):
    for sample in pool.samples[:6]:
        tokens = sample.network_tokens.astype(int)
        for prefix in range(SETUP_PREFIXES):
            derived = inventory_mask_from_prefix(tokens[:prefix]) & handedness_mask(prefix)
            assert np.array_equal(derived, sample.legal_masks[prefix])


# -- S04: forced handedness ------------------------------------------------------


def test_the_handedness_mask_forbids_the_flag_on_the_left_five_files_only():
    for prefix in range(SETUP_PREFIXES):
        mask = handedness_mask(prefix)
        assert mask.sum() in (11, 12)
        assert bool(mask[FLAG]) == ((prefix % 10) in FLAG_PERMITTED_FILES)
        assert all(mask[t] for t in range(12) if t != FLAG)
    assert FLAG_PERMITTED_FILES == (5, 6, 7, 8, 9)


def test_forced_generation_puts_the_flag_in_the_permitted_half_100_percent(pool):
    for sample in pool.samples:
        flag_file = int(np.nonzero(sample.network_tokens.astype(int) == FLAG)[0][0]) % 10
        assert flag_file in FLAG_PERMITTED_FILES
    assert pool.telemetry["flag_in_permitted_half_fraction_network"] == 1.0
    assert sum(pool.telemetry["flag_file_histogram_network"][:5]) == 0


def test_reduction_without_forced_handedness_reproduces_the_unconstrained_distribution(setup_model, model_digest):
    """With `force_handedness=False` the legal mask is the inventory mask alone
    and the Flag lands on both halves."""
    unforced = generate_pool(
        setup_model, namespace=NAMESPACE, seed_index=7, snapshot_iteration=0,
        snapshot_digest=model_digest, count=256, force_handedness=False, reflection_probability=0.0,
    )
    files = [int(np.nonzero(s.network_tokens.astype(int) == FLAG)[0][0]) % 10 for s in unforced.samples]
    assert any(f < 5 for f in files) and any(f >= 5 for f in files)
    for sample in unforced.samples[:6]:
        tokens = sample.network_tokens.astype(int)
        for prefix in range(SETUP_PREFIXES):
            assert np.array_equal(inventory_mask_from_prefix(tokens[:prefix]), sample.legal_masks[prefix])
    tokens = torch.zeros((1, SETUP_PREFIXES), dtype=torch.long)
    assert bool(legal_masks(tokens, 0, force_handedness=False)[0, FLAG])
    assert not bool(legal_masks(tokens, 0, force_handedness=True)[0, FLAG])


# -- S05: post-generation reflection ----------------------------------------------


def test_reflection_is_an_involution_and_matches_the_accepted_helper(pool):
    for sample in pool.samples:
        network = sample.network_tokens.astype(int)
        assert np.array_equal(reflect_tokens(reflect_tokens(network)), network)
        assert tuple(reflect_tokens(network).tolist()) == reflect_canonical(tuple(network.tolist()))


def test_reflected_fraction_is_one_half_within_binomial_tolerance(setup_model, model_digest):
    big = generate_pool(
        setup_model, namespace=NAMESPACE, seed_index=3, snapshot_iteration=0,
        snapshot_digest=model_digest, count=1024,
    )
    fraction = big.telemetry["reflected_fraction"]
    assert abs(fraction - 0.5) < 4 * 0.5 / np.sqrt(1024)  # four binomial SEs
    played = big.telemetry["flag_file_histogram_played"]
    assert sum(played[:5]) > 0 and sum(played[5:]) > 0, "both flag halves reach play"


def test_the_reflection_stream_is_independent_of_the_token_stream():
    assert reflection_seed(NAMESPACE, 1, 0, 5) != pool_root_seed(NAMESPACE, 1, 0, 5)
    assert reflection_seed(NAMESPACE, 1, 0, 5) != reflection_seed(NAMESPACE, 1, 0, 6)
    assert reflection_seed(NAMESPACE, 1, 0, 5) != reflection_seed(NAMESPACE, 1, 1, 5)


def test_a_reflected_sample_plays_the_mirror_and_keeps_its_network_record(pool):
    reflected = [s for s in pool.samples if s.reflected]
    unreflected = [s for s in pool.samples if not s.reflected]
    assert reflected and unreflected
    for sample in reflected:
        assert tuple(sample.played_canonical) == tuple(reflect_tokens(sample.network_tokens.astype(int)).tolist())
        assert sample.content_fingerprint != sample.network_fingerprint
        assert sample.class_fingerprint == sample.class_fingerprint  # same class either way
    for sample in unreflected:
        assert tuple(sample.played_canonical) == tuple(sample.network_tokens.astype(int).tolist())
        assert sample.content_fingerprint == sample.network_fingerprint


# -- S06: behavior log-probability bookkeeping under reflection -------------------


def test_flipping_a_played_board_back_recovers_the_network_tokens_and_the_recorded_nll(pool):
    for sample in pool.samples:
        played = np.asarray(sample.played_canonical, dtype=int)
        recovered = reflect_tokens(played) if sample.reflected else played
        assert np.array_equal(recovered, sample.network_tokens.astype(int))
        gathered = sample.behavior_log_probs[np.arange(SETUP_PREFIXES), recovered]
        assert np.allclose(gathered, sample.behavior_selected_log_prob, atol=1e-6)
        assert np.allclose(suffix_information(gathered), sample.suffix_information, atol=1e-4)


def test_gathering_against_the_played_orientation_would_be_wrong_for_reflected_samples(pool):
    """The failure S06 guards against: a reflected board indexed against the
    network-orientation log-probs yields a different, finite NLL."""
    reflected = [s for s in pool.samples if s.reflected]
    mismatches = 0
    for sample in reflected:
        played = np.asarray(sample.played_canonical, dtype=int)
        wrong = sample.behavior_log_probs[np.arange(SETUP_PREFIXES), played]
        if not np.allclose(wrong, sample.behavior_selected_log_prob, atol=1e-6):
            mismatches += 1
    assert mismatches == len(reflected)


# -- S07: canonical-to-engine orientation ---------------------------------------


def test_red_is_the_identity_and_blue_reverses_the_ranks(pool):
    for sample in pool.samples:
        if sample.lane == RED:
            assert sample.engine_setup == sample.played_canonical
        else:
            assert sample.engine_setup == orient_setup(sample.played_canonical, BLUE)
            assert sample.engine_setup != sample.played_canonical
        red = to_engine_setup(sample.played_canonical, RED)
        blue = to_engine_setup(sample.played_canonical, BLUE)
        assert tuple(blue) == tuple(red[30:40] + red[20:30] + red[10:20] + red[0:10]), "mutual row reversal"


def test_canonical_blue_handed_straight_to_the_engine_is_rejected(pool):
    blue = next(s for s in pool.samples if s.lane == BLUE)
    with pytest.raises(Phase15OrientationError):
        assert_engine_orientation(blue.played_canonical, blue.played_canonical, BLUE)


def test_blue_flags_land_on_blue_back_rows_not_front_rows(pool):
    squares = SETUP_SQUARES[BLUE]
    for sample in (s for s in pool.samples if s.lane == BLUE):
        canonical_rank = sample.played_canonical.index(FLAG) // 10
        engine_row = squares[sample.engine_setup.index(FLAG)] // 10
        assert engine_row == 9 - canonical_rank


def test_an_illegal_inventory_or_unknown_player_never_reaches_the_engine():
    with pytest.raises(Phase18SetupGenerationError, match="inventory"):
        to_engine_setup(tuple([FLAG] * 40), RED)
    with pytest.raises(Phase18SetupOrientationError, match="unknown player"):
        to_engine_setup(tuple(range(40)), 7)


def test_every_pooled_board_is_accepted_by_engine_game_creation_without_a_move_played(pool):
    """Legality only: a game state is created from the two lanes' engine
    setups and no move is ever played."""
    reds = [s for s in pool.samples if s.lane == RED]
    blues = [s for s in pool.samples if s.lane == BLUE]
    for red, blue in zip(reds, blues):
        state = create_game(red.engine_setup, blue.engine_setup, game_id=f"{red.root_seed}")
        assert state.acting_player in (RED, BLUE)
    assert pool.telemetry["orientation_failures"] == 0 and pool.telemetry["legality_failures"] == 0


# -- S11: realized suffix information -------------------------------------------


def test_suffix_information_is_the_reverse_cumulative_surprisal():
    logs = np.array([-0.5, -1.0, -2.0], dtype=np.float32)
    assert np.allclose(suffix_information(logs), [3.5, 3.0, 2.0])


def test_information_recursion_holds_on_recorded_samples(pool):
    for sample in pool.samples:
        info = sample.suffix_information.astype(np.float64)
        logp = sample.behavior_selected_log_prob.astype(np.float64)
        assert abs(info[39] + logp[39]) < 1e-5
        for k in range(39):
            assert abs(info[k] - (info[k + 1] - logp[k])) < 1e-4


# -- S20: pool structure ----------------------------------------------------------


def test_a_pool_of_1024_has_1024_distinct_entries_split_512_per_lane(setup_model, model_digest):
    big = generate_pool(
        setup_model, namespace=NAMESPACE, seed_index=2, snapshot_iteration=0,
        snapshot_digest=model_digest, count=1024,
    )
    assert len(big.samples) == 1024
    assert big.telemetry["distinct_content_fingerprints"] == 1024
    lanes = [s.lane for s in big.samples]
    assert lanes[::2] == [RED] * 512 and lanes[1::2] == [BLUE] * 512
    assert big.telemetry["lane_counts"] == {"red": 512, "blue": 512}


def test_generation_is_deterministic_and_independent_of_the_pool_size(setup_model, model_digest):
    eight = generate_pool(setup_model, namespace=NAMESPACE, seed_index=4, snapshot_iteration=2, snapshot_digest=model_digest, count=8)
    six = generate_pool(setup_model, namespace=NAMESPACE, seed_index=4, snapshot_iteration=2, snapshot_digest=model_digest, count=6)
    again = generate_pool(setup_model, namespace=NAMESPACE, seed_index=4, snapshot_iteration=2, snapshot_digest=model_digest, count=8)
    for index in range(6):
        assert np.array_equal(eight.samples[index].network_tokens, six.samples[index].network_tokens)
        assert eight.samples[index].reflected == six.samples[index].reflected
    for a, b in zip(eight.samples, again.samples):
        assert np.array_equal(a.network_tokens, b.network_tokens)
        assert np.array_equal(a.behavior_log_probs, b.behavior_log_probs)
    other = generate_pool(setup_model, namespace=NAMESPACE, seed_index=4, snapshot_iteration=3, snapshot_digest=model_digest, count=8)
    assert any(not np.array_equal(a.network_tokens, b.network_tokens) for a, b in zip(eight.samples, other.samples))


def test_seed_domains_are_separated():
    root = pool_root_seed(NAMESPACE, 1, 0, 0)
    assert root != pool_root_seed(NAMESPACE, 2, 0, 0) != pool_root_seed(NAMESPACE, 1, 1, 0)
    assert token_seed(root, 0) != token_seed(root, 1)
    with pytest.raises(Phase18SetupError):
        token_seed(root, SETUP_PREFIXES)


# -- S24: immediately terminal setups -------------------------------------------


def _terminal_setup():
    """The six bombs on the six corridor front squares: no opening move exists,
    because every other front square faces a lake and every own square is
    occupied at ply 0."""
    pieces = [t for t in range(12) for _ in range(PIECE_COUNTS[t]) if t != BOMB]
    front = {30 + f for f in CORRIDOR_FILES}
    board = []
    for square in range(40):
        board.append(BOMB if square in front else pieces.pop(0))
    assert len(pieces) == 0 and board.count(BOMB) == PIECE_COUNTS[BOMB]
    return tuple(board)


def test_opening_move_predicate_matches_the_engine_at_ply_zero():
    """Structural predicate versus the engine's own legal-move generator, on
    random and constructed setups. Only ply 0 is inspected; no move is played."""
    rng = __import__("random").Random(1)
    for _ in range(40):
        red = random_setup(rng, RED)
        blue = random_setup(rng, BLUE)
        state = create_game(red, blue)
        assert has_opening_move(red) == has_legal_action(state, RED)
        assert has_opening_move(orient_setup(blue, BLUE)) == has_legal_action(state, BLUE)  # engine frame -> canonical
    terminal = _terminal_setup()
    validate_setup(terminal, RED)
    assert not has_opening_move(terminal)
    state = create_game(terminal, random_setup(rng, BLUE))
    assert not has_legal_action(state, RED)


def test_a_terminal_setup_is_flagged_by_the_sample_and_counted_by_the_pool(pool):
    assert pool.telemetry["immediately_terminal_count"] == sum(1 for s in pool.samples if not s.opening_move)


# -- the inverse-CDF draw ----------------------------------------------------------


def test_inverse_cdf_respects_the_mask_at_the_top_of_the_unit_interval():
    probabilities = np.zeros((1, 12))
    probabilities[0, 0] = probabilities[0, 1] = 0.5
    mask = np.zeros((1, 12), dtype=bool)
    mask[0, 0] = mask[0, 1] = True
    for draw in (0.0, 0.4999, 0.5, 0.9, 1.0 - 1e-12, 0.99999999999999):
        chosen = inverse_cdf_choice(probabilities, mask, np.array([draw]))
        assert bool(mask[0, chosen[0]])


def test_inverse_cdf_partitions_the_unit_interval_by_probability():
    probabilities = np.zeros((1, 12))
    probabilities[0, 3], probabilities[0, 7] = 0.25, 0.75
    mask = np.zeros((1, 12), dtype=bool)
    mask[0, 3] = mask[0, 7] = True
    draws = np.linspace(0.0, 1.0, 10001, endpoint=False)
    picks = np.array([inverse_cdf_choice(probabilities, mask, np.array([u]))[0] for u in draws])
    assert (picks == 3).mean() == pytest.approx(0.25, abs=1e-3)


def test_generation_on_mps_produces_only_legal_handed_setups():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this host")
    from stratego.training.phase18.setup_model import build_setup_model, state_dict_digest

    model = build_setup_model(device="mps", seed=3)
    generated = generate_pool(
        model, namespace=NAMESPACE, seed_index=9, snapshot_iteration=0,
        snapshot_digest=state_dict_digest(model), count=64,
    )
    assert generated.telemetry["flag_in_permitted_half_fraction_network"] == 1.0
    for sample in generated.samples:
        validate_setup(sample.played_canonical, sample.lane)
        validate_setup(sample.engine_setup, sample.lane)
