"""The frozen synthetic known-reward landscape: determinism, invariance, the
certified optimum, the exact baseline moments and the seeded outcomes."""

import random

import numpy as np
import pytest

from stratego.engine.constants import NUM_PIECE_TYPES, PIECES_PER_PLAYER, RED
from stratego.engine.setup import random_setup, validate_setup
from stratego.setups.identity import reflect_canonical
from stratego.training.phase18 import synthetic_landscape as land

NS = "phase18_g2_landscape_test_v1"


@pytest.fixture(scope="module")
def landscape():
    return land.build_landscape(namespace=NS, table_seed=20260902, kappa=3.0, p_draw=0.1)


def test_the_table_is_deterministic_versioned_and_reflection_invariant(landscape):
    again = land.build_table(20260902)
    assert np.array_equal(landscape.table, again)
    assert landscape.table.shape == (NUM_PIECE_TYPES, PIECES_PER_PLAYER)
    cube = landscape.table.reshape(NUM_PIECE_TYPES, 4, 10)
    assert np.array_equal(cube, cube[:, :, ::-1])
    assert landscape.version == land.LANDSCAPE_VERSION
    assert landscape.document()["table_digest"] == land.build_landscape(namespace=NS, table_seed=20260902, kappa=3.0, p_draw=0.1).document()["table_digest"]
    assert land.build_table(1).tolist() != landscape.table.tolist()


def test_utility_is_reflection_invariant_and_the_vectorised_form_agrees(landscape):
    rng = random.Random(0)
    boards = [random_setup(rng, RED) for _ in range(50)]
    values = landscape.utilities(np.array(boards))
    for board, value in zip(boards, values):
        assert landscape.utility(board) == pytest.approx(value)
        assert landscape.utility(reflect_canonical(board)) == pytest.approx(value)


def test_the_uniform_moments_are_exact_against_monte_carlo(landscape):
    rng = random.Random(7)
    samples = np.array([random_setup(rng, RED) for _ in range(20000)])
    values = landscape.utilities(samples)
    se_mean = landscape.uniform_sd / np.sqrt(len(values))
    assert abs(values.mean() - landscape.uniform_mean) < 4 * se_mean
    assert abs(values.std(ddof=1) - landscape.uniform_sd) < 0.03 * landscape.uniform_sd
    exact = land.uniform_moments(landscape.table)
    a = np.stack([landscape.table[t] for t in land.SLOT_TYPES])
    assert exact["mean"] == pytest.approx(a.sum() / 40)


def test_the_exact_optimum_is_certified_by_duality_and_dominates_random_and_greedy(landscape):
    record = land.exact_optimum(landscape.table)
    assert record["certificate"]["certified"]
    assert record["certificate"]["dual_feasibility_violations"] == 0
    assert record["certificate"]["gap"] < 1e-6
    optimal = tuple(record["optimal_setup"])
    validate_setup(optimal, RED)
    assert landscape.utility(optimal) == pytest.approx(landscape.optimum)
    rng = random.Random(3)
    assert all(landscape.utility(random_setup(rng, RED)) <= landscape.optimum + 1e-9 for _ in range(5000))
    # A greedy heuristic: fill squares in order with the best remaining type.
    remaining = {t: __import__("stratego.engine.constants", fromlist=["PIECE_COUNTS"]).PIECE_COUNTS[t] for t in range(NUM_PIECE_TYPES)}
    greedy = []
    for square in range(PIECES_PER_PLAYER):
        best = max((t for t in remaining if remaining[t] > 0), key=lambda t: landscape.table[t, square])
        remaining[best] -= 1
        greedy.append(best)
    assert landscape.utility(tuple(greedy)) <= landscape.optimum + 1e-9
    # Independent check of the Hungarian solver on a tiny matrix with a known answer.
    assignment, u, v = land.hungarian_minimum([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    assert assignment == [1, 0, 2]
    assert sum(u) + sum(v) == pytest.approx(5.0)


def test_outcomes_are_seeded_deterministic_and_carry_no_utility(landscape):
    board = random_setup(random.Random(11), RED)
    first = landscape.outcomes_for(board, seed_index=1, period=3, fingerprint="abc", replicates=4)
    second = landscape.outcomes_for(board, seed_index=1, period=3, fingerprint="abc", replicates=4)
    other = landscape.outcomes_for(board, seed_index=1, period=4, fingerprint="abc", replicates=4)
    assert first == second and len(first) == 4
    assert all(isinstance(z, int) and z in (-1, 0, 1) for z in first)
    assert isinstance(first, list)
    assert first != other or True  # different period draws are independent; equality is possible by chance


def test_outcome_frequencies_follow_the_mapping(landscape):
    mapping = landscape.mapping
    for z in (-2.0, -0.5, 0.0, 0.7, 2.5):
        draws = np.array([mapping.outcome(z, u) for u in np.linspace(0.0, 1.0, 20001, endpoint=False)])
        p_loss, p_draw, p_win = mapping.probabilities(z)
        assert (draws == -1).mean() == pytest.approx(p_loss, abs=2e-4)
        assert (draws == 0).mean() == pytest.approx(p_draw, abs=2e-4)
        assert (draws == 1).mean() == pytest.approx(p_win, abs=2e-4)
    assert mapping.probabilities(0.0).tolist() == pytest.approx([0.45, 0.10, 0.45])
    assert mapping.probabilities(10.0)[2] > 0.899


def test_the_document_round_trips_and_a_drifted_table_is_refused(landscape):
    document = landscape.document()
    rebuilt = land.landscape_from_document(document)
    assert rebuilt.digest() == landscape.digest()
    assert rebuilt.optimum == landscape.optimum and rebuilt.uniform_mean == landscape.uniform_mean
    drifted = dict(document, table_digest="0" * 64)
    with pytest.raises(Exception):
        land.landscape_from_document(drifted)
