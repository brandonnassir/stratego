"""Phase 11 Agent 3: the independent verification path, cross-validated.

The independent module rebuilds the frozen constants and derivations from
the engine authority and the published contract text; these tests are where
the two implementations meet. Agreement here is meaningful precisely
because `phase11_sampler_audit` imports none of the modules it is checking.
"""

import numpy as np
import pytest

from stratego.evaluation import phase11_sampler_audit as ind
from stratego.evaluation.phase11_baselines import remaining_counts, sample_world
from stratego.evaluation.phase11_public_state import (
    hidden_opponent_pieces,
    legal_rank_mask,
    public_state_identity,
)
from stratego.evaluation.phase11_sampler import sample_belief_world
from stratego.training.phase11_contract import (
    IMMOVABLE_RANK_INDICES,
    MOVABLE_RANK_INDICES,
    RANK_COUNT,
    RANK_INITIAL_COUNTS,
)
from stratego.training.phase11_seed import (
    derive_phase11_seed,
    phase11_sample_token,
    unit_uniform,
    world_categorical_uniform,
    world_order_key,
    world_sample_seed,
)

from .test_phase11_sampler import make_document, make_request, uniform_probabilities


def rich_document():
    return make_document(
        known_slots=(2, 14, 15), dead_slots=(9, 26), moved_slots=(1, 3, 18, 22, 29)
    )


# ---------------------------------------------------------------------------
# The rebuilt constants and derivations agree with the frozen originals
# ---------------------------------------------------------------------------


def test_the_engine_authority_matches_the_frozen_rank_space():
    assert ind.RANKS == RANK_COUNT
    assert ind.INITIAL_COUNTS == RANK_INITIAL_COUNTS
    assert tuple(sorted(ind.IMMOVABLE)) == IMMOVABLE_RANK_INDICES
    assert ind.MOVABLE == MOVABLE_RANK_INDICES


def test_the_raw_blake2b_derivation_matches_the_frozen_streams():
    document = rich_document()
    identity = public_state_identity(document)
    token = phase11_sample_token("belief_sampler_v1", identity, 12)
    assert ind.independent_sample_token("belief_sampler_v1", identity, 12) == token
    for slot in (0, 7, 39):
        assert ind.independent_seed("world_order", token, slot) == world_order_key(
            token, slot
        )
    for step in (0, 5, 31):
        seed = ind.independent_seed("world_categorical", token, step)
        assert ind.independent_unit_uniform(seed) == world_categorical_uniform(
            token, step
        )
        assert unit_uniform(seed) == ind.independent_unit_uniform(seed)
    assert ind.independent_seed("world_sample", token) == world_sample_seed(token)
    assert derive_phase11_seed("world_sample", token) == world_sample_seed(token)


def test_the_raw_document_walk_matches_the_frozen_public_facts():
    document = rich_document()
    assert ind.independent_identity(document) == public_state_identity(document)
    assert tuple(ind.independent_inventory(document)) == remaining_counts(document)
    primary = [int(piece["piece_slot"]) for piece in hidden_opponent_pieces(document)]
    rebuilt = [
        int(piece["piece_slot"]) for piece in ind.independent_hidden_pieces(document)
    ]
    assert rebuilt == primary
    assert ind.independent_mask(True) == legal_rank_mask(True)
    assert ind.independent_mask(False) == legal_rank_mask(False)


def test_the_scalar_softmax_matches_the_frozen_extraction():
    from stratego.evaluation.phase11_belief import softmax_float64

    logits = np.array([2.5, -1.0, 0.0, 4.25, -3.5, 1.5, 0.75, -0.25, 2.0, 1.0, -2.0, 0.5])
    primary = softmax_float64(logits.astype(np.float32))
    local = ind.independent_softmax(logits.astype(np.float32))
    assert max(abs(primary[rank] - local[rank]) for rank in range(RANK_COUNT)) <= 1e-15


# ---------------------------------------------------------------------------
# Full worlds re-derive
# ---------------------------------------------------------------------------


def test_learned_worlds_re_derive_exactly():
    document = rich_document()
    probabilities = uniform_probabilities(document)
    for ordinal in range(8):
        world = sample_belief_world(
            make_request(document, ordinal=ordinal, probabilities=probabilities)
        )
        report = ind.verify_world_independently(document, probabilities, world)
        assert report["agrees"], report["findings"]
        assert report["knife_edge_events"] == 0
        assert report["steps"] == len(world["assignment"])


def test_count_uniform_worlds_re_derive_exactly():
    document = rich_document()
    for ordinal in range(4):
        world = sample_world(document, ordinal)
        report = ind.verify_world_independently(document, None, world)
        assert report["agrees"], report["findings"]


def test_the_guard_recomputation_counts_pruned_steps():
    from .test_phase11_sampler import scout_and_bombs_document

    document = scout_and_bombs_document()
    probabilities = uniform_probabilities(document)
    world = sample_belief_world(make_request(document, probabilities=probabilities))
    report = ind.verify_world_independently(document, probabilities, world)
    assert report["agrees"], report["findings"]
    # Whenever an unmoved piece precedes the moved scout-taker, the guard
    # visibly pruned the scout from its legal set.
    assert report["guard_pruned_steps"] >= 1


# ---------------------------------------------------------------------------
# The verifier detects corruption
# ---------------------------------------------------------------------------


def test_a_corrupted_assignment_is_detected():
    document = rich_document()
    probabilities = uniform_probabilities(document)
    world = sample_belief_world(make_request(document, probabilities=probabilities))
    tampered = dict(world, assignment=dict(world["assignment"]))
    slots = sorted(tampered["assignment"])
    first = tampered["assignment"][slots[0]]
    tampered["assignment"][slots[0]] = (first + 1) % RANK_COUNT
    report = ind.verify_world_independently(document, probabilities, tampered)
    assert not report["agrees"]
    assert any("does not re-derive" in finding for finding in report["findings"])
    assert any("multiset" in finding for finding in report["findings"])


def test_a_corrupted_piece_order_is_detected():
    document = rich_document()
    probabilities = uniform_probabilities(document)
    world = sample_belief_world(make_request(document, probabilities=probabilities))
    tampered = dict(world, piece_order=list(reversed(world["piece_order"])))
    report = ind.verify_world_independently(document, probabilities, tampered)
    assert not report["agrees"]
    assert any("order" in finding for finding in report["findings"])


def test_a_corrupted_token_is_detected():
    document = rich_document()
    probabilities = uniform_probabilities(document)
    world = sample_belief_world(make_request(document, probabilities=probabilities))
    tampered = dict(world, sample_ordinal=int(world["sample_ordinal"]) + 1)
    report = ind.verify_world_independently(document, probabilities, tampered)
    assert not report["agrees"]
    assert any("token" in finding for finding in report["findings"])


def test_an_immobility_violation_is_detected():
    document = rich_document()
    probabilities = uniform_probabilities(document)
    world = sample_belief_world(make_request(document, probabilities=probabilities))
    tampered = dict(world, assignment=dict(world["assignment"]))
    tampered["assignment"][1] = 11  # slot 1 is publicly moved
    report = ind.verify_world_independently(document, probabilities, tampered)
    assert not report["agrees"]
    assert any("moved slot 1" in finding for finding in report["findings"])


def test_the_module_shares_no_phase11_implementation():
    import ast
    import inspect

    import stratego.evaluation.phase11_sampler_audit as module

    tree = ast.parse(inspect.getsource(module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any("phase11" in name for name in imported), imported
    assert not any("phase10" in name for name in imported), imported
    allowed = {"hashlib", "json", "math", "__future__"}
    relative = {name for name in imported if "engine" in name}
    assert imported <= allowed | relative, imported
