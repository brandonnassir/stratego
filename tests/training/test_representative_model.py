"""Correctness tests for the Phase 3 Agent 4 representative benchmark probe.

The network here is throw-away scaffolding for measuring Metal inference cost,
so these tests deliberately assert nothing about playing strength. What they do
assert is everything the benchmark's conclusions depend on:

- output shapes and finiteness on every available device;
- dense legality actually eliminates illegal actions;
- a sampled action is always one the engine declared legal;
- the value probe has exactly three classes (win, draw, loss);
- repeated forwards on one device/precision are numerically stable;
- the compact legality path gives the same normalised legal-set distribution
  as the dense mask;
- no hidden engine information reaches the model beyond the approved
  `observation_v2_1_127ch` tensor and the legality input.

Bitwise central-processing-unit vs Metal equality is explicitly *not* required
and is not tested; distributional agreement is.

The full batch-size / precision sweep lives in `scripts/run_phase3_agent04.py`.
"""

import random

import numpy as np
import pytest
import torch

from stratego.engine.constants import (
    ACTION_SPACE_SIZE,
    NUM_PIECE_TYPES,
    NUM_SQUARES,
    OBSERVATION_CHANNELS,
)
from stratego.engine.observation import belief_target, build_observation
from stratego.engine.permutation import belief_targets_differ, permute_hidden_identities
from stratego.training.batch_simulation import BatchSimulator
from stratego.training.representative_model import (
    IS_BENCHMARK_PROBE,
    REPRESENTATIVE_MODEL_VERSION,
    VALUE_CLASSES,
    CompactLegality,
    RepresentativeConfig,
    RepresentativeTransformer,
    build_compact_legality,
    build_representative_model,
    compact_legal_probabilities,
    dense_legal_probabilities,
    dense_mask_to_bool,
    observation_to_tokens,
    sample_compact,
    sample_dense,
    scatter_compact_probabilities,
)

MPS_AVAILABLE = torch.backends.mps.is_available()

DEVICES = ["cpu"] + (["mps"] if MPS_AVAILABLE else [])

# float16/bfloat16 on the central processing unit are slow and not what the
# benchmark cares about, so reduced precision is only exercised on Metal.
PRECISIONS = {
    "cpu": [torch.float32],
    "mps": [torch.float32, torch.float16, torch.bfloat16],
}


# ---------------------------------------------------------------------------
# Shared fixtures: real positions from the frozen engine
# ---------------------------------------------------------------------------


def _advance(simulator: BatchSimulator, plies: int, seed: int = 7) -> None:
    rng = random.Random(seed)
    for _ in range(plies):
        active = simulator.active_slots()
        if not active:
            return
        actions = {}
        for slot in active:
            legal = simulator.legal_actions(slot)
            actions[slot] = rng.choice(legal)
        simulator.step(actions)
        simulator.reset_finished()


@pytest.fixture(scope="module")
def simulator() -> BatchSimulator:
    batch = BatchSimulator(num_environments=16, root_seed=404)
    _advance(batch, 40)
    return batch


@pytest.fixture(scope="module")
def positions(simulator: BatchSimulator):
    """`(tokens, dense_mask, legal_lists)` for one real batch."""
    slots = simulator.active_slots()
    observations = simulator.observations(slots)
    masks = simulator.legal_action_masks(slots)
    lists = [list(row) for row in simulator.legal_action_lists(slots)]
    tokens = torch.from_numpy(observation_to_tokens(observations))
    return tokens, dense_mask_to_bool(masks), lists


@pytest.fixture(scope="module")
def cpu_model() -> RepresentativeTransformer:
    return build_representative_model(seed=0, device="cpu", dtype=torch.float32)


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


def test_probe_is_labelled_as_a_benchmark_probe(cpu_model):
    """The report is not the only place this must be marked temporary."""
    assert IS_BENCHMARK_PROBE is True
    assert cpu_model.is_benchmark_probe is True
    summary = cpu_model.architecture_summary()
    assert summary["benchmark_probe"] is True
    assert summary["trained"] is False
    assert "not the frozen model design" in summary["note"].lower()
    assert summary["version"] == REPRESENTATIVE_MODEL_VERSION


def test_architecture_matches_the_planning_target(cpu_model):
    config = cpu_model.config
    assert (config.num_tokens, config.input_features) == (NUM_SQUARES, OBSERVATION_CHANNELS)
    assert (config.width, config.num_layers, config.num_heads) == (128, 4, 4)
    assert config.feedforward_width == 512
    # A probe of the planned shape, not a scaled-up stand-in.
    assert 0.5e6 < cpu_model.parameter_count() < 3e6


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def test_tokenisation_is_a_pure_layout_change(simulator):
    observation = simulator.observation(0)
    tokens = observation_to_tokens(observation)
    assert tokens.shape == (NUM_SQUARES, OBSERVATION_CHANNELS)
    assert tokens.dtype == np.float32
    for channel in range(OBSERVATION_CHANNELS):
        plane = observation[channel].reshape(NUM_SQUARES)
        assert np.array_equal(tokens[:, channel], plane)


def test_tokenisation_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        observation_to_tokens(np.zeros((10, 10), dtype=np.float32))
    with pytest.raises(ValueError):
        observation_to_tokens(np.zeros((64, 10, 10), dtype=np.float32))


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_output_shapes_and_finiteness(device, positions):
    tokens, _, _ = positions
    for dtype in PRECISIONS[device]:
        model = build_representative_model(device=device, dtype=dtype)
        with torch.inference_mode():
            outputs = model(tokens.to(device=device, dtype=dtype))
        batch = tokens.shape[0]
        assert outputs.policy_logits.shape == (batch, ACTION_SPACE_SIZE)
        assert outputs.value_logits.shape == (batch, VALUE_CLASSES)
        assert outputs.belief_logits.shape == (batch, NUM_SQUARES, NUM_PIECE_TYPES)
        assert outputs.all_finite(), f"non-finite outputs on {device}/{dtype}"


@pytest.mark.parametrize("device", DEVICES)
def test_value_probe_has_three_classes(device, positions):
    """Win, draw and loss -- and they must form a distribution."""
    tokens, _, _ = positions
    model = build_representative_model(device=device, dtype=torch.float32)
    with torch.inference_mode():
        value_logits = model(tokens.to(device=device)).value_logits
    assert value_logits.shape[1] == 3 == VALUE_CLASSES
    probabilities = torch.softmax(value_logits.float(), dim=1)
    assert torch.allclose(
        probabilities.sum(dim=1), torch.ones(tokens.shape[0], device=probabilities.device)
    )


def test_forward_rejects_wrong_token_shape(cpu_model):
    with pytest.raises(ValueError):
        cpu_model(torch.zeros(2, NUM_SQUARES))
    with pytest.raises(ValueError):
        cpu_model(torch.zeros(2, 64, OBSERVATION_CHANNELS))


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_dense_legality_eliminates_illegal_actions(device, positions):
    tokens, mask, legal_lists = positions
    model = build_representative_model(device=device, dtype=torch.float32)
    mask_device = mask.to(device)
    with torch.inference_mode():
        probabilities = dense_legal_probabilities(
            model(tokens.to(device=device)).policy_logits, mask_device
        )
    probabilities = probabilities.cpu()
    # Every illegal entry is exactly zero, and the legal set carries all the mass.
    assert float(probabilities[~mask].abs().max()) == 0.0
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(tokens.shape[0]), atol=1e-5)
    for row, legal in enumerate(legal_lists):
        assert float(probabilities[row, legal].sum()) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("legality", ["dense", "compact"])
def test_sampled_action_is_always_legal(device, legality, positions):
    tokens, mask, legal_lists = positions
    model = build_representative_model(device=device, dtype=torch.float32)
    compact = build_compact_legality(legal_lists).to(device)
    mask_device = mask.to(device)
    legal_sets = [set(row) for row in legal_lists]

    with torch.inference_mode():
        policy_logits = model(tokens.to(device=device)).policy_logits
        for draw in range(25):
            generator = None
            torch.manual_seed(1000 + draw)
            if legality == "dense":
                actions = sample_dense(policy_logits, mask_device, generator=generator)
            else:
                actions = sample_compact(policy_logits, compact, generator=generator)
            for row, action in enumerate(actions.cpu().tolist()):
                assert action in legal_sets[row], (
                    f"{legality} sampling produced illegal action {action} "
                    f"for row {row} on {device}"
                )


@pytest.mark.parametrize("device", DEVICES)
def test_compact_and_dense_agree_on_legal_probabilities(device, positions):
    tokens, mask, legal_lists = positions
    model = build_representative_model(device=device, dtype=torch.float32)
    compact = build_compact_legality(legal_lists).to(device)
    with torch.inference_mode():
        policy_logits = model(tokens.to(device=device)).policy_logits
        dense = dense_legal_probabilities(policy_logits, mask.to(device))
        scattered = scatter_compact_probabilities(
            compact_legal_probabilities(policy_logits, compact), compact
        )
    assert float((dense - scattered).abs().max()) < 1e-5


def test_compact_padding_is_masked_out(cpu_model, positions):
    """Padded entries must carry exactly zero probability."""
    tokens, _, legal_lists = positions
    compact = build_compact_legality(legal_lists, capacity=128)
    with torch.inference_mode():
        probabilities = compact_legal_probabilities(
            cpu_model(tokens).policy_logits, compact
        )
    assert float(probabilities[~compact.valid].abs().max()) == 0.0
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(tokens.shape[0]), atol=1e-6)


def test_compact_capacity_overflow_fails_loudly():
    """Silently truncating a legal move would be a correctness bug sold as speed."""
    with pytest.raises(ValueError, match="capacity"):
        build_compact_legality([[1, 2, 3, 4]], capacity=2)


def test_dense_mask_shape_is_validated():
    with pytest.raises(ValueError):
        dense_mask_to_bool(np.zeros((4, 999), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_repeated_forward_is_stable_for_one_device_and_precision(device, positions):
    """Not a cross-device equality claim: same device, same precision, same input."""
    tokens, _, _ = positions
    for dtype in PRECISIONS[device]:
        model = build_representative_model(device=device, dtype=dtype)
        staged = tokens.to(device=device, dtype=dtype)
        with torch.inference_mode():
            first = model(staged).policy_logits.float().cpu().clone()
            for _ in range(4):
                again = model(staged).policy_logits.float().cpu()
                assert float((again - first).abs().max()) == 0.0, (
                    f"repeated forward drifted on {device}/{dtype}"
                )


@pytest.mark.skipif(not MPS_AVAILABLE, reason="requires Metal Performance Shaders")
def test_metal_and_cpu_agree_distributionally(positions):
    """Explicitly a *distributional* check; bitwise equality is not required."""
    tokens, _, legal_lists = positions
    cpu = build_representative_model(device="cpu", dtype=torch.float32)
    metal = build_representative_model(device="mps", dtype=torch.float32)
    compact_cpu = build_compact_legality(legal_lists)
    compact_mps = compact_cpu.to("mps")
    with torch.inference_mode():
        cpu_probabilities = compact_legal_probabilities(cpu(tokens).policy_logits, compact_cpu)
        metal_probabilities = compact_legal_probabilities(
            metal(tokens.to("mps")).policy_logits, compact_mps
        ).cpu()
    assert float((cpu_probabilities - metal_probabilities).abs().max()) < 1e-4
    agreement = float(
        (cpu_probabilities.argmax(1) == metal_probabilities.argmax(1)).float().mean()
    )
    assert agreement > 0.95


# ---------------------------------------------------------------------------
# Hidden information
# ---------------------------------------------------------------------------


def test_hidden_identities_cannot_reach_the_model(cpu_model, simulator):
    """Permuting hidden opponent types must not move a single model output.

    This is the anti-leak argument for the whole benchmark path: the model sees
    `build_observation` plus the legality input and nothing else, so a state
    and a hidden-identity permutation of it are indistinguishable to it, while
    the privileged belief target does change.
    """
    rng = random.Random(99)
    checked = 0
    for slot in simulator.active_slots():
        state = simulator.game_state(slot)
        observer = state.acting_player
        permuted, info = permute_hidden_identities(state, observer, rng)
        if not (info["valid"] and info["changed"]):
            continue
        checked += 1

        original_tokens = torch.from_numpy(
            observation_to_tokens(build_observation(state, observer))
        )[None]
        permuted_tokens = torch.from_numpy(
            observation_to_tokens(build_observation(permuted, observer))
        )[None]
        assert torch.equal(original_tokens, permuted_tokens)

        with torch.inference_mode():
            first = cpu_model(original_tokens)
            second = cpu_model(permuted_tokens)
        assert torch.equal(first.policy_logits, second.policy_logits)
        assert torch.equal(first.value_logits, second.value_logits)
        assert torch.equal(first.belief_logits, second.belief_logits)

        # Positive control: the privileged target really did change, so the
        # test above is not vacuous.
        assert belief_targets_differ(state, permuted, observer)
    assert checked >= 4, "expected several valid changed permutations to compare"


def test_belief_targets_are_never_a_model_input(monkeypatch, cpu_model, simulator):
    """Building and running the model must not touch the privileged labels."""
    import stratego.engine.observation as observation_module

    def explode(*args, **kwargs):  # pragma: no cover - only runs on failure
        raise AssertionError("belief_target must not be reachable from the model path")

    monkeypatch.setattr(observation_module, "belief_target", explode)

    state = simulator.game_state(0)
    tokens = torch.from_numpy(observation_to_tokens(build_observation(state)))[None]
    mask = dense_mask_to_bool(simulator.legal_action_masks([0]))
    with torch.inference_mode():
        actions = sample_dense(cpu_model(tokens).policy_logits, mask)
    assert int(actions[0]) in set(simulator.legal_actions(0))


def test_model_input_width_is_exactly_the_observation_contract(cpu_model):
    """127 features per token: no extra privileged channel can be smuggled in."""
    assert cpu_model.input_projection.in_features == OBSERVATION_CHANNELS
    assert cpu_model.config.input_features == OBSERVATION_CHANNELS
    with pytest.raises(ValueError):
        cpu_model(torch.zeros(1, NUM_SQUARES, OBSERVATION_CHANNELS + 1))


# ---------------------------------------------------------------------------
# Determinism of construction
# ---------------------------------------------------------------------------


def test_same_seed_gives_the_same_weights():
    first = build_representative_model(seed=5)
    second = build_representative_model(seed=5)
    third = build_representative_model(seed=6)
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)
    assert any(
        not torch.equal(left, right)
        for left, right in zip(first.parameters(), third.parameters())
    )


def test_configuration_rejects_indivisible_head_count():
    with pytest.raises(ValueError):
        RepresentativeTransformer(RepresentativeConfig(width=130, num_heads=4))


def test_compact_legality_reports_capacity_and_counts(positions):
    _, _, legal_lists = positions
    compact = build_compact_legality(legal_lists, capacity=64)
    assert isinstance(compact, CompactLegality)
    assert compact.capacity == 64
    assert compact.counts.tolist() == [len(row) for row in legal_lists]
    assert int(compact.valid.sum()) == sum(len(row) for row in legal_lists)
