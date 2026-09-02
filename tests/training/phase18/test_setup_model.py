"""S01, S03, S30: the setup network's shape, heads, alignment and causality."""

import numpy as np
import pytest
import torch

from stratego.engine.constants import NUM_PIECE_TYPES
from stratego.training.phase18.setup_contract import (
    POSITIONAL_INIT_STD,
    SETUP_PARAMETER_TARGET,
    SETUP_PREFIXES,
    SETUP_SEQUENCE_LENGTH,
    START_TOKEN,
    Phase18SetupError,
)
from stratego.training.phase18.setup_model import (
    Phase18SetupModel,
    assert_architecture,
    build_setup_model,
    count_parameters,
    state_dict_digest,
)


# -- S30: the frozen architecture ---------------------------------------------


def test_parameter_count_is_exactly_802320(setup_model):
    assert count_parameters(setup_model) == SETUP_PARAMETER_TARGET == 802_320


def test_architecture_is_4_128_4_512_pre_layernorm(setup_model):
    architecture = assert_architecture(setup_model)
    assert (architecture["blocks"], architecture["width"]) == (4, 128)
    assert (architecture["heads"], architecture["feed_forward_width"]) == (4, 512)
    assert architecture["normalization"] == "pre_layernorm"
    assert architecture["vocabulary"] == 13
    assert architecture["sequence_length"] == SETUP_SEQUENCE_LENGTH == 41
    assert architecture["prefixes"] == SETUP_PREFIXES == 40
    assert (architecture["piece_classes"], architecture["wdl_classes"]) == (12, 3)


def test_a_different_width_is_refused_rather_than_accepted_quietly():
    narrow = Phase18SetupModel(feed_forward_width=51)
    assert count_parameters(narrow) != SETUP_PARAMETER_TARGET
    with pytest.raises(Phase18SetupError, match="outside the frozen band"):
        assert_architecture(narrow)


def test_state_dict_shapes_match_the_accepted_phase17_architecture(setup_model):
    """The Phase 17 module is read for comparison only; nothing imports it on
    the Phase 18 training path."""
    from stratego.training.phase17.setup_model import build_setup_model as phase17_build

    phase17 = phase17_build(device="cpu", seed=1)
    phase18_shapes = {name: tuple(t.shape) for name, t in setup_model.state_dict().items()}
    phase17_shapes = {name: tuple(t.shape) for name, t in phase17.state_dict().items()}
    # The three head names differ by contract (published order and units are
    # documented on the Phase 18 side); every other tensor is identical.
    rename = {"conditional_entropy_head.weight": "entropy_head.weight", "conditional_entropy_head.bias": "entropy_head.bias"}
    phase17_shapes = {rename.get(name, name): shape for name, shape in phase17_shapes.items()}
    assert phase18_shapes == phase17_shapes
    assert count_parameters(phase17) == count_parameters(setup_model) == 802_320


def test_positional_embeddings_initialise_at_std_0_1():
    """Paper Table 23 / published `pos_emb_std = 0.1`, measured over 40 seeds."""
    stds = [float(build_setup_model(seed=seed).positional_embedding.weight.std()) for seed in range(40)]
    assert abs(np.mean(stds) - POSITIONAL_INIT_STD) < 0.004
    assert all(0.09 < s < 0.11 for s in stds)


def test_build_is_reproducible_from_a_seed_and_leaves_the_global_rng_alone():
    torch.manual_seed(5)
    before = torch.rand(3)
    torch.manual_seed(5)
    first = build_setup_model(seed=99)
    after = torch.rand(3)
    second = build_setup_model(seed=99)
    third = build_setup_model(seed=100)
    assert state_dict_digest(first) == state_dict_digest(second)
    assert state_dict_digest(first) != state_dict_digest(third)
    assert torch.equal(before, after), "building a model must not consume the global stream"


# -- S01: alignment and causality --------------------------------------------


def test_every_head_reports_at_all_forty_prefixes(setup_model):
    tokens = torch.full((3, SETUP_SEQUENCE_LENGTH), START_TOKEN, dtype=torch.long)
    outputs = setup_model(tokens)
    assert outputs["piece_logits"].shape == (3, SETUP_PREFIXES, NUM_PIECE_TYPES)
    assert outputs["wdl_logits"].shape == (3, SETUP_PREFIXES, 3)
    assert outputs["entropy_prediction"].shape == (3, SETUP_PREFIXES)


def test_prefix_k_is_a_function_of_the_first_k_placements_only(setup_model):
    """Causality: perturbing any placement j > k leaves output k bit-identical."""
    generator = torch.Generator().manual_seed(11)
    base = torch.randint(0, 12, (2, SETUP_SEQUENCE_LENGTH), generator=generator)
    base[:, 0] = START_TOKEN
    with torch.no_grad():
        reference = setup_model(base)
    for k in (0, 7, 20, 38):
        mutated = base.clone()
        mutated[:, k + 1 :] = (mutated[:, k + 1 :] + 5) % 12  # placements k.. onwards
        with torch.no_grad():
            outputs = setup_model(mutated)
        for name in ("piece_logits", "wdl_logits", "entropy_prediction"):
            assert torch.equal(outputs[name][:, : k + 1], reference[name][:, : k + 1]), (name, k)


def test_prefix_k_has_seen_exactly_k_placements(setup_model):
    """Alignment: changing placement k-1 changes output k but not output k-1."""
    generator = torch.Generator().manual_seed(12)
    base = torch.randint(0, 12, (1, SETUP_SEQUENCE_LENGTH), generator=generator)
    base[:, 0] = START_TOKEN
    with torch.no_grad():
        reference = setup_model(base)
    for k in (1, 10, 39):
        mutated = base.clone()
        mutated[0, k] = (mutated[0, k] + 3) % 12  # placement k-1 sits at sequence position k
        with torch.no_grad():
            outputs = setup_model(mutated)
        assert torch.equal(outputs["piece_logits"][:, k - 1], reference["piece_logits"][:, k - 1])
        assert not torch.allclose(outputs["piece_logits"][:, k], reference["piece_logits"][:, k], atol=1e-6)


def test_sequence_longer_than_the_contract_is_refused(setup_model):
    tokens = torch.full((1, SETUP_SEQUENCE_LENGTH + 1), START_TOKEN, dtype=torch.long)
    with pytest.raises(Phase18SetupError, match="sequence length"):
        setup_model(tokens)


# -- S03: the 12-way head is the published 14-way head restricted ---------------


def test_twelve_way_softmax_equals_the_published_fourteen_way_softmax_restricted_to_live_classes():
    """The published head carries `lake` and `empty` with inventory 0, so its
    mask fills them with `finfo.min`; the softmax over 14 then equals the
    12-way softmax on the live classes exactly."""
    generator = torch.Generator().manual_seed(3)
    live = torch.randn(5, 12, generator=generator)
    fourteen = torch.cat([live, torch.full((5, 2), torch.finfo(torch.float32).min)], dim=1)
    restricted = torch.softmax(fourteen, dim=-1)
    assert torch.equal(restricted[:, 12:], torch.zeros(5, 2))
    assert torch.allclose(restricted[:, :12], torch.softmax(live, dim=-1), atol=1e-7)
