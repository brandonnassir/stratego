"""Phase 17 Agent 3 section 2: the setup network's shape and causality."""

import pytest
import torch

from stratego.training.phase17.setup_contract import (
    SETUP_PARAMETER_TARGET,
    SETUP_PREFIXES,
    SETUP_SEQUENCE_LENGTH,
    START_TOKEN,
    Phase17SetupError,
)
from stratego.training.phase17.setup_model import (
    Phase17SetupModel,
    assert_architecture,
    build_setup_model,
    count_parameters,
)


def test_parameter_count_is_agent_1s_frozen_target(setup_model):
    """802,320 is arithmetic Agent 1 froze before any model existed."""
    assert count_parameters(setup_model) == SETUP_PARAMETER_TARGET


def test_architecture_is_4_128_4_512(setup_model):
    architecture = assert_architecture(setup_model)
    assert (architecture["blocks"], architecture["width"]) == (4, 128)
    assert (architecture["heads"], architecture["feed_forward_width"]) == (4, 512)
    assert architecture["normalization"] == "pre_layernorm"
    assert architecture["vocabulary"] == 13
    assert architecture["sequence_length"] == SETUP_SEQUENCE_LENGTH


def test_a_different_width_is_refused_rather_than_accepted_quietly():
    """Section 2: fail the gate rather than quietly changing widths."""
    narrow = Phase17SetupModel(feed_forward_width=51)
    assert count_parameters(narrow) == 328_412
    with pytest.raises(Phase17SetupError, match="outside the frozen band"):
        assert_architecture(narrow)


def test_every_head_reports_at_all_forty_prefixes(setup_model):
    tokens = torch.full((3, SETUP_SEQUENCE_LENGTH), START_TOKEN, dtype=torch.long)
    outputs = setup_model(tokens)
    assert outputs["piece_logits"].shape == (3, SETUP_PREFIXES, 12)
    assert outputs["wdl_logits"].shape == (3, SETUP_PREFIXES, 3)
    assert outputs["conditional_entropy"].shape == (3, SETUP_PREFIXES)


def test_suffix_mutation_cannot_change_an_earlier_prefix(setup_model):
    """The required causality test: prefix k does not see tokens > k."""
    base = torch.randint(0, 12, (2, SETUP_SEQUENCE_LENGTH), dtype=torch.long)
    base[:, 0] = START_TOKEN
    mutated = base.clone()
    # Change everything from placement 20 onward.
    mutated[:, 21:] = (mutated[:, 21:] + 5) % 12

    with torch.no_grad():
        first = setup_model(base)
        second = setup_model(mutated)

    for name in ("piece_logits", "wdl_logits", "conditional_entropy"):
        earlier = slice(None, 20)
        assert torch.allclose(first[name][:, earlier], second[name][:, earlier], atol=1e-6), name
    # And the change is genuinely visible later, so the test is not vacuous.
    assert not torch.allclose(
        first["piece_logits"][:, 25], second["piece_logits"][:, 25], atol=1e-6
    )


def test_sequence_longer_than_the_contract_is_refused(setup_model):
    tokens = torch.full((1, SETUP_SEQUENCE_LENGTH + 1), START_TOKEN, dtype=torch.long)
    with pytest.raises(Phase17SetupError, match="sequence length"):
        setup_model(tokens)


def test_build_is_reproducible_from_a_seed():
    from stratego.training.phase9_behavior import state_dict_digest

    first = build_setup_model(seed=99)
    second = build_setup_model(seed=99)
    third = build_setup_model(seed=100)
    assert state_dict_digest(first) == state_dict_digest(second)
    assert state_dict_digest(first) != state_dict_digest(third)
