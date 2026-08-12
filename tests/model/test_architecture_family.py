"""The Phase 6 candidate family: one implementation, seven reproducible sizes.

Covers the Agent 2 completion gates: the C0-C6 ladder is explicit and
serializable, construction is deterministic from `(configuration, seed)`, the
three heads match `model_contract_v2`, the policy head stays in the normalized
frame, no privileged input exists, and a checkpoint written for one candidate
cannot be loaded as another even when every tensor shape agrees.

Nothing here measures speed or playing strength. Agent 3 owns benchmarking, and
the Phase 6 rules forbid treating random-weight strength as evidence at all.
"""

from __future__ import annotations

import pytest
import torch

from stratego.engine.constants import BOARD_COLUMNS, BOARD_ROWS
from stratego.evaluation.policy import PolicyRequirements, build_policy_input
from stratego.model.architecture_configs import (
    ARCHITECTURE_FAMILY,
    ARCHITECTURE_FAMILY_VERSION,
    CANDIDATE_IDS,
    CANDIDATE_ROLES,
    CANDIDATES,
    CONFIG_FIELDS,
    FAMILY_CONSTANTS,
    FAMILY_INITIALIZATION_SEED,
    ArchitectureConfigError,
    CandidateConfig,
    architecture_family_digest,
    candidate_config,
    candidate_configs,
    candidate_table,
    config_digests,
    family_summary,
)
from stratego.model.base import StrategoModel
from stratego.model.checkpoint import (
    CheckpointCompatibilityError,
    architecture_registration,
    build_checkpoint_payload,
    load_checkpoint,
    load_checkpoint_into,
    registered_architectures,
    save_checkpoint,
    state_dict_digest,
    validate_checkpoint_payload,
)
from stratego.model.contract import (
    BELIEF_IGNORE_INDEX,
    BELIEF_TYPE_COUNT,
    MODEL_CONTRACT_VERSION,
    POLICY_ACTION_FRAME,
    POLICY_LOGIT_COUNT,
    TOKEN_COUNT,
    TOKEN_FEATURES,
    VALUE_CLASS_COUNT,
    ModelContractError,
)
from stratego.model.integration_model import MODEL_ARCHITECTURE_ID, build_integration_model
from stratego.model.losses import multi_head_loss
from stratego.model.policy_adapter import GreedyNeuralPolicy
from stratego.model.production_model import (
    ProductionModel,
    benchmark_observation_batch,
    benchmark_token_batch,
    build_candidate_model,
    validate_candidate_outputs,
)
from stratego.model.tokenization import square_to_row_column

from ..helpers import nonterminal_state

#: The literal instruction table. Written out a second time, by hand, so that a
#: typo in `architecture_configs._LADDER` is a test failure rather than a new
#: source of truth silently agreeing with itself.
INSTRUCTION_LADDER = {
    "C0": (64, 2, 4, 256),
    "C1": (128, 4, 4, 512),
    "C2": (192, 4, 6, 768),
    "C3": (192, 6, 6, 768),
    "C4": (256, 6, 8, 1024),
    "C5": (256, 8, 8, 1024),
    "C6": (384, 8, 8, 1536),
}

#: Exact trainable parameter counts, pinned. These are what Agent 3's compute
#: and memory projections are built on: if an initialization or head change
#: moves one of them, that must be a decision, not a surprise.
EXPECTED_PARAMETERS = {
    "C0": 123_223,
    "C1": 863_959,
    "C2": 1_922_519,
    "C3": 2_812_247,
    "C4": 4_978_391,
    "C5": 6_557_911,
    "C6": 14_702_807,
}

#: The candidates cheap enough to build repeatedly in a unit test. The heavy
#: ones are covered once each by the shared `every_candidate` fixture and
#: exhaustively by `scripts/run_phase6_agent02.py`.
SMALL_CANDIDATES = ("C0", "C1")

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal is not available on this host"
)


@pytest.fixture(scope="module")
def candidate_models() -> dict:
    """Every candidate, built once, on CPU in float32 at the family seed."""
    return {
        candidate_id: build_candidate_model(candidate_id) for candidate_id in CANDIDATE_IDS
    }


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def test_the_ladder_is_the_instruction_table():
    assert list(CANDIDATE_IDS) == sorted(INSTRUCTION_LADDER)
    for candidate_id, (width, blocks, heads, feed_forward) in INSTRUCTION_LADDER.items():
        config = candidate_config(candidate_id)
        assert (config.width, config.blocks, config.heads, config.feed_forward_width) == (
            width,
            blocks,
            heads,
            feed_forward,
        )


def test_every_candidate_satisfies_the_one_hard_pytorch_constraint():
    """`nn.MultiheadAttention` requires `width % heads == 0`, and no row needed
    an adjustment -- which is the claim the report makes, so it is tested."""
    for config in CANDIDATES.values():
        assert config.width % config.heads == 0
        assert config.head_dimension == config.width // config.heads


def test_every_candidate_has_a_role_and_a_table_row():
    table = {row["candidate_id"]: row for row in candidate_table()}
    assert set(table) == set(CANDIDATE_IDS)
    for candidate_id, row in table.items():
        assert CANDIDATE_ROLES[candidate_id]
        assert row["role"] == CANDIDATE_ROLES[candidate_id]
        assert row["config_digest"] == CANDIDATES[candidate_id].digest()


def test_the_ladder_is_monotone_in_capacity():
    """C0 through C6 is a scaling ladder; a row that shrank would make Agent 3's
    frontier non-monotone for a reason unrelated to the hardware."""
    counts = [
        CANDIDATES[candidate_id].width * CANDIDATES[candidate_id].blocks
        for candidate_id in CANDIDATE_IDS
    ]
    assert counts == sorted(counts)


def test_the_family_is_not_the_phase_5_fixture():
    assert ARCHITECTURE_FAMILY != MODEL_ARCHITECTURE_ID
    assert ARCHITECTURE_FAMILY not in {"integration_model_v1", "ataraxos_full_v1"}
    assert set(registered_architectures()) == {MODEL_ARCHITECTURE_ID, ARCHITECTURE_FAMILY}


# ---------------------------------------------------------------------------
# Configuration: serialization and refusal
# ---------------------------------------------------------------------------


def test_a_configuration_round_trips_exactly():
    for config in CANDIDATES.values():
        payload = config.to_dict()
        assert list(payload) == list(CONFIG_FIELDS)
        assert CandidateConfig.from_dict(payload) == config
        assert CandidateConfig.from_dict(payload).digest() == config.digest()


def test_a_configuration_is_json_serializable():
    import json

    restored = json.loads(json.dumps(candidate_configs()))
    for candidate_id, payload in restored.items():
        assert CandidateConfig.from_dict(payload) == CANDIDATES[candidate_id]


def test_an_unknown_or_missing_configuration_field_is_refused():
    payload = CANDIDATES["C1"].to_dict()
    with pytest.raises(ArchitectureConfigError, match="unknown"):
        CandidateConfig.from_dict({**payload, "mystery": 3})
    incomplete = {key: value for key, value in payload.items() if key != "width"}
    with pytest.raises(ArchitectureConfigError, match="missing"):
        CandidateConfig.from_dict(incomplete)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"heads": 5}, "divide evenly"),
        ({"width": 0}, "positive"),
        ({"blocks": -1}, "positive"),
        ({"belief_classes": 13}, "belief_classes"),
        ({"value_classes": 2}, "value_classes"),
        ({"input_channels": 128}, "input_channels"),
        ({"board_tokens": 64}, "board_tokens"),
        ({"policy_size": 9999}, "policy_size"),
        ({"dropout": 1.0}, "dropout"),
        ({"dropout": -0.1}, "dropout"),
        ({"position_encoding": "sinusoidal"}, "position_encoding"),
        ({"normalization": "post_layernorm"}, "normalization"),
        ({"architecture_family_version": "architecture_family_v2"}, "architecture_family_version"),
        ({"candidate_id": "  "}, "candidate_id"),
    ],
)
def test_an_impossible_configuration_is_refused(changes, message):
    with pytest.raises(ArchitectureConfigError, match=message):
        CANDIDATES["C2"].replace(**changes)


def test_a_configuration_error_is_a_model_contract_error():
    """One failure type at the model boundary, so a caller catching
    `ModelContractError` cannot miss a configuration failure."""
    assert issubclass(ArchitectureConfigError, ModelContractError)


def test_the_contract_pins_the_shared_fields():
    for config in CANDIDATES.values():
        assert config.input_channels == TOKEN_FEATURES
        assert config.board_tokens == TOKEN_COUNT
        assert config.policy_size == POLICY_LOGIT_COUNT == TOKEN_COUNT * TOKEN_COUNT
        assert config.value_classes == VALUE_CLASS_COUNT
        assert config.belief_classes == BELIEF_TYPE_COUNT
        assert config.architecture_family_version == ARCHITECTURE_FAMILY_VERSION
        assert config.dropout == 0.0


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def test_digests_are_stable_and_distinct():
    first = config_digests()
    assert first == config_digests()
    assert len(set(first.values())) == len(CANDIDATE_IDS)
    assert architecture_family_digest() == architecture_family_digest()


def test_the_digest_covers_the_family_constants():
    """A digest over the configuration alone would call two networks identical
    after someone changed the activation for everybody."""
    config = CANDIDATES["C1"]
    baseline = config.digest()
    identity = config.identity()
    assert identity["family"] == dict(FAMILY_CONSTANTS)
    assert identity["config"] == config.to_dict()
    # Simulated family change: the digest is computed over `identity()`, so a
    # different family constant must produce a different digest.
    import hashlib
    import json

    altered = {"config": config.to_dict(), "family": {**FAMILY_CONSTANTS, "activation": "relu"}}
    altered_digest = hashlib.sha256(
        json.dumps(altered, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert altered_digest != baseline


def test_head_count_is_invisible_to_shapes_but_not_to_the_digest():
    """The reason configuration equality, not shape compatibility, is the gate.

    `nn.MultiheadAttention` packs every head into one `(3D, D)` projection, so
    C2 (six heads) and a four-head variant of the same width have byte-identical
    state dicts. Only the configuration separates them.
    """
    six_heads = CANDIDATES["C2"]
    four_heads = six_heads.replace(heads=4)
    shapes = lambda model: {  # noqa: E731 -- a one-line local, used twice
        name: tuple(tensor.shape) for name, tensor in model.state_dict().items()
    }
    assert shapes(ProductionModel(six_heads)) == shapes(ProductionModel(four_heads))
    assert six_heads.digest() != four_heads.digest()


# ---------------------------------------------------------------------------
# Deterministic construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate_id", SMALL_CANDIDATES)
def test_the_same_seed_gives_a_bit_identical_state_dict(candidate_id):
    first = ProductionModel(candidate_id, seed=4242).state_dict()
    second = ProductionModel(candidate_id, seed=4242).state_dict()
    assert state_dict_digest(first) == state_dict_digest(second)
    for name, tensor in first.items():
        assert torch.equal(tensor, second[name]), name


@pytest.mark.parametrize("candidate_id", SMALL_CANDIDATES)
def test_a_different_seed_gives_different_weights(candidate_id):
    first = ProductionModel(candidate_id, seed=1).state_dict()
    second = ProductionModel(candidate_id, seed=2).state_dict()
    assert state_dict_digest(first) != state_dict_digest(second)
    # Not merely "some tensor differs": the projections that carry the input
    # must differ, or a partially-seeded initialization would pass.
    assert not torch.equal(first["input_projection.weight"], second["input_projection.weight"])
    assert not torch.equal(first["row_embedding"], second["row_embedding"])


def test_reset_parameters_returns_to_the_seeded_state():
    model = ProductionModel("C0", seed=7)
    before = state_dict_digest(model.state_dict())
    model.reset_parameters(seed=8)
    assert state_dict_digest(model.state_dict()) != before
    model.reset_parameters(seed=7)
    assert state_dict_digest(model.state_dict()) == before


def test_the_declared_family_seed_is_the_default():
    assert ProductionModel("C0").initialisation_seed == FAMILY_INITIALIZATION_SEED
    assert build_candidate_model("C0").initialisation_seed == FAMILY_INITIALIZATION_SEED


def test_parameter_counts_are_exact(candidate_models):
    for candidate_id, model in candidate_models.items():
        assert model.parameter_count() == EXPECTED_PARAMETERS[candidate_id]
        assert model.trainable_parameter_count() == EXPECTED_PARAMETERS[candidate_id]


def test_parameter_groups_account_for_every_parameter(candidate_models):
    for candidate_id, model in candidate_models.items():
        breakdown = model.parameter_breakdown()
        assert set(breakdown) == {"encoder", "policy_head", "value_head", "belief_head"}
        assert sum(breakdown.values()) == EXPECTED_PARAMETERS[candidate_id]
        assert all(count > 0 for count in breakdown.values())


def test_parameter_bytes_follow_precision(candidate_models):
    model = candidate_models["C1"]
    assert model.parameter_bytes(torch.float32) == 4 * model.parameter_count()
    assert model.parameter_bytes(torch.float16) == 2 * model.parameter_count()


def test_a_candidate_can_be_reconstructed_from_config_and_seed_alone():
    """Exactly what Agent 3 is promised: config + family version + seed."""
    original = build_candidate_model("C2", seed=99)
    payload = original.config.to_dict()
    rebuilt = build_candidate_model(CandidateConfig.from_dict(payload), seed=99)
    assert state_dict_digest(original.state_dict()) == state_dict_digest(rebuilt.state_dict())


# ---------------------------------------------------------------------------
# Forward pass and the contract
# ---------------------------------------------------------------------------


def test_every_candidate_produces_the_contract_shapes(candidate_models):
    tokens = benchmark_token_batch(3, seed=11)
    for candidate_id, model in candidate_models.items():
        with torch.no_grad():
            outputs = model(tokens)
        summary = validate_candidate_outputs(outputs, batch=3)
        assert summary["policy_shape"] == [3, POLICY_LOGIT_COUNT], candidate_id
        assert summary["value_shape"] == [3, VALUE_CLASS_COUNT], candidate_id
        assert summary["belief_shape"] == [3, TOKEN_COUNT, BELIEF_TYPE_COUNT], candidate_id
        assert summary["all_finite"], candidate_id
        assert summary["model_contract_version"] == MODEL_CONTRACT_VERSION
        assert summary["policy_action_frame"] == POLICY_ACTION_FRAME


def test_the_observation_path_and_the_token_path_agree():
    model = build_candidate_model("C0")
    observation = benchmark_observation_batch(2, seed=5)
    with torch.no_grad():
        from_observation = model.forward_observation(observation)
        from_tokens = model(benchmark_token_batch(2, seed=5))
    assert torch.equal(from_observation.policy_logits, from_tokens.policy_logits)


def test_evaluation_mode_forward_is_deterministic(candidate_models):
    tokens = benchmark_token_batch(2, seed=3)
    for candidate_id, model in candidate_models.items():
        with torch.no_grad():
            first = model(tokens)
            second = model(tokens)
        assert torch.equal(first.policy_logits, second.policy_logits), candidate_id
        assert torch.equal(first.value_logits, second.value_logits), candidate_id
        assert torch.equal(first.belief_logits, second.belief_logits), candidate_id


def test_two_separately_built_models_agree_bit_for_bit():
    tokens = benchmark_token_batch(2, seed=17)
    with torch.no_grad():
        first = build_candidate_model("C1")(tokens)
        second = build_candidate_model("C1")(tokens)
    assert torch.equal(first.policy_logits, second.policy_logits)


def test_a_wrong_input_shape_is_refused():
    model = build_candidate_model("C0")
    with pytest.raises(ModelContractError):
        model(torch.randn(2, TOKEN_COUNT, TOKEN_FEATURES + 1))
    with pytest.raises(ModelContractError):
        model(torch.randn(TOKEN_COUNT, TOKEN_FEATURES))


def test_the_benchmark_input_builder_is_deterministic_and_valid():
    first = benchmark_observation_batch(4, seed=2)
    second = benchmark_observation_batch(4, seed=2)
    assert torch.equal(first, second)
    assert tuple(first.shape) == (4, TOKEN_FEATURES, BOARD_ROWS, BOARD_COLUMNS)
    assert not torch.equal(first, benchmark_observation_batch(4, seed=3))
    with pytest.raises(ArchitectureConfigError):
        benchmark_observation_batch(0)


def test_output_validation_rejects_a_malformed_result():
    model = build_candidate_model("C0")
    with torch.no_grad():
        outputs = model(benchmark_token_batch(2, seed=1))
    with pytest.raises(ModelContractError):
        validate_candidate_outputs(outputs, batch=3)


# ---------------------------------------------------------------------------
# Position representation
# ---------------------------------------------------------------------------


def test_the_position_term_is_the_learned_row_plus_the_learned_column():
    model = ProductionModel("C0", seed=5)
    position = model.position_embedding()
    assert tuple(position.shape) == (TOKEN_COUNT, model.config.width)
    for square in range(TOKEN_COUNT):
        row, column = square_to_row_column(square)
        expected = model.row_embedding[row] + model.column_embedding[column]
        assert torch.allclose(position[square], expected, atol=0, rtol=0)


def test_the_row_and_column_indexing_is_normalized_row_major():
    """A transposed indexing would still produce a `[100, D]` tensor, so the
    mapping is pinned with distinguishable embeddings rather than by shape."""
    model = ProductionModel("C0", seed=5)
    with torch.no_grad():
        model.row_embedding.zero_()
        model.column_embedding.zero_()
        # Row `r` gets value `r`; column `c` gets value `100 * c`. Every square
        # then reads back its own `(row, column)` uniquely.
        for row in range(BOARD_ROWS):
            model.row_embedding[row].fill_(float(row))
        for column in range(BOARD_COLUMNS):
            model.column_embedding[column].fill_(100.0 * column)
    position = model.position_embedding()
    for square in range(TOKEN_COUNT):
        expected = float(square // BOARD_COLUMNS) + 100.0 * float(square % BOARD_COLUMNS)
        assert position[square, 0].item() == expected, square


def test_the_position_term_is_separable_not_per_square():
    """20 vectors, not 100: the point of a row/column scheme is that a step in
    one direction means the same thing everywhere on the board."""
    model = ProductionModel("C0")
    parameters = dict(model.named_parameters())
    assert tuple(parameters["row_embedding"].shape) == (BOARD_ROWS, model.config.width)
    assert tuple(parameters["column_embedding"].shape) == (BOARD_COLUMNS, model.config.width)
    assert not any(
        tuple(tensor.shape)[:1] == (TOKEN_COUNT,) and name.endswith("embedding")
        for name, tensor in parameters.items()
    )


def test_the_position_buffers_do_not_reach_a_checkpoint():
    """`token_rows` and `token_columns` are constants. A checkpoint carrying
    them would be claiming they are weights that could have been trained."""
    state_dict = ProductionModel("C0").state_dict()
    assert "token_rows" not in state_dict
    assert "token_columns" not in state_dict


# ---------------------------------------------------------------------------
# Policy head: the normalized source/destination frame
# ---------------------------------------------------------------------------


def test_the_policy_matrix_flattens_row_major_as_source_destination():
    model = build_candidate_model("C0")
    tokens = benchmark_token_batch(1, seed=21)
    with torch.no_grad():
        hidden = model.encode(tokens)
        query = model.policy_query(hidden)
        key = model.policy_key(hidden)
        expected = torch.matmul(query, key.transpose(1, 2)) * model._policy_scale
        expected = (
            expected
            + model.policy_source_bias.view(1, -1, 1)
            + model.policy_destination_bias.view(1, 1, -1)
        )
        logits = model(tokens).policy_logits
    assert torch.equal(logits.reshape(1, TOKEN_COUNT, TOKEN_COUNT), expected)


def test_the_source_bias_moves_exactly_one_source_row():
    """`action_id = 100 * source + destination`, in normalized squares."""
    model = build_candidate_model("C0")
    tokens = benchmark_token_batch(1, seed=22)
    with torch.no_grad():
        before = model(tokens).policy_logits[0].clone()
        model.policy_source_bias[7] += 5.0
        after = model(tokens).policy_logits[0]
    difference = after - before
    moved = torch.nonzero(difference.abs() > 1e-4).flatten().tolist()
    assert moved == list(range(700, 800))


def test_the_destination_bias_moves_exactly_one_destination_column():
    model = build_candidate_model("C0")
    tokens = benchmark_token_batch(1, seed=23)
    with torch.no_grad():
        before = model(tokens).policy_logits[0].clone()
        model.policy_destination_bias[3] += 5.0
        after = model(tokens).policy_logits[0]
    difference = after - before
    moved = torch.nonzero(difference.abs() > 1e-4).flatten().tolist()
    assert moved == [100 * source + 3 for source in range(TOKEN_COUNT)]


def test_the_network_never_converts_action_frames():
    """The conversion lives in `action_frame.py` and nowhere else. A network
    that converted internally would make the checkpoint's frame fields a lie."""
    import ast
    import inspect

    from stratego.model import production_model

    tree = ast.parse(inspect.getsource(production_model))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any("action_frame" in name for name in imported)
    assert not hasattr(production_model, "absolute_action_to_model")
    assert not hasattr(production_model, "model_action_to_absolute")


# ---------------------------------------------------------------------------
# Value and belief heads
# ---------------------------------------------------------------------------


def test_the_value_head_is_a_three_class_distribution():
    from stratego.model.contract import value_probabilities

    model = build_candidate_model("C1")
    with torch.no_grad():
        outputs = model(benchmark_token_batch(4, seed=31))
    probabilities = value_probabilities(outputs.value_logits)
    assert tuple(probabilities.shape) == (4, 3)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_the_belief_head_is_per_token_and_shares_the_encoder():
    model = build_candidate_model("C0")
    parameters = dict(model.named_parameters())
    # A lightweight per-token head: one linear layer, not a second decoder.
    assert tuple(parameters["belief_output.weight"].shape) == (BELIEF_TYPE_COUNT, 64)
    with torch.no_grad():
        outputs = model(benchmark_token_batch(2, seed=32))
    assert tuple(outputs.belief_logits.shape) == (2, TOKEN_COUNT, BELIEF_TYPE_COUNT)


# ---------------------------------------------------------------------------
# No privileged inputs
# ---------------------------------------------------------------------------


def test_the_forward_pass_takes_tokens_and_nothing_else():
    import inspect

    signature = inspect.signature(ProductionModel.forward)
    assert list(signature.parameters) == ["self", "tokens"]
    observation_signature = inspect.signature(ProductionModel.forward_observation)
    assert list(observation_signature.parameters) == ["self", "observation"]


def test_the_family_imports_no_privileged_engine_product():
    """The same object-graph rule Phase 5 established, restated for the new
    modules: no `GameState`, no piece records, no belief targets."""
    import inspect

    from stratego.model import architecture_configs, base, production_model

    for module in (production_model, architecture_configs, base):
        source = inspect.getsource(module)
        for forbidden in ("GameState", "PieceRecord", "belief_target", "hidden_identit"):
            assert forbidden not in source, f"{module.__name__} mentions {forbidden}"


def test_no_absolute_colour_feature_exists():
    """Agent 1's symmetry decision: the network sees normalized squares only,
    so nothing in it may key off which colour is acting."""
    model = ProductionModel("C0")
    names = " ".join(name for name, _ in model.named_parameters()).lower()
    for forbidden in ("red", "blue", "colour", "color", "player"):
        assert forbidden not in names


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------


def test_the_family_default_has_no_dropout():
    for config in CANDIDATES.values():
        assert config.dropout == 0.0


def test_dropout_is_disabled_in_evaluation_mode():
    """If a later agent ever turns dropout on, benchmark and evaluation mode
    must still be deterministic. Proven both ways: identical in `eval`,
    different in `train`."""
    config = CANDIDATES["C0"].replace(dropout=0.5)
    model = ProductionModel(config, seed=3)
    tokens = benchmark_token_batch(2, seed=41)

    model.eval()
    with torch.no_grad():
        assert torch.equal(model(tokens).policy_logits, model(tokens).policy_logits)

    model.train()
    torch.manual_seed(0)
    with torch.no_grad():
        first = model(tokens).policy_logits.clone()
        second = model(tokens).policy_logits
    assert not torch.equal(first, second)


def test_dropout_does_not_change_the_parameter_set():
    """A checkpoint written at one dropout value must load at another: dropout
    is a training-time policy, not a change of architecture."""
    zero = ProductionModel(CANDIDATES["C0"], seed=1).state_dict()
    half = ProductionModel(CANDIDATES["C0"].replace(dropout=0.5), seed=1).state_dict()
    assert set(zero) == set(half)
    assert state_dict_digest(zero) == state_dict_digest(half)


# ---------------------------------------------------------------------------
# Backward connectivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate_id", SMALL_CANDIDATES)
def test_every_parameter_receives_a_finite_gradient(candidate_id):
    """Connectivity only. Phase 6 authorises a backward pass to measure compute
    and to prove the graph is connected -- not to train anything."""
    model = ProductionModel(candidate_id, seed=2)
    model.train()
    outputs = model(benchmark_token_batch(2, seed=51))
    loss = (
        outputs.policy_logits.square().mean()
        + outputs.value_logits.square().mean()
        + outputs.belief_logits.square().mean()
    )
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum().item() > 0.0, name


def test_the_phase_5_multi_head_loss_still_applies():
    """The heads did not change shape, so the existing loss module must work
    against a candidate unmodified."""
    model = ProductionModel("C0", seed=2)
    model.train()
    outputs = model(benchmark_token_batch(2, seed=52))
    legal_mask = torch.zeros(2, POLICY_LOGIT_COUNT, dtype=torch.bool)
    legal_mask[:, :8] = True
    belief_mask = torch.zeros(2, TOKEN_COUNT, dtype=torch.bool)
    belief_mask[:, :4] = True
    # The belief head is supervised on unresolved hidden squares only, so the
    # labels are the ignore index everywhere the mask is false.
    belief_labels = torch.full((2, TOKEN_COUNT), BELIEF_IGNORE_INDEX, dtype=torch.long)
    belief_labels[:, :4] = 0
    loss = multi_head_loss(
        outputs,
        target_actions=torch.zeros(2, dtype=torch.long),
        legal_mask=legal_mask,
        target_value_classes=torch.zeros(2, dtype=torch.long),
        belief_labels=belief_labels,
        belief_mask=belief_mask,
    )
    loss.total.backward()
    assert torch.isfinite(loss.total)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_a_candidate_checkpoint_round_trips(tmp_path):
    model = build_candidate_model("C1", seed=123)
    path = save_checkpoint(model, tmp_path / "c1.pt", training_iteration=0, training_step=0)
    restored, metadata = load_checkpoint(path)

    assert metadata["model_architecture_id"] == ARCHITECTURE_FAMILY
    assert metadata["model_contract_version"] == MODEL_CONTRACT_VERSION
    assert metadata["model_configuration"] == model.config.to_dict()
    assert metadata["policy_action_frame"] == POLICY_ACTION_FRAME
    assert isinstance(restored, ProductionModel)
    assert state_dict_digest(restored.state_dict()) == state_dict_digest(model.state_dict())

    tokens = benchmark_token_batch(2, seed=61)
    with torch.no_grad():
        assert torch.equal(model(tokens).policy_logits, restored(tokens).policy_logits)


def test_a_checkpoint_states_the_candidate_it_holds(tmp_path):
    model = build_candidate_model("C3")
    payload = build_checkpoint_payload(model)
    assert payload["model_configuration"]["candidate_id"] == "C3"
    assert payload["provenance"]["parameter_count"] == EXPECTED_PARAMETERS["C3"]
    assert payload["provenance"]["initialisation_seed"] == FAMILY_INITIALIZATION_SEED
    assert validate_checkpoint_payload(payload)["model_architecture_id"] == ARCHITECTURE_FAMILY


def test_one_candidate_cannot_be_loaded_as_another(tmp_path):
    """The gate the whole registry exists for. C2 and C3 differ in depth, so
    this one is caught by shapes; the next test covers the harder case."""
    path = save_checkpoint(build_candidate_model("C2"), tmp_path / "c2.pt")
    target = build_candidate_model("C3")
    with pytest.raises(CheckpointCompatibilityError):
        load_checkpoint_into(target, path)


def test_a_shape_compatible_but_different_candidate_is_refused(tmp_path):
    """Same width, same depth, same feed-forward, different head count: every
    tensor shape agrees and `load_state_dict` would succeed silently."""
    six_heads = build_candidate_model(CANDIDATES["C2"].replace(candidate_id="C2-six"))
    four_heads = build_candidate_model(
        CANDIDATES["C2"].replace(candidate_id="C2-four", heads=4)
    )
    assert {name: tuple(t.shape) for name, t in six_heads.state_dict().items()} == {
        name: tuple(t.shape) for name, t in four_heads.state_dict().items()
    }
    path = save_checkpoint(six_heads, tmp_path / "six.pt")
    with pytest.raises(CheckpointCompatibilityError, match="configuration does not match"):
        load_checkpoint_into(four_heads, path)


def test_a_checkpoint_that_misstates_its_candidate_is_refused(tmp_path):
    """A file claiming to be C3 while carrying C2's dimensions would attribute
    C2's measurements to C3 in every downstream report."""
    payload = build_checkpoint_payload(build_candidate_model("C2"))
    payload["model_configuration"] = dict(payload["model_configuration"], candidate_id="C3")
    with pytest.raises(CheckpointCompatibilityError, match="claims candidate"):
        validate_checkpoint_payload(payload)


def test_the_expected_identity_arguments_are_honoured(tmp_path):
    path = save_checkpoint(build_candidate_model("C0"), tmp_path / "c0.pt")
    load_checkpoint(
        path,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=CANDIDATES["C0"],
    )
    with pytest.raises(CheckpointCompatibilityError, match="expected"):
        load_checkpoint(path, expected_architecture_id=MODEL_ARCHITECTURE_ID)
    with pytest.raises(CheckpointCompatibilityError, match="configuration does not match"):
        load_checkpoint(path, expected_configuration=CANDIDATES["C1"])


def test_a_fixture_checkpoint_cannot_be_loaded_as_a_candidate(tmp_path):
    """`integration_model_v1` is Phase 5 scaffolding and must never be mistaken
    for a Phase 6 candidate, in either direction."""
    fixture_path = save_checkpoint(build_integration_model(), tmp_path / "fixture.pt")
    with pytest.raises(CheckpointCompatibilityError):
        load_checkpoint_into(build_candidate_model("C0"), fixture_path)

    candidate_path = save_checkpoint(build_candidate_model("C0"), tmp_path / "candidate.pt")
    with pytest.raises(CheckpointCompatibilityError):
        load_checkpoint_into(build_integration_model(), candidate_path)


def test_an_unregistered_architecture_is_refused():
    payload = build_checkpoint_payload(build_candidate_model("C0"))
    payload["model_architecture_id"] = "ataraxos_full_v1"
    with pytest.raises(CheckpointCompatibilityError, match="unknown model_architecture_id"):
        validate_checkpoint_payload(payload)


def test_the_registry_describes_both_shipped_architectures():
    family = architecture_registration(ARCHITECTURE_FAMILY)
    assert family.model_class is ProductionModel
    assert family.check_configuration is not None
    fixture = architecture_registration(MODEL_ARCHITECTURE_ID)
    assert fixture.model_class.__name__ == "IntegrationModel"


def test_registering_a_duplicate_architecture_id_is_refused():
    from stratego.model.checkpoint import ArchitectureRegistration, CheckpointError
    from stratego.model.checkpoint import register_architecture

    duplicate = ArchitectureRegistration(
        architecture_id=ARCHITECTURE_FAMILY,
        model_class=ProductionModel,
        config_from_dict=CandidateConfig.from_dict,
        build=ProductionModel,
    )
    with pytest.raises(CheckpointError, match="already registered"):
        register_architecture(duplicate)


# ---------------------------------------------------------------------------
# The evaluation boundary
# ---------------------------------------------------------------------------


def test_a_candidate_drives_the_existing_policy_adapter(tmp_path):
    """Agent 5's requirement, checked now: a candidate must reach a decision
    through the *existing* normalized decision path, with no second adapter."""
    path = save_checkpoint(build_candidate_model("C0"), tmp_path / "c0.pt")
    policy = GreedyNeuralPolicy.from_checkpoint(path)
    assert isinstance(policy.model, StrategoModel)

    state = nonterminal_state(40)
    request = build_policy_input(
        state,
        policy=policy.ref,
        policy_seed=3,
        requirements=PolicyRequirements(observation=True, legal_action_mask=True),
    )
    result = policy.decide(request)
    from stratego.engine.legal_moves import legal_actions

    # The engine remains the legality authority: the returned identifier is an
    # absolute engine action, converted back out of the model's frame.
    assert result.selected_action_id in set(legal_actions(state))
    assert result.diagnostics["model_architecture_id"] == ARCHITECTURE_FAMILY
    assert result.diagnostics["policy_action_frame"] == POLICY_ACTION_FRAME


def test_the_adapter_still_refuses_a_non_model():
    from stratego.model.policy_adapter import NeuralPolicyError

    with pytest.raises(NeuralPolicyError, match="StrategoModel"):
        GreedyNeuralPolicy(torch.nn.Linear(3, 3))


# ---------------------------------------------------------------------------
# Metal
# ---------------------------------------------------------------------------


@requires_mps
@pytest.mark.parametrize("candidate_id", SMALL_CANDIDATES)
def test_a_candidate_constructs_and_runs_on_metal_in_float32(candidate_id):
    model = build_candidate_model(candidate_id, device="mps")
    tokens = benchmark_token_batch(2, seed=71, device="mps")
    with torch.no_grad():
        outputs = model(tokens)
    torch.mps.synchronize()
    summary = validate_candidate_outputs(outputs, batch=2)
    assert summary["device"].startswith("mps")
    assert summary["all_finite"]


@requires_mps
@pytest.mark.parametrize("candidate_id", SMALL_CANDIDATES)
def test_a_candidate_constructs_and_runs_on_metal_in_float16(candidate_id):
    model = build_candidate_model(candidate_id, device="mps", dtype=torch.float16)
    tokens = benchmark_token_batch(2, seed=72, device="mps", dtype=torch.float16)
    with torch.no_grad():
        outputs = model(tokens)
    torch.mps.synchronize()
    summary = validate_candidate_outputs(outputs, batch=2)
    assert summary["dtype"] == "torch.float16"
    assert summary["all_finite"]


@requires_mps
def test_metal_and_cpu_agree_to_float32_tolerance():
    """Not bit-equality -- different kernels legitimately reassociate -- but a
    disagreement beyond this is a numerical problem, not a rounding difference."""
    cpu_model = build_candidate_model("C0")
    mps_model = build_candidate_model("C0", device="mps")
    tokens = benchmark_token_batch(2, seed=73)
    with torch.no_grad():
        cpu_outputs = cpu_model(tokens)
        mps_outputs = mps_model(tokens.to("mps"))
    torch.mps.synchronize()
    assert torch.allclose(
        cpu_outputs.policy_logits, mps_outputs.policy_logits.cpu(), atol=1e-4, rtol=1e-4
    )
    assert torch.allclose(
        cpu_outputs.value_logits, mps_outputs.value_logits.cpu(), atol=1e-4, rtol=1e-4
    )


# ---------------------------------------------------------------------------
# The reported summary
# ---------------------------------------------------------------------------


def test_the_family_summary_is_complete_and_serializable():
    import json

    summary = family_summary()
    assert summary["architecture_family"] == ARCHITECTURE_FAMILY
    assert summary["architecture_family_version"] == ARCHITECTURE_FAMILY_VERSION
    assert summary["model_contract_version"] == MODEL_CONTRACT_VERSION
    assert summary["initialization_seed"] == FAMILY_INITIALIZATION_SEED
    assert list(summary["candidate_ids"]) == list(CANDIDATE_IDS)
    assert summary["ladder_adjustments"] == []
    json.dumps(summary)


def test_the_architecture_summary_states_it_is_untrained(candidate_models):
    for candidate_id, model in candidate_models.items():
        summary = model.architecture_summary()
        assert summary["candidate_id"] == candidate_id
        assert summary["trained"] is False
        assert summary["integration_fixture"] is False
        assert summary["config_digest"] == CANDIDATES[candidate_id].digest()
        assert summary["parameter_count"] == EXPECTED_PARAMETERS[candidate_id]
        assert "untrained" in summary["note"].lower()
