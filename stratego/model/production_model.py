"""`stratego_transformer_v1` -- the configurable Phase 6 candidate network.

Specification source: Phase 6 Agent 2 instructions ("Fixed high-level
architecture", "Position representation", "Transformer blocks", the three head
sections) and the shapes frozen by `model_contract_v2`.

One class, seven candidates
---------------------------
Every C0-C6 candidate is *this* module built from a different
:class:`~stratego.model.architecture_configs.CandidateConfig`. There are no
per-candidate branches anywhere below -- not one `if config.width == ...` --
because Agent 3's benchmark is only meaningful if the sole difference between
two candidates is four integers.

.. code-block:: text

    observation [B, 127, 10, 10]
      -> tokens [B, 100, 127]                       pure relayout, row-major
      -> input projection                           [B, 100, D]
      -> + learned row embedding + learned column embedding
      -> `blocks` x pre-norm Transformer block      no mask of any kind
      -> final LayerNorm                            [B, 100, D]  shared encoder
         |- policy: source query . destination key  [B, 100, 100] -> [B, 10000]
         |- value:  mean pool -> MLP                [B, 3]   WIN, DRAW, LOSS
         |- belief: per-token linear                [B, 100, 12]

Frames
------
Tokens and policy logits are both in the **acting player's normalized** frame,
which is the whole content of `model_contract_v2`. This module therefore never
imports :mod:`stratego.model.action_frame` and never sees an absolute engine
action: policy entry `100 * source + destination` names normalized squares, and
converting that to the engine's frame is the adapter's job at the boundary. A
conversion inside the network would put the same operation in two places and
would be invisible to the checkpoint's frame fields.

Nothing here is trained. Random weights carry no strategic meaning, and the
Phase 6 rules forbid using random-weight playing strength as evidence for
anything.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn

from ..engine.constants import BOARD_COLUMNS, BOARD_ROWS
from .architecture_configs import (
    ACTIVATION,
    ARCHITECTURE_FAMILY,
    ARCHITECTURE_FAMILY_VERSION,
    ATTENTION_IMPLEMENTATION,
    BELIEF_HEAD,
    CANDIDATE_IDS,
    CANDIDATE_ROLES,
    FAMILY_CONSTANTS,
    FAMILY_INITIALIZATION_SEED,
    LAYER_NORM_EPS,
    POLICY_HEAD,
    VALUE_HEAD,
    ArchitectureConfigError,
    CandidateConfig,
    candidate_config,
)
from .base import StrategoModel
from .contract import (
    MODEL_CONTRACT_VERSION,
    POLICY_ACTION_FRAME,
    ModelOutputs,
    validate_belief_logits,
    validate_observation_batch,
    validate_policy_logits,
    validate_token_batch,
    validate_value_logits,
)
from .tokenization import observation_to_tokens

#: Groups used for the per-head parameter accounting Agent 3 needs. A parameter
#: name is assigned to the first group whose prefix it starts with, so the order
#: matters and every parameter must land in exactly one group (a test asserts
#: the groups sum to the total).
_PARAMETER_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "policy_head",
        ("policy_query.", "policy_key.", "policy_source_bias", "policy_destination_bias"),
    ),
    ("value_head", ("value_body.", "value_output.")),
    ("belief_head", ("belief_output.",)),
    (
        "encoder",
        ("input_projection.", "row_embedding", "column_embedding", "blocks.", "encoder_norm."),
    ),
)


class _EncoderBlock(nn.Module):
    """One pre-normalization block: norm -> attention -> residual, then again.

    Pre-norm (normalize *before* each sublayer rather than after the residual)
    is what lets the deeper candidates -- C5 and C6 at eight blocks -- be
    constructed and run without a warmup schedule, and it keeps activations
    well-scaled at initialization, which is what makes the float16 smoke check
    on Metal a test of the kernels rather than of the initialization.

    Identical for every candidate by instruction: same activation, same epsilon,
    same biases, same dropout policy, same attention implementation.
    """

    def __init__(self, config: CandidateConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width, eps=LAYER_NORM_EPS)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.width,
            num_heads=config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(config.width, eps=LAYER_NORM_EPS)
        self.feedforward = nn.Sequential(
            nn.Linear(config.width, config.feed_forward_width),
            nn.GELU(),
            nn.Linear(config.feed_forward_width, config.width),
        )
        # Residual dropout on both sublayers. `nn.Dropout(0.0)` is the identity,
        # so the default family configuration has no randomness to disable --
        # but the modules exist unconditionally, because a candidate whose
        # *module list* changed with `dropout` would not round-trip through a
        # checkpoint written at a different dropout value.
        self.attention_dropout = nn.Dropout(config.dropout)
        self.feedforward_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normed = self.attention_norm(hidden)
        # Every square attends to every other square. No causal mask and no
        # padding mask: a board position is not a sequence, and all 100 tokens
        # are always present.
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        hidden = hidden + self.attention_dropout(attended)
        return hidden + self.feedforward_dropout(self.feedforward(self.feedforward_norm(hidden)))


class ProductionModel(StrategoModel):
    """A Phase 6 candidate network, fully described by its `CandidateConfig`."""

    architecture_id = ARCHITECTURE_FAMILY
    is_integration_fixture = False

    def __init__(
        self,
        config: "CandidateConfig | str",
        *,
        seed: int = FAMILY_INITIALIZATION_SEED,
    ):
        super().__init__()
        if isinstance(config, str):
            config = candidate_config(config)
        if not isinstance(config, CandidateConfig):
            raise ArchitectureConfigError(
                f"expected a CandidateConfig or a candidate id, got {type(config).__name__}"
            )
        self.config = config
        self._init_seed = int(seed)
        width = config.width

        self.input_projection = nn.Linear(config.input_channels, width)

        # h[r,c] = W_x x[r,c] + e_row[r] + e_col[c]. Ten row vectors and ten
        # column vectors rather than one hundred square vectors: the board is a
        # grid, and separable embeddings let "one row forward" mean the same
        # thing everywhere instead of being relearned per square.
        self.row_embedding = nn.Parameter(torch.zeros(BOARD_ROWS, width))
        self.column_embedding = nn.Parameter(torch.zeros(BOARD_COLUMNS, width))
        # Token `i` is normalized row-major square `i` (see `tokenization.py`),
        # so its row is `i // 10` and its column is `i % 10`. Held as buffers so
        # they follow the model across devices; non-persistent so they never
        # appear in a checkpoint, where they would be constants pretending to be
        # weights.
        squares = torch.arange(config.board_tokens, dtype=torch.long)
        self.register_buffer("token_rows", squares // BOARD_COLUMNS, persistent=False)
        self.register_buffer("token_columns", squares % BOARD_COLUMNS, persistent=False)

        self.blocks = nn.ModuleList(_EncoderBlock(config) for _ in range(config.blocks))
        self.encoder_norm = nn.LayerNorm(width, eps=LAYER_NORM_EPS)

        # Policy: every square emits a "move from here" query and a "move to
        # here" key; their scaled dot product scores one (source, destination)
        # pair. The two bias vectors give the head a way to express "this square
        # is worth moving from" and "this square is worth moving to" without
        # spending attention capacity on it.
        self.policy_query = nn.Linear(width, width)
        self.policy_key = nn.Linear(width, width)
        self.policy_source_bias = nn.Parameter(torch.zeros(config.board_tokens))
        self.policy_destination_bias = nn.Parameter(torch.zeros(config.board_tokens))
        self._policy_scale = 1.0 / math.sqrt(width)

        # Value: pooled position -> WIN / DRAW / LOSS, for the acting player.
        self.value_body = nn.Linear(width, width)
        self.value_output = nn.Linear(width, config.value_classes)

        # Belief: one 12-way distribution per square, sharing the encoder. Phase
        # 6 deliberately does not build the paper's separate belief decoder.
        self.belief_output = nn.Linear(width, config.belief_classes)

        self.reset_parameters(seed=self._init_seed)
        # Deterministic inference is the default state. With the family's zero
        # dropout there is no randomness to disable, but a benchmark that
        # measured a training-mode forward pass would not be measuring what
        # Agent 5 runs.
        self.eval()

    # -- construction ------------------------------------------------------

    def reset_parameters(self, *, seed: int | None = None) -> None:
        """Deterministic initialization from an explicit CPU generator.

        An explicit `torch.Generator` rather than the global RNG: two processes
        then produce bit-identical weights without agreeing about global seeding,
        which is what makes "same seed -> same state dict" a property of the
        model rather than of the caller.
        """
        if seed is not None:
            self._init_seed = int(seed)
        generator = torch.Generator(device="cpu").manual_seed(self._init_seed)
        # `nn.MultiheadAttention` owns an `out_proj` that is itself an
        # `nn.Linear`, so it would otherwise be initialized twice -- once by the
        # attention branch and once when `modules()` reaches the child. Drawing
        # from the generator twice for one tensor is not wrong, but it makes the
        # sequence depend on a PyTorch internal, so the child is skipped.
        attention_children: set[int] = set()
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.MultiheadAttention):
                    attention_children.add(id(module.out_proj))
                    bound = 1.0 / math.sqrt(module.embed_dim)
                    module.in_proj_weight.uniform_(-bound, bound, generator=generator)
                    if module.in_proj_bias is not None:
                        module.in_proj_bias.zero_()
                    module.out_proj.weight.uniform_(-bound, bound, generator=generator)
                    if module.out_proj.bias is not None:
                        module.out_proj.bias.zero_()
                elif isinstance(module, nn.Linear) and id(module) not in attention_children:
                    bound = 1.0 / math.sqrt(module.in_features)
                    module.weight.uniform_(-bound, bound, generator=generator)
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.weight.fill_(1.0)
                    module.bias.zero_()
            self.row_embedding.normal_(mean=0.0, std=0.02, generator=generator)
            self.column_embedding.normal_(mean=0.0, std=0.02, generator=generator)
            # Zero, so an untrained model expresses no square preference at all.
            self.policy_source_bias.zero_()
            self.policy_destination_bias.zero_()

    @property
    def initialisation_seed(self) -> int:
        return self._init_seed

    @property
    def candidate_id(self) -> str:
        return self.config.candidate_id

    def parameter_breakdown(self) -> dict[str, int]:
        """Trainable parameters per component, summing to the model total."""
        totals = {name: 0 for name, _prefixes in _PARAMETER_GROUPS}
        for name, parameter in self.named_parameters():
            for group, prefixes in _PARAMETER_GROUPS:
                if name.startswith(prefixes):
                    totals[group] += parameter.numel()
                    break
            else:  # pragma: no cover -- guarded by test_parameter_groups_are_total
                raise ArchitectureConfigError(
                    f"parameter {name!r} belongs to no accounting group; the parameter "
                    "breakdown would silently under-count"
                )
        return totals

    def parameter_bytes(self, dtype: torch.dtype = torch.float32) -> int:
        """Parameter bytes at a given precision, independent of current dtype."""
        element = torch.empty((), dtype=dtype).element_size()
        return self.parameter_count() * element

    def architecture_summary(self) -> dict:
        """Serializable description, carried into checkpoints and reports."""
        return {
            "architecture_family": ARCHITECTURE_FAMILY,
            "architecture_family_version": ARCHITECTURE_FAMILY_VERSION,
            "model_architecture_id": self.architecture_id,
            "model_contract_version": MODEL_CONTRACT_VERSION,
            "candidate_id": self.config.candidate_id,
            "role": CANDIDATE_ROLES.get(self.config.candidate_id, "custom"),
            "config": self.config.to_dict(),
            "config_digest": self.config.digest(),
            "family_constants": dict(FAMILY_CONSTANTS),
            "parameter_count": self.parameter_count(),
            "trainable_parameter_count": self.trainable_parameter_count(),
            "parameter_breakdown": self.parameter_breakdown(),
            "parameter_bytes_float32": self.parameter_bytes(torch.float32),
            "parameter_bytes_float16": self.parameter_bytes(torch.float16),
            "initialisation_seed": self._init_seed,
            "integration_fixture": False,
            "trained": False,
            "policy_head": POLICY_HEAD,
            "value_head": VALUE_HEAD,
            "belief_head": BELIEF_HEAD,
            "activation": ACTIVATION,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "note": (
                "Phase 6 candidate architecture. Untrained: its weights carry no "
                "strategic meaning and its playing strength is not evidence of anything."
            ),
        }

    # -- forward -----------------------------------------------------------

    def position_embedding(self) -> torch.Tensor:
        """`[100, D]` learned position term, `e_row[r] + e_col[c]` per square."""
        rows = torch.index_select(self.row_embedding, 0, self.token_rows)
        columns = torch.index_select(self.column_embedding, 0, self.token_columns)
        return rows + columns

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """`[B, 100, 127]` -> `[B, 100, D]` shared board representation."""
        validate_token_batch(tokens)
        weight_dtype = self.input_projection.weight.dtype
        if tokens.dtype != weight_dtype:
            # Casting rather than raising keeps a float16 model callable with a
            # float32 input, which is how the precision smoke checks feed it.
            tokens = tokens.to(weight_dtype)
        hidden = self.input_projection(tokens) + self.position_embedding().unsqueeze(0)
        for block in self.blocks:
            hidden = block(hidden)
        return self.encoder_norm(hidden)

    def forward(self, tokens: torch.Tensor) -> ModelOutputs:
        """One forward pass over tokenized observations. Returns validated heads."""
        hidden = self.encode(tokens)
        batch = hidden.shape[0]

        # L[i,j] = (Q_i . K_j) / sqrt(D) + b_source[i] + b_destination[j].
        query = self.policy_query(hidden)
        key = self.policy_key(hidden)
        scores = torch.matmul(query, key.transpose(1, 2)) * self._policy_scale
        scores = (
            scores
            + self.policy_source_bias.to(scores.dtype).view(1, -1, 1)
            + self.policy_destination_bias.to(scores.dtype).view(1, 1, -1)
        )
        # Row-major flatten: entry `(source, destination)` lands at index
        # `100 * source + destination`, which is the frozen action encoding read
        # in normalized squares. `reshape` -- not `view` -- because `scores` is
        # the output of an add and need not be contiguous on every backend.
        policy_logits = scores.reshape(batch, self.config.policy_size)

        pooled = hidden.mean(dim=1)
        value_logits = self.value_output(torch.nn.functional.gelu(self.value_body(pooled)))

        belief_logits = self.belief_output(hidden)

        return ModelOutputs.validated(policy_logits, value_logits, belief_logits, batch=batch)

    def forward_observation(self, observation: torch.Tensor) -> ModelOutputs:
        """Convenience path from the canonical `[B, 127, 10, 10]` input."""
        return self.forward(observation_to_tokens(observation))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_candidate_model(
    candidate: "CandidateConfig | str",
    *,
    seed: int = FAMILY_INITIALIZATION_SEED,
    device: "torch.device | str" = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ProductionModel:
    """Build a candidate on CPU with fixed weights, then move and cast.

    Building on CPU first is what makes a CPU/Metal comparison meaningful: both
    devices start from bit-identical float32 weights, so any difference measured
    later comes from the kernels rather than from initialization. It is also why
    `(config, seed)` alone is enough for Agent 3 to reconstruct any candidate.
    """
    model = ProductionModel(candidate, seed=seed)
    model = model.to(device=torch.device(device), dtype=dtype)
    model.eval()
    return model


def build_all_candidates(
    *, seed: int = FAMILY_INITIALIZATION_SEED
) -> dict[str, ProductionModel]:
    """Every C0-C6 candidate on CPU in float32, smallest first."""
    return {
        candidate_id: build_candidate_model(candidate_id, seed=seed)
        for candidate_id in CANDIDATE_IDS
    }


# ---------------------------------------------------------------------------
# Benchmark input and output validation (the Agent 3 surface)
# ---------------------------------------------------------------------------


def benchmark_observation_batch(
    batch: int = 1,
    *,
    seed: int = 0,
    device: "torch.device | str" = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """A deterministic, contract-valid `[B, 127, 10, 10]` benchmark input.

    Reproducible from `(batch, seed)` alone and validated against the canonical
    input boundary before it is returned.

    It is a *shape*-valid input, not a real board: the values are pseudo-random,
    and throughput does not depend on them, so this is the right input for
    timing. Anything that depends on the values -- legality, decisions, hidden
    information -- must use real engine positions instead
    (`stratego.training.mps_benchmark.build_position_pool`).
    """
    if batch < 1:
        raise ArchitectureConfigError(f"batch must be at least 1, got {batch}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    observation = torch.randn(
        batch, 127, BOARD_ROWS, BOARD_COLUMNS, generator=generator, dtype=torch.float32
    )
    observation = observation.to(device=torch.device(device), dtype=dtype)
    validate_observation_batch(observation)
    return observation


def benchmark_token_batch(
    batch: int = 1,
    *,
    seed: int = 0,
    device: "torch.device | str" = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The same input as :func:`benchmark_observation_batch`, tokenized."""
    return observation_to_tokens(
        benchmark_observation_batch(batch, seed=seed, device=device, dtype=dtype)
    )


def validate_candidate_outputs(
    outputs: ModelOutputs, *, batch: int | None = None, require_finite: bool = True
) -> dict:
    """Check one forward pass against `model_contract_v2`; return a summary.

    Re-validates rather than trusting :meth:`ModelOutputs.validated`, because
    this is the function Agent 3 calls on results that may have crossed a device
    or a precision boundary since they were built. Raises
    :class:`~stratego.model.contract.ModelContractError` on any violation; the
    returned dictionary is evidence for a report, not a status code.
    """
    resolved = validate_policy_logits(
        outputs.policy_logits, batch=batch, require_finite=require_finite
    )
    validate_value_logits(outputs.value_logits, batch=resolved, require_finite=require_finite)
    validate_belief_logits(outputs.belief_logits, batch=resolved, require_finite=require_finite)
    return {
        "batch": resolved,
        "policy_shape": list(outputs.policy_logits.shape),
        "value_shape": list(outputs.value_logits.shape),
        "belief_shape": list(outputs.belief_logits.shape),
        "device": str(outputs.policy_logits.device),
        "dtype": str(outputs.policy_logits.dtype),
        "all_finite": bool(outputs.all_finite()),
        "policy_action_frame": POLICY_ACTION_FRAME,
        "model_contract_version": MODEL_CONTRACT_VERSION,
    }


def candidate_state_dict(
    candidate: "CandidateConfig | str", *, seed: int = FAMILY_INITIALIZATION_SEED
) -> Mapping[str, torch.Tensor]:
    """The CPU float32 initial state dict for a candidate. Used by the
    determinism checks, which compare two of these byte for byte."""
    return ProductionModel(candidate, seed=seed).state_dict()


def config_from_checkpoint_payload(payload: Mapping[str, Any]) -> CandidateConfig:
    """The candidate configuration a checkpoint payload describes."""
    return CandidateConfig.from_dict(payload["model_configuration"])


__all__ = [
    "ProductionModel",
    "benchmark_observation_batch",
    "benchmark_token_batch",
    "build_all_candidates",
    "build_candidate_model",
    "candidate_state_dict",
    "config_from_checkpoint_payload",
    "validate_candidate_outputs",
]
