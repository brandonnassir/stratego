"""The Phase 6 candidate architecture family: one config, seven candidates.

Specification source: Phase 6 Agent 2 instructions ("Candidate Architecture
Family"), and the common contract in
`instructions/phase_6_sequential_agent_instructions/00_PHASE_6_SEQUENCE_AND_COMMON_CONTRACT.md`.

What this module is
-------------------
A serializable description of *shape*, and nothing else. It holds no tensors, no
weights and no PyTorch modules; :mod:`stratego.model.production_model` turns one
of these configurations into a network. The split exists so a candidate can be
named, digested, written into a checkpoint and compared for identity without
constructing 14 million parameters first.

One family, seven candidates
----------------------------
The instruction is explicit that this must be *one configurable family* rather
than seven hand-edited model classes: the whole point of Agent 3's benchmark is
that the only thing that differs between C0 and C6 is four integers. Anything
else that differed -- an activation, a normalization epsilon, a bias policy --
would make the measured compute/capacity frontier a comparison of architectures
instead of a comparison of sizes. So everything that is *not* one of those four
integers lives here as a module-level family constant, is asserted identical for
every candidate, and is folded into the architecture digest.

============  =====  ======  =====  =============  =================================
Candidate     Width  Blocks  Heads  Feed-forward   Role
============  =====  ======  =====  =============  =================================
C0               64       2      4            256  small control
C1              128       4      4            512  small practical
C2              192       4      6            768  wider
C3              192       6      6            768  deeper
C4              256       6      8          1,024  medium-large
C5              256       8      8          1,024  deeper medium-large
C6              384       8      8          1,536  paper-width/depth ceiling reference
============  =====  ======  =====  =============  =================================

C6 is an upper-region benchmark reference, not a presumed choice. Every row is
the literal instruction table: each width divides evenly across its head count
(16, 32, 32, 32, 32, 32, 48 channels per head), so no row violated a hard
PyTorch constraint and **no adjustment to the ladder was necessary**.

Identity
--------
Two things are deliberately distinct:

``ARCHITECTURE_FAMILY``
    the checkpoint's `model_architecture_id`, shared by every candidate, and
    distinct from the Phase 5 fixture's `integration_model_v1`.
``candidate_id``
    a field *inside* the serialized configuration.

That split is what makes a cross-candidate load fail. C2 (192/4/6/768) and a
hypothetical 192/4/**4**/768 produce byte-identical tensor shapes -- head count
never appears in a parameter shape, because `nn.MultiheadAttention` packs all
heads into one `(3D, D)` projection. A shape check cannot separate them; the
configuration can, so configuration equality (not shape compatibility) is the
gate in :mod:`stratego.model.checkpoint`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping

from ..engine.constants import BOARD_COLUMNS, BOARD_ROWS
from .contract import (
    BELIEF_TYPE_COUNT,
    MODEL_CONTRACT_VERSION,
    POLICY_LOGIT_COUNT,
    TOKEN_COUNT,
    TOKEN_FEATURES,
    VALUE_CLASS_COUNT,
    ModelContractError,
)

# ---------------------------------------------------------------------------
# Family identity
# ---------------------------------------------------------------------------

#: The checkpoint architecture identifier for every candidate in this family.
#: Deliberately not `integration_model_v1`: these networks are candidates for the
#: production model, and no Phase 5 fixture checkpoint may ever load as one.
ARCHITECTURE_FAMILY = "stratego_transformer_v1"

#: Bumped whenever anything that changes the *meaning* or *layout* of a
#: candidate's weights changes -- a family constant below, the block design, a
#: head design, or the initialization scheme. Not bumped for a new candidate row.
ARCHITECTURE_FAMILY_VERSION = "architecture_family_v1"

#: The one declared initialization seed for the whole Phase 6 benchmark family.
#: Every candidate Agent 3 benchmarks is built from this seed unless a later
#: agent explicitly documents a narrow sensitivity check.
FAMILY_INITIALIZATION_SEED = 20250601

# ---------------------------------------------------------------------------
# Family constants: identical across candidates, by instruction
# ---------------------------------------------------------------------------

#: Learned row embedding + learned column embedding, `h = W_x x + e_row + e_col`.
#: Twenty vectors rather than a hundred: the board's geometry is a grid, and a
#: per-square table cannot express "same column, one row further" at all.
POSITION_ENCODING = "learned_row_column_v1"

#: Pre-normalization blocks: LayerNorm -> sublayer -> residual, twice.
NORMALIZATION = "pre_layernorm"

#: The remaining fixed choices. These are *not* per-candidate fields, because a
#: candidate that varied one of them would not be a size comparison any more.
ACTIVATION = "gelu"
LAYER_NORM_EPS = 1e-5
LINEAR_BIAS = True
ATTENTION_IMPLEMENTATION = "torch.nn.MultiheadAttention(batch_first=True, need_weights=False)"
ATTENTION_MASK = "none"
POLICY_HEAD = "source_query_destination_key_scaled_with_source_and_destination_biases"
VALUE_HEAD = "mean_pool_tokens_then_two_layer_mlp"
BELIEF_HEAD = "per_token_linear"
INITIALIZATION = "uniform_fan_in_linear_zero_bias_normal_0.02_embeddings"

#: The family constants in one mapping, folded into every digest so that a
#: change to any of them is visible as a changed architecture identity rather
#: than as a silent difference between two runs.
FAMILY_CONSTANTS: dict[str, Any] = {
    "architecture_family": ARCHITECTURE_FAMILY,
    "architecture_family_version": ARCHITECTURE_FAMILY_VERSION,
    "model_contract_version": MODEL_CONTRACT_VERSION,
    "position_encoding": POSITION_ENCODING,
    "normalization": NORMALIZATION,
    "layer_norm_eps": LAYER_NORM_EPS,
    "activation": ACTIVATION,
    "linear_bias": LINEAR_BIAS,
    "attention_implementation": ATTENTION_IMPLEMENTATION,
    "attention_mask": ATTENTION_MASK,
    "policy_head": POLICY_HEAD,
    "value_head": VALUE_HEAD,
    "belief_head": BELIEF_HEAD,
    "initialization": INITIALIZATION,
}


class ArchitectureConfigError(ModelContractError):
    """An impossible or contract-violating candidate configuration.

    Subclasses :class:`~stratego.model.contract.ModelContractError` so that a
    caller catching the one model-boundary failure type catches this too.
    """


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------

#: The serialized field order. Fixed, because it is also the digest's field
#: order, and a digest that depended on dictionary insertion order would not be
#: stable across a save/reload cycle.
CONFIG_FIELDS: tuple[str, ...] = (
    "candidate_id",
    "width",
    "blocks",
    "heads",
    "feed_forward_width",
    "input_channels",
    "board_tokens",
    "policy_size",
    "value_classes",
    "belief_classes",
    "position_encoding",
    "normalization",
    "dropout",
    "architecture_family_version",
)


@dataclass(frozen=True)
class CandidateConfig:
    """One candidate's complete, serializable shape.

    Frozen because a configuration is an identity: it is digested, written into
    checkpoints and compared for equality, and every one of those uses is wrong
    if the object can be edited afterwards. Use :meth:`replace` for a variant.

    The four fields that actually vary across the ladder are `width`, `blocks`,
    `heads` and `feed_forward_width`. The rest are pinned by
    `model_contract_v2` or by the family, and are carried explicitly anyway so
    that a checkpoint states them rather than assuming the loading build agrees.
    """

    candidate_id: str
    width: int
    blocks: int
    heads: int
    feed_forward_width: int
    input_channels: int = TOKEN_FEATURES
    board_tokens: int = TOKEN_COUNT
    policy_size: int = POLICY_LOGIT_COUNT
    value_classes: int = VALUE_CLASS_COUNT
    belief_classes: int = BELIEF_TYPE_COUNT
    position_encoding: str = POSITION_ENCODING
    normalization: str = NORMALIZATION
    dropout: float = 0.0
    architecture_family_version: str = ARCHITECTURE_FAMILY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ArchitectureConfigError("candidate_id must be a non-empty string")

        for name in ("width", "blocks", "heads", "feed_forward_width"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ArchitectureConfigError(f"{name} must be a positive integer, got {value!r}")

        # `nn.MultiheadAttention` requires this exactly; it is the one hard
        # PyTorch constraint the ladder could have violated, so it is checked
        # here rather than surfacing from inside a constructor.
        if self.width % self.heads:
            raise ArchitectureConfigError(
                f"width {self.width} must divide evenly across {self.heads} heads "
                f"({self.candidate_id})"
            )

        # The contract owns these four numbers. A configuration that disagreed
        # would build a network whose outputs the boundary validators reject, so
        # it is refused at configuration time where the message is clearer.
        for name, expected in (
            ("input_channels", TOKEN_FEATURES),
            ("board_tokens", TOKEN_COUNT),
            ("policy_size", POLICY_LOGIT_COUNT),
            ("value_classes", VALUE_CLASS_COUNT),
            ("belief_classes", BELIEF_TYPE_COUNT),
        ):
            actual = getattr(self, name)
            if actual != expected:
                raise ArchitectureConfigError(
                    f"{name} must be {expected} under {MODEL_CONTRACT_VERSION}, got {actual!r}"
                )
        # The policy head is a 100x100 source/destination matrix flattened
        # row-major; if that identity ever failed the flatten would silently
        # stop meaning `100 * source + destination`.
        if self.policy_size != self.board_tokens * self.board_tokens:
            raise ArchitectureConfigError(
                f"policy_size {self.policy_size} must equal board_tokens^2 "
                f"({self.board_tokens * self.board_tokens})"
            )
        if self.board_tokens != BOARD_ROWS * BOARD_COLUMNS:
            raise ArchitectureConfigError(
                f"board_tokens {self.board_tokens} must equal the {BOARD_ROWS}x{BOARD_COLUMNS} "
                "board"
            )

        if self.position_encoding != POSITION_ENCODING:
            raise ArchitectureConfigError(
                f"position_encoding must be {POSITION_ENCODING!r} for this family, got "
                f"{self.position_encoding!r}"
            )
        if self.normalization != NORMALIZATION:
            raise ArchitectureConfigError(
                f"normalization must be {NORMALIZATION!r} for this family, got "
                f"{self.normalization!r}"
            )
        if self.architecture_family_version != ARCHITECTURE_FAMILY_VERSION:
            raise ArchitectureConfigError(
                f"architecture_family_version {self.architecture_family_version!r} is not this "
                f"build's {ARCHITECTURE_FAMILY_VERSION!r}; the weights would be laid out "
                "differently"
            )

        dropout = self.dropout
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise ArchitectureConfigError(f"dropout must be a number, got {dropout!r}")
        if not 0.0 <= float(dropout) < 1.0:
            raise ArchitectureConfigError(f"dropout must be in [0, 1), got {dropout!r}")
        # Normalize `0` to `0.0` so an int and a float configuration digest
        # identically; `object.__setattr__` because the dataclass is frozen.
        object.__setattr__(self, "dropout", float(dropout))

    # -- identity ----------------------------------------------------------

    @property
    def architecture_id(self) -> str:
        """The checkpoint architecture id. Shared by the whole family."""
        return ARCHITECTURE_FAMILY

    @property
    def head_dimension(self) -> int:
        """Channels per attention head. The policy head is separate and scales by
        `sqrt(width)`, since its query and key projections are full-width."""
        return self.width // self.heads

    def to_dict(self) -> dict:
        """Serializable form, in the fixed :data:`CONFIG_FIELDS` order."""
        return {name: getattr(self, name) for name in CONFIG_FIELDS}

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "CandidateConfig":
        """Rebuild from :meth:`to_dict`. Unknown or missing fields are refused.

        Strict in both directions on purpose. An unknown field means the file
        was written by a build that knew something this one does not; a missing
        field would have to be defaulted, and defaulting is exactly how a
        checkpoint silently acquires semantics it was not saved under.
        """
        if not isinstance(payload, Mapping):
            raise ArchitectureConfigError(
                f"expected a configuration mapping, got {type(payload).__name__}"
            )
        unexpected = sorted(set(payload) - set(CONFIG_FIELDS))
        if unexpected:
            raise ArchitectureConfigError(
                f"unknown candidate configuration field(s): {', '.join(unexpected)}"
            )
        missing = [name for name in CONFIG_FIELDS if name not in payload]
        if missing:
            raise ArchitectureConfigError(
                f"candidate configuration is missing field(s): {', '.join(missing)}"
            )
        return CandidateConfig(**{name: payload[name] for name in CONFIG_FIELDS})

    def replace(self, **changes: Any) -> "CandidateConfig":
        """A validated variant. Used by tests and by later sensitivity checks."""
        return replace(self, **changes)

    def identity(self) -> dict:
        """Configuration plus the family constants: everything the weights mean."""
        return {"config": self.to_dict(), "family": dict(FAMILY_CONSTANTS)}

    def digest(self) -> str:
        """Stable SHA-256 over :meth:`identity`.

        Covers the family constants as well as the configuration, so changing
        the activation or the normalization epsilon changes every candidate's
        digest -- which is the honest answer, since it changes what the weights
        mean. `sort_keys` makes the digest independent of field order in the
        serialized form even though :data:`CONFIG_FIELDS` already fixes it.
        """
        canonical = json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def short_digest(self) -> str:
        """First 16 hex characters of :meth:`digest`, for tables and filenames."""
        return self.digest()[:16]

    def describe(self) -> str:
        """One-line human description, e.g. `C3 192w x 6 blocks x 6 heads, ff 768`."""
        return (
            f"{self.candidate_id} {self.width}w x {self.blocks} blocks x {self.heads} heads, "
            f"ff {self.feed_forward_width}"
        )


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

#: The literal instruction table. `(candidate_id, width, blocks, heads, ff, role)`.
_LADDER: tuple[tuple[str, int, int, int, int, str], ...] = (
    ("C0", 64, 2, 4, 256, "small control"),
    ("C1", 128, 4, 4, 512, "small practical"),
    ("C2", 192, 4, 6, 768, "wider"),
    ("C3", 192, 6, 6, 768, "deeper"),
    ("C4", 256, 6, 8, 1024, "medium-large"),
    ("C5", 256, 8, 8, 1024, "deeper medium-large"),
    ("C6", 384, 8, 8, 1536, "paper-width/depth ceiling reference"),
)

#: Candidate id -> frozen configuration. The single source of truth for C0-C6.
CANDIDATES: dict[str, CandidateConfig] = {
    candidate_id: CandidateConfig(
        candidate_id=candidate_id,
        width=width,
        blocks=blocks,
        heads=heads,
        feed_forward_width=feed_forward,
    )
    for candidate_id, width, blocks, heads, feed_forward, _role in _LADDER
}

#: Candidate id -> the role the instruction assigns it. Kept out of
#: :class:`CandidateConfig` deliberately: a role is documentation, and folding it
#: into the configuration would make two identically-shaped networks compare
#: unequal because someone reworded a table cell.
CANDIDATE_ROLES: dict[str, str] = {
    candidate_id: role for candidate_id, _w, _b, _h, _f, role in _LADDER
}

#: Benchmark order, smallest first.
CANDIDATE_IDS: tuple[str, ...] = tuple(candidate_id for candidate_id, *_ in _LADDER)


def candidate_config(candidate_id: str) -> CandidateConfig:
    """Look up a ladder candidate by id, with a listing in the failure message."""
    try:
        return CANDIDATES[candidate_id]
    except KeyError:
        raise ArchitectureConfigError(
            f"unknown candidate {candidate_id!r}; this family defines "
            f"{', '.join(CANDIDATE_IDS)}"
        ) from None


def is_ladder_candidate(candidate_id: Any) -> bool:
    """Whether `candidate_id` names one of the frozen C0-C6 rows."""
    return isinstance(candidate_id, str) and candidate_id in CANDIDATES


def candidate_table() -> list[dict]:
    """The C0-C6 table as report-ready rows, roles and digests included."""
    return [
        {
            "candidate_id": config.candidate_id,
            "width": config.width,
            "blocks": config.blocks,
            "heads": config.heads,
            "feed_forward_width": config.feed_forward_width,
            "head_dimension": config.head_dimension,
            "role": CANDIDATE_ROLES[config.candidate_id],
            "config_digest": config.digest(),
        }
        for config in (CANDIDATES[candidate_id] for candidate_id in CANDIDATE_IDS)
    ]


def candidate_configs() -> dict[str, dict]:
    """Every ladder configuration in serialized form, keyed by candidate id."""
    return {candidate_id: CANDIDATES[candidate_id].to_dict() for candidate_id in CANDIDATE_IDS}


def config_digests() -> dict[str, str]:
    """Candidate id -> full configuration digest."""
    return {candidate_id: CANDIDATES[candidate_id].digest() for candidate_id in CANDIDATE_IDS}


def architecture_family_digest() -> str:
    """One digest over the family constants *and* every candidate digest.

    Changes if a family constant changes, if a candidate's shape changes, or if
    a candidate is added or removed. Agent 3 records it alongside its numbers so
    a later reader can tell whether a benchmark was run against this exact
    ladder.
    """
    canonical = json.dumps(
        {"family": dict(FAMILY_CONSTANTS), "candidates": config_digests()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def family_summary() -> dict:
    """Serializable statement of the whole family, for reports and checkpoints."""
    return {
        "architecture_family": ARCHITECTURE_FAMILY,
        "architecture_family_version": ARCHITECTURE_FAMILY_VERSION,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "initialization_seed": FAMILY_INITIALIZATION_SEED,
        "family_constants": dict(FAMILY_CONSTANTS),
        "candidate_ids": list(CANDIDATE_IDS),
        "candidate_table": candidate_table(),
        "candidate_configs": candidate_configs(),
        "config_digests": config_digests(),
        "architecture_family_digest": architecture_family_digest(),
        "ladder_adjustments": [],
    }


__all__ = [
    "ACTIVATION",
    "ARCHITECTURE_FAMILY",
    "ARCHITECTURE_FAMILY_VERSION",
    "ATTENTION_IMPLEMENTATION",
    "BELIEF_HEAD",
    "CANDIDATES",
    "CANDIDATE_IDS",
    "CANDIDATE_ROLES",
    "CONFIG_FIELDS",
    "FAMILY_CONSTANTS",
    "FAMILY_INITIALIZATION_SEED",
    "INITIALIZATION",
    "LAYER_NORM_EPS",
    "LINEAR_BIAS",
    "NORMALIZATION",
    "POLICY_HEAD",
    "POSITION_ENCODING",
    "VALUE_HEAD",
    "ArchitectureConfigError",
    "CandidateConfig",
    "architecture_family_digest",
    "candidate_config",
    "candidate_configs",
    "candidate_table",
    "config_digests",
    "family_summary",
    "is_ladder_candidate",
]
