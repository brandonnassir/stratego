"""Phase 17 Agent 3: inventory masking, orientation, and batched generation.

Specification sources:

- `03_AGENT_3_AUTOREGRESSIVE_SETUP_NETWORK.md` sections 3 and 4
- `00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md` section 7
- `reports/phase17/ataraxos_method_map_v1.md` rows S02, S03, S04, S05

The three refusals this module exists to make
---------------------------------------------
1. An exhausted piece type is excluded *before* normalization, so it carries
   probability exactly zero and cannot be drawn by any sampler, adversarial
   logits included.
2. Generation happens in canonical own-side coordinates and reaches the
   engine only through the accepted Phase 15 helper. Canonical Blue passed
   straight to `create_game` is the Phase 11B defect that put a flag on the
   front row of 77.0% of Blue boards; `assert_engine_orientation` is imported,
   never re-derived.
3. There is no library, no template, no repair and no fallback. A malformed
   prefix or a failed inventory check raises; it is never patched up.

Reproducibility
---------------
Each token is drawn by inverse CDF against a uniform derived from its own
`(run, game, side, prefix)` seed, so the draw at prefix `k` does not depend on
how the chain was batched. A pool of 512 chains and one chain alone therefore
produce byte-identical setups under the same raw snapshot. The cumulative sum
runs in float64 on CPU so the *sampling* step is bit-stable; the forward pass
that produced the probabilities is still device-dependent, and the determinism
claim is per-device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...belief.phase15.orientation import assert_engine_orientation
from ...engine.constants import NUM_PIECE_TYPES, PLAYERS, RED
from ...engine.setup import validate_setup
from ...setups.identity import class_fingerprint, content_fingerprint, orient_setup
from .setup_contract import (
    INVENTORY_VECTOR,
    ORIENTATION_RULE_VERSION,
    SETUP_PREFIXES,
    SETUP_SEQUENCE_LENGTH,
    START_TOKEN,
    Phase17SetupError,
    Phase17SetupGenerationError,
    Phase17SetupOrientationError,
    seed_uniform,
    setup_root_seed,
    setup_token_seed,
)
from .setup_model import Phase17SetupModel

_INVENTORY = np.array(INVENTORY_VECTOR, dtype=np.int64)


# ---------------------------------------------------------------------------
# Inventory mask
# ---------------------------------------------------------------------------


def remaining_counts(prefix: "np.ndarray | list[int] | tuple[int, ...]") -> np.ndarray:
    """Remaining inventory after `prefix`, derived from the prefix alone.

    Raises on an invalid prefix rather than repairing it: a negative remaining
    count means an earlier draw already violated the inventory, and continuing
    would produce a setup that fails validation 40 tokens later with no trace
    of which draw broke it.
    """
    tokens = np.asarray(prefix, dtype=np.int64).reshape(-1)
    if tokens.size > SETUP_PREFIXES:
        raise Phase17SetupError(f"prefix longer than {SETUP_PREFIXES}: {tokens.size}")
    if tokens.size and (tokens.min() < 0 or tokens.max() >= NUM_PIECE_TYPES):
        raise Phase17SetupError(f"prefix holds an unknown piece type: {tokens.tolist()}")
    used = np.bincount(tokens, minlength=NUM_PIECE_TYPES)
    remaining = _INVENTORY - used
    if (remaining < 0).any():
        over = [int(index) for index in np.nonzero(remaining < 0)[0]]
        raise Phase17SetupError(f"prefix over-uses piece type(s) {over}: {tokens.tolist()}")
    return remaining


def inventory_mask_from_prefix(
    prefix: "np.ndarray | list[int] | tuple[int, ...]",
) -> np.ndarray:
    """Boolean `[12]` mask of types still available after `prefix`."""
    return remaining_counts(prefix) > 0


def batched_remaining(tokens: torch.Tensor, prefix_length: int) -> torch.Tensor:
    """Remaining counts for every chain in a batch, as `[B, 12]`.

    `tokens` is the `[B, 40]` placement buffer; only its first `prefix_length`
    columns are read, so an uninitialised tail cannot leak into the mask.
    """
    batch = tokens.shape[0]
    counts = torch.zeros((batch, NUM_PIECE_TYPES), dtype=torch.int64, device=tokens.device)
    if prefix_length:
        drawn = tokens[:, :prefix_length]
        counts.scatter_add_(1, drawn, torch.ones_like(drawn))
    inventory = torch.as_tensor(INVENTORY_VECTOR, dtype=torch.int64, device=tokens.device)
    remaining = inventory[None, :] - counts
    if bool((remaining < 0).any()):
        raise Phase17SetupGenerationError(
            f"inventory went negative at prefix {prefix_length}; a mask was bypassed"
        )
    return remaining


def masked_probabilities(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the legal types only.

    The exclusion is applied to the logits *before* the softmax, so an
    exhausted type receives probability exactly 0.0 and no renormalisation
    residue. Adding the mask after the softmax would leave a type with a tiny
    but non-zero weight, which is precisely what the adversarial-logit test
    is written to catch.
    """
    if logits.shape != mask.shape:
        raise Phase17SetupError(
            f"logits {tuple(logits.shape)} and mask {tuple(mask.shape)} disagree"
        )
    if not bool(mask.any(dim=-1).all()):
        raise Phase17SetupGenerationError("a prefix has no legal next piece type")
    excluded = logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(excluded.to(torch.float32), dim=-1)


def inverse_cdf_choice(
    probabilities: np.ndarray, mask: np.ndarray, uniforms: np.ndarray
) -> np.ndarray:
    """Pick one legal index per row by inverse CDF against `uniforms`.

    Exact by construction rather than by luck. The mask is re-applied and each
    row renormalised in float64 *before* the cumulative sum, because a softmax
    row that sums to `1 - 1e-7` (CPU and MPS approach 1 from different sides)
    leaves a sliver of `[0, 1)` past the final cumulative value. A draw landing
    in that sliver runs off the end of the array, and clamping the result to
    the last index would place whatever piece type happens to sit at index 11
    -- exhausted or not.

    Clamping instead to the last index with nonzero mass is the correct
    resolution of a rounding artifact, not a repair of a bad sample: it is the
    value the exact arithmetic would have produced.

    Working in float64 on CPU also makes the draw independent of batching and
    of device, so a pool of 512 and a single chain choose identically.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if probabilities.shape != mask.shape:
        raise Phase17SetupError(
            f"probabilities {probabilities.shape} and mask {mask.shape} disagree"
        )
    if not mask.any(axis=1).all():
        raise Phase17SetupGenerationError("a row has no legal index to choose from")

    exact = np.where(mask, probabilities, 0.0)
    totals = exact.sum(axis=1, keepdims=True)
    if not np.isfinite(totals).all() or (totals <= 0.0).any():
        raise Phase17SetupGenerationError("a row holds no probability mass on a legal index")

    cumulative = np.cumsum(exact / totals, axis=1)
    draws = np.asarray(uniforms, dtype=np.float64)
    chosen = np.array(
        [
            int(np.searchsorted(cumulative[row], draws[row], side="right"))
            for row in range(cumulative.shape[0])
        ],
        dtype=np.int64,
    )
    last_legal = (mask.shape[1] - 1) - np.argmax(mask[:, ::-1], axis=1)
    return np.minimum(chosen, last_legal)


# ---------------------------------------------------------------------------
# Orientation boundary
# ---------------------------------------------------------------------------


def to_engine_setup(canonical: "tuple[int, ...]", player: int) -> tuple[int, ...]:
    """Validate a canonical 40-tuple, orient it, and re-check the placement.

    The inventory check runs on the *canonical* tuple, the orientation runs
    through the accepted helper, and the result is re-derived against the
    engine's own `SETUP_SQUARES` by `assert_engine_orientation` -- so a defect
    in `orient_setup` cannot hide behind a check written in terms of it.
    """
    if player not in PLAYERS:
        raise Phase17SetupOrientationError(f"unknown player: {player!r}")
    entries = tuple(int(value) for value in canonical)
    try:
        validate_setup(entries, player)
    except Exception as error:  # engine SetupError
        raise Phase17SetupGenerationError(
            f"sampled setup failed the engine inventory check: {error}"
        ) from error
    engine_setup = orient_setup(entries, player)
    try:
        assert_engine_orientation(entries, engine_setup, player)
    except Exception as error:
        raise Phase17SetupOrientationError(str(error)) from error
    validate_setup(engine_setup, player)
    return engine_setup


# ---------------------------------------------------------------------------
# One sampled setup and its behavior record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampledSetup:
    """One side's sampled setup with the complete behavior trace behind it.

    Every field the Agent 1 `SetupEpisode` schema needs from generation is
    here; `setup_episode.SetupEpisode.create` copies them across without
    recomputation, so the episode can never disagree with what was drawn.
    """

    color: int
    root_seed: int
    per_token_seeds: tuple
    canonical_setup: tuple
    engine_setup: tuple
    tokens: np.ndarray
    inventory_masks: np.ndarray
    behavior_probabilities: np.ndarray
    behavior_log_probabilities: np.ndarray
    suffix_information_content: np.ndarray
    prefix_wdl_predictions: np.ndarray
    prefix_conditional_entropy_predictions: np.ndarray
    setup_model_state_digest: str
    setup_snapshot_iteration: int

    @property
    def canonical_fingerprint(self) -> str:
        return content_fingerprint(self.canonical_setup)

    @property
    def reflection_class_fingerprint(self) -> str:
        return class_fingerprint(self.canonical_setup)

    @property
    def orientation_rule_version(self) -> str:
        return ORIENTATION_RULE_VERSION


def suffix_information(log_probabilities: np.ndarray) -> np.ndarray:
    """`I(sigma_bar | sigma_k) = -sum_{j >= k} log pi(t_j)` in nats.

    Method map row S07. This is the realized information content of the
    suffix -- the only quantity computable from one sampled setup, and the
    Monte Carlo estimator of the conditional entropy the paper's `H` names.
    """
    values = np.asarray(log_probabilities, dtype=np.float64)
    return np.flip(np.cumsum(np.flip(-values))).astype(np.float32)


# ---------------------------------------------------------------------------
# Vectorized generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_setups(
    model: Phase17SetupModel,
    *,
    run_id: str,
    game_ids: "list[str]",
    color: int,
    model_state_digest: str,
    snapshot_iteration: int,
    device: str | None = None,
) -> "list[SampledSetup]":
    """Sample one setup per `game_id`, all under one frozen raw snapshot.

    Vectorized across the batch: 40 forward passes over growing prefixes
    rather than 40 x B. The per-token seed makes the result independent of
    `len(game_ids)`, so a pool refill produces the same setup for a game id
    that a single-chain draw would.
    """
    if color not in PLAYERS:
        raise Phase17SetupError(f"unknown colour: {color!r}")
    if not game_ids:
        raise Phase17SetupError("generate_setups needs at least one game id")
    if len(set(game_ids)) != len(game_ids):
        raise Phase17SetupError("game ids must be unique within one generation call")

    target = device or next(model.parameters()).device
    batch = len(game_ids)

    root_seeds = [setup_root_seed(run_id, game_id, color) for game_id in game_ids]
    token_seeds = np.array(
        [[setup_token_seed(root, prefix) for prefix in range(SETUP_PREFIXES)] for root in root_seeds],
        dtype=np.int64,
    )
    uniforms = np.array(
        [[seed_uniform(int(seed)) for seed in row] for row in token_seeds], dtype=np.float64
    )

    tokens = torch.zeros((batch, SETUP_PREFIXES), dtype=torch.long, device=target)
    sequence = torch.full((batch, SETUP_SEQUENCE_LENGTH), START_TOKEN, dtype=torch.long, device=target)

    masks = np.zeros((batch, SETUP_PREFIXES, NUM_PIECE_TYPES), dtype=bool)
    probabilities = np.zeros((batch, SETUP_PREFIXES, NUM_PIECE_TYPES), dtype=np.float32)
    log_probabilities = np.zeros((batch, SETUP_PREFIXES), dtype=np.float32)
    wdl = np.zeros((batch, SETUP_PREFIXES, 3), dtype=np.float32)
    conditional_entropy = np.zeros((batch, SETUP_PREFIXES), dtype=np.float32)

    was_training = model.training
    model.eval()
    try:
        for prefix in range(SETUP_PREFIXES):
            outputs = model(sequence[:, : prefix + 1])
            logits = outputs["piece_logits"][:, prefix]
            mask = batched_remaining(tokens, prefix) > 0
            step_probabilities = masked_probabilities(logits, mask)

            row_probabilities = step_probabilities.to("cpu", torch.float32).numpy()
            row_mask = mask.to("cpu").numpy()

            chosen = inverse_cdf_choice(row_probabilities, row_mask, uniforms[:, prefix])
            if not row_mask[np.arange(batch), chosen].all():
                raise Phase17SetupGenerationError(
                    f"prefix {prefix} sampled an exhausted piece type"
                )
            picked = row_probabilities[np.arange(batch), chosen]
            if not np.isfinite(picked).all() or (picked <= 0.0).any():
                raise Phase17SetupGenerationError(
                    f"prefix {prefix} drew a zero-probability token"
                )

            masks[:, prefix] = row_mask
            probabilities[:, prefix] = row_probabilities
            log_probabilities[:, prefix] = np.log(picked.astype(np.float64)).astype(np.float32)
            wdl[:, prefix] = (
                torch.softmax(outputs["wdl_logits"][:, prefix].to(torch.float32), dim=-1)
                .to("cpu")
                .numpy()
            )
            conditional_entropy[:, prefix] = (
                outputs["conditional_entropy"][:, prefix].to("cpu", torch.float32).numpy()
            )

            step = torch.as_tensor(chosen, dtype=torch.long, device=target)
            tokens[:, prefix] = step
            sequence[:, prefix + 1] = step
    finally:
        model.train(was_training)

    drawn = tokens.to("cpu").numpy()
    results: "list[SampledSetup]" = []
    for row, game_id in enumerate(game_ids):
        canonical = tuple(int(value) for value in drawn[row])
        engine_setup = to_engine_setup(canonical, color)
        results.append(
            SampledSetup(
                color=int(color),
                root_seed=int(root_seeds[row]),
                per_token_seeds=tuple(int(value) for value in token_seeds[row]),
                canonical_setup=canonical,
                engine_setup=engine_setup,
                tokens=drawn[row].astype(np.int8),
                inventory_masks=masks[row],
                behavior_probabilities=probabilities[row],
                behavior_log_probabilities=log_probabilities[row],
                suffix_information_content=suffix_information(log_probabilities[row]),
                prefix_wdl_predictions=wdl[row],
                prefix_conditional_entropy_predictions=conditional_entropy[row],
                setup_model_state_digest=model_state_digest,
                setup_snapshot_iteration=int(snapshot_iteration),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------


class SetupPool:
    """A per-side pool generated under one frozen raw setup snapshot.

    The pool is keyed by game id rather than by draw order, because a game's
    setup must be reproducible from `(run, game, side)` alone. Rebinding the
    snapshot discards whatever is left: section 4 forbids relabelling old pool
    entries as current, and a stale entry carries the *old* behavior
    probabilities that the ratio denominator would then silently misattribute.
    """

    def __init__(
        self,
        model: Phase17SetupModel,
        *,
        run_id: str,
        color: int,
        model_state_digest: str,
        snapshot_iteration: int,
        size: int,
    ) -> None:
        self.model = model
        self.run_id = run_id
        self.color = int(color)
        self.model_state_digest = model_state_digest
        self.snapshot_iteration = int(snapshot_iteration)
        self.size = int(size)
        self._entries: "dict[str, SampledSetup]" = {}
        self.generated_count = 0
        self.consumed_count = 0
        self.refill_count = 0

    @property
    def unused_count(self) -> int:
        return len(self._entries)

    def take(self, game_id: str) -> SampledSetup:
        """The pool entry for `game_id`, generating it if the pool is short."""
        entry = self._entries.pop(game_id, None)
        if entry is None:
            self.refill_count += 1
            entry = generate_setups(
                self.model,
                run_id=self.run_id,
                game_ids=[game_id],
                color=self.color,
                model_state_digest=self.model_state_digest,
                snapshot_iteration=self.snapshot_iteration,
            )[0]
            self.generated_count += 1
        self.consumed_count += 1
        return entry

    def prefetch(self, game_ids: "list[str]") -> None:
        """Generate ahead for `game_ids`, capped at the pool size."""
        pending = [game_id for game_id in game_ids if game_id not in self._entries]
        pending = pending[: max(0, self.size - len(self._entries))]
        if not pending:
            return
        samples = generate_setups(
            self.model,
            run_id=self.run_id,
            game_ids=pending,
            color=self.color,
            model_state_digest=self.model_state_digest,
            snapshot_iteration=self.snapshot_iteration,
        )
        for game_id, sample in zip(pending, samples):
            self._entries[game_id] = sample
        self.generated_count += len(samples)

    def rebind(self, model: Phase17SetupModel, *, model_state_digest: str, snapshot_iteration: int) -> int:
        """Adopt a new raw snapshot and discard every entry drawn under the old one."""
        discarded = len(self._entries)
        self._entries.clear()
        self.model = model
        self.model_state_digest = model_state_digest
        self.snapshot_iteration = int(snapshot_iteration)
        return discarded

    def telemetry(self) -> dict:
        return {
            "color": self.color,
            "size": self.size,
            "snapshot_iteration": self.snapshot_iteration,
            "setup_model_state_digest": self.model_state_digest,
            "generated": self.generated_count,
            "consumed": self.consumed_count,
            "unused": self.unused_count,
            "refills": self.refill_count,
        }


__all__ = [
    "SampledSetup",
    "SetupPool",
    "batched_remaining",
    "generate_setups",
    "inverse_cdf_choice",
    "inventory_mask_from_prefix",
    "masked_probabilities",
    "remaining_counts",
    "suffix_information",
    "to_engine_setup",
]
