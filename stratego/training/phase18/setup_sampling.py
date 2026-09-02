"""Phase 18: inventory and handedness masking, seeded generation, reflection,
and the orientation boundary (S02, S04, S05, S06, S07, S11, S20, S24).

Generation, once
----------------
```text
for prefix k in 0..39 (one forward pass per prefix over the growing batch):
    legal_k    = inventory(prefix) AND handedness(square k)      # S02, S04
    log_pi_k   = log_softmax(logits_k masked with -inf)          # exact zeros
    t_k        = inverse-CDF draw against a per-(chain, prefix) seeded uniform
    record log_pi_k, log_pi_k[t_k], W/D/L LOGITS, h_k
after square 39:
    I_k        = -sum_{j >= k} log_pi_j[t_j]                     # S11
    reflected  = independent seeded uniform < 0.5                # S05
    played     = reflect(tokens) if reflected else tokens
    engine     = orient(played, lane) through the accepted helper # S07
```

The network only ever sees and is only ever trained on the NETWORK
orientation, in which the Flag sits in the permitted right half. The board
that is played may be its mirror. Every log-probability, mask and prediction
recorded here belongs to the network orientation, and `setup_buffer` flips a
played board back before any gather (S06).

Reproducibility
---------------
Each token is drawn by inverse CDF against a uniform derived from its own
`(namespace, seed, snapshot, chain, prefix)` seed, so the draw at prefix `k`
does not depend on how the chain was batched; the cumulative sum runs in
float64 on CPU so the sampling step is bit-stable. The forward pass that
produced the probabilities remains device-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...belief.phase15.orientation import assert_engine_orientation
from ...engine.constants import BOMB, FLAG, NUM_PIECE_TYPES, PLAYERS, RED, BLUE
from ...engine.setup import validate_setup
from ...setups.identity import (
    CANONICAL_FILES,
    CANONICAL_RANKS,
    FRONT_RANK,
    canonical_rank_file,
    class_fingerprint,
    content_fingerprint,
    orient_setup,
    reflect_canonical,
)
from .setup_contract import (
    FLAG_PERMITTED_FILES,
    INVENTORY_VECTOR,
    ORIENTATION_RULE_VERSION,
    REFLECTION_PROBABILITY,
    SETUP_PREFIXES,
    SETUP_SEQUENCE_LENGTH,
    START_TOKEN,
    Phase18SetupError,
    Phase18SetupGenerationError,
    Phase18SetupOrientationError,
    pool_root_seed,
    reflection_seed,
    seed_uniform,
    token_seed,
)
from .setup_model import Phase18SetupModel

_INVENTORY = np.array(INVENTORY_VECTOR, dtype=np.int64)

#: Files whose front-rank square faces open board rather than a lake: the
#: lakes occupy columns 2-3 and 6-7 of the two middle rows.
CORRIDOR_FILES = (0, 1, 4, 5, 8, 9)
IMMOVABLE = (FLAG, BOMB)


# ---------------------------------------------------------------------------
# Inventory mask (S02)
# ---------------------------------------------------------------------------


def remaining_counts(prefix) -> np.ndarray:
    """Remaining inventory after `prefix`, derived from the prefix alone.

    Raises on an invalid prefix rather than repairing it.
    """
    tokens = np.asarray(prefix, dtype=np.int64).reshape(-1)
    if tokens.size > SETUP_PREFIXES:
        raise Phase18SetupError(f"prefix longer than {SETUP_PREFIXES}: {tokens.size}")
    if tokens.size and (tokens.min() < 0 or tokens.max() >= NUM_PIECE_TYPES):
        raise Phase18SetupError(f"prefix holds an unknown piece type: {tokens.tolist()}")
    used = np.bincount(tokens, minlength=NUM_PIECE_TYPES)
    remaining = _INVENTORY - used
    if (remaining < 0).any():
        over = [int(index) for index in np.nonzero(remaining < 0)[0]]
        raise Phase18SetupError(f"prefix over-uses piece type(s) {over}: {tokens.tolist()}")
    return remaining


def inventory_mask_from_prefix(prefix) -> np.ndarray:
    """Boolean `[12]` mask of types still available after `prefix`."""
    return remaining_counts(prefix) > 0


def batched_remaining(tokens: torch.Tensor, prefix_length: int) -> torch.Tensor:
    """Remaining counts for every chain in a batch, `[B, 12]`, from the prefix
    columns only."""
    batch = tokens.shape[0]
    counts = torch.zeros((batch, NUM_PIECE_TYPES), dtype=torch.int64, device=tokens.device)
    if prefix_length:
        drawn = tokens[:, :prefix_length]
        counts.scatter_add_(1, drawn, torch.ones_like(drawn))
    inventory = torch.as_tensor(INVENTORY_VECTOR, dtype=torch.int64, device=tokens.device)
    remaining = inventory[None, :] - counts
    if bool((remaining < 0).any()):
        raise Phase18SetupGenerationError(
            f"inventory went negative at prefix {prefix_length}; a mask was bypassed"
        )
    return remaining


# ---------------------------------------------------------------------------
# Handedness mask (S04)
# ---------------------------------------------------------------------------


def handedness_mask(prefix: int) -> np.ndarray:
    """`[12]` mask that forbids the Flag outside the permitted half at square
    `prefix`; every other type is unaffected.

    The published `right_side` buffer marks files 5..9 of each rank, and
    `legal_mask[:, ~right_side, FLAG_IDX] = False`.
    """
    if not 0 <= int(prefix) < SETUP_PREFIXES:
        raise Phase18SetupError(f"prefix out of range: {prefix!r}")
    mask = np.ones(NUM_PIECE_TYPES, dtype=bool)
    _, file = canonical_rank_file(int(prefix))
    if file not in FLAG_PERMITTED_FILES:
        mask[FLAG] = False
    return mask


def legal_masks(tokens: torch.Tensor, prefix: int, *, force_handedness: bool) -> torch.Tensor:
    """`[B, 12]` legal-next-type mask at `prefix`: inventory, and handedness
    when forced. Derived from the prefix and the square; never passed in."""
    mask = batched_remaining(tokens, prefix) > 0
    if force_handedness:
        handed = torch.as_tensor(handedness_mask(prefix), device=tokens.device)
        mask = mask & handed[None, :]
    if not bool(mask.any(dim=-1).all()):
        raise Phase18SetupGenerationError(f"prefix {prefix} has a chain with no legal type")
    return mask


def masked_log_probabilities(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`log_softmax` over the legal types only; illegal entries are `-inf`
    (probability exactly zero), matching the published `finfo.min` fill in
    effect.
    """
    if logits.shape != mask.shape:
        raise Phase18SetupError(f"logits {tuple(logits.shape)} and mask {tuple(mask.shape)} disagree")
    if not bool(mask.any(dim=-1).all()):
        raise Phase18SetupGenerationError("a prefix has no legal next piece type")
    excluded = logits.to(torch.float32).masked_fill(~mask, float("-inf"))
    return torch.log_softmax(excluded, dim=-1)


def inverse_cdf_choice(probabilities: np.ndarray, mask: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Pick one legal index per row by inverse CDF against `uniforms`.

    The mask is re-applied and each row renormalised in float64 before the
    cumulative sum; a draw that lands past the final cumulative value because
    of float rounding is clamped to the last index with nonzero mass, never to
    index 11.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if probabilities.shape != mask.shape:
        raise Phase18SetupError("probabilities and mask disagree")
    if not mask.any(axis=1).all():
        raise Phase18SetupGenerationError("a row has no legal index to choose from")
    exact = np.where(mask, probabilities, 0.0)
    totals = exact.sum(axis=1, keepdims=True)
    if not np.isfinite(totals).all() or (totals <= 0.0).any():
        raise Phase18SetupGenerationError("a row holds no probability mass on a legal index")
    cumulative = np.cumsum(exact / totals, axis=1)
    draws = np.asarray(uniforms, dtype=np.float64)
    chosen = np.array(
        [int(np.searchsorted(cumulative[row], draws[row], side="right")) for row in range(cumulative.shape[0])],
        dtype=np.int64,
    )
    last_legal = (mask.shape[1] - 1) - np.argmax(mask[:, ::-1], axis=1)
    return np.minimum(chosen, last_legal)


# ---------------------------------------------------------------------------
# Reflection (S05) and orientation (S07)
# ---------------------------------------------------------------------------


def reflect_tokens(tokens: np.ndarray) -> np.ndarray:
    """Left-right reflection of `[..., 40]` canonical boards: file `f -> 9 - f`
    inside every rank. Vectorised form of `identity.reflect_canonical`."""
    array = np.asarray(tokens)
    if array.shape[-1] != SETUP_PREFIXES:
        raise Phase18SetupError(f"expected a trailing dimension of {SETUP_PREFIXES}")
    return array.reshape(*array.shape[:-1], CANONICAL_RANKS, CANONICAL_FILES)[..., ::-1].reshape(array.shape).copy()


def to_engine_setup(canonical, player: int) -> tuple:
    """Validate a canonical 40-tuple, orient it through the accepted helper, and
    re-check the placement against the engine's own squares (S07)."""
    if player not in PLAYERS:
        raise Phase18SetupOrientationError(f"unknown player: {player!r}")
    entries = tuple(int(value) for value in canonical)
    try:
        validate_setup(entries, player)
    except Exception as error:  # engine SetupError
        raise Phase18SetupGenerationError(f"setup failed the engine inventory check: {error}") from error
    engine_setup = orient_setup(entries, player)
    try:
        assert_engine_orientation(entries, engine_setup, player)
    except Exception as error:
        raise Phase18SetupOrientationError(str(error)) from error
    validate_setup(engine_setup, player)
    return engine_setup


def has_opening_move(canonical) -> bool:
    """S24: whether the owner of this setup has any legal move at ply 0.

    At ply 0 every own square is occupied, so a piece can move only forward
    off the front rank into the middle rows, and only where a lake does not
    block it: a movable piece on the front rank at a corridor file.
    """
    entries = tuple(int(value) for value in canonical)
    if len(entries) != SETUP_PREFIXES:
        raise Phase18SetupError(f"expected {SETUP_PREFIXES} entries, got {len(entries)}")
    for file in CORRIDOR_FILES:
        if entries[FRONT_RANK * CANONICAL_FILES + file] not in IMMOVABLE:
            return True
    return False


def suffix_information(selected_log_probabilities) -> np.ndarray:
    """`I_k = -sum_{j >= k} log pi_behavior(t_j | sigma_j)` in nats (S11)."""
    values = np.asarray(selected_log_probabilities, dtype=np.float64)
    return np.flip(np.cumsum(np.flip(-values))).astype(np.float32)


# ---------------------------------------------------------------------------
# One sampled setup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampledSetup:
    """One pooled setup with the complete behavior record behind it.

    `network_tokens` is what the network generated (Flag in the permitted
    half); `played_canonical` is the board actually played, which is the
    mirror when `reflected`. Every recorded probability belongs to the
    network orientation.
    """

    index: int
    lane: int
    root_seed: int
    reflection_seed: int
    reflected: bool
    network_tokens: np.ndarray
    played_canonical: tuple
    engine_setup: tuple
    legal_masks: np.ndarray
    behavior_log_probs: np.ndarray
    behavior_selected_log_prob: np.ndarray
    suffix_information: np.ndarray
    wdl_logits: np.ndarray
    entropy_prediction: np.ndarray
    snapshot_digest: str
    snapshot_iteration: int

    @property
    def content_fingerprint(self) -> str:
        """Identity of the PLAYED board (S10): a setup and its mirror differ."""
        return content_fingerprint(self.played_canonical)

    @property
    def class_fingerprint(self) -> str:
        return class_fingerprint(self.played_canonical)

    @property
    def network_fingerprint(self) -> str:
        return content_fingerprint(tuple(int(v) for v in self.network_tokens))

    @property
    def orientation_rule_version(self) -> str:
        return ORIENTATION_RULE_VERSION

    @property
    def opening_move(self) -> bool:
        return has_opening_move(self.played_canonical)


@dataclass
class PoolGeneration:
    samples: list
    telemetry: dict


# ---------------------------------------------------------------------------
# Vectorised generation (S20: one immutable pool per snapshot)
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_pool(
    model: Phase18SetupModel,
    *,
    namespace: str,
    seed_index: int,
    snapshot_iteration: int,
    snapshot_digest: str,
    count: int,
    force_handedness: bool = True,
    reflection_probability: float = REFLECTION_PROBABILITY,
    device: str | None = None,
    purpose: str = "pool",
) -> PoolGeneration:
    """Sample `count` setups under one frozen raw snapshot.

    `purpose` selects the seed stream (`pool` for training pools, `eval` for
    held-out EMA samples) so the two can never share a draw. Even pool indices
    are the Red lane, odd the Blue lane (S20).
    """
    if count < 1:
        raise Phase18SetupError("count must be positive")
    target = device or next(model.parameters()).device
    batch = int(count)

    if purpose == "pool":
        roots = [pool_root_seed(namespace, seed_index, snapshot_iteration, index) for index in range(batch)]
        flips = [reflection_seed(namespace, seed_index, snapshot_iteration, index) for index in range(batch)]
    else:
        from .setup_contract import stream_seed

        roots = [stream_seed(namespace, purpose, int(seed_index), int(snapshot_iteration), index) for index in range(batch)]
        flips = [
            stream_seed(namespace, f"{purpose}_reflection", int(seed_index), int(snapshot_iteration), index)
            for index in range(batch)
        ]
    uniforms = np.array(
        [[seed_uniform(token_seed(root, prefix)) for prefix in range(SETUP_PREFIXES)] for root in roots],
        dtype=np.float64,
    )

    tokens = torch.zeros((batch, SETUP_PREFIXES), dtype=torch.long, device=target)
    sequence = torch.full((batch, SETUP_SEQUENCE_LENGTH), START_TOKEN, dtype=torch.long, device=target)

    masks = np.zeros((batch, SETUP_PREFIXES, NUM_PIECE_TYPES), dtype=bool)
    log_probs = np.zeros((batch, SETUP_PREFIXES, NUM_PIECE_TYPES), dtype=np.float32)
    selected = np.zeros((batch, SETUP_PREFIXES), dtype=np.float32)
    wdl = np.zeros((batch, SETUP_PREFIXES, 3), dtype=np.float32)
    entropy = np.zeros((batch, SETUP_PREFIXES), dtype=np.float32)

    was_training = model.training
    model.eval()
    try:
        for prefix in range(SETUP_PREFIXES):
            outputs = model(sequence[:, : prefix + 1])
            logits = outputs["piece_logits"][:, prefix]
            mask = legal_masks(tokens, prefix, force_handedness=force_handedness)
            step_log = masked_log_probabilities(logits, mask)
            row_log = step_log.to("cpu", torch.float32).numpy()
            row_mask = mask.to("cpu").numpy()
            row_probabilities = np.where(row_mask, np.exp(row_log.astype(np.float64)), 0.0)

            chosen = inverse_cdf_choice(row_probabilities, row_mask, uniforms[:, prefix])
            if not row_mask[np.arange(batch), chosen].all():
                raise Phase18SetupGenerationError(f"prefix {prefix} sampled an illegal piece type")
            picked = row_log[np.arange(batch), chosen]
            if not np.isfinite(picked).all():
                raise Phase18SetupGenerationError(f"prefix {prefix} drew a zero-probability token")

            masks[:, prefix] = row_mask
            log_probs[:, prefix] = np.where(row_mask, row_log, 0.0).astype(np.float32)
            selected[:, prefix] = picked
            wdl[:, prefix] = outputs["wdl_logits"][:, prefix].to("cpu", torch.float32).numpy()
            entropy[:, prefix] = outputs["entropy_prediction"][:, prefix].to("cpu", torch.float32).numpy()

            step = torch.as_tensor(chosen, dtype=torch.long, device=target)
            tokens[:, prefix] = step
            sequence[:, prefix + 1] = step
    finally:
        model.train(was_training)

    drawn = tokens.to("cpu").numpy().astype(np.int64)
    flag_files_network = np.zeros(CANONICAL_FILES, dtype=np.int64)
    flag_files_played = np.zeros(CANONICAL_FILES, dtype=np.int64)
    samples: list = []
    reflected_count = 0
    terminal_count = 0
    for index in range(batch):
        network = drawn[index]
        flag_file = int(np.nonzero(network == FLAG)[0][0]) % CANONICAL_FILES
        flag_files_network[flag_file] += 1
        if force_handedness and flag_file not in FLAG_PERMITTED_FILES:
            raise Phase18SetupGenerationError(
                f"chain {index}: the Flag landed at file {flag_file}, outside the permitted half"
            )
        reflected = bool(seed_uniform(flips[index]) < reflection_probability)
        played = tuple(int(v) for v in (reflect_tokens(network) if reflected else network))
        if reflected:
            reflected_count += 1
            if tuple(int(v) for v in reflect_canonical(played)) != tuple(int(v) for v in network):
                raise Phase18SetupGenerationError(f"chain {index}: reflection is not an involution")
        flag_files_played[played.index(FLAG) % CANONICAL_FILES] += 1
        lane = RED if index % 2 == 0 else BLUE
        engine_setup = to_engine_setup(played, lane)
        sample = SampledSetup(
            index=index,
            lane=lane,
            root_seed=int(roots[index]),
            reflection_seed=int(flips[index]),
            reflected=reflected,
            network_tokens=network.astype(np.int8),
            played_canonical=played,
            engine_setup=engine_setup,
            legal_masks=masks[index],
            behavior_log_probs=log_probs[index],
            behavior_selected_log_prob=selected[index],
            suffix_information=suffix_information(selected[index]),
            wdl_logits=wdl[index],
            entropy_prediction=entropy[index],
            snapshot_digest=snapshot_digest,
            snapshot_iteration=int(snapshot_iteration),
        )
        if not sample.opening_move:
            terminal_count += 1
        samples.append(sample)

    contents = {sample.content_fingerprint for sample in samples}
    classes = {sample.class_fingerprint for sample in samples}
    telemetry = {
        "count": batch,
        "purpose": purpose,
        "snapshot_iteration": int(snapshot_iteration),
        "snapshot_digest": snapshot_digest,
        "force_handedness": bool(force_handedness),
        "reflection_probability": float(reflection_probability),
        "flag_file_histogram_network": flag_files_network.tolist(),
        "flag_file_histogram_played": flag_files_played.tolist(),
        "flag_in_permitted_half_fraction_network": float(
            flag_files_network[list(FLAG_PERMITTED_FILES)].sum() / batch
        ),
        "reflected_fraction": reflected_count / batch,
        "reflected_count": reflected_count,
        "legality_failures": 0,
        "orientation_failures": 0,
        "distinct_content_fingerprints": len(contents),
        "distinct_class_fingerprints": len(classes),
        "immediately_terminal_count": terminal_count,
        "lane_counts": {"red": sum(1 for s in samples if s.lane == RED), "blue": sum(1 for s in samples if s.lane == BLUE)},
        "mean_sequence_information_nats": float(np.mean([s.suffix_information[0] for s in samples])),
    }
    return PoolGeneration(samples=samples, telemetry=telemetry)


__all__ = [
    "CORRIDOR_FILES",
    "PoolGeneration",
    "SampledSetup",
    "batched_remaining",
    "generate_pool",
    "handedness_mask",
    "has_opening_move",
    "inventory_mask_from_prefix",
    "inverse_cdf_choice",
    "legal_masks",
    "masked_log_probabilities",
    "reflect_tokens",
    "remaining_counts",
    "suffix_information",
    "to_engine_setup",
]
