"""Phase 18: reusable setup pools, identity, outcome aggregation, the flat
advantage, and minibatches (S06, S08, S09, S10, S13, S14, S15, S19, S21-S24).

The published `ArrangementBuffer`, transcribed
-----------------------------------------------
```text
add_pool(samples, period)     new rows are prepended; identical played boards
                              collapse to the NEWEST snapshot (S10); counts,
                              running means and ready flags are reallocated
                              to zero for EVERY row, so aggregation spans
                              exactly one collection period (S23)
add_outcome(fingerprint, z)   running mean of the one-hot W/D/L outcome per
                              row (S09); an unknown fingerprint is fatal (S10)
process(alpha)                for ready rows only:
                                z_bar     = mean one-hot @ (-1, 0, +1)     (S08)
                                E[v_k]    = softmax(wdl_logits_k) @ (-1, 0, +1)
                                residual  = I_k - 10 h_k                   (S13)
                                delta_k   = (z_bar - E[v_k]) + alpha * residual
                                target_h  = I_k / 10                       (S12)
                              lambda != 1 is refused (S15)
minibatches(size, seed)       shuffled minibatches of ready rows; a played
                              board is flipped back to network orientation
                              before any gather and checked against the
                              recorded network tokens (S06)
filter(period)                rows older than the retention window expire
                              (S21); the buffer then needs a new pool
```

A row with no completed outcome in the period is never trained and never
treated as a draw: it is simply not ready.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...engine.constants import NUM_PIECE_TYPES
from .setup_contract import (
    CATEGORICAL_AGGREGATION,
    ENTROPY_NORMALIZER,
    SETUP_BUFFER_VERSION,
    SETUP_PREFIXES,
    SETUP_SEQUENCE_LENGTH,
    START_TOKEN,
    WDL_DRAW,
    WDL_LOSS,
    WDL_WIN,
    Phase18SetupAttributionError,
    Phase18SetupConfigError,
    Phase18SetupError,
)
from .setup_sampling import SampledSetup, reflect_tokens

_AGGREGATION = np.array(CATEGORICAL_AGGREGATION, dtype=np.float64)


def outcome_one_hot(outcome: int) -> np.ndarray:
    """`+1 / 0 / -1` to the published (loss, draw, win) one-hot: index z + 1."""
    if outcome not in (-1, 0, 1):
        raise Phase18SetupError(f"outcome must be -1, 0 or +1, got {outcome!r}")
    vector = np.zeros(3, dtype=np.float64)
    vector[int(outcome) + 1] = 1.0
    assert (vector[WDL_LOSS], vector[WDL_DRAW], vector[WDL_WIN]) == (
        float(outcome == -1),
        float(outcome == 0),
        float(outcome == 1),
    )
    return vector


def expected_value(wdl_probabilities: np.ndarray) -> np.ndarray:
    """`E[v] = p_win - p_loss` = probabilities @ (-1, 0, +1) (S08)."""
    probabilities = np.asarray(wdl_probabilities, dtype=np.float64)
    if probabilities.shape[-1] != 3:
        raise Phase18SetupError("W/D/L probabilities must have a trailing dimension of 3")
    return probabilities @ _AGGREGATION


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


@dataclass
class SetupBatch:
    """One consumable minibatch of ready rows, as device tensors."""

    sequence: torch.Tensor          # [B, 41] start token + 40 NETWORK tokens
    tokens: torch.Tensor            # [B, 40]
    masks: torch.Tensor             # [B, 40, 12] legal-by-inventory-and-handedness
    behavior_log_probs: torch.Tensor          # [B, 40, 12] masked log pi_b; illegal = 0
    behavior_selected_log_prob: torch.Tensor  # [B, 40] log pi_b(t_k | sigma_k)
    advantage: torch.Tensor         # [B, 40]
    value_target: torch.Tensor      # [B, 3] mean one-hot (loss, draw, win)
    entropy_target: torch.Tensor    # [B, 40] I_k / 10
    outcome_counts: torch.Tensor    # [B]
    fingerprints: list
    count: int

    def to(self, device) -> "SetupBatch":
        moved = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in self.__dict__.items()
        }
        return SetupBatch(**moved)


@dataclass
class ProcessedRows:
    """What `process` computed for the ready rows, plus its telemetry."""

    indices: np.ndarray
    advantage: np.ndarray
    outcome_term: np.ndarray
    entropy_residual: np.ndarray
    entropy_target: np.ndarray
    value_target: np.ndarray
    z_bar: np.ndarray
    expected_values: np.ndarray
    alpha: float
    telemetry: dict


class SetupBuffer:
    """The reusable pool buffer with published semantics."""

    version = SETUP_BUFFER_VERSION

    def __init__(self, *, storage_duration: int, device: str = "cpu") -> None:
        if int(storage_duration) < 0:
            raise Phase18SetupConfigError("storage_duration must be non-negative")
        self.storage_duration = int(storage_duration)
        self.device = device
        self.need_pool = True
        self._samples: list = []
        self._period_added: np.ndarray = np.zeros(0, dtype=np.int64)
        self._counts = np.zeros(0, dtype=np.int64)
        self._mean_one_hot = np.zeros((0, 3), dtype=np.float64)
        self._ready = np.zeros(0, dtype=bool)
        self._lookup: dict = {}
        self._processed: ProcessedRows | None = None
        self.duplicates_collapsed_total = 0
        self.attribution_failures = 0
        self.pools_added = 0
        self.outcomes_added = 0

    # -- rows ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> list:
        return list(self._samples)

    def ready_count(self) -> int:
        return int(self._ready.sum())

    def ready_indices(self) -> np.ndarray:
        return np.nonzero(self._ready)[0]

    def index_of(self, fingerprint: str) -> int:
        try:
            return self._lookup[fingerprint]
        except KeyError:
            self.attribution_failures += 1
            raise Phase18SetupAttributionError(
                f"setup {fingerprint[:16]} is not in the buffer; an outcome cannot be attributed "
                "(the retention window is too short, or the setup never entered a pool)"
            ) from None

    # -- add a pool (S10, S20, S23) -----------------------------------------

    def add_pool(self, samples, *, period: int) -> dict:
        samples = list(samples)
        if not samples:
            raise Phase18SetupError("a pool must hold at least one setup")
        for sample in samples:
            if not isinstance(sample, SampledSetup):
                raise Phase18SetupError("pools hold SampledSetup rows only")
        combined = samples + self._samples
        periods = np.concatenate(
            [np.full(len(samples), int(period), dtype=np.int64), self._period_added]
        )
        keep = mark_most_recent_appearance([s.content_fingerprint for s in combined], periods)
        duplicates = len(combined) - int(sum(keep))
        self._samples = [sample for sample, flag in zip(combined, keep) if flag]
        self._period_added = periods[np.asarray(keep, dtype=bool)]
        rows = len(self._samples)
        # Reallocated to zero for EVERY row: aggregation spans one period only.
        self._counts = np.zeros(rows, dtype=np.int64)
        self._mean_one_hot = np.zeros((rows, 3), dtype=np.float64)
        self._ready = np.zeros(rows, dtype=bool)
        self._lookup = {sample.content_fingerprint: index for index, sample in enumerate(self._samples)}
        if len(self._lookup) != rows:
            raise Phase18SetupError("duplicate fingerprints survived de-duplication")
        self._processed = None
        self.need_pool = False
        self.duplicates_collapsed_total += duplicates
        self.pools_added += 1
        return {
            "period": int(period),
            "added": len(samples),
            "duplicates_collapsed": duplicates,
            "rows": rows,
            "surviving_older_rows": int((self._period_added < int(period)).sum()),
        }

    # -- outcomes (S09) -------------------------------------------------------

    def add_outcome(self, fingerprint: str, outcome: int) -> None:
        if self.need_pool:
            raise Phase18SetupError("the buffer needs a new pool before outcomes can be added")
        index = self.index_of(fingerprint)
        one_hot = outcome_one_hot(int(outcome))
        count = self._counts[index]
        self._mean_one_hot[index] = (count * self._mean_one_hot[index] + one_hot) / (count + 1)
        self._counts[index] = count + 1
        self._ready[index] = True
        self._processed = None
        self.outcomes_added += 1

    def add_outcomes(self, pairs) -> int:
        added = 0
        for fingerprint, outcome in pairs:
            self.add_outcome(fingerprint, outcome)
            added += 1
        return added

    def outcome_record(self, fingerprint: str) -> dict:
        index = self.index_of(fingerprint)
        mean = self._mean_one_hot[index]
        return {
            "count": int(self._counts[index]),
            "mean_one_hot": mean.tolist(),
            "z_bar": float(mean @ _AGGREGATION),
            "ready": bool(self._ready[index]),
        }

    # -- process (S08, S12, S13, S14, S15, S19) ------------------------------

    def process(self, *, alpha: float, td_lambda: float = 1.0, gae_lambda: float = 1.0) -> ProcessedRows:
        if self.need_pool:
            raise Phase18SetupError("the buffer needs a new pool before it can be processed")
        if float(td_lambda) != 1.0 or float(gae_lambda) != 1.0:
            raise Phase18SetupConfigError(
                "process implements the flat advantage, which equals the published "
                f"recursion only at lambda = 1.0; refused td_lambda={td_lambda}, gae_lambda={gae_lambda}"
            )
        indices = self.ready_indices()
        excluded = len(self._samples) - indices.size
        if indices.size == 0:
            raise Phase18SetupError("no setup received a completed outcome in this period")

        rows = [self._samples[i] for i in indices]
        value_target = self._mean_one_hot[indices]                          # [R, 3]
        z_bar = value_target @ _AGGREGATION                                 # [R]
        wdl_logits = np.stack([r.wdl_logits for r in rows]).astype(np.float64)   # [R, 40, 3]
        expected = expected_value(softmax(wdl_logits))                      # [R, 40]
        information = np.stack([r.suffix_information for r in rows]).astype(np.float64)   # [R, 40]
        predicted = np.stack([r.entropy_prediction for r in rows]).astype(np.float64)     # [R, 40]

        outcome_term = z_bar[:, None] - expected
        entropy_residual = information - ENTROPY_NORMALIZER * predicted
        advantage = outcome_term + float(alpha) * entropy_residual
        entropy_target = information / ENTROPY_NORMALIZER
        counts = self._counts[indices]

        flat_outcome = outcome_term.reshape(-1)
        flat_entropy = (float(alpha) * entropy_residual).reshape(-1)
        flat_advantage = advantage.reshape(-1)
        telemetry = {
            "ready_rows": int(indices.size),
            "excluded_zero_outcome_rows": int(excluded),
            "prefix_rows": int(indices.size * SETUP_PREFIXES),
            "alpha": float(alpha),
            "td_lambda": float(td_lambda),
            "gae_lambda": float(gae_lambda),
            "outcome_count": {
                "mean": float(counts.mean()),
                "median": float(np.median(counts)),
                "min": int(counts.min()),
                "max": int(counts.max()),
                "fraction_single_outcome": float((counts == 1).mean()),
            },
            "z_bar_mean": float(z_bar.mean()),
            "z_bar_std": float(z_bar.std()),
            "outcome_variance_per_setup_mean": float(
                np.mean(1.0 - (value_target[:, 0] ** 2 + value_target[:, 1] ** 2 + value_target[:, 2] ** 2))
            ),
            "advantage_terms": _term_telemetry(flat_outcome, flat_entropy, flat_advantage),
            "information_mean_nats": float(information.mean()),
            "information_prefix0_mean_nats": float(information[:, 0].mean()),
            "entropy_prediction_mean": float(predicted.mean()),
            "entropy_residual_mean_nats": float(entropy_residual.mean()),
            "normalized_residual_abs_mean": float(np.abs(entropy_target - predicted).mean()),
            "expected_value_mean": float(expected.mean()),
        }
        self._processed = ProcessedRows(
            indices=indices,
            advantage=advantage.astype(np.float32),
            outcome_term=outcome_term.astype(np.float32),
            entropy_residual=entropy_residual.astype(np.float32),
            entropy_target=entropy_target.astype(np.float32),
            value_target=value_target.astype(np.float32),
            z_bar=z_bar.astype(np.float32),
            expected_values=expected.astype(np.float32),
            alpha=float(alpha),
            telemetry=telemetry,
        )
        return self._processed

    # -- minibatches (S06, S19, S26) ------------------------------------------

    def minibatches(self, batch_size: int, *, seed: int):
        if self._processed is None:
            raise Phase18SetupError("call process() before sampling minibatches")
        if int(batch_size) < 1:
            raise Phase18SetupError("batch_size must be positive")
        processed = self._processed
        ready = processed.indices
        order = np.random.RandomState(int(seed) % (2**32)).permutation(ready.size)
        for start in range(0, ready.size, int(batch_size)):
            position = order[start : start + int(batch_size)]
            yield self._batch(processed, position)

    def _batch(self, processed: ProcessedRows, position: np.ndarray) -> SetupBatch:
        rows = [self._samples[processed.indices[p]] for p in position]
        count = len(rows)
        tokens = np.zeros((count, SETUP_PREFIXES), dtype=np.int64)
        for row_index, sample in enumerate(rows):
            played = np.asarray(sample.played_canonical, dtype=np.int64)
            network = reflect_tokens(played) if sample.reflected else played
            if not np.array_equal(network, np.asarray(sample.network_tokens, dtype=np.int64)):
                raise Phase18SetupError(
                    f"setup {sample.content_fingerprint[:16]}: the played board flipped back does "
                    "not reproduce the recorded network tokens; refusing to gather log-probabilities"
                )
            tokens[row_index] = network
        sequence = np.full((count, SETUP_SEQUENCE_LENGTH), START_TOKEN, dtype=np.int64)
        sequence[:, 1:] = tokens
        masks = np.stack([r.legal_masks for r in rows])
        log_probs = np.stack([r.behavior_log_probs for r in rows]).astype(np.float32)
        selected = np.stack([r.behavior_selected_log_prob for r in rows]).astype(np.float32)
        device = self.device
        return SetupBatch(
            sequence=torch.as_tensor(sequence, device=device),
            tokens=torch.as_tensor(tokens, device=device),
            masks=torch.as_tensor(masks, device=device),
            behavior_log_probs=torch.as_tensor(log_probs, device=device),
            behavior_selected_log_prob=torch.as_tensor(selected, device=device),
            advantage=torch.as_tensor(processed.advantage[position], device=device),
            value_target=torch.as_tensor(processed.value_target[position], device=device),
            entropy_target=torch.as_tensor(processed.entropy_target[position], device=device),
            outcome_counts=torch.as_tensor(self._counts[processed.indices[position]], device=device),
            fingerprints=[r.content_fingerprint for r in rows],
            count=count,
        )

    # -- retention (S21) -------------------------------------------------------

    def filter(self, current_period: int) -> dict:
        expiration = self._period_added + self.storage_duration
        keep = expiration >= int(current_period)
        dropped = int((~keep).sum())
        self._samples = [sample for sample, flag in zip(self._samples, keep) if flag]
        self._period_added = self._period_added[keep]
        self._counts = self._counts[keep]
        self._mean_one_hot = self._mean_one_hot[keep]
        self._ready = self._ready[keep]
        self._lookup = {sample.content_fingerprint: index for index, sample in enumerate(self._samples)}
        self._processed = None
        self.need_pool = True
        return {"current_period": int(current_period), "dropped": dropped, "rows": len(self._samples)}

    def telemetry(self) -> dict:
        counts = self._counts
        return {
            "buffer_version": self.version,
            "rows": len(self._samples),
            "ready_rows": self.ready_count(),
            "zero_outcome_rows": int((counts == 0).sum()) if counts.size else 0,
            "pools_added": self.pools_added,
            "outcomes_added": self.outcomes_added,
            "duplicates_collapsed_total": self.duplicates_collapsed_total,
            "attribution_failures": self.attribution_failures,
            "storage_duration": self.storage_duration,
            "need_pool": self.need_pool,
            "snapshot_iterations_present": sorted({int(s.snapshot_iteration) for s in self._samples}),
        }


def mark_most_recent_appearance(values, timestamps) -> list:
    """One True per unique value, at its maximal-timestamp appearance (first
    such appearance in list order). Transcribed from the published helper."""
    timestamps = [int(t) for t in np.asarray(timestamps).reshape(-1)]
    if len(values) != len(timestamps):
        raise Phase18SetupError("values and timestamps disagree in length")
    latest: dict = {}
    for value, timestamp in zip(values, timestamps):
        if value not in latest or timestamp > latest[value]:
            latest[value] = timestamp
    seen: set = set()
    mask = []
    for value, timestamp in zip(values, timestamps):
        if timestamp == latest[value] and value not in seen:
            seen.add(value)
            mask.append(True)
        else:
            mask.append(False)
    return mask


def _term_telemetry(outcome: np.ndarray, entropy: np.ndarray, total: np.ndarray) -> dict:
    """S13's required telemetry: both advantage terms separately."""
    quantiles = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)

    def describe(values: np.ndarray) -> dict:
        return {
            "mean": float(values.mean()),
            "abs_mean": float(np.abs(values).mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "quantiles": {str(q): float(np.quantile(values, q)) for q in quantiles},
        }

    def correlation(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() == 0.0 or b.std() == 0.0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "outcome_term": describe(outcome),
        "entropy_term": describe(entropy),
        "total_advantage": describe(total),
        "entropy_to_outcome_abs_ratio": float(np.abs(entropy).mean() / max(np.abs(outcome).mean(), 1e-12)),
        "outcome_term_correlation_with_total": correlation(outcome, total),
        "entropy_term_correlation_with_total": correlation(entropy, total),
    }


__all__ = [
    "ProcessedRows",
    "SetupBatch",
    "SetupBuffer",
    "expected_value",
    "mark_most_recent_appearance",
    "outcome_one_hot",
    "softmax",
]
