"""Phase 15 Agent 1: wiring the pieces into the deliverables.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` sections 8-13.

This module owns the *order* of the work and nothing else: every rule it
applies lives in :mod:`.contract`, :mod:`.train`, :mod:`.calibration`,
:mod:`.metrics` or :mod:`.checkpoint`. It exists so the runner script stays
a command-line interface rather than a second implementation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ...model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
from ...model.checkpoint import load_checkpoint
from .calibration import decide, fit_temperature
from .contract import (
    RECIPE,
    SPECIALISTS,
    SPECIALIST_SOURCE,
    SPLIT_CALIBRATION,
    SPLIT_DEVELOPMENT,
    SPLIT_TRAIN,
    Phase15Error,
)
from .features import build_cache, cache_path, load_cache
from .heads import Phase15BeliefSpecialist
from .metrics import baseline_probabilities, evaluate, paired_comparison, uniform_reference
from .storage import load_split
from .train import TrainConfig, predict_probabilities, train_specialist

#: The pipeline identity.
PIPELINE_VERSION = "phase15_agent01_pipeline_v1"

#: The accepted Phase 11B Agent 1C reference, and the backbone it needs.
AGENT1C_CHECKPOINT = Path("checkpoints/phase11b/agent01_1c_final_block_plus_mlp.pt")
AGENT1C_BACKBONE = Path("checkpoints/phase11b/phase9_c1_readonly_copy.pt")


class Phase15PipelineError(Phase15Error):
    """A pipeline stage could not run."""


def load_policy(path: "Path | str", *, device: str = "cpu"):
    """A frozen policy/value backbone, read-only."""
    model, metadata = load_checkpoint(
        Path(path),
        device=torch.device(device),
        dtype=torch.float32,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=candidate_config("C1"),
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, metadata


def ensure_caches(
    policy_model,
    specialist_id: str,
    splits: dict,
    cache_root: "Path | str",
    *,
    device: str = "cpu",
    rebuild: bool = False,
    progress=None,
) -> dict:
    """Build (or reuse) this backbone's frozen-prefix cache for each split."""
    records = {}
    for split, data in splits.items():
        path = cache_path(cache_root, specialist_id, split)
        if rebuild or not path.is_file():
            records[split] = build_cache(
                policy_model, data, path, device=device, progress=progress
            )
        else:
            existing = load_cache(path)
            if int(existing.shape[0]) != int(data["samples"]):
                records[split] = build_cache(
                    policy_model, data, path, device=device, progress=progress
                )
            else:
                records[split] = {
                    "path": str(path),
                    "split": split,
                    "samples": int(existing.shape[0]),
                    "reused": True,
                }
            del existing
    return records


def train_one(
    specialist_id: str,
    source_identity: dict,
    splits: dict,
    cache_root: "Path | str",
    *,
    device: str = "cpu",
    batch_size: "int | None" = None,
    batch_size_reason: "str | None" = None,
    rebuild_caches: bool = False,
    progress=None,
) -> dict:
    """Train one specialist end to end: cache, fit, calibrate, evaluate.

    Returns everything the checkpoint writer and the report need. The
    frozen backbone is loaded here and its digest re-checked afterwards, so
    "P18 unchanged by training" is measured rather than assumed.
    """
    from ...training.phase9_behavior import state_dict_digest

    source_id = SPECIALIST_SOURCE[specialist_id]
    policy_model, _metadata = load_policy(
        source_identity["phase15_copy_path"], device=device
    )
    digest_before = state_dict_digest(policy_model)
    if digest_before != source_identity["model_state_digest"]:
        raise Phase15PipelineError(
            f"{source_id} loaded as {digest_before[:16]}, the binding records "
            f"{source_identity['model_state_digest'][:16]}"
        )

    caches = ensure_caches(
        policy_model,
        specialist_id,
        splits,
        cache_root,
        device=device,
        rebuild=rebuild_caches,
        progress=progress,
    )
    train_cache = load_cache(caches[SPLIT_TRAIN]["path"])
    dev_cache = load_cache(caches[SPLIT_DEVELOPMENT]["path"])
    calibration_cache = load_cache(caches[SPLIT_CALIBRATION]["path"])

    config = TrainConfig(specialist_id=specialist_id, device=device)
    if batch_size is not None and int(batch_size) != config.batch_size:
        config.batch_size_changed_from = config.batch_size
        config.batch_size = int(batch_size)
        config.batch_size_change_reason = (
            batch_size_reason or "memory/throughput safety; the statistical recipe is unchanged"
        )

    model = Phase15BeliefSpecialist.from_policy(
        policy_model, specialist_id=specialist_id
    )
    record = train_specialist(
        model,
        train_cache,
        splits[SPLIT_TRAIN],
        dev_cache,
        splits[SPLIT_DEVELOPMENT],
        config,
        policy_model=policy_model,
        progress=progress,
    )

    digest_after = state_dict_digest(policy_model)
    record["source_unchanged"] = {
        "source_id": source_id,
        "model_state_digest_before": digest_before,
        "model_state_digest_after": digest_after,
        "unchanged": digest_before == digest_after,
    }
    if digest_before != digest_after:
        raise Phase15PipelineError(
            f"{source_id} changed during training: {digest_before[:16]} -> "
            f"{digest_after[:16]}"
        )

    # -- calibration, on the calibration split only ------------------------
    calibration_logits = _logits(
        model, calibration_cache, splits[SPLIT_CALIBRATION], device=device
    )
    fit = fit_temperature(
        calibration_logits,
        np.asarray(splits[SPLIT_CALIBRATION]["true_rank"], dtype=np.int64),
    )

    development = splits[SPLIT_DEVELOPMENT]
    dev_baseline = baseline_probabilities(development)
    raw = predict_probabilities(
        model, dev_cache, development, device=device, temperature=1.0
    )
    calibrated = predict_probabilities(
        model, dev_cache, development, device=device, temperature=fit["temperature"]
    )
    raw_metrics = evaluate(raw, development, baseline=dev_baseline)
    calibrated_metrics = evaluate(calibrated, development, baseline=dev_baseline)
    decision = decide(raw_metrics, calibrated_metrics)
    model.set_temperature(fit["temperature"] if decision["keep_calibrated"] else 1.0)

    return {
        "specialist_id": specialist_id,
        "source_id": source_id,
        "model": model,
        "policy_model": policy_model,
        "caches": caches,
        "training": record,
        "calibration": {**fit, **decision, "applied_temperature": model.temperature},
        "development_raw": raw_metrics,
        "development_calibrated": calibrated_metrics,
        "development_probabilities": (
            calibrated if decision["keep_calibrated"] else raw
        ),
    }


@torch.no_grad()
def _logits(model, cache, data, *, device: str = "cpu", batch_size: int = 512):
    """`float64[M, 12]` raw logits, in the corpus's own piece order."""
    from .train import sample_batches

    model.eval()
    offsets = np.asarray(data["piece_offset"], dtype=np.int64)
    out = np.empty((int(data["pieces"]), 12), dtype=np.float64)
    rows = np.arange(int(data["samples"]), dtype=np.int64)
    for block, token_rows, token_squares, _labels in sample_batches(
        data, rows, batch_size
    ):
        tokens = torch.from_numpy(np.array(cache[block], dtype=np.float32, copy=True)).to(
            device
        )
        gather = (
            torch.from_numpy(token_rows).to(device),
            torch.from_numpy(token_squares).to(device),
        )
        values = model(tokens, gather).detach().cpu().to(torch.float64).numpy()
        cursor = 0
        for row in block:
            width = int(offsets[row + 1] - offsets[row])
            out[offsets[row] : offsets[row + 1]] = values[cursor : cursor + width]
            cursor += width
    return out


# ---------------------------------------------------------------------------
# The Phase 11B Agent 1C reference
# ---------------------------------------------------------------------------


def agent1c_reference(
    development: dict, *, device: str = "cpu", batch_size: int = 256
) -> dict:
    """Score the surviving Agent 1C belief model on the NEW development split.

    The old contaminated development result is *not* the comparison set:
    section 11 asks for 1C measured on this corpus. 1C is loaded through the
    accepted Phase 11B objects and reads the accepted Phase 9 backbone's
    penultimate representation, exactly as it was trained to.
    """
    from ..phase11b.features import LAYER_PENULTIMATE, encode_batch
    from ..phase11b.heads import CANDIDATE_1C, build_candidate

    if not AGENT1C_CHECKPOINT.is_file():
        raise Phase15PipelineError(f"no Agent 1C checkpoint at {AGENT1C_CHECKPOINT}")
    if not AGENT1C_BACKBONE.is_file():
        raise Phase15PipelineError(
            f"no Phase 9 read-only backbone at {AGENT1C_BACKBONE}; Agent 1C cannot "
            "be scored without the encoder it was attached to"
        )
    backbone, _metadata = load_policy(AGENT1C_BACKBONE, device=device)
    payload = torch.load(AGENT1C_CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_candidate(CANDIDATE_1C, backbone)
    model.load_state_dict(payload["state_dict"])
    model.to(torch.device(device)).eval()

    offsets = np.asarray(development["piece_offset"], dtype=np.int64)
    squares = np.asarray(development["perspective_square"], dtype=np.int64)
    samples = int(development["samples"])
    out = np.empty((int(development["pieces"]), 12), dtype=np.float64)
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, samples, batch_size):
            stop = min(start + batch_size, samples)
            tokens = encode_batch(
                backbone,
                np.asarray(development["observations"][start:stop]),
                LAYER_PENULTIMATE,
            )
            features = model.encode(tokens)
            for row in range(start, stop):
                low, high = int(offsets[row]), int(offsets[row + 1])
                if high <= low:  # pragma: no cover - every sample has a piece
                    continue
                logits = model.head(features[row - start, squares[low:high]])
                out[low:high] = torch.softmax(
                    logits.detach().cpu().to(torch.float64), dim=1
                ).numpy()
    return {
        "reference_id": "phase11b_agent01_1c_final_block_plus_mlp",
        "checkpoint": str(AGENT1C_CHECKPOINT),
        "backbone": str(AGENT1C_BACKBONE),
        "trained_on": payload.get("corpus_version"),
        "old_development_r_ce": (payload.get("dev_metrics") or {}).get("r_ce"),
        "old_development_note": (
            "measured on phase11b_common_corpus_v1, whose Blue setups are "
            "mis-oriented; reported only to identify the artifact, never as the "
            "comparison set"
        ),
        "probabilities": out,
        "seconds": round(time.perf_counter() - started, 3),
    }


# ---------------------------------------------------------------------------
# The report block
# ---------------------------------------------------------------------------


def comparison_block(
    trained: dict, reference: dict, development: dict, dev_baseline: np.ndarray
) -> dict:
    """B18 vs B24 vs Agent 1C vs the baseline, on the same development pieces."""
    reference_metrics = evaluate(
        reference["probabilities"], development, baseline=dev_baseline
    )
    block = {
        "development_positions": int(development["samples"]),
        "development_pieces": int(development["pieces"]),
        "remaining_count_baseline": {
            "ce": float(reference_metrics["baseline_ce"]),
            "top1": float(reference_metrics["baseline_top1"]),
            "brier": float(reference_metrics["baseline_brier"]),
        },
        "uniform_reference": {
            key: uniform_reference(development)[key] for key in ("ce", "r_ce", "top1")
        },
        "agent1c_reference": {
            **{
                key: reference_metrics[key]
                for key in (
                    "ce",
                    "r_ce",
                    "r_ce_ci95",
                    "top1",
                    "brier",
                    "expected_calibration_error",
                    "maximum_calibration_error",
                )
            },
            "reference_id": reference["reference_id"],
            "old_development_r_ce": reference["old_development_r_ce"],
        },
        "specialists": {},
        "paired": {},
    }
    for specialist_id in SPECIALISTS:
        result = trained[specialist_id]
        block["specialists"][specialist_id] = {
            key: result["development_calibrated" if result["calibration"]["keep_calibrated"] else "development_raw"][key]
            for key in (
                "ce",
                "r_ce",
                "r_ce_ci95",
                "top1",
                "brier",
                "expected_calibration_error",
                "maximum_calibration_error",
            )
        }
        block["paired"][f"{specialist_id}_vs_agent1c"] = paired_comparison(
            result["development_probabilities"], reference["probabilities"], development
        )
    block["paired"]["b18_vs_b24"] = paired_comparison(
        trained["b18"]["development_probabilities"],
        trained["b24"]["development_probabilities"],
        development,
    )
    return block


def load_corpus(root: "Path | str") -> dict:
    """All three splits, with labels, ready for training and scoring."""
    return {
        split: load_split(root, split, labels=True)
        for split in (SPLIT_TRAIN, SPLIT_CALIBRATION, SPLIT_DEVELOPMENT)
    }


__all__ = [
    "AGENT1C_BACKBONE",
    "AGENT1C_CHECKPOINT",
    "PIPELINE_VERSION",
    "Phase15PipelineError",
    "agent1c_reference",
    "comparison_block",
    "ensure_caches",
    "load_corpus",
    "load_policy",
    "train_one",
]
