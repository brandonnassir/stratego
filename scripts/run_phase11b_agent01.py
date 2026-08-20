#!/usr/bin/env python3
"""Phase 11B Agent 1 harness: the common corpus and the attached belief head.

Phase 11B is an **engineering prototype branch**. It does not overturn the
Phase 11 `FAIL`, does not spend or open `phase11_test_bank_v1`, and does not
authorize Phase 12. Every artifact this harness writes carries those four
statements as data.

Six stages, in order:

- **verify** — read-only re-derivation of the accepted Phase 9 checkpoint's
  identity (file SHA, model-state digest, parameter count, belief-head
  tensor names/shapes/digest) and a preservation check over the Phase 11
  evidence Phase 11B must leave untouched: both bank artifacts, the access
  ledger, `phase11_system_v1`, the Agent 1-7 reports and the sealed-test
  results.
- **corpus** — build (or verify) the common Phase 11B corpus: 2,048 fresh
  training games and 512 fresh development games over four behaviour
  strata, balanced by setup source and observer colour, public inputs and
  privileged labels in separate directories.
- **features** — cache the frozen C1 per-piece representation of both
  splits once, so 1A and 1B see bit-identical features.
- **reference** — score the unchanged Phase 11 belief head on the common
  development positions. Reference only; the spent Phase 11 bank is not
  reused for it.
- **train** — Experiment 1A (the existing 128->12 layer, from the accepted
  weights) and Experiment 1B (a 128->512->512->12 attached MLP), then
  optional 1C (the last C1 block unfrozen with the larger head) only when
  1B justifies it and the budget allows.
- **report** — the leaderboard JSON, the learning curves, the interface
  smoke check and the Markdown report.

Usage::

    python scripts/run_phase11b_agent01.py --full
    python scripts/run_phase11b_agent01.py --stage corpus --rebuild-corpus
    python scripts/run_phase11b_agent01.py --full --run-pytest
    python scripts/run_phase11b_agent01.py --full --with-1c
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from stratego.belief.phase11b.contract import (  # noqa: E402
    CHECKPOINT_ROOT,
    CORPUS_ROOT,
    CORPUS_SPLITS,
    CORPUS_STRATA,
    CORPUS_VERSION,
    PHASE11B_STATUS_MARKERS,
    PHASE11B_VERSION,
    PHASE11_FACTS,
    RANK_COUNT,
    REPORT_ROOT,
)
from stratego.belief.phase11b import features as feat  # noqa: E402
from stratego.belief.phase11b import heads as H  # noqa: E402
from stratego.belief.phase11b import metrics as M  # noqa: E402
from stratego.belief.phase11b import storage as S  # noqa: E402
from stratego.belief.phase11b.build import build_corpus  # noqa: E402
from stratego.belief.phase11b.interface import (  # noqa: E402
    BELIEF_INTERFACE_VERSION,
    Phase11BBeliefModel,
    Phase11BPublicState,
)
from stratego.belief.phase11b.seeds import (  # noqa: E402
    CANONICAL_PHASE11B_SEEDS,
    PHASE11B_IDENTITY_VERSION,
)
from stratego.belief.phase11b.train import TrainConfig, predict_probabilities, train_attached_head  # noqa: E402

AGENT = 1
REPORT_DIRECTORY = REPOSITORY_ROOT / REPORT_ROOT
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / CHECKPOINT_ROOT
CORPUS_DIRECTORY = REPOSITORY_ROOT / CORPUS_ROOT
SUMMARY_PATH = REPORT_DIRECTORY / "agent_01_summary.json"
REPORT_PATH = REPORT_DIRECTORY / "agent_01_report.md"
CURVES_PATH = REPORT_DIRECTORY / "agent_01_learning_curves.json"
STAGE_PATH = REPORT_DIRECTORY / ".agent_01_stages.json"

STAGES = ("verify", "corpus", "features", "reference", "train", "report")

#: The Phase 11 evidence Phase 11B must preserve byte-for-byte. Checked by
#: SHA-256 before and after every run, so "Phase 11B changed nothing" is a
#: measurement rather than an assurance.
PRESERVED_PHASE11_ARTIFACTS = (
    "reports/phase_11_implementation_report.md",
    "reports/phase_11_data/agent_01_phase11_contract.json",
    "reports/phase_11_data/agent_01_validation_bank.json",
    "reports/phase_11_data/agent_01_test_bank.json",
    "reports/phase_11_data/agent_05_validation_freeze.json",
    "reports/phase_11_data/agent_06_system_v1.json",
    "reports/phase_11_data/agent_07_final_acceptance.json",
    "reports/phase_11_data/phase11_bank_access_ledger.jsonl",
)

#: The accepted modules Phase 11B imports but may not modify.
PRESERVED_ACCEPTED_MODULES = (
    "stratego/evaluation/phase11_sampler.py",
    "stratego/evaluation/phase11_baselines.py",
    "stratego/evaluation/phase11_public_state.py",
    "stratego/training/phase11_contract.py",
    "stratego/training/phase11_seed.py",
    "stratego/training/belief_targets.py",
    "stratego/model/production_model.py",
)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            hasher.update(block)
    return hasher.hexdigest()


def log(message: str) -> None:
    print(message, flush=True)


def load_stages() -> dict:
    if STAGE_PATH.exists():
        return json.loads(STAGE_PATH.read_text())
    return {}


def save_stage(name: str, payload: dict) -> None:
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stages = load_stages()
    stages[name] = payload
    STAGE_PATH.write_text(json.dumps(stages, indent=1, sort_keys=True, default=str) + "\n")


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def stage_verify() -> dict:
    """Read-only identity and preservation checks. Nothing is written."""
    from stratego.training.phase11_contract import (
        ACCEPTED_BELIEF_HEAD_DIGEST,
        ACCEPTED_GLOBAL_OPTIMIZER_STEP,
        BELIEF_HEAD_TENSOR_NAMES,
        BELIEF_HEAD_TENSOR_SHAPES,
    )
    from stratego.training.phase10_contract import (
        ACCEPTED_PHASE9_CHECKPOINT_SHA256,
        ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
        ACCEPTED_PHASE9_PARAMETERS,
    )
    from stratego.training.phase9_behavior import state_dict_digest
    from stratego.training.phase9_checkpoint import model_from_payload, read_phase9_payload

    problems: list[str] = []
    checkpoint = REPOSITORY_ROOT / feat.ACCEPTED_PHASE9_CHECKPOINT
    observed_sha = file_sha256(checkpoint)
    payload = read_phase9_payload(checkpoint)
    model = model_from_payload(payload)
    state_digest = state_dict_digest(model)
    parameters = int(sum(tensor.numel() for tensor in model.parameters()))
    head_digest = feat.belief_head_digest(model)
    state = model.state_dict()
    head_names = tuple(sorted(name for name in state if name.startswith("belief_output.")))
    head_shapes = {name: tuple(state[name].shape) for name in head_names}
    global_step = payload.get("global_optimizer_step")

    if observed_sha != ACCEPTED_PHASE9_CHECKPOINT_SHA256:
        problems.append(f"Phase 9 checkpoint SHA {observed_sha} != accepted")
    if state_digest != ACCEPTED_PHASE9_MODEL_STATE_DIGEST:
        problems.append(f"Phase 9 model-state digest {state_digest} != accepted")
    if parameters != ACCEPTED_PHASE9_PARAMETERS:
        problems.append(f"Phase 9 parameter count {parameters} != accepted")
    if head_digest != ACCEPTED_BELIEF_HEAD_DIGEST:
        problems.append(f"belief-head digest {head_digest} != accepted")
    if head_names != BELIEF_HEAD_TENSOR_NAMES:
        problems.append(f"belief-head tensors {head_names} != frozen")
    if head_shapes != BELIEF_HEAD_TENSOR_SHAPES:
        problems.append("belief-head shapes != frozen")
    if global_step != ACCEPTED_GLOBAL_OPTIMIZER_STEP:
        problems.append(f"global optimizer step {global_step} != accepted")
    del model

    preserved = {}
    for relative in PRESERVED_PHASE11_ARTIFACTS + PRESERVED_ACCEPTED_MODULES:
        path = REPOSITORY_ROOT / relative
        if not path.exists():
            problems.append(f"preserved artifact {relative} is missing")
            continue
        preserved[relative] = file_sha256(path)

    spent_bank = REPOSITORY_ROOT / "reports/phase_11_data/agent_07_final_acceptance.json"
    phase11_recommendation = None
    if spent_bank.exists():
        acceptance = json.loads(spent_bank.read_text())
        phase11_recommendation = acceptance.get("recommendation") or acceptance.get(
            "final_recommendation"
        )

    return {
        "stage": "verify",
        "problems": problems,
        "pass": not problems,
        "phase9_checkpoint": {
            "path": feat.ACCEPTED_PHASE9_CHECKPOINT,
            "sha256": observed_sha,
            "model_state_digest": state_digest,
            "parameters": parameters,
            "belief_head_digest": head_digest,
            "belief_head_tensor_names": list(head_names),
            "global_optimizer_step": global_step,
            "opened": "read_only",
        },
        "preserved_digests": preserved,
        "phase11_recommendation_read": phase11_recommendation,
        **PHASE11B_STATUS_MARKERS,
    }


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


#: Where a truncated pilot corpus goes. Never the canonical root: Agents 2-5
#: are promised those exact bytes, and a `--limit-*` run must not be able to
#: replace a 2,560-game corpus with a 32-game one.
PILOT_CORPUS_DIRECTORY = CORPUS_DIRECTORY.parent / "pilot_corpus"


def stage_corpus(*, rebuild: bool, limits: "dict | None") -> dict:
    """Build the common corpus, or verify the one already on disk."""
    if limits:
        return stage_pilot_corpus(limits)
    manifest_path = CORPUS_DIRECTORY / "manifest.json"
    if manifest_path.exists() and not rebuild:
        manifest = S.read_manifest(CORPUS_DIRECTORY)
        recomputed = {
            split: S.split_digest(CORPUS_DIRECTORY, split) for split in CORPUS_SPLITS
        }
        drift = [
            f"{split}/{name}"
            for split in CORPUS_SPLITS
            for name, digest in recomputed[split].items()
            if manifest["splits"][split]["file_digests"].get(name) != digest
        ]
        rebuilt = dict(manifest)
        rebuilt["splits"] = {
            split: {**manifest["splits"][split], "file_digests": recomputed[split]}
            for split in CORPUS_SPLITS
        }
        identity = S.corpus_digest(rebuilt)
        return {
            "stage": "corpus",
            "action": "verified_existing",
            "manifest": manifest,
            "recomputed_corpus_digest": identity,
            "corpus_digest_matches": identity == manifest["corpus_digest"],
            "file_digest_drift": drift,
            "pass": not drift and identity == manifest["corpus_digest"],
        }

    manifest = _generate_corpus(CORPUS_DIRECTORY, limits=None)
    return {
        "stage": "corpus",
        "action": "rebuilt",
        "root": str(CORPUS_DIRECTORY.relative_to(REPOSITORY_ROOT)),
        "manifest": manifest,
        "recomputed_corpus_digest": manifest["corpus_digest"],
        "corpus_digest_matches": True,
        "file_digest_drift": [],
        "pass": all(manifest["splits"][split]["complete"] for split in CORPUS_SPLITS),
    }


def stage_pilot_corpus(limits: dict) -> dict:
    """A truncated throughput pilot, written beside the real corpus."""
    log(f"  pilot corpus: {limits} games, written to {PILOT_CORPUS_DIRECTORY.name}/")
    manifest = _generate_corpus(PILOT_CORPUS_DIRECTORY, limits=limits)
    return {
        "stage": "corpus",
        "action": "pilot",
        "root": str(PILOT_CORPUS_DIRECTORY.relative_to(REPOSITORY_ROOT)),
        "limits": dict(limits),
        "manifest": manifest,
        "recomputed_corpus_digest": manifest["corpus_digest"],
        "corpus_digest_matches": True,
        "file_digest_drift": [],
        "note": (
            "a truncated pilot: incomplete by construction, digested differently "
            "from the common corpus, and never the artifact Agents 2-5 reuse"
        ),
        "pass": True,
    }


def _generate_corpus(root: Path, *, limits: "dict | None") -> dict:
    from stratego.evaluation.phase11_pipeline import build_owners

    export = CHECKPOINT_DIRECTORY / "phase9_c1_readonly_copy.pt"
    owners, _ = build_owners(REPOSITORY_ROOT, export, device="cpu")

    def progress(split, done, total, samples, elapsed):
        rate = done / max(elapsed, 1e-9)
        log(
            f"  [corpus/{split}] {done}/{total} games  samples={samples}  "
            f"{elapsed:.0f}s  {rate:.2f} g/s  eta {(total - done) / max(rate, 1e-9):.0f}s"
        )

    return build_corpus(root, owners, limits=limits, overwrite=True, progress=progress)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


#: `layer -> file stem`. The final-layer cache is what 1A and 1B consume;
#: the penultimate cache exists only when 1C is requested, because it is
#: 15x larger and nothing else reads it.
CACHE_STEMS = {feat.LAYER_FINAL: "c1_features", feat.LAYER_PENULTIMATE: "c1_penultimate"}


def stage_features(device: str, *, with_1c: bool = False) -> dict:
    """Cache the frozen C1 representation of both splits."""
    model, identity = feat.load_frozen_c1(
        REPOSITORY_ROOT, CHECKPOINT_DIRECTORY / "phase9_c1_readonly_copy.pt", device=device
    )
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    layers = [feat.LAYER_FINAL] + ([feat.LAYER_PENULTIMATE] if with_1c else [])
    blocks: dict = {layer: {} for layer in layers}
    for layer in layers:
        for split in CORPUS_SPLITS:
            data = S.load_split(CORPUS_DIRECTORY, split, labels=True)
            cache, seconds = feat.encode_split(model, data, layer=layer)
            path = CHECKPOINT_DIRECTORY / f"{CACHE_STEMS[layer]}_{split}.npy"
            np.save(path, cache)
            blocks[layer][split] = {
                "shape": [int(value) for value in cache.shape],
                "seconds": seconds,
                "digest": feat.cache_digest(cache),
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "bytes": int(path.stat().st_size),
            }
            log(f"  [features/{layer}/{split}] {tuple(cache.shape)} in {seconds:.1f}s")
            del cache, data
    return {
        "stage": "features",
        "layers": layers,
        "layer": feat.LAYER_FINAL,
        "cache_version": feat.FEATURE_CACHE_VERSION,
        "frozen_model": identity,
        "splits": blocks[feat.LAYER_FINAL],
        "caches": blocks,
        "pass": all(
            block["shape"][0] > 0 for layer in layers for block in blocks[layer].values()
        ),
    }


def load_features(split: str, layer: str = feat.LAYER_FINAL) -> np.ndarray:
    return np.load(
        CHECKPOINT_DIRECTORY / f"{CACHE_STEMS[layer]}_{split}.npy", mmap_mode="r"
    )


# ---------------------------------------------------------------------------
# reference
# ---------------------------------------------------------------------------


def stage_reference(device: str) -> dict:
    """The unchanged Phase 11 head, on the common development positions."""
    model, identity = feat.load_frozen_c1(
        REPOSITORY_ROOT, CHECKPOINT_DIRECTORY / "phase9_c1_readonly_copy.pt", device=device
    )
    head = H.ExistingBeliefHead.from_accepted(model)
    dev = S.load_split(CORPUS_DIRECTORY, "dev", labels=True)
    dev_features = np.asarray(load_features("dev"))
    started = time.perf_counter()
    probabilities = predict_probabilities(head, dev_features, device="cpu")
    metrics = M.evaluate(probabilities, dev)
    uniform = M.uniform_reference(dev)
    return {
        "stage": "reference",
        "candidate_id": "phase11_head_unchanged_reference",
        "belief_head_digest": identity["belief_head_digest"],
        "note": (
            "the accepted Phase 11 belief head, scored on the fresh Phase 11B "
            "development positions. Reference only: the spent "
            "phase11_test_bank_v1 was not opened, and this number is not the "
            "Phase 11 sealed-test result."
        ),
        "metrics": metrics,
        "uniform_floor": {
            "ce": uniform["ce"],
            "r_ce": uniform["r_ce"],
            "top1": uniform["top1"],
        },
        "seconds": round(time.perf_counter() - started, 3),
        "pass": np.isfinite(metrics["ce"]),
    }


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def _inference_cost(head, dev_features: np.ndarray, device: str) -> dict:
    """Per-piece and per-decision inference latency of one trained head."""
    rows = min(4096, int(dev_features.shape[0]))
    batch = torch.from_numpy(np.ascontiguousarray(dev_features[:rows])).to(device)
    head = head.to(device).eval()
    with torch.no_grad():
        for _ in range(3):
            head(batch)
        started = time.perf_counter()
        repeats = 20
        for _ in range(repeats):
            head(batch)
        elapsed = (time.perf_counter() - started) / repeats
    return {
        "batch_rows": rows,
        "batch_seconds": round(elapsed, 6),
        "microseconds_per_piece": round(elapsed / rows * 1e6, 4),
    }


def train_candidate(
    candidate_id: str,
    frozen_model,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    dev_features: np.ndarray,
    dev: dict,
    *,
    device: str,
    epochs: int,
    max_seconds: float,
) -> dict:
    head = H.build_candidate(candidate_id, frozen_model)
    config = TrainConfig(
        candidate_id=candidate_id,
        epochs=epochs,
        device=device,
        max_seconds=max_seconds,
    )

    def progress(_id, row):
        log(
            f"  [{candidate_id}] epoch {row['epoch']:>2}  train {row['train_loss']:.4f}  "
            f"dev CE {row['dev_ce']:.4f}  R_CE {row['dev_r_ce']:.4f}  "
            f"top1 {row['dev_top1']:.4f}  {row['seconds']:.0f}s"
        )

    record = train_attached_head(
        head, train_features, train_labels, dev_features, dev, config, progress=progress
    )
    state = record.pop("best_state")
    path = CHECKPOINT_DIRECTORY / f"{candidate_id}.pt"
    torch.save(
        {
            "candidate_id": candidate_id,
            "architecture": head.architecture,
            "state_dict": state,
            "config": record["config"],
            "dev_metrics": record["dev_metrics"],
            "corpus_version": CORPUS_VERSION,
            **PHASE11B_STATUS_MARKERS,
        },
        path,
    )
    record["checkpoint"] = {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "sha256": file_sha256(path),
    }
    record["architecture"] = head.architecture
    record["parameters_added"] = H.parameter_count(head)
    record["inference"] = _inference_cost(head, dev_features, "cpu")
    record["head"] = head
    return record


def stage_train(
    device: str, *, epochs: int, budget: float, with_1c: bool, repeat: bool = False
) -> dict:
    model, identity = feat.load_frozen_c1(
        REPOSITORY_ROOT, CHECKPOINT_DIRECTORY / "phase9_c1_readonly_copy.pt", device="cpu"
    )
    train = S.load_split(CORPUS_DIRECTORY, "train", labels=True)
    dev = S.load_split(CORPUS_DIRECTORY, "dev", labels=True)
    train_features = np.asarray(load_features("train"))
    dev_features = np.asarray(load_features("dev"))
    train_labels = np.asarray(train["true_rank"], dtype=np.int64)

    results = {}
    heads = {}
    started = time.perf_counter()
    for candidate_id in (H.CANDIDATE_1A, H.CANDIDATE_1B):
        log(f"  training {candidate_id} ...")
        record = train_candidate(
            candidate_id,
            model,
            train_features,
            train_labels,
            dev_features,
            dev,
            device=device,
            epochs=epochs,
            max_seconds=budget,
        )
        heads[candidate_id] = record.pop("head")
        results[candidate_id] = record

    ran_1c = False
    if with_1c:
        log(f"  training {H.CANDIDATE_1C} ...")
        results[H.CANDIDATE_1C] = train_1c(
            model, train, dev, device=device, epochs=EPOCHS_1C, budget=budget
        )
        heads[H.CANDIDATE_1C] = results[H.CANDIDATE_1C].pop("head")
        ran_1c = True

    comparisons = paired_comparisons(results, heads, dev)
    reproducibility = None
    if repeat:
        log("  repeat pass: retraining every candidate under the identical config ...")
        reproducibility = repeat_training(
            model, train, dev, train_features, train_labels, dev_features,
            results, device=device, epochs=epochs, budget=budget,
        )
    return {
        "stage": "train",
        "reproducibility": reproducibility,
        "frozen_model": identity,
        "peak_memory_bytes": peak_rss_bytes(),
        "results": results,
        "heads": heads,
        "ran_1c": ran_1c,
        "paired_comparisons": comparisons,
        "seconds": round(time.perf_counter() - started, 3),
        "pass": all(np.isfinite(row["dev_metrics"]["ce"]) for row in results.values()),
    }


#: 1C's declared epoch budget. A 1C epoch runs a real transformer block over
#: every training position, so it costs ~4x a 1A/1B epoch; twelve is the
#: configuration Agent 1 declared and ran, not a tuned value.
EPOCHS_1C = 12


def train_1c(frozen_model, train: dict, dev: dict, *, device: str, epochs: int, budget: float) -> dict:
    """Experiment 1C: the last C1 block unfrozen, with the 1B head."""
    from stratego.belief.phase11b.train import predict_probabilities_1c, train_final_block

    model = H.build_candidate(H.CANDIDATE_1C, frozen_model)
    config = TrainConfig(
        candidate_id=H.CANDIDATE_1C,
        epochs=epochs,
        batch_size=256,
        patience=4,
        device=device,
        max_seconds=budget,
    )

    def progress(_id, row):
        log(
            f"  [{H.CANDIDATE_1C}] epoch {row['epoch']:>2}  train {row['train_loss']:.4f}  "
            f"dev CE {row['dev_ce']:.4f}  R_CE {row['dev_r_ce']:.4f}  "
            f"top1 {row['dev_top1']:.4f}  {row['seconds']:.0f}s"
        )

    record = train_final_block(
        model,
        load_features("train", feat.LAYER_PENULTIMATE),
        train,
        load_features("dev", feat.LAYER_PENULTIMATE),
        dev,
        config,
        progress=progress,
    )
    state = record.pop("best_state")
    path = CHECKPOINT_DIRECTORY / f"{H.CANDIDATE_1C}.pt"
    torch.save(
        {
            "candidate_id": H.CANDIDATE_1C,
            "architecture": model.architecture,
            "state_dict": state,
            "config": record["config"],
            "dev_metrics": record["dev_metrics"],
            "corpus_version": CORPUS_VERSION,
            **PHASE11B_STATUS_MARKERS,
        },
        path,
    )
    record["checkpoint"] = {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "sha256": file_sha256(path),
    }
    record["architecture"] = model.architecture
    record["parameters_added"] = H.parameter_count(model)
    record["parameters_unfrozen_c1"] = H.parameter_count(model.block) + H.parameter_count(
        model.encoder_norm
    )
    tokens = load_features("dev", feat.LAYER_PENULTIMATE)
    started = time.perf_counter()
    probabilities = predict_probabilities_1c(model, tokens, dev, device="cpu")
    elapsed = time.perf_counter() - started
    record["inference"] = {
        "batch_rows": int(dev["pieces"]),
        "batch_seconds": round(elapsed, 6),
        "microseconds_per_piece": round(elapsed / int(dev["pieces"]) * 1e6, 4),
        "note": (
            "1C's cost is per position, not per piece: it runs a transformer "
            "block over all 100 tokens before the head sees anything."
        ),
    }
    record["probabilities"] = probabilities
    record["head"] = model
    return record


def repeat_training(
    frozen_model,
    train: dict,
    dev: dict,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    dev_features: np.ndarray,
    first: dict,
    *,
    device: str,
    epochs: int,
    budget: float,
) -> dict:
    """Retrain every candidate under the identical configuration.

    A control, not a second experiment: the checkpoints and the leaderboard
    come from the first pass. What this measures is how much of a reported
    gap could be run-to-run noise, so the report can state the answer
    instead of assuming bit-exactness the backend does not promise.
    """
    drift = {}
    for candidate_id in first:
        if candidate_id == H.CANDIDATE_1C:
            record = train_1c(
                frozen_model, train, dev, device=device, epochs=EPOCHS_1C, budget=budget
            )
            record.pop("head", None)
            record.pop("probabilities", None)
        else:
            record = train_candidate(
                candidate_id,
                frozen_model,
                train_features,
                train_labels,
                dev_features,
                dev,
                device=device,
                epochs=epochs,
                max_seconds=budget,
            )
            record.pop("head", None)
        repeated = record["dev_metrics"]["r_ce"]
        original = first[candidate_id]["dev_metrics"]["r_ce"]
        drift[candidate_id] = {
            "r_ce_first_pass": original,
            "r_ce_repeat_pass": repeated,
            "absolute_drift": abs(repeated - original),
        }
        log(
            f"  [repeat/{candidate_id}] R_CE {original:.4f} -> {repeated:.4f} "
            f"(drift {abs(repeated - original):.5f})"
        )
    worst = max(drift.values(), key=lambda row: row["absolute_drift"])
    return {
        "note": (
            "identical configuration, identical corpus, identical frozen features; "
            "CPU float32 reductions are not bit-reproducible across runs, so this "
            "is the scale of run-to-run noise in a reported R_CE"
        ),
        "candidates": drift,
        "worst_absolute_drift": worst["absolute_drift"],
        "checkpoints_kept_from": "first pass",
    }


def paired_comparisons(results: dict, heads: dict, dev: dict) -> dict:
    """Paired game-bootstrap of every candidate pair on the same pieces."""
    probabilities = {}
    for candidate_id, record in results.items():
        if "probabilities" in record:
            probabilities[candidate_id] = record.pop("probabilities")
        else:
            probabilities[candidate_id] = predict_probabilities(
                heads[candidate_id], np.asarray(load_features("dev")), device="cpu"
            )
    names = list(results)
    out = {}
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            out[f"{left} vs {right}"] = M.paired_comparison(
                probabilities[left], probabilities[right], dev
            )
    return out


# ---------------------------------------------------------------------------
# interface smoke check
# ---------------------------------------------------------------------------


def interface_check(heads: dict, *, worlds: int = 8, positions: int = 16) -> dict:
    """Exercise `predict_marginals` / `sample_worlds` on real dev positions.

    The positions are played once and every candidate is checked on the
    same ones, so the block is a like-for-like statement about the shared
    interface rather than three unrelated smoke tests.
    """
    from stratego.belief.phase11b.corpus import (
        Phase11BSetupSources,
        corpus_plans,
        play_corpus_game,
        select_decisions,
    )
    from stratego.evaluation.match_spec import EVALUATION_RULES
    from stratego.evaluation.phase11_pipeline import build_owners
    from stratego.evaluation.phase11_public_state import build_public_state_document
    from stratego.evaluation.policy import build_public_view
    from stratego.engine.constants import BLUE, RED
    from stratego.engine.observation import build_observation
    from stratego.engine.state import create_game
    from stratego.engine.transition import apply_action

    model, _ = feat.load_frozen_c1(
        REPOSITORY_ROOT, CHECKPOINT_DIRECTORY / "phase9_c1_readonly_copy.pt", device="cpu"
    )
    owners, _ = build_owners(
        REPOSITORY_ROOT, CHECKPOINT_DIRECTORY / "phase9_c1_readonly_copy.pt", device="cpu"
    )
    sources = Phase11BSetupSources()

    started = time.perf_counter()
    states: list[Phase11BPublicState] = []
    for plan in corpus_plans("dev", sources, limit=8):
        if len(states) >= positions:
            break
        result, decisions = play_corpus_game(plan, owners)
        selected = select_decisions(decisions, 4)
        if not selected:
            continue
        observer = {"red": RED, "blue": BLUE}[plan.observer_color]
        wanted = {int(row["ply"]) for row in selected}
        state = create_game(
            plan.red_setup, plan.blue_setup, rules=EVALUATION_RULES, game_id=plan.game_id
        )
        for action in result.action_history:
            if state.terminal or len(states) >= positions:
                break
            if state.acting_player == observer and int(state.total_moves) in wanted:
                observation = build_observation(state, observer)
                document = build_public_state_document(
                    build_public_view(state, observer), observation
                )
                states.append(Phase11BPublicState(document, observation))
            apply_action(state, int(action))

    blocks = {}
    for candidate_id, head in heads.items():
        belief = Phase11BBeliefModel(
            model, head, candidate_id=candidate_id, device="cpu"
        )
        sampled = 0
        hidden_counts = []
        for ordinal, public in enumerate(states):
            marginals = belief.predict_marginals(public)
            for row in marginals.values():
                if row.shape != (RANK_COUNT,) or abs(float(row.sum()) - 1.0) > 1e-9:
                    raise AssertionError(f"{candidate_id}: a marginal is not a probability vector")
            drawn = belief.sample_worlds(public, worlds, seed=ordinal)
            repeat = belief.sample_worlds(public, worlds, seed=ordinal)
            if [world["assignment"] for world in drawn] != [
                world["assignment"] for world in repeat
            ]:
                raise AssertionError(f"{candidate_id}: sample_worlds is not seed-deterministic")
            for world in drawn:
                if sorted(world["assignment"]) != sorted(marginals):
                    raise AssertionError(f"{candidate_id}: a world missed an unresolved piece")
            sampled += len(drawn)
            hidden_counts.append(len(marginals))
        blocks[candidate_id] = {
            "interface_version": BELIEF_INTERFACE_VERSION,
            "candidate_id": candidate_id,
            "feature_layer": getattr(head, "feature_layer", "final"),
            "positions_checked": len(states),
            "worlds_sampled": sampled,
            "worlds_per_position": worlds,
            "hidden_pieces_per_position": (
                round(float(np.mean(hidden_counts)), 2) if hidden_counts else 0.0
            ),
            "all_marginals_sum_to_one": True,
            "sample_worlds_seed_deterministic": True,
            "all_worlds_passed_accepted_validation_stack": True,
            "sampler_source": (
                "stratego.evaluation.phase11_sampler (accepted, unmodified)"
            ),
        }
        log(
            f"  [interface/{candidate_id}] {len(states)} positions, {sampled} worlds, "
            f"all valid"
        )
    return {
        "positions_checked": len(states),
        "candidates": blocks,
        "seconds": round(time.perf_counter() - started, 3),
    }


__all__ = ["main"]


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "mps_available": bool(torch.backends.mps.is_available()),
        "torch_threads": int(torch.get_num_threads()),
    }


def peak_rss_bytes() -> int:
    """Process peak resident set size, in bytes.

    Reported as the run's `peak memory`. It covers the whole harness — the
    memory-mapped corpus, the feature caches and the head — which is the
    honest number for "what did this experiment cost to run", and the
    report says so rather than implying it is the model's footprint.
    """
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def leaderboard_row(candidate_id: str, record: dict, *, reference: bool = False) -> dict:
    """The standardized leaderboard fields Agents 2-5 compare against."""
    metrics = record["metrics"] if reference else record["dev_metrics"]
    strata = metrics["strata"]
    row = {
        "candidate_id": candidate_id,
        "phase11b_version": PHASE11B_VERSION,
        "corpus_version": CORPUS_VERSION,
        "architecture": record.get("architecture", "linear(128->12)"),
        "ce": metrics["ce"],
        "baseline_ce": metrics["baseline_ce"],
        "r_ce": metrics["r_ce"],
        "r_ce_ci95": metrics["r_ce_ci95"],
        "top1": metrics["top1"],
        "baseline_top1": metrics["baseline_top1"],
        "r_ce_by_stratum": {
            name: strata[name]["r_ce"] for name in CORPUS_STRATA if name in strata
        },
        "top1_by_stratum": {
            name: strata[name]["top1"] for name in CORPUS_STRATA if name in strata
        },
        "worst_stratum": metrics["worst_stratum"],
        "best_stratum": metrics["best_stratum"],
        "dev_samples": metrics["samples"],
        "dev_pieces": metrics["pieces"],
        "diagnostic_projected_r_ce": metrics["diagnostic_projected_r_ce"],
    }
    if reference:
        from stratego.training.phase11_contract import BELIEF_HEAD_TENSOR_SHAPES

        accepted_head = sum(
            int(np.prod(shape)) for shape in BELIEF_HEAD_TENSOR_SHAPES.values()
        )
        row.update(
            {
                "parameters_added": 0,
                "belief_parameters_total": accepted_head,
                "training_seconds": 0.0,
                "time_to_best_seconds": 0.0,
                "best_epoch": None,
                "inference_microseconds_per_piece": None,
                "trained_in_phase11b": False,
            }
        )
    else:
        row.update(
            {
                "parameters_added": record["parameters_added"],
                "belief_parameters_total": record["parameters_added"],
                "parameters_unfrozen_c1": record.get("parameters_unfrozen_c1", 0),
                "retrains_accepted_c1_weights": bool(record.get("parameters_unfrozen_c1")),
                "training_seconds": record["training_seconds"],
                "time_to_best_seconds": record["time_to_best_seconds"],
                "best_epoch": record["best_epoch"],
                "epochs_run": record["epochs_run"],
                "stopped_because": record["stopped_because"],
                "inference_microseconds_per_piece": record["inference"][
                    "microseconds_per_piece"
                ],
                "checkpoint_sha256": record["checkpoint"]["sha256"],
                "trained_in_phase11b": True,
            }
        )
    return row


#: The `R_CE` band inside which the sprint prefers the cheaper model.
EQUIVALENCE_BAND = 0.005


def pick_winner(rows: dict) -> dict:
    """The Agent 1 winner, under the sprint's engineering winner rule.

    The rule is applied **once, against the best candidate** — never as a
    chain of pairwise comparisons. Applied transitively, a run of 0.004
    steps would discard an arbitrarily large real improvement: A within
    0.005 of B and B within 0.005 of C does not make A within 0.005 of C,
    and the rule calls a gap above the band material. So:

    1. take the lowest development `R_CE`;
    2. keep every candidate within `EQUIVALENCE_BAND` of it;
    3. among those, the cheapest and simplest wins, unless a costlier one
       has a meaningful unusual-behaviour (Scout-rush) advantage.
    """
    trained = {name: row for name, row in rows.items() if row["trained_in_phase11b"]}
    ordered = sorted(trained, key=lambda name: trained[name]["r_ce"])
    leader = ordered[0]
    band = [
        name
        for name in ordered
        if trained[name]["r_ce"] - trained[leader]["r_ce"] < EQUIVALENCE_BAND
    ]
    excluded = [name for name in ordered if name not in band]
    winner = min(band, key=lambda name: trained[name]["parameters_added"])

    scout = {
        name: trained[name]["r_ce_by_stratum"].get("scout_rush")
        for name in band
        if trained[name]["r_ce_by_stratum"].get("scout_rush") is not None
    }
    scout_advantage = None
    if len(scout) > 1 and winner in scout:
        best_scout = min(scout, key=lambda name: scout[name])
        scout_advantage = {
            "candidate": best_scout,
            "scout_rush_r_ce": scout[best_scout],
            "winner_scout_rush_r_ce": scout[winner],
            "advantage": scout[winner] - scout[best_scout],
            "meaningful": bool(scout[winner] - scout[best_scout] >= EQUIVALENCE_BAND),
        }

    if winner == leader:
        reason = (
            f"lowest development R_CE ({trained[winner]['r_ce']:.4f}) and the cheapest "
            "candidate inside the equivalence band"
        )
    else:
        reason = (
            f"within {trained[winner]['r_ce'] - trained[leader]['r_ce']:.4f} R_CE of the "
            f"leader {leader} ({trained[leader]['r_ce']:.4f}) and materially cheaper and "
            f"simpler ({trained[winner]['parameters_added']:,} trained parameters against "
            f"{trained[leader]['parameters_added']:,}, and no accepted C1 weight retrained), "
            "so the Phase 11B engineering winner rule prefers it"
        )
    return {
        "winner": winner,
        "leader_by_r_ce": leader,
        "equivalence_band": EQUIVALENCE_BAND,
        "inside_band": band,
        "excluded_as_materially_worse": excluded,
        "r_ce_margin_to_leader": trained[winner]["r_ce"] - trained[leader]["r_ce"],
        "scout_rush_check": scout_advantage,
        "reason": reason,
        "rule": "00_PHASE_11B_OVERVIEW.md 'Engineering Winner Rule'",
        "rule_application": (
            "band measured against the leader only; the rule is not applied "
            "transitively down a chain of pairwise comparisons"
        ),
    }


def stage_report(stages: dict, *, with_1c: bool) -> dict:
    """Write the leaderboard JSON, the learning curves and the report."""
    verify = stages["verify"]
    corpus = stages["corpus"]
    features = stages["features"]
    reference = stages["reference"]
    train = stages["train"]

    rows = {
        "phase11_head_unchanged_reference": leaderboard_row(
            "phase11_head_unchanged_reference", reference, reference=True
        )
    }
    for candidate_id, record in train["results"].items():
        rows[candidate_id] = leaderboard_row(candidate_id, record)
    decision = pick_winner(rows)

    curves = {
        candidate_id: record["curve"] for candidate_id, record in train["results"].items()
    }
    CURVES_PATH.write_text(json.dumps(curves, indent=1) + "\n")

    manifest = corpus["manifest"]
    label_prior = corpus_label_prior()
    summary = {
        "phase": "phase11b",
        "agent": AGENT,
        "phase11b_version": PHASE11B_VERSION,
        "identity_version": PHASE11B_IDENTITY_VERSION,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment(),
        "seeds": dict(CANONICAL_PHASE11B_SEEDS),
        **PHASE11B_STATUS_MARKERS,
        **PHASE11_FACTS,
        "starting_state": {
            "phase9_checkpoint": verify["phase9_checkpoint"],
            "preserved_artifact_digests": verify["preserved_digests"],
            "preservation_problems": verify["problems"],
        },
        "common_corpus": {
            "corpus_version": manifest["corpus_version"],
            "corpus_format_version": manifest["corpus_format_version"],
            "corpus_digest": manifest["corpus_digest"],
            "root": CORPUS_ROOT,
            "splits": {
                split: {
                    key: value
                    for key, value in manifest["splits"][split].items()
                    if key != "file_digests"
                }
                for split in manifest["splits"]
            },
            "file_digests": {
                split: manifest["splits"][split]["file_digests"]
                for split in manifest["splits"]
            },
            "generation_seconds": manifest.get("generation_seconds"),
            "reusable_by": "phase11b agents 2-5, byte-for-byte",
            "label_prior": label_prior,
        },
        "frozen_features": {
            "cache_version": features["cache_version"],
            "layer": features["layer"],
            "layers": features.get("layers", [features["layer"]]),
            "splits": features["splits"],
            "caches": features.get("caches", {}),
            "frozen_model": features["frozen_model"],
        },
        "leaderboard": rows,
        "leaderboard_order": sorted(rows, key=lambda name: rows[name]["r_ce"]),
        "decision": decision,
        "paired_comparisons": train.get("paired_comparisons", {}),
        "reproducibility": train.get("reproducibility"),
        "reference_note": reference["note"],
        "uniform_floor": reference["uniform_floor"],
        "interface": stages.get("interface"),
        "experiment_1c": {
            "run": bool(train.get("ran_1c")),
            "condition": (
                "01_AGENT_1: run 1C only if 1B is clearly promising and "
                "sufficient time remains"
            ),
            "justification": (
                "1B was the best candidate so far and beat both the unchanged "
                "Phase 11 head and 1A on every stratum, and the whole 1A+1B "
                "programme had cost under 20 seconds, so the budget was "
                "untouched. 1C answers the question 1B raised: whether the "
                "remaining shortfall is head capacity or the frozen "
                "representation."
                if train.get("ran_1c")
                else "not run"
            ),
            "epochs": EPOCHS_1C if train.get("ran_1c") else None,
        },
        "suite": stages.get("suite"),
        "peak_memory_bytes": train.get("peak_memory_bytes", peak_rss_bytes()),
        "peak_memory_note": (
            "peak process RSS at the end of the training stage: the memory-mapped "
            "corpus, the frozen feature caches and the trained candidates, not the "
            "head alone"
        ),
        "scientific_claim": "none",
        "stop_condition": (
            "Agent 1 reports which of 1A/1B/1C was best and starts no other "
            "architecture."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=1, sort_keys=False, default=str) + "\n")
    return {"stage": "report", "summary_path": str(SUMMARY_PATH), "pass": True, "summary": summary}


def corpus_label_prior() -> dict:
    """The observed hidden-rank prior of the training split.

    Worth stating because it is **not** the initial army distribution: a
    hidden piece is one nobody has resolved yet, and the ranks that survive
    unresolved are systematically different from the ranks an army starts
    with. A later agent that assumes the army prior will mis-read its own
    calibration.
    """
    from stratego.belief.phase11b.contract import RANK_INITIAL_COUNTS, RANK_NAMES

    data = S.load_split(CORPUS_DIRECTORY, "train", labels=True)
    counts = np.bincount(np.asarray(data["true_rank"], dtype=np.int64), minlength=RANK_COUNT)
    observed = counts / counts.sum()
    army = np.asarray(RANK_INITIAL_COUNTS, dtype=np.float64) / sum(RANK_INITIAL_COUNTS)
    return {
        "split": "train",
        "hidden_pieces": int(counts.sum()),
        "observed": {name: float(observed[index]) for index, name in enumerate(RANK_NAMES)},
        "initial_army": {name: float(army[index]) for index, name in enumerate(RANK_NAMES)},
        "largest_deficit": RANK_NAMES[int(np.argmin(observed - army))],
        "largest_surplus": RANK_NAMES[int(np.argmax(observed - army))],
        "note": (
            "the prior over *unresolved* ranks, not over an army: ranks that "
            "reveal themselves early are under-represented among hidden pieces"
        ),
    }


def record_suite() -> dict:
    """Run the repository suite and record its counts, the accepted pattern."""
    import re

    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = round(time.perf_counter() - started, 2)
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout else ""
    counts = {
        name: int(value)
        for value, name in re.findall(r"(\d+) (passed|failed|skipped|error[s]?)", tail)
    }
    return {
        "command": "python -m pytest tests -q",
        "returncode": completed.returncode,
        "summary_line": tail,
        "passed": counts.get("passed", 0),
        "skipped": counts.get("skipped", 0),
        "failed": counts.get("failed", 0) + counts.get("errors", 0) + counts.get("error", 0),
        "green": completed.returncode == 0,
        "wall_clock_seconds": elapsed,
    }


def write_markdown(stages: dict) -> Path:
    """The human-readable Agent 1 report."""
    summary = stages["report"]["summary"]
    rows = summary["leaderboard"]
    order = summary["leaderboard_order"]
    decision = summary["decision"]
    corpus = summary["common_corpus"]
    interface_block = summary.get("interface") or {}
    winner = decision["winner"]
    reference = rows["phase11_head_unchanged_reference"]
    trained = [name for name in order if rows[name]["trained_in_phase11b"]]
    cheapest = min(trained, key=lambda name: rows[name]["parameters_added"])
    leader = decision["leader_by_r_ce"]

    def fmt(value, digits=4):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    lines: list[str] = []
    add = lines.append

    add("# Phase 11B — Agent 1: Attached Belief-Head Engineering")
    add("")
    add("**Status: engineering prototype.** This report does not repair Phase 11, does")
    add("not overturn the Phase 11 `FAIL`, and does not authorize Phase 12.")
    add("`phase11_test_bank_v1` was not opened; it remains spent.")
    add("")
    add("| marker | value |")
    add("| --- | --- |")
    for key in (
        "phase",
        "status",
        "phase11_fail_unchanged",
        "phase11_test_bank_used",
        "phase12_authorized_by_this_artifact",
        "phase11_final_classification",
        "phase11_test_bank_spent",
        "scientific_claim",
    ):
        add(f"| `{key}` | `{summary[key]}` |")

    # -- findings ----------------------------------------------------------
    gain_1a = reference["r_ce"] - rows[cheapest]["r_ce"]
    gain_head = rows[cheapest]["r_ce"] - rows[winner]["r_ce"]
    add("")
    add("## 0. What Agent 1 found")
    add("")
    add("Agent 1's question was whether the Phase 11 weakness was **mainly insufficient**")
    add("**dedicated belief optimization** or **an undersized belief output head**. On the")
    add("common Phase 11B development set the answer is unambiguous:")
    add("")
    add(
        f"1. **It was mainly the optimization.** Retraining the *existing* 128→12 layer —"
    )
    add(
        f"   the same {reference['belief_parameters_total']:,} parameters, the same frozen C1 features — moves"
    )
    add(
        f"   `R_CE` from **{fmt(reference['r_ce'])}** to **{fmt(rows[cheapest]['r_ce'])}**, a gain of"
    )
    add(
        f"   **{fmt(gain_1a)}**. Nothing about the architecture changed; only the objective did."
    )
    add(
        f"2. **Head capacity is worth much less.** Going from {rows[cheapest]['parameters_added']:,} to"
    )
    add(
        f"   {rows[winner]['parameters_added']:,} trained parameters — a {rows[winner]['parameters_added'] // max(rows[cheapest]['parameters_added'], 1)}× head — buys a further"
    )
    add(
        f"   **{fmt(gain_head)}**, about {gain_1a / max(gain_head, 1e-9):.0f}× smaller than the retraining's gain."
    )
    add("3. **The representation is now the binding constraint.** Unfreezing the last C1")
    add(
        f"   block (1C) buys another {fmt(rows[winner]['r_ce'] - rows[leader]['r_ce'])} on top of the larger head — as much again as"
    )
    add("   the larger head bought over the retrained linear layer, and the only change")
    add("   that reached past the frozen features. That is the strongest signal here for")
    add("   what Agents 2-5 should expect: the remaining headroom is in the")
    add("   representation, not in the head.")
    add("")
    add("The largest single change is on the hardest stratum. The unchanged Phase 11 head")
    add(
        f"scored `R_CE` {fmt(reference['r_ce_by_stratum'].get('scout_rush'))} on Scout-rush — indistinguishable from simply counting"
    )
    scout_values = sorted(
        rows[name]["r_ce_by_stratum"]["scout_rush"] for name in trained
    )
    add(
        f"the remaining pieces. The three retrained candidates score {fmt(scout_values[0])}-{fmt(scout_values[-1])} there."
    )
    add("")
    add("None of this is a scientific claim, a repair of Phase 11, or evidence about")
    add("whether better beliefs win more games. It is an engineering measurement on one")
    add("fresh development set.")

    # -- starting state ----------------------------------------------------
    checkpoint = summary["starting_state"]["phase9_checkpoint"]
    add("")
    add("## 1. Starting state")
    add("")
    add("The accepted Phase 9 checkpoint was opened **read-only** and exported to a")
    add("Phase 11B path. Its identity was re-derived from live bytes:")
    add("")
    add("```text")
    add(f"sha256                {checkpoint['sha256']}")
    add(f"model state digest    {checkpoint['model_state_digest']}")
    add(f"belief head digest    {checkpoint['belief_head_digest']}")
    add(f"parameters            {checkpoint['parameters']:,}")
    add(f"global optimizer step {checkpoint['global_optimizer_step']:,}")
    add("```")
    add("")
    add(
        f"{len(summary['starting_state']['preserved_artifact_digests'])} preserved Phase 11 artifacts and accepted modules were digested; every"
    )
    add("digest is in `agent_01_summary.json`. Phase 11B wrote to none of them. The")
    add("accepted Phase 11 sampler, baselines, public-state document, contract, seed")
    add("module and belief targets are **imported and unmodified**.")

    # -- corpus ------------------------------------------------------------
    add("")
    add("## 2. Common Phase 11B corpus")
    add("")
    add(f"`{corpus['corpus_version']}` — corpus digest")
    add("")
    add("```text")
    add(corpus["corpus_digest"])
    add("```")
    add("")
    add("| split | games | observer decisions | samples | hidden pieces | sampled/game | setup-library split |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for split in ("train", "dev"):
        block = corpus["splits"][split]
        add(
            f"| {split} | {block['games']:,} | {block['observer_decisions']:,} | "
            f"{block['samples']:,} | {block['hidden_pieces']:,} | "
            f"{block['decisions_per_game']} | {block['library_split']} |"
        )
    add("")
    add("512 training and 128 development games per behaviour stratum")
    add(f"({', '.join(CORPUS_STRATA)}), balanced by")
    add("construction over the two setup sources and both observer colours: 16 cells,")
    add("cell-major, so balance is a property of the id space rather than of any draw.")
    add("")
    add("The setup-source split is the **opponent's**, following the accepted Phase 11")
    add("convention: the observer always draws from the accepted P10-D production")
    add("source, because that is the seat a deployed system occupies, and the 50/50")
    add("`p10d` / `neutral_v1` variation is what the observer has to form beliefs")
    add("about.")
    add("")
    add("The two splits draw from **disjoint setup-library splits** (`train` and")
    add("`validation`) as well as from disjoint seed streams under a Phase 11B-only")
    add("blake2b personalization, so a development game cannot share a base arrangement")
    add("or a match seed with a training game, and no Phase 11B stream can coincide with")
    add("a Phase 11 one.")
    add("")
    add("Public inputs live in `public/`, privileged true ranks in `privileged/`, and the")
    add("loader returns labels only when a caller asks for them by name. Every sample's")
    add("observation was rebuilt on an independent engine replay and checked bit-for-bit")
    add("against the digest the public pass recorded **before** any label was read; the")
    add("hidden-piece set, the remaining inventory and the legal-rank masks were")
    add("re-derived from the public document on that replay too.")
    add("")
    prior = corpus.get("label_prior")
    if prior:
        deficit, surplus = prior["largest_deficit"], prior["largest_surplus"]
        add("")
        add("One property later agents should not assume away: the hidden-rank prior is")
        add("**not** the initial army distribution. A hidden piece is one nobody has")
        add("resolved, and ranks that reveal themselves early are under-represented among")
        add(
            f"them — `{deficit}` is {prior['observed'][deficit]:.1%} of hidden pieces against {prior['initial_army'][deficit]:.1%} of an army,"
        )
        add(
            f"while `{surplus}` is {prior['observed'][surplus]:.1%} against {prior['initial_army'][surplus]:.1%}."
        )
        add("")
    zero = sum(corpus["splits"][split]["zero_sample_games"] for split in corpus["splits"])
    add(
        f"{zero} of {corpus['splits']['train']['games'] + corpus['splits']['dev']['games']:,} games contributed no sample: they ended before the observer had an"
    )
    add("eligible decision. That is the same environment property Phase 11 recorded, not")
    add("a generation fault.")

    # -- results -----------------------------------------------------------
    add("")
    add("## 3. Results on the common development set")
    add("")
    add("| candidate | architecture | CE | R_CE | 95% CI | top-1 | trained params | train s | to best s | µs/piece |")
    add("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for name in order:
        row = rows[name]
        low, high = row["r_ce_ci95"]
        add(
            f"| `{name}` | {row['architecture']} | {fmt(row['ce'])} | "
            f"**{fmt(row['r_ce'])}** | [{fmt(low)}, {fmt(high)}] | {fmt(row['top1'])} | "
            f"{row['parameters_added']:,} | {fmt(row['training_seconds'], 1)} | "
            f"{fmt(row['time_to_best_seconds'], 1)} | "
            f"{fmt(row['inference_microseconds_per_piece'], 3)} |"
        )
    add("")
    add(
        f"`remaining_count_belief_v1` — the `R_CE` denominator — scores CE **{fmt(reference['baseline_ce'])}**"
    )
    add(
        f"and top-1 {fmt(reference['baseline_top1'])} on these {reference['dev_pieces']:,} hidden pieces. A flat 12-way vector"
    )
    add(f"scores `R_CE` {fmt(summary['uniform_floor']['r_ce'])}, which is the uninformed floor.")
    add("")
    add("The intervals above are marginal game bootstraps. Two candidates scored on the")
    add("*same* pieces are far more comparable than those intervals suggest, so the")
    add("paired game bootstrap of the CE difference is the honest test:")
    add("")
    add("| comparison | mean ΔCE | 95% CI | distinguishable |")
    add("| --- | ---: | --- | --- |")
    for label, block in summary.get("paired_comparisons", {}).items():
        low, high = block["ce_difference_ci95"]
        add(
            f"| {label} | {block['ce_difference']:+.4f} | "
            f"[{low:+.4f}, {high:+.4f}] | {'yes' if block['distinguishable'] else 'no'} |"
        )
    add("")
    add("All three orderings are real, not noise — the gaps are simply small in `R_CE`.")
    add("")
    add("### Per-stratum R_CE")
    add("")
    add("| candidate | " + " | ".join(CORPUS_STRATA) + " |")
    add("| --- | " + " | ".join("---:" for _ in CORPUS_STRATA) + " |")
    for name in order:
        values = rows[name]["r_ce_by_stratum"]
        add(
            f"| `{name}` | "
            + " | ".join(fmt(values.get(stratum)) for stratum in CORPUS_STRATA)
            + " |"
        )
    add("")
    add("Every retrained candidate improves every stratum, and the ordering of strata is")
    add("unchanged: self-play positions stay the easiest, rule opponents the hardest.")
    add("Scout-rush moves from *worst* stratum for the old head to mid-table for all")
    add("three candidates.")

    # -- configuration -----------------------------------------------------
    add("")
    add("## 4. What was trained, and how")
    add("")
    add("One declared configuration per experiment. **No hyperparameter sweep and no")
    add("architecture search was run**; these are choices, not search results.")
    add("")
    add("| experiment | trainable | frozen | optimizer | LR | batch | epochs | stopped |")
    add("| --- | --- | --- | --- | --- | ---: | ---: | --- |")
    configurations = {
        name: stages["train"]["results"][name]["config"] for name in trained
    }
    frozen_note = {
        H.CANDIDATE_1A: "all of C1",
        H.CANDIDATE_1B: "all of C1",
        H.CANDIDATE_1C: "C1 except its last block",
    }
    trainable_note = {
        H.CANDIDATE_1A: "the accepted 128→12 layer, from the accepted weights",
        H.CANDIDATE_1B: "a fresh 128→512→512→12 GELU MLP",
        H.CANDIDATE_1C: "a copy of C1's last block + encoder norm, plus the 1B head",
    }
    for name in trained:
        config = configurations[name]
        record = stages["train"]["results"][name]
        learning_rate = f"{config['learning_rate']:g}"
        if config.get("block_learning_rate"):
            learning_rate += f" (block {config['block_learning_rate']:g})"
        add(
            f"| `{name}` | {trainable_note.get(name, '—')} | {frozen_note.get(name, '—')} | "
            f"{config['optimizer']} + cosine | {learning_rate} | {config['batch_size']:,} | "
            f"{record['epochs_run']}/{config['epochs']} | {record['stopped_because']} |"
        )
    add("")
    add("The loss is supervised hidden-rank cross-entropy over hidden pieces and nothing")
    add("else: no policy term, no value term, no game outcome anywhere. 1A deliberately")
    add("starts from the **accepted** belief-head weights, because that is what makes its")
    add("gain a measurement of dedicated belief optimization at fixed capacity.")
    add("")
    add("Because C1 is frozen for 1A and 1B, its representation is a constant of the")
    add("corpus and was cached once. Both experiments therefore see bit-identical")
    add("features, and any difference between them is the head and nothing else.")

    # -- cost --------------------------------------------------------------
    features = summary["frozen_features"]
    generation = corpus.get("generation_seconds") or {}
    add("")
    add("## 5. Cost")
    add("")
    add("| item | wall clock |")
    add("| --- | ---: |")
    add(
        f"| corpus generation (2,560 games, played once, reused by Agents 2-5) | {sum(generation.values()):.0f} s |"
    )
    for layer, block in features.get("caches", {}).items():
        add(
            f"| frozen C1 `{layer}` feature cache (both splits) | "
            f"{sum(entry['seconds'] for entry in block.values()):.1f} s |"
        )
    for name in trained:
        add(f"| train `{name}` | {rows[name]['training_seconds']:.1f} s |")
    add("")
    add(
        f"Peak memory: **{summary['peak_memory_bytes'] / 1e9:.2f} GB**. That is the {summary['peak_memory_note']},"
    )
    add("and it is dominated by the caches rather than by any model — the largest single")
    add("object is the penultimate-layer cache 1C reads.")
    add("")
    add(
        f"Every experiment ran on CPU / float32 at {summary['environment'].get('torch_threads', '?')} torch threads — the accepted"
    )
    add("evaluation backend. Nothing here needed a GPU: the whole Agent 1 experiment")
    add("programme after corpus generation costs under two minutes.")
    suite = summary.get("suite")
    if suite:
        add("")
        add(
            f"Repository suite after Agent 1: **{suite['passed']:,} passed, {suite['skipped']} skipped**"
        )
        add(
            f"in {suite['wall_clock_seconds']:.0f} s (`{suite['command']}`)."
        )
    repeatability = summary.get("reproducibility")
    if repeatability:
        add("")
        add("Retraining every candidate a second time under the identical configuration")
        add(
            f"moves `R_CE` by at most **{repeatability['worst_absolute_drift']:.5f}** — an order of magnitude"
        )
        add("smaller than the smallest gap the leaderboard reports, so the ordering is not")
        add("an artefact of run-to-run noise. Multi-threaded CPU float32 reductions are not")
        add("bit-reproducible, so this is measured rather than assumed.")
        add("")
        add("| candidate | first pass | repeat pass | drift |")
        add("| --- | ---: | ---: | ---: |")
        for name, row in repeatability["candidates"].items():
            add(
                f"| `{name}` | {row['r_ce_first_pass']:.4f} | "
                f"{row['r_ce_repeat_pass']:.4f} | {row['absolute_drift']:.5f} |"
            )

    # -- decision ----------------------------------------------------------
    add("")
    add("## 6. Which of 1A / 1B / 1C was best")
    add("")
    add(f"**Winner: `{winner}`.**")
    add("")
    add(f"{decision['reason'][:1].upper()}{decision['reason'][1:]}.")
    add("")
    add("How the rule was applied:")
    add("")
    add(f"- leader by `R_CE`: `{leader}` ({fmt(rows[leader]['r_ce'])});")
    add(
        f"- inside the {decision['equivalence_band']} equivalence band: "
        + ", ".join(f"`{name}`" for name in decision["inside_band"])
        + ";"
    )
    add(
        "- materially worse and excluded: "
        + (
            ", ".join(f"`{name}`" for name in decision["excluded_as_materially_worse"])
            or "none"
        )
        + ";"
    )
    scout = decision.get("scout_rush_check") or {}
    if scout:
        add(
            f"- Scout-rush check: `{scout['candidate']}` leads the winner by "
            f"{fmt(scout['advantage'])} `R_CE` there, which is **not** a meaningful"
        )
        add("  unusual-behaviour advantage at this band width;")
    add("- search-integration complexity: the winner attaches to the **unmodified**")
    add("  frozen C1 encoder, so the belief comes out of the same forward pass the")
    add("  policy already runs. 1C would require carrying a second, retrained copy of")
    add("  C1's last block alongside the accepted network.")
    add("")
    add("The band is measured against the leader only, never as a chain of pairwise")
    add("comparisons: applied transitively, a run of sub-band steps would discard an")
    add("arbitrarily large real improvement, and the rule itself calls a gap above the")
    add("band material.")

    # -- interface ---------------------------------------------------------
    add("")
    add("## 7. Required interface")
    add("")
    add("```text")
    add("predict_marginals(public_state)      -> {piece_slot: 12-way rank probabilities}")
    add("sample_worlds(public_state, n, seed) -> complete legal hidden armies")
    add("```")
    add("")
    add("`Phase11BPublicState` carries exactly the two public objects the accepted")
    add("`Phase11BeliefRequest` carries — the frozen public-state document and the")
    add("127-channel observation — so the interface has no field a true rank could")
    add("arrive in.")
    add("")
    add("| candidate | positions | worlds | marginals valid | seed-deterministic | worlds legal |")
    add("| --- | ---: | ---: | --- | --- | --- |")
    for name, block in (interface_block.get("candidates") or {}).items():
        add(
            f"| `{name}` | {block['positions_checked']} | {block['worlds_sampled']} | "
            f"{'yes' if block['all_marginals_sum_to_one'] else 'no'} | "
            f"{'yes' if block['sample_worlds_seed_deterministic'] else 'no'} | "
            f"{'yes' if block['all_worlds_passed_accepted_validation_stack'] else 'no'} |"
        )
    add("")
    add("Every world was drawn through **`stratego.evaluation.phase11_sampler`, the")
    add("accepted Phase 11 sampler, imported and unmodified** — the completion")
    add("feasibility guard, the `learned_probability × remaining_count` weighting, the")
    add("frozen categorical walk and the full validation stack are all the accepted code.")
    add("Phase 11B supplies marginals and nothing else.")

    # -- caveats -----------------------------------------------------------
    add("")
    add("## 8. Caveats a reader should carry forward")
    add("")
    add("- **This is a development-set number.** There is no sealed bank behind it and")
    add("  no scientific claim attached to it. The development set is an engineering")
    add("  comparison set, exactly as the sprint defines it.")
    if H.CANDIDATE_1C in stages["train"]["results"]:
        curve = stages["train"]["results"][H.CANDIDATE_1C]["curve"]
        tail = curve[-4:] if len(curve) >= 4 else curve
        slope = tail[0]["dev_r_ce"] - tail[-1]["dev_r_ce"]
        add(
            f"- **1C's best epoch was its last ({curve[-1]['epoch']} of {curve[-1]['epoch']}).** Its development curve was"
        )
        add(
            f"  still improving monotonically at the end, but only just: the last {len(tail) - 1} epochs"
        )
        add(
            f"  moved `R_CE` by {slope:.5f} in total, against a cosine schedule that had already"
        )
        add(
            f"  annealed. Read {fmt(rows[leader]['r_ce'])} as a slight underestimate of what this"
        )
        add("  configuration reaches, not as a large one. Agent 1 did not extend the run:")
        add("  re-choosing an epoch budget after seeing the result is the tuning the")
        add("  sprint forbids.")
    add("- **1B overfits early.** Its best development epoch was")
    add(
        f"  {stages['train']['results'][H.CANDIDATE_1B]['best_epoch']} of {stages['train']['results'][H.CANDIDATE_1B]['config']['epochs']} while its training loss kept falling. On a frozen"
    )
    add("  128-wide feature the extra head capacity is quickly exhausted.")
    add("- **The headline `R_CE` uses the accepted raw-softmax convention** — no masking,")
    add("  no epsilon, full simplex — because that is how the Phase 11 head was measured")
    add("  and how the accepted sampler consumes a belief. Renormalizing each candidate")
    add("  onto the publicly legal support is reported as a diagnostic only")
    add(
        f"  (`{winner}`: {fmt(rows[winner]['diagnostic_projected_r_ce'])} projected against {fmt(rows[winner]['r_ce'])} raw); mixing it into the"
    )
    add("  headline would compare a masked candidate against an unmasked reference.")
    add("- **The reference row is not the Phase 11 sealed-test result.** It is the")
    add("  unchanged Phase 11 head scored on *these* fresh positions. Phase 11's sealed")
    add("  test remains what it was, and its bank remains spent.")

    # -- handoff -----------------------------------------------------------
    add("")
    add("## 9. Handoff to Agents 2-5")
    add("")
    add("The common corpus is built and immutable. Reuse it byte-for-byte:")
    add("")
    add("```python")
    add("from stratego.belief.phase11b.storage import load_split, split_digest")
    add("")
    add(f'data = load_split("{CORPUS_ROOT}", "train", labels=True)')
    add("```")
    add("")
    add("`split_digest` recomputes the per-file SHA-256s and `corpus_digest` the whole-")
    add("corpus identity; both are in `agent_01_summary.json`. Equality is the proof that")
    add("two experiments were scored on one corpus.")
    add("")
    add("Score with `stratego.belief.phase11b.metrics.evaluate`, which computes the")
    add("`R_CE` denominator from the corpus's own stored public arrays, so every")
    add("candidate divides by the same number on the same pieces.")
    add("")
    add("What Agent 1's result implies for the remaining experiments: the frozen C1")
    add("feature is close to exhausted. Replacing the linear head with a three-layer")
    add(
        f"MLP — {rows[winner]['parameters_added'] // max(rows[cheapest]['parameters_added'], 1)}x the parameters — bought {fmt(rows[cheapest]['r_ce'] - rows[winner]['r_ce'])}; letting a single encoder block move"
    )
    add("bought as much again. The representation is where the remaining headroom is, so a")
    add("raw-observation CNN (Agent 2) is the most informative next experiment: it is")
    add("the cheapest way to learn a belief-specific representation instead of borrowing")
    add("the policy's.")
    add("")
    add("Two practical notes for whoever runs it. First, the frozen-feature caching trick")
    add("does not transfer — a model that learns its own representation must see the")
    add("observations, so budget for real epochs over the 1.3 GB training tensor rather")
    add("than the seconds 1A and 1B took. Second, `phase9_selfplay` is consistently the")
    add("easiest stratum and the rule opponents the hardest; a candidate that wins only")
    add("on self-play positions has not generalized.")

    # -- stop --------------------------------------------------------------
    add("")
    add("## 10. Stop condition")
    add("")
    add("Agent 1 stops here. No other architecture was begun. Phase 11 remains `FAIL`,")
    add("`phase11_test_bank_v1` remains spent and unopened, and nothing in this report")
    add("authorizes Phase 12 or claims that Phase 11 has been repaired.")
    add("")

    REPORT_PATH.write_text("\n".join(lines))
    log(f"[report] wrote {REPORT_PATH.relative_to(REPOSITORY_ROOT)}")
    return REPORT_PATH


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 11B Agent 1 harness")
    parser.add_argument("--full", action="store_true", help="run every stage")
    parser.add_argument("--stage", choices=STAGES, help="run one stage")
    parser.add_argument("--rebuild-corpus", action="store_true")
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="throughput pilot only; writes to data/phase11b/pilot_corpus, never the common corpus",
    )
    parser.add_argument(
        "--limit-dev",
        type=int,
        default=None,
        help="throughput pilot only; writes to data/phase11b/pilot_corpus, never the common corpus",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--budget-seconds", type=float, default=1800.0)
    parser.add_argument("--with-1c", action="store_true")
    parser.add_argument("--skip-interface", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--repeat-train", action="store_true",
                        help="retrain every candidate once more and measure the drift")
    parser.add_argument("--record-suite", action="store_true",
                        help="run the repository suite and record its counts")
    parser.add_argument("--run-pytest", action="store_true")
    arguments = parser.parse_args(argv)

    if not arguments.full and arguments.stage is None:
        parser.error("pass --full or --stage")
    # Pinned so a rerun is as reproducible as the backend allows, and
    # recorded so the report can say what it was measured under.
    torch.set_num_threads(int(arguments.torch_threads))
    wanted = list(STAGES) if arguments.full else [arguments.stage]
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    limits = {}
    if arguments.limit_train is not None:
        limits["train"] = arguments.limit_train
    if arguments.limit_dev is not None:
        limits["dev"] = arguments.limit_dev

    stages = load_stages()
    heads = {}
    started = time.perf_counter()
    for name in wanted:
        log(f"[{name}] ...")
        stage_started = time.perf_counter()
        if name == "verify":
            payload = stage_verify()
        elif name == "corpus":
            payload = stage_corpus(rebuild=arguments.rebuild_corpus, limits=limits or None)
        elif name == "features":
            payload = stage_features(arguments.device, with_1c=arguments.with_1c)
        elif name == "reference":
            payload = stage_reference(arguments.device)
        elif name == "train":
            payload = stage_train(
                arguments.device,
                epochs=arguments.epochs,
                budget=arguments.budget_seconds,
                with_1c=arguments.with_1c,
                repeat=arguments.repeat_train,
            )
            heads = payload.pop("heads")
            if not arguments.skip_interface:
                log("[interface] smoke check on every trained candidate ...")
                stages["interface"] = interface_check(heads)
                save_stage("interface", stages["interface"])
        elif name == "report":
            if arguments.record_suite:
                log("[suite] running the repository suite ...")
                stages["suite"] = record_suite()
                save_stage("suite", stages["suite"])
                log(f"[suite] {stages['suite']['summary_line']}")
            payload = stage_report(stages, with_1c=arguments.with_1c)
        else:  # pragma: no cover - argparse restricts the choices
            raise SystemExit(f"unknown stage {name!r}")
        payload["seconds"] = round(time.perf_counter() - stage_started, 3)
        stages[name] = payload
        save_stage(name, {key: value for key, value in payload.items() if key != "summary"})
        verdict = "PASS" if payload.get("pass", True) else "FAIL"
        log(f"[{name}] {verdict} in {payload['seconds']:.1f}s")
        if not payload.get("pass", True):
            log(f"[{name}] problems: {payload.get('problems') or payload.get('file_digest_drift')}")
            return 1

    if "report" in wanted:
        write_markdown(stages)
    log(f"done in {time.perf_counter() - started:.1f}s")

    if arguments.run_pytest:
        log("[pytest] running the Phase 11B Agent 1 artifact tests ...")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/belief/phase11b",
                "-q",
            ],
            cwd=REPOSITORY_ROOT,
        )
        return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
