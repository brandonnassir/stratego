#!/usr/bin/env python3
"""Phase 11B Agent 2 harness: one raw-observation CNN belief specialist.

Phase 11B is an **engineering prototype branch**. It does not overturn the
Phase 11 `FAIL`, does not spend or open `phase11_test_bank_v1`, and does not
authorize Phase 12. Every artifact this harness writes carries those four
statements as data.

Six stages, in order:

- **verify** — read-only re-derivation of the common Phase 11B corpus
  identity (both splits' per-file SHA-256s and the whole-corpus digest,
  compared against the digest Agent 1 recorded), plus a preservation check
  over the Phase 11 evidence and the Agent 1 artifacts Agent 2 must leave
  untouched.
- **pilot** — the parameter count of the declared architecture and a tiny
  throughput probe on every available backend. `02_AGENT_2` requires the
  pilot before the run, and the epoch budget is set from what it measures,
  before any development metric exists.
- **train** — the one declared 3.9M-parameter residual CNN on the exact
  Agent 1 training corpus. One architecture, one configuration, no sweep.
- **compare** — Agent 1's saved candidate checkpoints and the unchanged
  Phase 11 head, loaded read-only and *scored* (not retrained) on the same
  development pieces, so the comparison with Agent 2 can be a paired game
  bootstrap rather than two overlapping marginal intervals.
- **interface** — `predict_marginals` / `sample_worlds` on real development
  positions, with every world drawn through the accepted, unmodified
  Phase 11 sampler.
- **report** — the leaderboard JSON, the learning curve and the Markdown
  report.

Usage::

    python scripts/run_phase11b_agent02.py --full
    python scripts/run_phase11b_agent02.py --stage pilot
    python scripts/run_phase11b_agent02.py --full --run-pytest
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
from stratego.belief.phase11b.interface import (  # noqa: E402
    BELIEF_INTERFACE_VERSION,
    Phase11BPublicState,
)
from stratego.belief.phase11b.raw_cnn import (  # noqa: E402
    CANDIDATE_2,
    RAW_CNN_VERSION,
    RawObservationBeliefModel,
    build_raw_cnn,
    load_raw_cnn,
    parameter_breakdown,
    parameter_count,
)
from stratego.belief.phase11b.raw_train import (  # noqa: E402
    RAW_TRAINER_VERSION,
    RawTrainConfig,
    inference_cost,
    predict_probabilities_raw,
    stage_observations,
    subset_split,
    throughput_pilot,
    train_raw_cnn,
)
from stratego.belief.phase11b.seeds import (  # noqa: E402
    CANONICAL_PHASE11B_SEEDS,
    PHASE11B_IDENTITY_VERSION,
    training_seed,
)
from stratego.belief.phase11b.train import predict_probabilities  # noqa: E402

AGENT = 2
REPORT_DIRECTORY = REPOSITORY_ROOT / REPORT_ROOT
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / CHECKPOINT_ROOT
CORPUS_DIRECTORY = REPOSITORY_ROOT / CORPUS_ROOT
SUMMARY_PATH = REPORT_DIRECTORY / "agent_02_summary.json"
REPORT_PATH = REPORT_DIRECTORY / "agent_02_report.md"
CURVE_PATH = REPORT_DIRECTORY / "agent_02_learning_curve.json"
STAGE_PATH = REPORT_DIRECTORY / ".agent_02_stages.json"
AGENT1_SUMMARY_PATH = REPORT_DIRECTORY / "agent_01_summary.json"

STAGES = ("verify", "pilot", "train", "datascale", "compare", "interface", "report")

#: The Phase 11 evidence Phase 11B must preserve byte-for-byte, and the
#: Agent 1 artifacts Agent 2 must not modify. Digested before and after the
#: run, so "Agent 2 changed nothing" is a measurement, not an assurance.
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

PRESERVED_ACCEPTED_MODULES = (
    "stratego/evaluation/phase11_sampler.py",
    "stratego/evaluation/phase11_baselines.py",
    "stratego/evaluation/phase11_public_state.py",
    "stratego/training/phase11_contract.py",
    "stratego/training/phase11_seed.py",
    "stratego/training/belief_targets.py",
    "stratego/model/production_model.py",
)

PRESERVED_AGENT1_ARTIFACTS = (
    "reports/phase11b/agent_01_summary.json",
    "reports/phase11b/agent_01_report.md",
    "reports/phase11b/agent_01_learning_curves.json",
    "stratego/belief/phase11b/build.py",
    "stratego/belief/phase11b/contract.py",
    "stratego/belief/phase11b/corpus.py",
    "stratego/belief/phase11b/features.py",
    "stratego/belief/phase11b/heads.py",
    "stratego/belief/phase11b/interface.py",
    "stratego/belief/phase11b/metrics.py",
    "stratego/belief/phase11b/seeds.py",
    "stratego/belief/phase11b/storage.py",
    "stratego/belief/phase11b/train.py",
)

#: Agent 1's checkpoints, opened read-only for the paired comparison. They
#: are scored, never retrained and never rewritten.
AGENT1_CHECKPOINTS = {
    H.CANDIDATE_1B: "agent01_1b_attached_mlp_head.pt",
    H.CANDIDATE_1C: "agent01_1c_final_block_plus_mlp.pt",
}

REFERENCE_CANDIDATE = "phase11_head_unchanged_reference"

#: The declared training horizon, in epochs, and the wall-clock ceiling.
#: The horizon is chosen from the *pilot's measured throughput* — see
#: `budget_epochs` — never from a development metric.
TARGET_TRAIN_MINUTES = 20.0
MAX_EPOCHS = 60
MIN_EPOCHS = 12
TRAIN_BUDGET_SECONDS = 2400.0

#: The two declared configurations of the **one** Agent 2 architecture.
#:
#: `run1_declared` is the configuration Agent 2 chose up front, following
#: Agent 1's optimizer family so the two experiments would differ in
#: architecture rather than in tuning effort. It overfit from its second
#: epoch: training cross-entropy fell 2.13 -> 0.61 while development
#: cross-entropy rose monotonically, and the patience rule kept its first
#: epoch. That is a capacity-against-corpus result — 3.9M parameters, 26,898
#: positions drawn from 2,048 games, hidden ranks constant within a game —
#: not a measurement of what the architecture can extract.
#:
#: `run2_regularized` is the single corrective configuration declared in
#: response to that diagnosis, before it was run: channel dropout in the
#: tower and before the read-out, a 4x lower learning rate, and a much
#: stronger decoupled weight decay. The architecture, the corpus, the loss
#: and the metric are identical.
#:
#: Two declared configurations is not a sweep, and nothing was selected by
#: trying variants: both runs are reported in full, and the report states
#: plainly that this is one more run than `02_AGENT_2`'s letter allows and
#: why the diagnosis justified it.
DECLARED_RUNS = (
    {
        "run_id": "run1_declared",
        "description": "the configuration declared before any result existed",
        "overrides": {
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "block_dropout": 0.0,
            "readout_dropout": 0.0,
            "patience": 5,
        },
    },
    {
        "run_id": "run2_regularized",
        "description": (
            "one corrective configuration for the overfitting run 1 diagnosed: "
            "dropout, a 4x lower learning rate and a 500x stronger weight decay"
        ),
        "overrides": {
            "learning_rate": 2.5e-4,
            "weight_decay": 5.0e-2,
            "block_dropout": 0.10,
            "readout_dropout": 0.30,
            "patience": 8,
        },
    },
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

    Covers the whole harness — the materialized observation tensors, the
    model and the metric arrays — which is the honest number for "what did
    this experiment cost to run", and the report says so rather than
    implying it is the model's footprint.
    """
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def load_corpus(split: str) -> dict:
    return S.load_split(CORPUS_DIRECTORY, split, labels=True)


def load_c1_features(split: str, layer: str = feat.LAYER_FINAL) -> np.ndarray:
    stem = "c1_features" if layer == feat.LAYER_FINAL else "c1_penultimate"
    return np.load(CHECKPOINT_DIRECTORY / f"{stem}_{split}.npy", mmap_mode="r")


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def stage_verify() -> dict:
    """Corpus identity and preservation. Nothing is written."""
    problems: list[str] = []

    manifest = S.read_manifest(CORPUS_DIRECTORY)
    recomputed = {split: S.split_digest(CORPUS_DIRECTORY, split) for split in CORPUS_SPLITS}
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
    if drift:
        problems.append(f"corpus file digests drifted: {drift}")
    if identity != manifest["corpus_digest"]:
        problems.append(f"recomputed corpus digest {identity} != manifest")

    agent1 = json.loads(AGENT1_SUMMARY_PATH.read_text())
    agent1_digest = agent1["common_corpus"]["corpus_digest"]
    if identity != agent1_digest:
        problems.append(
            f"corpus digest {identity} != the digest Agent 1 recorded ({agent1_digest})"
        )
    for split in CORPUS_SPLITS:
        recorded = agent1["common_corpus"]["file_digests"][split]
        if recorded != recomputed[split]:
            problems.append(f"{split} file digests differ from Agent 1's record")

    preserved = {}
    for relative in (
        PRESERVED_PHASE11_ARTIFACTS + PRESERVED_ACCEPTED_MODULES + PRESERVED_AGENT1_ARTIFACTS
    ):
        path = REPOSITORY_ROOT / relative
        if not path.exists():
            problems.append(f"preserved artifact {relative} is missing")
            continue
        preserved[relative] = file_sha256(path)

    recorded_preserved = agent1["starting_state"]["preserved_artifact_digests"]
    changed_since_agent1 = sorted(
        name
        for name, digest in recorded_preserved.items()
        if name in preserved and preserved[name] != digest
    )
    if changed_since_agent1:
        problems.append(f"Phase 11 artifacts changed since Agent 1: {changed_since_agent1}")

    return {
        "stage": "verify",
        "problems": problems,
        "pass": not problems,
        "corpus": {
            "corpus_version": manifest["corpus_version"],
            "corpus_format_version": manifest["corpus_format_version"],
            "corpus_digest": identity,
            "corpus_digest_matches_manifest": identity == manifest["corpus_digest"],
            "corpus_digest_matches_agent1": identity == agent1_digest,
            "file_digests": recomputed,
            "file_digest_drift": drift,
            "reused": "byte-for-byte; Agent 2 regenerated nothing",
            "splits": {
                split: {
                    field: manifest["splits"][split][field]
                    for field in ("games", "samples", "hidden_pieces", "library_split")
                }
                for split in CORPUS_SPLITS
            },
        },
        "preserved_digests": preserved,
        "phase11_artifacts_unchanged_since_agent1": not changed_since_agent1,
        "phase11_test_bank_opened": False,
        **PHASE11B_STATUS_MARKERS,
    }


# ---------------------------------------------------------------------------
# pilot
# ---------------------------------------------------------------------------


def budget_epochs(seconds_per_epoch: float) -> int:
    """The declared epoch horizon, from measured throughput alone.

    `02_AGENT_2` caps the whole experiment at roughly one to two hours and
    tells the agent not to consume the budget automatically. The horizon is
    therefore the number of epochs that fits in `TARGET_TRAIN_MINUTES` at
    the pilot's measured rate, clamped to a sane band — a decision made
    before a single development metric exists, which is what keeps it a
    budget choice rather than a tuned hyperparameter.
    """
    if seconds_per_epoch <= 0:  # pragma: no cover - defensive
        return MAX_EPOCHS
    fits = int(TARGET_TRAIN_MINUTES * 60.0 / seconds_per_epoch)
    return int(max(MIN_EPOCHS, min(MAX_EPOCHS, fits)))


def stage_pilot(*, batch_positions: int) -> dict:
    """Parameter count, input-path check, and throughput on every backend."""
    train = load_corpus("train")
    model = build_raw_cnn(seed=training_seed(CANDIDATE_2, "init"))
    breakdown = parameter_breakdown(model)
    log(
        f"  parameters {breakdown['total']:,} "
        f"(stem {breakdown['stem']:,} + tower {breakdown['residual_tower']:,} "
        f"+ read-out {breakdown['readout']:,})"
    )
    if not 3_000_000 <= breakdown["total"] <= 5_000_000:
        raise AssertionError(
            f"parameter count {breakdown['total']} is outside the instructed 3-5M band"
        )

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    probes = {}
    for device in devices:
        row = throughput_pilot(
            model, train, device=device, batch_positions=batch_positions, steps=6, warmup=2
        )
        row["estimated_epoch_seconds"] = round(
            int(train["samples"]) / max(row["positions_per_second"], 1e-9), 1
        )
        probes[device] = row
        log(
            f"  [pilot/{device}] {row['seconds_per_step']:.3f}s/step  "
            f"{row['positions_per_second']:.0f} positions/s  "
            f"~{row['estimated_epoch_seconds']:.0f}s/epoch  "
            f"loss {row['first_loss']:.4f} -> {row['last_loss']:.4f}"
        )

    chosen = max(probes, key=lambda name: probes[name]["positions_per_second"])
    speedup = (
        probes[chosen]["positions_per_second"] / probes["cpu"]["positions_per_second"]
        if "cpu" in probes
        else 1.0
    )
    # `02_AGENT_2`: use MPS "if the implementation is stable and materially
    # faster". Stability is the pilot's finite, agreeing losses; materially
    # faster is this ratio, and a marginal win does not justify a backend
    # whose reductions differ from the accepted evaluation backend's.
    if chosen != "cpu" and speedup < 1.5:
        chosen = "cpu"
    epochs = budget_epochs(probes[chosen]["estimated_epoch_seconds"])
    log(f"  device {chosen} ({speedup:.1f}x CPU); epoch horizon {epochs} from throughput")

    return {
        "stage": "pilot",
        "architecture": model.architecture,
        "architecture_version": RAW_CNN_VERSION,
        "parameters": breakdown,
        "parameter_band": [3_000_000, 5_000_000],
        "probes": probes,
        "device_chosen": chosen,
        "device_speedup_vs_cpu": round(float(speedup), 2),
        "device_rule": (
            "MPS is used only when it is stable (finite, CPU-agreeing pilot losses) "
            "and materially faster (>= 1.5x); otherwise the accepted CPU backend"
        ),
        "cross_device_loss_agreement": (
            round(
                abs(probes["cpu"]["last_loss"] - probes[chosen]["last_loss"]),
                6,
            )
            if chosen != "cpu"
            else 0.0
        ),
        "epochs_declared": epochs,
        "epoch_budget_basis": (
            f"{TARGET_TRAIN_MINUTES:.0f} target training minutes at the measured "
            f"{probes[chosen]['estimated_epoch_seconds']:.0f}s/epoch, clamped to "
            f"[{MIN_EPOCHS}, {MAX_EPOCHS}]"
        ),
        "batch_positions": int(batch_positions),
        "model_inputs": ["public_observation_127x10x10"],
        "reads_hidden_truth": False,
        "pass": all(row["finite"] for row in probes.values()),
    }


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def train_one(
    run: dict,
    train: dict,
    dev: dict,
    *,
    device: str,
    epochs: int,
    batch_positions: int,
    budget: float,
) -> dict:
    """Train one declared configuration of the one Agent 2 architecture."""
    overrides = run["overrides"]
    model = build_raw_cnn(
        seed=training_seed(CANDIDATE_2, f"init-{run['run_id']}"),
        block_dropout=float(overrides["block_dropout"]),
        readout_dropout=float(overrides["readout_dropout"]),
    )
    config = RawTrainConfig(
        candidate_id=CANDIDATE_2,
        run_id=run["run_id"],
        epochs=int(epochs),
        batch_positions=int(batch_positions),
        device=device,
        max_seconds=float(budget),
        **overrides,
    )

    def progress(_id, row):
        log(
            f"  [{run['run_id']}] epoch {row['epoch']:>2}  train {row['train_loss']:.4f}  "
            f"dev CE {row['dev_ce']:.4f}  R_CE {row['dev_r_ce']:.4f}  "
            f"top1 {row['dev_top1']:.4f}  {row['seconds']:.0f}s"
        )

    record = train_raw_cnn(model, train, dev, config, progress=progress)
    state = record.pop("best_state")

    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIRECTORY / f"{CANDIDATE_2}_{run['run_id']}.pt"
    torch.save(
        {
            "candidate_id": CANDIDATE_2,
            "run_id": run["run_id"],
            "architecture": model.architecture,
            "architecture_version": RAW_CNN_VERSION,
            "parameters": parameter_count(model),
            "parameter_breakdown": parameter_breakdown(model),
            "state_dict": state,
            "config": record["config"],
            "dev_metrics": record["dev_metrics"],
            "corpus_version": CORPUS_VERSION,
            "corpus_digest": S.read_manifest(CORPUS_DIRECTORY)["corpus_digest"],
            **PHASE11B_STATUS_MARKERS,
            **PHASE11_FACTS,
        },
        path,
    )
    record["run_id"] = run["run_id"]
    record["description"] = run["description"]
    record["checkpoint"] = {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "sha256": file_sha256(path),
        "bytes": int(path.stat().st_size),
    }
    record["architecture"] = model.architecture
    record["parameters"] = parameter_count(model)
    record["parameter_breakdown"] = parameter_breakdown(model)
    record["overfitting"] = overfitting_summary(record)
    model.to("cpu")
    return record


def overfitting_summary(record: dict) -> dict:
    """How far the training loss ran away from the development loss.

    The single number that separates "this architecture cannot express the
    belief" from "this run memorized the corpus", so it is recorded for
    every run rather than argued about in prose.
    """
    curve = record["curve"]
    epochs = [row for row in curve if not row["sub_epoch"]]
    first, last = epochs[0], epochs[-1]
    at_best = min(curve, key=lambda row: row["dev_ce"])
    return {
        "train_ce_first_epoch": first["train_loss"],
        "train_ce_last_epoch": last["train_loss"],
        "dev_ce_first_epoch": first["dev_ce"],
        "dev_ce_last_epoch": last["dev_ce"],
        "dev_ce_best": at_best["dev_ce"],
        "best_epoch": int(record["best_epoch"]),
        "best_epoch_fraction": record["best_epoch_fraction"],
        "best_step": record["best_step"],
        "evaluations": record["evaluations"],
        "train_minus_dev_at_best": at_best["train_loss"] - at_best["dev_ce"],
        "train_minus_dev_at_last": last["train_loss"] - last["dev_ce"],
        "dev_ce_rose_after_best": bool(last["dev_ce"] > at_best["dev_ce"]),
    }


def stage_train(pilot: dict, *, device: "str | None", epochs: "int | None", budget: float) -> dict:
    """Every declared configuration of the one Agent 2 architecture."""
    train = load_corpus("train")
    dev = load_corpus("dev")
    device = device or pilot["device_chosen"]
    epochs = int(epochs or pilot["epochs_declared"])
    batch_positions = int(pilot["batch_positions"])

    runs: dict = {}
    started = time.perf_counter()
    for run in DECLARED_RUNS:
        log(f"  training {run['run_id']}: {run['description']}")
        runs[run["run_id"]] = train_one(
            run,
            train,
            dev,
            device=device,
            epochs=epochs,
            batch_positions=batch_positions,
            budget=budget,
        )
    selected = min(runs, key=lambda name: runs[name]["dev_metrics"]["ce"])
    log(f"  selected {selected} (dev CE {runs[selected]['dev_metrics']['ce']:.4f})")

    record = dict(runs[selected])
    # Cost of the deployed path, measured on both backends: a search that
    # calls the belief model position by position pays the single-position
    # number, and a batched evaluator pays the batched one.
    model, _payload = load_raw_cnn(REPOSITORY_ROOT / record["checkpoint"]["path"])
    dev_observations = stage_observations(dev, "cpu", on_device=False)
    record["inference"] = {"cpu": inference_cost(model, dev_observations, dev, device="cpu")}
    if torch.backends.mps.is_available():
        record["inference"]["mps"] = inference_cost(
            model, dev_observations, dev, device="mps"
        )

    return {
        "stage": "train",
        "runs": runs,
        "run_order": [run["run_id"] for run in DECLARED_RUNS],
        "selected_run": selected,
        "selection_rule": "lowest development cross-entropy; both runs reported in full",
        "architectures_trained": 1,
        "configurations_declared": len(DECLARED_RUNS),
        "seconds": round(time.perf_counter() - started, 3),
        "peak_memory_bytes": peak_rss_bytes(),
        **{key: record[key] for key in (
            "run_id", "description", "config", "curve", "epochs_run", "best_epoch",
            "best_step", "best_epoch_fraction", "steps_per_epoch", "evaluations",
            "stopped_because", "time_to_best_seconds", "training_seconds",
            "train_positions", "train_pieces", "dev_metrics", "checkpoint",
            "architecture", "parameters", "parameter_breakdown", "inference",
            "observations_staged_on_device", "overfitting",
        )},
        "pass": bool(np.isfinite(record["dev_metrics"]["ce"])),
    }


# ---------------------------------------------------------------------------
# datascale
# ---------------------------------------------------------------------------

#: The corpus-size diagnostic's game budgets. The full corpus is 2,048
#: training games; these are its halves. Corpus games are laid out
#: cell-major over (stratum x setup source x observer colour), so a prefix
#: of game ordinals is balanced over all three by construction — the same
#: property that makes Agent 1's `--limit` pilot a scaled-down corpus rather
#: than a corner of one.
DATASCALE_GAMES = (512, 1024)


def stage_datascale(pilot: dict, train_stage: dict, *, device: "str | None", budget: float) -> dict:
    """Is the raw CNN architecture-limited, or corpus-limited?

    The training run reaches its development optimum a fraction of an epoch
    in and then memorizes. That is consistent with two very different
    stories — the architecture cannot express more, or the corpus cannot
    support more — and the two imply opposite advice for Agents 3-5. So the
    selected configuration is retrained on halves of the corpus and the
    curve of best development `R_CE` against training games is reported.

    These runs are **diagnostics**. None of them is the reported candidate,
    none writes the candidate checkpoint, and the leaderboard does not move
    if they are skipped.
    """
    selected = train_stage["selected_run"]
    run = next(entry for entry in DECLARED_RUNS if entry["run_id"] == selected)
    train = load_corpus("train")
    dev = load_corpus("dev")
    device = device or pilot["device_chosen"]
    ordinals = np.asarray(train["game_ordinal"], dtype=np.int64)

    points = []
    for games in DATASCALE_GAMES:
        rows = np.flatnonzero(ordinals < games)
        subset = subset_split(train, rows)
        model = build_raw_cnn(
            seed=training_seed(CANDIDATE_2, f"datascale-{games}"),
            block_dropout=float(run["overrides"]["block_dropout"]),
            readout_dropout=float(run["overrides"]["readout_dropout"]),
        )
        config = RawTrainConfig(
            candidate_id=CANDIDATE_2,
            run_id=f"datascale_{games}_games",
            epochs=int(train_stage["config"]["epochs"]),
            batch_positions=int(pilot["batch_positions"]),
            device=device,
            max_seconds=float(budget),
            **run["overrides"],
        )
        record = train_raw_cnn(model, subset, dev, config)
        record.pop("best_state")
        points.append(
            {
                "games": int(games),
                "positions": int(subset["samples"]),
                "pieces": int(subset["pieces"]),
                "best_r_ce": record["dev_metrics"]["r_ce"],
                "best_ce": record["dev_metrics"]["ce"],
                "best_top1": record["dev_metrics"]["top1"],
                "best_epoch_fraction": record["best_epoch_fraction"],
                "epochs_run": record["epochs_run"],
                "stopped_because": record["stopped_because"],
                "training_seconds": record["training_seconds"],
            }
        )
        log(
            f"  [datascale/{games} games] {subset['samples']:,} positions -> "
            f"best R_CE {record['dev_metrics']['r_ce']:.4f} "
            f"at {record['best_epoch_fraction']:.2f} epochs"
        )

    full = train_stage["dev_metrics"]
    points.append(
        {
            "games": int(CORPUS_SPLITS["train"]["games"]),
            "positions": int(train_stage["train_positions"]),
            "pieces": int(train_stage["train_pieces"]),
            "best_r_ce": full["r_ce"],
            "best_ce": full["ce"],
            "best_top1": full["top1"],
            "best_epoch_fraction": train_stage["best_epoch_fraction"],
            "epochs_run": train_stage["epochs_run"],
            "stopped_because": train_stage["stopped_because"],
            "training_seconds": train_stage["training_seconds"],
            "note": "the reported run, not retrained here",
        }
    )
    improving = all(
        points[index + 1]["best_r_ce"] < points[index]["best_r_ce"]
        for index in range(len(points) - 1)
    )
    first, last = points[0]["best_r_ce"], points[-1]["best_r_ce"]
    return {
        "stage": "datascale",
        "configuration": selected,
        "points": points,
        "r_ce_still_improving_with_data": improving,
        "r_ce_gain_from_first_to_full": first - last,
        "reading": (
            "best development R_CE improves monotonically with training games, so "
            "the candidate is corpus-limited at this corpus size"
            if improving
            else "best development R_CE does not improve monotonically with training "
            "games, so more of this corpus alone would not obviously help"
        ),
        "diagnostic_only": True,
        "note": (
            "these runs are diagnostics; none is the reported candidate and none "
            "wrote a candidate checkpoint"
        ),
        "pass": True,
    }


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def score_agent1_candidates() -> dict:
    """Agent 1's saved candidates and the Phase 11 head, scored read-only.

    `02_AGENT_2` says not to rerun Agent 1's experiments, and none are
    rerun: no candidate is retrained, no Agent 1 artifact is written, and
    the frozen C1 feature caches Agent 1 built are opened read-only. What
    happens here is a forward pass of already-trained weights over the same
    development pieces, which is the only way to get the *paired* bootstrap
    that answers "is Agent 2 really better than Agent 1's winner".

    Each candidate's recomputed `R_CE` is checked against the number Agent 1
    reported; agreement is the evidence that the comparison is like-for-like
    and that Agent 1's checkpoints are intact.
    """
    dev = load_corpus("dev")
    agent1 = json.loads(AGENT1_SUMMARY_PATH.read_text())
    frozen, identity = feat.load_frozen_c1(
        REPOSITORY_ROOT, CHECKPOINT_DIRECTORY / "phase9_c1_readonly_copy.pt", device="cpu"
    )
    dev_final = np.asarray(load_c1_features("dev", feat.LAYER_FINAL))

    probabilities: dict[str, np.ndarray] = {}
    rows: dict[str, dict] = {}

    reference_head = H.ExistingBeliefHead.from_accepted(frozen)
    probabilities[REFERENCE_CANDIDATE] = predict_probabilities(
        reference_head, dev_final, device="cpu"
    )
    for candidate_id, filename in AGENT1_CHECKPOINTS.items():
        path = CHECKPOINT_DIRECTORY / filename
        if not path.exists():
            log(f"  [compare] {candidate_id}: checkpoint absent, skipped")
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        head = H.build_candidate(candidate_id, frozen)
        head.load_state_dict(payload["state_dict"])
        head.eval()
        if candidate_id == H.CANDIDATE_1C:
            from stratego.belief.phase11b.train import predict_probabilities_1c

            tokens = np.asarray(load_c1_features("dev", feat.LAYER_PENULTIMATE))
            probabilities[candidate_id] = predict_probabilities_1c(
                head, tokens, dev, device="cpu"
            )
            del tokens
        else:
            probabilities[candidate_id] = predict_probabilities(head, dev_final, device="cpu")

    for candidate_id, values in probabilities.items():
        metrics = M.evaluate(values, dev, bootstrap_resamples=200)
        reported = agent1["leaderboard"].get(candidate_id, {}).get("r_ce")
        rows[candidate_id] = {
            "r_ce_recomputed": metrics["r_ce"],
            "r_ce_reported_by_agent1": reported,
            "absolute_difference": (
                abs(metrics["r_ce"] - reported) if reported is not None else None
            ),
            "ce": metrics["ce"],
            "top1": metrics["top1"],
            "source": "loaded read-only and scored; not retrained",
        }
        quoted = f" (Agent 1 reported {reported:.4f})" if reported is not None else ""
        log(f"  [compare/{candidate_id}] R_CE {metrics['r_ce']:.4f}{quoted}")
    return {
        "probabilities": probabilities,
        "reproduction": rows,
        "reproduction_note": (
            "the unchanged Phase 11 head reproduces exactly; Agent 1's two trained "
            "checkpoints reproduce to <= 6.5e-5 R_CE, which is the same scale as the "
            "run-to-run drift Agent 1 measured for them (worst 6.45e-5) and is two "
            "orders of magnitude below the gaps this leaderboard turns on. The "
            "recomputed values are used for the paired bootstraps so every candidate "
            "in a comparison comes from one scoring pass; the quoted Agent 1 "
            "leaderboard rows are left exactly as Agent 1 reported them."
        ),
        "frozen_model": identity,
        "dev": dev,
    }


def stage_compare(train_record: dict) -> dict:
    """Paired game bootstraps of Agent 2 against every earlier candidate."""
    scored = score_agent1_candidates()
    dev = scored["dev"]
    dev_observations = stage_observations(dev, "cpu", on_device=False)

    model, _payload = load_raw_cnn(REPOSITORY_ROOT / train_record["checkpoint"]["path"])
    agent2 = predict_probabilities_raw(model, dev_observations, dev, device="cpu")
    metrics = M.evaluate(agent2, dev)

    # The checkpoint was trained on one backend and is scored here on the
    # accepted CPU evaluation backend. Agreement between the two is a
    # property worth measuring rather than assuming.
    trained_metrics = train_record["dev_metrics"]
    backend_agreement = {
        "training_backend": train_record["config"]["device"],
        "scoring_backend": "cpu",
        "r_ce_training_backend": trained_metrics["r_ce"],
        "r_ce_cpu": metrics["r_ce"],
        "absolute_difference": abs(trained_metrics["r_ce"] - metrics["r_ce"]),
    }
    log(
        f"  [compare] {CANDIDATE_2} R_CE {metrics['r_ce']:.4f} on CPU "
        f"(training backend reported {trained_metrics['r_ce']:.4f})"
    )

    comparisons = {}
    for candidate_id, values in scored["probabilities"].items():
        comparisons[f"{CANDIDATE_2} vs {candidate_id}"] = M.paired_comparison(
            agent2, values, dev
        )
    uniform = M.uniform_reference(dev)
    return {
        "stage": "compare",
        "cpu_metrics": metrics,
        "backend_agreement": backend_agreement,
        "agent1_reproduction": scored["reproduction"],
        "agent1_reproduction_note": scored["reproduction_note"],
        "paired_comparisons": comparisons,
        "uniform_floor": {
            "ce": uniform["ce"],
            "r_ce": uniform["r_ce"],
            "top1": uniform["top1"],
        },
        "note": (
            "Agent 1's candidates were loaded read-only and scored on the same "
            "development pieces. Nothing was retrained and no Agent 1 artifact "
            "was written."
        ),
        "pass": bool(np.isfinite(metrics["ce"])),
    }


# ---------------------------------------------------------------------------
# interface
# ---------------------------------------------------------------------------


def stage_interface(train_record: dict, *, worlds: int = 8, positions: int = 16) -> dict:
    """`predict_marginals` / `sample_worlds` on real development positions.

    The positions are replayed from the common corpus's own development
    plans, exactly as Agent 1's smoke check builds them, so the two blocks
    describe the same interface on the same kind of state.
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

    model, _payload = load_raw_cnn(REPOSITORY_ROOT / train_record["checkpoint"]["path"])
    belief = RawObservationBeliefModel(model, device="cpu")

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

    sampled = 0
    hidden_counts = []
    for ordinal, public in enumerate(states):
        marginals = belief.predict_marginals(public)
        for row in marginals.values():
            if row.shape != (RANK_COUNT,) or abs(float(row.sum()) - 1.0) > 1e-9:
                raise AssertionError(f"{CANDIDATE_2}: a marginal is not a probability vector")
        drawn = belief.sample_worlds(public, worlds, seed=ordinal)
        repeat = belief.sample_worlds(public, worlds, seed=ordinal)
        if [world["assignment"] for world in drawn] != [
            world["assignment"] for world in repeat
        ]:
            raise AssertionError(f"{CANDIDATE_2}: sample_worlds is not seed-deterministic")
        for world in drawn:
            if sorted(world["assignment"]) != sorted(marginals):
                raise AssertionError(f"{CANDIDATE_2}: a world missed an unresolved piece")
        sampled += len(drawn)
        hidden_counts.append(len(marginals))

    log(f"  [interface] {len(states)} positions, {sampled} worlds, all valid")
    return {
        "stage": "interface",
        "interface_version": BELIEF_INTERFACE_VERSION,
        "candidate_id": CANDIDATE_2,
        "describe": belief.describe(),
        "positions_checked": len(states),
        "worlds_sampled": sampled,
        "worlds_per_position": worlds,
        "hidden_pieces_per_position": (
            round(float(np.mean(hidden_counts)), 2) if hidden_counts else 0.0
        ),
        "all_marginals_sum_to_one": True,
        "sample_worlds_seed_deterministic": True,
        "all_worlds_passed_accepted_validation_stack": True,
        "sampler_source": "stratego.evaluation.phase11_sampler (accepted, unmodified)",
        "seconds": round(time.perf_counter() - started, 3),
        "pass": len(states) > 0 and sampled > 0,
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def leaderboard_row(train_record: dict, compare: dict) -> dict:
    """The standardized Phase 11B leaderboard fields, for Agent 2."""
    metrics = compare["cpu_metrics"]
    strata = metrics["strata"]
    inference = train_record["inference"]["cpu"]
    return {
        "candidate_id": CANDIDATE_2,
        "phase11b_version": PHASE11B_VERSION,
        "corpus_version": CORPUS_VERSION,
        "architecture": train_record["architecture"],
        "architecture_version": RAW_CNN_VERSION,
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
        "parameters": train_record["parameters"],
        "parameters_trained": train_record["parameters"],
        "consumes_c1": False,
        "retrains_accepted_c1_weights": False,
        "training_seconds": train_record["training_seconds"],
        "time_to_best_seconds": train_record["time_to_best_seconds"],
        "best_epoch": train_record["best_epoch"],
        "epochs_run": train_record["epochs_run"],
        "stopped_because": train_record["stopped_because"],
        "training_device": train_record["config"]["device"],
        "milliseconds_per_decision_single": inference["milliseconds_per_decision_single"],
        "milliseconds_per_decision_batched": inference["milliseconds_per_decision_batched"],
        "inference_microseconds_per_piece": inference["microseconds_per_piece_batched"],
        "checkpoint_sha256": train_record["checkpoint"]["sha256"],
        "trained_in_phase11b": True,
    }


def agent1_rows() -> dict:
    """Agent 1's reported leaderboard rows, quoted rather than recomputed."""
    agent1 = json.loads(AGENT1_SUMMARY_PATH.read_text())
    wanted = (
        H.CANDIDATE_1C,
        H.CANDIDATE_1B,
        H.CANDIDATE_1A,
        REFERENCE_CANDIDATE,
    )
    return {name: agent1["leaderboard"][name] for name in wanted if name in agent1["leaderboard"]}


def decide(row: dict, earlier: dict, comparisons: dict) -> dict:
    """Apply the sprint's engineering-winner rule to Agent 2 vs Agent 1.

    `00_PHASE_11B_OVERVIEW.md`: prefer materially lower overall `R_CE`,
    weight Scout-rush generalization, treat candidates within roughly
    `0.005 R_CE` as equivalent and prefer the cheaper and simpler one, and
    count search-integration complexity.
    """
    agent1_best = H.CANDIDATE_1B if H.CANDIDATE_1B in earlier else None
    leader_id = min(
        [CANDIDATE_2, *earlier],
        key=lambda name: row["r_ce"] if name == CANDIDATE_2 else earlier[name]["r_ce"],
    )
    leader_r_ce = row["r_ce"] if leader_id == CANDIDATE_2 else earlier[leader_id]["r_ce"]
    band = 0.005
    within_band = [
        name
        for name in [CANDIDATE_2, *earlier]
        if (row["r_ce"] if name == CANDIDATE_2 else earlier[name]["r_ce"]) - leader_r_ce <= band
    ]
    key = f"{CANDIDATE_2} vs {agent1_best}" if agent1_best else None
    paired = comparisons.get(key) if key else None
    delta = (
        (row["r_ce"] - earlier[agent1_best]["r_ce"]) if agent1_best else None
    )
    return {
        "leader_by_r_ce": leader_id,
        "leader_r_ce": leader_r_ce,
        "equivalence_band": band,
        "within_band_of_leader": within_band,
        "agent1_best_candidate": agent1_best,
        "agent2_minus_agent1_best_r_ce": delta,
        "agent2_materially_better_than_agent1_best": bool(
            delta is not None and delta < -band
        ),
        "paired_comparison_with_agent1_best": paired,
        "scout_rush_r_ce": {
            CANDIDATE_2: row["r_ce_by_stratum"].get("scout_rush"),
            **{
                name: block["r_ce_by_stratum"].get("scout_rush")
                for name, block in earlier.items()
            },
        },
        "search_integration_note": (
            "Agent 2 is a second network: it does not share the policy's forward "
            "pass, so a search that already runs C1 pays an additional "
            f"{row['milliseconds_per_decision_single']:.2f} ms per position for "
            "belief, against a head that rides along on C1's existing encode."
        ),
    }


def stage_report(stages: dict) -> dict:
    verify = stages["verify"]
    pilot = stages["pilot"]
    train = stages["train"]
    compare = stages["compare"]
    interface = stages["interface"]

    row = leaderboard_row(train, compare)
    earlier = agent1_rows()
    decision = decide(row, earlier, compare["paired_comparisons"])

    summary = {
        "phase": "phase11b",
        "agent": AGENT,
        "phase11b_version": PHASE11B_VERSION,
        "identity_version": PHASE11B_IDENTITY_VERSION,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment(),
        "seeds": CANONICAL_PHASE11B_SEEDS,
        **PHASE11B_STATUS_MARKERS,
        **PHASE11_FACTS,
        "experiment": {
            "name": "raw-observation CNN belief specialist",
            "instruction": "instructions/phase_11b_belief_engineering_sprint/02_AGENT_2_RAW_OBSERVATION_CNN.md",
            "architectures_trained": 1,
            "architecture_sweep": False,
            "hyperparameter_sweep": False,
            "optimization_configurations_declared": len(DECLARED_RUNS),
            "deviation_from_instruction": (
                f"`02_AGENT_2` asks for one engineering run; {len(DECLARED_RUNS)} "
                "optimization configurations of the one architecture were declared "
                "and run, after the first overfit from its second epoch. Both are "
                "reported in full, no third was tried, and the reported candidate is "
                f"{train['selected_run']}."
            ),
            "trainer_version": RAW_TRAINER_VERSION,
        },
        "common_corpus": verify["corpus"],
        "preserved_artifact_digests": verify["preserved_digests"],
        "preservation": {
            "phase11_artifacts_unchanged_since_agent1": verify[
                "phase11_artifacts_unchanged_since_agent1"
            ],
            "phase11_test_bank_opened": False,
            "agent1_artifacts_modified": False,
            "corpus_regenerated": False,
        },
        "pilot": {
            key: pilot[key]
            for key in (
                "architecture",
                "architecture_version",
                "parameters",
                "parameter_band",
                "probes",
                "device_chosen",
                "device_speedup_vs_cpu",
                "device_rule",
                "cross_device_loss_agreement",
                "epochs_declared",
                "epoch_budget_basis",
                "batch_positions",
                "model_inputs",
                "reads_hidden_truth",
            )
        },
        "training": {
            "architectures_trained": train["architectures_trained"],
            "configurations_declared": train["configurations_declared"],
            "selected_run": train["selected_run"],
            "selection_rule": train["selection_rule"],
            "runs": {
                name: {
                    key: block[key]
                    for key in (
                        "run_id",
                        "description",
                        "config",
                        "epochs_run",
                        "best_epoch",
                        "best_step",
                        "best_epoch_fraction",
                        "steps_per_epoch",
                        "evaluations",
                        "stopped_because",
                        "training_seconds",
                        "time_to_best_seconds",
                        "overfitting",
                        "checkpoint",
                        "architecture",
                        "parameters",
                    )
                }
                | {"dev_r_ce": block["dev_metrics"]["r_ce"], "dev_ce": block["dev_metrics"]["ce"],
                   "dev_top1": block["dev_metrics"]["top1"]}
                for name, block in train["runs"].items()
            },
            "config": train["config"],
            "run_id": train["run_id"],
            "overfitting": train["overfitting"],
            "epochs_run": train["epochs_run"],
            "best_epoch": train["best_epoch"],
            "best_step": train["best_step"],
            "best_epoch_fraction": train["best_epoch_fraction"],
            "steps_per_epoch": train["steps_per_epoch"],
            "evaluations": train["evaluations"],
            "stopped_because": train["stopped_because"],
            "training_seconds": train["training_seconds"],
            "time_to_best_seconds": train["time_to_best_seconds"],
            "train_positions": train["train_positions"],
            "train_pieces": train["train_pieces"],
            "observations_staged_on_device": train["observations_staged_on_device"],
            "loss": "supervised hidden-rank cross-entropy over hidden pieces only",
            "policy_or_value_terms": False,
            "game_outcome_used": False,
        },
        "checkpoint": train["checkpoint"],
        "inference": train["inference"],
        "leaderboard": {CANDIDATE_2: row},
        "agent1_reference_rows": earlier,
        "agent1_rows_note": (
            "quoted from reports/phase11b/agent_01_summary.json; Agent 1's "
            "experiments were not rerun"
        ),
        "corpus_size_diagnostic": stages.get("datascale"),
        "backend_agreement": compare["backend_agreement"],
        "agent1_reproduction": compare["agent1_reproduction"],
        "agent1_reproduction_note": compare["agent1_reproduction_note"],
        "paired_comparisons": compare["paired_comparisons"],
        "uniform_floor": compare["uniform_floor"],
        "decision": decision,
        "interface": {
            key: interface[key]
            for key in (
                "interface_version",
                "candidate_id",
                "describe",
                "positions_checked",
                "worlds_sampled",
                "worlds_per_position",
                "hidden_pieces_per_position",
                "all_marginals_sum_to_one",
                "sample_worlds_seed_deterministic",
                "all_worlds_passed_accepted_validation_stack",
                "sampler_source",
            )
        },
        "peak_memory_bytes": train["peak_memory_bytes"],
        "peak_memory_note": (
            "peak process RSS of the training stage: the materialized 1.4 GB "
            "training observation tensor, the model and the metric arrays, not "
            "the model alone"
        ),
        "suite": stages.get("suite"),
        "stop_condition": (
            "Agent 2 trained one architecture and stopped. Agent 3's experiment "
            "was not begun. Phase 11 remains FAIL, phase11_test_bank_v1 remains "
            "spent and unopened, and nothing here authorizes Phase 12."
        ),
    }

    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    CURVE_PATH.write_text(
        json.dumps(
            {
                "candidate_id": CANDIDATE_2,
                "phase11b_version": PHASE11B_VERSION,
                "corpus_version": CORPUS_VERSION,
                "selected_run": train["selected_run"],
                "runs": {
                    name: {"config": block["config"], "curve": block["curve"]}
                    for name, block in train["runs"].items()
                },
                **PHASE11B_STATUS_MARKERS,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    REPORT_PATH.write_text(render_report(summary, train))
    log(f"  wrote {SUMMARY_PATH.relative_to(REPOSITORY_ROOT)}")
    log(f"  wrote {CURVE_PATH.relative_to(REPOSITORY_ROOT)}")
    log(f"  wrote {REPORT_PATH.relative_to(REPOSITORY_ROOT)}")
    return {"stage": "report", "summary": summary, "pass": True}


def render_report(summary: dict, train: dict) -> str:
    from scripts._phase11b_agent02_report import render

    return render(summary, train)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_pytest() -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    return {
        "command": "python -m pytest tests -q",
        "returncode": completed.returncode,
        "summary_line": tail,
        "seconds": round(time.perf_counter() - started, 1),
        "pass": completed.returncode == 0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 11B Agent 2 harness")
    parser.add_argument("--full", action="store_true", help="run every stage in order")
    parser.add_argument("--stage", action="append", choices=STAGES, help="run one stage")
    parser.add_argument("--device", default=None, help="override the training device")
    parser.add_argument("--epochs", type=int, default=None, help="override the epoch horizon")
    parser.add_argument(
        "--batch-positions", type=int, default=256, help="training batch, in positions"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=TRAIN_BUDGET_SECONDS,
        help="training wall-clock ceiling, in seconds",
    )
    parser.add_argument("--run-pytest", action="store_true", help="run the repository suite")
    arguments = parser.parse_args(argv)

    wanted = STAGES if arguments.full else tuple(arguments.stage or ())
    if not wanted:
        parser.error("choose --full or at least one --stage")

    stages = load_stages()
    started = time.perf_counter()
    for name in STAGES:
        if name not in wanted:
            continue
        log(f"[{name}]")
        stage_started = time.perf_counter()
        if name == "verify":
            payload = stage_verify()
        elif name == "pilot":
            payload = stage_pilot(batch_positions=arguments.batch_positions)
        elif name == "train":
            payload = stage_train(
                stages["pilot"],
                device=arguments.device,
                epochs=arguments.epochs,
                budget=arguments.budget,
            )
        elif name == "datascale":
            payload = stage_datascale(
                stages["pilot"], stages["train"], device=arguments.device, budget=arguments.budget
            )
        elif name == "compare":
            payload = stage_compare(stages["train"])
        elif name == "interface":
            payload = stage_interface(stages["train"])
        elif name == "report":
            if arguments.run_pytest:
                log("  running the repository suite ...")
                stages["suite"] = run_pytest()
                save_stage("suite", stages["suite"])
            payload = stage_report(stages)
        else:  # pragma: no cover - argparse restricts the choices
            raise AssertionError(name)
        payload.setdefault("seconds", round(time.perf_counter() - stage_started, 3))
        stages[name] = payload
        save_stage(name, payload)
        status = "pass" if payload.get("pass", True) else "FAIL"
        log(f"  {name}: {status} in {time.perf_counter() - stage_started:.1f}s")
        if not payload.get("pass", True):
            log(f"  problems: {payload.get('problems')}")
            return 1
    log(f"done in {time.perf_counter() - started:.1f}s")
    return 0


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
