#!/usr/bin/env python3
"""Phase 11B Agent 4 harness: a fusion CNN over the raw observation and C1.

Phase 11B is an **engineering prototype branch**. It does not overturn the
Phase 11 `FAIL`, does not spend or open `phase11_test_bank_v1`, and does not
authorize Phase 12. Every artifact this harness writes carries those four
statements as data.

Eight stages, in order:

- **verify** — read-only re-derivation of the common Phase 11B corpus
  identity, plus a preservation check over the Phase 11 evidence and the
  Agent 1, 2 and 3 artifacts Agent 4 must leave untouched.
- **cache** — Agent 3's frozen C1 field cache, reused after its digests are
  checked against what Agent 3 recorded, plus the fused two-branch input the
  shared trainer stages. A random sample of every split is re-fused from the
  corpus bytes and Agent 3's cache and compared half by half.
- **pilot** — the declared architecture's parameter count and a tiny
  throughput probe on every backend. The epoch budget is set from what the
  pilot measures, before any development metric exists.
- **train** — the one declared fusion CNN on the exact common training
  corpus. One architecture, one configuration, no sweep, and no gradient
  anywhere near C1.
- **repeat** — the same configuration and the same seed trained a second
  time, to measure this candidate's run-to-run spread. It writes no
  checkpoint and never becomes the reported candidate.
- **compare** — Agent 1's, Agent 2's and Agent 3's saved checkpoints and the
  unchanged Phase 11 head, loaded read-only and *scored* (never retrained)
  on the same development pieces, so the three-way raw / C1 / hybrid
  comparison can be a paired game bootstrap rather than three marginal
  intervals.
- **interface** — `predict_marginals` / `sample_worlds` on real development
  positions, every world drawn through the accepted Phase 11 sampler.
- **report** — the leaderboard JSON, the learning curve and the Markdown
  report.

Usage::

    python scripts/run_phase11b_agent04.py --full
    python scripts/run_phase11b_agent04.py --stage pilot
    python scripts/run_phase11b_agent04.py --full --run-pytest
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
from stratego.belief.phase11b import feature_seam as seam  # noqa: E402
from stratego.belief.phase11b import heads as H  # noqa: E402
from stratego.belief.phase11b import metrics as M  # noqa: E402
from stratego.belief.phase11b import storage as S  # noqa: E402
from stratego.belief.phase11b.feature_cnn import (  # noqa: E402
    CANDIDATE_3,
    feature_split_view,
    load_feature_cnn,
)
from stratego.belief.phase11b.hybrid_cnn import (  # noqa: E402
    CANDIDATE_4,
    FUSED_CACHE_VERSION,
    FUSED_SHAPE,
    HYBRID_CNN_VERSION,
    HYBRID_FUSION,
    HybridBeliefModel,
    build_fused_cache,
    build_hybrid_cnn,
    fused_cache_path,
    fused_digest,
    hybrid_split_view,
    inference_cost,
    load_fused_cache,
    load_hybrid_cnn,
    parameter_breakdown,
    verify_fused_cache,
)
from stratego.belief.phase11b.interface import (  # noqa: E402
    BELIEF_INTERFACE_VERSION,
    Phase11BPublicState,
)
from stratego.belief.phase11b.raw_cnn import (  # noqa: E402
    CANDIDATE_2,
    load_raw_cnn,
    parameter_count,
)
from stratego.belief.phase11b.raw_train import (  # noqa: E402
    RAW_TRAINER_VERSION,
    RawTrainConfig,
    predict_probabilities_raw,
    stage_observations,
    throughput_pilot,
    train_raw_cnn,
)
from stratego.belief.phase11b.seeds import (  # noqa: E402
    CANONICAL_PHASE11B_SEEDS,
    PHASE11B_IDENTITY_VERSION,
    training_seed,
)
from stratego.belief.phase11b.train import predict_probabilities  # noqa: E402

AGENT = 4
REPORT_DIRECTORY = REPOSITORY_ROOT / REPORT_ROOT
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / CHECKPOINT_ROOT
CORPUS_DIRECTORY = REPOSITORY_ROOT / CORPUS_ROOT
SUMMARY_PATH = REPORT_DIRECTORY / "agent_04_summary.json"
REPORT_PATH = REPORT_DIRECTORY / "agent_04_report.md"
CURVE_PATH = REPORT_DIRECTORY / "agent_04_learning_curve.json"
STAGE_PATH = REPORT_DIRECTORY / ".agent_04_stages.json"
AGENT1_SUMMARY_PATH = REPORT_DIRECTORY / "agent_01_summary.json"
AGENT2_SUMMARY_PATH = REPORT_DIRECTORY / "agent_02_summary.json"
AGENT3_SUMMARY_PATH = REPORT_DIRECTORY / "agent_03_summary.json"

STAGES = (
    "verify",
    "cache",
    "pilot",
    "train",
    "repeat",
    "compare",
    "interface",
    "report",
)

#: The Phase 11 evidence Phase 11B must preserve byte-for-byte. Digested
#: here and compared against what Agent 3 recorded, so "Agent 4 changed
#: nothing" is a measurement rather than an assurance.
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

#: Agent 2's artifacts. Agent 4 reuses Agent 2's residual tower, its
#: read-out and its trainer by **import**, so these files are load-bearing
#: for this experiment and are digested for the same reason Agent 1's are.
PRESERVED_AGENT2_ARTIFACTS = (
    "reports/phase11b/agent_02_summary.json",
    "reports/phase11b/agent_02_report.md",
    "reports/phase11b/agent_02_learning_curve.json",
    "stratego/belief/phase11b/raw_cnn.py",
    "stratego/belief/phase11b/raw_train.py",
)

#: Agent 3's artifacts. `04_AGENT_4` requires Agent 3's "exact frozen C1
#: feature seam" and permits reusing its cache "if compatible and exact",
#: so the seam module is digested and the cache files' contents are checked
#: against the digests Agent 3 published.
PRESERVED_AGENT3_ARTIFACTS = (
    "reports/phase11b/agent_03_summary.json",
    "reports/phase11b/agent_03_report.md",
    "reports/phase11b/agent_03_learning_curve.json",
    "stratego/belief/phase11b/feature_seam.py",
    "stratego/belief/phase11b/feature_cnn.py",
)

#: Earlier candidates' checkpoints, opened read-only for the paired
#: comparison. They are scored, never retrained and never rewritten.
EARLIER_CHECKPOINTS = {
    H.CANDIDATE_1B: "agent01_1b_attached_mlp_head.pt",
    H.CANDIDATE_1C: "agent01_1c_final_block_plus_mlp.pt",
    CANDIDATE_2: "agent02_raw_observation_cnn_run1_declared.pt",
    CANDIDATE_3: "agent03_c1_feature_cnn.pt",
}

REFERENCE_CANDIDATE = "phase11_head_unchanged_reference"

#: The declared training horizon, in epochs, and the wall-clock ceiling.
#: The horizon comes from the pilot's *measured throughput* — see
#: `budget_epochs` — never from a development metric. The constants are
#: Agent 2's and Agent 3's, unchanged, so the three runs are budgeted by the
#: same rule.
TARGET_TRAIN_MINUTES = 20.0
MAX_EPOCHS = 60
MIN_EPOCHS = 12
TRAIN_BUDGET_SECONDS = 2400.0

#: The **one** declared configuration. It is Agent 2's `run1_declared` and
#: Agent 3's, verbatim — Agent 1's optimizer family, no dropout — because
#: the whole point of Agent 4 is to change the input and nothing else.
#: `04_AGENT_4` forbids branch-width, fusion-method, depth and
#: learning-rate sweeps; none was run, and no second configuration was
#: declared or considered.
DECLARED_RUN = {
    "run_id": "run1_declared",
    "description": (
        "Agent 2's declared optimization configuration, inherited unchanged through "
        "Agent 3 so the three candidates differ in input representation rather than "
        "in tuning"
    ),
    "overrides": {
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "block_dropout": 0.0,
        "readout_dropout": 0.0,
        "patience": 5,
    },
}

#: The fused training input is 2.7 GB. Staging it on the training device
#: costs one copy and saves a host-to-device transfer per step; above this
#: ceiling the harness keeps it in host memory instead. A *declared* rule,
#: applied before the run, and the summary records which branch was taken.
STAGE_ON_DEVICE_CEILING_BYTES = 4 * 1024**3


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

    Covers the whole harness — the materialized 2.7 GB fused tensor, the
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


def frozen_c1(device: str = "cpu"):
    return feat.load_frozen_c1(
        REPOSITORY_ROOT, CHECKPOINT_DIRECTORY / "phase9_c1_readonly_copy.pt", device=device
    )


def field_path(split: str) -> Path:
    return seam.field_cache_path(CHECKPOINT_DIRECTORY, split)


CACHE_DEVICE = "cpu"


def fused_path(split: str) -> Path:
    return fused_cache_path(CHECKPOINT_DIRECTORY, split)


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
    agent2 = json.loads(AGENT2_SUMMARY_PATH.read_text())
    agent3 = json.loads(AGENT3_SUMMARY_PATH.read_text())
    for name, block in (("Agent 1", agent1), ("Agent 2", agent2), ("Agent 3", agent3)):
        recorded = block["common_corpus"]["corpus_digest"]
        if identity != recorded:
            problems.append(f"corpus digest {identity} != the digest {name} recorded")
    for split in CORPUS_SPLITS:
        if agent1["common_corpus"]["file_digests"][split] != recomputed[split]:
            problems.append(f"{split} file digests differ from Agent 1's record")

    preserved = {}
    for relative in (
        PRESERVED_PHASE11_ARTIFACTS
        + PRESERVED_ACCEPTED_MODULES
        + PRESERVED_AGENT1_ARTIFACTS
        + PRESERVED_AGENT2_ARTIFACTS
        + PRESERVED_AGENT3_ARTIFACTS
    ):
        path = REPOSITORY_ROOT / relative
        if not path.exists():
            problems.append(f"preserved artifact {relative} is missing")
            continue
        preserved[relative] = file_sha256(path)

    recorded_preserved = agent3["preserved_artifact_digests"]
    changed_since_agent3 = sorted(
        name
        for name, digest in recorded_preserved.items()
        if name in preserved and preserved[name] != digest
    )
    if changed_since_agent3:
        problems.append(f"artifacts changed since Agent 3: {changed_since_agent3}")

    return {
        "stage": "verify",
        "problems": problems,
        "pass": not problems,
        "corpus": {
            "corpus_version": manifest["corpus_version"],
            "corpus_format_version": manifest["corpus_format_version"],
            "corpus_digest": identity,
            "corpus_digest_matches_manifest": identity == manifest["corpus_digest"],
            "corpus_digest_matches_agent1": identity
            == agent1["common_corpus"]["corpus_digest"],
            "corpus_digest_matches_agent2": identity
            == agent2["common_corpus"]["corpus_digest"],
            "corpus_digest_matches_agent3": identity
            == agent3["common_corpus"]["corpus_digest"],
            "file_digests": recomputed,
            "file_digest_drift": drift,
            "reused": "byte-for-byte; Agent 4 regenerated nothing",
            "splits": {
                split: {
                    field: manifest["splits"][split][field]
                    for field in ("games", "samples", "hidden_pieces", "library_split")
                }
                for split in CORPUS_SPLITS
            },
        },
        "preserved_digests": preserved,
        "artifacts_unchanged_since_agent3": not changed_since_agent3,
        "phase11_test_bank_opened": False,
        **PHASE11B_STATUS_MARKERS,
    }


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def stage_cache(*, batch_size: int, rebuild: bool) -> dict:
    """Agent 3's C1 field, reused and checked, plus the fused input.

    `04_AGENT_4` says to "reuse Agent 3's exact frozen C1 feature seam", to
    "not change it", and to "reuse Agent 3's C1 feature cache if compatible
    and exact". So the field cache is not rebuilt: it is opened read-only,
    its content digest is recomputed and compared to the digest Agent 3
    published, and a random sample is re-encoded from the public
    observations through the frozen C1 to confirm it is still a function of
    those two things.

    The fused input Agent 4 does build is not a third representation. It is
    the corpus's own observation bytes and Agent 3's field, laid side by
    side so Agent 2's single-tensor trainer can be reused unforked, and the
    verification compares each half separately to its source.
    """
    agent3 = json.loads(AGENT3_SUMMARY_PATH.read_text())
    recorded_seam = agent3["frozen_seam"]
    recorded_caches = agent3["feature_cache"]["caches"]
    problems: list[str] = []

    if recorded_seam["seam_id"] != seam.SEAM_ID:
        problems.append(f"seam id {seam.SEAM_ID} != Agent 3's {recorded_seam['seam_id']}")
    if recorded_seam["layer_token"] != feat.LAYER_FINAL:
        problems.append("Agent 3's seam layer token is not LAYER_FINAL")
    if recorded_seam != seam.SEAM_DESCRIPTION:
        problems.append("the seam description differs from the one Agent 3 recorded")

    model, identity = frozen_c1(CACHE_DEVICE)
    for key, value in agent3["frozen_model"].items():
        if key in identity and identity[key] != value:
            problems.append(f"frozen C1 {key} differs from Agent 3's record")

    field_caches = {}
    field_verification = {}
    fused_caches = {}
    fused_verification = {}
    started = time.perf_counter()
    for split in CORPUS_SPLITS:
        data = load_corpus(split)
        samples = int(data["samples"])

        path = field_path(split)
        if not path.exists():
            problems.append(f"Agent 3's field cache {path.name} is missing")
            continue
        field = seam.load_field_cache(path, expected_samples=samples)
        observed = seam.field_digest(field)
        expected = recorded_caches[split]["digest"]
        if observed != expected:
            problems.append(
                f"{split} field cache digest {observed} != Agent 3's {expected}"
            )
        field_caches[split] = {
            **{
                key: recorded_caches[split][key]
                for key in ("cache_version", "seam_id", "layer_token", "shape", "dtype")
            },
            "split": split,
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "bytes": int(path.stat().st_size),
            "digest": observed,
            "digest_matches_agent3": observed == expected,
            "built_by": "agent_03",
            "rebuilt_by_agent4": False,
            "derived_from": "public observations + accepted frozen C1",
            "contains_labels": False,
        }
        field_verification[split] = seam.verify_field_cache(model, data, field, rows=64)
        log(
            f"  [field/{split}] {field.shape}  digest matches Agent 3: "
            f"{observed == expected}  re-encoded "
            f"{field_verification[split]['rows_checked']} rows, max |diff| "
            f"{field_verification[split]['max_absolute_difference']:.2e}"
        )

        target = fused_path(split)
        if target.exists() and not rebuild:
            cache = load_fused_cache(target, expected_samples=samples)
            fused_caches[split] = {
                "cache_version": FUSED_CACHE_VERSION,
                "seam_id": seam.SEAM_ID,
                "layer_token": feat.LAYER_FINAL,
                "split": split,
                "path": str(target),
                "shape": list(cache.shape),
                "dtype": "float32",
                "channel_layout": {
                    "public_observation": [0, 127],
                    "frozen_c1_field": [127, 255],
                },
                "bytes": int(target.stat().st_size),
                "digest": fused_digest(cache),
                "seconds": 0.0,
                "reused_existing_file": True,
                "derived_from": [
                    "the common corpus public observations",
                    "Agent 3's frozen C1 field cache",
                ],
                "contains_labels": False,
            }
        else:

            def progress(done, total, elapsed, _split=split):
                if done % (batch_size * 10) == 0 or done == total:
                    log(f"  [fuse/{_split}] {done}/{total} positions in {elapsed:.1f}s")

            fused_caches[split] = build_fused_cache(
                data, field, target, batch_size=batch_size, progress=progress
            )
            fused_caches[split]["reused_existing_file"] = False
            cache = load_fused_cache(target, expected_samples=samples)
        fused_caches[split]["path"] = str(target.relative_to(REPOSITORY_ROOT))
        fused_verification[split] = verify_fused_cache(data, field, cache, rows=64)
        log(
            f"  [fuse/{split}] {cache.shape}  "
            f"{fused_caches[split]['bytes'] / 1e9:.2f} GB  re-fused "
            f"{fused_verification[split]['rows_checked']} rows, raw half exact: "
            f"{fused_verification[split]['raw_half_is_the_corpus_observation']}, "
            f"C1 half exact: {fused_verification[split]['c1_half_is_agent3s_field']}"
        )
        del cache, field

    problems += [
        f"{split}: Agent 3's field cache is not bit-identical on re-derivation "
        f"(max |diff| {block['max_absolute_difference']:.3e})"
        for split, block in field_verification.items()
        if not block["bit_identical"]
    ]
    problems += [
        f"{split}: the fused input is not exactly its two sources side by side "
        f"(max |diff| {block['max_absolute_difference']:.3e})"
        for split, block in fused_verification.items()
        if not block["bit_identical"]
    ]
    return {
        "stage": "cache",
        "seam": seam.SEAM_DESCRIPTION,
        "seam_source": "agent_03; reused unchanged",
        "seam_matches_agent3_record": recorded_seam == seam.SEAM_DESCRIPTION,
        "frozen_model": identity,
        "cache_device": CACHE_DEVICE,
        "cache_device_note": (
            "Agent 3's field cache was built on the accepted CPU evaluation "
            "backend and is reused as those exact bytes; the fusion is a "
            "re-layout of two arrays and involves no backend at all"
        ),
        "field_caches": field_caches,
        "field_verification": field_verification,
        "field_cache_rebuilt": False,
        "fused_caches": fused_caches,
        "fused_verification": fused_verification,
        "total_seconds": round(time.perf_counter() - started, 3),
        "gradients_reaching_c1": False,
        "problems": problems,
        "pass": not problems,
    }


# ---------------------------------------------------------------------------
# pilot
# ---------------------------------------------------------------------------


def budget_epochs(seconds_per_epoch: float) -> int:
    """The declared epoch horizon, from measured throughput alone.

    `04_AGENT_4` caps the whole experiment at roughly one to two hours and
    asks for "a brief throughput/sanity pilot first". The horizon is the
    number of epochs that fits in `TARGET_TRAIN_MINUTES` at the pilot's
    measured rate, clamped to a sane band — a decision made before a single
    development metric exists.
    """
    if seconds_per_epoch <= 0:  # pragma: no cover - defensive
        return MAX_EPOCHS
    fits = int(TARGET_TRAIN_MINUTES * 60.0 / seconds_per_epoch)
    return int(max(MIN_EPOCHS, min(MAX_EPOCHS, fits)))


def stage_pilot(cache: dict, *, batch_positions: int) -> dict:
    """Parameter count, input-path check, and throughput on every backend."""
    train = load_corpus("train")
    fused = load_fused_cache(fused_path("train"), expected_samples=int(train["samples"]))
    # The pilot must measure throughput over the *same* bytes training will
    # read, so the cache stage's recorded digest is re-checked here rather
    # than trusted: a stale or half-written cache would otherwise show up as
    # a plausible-looking epoch estimate.
    recorded = cache["fused_caches"]["train"]["digest"]
    observed = fused_digest(fused)
    if observed != recorded:
        raise AssertionError(f"train fused cache digest {observed} != {recorded}")
    view = hybrid_split_view(train, fused)
    model = build_hybrid_cnn(seed=training_seed(CANDIDATE_4, "init"))
    breakdown = parameter_breakdown(model)
    log(
        f"  parameters {breakdown['total']:,} "
        f"(raw branch {breakdown['raw_branch']:,} + C1 branch "
        f"{breakdown['c1_branch']:,} + tower {breakdown['residual_tower']:,} "
        f"+ read-out {breakdown['readout']:,})"
    )
    if not 3_000_000 <= breakdown["total"] <= 5_000_000:
        raise AssertionError(
            f"parameter count {breakdown['total']} is outside the instructed 3-5M band"
        )
    if any(not tensor.requires_grad for tensor in model.parameters()):
        raise AssertionError("the belief specialist has a frozen parameter")

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    probes = {}
    for device in devices:
        row = throughput_pilot(
            model, view, device=device, batch_positions=batch_positions, steps=6, warmup=2
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
    # Agent 2's rule, unchanged: MPS only when it is stable (finite,
    # CPU-agreeing pilot losses) and materially faster.
    if chosen != "cpu" and speedup < 1.5:
        chosen = "cpu"
    epochs = budget_epochs(probes[chosen]["estimated_epoch_seconds"])
    staged_bytes = int(np.prod(fused.shape)) * 4
    stage_on_device = staged_bytes <= STAGE_ON_DEVICE_CEILING_BYTES
    log(f"  device {chosen} ({speedup:.1f}x CPU); epoch horizon {epochs} from throughput")
    log(
        f"  fused training tensor {staged_bytes / 1e9:.2f} GB; staged on device: "
        f"{stage_on_device}"
    )
    del fused, view

    return {
        "stage": "pilot",
        "architecture": model.architecture,
        "architecture_version": HYBRID_CNN_VERSION,
        "seam_id": seam.SEAM_ID,
        "fusion": HYBRID_FUSION,
        "parameters": breakdown,
        "parameter_band": [3_000_000, 5_000_000],
        "fused_cache_digest_verified": True,
        "agent2_parameters": 3_897_004,
        "agent3_parameters": 3_898_444,
        "parameters_minus_agent2": breakdown["total"] - 3_897_004,
        "parameters_minus_agent3": breakdown["total"] - 3_898_444,
        "probes": probes,
        "device_chosen": chosen,
        "device_speedup_vs_cpu": round(float(speedup), 2),
        "device_rule": (
            "MPS is used only when it is stable (finite, CPU-agreeing pilot losses) "
            "and materially faster (>= 1.5x); otherwise the accepted CPU backend"
        ),
        "cross_device_loss_agreement": (
            round(abs(probes["cpu"]["last_loss"] - probes[chosen]["last_loss"]), 6)
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
        "staged_input_bytes": staged_bytes,
        "stage_on_device": bool(stage_on_device),
        "stage_on_device_rule": (
            f"stage the fused tensor on the training device when it is at most "
            f"{STAGE_ON_DEVICE_CEILING_BYTES / 1024 ** 3:.0f} GiB, otherwise keep it "
            "in host memory and transfer per batch"
        ),
        "model_inputs": ["public_observation_127x10x10", "frozen_c1_field_100x128"],
        "consumes_raw_observation": True,
        "consumes_c1": True,
        "reads_hidden_truth": False,
        "pass": all(row["finite"] for row in probes.values()),
    }


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def overfitting_summary(record: dict) -> dict:
    """How far the training loss ran away from the development loss.

    Agent 2's diagnostic, computed the same way, because the single most
    useful thing Agent 2 handed forward was that a specialist of this size
    memorizes this corpus inside its first epoch — and Agent 3 confirmed it
    on a different representation.
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
    """The one declared configuration of the one Agent 4 architecture."""
    train = load_corpus("train")
    dev = load_corpus("dev")
    train_fused = load_fused_cache(fused_path("train"), expected_samples=int(train["samples"]))
    dev_fused = load_fused_cache(fused_path("dev"), expected_samples=int(dev["samples"]))
    train_view = hybrid_split_view(train, train_fused)
    dev_view = hybrid_split_view(dev, dev_fused)

    device = device or pilot["device_chosen"]
    epochs = int(epochs or pilot["epochs_declared"])
    batch_positions = int(pilot["batch_positions"])
    overrides = DECLARED_RUN["overrides"]

    model = build_hybrid_cnn(
        seed=training_seed(CANDIDATE_4, f"init-{DECLARED_RUN['run_id']}"),
        block_dropout=float(overrides["block_dropout"]),
        readout_dropout=float(overrides["readout_dropout"]),
    )
    config = RawTrainConfig(
        candidate_id=CANDIDATE_4,
        run_id=DECLARED_RUN["run_id"],
        epochs=epochs,
        batch_positions=batch_positions,
        device=device,
        max_seconds=float(budget),
        stage_observations_on_device=bool(pilot["stage_on_device"]),
        **overrides,
    )

    def progress(_id, row):
        log(
            f"  [{DECLARED_RUN['run_id']}] epoch {row['epoch']:>2}  "
            f"train {row['train_loss']:.4f}  dev CE {row['dev_ce']:.4f}  "
            f"R_CE {row['dev_r_ce']:.4f}  top1 {row['dev_top1']:.4f}  {row['seconds']:.0f}s"
        )

    started = time.perf_counter()
    record = train_raw_cnn(model, train_view, dev_view, config, progress=progress)
    state = record.pop("best_state")

    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIRECTORY / f"{CANDIDATE_4}.pt"
    torch.save(
        {
            "candidate_id": CANDIDATE_4,
            "run_id": DECLARED_RUN["run_id"],
            "architecture": model.architecture,
            "architecture_version": HYBRID_CNN_VERSION,
            "fusion": HYBRID_FUSION,
            "seam": seam.SEAM_DESCRIPTION,
            "parameters": parameter_count(model),
            "parameter_breakdown": parameter_breakdown(model),
            "state_dict": state,
            "config": record["config"],
            "dev_metrics": record["dev_metrics"],
            "corpus_version": CORPUS_VERSION,
            "corpus_digest": S.read_manifest(CORPUS_DIRECTORY)["corpus_digest"],
            "frozen_c1": {
                "source_checkpoint": feat.ACCEPTED_PHASE9_CHECKPOINT,
                "retrained": False,
            },
            **PHASE11B_STATUS_MARKERS,
            **PHASE11_FACTS,
        },
        path,
    )
    model.to("cpu")

    record["run_id"] = DECLARED_RUN["run_id"]
    record["description"] = DECLARED_RUN["description"]
    record["checkpoint"] = {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "sha256": file_sha256(path),
        "bytes": int(path.stat().st_size),
    }
    record["architecture"] = model.architecture
    record["parameters"] = parameter_count(model)
    record["parameter_breakdown"] = parameter_breakdown(model)
    record["overfitting"] = overfitting_summary(record)

    # Cost of the deployed path. Two readings, both reported: the specialist
    # alone (what a search already running C1 adds) and end to end (what is
    # comparable to Agent 2, which has no C1 stage).
    scored, _payload = load_hybrid_cnn(path)
    encoder, _identity = frozen_c1("cpu")
    record["inference"] = {
        "cpu": inference_cost(
            scored, encoder, dev_fused, dev["observations"], dev, device="cpu"
        )
    }
    if torch.backends.mps.is_available():
        record["inference"]["mps"] = inference_cost(
            scored, encoder, dev_fused, dev["observations"], dev, device="mps"
        )

    return {
        "stage": "train",
        "architectures_trained": 1,
        "configurations_declared": 1,
        "seconds": round(time.perf_counter() - started, 3),
        "peak_memory_bytes": peak_rss_bytes(),
        "c1_parameters_updated": 0,
        **{
            key: record[key]
            for key in (
                "run_id", "description", "config", "curve", "epochs_run", "best_epoch",
                "best_step", "best_epoch_fraction", "steps_per_epoch", "evaluations",
                "stopped_because", "time_to_best_seconds", "training_seconds",
                "train_positions", "train_pieces", "dev_metrics", "checkpoint",
                "architecture", "parameters", "parameter_breakdown", "inference",
                "observations_staged_on_device", "overfitting",
            )
        },
        "pass": bool(np.isfinite(record["dev_metrics"]["ce"])),
    }


# ---------------------------------------------------------------------------
# repeat
# ---------------------------------------------------------------------------


def stage_repeat(pilot: dict, train_stage: dict, *, device: "str | None", budget: float) -> dict:
    """Train the identical configuration a second time, and compare.

    Agent 3's diagnostic, for Agent 3's reason: a reader of a one-run
    experiment cannot tell a result from a lucky draw. Same architecture,
    same declared configuration, same seed, same corpus, same probe
    schedule, same device.

    This run is a **diagnostic**. It writes no checkpoint, it cannot become
    the reported candidate, and the leaderboard is identical without it.
    Because the seed is the same, a difference here is backend
    nondeterminism and nothing else — which is worth knowing separately for
    this candidate, since it is the first Phase 11B model whose forward pass
    contains a concatenation of two independently normalized branches.
    """
    train = load_corpus("train")
    dev = load_corpus("dev")
    train_fused = load_fused_cache(fused_path("train"), expected_samples=int(train["samples"]))
    dev_fused = load_fused_cache(fused_path("dev"), expected_samples=int(dev["samples"]))
    device = device or pilot["device_chosen"]
    overrides = DECLARED_RUN["overrides"]

    model = build_hybrid_cnn(
        seed=training_seed(CANDIDATE_4, f"init-{DECLARED_RUN['run_id']}"),
        block_dropout=float(overrides["block_dropout"]),
        readout_dropout=float(overrides["readout_dropout"]),
    )
    config = RawTrainConfig(
        candidate_id=CANDIDATE_4,
        run_id=DECLARED_RUN["run_id"],
        epochs=int(train_stage["config"]["epochs"]),
        batch_positions=int(train_stage["config"]["batch_positions"]),
        device=device,
        max_seconds=float(budget),
        stage_observations_on_device=bool(train_stage["observations_staged_on_device"]),
        **overrides,
    )
    started = time.perf_counter()
    record = train_raw_cnn(
        model, hybrid_split_view(train, train_fused), hybrid_split_view(dev, dev_fused), config
    )
    record.pop("best_state")
    model.to("cpu")

    reported = train_stage["dev_metrics"]
    repeated = record["dev_metrics"]
    epochs_first = [
        row["train_loss"] for row in train_stage["curve"] if not row["sub_epoch"]
    ]
    epochs_again = [row["train_loss"] for row in record["curve"] if not row["sub_epoch"]]
    shared = min(len(epochs_first), len(epochs_again))
    epoch_gap = (
        max(abs(a - b) for a, b in zip(epochs_first[:shared], epochs_again[:shared]))
        if shared
        else None
    )
    difference = abs(reported["r_ce"] - repeated["r_ce"])
    log(
        f"  [repeat] R_CE {repeated['r_ce']:.6f} against the reported "
        f"{reported['r_ce']:.6f}  |diff| {difference:.2e}"
    )
    return {
        "stage": "repeat",
        "purpose": "run-to-run spread under the identical declared configuration",
        "is_the_reported_candidate": False,
        "checkpoint_written": False,
        "identical_seed": True,
        "identical_config": {
            key: train_stage["config"][key] == config.to_dict()[key]
            for key in ("learning_rate", "weight_decay", "batch_positions", "epochs", "device")
        },
        "reported_r_ce": reported["r_ce"],
        "repeated_r_ce": repeated["r_ce"],
        "absolute_r_ce_difference": difference,
        "reported_top1": reported["top1"],
        "repeated_top1": repeated["top1"],
        "max_epoch_train_loss_difference": epoch_gap,
        "epochs_run": record["epochs_run"],
        "best_step": record["best_step"],
        "best_step_matches": record["best_step"] == train_stage["best_step"],
        "stopped_because": record["stopped_because"],
        "training_seconds": record["training_seconds"],
        "seconds": round(time.perf_counter() - started, 3),
        "pass": bool(np.isfinite(repeated["ce"])),
    }


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def score_earlier_candidates() -> dict:
    """Earlier candidates, loaded read-only and scored on the same pieces.

    `04_AGENT_4` says "use earlier reports on disk, do not rerun prior
    candidates", and none is rerun: nothing is retrained, no earlier
    artifact is written, and every quoted headline number in the report's
    comparison table comes from the earlier summary JSON. What happens here
    is a forward pass of already-trained weights over the same development
    pieces, which is the only way to get the *paired* bootstrap that can say
    whether the hybrid's difference from raw-only and C1-only is real. Each
    recomputed `R_CE` is checked against what its own agent reported.
    """
    dev = load_corpus("dev")
    agent1 = json.loads(AGENT1_SUMMARY_PATH.read_text())
    agent2 = json.loads(AGENT2_SUMMARY_PATH.read_text())
    agent3 = json.loads(AGENT3_SUMMARY_PATH.read_text())
    reported = {
        **{name: block.get("r_ce") for name, block in agent1["leaderboard"].items()},
        **{name: block.get("r_ce") for name, block in agent2["leaderboard"].items()},
        **{name: block.get("r_ce") for name, block in agent3["leaderboard"].items()},
    }
    encoder, identity = frozen_c1("cpu")
    dev_final = np.asarray(
        np.load(CHECKPOINT_DIRECTORY / "c1_features_dev.npy", mmap_mode="r")
    )

    probabilities: dict[str, np.ndarray] = {}
    probabilities[REFERENCE_CANDIDATE] = predict_probabilities(
        H.ExistingBeliefHead.from_accepted(encoder), dev_final, device="cpu"
    )
    for candidate_id, filename in EARLIER_CHECKPOINTS.items():
        path = CHECKPOINT_DIRECTORY / filename
        if not path.exists():
            log(f"  [compare] {candidate_id}: checkpoint absent, skipped")
            continue
        if candidate_id == CANDIDATE_2:
            raw, _payload = load_raw_cnn(path)
            observations = stage_observations(dev, "cpu", on_device=False)
            probabilities[candidate_id] = predict_probabilities_raw(
                raw, observations, dev, device="cpu"
            )
            del observations
            continue
        if candidate_id == CANDIDATE_3:
            specialist, _payload = load_feature_cnn(path)
            dev_field = seam.load_field_cache(
                field_path("dev"), expected_samples=int(dev["samples"])
            )
            staged = stage_observations(
                feature_split_view(dev, dev_field), "cpu", on_device=False
            )
            probabilities[candidate_id] = predict_probabilities_raw(
                specialist, staged, dev, device="cpu"
            )
            del staged, dev_field
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        head = H.build_candidate(candidate_id, encoder)
        head.load_state_dict(payload["state_dict"])
        head.eval()
        if candidate_id == H.CANDIDATE_1C:
            from stratego.belief.phase11b.train import predict_probabilities_1c

            tokens = np.asarray(
                np.load(CHECKPOINT_DIRECTORY / "c1_penultimate_dev.npy", mmap_mode="r")
            )
            probabilities[candidate_id] = predict_probabilities_1c(
                head, tokens, dev, device="cpu"
            )
            del tokens
        else:
            probabilities[candidate_id] = predict_probabilities(head, dev_final, device="cpu")

    rows: dict[str, dict] = {}
    for candidate_id, values in probabilities.items():
        metrics = M.evaluate(values, dev, bootstrap_resamples=200)
        quoted = reported.get(candidate_id)
        rows[candidate_id] = {
            "r_ce_recomputed": metrics["r_ce"],
            "r_ce_reported_by_its_agent": quoted,
            "absolute_difference": (
                abs(metrics["r_ce"] - quoted) if quoted is not None else None
            ),
            "ce": metrics["ce"],
            "top1": metrics["top1"],
            "source": "loaded read-only and scored; not retrained",
        }
        suffix = f" (reported {quoted:.4f})" if quoted is not None else ""
        log(f"  [compare/{candidate_id}] R_CE {metrics['r_ce']:.4f}{suffix}")
    return {
        "probabilities": probabilities,
        "reproduction": rows,
        "reproduction_note": (
            "the unchanged Phase 11 head reproduces exactly; the trained "
            "checkpoints reproduce to the scale of the run-to-run drift their own "
            "agents measured, orders of magnitude below the gaps this leaderboard "
            "turns on. The recomputed values are used for the paired bootstraps so "
            "every candidate in a comparison comes from one scoring pass; the "
            "quoted leaderboard rows are left exactly as Agents 1, 2 and 3 "
            "reported them."
        ),
        "frozen_model": identity,
        "dev": dev,
    }


def stage_compare(train_record: dict) -> dict:
    """Paired game bootstraps of Agent 4 against every earlier candidate."""
    scored = score_earlier_candidates()
    dev = scored["dev"]
    dev_fused = load_fused_cache(fused_path("dev"), expected_samples=int(dev["samples"]))

    model, _payload = load_hybrid_cnn(REPOSITORY_ROOT / train_record["checkpoint"]["path"])
    staged = stage_observations(hybrid_split_view(dev, dev_fused), "cpu", on_device=False)
    agent4 = predict_probabilities_raw(model, staged, dev, device="cpu")
    metrics = M.evaluate(agent4, dev)

    trained_metrics = train_record["dev_metrics"]
    backend_agreement = {
        "training_backend": train_record["config"]["device"],
        "scoring_backend": "cpu",
        "r_ce_training_backend": trained_metrics["r_ce"],
        "r_ce_cpu": metrics["r_ce"],
        "absolute_difference": abs(trained_metrics["r_ce"] - metrics["r_ce"]),
    }
    log(
        f"  [compare] {CANDIDATE_4} R_CE {metrics['r_ce']:.4f} on CPU "
        f"(training backend reported {trained_metrics['r_ce']:.4f})"
    )

    comparisons = {}
    for candidate_id, values in scored["probabilities"].items():
        comparisons[f"{CANDIDATE_4} vs {candidate_id}"] = M.paired_comparison(
            agent4, values, dev
        )
    uniform = M.uniform_reference(dev)
    del staged
    return {
        "stage": "compare",
        "cpu_metrics": metrics,
        "backend_agreement": backend_agreement,
        "earlier_reproduction": scored["reproduction"],
        "earlier_reproduction_note": scored["reproduction_note"],
        "paired_comparisons": comparisons,
        "uniform_floor": {
            "ce": uniform["ce"],
            "r_ce": uniform["r_ce"],
            "top1": uniform["top1"],
        },
        "note": (
            "Agents 1, 2 and 3's candidates were loaded read-only and scored on the "
            "same development pieces. Nothing was retrained and no earlier artifact "
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
    plans, exactly as Agents 1, 2 and 3 build them, so the four blocks
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

    model, _payload = load_hybrid_cnn(REPOSITORY_ROOT / train_record["checkpoint"]["path"])
    encoder, _identity = frozen_c1("cpu")
    belief = HybridBeliefModel(encoder, model, device="cpu")

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
                raise AssertionError(f"{CANDIDATE_4}: a marginal is not a probability vector")
        drawn = belief.sample_worlds(public, worlds, seed=ordinal)
        repeat = belief.sample_worlds(public, worlds, seed=ordinal)
        if [world["assignment"] for world in drawn] != [
            world["assignment"] for world in repeat
        ]:
            raise AssertionError(f"{CANDIDATE_4}: sample_worlds is not seed-deterministic")
        for world in drawn:
            if sorted(world["assignment"]) != sorted(marginals):
                raise AssertionError(f"{CANDIDATE_4}: a world missed an unresolved piece")
        sampled += len(drawn)
        hidden_counts.append(len(marginals))

    log(f"  [interface] {len(states)} positions, {sampled} worlds, all valid")
    return {
        "stage": "interface",
        "interface_version": BELIEF_INTERFACE_VERSION,
        "candidate_id": CANDIDATE_4,
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

#: The sprint's equivalence band. `00_PHASE_11B_OVERVIEW.md`: "if two
#: candidates are within roughly 0.005 R_CE, prefer the cheaper and simpler".
EQUIVALENCE_BAND = 0.005


def leaderboard_row(train_record: dict, compare: dict) -> dict:
    """The standardized Phase 11B leaderboard fields, for Agent 4."""
    metrics = compare["cpu_metrics"]
    strata = metrics["strata"]
    inference = train_record["inference"]["cpu"]
    return {
        "candidate_id": CANDIDATE_4,
        "phase11b_version": PHASE11B_VERSION,
        "corpus_version": CORPUS_VERSION,
        "architecture": train_record["architecture"],
        "architecture_version": HYBRID_CNN_VERSION,
        "seam_id": seam.SEAM_ID,
        "fusion": HYBRID_FUSION,
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
        "consumes_c1": True,
        "consumes_raw_observation": True,
        "retrains_accepted_c1_weights": False,
        "training_seconds": train_record["training_seconds"],
        "time_to_best_seconds": train_record["time_to_best_seconds"],
        "best_epoch": train_record["best_epoch"],
        "epochs_run": train_record["epochs_run"],
        "stopped_because": train_record["stopped_because"],
        "training_device": train_record["config"]["device"],
        "milliseconds_per_decision_single": inference["specialist"][
            "milliseconds_per_decision_single"
        ],
        "milliseconds_per_decision_single_end_to_end": inference["end_to_end"][
            "milliseconds_per_decision_single"
        ],
        "milliseconds_per_decision_batched": inference["specialist"][
            "milliseconds_per_decision_batched"
        ],
        "inference_microseconds_per_piece": inference["specialist"][
            "microseconds_per_piece_batched"
        ],
        "checkpoint_sha256": train_record["checkpoint"]["sha256"],
        "trained_in_phase11b": True,
    }


def earlier_rows() -> dict:
    """Agents 1, 2 and 3's reported rows, quoted rather than recomputed."""
    agent1 = json.loads(AGENT1_SUMMARY_PATH.read_text())
    agent2 = json.loads(AGENT2_SUMMARY_PATH.read_text())
    agent3 = json.loads(AGENT3_SUMMARY_PATH.read_text())
    combined = {**agent1["leaderboard"], **agent2["leaderboard"], **agent3["leaderboard"]}
    wanted = (
        H.CANDIDATE_1C,
        H.CANDIDATE_1B,
        H.CANDIDATE_1A,
        CANDIDATE_2,
        CANDIDATE_3,
        REFERENCE_CANDIDATE,
    )
    return {name: combined[name] for name in wanted if name in combined}


def comparison_table(row: dict, earlier: dict) -> dict:
    """The no-rerun table `04_AGENT_4` asks for, in the order it asks for it.

    ```text
    old Phase 11 head
    Agent 1 best attached head
    Agent 2 raw CNN
    Agent 3 C1-feature CNN
    Agent 4 hybrid
    ```

    "Agent 1 best attached head" is resolved from Agent 1's own leaderboard
    by lowest `R_CE` rather than hard-coded, so the table cannot silently
    quote the wrong one. Every earlier number is read from its agent's
    summary JSON; nothing here is recomputed and nothing is rerun.
    """
    agent1_candidates = {
        name: block
        for name, block in earlier.items()
        if name.startswith("agent01_")
    }
    best_agent1 = min(agent1_candidates, key=lambda name: agent1_candidates[name]["r_ce"])

    def entry(label: str, candidate_id: str, block: dict) -> dict:
        return {
            "label": label,
            "candidate_id": candidate_id,
            "architecture": block.get("architecture"),
            "r_ce": block.get("r_ce"),
            "r_ce_ci95": block.get("r_ce_ci95"),
            "ce": block.get("ce"),
            "top1": block.get("top1"),
            "scout_rush_r_ce": (block.get("r_ce_by_stratum") or {}).get("scout_rush"),
            "parameters": block.get("belief_parameters_total", block.get("parameters")),
            "training_seconds": block.get("training_seconds"),
            "inference_microseconds_per_piece": block.get(
                "inference_microseconds_per_piece"
            ),
            "rerun_by_agent4": False,
            "quoted_from": (
                "reports/phase11b/agent_04_summary.json (this report)"
                if candidate_id == CANDIDATE_4
                else "its own agent's summary JSON"
            ),
        }

    rows = [
        entry("old Phase 11 head", REFERENCE_CANDIDATE, earlier.get(REFERENCE_CANDIDATE, {})),
        entry(
            "Agent 1 best attached head", best_agent1, agent1_candidates.get(best_agent1, {})
        ),
        entry("Agent 2 raw CNN", CANDIDATE_2, earlier.get(CANDIDATE_2, {})),
        entry("Agent 3 C1-feature CNN", CANDIDATE_3, earlier.get(CANDIDATE_3, {})),
        entry("Agent 4 hybrid", CANDIDATE_4, row),
    ]
    return {
        "rows": rows,
        "agent1_best_resolved_as": best_agent1,
        "agent1_best_resolution_rule": "lowest R_CE on Agent 1's own leaderboard",
        "prior_candidates_rerun": False,
        "source": (
            "reports/phase11b/agent_01_summary.json, agent_02_summary.json and "
            "agent_03_summary.json"
        ),
    }


def complementarity(row: dict, earlier: dict, comparisons: dict) -> dict:
    """Agent 4's actual engineering question, answered arithmetically.

    `04_AGENT_4`: "This model asks whether C1 provides useful high-level
    strategic abstractions while raw observation restores belief-specific
    details C1 may have compressed. Compare directly with Agent 2: raw-only
    CNN, Agent 3: C1-only CNN."

    Complementarity is a claim about the hybrid beating *both* single-source
    towers of the same size — beating only the weaker one would show
    nothing, since the hybrid contains the stronger one's input. So the
    reference is the better of the two, the threshold is the sprint's own
    `0.005` band, and the reading is computed from the number rather than
    written by hand.
    """
    raw_only = earlier.get(CANDIDATE_2, {}).get("r_ce")
    c1_only = earlier.get(CANDIDATE_3, {}).get("r_ce")
    hybrid = row["r_ce"]
    pairs = {name: value for name, value in (("raw", raw_only), ("c1", c1_only)) if value}
    best_single_name = min(pairs, key=lambda name: pairs[name]) if pairs else None
    best_single = pairs.get(best_single_name)
    delta_best = hybrid - best_single if best_single is not None else None
    delta_raw = hybrid - raw_only if raw_only is not None else None
    delta_c1 = hybrid - c1_only if c1_only is not None else None

    if delta_best is None:  # pragma: no cover - both reports are on disk
        reading = "an earlier report was not available"
    elif delta_best < -EQUIVALENCE_BAND:
        reading = (
            "complementary: the fusion is materially better than either "
            "single-source tower of the same size, so raw observation and the "
            "frozen C1 representation each carry belief signal the other lacks"
        )
    elif abs(delta_best) <= EQUIVALENCE_BAND:
        reading = (
            "not complementary: the fusion is no better than its stronger branch "
            "alone, so the second input adds no belief signal this corpus can "
            "expose beyond what the stronger one already carries"
        )
    else:
        reading = (
            "the fusion is worse than its stronger branch alone: splitting the "
            "stem between two branches cost more than the second representation "
            "returned"
        )
    return {
        "agent2_raw_only_r_ce": raw_only,
        "agent3_c1_only_r_ce": c1_only,
        "agent4_hybrid_r_ce": hybrid,
        "better_single_source": best_single_name,
        "better_single_source_r_ce": best_single,
        "hybrid_minus_better_single_source": delta_best,
        "hybrid_minus_agent2_raw_only": delta_raw,
        "hybrid_minus_agent3_c1_only": delta_c1,
        "equivalence_band": EQUIVALENCE_BAND,
        "complementary": bool(delta_best is not None and delta_best < -EQUIVALENCE_BAND),
        "paired_comparison_with_agent2": comparisons.get(f"{CANDIDATE_4} vs {CANDIDATE_2}"),
        "paired_comparison_with_agent3": comparisons.get(f"{CANDIDATE_4} vs {CANDIDATE_3}"),
        "capacity_held_fixed": {
            "agent02_raw_observation_cnn": 3_897_004,
            "agent03_c1_feature_cnn": 3_898_444,
            "agent04_hybrid_raw_c1_cnn": row["parameters"],
            "spread": 3_898_444 - 3_897_004,
            "note": (
                "the three candidates share Agent 2's residual tower and read-out "
                "and differ only in the stem, so a difference between them is a "
                "difference of input representation and not of capacity"
            ),
        },
        "interpretation": reading,
        "interpretation_source": "04_AGENT_4_HYBRID_RAW_C1_CNN.md, 'Engineering Question'",
        "agent2_rerun": False,
        "agent3_rerun": False,
    }


def decide(row: dict, earlier: dict, comparisons: dict) -> dict:
    """Apply the sprint's engineering-winner rule to Agent 4.

    `00_PHASE_11B_OVERVIEW.md`: prefer materially lower overall `R_CE`,
    weight Scout-rush generalization, treat candidates within roughly
    `0.005 R_CE` as equivalent and prefer the cheaper and simpler one, and
    count search-integration complexity.
    """
    everyone = {
        **{name: block["r_ce"] for name, block in earlier.items()},
        CANDIDATE_4: row["r_ce"],
    }
    leader_id = min(everyone, key=lambda name: everyone[name])
    leader_r_ce = everyone[leader_id]
    within_band = [name for name in everyone if everyone[name] - leader_r_ce <= EQUIVALENCE_BAND]
    best_earlier = min(
        (name for name in earlier if name != REFERENCE_CANDIDATE),
        key=lambda name: earlier[name]["r_ce"],
        default=None,
    )
    delta = row["r_ce"] - earlier[best_earlier]["r_ce"] if best_earlier else None
    return {
        "leader_by_r_ce": leader_id,
        "leader_r_ce": leader_r_ce,
        "equivalence_band": EQUIVALENCE_BAND,
        "within_band_of_leader": within_band,
        "best_earlier_candidate": best_earlier,
        "agent4_minus_best_earlier_r_ce": delta,
        "agent4_materially_better_than_best_earlier": bool(
            delta is not None and delta < -EQUIVALENCE_BAND
        ),
        "agent4_is_the_leader": leader_id == CANDIDATE_4,
        "paired_comparison_with_best_earlier": comparisons.get(
            f"{CANDIDATE_4} vs {best_earlier}"
        ),
        "scout_rush_r_ce": {
            CANDIDATE_4: row["r_ce_by_stratum"].get("scout_rush"),
            **{
                name: block["r_ce_by_stratum"].get("scout_rush")
                for name, block in earlier.items()
            },
        },
        "search_integration_note": (
            "Agent 4 needs both C1's encode and its own tower: a search already "
            "running C1 for its policy pays the specialist's "
            f"{row['milliseconds_per_decision_single']:.2f} ms per position, and a "
            f"belief query in isolation costs "
            f"{row['milliseconds_per_decision_single_end_to_end']:.2f} ms. That is "
            "Agent 3's integration cost plus a second input path, against Agent 1's "
            "head, which is a tensor on a pass the search already makes."
        ),
    }


def stage_report(stages: dict) -> dict:
    verify = stages["verify"]
    cache = stages["cache"]
    pilot = stages["pilot"]
    train = stages["train"]
    compare = stages["compare"]
    interface = stages["interface"]

    row = leaderboard_row(train, compare)
    earlier = earlier_rows()
    table = comparison_table(row, earlier)
    complement = complementarity(row, earlier, compare["paired_comparisons"])
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
            "name": "hybrid raw + C1 fusion CNN belief specialist",
            "instruction": (
                "instructions/phase_11b_belief_engineering_sprint/"
                "04_AGENT_4_HYBRID_RAW_C1_CNN.md"
            ),
            "question": (
                "do raw public information and C1's learned strategic "
                "representation provide complementary belief signal?"
            ),
            "architectures_trained": 1,
            "architecture_sweep": False,
            "branch_width_sweep": False,
            "fusion_method_sweep": False,
            "depth_sweep": False,
            "learning_rate_sweep": False,
            "hyperparameter_sweep": False,
            "optimization_configurations_declared": 1,
            "fusion": HYBRID_FUSION,
            "tower_inherited_from": CANDIDATE_2,
            "seam_inherited_from": CANDIDATE_3,
            "architecture_inheritance_reason": (
                "Agent 2's residual tower and read-out verbatim (width 160, 8 "
                "residual 3x3 blocks, 1x1 read-out 128) and Agent 3's frozen C1 "
                "seam unchanged, so the only difference between the three "
                "candidates is what enters the tower"
            ),
            "branch_width_rationale": (
                "the fused width had to be 160 for the tower to be Agent 2's, and "
                "160 is divided evenly between the two branches because the "
                "experiment asks whether the sources are complementary and an "
                "uneven split would prejudge which one carries more"
            ),
            "trainer_version": RAW_TRAINER_VERSION,
            "trainer_shared_with": [CANDIDATE_2, CANDIDATE_3],
            "prior_candidates_rerun": False,
        },
        "common_corpus": verify["corpus"],
        "preserved_artifact_digests": verify["preserved_digests"],
        "preservation": {
            "artifacts_unchanged_since_agent3": verify["artifacts_unchanged_since_agent3"],
            "phase11_test_bank_opened": False,
            "agent1_artifacts_modified": False,
            "agent2_artifacts_modified": False,
            "agent3_artifacts_modified": False,
            "c1_modified": False,
            "corpus_regenerated": False,
            "agent3_field_cache_rebuilt": False,
        },
        "frozen_seam": cache["seam"],
        "frozen_seam_source": cache["seam_source"],
        "frozen_seam_matches_agent3_record": cache["seam_matches_agent3_record"],
        "frozen_model": cache["frozen_model"],
        "feature_cache": {
            "cache_device": cache["cache_device"],
            "cache_device_note": cache["cache_device_note"],
            "c1_field_reused_from_agent3": cache["field_caches"],
            "c1_field_verification": cache["field_verification"],
            "c1_field_rebuilt": cache["field_cache_rebuilt"],
            "fused_input": cache["fused_caches"],
            "fused_input_verification": cache["fused_verification"],
            "total_seconds": cache["total_seconds"],
            "gradients_reaching_c1": False,
        },
        "pilot": {
            key: pilot[key]
            for key in (
                "architecture",
                "architecture_version",
                "seam_id",
                "fusion",
                "parameters",
                "parameter_band",
                "agent2_parameters",
                "agent3_parameters",
                "parameters_minus_agent2",
                "parameters_minus_agent3",
                "probes",
                "device_chosen",
                "device_speedup_vs_cpu",
                "device_rule",
                "cross_device_loss_agreement",
                "epochs_declared",
                "epoch_budget_basis",
                "batch_positions",
                "staged_input_bytes",
                "stage_on_device",
                "stage_on_device_rule",
                "model_inputs",
                "consumes_raw_observation",
                "consumes_c1",
                "reads_hidden_truth",
                "fused_cache_digest_verified",
            )
        },
        "training": {
            "architectures_trained": train["architectures_trained"],
            "configurations_declared": train["configurations_declared"],
            "run_id": train["run_id"],
            "description": train["description"],
            "config": train["config"],
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
            "input_staged_on_device": train["observations_staged_on_device"],
            "loss": "supervised hidden-rank cross-entropy over hidden pieces only",
            "policy_or_value_terms": False,
            "game_outcome_used": False,
            "c1_parameters_updated": train["c1_parameters_updated"],
            "trainable_parameters": train["parameters"],
        },
        "repeat_run": stages.get("repeat"),
        "checkpoint": train["checkpoint"],
        "inference": train["inference"],
        "leaderboard": {CANDIDATE_4: row},
        "earlier_reference_rows": earlier,
        "earlier_rows_note": (
            "quoted from reports/phase11b/agent_01_summary.json, "
            "agent_02_summary.json and agent_03_summary.json; no earlier "
            "experiment was rerun"
        ),
        "comparison_table": table,
        "complementarity": complement,
        "backend_agreement": compare["backend_agreement"],
        "earlier_reproduction": compare["earlier_reproduction"],
        "earlier_reproduction_note": compare["earlier_reproduction_note"],
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
            "peak process RSS of the training stage: the materialized 2.7 GB fused "
            "input tensor, the model and the metric arrays, not the model alone"
        ),
        "suite": stages.get("suite"),
        "stop_condition": (
            "Agent 4 trained one architecture and stopped. The Transformer "
            "(Agent 5) was not begun and is not authorized by this artifact. "
            "Phase 11 remains FAIL, phase11_test_bank_v1 remains spent and "
            "unopened, and nothing here authorizes Phase 12."
        ),
    }

    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    CURVE_PATH.write_text(
        json.dumps(
            {
                "candidate_id": CANDIDATE_4,
                "phase11b_version": PHASE11B_VERSION,
                "corpus_version": CORPUS_VERSION,
                "run_id": train["run_id"],
                "config": train["config"],
                "curve": train["curve"],
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
    from scripts._phase11b_agent04_report import render

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
    parser = argparse.ArgumentParser(description="Phase 11B Agent 4 harness")
    parser.add_argument("--full", action="store_true", help="run every stage in order")
    parser.add_argument("--stage", action="append", choices=STAGES, help="run one stage")
    parser.add_argument("--device", default=None, help="override the training device")
    parser.add_argument("--epochs", type=int, default=None, help="override the epoch horizon")
    parser.add_argument(
        "--batch-positions", type=int, default=256, help="training batch, in positions"
    )
    parser.add_argument(
        "--cache-batch", type=int, default=512, help="fusion batch, in positions"
    )
    parser.add_argument(
        "--rebuild-cache", action="store_true", help="rebuild the fused input cache"
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
        elif name == "cache":
            payload = stage_cache(
                batch_size=arguments.cache_batch, rebuild=arguments.rebuild_cache
            )
        elif name == "pilot":
            payload = stage_pilot(stages["cache"], batch_positions=arguments.batch_positions)
        elif name == "train":
            payload = stage_train(
                stages["pilot"],
                device=arguments.device,
                epochs=arguments.epochs,
                budget=arguments.budget,
            )
        elif name == "repeat":
            payload = stage_repeat(
                stages["pilot"], stages["train"], device=arguments.device,
                budget=arguments.budget,
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
