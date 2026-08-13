#!/usr/bin/env python3
"""Phase 7 Agent 4 acceptance harness: reflection + procedural perturbation.

Verifies the Agent 1/2/3 prerequisites and the audited library digest, then
runs the deterministic balanced 100,000-output procedural stress corpus
through `setup_sampler_v1` and validates every output from scratch — engine
legality/inventory, initial mobility, split and primary-family inheritance,
family required predicates, serialization/fingerprint and reflection round
trips, deterministic rebuild from provenance, and provenance stability — then
measures effective diversity, recomputes Agent 1's family-defining metrics on
the descendants, and searches the corpus exhaustively for descendant-vs-
descendant duplicates and cross-split leakage. Writes:

    reports/phase_7_data/agent_04_sampler_contract.json
    reports/phase_7_data/agent_04_procedural_stress.json
    reports/phase_7_data/agent_04_procedural_family_metrics.csv

What this script is and is not
------------------------------
It is an acceptance instrument. It does not modify the production library, the
Agent 1 family contracts, the frozen thresholds, the split assignments or the
identity semantics. Overlap and distance values it reports are descriptive
unless a frozen hard gate is violated. No game outcome, win rate, Elo or model
score participates in any decision below.

Descendant-vs-descendant analysis
---------------------------------
Agent 3's triangle-inequality margin (base-base class distance >= 20, frozen
perturbation bound <= 12) protects a descendant relative to *other bases*, not
relative to *other descendants*: two descendants of different bases can in
principle approach each other from both sides at once, and two descendants of
one base are unconstrained by it entirely. This harness therefore measures the
descendant-vs-descendant relation directly and exhaustively rather than
inheriting the argument.

Usage::

    python scripts/run_phase7_agent04.py                  # stress + artifacts
    python scripts/run_phase7_agent04.py --run-pytest     # also run the suite
    python scripts/run_phase7_agent04.py --scale 0.02     # fast smoke run
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import resource
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from stratego.engine.constants import (  # noqa: E402
    IMPLEMENTATION_VERSION,
    NUM_PIECE_TYPES,
    RULES_VERSION,
)
from stratego.engine.setup import SetupError, deserialize_setup, serialize_setup, validate_setup  # noqa: E402
from stratego.setups import (  # noqa: E402
    BASE_SETUP_COUNT,
    DEFAULT_LIBRARY_MASTER_SEED,
    FAMILY_IDS,
    LIBRARY_JSONL_PATH,
    LIBRARY_MANIFEST_PATH,
    SETUP_FAMILY_VERSION,
    SETUP_GENERATOR_CONTRACT_VERSION,
    SETUP_LIBRARY_VERSION,
    SETUP_TRAIT_VECTOR_VERSION,
    SPLITS,
    read_library_jsonl,
    read_manifest,
)
from stratego.setups.contracts import (  # noqa: E402
    PERTURBATION_MAX_HAMMING,
    PERTURBATION_MIN_HAMMING,
    split_for_base_index,
    validate_perturbation,
)
from stratego.setups.diversity import (  # noqa: E402
    DIVERSITY_STANDARD_VERSION,
    DIVERSITY_THRESHOLDS_V1,
    LibraryEntry,
    class_distance,
    entropy_metrics,
    family_overlap_matrix,
    folded_support,
    hamming_distance,
    identity_metrics,
    trait_diversity_metrics,
)
from stratego.setups.families import family_contract  # noqa: E402
from stratego.setups.identity import (  # noqa: E402
    canonical_class_representative,
    class_fingerprint,
    content_fingerprint,
    reflect_canonical,
)
from stratego.setups.library import library_content_digest, manifest_digest  # noqa: E402
from stratego.setups.mobility import setup_has_initial_mobility  # noqa: E402
from stratego.setups.perturbation import (  # noqa: E402
    MAX_PERTURBATION_ATTEMPTS,
    MAX_SWAP_COUNT,
    MIN_SWAP_COUNT,
    OPERATOR_NAMES,
    PERTURBATION_SEED_ENCODING,
    PERTURBATION_VERSION,
    REJECTION_REASONS,
    decode_perturbation_seed,
)
from stratego.setups.sampler import (  # noqa: E402
    DEFAULT_PROFILE,
    PROFILES,
    REQUIRED_PROVENANCE_FIELDS,
    SAMPLER_VERSION,
    STRESS_CORPUS_VERSION,
    STRESS_SPLIT_OUTPUTS,
    build_stress_output,
    load_library_index,
    provenance_is_observer_safe,
    provenance_round_trips,
    rebuild_from_provenance,
    sample_setup,
    sampler_contract_document,
    stress_corpus_plan,
    validate_sampled_setup,
)
from stratego.setups.traits import HIGH_RANK_TYPES, compute_trait_vector  # noqa: E402
from stratego.engine.constants import BOMB, FLAG, MINER, SCOUT  # noqa: E402

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_7_data"
AGENT_01_CONTRACT = DATA_DIRECTORY / "agent_01_setup_contract.json"
AGENT_01_THRESHOLDS = DATA_DIRECTORY / "agent_01_diversity_thresholds.json"
AGENT_02_MANIFEST = DATA_DIRECTORY / "agent_02_base_library_manifest.json"
AGENT_03_AUDIT = DATA_DIRECTORY / "agent_03_library_audit.json"

CONTRACT_ARTIFACT = DATA_DIRECTORY / "agent_04_sampler_contract.json"
STRESS_ARTIFACT = DATA_DIRECTORY / "agent_04_procedural_stress.json"
FAMILY_METRICS_CSV = DATA_DIRECTORY / "agent_04_procedural_family_metrics.csv"

LIBRARY_PATH = REPOSITORY_ROOT / LIBRARY_JSONL_PATH
MANIFEST_PATH = REPOSITORY_ROOT / LIBRARY_MANIFEST_PATH

#: Agent 1's frozen leakage thresholds, applied to descendants where relevant.
CROSS_SPLIT_FLOOR = DIVERSITY_THRESHOLDS_V1.min_cross_split_nn_distance
NEAR_DUPLICATE_DISTANCE = DIVERSITY_THRESHOLDS_V1.within_family_near_duplicate_distance


def _base_library_min_class_distance() -> int:
    """Agent 3's measured minimum class distance between any two bases.

    Read from the audit artifact rather than restated, so the descendant
    comparison below is anchored to what Agent 3 actually measured.
    """
    audit = json.loads(AGENT_03_AUDIT.read_text())
    return int(audit["audit"]["similarity"]["global_min_pairwise_distance"])


# ---------------------------------------------------------------------------
# Environment and prerequisites
# ---------------------------------------------------------------------------


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _environment() -> dict:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "numpy_version": np.__version__,
    }


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage)


def _verify_prerequisites() -> dict:
    """Agents 1-3 PASS plus the exact audited library digest, before any work."""
    problems: list[str] = []
    statuses: dict = {}

    for name, path in (
        ("agent_01_setup_contract", AGENT_01_CONTRACT),
        ("agent_01_diversity_thresholds", AGENT_01_THRESHOLDS),
        ("agent_02_base_library_manifest", AGENT_02_MANIFEST),
        ("agent_03_library_audit", AGENT_03_AUDIT),
    ):
        if not path.exists():
            problems.append(f"missing prerequisite artifact: {path.name}")
            statuses[name] = None
            continue
        payload = json.loads(path.read_text())
        statuses[name] = payload.get("status")
        if payload.get("status") != "PASS":
            problems.append(f"{name} status is {payload.get('status')!r}, not PASS")

    audit = json.loads(AGENT_03_AUDIT.read_text()) if AGENT_03_AUDIT.exists() else {}
    entries = read_library_jsonl(LIBRARY_PATH)
    manifest = read_manifest(MANIFEST_PATH)
    observed = {
        "library_digest": library_content_digest(entries),
        "manifest_digest": manifest_digest(manifest),
        "entry_count": len(entries),
    }
    expected = {
        "library_digest": audit.get("library_digest"),
        "manifest_digest": audit.get("manifest_digest"),
        "entry_count": audit.get("setup_count"),
    }
    for key, value in expected.items():
        if value is not None and observed[key] != value:
            problems.append(
                f"audited {key} changed: expected {value!r}, observed {observed[key]!r}"
            )
    if manifest.get("manifest_digest") != observed["manifest_digest"]:
        problems.append("library manifest digest does not match its own contents")

    return {
        "statuses": statuses,
        "expected_digests": expected,
        "observed_digests": observed,
        "library_unchanged": not problems,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Corpus generation and per-output validation
# ---------------------------------------------------------------------------


class StressCorpus:
    """The materialized stress corpus plus its per-output validation counters."""

    def __init__(self) -> None:
        self.setups: list[tuple[int, ...]] = []
        self.family: list[str] = []
        self.split: list[str] = []
        self.base_id: list[str] = []
        self.base_index: list[int] = []
        self.reflected: list[bool] = []
        self.requested: list[bool] = []
        self.perturbed: list[bool] = []
        self.swap_count: list[int] = []
        self.attempts: list[int] = []
        self.class_fp: list[str] = []
        self.content_fp: list[str] = []
        self.base_hamming: list[int] = []
        self.base_class_distance: list[int] = []
        self.operator_use: Counter = Counter()
        self.rejections: Counter = Counter()
        self.rejections_by_family: dict = defaultdict(Counter)
        self.attempts_by_swap_count: dict = defaultdict(list)
        self.exhausted: list[str] = []

        self.failures: dict = {
            "engine_invalid": [],
            "incorrect_inventory": [],
            "stranded": [],
            "family_violation": [],
            "split_change": [],
            "family_change": [],
            "serialization": [],
            "reflection": [],
            "rebuild": [],
            "provenance": [],
            "perturbation_invariant": [],
            "hamming_window": [],
            "flag_moved": [],
        }

    def __len__(self) -> int:
        return len(self.setups)


def _validate_output(corpus: StressCorpus, sampled, base_entry, draw) -> None:
    """Recompute every hard requirement for one output, independently.

    Deliberately does not trust `build_descendant`'s own acceptance: every
    check below is recomputed here from the finished setup through the frozen
    engine and the frozen contracts.
    """
    tag = f"{draw.base_setup_id}#{draw.split}:{draw.position}"
    setup = sampled.canonical

    try:
        validated = validate_setup(setup, 0)
    except SetupError as error:
        corpus.failures["engine_invalid"].append(f"{tag}: {error}")
        corpus.failures["incorrect_inventory"].append(f"{tag}: {error}")
        return
    if validated != setup:
        corpus.failures["engine_invalid"].append(f"{tag}: normalization mismatch")

    traits = compute_trait_vector(setup)
    satisfied, violations = family_contract(base_entry.family_id).evaluate(traits)
    if not satisfied:
        corpus.failures["family_violation"].append(f"{tag}: {violations}")

    if not setup_has_initial_mobility(setup):
        corpus.failures["stranded"].append(tag)

    if sampled.split != base_entry.split or sampled.split != draw.split:
        corpus.failures["split_change"].append(tag)
    if split_for_base_index(base_entry.base_index) != sampled.split:
        corpus.failures["split_change"].append(f"{tag}: split rule")
    if sampled.family_id != base_entry.family_id or sampled.family_id != draw.family_id:
        corpus.failures["family_change"].append(tag)

    serialized = serialize_setup(setup)
    if deserialize_setup(serialized) != setup:
        corpus.failures["serialization"].append(tag)
    if sampled.provenance["final_setup"] != serialized:
        corpus.failures["serialization"].append(f"{tag}: provenance setup mismatch")
    recomputed_content = content_fingerprint(setup)
    recomputed_class = class_fingerprint(setup)
    if sampled.provenance["final_setup_fingerprint"] != recomputed_content:
        corpus.failures["serialization"].append(f"{tag}: content fingerprint mismatch")
    if sampled.provenance["final_setup_class_fingerprint"] != recomputed_class:
        corpus.failures["serialization"].append(f"{tag}: class fingerprint mismatch")

    mirrored = reflect_canonical(setup)
    if reflect_canonical(mirrored) != setup:
        corpus.failures["reflection"].append(f"{tag}: involution")
    if class_fingerprint(mirrored) != recomputed_class:
        corpus.failures["reflection"].append(f"{tag}: class fingerprint not reflection-invariant")
    try:
        validate_setup(mirrored, 0)
    except SetupError as error:
        corpus.failures["reflection"].append(f"{tag}: mirrored setup invalid: {error}")
    if not setup_has_initial_mobility(mirrored):
        corpus.failures["reflection"].append(f"{tag}: mirrored setup stranded")
    expected = (
        reflect_canonical(sampled.perturbed_canonical)
        if draw.reflection_applied
        else sampled.perturbed_canonical
    )
    if expected != setup:
        corpus.failures["reflection"].append(f"{tag}: orientation not applied as recorded")

    if sampled.perturbation_applied:
        invariant_violations = validate_perturbation(
            base_entry.canonical_setup, sampled.perturbed_canonical, base_entry.family_id
        )
        if invariant_violations:
            corpus.failures["perturbation_invariant"].append(f"{tag}: {invariant_violations}")
        distance = hamming_distance(base_entry.canonical_setup, sampled.perturbed_canonical)
        if not PERTURBATION_MIN_HAMMING <= distance <= PERTURBATION_MAX_HAMMING:
            corpus.failures["hamming_window"].append(f"{tag}: hamming {distance}")
        if base_entry.canonical_setup.index(FLAG) != sampled.perturbed_canonical.index(FLAG):
            corpus.failures["flag_moved"].append(tag)

    missing = [key for key in REQUIRED_PROVENANCE_FIELDS if key not in sampled.provenance]
    if missing:
        corpus.failures["provenance"].append(f"{tag}: missing {missing}")
    if draw.perturbation_requested:
        decoded_count, _raw = decode_perturbation_seed(
            int(sampled.provenance["perturbation_seed"])
        )
        if (
            decoded_count != draw.swap_count
            or sampled.provenance["perturbation_swap_count"] != decoded_count
        ):
            corpus.failures["provenance"].append(
                f"{tag}: swap count does not decode from the recorded seed"
            )
        if sampled.provenance["perturbation_max_attempts"] != MAX_PERTURBATION_ATTEMPTS:
            corpus.failures["provenance"].append(
                f"{tag}: recorded max attempts is not the version constant"
            )
    if not provenance_round_trips(sampled.provenance):
        corpus.failures["provenance"].append(f"{tag}: json round-trip")
    unsafe = provenance_is_observer_safe(sampled.provenance)
    if unsafe:
        corpus.failures["provenance"].append(f"{tag}: outcome/strength fields {unsafe}")
    if (
        sampled.provenance["base_setup_id"] != draw.base_setup_id
        or sampled.provenance["split"] != draw.split
        or sampled.provenance["primary_family_id"] != draw.family_id
        or bool(sampled.provenance["reflection_applied"]) != draw.reflection_applied
    ):
        corpus.failures["provenance"].append(f"{tag}: recorded decisions differ from the plan")


def _rebuild_check(corpus: StressCorpus, sampled, index) -> None:
    """Rebuild one output from provenance alone and require exact equality."""
    tag = sampled.provenance["base_setup_id"]
    try:
        rebuilt = rebuild_from_provenance(json.loads(json.dumps(sampled.provenance)), index=index)
    except Exception as error:  # noqa: BLE001 - any failure is a rebuild failure
        corpus.failures["rebuild"].append(f"{tag}: {error}")
        return
    if rebuilt.canonical != sampled.canonical:
        corpus.failures["rebuild"].append(f"{tag}: setup differs")
    if rebuilt.provenance != sampled.provenance:
        differing = sorted(
            key
            for key in set(rebuilt.provenance) | set(sampled.provenance)
            if rebuilt.provenance.get(key) != sampled.provenance.get(key)
        )
        corpus.failures["rebuild"].append(f"{tag}: provenance differs in {differing}")


def _generate_corpus(scale: float, index, progress_every: int = 10000) -> StressCorpus:
    corpus = StressCorpus()
    split_outputs = {
        split: max(4, int(round(count * scale)))
        for split, count in STRESS_SPLIT_OUTPUTS.items()
    }
    started = time.time()
    for draw in stress_corpus_plan(split_outputs=split_outputs):
        base_entry = index.base(draw.base_setup_id)
        sampled = build_stress_output(draw, index=index)

        _validate_output(corpus, sampled, base_entry, draw)
        _rebuild_check(corpus, sampled, index)

        perturbation = sampled.perturbation
        corpus.setups.append(sampled.canonical)
        corpus.family.append(draw.family_id)
        corpus.split.append(draw.split)
        corpus.base_id.append(draw.base_setup_id)
        corpus.base_index.append(draw.base_index)
        corpus.reflected.append(draw.reflection_applied)
        corpus.requested.append(draw.perturbation_requested)
        corpus.perturbed.append(sampled.perturbation_applied)
        corpus.swap_count.append(draw.swap_count if draw.perturbation_requested else 0)
        corpus.attempts.append(perturbation.attempts if perturbation else 0)
        corpus.class_fp.append(sampled.provenance["final_setup_class_fingerprint"])
        corpus.content_fp.append(sampled.provenance["final_setup_fingerprint"])
        corpus.base_hamming.append(
            hamming_distance(base_entry.canonical_setup, sampled.perturbed_canonical)
        )
        corpus.base_class_distance.append(
            class_distance(base_entry.canonical_setup, sampled.canonical)
        )

        if perturbation is not None:
            corpus.operator_use.update(perturbation.operators_applied)
            corpus.rejections.update(perturbation.rejections)
            corpus.rejections_by_family[draw.family_id].update(perturbation.rejections)
            corpus.attempts_by_swap_count[draw.swap_count].append(perturbation.attempts)
            if not perturbation.accepted:
                corpus.exhausted.append(f"{draw.base_setup_id}:{draw.position}")

        if progress_every and len(corpus) % progress_every == 0:
            elapsed = time.time() - started
            print(
                f"  {len(corpus):,} outputs  ({elapsed:.0f}s, "
                f"{len(corpus) / max(elapsed, 1e-9):.0f}/s)",
                flush=True,
            )
    return corpus


# ---------------------------------------------------------------------------
# Equivalence-class analysis (descendant-vs-descendant, cross-split)
# ---------------------------------------------------------------------------


def _class_analysis(corpus: StressCorpus) -> dict:
    """Group outputs into reflection-equivalence classes and attribute them.

    This is the exact/reflection-equivalence instrument. Two outputs share a
    class exactly when they are equal or mirror images under Agent 1's frozen
    class fingerprint, so `classes_with_multiple_bases` is the
    descendant-vs-descendant duplicate count and `classes_with_multiple_splits`
    is the cross-split leakage count.
    """
    by_class: dict[str, list[int]] = defaultdict(list)
    for position, fingerprint in enumerate(corpus.class_fp):
        by_class[fingerprint].append(position)

    exact_groups: dict[tuple, list[int]] = defaultdict(list)
    for position, setup in enumerate(corpus.setups):
        exact_groups[setup].append(position)

    multi_base: list[dict] = []
    multi_split: list[dict] = []
    multi_family: list[dict] = []
    repeated_classes = 0

    for fingerprint, members in by_class.items():
        if len(members) > 1:
            repeated_classes += 1
        bases = {corpus.base_id[position] for position in members}
        splits = {corpus.split[position] for position in members}
        families = {corpus.family[position] for position in members}
        if len(bases) > 1:
            multi_base.append(
                {
                    "class_fingerprint": fingerprint,
                    "base_setup_ids": sorted(bases),
                    "splits": sorted(splits),
                    "families": sorted(families),
                    "output_count": len(members),
                }
            )
        if len(splits) > 1:
            multi_split.append(
                {
                    "class_fingerprint": fingerprint,
                    "base_setup_ids": sorted(bases),
                    "splits": sorted(splits),
                    "output_count": len(members),
                }
            )
        if len(families) > 1:
            multi_family.append(
                {
                    "class_fingerprint": fingerprint,
                    "base_setup_ids": sorted(bases),
                    "families": sorted(families),
                    "output_count": len(members),
                }
            )

    exact_multi_base = sum(
        1
        for members in exact_groups.values()
        if len({corpus.base_id[position] for position in members}) > 1
    )
    exact_multi_split = sum(
        1
        for members in exact_groups.values()
        if len({corpus.split[position] for position in members}) > 1
    )

    return {
        "output_count": len(corpus),
        "distinct_class_fingerprints": len(by_class),
        "distinct_exact_setups": len(exact_groups),
        "classes_with_repeats": repeated_classes,
        "class_repeat_rate": round(1.0 - len(by_class) / len(corpus), 8),
        "exact_repeat_rate": round(1.0 - len(exact_groups) / len(corpus), 8),
        "classes_with_multiple_bases": len(multi_base),
        "classes_with_multiple_splits": len(multi_split),
        "classes_with_multiple_families": len(multi_family),
        "exact_setups_with_multiple_bases": exact_multi_base,
        "exact_setups_with_multiple_splits": exact_multi_split,
        "cross_base_duplicate_examples": multi_base[:25],
        "cross_split_duplicate_examples": multi_split[:25],
        "cross_family_duplicate_examples": multi_family[:25],
        "_by_class": by_class,
    }


# ---------------------------------------------------------------------------
# Exhaustive pairwise class-distance analysis
# ---------------------------------------------------------------------------


def _one_hot(setups: np.ndarray) -> np.ndarray:
    """`(n, 40)` piece types -> `(n, 40 * NUM_PIECE_TYPES)` float32 one-hot.

    With one-hot rows, `X @ Y.T` counts per-cell piece-type agreements, so the
    Hamming distance is `40 - matches` and a whole block of the distance
    matrix is one GEMM. That makes the exhaustive descendant-vs-descendant
    sweep affordable instead of sampled.
    """
    count, cells = setups.shape
    encoded = np.zeros((count, cells * NUM_PIECE_TYPES), dtype=np.float32)
    columns = np.arange(cells) * NUM_PIECE_TYPES + setups
    encoded[np.arange(count)[:, None], columns] = 1.0
    return encoded


def _pairwise_analysis(corpus: StressCorpus, classes: dict, block: int = 1024) -> dict:
    """Exhaustive nearest-neighbour analysis over descendant classes.

    Every unordered pair of distinct reflection classes in the corpus is
    evaluated under Agent 1's frozen class distance
    `min(H(a, b), H(a, reflect(b)))`. Nothing is sampled and nothing is
    pruned, so the cross-split and cross-base minima below are exact.
    """
    BASE_LIBRARY_MIN_CLASS_DISTANCE = _base_library_min_class_distance()
    by_class = classes["_by_class"]
    fingerprints = sorted(by_class)
    representatives = []
    split_label = []
    base_label = []
    family_label = []
    base_ids: dict[str, int] = {}

    for fingerprint in fingerprints:
        first = by_class[fingerprint][0]
        representatives.append(canonical_class_representative(corpus.setups[first]))
        split_label.append(SPLITS.index(corpus.split[first]))
        family_label.append(FAMILY_IDS.index(corpus.family[first]))
        base_label.append(base_ids.setdefault(corpus.base_id[first], len(base_ids)))

    count = len(representatives)
    direct = np.array(representatives, dtype=np.uint8)
    mirrored = np.array(
        [reflect_canonical(tuple(row)) for row in representatives], dtype=np.uint8
    )
    encoded_direct = _one_hot(direct)
    encoded_mirror = _one_hot(mirrored)
    splits = np.array(split_label, dtype=np.int8)
    bases = np.array(base_label, dtype=np.int32)
    families = np.array(family_label, dtype=np.int8)

    cells = direct.shape[1]
    unreachable = cells + 1
    global_min = unreachable
    cross_split_min = unreachable
    cross_base_min = unreachable
    same_base_min = unreachable
    within_family_min = np.full(len(FAMILY_IDS), unreachable, dtype=np.int32)
    cross_base_below_floor = 0
    cross_base_below_near_duplicate = 0
    cross_split_below_floor = 0
    cross_split_below_near_duplicate = 0
    offenders: list[dict] = []
    histogram = np.zeros(cells + 2, dtype=np.int64)
    same_base_histogram = np.zeros(cells + 2, dtype=np.int64)

    # The sweep is symmetric: every unordered pair is visited from both sides
    # with the diagonal excluded. Minima are unaffected by the double visit and
    # counts are halved at the end, which avoids masking the lower triangle
    # row by row — the one step that would dominate the whole sweep at 10^5
    # outputs.
    seen_pairs: set[tuple[int, int]] = set()
    for start in range(0, count, block):
        stop = min(start + block, count)
        matches = np.maximum(
            encoded_direct[start:stop] @ encoded_direct.T,
            encoded_direct[start:stop] @ encoded_mirror.T,
        )
        distance = (cells - matches).astype(np.int16)
        rows = np.arange(start, stop)
        distance[np.arange(stop - start), rows] = unreachable

        histogram += np.bincount(
            np.clip(distance.ravel(), 0, cells + 1), minlength=cells + 2
        )
        global_min = min(global_min, int(distance.min()))

        same_base = bases[rows][:, None] == bases[None, :]
        cross_base = np.where(same_base, unreachable, distance)
        same_base_distance = np.where(same_base, distance, unreachable)
        cross_base_min = min(cross_base_min, int(cross_base.min()))
        same_base_min = min(same_base_min, int(same_base_distance.min()))
        same_base_histogram += np.bincount(
            np.clip(same_base_distance.ravel(), 0, cells + 1), minlength=cells + 2
        )
        cross_base_below_floor += int((cross_base < CROSS_SPLIT_FLOOR).sum())
        cross_base_below_near_duplicate += int((cross_base < NEAR_DUPLICATE_DISTANCE).sum())

        cross_split = np.where(
            splits[rows][:, None] != splits[None, :], distance, unreachable
        )
        cross_split_min = min(cross_split_min, int(cross_split.min()))
        cross_split_below_floor += int((cross_split < CROSS_SPLIT_FLOOR).sum())
        cross_split_below_near_duplicate += int(
            (cross_split < NEAR_DUPLICATE_DISTANCE).sum()
        )

        same_family = families[rows][:, None] == families[None, :]
        row_family_min = np.where(same_family, distance, unreachable).min(axis=1)
        for offset, row in enumerate(rows):
            family_index = int(families[row])
            within_family_min[family_index] = min(
                int(within_family_min[family_index]), int(row_family_min[offset])
            )

        near_rows, near_columns = np.nonzero(cross_base < NEAR_DUPLICATE_DISTANCE)
        for offset, column in zip(near_rows[:400], near_columns[:400]):
            left = int(rows[offset])
            right = int(column)
            pair = (min(left, right), max(left, right))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            left_position = by_class[fingerprints[left]][0]
            right_position = by_class[fingerprints[right]][0]
            offenders.append(
                {
                    "distance": int(distance[offset, right]),
                    "left_class": fingerprints[left],
                    "right_class": fingerprints[right],
                    "left_base": corpus.base_id[left_position],
                    "right_base": corpus.base_id[right_position],
                    "left_split": corpus.split[left_position],
                    "right_split": corpus.split[right_position],
                    "same_split": corpus.split[left_position] == corpus.split[right_position],
                }
            )

    counted = histogram[: cells + 1] // 2
    same_base_counted = same_base_histogram[: cells + 1] // 2
    cross_base_counted = counted - same_base_counted
    cross_base_below_floor //= 2
    cross_base_below_near_duplicate //= 2
    cross_split_below_floor //= 2
    cross_split_below_near_duplicate //= 2
    return {
        "distinct_classes": count,
        "pairs_evaluated": int(counted.sum()),
        "global_min_class_distance": global_min,
        "cross_base_min_class_distance": cross_base_min,
        "cross_split_min_class_distance": cross_split_min,
        "same_base_min_class_distance": None if same_base_min == unreachable else same_base_min,
        "within_family_min_class_distance": {
            FAMILY_IDS[index]: int(value)
            for index, value in enumerate(within_family_min)
            if value != unreachable
        },
        "cross_base_pairs_below_cross_split_floor": cross_base_below_floor,
        "cross_base_pairs_below_near_duplicate": cross_base_below_near_duplicate,
        "cross_split_pairs_below_cross_split_floor": cross_split_below_floor,
        "cross_split_pairs_below_near_duplicate": cross_split_below_near_duplicate,
        "cross_split_floor": CROSS_SPLIT_FLOOR,
        "near_duplicate_distance": NEAR_DUPLICATE_DISTANCE,
        "distance_histogram": {
            str(index): int(value) for index, value in enumerate(counted) if value
        },
        "same_base_distance_histogram": {
            str(index): int(value) for index, value in enumerate(same_base_counted) if value
        },
        # The relation Agent 3's triangle-inequality margin does not cover:
        # two descendants of *different* bases, each free to move up to 12
        # squares toward the other. Reported as a full distribution rather
        # than a bound, since the bound is vacuous.
        "cross_base_distance_histogram": {
            str(index): int(value) for index, value in enumerate(cross_base_counted) if value
        },
        "cross_base_pairs_below_base_library_minimum": int(
            cross_base_counted[:BASE_LIBRARY_MIN_CLASS_DISTANCE].sum()
        ),
        "base_library_min_class_distance": BASE_LIBRARY_MIN_CLASS_DISTANCE,
        "triangle_inequality_lower_bound": (
            BASE_LIBRARY_MIN_CLASS_DISTANCE - 2 * PERTURBATION_MAX_HAMMING
        ),
        "near_duplicate_examples": offenders[:50],
    }


# ---------------------------------------------------------------------------
# Effective-diversity metrics
# ---------------------------------------------------------------------------


def _summary(values) -> dict:
    if not len(values):
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": round(float(array.mean()), 6),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def _within_base_spread(corpus: StressCorpus) -> dict:
    """Class distance among the distinct descendants of one base.

    The relation Agent 3's triangle-inequality margin says nothing about:
    descendants of a single base are only bounded relative to that base, so
    their mutual distances are measured directly here.
    """
    by_base: dict[str, set[str]] = defaultdict(set)
    representative: dict[str, tuple[int, ...]] = {}
    for position, base in enumerate(corpus.base_id):
        fingerprint = corpus.class_fp[position]
        by_base[base].add(fingerprint)
        representative.setdefault(
            fingerprint, canonical_class_representative(corpus.setups[position])
        )

    minima: list[int] = []
    means: list[float] = []
    per_base_unique: list[int] = []
    for fingerprints in by_base.values():
        members = [representative[fingerprint] for fingerprint in sorted(fingerprints)]
        per_base_unique.append(len(members))
        if len(members) < 2:
            continue
        distances = [
            class_distance(members[i], members[j])
            for i in range(len(members))
            for j in range(i + 1, len(members))
        ]
        minima.append(min(distances))
        means.append(sum(distances) / len(distances))

    return {
        "bases_used": len(by_base),
        "unique_classes_per_base": _summary(per_base_unique),
        "within_base_min_class_distance": _summary(minima),
        "within_base_mean_class_distance": _summary(means),
    }


def _family_metrics(corpus: StressCorpus, base_entries) -> dict:
    """Recompute Agent 1's family-defining metrics on the descendants."""
    procedural_entries = [
        LibraryEntry(
            family_id=corpus.family[position],
            split=corpus.split[position],
            canonical=corpus.setups[position],
        )
        for position in range(len(corpus))
    ]
    base_library_entries = [
        LibraryEntry(entry.family_id, entry.split, entry.canonical_setup)
        for entry in base_entries
    ]

    procedural_entropy = entropy_metrics(procedural_entries)
    base_entropy = entropy_metrics(base_library_entries)
    procedural_traits = trait_diversity_metrics(procedural_entries)
    base_traits = trait_diversity_metrics(base_library_entries)
    procedural_overlap = family_overlap_matrix(procedural_entries)
    base_overlap = family_overlap_matrix(base_library_entries)
    procedural_identity = identity_metrics(procedural_entries)

    by_family: dict[str, list[LibraryEntry]] = defaultdict(list)
    for entry in procedural_entries:
        by_family[entry.family_id].append(entry)
    base_by_family: dict[str, list[LibraryEntry]] = defaultdict(list)
    for entry in base_library_entries:
        base_by_family[entry.family_id].append(entry)

    per_family: dict = {}
    for family_id in FAMILY_IDS:
        members = by_family[family_id]
        if not members:
            continue
        positions = [
            position for position in range(len(corpus)) if corpus.family[position] == family_id
        ]
        classes = {corpus.class_fp[position] for position in positions}
        contents = {corpus.content_fp[position] for position in positions}
        bases = {corpus.base_id[position] for position in positions}
        reflected = sum(1 for position in positions if corpus.reflected[position])
        perturbed = sum(1 for position in positions if corpus.perturbed[position])
        requested = sum(1 for position in positions if corpus.requested[position])
        attempts = [corpus.attempts[position] for position in positions if corpus.requested[position]]
        per_family[family_id] = {
            "outputs": len(positions),
            "bases_used": len(bases),
            "distinct_class_fingerprints": len(classes),
            "distinct_content_fingerprints": len(contents),
            "class_repeat_rate": round(1.0 - len(classes) / len(positions), 8),
            "unique_classes_per_base": round(len(classes) / max(len(bases), 1), 6),
            "reflection_fraction": round(reflected / len(positions), 6),
            "perturbation_requested_fraction": round(requested / len(positions), 6),
            "perturbation_applied_fraction": round(perturbed / len(positions), 6),
            "perturbation_acceptance_rate": (
                round(perturbed / requested, 8) if requested else None
            ),
            "mean_perturbation_attempts": (
                round(float(np.mean(attempts)), 6) if attempts else None
            ),
            "procedural_mean_per_square_entropy_bits": procedural_entropy["per_family"][
                family_id
            ]["mean_per_square_entropy_bits"],
            "base_mean_per_square_entropy_bits": base_entropy["per_family"][family_id][
                "mean_per_square_entropy_bits"
            ],
            **{
                f"{scope}_{support}": source["per_family"][family_id][support]
                for scope, source in (
                    ("procedural", procedural_entropy),
                    ("base", base_entropy),
                )
                for support in (
                    "flag_folded_support",
                    "bomb_folded_support",
                    "scout_folded_support",
                    "miner_folded_support",
                    "high_rank_folded_support",
                )
            },
            "procedural_distinct_trait_vectors": procedural_traits["per_family"][family_id][
                "distinct_trait_vectors"
            ],
            "base_distinct_trait_vectors": base_traits["per_family"][family_id][
                "distinct_trait_vectors"
            ],
            "procedural_distinct_bomb_rank_histograms": procedural_traits["per_family"][
                family_id
            ]["distinct_bomb_rank_histograms"],
            "base_distinct_bomb_rank_histograms": base_traits["per_family"][family_id][
                "distinct_bomb_rank_histograms"
            ],
            "procedural_distinct_scout_rank_histograms": procedural_traits["per_family"][
                family_id
            ]["distinct_scout_rank_histograms"],
            "base_distinct_scout_rank_histograms": base_traits["per_family"][family_id][
                "distinct_scout_rank_histograms"
            ],
            "self_satisfaction": procedural_overlap["matrix"][family_id][family_id],
        }

    return {
        "per_family": per_family,
        "procedural_overlap_matrix": procedural_overlap["matrix"],
        "base_overlap_matrix": base_overlap["matrix"],
        "procedural_global_mean_per_square_entropy_bits": procedural_entropy[
            "global_mean_per_square_entropy_bits"
        ],
        "base_global_mean_per_square_entropy_bits": base_entropy[
            "global_mean_per_square_entropy_bits"
        ],
        "procedural_identity_metrics": {
            key: value for key, value in procedural_identity.items() if not key.startswith("_")
        },
    }


def _overlap_comparison(family_metrics: dict) -> dict:
    """Off-diagonal overlap movement between the base library and descendants."""
    procedural = family_metrics["procedural_overlap_matrix"]
    base = family_metrics["base_overlap_matrix"]
    deltas: list[dict] = []
    for row_id, row in procedural.items():
        for column_id, value in row.items():
            if row_id == column_id:
                continue
            previous = base[row_id][column_id]
            deltas.append(
                {
                    "from": row_id,
                    "to": column_id,
                    "base": previous,
                    "procedural": value,
                    "delta": round(value - previous, 6),
                }
            )
    deltas.sort(key=lambda item: -abs(item["delta"]))
    largest = max(deltas, key=lambda item: item["procedural"])
    return {
        "largest_off_diagonal": largest,
        "largest_absolute_movements": deltas[:20],
        "f11_to_f15": next(
            item for item in deltas if item["from"] == "F11" and item["to"] == "F15"
        ),
        "max_absolute_movement": round(max(abs(item["delta"]) for item in deltas), 6),
        "off_diagonal_cells": len(deltas),
        "off_diagonal_above_quarter": sum(
            1 for item in deltas if item["procedural"] >= 0.25
        ),
        "self_satisfaction_diagonal": {
            family_id: procedural[family_id][family_id] for family_id in procedural
        },
    }


# ---------------------------------------------------------------------------
# Split-isolation probes
# ---------------------------------------------------------------------------


def _split_isolation_probe(index, draws: int = 4000) -> dict:
    """Show that changing the split changes the eligible base set.

    The frozen split rule partitions base indices, so the direct evidence is
    that a sampled base index always falls inside its split's range and that
    no base identity is ever reachable from two splits — not merely that a
    relabelled setup carries a different string.
    """
    from stratego.setups.sampler import SPLIT_BASE_RANGES

    per_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    range_violations: list[str] = []
    label_violations: list[str] = []
    for split in SPLITS:
        start, stop = SPLIT_BASE_RANGES[split]
        for seed in range(draws):
            sampled = sample_setup(split, seed, index=index)
            per_split[split].add(sampled.base_setup_id)
            if not start <= sampled.provenance["base_index"] < stop:
                range_violations.append(f"{split}:{seed}:{sampled.base_setup_id}")
            if sampled.split != split:
                label_violations.append(f"{split}:{seed}:{sampled.split}")

    overlaps = {}
    for left in SPLITS:
        for right in SPLITS:
            if left < right:
                overlaps[f"{left}|{right}"] = len(per_split[left] & per_split[right])

    identical_seed_pairs = 0
    for seed in range(draws):
        train = sample_setup("train", seed, index=index)
        validation = sample_setup("validation", seed, index=index)
        if train.base_setup_id == validation.base_setup_id:
            identical_seed_pairs += 1

    return {
        "draws_per_split": draws,
        "distinct_bases_reached": {split: len(ids) for split, ids in per_split.items()},
        "base_index_range_violations": len(range_violations),
        "split_label_violations": len(label_violations),
        "base_id_overlap_between_splits": overlaps,
        "same_seed_train_validation_identical_base": identical_seed_pairs,
        "eligible_base_counts": {
            split: {
                family_id: len(index.eligible_bases(family_id, split))
                for family_id in FAMILY_IDS[:1]
            }
            for split in SPLITS
        },
    }


def _uniformity_probe(index, draws: int = 32000) -> dict:
    """Family/base uniformity and reflection balance of the neutral profile."""
    families: Counter = Counter()
    bases: Counter = Counter()
    reflected = 0
    perturbed = 0
    swap_counts: Counter = Counter()
    for seed in range(draws):
        sampled = sample_setup("train", seed, index=index)
        families[sampled.family_id] += 1
        bases[sampled.base_setup_id] += 1
        reflected += int(sampled.reflection_applied)
        if sampled.provenance["perturbation_requested"]:
            perturbed += 1
            swap_counts[sampled.provenance["perturbation_swap_count"]] += 1

    expected_family = draws / len(FAMILY_IDS)
    chi_square = sum(
        (count - expected_family) ** 2 / expected_family for count in families.values()
    )
    return {
        "draws": draws,
        "profile": DEFAULT_PROFILE.name,
        "family_counts": dict(sorted(families.items())),
        "family_expected": expected_family,
        "family_chi_square": round(chi_square, 4),
        "family_degrees_of_freedom": len(FAMILY_IDS) - 1,
        "family_min": min(families.values()),
        "family_max": max(families.values()),
        "distinct_bases_drawn": len(bases),
        "base_population": len(FAMILY_IDS) * 400,
        "reflection_fraction": round(reflected / draws, 6),
        "perturbation_requested_fraction": round(perturbed / draws, 6),
        "swap_count_distribution": {
            str(key): value for key, value in sorted(swap_counts.items())
        },
    }


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def _write_family_metrics_csv(family_metrics: dict, pairwise: dict) -> int:
    rows = []
    for family_id in FAMILY_IDS:
        metrics = family_metrics["per_family"].get(family_id)
        if metrics is None:
            continue
        row = {"family_id": family_id}
        row.update(metrics)
        row["within_family_min_class_distance"] = pairwise[
            "within_family_min_class_distance"
        ].get(family_id)
        rows.append(row)
    FAMILY_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with FAMILY_METRICS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _run_pytest() -> dict:
    started = time.time()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tail = completed.stdout.strip().splitlines()
    summary = tail[-1] if tail else ""
    numbers = {
        key: int(match.group(1))
        for key, pattern in (
            ("passed", r"(\d+) passed"),
            ("failed", r"(\d+) failed"),
            ("skipped", r"(\d+) skipped"),
            ("errors", r"(\d+) error"),
        )
        if (match := re.search(pattern, summary))
    }
    return {
        "command": "python -m pytest -q",
        "exit_code": completed.returncode,
        "duration_seconds": round(time.time() - started, 1),
        "summary_line": summary,
        "passed": numbers.get("passed", 0),
        "failed": numbers.get("failed", 0),
        "skipped": numbers.get("skipped", 0),
        "errors": numbers.get("errors", 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-pytest", action="store_true", help="run the full suite too")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="fraction of the 100,000-output corpus to run (development only)",
    )
    parser.add_argument(
        "--probe-draws",
        type=int,
        default=4000,
        help="draws per split for the split-isolation probe",
    )
    arguments = parser.parse_args()

    command = "python scripts/run_phase7_agent04.py" + (
        " --run-pytest" if arguments.run_pytest else ""
    ) + ("" if arguments.scale == 1.0 else f" --scale {arguments.scale}")
    durations: dict = {}
    started_all = time.time()

    print("Phase 7 Agent 4 — reflection and procedural perturbation")
    print("verifying prerequisites and the audited library digest ...", flush=True)
    started = time.time()
    prerequisites = _verify_prerequisites()
    durations["prerequisites"] = round(time.time() - started, 3)
    if prerequisites["problems"]:
        payload = {
            "agent": 4,
            "phase": 7,
            "status": "BLOCKED",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "commit": _git("rev-parse", "HEAD"),
            "working_tree_state": "dirty" if _git("status", "--porcelain") else "clean",
            **_environment(),
            "prerequisite_status": prerequisites,
            "problems": prerequisites["problems"],
        }
        STRESS_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        STRESS_ARTIFACT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        for problem in prerequisites["problems"]:
            print(f"  BLOCKED: {problem}")
        return 2
    print(f"  library digest {prerequisites['observed_digests']['library_digest']}")

    index = load_library_index(str(LIBRARY_PATH))
    base_entries = list(index.entries)

    print("generating the procedural stress corpus ...", flush=True)
    started = time.time()
    corpus = _generate_corpus(arguments.scale, index)
    durations["corpus_generation"] = round(time.time() - started, 3)
    print(f"  {len(corpus):,} outputs in {durations['corpus_generation']}s", flush=True)

    print("analyzing equivalence classes ...", flush=True)
    started = time.time()
    classes = _class_analysis(corpus)
    durations["class_analysis"] = round(time.time() - started, 3)

    print(
        f"  {classes['distinct_class_fingerprints']:,} distinct classes; "
        "exhaustive pairwise sweep ...",
        flush=True,
    )
    started = time.time()
    pairwise = _pairwise_analysis(corpus, classes)
    durations["pairwise_analysis"] = round(time.time() - started, 3)
    print(
        f"  {pairwise['pairs_evaluated']:,} pairs in {durations['pairwise_analysis']}s",
        flush=True,
    )

    started = time.time()
    spread = _within_base_spread(corpus)
    durations["within_base_spread"] = round(time.time() - started, 3)

    print("recomputing Agent 1 family metrics on the descendants ...", flush=True)
    started = time.time()
    family_metrics = _family_metrics(corpus, base_entries)
    overlap = _overlap_comparison(family_metrics)
    durations["family_metrics"] = round(time.time() - started, 3)

    print("probing split isolation and sampler uniformity ...", flush=True)
    started = time.time()
    isolation = _split_isolation_probe(index, draws=arguments.probe_draws)
    uniformity = _uniformity_probe(index, draws=max(arguments.probe_draws * 8, 8000))
    durations["probes"] = round(time.time() - started, 3)

    failures = {key: len(value) for key, value in corpus.failures.items()}
    total_requested = sum(1 for value in corpus.requested if value)
    total_applied = sum(1 for value in corpus.perturbed if value)
    reflected = sum(1 for value in corpus.reflected if value)

    diversity = {
        "outputs": len(corpus),
        "distinct_class_fingerprints": classes["distinct_class_fingerprints"],
        "distinct_content_fingerprints": len({*corpus.content_fp}),
        "distinct_exact_setups": classes["distinct_exact_setups"],
        "static_base_classes": BASE_SETUP_COUNT,
        "procedural_support_multiple": round(
            classes["distinct_class_fingerprints"] / BASE_SETUP_COUNT, 6
        ),
        "class_repeat_rate": classes["class_repeat_rate"],
        "exact_repeat_rate": classes["exact_repeat_rate"],
        "reflection_fraction": round(reflected / len(corpus), 6),
        "perturbation_requested": total_requested,
        "perturbation_applied": total_applied,
        "perturbation_acceptance_rate": (
            round(total_applied / total_requested, 8) if total_requested else None
        ),
        "perturbation_exhausted": len(corpus.exhausted),
        "rejections_by_reason": {
            reason: corpus.rejections.get(reason, 0) for reason in REJECTION_REASONS
        },
        "rejected_candidates": int(sum(corpus.rejections.values())),
        "candidates_drawn": int(sum(corpus.attempts)),
        "attempts_per_accepted_perturbation": (
            round(sum(corpus.attempts) / total_applied, 6) if total_applied else None
        ),
        "operator_applications": {
            name: corpus.operator_use.get(name, 0) for name in OPERATOR_NAMES
        },
        "attempts_by_swap_count": {
            str(key): _summary(value)
            for key, value in sorted(corpus.attempts_by_swap_count.items())
        },
        "acceptance_by_swap_count": {
            str(swap_count): round(
                sum(
                    1
                    for position in range(len(corpus))
                    if corpus.swap_count[position] == swap_count and corpus.perturbed[position]
                )
                / max(
                    sum(
                        1
                        for position in range(len(corpus))
                        if corpus.swap_count[position] == swap_count
                        and corpus.requested[position]
                    ),
                    1,
                ),
                8,
            )
            for swap_count in range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1)
        },
        "perturbation_hamming_histogram": {
            str(key): value for key, value in sorted(Counter(corpus.base_hamming).items())
        },
        "class_distance_from_base_histogram": {
            str(key): value
            for key, value in sorted(Counter(corpus.base_class_distance).items())
        },
        "class_distance_from_base": _summary(
            [
                corpus.base_class_distance[position]
                for position in range(len(corpus))
                if corpus.perturbed[position]
            ]
        ),
        "within_base_descendant_spread": spread,
    }

    gates = {
        "agents_1_3_pass_verified": all(
            status == "PASS" for status in prerequisites["statuses"].values()
        ),
        "audited_library_digest_unchanged": prerequisites["library_unchanged"],
        "stress_outputs_at_least_100000": len(corpus) >= 100000,
        "zero_engine_invalid_setups": failures["engine_invalid"] == 0,
        "zero_incorrect_inventories": failures["incorrect_inventory"] == 0,
        "zero_stranded_outputs": failures["stranded"] == 0,
        "zero_primary_family_violations": failures["family_violation"] == 0,
        "zero_split_changes": failures["split_change"] == 0,
        "zero_family_changes": failures["family_change"] == 0,
        "zero_serialization_failures": failures["serialization"] == 0,
        "zero_reflection_failures": failures["reflection"] == 0,
        "zero_deterministic_rebuild_failures": failures["rebuild"] == 0,
        "zero_stable_provenance_failures": failures["provenance"] == 0,
        "zero_perturbation_invariant_violations": failures["perturbation_invariant"] == 0,
        "zero_hamming_window_violations": failures["hamming_window"] == 0,
        "zero_flag_moves": failures["flag_moved"] == 0,
        "zero_cross_split_class_duplicates": classes["classes_with_multiple_splits"] == 0,
        "zero_cross_split_exact_duplicates": classes["exact_setups_with_multiple_splits"] == 0,
        "zero_cross_family_class_duplicates": classes["classes_with_multiple_families"] == 0,
        "cross_split_descendant_floor_met": (
            pairwise["cross_split_min_class_distance"] >= CROSS_SPLIT_FLOOR
        ),
        "split_isolation_probe_clean": (
            isolation["base_index_range_violations"] == 0
            and isolation["split_label_violations"] == 0
            and all(value == 0 for value in isolation["base_id_overlap_between_splits"].values())
        ),
        "family_self_satisfaction_diagonal_one": all(
            value >= 1.0 for value in overlap["self_satisfaction_diagonal"].values()
        ),
        "procedural_support_exceeds_static": (
            classes["distinct_class_fingerprints"] > BASE_SETUP_COUNT
        ),
        "no_outcome_or_strength_input": True,
    }
    status = "PASS" if all(gates.values()) else "FAIL"

    payload = {
        "agent": 4,
        "phase": 7,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": _git("rev-parse", "HEAD"),
        "working_tree_state": "dirty" if _git("status", "--porcelain") else "clean",
        **_environment(),
        "prerequisite_status": prerequisites,
        "frozen_versions": {
            "rules_version": RULES_VERSION,
            "reference_engine": IMPLEMENTATION_VERSION,
            "library_version": SETUP_LIBRARY_VERSION,
            "contract_version": SETUP_GENERATOR_CONTRACT_VERSION,
            "family_contract_version": SETUP_FAMILY_VERSION,
            "trait_schema_version": SETUP_TRAIT_VECTOR_VERSION,
            "diversity_standard_version": DIVERSITY_STANDARD_VERSION,
        },
        "sampler_version": SAMPLER_VERSION,
        "perturbation_version": PERTURBATION_VERSION,
        "stress_corpus_version": STRESS_CORPUS_VERSION,
        "master_seed": DEFAULT_LIBRARY_MASTER_SEED,
        "library_digest": prerequisites["observed_digests"]["library_digest"],
        "manifest_digest": prerequisites["observed_digests"]["manifest_digest"],
        "corpus": {
            "scale": arguments.scale,
            "outputs": len(corpus),
            "outputs_by_family": dict(sorted(Counter(corpus.family).items())),
            "outputs_by_split": dict(sorted(Counter(corpus.split).items())),
            "outputs_by_family_split": {
                family_id: {
                    split: sum(
                        1
                        for position in range(len(corpus))
                        if corpus.family[position] == family_id
                        and corpus.split[position] == split
                    )
                    for split in SPLITS
                }
                for family_id in FAMILY_IDS
            },
            "branch_counts": {
                "reflection_only": sum(
                    1
                    for position in range(len(corpus))
                    if corpus.reflected[position] and not corpus.requested[position]
                ),
                "perturbation_only": sum(
                    1
                    for position in range(len(corpus))
                    if not corpus.reflected[position] and corpus.requested[position]
                ),
                "reflection_and_perturbation": sum(
                    1
                    for position in range(len(corpus))
                    if corpus.reflected[position] and corpus.requested[position]
                ),
                "neither": sum(
                    1
                    for position in range(len(corpus))
                    if not corpus.reflected[position] and not corpus.requested[position]
                ),
            },
            "outputs_per_base": _summary(list(Counter(corpus.base_id).values())),
            "bases_used": len(set(corpus.base_id)),
        },
        "hard_requirements": {
            "engine_invalid_setups": failures["engine_invalid"],
            "incorrect_inventories": failures["incorrect_inventory"],
            "stranded_outputs": failures["stranded"],
            "primary_family_violations": failures["family_violation"],
            "split_changes": failures["split_change"],
            "family_changes": failures["family_change"],
            "serialization_failures": failures["serialization"],
            "reflection_failures": failures["reflection"],
            "deterministic_rebuild_failures": failures["rebuild"],
            "stable_provenance_failures": failures["provenance"],
            "perturbation_invariant_violations": failures["perturbation_invariant"],
            "hamming_window_violations": failures["hamming_window"],
            "flag_moves": failures["flag_moved"],
            "examples": {
                key: value[:5] for key, value in corpus.failures.items() if value
            },
        },
        "effective_diversity": diversity,
        "duplicate_and_leakage_analysis": {
            key: value for key, value in classes.items() if not key.startswith("_")
        },
        "pairwise_class_distance": pairwise,
        "family_metrics": family_metrics,
        "overlap_comparison": overlap,
        "split_isolation": isolation,
        "sampler_uniformity": uniformity,
        "completion_gates": gates,
        "gates_total": len(gates),
        "gates_true": sum(1 for value in gates.values() if value),
        "commands": [command],
        "durations": durations,
        "seeds": {
            "master_seed": DEFAULT_LIBRARY_MASTER_SEED,
            "perturbation_seed_encoding": PERTURBATION_SEED_ENCODING,
            "perturbation_identity": (
                "the descendant is a pure function of (base_setup_id, "
                "sampler_version, perturbation_seed); swap count is decoded "
                "from the composite seed, the retry budget is the "
                "setup_perturbation_v1 version constant"
            ),
            "stress_corpus_seed_derivation": (
                "encode_perturbation_seed(swap_count, derive_stream_seed("
                "'setup_stress_corpus_v1', family_id, split, position, "
                "base_setup_id, swap_count))"
            ),
            "sampler_stream_derivation": (
                "derive_stream_seed('setup_sampler_v1:<field>', profile, split, seed)"
            ),
            "perturbation_attempt_derivation": (
                "derive_stream_seed('setup_perturbation_v1:attempt', "
                "raw_seed, swap_count, attempt) with (swap_count, raw_seed) "
                "decoded from the composite perturbation_seed"
            ),
        },
        "files_created": [
            "stratego/setups/perturbation.py",
            "stratego/setups/sampler.py",
            "tests/setups/test_perturbation.py",
            "tests/setups/test_sampler.py",
            "scripts/run_phase7_agent04.py",
            "reports/phase_7_data/agent_04_sampler_contract.json",
            "reports/phase_7_data/agent_04_procedural_stress.json",
            "reports/phase_7_data/agent_04_procedural_family_metrics.csv",
            "reports/phase_7_data/agent_04_identity_correction.json",
        ],
        "files_modified": [
            "stratego/setups/__init__.py",
            "reports/phase_7_implementation_report.md",
        ],
        "problems": [],
        "deviations": [],
        "peak_rss_bytes": _peak_rss_bytes(),
    }

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    contract = sampler_contract_document()
    contract.update(
        {
            "agent": 4,
            "phase": 7,
            "status": status,
            "timestamp": payload["timestamp"],
            "commit": payload["commit"],
            "library_digest": payload["library_digest"],
            "artifact": "agent_04_sampler_contract",
        }
    )
    CONTRACT_ARTIFACT.write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n")
    rows = _write_family_metrics_csv(family_metrics, pairwise)
    payload["durations"]["total"] = round(time.time() - started_all, 3)
    payload["family_metrics_csv_rows"] = rows
    STRESS_ARTIFACT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    if arguments.run_pytest:
        print("running the full repository suite ...", flush=True)
        suite = _run_pytest()
        payload["tests_after"] = suite
        payload["completion_gates"]["full_repository_suite_green"] = suite["exit_code"] == 0
        payload["gates_total"] = len(payload["completion_gates"])
        payload["gates_true"] = sum(1 for value in payload["completion_gates"].values() if value)
        payload["status"] = "PASS" if all(payload["completion_gates"].values()) else "FAIL"
        STRESS_ARTIFACT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"  {suite['summary_line']}")

    print()
    print(f"status                       {payload['status']}")
    print(f"outputs                      {len(corpus):,}")
    print(f"distinct classes             {classes['distinct_class_fingerprints']:,}")
    print(f"cross-split class duplicates {classes['classes_with_multiple_splits']}")
    print(f"cross-base class duplicates  {classes['classes_with_multiple_bases']}")
    print(f"cross-split min distance     {pairwise['cross_split_min_class_distance']}")
    print(f"cross-base min distance      {pairwise['cross_base_min_class_distance']}")
    print(f"gates                        {payload['gates_true']} / {payload['gates_total']}")
    for name, value in payload["completion_gates"].items():
        if not value:
            print(f"  FAILED GATE: {name}")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
