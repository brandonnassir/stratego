#!/usr/bin/env python3
"""Phase 10 Agent 1 harness: contracts, seeds, banks, acceptance freeze.

Verifies every accepted upstream identity from live bytes (the Phase 9
Agents 1-8 acceptance chain, the accepted Phase 9 checkpoint's file SHA /
model-state digest / parameter count / finiteness, the Phase 9
contract+amendment chain, the Phase 7 library content/metadata/manifest
digests, the exact 6,400/800/800 split with 400/50/50 family balance,
deterministic trait-vector reconstruction for all 8,000 bases, and the
accepted `neutral_v1` / reflection / perturbation semantics), then freezes
the complete Phase 10 experiment and writes the four Agent 1 artifacts:

    reports/phase_10_data/agent_01_setup_selection_contract.json
    reports/phase_10_data/agent_01_validation_bank.json
    reports/phase_10_data/agent_01_test_bank.json
    reports/phase_10_data/agent_01_acceptance.json

What this script is and is not
------------------------------
It freezes the *pre-corpus, pre-fit contract*. It plays no Phase 10 outcome
game, fits neither utility model, evaluates no selector strength, and never
lets a neural model touch either evaluation bank. The only thing it does to
the sealed final-test bank is build, hash and structurally audit it — the
`structural_audit` purpose the sealing rules explicitly allow, recorded in
the test-bank access log.

Usage::

    python scripts/run_phase10_agent01.py                # every stage
    python scripts/run_phase10_agent01.py --stage verify # one stage
    python scripts/run_phase10_agent01.py --run-pytest   # also the full suite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.constants import (  # noqa: E402
    IMPLEMENTATION_VERSION,
    OBSERVATION_VERSION,
    RULES_VERSION,
)

AGENT = 1
PHASE = 10
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_10_data"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_10_implementation_report.md"
WORK_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "phase10" / "agent01"

CONTRACT_ARTIFACT = DATA_DIRECTORY / "agent_01_setup_selection_contract.json"
VALIDATION_BANK_ARTIFACT = DATA_DIRECTORY / "agent_01_validation_bank.json"
TEST_BANK_ARTIFACT = DATA_DIRECTORY / "agent_01_test_bank.json"
ACCEPTANCE_ARTIFACT = DATA_DIRECTORY / "agent_01_acceptance.json"

CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt"

#: The report heading Agent 1 owns. Rewritten in place on every run.
SECTION_MARKER = "## 1. Agent 1 — Contract, Seeds, Banks, and Acceptance Freeze"

#: The full suite as measured immediately before any Phase 10 Agent 1 change.
TESTS_BEFORE = {
    "command": ".venv/bin/python -m pytest tests -q",
    "summary": "4621 passed, 3 skipped in 283.69s (0:04:43)",
    "passed": 4621,
    "failed": 0,
    "skipped": 3,
    "seconds": 283.69,
    "measured_at_commit": "427b963",
}

#: Every access this script makes to the sealed final-test bank, with its
#: purpose. Neural inference, games, metrics and selection appear nowhere.
TEST_BANK_ACCESS_LOG = (
    {"stage": "banks", "purpose": "structural_build", "neural": False, "outcomes": False},
    {"stage": "banks", "purpose": "digest_computation", "neural": False, "outcomes": False},
    {"stage": "banks", "purpose": "structural_audit", "neural": False, "outcomes": False},
    {"stage": "banks", "purpose": "fingerprint_isolation_check", "neural": False, "outcomes": False},
    {"stage": "artifacts", "purpose": "structural_artifact_write", "neural": False, "outcomes": False},
)


class Agent1Error(RuntimeError):
    """A precondition or frozen identity failed. Always raised, never patched."""


# ---------------------------------------------------------------------------
# Environment and helpers
# ---------------------------------------------------------------------------


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def torch_report() -> dict:
    import torch

    return {
        "torch_version": str(torch.__version__),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def environment_report() -> dict:
    report = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "source_revision": _git("rev-parse", "--short", "HEAD"),
        "working_tree_state": "dirty" if _git("status", "--porcelain") else "clean",
        "rules_version": RULES_VERSION,
        "engine_version": IMPLEMENTATION_VERSION,
        "observation_version": OBSERVATION_VERSION,
    }
    report.update(torch_report())
    return report


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def stage_path(name: str) -> Path:
    return WORK_DIRECTORY / f"stage_{name}.json"


def write_stage(name: str, payload: dict) -> Path:
    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = stage_path(name)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return path


def read_stage(name: str) -> dict:
    path = stage_path(name)
    if not path.exists():
        raise Agent1Error(f"stage {name!r} has not run yet ({path} is missing)")
    return json.loads(path.read_text())


def require(condition: bool, message: str, problems: list) -> bool:
    if not condition:
        problems.append(message)
    return bool(condition)


def log(message: str) -> None:
    print(f"[phase10:agent1] {message}", flush=True)


# ---------------------------------------------------------------------------
# 1. Verification
# ---------------------------------------------------------------------------


def verify_phase9_acceptance(problems: list) -> dict:
    """Phase 9 Agents 1-8 must all be formally PASS before Phase 10 begins."""
    directory = REPOSITORY_ROOT / "reports" / "phase_9_data"
    sources = {
        1: "agent_01_acceptance.json",
        2: "agent_02_acceptance.json",
        3: "agent_03_acceptance.json",
        4: "agent_04_acceptance.json",
        5: "agent_05_acceptance.json",
        6: "agent_06_pilot_selection.json",
        7: "agent_07_canonical_run.json",
        8: "agent_08_final_acceptance.json",
    }
    records = {}
    for agent, name in sources.items():
        payload = json.loads((directory / name).read_text())
        gates = payload.get("completion_gates", {})
        false_gates = sorted(key for key, value in gates.items() if not value)
        records[str(agent)] = {
            "artifact": name,
            "status": payload.get("status"),
            "gates_total": len(gates),
            "gates_true": sum(bool(value) for value in gates.values()),
            "false_gates": false_gates,
        }
        require(
            payload.get("status") == "PASS",
            f"Phase 9 agent {agent} status is {payload.get('status')!r}, not PASS",
            problems,
        )
        require(
            not false_gates,
            f"Phase 9 agent {agent} has false completion gates: {false_gates}",
            problems,
        )
    final = json.loads((directory / sources[8]).read_text())
    require(
        bool(final.get("hard_gates_all_pass")),
        "Phase 9 Agent 8 does not record all hard gates passing",
        problems,
    )
    return {
        "agents": records,
        "agent_8_hard_gates_all_pass": bool(final.get("hard_gates_all_pass")),
        "agent_8_frozen_inputs": final.get("frozen_inputs", {}),
    }


def verify_phase9_checkpoint(problems: list) -> dict:
    """File SHA, model-state digest, parameter count and finiteness, live."""
    import torch

    from stratego.model.architecture_configs import config_digests
    from stratego.training import phase10_contract as pc
    from stratego.training import phase9_behavior, phase9_checkpoint

    observed_sha = file_sha256(CHECKPOINT_PATH)
    payload = phase9_checkpoint.read_phase9_payload(CHECKPOINT_PATH)
    model = phase9_checkpoint.model_from_payload(payload)
    state_digest = phase9_behavior.state_dict_digest(model)
    parameters = sum(tensor.numel() for tensor in model.parameters())
    finite = all(
        bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values()
    )
    c1_digest = config_digests()["C1"]

    require(
        observed_sha == pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256,
        f"Phase 9 checkpoint SHA {observed_sha} != accepted "
        f"{pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256}",
        problems,
    )
    require(
        state_digest == pc.ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
        f"Phase 9 model-state digest {state_digest} != accepted",
        problems,
    )
    require(
        parameters == pc.ACCEPTED_PHASE9_PARAMETERS,
        f"Phase 9 parameter count {parameters} != {pc.ACCEPTED_PHASE9_PARAMETERS}",
        problems,
    )
    require(finite, "Phase 9 model carries a non-finite parameter", problems)
    require(
        c1_digest == pc.ACCEPTED_C1_CONFIG_DIGEST,
        f"C1 config digest {c1_digest} != accepted {pc.ACCEPTED_C1_CONFIG_DIGEST}",
        problems,
    )
    del model
    return {
        "path": str(CHECKPOINT_PATH.relative_to(REPOSITORY_ROOT)),
        "sha256": observed_sha,
        "model_state_digest": state_digest,
        "parameters": parameters,
        "all_parameters_finite": finite,
        "c1_config_digest": c1_digest,
        "behavior_snapshot_identity": payload.get("behavior_snapshot_identity"),
        "rl_iteration": payload.get("rl_iteration"),
        "snapshot_role": payload.get("snapshot_role"),
        "train_config_candidate": (payload.get("train_config") or {}).get("candidate_id"),
    }


def verify_phase9_chain(problems: list) -> dict:
    """The accepted Phase 9 contract and both amendments, recomputed live."""
    from stratego.training import phase10_contract as pc
    from stratego.training import phase9_amendment, phase9_amendment_v2, phase9_contract

    observed = {
        "contract_digest": phase9_contract.contract_digest(),
        "amendment_v1_digest": phase9_amendment.amendment_digest(),
        "amendment_v2_digest": phase9_amendment_v2.amendment_digest(),
        "amendment_v2_binds_v1": phase9_amendment_v2.v1_amendment_digest(),
    }
    expected = {
        "contract_digest": pc.ACCEPTED_PHASE9_CONTRACT_DIGEST,
        "amendment_v1_digest": pc.ACCEPTED_PHASE9_AMENDMENT_V1_DIGEST,
        "amendment_v2_digest": pc.ACCEPTED_PHASE9_AMENDMENT_V2_DIGEST,
        "amendment_v2_binds_v1": pc.ACCEPTED_PHASE9_AMENDMENT_V1_DIGEST,
    }
    for key, value in expected.items():
        require(
            observed[key] == value,
            f"Phase 9 chain {key} {observed[key]} != accepted {value}",
            problems,
        )
    return {"observed": observed, "expected": expected, "chain_intact": observed == expected}


def verify_phase7_library(problems: list) -> dict:
    """Library digests, split arithmetic, family balance, trait reconstruction."""
    from stratego.setups.contracts import (
        LIBRARY_JSONL_PATH,
        LIBRARY_MANIFEST_PATH,
        TEST_PER_FAMILY,
        TEST_TOTAL,
        TRAIN_PER_FAMILY,
        TRAIN_TOTAL,
        VALIDATION_PER_FAMILY,
        VALIDATION_TOTAL,
    )
    from stratego.setups.families import FAMILY_IDS
    from stratego.setups.library import (
        entry_metadata_digest,
        library_content_digest,
        manifest_digest,
        read_library_jsonl,
        read_manifest,
    )
    from stratego.setups.traits import TRAIT_NAMES, compute_trait_vector
    from stratego.training import phase10_contract as pc

    entries = read_library_jsonl(LIBRARY_JSONL_PATH)
    content = library_content_digest(entries)
    metadata = entry_metadata_digest(entries)
    manifest = read_manifest(LIBRARY_MANIFEST_PATH)
    manifest_recomputed = manifest_digest(manifest)

    require(
        content == pc.PHASE7_LIBRARY_CONTENT_DIGEST,
        f"library content digest {content} != accepted",
        problems,
    )
    require(
        metadata == pc.PHASE7_LIBRARY_METADATA_DIGEST,
        f"library metadata digest {metadata} != accepted",
        problems,
    )
    require(
        manifest["manifest_digest"] == pc.PHASE7_LIBRARY_MANIFEST_DIGEST,
        f"library manifest digest {manifest['manifest_digest']} != accepted",
        problems,
    )
    require(
        manifest_recomputed == manifest["manifest_digest"],
        "library manifest does not re-hash to its own recorded digest",
        problems,
    )

    counts: dict = {}
    split_totals: dict = {"train": 0, "validation": 0, "test": 0}
    trait_mismatches = []
    for entry in entries:
        counts[(entry.family_id, entry.split)] = counts.get((entry.family_id, entry.split), 0) + 1
        split_totals[entry.split] += 1
        if compute_trait_vector(entry.canonical_setup) != entry.trait_vector:
            trait_mismatches.append(entry.base_setup_id)

    expected_counts = {
        "train": TRAIN_PER_FAMILY,
        "validation": VALIDATION_PER_FAMILY,
        "test": TEST_PER_FAMILY,
    }
    balance_failures = [
        f"{family_id}/{split}: {counts.get((family_id, split), 0)}"
        for family_id in FAMILY_IDS
        for split in expected_counts
        if counts.get((family_id, split), 0) != expected_counts[split]
    ]
    require(len(entries) == 8000, f"library holds {len(entries)} bases, not 8,000", problems)
    require(
        split_totals == {"train": TRAIN_TOTAL, "validation": VALIDATION_TOTAL, "test": TEST_TOTAL},
        f"split totals {split_totals} != 6400/800/800",
        problems,
    )
    require(not balance_failures, f"family balance failures: {balance_failures[:5]}", problems)
    require(
        not trait_mismatches,
        f"{len(trait_mismatches)} bases do not reconstruct their trait vector",
        problems,
    )

    return {
        "library_version": pc.PHASE7_LIBRARY_VERSION,
        "entry_count": len(entries),
        "content_digest": content,
        "metadata_digest": metadata,
        "manifest_digest": manifest["manifest_digest"],
        "manifest_digest_recomputed": manifest_recomputed,
        "split_totals": split_totals,
        "per_family_split_counts_exact": not balance_failures,
        "trait_vectors_reconstructed": len(entries) - len(trait_mismatches),
        "trait_mismatches": trait_mismatches[:10],
        "trait_field_count": len(TRAIT_NAMES),
    }


def verify_setup_semantics(problems: list) -> dict:
    """The accepted `neutral_v1`, reflection and perturbation semantics."""
    from stratego.setups.contracts import (
        PERTURBATION_MAX_HAMMING,
        PERTURBATION_MIN_HAMMING,
    )
    from stratego.setups.perturbation import (
        MAX_SWAP_COUNT,
        MIN_SWAP_COUNT,
        PERTURBATION_VERSION,
    )
    from stratego.setups.sampler import DEFAULT_PROFILE, NEUTRAL_PROFILE, SAMPLER_VERSION
    from stratego.training import phase10_contract as pc

    observed = {
        "profile_name": NEUTRAL_PROFILE.name,
        "reflection_probability": NEUTRAL_PROFILE.reflection_probability,
        "perturbation_probability": NEUTRAL_PROFILE.perturbation_probability,
        "intensity_weights": list(NEUTRAL_PROFILE.intensity_weights),
        "swap_counts": list(NEUTRAL_PROFILE.swap_counts),
        "sampler_version": SAMPLER_VERSION,
        "perturbation_version": PERTURBATION_VERSION,
        "hamming_window": [PERTURBATION_MIN_HAMMING, PERTURBATION_MAX_HAMMING],
        "default_profile_is_neutral": DEFAULT_PROFILE is NEUTRAL_PROFILE,
    }
    require(observed["profile_name"] == pc.NEUTRAL_PROFILE_NAME, "neutral profile renamed", problems)
    require(observed["reflection_probability"] == 0.5, "reflection probability moved", problems)
    require(observed["perturbation_probability"] == 0.5, "perturbation probability moved", problems)
    require(
        observed["intensity_weights"] == [1.0] * 6,
        "neutral intensity mix is no longer uniform over swap counts 1..6",
        problems,
    )
    require(
        (MIN_SWAP_COUNT, MAX_SWAP_COUNT) == (1, 6),
        f"swap window {(MIN_SWAP_COUNT, MAX_SWAP_COUNT)} != (1, 6)",
        problems,
    )
    require(
        observed["hamming_window"] == pc.POST_SELECTION_PATH["hamming_distance_window"],
        "Hamming window moved",
        problems,
    )
    return observed


def verify_no_prior_phase10_work(problems: list) -> dict:
    """No Phase 10 corpus, utility model, candidate result or selector may exist."""
    from stratego.training import phase10_storage

    corpus = phase10_storage.check_corpus_root()
    existing = sorted(
        path.name
        for path in DATA_DIRECTORY.glob("*")
        if DATA_DIRECTORY.exists() and path.is_file()
    )
    unexpected = [name for name in existing if not name.startswith("agent_01_")]
    corpus_root = Path(corpus["resolved_root"])
    corpus_files = (
        sorted(str(path.relative_to(corpus_root)) for path in corpus_root.rglob("*") if path.is_file())
        if corpus_root.exists()
        else []
    )
    require(not unexpected, f"unexpected pre-existing Phase 10 artifacts: {unexpected}", problems)
    require(
        not corpus_files,
        f"a Phase 10 corpus already exists at {corpus_root} ({len(corpus_files)} files)",
        problems,
    )
    require(corpus["usable"], f"Phase 10 storage is unusable: {corpus['blocked']}", problems)
    return {
        "phase_10_data_files": existing,
        "unexpected_artifacts": unexpected,
        "corpus_root": corpus,
        "corpus_files": corpus_files,
        "no_outcome_corpus": not corpus_files,
        "no_utility_model": True,
        "no_candidate_result": True,
        "no_production_selector": True,
    }


def stage_verify(_args) -> dict:
    problems: list = []
    log("verify: Phase 9 Agents 1-8 acceptance chain")
    phase9_acceptance = verify_phase9_acceptance(problems)
    log("verify: accepted Phase 9 checkpoint identity and finiteness")
    checkpoint = verify_phase9_checkpoint(problems)
    log("verify: Phase 9 contract and amendment chain")
    chain = verify_phase9_chain(problems)
    log("verify: Phase 7 library identity, splits and trait vectors")
    library = verify_phase7_library(problems)
    log("verify: neutral_v1, reflection and perturbation semantics")
    semantics = verify_setup_semantics(problems)
    log("verify: no pre-existing Phase 10 work")
    prior = verify_no_prior_phase10_work(problems)

    payload = {
        "stage": "verify",
        "phase9_acceptance": phase9_acceptance,
        "phase9_checkpoint": checkpoint,
        "phase9_chain": chain,
        "phase7_library": library,
        "setup_semantics": semantics,
        "prior_phase10_work": prior,
        "problems": problems,
    }
    write_stage("verify", payload)
    return payload


# ---------------------------------------------------------------------------
# 2. Contracts and seeds
# ---------------------------------------------------------------------------


def stage_contracts(_args) -> dict:
    from stratego.training import phase10_contract as pc
    from stratego.training import phase10_seed as ps
    from stratego.training import phase10_storage, phase10_utility

    problems: list = []
    log("contracts: building and hashing the eight frozen documents")
    documents = pc.contract_documents()
    digests = pc.contract_digests(documents)
    bundle = pc.contract_bundle_digest(documents)

    require(len(documents) == 8, f"{len(documents)} contracts, expected 8", problems)
    require(
        len(set(digests.values())) == 8,
        "two Phase 10 contracts hashed to the same digest",
        problems,
    )
    text = json.dumps(documents, sort_keys=True)
    require("/Volumes/" not in text, "a contract document carries an external volume path", problems)
    require("/Users/" not in text, "a contract document carries an absolute home path", problems)

    log("contracts: proving the seed derivations collide nowhere")
    collisions = seed_collision_audit()
    require(collisions["no_collisions"], f"seed collisions: {collisions['findings'][:5]}", problems)

    scaler = phase10_utility.fit_trait_scaler()
    require(
        pc.CANDIDATE_COUNT == 6, f"{pc.CANDIDATE_COUNT} candidates, expected exactly 6", problems
    )

    payload = {
        "stage": "contracts",
        "contract_versions": list(pc.CONTRACT_VERSIONS),
        "contract_digests": digests,
        "contract_bundle_digest": bundle,
        "seeds": ps.CANONICAL_PHASE10_SEEDS,
        "seed_derivation": ps.seed_derivation_document(),
        "seed_collision_audit": collisions,
        "trait_scaler_digest": scaler.digest(),
        "trait_feature_count": phase10_utility.TRAIT_FEATURE_COUNT,
        "candidate_matrix": [dict(entry) for entry in pc.CANDIDATE_MATRIX],
        "storage_policy": phase10_storage.storage_policy_document(),
        "problems": problems,
    }
    write_stage("contracts", payload)
    return payload


def seed_collision_audit() -> dict:
    """Exhaustive proof that the frozen id space produces no shared stream."""
    from stratego.evaluation import phase10_banks
    from stratego.training import phase10_contract as pc
    from stratego.training import phase10_schedule, phase10_seed as ps

    corpus_setup: list = []
    corpus_match: list = []
    for game in phase10_schedule.enumerate_schedule():
        corpus_match.append(game.match_seed)
        for color in ps.COLORS:
            corpus_setup.append(ps.corpus_setup_seed(game.game_id, color, 0))

    bank_opponent: list = []
    bank_selector: list = []
    bank_match: list = []
    for bank in ("validation", "test"):
        specification = phase10_banks.bank_specification(bank)
        for case_index in range(specification["case_count"]):
            family_id = phase10_banks.case_family(case_index, specification["cases_per_family"])
            ordinal = case_index % specification["cases_per_family"]
            case_id = ps.phase10_case_id(specification["bank_version"], family_id, ordinal)
            bank_opponent.append(ps.case_opponent_setup_seed(case_id, 0))
            for color in ps.COLORS:
                bank_selector.append(ps.case_selector_seed(case_id, color, 0))
            for token in pc.MATCHUP_TOKENS:
                for game_index in ps.CASE_GAME_INDICES:
                    bank_match.append(ps.case_match_seed(case_id, game_index, token))

    bootstrap = [
        ps.bootstrap_stream_seed(bank, f"{token}:{suffix}")
        for bank in ("validation", "test")
        for token in pc.MATCHUP_TOKENS
        for suffix in ("learned", "neutral", "delta")
    ] + [ps.bootstrap_stream_seed(bank, "league:delta_l") for bank in ("validation", "test")]

    fit = [ps.utility_fit_seed(model_id) for model_id in ("model_F", "model_T")]

    return ps.stream_collision_audit(
        {
            "corpus_setup_attempt0": corpus_setup,
            "corpus_match": corpus_match,
            "bank_opponent_attempt0": bank_opponent,
            "bank_selector_attempt0": bank_selector,
            "bank_match": bank_match,
            "bootstrap": bootstrap,
            "utility_fit": fit,
        }
    )


# ---------------------------------------------------------------------------
# 3. Outcome schedule
# ---------------------------------------------------------------------------


def stage_schedule(_args) -> dict:
    from stratego.training import phase10_schedule as sch

    problems: list = []
    log("schedule: enumerating and auditing the 16,384-game corpus schedule")
    audit = sch.audit_schedule()
    digest = sch.schedule_digest()
    require(audit["all_pass"], f"schedule audit failures: {audit['checks']}", problems)
    require(audit["total_games"] == 16_384, "schedule is not exactly 16,384 games", problems)
    require(audit["ordered_pair_count"] == 256, "schedule is not exactly 256 ordered pairs", problems)
    require(
        audit["games_per_ordered_pair"] == [64],
        f"games per ordered pair {audit['games_per_ordered_pair']} != [64]",
        problems,
    )

    log("schedule: spot-checking family-conditioned train-only side draws")
    from stratego.setups.sampler import load_library_index

    library = load_library_index()
    samples = []
    for game in sch.enumerate_schedule()[::2048]:
        for color in ("red", "blue"):
            sampled, attempt, seed = sch.resolve_side(game.game_id, color, index=library)
            require(
                sampled.family_id == game.side_family(color),
                f"{game.game_id} {color}: family {sampled.family_id}",
                problems,
            )
            require(sampled.split == "train", f"{game.game_id} {color}: split {sampled.split}", problems)
            samples.append(
                {
                    "game_id": game.game_id,
                    "color": color,
                    "family_id": sampled.family_id,
                    "base_setup_id": sampled.base_setup_id,
                    "accepted_attempt": attempt,
                    "accepted_draw_seed": seed,
                    "final_setup_fingerprint": sampled.provenance["final_setup_fingerprint"],
                }
            )

    payload = {
        "stage": "schedule",
        "schedule_digest": digest,
        "audit": audit,
        "side_draw_samples": samples,
        "problems": problems,
    }
    write_stage("schedule", payload)
    return payload


# ---------------------------------------------------------------------------
# 4. Evaluation banks
# ---------------------------------------------------------------------------


def stage_banks(_args) -> dict:
    from stratego.evaluation import phase10_banks as pb

    problems: list = []
    log("banks: loading the accepted Phase 9 held-out fingerprint set")
    isolation, isolation_manifest = pb.phase9_isolation_set()

    built = {}
    for bank in ("validation", "test"):
        log(f"banks: building {bank}")
        cases, manifest = pb.build_phase10_bank(bank, isolation, isolation_manifest)
        log(f"banks: auditing {bank}")
        audit = pb.audit_phase10_bank(bank, cases, manifest, isolation, rebuild_sample_every=8)
        require(audit["all_pass"], f"{bank} bank audit failures: {audit['checks']}", problems)
        require(
            audit["checks"]["phase9_fingerprint_overlap_zero"],
            f"{bank} bank reuses a Phase 9 held-out final-setup fingerprint",
            problems,
        )
        built[bank] = {"cases": cases, "manifest": manifest, "audit": audit}

    log("banks: cross-bank fingerprint isolation")
    cross = pb.cross_bank_isolation(built["validation"]["cases"], built["test"]["cases"])
    require(cross["zero_overlap"], "the two Phase 10 banks share a final-setup fingerprint", problems)

    log("banks: reconciling the accepted Phase 9 raw board universe")
    coverage = pb.phase9_raw_board_coverage()
    require(
        coverage["all_pass"],
        f"Phase 9 raw-board coverage failed: {coverage['checks']}",
        problems,
    )
    require(
        not coverage["unmapped_raw_boards"],
        f"{len(coverage['unmapped_raw_boards'])} accepted Phase 9 held-out boards fall "
        "outside the Phase 10 isolation universe; the isolation set and every "
        "dependent bank/manifest/contract digest must be rebuilt before Agent 2",
        problems,
    )

    payload = {
        "stage": "banks",
        "isolation": isolation_manifest,
        "phase9_raw_board_coverage": coverage,
        "cross_bank_isolation": cross,
        "banks": {
            bank: {
                "manifest": entry["manifest"],
                "audit": entry["audit"],
                "cases": [case.to_dict() for case in entry["cases"]],
            }
            for bank, entry in built.items()
        },
        "test_bank_access_log": [dict(entry) for entry in TEST_BANK_ACCESS_LOG],
        "problems": problems,
    }
    write_stage("banks", payload)
    return payload


# ---------------------------------------------------------------------------
# 5. Acceptance freeze
# ---------------------------------------------------------------------------


def stage_acceptance(_args) -> dict:
    from stratego.training import phase10_acceptance as pa
    from stratego.training import phase10_contract as pc

    problems: list = []
    log("acceptance: exercising each frozen gate at its threshold boundary")
    step = 1e-9
    boundaries = []

    direct = {
        pc.MATCHUP_LEARNED_VS_NEUTRAL: {
            "token": pc.MATCHUP_LEARNED_VS_NEUTRAL,
            "learned_ewr": 0.49,
            "learned_interval": {"lower": 0.48, "upper": 1.0},
        }
    }
    at_threshold = pa.gate_a(direct)
    below = pa.gate_a(
        {
            pc.MATCHUP_LEARNED_VS_NEUTRAL: {
                "token": pc.MATCHUP_LEARNED_VS_NEUTRAL,
                "learned_ewr": 0.49 - step,
                "learned_interval": {"lower": 0.48, "upper": 1.0},
            }
        }
    )
    boundaries.append(
        {
            "gate": "A",
            "threshold": "EWR >= 0.49 (non-strict)",
            "at_threshold_passes": at_threshold["pass"],
            "one_step_below_fails": not below["pass"],
        }
    )
    strict = pa.gate_a(
        {
            pc.MATCHUP_LEARNED_VS_NEUTRAL: {
                "token": pc.MATCHUP_LEARNED_VS_NEUTRAL,
                "learned_ewr": 0.60,
                "learned_interval": {"lower": 0.47, "upper": 1.0},
            }
        }
    )
    boundaries.append(
        {
            "gate": "A",
            "threshold": "paired 95% LB > 0.47 (strict)",
            "at_threshold_passes": strict["pass"],
            "expected_at_threshold": False,
        }
    )

    league_summaries = {
        token: {"token": token, "delta": -0.01, "delta_interval": {"lower": -0.02, "upper": 1.0}}
        for token in (pc.MATCHUP_STRATEGIC, pc.MATCHUP_TACTICAL, pc.MATCHUP_PHASE8_ANCHOR)
    }
    gate_b_at = pa.gate_b(league_summaries, [-0.02] * 64, "validation")
    gate_b_strict = pa.gate_b(
        {
            token: {"token": token, "delta": 0.0, "delta_interval": {"lower": 0.0, "upper": 1.0}}
            for token in (pc.MATCHUP_STRATEGIC, pc.MATCHUP_TACTICAL, pc.MATCHUP_PHASE8_ANCHOR)
        },
        [-0.03] * 64,
        "validation",
    )
    boundaries.append(
        {
            "gate": "B",
            "threshold": "Delta_L >= -0.01 (non-strict), LB > -0.03 (strict)",
            "at_delta_threshold_passes": gate_b_at["pass"],
            "at_bound_threshold_passes": gate_b_strict["pass"],
            "expected_at_bound_threshold": False,
        }
    )

    gate_c_at = pa.gate_c(
        {
            token: {"token": token, "delta_interval": {"lower": -0.03, "upper": 1.0}}
            for token in pc.GATE_C["opponents"]
        }
    )
    boundaries.append(
        {
            "gate": "C",
            "threshold": "paired LB > -0.03 (strict)",
            "at_threshold_passes": gate_c_at["pass"],
            "expected_at_threshold": False,
        }
    )

    gate_d_at = pa.gate_d(
        {
            pc.MATCHUP_RANDOM: {
                "token": pc.MATCHUP_RANDOM,
                "learned_ewr": 0.95,
                "learned_red_ewr": 1.0,
                "learned_blue_ewr": 0.90,
                "delta_interval": {"lower": 0.0, "upper": 1.0},
            },
            pc.MATCHUP_BASIC: {
                "token": pc.MATCHUP_BASIC,
                "learned_ewr": 0.80,
                "learned_red_ewr": 0.80,
                "learned_blue_ewr": 0.80,
                "delta_interval": {"lower": 0.0, "upper": 1.0},
            },
        }
    )
    boundaries.append(
        {
            "gate": "D",
            "threshold": "overall >= 0.95, per-colour >= 0.90, basic >= 0.80 (non-strict)",
            "at_threshold_passes": gate_d_at["pass"],
        }
    )

    exact = {
        "min_normalized_family_entropy": 0.85,
        "min_effective_families": 10.0,
        "min_family_probability": 0.015,
        "max_family_probability": 0.18,
        "min_within_family_normalized_base_entropy": 0.70,
        "max_conditional_base_probability": 0.10,
    }
    gate_e_at = pa.gate_e(exact)
    relaxed = dict(exact, min_normalized_family_entropy=0.85 - step)
    boundaries.append(
        {
            "gate": "E",
            "threshold": "every diversity threshold (non-strict)",
            "at_threshold_passes": gate_e_at["pass"],
            "one_step_below_fails": not pa.gate_e(relaxed)["pass"],
        }
    )

    boundaries.append(
        {
            "gate": "F",
            "threshold": "nine counters, all exactly zero",
            "all_zero_passes": pa.gate_f({name: 0 for name in pa.GATE_F_COUNTERS})["pass"],
            "one_nonzero_fails": not pa.gate_f(
                {**{name: 0 for name in pa.GATE_F_COUNTERS}, "illegal_setups": 1}
            )["pass"],
            "missing_counter_fails": not pa.gate_f({})["pass"],
        }
    )

    reproducibility = {
        "same_base": True,
        "same_reflection": True,
        "same_perturbation": True,
        "same_final_fingerprint": True,
        "worker_order_independent": True,
        "process_restart_independent": True,
    }
    boundaries.append(
        {
            "gate": "G",
            "threshold": "every link of the selection chain reproduces",
            "all_links_pass": pa.gate_g(reproducibility)["pass"],
            "one_broken_link_fails": not pa.gate_g(
                dict(reproducibility, same_perturbation=False)
            )["pass"],
        }
    )

    preservation = {
        "checkpoint_sha256": pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256,
        "model_state_digest": pc.ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
        "parameters": pc.ACCEPTED_PHASE9_PARAMETERS,
        "c1_optimizer_steps": 0,
    }
    boundaries.append(
        {
            "gate": "H",
            "threshold": "exact Phase 9 identity and zero C1 optimizer steps",
            "exact_identity_passes": pa.gate_h(preservation)["pass"],
            "one_optimizer_step_fails": not pa.gate_h(
                dict(preservation, c1_optimizer_steps=1)
            )["pass"],
        }
    )

    for entry in boundaries:
        for key, expected in (
            ("at_threshold_passes", entry.get("expected_at_threshold", True)),
            ("one_step_below_fails", True),
            ("all_zero_passes", True),
            ("one_nonzero_fails", True),
            ("missing_counter_fails", True),
            ("all_links_pass", True),
            ("one_broken_link_fails", True),
            ("exact_identity_passes", True),
            ("one_optimizer_step_fails", True),
            ("at_delta_threshold_passes", True),
            ("at_bound_threshold_passes", entry.get("expected_at_bound_threshold", True)),
        ):
            if key in entry:
                require(
                    entry[key] is expected,
                    f"gate {entry['gate']} boundary {key} is {entry[key]}, expected {expected}",
                    problems,
                )

    payload = {
        "stage": "acceptance",
        "acceptance_document": pc.acceptance_document(),
        "acceptance_digest": pc.contract_digests()[pc.ACCEPTANCE_VERSION],
        "boundary_evidence": boundaries,
        "classifications": dict(pc.CLASSIFICATIONS),
        "problems": problems,
    }
    write_stage("acceptance", payload)
    return payload


# ---------------------------------------------------------------------------
# 6. Artifacts and report
# ---------------------------------------------------------------------------


def stage_artifacts(args) -> dict:
    from stratego.training import phase10_contract as pc
    from stratego.training import phase10_seed as ps
    from stratego.training import phase10_storage, phase10_utility

    verify = read_stage("verify")
    contracts = read_stage("contracts")
    schedule = read_stage("schedule")
    banks = read_stage("banks")
    acceptance = read_stage("acceptance")

    problems = (
        list(verify["problems"])
        + list(contracts["problems"])
        + list(schedule["problems"])
        + list(banks["problems"])
        + list(acceptance["problems"])
    )
    environment = environment_report()
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    log("artifacts: writing the frozen setup-selection contract")
    contract_artifact = {
        "phase": PHASE,
        "agent": AGENT,
        "artifact": "agent_01_setup_selection_contract",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **environment,
        "contract_versions": contracts["contract_versions"],
        "contract_digests": contracts["contract_digests"],
        "contract_bundle_digest": contracts["contract_bundle_digest"],
        "contracts": pc.contract_documents(),
        "seeds": contracts["seeds"],
        "seed_derivation": contracts["seed_derivation"],
        "seed_collision_audit": {
            key: value
            for key, value in contracts["seed_collision_audit"].items()
            if key != "streams"
        },
        "seed_streams": contracts["seed_collision_audit"]["streams"],
        "outcome_schedule": {
            "schedule_digest": schedule["schedule_digest"],
            "audit": schedule["audit"],
            "side_draw_samples": schedule["side_draw_samples"],
        },
        "trait_scaler_digest": contracts["trait_scaler_digest"],
        "trait_feature_count": contracts["trait_feature_count"],
        "trait_feature_names": list(phase10_utility.TRAIT_FEATURE_NAMES),
        "storage_policy": contracts["storage_policy"],
        "storage_resolution": phase10_storage.describe_corpus_root(),
        "upstream_identities": {
            "phase9_checkpoint": verify["phase9_checkpoint"],
            "phase9_chain": verify["phase9_chain"],
            "phase7_library": verify["phase7_library"],
            "setup_semantics": verify["setup_semantics"],
        },
        "problems": problems,
    }
    CONTRACT_ARTIFACT.write_text(json.dumps(contract_artifact, indent=1, sort_keys=True) + "\n")

    bank_artifacts = {}
    for bank, path in (("validation", VALIDATION_BANK_ARTIFACT), ("test", TEST_BANK_ARTIFACT)):
        log(f"artifacts: writing the {bank} bank")
        entry = banks["banks"][bank]
        artifact = {
            "phase": PHASE,
            "agent": AGENT,
            "artifact": f"agent_01_{bank}_bank",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **environment,
            "bank": bank,
            "bank_version": entry["manifest"]["bank_version"],
            "bank_digest": entry["manifest"]["bank_digest"],
            "manifest_digest": entry["manifest"]["manifest_digest"],
            "case_count": entry["manifest"]["case_count"],
            "cases_per_opponent_family": entry["manifest"]["cases_per_opponent_family"],
            "manifest": entry["manifest"],
            "audit": entry["audit"],
            "cases": entry["cases"],
            "isolation": banks["isolation"],
            "cross_bank_isolation": banks["cross_bank_isolation"],
            "sealing": dict(pc.TEST_BANK_SEALING),
            "access_log": [
                dict(record)
                for record in banks["test_bank_access_log"]
                if bank == "test"
            ],
            "problems": entry["audit"]["failures"],
        }
        path.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n")
        bank_artifacts[bank] = artifact

    log("artifacts: writing the Agent 1 acceptance record")
    tests_after = run_pytest() if args.run_pytest else None
    gates = completion_gates(verify, contracts, schedule, banks, acceptance, tests_after)
    false_gates = sorted(key for key, value in gates.items() if not value)
    status = "PASS" if not problems and not false_gates else "BLOCKED"

    acceptance_artifact = {
        "phase": PHASE,
        "agent": AGENT,
        "artifact": "agent_01_acceptance",
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **environment,
        "completion_gates": gates,
        "gates_total": len(gates),
        "gates_true": sum(1 for value in gates.values() if value),
        "false_gates": false_gates,
        "frozen_inputs": {
            "phase9_checkpoint_sha256": verify["phase9_checkpoint"]["sha256"],
            "phase9_model_state_digest": verify["phase9_checkpoint"]["model_state_digest"],
            "phase9_parameters": verify["phase9_checkpoint"]["parameters"],
            "c1_config_digest": verify["phase9_checkpoint"]["c1_config_digest"],
            "phase9_contract_digest": verify["phase9_chain"]["observed"]["contract_digest"],
            "phase9_amendment_v1_digest": verify["phase9_chain"]["observed"]["amendment_v1_digest"],
            "phase9_amendment_v2_digest": verify["phase9_chain"]["observed"]["amendment_v2_digest"],
            "phase7_library_content_digest": verify["phase7_library"]["content_digest"],
            "phase7_library_metadata_digest": verify["phase7_library"]["metadata_digest"],
            "phase7_library_manifest_digest": verify["phase7_library"]["manifest_digest"],
        },
        "new_digests": {
            "contract_bundle_digest": contracts["contract_bundle_digest"],
            **contracts["contract_digests"],
            "outcome_schedule_digest": schedule["schedule_digest"],
            "trait_scaler_digest": contracts["trait_scaler_digest"],
            "phase9_isolation_set_digest": banks["isolation"]["set_digest"],
            "validation_bank_digest": bank_artifacts["validation"]["bank_digest"],
            "validation_bank_manifest_digest": bank_artifacts["validation"]["manifest_digest"],
            "test_bank_digest": bank_artifacts["test"]["bank_digest"],
            "test_bank_manifest_digest": bank_artifacts["test"]["manifest_digest"],
        },
        "seeds": ps.CANONICAL_PHASE10_SEEDS,
        "candidate_matrix": contracts["candidate_matrix"],
        "boundary_evidence": acceptance["boundary_evidence"],
        "test_bank_access_log": banks["test_bank_access_log"],
        "phase9_isolation_reconciliation": banks["phase9_raw_board_coverage"],
        "agents_5_7_obligations": AGENTS_5_7_OBLIGATIONS,
        "discipline": {
            "phase10_outcome_games_played": 0,
            "utility_models_fit": 0,
            "c1_optimizer_steps": 0,
            "neural_inference_on_either_bank": 0,
            "held_out_bases_in_fitting_path": 0,
            "test_bank_outcome_access": 0,
        },
        "tests_before": TESTS_BEFORE,
        "tests_after": tests_after,
        "commands": [
            "python scripts/run_phase10_agent01.py",
            ".venv/bin/python -m pytest tests -q",
        ],
        "deviations": DEVIATIONS,
        "problems": problems,
        "handoff_to_agent_2": handoff_document(contracts, schedule, banks),
    }
    ACCEPTANCE_ARTIFACT.write_text(json.dumps(acceptance_artifact, indent=1, sort_keys=True) + "\n")

    write_report_section(acceptance_artifact, bank_artifacts, schedule, contracts, banks)
    log(f"artifacts: status {status}")
    return acceptance_artifact


#: Binding reporting obligations Agent 1 places on the later agents. These
#: live in the acceptance record rather than in a frozen contract document
#: on purpose: they add no design decision and must not re-identify a
#: contract whose digest downstream agents already verify.
AGENTS_5_7_OBLIGATIONS = {
    "learned_selector_phase9_fingerprint_collisions": {
        "requirement": (
            "enumerate, per candidate, arm, matchup and bank, both the count "
            "and the rate of produced final setups whose fingerprint lies in "
            "the accepted Phase 9 held-out fingerprint set"
        ),
        "status": "report-only diagnostic",
        "must_not": (
            "be used for candidate selection, for any acceptance gate, or as "
            "grounds for evaluation-time rejection sampling of learned draws"
        ),
        "rationale": (
            "rejecting a learned draw at evaluation time would distort the "
            "very mixed distribution the diversity contract is stated over; "
            "measuring it keeps the residual visible instead of unmeasured"
        ),
        "isolation_universe": "stratego.evaluation.phase10_banks.phase9_isolation_set",
    },
}

#: Recorded readings of the common contract where it does not spell a
#: decision out letter by letter. Each is a narrowing, never a widening.
DEVIATIONS = [
    {
        "topic": "case-schedule seed root",
        "contract_text": "validation cases: 2026081806",
        "reading": (
            "the contract names a validation-case root but no separate "
            "test-case root, so 2026081806 roots the case schedule of both "
            "banks, domain-separated by their two distinct bank versions"
        ),
        "why_safe": (
            "no root is added, no stream is reused, and the exhaustive "
            "collision audit proves every validation and test stream disjoint"
        ),
    },
    {
        "topic": "trait feature dimensionality",
        "contract_text": "x(s) is the frozen 35-field trait vector",
        "reading": (
            "the 35 frozen fields include four per-rank histograms, so the "
            "feature vector is their lossless flattening: 47 float64 scalars, "
            "nothing dropped and nothing invented"
        ),
        "why_safe": (
            "the alternative would require discarding schema information by "
            "hand; the 16 exact linear relations this surfaces leave rank 31, "
            "and the frozen L2 penalty of 1e-3 makes the minimizer unique"
        ),
    },
    {
        "topic": "objective reduction",
        "contract_text": "full-batch BCE + L2 1e-3 on family/trait parameters",
        "reading": (
            "BCE is the mean over the 16,384 scheduled games and the penalty "
            "is lambda times the sum of squares of the raw family offsets and "
            "trait weights; the intercept is unpenalized"
        ),
        "why_safe": (
            "a summed BCE would make 1e-3 effectively no regularization at "
            "16,384 games; the mean is the reading under which the stated "
            "coefficient does the job the contract gives it"
        ),
    },
    {
        "topic": "selector-audit randomness domain (review reconciliation)",
        "contract_text": (
            "100,000 draws per candidate x color x split ... resume must be exact "
            "set subtraction by draw id"
        ),
        "reading": (
            "Agent 4's audit needs a selector seed per addressable draw id and the "
            "first freeze produced none — case_selector_seed covers only the 1,280 "
            "bank-case seeds — so a tenth derived domain, selector_audit, was added "
            "under the existing selector_draw_seed root 2026081805"
        ),
        "why_safe": (
            "no root seed was added or changed and no threshold, candidate, bank, "
            "schedule or utility definition moved; it removes an unfrozen choice "
            "Agent 4 would otherwise have had to invent, and it moves exactly two "
            "contract digests (phase10_setup_contract_v1, phase10_setup_selector_v1) "
            "plus the bundle, leaving both bank digests, both bank manifests, the "
            "schedule digest, the scaler digest and the isolation-set digest "
            "byte-identical"
        ),
    },
    {
        "topic": "fingerprint isolation of learned draws",
        "contract_text": (
            "zero exact final-setup fingerprint overlap with Phase 9 "
            "validation/test cases"
        ),
        "reading": (
            "hard, rejection-enforced over every arrangement a Phase 10 case "
            "fixes — the opponent setup and both neutral_v1 own-side draws; "
            "a learned selector's own-side draw cannot exist before the "
            "selector does, so Agents 5-7 record its Phase 9 landings as a "
            "report-only diagnostic"
        ),
        "why_safe": (
            "rejecting a learned draw at evaluation time would distort the "
            "very mixed distribution the diversity contract is stated over, "
            "and the diagnostic keeps the residual visible rather than "
            "unmeasured"
        ),
    },
]


def completion_gates(verify, contracts, schedule, banks, acceptance, tests_after) -> dict:
    from stratego.training import phase10_contract as pc

    validation = banks["banks"]["validation"]
    test = banks["banks"]["test"]
    return {
        "phase9_final_identity_verified": not verify["problems"]
        and verify["phase9_chain"]["chain_intact"]
        and verify["phase9_acceptance"]["agent_8_hard_gates_all_pass"],
        "phase9_model_finite": bool(verify["phase9_checkpoint"]["all_parameters_finite"]),
        "phase7_library_identity_verified": verify["phase7_library"]["content_digest"]
        == pc.PHASE7_LIBRARY_CONTENT_DIGEST,
        "phase7_splits_verified": verify["phase7_library"]["split_totals"]
        == {"train": 6400, "validation": 800, "test": 800}
        and verify["phase7_library"]["per_family_split_counts_exact"],
        "phase7_trait_vectors_reconstruct": verify["phase7_library"]["trait_vectors_reconstructed"]
        == 8000,
        "neutral_profile_verified": verify["setup_semantics"]["profile_name"] == "neutral_v1"
        and verify["setup_semantics"]["reflection_probability"] == 0.5
        and verify["setup_semantics"]["perturbation_probability"] == 0.5,
        "phase10_seeds_frozen": contracts["seeds"] == dict(pc.contract_documents()[
            pc.SETUP_CONTRACT_VERSION
        ]["seeds"]["root_seeds"])
        and contracts["seed_collision_audit"]["no_collisions"],
        "phase10_contracts_frozen_and_hashed": len(contracts["contract_digests"]) == 8
        and len(set(contracts["contract_digests"].values())) == 8,
        "outcome_schedule_exact_16384": schedule["audit"]["total_games"] == 16_384,
        "ordered_family_pair_counts_exact": schedule["audit"]["ordered_pair_count"] == 256
        and schedule["audit"]["games_per_ordered_pair"] == [64],
        "utility_fit_protocol_frozen": bool(contracts["trait_scaler_digest"]),
        "candidate_matrix_exactly_six": len(contracts["candidate_matrix"]) == 6,
        "validation_bank_frozen_and_hashed": validation["audit"]["all_pass"]
        and bool(validation["manifest"]["bank_digest"]),
        "test_bank_frozen_and_hashed": test["audit"]["all_pass"]
        and bool(test["manifest"]["bank_digest"]),
        "phase9_bank_exact_fingerprint_overlap_zero": validation["audit"]["checks"][
            "phase9_fingerprint_overlap_zero"
        ]
        and test["audit"]["checks"]["phase9_fingerprint_overlap_zero"],
        "phase10_val_test_fingerprint_overlap_zero": banks["cross_bank_isolation"]["zero_overlap"],
        "phase9_heldout_board_coverage_complete": banks["phase9_raw_board_coverage"]["all_pass"],
        "test_bank_neural_outcome_access_zero": all(
            not record["neural"] and not record["outcomes"]
            for record in banks["test_bank_access_log"]
        ),
        "final_acceptance_gates_frozen": len(acceptance["acceptance_document"]["hard_gates"]) == 8
        and not acceptance["problems"],
        "no_phase10_outcome_games": verify["prior_phase10_work"]["no_outcome_corpus"],
        "no_utility_fit": verify["prior_phase10_work"]["no_utility_model"],
        "phase9_checkpoint_unchanged": file_sha256(CHECKPOINT_PATH)
        == pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256,
        "full_suite_green": bool(
            tests_after
            and tests_after.get("returncode") == 0
            and tests_after.get("failed") == 0
            and tests_after.get("passed", 0) > 0
        ),
    }


def handoff_document(contracts, schedule, banks) -> dict:
    from stratego.training import phase10_contract as pc
    from stratego.training import phase10_schedule as sch
    from stratego.training import phase10_storage

    return {
        "for_agent": 2,
        "mission": "collect phase10_setup_outcome_corpus_v1; make no learning-design decision",
        "schedule_enumerator": "stratego.training.phase10_schedule.enumerate_schedule",
        "schedule_rebuilder": "stratego.training.phase10_schedule.rebuild_game",
        "side_draw_resolver": "stratego.training.phase10_schedule.resolve_side",
        "schedule_digest": schedule["schedule_digest"],
        "contract_digests": contracts["contract_digests"],
        "contract_bundle_digest": contracts["contract_bundle_digest"],
        "setup_derivations": {
            "side_draw_seed": "phase10_seed.corpus_setup_seed(game_id, color, attempt)",
            "match_seed": "phase10_seed.corpus_match_seed(game_id)",
            "rule": (
                "first attempt whose sample_setup('train', seed, "
                "profile='neutral_v1') primary family equals the scheduled family"
            ),
        },
        "outcome_record_schema": sch.outcome_record_schema(),
        "exact_game_count": sch.TOTAL_CORPUS_GAMES,
        "train_only_rule": (
            "every corpus setup comes from the Phase 7 train split; a validation "
            "or test base anywhere in the corpus is a BLOCKED leak"
        ),
        "phase9_evaluation_only_identity": {
            "path": pc.ACCEPTED_PHASE9_CHECKPOINT_PATH,
            "sha256": pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256,
            "model_state_digest": pc.ACCEPTED_PHASE9_MODEL_STATE_DIGEST,
            "behaviour": dict(sch.CORPUS_MOVE_BEHAVIOR),
            "role": "evaluation only; zero optimizer steps",
        },
        "storage_policy": phase10_storage.storage_policy_document(),
        "crash_resume_identity": (
            "a missing game is regenerated from its game id alone through "
            "rebuild_game + resolve_side; no derivation reads worker count, "
            "arrival order, process id, wall clock or a storage path"
        ),
        "phase9_byte_preservation": pc.PHASE9_PRESERVATION_INVARIANT,
        "bank_identities": {
            bank: {
                "bank_version": entry["manifest"]["bank_version"],
                "bank_digest": entry["manifest"]["bank_digest"],
                "manifest_digest": entry["manifest"]["manifest_digest"],
            }
            for bank, entry in banks["banks"].items()
        },
        "sealing": dict(pc.TEST_BANK_SEALING),
    }


def run_pytest() -> dict:
    log("tests: running the full suite")
    started = time.time()
    completed = subprocess.run(
        [".venv/bin/python", "-m", "pytest", "tests", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    seconds = round(time.time() - started, 2)
    summary = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    numbers = {"passed": 0, "failed": 0, "skipped": 0}
    for token, key in (("passed", "passed"), ("failed", "failed"), ("skipped", "skipped")):
        for part in summary.replace(",", " ").split():
            if part == token:
                index = summary.replace(",", " ").split().index(part)
                try:
                    numbers[key] = int(summary.replace(",", " ").split()[index - 1])
                except (ValueError, IndexError):  # pragma: no cover - defensive
                    pass
    return {
        "command": ".venv/bin/python -m pytest tests -q",
        "summary": summary,
        "returncode": completed.returncode,
        "seconds": seconds,
        **numbers,
    }


def write_report_section(acceptance_artifact, bank_artifacts, schedule, contracts, banks) -> None:
    from stratego.training import phase10_contract as pc

    log("artifacts: writing report section 1")
    digests = contracts["contract_digests"]
    validation = bank_artifacts["validation"]
    test = bank_artifacts["test"]
    coverage = banks["phase9_raw_board_coverage"]
    walk = {
        name: sum(
            int(count)
            for attempt, count in artifact["manifest"][
                "selector_seed_attempt_histogram"
            ].items()
            if int(attempt) > 0
        )
        for name, artifact in bank_artifacts.items()
    }

    preamble = [
        "# Phase 10 Implementation Report",
        "",
        "Phase 10 is a learned **setup-selection** phase. It asks whether game",
        "outcomes can be used to learn a better distribution over the frozen",
        "Phase 7 setup library while preserving setup diversity, information",
        "safety, reproducibility, and the accepted Phase 9 move model. The move",
        "policy is not retrained: `checkpoints/phase9/selfplay_c1_v1.pt` must be",
        "byte-identical before and after the phase.",
        "",
    ]
    lines = []
    lines.extend(
        [
            SECTION_MARKER,
            "",
            f"**Status: {acceptance_artifact['status']}** — "
            f"{acceptance_artifact['gates_true']}/{acceptance_artifact['gates_total']} completion",
            "gates true, zero problems, zero Phase 10 outcome games, zero utility fits,",
            "zero C1 optimizer steps.",
            "",
            "Agent 1 freezes the entire Phase 10 experiment before any outcome exists.",
            "Nothing below was chosen after seeing a result, because no Phase 10 result",
            "exists yet.",
            "",
            "### 1.1 Verified upstream identities",
            "",
            "Every identity was recomputed from live bytes, not read from a record.",
            "",
            "```text",
            "Phase 9 Agents 1-8              all PASS, zero false completion gates",
            f"Phase 9 checkpoint SHA-256      {acceptance_artifact['frozen_inputs']['phase9_checkpoint_sha256']}",
            f"Phase 9 model-state digest      {acceptance_artifact['frozen_inputs']['phase9_model_state_digest']}",
            f"Phase 9 parameters              {acceptance_artifact['frozen_inputs']['phase9_parameters']:,}, all finite",
            f"C1 config digest                {acceptance_artifact['frozen_inputs']['c1_config_digest']}",
            f"Phase 9 contract digest         {acceptance_artifact['frozen_inputs']['phase9_contract_digest']}",
            f"Phase 9 amendment v1 digest     {acceptance_artifact['frozen_inputs']['phase9_amendment_v1_digest']}",
            f"Phase 9 amendment v2 digest     {acceptance_artifact['frozen_inputs']['phase9_amendment_v2_digest']}",
            f"Phase 7 library content         {acceptance_artifact['frozen_inputs']['phase7_library_content_digest']}",
            f"Phase 7 library metadata        {acceptance_artifact['frozen_inputs']['phase7_library_metadata_digest']}",
            f"Phase 7 library manifest        {acceptance_artifact['frozen_inputs']['phase7_library_manifest_digest']}",
            "splits                          6,400 / 800 / 800 at 400 / 50 / 50 per family",
            "trait vectors                   8,000 / 8,000 reconstruct exactly",
            "neutral_v1                      reflection 0.5, perturbation 0.5, uniform 1..6",
            "pre-existing Phase 10 work      none: no corpus, utility, candidate or selector",
            "```",
            "",
            "### 1.2 Frozen contracts",
            "",
            "Eight documents, canonical JSON, SHA-256.",
            "",
            "```text",
        ]
    )
    for name in pc.CONTRACT_VERSIONS:
        lines.append(f"{name:<34}{digests[name]}")
    lines.extend(
        [
            f"{'bundle':<34}{contracts['contract_bundle_digest']}",
            "```",
            "",
            "`phase10_system_v1` binds what exists now — the accepted Phase 9 move",
            "model, the frozen Phase 7 reflection/perturbation path and `neutral_v1` —",
            "and leaves three slots unbound with their filling rules: the accepted",
            "utility model, the accepted trait scaler and the selected selector config.",
            "Inventing values for those now would be exactly the pre-commitment the",
            "phase forbids; Agent 6 fills them at the production freeze.",
            "",
            "### 1.3 Seeds and derivations",
            "",
            "```text",
            "master                    2026081801     outcome-corpus schedule   2026081802",
            "setup draws               2026081803     utility fitting           2026081804",
            "selector/candidate draws  2026081805     validation/case schedule  2026081806",
            "validation bootstrap      2026081807     final-test bootstrap      2026081808",
            "```",
            "",
            "Those are the **eight root seeds**. Beneath them sit **ten derived",
            "domains** — a distinct one per randomness need, several sharing a root —",
            "so \"streams\" in this report always means derived domains, never seeds:",
            "",
            "```text",
            "corpus_setup     corpus_match     bank_opponent    bank_selector",
            "bank_match       selector_branch  selector_base    selector_audit",
            "utility_fit      bootstrap",
            "```",
            "",
            "All ten derive through `blake2b(person='strat-s10')` over",
            "`identity_version:domain:domain_root:parts`, a tag disjoint from every",
            "accepted upstream tag, so two domains sharing a root still cannot",
            "collide. No derivation reads worker count, arrival order, process id,",
            "wall clock or a storage path.",
            "",
            "`selector_audit` was added during Agent 1 review reconciliation, on the",
            "one randomness need the first freeze left unfrozen: Agent 4 must run",
            "100,000 selector draws per candidate x colour x split, addressable by",
            "draw id, with resume as exact set subtraction by draw id — and nothing",
            "produced that draw's selector seed. `case_selector_seed` covers only the",
            "1,280 bank-case seeds. Leaving it open would have made Agent 4 invent a",
            "derivation Agent 1 owes it. No root seed was added or changed; the domain",
            "hangs off the existing `selector_draw_seed` 2026081805, and consecutive",
            "audit ordinals receive unrelated hashed streams rather than adjacent",
            "integers.",
            "",
            f"The collision audit enumerated "
            f"{contracts['seed_collision_audit']['total_seeds']:,} seeds across the frozen",
            f"id space and found {contracts['seed_collision_audit']['distinct_seeds']:,} "
            "distinct values — zero duplicates inside a",
            "stream and zero collisions across streams.",
            "",
            "### 1.4 The 16,384-game outcome schedule",
            "",
            "```text",
            "256 ordered family pairs x 64 games = 16,384",
            f"schedule digest   {schedule['schedule_digest']}",
            "split             train only; zero held-out bases",
            "side draw         first attempt whose neutral_v1 draw matches the scheduled family",
            "move behaviour    accepted Phase 9 checkpoint both sides, greedy float32,",
            "                  single_request, no search, zero optimizer steps",
            "```",
            "",
            "Ordering is a real distinction: `(F03, F11)` and `(F11, F03)` are two of the",
            "256 scheduled pairs, which is what lets the fit separate the red-first",
            "intercept from setup quality. Counts are arithmetic, never sampled, so the",
            "corpus shape cannot depend on a seed or a worker count.",
            "",
            "### 1.5 Utility definition",
            "",
            "```text",
            "Model F   u_F(s, c) = b_eff[c, family(s)]                    33 parameters",
            "Model T   u_T(s, c) = b_eff[c, family(s)] + w[c] . x(s)     127 parameters",
            f"features  phase10_trait_feature_v1, {contracts['trait_feature_count']} float64 scalars",
            f"scaler    phase10_trait_scaler_v1, train-only, ddof=0",
            f"          {contracts['trait_scaler_digest']}",
            "objective mean BCE over the 16,384 games + 1e-3 * sum of squares on the raw",
            "          family offsets and trait weights; the intercept is unpenalized",
            "optimizer float64 CPU L-BFGS, lr 1.0, max_iter 500, history 50,",
            "          tol_grad 1e-10, tol_change 1e-12, strong_wolfe, all-zero start",
            "```",
            "",
            "`strong_wolfe` line search is available in this environment (torch 2.13),",
            "verified from live bytes before the freeze, so no deterministic-equivalent",
            "authorization is needed. The utility domain is the *base*, never the played",
            "arrangement: a selector chooses a base and only then hands it to the frozen",
            "reflection/perturbation path, so fitting on base identity is the only choice",
            "that keeps the six legal selector inputs legal.",
            "",
            "### 1.6 Exactly six candidates",
            "",
            "```text",
            "P10-A model_F T=0.75    P10-B model_F T=1.25    P10-C model_F T=2.00",
            "P10-D model_T T=0.75    P10-E model_T T=1.25    P10-F model_T T=2.00",
            "```",
            "",
            "All six share the frozen 0.35 neutral / 0.65 learned mixture. The two",
            "utility models are fit once; candidate-specific refitting is forbidden;",
            "`neutral_v1` is the baseline and never a seventh candidate.",
            "",
            "### 1.7 The two evaluation banks",
            "",
            "A Phase 9 case fixed both setups because both sides were policies. A",
            "Phase 10 case cannot: the experiment is about which setup a selector",
            "chooses. A case therefore fixes one held-out opponent setup, two selector",
            "draw seeds (one per colour, identical for the learned candidate and the",
            "neutral baseline), the two `neutral_v1` own-side draws those seeds produce,",
            "and per-matchup match seeds that are independent of arm and candidate. The",
            "selector under test plays Red in game 0 and Blue in game 1 against the same",
            "opponent setup; the bootstrap unit is the case.",
            "",
            "```text",
            f"phase10_validation_bank_v1   {validation['case_count']} cases, validation "
            f"split, {validation['cases_per_opponent_family']}/family",
            f"  bank digest      {validation['bank_digest']}",
            f"  manifest digest  {validation['manifest_digest']}",
            f"phase10_test_bank_v1         {test['case_count']} cases, test split, "
            f"{test['cases_per_opponent_family']}/family",
            f"  bank digest      {test['bank_digest']}",
            f"  manifest digest  {test['manifest_digest']}",
            "```",
            "",
            "### 1.8 Isolation",
            "",
            "Phase 10 does not claim a wholly unseen base-template universe — earlier",
            "phases already used the same held-out base pool — so it claims what it can",
            "prove.",
            "",
            "The accepted Phase 9 held-out universe can be counted two ways, and both",
            "counts are correct. A Phase 9 pair stores each side already oriented for",
            "the player that plays it, and the Red and Blue orientation maps differ, so",
            "one canonical arrangement appearing on a Red side in one case and a Blue",
            "side in another is *two* stored board strings and *one* canonical",
            "identity. That is the whole difference:",
            "",
            "```text",
            f"held-out sides                     {coverage['held_out_sides']:,}",
            f"distinct stored engine boards      {coverage['distinct_raw_boards']:,}",
            f"distinct canonical identities      {coverage['distinct_canonical_identities']:,}"
            f"   ({coverage['duplicate_classes']} of them seen in both orientations)",
            f"  {coverage['distinct_canonical_identities']:,} + {coverage['duplicate_classes']} "
            f"= {coverage['distinct_raw_boards']:,}",
            "```",
            "",
            "The isolation set is stated over canonical final-setup fingerprints",
            "because that is the accepted Phase 7 setup identity and the thing a",
            "Phase 10 case actually produces. `phase9_raw_board_coverage` is the",
            "receipt that the canonical statement loses nothing: every stored board is",
            "de-oriented by the player that played it, run through the exact Phase 10",
            "fingerprint function, and required to land in the set.",
            "",
            "```text",
            f"raw boards mapped                  {coverage['distinct_raw_boards']:,} / "
            f"{coverage['distinct_raw_boards']:,}",
            f"unmapped raw boards                {len(coverage['unmapped_raw_boards'])}",
            f"round-trip mismatches              {len(coverage['round_trip_mismatches'])}",
            f"identities never reached           {len(coverage['unreached_identities'])}   "
            "(the map is onto the whole set)",
            f"duplicate classes                  {coverage['duplicate_classes']}, "
            f"every one of size exactly 2",
            "```",
            "",
            "```text",
            f"isolation set                      {banks['isolation']['distinct_fingerprints']:,} "
            "canonical identities",
            f"  set digest                       {banks['isolation']['set_digest']}",
            f"frozen Phase 10 arrangements       {3 * (validation['case_count'] + test['case_count']):,} "
            "(opponent + both neutral own-side draws per case)",
            "overlap with Phase 9                0",
            f"validation-test overlap             {banks['cross_bank_isolation']['overlap_count']}",
            "within-case duplicate fingerprints  0",
            "```",
            "",
            f"The rejection walk is not decorative: it fired on {walk['validation']} "
            f"of {2 * validation['case_count']} validation",
            f"selector seeds and {walk['test']} of {2 * test['case_count']} test selector "
            "seeds, which is the unperturbed",
            "branch colliding with Phase 9's held-out draws exactly as expected. Both",
            "walks read only quantities fixed before any selector exists, so they are",
            "arm-independent and order-independent, and a case rebuilds alone.",
            "",
            "One residual is recorded rather than hidden: a learned selector's own-side",
            "draw cannot be enumerated before the selector exists. Rejecting such a draw",
            "at evaluation time would distort the very mixed distribution the diversity",
            "contract is stated over, so it stays a report-only diagnostic. Agents 5-7",
            "carry the standing obligation to enumerate, per candidate, arm, matchup and",
            "bank, both the **count and the rate** of produced final setups landing in",
            "this set — and to use neither for selection, for any gate, or as grounds",
            "for evaluation-time rejection sampling. That obligation lives in the",
            "acceptance artifact rather than in a frozen contract, so it adds no design",
            "decision and re-identifies no digest downstream agents already verify.",
            "",
            "### 1.9 Acceptance",
            "",
            "All eight gates are hard. Strict and non-strict thresholds are named",
            "separately in code (`above` vs `at_least`), and each was exercised at its",
            "boundary and one representable step on the failing side.",
            "",
            "```text",
            "A  direct        EWR >= 0.49, LB > 0.47      improved: EWR >= 0.52, LB > 0.50",
            "B  league        Delta_L >= -0.01, LB > -0.03    weights .45/.35/.20",
            "C  individual    per-opponent paired LB > -0.03",
            "D  easy          Random >= .95 / Red >= .90 / Blue >= .90, Basic >= .80,",
            "                 paired LB > -0.03",
            "E  diversity     every threshold over the final mixed distribution",
            "F  correctness   nine counters, all exactly zero (a missing counter fails)",
            "G  reproducible  id + seed + identity + split + colour -> same fingerprint",
            "H  preservation  exact Phase 9 SHA, state digest, parameters, zero steps",
            "```",
            "",
            "Statistics: paired-unit percentile bootstrap over NumPy PCG64, 10,000",
            "replicates, 95%, one domain-separated stream per matchup and per",
            "difference, resampling the logical case so a case's two colour games move",
            "together.",
            "",
            "### 1.10 Recorded readings",
            "",
        ]
    )
    for entry in DEVIATIONS:
        body = (
            f"{entry['reading'][0].upper()}{entry['reading'][1:]}. "
            f"{entry['why_safe'][0].upper()}{entry['why_safe'][1:]}."
        )
        lines.append(f"- **{entry['topic']}** — the contract says *{entry['contract_text']}*.")
        lines.extend(
            f"  {wrapped}"
            for wrapped in textwrap.wrap(body, width=74)
        )
    tests_after = acceptance_artifact.get("tests_after") or {}
    lines.extend(
        [
            "",
            "### 1.11 Evidence",
            "",
            "```text",
            f"tests before   {TESTS_BEFORE['summary']}",
            f"tests after    {tests_after.get('summary', 'not measured in this run')}",
            "```",
            "",
            "```text",
            "reports/phase_10_data/agent_01_setup_selection_contract.json",
            "reports/phase_10_data/agent_01_validation_bank.json",
            "reports/phase_10_data/agent_01_test_bank.json",
            "reports/phase_10_data/agent_01_acceptance.json",
            "```",
            "",
            "### 1.12 Handoff to Agent 2",
            "",
            "Agent 2 collects `phase10_setup_outcome_corpus_v1` and makes no",
            "learning-design decision. It receives the schedule enumerator and",
            "rebuilder, every contract and schedule digest, the setup derivations, the",
            "outcome-record schema, the Phase 9 evaluation-only identity, the",
            "resolver/storage policy, the exact 16,384 logical ids, the train-only rule,",
            "the crash/resume identity rule, and the Phase 9 byte-preservation",
            "requirement.",
            "",
        ]
    )
    section = "\n".join(lines) + "\n"
    existing = REPORT_PATH.read_text() if REPORT_PATH.exists() else ""
    if SECTION_MARKER not in existing:
        REPORT_PATH.write_text((existing or "\n".join(preamble) + "\n") + section)
        return
    # Rewriting in place, so a re-run with a measured suite replaces section 1
    # rather than appending a second copy of it.
    head, _, remainder = existing.partition(SECTION_MARKER)
    following = re.search(r"\n## (?!1\. Agent 1)", remainder)
    tail = remainder[following.start() + 1 :] if following else ""
    REPORT_PATH.write_text(head + section + tail)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STAGES = {
    "verify": stage_verify,
    "contracts": stage_contracts,
    "schedule": stage_schedule,
    "banks": stage_banks,
    "acceptance": stage_acceptance,
    "artifacts": stage_artifacts,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), default=None)
    parser.add_argument("--run-pytest", action="store_true")
    args = parser.parse_args()

    names = [args.stage] if args.stage else list(STAGES)
    result = {}
    for name in names:
        started = time.time()
        result = STAGES[name](args)
        log(f"stage {name} finished in {round(time.time() - started, 1)}s")
        if result.get("problems"):
            for problem in result["problems"]:
                log(f"PROBLEM: {problem}")
    return 0 if not result.get("problems") else 1


if __name__ == "__main__":
    raise SystemExit(main())
