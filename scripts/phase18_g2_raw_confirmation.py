#!/usr/bin/env python3
"""Phase 18 Gate G2, bounded raw-actor confirmation (Agent 5).

The frozen primary question:

    Using the parity-correct setup-learning method, does the raw generation
    actor reliably learn an independently generated synthetic setup landscape
    within 64 setup updates?

The raw actor is the primary endpoint of THIS synthetic trainability assay
only. The EMA remains the required evaluation/deployment model for every later
Stratego-facing stage (S28) and is recorded here as secondary mechanism
telemetry that cannot change the decision.

The learning method is the verified G2 implementation, byte for byte. This
driver imports `run_seed`, `AssayDesign` and the landscape builder unchanged,
freezes a design whose every method field equals G2's, and refuses to verify or
launch if any learning-method file's digest differs from the one the G2 launch
manifest bound at G2_SOURCE_COMMIT 354a4cad.

Every seed derives deterministically from the reviewed base commit (the
published G2 branch head), the fixed artifact namespace and a domain-separated
label through the recorded `derive_stream_seed`:

    seed_namespace = "phase18_g2_raw_confirmation_v1:<reviewed base commit>"
    seed           = derive_stream_seed(seed_namespace, label, *parts)

Stages, in order; each must finish before the next may start:

* `--freeze`            derive the seeds, build the new landscape, certify its
                        optimum by LP duality and re-verify the certificate from
                        the recorded potentials, audit freshness against G2 and
                        the development smokes, write the contract and the
                        landscape document (no training)
* `--verify`            preflight: learning-method digests against the G2 launch
                        manifest, the evaluator and setup suites with JUnit, the
                        canned parity oracle, the S01-S30 coverage table
* `--launch-manifest`   from the clean detached execution worktree at the frozen
                        source commit; before any outcome exists
* `--run --seed-index K` one frozen seed into the git-ignored artifact root
* `--replay`            landscape, initial models, first-period outcomes, every
                        period digest and all four evaluation endpoints per seed
* `--analyse`           the three seeds -> results and the frozen raw criteria
                        (EMA criteria recorded as telemetry)
* `--bind`              the binding ledger over every confirmation artifact
* `--decide`            the frozen decision rule over the results and the ledger

Nothing here opens a Stratego game or a sealed Phase 8 example: the assay's
only environment is the synthetic landscape, and a source-scan test pins that
this driver imports no game runner, corpus reader or evaluation bank.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.training.phase18 import PHASE18_SETUP_PACKAGE_VERSION  # noqa: E402
from stratego.training.phase18.setup_contract import (  # noqa: E402
    ENTROPY_NORMALIZER,
    METHOD_MAP,
    PAPER_ID,
    PAPER_SHA256,
    PUBLISHED_SOURCE,
    PUBLISHED_SOURCE_COMMIT,
    SETUP_EMA_DECAY,
    WORK_PACKAGE,
    file_sha256,
    json_document_digest,
    stream_seed,
)
from stratego.training.phase18.synthetic_assay import ASSAY_VERSION, AssayDesign, run_seed  # noqa: E402
from stratego.training.phase18.synthetic_landscape import (  # noqa: E402
    SLOT_TYPES,
    build_landscape,
    build_table,
    exact_optimum,
    hungarian_minimum,
    landscape_from_document,
    uniform_moments,
    utility,
)

RUN_ID = "G2-RAW-CONFIRMATION-2026-A"
GATE = "G2"
AGENT = "phase_18_agent_5"

#: The reviewed G2 branch head (published at origin/phase18/g2-setup-parity);
#: the branch point of this work and one of the seed-derivation inputs.
BASE_COMMIT = "6afa13bed355884a3327d2661fd739784260dc2b"
ARTIFACT_NAMESPACE = "phase18_g2_raw_confirmation_v1"
SEED_NAMESPACE = f"{ARTIFACT_NAMESPACE}:{BASE_COMMIT}"

#: The canonical tree the artifacts live in, whatever tree the code runs from.
CANONICAL_ROOT = Path("/Users/brandonwashington/Dev/Github/stratego/gpt_agent")
ARTIFACT_RELATIVE = Path("artifacts/phase18/g2_raw_confirmation_v1")
ARTIFACT_ROOT = CANONICAL_ROOT / ARTIFACT_RELATIVE
EXECUTION_WORKTREE = Path("/Users/brandonwashington/Dev/Github/stratego/gpt_agent_phase18_g2_raw_exec")

DIRECTORY = "g2_raw_confirmation"
CONTRACT_NAME = "phase18_g2_raw_confirmation_contract_v1.json"
LANDSCAPE_NAME = "phase18_g2_raw_confirmation_landscape_v1.json"
LAUNCH_NAME = "phase18_g2_raw_confirmation_launch_manifest_v1.json"
RESULTS_NAME = "phase18_g2_raw_confirmation_results_v1.json"
BINDING_NAME = "phase18_g2_raw_confirmation_binding_v1.json"
DECISION_INPUT_NAME = "phase18_g2_raw_confirmation_decision_input_v1.json"
VERIFICATION_NAME = "phase18_g2_raw_confirmation_verification_v1.json"
COVERAGE_NAME = "phase18_g2_raw_confirmation_parity_coverage_v1.json"
ORACLE_NAME = "phase18_g2_raw_confirmation_parity_oracle_v1.json"
REPLAY_NAME = "phase18_g2_raw_confirmation_replay_v1.json"

#: The G2 artifacts this confirmation is measured against (read only).
G2_CONTRACT_NAME = "phase18_g2_contract_v1.json"
G2_LANDSCAPE_NAME = "phase18_g2_synthetic_landscape_v1.json"
G2_LAUNCH_NAME = "phase18_g2_launch_manifest_v1.json"
G2_DEV_SMOKE_NAME = "g2/dev_smoke_v1.json"
G2_DRIVER = "scripts/phase18_g2_setup_parity.py"

AUTHORIZATION_FILES = (
    "reports/phase18/decisions/P18-D004.json",
    "reports/phase18/decisions/P18-D004.md",
    "reports/phase18/reviews/P18-D004_REVIEW.md",
    "instructions/phase_18_setup_integrated_warmstart/07_AGENT_5_G2_RAW_ACTOR_CONFIRMATION.md",
    "reports/phase18/ataraxos_setup_method_map_v2.json",
)

#: The learning method: every file the G2 launch manifest digested. They must
#: be byte-identical to the G2 launch manifest's digests for this confirmation
#: to freeze, verify or launch.
METHOD_FILES = (
    "scripts/phase18_g2_setup_parity.py",
    "stratego/training/phase18/__init__.py",
    "stratego/training/phase18/setup_contract.py",
    "stratego/training/phase18/setup_model.py",
    "stratego/training/phase18/setup_sampling.py",
    "stratego/training/phase18/setup_buffer.py",
    "stratego/training/phase18/setup_learning.py",
    "stratego/training/phase18/reference_oracle.py",
    "stratego/training/phase18/synthetic_landscape.py",
    "stratego/training/phase18/synthetic_assay.py",
    "stratego/training/phase18/coverage.py",
    "stratego/setups/identity.py",
    "stratego/belief/phase15/orientation.py",
    "stratego/engine/setup.py",
    "stratego/engine/constants.py",
    "stratego/evaluation/phase18/noninferiority.py",
)
SOURCE_FILES = METHOD_FILES + ("scripts/phase18_g2_raw_confirmation.py",)
TEST_FILES = (
    "tests/training/phase18/conftest.py",
    "tests/training/phase18/test_setup_model.py",
    "tests/training/phase18/test_setup_sampling.py",
    "tests/training/phase18/test_setup_buffer.py",
    "tests/training/phase18/test_setup_learning.py",
    "tests/training/phase18/test_reference_oracle.py",
    "tests/training/phase18/test_synthetic_landscape.py",
    "tests/training/phase18/test_synthetic_assay.py",
    "tests/training/phase18/test_coverage.py",
    "tests/training/phase18/test_g2_driver.py",
    "tests/training/phase18/test_g2_raw_confirmation_driver.py",
)

EVALUATOR_SUITE = "tests/evaluation/phase18"
SETUP_SUITE = "tests/training/phase18"

#: The design fields that legitimately differ from G2 (identity, not method).
DESIGN_FIELDS_EXPECTED_TO_DIFFER = frozenset({"namespace", "run_id", "model_seeds", "landscape_table_seed", "bootstrap_seed", "training_config_digest"})

ENDPOINT_ARRAYS = ("initial", "final", "initial_raw", "final_raw")
INTEGRITY_FAILURE_KEYS = ("legality_failures", "orientation_failures", "attribution_failures", "non_finite_events", "checkpoint_identity_failures")
SYMBOL = {-1: "-", 0: "0", 1: "+"}

QUESTION_TEXT = (
    "Using the parity-correct setup-learning method, does the raw generation actor reliably learn an "
    "independently generated synthetic setup landscape within 64 setup updates?"
)


class G2RawError(RuntimeError):
    """A frozen identity, accounting or sealing precondition failed."""


def log(message: str) -> None:
    print(f"[g2-raw {time.strftime('%H:%M:%S')}] {message}", flush=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    return path


def git_output(*arguments: str, cwd: Path = REPOSITORY_ROOT) -> str:
    return subprocess.run(["git", *arguments], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def environment() -> dict:
    import torch

    porcelain = git_output("status", "--porcelain")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "executing_tree": str(REPOSITORY_ROOT),
        "source_commit": git_output("rev-parse", "HEAD"),
        "source_tree": git_output("rev-parse", "HEAD^{tree}"),
        "working_tree_state": "clean" if not porcelain else f"dirty ({len(porcelain.splitlines())} paths)",
        "dirty_paths": porcelain.splitlines()[:40],
    }


def digests_of(names, root: Path = REPOSITORY_ROOT) -> dict:
    record = {}
    for name in names:
        path = root / name
        if not path.exists():
            raise G2RawError(f"BLOCKED: {name} is missing from {root}")
        record[name] = file_sha256(path)
    return record


def load_g2_driver():
    """The verified G2 driver, imported by path: its pytest runner, JUnit parser
    and canned parity oracle are reused unchanged."""
    spec = importlib.util.spec_from_file_location("phase18_g2_setup_parity", REPOSITORY_ROOT / G2_DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The frozen design and its seeds
# ---------------------------------------------------------------------------


def frozen_design() -> AssayDesign:
    """The one frozen design: every method field is the G2 default (the
    instruction's minimum or the published value); only the identity differs."""
    return AssayDesign(namespace=SEED_NAMESPACE, run_id=RUN_ID)


def bootstrap_seeds(design: AssayDesign) -> dict:
    """Domain-separated bootstrap seeds: the design's own bootstrap seed is the
    primary (raw, pooled) draw; every other draw carries its own label."""
    ns = design.namespace
    return {
        "raw_pooled": design.bootstrap_seed(),
        "raw_per_seed": {str(k): stream_seed(ns, "paired_bootstrap", "raw", "seed", k) for k in design.seed_indices},
        "ema_pooled": stream_seed(ns, "paired_bootstrap", "ema", "pooled"),
        "ema_per_seed": {str(k): stream_seed(ns, "paired_bootstrap", "ema", "seed", k) for k in design.seed_indices},
    }


def seed_derivation(design: AssayDesign) -> dict:
    ns = design.namespace
    return {
        "artifact_namespace": ARTIFACT_NAMESPACE,
        "seed_namespace": ns,
        "base_commit": BASE_COMMIT,
        "base_commit_role": "the reviewed G2 branch head, published at origin/phase18/g2-setup-parity; the branch point of this work and the second seed-derivation input",
        "rule": "seed = derive_stream_seed(seed_namespace, label, *parts) with seed_namespace = artifact_namespace + ':' + base_commit; every stream carries a distinct label",
        "seed_function": "stratego.setups.identity.derive_stream_seed",
        "model_seeds": {str(k): design.model_seed(k) for k in design.seed_indices},
        "landscape_table_seed": design.landscape_table_seed(),
        "bootstrap_seeds": bootstrap_seeds(design),
        "first_pool_root_seed_seed1_snapshot0_index0": stream_seed(ns, "pool", 1, 0, 0),
        "first_evaluation_root_seed_seed1_index0": stream_seed(ns, "eval", 1, 0, 0),
        "first_evaluation_reflection_seed_seed1_index0": stream_seed(ns, "eval_reflection", 1, 0, 0),
        "labels": {
            "model_init": "derive_stream_seed(seed_namespace, 'model_init', k)",
            "pool_tokens": "derive_stream_seed(seed_namespace, 'pool', k, snapshot, index) -> per-prefix derive_stream_seed('phase18_setup_token', root, prefix)",
            "pool_reflection": "derive_stream_seed(seed_namespace, 'reflection', k, snapshot, index)",
            "outcomes": "derive_stream_seed(seed_namespace, 'outcome', k, period, content_fingerprint, replicate)",
            "shuffle": "derive_stream_seed(seed_namespace, 'shuffle', k, update, epoch)",
            "evaluation_tokens": "derive_stream_seed(seed_namespace, 'eval', k, 0, index); snapshot fixed at 0 so every endpoint (raw and EMA, initial and final) shares the same uniforms",
            "evaluation_reflection": "derive_stream_seed(seed_namespace, 'eval_reflection', k, 0, index)",
            "landscape_table": "derive_stream_seed(seed_namespace, 'landscape_table')",
            "paired_bootstrap_raw_pooled": "derive_stream_seed(seed_namespace, 'paired_bootstrap')",
            "paired_bootstrap_raw_per_seed": "derive_stream_seed(seed_namespace, 'paired_bootstrap', 'raw', 'seed', k)",
            "paired_bootstrap_ema_pooled": "derive_stream_seed(seed_namespace, 'paired_bootstrap', 'ema', 'pooled')",
            "paired_bootstrap_ema_per_seed": "derive_stream_seed(seed_namespace, 'paired_bootstrap', 'ema', 'seed', k)",
        },
    }


def build_frozen_landscape(design: AssayDesign):
    return build_landscape(
        namespace=design.namespace,
        table_seed=design.landscape_table_seed(),
        kappa=design.landscape_kappa,
        p_draw=design.landscape_p_draw,
    )


# ---------------------------------------------------------------------------
# The optimum certificate, re-verified from recorded potentials
# ---------------------------------------------------------------------------


def certificate_with_potentials(table) -> dict:
    """Solve the assignment once more and keep the dual potentials so the
    certificate can be re-verified by arithmetic alone."""
    a = np.stack([table[t] for t in SLOT_TYPES])
    assignment, u, v = hungarian_minimum((-a).tolist())
    board = [None] * len(SLOT_TYPES)
    for slot, square in enumerate(assignment):
        board[square] = SLOT_TYPES[slot]
    return {"assignment": [int(s) for s in assignment], "u": [float(x) for x in u], "v": [float(x) for x in v], "optimal_setup": [int(t) for t in board]}


def verify_certificate(table, u, v, optimal_setup, *, tolerance: float = 1e-9) -> dict:
    """LP-duality optimality from recorded potentials, without any solver.

    For cost = -T[type(slot), square]: dual feasibility `u_i + v_j <= cost_ij`
    for every (slot, square) makes `-(sum u + sum v)` an upper bound on the
    utility of every legal setup (weak duality over the assignment polytope),
    and a legal setup whose utility equals that bound is optimal.
    """
    from stratego.engine.constants import PIECE_COUNTS
    from stratego.engine.setup import validate_setup

    table = np.asarray(table, dtype=np.float64)
    a = np.stack([table[t] for t in SLOT_TYPES])
    cost = -a
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != (len(SLOT_TYPES),) or v.shape != (len(SLOT_TYPES),):
        raise G2RawError("the certificate potentials have the wrong shape")
    slack = cost - u[:, None] - v[None, :]
    violations = int((slack < -tolerance).sum())
    worst = float(slack.min())
    upper_bound = float(-(u.sum() + v.sum()))
    canonical = tuple(int(t) for t in optimal_setup)
    validate_setup(canonical, 0)
    counts = {int(t): canonical.count(t) for t in set(canonical)}
    inventory_ok = all(counts.get(t, 0) == PIECE_COUNTS[t] for t in range(len(PIECE_COUNTS)))
    primal = float(sum(table[piece, square] for square, piece in enumerate(canonical)))
    gap = abs(upper_bound - primal)
    return {
        "method": "recorded dual potentials: u_i + v_j <= cost_ij for all (slot, square); -(sum u + sum v) is an upper bound on every legal setup's utility; the recorded setup attains it",
        "dual_feasibility_violations": violations,
        "worst_slack": worst,
        "utility_upper_bound_from_potentials": upper_bound,
        "optimal_setup_utility_by_direct_summation": primal,
        "gap": gap,
        "optimal_setup_legal": True,
        "optimal_setup_inventory_ok": bool(inventory_ok),
        "certified": bool(violations == 0 and inventory_ok and gap <= 1e-6),
    }


# ---------------------------------------------------------------------------
# Method identity against G2 and freshness against every previous table
# ---------------------------------------------------------------------------


def method_identity(references: Path) -> dict:
    """The learning method is unchanged: every method file's digest equals the
    one the G2 launch manifest bound, the design equals G2's on every method
    field, and the training configuration equals G2's once the run id is
    removed."""
    g2_launch = json.loads((references / G2_LAUNCH_NAME).read_text())
    g2_contract = json.loads((references / G2_CONTRACT_NAME).read_text())
    files = {}
    for name in METHOD_FILES:
        g2_digest = g2_launch["source_digests"].get(name)
        path = REPOSITORY_ROOT / name
        current = file_sha256(path) if path.exists() else None
        files[name] = {"g2_digest": g2_digest, "current_digest": current, "identical": bool(g2_digest) and g2_digest == current}
    design = frozen_design()
    g2_design = AssayDesign(namespace=g2_contract["design"]["namespace"], run_id=g2_contract["run_id"])
    g2_document = g2_design.document()
    mine = {k: v for k, v in design.document().items() if k not in DESIGN_FIELDS_EXPECTED_TO_DIFFER}
    theirs = {k: v for k, v in g2_document.items() if k not in DESIGN_FIELDS_EXPECTED_TO_DIFFER}
    differing = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
    my_config = design.training_config().document()
    their_config = g2_design.training_config().document()
    my_config.pop("run_id")
    their_config.pop("run_id")
    return {
        "g2_source_commit": g2_launch["source"]["g2_source_commit"],
        "g2_launch_manifest_sha256": file_sha256(references / G2_LAUNCH_NAME),
        "g2_contract_sha256": file_sha256(references / G2_CONTRACT_NAME),
        "g2_design_rederives_from_code": json_document_digest(g2_document) == json_document_digest(g2_contract["design"]),
        "files": files,
        "all_method_files_identical_to_g2": all(f["identical"] for f in files.values()),
        "design_fields_compared": sorted(mine),
        "design_fields_expected_to_differ": sorted(DESIGN_FIELDS_EXPECTED_TO_DIFFER),
        "design_fields_differing": differing,
        "design_identical_on_every_method_field": not differing,
        "training_config_digest_this_run": design.training_config().config_digest(),
        "g2_training_config_digest": g2_design.training_config().config_digest(),
        "method_config_digest_run_id_removed": json_document_digest(my_config),
        "g2_method_config_digest_run_id_removed": json_document_digest(their_config),
        "method_config_identical": my_config == their_config,
        "ema_decay": SETUP_EMA_DECAY,
        "ema_retained_initial_fraction_after_budget": SETUP_EMA_DECAY ** design.updates,
        "ema_time_constant_updates": 1.0 / (1.0 - SETUP_EMA_DECAY),
    }


def method_unchanged(identity: dict) -> bool:
    return bool(identity["all_method_files_identical_to_g2"] and identity["design_identical_on_every_method_field"] and identity["method_config_identical"] and identity["g2_design_rederives_from_code"])


def freshness_audit(design: AssayDesign, landscape_document: dict, references: Path) -> dict:
    """Every seed and the table differ from G2's and from both development
    smokes'; the three model seeds are pairwise distinct."""
    g2_contract = json.loads((references / G2_CONTRACT_NAME).read_text())
    g2_landscape = json.loads((references / G2_LANDSCAPE_NAME).read_text())
    smokes = json.loads((references / G2_DEV_SMOKE_NAME).read_text())
    previous = {}
    g2_ns = g2_contract["design"]["namespace"]
    previous[g2_ns] = {
        "role": "the G2 frozen assay (P18-D004)",
        "table_seed": int(g2_contract["design"]["landscape_table_seed"]),
        "table_digest": g2_landscape["table_digest"],
        "model_seeds": {k: int(v) for k, v in g2_contract["design"]["model_seeds"].items()},
        "bootstrap_seed": int(g2_contract["design"]["bootstrap_seed"]),
        "exact_optimum": g2_landscape["exact_optimum"]["optimum"],
        "uniform_mean": g2_landscape["uniform_baseline"]["mean"],
        "first_pool_root_seed_seed1_snapshot0_index0": stream_seed(g2_ns, "pool", 1, 0, 0),
        "first_evaluation_root_seed_seed1_index0": stream_seed(g2_ns, "eval", 1, 0, 0),
    }
    for smoke in smokes["smokes"]:
        ns = smoke["namespace"]
        smoke_design = AssayDesign(namespace=ns)
        table = build_table(smoke_design.landscape_table_seed())
        moments = uniform_moments(table)
        optimum = exact_optimum(table)
        match = re.search(r"uniform mean (\S+) sd (\S+) optimum (\S+)", smoke.get("log_text", ""))
        recorded = [float(x) for x in match.groups()] if match else None
        rebuilt = [round(moments["mean"], 4), round(moments["sd"], 4), round(optimum["optimum"], 4)]
        previous[ns] = {
            "role": f"development smoke {smoke.get('label')} ({smoke.get('run_id')})",
            "table_seed": smoke_design.landscape_table_seed(),
            "table_digest": json_document_digest([[float(x) for x in row] for row in table]),
            "model_seeds": {str(k): smoke_design.model_seed(k) for k in smoke_design.seed_indices},
            "bootstrap_seed": smoke_design.bootstrap_seed(),
            "exact_optimum": optimum["optimum"],
            "uniform_mean": moments["mean"],
            "rebuilt_mean_sd_optimum": rebuilt,
            "recorded_mean_sd_optimum_in_smoke_log": recorded,
            "rebuild_matches_recorded_log": recorded is not None and all(abs(a - b) < 5e-5 for a, b in zip(rebuilt, recorded)),
            "first_pool_root_seed_seed1_snapshot0_index0": stream_seed(ns, "pool", 1, 0, 0),
            "first_evaluation_root_seed_seed1_index0": stream_seed(ns, "eval", 1, 0, 0),
        }
    mine = seed_derivation(design)
    my_model_seeds = set(mine["model_seeds"].values())
    previous_model_seeds = {s for p in previous.values() for s in p["model_seeds"].values()}
    checks = {
        "table_seed_new": all(mine["landscape_table_seed"] != p["table_seed"] for p in previous.values()),
        "table_digest_new": all(landscape_document["table_digest"] != p["table_digest"] for p in previous.values()),
        "model_seeds_pairwise_distinct": len(my_model_seeds) == len(design.seed_indices),
        "model_seeds_disjoint_from_previous": not (my_model_seeds & previous_model_seeds),
        "bootstrap_seed_new": all(mine["bootstrap_seeds"]["raw_pooled"] != p["bootstrap_seed"] for p in previous.values()),
        "bootstrap_seeds_pairwise_distinct": len({mine["bootstrap_seeds"]["raw_pooled"], mine["bootstrap_seeds"]["ema_pooled"], *mine["bootstrap_seeds"]["raw_per_seed"].values(), *mine["bootstrap_seeds"]["ema_per_seed"].values()}) == 2 + 2 * len(design.seed_indices),
        "first_pool_root_seed_new": all(mine["first_pool_root_seed_seed1_snapshot0_index0"] != p["first_pool_root_seed_seed1_snapshot0_index0"] for p in previous.values()),
        "first_evaluation_root_seed_new": all(mine["first_evaluation_root_seed_seed1_index0"] != p["first_evaluation_root_seed_seed1_index0"] for p in previous.values()),
        "seed_namespace_new": all(design.namespace != ns for ns in previous),
    }
    return {
        "rule": "no table, model seed, pool seed, outcome seed, evaluation seed or bootstrap seed may coincide with the G2 frozen assay's or either development smoke's; all derive from the new seed namespace, so freshness is by construction and is verified here explicitly",
        "previous_namespaces": previous,
        "this_run": {"seed_namespace": design.namespace, "table_seed": mine["landscape_table_seed"], "table_digest": landscape_document["table_digest"], "model_seeds": mine["model_seeds"], "bootstrap_seeds": mine["bootstrap_seeds"], "exact_optimum": landscape_document["exact_optimum"]["optimum"], "uniform_mean": landscape_document["uniform_baseline"]["mean"]},
        "checks": checks,
        "fresh": all(checks.values()),
        "selection_rule": "the table and every seed are the first derivation from the fixed namespace and the reviewed base commit; nothing was drawn twice or chosen by inspection",
    }


# ---------------------------------------------------------------------------
# Stage 1: freeze
# ---------------------------------------------------------------------------


def stage_freeze(reports: Path, *, references: Path | None = None) -> dict:
    references = references or (CANONICAL_ROOT / "reports" / "phase18")
    started = time.perf_counter()
    design = frozen_design()
    if design.reduced:
        raise G2RawError("BLOCKED: the frozen design must not be reduced")
    identity = method_identity(references)
    if not method_unchanged(identity):
        raise G2RawError("BLOCKED: the learning method differs from the verified G2 implementation")
    landscape = build_frozen_landscape(design)
    landscape_document = landscape.document()
    if not landscape_document["exact_optimum"]["certificate"]["certified"]:
        raise G2RawError("BLOCKED: the exact optimum could not be certified")
    potentials = certificate_with_potentials(landscape.table)
    if potentials["optimal_setup"] != landscape_document["exact_optimum"]["optimal_setup"]:
        raise G2RawError("BLOCKED: the recorded optimal setup does not match the certified one")
    independent = verify_certificate(landscape.table, potentials["u"], potentials["v"], potentials["optimal_setup"])
    if not independent["certified"] or abs(independent["optimal_setup_utility_by_direct_summation"] - landscape.optimum) > 1e-9:
        raise G2RawError("BLOCKED: the independent certificate check failed")
    freshness = freshness_audit(design, landscape_document, references)
    if not freshness["fresh"]:
        raise G2RawError(f"BLOCKED: the landscape or a seed is not fresh: {freshness['checks']}")
    seeds = seed_derivation(design)

    contract = {
        "artifact": "phase18_g2_raw_confirmation_contract_v1",
        "work_package": WORK_PACKAGE,
        "agent": AGENT,
        "gate": GATE,
        "run_id": RUN_ID,
        "package_version": PHASE18_SETUP_PACKAGE_VERSION,
        "assay_version": ASSAY_VERSION,
        "timestamp_utc": utc_now(),
        "authorizing_decision": "P18-D004 accepted as REVISE; operator work package 07 (2026-09-02)",
        "authorization": digests_of(AUTHORIZATION_FILES, references.parent.parent),
        "references": {
            "paper": PAPER_ID,
            "paper_sha256": PAPER_SHA256,
            "published_source": PUBLISHED_SOURCE,
            "published_source_commit": PUBLISHED_SOURCE_COMMIT,
            "method_map": METHOD_MAP,
            "g2_contract": G2_CONTRACT_NAME,
            "g2_landscape": G2_LANDSCAPE_NAME,
            "g2_launch_manifest": G2_LAUNCH_NAME,
            "note": "the paper and the published source are technical references; the Phase 18 contracts, the accepted P18-D004 and instruction 07 govern",
        },
        "question": {
            "text": QUESTION_TEXT,
            "primary_endpoint": "the RAW generation actor's held-out expected landscape utility at 0 updates and after update 64; this is the primary endpoint for this synthetic trainability assay only",
            "ema_role": "secondary mechanism telemetry recorded at the same endpoints on the same evaluation stream with the same paired bootstrap; EMA results cannot change the confirmation decision. The EMA remains the required evaluation/deployment model for every later Stratego-facing stage (S28).",
            "null_hypothesis": "the parity-correct raw learner does not reliably learn the landscape: for at least one seed U_final(raw) <= U_initial(raw), or the pooled paired 95% lower bound is <= 0, or the median seed closes less than 10% of its initial-to-optimum gap",
            "alternative_hypothesis": "for every seed U_final(raw) > U_initial(raw), the pooled paired 95% lower bound is strictly above zero, and the median seed closes at least 10% of its gap",
            "primary_metric": "expected landscape utility: the mean utility U(s) over the 4,096 held-out raw-actor samples per seed per endpoint, initial (0 updates) versus final (after update 64)",
            "practical_margin": {"median_gap_closure_fraction": design.gap_closure_threshold, "definition": "(U_final - U_initial) / (U_optimum - U_initial) per seed on the raw actor, median over the three seeds, must be >= 0.10"},
            "uncertainty": {
                "unit": "one held-out evaluation sample: the initial and final raw samples share the same per-token uniforms (common random numbers), so the per-sample difference U_final[i] - U_initial[i] is paired",
                "pooled_statistic": "mean paired difference over all seeds' samples (3 x 4,096 = 12,288 pairs)",
                "method": "two-sided 95% paired percentile bootstrap over the pooled paired differences, 10,000 replicates, frozen seed; the lower endpoint must be strictly greater than zero",
                "implementation": "stratego.evaluation.phase18.noninferiority.paired_unit_delta (the accepted paired bootstrap; one shared index draw for both endpoints)",
                "bootstrap_seeds": seeds["bootstrap_seeds"],
                "per_seed_intervals": "reported as diagnostics with their own domain-separated seeds; they do not decide the confirmation",
            },
            "checkpoint_rule": "the raw actor after the final fixed update (update 64) decides; intermediate curve points are telemetry and never select a checkpoint",
            "evaluation_rule": {
                "samples_per_endpoint": design.evaluation_samples,
                "common_random_numbers": "every endpoint (raw and EMA, initial and final) draws the same 4,096 per-token uniforms from the 'eval' stream at snapshot 0",
                "immediately_terminal_setups": "excluded from play and training under the existing rule (S24: a setup without an opening move is never trained and never a draw); the held-out evaluation scores every generated sample exactly as in G2 (the landscape utility is defined for every legal setup) and records the immediately-terminal count among the evaluation samples at every endpoint",
                "sample_count_integrity": "each of the four held-out utility arrays per seed (initial/final x raw/EMA) must hold exactly 4,096 finite values",
            },
            "sample_size_basis": "the G2 instruction minimum, unchanged: 4,096 held-out samples per endpoint per seed, three seeds, 10,000 replicates",
            "frozen_before_outcomes": True,
        },
        "design": design.document(),
        "seed_derivation": seeds,
        "freshness_audit": freshness,
        "method_identity": identity,
        "landscape": {
            "version": landscape_document["landscape_version"],
            "family": "the G2 family and methodology: a reflection-symmetric additive piece-type-by-square table drawn from a seeded standard normal (left five files drawn, mirrored), exact uniform-random moments by Hoeffding, W/D/L outcomes by kappa-scaled sigmoid with a fixed draw share",
            "table_seed": landscape_document["table_seed"],
            "table_digest": landscape_document["table_digest"],
            "reflection_invariant": landscape_document["reflection_invariant"],
            "uniform_baseline": landscape_document["uniform_baseline"],
            "exact_optimum": landscape_document["exact_optimum"]["optimum"],
            "optimal_setup": landscape_document["exact_optimum"]["optimal_setup"],
            "optimum_certified": landscape_document["exact_optimum"]["certificate"]["certified"],
            "solver_certificate": landscape_document["exact_optimum"]["certificate"],
            "certificate_potentials": {"u": potentials["u"], "v": potentials["v"], "assignment": potentials["assignment"]},
            "independent_certificate_check": independent,
            "outcome_mapping": landscape_document["outcome_mapping"],
            "outcomes_per_eligible_setup": design.outcomes_per_setup,
            "learner_interface": landscape_document["learner_interface"],
            "document": LANDSCAPE_NAME,
            "fresh_policy_baseline": "estimated separately by the assay from each seed's initial raw sample (4,096 draws) and recorded per seed; the exact uniform-random baseline above is the model-free reference",
        },
        "parity_requirements": {
            "method_files": "every learning-method file byte-identical to the G2 launch manifest's digest (method_identity.files)",
            "rows": "S01-S30 of the method map, each mapped to implementation symbols and tests in stratego/training/phase18/coverage.py; a row is complete only when every cited test passed in the recorded --verify run",
            "canned_oracle": "the G2 canned parity oracle (stratego/training/phase18/reference_oracle.py through the G2 driver's run_oracle) must pass on this run's seed",
            "suites": "the pre-existing Phase 18 evaluator suite and the Phase 18 setup suite pass with a JUnit record",
        },
        "integrity_requirements": {
            "legality_failures": 0,
            "orientation_failures": 0,
            "attribution_failures": 0,
            "non_finite_events": 0,
            "checkpoint_identity_failures": 0,
            "sample_count": f"exactly {design.evaluation_samples} finite values in each of the four held-out utility arrays per seed",
            "budget": f"exactly {design.updates} updates, {design.updates * design.epochs_per_update * -(-design.pool_size // design.batch_size)} optimizer steps and {design.updates} EMA updates per completed seed",
            "replay": "landscape, initial models, first-period outcomes, every period digest and all four evaluation endpoints of every seed re-derive from the frozen seeds and the three-object checkpoints",
            "binding": "every confirmation artifact binds one source commit",
        },
        "decision_rules": {
            "PROCEED": "all parity, replay, binding and integrity checks pass; final raw-actor mean utility > initial raw-actor mean utility in all three seeds; the lower bound of the pooled paired 95% bootstrap interval > 0; median raw-actor gap closure across the three seeds >= 0.10",
            "STOP": "the parity-correct raw learner fails these criteria without a concrete defect",
            "REVISE": "only if a specific implementation or measurement defect invalidates the run; an unfavourable valid result is never reclassified as an instrument problem",
            "BLOCKED": "required evidence cannot be produced because of an unresolved dependency",
            "ema_results": "recorded as secondary mechanism telemetry; they cannot change the decision",
            "scope_of_a_pass": "closes only the synthetic trainability portion of G2 and authorizes designing the next gate; it does not authorize launching G3 or the full warmstart",
        },
        "execution": {
            "device": design.device,
            "threads": design.threads,
            "reproducibility": "CPU float32 with pinned threads, as G2; the S29 checkpoint round trip is additionally proved on MPS by the setup suite",
            "artifact_root_absolute": str(ARTIFACT_ROOT),
            "artifact_root_relative": str(ARTIFACT_RELATIVE),
            "execution_worktree": str(EXECUTION_WORKTREE),
            "post_freeze_rule": "no frozen field changes after the first outcome is generated; a post-freeze implementation correction abandons this namespace and is documented, never silently amended",
            "sealed_phase8_test_access": {"planned": 0, "rule": "no Stratego game and no sealed Phase 8 example is opened by this work package"},
            "stratego_games_played": 0,
        },
        "entropy_normalizer": ENTROPY_NORMALIZER,
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_json(reports / CONTRACT_NAME, contract)
    write_json(reports / LANDSCAPE_NAME, landscape_document)
    log(f"contract and landscape frozen under {reports}: optimum {landscape.optimum:.4f}, uniform mean {landscape.uniform_mean:.4f} sd {landscape.uniform_sd:.4f}; fresh {freshness['fresh']}; method unchanged {method_unchanged(identity)}")
    return contract


# ---------------------------------------------------------------------------
# Frozen-contract verification
# ---------------------------------------------------------------------------


def load_frozen(reports: Path) -> tuple:
    contract_path, landscape_path = reports / CONTRACT_NAME, reports / LANDSCAPE_NAME
    if not contract_path.exists() or not landscape_path.exists():
        raise G2RawError(f"BLOCKED: no frozen contract/landscape under {reports}; run --freeze and commit first")
    contract = json.loads(contract_path.read_text())
    landscape_document = json.loads(landscape_path.read_text())
    return contract, landscape_document, file_sha256(contract_path), file_sha256(landscape_path)


def verify_frozen_identity(contract: dict, landscape_document: dict):
    """Rebuild every frozen identity and refuse on any drift."""
    design = frozen_design()
    if json_document_digest(design.document()) != json_document_digest(contract["design"]):
        raise G2RawError("BLOCKED: the frozen design does not re-derive from the code")
    if contract["run_id"] != RUN_ID or contract["design"]["namespace"] != SEED_NAMESPACE:
        raise G2RawError("BLOCKED: run id or namespace drift")
    if contract["seed_derivation"]["base_commit"] != BASE_COMMIT or contract["seed_derivation"]["seed_namespace"] != SEED_NAMESPACE:
        raise G2RawError("BLOCKED: the seed derivation does not bind the reviewed base commit")
    if json_document_digest(contract["seed_derivation"]) != json_document_digest(seed_derivation(design)):
        raise G2RawError("BLOCKED: the frozen seeds do not re-derive")
    landscape = landscape_from_document(landscape_document)
    if landscape_document["table_digest"] != contract["landscape"]["table_digest"]:
        raise G2RawError("BLOCKED: the landscape document and the contract disagree on the table digest")
    if abs(landscape.optimum - contract["landscape"]["exact_optimum"]) > 1e-9:
        raise G2RawError("BLOCKED: the exact optimum does not re-derive")
    potentials = contract["landscape"]["certificate_potentials"]
    check = verify_certificate(landscape.table, potentials["u"], potentials["v"], contract["landscape"]["optimal_setup"])
    if not check["certified"]:
        raise G2RawError("BLOCKED: the recorded certificate no longer verifies")
    if contract["question"]["uncertainty"]["bootstrap_seeds"] != json.loads(json.dumps(bootstrap_seeds(design))):
        raise G2RawError("BLOCKED: the bootstrap seeds do not re-derive")
    return design, landscape


# ---------------------------------------------------------------------------
# Stage 2: verify (preflight: method identity, tests, oracle, coverage)
# ---------------------------------------------------------------------------


def stage_verify(reports: Path, *, references: Path | None = None) -> dict:
    from stratego.training.phase18.coverage import attach_test_outcomes, verify_coverage

    references = references or reports
    g2 = load_g2_driver()
    contract, landscape_document, contract_sha, landscape_sha = load_frozen(reports)
    design, landscape = verify_frozen_identity(contract, landscape_document)
    identity = method_identity(references)
    if not method_unchanged(identity):
        raise G2RawError("BLOCKED: a learning-method file differs from the G2 launch manifest; the namespace must be abandoned, not amended")
    if json_document_digest(identity["files"]) != json_document_digest(contract["method_identity"]["files"]):
        raise G2RawError("BLOCKED: the method-file digests moved since the freeze")
    directory = reports / DIRECTORY
    log("checking the learning-method digests against the G2 launch manifest: identical")
    log("running the pre-existing Phase 18 evaluator suite")
    evaluator = g2.run_pytest(EVALUATOR_SUITE, directory / "junit_evaluator_suite.xml")
    log(f"evaluator suite: {evaluator['summary_line']}")
    log("running the Phase 18 setup tests")
    setup = g2.run_pytest(SETUP_SUITE, directory / "junit_setup_suite.xml")
    log(f"setup suite: {setup['summary_line']}")
    log("running the canned parity oracle on this run's seed")
    oracle_record = g2.run_oracle(design)
    oracle_record["artifact"] = "phase18_g2_raw_confirmation_parity_oracle_v1"
    oracle_record["oracle_driver"] = {"module": G2_DRIVER, "sha256": file_sha256(REPOSITORY_ROOT / G2_DRIVER), "note": "the G2 driver's run_oracle, reused unchanged"}
    log(f"oracle: {'PASS' if oracle_record['passed'] else 'FAIL'}")
    coverage = attach_test_outcomes(verify_coverage(REPOSITORY_ROOT), setup["outcomes"])
    coverage.update({
        "artifact": "phase18_g2_raw_confirmation_parity_coverage_v1",
        "method_map": METHOD_MAP,
        "method_map_sha256": file_sha256(REPOSITORY_ROOT / METHOD_MAP),
        "recorded_run": {"junit": setup["junit"], "junit_sha256": setup["junit_sha256"], "counts": setup["counts"]},
        "rule": "a row is complete only when every cited test passed in the recorded run; documentation alone never completes a row",
        "timestamp_utc": utc_now(),
    })
    write_json(directory / COVERAGE_NAME, coverage)
    write_json(directory / ORACLE_NAME, oracle_record)
    record = {
        "artifact": "phase18_g2_raw_confirmation_verification_v1",
        "run_id": RUN_ID,
        "contract_sha256": contract_sha,
        "landscape_sha256": landscape_sha,
        "method_identity": identity,
        "method_unchanged": method_unchanged(identity),
        "evaluator_suite": {k: v for k, v in evaluator.items() if k != "outcomes"},
        "setup_suite": {k: v for k, v in setup.items() if k != "outcomes"},
        "oracle_passed": oracle_record["passed"],
        "coverage_all_rows_complete": coverage["all_g2_rows_complete"],
        "coverage_rows_complete": coverage["rows_complete"],
        "coverage_problems": coverage["problems"],
        "parity_passed": bool(method_unchanged(identity) and oracle_record["passed"] and coverage["all_g2_rows_complete"] and evaluator["return_code"] == 0 and setup["return_code"] == 0),
        "environment": environment(),
        "timestamp_utc": utc_now(),
    }
    write_json(directory / VERIFICATION_NAME, record)
    log(f"coverage: {coverage['rows_complete']}/{coverage['rows_total']} rows complete; parity {'PASS' if record['parity_passed'] else 'FAIL'}")
    return record


# ---------------------------------------------------------------------------
# Stage 3: launch manifest (from the clean execution worktree)
# ---------------------------------------------------------------------------


def stage_launch_manifest(reports: Path, *, source_commit: str) -> dict:
    porcelain = git_output("status", "--porcelain")
    if porcelain:
        raise G2RawError(f"BLOCKED: the execution worktree is not clean:\n{porcelain}")
    head = git_output("rev-parse", "HEAD")
    if head != source_commit:
        raise G2RawError(f"BLOCKED: HEAD {head} is not the frozen source commit {source_commit}")
    if REPOSITORY_ROOT != EXECUTION_WORKTREE:
        raise G2RawError(f"BLOCKED: the launch manifest must be written from {EXECUTION_WORKTREE}, not {REPOSITORY_ROOT}")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, head], cwd=REPOSITORY_ROOT, capture_output=True).returncode == 0
    if not ancestor:
        raise G2RawError(f"BLOCKED: the reviewed base commit {BASE_COMMIT} is not an ancestor of {head}")
    contract, landscape_document, contract_sha, landscape_sha = load_frozen(reports)
    design, landscape = verify_frozen_identity(contract, landscape_document)
    worktree_contract = REPOSITORY_ROOT / "reports" / "phase18" / CONTRACT_NAME
    if file_sha256(worktree_contract) != contract_sha or file_sha256(REPOSITORY_ROOT / "reports" / "phase18" / LANDSCAPE_NAME) != landscape_sha:
        raise G2RawError("BLOCKED: the worktree's committed contract or landscape differs from the canonical copy")
    identity = method_identity(REPOSITORY_ROOT / "reports" / "phase18")
    if not method_unchanged(identity):
        raise G2RawError("BLOCKED: the learning method differs from the G2 launch manifest in the execution worktree")
    ignored = subprocess.run(["git", "check-ignore", "-q", str(ARTIFACT_RELATIVE / "probe")], cwd=CANONICAL_ROOT, capture_output=True).returncode == 0
    if not ignored:
        raise G2RawError(f"BLOCKED: {ARTIFACT_RELATIVE} is not git-ignored in the canonical tree")
    if ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.iterdir()):
        raise G2RawError(f"BLOCKED: the artifact root {ARTIFACT_ROOT} already exists and is not empty")
    manifest = {
        "artifact": "phase18_g2_raw_confirmation_launch_manifest_v1",
        "work_package": WORK_PACKAGE,
        "agent": AGENT,
        "gate": GATE,
        "run_id": RUN_ID,
        "timestamp_utc": utc_now(),
        "authorization": digests_of(AUTHORIZATION_FILES),
        "source": {
            "source_commit": source_commit,
            "source_tree": git_output("rev-parse", f"{source_commit}^{{tree}}"),
            "base_commit": BASE_COMMIT,
            "base_commit_is_ancestor": True,
            "execution_worktree": str(REPOSITORY_ROOT),
            "expected_execution_worktree": str(EXECUTION_WORKTREE),
            "worktree_porcelain_empty": True,
            "canonical_tree": str(CANONICAL_ROOT),
        },
        "artifacts": {
            "root_absolute": str(ARTIFACT_ROOT),
            "root_relative": str(ARTIFACT_RELATIVE),
            "git_ignored": ignored,
            "g2_artifacts": "untouched under artifacts/phase18/g2_setup_parity_v1",
            "g1_artifacts": "untouched; they live outside the repository under /Users/brandonwashington/Dev/stratego_phase18",
        },
        "contract_sha256": contract_sha,
        "landscape_sha256": landscape_sha,
        "landscape_table_digest": landscape_document["table_digest"],
        "design_digest": json_document_digest(design.document()),
        "seed_derivation": seed_derivation(design),
        "budget": {"updates": design.updates, "seeds": list(design.seed_indices), "pool_size": design.pool_size, "outcomes_per_setup": design.outcomes_per_setup, "evaluation_samples": design.evaluation_samples, "bootstrap_replicates": design.bootstrap_replicates, "gap_closure_threshold": design.gap_closure_threshold, "optimizer_steps_per_seed": design.updates * design.epochs_per_update * -(-design.pool_size // design.batch_size), "ema_updates_per_seed": design.updates},
        "method_identity": identity,
        "source_digests": digests_of(SOURCE_FILES),
        "test_digests": digests_of(TEST_FILES),
        "environment": environment(),
        "sealed_test_access": {"planned": 0},
        "stratego_games_planned": 0,
        "outcomes_generated_before_this_manifest": 0,
    }
    write_json(reports / LAUNCH_NAME, manifest)
    log(f"launch manifest bound to {source_commit[:12]} (base {BASE_COMMIT[:12]}) under {reports}")
    return manifest


# ---------------------------------------------------------------------------
# Stage 4: run one seed
# ---------------------------------------------------------------------------


def sample_count_integrity(record: dict, expected: int) -> dict:
    arrays = {}
    for name in ENDPOINT_ARRAYS:
        array = np.load(record["utilities"][name]["path"])
        finite = bool(np.isfinite(array).all())
        arrays[name] = {"count": int(array.size), "finite": finite, "ok": bool(array.size == expected and finite)}
    terminal = {
        "initial": record["initial"]["generation_telemetry"]["immediately_terminal_count"],
        "final": record["final"]["generation_telemetry"]["immediately_terminal_count"],
        "initial_raw": record["raw_diagnostic"]["initial"]["generation_telemetry"]["immediately_terminal_count"],
        "final_raw": record["raw_diagnostic"]["final"]["generation_telemetry"]["immediately_terminal_count"],
    }
    return {"expected_per_endpoint": expected, "arrays": arrays, "all_ok": all(a["ok"] for a in arrays.values()), "evaluation_immediately_terminal_counts": terminal}


def stage_run(reports: Path, *, seed_index: int, artifact_root: Path) -> dict:
    contract, landscape_document, contract_sha, landscape_sha = load_frozen(reports)
    design, landscape = verify_frozen_identity(contract, landscape_document)
    launch_path = reports / LAUNCH_NAME
    if not launch_path.exists():
        raise G2RawError("BLOCKED: no launch manifest; run --launch-manifest from the clean worktree first")
    launch = json.loads(launch_path.read_text())
    if launch["contract_sha256"] != contract_sha or launch["landscape_sha256"] != landscape_sha:
        raise G2RawError("BLOCKED: the launch manifest binds a different contract or landscape")
    if launch["source"]["base_commit"] != BASE_COMMIT:
        raise G2RawError("BLOCKED: the launch manifest binds a different base commit")
    if seed_index not in design.seed_indices:
        raise G2RawError(f"BLOCKED: seed index {seed_index} is not frozen")
    output = artifact_root / f"seed_{seed_index}"
    if output.exists():
        raise G2RawError(f"BLOCKED: {output} already exists; a seed is never rerun or overwritten")
    log(f"seed {seed_index}: {design.updates} updates, pool {design.pool_size}, {design.outcomes_per_setup} outcomes per setup, {design.evaluation_samples} held-out samples; raw actor primary, EMA telemetry")
    record = run_seed(design, landscape, seed_index, output, log=log)
    record["contract_sha256"] = contract_sha
    record["landscape_sha256"] = landscape_sha
    record["source_commit"] = launch["source"]["source_commit"]
    record["base_commit"] = BASE_COMMIT
    record["primary_endpoint"] = "raw_diagnostic (the raw generation actor) - the field name is the unchanged G2 runner's; here it is the primary endpoint and 'initial'/'final' (EMA) are secondary telemetry"
    record["sample_count_integrity"] = sample_count_integrity(record, design.evaluation_samples)
    record["environment"] = environment()
    write_json(output / "seed_result.json", record)
    write_json(reports / DIRECTORY / f"phase18_g2_raw_confirmation_seed_{seed_index}_result_v1.json", record)
    raw = record["raw_diagnostic"]
    log(f"seed {seed_index}: RAW {raw['initial']['mean_utility']:.4f} -> {raw['final']['mean_utility']:.4f} (gap closed {raw['gap']['fraction_closed']:+.4%}); EMA telemetry {record['initial']['mean_utility']:.4f} -> {record['final']['mean_utility']:.4f} (gap closed {record['gap']['fraction_closed']:+.4%}); sample counts ok {record['sample_count_integrity']['all_ok']}")
    return record


# ---------------------------------------------------------------------------
# Stage 5: replay
# ---------------------------------------------------------------------------


def stage_replay(reports: Path, *, artifact_root: Path) -> dict:
    import torch

    from stratego.training.phase18.setup_learning import SetupTrainer
    from stratego.training.phase18.setup_model import build_setup_model, state_dict_digest
    from stratego.training.phase18.setup_sampling import generate_pool
    from stratego.training.phase18.synthetic_assay import evaluate_policy

    started = time.perf_counter()
    contract, landscape_document, contract_sha, landscape_sha = load_frozen(reports)
    design, landscape = verify_frozen_identity(contract, landscape_document)
    launch = json.loads((reports / LAUNCH_NAME).read_text())
    torch.set_num_threads(design.threads)

    rebuilt = landscape.document()
    moments = uniform_moments(landscape.table)
    optimum = exact_optimum(landscape.table)
    potentials = contract["landscape"]["certificate_potentials"]
    landscape_check = {
        "table_digest_matches": rebuilt["table_digest"] == landscape_document["table_digest"],
        "document_digest_matches": json_document_digest(rebuilt) == json_document_digest(landscape_document),
        "optimum_matches": abs(optimum["optimum"] - landscape_document["exact_optimum"]["optimum"]) < 1e-9,
        "optimum_certified": optimum["certificate"]["certified"],
        "independent_certificate_verifies": verify_certificate(landscape.table, potentials["u"], potentials["v"], contract["landscape"]["optimal_setup"])["certified"],
        "uniform_mean_matches": abs(moments["mean"] - landscape_document["uniform_baseline"]["mean"]) < 1e-12,
        "uniform_sd_matches": abs(moments["sd"] - landscape_document["uniform_baseline"]["sd"]) < 1e-12,
        "reflection_invariant": bool(rebuilt["reflection_invariant"]),
    }
    design_check = {"design_digest_matches_launch": json_document_digest(design.document()) == launch["design_digest"]}
    log(f"landscape replay: {landscape_check}")

    seeds: dict = {}
    for k in design.seed_indices:
        directory = artifact_root / f"seed_{k}"
        record = json.loads((directory / "seed_result.json").read_text())
        check: dict = {}
        model = build_setup_model(device=design.device, seed=design.model_seed(k))
        check["initial_raw_digest_matches"] = state_dict_digest(model) == record["initial_raw_digest"]
        receipts = [json.loads(line) for line in (directory / "outcome_receipts.jsonl").read_text().splitlines()]
        check["receipts_sha256_matches"] = file_sha256(directory / "outcome_receipts.jsonl") == record["outcome_receipts"]["sha256"]
        check["telemetry_sha256_matches"] = file_sha256(directory / "telemetry.jsonl") == record["telemetry"]["sha256"]
        by_period: dict = {}
        for row in receipts:
            by_period.setdefault(row["period"], []).append(row)
        period_digests = [json_document_digest([[r["fingerprint"], r["outcomes"]] for r in by_period[p]]) for p in sorted(by_period)]
        check["all_period_digests_match"] = period_digests == record["period_outcome_digests"]
        check["periods"] = len(by_period)

        first = generate_pool(model, namespace=design.namespace, seed_index=k, snapshot_iteration=0, snapshot_digest=state_dict_digest(model), count=design.pool_size, device=design.device)
        recorded = {row["index"]: row for row in by_period[1]}
        replayed_rows = fingerprint_mismatches = outcome_mismatches = 0
        for sample in first.samples:
            if not sample.opening_move:
                continue
            row = recorded.get(sample.index)
            if row is None or row["fingerprint"] != sample.content_fingerprint:
                fingerprint_mismatches += 1
                continue
            outcomes = landscape.outcomes_for(sample.played_canonical, seed_index=k, period=1, fingerprint=sample.content_fingerprint, replicates=design.outcomes_per_setup)
            if "".join(SYMBOL[z] for z in outcomes) != row["outcomes"]:
                outcome_mismatches += 1
            replayed_rows += 1
        check["period_1_rows_replayed"] = replayed_rows
        check["period_1_fingerprint_mismatches"] = fingerprint_mismatches
        check["period_1_outcome_mismatches"] = outcome_mismatches
        check["period_1_replays_exactly"] = replayed_rows == len(by_period[1]) and fingerprint_mismatches == 0 and outcome_mismatches == 0

        def compare(name: str, evaluation: dict, stored_name: str) -> None:
            stored = np.load(directory / stored_name)
            difference = float(np.abs(evaluation["utilities"] - stored).max()) if evaluation["utilities"].shape == stored.shape else float("inf")
            check[f"{name}_max_abs_diff"] = difference
            check[f"{name}_replays"] = bool(evaluation["utilities"].shape == stored.shape and np.allclose(evaluation["utilities"], stored, atol=1e-9))
            check[f"{name}_bitwise"] = bool(difference == 0.0)

        initial = evaluate_policy(model.eval(), landscape, design, k, "replay_initial", ema_updates=0)
        compare("initial_ema_evaluation", initial, "utilities_initial.npy")
        compare("initial_raw_evaluation", initial, "utilities_initial_raw.npy")
        check["initial_ema_digest_matches"] = initial["ema_digest"] == record["initial"]["ema_digest"]
        check["initial_raw_digest_matches_initial_ema"] = record["raw_diagnostic"]["initial"]["ema_digest"] == record["initial"]["ema_digest"]

        trainer, manifest = SetupTrainer.load_checkpoint(directory / "checkpoint_final", design.training_config(), namespace=design.namespace, seed_index=k, device=design.device)
        final_ema = evaluate_policy(trainer.evaluation_model(device=design.device), landscape, design, k, "replay_final", ema_updates=trainer.ema.updates)
        compare("final_ema_evaluation", final_ema, "utilities_final.npy")
        check["final_ema_digest_matches"] = final_ema["ema_digest"] == record["final"]["ema_digest"] == manifest["ema"]["state_digest"]
        trainer.model.eval()
        final_raw = evaluate_policy(trainer.model, landscape, design, k, "replay_final_raw", ema_updates=None)
        compare("final_raw_evaluation", final_raw, "utilities_final_raw.npy")
        check["final_raw_digest_matches"] = final_raw["ema_digest"] == record["raw_diagnostic"]["final"]["ema_digest"] == manifest["raw"]["state_digest"]
        check["checkpoint_ema_updates"] = trainer.ema.updates
        check["checkpoint_optimizer_steps"] = trainer.optimizer_step_count
        check["checkpoint_budget_matches"] = trainer.ema.updates == design.updates and trainer.optimizer_step_count == record["optimizer_steps"]
        boolean_checks = (
            "initial_raw_digest_matches", "receipts_sha256_matches", "telemetry_sha256_matches", "all_period_digests_match",
            "period_1_replays_exactly", "initial_ema_evaluation_replays", "initial_raw_evaluation_replays", "initial_ema_digest_matches",
            "initial_raw_digest_matches_initial_ema", "final_ema_evaluation_replays", "final_ema_digest_matches",
            "final_raw_evaluation_replays", "final_raw_digest_matches", "checkpoint_budget_matches",
        )
        check["boolean_checks"] = list(boolean_checks)
        check["all"] = all(bool(check[name]) for name in boolean_checks)
        check["all_endpoints_bitwise"] = all(check[f"{n}_bitwise"] for n in ("initial_ema_evaluation", "initial_raw_evaluation", "final_ema_evaluation", "final_raw_evaluation"))
        seeds[str(k)] = check
        log(f"seed {k}: replay {'OK' if check['all'] else 'MISMATCH'} - period 1 rows {replayed_rows}, raw final diff {check['final_raw_evaluation_max_abs_diff']:.2e}, ema final diff {check['final_ema_evaluation_max_abs_diff']:.2e}")

    record = {
        "artifact": "phase18_g2_raw_confirmation_replay_v1",
        "work_package": WORK_PACKAGE,
        "agent": AGENT,
        "gate": GATE,
        "run_id": RUN_ID,
        "source_commit": launch["source"]["source_commit"],
        "base_commit": BASE_COMMIT,
        "contract_sha256": contract_sha,
        "landscape_sha256": landscape_sha,
        "landscape": landscape_check,
        "design": design_check,
        "seeds": seeds,
        "all_replays_exact": all(landscape_check.values()) and design_check["design_digest_matches_launch"] and all(s["all"] for s in seeds.values()),
        "all_endpoints_bitwise": all(s["all_endpoints_bitwise"] for s in seeds.values()),
        "note": "re-derives, from the frozen documents and seeds alone, the landscape (table, optimum, certificate, moments), the initial model of every seed, the first period's pool and all of its outcomes, every period digest from the receipts, and all four evaluation endpoints (raw and EMA, initial and final) against the stored utility arrays and the three-object checkpoint. A mismatch is reported, never repaired.",
        "seconds": round(time.perf_counter() - started, 3),
        "timestamp_utc": utc_now(),
    }
    write_json(reports / DIRECTORY / REPLAY_NAME, record)
    log(f"replay {'EXACT' if record['all_replays_exact'] else 'MISMATCH'}")
    return record


# ---------------------------------------------------------------------------
# Stage 6: analyse
# ---------------------------------------------------------------------------


def criteria(design: AssayDesign, endpoints: dict, arrays: dict, *, pooled_seed: int, per_seed_seeds: dict) -> dict:
    """The three frozen criteria on one pair of endpoint arrays."""
    from stratego.evaluation.phase18.noninferiority import paired_unit_delta

    initial_all = np.concatenate([arrays[k]["initial"] for k in design.seed_indices])
    final_all = np.concatenate([arrays[k]["final"] for k in design.seed_indices])
    pooled = paired_unit_delta(final_all, initial_all, seed=int(pooled_seed), replicates=design.bootstrap_replicates, confidence=design.bootstrap_confidence)
    per_seed = {}
    for k in design.seed_indices:
        interval = paired_unit_delta(arrays[k]["final"], arrays[k]["initial"], seed=int(per_seed_seeds[str(k)]), replicates=design.bootstrap_replicates, confidence=design.bootstrap_confidence)
        per_seed[str(k)] = {"interval": interval.to_dict(), "gap_fraction_closed": endpoints[k]["gap_fraction"], "initial": endpoints[k]["initial"], "final": endpoints[k]["final"], "improved": bool(endpoints[k]["final"] > endpoints[k]["initial"])}
    fractions = sorted(endpoints[k]["gap_fraction"] for k in design.seed_indices)
    median = float(np.median(fractions))
    return {
        "all_seeds_improved": all(endpoints[k]["final"] > endpoints[k]["initial"] for k in design.seed_indices),
        "pooled_paired_interval": pooled.to_dict(),
        "pooled_lower_bound_strictly_above_zero": bool(pooled.lower > 0.0),
        "median_gap_fraction_closed": median,
        "median_gap_closure_meets_threshold": bool(median >= design.gap_closure_threshold),
        "gap_closure_threshold": design.gap_closure_threshold,
        "per_seed": per_seed,
        "gap_fractions_sorted": fractions,
    }


def apply_decision_rule(*, parity: bool, integrity: bool, replay: bool, binding, raw: dict) -> dict:
    """The frozen rule. `binding` is None until the ledger exists."""
    prerequisites = {"parity": bool(parity), "integrity": bool(integrity), "replay": bool(replay), "binding": binding}
    raw_criteria = {
        "all_seeds_improved": bool(raw["all_seeds_improved"]),
        "pooled_lower_bound_strictly_above_zero": bool(raw["pooled_lower_bound_strictly_above_zero"]),
        "median_gap_closure_meets_threshold": bool(raw["median_gap_closure_meets_threshold"]),
    }
    raw_pass = all(raw_criteria.values())
    failed = [name for name, ok in prerequisites.items() if ok is False]
    if binding is None:
        decision, basis = "PENDING", "the binding ledger has not been written; --decide applies the frozen rule after --bind"
    elif failed:
        decision = "REVISE"
        basis = f"a prerequisite check failed ({', '.join(failed)}): PROCEED is unavailable; the packet must name the concrete implementation or measurement defect, and an unfavourable valid result is never reclassified as an instrument problem"
    elif raw_pass:
        decision, basis = "PROCEED", "all parity, replay, binding and integrity checks pass; the raw actor improved in every seed, the pooled paired 95% lower bound is above zero and the median gap closure meets the 10% threshold"
    else:
        failing = [name for name, ok in raw_criteria.items() if not ok]
        decision, basis = "STOP", f"all parity, replay, binding and integrity checks pass and the parity-correct raw learner fails the frozen criteria ({', '.join(failing)}) with no concrete defect"
    return {"decision": decision, "basis": basis, "prerequisites": prerequisites, "raw_criteria": raw_criteria, "raw_criteria_pass": raw_pass, "ema_results_considered": False}


def stage_analyse(reports: Path, *, artifact_root: Path) -> dict:
    contract, landscape_document, contract_sha, landscape_sha = load_frozen(reports)
    design, landscape = verify_frozen_identity(contract, landscape_document)
    launch = json.loads((reports / LAUNCH_NAME).read_text())
    directory = reports / DIRECTORY
    verification = json.loads((directory / VERIFICATION_NAME).read_text())
    coverage = json.loads((directory / COVERAGE_NAME).read_text())
    oracle_record = json.loads((directory / ORACLE_NAME).read_text())
    replay = json.loads((directory / REPLAY_NAME).read_text())
    if verification["contract_sha256"] != contract_sha or replay["contract_sha256"] != contract_sha:
        raise G2RawError("BLOCKED: the verification or replay record binds a different contract")
    expected_steps = design.updates * design.epochs_per_update * -(-design.pool_size // design.batch_size)

    seeds: dict = {}
    arrays_raw: dict = {}
    arrays_ema: dict = {}
    integrity_total: dict = {}
    sample_counts: dict = {}
    source_commits = set()
    for k in design.seed_indices:
        seed_directory = artifact_root / f"seed_{k}"
        record = json.loads((seed_directory / "seed_result.json").read_text())
        if record["contract_sha256"] != contract_sha or record["landscape_sha256"] != landscape_sha:
            raise G2RawError(f"BLOCKED: seed {k} ran under a different contract or landscape")
        if record["updates"] != design.updates or record["optimizer_steps"] != expected_steps or record["ema_updates"] != design.updates:
            raise G2RawError(f"BLOCKED: seed {k} did not complete the frozen budget")
        for name in ENDPOINT_ARRAYS:
            path = Path(record["utilities"][name]["path"])
            if file_sha256(path) != record["utilities"][name]["sha256"]:
                raise G2RawError(f"BLOCKED: seed {k} utilities {name} digest moved")
        arrays_raw[k] = {"initial": np.load(record["utilities"]["initial_raw"]["path"]), "final": np.load(record["utilities"]["final_raw"]["path"])}
        arrays_ema[k] = {"initial": np.load(record["utilities"]["initial"]["path"]), "final": np.load(record["utilities"]["final"]["path"])}
        for name, array in (("initial", arrays_raw[k]["initial"]), ("final", arrays_raw[k]["final"])):
            if abs(float(array.mean()) - record["raw_diagnostic"][name]["mean_utility"]) > 1e-9:
                raise G2RawError(f"BLOCKED: seed {k} raw {name} array does not reproduce the recorded mean")
        for name, array in (("initial", arrays_ema[k]["initial"]), ("final", arrays_ema[k]["final"])):
            if abs(float(array.mean()) - record[name]["mean_utility"]) > 1e-9:
                raise G2RawError(f"BLOCKED: seed {k} EMA {name} array does not reproduce the recorded mean")
        counts = sample_count_integrity(record, design.evaluation_samples)
        sample_counts[str(k)] = counts
        source_commits.add(record["source_commit"])
        for key, value in record["integrity"].items():
            integrity_total[key] = integrity_total.get(key, 0) + int(value)
        seeds[k] = {
            "raw": {"initial": record["raw_diagnostic"]["initial"]["mean_utility"], "final": record["raw_diagnostic"]["final"]["mean_utility"], "gap_fraction": record["raw_diagnostic"]["gap"]["fraction_closed"], "paired": record["raw_diagnostic"]["paired"]},
            "ema": {"initial": record["initial"]["mean_utility"], "final": record["final"]["mean_utility"], "gap_fraction": record["gap"]["fraction_closed"], "paired": record["paired"]},
            "curve": record["curve"],
            "integrity": record["integrity"],
            "sample_count_integrity": counts,
            "wall_seconds": record["wall_seconds"],
            "checkpoint": record["checkpoint"],
            "period_outcome_digests_sha256": hashlib.sha256(json.dumps(record["period_outcome_digests"]).encode()).hexdigest(),
            "receipts_sha256": record["outcome_receipts"]["sha256"],
            "telemetry_sha256": record["telemetry"]["sha256"],
            "initial_raw_digest": record["initial_raw_digest"],
            "final_raw_digest": record["raw_diagnostic"]["final"]["ema_digest"],
            "final_ema_digest": record["final"]["ema_digest"],
            "model_seed": record["model_seed"],
        }
    if len(source_commits) != 1 or next(iter(source_commits)) != launch["source"]["source_commit"]:
        raise G2RawError(f"BLOCKED: the seeds bind different source commits: {source_commits}")

    seeds_for_bootstrap = bootstrap_seeds(design)
    raw = criteria(design, {k: seeds[k]["raw"] for k in seeds}, arrays_raw, pooled_seed=seeds_for_bootstrap["raw_pooled"], per_seed_seeds=seeds_for_bootstrap["raw_per_seed"])
    ema = criteria(design, {k: seeds[k]["ema"] for k in seeds}, arrays_ema, pooled_seed=seeds_for_bootstrap["ema_pooled"], per_seed_seeds=seeds_for_bootstrap["ema_per_seed"])
    integrity_failures = {k: v for k, v in integrity_total.items() if k in INTEGRITY_FAILURE_KEYS}
    zero_integrity = all(v == 0 for v in integrity_failures.values()) and len(integrity_failures) == len(INTEGRITY_FAILURE_KEYS)
    all_sample_counts_ok = all(c["all_ok"] for c in sample_counts.values())
    integrity_pass = bool(zero_integrity and all_sample_counts_ok)
    parity = bool(verification["parity_passed"] and coverage["all_g2_rows_complete"] and oracle_record["passed"] and verification["method_unchanged"])
    replay_pass = bool(replay["all_replays_exact"])
    decision_input = apply_decision_rule(parity=parity, integrity=integrity_pass, replay=replay_pass, binding=None, raw=raw)
    ema_movement = {str(k): (seeds[k]["ema"]["final"] - seeds[k]["ema"]["initial"]) / (seeds[k]["raw"]["final"] - seeds[k]["raw"]["initial"]) if seeds[k]["raw"]["final"] != seeds[k]["raw"]["initial"] else None for k in seeds}

    results = {
        "artifact": "phase18_g2_raw_confirmation_results_v1",
        "work_package": WORK_PACKAGE,
        "agent": AGENT,
        "gate": GATE,
        "run_id": RUN_ID,
        "timestamp_utc": utc_now(),
        "source_commit": launch["source"]["source_commit"],
        "base_commit": BASE_COMMIT,
        "contract_sha256": contract_sha,
        "landscape_sha256": landscape_sha,
        "launch_manifest_sha256": file_sha256(reports / LAUNCH_NAME),
        "question": QUESTION_TEXT,
        "primary_endpoint": "raw generation actor (this synthetic trainability assay only); the EMA is secondary telemetry",
        "landscape": {"exact_optimum": landscape.optimum, "uniform_mean": landscape.uniform_mean, "uniform_sd": landscape.uniform_sd, "table_digest": landscape_document["table_digest"], "table_seed": landscape_document["table_seed"]},
        "design": {"seeds": list(design.seed_indices), "updates": design.updates, "pool_size": design.pool_size, "outcomes_per_setup": design.outcomes_per_setup, "evaluation_samples": design.evaluation_samples, "bootstrap_replicates": design.bootstrap_replicates, "gap_closure_threshold": design.gap_closure_threshold, "optimizer_steps_per_seed": expected_steps, "ema_updates_per_seed": design.updates, "ema_retained_initial_fraction": SETUP_EMA_DECAY ** design.updates, "ema_time_constant_updates": 1.0 / (1.0 - SETUP_EMA_DECAY)},
        "parity": {"method_unchanged": verification["method_unchanged"], "coverage_rows_complete": coverage["rows_complete"], "coverage_rows_total": coverage["rows_total"], "all_rows_complete": coverage["all_g2_rows_complete"], "oracle_passed": oracle_record["passed"], "evaluator_suite": verification["evaluator_suite"]["counts"], "setup_suite": verification["setup_suite"]["counts"], "passed": parity},
        "integrity": {"totals": integrity_total, "failures": integrity_failures, "zero_failures": zero_integrity, "sample_counts": sample_counts, "all_sample_counts_ok": all_sample_counts_ok, "passed": integrity_pass},
        "replay": {"all_replays_exact": replay_pass, "all_endpoints_bitwise": replay["all_endpoints_bitwise"], "per_seed": {k: {"all": v["all"], "all_endpoints_bitwise": v["all_endpoints_bitwise"]} for k, v in replay["seeds"].items()}},
        "seeds": {str(k): seeds[k] for k in seeds},
        "raw_criteria": {"role": "PRIMARY - decides the confirmation", **raw},
        "ema_criteria": {"role": "SECONDARY TELEMETRY - cannot change the decision", **ema},
        "ema_movement_as_fraction_of_raw_displacement": ema_movement,
        "criteria_summary": {"parity": parity, "integrity": integrity_pass, "replay": replay_pass, "raw_all_seeds_improved": raw["all_seeds_improved"], "raw_pooled_lower_bound_above_zero": raw["pooled_lower_bound_strictly_above_zero"], "raw_median_gap_closure_meets_threshold": raw["median_gap_closure_meets_threshold"], "raw_criteria_pass": decision_input["raw_criteria_pass"], "ema_all_seeds_improved": ema["all_seeds_improved"], "ema_pooled_lower_bound_above_zero": ema["pooled_lower_bound_strictly_above_zero"], "ema_median_gap_closure_meets_threshold": ema["median_gap_closure_meets_threshold"]},
        "decision_input_pending_binding": decision_input,
        "decision_rules": contract["decision_rules"],
        "sealed_test_access": {"examples_opened": 0, "multiplicity_increment": 0},
        "stratego_games_played": 0,
        "environment": environment(),
    }
    write_json(reports / RESULTS_NAME, results)
    log(f"RAW (primary): all improved {raw['all_seeds_improved']}, lower {raw['pooled_paired_interval']['lower']:+.5f}, median gap {raw['median_gap_fraction_closed']:+.4%}; EMA (telemetry): lower {ema['pooled_paired_interval']['lower']:+.5f}, median gap {ema['median_gap_fraction_closed']:+.4%}; parity {parity}, integrity {integrity_pass}, replay {replay_pass} -> {decision_input['decision']} pending binding")
    return results


# ---------------------------------------------------------------------------
# Stage 7: bind
# ---------------------------------------------------------------------------


def bound_commit(payload: dict):
    return payload.get("source_commit") or (payload.get("source") or {}).get("source_commit")


def stage_bind(reports: Path) -> dict:
    launch = json.loads((reports / LAUNCH_NAME).read_text())
    commit = launch["source"]["source_commit"]
    entries: dict = {}
    mismatched = []
    names = [CONTRACT_NAME, LANDSCAPE_NAME, LAUNCH_NAME, RESULTS_NAME] + sorted(p.name for p in (reports / DIRECTORY).glob("*.json"))
    for name in names:
        path = reports / name if (reports / name).exists() else reports / DIRECTORY / name
        payload = json.loads(path.read_text())
        bound = bound_commit(payload)
        form = "full" if bound else "pre-commit artifact; bound by this ledger's digest and the launch manifest"
        agrees = (bound == commit) if bound else None
        if bound and not agrees:
            mismatched.append(name)
        entries[str(path.relative_to(reports))] = {"sha256": file_sha256(path), "bytes": path.stat().st_size, "source_binding": {"form": form, "value": bound, "agrees": agrees}}
    ledger = {
        "artifact": "phase18_g2_raw_confirmation_binding_v1",
        "work_package": WORK_PACKAGE,
        "agent": AGENT,
        "gate": GATE,
        "run_id": RUN_ID,
        "source_commit": commit,
        "source_tree": launch["source"]["source_tree"],
        "base_commit": BASE_COMMIT,
        "artifacts": entries,
        "mismatched_artifacts": mismatched,
        "all_artifacts_bind_one_source_commit": not mismatched,
        "artifact_root": launch["artifacts"],
        "timestamp_utc": utc_now(),
    }
    write_json(reports / BINDING_NAME, ledger)
    log(f"binding ledger: {len(entries)} artifacts, {len(mismatched)} mismatched")
    return ledger


# ---------------------------------------------------------------------------
# Stage 8: decide
# ---------------------------------------------------------------------------


def stage_decide(reports: Path) -> dict:
    results = json.loads((reports / RESULTS_NAME).read_text())
    ledger = json.loads((reports / BINDING_NAME).read_text())
    results_sha = file_sha256(reports / RESULTS_NAME)
    if ledger["artifacts"].get(RESULTS_NAME, {}).get("sha256") != results_sha:
        raise G2RawError("BLOCKED: the binding ledger does not bind the current results file")
    if ledger["source_commit"] != results["source_commit"]:
        raise G2RawError("BLOCKED: the ledger and the results bind different source commits")
    binding = bool(ledger["all_artifacts_bind_one_source_commit"])
    decision = apply_decision_rule(parity=results["parity"]["passed"], integrity=results["integrity"]["passed"], replay=results["replay"]["all_replays_exact"], binding=binding, raw=results["raw_criteria"])
    record = {
        "artifact": "phase18_g2_raw_confirmation_decision_input_v1",
        "work_package": WORK_PACKAGE,
        "agent": AGENT,
        "gate": GATE,
        "run_id": RUN_ID,
        "source_commit": results["source_commit"],
        "base_commit": BASE_COMMIT,
        "results_sha256": results_sha,
        "binding_ledger_sha256": file_sha256(reports / BINDING_NAME),
        "question": QUESTION_TEXT,
        "rule": results["decision_rules"],
        "raw_criteria_summary": {k: v for k, v in results["raw_criteria"].items() if k in ("all_seeds_improved", "pooled_lower_bound_strictly_above_zero", "median_gap_fraction_closed", "median_gap_closure_meets_threshold", "gap_fractions_sorted")},
        "raw_pooled_paired_interval": results["raw_criteria"]["pooled_paired_interval"],
        "ema_telemetry_summary": {k: v for k, v in results["ema_criteria"].items() if k in ("all_seeds_improved", "pooled_lower_bound_strictly_above_zero", "median_gap_fraction_closed", "median_gap_closure_meets_threshold")},
        "decision": decision,
        "scope": "a PROCEED closes only the synthetic trainability portion of G2 and authorizes designing the next gate; it does not authorize launching G3 or the full warmstart",
        "timestamp_utc": utc_now(),
    }
    write_json(reports / DECISION_INPUT_NAME, record)
    log(f"frozen rule -> {decision['decision']}: {decision['basis']}")
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stage = parser.add_mutually_exclusive_group(required=True)
    for name in ("freeze", "verify", "launch-manifest", "run", "replay", "analyse", "bind", "decide"):
        stage.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--reports", type=Path, default=CANONICAL_ROOT / "reports" / "phase18")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--seed-index", type=int)
    parser.add_argument("--source-commit")
    arguments = parser.parse_args(argv)
    try:
        if arguments.freeze:
            stage_freeze(arguments.reports, references=arguments.reports)
        elif arguments.verify:
            stage_verify(arguments.reports, references=arguments.reports)
        elif arguments.launch_manifest:
            if not arguments.source_commit:
                parser.error("--launch-manifest needs --source-commit")
            stage_launch_manifest(arguments.reports, source_commit=arguments.source_commit)
        elif arguments.run:
            if arguments.seed_index is None:
                parser.error("--run needs --seed-index")
            stage_run(arguments.reports, seed_index=arguments.seed_index, artifact_root=arguments.artifacts)
        elif arguments.replay:
            return 0 if stage_replay(arguments.reports, artifact_root=arguments.artifacts)["all_replays_exact"] else 1
        elif arguments.analyse:
            stage_analyse(arguments.reports, artifact_root=arguments.artifacts)
        elif arguments.bind:
            stage_bind(arguments.reports)
        elif arguments.decide:
            stage_decide(arguments.reports)
    except G2RawError as error:
        log(str(error))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
