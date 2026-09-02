#!/usr/bin/env python3
"""Phase 18 Gate G2: setup implementation parity and the synthetic learning assay.

Stages, in order; each must finish before the next may start:

* `--freeze`            build the frozen assay design and the synthetic landscape;
                        write the contract and the landscape document (no training)
* `--verify`            run the evaluator suite and the Phase 18 setup tests with a
                        JUnit record, run the canned parity oracle, and write the
                        coverage table with the recorded outcome of every cited test
* `--launch-manifest`   from the clean detached execution worktree: bind
                        G2_SOURCE_COMMIT, its tree, every source and test digest, the
                        frozen contract and landscape digests, and the artifact root
* `--run --seed-index K` run one frozen seed of the assay into the artifact root
* `--analyse`           the three seeds -> results, the pooled paired bootstrap, the
                        per-seed gap closure, and every decision input
* `--bind`              the binding ledger over every G2 artifact

Nothing here opens a Stratego game or a sealed Phase 8 example: the assay's
only environment is the synthetic landscape, and a source-scan test pins that
this driver imports no game runner, corpus reader or evaluation bank.
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
)
from stratego.training.phase18.synthetic_assay import ASSAY_VERSION, AssayDesign, run_seed  # noqa: E402
from stratego.training.phase18.synthetic_landscape import build_landscape, landscape_from_document  # noqa: E402

RUN_ID = "G2-SYNTHETIC-ASSAY-2026-A"
GATE = "G2"
AGENT = "phase_18_agent_4"
NAMESPACE = "phase18_g2_setup_parity_v1"

#: The canonical tree the artifacts live in, whatever tree the code runs from.
CANONICAL_ROOT = Path("/Users/brandonwashington/Dev/Github/stratego/gpt_agent")
ARTIFACT_RELATIVE = Path("artifacts/phase18/g2_setup_parity_v1")
ARTIFACT_ROOT = CANONICAL_ROOT / ARTIFACT_RELATIVE
EXECUTION_WORKTREE = Path("/Users/brandonwashington/Dev/Github/stratego/gpt_agent_phase18_g2_exec")

CONTRACT_NAME = "phase18_g2_contract_v1.json"
LANDSCAPE_NAME = "phase18_g2_synthetic_landscape_v1.json"
LAUNCH_NAME = "phase18_g2_launch_manifest_v1.json"
RESULTS_NAME = "phase18_g2_results_v1.json"
BINDING_NAME = "phase18_g2_binding_v1.json"
G2_DIRECTORY = "g2"

AUTHORIZATION_FILES = (
    "reports/phase18/decisions/P18-D003.json",
    "reports/phase18/decisions/P18-D003.md",
    "reports/phase18/reviews/P18-D003_REVIEW.md",
    "instructions/phase_18_setup_integrated_warmstart/06_AGENT_4_G2_SETUP_PARITY_AND_SYNTHETIC_ASSAY.md",
    "reports/phase18/ataraxos_setup_method_map_v2.json",
)

SOURCE_FILES = (
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
)

EVALUATOR_SUITE = "tests/evaluation/phase18"
SETUP_SUITE = "tests/training/phase18"


class G2Error(RuntimeError):
    """A frozen identity, accounting or sealing precondition failed."""


def log(message: str) -> None:
    print(f"[g2 {time.strftime('%H:%M:%S')}] {message}", flush=True)


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
            raise G2Error(f"BLOCKED: {name} is missing from {root}")
        record[name] = file_sha256(path)
    return record


def frozen_design() -> AssayDesign:
    """The one frozen design. Every field is the instruction's minimum or the
    published value; nothing is tuned."""
    return AssayDesign(namespace=NAMESPACE, run_id=RUN_ID)


def build_frozen_landscape(design: AssayDesign):
    return build_landscape(
        namespace=design.namespace,
        table_seed=design.landscape_table_seed(),
        kappa=design.landscape_kappa,
        p_draw=design.landscape_p_draw,
    )


# ---------------------------------------------------------------------------
# Stage 1: freeze
# ---------------------------------------------------------------------------


def stage_freeze(reports: Path) -> dict:
    started = time.perf_counter()
    design = frozen_design()
    if design.reduced:
        raise G2Error("BLOCKED: the frozen design must not be reduced")
    landscape = build_frozen_landscape(design)
    landscape_document = landscape.document()
    if not landscape_document["exact_optimum"]["certificate"]["certified"]:
        raise G2Error("BLOCKED: the exact optimum could not be certified")
    ema_retained = SETUP_EMA_DECAY ** design.updates

    contract = {
        "artifact": "phase18_g2_contract_v1",
        "work_package": WORK_PACKAGE,
        "agent": AGENT,
        "gate": GATE,
        "run_id": RUN_ID,
        "package_version": PHASE18_SETUP_PACKAGE_VERSION,
        "assay_version": ASSAY_VERSION,
        "timestamp_utc": utc_now(),
        "authorization": digests_of(AUTHORIZATION_FILES),
        "references": {
            "paper": PAPER_ID,
            "paper_sha256": PAPER_SHA256,
            "published_source": PUBLISHED_SOURCE,
            "published_source_commit": PUBLISHED_SOURCE_COMMIT,
            "method_map": METHOD_MAP,
            "note": "the paper and the published source are technical references; the Phase 18 contracts and instruction 06 govern",
        },
        "question": {
            "text": (
                "Does the local scaled setup-policy implementation match the paper and the pinned published "
                "implementation at loss, gradient, sampling, aggregation, optimizer, checkpoint and EMA "
                "semantics, and can it reliably learn a known synthetic setup-reward landscape from "
                "outcome-only feedback across three fresh seeds?"
            ),
            "sub_gates": ["method and implementation parity (S01-S30 + canned oracle)", "synthetic learning with a known answer"],
            "null_hypothesis": "the setup learner does not improve the EMA model's expected landscape utility: for at least one seed U_final(EMA) <= U_initial(EMA), or the pooled paired lower bound is <= 0, or the median seed closes less than 10% of its initial-to-optimum gap",
            "alternative_hypothesis": "for every seed U_final(EMA) > U_initial(EMA), the pooled paired 95% lower bound is strictly above zero, and the median seed closes at least 10% of its gap",
            "primary_metric": "expected landscape utility: the mean utility U(s) over the held-out EMA samples (4,096 per seed per endpoint), initial (0 updates) versus final (after the last fixed update)",
            "practical_margin": {"median_gap_closure_fraction": design.gap_closure_threshold, "definition": "(U_final - U_initial) / (U_optimum - U_initial) per seed, median over the three seeds, must be >= 0.10"},
            "uncertainty": {
                "unit": "one held-out evaluation sample: initial and final EMA samples share the same per-token uniforms (common random numbers), so the per-sample difference U_final[i] - U_initial[i] is paired",
                "pooled_statistic": "mean paired difference over all seeds' samples (3 x 4,096 = 12,288 pairs)",
                "method": "two-sided 95% paired percentile bootstrap over the pooled paired differences, 10,000 replicates, frozen seed; the lower endpoint must be strictly greater than zero",
                "implementation": "stratego.evaluation.phase18.noninferiority.paired_unit_delta (the accepted paired bootstrap; one shared index draw for both endpoints)",
                "per_seed_intervals": "reported as diagnostics with the same method and seed offsets; they do not decide the gate",
                "bootstrap_seed": design.bootstrap_seed(),
            },
            "checkpoint_rule": "the EMA after the final fixed update (update 64) decides; intermediate curve points are telemetry and never select a checkpoint",
            "sample_size_basis": "instruction minimum: 4,096 held-out EMA setups per endpoint per seed, three seeds, 10,000 replicates",
        },
        "design": design.document(),
        "landscape": {
            "version": landscape_document["landscape_version"],
            "table_seed": landscape_document["table_seed"],
            "table_digest": landscape_document["table_digest"],
            "reflection_invariant": landscape_document["reflection_invariant"],
            "uniform_baseline": landscape_document["uniform_baseline"],
            "exact_optimum": landscape_document["exact_optimum"]["optimum"],
            "optimum_certified": landscape_document["exact_optimum"]["certificate"]["certified"],
            "outcome_mapping": landscape_document["outcome_mapping"],
            "outcomes_per_eligible_setup": design.outcomes_per_setup,
            "learner_interface": landscape_document["learner_interface"],
            "document": LANDSCAPE_NAME,
            "fresh_policy_baseline": "estimated separately by the assay from each seed's initial EMA sample (4,096 draws) and recorded per seed; the exact uniform-random baseline above is the model-free reference",
        },
        "parity_requirements": {
            "rows": "S01-S30 of the method map, each mapped to implementation symbols and tests in stratego/training/phase18/coverage.py; a row is complete only when every cited test passed in the recorded --verify run",
            "canned_oracle": "stratego/training/phase18/reference_oracle.py must agree with the production loss on every term and with production autograd on representative parameters (central finite differences in float64)",
            "integrity": "zero legality, orientation, attribution, non-finite or checkpoint-identity failures across all three seeds",
        },
        "decision_rules": {
            "PROCEED": "every S01-S30 row complete; canned forward-loss and gradient parity pass; zero integrity failures; final EMA utility > initial EMA utility for all three seeds; pooled paired 95% lower bound strictly > 0; median seed gap closure >= 0.10",
            "REVISE": "a concrete, isolated implementation or instrument defect prevents a valid result (see the predeclared instrument finding below)",
            "STOP": "a valid parity-correct implementation fails the frozen synthetic learning criteria with no isolated defect: the learner itself does not improve",
            "BLOCKED": "required evidence cannot be produced because of an unresolved dependency",
        },
        "predeclared_instrument_finding": {
            "statement": (
                "The evaluation model is the EMA, updated once after each complete setup update with decay 0.999, "
                "and the budget is 64 updates. By arithmetic the final EMA retains 0.999^64 = "
                f"{ema_retained:.6f} of the initial parameters in its geometric weighting; the EMA can move only about "
                "3% of the raw actor's total parameter displacement inside this budget. The EMA-based criteria therefore "
                "have very little power at this budget whatever the raw learner does. This was established before any frozen seed ran, "
                "from the decay arithmetic and from a development smoke on a separate namespace (phase18_g2_dev_smoke_v1/v2, recorded in reports/phase18/g2/dev_smoke_v1.json)."
            ),
            "raw_diagnostic": (
                "The assay additionally records the RAW actor's held-out utility on the same evaluation stream at every endpoint, "
                "with the same 4,096 paired samples per seed, the same pooled paired bootstrap and the same gap-closure fraction. "
                "It is a DIAGNOSTIC: it never decides the gate and the raw model is never the evaluation model in the implementation (S28)."
            ),
            "predeclared_interpretation": {
                "ema_criteria_pass": "PROCEED (the frozen criteria are what the instruction requires)",
                "ema_criteria_fail_and_raw_diagnostic_satisfies_the_same_three_criteria": "REVISE: instrument defect 'EMA horizon exceeds the update budget'; the learner demonstrably learns the landscape and the gate's evaluation model cannot reflect it within 64 updates",
                "ema_criteria_fail_and_raw_diagnostic_also_fails": "STOP unless a concrete isolated implementation defect is found, in which case REVISE",
                "any_integrity_failure_or_parity_failure": "REVISE (defect) or BLOCKED (dependency), never PROCEED",
            },
            "frozen_fields_unchanged": "no landscape, threshold, budget, seed or rule was altered after the smoke; the smoke used a different namespace and therefore different seeds and a different table",
        },
        "execution": {
            "device": design.device,
            "threads": design.threads,
            "reproducibility": "CPU float32 with pinned threads; the S29 checkpoint round trip is additionally proved on MPS by tests/training/phase18/test_setup_learning.py::test_save_reload_and_one_more_update[mps]",
            "artifact_root_absolute": str(ARTIFACT_ROOT),
            "artifact_root_relative": str(ARTIFACT_RELATIVE),
            "execution_worktree": str(EXECUTION_WORKTREE),
            "sealed_phase8_test_access": {"planned": 0, "rule": "no Stratego game and no sealed Phase 8 example is opened by this work package"},
            "stratego_games_played": 0,
        },
        "entropy_normalizer": ENTROPY_NORMALIZER,
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_json(reports / CONTRACT_NAME, contract)
    write_json(reports / LANDSCAPE_NAME, landscape_document)
    log(f"contract and landscape frozen under {reports}: optimum {landscape.optimum:.4f}, uniform mean {landscape.uniform_mean:.4f} sd {landscape.uniform_sd:.4f}")
    return contract


# ---------------------------------------------------------------------------
# Frozen-contract verification
# ---------------------------------------------------------------------------


def load_frozen(reports: Path) -> tuple:
    contract_path, landscape_path = reports / CONTRACT_NAME, reports / LANDSCAPE_NAME
    if not contract_path.exists() or not landscape_path.exists():
        raise G2Error(f"BLOCKED: no frozen contract/landscape under {reports}; run --freeze and commit first")
    contract = json.loads(contract_path.read_text())
    landscape_document = json.loads(landscape_path.read_text())
    return contract, landscape_document, file_sha256(contract_path), file_sha256(landscape_path)


def verify_frozen_identity(contract: dict, landscape_document: dict):
    """Rebuild every frozen identity and refuse on any drift."""
    design = frozen_design()
    if json_document_digest(design.document()) != json_document_digest(contract["design"]):
        raise G2Error("BLOCKED: the frozen design does not re-derive from the code")
    landscape = landscape_from_document(landscape_document)
    if landscape_document["table_digest"] != contract["landscape"]["table_digest"]:
        raise G2Error("BLOCKED: the landscape document and the contract disagree on the table digest")
    if int(contract["question"]["uncertainty"]["bootstrap_seed"]) != design.bootstrap_seed():
        raise G2Error("BLOCKED: the bootstrap seed does not re-derive")
    if contract["run_id"] != RUN_ID or contract["design"]["namespace"] != NAMESPACE:
        raise G2Error("BLOCKED: run id or namespace drift")
    return design, landscape


# ---------------------------------------------------------------------------
# Stage 2: verify (tests, oracle, coverage)
# ---------------------------------------------------------------------------


def run_pytest(target: str, junit: Path) -> dict:
    junit.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", target, "-q", "--no-header", "-p", "no:cacheprovider", f"--junitxml={junit}"]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True)
    outcomes = parse_junit(junit)
    counts = {status: sum(1 for s in outcomes.values() if s == status) for status in ("passed", "failed", "skipped", "error")}
    return {
        "target": target,
        "command": " ".join(command),
        "return_code": completed.returncode,
        "seconds": round(time.perf_counter() - started, 3),
        "summary_line": completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "",
        "counts": counts,
        "counts_note": "outcomes are keyed by test function (a parametrised test contributes its worst case); testcases_total counts every JUnit case",
        "testcases_total": count_junit_cases(junit),
        "total": len(outcomes),
        "junit": str(junit),
        "junit_sha256": file_sha256(junit),
        "outcomes": outcomes,
    }


def count_junit_cases(path: Path) -> int:
    import xml.etree.ElementTree as ET

    return sum(1 for _ in ET.parse(path).getroot().iter("testcase"))


def parse_junit(path: Path) -> dict:
    """`file::function` -> status; a parametrised test contributes its worst case."""
    import xml.etree.ElementTree as ET

    rank = {"passed": 0, "skipped": 1, "failed": 2, "error": 3}
    outcomes: dict = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        classname = case.get("classname", "")
        name = case.get("name", "")
        base = name.split("[")[0]
        module = classname.split(".")[-1] + ".py" if classname else ""
        key = f"{module}::{base}"
        status = "passed"
        for child in case:
            if child.tag in ("failure", "error"):
                status = "failed" if child.tag == "failure" else "error"
            elif child.tag == "skipped":
                status = "skipped"
        if rank[status] >= rank.get(outcomes.get(key, "passed"), 0):
            outcomes[key] = status
    return outcomes


def run_oracle(design: AssayDesign) -> dict:
    """The canned parity oracle against the production loss and autograd."""
    import copy

    import numpy as np
    import torch

    from stratego.engine.constants import FLAG
    from stratego.training.phase18 import reference_oracle as oracle
    from stratego.training.phase18.setup_buffer import SetupBuffer
    from stratego.training.phase18.setup_contract import START_TOKEN
    from stratego.training.phase18.setup_learning import setup_batch_loss
    from stratego.training.phase18.setup_model import build_setup_model, state_dict_digest
    from stratego.training.phase18.setup_sampling import generate_pool

    torch.set_num_threads(design.threads)
    namespace = "phase18_g2_parity_oracle_v1"
    model = build_setup_model(device="cpu", seed=design.model_seed(1))
    pool = generate_pool(model, namespace=namespace, seed_index=1, snapshot_iteration=0, snapshot_digest=state_dict_digest(model), count=8)
    pattern = [(1, 1, 0, -1), (1, -1, -1, -1), (0, 0, 1, 1), (1, 1, 1, 1), (-1, 0, -1, 1), (0, 1, -1, 1)]
    outcomes = {s.content_fingerprint: pattern[i % len(pattern)] for i, s in enumerate(pool.samples)}
    buffer = SetupBuffer(storage_duration=1)
    buffer.add_pool(pool.samples, period=1)
    for fingerprint, zs in outcomes.items():
        buffer.add_outcomes((fingerprint, z) for z in zs)
    alpha = 0.1
    processed = buffer.process(alpha=alpha)
    batch = next(buffer.minibatches(64, seed=0))
    fixture = {
        "sequence": batch.sequence.numpy().astype(np.int64),
        "tokens": batch.tokens.numpy().astype(np.int64),
        "masks": batch.masks.numpy().astype(bool),
        "behavior_log_probs": batch.behavior_log_probs.numpy().astype(np.float64),
        "behavior_selected_log_prob": batch.behavior_selected_log_prob.numpy().astype(np.float64),
        "advantage": batch.advantage.numpy().astype(np.float64),
        "value_target": batch.value_target.numpy().astype(np.float64),
        "entropy_target": batch.entropy_target.numpy().astype(np.float64),
    }
    coefficients = {"clip_epsilon": 0.2, "policy_weight": 1.0, "value_weight": 0.5, "entropy_weight": 1.0, "kl_weight": 0.1}
    by_fingerprint = {s.content_fingerprint: s for s in pool.samples}

    # Forward quantities: masks, I, E[v], aggregation, advantage, recursion.
    # `processed` rows are in ready-index order; the minibatch is shuffled, so
    # the processed position is looked up by fingerprint.
    processed_position = {buffer.samples[int(index)].content_fingerprint: position for position, index in enumerate(processed.indices)}
    quantity_checks = []
    for row, fingerprint in enumerate(batch.fingerprints):
        sample = by_fingerprint[fingerprint]
        position = processed_position[fingerprint]
        information = oracle.oracle_suffix_information(sample.behavior_selected_log_prob)
        mean, count, z_bar = oracle.oracle_running_mean(outcomes[fingerprint])
        expected = oracle.oracle_expected_value(oracle.oracle_softmax(sample.wdl_logits))
        advantage = oracle.oracle_flat_advantage(z_bar, expected, information, sample.entropy_prediction, alpha)
        recursion = oracle.oracle_published_recursion(mean, oracle.oracle_softmax(sample.wdl_logits), -sample.behavior_selected_log_prob.astype(np.float64), sample.entropy_prediction, td_lambda=1.0, gae_lambda=1.0, reg_temp=alpha, reg_norm=ENTROPY_NORMALIZER)
        quantity_checks.append({
            "fingerprint": fingerprint[:16],
            "masks_match": bool(np.array_equal(oracle.oracle_legal_masks(fixture["tokens"][row]), fixture["masks"][row])),
            "prefix_alignment": bool(fixture["sequence"][row, 0] == START_TOKEN and np.array_equal(fixture["sequence"][row, 1:], fixture["tokens"][row])),
            "information_max_abs_diff": float(np.abs(information - sample.suffix_information).max()),
            "entropy_target_max_abs_diff": float(np.abs(information / ENTROPY_NORMALIZER - fixture["entropy_target"][row]).max()),
            "aggregation": {"count": count, "z_bar_oracle": z_bar, "z_bar_production": buffer.outcome_record(fingerprint)["z_bar"], "mean_one_hot_max_abs_diff": float(np.abs(mean - fixture["value_target"][row]).max())},
            "advantage_max_abs_diff": float(np.abs(advantage - fixture["advantage"][row]).max()),
            "published_recursion_max_abs_diff": float(np.abs(recursion["advantage"] - fixture["advantage"][row]).max()),
            "i_minus_10h_max_abs_diff": float(np.abs((information - ENTROPY_NORMALIZER * sample.entropy_prediction) - processed.entropy_residual[position]).max()),
        })

    # Loss terms in double precision.
    double = copy.deepcopy(model).to(torch.float64)
    from stratego.training.phase18.setup_contract import SetupTrainingConfig

    config = SetupTrainingConfig(run_id=RUN_ID, device="cpu")
    total, terms = setup_batch_loss(double.eval(), batch, config=config)
    expected_terms = oracle.oracle_loss_from_model(model, fixture, coefficients)
    loss_checks = {
        name: {"production": float(terms[name].detach()), "oracle": float(expected_terms[name]), "abs_diff": abs(float(terms[name].detach()) - float(expected_terms[name]))}
        for name in ("policy_loss", "value_loss", "entropy_prediction_loss", "behavior_kl", "total_loss")
    }
    loss_checks["clip_fraction"] = {"production": float(terms["clip_fraction"].detach()), "oracle": float(expected_terms["clip_fraction"])}

    # Gradients: autograd of the production loss vs central differences of the oracle loss.
    parameters = [
        ("piece_head.bias", int(FLAG)), ("piece_head.weight", 5), ("wdl_head.bias", 2), ("wdl_head.weight", 100),
        ("entropy_head.bias", 0), ("entropy_head.weight", 17), ("layers.0.attention.query.weight", 130),
        ("layers.1.feed_forward.0.weight", 2048), ("layers.3.feed_forward.2.bias", 9), ("layers.2.attention_norm.weight", 7),
        ("token_embedding.weight", START_TOKEN * 128 + 4), ("positional_embedding.weight", 5 * 128 + 3), ("final_norm.weight", 40),
    ]
    double.train()
    double.zero_grad()
    total, _ = setup_batch_loss(double, batch, config=config)
    total.backward()
    named = dict(double.named_parameters())
    gradient_checks = []
    for name, index in parameters:
        autograd = float(named[name].grad.view(-1)[index])
        numeric = oracle.oracle_finite_difference_gradient(model, fixture, coefficients, name, index, epsilon=1e-4)
        gradient_checks.append({"parameter": name, "flat_index": index, "autograd": autograd, "finite_difference": numeric, "abs_diff": abs(autograd - numeric), "rel_diff": abs(autograd - numeric) / max(abs(numeric), 1e-12)})

    # Optimizer equivalence, clipping, EMA closed form, step counts.
    torch.manual_seed(0)
    parameter = torch.nn.Parameter(torch.randn(6, dtype=torch.float64))
    adamw = torch.optim.AdamW([parameter], lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    values = parameter.detach().numpy().copy()
    state: dict = {}
    adam_max_diff = 0.0
    for step in range(5):
        gradient = np.array([0.3, -0.2, 0.05, 1.0, -0.7, 0.0]) * (step + 1)
        adamw.zero_grad()
        parameter.grad = torch.as_tensor(gradient)
        adamw.step()
        values, state = oracle.oracle_adam_step(values, gradient, state, lr=5e-5, betas=(0.9, 0.999), eps=1e-8)
        adam_max_diff = max(adam_max_diff, float(np.abs(parameter.detach().numpy() - values).max()))

    single = copy.deepcopy(model)
    total_single, _ = setup_batch_loss(single, batch, config=config)
    total_single.backward()
    norms = [float(p.grad.norm()) for p in single.parameters()]
    scale = oracle.oracle_clip_scale(norms, 0.5)
    before = [p.grad.clone() for p in single.parameters()]
    pre_clip = float(torch.nn.utils.clip_grad_norm_(single.parameters(), 0.5))
    clip_max_diff = max(float((g - b * scale).abs().max()) for g, b in zip((p.grad for p in single.parameters()), before))
    post_clip = float(np.sqrt(sum(float(p.grad.pow(2).sum()) for p in single.parameters())))

    from stratego.training.phase18.setup_learning import SetupEMA, SetupTrainer

    ema = SetupEMA(copy.deepcopy(model), 0.999)
    shadow0 = {k: v.clone().numpy() for k, v in ema.state_dict().items()}
    raw = model.state_dict()
    for _ in range(7):
        ema.update(model)
    ema_max_diff = max(float(np.abs(tensor.numpy() - oracle.oracle_ema_closed_form(shadow0[name], raw[name].numpy(), 0.999, 7)).max()) for name, tensor in ema.state_dict().items())
    trainer = SetupTrainer(copy.deepcopy(model), config.replace(batch_size=8), namespace=namespace, seed_index=1)
    result = trainer.update(buffer, global_iteration=1)
    steps = oracle.oracle_step_counts(int(processed.indices.size), 8, 5)

    # Save / reload / one more update on the production device (MPS) when available.
    device_checks = {}
    for device in (["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]):
        import tempfile

        cfg = config.replace(device=device, batch_size=8)
        m = build_setup_model(device=device, seed=21)
        t = SetupTrainer(m, cfg, namespace=namespace, seed_index=2)
        g = generate_pool(m, namespace=namespace, seed_index=2, snapshot_iteration=0, snapshot_digest=state_dict_digest(m), count=16, device=device)
        b = SetupBuffer(storage_duration=1, device=device)
        b.add_pool(g.samples, period=1)
        for s in g.samples:
            b.add_outcomes((s.content_fingerprint, z) for z in (1, -1, 0, 1))
        t.update(b, global_iteration=1)
        b.filter(1)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = t.save_checkpoint(Path(tmp) / "ckpt")
            restored, _ = SetupTrainer.load_checkpoint(Path(tmp) / "ckpt", cfg, namespace=namespace, seed_index=2, device=device)
            raw_match = state_dict_digest(restored.model) == state_dict_digest(t.model)
            ema_match = state_dict_digest(restored.ema.as_model()) == state_dict_digest(t.ema.as_model())
            g2 = generate_pool(restored.model, namespace=namespace, seed_index=2, snapshot_iteration=1, snapshot_digest=state_dict_digest(restored.model), count=16, device=device)
            b2 = SetupBuffer(storage_duration=1, device=device)
            b2.add_pool(g2.samples, period=2)
            for s in g2.samples:
                b2.add_outcomes((s.content_fingerprint, z) for z in (1, 1, -1, 0))
            more = restored.update(b2, global_iteration=2)
        device_checks[device] = {"raw_restored": raw_match, "ema_restored": ema_match, "ema_device": str(restored.ema.device), "one_more_update_steps": more.optimizer_steps, "ema_updates_after": restored.ema.updates, "three_objects": sorted(k for k in manifest if k in ("raw", "ema", "optimizer"))}

    tolerance = {"loss_abs": 1e-8, "gradient_rel": 1e-5, "gradient_abs": 1e-7, "quantity_abs": 1e-4, "ema_abs": 1e-5, "adam_abs": 1e-12, "clip_abs": 1e-6}
    passed = (
        all(c["masks_match"] and c["prefix_alignment"] and c["information_max_abs_diff"] < tolerance["quantity_abs"] and c["advantage_max_abs_diff"] < tolerance["quantity_abs"] and c["published_recursion_max_abs_diff"] < tolerance["quantity_abs"] and c["i_minus_10h_max_abs_diff"] < tolerance["quantity_abs"] and c["aggregation"]["mean_one_hot_max_abs_diff"] < 1e-9 for c in quantity_checks)
        and all(v["abs_diff"] < tolerance["loss_abs"] for k, v in loss_checks.items() if k != "clip_fraction")
        and all(g["abs_diff"] < tolerance["gradient_abs"] or g["rel_diff"] < tolerance["gradient_rel"] for g in gradient_checks)
        and adam_max_diff < tolerance["adam_abs"] and clip_max_diff < tolerance["clip_abs"] and post_clip <= 0.5 + 1e-5 and ema_max_diff < tolerance["ema_abs"]
        and result.optimizer_steps == steps["optimizer_steps"] and result.ema_updates == 1
        and all(v["raw_restored"] and v["ema_restored"] and v["ema_updates_after"] == 2 for v in device_checks.values())
    )
    return {
        "artifact": "phase18_g2_parity_oracle_v1",
        "oracle_module": "stratego/training/phase18/reference_oracle.py",
        "independence": "the oracle imports no production loss, buffer or trainer; a test scans its imports",
        "fixture": {"namespace": namespace, "model_seed": design.model_seed(1), "setups": len(pool.samples), "alpha": alpha, "outcomes_per_setup": 4},
        "quantities": quantity_checks,
        "loss_terms_double_precision": loss_checks,
        "gradients": gradient_checks,
        "optimizer": {"adamw_vs_oracle_adam_max_abs_diff": adam_max_diff, "steps": 5, "weight_decay": 0.0},
        "clipping": {"pre_clip_norm": pre_clip, "post_clip_norm": post_clip, "oracle_scale": scale, "max_abs_diff_after_scale": clip_max_diff},
        "ema": {"closed_form_max_abs_diff_after_7_updates": ema_max_diff, "decay": 0.999, "note": "the shadow accumulates in float32; the closed form is float64, so a few float32 ulps of drift are expected"},
        "step_counts": {"ready": int(processed.indices.size), "batch_size": 8, "epochs": 5, "production_steps": result.optimizer_steps, "oracle_steps": steps["optimizer_steps"], "ema_updates": result.ema_updates},
        "checkpoint_round_trip": device_checks,
        "tolerances": tolerance,
        "passed": bool(passed),
    }


def stage_verify(reports: Path) -> dict:
    from stratego.training.phase18.coverage import attach_test_outcomes, verify_coverage

    contract, landscape_document, contract_sha, landscape_sha = load_frozen(reports)
    design, landscape = verify_frozen_identity(contract, landscape_document)
    g2 = reports / G2_DIRECTORY
    log("running the pre-existing Phase 18 evaluator suite")
    evaluator = run_pytest(EVALUATOR_SUITE, g2 / "junit_evaluator_suite.xml")
    log(f"evaluator suite: {evaluator['summary_line']}")
    log("running the Phase 18 setup tests")
    setup = run_pytest(SETUP_SUITE, g2 / "junit_setup_suite.xml")
    log(f"setup suite: {setup['summary_line']}")
    log("running the canned parity oracle")
    oracle_record = run_oracle(design)
    log(f"oracle: {'PASS' if oracle_record['passed'] else 'FAIL'}")
    coverage = attach_test_outcomes(verify_coverage(REPOSITORY_ROOT), setup["outcomes"])
    coverage.update({
        "artifact": "phase18_g2_parity_coverage_v1",
        "method_map": METHOD_MAP,
        "method_map_sha256": file_sha256(REPOSITORY_ROOT / METHOD_MAP),
        "recorded_run": {"junit": setup["junit"], "junit_sha256": setup["junit_sha256"], "counts": setup["counts"]},
        "rule": "a row is complete only when every cited test passed in the recorded run; documentation alone never completes a row",
        "timestamp_utc": utc_now(),
    })
    write_json(g2 / "phase18_g2_parity_coverage_v1.json", coverage)
    write_json(g2 / "phase18_g2_parity_oracle_v1.json", oracle_record)
    record = {
        "artifact": "phase18_g2_verification_v1",
        "contract_sha256": contract_sha,
        "landscape_sha256": landscape_sha,
        "evaluator_suite": {k: v for k, v in evaluator.items() if k != "outcomes"},
        "setup_suite": {k: v for k, v in setup.items() if k != "outcomes"},
        "oracle_passed": oracle_record["passed"],
        "coverage_all_rows_complete": coverage["all_g2_rows_complete"],
        "coverage_rows_complete": coverage["rows_complete"],
        "coverage_problems": coverage["problems"],
        "environment": environment(),
        "timestamp_utc": utc_now(),
    }
    write_json(g2 / "phase18_g2_verification_v1.json", record)
    log(f"coverage: {coverage['rows_complete']}/{coverage['rows_total']} rows complete")
    return record


# ---------------------------------------------------------------------------
# Stage 3: launch manifest (from the clean execution worktree)
# ---------------------------------------------------------------------------


def stage_launch_manifest(reports: Path, *, source_commit: str) -> dict:
    porcelain = git_output("status", "--porcelain")
    if porcelain:
        raise G2Error(f"BLOCKED: the execution worktree is not clean:\n{porcelain}")
    head = git_output("rev-parse", "HEAD")
    if head != source_commit:
        raise G2Error(f"BLOCKED: HEAD {head} is not G2_SOURCE_COMMIT {source_commit}")
    contract, landscape_document, contract_sha, landscape_sha = load_frozen(reports)
    design, landscape = verify_frozen_identity(contract, landscape_document)
    ignored = subprocess.run(["git", "check-ignore", "-q", str(ARTIFACT_RELATIVE / "probe")], cwd=CANONICAL_ROOT, capture_output=True).returncode == 0
    if not ignored:
        raise G2Error(f"BLOCKED: {ARTIFACT_RELATIVE} is not git-ignored in the canonical tree")
    if ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.iterdir()):
        raise G2Error(f"BLOCKED: the artifact root {ARTIFACT_ROOT} already exists and is not empty")
    manifest = {
        "artifact": "phase18_g2_launch_manifest_v1",
        "work_package": WORK_PACKAGE,
        "agent": AGENT,
        "gate": GATE,
        "run_id": RUN_ID,
        "timestamp_utc": utc_now(),
        "authorization": digests_of(AUTHORIZATION_FILES),
        "source": {
            "g2_source_commit": source_commit,
            "g2_source_tree": git_output("rev-parse", f"{source_commit}^{{tree}}"),
            "execution_worktree": str(REPOSITORY_ROOT),
            "expected_execution_worktree": str(EXECUTION_WORKTREE),
            "worktree_porcelain_empty": True,
            "canonical_tree": str(CANONICAL_ROOT),
        },
        "artifacts": {
            "root_absolute": str(ARTIFACT_ROOT),
            "root_relative": str(ARTIFACT_RELATIVE),
            "git_ignored": ignored,
            "g1_artifacts": "untouched; they live outside the repository under /Users/brandonwashington/Dev/stratego_phase18",
        },
        "contract_sha256": contract_sha,
        "landscape_sha256": landscape_sha,
        "landscape_table_digest": landscape_document["table_digest"],
        "design_digest": json_document_digest(design.document()),
        "seeds": {
            "namespace": NAMESPACE,
            "seed_function": "stratego.setups.identity.derive_stream_seed",
            "model_seeds": {str(k): design.model_seed(k) for k in design.seed_indices},
            "landscape_table_seed": design.landscape_table_seed(),
            "bootstrap_seed": design.bootstrap_seed(),
        },
        "budget": {"updates": design.updates, "seeds": list(design.seed_indices), "pool_size": design.pool_size, "outcomes_per_setup": design.outcomes_per_setup, "evaluation_samples": design.evaluation_samples, "bootstrap_replicates": design.bootstrap_replicates, "gap_closure_threshold": design.gap_closure_threshold},
        "source_digests": digests_of(SOURCE_FILES),
        "test_digests": digests_of(TEST_FILES),
        "environment": environment(),
        "sealed_test_access": {"planned": 0},
        "stratego_games_planned": 0,
    }
    write_json(reports / LAUNCH_NAME, manifest)
    log(f"launch manifest bound to {source_commit[:12]} under {reports}")
    return manifest


# ---------------------------------------------------------------------------
# Stage 4: run one seed
# ---------------------------------------------------------------------------


def stage_run(reports: Path, *, seed_index: int, artifact_root: Path) -> dict:
    contract, landscape_document, contract_sha, landscape_sha = load_frozen(reports)
    design, landscape = verify_frozen_identity(contract, landscape_document)
    launch_path = reports / LAUNCH_NAME
    if not launch_path.exists():
        raise G2Error("BLOCKED: no launch manifest; run --launch-manifest from the clean worktree first")
    launch = json.loads(launch_path.read_text())
    if launch["contract_sha256"] != contract_sha or launch["landscape_sha256"] != landscape_sha:
        raise G2Error("BLOCKED: the launch manifest binds a different contract or landscape")
    if seed_index not in design.seed_indices:
        raise G2Error(f"BLOCKED: seed index {seed_index} is not frozen")
    output = artifact_root / f"seed_{seed_index}"
    if output.exists():
        raise G2Error(f"BLOCKED: {output} already exists; a seed is never rerun or overwritten")
    log(f"seed {seed_index}: {design.updates} updates, pool {design.pool_size}, {design.outcomes_per_setup} outcomes per setup, {design.evaluation_samples} held-out samples")
    record = run_seed(design, landscape, seed_index, output, log=log)
    record["contract_sha256"] = contract_sha
    record["landscape_sha256"] = landscape_sha
    record["g2_source_commit"] = launch["source"]["g2_source_commit"]
    record["environment"] = environment()
    write_json(output / "seed_result.json", record)
    write_json(reports / G2_DIRECTORY / f"phase18_g2_seed_{seed_index}_result_v1.json", record)
    log(f"seed {seed_index}: EMA {record['initial']['mean_utility']:.4f} -> {record['final']['mean_utility']:.4f} (gap closed {record['gap']['fraction_closed']:+.4%}); raw {record['raw_diagnostic']['initial']['mean_utility']:.4f} -> {record['raw_diagnostic']['final']['mean_utility']:.4f} (gap closed {record['raw_diagnostic']['gap']['fraction_closed']:+.4%})")
    return record


# ---------------------------------------------------------------------------
# Stage 5: analyse
# ---------------------------------------------------------------------------


def _criteria(design: AssayDesign, seeds: dict, arrays: dict, *, seed_offset: int) -> dict:
    """The three frozen criteria on one pair of endpoint arrays (EMA or raw)."""
    import numpy as np

    from stratego.evaluation.phase18.noninferiority import paired_unit_delta

    initial_all = np.concatenate([arrays[k]["initial"] for k in design.seed_indices])
    final_all = np.concatenate([arrays[k]["final"] for k in design.seed_indices])
    pooled = paired_unit_delta(final_all, initial_all, seed=design.bootstrap_seed() + seed_offset, replicates=design.bootstrap_replicates, confidence=design.bootstrap_confidence)
    per_seed = {}
    for k in design.seed_indices:
        interval = paired_unit_delta(arrays[k]["final"], arrays[k]["initial"], seed=design.bootstrap_seed() + seed_offset + 1000 * k, replicates=design.bootstrap_replicates, confidence=design.bootstrap_confidence)
        per_seed[str(k)] = {"interval": interval.to_dict(), "gap_fraction_closed": seeds[k]["gap_fraction"], "initial": seeds[k]["initial"], "final": seeds[k]["final"], "improved": seeds[k]["final"] > seeds[k]["initial"]}
    fractions = sorted(seeds[k]["gap_fraction"] for k in design.seed_indices)
    median = float(np.median(fractions))
    return {
        "all_seeds_improved": all(seeds[k]["final"] > seeds[k]["initial"] for k in design.seed_indices),
        "pooled_paired_interval": pooled.to_dict(),
        "pooled_lower_bound_strictly_above_zero": bool(pooled.lower > 0.0),
        "median_gap_fraction_closed": median,
        "median_gap_closure_meets_threshold": bool(median >= design.gap_closure_threshold),
        "per_seed": per_seed,
        "gap_fractions_sorted": fractions,
    }


def stage_analyse(reports: Path, *, artifact_root: Path) -> dict:
    import numpy as np

    contract, landscape_document, contract_sha, landscape_sha = load_frozen(reports)
    design, landscape = verify_frozen_identity(contract, landscape_document)
    launch = json.loads((reports / LAUNCH_NAME).read_text())
    verification = json.loads((reports / G2_DIRECTORY / "phase18_g2_verification_v1.json").read_text())
    coverage = json.loads((reports / G2_DIRECTORY / "phase18_g2_parity_coverage_v1.json").read_text())
    oracle_record = json.loads((reports / G2_DIRECTORY / "phase18_g2_parity_oracle_v1.json").read_text())

    seeds: dict = {}
    arrays_ema: dict = {}
    arrays_raw: dict = {}
    integrity_total: dict = {}
    source_commits = set()
    for k in design.seed_indices:
        directory = artifact_root / f"seed_{k}"
        record = json.loads((directory / "seed_result.json").read_text())
        if record["contract_sha256"] != contract_sha or record["landscape_sha256"] != landscape_sha:
            raise G2Error(f"BLOCKED: seed {k} ran under a different contract or landscape")
        if record["updates"] != design.updates or record["optimizer_steps"] != design.updates * design.epochs_per_update * -(-design.pool_size // design.batch_size):
            raise G2Error(f"BLOCKED: seed {k} did not complete the frozen budget")
        for name in ("initial", "final", "initial_raw", "final_raw"):
            path = Path(record["utilities"][name]["path"])
            if file_sha256(path) != record["utilities"][name]["sha256"]:
                raise G2Error(f"BLOCKED: seed {k} utilities {name} digest moved")
        arrays_ema[k] = {"initial": np.load(record["utilities"]["initial"]["path"]), "final": np.load(record["utilities"]["final"]["path"])}
        arrays_raw[k] = {"initial": np.load(record["utilities"]["initial_raw"]["path"]), "final": np.load(record["utilities"]["final_raw"]["path"])}
        for name, array in (("initial", arrays_ema[k]["initial"]), ("final", arrays_ema[k]["final"])):
            if abs(float(array.mean()) - record[name]["mean_utility"]) > 1e-9:
                raise G2Error(f"BLOCKED: seed {k} {name} array does not reproduce the recorded mean")
        if arrays_ema[k]["initial"].size != design.evaluation_samples:
            raise G2Error(f"BLOCKED: seed {k} evaluated {arrays_ema[k]['initial'].size} samples, not {design.evaluation_samples}")
        source_commits.add(record["g2_source_commit"])
        for key, value in record["integrity"].items():
            integrity_total[key] = integrity_total.get(key, 0) + int(value)
        seeds[k] = {
            "ema": {"initial": record["initial"]["mean_utility"], "final": record["final"]["mean_utility"], "gap_fraction": record["gap"]["fraction_closed"]},
            "raw": {"initial": record["raw_diagnostic"]["initial"]["mean_utility"], "final": record["raw_diagnostic"]["final"]["mean_utility"], "gap_fraction": record["raw_diagnostic"]["gap"]["fraction_closed"]},
            "curve": record["curve"],
            "integrity": record["integrity"],
            "wall_seconds": record["wall_seconds"],
            "checkpoint": record["checkpoint"],
            "period_outcome_digests_sha256": hashlib.sha256(json.dumps(record["period_outcome_digests"]).encode()).hexdigest(),
            "receipts_sha256": record["outcome_receipts"]["sha256"],
            "telemetry_sha256": record["telemetry"]["sha256"],
            "initial_raw_digest": record["initial_raw_digest"],
            "final_ema_digest": record["final"]["ema_digest"],
        }
    if len(source_commits) != 1 or next(iter(source_commits)) != launch["source"]["g2_source_commit"]:
        raise G2Error(f"BLOCKED: the seeds bind different source commits: {source_commits}")

    ema = _criteria(design, {k: seeds[k]["ema"] for k in seeds}, arrays_ema, seed_offset=0)
    raw = _criteria(design, {k: seeds[k]["raw"] for k in seeds}, arrays_raw, seed_offset=7)
    integrity_failures = {k: v for k, v in integrity_total.items() if k in ("legality_failures", "orientation_failures", "attribution_failures", "non_finite_events", "checkpoint_identity_failures")}
    zero_integrity = all(v == 0 for v in integrity_failures.values())
    parity = bool(coverage["all_g2_rows_complete"] and oracle_record["passed"] and verification["evaluator_suite"]["return_code"] == 0 and verification["setup_suite"]["return_code"] == 0)
    ema_pass = ema["all_seeds_improved"] and ema["pooled_lower_bound_strictly_above_zero"] and ema["median_gap_closure_meets_threshold"]
    raw_pass = raw["all_seeds_improved"] and raw["pooled_lower_bound_strictly_above_zero"] and raw["median_gap_closure_meets_threshold"]
    if not parity or not zero_integrity:
        decision = "REVISE"
        basis = "a parity row, the oracle, a suite or an integrity counter failed"
    elif ema_pass:
        decision = "PROCEED"
        basis = "the frozen EMA criteria pass"
    elif raw_pass:
        decision = "REVISE"
        basis = "predeclared: the EMA criteria fail while the raw diagnostic satisfies the same three criteria - instrument defect 'EMA horizon exceeds the update budget'"
    else:
        decision = "STOP"
        basis = "the learner does not satisfy the criteria on the EMA or on the raw diagnostic"

    results = {
        "artifact": "phase18_g2_results_v1",
        "work_package": WORK_PACKAGE,
        "agent": AGENT,
        "gate": GATE,
        "run_id": RUN_ID,
        "timestamp_utc": utc_now(),
        "g2_source_commit": launch["source"]["g2_source_commit"],
        "contract_sha256": contract_sha,
        "landscape_sha256": landscape_sha,
        "launch_manifest_sha256": file_sha256(reports / LAUNCH_NAME),
        "landscape": {"exact_optimum": landscape.optimum, "uniform_mean": landscape.uniform_mean, "uniform_sd": landscape.uniform_sd, "table_digest": landscape_document["table_digest"]},
        "design": {"seeds": list(design.seed_indices), "updates": design.updates, "pool_size": design.pool_size, "outcomes_per_setup": design.outcomes_per_setup, "evaluation_samples": design.evaluation_samples, "bootstrap_replicates": design.bootstrap_replicates, "gap_closure_threshold": design.gap_closure_threshold, "ema_retained_initial_fraction": SETUP_EMA_DECAY ** design.updates},
        "parity": {"coverage_rows_complete": coverage["rows_complete"], "coverage_rows_total": coverage["rows_total"], "all_rows_complete": coverage["all_g2_rows_complete"], "oracle_passed": oracle_record["passed"], "evaluator_suite": verification["evaluator_suite"]["counts"], "setup_suite": verification["setup_suite"]["counts"], "passed": parity},
        "integrity": {"totals": integrity_total, "failures": integrity_failures, "zero_failures": zero_integrity},
        "seeds": {str(k): seeds[k] for k in seeds},
        "ema_criteria": ema,
        "raw_diagnostic_criteria": {"role": "DIAGNOSTIC ONLY - never decides PROCEED", **raw},
        "criteria_summary": {"parity": parity, "zero_integrity_failures": zero_integrity, "ema_all_seeds_improved": ema["all_seeds_improved"], "ema_pooled_lower_bound_above_zero": ema["pooled_lower_bound_strictly_above_zero"], "ema_median_gap_closure_meets_threshold": ema["median_gap_closure_meets_threshold"], "ema_criteria_pass": ema_pass, "raw_diagnostic_criteria_pass": raw_pass},
        "decision_input": {"decision": decision, "basis": basis, "rule": contract["decision_rules"], "predeclared_interpretation": contract["predeclared_instrument_finding"]["predeclared_interpretation"]},
        "sealed_test_access": {"examples_opened": 0, "multiplicity_increment": 0},
        "stratego_games_played": 0,
        "environment": environment(),
    }
    write_json(reports / RESULTS_NAME, results)
    log(f"EMA: all improved {ema['all_seeds_improved']}, lower {ema['pooled_paired_interval']['lower']:+.5f}, median gap {ema['median_gap_fraction_closed']:+.4%}; raw: all improved {raw['all_seeds_improved']}, lower {raw['pooled_paired_interval']['lower']:+.5f}, median gap {raw['median_gap_fraction_closed']:+.4%} -> {decision}")
    return results


# ---------------------------------------------------------------------------
# Stage 6: bind
# ---------------------------------------------------------------------------


def stage_bind(reports: Path) -> dict:
    launch = json.loads((reports / LAUNCH_NAME).read_text())
    commit = launch["source"]["g2_source_commit"]
    entries: dict = {}
    mismatched = []
    names = [CONTRACT_NAME, LANDSCAPE_NAME, LAUNCH_NAME, RESULTS_NAME] + sorted(p.name for p in (reports / G2_DIRECTORY).glob("*.json"))
    for name in names:
        path = reports / name if (reports / name).exists() else reports / G2_DIRECTORY / name
        payload = json.loads(path.read_text())
        bound = payload.get("g2_source_commit") or (payload.get("source") or {}).get("g2_source_commit")
        form = "full" if bound else "pre-commit artifact; bound by this ledger's digest and the launch manifest"
        agrees = (bound == commit) if bound else None
        if bound and not agrees:
            mismatched.append(name)
        entries[str(path.relative_to(reports))] = {"sha256": file_sha256(path), "bytes": path.stat().st_size, "source_binding": {"form": form, "value": bound, "agrees": agrees}}
    ledger = {
        "artifact": "phase18_g2_binding_v1",
        "work_package": WORK_PACKAGE,
        "agent": AGENT,
        "gate": GATE,
        "run_id": RUN_ID,
        "g2_source_commit": commit,
        "g2_source_tree": launch["source"]["g2_source_tree"],
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
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--freeze", action="store_true")
    stage.add_argument("--verify", action="store_true")
    stage.add_argument("--launch-manifest", action="store_true")
    stage.add_argument("--run", action="store_true")
    stage.add_argument("--analyse", action="store_true")
    stage.add_argument("--bind", action="store_true")
    parser.add_argument("--reports", type=Path, default=CANONICAL_ROOT / "reports" / "phase18")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--seed-index", type=int)
    parser.add_argument("--source-commit")
    arguments = parser.parse_args(argv)
    try:
        if arguments.freeze:
            stage_freeze(arguments.reports)
        elif arguments.verify:
            stage_verify(arguments.reports)
        elif arguments.launch_manifest:
            if not arguments.source_commit:
                parser.error("--launch-manifest needs --source-commit")
            stage_launch_manifest(arguments.reports, source_commit=arguments.source_commit)
        elif arguments.run:
            if arguments.seed_index is None:
                parser.error("--run needs --seed-index")
            stage_run(arguments.reports, seed_index=arguments.seed_index, artifact_root=arguments.artifacts)
        elif arguments.analyse:
            stage_analyse(arguments.reports, artifact_root=arguments.artifacts)
        elif arguments.bind:
            stage_bind(arguments.reports)
    except G2Error as error:
        log(str(error))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
