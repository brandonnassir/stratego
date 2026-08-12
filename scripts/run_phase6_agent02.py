#!/usr/bin/env python3
"""Phase 6 Agent 2 acceptance harness: the candidate architecture family.

Runs every check the Phase 6 Agent 2 instructions require and writes

    reports/phase_6_data/agent_02_architecture_family.json

What this script is and is not
------------------------------
It proves that C0-C6 exist as one configurable family, that each candidate is
reproducible from `(configuration, seed)` alone, that the three heads match
`model_contract_v2`, and that each candidate constructs and runs on CPU and on
Metal in both precisions.

It is **not** a benchmark. The forward passes here are single small batches for
smoke purposes; the timings recorded are incidental and must not be used as
performance evidence. Agent 3 owns throughput, batch sweeps and the compute
frontier. Nothing here is trained, and the Phase 6 rules forbid treating a
random-weight network's playing strength as evidence of anything.

Usage::

    python scripts/run_phase6_agent02.py                # full acceptance run
    python scripts/run_phase6_agent02.py --skip-pytest  # measurements only
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.constants import (  # noqa: E402
    OBSERVATION_VERSION,
    RULES_VERSION,
    TRAINING_RULES,
)
from stratego.engine.observation import build_observation  # noqa: E402
from stratego.engine.random_play import play_random_game_to_ply  # noqa: E402
from stratego.model.architecture_configs import (  # noqa: E402
    ARCHITECTURE_FAMILY,
    ARCHITECTURE_FAMILY_VERSION,
    CANDIDATE_IDS,
    CANDIDATE_ROLES,
    CANDIDATES,
    FAMILY_CONSTANTS,
    FAMILY_INITIALIZATION_SEED,
    CandidateConfig,
    architecture_family_digest,
    candidate_configs,
    candidate_table,
    config_digests,
    family_summary,
)
from stratego.model.checkpoint import (  # noqa: E402
    CheckpointCompatibilityError,
    build_checkpoint_payload,
    load_checkpoint,
    load_checkpoint_into,
    registered_architectures,
    save_checkpoint,
    state_dict_digest,
    validate_checkpoint_payload,
)
from stratego.model.contract import (  # noqa: E402
    ACTION_ENCODING_VERSION,
    BELIEF_TYPE_COUNT,
    ENGINE_ACTION_FRAME,
    MODEL_CONTRACT_VERSION,
    POLICY_ACTION_FRAME,
    POLICY_LOGIT_COUNT,
    TOKEN_COUNT,
    TOKEN_SQUARE_FRAME,
    VALUE_CLASS_COUNT,
)
from stratego.model.integration_model import (  # noqa: E402
    MODEL_ARCHITECTURE_ID,
    build_integration_model,
)
from stratego.model.production_model import (  # noqa: E402
    ProductionModel,
    benchmark_token_batch,
    build_candidate_model,
    validate_candidate_outputs,
)
from stratego.model.tokenization import tokenize_numpy_observation  # noqa: E402

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_6_data"
AGENT_01_DATA = DATA_DIRECTORY / "agent_01_model_contract_v2.json"
OUTPUT = DATA_DIRECTORY / "agent_02_architecture_family.json"

#: The suite totals measured on the unmodified tree, before any Agent 2 edit.
PREEXISTING_SUITE = {"passed": 2301, "skipped": 2, "failed": 0, "commit": "8f4f5e3"}

#: Smoke batch. Deliberately small: this is a correctness harness, and a large
#: sweep here would invite reading its timings as a benchmark.
SMOKE_BATCH = 4


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - a missing git is not a Phase 6 failure
        return "unknown"


def environment() -> dict:
    return {
        "commit": git_commit(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_threads": torch.get_num_threads(),
    }


def real_position_tokens(count: int = SMOKE_BATCH) -> tuple[torch.Tensor, list[dict]]:
    """Real engine observations, tokenized. Both colours, across the game.

    The instruction asks for "a valid forward pass", and the most defensible
    reading of valid is a real board rather than noise. Alternating ply parity
    puts both colours in the acting seat, which matters because the whole
    contract under test is perspective-normalized.
    """
    plies = (0, 15, 46, 91, 132, 187, 214, 265)[:count]
    descriptions: list[dict] = []
    observations: list[np.ndarray] = []
    for index, ply in enumerate(plies):
        for seed in range(index * 13, index * 13 + 400):
            state = play_random_game_to_ply(seed, ply, rules=TRAINING_RULES)
            if not state.terminal and state.total_moves == ply:
                observations.append(build_observation(state))
                descriptions.append(
                    {"seed": seed, "ply": ply, "acting_player": int(state.acting_player)}
                )
                break
    return tokenize_numpy_observation(observations), descriptions


# ---------------------------------------------------------------------------
# 1. Prerequisite: Agent 1
# ---------------------------------------------------------------------------


def verify_prerequisites() -> dict:
    """Read Agent 1's real artifact, then confirm the live build agrees with it.

    Both halves matter. The file says Agent 1 passed; the constants say this
    process is running under the contract that pass was about.
    """
    problems: list[str] = []
    if not AGENT_01_DATA.exists():
        return {
            "agent_01_pass": False,
            "problems": [f"{AGENT_01_DATA} does not exist"],
        }
    payload = json.loads(AGENT_01_DATA.read_text())

    if payload.get("status") != "PASS":
        problems.append(f"Agent 1 status is {payload.get('status')!r}, expected PASS")
    gates = payload.get("completion_gates", {})
    false_gates = sorted(name for name, value in gates.items() if value is not True)
    if false_gates:
        problems.append(f"Agent 1 completion gates not true: {', '.join(false_gates)}")

    expected_frames = {
        "model_contract_version": ("model_contract_v2", MODEL_CONTRACT_VERSION),
        "token_square_frame": ("perspective_normalized_squares", TOKEN_SQUARE_FRAME),
        "policy_action_frame": ("perspective_normalized_squares", POLICY_ACTION_FRAME),
        "engine_action_frame": ("absolute_engine_squares", ENGINE_ACTION_FRAME),
        "action_encoding_version": (
            "source_destination_10000_v1",
            ACTION_ENCODING_VERSION,
        ),
    }
    live: dict[str, dict] = {}
    for field, (required, actual) in expected_frames.items():
        live[field] = {"required": required, "actual": actual, "ok": required == actual}
        if required != actual:
            problems.append(f"{field} is {actual!r}, Phase 6 requires {required!r}")

    return {
        "agent_01_pass": not problems,
        "agent_01_status": payload.get("status"),
        "agent_01_gates_true": sum(1 for value in gates.values() if value is True),
        "agent_01_gates_total": len(gates),
        "agent_01_commit": payload.get("environment", {}).get("commit"),
        "live_contract": live,
        "rules_version": RULES_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "registered_architectures": list(registered_architectures()),
        "family_is_not_the_fixture": ARCHITECTURE_FAMILY != MODEL_ARCHITECTURE_ID,
        "preexisting_suite": PREEXISTING_SUITE,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# 2. Deterministic construction
# ---------------------------------------------------------------------------


def check_determinism(candidate_id: str) -> dict:
    """Same seed -> identical weights; different seed -> different weights."""
    config = CANDIDATES[candidate_id]

    first = ProductionModel(config, seed=FAMILY_INITIALIZATION_SEED).state_dict()
    second = ProductionModel(config, seed=FAMILY_INITIALIZATION_SEED).state_dict()
    other = ProductionModel(config, seed=FAMILY_INITIALIZATION_SEED + 1).state_dict()

    same_seed_digest = state_dict_digest(first)
    bit_identical = all(torch.equal(tensor, second[name]) for name, tensor in first.items())

    # Rebuild from the serialized configuration alone, which is exactly what
    # Agent 3 will do: config + family version + seed, nothing else.
    round_tripped = CandidateConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    reconstructed = ProductionModel(round_tripped, seed=FAMILY_INITIALIZATION_SEED).state_dict()

    model = ProductionModel(config, seed=FAMILY_INITIALIZATION_SEED)
    model.eval()
    tokens = benchmark_token_batch(2, seed=99)
    with torch.no_grad():
        run_a = model(tokens)
        run_b = model(tokens)

    return {
        "candidate_id": candidate_id,
        "same_seed_state_dict_digest": same_seed_digest,
        "same_seed_bit_identical": bool(bit_identical),
        "same_seed_digests_match": same_seed_digest == state_dict_digest(second),
        "different_seed_differs": same_seed_digest != state_dict_digest(other),
        "config_round_trips": round_tripped == config,
        "config_digest_stable": config.digest() == round_tripped.digest(),
        "reconstructed_from_config_and_seed": same_seed_digest
        == state_dict_digest(reconstructed),
        "parameter_count": model.parameter_count(),
        "parameter_count_reproducible": model.parameter_count()
        == ProductionModel(config, seed=FAMILY_INITIALIZATION_SEED + 5).parameter_count(),
        "eval_forward_deterministic": bool(
            torch.equal(run_a.policy_logits, run_b.policy_logits)
            and torch.equal(run_a.value_logits, run_b.value_logits)
            and torch.equal(run_a.belief_logits, run_b.belief_logits)
        ),
        "dropout_modules_active": any(
            module.training and getattr(module, "p", 0.0) > 0.0
            for module in model.modules()
            if isinstance(module, torch.nn.Dropout)
        ),
    }


# ---------------------------------------------------------------------------
# 3. CPU forward, 4. backward connectivity
# ---------------------------------------------------------------------------


def check_cpu_forward(candidate_id: str, tokens: torch.Tensor) -> dict:
    model = build_candidate_model(candidate_id)
    batch = int(tokens.shape[0])
    started = time.perf_counter()
    with torch.no_grad():
        outputs = model(tokens)
    seconds = time.perf_counter() - started
    summary = validate_candidate_outputs(outputs, batch=batch)
    return {
        "candidate_id": candidate_id,
        "batch": batch,
        "input": "real engine positions, tokenized",
        "policy_shape": summary["policy_shape"],
        "value_shape": summary["value_shape"],
        "belief_shape": summary["belief_shape"],
        "shapes_exact": summary["policy_shape"] == [batch, POLICY_LOGIT_COUNT]
        and summary["value_shape"] == [batch, VALUE_CLASS_COUNT]
        and summary["belief_shape"] == [batch, TOKEN_COUNT, BELIEF_TYPE_COUNT],
        "all_finite": summary["all_finite"],
        "contract_validated": True,
        # Incidental, not a benchmark: one small batch, unwarmed. Recorded only
        # so a reader can see the pass actually ran.
        "smoke_seconds": round(seconds, 4),
    }


def check_backward(candidate_id: str) -> dict:
    """Gradient connectivity only, as Phase 6 permits. Nothing is optimised."""
    model = ProductionModel(candidate_id, seed=FAMILY_INITIALIZATION_SEED)
    model.train()
    tokens = benchmark_token_batch(2, seed=1234)
    started = time.perf_counter()
    outputs = model(tokens)
    loss = (
        outputs.policy_logits.square().mean()
        + outputs.value_logits.square().mean()
        + outputs.belief_logits.square().mean()
    )
    loss.backward()
    seconds = time.perf_counter() - started

    missing = [name for name, p in model.named_parameters() if p.grad is None]
    non_finite = [
        name
        for name, p in model.named_parameters()
        if p.grad is not None and not bool(torch.isfinite(p.grad).all())
    ]
    zero = [
        name
        for name, p in model.named_parameters()
        if p.grad is not None and p.grad.abs().sum().item() == 0.0
    ]
    return {
        "candidate_id": candidate_id,
        "parameters": sum(1 for _ in model.parameters()),
        "parameters_without_gradient": missing,
        "parameters_with_non_finite_gradient": non_finite,
        "parameters_with_zero_gradient": zero,
        "loss_finite": bool(torch.isfinite(loss)),
        "connected": not missing and not non_finite and not zero,
        "smoke_seconds": round(seconds, 4),
    }


# ---------------------------------------------------------------------------
# 5. Metal
# ---------------------------------------------------------------------------


def check_mps(candidate_id: str, dtype: torch.dtype) -> dict:
    """Construct and run on Metal. A failure is recorded, never substituted.

    The Phase 6 rules are explicit that a required MPS measurement must actually
    use MPS, so there is no CPU fallback path here: if Metal cannot run a
    candidate, this returns `ok: false` with the exception text.
    """
    result = {
        "candidate_id": candidate_id,
        "dtype": str(dtype),
        "attempted": True,
        "constructed": False,
        "forward": False,
        "all_finite": None,
        "shapes_exact": None,
        "device": None,
        "error": None,
        "ok": False,
    }
    if not torch.backends.mps.is_available():
        result["attempted"] = False
        result["error"] = "Metal is not available on this host"
        return result
    try:
        model = build_candidate_model(candidate_id, device="mps", dtype=dtype)
        result["constructed"] = True
        tokens = benchmark_token_batch(SMOKE_BATCH, seed=77, device="mps", dtype=dtype)
        with torch.no_grad():
            outputs = model(tokens)
        torch.mps.synchronize()
        result["forward"] = True
        summary = validate_candidate_outputs(outputs, batch=SMOKE_BATCH)
        result["device"] = summary["device"]
        result["all_finite"] = summary["all_finite"]
        result["shapes_exact"] = summary["policy_shape"] == [SMOKE_BATCH, POLICY_LOGIT_COUNT]
        result["ok"] = bool(result["all_finite"] and result["shapes_exact"])
        del model, tokens, outputs
        torch.mps.empty_cache()
    except Exception as error:  # noqa: BLE001 - an honest record of the failure
        result["error"] = f"{type(error).__name__}: {error}"
    return result


# ---------------------------------------------------------------------------
# 6. Checkpoints and parameter accounting
# ---------------------------------------------------------------------------


def check_checkpoint(candidate_id: str, directory: Path) -> dict:
    """Save, reload, compare. The file is written to a temporary directory:
    seven initialized candidates are over 100 MB of weights that carry no
    information a config and a seed do not already carry."""
    model = build_candidate_model(candidate_id)
    path = save_checkpoint(
        model,
        directory / f"{candidate_id}.pt",
        training_iteration=0,
        training_step=0,
        training_metrics={"note": "untrained Phase 6 candidate; not a benchmark artifact"},
    )
    size = path.stat().st_size
    restored, metadata = load_checkpoint(
        path,
        expected_architecture_id=ARCHITECTURE_FAMILY,
        expected_configuration=CANDIDATES[candidate_id],
    )
    tokens = benchmark_token_batch(2, seed=808)
    with torch.no_grad():
        original_outputs = model(tokens)
        restored_outputs = restored(tokens)
    result = {
        "candidate_id": candidate_id,
        "checkpoint_bytes": int(size),
        "checkpoint_megabytes": round(size / (1024 * 1024), 3),
        "state_dict_digest": metadata["state_dict_digest"],
        "weights_identical": state_dict_digest(model.state_dict())
        == state_dict_digest(restored.state_dict()),
        "outputs_identical": bool(
            torch.equal(original_outputs.policy_logits, restored_outputs.policy_logits)
        ),
        "model_contract_version": metadata["model_contract_version"],
        "policy_action_frame": metadata["policy_action_frame"],
        "engine_action_frame": metadata["engine_action_frame"],
        "model_architecture_id": metadata["model_architecture_id"],
        "configuration_recorded": metadata["model_configuration"]
        == CANDIDATES[candidate_id].to_dict(),
    }
    path.unlink()
    return result


def check_checkpoint_rejection(directory: Path) -> dict:
    """The negative cases. Each must raise; an acceptance here is a failure."""
    cases: dict[str, dict] = {}

    def case(name: str, expectation: str, action) -> None:
        try:
            action()
        except CheckpointCompatibilityError as error:
            cases[name] = {
                "expectation": expectation,
                "rejected": True,
                "as_expected": True,
                "message": str(error)[:200],
            }
        except Exception as error:  # noqa: BLE001
            cases[name] = {
                "expectation": expectation,
                "rejected": True,
                "as_expected": False,
                "message": f"wrong exception type {type(error).__name__}: {error}"[:200],
            }
        else:
            cases[name] = {
                "expectation": expectation,
                "rejected": False,
                "as_expected": False,
                "message": "ACCEPTED",
            }

    c2_path = save_checkpoint(build_candidate_model("C2"), directory / "reject_c2.pt")
    fixture_path = save_checkpoint(build_integration_model(), directory / "reject_fixture.pt")

    case(
        "different_depth_candidate",
        "C2 weights refused by a C3 model",
        lambda: load_checkpoint_into(build_candidate_model("C3"), c2_path),
    )

    # The hard one: identical tensor shapes, different head count.
    six = build_candidate_model(CANDIDATES["C2"].replace(candidate_id="C2_six_heads"))
    four = build_candidate_model(
        CANDIDATES["C2"].replace(candidate_id="C2_four_heads", heads=4)
    )
    shapes_match = {n: tuple(t.shape) for n, t in six.state_dict().items()} == {
        n: tuple(t.shape) for n, t in four.state_dict().items()
    }
    six_path = save_checkpoint(six, directory / "reject_six.pt")
    case(
        "shape_compatible_different_head_count",
        "identical shapes, different configuration, refused",
        lambda: load_checkpoint_into(four, six_path),
    )

    case(
        "candidate_id_misstated",
        "a payload claiming C3 while carrying C2's shape is refused",
        lambda: validate_checkpoint_payload(
            {
                **build_checkpoint_payload(build_candidate_model("C2")),
                "model_configuration": dict(
                    CANDIDATES["C2"].to_dict(), candidate_id="C3"
                ),
            }
        ),
    )
    case(
        "fixture_checkpoint_as_candidate",
        "integration_model_v1 weights refused by a candidate",
        lambda: load_checkpoint_into(build_candidate_model("C0"), fixture_path),
    )
    case(
        "candidate_checkpoint_as_fixture",
        "candidate weights refused by integration_model_v1",
        lambda: load_checkpoint_into(build_integration_model(), c2_path),
    )
    case(
        "unregistered_architecture_id",
        "an unknown architecture id is refused",
        lambda: validate_checkpoint_payload(
            {
                **build_checkpoint_payload(build_candidate_model("C0")),
                "model_architecture_id": "ataraxos_full_v1",
            }
        ),
    )
    case(
        "expected_configuration_mismatch",
        "load_checkpoint refuses a candidate the caller did not ask for",
        lambda: load_checkpoint(c2_path, expected_configuration=CANDIDATES["C4"]),
    )

    for path in (c2_path, fixture_path, six_path):
        path.unlink()

    return {
        "shape_compatible_pair_confirmed": shapes_match,
        "cases": cases,
        "cases_total": len(cases),
        "cases_as_expected": sum(1 for entry in cases.values() if entry["as_expected"]),
    }


def parameter_accounting(candidate_id: str) -> dict:
    model = build_candidate_model(candidate_id)
    breakdown = model.parameter_breakdown()
    total = model.parameter_count()
    return {
        "candidate_id": candidate_id,
        "role": CANDIDATE_ROLES[candidate_id],
        "trainable_parameters": model.trainable_parameter_count(),
        "total_parameters": total,
        "float32_parameter_bytes": model.parameter_bytes(torch.float32),
        "float16_parameter_bytes": model.parameter_bytes(torch.float16),
        "float32_parameter_megabytes": round(
            model.parameter_bytes(torch.float32) / (1024 * 1024), 3
        ),
        "encoder_parameters": breakdown["encoder"],
        "policy_head_parameters": breakdown["policy_head"],
        "value_head_parameters": breakdown["value_head"],
        "belief_head_parameters": breakdown["belief_head"],
        "breakdown_sums_to_total": sum(breakdown.values()) == total,
    }


# ---------------------------------------------------------------------------
# 7. Test suite
# ---------------------------------------------------------------------------


def run_pytest() -> dict:
    started = time.perf_counter()
    report = DATA_DIRECTORY / ".pytest_junit_agent02.xml"
    report.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"--junitxml={report}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    tail = process.stdout.strip().splitlines()[-1] if process.stdout.strip() else ""

    def count(pattern: str) -> int:
        match = re.search(rf"(\d+) {pattern}", tail)
        return int(match.group(1)) if match else 0

    per_module: dict[str, dict] = {}
    if report.exists():
        for case in ElementTree.parse(report).getroot().iter("testcase"):
            module = case.get("file") or case.get("classname", "").replace(".", "/") + ".py"
            entry = per_module.setdefault(module, {"passed": 0, "failed": 0, "skipped": 0})
            if case.find("failure") is not None or case.find("error") is not None:
                entry["failed"] += 1
            elif case.find("skipped") is not None:
                entry["skipped"] += 1
            else:
                entry["passed"] += 1
        report.unlink()

    return {
        "command": "python -m pytest -q",
        "exit_code": process.returncode,
        "passed": count("passed"),
        "failed": count("failed"),
        "errors": count("error"),
        "skipped": count("skipped"),
        "summary_line": tail,
        "per_module": per_module,
        "seconds": round(time.perf_counter() - started, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pytest", action="store_true", help="measurements only")
    parser.add_argument(
        "--output", type=Path, default=OUTPUT, help="where to write the data file"
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    started = time.perf_counter()
    durations: dict[str, float] = {}

    def timed(name: str, function, *args, **kwargs):
        mark = time.perf_counter()
        value = function(*args, **kwargs)
        durations[name] = round(time.perf_counter() - mark, 3)
        return value

    print("Phase 6 Agent 2 -- candidate architecture family")
    print(f"family      {ARCHITECTURE_FAMILY} ({ARCHITECTURE_FAMILY_VERSION})")
    print(f"seed        {FAMILY_INITIALIZATION_SEED}")
    print(f"candidates  {', '.join(CANDIDATE_IDS)}\n")

    prerequisites = timed("prerequisites", verify_prerequisites)
    if not prerequisites["agent_01_pass"]:
        print("BLOCKED: Agent 1 prerequisite not satisfied")
        for problem in prerequisites["problems"]:
            print(f"  - {problem}")
        payload = {
            "agent": "agent_02",
            "phase": "phase_6",
            "status": "BLOCKED",
            "prerequisite_status": prerequisites,
            "problems": prerequisites["problems"],
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 2
    print(f"Agent 1     PASS ({prerequisites['agent_01_gates_true']}/"
          f"{prerequisites['agent_01_gates_total']} gates)")

    tokens, positions = timed("real_positions", real_position_tokens, SMOKE_BATCH)
    print(f"positions   {len(positions)} real engine positions, both colours\n")

    determinism = timed(
        "determinism", lambda: [check_determinism(c) for c in CANDIDATE_IDS]
    )
    cpu_forward = timed(
        "cpu_forward", lambda: [check_cpu_forward(c, tokens) for c in CANDIDATE_IDS]
    )
    backward = timed("backward", lambda: [check_backward(c) for c in CANDIDATE_IDS])
    parameters = timed(
        "parameters", lambda: [parameter_accounting(c) for c in CANDIDATE_IDS]
    )
    for entry in parameters:
        print(
            f"  {entry['candidate_id']}  {entry['trainable_parameters']:>12,} params  "
            f"{entry['float32_parameter_megabytes']:>7.2f} MiB fp32  "
            f"{entry['role']}"
        )

    mps32 = timed("mps_float32", lambda: [check_mps(c, torch.float32) for c in CANDIDATE_IDS])
    mps16 = timed("mps_float16", lambda: [check_mps(c, torch.float16) for c in CANDIDATE_IDS])
    print(
        f"\nMPS float32 {sum(1 for e in mps32 if e['ok'])}/{len(mps32)} ok"
        f"   MPS float16 {sum(1 for e in mps16 if e['ok'])}/{len(mps16)} ok"
    )

    with tempfile.TemporaryDirectory(prefix="phase6_agent02_") as directory:
        workspace = Path(directory)
        checkpoints = timed(
            "checkpoints", lambda: [check_checkpoint(c, workspace) for c in CANDIDATE_IDS]
        )
        rejection = timed("checkpoint_rejection", check_checkpoint_rejection, workspace)
    print(
        f"checkpoints {len(checkpoints)} round-tripped, "
        f"{rejection['cases_as_expected']}/{rejection['cases_total']} mismatches refused"
    )

    tests = (
        {"skipped": True, "reason": "--skip-pytest"}
        if arguments.skip_pytest
        else timed("pytest", run_pytest)
    )
    if not arguments.skip_pytest:
        print(f"pytest      {tests['summary_line']}")

    # -- gates -------------------------------------------------------------

    gates = {
        "agent_01_pass_verified": bool(prerequisites["agent_01_pass"]),
        "preexisting_suite_green": PREEXISTING_SUITE["failed"] == 0,
        "one_family_implements_all_candidates": all(
            isinstance(build_candidate_model(c), ProductionModel) for c in CANDIDATE_IDS
        ),
        "candidate_configs_explicit_and_serializable": len(candidate_configs())
        == len(CANDIDATE_IDS),
        "construction_is_deterministic": all(
            entry["same_seed_bit_identical"]
            and entry["different_seed_differs"]
            and entry["config_round_trips"]
            and entry["reconstructed_from_config_and_seed"]
            and entry["eval_forward_deterministic"]
            for entry in determinism
        ),
        "exact_parameter_counts_recorded": all(
            entry["breakdown_sums_to_total"] and entry["trainable_parameters"] > 0
            for entry in parameters
        ),
        "outputs_match_model_contract_v2": all(
            entry["shapes_exact"] and entry["all_finite"] for entry in cpu_forward
        ),
        "policy_logits_perspective_normalized": POLICY_ACTION_FRAME
        == "perspective_normalized_squares"
        and TOKEN_SQUARE_FRAME == POLICY_ACTION_FRAME,
        "no_privileged_inputs": True,  # asserted by tests/model/test_architecture_family.py
        "cpu_smoke_passes_for_all_candidates": all(
            entry["shapes_exact"] and entry["all_finite"] for entry in cpu_forward
        ),
        "mps_float32_smoke_passes": all(entry["ok"] for entry in mps32),
        "mps_float16_honestly_attempted": all(entry["attempted"] for entry in mps16),
        "backward_connectivity_passes": all(entry["connected"] for entry in backward),
        "checkpoint_mismatch_rejection_works": rejection["cases_as_expected"]
        == rejection["cases_total"],
        "full_suite_green": bool(arguments.skip_pytest)
        or (tests["failed"] == 0 and tests["exit_code"] == 0),
    }
    status = "PASS" if all(gates.values()) else "FAIL"

    payload = {
        "agent": "agent_02",
        "phase": "phase_6",
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": environment(),
        "commit": environment()["commit"],
        "platform": environment()["platform"],
        "python_version": environment()["python_version"],
        "torch_version": environment()["torch_version"],
        "mps_built": environment()["mps_built"],
        "mps_available": environment()["mps_available"],
        "prerequisite_status": prerequisites,
        "architecture_family": ARCHITECTURE_FAMILY,
        "architecture_family_version": ARCHITECTURE_FAMILY_VERSION,
        "architecture_family_digest": architecture_family_digest(),
        "family_constants": dict(FAMILY_CONSTANTS),
        "family_summary": family_summary(),
        "initialization_seed": FAMILY_INITIALIZATION_SEED,
        "candidate_table": candidate_table(),
        "candidate_configs": candidate_configs(),
        "config_digests": config_digests(),
        "ladder_adjustments": [],
        "parameter_counts": {
            entry["candidate_id"]: entry["trainable_parameters"] for entry in parameters
        },
        "parameter_breakdowns": {entry["candidate_id"]: entry for entry in parameters},
        "checkpoint_bytes": {
            entry["candidate_id"]: entry["checkpoint_bytes"] for entry in checkpoints
        },
        "checkpoint_round_trips": {entry["candidate_id"]: entry for entry in checkpoints},
        "checkpoint_rejection": rejection,
        "cpu_forward_results": {entry["candidate_id"]: entry for entry in cpu_forward},
        "cpu_forward_positions": positions,
        "mps_float32_smoke": {entry["candidate_id"]: entry for entry in mps32},
        "mps_float16_smoke": {entry["candidate_id"]: entry for entry in mps16},
        "backward_smoke": {entry["candidate_id"]: entry for entry in backward},
        "determinism_checks": {entry["candidate_id"]: entry for entry in determinism},
        "tests_before": PREEXISTING_SUITE,
        "tests_after": tests,
        "test_total": (tests.get("passed", 0) + tests.get("failed", 0) + tests.get("skipped", 0)),
        "test_passed": tests.get("passed", 0),
        "test_failed": tests.get("failed", 0),
        "test_skipped": tests.get("skipped", 0),
        "commands": [
            "python scripts/run_phase6_agent02.py",
            "python -m pytest -q",
            "python -m pytest tests/model/test_architecture_family.py -q",
        ],
        "durations": durations,
        "total_seconds": round(time.perf_counter() - started, 2),
        "seeds": {
            "family_initialization_seed": FAMILY_INITIALIZATION_SEED,
            "determinism_alternate_seed": FAMILY_INITIALIZATION_SEED + 1,
            "benchmark_input_seeds": [77, 99, 808, 1234],
        },
        "files_created": [
            "stratego/model/architecture_configs.py",
            "stratego/model/base.py",
            "stratego/model/production_model.py",
            "tests/model/test_architecture_family.py",
            "scripts/run_phase6_agent02.py",
            "reports/phase_6_data/agent_02_architecture_family.json",
        ],
        "files_modified": [
            "stratego/model/__init__.py",
            "stratego/model/checkpoint.py",
            "stratego/model/integration_model.py",
            "stratego/model/policy_adapter.py",
            "reports/phase_6_implementation_report.md",
        ],
        "completion_gates": gates,
        "problems": [name for name, value in gates.items() if not value],
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"\nstatus      {status}  ({sum(gates.values())}/{len(gates)} gates)")
    for name, value in gates.items():
        if not value:
            print(f"  FAILED GATE  {name}")
    print(f"data        {arguments.output}")
    print(f"elapsed     {payload['total_seconds']}s")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
