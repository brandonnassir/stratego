"""The G2 driver: determinism of the freeze, refusal on frozen-identity drift,
the recorded-outcome parser, the decision branches and the binding ledger."""

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DRIVER = REPOSITORY_ROOT / "scripts" / "phase18_g2_setup_parity.py"


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("phase18_g2_setup_parity", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_driver_reaches_no_game_runner_corpus_or_evaluation_bank():
    """No Stratego game and no sealed Phase 8 example is opened: the driver
    imports nothing that could play a game or read the corpus."""
    forbidden = ("match_runner", "neural_worker", "warmstart_dataset", "warmstart_examples", "setup_bank", "corpus", "engine.state", "phase9", "phase17", "rollout", "legal_moves")
    for node in ast.walk(ast.parse(DRIVER.read_text())):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = ast.unparse(node)
            for name in forbidden:
                assert name not in text, text
    source = DRIVER.read_text()
    assert "create_game" not in source and "run_neural_schedule" not in source


def test_the_frozen_design_is_not_reduced_and_the_freeze_is_deterministic(driver, tmp_path):
    design = driver.frozen_design()
    assert design.reduced is False and design.namespace == "phase18_g2_setup_parity_v1"
    first = driver.stage_freeze(tmp_path / "a")
    second = driver.stage_freeze(tmp_path / "b")
    for record in (first, second):
        record.pop("timestamp_utc")
        record.pop("seconds")
    assert first == second
    landscape_a = json.loads((tmp_path / "a" / driver.LANDSCAPE_NAME).read_text())
    landscape_b = json.loads((tmp_path / "b" / driver.LANDSCAPE_NAME).read_text())
    assert landscape_a == landscape_b
    assert landscape_a["exact_optimum"]["certificate"]["certified"]
    assert first["landscape"]["table_digest"] == landscape_a["table_digest"]
    assert first["design"]["model_seeds"] == {str(k): design.model_seed(k) for k in design.seed_indices}
    assert first["decision_rules"].keys() == {"PROCEED", "REVISE", "STOP", "BLOCKED"}
    assert first["predeclared_instrument_finding"]["predeclared_interpretation"]["ema_criteria_pass"].startswith("PROCEED")


def test_verify_frozen_identity_refuses_any_drift(driver, tmp_path):
    driver.stage_freeze(tmp_path)
    contract, landscape_document, _, _ = driver.load_frozen(tmp_path)
    design, landscape = driver.verify_frozen_identity(contract, landscape_document)
    assert design.updates == 64 and landscape.optimum == landscape_document["exact_optimum"]["optimum"]
    drifted = json.loads(json.dumps(contract))
    drifted["design"]["updates"] = 32
    with pytest.raises(driver.G2Error, match="does not re-derive"):
        driver.verify_frozen_identity(drifted, landscape_document)
    drifted_landscape = dict(landscape_document, table_digest="0" * 64)
    with pytest.raises(Exception):
        driver.verify_frozen_identity(contract, drifted_landscape)


def test_parse_junit_keeps_the_worst_case_of_a_parametrised_test(driver, tmp_path):
    xml = tmp_path / "junit.xml"
    xml.write_text(
        '<?xml version="1.0"?><testsuites><testsuite name="pytest">'
        '<testcase classname="tests.training.phase18.test_x" name="test_a[cpu]"/>'
        '<testcase classname="tests.training.phase18.test_x" name="test_a[mps]"><failure message="boom"/></testcase>'
        '<testcase classname="tests.training.phase18.test_x" name="test_b"><skipped message="no mps"/></testcase>'
        '<testcase classname="tests.training.phase18.test_x" name="test_c"/>'
        "</testsuite></testsuites>"
    )
    outcomes = driver.parse_junit(xml)
    assert outcomes == {"test_x.py::test_a": "failed", "test_x.py::test_b": "skipped", "test_x.py::test_c": "passed"}


def test_the_criteria_read_all_three_seeds_the_pooled_lower_bound_and_the_median_gap(driver):
    design = driver.frozen_design()
    rng = np.random.default_rng(0)
    arrays = {k: {"initial": rng.normal(2.0, 5.0, 64), "final": None} for k in design.seed_indices}
    for k in design.seed_indices:
        arrays[k]["final"] = arrays[k]["initial"] + 1.5 + rng.normal(0.0, 0.2, 64)
    seeds = {k: {"initial": float(arrays[k]["initial"].mean()), "final": float(arrays[k]["final"].mean()), "gap_fraction": (0.12, 0.08, 0.15)[i]} for i, k in enumerate(design.seed_indices)}
    verdict = driver._criteria(design, seeds, arrays, seed_offset=0)
    assert verdict["all_seeds_improved"] and verdict["pooled_lower_bound_strictly_above_zero"]
    assert verdict["median_gap_fraction_closed"] == pytest.approx(0.12) and verdict["median_gap_closure_meets_threshold"]
    assert verdict["pooled_paired_interval"]["sample_size"] == 192 and verdict["pooled_paired_interval"]["replicates"] == 10000
    seeds[design.seed_indices[1]]["gap_fraction"] = 0.05
    seeds[design.seed_indices[2]]["gap_fraction"] = 0.05
    assert not driver._criteria(design, seeds, arrays, seed_offset=0)["median_gap_closure_meets_threshold"]
    flat = {k: {"initial": arrays[k]["initial"], "final": arrays[k]["initial"] + rng.normal(0.0, 0.2, 64)} for k in design.seed_indices}
    assert not driver._criteria(design, seeds, flat, seed_offset=0)["pooled_lower_bound_strictly_above_zero"]


def test_the_binding_ledger_flags_a_mismatched_source_commit(driver, tmp_path):
    reports = tmp_path
    (reports / driver.G2_DIRECTORY).mkdir()
    driver.write_json(reports / driver.CONTRACT_NAME, {"artifact": "contract"})
    driver.write_json(reports / driver.LANDSCAPE_NAME, {"artifact": "landscape"})
    driver.write_json(reports / driver.LAUNCH_NAME, {"source": {"g2_source_commit": "a" * 40, "g2_source_tree": "b" * 40}, "artifacts": {}})
    driver.write_json(reports / driver.RESULTS_NAME, {"g2_source_commit": "c" * 40})
    driver.write_json(reports / driver.G2_DIRECTORY / "phase18_g2_seed_1_result_v1.json", {"g2_source_commit": "a" * 40})
    ledger = driver.stage_bind(reports)
    assert ledger["mismatched_artifacts"] == [driver.RESULTS_NAME]
    assert not ledger["all_artifacts_bind_one_source_commit"]
    assert ledger["artifacts"][driver.CONTRACT_NAME]["source_binding"]["agrees"] is None
