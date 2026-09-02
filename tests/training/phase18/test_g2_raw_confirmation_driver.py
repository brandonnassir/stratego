"""The G2 raw-actor confirmation driver: the sealing scan, deterministic
freezing from the reviewed base commit, freshness against every previous
table, method identity against the G2 launch manifest, the independent
certificate check, refusal on frozen-identity drift, the frozen criteria, the
decision rule, sample-count integrity and the binding ledger."""

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DRIVER = REPOSITORY_ROOT / "scripts" / "phase18_g2_raw_confirmation.py"
REFERENCES = REPOSITORY_ROOT / "reports" / "phase18"


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("phase18_g2_raw_confirmation", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frozen(driver, tmp_path_factory):
    reports = tmp_path_factory.mktemp("frozen")
    contract = driver.stage_freeze(reports, references=REFERENCES)
    landscape = json.loads((reports / driver.LANDSCAPE_NAME).read_text())
    return reports, contract, landscape


def test_the_driver_reaches_no_game_runner_corpus_or_evaluation_bank():
    forbidden = ("match_runner", "neural_worker", "warmstart_dataset", "warmstart_examples", "setup_bank", "corpus", "engine.state", "phase9", "phase17", "rollout", "legal_moves")
    for node in ast.walk(ast.parse(DRIVER.read_text())):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = ast.unparse(node)
            for name in forbidden:
                assert name not in text, text
    source = DRIVER.read_text()
    assert "create_game" not in source and "run_neural_schedule" not in source


def test_every_seed_derives_from_the_base_commit_and_the_namespace(driver):
    from stratego.setups.identity import derive_stream_seed

    design = driver.frozen_design()
    assert design.reduced is False
    assert design.namespace == f"{driver.ARTIFACT_NAMESPACE}:{driver.BASE_COMMIT}"
    assert driver.BASE_COMMIT == "6afa13bed355884a3327d2661fd739784260dc2b"
    seeds = driver.seed_derivation(design)
    for k in design.seed_indices:
        assert seeds["model_seeds"][str(k)] == derive_stream_seed(design.namespace, "model_init", str(k))
    assert seeds["landscape_table_seed"] == derive_stream_seed(design.namespace, "landscape_table")
    boots = seeds["bootstrap_seeds"]
    assert boots["raw_pooled"] == derive_stream_seed(design.namespace, "paired_bootstrap")
    assert boots["ema_pooled"] == derive_stream_seed(design.namespace, "paired_bootstrap", "ema", "pooled")
    assert boots["raw_per_seed"]["2"] == derive_stream_seed(design.namespace, "paired_bootstrap", "raw", "seed", "2")
    all_seeds = [boots["raw_pooled"], boots["ema_pooled"], *boots["raw_per_seed"].values(), *boots["ema_per_seed"].values(), *seeds["model_seeds"].values(), seeds["landscape_table_seed"]]
    assert len(set(all_seeds)) == len(all_seeds), "every stream carries a distinct seed"
    # A different base commit changes every seed: the derivation really binds it.
    other = driver.AssayDesign(namespace=f"{driver.ARTIFACT_NAMESPACE}:{'0' * 40}", run_id=driver.RUN_ID)
    assert other.model_seed(1) != design.model_seed(1) and other.landscape_table_seed() != design.landscape_table_seed()


def test_the_freeze_is_deterministic_fresh_and_method_identical_to_g2(driver, frozen, tmp_path):
    reports, contract, landscape = frozen
    again = driver.stage_freeze(tmp_path, references=REFERENCES)
    for record in (contract, again):
        record.pop("timestamp_utc")
        record.pop("seconds")
    assert contract == again
    assert landscape == json.loads((tmp_path / driver.LANDSCAPE_NAME).read_text())
    assert landscape["exact_optimum"]["certificate"]["certified"]
    assert contract["landscape"]["independent_certificate_check"]["certified"]
    assert contract["freshness_audit"]["fresh"] and all(contract["freshness_audit"]["checks"].values())
    g2_landscape = json.loads((REFERENCES / "phase18_g2_synthetic_landscape_v1.json").read_text())
    g2_contract = json.loads((REFERENCES / "phase18_g2_contract_v1.json").read_text())
    assert landscape["table_digest"] != g2_landscape["table_digest"]
    assert set(contract["design"]["model_seeds"].values()).isdisjoint(set(g2_contract["design"]["model_seeds"].values()))
    assert contract["design"]["bootstrap_seed"] != g2_contract["design"]["bootstrap_seed"]
    for ns in ("phase18_g2_dev_smoke_v1", "phase18_g2_dev_smoke_v2", "phase18_g2_setup_parity_v1"):
        assert ns in contract["freshness_audit"]["previous_namespaces"]
    identity = contract["method_identity"]
    assert identity["all_method_files_identical_to_g2"] and identity["design_identical_on_every_method_field"] and identity["method_config_identical"]
    assert identity["design_fields_differing"] == []
    assert identity["training_config_digest_this_run"] != identity["g2_training_config_digest"], "the run id alone separates the two configurations"
    assert contract["decision_rules"].keys() == {"PROCEED", "STOP", "REVISE", "BLOCKED", "ema_results", "scope_of_a_pass"}
    assert contract["question"]["frozen_before_outcomes"] is True
    design = contract["design"]
    assert (design["updates"], design["pool_size"], design["batch_size"], design["epochs_per_update"], design["outcomes_per_setup"], design["evaluation_samples"], design["bootstrap_replicates"], design["gap_closure_threshold"]) == (64, 1024, 1024, 5, 4, 4096, 10000, 0.1)


def test_the_certificate_verifies_from_the_recorded_potentials_and_fails_when_perturbed(driver, frozen):
    reports, contract, landscape = frozen
    table = np.array(landscape["table"], dtype=np.float64)
    potentials = contract["landscape"]["certificate_potentials"]
    check = driver.verify_certificate(table, potentials["u"], potentials["v"], contract["landscape"]["optimal_setup"])
    assert check["certified"] and check["dual_feasibility_violations"] == 0
    assert check["optimal_setup_utility_by_direct_summation"] == pytest.approx(contract["landscape"]["exact_optimum"], abs=1e-9)
    assert check["utility_upper_bound_from_potentials"] == pytest.approx(contract["landscape"]["exact_optimum"], abs=1e-6)
    perturbed = list(potentials["u"])
    perturbed[3] += 0.5
    assert not driver.verify_certificate(table, perturbed, potentials["v"], contract["landscape"]["optimal_setup"])["certified"]
    # A different legal setup cannot attain the bound.
    other = list(contract["landscape"]["optimal_setup"])
    i = next(i for i in range(40) if other[i] != other[(i + 1) % 40])
    j = next(j for j in range(40) if other[j] != other[i])
    other[i], other[j] = other[j], other[i]
    assert not driver.verify_certificate(table, potentials["u"], potentials["v"], other)["certified"]


def test_verify_frozen_identity_refuses_any_drift(driver, frozen):
    reports, contract, landscape = frozen
    loaded, landscape_document, _, _ = driver.load_frozen(reports)
    design, built = driver.verify_frozen_identity(loaded, landscape_document)
    assert design.updates == 64 and built.optimum == landscape_document["exact_optimum"]["optimum"]
    drifted = json.loads(json.dumps(loaded))
    drifted["design"]["updates"] = 32
    with pytest.raises(driver.G2RawError, match="does not re-derive"):
        driver.verify_frozen_identity(drifted, landscape_document)
    drifted = json.loads(json.dumps(loaded))
    drifted["seed_derivation"]["base_commit"] = "0" * 40
    with pytest.raises(driver.G2RawError, match="base commit"):
        driver.verify_frozen_identity(drifted, landscape_document)
    drifted = json.loads(json.dumps(loaded))
    drifted["landscape"]["certificate_potentials"]["u"][0] += 1.0
    with pytest.raises(driver.G2RawError, match="certificate"):
        driver.verify_frozen_identity(drifted, landscape_document)
    with pytest.raises(Exception):
        driver.verify_frozen_identity(loaded, dict(landscape_document, table_digest="0" * 64))


def test_the_criteria_read_all_three_seeds_the_pooled_lower_bound_and_the_median_gap(driver):
    design = driver.frozen_design()
    boots = driver.bootstrap_seeds(design)
    rng = np.random.default_rng(0)
    arrays = {k: {"initial": rng.normal(2.0, 5.0, 64), "final": None} for k in design.seed_indices}
    for k in design.seed_indices:
        arrays[k]["final"] = arrays[k]["initial"] + 1.5 + rng.normal(0.0, 0.2, 64)
    endpoints = {k: {"initial": float(arrays[k]["initial"].mean()), "final": float(arrays[k]["final"].mean()), "gap_fraction": (0.12, 0.08, 0.15)[i]} for i, k in enumerate(design.seed_indices)}
    verdict = driver.criteria(design, endpoints, arrays, pooled_seed=boots["raw_pooled"], per_seed_seeds=boots["raw_per_seed"])
    assert verdict["all_seeds_improved"] and verdict["pooled_lower_bound_strictly_above_zero"]
    assert verdict["median_gap_fraction_closed"] == pytest.approx(0.12) and verdict["median_gap_closure_meets_threshold"]
    assert verdict["pooled_paired_interval"]["sample_size"] == 192 and verdict["pooled_paired_interval"]["replicates"] == 10000
    assert verdict["pooled_paired_interval"]["seed"] == boots["raw_pooled"]
    endpoints[design.seed_indices[1]]["gap_fraction"] = 0.05
    endpoints[design.seed_indices[2]]["gap_fraction"] = 0.05
    assert not driver.criteria(design, endpoints, arrays, pooled_seed=boots["raw_pooled"], per_seed_seeds=boots["raw_per_seed"])["median_gap_closure_meets_threshold"]
    flat = {k: {"initial": arrays[k]["initial"], "final": arrays[k]["initial"] + rng.normal(0.0, 0.2, 64)} for k in design.seed_indices}
    assert not driver.criteria(design, endpoints, flat, pooled_seed=boots["raw_pooled"], per_seed_seeds=boots["raw_per_seed"])["pooled_lower_bound_strictly_above_zero"]


def test_the_frozen_decision_rule(driver):
    passing = {"all_seeds_improved": True, "pooled_lower_bound_strictly_above_zero": True, "median_gap_closure_meets_threshold": True}
    assert driver.apply_decision_rule(parity=True, integrity=True, replay=True, binding=True, raw=passing)["decision"] == "PROCEED"
    assert driver.apply_decision_rule(parity=True, integrity=True, replay=True, binding=None, raw=passing)["decision"] == "PENDING"
    failing = dict(passing, median_gap_closure_meets_threshold=False)
    verdict = driver.apply_decision_rule(parity=True, integrity=True, replay=True, binding=True, raw=failing)
    assert verdict["decision"] == "STOP" and "median_gap_closure_meets_threshold" in verdict["basis"]
    for broken in ("parity", "integrity", "replay", "binding"):
        checks = {"parity": True, "integrity": True, "replay": True, "binding": True}
        checks[broken] = False
        verdict = driver.apply_decision_rule(raw=passing, **checks)
        assert verdict["decision"] == "REVISE" and broken in verdict["basis"]
    # An unfavourable valid result with an EMA that happens to pass is still STOP:
    # the EMA is never consulted.
    verdict = driver.apply_decision_rule(parity=True, integrity=True, replay=True, binding=True, raw=dict(passing, all_seeds_improved=False))
    assert verdict["decision"] == "STOP" and verdict["ema_results_considered"] is False


def test_sample_count_integrity_flags_a_short_or_non_finite_array(driver, tmp_path):
    arrays = {}
    for name, values in (("initial", np.zeros(8)), ("final", np.ones(8)), ("initial_raw", np.zeros(7)), ("final_raw", np.array([1.0] * 7 + [np.nan]))):
        path = tmp_path / f"{name}.npy"
        np.save(path, values)
        arrays[name] = {"path": str(path), "sha256": "unused"}
    telemetry = {"generation_telemetry": {"immediately_terminal_count": 0}}
    record = {"utilities": arrays, "initial": telemetry, "final": telemetry, "raw_diagnostic": {"initial": telemetry, "final": telemetry}}
    check = driver.sample_count_integrity(record, 8)
    assert check["arrays"]["initial"]["ok"] and check["arrays"]["final"]["ok"]
    assert not check["arrays"]["initial_raw"]["ok"] and check["arrays"]["initial_raw"]["count"] == 7
    assert not check["arrays"]["final_raw"]["ok"] and not check["arrays"]["final_raw"]["finite"]
    assert not check["all_ok"]


def test_the_binding_ledger_flags_a_mismatched_source_commit(driver, tmp_path):
    reports = tmp_path
    (reports / driver.DIRECTORY).mkdir()
    driver.write_json(reports / driver.CONTRACT_NAME, {"artifact": "contract"})
    driver.write_json(reports / driver.LANDSCAPE_NAME, {"artifact": "landscape"})
    driver.write_json(reports / driver.LAUNCH_NAME, {"source": {"source_commit": "a" * 40, "source_tree": "b" * 40}, "artifacts": {}})
    driver.write_json(reports / driver.RESULTS_NAME, {"source_commit": "c" * 40})
    driver.write_json(reports / driver.DIRECTORY / "phase18_g2_raw_confirmation_seed_1_result_v1.json", {"source_commit": "a" * 40})
    ledger = driver.stage_bind(reports)
    assert ledger["mismatched_artifacts"] == [driver.RESULTS_NAME]
    assert not ledger["all_artifacts_bind_one_source_commit"]
    assert ledger["artifacts"][driver.CONTRACT_NAME]["source_binding"]["agrees"] is None
    assert ledger["base_commit"] == driver.BASE_COMMIT
