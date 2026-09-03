"""The G3 pilot driver: the frozen production configuration, the deterministic
freeze, refusal on frozen-identity drift, the diagnostic-arm gating and the
horizon that nothing continues past."""

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DRIVER = REPOSITORY_ROOT / "scripts" / "phase18_g3_pilot.py"


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("phase18_g3_pilot", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frozen(driver, tmp_path_factory):
    reports = tmp_path_factory.mktemp("g3_reports")
    contract = driver.stage_freeze(reports, c1_device="cpu")
    return reports, contract


def test_the_production_configuration_is_the_reviewers_frozen_defaults(driver):
    from stratego.training.phase18.g3_contract import G3_DESIGN_COMMIT

    candidate = driver.production_config("candidate", c1_device="cpu")
    control = driver.production_config("control", c1_device="cpu")
    assert candidate.matched_digest() == control.matched_digest()
    assert candidate.config_digest() != control.config_digest()
    assert candidate.setup_updates_enabled and not control.setup_updates_enabled
    assert candidate.is_production_scale()
    assert (candidate.periods, candidate.c1_updates_per_period, candidate.bundle_cadence_periods) == (256, 64, 32)
    assert (candidate.canonical_per_batch, candidate.live_per_batch, candidate.c1_train_config.batch_size) == (128, 128, 256)
    assert (candidate.live_retention_periods, candidate.buffer_storage_periods) == (32, 21)
    assert (candidate.slots, candidate.plies_per_period, candidate.pool_size) == (2560, 202, 1024)
    assert candidate.c1_train_config.candidate_id == "ws_pilot_lr1e-3_balanced" and candidate.c1_train_config.model_init_seed == 2026081302
    assert candidate.c1_train_config.learning_rate == 1e-3 and candidate.c1_train_config.weight_decay == 0.01 and candidate.c1_train_config.warmup_steps == 500
    assert driver.NAMESPACE == f"phase18_g3_pilot_v1:{G3_DESIGN_COMMIT}"
    assert candidate.setup_device == "cpu"


def test_the_freeze_is_deterministic_and_binds_the_evaluation_schedule(driver, frozen, tmp_path):
    reports, contract = frozen
    again = driver.stage_freeze(tmp_path / "again", c1_device="cpu")
    for record in (contract, again):
        record.pop("timestamp_utc")
        record.pop("seconds")
    assert contract == again
    assert contract["evaluation"]["cases_per_arm"] == 2560 and contract["evaluation"]["schedule"]["matches"] == 2560
    assert contract["evaluation"]["bases"]["count"] == 160 and contract["evaluation"]["bases"]["reserved_untouched"] == list(range(410, 450))
    assert contract["evaluation"]["opponents"] == list(driver.HANDCRAFTED_OPPONENTS) and len(contract["evaluation"]["opponents"]) == 8
    assert contract["lineages"]["candidate"]["setup_updates_enabled"] and not contract["lineages"]["control"]["setup_updates_enabled"]
    assert contract["status"].startswith("CONTRACT FROZEN") and "NOT authorised" in contract["status"]
    assert contract["frozen_defaults"]["evaluation_rules"].startswith("EVALUATION_RULES (battleless 200")


def test_verify_frozen_identity_refuses_any_drift(driver, frozen):
    reports, contract = frozen
    contract = json.loads((reports / driver.CONTRACT_NAME).read_text())
    rebuilt = driver.verify_frozen_identity(contract)
    assert len(rebuilt["cases"]) == 2560 and rebuilt["schedule"]["digest"] == contract["evaluation"]["schedule"]["digest"]
    drifted = json.loads(json.dumps(contract))
    drifted["matched_digest"] = "0" * 64
    with pytest.raises(driver.G3Error, match="does not re-derive"):
        driver.verify_frozen_identity(drifted)
    drifted = json.loads(json.dumps(contract))
    drifted["evaluation"]["schedule"]["digest"] = "0" * 64
    with pytest.raises(driver.G3Error, match="schedule does not re-derive"):
        driver.verify_frozen_identity(drifted)


def test_diagnostic_arms_are_gated_and_the_final_arms_are_the_primary(driver):
    assert driver.ARMS["candidate_final"] == ("candidate", 256, False)
    assert driver.ARMS["control_final"] == ("control", 256, False)
    assert driver.ARMS["candidate_128"][2] and driver.ARMS["candidate_0"][2] and driver.ARMS["control_128"][2]


def test_the_driver_never_continues_past_the_horizon(driver, frozen):
    from stratego.training.phase18.g3_contract import Phase18G3Error

    config = driver.production_config("candidate", c1_device="cpu")
    assert config.periods == 256
    from stratego.training.phase18.g3_pilot import LineageRunner

    runner = LineageRunner.__new__(LineageRunner)
    runner.config = config
    runner.period = 256
    with pytest.raises(Phase18G3Error, match="bounded horizon"):
        runner.run_period()


# ---------------------------------------------------------------------------
# Corrective commit: the launch binding
# ---------------------------------------------------------------------------


def _binding_root(tmp_path, driver):
    root = tmp_path / "root"
    for name in driver.HARNESS_SOURCES + driver.TEST_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"content of {name}\n")
    reports = root / "reports" / "phase18" / "g3_pilot"
    reports.mkdir(parents=True)
    contract = reports / driver.CONTRACT_NAME
    contract.write_text("{}\n")
    return root, reports, contract


def test_stale_verification_evidence_is_refused_at_launch(driver, tmp_path):
    root, reports, contract = _binding_root(tmp_path, driver)
    sha = driver.file_sha256(contract)
    verification = {
        "contract_sha256": sha,
        "source_digests": driver.digests_of(driver.HARNESS_SOURCES, root),
        "test_digests": driver.digests_of(driver.TEST_FILES, root),
    }
    assert driver.verification_binding_problems(verification, root=root, contract_sha256=sha) == []
    drifted = driver.HARNESS_SOURCES[3]
    (root / drifted).write_text("drift\n")
    problems = driver.verification_binding_problems(verification, root=root, contract_sha256=sha)
    assert problems == [f"stale verification evidence: source file {drifted} changed since the record was written"]
    (root / drifted).write_text(f"content of {drifted}\n")
    (root / driver.TEST_FILES[0]).write_text("drift\n")
    assert any("test file" in p for p in driver.verification_binding_problems(verification, root=root, contract_sha256=sha))
    (root / driver.TEST_FILES[0]).write_text(f"content of {driver.TEST_FILES[0]}\n")
    assert any("different contract" in p for p in driver.verification_binding_problems(verification, root=root, contract_sha256="0" * 64))
    (root / driver.TEST_FILES[1]).unlink()
    assert any("is missing" in p for p in driver.verification_binding_problems(verification, root=root, contract_sha256=sha))


def test_source_drift_or_a_mismatched_commit_is_refused_before_every_stage(driver, tmp_path):
    root, reports, contract = _binding_root(tmp_path, driver)
    commit = "a" * 40
    manifest = {
        "source": {"g3_source_commit": commit},
        "contract_sha256": driver.file_sha256(contract),
        "source_digests": driver.digests_of(driver.HARNESS_SOURCES, root),
        "test_digests": driver.digests_of(driver.TEST_FILES, root),
    }
    assert driver.launch_binding_problems(manifest, root=root, head=commit) == []
    assert driver.launch_binding_problems(manifest, root=root, head="b" * 40) == [f"HEAD {'b' * 40} is not the launch manifest's source commit {commit}"]
    (root / driver.HARNESS_SOURCES[0]).write_text("drift\n")
    assert driver.launch_binding_problems(manifest, root=root, head=commit) == [f"tracked source file {driver.HARNESS_SOURCES[0]} differs from the launch manifest"]
    (root / driver.HARNESS_SOURCES[0]).write_text(f"content of {driver.HARNESS_SOURCES[0]}\n")
    (root / driver.TEST_FILES[2]).write_text("drift\n")
    assert any("test file" in p and "differs" in p for p in driver.launch_binding_problems(manifest, root=root, head=commit))
    (root / driver.TEST_FILES[2]).write_text(f"content of {driver.TEST_FILES[2]}\n")
    # Generated reports and ignored runtime output are not source: they never trip the check.
    (reports / driver.MATCHING_NAME).write_text("{}\n")
    runtime = root / "output" / "phase18" / "runtime" / "g3_pilot_v1"
    runtime.mkdir(parents=True)
    (runtime / "candidate.log").write_text("log\n")
    assert driver.launch_binding_problems(manifest, root=root, head=commit) == []
    driver.write_json(reports / driver.LAUNCH_NAME, manifest)
    bound = driver.require_launch_binding(reports, contract_sha256=manifest["contract_sha256"], root=root, head=commit)
    assert bound["source"]["g3_source_commit"] == commit
    with pytest.raises(driver.G3Error, match="not the launched source"):
        driver.require_launch_binding(reports, root=root, head="b" * 40)
    with pytest.raises(driver.G3Error, match="different contract"):
        driver.require_launch_binding(reports, contract_sha256="0" * 64, root=root, head=commit)
    (reports / driver.LAUNCH_NAME).unlink()
    with pytest.raises(driver.G3Error, match="no launch manifest"):
        driver.require_launch_binding(reports, root=root, head=commit)


def test_every_post_launch_stage_refuses_without_the_launch_manifest(driver, frozen, tmp_path):
    reports, _contract = frozen
    for argv in (
        ["--check-matching"],
        ["--run", "--lineage", "candidate"],
        ["--evaluate", "--arm", "control_final"],
        ["--analyse"],
    ):
        assert driver.main(argv + ["--reports", str(reports), "--runtime", str(tmp_path)]) == 2


def test_the_runtime_root_is_bound_to_the_launch_manifest(driver, tmp_path):
    """A repeated pilot runs in a fresh runtime namespace; every post-launch stage
    must name the runtime root the manifest bound, never another."""
    bound = tmp_path / "runtime" / "g3_pilot_v2"
    manifest = {"runtime": {"root_absolute": str(bound)}}
    assert driver.runtime_binding_problems(manifest, bound) == []
    assert driver.runtime_binding_problems(manifest, tmp_path / "runtime" / ".." / "runtime" / "g3_pilot_v2") == []
    other = tmp_path / "runtime" / "g3_pilot_v1"
    problems = driver.runtime_binding_problems(manifest, other)
    assert len(problems) == 1 and "is not the launch manifest's" in problems[0]
    assert driver.runtime_binding_problems({}, bound) == ["the launch manifest records no runtime root"]
    with pytest.raises(driver.G3Error, match="BLOCKED"):
        driver.require_runtime_binding(manifest, other)
    driver.require_runtime_binding(manifest, bound)
