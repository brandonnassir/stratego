"""Stage 6B G3-ENG-04 focused tests: the reserved bases stay closed, the
schedule is identical across arms with matched opponents, formations,
colours and seeds, own setups are a pure function of (bundle, case seed),
cross-lineage pairing is refused, a mismatched component digest is refused,
accounting reconciles on a tiny schedule, and a handcrafted-opponent game
runs under EVALUATION_RULES and writes a receipt."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from stratego.engine.constants import BLUE, RED
from stratego.evaluation.match_spec import schedule_digest
from stratego.setups.identity import orient_setup
from stratego.training.phase18 import g3_evaluation as ev
from stratego.training.phase18 import g3_pilot as gp
from stratego.training.phase18.g3_bundle import read_manifest, verify_bundle
from stratego.training.phase18.g3_contract import (
    EVALUATION_BASE_INDICES,
    HANDCRAFTED_OPPONENTS,
    RESERVED_BASE_INDICES,
    Phase18G3Error,
    Phase18G3LineageError,
)
from stratego.training.phase18.g3_smoke import UNIFORM_PRIOR, smoke_pilot_config
from stratego.training.phase18.setup_model import build_setup_model, state_dict_digest
from stratego.training.warmstart_checkpoint import verify_corpus_identity

NAMESPACE = "phase18_g3_evaluation_test_v1"
TINY_FAMILIES = ("F00", "F01")
TINY_OPPONENTS = ("basic_heuristic", "stress_chaos")


@pytest.fixture(scope="module")
def bases():
    return ev.load_evaluation_bases(families=TINY_FAMILIES, base_indices=EVALUATION_BASE_INDICES[:2])


@pytest.fixture(scope="module")
def cases(bases):
    return ev.build_cases(bases, opponents=TINY_OPPONENTS)


@pytest.fixture(scope="module")
def mini_corpus(tmp_path_factory):
    """A four-game committed corpus (three train games), as tests/training builds it."""
    from stratego.training import synthetic_corpus as sc
    from stratego.training.warmstart_seed import synthetic_game_id

    ids = (
        synthetic_game_id("train", "strategic_rule_based@1.1.0", "random_legal@1.0.0", 0),
        synthetic_game_id("train", "tactical_rule_based@1.0.0", "basic_heuristic@1.0.0", 0),
        synthetic_game_id("train", "random_legal@1.0.0", "random_legal@1.0.0", 0),
        synthetic_game_id("validation", "basic_heuristic@1.0.0", "stress_chaos@1.0.0", 0),
    )
    root = tmp_path_factory.mktemp("g3_eval_mini_corpus")
    sc.generate_corpus(root, worker_count=1, chunks_per_worker=1, game_ids=ids)
    return root


@pytest.fixture(scope="module")
def pilot_bundles(tmp_path_factory, mini_corpus):
    """Final bundles of both lineages of a two-period smoke pilot."""
    torch.set_num_threads(1)
    root = mini_corpus
    identity = verify_corpus_identity(root, None, check_payload_bytes=False)
    run_root = tmp_path_factory.mktemp("eval_pilot")
    bundles = {}
    configs = {}
    for lineage in ("candidate", "control"):
        config = smoke_pilot_config(lineage=lineage, namespace=NAMESPACE, run_id="G3-EVAL-TEST", overrides={"periods": 2, "plies_per_period": 160})
        runner = gp.LineageRunner.fresh(config, run_root=run_root, corpus_root=root, corpus_identity=identity, value_prior=UNIFORM_PRIOR, require_complete_split=False, log=lambda m: None)
        try:
            runner.run()
        finally:
            runner.close()
        bundles[lineage] = runner.bundle_path(2)
        configs[lineage] = config
    return run_root, bundles, configs


def test_the_full_case_set_is_160_bases_by_16_cases_and_never_opens_a_reserved_base():
    full = ev.load_evaluation_bases()
    assert len(full) == 160 and {e.base_index for e in full} == set(EVALUATION_BASE_INDICES)
    assert len({e.family_id for e in full}) == 16
    assert not ({e.base_index for e in full} & set(RESERVED_BASE_INDICES))
    cases = ev.build_cases(full)
    assert len(cases) == 2560 and all(c.case_index % 2 == c.colour for c in cases)
    assert [c.opponent_id for c in cases[:16:2]] == list(HANDCRAFTED_OPPONENTS)
    with pytest.raises(Phase18G3Error, match="reserved"):
        ev.load_evaluation_bases(base_indices=(410,))
    with pytest.raises(Phase18G3Error, match="sealed"):
        ev.load_evaluation_bases(base_indices=(450,))


def test_the_schedule_is_identical_across_arms_with_matched_cases(cases):
    a = ev.build_schedule(cases, namespace=NAMESPACE)
    b = ev.build_schedule(cases, namespace=NAMESPACE)
    assert schedule_digest(a) == schedule_digest(b) and len(a) == len(cases) == 16
    for spec, case in zip(a, cases):
        assert spec.candidate_color == case.colour and spec.setup_pair_id == case.case_index
        assert spec.opponent.policy_id == case.opponent_id
        assert spec.rules.battleless_move_limit == 200 and spec.rules.context == "evaluation"
    assert schedule_digest(ev.build_schedule(cases, namespace="other")) != schedule_digest(a)
    record = ev.schedule_record(cases, a, namespace=NAMESPACE)
    assert record["candidate_token"] == "phase6_g3_bundle_greedy@0.2.0+float32" and record["bases"] == 4


def test_own_setups_are_a_pure_function_of_the_bundle_and_the_case_seed(cases):
    torch.set_num_threads(1)
    model_a = build_setup_model(device="cpu", seed=3)
    model_b = build_setup_model(device="cpu", seed=4)
    first = ev.resolve_own_setups(model_a, cases, namespace=NAMESPACE, seed_index=1)
    again = ev.resolve_own_setups(model_a, cases, namespace=NAMESPACE, seed_index=1)
    other = ev.resolve_own_setups(model_b, cases, namespace=NAMESPACE, seed_index=1)
    assert [s.content_fingerprint for s in first.samples] == [s.content_fingerprint for s in again.samples]
    assert [s.root_seed for s in first.samples] == [s.root_seed for s in other.samples], "shared case seeds"
    assert [s.content_fingerprint for s in first.samples] != [s.content_fingerprint for s in other.samples]
    bank = ev.build_arm_bank(cases, first.samples)
    for case, sample in zip(cases, first.samples):
        pair = bank.pair(case.case_index)
        own = pair.red_setup if case.colour == RED else pair.blue_setup
        base = pair.blue_setup if case.colour == RED else pair.red_setup
        assert tuple(own) == tuple(sample.engine_setup)
        assert tuple(base) == orient_setup(case.base_canonical, BLUE if case.colour == RED else RED)


def test_cross_lineage_pairing_and_mismatched_digests_are_refused(pilot_bundles, tmp_path):
    run_root, bundles, configs = pilot_bundles
    candidate = read_manifest(bundles["candidate"])
    control = read_manifest(bundles["control"])
    ev.assert_internally_matched(ev.component_tag(candidate, "c1"), ev.component_tag(candidate, "setup_ema"))
    with pytest.raises(Phase18G3LineageError, match="never crossed"):
        ev.assert_internally_matched(ev.component_tag(candidate, "c1"), ev.component_tag(control, "setup_ema"))
    with pytest.raises(Phase18G3LineageError):
        ev.assert_internally_matched(ev.component_tag(control, "c1"), ev.component_tag(candidate, "setup_ema"))
    earlier = read_manifest(run_root / "candidate" / gp.BUNDLES_DIRECTORY / gp.bundle_name(1))
    with pytest.raises(Phase18G3Error, match="evaluated whole"):
        ev.assert_internally_matched(ev.component_tag(candidate, "c1"), ev.component_tag(earlier, "setup_ema"))
    # The adapter refuses a bundle whose declared lineage is not the one asked for.
    with pytest.raises(Phase18G3LineageError):
        ev.evaluate_bundle(bundles["candidate"], config=configs["control"], lineage="control", label="x", cases=[], work=tmp_path / "x")
    # A mismatched component digest is refused before any game.
    import shutil

    tampered = tmp_path / "tampered"
    shutil.copytree(bundles["candidate"], tampered)
    with (tampered / "setup" / "ema.pt").open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(Phase18G3Error, match="does not verify"):
        verify_bundle(tampered)
    with pytest.raises(Phase18G3Error, match="does not verify"):
        ev.evaluate_bundle(tampered, config=configs["candidate"], lineage="candidate", label="x", cases=[], work=tmp_path / "y")


def test_a_tiny_schedule_plays_reconciles_and_pairs_across_arms(pilot_bundles, cases, tmp_path):
    torch.set_num_threads(1)
    run_root, bundles, configs = pilot_bundles
    rows_by_arm = {}
    records = {}
    for lineage in ("candidate", "control"):
        record, rows = ev.evaluate_bundle(
            bundles[lineage], config=configs[lineage], lineage=lineage, label=f"{lineage}_final", cases=cases, work=tmp_path / lineage, device="cpu", workers=1, chunk_units=4, log=lambda m: None
        )
        records[lineage] = record
        rows_by_arm[lineage] = rows
        assert record["accounting"]["complete_for_primary"] and record["accounting"]["planned"] == 16
        assert record["receipts"]["rows"] == 16 and Path(record["receipts"]["path"]).exists()
        assert record["bundle_id"] == read_manifest(bundles[lineage])["bundle_id"]
        receipts = [json.loads(line) for line in Path(record["receipts"]["path"]).read_text().splitlines()]
        for receipt in receipts:
            assert receipt["bundle_id"] == record["bundle_id"] and receipt["lineage"] == lineage
            assert receipt["rules"].endswith("context=evaluation") and "battleless_move_limit=200" in receipt["rules"]
            assert receipt["candidate_score"] in (0.0, 0.5, 1.0) and not receipt["errored"]
            assert receipt["opponent_id"] in TINY_OPPONENTS and receipt["base_index"] in EVALUATION_BASE_INDICES
    assert records["candidate"]["schedule"]["digest"] == records["control"]["schedule"]["digest"]
    assert records["candidate"]["setup_model_digest"] != records["control"]["setup_model_digest"]
    identity = ev.prove_arm_identity(rows_by_arm, cases)
    assert not identity["problems"], identity
    # Reused chunks come back byte-for-byte on a rerun.
    again, rows_again = ev.evaluate_bundle(bundles["control"], config=configs["control"], lineage="control", label="control_final", cases=cases, work=tmp_path / "control", device="cpu", chunk_units=4, log=lambda m: None)
    assert all(chunk["reused"] for chunk in again["chunks"])
    assert [r.match_id for r in rows_again] == [r.match_id for r in rows_by_arm["control"]]
    analysis = ev.paired_analysis(rows_by_arm["candidate"], rows_by_arm["control"], cases, namespace=NAMESPACE, replicates=200)
    assert analysis["cases"] == 16 and analysis["bases"] == 4 and analysis["families"] == 2
    assert analysis["lower"] <= analysis["point"] <= analysis["upper"]
    assert isinstance(analysis["passes"], bool) and "by_opponent" in analysis


def test_the_stratified_bootstrap_reproduces_the_design_formula_and_the_rule():
    rng = np.random.default_rng(1)
    per_base = rng.normal(0.06, 0.1, size=160)
    families = np.repeat(np.arange(16), 10)
    means = ev.stratified_cluster_bootstrap(per_base, families, replicates=2000, seed=7)
    # Equal family sizes: the 6A formula (global resample means rescaled by sqrt(10/9)).
    rng2 = np.random.default_rng(7 % (2**32))
    reference = []
    labels = sorted(set(families.tolist()))
    groups = [np.nonzero(families == label)[0] for label in labels]
    acc = np.zeros(2000)
    for group in groups:
        values = per_base[group]
        draws = values[rng2.integers(0, 10, size=(2000, 10))].mean(axis=1)
        acc += (values.mean() + (draws - values.mean()) * np.sqrt(10 / 9)) * (10 / 160)
    assert np.allclose(means, acc)
    assert abs(means.mean() - per_base.mean()) < 0.01
    assert ev.direct_per_base_standard_error(per_base, families) == pytest.approx(np.sqrt(np.mean([per_base[families == f].var(ddof=1) for f in labels]) / 160))
    with pytest.raises(Phase18G3Error, match="at least two"):
        ev.stratified_cluster_bootstrap(np.zeros(3), np.array([0, 0, 1]), replicates=10, seed=1)


# ---------------------------------------------------------------------------
# P18-A002: the persisted-receipt reader. stage_analyse is the only caller that
# reads rows back from disk, and the tests above hand prove_arm_identity the
# in-memory rows, so the round trip through read_receipt_rows was never covered.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evaluated_arms(pilot_bundles, cases, tmp_path_factory):
    """Both arms played once: the in-memory rows, and the receipts they wrote."""
    torch.set_num_threads(1)
    run_root, bundles, configs = pilot_bundles
    work = tmp_path_factory.mktemp("receipt_round_trip")
    arms = {}
    for lineage in ("candidate", "control"):
        record, rows = ev.evaluate_bundle(
            bundles[lineage], config=configs[lineage], lineage=lineage, label=f"{lineage}_final",
            cases=cases, work=work / lineage, device="cpu", workers=1, chunk_units=4, log=lambda m: None
        )
        arms[lineage] = {"record": record, "rows": rows, "receipts": Path(record["receipts"]["path"])}
    return arms


def test_every_invariant_field_survives_the_receipt_round_trip(evaluated_arms):
    for lineage, arm in evaluated_arms.items():
        loaded = ev.read_receipt_rows(arm["receipts"])
        assert len(loaded) == len(arm["rows"])
        by_match = {row.match_id: row for row in arm["rows"]}
        for row in loaded:
            original = by_match[row.match_id]
            for field in ev.ARM_INVARIANT_FIELDS:
                assert hasattr(row, field), f"{lineage}: {field} did not survive persistence"
                assert getattr(row, field) == getattr(original, field), f"{lineage}: {field} changed"
        # the three reconstructed aliases are exactly the documented ones
        assert set(ev.RECEIPT_ALIASES) <= set(ev.ARM_INVARIANT_FIELDS)


def test_prove_arm_identity_and_case_scores_accept_loaded_receipts(evaluated_arms, cases):
    loaded = {f"{lineage}_final": ev.read_receipt_rows(arm["receipts"]) for lineage, arm in evaluated_arms.items()}
    identity = ev.prove_arm_identity(loaded, cases)
    assert not identity["problems"], identity
    assert identity["opponent_formation_mismatches"] == 0
    for rows in loaded.values():
        scores = ev.case_scores(rows, cases)
        assert scores.shape == (len(cases),)
        assert set(scores.tolist()) <= {0.0, 0.5, 1.0}


def test_analysis_of_loaded_receipts_equals_analysis_of_the_in_memory_rows(evaluated_arms, cases):
    memory = ev.paired_analysis(
        evaluated_arms["candidate"]["rows"], evaluated_arms["control"]["rows"], cases,
        namespace=NAMESPACE, replicates=200,
    )
    disk = ev.paired_analysis(
        ev.read_receipt_rows(evaluated_arms["candidate"]["receipts"]),
        ev.read_receipt_rows(evaluated_arms["control"]["receipts"]),
        cases, namespace=NAMESPACE, replicates=200,
    )
    for field in ("point", "lower", "upper", "cases", "bases", "families", "passes", "near_boundary"):
        assert disk[field] == memory[field], f"{field} differs between the disk and memory analyses"


def _one_receipt(evaluated_arms):
    return json.loads(evaluated_arms["candidate"]["receipts"].read_text().splitlines()[0])


@pytest.mark.parametrize("drop", ["case_index", "opponent_policy", "opponent_id"])
def test_a_receipt_missing_a_reconstruction_input_is_rejected(evaluated_arms, tmp_path, drop):
    receipt = _one_receipt(evaluated_arms)
    receipt.pop(drop)
    path = tmp_path / f"missing_{drop}.jsonl"
    path.write_text(json.dumps(receipt) + "\n")
    with pytest.raises(Phase18G3Error, match=drop):
        ev.read_receipt_rows(path)


@pytest.mark.parametrize("policy", ["no_separator", "@only_version", "only_id@"])
def test_a_malformed_combined_policy_name_is_rejected(evaluated_arms, tmp_path, policy):
    receipt = _one_receipt(evaluated_arms)
    receipt["opponent_policy"] = policy
    path = tmp_path / "malformed_policy.jsonl"
    path.write_text(json.dumps(receipt) + "\n")
    with pytest.raises(Phase18G3Error, match="opponent_policy"):
        ev.read_receipt_rows(path)


def test_a_policy_id_disagreeing_with_opponent_id_is_rejected(evaluated_arms, tmp_path):
    receipt = _one_receipt(evaluated_arms)
    receipt["opponent_policy"] = f"someone_else@{receipt['opponent_policy'].rpartition('@')[2]}"
    path = tmp_path / "policy_disagrees.jsonl"
    path.write_text(json.dumps(receipt) + "\n")
    with pytest.raises(Phase18G3Error, match="disagrees with the persisted opponent_id"):
        ev.read_receipt_rows(path)


@pytest.mark.parametrize("alias", ["setup_pair_id", "opponent_policy_id", "opponent_policy_version"])
def test_a_conflicting_persisted_alias_is_rejected(evaluated_arms, tmp_path, alias):
    receipt = _one_receipt(evaluated_arms)
    receipt[alias] = 999999 if alias == "setup_pair_id" else "conflicting_value"
    path = tmp_path / f"conflict_{alias}.jsonl"
    path.write_text(json.dumps(receipt) + "\n")
    with pytest.raises(Phase18G3Error, match="conflicting with the reconstruction"):
        ev.read_receipt_rows(path)


def test_an_agreeing_persisted_alias_is_accepted(evaluated_arms, tmp_path):
    """A future receipt that also writes the alias is fine, as long as it agrees."""
    receipt = _one_receipt(evaluated_arms)
    identifier, _, version = str(receipt["opponent_policy"]).rpartition("@")
    receipt["setup_pair_id"] = int(receipt["case_index"])
    receipt["opponent_policy_id"] = identifier
    receipt["opponent_policy_version"] = version
    path = tmp_path / "agreeing_alias.jsonl"
    path.write_text(json.dumps(receipt) + "\n")
    row = ev.read_receipt_rows(path)[0]
    assert row.setup_pair_id == int(receipt["case_index"])
    assert row.opponent_policy_id == identifier and row.opponent_policy_version == version


# ---------------------------------------------------------------------------
# The command-level analysis path, exercised over PERSISTED receipts. This is
# the coverage whose absence let P18-A002's defect reach a completed run:
# stage_analyse is the only caller that reads rows back from disk.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def driver():
    import importlib.util

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("phase18_g3_pilot", root / "scripts" / "phase18_g3_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analysable_run(driver, pilot_bundles, cases, tmp_path_factory):
    """A complete miniature G3 run: two smoke periods per lineage, both arms
    evaluated into the runtime, and every record stage_analyse reads."""
    torch.set_num_threads(1)
    run_root, bundles, configs = pilot_bundles
    reports = tmp_path_factory.mktemp("stage_analyse_reports")
    matches = ev.build_schedule(cases, namespace=NAMESPACE)
    schedule = ev.schedule_record(cases, matches, namespace=NAMESPACE)

    for lineage in ("candidate", "control"):
        work = run_root / "evaluation" / f"{lineage}_final"
        record, _rows = ev.evaluate_bundle(
            bundles[lineage], config=configs[lineage], lineage=lineage, label=f"{lineage}_final",
            cases=cases, work=work, device="cpu", workers=1, chunk_units=4, log=lambda m: None,
        )
        record = record | {"contract_sha256": None, "diagnostic": False}
        driver.write_json(work / "arm_record.json", record)

    contract = {
        "question": {"decision_rule": "test rule", "near_boundary_rule": "test near-boundary rule"},
        "frozen_defaults": {"c1_device": "cpu"},
    }
    driver.write_json(reports / driver.CONTRACT_NAME, contract)
    contract_sha = driver.file_sha256(reports / driver.CONTRACT_NAME)
    for work in (run_root / "evaluation" / "candidate_final", run_root / "evaluation" / "control_final"):
        record = json.loads((work / "arm_record.json").read_text())
        driver.write_json(work / "arm_record.json", record | {"contract_sha256": contract_sha})

    matching = gp.matching_check(run_root, c1_device="cpu")
    matching["contract_sha256"] = contract_sha
    driver.write_json(reports / driver.MATCHING_NAME, matching)
    driver.write_json(reports / driver.VERIFICATION_NAME, {"restart_check": {"passed": True}})
    driver.write_json(reports / driver.LAUNCH_NAME, {
        "contract_sha256": contract_sha,
        "source": {"g3_source_commit": "0" * 40},
        "source_digests": {}, "test_digests": {},
        "runtime": {"root_absolute": str(run_root)},
    })
    frozen = {"configs": configs, "cases": cases, "matches": matches, "schedule": schedule}
    return driver, reports, run_root, frozen, contract_sha


def test_stage_analyse_reads_persisted_receipts_end_to_end(analysable_run, monkeypatch):
    driver, reports, run_root, frozen, _sha = analysable_run
    monkeypatch.setattr(driver, "verify_frozen_identity", lambda contract: frozen)
    monkeypatch.setattr(driver, "require_launch_binding", lambda reports, **kw: json.loads((reports / driver.LAUNCH_NAME).read_text()))
    results = driver.stage_analyse(reports, runtime=run_root)
    assert (reports / driver.RESULTS_NAME).exists()
    assert results["arm_identity_proof"]["problems"] == []
    assert results["primary"]["cases"] == len(frozen["cases"])
    assert results["decision_input"]["decision"] in ("PROCEED", "NEAR_BOUNDARY", "FAIL", "BLOCKED")
    # the analysis really went through the on-disk receipts
    for arm in ("candidate_final", "control_final"):
        assert results["arms"][arm]["receipts"]["rows"] == len(frozen["cases"])


def test_stage_analyse_refuses_a_receipt_that_lost_its_reconstruction_inputs(analysable_run, monkeypatch, tmp_path):
    """The exact failure P18-A002 repaired must stay a loud refusal, not a crash
    that only appears after a completed run."""
    driver, reports, run_root, frozen, _sha = analysable_run
    monkeypatch.setattr(driver, "verify_frozen_identity", lambda contract: frozen)
    monkeypatch.setattr(driver, "require_launch_binding", lambda reports, **kw: json.loads((reports / driver.LAUNCH_NAME).read_text()))
    record_path = run_root / "evaluation" / "candidate_final" / "arm_record.json"
    record = json.loads(record_path.read_text())
    receipts = Path(record["receipts"]["path"])
    original = receipts.read_text()
    damaged = [json.loads(line) for line in original.splitlines() if line.strip()]
    for receipt in damaged:
        receipt.pop("case_index")
    receipts.write_text("".join(json.dumps(r) + "\n" for r in damaged))
    try:
        record["receipts"]["sha256"] = driver.file_sha256(receipts)
        driver.write_json(record_path, record)
        with pytest.raises(Phase18G3Error, match="case_index"):
            driver.stage_analyse(reports, runtime=run_root)
    finally:
        receipts.write_text(original)
        record["receipts"]["sha256"] = driver.file_sha256(receipts)
        driver.write_json(record_path, record)
