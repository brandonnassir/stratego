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
