"""Stage 6B critical implementation checks on the two-lineage pilot harness:
identical lineages before setup learning, the frozen control setup model, the
moving candidate setup model, the equal C1 update budget, whole-bundle
identity, and the design-section-6 restart test."""

import copy
import json

import pytest
import torch

from stratego.training.phase18 import g3_pilot as gp
from stratego.training.phase18.g3_bundle import (
    C1_NAME,
    COLLECTOR_NAME,
    compare_bundles,
    read_manifest,
    verify_bundle,
)
from stratego.training.phase18.g3_contract import Phase18G3Error, Phase18G3LineageError
from stratego.training.phase18.g3_smoke import UNIFORM_PRIOR, restart_check, smoke_pilot_config
from stratego.training.warmstart_checkpoint import verify_corpus_identity

NAMESPACE = "phase18_g3_pilot_test_v1"


@pytest.fixture(scope="module")
def corpus(warmstart_mini_corpus):
    root, _ids = warmstart_mini_corpus
    return root, verify_corpus_identity(root, None, check_payload_bytes=False)


def keywords(run_root, corpus):
    root, identity = corpus
    return dict(run_root=run_root, corpus_root=root, corpus_identity=identity, value_prior=UNIFORM_PRIOR, require_complete_split=False, log=lambda m: None)


@pytest.fixture(scope="module")
def two_lineages(tmp_path_factory, corpus):
    """Both lineages of one smoke pilot, run to the bounded horizon of 3 periods."""
    torch.set_num_threads(1)
    run_root = tmp_path_factory.mktemp("pilot")
    records = {}
    for lineage in ("candidate", "control"):
        config = smoke_pilot_config(lineage=lineage, namespace=NAMESPACE, run_id="G3-PILOT-TEST")
        runner = gp.LineageRunner.fresh(config, **keywords(run_root, corpus))
        try:
            records[lineage] = runner.run()
        finally:
            runner.close()
    return run_root, records


def test_both_lineages_are_identical_before_setup_learning_begins(two_lineages):
    run_root, records = two_lineages
    report = gp.matching_check(run_root, c1_device="cpu")
    assert report["matched"], report["problems"]
    assert all(report["init"].values())
    assert report["bundle_0"]["c1"] and report["bundle_0"]["setup_raw"] and report["bundle_0"]["setup_ema"]
    assert report["bundle_0"]["lineage_differs"]
    assert all(v for k, v in report["period_1"].items() if k != "c1_state_digest_note")
    # After the candidate's first applied setup update its pools come from a
    # moved model; the control's never do.
    applied = [r["period"] for r in records["candidate"] if r["setup"]["applied"]]
    assert applied, "the smoke pilot applied no setup update"
    following = applied[0]  # period index of the record after the first update
    if following < len(records["candidate"]):
        assert records["candidate"][following]["pool"]["snapshot_digest"] != records["control"][following]["pool"]["snapshot_digest"]
        assert records["candidate"][following]["pool"]["content_digest"] != records["control"][following]["pool"]["content_digest"]
    for record in records["control"]:
        assert record["pool"]["snapshot_digest"] == gp.read_init_record(run_root / "control")["setup_init_state_digest"]


def test_the_control_setup_model_hash_never_changes(two_lineages):
    run_root, records = two_lineages
    init = gp.read_init_record(run_root / "control")
    for record in records["control"]:
        assert record["setup_raw_digest"] == init["setup_init_state_digest"]
        assert record["setup_ema_digest"] == init["setup_init_state_digest"]
        assert record["setup"]["applied"] is False and record["setup"]["skipped"] is False
        assert record["setup_updates"] == 0 and record["setup_optimizer_steps"] == 0 and record["setup_ema_updates"] == 0
    for period in (0, 1, 2, 3):
        manifest = read_manifest(run_root / "control" / gp.BUNDLES_DIRECTORY / gp.bundle_name(period))
        assert manifest["components"]["setup_raw"]["state_digest"] == init["setup_init_state_digest"]
        assert manifest["components"]["setup_ema"]["state_digest"] == init["setup_init_state_digest"]
        assert manifest["setup_updates_enabled"] is False


def test_the_candidate_setup_model_changes_after_a_real_update(two_lineages):
    run_root, records = two_lineages
    init = gp.read_init_record(run_root / "candidate")
    applied = [r for r in records["candidate"] if r["setup"]["applied"]]
    assert applied, "no candidate period applied a setup update; the smoke configuration must complete games"
    for record in applied:
        update = record["setup"]["update"]
        assert update["optimizer_steps"] >= 1 and update["epochs"] == 5 and update["ema_updates"] >= 1
        assert update["raw_digest_after"] != update["raw_digest_before"]
        assert record["setup_raw_digest"] == update["raw_digest_after"]
        assert record["setup_ema_digest"] != record["setup_raw_digest"]
    assert records["candidate"][-1]["setup_raw_digest"] != init["setup_init_state_digest"]
    assert records["candidate"][-1]["setup_ema_digest"] != init["setup_init_state_digest"]
    assert records["candidate"][-1]["setup_updates"] == len(applied)


def test_both_lineages_receive_the_equal_gameplay_update_budget(two_lineages):
    run_root, records = two_lineages
    report = gp.matching_check(run_root, c1_device="cpu")
    budget = report["budget"]
    assert budget["candidate"]["c1_updates"] == budget["control"]["c1_updates"] == 3 * 2
    assert budget["candidate"]["c1_global_step"] == budget["control"]["c1_global_step"] == 6
    for lineage in ("candidate", "control"):
        for record in records[lineage]:
            assert record["c1"]["updates_completed"] == 2 and record["c1"]["updates_planned"] == 2
            assert record["c1"]["canonical_rows_planned"] + record["c1"]["live_rows_planned"] == 2 * 8
            assert record["c1_counters"]["non_finite_losses"] == 0 and record["c1_counters"]["non_finite_gradients"] == 0
            assert record["c1"]["losses"]["all_finite"]
            assert record["integrity"]["non_finite_events"] == 0
            assert record["integrity"]["legality_failures"] == 0 and record["integrity"]["orientation_failures"] == 0
    # Same canonical cursor, same live draw seeds; the streams differ only through the pools.
    assert records["candidate"][2]["c1"]["cursor_after_planned"] == records["control"][2]["c1"]["cursor_after_planned"]
    assert records["candidate"][2]["c1"]["live_seeds"] == records["control"][2]["c1"]["live_seeds"]


def test_each_lineage_uses_only_its_own_matched_bundle(two_lineages):
    run_root, _records = two_lineages
    candidate = run_root / "candidate" / gp.BUNDLES_DIRECTORY / gp.bundle_name(3)
    control = run_root / "control" / gp.BUNDLES_DIRECTORY / gp.bundle_name(3)
    verify_bundle(candidate, expected_run_id="G3-PILOT-TEST", expected_lineage="candidate", expected_period=3)
    with pytest.raises(Phase18G3LineageError, match="never paired across lineages"):
        verify_bundle(candidate, expected_lineage="control")
    with pytest.raises(Phase18G3LineageError):
        verify_bundle(control, expected_lineage="candidate")
    with pytest.raises(Phase18G3Error, match="belongs to run"):
        verify_bundle(control, expected_run_id="OTHER")
    with pytest.raises(Phase18G3Error, match="is period"):
        verify_bundle(control, expected_period=2)
    same = compare_bundles(read_manifest(candidate), read_manifest(control))
    assert not same["setup_raw"] and not same["setup_ema"]
    # A resume refuses the other lineage's bundle outright.
    config = smoke_pilot_config(lineage="control", namespace=NAMESPACE, run_id="G3-PILOT-TEST")
    with pytest.raises(Phase18G3LineageError):
        gp.LineageRunner.resume(config, bundle_directory=candidate, run_root=run_root, corpus_root="unused", corpus_identity=read_manifest(candidate) and _identity_stub(), value_prior=UNIFORM_PRIOR, require_complete_split=False, log=lambda m: None)


def _identity_stub():
    from stratego.training.warmstart_checkpoint import CorpusIdentity

    return CorpusIdentity(corpus_version="synthetic_warmstart_corpus_v1", content_digest="0" * 64, metadata_digest="0" * 64, commit_index_digest="0" * 64)


def test_required_files_are_written_and_a_tampered_component_is_refused(two_lineages, tmp_path):
    run_root, _records = two_lineages
    bundle = run_root / "candidate" / gp.BUNDLES_DIRECTORY / gp.bundle_name(2)
    manifest = verify_bundle(bundle)
    for name, entry in manifest["components"].items():
        assert (bundle / entry["file"]).exists(), name
    assert manifest["bundle_id"] and manifest["parent_bundle_id"] == read_manifest(run_root / "candidate" / gp.BUNDLES_DIRECTORY / gp.bundle_name(1))["bundle_id"]
    assert [entry["period"] for entry in manifest["live_periods"]] == [1, 2]
    for name in (gp.PERIODS_NAME, gp.C1_ROWS_NAME, gp.INIT_NAME, gp.STATE_NAME):
        assert (run_root / "candidate" / name).exists()
    state = json.loads((run_root / "candidate" / gp.STATE_NAME).read_text())
    assert state["period"] == 3 and state["c1_global_step"] == 6
    # Tamper with one component: the bundle no longer verifies.
    import shutil

    copy_dir = tmp_path / "tampered"
    shutil.copytree(bundle, copy_dir)
    with (copy_dir / COLLECTOR_NAME).open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(Phase18G3Error, match="does not verify"):
        verify_bundle(copy_dir)
    # And a manifest edit breaks the bundle id.
    shutil.copytree(bundle, tmp_path / "edited")
    edited = json.loads((tmp_path / "edited" / "manifest.json").read_text())
    edited["period"] = 99
    (tmp_path / "edited" / "manifest.json").write_text(json.dumps(edited))
    with pytest.raises(Phase18G3Error, match="bundle_id"):
        read_manifest(tmp_path / "edited")


def test_duplicate_boards_collapse_to_one_row_and_both_games_attribute_to_it(corpus, tmp_path):
    """Basic duplicate-board handling: an identical played board in one pool is
    one buffer row (S10); every game on either copy attributes to that row."""
    import dataclasses

    from stratego.engine.constants import BLUE, RED
    from stratego.training.phase18.g3_collector import PeriodCollector
    from stratego.training.phase18.setup_buffer import SetupBuffer
    from stratego.training.phase18.setup_model import build_setup_model, state_dict_digest
    from stratego.training.phase18.setup_sampling import generate_pool

    torch.set_num_threads(1)
    config = smoke_pilot_config(namespace=NAMESPACE, run_id="G3-DUP-TEST", overrides={"slots": 4, "plies_per_period": 4000, "pool_size": 8})
    model = build_setup_model(device="cpu", seed=5)
    digest = state_dict_digest(model)
    samples = list(generate_pool(model, namespace=NAMESPACE, seed_index=1, snapshot_iteration=0, snapshot_digest=digest, count=8).samples)
    red = [s for s in samples if s.lane == RED]
    # Replace the second red row by an exact duplicate of the first (same played board).
    duplicate = dataclasses.replace(red[1], index=red[1].index)
    duplicate = dataclasses.replace(red[0], index=red[1].index)
    samples[samples.index(red[1])] = duplicate
    buffer = SetupBuffer(storage_duration=config.buffer_storage_periods)
    record = buffer.add_pool(samples, period=1)
    assert record["duplicates_collapsed"] == 1 and record["rows"] == 7
    collector = PeriodCollector(config, buffer, live_root=tmp_path / "live")
    collector.begin_period(1, samples, snapshot_digest=digest)
    collector.run_period()
    document = collector.end_period()
    assert document["completed"] >= 4
    shared = buffer.outcome_record(red[0].content_fingerprint)
    assert shared["count"] >= 2 and shared["ready"]
    assert buffer.outcomes_added == 2 * document["completed"]
    assert buffer.attribution_failures == 0


def test_checkpoint_save_and_resume_across_an_unfinished_game_reproduces_the_continuation(corpus, tmp_path):
    root, _identity = corpus
    report = restart_check(root=tmp_path, corpus_root=root, namespace=NAMESPACE + ":restart", restart_after=1)
    assert report["unfinished_games_at_save"] >= 1
    assert report["passed"], report
    assert all(report["comparisons"].values()) and all(report["bundle_comparison"].values())
    assert report["control_next_period_completed"] >= 1
