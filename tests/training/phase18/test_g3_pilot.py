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


# ---------------------------------------------------------------------------
# Corrective commit: consistent recovery, the gate bundle, fairness conditions
# ---------------------------------------------------------------------------


def test_resume_selects_the_latest_verified_bundle_and_replays_without_duplicates(corpus, tmp_path):
    """Cadence 2, horizon 4: run three periods (bundles 0, 1, 2; none at 3),
    interrupt, leave a partial newer bundle and a lying run_state.json, resume
    from bundle_2, replay period 3 and finish period 4: exactly one record per
    period, exactly K C1 rows per period, the progress after bundle_2 archived
    (never deleted), and the state naming the real last bundle."""
    import shutil
    from collections import Counter
    from pathlib import Path

    from stratego.training.phase18 import g3_live_store as ls

    torch.set_num_threads(1)
    config = smoke_pilot_config(namespace=NAMESPACE + ":cadence2", run_id="G3-RESUME-TEST", overrides={"periods": 4, "bundle_cadence_periods": 2})
    run_root = tmp_path / "run"
    runner = gp.LineageRunner.fresh(config, **keywords(run_root, corpus))
    try:
        first = runner.run(periods=3)
    finally:
        runner.close()
    lineage_root = run_root / "candidate"
    bundles = lineage_root / gp.BUNDLES_DIRECTORY
    # The gate bundle after period 1 exists alongside the cadence bundle at 2.
    assert sorted(p.name for p in bundles.iterdir()) == ["bundle_0000", "bundle_0001", "bundle_0002"]
    assert first[0]["bundle_planned"] and not first[2]["bundle_planned"]
    assert [r["period"] for r in gp.read_period_records(lineage_root)] == [1, 2, 3]
    assert len(gp.read_c1_rows(lineage_root)) == 3 * 2
    state_path = lineage_root / gp.STATE_NAME
    assert json.loads(state_path.read_text())["last_bundle"] == str(bundles / "bundle_0002")
    interrupted_period_3 = first[2]

    # Debris of an interruption: a partial newer bundle and a lying state file.
    (bundles / "bundle_0003.partial").mkdir()
    (bundles / "bundle_0003.partial" / "junk").write_text("x")
    state_path.write_text(json.dumps({"period": 3, "last_bundle": str(bundles / "bundle_0003"), "last_bundle_period": 3}))

    # Selection reads the bundles, not the state file, and skips a tampered newer one.
    copy = tmp_path / "copy"
    shutil.copytree(lineage_root, copy / "candidate")
    with (copy / "candidate" / gp.BUNDLES_DIRECTORY / "bundle_0002" / COLLECTOR_NAME).open("ab") as handle:
        handle.write(b"x")
    fallback = gp.select_resume_bundle(copy / "candidate", config)
    assert fallback["period"] == 1 and [s["period"] for s in fallback["skipped"]] == [2]

    resumed = gp.LineageRunner.resume(config, bundle_directory=None, **keywords(run_root, corpus))
    try:
        record = resumed.resume_record
        assert record["selection"]["period"] == 2 and record["selection"]["skipped"] == [] and resumed.period == 2
        archive = record["archive"]
        assert archive["archive"] and archive["period_records_archived"] == 1 and archive["c1_rows_archived"] == 2
        assert sorted(Path(m["from"]).name for m in archive["live_periods_moved"]) == [
            "period_0003.done.json", "period_0003.journal.jsonl", "period_0003.meta.jsonl", "period_0003.records"
        ]
        assert [Path(m["from"]).name for m in archive["bundles_moved"]] == ["bundle_0003.partial"]
        assert archive["previous_state_archived"]
        assert [r["period"] for r in gp.read_period_records(lineage_root)] == [1, 2]
        assert len(gp.read_c1_rows(lineage_root)) == 4
        assert ls.available_periods(lineage_root / gp.LIVE_DIRECTORY) == (1, 2)
        assert json.loads(state_path.read_text())["last_bundle"] == str(bundles / "bundle_0002")
        later = resumed.run()
    finally:
        resumed.close()
    assert [r["period"] for r in later] == [3, 4]
    records = gp.read_period_records(lineage_root)
    assert [r["period"] for r in records] == [1, 2, 3, 4]
    assert dict(Counter(r["period"] for r in gp.read_c1_rows(lineage_root))) == {1: 2, 2: 2, 3: 2, 4: 2}
    assert ls.available_periods(lineage_root / gp.LIVE_DIRECTORY) == (1, 2, 3, 4)
    assert sorted(p.name for p in bundles.iterdir()) == ["bundle_0000", "bundle_0001", "bundle_0002", "bundle_0004"]
    state = json.loads(state_path.read_text())
    assert state["period"] == 4 and state["last_bundle_period"] == 4 and state["last_bundle"] == str(bundles / "bundle_0004")
    assert state["resume"]["selection"]["period"] == 2
    # The replayed period 3 is the interrupted period 3.
    replayed = records[2]
    for key in ("completed_game_ids_digest", "outcome_records_digest", "plies_advanced", "in_flight_at_end"):
        assert replayed["collection"][key] == interrupted_period_3["collection"][key], key
    assert replayed["c1"]["keys_digests"] == interrupted_period_3["c1"]["keys_digests"]
    assert replayed["c1_state_digest"] == interrupted_period_3["c1_state_digest"]
    assert replayed["setup_raw_digest"] == interrupted_period_3["setup_raw_digest"]
    # The archive preserves everything that was made after bundle_2.
    archive_dir = Path(archive["archive"])
    assert json.loads((archive_dir / "periods_after.jsonl").read_text().splitlines()[0])["period"] == 3
    assert len((archive_dir / "c1_rows_after.jsonl").read_text().splitlines()) == 2
    assert sorted(p.name for p in (archive_dir / gp.LIVE_DIRECTORY).iterdir()) == [
        "period_0003.done.json", "period_0003.journal.jsonl", "period_0003.meta.jsonl", "period_0003.records"
    ]
    assert (archive_dir / gp.BUNDLES_DIRECTORY / "bundle_0003.partial" / "junk").exists()
    assert (archive_dir / gp.STATE_NAME).exists()
    # A resume that finds nothing after its bundle archives nothing.
    again = gp.LineageRunner.resume(config, bundle_directory=None, **keywords(run_root, corpus))
    try:
        assert again.period == 4 and again.resume_record["archive"]["archive"] is None
        with pytest.raises(Phase18G3Error, match="bounded horizon"):
            again.run_period()
    finally:
        again.close()


def test_recovery_refuses_a_history_it_cannot_restore_exactly(corpus, tmp_path):
    """A missing period record below the bundle period is corruption, not
    something to paper over: the resume refuses before touching anything."""
    torch.set_num_threads(1)
    config = smoke_pilot_config(namespace=NAMESPACE + ":refuse", run_id="G3-REFUSE-TEST", overrides={"periods": 2, "bundle_cadence_periods": 2})
    run_root = tmp_path / "run"
    runner = gp.LineageRunner.fresh(config, **keywords(run_root, corpus))
    try:
        runner.run()
    finally:
        runner.close()
    lineage_root = run_root / "candidate"
    periods_path = lineage_root / gp.PERIODS_NAME
    lines = periods_path.read_text().splitlines()
    periods_path.write_text(lines[1] + "\n")  # period 1's record lost
    with pytest.raises(Phase18G3Error, match="exactly one record"):
        gp.archive_progress_after(lineage_root, 2, updates_per_period=2)
    periods_path.write_text("\n".join(lines) + "\n")
    rows_path = lineage_root / gp.C1_ROWS_NAME
    rows = rows_path.read_text().splitlines()
    rows_path.write_text("\n".join(rows + [rows[-1]]) + "\n")  # a duplicated C1 row
    with pytest.raises(Phase18G3Error, match="C1 rows"):
        gp.archive_progress_after(lineage_root, 2, updates_per_period=2)
    rows_path.write_text("\n".join(rows) + "\n")
    assert gp.archive_progress_after(lineage_root, 2, updates_per_period=2)["archive"] is None
    # An interrupted append (a partial trailing line) is archived, not refused.
    with periods_path.open("a") as handle:
        handle.write('{"period": 3, "truncated')
    report = gp.archive_progress_after(lineage_root, 2, updates_per_period=2)
    assert report["partial_lines_archived"] == 1 and report["archive"]
    assert [r["period"] for r in gp.read_period_records(lineage_root)] == [1, 2]


def test_fairness_conditions_hold_on_the_smoke_pilot_and_fail_on_tampering(two_lineages, tmp_path):
    import shutil

    run_root, _records = two_lineages
    matching = gp.matching_check(run_root, c1_device="cpu") | {"contract_sha256": "c" * 64}
    fairness = gp.fairness_conditions(run_root, periods=3, updates_per_period=2, matching=matching, contract_sha256="c" * 64)
    assert fairness["all_hold"], fairness["problems"]
    assert all(fairness["conditions"].values())
    assert fairness["budgets"]["candidate"] == fairness["budgets"]["control"] == {"completed_updates": 6, "final_global_step": 6}
    assert all(entry["verified"] and entry["consistent_with_records"] for entry in fairness["final_bundles"].values())
    # Missing, failed or foreign matching evidence is a failed condition, never telemetry.
    missing = gp.fairness_conditions(run_root, periods=3, updates_per_period=2, matching=None, contract_sha256="c" * 64)
    assert not missing["all_hold"] and not missing["conditions"]["matching_record_present"] and any("not telemetry" in p for p in missing["problems"])
    failed = gp.fairness_conditions(run_root, periods=3, updates_per_period=2, matching=dict(matching, matched=False, problems=["x"]), contract_sha256="c" * 64)
    assert not failed["all_hold"] and not failed["conditions"]["matching_record_matched"]
    stale = gp.fairness_conditions(run_root, periods=3, updates_per_period=2, matching=matching, contract_sha256="d" * 64)
    assert not stale["all_hold"] and not stale["conditions"]["matching_record_contract_bound"]
    # A duplicated period record breaks the exact-history condition.
    duplicated = tmp_path / "dup"
    shutil.copytree(run_root, duplicated)
    periods_path = duplicated / "candidate" / gp.PERIODS_NAME
    lines = periods_path.read_text().splitlines()
    periods_path.write_text("\n".join(lines + [lines[-1]]) + "\n")
    dup = gp.fairness_conditions(duplicated, periods=3, updates_per_period=2, matching=matching, contract_sha256="c" * 64)
    assert not dup["all_hold"] and not dup["conditions"]["candidate_periods_exactly_1_to_3"]
    # A control whose records disagree with its recorded initial digest is not frozen.
    tampered = tmp_path / "init"
    shutil.copytree(run_root, tampered)
    init_path = tampered / "control" / gp.INIT_NAME
    init = json.loads(init_path.read_text())
    init["setup_init_state_digest"] = "0" * 64
    init_path.write_text(json.dumps(init))
    frozen = gp.fairness_conditions(tampered, periods=3, updates_per_period=2, matching=matching, contract_sha256="c" * 64)
    assert not frozen["all_hold"] and not frozen["conditions"]["control_setup_digest_unchanged"]
    # A horizon the lineages never reached has no complete final bundle and an incomplete budget.
    short = gp.fairness_conditions(run_root, periods=4, updates_per_period=2, matching=matching, contract_sha256="c" * 64)
    assert not short["all_hold"]
    assert not short["conditions"]["candidate_final_bundle_complete"] and not short["conditions"]["equal_completed_c1_update_budget"]


def test_the_period_one_gate_passes_with_completed_games_and_unequal_raw_digests(corpus, tmp_path):
    """The first pilot's failure mode reproduced and corrected: period 1 completes
    games in both lineages, so the raw live commit digests differ (each hashed
    metadata line carries the lineage stamp); the gate reads the semantic
    identity instead, both lineages serve exactly the planned mixture, and
    `matched` is true. A non-lineage difference in a store fails the gate."""
    import shutil

    torch.set_num_threads(1)
    run_root = tmp_path / "run"
    records = {}
    for lineage in ("candidate", "control"):
        config = smoke_pilot_config(lineage=lineage, namespace=NAMESPACE + ":gate", run_id="G3-GATE-TEST", overrides={"periods": 1, "slots": 2, "plies_per_period": 400})
        runner = gp.LineageRunner.fresh(config, **keywords(run_root, corpus))
        try:
            records[lineage] = runner.run()
        finally:
            runner.close()
    first = {lineage: records[lineage][0] for lineage in records}
    assert first["candidate"]["collection"]["completed"] >= 1
    assert first["candidate"]["collection"]["live"]["commit_digest"] != first["control"]["collection"]["live"]["commit_digest"]

    report = gp.matching_check(run_root, c1_device="cpu")
    assert report["matched"], report["problems"]
    assert all(v for k, v in report["period_1"].items() if k != "c1_state_digest_note")
    assert report["period_1"]["collection/live/semantic_identity"] is True
    assert report["period_1"]["collection/live/raw_commit_digest_recorded"] is True
    assert report["period_1"]["c1/served_rows_equal_planned"] is True
    assert "collection/live/commit_digest" not in report["period_1"]
    live = report["live_store"]
    assert live["matched"] and live["raw_commit_digest_equal"] is False
    assert live["raw_commit_digest"] == {lineage: first[lineage]["collection"]["live"]["commit_digest"] for lineage in first}
    assert "lineage stamp" in live["raw_commit_digest_note"]
    assert all(live["semantic"]["checks"].values())
    assert live["semantic"]["lineage_neutral_commit_digest"]["candidate"] == live["semantic"]["lineage_neutral_commit_digest"]["control"]
    assert live["semantic"]["commits"] == {"candidate": first["candidate"]["collection"]["completed"], "control": first["control"]["collection"]["completed"]}
    for lineage in first:
        c1 = first[lineage]["c1"]
        assert c1["live_rows_served"] == c1["live_rows_planned"] == 2 * 4
        assert c1["canonical_rows_served"] == c1["canonical_rows_planned"] == 2 * 4
        assert c1["served_equals_planned"] is True
        rows = gp.read_c1_rows(run_root / lineage)
        assert [(r["canonical_examples"], r["live_examples"], r["examples"]) for r in rows] == [(4, 4, 8)] * 2
        assert [r["live_seed"] for r in rows] == c1["live_seeds"]
    assert report["budget"]["candidate"]["live_rows_served"] == report["budget"]["control"]["live_rows_served"] == 8
    assert report["budget"]["candidate"]["canonical_rows_served"] == report["budget"]["control"]["canonical_rows_served"] == 8

    # A non-lineage metadata difference in the control store fails the gate.
    tampered = tmp_path / "tampered"
    shutil.copytree(run_root, tampered)
    meta_path = tampered / "control" / gp.LIVE_DIRECTORY / "period_0001.meta.jsonl"
    lines = meta_path.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["blue_policy_seed"] = int(entry["blue_policy_seed"]) + 1
    lines[0] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    meta_path.write_text("\n".join(lines) + "\n")
    failed = gp.matching_check(tampered, c1_device="cpu")
    assert not failed["matched"]
    assert failed["period_1"]["collection/live/semantic_identity"] is False
    assert any("blue_policy_seed" in p for p in failed["live_store"]["problems"])
    assert any("semantically identical" in p for p in failed["problems"])

    # A period record whose served counts do not equal the plan fails the gate.
    short = tmp_path / "short"
    shutil.copytree(run_root, short)
    periods_path = short / "candidate" / gp.PERIODS_NAME
    record = json.loads(periods_path.read_text().splitlines()[0])
    record["c1"]["live_rows_served"] = 0
    periods_path.write_text(json.dumps(record, sort_keys=True) + "\n")
    failed = gp.matching_check(short, c1_device="cpu")
    assert not failed["matched"] and failed["period_1"]["c1/served_rows_equal_planned"] is False
    assert failed["period_1"]["c1/live_rows_served"] is False


def test_decision_input_requires_every_condition():
    passes = {"passes": True, "near_boundary": False}
    gates = {"G1": True, "G2": True, "G10_clean_deliverable": None}
    good = {"all_hold": True, "problems": []}
    assert gp.decision_input(passes, gates, good)["decision"] == "PROCEED"
    blocked = gp.decision_input(passes, gates, {"all_hold": False, "problems": ["the lineage-matching record is missing"]})
    assert blocked["decision"] == "BLOCKED" and "missing" in blocked["basis"]
    assert gp.decision_input(passes, dict(gates, G2=False), good)["decision"] == "BLOCKED"
    assert gp.decision_input({"passes": False, "near_boundary": True}, gates, good)["decision"] == "NEAR_BOUNDARY"
    assert gp.decision_input({"passes": False, "near_boundary": False}, gates, good)["decision"] == "FAIL"
    assert gp.decision_input(passes, gates, good)["uncomputed_gates"] == ["G10_clean_deliverable"]
