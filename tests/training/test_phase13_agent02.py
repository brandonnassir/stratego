"""Phase 13 Agent 2: tests for the integrated Phase 14 final-training runner.

Two kinds of test live here.

*Mechanical* checks — the contract binding, the seed streams, the pool
function, the clock, the schedule, the checkpoint format, the storage guard,
the telemetry surface — are pure and cost milliseconds. They are where the
frozen values are actually pinned.

*Integration* checks run the real runner end to end on a **scaled population**
and a **manual clock**, the two declared test seams. One session-scoped fixture
walks a whole scripted run — start, three units either side of the 132-hour
transition, a resume, and the 168-hour stop — and every end-to-end assertion
reads from that one walk rather than paying for its own.

No 90-minute rehearsal and no strength tournament happens here, by design:
Agent 2 proves the machinery runs, not that the model is good.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from stratego.training import phase14_contract as contract
from stratego.training import phase14_pool as pool_module
from stratego.training import phase14_seed as seeds
from stratego.training.phase14_clock import (
    DeadlineController,
    ManualClock,
    Phase14ClockError,
    RunWindow,
    SystemClock,
    archive_index_for_elapsed,
    candidate_index_for_elapsed,
    hot_index_for_elapsed,
    parse_utc,
    require_production_clock,
    segment_for_elapsed,
)
from stratego.training.phase14_config import (
    integrated_config_digest,
    integrated_config_document,
)
from stratego.training.phase14_contract import Population
from stratego.training.phase14_pool import ActivePool, HistoricalArchive
from stratego.training.phase14_telemetry import (
    ControlSurface,
    Phase14TelemetryError,
    build_snapshot,
    missing_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CONTRACT = REPO_ROOT / contract.FROZEN_CONTRACT_RELATIVE_PATH

pytestmark = pytest.mark.skipif(
    not FROZEN_CONTRACT.exists(), reason="phase13 agent 1 contract not present"
)

#: The scaled population every integration test uses. Small enough that a unit
#: costs seconds, wide enough that all four buckets and all five handcrafted
#: behaviours still appear.
TEST_POPULATION = Population.scaled(512)


# ---------------------------------------------------------------------------
# The contract binding
# ---------------------------------------------------------------------------


def test_implementation_matches_the_frozen_agent_1_contract():
    assert contract.verify_against_frozen_contract() == []
    report = contract.assert_matches_frozen_contract()
    assert report["disagreements"] == 0
    assert report["implementation_contract_digest"] == contract.contract_digest()


def test_mixtures_are_exactly_the_frozen_counts():
    main = contract.bucket_counts("main")
    late = contract.bucket_counts("late")
    assert sum(main.values()) == sum(late.values()) == 2048
    assert main["current"] == 1188 and main["historical"] == 615
    assert late["current"] == 819 and late["historical"] == 984
    # The handcrafted share is identical in both segments, exactly as frozen.
    assert main["rule"] == late["rule"] == 122
    assert main["stress"] == late["stress"] == 123
    assert sum(contract.HANDCRAFTED_COUNTS.values()) == 245


def test_realized_percentages_sit_in_the_frozen_bands():
    for segment, current, historical in (("main", 0.58008, 0.30029), ("late", 0.3999, 0.48047)):
        counts = contract.bucket_counts(segment)
        total = sum(counts.values())
        assert counts["current"] / total == pytest.approx(current, abs=5e-5)
        assert counts["historical"] / total == pytest.approx(historical, abs=5e-5)
        handcrafted = (counts["rule"] + counts["stress"]) / total
        assert 0.10 <= handcrafted <= 0.15
        assert 0.85 <= (counts["current"] + counts["historical"]) / total <= 0.90


def test_handcrafted_layout_is_contiguous_and_exact():
    for bucket, families in (
        ("rule", contract.RULE_FAMILY_ORDER),
        ("stress", contract.STRESS_FAMILY_ORDER),
    ):
        seen = {}
        total = sum(contract.HANDCRAFTED_COUNTS[name] for name in families)
        for ordinal in range(total):
            name = contract.handcrafted_policy_for_ordinal(bucket, ordinal)
            seen[name] = seen.get(name, 0) + 1
        assert seen == {name: contract.HANDCRAFTED_COUNTS[name] for name in families}
        with pytest.raises(contract.Phase14ContractError):
            contract.handcrafted_policy_for_ordinal(bucket, total)


def test_learning_rate_and_entropy_are_frozen_per_segment():
    assert contract.learning_rate("main") == 7.5e-05 == 0.25 * contract.LR9
    assert contract.learning_rate("late") == 3.75e-05 == 0.125 * contract.LR9
    assert contract.entropy_coefficient() == 0.001
    with pytest.raises(contract.Phase14ContractError):
        contract.learning_rate("middle")


def test_candidate_hours_are_the_frozen_twenty_nine():
    assert contract.CANDIDATE_HOURS[0] == 0
    assert contract.CANDIDATE_HOURS[-1] == 168
    assert contract.CANDIDATE_COUNT == 29
    assert all(hour % 6 == 0 for hour in contract.CANDIDATE_HOURS)


def test_belief_auxiliary_is_retained_at_the_accepted_weight():
    values = contract.inherited_phase9_values()
    assert values["belief_loss_weight"] == 0.25
    assert values["value_loss_weight"] == 0.5
    assert values["ppo_clip_epsilon"] == 0.20
    assert values["minibatch_size"] == 512
    assert values["epochs_per_rollout"] == 2


def test_ema_absence_is_recorded_rather_than_omitted():
    assert contract.EMA_PRESENT is False
    assert "no EMA" in contract.EMA_STATE_RECORD


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def test_game_ids_round_trip_and_refuse_foreign_identifiers():
    identifier = seeds.game_id(12, "historical", 300)
    fields = seeds.parse_game_id(identifier)
    assert fields["iteration"] == 12
    assert fields["bucket"] == "historical"
    assert fields["ordinal"] == 300
    assert fields["namespace"] == contract.PHASE14_NAMESPACE
    for foreign in (
        "phase9_rollout_v1|ms=2026081601|ns=canonical|it=012|b=historical|g=0300",
        "phase14_rollout_v1|ms=1|ns=phase14|it=0012|b=historical|g=0300",
        "nonsense",
    ):
        with pytest.raises(seeds.Phase14SeedError):
            seeds.parse_game_id(foreign)


def test_a_phase9_parser_refuses_a_phase14_id():
    from stratego.training.phase9_seed import Phase9SeedError, parse_phase9_game_id

    with pytest.raises(Phase9SeedError):
        parse_phase9_game_id(seeds.game_id(1, "current", 0))


def test_streams_are_domain_separated():
    identifier = seeds.game_id(3, "rule", 5)
    red = seeds.side_selector_seed(identifier, "red")
    blue = seeds.side_selector_seed(identifier, "blue")
    assert red != blue
    assert seeds.policy_seed(identifier, "red") not in (red, blue)
    assert seeds.setup_root_seed(identifier) not in (red, blue)
    uniforms = [seeds.action_sampling_uniform(identifier, ply) for ply in range(64)]
    assert all(0.0 < value <= 1.0 for value in uniforms)
    assert len(set(uniforms)) == len(uniforms)


def test_seed_streams_do_not_collide_with_phase10b():
    from stratego.training import phase10b_seed

    identifier = seeds.game_id(1, "current", 0)
    other = phase10b_seed.game_id(1, "current", 0)
    assert seeds.derive_seed(seeds.DOMAIN_RED_SETUP, identifier) != phase10b_seed.derive_seed(
        phase10b_seed.DOMAIN_RED_SETUP, other
    )


def test_ordinal_validation_is_segment_aware():
    identifier = seeds.game_id(1, "current", 1000)
    assert seeds.validate_ordinal(identifier, "main")["ordinal"] == 1000
    with pytest.raises(seeds.Phase14SeedError):
        seeds.validate_ordinal(identifier, "late")  # late holds only 819 current games


# ---------------------------------------------------------------------------
# The active pool
# ---------------------------------------------------------------------------


def _archive(k: int) -> HistoricalArchive:
    archive = HistoricalArchive()
    for index in range(1, k + 1):
        archive.append(
            archive_mark=index,
            path=f"/tmp/phase14/archive_{index:04d}.pt",
            sha256=f"{index:064x}",
            model_state_digest=f"{index:064x}",
            elapsed_seconds=7200.0 * index,
            written_utc="2026-08-21T00:00:00.000Z",
            iteration=index,
            global_optimizer_step=100 * index,
        )
    return archive


def test_pool_membership_is_bounded_and_distinct_at_every_archive_size():
    for k in range(0, 85):
        bands = pool_module.active_pool_positions(k)
        members = [position for band in bands.values() for position in band]
        assert len(members) == len(set(members))
        assert len(members) == min(k, contract.POOL_SNAPSHOT_SLOTS)
        assert all(1 <= position <= k for position in members)
        if k > contract.POOL_SNAPSHOT_SLOTS:
            assert bands["recent"] == tuple(range(k - 5, k + 1))
            assert len(bands["older"]) == 4 and len(bands["middle"]) == 4


def test_anchors_are_permanent_and_the_pool_never_exceeds_sixteen():
    for k in (0, 1, 7, 14, 15, 40, 84):
        pool = ActivePool.for_archive(_archive(k))
        assert pool.categories["anchor"] == ("P8", "P9")
        assert len(pool.members()) == min(2 + k, contract.POOL_SIZE)


def test_empty_category_weight_is_redistributed():
    from fractions import Fraction

    empty = ActivePool.for_archive(_archive(0)).member_weights()
    assert empty == {"P8": Fraction(1, 2), "P9": Fraction(1, 2)}
    single = ActivePool.for_archive(_archive(1)).member_weights()
    assert single["S0001"] == Fraction(8, 10)
    assert single["P8"] == single["P9"] == Fraction(1, 10)
    full = ActivePool.for_archive(_archive(20))
    weights = full.member_weights()
    assert sum(weights.values()) == 1
    assert sum(weights[name] for name in full.categories["recent"]) == Fraction(3, 10)


def test_historical_bucket_is_partitioned_exactly_never_sampled():
    for k in (0, 1, 3, 14, 20, 84):
        pool = ActivePool.for_archive(_archive(k))
        for total in (615, 984):
            counts = pool_module.exact_member_counts(total, pool, 1)
            assert sum(counts.values()) == total
            assert set(counts) == set(pool.members())
            ranges = pool_module.member_ordinal_ranges(total, pool, 1)
            assert ranges[0][1] == 0
            assert ranges[-1][2] == total
            for ordinal in (0, total // 2, total - 1):
                assert pool_module.member_for_ordinal(ordinal, total, pool, 1) in counts


def test_remainder_rotates_across_iterations():
    pool = ActivePool.for_archive(_archive(20))
    profiles = {
        iteration: tuple(
            pool_module.exact_member_counts(615, pool, iteration)[name]
            for name in pool.members()
        )
        for iteration in range(1, 6)
    }
    assert len(set(profiles.values())) > 1
    assert all(sum(profile) == 615 for profile in profiles.values())


def test_pool_digest_detects_a_membership_change_and_resume_refuses_it():
    archive = _archive(20)
    recorded = ActivePool.for_archive(archive).to_dict()
    assert pool_module.assert_pool_matches(archive, recorded).k == 20
    grown = _archive(21)
    with pytest.raises(pool_module.Phase14PoolError):
        pool_module.assert_pool_matches(grown, recorded)


def test_archive_ordering_is_its_identity():
    archive = _archive(3)
    payload = archive.to_dict()
    payload["entries"][0]["position"] = 2
    with pytest.raises(pool_module.Phase14PoolError):
        HistoricalArchive.from_dict(payload)


# ---------------------------------------------------------------------------
# The clock, the deadline and the test seam
# ---------------------------------------------------------------------------


def test_window_is_exactly_168_hours_with_a_132_hour_transition():
    clock = ManualClock("2026-09-01T00:00:00.000Z")
    window = RunWindow.start(clock.now())
    assert (window.run_deadline_utc - window.run_start_utc).total_seconds() == 604800
    assert (window.transition_utc - window.run_start_utc).total_seconds() == 475200


def test_segment_switches_at_the_exact_frozen_mark():
    assert segment_for_elapsed(475199.0) == "main"
    assert segment_for_elapsed(475200.0) == "late"
    assert segment_for_elapsed(475200.1) == "late"


def test_cadence_indices_follow_the_frozen_cadences():
    assert hot_index_for_elapsed(899) == 0 and hot_index_for_elapsed(900) == 1
    assert archive_index_for_elapsed(7199) == 0 and archive_index_for_elapsed(7200) == 1
    assert archive_index_for_elapsed(10**9) == contract.ARCHIVE_SNAPSHOTS_IN_RUN
    assert candidate_index_for_elapsed(21599) == 0
    assert candidate_index_for_elapsed(21600) == 1
    assert candidate_index_for_elapsed(10**9) == contract.CANDIDATE_COUNT - 1


def test_downtime_counts_and_a_restart_never_creates_a_new_deadline():
    clock = ManualClock("2026-09-01T00:00:00.000Z")
    controller = DeadlineController.start(clock)
    persisted = controller.window.to_dict()
    clock.advance_hours(50)  # the machine was off for fifty hours
    resumed = DeadlineController.resume(persisted, clock)
    assert resumed.window.to_dict() == persisted
    assert resumed.elapsed_hours() == pytest.approx(50.0)
    assert resumed.remaining() == pytest.approx((168 - 50) * 3600)
    clock.advance_hours(118.1)
    assert resumed.expired() is True
    assert resumed.may_start_collection_unit() is False
    assert resumed.may_start_optimizer_step() is False


def test_a_window_that_is_not_168_hours_is_refused():
    start = parse_utc("2026-09-01T00:00:00.000Z")
    with pytest.raises(Phase14ClockError):
        RunWindow(
            run_start_utc=start,
            run_deadline_utc=start.replace(day=2),
            transition_utc=start.replace(day=2),
        )


def test_production_refuses_the_test_clock():
    assert require_production_clock(SystemClock()) is not None
    with pytest.raises(Phase14ClockError):
        require_production_clock(ManualClock("2026-09-01T00:00:00.000Z"))


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def setup_source():
    from stratego.training.phase14_setup_source import Phase14SetupSource

    return Phase14SetupSource.build()


def test_iteration_composition_matches_the_frozen_mixture(setup_source):
    from collections import Counter

    from stratego.training.phase14_schedule import iter_iteration_schedule

    pool = ActivePool.for_archive(_archive(20))
    for segment in ("main", "late"):
        records = list(
            iter_iteration_schedule(7, segment=segment, pool=pool, setup_source=setup_source)
        )
        assert len(records) == 2048
        kinds = Counter(record.opponent_kind for record in records)
        counts = contract.bucket_counts(segment)
        assert kinds["current_policy"] == counts["current"]
        assert kinds["historical_snapshot"] == counts["historical"]
        handcrafted = Counter(
            record.handcrafted_policy_id for record in records if record.handcrafted_policy_id
        )
        assert dict(handcrafted) == contract.HANDCRAFTED_COUNTS
        members = {record.historical_snapshot_identity for record in records} - {None}
        assert members <= set(pool.members())


def test_scheduled_records_rebuild_purely_from_their_identifiers(setup_source):
    from stratego.training.phase14_schedule import (
        rebuild_scheduled_game,
        scheduled_game_record,
    )

    pool = ActivePool.for_archive(_archive(9))
    for bucket, ordinal in (("current", 3), ("historical", 100), ("rule", 70), ("stress", 90)):
        record = scheduled_game_record(
            4, bucket, ordinal, segment="main", pool=pool, setup_source=setup_source
        )
        rebuilt = rebuild_scheduled_game(
            record.rollout_game_id, segment="main", pool=pool, setup_source=setup_source
        )
        assert rebuilt == record
        assert record.phase9_game_id == record.rollout_game_id


def test_colour_balance_splits_each_asymmetric_bucket(setup_source):
    from collections import Counter

    from stratego.training.phase14_schedule import iter_iteration_schedule

    pool = ActivePool.for_archive(_archive(5))
    records = list(
        iter_iteration_schedule(2, segment="main", pool=pool, setup_source=setup_source)
    )
    asymmetric = [record for record in records if record.learner_color is not None]
    colours = Counter(record.learner_color for record in asymmetric)
    assert abs(colours["red"] - colours["blue"]) <= 2
    self_play = [record for record in records if record.learner_color is None]
    assert all(record.learner_control == "both" for record in self_play)


# ---------------------------------------------------------------------------
# The setup source
# ---------------------------------------------------------------------------


def test_setup_source_orients_blue_through_the_accepted_helper(setup_source):
    from stratego.training.phase14_setup_source import assert_orientation_path

    probe = assert_orientation_path(setup_source, seeds.game_id(1, "current", 0))
    assert probe["engine_is_oriented"] is True
    assert probe["canonical_differs_from_oriented"] is True


def test_setup_assignments_are_pure_functions_of_the_game_id(setup_source):
    from stratego.training.phase14_setup_source import (
        Phase14SetupSource,
        validate_assignment_provenance,
    )

    identifier = seeds.game_id(2, "current", 1)
    first = setup_source.assign(root_seed=0, environment_id=0, generation=0, game_id=identifier)
    second = Phase14SetupSource.build().assign(
        root_seed=0, environment_id=0, generation=0, game_id=identifier
    )
    assert first.red_setup == second.red_setup
    assert first.blue_setup == second.blue_setup
    assert validate_assignment_provenance(first.provenance) == []
    assert first.provenance["setup_source_identity"] == "phase14_setup_source_v1"


def test_the_phase11b_glue_is_not_on_the_training_path():
    source = (REPO_ROOT / "stratego" / "training" / "phase14_setup_source.py").read_text()
    assert "phase11b" not in source.replace(
        "stratego/belief/phase11b/corpus.py", ""
    ).replace("Phase11BSetupSources", "").replace("Phase 11B", "")


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_payload():
    """A real payload built from the accepted starting checkpoint."""
    import torch

    from stratego.training.phase14_checkpoint import build_payload, SNAPSHOT_ROLE_HOT
    from stratego.training.phase14_trainer import load_starting_model

    model = load_starting_model(device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=contract.MAIN_LEARNING_RATE)
    window = RunWindow.start(parse_utc("2026-09-01T00:00:00.000Z")).to_dict()
    return build_payload(
        model=model,
        optimizer=optimizer,
        snapshot_role=SNAPSHOT_ROLE_HOT,
        trainer_state={"global_optimizer_step": 7, "cursor": None},
        run_window=window,
        schedule_state={"segment": "main"},
        population_schedule_state={"iteration": 1},
        active_historical_pool=ActivePool.for_archive(_archive(2)).to_dict(),
        historical_archive_state=_archive(2).to_dict(),
        shard_cursor={"last_committed_iteration": 1},
        storage_state={"free_gib": 900},
        candidate_evaluation_state={"marks": 1},
        device="cpu",
    )


def test_hot_payload_covers_every_frozen_resume_field(sample_payload):
    from stratego.training.phase14_checkpoint import _covers

    for field in contract.HOT_CHECKPOINT_REQUIRED_FIELDS:
        assert _covers(sample_payload, field), field
    assert sample_payload["ema_state"]["present"] is False
    assert sample_payload["run_window"]["deadline_seconds"] == 604800


def test_a_payload_without_the_window_is_refused():
    import torch

    from stratego.training.phase14_checkpoint import (
        Phase14CheckpointError,
        SNAPSHOT_ROLE_HOT,
        build_payload,
    )
    from stratego.training.phase14_trainer import load_starting_model

    model = load_starting_model(device="cpu")
    with pytest.raises(Phase14CheckpointError):
        build_payload(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
            snapshot_role=SNAPSHOT_ROLE_HOT,
            trainer_state={"global_optimizer_step": 0},
            run_window={"run_start_utc": "2026-09-01T00:00:00.000Z"},
            schedule_state={},
            population_schedule_state={},
            active_historical_pool=ActivePool.for_archive(_archive(0)).to_dict(),
            historical_archive_state=_archive(0).to_dict(),
            shard_cursor={},
            storage_state={},
            candidate_evaluation_state={},
            device="cpu",
        )


def test_hot_ring_validates_before_pruning_and_keeps_four(tmp_path, sample_payload):
    from stratego.training.phase14_checkpoint import HotCheckpointRing, is_valid

    ring = HotCheckpointRing(tmp_path / "hot")
    for _ in range(6):
        ring.write(sample_payload, fsync=False)
    files = ring.files()
    assert len(files) == contract.HOT_CHECKPOINT_RETAIN
    assert all(is_valid(path) for path in files)
    latest = ring.latest_valid()
    assert latest is not None and latest.name == files[0].name

    # A torn newest file costs one cadence, not the run.
    latest.write_bytes(b"not a checkpoint")
    assert is_valid(latest) is False
    recovered = ring.latest_valid()
    assert recovered is not None and recovered != latest


def test_a_tampered_payload_is_refused(tmp_path, sample_payload):
    import torch

    from stratego.training.phase14_checkpoint import Phase14CheckpointError, read, save

    path = tmp_path / "hot.pt"
    save(sample_payload, path, fsync=False)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["trainer_state"]["global_optimizer_step"] = 999
    torch.save(payload, path)
    with pytest.raises(Phase14CheckpointError):
        read(path)


def test_the_archive_is_append_only(tmp_path, sample_payload):
    from stratego.training.phase14_checkpoint import (
        Phase14CheckpointError,
        SNAPSHOT_ROLE_ARCHIVE,
        write_archive_snapshot,
    )

    payload = dict(sample_payload)
    payload["snapshot_role"] = SNAPSHOT_ROLE_ARCHIVE
    from stratego.training.warmstart_checkpoint import payload_integrity_digest

    payload["integrity_digest"] = payload_integrity_digest(
        {key: value for key, value in payload.items() if key != "integrity_digest"}
    )
    write_archive_snapshot(tmp_path, payload, position=1, fsync=False)
    with pytest.raises(Phase14CheckpointError):
        write_archive_snapshot(tmp_path, payload, position=1, fsync=False)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_the_deletion_guard_refuses_everything_outside_the_phase14_shards(tmp_path):
    from stratego.training.phase14_storage import (
        Phase14Storage,
        Phase14StorageError,
        assert_not_project_evidence,
    )

    storage = Phase14Storage.under(tmp_path)
    shard = storage.rollout_root / "phase14" / "iteration_001" / "shards" / "w00_s0000.stgshard"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(b"x")
    assert assert_not_project_evidence(shard, storage) == shard.resolve()
    for forbidden in (
        REPO_ROOT / "checkpoints" / "phase9" / "selfplay_c1_v1.pt",
        storage.archive_root / "archive_0001.pt",
        shard.with_suffix(".meta.jsonl"),
    ):
        with pytest.raises(Phase14StorageError):
            assert_not_project_evidence(forbidden, storage)


def test_rolling_deletion_needs_both_authorization_and_a_disposable_mark(tmp_path):
    from stratego.training.phase14_storage import (
        Phase14Storage,
        Phase14StorageError,
        execute_rolling_deletion,
        mark_shards_disposable,
        plan_rolling_deletion,
        read_disposable_mark,
    )

    storage = Phase14Storage.under(tmp_path)
    directory = storage.iteration_directory(1)
    (directory / "shards").mkdir(parents=True, exist_ok=True)
    (directory / "shards" / "w00_s0000.stgshard").write_bytes(b"x" * 32)

    plan = plan_rolling_deletion(storage)
    assert plan["authorized"] is False  # a healthy volume authorizes nothing
    assert plan["delete"] == []
    with pytest.raises(Phase14StorageError):
        execute_rolling_deletion(storage, plan)

    mark = mark_shards_disposable(directory, iteration=1, reason="test")
    assert mark["consumed"] and mark["disposable"] and mark["safe_to_delete"]
    assert read_disposable_mark(directory, 1) == mark


def test_storage_layout_matches_the_frozen_paths():
    from stratego.training.phase14_storage import Phase14Storage

    production = Phase14Storage.production()
    assert str(production.external_root) == contract.EXTERNAL_RUN_DIRECTORY
    assert production.hot_root == contract.repository_root() / contract.HOT_CHECKPOINT_DIRECTORY
    assert production.is_production_layout() is True
    assert production.archive_root.name == "archive"
    assert production.rollout_root.name == "rollouts"


# ---------------------------------------------------------------------------
# Telemetry and control
# ---------------------------------------------------------------------------


def test_a_snapshot_covers_the_whole_frozen_metric_list():
    from stratego.training.phase14_telemetry import METRIC_PATHS

    sections: dict = {}
    for section, key in METRIC_PATHS.values():
        sections.setdefault(section, {})[key] = 0
    snapshot = build_snapshot(
        clock=sections.get("clock", {}),
        training=sections.get("training", {}),
        collection=sections.get("collection", {}),
        population=sections.get("population", {}),
        checkpoints=sections.get("checkpoints", {}),
        candidates=sections.get("candidates", {}),
        storage=sections.get("storage", {}),
        workers=sections.get("workers", {}),
        counters=sections.get("counters", {}),
        failures={},
    )
    assert missing_metrics(snapshot) == []
    assert set(contract.FROZEN_METRIC_LIST) == set(METRIC_PATHS)


def test_the_control_surface_offers_only_emergency_stop():
    control = ControlSurface()
    assert control.should_continue() is True
    for key in contract.IMMUTABLE_CONTROL_KEYS:
        with pytest.raises(Phase14TelemetryError):
            control.set(key, 1)
    with pytest.raises(Phase14TelemetryError):
        control.set("anything_else", 1)
    assert control.emergency_stop("test")["stop_requested"] is True
    assert control.should_continue() is False


# ---------------------------------------------------------------------------
# Identity binding
# ---------------------------------------------------------------------------


def test_the_integrated_config_binds_every_section_17_input():
    document = integrated_config_document()
    for key in (
        "starting_checkpoint",
        "training_objective",
        "learning_rate",
        "transition",
        "opponent_mixture",
        "historical_pool",
        "setup_source",
        "checkpoint_cadences",
        "candidate_evaluation",
        "storage_policy",
        "deadline_semantics",
    ):
        assert key in document, key
    assert document["learning_rate"]["main"] == contract.MAIN_LEARNING_RATE
    assert document["learning_rate"]["late"] == contract.LATE_LEARNING_RATE
    assert document["transition"]["transition_seconds"] == 475200
    assert document["deadline_semantics"]["deadline_seconds"] == 604800
    assert document["candidate_evaluation"]["pack_digest"] == contract.SELECTION_PACK_DIGEST
    assert document["starting_checkpoint"]["sha256"] == contract.STARTING_CHECKPOINT_SHA256
    assert integrated_config_digest() == integrated_config_digest()
    assert len(integrated_config_digest()) == 64


# ---------------------------------------------------------------------------
# Search is absent from the training path
# ---------------------------------------------------------------------------


def test_no_search_module_is_imported_by_the_phase14_training_path():
    """Walk the real import graph rather than trusting a grep."""
    program = (
        "import sys;"
        "import stratego.training.phase14_runner;"
        "import stratego.training.phase14_trainer;"
        "import stratego.training.phase14_collector;"
        "import stratego.evaluation.phase14_candidates;"
        "print([name for name in sys.modules if name.startswith('stratego.search')])"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=True,
    )
    assert result.stdout.strip() == "[]"


def test_the_contract_states_the_search_prohibition():
    frozen = json.loads(FROZEN_CONTRACT.read_text())
    assert all(
        frozen["search_prohibition"][key] == "NOT USED"
        for key in ("phase12_tiny", "phase12_small", "phase12_medium")
    )
    assert "search" in contract.SEARCH_PROHIBITION


# ---------------------------------------------------------------------------
# End to end: one scripted run on the two declared test seams
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scripted_run(tmp_path_factory):
    """Start, three units across the transition, a resume, and the 168h stop.

    One walk, reused by every end-to-end assertion below: the scaled population
    keeps a unit at seconds rather than the ~20 minutes 2,048 games would cost,
    and the manual clock is the only way a short test can stand at hour 132.
    """
    from stratego.training.phase14_runner import MODE_TEST, Phase14Runner
    from stratego.training.phase14_storage import Phase14Storage
    from stratego.training.phase9_trainer import LoaderTopology

    root = tmp_path_factory.mktemp("phase14_run")
    storage = Phase14Storage.under(root)
    clock = ManualClock("2026-09-01T00:00:00.000Z")

    def build(active_clock):
        return Phase14Runner(
            storage,
            clock=active_clock,
            mode=MODE_TEST,
            device="cpu",
            inference_device="cpu",
            topology=LoaderTopology(workers=1),
            games_in_flight=8,
            population=TEST_POPULATION,
        )

    runner = build(clock)
    start = runner.start()
    first = runner.run_iteration()

    clock.advance_hours(6.1)  # crosses the 2h archive and the 6h candidate marks
    second = runner.run_iteration()

    clock.advance_hours(126.0)  # crosses the 132h main -> late transition
    third = runner.run_iteration()

    before = {
        "step": runner.trainer.global_step,
        "iteration": runner.progress.iteration,
        "pool": runner.pool.digest(),
        "archive": runner.archive.digest(),
        "window": runner.controller.window.to_dict(),
        "model": runner.trainer.model_state_digest,
        "beta": runner.trainer.controller.beta,
        "examples": runner.trainer.examples_consumed,
    }

    resumed_clock = ManualClock(clock.now())
    resumed = build(resumed_clock)
    resume_report = resumed.resume()
    after = {
        "step": resumed.trainer.global_step,
        "iteration": resumed.progress.iteration,
        "pool": resumed.pool.digest(),
        "archive": resumed.archive.digest(),
        "window": resumed.controller.window.to_dict(),
        "model": resumed.trainer.model_state_digest,
        "beta": resumed.trainer.controller.beta,
        "examples": resumed.trainer.examples_consumed,
    }

    resumed_clock.advance_hours(36.1)  # past the 168h deadline
    refused = resumed.run_iteration()
    final = resumed.finalize(reason="deadline")

    return {
        "storage": storage,
        "start": start,
        "units": [first, second, third],
        "before": before,
        "after": after,
        "resume_report": resume_report,
        "refused": refused,
        "final": final,
        "runner": resumed,
    }


def test_units_collect_the_scheduled_mixture_and_seal(scripted_run):
    for unit in scripted_run["units"]:
        assert unit["launched"] and unit["sealed"] and unit["trained"]
        counts = unit["collection"]["bucket_counts"]
        assert counts == TEST_POPULATION.bucket_counts(unit["segment"])
        assert unit["collection"]["games_collected"] == TEST_POPULATION.games_per_iteration(
            unit["segment"]
        )
        assert unit["collection"]["observer_probe_failures"] == 0


def test_the_transition_moves_the_learning_rate_and_the_mixture(scripted_run):
    main, _, late = scripted_run["units"]
    assert main["segment"] == "main"
    assert late["segment"] == "late"
    assert main["telemetry"]["training"]["learning_rate"] == contract.MAIN_LEARNING_RATE
    assert late["telemetry"]["training"]["learning_rate"] == contract.LATE_LEARNING_RATE
    assert (
        late["collection"]["bucket_counts"]["historical"]
        >= main["collection"]["bucket_counts"]["historical"]
    )


def test_the_model_updates_and_every_loss_component_is_finite(scripted_run):
    import math

    unit = scripted_run["units"][0]
    assert unit["updates"] > 0
    telemetry = unit["telemetry"]["training"]
    for key in ("policy_loss", "value_loss", "belief_loss", "kl", "grad_norm"):
        value = telemetry[key]
        assert value is not None, key
        assert math.isfinite(float(value)), key
    # The belief auxiliary is contracted, so it must actually contribute.
    assert float(telemetry["belief_loss"]) != 0.0
    assert scripted_run["before"]["model"] != contract.STARTING_MODEL_STATE_DIGEST
    assert scripted_run["before"]["step"] >= 3


def test_the_archive_and_candidates_follow_the_frozen_cadences(scripted_run):
    first, second, third = scripted_run["units"]
    assert first["archived"] is None and first["candidate"] is None
    assert second["archived"] is not None
    assert second["candidate"]["hour"] == 6
    assert third["candidate"]["hour"] == 132
    runner = scripted_run["runner"]
    assert runner.archive.k >= 3
    hours = [mark["hour"] for mark in scripted_run["final"]["manifest"]["candidates"]]
    assert hours[0] == 0 and hours[-1] == 168


def test_the_pool_grows_deterministically_with_the_archive(scripted_run):
    runner = scripted_run["runner"]
    recomputed = ActivePool.for_archive(runner.archive)
    assert recomputed.digest() == runner.pool.digest()
    assert recomputed.members()[:2] == ("P8", "P9")
    assert all(
        identity in recomputed.checkpoints for identity in recomputed.members()
    )


def test_resume_restores_the_exact_logical_state(scripted_run):
    assert scripted_run["before"] == scripted_run["after"]
    report = scripted_run["resume_report"]
    assert report["resumed"] is True
    assert report["run_start_utc"] == scripted_run["start"]["run_start_utc"]
    assert report["run_deadline_utc"] == scripted_run["start"]["run_deadline_utc"]


def test_the_deadline_stops_new_work_and_finalization_preserves_hour_168(scripted_run):
    assert scripted_run["refused"]["launched"] is False
    assert scripted_run["refused"]["reason"] == "deadline"
    final = scripted_run["final"]
    assert final["closed"] is True
    assert final["hour_168_candidate"]["hour"] == 168
    assert final["hour_168_candidate"]["evaluation_status"] == "pending"
    assert final["manifest"]["progress"]["closed"] is True


def test_storage_and_telemetry_paths_carry_real_content(scripted_run):
    storage = scripted_run["storage"]
    assert (storage.log_root / "phase14_telemetry.jsonl").exists()
    assert storage.run_state_path.exists()
    rows = json.loads(storage.run_state_path.read_text())
    assert rows["artifact"] == "phase14_run_manifest_v1"
    unit = scripted_run["units"][-1]
    assert unit["telemetry"]["missing_metrics"] == []
    assert unit["retention"]["disposable_mark"]["safe_to_delete"] is True
    assert list(storage.archive_root.glob("archive_*.pt"))


def test_the_hot_checkpoint_reloads_and_names_its_run(scripted_run):
    from stratego.training.phase14_checkpoint import read

    runner = scripted_run["runner"]
    latest = runner.hot.latest_valid()
    assert latest is not None
    payload = read(latest)
    assert payload["run_window"]["run_start_utc"] == scripted_run["start"]["run_start_utc"]
    assert payload["upstream"]["phase14_contract_digest"] == contract.contract_digest()
    assert payload["upstream"]["parent_sha256"] == contract.STARTING_CHECKPOINT_SHA256
    assert payload["ema_state"]["present"] is False
    assert payload["active_historical_pool"]["k"] == runner.archive.k


def test_production_mode_refuses_both_test_seams(tmp_path):
    from stratego.training.phase14_runner import MODE_PRODUCTION, Phase14Runner, Phase14RunnerError
    from stratego.training.phase14_storage import Phase14Storage

    storage = Phase14Storage.under(tmp_path)
    with pytest.raises(Phase14ClockError):
        Phase14Runner(storage, clock=ManualClock("2026-09-01T00:00:00.000Z"), mode=MODE_PRODUCTION)
    with pytest.raises(Phase14RunnerError):
        Phase14Runner(storage, mode=MODE_PRODUCTION, population=Population.scaled(8))


def test_a_crash_during_training_resumes_from_the_checkpointed_cursor(tmp_path):
    """The unit's games are already sealed; only its epochs need finishing."""
    from stratego.training.phase14_runner import MODE_TEST, Phase14Runner
    from stratego.training.phase14_storage import Phase14Storage
    from stratego.training.phase9_trainer import LoaderTopology

    storage = Phase14Storage.under(tmp_path)
    clock = ManualClock("2026-09-01T00:00:00.000Z")

    def build(active_clock):
        return Phase14Runner(
            storage,
            clock=active_clock,
            mode=MODE_TEST,
            device="cpu",
            inference_device="cpu",
            topology=LoaderTopology(workers=1),
            games_in_flight=8,
            population=TEST_POPULATION,
        )

    runner = build(clock)
    runner.start()
    partial = runner.run_iteration(updates=1)
    assert partial["sealed"] is True
    assert partial["trained"] is False
    assert partial["updates"] == 1
    interrupted_step = runner.trainer.global_step

    resumed = build(ManualClock(clock.now()))
    resumed.resume()
    assert resumed.trainer.cursor is not None
    assert resumed.trainer.cursor.finished is False
    finished = resumed.run_iteration()
    assert finished["resumed_training_only"] is True
    assert finished["resumed_from_cursor"] is True
    assert finished["trained"] is True
    assert finished["collection"]["games_collected"] == 0  # nothing was re-played
    assert resumed.trainer.global_step > interrupted_step
    assert resumed.progress.iteration == 1

    # And the next unit is a fresh one, not a replay of the committed iteration.
    following = resumed.run_iteration(updates=1)
    assert following["iteration"] == 2
    assert following.get("resumed_training_only") is None


def test_a_run_without_a_valid_hot_checkpoint_refuses_to_resume(tmp_path):
    from stratego.training.phase14_runner import (
        MODE_TEST,
        Phase14IntegrityError,
        Phase14Runner,
    )
    from stratego.training.phase14_storage import Phase14Storage

    storage = Phase14Storage.under(tmp_path)
    storage.prepare()
    runner = Phase14Runner(
        storage,
        clock=ManualClock("2026-09-01T00:00:00.000Z"),
        mode=MODE_TEST,
        device="cpu",
        population=TEST_POPULATION,
    )
    with pytest.raises(Phase14IntegrityError):
        runner.resume()


# ---------------------------------------------------------------------------
# The candidate evaluator
# ---------------------------------------------------------------------------


def test_the_frozen_pack_loads_and_verifies_its_own_digest():
    from stratego.evaluation.phase14_candidates import (
        Phase14CandidateError,
        load_pack,
        load_selection_rule,
    )

    pack = load_pack()
    assert pack["pack_content_digest"] == contract.SELECTION_PACK_DIGEST
    assert len(pack["games"]) == 128
    rule = load_selection_rule()
    assert rule["pack_binding"]["games_per_candidate"] == 128
    with pytest.raises(Phase14CandidateError):
        load_pack(REPO_ROOT / contract.SELECTION_RULE_RELATIVE_PATH)


def test_the_evaluator_plays_the_pack_with_direct_policies_only(tmp_path):
    from stratego.evaluation.phase14_candidates import evaluate_candidate, load_pack
    from stratego.training.phase14_checkpoint import export_evaluation_weights

    weights = tmp_path / "anchor.pt"
    export_evaluation_weights(
        contract.repository_root() / contract.STARTING_CHECKPOINT, weights
    )
    pack = dict(load_pack())
    pack["games"] = [
        game
        for stratum in contract.SELECTION_STRATA
        for game in [g for g in pack["games"] if g["opponent"] == stratum][:1]
    ]
    result = evaluate_candidate(weights, anchor_weights=weights, pack=pack, device="cpu")
    assert result["games_played"] == 4
    assert set(result["strata"]) == set(contract.SELECTION_STRATA)
    assert result["search_used"] is False
    assert result["complete"] is False  # a 4-game pass is not a scored candidate
    assert 0.0 <= result["mean_ewr"] <= 1.0


def test_the_ledger_survives_a_failed_evaluation(tmp_path):
    from stratego.evaluation.phase14_candidates import (
        CandidateLedger,
        Phase14CandidateError,
        select_final_candidate,
    )

    ledger = CandidateLedger.at(tmp_path)
    ledger.record_candidate(6, {"snapshot_path": "/tmp/x.pt"})
    ledger.record_failure(6, "device fell over")
    entry = ledger.read()["candidates"]["6"]
    assert entry["status"] == "failed"
    assert entry["rerunnable"] is True
    assert ledger.pending()[0]["hour"] == 6
    with pytest.raises(Phase14CandidateError):
        select_final_candidate(ledger.read()["candidates"].values())


def test_the_selection_rule_ranks_by_mean_then_minimum_then_hour():
    from stratego.evaluation.phase14_candidates import select_final_candidate

    entries = [
        {"hour": 6, "complete": True, "mean_ewr": 0.60, "min_stratum_ewr": 0.50},
        {"hour": 12, "complete": True, "mean_ewr": 0.65, "min_stratum_ewr": 0.40},
        {"hour": 18, "complete": True, "mean_ewr": 0.65, "min_stratum_ewr": 0.55},
        {"hour": 24, "complete": True, "mean_ewr": 0.65, "min_stratum_ewr": 0.55},
        {"hour": 30, "complete": False, "mean_ewr": 0.99, "min_stratum_ewr": 0.99},
    ]
    selected = select_final_candidate(entries)
    assert selected["selected_hour"] == 24  # highest mean, highest min, later hour
    assert selected["candidates_considered"] == 4
