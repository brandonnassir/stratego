"""Phase 16 Agent 3: checkpoints, resume identity, exports and the arm clock."""

import copy
import json

import pytest
import torch

from stratego.training.phase16 import checkpoint as CK
from stratego.training.phase16 import contract as C
from stratego.training.phase16 import runner as R
from stratego.training.phase16 import seat as SEAT
from stratego.training.phase16.trainer import WindowTrainer


@pytest.fixture
def arm_state(starting_model):
    """A live arm's three mutable objects, cheap to build."""
    config = C.ARM_B.replace(arm_id="test_tiny", device="cpu", collection_device="cpu")
    model = copy.deepcopy(starting_model)
    trainer = WindowTrainer(config, model, device="cpu")
    return config, model, trainer


# ---------------------------------------------------------------------------
# The starting model
# ---------------------------------------------------------------------------


def test_the_starting_model_is_the_bound_p24_copy_and_is_trainable(starting_model):
    from stratego.training.phase9_behavior import state_dict_digest

    assert state_dict_digest(starting_model) == C.STARTING_MODEL_STATE_DIGEST
    assert all(parameter.requires_grad for parameter in starting_model.parameters())


def test_another_checkpoint_cannot_be_the_starting_model(tmp_path, repository_root):
    other = repository_root / "checkpoints/phase15/p18_source_readonly.pt"
    if not other.is_file():
        pytest.skip("the P18 copy is not present")
    with pytest.raises(CK.Phase16CheckpointError):
        CK.load_starting_model(other, device="cpu", root=repository_root)
    with pytest.raises(CK.Phase16CheckpointError):
        CK.load_starting_model(tmp_path / "absent.pt", device="cpu", root=repository_root)


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def test_a_payload_carries_every_field_a_resume_needs(arm_state):
    config, model, trainer = arm_state
    payload = CK.build_payload(
        config=config,
        model=model,
        optimizer=trainer.optimizer,
        ema=trainer.ema,
        trainer_state=trainer.trainer_state(),
        collector_state={"iteration": 3},
        clock={"elapsed_seconds": 12.0},
    )
    assert all(key in payload for key in CK.REQUIRED_KEYS)
    assert payload["arm_digest"] == config.digest()
    assert payload["ema_state"]["present"] is True
    assert payload["starting_checkpoint"]["sha256"] == C.STARTING_CHECKPOINT_SHA256


def test_an_arm_without_an_ema_records_the_absence_rather_than_omitting_it(starting_model):
    config = C.ARM_A.replace(arm_id="test_control", device="cpu")
    model = copy.deepcopy(starting_model)
    trainer = WindowTrainer(config, model, device="cpu")
    payload = CK.build_payload(
        config=config,
        model=model,
        optimizer=trainer.optimizer,
        ema=None,
        trainer_state=trainer.trainer_state(),
        collector_state={},
        clock={},
    )
    assert payload["ema_state"]["present"] is False
    assert payload["ema_state"]["statement"]


def test_a_checkpoint_round_trips_through_disk(tmp_path, arm_state):
    config, model, trainer = arm_state
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01)
    trainer.ema.update(model)
    trainer.global_step = 17
    payload = CK.build_payload(
        config=config,
        model=model,
        optimizer=trainer.optimizer,
        ema=trainer.ema,
        trainer_state=trainer.trainer_state(),
        collector_state={"iteration": 4, "draw_counts": [1, 2]},
        clock={"elapsed_seconds": 60.0, "started_utc": "2026-08-26T00:00:00.000Z"},
    )
    written = CK.save(payload, tmp_path / "hot.pt")
    assert written["arm_digest"] == config.digest()
    reread = CK.read(tmp_path / "hot.pt")
    assert reread["model_state_digest"] == payload["model_state_digest"]

    fresh_model = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in fresh_model.parameters():
            parameter.mul_(0.0)
    fresh = WindowTrainer(config, fresh_model, device="cpu")
    report = CK.restore(
        reread, config=config, model=fresh_model, optimizer=fresh.optimizer, ema=fresh.ema
    )
    assert report["model_state_digest"] == payload["model_state_digest"]
    assert fresh.ema.updates == trainer.ema.updates
    fresh.restore_state(reread["trainer_state"])
    assert fresh.global_step == 17


def test_a_resume_refuses_a_different_arm(tmp_path, arm_state):
    config, model, trainer = arm_state
    payload = CK.build_payload(
        config=config,
        model=model,
        optimizer=trainer.optimizer,
        ema=trainer.ema,
        trainer_state=trainer.trainer_state(),
        collector_state={},
        clock={},
    )
    other = config.replace(epochs=2)
    with pytest.raises(CK.Phase16CheckpointError, match="arm"):
        CK.restore(payload, config=other, model=model, optimizer=trainer.optimizer, ema=trainer.ema)


def test_reading_refuses_a_foreign_payload(tmp_path):
    torch.save({"phase14_checkpoint_version": "phase14_checkpoint_v1"}, tmp_path / "foreign.pt")
    with pytest.raises(CK.Phase16CheckpointError):
        CK.read(tmp_path / "foreign.pt")
    torch.save([1, 2, 3], tmp_path / "notamap.pt")
    with pytest.raises(CK.Phase16CheckpointError):
        CK.read(tmp_path / "notamap.pt")


# ---------------------------------------------------------------------------
# The evaluation export
# ---------------------------------------------------------------------------


def test_the_export_round_trips_bitwise_and_loads_as_c1(tmp_path, arm_state):
    config, model, trainer = arm_state
    payload = CK.build_payload(
        config=config,
        model=model,
        optimizer=trainer.optimizer,
        ema=trainer.ema,
        trainer_state=trainer.trainer_state(),
        collector_state={},
        clock={},
    )
    report = CK.export_evaluation_weights(payload, tmp_path / "hour_00.pt")
    assert report["bitwise_state_dict_match"]
    assert report["source"] == "raw"
    assert report["model_state_digest"] == payload["model_state_digest"]
    assert report["parameters"] > 0
    assert CK.file_sha256(tmp_path / "hour_00.pt") == report["export_sha256"]


def test_an_ema_export_is_the_ema_and_not_the_raw_weights(tmp_path, arm_state):
    config, model, trainer = arm_state
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05)
    trainer.ema.update(model)
    payload = CK.build_payload(
        config=config,
        model=model,
        optimizer=trainer.optimizer,
        ema=trainer.ema,
        trainer_state=trainer.trainer_state(),
        collector_state={},
        clock={},
    )
    raw = CK.export_evaluation_weights(payload, tmp_path / "raw.pt", use_ema=False)
    ema = CK.export_evaluation_weights(payload, tmp_path / "ema.pt", use_ema=True)
    assert ema["source"] == "ema"
    assert ema["model_state_digest"] != raw["model_state_digest"]


def test_an_ema_export_is_refused_when_the_arm_has_none(tmp_path, starting_model):
    config = C.ARM_A.replace(arm_id="test_control", device="cpu")
    model = copy.deepcopy(starting_model)
    trainer = WindowTrainer(config, model, device="cpu")
    payload = CK.build_payload(
        config=config,
        model=model,
        optimizer=trainer.optimizer,
        ema=None,
        trainer_state=trainer.trainer_state(),
        collector_state={},
        clock={},
    )
    with pytest.raises(CK.Phase16CheckpointError):
        CK.export_evaluation_weights(payload, tmp_path / "ema.pt", use_ema=True)


def test_an_exported_arm_seats_through_the_accepted_direct_seat(tmp_path, arm_state):
    config, model, trainer = arm_state
    payload = CK.build_payload(
        config=config,
        model=model,
        optimizer=trainer.optimizer,
        ema=trainer.ema,
        trainer_state=trainer.trainer_state(),
        collector_state={},
        clock={},
    )
    report = CK.export_evaluation_weights(payload, tmp_path / "hour_00.pt")
    built = SEAT.build_seat(
        weights_path=str(tmp_path / "hour_00.pt"),
        arm_id="test_tiny_h0",
        device="cpu",
        expected_sha256=report["export_sha256"],
    )
    assert built.arm_id == "test_tiny_h0"
    assert built.kind == "direct"
    assert hasattr(built, "decide") and hasattr(built, "pairing")
    with pytest.raises(SEAT.Phase16SeatError):
        SEAT.build_seat(
            weights_path=str(tmp_path / "hour_00.pt"),
            arm_id="x",
            device="cpu",
            expected_sha256="0" * 64,
        )


def test_the_provider_spec_is_what_agent_1s_runner_normalizes(tmp_path):
    from stratego.evaluation.phase16.runner import normalize_seat_spec

    spec = SEAT.provider_spec(str(tmp_path / "w.pt"), arm_id="b_damped_h6", expected_sha256="ab" * 32)
    normalized = normalize_seat_spec(spec)
    assert normalized[0] == "stratego.training.phase16.seat:build_seat"
    assert json.loads(normalized[1])["arm_id"] == "b_damped_h6"
    assert normalized[2] == "b_damped_h6"


# ---------------------------------------------------------------------------
# The clock
# ---------------------------------------------------------------------------


def test_the_clock_measures_the_arms_own_elapsed_time():
    clock = R.ArmClock(hours=6.0)
    assert not clock.expired
    assert clock.elapsed_hours >= 0.0
    expired = R.ArmClock(hours=0.0)
    assert expired.expired


def test_a_resumed_clock_carries_the_elapsed_time_forward():
    original = R.ArmClock(hours=6.0)
    payload = original.to_dict()
    payload["elapsed_seconds"] = 3600.0
    resumed = R.ArmClock.resume(payload, hours=6.0)
    assert resumed.started_utc == original.started_utc
    assert resumed.elapsed_hours >= 1.0
    assert not resumed.expired
    # a run that already used its budget stays expired across the restart
    payload["elapsed_seconds"] = 6 * 3600.0
    assert R.ArmClock.resume(payload, hours=6.0).expired


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def test_telemetry_round_trips_and_summarizes(tmp_path):
    rows = [
        {
            "iteration": index,
            "elapsed_hours": index * 0.1,
            "iteration_wall_seconds": 300.0 + index,
            "collection": {
                "plies_per_second": 40.0,
                "game_length": {"mean": 200.0 + index},
            },
            "optimization": {
                "mean_policy_entropy": 0.7 - index * 0.01,
                "mean_behavior_kl": 0.02,
                "mean_clip_fraction": 0.2,
                "advantage_statistics": {"retention_fraction": 0.25},
            },
        }
        for index in range(1, 6)
    ]
    path = tmp_path / "arm_windows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert R.read_telemetry(path) == rows
    summary = R.telemetry_summary(rows)
    assert summary["iterations"] == 5
    assert summary["iteration_wall_seconds"]["mean"] == pytest.approx(303.0)
    assert summary["move_entropy"] == {"first": 0.69, "last": pytest.approx(0.65)}
    assert summary["advantage_retention"]["mean"] == pytest.approx(0.25)
    assert summary["elapsed_hours"] == pytest.approx(0.5)
    assert R.read_telemetry(tmp_path / "absent.jsonl") == []
    assert R.telemetry_summary([]) == {}


def test_the_iteration_wall_time_spread_is_the_number_section_2_1_targets():
    """A window collector's whole claim is that this coefficient stays small."""
    steady = [
        {"iteration": n, "iteration_wall_seconds": 300.0, "collection": {}, "optimization": {}}
        for n in range(10)
    ]
    assert R.telemetry_summary(steady)["iteration_wall_seconds"]["coefficient_of_variation"] == 0.0
    ballooning = [
        {"iteration": n, "iteration_wall_seconds": 300.0 * (1 + n), "collection": {}, "optimization": {}}
        for n in range(10)
    ]
    assert (
        R.telemetry_summary(ballooning)["iteration_wall_seconds"]["coefficient_of_variation"] > 0.5
    )


# ---------------------------------------------------------------------------
# The arm's own historical pool
# ---------------------------------------------------------------------------


def test_only_a_mixture_arm_archives_its_own_weights(starting_model, tmp_path, library_source):
    """A `pure_current` arm never draws a historical opponent, so it keeps none."""
    from stratego.training.phase16.collector import WindowCollector
    from stratego.training.phase16.population import HistoricalPool
    from stratego.training.phase16.snapshots import bind_anchor, participants_for

    def build(config):
        runner = R.ArmRunner(
            config,
            root=REPOSITORY_ROOT_FOR_TESTS,
            storage_root=tmp_path / "arms",
            telemetry_root=tmp_path / "telemetry",
            hours=6.0,
            device="cpu",
            collection_device="cpu",
        )
        runner.model = starting_model
        runner.pool = HistoricalPool(R.ANCHOR_IDENTITY)
        runner._historical = bind_anchor(
            starting_model, identity=R.ANCHOR_IDENTITY, device="cpu"
        )
        runner.collector = WindowCollector(
            config,
            participants_for(
                starting_model,
                identity=R.CURRENT_IDENTITY,
                device="cpu",
                historical=runner._historical,
            ),
            setup_source=library_source,
            pool=runner.pool,
        )
        runner.collector.iteration = 7
        return runner

    pure = build(C.ARM_B.replace(arm_id="pure", device="cpu", collection_device="cpu"))
    assert pure._maybe_archive() is None
    assert pure.pool.members() == (R.ANCHOR_IDENTITY,)

    mixture = build(C.ARM_A.replace(arm_id="mix", device="cpu", collection_device="cpu"))
    entry = mixture._maybe_archive()
    assert entry is not None
    assert entry["identity"] == "W0007"
    assert entry["pool_size"] == 2
    assert mixture.pool.members() == (R.ANCHOR_IDENTITY, "W0007")
    assert "W0007" in mixture._historical
    # and the cadence holds: an immediate second call adds nothing
    assert mixture._maybe_archive() is None
    assert len(mixture.pool.members()) == 2


REPOSITORY_ROOT_FOR_TESTS = __import__("pathlib").Path(__file__).resolve().parents[3]


def test_a_hard_veto_stops_the_arm_cleanly_rather_than_crashing(monkeypatch, tmp_path, starting_model, library_source):
    """The limit still fires; only the way the process ends changes."""
    from stratego.training.phase16.collector import WindowCollector, WindowResult
    from stratego.training.phase16.population import HistoricalPool
    from stratego.training.phase16.snapshots import bind_anchor, participants_for
    from stratego.training.phase16.trainer import Phase16TrainerError

    config = C.ARM_B.replace(arm_id="veto", device="cpu", collection_device="cpu",
                             population=2, window_decisions=8, minibatch_size=2)
    runner = R.ArmRunner(config, root=REPOSITORY_ROOT_FOR_TESTS,
                         storage_root=tmp_path / "arms", telemetry_root=tmp_path / "tel",
                         hours=6.0, device="cpu", collection_device="cpu")
    runner.model = starting_model
    runner.pool = HistoricalPool(R.ANCHOR_IDENTITY)
    runner._historical = bind_anchor(starting_model, identity=R.ANCHOR_IDENTITY, device="cpu")
    runner.collector = WindowCollector(
        config,
        participants_for(starting_model, identity=R.CURRENT_IDENTITY, device="cpu",
                         historical=runner._historical),
        setup_source=library_source, pool=runner.pool,
    )
    from stratego.training.phase16.trainer import WindowTrainer

    runner.trainer = WindowTrainer(config, starting_model, device="cpu")

    class _Row:
        advantage = 0.0
    def fake_collect(**kwargs):
        result = WindowResult(iteration=runner.collector.iteration + 1)
        runner.collector.iteration += 1
        result.rows = [_Row()]
        result.plies_advanced = 1
        result.seconds = 0.01
        return result
    monkeypatch.setattr(runner.collector, "collect_window", fake_collect)
    monkeypatch.setattr(runner, "_maybe_evaluate", lambda **kw: None)
    monkeypatch.setattr(runner, "_write_hot_checkpoint", lambda: {})
    monkeypatch.setattr(runner, "_maybe_archive", lambda: None)
    monkeypatch.setattr(runner, "_rotate_behavior", lambda: {})

    def veto(rows, *, iteration, may_start_step=None):
        raise Phase16TrainerError(f"window {iteration} epoch 1: mean behavior KL 0.09 exceeds 0.08")
    monkeypatch.setattr(runner.trainer, "train_window", veto)

    summary = runner.run()          # must not raise
    assert summary["stopped"]["reason"] == "hard_veto"
    assert "exceeds" in summary["stopped"]["error"]
    assert summary["stopped"]["iteration"] == 1
    # and the breach is on the telemetry, not only in the return value
    rows = R.read_telemetry(runner.telemetry_path)
    assert rows and rows[-1]["stopped"]["reason"] == "hard_veto"
