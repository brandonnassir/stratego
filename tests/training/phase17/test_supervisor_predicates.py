"""Agent 4: the collapse supervisor's predicate arithmetic and the D9-B rule.

These tests are about counting and classification, not about training. Agent 4
instruction section 9: "Unit tests cover predicate arithmetic; no broad
failure-injection campaign is required."
"""

from __future__ import annotations

import pytest

from stratego.training.phase17.supervisor import (
    MODE_INTEGRATION,
    MODE_PRODUCTION,
    SEVERITY_DIAGNOSTIC,
    SEVERITY_STOP,
    CollapseSupervisor,
    Phase17SupervisorError,
)

BASELINE = 1.542894478885798
FLOOR = BASELINE * 0.60


def make(mode=MODE_PRODUCTION) -> CollapseSupervisor:
    return CollapseSupervisor(
        run_id="RUN-TEST-A", mode=mode, setup_entropy_baseline=BASELINE
    )


def test_a_baseline_is_never_defaulted():
    with pytest.raises(Phase17SupervisorError, match="baseline must be positive"):
        CollapseSupervisor(run_id="RUN-TEST-A", setup_entropy_baseline=0.0)


def test_one_reading_is_a_warning_not_a_stop():
    supervisor = make()
    verdict = supervisor.observe_setup_entropy(FLOOR - 0.1)
    assert verdict["tripped"] and not verdict["fired"]
    assert not supervisor.should_stop


def test_three_consecutive_readings_stop_in_production():
    supervisor = make()
    for _ in range(2):
        assert not supervisor.observe_setup_entropy(FLOOR - 0.1)["fired"]
    verdict = supervisor.observe_setup_entropy(FLOOR - 0.1)
    assert verdict["fired"] and verdict["severity"] == SEVERITY_STOP
    assert supervisor.should_stop
    assert supervisor.stop_record()["code"] == "P4"


def test_a_good_reading_resets_the_consecutive_run():
    """Three trips spread across a run must not accumulate into a stop."""
    supervisor = make()
    supervisor.observe_setup_entropy(FLOOR - 0.1)
    supervisor.observe_setup_entropy(FLOOR + 0.1)
    supervisor.observe_setup_entropy(FLOOR - 0.1)
    supervisor.observe_setup_entropy(FLOOR + 0.1)
    verdict = supervisor.observe_setup_entropy(FLOOR - 0.1)
    assert verdict["tripped"] and not verdict["fired"]
    assert verdict["consecutive"] == 1
    assert not supervisor.should_stop
    assert supervisor.predicates["P4"].trips == 3


def test_d9b_makes_p4_a_diagnostic_in_integration_mode_only():
    """Decision D9-B section 6: the relative reading does not veto integration."""
    supervisor = make(MODE_INTEGRATION)
    for _ in range(3):
        verdict = supervisor.observe_setup_entropy(FLOOR - 0.2)
    assert verdict["fired"]
    assert verdict["severity"] == SEVERITY_DIAGNOSTIC
    assert not supervisor.should_stop, "a relative-only reading may not stop Agent 4"
    assert supervisor.stop_record() is None


def test_d9b_does_not_move_the_threshold():
    production = make(MODE_PRODUCTION)
    integration = make(MODE_INTEGRATION)
    assert production.setup_entropy_floor == integration.setup_entropy_floor
    assert production.setup_entropy_floor == pytest.approx(0.9257366873314787)
    assert (
        production.predicates["P4"].consecutive_required
        == integration.predicates["P4"].consecutive_required
        == 3
    )


def test_absolute_floors_stay_hard_in_integration_mode():
    """P5 is an absolute floor; D9-B leaves every absolute floor a stop."""
    supervisor = make(MODE_INTEGRATION)
    verdict = supervisor.observe_flag_support(3.9)
    assert verdict["fired"] and verdict["severity"] == SEVERITY_STOP
    assert supervisor.should_stop
    assert supervisor.stop_record()["code"] == "P5"


def test_flag_support_at_the_floor_does_not_trip():
    assert not make(MODE_INTEGRATION).observe_flag_support(4.0)["tripped"]


def test_setup_kl_hard_limit_needs_three_consecutive_updates():
    supervisor = make()
    for _ in range(2):
        assert not supervisor.observe_setup_kl(0.09)["fired"]
    assert supervisor.observe_setup_kl(0.09)["fired"]
    assert supervisor.stop_record()["code"] == "P3"


def test_setup_kl_at_the_limit_does_not_trip():
    assert not make().observe_setup_kl(0.08)["tripped"]


def test_move_kl_hard_limit():
    supervisor = make()
    for _ in range(3):
        verdict = supervisor.observe_move_kl(0.081)
    assert verdict["fired"] and supervisor.stop_record()["code"] == "P2"


def test_nonfinite_stops_immediately():
    supervisor = make()
    verdict = supervisor.check_finite({"move_loss": float("nan"), "lr": 1e-4})
    assert verdict["fired"] and supervisor.should_stop
    assert supervisor.stop_record()["code"] == "I3"
    assert "move_loss" in verdict["evidence"]["nonfinite"]


def test_a_none_scalar_is_not_a_nonfinite_reading():
    assert not make().check_finite({"move_loss": None})["tripped"]


def test_a_foreign_participant_is_an_immediate_stop():
    supervisor = make()
    verdicts = supervisor.check_participant_ledger(
        {
            "unknown_model_states": {},
            "rule_or_stress_decisions": 0,
            "historical_participants": 1,
            "search_participants": 0,
        }
    )
    assert verdicts[1]["fired"]
    assert supervisor.stop_record()["code"] == "I5"


def test_a_stale_digest_is_an_immediate_stop():
    supervisor = make()
    verdicts = supervisor.check_participant_ledger(
        {
            "unknown_model_states": {"deadbeef": 12},
            "rule_or_stress_decisions": 0,
            "historical_participants": 0,
            "search_participants": 0,
        }
    )
    assert verdicts[0]["fired"] and supervisor.stop_record()["code"] == "I2"


def test_a_clean_ledger_trips_nothing():
    supervisor = make()
    verdicts = supervisor.check_participant_ledger(
        {
            "unknown_model_states": {},
            "rule_or_stress_decisions": 0,
            "historical_participants": 0,
            "search_participants": 0,
        }
    )
    assert not any(v["tripped"] for v in verdicts)
    assert not supervisor.should_stop


def test_setup_generation_failure_is_i4():
    supervisor = make()
    verdict = supervisor.check_setup_generation(
        legality_failures=0, orientation_failures=0, fallback_attempts=1
    )
    assert verdict["fired"] and supervisor.stop_record()["code"] == "I4"


def test_setup_silence_only_trips_when_work_was_available():
    supervisor = make()
    assert not supervisor.observe_setup_update_activity(
        updated=False, interval_complete=True, warmed_up=False, episodes_available=True
    )["tripped"]
    assert not supervisor.observe_setup_update_activity(
        updated=False, interval_complete=True, warmed_up=True, episodes_available=False
    )["tripped"]
    assert supervisor.observe_setup_update_activity(
        updated=False, interval_complete=True, warmed_up=True, episodes_available=True
    )["fired"]


def test_move_entropy_needs_a_first_hour_median_before_it_can_trip():
    supervisor = make()
    assert not supervisor.observe_move_entropy(0.001)["tripped"]
    supervisor.observe_move_entropy(0.001, first_hour_median=1.0)
    for _ in range(4):
        verdict = supervisor.observe_move_entropy(0.001)
    assert verdict["fired"] and supervisor.stop_record()["code"] == "P6"


def test_ewr_collapse_is_measured_against_hour_zero():
    supervisor = make()
    supervisor.observe_ewr(0.70, hour0=0.70)
    for _ in range(3):
        verdict = supervisor.observe_ewr(0.54)
    assert verdict["fired"] and supervisor.stop_record()["code"] == "P1"
    assert verdict["evidence"]["drop"] == pytest.approx(0.16)


def test_queue_alarms_map_to_p8():
    supervisor = make()
    for _ in range(3):
        verdict = supervisor.observe_queue(
            {"backlog": {"over": True, "depth": 5000}, "age": {"over": False}}
        )
    assert verdict["fired"] and supervisor.stop_record()["code"] == "P8"


def test_the_supervisor_changes_no_hyperparameter():
    """The document names what it may not touch; the class has no setter for any."""
    supervisor = make()
    for name in (
        "learning_rate",
        "kl_target",
        "entropy_coefficient",
        "population",
        "epochs",
        "setup_batch",
    ):
        assert not hasattr(supervisor, f"set_{name}")
    assert "learning rate" in supervisor.document()["may_not_change"]


def test_consecutive_state_survives_a_round_trip():
    supervisor = make()
    supervisor.observe_setup_entropy(FLOOR - 0.1)
    supervisor.observe_setup_entropy(FLOOR - 0.1)
    supervisor.observe_ewr(0.7, hour0=0.7)
    resumed = make()
    resumed.load_state_document(supervisor.state_document())
    assert resumed.predicates["P4"].consecutive == 2
    assert resumed.hour0_ewr == pytest.approx(0.7)
    # The third reading after a resume still fires, exactly as it would have.
    assert resumed.observe_setup_entropy(FLOOR - 0.1)["fired"]


def test_a_foreign_supervisor_state_is_refused():
    supervisor = make()
    with pytest.raises(Phase17SupervisorError, match="supervisor state is"):
        supervisor.load_state_document({"supervisor_version": "something_else"})


def test_an_unknown_predicate_is_refused():
    with pytest.raises(Phase17SupervisorError, match="unknown stop predicate"):
        make().observe("P99", True)


# -- fail-closed reading of the documents the guards depend on --------------


def test_a_ledger_missing_a_required_field_is_refused_not_passed():
    """A `.get(name, 0)` here would turn a renamed field into a permanent pass."""
    supervisor = make()
    with pytest.raises(Phase17SupervisorError, match="missing"):
        supervisor.check_participant_ledger({"unknown_model_states": {}})


def test_a_malformed_queue_alarm_is_refused_not_read_as_not_over():
    supervisor = make()
    with pytest.raises(Phase17SupervisorError, match="cannot answer"):
        supervisor.observe_queue({"backlog": {"depth": 5000}, "age": {"over": False}})


def test_queue_telemetry_without_a_depth_is_refused():
    from stratego.training.phase17.queue import Phase17BudgetError, SetupBudgetPolicy

    policy = SetupBudgetPolicy.freeze(games_per_iteration=100.0)
    with pytest.raises(Phase17BudgetError, match="no 'depth'"):
        policy.alarms({"oldest_age": 3})
    # An empty queue legitimately has no oldest age; that is absence, not a
    # missing field, and must not raise.
    assert policy.alarms({"depth": 0, "oldest_age": None})["age"]["over"] is False
