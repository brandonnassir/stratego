"""The run supervisor's predicate arithmetic and D10's two families.

These tests are about counting and classification, not about training. Their
subject is the line D10 section 7 drew: `I1`-`I8` stop a run and `P1`-`P7` are
warnings that never can.
"""

from __future__ import annotations

import pytest

from stratego.training.phase17.supervisor import (
    MODE_INTEGRATION,
    MODE_PRODUCTION,
    SEVERITY_STOP,
    SEVERITY_WARNING,
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


def test_three_consecutive_readings_still_only_warn():
    """D10 section 7: setup entropy decline is telemetry, not a stop.

    The consecutive count still fires -- that is what turns one noisy reading
    into a warning worth reading -- but firing a statistical predicate can
    never set `stopped`.
    """
    supervisor = make()
    for _ in range(2):
        assert not supervisor.observe_setup_entropy(FLOOR - 0.1)["fired"]
    verdict = supervisor.observe_setup_entropy(FLOOR - 0.1)
    assert verdict["fired"] and verdict["severity"] == SEVERITY_WARNING
    assert verdict["stops_the_run"] is False
    assert not supervisor.should_stop
    assert supervisor.stop_record() is None
    assert any(entry["code"] == "P4" for entry in supervisor.warnings)


def test_every_statistical_predicate_is_a_warning_in_every_mode():
    for mode in (MODE_PRODUCTION, MODE_INTEGRATION):
        supervisor = make(mode)
        for code in supervisor.WARNING_CODES:
            assert supervisor.predicates[code].severity == SEVERITY_WARNING, code
        for code in ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8"):
            assert supervisor.predicates[code].severity == SEVERITY_STOP, code


def test_no_statistical_reading_can_stop_a_run():
    """Every P-family entry point, driven past its consecutive count."""
    supervisor = make()
    for _ in range(6):
        supervisor.observe_setup_entropy(FLOOR - 0.2)
        supervisor.observe_flag_support(1.0)
        supervisor.observe_setup_kl(10.0)
        supervisor.observe_move_kl(10.0)
        supervisor.observe_move_entropy(0.0, first_hour_median=1.0)
        supervisor.observe_ewr(0.10, hour0=0.90)
        supervisor.observe_setup_update_activity(
            updated=False, interval_complete=True, episodes_available=True
        )
    assert not supervisor.should_stop
    assert supervisor.stop_record() is None
    assert {entry["code"] for entry in supervisor.warnings} == set(
        supervisor.WARNING_CODES
    )


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


def test_the_descriptive_thresholds_are_unchanged():
    """D10 demoted the consequences, not the numbers.

    The thresholds survive so a warning can still say "this is below the level
    Agent 3 measured as collapse". Rewriting one to manufacture a quiet run
    would be the opposite of what the demotion is for.
    """
    supervisor = make()
    assert supervisor.setup_entropy_floor == pytest.approx(0.9257366873314787)
    assert supervisor.flag_support_floor == 4.0
    assert supervisor.predicates["P4"].consecutive_required == 3


def test_flag_support_at_the_floor_does_not_trip():
    assert not make().observe_flag_support(4.0)["tripped"]


def test_setup_kl_hard_limit_needs_three_consecutive_updates():
    supervisor = make()
    for _ in range(2):
        assert not supervisor.observe_setup_kl(0.09)["fired"]
    verdict = supervisor.observe_setup_kl(0.09)
    assert verdict["fired"] and verdict["severity"] == SEVERITY_WARNING
    assert not supervisor.should_stop


def test_setup_kl_at_the_limit_does_not_trip():
    assert not make().observe_setup_kl(0.08)["tripped"]


def test_move_kl_hard_limit():
    supervisor = make()
    for _ in range(3):
        verdict = supervisor.observe_move_kl(0.081)
    assert verdict["fired"] and not supervisor.should_stop


def test_a_fixed_transition_count_violation_stops_immediately():
    """One of D10 section 7's named stops: the iteration already trained."""
    supervisor = make()
    assert not supervisor.check_transition_count(harvested=65536, budget=65536)["tripped"]
    verdict = supervisor.check_transition_count(harvested=65535, budget=65536)
    assert verdict["fired"] and verdict["severity"] == SEVERITY_STOP
    assert supervisor.should_stop
    assert supervisor.stop_record()["code"] == "I8"


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
        updated=True, interval_complete=True, episodes_available=True
    )["tripped"]
    assert not supervisor.observe_setup_update_activity(
        updated=False, interval_complete=True, episodes_available=False
    )["tripped"]
    verdict = supervisor.observe_setup_update_activity(
        updated=False, interval_complete=True, episodes_available=True
    )
    assert verdict["fired"] and not supervisor.should_stop


def test_move_entropy_needs_a_first_hour_median_before_it_can_trip():
    supervisor = make()
    assert not supervisor.observe_move_entropy(0.001)["tripped"]
    supervisor.observe_move_entropy(0.001, first_hour_median=1.0)
    for _ in range(4):
        verdict = supervisor.observe_move_entropy(0.001)
    assert verdict["fired"] and verdict["code"] == "P6"
    assert not supervisor.should_stop


def test_ewr_collapse_is_measured_against_hour_zero():
    supervisor = make()
    supervisor.observe_ewr(0.70, hour0=0.70)
    for _ in range(3):
        verdict = supervisor.observe_ewr(0.54)
    assert verdict["fired"] and verdict["code"] == "P1"
    assert verdict["evidence"]["drop"] == pytest.approx(0.16)
    assert not supervisor.should_stop


def test_the_retired_queue_backlog_predicate_is_gone():
    """P8 watched a backlog. D10 drains the buffer every iteration."""
    supervisor = make()
    assert "P8" not in supervisor.predicates
    assert not hasattr(supervisor, "observe_queue")


def test_the_supervisor_changes_no_hyperparameter():
    """The document names what it may not touch; the class has no setter for any."""
    supervisor = make()
    for name in (
        "learning_rate",
        "kl_coefficient",
        "entropy_coefficient",
        "population",
        "epochs",
        "setup_batch",
    ):
        assert not hasattr(supervisor, f"set_{name}")
    assert "learning rate" in supervisor.document()["may_not_change"]
    assert (
        "the fixed setup behavior-KL coefficient"
        in supervisor.document()["may_not_change"]
    )


def test_consecutive_state_survives_a_round_trip():
    supervisor = make()
    supervisor.observe_setup_entropy(FLOOR - 0.1)
    supervisor.observe_setup_entropy(FLOOR - 0.1)
    supervisor.observe_ewr(0.7, hour0=0.7)
    resumed = make()
    resumed.load_state_document(supervisor.state_document())
    assert resumed.predicates["P4"].consecutive == 2
    assert resumed.hour0_ewr == pytest.approx(0.7)
    # The third reading after a resume still fires as a warning, exactly as it
    # would have without the resume.
    assert resumed.observe_setup_entropy(FLOOR - 0.1)["fired"]
    assert not resumed.should_stop


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


def test_the_retired_budget_policy_module_is_gone():
    """D10 removed the quota, the warm-up and the backlog alarm together."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("stratego.training.phase17.queue")
