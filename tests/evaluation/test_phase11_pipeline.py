"""Phase 11 Agent 5: the integrated pipeline and the sealed-bank refusal.

These tests pin the properties Agents 6 and 7 depend on: the pipeline's
stage list and entry point are stable, the sealed test bank cannot be
scored without an explicit authorization, the frozen sample schedule is a
deterministic function of the recorded decisions, the gate quantities are
pulled from the metric block by name, and the freeze identity is a pure
function of the logical document.

Nothing here plays a game or runs a forward pass; the heavy integrated run
lives in the harness and its artifacts are checked by
`test_phase11_agent05_artifacts.py`.
"""

from __future__ import annotations

import pytest

from stratego.evaluation import phase11_pipeline as pipeline
from stratego.training import phase11_contract as contract


# ---------------------------------------------------------------------------
# Versions, stages and bindings
# ---------------------------------------------------------------------------


def test_pipeline_version_and_entry_point_are_frozen():
    assert pipeline.PIPELINE_VERSION == "phase11_validation_freeze_v1"
    assert (
        pipeline.FINAL_TEST_ENTRY_POINT
        == "stratego.evaluation.phase11_pipeline.run_phase11_pipeline"
    )
    module_name, _, function_name = pipeline.FINAL_TEST_ENTRY_POINT.rpartition(".")
    assert module_name == pipeline.__name__
    assert callable(getattr(pipeline, function_name))


def test_pipeline_stages_are_the_eight_scored_stages_in_order():
    assert pipeline.PIPELINE_STAGES == (
        "generate",
        "targets",
        "score",
        "metrics",
        "slices",
        "sampler_checks",
        "bound_evidence",
        "gate_quantities",
    )


def test_bank_bindings_match_the_frozen_contract():
    validation = pipeline.bank_binding("validation")
    test = pipeline.bank_binding("test")
    assert validation["bank_version"] == contract.VALIDATION_BANK_VERSION
    assert validation["games_expected"] == contract.VALIDATION_BANK_GAMES
    assert test["bank_version"] == contract.TEST_BANK_VERSION
    assert test["games_expected"] == contract.TEST_BANK_GAMES


def test_bank_binding_refuses_an_unknown_bank():
    with pytest.raises(pipeline.Phase11PipelineError):
        pipeline.bank_binding("holdout")


def test_bank_binding_returns_a_copy():
    binding = pipeline.bank_binding("validation")
    binding["games_expected"] = 1
    assert pipeline.bank_binding("validation")["games_expected"] == (
        contract.VALIDATION_BANK_GAMES
    )


# ---------------------------------------------------------------------------
# The seal
# ---------------------------------------------------------------------------


def test_only_the_test_bank_is_sealed():
    assert pipeline.SEALED_BANKS == (contract.TEST_BANK_VERSION,)


def test_sealed_bank_is_refused_without_authorization():
    with pytest.raises(pipeline.Phase11SealError):
        pipeline.assert_seal("test", sealed_bank_authorized=False)


def test_validation_bank_is_never_sealed():
    pipeline.assert_seal("validation", sealed_bank_authorized=False)
    pipeline.assert_seal("validation", sealed_bank_authorized=True)


def test_sealed_bank_opens_only_with_the_explicit_authorization():
    pipeline.assert_seal("test", sealed_bank_authorized=True)


def test_run_pipeline_refuses_the_sealed_bank_before_touching_anything(tmp_path):
    """The refusal precedes every read, so a sealed call cannot half-run."""
    with pytest.raises(pipeline.Phase11SealError):
        pipeline.run_phase11_pipeline(
            "test",
            tmp_path,
            bound_evidence={},
            preservation={},
            store_root=tmp_path / "store",
        )
    assert not (tmp_path / "store").exists()


def test_sealed_bank_authorization_defaults_to_false():
    import inspect

    signature = inspect.signature(pipeline.run_phase11_pipeline)
    assert signature.parameters["sealed_bank_authorized"].default is False


# ---------------------------------------------------------------------------
# The frozen sample schedule
# ---------------------------------------------------------------------------


def _rows(game_id: str, decisions):
    return [
        {
            "game_id": game_id,
            "case_id": f"case|{game_id}",
            "game_index": 0,
            "observer_color": "red",
            "opponent_stratum": "basic_rule",
            "opponent_setup_source": "neutral",
            "decision_index": index,
            "public_state_identity": f"{game_id}-{index:04d}" + "0" * 16,
            "unresolved_pieces": 5,
            "progress_bucket": "early",
        }
        for index in decisions
    ]


def test_schedule_takes_evenly_spaced_decisions_per_game():
    rows = _rows("g0", range(0, 40, 2))
    schedule = pipeline.frozen_sample_schedule(rows)
    assert len(schedule) == pipeline.SAMPLE_DECISIONS_PER_GAME
    taken = [row["decision_index"] for row in schedule]
    assert taken == [0, 10, 20, 30]


def test_schedule_takes_what_a_short_game_has():
    schedule = pipeline.frozen_sample_schedule(_rows("g0", [0, 4]))
    assert [row["decision_index"] for row in schedule] == [0, 4]


def test_schedule_covers_every_game_and_is_ordinal_dense():
    rows = _rows("g0", range(0, 20)) + _rows("g1", range(0, 20))
    schedule = pipeline.frozen_sample_schedule(rows)
    assert {row["game_id"] for row in schedule} == {"g0", "g1"}
    assert [row["schedule_ordinal"] for row in schedule] == list(range(len(schedule)))
    assert [row["request_ordinal"] for row in schedule] == list(range(len(schedule)))
    assert len(set(row["request_id"] for row in schedule)) == len(schedule)


def test_schedule_is_independent_of_input_row_order():
    rows = _rows("g0", range(0, 20)) + _rows("g1", range(0, 20))
    forward = pipeline.frozen_sample_schedule(rows)
    reverse = pipeline.frozen_sample_schedule(list(reversed(rows)))
    assert [row["request_id"] for row in forward] == [
        row["request_id"] for row in reverse
    ]


def test_schedule_does_not_mutate_its_input_rows():
    rows = _rows("g0", range(0, 20))
    pipeline.frozen_sample_schedule(rows)
    assert all("schedule_ordinal" not in row for row in rows)


def test_sample_schedule_nominal_size_is_the_frozen_world_floor_or_more():
    """The nominal size clears the floor; the realized size may not.

    A game with fewer eligible decisions than the quota contributes what it
    has, so this is a ceiling on the integrated pass, not a guarantee — the
    frozen floor itself belongs to Agent 3's large audit.
    """
    states = contract.VALIDATION_BANK_GAMES * pipeline.SAMPLE_DECISIONS_PER_GAME
    assert states * pipeline.SAMPLE_WORLD_ORDINALS >= pipeline.SAMPLE_WORLD_FLOOR
    assert pipeline.SAMPLE_WORLD_FLOOR == contract.SAMPLER_AUDIT_MIN_WORLDS


def test_sample_world_ordinals_match_the_measured_request_shape():
    from stratego.evaluation.phase11_repro import REQUEST_WORLD_COUNT

    assert pipeline.SAMPLE_WORLD_ORDINALS == REQUEST_WORLD_COUNT


# ---------------------------------------------------------------------------
# Gate quantities
# ---------------------------------------------------------------------------


def _interval(point, lower=None, upper=None):
    return {
        "point": point,
        "lower": point if lower is None else lower,
        "upper": point if upper is None else upper,
    }


def _metric_block(r_ce=0.95, ce_delta_upper=-0.01, top1=0.04, brier_upper=-0.02):
    return {
        "metrics": {
            "r_ce": _interval(r_ce, r_ce - 0.01, r_ce + 0.01),
            "ce_delta": _interval(-0.05, -0.06, ce_delta_upper),
            "top1_delta": _interval(top1, top1 - 0.005, top1 + 0.005),
            "brier_delta": _interval(-0.03, -0.04, brier_upper),
        },
        "ece_learned": {"ece": 0.04},
        "ece_baseline": {"ece": 0.002},
    }


def _slices(r_ce=1.0, ece=0.05):
    return {
        "opponent_stratum": {
            name: {"r_ce": _interval(r_ce), "ece_learned": {"ece": ece}}
            for name in contract.OPPONENT_STRATA
        }
    }


def test_gate_quantities_are_pulled_by_name():
    quantities = pipeline.gate_quantities(_metric_block(), _slices())
    assert quantities["r_ce"] == pytest.approx(0.95)
    assert quantities["ce_delta_upper"] == pytest.approx(-0.01)
    assert quantities["delta_top1"] == pytest.approx(0.04)
    assert quantities["brier_delta_upper"] == pytest.approx(-0.02)
    assert quantities["ece_overall"] == pytest.approx(0.04)
    assert sorted(quantities["stratum_r_ce"]) == sorted(contract.OPPONENT_STRATA)
    assert sorted(quantities["stratum_ece"]) == sorted(contract.OPPONENT_STRATA)


def test_gate_quantities_refuse_a_missing_stratum():
    slices = _slices()
    slices["opponent_stratum"].pop("scout_rush")
    with pytest.raises(pipeline.Phase11PipelineError):
        pipeline.gate_quantities(_metric_block(), slices)


def _evidence(sampler_zero=True, safety_zero=True, legs_exact=True, p95=48.5):
    counters = {name: 0 for name in contract.SAMPLER_ZERO_TOLERANCE_COUNTERS}
    if not sampler_zero:
        counters["inventory_errors"] = 1
    safety = {name: 0 for name in contract.INFORMATION_SAFETY_ZERO_COUNTERS}
    if not safety_zero:
        safety[contract.INFORMATION_SAFETY_ZERO_COUNTERS[0]] = 1
    legs = {name: legs_exact for name in contract.REPRODUCIBILITY_TOPOLOGY_LEGS}
    return counters, safety, legs, p95


def _preservation():
    return {
        key: value
        for key, value in contract.GATE_H.items()
        if key not in ("gate", "name")
    }


def test_evaluate_all_gates_passes_on_clean_evidence():
    counters, safety, legs, p95 = _evidence()
    gates = pipeline.evaluate_all_gates(
        pipeline.gate_quantities(_metric_block(), _slices()),
        sampler_counters=counters,
        safety_counters=safety,
        leg_exact=legs,
        p95_forward_64_ms=p95,
        preservation=_preservation(),
    )
    assert sorted(gates) == list(contract.HARD_GATE_IDS)
    assert all(block["passed"] for block in gates.values())


@pytest.mark.parametrize(
    "kwargs, failing",
    [
        ({"sampler_zero": False}, "E"),
        ({"safety_zero": False}, "F"),
        ({"legs_exact": False}, "G"),
        ({"p95": 501.0}, "G"),
    ],
)
def test_evaluate_all_gates_fails_the_right_gate(kwargs, failing):
    counters, safety, legs, p95 = _evidence(**kwargs)
    gates = pipeline.evaluate_all_gates(
        pipeline.gate_quantities(_metric_block(), _slices()),
        sampler_counters=counters,
        safety_counters=safety,
        leg_exact=legs,
        p95_forward_64_ms=p95,
        preservation=_preservation(),
    )
    assert gates[failing]["passed"] is False
    assert all(
        gates[gate]["passed"] for gate in contract.HARD_GATE_IDS if gate != failing
    )


def test_gate_a_fails_on_the_known_validation_r_ce():
    """The 0.9750 reading must fail Gate A, and nothing may soften it."""
    counters, safety, legs, p95 = _evidence()
    gates = pipeline.evaluate_all_gates(
        pipeline.gate_quantities(_metric_block(r_ce=0.9750), _slices()),
        sampler_counters=counters,
        safety_counters=safety,
        leg_exact=legs,
        p95_forward_64_ms=p95,
        preservation=_preservation(),
    )
    assert gates["A"]["passed"] is False
    assert gates["A"]["checks"]["r_ce_le_0_97"] is False
    # The other Gate A check still holds: significantly better, not better
    # by the required margin.
    assert gates["A"]["checks"]["ce_delta_upper_lt_0"] is True


def test_gate_h_fails_when_a_preservation_identity_moves():
    counters, safety, legs, p95 = _evidence()
    preservation = _preservation()
    preservation["phase9_parameters"] = 863_960
    gates = pipeline.evaluate_all_gates(
        pipeline.gate_quantities(_metric_block(), _slices()),
        sampler_counters=counters,
        safety_counters=safety,
        leg_exact=legs,
        p95_forward_64_ms=p95,
        preservation=preservation,
    )
    assert gates["H"]["passed"] is False


# ---------------------------------------------------------------------------
# The freeze identity
# ---------------------------------------------------------------------------


def test_freeze_identity_is_order_independent_and_content_sensitive():
    left = pipeline.freeze_identity({"a": 1, "b": {"c": 2, "d": 3}})
    right = pipeline.freeze_identity({"b": {"d": 3, "c": 2}, "a": 1})
    assert left == right
    assert left != pipeline.freeze_identity({"a": 1, "b": {"c": 2, "d": 4}})


def test_frozen_implementation_modules_all_exist():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for name in pipeline.FROZEN_IMPLEMENTATION_MODULES:
        assert (root / name).exists(), name


def test_frozen_implementation_covers_every_phase11_module():
    """A new Phase 11 implementation module must join the freeze list."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    live = {
        f"stratego/evaluation/{path.name}"
        for path in (root / "stratego" / "evaluation").glob("phase11_*.py")
    } | {
        f"stratego/training/{path.name}"
        for path in (root / "stratego" / "training").glob("phase11_*.py")
    }
    assert live == set(pipeline.FROZEN_IMPLEMENTATION_MODULES)


def test_module_sha256_is_stable_and_covers_the_freeze_list():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    first = pipeline.module_sha256(root)
    assert sorted(first) == sorted(pipeline.FROZEN_IMPLEMENTATION_MODULES)
    assert first == pipeline.module_sha256(root)
    assert all(len(digest) == 64 for digest in first.values())


# ---------------------------------------------------------------------------
# The content-only store identity
# ---------------------------------------------------------------------------


def _manifest(forward_seconds=(0.1, 0.2)):
    return {
        "bank_digest": "bank",
        "bank_version": "phase11_validation_bank_v1",
        "belief_forwards": 10,
        "belief_rows": 100,
        "complete_bank": True,
        "games": 2,
        "games_expected": 2,
        "model_identity": "model",
        "observer_decisions": 20,
        "prediction_events": 200,
        "record_version": "phase11_prediction_record_v1",
        "request_digest_rollup": "rollup",
        "run_version": "phase11_belief_validation_v1",
        "store_version": "phase11_prediction_store_v1",
        "games_index": [
            {
                "bank_version": "phase11_validation_bank_v1",
                "case_id": f"case{index}",
                "case_index": index,
                "decisions": 10,
                "empty_decisions": 0,
                "events": 100,
                "forward_seconds": seconds,
                "game_id": f"game{index}",
                "game_index": index,
                "match_id": f"match{index}",
                "match_seed": 7 + index,
                "observer_color": "red",
                "observer_decisions": 10,
                "observer_result": "win",
                "opponent_setup_source": "neutral",
                "opponent_stratum": "basic_rule",
                "plies": 40,
                "public_shard_digest": f"public{index}",
                "record_version": "phase11_prediction_record_v1",
                "replay_digest": f"replay{index}",
                "store_version": "phase11_prediction_store_v1",
                "terminal_reason": "flag_capture",
                "truth_shard_digest": f"truth{index}",
            }
            for index, seconds in enumerate(forward_seconds)
        ],
    }


def test_store_content_digest_ignores_wall_clock_durations():
    """The whole reason the content digest exists."""
    assert pipeline.store_content_digest(
        _manifest((0.1, 0.2))
    ) == pipeline.store_content_digest(_manifest((9.9, 0.0001)))


def test_store_content_digest_is_sensitive_to_recorded_content():
    baseline = pipeline.store_content_digest(_manifest())
    for field, value in (
        ("public_shard_digest", "tampered"),
        ("truth_shard_digest", "tampered"),
        ("replay_digest", "tampered"),
        ("events", 101),
        ("match_seed", 999),
        ("terminal_reason", "battleless_move_limit_draw"),
    ):
        manifest = _manifest()
        manifest["games_index"][0][field] = value
        assert pipeline.store_content_digest(manifest) != baseline, field


def test_store_content_digest_is_sensitive_to_run_level_content():
    baseline = pipeline.store_content_digest(_manifest())
    for field, value in (
        ("request_digest_rollup", "tampered"),
        ("prediction_events", 201),
        ("model_identity", "other"),
        ("bank_digest", "other"),
    ):
        manifest = _manifest()
        manifest[field] = value
        assert pipeline.store_content_digest(manifest) != baseline, field


def test_store_content_digest_is_independent_of_game_order():
    manifest = _manifest()
    reversed_manifest = _manifest()
    reversed_manifest["games_index"].reverse()
    assert pipeline.store_content_digest(manifest) == pipeline.store_content_digest(
        reversed_manifest
    )


def test_store_content_fields_exclude_every_timing():
    assert "forward_seconds" not in pipeline.STORE_CONTENT_FIELDS
    assert not [
        name
        for name in pipeline.STORE_CONTENT_FIELDS + pipeline.STORE_CONTENT_MANIFEST_FIELDS
        if "second" in name or "time" in name or "clock" in name
    ]


# ---------------------------------------------------------------------------
# Schedule accounting
# ---------------------------------------------------------------------------


def test_schedule_accounting_separates_nominal_attainable_and_realized():
    rows = _rows("g0", range(0, 20)) + _rows("g1", [0, 3]) + _rows("g2", [7])
    schedule = pipeline.frozen_sample_schedule(rows)
    accounting = pipeline.schedule_accounting(rows, schedule, games=5)
    assert accounting["games"] == 5
    assert accounting["games_with_eligible_decisions"] == 3
    assert accounting["games_without_eligible_decisions"] == 2
    assert accounting["schedule_slots_nominal"] == 5 * 4
    assert accounting["schedule_slots_attainable"] == 4 + 2 + 1
    assert accounting["schedule_slots_realized"] == 7
    assert accounting["realized_equals_attainable"] is True
    assert accounting["every_eligible_game_contributes"] is True
    assert accounting["games_below_quota"] == 2
    assert accounting["games_contributing"] == 3


def test_schedule_accounting_notices_a_dropped_eligible_game():
    rows = _rows("g0", range(0, 8)) + _rows("g1", range(0, 8))
    schedule = [row for row in pipeline.frozen_sample_schedule(rows) if row["game_id"] == "g0"]
    accounting = pipeline.schedule_accounting(rows, schedule, games=2)
    assert accounting["every_eligible_game_contributes"] is False
    assert accounting["realized_equals_attainable"] is False


def test_schedule_accounting_on_a_bank_where_every_game_is_full():
    rows = _rows("g0", range(0, 40)) + _rows("g1", range(0, 40))
    schedule = pipeline.frozen_sample_schedule(rows)
    accounting = pipeline.schedule_accounting(rows, schedule, games=2)
    assert accounting["schedule_slots_nominal"] == accounting["schedule_slots_attainable"]
    assert accounting["schedule_slots_realized"] == accounting["schedule_slots_nominal"]
    assert accounting["games_without_eligible_decisions"] == 0
    assert accounting["games_below_quota"] == 0
