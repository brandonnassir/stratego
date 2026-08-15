"""Phase 8 Agent 5: pilot-selection logic and the frozen train config.

Two layers. The pure layer proves the *decision* — matrix exactness, score
arithmetic, veto, tie-break, config completeness, held-out gates — against
hand-built records, so it holds regardless of what any particular run
measured. The artifact layer re-runs the same pure functions against the
shipped `agent_05_*` evidence and requires the published winner back, which is
what makes "reproducible from the CSV" a test rather than a claim.

The full-scale MPS evidence (six 5,000-update pilots) lives in
`scripts/run_phase8_agent05.py` and its artifacts; nothing here trains.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from stratego.training import warmstart_contract as wc
from stratego.training import warmstart_pilot as wp
from stratego.training.warmstart_dataset import (
    WarmstartBatch,
    WarmstartTargets,
    batch_from_arrays,
)
from stratego.training.warmstart_metrics import run_validation, spread_batch_positions
from stratego.training.warmstart_trainer import (
    WarmstartTrainConfig,
    pilot_candidate_ids,
)

ARTIFACT_DIRECTORY = Path(__file__).resolve().parents[2] / "reports" / "phase_8_data"


def record(
    candidate_id: str,
    *,
    policy: "float | None" = 0.90,
    value: "float | None" = 0.95,
    belief: "float | None" = 0.98,
    examples_per_second: float = 1900.0,
    completed: int = wp.PILOT_UPDATE_BUDGET,
    **counters,
) -> dict:
    """One synthetic selection record with the frozen field names."""
    base = {
        "non_finite_losses": 0,
        "non_finite_gradients": 0,
        "non_finite_parameters": 0,
        "illegal_targets": 0,
        "data_mismatches": 0,
        "checkpoint_errors": 0,
    }
    base.update(counters)
    ratios = [policy, value, belief]
    return {
        "candidate_id": candidate_id,
        "completed_updates": completed,
        "update_budget": wp.PILOT_UPDATE_BUDGET,
        "examples_per_second": examples_per_second,
        "counters": base,
        "final_checkpoint": {
            "global_step": completed,
            "scope": wp.SELECTION_SCOPE,
            "policy_ce_ratio": policy,
            "value_ce_ratio": value,
            "belief_ce_ratio": belief,
            "selection_score": wp.selection_score(ratios),
        },
    }


class TestCandidateMatrixExactness:
    def test_the_live_matrix_is_agent_1s_six_candidates_at_the_cap(self):
        matrix = wp.frozen_candidate_matrix()
        assert len(matrix) == 6 == wp.PILOT_CANDIDATE_LIMIT
        assert [entry["candidate_id"] for entry in matrix] == list(pilot_candidate_ids())
        assert wp.verify_candidate_matrix() == []

    def test_only_learning_rate_and_loss_weights_vary(self):
        matrix = wp.frozen_candidate_matrix()
        assert {entry["learning_rate"] for entry in matrix} == {1e-3, 3e-4, 1e-4}
        assert {
            (entry["lambda_policy"], entry["lambda_value"], entry["lambda_belief"])
            for entry in matrix
        } == {(1.0, 1.0, 1.0), (1.0, 0.5, 0.5)}
        for entry in matrix:
            assert set(entry) == {
                "candidate_id",
                "learning_rate",
                "loss_profile",
                "lambda_policy",
                "lambda_value",
                "lambda_belief",
            }

    def test_a_drifted_recorded_matrix_is_reported(self):
        drifted = wc.pilot_matrix()
        drifted["candidates"] = drifted["candidates"][:5]
        problems = wp.verify_candidate_matrix(drifted)
        assert problems and any("differs from Agent 1" in entry for entry in problems)

    def test_drifted_fixed_controls_are_reported(self):
        drifted = wc.pilot_matrix()
        drifted["fixed_controls"] = dict(drifted["fixed_controls"], batch_size=512)
        problems = wp.verify_candidate_matrix(drifted)
        assert problems and any("fixed controls differ" in entry for entry in problems)

    def test_the_live_matrix_matches_the_accepted_artifact(self):
        path = ARTIFACT_DIRECTORY / "agent_01_warmstart_contract.json"
        if not path.exists():
            pytest.skip("Agent 1 contract artifact not written yet")
        recorded = json.loads(path.read_text())["contract"]["pilot_matrix"]
        assert wp.verify_candidate_matrix(recorded) == []

    def test_the_matrix_digest_is_stable_and_covers_the_controls(self):
        assert wp.candidate_matrix_digest() == wp.candidate_matrix_digest()
        assert len(wp.candidate_matrix_digest()) == 64

    def test_the_frozen_budgets_are_agent_1s(self):
        assert wp.PILOT_UPDATE_BUDGET == 5000
        assert wp.FINAL_UPDATE_BUDGET_MAX == 25000
        assert wp.PILOT_VALIDATION_CADENCE == 500


class TestInitialStateChecksum:
    def test_the_canonical_seed_reproduces_one_checksum(self):
        from stratego.model.production_model import build_candidate_model

        first = build_candidate_model("C0", seed=4242, device="cpu", dtype=torch.float32)
        second = build_candidate_model("C0", seed=4242, device="cpu", dtype=torch.float32)
        other = build_candidate_model("C0", seed=4243, device="cpu", dtype=torch.float32)
        assert wp.model_state_checksum(first.state_dict()) == wp.model_state_checksum(
            second.state_dict()
        )
        assert wp.model_state_checksum(first.state_dict()) != wp.model_state_checksum(
            other.state_dict()
        )

    def test_a_single_changed_weight_changes_the_checksum(self):
        from stratego.model.production_model import build_candidate_model

        model = build_candidate_model("C0", seed=7, device="cpu", dtype=torch.float32)
        before = wp.model_state_checksum(model.state_dict())
        state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        name = sorted(state)[0]
        state[name] = state[name] + 1e-6
        assert wp.model_state_checksum(state) != before

    def test_the_checksum_ignores_device_and_dtype_container(self):
        state = {"a": torch.arange(6, dtype=torch.float64).reshape(2, 3)}
        as_float32 = {"a": state["a"].to(torch.float32)}
        assert wp.model_state_checksum(state) == wp.model_state_checksum(as_float32)


class TestBatchIdentitySequence:
    def test_equal_sequences_fold_to_equal_digests(self):
        left = ["a" * 64, "b" * 64, "c" * 64]
        assert wp.batch_sequence_digest(left) == wp.batch_sequence_digest(list(left))

    def test_order_is_part_of_the_identity(self):
        left = ["a" * 64, "b" * 64]
        assert wp.batch_sequence_digest(left) != wp.batch_sequence_digest(left[::-1])

    def test_length_is_part_of_the_identity(self):
        assert wp.batch_sequence_digest(["a"]) != wp.batch_sequence_digest(["a", "a"])

    def test_validation_batch_positions_are_a_pure_function(self):
        # The fairness property behind "every candidate validates on the same
        # held-out examples": the positions depend on nothing run-specific.
        first = spread_batch_positions(249963, 256, 64)
        second = spread_batch_positions(249963, 256, 64)
        assert first == second
        assert len(first) == 64


class TestSelectionScoreMath:
    def test_the_score_is_the_mean_of_the_three_ratios(self):
        assert wp.selection_score([0.9, 0.99, 1.02]) == pytest.approx(
            (0.9 + 0.99 + 1.02) / 3
        )

    def test_a_missing_component_leaves_the_score_undefined(self):
        assert wp.selection_score([0.9, None, 1.0]) is None

    def test_the_wrong_number_of_ratios_raises(self):
        with pytest.raises(wp.WarmstartPilotError):
            wp.selection_score([0.9, 1.0])

    def test_the_score_reads_the_frozen_field_names(self):
        metrics = {"policy_ce_ratio": 0.8, "value_ce_ratio": 0.9, "belief_ce_ratio": 1.0}
        assert wp.selection_score_from_metrics(metrics) == pytest.approx(0.9)
        assert wp.RATIO_FIELDS == (
            "policy_ce_ratio",
            "value_ce_ratio",
            "belief_ce_ratio",
        )

    def test_lower_is_better(self):
        better = record("better", policy=0.80, value=0.90, belief=0.95)
        worse = record("worse", policy=0.95, value=0.99, belief=1.00)
        assert wp.select_winner([worse, better])["winner"] == "better"


class TestVetoLogic:
    @pytest.mark.parametrize(
        "counter, expected",
        [
            ("non_finite_losses", wp.VETO_NON_FINITE),
            ("non_finite_gradients", wp.VETO_NON_FINITE),
            ("non_finite_parameters", wp.VETO_NON_FINITE),
            ("illegal_targets", wp.VETO_TARGET_MISMATCH),
            ("data_mismatches", wp.VETO_SPLIT_LEAK),
            ("checkpoint_errors", wp.VETO_CHECKPOINT_FAILURE),
        ],
    )
    def test_every_frozen_hard_failure_vetoes(self, counter, expected):
        assert expected in wp.veto_reasons(record("c", **{counter: 1}))

    def test_a_clean_record_is_not_vetoed(self):
        assert wp.veto_reasons(record("c")) == ()

    @pytest.mark.parametrize("component", ["policy", "value", "belief"])
    def test_a_component_ratio_above_1_05_vetoes(self, component):
        vetoed = wp.veto_reasons(record("c", **{component: 1.0501}))
        assert any(entry.startswith(wp.VETO_RATIO_ABOVE_THRESHOLD) for entry in vetoed)

    @pytest.mark.parametrize("component", ["policy", "value", "belief"])
    def test_the_threshold_boundary_is_inclusive(self, component):
        assert wp.veto_reasons(record("c", **{component: 1.05})) == ()

    def test_an_incomplete_budget_vetoes(self):
        assert wp.VETO_INCOMPLETE_BUDGET in wp.veto_reasons(record("c", completed=4999))

    def test_a_missing_final_score_vetoes(self):
        assert wp.VETO_MISSING_FINAL_SCORE in wp.veto_reasons(record("c", value=None))

    def test_all_vetoed_is_blocked_not_a_winner(self):
        decision = wp.select_winner(
            [record("a", policy=1.2), record("b", non_finite_losses=1)]
        )
        assert decision["status"] == "BLOCKED"
        assert decision["winner"] is None
        assert decision["candidates_eligible"] == 0
        assert "broadening" in decision["reason"]

    def test_a_vetoed_candidate_cannot_win_on_score(self):
        # The best score in the matrix, but non-finite training: veto first.
        decision = wp.select_winner(
            [
                record("best_but_broken", policy=0.1, value=0.1, belief=0.1,
                       non_finite_gradients=1),
                record("sound", policy=0.9, value=0.9, belief=0.9),
            ]
        )
        assert decision["winner"] == "sound"
        assert decision["vetoed"][0]["candidate_id"] == "best_but_broken"


class TestTieBreakLogic:
    def test_first_key_is_the_selection_score(self):
        decision = wp.select_winner(
            [record("a", policy=0.90), record("b", policy=0.80)]
        )
        assert decision["winner"] == "b"
        assert decision["tie_break_used"] == "selection_score"

    def test_second_key_is_the_policy_ratio(self):
        # Equal mean, different policy ratio: the lower policy ratio wins.
        left = record("left", policy=0.90, value=0.95, belief=1.00)
        right = record("right", policy=0.85, value=1.00, belief=1.00)
        assert wp.selection_score_from_metrics(
            left["final_checkpoint"]
        ) == pytest.approx(wp.selection_score_from_metrics(right["final_checkpoint"]))
        decision = wp.select_winner([left, right])
        assert decision["winner"] == "right"
        assert decision["tie_break_used"] == "policy_ce_ratio"

    def test_third_key_is_measured_throughput(self):
        slow = record("slow", examples_per_second=1500.0)
        fast = record("fast", examples_per_second=1900.0)
        decision = wp.select_winner([slow, fast])
        assert decision["winner"] == "fast"
        assert decision["tie_break_used"] == "examples_per_second"

    def test_a_complete_tie_still_resolves_deterministically_and_says_so(self):
        first = record("aaa")
        second = record("bbb")
        decision = wp.select_winner([first, second])
        assert decision["winner"] == "aaa"
        assert decision["tie_break_used"] == "candidate_id_determinism_fallback"

    def test_selection_is_independent_of_input_order(self):
        records = [
            record("a", policy=0.91),
            record("b", policy=0.80),
            record("c", policy=0.85),
        ]
        forward = wp.select_winner(records)
        backward = wp.select_winner(records[::-1])
        assert forward["winner"] == backward["winner"] == "b"
        assert [entry["candidate_id"] for entry in forward["ranking"]] == [
            entry["candidate_id"] for entry in backward["ranking"]
        ]

    def test_the_ranking_covers_exactly_the_non_vetoed_candidates(self):
        records = [record("a"), record("b", policy=1.3), record("c", policy=0.7)]
        decision = wp.select_winner(records)
        assert [entry["candidate_id"] for entry in decision["ranking"]] == ["c", "a"]
        assert decision["candidates_vetoed"] == 1

    def test_the_tie_break_order_is_agent_1s(self):
        assert list(wp.TIE_BREAK_ORDER) == [
            "lower selection score",
            "lower validation policy ratio",
            "higher measured training examples/s",
        ]


class TestReproductionFromRows:
    def rows_for(self, records):
        rows = []
        for entry in records:
            final = entry["final_checkpoint"]
            rows.append(
                {
                    "candidate_id": entry["candidate_id"],
                    "validation_scope": wp.SELECTION_SCOPE,
                    "global_step": entry["completed_updates"],
                    "update_budget": entry["update_budget"],
                    "examples_per_second": entry["examples_per_second"],
                    "policy_ce_ratio": final["policy_ce_ratio"],
                    "value_ce_ratio": final["value_ce_ratio"],
                    "belief_ce_ratio": final["belief_ce_ratio"],
                    "selection_score": final["selection_score"],
                    **entry["counters"],
                }
            )
            # Cadence rows must be ignored by the reproduction.
            rows.append(dict(rows[-1], validation_scope=wp.CADENCE_SCOPE,
                             global_step=500, selection_score=0.01))
        return rows

    def test_rows_reproduce_the_same_winner(self):
        records = [record("a", policy=0.91), record("b", policy=0.80)]
        rebuilt = wp.records_from_rows(self.rows_for(records))
        assert wp.select_winner(rebuilt)["winner"] == wp.select_winner(records)["winner"]

    def test_cadence_rows_are_not_selection_inputs(self):
        # The cadence rows carry a far better (0.01) score; if they leaked into
        # the selection the winner would be decided by a 500-update sample.
        records = [record("a", policy=0.91), record("b", policy=0.80)]
        rebuilt = wp.records_from_rows(self.rows_for(records))
        assert len(rebuilt) == 2
        assert all(
            entry["final_checkpoint"]["global_step"] == wp.PILOT_UPDATE_BUDGET
            for entry in rebuilt
        )

    def test_duplicate_selection_rows_are_refused(self):
        records = [record("a")]
        rows = self.rows_for(records)
        rows.append(dict(rows[0]))
        with pytest.raises(wp.WarmstartPilotError, match="more than one"):
            wp.records_from_rows(rows)

    def test_veto_columns_survive_the_round_trip(self):
        records = [record("a", non_finite_losses=3), record("b")]
        rebuilt = wp.records_from_rows(self.rows_for(records))
        decision = wp.select_winner(rebuilt)
        assert decision["winner"] == "b"
        assert decision["vetoed"][0]["reasons"] == [wp.VETO_NON_FINITE]


class TestFrozenTrainConfig:
    def build(self, **overrides):
        candidate_id = overrides.pop("winner_candidate_id", "ws_pilot_lr3e-4_balanced")
        config = WarmstartTrainConfig.from_pilot_candidate(
            candidate_id, device="mps", validation_batches=64
        )
        arguments = {
            "winner_candidate_id": candidate_id,
            "train_config_identity": config.identity(),
            "train_config_digest": config.digest(),
            "model_config_digest": wc.EXPECTED_C1_CONFIG_DIGEST,
            "expected_fresh_init_checksum": "f" * 64,
            "corpus_identity": {
                "corpus_version": "synthetic_warmstart_corpus_v1",
                "content_digest": "a" * 64,
                "metadata_digest": "b" * 64,
                "commit_index_digest": "c" * 64,
            },
            "max_final_updates": 25000,
            "checkpoint_cadence_updates": 500,
            "best_checkpoint_metric": "validation selection_score",
            "early_stop_rule": {"rule": "none"},
            "loader_topology": {"workers": 12, "prefetch": 2, "record_cache_size": 512},
            "seeds": {"train_order_seed": 2026081303},
            "validation_batches": 64,
        }
        arguments.update(overrides)
        return wp.build_frozen_train_config(**arguments)

    def test_every_required_field_is_present_and_versioned(self):
        payload = self.build()
        assert payload["train_config_version"] == wp.WARMSTART_TRAIN_CONFIG_VERSION
        assert wp.verify_frozen_train_config(payload) == []
        for name in wp.REQUIRED_TRAIN_CONFIG_FIELDS:
            assert name in payload["config"], name

    def test_the_hyperparameters_come_from_the_frozen_candidate(self):
        payload = self.build(winner_candidate_id="ws_pilot_lr1e-4_policy_led")
        config = payload["config"]
        assert config["learning_rate"] == 1e-4
        assert (config["lambda_policy"], config["lambda_value"], config["lambda_belief"]) == (
            1.0,
            0.5,
            0.5,
        )
        assert config["model_candidate"] == "C1"
        assert config["batch_size"] == 256
        assert config["optimizer"] == "AdamW"
        assert config["precision"] == "float32"

    def test_an_off_matrix_winner_is_refused(self):
        # Defence in depth: `from_pilot_candidate` already refuses off-matrix
        # ids, so the freeze is handed a *valid* trainer identity under a bogus
        # winner name — and must still refuse on its own.
        valid = WarmstartTrainConfig.from_pilot_candidate(
            "ws_pilot_lr3e-4_balanced", device="mps", validation_batches=64
        )
        with pytest.raises(wp.WarmstartPilotError, match="frozen candidates"):
            wp.build_frozen_train_config(
                winner_candidate_id="ws_pilot_lr5e-4_balanced",
                train_config_identity=valid.identity(),
                train_config_digest=valid.digest(),
                model_config_digest=wc.EXPECTED_C1_CONFIG_DIGEST,
                expected_fresh_init_checksum="f" * 64,
                corpus_identity={},
                max_final_updates=25000,
                checkpoint_cadence_updates=500,
                best_checkpoint_metric="validation selection_score",
                early_stop_rule={"rule": "none"},
                loader_topology={"workers": 12, "prefetch": 2, "record_cache_size": 512},
                seeds={"train_order_seed": 2026081303},
                validation_batches=64,
            )

    def test_a_final_budget_above_25k_is_refused(self):
        with pytest.raises(wp.WarmstartPilotError, match="outside the frozen limit"):
            self.build(max_final_updates=25001)

    def test_the_digest_covers_the_config_body(self):
        first = self.build()
        second = self.build(max_final_updates=20000)
        assert first["train_config_digest"] != second["train_config_digest"]
        assert wp.verify_frozen_train_config(second) == []

    def test_a_tampered_config_fails_verification(self):
        payload = self.build()
        payload["config"]["learning_rate"] = 5e-4
        problems = wp.verify_frozen_train_config(payload)
        assert any("learning_rate" in entry for entry in problems)
        assert any("digest does not match" in entry for entry in problems)

    def test_a_missing_field_fails_verification(self):
        payload = self.build()
        payload["config"].pop("gradient_clip_norm")
        assert any(
            "missing required fields" in entry
            for entry in wp.verify_frozen_train_config(payload)
        )

    def test_the_payload_names_the_reconstruction_call_and_forbids_continuation(self):
        payload = self.build()
        assert "from_pilot_candidate" in payload["trainer_construction"]
        assert any("never continue a pilot checkpoint" in rule
                   for rule in payload["agent_6_rules"])
        assert "checkpoint_path" not in payload


class TestHeldOutAccessIsMeasured:
    def batch(self, splits):
        size = len(splits)
        targets = WarmstartTargets(
            legal_mask=torch.zeros((size, 4), dtype=torch.bool),
            policy_action_model=torch.zeros(size, dtype=torch.int64),
            policy_weight=torch.zeros(size, dtype=torch.float32),
            value_target=torch.zeros(size, dtype=torch.int64),
            belief_target=torch.zeros((size, 2), dtype=torch.int64),
            belief_mask=torch.zeros((size, 2), dtype=torch.bool),
            policy_action_abs=torch.zeros(size, dtype=torch.int32),
            acting_player=torch.zeros(size, dtype=torch.int8),
        )
        return WarmstartBatch(
            observations=torch.zeros((size, 1)),
            targets=targets,
            keys=tuple((f"g{index}", index) for index in range(size)),
            source_policy_ids=("basic",) * size,
            corpus_splits=tuple(splits),
        )

    def test_the_boundary_counts_examples_by_split(self):
        with wp.record_model_input_access() as log:
            self.batch(["train"] * 3).model_input()
            self.batch(["validation"] * 2).model_input()
        assert log.examples_by_split == {"train": 3, "validation": 2}
        assert log.batches_by_split == {"train": 1, "validation": 1}
        assert log.test_examples == 0

    def test_a_test_example_would_be_counted_not_hidden(self):
        # The positive control: the log reports what happens, so a zero in the
        # artifact means zero happened rather than "we did not look".
        with wp.record_model_input_access() as log:
            self.batch(["test", "test"]).model_input()
        assert log.test_examples == 2
        assert log.to_dict()["test_examples_evaluated_by_model"] == 2

    def test_the_instrumentation_is_removed_afterwards(self):
        original = WarmstartBatch.model_input
        with wp.record_model_input_access():
            assert WarmstartBatch.model_input is not original
        assert WarmstartBatch.model_input is original

    def test_the_boundary_still_returns_the_observations(self):
        batch = self.batch(["train"])
        with wp.record_model_input_access():
            assert batch.model_input() is batch.observations

    def test_phase4_entry_points_are_counted_and_restored(self):
        from stratego.evaluation import match_runner, neural_worker

        original = match_runner.play_match
        with wp.record_phase4_access() as log:
            assert match_runner.play_match is not original
        assert match_runner.play_match is original
        assert log.neural_evaluation_games == 0
        assert log.to_dict()["phase4_neural_evaluation_games"] == 0
        assert neural_worker.checkpoint_load_count() >= 0


class TestSealingGatesRefuseAgent5:
    @pytest.mark.parametrize(
        "purpose",
        ["model_inference", "model_metric", "checkpoint_selection",
         "hyperparameter_selection", "early_stopping", "final_evaluation"],
    )
    def test_the_test_corpus_is_sealed_against_agent_5(self, purpose):
        with pytest.raises(wc.HeldOutAccessError):
            wc.check_test_corpus_access(purpose, phase8_agent=5)

    @pytest.mark.parametrize(
        "purpose",
        ["neural_playing_strength", "pilot_selection", "config_selection",
         "checkpoint_selection", "final_random_evaluation"],
    )
    def test_phase4_strength_is_sealed_against_agent_5(self, purpose):
        with pytest.raises(wc.HeldOutAccessError):
            wc.check_phase4_bank_access(purpose, phase8_agent=5)

    def test_structural_reads_stay_available_to_agent_5(self):
        assert wc.check_test_corpus_access("structural_audit", phase8_agent=5)
        assert wc.check_phase4_bank_access("non_neural_regression", phase8_agent=5)

    def test_a_validation_pass_never_reaches_the_test_split(self, warmstart_mini_corpus):
        from stratego.training.warmstart_dataset import WarmstartDataset

        root, _game_ids = warmstart_mini_corpus
        dataset = WarmstartDataset(root, record_cache_size=8, require_complete_split=False)
        with pytest.raises(wc.HeldOutAccessError):
            run_validation(
                object(),
                dataset,
                split="test",
                value_prior=(1 / 3, 1 / 3, 1 / 3),
                phase8_agent=5,
            )

    def test_the_trainer_config_cannot_name_the_test_split(self):
        from stratego.training.warmstart_trainer import (
            WarmstartTrainerError,
            unit_test_config,
        )

        with pytest.raises(WarmstartTrainerError, match="sealed test split"):
            unit_test_config(validation_split="test")


class TestAgent5Artifacts:
    """PASS-gated once the Agent 5 runner has written its artifacts."""

    @pytest.fixture()
    def selection(self):
        path = ARTIFACT_DIRECTORY / "agent_05_pilot_selection.json"
        if not path.exists():
            pytest.skip("Agent 5 selection artifact not written yet")
        return json.loads(path.read_text())

    @pytest.fixture()
    def frozen(self):
        path = ARTIFACT_DIRECTORY / "agent_05_frozen_train_config.json"
        if not path.exists():
            pytest.skip("Agent 5 frozen train config not written yet")
        return json.loads(path.read_text())

    @pytest.fixture()
    def rows(self):
        path = ARTIFACT_DIRECTORY / "agent_05_pilot_runs.csv"
        if not path.exists():
            pytest.skip("Agent 5 pilot runs CSV not written yet")
        with path.open() as handle:
            return list(csv.DictReader(handle))

    def test_the_artifact_declares_pass_and_the_live_versions(self, selection):
        assert selection["status"] == "PASS"
        assert selection["pilot_version"] == wp.WARMSTART_PILOT_VERSION
        assert selection["prerequisite_digests"]["agent_01_contract"] == wc.contract_digest()
        assert (
            selection["prerequisite_digests"]["candidate_matrix"]
            == wp.candidate_matrix_digest()
        )
        gates = selection["completion_gates"]
        assert gates and all(gates.values())

    def test_exactly_the_frozen_matrix_ran(self, selection):
        ran = [run["candidate_id"] for run in selection["pilot_runs"]]
        assert sorted(ran) == sorted(pilot_candidate_ids())
        assert len(ran) <= wp.PILOT_CANDIDATE_LIMIT
        assert selection["fairness"]["unregistered_configs_run"] == []
        assert selection["fairness"]["registered_candidates_not_run"] == []

    def test_every_pilot_started_from_the_same_initialization(self, selection):
        fairness = selection["fairness"]
        assert fairness["all_init_checksums_identical"]
        assert len(set(fairness["init_checksums"].values())) == 1
        assert fairness["expected_fresh_init_checksum"]

    def test_every_pilot_saw_the_same_ordered_batch_identities(self, selection):
        fairness = selection["fairness"]
        assert fairness["all_batch_sequences_identical"]
        assert fairness["all_budgets_equal"]
        assert set(fairness["completed_updates"].values()) == {wp.PILOT_UPDATE_BUDGET}

    def test_validation_ran_at_the_same_update_numbers_on_the_same_examples(
        self, selection
    ):
        fairness = selection["fairness"]
        assert fairness["all_validation_update_numbers_identical"]
        assert fairness["validation_batch_positions_identical"]
        numbers = list(fairness["validation_update_numbers"].values())[0]
        assert numbers == list(
            range(
                wp.PILOT_VALIDATION_CADENCE,
                wp.PILOT_UPDATE_BUDGET + 1,
                wp.PILOT_VALIDATION_CADENCE,
            )
        )

    def test_the_published_scores_are_the_frozen_arithmetic(self, selection):
        assert selection["selection_score_recomputation"]["all_match"]
        assert selection["selection_score_recomputation"]["mismatches"] == []

    def test_the_winner_is_reproducible_from_the_published_csv(self, selection, rows):
        rebuilt = wp.records_from_rows(rows)
        decision = wp.select_winner(rebuilt)
        assert decision["status"] == "PASS"
        assert decision["winner"] == selection["selection"]["winner"]
        assert [entry["candidate_id"] for entry in decision["ranking"]] == [
            entry["candidate_id"] for entry in selection["selection"]["ranking"]
        ]

    def test_the_csv_selection_rows_are_the_full_validation_split(self, rows):
        selection_rows = [
            row for row in rows if row["validation_scope"] == wp.SELECTION_SCOPE
        ]
        assert len(selection_rows) == len(pilot_candidate_ids())
        for row in selection_rows:
            assert int(row["global_step"]) == wp.PILOT_UPDATE_BUDGET

    def test_the_frozen_config_is_complete_versioned_and_the_winner(
        self, frozen, selection
    ):
        assert wp.verify_frozen_train_config(frozen) == []
        assert frozen["problems"] == []
        assert frozen["winning_candidate_id"] == selection["selection"]["winner"]
        assert frozen["config"]["max_final_updates"] <= wp.FINAL_UPDATE_BUDGET_MAX
        assert frozen["candidate_matrix_digest"] == wp.candidate_matrix_digest()

    def test_the_frozen_config_reconstructs_through_the_trainer_api(self, frozen):
        config = WarmstartTrainConfig.from_pilot_candidate(
            frozen["winning_candidate_id"],
            device=frozen["config"]["device"],
            validation_batches=frozen["config"]["validation_batches"],
        )
        assert config.identity() == frozen["trainer_config_identity"]
        assert config.digest() == frozen["trainer_config_digest"]

    def test_no_test_example_reached_a_model_and_no_phase4_game_was_played(
        self, selection
    ):
        access = selection["held_out_access_log"]
        assert access["test_examples_evaluated_by_model_agent_5"] == 0
        assert access["phase4_neural_evaluation_games_agent_5"] == 0
        assert access["phase4_neural_checkpoint_loads_agent_5"] == 0
        assert access["all_gates_refuse_agent_5"]
        assert access["examples_by_split"].get("test", 0) == 0
        assert access["examples_by_split"]["validation"] > 0

    def test_no_pilot_checkpoint_is_handed_to_agent_6(self, selection, frozen):
        handoff = selection["handoff_to_agent_6"]
        assert handoff["winning_candidate_id"] == frozen["winning_candidate_id"]
        assert handoff["expected_fresh_init_checksum"]
        assert handoff["final_training_budget"] <= wp.FINAL_UPDATE_BUDGET_MAX
        serialized = json.dumps(handoff)
        assert ".pt" not in serialized and "checkpoint_path" not in serialized

    def test_the_sanity_extension_was_not_invented(self, selection):
        extension = selection["sanity_extension"]
        assert extension["run"] is False
        assert (
            extension["agent_1_development_budget"]["final_run_optimizer_steps_max"]
            == wp.FINAL_UPDATE_BUDGET_MAX
        )
