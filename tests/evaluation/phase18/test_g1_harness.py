"""Tests for the Phase 18 Gate G1 control and evaluation harness.

These cover the parts the wrapper adds - sealing refusals, the per-arm data
overlay, read-only checkpoint staging, the pairing proof, and the assembly of
the 42 original Phase 8 gates. They deliberately do not re-test Phase 8
training or evaluation semantics: the wrapper calls the accepted code for all
of that, and re-testing it here would only test a copy.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import phase18_g1_evaluate as g1  # noqa: E402

ACCEPTED_ACCEPTANCE = (
    REPOSITORY_ROOT / "reports" / "phase_8_data" / "agent_07_final_acceptance.json"
)


def _manifest(**overrides) -> dict:
    payload = {
        "selected_global_step": 24000,
        "selection_protocol": {
            "split": "validation",
            "test_split_used": False,
            "phase4_strength_used": False,
        },
    }
    payload.update(overrides)
    return payload


def _run(test_examples: int = 0) -> dict:
    return {
        "held_out_discipline": {
            "test_examples_evaluated_by_model": test_examples,
            "phase4_neural_evaluation_games": 0,
        }
    }


def _control(tmp_path: Path, *, manifest=None, run=None) -> Path:
    artifacts = tmp_path / "dry_run_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "agent_06_checkpoint_manifest.json").write_text(
        json.dumps(manifest if manifest is not None else _manifest())
    )
    (artifacts / "agent_06_warmstart_run.json").write_text(
        json.dumps(run if run is not None else _run())
    )
    return tmp_path


class TestSealing:
    """The sealed split must not open before the selection is closed."""

    def test_a_finalized_selection_is_accepted(self, tmp_path):
        report = g1.assert_selection_finalized(_control(tmp_path))
        assert report["selection_finalized"] is True
        assert report["selected_global_step"] == 24000

    def test_a_missing_manifest_refuses(self, tmp_path):
        (tmp_path / "dry_run_artifacts").mkdir()
        with pytest.raises(g1.G1EvaluationError, match="selection is not finalized"):
            g1.assert_selection_finalized(tmp_path)

    def test_no_selected_checkpoint_refuses(self, tmp_path):
        control = _control(tmp_path, manifest=_manifest(selected_global_step=None))
        with pytest.raises(g1.G1EvaluationError, match="froze no selected checkpoint"):
            g1.assert_selection_finalized(control)

    @pytest.mark.parametrize(
        "protocol",
        [
            {"split": "test", "test_split_used": False, "phase4_strength_used": False},
            {"split": "validation", "test_split_used": True, "phase4_strength_used": False},
            {"split": "validation", "test_split_used": False, "phase4_strength_used": True},
        ],
    )
    def test_a_selection_that_saw_held_out_data_refuses(self, tmp_path, protocol):
        control = _control(tmp_path, manifest=_manifest(selection_protocol=protocol))
        with pytest.raises(g1.G1EvaluationError, match="not validation-only"):
            g1.assert_selection_finalized(control)

    def test_a_control_that_already_touched_test_refuses(self, tmp_path):
        control = _control(tmp_path, run=_run(test_examples=1))
        with pytest.raises(g1.G1EvaluationError, match="sealing is broken"):
            g1.assert_selection_finalized(control)


class TestOverlay:
    def test_the_reference_overlay_is_the_accepted_evidence(self, tmp_path):
        overlay = g1.build_overlay(tmp_path / "reference", None)
        accepted = {path.name for path in g1.ACCEPTED_DATA.iterdir() if path.is_file()}
        assert {path.name for path in overlay.iterdir()} == accepted
        for name in accepted:
            assert (overlay / name).read_bytes() == (g1.ACCEPTED_DATA / name).read_bytes()

    def test_only_the_two_agent_6_artifacts_are_swapped(self, tmp_path):
        replacement = tmp_path / "arm"
        replacement.mkdir()
        for name in g1.ARM_SPECIFIC:
            (replacement / name).write_text(json.dumps({"marker": name}))
        overlay = g1.build_overlay(
            tmp_path / "candidate", {name: replacement / name for name in g1.ARM_SPECIFIC}
        )
        for name in g1.ARM_SPECIFIC:
            assert json.loads((overlay / name).read_text()) == {"marker": name}
        for path in g1.ACCEPTED_DATA.iterdir():
            if path.is_file() and path.name not in g1.ARM_SPECIFIC:
                assert (overlay / path.name).read_bytes() == path.read_bytes()

    def test_a_non_agent_6_swap_is_refused(self, tmp_path):
        with pytest.raises(g1.G1EvaluationError, match="not an arm-specific artifact"):
            g1.build_overlay(
                tmp_path / "bad", {"agent_05_frozen_train_config.json": tmp_path}
            )


class TestCheckpointStaging:
    def test_the_copy_is_byte_identical_and_read_only(self, tmp_path):
        source = tmp_path / "source.pt"
        source.write_bytes(b"warmstart bytes")
        record = g1.stage_checkpoint(source, tmp_path / "staged" / "copy.pt")
        staged = Path(record["staged"])
        assert staged.read_bytes() == source.read_bytes()
        assert record["sha256"] == g1.sha256(source)
        assert staged.stat().st_mode & 0o222 == 0, "a staged checkpoint must not be writable"

    def test_restaging_replaces_a_read_only_copy(self, tmp_path):
        source = tmp_path / "source.pt"
        source.write_bytes(b"first")
        destination = tmp_path / "staged" / "copy.pt"
        g1.stage_checkpoint(source, destination)
        source.write_bytes(b"second")
        record = g1.stage_checkpoint(source, destination)
        assert destination.read_bytes() == b"second"
        assert record["sha256"] == g1.sha256(source)


def _arm(games, *, schedule="digest-a", offset=0.0, pairs=(0, 1, 2)):
    per_game = {
        game: {
            "policy_weighted_ce": 1.0 + offset,
            "policy_weighted_baseline_ce": 2.0,
            "policy_weighted_top1": 0.5,
            "policy_weight_sum": 1.0,
            "policy_weighted_expected_top1": 0.1,
            "policy_examples": 3.0,
            "value_ce": 1.0,
            "value_baseline_ce": 2.0,
            "value_brier": 0.3,
            "value_baseline_brier": 0.4,
            "value_examples": 5.0,
            "belief_ce": 1.0,
            "belief_baseline_ce": 1.5,
            "belief_top1": 2.0,
            "belief_baseline_top1": 1.0,
            "belief_pieces": 8.0,
        }
        for game in games
    }
    harness = {"schedule_digest": schedule}
    return {
        "per_game": per_game,
        "random_gate": {"harness": dict(harness)},
        "vs_init": {"harness": dict(harness)},
        "random_units": {pair: 0.5 for pair in pairs},
        "vs_init_units": {pair: 0.5 for pair in pairs},
    }


class TestPairingProof:
    def test_matched_arms_have_no_problems(self):
        proof = g1.prove_pairing(_arm(["g1", "g2"]), _arm(["g1", "g2"]))
        assert proof["problems"] == []
        assert proof["identical_game_sets"] is True
        assert proof["sealed_test_games"] == 2

    def test_a_different_game_set_is_caught(self):
        proof = g1.prove_pairing(_arm(["g1", "g2"]), _arm(["g1"]))
        assert any("game sets differ" in problem for problem in proof["problems"])

    def test_a_model_independent_field_that_moved_is_caught(self):
        """Baselines and denominators are properties of the data. If one differs
        between arms, the two arms did not score the same examples."""
        candidate = _arm(["g1"])
        reference = _arm(["g1"])
        reference["per_game"]["g1"]["belief_pieces"] = 9.0
        proof = g1.prove_pairing(candidate, reference)
        assert proof["model_independent_mismatches"] == {"belief_pieces": 1}
        assert any("model-independent" in problem for problem in proof["problems"])

    def test_a_model_dependent_difference_is_not_a_problem(self):
        proof = g1.prove_pairing(_arm(["g1"], offset=0.25), _arm(["g1"]))
        assert proof["problems"] == []

    def test_a_different_schedule_is_caught(self):
        proof = g1.prove_pairing(
            _arm(["g1"], schedule="digest-a"), _arm(["g1"], schedule="digest-b")
        )
        assert sum("same schedule" in problem for problem in proof["problems"]) == 2

    def test_different_setup_pairs_are_caught(self):
        proof = g1.prove_pairing(_arm(["g1"], pairs=(0, 1)), _arm(["g1"], pairs=(0, 2)))
        assert sum("different setup pairs" in problem for problem in proof["problems"]) == 2


class TestTheFortyTwoGates:
    """The wrapper must produce the accepted gate set, name for name."""

    def test_the_gate_names_round_trip_from_the_accepted_artifact(self):
        accepted = json.loads(ACCEPTED_ACCEPTANCE.read_text())["completion_gates"]
        assert len(accepted) == 42

        groups = {"heldout_": {}, "random_": {}, "vs_init_": {}, "discipline_": {}}
        literals = {}
        for name, value in accepted.items():
            for prefix in groups:
                if name.startswith(prefix):
                    groups[prefix][name[len(prefix):]] = value
                    break
            else:
                literals[name] = value

        arm = {
            "heads": {"gates": groups["heldout_"]},
            "random_gate": {"gates": groups["random_"]},
            "vs_init": {"gates": groups["vs_init_"]},
            "audit": {"gates": groups["discipline_"]},
            "verify": {
                "prior_agents": {
                    "agents_1_to_6_all_pass": literals["prerequisites_agents_1_to_6_pass"]
                },
                "corpus": {
                    "resolved_root_matches_accepted_location": literals[
                        "corpus_resolved_through_resolver"
                    ],
                    "problems": [],
                },
                "upstream": {"frozen_upstream_problems": []},
                "checkpoint_identity": {"problems": []},
            },
            "export": {"exports": {"warmstart": {"bitwise_state_dict_match": True}}},
        }
        assert g1.completion_gates(arm) == accepted

    def test_a_verification_problem_fails_its_gate(self):
        accepted = json.loads(ACCEPTED_ACCEPTANCE.read_text())["completion_gates"]
        groups = {"heldout_": {}, "random_": {}, "vs_init_": {}, "discipline_": {}}
        for name, value in accepted.items():
            for prefix in groups:
                if name.startswith(prefix):
                    groups[prefix][name[len(prefix):]] = value
                    break
        arm = {
            "heads": {"gates": groups["heldout_"]},
            "random_gate": {"gates": groups["random_"]},
            "vs_init": {"gates": groups["vs_init_"]},
            "audit": {"gates": groups["discipline_"]},
            "verify": {
                "prior_agents": {"agents_1_to_6_all_pass": True},
                "corpus": {
                    "resolved_root_matches_accepted_location": True,
                    "problems": ["a digest moved"],
                },
                "upstream": {"frozen_upstream_problems": []},
                "checkpoint_identity": {"problems": []},
            },
            "export": {"exports": {"warmstart": {"bitwise_state_dict_match": False}}},
        }
        gates = g1.completion_gates(arm)
        assert gates["corpus_digests_match_accepted"] is False
        assert gates["evaluation_export_bitwise_faithful"] is False


class TestPairedUnitLoading:
    def test_missing_results_are_refused(self, tmp_path):
        with pytest.raises(g1.G1EvaluationError, match="no persisted results"):
            g1.load_paired_units(tmp_path, "random")


class TestControlDriverLocationOverride:
    """The one declared deviation must be exactly one constant, and reversible."""

    def test_the_override_rebinds_only_the_relative_constant(self):
        import phase18_g1_control as control

        module = control.a6
        before = module.REQUIRED_CORPUS_ROOT_RELATIVE
        try:
            record = control.apply_location_override()
            assert record["before"] == before
            assert record["after"] == module.REQUIRED_CORPUS_ROOT
            assert module.REQUIRED_CORPUS_ROOT_RELATIVE == module.REQUIRED_CORPUS_ROOT
            # An absolute right-hand operand wins, so the repository-relative
            # comparison becomes the resolver comparison.
            assert (
                module.REPOSITORY_ROOT / module.REQUIRED_CORPUS_ROOT_RELATIVE
                == Path(module.REQUIRED_CORPUS_ROOT)
            )
            assert record["phase8_semantics_changed"] == "none"
        finally:
            module.REQUIRED_CORPUS_ROOT_RELATIVE = before

    def test_the_untouched_assertion_depends_on_the_checkout_location(self):
        import phase18_g1_control as control

        module = control.a6
        before = module.REQUIRED_CORPUS_ROOT_RELATIVE
        try:
            elsewhere = Path("/somewhere/else/entirely")
            assert elsewhere / before != Path(module.REQUIRED_CORPUS_ROOT)
        finally:
            module.REQUIRED_CORPUS_ROOT_RELATIVE = before
