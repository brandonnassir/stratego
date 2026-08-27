"""Phase 15 Agent 1: the produced artifacts, and what they promised not to do.

Artifacts are skipped when absent so a fresh clone still runs green — the
accepted Phase 9-14 pattern. When they are present these tests hold them to
the instruction: the boundary was respected, the orientation gate passed
before any corpus existed, the corpus is the size and shape section 5 asked
for, the specialists did not move P18 or P24, the handoff binds every digest
section 13 names, and nothing here claims a playing strength.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stratego.belief.phase15 import contract as C
from stratego.belief.phase15.handoff import HANDOFF_VERSION, verify_handoff

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPOSITORY_ROOT / "reports" / "phase15"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "phase15"
DATA_ROOT = REPOSITORY_ROOT / "data" / "phase15"


def _load(path: Path):
    if not path.is_file():
        pytest.skip(f"{path.name} has not been produced yet")
    return json.loads(path.read_text())


@pytest.fixture()
def boundary():
    return _load(REPORT_ROOT / "agent_01_process_boundary.json")


@pytest.fixture()
def orientation():
    return _load(REPORT_ROOT / "agent_01_orientation_gate.json")


@pytest.fixture()
def manifest():
    return _load(DATA_ROOT / "phase15_belief_corpus_v1_manifest.json")


@pytest.fixture()
def verification():
    return _load(REPORT_ROOT / "agent_01_corpus_verification.json")


@pytest.fixture()
def metrics():
    return _load(REPORT_ROOT / "agent_01_metrics.json")


@pytest.fixture()
def checks():
    return _load(REPORT_ROOT / "agent_01_interface_checks.json")


@pytest.fixture()
def handoff():
    return _load(REPORT_ROOT / "phase15_search_handoff_v1.json")


@pytest.fixture()
def summary():
    return _load(REPORT_ROOT / "agent_01_summary.json")


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


def test_no_process_control_was_claimed(boundary):
    assert boundary["phase15_authorizes_process_control"] is False
    assert boundary["competing_phase14_processes"] == []
    assert boundary["verdict"] == "ready_for_compute"


def test_no_phase15_module_can_reach_a_control_command():
    package = REPOSITORY_ROOT / "stratego" / "belief" / "phase15"
    forbidden = (
        "phase14_emergency_stop",
        "emergency_stop",
        "os.kill",
        "SIGKILL",
        "SIGTERM",
        "--role finalize",
        "phase14_select_final",
    )
    offenders = []
    for path in sorted(package.glob("*.py")):
        text = path.read_text()
        for token in forbidden:
            # The runbook name appears in prose; only code may not use it.
            for line in text.splitlines():
                stripped = line.strip()
                if token in stripped and not stripped.startswith(("#", '"', "'")):
                    offenders.append(f"{path.name}: {stripped[:70]}")
    assert offenders == []


def test_the_status_markers_say_this_is_not_a_strength_claim(metrics, handoff):
    for document in (metrics, handoff):
        assert document["phase"] == "phase_15"
        assert document["agent"] == "agent_01"
        assert document["search_implemented"] is False
        assert document["status"] == "engineering_deliverable_not_a_strength_claim"


def test_the_handoff_excludes_everything_section_14_forbids(handoff):
    excluded = " ".join(handoff["not_included"]).lower()
    for token in ("search", "final combined player", "phase 12", "phase 14"):
        assert token in excluded


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


def test_the_orientation_gate_passed_at_scale(orientation):
    assert orientation["passed"] is True
    assert orientation["armies_checked"] >= 2000
    assert orientation["negative_canary"]["detected"] is True
    assert orientation["inventory_exact"] is True
    assert orientation["paired_orientation"] is True


def test_the_corpus_shows_the_defect_is_gone(orientation):
    assert orientation["front_row_flag_rate"] < 0.05
    assert orientation["defect_counterfactual"]["rate"] > 0.5


def test_the_manifest_carries_the_orientation_evidence(manifest):
    assert manifest["orientation"]["passed"] is True
    assert "canonical rank" in manifest["orientation"]["orientation_rule"]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def test_every_split_met_its_target_or_at_least_its_floor(manifest):
    for split, block in manifest["splits"].items():
        assert block["samples"] >= C.POSITION_FLOOR[split], split
        if not block["met_target"]:
            pytest.fail(
                f"{split} fell back to the floor without recorded pilot evidence"
            )


def test_the_corpus_is_not_the_contaminated_one(manifest):
    assert manifest["corpus_version"] == "phase15_belief_corpus_v1"
    assert manifest["corpus_version"] != "phase11b_common_corpus_v1"
    assert manifest["run_version"].startswith("phase15_")


def test_the_accepted_termination_cap_was_preserved(manifest):
    cap = manifest["termination_cap"]
    assert cap["battleless_move_limit"] == 200
    assert cap["absolute_move_limit"] == 4000
    assert "unchanged" in cap["rules"]


def test_the_corpus_digest_does_not_embed_wall_clock(manifest):
    from stratego.belief.phase15.storage import corpus_digest

    assert corpus_digest(manifest) == manifest["corpus_digest"]
    assert "generation_seconds" in manifest


def test_the_splits_are_disjoint(verification):
    assert verification["passed"] is True
    assert verification["disjointness"]["disjoint"] is True
    for name, block in verification["disjointness"]["pairs"].items():
        assert block["shared_game_ids"] == 0, name
        assert block["shared_public_state_identities"] == 0, name


def test_the_achieved_mixture_is_close_to_the_design(verification):
    for split, block in verification["splits"].items():
        deviation = block["mixture"]["max_absolute_deviation"]
        assert deviation["observer_model"] < 0.05, split
        assert deviation["observer_color"] < 0.05, split
        assert deviation["opponent"] < 0.05, split
        assert deviation["setup_source"] < 0.05, split


def test_every_targeted_family_reached_every_split(verification):
    for split, block in verification["splits"].items():
        assert block["mixture"]["targeted_families_missing"] == [], split


def test_every_stored_label_is_publicly_admissible(verification):
    for split, block in verification["splits"].items():
        labels = block["labels"]
        assert labels["all_ranks_publicly_admissible"] is True, split
        assert labels["all_ranks_have_remaining_inventory"] is True, split
        assert labels["moved_pieces_with_immobile_rank"] == 0, split


def test_the_privileged_labels_live_in_their_own_directory():
    for split in C.CORPUS_SPLITS:
        root = DATA_ROOT / "phase15_belief_corpus_v1" / split
        if not root.is_dir():
            pytest.skip("the corpus has not been generated yet")
        assert (root / C.PRIVILEGED_DIRECTORY / "true_rank.npy").is_file()
        public = sorted(path.name for path in (root / C.PUBLIC_DIRECTORY).iterdir())
        assert "true_rank.npy" not in public


# ---------------------------------------------------------------------------
# The specialists
# ---------------------------------------------------------------------------


def test_the_source_policies_were_not_changed_by_training(metrics):
    for specialist_id, block in metrics["specialists"].items():
        unchanged = block["training"]["source_unchanged"]
        assert unchanged["unchanged"] is True, specialist_id
        assert (
            unchanged["model_state_digest_before"]
            == unchanged["model_state_digest_after"]
        )


def test_no_policy_or_value_parameter_took_a_gradient(metrics):
    for specialist_id, block in metrics["specialists"].items():
        isolation = block["training"]["gradient_isolation"]
        assert isolation["policy_value_parameters_with_gradient"] == 0, specialist_id
        assert isolation["policy_value_parameters_requiring_grad"] == 0, specialist_id
        assert isolation["checked_parameters"] > 0


def test_both_specialists_ran_the_same_declared_recipe(metrics):
    configs = {}
    for specialist_id, block in metrics["specialists"].items():
        config = dict(block["training"]["config"])
        config.pop("specialist_id")
        configs[specialist_id] = config
    values = list(configs.values())
    assert values[0] == values[1], configs


def test_the_recipe_matches_the_contract(metrics):
    for block in metrics["specialists"].values():
        config = block["training"]["config"]
        assert config["head_learning_rate"] == C.RECIPE["head_learning_rate"]
        assert config["block_learning_rate"] == C.RECIPE["final_block_learning_rate"]
        assert config["weight_decay"] == C.RECIPE["weight_decay"]
        assert config["epochs"] == C.RECIPE["max_epochs"]
        assert config["patience"] == C.RECIPE["early_stop_patience"]
        assert config["optimizer"] == "adamw"
        assert config["schedule"] == "cosine"


def test_the_selected_epoch_is_the_best_development_cross_entropy():
    curves = _load(REPORT_ROOT / "agent_01_learning_curves.json")
    metrics = _load(REPORT_ROOT / "agent_01_metrics.json")
    for specialist_id, curve in curves["curves"].items():
        best = min(row["dev_ce"] for row in curve)
        chosen = metrics["specialists"][specialist_id]["training"]["best_epoch"]
        assert next(row for row in curve if row["epoch"] == chosen)["dev_ce"] == best


def test_each_belief_checkpoint_holds_no_policy_or_value_tensor():
    import torch

    for specialist_id in C.SPECIALISTS:
        path = CHECKPOINT_ROOT / f"{specialist_id}_belief_v1.pt"
        if not path.is_file():
            pytest.skip("the specialists have not been trained yet")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["holds_policy_parameters"] is False
        assert payload["holds_value_parameters"] is False
        for name in payload["state_dict"]:
            assert name.startswith(
                ("block.", "encoder_norm.", "head.", "log_temperature")
            )


def test_each_specialist_is_bound_to_its_own_source(metrics):
    assert metrics["specialists"]["b18"]["source_id"] == "p18"
    assert metrics["specialists"]["b24"]["source_id"] == "p24"


# ---------------------------------------------------------------------------
# Calibration and metrics
# ---------------------------------------------------------------------------


def test_calibration_used_only_the_calibration_split(metrics, manifest):
    for block in metrics["specialists"].values():
        calibration = block["calibration"]
        assert calibration["calibration_pieces"] == manifest["splits"]["calibration"][
            "pieces"
        ]
        assert calibration["temperature"] > 0.0
        assert calibration["top1_labels_changed"] == 0


def test_a_kept_temperature_improved_both_nll_and_calibration(metrics):
    for specialist_id, block in metrics["specialists"].items():
        calibration = block["calibration"]
        if calibration["keep_calibrated"]:
            assert calibration["development_nll_improved"] is True, specialist_id
            assert calibration["development_ece_improved"] is True, specialist_id
            assert calibration["top1_unchanged"] is True, specialist_id
            assert calibration["applied_temperature"] == pytest.approx(
                calibration["temperature"]
            )
        else:
            assert calibration["applied_temperature"] == pytest.approx(1.0)


def test_temperature_scaling_left_top_one_accuracy_untouched(metrics):
    for specialist_id, block in metrics["specialists"].items():
        assert block["development_raw"]["top1"] == pytest.approx(
            block["development_calibrated"]["top1"]
        ), specialist_id


def test_every_required_metric_is_present(metrics):
    required = (
        "ce",
        "nll",
        "baseline_ce",
        "r_ce",
        "top1",
        "brier",
        "expected_calibration_error",
        "maximum_calibration_error",
    )
    for specialist_id, block in metrics["specialists"].items():
        for view in ("development_raw", "development_calibrated"):
            for key in required:
                assert key in block[view], f"{specialist_id}.{view}.{key}"


def test_every_required_breakdown_is_present(metrics):
    required = (
        "observer_color",
        "observer_source",
        "opponent",
        "opponent_class",
        "setup_source",
        "opponent_setup_family",
        "game_band",
    )
    for specialist_id, block in metrics["specialists"].items():
        breakdowns = block["development_calibrated"]["breakdowns"]
        for dimension in required:
            assert dimension in breakdowns, f"{specialist_id}.{dimension}"
            assert breakdowns[dimension], f"{specialist_id}.{dimension} is empty"


def test_the_agent1c_reference_was_scored_on_the_new_corpus(metrics):
    comparison = metrics["comparison"]
    reference = comparison["agent1c_reference"]
    assert reference["reference_id"].startswith("phase11b_agent01_1c")
    # The old number is quoted only to identify the artifact; the reported
    # R_CE is the one measured here, and the two must not be the same object.
    assert reference["r_ce"] != reference["old_development_r_ce"]
    assert comparison["development_positions"] > 10000


def test_the_baseline_and_the_uniform_floor_bracket_the_models(metrics):
    comparison = metrics["comparison"]
    assert comparison["uniform_reference"]["r_ce"] > 1.0
    for specialist_id, block in comparison["specialists"].items():
        assert block["r_ce"] > 0.0, specialist_id
        assert block["r_ce"] < comparison["uniform_reference"]["r_ce"], specialist_id


# ---------------------------------------------------------------------------
# Interface and handoff
# ---------------------------------------------------------------------------


def test_every_provider_check_passed(checks):
    for specialist_id, block in checks["providers"].items():
        assert block["passed"] is True, specialist_id
        assert block["probabilities_finite"] is True
        assert block["probabilities_sum_to_one"] is True
        assert block["fixed_seed_reproduces_worlds"] is True
        assert block["remaining_piece_counts_exact"] is True
        assert block["moved_pieces_never_immobile"] is True
        assert block["all_worlds_pass_accepted_validation"] is True
        assert block["independent_per_piece_sampling"] is False
        assert block["worlds_checked"] > 0


def test_the_provider_uses_the_accepted_sampler_by_import(checks):
    for block in checks["providers"].values():
        assert "phase11_sampler" in block["sampler_is_accepted_by_import"]
        assert "unmodified" in block["sampler_is_accepted_by_import"]
        assert block["describe"]["uses_hidden_truth"] is False


def test_no_true_rank_is_reachable_through_the_public_interface(checks):
    for specialist_id, block in checks["providers"].items():
        isolation = block["truth_isolation"]
        assert isolation["public_state_fields"] == [
            "observation",
            "public_state_document",
        ], specialist_id
        assert isolation["public_document_exposes_unresolved_ranks"] is False
        assert isolation["provider_uses_hidden_truth"] is False
        assert isolation["answers_from_the_public_document_alone"] is True


def test_the_handoff_binds_every_required_digest(handoff):
    assert handoff["artifact"] == HANDOFF_VERSION
    for source_id in ("p18", "p24"):
        block = handoff["policy_models"][source_id]
        assert len(block["model_state_digest"]) == 64
        assert len(block["checkpoint_sha256"]) == 64
        assert len(block["phase14_archive_sha256"]) == 64
    for specialist_id in ("b18", "b24"):
        block = handoff["belief_models"][specialist_id]
        assert len(block["state_digest"]) == 64
        assert len(block["checkpoint_sha256"]) == 64
        assert block["calibration"]["applied_temperature"] > 0.0
        assert block["calibration"]["fitted_temperature"] > 0.0
    assert len(handoff["corpus"]["corpus_digest"]) == 64
    assert handoff["provider"]["interface_version"]
    assert handoff["provider"]["accepted_sampler_version"]


def test_the_handoff_still_describes_the_bytes_on_disk(handoff):
    report = verify_handoff(handoff, root=REPOSITORY_ROOT)
    assert report["findings"] == []
    assert report["verified"] is True
    assert report["artifacts_checked"] == 4


def test_the_handoff_binds_each_specialist_to_the_right_policy(handoff):
    assert handoff["belief_models"]["b18"]["bound_policy"] == "p18"
    assert handoff["belief_models"]["b24"]["bound_policy"] == "p24"
    assert handoff["policy_models"]["p18"]["phase14_candidate_hour"] == 18
    assert handoff["policy_models"]["p24"]["phase14_candidate_hour"] == 24


def test_the_summary_agrees_with_the_metrics(summary, metrics):
    for specialist_id, block in summary["specialists"].items():
        assert block["r_ce"] == pytest.approx(
            metrics["comparison"]["specialists"][specialist_id]["r_ce"]
        )
        assert block["checkpoint_sha256"] == metrics["specialists"][specialist_id][
            "checkpoint"
        ]["sha256"]


def test_the_report_exists_and_names_no_strength_claim():
    path = REPORT_ROOT / "agent_01_report.md"
    if not path.is_file():
        pytest.skip("the report has not been written yet")
    text = path.read_text()
    assert "not** a playing-strength claim" in text or "not a playing-strength" in text
    assert "phase15_belief_corpus_v1" in text
    assert "47-of-64" in text or "47 of 64" in text
