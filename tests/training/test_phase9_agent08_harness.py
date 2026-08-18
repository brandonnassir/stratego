"""Phase 9 Agent 8: the final-acceptance harness's own logic, with controls.

Agent 8 adds three kinds of logic that are not already frozen in a library
module, and each can fail silently:

```text
the observer probe-count rule that reconciles the resumed iteration 30
the hard-gate table whose booleans must equal their own observed/threshold rows
the paired-difference bootstrap token the improvement CIs are seeded from
```

Every check is paired with a control that must fail: a probe rule that
counted rule-opponent plies as neural, a gate table whose published boolean
contradicts its numbers, or a recommendation that ignores a failed gate has
to be caught here rather than in review.

The module is imported by path because it is a script; nothing in it runs at
import time beyond its own imports (and it stays torch-free at module scope,
which `test_module_scope_is_torch_free` pins).
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPOSITORY_ROOT / "scripts" / "run_phase9_agent08.py"


@pytest.fixture(scope="module")
def harness():
    specification = importlib.util.spec_from_file_location(
        "run_phase9_agent08_under_test", HARNESS_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_module_scope_is_torch_free(harness):
    # Game workers spawn re-imports of __main__; the harness must stay
    # importable without torch, exactly like the accepted Agent 1/6/7 scripts.
    assert "torch" not in sys.modules or True  # torch may be loaded by other tests
    source = HARNESS_PATH.read_text(encoding="utf-8")
    module_scope = [
        line
        for line in source.splitlines()
        if line.startswith("import torch") or line.startswith("from torch")
    ]
    assert module_scope == []


# ---------------------------------------------------------------------------
# The observer probe-count rule
# ---------------------------------------------------------------------------


def test_probe_rule_current_and_historical_count_every_ply(harness):
    # Both sides are neural, so the quota fills on the first two plies.
    assert harness.probes_for_game("current_policy", None, 1) == 1
    assert harness.probes_for_game("current_policy", None, 2) == 2
    assert harness.probes_for_game("current_policy", None, 250) == 2
    assert harness.probes_for_game("historical_snapshot", "red", 9) == 2
    assert harness.probes_for_game("historical_snapshot", None, 1) == 1


def test_probe_rule_rule_and_stress_count_learner_plies_only(harness):
    # Red acts on plies 1, 3, 5, ...: a red learner reaches two neural plies
    # at ply 3; a blue learner (plies 2, 4) only at ply 4.
    assert harness.probes_for_game("rule_policy", "red", 1) == 1
    assert harness.probes_for_game("rule_policy", "red", 2) == 1
    assert harness.probes_for_game("rule_policy", "red", 3) == 2
    assert harness.probes_for_game("stress_policy", "blue", 1) == 0
    assert harness.probes_for_game("stress_policy", "blue", 2) == 1
    assert harness.probes_for_game("stress_policy", "blue", 3) == 1
    assert harness.probes_for_game("stress_policy", "blue", 4) == 2


def test_probe_rule_control_wrong_parity_would_disagree(harness):
    # The control the rule validation rests on: if rule-opponent plies were
    # wrongly counted as neural, a 3-ply blue-learner game would claim 2
    # probes where the collector takes 1. The two answers must differ, or the
    # 60-iteration validation could not distinguish the rules.
    wrong = min(2, 3)  # counting every ply
    right = harness.probes_for_game("rule_policy", "blue", 3)
    assert wrong != right


def test_probe_rule_refuses_unattributable_asymmetric_game(harness):
    with pytest.raises(Exception):
        harness.probes_for_game("rule_policy", None, 10)


# ---------------------------------------------------------------------------
# The paired-difference token
# ---------------------------------------------------------------------------


def test_diff_token_is_the_frozen_shape(harness):
    token = harness.diff_matchup_token("cand@1", "anchor@2", "opp@3")
    assert token == "diff|cand@1|anchor@2|opp@3"


def test_diff_token_orders_candidate_before_anchor(harness):
    forward = harness.diff_matchup_token("a", "b", "c")
    swapped = harness.diff_matchup_token("b", "a", "c")
    assert forward != swapped  # the seed must not survive a role swap


# ---------------------------------------------------------------------------
# The hard-gate table and its self-consistency validator
# ---------------------------------------------------------------------------


def _passing_final_stage() -> dict:
    return {
        "gates": {
            "gate_a": {
                "ewr": 0.70,
                "ewr_min": 0.58,
                "ci_lower": 0.66,
                "ci_lower_exclusive": 0.53,
                "games": 1024,
                "passed": True,
            },
            "gate_b": {
                "ewr": 0.80,
                "ewr_min": 0.52,
                "anchor_ewr": 0.45,
                "paired_improvement": 0.35,
                "paired_improvement_min": 0.05,
                "improvement_ci_lower": 0.30,
                "improvement_ci_lower_exclusive": 0.0,
                "stretch_0_55_report_only": True,
                "passed": True,
            },
            "gate_c": {
                "ewr": 0.79,
                "ewr_min": 0.52,
                "anchor_ewr": 0.46,
                "paired_improvement": 0.33,
                "paired_improvement_min": 0.05,
                "improvement_ci_lower": 0.28,
                "improvement_ci_lower_exclusive": 0.0,
                "stretch_0_55_report_only": True,
                "passed": True,
            },
            "gate_d": {
                "ewr": 0.99,
                "overall_min": 0.94,
                "red_ewr": 1.0,
                "blue_ewr": 0.98,
                "color_min": 0.90,
                "ci_lower": 0.97,
                "ci_lower_exclusive": 0.92,
                "passed": True,
            },
            "gate_e": {
                "ewr": 0.84,
                "ewr_min": 0.65,
                "ci_lower": 0.80,
                "ci_lower_exclusive": 0.60,
                "passed": True,
            },
            "gate_f": {
                "illegal_actions": 0,
                "model_failures": 0,
                "non_finite_outputs": 0,
                "observer_probes_on_final_games": 11776,
                "observer_safety_failures": 0,
                "action_reproduction_mismatches": 0,
                "passed": True,
            },
            "gate_g": {
                "population": "every final-candidate decision across the final-test games",
                "decisions": 700000,
                "decisions_above_0_999": 35000,
                "fraction_above_0_999": 0.05,
                "fraction_max_exclusive": 0.25,
                "passed": True,
            },
            "gate_h": {
                "belief_ce_ratio": 0.90,
                "ratio_max": 0.98,
                "belief_top1": 0.42,
                "remaining_count_top1": 0.35,
                "passed": True,
            },
        }
    }


def test_hard_gate_table_carries_the_frozen_eight(harness):
    table = harness.hard_gate_table(_passing_final_stage())
    assert sorted(table) == sorted(name for name, _key in harness.HARD_GATE_ROWS)
    for row in table.values():
        assert "observed" in row and "threshold" in row and "passed" in row


def test_recompute_agrees_with_a_consistent_table(harness):
    table = harness.hard_gate_table(_passing_final_stage())
    recomputed = harness.recompute_gate_booleans(table)
    assert all(recomputed.values())


def test_recompute_detects_a_failed_gate_published_as_passed(harness):
    final = _passing_final_stage()
    final["gates"]["gate_e"]["ewr"] = 0.60  # below the frozen 0.65 floor
    table = harness.hard_gate_table(final)
    # The builder honestly copies the (stale) boolean; recompute must disagree.
    assert table["gate_e_basic_guard"]["passed"] is True
    recomputed = harness.recompute_gate_booleans(table)
    assert recomputed["gate_e_basic_guard"] is False


def test_recompute_detects_a_ci_lower_bound_at_the_boundary(harness):
    # The frozen bounds are exclusive: a lower bound exactly at 0.53 fails A.
    final = _passing_final_stage()
    final["gates"]["gate_a"]["ci_lower"] = 0.53
    table = harness.hard_gate_table(final)
    assert harness.recompute_gate_booleans(table)["gate_a_vs_phase8_anchor"] is False


def test_recompute_detects_nonzero_safety_counters(harness):
    final = _passing_final_stage()
    final["gates"]["gate_f"]["illegal_actions"] = 1
    table = harness.hard_gate_table(final)
    assert harness.recompute_gate_booleans(table)["gate_f_safety"] is False


def test_validator_accepts_a_consistent_artifact(harness):
    table = harness.hard_gate_table(_passing_final_stage())
    artifact = {
        "hard_gates": table,
        "recommendation": "PASS",
        "completion_gates": {"anything": True},
    }
    assert harness.validate_acceptance_artifact(artifact) == []


def test_validator_rejects_a_tampered_boolean(harness):
    final = _passing_final_stage()
    final["gates"]["gate_c"]["paired_improvement"] = 0.01  # below +0.05
    table = harness.hard_gate_table(final)
    artifact = {
        "hard_gates": table,
        "recommendation": "FAIL",
        "completion_gates": {"anything": True},
    }
    problems = harness.validate_acceptance_artifact(artifact)
    assert any("gate_c_tactical" in problem for problem in problems)


def test_validator_rejects_pass_with_a_false_gate(harness):
    final = _passing_final_stage()
    final["gates"]["gate_b"]["ewr"] = 0.40
    final["gates"]["gate_b"]["passed"] = False
    table = harness.hard_gate_table(final)
    artifact = {
        "hard_gates": table,
        "recommendation": "PASS",
        "completion_gates": {"anything": True},
    }
    problems = harness.validate_acceptance_artifact(artifact)
    assert any("recommendation PASS" in problem for problem in problems)


def test_validator_rejects_fail_when_everything_passes(harness):
    table = harness.hard_gate_table(_passing_final_stage())
    artifact = {
        "hard_gates": table,
        "recommendation": "FAIL",
        "completion_gates": {"anything": True},
    }
    problems = harness.validate_acceptance_artifact(artifact)
    assert any("recommendation FAIL" in problem for problem in problems)


def test_validator_requires_the_frozen_row_names(harness):
    table = harness.hard_gate_table(_passing_final_stage())
    del table["gate_h_belief_retention"]
    artifact = {
        "hard_gates": table,
        "recommendation": "PASS",
        "completion_gates": {},
    }
    problems = harness.validate_acceptance_artifact(artifact)
    assert problems  # a missing row can never validate


# ---------------------------------------------------------------------------
# The frozen constants the harness pins
# ---------------------------------------------------------------------------


def test_accepted_digests_match_the_frozen_modules(harness):
    from stratego.training.phase9_amendment import amendment_digest
    from stratego.training.phase9_amendment_v2 import (
        amendment_digest as amendment_v2_digest,
    )
    from stratego.training.phase9_contract import contract_digest

    assert harness.ACCEPTED_CONTRACT_DIGEST == contract_digest()
    assert harness.ACCEPTED_AMENDMENT_DIGEST == amendment_digest()
    assert harness.ACCEPTED_AMENDMENT_V2_DIGEST == amendment_v2_digest()


def test_ceiling_chain_is_the_reviewed_12_15_24(harness):
    assert harness.ACCEPTED_CEILING_CHAIN == ((12, 43_200), (15, 54_000), (24, 86_400))


def test_selected_iteration_and_source_snapshot_are_pinned(harness):
    assert harness.ACCEPTED_SELECTED_ITERATION == 40
    assert harness.ACCEPTED_SOURCE_SNAPSHOT == "behavior_B041.pt"


def test_working_tree_freeze_reports_the_head_commit(harness):
    freeze = harness.working_tree_freeze()
    assert len(freeze["head_commit"]) == 40
    assert isinstance(freeze["tracked_drift"], list)
    assert isinstance(freeze["untracked_files"], list)
    assert len(freeze["agent7_artifacts_in_head"]) >= 4
