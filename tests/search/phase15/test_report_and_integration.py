"""End-to-end shape checks: the report, the summary, and the runner's roles."""

import json
import subprocess
import sys

import pytest

from stratego.search.phase15.report_text import build_report, build_summary


def _artifacts():
    """A minimal but complete set of the artifacts the report reads."""
    candidate = {
        "artifact": "phase15_search_candidate_v1",
        "generated_utc": "2026-08-24T12:00:00Z",
        "selected_system": {
            "pairing_id": "p24_b18",
            "move_model": "P24",
            "belief_model": "B18",
        },
        "move_model": {"logical_identity": "P24"},
        "belief_model": {"prefix_backbone": "p18"},
        "search": {
            "search_version": "phase12_root_world_search_v1",
            "score_definition": "S(a) = Q(a) + beta * log(pi(a) + epsilon)",
            "selected_preset": "TINY",
            "worlds": 8,
            "root_candidates": "<= 8",
            "rollout_depth": 4,
        },
        "maximum_strength": {"preset_id": "MEDIUM", "worlds": 32, "rollout_depth": 8},
        "time_caps_seconds": {"selected_search": 0.5, "maximum_strength": 3.5},
        "direct_fallback": {"rule": "play the direct legal move"},
        "known_limitations": ["a compact engineering pack"],
        "oracle_available_in_production": False,
        "scientific_validation_status": "not performed",
    }
    arm = {
        "arm_id": "p24_b18",
        "games": 4,
        "wins": 2,
        "draws": 1,
        "losses": 1,
        "ewr": 0.625,
        "ewr_by_opponent": {"p18": {"ewr": 0.5, "games": 4}},
        "min_opponent": {"name": "p18", "ewr": 0.5},
        "min_family": {"name": "miner_forward", "ewr": 0.5},
        "weakness_pack_family_ewr": 0.5,
        "median_seconds_per_move": 0.12,
        "p95_seconds_per_move": 0.15,
        "move_change_rate": 0.1,
        "fallback_rate": 0.0,
        "paired_vs_direct": {"delta": 0.1, "standard_error": 0.05},
    }
    return {
        "boundary": {"verdict": "clear_to_run"},
        "match_manifest": {
            "board_count": 4,
            "manifest_digest": "a" * 64,
            "orientation_rule": "red engine row == canonical rank",
            "library_split": "validation",
            "balance": {
                "by_setup_source": {"neutral_v1": 2},
                "by_color": {"red": 2, "blue": 2},
            },
        },
        "position_manifest": {"position_count": 8, "manifest_digest": "b" * 64},
        "gate": {
            "passed": True,
            "checks_passed": 10,
            "checks_run": 10,
            "seconds": 51.0,
            "failed": [],
            "gate_version": "phase15_correctness_gate_v1",
            "checks": {
                "identities": {"passed": True},
                "model_roles": {
                    "passed": True,
                    "direct_action_provider_invariant": 24,
                    "positions_where_search_differed_by_provider": 24,
                    "positions_where_p18_and_p24_differ": 3,
                    "positions": 12,
                },
                "permutation_invariance": {
                    "passed": True,
                    "production_checks": 48,
                    "oracle_sensitive": 6,
                    "oracle_checks": 24,
                },
                "fallback": {
                    "passed": True,
                    "timeout_fallbacks": 48,
                    "error_fallbacks": 48,
                },
                "oracle_refusals": {"passed": True, "refusals": {"a": "x", "b": "y"}},
                "decisions": {
                    "passed": True,
                    "decisions": 48,
                    "candidates_checked": 300,
                    "worlds_checked": 350,
                },
                "phase12_frozen_candidate_regression": {
                    "passed": True,
                    "result": "131 passed",
                },
            },
        },
        "stage_a": {
            "positions": 120,
            "preset": "TINY",
            "arms": {
                "p24_b18": {
                    "move_change_rate_vs_direct": 0.1,
                    "oracle_agreement": 0.92,
                    "legal_decision_rate": 1.0,
                    "median_seconds": 0.12,
                    "p95_seconds": 0.15,
                    "mean_c1_forwards": 245,
                    "world_uniqueness": 0.9,
                    "median_score_margin": 0.01,
                }
            },
            "interpretation": {
                "p24": {"reading": "learned_belief_tracks_oracle", "note": "it does"}
            },
        },
        "stage_b": {
            "games_played": 8,
            "arms": ["p24_b18", "p24_direct"],
            "boards": 4,
            "preset": "TINY",
            "wall_seconds": 60.0,
            "workers": 10,
            "summaries": {"p24_b18|TINY": arm},
            "probes": {
                "p24_b18": {
                    "permutation_checks": 4,
                    "permutation_sensitive": 0,
                    "expects_hidden_truth": False,
                    "passed": True,
                }
            },
            "probe_passed": True,
        },
        "budget": {
            "games_played": 8,
            "pairings": ["p24_b18"],
            "presets": ["TINY", "SMALL", "MEDIUM"],
            "ladder_boards": 4,
            "wall_seconds": 120.0,
            "profiles": {
                "p24_b18": {
                    "ladder": {
                        "order": ["TINY", "SMALL", "MEDIUM"],
                        "rungs": [
                            {
                                "preset_id": "TINY",
                                "worlds": 8,
                                "rollout_depth": 4,
                                "ewr": 0.625,
                                "search_seconds_per_game": 6.0,
                                "ewr_gain_per_added_search_second": None,
                                "median_seconds_per_move": 0.12,
                                "p95_seconds_per_move": 0.15,
                                "human_play": {"verdict": "comfortable"},
                                "paired_vs_cheapest": {"delta": 0.0},
                            }
                        ]
                    },
                    "selection": {
                        "selected_preset": "TINY",
                        "strongest_observed_preset": "TINY",
                        "strongest_observed_ewr": 0.625,
                        "rule": "cheapest adequate rung",
                    },
                    "maximum_strength": {"mode": "MEDIUM"},
                    "strong_gate": {
                        "allowed": False,
                        "reason": "MEDIUM did not show a useful improvement",
                        "improvement_over_cheaper": 0.01,
                        "useful_improvement_required": 0.1,
                    },
                }
            },
        },
        "matrix": {
            "matrix": {
                "p24_b18": {
                    "direct_ewr": 0.5,
                    "search_ewr": 0.625,
                    "paired_delta_vs_direct": 0.1,
                    "paired_standard_error": 0.05,
                    "worst_opponent": {"name": "p18", "ewr": 0.5},
                    "weakness_pack_family_ewr": 0.5,
                    "median_seconds_per_move": 0.12,
                    "p95_seconds_per_move": 0.15,
                    "fallback_rate": 0.0,
                }
            },
            "selected_pairing": "p24_b18",
            "selected_preset": "TINY",
            "maximum_strength_preset": "MEDIUM",
            "selection": {
                "rule": "composite score",
                "margin": 0.1,
                "contenders_within_margin": ["p24_b18"],
            },
        },
        "candidate": candidate,
    }


def test_the_report_renders_and_states_what_it_is_not():
    text = build_report(_artifacts())
    assert text.startswith("# Phase 15 — Agent 2")
    assert "P24 + B18" in text
    assert "scientific_validation_status: not performed" in text
    assert "No significance claim is made" in text
    assert "oracle_available_in_production = false" in text
    assert "it did not train or modify B18, B24, P18 or P24" in text
    assert "| system |" in text


def test_the_report_names_the_cross_pairing_backbone():
    text = build_report(_artifacts())
    assert "marginals over **P18**'s" in text


def test_the_summary_carries_the_selection_and_the_evidence_digests():
    summary = build_summary(_artifacts())
    assert summary["phase"] == "phase_15"
    assert summary["agent"] == "agent_02"
    assert summary["selected_system"]["pairing_id"] == "p24_b18"
    assert summary["evidence"]["match_manifest_digest"] == "a" * 64
    assert summary["gate"]["passed"] is True
    assert summary["oracle_available_in_production"] is False
    assert summary["scientific_validation_status"] == "not performed"
    assert summary["budget"]["p24_b18"]["strong_gate_allowed"] is False


def test_the_summary_is_json_serializable():
    json.dumps(build_summary(_artifacts()))


# -- the runner -------------------------------------------------------------


def test_the_runner_exposes_every_instructed_role(repository_root):
    result = subprocess.run(
        [sys.executable, "scripts/run_phase15_agent02.py", "--help"],
        cwd=str(repository_root),
        capture_output=True,
        text=True,
        check=True,
    )
    for role in (
        "boundary",
        "boards",
        "positions",
        "gate",
        "stage_a",
        "stage_b",
        "stage_c",
        "select",
        "candidate",
        "report",
    ):
        assert role in result.stdout


def test_the_runner_refuses_an_unknown_role(repository_root):
    result = subprocess.run(
        [sys.executable, "scripts/run_phase15_agent02.py", "--role", "launch_phase14"],
        cwd=str(repository_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def _executable_source(path):
    """A file's source with its docstrings and comments removed.

    The prose in these modules *describes* the things they must not do
    ("creates no emergency-stop file"), so a scan of the raw text would
    match its own safety documentation. Stripping docstrings and comments
    leaves only what actually runs.
    """
    import ast
    import io
    import tokenize

    source = path.read_text()
    without_comments = tokenize.untokenize(
        token
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type != tokenize.COMMENT
    )
    tree = ast.parse(without_comments)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) and ast.get_docstring(node) is not None:
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


def _subprocess_literals(path):
    """Every literal string that reaches a subprocess call in `path`."""
    import ast

    tree = ast.parse(path.read_text())
    launched = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = getattr(target, "attr", getattr(target, "id", ""))
        if name not in ("run", "Popen", "call", "check_output", "check_call"):
            continue
        owner = getattr(getattr(target, "value", None), "id", "")
        if owner and owner != "subprocess":
            continue
        for argument in ast.walk(node):
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                launched.append(argument.value)
    return launched


def test_the_runner_never_signals_or_finalizes_anything(repository_root):
    """Section 2: this script controls no process it did not start.

    Two separate claims. First, nothing in the executable source signals a
    process at all. Second — and this is the one a text scan gets wrong —
    the `boundary` role legitimately *reads* `ps` output looking for the
    string `phase14_launch`, which is exactly the read-only inspection
    section 2 asks for; what must never happen is a Phase 14 command being
    *launched*. So the process-control check reads the arguments that
    actually reach a subprocess call, not the file's text.
    """
    runner = repository_root / "scripts" / "run_phase15_agent02.py"
    source = _executable_source(runner)
    for forbidden in ("os.kill", "SIGTERM", "SIGKILL", "killpg", ".terminate(", "psutil"):
        assert forbidden not in source, f"the runner references {forbidden!r}"

    launched = _subprocess_literals(runner)
    assert launched, "the boundary role does run `ps`; the scan found nothing"
    for literal in launched:
        lowered = literal.lower()
        for forbidden in ("phase14", "finalize", "close", "stop", "kill", "emergency"):
            assert forbidden not in lowered, (
                f"the runner passes {literal!r} to a subprocess call"
            )


def test_no_phase15_module_signals_a_process(repository_root):
    package = repository_root / "stratego" / "search" / "phase15"
    for path in sorted(package.glob("*.py")):
        source = _executable_source(path)
        for forbidden in ("os.kill", "SIGKILL", "SIGTERM", "killpg", "psutil"):
            assert forbidden not in source, f"{path.name} references {forbidden!r}"


def test_the_package_never_writes_outside_phase15(repository_root):
    """No Phase 15 module may write into another phase's directory."""
    package = repository_root / "stratego" / "search" / "phase15"
    for path in sorted(package.glob("*.py")):
        source = _executable_source(path)
        for forbidden in (
            "checkpoints/phase12",
            "checkpoints/phase14",
            "reports/phase12",
            "reports/phase11b",
            "data/phase11b",
        ):
            assert forbidden not in source, f"{path.name} references {forbidden}"


# -- the human-play CLI -----------------------------------------------------


def test_the_cli_offers_no_oracle_seat(repository_root):
    result = subprocess.run(
        [sys.executable, "scripts/play_phase15.py", "--help"],
        cwd=str(repository_root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "oracle" not in result.stdout
    assert "selected_search" in result.stdout
    assert "maximum_strength" in result.stdout


def test_the_cli_never_draws_through_the_contaminated_glue(repository_root):
    """`play_phase12.py` draws through `Phase11BSetupSources`; this must not."""
    source = (repository_root / "scripts" / "play_phase15.py").read_text()
    assert "Phase11BSetupSources" not in source.replace(
        "through `Phase11BSetupSources`", ""
    )
    assert "Phase15MatchSetupSources" in source
    assert "check_board" in source


def test_the_cli_setups_pass_the_orientation_gate(repository_root):
    """The human-play path is gated exactly like the match path."""
    sys.path.insert(0, str(repository_root / "scripts"))
    try:
        import play_phase15
    finally:
        sys.path.pop(0)

    class _Args:
        setup_source = "phase14_learned"
        setup_seed = 3

    red, blue = play_phase15.draw_setups(_Args())
    assert len(red) == len(blue) == 40
