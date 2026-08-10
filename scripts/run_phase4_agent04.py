#!/usr/bin/env python3
"""Phase 4 Agent 4 acceptance harness -- calibration, security audit, sign-off.

Runs every Phase 4 completion gate and writes
`reports/phase_4_data/agent_04_calibration_security.json`:

- prerequisites: Agents 1-3 all `PASS`, and Agent 1's frozen setup-bank artefact
  still regenerates byte-for-byte;
- a >= 100,000 trial policy-level hidden-information audit across all ten
  catalogued policies, with a positive control and a leak detector;
- the tuning record: the one policy change this agent made, with the ablation
  that justifies it re-derivable rather than merely asserted;
- a screening league, then the final calibration league with paired confidence
  intervals over the paired unit;
- strength-tier partitioning from the pairwise intervals;
- stress-policy behavioural characterisation by replay of the league's own rows;
- a final reproducibility check: serial, parallel and shuffled;
- checkpoint readiness: an observation-consuming policy driven through the
  gauntlet path a future neural checkpoint will use.

What this script is not: it is not a match runner, not a statistics library and
not a policy suite. Agents 1-3 own those and this script calls them.

Usage:

    python scripts/run_phase4_agent04.py                  # full acceptance run
    python scripts/run_phase4_agent04.py --quick          # fast smoke run
    python scripts/run_phase4_agent04.py --skip-pytest    # measurements only
    python scripts/run_phase4_agent04.py --stage audit    # one stage only
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.engine.constants import (  # noqa: E402
    EVALUATION_RULES,
    IMPLEMENTATION_VERSION,
    ACTION_SPACE_SIZE,
    OBSERVATION_VERSION,
    RULES_VERSION,
)
from stratego.evaluation.baselines import (  # noqa: E402
    BASELINE_SUITE_VERSION,
    STRATEGIC_WEIGHTS,
    ScoredMove,
    StrategicRuleBasedPolicy,
)
from stratego.evaluation.calibration import (  # noqa: E402
    AUDIT_PLIES,
    CALIBRATION_VERSION,
    POSITION_SOURCES,
    STRESS_COMPARISON_METRICS,
    behavior_divergence,
    profile_replay,
    run_hidden_information_audit,
    strength_tiers,
    summarise_behavior,
)
from stratego.evaluation.heuristics import (  # noqa: E402
    HEURISTICS_VERSION,
    PIECE_VALUES,
    CandidateMove,
    DecisionContext,
)
from stratego.evaluation.match_runner import (  # noqa: E402
    MATCH_RESULT_SCHEMA_VERSION,
    MATCH_RUNNER_VERSION,
    compare_results,
    run_schedule,
)
from stratego.evaluation.match_spec import (  # noqa: E402
    EVALUATION_SUITE_VERSION,
    MATCH_SPEC_VERSION,
    build_paired_schedule,
    schedule_matches,
)
from stratego.evaluation.policy import (  # noqa: E402
    POLICY_INTERFACE_VERSION,
    Policy,
    PolicyInput,
    PolicyRequirements,
    PolicyResult,
)
from stratego.evaluation.registry import (  # noqa: E402
    ALL_POLICY_IDS,
    LADDER_POLICY_IDS,
    STRESS_POLICY_IDS,
    build_policy,
    policy_catalog,
    policy_ref,
)
from stratego.evaluation.reporting import (  # noqa: E402
    REPORTING_VERSION,
    write_json,
    write_results_csv,
)
from stratego.evaluation.scheduler import (  # noqa: E402
    SCHEDULER_VERSION,
    build_league_schedule,
    build_matchup_schedule,
    merge_schedules,
    require_valid_schedule,
)
from stratego.evaluation.setup_bank import (  # noqa: E402
    SETUP_BANK_VERSION,
    SetupBank,
    bank_digest,
)
from stratego.evaluation.statistics import (  # noqa: E402
    STATISTICS_VERSION,
    summarize_matchup,
    summarize_run,
)

DATA_DIRECTORY = REPOSITORY_ROOT / "reports/phase_4_data"
DEFAULT_OUTPUT = DATA_DIRECTORY / "agent_04_calibration_security.json"
LEAGUE_CSV = DATA_DIRECTORY / "agent_04_baseline_league_raw.csv"
AUDIT_JSON = DATA_DIRECTORY / "agent_04_hidden_information_audit.json"
BEHAVIOR_CSV = DATA_DIRECTORY / "agent_04_stress_behavior.csv"
TIER_CSV = DATA_DIRECTORY / "agent_04_pairwise_calibration.csv"

#: The audit floor from the Phase 4 contract.
AUDIT_TRIAL_TARGET = 100_000

#: Paired units per matchup class. The Phase 4 instructions ask for at least 512
#: paired units on "important adjacent-tier comparisons" and explicitly permit a
#: league that does not evaluate every pair at the maximum sample size. Spending
#: the same 1,024 units on `stress_chaos` versus `stress_draw_seeker` -- a pair
#: nothing in the acceptance gate cites -- would quadruple both runtime and the
#: raw artefact for no statistical gain, so the budget goes where the tier gate
#: actually reads it. Every reduction is reported in the data file rather than
#: applied silently.
LADDER_UNITS = 1024
LADDER_VS_STRESS_UNITS = 512
STRESS_UNITS = 256

#: Behaviour is profiled by replaying stored action histories, which needs the
#: histories in memory. Profiling all ~44,000 league games would hold tens of
#: millions of integers at once, so a deterministic prefix of setup pairs from
#: the *same* schedule is replayed instead -- a genuine subset of the final
#: league, identified by the same `match_id`s.
BEHAVIOR_UNITS = 96

#: Matchups whose reproduction is re-verified serially, in parallel and shuffled.
REPRODUCIBILITY_UNITS = 64
REPRODUCIBILITY_WORKERS = (1, 2, 4, 8)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260408
AUDIT_ROOT_SEED = 20260407


def log(message: str) -> None:
    print(f"[agent04] {message}", flush=True)


def ladder_pairs() -> list[tuple[str, str]]:
    """The six unordered pairs among the four core ladder policies."""
    identifiers = list(LADDER_POLICY_IDS)
    return [
        (identifiers[i], identifiers[j])
        for i in range(len(identifiers))
        for j in range(i + 1, len(identifiers))
    ]


# ---------------------------------------------------------------------------
# Stage 1 -- prerequisites
# ---------------------------------------------------------------------------


def prerequisite_stage() -> dict:
    """Agents 1-3 must all be PASS, and Agent 1's bank must still regenerate."""
    report = REPOSITORY_ROOT / "reports/phase_4_implementation_report.md"
    text = report.read_text(encoding="utf-8") if report.exists() else ""

    agents: dict[str, dict] = {}
    for agent, filename in (
        ("agent_01", "agent_01_evaluation_foundations.json"),
        ("agent_02", "agent_02_baseline_agents.json"),
        ("agent_03", "agent_03_match_runner_statistics.json"),
    ):
        path = DATA_DIRECTORY / filename
        entry: dict = {"data_file": filename, "present": path.exists()}
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            entry["status"] = payload.get("status")
        agents[agent] = entry

    section_headings = re.findall(r"^## (\d)\. Agent \d .*$", text, flags=re.MULTILINE)
    bank = SetupBank.generate()
    fresh_digest = bank_digest(bank)
    stored_path = DATA_DIRECTORY / "agent_01_setup_bank_v1.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8")) if stored_path.exists() else {}
    stored_digest = (
        json.loads((DATA_DIRECTORY / "agent_01_evaluation_foundations.json").read_text())
        .get("setup_bank", {})
        .get("digest")
    )

    problems: list[str] = []
    for agent, entry in agents.items():
        if entry.get("status") != "PASS":
            problems.append(f"{agent} is not PASS ({entry.get('status')!r})")
    if fresh_digest != stored_digest:
        problems.append("Agent 1's setup bank no longer regenerates to its stored digest")
    if stored.get("pair_count") != len(bank):
        problems.append("the stored bank artefact has a different pair count")
    for expected in ("1", "2", "3"):
        if expected not in section_headings:
            problems.append(f"report section {expected} is missing")

    return {
        "agents": agents,
        "report_sections_present": section_headings,
        "setup_bank_pairs": len(bank),
        "setup_bank_digest_fresh": fresh_digest,
        "setup_bank_digest_stored": stored_digest,
        "setup_bank_digest_matches": fresh_digest == stored_digest,
        "problems": problems,
        "passed": not problems,
    }


# ---------------------------------------------------------------------------
# Stage 2 -- hidden-information audit
# ---------------------------------------------------------------------------


def audit_stage(target_trials: int, workers: int) -> dict:
    """The >= 100,000 trial policy-level hidden-state permutation audit."""
    log(f"audit: {target_trials:,} trials across {len(ALL_POLICY_IDS)} policies, {workers} workers")
    started = time.perf_counter()
    audit = run_hidden_information_audit(
        target_trials,
        workers=workers,
        root_seed=AUDIT_ROOT_SEED,
        policy_ids=list(ALL_POLICY_IDS),
        sources=POSITION_SOURCES,
        plies=AUDIT_PLIES,
    )
    audit["wall_clock_seconds"] = time.perf_counter() - started
    audit["target_trials"] = target_trials

    problems: list[str] = []
    if audit["trials"] < target_trials:
        problems.append(f"only {audit['trials']} trials, target was {target_trials}")
    if audit["total_mismatches"]:
        problems.append(f"{audit['total_mismatches']} hidden-information mismatches")
    if audit["positive_control_failures"]:
        problems.append(
            f"{audit['positive_control_failures']} trials where the privileged belief "
            "target did not change"
        )
    if audit["leak_detector_failures"]:
        problems.append(
            f"{audit['leak_detector_failures']} trials where the hidden types were identical"
        )
    if len(audit["trials_by_policy"]) != len(ALL_POLICY_IDS):
        problems.append("not every catalogued policy was audited")
    if len(audit["trials_by_ply"]) < 4:
        problems.append("fewer than four game phases represented")
    audit["problems"] = problems
    audit["passed"] = not problems
    log(
        f"audit: {audit['trials']:,} trials, {audit['policy_comparisons']:,} comparisons, "
        f"{audit['total_mismatches']} mismatches in {audit['wall_clock_seconds']:.0f}s"
    )
    return audit


# ---------------------------------------------------------------------------
# Stage 3 -- the tuning record
# ---------------------------------------------------------------------------


class LegacyExposureStrategic(StrategicRuleBasedPolicy):
    """`strategic_rule_based` as Agent 2 shipped it, for the ablation only.

    Restores the 1.0.0 exposure rule -- a penalty proportional to the *material
    value* of an identified piece -- on top of the current 1.1.0 code, so the
    change this agent made stays measurable after the fact instead of resting on
    a number quoted in a report. Not catalogued, and never an opponent in the
    calibration league.
    """

    policy_version = "1.0.0-legacy-exposure"

    def score(self, context: DecisionContext, move: CandidateMove) -> ScoredMove:
        scored = super().score(context, move)
        if scored.family in ("flag_capture", "flag_defence"):
            return scored
        total = scored.score
        components = [
            (name, value) for name, value in scored.components if name != "exposure"
        ]
        total -= sum(value for name, value in scored.components if name == "exposure")
        weight = self.weights.exposure
        if weight and context.is_exposed(move.piece_id):
            legacy = PIECE_VALUES[move.piece_type] * move.advance
            if legacy > 0.0:
                value = -weight * legacy
                total += value
                components.append(("exposure", value))
        return ScoredMove(scored.action_id, total, scored.family, tuple(components))


def _ablation_job(payload: tuple[str, int]) -> dict:
    """One variant against `tactical_rule_based`, serially in its own process.

    Serial because `run_schedule`'s worker path resolves policies through the
    catalogue, and the legacy variant is deliberately not in it.
    """
    variant, units = payload
    candidate: Policy = (
        LegacyExposureStrategic() if variant == "legacy" else build_policy("strategic_rule_based")
    )
    opponent = build_policy("tactical_rule_based")
    bank = SetupBank.generate()
    matches = schedule_matches(
        build_paired_schedule(candidate.ref, opponent.ref, list(range(units)))
    )
    policies = {candidate.ref.token: candidate, opponent.ref.token: opponent}
    run = run_schedule(matches, bank, policies=policies, worker_count=1, record_actions=False)
    summary = summarize_matchup(run.results, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    return {
        "variant": variant,
        "policy": candidate.ref.token,
        "paired_units": summary.paired_units,
        "effective_win_rate": summary.effective_win_rate,
        "interval": summary.interval.to_dict(),
        "separated_from_even": summary.separated_from_even,
        "wins": summary.counts.wins,
        "draws": summary.counts.draws,
        "losses": summary.counts.losses,
        "mean_plies": summary.plies["mean"],
    }


def tuning_stage(units: int, workers: int) -> dict:
    """Re-derive the ablation that justifies the one policy change made here.

    The exposure term at 1.0.0 was priced by the material value of the
    identified piece. That is anti-correlated with the thing it was meant to
    price: a piece becomes identified by *winning* a fight, and the Marshal --
    the most valuable piece and the one the old rule taxed hardest -- is the
    piece the opponent can least answer. 1.1.0 prices the same term by
    `expected_defence_value`, the publicly deducible expected cost of being
    attacked by an unknown mover, which is negative only while the opponent still
    holds something that beats this type.
    """
    log(f"tuning: legacy vs current exposure rule, {units} paired units each")
    payloads = [("legacy", units), ("current", units)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(2, workers)) as pool:
            rows = list(pool.map(_ablation_job, payloads))
    else:
        rows = [_ablation_job(payload) for payload in payloads]

    by_variant = {row["variant"]: row for row in rows}
    legacy = by_variant["legacy"]
    current = by_variant["current"]
    vulnerability_table = _exposure_pricing_table()

    iterations = [
        {
            "iteration": 1,
            "change": "diagnosis only, no code change",
            "rationale": (
                "Agent 2 handed off `tactical_rule_based` vs `strategic_rule_based` as "
                "statistically inseparable at 0.542 over 192 units, and flagged `exposure` "
                "as the only Strategic term whose removal improved Strategic. Measured the "
                "term's pricing directly: it penalises a revealed Marshal 4x harder than a "
                "revealed Spy while the Marshal's expected cost of being attacked is "
                "positive and the Spy's is the most negative on the board."
            ),
            "result": "exposure pricing confirmed anti-correlated with vulnerability",
        },
        {
            "iteration": 2,
            "change": "re-price exposure by `expected_defence_value`, no weight moved",
            "rationale": (
                "Preserves the conceptual role -- an identified piece the opponent can "
                "answer is a target -- and fixes the scaling. Uses only the publicly "
                "deducible unresolved inventory, so the term stays invariant under "
                "`permute_hidden_identities`."
            ),
            "result": (
                f"Strategic vs Tactical {legacy['effective_win_rate']:.3f} -> "
                f"{current['effective_win_rate']:.3f} over {units} paired units"
            ),
        },
        {
            "iteration": 3,
            "change": "swept `pressure` x1.75/x2.5, `mobility` x1.5, `miner_preservation` x2, "
            "and dropping the Strategic scout-information bonus; none adopted",
            "rationale": (
                "Checked whether the load-bearing terms wanted rescaling on top of the "
                "exposure fix. At 256 paired units every variant landed inside the "
                "seed-noise band of the unmodified fix, so adopting one would have been "
                "fitting noise. Behaviour-identical replicas differing only in "
                "`policy_version` spanned 0.529-0.578 at that sample size, which is what "
                "established the band and why the decisive comparison was rerun at "
                f"{units} units."
            ),
            "result": "no further change adopted; one rule changed in total",
        },
    ]

    problems: list[str] = []
    if not current["separated_from_even"]:
        problems.append("the current Strategic does not separate from Tactical")
    if current["effective_win_rate"] <= legacy["effective_win_rate"]:
        problems.append("the exposure fix did not improve Strategic")

    return {
        "policy_changed": "strategic_rule_based",
        "version_before": "1.0.0",
        "version_after": policy_ref("strategic_rule_based").policy_version,
        "weights_changed": [],
        "rules_changed": ["exposure"],
        "paired_units": units,
        "ablation": rows,
        "improvement": current["effective_win_rate"] - legacy["effective_win_rate"],
        "exposure_pricing_table": vulnerability_table,
        "iterations": iterations,
        "iteration_count": len(iterations),
        "problems": problems,
        "passed": not problems,
    }


def _exposure_pricing_table() -> list[dict]:
    """The measurement that diagnosed the old exposure rule.

    Old penalty at `advance=1` beside the expected cost of being attacked by an
    unknown mover, for a mid-game position. The two run in opposite directions,
    which is the whole finding.
    """
    from stratego.engine.constants import PIECE_TYPE_NAMES
    from stratego.evaluation.calibration import sample_positions
    from stratego.evaluation.heuristics import build_context
    from stratego.evaluation.policy import build_policy_input

    policy = build_policy("strategic_rule_based")
    positions = sample_positions(11, source="random_walk")
    state = positions[len(positions) // 2]
    context = build_context(
        build_policy_input(
            state,
            policy=policy.ref,
            policy_seed=1,
            requirements=policy.requirements,
            suite_version=EVALUATION_SUITE_VERSION,
            match_id="exposure-probe",
            paired_unit_id="exposure-probe",
        )
    )
    rows = []
    for piece_type, name in enumerate(PIECE_TYPE_NAMES):
        rows.append(
            {
                "piece": name,
                "material_value": PIECE_VALUES[piece_type],
                "legacy_penalty_at_advance_1": -STRATEGIC_WEIGHTS.exposure
                * PIECE_VALUES[piece_type],
                "expected_defence_value": context.expected_defence_value(piece_type),
                "current_penalty_at_advance_1": (
                    STRATEGIC_WEIGHTS.exposure * context.expected_defence_value(piece_type)
                    if context.expected_defence_value(piece_type) < 0.0
                    else 0.0
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Stage 4 -- screening and calibration leagues
# ---------------------------------------------------------------------------


def build_calibration_schedule(
    ladder_units: int, ladder_vs_stress_units: int, stress_units: int
):
    """The full 45-pair league, sampled by matchup class.

    Every unordered pair of the ten catalogued policies appears; only the number
    of paired units differs, and the three classes are reported separately so the
    sampling is visible rather than implied.
    """
    ladder = set(LADDER_POLICY_IDS)
    schedules = []
    classes: dict[str, list[str]] = {"ladder": [], "ladder_vs_stress": [], "stress": []}
    identifiers = list(ALL_POLICY_IDS)
    for index, first in enumerate(identifiers):
        for second in identifiers[index + 1 :]:
            both = (first in ladder) + (second in ladder)
            if both == 2:
                units, label = ladder_units, "ladder"
            elif both == 1:
                units, label = ladder_vs_stress_units, "ladder_vs_stress"
            else:
                units, label = stress_units, "stress"
            schedules.append(
                build_matchup_schedule(first, second, list(range(units)), name=f"{label}")
            )
            classes[label].append(f"{policy_ref(first).token} vs {policy_ref(second).token}")
    schedule = merge_schedules(schedules, name="phase4_calibration")
    return schedule, classes


def screen_stage(units: int, workers: int) -> dict:
    """Short-schedule screen: broken policies, twins, draw loops, inversions."""
    log(f"screen: full 45-pair league at {units} paired units")
    bank = SetupBank.generate()
    schedule = build_league_schedule(
        list(ALL_POLICY_IDS), list(range(units)), name="phase4_screen"
    )
    require_valid_schedule(schedule, bank)
    started = time.perf_counter()
    run = run_schedule(schedule.matches, bank, worker_count=workers, record_actions=False)
    elapsed = time.perf_counter() - started
    summary = summarize_run(run.results, resamples=2000, seed=BOOTSTRAP_SEED)

    findings: list[str] = []
    twins: list[str] = []
    draw_loops: list[str] = []
    for matchup, entry in summary["per_matchup"].items():
        draw_share = entry["terminal_reasons"]["shares"].get("battleless_move_limit_draw", 0.0)
        if draw_share >= 0.9:
            draw_loops.append(matchup)
        interval = entry["confidence_interval"]
        if interval["width"] <= 0.02 and abs(entry["effective_win_rate"] - 0.5) < 0.02:
            twins.append(matchup)
    if run.policy_errors:
        findings.append(f"{run.policy_errors} policy errors")

    # A strength inversion in the ladder: a lower tier separating above a higher.
    inversions: list[str] = []
    order = {policy_ref(pid).token: rank for rank, pid in enumerate(LADDER_POLICY_IDS)}
    for matchup, entry in summary["per_matchup"].items():
        candidate, opponent = entry["candidate"], entry["opponent"]
        if candidate not in order or opponent not in order:
            continue
        stronger_is_candidate = order[candidate] > order[opponent]
        rate = entry["effective_win_rate"]
        if entry["separated_from_even"] and ((rate > 0.5) != stronger_is_candidate):
            inversions.append(f"{matchup} at {rate:.3f}")

    return {
        "paired_units": units,
        "matches": len(run.results),
        "wall_clock_seconds": elapsed,
        "matches_per_second": len(run.results) / elapsed,
        "mean_plies": run.plies / len(run.results),
        "policy_errors": run.policy_errors,
        "illegal_policy_actions": run.illegal_policy_actions,
        "results_digest": run.results_digest,
        "pathological_draw_matchups": draw_loops,
        "near_identical_matchups": twins,
        "ladder_inversions": inversions,
        "terminal_reasons": summary["terminal_reasons"],
        "per_matchup_effective_win_rates": {
            matchup: entry["effective_win_rate"]
            for matchup, entry in summary["per_matchup"].items()
        },
        "findings": findings,
        "passed": not findings,
    }


def calibration_stage(
    ladder_units: int,
    ladder_vs_stress_units: int,
    stress_units: int,
    workers: int,
    write_csv: bool,
) -> dict:
    """The final league: paired colour/setup evaluation with bootstrap intervals."""
    bank = SetupBank.generate()
    schedule, classes = build_calibration_schedule(
        ladder_units, ladder_vs_stress_units, stress_units
    )
    require_valid_schedule(schedule, bank)
    log(
        f"calibration: {len(schedule.matches):,} matches over {len(schedule.matchups)} matchups "
        f"({ladder_units}/{ladder_vs_stress_units}/{stress_units} units by class), {workers} workers"
    )
    started = time.perf_counter()
    run = run_schedule(schedule.matches, bank, worker_count=workers, record_actions=False)
    elapsed = time.perf_counter() - started
    log(f"calibration: ran in {elapsed:.0f}s at {len(run.results)/elapsed:.1f} matches/s")

    summary = summarize_run(
        run.results,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        include_setup_table=False,
    )
    csv_path = None
    if write_csv:
        csv_path = write_results_csv(LEAGUE_CSV, run.results)
        log(f"calibration: raw rows -> {csv_path.name} ({csv_path.stat().st_size/1e6:.1f} MB)")

    problems: list[str] = []
    if run.policy_errors:
        problems.append(f"{run.policy_errors} policy errors in the final league")
    if run.illegal_policy_actions:
        problems.append(f"{run.illegal_policy_actions} illegal policy actions")
    if summary["problems"]:
        problems.extend(summary["problems"])
    for matchup, entry in summary["per_matchup"].items():
        if entry["paired_units"] * 2 != entry["games"]:
            problems.append(f"{matchup} has incomplete paired units")

    return {
        "schedule_digest": run.schedule_digest,
        "results_digest": run.results_digest,
        "matchup_classes": classes,
        "units_by_class": {
            "ladder": ladder_units,
            "ladder_vs_stress": ladder_vs_stress_units,
            "stress": stress_units,
        },
        "matches": len(run.results),
        "paired_units": run.paired_units_run,
        "matchups": len(summary["per_matchup"]),
        "wall_clock_seconds": elapsed,
        "matches_per_second": len(run.results) / elapsed,
        "plies": summary["plies"],
        "terminal_reasons": summary["terminal_reasons"],
        "policy_errors": run.policy_errors,
        "illegal_policy_actions": run.illegal_policy_actions,
        "raw_csv": None if csv_path is None else csv_path.name,
        "summary": summary,
        "problems": problems,
        "passed": not problems,
    }


# ---------------------------------------------------------------------------
# Stage 5 -- strength tiers
# ---------------------------------------------------------------------------


def tier_stage(summary: dict) -> dict:
    """Partition the core ladder into statistically distinguishable tiers."""
    tokens = [policy_ref(pid).token for pid in LADDER_POLICY_IDS]
    tiers = strength_tiers(tokens, summary["per_matchup"])

    floor_token = policy_ref("random_legal").token
    floor_rank = tiers["membership"].get(floor_token)
    problems: list[str] = []
    if tiers["tier_count"] < 3:
        problems.append(f"only {tiers['tier_count']} strength tiers, at least 3 required")
    if floor_rank != tiers["tier_count"]:
        problems.append("random_legal is not the floor of the ladder")
    if tiers["missing_comparisons"]:
        problems.append(f"missing direct comparisons: {tiers['missing_comparisons']}")
    if not tiers["fully_ordered"]:
        problems.append(
            f"cross-tier pairs that do not separate: {tiers['unseparated_cross_tier_pairs']}"
        )

    ladder_table = []
    for first, second in ladder_pairs():
        first_token, second_token = policy_ref(first).token, policy_ref(second).token
        entry = summary["per_matchup"].get(f"{first_token} vs {second_token}") or summary[
            "per_matchup"
        ].get(f"{second_token} vs {first_token}")
        if entry is None:
            continue
        ladder_table.append(
            {
                "candidate": entry["candidate"],
                "opponent": entry["opponent"],
                "paired_units": entry["paired_units"],
                "effective_win_rate": entry["effective_win_rate"],
                "confidence_interval": entry["confidence_interval"],
                "normal_interval": entry["normal_interval"],
                "separated_from_even": entry["separated_from_even"],
                "wins": entry["wins"],
                "draws": entry["draws"],
                "losses": entry["losses"],
                "color_split": entry["color_split"],
                "paired_unit_score_histogram": entry["paired_unit_score_histogram"],
            }
        )

    tiers["ladder_comparisons"] = ladder_table
    tiers["random_is_floor"] = floor_rank == tiers["tier_count"]
    tiers["problems"] = problems
    tiers["passed"] = not problems
    log(f"tiers: {tiers['tier_count']} tiers, fully_ordered={tiers['fully_ordered']}")
    return tiers


# ---------------------------------------------------------------------------
# Stage 6 -- stress behaviour
# ---------------------------------------------------------------------------


def _behavior_job(payload: dict) -> dict[str, dict[str, int]]:
    """Play a matchup subset and profile both sides by replay."""
    first, second, units = payload["first"], payload["second"], payload["units"]
    bank = SetupBank.generate()
    schedule = build_matchup_schedule(first, second, list(range(units)))
    run = run_schedule(schedule.matches, bank, worker_count=1, record_actions=True)
    totals: dict[str, Counter] = {}
    for row in run.results:
        for token, counter in profile_replay(row).items():
            totals.setdefault(token, Counter()).update(counter)
    return {token: dict(counter) for token, counter in totals.items()}


def stress_stage(units: int, workers: int) -> dict:
    """Characterise every policy's behaviour over a subset of the final league."""
    identifiers = list(ALL_POLICY_IDS)
    payloads = [
        {"first": first, "second": second, "units": units}
        for index, first in enumerate(identifiers)
        for second in identifiers[index + 1 :]
    ]
    log(f"behaviour: {len(payloads)} matchups x {units} paired units, profiled by replay")
    started = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(_behavior_job, payloads))
    else:
        chunks = [_behavior_job(payload) for payload in payloads]
    elapsed = time.perf_counter() - started

    pooled: dict[str, Counter] = {}
    for chunk in chunks:
        for token, counter in chunk.items():
            pooled.setdefault(token, Counter()).update(counter)

    profiles = {token: summarise_behavior(counter) for token, counter in pooled.items()}
    reference_token = policy_ref("strategic_rule_based").token
    reference = profiles[reference_token]
    divergences = {
        token: behavior_divergence(profile, reference)
        for token, profile in profiles.items()
        if token != reference_token
    }
    stress_tokens = [policy_ref(pid).token for pid in STRESS_POLICY_IDS]
    materially_different = [
        token for token in stress_tokens if divergences[token]["materially_different"]
    ]

    problems: list[str] = []
    if len(materially_different) < 4:
        problems.append(
            f"only {len(materially_different)} stress policies differ materially from "
            "the Strategic baseline"
        )
    missing = [token for token in stress_tokens if token not in profiles]
    if missing:
        problems.append(f"no behavioural profile for {missing}")

    log(
        f"behaviour: {len(materially_different)}/{len(stress_tokens)} stress policies "
        f"materially different in {elapsed:.0f}s"
    )
    return {
        "paired_units_per_matchup": units,
        "matchups": len(payloads),
        "wall_clock_seconds": elapsed,
        "reference": reference_token,
        "metrics_compared": list(STRESS_COMPARISON_METRICS),
        "profiles": profiles,
        "divergence_from_strategic": divergences,
        "stress_policies_materially_different": materially_different,
        "problems": problems,
        "passed": not problems,
    }


# ---------------------------------------------------------------------------
# Stage 7 -- final reproducibility
# ---------------------------------------------------------------------------


def reproducibility_stage(units: int, worker_counts: "tuple[int, ...]") -> dict:
    """Rerun a league subset serially, in parallel, and with the order shuffled."""
    bank = SetupBank.generate()
    subset = merge_schedules(
        [
            build_matchup_schedule(first, second, list(range(units)))
            for first, second in (
                ("random_legal", "basic_heuristic"),
                ("tactical_rule_based", "strategic_rule_based"),
                ("strategic_rule_based", "stress_information_miser"),
                ("stress_chaos", "stress_draw_seeker"),
            )
        ],
        name="phase4_reproducibility",
    )
    require_valid_schedule(subset, bank)
    log(f"reproducibility: {len(subset.matches)} matches at workers {worker_counts} plus shuffled")

    executions: list[dict] = []
    baseline: "tuple" = ()
    digests: set[str] = set()
    mismatches = 0
    compared = 0
    for worker_count in worker_counts:
        run = run_schedule(
            subset.matches, bank, worker_count=worker_count, record_actions=True
        )
        digests.add(run.results_digest)
        if not baseline:
            baseline = run.results
        else:
            problems = compare_results(baseline, run.results)
            mismatches += len(problems)
            compared += len(run.results)
        executions.append(
            {
                "label": f"workers={worker_count}",
                "worker_count": worker_count,
                "chunk_count": run.chunk_count,
                "matches": len(run.results),
                "seconds": run.wall_clock_seconds,
                "results_digest": run.results_digest,
                "replay_digest_of_digests": _digest_of_replays(run.results),
            }
        )

    shuffled = list(subset.matches)
    random.Random(4242).shuffle(shuffled)
    run = run_schedule(shuffled, bank, worker_count=max(worker_counts), record_actions=True)
    digests.add(run.results_digest)
    problems = compare_results(baseline, run.results)
    mismatches += len(problems)
    compared += len(run.results)
    executions.append(
        {
            "label": "shuffled",
            "worker_count": max(worker_counts),
            "chunk_count": run.chunk_count,
            "matches": len(run.results),
            "seconds": run.wall_clock_seconds,
            "results_digest": run.results_digest,
            "replay_digest_of_digests": _digest_of_replays(run.results),
        }
    )

    # Checked separately from `compare_results`, which would also catch a
    # divergence: the instructions ask for identical per-match results *and*
    # replay digests, so the replay digests get their own one-line statement
    # rather than being implied by a field-level comparison passing.
    replay_digests = {execution["replay_digest_of_digests"] for execution in executions}

    stage_problems: list[str] = []
    if len(digests) != 1:
        stage_problems.append(f"{len(digests)} distinct results digests across executions")
    if len(replay_digests) != 1:
        stage_problems.append(f"{len(replay_digests)} distinct replay-digest sets across executions")
    if mismatches:
        stage_problems.append(f"{mismatches} per-match mismatches")

    return {
        "matches_per_execution": len(subset.matches),
        "executions": executions,
        "distinct_results_digests": sorted(digests),
        "distinct_replay_digest_sets": sorted(replay_digests),
        "rerun_matches": compared,
        "mismatches": mismatches,
        "problems": stage_problems,
        "passed": not stage_problems,
    }


def _digest_of_replays(results) -> str:
    from hashlib import sha256

    digest = sha256()
    for row in sorted(results, key=lambda item: item.match_id):
        digest.update(row.match_id.encode("utf-8"))
        digest.update(row.replay_digest.encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Stage 8 -- checkpoint readiness
# ---------------------------------------------------------------------------


class CheckpointShapedPolicy(Policy):
    """A checkpoint-shaped policy: reads the 127-channel tensor, masks, argmaxes.

    Deliberately **not** a Stratego opponent and deliberately not catalogued. Its
    only job is to prove the path a future neural checkpoint takes actually works
    end to end: declare `observation=True` and `legal_action_mask=True`, receive
    the `observation_v2_1_127ch` tensor, project it to a score per action
    identifier, mask the illegal ones and take the best. That is exactly the
    shape of a trained network's `decide`, with a fixed seeded projection in
    place of learned weights, so the harness is exercised without any of the
    neural-network work Phase 4 excludes.

    The projection is a fixed function of the tensor, so this policy is fully
    deterministic and its games reproduce like any other.
    """

    policy_id = "probe_observation_checkpoint"
    policy_version = "1.0.0"
    requirements = PolicyRequirements(
        observation=True, legal_action_mask=True, public_view=False
    )
    stochastic = False
    description = "Checkpoint-shaped observation consumer; harness probe, not an opponent."

    def __init__(self) -> None:
        generator = np.random.default_rng(20260409)
        # A fixed random projection from a pooled observation to action scores.
        self._projection = generator.standard_normal((127, 64), dtype=np.float64)
        self._head = generator.standard_normal((64, ACTION_SPACE_SIZE), dtype=np.float64)

    def decide(self, request: PolicyInput) -> PolicyResult:
        observation = request.require_observation()
        mask = request.require_legal_action_mask()
        # Channel means: the cheapest pooling that still reads every plane.
        pooled = observation.reshape(observation.shape[0], -1).mean(axis=1)
        scores = np.tanh(pooled @ self._projection) @ self._head
        scores = np.where(mask.astype(bool), scores, -np.inf)
        action_id = int(np.argmax(scores))
        return self.result(
            request,
            action_id,
            {"family": "observation_argmax", "score": float(scores[action_id])},
        )


def checkpoint_stage(units: int) -> dict:
    """Drive the checkpoint-shaped probe through the gauntlet path.

    Serial by necessity: the parallel path resolves policies through the
    catalogue and the probe is not in it. That constraint is itself worth
    recording -- a real checkpoint evaluation must either register its policy or
    run serially.
    """
    probe = CheckpointShapedPolicy()
    opponents = list(LADDER_POLICY_IDS)
    log(f"checkpoint: probe gauntlet against {len(opponents)} ladder policies, {units} units")
    bank = SetupBank.generate()

    rows = []
    illegal = 0
    reproduction_mismatches = 0
    for opponent_id in opponents:
        opponent = build_policy(opponent_id)
        matches = schedule_matches(
            build_paired_schedule(probe.ref, opponent.ref, list(range(units)))
        )
        policies = {probe.ref.token: probe, opponent.ref.token: opponent}
        run = run_schedule(matches, bank, policies=policies, worker_count=1, record_actions=True)
        illegal += run.illegal_policy_actions
        summary = summarize_matchup(run.results, resamples=2000, seed=BOOTSTRAP_SEED)
        # Rerun a paired unit from its identity alone; a checkpoint evaluation
        # must reproduce exactly like a baseline one. `run.results` is sorted by
        # `match_id`, so the originals are selected by identity rather than by
        # schedule position.
        repeated = matches[:2]
        by_id = {row.match_id: row for row in run.results}
        again = run_schedule(
            repeated, bank, policies=policies, worker_count=1, record_actions=True
        )
        reproduction_mismatches += len(
            compare_results([by_id[spec.match_id] for spec in repeated], again.results)
        )
        rows.append(
            {
                "opponent": opponent.ref.token,
                "paired_units": summary.paired_units,
                "effective_win_rate": summary.effective_win_rate,
                "confidence_interval": summary.interval.to_dict(),
                "mean_plies": summary.plies["mean"],
                "wins": summary.counts.wins,
                "draws": summary.counts.draws,
                "losses": summary.counts.losses,
            }
        )

    problems: list[str] = []
    if illegal:
        problems.append(f"{illegal} illegal actions from the observation probe")
    if reproduction_mismatches:
        problems.append(f"{reproduction_mismatches} reproduction mismatches for the probe")

    return {
        "probe": probe.ref.token,
        "requirements": probe.requirements.to_dict(),
        "observation_version": OBSERVATION_VERSION,
        "paired_units_per_opponent": units,
        "gauntlet": rows,
        "illegal_actions": illegal,
        "reproduction_mismatches": reproduction_mismatches,
        "note": (
            "A checkpoint-shaped policy consuming the 127-channel observation plays "
            "the ladder through the same schedule, runner, statistics and reproduction "
            "path as a baseline. Parallel execution additionally requires the policy to "
            "be registered, because workers resolve policies through the catalogue."
        ),
        "problems": problems,
        "passed": not problems,
    }


# ---------------------------------------------------------------------------
# Stage 9 -- the test suite
# ---------------------------------------------------------------------------


def pytest_stage(target: str = "") -> dict:
    command = [sys.executable, "-m", "pytest", "-q"]
    if target:
        command.append(target)
    log(f"pytest: {' '.join(command[2:])}")
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False
    )
    elapsed = time.perf_counter() - started
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""

    def count(label: str) -> int:
        match = re.search(rf"(\d+) {label}", tail)
        return int(match.group(1)) if match else 0

    passed, failed = count("passed"), count("failed")
    errors, skipped = count("error"), count("skipped")
    return {
        "command": " ".join(command),
        "summary_line": tail,
        "returncode": completed.returncode,
        "total": passed + failed + errors + skipped,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "seconds": elapsed,
        "green": completed.returncode == 0 and failed == 0 and errors == 0,
    }


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------


def write_behavior_csv(path: Path, profiles: dict, divergences: dict) -> Path:
    import csv

    metrics = [
        "games",
        "moves",
        "mean_game_plies",
        "attack_rate",
        "scout_move_rate",
        "scout_run_rate",
        "miner_move_rate",
        "miner_attack_rate",
        "mean_move_distance",
        "piece_type_entropy_bits",
        "movement_entropy_bits",
        "own_reveal_rate",
        "draw_rate",
        "battleless_draw_rate",
        "flag_capture_win_rate",
        "effective_win_rate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["policy", "role", *metrics, "metrics_beyond_threshold", "largest_relative_difference"]
        )
        for token in sorted(profiles):
            profile = profiles[token]
            role = "stress" if token.split("@")[0] in STRESS_POLICY_IDS else "ladder"
            divergence = divergences.get(token, {})
            writer.writerow(
                [
                    token,
                    role,
                    *[profile[metric] for metric in metrics],
                    "|".join(divergence.get("metrics_beyond_threshold", [])),
                    divergence.get("largest_relative_difference", 0.0),
                ]
            )
    return path


def write_pairwise_csv(path: Path, summary: dict) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "candidate",
                "opponent",
                "paired_units",
                "games",
                "wins",
                "draws",
                "losses",
                "effective_win_rate",
                "ci_lower",
                "ci_upper",
                "ci_width",
                "separated_from_even",
                "normal_lower",
                "normal_upper",
                "mean_plies",
                "ewr_as_red",
                "ewr_as_blue",
            ]
        )
        for matchup in sorted(summary["per_matchup"]):
            entry = summary["per_matchup"][matchup]
            interval = entry["confidence_interval"]
            normal = entry["normal_interval"] or {}
            colors = entry["color_split"]
            writer.writerow(
                [
                    entry["candidate"],
                    entry["opponent"],
                    entry["paired_units"],
                    entry["games"],
                    entry["wins"],
                    entry["draws"],
                    entry["losses"],
                    entry["effective_win_rate"],
                    interval["lower"],
                    interval["upper"],
                    interval["width"],
                    entry["separated_from_even"],
                    normal.get("lower"),
                    normal.get("upper"),
                    entry["plies"]["mean"],
                    colors.get("red", {}).get("effective_win_rate"),
                    colors.get("blue", {}).get("effective_win_rate"),
                ]
            )
    return path


def environment_record() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
    }


def version_record() -> dict:
    return {
        "implementation": IMPLEMENTATION_VERSION,
        "rules": RULES_VERSION,
        "observation": OBSERVATION_VERSION,
        "policy_interface": POLICY_INTERFACE_VERSION,
        "match_spec": MATCH_SPEC_VERSION,
        "setup_bank": SETUP_BANK_VERSION,
        "evaluation_suite": EVALUATION_SUITE_VERSION,
        "baseline_suite": BASELINE_SUITE_VERSION,
        "heuristics": HEURISTICS_VERSION,
        "match_runner": MATCH_RUNNER_VERSION,
        "match_result_schema": MATCH_RESULT_SCHEMA_VERSION,
        "scheduler": SCHEDULER_VERSION,
        "statistics": STATISTICS_VERSION,
        "reporting": REPORTING_VERSION,
        "calibration": CALIBRATION_VERSION,
        "rules_context": EVALUATION_RULES.context,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fast smoke run; reports FAIL")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--audit-trials", type=int, default=AUDIT_TRIAL_TARGET)
    parser.add_argument("--ladder-units", type=int, default=LADDER_UNITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage",
        action="append",
        default=None,
        choices=[
            "prerequisites",
            "audit",
            "tuning",
            "screen",
            "calibration",
            "behavior",
            "reproducibility",
            "checkpoint",
        ],
        help="run only these stages (repeatable); skips the JSON write",
    )
    arguments = parser.parse_args()

    quick = arguments.quick
    audit_trials = 2_000 if quick else arguments.audit_trials
    ladder_units = 64 if quick else arguments.ladder_units
    vs_stress_units = 32 if quick else LADDER_VS_STRESS_UNITS
    stress_units = 16 if quick else STRESS_UNITS
    behavior_units = 8 if quick else BEHAVIOR_UNITS
    tuning_units = 64 if quick else 1024
    screen_units = 16 if quick else 64
    reproducibility_units = 8 if quick else REPRODUCIBILITY_UNITS
    worker_counts = (1, 2) if quick else REPRODUCIBILITY_WORKERS
    checkpoint_units = 8 if quick else 64
    workers = max(1, arguments.workers)

    selected = set(arguments.stage) if arguments.stage else None

    def wanted(name: str) -> bool:
        return selected is None or name in selected

    started = time.perf_counter()
    log(f"start (quick={quick}, workers={workers})")

    stages: dict[str, dict] = {}
    if wanted("prerequisites"):
        stages["prerequisites"] = prerequisite_stage()
        if not stages["prerequisites"]["passed"] and selected is None:
            log(f"BLOCKED: {stages['prerequisites']['problems']}")
            payload = {
                "agent": "agent_04",
                "status": "BLOCKED",
                "prerequisites": stages["prerequisites"],
                "versions": version_record(),
                "environment": environment_record(),
            }
            write_json(arguments.output, payload)
            return 2

    if wanted("audit"):
        stages["audit"] = audit_stage(audit_trials, workers)
        write_json(AUDIT_JSON, stages["audit"])
    if wanted("tuning"):
        stages["tuning"] = tuning_stage(tuning_units, workers)
    if wanted("screen"):
        stages["screen"] = screen_stage(screen_units, workers)
    if wanted("calibration"):
        stages["calibration"] = calibration_stage(
            ladder_units, vs_stress_units, stress_units, workers, write_csv=not quick
        )
        stages["tiers"] = tier_stage(stages["calibration"]["summary"])
        write_pairwise_csv(TIER_CSV, stages["calibration"]["summary"])
    if wanted("behavior"):
        stages["behavior"] = stress_stage(behavior_units, workers)
        write_behavior_csv(
            BEHAVIOR_CSV,
            stages["behavior"]["profiles"],
            stages["behavior"]["divergence_from_strategic"],
        )
    if wanted("reproducibility"):
        stages["reproducibility"] = reproducibility_stage(reproducibility_units, worker_counts)
    if wanted("checkpoint"):
        stages["checkpoint"] = checkpoint_stage(checkpoint_units)

    tests = (
        {"skipped": True, "green": False, "total": 0, "passed": 0, "failed": 0, "errors": 0}
        if arguments.skip_pytest
        else pytest_stage()
    )

    if selected is not None:
        log(f"stage selection {sorted(selected)} complete; JSON not written")
        for name, stage in stages.items():
            log(f"  {name}: passed={stage.get('passed')} problems={stage.get('problems')}")
        return 0

    calibration = stages["calibration"]
    summary = calibration["summary"]
    tiers = stages["tiers"]
    audit = stages["audit"]
    behavior = stages["behavior"]
    reproducibility = stages["reproducibility"]
    checkpoint = stages["checkpoint"]
    tuning = stages["tuning"]

    gates = {
        "all_prior_agents_passed": stages["prerequisites"]["passed"],
        "hidden_information_trials_met": audit["trials"] >= AUDIT_TRIAL_TARGET,
        "hidden_information_zero_mismatches": audit["total_mismatches"] == 0,
        "positive_controls_verified": (
            audit["positive_control_failures"] == 0 and audit["positive_control_trials"] > 0
        ),
        "leak_detector_verified": audit["leak_detector_failures"] == 0,
        "every_policy_legal": (
            calibration["illegal_policy_actions"] == 0 and calibration["policy_errors"] == 0
        ),
        "league_reproducible": reproducibility["passed"],
        "paired_confidence_intervals_work": all(
            entry["confidence_interval"]["resampling_unit"] == "paired_unit"
            and entry["confidence_interval"]["sample_size"] == entry["paired_units"]
            for entry in summary["per_matchup"].values()
        ),
        "three_or_more_strength_tiers": tiers["tier_count"] >= 3,
        # The instructions name these two explicitly: Random must function as the
        # floor, and stronger baselines must not collapse into statistically
        # indistinguishable copies of one another.
        "random_is_the_ladder_floor": tiers["random_is_floor"],
        "ladder_tiers_fully_ordered": tiers["fully_ordered"],
        "strategic_separates_from_tactical": tuning["passed"],
        "screen_found_no_broken_policy": stages["screen"]["passed"],
        "stress_policies_distinct": behavior["passed"],
        "raw_results_permit_reproduction": calibration["raw_csv"] is not None,
        "checkpoint_harness_ready": checkpoint["passed"],
        "test_suite_green": bool(tests.get("green")),
    }
    if quick:
        gates["not_a_quick_run"] = False

    decision = "PASS" if all(gates.values()) else "FAIL"
    elapsed = time.perf_counter() - started

    payload = {
        "agent": "agent_04",
        "status": decision,
        "phase4_decision": decision,
        "quick": quick,
        "wall_clock_seconds": elapsed,
        "versions": version_record(),
        "environment": environment_record(),
        "policy_catalog": policy_catalog(),
        "prerequisites": stages["prerequisites"],
        # -- audit ----------------------------------------------------------
        "hidden_information_trials": audit["trials"],
        "hidden_information_mismatches": audit["total_mismatches"],
        "hidden_information_policy_comparisons": audit["policy_comparisons"],
        "positive_control_trials": audit["positive_control_trials"],
        "positive_control_failures": audit["positive_control_failures"],
        "leak_detector_failures": audit["leak_detector_failures"],
        "policies_audited": audit["policies_audited"],
        "hidden_information_audit": audit,
        # -- league ---------------------------------------------------------
        "league_match_count": calibration["matches"],
        "league_paired_unit_count": calibration["paired_units"],
        "league_matchup_count": calibration["matchups"],
        "league": calibration,
        "screen": stages["screen"],
        "core_policy_results": {
            token: summary["per_opponent"].get(token)
            for token in (policy_ref(pid).token for pid in LADDER_POLICY_IDS)
        },
        "pairwise_effective_win_rates": {
            matchup: entry["effective_win_rate"]
            for matchup, entry in summary["per_matchup"].items()
        },
        "pairwise_confidence_intervals": {
            matchup: entry["confidence_interval"]
            for matchup, entry in summary["per_matchup"].items()
        },
        "league_ratings": summary.get("league"),
        # -- tiers ----------------------------------------------------------
        "strength_tier_count": tiers["tier_count"],
        "strength_tier_membership": tiers["membership"],
        "strength_tiers": tiers,
        # -- stress ---------------------------------------------------------
        "stress_behavior_metrics": behavior["profiles"],
        "stress_behavior": behavior,
        # -- reproducibility ------------------------------------------------
        "reproducibility_rerun_matches": reproducibility["rerun_matches"],
        "reproducibility_mismatches": reproducibility["mismatches"],
        "reproducibility": reproducibility,
        # -- tuning ---------------------------------------------------------
        "policy_tuning_iterations": tuning["iteration_count"],
        "policy_tuning": tuning,
        # -- checkpoint readiness -------------------------------------------
        "evaluation_harness_ready_for_future_checkpoints": checkpoint["passed"],
        "checkpoint_readiness": checkpoint,
        # -- tests ----------------------------------------------------------
        "test_total": tests.get("total", 0),
        "test_passed": tests.get("passed", 0),
        "test_failed": tests.get("failed", 0) + tests.get("errors", 0),
        "tests": tests,
        "completion_gates": gates,
        "files_created": [
            "stratego/evaluation/calibration.py",
            "scripts/run_phase4_agent04.py",
            "tests/evaluation/test_phase4_acceptance.py",
            DEFAULT_OUTPUT.name,
            AUDIT_JSON.name,
            LEAGUE_CSV.name,
            BEHAVIOR_CSV.name,
            TIER_CSV.name,
        ],
        "files_modified": [
            "stratego/evaluation/baselines.py",
            "stratego/evaluation/__init__.py",
            "tests/evaluation/test_match_runner.py",
            "reports/phase_4_implementation_report.md",
        ],
    }
    write_json(arguments.output, payload)

    # The suite above ran *before* this file existed in its final form, so
    # `test_the_published_acceptance_record_is_internally_consistent` could only
    # skip. Re-run the acceptance module now that the artefact is on disk, so the
    # published record is checked against the shipped assertions rather than
    # trusted. A failure here downgrades the decision instead of being reported
    # as a green run with a broken data file.
    if not arguments.skip_pytest:
        verification = pytest_stage("tests/evaluation/test_phase4_acceptance.py")
        payload["artefact_verification"] = verification
        gates["published_record_verified"] = bool(verification["green"])
        if not verification["green"]:
            decision = "FAIL"
            payload["status"] = decision
            payload["phase4_decision"] = decision
        payload["completion_gates"] = gates
        write_json(arguments.output, payload)

    log("")
    log(f"Phase 4 decision: {decision}")
    log(f"Baseline tiers established: {tiers['tier_count']}")
    log(f"Hidden-information audit trials: {audit['trials']:,}")
    log(f"Hidden-information mismatches: {audit['total_mismatches']}")
    log(
        "Evaluation harness ready for future checkpoints: "
        f"{'yes' if checkpoint['passed'] else 'no'}"
    )
    for name, value in gates.items():
        log(f"  {'ok ' if value else 'FAIL'} {name}")
    log(f"total {elapsed:.0f}s -> {arguments.output}")
    return 0 if decision == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
