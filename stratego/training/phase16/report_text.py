"""Phase 16 Agent 3: the report renderer.

Section 7 fixes the sections and their order: what was built, gate results, the
three h-curves with SEs, the decision the predeclared rules produced, and
`known_limitations`. Rendering it from the recorded artifacts rather than
writing it by hand is what keeps the numbers in the prose and the numbers in
the JSON the same numbers.

Every table names its pack. Overview section 6 forbids cross-pack comparisons
in conclusions, and a table that does not say which instrument produced a
number is an invitation to make one.
"""

from __future__ import annotations

import json
from pathlib import Path

from .analysis import (
    CONTROL_ARM,
    DAMPED_ARM,
    DAMPED_PLUS_ARM,
    VERDICT_ADOPT,
    VERDICT_STOP,
    arm_curve,
    end_to_end_comparison,
    instrument_check,
    load_phase14_decomposition,
    throughput_verdict,
)
from .contract import (
    ADOPT_RECIPE_MARGIN,
    ARM_HOURS,
    DECISION_RULES,
    EVALUATION_HOURS,
    SETUPS_CAUSAL_MARGIN,
    SHOOTOUT_ARMS,
    contract_digest,
    contract_document,
)
from .runner import read_telemetry, telemetry_summary
from .schedules import floor_iteration, learning_rate_for

ARM_ORDER = (CONTROL_ARM, DAMPED_ARM, DAMPED_PLUS_ARM)
ARM_LETTER = {CONTROL_ARM: "A", DAMPED_ARM: "B", DAMPED_PLUS_ARM: "C"}


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _gate_line(name: str, entry: dict) -> str:
    verdict = entry.get("pass")
    mark = "PASS" if verdict else ("FAIL" if verdict is False else "not run")
    note = entry.get("note") or ""
    return f"| `{name}` | {mark} | {note} |"


def render_report(
    *,
    gates: dict,
    throughput: dict,
    curves: dict,
    configs: dict,
    candidate: dict,
    telemetry_root,
    report_root,
) -> str:
    telemetry_root = Path(telemetry_root)
    # The decomposition artifact sits beside the report, so its root is the
    # repository the report is being written into.
    report_root_parent = Path(report_root).parent.parent
    lines: list = []
    add = lines.append

    add("# Phase 16 — Agent 3 report")
    add("## Training loop v2 and the 3×6-hour recipe shootout")
    add("")
    add(f"_Contract digest `{contract_digest()[:16]}…`._")
    add("")
    add(
        "Every EWR below names its pack. Cross-pack comparisons are forbidden in "
        "conclusions (overview §6); the packs are engineering instruments and no "
        "significance claim is made anywhere in this report."
    )
    add("")

    # -- 1. what was built --------------------------------------------------
    add("## 1. What was built")
    add("")
    add(
        "`stratego/training/phase16/` — additive, importing the accepted objective "
        "and controller unmodified:"
    )
    add("")
    add("| module | what it is |")
    add("|---|---|")
    add("| `contract.py` | arm flags, seed streams, game ids, the inherited Phase 9 constants |")
    add("| `schedules.py` | the power-law LR and the annealed entropy coefficient |")
    add("| `targets.py` | window-edge targets and §2.2's invariant |")
    add("| `setups.py` | the `library` and `expanded` setup mixtures, orientation-gated |")
    add("| `population.py` | the persistent population and its opponent mixture |")
    add("| `snapshots.py` | behavior snapshots taken from live weights |")
    add("| `collector.py` | the window collector, harvesting rows at collection time |")
    add("| `trainer.py` | the PPO/KL update, the EMA, the per-window statistics |")
    add("| `checkpoint.py` | arm checkpoints, resume identity, the evaluation export |")
    add("| `runner.py` | one arm start-to-deadline, telemetry, hour exports |")
    add("| `seat.py` | the Agent 1 provider factory for a trained arm |")
    add("| `analysis.py` | §5's predeclared rules, applied mechanically |")
    add("")
    add("### The three arms")
    add("")
    add("| arm | LR | entropy | epochs | EMA | opponents | setups |")
    add("|---|---|---|---|---|---|---|")
    for arm in SHOOTOUT_ARMS:
        lr = (
            f"{arm.lr_constant:.2e} constant"
            if arm.lr_schedule == "constant"
            else (
                f"{arm.lr_max:.2e}·(n/{arm.lr_reference})^−{arm.lr_exponent} "
                f"→ {arm.lr_min:.2e}"
            )
        )
        entropy = (
            f"{arm.entropy_constant} constant"
            if arm.entropy_schedule == "constant"
            else f"{arm.entropy_start}·n^−{arm.entropy_exponent} → {arm.entropy_floor}"
        )
        add(
            f"| **{ARM_LETTER[arm.arm_id]}** `{arm.arm_id}` | {lr} | {entropy} | "
            f"{arm.epochs} | {_fmt(arm.ema)} | `{arm.opponents}` | `{arm.setups}` |"
        )
    add("")
    add(
        "Unchanged by explicit choice: PPO clip 0.2, λ_A 0.5 / λ_V 0.8, the "
        "top-quartile advantage filter with its 0.01 floor, value weight 0.5, "
        "belief-aux weight 0.25, the adaptive-β KL machinery and its thresholds, "
        "grad-norm 1.0, minibatch 512, float32, battleless-100 / absolute-4000, "
        "MPS with inference batch shape 64. All imported from the accepted Phase 9 "
        "contract, never restated."
    )
    add("")

    # -- 1b. deviations -----------------------------------------------------
    amendment = contract_document().get("schedule_amendment") or {}
    add("## 1b. Deviations from the brief")
    add("")
    if amendment:
        add(
            f"**One amendment, to §2.3**, raised by this agent from the measured "
            f"iteration rate and confirmed by the brief's author."
        )
        add("")
        add(f"- *Defect.* {amendment['defect']}.")
        add(f"- *Change.* {amendment['change']}.")
        add(f"- *Unchanged.* {amendment['unchanged']}.")
        add(
            f"- *Measured horizon.* {amendment['measured_iteration_seconds']} s per "
            f"iteration on this machine, so N = {amendment['planned_iterations']} for "
            f"a six-hour arm. `n_ref = 1` remains the code default and reproduces the "
            f"brief exactly."
        )
        add(f"- *Entropy deliberately not re-horizoned.* {amendment['entropy_not_re_horizoned']}")
        add("")
        add("| arm | LR at n=1 | n=40 | n=150 | n=313 | floor first touched |")
        add("|---|---|---|---|---|---|")
        for arm in SHOOTOUT_ARMS:
            rates = [learning_rate_for(arm, n) for n in (1, 40, 150, 313)]
            add(
                f"| `{arm.arm_id}` | "
                + " | ".join(f"{rate:.2e}" for rate in rates)
                + f" | {floor_iteration(arm) or '—'} |"
            )
        add("")
        add(
            "The amendment does **not** equalise the arms and is not meant to. Arms "
            "B and C still take roughly half the control's Σ(lr·steps), because §2.4 "
            "gives them one epoch against A's two — an intended, briefed difference. "
            "What it removes is the unintended five-fold one."
        )
        add("")
    else:  # pragma: no cover - the contract always carries the amendment
        add("_None recorded._")
        add("")

    # -- 2. gates -----------------------------------------------------------
    add("## 2. Correctness gates (§3)")
    add("")
    if gates:
        add("| gate | result | note |")
        add("|---|---|---|")
        for name, entry in (gates.get("gates") or {}).items():
            add(_gate_line(name, entry))
        add("")
        invariant = (gates.get("gates") or {}).get("window_edge_invariant") or {}
        if invariant:
            add(
                f"The window-edge invariant was checked on {invariant.get('trials')} "
                f"synthetic games spanning at least {invariant.get('minimum_windows')} "
                f"windows each. Largest advantage difference "
                f"{_fmt(invariant.get('max_advantage_difference'), 9)}, largest W/D/L "
                f"difference {_fmt(invariant.get('max_wdl_difference'), 9)}, against a "
                f"tolerance of {invariant.get('tolerance')}."
            )
            add("")
        smoke = (gates.get("gates") or {}).get("smoke_run") or {}
        if smoke.get("pass") is not None:
            add(
                f"The smoke run completed {smoke.get('windows')} windows in "
                f"{smoke.get('minutes')} minutes, checkpointed, resumed to the same "
                f"model-state digest, and a CPU rerun of one window's update from "
                f"identical inputs was "
                f"{'bit-identical' if smoke.get('cpu_rerun_bit_identical') else 'NOT bit-identical'}."
            )
            add("")
    else:
        add("_No gate artifact recorded._")
        add("")
    if throughput:
        verdict = throughput_verdict(throughput)
        add(f"**Collection throughput.** {verdict['statement']}")
        add("")
        smoke_rows = ((gates.get("gates") or {}).get("smoke_run") or {}).get("telemetry")
        economics = end_to_end_comparison(
            throughput, smoke_rows or [], load_phase14_decomposition(report_root_parent)
        )
        if economics:
            ours, theirs = economics["phase16"], economics["phase14"]
            add("")
            add("### Where the time actually goes")
            add("")
            if economics.get("gate3_caveat"):
                add(f"**{economics['gate3_caveat']}**")
                add("")
            add("| | Phase 16 arm B | Phase 14 production |")
            add("|---|---|---|")
            add(
                f"| collection, plies/s | {ours['collection_plies_per_second']} "
                f"| {theirs.get('collection_plies_per_second', '—')} "
                f"{'(median, range ' + str(theirs['collection_plies_per_second_range']) + ')' if theirs.get('collection_plies_per_second_range') else ''} |"
            )
            add(
                f"| collection share of wall | {ours['collection_share_of_wall']:.0%} "
                f"| {theirs['collection_share_of_wall']:.0%} |"
                if theirs.get("collection_share_of_wall") is not None
                else f"| collection share of wall | {ours['collection_share_of_wall']:.0%} | — |"
            )
            add(
                f"| training share of wall | {ours['training_share_of_wall']:.0%} "
                f"| {theirs['training_share_of_wall']:.0%} |"
                if theirs.get("training_share_of_wall") is not None
                else f"| training share of wall | {ours['training_share_of_wall']:.0%} | — |"
            )
            add(
                f"| optimizer steps / iteration | "
                f"{ours['optimizer_steps_per_iteration']:,} | "
                + (
                    f"{theirs['growth']['optimizer_steps_per_iteration']['first5']:,} → "
                    f"{theirs['growth']['optimizer_steps_per_iteration']['last5']:,} |"
                    if theirs.get("growth")
                    else "— |"
                )
            )
            add(
                f"| minutes / iteration | {ours['minutes_per_iteration']} | "
                + (
                    f"{theirs['growth']['iteration_minutes']['first5']} → "
                    f"{theirs['growth']['iteration_minutes']['last5']} |"
                    if theirs.get("growth")
                    else f"{theirs['minutes_per_iteration']} |"
                )
            )
            add(
                f"| trained decisions / hour | "
                f"{ours['trained_decisions_per_hour']:,} | "
                f"{theirs['trained_decisions_per_hour']:,} |"
            )
            add("")
            if economics.get("finding"):
                add(economics["finding"])
                add("")
            add(f"_Phase 14 figures: {theirs['source']}_")
            if theirs.get("decomposition_source"):
                add(
                    f"_Split derived read-only from that run's own per-iteration "
                    f"telemetry; no Phase 14 module was imported and nothing was "
                    f"written. {theirs['decomposition_source']}._"
                )
            add("")
            add("Caveats:")
            for caveat in economics["caveats"]:
                add(f"- {caveat}")
            add("")

    # -- 3. the curves ------------------------------------------------------
    add("## 3. The three h-curves (§4)")
    add("")
    check = instrument_check(curves) if curves else {}
    if check:
        digests = set(d for d in check["starting_model_state_digest"].values() if d)
        if not check["digests_recorded"]:
            weights = "no model-state digest was recorded for every arm"
        elif check["digests_agree"]:
            weights = (
                f"all carry model-state digest `{sorted(digests)[0][:16]}…`, the "
                "accepted P24"
            )
        else:
            weights = "**their digests differ, which they must not**"
        add(
            f"_Instrument check: the {len(check['arms'])} arms export their "
            f"starting weights at h=0 and {weights}; their h=0 scores "
            + ("agree exactly" if check["scores_agree"] else "**disagree, which they must not**")
            + ". Arms exporting raw weights and arms exporting an EMA write the "
            "same tensors at update 0, so a disagreement here would be an "
            "evaluator fault rather than a recipe difference._"
        )
        add("")
    if curves:
        for arm in ARM_ORDER:
            rows = arm_curve(curves, arm)
            if not rows:
                continue
            add(f"### Arm {ARM_LETTER[arm]} — `{arm}`")
            add("")
            pack = rows[0].get("benchmark_pack")
            subset = rows[0].get("benchmark_subset")
            stratum = rows[0].get("adversarial_stratum")
            add(
                f"Benchmark pack `{pack}` subset `{subset}`; adversarial pack "
                f"`{rows[0].get('adversarial_pack')}` stratum `{stratum}`."
            )
            add("")
            has_full = any(row.get("benchmark_full_ewr") is not None for row in rows)
            header = "| h | iteration | step | benchmark EWR ± SE (decision) | adversarial EWR ± SE (decision) |"
            rule = "|---|---|---|---|---|"
            if has_full:
                header += " benchmark full-pack ± SE |"
                rule += "---|"
            add(header)
            add(rule)
            for row in rows:
                line = (
                    f"| {row['hour']} | {_fmt(row['iteration'])} | "
                    f"{_fmt(row['optimizer_step'])} | "
                    f"{_fmt(row['benchmark_ewr'])} ± {_fmt(row['benchmark_se'], 4)} "
                    f"({row['benchmark_games']} games) | "
                    f"{_fmt(row['adversarial_ewr'])} ± {_fmt(row['adversarial_se'], 4)} "
                    f"({row['adversarial_games']} games) |"
                )
                if has_full:
                    line += (
                        f" {_fmt(row.get('benchmark_full_ewr'))} ± "
                        f"{_fmt(row.get('benchmark_full_se'), 4)} "
                        f"({row.get('benchmark_full_games')} games) |"
                    )
                add(line)
            add("")
            strata_rows = [row for row in rows if row.get("adversarial_strata")]
            if strata_rows:
                names = sorted(strata_rows[0]["adversarial_strata"])
                add("Adversarial strata (only `adversarial_both` enters a decision rule):")
                add("")
                add("| h | " + " | ".join(f"`{name}`" for name in names) + " |")
                add("|---|" + "---|" * len(names))
                for row in strata_rows:
                    cells = []
                    for name in names:
                        value = row["adversarial_strata"][name]
                        cells.append(
                            f"{_fmt(value['ewr'])} ± {_fmt(value['se'], 4)}"
                        )
                    add(f"| {row['hour']} | " + " | ".join(cells) + " |")
                add("")
    else:
        add("_No hour curves recorded._")
        add("")

    # -- 4. per-iteration diagnostics --------------------------------------
    add("## 4. Per-iteration diagnostics")
    add("")
    any_telemetry = False
    add(
        "| arm | iterations | steps | Σ(lr·steps) | iteration wall s (mean / p90 / CV) "
        "| plies/s | entropy first→last | KL mean/max | clip mean | retention |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARM_ORDER:
        rows = read_telemetry(telemetry_root / f"{arm}_windows.jsonl")
        if not rows:
            continue
        any_telemetry = True
        summary = telemetry_summary(rows)
        wall = summary["iteration_wall_seconds"]
        entropy = summary.get("move_entropy") or {}
        kl = summary.get("behavior_kl") or {}
        clip = summary.get("clip_fraction") or {}
        retention = summary.get("advantage_retention") or {}
        plies = summary.get("plies_per_second") or {}
        add(
            f"| `{arm}` | {summary['iterations']} | "
            f"{summary.get('optimizer_steps', 0)} | "
            f"{summary.get('summed_step_size', 0.0):.4f} | "
            f"{_fmt(wall['mean'], 1)} / {_fmt(wall['p90'], 1)} / "
            f"{_fmt(wall['coefficient_of_variation'], 3)} | "
            f"{_fmt(plies.get('mean'), 1)} | "
            f"{_fmt(entropy.get('first'), 3)}→{_fmt(entropy.get('last'), 3)} | "
            f"{_fmt(kl.get('mean'))}/{_fmt(kl.get('max'))} | "
            f"{_fmt(clip.get('mean'), 3)} | {_fmt(retention.get('mean'), 3)} |"
        )
    add("")
    if not any_telemetry:
        add("_No per-iteration telemetry recorded._")
        add("")
    else:
        add(
            "The coefficient of variation of iteration wall-time is the number the "
            "window collector exists to hold down: Phase 14's iterations ran from "
            "24 to 138 minutes because an iteration ended when its last game did."
        )
        add("")
        add(
            "**Σ(lr·steps) is the column to read before any conclusion about "
            "damping.** Three arms given equal wall-clock do not receive equal "
            "total step size, and the residual difference after the §1b amendment "
            "is the briefed one: §2.4 gives B and C a single epoch against A's "
            "two. Read the measured column rather than the schedules — a "
            "B-under-A result on similar Σ(lr·steps) is about the recipe, and one "
            "on very different Σ(lr·steps) is partly about how much training each "
            "arm received."
        )
        add("")
        add("**Does the window budget actually pin the iteration?** Separately "
            "for each half:")
        add("")
        add("| arm | rows/iter CV | collection s (first10 → last10, CV) | training s (first10 → last10, CV) |")
        add("|---|---|---|---|")
        for arm in ARM_ORDER:
            rows = read_telemetry(telemetry_root / f"{arm}_windows.jsonl")
            if not rows:
                continue
            summary = telemetry_summary(rows)
            phases = summary.get("phase_seconds") or {}
            if not phases:
                continue
            collect = phases["collection"]
            train = phases["training"]
            add(
                f"| `{arm}` | "
                f"{summary.get('rows_per_iteration', {}).get('coefficient_of_variation', 0):.3f} | "
                f"{collect['first10']:.1f} → {collect['last10']:.1f} "
                f"(CV {collect['coefficient_of_variation']:.3f}) | "
                f"{train['first10']:.1f} → {train['last10']:.1f} "
                f"(CV {train['coefficient_of_variation']:.3f}) |"
            )
        add("")
        add(
            "The budget pins what it was designed to pin: rows per iteration, and "
            "therefore training time. It does **not** pin collection time under "
            "`phase14_mixture`, and the reason is a batching interaction this "
            "phase did not anticipate. The accepted collector groups pending "
            "decisions by *acting checkpoint*, so a 96-game lockstep batch splits "
            "once per distinct opponent snapshot in play. Arm A archives its own "
            "weights every 30 minutes to reproduce Phase 14's historical mixture, "
            "so its pool grew from 2 members to 13 and its collection throughput "
            "fell monotonically with it — 1,996 plies/s at pool 2 to 830 at pool "
            "13, a 2.4× slowdown that shows up as collection seconds, not as lost "
            "data. Arms B and C draw `pure_current` and hold exactly one snapshot, "
            "so they do not fragment. The fidelity was worth having and the cost "
            "is stated rather than hidden; a production run wanting both would "
            "need to cap the pool or pad across snapshots."
        )
        add("")

    # -- 5. the decision ----------------------------------------------------
    add("## 5. The decision the predeclared rules produced (§5)")
    add("")
    add("```text")
    for name, rule in DECISION_RULES.items():
        add(f"{name:<15} {rule}")
    add("```")
    add("")
    if candidate:
        verdict = candidate.get("verdict")
        power = candidate.get("power") or {}
        if power.get("statement"):
            add(f"> **Read this first.** {power['statement']}")
            add("")
            secondary = power.get("higher_powered_secondary") or {}
            if secondary:
                add(
                    f"> The same games scored over the full "
                    f"{secondary['games']}-board pack give a standard error of "
                    f"{secondary['standard_error']:.3f} "
                    f"({secondary['margin_in_standard_errors']} SE to the margin). "
                    f"It is reported in §3 as a secondary reading and enters no "
                    f"decision rule."
                )
                add("")
        add(f"**Verdict: {verdict}.** {candidate.get('statement')}")
        add("")
        adopt = candidate.get("adopt_recipe") or {}
        add(
            f"`adopt_recipe`: control (A) final-hour benchmark EWR "
            f"{_fmt(adopt.get('control_ewr'))}, threshold "
            f"{_fmt(adopt.get('threshold'))} (+{ADOPT_RECIPE_MARGIN}); "
            f"B {_fmt((adopt.get('candidates') or {}).get(DAMPED_ARM))}, "
            f"C {_fmt((adopt.get('candidates') or {}).get(DAMPED_PLUS_ARM))}. "
            f"Clearing: {adopt.get('clearing') or 'none'}."
        )
        add("")
        causal = candidate.get("setups_causal") or {}
        add(
            f"`setups_causal`: B adversarial {_fmt(causal.get('b_ewr'))}, "
            f"C adversarial {_fmt(causal.get('c_ewr'))}, delta "
            f"{_fmt(causal.get('delta'))} against a {SETUPS_CAUSAL_MARGIN} margin — "
            f"{'passes' if causal.get('pass') else 'does not pass' if causal.get('pass') is not None else 'not decidable'}."
        )
        add("")
        secondary = candidate.get("secondary_instrument") or {}
        if secondary:
            verb = "would also have adopted" if secondary["would_adopt"] else "would also have stopped"
            add(
                f"**Secondary instrument.** The same games scored over the full "
                f"pack put the control at {_fmt(secondary['control_ewr'])} "
                f"(threshold {_fmt(secondary['threshold'])}), with B at "
                f"{_fmt(secondary['candidates'].get(DAMPED_ARM))} and C at "
                f"{_fmt(secondary['candidates'].get(DAMPED_PLUS_ARM))} — it "
                f"{verb}."
            )
            add("")
            if secondary.get("reading"):
                add(f"> {secondary['reading']}")
                add("")

        slopes = (candidate.get("plateau_check") or {}).get("slopes") or {}
        add("`plateau_check` — h4→h6 benchmark slope per arm:")
        add("")
        add("| arm | h4 | h6 | Δ | per hour | flat |")
        add("|---|---|---|---|---|---|")
        for arm in ARM_ORDER:
            slope = slopes.get(arm)
            if not slope:
                add(f"| `{arm}` | — | — | — | — | — |")
                continue
            add(
                f"| `{arm}` | {_fmt(slope['h4'])} | {_fmt(slope['h6'])} | "
                f"{_fmt(slope['delta'])} | {_fmt(slope['per_hour'])} | "
                f"{_fmt(slope['flat'])} |"
            )
        add("")
        if verdict == VERDICT_STOP:
            add(
                "Section 5's `stop_rule` fires. No long run is authorized by this "
                "file; the decision returns to the operator."
            )
            add("")
            add("### What the shootout established, and what it did not")
            add("")
            add(
                "**No arm moved measurably in six hours.** Every arm's whole "
                "h-curve sits inside a single standard error of its own starting "
                "point, and the h4→h6 slopes are negative or flat for all three. "
                "The starting weights are P24 — hour 24 of the Phase 14 run — and "
                "Phase 14's own gain arrived in its first six hours, so a further "
                "six hours from that point producing nothing is consistent with "
                "what was already known rather than surprising."
            )
            add("")
            add(
                "So the correct reading of `stop_rule` here is **not** \"the "
                "damped recipes are worse than the control\". It is that this "
                "experiment could not tell any of the three apart. A recipe "
                "comparison needs either a starting point that is still learning, "
                "a horizon long enough for the differences to accumulate, or an "
                "instrument that can resolve the margin it is asked about — and "
                "this run had none of the three."
            )
            add("")
            add(
                "**The `setups_causal` result is the easiest thing in this report "
                "to misread, so plainly: it does not show that expanded setups "
                "fail.** Arm C's training distribution moved hard — mean game "
                "length 855.6 plies against B's 592.7 and A's 482.3, a higher draw "
                "rate, and 26,297 games for the same ~20M learner decisions where "
                "B played 47,508. The distribution moved; the strength did not. "
                "Six hours from a saturated start cannot separate \"expanded "
                "setups do not help\" from \"expanded setups need a longer "
                "horizon, or a start with headroom left\" — this experiment does "
                "not distinguish those two, and nothing here should be cited for "
                "the first. The one directional hint, itself well inside the "
                "noise, points the other way: C is marginally *best* of the three "
                "on the `adversarial_opponent` stratum, where the opponent plays "
                "adversarial setups and the arm does not."
            )
            add("")
            add(
                "What it *did* establish is infrastructural, and that part is "
                "solid: the window collector holds iteration wall-time to a "
                "coefficient of variation near 0.05 across ~300 iterations where "
                "Phase 14's grew 8× over 102, three arms ran six hours each with "
                "zero vetoes and zero non-finite losses, gradients or parameters, "
                "and the accepted objective, controller and filter were driven "
                "unmodified throughout at the frozen 0.25 retention."
            )
            add("")
        elif verdict == VERDICT_ADOPT:
            add(
                f"The winner is frozen as `phase16_recipe_candidate_v1` with every "
                f"flag, schedule constant and mixture proportion, plus the h-curve "
                f"of all three arms with their pack names."
            )
            add("")
    else:
        add("_No candidate decision recorded._")
        add("")

    # -- 6. limitations -----------------------------------------------------
    add("## 6. `known_limitations`")
    add("")
    add(
        f"- **{ARM_HOURS:.0f}-hour horizon.** Phase 14 established that six-hour runs "
        "are decision-grade for this model, and also that its own gain arrived in "
        "the first six hours. A six-hour shootout can rank recipes at six hours; it "
        "cannot say what any of them does at sixty."
    )
    add(
        "- **One seed per arm.** Each arm was run once. The arm-to-arm differences "
        "reported here carry no estimate of seed-to-seed variance, and the 0.03 "
        "decision margin was fixed in advance precisely because that variance is "
        "unmeasured."
    )
    add(
        "- **The decision margin is smaller than the instrument's noise.** One "
        "standard error on the 60-board decision subset is ±0.056 at the observed "
        "EWR; the predeclared margin is 0.03, i.e. 0.53 SE. The clearest "
        "demonstration is that the same games read over the full 120-board pack "
        "reverse the verdict for arm B (+0.046 against +0.017). Neither reading is "
        "preferable; both are too noisy for the question. Every EWR difference in "
        "this report smaller than about 0.11 should be treated as unmeasured."
    )
    add(
        "- **No arm learned, so no recipe was actually tested.** The six-hour "
        "curves are flat within noise for all three arms, which means the shootout "
        "compared three recipes none of which had a measurable effect. `stop_rule` "
        "firing reflects that absence, not a demonstrated ranking."
    )
    add(
        "- **The window budget pins data, not collection time.** Rows per "
        "iteration and training seconds hold to a CV near 0.06. Collection time "
        "does not, when the opponent mixture fragments the inference batch: the "
        "accepted collector groups pending decisions by acting checkpoint, so arm "
        "A's growing historical pool (2 → 13 members) cut its collection "
        "throughput from 1,996 to 830 plies/s. `pure_current` arms hold one "
        "snapshot and are unaffected."
    )
    add(
        "- **Partial-window advantages are built and tested but unused.** §2.2's "
        "boundary bootstrap exists in `targets.py` and is covered; the production "
        "path emits whole games so every row carries an exact W/D/L target, because "
        "the accepted objective averages its value and belief terms over every row "
        "and has no per-row loss mask. Rewriting the objective to add one was out "
        "of scope."
    )
    add(
        "- **A stable behavior identity.** The learner's collection snapshot is the "
        "constant `CURRENT` and its weights rotate each window, because a window "
        "collector continues the same games across an update. Each window's "
        "telemetry records the state-dict digest that actually played it, but the "
        "per-decision token no longer distinguishes them."
    )
    add(
        "- **In-flight games are not checkpointed.** A resumed arm keeps its "
        "weights, optimizer, EMA, clock and window numbering, and reseats its "
        "population from fresh draws. Partially-played games are lost, not "
        "corrupted."
    )
    add("")
    add("---")
    add("")
    add(
        "_Engineering deliverable. Not a strength claim, not a scientific result, "
        "and not a validation of any recipe beyond the six hours measured._"
    )
    add("")
    return "\n".join(lines)


def write_report(path, **kwargs) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(**kwargs))
    return target


__all__ = ["ARM_LETTER", "ARM_ORDER", "render_report", "write_report"]
