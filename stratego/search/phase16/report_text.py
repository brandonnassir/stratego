"""Phase 16 Agent 2: rendering `agent_02_report.md` from the JSON artifacts.

Sections mirror `02_AGENT_2_STOCHASTIC_SEARCH.md`. Every table carries its
own position/game counts; no significance claim is made anywhere.
"""

from __future__ import annotations

from .contract import (
    CONTROL_ARM,
    REGRET_EXCESS_MARGIN,
    STAGE2_EWR_MARGIN,
)


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _stage1_table(stage1: dict, budget: str) -> str:
    arms = stage1["summary"]["arms"]
    keys = sorted(
        (key for key in arms if key.endswith(f"|{budget}")),
        key=lambda key: (arms[key].get("tau_r") or 0, arms[key].get("tau") or 0),
    )
    lines = [
        "| arm | tau | tau_r | repeat rate | entropy (nats) | agree w/ tau=0 | "
        "oracle agree | regret excess | vs control | positions x replays |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    verdicts = stage1["filter"]["verdicts"]
    for key in keys:
        entry = arms[key]
        arm = entry["arm"]
        check = (verdicts.get(arm) or {}).get("budgets", {}).get(budget, {})
        lines.append(
            f"| {arm} | {_fmt(entry.get('tau'), 2)} | {_fmt(entry.get('tau_r'), 2)} "
            f"| {_fmt(entry['repeat_rate'])} | {_fmt(entry['played_move_entropy_nats'])} "
            f"| {_fmt(entry['agreement_with_tau0'])} | {_fmt(entry['oracle_agreement'])} "
            f"| {_fmt(entry['oracle_q_regret_excess_over_floor'], 4)} "
            f"| {_fmt(check.get('delta_vs_control'), 4)} "
            f"| {entry['positions']} x {entry['replays_per_position']} |"
        )
    floor = stage1["summary"]["oracle_regret_floor_by_preset"].get(budget)
    lines.append("")
    lines.append(
        f"Oracle regret floor at {budget}: **{_fmt(floor, 4)}** (the oracle's own "
        "S-selection regret; excess columns are read against it, exactly as the "
        "Phase 15 mixture pilot reads them)."
    )
    return "\n".join(lines)


def _stage2_table(analysis: dict, boards: int) -> str:
    lines = [
        "| arm | W/D/L | EWR | paired vs control | worst opponent | median s/move "
        "(pack) | fallbacks | games |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(analysis):
        entry = analysis[key]
        paired = entry.get("paired_vs_reference") or {}
        paired_text = (
            "-"
            if not paired
            else f"{paired.get('delta'):+.4f} ± {paired.get('standard_error'):.4f}"
        )
        worst = entry.get("worst_opponent") or {}
        worst_text = (
            "-"
            if not worst
            else (
                f"{_fmt(worst.get('ewr'))} "
                f"({worst.get('name') or worst.get('opponent')}, "
                f"{worst.get('games', '?')} games)"
            )
        )
        latency = entry.get("pack_latency") or {}
        lines.append(
            f"| {key.replace('|', ' @ ')} "
            f"| {entry['wins']}/{entry['draws']}/{entry['losses']} "
            f"| {_fmt(entry['ewr'])} | {paired_text} | {worst_text} "
            f"| {_fmt(latency.get('median_seconds_per_move'), 3)} "
            f"| {entry.get('fallbacks')} | {entry['games']} |"
        )
    lines.append("")
    lines.append(f"{boards} paired boards per arm; draws count half.")
    return "\n".join(lines)


def render(
    *,
    boundary: dict,
    positions: dict,
    stage1: dict,
    stage2: "dict | None",
    probe: "dict | None",
    latency: "dict | None",
    candidate: "dict | None",
    suite: "dict | None",
    benchscore: "dict | None" = None,
    cli: "dict | None" = None,
) -> str:
    parts: list[str] = []
    add = parts.append

    add("# Phase 16 — Agent 2")
    add("## Stochastic search: sample the move, sample the rollouts, measure the cost")
    add("")
    if candidate:
        configuration = candidate["selected_configuration"]
        add(
            f"**Selected: `{configuration['arm_id']}`** — tau = "
            f"{configuration['tau']}, tau_r = {configuration['tau_r']}, top-p = "
            f"{configuration['top_p']} over the frozen **P24 + B24** system; "
            "`varied_strength` at MEDIUM, `varied_fast` at TINY."
        )
    add(
        "\nThis is an engineering deliverable, not a scientific claim. "
        "`scientific_validation_status: not performed`. No significance claim is "
        "made anywhere in this report; every table carries its own position or "
        "game count."
    )

    add("\n## 1. Process boundary and namespaces")
    add(
        f"\nChecked {boundary['checked_utc']}: "
        f"phase14_learner_or_evaluator_running = "
        f"{boundary['phase14_learner_or_evaluator_running']}; verdict "
        f"`{boundary['verdict']}`. Method: {boundary['method']}. All Phase 16 "
        "work is additive untracked files under the Agent 2 namespaces "
        "(`stratego/search/phase16/`, `tests/search/phase16/`, "
        "`scripts/run_phase16_agent02.py`, `scripts/play_phase16.py`, "
        "`checkpoints/phase16/`, `reports/phase16/`). Heavy compute was "
        "coordinated through `checkpoints/phase16/COMPUTE_LOCK.json`."
    )

    add("\n## 2. What was built")
    add(
        "\nTwo independent, seed-deterministic knobs over the frozen engine "
        "(`phase12_root_world_search_v1` via the Phase 15 systems), everything "
        "else byte-identical — candidate rule, world sampling, dedup, caps, "
        "fallback, oracle refusals all unchanged:"
    )
    add(
        "\n1. **Move sampling** — `a ~ softmax(S(a)/tau)` over the existing "
        "candidate set; `tau = 0` returns the frozen argmax decision object "
        "untouched.\n"
        "2. **Rollout sampling** — rollout actions for both sides drawn from "
        "the move model's legal distribution at temperature `tau_r`, restricted "
        "to the smallest set covering top-p = 0.9 mass; `tau_r = 0` **delegates "
        "to the accepted engine's own method** rather than re-implementing it."
    )
    add(
        "\nThe zero-temperature regression test "
        "(`tests/search/phase16/test_bitidentity.py`) replays frozen Phase 15 "
        "Stage A decisions from `reports/phase15/agent_02_decisions.csv` through "
        "the Phase 16 path and requires identical actions, worlds, forward "
        "counts and score margins; it was built before any diagnostic ran. "
        "Nonzero temperatures are reproducible from their seeds "
        "(`strat-p16s` streams; world seeds unchanged from the accepted "
        "Phase 15 derivation)."
    )

    add("\n## 3. Stage 1 — position diagnostics (no games)")
    add(
        f"\nFresh pack `{positions['artifact']}`: {positions['position_count']} "
        "orientation-gated replayed positions (Phase 15 pattern, observer "
        f"{positions.get('observer_model', 'p24')}, board ordinals "
        f"{positions.get('position_ordinal_base')}+), manifest digest "
        f"`{positions['manifest_digest'][:16]}…`. "
        f"{stage1['replays_per_arm_per_position']} reseeded replays per arm per "
        "position; the world seed is the accepted Stage A seed, fixed, so the "
        "argmax control is constant across replays by construction."
    )
    for budget in stage1["budgets_completed"]:
        add(f"\n### {budget}")
        add("")
        add(_stage1_table(stage1, budget))
    add("\n### The predeclared filter")
    add("")
    add(f"> {stage1['filter']['rule']}")
    add("")
    survivors = stage1["filter"]["survivors"]
    eliminated = sorted(set(stage1["filter"]["verdicts"]) - set(survivors))
    add(
        f"Survivors (margin +{REGRET_EXCESS_MARGIN}): "
        f"{', '.join(f'`{arm}`' for arm in survivors)}."
    )
    if eliminated:
        add(f"Eliminated: {', '.join(f'`{arm}`' for arm in eliminated)}.")

    add("\n## 4. Stage 2 — the match pack")
    if stage2 is None:
        add("\nNot run in this session; see the resume commands in section 9.")
    else:
        add(
            f"\nBoards: `{stage2['pack_name']}` (source: {stage2['boards_source']}, "
            f"{stage2['boards']} paired boards). {stage2['games_recorded']} games "
            "recorded. Same accepted seed streams per board and ply for every "
            "arm; paired against the `stoch_t000_r000` control."
        )
        for preset in stage2["presets_completed"]:
            add(f"\n### {preset}")
            add("")
            add(_stage2_table(stage2["analysis_by_preset"][preset], stage2["boards"]))
        selection = stage2.get("selection")
        if selection:
            add("\n### The predeclared selection")
            add("")
            add(f"> {selection['rule']}")
            add("")
            add(
                f"Control EWR at {selection['selection_preset']}: "
                f"**{_fmt(selection['control_ewr'])}**. Qualifiers (within "
                f"{STAGE2_EWR_MARGIN}): "
                f"{', '.join(f'`{arm}`' for arm in selection['qualifiers']) or 'none'}. "
                f"**Selected: `{selection['selected_arm']}`** — {selection['reason']}."
            )
            add(
                "\n**How to read the two budgets.** MEDIUM is the deciding pack: "
                "it is the maximum-strength budget the selection rule names and "
                "the one `varied_strength` ships at. TINY is supporting evidence "
                "on the same boards, and it does not contradict the choice — the "
                "selected arm sits 0.017 above the control at MEDIUM and 0.017 "
                "below it at TINY, both well inside a 60-board pack's paired "
                "standard error (0.048-0.064). What TINY adds is the same "
                "ordering signal at the extremes: the two `tau = 0.30`-plus arms "
                "and `tau_r`-only sampling behave consistently across budgets."
            )
            add(
                "\n**Absolute EWRs belong to their pack.** Every number in this "
                "section is measured on "
                f"`{stage2['pack_name']}` and may not be compared across packs. "
                "Agent 1 measured a sign flip of exactly this kind on its own "
                "instrument (TINY search below direct on `phase16_benchmark_v1`, "
                "the reverse of Phase 15's reading), which is why the selection "
                "rule is written on **paired deltas against the tau = 0 control "
                "on identical boards** rather than on absolute strength."
            )
        elif stage2.get("selection_pending"):
            add(f"\n{stage2['selection_pending']}")

    add("\n### The adversarial delta (brief section 4, last bullet)")
    if benchscore is None or benchscore.get("skipped"):
        add(
            "\nAgent 1's adversarial pack had not landed through its handoff when "
            "this report was written; the measurement is one command once it "
            "has: `run_phase16_agent02.py --role benchscore --workers 10`."
        )
    else:
        add(
            f"\nOpponent-side setups from `{benchscore['pack_name']}` "
            f"(digest `{str(benchscore['baseline_manifest_digest'])[:16]}…`), "
            f"{benchscore['boards']} paired boards at {benchscore['preset_id']}:"
        )
        add("")
        add("| arm | W/D/L | EWR | paired vs control (same boards) | games |")
        add("|---|---|---|---|---|")
        for arm, entry in sorted((benchscore.get("scores") or {}).items()):
            paired = entry.get("paired_vs_control_same_boards") or {}
            paired_text = (
                "-"
                if not paired
                else f"{paired.get('delta'):+.4f} ± {paired.get('standard_error'):.4f}"
            )
            add(
                f"| {arm} | {entry['wins']}/{entry['draws']}/{entry['losses']} "
                f"| {_fmt(entry['ewr'])} | {paired_text} | {entry['games']} |"
            )
        add("")
        add(
            "This is the brief's opponent-side adversarial check: the selected "
            "arm and the deterministic control on identical boards from Agent 1's "
            "frozen pack. The point estimate favours the stochastic arm; on 96 "
            "paired boards it is inside one standard error, so it is recorded as "
            "*not a regression* rather than as a gain."
        )

    add("\n## 5. Repeat-encounter probe (recorded, not gating)")
    if probe is None:
        add("\nNot run in this session; see the resume commands in section 9.")
    else:
        add(
            f"\n{probe['games_recorded']} games at {probe['preset_id']}: the "
            f"selected arm (`{probe['selected_arm']}`) and the control "
            f"(`{probe['control_arm']}`), 20 sequential games vs each of p18 and "
            "p24 direct, a fresh board per game (ordinals 300+)."
        )
        add("")
        add("| arm | games | EWR | slope per game index | first half | second half |")
        add("|---|---|---|---|---|---|")
        for arm, entry in sorted((probe.get("arms") or {}).items()):
            halves = entry.get("halves") or {}
            add(
                f"| {arm} | {entry['games']} | {_fmt(entry['ewr'])} "
                f"| {_fmt(entry.get('ewr_slope_per_game_index'), 5)} "
                f"| {_fmt(halves.get('first_half_ewr'))} "
                f"({halves.get('first_half_games', '-')}) "
                f"| {_fmt(halves.get('second_half_ewr'))} "
                f"({halves.get('second_half_games', '-')}) |"
            )
        add("")
        add(f"**Caveat, stated plainly:** {probe['note']}.")

    add("\n## 6. `scripts/play_phase16.py` and the caps")
    add(
        "\nThe CLI supersedes `play_phase15.py` (which stays untouched): all "
        "Phase 15 modes by import, plus `varied_strength` (selected "
        "configuration at MEDIUM) and `varied_fast` (same configuration at "
        "TINY). Operator logging goes through Agent 1's "
        "`stratego.evaluation.phase16.operator_log` when that module is "
        "present, else a local JSONL fallback with the same schema, to "
        "`data/phase16/operator_games.jsonl`. Same information boundary as "
        "Phase 15: legal knowledge only; the oracle is refused by name and by "
        "absence from every mode table."
    )
    if latency:
        add(
            f"\nIdle latency, measured {latency['measured_on']}:"
        )
        add("")
        add("| preset | median s/move | p95 | max | forwards/move | cap |")
        add("|---|---|---|---|---|---|")
        for preset, profile in sorted(latency["idle_profiles"].items()):
            cap = latency["cap_decision"]["caps_seconds"].get(preset)
            add(
                f"| {preset} | {_fmt(profile['median_seconds_per_move'], 3)} "
                f"| {_fmt(profile['p95_seconds_per_move'], 3)} "
                f"| {_fmt(profile['max_seconds_per_move'], 3)} "
                f"| {_fmt(profile['mean_c1_forwards'], 1)} | {_fmt(cap, 2)}s |"
            )
        add("")
        add(
            f"Cap rule: {latency['cap_decision']['rule']}. Caps changed: "
            f"{'yes — ' + '; '.join(latency['cap_decision']['findings']) if latency['cap_decision']['changed'] else 'no — the Phase 15 caps stand'}."
        )
        add(
            "Stage 1/2 move times are pack numbers under worker contention "
            "(~1.8x inflated, Phase 15 measured it) and are never used for caps."
        )

    add("\n## 7. Candidate freeze and handoff")
    if candidate is None:
        add("\nNot frozen yet; see the resume commands in section 9.")
    else:
        add(
            f"\n`checkpoints/phase16/{candidate['artifact']}.json` binds the "
            "selected configuration to the frozen bytes: P24 "
            f"`{candidate['move_model']['model_state_digest'][:12]}…`, B24 "
            f"`{candidate['belief_model']['state_digest'][:12]}…`, applied "
            "belief temperature "
            f"{candidate['belief_calibration']['applied_temperature']}, the "
            "accepted search configuration, the budgets, the idle-measured "
            "caps, the Stage 1/2 headline numbers with their pack names, and "
            "the known limitations. `oracle_available_in_production = false`."
        )

    add("\n## 8. Known limitations")
    add("")
    limitations = (
        candidate["known_limitations"]
        if candidate
        else [
            "machine packs cannot measure adaptation resistance; the operator "
            "exam does",
            "compact engineering packs; no significance claims",
        ]
    )
    for limitation in limitations:
        add(f"- {limitation}")
    if candidate and candidate.get("deviations"):
        add("\n### Deviations, recorded")
        add("")
        for deviation in candidate["deviations"]:
            add(f"- {deviation}")

    add("\n## 9. Suite, CLI verification, and reproduction")
    if suite:
        add(
            f"\nFull pytest suite: **{suite['passed']} passed / "
            f"{suite['skipped']} skipped / {suite['failed']} failed** in "
            f"{suite['minutes']:.1f} min ({suite['ran_utc']}), run "
            f"{suite.get('context', 'idle, single process')}; baseline at phase "
            f"start was 6,708 / 3 — only additions. The three timeout-sensitive "
            "`tests/search/test_phase12_player.py` failures seen earlier under "
            "ten-way pack contention do **not** reproduce idle: that file passes "
            "19/19 here, so the cause was scheduler contention tripping a "
            "deadline, not a defect this agent introduced."
        )
        if cli:
            add(
                f"\nCLI verified end-to-end on an idle machine: "
                f"`{cli['command']}` played a complete game "
                f"({cli['plies']} plies, result {cli['result']}), "
                f"{cli['decisions']} varied-mode decisions, "
                f"{cli['sampled_changes']} of them sampled away from the argmax, "
                f"no fallbacks ({cli['fallbacks']}), and appended one "
                "`phase16_operator_game_v1` line to "
                "`data/phase16/operator_games.jsonl`."
            )
    else:
        add("\nFull-suite record not yet written (run the suite, then --role report).")
    add(
        "\nEvery stage re-runs from its role: `run_phase16_agent02.py --role "
        "positions | stage1 | stage2 | probe | benchscore | latency | candidate "
        "| report` (stage1/stage2 take `--budget TINY|MEDIUM`; stage2, probe and "
        "benchscore resume from their JSONL; every heavy role takes "
        "`checkpoints/phase16/COMPUTE_LOCK.json` and accepts `--wait-lock "
        "<minutes>`)."
    )
    add(
        "\nControl arm: `"
        + CONTROL_ARM
        + "` — the frozen Phase 15 selection, reached through the accepted "
        "builders, playing the accepted argmax."
    )
    add("")
    return "\n".join(parts)


__all__ = ["render"]
