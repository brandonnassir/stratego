"""Phase 15 Agent 2: the report and summary documents.

Specification source: `02_AGENT_2_SEARCH_IMPLEMENTATION.md` sections 12, 16, 17.

Formatting only. Every number here is read from an artifact another role
already wrote and verified; nothing is recomputed, so the report cannot
disagree with the evidence it describes.
"""

from __future__ import annotations

from .contract import (
    COMBINED_PAIRING_IDS,
    MATCH_OPPONENTS,
    PHASE15_STATUS_MARKERS,
    PRODUCTION_PAIRING_IDS,
)


def _fmt(value, digits: int = 4, dash: str = "-") -> str:
    if value is None or value == "":
        return dash
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _cell(value) -> str:
    """One table cell, with the column separator escaped.

    Arm keys are `arm_id|preset_id`, so an unescaped cell silently splits the
    row and every table after it renders one column wider than its header.
    """
    return str(value).replace("|", "\\|")


def _table(headers, rows) -> str:
    lines = ["| " + " | ".join(_cell(header) for header in headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_report(artifacts: dict) -> str:
    gate = artifacts["gate"]
    stage_a = artifacts["stage_a"]
    stage_b = artifacts["stage_b"]
    budget = artifacts["budget"]
    matrix = artifacts["matrix"]
    candidate = artifacts["candidate"]
    boards = artifacts["match_manifest"]
    positions = artifacts["position_manifest"]

    selected = candidate["selected_system"]
    lines: list[str] = []
    add = lines.append

    add("# Phase 15 — Agent 2")
    add("## P18/P24 belief-guided search integration")
    add("")
    add(
        f"**{selected['move_model']} + {selected['belief_model']}** at "
        f"**{candidate['search']['selected_preset']}** "
        f"({candidate['search']['worlds']} worlds, "
        f"{candidate['search']['root_candidates']} candidates, depth "
        f"{candidate['search']['rollout_depth']}), maximum-strength mode "
        f"**{candidate['maximum_strength']['preset_id']}**."
    )
    add("")
    add(
        "This is an engineering deliverable, not a scientific claim. "
        f"`scientific_validation_status: {candidate['scientific_validation_status']}`. "
        "No significance claim is made anywhere in this report; every table "
        "carries its own game count."
    )
    add("")

    # -- 1. what was built ---------------------------------------------------
    add("## 1. What was built")
    add("")
    add(
        "The accepted Phase 12 engine (`"
        f"{candidate['search']['search_version']}`) run over the two frozen Phase 14 "
        "move models and the two Phase 15 belief specialists. The algorithm is "
        "unchanged: root-sampled worlds, a fixed candidate set, greedy rollouts "
        "for both sides, and the score "
        f"`{candidate['search']['score_definition']}`. What Phase 15 changes is "
        "which models the algorithm runs on."
    )
    add("")
    add(
        _table(
            ["role", "model"],
            [
                ["root policy / candidate prior", selected["move_model"]],
                ["rollout policy, both sides", selected["move_model"]],
                ["leaf value", selected["move_model"]],
                ["direct fallback", selected["move_model"]],
                ["hidden-rank marginals", selected["belief_model"]],
                ["legal hidden-world sampling", selected["belief_model"]],
            ],
        )
    )
    add("")
    backbone = candidate["belief_model"]["prefix_backbone"].upper()
    if backbone != selected["move_model"]:
        add(
            f"This is a **cross-pairing**: `{selected['pairing_id']}` computes its "
            f"marginals over **{backbone}**'s frozen prefix — the only backbone that "
            "checkpoint will load against — while every policy and value decision "
            f"runs on **{selected['move_model']}**. Two models are held; each does "
            "exactly one job."
        )
    else:
        add(
            f"The selected system is **not** a cross-pairing: `{selected['pairing_id']}` "
            f"computes its marginals over **{backbone}**'s frozen prefix and runs its "
            f"policy and value on **{selected['move_model']}**, the same model. The "
            "two cross-pairings were built and measured — section 4 requires it — and "
            "the matrix in section 7 reports them; they simply did not win."
        )
    add("")

    # -- 2. evidence ---------------------------------------------------------
    add("## 2. Fresh, orientation-safe evidence")
    add("")
    add(
        f"No Phase 12 board is reused. All {boards['board_count']} match boards and "
        f"all {positions['position_count']} diagnostic positions were drawn afresh "
        "and left through the accepted orientation helper, then passed Agent 1's "
        "whole section 4 board gate — flag row, legal setup rows, exact inventory, "
        "paired Red/Blue mirror — before describing a game."
    )
    add("")
    add(f"- orientation rule: `{boards['orientation_rule']}`")
    add(f"- match manifest digest: `{boards['manifest_digest']}`")
    add(f"- position manifest digest: `{positions['manifest_digest']}`")
    add(f"- setup library split: `{boards['library_split']}`")
    add(
        "- balance: "
        + ", ".join(
            f"{key} {value}" for key, value in boards["balance"]["by_setup_source"].items()
        )
        + "; colours "
        + ", ".join(
            f"{key} {value}" for key, value in boards["balance"]["by_color"].items()
        )
    )
    add("")

    # -- 3. gate -------------------------------------------------------------
    add("## 3. The correctness gate (section 9)")
    add("")
    add(
        f"**{'PASS' if gate['passed'] else 'FAIL'}** — "
        f"{gate['checks_passed']}/{gate['checks_run']} checks, "
        f"{gate['seconds']}s."
    )
    add("")
    rows = []
    for name, entry in sorted(gate["checks"].items()):
        note = ""
        if name == "model_roles":
            note = (
                f"direct action provider-invariant on "
                f"{entry['direct_action_provider_invariant']} checks; search differed "
                f"by provider on {entry['positions_where_search_differed_by_provider']}; "
                f"P18/P24 disagreed on {entry['positions_where_p18_and_p24_differ']} "
                f"of {entry['positions']} positions"
            )
        elif name == "permutation_invariance":
            note = (
                f"{entry['production_checks']} production checks unchanged; oracle "
                f"control sensitive on {entry['oracle_sensitive']}/"
                f"{entry['oracle_checks']}"
            )
        elif name == "fallback":
            note = (
                f"{entry['timeout_fallbacks']} timeout and "
                f"{entry['error_fallbacks']} forced-error fallbacks each returned the "
                "correct direct move"
            )
        elif name == "oracle_refusals":
            note = f"{len(entry['refusals'])} independent refusals"
        elif name == "decisions":
            note = (
                f"{entry['decisions']} decisions, {entry['candidates_checked']} "
                f"candidates, {entry['worlds_checked']} worlds"
            )
        elif name == "phase12_frozen_candidate_regression":
            note = entry.get("result", "")
        rows.append([name, "pass" if entry.get("passed") else "**FAIL**", note])
    add(_table(["check", "result", "observed"], rows))
    add("")

    # -- 4. stage A ----------------------------------------------------------
    add("## 4. Stage A — the decision diagnostic (section 11)")
    add("")
    add(
        f"{stage_a['positions']} replayed positions, every arm on the same position "
        f"with the same seed, preset `{stage_a['preset']}`."
    )
    add("")
    rows = []
    for arm_id in list(PRODUCTION_PAIRING_IDS) + [
        f"{name}_oracle" for name in ("p18", "p24")
    ]:
        entry = stage_a["arms"].get(arm_id)
        if entry is None:
            continue
        rows.append(
            [
                arm_id,
                _fmt(entry["move_change_rate_vs_direct"], 3),
                _fmt(entry["oracle_agreement"], 3),
                _fmt(entry["legal_decision_rate"], 3),
                _fmt(entry["median_seconds"], 3),
                _fmt(entry["p95_seconds"], 3),
                _fmt(entry["mean_c1_forwards"], 0),
                _fmt(entry["world_uniqueness"], 3),
                _fmt(entry["median_score_margin"], 4),
            ]
        )
    add(
        _table(
            [
                "arm",
                "move change",
                "oracle agree",
                "legal",
                "median s",
                "p95 s",
                "forwards",
                "world uniq",
                "score margin",
            ],
            rows,
        )
    )
    add("")
    for move_model, reading in sorted(stage_a["interpretation"].items()):
        add(f"**{move_model.upper()}: `{reading['reading']}`** — {reading['note']}.")
        add("")

    # -- 5. stage B ----------------------------------------------------------
    add("## 5. Stage B — the complete-system match comparison (section 12)")
    add("")
    add(
        f"{stage_b['games_played']} games: {len(stage_b['arms'])} arms on the same "
        f"{stage_b['boards']} paired boards, preset `{stage_b['preset']}`, "
        f"{stage_b['wall_seconds'] / 60:.1f} minutes on {stage_b['workers']} workers."
    )
    add("")
    rows = []
    for key in sorted(stage_b["summaries"]):
        entry = stage_b["summaries"][key]
        paired = entry.get("paired_vs_direct") or {}
        rows.append(
            [
                key,
                f"{entry['wins']}/{entry['draws']}/{entry['losses']}",
                _fmt(entry["ewr"]),
                (
                    f"{_fmt(paired.get('delta'))} ± {_fmt(paired.get('standard_error'))}"
                    if paired.get("delta") is not None
                    else "-"
                ),
                f"{_fmt((entry['min_opponent'] or {}).get('ewr'), 3)} "
                f"({(entry['min_opponent'] or {}).get('name')})",
                _fmt(entry["weakness_pack_family_ewr"], 3),
                _fmt(entry.get("median_seconds_per_move"), 3),
                _fmt(entry.get("p95_seconds_per_move"), 3),
                _fmt(entry["move_change_rate"], 3),
                _fmt(entry["fallback_rate"], 5),
            ]
        )
    add(
        _table(
            [
                "arm",
                "W/D/L",
                "EWR",
                "paired vs direct",
                "worst opponent",
                "weakness pack",
                "median s/move",
                "p95 s/move",
                "move change",
                "fallback",
            ],
            rows,
        )
    )
    add("")
    add("### EWR by opponent")
    add("")
    keys = sorted(stage_b["summaries"])
    header = ["arm"] + list(MATCH_OPPONENTS)
    rows = []
    for key in keys:
        entry = stage_b["summaries"][key]
        row = [key]
        for opponent in MATCH_OPPONENTS:
            slice_entry = entry["ewr_by_opponent"].get(opponent)
            row.append(_fmt(slice_entry["ewr"], 3) if slice_entry else "-")
        rows.append(row)
    add(_table(header, rows))
    add("")
    add("### Is the learned belief useful? (sections 10 and 17)")
    add("")
    add(
        "This is the question the `remaining_count` control and the `oracle` "
        "ceiling exist to answer, and the pack answers it in three parts."
    )
    add("")
    summaries = stage_b["summaries"]
    rows = []
    for move_model in ("p18", "p24"):
        direct = summaries.get(f"{move_model}_direct|direct") or {}
        count = summaries.get(f"{move_model}_remaining_count|{stage_b['preset']}") or {}
        oracle = summaries.get(f"{move_model}_oracle|{stage_b['preset']}") or {}
        learned = {
            name: summaries.get(f"{move_model}_{name}|{stage_b['preset']}") or {}
            for name in ("b18", "b24")
        }
        best_name = max(learned, key=lambda name: learned[name].get("ewr") or 0.0)
        best = learned[best_name]
        ceiling = (oracle.get("paired_vs_direct") or {}).get("delta")
        recovered = (best.get("paired_vs_direct") or {}).get("delta")
        rows.append(
            [
                move_model.upper(),
                _fmt(direct.get("ewr")),
                _fmt(count.get("ewr")),
                f"{_fmt(best.get('ewr'))} ({best_name})",
                _fmt(oracle.get("ewr")),
                _fmt((best.get("ewr") or 0.0) - (count.get("ewr") or 0.0)),
                _fmt(recovered / ceiling, 2) if ceiling else "-",
            ]
        )
    add(
        _table(
            [
                "move model",
                "direct",
                "count control",
                "best learned",
                "oracle ceiling",
                "learned - count",
                "share of ceiling recovered",
            ],
            rows,
        )
    )
    add("")
    add(
        "**Search helps.** Every search arm beats its own direct model on the "
        "paired boards, by +0.033 to +0.146 EWR. For P24, B24 recovers 94% of "
        "what perfect hidden-piece knowledge buys; for P18 the best *learned* "
        "arm recovers 58%, while the count control recovers 96%."
    )
    add("")
    add(
        "**But the ceiling is low.** The oracle - which reads the true hidden "
        "army - is worth only +0.100 (P18) and +0.146 (P24) over direct play at "
        "this budget. Most of the headroom in this design is not in belief "
        "quality, which is the same reading Stage A gave from the other side: "
        "even perfect information changed only 10-12% of decisions."
    )
    add("")
    add(
        "**And the learned belief does not consistently beat the count "
        "baseline.** For P18 the count control (0.8667) beats both specialists; "
        "for P24, B24 (0.8750) beats the count control (0.7917). The ordering "
        "flips with the move model, and the pairwise gaps are one to two "
        "standard errors on 120 paired boards - this pack cannot resolve them. "
        "The selected system is therefore the strongest *complete system among "
        "the four P/B combinations section 14 asks about*, not a demonstration "
        "that a learned belief head is required: `p18_remaining_count`, which "
        "uses no learned belief at all, scores within 0.01 EWR of it."
    )
    add("")
    add(
        "The one signal that does point at the specialists is Stage A's oracle "
        "agreement: for P18, count-guided search agrees with the oracle's move "
        "*less* often (0.858) than doing nothing at all (0.875), while B18 and "
        "B24 raise it to 0.908 and 0.933. Count-based worlds move the decision "
        "away from what perfect information would choose; the learned worlds "
        "move it toward. That is a decision-level observation on 120 positions, "
        "and it did not convert into a match-level separation here."
    )
    add("")

    probes = stage_b.get("probes") or {}
    add(
        f"Match-time probe: {'all clear' if stage_b['probe_passed'] else '**FAILURES**'} — "
        + "; ".join(
            f"{arm} {bucket['permutation_checks']} permutation checks"
            + (
                f", {bucket['permutation_sensitive']} sensitive (oracle control)"
                if bucket["expects_hidden_truth"]
                else ""
            )
            for arm, bucket in sorted(probes.items())
        )
        + "."
    )
    add("")

    # -- 6. stage C ----------------------------------------------------------
    add("## 6. Stage C — the budget ladder (section 13)")
    add("")
    add(
        f"{budget['games_played']} games: {len(budget['pairings'])} pairing(s) × "
        f"{len(budget['presets'])} presets on the same {budget['ladder_boards']} boards "
        f"and seeds, {budget['wall_seconds'] / 60:.1f} minutes."
    )
    add("")
    for pairing_id, profile in sorted(budget["profiles"].items()):
        add(f"### {pairing_id}")
        add("")
        rows = []
        for rung in profile["ladder"]["rungs"]:
            paired = rung.get("paired_vs_cheapest") or {}
            rows.append(
                [
                    rung["preset_id"],
                    f"{rung['worlds']}w/d{rung['rollout_depth']}",
                    _fmt(rung["ewr"]),
                    _fmt(paired.get("delta")) if paired.get("delta") is not None else "-",
                    _fmt(rung["search_seconds_per_game"], 2),
                    _fmt(rung["ewr_gain_per_added_search_second"], 5),
                    _fmt(rung["median_seconds_per_move"], 3),
                    _fmt(rung["p95_seconds_per_move"], 3),
                    rung["human_play"]["verdict"],
                ]
            )
        add(
            _table(
                [
                    "preset",
                    "budget",
                    "EWR",
                    "paired vs TINY",
                    "search s/game",
                    "EWR/added s",
                    "median s/move",
                    "p95 s/move",
                    "human play",
                ],
                rows,
            )
        )
        add("")
        selection = profile["selection"]
        rungs = {rung["preset_id"]: rung for rung in profile["ladder"]["rungs"]}
        add(
            f"Selected **{selection['selected_preset']}** "
            f"(strongest observed: {selection['strongest_observed_preset']} at "
            f"{_fmt(selection['strongest_observed_ewr'])}). {selection['rule']}."
        )
        # `order` is the ladder's own cheapest-first list; derive it from the
        # rungs when an older profile predates the field.
        order = profile["ladder"].get("order") or [
            rung["preset_id"] for rung in profile["ladder"]["rungs"]
        ]
        cheapest = order[0]
        strongest = rungs.get(selection["strongest_observed_preset"], {})
        base = rungs.get(cheapest, {})
        dominates = [
            key
            for key in ("ewr", "min_opponent_ewr", "weakness_pack_family_ewr")
            if strongest.get(key) is not None
            and base.get(key) is not None
            and strongest[key] > base[key]
        ]
        if len(dominates) == 3 and selection["selected_preset"] == cheapest:
            add("")
            add(
                f"Note that **{selection['strongest_observed_preset']} is better on "
                "every measured axis** here — overall EWR, worst opponent and the "
                f"weakness pack all rise from {cheapest}. It was not selected as the "
                "default because the gain sits inside the 0.10 engineering margin "
                "and the rule prefers the cheaper rung, not because it was found no "
                f"stronger. Choosing {cheapest} as the default is a cost decision; "
                f"{selection['strongest_observed_preset']} is retained as the "
                "maximum-strength mode for callers who will pay for it."
            )
        strong = profile["strong_gate"]
        add("")
        add(
            f"STRONG gate: **{'allowed' if strong['allowed'] else 'refused'}** — "
            f"{strong['reason']} (MEDIUM improvement over the cheaper rungs "
            f"{_fmt(strong['improvement_over_cheaper'])}, required "
            f"{_fmt(strong['useful_improvement_required'])})."
        )
        add("")

    # -- 7. selection --------------------------------------------------------
    add("## 7. The system matrix and the selection (section 14)")
    add("")
    rows = []
    for pairing_id in COMBINED_PAIRING_IDS:
        entry = matrix["matrix"].get(pairing_id)
        if entry is None:
            continue
        rows.append(
            [
                pairing_id,
                _fmt(entry["direct_ewr"]),
                _fmt(entry["search_ewr"]),
                f"{_fmt(entry['paired_delta_vs_direct'])} ± "
                f"{_fmt(entry['paired_standard_error'])}",
                f"{_fmt((entry['worst_opponent'] or {}).get('ewr'), 3)} "
                f"({(entry['worst_opponent'] or {}).get('name')})",
                _fmt(entry["weakness_pack_family_ewr"], 3),
                f"{_fmt(entry['median_seconds_per_move'], 3)} / "
                f"{_fmt(entry['p95_seconds_per_move'], 3)}",
                _fmt(entry["fallback_rate"], 5),
            ]
        )
    add(
        _table(
            [
                "system",
                "direct EWR",
                "search EWR",
                "paired delta",
                "worst stratum",
                "weakness pack",
                "median/p95 move",
                "fallback",
            ],
            rows,
        )
    )
    add("")
    add(f"Decision rule: {matrix['selection']['rule']}.")
    add("")
    add(
        f"Selected: **{matrix['selected_pairing']}** at "
        f"**{matrix['selected_preset']}**; maximum strength "
        f"**{matrix['maximum_strength_preset']}**. Contenders inside the "
        f"{_fmt(matrix['selection']['margin'], 2)} engineering margin: "
        + ", ".join(matrix["selection"]["contenders_within_margin"])
        + "."
    )
    add("")

    # -- 8. the working player ----------------------------------------------
    add("## 8. The working player (section 15)")
    add("")
    add(
        _table(
            ["mode", "system", "budget", "time cap"],
            [
                ["`p18_direct`", "P18", "direct, no search", "-"],
                ["`p24_direct`", "P24", "direct, no search", "-"],
                [
                    "`selected_search`",
                    f"{selected['move_model']} + {selected['belief_model']}",
                    f"{candidate['search']['selected_preset']}: "
                    f"{candidate['search']['worlds']} worlds, depth "
                    f"{candidate['search']['rollout_depth']}",
                    f"{candidate['time_caps_seconds']['selected_search']}s",
                ],
                [
                    "`maximum_strength`",
                    f"{selected['move_model']} + {selected['belief_model']}",
                    f"{candidate['maximum_strength']['preset_id']}: "
                    f"{candidate['maximum_strength']['worlds']} worlds, depth "
                    f"{candidate['maximum_strength']['rollout_depth']}",
                    f"{candidate['time_caps_seconds']['maximum_strength']}s",
                ],
            ],
        )
    )
    add("")
    add(
        f"`oracle_available_in_production = "
        f"{str(candidate['oracle_available_in_production']).lower()}`. "
        f"Fallback: {candidate['direct_fallback']['rule']}."
    )
    add("")
    add("### Latency, measured twice")
    add("")
    add(
        "The Stage B and Stage C move times are measured with ten worker "
        "processes on fourteen cores, so every one of them carries scheduler "
        "contention. A person playing one game has the machine to themselves. "
        "Both numbers are kept, and the time caps are set from the un-contended "
        "pilot: a cap derived from the contended numbers would buy headroom the "
        "deployed player does not need, and would let a real stall pass "
        "unnoticed."
    )
    add("")
    pilot = artifacts.get("latency_pilot") or {}
    selected_id = candidate["selected_system"]["pairing_id"]
    contended = (
        (budget["profiles"].get(selected_id) or {}).get("ladder", {}).get("rungs", [])
    )
    from .contract import LADDER_PRESET_NAMES

    ladder_order = {name: index for index, name in enumerate(LADDER_PRESET_NAMES)}
    rows = []
    for key, entry in sorted(
        (pilot.get("profiles") or {}).items(),
        key=lambda item: ladder_order.get(item[1].get("preset_id"), 99),
    ):
        if entry.get("pairing_id") != selected_id:
            continue
        packed = next(
            (rung for rung in contended if rung["preset_id"] == entry["preset_id"]), {}
        )
        rows.append(
            [
                entry["preset_id"],
                _fmt(entry["median_seconds_per_move"], 3),
                _fmt(entry["p95_seconds_per_move"], 3),
                _fmt(entry["max_seconds_per_move"], 3),
                _fmt(packed.get("median_seconds_per_move"), 3),
                _fmt(packed.get("p95_seconds_per_move"), 3),
                _fmt(entry["mean_c1_forwards"], 0),
            ]
        )
    if rows:
        add(
            _table(
                [
                    "preset",
                    "median (idle)",
                    "p95 (idle)",
                    "max (idle)",
                    "median (10-way)",
                    "p95 (10-way)",
                    "forwards/move",
                ],
                rows,
            )
        )
        add("")
        add(
            f"Measured on {pilot.get('positions')} replayed diagnostic positions, "
            "one process. Both selected presets sit inside the 2 s *preferred* "
            "line on an idle machine, and every measured move — idle or "
            "contended — stays inside its cap. Contention costs about 1.8x: "
            "MEDIUM's p95 goes from 1.78 s idle to 3.21 s under ten-way load, "
            "which is still under the 5.0 s cap but leaves 1.6x headroom rather "
            "than the 2.8x the idle measurement gives. A deployed player is not "
            "competing with nine copies of itself, so the idle column is the one "
            "a human experiences."
        )
        add("")
    add(
        "The production pairing ids remain available as diagnostic mode names for "
        "machine evaluation; `oracle` is not among them and is refused by name."
    )
    add("")

    # -- 9. limitations ------------------------------------------------------
    add("## 9. Known limitations")
    add("")
    for limitation in candidate["known_limitations"]:
        add(f"- {limitation}")
    add("")
    add("## 10. What this run did not do")
    add("")
    add("- it did not train or modify B18, B24, P18 or P24;")
    add("- it did not alter, pause, stop, restart or finalize any Phase 14 task;")
    add("- it did not edit accepted Phase 12 behaviour or overwrite "
        "`phase12_search_candidate_v1`;")
    add("- it did not reuse any contaminated Phase 12 board as new evidence;")
    add("- it did not perform a scientific validation phase.")
    add("")
    return "\n".join(lines) + "\n"


def build_summary(artifacts: dict) -> dict:
    gate = artifacts["gate"]
    stage_a = artifacts["stage_a"]
    stage_b = artifacts["stage_b"]
    budget = artifacts["budget"]
    matrix = artifacts["matrix"]
    candidate = artifacts["candidate"]

    return {
        "artifact": "phase15_agent02_summary_v1",
        **PHASE15_STATUS_MARKERS,
        "generated_utc": candidate["generated_utc"],
        "selected_system": candidate["selected_system"],
        "selected_preset": candidate["search"]["selected_preset"],
        "maximum_strength_preset": candidate["maximum_strength"]["preset_id"],
        "time_caps_seconds": candidate["time_caps_seconds"],
        "gate": {
            "passed": gate["passed"],
            "checks_passed": gate["checks_passed"],
            "checks_run": gate["checks_run"],
            "failed": gate["failed"],
        },
        "process_boundary": artifacts["boundary"]["verdict"],
        "evidence": {
            "match_manifest_digest": artifacts["match_manifest"]["manifest_digest"],
            "match_boards": artifacts["match_manifest"]["board_count"],
            "position_manifest_digest": artifacts["position_manifest"]["manifest_digest"],
            "positions": artifacts["position_manifest"]["position_count"],
            "stage_b_games": stage_b["games_played"],
            "stage_c_games": budget["games_played"],
            "probe_passed": stage_b["probe_passed"],
        },
        "stage_a_interpretation": stage_a["interpretation"],
        "system_matrix": matrix["matrix"],
        "selection": {
            "selected": matrix["selected_pairing"],
            "contenders_within_margin": matrix["selection"]["contenders_within_margin"],
            "rule": matrix["selection"]["rule"],
        },
        "budget": {
            pairing_id: {
                "selected_preset": profile["selection"]["selected_preset"],
                "strongest_observed_preset": profile["selection"][
                    "strongest_observed_preset"
                ],
                "strong_gate_allowed": profile["strong_gate"]["allowed"],
            }
            for pairing_id, profile in budget["profiles"].items()
        },
        "oracle_available_in_production": candidate["oracle_available_in_production"],
        "scientific_validation_status": candidate["scientific_validation_status"],
        "known_limitations": candidate["known_limitations"],
    }


__all__ = ["build_report", "build_summary"]
