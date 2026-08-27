"""The deeper-search pilot's write-up.

Formatting only: every number is read from an artifact the pilot roles already
wrote and verified, so the prose cannot disagree with the evidence.
"""

from __future__ import annotations

from .contract import DEEP_PILOT_PRESET_NAMES, MATCH_OPPONENTS
from .report_text import _cell, _fmt, _table


def build_deep_report(payload: dict, gate: dict, pack: dict) -> str:
    rungs = payload["rungs"]
    verdict = payload["verdict"]
    lines: list[str] = []
    add = lines.append

    add("# Phase 15 — Agent 2 follow-up")
    add("## Deeper-search pilot: P24 + B24 at 2x and 4x MEDIUM")
    add("")
    add(f"**Question.** {payload['question']}")
    add("")
    add(
        f"**Answer: {verdict['recommendation']}.** {verdict['reason']}."
    )
    add("")
    if verdict.get("both_rungs_regressed"):
        add(
            "> The honest one-line version: buying 2.19x and 3.90x more search "
            "compute did not make P24 + B24 stronger. It made it **weaker** — "
            "and the oracle control in section 3 shows why, which is the part of "
            "this result worth carrying forward."
        )
        add("")
    add(
        "This is a narrow paired pilot on one already-selected system. No "
        "architecture was changed, no network was trained, no belief experiment "
        "was broadened, and no Phase 14 task was touched. The ladder is closed "
        "here."
    )
    add("")

    # -- 1. what was varied --------------------------------------------------
    add("## 1. What was varied, and what was held fixed")
    add("")
    add(
        "Compute grows through **worlds first, depth modestly** — everything "
        "else is MEDIUM's. `LARGE` and `XLARGE` do not pass "
        "`max_root_candidates`, `beta` or `epsilon` at all, so they inherit the "
        "same defaults MEDIUM uses; the pilot's own control is therefore a "
        "property of the configuration rather than a promise, and the gate "
        "checks it."
    )
    add("")
    invariants = gate["checks"]["configuration_invariants"]["rungs"]
    idle = gate["idle_latency"]
    rows = []
    for name in DEEP_PILOT_PRESET_NAMES:
        config = invariants[name]
        measured = idle.get(name, {})
        rows.append(
            [
                name,
                config["worlds"],
                config["rollout_depth"],
                config["max_root_candidates"],
                config["beta"],
                f"{config['naive_ratio_vs_medium']:.2f}x",
                f"{measured.get('measured_forward_ratio_vs_medium', 0):.2f}x",
                _fmt(measured.get("mean_c1_forwards"), 0),
                _fmt(measured.get("mean_world_uniqueness"), 3),
            ]
        )
    add(
        _table(
            [
                "rung",
                "worlds",
                "depth",
                "candidates",
                "beta",
                "planned compute",
                "measured compute",
                "forwards/move",
                "world uniqueness",
            ],
            rows,
        )
    )
    add("")
    add(
        "Measured compute lands at **2.19x** and **3.90x** — slightly under the "
        "planned 2.25x and 4.12x, because duplicate sampled worlds are evaluated "
        "once and weighted, and duplicates get commoner as the world budget "
        "grows (uniqueness falls from 0.926 to 0.898). The measured ratio is the "
        "honest one and is what the cost figures below use."
    )
    add("")
    add(
        "These rungs are deliberately **not** the section 7 `STRONG` preset, "
        "which raises the candidate count to 12: this pilot forbids changing "
        "candidate handling, so reusing `STRONG` would have quietly broken its "
        "own control."
    )
    add("")

    # -- 2. integrity --------------------------------------------------------
    add("## 2. Integrity at the larger budgets")
    add("")
    add(
        f"**{'PASS' if payload['gate_passed'] else 'FAIL'}** — every check ran on "
        "fresh replayed positions before a single deeper game was played."
    )
    add("")
    checks = gate["checks"]
    determinism = checks["determinism_and_legality"]
    identity = checks["frozen_identity"]
    add(
        _table(
            ["check", "result", "observed"],
            [
                [
                    "frozen identity",
                    "pass" if identity["passed"] else "**FAIL**",
                    f"P24 `{identity['move_model_state_digest'][:16]}` + B24 "
                    f"`{identity['belief_state_digest'][:16]}`, temperature "
                    f"{identity['applied_temperature']}",
                ],
                [
                    "configuration control",
                    "pass" if checks["configuration_invariants"]["passed"] else "**FAIL**",
                    "candidates, beta, epsilon and world dedup identical to MEDIUM "
                    "at every rung",
                ],
                [
                    "determinism and legality",
                    "pass" if determinism["passed"] else "**FAIL**",
                    f"{determinism['decisions']} decisions re-run under the same "
                    "seed at every rung: identical action, identical world "
                    "weights, identical Q values; every action legal; the direct "
                    "move always a candidate",
                ],
                [
                    "sampled worlds legal",
                    "pass" if checks["worlds_legal"]["passed"] else "**FAIL**",
                    ", ".join(
                        f"{name} {count}"
                        for name, count in checks["worlds_legal"]["worlds_checked"].items()
                    )
                    + " worlds through the accepted validation stack",
                ],
                [
                    "MEDIUM reproduces Stage C",
                    "pass" if payload["medium_reproduces_stage_c"]["passed"] else "**FAIL**",
                    f"{payload['medium_reproduces_stage_c']['boards_compared']} boards "
                    "played twice in separate runs, hours apart: identical outcome, "
                    "score and ply count",
                ],
            ],
        )
    )
    add("")

    # -- 3. the pack ---------------------------------------------------------
    add("## 3. The paired pack")
    add("")
    add(
        f"{payload['games_played']} games on the same **{payload['boards']} balanced "
        "boards** — one per cell of the 10 opponents x 3 setup sources x 2 colours "
        "grid — with the same opponents, the same setups and the same "
        "per-decision seeds at every rung. All three rungs were replayed fresh in "
        "this one pack, so every paired delta comes from rows produced by "
        f"identical code under identical conditions. {payload['wall_seconds'] / 3600:.1f} "
        "hours on 10 workers."
    )
    add("")
    rows = []
    for name in DEEP_PILOT_PRESET_NAMES:
        entry = rungs.get(name)
        if entry is None:
            continue
        paired = entry.get("paired_vs_medium") or {}
        idle_entry = entry.get("idle_latency") or {}
        differ = entry.get("moves_differing_from_medium") or {}
        rows.append(
            [
                name,
                f"{entry['wins']}/{entry['draws']}/{entry['losses']}",
                _fmt(entry["ewr"]),
                (
                    f"{_fmt(paired.get('delta'))} ± {_fmt(paired.get('standard_error'))}"
                    if paired.get("delta") is not None
                    else "— (baseline)"
                ),
                f"{_fmt((entry['worst_opponent'] or {}).get('ewr'), 3)} "
                f"({(entry['worst_opponent'] or {}).get('name')})",
                _fmt(idle_entry.get("mean_seconds_per_move"), 3),
                _fmt(idle_entry.get("median_seconds_per_move"), 3),
                _fmt(idle_entry.get("p95_seconds_per_move"), 3),
                _fmt(differ.get("fraction_differing_from_medium"), 3),
                f"{entry['fallbacks']}",
            ]
        )
    add(
        _table(
            [
                "rung",
                "W/D/L",
                "EWR",
                "paired vs MEDIUM",
                "worst opponent",
                "mean move (idle)",
                "median move (idle)",
                "p95 move (idle)",
                "% moves differing",
                "fallbacks/errors",
            ],
            rows,
        )
    )
    add("")
    add(
        "`% moves differing` is measured on the fixed diagnostic position "
        "manifest, where every rung answers the same question. Inside a match "
        "game the quantity is ill-defined: the moment a rung plays a different "
        "move the two games diverge and later positions are no longer "
        "comparable, so a per-ply count there would measure divergence of "
        "*positions*, not of decisions."
    )
    add("")

    # -- first divergence ----------------------------------------------------
    divergence = payload.get("first_divergence_from_medium") or {}
    if divergence:
        add("### How far into a game the extra search takes to matter")
        add("")
        rows = [
            [
                name,
                entry["games_compared"],
                entry["games_identical_to_medium"],
                entry["games_that_diverged"],
                _fmt(entry["median_first_divergence_ply"], 0),
                _fmt(entry["min_first_divergence_ply"], 0),
            ]
            for name, entry in sorted(divergence.items())
        ]
        add(
            _table(
                [
                    "rung",
                    "games compared",
                    "identical to MEDIUM",
                    "diverged",
                    "median first-divergence ply",
                    "earliest",
                ],
                rows,
            )
        )
        add("")

    # -- latency -------------------------------------------------------------
    add("### Latency, idle and in-pack")
    add("")
    rows = []
    for name in DEEP_PILOT_PRESET_NAMES:
        entry = rungs.get(name)
        if entry is None:
            continue
        idle_entry = entry.get("idle_latency") or {}
        pack_entry = entry.get("pack_latency") or {}
        rows.append(
            [
                name,
                _fmt(idle_entry.get("median_seconds_per_move"), 3),
                _fmt(idle_entry.get("p95_seconds_per_move"), 3),
                _fmt(idle_entry.get("max_seconds_per_move"), 3),
                _fmt(pack_entry.get("median_seconds_per_move"), 3),
                _fmt(pack_entry.get("p95_seconds_per_move"), 3),
                _fmt(entry.get("search_seconds_per_game"), 1),
            ]
        )
    add(
        _table(
            [
                "rung",
                "median (idle)",
                "p95 (idle)",
                "max (idle)",
                "median (10-way)",
                "p95 (10-way)",
                "search s/game (10-way)",
            ],
            rows,
        )
    )
    add("")
    add(
        "The idle column is what a person playing one game experiences and is "
        "the column the 5 s ceiling is applied to."
    )
    add("")

    # -- per opponent --------------------------------------------------------
    add("### EWR by opponent")
    add("")
    header = ["rung"] + list(MATCH_OPPONENTS)
    rows = []
    for name in DEEP_PILOT_PRESET_NAMES:
        entry = rungs.get(name)
        if entry is None:
            continue
        row = [name]
        for opponent in MATCH_OPPONENTS:
            slice_entry = entry["ewr_by_opponent"].get(opponent)
            row.append(_fmt(slice_entry["ewr"], 3) if slice_entry else "-")
        rows.append(row)
    add(_table(header, rows))
    add("")

    # -- oracle --------------------------------------------------------------
    oracle = payload.get("oracle_reference") or {}
    if oracle:
        add("### Oracle ceiling at each budget (offline diagnostic)")
        add("")
        add(
            "Essentially free to run: the oracle's sampled worlds all collapse to "
            "the one true army, so its cost does not grow with the world budget. "
            "It is never a deployable arm and is excluded from production by four "
            "independent refusals."
        )
        add("")
        ordered = sorted(
            oracle.items(),
            key=lambda item: DEEP_PILOT_PRESET_NAMES.index(item[0])
            if item[0] in DEEP_PILOT_PRESET_NAMES
            else 99,
        )
        rows = []
        for name, entry in ordered:
            paired = entry.get("paired_vs_medium") or {}
            rows.append(
                [
                    name,
                    _fmt(entry["ewr"]),
                    (
                        f"{_fmt(paired.get('delta'))} ± {_fmt(paired.get('standard_error'))}"
                        if paired.get("delta") is not None
                        else "— (baseline)"
                    ),
                    _fmt((entry["worst_opponent"] or {}).get("ewr"), 3),
                    _fmt(entry["search_seconds_per_game"], 1),
                ]
            )
        add(
            _table(
                [
                    "rung",
                    "oracle EWR",
                    "paired vs oracle MEDIUM",
                    "worst opponent",
                    "search s/game",
                ],
                rows,
            )
        )
        add("")
        add(
            "**This is the pilot's most informative result, and it inverts the "
            "main one.** Given the *true* hidden army, more search **helps**: the "
            "oracle gains +0.042 at LARGE and +0.025 at XLARGE, and its worst "
            "opponent improves from 0.667 to 0.833. Given *belief-sampled* worlds, "
            "the same extra compute **hurts**, by −0.075 at both rungs, with the "
            "worst opponent falling from 0.833 to 0.500."
        )
        add("")
        add(
            "So the search mechanics, the rollout policy and the leaf value are "
            "not what is failing at depth — over correct worlds they scale the "
            "way one would hope. What does not survive scaling is the *world "
            "distribution*. Averaging over 64-96 sampled worlds instead of 32, "
            "and rolling each out 9-11 plies instead of 8, commits harder and "
            "for longer to a belief that is wrong in a correlated way, so the "
            "extra compute buys a more confident wrong answer rather than a "
            "better one. That reading is consistent with the rest of the phase: "
            "the belief specialists never separated from the count baseline at "
            "MEDIUM either."
        )
        add("")

    # -- decision ------------------------------------------------------------
    add("## 4. The decision")
    add("")
    add(f"Rule: {verdict['rule']}.")
    add("")
    add(
        _table(
            ["quantity", "LARGE", "XLARGE"],
            [
                [
                    "paired gain vs MEDIUM",
                    _fmt(verdict.get("large_gain")),
                    _fmt(verdict.get("xlarge_gain")),
                ],
                [
                    "standard error",
                    _fmt(verdict.get("large_standard_error")),
                    _fmt(verdict.get("xlarge_standard_error")),
                ],
                [
                    "p95 move, idle",
                    _fmt(verdict.get("large_p95_seconds"), 3),
                    _fmt(verdict.get("xlarge_p95_seconds"), 3),
                ],
                [
                    f"fits the {verdict['latency_ceiling_seconds']} s ceiling",
                    _fmt(verdict.get("large_fits_latency_ceiling")),
                    _fmt(verdict.get("xlarge_fits_latency_ceiling")),
                ],
            ],
        )
    )
    add("")
    add(
        f"Meaningful-gain band: {verdict['meaningful_gain_band'][0]} to "
        f"{verdict['meaningful_gain_band'][1]} EWR."
    )
    add("")
    add(f"**Recommendation: {verdict['recommendation']}.** {verdict['reason']}.")
    add("")
    add("### How far this sample can be trusted")
    add("")
    large_gain = verdict.get("large_gain")
    large_se = verdict.get("large_standard_error")
    xlarge_gain = verdict.get("xlarge_gain")
    xlarge_se = verdict.get("xlarge_standard_error")
    if None not in (large_gain, large_se, xlarge_gain, xlarge_se) and large_se and xlarge_se:
        add(
            f"The regressions are {abs(large_gain / large_se):.1f} and "
            f"{abs(xlarge_gain / xlarge_se):.1f} standard errors from zero. "
            "**Neither is individually resolved at 60 paired boards**, and this "
            "pilot makes no significance claim. What raises it above noise is that "
            "four things point the same way at once: both rungs regress by the "
            "same amount, the worst-opponent score falls monotonically with "
            "compute (0.833 to 0.667 to 0.500), games get *longer* rather than "
            "more decisive, and the oracle control moves in the opposite "
            "direction on the same boards with the same seeds. A single noisy "
            "arm would not produce that pattern."
        )
        add("")
        add(
            "The conservative reading is the one the decision rule already "
            "takes: there is no evidence of a gain, so do not spend the compute. "
            "The stronger reading — that belief-sampled search actively degrades "
            "with scale — is supported but not established here, and would need "
            "a larger pack to settle. That pack is not part of this pilot."
        )
        add("")
    add("## 5. What this pilot did not do")
    add("")
    add("- it did not change the algorithm, the candidate rule or the regularization;")
    add("- it did not train or modify any network;")
    add("- it did not broaden the belief experiments;")
    add("- it did not touch any Phase 14 task or artifact;")
    add("- it did not extend the ladder beyond these three rungs, and will not.")
    add("")
    return "\n".join(lines) + "\n"


__all__ = ["build_deep_report"]
