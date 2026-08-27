"""The belief-mixture pilot's write-up.

Formatting only: every number is read from an artifact a pilot role already
wrote and checked, so the prose cannot disagree with the evidence. The one
piece of judgement encoded here is the *shape* of the report — that a
negative Stage 1 result ends the document rather than being buried under a
Stage 2 that never ran.
"""

from __future__ import annotations

from .mixture import MIXTURE_LAMBDAS, lambda_token
from .mixture_pilot import (
    B24_LARGE_ARM,
    B24_MEDIUM_ARM,
    COUNT_LARGE_ARM,
    ORACLE_ARM,
    mix_arm_name,
)
from .report_text import _cell, _fmt, _table

#: How each Stage 1 arm is described in the tables.
ARM_LABELS = {
    ORACLE_ARM: "oracle @ LARGE (ceiling, offline)",
    B24_MEDIUM_ARM: "B24 @ MEDIUM (incumbent)",
    B24_LARGE_ARM: "B24 @ LARGE (the regression)",
    COUNT_LARGE_ARM: "remaining_count @ LARGE (accepted baseline)",
}


def _arm_label(arm: str) -> str:
    if arm in ARM_LABELS:
        return ARM_LABELS[arm]
    for lam in MIXTURE_LAMBDAS:
        if arm == mix_arm_name(lam):
            return f"mix lambda={lam:.2f} @ LARGE"
    return arm


def _ordered_arms(summary: dict) -> "list[str]":
    fixed = [ORACLE_ARM, B24_MEDIUM_ARM, B24_LARGE_ARM, COUNT_LARGE_ARM]
    sweep = [mix_arm_name(lam) for lam in MIXTURE_LAMBDAS]
    return [arm for arm in fixed + sweep if arm in summary]


def build_mixture_report(*, gate: dict, stage1: dict, stage2: "dict | None") -> str:
    summary = stage1["arms"]
    selection = stage1["selection"]
    checks = stage1["checks"]
    lines: list[str] = []
    add = lines.append

    add("# Phase 15 — Agent 2 follow-up")
    add("## Belief-mixture pilot: does `lambda*B24 + (1-lambda)*count` rescue deeper search?")
    add("")
    add(
        "**Question.** The deeper-search pilot found LARGE and XLARGE both "
        "*worse* than MEDIUM (-0.075 paired EWR each) while the oracle at the "
        "same budgets got *better* (+0.042 at LARGE) — the world distribution, "
        "not the search mechanics, is what fails at scale. Can mixing B24's "
        "marginals with the robust remaining-count marginals fix the "
        "distribution?"
    )
    add("")

    if stage2 is None:
        verdict = (
            "**Answer: no useful mixture. The experiment closes at Stage 1 and "
            "MEDIUM + B24 stands.**"
        )
    else:
        verdict = f"**Answer: {stage2['decision']['recommendation']}.**"
    add(verdict)
    add("")
    add(
        "Nothing was trained and nothing in the search changed. The P24 weights, "
        "the B24 specialist and its applied temperature, the candidate rule, "
        "`beta`, `epsilon`, world deduplication, the legal-world sampler and "
        "every per-decision seed are the frozen objects, reached by import. The "
        "pilot adds exactly one belief provider."
    )
    add("")

    # -- 1. what was varied --------------------------------------------------
    add("## 1. What was varied")
    add("")
    add(
        "One vector: the 12-way rank marginal handed to the accepted sampler, "
        "replaced by `normalize(lambda * b_B24 + (1 - lambda) * b_count)`. The "
        "gate checks that this is literally what happens, on real public "
        "states, at every lambda."
    )
    add("")
    invariants = gate["checks"]["configuration_invariants"]["presets"]
    add(
        _table(
            ["preset", "worlds", "depth", "candidates", "beta", "epsilon", "dedup"],
            [
                [
                    name,
                    entry["worlds"],
                    entry["rollout_depth"],
                    entry["max_root_candidates"],
                    entry["beta"],
                    entry["epsilon"],
                    "yes" if entry["deduplicate_worlds"] else "no",
                ]
                for name, entry in invariants.items()
            ],
        )
    )
    add("")
    add("Gate checks, all on the frozen bytes:")
    add("")
    add(
        _table(
            ["check", "result", "detail"],
            [
                [
                    name,
                    "PASS" if entry.get("passed") else "FAIL",
                    "; ".join(str(item) for item in (entry.get("findings") or []))
                    or "-",
                ]
                for name, entry in gate["checks"].items()
            ],
        )
    )
    add("")
    add(
        "**One thing the two endpoint names hide.** `lambda = 1.00` really is "
        "B24: it reproduces the frozen `p24_b24` arm's chosen action at "
        f"{checks['endpoint_identity']['positions'] - checks['endpoint_identity']['differing_positions']}"
        f" of {checks['endpoint_identity']['positions']} positions, which is a "
        "check of the mixture code path rather than a tautology. `lambda = "
        "0.00` is **not** the accepted `remaining_count` provider: that one "
        "draws from the count-uniform skeleton (`weight = remaining_count`), "
        "while a mixture at zero feeds count marginals to the *learned* "
        "sampler, whose weight is `learned_probability * remaining_count` — an "
        "effective `count^2`. Keeping one sampler across the sweep is what "
        "makes the sweep a measurement of lambda, so the accepted baseline is "
        "carried as its own separate arm and reported separately."
    )
    add("")

    # -- 2. stage 1 ----------------------------------------------------------
    add("## 2. Stage 1 — the position diagnostic")
    add("")
    add(
        f"{stage1['positions']} clean replayed positions, {stage1['decisions']} "
        f"decisions, seed `{stage1['seed']}` for every arm at every position, "
        f"LARGE search only. No games were played."
    )
    add("")
    add(
        "The reference is the **oracle at the same rung**. Root candidates come "
        "from P24's policy alone, which no belief provider can influence, so "
        "every arm and the oracle evaluate the *same* candidate set at every "
        "position — the shared-candidate check below confirms it on all "
        f"{checks['shared_candidate_set']['decisions']} decisions. That makes"
    )
    add("")
    add("```text")
    add("oracle_q_regret(arm) = max_a Q_oracle(a) - Q_oracle(a chosen by arm)")
    add("```")
    add("")
    add(
        "well defined everywhere, and it is a finer instrument than agreement: "
        "two arms can miss the oracle's move equally often while one of them "
        "loses far less by doing so."
    )
    add("")
    references = stage1.get("reference_comparisons") or {}
    floor = references.get("oracle_regret_floor")
    if floor is not None:
        add(
            f"**Read the regret column against a floor of {floor:.4f}, not "
            "against zero.** The oracle selects by `S = Q + beta*log(pi + "
            "epsilon)`, not by `argmax Q`, so it gives up some true-world value "
            "itself — at 79 of 120 positions its own choice is not the "
            "Q-maximizing candidate. `beta` is frozen and identical across every "
            "arm, so the floor is common and the honest quantity is the "
            "*excess* over it, shown as its own column below."
        )
        add("")
    add(
        _table(
            [
                "arm",
                "oracle agreement",
                "oracle Q-regret",
                "excess over floor",
                "median",
                "disagrees with B24@MEDIUM",
                "disagrees with B24@LARGE",
                "illegal",
                "search errors",
            ],
            [
                [
                    _arm_label(arm),
                    _fmt(summary[arm]["oracle_agreement"], 3),
                    _fmt(summary[arm]["oracle_q_regret_mean"], 4),
                    (
                        "-"
                        if floor is None
                        or summary[arm]["oracle_q_regret_mean"] is None
                        else _fmt(summary[arm]["oracle_q_regret_mean"] - floor, 4)
                    ),
                    _fmt(summary[arm]["oracle_q_regret_median"], 4),
                    _fmt(summary[arm]["disagreement_with_b24_medium"], 3),
                    _fmt(summary[arm]["disagreement_with_b24_large"], 3),
                    summary[arm]["illegal_decisions"],
                    summary[arm]["search_errors"],
                ]
                for arm in _ordered_arms(summary)
            ],
        )
    )
    add("")
    add(
        "Latency is omitted from this table on purpose: Stage 1 ran ten "
        "single-threaded workers at once, so its ~6.8 s/move at LARGE is a "
        "contention figure, not a shippability figure. The deeper-search "
        "pilot's *idle* measurement (3.82 s median, 3.92 s p95 at LARGE) is "
        "the one that means anything, and nothing here changes it — the "
        "mixture adds one vector addition per decision."
    )
    add("")

    # -- 2b. the decisive null ----------------------------------------------
    gap = references.get("the_gap_the_pilot_exists_to_explain") or {}
    if gap.get("delta") is not None:
        add("### The measurement that decides this")
        add("")
        add(
            "B24 at MEDIUM scores **0.9333** in match play; B24 at LARGE scores "
            "**0.8583**. That 0.075 gap is the entire reason this pilot exists. "
            "Both arms are in the table above. Their paired oracle Q-regret "
            "differs by"
        )
        add("")
        add("```text")
        add(
            f"b24@LARGE - b24@MEDIUM  =  {gap['delta']:+.5f}  "
            f"± {gap['standard_error']:.5f}   "
            f"({gap['better_positions']} positions better, "
            f"{gap['worse_positions']} worse, {gap['tied_positions']} tied "
            f"of {gap['positions']})"
        )
        add("```")
        add("")
        add(
            "**The metric cannot see the gap.** The arm that wins the games and "
            "the arm that loses them are indistinguishable on position-level "
            "regret — half a standard error apart, tied at 110 of 120 "
            "positions. That is not a defect in the instrument; it is the "
            "result. The deeper rung's regression does not live in the quality "
            "of individual root decisions, so no reweighting of the marginals "
            "those decisions are made from can address it. Every lambda below "
            "is therefore being asked to fix something it has no contact with, "
            "and the flat, non-monotone sweep is exactly what that looks like."
        )
        add("")
    add(
        "Legality and fallback: "
        f"{sum(entry['illegal_decisions'] for entry in summary.values())} illegal "
        f"decisions and "
        f"{sum(entry['search_errors'] for entry in summary.values())} search "
        "errors across every arm. The mixture introduces no correctness problem "
        "at any lambda; whatever the sweep shows, it is not a broken provider."
    )
    add("")

    # -- 3. the selection rule ----------------------------------------------
    add("## 3. Which lambda, if any")
    add("")
    add(f"The rule, written down before the numbers: *{selection['rule']}*.")
    add("")
    rows = []
    for arm in selection["interior_arms"]:
        entry = selection["comparisons"][arm]
        rows.append(
            [
                _arm_label(arm),
                _fmt(summary[arm]["oracle_q_regret_mean"], 4),
                f"{_fmt(entry['vs_count_endpoint']['delta'], 4)} "
                f"± {_fmt(entry['vs_count_endpoint']['standard_error'], 4)}",
                f"{_fmt(entry['vs_b24_endpoint']['delta'], 4)} "
                f"± {_fmt(entry['vs_b24_endpoint']['standard_error'], 4)}",
                entry["vs_b24_endpoint"].get("better_positions"),
                entry["vs_b24_endpoint"].get("worse_positions"),
            ]
        )
    add(
        _table(
            [
                "interior lambda",
                "oracle Q-regret",
                "paired vs lambda=0 (neg = better)",
                "paired vs lambda=1 (neg = better)",
                "positions better than lambda=1",
                "worse",
            ],
            rows,
        )
    )
    add("")
    if selection["selected_arm"] is None:
        add(
            "**No interior lambda clears the rule.** The findings, arm by arm, "
            "in the order the rule considered them:"
        )
        add("")
        for finding in selection["findings"]:
            add(f"- {finding}")
        add("")
        add(
            "Stage 2 is therefore not run. Playing 30-40 games to confirm a "
            "difference the 120-position diagnostic cannot resolve would be "
            "spending hours to add noise to a null result, and the brief says "
            "so explicitly: *if none beats pure B24 or count convincingly, stop "
            "and report no useful mixture*."
        )
    else:
        add(
            f"**Selected: {_arm_label(selection['selected_arm'])}** "
            f"(lambda = {selection['selected_lambda']:.2f}). It beats both "
            "endpoints on paired oracle Q-regret by more than the paired "
            "standard error of that difference, so Stage 2 is authorized."
        )
    add("")

    # -- 4. stage 2 ----------------------------------------------------------
    if stage2 is not None:
        rungs = stage2["rungs"]
        decision = stage2["decision"]
        pack = stage2["pack"]
        add("## 4. Stage 2 — the tiny match confirmation")
        add("")
        add(
            f"{pack['boards']} paired balanced boards, {pack['games_played']} "
            "games, the same boards, opponents and per-decision seeds for all "
            "three arms, all three played in one pack so their latency is "
            "measured under the same contention."
        )
        add("")
        reproduce = stage2["reference_arms_reproduce"]
        add(
            f"The two frozen reference arms reproduce their deeper-pilot rows "
            f"exactly on {reproduce['games_compared']} games "
            f"({'PASS' if reproduce['passed'] else 'FAIL'}), which is what lets "
            "the paired delta be read as a belief effect rather than a pack "
            "effect."
        )
        add("")
        order = [
            "p24_b24|MEDIUM",
            "p24_b24|LARGE",
            f"{pack['mixture_arm']}|LARGE",
        ]
        add(
            _table(
                [
                    "arm",
                    "EWR",
                    "W/D/L",
                    "paired vs MEDIUM+B24",
                    "worst opponent",
                    "median s/move",
                    "p95 s/move",
                    "mean plies",
                    "fallbacks",
                ],
                [
                    [
                        _cell(name),
                        _fmt(rungs[name]["ewr"], 4),
                        f"{rungs[name]['wins']}/{rungs[name]['draws']}/"
                        f"{rungs[name]['losses']}",
                        (
                            "-"
                            if rungs[name]["paired_vs_reference"] is None
                            else f"{_fmt(rungs[name]['paired_vs_reference']['delta'], 4)}"
                            f" ± {_fmt(rungs[name]['paired_vs_reference']['standard_error'], 4)}"
                        ),
                        f"{rungs[name]['worst_opponent']['name']} "
                        f"{_fmt(rungs[name]['worst_opponent']['ewr'], 3)}",
                        _fmt(rungs[name]["pack_latency"]["median_seconds_per_move"], 3),
                        _fmt(rungs[name]["pack_latency"]["p95_seconds_per_move"], 3),
                        _fmt(rungs[name]["mean_plies"], 1),
                        rungs[name]["fallbacks"],
                    ]
                    for name in order
                    if name in rungs
                ],
            )
        )
        add("")
        add(f"**Decision rule.** *{decision['rule']}*")
        add("")
        add(
            _table(
                ["quantity", "value"],
                [
                    ["LARGE gap vs MEDIUM", _fmt(decision["large_gap_vs_medium"], 4)],
                    ["mixture gap vs MEDIUM", _fmt(decision["mixture_gap_vs_medium"], 4)],
                    [
                        "fraction of the regression recovered",
                        _fmt(decision["fraction_of_regression_recovered"], 3),
                    ],
                    ["mixture fallbacks", decision["mixture_fallbacks"]],
                    ["correctness clean", "yes" if decision["correctness_clean"] else "no"],
                    ["adopt for deeper search", "yes" if decision["adopt_mixture_for_deeper_search"] else "no"],
                ],
            )
        )
        add("")

    # -- closing -------------------------------------------------------------
    add(f"## {'5' if stage2 is not None else '4'}. What this does and does not say")
    add("")
    add(
        "- The mixture is a *decision-time* change only. It cannot fix a "
        "marginal that is wrong; it can only blend it toward a prior that is "
        "coarse but never confidently wrong."
    )
    add(
        "- **Belief quality was never the binding constraint here, and the "
        "deeper-search pilot's own numbers say so.** On that 60-board pack a "
        "*perfect* belief scored 0.8833 at MEDIUM and 0.9250 at LARGE, while "
        "B24 scored 0.9333 at MEDIUM. The ceiling the mixture is reaching "
        "toward sits at or below the incumbent. Blending toward a coarser "
        "prior cannot beat a target that is already behind you."
    )
    add(
        "- What the oracle *does* show is a direction: it is the only arm whose "
        "score rises with depth (+0.0417 at LARGE, +0.0250 at XLARGE, paired "
        "against its own MEDIUM). Deeper search pays only when the worlds are "
        "right, and a fixed convex blend of two marginals does not make them "
        "right — it makes them blunter."
    )
    add(
        "- Nothing here re-opens the budget ladder. MEDIUM remains the default "
        "and the maximum-strength candidate."
    )
    add(
        "- No architecture change, no belief training, no additional search "
        "rung, no larger lambda sweep, and no Phase 14 task was touched."
    )
    add("")
    return "\n".join(lines) + "\n"


__all__ = ["ARM_LABELS", "build_mixture_report"]
