"""Phase 16 Agent 3: the predeclared decision rules of section 5.

```text
adopt_recipe    max(B, C) final-hour benchmark EWR >= A + 0.03
setups_causal   C adversarial-stratum EWR >= B + 0.03
plateau_check   report each arm's h4->h6 slope; a flat B/C with a passing h6
                still adopts, but the report must say the plateau moved, not
                vanished
stop_rule       if neither B nor C clears adopt_recipe: STOP
```

These are applied here, mechanically, to the recorded hour curves. The module
computes a verdict; it does not soften one. `stop_rule` firing is a result,
not a failure of the phase, and the function returns it in the same shape as
an adoption so a reader cannot tell from the plumbing which way it went.

Standard errors, not significance
---------------------------------
Overview section 6: these are engineering packs. Every EWR is reported with
the binomial standard error of its own pack and no test is run against it. The
0.03 margins are *decision* thresholds fixed before the runs, which is what
makes applying them afterwards legitimate.
"""

from __future__ import annotations

import hashlib
import json
import math

from .contract import (
    ADOPT_RECIPE_MARGIN,
    EVALUATION_HOURS,
    SETUPS_CAUSAL_MARGIN,
    Phase16TrainingError,
)

CONTROL_ARM = "a_control"
DAMPED_ARM = "b_damped"
DAMPED_PLUS_ARM = "c_damped_plus"

VERDICT_ADOPT = "ADOPT"
VERDICT_STOP = "STOP"
VERDICT_INCOMPLETE = "INCOMPLETE"


def standard_error(ewr: float, games: int) -> float:
    """The binomial SE of one pack's EWR. Reported, never tested against."""
    if not games:
        return 0.0
    value = min(max(float(ewr), 0.0), 1.0)
    return math.sqrt(max(value * (1.0 - value), 0.0) / float(games))


def _hour_entry(curves: dict, arm: str, hour: int) -> "dict | None":
    return (curves.get(arm) or {}).get(str(int(hour)))


def arm_curve(curves: dict, arm: str) -> list:
    """One arm's recorded hours, in order, with SEs attached."""
    rows = []
    for hour in EVALUATION_HOURS:
        entry = _hour_entry(curves, arm, hour)
        if entry is None:
            continue
        benchmark = entry.get("benchmark") or {}
        adversarial = entry.get("adversarial") or {}
        rows.append(
            {
                "hour": int(hour),
                "iteration": entry.get("iteration"),
                "optimizer_step": entry.get("optimizer_step"),
                "source": entry.get("source"),
                "benchmark_pack": benchmark.get("pack"),
                "benchmark_subset": benchmark.get("subset"),
                "benchmark_games": benchmark.get("games"),
                "benchmark_ewr": benchmark.get("ewr"),
                "benchmark_se": standard_error(
                    benchmark.get("ewr") or 0.0, benchmark.get("games") or 0
                ),
                "adversarial_pack": adversarial.get("pack"),
                "adversarial_stratum": adversarial.get("stratum"),
                "adversarial_games": adversarial.get("games"),
                "adversarial_ewr": adversarial.get("ewr"),
                "adversarial_se": standard_error(
                    adversarial.get("ewr") or 0.0, adversarial.get("games") or 0
                ),
            }
        )
        full = entry.get("benchmark_full") or {}
        if full:
            rows[-1].update(
                {
                    "benchmark_full_games": full.get("games"),
                    "benchmark_full_ewr": full.get("ewr"),
                    "benchmark_full_se": standard_error(
                        full.get("ewr") or 0.0, full.get("games") or 0
                    ),
                }
            )
        strata = entry.get("adversarial_strata") or {}
        if strata:
            rows[-1]["adversarial_strata"] = {
                name: {
                    "ewr": value.get("ewr"),
                    "games": value.get("games"),
                    "se": standard_error(value.get("ewr") or 0.0, value.get("games") or 0),
                }
                for name, value in strata.items()
            }
    return rows


def final_hour(curves: dict, arm: str) -> "dict | None":
    rows = arm_curve(curves, arm)
    return rows[-1] if rows else None


def plateau_slope(curves: dict, arm: str, *, key: str = "benchmark_ewr") -> "dict | None":
    """The h4 -> h6 slope section 5's `plateau_check` asks each arm to report."""
    early, late = _hour_entry(curves, arm, 4), _hour_entry(curves, arm, 6)
    if early is None or late is None:
        return None
    field = "benchmark" if key.startswith("benchmark") else "adversarial"
    first = (early.get(field) or {}).get("ewr")
    second = (late.get(field) or {}).get("ewr")
    if first is None or second is None:
        return None
    return {
        "h4": first,
        "h6": second,
        "delta": round(float(second) - float(first), 5),
        "per_hour": round((float(second) - float(first)) / 2.0, 5),
        "flat": abs(float(second) - float(first)) < 0.02,
    }


def decide_recipe(curves: dict, configs: "dict | None" = None) -> dict:
    """Apply section 5's rules to the recorded curves and return the verdict."""
    configs = configs or {}
    arms = {arm: arm_curve(curves, arm) for arm in (CONTROL_ARM, DAMPED_ARM, DAMPED_PLUS_ARM)}
    finals = {arm: (rows[-1] if rows else None) for arm, rows in arms.items()}
    missing = [arm for arm, entry in finals.items() if entry is None]

    control = finals[CONTROL_ARM]
    damped = finals[DAMPED_ARM]
    damped_plus = finals[DAMPED_PLUS_ARM]

    adopt = {
        "rule": f"max(B, C) final-hour benchmark EWR >= A + {ADOPT_RECIPE_MARGIN}",
        "margin": ADOPT_RECIPE_MARGIN,
        "control_ewr": control["benchmark_ewr"] if control else None,
        "threshold": (
            round(float(control["benchmark_ewr"]) + ADOPT_RECIPE_MARGIN, 5)
            if control and control["benchmark_ewr"] is not None
            else None
        ),
        "candidates": {
            arm: (finals[arm]["benchmark_ewr"] if finals[arm] else None)
            for arm in (DAMPED_ARM, DAMPED_PLUS_ARM)
        },
    }
    winner = None
    if adopt["threshold"] is not None:
        clearing = {
            arm: value
            for arm, value in adopt["candidates"].items()
            if value is not None and float(value) >= adopt["threshold"]
        }
        adopt["clearing"] = sorted(clearing)
        if clearing:
            winner = max(clearing, key=lambda arm: float(clearing[arm]))
        adopt["pass"] = bool(clearing)
    else:
        adopt["clearing"] = []
        adopt["pass"] = None

    causal = {
        "rule": f"C adversarial-stratum EWR >= B + {SETUPS_CAUSAL_MARGIN}",
        "margin": SETUPS_CAUSAL_MARGIN,
        "b_ewr": damped["adversarial_ewr"] if damped else None,
        "c_ewr": damped_plus["adversarial_ewr"] if damped_plus else None,
    }
    if causal["b_ewr"] is not None and causal["c_ewr"] is not None:
        causal["threshold"] = round(float(causal["b_ewr"]) + SETUPS_CAUSAL_MARGIN, 5)
        causal["delta"] = round(float(causal["c_ewr"]) - float(causal["b_ewr"]), 5)
        causal["pass"] = bool(float(causal["c_ewr"]) >= causal["threshold"])
    else:
        causal["pass"] = None

    plateaus = {
        arm: plateau_slope(curves, arm) for arm in (CONTROL_ARM, DAMPED_ARM, DAMPED_PLUS_ARM)
    }

    if missing:
        verdict = VERDICT_INCOMPLETE
    elif winner is not None:
        verdict = VERDICT_ADOPT
    else:
        verdict = VERDICT_STOP

    statement = {
        VERDICT_ADOPT: (
            f"{winner} clears the control by the predeclared {ADOPT_RECIPE_MARGIN} "
            "margin on the final hour; it is the Phase 16 production recipe"
        ),
        VERDICT_STOP: (
            "neither B nor C clears adopt_recipe; section 5's stop_rule applies. "
            "No long run is authorized by this file. The report is written and "
            "the decision returns to the operator."
        ),
        VERDICT_INCOMPLETE: (
            f"no final hour recorded for {missing}; the shootout has not finished "
            "and no rule can be applied yet"
        ),
    }[verdict]

    # What the instrument can actually resolve, stated next to the rule that
    # asks it to resolve 0.03. A decision margin well inside one standard error
    # is a coin flip dressed as a threshold, and the report must not let a
    # reader mistake the second for the first.
    power = {}
    observed = [
        row["benchmark_ewr"]
        for rows in arms.values()
        for row in rows
        if row.get("benchmark_ewr") is not None
    ]
    if observed:
        games = next(
            (
                row.get("benchmark_games")
                for rows in arms.values()
                for row in rows
                if row.get("benchmark_games")
            ),
            0,
        )
        typical = sum(observed) / len(observed)
        se = standard_error(typical, games)
        full_games = next(
            (
                row.get("benchmark_full_games")
                for rows in arms.values()
                for row in rows
                if row.get("benchmark_full_games")
            ),
            None,
        )
        power = {
            "decision_instrument": "the predeclared quick subset",
            "games": games,
            "typical_ewr": round(typical, 4),
            "standard_error": round(se, 4),
            "decision_margin": ADOPT_RECIPE_MARGIN,
            "margin_in_standard_errors": round(ADOPT_RECIPE_MARGIN / se, 2) if se else None,
            "resolvable": bool(se and ADOPT_RECIPE_MARGIN >= se),
            "statement": (
                f"one standard error on the {games}-board decision instrument is "
                f"{se:.3f} at the observed EWR; the predeclared margin is "
                f"{ADOPT_RECIPE_MARGIN}, i.e. "
                f"{ADOPT_RECIPE_MARGIN / se:.2f} standard errors. A difference this "
                "rule calls decisive is well inside the noise of the instrument it "
                "reads, so the verdict below should be read as the mechanical "
                "output of a predeclared rule and not as evidence that one recipe "
                "is better."
            )
            if se
            else "",
        }
        if full_games:
            full_se = standard_error(typical, full_games)
            power["higher_powered_secondary"] = {
                "games": full_games,
                "standard_error": round(full_se, 4),
                "margin_in_standard_errors": round(ADOPT_RECIPE_MARGIN / full_se, 2)
                if full_se
                else None,
                "note": (
                    "the full pack contains the quick subset, so this is the same "
                    "games plus the rest; it does not enter any decision rule"
                ),
            }

    # Would the higher-powered reading of the *same games* have produced the
    # same verdict? If not, that disagreement is the sharpest available evidence
    # that the margin is below what the instrument can resolve. It is reported,
    # and it changes nothing: the rule was predeclared on the quick subset, and
    # re-reading it on another instrument after seeing the answer is precisely
    # what predeclaring it was meant to prevent.
    secondary = {}
    control_full = (control or {}).get("benchmark_full_ewr")
    if control_full is not None:
        threshold_full = round(float(control_full) + ADOPT_RECIPE_MARGIN, 5)
        candidates_full = {
            arm: (finals[arm] or {}).get("benchmark_full_ewr")
            for arm in (DAMPED_ARM, DAMPED_PLUS_ARM)
        }
        clearing_full = sorted(
            arm
            for arm, value in candidates_full.items()
            if value is not None and float(value) >= threshold_full
        )
        secondary = {
            "instrument": "the full benchmark pack, which contains the quick subset",
            "control_ewr": control_full,
            "threshold": threshold_full,
            "candidates": candidates_full,
            "clearing": clearing_full,
            "would_adopt": bool(clearing_full),
            "agrees_with_decision": bool(clearing_full) == bool(adopt.get("pass")),
            "binding": (
                "reported only; no decision rule reads it, and the verdict is "
                "the quick subset's"
            ),
        }
        if not secondary["agrees_with_decision"]:
            secondary["reading"] = (
                "the two instruments disagree on the same games. That is not a "
                "reason to prefer either: it is a direct measurement of the fact "
                "that a 0.03 margin sits inside the noise of both, and the "
                "decision stands on the instrument that was named in advance."
            )

    document = {
        "verdict": verdict,
        "statement": statement,
        "power": power,
        "secondary_instrument": secondary,
        "winner": winner,
        "adopt_recipe": adopt,
        "setups_causal": causal,
        "plateau_check": {
            "rule": (
                "a flat B/C with a passing h6 still adopts; the plateau moved, "
                "it did not vanish"
            ),
            "slopes": plateaus,
            "note": (
                "Phase 14 gained +0.0414 EWR in its first six hours and nothing "
                "measurable in the next 54; a six-hour shootout cannot see past "
                "its own horizon"
            ),
        },
        "curves": arms,
        "missing_arms": missing,
    }
    if winner is not None and winner in configs:
        document["recipe"] = configs[winner]
    return document


def throughput_verdict(payload: dict) -> dict:
    """Gate 3 as a sentence, from the measured pair."""
    gate = payload.get("gate") or {}
    if gate.get("pass") is None:
        return {"pass": None, "statement": gate.get("note", "not measured")}
    ours = (payload.get("phase16") or {}).get("plies_per_second")
    theirs = (payload.get("phase14") or {}).get("plies_per_second")
    return {
        "pass": bool(gate["pass"]),
        "statement": (
            f"the window collector advanced {ours} plies/s against the accepted "
            f"Phase 14 collector's {theirs} plies/s on the same machine "
            f"(ratio {gate.get('ratio_phase16_over_phase14')}); the gate asks for "
            "within 2x"
        ),
    }


#: Phase 14's own recorded production facts, for the end-to-end comparison.
#: Not re-measured here -- these are what that run reported when it stopped.
PHASE14_RUN_HOURS = 59.97
PHASE14_RUN_STEPS = 202_504
PHASE14_RUN_ITERATIONS = 102
PHASE14_MINIBATCH = 512
PHASE14_EPOCHS = 2

#: The per-iteration decomposition extracted from that run's own telemetry.
PHASE14_DECOMPOSITION_PATH = "reports/phase16/agent_03_phase14_decomposition.json"


def load_phase14_decomposition(root: "str | Path" = ".") -> dict:
    """Phase 14's collection/training split, if it has been extracted."""
    from pathlib import Path

    path = Path(root) / PHASE14_DECOMPOSITION_PATH
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def end_to_end_comparison(throughput: dict, telemetry_rows, decomposition: "dict | None" = None) -> dict:
    """Where the window collector's time actually goes, against where Phase 14's did.

    The honest version of this comparison is *not* collector-against-collector.
    Phase 14's own telemetry shows its collection ran at a median 1,784 plies/s
    and never slowed materially: collection was 17% of its 59.97 hours. What
    grew was **training**, from 15.7 to 153.8 minutes an iteration, because a
    2,048-*game* iteration carries more data as games get longer -- mean game
    length went 265 to 734 plies and optimizer steps per iteration went 1,712 to
    5,247. A window collector pins that quantity by construction.

    So this reports the split, not a speedup. Gate 3's collector ratio is
    reported alongside and explicitly deflated: it compares two cold starts.
    """
    rows = [row for row in telemetry_rows if row.get("collection")]
    if not rows:
        return {}
    # The first window of a run is a cold start -- 96 games all at ply 0, no
    # desynchronisation yet -- so steady state is everything after it.
    steady = rows[1:] or rows
    plies = sum(row["collection"].get("plies_advanced", 0) for row in steady)
    trained = sum(row["collection"].get("rows", 0) for row in steady)
    collection = sum(row["collection"].get("seconds", 0.0) for row in steady)
    training = sum((row.get("optimization") or {}).get("seconds", 0.0) for row in steady)
    steps = sum((row.get("optimization") or {}).get("optimizer_steps", 0) for row in steady)
    wall = collection + training
    if wall <= 0 or plies <= 0:
        return {}

    ours = {
        "windows": len(steady),
        "collection_plies_per_second": round(plies / collection, 1) if collection else None,
        "collection_share_of_wall": round(collection / wall, 4),
        "training_share_of_wall": round(training / wall, 4),
        "seconds_per_optimizer_step": round(training / steps, 4) if steps else None,
        "optimizer_steps_per_iteration": round(steps / len(steady)),
        "rows_per_iteration": round(trained / len(steady)),
        "minutes_per_iteration": round(wall / len(steady) / 60.0, 3),
        "trained_decisions_per_hour": round(trained / wall * 3600.0),
        "isolated_plies_per_second": (throughput.get("phase16") or {}).get(
            "plies_per_second"
        ),
    }

    examples = PHASE14_RUN_STEPS * PHASE14_MINIBATCH / PHASE14_EPOCHS
    theirs = {
        "source": (
            f"the run's own recorded facts: {PHASE14_RUN_HOURS} h, step "
            f"{PHASE14_RUN_STEPS}, {PHASE14_RUN_ITERATIONS} iterations at "
            f"minibatch {PHASE14_MINIBATCH} x {PHASE14_EPOCHS} epochs"
        ),
        "trained_decisions_per_hour": round(examples / PHASE14_RUN_HOURS),
        "minutes_per_iteration": round(
            PHASE14_RUN_HOURS * 60.0 / PHASE14_RUN_ITERATIONS, 1
        ),
        "isolated_plies_per_second": (throughput.get("phase14") or {}).get(
            "plies_per_second"
        ),
    }
    decomposition = decomposition or {}
    if decomposition:
        whole = decomposition["whole_run"]
        rates = decomposition["collection_plies_per_second"]
        growth = decomposition["growth"]
        theirs.update(
            {
                "collection_plies_per_second": rates["median"],
                "collection_plies_per_second_range": [rates["min"], rates["max"]],
                "collection_share_of_wall": whole["collection_share"],
                "training_share_of_wall": whole["training_share"],
                "collection_hours": whole["collection_hours"],
                "training_hours": whole["training_hours"],
                "seconds_per_optimizer_step": growth["seconds_per_optimizer_step"]["first5"],
                "growth": growth,
                "decomposition_source": decomposition["source"]["note"],
            }
        )

    comparison = {
        "phase16": ours,
        "phase14": theirs,
        "ratios": {
            "trained_decisions_per_hour": round(
                ours["trained_decisions_per_hour"] / theirs["trained_decisions_per_hour"], 2
            )
            if theirs["trained_decisions_per_hour"]
            else None,
        },
        "caveats": [
            "the Phase 16 figures come from the 20-minute smoke run at the "
            "production window on an idle machine, not from a six-hour arm",
            "an iteration is not the same unit in the two systems (2,048 games "
            "against a 65,536-decision budget), so minutes/iteration is reported "
            "for scale and is not a speedup",
            "Phase 14's hours include its in-run candidate evaluations, which are "
            "charged to the training remainder rather than separated out",
        ],
    }
    if decomposition:
        comparison["ratios"]["collection_plies_per_second"] = (
            round(ours["collection_plies_per_second"] / theirs["collection_plies_per_second"], 2)
            if ours["collection_plies_per_second"] and theirs["collection_plies_per_second"]
            else None
        )
        comparison["gate3_caveat"] = (
            "Gate 3's collector ratio compares two *cold starts* -- 96 games all "
            "at ply 0 -- and the Phase 14 side of it carries model-load time "
            "inside a 16.6-second measurement. Against that run's own production "
            f"telemetry (median {theirs['collection_plies_per_second']} plies/s) the "
            f"window collector's steady-state {ours['collection_plies_per_second']} "
            "plies/s is not faster. Collection is a wash; the gate is a floor "
            "check and it passes, but it is not the finding."
        )
        comparison["finding"] = (
            "Phase 14 spent "
            f"{theirs['training_share_of_wall']:.0%} of its {PHASE14_RUN_HOURS} hours "
            "in the training phase, and that phase grew from "
            f"{growth['training_minutes']['first5']} to "
            f"{growth['training_minutes']['last5']} minutes an iteration while "
            f"collection went only {growth['collection_minutes']['first5']} to "
            f"{growth['collection_minutes']['last5']}. The cause is iteration "
            "*sizing*: 2,048 whole games carry more data as games lengthen "
            f"({growth['mean_game_length']['first5']} to "
            f"{growth['mean_game_length']['last5']} plies, "
            f"{growth['optimizer_steps_per_iteration']['first5']} to "
            f"{growth['optimizer_steps_per_iteration']['last5']} optimizer steps an "
            "iteration). A fixed decision budget pins that quantity by "
            "construction, which is the whole of the design change."
        )
    return comparison


def instrument_check(curves: dict) -> dict:
    """Do the arms' h=0 readings agree, as they must?

    Every arm exports its starting weights at h=0, and every arm starts from the
    same read-only P24 copy. Three identical policies scored on the same boards
    must produce the same number; if they do not, the evaluator is not
    deterministic and no later difference between arms means anything. This is
    the cheapest available check on the instrument and it costs nothing to run,
    because the readings already exist.
    """
    zero = {
        arm: entry
        for arm, entry in (
            (arm, (curves.get(arm) or {}).get("0")) for arm in curves
        )
        if entry
    }
    if len(zero) < 2:
        return {}
    digests = {arm: entry.get("model_state_digest") for arm, entry in zero.items()}
    benchmark = {arm: (entry.get("benchmark") or {}).get("ewr") for arm, entry in zero.items()}
    adversarial = {
        arm: (entry.get("adversarial") or {}).get("ewr") for arm, entry in zero.items()
    }
    sources = {arm: entry.get("source") for arm, entry in zero.items()}
    return {
        "arms": sorted(zero),
        "starting_model_state_digest": digests,
        "h0_benchmark_ewr": benchmark,
        "h0_adversarial_ewr": adversarial,
        "export_source": sources,
        "scores_agree": len(set(v for v in benchmark.values() if v is not None)) == 1
        and len(set(v for v in adversarial.values() if v is not None)) == 1,
        # "not recorded" and "recorded and different" are different facts, and
        # only the second is a fault.
        "digests_recorded": all(digests.values()),
        "digests_agree": (
            len(set(digests.values())) == 1 if all(digests.values()) else None
        ),
        "note": (
            "arms exporting raw weights and arms exporting an EMA both write the "
            "same tensors at update 0, so an h=0 disagreement would be an "
            "evaluator fault rather than a recipe difference"
        ),
    }


def horizon_evidence(root: "str | Path" = ".") -> dict:
    """What a longer run must re-derive its schedule constants from.

    Phase 16's schedule constants are horizon-dependent: `n_ref` is a fraction
    of the *planned iteration count*, and that count follows from how long an
    iteration takes on the machine the run is on. A production run at a
    different length, window size or population has a different N and therefore
    a different `n_ref`. Carrying the inputs rather than only the outputs is
    what lets the next agent recompute instead of inheriting a number whose
    horizon no longer applies -- which is the exact failure this phase already
    caught once.
    """
    from pathlib import Path

    from .contract import (
        LR_HORIZON_FRACTION,
        LR_REFERENCE_ITERATION,
        MEASURED_ITERATION_SECONDS,
        PLANNED_ITERATIONS,
    )

    decomposition = load_phase14_decomposition(root)
    path = Path(root) / PHASE14_DECOMPOSITION_PATH
    evidence = {
        "why": (
            "the section 2.3 constants are horizon-dependent; a run of a "
            "different length, window size or population has a different N and "
            "therefore a different n_ref"
        ),
        "measured": {
            "seconds_per_iteration": MEASURED_ITERATION_SECONDS,
            "conditions": (
                "population 96, window 65,536 learner decisions, minibatch 512, "
                "1 epoch, MPS, idle machine"
            ),
            "planned_iterations_for_six_hours": PLANNED_ITERATIONS,
        },
        "derivation": {
            "rule": "n_ref = ceil(lr_horizon_fraction * N)",
            "lr_horizon_fraction": LR_HORIZON_FRACTION,
            "n_ref": LR_REFERENCE_ITERATION,
            "recompute_for_a_longer_run": (
                "N = run_seconds / measured seconds_per_iteration at the run's own "
                "window size and population; then n_ref = ceil(0.125 * N). Do not "
                "carry n_ref = 40 into a run that is not ~313 iterations."
            ),
        },
        "phase14_decomposition": {
            "path": PHASE14_DECOMPOSITION_PATH,
            "present": bool(decomposition),
            "regenerate": "scripts/run_phase16_agent03.py --role decompose",
            "sha256": (
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            ),
        },
        "for_agent_5": (
            "brief 05 section 4 asks for horizon constants for the production "
            "run: re-derive them from the measured seconds_per_iteration above at "
            "the production window size, not from this phase's n_ref"
        ),
    }
    if decomposition:
        evidence["phase14_decomposition"]["whole_run"] = decomposition["whole_run"]
        evidence["phase14_decomposition"]["finding"] = (
            "collection was 17% of Phase 14's 59.97 hours and its rate held "
            "(median 1,784 plies/s); training grew 15.7 -> 153.8 minutes an "
            "iteration because a 2,048-game iteration carries more data as games "
            "lengthen. A fixed decision budget pins that; a game count does not."
        )
    return evidence


def require_curves(curves: dict) -> None:
    if not curves:
        raise Phase16TrainingError("no hour curves recorded; run --role evaluate first")


__all__ = [
    "CONTROL_ARM",
    "end_to_end_comparison",
    "horizon_evidence",
    "instrument_check",
    "load_phase14_decomposition",
    "DAMPED_ARM",
    "DAMPED_PLUS_ARM",
    "VERDICT_ADOPT",
    "VERDICT_INCOMPLETE",
    "VERDICT_STOP",
    "arm_curve",
    "decide_recipe",
    "final_hour",
    "plateau_slope",
    "require_curves",
    "standard_error",
    "throughput_verdict",
]
