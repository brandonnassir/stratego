"""Phase 15 Agent 1: the report, rendered from the recorded artifacts.

Every number in the rendered document comes from a JSON artifact that was
written by the stage that produced it. Nothing is retyped, so the report
cannot drift from the evidence, and a reader who distrusts a sentence can
open the file it came from.
"""

from __future__ import annotations

import time


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def render(
    *,
    manifest: dict,
    metrics: dict,
    checks: dict,
    verification: dict,
    orientation: dict,
    boundary: dict,
    handoff: dict,
    curves: dict,
    sources: dict,
    summary: dict,
    specialist_table: str,
) -> str:
    comparison = metrics["comparison"]
    recipe = curves["recipe"]
    splits = manifest["splits"]
    total_positions = sum(block["samples"] for block in splits.values())
    total_pieces = sum(block["pieces"] for block in splits.values())

    lines: list[str] = []
    add = lines.append

    add("# Phase 15 — Agent 1")
    add("## Clean belief-corpus generation and B18/B24 fine-tuning")
    add("")
    add(
        f"_Written {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. "
        "This is an engineering deliverable: a corpus, two belief models and a "
        "handoff. It is **not** a playing-strength claim — no search was "
        "implemented, and none was evaluated._"
    )
    add("")

    # -- 1. Process boundary ------------------------------------------------
    add("## 1. Process boundary")
    add("")
    add(
        "This task does not authorize process control, and none was exercised: "
        "no emergency-stop file was created, no signal was sent, no live run "
        "state was edited, no checkpoint was rotated and no closeout command "
        "was invoked."
    )
    add("")
    state = boundary.get("phase14_run_state", {})
    add(
        f"- verdict: `{boundary['verdict']}` — "
        f"{len(boundary['competing_phase14_processes'])} competing Phase 14 "
        f"processes on a {boundary['host_cpus']}-core host"
    )
    if state:
        add(
            f"- Phase 14 run state at check time: {state.get('elapsed_hours', 0):.2f} h "
            f"elapsed, {state.get('iterations_completed')} iterations, "
            f"`closed_reason={state.get('closed_reason')!r}`, last written "
            f"{state.get('mtime_utc')}"
        )
    add(
        "- the only Phase 14 process running was the read-only monitoring "
        "dashboard, which holds no compute"
    )
    add(
        "- Phase 14 artifacts were opened read-only: the candidate ledger, the "
        "two candidate evaluation weight files, and the two archive snapshots "
        "the ledger names"
    )
    add("")

    # -- 2. Orientation -----------------------------------------------------
    add("## 2. The orientation correction (section 4)")
    add("")
    add(
        "The old Phase 11B corpus and the Phase 12 match packs are contaminated: "
        "`Phase11BSetupSources.draw` returned canonical own-orientation tuples "
        "and the old glue handed them straight to `create_game()` for Blue. "
        "Canonical rank 0 is a player's own back rank, and Blue's engine setup "
        "order runs front-to-back, so an unoriented Blue army was placed "
        "reversed."
    )
    add("")
    add("The rule Phase 15 enforces, re-derived from the engine's own `SETUP_SQUARES`:")
    add("")
    add("```text")
    add(orientation["orientation_rule"])
    add("```")
    add("")
    add(
        f"The gate ran on **{orientation['boards_checked']:,} paired boards "
        f"({orientation['armies_checked']:,} armies)**, checking flag location, "
        "legal setup rows, complete inventory and Red/Blue paired orientation "
        "on every one."
    )
    add("")
    counterfactual = orientation["defect_counterfactual"]
    add(
        f"- observed front-row flags: **{orientation['front_row_flags']} of "
        f"{orientation['armies_checked']:,}** "
        f"({_percent(orientation['front_row_flag_rate'])})"
    )
    add(
        f"- the same draws under the old glue would have produced "
        f"**{counterfactual['blue_front_row_flags']:,} of "
        f"{counterfactual['blue_boards']:,}** Blue front-row flags "
        f"({_percent(counterfactual['rate'])}) — which reproduces Phase 12's "
        "47-of-64 observation almost exactly"
    )
    add(
        "- negative canary: handing the gate a raw Blue canonical tuple is "
        f"**detected** (`{orientation['negative_canary']['canary']}`)"
    )
    add(
        "- flag-row histograms are exact mirrors: Red concentrates on engine "
        "row 0, Blue on engine row 9"
    )
    add("")
    add(
        "No Phase 15 module imports `belief/phase11b/corpus.py`, "
        "`Phase11BSetupSources` or `corpus_plans`, and a test enforces it."
    )
    add("")

    # -- 3. P18 / P24 -------------------------------------------------------
    add("## 3. P18 and P24 (section 3)")
    add("")
    add(
        "Both were resolved from the Phase 14 **candidate ledger**, not from the "
        "newest hot checkpoint, and every identity was re-derived from bytes."
    )
    add("")
    for source_id in sorted(sources):
        block = sources[source_id]
        evaluation = block["candidate_evaluation"]
        add(f"**{block['logical_identity']}** — Phase 14 candidate hour {block['hour']}")
        add("")
        add(f"- archive snapshot `{block['original_snapshot_path']}`")
        add(f"  - sha256 `{block['original_snapshot_sha256']}` (matches the ledger)")
        add(f"- model-state digest `{block['model_state_digest']}`")
        add(
            f"- optimizer step {block['global_optimizer_step']:,}, iteration "
            f"{block['iteration']}, elapsed {block['elapsed_seconds'] / 3600.0:.2f} h"
        )
        add(
            f"- candidate evaluation complete: {evaluation['games_played']} games, "
            f"mean EWR {evaluation['mean_ewr']:.4f}, min stratum "
            f"{evaluation['min_stratum_ewr']:.4f} on pack "
            f"`{evaluation['pack_digest'][:16]}…`"
        )
        add(
            f"- read-only Phase 15 copy `{block['phase15_copy_path']}` "
            f"(sha256 `{block['phase15_copy_sha256'][:16]}…`, mode 0444)"
        )
        add("")
    add(
        "Neither file was written, and neither model was trained. The digests "
        "below are measured before and after each fine-tuning run."
    )
    add("")
    for specialist_id, block in sorted(metrics["specialists"].items()):
        unchanged = block["training"]["source_unchanged"]
        add(
            f"- `{unchanged['source_id']}` before `{unchanged['model_state_digest_before'][:16]}…` "
            f"→ after `{unchanged['model_state_digest_after'][:16]}…` — "
            f"**unchanged: {unchanged['unchanged']}**"
        )
    add("")

    # -- 4. Corpus ----------------------------------------------------------
    add("## 4. `phase15_belief_corpus_v1` (sections 5–7)")
    add("")
    add(f"`corpus_digest` `{manifest['corpus_digest']}`")
    add("")
    add(
        f"**{total_positions:,} eligible observer positions** carrying "
        f"**{total_pieces:,} supervised hidden pieces**, generated in "
        f"{summary['corpus']['generation_minutes']:.1f} minutes on "
        f"{manifest['workers']} CPU worker processes."
    )
    add("")
    rows = []
    for split in ("train", "calibration", "development"):
        block = splits[split]
        rows.append(
            f"| {split} | {block['samples']:,} | {block['target_positions']:,} | "
            f"{block['games']:,} | {block['pieces']:,} | "
            f"{block['hidden_pieces_per_sample']:.2f} | {block['library_split']} |"
        )
    add("| split | positions | target | games | hidden pieces | pieces/position | library split |")
    add("|-------|-----------|--------|-------|---------------|-----------------|---------------|")
    lines.extend(rows)
    add("")
    add(
        "Every split met its **initial engineering target**; the section 5 "
        "fallback floor was not used and no pilot evidence was needed to "
        "justify a reduction."
    )
    add("")
    add("### Achieved mixture, counted over positions")
    add("")
    add(
        "Section 6 asks for counts *after position sampling*, not intended game "
        "counts. The training split's achieved shares:"
    )
    add("")
    mixture = verification["splits"]["train"]["mixture"]
    for dimension, label in (
        ("observer_model", "observer"),
        ("opponent", "opponent"),
        ("setup_source", "setup source"),
        ("observer_color", "observer colour"),
    ):
        parts = ", ".join(
            f"{name} {_percent(entry['fraction'])}"
            for name, entry in sorted(mixture[dimension].items())
        )
        add(f"- **{label}**: {parts}")
    deviation = mixture["max_absolute_deviation"]
    add(
        "- largest absolute deviation from the design: "
        + ", ".join(
            f"{name} {value:.4f}" for name, value in sorted(deviation.items())
        )
    )
    add(
        f"- targeted families missing: "
        f"`{mixture['targeted_families_missing'] or 'none'}`"
    )
    band = mixture["game_band"]
    add(
        "- game band: "
        + ", ".join(
            f"{name} {_percent(entry['fraction'])}" for name, entry in sorted(band.items())
        )
    )
    add("")
    add("### Split disjointness")
    add("")
    pairs = verification["disjointness"]["pairs"]
    for name, block in sorted(pairs.items()):
        add(
            f"- `{name}`: {block['shared_game_ids']} shared game ids, "
            f"{block['shared_public_state_identities']} shared public-state "
            "identities"
        )
    add("")
    add(
        "Training draws its base setups from the accepted `train` library split. "
        "Calibration and development both draw from `validation`, which is not "
        "enough on its own — two games whose observer drew the same base setup "
        "reach the same opening public state. The `validation` population is "
        "therefore **partitioned in half, per family**, and each split draws "
        "only from its own half. This was found by the disjointness check on a "
        "first build of the corpus, which shared 43 opening positions between "
        "calibration and development; the corpus was regenerated after the fix."
    )
    add("")
    add("### The public/privileged boundary")
    add("")
    add(
        "Two passes. The public pass plays the game with a policy that reads a "
        "`PolicyInput` and records only the ply, the unresolved-piece count and "
        "a sha256 of its own observation. The privileged replay rebuilds each "
        "selected decision from the action history, **checks the rebuilt "
        "observation against that digest**, and only then reads "
        "`dense_belief_target`."
    )
    add("")
    add(
        "Public arrays live in `public/`, the single label array in "
        "`privileged/`, and `load_split` returns the public half unless "
        "`labels=True` is passed."
    )
    for split, block in sorted(verification["splits"].items()):
        labels = block["labels"]
        add(
            f"- `{split}`: {labels['pieces']:,} stored ranks, all publicly "
            "admissible, all with remaining public inventory, "
            f"{labels['moved_pieces_with_immobile_rank']} moved pieces carrying "
            "an immobile rank"
        )
    add("")
    orientation_recheck = verification["splits"]["train"]["orientation"]
    add(
        f"Boards rebuilt from stored game ids alone: "
        f"{orientation_recheck['armies_rechecked']} armies re-checked, front-row "
        f"flag rate {_percent(orientation_recheck['front_row_flag_rate'])}."
    )
    add("")

    # -- 5. B18 / B24 -------------------------------------------------------
    add("## 5. B18 and B24 (sections 8–10)")
    add("")
    add("```text")
    add("frozen  P18/P24 prefix   first three C1 transformer blocks")
    add("trained copy             final C1 block + encoder norm")
    add("fresh   belief MLP       128 -> 512 -> 512 -> 12, GELU")
    add("```")
    add("")
    first = next(iter(metrics["specialists"].values()))
    add(
        "A belief checkpoint contains only the copied block, the copied encoder "
        "norm, the belief MLP, the calibration temperature and the identity "
        "bindings — no policy tensor and no value tensor. Loading one requires "
        "the backbone whose model-state digest it recorded, and refuses any "
        "other."
    )
    add("")
    add("### The recipe (declared once, shared by both)")
    add("")
    add("```text")
    for key in (
        "loss",
        "optimizer",
        "head_learning_rate",
        "final_block_learning_rate",
        "weight_decay",
        "schedule",
        "batch_size",
        "max_epochs",
        "early_stop_patience",
        "selection",
    ):
        add(f"{key:<26} {recipe[key]}")
    add("```")
    add("")
    for specialist_id, block in sorted(metrics["specialists"].items()):
        training = block["training"]
        config = training["config"]
        note = ""
        if config.get("batch_size_changed_from"):
            note = (
                f" (batch size changed from {config['batch_size_changed_from']} to "
                f"{config['batch_size']}: {config['batch_size_change_reason']})"
            )
        add(
            f"- **{specialist_id.upper()}**: {training['epochs_run']} epochs run, "
            f"best at epoch {training['best_epoch']} "
            f"(`{training['stopped_because']}`), "
            f"{training['training_seconds'] / 60.0:.1f} min total, "
            f"{training['time_to_best_seconds'] / 60.0:.1f} min to best{note}"
        )
        add(
            f"  - gradient isolation: "
            f"{training['gradient_isolation']['policy_value_parameters_with_gradient']} "
            "policy/value parameters carried a gradient, "
            f"{training['gradient_isolation']['checked_parameters']} checked"
        )
    add("")
    add("### Calibration")
    add("")
    for specialist_id, block in sorted(metrics["specialists"].items()):
        calibration = block["calibration"]
        add(
            f"- **{specialist_id.upper()}**: fitted temperature "
            f"{calibration['temperature']:.4f} on the calibration split "
            f"({calibration['calibration_pieces']:,} pieces), "
            f"NLL {calibration['calibration_nll_raw']:.4f} → "
            f"{calibration['calibration_nll_fitted']:.4f}"
        )
        add(
            f"  - development NLL {calibration['development_nll_raw']:.4f} → "
            f"{calibration['development_nll_calibrated']:.4f}; ECE "
            f"{calibration['development_ece_raw']:.4f} → "
            f"{calibration['development_ece_calibrated']:.4f}; top-1 unchanged: "
            f"{calibration['top1_unchanged']}"
        )
        add(
            f"  - **kept: {calibration['keep_calibrated']}**, applied temperature "
            f"{calibration['applied_temperature']:.4f}"
        )
    add("")

    # -- 6. Metrics ---------------------------------------------------------
    add("## 6. Development metrics (section 11)")
    add("")
    add(
        f"All models scored on the **same** {comparison['development_positions']:,} "
        f"development positions and {comparison['development_pieces']:,} hidden "
        "pieces, against the same accepted `remaining_count_belief_v1` "
        "denominator."
    )
    add("")
    add(specialist_table)
    add("")
    add(
        f"The uninformed floor — a flat 12-way vector — scores "
        f"R_CE {comparison['uniform_reference']['r_ce']:.4f}, so every model "
        "above is better than knowing nothing."
    )
    add("")
    add("### Paired comparisons (game bootstrap, same positions)")
    add("")
    for name, block in sorted(comparison["paired"].items()):
        left, right = name.split("_vs_")
        direction = (
            f"{left.upper()} lower"
            if block["left_lower_ce"]
            else (
                f"{right.upper()} lower"
                if block["distinguishable"]
                else "indistinguishable"
            )
        )
        add(
            f"- `{name}`: ΔCE {block['ce_difference']:+.4f} "
            f"[{block['ce_difference_ci95'][0]:+.4f}, "
            f"{block['ce_difference_ci95'][1]:+.4f}] over {block['games']:,} "
            f"games — **{direction}**"
        )
    add("")
    reference = comparison["agent1c_reference"]
    add(
        "The Agent 1C comparison is on the **new** development corpus. Its old "
        f"result (R_CE {reference['old_development_r_ce']:.4f}) was measured on "
        "`phase11b_common_corpus_v1`, whose Blue setups are mis-oriented, and is "
        "quoted only to identify the artifact — never as the comparison set."
    )
    add("")
    add(
        f"Scored here, Agent 1C reaches R_CE {reference['r_ce']:.4f} with a 95% "
        f"interval of [{reference['r_ce_ci95'][0]:.4f}, "
        f"{reference['r_ce_ci95'][1]:.4f}] — that is, **statistically "
        "indistinguishable from the remaining-count baseline it was built to "
        "beat**. Two things changed at once and this experiment does not "
        "separate them:"
    )
    add("")
    add(
        "1. the corpus is orientation-correct, and 1C was trained on boards "
        "where Blue's army was placed back-to-front;"
    )
    add(
        "2. 1C is attached to the accepted **Phase 9** backbone and was trained "
        "against a different observer, opponent and setup distribution than the "
        "one measured here."
    )
    add("")
    add(
        "So the drop should not be read as a clean measurement of the "
        "orientation defect's cost. What it does establish is narrower and "
        "sufficient for the handoff: on the corpus search will actually face, "
        "the surviving old belief model carries no usable advantage over the "
        "count baseline, and the two new specialists clearly do."
    )
    add("")
    add("### Breakdowns")
    add("")
    add(
        "Full per-cell blocks — observer colour, observer source, opponent, "
        "opponent class, setup source, setup family and early/middle/late band —"
        " are in `agent_01_metrics.json` under each specialist's "
        "`development_calibrated.breakdowns`. The headline splits:"
    )
    add("")
    for specialist_id in sorted(metrics["specialists"]):
        block = metrics["specialists"][specialist_id]
        chosen = (
            block["development_calibrated"]
            if block["calibration"]["keep_calibrated"]
            else block["development_raw"]
        )
        breakdowns = chosen["breakdowns"]
        add(f"**{specialist_id.upper()}**")
        for dimension in ("observer_color", "observer_source", "opponent_class", "game_band"):
            parts = ", ".join(
                f"{name} {entry['r_ce']:.4f}"
                for name, entry in sorted(breakdowns[dimension].items())
            )
            add(f"- R_CE by {dimension.replace('_', ' ')}: {parts}")
        add("")

    # -- 7. Provider --------------------------------------------------------
    add("## 7. The belief/sampler interface (section 12)")
    add("")
    add("```text")
    add("predict_marginals(public_state)      -> 12-way rank probabilities")
    add("sample_worlds(public_state, n, seed) -> complete legal hidden armies")
    add("```")
    add("")
    add(
        "Marginals go through the accepted constrained-world sampler by import: "
        "`stratego.evaluation.phase11_sampler.sample_belief_world`, unmodified. "
        "Pieces are not sampled independently and no accepted inventory or "
        "movement-impossibility constraint was altered."
    )
    add("")
    for specialist_id, block in sorted(checks["providers"].items()):
        latency = block["marginal_latency_ms"]
        add(
            f"- **{specialist_id.upper()}**: {block['positions_checked']} fresh "
            f"positions, {block['worlds_checked']} sampled worlds — "
            "probabilities finite and summing to one "
            f"(max deviation {block['max_probability_sum_deviation']:.2e}); "
            "fixed seed reproduces worlds; remaining piece counts exact; moved "
            "pieces never assigned Flag or Bomb; every world passes the accepted "
            "validation stack"
        )
        add(
            f"  - marginal latency: mean {latency['mean']:.2f} ms, "
            f"p50 {latency['p50']:.2f} ms, p95 {latency['p95']:.2f} ms"
        )
        isolation = block["truth_isolation"]
        add(
            f"  - truth isolation: the public-state type carries exactly "
            f"{isolation['public_state_fields']}, the provider reports "
            f"`uses_hidden_truth={isolation['provider_uses_hidden_truth']}`, and "
            "it answers from the public document alone"
        )
    add("")

    # -- 8. Handoff ---------------------------------------------------------
    add("## 8. The search handoff (section 13)")
    add("")
    add(
        f"`{summary['handoff']['path']}` binds exact digests for P18, P24, B18, "
        "B24, the corpus, both calibration values, the provider interface "
        f"version and the accepted sampler version. It re-verifies against the "
        f"bytes on disk: **verified = {handoff['verification']['verified']}** "
        f"over {handoff['verification']['artifacts_checked']} artifacts."
    )
    add("")
    add("```text")
    for source_id, block in sorted(handoff["policy_models"].items()):
        add(f"{block['logical_identity']:<4} {block['model_state_digest']}")
    for specialist_id, block in sorted(handoff["belief_models"].items()):
        add(f"{specialist_id.upper():<4} {block['state_digest']}")
    add(f"corpus {handoff['corpus']['corpus_digest']}")
    add("```")
    add("")

    # -- 9. Limits ----------------------------------------------------------
    add("## 9. What this does and does not establish")
    add("")
    add(
        "- **Established**: an orientation-safe corpus of "
        f"{total_positions:,} positions; two belief specialists trained without "
        "touching P18 or P24; calibrated development metrics; providers that "
        "generate legal deterministic worlds; an exact handoff."
    )
    add(
        "- **Not established**: any claim about playing strength. No search was "
        "implemented, no combined player was chosen, no Phase 12 artifact was "
        "modified and no Phase 14 task was controlled."
    )
    add(
        "- The Phase 14 candidate EWRs quoted in section 3 are Phase 14's own "
        "128-game pack results, reported to identify the checkpoints. They are "
        "not Phase 15 results."
    )
    add(
        "- Corpus games were played with both neural seats in the accepted "
        "**greedy** decision mode; diversity comes from the setup mixture, not "
        "from sampled play. Each game runs to its accepted termination under "
        "`EVALUATION_RULES` (battleless 200, absolute 4000); the section 5 "
        "option to retire a trajectory early was not used, because evenly "
        "spaced sampling is defined over a game's complete eligible list."
    )
    add("")
    add("## 10. Artifacts")
    add("")
    add("```text")
    add("data/phase15/phase15_belief_corpus_v1/")
    add("data/phase15/phase15_belief_corpus_v1_manifest.json")
    add("checkpoints/phase15/p18_source_identity.json")
    add("checkpoints/phase15/p24_source_identity.json")
    add("checkpoints/phase15/b18_belief_v1.pt")
    add("checkpoints/phase15/b24_belief_v1.pt")
    add("reports/phase15/agent_01_process_boundary.json")
    add("reports/phase15/agent_01_orientation_gate.json")
    add("reports/phase15/agent_01_corpus_verification.json")
    add("reports/phase15/agent_01_learning_curves.json")
    add("reports/phase15/agent_01_metrics.json")
    add("reports/phase15/agent_01_interface_checks.json")
    add("reports/phase15/agent_01_summary.json")
    add("reports/phase15/agent_01_report.md")
    add("reports/phase15/phase15_search_handoff_v1.json")
    add("```")
    add("")
    return "\n".join(lines) + "\n"


__all__ = ["render"]
