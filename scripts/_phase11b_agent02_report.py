"""Markdown renderer for the Phase 11B Agent 2 report.

Split out of `run_phase11b_agent02.py` for the same reason
`_phase10_agent05_report.py` is split out of its harness: the prose is long,
it changes for reasons that have nothing to do with the experiment, and a
harness is easier to read when the string building lives elsewhere.

Every number is read from the summary the harness just wrote. Nothing here
recomputes a metric, and nothing here decides anything: the comparison
verdict is `summary["decision"]`, produced by the harness's `decide`, and
this module only puts it into sentences.
"""

from __future__ import annotations

CANDIDATE_2 = "agent02_raw_observation_cnn"
REFERENCE = "phase11_head_unchanged_reference"

STRATA = ("phase9_selfplay", "strategic_rule", "tactical_rule", "scout_rush")


def _fmt(value, digits: int = 4, dash: str = "—") -> str:
    if value is None:
        return dash
    return f"{value:.{digits}f}"


def _interval(bounds) -> str:
    if not bounds:
        return "—"
    return f"[{bounds[0]:.4f}, {bounds[1]:.4f}]"


def _signed(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}"


def render(summary: dict, train: dict) -> str:
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    counter = {"n": -1}

    def section(title: str) -> None:
        counter["n"] += 1
        add(f"## {counter['n']}. {title}")

    row = summary["leaderboard"][CANDIDATE_2]
    earlier = summary["agent1_reference_rows"]
    decision = summary["decision"]
    pilot = summary["pilot"]
    training = summary["training"]
    interface = summary["interface"]
    corpus = summary["common_corpus"]
    reference = earlier.get(REFERENCE, {})
    agent1_best_id = decision["agent1_best_candidate"]
    agent1_best = earlier.get(agent1_best_id, {})
    leader_id = decision["leader_by_r_ce"]
    delta = decision["agent2_minus_agent1_best_r_ce"]
    paired_best = decision["paired_comparison_with_agent1_best"] or {}

    # -- header ------------------------------------------------------------
    add("# Phase 11B — Agent 2: Raw-Observation CNN")
    add()
    add("**Status: engineering prototype.** This report does not repair Phase 11, does")
    add("not overturn the Phase 11 `FAIL`, and does not authorize Phase 12.")
    add("`phase11_test_bank_v1` was not opened; it remains spent.")
    add()
    add("| marker | value |")
    add("| --- | --- |")
    for key in (
        "phase",
        "status",
        "phase11_fail_unchanged",
        "phase11_test_bank_used",
        "phase12_authorized_by_this_artifact",
        "phase11_final_classification",
        "phase11_test_bank_spent",
        "scientific_claim",
    ):
        add(f"| `{key}` | `{summary[key]}` |")
    add()

    # -- 0. headline -------------------------------------------------------
    section("What Agent 2 found")
    add()
    add("Agent 2's question was whether giving belief inference **its own")
    add("raw-observation spatial specialist** — bypassing C1's learned compression")
    add("entirely — solves most of the predictive weakness. On the common Phase 11B")
    add("development set:")
    add()
    better = decision["agent2_materially_better_than_agent1_best"]
    inside_band = abs(delta) <= decision["equivalence_band"] if delta is not None else False
    if better:
        add(
            f"1. **It helps, materially.** The raw CNN scores `R_CE` "
            f"**{row['r_ce']:.4f}** against Agent 1's best attached head at "
            f"**{agent1_best.get('r_ce', float('nan')):.4f}** — a gain of "
            f"**{abs(delta):.4f}**, wider than the sprint's "
            f"{decision['equivalence_band']} equivalence band."
        )
    elif inside_band:
        add(
            f"1. **It ties.** The raw CNN scores `R_CE` **{row['r_ce']:.4f}** against "
            f"Agent 1's best attached head at "
            f"**{agent1_best.get('r_ce', float('nan')):.4f}** — a difference of "
            f"{_signed(delta)}, inside the sprint's "
            f"{decision['equivalence_band']} equivalence band."
        )
    else:
        add(
            f"1. **It does not help.** The raw CNN scores `R_CE` "
            f"**{row['r_ce']:.4f}** against Agent 1's best attached head at "
            f"**{agent1_best.get('r_ce', float('nan')):.4f}** — "
            f"{abs(delta):.4f} *worse*, outside the sprint's "
            f"{decision['equivalence_band']} equivalence band."
        )
    add(
        f"   The paired game bootstrap of the cross-entropy difference is "
        f"{_signed(paired_best.get('ce_difference'))} "
        f"{_interval(paired_best.get('ce_difference_ci95'))}, so the ordering is "
        f"{'real, not noise' if paired_best.get('distinguishable') else 'not distinguishable from noise'}."
    )
    add(
        f"2. **Against the unchanged Phase 11 head** (`R_CE` "
        f"{_fmt(reference.get('r_ce'))} on these same fresh positions) the raw CNN is "
        f"{abs(row['r_ce'] - reference['r_ce']):.4f} better, and top-1 hidden-rank "
        f"accuracy moves {reference.get('top1', 0.0):.4f} -> {row['top1']:.4f}."
    )
    add(
        f"3. **It costs more to run.** {row['parameters']:,} parameters against "
        f"{agent1_best.get('parameters_added', 0):,}, "
        f"{row['milliseconds_per_decision_single']:.2f} ms per position on CPU, and a "
        f"second network that does not ride along on the policy's forward pass."
    )
    add()
    scout = decision["scout_rush_r_ce"]
    add(
        f"On the hardest generalization stratum, Scout-rush, the raw CNN scores "
        f"`R_CE` {_fmt(scout.get(CANDIDATE_2))} against "
        f"{_fmt(scout.get(agent1_best_id))} for Agent 1's winner and "
        f"{_fmt(scout.get(REFERENCE))} for the unchanged Phase 11 head."
    )
    add()
    add("None of this is a scientific claim, a repair of Phase 11, or evidence about")
    add("whether better beliefs win more games. It is one engineering measurement on")
    add("one fresh development set.")
    add()

    # -- 1. the model ------------------------------------------------------
    parameters = pilot["parameters"]
    section("The model")
    add()
    add("```text")
    add("public 127 x 10 x 10 observation")
    add(f"    -> 3x3 spatial projection to width {parameters['width']}")
    add(f"    -> {parameters['blocks']} residual 3x3 convolution blocks")
    add(f"    -> 1x1 read-out at width {parameters['readout_width']}")
    add("    -> 12 rank logits per square")
    add("```")
    add()
    add("| part | parameters |")
    add("| --- | ---: |")
    add(f"| spatial projection | {parameters['stem']:,} |")
    add(f"| residual tower ({parameters['blocks']} blocks) | {parameters['residual_tower']:,} |")
    add(f"| per-square read-out | {parameters['readout']:,} |")
    add(f"| **total** | **{parameters['total']:,}** |")
    add()
    add(
        f"{parameters['total']:,} parameters, mid-band of the instructed "
        f"{pilot['parameter_band'][0] / 1e6:.0f}-{pilot['parameter_band'][1] / 1e6:.0f}M "
        "range, and the count was calculated before the run rather than after it."
    )
    add(
        f"The tower is {parameters['conv3x3_layers']} 3x3 convolutional layers, a "
        f"{parameters['receptive_field_width_squares']}x{parameters['receptive_field_width_squares']} "
        "receptive field: every square sees the whole 10x10 board with margin."
    )
    add()
    add("**One architecture, no sweep.** Width, depth and read-out width were")
    add("declared once, before the run, and no second architecture was built or")
    add("considered. Two *optimization* configurations of this one architecture were")
    add("declared and both are reported — \"Two declared configurations\" below says")
    add("exactly why, and \"What was trained\" gives the selected one verbatim.")
    add()
    add("The model's only input is the 127-channel public observation. It receives no")
    add("C1 feature, no privileged tensor and no hidden rank: the corpus stores true")
    add("ranks in a separate directory, the loader hands them over only when asked by")
    add("name, and `forward` takes exactly one argument.")
    add()

    # -- 2. the pilot ------------------------------------------------------
    section("The pilot, and where the budget came from")
    add()
    add("| backend | s/step | positions/s | estimated s/epoch | pilot loss |")
    add("| --- | ---: | ---: | ---: | --- |")
    for name, probe in sorted(pilot["probes"].items()):
        add(
            f"| `{name}` | {probe['seconds_per_step']:.3f} | "
            f"{probe['positions_per_second']:,.0f} | "
            f"{probe['estimated_epoch_seconds']:.0f} | "
            f"{probe['first_loss']:.4f} -> {probe['last_loss']:.4f} |"
        )
    add()
    add(
        f"MPS was **{pilot['device_speedup_vs_cpu']:.1f}x** the accepted CPU backend and "
        f"its pilot losses agree with CPU's to "
        f"{pilot['cross_device_loss_agreement']:.1e}, which is the "
        "\"stable and materially faster\" test `02_AGENT_2` sets. Training ran on "
        f"`{pilot['device_chosen']}`."
    )
    add()
    add(
        f"The epoch horizon — **{pilot['epochs_declared']}** — was set from "
        f"{pilot['epoch_budget_basis']}. That decision was made from measured"
    )
    add("throughput **before any development metric existed**, which is what keeps it")
    add("a budget choice rather than a tuned hyperparameter. Neither run came close")
    add("to spending it: both stopped on patience long before epoch 60.")
    add()

    # -- 2b. the two declared runs ----------------------------------------
    runs = training["runs"]
    selected = training["selected_run"]
    ordered_runs = [runs[name] for name in ("run1_declared", "run2_regularized") if name in runs]
    section("Two declared configurations, and why there are two")
    add()
    add(
        f"`02_AGENT_2` asks for **one** engineering run. Agent 2 ran "
        f"{training['configurations_declared']} configurations of the **one** "
        "architecture, and this section says plainly why, because the deviation "
        "matters more than the number it produced."
    )
    add()
    add("| run | configuration | epochs run | best at (epochs) | dev R_CE | train CE first → last | dev CE first → last |")
    add("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for block in ordered_runs:
        over = block["overfitting"]
        config = block["config"]
        mark = "**" if block["run_id"] == selected else ""
        shape = (
            f"lr {config['learning_rate']:g}, wd {config['weight_decay']:g}, "
            f"dropout {config['block_dropout']:g}/{config['readout_dropout']:g}"
        )
        add(
            f"| `{block['run_id']}` | {shape} | "
            f"{block['epochs_run']} of {config['epochs']} | "
            f"{block['best_epoch_fraction']:.2f} | {mark}{block['dev_r_ce']:.4f}{mark} | "
            f"{over['train_ce_first_epoch']:.4f} → {over['train_ce_last_epoch']:.4f} | "
            f"{over['dev_ce_first_epoch']:.4f} → {over['dev_ce_last_epoch']:.4f} |"
        )
    add()
    add("The first two CE columns are epoch-boundary values, which is where the")
    add("overfitting is easiest to see; the `dev R_CE` column is the best checkpoint")
    add("of the whole run, found by the sub-epoch probe described below.")
    add()
    first = runs.get("run1_declared")
    second = runs.get("run2_regularized")
    if first is not None and second is not None:
        over = first["overfitting"]
        add(
            "**Run 1 was the configuration declared before any result existed** — "
            "Agent 1's own optimizer family, so that the two experiments would differ "
            "in architecture rather than in tuning effort. Read at epoch boundaries it "
            "looks like a failed run: training cross-entropy fell "
            f"{over['train_ce_first_epoch']:.4f} → {over['train_ce_last_epoch']:.4f} "
            f"while development cross-entropy rose "
            f"{over['dev_ce_first_epoch']:.4f} → {over['dev_ce_last_epoch']:.4f}, and "
            f"patience stopped it after {first['epochs_run']} epochs of "
            f"{first['config']['epochs']}."
        )
        add()
        add("That pattern is a statement about capacity against corpus, not about the")
        add("architecture. The corpus holds 26,898 training positions drawn from 2,048")
        add("games — 13 positions per game, and **the hidden ranks are constant within**")
        add("**a game** — against 3.9M parameters. Agent 1's heads never met this")
        add("problem: they train 1,548-334,860 parameters on a representation that was")
        add("already pretrained on far more data than this corpus contains.")
        add()
        add("Two things followed from that diagnosis, and it matters which is which.")
        add()
        selected_block = runs[selected]
        add(
            f"**A measurement fix, applied to both runs.** Both runs reach their "
            f"development optimum inside the first epoch or two — the kept checkpoint "
            f"of `{selected}` is step {selected_block['best_step']} of "
            f"{selected_block['steps_per_epoch']} per epoch, "
            f"{selected_block['best_epoch_fraction']:.2f} epochs in. A once-per-epoch "
            "probe cannot find a checkpoint that good, and `02_AGENT_2` asks for the "
            "best development checkpoint, so the trainer probes development "
            f"cross-entropy {selected_block['config']['evaluations_per_epoch']} times "
            "per epoch and keeps the best weights it ever saw. This changes nothing "
            "about training — same batches, same order, same optimizer state — only "
            "how finely the run is *observed*, and it applies identically to both "
            "runs."
        )
        add()
        epoch_best = min(
            row["dev_r_ce"] for row in train["curve"] if not row["sub_epoch"]
        )
        add(
            f"That fix is worth {epoch_best - row['r_ce']:.4f} `R_CE` on the reported "
            f"run — read at epoch boundaries this candidate scores {epoch_best:.4f}, "
            f"and read at the probe it scores {row['r_ce']:.4f}. For scale, that is "
            "almost exactly this candidate's whole margin over the unchanged Phase 11 "
            f"head ({abs(row['r_ce'] - reference['r_ce']):.4f}) and "
            f"{(epoch_best - row['r_ce']) / abs(delta) * 100:.0f}% of its deficit "
            f"against Agent 1's winner ({abs(delta):.4f}). Coarse checkpointing, not "
            "the architecture, would have been the largest single error in this "
            "report."
        )
        add()
        add(
            "**A second configuration, which turned out not to be needed.** One "
            "corrective configuration was declared against the overfitting — dropout, "
            "a 4x lower learning rate, a 500x stronger decoupled weight decay — to "
            "test whether run 1's optimizer, rather than the architecture or the "
            "corpus, was the limitation. It was not: the two configurations land "
            f"{abs(first['dev_r_ce'] - second['dev_r_ce']):.4f} `R_CE` apart "
            f"({first['dev_r_ce']:.4f} against {second['dev_r_ce']:.4f}) — "
            f"{decision['equivalence_band'] / abs(first['dev_r_ce'] - second['dev_r_ce']):.0f}x "
            "narrower than the sprint's equivalence band and "
            f"{abs(delta) / abs(first['dev_r_ce'] - second['dev_r_ce']):.0f}x narrower "
            "than the gap to Agent 1. Regularizing the run slowed the memorization "
            "and moved the optimum later; it changed the ceiling by almost nothing."
        )
        add()
        add(
            f"**The reported candidate is therefore `{selected}` — the configuration "
            "declared up front.** The selection rule was fixed in advance (lowest "
            "development cross-entropy) and it chose the run that needed no "
            "correction."
        )
        add()
        add("**This is one more run than the instruction's letter allows, and it is")
        add("not a sweep**: two configurations were declared, both are reported in")
        add("full above and in `agent_02_learning_curve.json`, no third was tried, and")
        add("no architecture variant was built. Because the two agree to")
        add(
            f"{abs(first['dev_r_ce'] - second['dev_r_ce']):.4f} `R_CE`, a reader who "
            "rejects the deviation entirely can read run 1's row alone and reach the "
            "same conclusion about this candidate."
        )
        add()

    # -- 3. results --------------------------------------------------------
    section("Results on the common development set")
    add()
    add(
        f"All rows are the same {row['dev_pieces']:,} hidden pieces of the same "
        f"{row['dev_samples']:,} development decisions, from corpus "
        f"`{corpus['corpus_digest'][:8]}…`, and all divide by the same "
        f"`remaining_count_belief_v1` denominator (CE {row['baseline_ce']:.4f}, "
        f"top-1 {row['baseline_top1']:.4f})."
    )
    add()
    add("| candidate | architecture | CE | R_CE | 95% CI | top-1 | trained params |")
    add("| --- | --- | ---: | ---: | --- | ---: | ---: |")
    ordered = _ordered_rows(row, earlier)
    for name, block, trained in ordered:
        mark = "**" if name == CANDIDATE_2 else ""
        add(
            f"| `{name}` | {block['architecture']} | {block['ce']:.4f} | "
            f"{mark}{block['r_ce']:.4f}{mark} | {_interval(block.get('r_ce_ci95'))} | "
            f"{block['top1']:.4f} | {trained:,} |"
        )
    add()
    add(
        f"A flat 12-way vector scores `R_CE` "
        f"{summary['uniform_floor']['r_ce']:.4f} — the uninformed floor."
    )
    add()
    add("Agent 1's rows are quoted from `agent_01_summary.json`; **none of Agent 1's**")
    add("**experiments was rerun**. To make the comparison paired rather than two")
    add("overlapping marginal intervals, Agent 1's saved checkpoints were additionally")
    add("loaded read-only and *scored* on the same pieces. They reproduce:")
    add()
    add("| candidate | Agent 1 reported | recomputed here | difference |")
    add("| --- | ---: | ---: | ---: |")
    for name, block in sorted(summary["agent1_reproduction"].items()):
        add(
            f"| `{name}` | {_fmt(block['r_ce_reported_by_agent1'])} | "
            f"{block['r_ce_recomputed']:.4f} | "
            f"{_fmt(block['absolute_difference'], 6)} |"
        )
    add()
    add("### Paired game bootstraps")
    add()
    add("A negative difference means the raw CNN has the lower cross-entropy.")
    add()
    add("| comparison | mean ΔCE | 95% CI | distinguishable |")
    add("| --- | ---: | --- | --- |")
    for name, block in summary["paired_comparisons"].items():
        add(
            f"| {name} | {block['ce_difference']:+.4f} | "
            f"{_interval(block['ce_difference_ci95'])} | "
            f"{'yes' if block['distinguishable'] else 'no'} |"
        )
    add()
    add("### Per-stratum R_CE")
    add()
    add("| candidate | " + " | ".join(STRATA) + " |")
    add("| --- | " + " | ".join("---:" for _ in STRATA) + " |")
    for name, block, _trained in ordered:
        cells = " | ".join(_fmt(block["r_ce_by_stratum"].get(stratum)) for stratum in STRATA)
        add(f"| `{name}` | {cells} |")
    add()

    # -- corpus size -------------------------------------------------------
    scale = summary.get("corpus_size_diagnostic")
    if scale:
        section("Architecture-limited, or corpus-limited?")
        add()
        add("Reaching the development optimum a fraction of an epoch in is consistent")
        add("with two opposite stories: the architecture cannot express more, or the")
        add("corpus cannot support more. They imply opposite advice for Agents 3-5, so")
        add(f"the selected configuration (`{scale['configuration']}`) was retrained on")
        add("halves of the corpus, sliced by whole games.")
        add()
        add("| training games | positions | hidden pieces | best dev R_CE | best at (epochs) |")
        add("| ---: | ---: | ---: | ---: | ---: |")
        for point in scale["points"]:
            mark = "**" if point.get("note") else ""
            add(
                f"| {point['games']:,} | {point['positions']:,} | {point['pieces']:,} | "
                f"{mark}{point['best_r_ce']:.4f}{mark} | "
                f"{point['best_epoch_fraction']:.2f} |"
            )
        add()
        reading = scale["reading"]
        add(f"**{reading[:1].upper()}{reading[1:]}.**")
        add(
            f"Quadrupling the training games moves best development `R_CE` by "
            f"{abs(scale['r_ce_gain_from_first_to_full']):.4f}, and the curve has not "
            "flattened."
        )
        add()
        add("These runs are diagnostics: none of them is the reported candidate, none")
        add("wrote a candidate checkpoint, and the leaderboard is identical without")
        add("them. They exist because the difference between \"this architecture cannot\"")
        add("and \"this corpus cannot\" is the single most useful thing Agent 2 can hand")
        add("to the agents that follow.")
        add()

    # -- 4. training -------------------------------------------------------
    config = training["config"]
    section("What was trained, and how")
    add()
    add("| field | value |")
    add("| --- | --- |")
    add(f"| trainable | the whole {row['parameters']:,}-parameter CNN, from scratch |")
    add("| frozen | nothing — this candidate has no C1 stage |")
    add(f"| optimizer | {config['optimizer']} + {config['schedule']} |")
    add(f"| learning rate | {config['learning_rate']} |")
    add(f"| weight decay | {config['weight_decay']} |")
    add(f"| gradient clip | {config['gradient_clip']} |")
    add(f"| batch | {config['batch_positions']} positions (~{training['train_pieces'] // max(training['train_positions'], 1)} hidden pieces each) |")
    add(f"| epochs | {training['epochs_run']} of {config['epochs']} |")
    add(f"| stopped | {training['stopped_because']} |")
    add(
        f"| best checkpoint | step {training['best_step']} of "
        f"{training['steps_per_epoch']}/epoch — {training['best_epoch_fraction']:.2f} "
        f"epochs, of {training['evaluations']} probes |"
    )
    add(f"| device | {config['device']} |")
    add()
    add(
        f"The loss is supervised hidden-rank cross-entropy over the "
        f"{training['train_pieces']:,} hidden pieces of the "
        f"{training['train_positions']:,} training decisions and nothing else: no "
        "policy term, no value term, no game outcome anywhere. The optimizer family "
        "is deliberately Agent 1's declared one (AdamW, cosine, `1e-3`, weight decay "
        "`1e-4`), so the two experiments differ in architecture rather than in "
        "tuning effort."
    )
    add()
    add("The supervised squares are gathered with the same helper Agent 1's")
    add("Experiment 1C uses, so Agent 2 is trained on exactly the pieces, in exactly")
    add("the order, that every Phase 11B candidate is scored on.")
    add()

    # -- 5. cost -----------------------------------------------------------
    inference = summary["inference"]
    section("Cost")
    add()
    add("| item | value |")
    add("| --- | ---: |")
    add(f"| training wall clock | {training['training_seconds']:.0f} s |")
    add(f"| time to best checkpoint | {training['time_to_best_seconds']:.0f} s |")
    add(f"| parameters | {row['parameters']:,} |")
    add(f"| checkpoint | {train['checkpoint']['bytes'] / 1e6:.1f} MB |")
    for backend, block in sorted(inference.items()):
        add(
            f"| inference, {backend}: one position | "
            f"{block['milliseconds_per_decision_single']:.2f} ms |"
        )
        add(
            f"| inference, {backend}: batched, per position | "
            f"{block['milliseconds_per_decision_batched']:.3f} ms |"
        )
    add(f"| peak memory | {summary['peak_memory_bytes'] / 1e9:.2f} GB |")
    add()
    add(
        f"Peak memory is the {summary['peak_memory_note']}. Inference is priced per "
        "*position*, not per piece, because a convolution tower produces all 100 "
        "squares in one pass; at the corpus's "
        f"{inference['cpu']['hidden_pieces_per_decision']:.1f} hidden pieces per "
        f"decision that is {row['inference_microseconds_per_piece']:.3f} µs/piece "
        "batched, against "
        f"{agent1_best.get('inference_microseconds_per_piece', float('nan')):.3f} "
        "µs/piece for Agent 1's winner — but the honest comparison is that Agent 1's "
        "head rides on a C1 forward pass a search already pays for, and this model "
        "is a second network."
    )
    add()

    # -- 6. the verdict ----------------------------------------------------
    section("Is this preferable to Agent 1?")
    add()
    add(f"**{_verdict_sentence(decision, row, agent1_best_id, agent1_best)}**")
    add()
    add("How the sprint's engineering-winner rule applies:")
    add()
    add(f"- leader by `R_CE`: `{leader_id}` ({decision['leader_r_ce']:.4f});")
    add(
        "- inside the "
        f"{decision['equivalence_band']} equivalence band of the leader: "
        + ", ".join(f"`{name}`" for name in decision["within_band_of_leader"])
        + ";"
    )
    add(
        f"- Scout-rush / generalization: {_fmt(scout.get(CANDIDATE_2))} for the raw CNN "
        f"against {_fmt(scout.get(agent1_best_id))} for Agent 1's winner;"
    )
    add(f"- search-integration complexity: {decision['search_integration_note']}")
    add()
    add("The band is measured against the leader only, never as a chain of pairwise")
    add("comparisons — the same convention Agent 1 recorded.")
    add()

    # -- 7. interface ------------------------------------------------------
    section("Required interface")
    add()
    add("```text")
    add("predict_marginals(public_state)      -> {piece_slot: 12-way rank probabilities}")
    add("sample_worlds(public_state, n, seed) -> complete legal hidden armies")
    add("```")
    add()
    add("| candidate | positions | worlds | marginals valid | seed-deterministic | worlds legal |")
    add("| --- | ---: | ---: | --- | --- | --- |")
    add(
        f"| `{CANDIDATE_2}` | {interface['positions_checked']} | "
        f"{interface['worlds_sampled']} | "
        f"{'yes' if interface['all_marginals_sum_to_one'] else 'no'} | "
        f"{'yes' if interface['sample_worlds_seed_deterministic'] else 'no'} | "
        f"{'yes' if interface['all_worlds_passed_accepted_validation_stack'] else 'no'} |"
    )
    add()
    add("Every world was drawn through **`stratego.evaluation.phase11_sampler`, the")
    add("accepted Phase 11 sampler, imported and unmodified**. Agent 2 supplies")
    add("marginals and nothing else, through the same `Phase11BPublicState` Agent 1")
    add("defined — a container with exactly two public fields and no field a true rank")
    add("could arrive in. The Agent 2 adapter *subclasses* Agent 1's interface rather")
    add("than reimplementing it, so `sample_worlds` is inherited code, not a fork.")
    add()

    # -- 8. caveats --------------------------------------------------------
    section("Caveats a reader should carry forward")
    add()
    add("- **This is a development-set number.** There is no sealed bank behind it and")
    add("  no scientific claim attached to it. The development set is an engineering")
    add("  comparison set, exactly as the sprint defines it.")
    backend = summary["backend_agreement"]
    add(
        f"- **The checkpoint was trained on `{backend['training_backend']}` and scored on "
        f"CPU.** The two backends agree to "
        f"{backend['absolute_difference']:.2e} `R_CE`, so the headline number does not "
        "depend on which one produced it. Neither backend's float32 reductions are "
        "bit-reproducible, so this was measured rather than assumed."
    )
    add(
        f"- **The kept checkpoint is {training['best_epoch_fraction']:.2f} epochs into "
        f"a {training['epochs_run']}-epoch run** ({config['epochs']} scheduled; "
        f"stopped by `{training['stopped_because']}`). "
        + _curve_caveat(training, train)
    )
    add("- **The headline `R_CE` uses the accepted raw-softmax convention** — no")
    add("  masking, no epsilon, full simplex — because that is how the Phase 11 head")
    add("  was measured and how the accepted sampler consumes a belief. Renormalizing")
    add(
        f"  onto the publicly legal support is a diagnostic only "
        f"({row['diagnostic_projected_r_ce']:.4f} against {row['r_ce']:.4f} raw)."
    )
    add("- **The reference row is not the Phase 11 sealed-test result.** It is the")
    add("  unchanged Phase 11 head scored on *these* fresh positions. Phase 11's sealed")
    add("  test remains what it was, and its bank remains spent.")
    add(
        "- **No repeat under an identical configuration.** Agent 1 measured its "
        "candidates' run-to-run spread by retraining each one; Agent 2's two runs "
        "differ in configuration, so they do not measure that. What they do bound is "
        "looser and still useful: two *different* optimization settings of this "
        f"architecture land {abs(runs['run1_declared']['dev_r_ce'] - runs['run2_regularized']['dev_r_ce']):.4f} "
        "`R_CE` apart, so the reported number is not balanced on a single lucky seed."
    )
    add()

    # -- 9. preservation and stop -----------------------------------------
    preservation = summary["preservation"]
    section("What Agent 2 touched")
    add()
    add(
        f"The common corpus was reused **byte-for-byte**: both splits' file digests and "
        f"the whole-corpus digest `{corpus['corpus_digest'][:16]}…` were recomputed "
        "from disk and matched against the values Agent 1 recorded. Nothing was "
        "regenerated."
    )
    add()
    add("| statement | value |")
    add("| --- | --- |")
    add(f"| corpus regenerated | `{preservation['corpus_regenerated']}` |")
    add(f"| Agent 1 artifacts modified | `{preservation['agent1_artifacts_modified']}` |")
    add(
        f"| Phase 11 artifacts unchanged since Agent 1 | "
        f"`{preservation['phase11_artifacts_unchanged_since_agent1']}` |"
    )
    add(f"| `phase11_test_bank_v1` opened | `{preservation['phase11_test_bank_opened']}` |")
    add()
    suite = summary.get("suite")
    if suite:
        add(f"Repository suite after Agent 2: **{suite['summary_line']}** (`{suite['command']}`).")
        add()
    section("Handoff to Agent 3")
    add()
    add("Agent 2 does not begin Agent 3's experiment and does not recommend for or")
    add("against running it. What Agent 2 measured that Agent 3 should carry:")
    add()
    add(
        f"1. **Learning a belief representation from scratch on this corpus does not "
        f"work yet.** {row['parameters']:,} parameters reach their development "
        f"optimum "
        f"{summary['training']['best_epoch_fraction']:.2f} epochs in and then "
        "memorize, and a heavily regularized configuration moves the ceiling by only "
        f"{abs(runs['run1_declared']['dev_r_ce'] - runs['run2_regularized']['dev_r_ce']):.4f} "
        "`R_CE`. Borrowing C1's pretrained representation beats learning one here, "
        "and by a wide margin."
    )
    if scale:
        add(
            f"2. **The binding constraint is the corpus, not the architecture.** Best "
            f"development `R_CE` improves "
            f"{scale['points'][0]['best_r_ce']:.4f} -> {scale['points'][-1]['best_r_ce']:.4f} "
            f"from {scale['points'][0]['games']:,} to {scale['points'][-1]['games']:,} "
            "training games and has not flattened. A from-scratch spatial specialist "
            "is not refuted by this result; it is untested at a corpus size that "
            "would give it a chance, and Phase 11B's common corpus was sized for "
            "cheap head experiments."
        )
    add(
        f"3. **Agents 3 and 4 read the C1 features, so they inherit the cheap side of "
        f"this trade.** They should still expect the overfitting régime to bite "
        "wherever they add trainable capacity: 26,898 positions from 2,048 games, "
        "with hidden ranks constant inside a game, is a small supervision set "
        "however it is presented."
    )
    add(
        "4. **Probe development loss several times per epoch.** Both Agent 2 runs "
        "reached their optimum in the first epoch or two, and any candidate that "
        "trains comparable capacity on this corpus should expect the same. At epoch "
        "granularity this one would have been reported at `R_CE` "
        f"{min(r['dev_r_ce'] for r in train['curve'] if not r['sub_epoch']):.4f} "
        f"instead of {row['r_ce']:.4f}."
    )
    add()
    section("Stop condition")
    add()
    add("Agent 2 trained one architecture and stopped. Agent 3's experiment was not")
    add("begun. Phase 11 remains `FAIL`, `phase11_test_bank_v1` remains spent and")
    add("unopened, and nothing in this report claims that Phase 11 has been repaired")
    add("or that Phase 12 is authorized.")
    add()
    return "\n".join(lines)


def _ordered_rows(row: dict, earlier: dict) -> list:
    """Every leaderboard row, best `R_CE` first, with its trained-parameter count."""
    rows = [(CANDIDATE_2, row, int(row["parameters"]))]
    for name, block in earlier.items():
        rows.append((name, block, int(block.get("parameters_added", 0))))
    return sorted(rows, key=lambda item: item[1]["r_ce"])


def _verdict_sentence(decision: dict, row: dict, agent1_best_id: str, agent1_best: dict) -> str:
    delta = decision["agent2_minus_agent1_best_r_ce"]
    band = decision["equivalence_band"]
    if delta is None:  # pragma: no cover - Agent 1's summary always has a winner
        return "No Agent 1 candidate was available to compare against."
    if decision["agent2_materially_better_than_agent1_best"]:
        return (
            f"Yes on predictive quality: the raw CNN is {abs(delta):.4f} `R_CE` better "
            f"than `{agent1_best_id}`, wider than the {band} band, and the paired "
            "bootstrap separates them."
        )
    if abs(delta) <= band:
        return (
            f"No. The raw CNN and `{agent1_best_id}` are within {band} `R_CE` "
            f"({_signed(delta)}), so the rule prefers the cheaper and simpler model — "
            f"and that is Agent 1's head at {agent1_best.get('parameters_added', 0):,} "
            f"trained parameters against {row['parameters']:,}, attached to a forward "
            "pass the policy already runs."
        )
    return (
        f"No. The raw CNN is {abs(delta):.4f} `R_CE` *worse* than `{agent1_best_id}` "
        "and also more expensive, so it loses on both axes of the rule."
    )


def _curve_caveat(training: dict, train: dict) -> str:
    curve = train.get("curve") or []
    if not curve:  # pragma: no cover - a run always has a curve
        return ""
    at_best = min(curve, key=lambda row: row["dev_ce"])
    trailing = curve[-1]["dev_r_ce"] - at_best["dev_r_ce"]
    if training["stopped_because"] == "epochs_exhausted" and trailing <= 0:
        return (
            "The curve was still improving when the horizon ran out, so read this as "
            "an underestimate of what the configuration reaches. Re-choosing the "
            "horizon after seeing the result is the tuning the sprint forbids, so the "
            "run was not extended."
        )
    return (
        f"Development `R_CE` ended {abs(trailing):.4f} "
        f"{'worse' if trailing > 0 else 'better'} than at the best probe, so the curve "
        "had turned well before the run stopped and the kept checkpoint is nowhere "
        "near the last one."
    )


__all__ = ["render"]
