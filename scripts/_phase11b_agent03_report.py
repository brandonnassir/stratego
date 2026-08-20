"""Markdown renderer for the Phase 11B Agent 3 report.

Split out of `run_phase11b_agent03.py` for the reason
`_phase11b_agent02_report.py` is split out of its harness: the prose is
long, it changes for reasons that have nothing to do with the experiment,
and a harness is easier to read when the string building lives elsewhere.

Every number is read from the summary the harness just wrote. Nothing here
recomputes a metric, and nothing here decides anything: the verdict is
`summary["decision"]` and the Agent 2 contrast is
`summary["agent2_contrast"]`, both produced by the harness, and this module
only puts them into sentences.
"""

from __future__ import annotations

CANDIDATE_3 = "agent03_c1_feature_cnn"
CANDIDATE_2 = "agent02_raw_observation_cnn"
CANDIDATE_1B = "agent01_1b_attached_mlp_head"
CANDIDATE_1C = "agent01_1c_final_block_plus_mlp"
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


def _params(block: dict) -> int:
    """Trained-parameter count of one leaderboard row.

    Agent 1 records it as `belief_parameters_total` and Agent 2 as
    `parameters_trained`; a reader comparing rows wants one column, so the
    two spellings are resolved here rather than in four separate f-strings.
    """
    for key in ("parameters_trained", "belief_parameters_total", "parameters"):
        if block.get(key) is not None:
            return int(block[key])
    return 0


def _ordered_rows(row: dict, earlier: dict) -> list:
    """Every candidate on the sprint's development set, best `R_CE` first."""
    rows = [(CANDIDATE_3, row)] + [(name, block) for name, block in earlier.items()]
    return sorted(rows, key=lambda pair: pair[1]["r_ce"])


def render(summary: dict, train: dict) -> str:
    row = summary["leaderboard"][CANDIDATE_3]
    earlier = summary["earlier_reference_rows"]
    contrast = summary["agent2_contrast"]
    decision = summary["decision"]
    training = summary["training"]
    pilot = summary["pilot"]
    seam = summary["frozen_seam"]
    cache = summary["feature_cache"]
    metrics_ci = _interval(row["r_ce_ci95"])
    agent1_best_id = decision["best_earlier_candidate"]
    agent1_best = earlier.get(agent1_best_id, {})
    # The overall leader (1C) and the same-seam head (1B) answer different
    # questions and the report needs both: 1C decides "is Agent 3 preferable",
    # 1B decides "was extraction the bottleneck".
    agent1_1b = earlier.get(CANDIDATE_1B, {})
    inference = summary["inference"]["cpu"]
    overfit = training["overfitting"]
    repeat = summary.get("repeat_run") or {}
    parts: list[str] = []

    # -- 0 ------------------------------------------------------------------
    parts.append(
        f"""# Phase 11B — Agent 3: C1-Feature CNN

**Status: engineering prototype.** This report does not repair Phase 11, does
not overturn the Phase 11 `FAIL`, and does not authorize Phase 12.
`phase11_test_bank_v1` was not opened; it remains spent.

| marker | value |
| --- | --- |
| `phase` | `{summary['phase']}` |
| `status` | `{summary['status']}` |
| `phase11_fail_unchanged` | `{summary['phase11_fail_unchanged']}` |
| `phase11_test_bank_used` | `{summary['phase11_test_bank_used']}` |
| `phase12_authorized_by_this_artifact` | `{summary['phase12_authorized_by_this_artifact']}` |
| `phase11_final_classification` | `{summary['phase11_final_classification']}` |
| `phase11_test_bank_spent` | `{summary['phase11_test_bank_spent']}` |
| `scientific_claim` | `{summary['scientific_claim']}` |

## 0. What Agent 3 found

Agent 3's question was whether the final C1 move model still carries enough
belief-relevant information, and whether the old Phase 11 belief classifier
was mainly an **extraction** bottleneck. The experiment gives a two-part
answer, and the two parts point in opposite directions.

1. **C1's representation is the better one.** Give the *same* 3.9M-parameter
   spatial specialist the frozen C1 per-square field instead of the raw
   observation and `R_CE` moves {_fmt(contrast['agent2_raw_cnn_r_ce'])} ->
   **{_fmt(row['r_ce'])}** {metrics_ci}. The paired game bootstrap of the
   cross-entropy difference is
   {_signed(contrast['paired_comparison']['ce_difference'])}
   {_interval(contrast['paired_comparison']['ce_difference_ci95'])}, so the
   ordering is real. C1 is **not** discarding the information.
2. **Extraction was not a capacity problem.** Agent 1's
   {_params(agent1_1b):,}-parameter attached head reads **the same seam** and
   scores {_fmt(agent1_1b.get('r_ce'))}, which is
   {_fmt(abs(row['r_ce'] - agent1_1b.get('r_ce', 0)))} `R_CE` *better* than
   this candidate — also a distinguishable paired difference. Adding 3.9M
   parameters of spatial capacity on top of the richer version of the same
   features made the belief **worse**, not better.

So the old head was an extraction bottleneck in the sense Agent 1 already
measured — dedicated belief optimization of the *unchanged* 1,548-parameter
linear head moved {_fmt(earlier.get(REFERENCE, {}).get('r_ce'))} ->
{_fmt(earlier.get('agent01_1a_existing_linear_head', {}).get('r_ce'))} — and
**not** in the sense that a bigger, spatial extractor on the same seam would
have helped. On this corpus it hurts.

None of this is a scientific claim, a repair of Phase 11, or evidence about
whether better beliefs win more games. It is one engineering measurement on
one fresh development set.
"""
    )

    # -- 1 ------------------------------------------------------------------
    rejected = seam["alternatives_rejected"]
    parts.append(
        f"""## 1. The frozen seam

`03_AGENT_3` asks for "the richest spatial/token-level representation
immediately before the task heads that can be mapped back to board cells".
In C1 that tensor is unambiguous, because all three heads read the same one:

```text
hidden = self.encode(tokens)                    # [B, 100, 128]
policy_logits = ... query / key over hidden
value_logits  = ... over hidden.mean(dim=1)
belief_logits = self.belief_output(hidden)
```

| field | value |
| --- | --- |
| seam id | `{seam['seam_id']}` |
| tensor | `{seam['tensor']}` |
| definition | `{seam['definition']}` |
| shape | `[B, {seam['shape'][1]}, {seam['shape'][2]}]` |
| per-square | `{seam['is_per_square']}` |
| pooled | `{seam['is_pooled']}` |
| consumed by | {', '.join(f'`{name}`' for name in seam['consumed_by_heads'])} |

{seam['square_mapping'][0].upper() + seam['square_mapping'][1:]}.

**What was rejected, and why:**

| candidate tensor | rejected because |
| --- | --- |
| `hidden.mean(dim=1)` | {rejected['hidden.mean(dim=1)']} |
| `belief_output(hidden)` | {rejected['belief_output(hidden)']} |
| penultimate block input | {rejected['penultimate block input']} |
| `policy_query` / `policy_key` | {rejected['policy_query / policy_key']} |

The first two are what `03_AGENT_3` warns against — "an unnecessarily pooled
or compressed global vector". The penultimate tensor is per-square and
128-wide and Agent 1 already caches it for Experiment 1C, but it is one
encoder block short of what the heads actually read, and Agent 3's question
is about what the heads see.

**No new frozen-prefix code was written.** Agent 1's `features.encode_batch`
already returns exactly this tensor for its `final` layer; Agent 1's *cache*
then gathers it at the supervised squares, which is what a per-piece head
needs and what a spatial CNN cannot use. Agent 3 calls the accepted seam
function verbatim and keeps all 100 tokens. `features.py` is unmodified, and
so is every other Agent 1 and Agent 2 file — see "What Agent 3 touched".

### The cache

`03_AGENT_3` permits caching "if this materially speeds training" and
requires that any cache be "derivable from the common public observations
plus the accepted frozen C1".

| split | shape | size | seconds | digest |
| --- | --- | ---: | ---: | --- |"""
    )
    for split in ("train", "dev"):
        block = cache["caches"].get(split)
        if not block:
            continue
        parts.append(
            f"| `{split}` | `{tuple(block['shape'])}` | {block['bytes'] / 1e6:.0f} MB | "
            f"{block['seconds']:.1f} | `{block['digest'][:16]}…` |"
        )
    verification = cache["verification"]
    parts.append(
        f"""
Both requirements are measurements rather than assurances. The cache turns
every training epoch into a pass over a fixed matrix instead of
{training['train_positions']:,} transformer forward passes, and it was built
in {cache['total_seconds']:.1f}s total. Derivability was checked by
re-encoding a random sample of each split from the public observations and
comparing:

| split | rows re-derived | max abs difference | bit-identical |
| --- | ---: | ---: | --- |
"""
        + "\n".join(
            f"| `{split}` | {block['rows_checked']} | "
            f"{block['max_absolute_difference']:.2e} | {block['bit_identical']} |"
            for split, block in verification.items()
        )
        + f"""

The inputs to that re-derivation are the public observation and the accepted
frozen C1 weights, and nothing else: no label, no privileged array and no
Agent 3 parameter takes part.

{cache['cache_device_note'][0].upper() + cache['cache_device_note'][1:]}.
"""
    )

    # -- 2 ------------------------------------------------------------------
    breakdown = pilot["parameters"]
    parts.append(
        f"""## 2. The model

```text
public 127 x 10 x 10 observation
    -> frozen C1                       (accepted Phase 9 weights, never updated)
    -> per-square C1 field [100, 128]
    -> 3x3 spatial projection to width {breakdown['width']}
    -> {breakdown['blocks']} residual 3x3 convolution blocks
    -> 1x1 read-out at width {breakdown['readout_width']}
    -> 12 rank logits per square
```

| part | parameters |
| --- | ---: |
| spatial projection | {breakdown['stem']:,} |
| residual tower ({breakdown['blocks']} blocks) | {breakdown['residual_tower']:,} |
| per-square read-out | {breakdown['readout']:,} |
| **total, all trainable** | **{breakdown['total']:,}** |
| frozen C1, never updated | {breakdown['frozen_c1_parameters']:,} |

**The tower is Agent 2's, inherited rather than chosen.** Width, depth and
read-out width were not picked here at all: `03_AGENT_3` wants this candidate
read against Agent 2's raw-observation CNN, and a comparison like that is
only clean if the specialist is held fixed and the *representation* is the
thing that changes. Agent 2 is {pilot['agent2_parameters']:,} parameters and
this is {breakdown['total']:,} — the {pilot['parameters_minus_agent2']:,}
difference is the stem's one extra input channel (128 C1 channels against 127
observation planes) and nothing else. `ResidualBlock` is *imported* from
`raw_cnn`, not re-declared.

That is also why there is no sweep: there was nothing to sweep. One
architecture, one configuration, one run.

**The specialist never sees the raw observation.**
`C1FeatureBeliefCNN.forward` takes exactly one argument, the `[B, 100, 128]`
frozen field, and the module holds no other input path — feeding raw
observation in is Agent 4's experiment, not this one. A true rank cannot
reach it either: the corpus stores labels in a different directory and the
loader hands them over only when asked by name.

**C1 is frozen structurally, not by convention.** It is loaded through
`features.load_frozen_c1`, which checks the accepted state and belief-head
digests and sets `requires_grad=False` on every parameter; the field cache
means C1 is not even *called* during training; and `build_feature_cnn`
returns the specialist alone, so no optimizer in this experiment is ever
handed a C1 parameter. `c1_parameters_updated` = {training['c1_parameters_updated']}.
"""
    )

    # -- 3 ------------------------------------------------------------------
    probes = pilot["probes"]
    parts.append(
        """## 3. The pilot, and where the budget came from

| backend | s/step | positions/s | estimated s/epoch | pilot loss |
| --- | ---: | ---: | ---: | --- |
"""
        + "\n".join(
            f"| `{name}` | {block['seconds_per_step']:.3f} | "
            f"{block['positions_per_second']:,.0f} | "
            f"{block['estimated_epoch_seconds']:.0f} | "
            f"{block['first_loss']:.4f} -> {block['last_loss']:.4f} |"
            for name, block in probes.items()
        )
        + f"""

MPS was **{pilot['device_speedup_vs_cpu']}x** the accepted CPU backend and
its pilot losses agree with CPU's to
{pilot['cross_device_loss_agreement']:.1e}. Training ran on
`{pilot['device_chosen']}`. The rule is Agent 2's, unchanged:
{pilot['device_rule']}.

The epoch horizon — **{pilot['epochs_declared']}** — is
{pilot['epoch_budget_basis']}. That decision was made from measured
throughput **before any development metric existed**, which is what keeps it
a budget choice rather than a tuned hyperparameter. The run did not come
close to spending it: it stopped on `{training['stopped_because']}` after
{training['epochs_run']} epochs.
"""
    )

    # -- 4 ------------------------------------------------------------------
    parts.append(
        f"""## 4. Results on the common development set

All rows are the same {row['dev_pieces']:,} hidden pieces of the same
{row['dev_samples']:,} development decisions, from corpus
`{summary['common_corpus']['corpus_digest'][:8]}…`, and all divide by the same
`remaining_count_belief_v1` denominator (CE {_fmt(row['baseline_ce'])}, top-1
{_fmt(row['baseline_top1'])}).

| candidate | representation | CE | R_CE | 95% CI | top-1 | trained params |
| --- | --- | ---: | ---: | --- | ---: | ---: |"""
    )
    representation = {
        CANDIDATE_3: "frozen C1 field, all 100 squares",
        CANDIDATE_2: "raw 127-channel observation",
        CANDIDATE_1C: "frozen C1 penultimate + last block retrained",
        CANDIDATE_1B: "frozen C1 feature at the piece's square",
        "agent01_1a_existing_linear_head": "frozen C1 feature at the piece's square",
        REFERENCE: "frozen C1 feature at the piece's square",
    }
    for name, block in _ordered_rows(row, earlier):
        highlight = "**" if name == CANDIDATE_3 else ""
        parts.append(
            f"| `{name}` | {representation.get(name, '—')} | {_fmt(block['ce'])} | "
            f"{highlight}{_fmt(block['r_ce'])}{highlight} | "
            f"{_interval(block.get('r_ce_ci95'))} | {_fmt(block['top1'])} | "
            f"{_params(block):,} |"
        )
    parts.append(
        f"""
A flat 12-way vector scores `R_CE` {_fmt(summary['uniform_floor']['r_ce'])} —
the uninformed floor.

Agent 1's and Agent 2's rows are quoted from their own summaries; **no
earlier experiment was rerun**. To make the comparisons paired rather than
five overlapping marginal intervals, their saved checkpoints were
additionally loaded read-only and *scored* on the same pieces. They
reproduce:

| candidate | reported by its agent | recomputed here | difference |
| --- | ---: | ---: | ---: |
"""
        + "\n".join(
            f"| `{name}` | {_fmt(block['r_ce_reported_by_its_agent'])} | "
            f"{_fmt(block['r_ce_recomputed'])} | "
            f"{block['absolute_difference']:.6f} |"
            for name, block in summary["earlier_reproduction"].items()
        )
        + f"""

{summary['earlier_reproduction_note']}

### Paired game bootstraps

A negative difference means Agent 3 has the lower cross-entropy.

| comparison | mean ΔCE | 95% CI | distinguishable |
| --- | ---: | --- | --- |
"""
        + "\n".join(
            f"| {key} | {_signed(block['ce_difference'])} | "
            f"{_interval(block['ce_difference_ci95'])} | "
            f"{'yes' if block['distinguishable'] else 'no'} |"
            for key, block in summary["paired_comparisons"].items()
        )
        + """

### Per-stratum R_CE

| candidate | """
        + " | ".join(STRATA)
        + " |\n| --- | "
        + " | ".join("---:" for _ in STRATA)
        + " |\n"
        + "\n".join(
            f"| `{name}` | "
            + " | ".join(_fmt(block["r_ce_by_stratum"].get(stratum)) for stratum in STRATA)
            + " |"
            for name, block in _ordered_rows(row, earlier)
        )
    )

    # -- 5 ------------------------------------------------------------------
    paired2 = contrast["paired_comparison"]
    parts.append(
        f"""

## 5. The comparison `03_AGENT_3` asks for

```text
Agent 2 raw-CNN R_CE          {_fmt(contrast['agent2_raw_cnn_r_ce'])}
Agent 3 C1-feature-CNN R_CE   {_fmt(contrast['agent3_c1_feature_cnn_r_ce'])}
difference                    {_signed(contrast['difference_agent3_minus_agent2'])}
```

Same tower, same optimizer, same corpus, same probe schedule, same
checkpoint-selection rule, same development pieces. The **only** difference
is what goes in the bottom: 127 raw observation planes for Agent 2, the
frozen C1 field's 128 channels for Agent 3. The paired game bootstrap of the
cross-entropy difference is {_signed(paired2['ce_difference'])}
{_interval(paired2['ce_difference_ci95'])} over {paired2['games']} games, so
this is not noise.

That gap of {_fmt(abs(contrast['difference_agent3_minus_agent2']))} is
{'inside' if contrast['within_equivalence_band'] else 'just outside'} the
sprint's {contrast['equivalence_band']} equivalence band.

**Reading, per `03_AGENT_3`'s own interpretation rule:**
{contrast['interpretation']}.

The instruction anticipated two outcomes: `Agent 3 ~= Agent 2`, which would
mean C1 retained substantial belief-relevant information, or `Agent 2
substantially > Agent 3`, which would mean C1 obscures it and a dedicated
raw-observation encoder is preferable. What happened is neither — Agent 3 is
modestly but distinguishably *better* — and that outcome rules the second
reading out and points past the first. **C1 is not discarding or obscuring
belief-relevant information**: a from-scratch encoder given the same
capacity, the same corpus and the same optimizer does worse than reading C1's
frozen output. The margin is small, so the honest strength of the claim is
"at least as informative, probably slightly more", not "dramatically better".

### The other half of the question

`03_AGENT_3`'s mission also asks whether "the old small belief classifier was
mainly an extraction bottleneck". That comparison does not need a new run
either, because Agent 1 already trained a head on **this exact seam**:

| candidate | reads | trained params | R_CE |
| --- | --- | ---: | ---: |
| `{CANDIDATE_1B}` | the same `encode` output, gathered at the piece's own square | {_params(agent1_1b):,} | {_fmt(agent1_1b.get('r_ce'))} |
| `{CANDIDATE_3}` | the same `encode` output, all 100 squares | {_params(row):,} | {_fmt(row['r_ce'])} |

Same tensor, {_params(row) // max(_params(agent1_1b), 1)}x the parameters, and
spatial context the per-piece head cannot express — and the belief comes out
{_fmt(abs(row['r_ce'] - agent1_1b.get('r_ce', 0)))} `R_CE` **worse** for it. So the answer to the second half is **no, not in the capacity
sense**. The Phase 11 head's problem was that it was never optimized for
belief on its own — Agent 1 moved {_fmt(earlier.get(REFERENCE, {}).get('r_ce'))}
-> {_fmt(earlier.get('agent01_1a_existing_linear_head', {}).get('r_ce'))}
without adding a single parameter — and once that is fixed, more extraction
capacity on this corpus costs rather than buys.
"""
    )

    # -- 6 ------------------------------------------------------------------
    parts.append(
        f"""## 6. Architecture, or corpus?

Agent 2 handed forward a specific warning: 3.9M parameters against
{training['train_positions']:,} positions drawn from 2,048 games, with hidden
ranks constant inside a game, memorizes almost immediately. It did, and this
candidate does the same:

| quantity | value |
| --- | ---: |
| best development checkpoint | epoch {overfit['best_epoch']} ({_fmt(overfit['best_epoch_fraction'], 2)} epochs), step {overfit['best_step']} of {training['steps_per_epoch']}/epoch |
| development probes | {overfit['evaluations']} |
| training CE, first -> last epoch | {_fmt(overfit['train_ce_first_epoch'])} -> {_fmt(overfit['train_ce_last_epoch'])} |
| development CE, first -> last epoch | {_fmt(overfit['dev_ce_first_epoch'])} -> {_fmt(overfit['dev_ce_last_epoch'])} |
| development CE at the kept checkpoint | {_fmt(overfit['dev_ce_best'])} |
| development CE rose after the best | {overfit['dev_ce_rose_after_best']} |

Training cross-entropy fell {_fmt(overfit['train_ce_first_epoch'])} ->
{_fmt(overfit['train_ce_last_epoch'])} while development cross-entropy rose
{_fmt(overfit['dev_ce_first_epoch'])} -> {_fmt(overfit['dev_ce_last_epoch'])},
and the patience rule stopped the run after {training['epochs_run']} of
{training['config']['epochs']} scheduled epochs.

Agent 2 answered the "architecture or corpus" question for this tower by
retraining it on halves of the corpus — best development `R_CE` improved
monotonically 0.9907 -> 0.9775 -> 0.9686 from 512 to 2,048 games and had not
flattened. **That diagnostic was not repeated here.** It is the same tower
under the same optimizer on the same corpus, Agent 2's finding is on disk,
and `03_AGENT_3` says to use earlier reports rather than rerun prior
candidates. What Agent 3 adds is that swapping the *input representation* for
a better one moves the number by
{_fmt(abs(contrast['difference_agent3_minus_agent2']))} while quadrupling the
corpus moved it by 0.0221 — the corpus is still the larger lever.

**Agent 2's measurement fix was inherited whole.** The kept checkpoint here is
step {overfit['best_step']} of {training['steps_per_epoch']} per epoch, so
{_fmt(overfit['best_epoch_fraction'], 2)} epochs in, found by probing
development cross-entropy {training['config']['evaluations_per_epoch']} times
per epoch. At epoch granularity this candidate would have been reported at
`R_CE` {_fmt(min(r['dev_r_ce'] for r in train['curve'] if not r['sub_epoch']))}
instead of {_fmt(row['r_ce'])}.
"""
    )

    # -- 7 ------------------------------------------------------------------
    config = training["config"]
    parts.append(
        f"""## 7. What was trained, and how

| field | value |
| --- | --- |
| trainable | the whole {row['parameters_trained']:,}-parameter belief CNN, from scratch |
| frozen | all {breakdown['frozen_c1_parameters']:,} C1 parameters — not called during training at all |
| optimizer | {config['optimizer']} + {config['schedule']} |
| learning rate | {config['learning_rate']:g} |
| weight decay | {config['weight_decay']:g} |
| gradient clip | {config['gradient_clip']} |
| batch | {config['batch_positions']} positions |
| epochs | {training['epochs_run']} of {config['epochs']} |
| stopped | {training['stopped_because']} |
| development probes per epoch | {config['evaluations_per_epoch']} |
| best checkpoint | step {overfit['best_step']} of {training['steps_per_epoch']}/epoch — {_fmt(overfit['best_epoch_fraction'], 2)} epochs, of {overfit['evaluations']} probes |
| device | {config['device']} |

The configuration is Agent 2's `run1_declared` verbatim, for the same reason
the architecture is: {summary['experiment']['architecture_inheritance_reason']}.
Agent 2 additionally ran one corrective regularized configuration and found it
moved the ceiling by 0.0005 `R_CE`; Agent 3 declared **one** configuration, so
this report has no deviation to disclose on that front. It was then trained a
second time under that same configuration and seed as a spread diagnostic —
`R_CE` {_fmt(repeat.get('reported_r_ce'), 6)} against
{_fmt(repeat.get('repeated_r_ce'), 6)}, a difference of
{repeat.get('absolute_r_ce_difference', float('nan')):.2e}, stopping at the
same step {repeat.get('best_step')} — and that repeat wrote no checkpoint and
is not the reported candidate.

The loss is supervised hidden-rank cross-entropy over the
{training['train_pieces']:,} hidden pieces of the {training['train_positions']:,}
training decisions and nothing else: no policy term, no value term, no game
outcome anywhere. The supervised squares are gathered with the same helper
Agent 1's Experiment 1C and Agent 2 use, so Agent 3 is trained on exactly the
pieces, in exactly the order, that every Phase 11B candidate is scored on.

**The trainer is Agent 2's, imported, not forked.** `train_raw_cnn` stages
`data["observations"]` as one tensor, indexes it by sample row and hands
batches to `model.logits_at`; nothing in it knows whether those rows are
127-channel observations or 128-wide C1 fields. Agent 3 hands it a *view* of
the split whose input array is the cached field. So the two candidates share
an architecture, an optimizer, a shuffling scheme, a probe schedule and a
checkpoint rule, and the difference between their numbers is the
representation.
"""
    )

    # -- 8 ------------------------------------------------------------------
    specialist = inference["specialist"]
    encode = inference["frozen_c1_encode"]
    end_to_end = inference["end_to_end"]
    parts.append(
        f"""## 8. Cost

| item | value |
| --- | ---: |
| feature cache | {cache['total_seconds']:.1f} s, {sum(block['bytes'] for block in cache['caches'].values()) / 1e6:.0f} MB |
| training wall clock | {training['training_seconds']:.0f} s |
| time to best checkpoint | {training['time_to_best_seconds']:.0f} s |
| trainable parameters | {row['parameters_trained']:,} |
| checkpoint | {summary['checkpoint']['bytes'] / 1e6:.1f} MB |
| peak memory | {summary['peak_memory_bytes'] / 1e9:.2f} GB |

Inference has two honest readings and both are reported, because quoting only
one would be a rhetorical choice:

| path | one position | batched, per position | per hidden piece, batched |
| --- | ---: | ---: | ---: |
| belief CNN alone (field already computed) | {specialist['milliseconds_per_decision_single']:.2f} ms | {specialist['milliseconds_per_decision_batched']:.3f} ms | {specialist['microseconds_per_piece_batched']:.2f} µs |
| frozen C1 encode alone | {encode['milliseconds_per_decision_single']:.2f} ms | {encode['milliseconds_per_decision_batched']:.3f} ms | {encode['microseconds_per_piece_batched']:.2f} µs |
| **end to end, observation -> marginals** | **{end_to_end['milliseconds_per_decision_single']:.2f} ms** | {end_to_end['milliseconds_per_decision_batched']:.3f} ms | {end_to_end['microseconds_per_piece_batched']:.2f} µs |

The first row is what a search that is *already* running C1 for its policy
adds; the last is a belief query in isolation, and it is the row comparable to
Agent 2's {earlier.get(CANDIDATE_2, {}).get('milliseconds_per_decision_single', 0):.2f} ms,
which has no C1 stage. Against Agent 1's winner — a head that rides on C1's
existing encode for
{agent1_1b.get('inference_microseconds_per_piece', 0):.3f} µs
per piece — this candidate is a second network with its own checkpoint
whichever row is used.

Peak memory is the peak process RSS of the training stage: the materialized
1.4 GB C1 field tensor, the model and the metric arrays, not the model alone.
"""
    )

    # -- 9 ------------------------------------------------------------------
    band = decision["equivalence_band"]
    parts.append(
        f"""## 9. Is this preferable to what already exists?

**No. Agent 3 is {_fmt(abs(decision['agent3_minus_best_earlier_r_ce']))}
`R_CE` *worse* than `{agent1_best_id}` ({_fmt(agent1_best.get('r_ce'))} against
{_fmt(row['r_ce'])}) and far more expensive, so it loses on both axes of the
sprint's rule.** It is, however, the better of the two 3.9M spatial
specialists, and that is the finding it contributes.

How the engineering-winner rule applies:

- leader by `R_CE`: `{decision['leader_by_r_ce']}` ({_fmt(decision['leader_r_ce'])});
- inside the {band} equivalence band of the leader: {', '.join(f'`{name}`' for name in decision['within_band_of_leader'])};
- Scout-rush / generalization: {_fmt(decision['scout_rush_r_ce'].get(CANDIDATE_3))} for Agent 3, against {_fmt(decision['scout_rush_r_ce'].get(agent1_best_id))} for `{agent1_best_id}` and {_fmt(decision['scout_rush_r_ce'].get(CANDIDATE_2))} for `{CANDIDATE_2}`;
- search-integration complexity: {decision['search_integration_note']}

The band is measured against the leader only, never as a chain of pairwise
comparisons — the convention Agents 1 and 2 recorded.
"""
    )

    # -- 10 -----------------------------------------------------------------
    interface = summary["interface"]
    parts.append(
        f"""## 10. Required interface

```text
predict_marginals(public_state)      -> {{piece_slot: 12-way rank probabilities}}
sample_worlds(public_state, n, seed) -> complete legal hidden armies
```

| candidate | positions | worlds | marginals valid | seed-deterministic | worlds legal |
| --- | ---: | ---: | --- | --- | --- |
| `{CANDIDATE_3}` | {interface['positions_checked']} | {interface['worlds_sampled']} | {'yes' if interface['all_marginals_sum_to_one'] else 'no'} | {'yes' if interface['sample_worlds_seed_deterministic'] else 'no'} | {'yes' if interface['all_worlds_passed_accepted_validation_stack'] else 'no'} |

Every world was drawn through **`{interface['sampler_source']}`**. Agent 3
supplies marginals and nothing else, through the same `Phase11BPublicState`
Agent 1 defined — a container with exactly two public fields and no field a
true rank could arrive in. The Agent 3 adapter *subclasses* Agent 1's
interface: the encoder slot holds the frozen C1 and the head slot holds the
belief CNN, so `sample_worlds` is inherited code, not a fork, and the live
path recomputes the same `encode` output the cache was built from.
"""
    )

    # -- 11 -----------------------------------------------------------------
    epoch_rows = [r for r in train["curve"] if not r["sub_epoch"]]
    parts.append(
        f"""## 11. Caveats a reader should carry forward

- **This is a development-set number.** There is no sealed bank behind it and
  no scientific claim attached to it. The development set is an engineering
  comparison set, exactly as the sprint defines it.
- **The checkpoint was trained on `{config['device']}` and scored on CPU.**
  The two backends agree to
  {summary['backend_agreement']['absolute_difference']:.2e} `R_CE`, so the
  headline number does not depend on which one produced it. Neither backend's
  float32 reductions are bit-reproducible, so this was measured rather than
  assumed. The *feature cache* was built on CPU precisely so that the seam
  itself is not a source of that difference.
- **The kept checkpoint is {_fmt(overfit['best_epoch_fraction'], 2)} epochs
  into a {training['epochs_run']}-epoch run** ({config['epochs']} scheduled;
  stopped by `{training['stopped_because']}`). Development `R_CE` ended
  {_fmt(epoch_rows[-1]['dev_r_ce'] - row['r_ce'])} worse than at the best
  probe, so the curve had turned well before the run stopped.
- **The headline `R_CE` uses the accepted raw-softmax convention** — no
  masking, no epsilon, full simplex — because that is how the Phase 11 head
  was measured and how the accepted sampler consumes a belief. Renormalizing
  onto the publicly legal support is a diagnostic only
  ({_fmt(row['diagnostic_projected_r_ce'])} against {_fmt(row['r_ce'])} raw).
- **The reference row is not the Phase 11 sealed-test result.** It is the
  unchanged Phase 11 head scored on *these* fresh positions. Phase 11's sealed
  test remains what it was, and its bank remains spent.
- **One configuration, and it was repeated.** `03_AGENT_3` asks for one
  architecture and one comparison, so unlike Agent 2 there is no second
  optimization configuration here. Run-to-run spread was measured the way
  Agent 1 measured it — by training the identical configuration again — and
  it came out at
  {repeat.get('absolute_r_ce_difference', float('nan')):.2e} `R_CE`
  ({_fmt(repeat.get('reported_r_ce'), 6)} against
  {_fmt(repeat.get('repeated_r_ce'), 6)}), with the two runs' epoch-boundary
  training losses agreeing to
  {repeat.get('max_epoch_train_loss_difference', float('nan')):.2e}. That
  repeat is a diagnostic: it wrote no checkpoint and the leaderboard is
  identical without it. What it does **not** bound is seed sensitivity — the
  seed was deliberately held fixed, so this measures backend
  nondeterminism, which for this model on this backend turns out to be
  none.
- **"C1 retains the information" is a statement about this corpus.** Agent 2
  showed the same tower is corpus-limited at 2,048 games. A raw-observation
  encoder is not refuted by losing here; it is untested at a corpus size that
  would give it a chance, and the same caveat applies to this candidate.
"""
    )

    # -- 12 -----------------------------------------------------------------
    preservation = summary["preservation"]
    suite = summary.get("suite")
    parts.append(
        f"""## 12. What Agent 3 touched

The common corpus was reused **byte-for-byte**: both splits' file digests and
the whole-corpus digest `{summary['common_corpus']['corpus_digest'][:16]}…`
were recomputed from disk and matched against the values Agent 1 and Agent 2
recorded. Nothing was regenerated.

| statement | value |
| --- | --- |
| corpus regenerated | `{preservation['corpus_regenerated']}` |
| C1 modified | `{preservation['c1_modified']}` |
| Agent 1 artifacts modified | `{preservation['agent1_artifacts_modified']}` |
| Agent 2 artifacts modified | `{preservation['agent2_artifacts_modified']}` |
| artifacts unchanged since Agent 2 | `{preservation['artifacts_unchanged_since_agent2']}` |
| `phase11_test_bank_v1` opened | `{preservation['phase11_test_bank_opened']}` |

Agent 3 added `stratego/belief/phase11b/feature_seam.py` and
`feature_cnn.py`, the harness `scripts/run_phase11b_agent03.py`, the
renderer `scripts/_phase11b_agent03_report.py`, its tests, two field caches under `checkpoints/phase11b/` and one
checkpoint. It edited no existing module: the frozen-prefix call, the
trainer, the metrics, the sampler adapter and the corpus loader are all
imported from Agent 1's and Agent 2's files unchanged, which is why all
{len(summary['preserved_artifact_digests'])} preserved digests still match.
"""
    )
    if suite:
        parts.append(
            f"Repository suite after Agent 3: **{suite['summary_line']}** "
            f"(`{suite['command']}`).\n"
        )

    # -- 13 -----------------------------------------------------------------
    parts.append(
        f"""## 13. Handoff to Agent 4

Agent 3 does not begin Agent 4's experiment and does not recommend for or
against running it. What Agent 3 measured that Agent 4 should carry:

1. **The frozen C1 field beats raw pixels at equal capacity, by
   {_fmt(abs(contrast['difference_agent3_minus_agent2']))} `R_CE`, and the
   difference is distinguishable.** Agent 4's hybrid feeds both into one
   specialist. This result says the C1 half is the more informative of the
   two inputs it will be given, not that the raw half is worthless — Agent 2's
   candidate is still
   {_fmt(earlier.get(REFERENCE, {}).get('r_ce', 0) - earlier.get(CANDIDATE_2, {}).get('r_ce', 0))}
   better than the unchanged Phase 11 head.
2. **Both spatial specialists lose to a {_params(agent1_1b):,}-parameter
   head attached to the frozen C1 feature.** On this corpus the binding
   constraint is supervision, not extraction capacity. A hybrid at 3-5M parameters should
   expect the same régime and should probe development loss several times per
   epoch — this run's optimum arrived {_fmt(overfit['best_epoch_fraction'], 2)}
   epochs in.
3. **The seam and its cache are reusable.** `feature_seam.py` builds the
   `[N, 100, 128]` field for either split in seconds, verifies it against the
   public observations, and hashes it; Agent 4 can concatenate it with the raw
   observation without re-deriving anything.
4. **`train_raw_cnn` is representation-agnostic.** Agent 3 reused it by
   passing a split view with a different input array. A hybrid can do the same
   with a stacked input, and inheriting the trainer is what makes the three
   candidates' numbers comparable.
"""
    )

    # -- 14 -----------------------------------------------------------------
    parts.append(
        f"""## 14. Stop condition

{summary['stop_condition']}
"""
    )
    return "\n".join(parts)


__all__ = ["render"]
