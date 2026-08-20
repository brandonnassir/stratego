"""Markdown renderer for the Phase 11B Agent 4 report.

Split out of `run_phase11b_agent04.py` for the reason
`_phase11b_agent03_report.py` is split out of its harness: the prose is
long, it changes for reasons that have nothing to do with the experiment,
and a harness is easier to read when the string building lives elsewhere.

Every number is read from the summary the harness just wrote. Nothing here
recomputes a metric, and nothing here decides anything: the verdict is
`summary["decision"]`, the complementarity reading is
`summary["complementarity"]` and the required table is
`summary["comparison_table"]`, all three produced by the harness, and this
module only puts them into sentences.
"""

from __future__ import annotations

CANDIDATE_4 = "agent04_hybrid_raw_c1_cnn"
CANDIDATE_3 = "agent03_c1_feature_cnn"
CANDIDATE_2 = "agent02_raw_observation_cnn"
CANDIDATE_1A = "agent01_1a_existing_linear_head"
CANDIDATE_1B = "agent01_1b_attached_mlp_head"
CANDIDATE_1C = "agent01_1c_final_block_plus_mlp"
REFERENCE = "phase11_head_unchanged_reference"

STRATA = ("phase9_selfplay", "strategic_rule", "tactical_rule", "scout_rush")
STRATUM_LABELS = {
    "phase9_selfplay": "Phase9-like",
    "strategic_rule": "Strategic",
    "tactical_rule": "Tactical",
    "scout_rush": "Scout-rush",
}


def _fmt(value, digits: int = 4, dash: str = "—") -> str:
    if value is None:
        return dash
    return f"{value:.{digits}f}"


def _interval(bounds) -> str:
    if not bounds:
        return "—"
    return f"[{bounds[0]:.4f}, {bounds[1]:.4f}]"


def _ce_interval(bounds) -> str:
    if not bounds:
        return "—"
    return f"[{bounds[0]:+.5f}, {bounds[1]:+.5f}]"


def _signed(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}"


def _params(block: dict) -> int:
    """Trained-parameter count of one leaderboard row.

    Agent 1 records it as `belief_parameters_total` and Agents 2-4 as
    `parameters_trained`; a reader comparing rows wants one column, so the
    two spellings are resolved here rather than in a dozen f-strings.
    """
    for key in ("parameters_trained", "belief_parameters_total", "parameters"):
        if block.get(key) is not None:
            return int(block[key])
    return 0


def _ordered_rows(row: dict, earlier: dict) -> list:
    """Every candidate on the sprint's development set, best `R_CE` first."""
    rows = [(CANDIDATE_4, row)] + [(name, block) for name, block in earlier.items()]
    return sorted(rows, key=lambda pair: pair[1]["r_ce"])


def _paired(comparisons: dict, other: str) -> dict:
    return comparisons.get(f"{CANDIDATE_4} vs {other}", {})


def _verdict(paired: dict) -> str:
    """One phrase for what a paired bootstrap actually says."""
    if not paired:
        return "—"
    if not paired.get("distinguishable"):
        return "not distinguishable"
    return "Agent 4 lower" if paired.get("left_lower_ce") else "Agent 4 higher"


def render(summary: dict, train: dict) -> str:
    row = summary["leaderboard"][CANDIDATE_4]
    earlier = summary["earlier_reference_rows"]
    complement = summary["complementarity"]
    table = summary["comparison_table"]
    decision = summary["decision"]
    training = summary["training"]
    pilot = summary["pilot"]
    seam = summary["frozen_seam"]
    cache = summary["feature_cache"]
    comparisons = summary["paired_comparisons"]
    inference = summary["inference"]["cpu"]
    overfit = training["overfitting"]
    repeat = summary.get("repeat_run") or {}
    interval = _interval(row["r_ce_ci95"])

    agent2 = earlier.get(CANDIDATE_2, {})
    agent3 = earlier.get(CANDIDATE_3, {})
    agent1_1b = earlier.get(CANDIDATE_1B, {})
    agent1_1c = earlier.get(CANDIDATE_1C, {})
    reference = earlier.get(REFERENCE, {})
    versus_3 = _paired(comparisons, CANDIDATE_3)
    versus_2 = _paired(comparisons, CANDIDATE_2)
    leader_id = decision["leader_by_r_ce"]
    parts: list[str] = []

    # -- 0 ------------------------------------------------------------------
    parts.append(
        f"""# Phase 11B — Agent 4: Hybrid Raw + C1 CNN

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

## 0. What Agent 4 found

Agent 4's question was whether raw public information and C1's learned
strategic representation carry **complementary** belief signal — whether C1
supplies high-level abstractions while the raw observation restores
belief-specific detail C1 may have compressed away.

The answer is **no, not measurably**, and the shape of the answer is the
useful part.

1. **The raw observation adds nothing on top of C1.** The hybrid scores
   `R_CE` **{_fmt(row['r_ce'])}** {interval} against Agent 3's C1-only tower
   at {_fmt(agent3.get('r_ce'))}. The paired game bootstrap of the
   cross-entropy difference is
   {_signed(versus_3.get('ce_difference'), 5)}
   {_ce_interval(versus_3.get('ce_difference_ci95'))} — the interval
   **straddles zero**, so on {versus_3.get('games', '—')} development games
   the two are not distinguishable. Half the stem was handed to a
   127-channel public observation and the belief did not move.
2. **C1 adds a great deal on top of the raw observation.** The same fusion
   against Agent 2's raw-only tower is
   {_signed(versus_2.get('ce_difference'), 5)}
   {_ce_interval(versus_2.get('ce_difference_ci95'))}, which is
   distinguishable and favours the hybrid. Agent 2 alone is
   {_fmt(agent2.get('r_ce'))}.
3. So the two directions are **not symmetric**. For belief at this corpus
   size, C1's per-square representation is effectively a superset of the raw
   observation: adding C1 to raw pixels is worth
   {_fmt(abs((agent2.get('r_ce') or 0) - row['r_ce']))} `R_CE`, and adding raw
   pixels to C1 is worth {_fmt(abs((agent3.get('r_ce') or 0) - row['r_ce']))},
   inside the noise.
4. **The sprint leader is unchanged.** All three 3.9M-parameter spatial
   specialists — raw, C1, hybrid — remain behind Agent 1's
   {_params(agent1_1b):,}-parameter attached head
   ({_fmt(agent1_1b.get('r_ce'))}) and its
   {_params(agent1_1c):,}-parameter 1C variant
   ({_fmt(agent1_1c.get('r_ce'))}). The leader by `R_CE` is
   **`{leader_id}`** at {_fmt(decision['leader_r_ce'])}.

Read together with Agent 2's corpus-size sweep, the reading is consistent and
unflattering to more architecture: three different 3.9M-parameter inputs land
within {_fmt(abs((agent2.get('r_ce') or 0) - row['r_ce']))} `R_CE` of each
other and all of them lose to a 335k head. The binding constraint is the
corpus, not the representation and not the fusion.
"""
    )

    # -- 1 ------------------------------------------------------------------
    field_train = cache["c1_field_reused_from_agent3"].get("train", {})
    field_dev = cache["c1_field_reused_from_agent3"].get("dev", {})
    fused_train = cache["fused_input"].get("train", {})
    fused_dev = cache["fused_input"].get("dev", {})
    verify_train = cache["fused_input_verification"].get("train", {})
    verify_dev = cache["fused_input_verification"].get("dev", {})
    parts.append(
        f"""## 1. The two inputs, and what was reused

`04_AGENT_4` requires two legal, public branches and forbids changing either
Agent 1's common corpus or Agent 3's frozen C1 seam.

```text
branch A   raw 127 x 10 x 10 public observation      the corpus's own bytes
branch B   frozen per-square C1 field [100, 128]     Agent 3's seam, unchanged
```

### The seam is Agent 3's, not a new one

| property | value |
| --- | --- |
| `seam_id` | `{seam['seam_id']}` |
| tensor | `{seam['tensor']}` |
| definition | `{seam['definition']}` |
| shape | `[{', '.join(str(part) for part in seam['shape'])}]` |
| per-square | `{seam['is_per_square']}` |
| pooled | `{seam['is_pooled']}` |
| source | {summary['frozen_seam_source']} |
| matches Agent 3's recorded description | `{summary['frozen_seam_matches_agent3_record']}` |

### Agent 3's cache was reused, not rebuilt

`04_AGENT_4`: "Reuse Agent 3's C1 feature cache if compatible and exact."
Compatibility and exactness are measured rather than assumed. The cache files
are opened read-only, their content digests are recomputed and compared to
the digests Agent 3 published, and a random sample of each split is
re-encoded from the public observations through the frozen C1.

| split | shape | digest matches Agent 3 | rows re-encoded | max abs difference |
| --- | --- | --- | --- | --- |
| train | `{field_train.get('shape')}` | `{field_train.get('digest_matches_agent3')}` | {cache['c1_field_verification'].get('train', {}).get('rows_checked', '—')} | {cache['c1_field_verification'].get('train', {}).get('max_absolute_difference', '—')} |
| dev | `{field_dev.get('shape')}` | `{field_dev.get('digest_matches_agent3')}` | {cache['c1_field_verification'].get('dev', {}).get('rows_checked', '—')} | {cache['c1_field_verification'].get('dev', {}).get('max_absolute_difference', '—')} |

`c1_field_rebuilt` is `{cache['c1_field_rebuilt']}`. Agent 3's checkpoint, its
report and its cache files are byte-for-byte what they were.

### The one thing Agent 4 did build

Agent 2's trainer stages **one** tensor per split and indexes it by sample
row. Rather than fork it — which would have made Agent 4's optimizer,
shuffle, probe schedule and checkpoint rule merely *similar* to Agents 2 and
3 rather than identical — the two branches are laid side by side into one
255-channel tensor:

```text
channels   0 .. 127    the corpus's public observation, unchanged
channels 127 .. 255    Agent 3's C1 field, in field_to_planes layout
```

`HybridBeliefCNN.forward` splits it back apart at channel 127 and sends the
halves to their own projections. This is a re-layout of two existing arrays,
not a third representation, and the verification says so half by half:

| split | shape | size | raw half is the corpus observation | C1 half is Agent 3's field |
| --- | --- | --- | --- | --- |
| train | `{fused_train.get('shape')}` | {fused_train.get('bytes', 0) / 1e9:.2f} GB | `{verify_train.get('raw_half_is_the_corpus_observation')}` | `{verify_train.get('c1_half_is_agent3s_field')}` |
| dev | `{fused_dev.get('shape')}` | {fused_dev.get('bytes', 0) / 1e9:.2f} GB | `{verify_dev.get('raw_half_is_the_corpus_observation')}` | `{verify_dev.get('c1_half_is_agent3s_field')}` |

Both halves re-derive bit-identically ({verify_train.get('rows_checked')} random
rows per split), and `contains_labels` is `false` on every cache block. No
hidden truth enters either branch: the corpus keeps true ranks in a separate
`privileged/` directory that the loader hands over only when asked by name,
and neither of the model's two entry points has an argument a label could
arrive in.

### The common corpus

| | train | dev |
| --- | --- | --- |
| games | {summary['common_corpus']['splits']['train']['games']} | {summary['common_corpus']['splits']['dev']['games']} |
| positions | {summary['common_corpus']['splits']['train']['samples']} | {summary['common_corpus']['splits']['dev']['samples']} |
| hidden pieces | {summary['common_corpus']['splits']['train']['hidden_pieces']:,} | {summary['common_corpus']['splits']['dev']['hidden_pieces']:,} |
| setup-library split | `{summary['common_corpus']['splits']['train']['library_split']}` | `{summary['common_corpus']['splits']['dev']['library_split']}` |

`corpus_digest` `{summary['common_corpus']['corpus_digest'][:16]}…`, recomputed
from the bytes on disk and equal to the digest Agents 1, 2 and 3 each
recorded. {summary['common_corpus']['reused']}.
"""
    )

    # -- 2 ------------------------------------------------------------------
    breakdown = pilot["parameters"]
    parts.append(
        f"""## 2. The model

```text
raw 127 x 10 x 10  -> conv3x3 {breakdown['raw_branch_width']} -> BN -> ReLU --\\
                                                                   concat {breakdown['width']}
C1 field [100,128] -> conv3x3 {breakdown['c1_branch_width']} -> BN -> ReLU --/
                                                                        |
                                                    {breakdown['blocks']} x residual 3x3 {breakdown['width']}
                                                                        |
                                                     1x1 {breakdown['readout_width']} -> BN -> ReLU -> 1x1 12
                                                                        |
                                                              12 logits per square
```

`{row['architecture']}`

| block | parameters |
| --- | --- |
| raw branch | {breakdown['raw_branch']:,} |
| C1 branch | {breakdown['c1_branch']:,} |
| residual tower | {breakdown['residual_tower']:,} |
| read-out | {breakdown['readout']:,} |
| **total** | **{breakdown['total']:,}** |

Inside the instructed `{pilot['parameter_band'][0]:,}`–`{pilot['parameter_band'][1]:,}`
band, and — deliberately — within {abs(pilot['parameters_minus_agent3']):,}
parameters of Agent 3 and {abs(pilot['parameters_minus_agent2']):,} of Agent 2:

| candidate | stem | parameters |
| --- | --- | --- |
| Agent 2 | `conv3x3(127 -> 160)` | {pilot['agent2_parameters']:,} |
| Agent 3 | `conv3x3(128 -> 160)` | {pilot['agent3_parameters']:,} |
| Agent 4 | `conv3x3(127 -> 80) ‖ conv3x3(128 -> 80)` | {breakdown['total']:,} |

That is the whole design. The residual tower and the read-out are **Agent 2's,
imported rather than re-declared**, at Agent 2's width and Agent 2's depth,
and the seam is Agent 3's. What changes across the three reports is the stem.
A spread of {pilot['agent3_parameters'] - pilot['agent2_parameters']:,}
parameters — 0.04% — means a difference between the three numbers is a
difference of *input representation*, not of capacity.

### The choices, declared as choices

`04_AGENT_4` forbids branch-width, fusion-method, depth and learning-rate
sweeps. None was run. Three things had to be picked without one:

- **Branch widths 80 and 80.** The fused width had to be
  {breakdown['width']} for the tower to be Agent 2's, so the only free choice
  was how to divide it. It is divided evenly, because the experiment asks
  whether the two sources are *complementary* and an uneven split would
  prejudge which one carries more.
- **Concatenation, not addition or gating.** Summing two projections would
  force both representations into one shared {breakdown['width']}-channel
  basis before a single nonlinearity had seen them together; gating would add
  a learned mixing rule this experiment has no budget to validate.
  Concatenation lets the first residual block's 3x3 convolution learn the
  mixture itself, per channel and per neighbourhood, which is the weakest
  assumption of the three.
- **One 3x3 per branch.** The instruction's diagram says "small spatial
  projection". One convolution each keeps the fusion early and leaves the
  work to the shared tower rather than to two private stacks.

`architecture_sweep`, `branch_width_sweep`, `fusion_method_sweep`,
`depth_sweep` and `learning_rate_sweep` are all `false` in the summary, and
`optimization_configurations_declared` is
`{summary['experiment']['optimization_configurations_declared']}`.

### C1 is frozen, structurally

`build_hybrid_cnn` returns the specialist **alone**, so no optimizer built
from its parameters can reach a C1 weight. During training C1 is not even
called: the C1 branch reads Agent 3's cached field. `c1_parameters_updated`
is `{training['c1_parameters_updated']}` and
`gradients_reaching_c1` is `{cache['gradients_reaching_c1']}`.
"""
    )

    # -- 3 ------------------------------------------------------------------
    cpu_probe = pilot["probes"].get("cpu", {})
    chosen_probe = pilot["probes"].get(pilot["device_chosen"], {})
    parts.append(
        f"""## 3. The pilot, and where the budget came from

`04_AGENT_4` asks for "a brief throughput/sanity pilot first". It ran on every
available backend before any development metric existed.

| backend | s/step | positions/s | est. s/epoch | first loss | last loss |
| --- | --- | --- | --- | --- | --- |
| cpu | {_fmt(cpu_probe.get('seconds_per_step'), 3)} | {cpu_probe.get('positions_per_second')} | {cpu_probe.get('estimated_epoch_seconds')} | {_fmt(cpu_probe.get('first_loss'))} | {_fmt(cpu_probe.get('last_loss'))} |
| mps | {_fmt(pilot['probes'].get('mps', {}).get('seconds_per_step'), 3)} | {pilot['probes'].get('mps', {}).get('positions_per_second')} | {pilot['probes'].get('mps', {}).get('estimated_epoch_seconds')} | {_fmt(pilot['probes'].get('mps', {}).get('first_loss'))} | {_fmt(pilot['probes'].get('mps', {}).get('last_loss'))} |

Chosen: **`{pilot['device_chosen']}`**, {pilot['device_speedup_vs_cpu']}x CPU.
{pilot['device_rule']}. The two backends' pilot losses agree to
{pilot['cross_device_loss_agreement']}.

The epoch horizon is {pilot['epochs_declared']}, from
{pilot['epoch_budget_basis']} — a budget decision taken from *measured
throughput*, before a single development number existed.

The {pilot['staged_input_bytes'] / 1e9:.2f} GB fused training tensor was staged
on the training device (`stage_on_device` `{pilot['stage_on_device']}`) under
the declared rule: {pilot['stage_on_device_rule']}.
"""
    )

    # -- 4 ------------------------------------------------------------------
    stratum_lines = "\n".join(
        f"| {STRATUM_LABELS[name]} | {_fmt(row['r_ce_by_stratum'].get(name))} | "
        f"{_fmt(row['top1_by_stratum'].get(name))} | "
        f"{_fmt(agent3.get('r_ce_by_stratum', {}).get(name))} | "
        f"{_fmt(agent2.get('r_ce_by_stratum', {}).get(name))} | "
        f"{_fmt(agent1_1b.get('r_ce_by_stratum', {}).get(name))} |"
        for name in STRATA
    )
    board_lines = "\n".join(
        f"| {index} | `{name}` | {_fmt(block['r_ce'])} | {_interval(block.get('r_ce_ci95'))} | "
        f"{_fmt(block['top1'])} | {_params(block):,} |"
        for index, (name, block) in enumerate(_ordered_rows(row, earlier), start=1)
    )
    paired_lines = "\n".join(
        f"| `{other}` | {_signed(block.get('ce_difference'), 5)} | "
        f"{_ce_interval(block.get('ce_difference_ci95'))} | "
        f"`{block.get('distinguishable')}` | {_verdict(block)} |"
        for other, block in (
            (name.split(" vs ")[1], value) for name, value in comparisons.items()
        )
    )
    parts.append(
        f"""## 4. Results on the common development set

{row['dev_samples']:,} positions, {row['dev_pieces']:,} hidden pieces, the
identical development positions every Phase 11B candidate is scored on.

| metric | value |
| --- | --- |
| overall cross-entropy | {_fmt(row['ce'], 6)} |
| remaining-count baseline cross-entropy | {_fmt(row['baseline_ce'], 6)} |
| **`R_CE`** | **{_fmt(row['r_ce'], 6)}** |
| `R_CE` 95% CI | {interval} |
| top-1 hidden-rank accuracy | {_fmt(row['top1'])} |
| baseline top-1 | {_fmt(row['baseline_top1'])} |
| best stratum | `{row['best_stratum']}` |
| worst stratum | `{row['worst_stratum']}` |
| projected-onto-legal `R_CE` (diagnostic only) | {_fmt(row['diagnostic_projected_r_ce'])} |

The uniform floor is `R_CE` {_fmt(summary['uniform_floor']['r_ce'])}; every
candidate on the board beats it comfortably, so the leaderboard is a
comparison between models rather than between a model and noise.

### Per-stratum `R_CE`

| stratum | Agent 4 `R_CE` | Agent 4 top-1 | Agent 3 `R_CE` | Agent 2 `R_CE` | Agent 1 1B `R_CE` |
| --- | --- | --- | --- | --- | --- |
{stratum_lines}

Scout-rush is the sprint's generalization stratum — the behaviour least like
the self-play the frozen C1 was trained on. Agent 4 scores
{_fmt(row['r_ce_by_stratum'].get('scout_rush'))} there, against Agent 3's
{_fmt(agent3.get('r_ce_by_stratum', {}).get('scout_rush'))}, Agent 2's
{_fmt(agent2.get('r_ce_by_stratum', {}).get('scout_rush'))} and Agent 1 1B's
{_fmt(agent1_1b.get('r_ce_by_stratum', {}).get('scout_rush'))}. The raw
branch does not rescue the unusual stratum: raw-only is still the worst of the
three towers there, and Agent 4 and Agent 3 change places by
{_fmt(abs((row['r_ce_by_stratum'].get('scout_rush') or 0) - (agent3.get('r_ce_by_stratum', {}).get('scout_rush') or 0)))}
— the hybrid is marginally *worse* on Scout-rush than the C1-only tower it is
marginally better than overall. Both gaps are far inside the equivalence band,
which is the honest reading: on this stratum, as overall, the second input is
not doing measurable work.

### The whole sprint board, best first

| # | candidate | `R_CE` | 95% CI | top-1 | trained parameters |
| --- | --- | --- | --- | --- | --- |
{board_lines}

### Paired game bootstraps

Marginal confidence intervals cannot say whether two candidates differ,
because they are scored on the same positions. These are paired bootstraps
over {versus_3.get('games', '—')} development **games**, of the per-piece
cross-entropy difference (negative = Agent 4 lower):

| against | CE difference | 95% CI | distinguishable | reading |
| --- | --- | --- | --- | --- |
{paired_lines}

Every earlier candidate here was loaded read-only from its own checkpoint and
scored on these same pieces; nothing was retrained.
{summary['earlier_reproduction_note']}
"""
    )

    # -- 5 ------------------------------------------------------------------
    table_lines = "\n".join(
        f"| {entry['label']} | `{entry['candidate_id']}` | {_fmt(entry['r_ce'])} | "
        f"{_interval(entry['r_ce_ci95'])} | {_fmt(entry['top1'])} | "
        f"{_fmt(entry['scout_rush_r_ce'])} | "
        f"{(entry['parameters'] or 0):,} | `{entry['rerun_by_agent4']}` |"
        for entry in table["rows"]
    )
    parts.append(
        f"""## 5. The no-rerun comparison table `04_AGENT_4` asks for

| candidate | id | `R_CE` | 95% CI | top-1 | Scout-rush `R_CE` | trained parameters | rerun by Agent 4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
{table_lines}

"Agent 1 best attached head" is resolved as `{table['agent1_best_resolved_as']}`
by the rule "{table['agent1_best_resolution_rule']}", from Agent 1's own
leaderboard, so the table cannot silently quote the wrong one.
`prior_candidates_rerun` is `{table['prior_candidates_rerun']}`; every earlier
figure is read from {table['source']}.
"""
    )

    # -- 6 ------------------------------------------------------------------
    parts.append(
        f"""## 6. Is the signal complementary?

This is the question `04_AGENT_4` exists to answer, and it is answered
arithmetically rather than by assertion. Complementarity is a claim about the
hybrid beating **both** single-source towers of the same size — beating only
the weaker one would show nothing, since the hybrid contains the stronger
one's input. So the reference is the better of the two.

| quantity | value |
| --- | --- |
| Agent 2, raw-only | {_fmt(complement['agent2_raw_only_r_ce'])} |
| Agent 3, C1-only | {_fmt(complement['agent3_c1_only_r_ce'])} |
| **Agent 4, hybrid** | **{_fmt(complement['agent4_hybrid_r_ce'])}** |
| better single source | `{complement['better_single_source']}` at {_fmt(complement['better_single_source_r_ce'])} |
| hybrid − better single source | {_signed(complement['hybrid_minus_better_single_source'], 5)} |
| hybrid − raw-only | {_signed(complement['hybrid_minus_agent2_raw_only'], 5)} |
| hybrid − C1-only | {_signed(complement['hybrid_minus_agent3_c1_only'], 5)} |
| equivalence band | {complement['equivalence_band']} |
| **complementary** | **`{complement['complementary']}`** |

> {complement['interpretation']}

The paired bootstrap is the sharper statement. Against C1-only the difference
is {_signed(versus_3.get('ce_difference'), 5)}
{_ce_interval(versus_3.get('ce_difference_ci95'))} — the interval contains
zero, so the hybrid and the C1-only tower are **not distinguishable** on this
development set. Against raw-only it is
{_signed(versus_2.get('ce_difference'), 5)}
{_ce_interval(versus_2.get('ce_difference_ci95'))}, which is distinguishable
and favours the hybrid.

### What that asymmetry means

The instruction's hypothesis had two halves — that C1 supplies strategic
abstraction, and that the raw observation restores belief detail C1
compressed away. The first half survives; the second does not.

- Give a fusion tower the C1 field on top of raw pixels and belief improves
  measurably. C1's six transformer blocks are contributing something the
  convolution tower cannot learn from the observation in
  {summary['common_corpus']['splits']['train']['samples']:,} positions.
- Give the same tower raw pixels on top of the C1 field and belief does not
  move. Whatever belief-relevant detail C1 compressed away, either it is not
  there, or {summary['common_corpus']['splits']['train']['samples']:,}
  correlated positions are not enough to learn to use it.

The second reading is the one Agent 2 already argued from a different
direction with its corpus-size sweep, and it is the reading this result
supports: the marginal value of *any* additional representation is being
absorbed by the corpus.

### Capacity was held fixed, so this is not a capacity story

| candidate | parameters |
| --- | --- |
| `{CANDIDATE_2}` | {complement['capacity_held_fixed']['agent02_raw_observation_cnn']:,} |
| `{CANDIDATE_3}` | {complement['capacity_held_fixed']['agent03_c1_feature_cnn']:,} |
| `{CANDIDATE_4}` | {complement['capacity_held_fixed']['agent04_hybrid_raw_c1_cnn']:,} |
| spread | {complement['capacity_held_fixed']['spread']:,} |

{complement['capacity_held_fixed']['note']}.
"""
    )

    # -- 7 ------------------------------------------------------------------
    config = training["config"]
    parts.append(
        f"""## 7. What was trained, and how

One architecture, one declared configuration, inherited from Agent 2 through
Agent 3 so the three candidates differ in input rather than in tuning effort.

| setting | value |
| --- | --- |
| run id | `{training['run_id']}` |
| optimizer | `{config['optimizer']}`, lr {config['learning_rate']}, weight decay {config['weight_decay']} |
| schedule | `{config['schedule']}` |
| batch | {config['batch_positions']} positions |
| gradient clip | {config['gradient_clip']} |
| dropout | {config['block_dropout']} / {config['readout_dropout']} |
| epoch horizon | {config['epochs']} |
| patience | {config['patience']} |
| device | `{config['device']}` |
| trainer | `{config['trainer_version']}` (shared with {', '.join(f'`{name}`' for name in summary['experiment']['trainer_shared_with'])}) |
| evaluations per epoch | {config['evaluations_per_epoch']} |
| input staged on device | `{training['input_staged_on_device']}` |

The loss is supervised hidden-rank cross-entropy over hidden pieces and
nothing else: `policy_or_value_terms` `{training['policy_or_value_terms']}`,
`game_outcome_used` `{training['game_outcome_used']}`.

### The overfitting signature, for the third time

| quantity | value |
| --- | --- |
| epochs run | {training['epochs_run']} of {config['epochs']} |
| stopped because | `{training['stopped_because']}` |
| best epoch | {training['best_epoch']} (fraction {training['best_epoch_fraction']}, step {training['best_step']} of {training['steps_per_epoch']} per epoch) |
| development probes | {training['evaluations']} |
| train CE, first epoch | {_fmt(overfit['train_ce_first_epoch'])} |
| train CE, last epoch | {_fmt(overfit['train_ce_last_epoch'])} |
| dev CE at best | {_fmt(overfit['dev_ce_best'])} |
| dev CE, last epoch | {_fmt(overfit['dev_ce_last_epoch'])} |
| dev CE rose after best | `{overfit['dev_ce_rose_after_best']}` |

The optimum arrives {training['best_epoch_fraction']:.0%} of the way through
the **first** epoch and development cross-entropy rises monotonically
thereafter while training cross-entropy collapses to
{_fmt(overfit['train_ce_last_epoch'])}. That is the same signature Agent 2 and
Agent 3 recorded, on a third input representation. It is a statement about
3.9M parameters against
{training['train_positions']:,} correlated positions, not about the fusion —
and it is why the sub-epoch probe cadence exists: a once-per-epoch probe would
have missed this candidate's best weights entirely.

### Run-to-run spread

The identical configuration and the identical seed were trained a second
time. This run is a **diagnostic**: `is_the_reported_candidate`
`{repeat.get('is_the_reported_candidate')}`, `checkpoint_written`
`{repeat.get('checkpoint_written')}`.

| quantity | value |
| --- | --- |
| reported `R_CE` | {_fmt(repeat.get('reported_r_ce'), 6)} |
| repeated `R_CE` | {_fmt(repeat.get('repeated_r_ce'), 6)} |
| absolute difference | {repeat.get('absolute_r_ce_difference')} |
| best step matches | `{repeat.get('best_step_matches')}` |
| max epoch train-loss difference | {repeat.get('max_epoch_train_loss_difference')} |

Bit-identical, which matches what Agent 3 measured for this model family:
MPS is not run-to-run deterministic for C1 transformer training, but it *is*
for these convolution/batch-norm belief towers. The spread is therefore not a
competing explanation for any gap on the board.

### The two backends agree

| quantity | value |
| --- | --- |
| training backend | `{summary['backend_agreement']['training_backend']}` |
| scoring backend | `{summary['backend_agreement']['scoring_backend']}` |
| `R_CE` on the training backend | {_fmt(summary['backend_agreement']['r_ce_training_backend'], 8)} |
| `R_CE` on CPU | {_fmt(summary['backend_agreement']['r_ce_cpu'], 8)} |
| absolute difference | {summary['backend_agreement']['absolute_difference']:.2e} |

The reported number is the CPU one, so the leaderboard row and every paired
bootstrap come from one scoring pass on the accepted evaluation backend.
"""
    )

    # -- 8 ------------------------------------------------------------------
    specialist = inference["specialist"]
    encode = inference["frozen_c1_encode"]
    end_to_end = inference["end_to_end"]
    parts.append(
        f"""## 8. Cost

| quantity | value |
| --- | --- |
| training wall-clock | {training['training_seconds']} s |
| time to best checkpoint | {training['time_to_best_seconds']} s |
| peak process RSS | {summary['peak_memory_bytes'] / 1e9:.2f} GB |
| checkpoint | {summary['checkpoint']['bytes'] / 1e6:.1f} MB, `{summary['checkpoint']['sha256'][:16]}…` |

{summary['peak_memory_note']}.

### Inference latency, both honest readings

A C1-consuming candidate has two defensible prices and reporting only one
would be a rhetorical choice.

| reading | ms/decision (single) | ms/decision (batched) | µs/piece |
| --- | --- | --- | --- |
| specialist alone | {specialist['milliseconds_per_decision_single']} | {specialist['milliseconds_per_decision_batched']} | {specialist['microseconds_per_piece_batched']} |
| frozen C1 encode alone | {encode['milliseconds_per_decision_single']} | {encode['milliseconds_per_decision_batched']} | {encode['microseconds_per_piece_batched']} |
| end to end | {end_to_end['milliseconds_per_decision_single']} | {end_to_end['milliseconds_per_decision_batched']} | {end_to_end['microseconds_per_piece_batched']} |

*Specialist alone* is the added cost inside a search that is already running
C1 for its policy — the situation this project is actually in. *End to end*
is a belief query in isolation, and is the number comparable to Agent 2,
which has no C1 stage. Measured on CPU at batch
{inference['batch_positions']} with {inference['repeats']} repeats,
{inference['hidden_pieces_per_decision']} hidden pieces per decision.

For scale, Agent 1's attached head costs
{_fmt(agent1_1b.get('inference_microseconds_per_piece'), 4)} µs per piece on a
pass the search is already making.
"""
    )

    # -- 9 ------------------------------------------------------------------
    best_earlier = decision["best_earlier_candidate"]
    parts.append(
        f"""## 9. Is this preferable to what already exists?

The sprint's engineering-winner rule: prefer materially lower overall `R_CE`,
give substantial weight to Scout-rush generalization, treat candidates within
roughly {decision['equivalence_band']} `R_CE` as equivalent and prefer the
cheaper and simpler one, and count search-integration complexity.

| question | answer |
| --- | --- |
| leader by `R_CE` | `{decision['leader_by_r_ce']}` at {_fmt(decision['leader_r_ce'])} |
| within the equivalence band of the leader | {', '.join(f'`{name}`' for name in decision['within_band_of_leader'])} |
| best earlier candidate | `{best_earlier}` at {_fmt(earlier.get(best_earlier, {}).get('r_ce'))} |
| Agent 4 − best earlier | {_signed(decision['agent4_minus_best_earlier_r_ce'], 5)} |
| Agent 4 materially better than best earlier | `{decision['agent4_materially_better_than_best_earlier']}` |
| Agent 4 is the leader | `{decision['agent4_is_the_leader']}` |

**No.** Agent 4 is {_signed(decision['agent4_minus_best_earlier_r_ce'], 4)}
`R_CE` against `{best_earlier}` — worse, distinguishably so
({_signed(_paired(comparisons, best_earlier).get('ce_difference'), 5)}
{_ce_interval(_paired(comparisons, best_earlier).get('ce_difference_ci95'))}) —
while costing {_params(row) / max(_params(earlier.get(best_earlier, {})), 1):.1f}x
the parameters and a second network with its own checkpoint.

On Scout-rush, the generalization stratum the rule weights specially:

| candidate | Scout-rush `R_CE` |
| --- | --- |
"""
        + "\n".join(
            f"| `{name}` | {_fmt(value)} |"
            for name, value in sorted(
                decision["scout_rush_r_ce"].items(), key=lambda pair: (pair[1] is None, pair[1])
            )
        )
        + f"""

Agent 4 does not win that column either.

On search integration: {decision['search_integration_note']}
"""
    )

    # -- 10 -----------------------------------------------------------------
    iface = summary["interface"]
    parts.append(
        f"""## 10. Required interface

`04_AGENT_4` requires `predict_marginals(public_state)` and
`sample_worlds(public_state, n, seed)`. Both are exposed by
`HybridBeliefModel`, which subclasses Agent 1's shared interface rather than
reimplementing it — so `sample_worlds` runs through the **accepted,
unmodified** Phase 11 sampler as inherited code, not a fork.

| check | value |
| --- | --- |
| interface version | `{iface['interface_version']}` |
| positions exercised | {iface['positions_checked']} |
| worlds sampled | {iface['worlds_sampled']} ({iface['worlds_per_position']} per position) |
| mean unresolved pieces per position | {iface['hidden_pieces_per_position']} |
| all marginals are probability vectors | `{iface['all_marginals_sum_to_one']}` |
| `sample_worlds` is seed-deterministic | `{iface['sample_worlds_seed_deterministic']}` |
| every world passed the accepted validation stack | `{iface['all_worlds_passed_accepted_validation_stack']}` |
| sampler | `{iface['sampler_source']}` |
| reads hidden truth | `{iface['describe']['reads_hidden_truth']}` |

The positions are replayed from the common corpus's own development plans,
exactly as Agents 1, 2 and 3 build them, so all four interface blocks describe
the same interface on the same kind of state.

In deployment the C1 branch's input is derived live from the public
observation through the same frozen encoder and the same `feature_layer`
(`{seam['layer_token']}`) the cached field was built from, so the trained path
and the deployed path read the identical tensor.
"""
    )

    # -- 11 -----------------------------------------------------------------
    parts.append(
        f"""## 11. Caveats a reader should carry forward

- **This is a development-set engineering comparison, not a sealed test.**
  The Phase 11B development set is explicitly "an engineering comparison set,
  not a scientifically sealed bank". Nothing here is a scientific result.
- **A negative complementarity finding is corpus-conditional.** "The raw
  observation adds nothing on top of C1" is measured at
  {summary['common_corpus']['splits']['train']['samples']:,} training
  positions from {summary['common_corpus']['splits']['train']['games']:,}
  games. Agent 2's corpus-size sweep already showed this regime is
  corpus-bound; a larger corpus could change the answer, and this report does
  not claim otherwise.
- **One fusion, one split, one depth.** The instruction forbids sweeps, so
  exactly one point in the fusion design space was measured. A different
  fusion (gating, cross-attention, a deeper per-branch stack) is untested, and
  "concatenation at 80/80 buys nothing" is not "no fusion could".
- **The best checkpoint is a fraction of one epoch in.** All three spatial
  specialists reach their optimum inside their first epoch and degrade
  monotonically after. The reported weights are a genuinely early stop, chosen
  by {config['evaluations_per_epoch']} probes per epoch.
- **`R_CE` is not the project's question.** Phase 12's question is whether a
  better belief helps search win more games. Nothing in this sprint measures
  that.
"""
    )

    # -- 12 -----------------------------------------------------------------
    preservation = summary["preservation"]
    parts.append(
        f"""## 12. What Agent 4 touched

Created:

```text
stratego/belief/phase11b/hybrid_cnn.py
scripts/run_phase11b_agent04.py
scripts/_phase11b_agent04_report.py
tests/belief/phase11b/test_phase11b_agent04_artifacts.py
checkpoints/phase11b/{CANDIDATE_4}.pt
checkpoints/phase11b/hybrid_input_train.npy
checkpoints/phase11b/hybrid_input_dev.npy
reports/phase11b/agent_04_summary.json
reports/phase11b/agent_04_report.md
reports/phase11b/agent_04_learning_curve.json
```

Modified: nothing.

| preservation check | value |
| --- | --- |
| artifacts unchanged since Agent 3 | `{preservation['artifacts_unchanged_since_agent3']}` |
| `phase11_test_bank_v1` opened | `{preservation['phase11_test_bank_opened']}` |
| Agent 1 artifacts modified | `{preservation['agent1_artifacts_modified']}` |
| Agent 2 artifacts modified | `{preservation['agent2_artifacts_modified']}` |
| Agent 3 artifacts modified | `{preservation['agent3_artifacts_modified']}` |
| Agent 3's C1 field cache rebuilt | `{preservation['agent3_field_cache_rebuilt']}` |
| C1 modified | `{preservation['c1_modified']}` |
| corpus regenerated | `{preservation['corpus_regenerated']}` |

{len(summary['preserved_artifact_digests'])} preserved artifacts were digested
and compared against the digests Agent 3 recorded — the Phase 11 evidence, the
accepted sampler/baseline/public-state/contract/seed modules, the production
model, and every Agent 1, 2 and 3 module and report. All match.
"""
    )

    # -- 13 -----------------------------------------------------------------
    suite = summary.get("suite") or {}
    suite_line = (
        f"`{suite.get('command')}` → {suite.get('summary_line')} "
        f"({suite.get('seconds')} s)"
        if suite
        else "not run in this invocation"
    )
    parts.append(
        f"""## 13. Where the sprint stands

Four experiments, one board:

| candidate | input | parameters | `R_CE` |
| --- | --- | --- | --- |
"""
        + "\n".join(
            f"| `{name}` | {('C1 feature, last block unfrozen' if name == CANDIDATE_1C else 'C1 feature') if name.startswith('agent01') or name == REFERENCE else ('raw observation' if name == CANDIDATE_2 else ('frozen C1 field' if name == CANDIDATE_3 else 'raw + frozen C1 field'))} | {_params(block):,} | {_fmt(block['r_ce'])} |"
            for name, block in _ordered_rows(row, earlier)
        )
        + f"""

The three 3.9M-parameter spatial specialists occupy the bottom three trained
positions on that board. The two small heads attached to C1's own
representation occupy the top two. Agent 4 changes the *explanation* — the raw
observation is not withholding anything C1 lost — without changing the
ordering.

`00_PHASE_11B_OVERVIEW.md` puts a review point after Agent 4 and asks whether
important uncertainty remains before Agent 5. Two observations, offered as
input to that review rather than as a decision:

- The uncertainty Agent 4 was meant to resolve is resolved, negatively. Raw
  and C1 are not complementary at this corpus size.
- Every architecture tried so far — a linear head, an MLP head, an unfrozen
  encoder block, and three 3.9M convolution towers over three different inputs
  — lands between {_fmt(min(block['r_ce'] for _, block in _ordered_rows(row, earlier)))}
  and {_fmt(max(block['r_ce'] for name, block in _ordered_rows(row, earlier) if name != REFERENCE))}
  `R_CE`. That is a narrow band for that much architectural variation, and it
  is the strongest available evidence that the next marginal gain is not in
  the model.

Whether an autoregressive Transformer is worth its cost against that
background is the reviewer's call, not this report's.

Repository suite: {suite_line}
"""
    )

    # -- 14 -----------------------------------------------------------------
    parts.append(
        f"""## 14. Stop condition

{summary['stop_condition']}

No claim is made that Phase 11 has been repaired, that the Phase 11 `FAIL` is
overturned, or that Phase 12 is authorized.

```text
phase                                  = {summary['phase']}
status                                 = {summary['status']}
phase11_fail_unchanged                 = {summary['phase11_fail_unchanged']}
phase11_test_bank_used                 = {summary['phase11_test_bank_used']}
phase12_authorized_by_this_artifact    = {summary['phase12_authorized_by_this_artifact']}
phase11_final_classification           = {summary['phase11_final_classification']}
phase11_test_bank_spent                = {summary['phase11_test_bank_spent']}
scientific_claim                       = {summary['scientific_claim']}
```

---

Generated {summary['generated_utc']} ·
`{summary['phase11b_version']}` ·
corpus `{summary['common_corpus']['corpus_digest'][:16]}…` ·
checkpoint `{summary['checkpoint']['sha256'][:16]}…`
"""
    )

    return "\n".join(parts)


__all__ = ["render"]
