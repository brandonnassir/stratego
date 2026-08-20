# Phase 11B — Agent 2: Raw-Observation CNN

**Status: engineering prototype.** This report does not repair Phase 11, does
not overturn the Phase 11 `FAIL`, and does not authorize Phase 12.
`phase11_test_bank_v1` was not opened; it remains spent.

| marker | value |
| --- | --- |
| `phase` | `phase11b` |
| `status` | `engineering_prototype` |
| `phase11_fail_unchanged` | `True` |
| `phase11_test_bank_used` | `False` |
| `phase12_authorized_by_this_artifact` | `False` |
| `phase11_final_classification` | `FAIL` |
| `phase11_test_bank_spent` | `True` |
| `scientific_claim` | `none` |

## 0. What Agent 2 found

Agent 2's question was whether giving belief inference **its own
raw-observation spatial specialist** — bypassing C1's learned compression
entirely — solves most of the predictive weakness. On the common Phase 11B
development set:

1. **It does not help.** The raw CNN scores `R_CE` **0.9686** against Agent 1's best attached head at **0.9495** — 0.0191 *worse*, outside the sprint's 0.005 equivalence band.
   The paired game bootstrap of the cross-entropy difference is +0.0420 [0.0373, 0.0469], so the ordering is real, not noise.
2. **Against the unchanged Phase 11 head** (`R_CE` 0.9834 on these same fresh positions) the raw CNN is 0.0148 better, and top-1 hidden-rank accuracy moves 0.2303 -> 0.2520.
3. **It costs more to run.** 3,897,004 parameters against 334,860, 3.42 ms per position on CPU, and a second network that does not ride along on the policy's forward pass.

On the hardest generalization stratum, Scout-rush, the raw CNN scores `R_CE` 0.9721 against 0.9497 for Agent 1's winner and 0.9997 for the unchanged Phase 11 head.

None of this is a scientific claim, a repair of Phase 11, or evidence about
whether better beliefs win more games. It is one engineering measurement on
one fresh development set.

## 1. The model

```text
public 127 x 10 x 10 observation
    -> 3x3 spatial projection to width 160
    -> 8 residual 3x3 convolution blocks
    -> 1x1 read-out at width 128
    -> 12 rank logits per square
```

| part | parameters |
| --- | ---: |
| spatial projection | 183,200 |
| residual tower (8 blocks) | 3,691,520 |
| per-square read-out | 22,284 |
| **total** | **3,897,004** |

3,897,004 parameters, mid-band of the instructed 3-5M range, and the count was calculated before the run rather than after it.
The tower is 17 3x3 convolutional layers, a 35x35 receptive field: every square sees the whole 10x10 board with margin.

**One architecture, no sweep.** Width, depth and read-out width were
declared once, before the run, and no second architecture was built or
considered. Two *optimization* configurations of this one architecture were
declared and both are reported — "Two declared configurations" below says
exactly why, and "What was trained" gives the selected one verbatim.

The model's only input is the 127-channel public observation. It receives no
C1 feature, no privileged tensor and no hidden rank: the corpus stores true
ranks in a separate directory, the loader hands them over only when asked by
name, and `forward` takes exactly one argument.

## 2. The pilot, and where the budget came from

| backend | s/step | positions/s | estimated s/epoch | pilot loss |
| --- | ---: | ---: | ---: | --- |
| `cpu` | 1.038 | 246 | 109 | 2.4071 -> 2.3064 |
| `mps` | 0.121 | 2,114 | 13 | 2.4070 -> 2.3062 |

MPS was **8.6x** the accepted CPU backend and its pilot losses agree with CPU's to 2.1e-04, which is the "stable and materially faster" test `02_AGENT_2` sets. Training ran on `mps`.

The epoch horizon — **60** — was set from 20 target training minutes at the measured 13s/epoch, clamped to [12, 60]. That decision was made from measured
throughput **before any development metric existed**, which is what keeps it
a budget choice rather than a tuned hyperparameter. Neither run came close
to spending it: both stopped on patience long before epoch 60.

## 3. Two declared configurations, and why there are two

`02_AGENT_2` asks for **one** engineering run. Agent 2 ran 2 configurations of the **one** architecture, and this section says plainly why, because the deviation matters more than the number it produced.

| run | configuration | epochs run | best at (epochs) | dev R_CE | train CE first → last | dev CE first → last |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run1_declared` | lr 0.001, wd 0.0001, dropout 0/0 | 6 of 60 | 0.86 | **0.9686** | 2.1382 → 0.6389 | 2.1581 → 3.2994 |
| `run2_regularized` | lr 0.00025, wd 0.05, dropout 0.1/0.3 | 10 of 60 | 1.47 | 0.9691 | 2.1861 → 0.7533 | 2.1347 → 3.0144 |

The first two CE columns are epoch-boundary values, which is where the
overfitting is easiest to see; the `dev R_CE` column is the best checkpoint
of the whole run, found by the sub-epoch probe described below.

**Run 1 was the configuration declared before any result existed** — Agent 1's own optimizer family, so that the two experiments would differ in architecture rather than in tuning effort. Read at epoch boundaries it looks like a failed run: training cross-entropy fell 2.1382 → 0.6389 while development cross-entropy rose 2.1581 → 3.2994, and patience stopped it after 6 epochs of 60.

That pattern is a statement about capacity against corpus, not about the
architecture. The corpus holds 26,898 training positions drawn from 2,048
games — 13 positions per game, and **the hidden ranks are constant within**
**a game** — against 3.9M parameters. Agent 1's heads never met this
problem: they train 1,548-334,860 parameters on a representation that was
already pretrained on far more data than this corpus contains.

Two things followed from that diagnosis, and it matters which is which.

**A measurement fix, applied to both runs.** Both runs reach their development optimum inside the first epoch or two — the kept checkpoint of `run1_declared` is step 91 of 106 per epoch, 0.86 epochs in. A once-per-epoch probe cannot find a checkpoint that good, and `02_AGENT_2` asks for the best development checkpoint, so the trainer probes development cross-entropy 8 times per epoch and keeps the best weights it ever saw. This changes nothing about training — same batches, same order, same optimizer state — only how finely the run is *observed*, and it applies identically to both runs.

That fix is worth 0.0146 `R_CE` on the reported run — read at epoch boundaries this candidate scores 0.9832, and read at the probe it scores 0.9686. For scale, that is almost exactly this candidate's whole margin over the unchanged Phase 11 head (0.0148) and 76% of its deficit against Agent 1's winner (0.0191). Coarse checkpointing, not the architecture, would have been the largest single error in this report.

**A second configuration, which turned out not to be needed.** One corrective configuration was declared against the overfitting — dropout, a 4x lower learning rate, a 500x stronger decoupled weight decay — to test whether run 1's optimizer, rather than the architecture or the corpus, was the limitation. It was not: the two configurations land 0.0005 `R_CE` apart (0.9686 against 0.9691) — 10x narrower than the sprint's equivalence band and 39x narrower than the gap to Agent 1. Regularizing the run slowed the memorization and moved the optimum later; it changed the ceiling by almost nothing.

**The reported candidate is therefore `run1_declared` — the configuration declared up front.** The selection rule was fixed in advance (lowest development cross-entropy) and it chose the run that needed no correction.

**This is one more run than the instruction's letter allows, and it is
not a sweep**: two configurations were declared, both are reported in
full above and in `agent_02_learning_curve.json`, no third was tried, and
no architecture variant was built. Because the two agree to
0.0005 `R_CE`, a reader who rejects the deviation entirely can read run 1's row alone and reach the same conclusion about this candidate.

## 4. Results on the common development set

All rows are the same 55,955 hidden pieces of the same 1,828 development decisions, from corpus `903bf10a…`, and all divide by the same `remaining_count_belief_v1` denominator (CE 2.1949, top-1 0.2038).

| candidate | architecture | CE | R_CE | 95% CI | top-1 | trained params |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `agent01_1c_final_block_plus_mlp` | c1_last_block + mlp(128->512->512->12, gelu) | 2.0763 | 0.9460 | [0.9434, 0.9488] | 0.2640 | 533,388 |
| `agent01_1b_attached_mlp_head` | mlp(128->512->512->12, gelu) | 2.0841 | 0.9495 | [0.9471, 0.9524] | 0.2603 | 334,860 |
| `agent01_1a_existing_linear_head` | linear(128->12) | 2.0920 | 0.9531 | [0.9508, 0.9558] | 0.2542 | 1,548 |
| `agent02_raw_observation_cnn` | raw_cnn(127 -> conv3x3 160 -> 8 x residual3x3 -> 1x1 128 -> 12/square, relu, batchnorm2d) | 2.1260 | **0.9686** | [0.9660, 0.9713] | 0.2520 | 3,897,004 |
| `phase11_head_unchanged_reference` | linear(128->12) | 2.1584 | 0.9834 | [0.9785, 0.9884] | 0.2303 | 0 |

A flat 12-way vector scores `R_CE` 1.1321 — the uninformed floor.

Agent 1's rows are quoted from `agent_01_summary.json`; **none of Agent 1's**
**experiments was rerun**. To make the comparison paired rather than two
overlapping marginal intervals, Agent 1's saved checkpoints were additionally
loaded read-only and *scored* on the same pieces. They reproduce:

| candidate | Agent 1 reported | recomputed here | difference |
| --- | ---: | ---: | ---: |
| `agent01_1b_attached_mlp_head` | 0.9495 | 0.9495 | 0.000064 |
| `agent01_1c_final_block_plus_mlp` | 0.9460 | 0.9459 | 0.000032 |
| `phase11_head_unchanged_reference` | 0.9834 | 0.9834 | 0.000000 |

### Paired game bootstraps

A negative difference means the raw CNN has the lower cross-entropy.

| comparison | mean ΔCE | 95% CI | distinguishable |
| --- | ---: | --- | --- |
| agent02_raw_observation_cnn vs agent01_1b_attached_mlp_head | +0.0420 | [0.0373, 0.0469] | yes |
| agent02_raw_observation_cnn vs agent01_1c_final_block_plus_mlp | +0.0498 | [0.0449, 0.0547] | yes |
| agent02_raw_observation_cnn vs phase11_head_unchanged_reference | -0.0324 | [-0.0434, -0.0217] | yes |

### Per-stratum R_CE

| candidate | phase9_selfplay | strategic_rule | tactical_rule | scout_rush |
| --- | ---: | ---: | ---: | ---: |
| `agent01_1c_final_block_plus_mlp` | 0.9317 | 0.9557 | 0.9523 | 0.9459 |
| `agent01_1b_attached_mlp_head` | 0.9350 | 0.9596 | 0.9556 | 0.9497 |
| `agent01_1a_existing_linear_head` | 0.9379 | 0.9628 | 0.9582 | 0.9552 |
| `agent02_raw_observation_cnn` | 0.9560 | 0.9760 | 0.9717 | 0.9721 |
| `phase11_head_unchanged_reference` | 0.9614 | 0.9883 | 0.9859 | 0.9997 |

## 5. Architecture-limited, or corpus-limited?

Reaching the development optimum a fraction of an epoch in is consistent
with two opposite stories: the architecture cannot express more, or the
corpus cannot support more. They imply opposite advice for Agents 3-5, so
the selected configuration (`run1_declared`) was retrained on
halves of the corpus, sliced by whole games.

| training games | positions | hidden pieces | best dev R_CE | best at (epochs) |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 6,702 | 199,816 | 0.9907 | 1.56 |
| 1,024 | 13,499 | 406,968 | 0.9775 | 1.25 |
| 2,048 | 26,898 | 817,255 | **0.9686** | 0.86 |

**Best development R_CE improves monotonically with training games, so the candidate is corpus-limited at this corpus size.**
Quadrupling the training games moves best development `R_CE` by 0.0221, and the curve has not flattened.

These runs are diagnostics: none of them is the reported candidate, none
wrote a candidate checkpoint, and the leaderboard is identical without
them. They exist because the difference between "this architecture cannot"
and "this corpus cannot" is the single most useful thing Agent 2 can hand
to the agents that follow.

## 6. What was trained, and how

| field | value |
| --- | --- |
| trainable | the whole 3,897,004-parameter CNN, from scratch |
| frozen | nothing — this candidate has no C1 stage |
| optimizer | adamw + cosine |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| gradient clip | 1.0 |
| batch | 256 positions (~30 hidden pieces each) |
| epochs | 6 of 60 |
| stopped | patience |
| best checkpoint | step 91 of 106/epoch — 0.86 epochs, of 54 probes |
| device | mps |

The loss is supervised hidden-rank cross-entropy over the 817,255 hidden pieces of the 26,898 training decisions and nothing else: no policy term, no value term, no game outcome anywhere. The optimizer family is deliberately Agent 1's declared one (AdamW, cosine, `1e-3`, weight decay `1e-4`), so the two experiments differ in architecture rather than in tuning effort.

The supervised squares are gathered with the same helper Agent 1's
Experiment 1C uses, so Agent 2 is trained on exactly the pieces, in exactly
the order, that every Phase 11B candidate is scored on.

## 7. Cost

| item | value |
| --- | ---: |
| training wall clock | 101 s |
| time to best checkpoint | 14 s |
| parameters | 3,897,004 |
| checkpoint | 15.7 MB |
| inference, cpu: one position | 3.42 ms |
| inference, cpu: batched, per position | 0.806 ms |
| inference, mps: one position | 1.28 ms |
| inference, mps: batched, per position | 0.128 ms |
| peak memory | 5.44 GB |

Peak memory is the peak process RSS of the training stage: the materialized 1.4 GB training observation tensor, the model and the metric arrays, not the model alone. Inference is priced per *position*, not per piece, because a convolution tower produces all 100 squares in one pass; at the corpus's 30.6 hidden pieces per decision that is 26.322 µs/piece batched, against 0.437 µs/piece for Agent 1's winner — but the honest comparison is that Agent 1's head rides on a C1 forward pass a search already pays for, and this model is a second network.

## 8. Is this preferable to Agent 1?

**No. The raw CNN is 0.0191 `R_CE` *worse* than `agent01_1b_attached_mlp_head` and also more expensive, so it loses on both axes of the rule.**

How the sprint's engineering-winner rule applies:

- leader by `R_CE`: `agent01_1c_final_block_plus_mlp` (0.9460);
- inside the 0.005 equivalence band of the leader: `agent01_1c_final_block_plus_mlp`, `agent01_1b_attached_mlp_head`;
- Scout-rush / generalization: 0.9721 for the raw CNN against 0.9497 for Agent 1's winner;
- search-integration complexity: Agent 2 is a second network: it does not share the policy's forward pass, so a search that already runs C1 pays an additional 3.42 ms per position for belief, against a head that rides along on C1's existing encode.

The band is measured against the leader only, never as a chain of pairwise
comparisons — the same convention Agent 1 recorded.

## 9. Required interface

```text
predict_marginals(public_state)      -> {piece_slot: 12-way rank probabilities}
sample_worlds(public_state, n, seed) -> complete legal hidden armies
```

| candidate | positions | worlds | marginals valid | seed-deterministic | worlds legal |
| --- | ---: | ---: | --- | --- | --- |
| `agent02_raw_observation_cnn` | 16 | 128 | yes | yes | yes |

Every world was drawn through **`stratego.evaluation.phase11_sampler`, the
accepted Phase 11 sampler, imported and unmodified**. Agent 2 supplies
marginals and nothing else, through the same `Phase11BPublicState` Agent 1
defined — a container with exactly two public fields and no field a true rank
could arrive in. The Agent 2 adapter *subclasses* Agent 1's interface rather
than reimplementing it, so `sample_worlds` is inherited code, not a fork.

## 10. Caveats a reader should carry forward

- **This is a development-set number.** There is no sealed bank behind it and
  no scientific claim attached to it. The development set is an engineering
  comparison set, exactly as the sprint defines it.
- **The checkpoint was trained on `mps` and scored on CPU.** The two backends agree to 3.00e-11 `R_CE`, so the headline number does not depend on which one produced it. Neither backend's float32 reductions are bit-reproducible, so this was measured rather than assumed.
- **The kept checkpoint is 0.86 epochs into a 6-epoch run** (60 scheduled; stopped by `patience`). Development `R_CE` ended 0.5346 worse than at the best probe, so the curve had turned well before the run stopped and the kept checkpoint is nowhere near the last one.
- **The headline `R_CE` uses the accepted raw-softmax convention** — no
  masking, no epsilon, full simplex — because that is how the Phase 11 head
  was measured and how the accepted sampler consumes a belief. Renormalizing
  onto the publicly legal support is a diagnostic only (0.9635 against 0.9686 raw).
- **The reference row is not the Phase 11 sealed-test result.** It is the
  unchanged Phase 11 head scored on *these* fresh positions. Phase 11's sealed
  test remains what it was, and its bank remains spent.
- **No repeat under an identical configuration.** Agent 1 measured its candidates' run-to-run spread by retraining each one; Agent 2's two runs differ in configuration, so they do not measure that. What they do bound is looser and still useful: two *different* optimization settings of this architecture land 0.0005 `R_CE` apart, so the reported number is not balanced on a single lucky seed.

## 11. What Agent 2 touched

The common corpus was reused **byte-for-byte**: both splits' file digests and the whole-corpus digest `903bf10a3e34cfc0…` were recomputed from disk and matched against the values Agent 1 recorded. Nothing was regenerated.

| statement | value |
| --- | --- |
| corpus regenerated | `False` |
| Agent 1 artifacts modified | `False` |
| Phase 11 artifacts unchanged since Agent 1 | `True` |
| `phase11_test_bank_v1` opened | `False` |

Repository suite after Agent 2: **5908 passed, 3 skipped in 342.68s (0:05:42)** (`python -m pytest tests -q`).

## 12. Handoff to Agent 3

Agent 2 does not begin Agent 3's experiment and does not recommend for or
against running it. What Agent 2 measured that Agent 3 should carry:

1. **Learning a belief representation from scratch on this corpus does not work yet.** 3,897,004 parameters reach their development optimum 0.86 epochs in and then memorize, and a heavily regularized configuration moves the ceiling by only 0.0005 `R_CE`. Borrowing C1's pretrained representation beats learning one here, and by a wide margin.
2. **The binding constraint is the corpus, not the architecture.** Best development `R_CE` improves 0.9907 -> 0.9686 from 512 to 2,048 training games and has not flattened. A from-scratch spatial specialist is not refuted by this result; it is untested at a corpus size that would give it a chance, and Phase 11B's common corpus was sized for cheap head experiments.
3. **Agents 3 and 4 read the C1 features, so they inherit the cheap side of this trade.** They should still expect the overfitting régime to bite wherever they add trainable capacity: 26,898 positions from 2,048 games, with hidden ranks constant inside a game, is a small supervision set however it is presented.
4. **Probe development loss several times per epoch.** Both Agent 2 runs reached their optimum in the first epoch or two, and any candidate that trains comparable capacity on this corpus should expect the same. At epoch granularity this one would have been reported at `R_CE` 0.9832 instead of 0.9686.

## 13. Stop condition

Agent 2 trained one architecture and stopped. Agent 3's experiment was not
begun. Phase 11 remains `FAIL`, `phase11_test_bank_v1` remains spent and
unopened, and nothing in this report claims that Phase 11 has been repaired
or that Phase 12 is authorized.
