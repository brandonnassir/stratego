# Phase 11B — Agent 4: Hybrid Raw + C1 CNN

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

## 0. What Agent 4 found

Agent 4's question was whether raw public information and C1's learned
strategic representation carry **complementary** belief signal — whether C1
supplies high-level abstractions while the raw observation restores
belief-specific detail C1 may have compressed away.

The answer is **no, not measurably**, and the shape of the answer is the
useful part.

1. **The raw observation adds nothing on top of C1.** The hybrid scores
   `R_CE` **0.9614** [0.9582, 0.9646] against Agent 3's C1-only tower
   at 0.9624. The paired game bootstrap of the
   cross-entropy difference is
   -0.00224
   [-0.00638, +0.00211] — the interval
   **straddles zero**, so on 497 development games
   the two are not distinguishable. Half the stem was handed to a
   127-channel public observation and the belief did not move.
2. **C1 adds a great deal on top of the raw observation.** The same fusion
   against Agent 2's raw-only tower is
   -0.01592
   [-0.02034, -0.01135], which is
   distinguishable and favours the hybrid. Agent 2 alone is
   0.9686.
3. So the two directions are **not symmetric**. For belief at this corpus
   size, C1's per-square representation is effectively a superset of the raw
   observation: adding C1 to raw pixels is worth
   0.0073 `R_CE`, and adding raw
   pixels to C1 is worth 0.0010,
   inside the noise.
4. **The sprint leader is unchanged.** All three 3.9M-parameter spatial
   specialists — raw, C1, hybrid — remain behind Agent 1's
   334,860-parameter attached head
   (0.9495) and its
   533,388-parameter 1C variant
   (0.9460). The leader by `R_CE` is
   **`agent01_1c_final_block_plus_mlp`** at 0.9460.

Read together with Agent 2's corpus-size sweep, the reading is consistent and
unflattering to more architecture: three different 3.9M-parameter inputs land
within 0.0073 `R_CE` of each
other and all of them lose to a 335k head. The binding constraint is the
corpus, not the representation and not the fusion.

## 1. The two inputs, and what was reused

`04_AGENT_4` requires two legal, public branches and forbids changing either
Agent 1's common corpus or Agent 3's frozen C1 seam.

```text
branch A   raw 127 x 10 x 10 public observation      the corpus's own bytes
branch B   frozen per-square C1 field [100, 128]     Agent 3's seam, unchanged
```

### The seam is Agent 3's, not a new one

| property | value |
| --- | --- |
| `seam_id` | `c1_encoder_output_all_tokens` |
| tensor | `ProductionModel.encode(tokens)` |
| definition | `encoder_norm(block_6(... block_1(input_projection(tokens) + position_embedding())))` |
| shape | `[batch, 100, 128]` |
| per-square | `True` |
| pooled | `False` |
| source | agent_03; reused unchanged |
| matches Agent 3's recorded description | `True` |

### Agent 3's cache was reused, not rebuilt

`04_AGENT_4`: "Reuse Agent 3's C1 feature cache if compatible and exact."
Compatibility and exactness are measured rather than assumed. The cache files
are opened read-only, their content digests are recomputed and compared to
the digests Agent 3 published, and a random sample of each split is
re-encoded from the public observations through the frozen C1.

| split | shape | digest matches Agent 3 | rows re-encoded | max abs difference |
| --- | --- | --- | --- | --- |
| train | `[26898, 100, 128]` | `True` | 64 | 0.0 |
| dev | `[1828, 100, 128]` | `True` | 64 | 0.0 |

`c1_field_rebuilt` is `False`. Agent 3's checkpoint, its
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
| train | `[26898, 255, 10, 10]` | 2.74 GB | `True` | `True` |
| dev | `[1828, 255, 10, 10]` | 0.19 GB | `True` | `True` |

Both halves re-derive bit-identically (64 random
rows per split), and `contains_labels` is `false` on every cache block. No
hidden truth enters either branch: the corpus keeps true ranks in a separate
`privileged/` directory that the loader hands over only when asked by name,
and neither of the model's two entry points has an argument a label could
arrive in.

### The common corpus

| | train | dev |
| --- | --- | --- |
| games | 2048 | 512 |
| positions | 26898 | 1828 |
| hidden pieces | 817,255 | 55,955 |
| setup-library split | `train` | `validation` |

`corpus_digest` `903bf10a3e34cfc0…`, recomputed
from the bytes on disk and equal to the digest Agents 1, 2 and 3 each
recorded. byte-for-byte; Agent 4 regenerated nothing.

## 2. The model

```text
raw 127 x 10 x 10  -> conv3x3 80 -> BN -> ReLU --\
                                                                   concat 160
C1 field [100,128] -> conv3x3 80 -> BN -> ReLU --/
                                                                        |
                                                    8 x residual 3x3 160
                                                                        |
                                                     1x1 128 -> BN -> ReLU -> 1x1 12
                                                                        |
                                                              12 logits per square
```

`hybrid(raw 127 -> conv3x3 80 || c1_field(128) -> conv3x3 80) -> concatenate 160 -> 8 x residual3x3 -> 1x1 128 -> 12/square, relu, batchnorm2d`

| block | parameters |
| --- | --- |
| raw branch | 91,600 |
| C1 branch | 92,320 |
| residual tower | 3,691,520 |
| read-out | 22,284 |
| **total** | **3,897,724** |

Inside the instructed `3,000,000`–`5,000,000`
band, and — deliberately — within 720
parameters of Agent 3 and 720 of Agent 2:

| candidate | stem | parameters |
| --- | --- | --- |
| Agent 2 | `conv3x3(127 -> 160)` | 3,897,004 |
| Agent 3 | `conv3x3(128 -> 160)` | 3,898,444 |
| Agent 4 | `conv3x3(127 -> 80) ‖ conv3x3(128 -> 80)` | 3,897,724 |

That is the whole design. The residual tower and the read-out are **Agent 2's,
imported rather than re-declared**, at Agent 2's width and Agent 2's depth,
and the seam is Agent 3's. What changes across the three reports is the stem.
A spread of 1,440
parameters — 0.04% — means a difference between the three numbers is a
difference of *input representation*, not of capacity.

### The choices, declared as choices

`04_AGENT_4` forbids branch-width, fusion-method, depth and learning-rate
sweeps. None was run. Three things had to be picked without one:

- **Branch widths 80 and 80.** The fused width had to be
  160 for the tower to be Agent 2's, so the only free choice
  was how to divide it. It is divided evenly, because the experiment asks
  whether the two sources are *complementary* and an uneven split would
  prejudge which one carries more.
- **Concatenation, not addition or gating.** Summing two projections would
  force both representations into one shared 160-channel
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
`1`.

### C1 is frozen, structurally

`build_hybrid_cnn` returns the specialist **alone**, so no optimizer built
from its parameters can reach a C1 weight. During training C1 is not even
called: the C1 branch reads Agent 3's cached field. `c1_parameters_updated`
is `0` and
`gradients_reaching_c1` is `False`.

## 3. The pilot, and where the budget came from

`04_AGENT_4` asks for "a brief throughput/sanity pilot first". It ran on every
available backend before any development metric existed.

| backend | s/step | positions/s | est. s/epoch | first loss | last loss |
| --- | --- | --- | --- | --- | --- |
| cpu | 1.063 | 240.8 | 111.7 | 2.4109 | 2.2912 |
| mps | 0.130 | 1966.4 | 13.7 | 2.4109 | 2.2915 |

Chosen: **`mps`**, 8.17x CPU.
MPS is used only when it is stable (finite, CPU-agreeing pilot losses) and materially faster (>= 1.5x); otherwise the accepted CPU backend. The two backends' pilot losses agree to
0.000317.

The epoch horizon is 60, from
20 target training minutes at the measured 14s/epoch, clamped to [12, 60] — a budget decision taken from *measured
throughput*, before a single development number existed.

The 2.74 GB fused training tensor was staged
on the training device (`stage_on_device` `True`) under
the declared rule: stage the fused tensor on the training device when it is at most 4 GiB, otherwise keep it in host memory and transfer per batch.

## 4. Results on the common development set

1,828 positions, 55,955 hidden pieces, the
identical development positions every Phase 11B candidate is scored on.

| metric | value |
| --- | --- |
| overall cross-entropy | 2.110101 |
| remaining-count baseline cross-entropy | 2.194927 |
| **`R_CE`** | **0.961354** |
| `R_CE` 95% CI | [0.9582, 0.9646] |
| top-1 hidden-rank accuracy | 0.2561 |
| baseline top-1 | 0.2038 |
| best stratum | `phase9_selfplay` |
| worst stratum | `strategic_rule` |
| projected-onto-legal `R_CE` (diagnostic only) | 0.9573 |

The uniform floor is `R_CE` 1.1321; every
candidate on the board beats it comfortably, so the leaderboard is a
comparison between models rather than between a model and noise.

### Per-stratum `R_CE`

| stratum | Agent 4 `R_CE` | Agent 4 top-1 | Agent 3 `R_CE` | Agent 2 `R_CE` | Agent 1 1B `R_CE` |
| --- | --- | --- | --- | --- | --- |
| Phase9-like | 0.9460 | 0.2741 | 0.9501 | 0.9560 | 0.9350 |
| Strategic | 0.9714 | 0.2534 | 0.9712 | 0.9760 | 0.9596 |
| Tactical | 0.9666 | 0.2576 | 0.9679 | 0.9717 | 0.9556 |
| Scout-rush | 0.9632 | 0.2373 | 0.9619 | 0.9721 | 0.9497 |

Scout-rush is the sprint's generalization stratum — the behaviour least like
the self-play the frozen C1 was trained on. Agent 4 scores
0.9632 there, against Agent 3's
0.9619, Agent 2's
0.9721 and Agent 1 1B's
0.9497. The raw
branch does not rescue the unusual stratum: raw-only is still the worst of the
three towers there, and Agent 4 and Agent 3 change places by
0.0014
— the hybrid is marginally *worse* on Scout-rush than the C1-only tower it is
marginally better than overall. Both gaps are far inside the equivalence band,
which is the honest reading: on this stratum, as overall, the second input is
not doing measurable work.

### The whole sprint board, best first

| # | candidate | `R_CE` | 95% CI | top-1 | trained parameters |
| --- | --- | --- | --- | --- | --- |
| 1 | `agent01_1c_final_block_plus_mlp` | 0.9460 | [0.9434, 0.9488] | 0.2640 | 533,388 |
| 2 | `agent01_1b_attached_mlp_head` | 0.9495 | [0.9471, 0.9524] | 0.2603 | 334,860 |
| 3 | `agent01_1a_existing_linear_head` | 0.9531 | [0.9508, 0.9558] | 0.2542 | 1,548 |
| 4 | `agent04_hybrid_raw_c1_cnn` | 0.9614 | [0.9582, 0.9646] | 0.2561 | 3,897,724 |
| 5 | `agent03_c1_feature_cnn` | 0.9624 | [0.9592, 0.9656] | 0.2569 | 3,898,444 |
| 6 | `agent02_raw_observation_cnn` | 0.9686 | [0.9660, 0.9713] | 0.2520 | 3,897,004 |
| 7 | `phase11_head_unchanged_reference` | 0.9834 | [0.9785, 0.9884] | 0.2303 | 1,548 |

### Paired game bootstraps

Marginal confidence intervals cannot say whether two candidates differ,
because they are scored on the same positions. These are paired bootstraps
over 497 development **games**, of the per-piece
cross-entropy difference (negative = Agent 4 lower):

| against | CE difference | 95% CI | distinguishable | reading |
| --- | --- | --- | --- | --- |
| `agent01_1b_attached_mlp_head` | +0.02611 | [+0.02226, +0.03013] | `True` | Agent 4 higher |
| `agent01_1c_final_block_plus_mlp` | +0.03386 | [+0.03011, +0.03774] | `True` | Agent 4 higher |
| `agent02_raw_observation_cnn` | -0.01592 | [-0.02034, -0.01135] | `True` | Agent 4 lower |
| `agent03_c1_feature_cnn` | -0.00224 | [-0.00638, +0.00211] | `False` | not distinguishable |
| `phase11_head_unchanged_reference` | -0.04830 | [-0.05845, -0.03842] | `True` | Agent 4 lower |

Every earlier candidate here was loaded read-only from its own checkpoint and
scored on these same pieces; nothing was retrained.
the unchanged Phase 11 head reproduces exactly; the trained checkpoints reproduce to the scale of the run-to-run drift their own agents measured, orders of magnitude below the gaps this leaderboard turns on. The recomputed values are used for the paired bootstraps so every candidate in a comparison comes from one scoring pass; the quoted leaderboard rows are left exactly as Agents 1, 2 and 3 reported them.

## 5. The no-rerun comparison table `04_AGENT_4` asks for

| candidate | id | `R_CE` | 95% CI | top-1 | Scout-rush `R_CE` | trained parameters | rerun by Agent 4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| old Phase 11 head | `phase11_head_unchanged_reference` | 0.9834 | [0.9785, 0.9884] | 0.2303 | 0.9997 | 1,548 | `False` |
| Agent 1 best attached head | `agent01_1c_final_block_plus_mlp` | 0.9460 | [0.9434, 0.9488] | 0.2640 | 0.9459 | 533,388 | `False` |
| Agent 2 raw CNN | `agent02_raw_observation_cnn` | 0.9686 | [0.9660, 0.9713] | 0.2520 | 0.9721 | 3,897,004 | `False` |
| Agent 3 C1-feature CNN | `agent03_c1_feature_cnn` | 0.9624 | [0.9592, 0.9656] | 0.2569 | 0.9619 | 3,898,444 | `False` |
| Agent 4 hybrid | `agent04_hybrid_raw_c1_cnn` | 0.9614 | [0.9582, 0.9646] | 0.2561 | 0.9632 | 3,897,724 | `False` |

"Agent 1 best attached head" is resolved as `agent01_1c_final_block_plus_mlp`
by the rule "lowest R_CE on Agent 1's own leaderboard", from Agent 1's own
leaderboard, so the table cannot silently quote the wrong one.
`prior_candidates_rerun` is `False`; every earlier
figure is read from reports/phase11b/agent_01_summary.json, agent_02_summary.json and agent_03_summary.json.

## 6. Is the signal complementary?

This is the question `04_AGENT_4` exists to answer, and it is answered
arithmetically rather than by assertion. Complementarity is a claim about the
hybrid beating **both** single-source towers of the same size — beating only
the weaker one would show nothing, since the hybrid contains the stronger
one's input. So the reference is the better of the two.

| quantity | value |
| --- | --- |
| Agent 2, raw-only | 0.9686 |
| Agent 3, C1-only | 0.9624 |
| **Agent 4, hybrid** | **0.9614** |
| better single source | `c1` at 0.9624 |
| hybrid − better single source | -0.00102 |
| hybrid − raw-only | -0.00725 |
| hybrid − C1-only | -0.00102 |
| equivalence band | 0.005 |
| **complementary** | **`False`** |

> not complementary: the fusion is no better than its stronger branch alone, so the second input adds no belief signal this corpus can expose beyond what the stronger one already carries

The paired bootstrap is the sharper statement. Against C1-only the difference
is -0.00224
[-0.00638, +0.00211] — the interval contains
zero, so the hybrid and the C1-only tower are **not distinguishable** on this
development set. Against raw-only it is
-0.01592
[-0.02034, -0.01135], which is distinguishable
and favours the hybrid.

### What that asymmetry means

The instruction's hypothesis had two halves — that C1 supplies strategic
abstraction, and that the raw observation restores belief detail C1
compressed away. The first half survives; the second does not.

- Give a fusion tower the C1 field on top of raw pixels and belief improves
  measurably. C1's six transformer blocks are contributing something the
  convolution tower cannot learn from the observation in
  26,898 positions.
- Give the same tower raw pixels on top of the C1 field and belief does not
  move. Whatever belief-relevant detail C1 compressed away, either it is not
  there, or 26,898
  correlated positions are not enough to learn to use it.

The second reading is the one Agent 2 already argued from a different
direction with its corpus-size sweep, and it is the reading this result
supports: the marginal value of *any* additional representation is being
absorbed by the corpus.

### Capacity was held fixed, so this is not a capacity story

| candidate | parameters |
| --- | --- |
| `agent02_raw_observation_cnn` | 3,897,004 |
| `agent03_c1_feature_cnn` | 3,898,444 |
| `agent04_hybrid_raw_c1_cnn` | 3,897,724 |
| spread | 1,440 |

the three candidates share Agent 2's residual tower and read-out and differ only in the stem, so a difference between them is a difference of input representation and not of capacity.

## 7. What was trained, and how

One architecture, one declared configuration, inherited from Agent 2 through
Agent 3 so the three candidates differ in input rather than in tuning effort.

| setting | value |
| --- | --- |
| run id | `run1_declared` |
| optimizer | `adamw`, lr 0.001, weight decay 0.0001 |
| schedule | `cosine` |
| batch | 256 positions |
| gradient clip | 1.0 |
| dropout | 0.0 / 0.0 |
| epoch horizon | 60 |
| patience | 5 |
| device | `mps` |
| trainer | `phase11b_raw_cnn_trainer_v1` (shared with `agent02_raw_observation_cnn`, `agent03_c1_feature_cnn`) |
| evaluations per epoch | 8 |
| input staged on device | `True` |

The loss is supervised hidden-rank cross-entropy over hidden pieces and
nothing else: `policy_or_value_terms` `False`,
`game_outcome_used` `False`.

### The overfitting signature, for the third time

| quantity | value |
| --- | --- |
| epochs run | 6 of 60 |
| stopped because | `patience` |
| best epoch | 1 (fraction 0.8585, step 91 of 106 per epoch) |
| development probes | 54 |
| train CE, first epoch | 2.0956 |
| train CE, last epoch | 0.6264 |
| dev CE at best | 2.1101 |
| dev CE, last epoch | 3.7511 |
| dev CE rose after best | `True` |

The optimum arrives 86% of the way through
the **first** epoch and development cross-entropy rises monotonically
thereafter while training cross-entropy collapses to
0.6264. That is the same signature Agent 2 and
Agent 3 recorded, on a third input representation. It is a statement about
3.9M parameters against
26,898 correlated positions, not about the fusion —
and it is why the sub-epoch probe cadence exists: a once-per-epoch probe would
have missed this candidate's best weights entirely.

### Run-to-run spread

The identical configuration and the identical seed were trained a second
time. This run is a **diagnostic**: `is_the_reported_candidate`
`False`, `checkpoint_written`
`False`.

| quantity | value |
| --- | --- |
| reported `R_CE` | 0.961354 |
| repeated `R_CE` | 0.961354 |
| absolute difference | 0.0 |
| best step matches | `True` |
| max epoch train-loss difference | 0.0 |

Bit-identical, which matches what Agent 3 measured for this model family:
MPS is not run-to-run deterministic for C1 transformer training, but it *is*
for these convolution/batch-norm belief towers. The spread is therefore not a
competing explanation for any gap on the board.

### The two backends agree

| quantity | value |
| --- | --- |
| training backend | `mps` |
| scoring backend | `cpu` |
| `R_CE` on the training backend | 0.96135379 |
| `R_CE` on CPU | 0.96135379 |
| absolute difference | 4.95e-09 |

The reported number is the CPU one, so the leaderboard row and every paired
bootstrap come from one scoring pass on the accepted evaluation backend.

## 8. Cost

| quantity | value |
| --- | --- |
| training wall-clock | 105.831 s |
| time to best checkpoint | 14.942 s |
| peak process RSS | 8.69 GB |
| checkpoint | 15.7 MB, `e25afd2480092bb2…` |

peak process RSS of the training stage: the materialized 2.7 GB fused input tensor, the model and the metric arrays, not the model alone.

### Inference latency, both honest readings

A C1-consuming candidate has two defensible prices and reporting only one
would be a rhetorical choice.

| reading | ms/decision (single) | ms/decision (batched) | µs/piece |
| --- | --- | --- | --- |
| specialist alone | 3.1474 | 1.0908 | 35.6354 |
| frozen C1 encode alone | 1.5067 | 0.2571 | 8.4005 |
| end to end | 4.6541 | 1.3479 | 44.0359 |

*Specialist alone* is the added cost inside a search that is already running
C1 for its policy — the situation this project is actually in. *End to end*
is a belief query in isolation, and is the number comparable to Agent 2,
which has no C1 stage. Measured on CPU at batch
256 with 10 repeats,
30.61 hidden pieces per decision.

For scale, Agent 1's attached head costs
0.4371 µs per piece on a
pass the search is already making.

## 9. Is this preferable to what already exists?

The sprint's engineering-winner rule: prefer materially lower overall `R_CE`,
give substantial weight to Scout-rush generalization, treat candidates within
roughly 0.005 `R_CE` as equivalent and prefer the
cheaper and simpler one, and count search-integration complexity.

| question | answer |
| --- | --- |
| leader by `R_CE` | `agent01_1c_final_block_plus_mlp` at 0.9460 |
| within the equivalence band of the leader | `agent01_1c_final_block_plus_mlp`, `agent01_1b_attached_mlp_head` |
| best earlier candidate | `agent01_1c_final_block_plus_mlp` at 0.9460 |
| Agent 4 − best earlier | +0.01539 |
| Agent 4 materially better than best earlier | `False` |
| Agent 4 is the leader | `False` |

**No.** Agent 4 is +0.0154
`R_CE` against `agent01_1c_final_block_plus_mlp` — worse, distinguishably so
(+0.03386
[+0.03011, +0.03774]) —
while costing 7.3x
the parameters and a second network with its own checkpoint.

On Scout-rush, the generalization stratum the rule weights specially:

| candidate | Scout-rush `R_CE` |
| --- | --- |
| `agent01_1c_final_block_plus_mlp` | 0.9459 |
| `agent01_1b_attached_mlp_head` | 0.9497 |
| `agent01_1a_existing_linear_head` | 0.9552 |
| `agent03_c1_feature_cnn` | 0.9619 |
| `agent04_hybrid_raw_c1_cnn` | 0.9632 |
| `agent02_raw_observation_cnn` | 0.9721 |
| `phase11_head_unchanged_reference` | 0.9997 |

Agent 4 does not win that column either.

On search integration: Agent 4 needs both C1's encode and its own tower: a search already running C1 for its policy pays the specialist's 3.15 ms per position, and a belief query in isolation costs 4.65 ms. That is Agent 3's integration cost plus a second input path, against Agent 1's head, which is a tensor on a pass the search already makes.

## 10. Required interface

`04_AGENT_4` requires `predict_marginals(public_state)` and
`sample_worlds(public_state, n, seed)`. Both are exposed by
`HybridBeliefModel`, which subclasses Agent 1's shared interface rather than
reimplementing it — so `sample_worlds` runs through the **accepted,
unmodified** Phase 11 sampler as inherited code, not a fork.

| check | value |
| --- | --- |
| interface version | `phase11b_belief_interface_v1` |
| positions exercised | 16 |
| worlds sampled | 128 (8 per position) |
| mean unresolved pieces per position | 32.75 |
| all marginals are probability vectors | `True` |
| `sample_worlds` is seed-deterministic | `True` |
| every world passed the accepted validation stack | `True` |
| sampler | `stratego.evaluation.phase11_sampler (accepted, unmodified)` |
| reads hidden truth | `False` |

The positions are replayed from the common corpus's own development plans,
exactly as Agents 1, 2 and 3 build them, so all four interface blocks describe
the same interface on the same kind of state.

In deployment the C1 branch's input is derived live from the public
observation through the same frozen encoder and the same `feature_layer`
(`final`) the cached field was built from, so the trained path
and the deployed path read the identical tensor.

## 11. Caveats a reader should carry forward

- **This is a development-set engineering comparison, not a sealed test.**
  The Phase 11B development set is explicitly "an engineering comparison set,
  not a scientifically sealed bank". Nothing here is a scientific result.
- **A negative complementarity finding is corpus-conditional.** "The raw
  observation adds nothing on top of C1" is measured at
  26,898 training
  positions from 2,048
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
  by 8 probes per epoch.
- **`R_CE` is not the project's question.** Phase 12's question is whether a
  better belief helps search win more games. Nothing in this sprint measures
  that.

## 12. What Agent 4 touched

Created:

```text
stratego/belief/phase11b/hybrid_cnn.py
scripts/run_phase11b_agent04.py
scripts/_phase11b_agent04_report.py
tests/belief/phase11b/test_phase11b_agent04_artifacts.py
checkpoints/phase11b/agent04_hybrid_raw_c1_cnn.pt
checkpoints/phase11b/hybrid_input_train.npy
checkpoints/phase11b/hybrid_input_dev.npy
reports/phase11b/agent_04_summary.json
reports/phase11b/agent_04_report.md
reports/phase11b/agent_04_learning_curve.json
```

Modified: nothing.

| preservation check | value |
| --- | --- |
| artifacts unchanged since Agent 3 | `True` |
| `phase11_test_bank_v1` opened | `False` |
| Agent 1 artifacts modified | `False` |
| Agent 2 artifacts modified | `False` |
| Agent 3 artifacts modified | `False` |
| Agent 3's C1 field cache rebuilt | `False` |
| C1 modified | `False` |
| corpus regenerated | `False` |

38 preserved artifacts were digested
and compared against the digests Agent 3 recorded — the Phase 11 evidence, the
accepted sampler/baseline/public-state/contract/seed modules, the production
model, and every Agent 1, 2 and 3 module and report. All match.

## 13. Where the sprint stands

Four experiments, one board:

| candidate | input | parameters | `R_CE` |
| --- | --- | --- | --- |
| `agent01_1c_final_block_plus_mlp` | C1 feature, last block unfrozen | 533,388 | 0.9460 |
| `agent01_1b_attached_mlp_head` | C1 feature | 334,860 | 0.9495 |
| `agent01_1a_existing_linear_head` | C1 feature | 1,548 | 0.9531 |
| `agent04_hybrid_raw_c1_cnn` | raw + frozen C1 field | 3,897,724 | 0.9614 |
| `agent03_c1_feature_cnn` | frozen C1 field | 3,898,444 | 0.9624 |
| `agent02_raw_observation_cnn` | raw observation | 3,897,004 | 0.9686 |
| `phase11_head_unchanged_reference` | C1 feature | 1,548 | 0.9834 |

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
  — lands between 0.9460
  and 0.9686
  `R_CE`. That is a narrow band for that much architectural variation, and it
  is the strongest available evidence that the next marginal gain is not in
  the model.

Whether an autoregressive Transformer is worth its cost against that
background is the reviewer's call, not this report's.

Repository suite: `python -m pytest tests -q` → 6016 passed, 3 skipped in 346.43s (0:05:46) (347.6 s)

## 14. Stop condition

Agent 4 trained one architecture and stopped. The Transformer (Agent 5) was not begun and is not authorized by this artifact. Phase 11 remains FAIL, phase11_test_bank_v1 remains spent and unopened, and nothing here authorizes Phase 12.

No claim is made that Phase 11 has been repaired, that the Phase 11 `FAIL` is
overturned, or that Phase 12 is authorized.

```text
phase                                  = phase11b
status                                 = engineering_prototype
phase11_fail_unchanged                 = True
phase11_test_bank_used                 = False
phase12_authorized_by_this_artifact    = False
phase11_final_classification           = FAIL
phase11_test_bank_spent                = True
scientific_claim                       = none
```

---

Generated 2026-08-20T03:53:25Z ·
`phase11b_engineering_v1` ·
corpus `903bf10a3e34cfc0…` ·
checkpoint `e25afd2480092bb2…`
