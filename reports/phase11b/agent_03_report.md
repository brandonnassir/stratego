# Phase 11B — Agent 3: C1-Feature CNN

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

## 0. What Agent 3 found

Agent 3's question was whether the final C1 move model still carries enough
belief-relevant information, and whether the old Phase 11 belief classifier
was mainly an **extraction** bottleneck. The experiment gives a two-part
answer, and the two parts point in opposite directions.

1. **C1's representation is the better one.** Give the *same* 3.9M-parameter
   spatial specialist the frozen C1 per-square field instead of the raw
   observation and `R_CE` moves 0.9686 ->
   **0.9624** [0.9592, 0.9656]. The paired game bootstrap of the
   cross-entropy difference is
   -0.0137
   [-0.0185, -0.0093], so the
   ordering is real. C1 is **not** discarding the information.
2. **Extraction was not a capacity problem.** Agent 1's
   334,860-parameter attached head reads **the same seam** and
   scores 0.9495, which is
   0.0129 `R_CE` *better* than
   this candidate — also a distinguishable paired difference. Adding 3.9M
   parameters of spatial capacity on top of the richer version of the same
   features made the belief **worse**, not better.

So the old head was an extraction bottleneck in the sense Agent 1 already
measured — dedicated belief optimization of the *unchanged* 1,548-parameter
linear head moved 0.9834 ->
0.9531 — and
**not** in the sense that a bigger, spatial extractor on the same seam would
have helped. On this corpus it hurts.

None of this is a scientific claim, a repair of Phase 11, or evidence about
whether better beliefs win more games. It is one engineering measurement on
one fresh development set.

## 1. The frozen seam

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
| seam id | `c1_encoder_output_all_tokens` |
| tensor | `ProductionModel.encode(tokens)` |
| definition | `encoder_norm(block_6(... block_1(input_projection(tokens) + position_embedding())))` |
| shape | `[B, 100, 128]` |
| per-square | `True` |
| pooled | `False` |
| consumed by | `policy_head`, `value_head`, `belief_head` |

Token i is row-major normalized square i (row i // 10, column i % 10), the accepted observation_to_tokens order, so the field maps back to board cells with a transpose and a reshape and nothing else.

**What was rejected, and why:**

| candidate tensor | rejected because |
| --- | --- |
| `hidden.mean(dim=1)` | pooled to one global 128-vector; not per-square |
| `belief_output(hidden)` | already compressed 128 -> 12 by the accepted head |
| penultimate block input | per-square but one encoder block short of the heads |
| `policy_query` / `policy_key` | a single head's own projection, not the shared tensor |

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
| --- | --- | ---: | ---: | --- |
| `train` | `(26898, 100, 128)` | 1377 MB | 0.0 | `d44453283cf8117f…` |
| `dev` | `(1828, 100, 128)` | 94 MB | 0.0 | `33b3f83af328b63e…` |

Both requirements are measurements rather than assurances. The cache turns
every training epoch into a pass over a fixed matrix instead of
26,898 transformer forward passes, and it was built
in 0.6s total. Derivability was checked by
re-encoding a random sample of each split from the public observations and
comparing:

| split | rows re-derived | max abs difference | bit-identical |
| --- | ---: | ---: | --- |
| `dev` | 64 | 0.00e+00 | True |
| `train` | 64 | 0.00e+00 | True |

The inputs to that re-derivation are the public observation and the accepted
frozen C1 weights, and nothing else: no label, no privileged array and no
Agent 3 parameter takes part.

Built on the accepted CPU evaluation backend even though MPS is available: the cache is what the specialist trains on and what the deployed interface recomputes live on CPU, and the two backends' encodes agree only to ~1e-7.

## 2. The model

```text
public 127 x 10 x 10 observation
    -> frozen C1                       (accepted Phase 9 weights, never updated)
    -> per-square C1 field [100, 128]
    -> 3x3 spatial projection to width 160
    -> 8 residual 3x3 convolution blocks
    -> 1x1 read-out at width 128
    -> 12 rank logits per square
```

| part | parameters |
| --- | ---: |
| spatial projection | 184,640 |
| residual tower (8 blocks) | 3,691,520 |
| per-square read-out | 22,284 |
| **total, all trainable** | **3,898,444** |
| frozen C1, never updated | 863,959 |

**The tower is Agent 2's, inherited rather than chosen.** Width, depth and
read-out width were not picked here at all: `03_AGENT_3` wants this candidate
read against Agent 2's raw-observation CNN, and a comparison like that is
only clean if the specialist is held fixed and the *representation* is the
thing that changes. Agent 2 is 3,897,004 parameters and
this is 3,898,444 — the 1,440
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
handed a C1 parameter. `c1_parameters_updated` = 0.

## 3. The pilot, and where the budget came from

| backend | s/step | positions/s | estimated s/epoch | pilot loss |
| --- | ---: | ---: | ---: | --- |
| `cpu` | 1.014 | 252 | 107 | 2.4433 -> 2.3200 |
| `mps` | 0.124 | 2,060 | 13 | 2.4433 -> 2.3197 |

MPS was **8.16x** the accepted CPU backend and
its pilot losses agree with CPU's to
2.7e-04. Training ran on
`mps`. The rule is Agent 2's, unchanged:
MPS is used only when it is stable (finite, CPU-agreeing pilot losses) and materially faster (>= 1.5x); otherwise the accepted CPU backend.

The epoch horizon — **60** — is
20 target training minutes at the measured 13s/epoch, clamped to [12, 60]. That decision was made from measured
throughput **before any development metric existed**, which is what keeps it
a budget choice rather than a tuned hyperparameter. The run did not come
close to spending it: it stopped on `patience` after
7 epochs.

## 4. Results on the common development set

All rows are the same 55,955 hidden pieces of the same
1,828 development decisions, from corpus
`903bf10a…`, and all divide by the same
`remaining_count_belief_v1` denominator (CE 2.1949, top-1
0.2038).

| candidate | representation | CE | R_CE | 95% CI | top-1 | trained params |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `agent01_1c_final_block_plus_mlp` | frozen C1 penultimate + last block retrained | 2.0763 | 0.9460 | [0.9434, 0.9488] | 0.2640 | 533,388 |
| `agent01_1b_attached_mlp_head` | frozen C1 feature at the piece's square | 2.0841 | 0.9495 | [0.9471, 0.9524] | 0.2603 | 334,860 |
| `agent01_1a_existing_linear_head` | frozen C1 feature at the piece's square | 2.0920 | 0.9531 | [0.9508, 0.9558] | 0.2542 | 1,548 |
| `agent03_c1_feature_cnn` | frozen C1 field, all 100 squares | 2.1123 | **0.9624** | [0.9592, 0.9656] | 0.2569 | 3,898,444 |
| `agent02_raw_observation_cnn` | raw 127-channel observation | 2.1260 | 0.9686 | [0.9660, 0.9713] | 0.2520 | 3,897,004 |
| `phase11_head_unchanged_reference` | frozen C1 feature at the piece's square | 2.1584 | 0.9834 | [0.9785, 0.9884] | 0.2303 | 1,548 |

A flat 12-way vector scores `R_CE` 1.1321 —
the uninformed floor.

Agent 1's and Agent 2's rows are quoted from their own summaries; **no
earlier experiment was rerun**. To make the comparisons paired rather than
five overlapping marginal intervals, their saved checkpoints were
additionally loaded read-only and *scored* on the same pieces. They
reproduce:

| candidate | reported by its agent | recomputed here | difference |
| --- | ---: | ---: | ---: |
| `agent01_1b_attached_mlp_head` | 0.9495 | 0.9495 | 0.000064 |
| `agent01_1c_final_block_plus_mlp` | 0.9460 | 0.9459 | 0.000032 |
| `agent02_raw_observation_cnn` | 0.9686 | 0.9686 | 0.000000 |
| `phase11_head_unchanged_reference` | 0.9834 | 0.9834 | 0.000000 |

the unchanged Phase 11 head reproduces exactly; the trained checkpoints reproduce to the scale of the run-to-run drift their own agents measured, two orders of magnitude below the gaps this leaderboard turns on. The recomputed values are used for the paired bootstraps so every candidate in a comparison comes from one scoring pass; the quoted leaderboard rows are left exactly as Agents 1 and 2 reported them.

### Paired game bootstraps

A negative difference means Agent 3 has the lower cross-entropy.

| comparison | mean ΔCE | 95% CI | distinguishable |
| --- | ---: | --- | --- |
| agent03_c1_feature_cnn vs agent01_1b_attached_mlp_head | +0.0284 | [0.0243, 0.0324] | yes |
| agent03_c1_feature_cnn vs agent01_1c_final_block_plus_mlp | +0.0361 | [0.0321, 0.0400] | yes |
| agent03_c1_feature_cnn vs agent02_raw_observation_cnn | -0.0137 | [-0.0185, -0.0093] | yes |
| agent03_c1_feature_cnn vs phase11_head_unchanged_reference | -0.0461 | [-0.0572, -0.0355] | yes |

### Per-stratum R_CE

| candidate | phase9_selfplay | strategic_rule | tactical_rule | scout_rush |
| --- | ---: | ---: | ---: | ---: |
| `agent01_1c_final_block_plus_mlp` | 0.9317 | 0.9557 | 0.9523 | 0.9459 |
| `agent01_1b_attached_mlp_head` | 0.9350 | 0.9596 | 0.9556 | 0.9497 |
| `agent01_1a_existing_linear_head` | 0.9379 | 0.9628 | 0.9582 | 0.9552 |
| `agent03_c1_feature_cnn` | 0.9501 | 0.9712 | 0.9679 | 0.9619 |
| `agent02_raw_observation_cnn` | 0.9560 | 0.9760 | 0.9717 | 0.9721 |
| `phase11_head_unchanged_reference` | 0.9614 | 0.9883 | 0.9859 | 0.9997 |


## 5. The comparison `03_AGENT_3` asks for

```text
Agent 2 raw-CNN R_CE          0.9686
Agent 3 C1-feature-CNN R_CE   0.9624
difference                    -0.0062
```

Same tower, same optimizer, same corpus, same probe schedule, same
checkpoint-selection rule, same development pieces. The **only** difference
is what goes in the bottom: 127 raw observation planes for Agent 2, the
frozen C1 field's 128 channels for Agent 3. The paired game bootstrap of the
cross-entropy difference is -0.0137
[-0.0185, -0.0093] over 497 games, so
this is not noise.

That gap of 0.0062 is
just outside the
sprint's 0.005 equivalence band.

**Reading, per `03_AGENT_3`'s own interpretation rule:**
Agent 3 is materially better than Agent 2: the frozen C1 representation is a better starting point for belief than a specialist encoder learned from this corpus.

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
| `agent01_1b_attached_mlp_head` | the same `encode` output, gathered at the piece's own square | 334,860 | 0.9495 |
| `agent03_c1_feature_cnn` | the same `encode` output, all 100 squares | 3,898,444 | 0.9624 |

Same tensor, 11x the parameters, and
spatial context the per-piece head cannot express — and the belief comes out
0.0129 `R_CE` **worse** for it. So the answer to the second half is **no, not in the capacity
sense**. The Phase 11 head's problem was that it was never optimized for
belief on its own — Agent 1 moved 0.9834
-> 0.9531
without adding a single parameter — and once that is fixed, more extraction
capacity on this corpus costs rather than buys.

## 6. Architecture, or corpus?

Agent 2 handed forward a specific warning: 3.9M parameters against
26,898 positions drawn from 2,048 games, with hidden
ranks constant inside a game, memorizes almost immediately. It did, and this
candidate does the same:

| quantity | value |
| --- | ---: |
| best development checkpoint | epoch 2 (1.35 epochs), step 143 of 106/epoch |
| development probes | 64 |
| training CE, first -> last epoch | 2.1113 -> 0.7359 |
| development CE, first -> last epoch | 2.1365 -> 3.3337 |
| development CE at the kept checkpoint | 2.1123 |
| development CE rose after the best | True |

Training cross-entropy fell 2.1113 ->
0.7359 while development cross-entropy rose
2.1365 -> 3.3337,
and the patience rule stopped the run after 7 of
60 scheduled epochs.

Agent 2 answered the "architecture or corpus" question for this tower by
retraining it on halves of the corpus — best development `R_CE` improved
monotonically 0.9907 -> 0.9775 -> 0.9686 from 512 to 2,048 games and had not
flattened. **That diagnostic was not repeated here.** It is the same tower
under the same optimizer on the same corpus, Agent 2's finding is on disk,
and `03_AGENT_3` says to use earlier reports rather than rerun prior
candidates. What Agent 3 adds is that swapping the *input representation* for
a better one moves the number by
0.0062 while quadrupling the
corpus moved it by 0.0221 — the corpus is still the larger lever.

**Agent 2's measurement fix was inherited whole.** The kept checkpoint here is
step 143 of 106 per epoch, so
1.35 epochs in, found by probing
development cross-entropy 8 times
per epoch. At epoch granularity this candidate would have been reported at
`R_CE` 0.9734
instead of 0.9624.

## 7. What was trained, and how

| field | value |
| --- | --- |
| trainable | the whole 3,898,444-parameter belief CNN, from scratch |
| frozen | all 863,959 C1 parameters — not called during training at all |
| optimizer | adamw + cosine |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| gradient clip | 1.0 |
| batch | 256 positions |
| epochs | 7 of 60 |
| stopped | patience |
| development probes per epoch | 8 |
| best checkpoint | step 143 of 106/epoch — 1.35 epochs, of 64 probes |
| device | mps |

The configuration is Agent 2's `run1_declared` verbatim, for the same reason
the architecture is: Agent 2's tower verbatim (width 160, 8 residual 3x3 blocks, 1x1 read-out 128) so the only difference between the two candidates is the input representation.
Agent 2 additionally ran one corrective regularized configuration and found it
moved the ceiling by 0.0005 `R_CE`; Agent 3 declared **one** configuration, so
this report has no deviation to disclose on that front. It was then trained a
second time under that same configuration and seed as a spread diagnostic —
`R_CE` 0.962373 against
0.962373, a difference of
0.00e+00, stopping at the
same step 143 — and that repeat wrote no checkpoint and
is not the reported candidate.

The loss is supervised hidden-rank cross-entropy over the
817,255 hidden pieces of the 26,898
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

## 8. Cost

| item | value |
| --- | ---: |
| feature cache | 0.6 s, 1471 MB |
| training wall clock | 119 s |
| time to best checkpoint | 23 s |
| trainable parameters | 3,898,444 |
| checkpoint | 15.7 MB |
| peak memory | 5.05 GB |

Inference has two honest readings and both are reported, because quoting only
one would be a rhetorical choice:

| path | one position | batched, per position | per hidden piece, batched |
| --- | ---: | ---: | ---: |
| belief CNN alone (field already computed) | 2.96 ms | 1.024 ms | 33.47 µs |
| frozen C1 encode alone | 1.44 ms | 0.249 ms | 8.13 µs |
| **end to end, observation -> marginals** | **4.41 ms** | 1.273 ms | 41.60 µs |

The first row is what a search that is *already* running C1 for its policy
adds; the last is a belief query in isolation, and it is the row comparable to
Agent 2's 3.42 ms,
which has no C1 stage. Against Agent 1's winner — a head that rides on C1's
existing encode for
0.437 µs
per piece — this candidate is a second network with its own checkpoint
whichever row is used.

Peak memory is the peak process RSS of the training stage: the materialized
1.4 GB C1 field tensor, the model and the metric arrays, not the model alone.

## 9. Is this preferable to what already exists?

**No. Agent 3 is 0.0164
`R_CE` *worse* than `agent01_1c_final_block_plus_mlp` (0.9460 against
0.9624) and far more expensive, so it loses on both axes of the
sprint's rule.** It is, however, the better of the two 3.9M spatial
specialists, and that is the finding it contributes.

How the engineering-winner rule applies:

- leader by `R_CE`: `agent01_1c_final_block_plus_mlp` (0.9460);
- inside the 0.005 equivalence band of the leader: `agent01_1c_final_block_plus_mlp`, `agent01_1b_attached_mlp_head`;
- Scout-rush / generalization: 0.9619 for Agent 3, against 0.9459 for `agent01_1c_final_block_plus_mlp` and 0.9721 for `agent02_raw_observation_cnn`;
- search-integration complexity: Agent 3 rides on C1's existing encode: a search already running C1 for its policy pays only the specialist's 2.96 ms per position, against 4.41 ms for a belief query in isolation. It is still a second network with its own checkpoint, unlike Agent 1's head.

The band is measured against the leader only, never as a chain of pairwise
comparisons — the convention Agents 1 and 2 recorded.

## 10. Required interface

```text
predict_marginals(public_state)      -> {piece_slot: 12-way rank probabilities}
sample_worlds(public_state, n, seed) -> complete legal hidden armies
```

| candidate | positions | worlds | marginals valid | seed-deterministic | worlds legal |
| --- | ---: | ---: | --- | --- | --- |
| `agent03_c1_feature_cnn` | 16 | 128 | yes | yes | yes |

Every world was drawn through **`stratego.evaluation.phase11_sampler (accepted, unmodified)`**. Agent 3
supplies marginals and nothing else, through the same `Phase11BPublicState`
Agent 1 defined — a container with exactly two public fields and no field a
true rank could arrive in. The Agent 3 adapter *subclasses* Agent 1's
interface: the encoder slot holds the frozen C1 and the head slot holds the
belief CNN, so `sample_worlds` is inherited code, not a fork, and the live
path recomputes the same `encode` output the cache was built from.

## 11. Caveats a reader should carry forward

- **This is a development-set number.** There is no sealed bank behind it and
  no scientific claim attached to it. The development set is an engineering
  comparison set, exactly as the sprint defines it.
- **The checkpoint was trained on `mps` and scored on CPU.**
  The two backends agree to
  2.19e-09 `R_CE`, so the
  headline number does not depend on which one produced it. Neither backend's
  float32 reductions are bit-reproducible, so this was measured rather than
  assumed. The *feature cache* was built on CPU precisely so that the seam
  itself is not a source of that difference.
- **The kept checkpoint is 1.35 epochs
  into a 7-epoch run** (60 scheduled;
  stopped by `patience`). Development `R_CE` ended
  0.5564 worse than at the best
  probe, so the curve had turned well before the run stopped.
- **The headline `R_CE` uses the accepted raw-softmax convention** — no
  masking, no epsilon, full simplex — because that is how the Phase 11 head
  was measured and how the accepted sampler consumes a belief. Renormalizing
  onto the publicly legal support is a diagnostic only
  (0.9577 against 0.9624 raw).
- **The reference row is not the Phase 11 sealed-test result.** It is the
  unchanged Phase 11 head scored on *these* fresh positions. Phase 11's sealed
  test remains what it was, and its bank remains spent.
- **One configuration, and it was repeated.** `03_AGENT_3` asks for one
  architecture and one comparison, so unlike Agent 2 there is no second
  optimization configuration here. Run-to-run spread was measured the way
  Agent 1 measured it — by training the identical configuration again — and
  it came out at
  0.00e+00 `R_CE`
  (0.962373 against
  0.962373), with the two runs' epoch-boundary
  training losses agreeing to
  0.00e+00. That
  repeat is a diagnostic: it wrote no checkpoint and the leaderboard is
  identical without it. What it does **not** bound is seed sensitivity — the
  seed was deliberately held fixed, so this measures backend
  nondeterminism, which for this model on this backend turns out to be
  none.
- **"C1 retains the information" is a statement about this corpus.** Agent 2
  showed the same tower is corpus-limited at 2,048 games. A raw-observation
  encoder is not refuted by losing here; it is untested at a corpus size that
  would give it a chance, and the same caveat applies to this candidate.

## 12. What Agent 3 touched

The common corpus was reused **byte-for-byte**: both splits' file digests and
the whole-corpus digest `903bf10a3e34cfc0…`
were recomputed from disk and matched against the values Agent 1 and Agent 2
recorded. Nothing was regenerated.

| statement | value |
| --- | --- |
| corpus regenerated | `False` |
| C1 modified | `False` |
| Agent 1 artifacts modified | `False` |
| Agent 2 artifacts modified | `False` |
| artifacts unchanged since Agent 2 | `True` |
| `phase11_test_bank_v1` opened | `False` |

Agent 3 added `stratego/belief/phase11b/feature_seam.py` and
`feature_cnn.py`, the harness `scripts/run_phase11b_agent03.py`, the
renderer `scripts/_phase11b_agent03_report.py`, its tests, two field caches under `checkpoints/phase11b/` and one
checkpoint. It edited no existing module: the frozen-prefix call, the
trainer, the metrics, the sampler adapter and the corpus loader are all
imported from Agent 1's and Agent 2's files unchanged, which is why all
33 preserved digests still match.

Repository suite after Agent 3: **5962 passed, 3 skipped in 348.71s (0:05:48)** (`python -m pytest tests -q`).

## 13. Handoff to Agent 4

Agent 3 does not begin Agent 4's experiment and does not recommend for or
against running it. What Agent 3 measured that Agent 4 should carry:

1. **The frozen C1 field beats raw pixels at equal capacity, by
   0.0062 `R_CE`, and the
   difference is distinguishable.** Agent 4's hybrid feeds both into one
   specialist. This result says the C1 half is the more informative of the
   two inputs it will be given, not that the raw half is worthless — Agent 2's
   candidate is still
   0.0148
   better than the unchanged Phase 11 head.
2. **Both spatial specialists lose to a 334,860-parameter
   head attached to the frozen C1 feature.** On this corpus the binding
   constraint is supervision, not extraction capacity. A hybrid at 3-5M parameters should
   expect the same régime and should probe development loss several times per
   epoch — this run's optimum arrived 1.35
   epochs in.
3. **The seam and its cache are reusable.** `feature_seam.py` builds the
   `[N, 100, 128]` field for either split in seconds, verifies it against the
   public observations, and hashes it; Agent 4 can concatenate it with the raw
   observation without re-deriving anything.
4. **`train_raw_cnn` is representation-agnostic.** Agent 3 reused it by
   passing a split view with a different input array. A hybrid can do the same
   with a stacked input, and inheriting the trainer is what makes the three
   candidates' numbers comparable.

## 14. Stop condition

Agent 3 trained one architecture and stopped. Agent 4's experiment was not begun. Phase 11 remains FAIL, phase11_test_bank_v1 remains spent and unopened, and nothing here authorizes Phase 12.
