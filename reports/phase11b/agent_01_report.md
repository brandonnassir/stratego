# Phase 11B — Agent 1: Attached Belief-Head Engineering

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

## 0. What Agent 1 found

Agent 1's question was whether the Phase 11 weakness was **mainly insufficient**
**dedicated belief optimization** or **an undersized belief output head**. On the
common Phase 11B development set the answer is unambiguous:

1. **It was mainly the optimization.** Retraining the *existing* 128→12 layer —
   the same 1,548 parameters, the same frozen C1 features — moves
   `R_CE` from **0.9834** to **0.9531**, a gain of
   **0.0303**. Nothing about the architecture changed; only the objective did.
2. **Head capacity is worth much less.** Going from 1,548 to
   334,860 trained parameters — a 216× head — buys a further
   **0.0036**, about 8× smaller than the retraining's gain.
3. **The representation is now the binding constraint.** Unfreezing the last C1
   block (1C) buys another 0.0036 on top of the larger head — as much again as
   the larger head bought over the retrained linear layer, and the only change
   that reached past the frozen features. That is the strongest signal here for
   what Agents 2-5 should expect: the remaining headroom is in the
   representation, not in the head.

The largest single change is on the hardest stratum. The unchanged Phase 11 head
scored `R_CE` 0.9997 on Scout-rush — indistinguishable from simply counting
the remaining pieces. The three retrained candidates score 0.9459-0.9552 there.

None of this is a scientific claim, a repair of Phase 11, or evidence about
whether better beliefs win more games. It is an engineering measurement on one
fresh development set.

## 1. Starting state

The accepted Phase 9 checkpoint was opened **read-only** and exported to a
Phase 11B path. Its identity was re-derived from live bytes:

```text
sha256                dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
model state digest    f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
belief head digest    a9df48a1adcd29b1a46c42ff1e605ede485119a36c247f1ae74f249f6d6f1dc7
parameters            863,959
global optimizer step 47,086
```

15 preserved Phase 11 artifacts and accepted modules were digested; every
digest is in `agent_01_summary.json`. Phase 11B wrote to none of them. The
accepted Phase 11 sampler, baselines, public-state document, contract, seed
module and belief targets are **imported and unmodified**.

## 2. Common Phase 11B corpus

`phase11b_common_corpus_v1` — corpus digest

```text
903bf10a3e34cfc0d91ba4a22761864fb91dc0cb832e7f5a07fc2a40bd743cf6
```

| split | games | observer decisions | samples | hidden pieces | sampled/game | setup-library split |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| train | 2,048 | 190,901 | 26,898 | 817,255 | 16 | train |
| dev | 512 | 49,230 | 1,828 | 55,955 | 4 | validation |

512 training and 128 development games per behaviour stratum
(phase9_selfplay, strategic_rule, tactical_rule, scout_rush), balanced by
construction over the two setup sources and both observer colours: 16 cells,
cell-major, so balance is a property of the id space rather than of any draw.

The setup-source split is the **opponent's**, following the accepted Phase 11
convention: the observer always draws from the accepted P10-D production
source, because that is the seat a deployed system occupies, and the 50/50
`p10d` / `neutral_v1` variation is what the observer has to form beliefs
about.

The two splits draw from **disjoint setup-library splits** (`train` and
`validation`) as well as from disjoint seed streams under a Phase 11B-only
blake2b personalization, so a development game cannot share a base arrangement
or a match seed with a training game, and no Phase 11B stream can coincide with
a Phase 11 one.

Public inputs live in `public/`, privileged true ranks in `privileged/`, and the
loader returns labels only when a caller asks for them by name. Every sample's
observation was rebuilt on an independent engine replay and checked bit-for-bit
against the digest the public pass recorded **before** any label was read; the
hidden-piece set, the remaining inventory and the legal-rank masks were
re-derived from the public document on that replay too.


One property later agents should not assume away: the hidden-rank prior is
**not** the initial army distribution. A hidden piece is one nobody has
resolved, and ranks that reveal themselves early are under-represented among
them — `scout` is 16.6% of hidden pieces against 20.0% of an army,
while `bomb` is 16.2% against 15.0%.

95 of 2,560 games contributed no sample: they ended before the observer had an
eligible decision. That is the same environment property Phase 11 recorded, not
a generation fault.

## 3. Results on the common development set

| candidate | architecture | CE | R_CE | 95% CI | top-1 | trained params | train s | to best s | µs/piece |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `agent01_1c_final_block_plus_mlp` | c1_last_block + mlp(128->512->512->12, gelu) | 2.0763 | **0.9460** | [0.9434, 0.9488] | 0.2640 | 533,388 | 73.2 | 73.2 | 2.773 |
| `agent01_1b_attached_mlp_head` | mlp(128->512->512->12, gelu) | 2.0841 | **0.9495** | [0.9471, 0.9524] | 0.2603 | 334,860 | 12.9 | 6.5 | 0.437 |
| `agent01_1a_existing_linear_head` | linear(128->12) | 2.0920 | **0.9531** | [0.9508, 0.9558] | 0.2542 | 1,548 | 5.0 | 4.4 | 0.017 |
| `phase11_head_unchanged_reference` | linear(128->12) | 2.1584 | **0.9834** | [0.9785, 0.9884] | 0.2303 | 0 | 0.0 | 0.0 | — |

`remaining_count_belief_v1` — the `R_CE` denominator — scores CE **2.1949**
and top-1 0.2038 on these 55,955 hidden pieces. A flat 12-way vector
scores `R_CE` 1.1321, which is the uninformed floor.

The intervals above are marginal game bootstraps. Two candidates scored on the
*same* pieces are far more comparable than those intervals suggest, so the
paired game bootstrap of the CE difference is the honest test:

| comparison | mean ΔCE | 95% CI | distinguishable |
| --- | ---: | --- | --- |
| agent01_1a_existing_linear_head vs agent01_1b_attached_mlp_head | +0.0078 | [+0.0054, +0.0103] | yes |
| agent01_1a_existing_linear_head vs agent01_1c_final_block_plus_mlp | +0.0156 | [+0.0126, +0.0186] | yes |
| agent01_1b_attached_mlp_head vs agent01_1c_final_block_plus_mlp | +0.0078 | [+0.0063, +0.0093] | yes |

All three orderings are real, not noise — the gaps are simply small in `R_CE`.

### Per-stratum R_CE

| candidate | phase9_selfplay | strategic_rule | tactical_rule | scout_rush |
| --- | ---: | ---: | ---: | ---: |
| `agent01_1c_final_block_plus_mlp` | 0.9317 | 0.9557 | 0.9523 | 0.9459 |
| `agent01_1b_attached_mlp_head` | 0.9350 | 0.9596 | 0.9556 | 0.9497 |
| `agent01_1a_existing_linear_head` | 0.9379 | 0.9628 | 0.9582 | 0.9552 |
| `phase11_head_unchanged_reference` | 0.9614 | 0.9883 | 0.9859 | 0.9997 |

Every retrained candidate improves every stratum, and the ordering of strata is
unchanged: self-play positions stay the easiest, rule opponents the hardest.
Scout-rush moves from *worst* stratum for the old head to mid-table for all
three candidates.

## 4. What was trained, and how

One declared configuration per experiment. **No hyperparameter sweep and no
architecture search was run**; these are choices, not search results.

| experiment | trainable | frozen | optimizer | LR | batch | epochs | stopped |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| `agent01_1c_final_block_plus_mlp` | a copy of C1's last block + encoder norm, plus the 1B head | C1 except its last block | adamw + cosine | 0.001 (block 0.0001) | 256 | 12/12 | epochs_exhausted |
| `agent01_1b_attached_mlp_head` | a fresh 128→512→512→12 GELU MLP | all of C1 | adamw + cosine | 0.001 | 4,096 | 10/40 | patience |
| `agent01_1a_existing_linear_head` | the accepted 128→12 layer, from the accepted weights | all of C1 | adamw + cosine | 0.001 | 4,096 | 40/40 | epochs_exhausted |

The loss is supervised hidden-rank cross-entropy over hidden pieces and nothing
else: no policy term, no value term, no game outcome anywhere. 1A deliberately
starts from the **accepted** belief-head weights, because that is what makes its
gain a measurement of dedicated belief optimization at fixed capacity.

Because C1 is frozen for 1A and 1B, its representation is a constant of the
corpus and was cached once. Both experiments therefore see bit-identical
features, and any difference between them is the head and nothing else.

## 5. Cost

| item | wall clock |
| --- | ---: |
| corpus generation (2,560 games, played once, reused by Agents 2-5) | 478 s |
| frozen C1 `final` feature cache (both splits) | 7.5 s |
| frozen C1 `penultimate` feature cache (both splits) | 5.7 s |
| train `agent01_1c_final_block_plus_mlp` | 73.2 s |
| train `agent01_1b_attached_mlp_head` | 12.9 s |
| train `agent01_1a_existing_linear_head` | 5.0 s |

Peak memory: **7.25 GB**. That is the peak process RSS at the end of the training stage: the memory-mapped corpus, the frozen feature caches and the trained candidates, not the head alone,
and it is dominated by the caches rather than by any model — the largest single
object is the penultimate-layer cache 1C reads.

Every experiment ran on CPU / float32 at 8 torch threads — the accepted
evaluation backend. Nothing here needed a GPU: the whole Agent 1 experiment
programme after corpus generation costs under two minutes.

Repository suite after Agent 1: **5,871 passed, 3 skipped**
in 350 s (`python -m pytest tests -q`).

Retraining every candidate a second time under the identical configuration
moves `R_CE` by at most **0.00006** — an order of magnitude
smaller than the smallest gap the leaderboard reports, so the ordering is not
an artefact of run-to-run noise. Multi-threaded CPU float32 reductions are not
bit-reproducible, so this is measured rather than assumed.

| candidate | first pass | repeat pass | drift |
| --- | ---: | ---: | ---: |
| `agent01_1a_existing_linear_head` | 0.9531 | 0.9531 | 0.00000 |
| `agent01_1b_attached_mlp_head` | 0.9495 | 0.9495 | 0.00006 |
| `agent01_1c_final_block_plus_mlp` | 0.9460 | 0.9459 | 0.00003 |

## 6. Which of 1A / 1B / 1C was best

**Winner: `agent01_1b_attached_mlp_head`.**

Within 0.0036 R_CE of the leader agent01_1c_final_block_plus_mlp (0.9460) and materially cheaper and simpler (334,860 trained parameters against 533,388, and no accepted C1 weight retrained), so the Phase 11B engineering winner rule prefers it.

How the rule was applied:

- leader by `R_CE`: `agent01_1c_final_block_plus_mlp` (0.9460);
- inside the 0.005 equivalence band: `agent01_1c_final_block_plus_mlp`, `agent01_1b_attached_mlp_head`;
- materially worse and excluded: `agent01_1a_existing_linear_head`;
- Scout-rush check: `agent01_1c_final_block_plus_mlp` leads the winner by 0.0038 `R_CE` there, which is **not** a meaningful
  unusual-behaviour advantage at this band width;
- search-integration complexity: the winner attaches to the **unmodified**
  frozen C1 encoder, so the belief comes out of the same forward pass the
  policy already runs. 1C would require carrying a second, retrained copy of
  C1's last block alongside the accepted network.

The band is measured against the leader only, never as a chain of pairwise
comparisons: applied transitively, a run of sub-band steps would discard an
arbitrarily large real improvement, and the rule itself calls a gap above the
band material.

## 7. Required interface

```text
predict_marginals(public_state)      -> {piece_slot: 12-way rank probabilities}
sample_worlds(public_state, n, seed) -> complete legal hidden armies
```

`Phase11BPublicState` carries exactly the two public objects the accepted
`Phase11BeliefRequest` carries — the frozen public-state document and the
127-channel observation — so the interface has no field a true rank could
arrive in.

| candidate | positions | worlds | marginals valid | seed-deterministic | worlds legal |
| --- | ---: | ---: | --- | --- | --- |
| `agent01_1a_existing_linear_head` | 16 | 128 | yes | yes | yes |
| `agent01_1b_attached_mlp_head` | 16 | 128 | yes | yes | yes |
| `agent01_1c_final_block_plus_mlp` | 16 | 128 | yes | yes | yes |

Every world was drawn through **`stratego.evaluation.phase11_sampler`, the
accepted Phase 11 sampler, imported and unmodified** — the completion
feasibility guard, the `learned_probability × remaining_count` weighting, the
frozen categorical walk and the full validation stack are all the accepted code.
Phase 11B supplies marginals and nothing else.

## 8. Caveats a reader should carry forward

- **This is a development-set number.** There is no sealed bank behind it and
  no scientific claim attached to it. The development set is an engineering
  comparison set, exactly as the sprint defines it.
- **1C's best epoch was its last (12 of 12).** Its development curve was
  still improving monotonically at the end, but only just: the last 3 epochs
  moved `R_CE` by 0.00014 in total, against a cosine schedule that had already
  annealed. Read 0.9460 as a slight underestimate of what this
  configuration reaches, not as a large one. Agent 1 did not extend the run:
  re-choosing an epoch budget after seeing the result is the tuning the
  sprint forbids.
- **1B overfits early.** Its best development epoch was
  5 of 40 while its training loss kept falling. On a frozen
  128-wide feature the extra head capacity is quickly exhausted.
- **The headline `R_CE` uses the accepted raw-softmax convention** — no masking,
  no epsilon, full simplex — because that is how the Phase 11 head was measured
  and how the accepted sampler consumes a belief. Renormalizing each candidate
  onto the publicly legal support is reported as a diagnostic only
  (`agent01_1b_attached_mlp_head`: 0.9487 projected against 0.9495 raw); mixing it into the
  headline would compare a masked candidate against an unmasked reference.
- **The reference row is not the Phase 11 sealed-test result.** It is the
  unchanged Phase 11 head scored on *these* fresh positions. Phase 11's sealed
  test remains what it was, and its bank remains spent.

## 9. Handoff to Agents 2-5

The common corpus is built and immutable. Reuse it byte-for-byte:

```python
from stratego.belief.phase11b.storage import load_split, split_digest

data = load_split("data/phase11b/common_corpus_v1", "train", labels=True)
```

`split_digest` recomputes the per-file SHA-256s and `corpus_digest` the whole-
corpus identity; both are in `agent_01_summary.json`. Equality is the proof that
two experiments were scored on one corpus.

Score with `stratego.belief.phase11b.metrics.evaluate`, which computes the
`R_CE` denominator from the corpus's own stored public arrays, so every
candidate divides by the same number on the same pieces.

What Agent 1's result implies for the remaining experiments: the frozen C1
feature is close to exhausted. Replacing the linear head with a three-layer
MLP — 216x the parameters — bought 0.0036; letting a single encoder block move
bought as much again. The representation is where the remaining headroom is, so a
raw-observation CNN (Agent 2) is the most informative next experiment: it is
the cheapest way to learn a belief-specific representation instead of borrowing
the policy's.

Two practical notes for whoever runs it. First, the frozen-feature caching trick
does not transfer — a model that learns its own representation must see the
observations, so budget for real epochs over the 1.3 GB training tensor rather
than the seconds 1A and 1B took. Second, `phase9_selfplay` is consistently the
easiest stratum and the rule opponents the hardest; a candidate that wins only
on self-play positions has not generalized.

## 10. Stop condition

Agent 1 stops here. No other architecture was begun. Phase 11 remains `FAIL`,
`phase11_test_bank_v1` remains spent and unopened, and nothing in this report
authorizes Phase 12 or claims that Phase 11 has been repaired.
