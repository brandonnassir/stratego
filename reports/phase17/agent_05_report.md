# Phase 17 — Agent 5

## Post-training local evaluation and checkpoint shortlist

**Run:** `RUN-2026-B` · **Recipe:** `phase17_simple_paper_tandem_v1` · **Written:** 2026-08-30

**Handoff:** `reports/phase17/phase17_local_eval_handoff_v1.json`
(`handoff_digest 244d8079…`)

---

## 1. Verdict

All 25 frozen paired EMA candidates were evaluated locally on both lanes. Zero
refusals, zero retries, zero failures. Every identity reproduced from bytes before
any weight was loaded.

**The run did not improve.**

| Question | Answer |
|---|---|
| Hour 6 → 12, move-only lane | **Degraded** — slope −0.0115 EWR/h, t = −2.97 |
| Hour 6 → 12, joint lane | **Flat** — slope +0.0003 EWR/h, t = 0.04 |
| Hour 6 → 12, mean of the two | Flat to slightly down — slope −0.0056 EWR/h, t = −1.30 |
| Did mean improvement hide a worst-stratum regression? | **No** — there was no mean improvement to hide one |
| Did anything beat the hour-0 start checkpoint? | **No candidate, on the move-only lane. 0 of 24.** |

The maximum of the move-only curve is hour 0 — the accepted Phase 9 C1 weights the
run started from. Twelve active hours of tandem self-play moved that lane down by
0.0625 EWR (paired, t = −1.74) and left the joint lane statistically where it began.

**No checkpoint is promoted. The operator makes that decision.**

---

## 2. Entry conditions, independently re-verified

I did not take Agent 7's closeout on trust; every claim below was recomputed here.

| Check | Result |
|---|---|
| Trainer process remaining | none (`ps` clean) |
| Ledger digest | `3961a83e…` recomputed = ledger claim = closeout claim |
| Ledger file sha256 | `bb30b6af…` matches closeout |
| All 25 candidate file sha256 | **25/25 reproduce the ledger**, byte-identical |
| Ordinals 0–24 | present, contiguous, no gaps |
| Telemetry file sha256 | matches closeout (`0061e75c…`) — unmodified by this work |
| `joint_00535.pt` sha256 | matches closeout — hashed only, **never loaded** |

Per D11, nothing remote was used, attempted, or required. The retired
transport/MacBook modules were left on disk untouched (instruction §1 preserves
them) and are not on the active path; `scripts/run_phase17_eval.py evaluate` is the
only entry point that ran.

---

## 3. One-candidate validation — `RUN-2026-B-cand-000` (h0)

Validated end to end, **result retained**, and not replayed for the batch (the
idempotent path returned it in 1 s).

| Required check | Evidence |
|---|---|
| Candidate/source/config/pack/evaluator identities | recomputed from bytes; move EMA `f1df694d…` = the canonical Phase 9 C1 digest, setup EMA `9dc73986…` = the seed-17 fresh init, both matching the ledger |
| Both lane completions | `move_only` 120 games, `joint_move_setup` 120 games |
| Deterministic case/seed accounting | identical `result_digest` on re-run **and under a different worker count (8 vs 3)**; joint-lane root and per-token seeds cross-checked against the pack on every case |
| Atomic result and receipt writing | `.partial` staging, `os.replace`; no residue anywhere |
| Receipt re-verification and ledger ingestion | `eligible: true`, zero mismatches against 12 ledger-bound fields |

Five refusal paths were exercised and all fired correctly:

- stale / mis-attributed candidate id;
- duplicate-conflicting result for one identity;
- wrong published file sha256 (partial or corrupt bundle);
- unbound benchmark pack digest;
- **the forbidden terminal checkpoint `joint_00535.pt`** — structurally impossible to
  evaluate: it carries schema `phase17_joint_checkpoint_v2` and the evaluator reads
  `phase17_paired_export_v1` only. Attempted deliberately, refused, recorded.

No candidate weights, cases, seeds, scoring, or training artifacts were modified.
The evaluator source digest is `0aa0b23a…` and is **identical across all 25
receipts** — it was not edited at any point during or after the batch.

### An evaluator defect found and *not* fixed

`role_evaluate`'s failure path writes a refusal receipt to
`<candidate_id>.result.json`. `existing_result()` then finds that file on any later
attempt, reads its `bundle_digest` as `None`, and refuses the candidate as
duplicate-conflicting — **permanently**. A candidate that once failed for a transient
reason (a mistyped `--expect-*` flag, say) can never be re-evaluated without deleting
the refusal file by hand, which contradicts the contract's requirement that failures
"can be repaired without changing the preserved training run."

I did not fix it. Fixing it edits `evaluator.py`, which changes
`evaluator_source_digest`, which would split the evaluator identity across the 25
receipts — cand-000's was already written. The defect did not affect this batch:
every candidate was evaluated exactly once into a clean directory. The fix is to
have `existing_result` ignore receipts whose `status` is `refused`, or to write
refusals under a distinct suffix.

---

## 4. Method — why the numbers below are paired

A 120-game lane carries a binomial SE near 0.04 EWR, and Agent 1 measured that **25
candidates drawn from pure noise spread 0.1435 EWR** on exactly this lane size. The
observed spreads here are 0.1208 (move-only, *under* the noise reference) and 0.2125
(joint, well above it). **No isolated peak in either curve is evidence of anything.**

Every candidate plays the same 120 cases against the same opponents from the same
setups, so per-board paired differences cancel the case-to-case variance that
dominates that spread. To get those rows I replayed all 25 candidates through the
evaluator's own `_worker_init`/`_play` and proved the replay is the scored games:
**all 25 reproduced their receipt's `result_digest` exactly.**

---

## 5. The learning curve

Full table in `agent_05_learning_curve.csv` (25 rows × 70 columns: identities, both
lanes, all strata, worst stratum, rolling medians, move/setup KL and entropy from the
frozen telemetry, setup diversity, integrity flags, receipt eligibility).

```text
  h    it     MO      JT    mean   worst   r3MO    r3JT   H_move  H_setup
 0.0    0   .7542   .6625   .7083   .3333    -       -      -       -
 1.0   46   .7167   .6000   .6583   .0833   .7250   .6542  1.041   1.758
 2.0   85   .6833   .5417   .6125   .2500   .6833   .5750  0.780   1.727
 3.0  126   .6750   .6292   .6521   .2500   .6833   .6292  0.646   1.702
 4.0  173   .6958   .7042   .7000   .3750   .6958   .6292  0.626   1.653
 5.0  217   .6583   .6375   .6479   .1250   .6792   .6375  0.522   1.644
 6.0  261   .6875   .6333   .6604   .1250   .6750   .6333  0.513   1.647   <- window opens
 7.0  302   .7375   .7083   .7229   .4167   .7250   .6333  0.500   1.592
 8.0  344   .6875   .6125   .6500   .2083   .7167   .6125  0.462   1.535
 9.0  382   .7417   .6375   .6896   .2917   .6917   .6375  0.428   1.461
10.0  425   .7000   .5750   .6375   .2083   .7000   .5750  0.449   1.414
11.0  468   .6667   .6292   .6479   .4167   .6667   .6292  0.417   1.429
12.0  510   .6708   .6875   .6792   .2917   .6667   .6292  0.400   1.382
```

Whole-run trend: move-only **−0.0035 EWR/h (t = −2.26)**, joint +0.0014 (t = 0.48),
mean −0.0010 (t = −0.59).

Hour 6–12 block against hour 0–5.5 block, paired per board: move-only **−0.0052**
(SE 0.0160), joint **+0.0143** (SE 0.0198). Neither block moved.

Hour 6–12 block against the h0 start, paired per board: move-only **−0.0625**
(SE 0.0359, t = −1.74), joint **−0.0388** (SE 0.0432, t = −0.90).

Per-candidate paired deltas against h0 on the move-only lane are negative for
**all 24** trained candidates. On the joint lane, three land above h0 (h4.0 +0.042,
h7.0 +0.046, h12.0 +0.025), none by more than one standard error.

---

## 6. Hour 6 → hour 12, answered directly

**Move-only degraded.** Within the window the slope is −0.0115 EWR/h with SE 0.0039
(t = −2.97) — about −0.069 EWR across the six hours. This is the one direction in the
whole analysis that clears conventional significance.

**Joint flattened.** Slope +0.0003 EWR/h (t = 0.04). The window ranges 0.5333 to
0.7083 with no direction whatsoever; that 0.175 range on 13 points is the noise
signature, not a trajectory.

**The mean is flat to slightly down** (−0.0056 EWR/h, t = −1.30), because the
degrading move lane and the flat joint lane roughly cancel.

Nothing here supports the phase's motivating hypothesis. The engineering question was
whether the paired system keeps improving from hour 6 through hour 12; it did not
improve from hour 0 either.

---

## 7. Worst stratum — no hidden regression, but a real redistribution

Mean EWR moved **+0.0046** from the early block to the late block. The worst stratum
moved **+0.0430** and the robust 36-game neural-opponent block **+0.0101**. Both moved
*up*. So the answer to "did mean improvement hide a worst-stratum regression" is no,
in the strict sense that there was no mean improvement, and in the useful sense that
the worst stratum did not regress either.

Two caveats matter more than that headline:

**A single candidate's worst stratum is mostly noise.** The worst bucket is a
by-opponent bucket at every one of the 25 candidates, and those hold 12 games each —
SE near 0.14 EWR. That is why the shortlist ranks on the 36-game neural block instead.
The worst stratum is a neural opponent (p18, p24, phase9_anchor) in 49 of the 50
lane-candidate entries — the sole exception is cand-000's joint lane, where
`strategic_rule_based` scores 0.3333 against the untrained setup network. It is never
a setup-source or colour stratum.

**The flat mean does hide a redistribution.** On the move-only lane the
neural-opponent block rose **+0.0554** (SE 0.0289, t = 1.92) while the non-neural block
fell **−0.0312** (SE 0.0186, t = −1.68). Per opponent, the late block gained most
against p24 (+0.121) and lost most against stress_scout_rush (−0.137) and
stress_information_miser (−0.072). The move policy did change; it traded performance
against handcrafted opponents for performance against neural ones, netting slightly
negative. That is consistent with a policy specializing on its own self-play
distribution.

---

## 8. The setup half, isolated

The two lanes fix the opponent, the opponent's setup, the colour and the match seed,
and both use the **same** candidate move weights. Only the player's setup differs —
network-generated in the joint lane, fixed accepted library in the move-only lane. The
per-board difference is therefore a direct read on the setup network, free of any
move-policy confound.

| Reading | Value |
|---|---|
| Gap at h0 (untrained, seed-17 init) | −0.0917 |
| Gap pooled over h6–12 | **−0.0679** (paired SE 0.0233, **t = −2.91**) |
| Gap at h12 | +0.0167 |
| Difference-in-differences (h6–12 gap − h0 gap) | **+0.0237** (SE 0.0545, t = **+0.44**) |
| Gap trend, whole run | +0.0049/h (t = 1.62) |

After 12 active hours, 37,875 setup optimizer steps and 467,872 consumed episodes,
**the setup network's boards are still measurably worse than the fixed accepted
library**, and it has **not** measurably improved on its own random initialization.
The trend is mildly positive and does not reach significance. The +0.0167 at h12 is a
single candidate, not a crossover.

Diversity did not collapse in the evaluation lane: every candidate, h0 through h12,
produced **120 distinct canonical setup fingerprints for the 120 joint cases**.
Training-side setup empirical entropy fell monotonically 1.769 → 1.382 nats (−22%) and
flag effective support bounced 12.99–31.82 with no trend. Concentration is real; it has
not collapsed the sampled distribution.

---

## 9. Integrity

Across every candidate's iteration in the frozen telemetry: **0** stop predicates,
**0** warnings, **0** setup legality failures, **0** orientation failures, **0** setup
fallback attempts, **0** non-finite gradients. Every generated joint setup passed the
accepted orientation gate at evaluation time (a failure would have been fatal, never
repaired). All 25 receipts re-verify as eligible with zero mismatches.

---

## 10. Shortlist and recommendation

Pareto front over the three measurable axes of contract §14 — mean composite-pack EWR,
worst stratum EWR, move-only non-regression — among the 13 eligible hour 6–12
candidates. The front has **two** members. Late rolling direction, setup stability and
training stability are applied as ranking criteria after the front, not as axes.

| | Candidate | Hour | mean | move-only | joint | worst | neural block | smoothed mean |
|---|---|---|---|---|---|---|---|---|
| **Recommended** | `RUN-2026-B-cand-014` | 7.0 | 0.7229 | 0.7375 | 0.7083 | 0.4167 | 0.5000 | 0.6792 |
| Alternative 1 | `RUN-2026-B-cand-018` | 9.0 | 0.6896 | 0.7417 | 0.6375 | 0.2917 | 0.3333 | 0.6708 |
| Alternative 2 | `RUN-2026-B-cand-022` | 11.0 | 0.6479 | 0.6667 | 0.6292 | 0.4167 | 0.4722 | 0.6417 |

**cand-014 (h7.0)** leads the window on mean EWR, worst stratum and the neural block
simultaneously, and is the only candidate not dominated on all three axes.
**It is also the exact shape the selection rule warns about**: its joint EWR is the
global maximum of the joint lane across all 25 candidates, its immediate neighbours
score 0.6333 and 0.5333, and 3-point centered smoothing pulls its mean from 0.7229 back
to 0.6792. Its paired margin over h0 is inside noise.

**cand-018 (h9.0)** is the second front member and the strongest move-only candidate of
the window (0.7417), with the smallest move-only paired regression against h0
(−0.0125).

**cand-022 (h11.0)** is *off* the front, listed on a declared late-window robustness
axis: worst stratum 0.4167 (tied best) and neural block 0.4722 (second best) at hour 11
rather than at an early peak. It is dominated on mean and move-only EWR.

The final timestamp did not win automatically — cand-024 (h12.0) is on neither the
front nor the shortlist.

**None of the three is distinguishable from the others, or from the window as a whole,
at this sample size.** Treat the ordering as a ranking under a declared rule, not a
measured difference.

### What conditions the whole recommendation

This evaluation does **not** support promoting any Phase 17 candidate over the
accepted Phase 9 C1 move weights. Those weights are what the h0 candidate carries
unchanged, and they outscored every trained candidate on the move-only lane. The
shortlist answers "which is the best member of the hour 6–12 window" — the question the
selection rule asks — not "is any of them worth promoting", which on this evidence is
no. The operator holds that decision.

---

## 11. Established and unknown

**Established**

- All 25 frozen paired EMA candidates evaluated locally, both lanes, zero refusals,
  zero retries, zero failures.
- Every candidate's bytes, both EMA state digests, config, source, pack and evaluator
  identities reproduced before any weight was loaded.
- The evaluation is bit-deterministic: identical result digests on re-run, under a
  different worker count, and on a full per-board replay of all 25.
- Hour 6 → 12 shows a degrading move-only lane and a flat joint lane. No lane improved.
- Nothing beat the h0 start checkpoint on the move-only lane; 0 of 24.
- The setup network remains measurably worse than the fixed accepted library and shows
  no measurable improvement over its own random initialization.
- Only EMA candidate exports were evaluated. `joint_00535.pt`, raw weights, and
  post-h12 state were not, and the schema check makes the first structurally
  impossible.

**Unknown**

- **Why** the move-only lane declines. This measures outcome, not cause. The KL/entropy
  anneal, the game-length collapse (270 → 145 plies), and genuine overfitting to the
  self-play distribution are all consistent with what is seen here, and this evaluation
  cannot separate them.
- Whether the 130 unrun iterations of the frozen 640-iteration horizon would have
  changed the direction. The move LR and entropy schedules never completed their anneal.
- Whether a longer or differently-shaped setup schedule would close the gap to the fixed
  library. Twelve hours did not, and the trend is not significant in either direction.
- How any of these EWRs relate to human play. The ten benchmark opponents are
  evaluation instruments, not a human distribution.

---

## 12. Artifacts

```text
reports/phase17/phase17_local_eval_handoff_v1.json    binding handoff
reports/phase17/agent_05_report.md                    this report
reports/phase17/agent_05_local_environment.json       host, evaluator, pack, opponents
reports/phase17/agent_05_candidate_receipts.jsonl     25 receipts, each re-verified
reports/phase17/agent_05_learning_curve.csv           25 x 70 curve
reports/phase17/agent_05_checkpoint_shortlist.json    Pareto front and recommendation
reports/phase17/local_eval/results/*.result.json      25 evaluator receipts
reports/phase17/local_eval/rows/*.rows.json           per-board rows (digest-proven)
scripts/phase17_local_eval_capture.py                 per-board replay + digest proof
scripts/phase17_local_eval_analysis.py                curve, statistics, deliverables
```

Total evaluation runtime 315 s for 6,000 games. Nothing under
`checkpoints/phase17/`, the telemetry, or any accepted checkpoint was written to.
