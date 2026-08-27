# Phase 16 — Agent 3 report
## Training loop v2 and the 3×6-hour recipe shootout

_Contract digest `f68fb7f7c9300294…`._

Every EWR below names its pack. Cross-pack comparisons are forbidden in conclusions (overview §6); the packs are engineering instruments and no significance claim is made anywhere in this report.

## 1. What was built

`stratego/training/phase16/` — additive, importing the accepted objective and controller unmodified:

| module | what it is |
|---|---|
| `contract.py` | arm flags, seed streams, game ids, the inherited Phase 9 constants |
| `schedules.py` | the power-law LR and the annealed entropy coefficient |
| `targets.py` | window-edge targets and §2.2's invariant |
| `setups.py` | the `library` and `expanded` setup mixtures, orientation-gated |
| `population.py` | the persistent population and its opponent mixture |
| `snapshots.py` | behavior snapshots taken from live weights |
| `collector.py` | the window collector, harvesting rows at collection time |
| `trainer.py` | the PPO/KL update, the EMA, the per-window statistics |
| `checkpoint.py` | arm checkpoints, resume identity, the evaluation export |
| `runner.py` | one arm start-to-deadline, telemetry, hour exports |
| `seat.py` | the Agent 1 provider factory for a trained arm |
| `analysis.py` | §5's predeclared rules, applied mechanically |

### The three arms

| arm | LR | entropy | epochs | EMA | opponents | setups |
|---|---|---|---|---|---|---|
| **A** `a_control` | 7.50e-05 constant | 0.001 constant | 2 | no | `phase14_mixture` | `library` |
| **B** `b_damped` | 1.50e-04·(n/40)^−1.1 → 1.50e-05 | 0.005·n^−0.3 → 0.001 | 1 | yes | `pure_current` | `library` |
| **C** `c_damped_plus` | 1.50e-04·(n/40)^−1.1 → 1.50e-05 | 0.005·n^−0.3 → 0.001 | 1 | yes | `pure_current` | `expanded` |

Unchanged by explicit choice: PPO clip 0.2, λ_A 0.5 / λ_V 0.8, the top-quartile advantage filter with its 0.01 floor, value weight 0.5, belief-aux weight 0.25, the adaptive-β KL machinery and its thresholds, grad-norm 1.0, minibatch 512, float32, battleless-100 / absolute-4000, MPS with inference batch shape 64. All imported from the accepted Phase 9 contract, never restated.

## 1b. Deviations from the brief

**One amendment, to §2.3**, raised by this agent from the measured iteration rate and confirmed by the brief's author.

- *Defect.* the exponents were transcribed from a ~43,000-iteration run; at n_ref = 1 the power law floors at n = 9, so a six-hour arm at this machine's ~313 iterations would spend ~97% of itself at 1.5e-5 -- five times below the control -- and the shootout would measure a starved learning rate rather than a damped schedule.
- *Change.* lr(n) = clamp(lr_max * (n/n_ref)**-1.1, lr_min, lr_max) with n_ref = ceil(0.125 * N) = 40 for N = 313.
- *Unchanged.* lr_max, lr_min, the exponent, the entropy anneal, and arm A in its entirety.
- *Measured horizon.* 69.0 s per iteration on this machine, so N = 313 for a six-hour arm. `n_ref = 1` remains the code default and reproduces the brief exactly.
- *Entropy deliberately not re-horizoned.* 0.005 * n**-0.3 reaches the 0.001 terminal floor at n = 213, ~68% of a 313-iteration run: already a smooth decay across most of the run followed by the terminal value, which is what section 2.3 asks for. Re-horizoning it to n_ref_H = 13 as first proposed would start it at 0.0108 -- above the accepted Phase 9 level the section restores -- and end at 0.0018, never reaching the floor.

| arm | LR at n=1 | n=40 | n=150 | n=313 | floor first touched |
|---|---|---|---|---|---|
| `a_control` | 7.50e-05 | 7.50e-05 | 7.50e-05 | 7.50e-05 | — |
| `b_damped` | 1.50e-04 | 1.50e-04 | 3.50e-05 | 1.56e-05 | 325 |
| `c_damped_plus` | 1.50e-04 | 1.50e-04 | 3.50e-05 | 1.56e-05 | 325 |

The amendment does **not** equalise the arms and is not meant to. Arms B and C still take roughly half the control's Σ(lr·steps), because §2.4 gives them one epoch against A's two — an intended, briefed difference. What it removes is the unintended five-fold one.

## 2. Correctness gates (§3)

| gate | result | note |
|---|---|---|
| `collection_throughput` | PASS | the Phase 14 reference is measured here, on this machine, into a scratch root; the open Phase 14 run state is never read or written |
| `full_pytest` | PASS | run after the shootout, per section 8 |
| `smoke_run` | PASS |  |
| `window_edge_invariant` | PASS |  |

The window-edge invariant was checked on 24 synthetic games spanning at least 4 windows each. Largest advantage difference 0.000000000, largest W/D/L difference 0.000000000, against a tolerance of 1e-06.

The smoke run completed 18 windows in 20.48 minutes, checkpointed, resumed to the same model-state digest, and a CPU rerun of one window's update from identical inputs was bit-identical.

**Collection throughput.** the window collector advanced 2001.7 plies/s against the accepted Phase 14 collector's 1672.62 plies/s on the same machine (ratio 1.1967); the gate asks for within 2x


### Where the time actually goes

**Gate 3's collector ratio compares two *cold starts* -- 96 games all at ply 0 -- and the Phase 14 side of it carries model-load time inside a 16.6-second measurement. Against that run's own production telemetry (median 1784.2 plies/s) the window collector's steady-state 1610.2 plies/s is not faster. Collection is a wash; the gate is a floor check and it passes, but it is not the finding.**

| | Phase 16 arm B | Phase 14 production |
|---|---|---|
| collection, plies/s | 1610.2 | 1784.2 (median, range [1325.5, 1900.6]) |
| collection share of wall | 59% | 17% |
| training share of wall | 41% | 83% |
| optimizer steps / iteration | 126 | 1,712 → 5,247 |
| minutes / iteration | 1.144 | 20.6 → 170.1 |
| trained decisions / hour | 3,360,203 | 864,449 |

Phase 14 spent 83% of its 59.97 hours in the training phase, and that phase grew from 15.7 to 153.8 minutes an iteration while collection went only 4.8 to 16.3. The cause is iteration *sizing*: 2,048 whole games carry more data as games lengthen (265.0 to 733.6 plies, 1712 to 5247 optimizer steps an iteration). A fixed decision budget pins that quantity by construction, which is the whole of the design change.

_Phase 14 figures: the run's own recorded facts: 59.97 h, step 202504, 102 iterations at minibatch 512 x 2 epochs_
_Split derived read-only from that run's own per-iteration telemetry; no Phase 14 module was imported and nothing was written. derived from the run's own per-iteration telemetry: collection seconds = 2048 games / games_per_second, training seconds = the remainder of the iteration's elapsed time._

Caveats:
- the Phase 16 figures come from the 20-minute smoke run at the production window on an idle machine, not from a six-hour arm
- an iteration is not the same unit in the two systems (2,048 games against a 65,536-decision budget), so minutes/iteration is reported for scale and is not a speedup
- Phase 14's hours include its in-run candidate evaluations, which are charged to the training remainder rather than separated out

## 3. The three h-curves (§4)

_Instrument check: the 3 arms export their starting weights at h=0 and all carry model-state digest `622d9e6caa723c93…`, the accepted P24; their h=0 scores agree exactly. Arms exporting raw weights and arms exporting an EMA write the same tensors at update 0, so a disagreement here would be an evaluator fault rather than a recipe difference._

### Arm A — `a_control`

Benchmark pack `phase16_benchmark_v1` subset `quick60`; adversarial pack `phase16_adversarial_baseline_v1` stratum `adversarial_both`.

| h | iteration | step | benchmark EWR ± SE (decision) | adversarial EWR ± SE (decision) | benchmark full-pack ± SE |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.7583 ± 0.0553 (60 games) | 0.8646 ± 0.0349 (96 games) | 0.8125 ± 0.0356 (120 games) |
| 2 | 61 | 15546 | 0.7250 ± 0.0576 (60 games) | 0.7760 ± 0.0425 (96 games) | 0.7500 ± 0.0395 (120 games) |
| 4 | 112 | 28736 | 0.8083 ± 0.0508 (60 games) | 0.7969 ± 0.0411 (96 games) | 0.8250 ± 0.0347 (120 games) |
| 6 | 159 | 40846 | 0.7667 ± 0.0546 (60 games) | 0.8125 ± 0.0398 (96 games) | 0.7458 ± 0.0397 (120 games) |

Adversarial strata (only `adversarial_both` enters a decision rule):

| h | `adversarial_both` | `adversarial_opponent` | `benchmark_control` |
|---|---|---|---|
| 0 | 0.8646 ± 0.0349 | 0.8021 ± 0.0407 | 0.8229 ± 0.0390 |
| 2 | 0.7760 ± 0.0425 | 0.7396 ± 0.0448 | 0.8125 ± 0.0398 |
| 4 | 0.7969 ± 0.0411 | 0.7656 ± 0.0432 | 0.7656 ± 0.0432 |
| 6 | 0.8125 ± 0.0398 | 0.7083 ± 0.0464 | 0.8438 ± 0.0371 |

### Arm B — `b_damped`

Benchmark pack `phase16_benchmark_v1` subset `quick60`; adversarial pack `phase16_adversarial_baseline_v1` stratum `adversarial_both`.

| h | iteration | step | benchmark EWR ± SE (decision) | adversarial EWR ± SE (decision) | benchmark full-pack ± SE |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.7583 ± 0.0553 (60 games) | 0.8646 ± 0.0349 (96 games) | 0.8125 ± 0.0356 (120 games) |
| 2 | 101 | 12900 | 0.7167 ± 0.0582 (60 games) | 0.8021 ± 0.0407 (96 games) | 0.7583 ± 0.0391 (120 games) |
| 4 | 201 | 25797 | 0.8000 ± 0.0516 (60 games) | 0.7969 ± 0.0411 (96 games) | 0.8292 ± 0.0344 (120 games) |
| 6 | 306 | 39247 | 0.7833 ± 0.0532 (60 games) | 0.7812 ± 0.0422 (96 games) | 0.7917 ± 0.0371 (120 games) |

Adversarial strata (only `adversarial_both` enters a decision rule):

| h | `adversarial_both` | `adversarial_opponent` | `benchmark_control` |
|---|---|---|---|
| 0 | 0.8646 ± 0.0349 | 0.8021 ± 0.0407 | 0.8229 ± 0.0390 |
| 2 | 0.8021 ± 0.0407 | 0.7396 ± 0.0448 | 0.7917 ± 0.0414 |
| 4 | 0.7969 ± 0.0411 | 0.7188 ± 0.0459 | 0.8594 ± 0.0355 |
| 6 | 0.7812 ± 0.0422 | 0.7396 ± 0.0448 | 0.7969 ± 0.0411 |

### Arm C — `c_damped_plus`

Benchmark pack `phase16_benchmark_v1` subset `quick60`; adversarial pack `phase16_adversarial_baseline_v1` stratum `adversarial_both`.

| h | iteration | step | benchmark EWR ± SE (decision) | adversarial EWR ± SE (decision) | benchmark full-pack ± SE |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.7583 ± 0.0553 (60 games) | 0.8646 ± 0.0349 (96 games) | 0.8125 ± 0.0356 (120 games) |
| 2 | 99 | 12608 | 0.6333 ± 0.0622 (60 games) | 0.7240 ± 0.0456 (96 games) | 0.6500 ± 0.0435 (120 games) |
| 4 | 198 | 25388 | 0.7083 ± 0.0587 (60 games) | 0.7448 ± 0.0445 (96 games) | 0.7708 ± 0.0384 (120 games) |
| 6 | 298 | 38261 | 0.7333 ± 0.0571 (60 games) | 0.7083 ± 0.0464 (96 games) | 0.7625 ± 0.0388 (120 games) |

Adversarial strata (only `adversarial_both` enters a decision rule):

| h | `adversarial_both` | `adversarial_opponent` | `benchmark_control` |
|---|---|---|---|
| 0 | 0.8646 ± 0.0349 | 0.8021 ± 0.0407 | 0.8229 ± 0.0390 |
| 2 | 0.7240 ± 0.0456 | 0.6719 ± 0.0479 | 0.7292 ± 0.0454 |
| 4 | 0.7448 ± 0.0445 | 0.6615 ± 0.0483 | 0.7240 ± 0.0456 |
| 6 | 0.7083 ± 0.0464 | 0.7448 ± 0.0445 | 0.7760 ± 0.0425 |

## 4. Per-iteration diagnostics

| arm | iterations | steps | Σ(lr·steps) | iteration wall s (mean / p90 / CV) | plies/s | entropy first→last | KL mean/max | clip mean | retention |
|---|---|---|---|---|---|---|---|---|---|
| `a_control` | 159 | 40846 | 3.0634 | 135.7 / 155.4 / 0.125 | 1058.9 | 0.799→0.881 | 0.0224/0.0303 | 0.197 | 0.250 |
| `b_damped` | 306 | 39247 | 2.1695 | 70.4 / 74.9 / 0.058 | 1589.1 | 0.777→0.589 | 0.0125/0.0669 | 0.127 | 0.250 |
| `c_damped_plus` | 298 | 38261 | 2.1496 | 72.3 / 76.9 / 0.051 | 1516.7 | 0.643→0.715 | 0.0122/0.0676 | 0.134 | 0.249 |

The coefficient of variation of iteration wall-time is the number the window collector exists to hold down: Phase 14's iterations ran from 24 to 138 minutes because an iteration ended when its last game did.

**Σ(lr·steps) is the column to read before any conclusion about damping.** Three arms given equal wall-clock do not receive equal total step size, and the residual difference after the §1b amendment is the briefed one: §2.4 gives B and C a single epoch against A's two. Read the measured column rather than the schedules — a B-under-A result on similar Σ(lr·steps) is about the recipe, and one on very different Σ(lr·steps) is partly about how much training each arm received.

**Does the window budget actually pin the iteration?** Separately for each half:

| arm | rows/iter CV | collection s (first10 → last10, CV) | training s (first10 → last10, CV) |
|---|---|---|---|
| `a_control` | 0.057 | 53.7 → 99.3 (CV 0.197) | 52.2 → 57.0 (CV 0.061) |
| `b_damped` | 0.080 | 39.0 → 35.9 (CV 0.066) | 27.1 → 27.4 (CV 0.084) |
| `c_damped_plus` | 0.091 | 39.8 → 39.6 (CV 0.052) | 28.0 → 29.1 (CV 0.092) |

The budget pins what it was designed to pin: rows per iteration, and therefore training time. It does **not** pin collection time under `phase14_mixture`, and the reason is a batching interaction this phase did not anticipate. The accepted collector groups pending decisions by *acting checkpoint*, so a 96-game lockstep batch splits once per distinct opponent snapshot in play. Arm A archives its own weights every 30 minutes to reproduce Phase 14's historical mixture, so its pool grew from 2 members to 13 and its collection throughput fell monotonically with it — 1,996 plies/s at pool 2 to 830 at pool 13, a 2.4× slowdown that shows up as collection seconds, not as lost data. Arms B and C draw `pure_current` and hold exactly one snapshot, so they do not fragment. The fidelity was worth having and the cost is stated rather than hidden; a production run wanting both would need to cap the pool or pad across snapshots.

## 5. The decision the predeclared rules produced (§5)

```text
adopt_recipe    max(B, C) final-hour benchmark EWR >= A + 0.03
setups_causal   C adversarial-stratum EWR >= B + 0.03
plateau_check   report each arm's h4->h6 slope; a flat B/C with a passing h6 still adopts, but the report must say the plateau moved, not vanished
stop_rule       if neither B nor C clears adopt_recipe: STOP, write the report, hand back to the operator; no long run is authorized
```

> **Read this first.** one standard error on the 60-board decision instrument is 0.056 at the observed EWR; the predeclared margin is 0.03, i.e. 0.53 standard errors. A difference this rule calls decisive is well inside the noise of the instrument it reads, so the verdict below should be read as the mechanical output of a predeclared rule and not as evidence that one recipe is better.

> The same games scored over the full 120-board pack give a standard error of 0.040 (0.75 SE to the margin). It is reported in §3 as a secondary reading and enters no decision rule.

**Verdict: STOP.** neither B nor C clears adopt_recipe; section 5's stop_rule applies. No long run is authorized by this file. The report is written and the decision returns to the operator.

`adopt_recipe`: control (A) final-hour benchmark EWR 0.7667, threshold 0.7967 (+0.03); B 0.7833, C 0.7333. Clearing: none.

`setups_causal`: B adversarial 0.7812, C adversarial 0.7083, delta -0.0729 against a 0.03 margin — does not pass.

**Secondary instrument.** The same games scored over the full pack put the control at 0.7458 (threshold 0.7758), with B at 0.7917 and C at 0.7625 — it would also have adopted.

> the two instruments disagree on the same games. That is not a reason to prefer either: it is a direct measurement of the fact that a 0.03 margin sits inside the noise of both, and the decision stands on the instrument that was named in advance.

`plateau_check` — h4→h6 benchmark slope per arm:

| arm | h4 | h6 | Δ | per hour | flat |
|---|---|---|---|---|---|
| `a_control` | 0.8083 | 0.7667 | -0.0417 | -0.0208 | no |
| `b_damped` | 0.8000 | 0.7833 | -0.0167 | -0.0083 | yes |
| `c_damped_plus` | 0.7083 | 0.7333 | 0.0250 | 0.0125 | no |

Section 5's `stop_rule` fires. No long run is authorized by this file; the decision returns to the operator.

### What the shootout established, and what it did not

**No arm moved measurably in six hours.** Every arm's whole h-curve sits inside a single standard error of its own starting point, and the h4→h6 slopes are negative or flat for all three. The starting weights are P24 — hour 24 of the Phase 14 run — and Phase 14's own gain arrived in its first six hours, so a further six hours from that point producing nothing is consistent with what was already known rather than surprising.

So the correct reading of `stop_rule` here is **not** "the damped recipes are worse than the control". It is that this experiment could not tell any of the three apart. A recipe comparison needs either a starting point that is still learning, a horizon long enough for the differences to accumulate, or an instrument that can resolve the margin it is asked about — and this run had none of the three.

**The `setups_causal` result is the easiest thing in this report to misread, so plainly: it does not show that expanded setups fail.** Arm C's training distribution moved hard — mean game length 855.6 plies against B's 592.7 and A's 482.3, a higher draw rate, and 26,297 games for the same ~20M learner decisions where B played 47,508. The distribution moved; the strength did not. Six hours from a saturated start cannot separate "expanded setups do not help" from "expanded setups need a longer horizon, or a start with headroom left" — this experiment does not distinguish those two, and nothing here should be cited for the first. The one directional hint, itself well inside the noise, points the other way: C is marginally *best* of the three on the `adversarial_opponent` stratum, where the opponent plays adversarial setups and the arm does not.

What it *did* establish is infrastructural, and that part is solid: the window collector holds iteration wall-time to a coefficient of variation near 0.05 across ~300 iterations where Phase 14's grew 8× over 102, three arms ran six hours each with zero vetoes and zero non-finite losses, gradients or parameters, and the accepted objective, controller and filter were driven unmodified throughout at the frozen 0.25 retention.

## 6. `known_limitations`

- **6-hour horizon.** Phase 14 established that six-hour runs are decision-grade for this model, and also that its own gain arrived in the first six hours. A six-hour shootout can rank recipes at six hours; it cannot say what any of them does at sixty.
- **One seed per arm.** Each arm was run once. The arm-to-arm differences reported here carry no estimate of seed-to-seed variance, and the 0.03 decision margin was fixed in advance precisely because that variance is unmeasured.
- **The decision margin is smaller than the instrument's noise.** One standard error on the 60-board decision subset is ±0.056 at the observed EWR; the predeclared margin is 0.03, i.e. 0.53 SE. The clearest demonstration is that the same games read over the full 120-board pack reverse the verdict for arm B (+0.046 against +0.017). Neither reading is preferable; both are too noisy for the question. Every EWR difference in this report smaller than about 0.11 should be treated as unmeasured.
- **No arm learned, so no recipe was actually tested.** The six-hour curves are flat within noise for all three arms, which means the shootout compared three recipes none of which had a measurable effect. `stop_rule` firing reflects that absence, not a demonstrated ranking.
- **The window budget pins data, not collection time.** Rows per iteration and training seconds hold to a CV near 0.06. Collection time does not, when the opponent mixture fragments the inference batch: the accepted collector groups pending decisions by acting checkpoint, so arm A's growing historical pool (2 → 13 members) cut its collection throughput from 1,996 to 830 plies/s. `pure_current` arms hold one snapshot and are unaffected.
- **Partial-window advantages are built and tested but unused.** §2.2's boundary bootstrap exists in `targets.py` and is covered; the production path emits whole games so every row carries an exact W/D/L target, because the accepted objective averages its value and belief terms over every row and has no per-row loss mask. Rewriting the objective to add one was out of scope.
- **A stable behavior identity.** The learner's collection snapshot is the constant `CURRENT` and its weights rotate each window, because a window collector continues the same games across an update. Each window's telemetry records the state-dict digest that actually played it, but the per-decision token no longer distinguishes them.
- **In-flight games are not checkpointed.** A resumed arm keeps its weights, optimizer, EMA, clock and window numbering, and reseats its population from fresh draws. Partially-played games are lost, not corrupted.

---

_Engineering deliverable. Not a strength claim, not a scientific result, and not a validation of any recipe beyond the six hours measured._
