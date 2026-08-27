# Phase 16 — Agent 3
## Training loop v2 and the 3×6-hour recipe shootout

## Mission

Phase 14 gained +0.0414 EWR in its first six hours and nothing measurable in the
next 54, while specializing against the anchor and losing ground on unseen styles.
The accepted update rule already matches the paper's core (PPO clip 0.2, λ 0.5/0.8,
top-quartile advantage filter with 0.01 floor, KL-to-behavior). What diverges is the
**schedule and the data distribution**. Build a Phase 16 training loop that closes
those divergences, then run a controlled three-arm, 6-hour shootout to decide the
Phase 16 production recipe.

Phase 14 established that 6-hour runs are decision-grade for this model. No run in
this instruction exceeds 6 hours.

Read `00_PHASE_16_OVERVIEW.md` first. Phase 9/14 training modules are read-only —
import what fits unchanged; rebuild the rest under `phase16`. Phase 14's files are in
the sealed launch closure: editing them breaks the still-open run. Never do it.

## 1. Process boundary and namespaces

Boundary identical to Agent 1 section 1, plus: your 6-hour runs are heavy compute —
take the compute lock (overview §5), and hold a power assertion (`caffeinate`) for
each run. Namespaces:

```text
stratego/training/phase16/       tests/training/phase16/
scripts/run_phase16_agent03.py   checkpoints/phase16/
data/phase16/                    reports/phase16/
```

Starting weights for every arm: the read-only P24 copy
(`checkpoints/phase15/p24_source_readonly.pt`), digest-verified
(`622d9e6caa723c93…` model state). Fresh AdamW moments. Never write to any
accepted checkpoint path.

## 2. The loop changes (each one a config flag, so arms are exact)

1. **Window collection** (the paper's fixed-iteration structure). Maintain a
   persistent population of in-flight games (default 96, matching the rehearsed
   topology). One iteration = advance the population until a fixed budget of
   **learner decisions** is collected (default 65,536), then train, then continue
   the same games. Iteration wall-time becomes independent of game length — the
   failure that ballooned Phase 14's iterations from 24 to 138 minutes goes away
   structurally. Games ending mid-window are replaced by fresh draws.
2. **Window-edge targets.** λ-returns must bootstrap at window boundaries:
   advantages via TD(λ_A = 0.5) over stored values within the window, tail
   bootstrapped from v at the boundary; W/D/L targets via λ = 0.8 blending toward
   the final outcome only once the game finishes (buffer per game until then).
   Reuse `phase9_targets` math by import where signatures allow; rebuild otherwise.
   **Required invariant test:** for a finished game whose stored values are
   identical, targets computed windowed (spanning ≥3 windows) equal targets
   computed whole-game, to float32 tolerance.
3. **Schedules** (defaults; record exact values in the run config):
   - LR power law: `lr(n) = clamp(lr_max * n^-1.1, lr_min, lr_max)` with
     `lr_max = 1.5e-4`, `lr_min = 1.5e-5`, n = 1-based iteration;
   - entropy coefficient annealed: `c_H(n) = max(0.001, 0.005 * n^-0.3)` —
     Phase 14 ran the terminal floor from step 0; this restores the accepted
     Phase 9 starting level and decays it smoothly;
   - constant-LR and constant-entropy remain available flags (control arm).
4. **Epochs per iteration**: 1 (paper) vs 2 (Phase 14) — flag.
5. **EMA** of weights, decay 0.999, evaluation-side only — flag. Checkpoints store
   both raw and EMA states.
6. **Opponent mixture**: `pure_current` (100% current self-play, the paper) vs
   `phase14_mixture` (58/30/12 current/historical/handcrafted) — flag. Handcrafted
   bots remain in *evaluation* regardless.
7. **Setup mixture**: `library` (Phase 14's 35% neutral + 65% P10-D, train split)
   vs `expanded` (50% library + 50% drawn uniformly over Agent 1's
   `phase16_adversarial_setups_v1` families) — flag. Every draw passes the imported
   orientation gate. Fallback if Agent 1's pack has not landed: build the authored
   families yourself from Agent 1 §5's family list, name the copy
   `phase16_agent03_interim_families_v1`, and reconcile digests in the report.

Unchanged, by explicit choice: PPO clip 0.2, advantage filter (0.75 quantile,
0.01 floor, PPO term only), value weight 0.5, belief-aux weight 0.25, adaptive-β KL
machinery with its accepted thresholds, grad-norm 1.0, minibatch 512, float32,
battleless-100 / absolute-4000 training rules, MPS device with fixed inference batch
shape 64.

## 3. Correctness gates before any 6-hour run

1. 20-minute fixed-seed smoke run: completes, checkpoints, resumes, and a CPU rerun
   of one iteration's update is bit-identical given identical inputs;
2. the window-edge invariant test (section 2.2) passes;
3. collection throughput within 2× of Phase 14's measured plies/s on this machine;
4. run full pytest — green.

## 4. The shootout — three arms, 6 hours each, sequential

All arms: same start (P24), same seeds where applicable, same evaluation.

```text
A  control   window collector + phase14 hyperparameters:
             LR 7.5e-5 constant, entropy 0.001 constant, 2 epochs,
             no EMA, phase14_mixture opponents, library setups
B  damped    LR power law, entropy annealed, 1 epoch, EMA,
             pure_current opponents, library setups
C  damped+   B plus expanded setup mixture
```

Arm A isolates infrastructure (same collector, old recipe); B isolates the
schedule/damping package; C adds distribution. Evaluation: at h = 0, 2, 4, 6, score
the checkpoint (EMA where present, raw in A) on Agent 1's **60-board quick subset**
of `phase16_benchmark_v1` plus the **adversarial stratum** (fallback: your interim
packs, named). Also record per-iteration: move entropy, KL diagnostics, clip
fraction, game-length distribution, iteration wall-time.

## 5. Predeclared decision rules

```text
adopt_recipe    max(B, C) final-hour benchmark EWR >= A + 0.03
setups_causal   C adversarial-stratum EWR >= B + 0.03
plateau_check   report each arm's h4->h6 slope; a flat B/C with a passing h6
                still adopts, but the report must say the plateau moved, not vanished
stop_rule       if neither B nor C clears adopt_recipe: STOP. Write the report,
                hand back to the operator. No long run is authorized by this file.
```

Freeze the winner as `checkpoints/phase16/phase16_recipe_candidate_v1.json`: every
flag, schedule constant, mixture proportion, and the h-curve of all three arms with
pack names.

## 6. Part B — capacity probe (only if `adopt_recipe` passed; may be deferred)

Under the winning recipe, one additional 6-hour run with a 2×-width C1 variant
(fresh init for widened tensors is not meaningful on a 6 h continuation — instead
widen via a new model trained from scratch is out of scope; therefore: run the probe
only if a principled warm-start exists, e.g. net2net-style width duplication built
in `phase16`; otherwise write a scoping note and defer to the operator). Also write
a scoping note (no build) for observation enrichment: threat/evasion/protection/
refused-capture planes computable from the action history, per the paper's
Appendix C. Honest sizing beats a rushed probe here.

## 7. Report

`reports/phase16/agent_03_report.md`, sections mirroring this file: what was built,
gate results, the three h-curves with SEs, the decision the predeclared rules
produced, and `known_limitations` (6-hour horizon; single seed per arm; quick-subset
evaluation noise ~±0.04 at 60 boards).

## APPENDIX 2026-08-27 — if a redesign is authorized, read this first

The 2026-08-27 shootout returned **STOP**: no arm moved, because §4 started every
arm from **P24**, and the run P24 came from had stopped learning 18 hours earlier.
Agent 3's refinement is correct and worth preserving — P24 was the right choice for
an *infrastructure* shakedown (it proved the harness on the weights the phase must
build on) and the wrong choice for a *comparison*. The brief asked one run to do
both. **The infrastructure claims stand and need no re-running; only the comparison
does.**

### Which starting checkpoint has headroom — settled by measurement

From the Phase 14 sidecar evaluation (all candidates, the *same* 2,200 games each,
`/Volumes/Brandon_Washington/stratego_phase14_sidecar_eval/`, SE ≈ 0.0087 per point):

```text
hour     0      6     12     18     24     30     36     42     48     54
EWR   .7600  .8014  .7909  .8086  .7886  .7868  .7841  .7752  .7882  .7905
min   .5209  .6309  .6691  .7018  .6364  .6555  .6627  .6300  .6491  .6264
```

`h0 -> h6` is **+0.0414, about 3.3 SE of the difference** — the only resolved jump
in the run. Every later step is inside noise (a random walk around .79). So:
**start a redesigned comparison at hour 0** (the Phase 9 warmstart the run itself
began from). Hour 6 and later have no demonstrated headroom and would reproduce
this phase's null.

### Ask a better-powered question than "which recipe is best"

A recipe *difference* is some fraction of the 0.0414 total available learning, i.e.
0.01–0.02 — which needs SE ≈ 0.005 to resolve at 2 SE, i.e. thousands of games per
arm per time point. That experiment is not affordable here. The affordable question
is a **curve-shape** question, where the signal is several times larger:

```text
H1  does the arm reproduce the +0.0414 first-six-hours jump?   (3.3 SE at 2,200 games)
H2  does its curve still climb after hour 6, where A's flatlined?
```

H2 is the actual claim §2.3's damping is supposed to support ("preventing
plateauing later"), and a continuing climb is a far easier signal to detect than a
marginal EWR gap at one time point. Score at hours 0/2/4/6 **and beyond hour 6** —
a 6-hour horizon cannot answer H2 at all, so a redesign needs ~12 h per arm.

### Consider selecting on worst-stratum, not mean

The `min` row above peaks at **h=18 (.7018)** and degrades afterward while the mean
stays flat — that is the anchor-specialisation signature, and Phase 14's frozen rule
picked h=18 correctly. For a goal defined by an opponent who *hunts weaknesses*,
worst-stratum is the better objective than mean EWR. Any redesign should predeclare
which it optimises.

## AMENDMENT 2026-08-26 — §2.3 schedule horizons (defect in this brief, caught by
## the executing agent before arms B/C ran)

The §2.3 defaults transcribed the paper's exponents without re-horizoning from its
~43,000 iterations to this loop's measured ~330 per 6-hour run: as written,
`lr(n) = clamp(1.5e-4·n^-1.1, 1.5e-5, ·)` reaches its floor at n≈9, so arms B/C
would train ~97% of the run at 1.5e-5 — 5× below the control — and the shootout
would measure a starved LR, not a damped schedule. Corrected schedules, with N =
the run's planned iteration count (recompute from measured s/iteration):

```text
lr(n) = clamp(lr_max * (n/n_ref)^-1.1, lr_min, lr_max),   n_ref = ceil(0.125*N)
```

Applied by the executing agent with measured N = 313 (69.0 s/iteration), n_ref = 40:
mean LR over the horizon moves from 1.7e-5 (unamended) to 5.6e-5 (same order as the
control's 7.5e-5). `n_ref = 1` remains the code default and reproduces the original
text.

**The entropy schedule is deliberately NOT re-horizoned.** The coordinator's first
proposed correction (`n_ref_H = ceil(0.04*N)`) was itself defective — no upper
clamp (it would *start* at 0.0108, above the accepted 0.005) and a terminal value
of 0.0019 that never reaches the named 0.001 floor. The §2.3 formula as originally
written (`max(0.001, 0.005*n^-0.3)`) starts exactly at 0.005 and reaches the floor
at n = 213 of 313 (68% of the run) — a smooth decay across most of the run followed
by the terminal value, which satisfies the section's intent: the failure it names
is Phase 14 running the terminal floor *from step 0*, not a floor arriving before
the final iteration.

lr_max/lr_min and the entropy endpoints are unchanged; arm A (control) is
unaffected. N, n_ref, and the full arithmetic are recorded in
`contract_document()["schedule_amendment"]`, carried in every arm's run config and
checkpoint, and noted in the report's deviations section.
