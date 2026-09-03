> **SUPERSEDED on 2026-09-02 by `phase18_g3_stage6a_joint_design_v2.md`.** The setup-only
> C0 evaluation design in this document — the frozen C0 mover, the fixed-library comparator,
> the P1/P2 contrasts, the confirmation sizing rule, the concentration review flags, the
> preflight, the teacher-regime probe and the in-process evaluator with R1–R11 — is no longer
> the active Phase 18 design. Its evidence is preserved unchanged: the throughput and storage
> measurements (section 2.6), the teacher-schedule facts (2.1), the library and leakage facts
> and the data-boundary record (2.3–2.5), the collector reconciliation with S21–S23 (3), the
> EMA parameter-aging record (4), the cluster-aware variance model, stratified bootstrap and
> direct per-base variance (5) and the appendix tables remain the reference the replacement
> design cites. Nothing below was re-run or edited for the supersession.

# Phase 18 Agent 6 — Stage 6A: Gate G3 evidence, power, and feasibility analysis (v1, corrected)

**Stage 6A of the G3 design work package. Analysis only.** This document freezes no contract
and authorizes no run. In producing it and its correction no Stratego game was played, no
setup pool was generated, no model was trained or loaded, no confirmation-slice board was
materialized, no confirmation outcome was observed, and no sealed artifact was opened. Every
numerical table below is reproduced by `scripts/phase18_g3_stage6a_tables.py` from tracked
evidence (`--write` regenerates `reports/phase18/g3_design/phase18_g3_stage6a_tables_v1.json`,
`--check` recomputes and compares); the tables are reprinted verbatim in Appendix A.

Date: 2026-09-02. Author: Phase 18 Agent 6 (G3 design agent). Corrected the same day on
reviewer instruction; the correction record is section 0.1.

---

## 0. Identity, the settled design choices, and the correction record

```text
backend-foundation commit   d37ebe77ae388d87267d47e75223967574332f94   (named by the operator)
                            tree eadbd2d91c7fd00fb499bc30041c7412b4c909ae
                            published: origin/phase18/g3-backend-foundation == d37ebe77 (git ls-remote)
                            carries: reviews/P18-D005_REVIEW.md (accepted PROCEED, G2 CLOSED)
                                     amendments/P18-A001_* and reviews/P18-A001_REVIEW.md (OPERATIVE)
                                     phase18_rule_identity_errata_v1.json: O-P18-EVALRULES-1 RESOLVED
                                     phase18_execution_storage_policy_v1.md (corrected record)
                                     .gitignore lines 110/114: output/phase18/worktrees/, output/phase18/runtime/
design branch               phase18/g3-design, created from d37ebe77 as a linked worktree at
                            output/phase18/worktrees/g3-design/  (git-ignored; policy item 2)
first Stage 6A commit       286acb33e741748a98b02a4fa883c47f492e659e  (preserved; this correction is a
                            new commit on top of it, nothing amended or rewritten)
runtime location            output/phase18/runtime/  for every future G3 runtime byte (policy item 3)
protected file              reports/phase13/phase14_launch_manifest_v1.json: never staged, never modified
```

Design choices settled by the operator on 2026-09-02 and applied throughout:

```text
collection regime           pool-versus-pool teacher games
collection rules            TRAINING_RULES   (battleless 100, absolute 4000, first player 0)
play evaluation rules       EVALUATION_RULES (battleless 200, absolute 4000, first player 0; P18-A001)
fresh setup initialization  independent initialization per training seed (three seeds)
decision model              the EMA; the raw actor is diagnostic only
primary contrasts           both use the frozen 0.05 practical-margin scope: paired 95% lower bound
                            above zero AND point estimate at least 0.05
unusual/procedural stratum  deferred to G4; G3 makes no generalization claim
evaluator                   in-process evaluator only after every R1–R11 credibility test passes;
                            the proven G1 harness is the fallback
execution topology          sequential versus concurrent seeds chosen from the preflight throughput
                            result, then frozen
screening                   the library arm is included; the teacher-regime probe is included
confirmation size           the smallest size the cluster-aware analysis supports, selected only at the
                            post-calibration review (section 5.6)
```

Reviewer decisions frozen on 2026-09-02 (correction round):

```text
powered effect              0.06 EWR for both primary contrasts
minimum confirmation B      160 bases, 10 per family
library arm                 one independent library draw per training seed (18 screening arms, 9 confirmation arms)
confirmation B              selected only at the post-calibration review by the frozen sizing rule (section 5.6)
confirmation bases          drawn within each family from the reserved validation bases 410..449 by a frozen
                            deterministic seed, never by taking the first indices
collector cadence           T = 202 plies per period retained; S = 2,560 slots provisional until preflight,
                            any adjustment constrained to keep about four completed outcomes per generated
                            setup and recorded before period 1
```

### 0.1 Correction record (this commit against 286acb33)

1. The global base bootstrap is replaced by a stratified cluster bootstrap: bases are
   resampled independently within each of the 16 families, each selected base carrying all
   of its opponents, colours, arms and seeds; a finite-stratum rescaling is applied
   (section 5.2).
2. Variance estimation is corrected. The raw cross-seed correlation of case differences
   estimates `rho_b + rho_w`, not `rho_w`; sizing now uses the direct stratified per-base
   variance, and `rho_b` / `rho_w` remain diagnostics with a tested joint decomposition
   (sections 5.1–5.2).
3. Sizing is uncertainty-aware: a predeclared one-sided 95% upper bound on the
   screening-derived per-base variance, explicit handling of values outside the tabulated
   range, and a return to review when no valid size fits within the 640 reserved bases
   (section 5.6).
4. The reviewer decisions above are frozen and applied (powered effect 0.06, minimum B 160,
   independent library arm per seed, B chosen only at the post-calibration review, seeded
   base selection).
5. Full-gate power is validated by Monte Carlo — P1, P2 and positive direction in all three
   realized seeds — and distinguished from per-contrast power; inference is stated as
   conditional on the three realized training seeds (section 5.4).
6. The data-boundary record is corrected: the first Stage 6A script decoded the structural
   content of all 8,000 library bases, including the reserved validation bases and the test
   bases. That is recorded as feature exposure and not claimed otherwise; no performance
   outcome or confirmation game was observed. Structural references and thresholds are now
   computed from the train split only (sections 2.5, 7).
7. `T = 202` is retained; `S = 2,560` is provisional until preflight under the constraint
   above (section 3.2).
8. The teacher-regime probe is included; harness preference and fallback are stated
   (sections 6.1, 8).
9. Every ready-row minibatch, including a final partial one, receives an optimizer step
   (section 3.2).
10. The script reads its frozen constants from their tracked source files (Appendix A, T3);
    the JSON and appendix are regenerated; the checks listed in Appendix B were run.

---

## 1. What G3 must answer

Common contract section 11, Gate G3 (setup-only Stratego benefit): the trained setup policy
improves over its fresh initialization; it beats the fixed library by the predeclared
practical margin with paired uncertainty excluding zero; direction is consistent across the
required seeds; diversity, legality, orientation and stability gates pass. Under the settled
choices both primary contrasts are read on the EMA setup model and both must show a paired
95% lower bound above zero and a point estimate of at least 0.05 effective win rate (EWR).

Constraints carried into the design:

- The EMA decides (method map S28, `exact`). P18-D004/P18-D005 established that after 64
  updates the EMA retained 0.938 of its initial parameter contribution and its synthetic
  utility moved 0.9–2.8% of the raw actor's movement. A raw-actor result may inform a
  development decision; it cannot close G3.
- The margin is scored in EWR from a known instrument resolution (P18-D005 review).
- Development data only: `setup_learning_development` is the only pack a setup-only assay
  may iterate against; no sealed Phase 8 test example and no `operator_sealed` entry may be
  read.
- G3 changes no C1 weight, so no sealed-test head metric is triggered.

---

## 2. Evidence inventory

### 2.1 The frozen Phase 8 teacher schedule (setup-outcome collection)

Source: `stratego/training/warmstart_contract.py`, `stratego/training/rule_population.py`,
`reports/phase_8_data/agent_02_matchup_counts.csv`.

```text
roster (frozen order)   random_legal@1.0.0, basic_heuristic@1.0.0, tactical_rule_based@1.0.0,
                        strategic_rule_based@1.1.0, stress_scout_rush, stress_miner_rush,
                        stress_draw_seeker, stress_berserker, stress_information_miser,
                        stress_chaos (all @1.0.0)
cells                   100 ordered (red, blue) cells, red-major, self-play cells included
corpus counts per cell  train 200 / validation 40 / test 40  (20,000 / 4,000 / 4,000 games)
rules                   CORPUS_RULES = TRAINING_RULES (battleless 100)
no neural model         the corpus is played entirely by the rule population
```

Common contract section 9.2 defines the live setup stream as: the raw setup model samples a
reusable pool; the frozen teacher matchup schedule plays games from that pool; completed
setup outcomes update the setup policy. That is the production mechanism and, with
pool-versus-pool settled, the G3 collection mechanism.

Outcome signal per cell in the train split (20,000 games, battleless 100): red wins 33.5%,
blue wins 32.6%, draws 33.9%, mean 258.8 plies. Nine cells are at least 95% draws (the
`stress_draw_seeker` / `stress_information_miser` / `stress_scout_rush` / `random_legal`
combinations; two self-play cells are 200/200 draws), 23 cells are at least 80% draws, 53
cells are at most 10% draws, 18 cells have one side winning at least 90%, and the mean
per-cell outcome variance Var(z) is 0.429 with 26 cells below 0.1. Under the frozen uniform
schedule roughly 9% of collection games carry no outcome information and about a quarter
carry very little; the schedule is a production identity and is not proposed to change, so
the cost model counts these games.

### 2.2 The frozen mover (C0) and the fresh setup-policy comparator

```text
C0 = G1 reproduction candidate   /Users/brandonwashington/Dev/stratego_phase18/g1_control_v1/
                                 dry_run_artifacts/warmstart_c1_v1.pt
                                 sha256 460a246be32b821a6d6d7feb928b272a4be1014ff55053f329980e21e3be074c
                                 10,459,947 bytes (Appendix A, T3); present and re-hashed 2026-09-02
accepted Phase 8 reference       checkpoints/phase8/warmstart_c1_v1.pt, sha256 f7e9c40d…eec7ca (not used in G3)
policy token                     phase6_c1_warmstart_greedy@0.2.0+float32, greedy, CPU float32
```

G1 closed on C0 (P18-D003: delta +0.006348, 95% [+0.000793, +0.011902] against the −0.010
margin on 4,096 independent pairs). Both the learned-setup and the fixed-library lanes use C0.

Fresh setup-policy comparator: `Phase18SetupModel`, 4 blocks / width 128 / 4 heads /
feed-forward 512, 802,320 parameters (tolerance 0), positional init std 0.1, forced flag
handedness plus independent 50% reflection, initialized per seed from
`derive_stream_seed(namespace, 'model_init', k)`. Under S28 the EMA at update 0 is the raw
initialization, so each seed's fresh comparator is its own `EMA_0 = raw_0`.

### 2.3 The fixed-library comparator

```text
setup_library_v1     digest 7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777
                     8,000 entries, 16 families F00–F15 x 500 bases; content and reflected fingerprints per entry
split rule           base_index 0..399 train (6,400), 400..449 validation (800), 450..499 test (800)
production source    training_setup_source('neutral_v1') -> train split, uniform over families
```

The fixed-library own-setup source is the production sampler over the train split: the
configuration the accepted Phase 8 corpus used. Under the frozen decision each training seed
gets its own independent library draw (`L_k`), so the library arm is played three times on the
same cases with seed-specific own-setup streams.

### 2.4 The development pack and the strata

`setup_learning_development` is constructible from existing assets (evaluation contract):
the accepted draw path exists (Phase 16's benchmark drew every board from the library
validation split through `Phase15MatchSetupSources`; `SetupBank` and `confirmation_bank.py`
provide seeded banks with a reflection-class separation audit).

Proposed construction (not frozen): opponent setups are validation-split bases; the pack is
split by base into a screening slice and a reserved confirmation range so that nothing
decided on the screening slice can see a confirmation board.

```text
screening slice        base_index 400..409  (10 per family, 160 bases)  -> 160 x 10 x 2 = 3,200 cases per arm
reserved range         base_index 410..449 (40 per family, 640 bases)
confirmation slice     B/16 bases per family drawn from the reserved range by a frozen deterministic
                       seed — derive_stream_seed(<G3 namespace>, 'confirmation_bases', family_id) applied
                       to the 40 reserved indices of that family — with the permutation recorded in the
                       contract before any confirmation board is opened; B itself is fixed only at the
                       post-calibration review (section 5.6)
case                   (opponent base, opponent policy, own colour, replicate 0)
own setup per arm      learned arms: one EMA sample under the case's own-setup seed;
                       library arms: one neutral_v1 draw from the train split under the case's seed and
                       the training seed's library stream
game seed              one recorded function of (pack, case, lane, colour, opponent); identical across
                       every arm of a contrast
"opened"               a slice is opened when any of its boards is materialized as a played or banked
                       setup; reading the library's recorded fingerprints for the leakage audit does not
                       open it. The confirmation range stays unopened in the calibration stage.
```

Strata: `familiar` = the library validation bases (all 16 families). `unusual_procedural` is
not populated (evaluation contract E1) and is deferred to G4; G3 reports it `not_run` and
makes no generalization claim. `operator_sealed` is absent and blocks G6 only. A descriptive
opponent stratum exists for free: teacher-seen opponents (strategic, tactical, five stress
policies) versus unseen neural opponents (p18, p24, phase9_anchor); it reads on transfer, not
on unfamiliar setups, and never gates.

Evaluation opponents (`stratego/evaluation/phase17/opponents.py`, bound by digest): p18
(sha256 aa2cc39b…), p24 (9bf256a9…), phase9_anchor (ed0f5198…), plus seven code opponents
(strategic_rule_based, tactical_rule_based, stress_scout_rush, stress_miner_rush,
stress_berserker, stress_information_miser, stress_chaos).

### 2.5 Leakage, reflection-class boundaries, and the data-boundary record

Rules (evaluation contract, frozen): a setup and its horizontal mirror are one class; packs
are separated by reflection-class fingerprint; pairwise content and class intersections
across all packs must be empty before a pack is opened; selection may read the validation
split and `setup_learning_development` only.

```text
setup learner training    pool-versus-pool teacher games: no library setup is ever sampled, seen or
                          scored by the learner (S32 on-policy requirement)
C0 mover                  trained on train-split bases only (Phase 8); validation bases were the
                          Phase 8 validation corpus (selection data) -> development use is allowed
dev pack opponents        validation bases -> disjoint by base from the fixed-library own-setup source
                          (train bases); class disjointness is PROVED by the pack audit over the
                          library's recorded content and reflected fingerprints, never assumed
sealed                    test bases 450..499 and the sealed Phase 8 test corpus: no game, outcome or
                          performance number touched; operator_sealed absent; the Phase 8 test
                          multiplicity counter unchanged
learned setups            every emitted class is recorded so the operator_sealed check can be run
                          retroactively once that pack exists
```

**Data-boundary record.** The first Stage 6A commit (`286acb33`) computed the library
structural references by decoding every one of the 8,000 library rows — piece positions of
the train bases, of the reserved validation bases 410..449 and of the test bases 450..499
alike. That was feature exposure of setup structure and is recorded here as such; it is not
undone and it is not claimed that those boards went unread. What was not observed: no
performance outcome, no game, no confirmation-slice board materialized or played, no
confirmation outcome, no sealed Phase 8 test example, no `operator_sealed` entry. The
corrective script decodes train-split rows only; the file is still read line by line because
it is one file, and a non-train line has only its split tag matched (Appendix A, T1 and the
JSON `library.data_boundary`). All structural references and thresholds in this document are
computed from the 6,400 train bases.

### 2.6 Throughput, runtime and storage evidence

Teacher-versus-teacher games (the collection cost; Appendix A, T3):

```text
Phase 4 baseline league     44,544 rule-vs-rule games: mean 0.0865 s per game in one process
                            (median 0.068, max 0.53), mean 388 plies, EVALUATION_RULES;
                            slowest policies involved: stress_chaos 0.133 s, information_miser 0.119 s
Phase 4 runner sweep        512 matches in 31.5 s (16.2/s, one worker) down to 4.85 s (105.6/s)
Phase 8 train corpus        mean 258.8 plies under battleless 100
planning bracket            30 (conservative) to 100 (measured ceiling) games per second; the exact
                            G3 collector is unmeasured, so the preflight microbenchmark is mandatory
```

C1-mover evaluation games:

```text
Phase 17 evaluator          240 games in 12.7 s mean, 8 workers, CPU float32, in-process owners
                            -> 18.9 games/s (this harness has not passed the Phase 18 R1–R11 tests)
G1 confirmation             16,384 games in 10,952.8 s -> 1.50 games/s wall; 4.84 s CPU per game,
                            15.6 ms per ply, mean 310 plies (neural_worker_v1, single_request;
                            the harness that passed R1–R11 for the random gate; the fallback)
```

The setup learner: G2 ran 64 updates plus four 4,096-sample evaluations in 913–930 s per seed
on CPU float32 with 4 threads (about 10 s per update including evaluation); Phase 17 measured
pool generation for 2 x 512 at 3.18 s CPU and a 5-epoch update on 320 episodes at 0.82 s CPU.
Learner compute is a few seconds per update.

Storage: G2 outcome receipts 20.9 MB for 262,144 outcomes (about 80 B each); telemetry 6.4 KB
per update; a setup checkpoint (raw + optimizer + EMA) 12.9 MB; a G1 receipt row about 1.46
KB. 158 GiB free on the data volume; 48 GiB RAM; Apple M4 Pro, 10 performance and 4
efficiency cores.

### 2.7 Instrument resolution and prior effect sizes

```text
own-setup instrument   SD of the paired per-board difference 0.5391 (Phase 17 rows, n = 3,000,
                       joint_move_setup minus move_only: same mover, opponent, opponent setup,
                       colour and seed; only the own setup differs); tie fraction 0.594;
                       single-lane SD 0.4236; implied same-case cross-arm correlation 0.190
predeclared margin     0.05 EWR for every G3/G4 EWR contrast (evaluation contract, frozen)
worst-stratum rule     a stratum is a REGRESSION only with at least 200 paired games
Phase 17 prior         fresh setup init vs fixed library: −0.0917 at h0; −0.0679 pooled h6–12
(defective method)     (paired SE 0.0233, t = −2.91); difference-in-differences +0.0237 (SE 0.0545)
```

The Phase 17 prior is the only real-game measurement of a fresh setup network against the
library with this mover family: a fresh initialization sat roughly 0.07–0.09 EWR below the
library. A learner that beats the library by 0.05 must therefore move its EMA by roughly
0.12–0.14 EWR relative to its own init, so the two primary contrasts are different-sized asks.

### 2.8 Learning dynamics observed in G2 (synthetic, observed only)

Raw utility z at updates 0/8/16/24/32/40/48/56/64:

```text
G2         seed 1  -0.763 -0.547 -0.302 +0.236 +0.722 +0.990 +1.197 +1.296 +1.315
           seed 2  -0.484 +0.060 +0.445 +0.880 +0.999 +1.093 +1.181 +1.260 +1.302
           seed 3  -0.187 -0.010 +0.347 +0.701 +0.844 +1.007 +1.086 +1.127 +1.199
G2 raw     seed 1  -0.135 +0.032 +0.370 +0.578 +0.606 +0.699 +0.784 +0.802 +0.834
confirm.   seed 2  -0.043 -0.055 +0.097 +0.434 +0.564 +0.616 +0.698 +0.777 +0.858
           seed 3  -0.046 +0.167 +0.437 +0.666 +0.747 +0.795 +0.767 +0.800 +0.819
EMA z moved 0.008–0.05 over the same 64 updates in every seed
```

These are synthetic-landscape observations with four outcomes per setup and
`P(win) = 0.9·sigmoid(3z)`. They say nothing quantitative about Stratego learning speed; real
outcomes are noisier (section 2.1), so the calibration stage measures the curve rather than
assuming it.

---

## 3. Setup-outcome collection reconciled with method-map rows S21–S23

### 3.1 What the published semantics require

```text
S21 pool lifetime      rows are retained for storage_duration after the period that added them;
                       a game started under pool A that finishes after the refresh to pool B is
                       attributed to A's row; an undersized window is a fatal error, never a loss
S22 snapshot binding   a setup stays bound to the behaviour snapshot that sampled it; the PPO ratio
                       and the advantage use the recorded behaviour quantities, never a re-forward
S23 window reset       counts, means and ready flags are zeroed at every pool refresh; aggregation
                       spans exactly one collection period
S09 / S10 / S24        variable completed-outcome counts m per setup; m = 0 rows are not trained and
                       never treated as a draw; identical boards collapse to one row bound to the
                       newer snapshot; immediately terminal setups are filtered from play
S26                    one optimizer step per minibatch of ready rows, minibatch size 1,024, five epochs
```

`stratego/training/phase18/setup_buffer.py` already implements these semantics (`add_pool`
reallocates counts and ready flags to zero for every row and collapses duplicates to the
newest period; `add_outcome` keeps a running mean per row and raises on an unknown
fingerprint; `filter(current_period)` drops rows whose `period_added + storage_duration <
current_period`; `minibatches` yields the permuted ready rows in slices of the batch size,
the last slice partial), and `SetupTrainer.update` takes one optimizer step for every yielded
minibatch, so a ready-row count that is not a multiple of 1,024 produces
`ceil(ready / 1,024)` steps per epoch with the final partial minibatch stepped like the
others. `tests/training/phase18/test_setup_buffer.py` pins the buffer semantics: an outcome
finishing under the next pool attributes to the old row; an undersized window raises rather
than dropping the outcome; counts and ready flags reset when a new pool arrives; identical
played boards collapse to one row bound to the newer snapshot; a zero-outcome setup is
excluded, not trained as a draw; an outcome for an unknown setup is fatal; the advantage uses
recorded behaviour quantities.

**The G2 assay exercised none of the retention, binding or reset semantics in its data
flow.** Its driver (`synthetic_assay.run_seed`) generated a pool, drew exactly
`outcomes_per_setup = 4` instantaneous outcomes for every setup, updated, and filtered with
`storage_duration = 1`: no game crossed a period boundary, no row was attributed under a later
pool, no behaviour-snapshot age exceeded zero, no row had m = 0. That was correct for a
synthetic landscape whose "games" are instantaneous, and those semantics were verified by the
unit tests above rather than by the assay. A G3 collector that reproduced the same synchronous
four-outcome scheme would leave them untested in real play before G4 and would not be the
published method, which the paper itself describes as "slightly off-policy" because most games
span several collection periods.

### 3.2 Proposed G3 collector (asynchronous, published semantics; not frozen)

```text
slots                  S = 2,560 concurrent game slots, PROVISIONAL until the preflight; any adjustment
                       must keep about four completed outcomes per generated setup per period
                       (Appendix A, T7a: S = 2,624 gives exactly 4.0 at the corpus mean length) and must
                       be recorded in the launch record before period 1
period                 every slot advances T = 202 plies per period (the published 2 x train_every_per_player
                       = 2 x 101; RETAINED); a slot whose game ends immediately starts a new game from the
                       CURRENT pool, so games started in period t may finish in t + 1 or later
pool                   one pool of 1,024 setups per period (512 per lane), sampled by the raw actor at the
                       period boundary and added to the buffer (S10 de-duplication applies)
cells                  each new game takes the next cell of the frozen 100-cell schedule in cyclic order,
                       so game STARTS are exactly uniform over cells in every window of 100; completions
                       per cell per period are length-weighted and are recorded
outcomes               every completed game attributes two outcomes (one per pool row, from the owner's
                       perspective) through add_outcome; counts and ready flags reset at every refresh
                       (S23); m per row is variable and recorded (S09)
retention              storage_duration = 21 periods = ceil(4,000 / 202) + 1, covering the absolute move
                       limit under TRAINING_RULES so that no attribution can fail for age (S21)
update                 after each period: process the ready rows, five epochs, ceil(ready / 1,024)
                       minibatches per epoch with one optimizer step for each, the final partial minibatch
                       included (S26), one EMA update after the update (S28), then filter and regenerate
                       the pool
rules                  TRAINING_RULES for every collection game; the rules token is stamped on every receipt
determinism            fixed slot order, per-game seeds derived from (period, slot, ordinal) through
                       derive_stream_seed, so the whole stream is replayable on CPU as in G2
```

Expected steady state (Appendix A, T7a): 1,998 completions per period at S = 2,560, 3,996
outcomes, a mean of 3.90 outcomes per generated setup split across the generating period and
the next, 173 CPU-seconds of games per period, 20–67 s of wall time per period at 100–30
games per second. The first period trains on fewer completions than steady state (games
started at period 1 mostly finish in period 2); the ready-row count, the m distribution, the
fraction of rows with m = 0, the snapshot-age distribution and the cross-period attribution
count are recorded every period.

### 3.3 The synchronous alternative, recorded as a divergence

`D-G3-SYNC` (not recommended): exactly m = 4 instantaneous outcomes per setup per period with
`storage_duration = 1`, as in G2. Same games per period, simpler, fully replayable, but it
never exercises S21–S23 in real play, never produces a behaviour-snapshot age above zero,
never produces m = 0 rows, and is not the published collection regime. If it were adopted for
G3 it would have to be declared as a G3-specific divergence and re-validated before G4. The
proposal here is to reproduce the published semantics (section 3.2).

---

## 4. EMA parameter aging (parameters only)

With decay 0.999 applied once per setup update, the EMA parameters after U updates are
`0.999^U · θ_0 + Σ_k (1 − 0.999)·0.999^(U−k) · θ_k`. Appendix A, T6 tabulates the retained
initial-parameter fraction and the weights: 0.938 at 64 updates, 0.880 at 128, 0.774 at 256,
0.368 at 1,000, 0.135 at 2,000, 0.050 at 3,000; the most recent 64 updates always carry
weight 0.062.

**No effective-win-rate response is inferred from this blend.** The only evidence on how the
parameter blend maps to performance is G2 (Appendix A, T2, lower table): at 64 updates the
trajectory weight was 0.062, yet the EMA closed 0.10–0.52% of its initial-to-optimum utility
gap while the raw actor closed 9.6–20.9% — a utility response of 0.9–2.8% of the raw actor's,
well below the parameter weight. Parameter interpolation therefore does not predict
performance, and this analysis makes no claim about the EMA's EWR at any budget. What the
calibration stage measures instead (section 6): the raw and EMA performance on the screening
slice at 128 and 256 updates, their divergence in parameter space (relative L2 distance per
update) and in performance, and the direction of the raw learning curve. The later reviewed
decision on whether to continue to 2,000 or 3,000 updates rests on those measurements plus the
cost table (Appendix A, T7b), not on the aging arithmetic.

---

## 5. Power and sample size around the actual resampling unit

### 5.1 Structure, correlation, and what can and cannot be estimated

Cases are `(base, opponent, colour)`: B bases x 10 opponents x 2 colours = 20B cases per arm,
with B/16 bases in each of the 16 families. Arms: for each seed k, the trained EMA `E_k`,
its own init `I_k`, and its own library draw `L_k`. Contrasts: P1 = `E_k − I_k`, P2 =
`E_k − L_k`, pooled over the three seeds.

Model (Appendix A, T3 and the script): `Y_a = F[family, type] + v[base, type] + w[case, type]
+ e[arm, case]`, with a FIXED family x type effect F (families are a design stratum, not a
sample), `2τ² = rho_b·V`, `2ω² = rho_w·V`, `2ε² = (1 − rho_b − rho_w)·V`, and the base and
case type-effects shared across seeds (worst case). Variance of one base's mean paired
difference (all 20 cases, all three seeds):

```text
frozen design (per-seed library arms; P1 and P2):  s2_base = V · [rho_b + rho_w/20 + (1 − rho_b − rho_w)/60]
one seed alone:                                    s2_base = V · [rho_b + rho_w/20 + (1 − rho_b − rho_w)/20]
pooled estimator:  Var = s2_base / B          effective independent paired units: n_eff = V / Var
```

At `rho_b = rho_w = 0` the frozen design gives n_eff = 60B; any base-level effect caps n_eff
near B / rho_b whatever the number of seeds, arms or cases: **bases are the scarce resource,
not games.** The shared-library alternative would have given 30B at zero correlation and
correlated the three seeds' P2 differences at 0.5 through the shared library game (Appendix A,
T4d); it is rejected by the frozen decision.

What the screening slice can estimate, and what it cannot:

- **The direct stratified per-base variance** `s2_base`: the sample variance of the per-base
  mean difference within each family (10 bases per family, 9 degrees of freedom each, pooled
  over 16 families to 144). It is the quantity sizing needs, it is unbiased for `Var` x B
  under the model (Appendix A, T5: 0.0204 against 0.0201; 0.0344 against 0.0344; 0.0048
  against 0.0048), and it needs no decomposition into `rho_b` and `rho_w`.
- **`rho_b` and `rho_w` as diagnostics only.** The raw cross-seed correlation of case
  differences estimates `rho_b + rho_w` (both the base-level and the case-level type effects
  are shared across seeds), so it cannot estimate `rho_w` on its own. The valid joint
  decomposition, after removing fixed family means, uses two moments: `m1` = the within-seed,
  within-base, across-case covariance of case differences (= `rho_b·V`) and `m2` = the
  across-seed, same-case covariance (= `(rho_b + rho_w)·V`), giving `rho_b = m1/V`, `rho_w =
  (m2 − m1)/V`. Tested in Appendix A, T5 at the screening structure: `rho_w` is recovered
  without bias (0.100 ± 0.010 at truth 0.10; 0.000 ± 0.011 at truth 0), `rho_b` with a small
  downward bias from family-mean centering (0.044 ± 0.008 at truth 0.05; 0.093 ± 0.009 at
  truth 0.10). Neither enters the sizing rule.

### 5.2 The stratified cluster bootstrap and its validation

Resampling unit: **within each family, its bases are resampled independently with
replacement** (B/16 draws per family), and each selected base carries every one of its
opponents, colours, arms and seeds; the resample statistic is the pooled contrast over the
resampled bases. Because resampling n_f bases inside a stratum reproduces only (n_f − 1)/n_f
of that stratum's variance, the resample distribution is rescaled about its centre by
`sqrt(n_f / (n_f − 1))` (1.054 at 10 bases per family, 1.033 at 16); the percentile interval
and the SE are read from the rescaled distribution.

Appendix A, T5 (3,000 replications per configuration for the SE, 300 datasets x 400
resamples for the bootstrap, a fixed family effect of 0.05 V present): Monte Carlo reproduces
the analytic SE within 2% for P1, P2 and a single seed; the stratified bootstrap's mean SE
equals the true SE (0.0112 / 0.0112, 0.0116 / 0.0116, 0.0055 / 0.0055) with 94–96% coverage;
the earlier global base bootstrap over-covers (0.977–1.000) because it treats the fixed family
differences as sampling variance. The naive SE that treats 60B paired games as independent is
half the truth at `rho_b = 0.05`.

### 5.3 The decision rule's per-contrast power

Both primary contrasts pass only if the stratified-bootstrap 95% lower bound exceeds zero
and the point estimate is at least 0.05. At a true effect of exactly 0.05 the point criterion
passes with probability 0.5 whatever the sample (Appendix A, T4a). The frozen powered effect
is **0.06 EWR**: 80% power requires `SE ≤ (0.06 − 0.05) / 0.8416 = 0.0119` (the lower-bound
criterion needs only `SE ≤ 0.0214` and is not binding). Appendix A, T4b gives n_eff, SE and
`d80` (the smallest true effect passing with 80% power) for B from 160 to 640 and `rho_b`
from 0 to 0.30 at `rho_w = 0.10`; T4e shows `rho_w` matters little.

### 5.4 Full-gate power, conditional on the realized seeds

The G3 statistical gate is the conjunction: P1 passes, P2 passes, and every one of the three
realized seeds has positive P1 and P2 point estimates. Its probability is smaller than either
per-contrast power. **Inference is conditional on the three realized training seeds**: they
are the three trained policies actually delivered, their true effects are fixed numbers, and
the gate makes a claim about them, not about a population of seeds. Appendix A, T8 simulates
true per-seed effects `(d − delta, d, d + delta)` with the stratified bootstrap for the lower
bounds:

```text
B = 160, rho_b = 0.05, rho_w = 0.10 (the model point the sizing rule maps to B = 176):
  true P2 0.06   P2 power 0.80–0.83   direction in all seeds 0.93–1.00   FULL GATE 0.80–0.82
  true P2 0.05   P2 power 0.48–0.51   direction 0.77–1.00 (weakest seed at 0.01 when delta = 0.04)  FULL GATE 0.48–0.50
  true P2 0.08   P2 power >= 0.99     FULL GATE >= 0.99
B = 160, rho_b = 0.10: true P2 0.06 -> P2 power 0.76, FULL GATE 0.76 (the rule sizes B = 304 there)
B = 256, rho_b = 0.10: true P2 0.06 -> P2 power 0.80, FULL GATE 0.80–0.81
P1 power is 1.00 in every scenario because the true P1 effect includes the init-to-library gap.
```

The direction check costs little when the weakest seed's true effect is at least 0.02, and it
is what fails when a seed sits near zero; that is the intended behaviour of a gate that
requires consistency across the delivered policies.

### 5.5 Stratum floors

The worst-stratum rule needs at least 200 paired games in a stratum before a regression may be
reported. Pooled over three seeds a per-opponent stratum holds 6B paired games and a
per-opponent-by-colour stratum 3B (Appendix A, T7d): at B = 160 those are 960 and 480, and a
single seed's per-opponent stratum holds 320. The rule counts raw paired games; their
effective count is smaller by the factors of section 5.1 and is reported beside it.

### 5.6 The frozen sizing rule (applied only at the post-calibration review)

```text
input      s2_base  = the pooled within-family sample variance of the per-base mean difference on the
                      screening slice at update 256 (10 bases per family, df = 144), taken as the LARGER
                      of the P1 and P2 values
bound      s2_upper = s2_base · df / chi2_{0.05, df}   (one-sided 95% upper confidence limit on the
                      variance; factor 1.228 at df = 144 by the Wilson–Hilferty quantile; coverage
                      0.93–0.95 in Appendix A, T5)
target     SE_target = (0.06 − 0.05) / z_0.80 = 0.0119
size       B = smallest multiple of 16 that is >= 160 and satisfies sqrt(s2_upper / B) <= SE_target
outside    a measured s2_base below the tabulated range floors at B = 160; above it, the formula still
           applies; if the resulting B exceeds the 640 reserved bases the design returns to review with
           the measured value and no confirmation size is fixed
selection  B/16 bases per family from the reserved range 410..449 by the frozen family seed (section 2.4)
```

Appendix A, T4f tabulates the rule against the measured per-base SD: 0.12 or below gives
B = 160; 0.14 gives 176; 0.16 gives 224; 0.18 gives 288; 0.20 gives 352; 0.22 gives 432;
0.24 gives 512; 0.26 gives 592; 0.28 or above returns to review. For orientation, T4g gives
the model-implied per-base SD: 0.142 at `(rho_b, rho_w) = (0.05, 0.10)` (B = 176), 0.185 at
`(0.10, 0.10)` (B = 304), 0.251 at `(0.20, 0.10)` (B = 560), and review at `(0.30, 0.10)`.
No confirmation size is fixed in this document; the rule is.

### 5.7 Multiple comparisons

The G3 decision requires both primary contrasts to pass: an intersection-union test, whose
familywise error of declaring PROCEED is at most the per-test alpha, so no adjustment is
applied to the two primaries. Seed direction is a consistency requirement on point estimates,
not a third test (its effect on full-gate power is in section 5.4); per-seed intervals are
diagnostics. Strata are descriptive except the pre-registered worst-stratum and colour rules,
which apply only where the 200-game floor holds. Decisions taken at the calibration review are
development decisions on a slice disjoint by base from the confirmation range; they never
select a checkpoint and enter no confirmation error accounting.

---

## 6. The bounded calibration stage (ends at 256 updates; mandatory review stop)

One training run per seed (three seeds, independent inits), run to exactly 256 periods and
stopped. Nothing continues automatically. Evaluation never feeds the learner.

### 6.1 What it measures

```text
throughput            exact collector games per second and seconds per period, per worker count,
                      sequential and concurrent (the preflight microbenchmark replays frozen Phase 8
                      corpus game ids through play_corpus_game with library setups: no learner, no pool,
                      nothing retained beyond the timing record); the topology (sequential or concurrent
                      seeds) and the final S are chosen from it and recorded before period 1
raw and EMA performance
                      at updates 128 and 256, raw_U and EMA_U per seed against EMA_0 (= raw_0) and the
                      seed's library arm on the screening slice (3,200 cases per arm), EVALUATION_RULES,
                      C0 as the mover; paired, stratified cluster bootstrap; the raw result is diagnostic,
                      the EMA result is telemetry: neither decides G3
learning-curve direction
                      raw_128 -> raw_256 on the screening slice, plus the teacher-regime probe every 32
                      periods: raw and EMA pools against library train-split setups under the teacher
                      schedule and TRAINING_RULES, 512 games each, outcomes never fed to the learner
                      (Appendix A, T7c: 8,192 teacher games, about two minutes)
outcome statistics    per period: completions, outcomes, ready rows, m distribution (mean, median,
                      fraction m = 0, fraction m = 1), per-row outcome variance, per-cell completed
                      counts, snapshot-age distribution, cross-period attributions, duplicates collapsed,
                      minibatches and optimizer steps per update
diversity, legality,  every pool and every evaluation sample: the section 7 statistics with their
orientation,          thresholds
concentration
raw/EMA divergence    relative L2 parameter distance per period; raw-minus-EMA performance at 128 and 256
variance inputs       s2_base for P1 and P2 (section 5.6), the diagnostics rho_b and rho_w (section 5.1),
                      and the between-library-draw variance across the three library arms
integrity             legality / orientation / attribution / non-finite / identity events; planned =
                      completed + failed + missing per period and per evaluation lane
```

Evaluator: the in-process evaluator (18.9 games/s) is used only once every R1–R11 credibility
test passes; otherwise the calibration evaluations run on the proven G1 harness (1.50 games/s),
which changes the wall time (Appendix A, T7c: 0.85 h against 10.7 h) and nothing else.

### 6.2 Stops

Only integrity failures stop a run automatically (section 7, rules A1–A9). Every learning
result — including a harmful direction on the screening slice — returns to review at 256
updates with the packet. No automatic learning-based stop rule exists because nothing
demonstrates that a 256-update raw reading predicts the eventual EMA endpoint.

### 6.3 What the review decides afterwards

The calibration packet (throughput, raw/EMA readings, curve direction, outcome statistics,
diversity, divergence, `s2_base`, `rho_b`, `rho_w`, integrity) goes to review. A later reviewed
decision determines whether continuing the same runs to 2,000 or 3,000 updates is justified
(cost only: Appendix A, T7b — at 2,000 periods 4.0 M games per seed, 96 CPU-hours, 11–37
wall-hours per seed; at 3,000 periods 6.0 M games, 144 CPU-hours, 17–56 wall-hours per seed),
fixes the confirmation B by the section 5.6 rule (or returns to review if none fits), draws the
confirmation bases by the frozen family seed, and fixes the confirmation checkpoint in the
Stage 6B contract. Continuation resumes from the period-256 checkpoint, which requires the S29
resume proof on the production device before any further period.

### 6.4 The confirmation stage (for orientation only; not designed here)

The EMA after the fixed final update decides; intermediate points are telemetry; no peak
selection. Primary contrasts P1 and P2 pooled over seeds, stratified cluster bootstrap, 10,000
replicates, frozen seed: each passes only with lower bound > 0 and point ≥ 0.05. Direction
positive in each realized seed for both contrasts; colour rule on P1; worst-stratum rule on P2
with the 200-game floor; integrity gates; the raw actor reported beside the EMA as a
diagnostic. PROCEED when everything passes; STOP when a valid run fails P1 with no isolated
defect; REVISE when P1 passes and P2 fails (common contract section 12: investigate estimator
variance, pool reuse, symmetry, evaluation distribution; do not integrate); BLOCKED when a
dependency prevents valid evidence.

---

## 7. Diversity and concentration thresholds (exact, with their evidence; train-split references)

Statistics are computed on canonical boards (rank 0 = back rank, rank 3 = front row) for every
1,024-setup pool and every 4,096-setup evaluation sample; "symmetrized" file shares average a
file with its mirror, which is what the 50% reflection produces in play. References: the
library **train split** (Appendix A, T1: 6,400 curated bases, per-family range) and the fresh
model (T2: 24 tracked G2 generation samples of 4,096; the 384 untracked 1,024-setup pool
records of the two G2 assays corroborate them with distinct classes 1,024/1,024 in every pool,
played file share at most 0.173, mirror-asymmetry z at most 3.71, reflected fraction
0.460–0.546, and 12 immediately terminal setups in total).

Automatic stops (integrity; the run halts at the period boundary with its checkpoint):

```text
A1  legality failures > 0            fresh model: 0 in every sample and pool; method map S02
A2  orientation failures > 0         0 everywhere; the Phase 12 defect class (77.0% front-row flags) is
                                     caught at the orientation gate itself (S07)
A3  attribution failures > 0         the buffer raises (test: unknown setup is fatal, never dropped)
A4  non-finite events > 0            G2: 0 in 384 updates
A5  checkpoint identity mismatch     raw / optimizer / EMA digests against the manifest (S29)
A6  distinct reflection classes < 922 of 1,024 in any pool (at least 10% duplicates); fresh model
                                     1,024/1,024 in all 384 pools; ten percent duplicates at this
                                     vocabulary has no benign explanation
A7  mirror-asymmetry z > 5.0 for any file pair, z = |n_f − n_(9−f)| / sqrt(n_f + n_(9−f));
                                     observed at most 2.50 (tracked) and 3.71 (untracked);
                                     a seeded fair reflection cannot reach 5.0 except by failure
A8  front-row flag share >= 0.60     the orientation-defect class: Phase 12 measured 0.770 against
                                     0.0177 when correct; square-uniform placement gives 0.25 and the
                                     most front-heavy curated train family 0.2925
A9  accounting failure               planned != completed + failed + missing in any period or lane
```

Review flags (recorded and reported; no automatic action):

```text
R1  distinct reflection classes < 1,014 of 1,024 (at least 1% duplicates); fresh baseline 0 duplicates
R2  max symmetrized flag-file share > 0.25     fresh 0.108–0.158; train library overall 0.1095; a curated
                                               corner family reaches 0.50, so concentration alone is not a defect
R3  front-row flag share > 0.2925              the most front-heavy curated train family (irregular_high_entropy)
R4  back-rank flag share < 0.1800              the least back-heavy curated train family
R5  mean bombs adjacent to the flag > 2.5750   the most fortress-like curated train family (near_corner_flag_fortress)
R6  front-row bomb share > 0.2522              the most bomb-forward curated train family (high_bomb_placement)
R7  max symmetrized bomb-file share > 0.20     twice the square-uniform 0.10; every curated train family <= 0.1533
R8  mean sequence information < 0.80 x the seed's update-0 value
                                               Phase 17 fell 22% over 12 hours without collapse; fresh
                                               samples rose 0.8–4.5% over 64 synthetic updates
R9  reflected fraction outside [0.44, 0.56] in any 1,024 pool (about 3.8 binomial SD; observed 0.460–0.546)
R10 immediately terminal setups > 10 in any pool (observed 12 in 384 pools)
```

The update-0 values of every statistic are recorded per seed so that drift is read against
the model's own start as well as against the references.

---

## 8. Evaluation accounting

Appendix A, T7c and T7d. Screening slice, 3,200 cases per arm, EMA evaluated at both 128 and
256, one library arm per seed:

```text
arms                    raw_128, EMA_128, raw_256, EMA_256 per seed (12) + EMA_0 per seed (3) + L_k per seed (3)
                        = 18 arms -> 57,600 games
wall time               0.85 h on the in-process harness (only after R1–R11 pass); 10.7 h on the G1 harness
teacher-regime probe    8 checkpoints x 2 models x 512 games = 8,192 teacher games (about 2.3 minutes at 60
                        games/s), included
```

Confirmation slice (not opened in the calibration stage; B fixed only at the review): 9 arms
x 20B games — 28,800 at B = 160 (0.42 h in-process, 5.35 h on the G1 harness), 46,080 at
B = 256, 115,200 at B = 640 (Appendix A, T7d). Total evaluation if the confirmation later runs
at B = 160: 86,400 C1-mover games plus the probe.

---

## 9. What exists and what must be built

```text
exists     setup model / sampling / buffer / trainer / EMA / checkpoints (stratego/training/phase18/*;
           86/86 evaluator and 131/131 setup tests at ccddceda); the synthetic assay driver; reflection-
           class identity (stratego/setups/identity.py); the paired bootstrap (evaluation/phase18/
           noninferiority.py); the power module; the library validation-split draw path (Phase 15/16);
           the bank separation audit (confirmation_bank.py); teacher builders and play_corpus_game
           (rule_population.py); MatchSpec pairing and receipts (the G1 script); C0; the opponent roster
G3-ENG-01  asynchronous pool-driven teacher-schedule collector (section 3.2): slots, fixed order,
           cyclic cells, explicit pool setups through the orientation boundary, attribution through
           the existing buffer, TRAINING_RULES, receipts, per-period accounting
G3-ENG-02  Phase 18 roster evaluator with lane-aware own-setup resolution (EMA sample or per-seed
           library draw per case, opponent setup and seeds fixed per case) meeting R1–R11 including the
           R9 retry regression; the in-process path is used only after every test passes, the G1
           harness otherwise
G3-ENG-03  development-pack builder: screening slice, the reserved range, the frozen per-family seed
           for confirmation bases, and the class-separation audit against the train-split own-setup
           source over recorded fingerprints
G3-ENG-04  run driver: checkpoints every 32 periods and at 128/256, exact CPU resume proof (S29),
           cooperative termination (S35), telemetry, screening evaluation hook, the teacher-regime
           probe, binding ledger, runtime under output/phase18/runtime/
G3-ENG-05  preflight microbenchmarks: collector games per second by worker count and topology
           (corpus-id replay), evaluation games per second, per-period wall; topology and S frozen
           from it and recorded before period 1
G3-ENG-06  analysis and packet scripts: stratified cluster bootstrap with the finite-stratum
           rescaling, direct per-base variance and its upper bound, the rho decomposition, strata,
           section 7 statistics, the sizing rule, calibration packet
```

Every item is engineering with tests; none requires a method change. The execution
instruction (Stage 6C) must place a mandatory preflight/review stop between these items and
the first collection game.

---

## 10. Risks

| risk | evidence | handling |
|---|---|---|
| Transfer: outcomes from teacher play, evaluation with the C1 mover | sections 2.1–2.2 | the production mechanism; P2 measures the transfer; the teacher-seen / unseen stratum reports it |
| Zero-signal cells: 9 of 100 cells at least 95% draws, 23 at least 80% | section 2.1 | counted in the cost; schedule unchanged; per-cell completions and Var(z) recorded |
| Collection under battleless 100, evaluation under 200 | P18-A001 invariant | both tokens on every receipt; no power claim rests on it |
| The EMA's response to training is unmeasured in Stratego | section 4 | measured at 128 and 256; no inference from parameter aging; continuation decided at review |
| Fresh init about 0.08 below the library, so P2 needs about 0.13 of EMA gain | section 2.7 | the calibration reads the remaining gap; no budget is promised |
| Per-base variance unknown; bases are the scarce resource | section 5 | measured on the screening slice; B fixed by the frozen rule with a 95% upper bound; review if none fits |
| The screening variance at 256 updates may not equal the confirmation's | section 5.6 | the upper bound and the larger of P1/P2 are used; the reserve reaches a per-base SD of 0.26 |
| Instrument SD may differ from the Phase 17 measurement | section 2.7 | sizing uses the measured slice variance, not the Phase 17 constant |
| Throughput unmeasured for the exact collector | section 2.6 | mandatory preflight; topology and S frozen from it |
| MPS is not bitwise reproducible; CPU greedy evaluation is | P18-D002/D003 | learner and games on CPU; replay of pool and EMA digests as in G2 |
| Diversity collapse, orientation defect, attribution loss | Phase 17, Phase 12, S21 | section 7 stops and flags; buffer raises on attribution |
| Concurrent seeds contend for the 10 performance cores | section 2.6 | preflight measures parallel efficiency; topology frozen |
| Evaluator credibility R1–R11 unmet for a roster evaluator | section 9 | G3-ENG-02 with tests; the G1 harness until then |
| Structural exposure of reserved and test bases in the first Stage 6A script | section 2.5 | recorded as feature exposure; references recomputed from the train split; no outcome observed |

---

## 11. What is decided where

Decided by the operator and the reviewer (sections 0 and 0.1): the collection regime and
rules, the evaluation rules, per-seed inits, the EMA as decision model, the primary-contrast
rule, the deferral of the unusual stratum, the evaluator preference and fallback, the powered
effect 0.06, the minimum B of 160, per-seed library arms, seeded base selection, T = 202, the
probe's inclusion, and that B is fixed only at the post-calibration review.

Decided at the preflight (recorded before period 1): the execution topology and the final S
under the four-outcomes-per-setup constraint.

Decided at the post-calibration review (with the calibration packet): whether to continue and
to which budget; the confirmation B by the frozen rule or a return to review; the confirmation
checkpoint. To be frozen in the Stage 6B contract: the G3 seed namespace string, the exact
probe schedule, and the confirmation-base permutation once B is known.

---

## 12. What this stage did not do

No game was played, no pool sampled, no model built, trained or loaded, no confirmation board
materialized or played, no confirmation outcome observed, no sealed artifact opened, no
contract frozen, no accepted artifact changed, nothing pushed, nothing amended or rewritten.
The only executable added reads tracked files and writes one JSON of tables; its access to
library content is recorded in section 2.5.

---

## Appendix B — checks run for this commit

```text
scripts/phase18_g3_stage6a_tables.py --check     recomputes every table and compares with the committed JSON
tests/training/phase18/test_setup_buffer.py       buffer semantics (S09, S10, S21–S23, S26)
tests/training/phase18/test_setup_learning.py     trainer, EMA, optimizer-step accounting
tests/training/phase18/test_reference_oracle.py   canned parity oracle
python -m json.tool                               JSON validity of the committed tables file
git diff --check                                  no trailing whitespace, no whitespace-only lines at EOF
```

The committed JSON `reports/phase18/g3_design/phase18_g3_stage6a_tables_v1.json` has SHA-256
`127f8e10648355ae68c06d61bf308cdc313a6457d216581ccea096abfa9a5e50`.

---

## Appendix A — tables (verbatim output of `scripts/phase18_g3_stage6a_tables.py --check`)

Reproduce with:

```bash
.venv/bin/python scripts/phase18_g3_stage6a_tables.py --check
```

## T1 Library structural references, TRAIN split only (setup_library_v1 base_index 0..399; canonical rank 0 = back rank, 3 = front row)

Lines scanned 8000 (split tags {'train': 6400, 'validation': 800, 'test': 800}); rows decoded and used 6400 (train only).

| statistic | train overall | family min (family) | family max (family) |
|---|---|---|---|
| front_row_flag_share | 0.0183 | 0.0000 (aggressive_high_rank_front) | 0.2925 (irregular_high_entropy) |
| back_rank_flag_share | 0.7705 | 0.1800 (irregular_high_entropy) | 1.0000 (balanced_conventional) |
| max_flag_file_share_symmetrized | 0.1095 | 0.1062 (aggressive_high_rank_front) | 0.5000 (corner_flag_fortress) |
| bombs_adjacent_to_flag_mean | 1.0092 | 0.0000 (lightly_defended_deceptive_flag) | 2.5750 (near_corner_flag_fortress) |
| front_row_bomb_share | 0.0823 | 0.0362 (balanced_conventional) | 0.2522 (high_bomb_placement) |
| back_rank_bomb_share | 0.2165 | 0.0460 (high_bomb_placement) | 0.2858 (central_back_flag_fortress) |
| max_bomb_file_share_symmetrized | 0.1010 | 0.1031 (miner_preservation) | 0.1533 (central_back_flag_fortress) |

## T2 Fresh-model baselines from the tracked G2 result files (24 generation samples of 4,096 setups)

| statistic | min | max |
|---|---|---|
| distinct_class_fraction | 1.0000 | 1.0000 |
| max_flag_file_share_played | 0.1082 | 0.1599 |
| max_flag_file_share_symmetrized | 0.1077 | 0.1578 |
| max_mirror_asymmetry_z | 0.6612 | 2.5017 |
| mean_sequence_information_nats | 62.3075 | 66.6954 |
| reflected_fraction | 0.4934 | 0.5015 |
| raw final / initial sequence-information ratio | 1.0079 | 1.0447 |

immediately terminal 0, legality failures 0, orientation failures 0 over all samples.

| G2 result file | EMA fraction of gap closed | raw fraction of gap closed | EMA retained initial-parameter fraction |
|---|---|---|---|
| phase18_g2_seed_1_result_v1.json | 0.0028 | 0.2092 | 0.9380 |
| phase18_g2_seed_2_result_v1.json | 0.0052 | 0.1850 | 0.9380 |
| phase18_g2_seed_3_result_v1.json | 0.0035 | 0.1482 | 0.9380 |
| phase18_g2_raw_confirmation_seed_1_result_v1.json | 0.0010 | 0.1065 | 0.9380 |
| phase18_g2_raw_confirmation_seed_2_result_v1.json | 0.0012 | 0.1001 | 0.9380 |
| phase18_g2_raw_confirmation_seed_3_result_v1.json | 0.0020 | 0.0960 | 0.9380 |

## T3 Frozen constants and their tracked sources

| constant | value | source |
|---|---|---|
| SD_PAIRED | 0.5391 | phase18_evaluation_contract_v1.json:power_and_precision_plan.measured_inputs.own_setup_instrument.sd_of_paired_per_board_difference |
| TIE_FRACTION | 0.5940 | phase18_evaluation_contract_v1.json:...own_setup_instrument.tie_fraction |
| SD_SINGLE_LANE | 0.4236 | phase18_evaluation_contract_v1.json:...measured_inputs.single_lane_sd |
| MARGIN | 0.0500 | phase18_evaluation_contract_v1.json:...predeclared_practical_margins.value |
| EMA_DECAY | 0.9990 | stratego/training/phase18/setup_contract.py:SETUP_EMA_DECAY |
| POOL_SIZE | 1024 | stratego/training/phase18/setup_contract.py:SETUP_POOL_SIZE |
| BATCH_SIZE | 1024 | stratego/training/phase18/setup_contract.py:SETUP_BATCH_SIZE |
| EPOCHS_PER_UPDATE | 5 | stratego/training/phase18/setup_contract.py:SETUP_EPOCHS_PER_UPDATE |
| TRAINING_BATTLELESS | 100 | stratego/engine/constants.py:TRAINING_RULES |
| TRAINING_ABSOLUTE | 4000 | stratego/engine/constants.py:TRAINING_RULES |
| EVALUATION_BATTLELESS | 200 | stratego/engine/constants.py:EVALUATION_RULES |
| EVALUATION_ABSOLUTE | 4000 | stratego/engine/constants.py:EVALUATION_RULES |
| MEAN_PLIES_TRAINING_RULES | 258.8130 | reports/phase_8_data/agent_02_corpus_manifest.json:decision_totals.per_split.train.mean_plies |
| SECONDS_PER_TEACHER_GAME | 0.0865 | reports/phase_4_data/agent_04_baseline_league_raw.csv: mean wall_clock_seconds over 44544 rule-vs-rule games |
| TEACHER_GAMES_MEASURED | 44544 | same file, row count |
| G1_HARNESS_GAMES_PER_SECOND | 1.4959 | reports/phase18/g1_random_confirmation/run_v1.json: completed games / seconds |
| PHASE17_HARNESS_GAMES_PER_SECOND | 18.9274 | reports/phase17/local_eval/results/*.result.json: games / (finished_utc - started_utc), 25 candidates |
| C0_CHECKPOINT_BYTES | 10459947 | reports/phase18/phase18_g1_random_confirmation_contract_v1.json:checkpoints.candidate.bytes |
| C0_CHECKPOINT_SHA256 | 460a246be32b821a6d6d7feb928b272a4be1014ff55053f329980e21e3be074c | same file, checkpoints.candidate.sha256 |
| LIBRARY_SPLIT_COUNTS | {'test': 800, 'train': 6400, 'validation': 800} | data/setups/setup_library_v1_manifest.json:split_counts |
| LIBRARY_SPLIT_RULE | base_index 0..399 train, 400..449 validation, 450..499 test | data/setups/setup_library_v1_manifest.json:split_rule |
| LIBRARY_DIGEST | 7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777 | data/setups/setup_library_v1_manifest.json:library_content_digest |

Derived: V = SD_paired^2 = 0.2906; implied same-case cross-arm correlation 0.1902; SE target for d80 <= 0.06: point criterion 0.0119, lower-bound criterion 0.0214, binding 0.0119; retention 21 periods = ceil(absolute limit / T) + 1.

## T4a Resolution per effective independent paired unit

| n_eff | SE | 95% half-width | MDE (lower>0, 80%) | d80 combined rule | P(pass at 0.05) | P(pass at 0.06) | P(pass at 0.08) |
|---|---|---|---|---|---|---|---|
| 256 | 0.0337 | 0.0660 | 0.0944 | 0.0944 | 0.317 | 0.429 | 0.661 |
| 512 | 0.0238 | 0.0467 | 0.0667 | 0.0701 | 0.500 | 0.663 | 0.896 |
| 913 | 0.0178 | 0.0350 | 0.0500 | 0.0650 | 0.500 | 0.712 | 0.954 |
| 1024 | 0.0168 | 0.0330 | 0.0472 | 0.0642 | 0.500 | 0.724 | 0.963 |
| 2048 | 0.0119 | 0.0233 | 0.0334 | 0.0600 | 0.500 | 0.799 | 0.994 |
| 4096 | 0.0084 | 0.0165 | 0.0236 | 0.0571 | 0.500 | 0.882 | 1.000 |
| 8192 | 0.0060 | 0.0117 | 0.0167 | 0.0550 | 0.500 | 0.953 | 1.000 |

## T4b Frozen design (three seeds, per-seed library arms; P1 and P2): n_eff / SE / d80, rho_w = 0.10

| B bases | cases/arm | games (9 arms) | rho_b=0.0 | rho_b=0.02 | rho_b=0.05 | rho_b=0.1 | rho_b=0.2 | rho_b=0.3 |
|---|---|---|---|---|---|---|---|---|
| 160 | 3200 | 28800 | 8000 / 0.0060 / 0.0551 | 4034 / 0.0085 / 0.0571 | 2313 / 0.0112 / 0.0594 | 1352 / 0.0147 / 0.0623 | 738 / 0.0198 / 0.0667 | 508 / 0.0239 / 0.0701 |
| 208 | 4160 | 37440 | 10400 / 0.0053 / 0.0544 | 5244 / 0.0074 / 0.0563 | 3007 / 0.0098 / 0.0583 | 1758 / 0.0129 / 0.0608 | 960 / 0.0174 / 0.0646 | 660 / 0.0210 / 0.0677 |
| 256 | 5120 | 46080 | 12800 / 0.0048 / 0.0540 | 6454 / 0.0067 / 0.0556 | 3701 / 0.0089 / 0.0575 | 2163 / 0.0116 / 0.0598 | 1182 / 0.0157 / 0.0632 | 813 / 0.0189 / 0.0659 |
| 320 | 6400 | 57600 | 16000 / 0.0043 / 0.0536 | 8067 / 0.0060 / 0.0551 | 4627 / 0.0079 / 0.0567 | 2704 / 0.0104 / 0.0587 | 1477 / 0.0140 / 0.0618 | 1016 / 0.0169 / 0.0642 |
| 400 | 8000 | 72000 | 20000 / 0.0038 / 0.0532 | 10084 / 0.0054 / 0.0545 | 5783 / 0.0071 / 0.0560 | 3380 / 0.0093 / 0.0578 | 1846 / 0.0125 / 0.0606 | 1270 / 0.0151 / 0.0627 |
| 480 | 9600 | 86400 | 24000 / 0.0035 / 0.0529 | 12101 / 0.0049 / 0.0541 | 6940 / 0.0065 / 0.0554 | 4056 / 0.0085 / 0.0571 | 2215 / 0.0115 / 0.0596 | 1524 / 0.0138 / 0.0616 |
| 560 | 11200 | 100800 | 28000 / 0.0032 / 0.0527 | 14118 / 0.0045 / 0.0538 | 8096 / 0.0060 / 0.0550 | 4732 / 0.0078 / 0.0566 | 2585 / 0.0106 / 0.0589 | 1778 / 0.0128 / 0.0608 |
| 640 | 12800 | 115200 | 32000 / 0.0030 / 0.0525 | 16134 / 0.0042 / 0.0536 | 9253 / 0.0056 / 0.0547 | 5408 / 0.0073 / 0.0562 | 2954 / 0.0099 / 0.0583 | 2032 / 0.0120 / 0.0601 |

## T4c One seed alone: n_eff / SE, rho_w = 0.10

| B bases | rho_b=0.0 | rho_b=0.02 | rho_b=0.05 | rho_b=0.1 | rho_b=0.2 | rho_b=0.3 |
|---|---|---|---|---|---|---|
| 160 | 3200 / 0.0095 | 2319 / 0.0112 | 1641 / 0.0133 | 1103 / 0.0162 | 667 / 0.0209 | 478 / 0.0247 |
| 208 | 4160 / 0.0084 | 3014 / 0.0098 | 2133 / 0.0117 | 1434 / 0.0142 | 867 / 0.0183 | 621 / 0.0216 |
| 256 | 5120 / 0.0075 | 3710 / 0.0089 | 2626 / 0.0105 | 1766 / 0.0128 | 1067 / 0.0165 | 764 / 0.0195 |
| 320 | 6400 / 0.0067 | 4638 / 0.0079 | 3282 / 0.0094 | 2207 / 0.0115 | 1333 / 0.0148 | 955 / 0.0174 |
| 400 | 8000 / 0.0060 | 5797 / 0.0071 | 4103 / 0.0084 | 2759 / 0.0103 | 1667 / 0.0132 | 1194 / 0.0156 |
| 480 | 9600 / 0.0055 | 6957 / 0.0065 | 4923 / 0.0077 | 3310 / 0.0094 | 2000 / 0.0121 | 1433 / 0.0142 |
| 560 | 11200 / 0.0051 | 8116 / 0.0060 | 5744 / 0.0071 | 3862 / 0.0087 | 2333 / 0.0112 | 1672 / 0.0132 |
| 640 | 12800 / 0.0048 | 9275 / 0.0056 | 6564 / 0.0067 | 4414 / 0.0081 | 2667 / 0.0104 | 1910 / 0.0123 |

## T4d Why per-seed library arms (B = 160, rho_w = 0.10): effective units per design

| rho_b | per-seed arms n_eff (28,800 games) | shared arm n_eff (22,400 games) |
|---|---|---|
| 0.0 | 8000 | 4571 |
| 0.02 | 4034 | 2945 |
| 0.05 | 2313 | 1920 |
| 0.1 | 1352 | 1215 |
| 0.2 | 738 | 701 |
| 0.3 | 508 | 492 |

## T4e rho_w sensitivity (frozen design)

| B | rho_b | rho_w | n_eff | SE | d80 |
|---|---|---|---|---|---|
| 160 | 0.05 | 0.0 | 2430 | 0.0109 | 0.0592 |
| 160 | 0.05 | 0.1 | 2313 | 0.0112 | 0.0594 |
| 160 | 0.05 | 0.3 | 2110 | 0.0117 | 0.0599 |
| 160 | 0.1 | 0.0 | 1391 | 0.0145 | 0.0622 |
| 160 | 0.1 | 0.1 | 1352 | 0.0147 | 0.0623 |
| 160 | 0.1 | 0.3 | 1280 | 0.0151 | 0.0627 |
| 256 | 0.05 | 0.0 | 3889 | 0.0086 | 0.0573 |
| 256 | 0.05 | 0.1 | 3701 | 0.0089 | 0.0575 |
| 256 | 0.05 | 0.3 | 3376 | 0.0093 | 0.0578 |
| 256 | 0.1 | 0.0 | 2226 | 0.0114 | 0.0596 |
| 256 | 0.1 | 0.1 | 2163 | 0.0116 | 0.0598 |
| 256 | 0.1 | 0.3 | 2048 | 0.0119 | 0.0600 |

## T4f Frozen sizing rule: measured per-base SD on the screening slice -> confirmation B (df = 144, one-sided 95% upper bound on the variance, SE target 0.0119)

| measured per-base SD | s2 | upper-bound factor | s2_upper | B unrounded | B (multiple of 16, >= 160) | decision |
|---|---|---|---|---|---|---|
| 0.070 | 0.0049 | 1.228 | 0.0060 | 42.6 | 160 | size |
| 0.100 | 0.0100 | 1.228 | 0.0123 | 87.0 | 160 | size |
| 0.120 | 0.0144 | 1.228 | 0.0177 | 125.2 | 160 | size |
| 0.140 | 0.0196 | 1.228 | 0.0241 | 170.5 | 176 | size |
| 0.160 | 0.0256 | 1.228 | 0.0314 | 222.7 | 224 | size |
| 0.180 | 0.0324 | 1.228 | 0.0398 | 281.8 | 288 | size |
| 0.200 | 0.0400 | 1.228 | 0.0491 | 347.9 | 352 | size |
| 0.220 | 0.0484 | 1.228 | 0.0594 | 421.0 | 432 | size |
| 0.240 | 0.0576 | 1.228 | 0.0707 | 501.0 | 512 | size |
| 0.260 | 0.0676 | 1.228 | 0.0830 | 588.0 | 592 | size |
| 0.280 | 0.0784 | 1.228 | 0.0963 | 681.9 | - | REVIEW: no valid size within the 640 reserved bases |
| 0.300 | 0.0900 | 1.228 | 0.1105 | 782.8 | - | REVIEW: no valid size within the 640 reserved bases |

## T4g Sizing-rule reference points implied by the variance model (for orientation; the rule uses the measured value)

| rho_b | rho_w | model per-base SD | s2_upper | B unrounded | B | decision |
|---|---|---|---|---|---|---|
| 0.0 | 0.0 | 0.0696 | 0.0059 | 42.1 | 160 | size |
| 0.02 | 0.1 | 0.1074 | 0.0142 | 100.3 | 160 | size |
| 0.05 | 0.1 | 0.1418 | 0.0247 | 174.8 | 176 | size |
| 0.1 | 0.1 | 0.1854 | 0.0422 | 299.1 | 304 | size |
| 0.2 | 0.1 | 0.2509 | 0.0773 | 547.7 | 560 | size |
| 0.3 | 0.1 | 0.3026 | 0.1124 | 796.3 | - | REVIEW: no valid size within the 640 reserved bases |

## T5 Monte Carlo: variance model, stratified cluster bootstrap, direct per-base variance, rho decomposition (fixed family effect 0.05 V)

| B | rho_b | rho_w | P1 SE sim / analytic | P2 SE sim / analytic | single-seed SE sim / analytic | naive SE (60B) | stratified boot SE / coverage | global boot SE / coverage | direct s2 mean / true | upper-bound factor / coverage | rho_b est mean +- sd | rho_w est mean +- sd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 160 | 0.05 | 0.1 | 0.0113 / 0.0112 | 0.0112 / 0.0112 | 0.0131 / 0.0133 | 0.0055 | 0.0112 / 0.943 | 0.0136 / 0.977 | 0.0204 / 0.0201 | 1.228 / 0.953 | 0.044 +- 0.008 | 0.100 +- 0.010 |
| 256 | 0.1 | 0.1 | 0.0117 / 0.0116 | 0.0117 / 0.0116 | 0.0129 / 0.0128 | 0.0043 | 0.0116 / 0.960 | 0.0130 / 0.983 | 0.0344 / 0.0344 | 1.170 / 0.930 | 0.093 +- 0.009 | 0.101 +- 0.008 |
| 160 | 0.0 | 0.0 | 0.0055 / 0.0055 | 0.0054 / 0.0055 | 0.0093 / 0.0095 | 0.0055 | 0.0055 / 0.943 | 0.0093 / 1.000 | 0.0048 / 0.0048 | 1.228 / 0.930 | -0.002 +- 0.003 | -0.000 +- 0.011 |

## T6 EMA parameter aging (decay 0.999, time constant 1000 updates) — parameters only

| U | retained initial-parameter fraction | weight on the raw trajectory | weight on the last 64 updates |
|---|---|---|---|
| 64 | 0.9380 | 0.0620 | 0.0620 |
| 128 | 0.8798 | 0.1202 | 0.0620 |
| 256 | 0.7740 | 0.2260 | 0.0620 |
| 500 | 0.6064 | 0.3936 | 0.0620 |
| 1000 | 0.3677 | 0.6323 | 0.0620 |
| 2000 | 0.1352 | 0.8648 | 0.0620 |
| 3000 | 0.0497 | 0.9503 | 0.0620 |

## T7a Collector per period: S = 2560 slots (provisional) x T = 202 plies / 258.8 mean plies; retention 21 periods

expected completions per period 1998, outcomes 3996, mean outcomes per generated setup 3.90 (S giving exactly 4.0: 2624); CPU s per period 173; wall s per period at 30/60/100 games/s: 67 / 33 / 20

## T7b Collection cost per seed by period budget (cost only; no play-strength inference)

| periods | games/seed | CPU h/seed | wall h/seed @30 | @60 | @100 | three seeds @30 | @60 | @100 | receipts MB | game rows MB | checkpoints MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 0.51 M | 12.3 | 4.7 | 2.4 | 1.4 | 14.2 | 7.1 | 4.3 | 82 | 153 | 103 |
| 1000 | 2.00 M | 48.0 | 18.5 | 9.3 | 5.6 | 55.5 | 27.8 | 16.7 | 320 | 599 | 400 |
| 2000 | 4.00 M | 96.0 | 37.0 | 18.5 | 11.1 | 111.0 | 55.5 | 33.3 | 639 | 1199 | 800 |
| 3000 | 5.99 M | 144.1 | 55.5 | 27.8 | 16.7 | 166.5 | 83.3 | 50.0 | 959 | 1798 | 1200 |

## T7c Screening-slice evaluation accounting (160 bases x 20 cases = 3,200 cases per arm; harness rates 18.9 and 1.50 games/s)

| arms | cases/arm | games | hours in-process (after R1–R11) | hours G1 harness (fallback) | probe teacher games | probe minutes @60 games/s |
|---|---|---|---|---|---|---|
| 18 | 3200 | 57600 | 0.85 | 10.70 | 8192 | 2.3 |

## T7d Confirmation-slice accounting by B (9 arms; not opened in the calibration stage)

| B | cases/arm | games | hours in-process | hours G1 harness | paired games per opponent (pooled) | per opponent x colour (pooled) | per family (pooled) | per opponent (one seed) |
|---|---|---|---|---|---|---|---|---|
| 160 | 3200 | 28800 | 0.42 | 5.35 | 960 | 480 | 600 | 320 |
| 208 | 4160 | 37440 | 0.55 | 6.95 | 1248 | 624 | 780 | 416 |
| 256 | 5120 | 46080 | 0.68 | 8.56 | 1536 | 768 | 960 | 512 |
| 320 | 6400 | 57600 | 0.85 | 10.70 | 1920 | 960 | 1200 | 640 |
| 400 | 8000 | 72000 | 1.06 | 13.37 | 2400 | 1200 | 1500 | 800 |
| 480 | 9600 | 86400 | 1.27 | 16.04 | 2880 | 1440 | 1800 | 960 |
| 560 | 11200 | 100800 | 1.48 | 18.72 | 3360 | 1680 | 2100 | 1120 |
| 640 | 12800 | 115200 | 1.69 | 21.39 | 3840 | 1920 | 2400 | 1280 |

## T8 Monte Carlo full-gate power, conditional on three realized seeds with true effects (d - delta, d, d + delta)

| rho_b | rho_w | B | true P1 | true P2 | seed spread | P1 power | P2 power | direction in all seeds | FULL GATE | analytic P2 pass prob |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.1 | 160 | 0.13 | 0.05 | 0.0 | 1.000 | 0.482 | 0.999 | 0.482 | 0.500 |
| 0.05 | 0.1 | 160 | 0.13 | 0.05 | 0.02 | 1.000 | 0.480 | 0.985 | 0.480 | 0.500 |
| 0.05 | 0.1 | 160 | 0.13 | 0.05 | 0.04 | 1.000 | 0.512 | 0.769 | 0.499 | 0.500 |
| 0.05 | 0.1 | 160 | 0.14 | 0.06 | 0.0 | 1.000 | 0.799 | 1.000 | 0.799 | 0.814 |
| 0.05 | 0.1 | 160 | 0.14 | 0.06 | 0.02 | 1.000 | 0.805 | 0.997 | 0.805 | 0.814 |
| 0.05 | 0.1 | 160 | 0.14 | 0.06 | 0.04 | 1.000 | 0.831 | 0.931 | 0.824 | 0.814 |
| 0.05 | 0.1 | 160 | 0.16 | 0.08 | 0.0 | 1.000 | 0.993 | 1.000 | 0.993 | 0.996 |
| 0.05 | 0.1 | 160 | 0.16 | 0.08 | 0.02 | 1.000 | 0.994 | 1.000 | 0.994 | 0.996 |
| 0.05 | 0.1 | 160 | 0.16 | 0.08 | 0.04 | 1.000 | 0.996 | 0.997 | 0.994 | 0.996 |
| 0.05 | 0.1 | 160 | 0.18 | 0.1 | 0.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.05 | 0.1 | 160 | 0.18 | 0.1 | 0.02 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.05 | 0.1 | 160 | 0.18 | 0.1 | 0.04 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.05 | 0.1 | 256 | 0.13 | 0.05 | 0.0 | 1.000 | 0.499 | 1.000 | 0.499 | 0.500 |
| 0.05 | 0.1 | 256 | 0.13 | 0.05 | 0.02 | 1.000 | 0.514 | 0.998 | 0.514 | 0.500 |
| 0.05 | 0.1 | 256 | 0.13 | 0.05 | 0.04 | 1.000 | 0.479 | 0.820 | 0.475 | 0.500 |
| 0.05 | 0.1 | 256 | 0.14 | 0.06 | 0.0 | 1.000 | 0.863 | 1.000 | 0.863 | 0.870 |
| 0.05 | 0.1 | 256 | 0.14 | 0.06 | 0.02 | 1.000 | 0.864 | 1.000 | 0.864 | 0.870 |
| 0.05 | 0.1 | 256 | 0.14 | 0.06 | 0.04 | 1.000 | 0.855 | 0.959 | 0.853 | 0.870 |
| 0.05 | 0.1 | 256 | 0.16 | 0.08 | 0.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.05 | 0.1 | 256 | 0.16 | 0.08 | 0.02 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.05 | 0.1 | 256 | 0.16 | 0.08 | 0.04 | 1.000 | 0.999 | 1.000 | 0.999 | 1.000 |
| 0.05 | 0.1 | 256 | 0.18 | 0.1 | 0.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.05 | 0.1 | 256 | 0.18 | 0.1 | 0.02 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.05 | 0.1 | 256 | 0.18 | 0.1 | 0.04 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.1 | 0.1 | 160 | 0.13 | 0.05 | 0.0 | 1.000 | 0.515 | 1.000 | 0.515 | 0.500 |
| 0.1 | 0.1 | 160 | 0.13 | 0.05 | 0.02 | 1.000 | 0.507 | 0.971 | 0.507 | 0.500 |
| 0.1 | 0.1 | 160 | 0.13 | 0.05 | 0.04 | 1.000 | 0.504 | 0.741 | 0.498 | 0.500 |
| 0.1 | 0.1 | 160 | 0.14 | 0.06 | 0.0 | 1.000 | 0.766 | 1.000 | 0.766 | 0.752 |
| 0.1 | 0.1 | 160 | 0.14 | 0.06 | 0.02 | 1.000 | 0.758 | 0.990 | 0.758 | 0.752 |
| 0.1 | 0.1 | 160 | 0.14 | 0.06 | 0.04 | 1.000 | 0.763 | 0.910 | 0.761 | 0.752 |
| 0.1 | 0.1 | 160 | 0.16 | 0.08 | 0.0 | 1.000 | 0.977 | 1.000 | 0.977 | 0.980 |
| 0.1 | 0.1 | 160 | 0.16 | 0.08 | 0.02 | 1.000 | 0.987 | 1.000 | 0.987 | 0.980 |
| 0.1 | 0.1 | 160 | 0.16 | 0.08 | 0.04 | 1.000 | 0.980 | 0.992 | 0.979 | 0.980 |
| 0.1 | 0.1 | 160 | 0.18 | 0.1 | 0.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.1 | 0.1 | 160 | 0.18 | 0.1 | 0.02 | 1.000 | 0.999 | 1.000 | 0.999 | 1.000 |
| 0.1 | 0.1 | 160 | 0.18 | 0.1 | 0.04 | 1.000 | 0.999 | 1.000 | 0.999 | 1.000 |
| 0.1 | 0.1 | 256 | 0.13 | 0.05 | 0.0 | 1.000 | 0.482 | 1.000 | 0.482 | 0.500 |
| 0.1 | 0.1 | 256 | 0.13 | 0.05 | 0.02 | 1.000 | 0.508 | 0.987 | 0.508 | 0.500 |
| 0.1 | 0.1 | 256 | 0.13 | 0.05 | 0.04 | 1.000 | 0.509 | 0.801 | 0.505 | 0.500 |
| 0.1 | 0.1 | 256 | 0.14 | 0.06 | 0.0 | 1.000 | 0.800 | 1.000 | 0.800 | 0.806 |
| 0.1 | 0.1 | 256 | 0.14 | 0.06 | 0.02 | 1.000 | 0.804 | 0.999 | 0.804 | 0.806 |
| 0.1 | 0.1 | 256 | 0.14 | 0.06 | 0.04 | 1.000 | 0.805 | 0.944 | 0.805 | 0.806 |
| 0.1 | 0.1 | 256 | 0.16 | 0.08 | 0.0 | 1.000 | 0.997 | 1.000 | 0.997 | 0.995 |
| 0.1 | 0.1 | 256 | 0.16 | 0.08 | 0.02 | 1.000 | 0.997 | 1.000 | 0.997 | 0.995 |
| 0.1 | 0.1 | 256 | 0.16 | 0.08 | 0.04 | 1.000 | 0.993 | 0.998 | 0.993 | 0.995 |
| 0.1 | 0.1 | 256 | 0.18 | 0.1 | 0.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.1 | 0.1 | 256 | 0.18 | 0.1 | 0.02 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.1 | 0.1 | 256 | 0.18 | 0.1 | 0.04 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
