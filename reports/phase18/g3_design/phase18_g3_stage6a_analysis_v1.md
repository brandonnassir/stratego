# Phase 18 Agent 6 — Stage 6A: Gate G3 evidence, power, and feasibility analysis (v1)

**Stage 6A of the G3 design work package. Analysis only.** This document freezes no
contract and authorizes no run. In producing it no Stratego game was played, no setup
pool was generated, no model was trained or loaded, no confirmation-slice board was
materialized, and no sealed artifact was opened. Every numerical table below is
reproduced by `scripts/phase18_g3_stage6a_tables.py` from tracked evidence
(`--write` regenerates `reports/phase18/g3_design/phase18_g3_stage6a_tables_v1.json`,
`--check` recomputes and compares it); the tables are reprinted verbatim in Appendix A.

Date: 2026-09-02. Author: Phase 18 Agent 6 (G3 design agent).

---

## 0. Identity and the settled design choices

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
evaluator                   in-process evaluator only after every R1–R11 credibility test passes
execution topology          sequential versus concurrent seeds chosen from the preflight throughput
                            result, then frozen
screening                   the library arm is included
confirmation size           the smallest size the cluster-aware analysis supports (section 5)
```

---

## 1. What G3 must answer

Common contract section 11, Gate G3 (setup-only Stratego benefit): the trained setup
policy improves over its fresh initialization; it beats the fixed library by the
predeclared practical margin with paired uncertainty excluding zero; direction is
consistent across the required seeds; diversity, legality, orientation and stability
gates pass. Under the settled choices both primary contrasts are read on the EMA setup
model and both must show a paired 95% lower bound above zero and a point estimate of at
least 0.05 effective win rate (EWR).

Constraints carried into the design:

- The EMA decides (method map S28, `exact`). P18-D004/P18-D005 established that after
  64 updates the EMA retained 0.938 of its initial parameter contribution and its
  synthetic utility moved 0.9–2.8% of the raw actor's movement. A raw-actor result may
  inform a development decision; it cannot close G3.
- The margin is scored in EWR from a known instrument resolution (P18-D005 review).
- Development data only: `setup_learning_development` is the only pack a setup-only
  assay may iterate against; no sealed Phase 8 test example and no `operator_sealed`
  entry may be read.
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

Common contract section 9.2 defines the live setup stream as: the raw setup model
samples a reusable pool; the frozen teacher matchup schedule plays games from that
pool; completed setup outcomes update the setup policy. That is the production
mechanism and, with pool-versus-pool settled, the G3 collection mechanism.

Outcome signal per cell in the train split (20,000 games, battleless 100): red wins
33.5%, blue wins 32.6%, draws 33.9%, mean 258.8 plies. Nine cells are at least 95%
draws (the `stress_draw_seeker` / `stress_information_miser` / `stress_scout_rush` /
`random_legal` combinations; two self-play cells are 200/200 draws), 23 cells are at
least 80% draws, 53 cells are at most 10% draws, 18 cells have one side winning at
least 90%, and the mean per-cell outcome variance Var(z) is 0.429 with 26 cells below
0.1. Under the frozen uniform schedule roughly 9% of collection games carry no outcome
information and about a quarter carry very little; the schedule is a production
identity and is not proposed to change, so the cost model counts these games.

### 2.2 The frozen mover (C0) and the fresh setup-policy comparator

```text
C0 = G1 reproduction candidate   /Users/brandonwashington/Dev/stratego_phase18/g1_control_v1/
                                 dry_run_artifacts/warmstart_c1_v1.pt
                                 sha256 460a246be32b821a6d6d7feb928b272a4be1014ff55053f329980e21e3be074c
                                 10,459,947 bytes; present and re-hashed 2026-09-02
accepted Phase 8 reference       checkpoints/phase8/warmstart_c1_v1.pt, sha256 f7e9c40d…eec7ca (not used in G3)
policy token                     phase6_c1_warmstart_greedy@0.2.0+float32, greedy, CPU float32
```

G1 closed on C0 (P18-D003: delta +0.006348, 95% [+0.000793, +0.011902] against the
−0.010 margin on 4,096 independent pairs). Both the learned-setup and the
fixed-library lanes use C0.

Fresh setup-policy comparator: `Phase18SetupModel`, 4 blocks / width 128 / 4 heads /
feed-forward 512, 802,320 parameters (tolerance 0), positional init std 0.1, forced
flag handedness plus independent 50% reflection, initialized per seed from
`derive_stream_seed(namespace, 'model_init', k)`. Under S28 the EMA at update 0 is the
raw initialization, so each seed's fresh comparator is its own `EMA_0 = raw_0`.

### 2.3 The fixed-library comparator

```text
setup_library_v1     digest 7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777
                     8,000 entries, 16 families F00–F15 x 500 bases; content and reflected fingerprints per entry
split rule           base_index 0..399 train (6,400), 400..449 validation (800), 450..499 test (800)
production source    training_setup_source('neutral_v1') -> train split, uniform over families
```

The fixed-library own-setup source is the production sampler over the train split: the
configuration the accepted Phase 8 corpus used.

### 2.4 The development pack and the strata

`setup_learning_development` is constructible from existing assets (evaluation
contract): the accepted draw path exists (Phase 16's benchmark drew every board from
the library validation split through `Phase15MatchSetupSources`; `SetupBank` and
`confirmation_bank.py` provide seeded banks with a reflection-class separation audit).

Proposed construction (not frozen): opponent setups are validation-split bases; the
pack is split by base into a screening slice and a confirmation slice so that nothing
decided on the screening slice can see a confirmation board.

```text
screening slice        base_index 400..409  (10 per family, 160 bases)  -> 160 x 10 x 2 = 3,200 cases per arm
confirmation slice     base_index 410..449 reserved (40 per family, 640 bases); the confirmation
                       design uses the smallest B the sizing rule supports (section 5.6), taken in
                       base-index order (B = 80 -> 410..414); the rest stays in reserve
case                   (opponent base, opponent policy, own colour, replicate 0)
own setup per arm      learned arms: one EMA sample under the case's own-setup seed;
                       library arms: one neutral_v1 draw from the train split under the case seed
game seed              one recorded function of (pack, case, lane, colour, opponent);
                       identical across every arm of a contrast
"opened"               a slice is opened when any of its boards is materialized as a played or
                       banked setup; reading the library's recorded fingerprints for the leakage
                       audit does not open it. The confirmation slice stays unopened in the
                       calibration stage.
```

Strata: `familiar` = the library validation bases (all 16 families). `unusual_procedural`
is not populated (evaluation contract E1) and is deferred to G4; G3 reports it
`not_run` and makes no generalization claim. `operator_sealed` is absent and blocks G6
only. A descriptive opponent stratum exists for free: teacher-seen opponents
(strategic, tactical, five stress policies) versus unseen neural opponents (p18, p24,
phase9_anchor); it reads on transfer, not on unfamiliar setups, and never gates.

Evaluation opponents (`stratego/evaluation/phase17/opponents.py`, bound by digest):
p18 (sha256 aa2cc39b…), p24 (9bf256a9…), phase9_anchor (ed0f5198…), plus seven code
opponents (strategic_rule_based, tactical_rule_based, stress_scout_rush,
stress_miner_rush, stress_berserker, stress_information_miser, stress_chaos).

### 2.5 Leakage and reflection-class boundaries

Rules (evaluation contract, frozen): a setup and its horizontal mirror are one class;
packs are separated by reflection-class fingerprint; pairwise content and class
intersections across all packs must be empty before a pack is opened; selection may
read the validation split and `setup_learning_development` only.

```text
setup learner training    pool-versus-pool teacher games: no library setup is ever sampled, seen or
                          scored by the learner (S32 on-policy requirement)
C0 mover                  trained on train-split bases only (Phase 8); validation bases were the
                          Phase 8 validation corpus (selection data) -> development use is allowed
dev pack opponents        validation bases -> disjoint by base from the fixed-library own-setup source
                          (train bases); class disjointness is PROVED by the pack audit over the
                          library's recorded content and reflected fingerprints, never assumed
sealed                    test bases 450..499 and the sealed Phase 8 test corpus untouched;
                          operator_sealed absent; the Phase 8 test multiplicity counter unchanged
learned setups            every emitted class is recorded so the operator_sealed check can be run
                          retroactively once that pack exists
```

### 2.6 Throughput, runtime and storage evidence

Teacher-versus-teacher games (the collection cost):

```text
Phase 4 baseline league     44,544 rule-vs-rule games: mean 0.087 s per game in one process
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
                            the harness that passed R1–R11 for the random gate)
```

The setup learner: G2 ran 64 updates plus four 4,096-sample evaluations in 913–930 s
per seed on CPU float32 with 4 threads (about 10 s per update including evaluation);
Phase 17 measured pool generation for 2 x 512 at 3.18 s CPU and a 5-epoch update on
320 episodes at 0.82 s CPU. Learner compute is a few seconds per update.

Storage: G2 outcome receipts 20.9 MB for 262,144 outcomes (about 80 B each); telemetry
6.4 KB per update; a setup checkpoint (raw + optimizer + EMA) 12.9 MB; a G1 receipt row
about 1.46 KB. 158 GiB free on the data volume; 48 GiB RAM; Apple M4 Pro, 10
performance and 4 efficiency cores.

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

The Phase 17 prior is the only real-game measurement of a fresh setup network against
the library with this mover family: a fresh initialization sat roughly 0.07–0.09 EWR
below the library. A learner that beats the library by 0.05 must therefore move its
EMA by roughly 0.12–0.14 EWR relative to its own init, so the two primary contrasts are
different-sized asks.

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
`P(win) = 0.9·sigmoid(3z)`. They say nothing quantitative about Stratego learning
speed; real outcomes are noisier (section 2.1), so the calibration stage measures the
curve rather than assuming it.

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
```

`stratego/training/phase18/setup_buffer.py` already implements these semantics
(`add_pool` reallocates counts and ready flags to zero for every row and collapses
duplicates to the newest period; `add_outcome` keeps a running mean per row and raises
on an unknown fingerprint; `filter(current_period)` drops rows whose
`period_added + storage_duration < current_period`), and
`tests/training/phase18/test_setup_buffer.py` pins them: an outcome finishing under the
next pool attributes to the old row; an undersized window raises rather than dropping
the outcome; counts and ready flags reset when a new pool arrives; identical played
boards collapse to one row bound to the newer snapshot; a zero-outcome setup is
excluded, not trained as a draw; an outcome for an unknown setup is fatal; the
advantage uses recorded behaviour quantities.

**The G2 assay exercised none of this in its data flow.** Its driver
(`synthetic_assay.run_seed`) generated a pool, drew exactly `outcomes_per_setup = 4`
instantaneous outcomes for every setup, updated, and filtered with
`storage_duration = 1`: no game crossed a period boundary, no row was attributed
under a later pool, no behaviour-snapshot age exceeded zero, no row had m = 0. That
was correct for a synthetic landscape whose "games" are instantaneous, and the S21–S23
semantics were verified by the unit tests above rather than by the assay. A G3
collector that reproduced the same synchronous four-outcome scheme would leave those
semantics untested in real play before G4 and would not be the published method,
which the paper itself describes as "slightly off-policy" because most games span
several collection periods.

### 3.2 Proposed G3 collector (asynchronous, published semantics; not frozen)

```text
slots                  S = 2,560 concurrent game slots (10 workers x 256), each stepped in a fixed order
period                 every slot advances T = 202 plies per period (the published 2 x train_every_per_player
                       = 2 x 101); a slot whose game ends immediately starts a new game from the
                       CURRENT pool, so games started in period t may finish in t + 1 or later
pool                   one pool of 1,024 setups per period (512 per lane), sampled by the raw actor at
                       the period boundary and added to the buffer (S10 de-duplication applies)
cells                  each new game takes the next cell of the frozen 100-cell schedule in cyclic
                       order, so game STARTS are exactly uniform over cells in every window of 100;
                       completions per cell per period are length-weighted and are recorded
outcomes               every completed game attributes two outcomes (one per pool row, from the
                       owner's perspective) through add_outcome; counts and ready flags reset at
                       every refresh (S23); m per row is variable and recorded (S09)
retention              storage_duration = 21 periods = ceil(4,000 / 202) + 1, covering the absolute
                       move limit under TRAINING_RULES so that no attribution can fail for age (S21)
update                 after each period: process the ready rows, 5 epochs, one optimizer step per
                       1,024 ready rows (S26; ceil(ready / 1,024) minibatches), one EMA update (S28),
                       then filter and regenerate the pool
rules                  TRAINING_RULES for every collection game; the rules token is stamped on every
                       receipt
determinism            fixed slot order, per-game seeds derived from (period, slot, ordinal) through
                       derive_stream_seed, so the whole stream is replayable on CPU as in G2
```

Expected steady state (Appendix A, T7a): 1,998 completions per period, 3,996 outcomes,
a mean of 3.90 outcomes per generated setup split across the generating period and the
next, 174 CPU-seconds of games per period, 20–67 s of wall time per period at 100–30
games per second. The first period trains on fewer completions than steady state
(games started at period 1 mostly finish in period 2); the ready-row count, the m
distribution, the fraction of rows with m = 0, the snapshot-age distribution and the
cross-period attribution count are recorded every period.

### 3.3 The synchronous alternative, recorded as a divergence

`D-G3-SYNC` (not recommended): exactly m = 4 instantaneous outcomes per setup per
period with `storage_duration = 1`, as in G2. Same games per period, simpler, fully
replayable, but it never exercises S21–S23 in real play, never produces a
behaviour-snapshot age above zero, never produces m = 0 rows, and is not the
published collection regime. If it were adopted for G3 it would have to be declared as
a G3-specific divergence and re-validated before G4. The proposal here is to reproduce
the published semantics (section 3.2).

---

## 4. EMA parameter aging (parameters only)

With decay 0.999 applied once per setup update, the EMA parameters after U updates are
`0.999^U · θ_0 + Σ_k (1 − 0.999)·0.999^(U−k) · θ_k`. Appendix A, T6 tabulates the
retained initial-parameter fraction and the weights: 0.938 at 64 updates, 0.880 at 128,
0.774 at 256, 0.368 at 1,000, 0.135 at 2,000, 0.050 at 3,000; the most recent 64 updates
always carry weight 0.062.

**No effective-win-rate response is inferred from this blend.** The only evidence on
how the parameter blend maps to performance is G2 (Appendix A, T2, lower table): at 64
updates the trajectory weight was 0.062, yet the EMA closed 0.10–0.52% of its
initial-to-optimum utility gap while the raw actor closed 9.6–20.9% — a utility
response of 0.9–2.8% of the raw actor's, well below the parameter weight. Parameter
interpolation therefore does not predict performance, and this analysis makes no claim
about the EMA's EWR at any budget. What the calibration stage measures instead
(section 6): the raw and EMA performance on the screening slice at 128 and 256
updates, their divergence in parameter space (relative L2 distance per update) and in
performance, and the direction of the raw learning curve. The later reviewed decision
on whether to continue to 2,000 or 3,000 updates rests on those measurements plus the
cost table (Appendix A, T7b), not on the aging arithmetic.

---

## 5. Power and sample size around the actual resampling unit

### 5.1 Structure of the evaluation and what is correlated

Cases are `(base, opponent, colour)`: B bases x 10 opponents x 2 colours = 20B cases
per arm. Arms: for each seed k, the trained EMA `E_k` and its own init `I_k`; the
library arm either once (`L`, shared) or once per seed (`L_k`). Contrasts: P1 =
`E_k − I_k`, P2 = `E_k − L` (or `E_k − L_k`), pooled over the three seeds.

Four correlation sources make the games far from independent:

1. **Repeated bases across opponents and colours.** A base whose opponent setup favours
   (or punishes) learned setups shifts every one of its 20 cases in the same direction
   in every seed. Modelled as a base x arm-type effect with variance fraction `rho_b`
   of the paired-difference variance V; treated as shared across seeds (worst case).
2. **The same cases reused across seeds.** A case-level own-setup-type effect shared
   across seeds, variance fraction `rho_w`.
3. **The shared fixed-library arm.** With one library game per case, every seed's P2
   difference contains that game's own noise; under the frozen constants this alone
   makes two seeds' P2 differences on a case correlate at 0.5 (V = 2σ²(1 − ρ_case),
   ρ_case = 0.190, so the library game carries exactly half of V).
4. **The Phase 17 instrument itself** (V = 0.5391²) already reflects the same-case
   pairing.

Model (Appendix A, T3 and the script): `Y_a = v[base, type] + w[case, type] + e[arm, case]`,
with `2τ² = rho_b·V`, `2ω² = rho_w·V`, `2ε² = (1 − rho_b − rho_w)·V`. Pooled-estimator
variance, three seeds, 20 cases per base:

```text
P2, per-seed library arms (and P1):  Var = (V / B) · [rho_b + rho_w/20 + (1 − rho_b − rho_w)/60]
P2, one shared library arm:          Var = (V / B) · [rho_b + rho_w/20 + (1 − rho_b − rho_w)/30]
one seed alone:                      Var = (V / B) · [rho_b + rho_w/20 + (1 − rho_b − rho_w)/20]
effective independent paired units   n_eff = V / Var
```

At `rho_b = rho_w = 0` the three seeds with per-seed library arms give n_eff = 60B (three
independent seeds), the shared arm gives 30B, and one seed gives 20B. Any base-level
effect caps n_eff at about B / rho_b whatever the number of seeds, arms or cases:
**bases are the scarce resource, not games.**

### 5.2 Validation of the variance model and of the base-cluster bootstrap

Appendix A, T5: Monte Carlo replications of the model reproduce the analytic SE to
within 2% for P1, P2 with per-seed library arms and P2 with the shared arm, at
(B, rho_b, rho_w) = (160, 0.05, 0.10), (80, 0.10, 0.10) and (160, 0, 0). The naive SE that
treats 60B paired games as independent (0.0055 at B = 160) understates the true SE by
a factor of 2.0 at rho_b = 0.05. The proposed bootstrap — resample bases with
replacement, carrying every case, colour, opponent, arm and seed of a base together —
recovers the true SE (mean bootstrap SE 0.0112 against 0.0112; 0.0205 against 0.0207)
with 94–95% coverage of the true mean over 300 datasets. The base-cluster bootstrap is
therefore the proposed resampling unit, and the naive claims of the earlier draft
(38,400 independent observations, half-width 0.0055) are withdrawn.

### 5.3 The decision rule's power

Both primary contrasts pass only if the paired 95% lower bound exceeds zero and the
point estimate is at least 0.05. At a true effect of exactly 0.05 the point criterion
passes with probability 0.5 whatever the sample (Appendix A, T4a). The design must
therefore state the effect it is powered for. With SE the pooled standard error,
80% power requires `d ≥ max(0.05 + 0.8416·SE, 2.8016·SE)`; the script reports this as
`d80`. P1 has the same rule and, having per-seed init arms, at least the resolution of
P2 with per-seed library arms, so P2 governs sizing.

### 5.4 Raw games versus effective units

Appendix A, T4b–T4d give n_eff, SE and d80 for B from 80 to 640 bases and rho_b from 0
to 0.20 (rho_w = 0.10; T4e shows rho_w matters little). Appendix A, T7d gives the raw
game counts for the same B. Two readings:

```text
B = 80  (5 per family)   1,600 cases per arm; 14,400 games with per-seed library arms (9 arms)
        n_eff  4,000 / 2,017 / 1,157 / 676 / 369   at rho_b = 0 / 0.02 / 0.05 / 0.10 / 0.20
        SE     0.0085 / 0.0120 / 0.0159 / 0.0207 / 0.0281
        d80    0.057 / 0.060 / 0.063 / 0.067 / 0.079
B = 160 (10 per family)  3,200 cases per arm; 28,800 games (9 arms)
        n_eff  8,000 / 4,034 / 2,313 / 1,352 / 738
        SE     0.0060 / 0.0085 / 0.0112 / 0.0147 / 0.0198
        d80    0.055 / 0.057 / 0.059 / 0.062 / 0.067
```

With the shared library arm at B = 80: n_eff 2,286 / 1,472 / 960 / 608 / 350 and d80
0.060 / 0.062 / 0.065 / 0.068 / 0.081 (T4c). The per-seed library arms cost 29% more
games (9 arms against 7) and buy 75% more effective units at rho_b = 0 but only 11%
more at rho_b = 0.10; whether they are worth it depends on the measured rho_b.

### 5.5 Stratum floors

The worst-stratum rule needs at least 200 paired games in a stratum before a regression
may be reported. Pooled over three seeds a per-opponent stratum holds 6B paired games
and a per-opponent-by-colour stratum 3B (T7d), so B ≥ 67 satisfies both; per-seed
strata at B = 80 hold 160 games per opponent and are reported as descriptive and
underpowered. The rule counts raw paired games; their effective count is smaller by the
same factors as above and is reported beside it.

### 5.6 Sizing rule and the provisional smallest confirmation size

Proposed rule (to be frozen in the Stage 6B contract): the confirmation uses the
smallest B that is a multiple of 16 (equal bases per family), at least 80 (stratum
floors), and gives `d80 ≤ 0.07` for P2 under the arm design chosen, with rho_b and rho_w
taken from the calibration stage's estimates on the screening slice (rho_b from a
one-way random-effects decomposition of per-seed case differences over 160 bases;
rho_w from the cross-seed correlation of case-level differences). Appendix A, T4f:
under the planning assumption rho_b ≤ 0.10 the rule gives B = 80 with per-seed library
arms (B = 64 would suffice; the stratum floor binds) and B = 80 with the shared arm;
at rho_b = 0.20 it gives 112 and 128. Powering for 0.06 instead of 0.07 would need
B = 144–256 (per-seed arms) or 176–272 (shared arm) at rho_b = 0.05–0.10.

**Provisional smallest confirmation size: B = 80 bases (base_index 410..414, 5 per
family), 1,600 cases per arm, 14,400 games with per-seed library arms or 11,200 with
the shared arm.** It is provisional because rho_b and rho_w are unmeasured; the
calibration stage measures them, and the frozen rule re-derives B at the review stop.
The choice of the powered effect (0.07 rather than 0.06) is a consequential decision
for review (section 11).

### 5.7 Multiple comparisons

The G3 decision requires both primary contrasts to pass: an intersection-union test,
whose familywise error of declaring PROCEED is at most the per-test alpha, so no
adjustment is applied to the two primaries. Seed consistency is a direction check on
point estimates, not a third test; per-seed intervals are diagnostics. Strata are
descriptive except the pre-registered worst-stratum and colour rules, which apply
only where the 200-game floor holds. Decisions taken at the calibration review are
development decisions on a slice disjoint by base from the confirmation slice; they
never select a checkpoint and enter no confirmation error accounting.

---

## 6. The bounded calibration stage (ends at 256 updates; mandatory review stop)

One training run per seed (three seeds, independent inits), run to exactly 256 periods
and stopped. Nothing continues automatically. Evaluation never feeds the learner.

### 6.1 What it measures

```text
throughput            exact collector games per second and seconds per period, per worker count,
                      sequential and concurrent (the preflight microbenchmark replays frozen
                      Phase 8 corpus game ids through play_corpus_game with library setups: no
                      learner, no pool, nothing retained beyond the timing record); the topology
                      (sequential or concurrent seeds) is chosen from it and frozen before period 1
raw and EMA performance
                      at updates 128 and 256, raw_U and EMA_U per seed against EMA_0 (= raw_0) and
                      the library arm(s) on the screening slice (3,200 cases per arm),
                      EVALUATION_RULES, C0 as the mover; paired, base-cluster bootstrap;
                      the raw result is diagnostic, the EMA result is telemetry: neither decides G3
learning-curve direction
                      raw_128 -> raw_256 on the screening slice, plus (optional) a teacher-regime probe
                      every 32 periods: raw and EMA pools against library train-split setups under
                      the teacher schedule and TRAINING_RULES, 512 games each, outcomes never fed
                      to the learner (Appendix A, T7c: 8,192 teacher games, about two minutes)
outcome statistics    per period: completions, outcomes, ready rows, m distribution (mean, median,
                      fraction m = 0, fraction m = 1), per-row outcome variance, per-cell completed
                      counts, snapshot-age distribution, cross-period attributions, duplicates collapsed
diversity, legality,  every pool and every evaluation sample: the section 7 statistics with their
orientation,          thresholds
concentration
raw/EMA divergence    relative L2 parameter distance per period; raw-minus-EMA performance at 128 and 256
correlation inputs    rho_b and rho_w estimated on the screening slice (section 5.6), and the
                      library-draw variance if per-seed library arms are used
integrity             legality / orientation / attribution / non-finite / identity events; planned =
                      completed + failed + missing per period and per evaluation lane
```

### 6.2 Stops

Only integrity failures stop a run automatically (section 7, rules A1–A9). Every
learning result — including a harmful direction on the screening slice — returns to
review at 256 updates with the packet; the earlier draft's automatic
`STOP-INEFFECTIVE (upper bound < 0.03)` rule is withdrawn because nothing demonstrates
that a 256-update raw reading predicts the eventual EMA endpoint.

### 6.3 What the review decides afterwards

The calibration packet (throughput, raw/EMA readings, curve direction, outcome
statistics, diversity, divergence, rho_b, rho_w, integrity) goes to review. A later
reviewed decision determines whether continuing the same runs to 2,000 or 3,000 updates
is justified (cost only: Appendix A, T7b — at 2,000 periods 4.0 M games per seed, 96.6
CPU-hours, 11–37 wall-hours per seed; at 3,000 periods 6.0 M games, 145 CPU-hours,
17–56 wall-hours per seed), and fixes the confirmation design (B by the section 5.6
rule, shared or per-seed library arms, the confirmation checkpoint) in the Stage 6B
contract. Continuation resumes from the period-256 checkpoint, which requires the S29
resume proof on the production device before any further period.

### 6.4 The confirmation stage (for orientation only; not designed here)

The EMA after the fixed final update decides; intermediate points are telemetry; no
peak selection. Primary contrasts P1 and P2 pooled over seeds, base-cluster bootstrap,
10,000 replicates, frozen seed: each passes only with lower bound > 0 and point ≥ 0.05.
Direction consistent in each seed; colour rule on P1; worst-stratum rule on P2 with the
200-game floor; integrity gates; the raw actor reported beside the EMA as a diagnostic.
PROCEED when everything passes; STOP when a valid run fails P1 with no isolated
defect; REVISE when P1 passes and P2 fails (common contract section 12: investigate
estimator variance, pool reuse, symmetry, evaluation distribution; do not integrate);
BLOCKED when a dependency prevents valid evidence.

---

## 7. Diversity and concentration thresholds (exact, with their evidence)

Statistics are computed on canonical boards (rank 0 = back rank, rank 3 = front row)
for every 1,024-setup pool and every 4,096-setup evaluation sample; "symmetrized" file
shares average a file with its mirror, which is what the 50% reflection produces in
play. References: the library (Appendix A, T1: 8,000 curated bases, per-family range)
and the fresh model (T2: 24 tracked G2 generation samples of 4,096; the 384 untracked
1,024-setup pool records of the two G2 assays corroborate them with distinct classes
1,024/1,024 in every pool, played file share at most 0.173, mirror-asymmetry z at most
3.71, reflected fraction 0.460–0.546, and 12 immediately terminal setups in total).

Automatic stops (integrity; the run halts at the period boundary with its checkpoint):

```text
A1  legality failures > 0            fresh model: 0 in every sample and pool; Phase 18 method map S02
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
                                     most front-heavy curated family 0.292
A9  accounting failure               planned != completed + failed + missing in any period or lane
```

Review flags (recorded and reported; no automatic action):

```text
R1  distinct reflection classes < 1,014 of 1,024 (at least 1% duplicates); fresh baseline 0 duplicates
R2  max symmetrized flag-file share > 0.25     fresh 0.108–0.158; library overall 0.110; a curated corner
                                               family reaches 0.50, so concentration alone is not a defect
R3  front-row flag share > 0.292               the most front-heavy curated family (irregular_high_entropy)
R4  back-rank flag share < 0.184               the least back-heavy curated family
R5  mean bombs adjacent to the flag > 2.572    the most fortress-like curated family
R6  front-row bomb share > 0.2526              the most bomb-forward curated family (high_bomb_placement)
R7  max symmetrized bomb-file share > 0.20     twice the square-uniform 0.10; every curated family <= 0.1533
R8  mean sequence information < 0.80 x the seed's update-0 value
                                               Phase 17 fell 22% over 12 hours without collapse; fresh
                                               samples rose 0.8–4.5% over 64 synthetic updates
R9  reflected fraction outside [0.44, 0.56] in any 1,024 pool (about 3.8 binomial SD; observed 0.460–0.546)
R10 immediately terminal setups > 10 in any pool (observed 12 in 384 pools)
```

The update-0 values of every statistic are recorded per seed so that drift is read
against the model's own start as well as against the references.

---

## 8. Evaluation accounting (corrected)

Appendix A, T7c and T7d. Screening slice, 3,200 cases per arm, EMA evaluated at both
128 and 256:

```text
arms with one shared library arm      raw_128, EMA_128, raw_256, EMA_256 per seed (12) + EMA_0 per seed (3)
                                      + library (1) = 16 arms -> 51,200 games
arms with per-seed library arms       18 arms -> 57,600 games
wall time                             0.75–0.85 h on an in-process harness at 18.9 games/s (only after
                                      R1–R11 pass); 9.5–10.7 h on the G1 harness at 1.5 games/s
optional teacher-regime probe         8,192 teacher games (about two minutes)
```

Confirmation slice (not opened in the calibration stage), provisional B = 80: 11,200
games with the shared arm (7 arms) or 14,400 with per-seed arms (9 arms); 0.2–2.7 h.
Totals if the confirmation later runs at B = 80: 62,400 games (shared arm) or 72,000
(per-seed arms). The earlier draft's "0.9 million evaluation games" figure was an
error and is withdrawn.

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
G3-ENG-02  Phase 18 roster evaluator with lane-aware own-setup resolution (EMA sample or library draw
           per case, opponent setup and seeds fixed per case) meeting R1–R11 including the R9 retry
           regression; the in-process path is used only after every test passes
G3-ENG-03  development-pack builder: screening slice and reserved confirmation bases, class-separation
           audit against the train-split own-setup source over recorded fingerprints
G3-ENG-04  run driver: checkpoints every 32 periods and at 128/256, exact CPU resume proof (S29),
           cooperative termination (S35), telemetry, screening evaluation hook, binding ledger,
           runtime under output/phase18/runtime/
G3-ENG-05  preflight microbenchmarks: collector games per second by worker count and topology
           (corpus-id replay), evaluation games per second, per-period wall; topology frozen from it
G3-ENG-06  analysis and packet scripts: base-cluster bootstrap, rho_b / rho_w estimation, strata,
           section 7 statistics, calibration packet
```

Every item is engineering with tests; none requires a method change. The execution
instruction (Stage 6C) must place a mandatory preflight/review stop between these
items and the first collection game.

---

## 10. Risks

| risk | evidence | handling |
|---|---|---|
| Transfer: outcomes from teacher play, evaluation with the C1 mover | sections 2.1–2.2 | the production mechanism; P2 measures the transfer; the teacher-seen / unseen stratum reports it |
| Zero-signal cells: 9 of 100 cells at least 95% draws, 23 at least 80% | section 2.1 | counted in the cost; schedule unchanged; per-cell completions and Var(z) recorded |
| Collection under battleless 100, evaluation under 200 | P18-A001 invariant | both tokens on every receipt; no power claim rests on it |
| The EMA's response to training is unmeasured in Stratego | section 4 | measured at 128 and 256; no inference from parameter aging; continuation decided at review |
| Fresh init about 0.08 below the library, so P2 needs about 0.13 of EMA gain | section 2.7 | the calibration reads the remaining gap; no budget is promised |
| rho_b / rho_w unknown; bases are the scarce resource | section 5 | estimated on the screening slice; B re-derived by the frozen rule |
| Instrument SD may differ from the Phase 17 measurement | section 2.7 | sizing rule uses the measured slice statistics at the review |
| Throughput unmeasured for the exact collector | section 2.6 | mandatory preflight; topology frozen from it |
| MPS is not bitwise reproducible; CPU greedy evaluation is | P18-D002/D003 | learner and games on CPU; replay of pool and EMA digests as in G2 |
| Diversity collapse, orientation defect, attribution loss | Phase 17, Phase 12, S21 | section 7 stops and flags; buffer raises on attribution |
| Concurrent seeds contend for the 10 performance cores | section 2.6 | preflight measures parallel efficiency; topology frozen |
| Evaluator credibility R1–R11 unmet for a roster evaluator | section 9 | G3-ENG-02 with tests before any confirmation game |

---

## 11. Remaining genuinely consequential decisions

1. **The powered effect for sizing the confirmation**: 0.07 (provisional B = 80) or
   0.06 (B = 144–272 depending on rho_b). The combined rule cannot be powered at 0.05.
2. **Library arm design**: one shared arm or one independent draw per seed. Proposed
   rule: per-seed arms if the calibration's rho_b estimate is at most 0.05 (where they
   buy 55–75% more effective units), otherwise the shared arm; alternatively fix one
   now.
3. **Collector parameters S = 2,560 slots and T = 202 plies per period** (published
   cadence; about 2,000 completions per period): keep, or re-derive from the preflight
   throughput before period 1.
4. **The optional teacher-regime learning-curve probe** (8,192 teacher games):
   include or omit.
5. **Whether the calibration evaluations use the G1 harness (proven, 1.5 games/s,
   9.5–10.7 h) or wait for the in-process evaluator to pass R1–R11** (0.75–0.85 h).

Everything else in this document is either settled by the operator, fixed by frozen
artifacts, or an engineering item with tests.

---

## 12. What this stage did not do

No game was played, no pool sampled, no model built, trained or loaded, no
confirmation board materialized, no sealed artifact opened, no contract frozen, no
accepted artifact changed, nothing pushed. The only executable added reads tracked
files and writes one JSON of tables.

---

## Appendix A — tables (verbatim output of `scripts/phase18_g3_stage6a_tables.py --check`)

Reproduce with:

```bash
.venv/bin/python scripts/phase18_g3_stage6a_tables.py --check
```

The committed JSON `reports/phase18/g3_design/phase18_g3_stage6a_tables_v1.json` has
SHA-256 `3a201c9a41dd729146f36ee0d8798077c6f80ca11648f1498e456f8036e0a059`; `--check`
exits non-zero if a recomputation differs.

## T1 Library concentration references (setup_library_v1, 8,000 bases; canonical rank 0 = back rank, 3 = front row)

| statistic | library overall | family min (family) | family max (family) |
|---|---|---|---|
| front_row_flag_share | 0.0182 | 0.0000 (aggressive_high_rank_front) | 0.2920 (irregular_high_entropy) |
| back_rank_flag_share | 0.7738 | 0.1840 (irregular_high_entropy) | 1.0000 (balanced_conventional) |
| max_flag_file_share_symmetrized | 0.1096 | 0.1030 (irregular_high_entropy) | 0.5000 (corner_flag_fortress) |
| bombs_adjacent_to_flag_mean | 1.0101 | 0.0000 (lightly_defended_deceptive_flag) | 2.5720 (central_back_flag_fortress) |
| front_row_bomb_share | 0.0826 | 0.0380 (balanced_conventional) | 0.2526 (high_bomb_placement) |
| back_rank_bomb_share | 0.2161 | 0.0472 (high_bomb_placement) | 0.2852 (near_corner_flag_fortress) |
| max_bomb_file_share_symmetrized | 0.1011 | 0.1018 (miner_preservation) | 0.1533 (central_back_flag_fortress) |

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

## T3 Planning constants

SD_paired 0.5391, V 0.2906, single-lane SD 0.4236, implied same-case cross-arm correlation 0.1902, tie fraction 0.594, margin 0.05, alpha two-sided 0.05, power 0.80.

## T4a Resolution per effective independent paired unit

| n_eff | SE | 95% half-width | MDE (lower>0, 80%) | d80 combined rule | P(pass at 0.05) | P(pass at 0.07) | P(pass at 0.10) |
|---|---|---|---|---|---|---|---|
| 256 | 0.0337 | 0.0660 | 0.0944 | 0.0944 | 0.317 | 0.547 | 0.843 |
| 512 | 0.0238 | 0.0467 | 0.0667 | 0.0701 | 0.500 | 0.799 | 0.982 |
| 913 | 0.0178 | 0.0350 | 0.0500 | 0.0650 | 0.500 | 0.869 | 0.997 |
| 1024 | 0.0168 | 0.0330 | 0.0472 | 0.0642 | 0.500 | 0.882 | 0.999 |
| 2048 | 0.0119 | 0.0233 | 0.0334 | 0.0600 | 0.500 | 0.953 | 1.000 |
| 4096 | 0.0084 | 0.0165 | 0.0236 | 0.0571 | 0.500 | 0.991 | 1.000 |
| 8192 | 0.0060 | 0.0117 | 0.0167 | 0.0550 | 0.500 | 1.000 | 1.000 |

## T4b P2 with per-seed library arms (also P1): n_eff / SE / d80, rho_w = 0.10

| B bases | cases/arm | rho_b=0.0 | rho_b=0.02 | rho_b=0.05 | rho_b=0.1 | rho_b=0.2 |
|---|---|---|---|---|---|---|
| 80 | 1600 | 4000 / 0.0085 / 0.0572 | 2017 / 0.0120 / 0.0601 | 1157 / 0.0159 / 0.0633 | 676 / 0.0207 / 0.0674 | 369 / 0.0281 / 0.0786 |
| 112 | 2240 | 5600 / 0.0072 / 0.0561 | 2824 / 0.0101 / 0.0585 | 1619 / 0.0134 / 0.0613 | 946 / 0.0175 / 0.0647 | 517 / 0.0237 / 0.0700 |
| 160 | 3200 | 8000 / 0.0060 / 0.0551 | 4034 / 0.0085 / 0.0571 | 2313 / 0.0112 / 0.0594 | 1352 / 0.0147 / 0.0623 | 738 / 0.0198 / 0.0667 |
| 240 | 4800 | 12000 / 0.0049 / 0.0541 | 6050 / 0.0069 / 0.0558 | 3470 / 0.0092 / 0.0577 | 2028 / 0.0120 / 0.0601 | 1108 / 0.0162 / 0.0636 |
| 320 | 6400 | 16000 / 0.0043 / 0.0536 | 8067 / 0.0060 / 0.0551 | 4627 / 0.0079 / 0.0567 | 2704 / 0.0104 / 0.0587 | 1477 / 0.0140 / 0.0618 |
| 480 | 9600 | 24000 / 0.0035 / 0.0529 | 12101 / 0.0049 / 0.0541 | 6940 / 0.0065 / 0.0554 | 4056 / 0.0085 / 0.0571 | 2215 / 0.0115 / 0.0596 |
| 640 | 12800 | 32000 / 0.0030 / 0.0525 | 16134 / 0.0042 / 0.0536 | 9253 / 0.0056 / 0.0547 | 5408 / 0.0073 / 0.0562 | 2954 / 0.0099 / 0.0583 |

## T4c P2 with one shared library arm: n_eff / SE / d80, rho_w = 0.10

| B bases | cases/arm | rho_b=0.0 | rho_b=0.02 | rho_b=0.05 | rho_b=0.1 | rho_b=0.2 |
|---|---|---|---|---|---|---|
| 80 | 1600 | 2286 / 0.0113 / 0.0595 | 1472 / 0.0140 / 0.0618 | 960 / 0.0174 / 0.0646 | 608 / 0.0219 / 0.0684 | 350 / 0.0288 / 0.0807 |
| 112 | 2240 | 3200 / 0.0095 / 0.0580 | 2061 / 0.0119 / 0.0600 | 1344 / 0.0147 / 0.0624 | 851 / 0.0185 / 0.0656 | 491 / 0.0243 / 0.0705 |
| 160 | 3200 | 4571 / 0.0080 / 0.0567 | 2945 / 0.0099 / 0.0584 | 1920 / 0.0123 / 0.0604 | 1215 / 0.0155 / 0.0630 | 701 / 0.0204 / 0.0671 |
| 240 | 4800 | 6857 / 0.0065 / 0.0555 | 4417 / 0.0081 / 0.0568 | 2880 / 0.0100 / 0.0585 | 1823 / 0.0126 / 0.0606 | 1051 / 0.0166 / 0.0640 |
| 320 | 6400 | 9143 / 0.0056 / 0.0547 | 5890 / 0.0070 / 0.0559 | 3840 / 0.0087 / 0.0573 | 2430 / 0.0109 / 0.0592 | 1401 / 0.0144 / 0.0621 |
| 480 | 9600 | 13714 / 0.0046 / 0.0539 | 8834 / 0.0057 / 0.0548 | 5760 / 0.0071 / 0.0560 | 3646 / 0.0089 / 0.0575 | 2102 / 0.0118 / 0.0599 |
| 640 | 12800 | 18286 / 0.0040 / 0.0534 | 11779 / 0.0050 / 0.0542 | 7680 / 0.0062 / 0.0552 | 4861 / 0.0077 / 0.0565 | 2803 / 0.0102 / 0.0586 |

## T4d one seed alone: n_eff / SE / d80, rho_w = 0.10

| B bases | cases/arm | rho_b=0.0 | rho_b=0.02 | rho_b=0.05 | rho_b=0.1 | rho_b=0.2 |
|---|---|---|---|---|---|---|
| 80 | 1600 | 1600 / 0.0135 / 0.0613 | 1159 / 0.0158 / 0.0633 | 821 / 0.0188 / 0.0658 | 552 / 0.0230 / 0.0693 | 333 / 0.0295 / 0.0827 |
| 112 | 2240 | 2240 / 0.0114 / 0.0596 | 1623 / 0.0134 / 0.0613 | 1149 / 0.0159 / 0.0634 | 772 / 0.0194 / 0.0663 | 467 / 0.0250 / 0.0710 |
| 160 | 3200 | 3200 / 0.0095 / 0.0580 | 2319 / 0.0112 / 0.0594 | 1641 / 0.0133 / 0.0612 | 1103 / 0.0162 / 0.0637 | 667 / 0.0209 / 0.0676 |
| 240 | 4800 | 4800 / 0.0078 / 0.0565 | 3478 / 0.0091 / 0.0577 | 2462 / 0.0109 / 0.0591 | 1655 / 0.0133 / 0.0612 | 1000 / 0.0170 / 0.0643 |
| 320 | 6400 | 6400 / 0.0067 / 0.0557 | 4638 / 0.0079 / 0.0567 | 3282 / 0.0094 / 0.0579 | 2207 / 0.0115 / 0.0597 | 1333 / 0.0148 / 0.0624 |
| 480 | 9600 | 9600 / 0.0055 / 0.0546 | 6957 / 0.0065 / 0.0554 | 4923 / 0.0077 / 0.0565 | 3310 / 0.0094 / 0.0579 | 2000 / 0.0121 / 0.0601 |
| 640 | 12800 | 12800 / 0.0048 / 0.0540 | 9275 / 0.0056 / 0.0547 | 6564 / 0.0067 / 0.0556 | 4414 / 0.0081 / 0.0568 | 2667 / 0.0104 / 0.0588 |

## T4e rho_w sensitivity (per-seed library arms)

| B | rho_b | rho_w | n_eff | SE | d80 |
|---|---|---|---|---|---|
| 80 | 0.05 | 0.0 | 1215 | 0.0155 | 0.0630 |
| 80 | 0.05 | 0.1 | 1157 | 0.0159 | 0.0633 |
| 80 | 0.05 | 0.3 | 1055 | 0.0166 | 0.0640 |
| 80 | 0.1 | 0.0 | 696 | 0.0204 | 0.0672 |
| 80 | 0.1 | 0.1 | 676 | 0.0207 | 0.0674 |
| 80 | 0.1 | 0.3 | 640 | 0.0213 | 0.0679 |
| 160 | 0.05 | 0.0 | 2430 | 0.0109 | 0.0592 |
| 160 | 0.05 | 0.1 | 2313 | 0.0112 | 0.0594 |
| 160 | 0.05 | 0.3 | 2110 | 0.0117 | 0.0599 |
| 160 | 0.1 | 0.0 | 1391 | 0.0145 | 0.0622 |
| 160 | 0.1 | 0.1 | 1352 | 0.0147 | 0.0623 |
| 160 | 0.1 | 0.3 | 1280 | 0.0151 | 0.0627 |

## T4f Smallest B (equal per family) with d80 at or below the target, rho_w = 0.10

| target d80 | design | rho_b=0.0 | rho_b=0.02 | rho_b=0.05 | rho_b=0.1 | rho_b=0.2 |
|---|---|---|---|---|---|---|
| 0.06 | per_seed_library | 48 | 96 | 144 | 256 | 448 |
| 0.06 | shared_library | 80 | 112 | 176 | 272 | 480 |
| 0.07 | per_seed_library | 16 | 32 | 48 | 64 | 112 |
| 0.07 | shared_library | 32 | 32 | 48 | 80 | 128 |

## T5 Monte Carlo check of the variance model and the base-cluster bootstrap

| B | rho_b | rho_w | P1 SE sim / analytic | P2 per-seed-L SE sim / analytic | P2 shared-L SE sim / analytic | naive SE (60B indep.) | bootstrap mean SE | 95% coverage |
|---|---|---|---|---|---|---|---|---|
| 160 | 0.05 | 0.1 | 0.0113 / 0.0112 | 0.0112 / 0.0112 | 0.0123 / 0.0123 | 0.0055 | 0.0112 | 0.950 |
| 80 | 0.1 | 0.1 | 0.0210 / 0.0207 | 0.0208 / 0.0207 | 0.0222 / 0.0219 | 0.0078 | 0.0205 | 0.940 |
| 160 | 0.0 | 0.0 | 0.0057 / 0.0055 | 0.0055 / 0.0055 | 0.0079 / 0.0078 | 0.0055 | 0.0055 | 0.950 |

## T6 EMA parameter aging (decay 0.999, time constant 1000 updates) — parameters only

| U | retained initial-parameter fraction 0.999^U | weight on the raw trajectory | weight on the last 64 updates |
|---|---|---|---|
| 64 | 0.9380 | 0.0620 | 0.0620 |
| 128 | 0.8798 | 0.1202 | 0.0620 |
| 256 | 0.7740 | 0.2260 | 0.0620 |
| 500 | 0.6064 | 0.3936 | 0.0620 |
| 1000 | 0.3677 | 0.6323 | 0.0620 |
| 2000 | 0.1352 | 0.8648 | 0.0620 |
| 3000 | 0.0497 | 0.9503 | 0.0620 |

## T7a Asynchronous collector per period: 2560 slots x 202 plies / 258.8 mean plies

expected completions per period 1998, outcomes 3996, mean outcomes per setup 3.90, CPU s per period 174, wall s per period at 30/60/100 games/s: 67 / 33 / 20

## T7b Collection cost per seed by period budget (games are cost only; no play-strength inference)

| periods | games/seed | CPU h/seed | wall h/seed @30 | @60 | @100 | three seeds @30 | @60 | @100 | receipts MB | game rows MB | checkpoints MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 0.51 M | 12.4 | 4.7 | 2.4 | 1.4 | 14.2 | 7.1 | 4.3 | 82 | 153 | 103 |
| 1000 | 2.00 M | 48.3 | 18.5 | 9.3 | 5.6 | 55.5 | 27.8 | 16.7 | 320 | 599 | 400 |
| 2000 | 4.00 M | 96.6 | 37.0 | 18.5 | 11.1 | 111.0 | 55.5 | 33.3 | 639 | 1199 | 800 |
| 3000 | 5.99 M | 144.9 | 55.5 | 27.8 | 16.7 | 166.5 | 83.3 | 50.0 | 959 | 1798 | 1200 |

## T7c Screening-slice evaluation accounting (160 bases x 20 cases = 3,200 cases per arm)

| design | arms | cases/arm | games | hours @18.9 games/s | hours @1.5 games/s |
|---|---|---|---|---|---|
| shared_library_16_arms | 16 | 3200 | 51200 | 0.75 | 9.48 |
| per_seed_library_18_arms | 18 | 3200 | 57600 | 0.85 | 10.67 |

Teacher-regime learning-curve probe (optional): 8 checkpoints x 2 models x 512 games = 8192 teacher games, about 2 minutes at 60 games/s.

## T7d Confirmation-slice accounting by B (not opened in the calibration stage)

| B | design | cases/arm | games | hours @18.9 | hours @1.5 | paired games per opponent (pooled) | per opponent x colour | per family |
|---|---|---|---|---|---|---|---|---|
| 80 | 7_arms_shared_library | 1600 | 11200 | 0.16 | 2.07 | 480 | 240 | 300 |
| 80 | 9_arms_per_seed_library | 1600 | 14400 | 0.21 | 2.67 | 480 | 240 | 300 |
| 112 | 7_arms_shared_library | 2240 | 15680 | 0.23 | 2.90 | 672 | 336 | 420 |
| 112 | 9_arms_per_seed_library | 2240 | 20160 | 0.30 | 3.73 | 672 | 336 | 420 |
| 160 | 7_arms_shared_library | 3200 | 22400 | 0.33 | 4.15 | 960 | 480 | 600 |
| 160 | 9_arms_per_seed_library | 3200 | 28800 | 0.42 | 5.33 | 960 | 480 | 600 |
| 240 | 7_arms_shared_library | 4800 | 33600 | 0.49 | 6.22 | 1440 | 720 | 900 |
| 240 | 9_arms_per_seed_library | 4800 | 43200 | 0.63 | 8.00 | 1440 | 720 | 900 |
| 320 | 7_arms_shared_library | 6400 | 44800 | 0.66 | 8.30 | 1920 | 960 | 1200 |
| 320 | 9_arms_per_seed_library | 6400 | 57600 | 0.85 | 10.67 | 1920 | 960 | 1200 |
| 480 | 7_arms_shared_library | 9600 | 67200 | 0.99 | 12.44 | 2880 | 1440 | 1800 |
| 480 | 9_arms_per_seed_library | 9600 | 86400 | 1.27 | 16.00 | 2880 | 1440 | 1800 |
| 640 | 7_arms_shared_library | 12800 | 89600 | 1.32 | 16.59 | 3840 | 1920 | 2400 |
| 640 | 9_arms_per_seed_library | 12800 | 115200 | 1.69 | 21.33 | 3840 | 1920 | 2400 |

