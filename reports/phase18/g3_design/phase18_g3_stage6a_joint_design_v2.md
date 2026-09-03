# Phase 18 Agent 6 — Stage 6A replacement design: the matched two-lineage joint-bundle pilot (v2, revised)

**Stage 6A of the G3 design work package. Analysis and design only; replaces the setup-only
C0 evaluation design.** This document freezes no contract and authorizes no run. In producing
it and its revisions no Stratego game was played, no setup pool was generated, no model was
trained or loaded, no bundle was written, no held-out material was read, and nothing was
pushed. Its tables are reproduced by `scripts/phase18_g3_joint_design_tables.py` (`--write`
regenerates `reports/phase18/g3_design/phase18_g3_joint_design_tables_v2.json`, `--check`
recomputes and compares) and are reprinted verbatim in the appendix. The superseded
setup-only design and all of its evidence stay in `phase18_g3_stage6a_analysis_v1.md`, marked
superseded at its head.

Date: 2026-09-02. Author: Phase 18 Agent 6 (G3 design agent). Revised the same day on the
operator's fourth-round instruction; the revision record is section 0.1.

```text
backend-foundation commit   d37ebe77ae388d87267d47e75223967574332f94   (operator-named, published)
design branch               phase18/g3-design, worktree output/phase18/worktrees/g3-design/
prior commits preserved     286acb33 (Stage 6A), 225cc77 (correction), d974452 (joint-bundle replacement);
                            this revision is a new commit on top of d974452
runtime location            output/phase18/runtime/  (storage policy item 3)
protected file              reports/phase13/phase14_launch_manifest_v1.json: never staged, never modified
```

### 0.1 Revision record (this commit against d974452)

1. The primary gate `bundle_final − bundle_0` is replaced by a matched two-lineage
   comparison: a candidate lineage in which C1 and the setup model start from their recorded
   initial states and train together with setup updates enabled, against a control lineage
   in which C1 starts from the same initial state, receives the same number and type of C1
   training updates, and trains throughout with a setup source frozen at the recorded initial
   setup model.
2. Each lineage keeps internally matched components; the candidate setup model is never
   evaluated with the control policy, nor the frozen setup model with the candidate policy.
3. The final candidate bundle is evaluated against the final control bundle on the same G1
   evaluator schedule, handcrafted rule-based opponents, opponent formations, colours and
   evaluation seeds.
4. `basic_heuristic` joins the seven handcrafted opponents (eight opponents; 16 cases per base).
5. The 0.05 EWR practical-margin rule is retained for the primary candidate-versus-control
   comparison.
6. The bounded pilot uses one seed; a second seed is a conditional follow-up only when the
   first result is near the decision boundary or operationally irregular. The three-seed
   consistency requirement is removed from the pilot.
7. `candidate_final` versus `candidate_128` and `candidate_final` versus `candidate_0` are
   progress diagnostics only; they cannot decide whether setup integration helped.
8. The estimand is stated: the primary comparison estimates the total benefit of enabling
   setup learning together with the policy co-adaptation it causes; it does not isolate the
   setup network's direct causal contribution.
9. The simplified gates, the slower G1 evaluator, the minimal restart test, the absence of a
   standalone preflight and the reduced reporting of d974452 are retained. Only the
   calculations affected by the comparison were updated (appendix J3, J5, J6).

---

## 1. Operator decisions applied

Third round (d974452): matched joint bundles that bind C1 weights and required state, setup
raw and EMA weights, both optimizer states, training counters and random-state identities
under one bundle identifier; evaluation of bundles from the same run, each with its own
contemporaneous policy and setup model, with cross-pairing across checkpoints or runs
forbidden; the fixed C0 mover and the fixed-library comparison removed as decision gates;
ten mandatory gates; the in-process evaluator and R1–R11, the standalone preflight, the
teacher-regime probe, the concentration review flags, the raw/EMA parameter distance,
non-gating strata reports, redundant Monte Carlo tables and unrelated tests removed or
optional; the proven G1 evaluator with a minimum adapter; one new restart test; operational
measurements taken during the pilot, not as gates.

Fourth round: the two-lineage primary comparison and the points of section 0.1.

Still in force from earlier rounds: pool-versus-pool collection under `TRAINING_RULES`
(battleless 100); play evaluation under `EVALUATION_RULES` (battleless 200, P18-A001);
the EMA is the setup model used in evaluation and the raw setup model generates and learns;
`T = 202` plies per period with `S = 2,560` slots provisional under the four-outcomes-per-
setup constraint; 160 evaluation bases, 10 per family; the unusual/procedural stratum
deferred to G4; `output/phase18/runtime/` for runtime data.

---

## 2. The bounded two-lineage pilot

### 2.1 Initial states shared by both lineages

```text
C1 (policy / value / belief)   the canonical fresh initialization, seed 2026081302, digest cfe60bb0…e042b8,
                               863,959 parameters (P18-D001; the reproduction contract's initialisation file)
setup model                    Phase18SetupModel, 802,320 parameters, one fresh initialization for the pilot seed
                               (derive_stream_seed(namespace, 'model_init', k)); this recorded initial version is
                               the candidate's starting point AND the control's frozen setup source
optimizers                     both fresh: C1 AdamW (the frozen Phase 8 configuration: lr 0.001, weight decay 0.01,
                               500-update warmup, batch 256); setup AdamW (lr 5e-5, weight decay 0)
bundle_0                       written by each lineage before any update from identical components; the two
                               bundle_0 manifests differ only in lineage id
```

Starting both models fresh is the common contract's rule for the final run (section 1) and
keeps the pilot's dynamics representative of the production run's first periods. Warm-starting
C1 from the G1 reproduction checkpoint is not adopted here and remains an operator decision
(section 9).

### 2.2 The two lineages

```text
candidate lineage      C1 and the setup model train together: every period ends with K supervised C1 updates
                       and one setup update (five epochs, one optimizer step per minibatch of ready rows
                       including a final partial one) followed by one EMA update; the next pool is sampled by
                       the updated raw setup model
control lineage        C1 starts from the same initial state and receives the same K supervised C1 updates per
                       period of the same type (the same canonical/live batch mixture, the same batch size,
                       the same schedule and shuffle seeds); the setup model is FROZEN at the recorded initial
                       version: no setup update, no EMA update, the setup optimizer never steps; every pool is
                       sampled by that frozen model under the same pool seeds the candidate uses
matched randomness     both lineages use the same collector seeds (period, slot, ordinal), the same pool seeds,
                       the same cell order and the same C1 shuffle streams; they differ only in whether setup
                       updates are applied, so their period-1 pools, games, live examples and C1 updates are
                       identical by construction and the lineages diverge from period 2 through the candidate's
                       changed pools
built-in check         after period 1 the two lineages' C1 digests and live-example digests must be identical
                       (recorded as an operational check of the matching; a difference is an implementation defect)
what differs by design the candidate's live stream is generated from setups its own setup model keeps changing;
                       the control's live stream is generated from setups of the frozen initial model. The C1 of
                       each lineage co-adapts to its own stream.
```

### 2.3 Streams, the period loop and the bounded budget

```text
canonical anchor stream        the accepted Phase 8 train corpus (digests verified through the checkpoint's corpus
                               identity), policy / value / belief supervision with the frozen teacher weights
live stream                    teacher-schedule games played from the lineage's setup pool (collector below); every
                               completed game yields two setup outcomes into the setup buffer (consumed only in the
                               candidate) and a trajectory converted by the Phase 8 example builder
                               (examples_for_game) into warmstart examples with the same targets and teacher
                               weights; live examples enter only the C1 training split (common contract 9.2)
collector                      unchanged from the superseded design, section 3.2: S = 2,560 slots (provisional),
                               T = 202 plies per period, cyclic cell order over the frozen 100-cell schedule,
                               1,024-setup pool per period, retention 21 periods, TRAINING_RULES, fixed slot order,
                               seeds through derive_stream_seed (about 1,998 completions and 3,996 outcomes per
                               period at the corpus mean length; appendix J2)
per period (each lineage)      (1) advance every slot T plies, starting new games from the current pool;
                               (2) K supervised C1 updates on the canonical/live mixture (batch 256);
                               (3) candidate only: one setup update and one EMA update;
                               (4) filter the buffer, regenerate the pool from the lineage's setup model;
                               (5) write a joint bundle every 32 periods and at the end
provisional variables          K (default 64), the canonical:live batch mixture (default 1:1), the live-example
                               retention window (default: the last 32 periods), S — research variables (method map
                               S33), identical in both lineages, recorded in the launch record before period 1 and
                               reported as observations, never tuned inside the pilot
bounded length and seed        256 periods per lineage; ONE seed. A second seed is run only as a conditional
                               follow-up (section 4.4)
```

Cost (appendix J2, J3): a period costs about 173 CPU-seconds of teacher games plus 9 s of C1
updates at K = 64 in each lineage; the two-lineage pilot at 256 periods is 1.02 M games,
32,768 C1 updates, about 63.8 M live examples and 12.9 GB of trajectories in total, 25.9
CPU-hours and 4.1–10.8 wall-hours across the 100–30 games-per-second bracket. Example
building and memory are unmeasured and are recorded in the pilot.

---

## 3. The joint checkpoint bundle

```text
component                                   existing infrastructure reused
C1 weights (policy, value, belief heads)    warmstart_checkpoint_v1 (stratego/training/warmstart_checkpoint.py):
+ required state, C1 optimizer state,       atomic temp-file -> fsync -> reload-validate -> replace; whole-payload
C1 update counter, corpus cursor, RNG       integrity digest; collect_rng_state; resume identity check (train config
state                                       and corpus identity by digest); load_model_for_evaluation for the evaluator
setup raw weights, setup optimizer state,   SetupTrainer.save_checkpoint / load_checkpoint (stratego/training/phase18/
setup EMA weights, setup update and EMA     setup_learning.py): raw.pt, optimizer.pt, ema.pt and a manifest with their
counters                                    sha256s, optimizer_step_count, ema_updates, config digest. In the control
                                            lineage these are the recorded initial raw weights, an unstepped optimizer
                                            and an EMA equal to the initial weights, with both counters at zero
collector state: every active slot's        Phase 17 capture_active_game / restore_active_game and
engine state, game id, cell, pool           write_joint_checkpoint / read_joint_checkpoint with CheckpointIdentity
fingerprints and ply; the setup buffer's    (stratego/training/phase17/checkpoint.py) for the slots; the SetupBuffer
rows, counts, means, ready flags and        rows and counters; the period and stream namespaces
period; the pool telemetry; the period
counter and stream seed identities
bundle manifest                             run id, LINEAGE id (candidate or control), setup_updates_enabled flag,
                                            seed index, period, C1 update count, setup update count, EMA update count,
                                            component file sha256s, RNG identities; bundle_id = sha256 over the manifest
```

Rules: a bundle is written and loaded only as a whole; every component's digest must equal
the digest bound in the manifest; the evaluator refuses a bundle whose components do not
verify or whose manifest names another run, lineage or period (gate 5). No component of one
bundle may be paired with a component of another bundle, lineage or run, in training or in
evaluation: the candidate setup model is never evaluated with the control policy, and the
frozen setup model is never evaluated with the candidate policy. Estimated size about 46 MB
per bundle (appendix J4); eight bundles per lineage per 256 periods.

---

## 4. Evaluation: final candidate bundle against final control bundle

### 4.1 The primary comparison and what it estimates

```text
arms                    candidate_final (the candidate lineage's period-256 bundle) and control_final (the control
                        lineage's period-256 bundle); each arm plays with its OWN C1 (greedy, CPU float32 through
                        the G1 harness) and its OWN setup model — the candidate's EMA, the control's frozen initial
                        model — sampled per case through the orientation boundary
opponents               the eight frozen handcrafted code opponents, by policy id: basic_heuristic,
                        strategic_rule_based, tactical_rule_based, stress_scout_rush, stress_miner_rush,
                        stress_berserker, stress_information_miser, stress_chaos (no neural opponent; no weights)
opponent formations     library validation bases 400..409 (160 bases, 10 per family); the same formations,
                        colours, opponents and evaluation seeds for every arm; identical schedule digest across
                        arms (as G1)
rules                   EVALUATION_RULES
cases                   160 bases x 8 opponents x 2 colours = 2,560 per arm
primary contrast        EWR(candidate_final) − EWR(control_final), paired by case; stratified cluster bootstrap
                        (bases resampled within families, carrying every case, colour and arm of the base;
                        finite-stratum rescaling sqrt(n_f / (n_f − 1)); validated in the superseded design's T5);
                        PROCEED requires the 95% lower bound above zero AND the point estimate at least 0.05
                        (the frozen practical-margin scope)
```

**What the primary comparison estimates.** It estimates the total benefit of enabling setup
learning: the effect of the learned setups on the integrated system together with the policy
co-adaptation that training on a changing setup stream causes, relative to a system that
received identical policy training on the frozen initial setup source. It does **not** isolate
the setup network's direct causal contribution: the candidate's policy and its setup model
differ from the control's jointly, and no arm separates them. It also says nothing about
unfamiliar setups.

### 4.2 Progress diagnostics (never decisive)

`candidate_final` versus `candidate_128` and `candidate_final` versus `candidate_0`, on the same
schedule, are progress diagnostics only: they show that the candidate lineage moved, not that
setup integration helped, because the control lineage moves too. `control_final` versus
`control_0` is optional telemetry (the two `bundle_0` arms share identical components, so
`candidate_0`'s games serve both). None of these readings may substitute for the primary
comparison.

### 4.3 Cost and resolution

Appendix J5: the primary comparison is 5,120 games, about 0.95 h on the G1 harness; with the
two candidate diagnostics 10,240 games and 1.9 h; with an optional `control_128` arm 12,800
games and 2.4 h. Appendix J6: at 160 bases and 16 cases per base the one-seed contrast has a
standard error of 0.011–0.021 and a 95% half-width of 0.021–0.042 across `rho_b` from 0 to
0.20; the combined rule passes a true effect of 0.08 with probability 0.92–0.998 and 0.10 with
at least 0.99, and a true effect of exactly 0.05 with probability 0.5 whatever the sample. The
planning variance V is the frozen paired per-game instrument, which may understate a contrast
whose two arms also differ in mover; the reported interval comes from the measured
stratified per-base variance, not from V.

### 4.4 One seed, and the conditional second seed

The bounded pilot runs one seed (one candidate lineage and one control lineage). A second
seed — a second matched lineage pair — is run only as a conditional follow-up, and its
contrast is pooled with the first by the same base-cluster bootstrap (appendix J6, seeds = 2),
when either holds:

```text
near the decision boundary   the 95% interval of the candidate-minus-control contrast contains 0.05 (the pass
                             or fail would turn on the interval's draw); a result whose interval lies entirely
                             above 0.05 passes, and one entirely below fails, without a second seed
operationally irregular      any mandatory gate failure, any accounting anomaly, or an operational observation
                             far outside the appendix J2 expectations (recorded in the pilot packet and judged
                             at review)
```

The three-seed consistency requirement of the superseded design does not apply to the pilot.

---

## 5. Mandatory gates and operational observations

```text
G1  legal setup generation          legality failures = 0 in every pool and evaluation sample (S02 mask)
G2  orientation / reflection        orientation failures = 0; every played setup passes the accepted boundary
                                    helper (S07); reflection flags recorded and the S06 round-trip test passes
G3  exact setup-to-outcome          attribution failures = 0: every completed game's two outcomes attribute to the
    attribution                     rows whose fingerprints it carries; the buffer raises on any other outcome
G4  completed-game accounting       per period and lineage: started = completed + in-flight + failed, no game id
                                    completed twice, no completed game without an outcome record; per evaluation
                                    lane: planned = completed + failed + missing
G5  exact joint-bundle identity     the manifest binds every component digest, the run and the lineage; only whole
                                    bundles are loaded; any digest, run, lineage or period mismatch is refused; no
                                    cross-checkpoint or cross-lineage pairing
G6  minimal checkpoint / resume     the restart test of section 6 passes
    equivalence
G7  paired evaluation               identical schedule digest, opponents, formations, colours and seeds across arms
G8  duplicate / diversity collapse  distinct reflection classes >= 922 of 1,024 in every pool (fresh model:
                                    1,024/1,024 in all 384 G2 pools); the single retained diversity check
G9  finite losses, valid files      non-finite events = 0 in every trainer; every bundle reloads and validates
                                    (whole-payload digests)
G10 clean committed deliverable     no protected or sealed artifact modified; the Phase 14 manifest never staged
```

Operational observations, recorded per period and lineage and reported, never gating:
seconds per period, collector games per second, C1 examples per second, memory high-water
mark, completions per period, outcomes per generated setup (mean, fraction with zero), ready
rows, minibatches and optimizer steps, snapshot-age distribution, cross-period attributions,
duplicates collapsed, the period-1 lineage-identity check of section 2.2, and the progress
diagnostics. Slow execution alone is not a stop condition. Optional telemetry, not gates: the
concentration statistics of the superseded design's section 7, the raw/EMA parameter
distance, and per-opponent, per-family and per-colour breakdowns.

---

## 6. The restart test (the one new end-to-end test)

```text
setup      a tiny candidate-lineage configuration on CPU float32 with fixed threads: few slots, short T, small pool,
           so that at least one game is unfinished at the save point
save       run to a period boundary n and write bundle_n with its unfinished games captured
control    continue uninterrupted through period n + 1; record the completed game ids and outcomes of
           period n + 1, the attribution map (fingerprint -> outcome multiset), the per-period accounting, and
           the digests of C1 weights, setup raw and EMA weights, buffer state and the bundle written at n + 1
restart    in a fresh process load bundle_n, finish period n + 1, record the same
assert     no outcome lost, duplicated or misattributed: identical completed-game id sets, identical outcome
           records, identical attribution map, identical accounting; identical next-period state and results:
           identical digests of C1 weights, setup raw and EMA weights, buffer state, receipts, and identical
           component digests in the bundle written at n + 1
```

The control lineage runs the same code path with setup updates disabled; the test may be
repeated in that configuration but is required once, on the candidate configuration. Reused:
`warmstart_checkpoint_v1` (C1, optimizer, cursor, RNG state, resume identity), `SetupTrainer`
checkpoints, Phase 17 `capture_active_game` / `restore_active_game`, the `SetupBuffer` state
and telemetry. Equivalence is proved on CPU; MPS is not bitwise reproducible (P18-D002).

---

## 7. Superseded, removed, and retained

```text
superseded (evidence kept)       from v1: the frozen C0 mover; the fixed-library comparator gate; the P1 / P2 setup
                                 contrasts; the confirmation sizing rule and range; the calibration stage as a
                                 setup-only screen; the full-gate Monte Carlo for P1 / P2; the three-seed consistency
                                 requirement. From d974452: bundle_final − bundle_0 as the primary gate (now a
                                 progress diagnostic)
removed or optional              the in-process evaluator and R1–R11; the separate preflight; the teacher-regime probe;
                                 the concentration review flags R2–R10; the raw/EMA parameter distance; non-gating
                                 opponent / family / colour reports; redundant Monte Carlo tables; tests unrelated to
                                 files or behaviour the agent changes
retained from v1                 the teacher-schedule facts and the outcome-signal profile (v1 2.1); the throughput,
                                 runtime and storage evidence (2.6); the library, leakage and data-boundary record
                                 (2.3–2.5); the asynchronous collector reconciled with S21–S23 (3.2); the EMA
                                 parameter-aging record with no performance inference (4); the variance model, the
                                 stratified cluster bootstrap and the direct per-base variance with their validation (5)
```

---

## 8. Engineering items

```text
G3-ENG-01  asynchronous pool-driven teacher-schedule collector (v1 section 3.2, unchanged), also emitting each
           completed game's trajectory to the live example stream; parameterized by the lineage's setup model
G3-ENG-02  joint period loop with a lineage switch: K supervised C1 updates per period from the canonical/live
           mixture through the accepted warmstart trainer and dataset machinery and the Phase 8 example builder;
           the setup update and EMA only when setup updates are enabled; period accounting; operational telemetry;
           the period-1 lineage-identity check
G3-ENG-03  joint bundle save / load with the manifest (run, lineage, setup_updates_enabled, counters, digests) and
           bundle_id, built on warmstart_checkpoint_v1, the SetupTrainer checkpoint and the Phase 17 active-game
           capture; the section 6 restart test
G3-ENG-04  the minimal G1 evaluator adapter for bundles and the eight handcrafted opponents: load a bundle, verify
           its manifest, export its C1 to the frozen evaluation checkpoint format with the bundle id stamped into
           the policy token and every receipt, resolve the arm's own setup per case from the bundle's own setup
           model under the case seed, resolve the opponent by policy id; the G1 schedule, pairing, receipts,
           accounting and paired bootstrap unchanged. Focused tests: a mismatched component digest is refused;
           a cross-lineage pairing (candidate C1 with the frozen setup model, or the reverse) is refused; the same
           case seed reproduces the same own setup for the same bundle and a different bundle produces its own;
           the schedule digest is identical across arms; accounting reconciles on a tiny schedule; a
           handcrafted-opponent game runs under EVALUATION_RULES and writes a receipt
G3-ENG-05  analysis and packet: stratified cluster bootstrap, the candidate-minus-control contrast with the
           near-boundary rule, the progress diagnostics, the ten gates, the operational observations
```

The pilot's execution instruction (Stage 6C) keeps a mandatory review stop before the first
collection game, but no separate preflight stage: throughput, memory, completion rate and
outcomes per setup are measured in the pilot itself.

---

## 9. Decisions for the operator

1. Pilot initialization: both models fresh (this design) or C1 warm-started from the G1
   reproduction checkpoint.
2. Provisional defaults: K = 64 C1 updates per period, canonical:live mixture 1:1, live
   retention 32 periods, pilot length 256 periods, bundle cadence 32, diagnostic bundle at 128.
3. Whether the optional `control_128` arm and the `control_final` versus `control_0`
   telemetry are included.
4. Whether the reserved validation bases 410..449 remain reserved for a later confirmation
   or are released.

---

## 10. What this stage did not do; checks run

No game, pool, model, bundle, training update, evaluation game, confirmation outcome or
held-out access; nothing pushed; prior commits preserved. Checks run for this commit:
`scripts/phase18_g3_joint_design_tables.py --check`, `scripts/phase18_g3_stage6a_tables.py
--check` (the superseded tables still reproduce), JSON validity of both table files,
`tests/training/phase18/test_setup_learning.py` (the setup checkpoint and EMA behaviour the
bundle reuses; no package code changed in this commit), and `git diff --check`.

---

## Appendix — tables (verbatim output of `scripts/phase18_g3_joint_design_tables.py --check`)

## J1 Constants used by the joint design (in addition to Stage 6A T3)

| constant | value | source |
|---|---|---|
| C1_SECONDS_PER_UPDATE | 0.1407 | reports/phase18/phase18_g1_control_run_v1.json: (run.wall_seconds - run.validation_seconds) / run.updates_completed |
| C1_UPDATES_MEASURED | 25000 | same file, run.updates_completed |
| C1_BATCH_SIZE | 256 | reports/phase18/phase18_g1_control_run_v1.json:/restart_proof/cursor_at_boundary/batch_size |
| SELECTED_DECISIONS_PER_GAME | 62.3590 | reports/phase_8_data/agent_02_corpus_manifest.json: decision_totals.per_split.train.mean_selected_decisions |
| TRAJECTORY_BYTES_PER_GAME | 12606.2661 | reports/phase_8_data/agent_02_corpus_manifest.json: storage.total_bytes / decision_totals.totals.games |
| SECONDS_PER_TEACHER_GAME | 0.0865 | reports/phase_4_data/agent_04_baseline_league_raw.csv: mean wall_clock_seconds over 44544 rule-vs-rule games |
| MEAN_PLIES_TRAINING_RULES | 258.8130 | reports/phase_8_data/agent_02_corpus_manifest.json:decision_totals.per_split.train.mean_plies |
| G1_HARNESS_GAMES_PER_SECOND | 1.4959 | reports/phase18/g1_random_confirmation/run_v1.json: completed games / seconds |
| C0_CHECKPOINT_BYTES | 10459947 | reports/phase18/phase18_g1_random_confirmation_contract_v1.json:checkpoints.candidate.bytes |

## J2 Joint period cost: S = 2560 slots (provisional) x T = 202 plies; 1998 completions, 3996 outcomes (3.90 per setup), 124596 live examples, 25.2 MB of trajectories per period

| C1 updates per period | C1 seconds | collector CPU s | wall s @30 games/s | @60 | @100 |
|---|---|---|---|---|---|
| 32 | 4.5 | 173 | 71 | 38 | 24 |
| 64 | 9.0 | 173 | 76 | 42 | 29 |
| 128 | 18.0 | 173 | 85 | 51 | 38 |
| 256 | 36.0 | 173 | 103 | 69 | 56 |

## J3 Bounded pilot budget: two lineages (candidate and control), one seed (cost only; example building and memory are measured in the pilot)

| periods | C1 updates/period | games per lineage | C1 updates per lineage | live examples per lineage | CPU h per lineage | CPU h per pilot | wall h per pilot @30 | @60 | @100 | trajectories GB per pilot | receipts MB per pilot |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 0.13 M | 4096 | 8.0 M | 3.2 | 6.5 | 2.7 | 1.5 | 1.0 | 3.2 | 41 |
| 64 | 128 | 0.13 M | 8192 | 8.0 M | 3.4 | 6.8 | 3.0 | 1.8 | 1.4 | 3.2 | 41 |
| 128 | 64 | 0.26 M | 8192 | 15.9 M | 6.5 | 12.9 | 5.4 | 3.0 | 2.1 | 6.4 | 82 |
| 128 | 128 | 0.26 M | 16384 | 15.9 M | 6.8 | 13.6 | 6.0 | 3.6 | 2.7 | 6.4 | 82 |
| 256 | 64 | 0.51 M | 16384 | 31.9 M | 12.9 | 25.9 | 10.8 | 6.0 | 4.1 | 12.9 | 164 |
| 256 | 128 | 0.51 M | 32768 | 31.9 M | 13.6 | 27.1 | 12.0 | 7.3 | 5.4 | 12.9 | 164 |

## J4 Joint bundle size (estimate): C1 weights 10.5 MB + C1 optimizer 6.9 MB + setup raw/optimizer/EMA 12.9 MB + collector state 15.9 MB = 46 MB; 8 bundles per 256 periods at every 32 = 369 MB

## J5 Candidate-versus-control evaluation on the G1 harness: 160 bases x 8 handcrafted opponents x 2 colours = 2560 cases per arm; 1.50 games/s

| seeds | arm set | arms | games | hours (G1 harness) | paired games per opponent | per opponent x colour | per family |
|---|---|---|---|---|---|---|---|
| 1 | primary: candidate_final vs control_final | 2 | 5120 | 0.95 | 320 | 160 | 160 |
| 1 | primary + candidate diagnostics (candidate_128, candidate_0) | 4 | 10240 | 1.90 | 320 | 160 | 160 |
| 1 | primary + candidate diagnostics + control_128 (optional) | 5 | 12800 | 2.38 | 320 | 160 | 160 |
| 2 | primary: candidate_final vs control_final | 2 | 10240 | 1.90 | 640 | 320 | 320 |
| 2 | primary + candidate diagnostics (candidate_128, candidate_0) | 4 | 20480 | 3.80 | 640 | 320 | 320 |
| 2 | primary + candidate diagnostics + control_128 (optional) | 5 | 25600 | 4.75 | 640 | 320 | 320 |

## J6 Resolution of the paired candidate-minus-control contrast at B = 160 bases, 16 cases per base (rho_w = 0.10); the near-boundary zone is the half-width around 0.05

| seeds | rho_b | per-base SD | n_eff | SE | 95% half-width | MDE (lower>0, 80%) | d80 combined rule | P(pass at 0.05) | P(pass at 0.08) | P(pass at 0.10) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0 | 0.1348 | 2560 | 0.0107 | 0.0209 | 0.0299 | 0.0590 | 0.500 | 0.998 | 1.000 |
| 1 | 0.05 | 0.1783 | 1463 | 0.0141 | 0.0276 | 0.0395 | 0.0619 | 0.500 | 0.983 | 1.000 |
| 1 | 0.1 | 0.2131 | 1024 | 0.0168 | 0.0330 | 0.0472 | 0.0642 | 0.500 | 0.963 | 0.999 |
| 1 | 0.2 | 0.2696 | 640 | 0.0213 | 0.0418 | 0.0597 | 0.0679 | 0.500 | 0.920 | 0.991 |
| 2 | 0.0 | 0.1000 | 4655 | 0.0079 | 0.0155 | 0.0221 | 0.0567 | 0.500 | 1.000 | 1.000 |
| 2 | 0.05 | 0.1551 | 1932 | 0.0123 | 0.0240 | 0.0344 | 0.0603 | 0.500 | 0.993 | 1.000 |
| 2 | 0.1 | 0.1953 | 1219 | 0.0154 | 0.0303 | 0.0433 | 0.0630 | 0.500 | 0.974 | 0.999 |
| 2 | 0.2 | 0.2575 | 701 | 0.0204 | 0.0399 | 0.0570 | 0.0671 | 0.500 | 0.930 | 0.993 |
