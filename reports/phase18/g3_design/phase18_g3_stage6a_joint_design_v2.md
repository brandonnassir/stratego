# Phase 18 Agent 6 — Stage 6A replacement design: the matched joint-bundle pilot (v2)

**Stage 6A of the G3 design work package. Analysis and design only; replaces the setup-only
C0 evaluation design.** This document freezes no contract and authorizes no run. In producing
it no Stratego game was played, no setup pool was generated, no model was trained or loaded,
no bundle was written, no held-out material was read, and nothing was pushed. Its tables are
reproduced by `scripts/phase18_g3_joint_design_tables.py` (`--write` regenerates
`reports/phase18/g3_design/phase18_g3_joint_design_tables_v2.json`, `--check` recomputes and
compares) and are reprinted verbatim in the appendix. The superseded design and all of its
evidence stay in `phase18_g3_stage6a_analysis_v1.md`, now marked superseded at its head.

Date: 2026-09-02. Author: Phase 18 Agent 6 (G3 design agent).

```text
backend-foundation commit   d37ebe77ae388d87267d47e75223967574332f94   (operator-named, published)
design branch               phase18/g3-design, worktree output/phase18/worktrees/g3-design/
prior commits preserved     286acb33 (Stage 6A), 225cc77 (Stage 6A correction); this is a new commit on top
runtime location            output/phase18/runtime/  (storage policy item 3)
protected file              reports/phase13/phase14_launch_manifest_v1.json: never staged, never modified
```

---

## 1. Operator decisions applied (2026-09-02, third round)

1. **Matched joint-system evaluation.** The trained setup model is never evaluated with an
   unrelated policy checkpoint. A joint checkpoint bundle binds move-policy weights,
   value/belief weights and required state, setup raw weights, setup EMA weights, both
   optimizer states, training counters and random-state identities under one bundle
   identifier. Evaluation compares a later bundle with a predeclared earlier bundle of the
   same run, each using its own contemporaneous policy and setup model; cross-pairing
   components across checkpoints or runs is forbidden. Both bundles face the same frozen
   handcrafted rule-based opponents, opponent formations, colours and game seeds. The primary
   conclusion is integrated-system improvement, not isolated setup-model causation. The
   fixed C0 mover and the fixed-library comparison are removed as G3 decision gates. The
   initial matched bundle is the fixed baseline; later matched bundles are candidates;
   candidate-versus-previous-checkpoint comparisons are progress diagnostics only.
2. **Reduced acceptance requirements**: the ten mandatory gates of section 5; the
   in-process evaluator and R1–R11, the separate preflight, the teacher-regime probe, the
   bomb/flag concentration review flags, the per-period raw/EMA parameter distance, the
   non-gating strata reports, redundant Monte Carlo tables and unrelated tests are removed
   or optional. The proven G1 evaluator is used with only the minimum adapter and focused
   tests.
3. **Minimal restart requirement**: reuse existing checkpoint infrastructure; one new
   end-to-end restart test (section 6).
4. **No standalone preflight**: speed, memory, completion rate and outcomes per setup are
   measured during the bounded joint-training pilot as operational observations, not gates;
   slow execution alone is not a stop condition.
5. **Deliverable**: this concise replacement design; the prior design marked superseded with
   its evidence preserved; only the supporting calculations that remain relevant updated.

Design choices settled earlier and still in force: pool-versus-pool collection under
`TRAINING_RULES` (battleless 100); play evaluation under `EVALUATION_RULES` (battleless 200,
P18-A001); independent setup initialization per training seed; the EMA is the setup model
used in evaluation, the raw setup model generates and learns; `T = 202` plies per period,
`S = 2,560` slots provisional under the four-outcomes-per-setup constraint; 160 evaluation
bases, 10 per family; the unusual/procedural stratum deferred to G4; `output/phase18/runtime/`
for runtime data.

---

## 2. The bounded joint-training pilot

### 2.1 Initialization and the baseline bundle

```text
C1 (policy / value / belief)   the canonical fresh initialization, seed 2026081302, digest cfe60bb0…e042b8,
                               863,959 parameters (P18-D001; the reproduction contract's initialisation file)
setup model                    Phase18SetupModel, 802,320 parameters, one independent fresh initialization per
                               training seed (derive_stream_seed(namespace, 'model_init', k))
optimizers                     both fresh: C1 AdamW (the frozen Phase 8 configuration: lr 0.001, weight decay 0.01,
                               500-update warmup, batch 256); setup AdamW (lr 5e-5, weight decay 0)
bundle_0                       the initial matched bundle written before any update: the FIXED BASELINE
```

Starting both models fresh is the common contract's rule for the final run (section 1) and
keeps the pilot's dynamics representative of the production run's first periods. The
alternative of warm-starting C1 from the G1 reproduction checkpoint is not adopted here and
is listed as an operator decision (section 9).

### 2.2 Streams and the period loop

```text
canonical anchor stream        the accepted Phase 8 train corpus (digests verified through the checkpoint's
                               corpus identity), policy / value / belief supervision with the frozen teacher weights
live stream                    teacher-schedule games played from the setup pool (collector below); every
                               completed game yields (a) two setup outcomes into the setup buffer and (b) a
                               trajectory converted by the Phase 8 example builder (examples_for_game) into
                               warmstart examples with the same targets and teacher weights; live examples enter
                               only the C1 training split (common contract 9.2)
collector                      unchanged from the superseded design, section 3.2: S = 2,560 slots (provisional),
                               T = 202 plies per period, cyclic cell order over the frozen 100-cell schedule,
                               1,024-setup pool per period, retention 21 periods, TRAINING_RULES, fixed slot order,
                               seeds through derive_stream_seed (about 1,998 completions and 3,996 outcomes per
                               period at the corpus mean length; appendix J2)
per period                     (1) advance every slot T plies, starting new games from the current pool;
                               (2) K supervised C1 updates on a canonical/live mixture (batch 256);
                               (3) one setup update: five epochs, one optimizer step per minibatch of ready rows
                               including a final partial minibatch, then one EMA update;
                               (4) filter the buffer, regenerate the pool;
                               (5) write a joint bundle every 32 periods and at the end
provisional variables          K (default 64), the canonical:live batch mixture (default 1:1), the live-example
                               retention window (default: the last 32 periods), S — all research variables
                               (method map S33); their values are recorded in the launch record before period 1
                               and reported as observations, never tuned inside the pilot
bounded length                 256 periods per seed by default; three seeds (three independent joint runs, each
                               with its own bundle_0), unless the operator bounds the pilot to fewer (section 9)
```

Cost (appendix J2, J3): a period costs about 173 CPU-seconds of teacher games plus 9 s of C1
updates at K = 64; 256 periods per seed are 0.51 M games, 16,384 C1 updates, about 31.9 M
live examples and 6.4 GB of trajectories; wall time 2.1–5.4 h per seed across the 100–30
games-per-second bracket. Example building and memory are unmeasured and are recorded in the
pilot.

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
counters                                    sha256s, optimizer_step_count, ema_updates, config digest
collector state: every active slot's        Phase 17 capture_active_game / restore_active_game and
engine state, game id, cell, pool           write_joint_checkpoint / read_joint_checkpoint with CheckpointIdentity
fingerprints and ply; the setup buffer's    (stratego/training/phase17/checkpoint.py) for the slots; the SetupBuffer
rows, counts, means, ready flags and        rows and counters; the period and stream namespaces
period; the pool telemetry; the period
counter and stream seed identities
bundle manifest                             run id, seed index, period, C1 update count, setup update count, EMA
                                            update count, component file sha256s, RNG identities; bundle_id = sha256
                                            over the manifest
```

Rules: a bundle is written and loaded only as a whole; every component's digest must equal
the digest bound in the manifest, and the evaluator refuses a bundle whose components do not
verify or whose manifest names another run or period (gate 5). No component of one bundle
may be paired with a component of another bundle or run, in training or in evaluation.
Estimated size about 46 MB per bundle (appendix J4); eight bundles per 256 periods.

---

## 4. Evaluation: fixed baseline against later matched bundles

```text
arms                    bundle_0 (fixed baseline) and the candidate bundles at predeclared periods (default 128
                        and 256); each arm plays with its OWN contemporaneous C1 (greedy, CPU float32 through the
                        G1 harness) and its OWN setup EMA, sampled per case through the orientation boundary
opponents               the seven frozen handcrafted code opponents of the accepted roster, by policy id:
                        strategic_rule_based, tactical_rule_based, stress_scout_rush, stress_miner_rush,
                        stress_berserker, stress_information_miser, stress_chaos (no neural opponent; no weights)
opponent formations     library validation bases 400..409 (160 bases, 10 per family); the same formations, colours,
                        opponents and game seeds for every arm; identical schedule digest across arms (as G1)
rules                   EVALUATION_RULES
cases                   160 bases x 7 opponents x 2 colours = 2,240 per arm
primary contrast        EWR(bundle_final) − EWR(bundle_0), paired by case, pooled over seeds; stratified cluster
                        bootstrap (bases resampled within families, carrying every case, colour and arm of the
                        base; finite-stratum rescaling sqrt(n_f / (n_f − 1)); validated in the superseded design's
                        T5); PROCEED requires the 95% lower bound above zero and the point estimate at least 0.05
                        (the frozen practical-margin scope), and a positive point estimate in every seed
diagnostics             bundle_U against bundle_(U−32) at every bundle: progress only; they never replace bundle_0
what is claimed         improvement of the integrated system (its policy and its setups together) over its own
                        start against the frozen handcrafted opponents; not setup-model causation, and nothing
                        about unfamiliar setups
```

Cost and resolution (appendix J5, J6): three seeds with the baseline and the final bundle
are 13,440 games, about 2.5 h on the G1 harness; adding the period-128 bundle makes 20,160
games and 3.7 h. At 160 bases with three seeds the pooled contrast has a standard error of
0.007–0.020 across `rho_b` from 0 to 0.20, so an integrated-system gain of 0.10 EWR passes
the rule with probability above 0.99 and a gain of 0.30 with certainty; the resolution is
what limits the progress diagnostics, not the gate. Because bundle_0 carries a
fresh-initialization mover, the expected gain over it is large, and the 0.05 margin is a
floor rather than a power target.

Minimal G1 evaluator adapter (`scripts/phase18_g1_random_confirmation.py` machinery reused):
load a bundle, verify its manifest, export its C1 to the frozen evaluation checkpoint format
with the bundle id stamped into the policy token and every receipt; resolve the arm's own
setup per case from the bundle's EMA under the case's own-setup seed; resolve the opponent
by policy id from the seven; keep the schedule, pairing, receipts, accounting and paired
bootstrap of the G1 script. Focused tests: a bundle with one mismatched component digest is
refused; the same case seed reproduces the same own setup for the same bundle and a different
bundle produces its own; the schedule digest is identical across arms; accounting reconciles
on a tiny schedule; a handcrafted-opponent game runs under `EVALUATION_RULES` and writes a
receipt.

---

## 5. Mandatory gates and operational observations

```text
G1  legal setup generation          legality failures = 0 in every pool and evaluation sample (S02 mask)
G2  orientation / reflection        orientation failures = 0; every played setup passes the accepted boundary
                                    helper (S07); reflection flags recorded and the S06 round-trip test passes
G3  exact setup-to-outcome          attribution failures = 0: every completed game's two outcomes attribute to the
    attribution                     rows whose fingerprints it carries; the buffer raises on any other outcome
G4  completed-game accounting       per period: started = completed + in-flight + failed, no game id completed
                                    twice, no completed game without an outcome record; per evaluation lane:
                                    planned = completed + failed + missing
G5  exact joint-bundle identity     the manifest binds every component digest; only whole bundles are loaded;
                                    any digest, run or period mismatch is refused; no cross-checkpoint pairing
G6  minimal checkpoint / resume     the restart test of section 6 passes
    equivalence
G7  paired evaluation               identical schedule digest, opponents, formations, colours and seeds across arms
G8  duplicate / diversity collapse  distinct reflection classes >= 922 of 1,024 in every pool (fresh model:
                                    1,024/1,024 in all 384 G2 pools); the single retained diversity check
G9  finite losses, valid files      non-finite events = 0 in both trainers; every bundle reloads and validates
                                    (whole-payload digests)
G10 clean committed deliverable     no protected or sealed artifact modified; the Phase 14 manifest never staged
```

Operational observations, recorded per period and reported, never gating: seconds per
period, collector games per second, C1 examples per second, memory high-water mark,
completions per period, outcomes per generated setup (mean, fraction with zero), ready rows,
minibatches and optimizer steps, snapshot-age distribution, cross-period attributions,
duplicates collapsed, and the raw-versus-EMA readings of the progress diagnostics. Slow
execution alone is not a stop condition. Optional telemetry, no longer gates: the
concentration statistics of the superseded design's section 7, the raw/EMA parameter
distance, and per-opponent, per-family and per-colour breakdowns.

---

## 6. The restart test (the one new end-to-end test)

```text
setup      a tiny configuration on CPU float32 with fixed threads: few slots, short T, small pool, so that at
           least one game is unfinished at the save point
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

Reused: `warmstart_checkpoint_v1` (C1, optimizer, cursor, RNG state, resume identity),
`SetupTrainer` checkpoints, Phase 17 `capture_active_game` / `restore_active_game`, the
`SetupBuffer` state and telemetry. Equivalence is proved on CPU; MPS is not bitwise
reproducible (P18-D002) and the operator's minimal requirement replaces the superseded
design's production-device resume proof.

---

## 7. Superseded, removed, and retained

```text
superseded (v1, evidence kept)   the frozen C0 mover; the fixed-library comparator gate; the P1 / P2 setup contrasts;
                                 the confirmation sizing rule and the confirmation range; the calibration stage as a
                                 setup-only screen; the full-gate Monte Carlo for P1 / P2
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
G3-ENG-01  asynchronous pool-driven teacher-schedule collector (v1 section 3.2, unchanged), now also emitting each
           completed game's trajectory to the live example stream
G3-ENG-02  joint period loop: K supervised C1 updates per period from the canonical/live mixture through the
           accepted warmstart trainer and dataset machinery and the Phase 8 example builder; the setup update
           and EMA; period accounting; operational telemetry
G3-ENG-03  joint bundle save / load with the manifest and bundle_id, built on warmstart_checkpoint_v1, the
           SetupTrainer checkpoint and the Phase 17 active-game capture; the section 6 restart test
G3-ENG-04  the minimal G1 evaluator adapter for bundles and handcrafted opponents, with the focused tests of
           section 4
G3-ENG-05  analysis and packet: stratified cluster bootstrap, the paired bundle contrast, the ten gates, the
           operational observations
```

The pilot's execution instruction (Stage 6C) keeps a mandatory review stop before the first
collection game, but no separate preflight stage: throughput, memory, completion rate and
outcomes per setup are measured in the pilot itself.

---

## 9. Decisions for the operator

1. Pilot initialization: both models fresh (this design) or C1 warm-started from the G1
   reproduction checkpoint.
2. Number of seeds for the bounded pilot: three (the contract's consistency requirement) or
   one (a third of the cost; no cross-seed direction check).
3. Provisional defaults: K = 64 C1 updates per period, canonical:live mixture 1:1, live
   retention 32 periods, pilot length 256 periods, bundle cadence 32, candidate bundles at
   128 and 256.
4. Whether `basic_heuristic` joins the seven handcrafted opponents.
5. Whether the 0.05 practical margin applies to the bundle contrast (this design keeps it).
6. Whether the reserved validation bases 410..449 remain reserved for a later confirmation
   or are released.

---

## 10. What this stage did not do; checks run

No game, pool, model, bundle, training update, evaluation game, confirmation outcome or
held-out access; nothing pushed; prior commits preserved. Checks run for this commit:
`scripts/phase18_g3_joint_design_tables.py --check`, `scripts/phase18_g3_stage6a_tables.py
--check` (the superseded tables still reproduce), JSON validity of both table files,
`tests/training/phase18/test_setup_learning.py` (the setup checkpoint and EMA behaviour the
bundle reuses), and `git diff --check`.

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

## J3 Bounded pilot budget per seed (cost only; example building and memory are measured in the pilot)

| periods | C1 updates/period | games | C1 updates | live examples | CPU h | wall h @30 | @60 | @100 | trajectories GB | receipts MB |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 0.13 M | 4096 | 8.0 M | 3.2 | 1.3 | 0.8 | 0.5 | 1.6 | 20 |
| 64 | 128 | 0.13 M | 8192 | 8.0 M | 3.4 | 1.5 | 0.9 | 0.7 | 1.6 | 20 |
| 128 | 64 | 0.26 M | 8192 | 15.9 M | 6.5 | 2.7 | 1.5 | 1.0 | 3.2 | 41 |
| 128 | 128 | 0.26 M | 16384 | 15.9 M | 6.8 | 3.0 | 1.8 | 1.4 | 3.2 | 41 |
| 256 | 64 | 0.51 M | 16384 | 31.9 M | 12.9 | 5.4 | 3.0 | 2.1 | 6.4 | 82 |
| 256 | 128 | 0.51 M | 32768 | 31.9 M | 13.6 | 6.0 | 3.6 | 2.7 | 6.4 | 82 |

## J4 Joint bundle size (estimate): C1 weights 10.5 MB + C1 optimizer 6.9 MB + setup raw/optimizer/EMA 12.9 MB + collector state 15.9 MB = 46 MB; 8 bundles per 256 periods at every 32 = 369 MB

## J5 Matched-bundle evaluation on the G1 harness: 160 bases x 7 handcrafted opponents x 2 colours = 2240 cases per arm; 1.50 games/s

| seeds | bundles per seed | games | hours (G1 harness) | paired games per opponent (pooled) | per opponent x colour | per family |
|---|---|---|---|---|---|---|
| 1 | 2 | 4480 | 0.83 | 320 | 160 | 140 |
| 1 | 3 | 6720 | 1.25 | 320 | 160 | 140 |
| 1 | 4 | 8960 | 1.66 | 320 | 160 | 140 |
| 1 | 9 | 20160 | 3.74 | 320 | 160 | 140 |
| 3 | 2 | 13440 | 2.50 | 960 | 480 | 420 |
| 3 | 3 | 20160 | 3.74 | 960 | 480 | 420 |
| 3 | 4 | 26880 | 4.99 | 960 | 480 | 420 |
| 3 | 9 | 60480 | 11.23 | 960 | 480 | 420 |

## J6 Resolution of the paired bundle contrast at B = 160 bases, 14 cases per base (rho_w = 0.10)

| seeds | rho_b | per-base SD | n_eff | SE | 95% half-width | MDE (lower>0, 80%) | d80 combined rule | P(pass at 0.10) | P(pass at 0.30) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0 | 0.1441 | 2240 | 0.0114 | 0.0223 | 0.0319 | 0.0596 | 1.000 | 1.000 |
| 1 | 0.05 | 0.1851 | 1358 | 0.0146 | 0.0287 | 0.0410 | 0.0623 | 1.000 | 1.000 |
| 1 | 0.1 | 0.2185 | 974 | 0.0173 | 0.0339 | 0.0484 | 0.0645 | 0.998 | 1.000 |
| 1 | 0.2 | 0.2734 | 622 | 0.0216 | 0.0424 | 0.0605 | 0.0682 | 0.990 | 1.000 |
| 3 | 0.0 | 0.0911 | 5600 | 0.0072 | 0.0141 | 0.0202 | 0.0561 | 1.000 | 1.000 |
| 3 | 0.05 | 0.1500 | 2068 | 0.0119 | 0.0232 | 0.0332 | 0.0600 | 1.000 | 1.000 |
| 3 | 0.1 | 0.1915 | 1268 | 0.0151 | 0.0297 | 0.0424 | 0.0627 | 1.000 | 1.000 |
| 3 | 0.2 | 0.2550 | 715 | 0.0202 | 0.0395 | 0.0565 | 0.0670 | 0.993 | 1.000 |
