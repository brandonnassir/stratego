# Stratego Artificial Intelligence Project Plan

> ## ⚠ HISTORICAL PLANNING DOCUMENT — PARTLY SUPERSEDED
>
> This is the **original plan**, preserved for the record. It is **not** a
> description of what happened, and its Phase 13–17 numbering **does not match
> execution**. Do not read any status, ordering or "next step" from this file.
>
> | For | Read |
> |---|---|
> | Current status | [`STATUS.md`](STATUS.md) |
> | What actually happened, Phases 1–18 | [`PHASE_HISTORY.md`](PHASE_HISTORY.md) |
> | Artifact status | [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) |
> | Rules for future work | [`EXPERIMENT_FRAMEWORK.md`](EXPERIMENT_FRAMEWORK.md) |
>
> Two corrections a reader must carry into every section below:
>
> 1. **The 85 percent target in §1 was formally retired on 2026-08-25** for lack
>    of a measurable human pool. It was **retired, not achieved**.
> 0. **"Phase 17" below means the planned casual human evaluation, which was
>    never reached. A *different* Phase 17 — tandem current-policy self-play —
>    was executed under the same number and completed with a negative result.
>    See [`PHASE_HISTORY.md`](PHASE_HISTORY.md) §12 and §13.
>
> 2. **The 168-hour final run was attempted once and interrupted** after
>    approximately 60 hours. It ran as *Phase 14*, not as the *Phase 15* this
>    document plans, and it selected no checkpoint. See §2A.
>
> Sections 1–12 remain accurate as the *record of what was planned*, and the
> frozen contracts they cite (rules, observation, action encoding, backend
> decision, evaluation stack) are still authoritative.

## 1. Goal

Build a local Stratego artificial intelligence system on an Apple M4 Pro Mac mini that can achieve **at least an 85 percent effective win rate against casual human Stratego players**.

Draws count as one-half of a win.

This is a practical reduced-scale system inspired by Ataraxos, not an attempt to reproduce its compute scale exactly.

---

## 2. Fixed project constraints

### Hardware

- Apple M4 Pro Mac mini;
- 14-core central processing unit;
- 20-core graphics processing unit;
- 16-core Neural Engine;
- 48 gigabytes unified memory;
- approximately 150 gigabytes free internal storage;
- 1 terabyte external drive.

### Training

- local hardware only;
- final training run lasts exactly **168 continuous wall-clock hours**;
- final large evaluation occurs after the 168-hour training period;
- the system must checkpoint and resume, but a crash does not extend the 168-hour wall-clock budget.

### Software direction

- Python project;
- PyTorch for neural-network training;
- Apple graphics processor through the supported PyTorch Metal backend;
- readable Python reference game engine first;
- optimized engine backend only if profiling shows it is needed;
- browser-based interface;
- browser training controls are allowed through a training-control service.

## 2A. Project status — superseded, see STATUS.md

This section formerly read "Phase 5 — **Next**". That has been untrue since
2026; the project ran through Phase 16. The authoritative status document is
[`STATUS.md`](STATUS.md) and the authoritative chronology is
[`PHASE_HISTORY.md`](PHASE_HISTORY.md). A summary, as of 2026-08-27:

| Phase | Actual outcome |
|---|---|
| Phases 1–4 | **`ACCEPTED`** — contracts, engine, throughput decision, evaluation harness |
| Phase 5 | **`ACCEPTED`** — model contract and end-to-end integration, `PASS` 22/22 |
| Phase 6 / 6B | **`ACCEPTED`** — C1 selected (863,959 params); engine advanced to `phase2_1_reference_1.2.0` |
| Phase 7 | **`ACCEPTED`** — `setup_library_v1`, 8,000 boards |
| Phase 8 | **`ACCEPTED`** — synthetic warm start, `PASS` 42/42 |
| Phase 9 | **`ACCEPTED`** — population self-play; **`checkpoints/phase9/selfplay_c1_v1.pt` is still the latest accepted direct policy** |
| Phase 10 | **`ACCEPTED`** — learned setup selection, `PASS-NONINFERIOR` |
| Phase 10B | **`INCOMPLETE`** — optional fine-tuning paused at 5 of 30 iterations, no classification |
| Phase 11 | **`FAILED GATE`** — sealed belief test `R_CE 0.9746` vs a `<= 0.97` ceiling; test bank permanently spent |
| Phase 11B | **`CONTAMINATED`** — belief sprint on a corpus carrying a Blue setup-orientation defect |
| Phase 12 | search **engine accepted and still in use**; its **match evidence is `CONTAMINATED`** by the same defect |
| Phase 13 | **`ACCEPTED`** — integrated rehearsal **plus** the configuration freeze this document assigns to Phase 14 |
| Phase 14 | **`INTERRUPTED`** — the attempted 168-hour final run, stopped at **59.97 h / step 202,504**; **no checkpoint selected**, not resumed |
| Phase 15 | **`ENGINEERING`** — unplanned corrective belief/search work (orientation fix, B18/B24, `p24_b24` search) |
| Phase 16 | **`ENGINEERING`** — unplanned robustness/distribution work; 3 of 5 agents ran; recipe shootout verdict **STOP** |
| Phase 17 (human evaluation) | **`PENDING`** — never reached; **zero human or operator games have been played** |
| Phase 17 (tandem self-play, `RUN-2026-B`) | **`COMPLETE`, result negative** — the number was reused for different work; 12.66 h, 535/640 iterations, **no checkpoint promoted**; move-only degraded, joint flat, learned setups worse than the fixed library |
| Phase 18 (setup-integrated warmstart) | **`IN PROGRESS`** — G0 and G1 accepted (G1 closed 2026-09-02); **G2 closed 2026-09-02**: implementation parity passes (`P18-D004` REVISE, accepted) and the bounded raw-actor confirmation on a fresh landscape (Agent 5, `P18-D005` PROCEED for the synthetic trainability portion) was **accepted** and published at `d0fd36d9`; it passed at the threshold's edge (median 10.0073% vs 10%), so the margin is not robust evidence for a precise 10% effect even though the learning effect is strong in all three seeds; **G3 unstarted** — no setup-only Stratego assay, pilot, rehearsal or production run is authorized |

**No new long training run is currently authorized.**

Frozen behavioral contracts:

- reference implementation: `phase2_1_reference_1.1.0`;
- rules: `stratego_project_v1`;
- observation: `observation_v2_1_127ch`;
- action encoding: 10,000 source-destination identifiers.

Phase 3 production result:

```text
backend decision: KEEP_PYTHON
R: 6.50
Agent 6: not required
```

Phase 4 evaluation result:

```text
policy interface: policy_interface_v1
match spec: match_spec_v1
setup bank: evaluation_setup_bank_v1
pairing: color_swap_same_board
calibration: phase4_calibration_v1
hidden-information audit: 100,000 trials / 0 mismatches
final calibration league: 44,544 games / 0 illegal actions / 0 policy errors
baseline strength tiers: 4
```

Calibrated ladder:

1. `strategic_rule_based@1.1.0`;
2. `tactical_rule_based@1.0.0`;
3. `basic_heuristic@1.0.0`;
4. `random_legal@1.0.0`.

Evaluation operating recommendation:

- 256 paired units for screening;
- 1,024 paired units for important/citable comparisons;
- 95% bootstrap confidence interval over `paired_unit_id`;
- Bradley-Terry/Elo-like ratings are secondary only.

Phase 3 measured collection settings remain benchmark starting points, not frozen final-training hyperparameters. Re-measure when the actual model architecture changes.

---

## 3. Ruleset

Use `stratego_project_v1` from `02_project_ruleset.md`.

Important deviations from the competitive rules described by the paper:

- two-square rule excluded;
- continuous-chasing rule excluded.

To guarantee practical termination:

- 100 battleless moves -> draw during training;
- 200 battleless moves -> draw during automated/human evaluation;
- 4,000 total move safety limit during training.

---

## 4. High-level system

```text
Browser interface
      |
      v
Training / evaluation control service
      |
      +----------------------+
      |                      |
      v                      v
Training coordinator     Human/evaluation runner
      |                      |
      v                      v
Compact Transformer + shared belief head
      |
      v
Game-engine interface
      |
      +----------------------+
      |                      |
      v                      v
Python reference engine   Optimized backend (optional after profiling)
```

The interface consumes stable records and service endpoints rather than model internals.

---

## 5. Model direction

### Core network

Use a compact Transformer encoder, initially targeting roughly:

- 100 board-square tokens;
- embedding width near 128;
- 4 Transformer layers;
- 4 attention heads;
- feedforward width near 512;
- approximately 1-2 million parameters, subject to benchmark results.

Final dimensions are selected after throughput tests on the Mac mini.

### Shared outputs

The shared encoder feeds:

1. **move policy head** using source-query / destination-key scoring;
2. **value head** predicting win, draw, and loss probabilities;
3. **belief head** predicting opponent hidden-piece type probabilities.

The first version does not train a separate large belief Transformer.

---

## 6. State representation

Use the approved `observation_v2_1_127ch` representation specified in `06_observation_v2_127ch.md`:

- 12 current own-piece identity planes;
- 12 current known-opponent identity planes;
- 1 hidden-opponent occupancy plane;
- 1 own-piece-known-to-opponent plane;
- 2 movement-status planes;
- 4 live-piece starting-coordinate planes;
- 12 persistent own-setup planes;
- 12 known-opponent setup-identity planes;
- 12 unresolved-opponent-inventory planes;
- 20 own behavioral-history planes;
- 20 opponent behavioral-history planes;
- 16 recent-move planes;
- 3 global/static planes.

Total: **127 feature planes**.

Behavioral history is formally defined through threat, evade, declined attack, protect, and was-protected events. A separate legal-action mask is supplied to the policy head.

---

## 7. Setup generator

### Initial target

Create **8,000 base setups**.

Recommended structure:

- 16 strategic families;
- approximately 500 legal variants per family;
- left-right reflection available for every setup;
- constrained procedural perturbations allow substantially more effective diversity than the stored 8,000 templates.

### Why 8,000

The move model must not see only a small repeated library, because setup structure directly changes hidden-state assumptions and gameplay. At the same time, storing hundreds of thousands of fixed setups is unnecessary when structured procedural variation is cheap.

Eight thousand gives:

- enough templates to avoid trivial memorization;
- broad strategic coverage;
- manageable inspection and validation;
- easy stratified evaluation by setup family;
- negligible storage cost.

### Initial setup families

The exact generation constraints are a later design task, but the library should include roughly balanced coverage of families such as:

1. corner Flag fortress;
2. near-corner Flag fortress;
3. central/back-row Flag fortress;
4. partially bombed Flag;
5. lightly defended/deceptive Flag;
6. false fortress / Bomb decoy;
7. distributed Bomb defense;
8. high Bomb placement;
9. aggressive high-rank front;
10. conservative high-rank rear;
11. Scout-forward information setup;
12. Scout-preservation setup;
13. Miner-forward setup;
14. Miner-preservation setup;
15. balanced conventional setup;
16. deliberately irregular/high-entropy setup.

The generator must preserve exact piece inventory while randomizing within strategic constraints.

### Learned setup selection

The first move learner uses the setup generator directly. Later, a lightweight selector learns probabilities over setup families and/or individual generated setups based on game outcomes.

A full autoregressive setup Transformer is a later research option, not a first-run requirement.

---

## 8. Training strategy

### Stage A: synthetic tactical warm start

Generate games from multiple rule-based agents with different styles.

Use these games to warm-start:

- policy behavior;
- value prediction;
- hidden-piece belief prediction.

All neural-network weights still begin from random initialization. No human game database is required.

Purpose: avoid spending a large portion of the 168-hour run learning elementary game facts from nearly random self-play.

### Stage B: population self-play

Train against a mixture of:

- current policy;
- historical checkpoints;
- rule-based agents;
- deliberately unusual/adversarial behavior agents.

An initial working mixture can be near:

- 50 percent current policy;
- 25 percent historical checkpoints;
- 15 percent rule-based agents;
- 10 percent unusual/adversarial agents.

The exact mixture is a hyperparameter and should be validated during development.

### Stage C: dynamically damped policy improvement

Retain the main Ataraxos-inspired ideas:

- direct policy sampling for self-play;
- no expensive search during routine data generation;
- importance-ratio clipping;
- penalty against the data-generating policy;
- early exploration/regularization that decays over training;
- decreasing learning rate;
- gradient clipping;
- exponential moving average of parameters;
- filtering toward positions with meaningful estimated advantage.

### Stage D: late league fine-tuning

Near the end of the run, emphasize:

- strongest recent checkpoints;
- strongest rule-based opponents;
- diverse setup families;
- unusual play styles;
- lower learning rate and smaller policy updates.

---

## 9. Decision-time search

Search is used for evaluation/human play, not routine self-play data generation.

Initial engineering range to benchmark:

- 8-12 candidate moves;
- 16-64 belief samples;
- rollout depth 8-16 plies;
- approximately 100-500 total rollouts;
- configurable human-play time budget.

Search must combine rollout values with the original policy using regularization rather than replacing the policy with raw rollout rankings.

The final search budget is selected by measured strength gained per second.

---

## 10. Tactical safety layer

Status: **deferred engineering option**.

It will not be part of the first model design or baseline evaluation.

If later introduced, it must be separately switchable so we can compare:

- model + search only;
- model + search + tactical safety layer.

---

## 11. Browser interface

### Monitoring

The browser should display:

- current training phase;
- wall-clock progress;
- model checkpoint;
- games completed;
- transitions processed;
- games/second and transitions/second;
- policy/value/belief losses;
- move entropy;
- learning rate;
- regularization strength;
- gradient norm;
- win/draw/loss rates by opponent;
- terminal-reason distribution;
- hardware/memory statistics;
- recent evaluation results.

### Training control

Feasible controls:

- start a development run;
- pause/stop a development run;
- save checkpoint immediately;
- launch evaluation;
- select a checkpoint for evaluation;
- emergency shutdown.

### Final 168-hour run lock

At final-run launch, training configuration is frozen.

The browser may still:

- monitor;
- request an additional checkpoint;
- initiate emergency termination.

It should not change architecture, learning rate schedule, opponent mixture, or major hyperparameters during the official 168-hour run.

---

## 12. Storage plan

### Internal storage

Use for:

- active project environment;
- current checkpoints;
- hot training logs;
- temporary batches/caches;
- current evaluation outputs;
- short-lived replay/reconstruction working sets.

### External 1 terabyte drive

Use for:

- archived checkpoints;
- selected game/replay archives;
- evaluation leagues;
- long-term metrics;
- packaged experiment snapshots;
- sampled/diagnostic trajectories rather than an unbounded complete self-play archive.

### Trajectory storage principle

Store compact setups + action histories + sparse behavior-policy probabilities + essential model outputs rather than full observation tensors.

Phase 3 measured approximately **5.59 GiB/hour** of encoded trajectory at the accepted collecting configuration. At that rate, retaining every trajectory for all 168 hours would approach the capacity of the external drive before checkpoints, logs, evaluation artifacts, and filesystem overhead.

Therefore use a rolling/managed trajectory buffer:

```text
self-play
   -> rolling training buffer
   -> consume/reconstruct selected positions for learning
   -> retain selected diagnostic/evaluation/sample games
   -> expire bulk trajectories no longer required
```

The exact retention window is a later training-system decision.

### Evaluation-game archival

Phase 4 showed that evaluation replay histories are tiny compared with training trajectories. Agent 3 measured roughly 2-3 KB of action-history replay data per evaluation game, and Agent 4 estimated that retaining the final 44,544-game calibration league's histories would cost only on the order of 100 MB.

The accepted Phase 4 calibration rows retain both setups and a replay digest, but the final league action histories were not archived. This does not invalidate Phase 4 because the games are reproducible from the frozen policy/setup/version records. However, the preferred long-term archival policy is to **retain complete action histories for accepted final evaluation leagues** whenever practical. If convenient, regenerate and archive the Phase 4 calibration histories later as a preservation improvement.

This is consistent with the project preference to preserve most generated games on the external drive when storage permits, while still using managed retention for high-volume training trajectories.

# 13. Collaborative development phases

## Phase 1 — Rules and engine specification

Deliverables:

- official/source rules document;
- project rules contract;
- engine specification;
- engine validation plan.

**Gate:** rules and interfaces approved before engine implementation.
### Additional Phase 1 authoritative contracts

Before implementation begins, Phase 1 also freezes:

- `observation_v2_1_127ch` in `06_observation_v2_127ch.md`;
- its validation matrix in `07_observation_validation_matrix.md`;
- compact internal state in `08_internal_state_spec.md`;
- privileged replay and browser-safe public events in `09_public_event_and_replay_schema.md`.

These contracts ensure the engine can reconstruct model observations and browser views from compact facts without exposing privileged hidden identities.


## Phase 2 — Python reference engine — COMPLETE

The Python reference engine was implemented, validated, and finalized in Phase 2.1.

Frozen implementation:

- `phase2_1_reference_1.1.0`;
- `stratego_project_v1`;
- `observation_v2_1_127ch`;
- fixed 10,000-entry action encoding.

**Gate:** passed.

## Phase 3 — High-throughput training architecture and optimization decision — COMPLETE

Implemented and validated:

1. bulk-synchronous collection;
2. one Metal-owning coordinator plus multiple CPU simulation workers;
3. persistent shared-memory buffers;
4. compact trajectories with periodic snapshots;
5. representative Metal model benchmark;
6. integrated end-to-end benchmark;
7. two-hour production soak.

Accepted measurements:

- no-model simulation pipeline: **96,963 positions/second**;
- sustainable representative-model inference: **14,922 positions/second**;
- `R = 6.50`;
- collecting soak: **8,871 positions/second**;
- zero soak swap;
- zero measured coordinator-memory growth;
- zero soak reconstruction mismatches;
- accepted representative configuration: 10 workers, 1,536 environments, batch 1,536, float16, dense legality, 32-ply snapshots.

Decision:

```text
KEEP_PYTHON
Agent 6 not required
```

**Gate:** passed.

## Phase 4 — Baseline opponents and evaluation harness — COMPLETE

Implemented and accepted:

- observer-safe `policy_interface_v1`;
- deterministic `match_spec_v1`;
- 1,024-pair `evaluation_setup_bank_v1` using the fixed `structured_v1` family;
- `color_swap_same_board` paired evaluation;
- 4-level calibrated baseline ladder;
- 6 unusual/stress policies;
- `match_runner_v1` and `match_result_v1`;
- `evaluation_scheduler_v1`;
- `evaluation_statistics_v1`;
- `evaluation_reporting_v1`;
- `phase4_calibration_v1`;
- parallel evaluation reproducible across worker counts and shuffled scheduling;
- checkpoint-shaped tensor-consuming policy compatibility.

Final core ladder:

```text
Tier 1 — strategic_rule_based@1.1.0
Tier 2 — tactical_rule_based@1.0.0
Tier 3 — basic_heuristic@1.0.0
Tier 4 — random_legal@1.0.0
```

Strategic 1.1.0 corrected the exposure heuristic to price publicly inferred vulnerability rather than material value. The policy version was bumped before final audit and calibration.

Final information-security audit:

- 100,000 valid hidden-state permutations;
- 1,000,000 policy comparisons;
- 0 action, diagnostic, score-vector, `PublicView`, or legal-action mismatches;
- 100,000 positive controls, 0 failures.

Final calibration league:

- 44,544 games;
- 22,272 paired units;
- 45 matchups;
- 0 illegal actions;
- 0 policy errors.

Direct Strategic-vs-Tactical result at 1,024 paired units:

- Strategic EWR = 0.5354;
- 95% paired interval = [0.5168, 0.5540].

Evaluation sampling guidance established by Phase 4:

- 256 paired units = screening;
- 1,024 paired units = important/citable comparison.

The paired unit is the resampling unit. Confidence intervals condition on the realized policy seeds; they do not measure all possible seed realizations. Seed-only replica variation was several percentage points at 256 units and <= 0.011 at 1,024 units.

**Gate:** passed. The evaluation harness is ready to serve as the permanent ruler for future checkpoints.

## Phase 5 — Integration confirmation of frozen state/action representation

_Executed and accepted: `PASS`, 22/22 hard gates. The "NEXT" marker that stood here is removed; see [`PHASE_HISTORY.md`](PHASE_HISTORY.md) §2._

The semantic freeze was completed early in Phase 2.1:

- `observation_v2_1_127ch`;
- behavioral event semantics;
- 10,000 source-destination action space.

Phase 5 therefore does **not** redefine these contracts. It confirms that the model/training stack consumes them exactly and that no integration layer introduces hidden-information leakage, shape drift, or legality mismatch.

**Gate:** model/training integration matches the already frozen reference contracts exactly.

## Phase 6 — Model design and hardware benchmark

Test compact Transformer sizes on the M4 Pro.

Measure:

- inference positions/second;
- training positions/second;
- memory;
- batch-size scaling.

**Gate:** choose final first-run model size based on throughput/quality tradeoff.

Because Phase 3's `R = 6.50` used an untrained representative probe rather than the final architecture, Phase 6 must re-measure end-to-end inference throughput for candidate real models and confirm that the Python simulator still has sufficient headroom.

## Phase 7 — Setup generator

Develop and validate 8,000 base setups across 16 structural families.

**Gate:** all setups legal, family distribution audited, diversity metrics acceptable.

## Phase 8 — Synthetic warm-start pipeline

Generate games from the rule-based population and train policy/value/belief outputs.

**Gate:** model decisively beats random baseline and learns nontrivial value/belief predictions.

## Phase 9 — Population self-play

Add current-policy and historical-checkpoint opponents with dynamically damped updates.

**Gate:** sustained improvement against frozen evaluation opponents and prior checkpoints without entropy collapse.

## Phase 10 — Learned setup selection

Train lightweight selection over setup families/templates.

**Gate:** learned selection improves or at minimum does not significantly degrade league performance while retaining setup diversity.

## Phase 11 — Belief validation

Evaluate shared belief head on held-out positions and opponent styles.

**Gate:** rule-consistent sampled assignments and measurable improvement over a count-constrained uninformed belief baseline.

## Phase 12 — Decision-time search

Benchmark rollout count, depth, and regularization.

**Gate:** search provides statistically credible performance gain over direct policy under a practical human-play time budget.

> ### ⚠ From here on, the numbering below is NOT what was executed
>
> Phases 13–17 as planned in this document do not correspond to the phases that
> ran under those numbers. The mapping is recorded in
> [`PHASE_HISTORY.md`](PHASE_HISTORY.md) §1 and is **not** being retroactively
> rewritten here:
>
> | Number | Planned below | Actually used for |
> |---|---|---|
> | 13 | Integrated rehearsal | Integrated rehearsal **plus** the configuration freeze planned as Phase 14 |
> | 14 | Configuration freeze | **The attempted final 168-hour run** — interrupted at 59.97 h |
> | 15 | Final 168-hour run | Corrective belief/search engineering (unplanned) |
> | 16 | Automated final evaluation | Robustness and distribution engineering (unplanned) |
> | 17 | Casual human evaluation | **Never reached** — `PENDING` |
>
> Read the four sections below as *the plan that was written*, not as a record
> of work. In particular, **the "Phase 15 — Final 168-hour run" section below
> describes a run that was executed under the name Phase 14 and did not
> complete**, and **"Phase 17 — Casual human evaluation" never happened**.

## Phase 13 — Integrated rehearsal

Run the entire stack continuously for several hours.

Validate:

- checkpoint/resume;
- logs;
- browser monitoring/control;
- no memory leak;
- no non-finite losses;
- evaluation automation.

**Gate:** no critical failure in rehearsal.

## Phase 14 — Configuration freeze

_**Number reused.** This freeze was actually performed in Phase 13. The name "Phase 14" now permanently denotes the interrupted 168-hour run. See [`PHASE_HISTORY.md`](PHASE_HISTORY.md) §§8–9._

Record:

- source revision;
- rules version;
- observation version;
- model architecture;
- random seeds;
- training schedule;
- opponent mixture;
- setup-library version;
- software versions;
- checkpoint policy.

**Gate:** final-run manifest signed off before timer begins.

## Phase 15 — Final 168-hour run

_**Number reused, and the plan below was not fulfilled.** This run was executed as **Phase 14** and was **interrupted after 59.97 hours**, producing 10 of 29 candidates and selecting no checkpoint. The number 15 was later used for corrective belief/search engineering. See [`STATUS.md`](STATUS.md) §5._

Exactly 168 continuous hours.

Suggested conceptual allocation, adjustable before configuration freeze:

- early period: warm start / transition to self-play;
- majority: population self-play;
- late period: stronger league and lower-update fine-tuning.

Periodic lightweight evaluations are permitted during the run because they are part of the fixed training system.

## Phase 16 — Automated final evaluation

_**Number reused.** The planned post-run final evaluation never happened, because the run it would have evaluated did not complete. The number 16 was used for robustness and distribution engineering. See [`PHASE_HISTORY.md`](PHASE_HISTORY.md) §11._

Run balanced machine matches against:

- all baselines;
- strongest historical checkpoints;
- direct policy vs searched policy;
- diverse setup families.

## Phase 17 — Casual human evaluation

_**Never reached.** No human or operator game has ever been played. The 85 percent target below was formally retired on 2026-08-25 — **retired, not achieved** — and its replacement (a 20-game operator exam at effective win rate ≥ 0.50) has also never been run. See [`STATUS.md`](STATUS.md) §10._

Primary target:

\[
\text{effective win rate} \ge 0.85.
\]

Record exact:

- wins;
- draws;
- losses;
- colors;
- setups;
- game lengths;
- opponent self-reported experience;
- search configuration;
- model checkpoint.

Human games occur after training is frozen.

---

## 14. Success criteria

> **Planned criteria, not achieved criteria.** As of 2026-08-27: the engine
> criteria were met; "belief head beats uninformed count-constrained baseline"
> **failed its formal gate** in Phase 11 and the learned belief never separated
> from the count control in Phase 15 match play; "search improves direct-policy
> performance" held on the Phase 15 pack and **did not reproduce** on the
> Phase 16 pack; and the **primary project target was formally retired on
> 2026-08-25 without ever being measured**. See [`STATUS.md`](STATUS.md).

### Engine success

- zero rule-validation failures;
- zero observation hidden-information leaks;
- deterministic replay works;
- stable long run;
- backend throughput documented.

### Initial learning success

- at least 95 percent effective win rate over 1,000 games against random legal play;
- clear superiority over early checkpoints;
- stable policy/value losses;
- move entropy does not collapse immediately.

### Complete-system success

- setup generator and selector preserve diversity;
- belief head beats uninformed count-constrained baseline;
- search improves direct-policy performance;
- model beats strongest internal rule-based baseline with a confidence interval excluding 50 percent effective win rate.

### Primary project target

- at least 85 percent effective win rate against casual human players.

The target is an empirical goal, not guaranteed by the architecture or training schedule.

_**Retired 2026-08-25 — retired, not achieved.** No pool of casual human players
was available to measure it against. Its replacement, `phase16_goal_v1` (an
effective win rate ≥ 0.50 over a predeclared 20-game operator exam under rematch
conditions), has also never been run: **zero human or operator games exist**.
See [`STATUS.md`](STATUS.md) §10._

---

## 15. Development priority

When tradeoffs arise, prioritize in this order:

1. correct rules;
2. reproducible evaluation;
3. self-play throughput;
4. stable learning;
5. complete first training run;
6. belief/search improvements;
7. interface polish;
8. optional tactical safety layer;
9. architectural experimentation.

This order is intended to maximize the probability of completing a functioning initial system before pursuing refinements.
