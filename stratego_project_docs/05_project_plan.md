# Stratego Artificial Intelligence Project Plan

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
- frozen Python reference engine selected as the current production simulator after Phase 3 profiling; a separate optimized backend only if a later model changes the bottleneck;
- browser-based interface;
- browser training controls are allowed through a training-control service.

## 2A. Current project status

| Phase | Status |
|---|---|
| Phase 1 — rules/specification | **Complete** |
| Phase 2 — Python reference engine | **Complete** |
| Phase 2.1 — symmetry + terminal-precedence correction | **Complete** |
| Phase 3 — high-throughput self-play architecture and backend decision | **Complete** |
| Phase 4 — baseline opponents and evaluation harness | **Next** |
| Later phases | Planned |

Frozen behavioral contracts:

- reference implementation: `phase2_1_reference_1.1.0`;
- rules: `stratego_project_v1`;
- observation: `observation_v2_1_127ch`;
- action encoding: 10,000 source-destination identifiers;
- replay/state semantics: frozen Phase 2.1 reference implementation.

Phase 3 production decision:

- backend: `KEEP_PYTHON`;
- measured \(R=6.50\);
- Agent 6 / optimized backend: not required;
- accepted initial integrated configuration: 10 workers, 1,536 environments, batch 1,536, float16 representative inference, dense legality, 32-ply snapshots;
- two-hour collecting rate: 8,871 positions/s;
- measured trajectory generation: approximately 5.59 GiB/hour.

The representative Phase 3 Transformer is an untrained systems probe, not the final playing model.

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
Python reference engine   Optimized backend (future only if re-profiling justifies it)
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
- current evaluation outputs.

### External 1 terabyte drive

Use for:

- archived checkpoints;
- game/replay archives;
- evaluation leagues;
- long-term metrics;
- packaged experiment snapshots.

### Trajectory storage principle

Store compact setups + action histories + essential model outputs rather than full 127 by 10 by 10 observations for every move. Reconstruct observations when practical.

Phase 3 measured approximately **5.59 GiB/hour** of encoded trajectory at the representative collecting configuration. At the same rate for 168 hours, retaining everything would be about 939 GiB before other run artifacts.

Therefore the production training system should use:

- a rolling trajectory/replay buffer sized to the training algorithm's consumption needs;
- selective archival of evaluation games, diagnostic/unusual games, and representative training samples;
- checkpoint/log storage independent from bulk trajectory retention;
- explicit deletion/expiration policy for consumed bulk trajectories.

Do not plan to retain every generated self-play decision for the entire final run.

---

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

## Phase 3 — High-throughput self-play architecture and optimization decision — COMPLETE

Implemented:

1. bulk-synchronous collection;
2. one Metal-owning coordinator plus multiple CPU simulation workers;
3. persistent preallocated shared-memory transport;
4. compact trajectories with periodic snapshots;
5. representative PyTorch/Metal inference benchmark;
6. integrated worker/model/trajectory pipeline;
7. two-hour production-style soak.

Accepted results:

- standalone simulation pipeline: 96,963 positions/s;
- representative model denominator: 14,922 positions/s;
- measured \(R=6.50\);
- backend decision: `KEEP_PYTHON`;
- Agent 6 / optimized backend: not required;
- integrated finalist without full recording: 12,838 positions/s;
- trajectory-recording soak: 8,871 positions/s;
- zero integrated correctness/reconstruction mismatches;
- zero unexplained memory growth;
- zero swap use.

Accepted initial configuration for similarly sized model-backed collection:

- 10 workers;
- 1,536 environments;
- inference batch 1,536;
- float16 representative inference;
- dense legality;
- snapshot interval 32.

Important conclusions:

- the current system is model/Metal-bound, not simulator-bound;
- compact legality is slower end to end on the current dense-mask transport and is not the production default;
- the final model must re-measure throughput and \(R\);
- the Phase 3 representative model is not a strength/model-design decision.

**Gate:** passed.

## Phase 4 — Baseline opponents and evaluation harness — NEXT

Build:

- random legal agent;
- elementary heuristic agent;
- stronger rule-based agent;
- unusual/adversarial style agents;
- checkpoint league runner;
- balanced color/setup assignment;
- reproducible match manifests and result summaries.

The Phase 3 representative model is not used as a training-strength baseline; it is an untrained systems probe.

**Gate:** reproducible balanced evaluation and self-play sanity checks suitable for later warm-start and self-play phases.

## Phase 5 — Integration confirmation of frozen state/action representation

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

Exactly 168 continuous hours.

Suggested conceptual allocation, adjustable before configuration freeze:

- early period: warm start / transition to self-play;
- majority: population self-play;
- late period: stronger league and lower-update fine-tuning.

Periodic lightweight evaluations are permitted during the run because they are part of the fixed training system.

## Phase 16 — Automated final evaluation

Run balanced machine matches against:

- all baselines;
- strongest historical checkpoints;
- direct policy vs searched policy;
- diverse setup families.

## Phase 17 — Casual human evaluation

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
