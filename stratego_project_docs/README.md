# Stratego Artificial Intelligence Project Documentation

This folder contains the current planning, specification, validation, and systems-architecture documents for the reduced-scale Stratego artificial intelligence project inspired by *Superhuman AI for Stratego Using Self-Play Reinforcement Learning and Test-Time Search* (Sokota et al., 2025).

## Documents

1. [`01_official_rules.md`](01_official_rules.md)  
   Source-faithful summary of the core Stratego rules and the additional competitive/online rules described in the paper.

2. [`02_project_ruleset.md`](02_project_ruleset.md)  
   The exact rules this project implements, including deliberate deviations from the competitive rules described in the paper.

3. [`03_game_engine_spec.md`](03_game_engine_spec.md)  
   Requirements and accepted contracts for the frozen Python reference engine, batch/self-play interface, deterministic replay, and backend-replacement policy.

4. [`04_engine_validation_plan.md`](04_engine_validation_plan.md)  
   Validation suite, accepted Phase 2/2.1 gates, accepted Phase 3 production-readiness gate, and future differential requirements.

5. [`05_project_plan.md`](05_project_plan.md)  
   High-level collaborative development plan from rules/engine work through the final 168-hour training run and human evaluation.

6. [`06_observation_v2_127ch.md`](06_observation_v2_127ch.md)  
   Authoritative 127-channel player-observation specification. The current identifier is `observation_v2_1_127ch`.

7. [`07_observation_validation_matrix.md`](07_observation_validation_matrix.md)  
   Channel-by-channel acceptance tests, mirror-equivalence requirements, hidden-information anti-leak tests, and replay/reconstruction validation.

8. [`08_internal_state_spec.md`](08_internal_state_spec.md)  
   Authoritative compact internal-state contract: piece records, knowledge flags, recent history, active threats, behavioral memory, snapshots, and belief-target separation.

9. [`09_public_event_and_replay_schema.md`](09_public_event_and_replay_schema.md)  
   Privileged replay, derived events, browser-safe observer events, training metadata, versioning, ordering, and storage policy.

10. [`10_phase_3_architecture.md`](10_phase_3_architecture.md)  
    Accepted high-throughput self-play architecture and Phase 3 performance/backend decision.

11. [`11_batch_simulation_spec.md`](11_batch_simulation_spec.md)  
    Accepted batch, worker, shared-memory, synchronization, failure, and model-decision transport contracts.

12. [`12_trajectory_buffer_spec.md`](12_trajectory_buffer_spec.md)  
    Accepted compact trajectory, sparse policy-record, snapshot, reconstruction, and retention/storage contracts.

## Project objective

Build a local Stratego system on an Apple M4 Pro Mac mini that uses a compact Transformer, a shared belief head, a diverse setup generator, population self-play, dynamically damped policy improvement, and decision-time search.

Primary empirical target:

\[
\text{effective win rate against casual human players} \ge 0.85.
\]

A draw counts as one-half of a win.

## Source hierarchy

- **Primary research source:** Sokota et al. (2025), especially Appendix A for rules and Sections 2 and D for architecture/training.
- **Official publisher reference:** Hasbro's official Stratego instruction page was checked as an external reference for the classic game instructions.
- **Project-specific decisions:** Explicitly identified as such in these documents and are not presented as claims from the paper.
- **Implementation evidence:** Phase 2/2.1 and Phase 3 implementation reports and their machine-readable metrics are the basis for accepted project-performance and validation statements.

Documentation baseline: **version 0.5**

## Current frozen behavioral contracts

The following are frozen entering Phase 4:

- reference implementation: `phase2_1_reference_1.1.0`;
- rules: `stratego_project_v1`;
- observation: `observation_v2_1_127ch`;
- action encoding: `action_id = 100 * source + destination`, 10,000 entries;
- replay/state semantics: frozen Phase 2.1 reference behavior.

The reference engine remains the behavioral correctness oracle.

## Completed phases

| Phase | Status |
|---|---|
| Phase 1 — rules/specification | **Complete** |
| Phase 2 — Python reference engine | **Complete** |
| Phase 2.1 — symmetry + terminal-precedence correction | **Complete** |
| Phase 3 — high-throughput self-play architecture and backend decision | **Complete** |
| Phase 4 — baseline opponents and evaluation harness | **Next** |

### Phase 2.1 reference-engine acceptance

Recorded evidence includes:

- 1,255 automated tests passed at the Phase 2.1 freeze;
- 120 combat cases, 0 failures;
- 1,804 mirrored position pairs / 3,608 observation comparisons, 0 mismatches;
- 103,625 valid hidden-information permutations, 0 public-information mismatches;
- 10,000 replay games / 5,078,406 plies, 0 state/observation/event/result mismatches;
- 600 snapshot/restore cases, 0 mismatches;
- 1,045,111 invariant-checked transitions, 0 violations.

### Phase 3 production-readiness acceptance

The complete repository suite reached **1,497 passing tests**.

Integrated Phase 3 evidence includes:

- 10,048 end-to-end worker/shared-memory/model environment-step comparisons, 0 mismatches;
- 11,251 stored decisions reconstructed through the compact trajectory path, 0 mismatches;
- 2.00-hour continuous production-style soak;
- 63,871,488 positions and 123,718 completed games during the soak;
- 411,818 reconstruction checks during the soak, 0 mismatches;
- 0 worker errors/restarts;
- 0 unexplained coordinator memory growth;
- 0 bytes of swap at start and end;
- throughput drift of only -0.76 percent from first to last quarter.

## Phase 3 backend decision

**Decision: `KEEP_PYTHON`. Agent 6 / a second optimized engine is not required.**

Measured decision inputs:

\[
R =
\frac{96{,}963\ \text{simulation-pipeline positions/s}}
{14{,}922\ \text{representative-model positions/s}}
=
6.50.
\]

The decision threshold to retain Python was \(R \ge 2.0\).

The integrated profile independently supports the same conclusion: the representative model/Metal path dominated the step while simulation workers spent most of their time waiting.

A future real model must re-measure this ratio before assuming the same headroom, but the current simulator should not be replaced merely for theoretical speed.

## Accepted Phase 3 starting configuration

For future model-backed collection benchmarks with a similarly sized model, start from:

- 10 simulation workers;
- 1,536 simultaneous environments;
- inference batch 1,536;
- `float16` representative inference;
- **dense** live legality;
- snapshot interval 32 plies.

Measured representative-probe rates:

- best 60-second integrated finalist without full trajectory recording: **12,838 positions/s**;
- two-hour trajectory-recording soak: **8,871 positions/s**.

These are system-engineering baselines, not strength or final-model performance guarantees.

## Storage planning note

The Phase 3 soak generated approximately:

- 96,965 encoded bytes per game;
- 187.8 bytes per decision;
- **5.59 GiB/hour** of encoded trajectory at the measured collecting rate.

At that unchanged rate, retaining every trajectory for 168 hours would be approximately **939 GiB**, before checkpoints, logs, evaluations, filesystem overhead, or other artifacts.

Therefore the final training system should use a **rolling trajectory/replay buffer** and selectively archive important samples rather than permanently retaining every generated decision.

## Important sampler regression

Phase 3 exposed a rare bug in the representative Gumbel-max sampler: a boundary uniform draw could create non-finite Gumbel noise, which could combine with an illegal action's `-inf` mask and produce `NaN` before `argmax`.

The frozen engine rejected the resulting illegal move before state mutation.

The representative sampler was corrected, regression-tested, and a coordinator-side legality check was added. This is now a general validation requirement for any future action sampler using masked infinities.

## Next phase

**Phase 4 — Baseline opponents and evaluation harness**

Build and validate:

- random legal agent;
- elementary heuristic agent;
- stronger rule-based agent;
- unusual/adversarial style agents;
- checkpoint league/evaluation runner;
- balanced, reproducible evaluation protocols.

Do not begin self-play training merely because the Phase 3 representative model can run. The Phase 3 network is an untrained systems benchmark probe, not the final playing model.
