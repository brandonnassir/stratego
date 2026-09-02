# Stratego Artificial Intelligence Project Documentation

This folder contains the **frozen behavioural specifications** for the reduced-scale Stratego artificial intelligence project inspired by *Superhuman AI for Stratego Using Self-Play Reinforcement Learning and Test-Time Search* (Sokota et al., 2025).

> **These specification files describe contracts, not current state.** For what
> is true right now, read the canonical documents below. The status sections
> further down this file were written at Phase 4 and are preserved as a
> historical planning snapshot.

## Canonical current documentation (read these first)

| Document | Function |
|---|---|
| [`../README.md`](../README.md) | Root project overview and orientation. |
| [`STATUS.md`](STATUS.md) | **Canonical current status — the single source of truth.** |
| [`PHASE_HISTORY.md`](PHASE_HISTORY.md) | Actual chronology of Phases 1–18 vs. the original plan's numbering. |
| [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) | Artifact-by-artifact status classification. |
| [`EXPERIMENT_FRAMEWORK.md`](EXPERIMENT_FRAMEWORK.md) | Rules for future experiments and the run-naming decision. |

Three layers, kept separate on purpose:

1. **Specifications** (`01_`–`12_` here) — frozen contracts; authoritative for
   *how the system is defined*.
2. **Current source of truth** (the table above) — authoritative for *what is
   true now*.
3. **Historical evidence** (`../reports/`, `../instructions/`, and the external
   Phase 14 run state) — read-only records; authoritative for *what was
   measured*, and never edited.

> **2026-08-31.** Phase 17 (tandem self-play, `RUN-2026-B`) is **complete with a
> negative result and no checkpoint promoted**; Phase 18 (setup-integrated
> warmstart) is **in progress: G0–G1 accepted, G2 parity passes and a bounded
> raw-actor confirmation (Agent 5) is running; nothing beyond G2 is
> authorized** (updated 2026-09-02). Note that the number
> 17 was reused: the *planned* Phase 17 was casual human evaluation and is still
> `PENDING`. See [`STATUS.md`](STATUS.md) §13–§14.

## Documents

1. [`01_official_rules.md`](01_official_rules.md)  
   Source-faithful summary of the core Stratego rules and the additional competitive/online rules described in the paper.

2. [`02_project_ruleset.md`](02_project_ruleset.md)  
   The exact rules this project will implement, including deliberate deviations from the competitive rules described in the paper.

3. [`03_game_engine_spec.md`](03_game_engine_spec.md)  
   Requirements for the Python reference game engine, its training-facing interface, deterministic replay, data contracts, and future optimized backend.

4. [`04_engine_validation_plan.md`](04_engine_validation_plan.md)  
   Validation suite and acceptance gates that the game engine must pass before model training begins.

5. [`05_project_plan.md`](05_project_plan.md)  
   High-level collaborative development plan from engine specification through the final 168-hour training run and human evaluation.

6. [`06_observation_v2_127ch.md`](06_observation_v2_127ch.md)  
   Authoritative 127-channel player-observation specification, including perspective normalization, setup memory, unresolved inventory, and formal behavioral-event semantics. The current identifier is `observation_v2_1_127ch`; the file name is retained so the cross-references in the other documents stay valid.

7. [`07_observation_validation_matrix.md`](07_observation_validation_matrix.md)  
   Channel-by-channel and behavior-by-behavior acceptance tests, hidden-information anti-leak tests, replay reconstruction tests, and optimized-backend differential requirements.

8. [`08_internal_state_spec.md`](08_internal_state_spec.md)  
   Authoritative compact internal-state contract: piece records, knowledge flags, recent history, active threats, behavioral memory, snapshot requirements, and belief-target separation.

9. [`09_public_event_and_replay_schema.md`](09_public_event_and_replay_schema.md)  
   Privileged replay, derived engine-event, browser-safe observer-event, versioning, ordering, and hidden-information filtering contracts.

10. [`11_batch_simulation_spec.md`](11_batch_simulation_spec.md)  
    Batch simulation and shared-memory contract: slot identity, deterministic seeding, worker partitioning, buffer layout and stale-publication safety, the hidden-information boundary, model ownership, and decision-record ordering.

11. [`12_trajectory_buffer_spec.md`](12_trajectory_buffer_spec.md)  
    Compact training-trajectory and reconstruction contract: game/decision records, probability precision, snapshot design, the serialization codec, retention policy, and reconstruction throughput planning.

### Note on document numbering

There is no `10_`. The former `10_phase_3_architecture.md` was folded into `03_game_engine_spec.md` sections 16, 17 and 20 during the Phase 4 documentation update, and was removed to avoid a second authoritative statement of the Phase 3 architecture and the `KEEP_PYTHON` decision.

`11_batch_simulation_spec.md` and `12_trajectory_buffer_spec.md` were **not** folded in. They remain the only written contracts for several behaviours the Phase 3 code implements directly — deterministic slot seeding, shared-buffer alignment and stale-publication detection, worker thread pinning, the "workers must not import PyTorch" rule, and the trajectory serialization codec. They keep their original numbers so that no existing reference needs rewriting.

## Project objective

Build a local Stratego system on an Apple M4 Pro Mac mini that uses a compact Transformer, a shared belief head, a diverse setup generator, population self-play, dynamically damped policy improvement, and decision-time search.

> **The original target — at least an 85 percent effective win rate against
> casual human Stratego players, draws counted as one-half — was formally
> retired on 2026-08-25** for lack of a measurable human pool. It was **retired,
> not achieved**: the project has never played a human under any protocol. Its
> replacement, `phase16_goal_v1` (effective win rate ≥ 0.50 over a predeclared
> 20-game operator exam under rematch conditions), has also never been run. See
> [`STATUS.md`](STATUS.md) §10.

## Source hierarchy

- **Primary research source:** Sokota et al. (2025), especially Appendix A for rules and Sections 2 and D for architecture/training.
- **Official publisher reference:** Hasbro's official Stratego instruction page was checked as an external reference for the existence of the classic game instructions.
- **Project-specific decisions:** Explicitly identified in `02_project_ruleset.md`; they are not presented as rules from the paper.

## Documentation synchronization note

This package incorporates the accepted Phase 4 implementation results in addition to the frozen Phase 2.1 and Phase 3 decisions.

It now records:

- the observer-safe `policy_interface_v1`;
- deterministic `match_spec_v1` and `color_swap_same_board` pairing;
- the 1,024-pair `evaluation_setup_bank_v1`;
- the permanent parallel match runner/statistics/reporting stack;
- 95% paired bootstrap confidence intervals over `paired_unit_id`;
- the calibrated four-tier baseline ladder;
- six distinct stress opponents;
- the 100,000-trial / 1,000,000-comparison hidden-information audit with zero mismatches;
- the 44,544-game final calibration league with zero illegal actions and zero policy errors;
- 256 paired units as screening scale and 1,024 paired units as the default important/citable scale;
- checkpoint-shaped consumption of `observation_v2_1_127ch` through the full evaluation path;
- the recommendation to retain complete action histories for accepted evaluation leagues when practical.

No Phase 4 result changed `stratego_project_v1`, `observation_v2_1_127ch`, the 10,000-entry action encoding, or the Phase 3 `KEEP_PYTHON` backend decision.

## Status

> **Historical snapshot — do not read as current.** Everything from here to the
> end of this file was written during the Phase 4 documentation update and
> describes the project as it stood then. It is preserved because it records
> the accepted Phase 1–4 contracts exactly as frozen. For current status see
> [`STATUS.md`](STATUS.md); for what happened afterwards see
> [`PHASE_HISTORY.md`](PHASE_HISTORY.md).

Planning baseline: **version 0.6**

### Completed as of the Phase 4 update

- **Phase 1:** rules, observation, internal-state, replay, and validation contracts.
- **Phase 2:** Python reference engine.
- **Phase 2.1:** perspective-symmetry correction and terminal-precedence clarification.
- **Phase 3:** high-throughput production pipeline, compact trajectories, representative Metal inference, end-to-end scaling, and production-backend decision.
- **Phase 4:** baseline/stress policies, reproducible paired evaluation, statistics/reporting, security audit, and calibration.

Frozen behavioral contracts:

- implementation: `phase2_1_reference_1.1.0`;
- rules: `stratego_project_v1`;
- observation: `observation_v2_1_127ch`;
- action encoding: 10,000 source-destination identifiers;
- production simulator: Python reference engine (`KEEP_PYTHON`, Phase 3 `R = 6.50`).

### Phase 4 accepted evaluation stack

```text
policy interface: policy_interface_v1
match specification: match_spec_v1
evaluation setup bank: evaluation_setup_bank_v1
pairing: color_swap_same_board
match runner: match_runner_v1
match result: match_result_v1
scheduler: evaluation_scheduler_v1
statistics: evaluation_statistics_v1
reporting: evaluation_reporting_v1
calibration: phase4_calibration_v1
```

Calibrated core ladder:

```text
Tier 1 — strategic_rule_based@1.1.0
Tier 2 — tactical_rule_based@1.0.0
Tier 3 — basic_heuristic@1.0.0
Tier 4 — random_legal@1.0.0
```

Final Phase 4 acceptance evidence:

- 100,000 valid hidden-information permutation trials;
- 1,000,000 policy comparisons;
- 0 hidden-information mismatches;
- 44,544 final calibration games;
- 22,272 paired units;
- 45 matchups;
- 0 illegal actions;
- 0 policy errors;
- exact reproduction across serial/parallel/shuffled executions.

Evaluation guidance:

- effective win rate is the primary metric;
- use 95% bootstrap intervals over the paired unit;
- 256 paired units are suitable for screening;
- 1,024 paired units are the default for important/citable comparisons;
- Bradley-Terry/Elo-like ratings are secondary convenience rankings.

The fixed evaluation bank is not the later training setup generator.

### What this snapshot called "Next" — superseded

This section previously read **"Next: Phase 5 — integration confirmation of the
frozen state/action representation."** That is no longer true and has not been
true for a long time.

**Phase 5 completed** (`PASS`, 22/22 hard gates) and the project ran through
**Phase 16**. Phases 5–10 are accepted, Phase 11 **failed** its primary belief
gate, Phases 11B and 12 are **contaminated** by a setup-orientation defect, the
168-hour final run (**Phase 14**) was **interrupted at 59.97 hours** and never
selected a checkpoint, and Phases 15–16 are **engineering-only** corrective
work. Human evaluation has **never been run**.

The full chronology, including how the executed phase numbers diverged from the
numbering in [`05_project_plan.md`](05_project_plan.md), is in
[`PHASE_HISTORY.md`](PHASE_HISTORY.md).

The Markdown files in this folder are project specifications/documentation. Implementations, tests, reports, and raw Phase 3/4 measurement artifacts live elsewhere in the repository.
