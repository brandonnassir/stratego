# Stratego Artificial Intelligence Project Documentation

This folder contains the current planning and specification documents for the reduced-scale Stratego artificial intelligence project inspired by *Superhuman AI for Stratego Using Self-Play Reinforcement Learning and Test-Time Search* (Sokota et al., 2025).

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
   Authoritative 127-channel player-observation specification, including perspective normalization, setup memory, unresolved inventory, and formal behavioral-event semantics.

7. [`07_observation_validation_matrix.md`](07_observation_validation_matrix.md)  
   Channel-by-channel and behavior-by-behavior acceptance tests, hidden-information anti-leak tests, replay reconstruction tests, and optimized-backend differential requirements.

8. [`08_internal_state_spec.md`](08_internal_state_spec.md)  
   Authoritative compact internal-state contract: piece records, knowledge flags, recent history, active threats, behavioral memory, snapshot requirements, and belief-target separation.

9. [`09_public_event_and_replay_schema.md`](09_public_event_and_replay_schema.md)  
   Privileged replay, derived engine-event, browser-safe observer-event, versioning, ordering, and hidden-information filtering contracts.

## Project objective

Build a local Stratego system on an Apple M4 Pro Mac mini that uses a compact Transformer, a shared belief head, a diverse setup generator, population self-play, dynamically damped policy improvement, and decision-time search. The final target is at least an 85 percent effective win rate against casual human Stratego players, with a draw counted as one-half of a win.

## Source hierarchy

- **Primary research source:** Sokota et al. (2025), especially Appendix A for rules and Sections 2 and D for architecture/training.
- **Official publisher reference:** Hasbro's official Stratego instruction page was checked as an external reference for the existence of the classic game instructions.
- **Project-specific decisions:** Explicitly identified in `02_project_ruleset.md`; they are not presented as rules from the paper.

## Status

Planning baseline: **version 0.2**  
Phase 1 rules, observation, internal-state, replay, and validation contracts are now specified. No model or game-engine implementation is included in these documents.
