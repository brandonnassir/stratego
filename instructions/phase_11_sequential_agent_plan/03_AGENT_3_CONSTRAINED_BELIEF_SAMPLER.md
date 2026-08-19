# Agent 3 — `belief_sampler_v1` and Complete-World Audit

## Mission

Implement the frozen count-constrained `belief_sampler_v1` and audit at least 250,000 complete worlds using validation public states only.

Do not change belief weights, probabilities, baselines, P10-D, or acceptance thresholds.

Do not access test-bank belief scores/truth.

## Prerequisites

Verify Agents 1-2 PASS and recompute:

- belief-head identity
- Agent 2 evaluator identity
- Agent 1 sampler contract
- Phase 7/9/10 identities
- validation bank
- test scored-access count = 0

## Implement exact frozen algorithm

For each sample:

1. public belief state + learned marginals only;
2. lock public known ranks;
3. compute exact remaining inventory;
4. apply public masks;
5. derive deterministic unresolved-piece order;
6. for each unresolved piece:
   - legal ranks have remaining count > 0 and public legality;
   - `weight = learned_probability * remaining_count`;
   - renormalize;
   - use frozen zero-mass fallback if needed;
   - deterministic categorical draw;
   - decrement count;
7. emit complete hidden assignment;
8. full validation stack;
9. provenance rebuildable from public-state identity + sample identity.

True hidden ranks cannot guide sampling.

## API boundary

Allow only public state/belief data, sampler identity, sample id/seed.

Reject true rank, private piece table, opponent setup truth, hidden start rank, winner/result/reward, future action/search result, storage path.

## Large audit

At least 250,000 complete worlds across thousands of distinct validation states.

Cover all 8 strata, both colors, early/mid/late, moved/unmoved uncertainty.

Zero counters:

- inventory_errors
- public_knowledge_violations
- known_rank_violations
- immobility_violations
- impossible_assignments
- nonfinite_probability_rows
- provenance_mismatches
- hidden_input_accesses

Report-only diagnostics:

- zero-mass fallback count/rate
- distinct worlds per position
- marginal agreement
- sampler entropy/diversity

## Independent audit

Second implementation path independently rebuilds remaining inventory, masks, exact multiset, public facts, and seed derivation.

Audit at least 25,000 worlds independently, or all if practical.

## Baseline sampler verification

Verify `count_uniform_world_sampler_v1` correctness on same public states. No strength comparison.

## Negative controls

Must fire on:

1. remove one remaining rank
2. allow Bomb/Flag on moved piece
3. duplicate Marshal count
4. alter public known rank
5. mutate sample seed
6. inject true hidden rank
7. corrupt provenance

## Basic reproducibility

Same public state + same sample id -> bit-identical world.

No mutable global RNG cursor.

Agent 4 owns full topology/restart gate.

## Preservation

No writes to Phase 9, Phase 10, Phase 7, or Agent 2 evidence. Optimizer steps 0.

## Completion gates

Minimum:

1. agents1_2_pass
2. sampler_contract_verified
3. sampler_request_boundary_exact
4. true_hidden_inputs_rejected
5. exact_inventory_enforced
6. public_masks_enforced
7. known_ranks_locked
8. piece_order_seeded
9. categorical_draw_seeded
10. zero_mass_fallback_exact
11. complete_world_validation_exact
12. sampler_worlds_ge_250k
13. thousands_distinct_states
14. all_8_strata_covered
15. both_colors_covered
16. all_zero_tolerance_counters_zero
17. independent_audit_pass
18. negative_controls_fire
19. deterministic_repeat_pass
20. baseline_world_sampler_valid
21. no_test_prediction_access
22. no_belief_updates
23. upstream_artifacts_unchanged
24. full_suite_green

## Deliverables

- `reports/phase_11_data/agent_03_sampler_contract.json`
- `reports/phase_11_data/agent_03_sampler_audit.json`
- `reports/phase_11_data/agent_03_sampler_diagnostics.csv`
- `reports/phase_11_data/agent_03_acceptance.json`
- report §3

## Handoff to Agent 4

Provide immutable sampler identity, provenance schema, sample-id rules, validation public-state set, audit evidence, zero-mass fallback behavior.

Agent 4 must not change sampler mathematics.

Stop and wait for reviewer acceptance.
