# Agent 2 — Belief Evaluator, Baselines, and Validation Predictive Evidence

## Mission

Implement the belief evaluation path and two frozen baselines, then run predictive evaluation on `phase11_validation_bank_v1` only.

Measure the existing belief head. Do not train it.

Do not implement the production complete-world sampler beyond the simple frozen baseline sampler.

Do not open the test bank for inference, scoring, privileged truth, or outcomes.

## Prerequisites

Recompute from live bytes:

- Agent 1 PASS and gates
- all Phase 11 contract digests/bundle
- validation/test bank digests
- Phase 9 checkpoint + belief-head identity
- P10-D identity
- Phase 7 identity

Require zero scored test-bank access.

## Work

### 1. Strict public/privileged separation

Production inference:
- public observation/history only
- outputs logits/probabilities

Privileged evaluator:
- receives true hidden ranks only after predictions
- scores only
- cannot feed target back to inference

Production request type must reject hidden truth structurally.

### 2. Prediction recorder

Implement Agent 1's exact record schema and public-state identity.

Persist enough primitive information to independently recompute all metrics and baselines.

### 3. `remaining_count_belief_v1`

Reconstruct remaining counts and public impossibility masks.

Test moved unknowns, revealed ranks, captures, near-endgame exhaustion, one-legal-rank cases, public Scout deductions.

### 4. `count_uniform_world_sampler_v1`

Implement structural fallback baseline and verify exact inventory/public legality.

### 5. Validation run

Exactly:

- 512 logical cases
- 1,024 games
- all 8 strata
- both colors
- exact setup-source balance

Collect all frozen hidden-piece prediction events.

W/D/L is report-only.

### 6. Metrics

Overall + required slices:

- CE learned/baseline
- R_CE
- top1 learned/baseline/delta
- Brier learned/baseline/delta
- ECE
- true-rank probability
- entropy
- per stratum
- Red/Blue
- early/mid/late
- moved/unmoved
- per rank

Use Agent 1 exact aggregation/bootstrap contract.

Validation values are diagnostic. Do not refit, calibrate, change thresholds/bins/baselines/bank, or drop bad strata.

### 7. Independent recomputation

Separate audit path from primitive rows must recompute targets, masks, baseline counts, CE, top1, Brier, ECE.

### 8. Negative controls

Must fire on:

- reversed rank mapping
- wrong true-rank label
- wrong remaining inventory
- publicly known pieces included in hidden denominator
- hidden-truth injection into inference request
- permuted probability columns

## Preservation

Before/after exact:

- Phase 9 SHA/state/params
- belief-head identity
- P10-D config
- utility/scaler
- Phase 7

C1 optimizer steps 0.

## Completion gates

Minimum:

1. agent1_pass
2. contracts_verified
3. validation_bank_verified
4. test_bank_structural_only
5. public_privileged_boundary_pass
6. prediction_schema_exact
7. rank_order_exact
8. remaining_count_baseline_complete
9. baseline_negative_controls_fire
10. count_uniform_world_baseline_complete
11. validation_games_exact
12. validation_strata_exact
13. validation_color_balance_exact
14. validation_setup_source_balance_exact
15. all_required_prediction_events_recorded
16. metrics_finite
17. independent_metric_recompute_pass
18. evaluator_negative_controls_fire
19. no_test_prediction_access
20. no_test_truth_access
21. no_belief_updates
22. phase9_checkpoint_unchanged
23. belief_head_unchanged
24. full_suite_green

## Deliverables

- validation prediction storage/manifest
- `reports/phase_11_data/agent_02_predictive_metrics.json`
- `reports/phase_11_data/agent_02_stratum_metrics.csv`
- `reports/phase_11_data/agent_02_baseline_audit.json`
- `reports/phase_11_data/agent_02_acceptance.json`
- report §2

Storage can be external; identity must be path-independent.

## Handoff to Agent 3

Provide accepted belief API/probability representation, public-state identity, validated count/mask reconstruction, validation public states, and Agent 1 sampler contract.

Agent 3 may use validation public states but not test predictions/truth.

Stop and wait for reviewer acceptance.
