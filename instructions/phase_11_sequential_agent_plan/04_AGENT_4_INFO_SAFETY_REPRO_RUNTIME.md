# Agent 4 — Information Safety, Reproducibility, and Runtime

## Mission

Prove belief model + `belief_sampler_v1`:

1. cannot see hidden truth;
2. reproduce exactly across topology/restart;
3. are fast enough for Phase 12 search.

No training/calibration/sampler redesign/P10-D change/test-bank scoring.

## Prerequisites

Verify Agents 1-3 PASS and recompute all load-bearing identities.

## Part A — hidden-truth permutation attack

Run >=50,000 trials.

For each:

1. choose validation public state with unresolved pieces;
2. construct alternative private hidden truth consistent with identical public history/state;
3. keep every public input byte-identical;
4. compute belief outputs on original/permuted private truth;
5. run fixed-seed sampler on both.

Required exact equality:

- logits/probabilities
- public masks
- sampler request
- sampled world
- sampler provenance

Injection attempts for private fields must be rejected.

## Part B — topology/restart reproducibility

Freeze a large deterministic request set spanning all strata.

Run under:

- 1 worker
- 4 workers
- 12 workers
- forward
- reverse
- round-robin/sharded
- fresh process
- kill/resume by exact request-id subtraction

Compare canonical digest of beliefs, masks, sampled world, provenance. All identical.

## Part C — performance benchmark

Use Agent 1 frozen benchmark config.

Representative states across both colors, all strata, early/mid/late, varying unresolved counts.

Measure:

1. one belief forward
2. forward + 16 worlds
3. forward + 32 worlds
4. forward + 64 worlds

Report median/p90/p95/p99/max, forward component, sampler component, RSS/memory, backend/device.

Hard Gate G quantity:

`p95(forward + 64 worlds) <= 500 ms`

Backend/device is frozen before measured benchmark. Do not switch after seeing results.

## Part D — sensitivity controls

Must fail when:

- private truth is deliberately read
- one belief probability is perturbed
- sample seed changes
- mutable global RNG is introduced
- provenance is corrupted

## Preservation

All upstream identities unchanged. No optimizer/refit/change.

## Completion gates

Minimum:

1. agents1_3_pass
2. hidden_truth_trials_ge_50k
3. belief_output_changes_zero
4. fixed_seed_sample_changes_zero
5. forbidden_hidden_access_zero
6. injection_controls_rejected
7. topology_request_set_frozen
8. worker_1_exact
9. worker_4_exact
10. worker_12_exact
11. forward_reverse_exact
12. fresh_process_exact
13. restart_resume_exact
14. mutable_rng_absent
15. benchmark_config_frozen
16. benchmark_states_representative
17. runtime_metrics_finite
18. p95_64_worlds_recorded
19. p95_64_worlds_le_500ms
20. negative_controls_fire
21. no_test_prediction_access
22. belief_head_unchanged
23. sampler_identity_unchanged
24. phase9_checkpoint_unchanged
25. full_suite_green

## Deliverables

- `reports/phase_11_data/agent_04_information_safety.json`
- `reports/phase_11_data/agent_04_reproducibility.json`
- `reports/phase_11_data/agent_04_runtime.csv`
- `reports/phase_11_data/agent_04_acceptance.json`
- report §4

## Handoff to Agent 5

Provide immutable evaluator/sampler identities, safety/topology evidence, measured runtime config/result.

Stop and wait for reviewer acceptance.
