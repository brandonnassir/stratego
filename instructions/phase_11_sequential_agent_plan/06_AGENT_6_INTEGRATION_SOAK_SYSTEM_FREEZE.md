# Agent 6 — Integration Soak and `phase11_system_v1` Production Freeze

## Mission

Exercise the full production belief path under repeated use and crash/restart, then fill/freeze `phase11_system_v1`.

No training, calibration, sampler redesign, or test-bank scoring.

## Prerequisites

Verify Agents 1-5 PASS and reviewer authorization.

Recompute all load-bearing identities.

Require test-bank scored access = 0 and live implementation matches Agent 5 freeze.

## Production integration soak

Use dedicated Agent 1-frozen Phase 11 soak namespace and train-only/nonbank games/states.

Minimum:

- >=8,192 complete belief requests
- every request does one real belief forward + 64 complete worlds
- thousands of unique public states
- both colors
- early/mid/late
- all eight behavior types where feasible without validation/test cases

Soak outcome/result is report-only.

## Crash/restart

At least three legs with different worker counts.

Include one real SIGKILL/process kill after committed work exists.

Resume by exact logical request-id set subtraction.

Final store exactly scheduled ids: no missing/duplicate/unscheduled.

## Per-request audit

For every committed request:

- rederive public-state identity
- rederive belief probabilities
- rederive all 64 sample ids
- rebuild worlds
- verify exact inventory/public constraints
- verify provenance
- verify no hidden input
- verify deterministic output

Replay substantial subset under another topology.

## Freeze `phase11_system_v1`

Fill Agent 1 template with:

- Phase 9 checkpoint
- belief-head tensor identity
- P10-D selector
- utility/scaler
- evaluator
- `remaining_count_belief_v1`
- `belief_sampler_v1`
- sampler implementation digest
- info-safety identity
- runtime benchmark config/result
- bank identities
- acceptance contract

This is the only belief stack Phase 12 may query if Phase 11 passes.

No absolute paths in logical identity.

## Preservation

Rehash before/after Phase 9, belief head, P10-D, utility/scaler, Phase 7, Agent 5 freeze. All exact.

## Completion gates

Minimum:

1. agents1_5_pass
2. test_scored_access_zero
3. soak_requests_ge_8192
4. soak_nonbank_train_only
5. thousands_unique_states
6. both_colors_covered
7. all_game_progress_buckets_covered
8. restart_resume_pass
9. missing_request_ids_zero
10. duplicate_request_ids_zero
11. unscheduled_request_ids_zero
12. inventory_errors_zero
13. public_constraint_errors_zero
14. provenance_mismatches_zero
15. hidden_input_access_zero
16. deterministic_rebuild_pass
17. cross_topology_replay_pass
18. phase11_system_v1_frozen
19. phase9_checkpoint_unchanged
20. belief_head_unchanged
21. phase10_selector_unchanged
22. no_optimizer_steps
23. full_suite_green

## Deliverables

- `reports/phase_11_data/agent_06_soak_manifest.json`
- `reports/phase_11_data/agent_06_soak_audit.json`
- `reports/phase_11_data/agent_06_system_v1.json`
- `reports/phase_11_data/agent_06_acceptance.json`
- report §6

## Handoff to Agent 7

Provide frozen `phase11_system_v1` digest, administrative freeze requirements, proof test-bank scored access remains zero, final-test entry point, all hard gates.

Agent 7 is first permitted to score the test bank.

Stop and wait for reviewer acceptance.
