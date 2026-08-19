# Agent 7 — Independent Final Acceptance and Phase 11 Freeze

## Mission

Perform independent administrative verification and then the **first sealed scored evaluation** of `phase11_test_bank_v1`.

Issue exactly one recommendation:

- `PASS-SEARCH-READY`
- `FAIL`
- `BLOCKED`

No training, calibration, repair, sampler change, threshold change, or rescue rerun.

Formal closure belongs to reviewing chat.

## Stage 0 — administrative freeze

Before opening test bank for scored predictions:

1. clean tracked tree at accepted Agent 6 commit;
2. recompute Agents 1-6 PASS/gates;
3. recompute Phase 11 contracts/bundle;
4. recompute eight seeds/derivations;
5. recompute Phase 9 checkpoint/state/params + belief-head digest;
6. recompute P10-D + utility/scaler;
7. recompute Phase 7;
8. recompute evaluator/sampler/safety/runtime identities;
9. verify `phase11_system_v1` against Agent 1 filling rules;
10. rebuild both banks structurally;
11. prove test scored-prediction/truth access was zero across Agents 1-6.

If integrity/sealing cannot be established -> `BLOCKED`; do not open bank.

## Stage 1 — first sealed test

Exactly:

- 2,048 logical paired cases
- 4,096 games
- 8 strata
- exact colors
- exact setup-source balance
- accepted Phase 9 observer belief model
- frozen evaluator/baselines/sampler

Record every frozen hidden-piece prediction event.

Game outcome cannot alter belief metrics.

## Stage 2 — final predictive metrics

Compute:

- learned CE
- baseline CE
- R_CE
- CE delta CI
- top1 learned/baseline/delta CI
- Brier learned/baseline/delta CI
- ECE overall/by stratum
- CE ratio all strata
- mandatory Red/Blue, early/mid/late, moved/unmoved, rank diagnostics

Use final bootstrap root `2026081908`, 10,000 replicates, logical-case resampling.

Independent statistics path must agree to frozen tolerance.

## Stage 3 — sampler/safety confirmation

Independently reconstruct a substantial test-derived sample set and verify exact inventory/public facts, hidden-input controls, fixed-seed reproducibility, unchanged sampler identity, unchanged Agent 4/6 evidence binding.

Additional test-position sampler checks are allowed if predeclared and do not alter sampler mathematics.

## Stage 4 — Gates A-H

### A
- R_CE <= 0.97
- CE learned-minus-baseline upper 95% bound < 0

### B
- Delta_top1 >= +0.03
- lower 95% bound > 0

### C
- overall ECE <= 0.08
- every stratum ECE <= 0.12
- Brier delta upper 95% bound <= +0.01

### D
Every stratum R_CE <= 1.05.

### E
Sampler correctness counters all zero.

### F
Information-safety counters all zero.

### G
All reproducibility checks exact and frozen p95 forward+64 <= 500 ms.

Do not rerun benchmark on a more favorable backend after seeing result.

### H
All preservation identities exact, optimizer steps zero.

## Classification

- all A-H PASS -> PASS-SEARCH-READY
- valid experiment with >=1 failed gate -> FAIL
- integrity/sealing failure -> BLOCKED

No discretionary override.

## Final preservation

After test work require exact:

- Phase 9 SHA/state/863,959 params
- belief-head identity
- C1 optimizer steps 0
- P10-D
- utility/scaler
- Phase 7

## Test discipline

One final scored run.

No tuning/calibration/metric edits/threshold edits/post-result bank changes/rescue reruns/alternate sampler/model swap.

If artifact-writing fails but primitive test evidence is intact, stop and ask reviewer rather than rerun.

## Completion gates

Minimum:

1. agents1_6_pass
2. administrative_freeze_verified
3. phase9_identity_verified
4. belief_head_identity_verified
5. phase10_identity_verified
6. phase7_identity_verified
7. phase11_contracts_verified
8. phase11_system_verified
9. validation_bank_rebuild_verified
10. test_bank_rebuild_verified
11. pre_agent7_test_score_access_zero
12. test_games_exact
13. test_strata_exact
14. test_color_balance_exact
15. test_setup_source_balance_exact
16. all_prediction_events_recorded
17. metric_recompute_pass
18. independent_bootstrap_pass
19. gate_a_recomputed
20. gate_b_recomputed
21. gate_c_recomputed
22. gate_d_recomputed
23. gate_e_recomputed
24. gate_f_recomputed
25. gate_g_recomputed
26. gate_h_recomputed
27. final_sampler_audit_pass
28. illegal_worlds_zero
29. hidden_input_access_zero
30. nonfinite_zero
31. phase9_checkpoint_unchanged_after_eval
32. belief_head_unchanged_after_eval
33. classification_recomputes_from_gate_rows
34. no_rescue_rerun
35. full_suite_green

## Deliverables

- `reports/phase_11_data/agent_07_final_acceptance.json`
- `reports/phase_11_data/agent_07_predictive_results.csv`
- `reports/phase_11_data/agent_07_calibration_results.csv`
- `reports/phase_11_data/agent_07_sampler_results.json`
- report §7

Report exact gate table, final recommendation, test-first-access proof, load-bearing identities, preservation, and whether Phase 12 is authorized.

## Closure

Do not commit Agent 7 result until reviewing chat accepts/rejects recommendation.

If PASS-SEARCH-READY, permanent Phase 11 freeze is accepted Phase 9 model+belief head, accepted P10-D, `remaining_count_belief_v1`, `belief_sampler_v1`, and Phase 11 system identity. Phase 12 may then implement search.

If FAIL, Phase 12 is not authorized.
