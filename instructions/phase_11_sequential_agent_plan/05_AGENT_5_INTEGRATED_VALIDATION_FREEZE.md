# Agent 5 — Integrated Validation Qualification and Implementation Freeze

## Mission

Run the complete Phase 11 validation pipeline exactly as it would run on the final test, using validation bank only, then freeze the implementation Agent 6/7 will use.

There is only one belief head and one frozen sampler design. This is not model selection.

No weights/calibration/metric/threshold/test-bank scoring changes.

## Prerequisites

Verify Agents 1-4 PASS and recompute all identities.

Require test-bank scored access = 0.

## Integrated validation

Using all 512 validation cases / 1,024 games:

1. generate/replay public states through frozen paths;
2. record exact targets;
3. score learned + remaining-count baseline;
4. compute all metrics/slices;
5. invoke sampler checks on frozen validation sample schedule;
6. bind Agent 4 safety/repro/runtime evidence;
7. recompute every final hard-gate quantity available on validation.

Validation hard-gate values are diagnostics/readiness evidence, **not** retuning signals.

Do not change weights, sampler, ECE bins, baseline, bank, thresholds, strata.

If a structural implementation defect is found before artifact freeze, stop and return to reviewer. Do not silently patch/rerun.

If predictive quality is simply weak, report it. Reviewer decides whether to proceed to sealed test.

## Independent recomputation

From primitive rows independently recompute:

- CE learned/baseline/ratio
- top1 delta
- Brier delta
- ECE
- per-stratum CE ratios
- all bootstrap intervals

## Leakage audit

Prove:

- targets used only after prediction
- no test predictions/truth
- no game result used as belief feature
- no diagnostic slice alters implementation

## Freeze implementation identity

Tracked artifact naming:

- belief-head identity
- evaluator version
- remaining-count baseline
- sampler version
- info-safety version
- bootstrap/statistics version
- runtime backend/config
- final-test entry point

Recommended `phase11_validation_freeze_v1`.

## Completion gates

Minimum:

1. agents1_4_pass
2. validation_bank_exact
3. validation_games_exact
4. full_pipeline_complete
5. predictive_metrics_complete
6. all_slices_complete
7. bootstrap_complete
8. independent_recompute_pass
9. sampler_evidence_bound
10. safety_evidence_bound
11. reproducibility_evidence_bound
12. runtime_evidence_bound
13. validation_privileged_boundary_clean
14. no_test_prediction_access
15. no_test_truth_access
16. no_threshold_change
17. no_calibration
18. no_belief_update
19. no_sampler_change
20. upstream_assets_unchanged
21. final_implementation_freeze_complete
22. full_suite_green

## Deliverables

- `reports/phase_11_data/agent_05_validation_metrics.json`
- `reports/phase_11_data/agent_05_validation_strata.csv`
- `reports/phase_11_data/agent_05_validation_freeze.json`
- `reports/phase_11_data/agent_05_acceptance.json`
- report §5

## Handoff to Agent 6

Provide the single frozen Phase 11 implementation identity and immutable dependencies.

Agent 6 performs soak/system freeze only.

Stop and wait for reviewer acceptance.
