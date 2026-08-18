# Agent 5 — Bounded Phase 10 Validation Selection

## Mission

Evaluate exactly the six frozen selector candidates on the **validation bank only**, apply predeclared eligibility, and freeze one selector configuration. Do not refit utilities, change temperatures/mixture weights, add candidates, or access final-test outcomes.

## Prerequisites

Verify Agents 1–4 PASS; exactly six candidate identities; utility/scaler and selector distribution digests; validation-bank digest; final-test outcome access zero; Phase 9 accepted identity; unchanged `neutral_v1`; and Agent 4 structural/diversity eligibility.

## No candidate-specific retraining

Model F and T are frozen. No refitting, feature-scaling changes, regularization changes, family pruning, temperature tuning, or mixture tuning based on validation results.

## Validation protocol

Use only `phase10_validation_bank_v1` (128 logical paired cases) with the accepted Phase 9 move policy, greedy float32 `single_request`, no search.

For every structurally eligible candidate run:

1. learned selector vs neutral selector, same Phase 9 policy both sides;
2. learned vs Strategic;
3. learned vs Tactical;
4. learned vs Phase 8 anchor;
5. learned vs Random;
6. learned vs Basic.

Also run/cache Phase9+`neutral_v1` against Strategic/Tactical/Phase8/Random/Basic on the exact same logical cases. Reuse case ids/colors/opponent setups/bootstrap units across candidates; ensure cache/game identities remain candidate-specific.

## Primitive metrics

Record direct learned-v-neutral EWR and `Delta_D`; learned and neutral W/D/L/EWR vs each external opponent; each `Delta_O`; color/family splits; terminal reasons; lengths; and all safety counters. Stress, if run, is report-only.

## Eligibility

A candidate is eligible only if Agent 4 correctness/reproducibility/diversity all pass, validation Random EWR >= 0.95, validation Basic EWR >= 0.80, and there are zero illegal setup/action/inference/non-finite/observer failures.

A high score cannot rescue an ineligible candidate.

If no candidate is eligible:

```text
PHASE 10 = FAIL
production setup source remains neutral_v1
```

Stop; Agents 6–7 do not run.

## Selection score

Compute exactly:

```text
S10 =
0.40*Delta_D
+ 0.30*Delta_Strategic
+ 0.20*Delta_Tactical
+ 0.10*Delta_Phase8
```

Tie-break:

1. higher S10;
2. higher Delta_Strategic;
3. higher Delta_D;
4. higher normalized family entropy;
5. higher effective base diversity;
6. lexicographically smaller candidate id.

Recompute score/tie-break independently from primitive results. Final-test evidence is forbidden.

## Freeze one config

Write `phase10_selector_config_v1` containing at minimum winner id, utility/scaler digests, temperature, 0.35/0.65 mixture, selector/source versions, Phase 7 and Phase 9 identities, all Phase 10 seeds, validation identity, score components, diversity metrics, and train/validation/test distribution digests.

The selector config and utility coefficients remain separate artifacts. No C1 checkpoint is created or altered.

## Access discipline

Record every validation-bank game-outcome access. Require zero test-bank neural/game-outcome/model-metric/checkpoint-selection accesses. Structural test-bank digest verification is allowed.

## Required artifacts

```text
reports/phase_10_data/agent_05_candidate_results.csv
reports/phase_10_data/agent_05_frozen_selector_config.json
reports/phase_10_data/agent_05_acceptance.json
```

Append §5 to the implementation report. CSV includes all six candidates, including ineligible ones with explicit reasons and no fabricated strength score if not run.

## Completion gates

```text
agents1_4_pass
candidate_count_6
unregistered_candidates_zero
utility_models_not_refit
validation_bank_identity_verified
neutral_baseline_fixed
same_cases_across_candidates
score_recomputes_exactly
tie_break_recomputes_exactly
eligibility_rules_exact
winner_unique_or_tiebreak_resolved
frozen_selector_config_complete
no_seventh_candidate
no_final_test_outcome_access
phase9_checkpoint_unchanged
full_suite_green
```

## Handoff to Agent 6

If a winner exists, hand forward the selector config/digest, utility/scaler artifacts, train-split production distribution digests, neutral baseline identity, Phase 9 identity, validation evidence, and proof the test bank has not been opened. Agent 6 may not reopen selection.
