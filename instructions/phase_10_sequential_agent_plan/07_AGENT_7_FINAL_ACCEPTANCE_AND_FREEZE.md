# Agent 7 — Independent Final Acceptance and Phase 10 Freeze

## Mission

Independently verify all identities/discipline evidence from live bytes, then perform the **first Phase 10 final-test game-outcome evaluation** on `phase10_test_bank_v1`. No training, candidate replacement, or threshold change.

Return exactly one recommendation:

```text
PASS-IMPROVED
PASS-NONINFERIOR
FAIL
BLOCKED
```

Formal closure belongs to the reviewing chat.

## Administrative freeze

Before final-test access, require Agents 1–6 accepted and the relevant source/artifact tree frozen to a recorded commit or byte-identical administrative freeze. Record untracked files separately. No relevant tracked byte may change between prerequisite verification and bank opening.

## Recompute critical identities

From live bytes verify accepted Phase 9 SHA/model-state/863,959 parameters/all finite; Phase 7 content/metadata/manifest/splits; every Phase 10 contract digest; utility/scaler; selector config; production Red/Blue vectors; `phase10_system_v1`; validation/test bank digests; and all eight seeds.

Independently rebuild the test bank and perform every-case structural audit plus Phase 9-bank exact-fingerprint isolation. Any mismatch before outcomes is `BLOCKED`.

## Discipline audit

Prove outcome corpus was train-only; exactly two utility models; exactly six candidates; no candidate-specific refit; Agent 5 selected from validation only; no final-test outcome/model metric before Agent 7; Agent 6 did not reopen selection; Phase 9 never changed; no C1 optimizer step in Phase 10.

Incomplete discipline evidence is `BLOCKED`; do not open test outcomes anyway.

## Open final bank once

Only after all prerequisites pass, authorize `phase10_test_bank_v1` for `final_evaluation` by Agent 7.

Use greedy float32 `single_request`, frozen color pairing, no search, final bootstrap seed 2026081808, 10,000 paired replicates.

Evaluate single frozen learned selector and neutral baseline on identical logical cases.

## Required final matchups

Direct:

```text
learned selector vs neutral selector
same Phase 9 move policy both sides
```

For Strategic, Tactical, Phase 8 anchor, Random, and Basic run both Phase9+learned and Phase9+neutral on the exact same cases.

Report W/D/L, EWR, color/family splits, terminal reasons, lengths. Stress only if pre-frozen as report-only.

## Recompute hard gates independently

### A — direct

Ordinary: EWR >= 0.49 and paired LB > 0.47. Improved: EWR >= 0.52 and paired LB > 0.50.

### B — strong league

```text
Delta_L = 0.45*Delta_Strategic + 0.35*Delta_Tactical + 0.20*Delta_Phase8
```

Require Delta_L >= -0.01 and 95% CI LB > -0.03. Significant improvement requires Delta_L > 0 and CI LB > 0.

### C — individual strong guards

Strategic, Tactical, Phase8 each require paired delta CI LB > -0.03.

### D — easy guards

Random overall >=0.95, Red>=0.90, Blue>=0.90; Basic>=0.80; paired learned-minus-neutral CI LB > -0.03 for both Random and Basic.

### E — diversity

Recompute/verify every frozen selector distribution metric and threshold.

### F — correctness/information safety

Zero illegal setup/action, inventory, stranded sampled setup, split, provenance, hidden-opponent selector input, non-finite, or inference failures.

### G — reproducibility

Re-run deterministic selector reconstruction on a substantial final-case sample across worker orders/processes and require identical final fingerprints.

### H — Phase 9 preservation

Re-hash Phase 9 after all final games and require exact accepted SHA/model-state/parameter count and zero C1 optimizer steps.

## Final classification

- any hard gate fails after correct execution -> `FAIL`;
- identity/sealing/discipline prevents valid evaluation -> `BLOCKED`;
- all gates pass plus Gate A improved and Gate B significantly positive -> `PASS-IMPROVED`;
- all gates pass otherwise -> `PASS-NONINFERIOR`.

Never switch winner after test, retrain, or use report-only metrics to rescue a gate.

## Final replay/safety audit

Prefer exhaustive final setup reconstruction. Rebuild selected setups from selector identity/seed; verify base/reflection/perturbation/fingerprint; replay move legality; verify neural actions legal; verify no opponent-private selector input; verify all outputs finite.

## Required artifacts

```text
reports/phase_10_data/agent_07_final_acceptance.json
reports/phase_10_data/agent_07_strength_results.csv
reports/phase_10_data/agent_07_diversity_results.csv
```

Append §7 to the implementation report. Acceptance JSON must contain primitive results, bootstrap intervals, all eight gate rows, classification logic, critical identities, access ledger, Phase 9 before/after proof, completion gates, and recommendation.

## Completion gates

```text
agents1_6_pass
administrative_freeze_verified
phase9_identity_verified
phase7_identity_verified
phase10_contracts_verified
utility_and_selector_digests_verified
phase10_system_identity_verified
validation_bank_rebuild_verified
test_bank_rebuild_verified
test_bank_structural_audit_pass
pre_agent7_test_outcome_access_zero
outcome_corpus_train_only_verified
candidate_count_6_verified
selection_validation_only_verified
phase9_checkpoint_unchanged_before_eval
gate_a_recomputed
gate_b_recomputed
gate_c_recomputed
gate_d_recomputed
gate_e_recomputed
gate_f_recomputed
gate_g_recomputed
gate_h_recomputed
final_setup_replay_audit_pass
illegal_actions_zero
nonfinite_zero
opponent_hidden_selector_inputs_zero
phase9_checkpoint_unchanged_after_eval
classification_recomputes_from_gate_rows
full_suite_green
```

## Closure

On PASS freeze permanently `neutral_v1`, accepted `learned_setup_source_v1`, selector config/utility/scaler, and accepted Phase 9 model. Phase 11 validates belief and does not retune setups.

On FAIL, production setup source remains `neutral_v1`; preserve evidence but do not force learned selection into the system.
