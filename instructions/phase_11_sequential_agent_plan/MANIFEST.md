# Phase 11 Sequential Agent Package Manifest

This ZIP contains **instructions only**, no Stratego implementation code.

## Execution order

1. `00_PHASE_11_SEQUENCE_AND_COMMON_CONTRACT.md`
2. `01_AGENT_1_CONTRACTS_SEEDS_BANKS_ACCEPTANCE.md`
3. `02_AGENT_2_BELIEF_EVALUATOR_BASELINES_VALIDATION.md`
4. `03_AGENT_3_CONSTRAINED_BELIEF_SAMPLER.md`
5. `04_AGENT_4_INFO_SAFETY_REPRO_RUNTIME.md`
6. `05_AGENT_5_INTEGRATED_VALIDATION_FREEZE.md`
7. `06_AGENT_6_INTEGRATION_SOAK_SYSTEM_FREEZE.md`
8. `07_AGENT_7_FINAL_ACCEPTANCE_AND_FREEZE.md`

Also included: `SHA256SUMS.md`.

## Sequential rule

A later agent begins only after the reviewing chat formally accepts the previous agent.

## Scientific boundary

Phase 11 validates the existing belief head and builds a count-constrained complete-world sampler. It does **not** train/fine-tune/calibrate the belief head and does not begin Phase 12 search.

Final outcomes:

- PASS-SEARCH-READY
- FAIL
- BLOCKED

Only PASS-SEARCH-READY authorizes Phase 12.
