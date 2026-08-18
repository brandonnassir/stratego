# Phase 10 Sequential Agent Plan — Manifest

This package contains implementation instructions only. It contains **no Stratego implementation code**.

## Files

1. `00_PHASE_10_SEQUENCE_AND_COMMON_CONTRACT.md` — shared mission, frozen design, seeds, utility models, selector semantics, evaluation banks, diversity rules, acceptance gates, sequencing.
2. `01_AGENT_1_CONTRACT_SEEDS_BANKS_ACCEPTANCE.md` — freeze contracts, schedules, fit protocol, six candidates, banks, test sealing, and acceptance thresholds before outcomes.
3. `02_AGENT_2_SETUP_OUTCOME_CORPUS.md` — generate the exact 16,384-game train-only setup-outcome corpus with Phase 9 frozen.
4. `03_AGENT_3_UTILITY_MODELS_AND_AUDIT.md` — fit exactly family-only and family+trait utility models and audit them independently.
5. `04_AGENT_4_SELECTOR_AND_PRODUCTION_SOURCE.md` — implement the 35/65 selector and large-scale diversity/legality/reproducibility audits.
6. `05_AGENT_5_BOUNDED_VALIDATION_SELECTION.md` — evaluate exactly six candidates on validation only and freeze one selector.
7. `06_AGENT_6_INTEGRATION_SOAK_AND_PRODUCTION_FREEZE.md` — production probability freeze, >=8,192-game soak, `phase10_system_v1`.
8. `07_AGENT_7_FINAL_ACCEPTANCE_AND_FREEZE.md` — first final-test outcome access, hard-gate evaluation, and final recommendation.
9. `SHA256SUMS.md` — integrity fingerprints for the Markdown instruction files.

## Required order

```text
Agent 1 -> review -> Agent 2 -> review -> Agent 3 -> review -> Agent 4
        -> review -> Agent 5

Agent 5 no eligible candidate -> Phase 10 FAIL; stop

Agent 5 winner -> review -> Agent 6 -> review -> Agent 7 -> reviewing-chat closure
```

## Central invariants

- Accepted Phase 9 checkpoint remains byte-identical.
- Phase 7 library and splits remain immutable.
- Utility fitting sees train outcomes only.
- Candidate selection sees validation outcomes only.
- Final-test outcomes are first opened by Agent 7.
- Exactly two utility models and exactly six candidates exist.
- `neutral_v1` remains a permanent baseline.
- Diversity is a hard gate.
- Physical storage paths never define logical identity.
