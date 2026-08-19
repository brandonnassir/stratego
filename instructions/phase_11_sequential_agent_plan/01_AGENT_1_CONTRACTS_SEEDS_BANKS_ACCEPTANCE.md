# Agent 1 — Contracts, Seeds, Banks, Metrics, and Acceptance Freeze

## Mission

Freeze the entire Phase 11 experiment **before any Phase 11 prediction score, validation result, sampler result, or test outcome exists**.

You own scientific design. Downstream agents should have no design choice left.

Do not run belief evaluation games, compute Phase 11 predictive metrics, implement the production sampler, or inspect test outcomes.

## Prerequisites

1. Verify Phase 10 has a formal closure commit.
2. Require clean tracked tree before freeze.
3. Recompute from live bytes:
   - Phase 9 checkpoint SHA/state/params/C1 config;
   - Phase 10 P10-D config;
   - utility/scaler;
   - Phase 10 production system;
   - Phase 7 library/splits/neutral_v1;
   - observation/model contracts.
4. Derive and record belief-head tensor digest from live checkpoint tensors.
5. Verify every neural parameter finite.

Do not trust report prose as identity evidence.

## Work

### 1. Freeze contracts

Canonical JSON + SHA-256 for at least:

- `phase11_belief_contract_v1`
- `phase11_belief_baseline_v1`
- `phase11_belief_bank_v1`
- `phase11_belief_metrics_v1`
- `phase11_belief_sampler_v1`
- `phase11_information_safety_v1`
- `phase11_acceptance_v1`
- `phase11_system_v1`

Also freeze a bundle digest.

`phase11_system_v1` begins as a template: upstream model/setup identities bound now; final accepted sampler/runtime/readiness slots fill later under explicit rules.

### 2. Freeze seeds and domains

Use the eight common-contract root seeds.

Freeze every derived domain now, including:

- validation/test case ids
- match seeds
- observer/opponent setup seeds
- belief/world sample ids
- sample piece-order streams
- categorical streams
- information-safety permutation streams
- topology/replay ids
- validation/final bootstrap streams
- Agent 6 soak identities

Run collision audit over every currently enumerable id space. For future million-scale spaces, freeze derivation now and carry a downstream exhaustive collision-check obligation.

### 3. Freeze banks

Build/hash exact validation and test banks from the common contract.

Every case must rebuild from case id alone and be independent of worker count/order/path.

Test bank can be structurally built/hashed but no inference, scoring, privileged truth, or outcome access.

### 4. Freeze observer/opponent semantics

Observer:
- accepted Phase 9 policy + belief head
- own setup source constant across both banks; use P10-D unless impossible for a verified upstream reason

Opponent:
- exactly one of the eight strata
- opponent setup-source stratum exactly P10-D or neutral_v1, balanced

Color swaps within each logical case.

### 5. Freeze target/event semantics

Specify exactly:

- which pieces count as hidden targets
- when predictions are recorded
- exclusion of publicly known ranks
- rank ordering/indexing
- public legal-rank mask semantics
- handling of zero true-rank probability
- numerical epsilon policy if any
- progress buckets
- moved/unmoved definitions

### 6. Freeze baselines

Fully specify `remaining_count_belief_v1` and `count_uniform_world_sampler_v1`.

### 7. Freeze sampler mathematics

Use common-contract weighting unless reviewer changes it now:

`weight = learned_probability * remaining_count`

Freeze:

- piece-order derivation
- categorical derivation
- zero-mass fallback
- count decrement
- validation stack
- provenance
- sampler identity inputs

### 8. Freeze metrics/statistics

Exact formulas for:

- CE
- CE ratio
- top-1
- Brier
- ECE
- true-rank probability
- entropy
- case aggregation
- bootstrap

Recommended ECE: 15 equal-width confidence bins on [0,1], confidence=max probability, accuracy=top1 correctness, weighted absolute gap.

### 9. Freeze Gates A-H

Use common-contract thresholds exactly. Add boundary tests for strict/non-strict operators.

### 10. Freeze classification

Exactly PASS-SEARCH-READY / FAIL / BLOCKED.

## Access ledger

Create append-only Phase 11 bank ledger with:

- agent/stage
- bank version
- structural-only flag
- neural inference count
- scored prediction count
- privileged truth count
- outcome count

Before Agent 7, every test-bank entry must be structural-only and all counters zero.

## Required tests

At minimum:

- contract digest recomputation
- root/domain separation
- bank arithmetic/balance
- isolated case rebuild
- color pairing
- exact strata/setup-source balance
- no logical-id/seed overlap between banks
- no hidden truth in production request schema
- baseline count conservation
- rank ordering
- gate boundary behavior
- classification logic
- test ledger starts at zero scored access
- upstream identities unchanged

## Completion gates

Minimum:

1. upstream_phase10_closed
2. phase9_identity_verified
3. belief_head_identity_frozen
4. phase10_selector_identity_verified
5. phase7_identity_verified
6. eight_contracts_frozen
7. contract_bundle_frozen
8. eight_root_seeds_frozen
9. randomness_domains_frozen
10. seed_collision_audit_clean
11. validation_bank_exact
12. test_bank_exact
13. validation_balance_exact
14. test_balance_exact
15. isolated_case_rebuild_pass
16. bank_overlap_zero
17. prediction_target_contract_frozen
18. baselines_frozen
19. sampler_math_frozen
20. metrics_frozen
21. bootstrap_frozen
22. acceptance_gates_frozen
23. classification_frozen
24. test_outcome_access_zero
25. no_phase11_predictions_scored
26. no_neural_updates
27. phase9_checkpoint_unchanged
28. full_suite_green

May add gates, not weaken them.

## Deliverables

- `reports/phase_11_data/agent_01_phase11_contract.json`
- `reports/phase_11_data/agent_01_validation_bank.json`
- `reports/phase_11_data/agent_01_test_bank.json`
- `reports/phase_11_data/agent_01_acceptance.json`
- report §1

## Handoff to Agent 2

Provide all contract/bank identities, belief-head digest, exact prediction schema/rank indexing, baseline formulas, metric formulas, validation-only authorization, and test-bank zero-access rule.

Agent 2 must not implement/tune `belief_sampler_v1`.

Stop and wait for reviewer acceptance.
