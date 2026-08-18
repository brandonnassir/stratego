# Agent 1 — Phase 10 Contract, Seeds, Banks, and Acceptance Freeze

## Mission

Freeze every Phase 10 learning/evaluation decision **before any Phase 10 outcome game is played and before either utility model is fit**. You establish the immutable experiment that Agents 2–7 execute.

Do not collect the 16,384-game corpus, fit utilities, evaluate selector strength, or access final-test outcomes.

Read `00_PHASE_10_SEQUENCE_AND_COMMON_CONTRACT.md` first; it is authoritative.

## Prerequisites

Verify from live bytes/artifacts:

- Phase 9 Agents 1–8 formally PASS;
- accepted Phase 9 checkpoint SHA, model-state digest, parameter count, and finiteness;
- complete Phase 9 contract/amendment chain intact;
- Phase 7 library content/metadata/manifest digests;
- exact 6,400/800/800 split and 400/50/50 family balance;
- deterministic `setup_trait_vector_v1` reconstruction for every base;
- accepted `neutral_v1`, reflection, perturbation semantics;
- no pre-existing Phase 10 outcome corpus, utility model, candidate result, or production selector.

Any mismatch is `BLOCKED`.

## Freeze contracts

Serialize and hash:

```text
phase10_setup_contract_v1
phase10_setup_outcome_corpus_v1
phase10_setup_utility_v1
phase10_setup_selector_v1
phase10_selector_schedule_v1
phase10_eval_bank_v1
phase10_acceptance_v1
phase10_system_v1
```

Each must use canonical JSON, stable versioning, stable SHA-256, upstream identities, and path-independent logical identity. Pin every digest in regression tests.

## Freeze seeds and derivations

Use exactly the eight root seeds in the common contract. Define domain-separated derivations for game ids, both setup roots, family-conditioned base draws, reflection, perturbation, selector branch/base draws, validation/test cases, color pairing, fit initialization, and bootstrap streams.

No derivation may depend on worker count, task arrival order, process id, wall clock, or physical storage path.

Exhaustively enumerate the outcome schedule plus all validation/test logical ids and prove zero duplicates/collisions.

## Freeze the 16,384-game outcome schedule

Exact arithmetic:

```text
256 ordered family pairs x 64 games = 16,384
```

For each logical game freeze ordered Red/Blue families, train-split setup draw identities, game id, setup roots, match seed, accepted Phase 9 policy token on both sides, greedy float32 single-request behavior, and outcome-record schema.

Prove exact pair counts, train-only use, no held-out bases, and no path-dependent ids.

## Freeze utility fitting

Implement exactly the Model F/Model T definitions and deterministic fit protocol from the common contract. Pin trait field order, train-only standardizer, draw target 0.5, Red-perspective orientation, red-first intercept, family-centering rule, L2 1e-3, float64 CPU fit, L-BFGS settings, model artifact schema, and independent objective formula.

If the environment cannot provide the specified deterministic strong-Wolfe L-BFGS behavior, stop before outcomes and obtain reviewer authorization for one deterministic equivalent. Do not improvise after seeing corpus outcomes.

## Freeze exactly six candidates

Pin:

```text
P10-A family-only T=0.75
P10-B family-only T=1.25
P10-C family-only T=2.00
P10-D family+traits T=0.75
P10-E family+traits T=1.25
P10-F family+traits T=2.00
```

All use 0.35 neutral / 0.65 learned. Pin score, eligibility, tie-break, and no-seventh-candidate rule.

## Build/freeze both evaluation banks

Build structurally only:

```text
phase10_validation_bank_v1: 128 cases, validation split, 8/family
phase10_test_bank_v1: 512 cases, test split, 32/family
```

Each case must rebuild fixed opponent setup/provenance, selector seed(s), color pairing, match seed, bootstrap unit, and final setup fingerprints.

### Isolation audit

Load accepted Phase 9 evaluation-bank structural identities and deterministically reject any Phase 10 case whose **final setup fingerprint** exactly overlaps a Phase 9 held-out evaluation fingerprint. Also require zero exact fingerprint overlap between Phase 10 validation and test banks.

Base-id reuse across phases is allowed; exact final setup fingerprint reuse is not. If deterministic construction cannot achieve zero overlap without changing Phase 7, return `BLOCKED`.

## Seal test-bank access

Before Agent 7, allow only structural build/audit/digest/fingerprint checks. Forbid neural inference, game outcomes, model metrics, selection, and hyperparameter selection. Record every access.

## Freeze acceptance/statistics

Serialize all eight hard gates, strict/non-strict inequality semantics, paired bootstrap method, 10,000 replicates, 95% interval, root seeds, matchup token derivations, and four final classifications.

Add negative tests at threshold boundaries.

## Required artifacts

```text
reports/phase_10_data/agent_01_setup_selection_contract.json
reports/phase_10_data/agent_01_validation_bank.json
reports/phase_10_data/agent_01_test_bank.json
reports/phase_10_data/agent_01_acceptance.json
```

Append §1 to `reports/phase_10_implementation_report.md`.

## Completion gates

```text
phase9_final_identity_verified
phase9_model_finite
phase7_library_identity_verified
phase7_splits_verified
phase7_trait_vectors_reconstruct
neutral_profile_verified
phase10_seeds_frozen
phase10_contracts_frozen_and_hashed
outcome_schedule_exact_16384
ordered_family_pair_counts_exact
utility_fit_protocol_frozen
candidate_matrix_exactly_six
validation_bank_frozen_and_hashed
test_bank_frozen_and_hashed
phase9_bank_exact_fingerprint_overlap_zero
phase10_val_test_fingerprint_overlap_zero
test_bank_neural_outcome_access_zero
final_acceptance_gates_frozen
no_phase10_outcome_games
no_utility_fit
phase9_checkpoint_unchanged
full_suite_green
```

## Handoff to Agent 2

Provide the schedule enumerator/rebuilder, all contract/schedule digests, setup derivations, outcome-record schema, Phase 9 evaluation-only identity, resolver/storage policy, exact 16,384 ids, train-only rule, crash/resume identity rules, and Phase 9 byte-preservation requirement. Agent 2 makes no learning-design decision.
