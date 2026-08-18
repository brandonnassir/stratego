# Agent 3 — Setup Utility Models and Independent Audit

## Mission

Fit exactly two lightweight setup-utility models from Agent 2's frozen **train-only** outcome corpus:

```text
Model F — family-only
Model T — family + 35 structural traits
```

Publish deterministic coefficients and an independent fit audit. Do not evaluate selector playing strength, inspect validation/test outcomes, or choose between models.

## Prerequisites

Verify Agents 1–2 PASS; corpus SEALED and digest-matching; exact 16,384 games with 64 per ordered family pair; every record train split; no validation/test outcome access; Phase 9 checkpoint unchanged; live utility contract matches Agent 1.

## Feature reconstruction

Do not trust stored features blindly. For every unique base in the corpus, resolve it through `setup_library_v1`, rebuild `setup_trait_vector_v1`, verify family/split, and compare to recorded trait identity.

Compute the 35-field standardizer from **all 6,400 train bases**, not merely sampled bases. Publish field order, means, stds, zero-std mask, and standardizer digest. Any validation/test contribution to standardization is a hard leak.

## Fit Model F

```text
u_F(s,c) = family_offset[c, family(s)]
```

Use the frozen pairwise outcome objective and fit protocol. Publish intercept, Red/Blue 16-family offsets, objective/BCE/L2, optimizer iterations/evaluations, termination gradient norm, and coefficient digest. Center family offsets under the accepted identifiability rule.

## Fit Model T

```text
u_T(s,c) = family_offset[c,family(s)] + trait_weight[c]^T standardized_traits(s)
```

Use the same corpus and fit protocol. No feature selection, interactions, nonlinear transforms, hidden layers, candidate-specific regularization, or extra tuning.

## Independent audit

Build an audit path that does not call the production fit helper for the quantities it verifies. Independently recompute for every corpus record: W/D/L target, Red/Blue orientation, standardized traits, logits, sigmoid probabilities, BCE, L2, full objective, family centering, and finite outputs. Add gradient or finite-difference spot checks.

## Deterministic refit

Fit each model at least twice in independent processes from the same frozen initialization/fit seed. CPU float64 should normally produce identical coefficients/objectives. If not, do not invent a loose tolerance after seeing differences; use only a tolerance frozen before comparison and justify it.

## Negative controls

Required non-vacuous controls:

1. swap Red/Blue setup orientation;
2. use wrong draw target;
3. use held-out statistics for standardization;
4. permute one trait-column order;
5. alter one family id;
6. alter one coefficient.

Each must be detected by the independent audit/identity checks.

## Production-input safety

The training objective may see both completed-game setups, but the exported production scorer must decompose to **own-side** `u(s,c)` only. Prove the artifact contains no opponent-conditioned table, opponent family/base, opponent policy id, matchup matrix, or outcome-conditioned production feature. The red-first intercept is not used for ranking setups.

## No model selection

Do not compare Model F vs T by validation/test playing strength. Training objective values are diagnostic only. Both models go forward to Agent 4.

## Required artifacts

```text
reports/phase_10_data/agent_03_utility_models.json
reports/phase_10_data/agent_03_utility_audit.json
reports/phase_10_data/agent_03_acceptance.json
```

Store production coefficients/scaler in the normal Phase 10 model/data hierarchy and reference by digest. Append §3 to the implementation report.

## Completion gates

```text
agents1_2_pass
corpus_digest_verified
corpus_train_only
trait_vectors_reconstructed
standardizer_train_only
model_f_fit_complete
model_t_fit_complete
coefficients_finite
objectives_finite
independent_objective_audit_pass
red_blue_orientation_audit_pass
deterministic_refit_pass
negative_controls_fire
production_scorer_own_side_only
no_validation_outcome_access
no_test_outcome_access
no_candidate_selection
phase9_checkpoint_unchanged
full_suite_green
```

## Handoff to Agent 4

Provide exact Model F/T artifacts and digests, standardizer/digest, pure own-side scoring contract, six frozen candidate definitions, proof no held-out outcomes were used, and proof Phase 9 remained unchanged. Agent 4 implements sampling/diversity only.
