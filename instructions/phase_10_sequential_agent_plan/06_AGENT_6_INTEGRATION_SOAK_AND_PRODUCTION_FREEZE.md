# Agent 6 — Phase 10 Integration Soak and Production Freeze

## Mission

Take Agent 5's single selected configuration **without retraining or reselection**, freeze production train-split distributions, and prove the combined Phase 9 policy + learned setup selector + Phase 7 perturbation path is operationally safe/reproducible at game scale. Outcomes are report-only.

## Prerequisites

Verify Agents 1–5 PASS; exactly one winner; selector/utility/scaler digests; final-test outcome access zero; accepted Phase 9 identity unchanged; and `neutral_v1` untouched. If Agent 5 returned overall Phase 10 FAIL, do not run.

## Production probability freeze

Materialize canonical train-split probability vectors for Red and Blue with base ids in canonical order, probabilities, family aggregation, utility scores, and all relevant digests. Publish SHA-256 of each vector and independently rebuild to exact canonical serialization.

Production vectors use train split only.

## Integration soak

Run at least 8,192 games using accepted Phase 9 move policy + accepted Phase 10 learned setup source. Use a dedicated soak namespace, deterministic ids, balanced colors, and no validation/test bank cases. Demonstrate all 16 families in actual games.

## Per-game integration checks

Verify selector identity, requested split=train, selected base in train, reflection/perturbation provenance rebuild, legal final setup, no stranded sampled setup, no inventory mismatch, and no opponent-private selector input.

## Preserve Phase 9

Before and after soak require exact accepted file SHA/model-state/863,959 parameters/all finite. No C1 optimizer/backward path.

## Parallelism/restart

Exercise multiple workers and at least one genuine process restart. Resume exactly by logical game id; no duplicate/missing ids; selector results for fixed logical ids must be identical across restart/topology.

## Storage/throughput

Measure bytes/game if persisted, games/s, decisions/s, peak RSS/MPS, and external-volume health if used. Physical path remains diagnostic only.

## Actual-game diversity diagnostics

Separately Red/Blue report family frequencies, family entropy/effective families, base frequencies, reflection rate, perturbation rate, swap-count distribution, and unique final setups. Hard diversity acceptance remains Agent 4's exact distribution metrics; soak frequencies cannot override them.

Material disagreement between empirical and exact selector probabilities beyond sampling expectations is an implementation problem.

## Outcome diagnostics

Red/Blue/draw, lengths, terminal reasons, or separately scheduled stress/rule games are report-only. They may not change candidate, coefficients, T, mixture, or final thresholds.

## Freeze phase10_system_v1

Bind:

```text
Phase 9 checkpoint SHA/model-state
Phase 7 library identities
utility/scaler identity
selector config identity
Red/Blue train distribution digests
learned_setup_source_v1
neutral_v1 identity
reflection/perturbation versions
all Phase 10 root seeds
acceptance contract
validation/test bank identities
```

No filesystem path in logical identity.

## Required artifacts

```text
reports/phase_10_data/agent_06_integration_soak.json
reports/phase_10_data/agent_06_production_selector_manifest.json
reports/phase_10_data/agent_06_acceptance.json
```

Append §6 to the implementation report.

## Completion gates

```text
agents1_5_pass
selector_config_digest_verified
production_red_distribution_frozen
production_blue_distribution_frozen
production_distribution_rebuild_exact
phase10_system_v1_frozen
soak_games_ge_8192
all_16_families_seen_in_soak
setup_legality_errors_zero
stranded_sampled_setups_zero
inventory_errors_zero
provenance_mismatches_zero
hidden_opponent_selector_inputs_zero
restart_resume_pass
duplicate_game_ids_zero
missing_game_ids_zero
phase9_checkpoint_unchanged
no_c1_optimizer_steps
no_reselection
no_test_outcome_access
full_suite_green
```

## Handoff to Agent 7

Provide `phase10_system_v1`, selected selector/utility/scaler identities, production distributions, validation selection record, Phase 9 identity, diversity evidence, final bank identities, and proof final-test outcome access is still zero. Agent 7 performs first final-test outcome evaluation and no training.
