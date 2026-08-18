# Agent 4 — Selector and Production Setup Source

## Mission

Implement and freeze:

```text
setup_selector_v1
learned_setup_source_v1
```

using Agent 3's two accepted utility models and the six frozen temperatures. Prove legality, split discipline, observer safety, diversity, and deterministic sampling at large scale. Do **not** perform candidate strength selection.

## Prerequisites

Verify Agents 1–3 PASS; both utility/scaler digests; six candidate definitions; Phase 7 library/neutral/reflection/perturbation identities; final-test outcome access still zero; Phase 9 unchanged.

## Distribution construction

For each candidate x color x split, enumerate every base in that split, reconstruct family/traits, score with the frozen utility, divide by T, compute finite softmax, and combine exactly:

```text
0.35 * neutral_v1 + 0.65 * learned
```

Publish a canonical probability-vector digest for every candidate/color/split. Never use opponent setup information.

## Sampling identity

A selector call is determined by:

```text
selector identity
requested split
own color
selector seed
```

Derive separate streams for mixture branch, neutral base draw, learned base draw, reflection, perturbation decision, and perturbation seed. No mutable global RNG cursor. Worker count/call order must not matter.

## Preserve Phase 7 semantics

After base selection, delegate to the accepted Phase 7 reflection/perturbation implementation unchanged: reflection 0.5, perturbation 0.5, swap count 1..6 uniform, Hamming 2..12, frozen retry rules. Any adapter must prove identical output.

## Exact distribution diversity audit

For all 6 candidates x 2 colors x 3 splits, compute and record:

- normalized family entropy;
- effective family count;
- min/max family probability;
- within-family normalized base entropy;
- max conditional base probability.

Apply every common-contract threshold. A strength result can never rescue a diversity failure.

## Large sampling audit

Run at least:

```text
100,000 draws per candidate x color x split
= at least 3,600,000 total draws
```

Every draw passes through selector -> base -> reflection -> perturbation -> engine validation. Require all 16 families represented and zero illegal setups, inventory errors, stranded sampled setups, split mismatches, provenance mismatches, determinism mismatches, or non-finite probabilities.

Compare empirical frequencies to exact probabilities as diagnostics.

## Topology/restart reproducibility

Rebuild a substantial fixed draw-id set under 1/3/8/13 workers; round-robin, contiguous, reversed ordering; and fresh process. Require identical base id, reflection, perturbation, final setup fingerprint, and provenance for every logical draw. Resume must be exact set subtraction by draw id.

## Permitted-input boundary

Prove selector inputs contain only requested split, own color, own candidate setup descriptors, selector seed/identity. Positive controls attempting to inject opponent family, base id, setup fingerprint, policy id, or game outcome must be rejected. Changing hidden opponent truth must not change a selector result when own inputs/seed are fixed.

## Preserve neutral_v1

No redefinition of `neutral_v1`. Neutral branch must match the accepted Phase 7 sampler for the same logical neutral draw identity. Learned source gets new versioned identities.

## Required artifacts

```text
reports/phase_10_data/agent_04_selector_contract.json
reports/phase_10_data/agent_04_diversity_audit.json
reports/phase_10_data/agent_04_acceptance.json
```

Append §4 to the implementation report. Include all probability-vector digests and raw diversity metrics.

## Completion gates

```text
agents1_3_pass
utility_digests_match
selector_contract_frozen
candidate_count_exactly_six
mixture_35_65_exact
probabilities_finite
probabilities_sum_to_one
distribution_diversity_audit_complete
all_diversity_thresholds_recorded
selector_draws_ge_required
all_16_families_represented
illegal_setups_zero
inventory_violations_zero
stranded_sampled_setups_zero
split_violations_zero
provenance_mismatches_zero
topology_reproducibility_pass
opponent_hidden_inputs_rejected
neutral_v1_unchanged
no_strength_selection_games
no_test_outcome_access
phase9_checkpoint_unchanged
full_suite_green
```

Individual candidates may be marked ineligible if they fail diversity; the agent can still PASS if the audit is correct. If all six fail structural/correctness requirements, return overall Phase 10 `FAIL` and stop.

## Handoff to Agent 5

Provide the six immutable selector configs, distribution digests, diversity eligibility, deterministic selector API, validation-bank identity, neutral baseline API, frozen score/tie-break, and explicit final-test prohibition.
