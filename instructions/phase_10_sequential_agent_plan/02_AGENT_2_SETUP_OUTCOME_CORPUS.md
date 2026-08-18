# Agent 2 — Controlled Phase 10 Setup-Outcome Corpus

## Mission

Generate the exact **16,384-game `phase10_setup_outcome_corpus_v1`** frozen by Agent 1. Create outcome evidence only: no setup learning, candidate selection, or Phase 9 model training.

## Prerequisites

Verify Agent 1 PASS; all live Phase 10 contract/bank digests; test-bank outcome access still zero; Phase 9 checkpoint SHA/model-state/parameter count; Phase 7 library/splits/traits; exact schedule reconstruction; and safe resolver-based storage. If external storage is used, prove it is really mounted, writable, and not the boot filesystem.

Any mismatch is `BLOCKED`.

## Move-policy discipline

Both sides use the accepted Phase 9 checkpoint with:

```text
greedy argmax
float32
single_request
no search
no temperature sampling
```

No optimizer, gradients, running-stat mutation, checkpoint rewrite, or parameter mutation. Hash Phase 9 before collection and after final sealing.

## Collect exactly the schedule

Play every Agent 1 logical game once:

```text
16 red families x 16 blue families x 64 = 16,384
```

No additions, substitutions, outcome-driven retries, or changed ids. A crash may regenerate only a logically missing uncommitted game under the same identity.

Every setup is Phase 7 **train split**, family-conditioned under frozen neutral semantics, then frozen reflection/perturbation.

## Outcome record

Persist one digest-checked record per game containing at minimum:

```text
game_id
ordered red/blue family
red/blue base id
red/blue trait-vector identity/digest
reflection/perturbation provenance
final setup fingerprints
setup seeds
Phase 9 checkpoint/model-state identity
W/D/L and winner
game length
terminal reason
match seed
record version
contract/schedule digests
payload/metadata/commit digests
```

Keep pre-game setup descriptors clearly separated from post-game outcome fields.

## Crash-safe store

Use append/commit semantics equivalent in rigor to prior phases. State:

```text
COLLECTING -> SEALED
```

A sealed corpus is immutable. Inject crashes before/after payload, metadata, commit, between games, and at shard rollover if sharded. Recovery exposes exactly committed games and discards uncommitted tails. Resume under different worker partitioning must converge to the same logical corpus.

## Determinism/replay

Rebuild every final setup from provenance. Verify legal terminal records. Independently replay at least 2,048 games end-to-end and require identical W/D/L, length, and terminal reason, with broad family coverage. Prefer exhaustive 16,384 replay if operationally cheap.

Run a wrong-checkpoint negative control and require the policy/result verifier to fail.

## Corpus balance audit

Require:

```text
total games 16,384
ordered pairs 256
games/pair 64
train split violations 0
duplicate game ids 0
duplicate commit identities 0
invalid setups 0
stranded sampled setups 0
inventory violations 0
setup provenance mismatches 0
policy identity mismatches 0
non-finite inference rows 0
illegal neural actions 0
```

Red/Blue/draw totals, lengths, and terminal reasons are diagnostics only; do not rank families.

## Storage diagnostics

Measure bytes/game, total size, compression ratio if any, games/s, decisions/s, peak RSS/MPS, and free storage. Paths are diagnostic only. If external storage disappears, stop `BLOCKED`; do not silently redirect.

## Required artifacts

```text
reports/phase_10_data/agent_02_outcome_corpus.json
reports/phase_10_data/agent_02_family_pair_audit.csv
reports/phase_10_data/agent_02_acceptance.json
```

Append §2 to the implementation report.

## Completion gates

```text
agent1_pass
contract_digests_match
phase9_checkpoint_verified_before
phase9_checkpoint_verified_after
phase9_model_state_unchanged
phase7_train_only
games_exact_16384
ordered_pairs_exact_256
games_per_pair_exact_64
duplicate_game_ids_zero
commit_protocol_pass
crash_resume_pass
invalid_setups_zero
stranded_sampled_setups_zero
inventory_violations_zero
setup_provenance_mismatches_zero
illegal_neural_actions_zero
nonfinite_inference_zero
replay_audit_pass
wrong_checkpoint_negative_control_fires
test_bank_neural_outcome_access_zero
no_setup_learning
full_suite_green
```

## Handoff to Agent 3

Provide read-only corpus resolver/identity, canonical record order, exact schema, setup descriptors, train-only standardization source, and proof no validation/test outcome or Phase 9 weight change occurred. Agent 3 fits exactly two models.
