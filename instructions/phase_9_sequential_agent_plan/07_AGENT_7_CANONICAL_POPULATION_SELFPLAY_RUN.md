# Phase 9 — Agent 7
# Canonical Population Self-Play Run

## Mission

Execute the one canonical Phase 9 population self-play RL run from the accepted Phase 8 checkpoint using the exact Agent 6 frozen configuration.

Select exactly one Phase 9 checkpoint by the frozen validation score.

Do not open the Phase 9 final-test bank.

## Prerequisites

Require Agents 1–6 `PASS` and formal acceptance.

Verify:

- accepted Phase 8 checkpoint SHA;
- Phase 9 frozen config document digest;
- trainer runtime identity if distinct;
- population/schedule/target/trainer versions;
- validation bank digest;
- starting model checksum.

### Mandatory corpus resolver check

Resolve the accepted Phase 8 corpus through `synthetic_corpus.default_corpus_root()` and require all three accepted digests. Do not hard-code the path into trainer/collector/checkpoint code. Any identity drift is `BLOCKED`.

## Fresh start

Start from:

```text
checkpoints/phase8/warmstart_c1_v1.pt
SHA-256
f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca
```

Do not load any pilot-trained weight or optimizer state.

Record starting model-state checksum before first Phase 9 optimizer update.

## Canonical run

Execute exactly:

```text
60 RL iterations
2,048 scheduled games / iteration
122,880 scheduled games maximum
2 optimizer epochs / rollout
validation every 5 iterations
archive every 5 iterations
12-hour operational ceiling
```

Use exactly `phase9_train_config_v1`.

No dynamic hyperparameter change is allowed except the already-frozen KL beta controller and entropy schedule.

## Per-iteration protocol

For each iteration:

1. create immutable behavior snapshot from current learner;
2. verify/hash snapshot;
3. construct exact Agent 2 logical schedule;
4. collect all scheduled games;
5. reconcile/verify/`SEALED`;
6. construct Agent 4 targets;
7. train exactly two epochs;
8. record all metrics;
9. run validation only at frozen cadence;
10. archive snapshot if cadence requires;
11. mark iteration `COMMITTED`;
12. only then proceed.

No next-iteration collection before current commit.

## Historical archive

Start:

```text
H000 = accepted Phase 8 checkpoint
```

Archive every 5 completed iterations as frozen.

Each archive member:

- immutable;
- SHA-256 recorded;
- model-state checksum recorded;
- iteration recorded;
- population eligibility recorded.

Maintain active window exactly:

```text
H000 + 8 most recent eligible archives
```

No outcome-prioritized sampling.

## Validation

Use only `phase9_validation_bank_v1`.

At each frozen validation cadence compute:

```text
Strategic EWR
Tactical EWR
Phase8-anchor EWR
Random EWR
Basic EWR
score = .45 Strategic + .35 Tactical + .20 Phase8
```

Checkpoint selection:

- strictly highest score wins;
- ties use Agent 1 tie-break;
- Random/Basic regression guards must be monitored;
- final iteration is not automatically selected.

Do not use final-test bank.

## Restart exercise

The canonical run must include at least one genuine process exit/restart from a normal Phase 9 checkpoint after several committed iterations.

Require exact logical continuity of:

```text
iteration
behavior snapshot
next schedule identity
active historical archive
rollout state
optimizer
scheduler
minibatch cursor
KL beta/controller
entropy schedule
global update count
best validation
validation history
archive cadence
```

Apply the accepted backend-aware MPS numerical-resume semantics. Do not demand impossible cross-process bit determinism.

## Hard-stop conditions

Stop immediately on:

```text
illegal neural action
non-finite loss
non-finite gradient
non-finite parameter
behavior identity mismatch
target reconstruction mismatch
observer leak
rollout digest mismatch
checkpoint corruption
population schedule mismatch
learner-control mismatch
mean KL > 0.08
PPO clip fraction > 0.75
test-bank model access
```

Do not skip bad batches or games and continue.

## Freeze accepted checkpoint

At the end:

- identify best validation checkpoint;
- copy/freeze to:

```text
checkpoints/phase9/selfplay_c1_v1.pt
```

- write SHA-256;
- independently reload with evaluation-only path;
- re-evaluate on the same frozen validation protocol;
- require metric reproduction within frozen deterministic tolerance;
- record model-state checksum;
- prove it differs from Phase 8 anchor;
- do not evaluate final-test bank.

Archive and rollout production bytes remain outside Git.

## Artifacts

Create:

```text
reports/phase_9_data/agent_07_canonical_run.json
reports/phase_9_data/agent_07_training_curve.csv
reports/phase_9_data/agent_07_population_archive.json
reports/phase_9_data/agent_07_checkpoint_manifest.json
```

Training curve should include per-iteration:

```text
games
positions
collection throughput
optimizer updates
PPO loss
value loss
belief loss
KL
KL beta
clip fraction
entropy
advantage stats
filter retention
validation metrics when present
archive identities
wall time
```

## Completion gates

Require:

```text
agents1_6_pass
corpus_resolver_verified
corpus_digests_match
fresh_phase8_anchor_start
pilot_checkpoint_loaded_no
exact_frozen_config_used
iterations_completed_60
games_scheduled_122880
rollout_identity_errors_zero
illegal_actions_zero
nonfinite_zero
target_mismatches_zero
observer_leaks_zero
kl_hard_limit_never_exceeded
clip_fraction_hard_limit_never_exceeded
restart_path_exercised
archive_schedule_exact
validation_only_checkpoint_selection
best_checkpoint_reload_reproduces
final_checkpoint_sha_written
final_test_model_access_zero
full_suite_green
```

If the operational ceiling arrives before completion, do not declare PASS; report incomplete/blocked with exact progress.

## Forbidden

Do not:

- tune after seeing validation beyond frozen controller behavior;
- open final-test bank;
- evaluate Phase 9 final hard gates;
- change opponent mixture;
- extend past 60 iterations;
- continue a pilot checkpoint;
- add search;
- add learned setup selection;
- begin Phase 10.

## Handoff to Agent 8

Provide:

- frozen Phase 9 checkpoint path/SHA/model checksum;
- selected iteration;
- full config identities;
- validation selection history;
- archive manifest;
- Phase 8 anchor identity;
- final-test bank digest;
- training-discipline access logs;
- all hard-stop counters.

Agent 8 performs no training.
