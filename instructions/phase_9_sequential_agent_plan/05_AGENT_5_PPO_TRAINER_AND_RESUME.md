# Phase 9 — Agent 5
# PPO Trainer, Dynamic Damping, Checkpoint/Resume, and Throughput Validation

## Mission

Build the real Phase 9 MPS optimizer path over sealed on-policy rollouts.

Implement the frozen PPO/value/belief/KL/entropy objective exactly, prove crash-safe logical resume, and demonstrate numerical stability before any bounded pilot selection.

Do not select a pilot winner.

## Prerequisites

Require Agents 1–4 `PASS` and formal acceptance.

Verify all RL target/example contract digests.

Verify Phase 8 accepted checkpoint identity.

### Mandatory corpus resolver check

Resolve via `synthetic_corpus.default_corpus_root()` and require accepted Phase 8 corpus digests. Checkpoint compatibility must use corpus version+digests, never absolute path.

## Trainer

Create:

```text
phase9_trainer_v1
phase9_checkpoint_v1
```

Use exactly:

```text
C1
float32 MPS
AdamW
weight decay 0.01
grad clip 1.0
minibatch 512
2 epochs per sealed rollout
```

Only learning rate and initial KL beta vary later in Agent 6.

## PPO loss

For PPO-eligible learner examples:

\[
r_t=\pi_\theta(a_t|s_t)/\pi_b(a_t|s_t)
\]

\[
L_\text{PPO}=-E[\min(r_tA_t,clip(r_t,0.8,1.2)A_t)]
\]

Require:

- illegal logits masked before normalization;
- behavior denominator validated finite/positive;
- action must be legal;
- no PPO contribution from non-eligible examples;
- no PPO contribution from opponent-only decisions.

## Value and belief losses

```text
value weight    0.5
belief weight   0.25
```

Value = categorical CE against Agent 4 WDL lambda targets.

Belief = accepted hidden-only CE over supervised squares.

Both use all learner decisions, independent of advantage filter.

## Behavior KL

Compute the Agent 1-frozen direction over legal distributions:

\[
D_{KL}(\pi_b\Vert\pi_\theta)
\]

Target:

```text
0.015
```

Adaptive beta after each optimization epoch:

```text
KL > 0.0300  -> beta *= 2
KL < 0.0075  -> beta *= 0.5
else         -> unchanged

clamp beta to [1e-4, 0.2]
```

Hard failure:

```text
mean KL > 0.08
```

## Entropy

Use legal policy entropy only.

Canonical entropy coefficient decays linearly:

```text
0.005 -> 0.001
```

over the canonical Phase 9 run position.

For pilot tests, use the same schedule mapped to the pilot's frozen progress definition from Agent 1.

## Total loss

\[
L=L_\text{PPO}+0.5L_\text{value}+0.25L_\text{belief}+\beta L_\text{KL}-c_HH
\]

Report every component independently.

## Iteration ownership

Trainer may consume only `SEALED` rollout iterations.

Before optimization:

- verify sealed rollout digest;
- verify behavior checkpoint identity;
- verify population contract;
- verify learner-control semantics;
- verify Phase 9 example/target versions.

After two epochs, mark training complete but do not mutate rollout bytes.

## Checkpoint/resume

Checkpoint must include all fields in the common contract.

Atomic writes only.

Reject:

- truncation;
- integrity digest mismatch;
- corpus identity drift;
- rollout digest drift;
- behavior snapshot drift;
- optimizer/config mismatch;
- population-version mismatch;
- cursor mismatch.

### CPU deterministic proof

Construct a small deterministic fixture and prove:

```text
uninterrupted
==
save / destroy process / reload / continue
```

bitwise where the CPU backend allows, including optimizer moments and KL-controller state.

### MPS proof

Use the reviewer-approved backend-aware principle established in Phase 8:

- exact logical state;
- exact next minibatch;
- exact scheduler/KL-controller/counters;
- immediate post-resume comparison against donor continuation under a predeclared tolerance;
- long-horizon divergence must remain within a measured no-checkpoint MPS control envelope.

Do not require impossible independent-run bit determinism.

Freeze exact Phase 9 resume acceptance measurements before using them as a gate.

## Stability soak

Run a non-selection infrastructure soak starting from the Phase 8 anchor.

Minimum:

```text
>= 2,000 optimizer updates
>= several sealed rollout iterations
```

Use a neutral middle pilot configuration chosen solely for infrastructure.

This is not model selection.

Require zero:

```text
non-finite losses
non-finite gradients
non-finite parameters
illegal targets
data mismatches
checkpoint errors
behavior identity mismatches
rollout identity mismatches
```

Record:

```text
KL
clip fraction
entropy
advantage retention
policy/value/belief losses
grad norms
parameter norms
examples/s
updates/s
data wait
MPS memory
CPU memory
```

Hard fail if KL/clip-fraction frozen instability limits are exceeded.

## Throughput

Measure complete iteration wall time split into:

```text
collection
sealing/audit
target construction
data wait
MPS forward/backward
checkpoint
validation infrastructure
```

Tune only execution topology knobs Agent 1 permits and prove identical logical minibatch identities.

Do not change training order for locality.

## Artifacts

Create:

```text
reports/phase_9_data/agent_05_trainer_contract.json
reports/phase_9_data/agent_05_resume_validation.json
reports/phase_9_data/agent_05_training_benchmark.csv
reports/phase_9_data/agent_05_stability_soak.json
```

## Completion gates

Require:

```text
agents1_4_pass
corpus_resolver_verified
corpus_digests_match
ppo_loss_matches_contract
illegal_logit_masking_pass
value_loss_matches_contract
belief_loss_matches_contract
kl_direction_and_beta_controller_pass
entropy_schedule_pass
opponent_only_gradients_zero
cpu_resume_pass
mps_backend_aware_resume_pass
atomic_checkpoint_tests_pass
soak_updates_ge_2000
nonfinite_zero
illegal_targets_zero
identity_mismatches_zero
kl_hard_limit_not_exceeded
clip_fraction_hard_limit_not_exceeded
throughput_measured
no_pilot_selection
no_final_test_access
full_suite_green
```

## Forbidden

Do not:

- run the six-pilot matrix;
- choose a winner;
- alter learning rates/betas beyond frozen test fixture;
- alter PPO clip/lambdas/loss weights;
- use stale rollouts in later iterations;
- open final-test bank;
- continue a soak checkpoint into Agent 6/7.

## Handoff to Agent 6

Provide:

- exact trainer constructor;
- exact pilot candidate constructor;
- checkpoint/resume API;
- sealed rollout consumption API;
- validated execution topology;
- stability/throughput evidence;
- hard veto counters.

Agent 6 may only run the frozen six candidates.
