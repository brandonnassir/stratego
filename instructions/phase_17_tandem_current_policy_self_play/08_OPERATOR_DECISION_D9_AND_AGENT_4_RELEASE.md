# Phase 17 — Operator Decision D9-B
## Agent 4 integration release and setup-concentration adjudication

> **Historical as of 2026-08-28.** D9-B authorized the completed Agent 4 integration
> experiment. Operator decision D10 supersedes its adaptive setup-KL recipe,
> re-horizoned entropy schedule, standalone-diversity interpretation, and launch gates.
> Preserve this file and its results as evidence; do not use it as the active recipe.

_Written 2026-08-27 after review of Agent 3's D7-B/D5 follow-up._

## 1. Decision

The operator accepts **D9-B**: defer the relative setup-entropy criterion to a bounded
real tandem soak in Agent 4.

Agent 3's standalone gate remains historically **FAIL** on S6 and its
`ready_for_tandem_integration: false` field must not be rewritten. The failure is real:
mean prefix entropy fell from `1.5428944789` to `0.8091`, below the 60% threshold
`0.9257366873` for consecutive checks. The standalone fixture also produced 83% draws
under uniform-random legal move play, so it did not establish how the setup learner
behaves under the trained Phase 9/current-policy outcome signal.

This file provides a narrow operator override to begin **Agent 4 integration and its
bounded tandem soak**. It is not production launch authorization, a passed setup gate,
or permission to weaken hard correctness floors.

## 2. Start condition

Before Agent 4 changes source:

1. commit the current Agent 2 and Agent 3 Phase 17 source, tests, reports, and handoffs
   as an immutable integration input;
2. record that commit SHA and verify the two handoff/source digests against it;
3. preserve the three unrelated pre-existing tracked modifications; and
4. stop if the upstream source changes after verification.

Agent 4 may not integrate uncommitted or moving Agent 2/3 inputs.

## 3. Frozen setup recipe consumed by Agent 4

Agent 4 must consume, not reinterpret, this setup recipe:

```text
model/update equation: phase17_setup_update_v2
setup advantage:        (outcome - expected W/D/L value)
                        + 0.9 * alpha * (suffix information / 10)
conditional h head:     retained with L_h weight 1.0; h is not read by advantage
PPO clip:               0.2
value-loss weight:      0.5
gradient clip:          0.5
setup epochs:           5
setup LR:               constant 5e-5
setup EMA decay:        0.999
pool size:              512 per side per raw setup snapshot

alpha(n): max(0.1 * n^-p, 0.1 * 42376^-0.3)
p:        0.3 * ln(42376) / ln(N), with N frozen by Agent 4

setup behavior KL:      reverse D_KL(current || behavior)
KL target:              0.0018
beta initial:           0.1
beta bounds:            [0.001, 1.0]
KL hard limit:          0.08
controller cadence:     once per setup iteration
controller input:       final epoch's mean KL
increase/decrease band: 1.5x / 0.5x target
increase/decrease step: 1.5x / (1/1.5)x beta
```

The setup throughput evidence is approximately `3.99 s/iteration`, projected at
`5.8%` of a 12-hour run. Five epochs therefore remain required.

## 4. Known documentation conflicts and precedence

The current Agent 3 handoff contains one stale statement under
`d5_resolution.controller_update_cadence` saying the controller updates once per
setup epoch. Agent 3's final report, final telemetry, tests, and implementation instead
use one controller update per setup iteration from the final epoch's KL. The latter is
the accepted D5 resolution and the recipe in section 3 governs.

Agent 4 must record the stale field as an upstream documentation irregularity and must
not implement the obsolete per-epoch behavior. Agent 4 does not edit Agent 3's report
or historical gate evidence.

The same handoff's readiness explanation ends by saying an earlier D7 decision would
"flip" readiness. It did not: Agent 3 correctly left readiness false after its D7-B/D5
rerun. This D9-B file supplies a limited integration override without changing that
field.

Agent 3's report labels the shared suite as `299 passed`. An independent rerun on
2026-08-27 produced `298 passed, 1 skipped` in 26.56 seconds; the skip is the MPS-only
sampling test because MPS is unavailable in this execution environment. Treat the
report wording as a count imprecision, not as proof the device-specific test ran.
Agent 4 must rerun that test on the actual training device if the setup sampler will
use MPS; if setup generation is frozen to CPU, record the test as not applicable and
bind that device choice in the config.

## 5. Required tandem concentration reading

Agent 4's bounded integration rehearsal must include real games in which both seats
sample moves from the current raw Phase 9-derived policy and both setups come from the
current raw setup policy. The setup learner must update from those completed outcomes.

At the same cadence and sample size practical for the bounded rehearsal, record:

- mean prefix entropy and percentage of the initial `1.5428944789` baseline;
- flag and bomb effective support and square support;
- reflection-class uniqueness;
- mean and minimum reflection-class distance;
- mean top-token concentration;
- W/D/L distribution, draw rate, setup optimizer steps, gradients, and raw digest
  changes;
- reverse setup KL, beta, and whether the controller is at either bound; and
- a direct comparison with Agent 3's standalone trajectory.

The question is not whether setup EWR improved in a short rehearsal. It is whether a
real move-policy outcome signal changes the rapid concentration pattern enough to
justify the unchanged production guard or a later explicit recalibration.

## 6. Agent 4 stop and reporting rules

During Agent 4 integration only, crossing the relative 60% entropy threshold is a
diagnostic finding, not an automatic integration veto. Stop immediately for:

- any illegal, inventory-invalid, masked-invalid, or misoriented setup;
- any setup-library fallback or setup episode bound to the wrong game/side/snapshot;
- no real setup optimizer/digest update from completed outcomes;
- reverse setup KL above `0.08`;
- flag effective support below `4`;
- severe near-duplicate/reflection-class collapse; or
- any checkpoint, source, config, or behavior-digest mismatch.

Do not tune alpha, KL, epoch count, model size, or the entropy threshold inside Agent
4. Report the measured trajectory and let Agent 6 adjudicate it.

Agent 4's handoff must contain a separate `setup_tandem_concentration_reading` and must
distinguish:

```text
ready_for_external_handshake
ready_for_preflight
production_setup_entropy_rule_unresolved
```

An absolute hard-floor or correctness failure makes both readiness fields false. A
relative-only entropy failure may proceed to Agent 6 with
`production_setup_entropy_rule_unresolved: true`; Agent 6, not Agent 4, owns the GO or
NO-GO decision.

## 7. Boundaries unchanged

Training remains 100% current-policy self-play with sampled legal moves. Raw weights
collect data; EMA weights are evaluation only. Search, belief training, historical
opponents, and handcrafted opponents remain prohibited from training.
