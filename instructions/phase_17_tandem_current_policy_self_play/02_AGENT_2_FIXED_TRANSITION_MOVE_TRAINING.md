# Phase 17 — Agent 2
## Fixed-transition move self-play from Phase 9

## Mission

Build the move half of Phase 17: exact Phase 9 weights, sampled 100% current-policy
self-play, true fixed-size transition training with boundary bootstrapping, one epoch,
dynamic schedules/KL, and evaluation-only EMA support.

You do not build the setup network, joint checkpoint runner, remote evaluator, or
production launch. Read `00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md` completely and
consume only a verified
`phase17_contract_handoff_v1` with `ready_for_agents_2_3: true`.

## 1. Ownership and reuse

Own:

```text
stratego/training/phase17/move_*.py
stratego/training/phase17/transition_*.py
tests/training/phase17/test_move_*.py
tests/training/phase17/test_transition_*.py
reports/phase17/agent_02_*
```

Agent 1 may have created the shared contract module; edit shared schema files only if
its handoff explicitly assigns them to you. Import accepted Phase 9 targets/objective
and correct Phase 16 schedule, snapshot, sampling, and EMA components where their
behavior matches. Do not edit earlier phase modules.

## 2. Exact Phase 9 loader

Implement a Phase 17 start loader that delegates to the accepted digest-checked
Phase 9 path and refuses any mismatched file/model-state identity. Build the same C1
architecture and confirm identical CPU logits on fixed observations before creating:

- fresh AdamW optimizer;
- fresh move KL controller at the frozen initial beta;
- iteration-1 schedule state;
- move EMA initialized from the raw Phase 9 state.

The old marginal belief head receives zero auxiliary loss. Add a test proving its
weight is zero in the Phase 17 objective and telemetry.

## 3. Current-policy seating and sampling

Provide one raw move-policy snapshot to both Red and Blue. There is no participant
mixture in training. Reject any training configuration naming historical, Phase 9
anchor, handcrafted, rule-based, stress, or search participants.

Reuse or implement legal categorical action sampling:

- normalize only over legal actions;
- sample from the policy, never argmax;
- derive an explicit stable action seed;
- store legal mask, behavior distribution, action, seed, and snapshot digest;
- reproduce the same action from the same stored distribution and seed.

Add a nondegenerate distribution test that empirically produces more than one legal
action over many seeds and matches the expected probabilities within a reasonable
sampling tolerance.

## 4. Fix in-flight rebinding

The existing Phase 16 collector only rebinds its top-level participants; each
in-flight runner retains the object supplied at game creation. Correct this in the
Phase 17 design. A runner's next action must resolve the current move snapshot
dynamically or receive an atomic propagated rebind.

Required forced-weight test:

1. start at least one game and take decisions under snapshot A;
2. create snapshot B with deliberately different legal logits and a different digest;
3. rebind without ending or recreating the game;
4. take both a Red and Blue decision where feasible;
5. prove behavior logits/action trace/digest are from B;
6. prove stored pre-rebind transitions remain bound to A.

Token equality or metadata-only assertions are insufficient.

## 5. Fixed transition windows

Advance a persistent game population until exactly the contract transition budget is
collected. Games that finish are replaced immediately. Games that do not finish stay
active after the update.

Unlike Phase 16 production, emit every transition in the current window for training;
do not retain whole games waiting for terminal outcomes. At each boundary, store the
post-window bootstrap predictions needed for both players.

Implement:

- scalar TD(lambda) advantage with the accepted lambda and boundary value bootstrap;
- W/D/L lambda targets with terminal outcomes when known and boundary W/D/L bootstrap
  when unfinished;
- exact perspective/sign handling across colors;
- no target leakage from hidden enemy piece identities or future actions;
- exactly one training appearance for every emitted transition.

Required invariant tests:

- **the reduction invariant (`G-M4a`, governing):** when the supplied boundary tail
  is the true terminal continuation, the Phase 17 recursion reproduces the accepted
  whole-game recursion entry for entry to float32 tolerance, for both the advantage
  and the W/D/L walk;
- an unfinished suffix changes only through the supplied bootstrap values;
- swapping player perspective transforms outcomes/targets correctly;
- the trained-row count is exactly the configured budget across short and forced-long
  game mixtures;
- resuming boundary carry state does not duplicate or omit transitions.

**Retired by operator decision D2 (2026-08-27):** the earlier requirement that a
finished trajectory split over at least three windows match whole-game targets to
float32 tolerance. It cannot hold alongside partial emission — see
`00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md` section 6 and
`reports/phase17/agent_01_boundary_target_probe.json`. Do not reinstate it, and do
not weaken the tolerance to make it pass.

Record instead, as telemetry rather than as a gate:

- `target_provenance` on every row: `terminal` or `boundary_bootstrapped`;
- boundary-target divergence: the per-row difference between the bootstrapped target
  and the target that row would have received from its eventual terminal outcome;
- bootstrap age: how many windows the row's trace had been carried when emitted.

Note also that `stratego.training.phase9_loss.phase9_batch_loss` averages the value
term over every row and has no per-row loss mask, which is why Phase 16 left partial
emission off. Build the per-row-maskable path in the Phase 17 namespace; do not edit
the accepted objective.

## 6. Move objective and update

Retain the accepted move objective unless Agent 1's frozen paper map says otherwise:

- PPO clip 0.2;
- accepted advantage standardization and top-quartile PPO filtering;
- scalar/WDL value weights and lambdas from the contract;
- behavior-KL controller with separately logged measured KL and beta;
- scheduled entropy coefficient;
- one epoch per transition window;
- accepted minibatch size and gradient clipping;
- raw update followed by evaluation-only EMA update.

The trainer returns detailed update telemetry and refuses nonfinite values, mean KL
above the hard limit, configuration mismatch, or behavior digest mismatch.

## 7. Structural no-search gate

The collector/trainer dependency graph must not import Phase 12/15/16 search players,
belief-world providers, or search configuration. Add a dependency/refusal test and a
runtime participant ledger proving only the current raw policy acted during a smoke
run.

Evaluation modules may later import the move model; the training module never invokes
them.

## 8. Bounded verification

Keep move-specific experimentation under 30 minutes:

- exact Phase 9 logit identity;
- fixed-transition and target invariant tests;
- forced in-flight rebind test;
- both-seat current-policy ledger;
- stochastic legal sampling test;
- one small update confirming raw changes and EMA follows without acting;
- boundary-state save/load round trip;
- relevant Phase 17 and reused component tests.

Do not run an LR arm, long strength comparison, or opponent-mixture experiment.

## 9. Handoff and report

Deliver:

```text
reports/phase17/phase17_move_handoff_v1.json
reports/phase17/agent_02_report.md
```

The handoff binds code/config/test digests, start identity, transition schema, target
equivalence evidence, forced-rebind evidence, runtime, and a minimal API Agent 4 can
consume. Set `ready_for_tandem_integration` true only when every silent-result gate in
this instruction passes.
