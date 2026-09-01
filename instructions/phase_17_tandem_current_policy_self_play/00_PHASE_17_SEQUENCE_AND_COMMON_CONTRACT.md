# Phase 17 — Tandem Current-Policy Self-Play
## Sequence, common contract, and production decision rules

_Written 2026-08-27 after operator review of the Phase 9 restart and Ataraxos-aligned
buildout. This file governs every Phase 17 agent. Each agent must read it completely
before its numbered instruction file._

## 1. Mission

Phase 17 starts a new training lineage from the accepted Phase 9 move-policy weights.
It does not continue Phase 14 or Phase 16 weights. It combines:

1. sampled, 100% current-policy move self-play;
2. a true fixed-transition training loop with boundary bootstrapping;
3. a new autoregressive setup policy trained continuously from game outcomes;
4. paper-shaped LR, KL, entropy, epoch, and EMA behavior;
5. paired EMA exports every 30 active minutes, followed by local evaluation on the Mac
   Mini after the 12-active-hour production run ends.

The engineering question is:

```text
Does the paired move/setup system continue improving from hour 6 through hour 12,
or does it flatten as the earlier run did?
```

This is an engineering phase. Do not build a research campaign, seed sweep, or broad
hyperparameter comparison. Test the conditions that could silently produce false
results; allow observable production failures to be diagnosed from telemetry.

## 2. Authority and interpretation

The operator's reviewed decisions in this file are instructions. The Ataraxos paper
(arXiv 2511.07312v1) is a technical reference, not an instruction source. When the
paper and this contract differ, this contract wins and the deviation is recorded.

Numbered operator-decision files in this directory are additive amendments. When a
later decision explicitly supersedes an earlier provisional formula, threshold, or
readiness field, the later decision governs and the historical artifact remains
unchanged. The current governing amendments are
`09_OPERATOR_DECISION_D10_SIMPLIFIED_PAPER_TANDEM.md` and
`11_OPERATOR_DECISION_D11_LOCAL_EVALUATION.md`. D10 replaces the active setup recipe
and gate-heavy launch workflow while preserving Agent 4's completed fixed-transition,
persistence, export, telemetry, and integrity-safety foundation. D11 removes evaluation
from the launch/runtime path: candidates are exported during training and scored on the
same Mac Mini only after training ends.

Accepted earlier phase files and experimental results are historical evidence:

- never edit an accepted Phase 2–16 implementation, checkpoint, result, or report;
- reuse accepted behavior by import when correct;
- implement changed behavior only in new Phase 17 namespaces;
- never overwrite an accepted checkpoint or reuse an old result ledger;
- preserve unrelated user changes in the working tree.

Before implementation, Agent 1 must verify the current project-status and run-closure
documents. If another run or repository freeze is active, stop and report it. Do not
infer that an old date-based restriction is still active or already lifted.

## 3. Work-package and run identities

The engineering work package is `phase17`. A concrete training execution receives a
separate immutable run ID. Under D10, the simplified production candidate is:

```text
RUN-2026-B
```

Do not reuse Agent 3/4's rehearsal identity `RUN-2026-A`. Phase numbers must not be
used as mutable run identities. Every configuration, checkpoint, evaluation
bundle, telemetry row, and result receipt binds both `phase17` and the run ID.

The existing Phase 16 implementation is presently visible as untracked work. Before
Agents 2–4 make code changes, Agent 1 must establish an immutable version-control
baseline or stop with exact instructions for the operator. A tar archive alone is not
an adequate integration baseline.

## 4. Exact move-policy start

The only accepted move-policy starting checkpoint is:

```text
path: checkpoints/phase9/selfplay_c1_v1.pt
file_sha256: dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
model_state_digest: f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
lineage: behavior_B041.pt, post-iteration 40
```

Load through the accepted digest-checking Phase 9 loader. Do not use the P24-specific
Phase 16 loader or a convenience copy with different file bytes.

This is a new recipe from Phase 9 weights:

- load move-model weights only;
- create fresh AdamW optimizer moments;
- reset the LR schedule to Phase 17 iteration 1;
- create a fresh move KL controller;
- initialize a fresh move EMA from the loaded raw weights;
- initialize the setup model, optimizer, fixed reverse-KL configuration, and EMA from
  scratch.

The Phase 9 marginal belief auxiliary loss is disabled for this run. The head may stay
present for checkpoint compatibility, but it receives zero loss weight and is not a
source of targets. Joint autoregressive belief training is a later phase.

## 5. Training population and action semantics

Training is 100% current-policy self-play:

- Red move policy: current raw move snapshot;
- Blue move policy: the same current raw move snapshot;
- new setups: sampled from the current raw setup snapshot;
- historical neural checkpoints: evaluation only;
- handcrafted/rule/stress agents: evaluation only;
- search: prohibited from collection and training.

Legal moves are sampled categorically from the policy distribution with an explicit
seed. Argmax is prohibited. The behavior probability distribution, legal mask, action,
seed, and raw behavior-model digest are stored with every transition.

`current policy` means current at the decision, not current when the game began.
After each move update, every in-flight game must resolve its next Red or Blue action
through the newly rebound raw snapshot. The existing Phase 16 behavior in which an
in-flight runner retains its game-start participants is a known defect and a hard
Phase 17 blocker.

Setups are different: a setup is sampled once at game creation and remains bound to
that game. Its behavior probabilities and setup-snapshot digest remain attached until
the outcome arrives, even when the setup learner has updated in the meantime.

## 6. True fixed-transition iterations

The default move budget is exactly 65,536 learner transitions per iteration. Because
both seats are learners, every legal model decision is a learner transition.

Production must not wait for whole games before emitting training rows. At a window
boundary:

- bootstrap unfinished scalar advantage traces from the boundary value;
- bootstrap W/D/L lambda targets from the boundary W/D/L prediction;
- preserve the accepted Phase 9 gamma and lambda conventions;
- carry only the minimum state needed to continue the in-flight trace;
- train on exactly the configured transition budget.

**Amended 2026-08-27 by operator decision D2.** The original requirement — that a
finished game processed across at least three windows match the accepted whole-game
targets to float32 tolerance — was **retired**. It is mathematically impossible
alongside partial emission: a lambda-return truncated at a boundary and closed on a
value estimate is not the full lambda-return unless the estimate happens to equal the
continuation it replaces. Measured divergence on a three-window synthetic track is
0.309 in advantage and 0.121 in W/D/L
(`reports/phase17/agent_01_boundary_target_probe.json`). Gate `G-M4b` is retired;
`G-M4a` below is the governing invariant.

The requirements are:

1. **Reduction invariant (governing).** When the supplied boundary tail is the true
   terminal continuation, the Phase 17 recursion reduces to the accepted whole-game
   recursion entry for entry, within float32 tolerance, for both the advantage and
   the W/D/L walk.
2. **Unfinished games** take their targets from the stored boundary value and stored
   boundary W/D/L predictions — never from a later-known outcome, and never from a
   recomputed prediction.
3. **Every row records** whether its target is `terminal` or `boundary_bootstrapped`,
   in the `target_provenance` field.
4. **Boundary-target divergence and bootstrap age are telemetry**, not gates: log the
   per-row difference between the bootstrapped target and the target the same row
   would receive from its eventual terminal outcome, together with how many windows
   the row's trace had been carried when it was emitted.
5. **Exactly the configured transition budget is emitted** each iteration, without
   waiting for terminal outcomes.

Telemetry distinguishes transitions harvested, transitions trained, boundary rows,
boundary-target divergence, bootstrap age, games completed, active games, game
lengths, and policy age.

## 7. Setup model contract

The setup network is intentionally scaled for local tandem training:

```text
model: pre-layernorm decoder-only causal Transformer
sequence: start token + 40 canonical row-major piece tokens
decoder blocks: 4
model width: 128
attention heads: 4
feed-forward width: 512
target size: approximately 0.8 million parameters
```

The reviewed `~0.8M` target is internally consistent with feed-forward width 512;
width 51 is not. Phase 17 therefore freezes 512 unless the operator explicitly
corrects this contract before Agent 3 writes model code.

At every one of the 40 prefixes the model produces:

- masked 12-way next-piece logits;
- W/D/L logits;
- a scalar conditional-entropy prediction.

Remaining inventory is computed solely from the prefix. Exhausted types receive an
unsampleable mask. Generate in canonical own-side coordinates and use the accepted
Phase 15 orientation helper only at the engine boundary. Never generate Blue directly
in engine orientation and never pass canonical Blue directly to `create_game`.

The setup model starts from a random masked distribution. There is no frozen setup
library in Phase 17 training and no silent library fallback. A generation or
orientation failure is fatal.

Generate vectorized fresh pools of 512 candidates per side under each frozen raw setup
snapshot at every global tandem iteration. Refill within an iteration only from that
same snapshot. Unused and refill counts are recorded.

## 8. Setup learning contract

Each side of a new game creates a setup episode containing:

- game ID, color, and owner perspective;
- canonical and engine-oriented setups plus fingerprints;
- setup behavior file/model-state identity;
- all 40 selected types, legal inventory masks, behavior probabilities/log-probabilities,
  W/D/L predictions, conditional-entropy predictions, and sampling seeds;
- terminal outcome from the setup owner's perspective when the game completes.

Both sides train from the result: win `+1`, draw `0`, loss `-1` from that side's
perspective. Use all 40 prefixes with no move-style top-quartile advantage filter.

Under D10, the setup update follows the paper-shaped path directly:

- PPO ratio clipping: `0.2`;
- behavior probabilities always come from the recorded raw setup snapshot;
- fixed reverse `D_KL(current || behavior)` coefficient: `0.1`;
- no adaptive setup-KL controller, target, beta bounds, or calibration;
- W/D/L value cross-entropy with weight `0.5`;
- conditional-entropy loss with weight `1.0` and target `I/10`;
- printed setup advantage
  `delta = (outcome - E[behavior value]) + alpha(n) * (I - h_behavior)`;
- `alpha(n) = 0.1 * n^-0.3`, with `n` the shared one-based global tandem iteration;
- setup gradient-norm clip: `0.5`;
- constant Adam learning rate `5e-5`;
- setup EMA decay `0.999`; and
- five setup epochs per global iteration.

Train on every setup episode whose game completed in the current fixed-transition
iteration, both sides, exactly once. Do not impose a fixed setup quota, warm-up
minimum, age-balancing rule, or independent diversity gate. If no game completes,
record a skipped setup update. Persist the current iteration's completed buffer only
to prevent outcome loss or duplication across a crash.

## 9. Move schedules and regularization

The first run is not a schedule shootout. Use the accepted paper-shaped Phase 16 move
schedule, correctly re-horizoned to the measured 12-hour iteration count `N`:

```text
n_ref = ceil(0.125 * N)
lr(n) = clamp(1.5e-4 * (n / n_ref)^-1.1, 1.5e-5, 1.5e-4)
c_H(n) = max(0.001, 0.005 * n^-0.3)
move epochs per iteration = 1
EMA decay = 0.999
```

`n` is the one-based completed iteration and is checkpointed. Agent 4 estimates `N`
from the bounded preflight throughput rehearsal and freezes `N`, `n_ref`, and the
entire schedule curve before the production launch. Do not recompute the horizon from
changing production speed.

Move behavior-KL retains the accepted Phase 9 controller unless Agent 1 documents a
paper-required difference:

```text
target KL: 0.015
initial beta: 0.005
beta bounds: [1e-4, 0.2]
hard mean KL limit: 0.08
```

The move controller remains unchanged. Setup behavior KL is a separate reverse KL with
fixed coefficient `0.1`; never report it as adaptive beta or as the move controller.

## 10. Raw, behavior, and EMA identities

Raw weights generate all training data. EMA weights never act in the training
population. Every checkpoint stores both.

A paired Phase 17 checkpoint binds:

- exact Phase 9 start identity;
- raw and EMA move states;
- raw and EMA setup states;
- both optimizers, the move KL controller, and the fixed setup reverse-KL coefficient;
- move and setup scheduler positions and optimizer-step counts;
- all RNG namespaces and counters;
- active-game engine states and their setup episodes;
- boundary-target carry state;
- current iteration's unconsumed completed-setup buffer and setup-pool identity;
- run/config/source digests and elapsed active-training time.

Planned checkpoint/resume must preserve the active population exactly. If exact
active-game persistence proves impossible, Agent 4 must stop for operator review;
repeatedly discarding in-flight games would bias setup outcomes toward short games.

Evaluation exports contain paired EMA weights and a manifest. A move-only evaluator
still records the paired setup digest and marks setup use as `false`.

## 11. Post-training local evaluation contract

No evaluator runs while `RUN-2026-B` is training. The trainer only exports immutable
paired EMA candidates at h0 and every 30 active minutes through h12. After Agent 7
freezes the completed run and all candidate ordinals, Agent 5 evaluates them on the
same Mac Mini. No MacBook, SSH, network transfer, shared cross-computer storage, or
remote worker is part of the run.

One immutable composite benchmark manifest contains two named lanes:

1. `move_only`: fixed boards/setups and fixed evaluation opponents, EMA move weights;
2. `joint_move_setup`: fixed setup RNG cases and fixed opponent cases, paired EMA
   move/setup weights.

The preferred move-only base is the accepted full `phase16_benchmark_v1`:

```text
digest: ebd130198ea500248b32df990bee876583a10d53546f38a6346ec522407320c2
```

The new composite pack receives a new digest. Both lanes report overall EWR and
opponent/setup/color strata. Historical, Phase 9, rule-based, and stress opponents
are evaluation instruments only.

Candidate times are hour 0 and every 30 active minutes through hour 12: 25 candidates.
Each candidate is immutable and atomically published. After training, the evaluator
recomputes candidate, model-state, pack, config, source, and evaluator identities. A
local receipt binds every identity, result digest, host/environment identity, and
runtime. Never evaluate a mutable `latest` filename. Validate one frozen candidate,
then score the full frozen set sequentially. Failures and retries remain explicit and
can be repaired without changing the preserved training run.

## 12. Gates and time budgets

The standalone setup-network gate and every evaluator gate are retired. Prelaunch uses
Agent 4B's accepted tandem smoke plus Agent 4C's focused attribution/resume checks; do
not run another production-shaped smoke merely to authorize launch.

### Accepted correctness evidence — do not repeat

- rules, counters, orientation, legal actions, and observation/action identities;
- exact Phase 9 start digest;
- sampled-not-argmax action selection;
- both seats use the current raw policy;
- forced rebind test on an already-running game;
- true fixed transition count, and boundary targets under the section 6
  reduction invariant (`G-M4a`; `G-M4b` retired by operator decision D2);
- checkpoint/save/load/export identity;
- structural no-search and no-training-opponent assertions.

### Accepted Agent 4B tandem integrity smoke — do not repeat

- exact Phase 9 move identity and fresh setup initialization;
- both move seats use the current raw policy and sample legal actions;
- legal, inventory-correct, correctly oriented setups from the current raw setup
  model, with no library fallback;
- exact fixed-transition output count and boundary bootstrapping;
- at least one completed outcome produces a real five-epoch setup update;
- setup reverse KL coefficient exactly `0.1`, with no adaptive controller update;
- setup alpha matches `0.1 * n^-0.3` at the shared global iteration;
- one paired checkpoint round trip without lost or duplicated setup outcomes; and
- structural absence of search, belief, historical, and handcrafted training
  participants.

Do not run another standalone diversity soak, setup entropy gate, controller
calibration, queue-arrival study, strength test, or broad failure-injection campaign.

### Safety wiring

Preserve the accepted safety features and Agent 4C's focused integrity-stop test. Do
not run another general safety smoke or failure-injection campaign before launch.

## 13. Production stop policy

Stop immediately on:

- rules, orientation, legality, candidate, or digest mismatch;
- any decision recorded under the wrong current move-policy digest;
- nonfinite loss, gradient, parameter, or schedule value;
- setup generation/masking failure or silent fallback attempt;
- search or a non-current training opponent entering collection;
- evaluation result bound to the wrong candidate or benchmark;
- unrecoverable checkpoint/resume identity failure.

Under D10, fixed-pack EWR decline, high but finite KL, entropy decline, setup
concentration, diversity loss, and changing game lengths are warnings and experiment
results, not automatic stops. Continue the 12-hour run without tuning when identities
and numerical state remain valid.

Stop for wrong routing or identity, illegal actions/setups, silent setup fallback,
nonfinite numerical state, prohibited training participants, fixed-transition count
violations, corrupt persistence, lost/duplicated setup outcomes, or unrecoverable
resource exhaustion. These are the conditions that could silently falsify the result
or prevent safe continuation.

## 14. Checkpoint-selection rule

The last checkpoint does not win automatically. Agent 7 produces a Pareto shortlist
from eligible hour 6–12 candidates using:

1. mean composite-pack EWR;
2. worst opponent/setup stratum EWR;
3. three-point rolling-median direction from hour 6 through hour 12;
4. move-only non-regression;
5. setup legality plus descriptive entropy/diversity context;
6. KL, completed-episode flow, and training stability.

The report recommends a checkpoint and up to two alternatives. A single isolated EWR
peak cannot override a poor worst stratum. Setup concentration informs the tradeoff but
does not invalidate an otherwise attributable checkpoint under D10. The operator makes
the final promotion decision.

## 15. Agents, dependencies, and handoffs

| Agent | Charter | May start | Handoff |
|---|---|---|---|
| 1 | contract, paper map, identities, baseline | now | `phase17_contract_handoff_v1` |
| 2 | fixed-transition move learner | after Agent 1 | `phase17_move_handoff_v1` |
| 3 | autoregressive setup learner | after Agent 1 | `phase17_setup_handoff_v1` |
| 4 | tandem runner, persistence, schedules | complete at `c2c0365` | `phase17_tandem_handoff_v1` |
| 4B | simplified paper-shaped recipe conversion | complete at `3be8bba` | `phase17_simple_tandem_handoff_v1` |
| 4C | attribution and resume correction | now, from `3be8bba` | `phase17_simple_tandem_handoff_v2` |
| 5 | post-training local evaluation and shortlist | after Agent 7 freezes the run | `phase17_local_eval_handoff_v1` |
| 6 | short launch freeze | after Agent 4C | `phase17_launch_decision_v2` |
| 7 | 12-hour training and candidate freeze | after Agent 6 GO and operator launch approval | `phase17_run_closeout_v1` |

Agents 2 and 3 may work in parallel only after Agent 1 freezes their shared schemas.
The execution order is Agent 4C -> Agent 6 -> Agent 7 -> Agent 5. Agent 5 does not
perform remote discovery, configure another computer, or run concurrently with
training. It consumes Agent 7's frozen candidate ledger. Consume other agents only
through verified handoff artifacts, never their work-in-progress state.

Agent 3's historical `ready_for_tandem_integration: false` and Agent 4's D9 results
remain evidence, but neither is a launch gate under D10.

## 16. Additive namespaces

Use additive Phase 17 paths. Agent 1 may refine the exact module split in its handoff:

```text
stratego/training/phase17/
stratego/evaluation/phase17/
tests/training/phase17/
tests/evaluation/phase17/
scripts/*phase17*
checkpoints/phase17/
data/phase17/
reports/phase17/
```

Do not write production artifacts into Phase 9, 14, 15, or 16 directories.

## 17. Reports and claims

Each agent writes:

```text
reports/phase17/agent_0N_report.md
reports/phase17/agent_0N_*.json|jsonl|csv
reports/phase17/<declared_handoff>.json
```

Every report states what was built, exact commands/tests run, runtime, artifact
digests, failures, known limitations, and what was not established. Do not claim that
an unrun gate passed, that fixed-board EWR measures the setup policy, or that a
production failure establishes a scientific conclusion.

## 18. Later-phase boundary

Phase 17 ends with a frozen paired move/setup checkpoint recommendation. Only after
the operator promotes it may a later phase:

1. collect selected-policy trajectories;
2. train the joint autoregressive belief model separately;
3. attach belief-guided search;
4. add stochastic human-facing move selection.

No Phase 17 agent implements or quietly prepares belief-guided search.
