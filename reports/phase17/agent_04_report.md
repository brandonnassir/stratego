# Phase 17 — Agent 4 report
## Tandem move/setup runner, persistence, schedules, telemetry, and guards

_Written 2026-08-28. Governing documents: `00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md`,
`04_AGENT_4_TANDEM_RUNNER_AND_PERSISTENCE.md`, and
`08_OPERATOR_DECISION_D9_AND_AGENT_4_RELEASE.md`. Where they conflict, D9-B governs._

## 0. What this agent established, and what it did not

**The headline, in one paragraph.** The tandem system runs, and it persists exactly. Over
200 setup iterations against a real current-policy move signal the setup policy did
**not** concentrate the way Agent 3's standalone one did — 99.6% of baseline entropy
against 84.2%, flag effective support 12.74 against 10.11, and the 60% relative
predicate never came within reach of firing. But beta saturated at its ceiling for 97.5%
of those iterations, so part of that difference is a KL penalty holding the policy still
rather than a better outcome signal, and this rehearsal cannot decompose the two. Agent 3's
standalone gate remains failed, and Agent 6 owns the production decision.

Established:

- one bulk-synchronous tandem iteration that runs the accepted Agent 2 move learner and
  Agent 3 setup learner together, in the contractual order, with no change to either;
- that the tandem outcome signal is **decisive**: 1.87% draws across the 32,043 games of
  the concentration soak, and 0.24% across the 3,325 of the arrival diagnostic, against
  83.3% under Agent 3's uniform-random legal fixture. That is the confound decision D9-B
  set out to remove, and it is removed;
- that the completed-episode arrival rate **is not stationary** — it rises as games
  shorten — which is the main operational risk this agent found and could not fully
  settle;
- **exact** active-game persistence — the question common contract section 10 said to
  stop for operator review over. It does not need a stop: an interrupted run resumed
  from its paired checkpoint reproduces the uninterrupted run's next iteration
  bit for bit;
- a measured 12-hour horizon `N = 640` and the complete frozen schedule curve;
- a fixed, unbiased setup-episode budget of 572 with its capacity, warm-up, age and
  alarm constants;
- twelve of twelve bounded integration checks, including one injected supervisor stop;
- the `setup_tandem_concentration_reading` decision D9-B section 5 requires.

Not established, and not claimed:

- **any strength result.** No benchmark lane was run, no EWR was produced, no opponent
  was played. Nothing here says the tandem system is better or worse at anything.
- **that Agent 3's standalone setup gate passed.** It failed on S6 and its
  `ready_for_tandem_integration: false` is unchanged and correct. D9-B is a narrow
  integration override, not a pass.
- **the production setup-entropy decision.** Agent 6 owns it. This report supplies the
  tandem evidence and states the reading plainly; it does not move a threshold.
- **that a real move signal *causes* the setup policy to stay diverse.** Section 13.3:
  the behavior-KL penalty saturated, and this rehearsal cannot separate its restraining
  effect from the outcome signal's.
- **what indexes the setup alpha schedule.** Section 7.4. Agent 4 may not tune alpha, so
  the question is surfaced with its arithmetic rather than answered.
- **where the mean game length settles.** Twenty iterations is 3% of a 640-iteration
  run, and the trend had not flattened.
- **the external evaluation cadence.** Agent 5's remit; nothing was transferred.
- **that production speed will match the rehearsal.** The horizon is frozen from the
  measurement below and is never recomputed from production speed; telemetry records
  the difference.

## 1. Integration baseline

Decision D9-B section 2 forbids integrating uncommitted or moving Agent 2/3 inputs, so
their source, tests, gate artifacts, reports and handoffs were committed first, before
any Agent 4 source existed:

```text
baseline commit  22e967e908a951dd5ec25b378e362f8872feae2d
subject          Freeze Phase 17 Agent 2/3 integration baseline for Agent 4
parent           2403fb40d44c269f341fd081341644db90d0fd4a
```

The three unrelated pre-existing tracked modifications named in Agent 1's
`source_identity.deliberately_left_unstaged` — `reports/phase13/phase14_launch_manifest_v1.json`,
`stratego_project_docs/05_project_plan.md` and `stratego_project_docs/README.md` — were
left unstaged and untouched. No `git clean`, `stash`, `checkout` of a modified tracked
file, `reset`, or history rewrite was used.

`scripts/run_phase17_preflight.py --role verify-inputs` re-derives every input digest
from the working tree: **67/67 pass** (`reports/phase17/agent_04_input_verification.json`).
That covers Agent 2's nine source and nine test files with both closure digests and its
four bound artifacts, Agent 3's fifteen source files with its closure digest and two
bound artifacts, both handoff document digests, the Agent 1 contract handoff, and the
Phase 9 start checkpoint's file sha256 `dfd698e5…`.

| Input | Digest |
|---|---|
| `phase17_contract_handoff_v1` (json document) | `0ad902599a2a17e69ecf28d32c13211f4e0e4739db6ae93544081a8b73330b02` |
| `phase17_move_handoff_v1` | `6198bd6f32122976329d9ac4fd65c1ac696182236a7c94d764d35cb41a0df464` |
| `phase17_setup_handoff_v1` | `7d77cf2d47a7c0bb076ddd1ffdeb73f8bcb16297a58902819500b4de0cf84f00` |
| Phase 9 start `model_state_digest` | `f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd` |

## 2. Upstream documentation irregularities

Recorded, not repaired. Agent 4 does not edit Agent 3's report or its historical gate
evidence. All three are recorded in `agent_04_input_verification.json` under
`upstream_documentation_irregularities`, and in all three the **implementation is
already correct** — only the prose is wrong.

| id | field | says | governing |
|---|---|---|---|
| A4-UI1 | `d5_resolution.controller_update_cadence` | once per setup **epoch** | once per setup **iteration**, on the **final** epoch's mean reverse KL (D9-B §3) |
| A4-UI2 | `operator_decisions_resolved[D5].resolution` | KL target `0.0037` | `0.0018` (D9-B §3, and the same handoff's own `config.kl_controller.target`) |
| A4-UI3 | `consumes.handoff_digest` | a `*_digest` field holding a **file** sha256 | Agent 1's encoding rules reserve `*_digest` for the json-document digest |

A4-UI1 is the one the instruction singles out. `setup_learning.SetupTrainer.update`
already sets `control_kl = iteration_kl[-1]` and calls `controller.update` **once**,
after the five-epoch loop — the accepted D5 resolution. The obsolete per-epoch behavior
was not implemented and is asserted against in
`tests/training/phase17/test_runner_tandem.py::test_the_setup_controller_steps_once_per_iteration_on_the_final_epoch`.

A4-UI3 is why a naive verifier reports a false failure: reading
`consumes.handoff_digest` as a json-document digest gives `0ad90259…` against the
recorded `75568fc8…`. The recorded value is the file sha256 and verifies exactly.

## 3. The MPS-only test, resolved on the training device

D9-B section 4 records `298 passed, 1 skipped` on a host without MPS, and requires the
skip be resolved on the actual training device or setup sampling be bound to CPU.

MPS **is** available here (`torch 2.13.0`, `macOS-26.5.2-arm64`), so
`test_generation_on_mps_produces_only_legal_setups` ran rather than skipping:

```text
pytest tests/training/phase17 -q     ->  299 passed in 27.15s   (0 skipped)
```

Setup sampling is therefore not bound to CPU, and the production config runs setup
generation on MPS. Note this resolves the *device* question only: the report wording
`299 passed` was a count imprecision on the host D9-B used, and that remains true of
that host.

## 4. What was built

| Module | Purpose |
|---|---|
| `stratego/training/phase17/runner.py` | `TandemConfig`, `Phase17SetupProvider`, `TandemCollector`, `TandemRunner`: the ten contractual steps, exact capture/restore, and the D9-B concentration reading |
| `stratego/training/phase17/checkpoint.py` | atomic paired checkpoint, fail-closed load, exact active-game capture/restore |
| `stratego/training/phase17/queue.py` | the frozen setup-episode budget policy and its sustainability arithmetic |
| `stratego/training/phase17/telemetry.py` | the frozen JSONL row schema and a checkpointable, verifiable append position |
| `stratego/training/phase17/supervisor.py` | every section 13 predicate with its consecutive-count state and reset rule |
| `stratego/training/phase17/export.py` | immutable paired EMA candidates and the 30-minute cadence |
| `scripts/run_phase17_preflight.py` | input verification, throughput probe, schedule freeze, integration rehearsal, concentration soak |
| `scripts/run_phase17_training.py` | `TrainingSession` and the production entry point Agent 6 binds |

Neither Agent 2's nor Agent 3's source was modified. The one integration seam is
`TandemCollector._retire`, which calls `super()._retire` unchanged and then enqueues
the finished game's two setup episodes. `_retire` is the single point at which a game is
known finished and still addressable; scanning for disappeared game ids afterwards would
have worked until the first game that finished and was replaced inside the same window.

`stratego/training/phase17/queue.py` is deliberately **not** a second queue. Agent 3's
`SetupEpisodeQueue` is accepted and is consumed unchanged; this module holds only the
budget arithmetic. Two places where an episode can be dropped is exactly the failure
mode common contract section 8 forbids.

## 5. Exact joint persistence

Common contract section 10 says a planned checkpoint/resume must preserve the active
population **exactly**, and that Agent 4 stops for operator review if that proves
impossible. It does not prove impossible, and there is no blocker to report.

Three things have to survive, and the accepted code already knows how to carry all three:

```text
engine state   create_snapshot(state, include_history=True) / restore_snapshot
               -- the engine's own codec, every field, including the derived
               event log and the action history
trajectory     the builder's decisions, per-ply snapshots and actions, which is
               what finish() seals a valid GameRecord from
target carry   Agent 2's SeatTrace.to_dict() / from_dict()
```

Nothing is replayed and nothing is recomputed. Replaying the action history would
rebuild the engine state correctly but could not rebuild the *builder*, whose
per-decision records hold the behavior probabilities the model produced at the time —
probabilities a changed model can no longer reproduce. So the records are stored.

`rules` is stored as a version string, not as an object, and the load rebinds the
accepted `CORPUS_RULES` after checking it. A checkpoint that carried the frozen
dataclass would be a checkpoint that could silently reintroduce a foreign ruleset.

A resumed game keeps the setups it was created with: `Phase17SetupProvider.assign`
consumes a staged `resuming` entry before it will touch a pool, so a re-seated game
cannot be handed a fresh draw.

### 5.1 The proof

`agent_04_resume_rehearsal.json`. An uninterrupted run and an interrupted-then-resumed
run were both advanced to the same point and their **next** iteration compared field for
field across 512 transitions and 16 fields:

```text
sampled_actions            the same games, at the same plies, choosing the same actions
advantage_targets          identical to float32
wdl_targets                identical to float32
target_provenance          identical
terminal_results           identical
games_finished             identical
move_raw / move_ema        identical model-state digests after the update
setup_raw / setup_ema      identical model-state digests after the update
cell_digest                identical
queue_depth                identical
setup_skipped              identical
setup_optimizer_steps      identical

differing_fields: []
```

The same comparison is a unit test:
`tests/training/phase17/test_runner_tandem.py::test_a_round_trip_reproduces_the_next_iteration_exactly`.

### 5.2 The proof runs on CPU, deliberately — and why that matters for production

**MPS is not bitwise reproducible run to run on this host.** This was measured before
any resume claim was made, by running two *identical uninterrupted* runs and comparing
them (`agent_04_integration.json` → `device_determinism_baseline`):

| device | bitwise reproducible | max advantage difference | max stored W/D/L difference | actions identical | terminal results identical |
|---|---|---|---|---|---|
| cpu | **yes** | 0.0 | 0.0 | yes | yes |
| mps | **no** | 9.83e-07 | 5.07e-07 | yes | yes |

over 16,384 rows. So a bitwise resume assertion cannot pass on MPS for *any*
implementation, correct or not, and a resume comparison run there would have reported a
"failure" that was the device. The persistence contract is therefore proved on the
deterministic device, and the production device's own noise floor is measured separately
so the MPS reading can be read against something.

**Consequence Agent 6 must carry into the launch manifest:** no production stop
predicate may be wired to a bitwise model-digest comparison across a resume while
training on MPS. Digest equality is the right check for a checkpoint's *contents*
(and is enforced, fail-closed, on load); it is not a valid check for
"did the resumed run continue identically" on a non-deterministic device. The two
absolute readings above are what a tolerance would have to be set against.

### 5.3 Known limitation, recorded in the checkpoint schema

`divergence_rows_lost_to_resume`. `whole_game_divergence` covers only the post-resume
rows of a game that spanned a resume, because the pre-resume rows were emitted into a
closed window and their row objects are gone. This is telemetry that operator decision
D2 explicitly made non-gating, it is named in `checkpoint_schema()`, and it affects no
target, no update and no stop predicate.

## 6. Schedule horizon

`scripts/run_phase17_preflight.py --role throughput`, on the production device, at the
production budget, with real move forward passes, real boundary target creation, one
real move epoch, real setup generation and five real setup epochs. Eight iterations;
the first is discarded as warm-up.

```text
mean iteration            67.40 s
measurement rows          2..8      (iteration 1 discarded)
N  = 12 h / 67.40 s     = 640
n_ref = ceil(0.125 * N) = 80
p_setup = 0.3*ln(42376)/ln(640) = 0.4946710477758662

schedule_digest  2607502309d51f2c6d7ecb9796d74d1f9fab5de43dd32bb590a2e5e9337787df
curve_digest     e3bb29ac0744daa38a98126b684d85e9383c14163956a8f568035e5769f5bb8a
```

Where the time goes, per iteration:

| component | seconds | share |
|---|---|---|
| collection (both seats, sampled, batched) | 35.75 | 53.0% |
| move optimization (one epoch) | 29.32 | 43.5% |
| setup generation (2 × 512 pool) | 1.46 | 2.2% |
| setup optimization (five epochs, 572 episodes) | 0.97 | 1.4% |

The setup half is **3.6%** of an iteration — below Agent 3's CPU projection of 5.8%, so
five epochs are comfortably affordable and no reduction was considered. Peak process
memory was 13,005 MiB.

`transitions_trained` is 16,384 of 65,536 harvested — exactly the accepted 0.75-quantile
advantage filter, not a shortfall.

### 6.1 A measurement error worth recording: never size the population from a reduced budget

The first population probe ran at a 4,096-transition budget and ranked
`population 96 > 256`. At the production 65,536 budget the ranking **inverts**:

| device | population | 4,096 budget | 65,536 budget |
|---|---|---|---|
| mps | 96 | 631 t/s | 778 t/s |
| mps | 256 | 501 t/s | **1028 t/s** |
| mps | 512 | — | 911 t/s |

The cause is Agent 2's carry-forward A2-CF3: each window close costs one extra forward
pass per open seat trace, up to `2P` rows on top of the budget. At 4,096 transitions
that is 12.5% overhead for `P = 256` and 4.7% for `P = 96`; at 65,536 it is 0.8% and
0.3% and stops mattering, leaving batching efficiency to decide. A reduced-budget probe
does not merely have more noise here — it answers a different question.

Frozen: **population 256 on MPS**, the peak of the production-budget curve.

### 6.2 CF1 is recorded, not changed

Agent 1's carry-forward CF1 asks whether the contract's move LR shape should be
replaced by the paper's before Agent 4 freezes the schedule. Measured against `N = 640`:

```text
move LR      1.5e-4 -> 1.5229732431679412e-05 ; the 1.5e-5 floor is NEVER reached
move c_H     0.005  -> 0.001 ; the floor is reached at n = 214, i.e. 33.4% of the run
```

Both are exactly what CF1 predicted. The common contract wins over the paper
(section 2) and no agent may amend it without the operator, so the **contract** form is
what is frozen. This is recorded in `agent_04_schedule.json` under `carry_forward_cf1`
for Agent 6 and the operator to decide on; Agent 4 did not change it.

## 7. The setup budget: three things the rehearsals changed

This is the part of the work that went furthest from where it started, and every
correction came from a rehearsal rather than from reading.

### 7.1 The margin has to point up, and my first version pointed it down

I initially wrote `budget = floor(arrivals / 1.25)`, reasoning that a budget below the
arrival rate leaves slack for variance. It does the opposite. Agent 3's
`SetupEpisodeQueue` **raises** at capacity — it never evicts, because silent dropping is
what section 8 forbids — so a budget below the arrival rate makes the queue grow without
bound and kills the run some hours in. The production-scale probe found it immediately,
with `setup episode queue is at capacity 256; refusing to evict`.

Writing the dynamics down: with arrivals `A` and fixed budget `B`, depth moves as
`D + A - B` on an update and `D + A` on a skip.

```text
B <  A   depth grows linearly            -> capacity overflow, run dies
B == A   zero-drift random walk          -> overflows on variance alone
B >  A   bounded; skips at rate 1 - A/B  -> the intended equilibrium
```

So `B > A`, and the price is a skip rate of `1 - A/B` — a *counted* event, never a
silently shrunk update. The batch size never varies; only *whether* an update happens
varies, so a fixed budget cannot invisibly prefer short games.

Two further sizing choices follow: the budget is derived from the **peak** steady-state
completion rate rather than the mean (a fresh population ramps — the throughput
rehearsal's first window finished 125 games against a steady-state 240–260 — and
understating arrivals is the one error this queue cannot absorb), and warm-up is **two**
budgets deep rather than one (the first games to finish in a fresh population are the
*shortest* games in it, and a first update sized to one budget would train on that
biased tail).

### 7.2 The backlog alarm was too close to the ceiling to ever fire

`P8` needs **three consecutive** windows before it stops. My first alarm sat at 90% of
capacity. With capacity at eight budgets that leaves 0.8 budgets of headroom, so a run
whose arrival rate had risen reached the capacity — which *raises* — before the second
reading. The alarm could not complete, and the clean supervisor stop it exists to
produce was unreachable.

The alarm is now at **50% of capacity**, which leaves four budgets of headroom: even at
a *doubled* arrival rate, where the queue gains a whole budget per window, four windows
fit under the ceiling and `P8` has room to accumulate. `SetupBudgetPolicy.__post_init__`
now refuses at freeze time any alarm whose `headroom_windows` does not exceed its own
consecutive count, so this cannot be reintroduced by changing a constant.

As a backstop, `TandemRunner.run_iteration` refuses to *start* a window the queue could
not absorb. Agent 3's raise is correct but it lands in the middle of a window, after
tens of thousands of transitions have been collected, and discards all of it. Checking
first means a run that has genuinely run out of queue stops having lost nothing.

### 7.3 The arrival rate is not stationary — it rises, because games get shorter

This is the finding that matters most for the production run, and it was not visible
until the tandem system was actually trained.

`reports/phase17/agent_04_arrival_rate.json`, twenty iterations at a 40,000-transition
budget with the queue capacity deliberately unbounded so the trend would be visible
rather than fatal:

| iteration | mean game length (plies) | arrivals / iteration | queue depth |
|---|---|---|---|
| 3 | 287.9 | 292 | 324 |
| 6 | 286.6 | 292 | 214 |
| 12 | 242.2 | 356 | 338 |
| 16 | 236.0 | 390 | 386 |
| 20 | **166.9** | **492** | **890** |

Mean game length falls **42%** in seventeen iterations and the arrival rate rises
**68%** with it, because the identity is exact:

```text
arrivals = 2 * move_budget_transitions / mean_game_length
```

The queue drained to 214 by iteration 6 and then grew monotonically from iteration 16.
Twenty iterations is 3% of a 640-iteration run, and where the mean game length settles
is **not established**.

The consequence for the budget is direct: a budget frozen from an early measurement is
overrun later, and the correct sizing target is the *high* end of the arrival rate, not
the observed one. The cost of that is a higher skip rate early rather than an overflow
late — a good trade, because a skip is a recorded event and an overflow is a dead run.

I want to be careful about what this is not. A shorter game is not a better game and
this says nothing about strength; it is a throughput observation and nothing else.

### 7.4 A blocking question this raises for Agent 6: what indexes the setup alpha?

Sizing the budget for a rising arrival rate means a larger budget, which means **fewer,
larger setup updates**. That collides with something I may not decide.

`SetupTrainingConfig.alpha` is indexed by the **setup** iteration. In Agent 3's
standalone soak that was unambiguous, because one setup iteration *was* one iteration.
In tandem they are different counters: a run of `N = 640` move iterations performs `S`
setup updates where `S = 640 * A / B` and `S < 640` whenever the budget exceeds the
arrival rate. Agent 1's contract section 8 defines the exponent against "`N`, frozen by
Agent 4" — which is the twelve-hour *iteration* count, i.e. the move counter.

The anneal therefore does not complete, and by a quantified amount:

| setup updates `S` | `alpha(S)` | × the paper endpoint `0.004091` |
|---|---|---|
| 125 | 0.009177 | **2.24** |
| 200 | 0.007274 | 1.78 |
| 300 | 0.005952 | 1.45 |
| 500 | 0.004623 | 1.13 |
| 640 | 0.004091 | 1.00 |

Decision D9-B section 6 says plainly: *"Do not tune alpha, KL, epoch count, model size,
or the entropy threshold inside Agent 4."* Choosing a budget that determines how far
alpha anneals is tuning alpha by another route, so **Agent 4 does not choose it**. The
question is surfaced with its arithmetic instead, as
`setup_alpha_indexing_unresolved: true` in the handoff.

For what it is worth, my reading is that indexing alpha by the **move** iteration `n`
resolves it cleanly — the anneal then completes over the twelve hours regardless of how
many setup updates the arrival rate permits, which is what "re-horizoned to `N`" appears
to mean — but that is a recipe change and it belongs to the operator, not to me.

### 7.5 What is frozen

`agent_04_throughput.json` → `setup_budget_policy`, derived from the throughput
rehearsal at the production 65,536-transition budget:

```text
budget                 572 episodes per setup update
capacity              4576   (8 budgets)
warm-up minimum       1144   (2 budgets)
max age                  8   iterations
backlog alarm depth   2288   (50% of capacity), 3 consecutive windows -> P8
headroom below the alarm  4.0 windows at a doubled arrival rate
measured arrivals      519.6 completed episodes per iteration
sustainability margin    1.1008
expected skip fraction   9.2%
```

The rehearsal confirmed the equilibrium exactly: at budget 572 against ~520 arrivals the
queue ran `250 → 106 → 20 → 516 → 432 → 340 → 224 → 134`, staying far below capacity,
with one explicit skip in seven update-eligible iterations.

**This budget is sized against the arrival rate observed in that eight-iteration
window, and section 7.3 shows that rate rises.** At the arrival rate measured at
iteration 20 of the arrival diagnostic — 785 completed episodes per iteration at the
production budget — a budget of 572 is below the arrival rate and the run would back up.
`P8` fires at depth 2288 and stops it cleanly, and the pre-window check refuses the
window that would raise, so the failure mode is a clean early stop rather than a dead
process or silent mis-training. But it *is* an early stop, and Agent 6 must decide
whether to accept that, re-size the budget against a stated game-length floor, or
resolve section 7.4 first. This is carry-forward **A4-CF4** and it is the single item I
would put in front of the operator before launch.

## 8. Telemetry

`phase17_tandem_telemetry_v1`. JSONL, one row per tandem iteration, `flush()` +
`os.fsync()` before the write returns. Required keys are asserted on every write, so a
missing field fails at the iteration that dropped it rather than at closeout: 27 move
keys, 27 setup keys, 13 system keys, all three blocks required.

### 8.1 Two silent zeros, found and removed

Both were guards that looked live and were not.

`MoveUpdate.means` stores every value under a `mean_` prefix — `mean_behavior_kl`, not
`behavior_kl`. My first telemetry row and my first supervisor wiring read the unprefixed
names through `.get(name, 0.0)`, which returns `0.0` forever. That fed **P2** (move mean
KL above 0.08) and **P6** (move entropy below 25% of its first-hour median) a constant
zero: P2 could never fire, and P6's first-hour median would have been zero, making it
unfireable too — while the telemetry row said both were being evaluated. A guard that is
quietly switched off is worse than no guard.

`move_means()` now raises on an unknown name rather than defaulting, and a test asserts
both that it raises and that the supervisor sees a genuinely nonzero move KL from a real
update.

The move **gradient norm** is a different case: Agent 2's trainer computes the pre-clip
norm inside `_step` and uses it only to refuse a non-finite gradient. Nothing carries it
out, and Agent 4 does not alter Agent 2's behavior. It is therefore reported as `null`
with `grad_norm_unavailable_reason` beside it, plus the fixed clip norm and the
non-finite-gradient counter — never as `0.0`, which a reader would take for a measured
collapse. Surfacing it would need a versioned amendment to Agent 2's trainer.

The setup side corroborates Agent 3's carry-forward A3-CF3 exactly: a real row shows
`L_h = 11.6` against a policy loss of order 0.3, and a pre-clip setup gradient norm of
41.2 against a 0.5 clip. That is the conditional-entropy head converging, not
instability, and the telemetry now expects it.

The same row shows the D5 finding directly: `control_kl = 0.000499` against
`mean_kl = 0.000188`, so the final epoch's KL is 2.7× the epoch mean. Averaging across
epochs really does understate the drift, and the controller reads the final epoch.

### 8.2 Durability

The append position is checkpointed as `(records, offset, last_record_digest)` and
`TelemetryWriter.resume` refuses a file whose tail does not reproduce that digest at
that offset. A crash between the fsync and the checkpoint leaves the file *longer* than
the recorded offset; that excess is a row the resumed run is about to produce again, so
it is truncated back rather than appended past — appending would leave two rows for one
iteration and no later reader could tell which one the training used.

## 9. The collapse supervisor

Every immediate (`I1`–`I7`) and persistent (`P1`–`P8`) predicate from common contract
section 13, each with a stable code, evidence payload, consecutive-count state,
severity, and a per-predicate reset rule. A reading that does not trip resets that
predicate's run to zero; without the reset, three trips spread across twelve hours would
eventually stop a healthy run. The consecutive state rides in the checkpoint, so a
resumed run damps from where it left off.

The supervisor may checkpoint and stop. It has no setter for a learning rate, a KL
target, an entropy coefficient, a population size, an epoch count, a setup batch, or a
benchmark case, and a test asserts that.

**The D9-B exception, precisely.** `P4` — setup mean prefix entropy below 60% of its
initial baseline for three checks — remains the default **production** stop predicate at
its unchanged threshold of `0.9257366873314787` nats. In `MODE_INTEGRATION` its severity
becomes `diagnostic` rather than `stop`: it still fires, still carries its full evidence
payload and consecutive count, and still appears in the telemetry row — it simply does
not set `should_stop`. Nothing else moves. `P5` (flag effective support below four) and
every other absolute floor stay hard in both modes; a test asserts that the two modes
compute the identical floor and the identical consecutive requirement.

## 10. Bounded integration rehearsal

`scripts/run_phase17_preflight.py --role integration`, 6 iterations at a 16,384
transition budget with population 256 on MPS. **12/12 checks pass**
(`agent_04_integration.json`), 441.7 s.

| check | result |
|---|---|
| h0 paired EMA export and digest re-verification | pass |
| h0 move EMA equals the accepted Phase 9 start | pass |
| one full transition iteration with completed and unfinished games | pass |
| at least one real setup update from completed outcomes | pass |
| five setup epochs observed and timed | pass |
| forced move-policy rebind observed in active games | pass |
| no search or training-opponent participants | pass |
| no search or training-opponent imports reachable from the runner | pass |
| telemetry schema and append-resume continuity | pass |
| the device the persistence proof runs on is bitwise reproducible | pass |
| paired checkpoint save/load continuation equivalence | pass |
| one injected supervisor stop, recorded with a safe exit | pass |

The rebind check is the one that matters most, because it is the Phase 16 defect this
phase exists to fix. Every game alive across all six windows recorded a *different*
acting `behavior_model_state_digest` in the first and last window, and the participant
ledger holds: zero unknown model states, zero rule or stress decisions, zero historical
participants, zero search participants.

The injected stop used an absolute floor (`P5`, flag effective support 2.0), which is
hard in every mode. The supervisor recorded the reason, the session wrote a paired
checkpoint, and that checkpoint re-read and re-validated cleanly — a safe exit, not a
corpse.

## 11. Tests

```text
pytest tests/training/phase17 -q                     386 passed, 0 skipped in 80.35s
pytest tests --ignore=tests/training/phase17 -q     7028 passed, 3 skipped in 604.69s
```

386 = Agent 2/3's 299 (all of which run on this device, including the MPS-only one)
plus 87 new Agent 4 tests: `test_runner_tandem.py` 41,
`test_checkpoint_persistence.py` 18, `test_supervisor_predicates.py` 28.

The 7,028/3 regression figure is byte-identical to what Agent 2's handoff recorded, and
was re-run after every Agent 4 change, so no accepted behavior moved. The second run
shared the machine with the concentration soak, which is why it is six seconds slower.

## 12. Production entry point

`scripts/run_phase17_training.py` holds `TrainingSession` — everything with a cadence:
checkpoints, 30-minute exports, telemetry rows, stop checks. `TandemRunner` holds the
ten steps of one iteration. The rehearsal and the production run therefore drive the
identical iteration code and differ only in how long they are asked to run.

The script **refuses to start** without both `--start` and `--i-am-agent-7`. Agent 4
rehearses; it does not launch. `--describe` prints the exact bound configuration for
Agent 6's launch manifest.

## 13. The setup tandem concentration reading (decision D9-B section 5)

`reports/phase17/agent_04_concentration.json`. The one thing Agent 3 could not do: run
the *same* setup learner against a real current-policy move signal instead of a
uniform-random legal fixture. 200 setup iterations, 396 move iterations, 103 minutes.

Everything on the setup side is held at Agent 3's soak values so the trajectories are
comparable iteration for iteration — 320 episodes per update, 64-episode minibatches,
five epochs, the D5 controller, diversity readings every 25 setup iterations, and the
same sample shape (160 Red + 160 Blue profiled together). The alpha horizon is `N = 640`
against Agent 3's 626, giving `p = 0.4947` against `0.4964` — a 0.3% difference in the
exponent. The **only** substantive difference is where the outcomes come from.

### 13.1 The confound is removed

```text
draw rate, tandem       1.87%   (32,043 completed games, this soak)
draw rate, tandem       0.24%   ( 3,325 completed games, the arrival diagnostic)
draw rate, Agent 3      83.3%   (uniform-random legal move fixture)
```

The two tandem figures differ because the arrival diagnostic ran a shorter, earlier
stretch at a larger move budget; both are two orders of magnitude below the fixture and
nothing here turns on which one is used.

Agent 3 named this as the confound it could not remove: "83% of its games draw and the
outcome term is largely noise. A PPO policy fitting noise concentrates." Under the real
Phase 9-derived current policy, 98% of games are decisive. Whatever else this reading
shows, the outcome term now carries signal.

### 13.2 The trajectory, side by side

| setup iteration | tandem H (nats) | tandem % of baseline | standalone % | tandem flag ES | standalone flag ES | tandem flag squares | standalone flag squares |
|---|---|---|---|---|---|---|---|
| 25 | 1.7055 | **110.5%** | 107.9% | **23.21** | 10.67 | 32 | 17 |
| 50 | 1.7311 | **112.2%** | 104.0% | **23.14** | 10.06 | 34 | 16 |
| 75 | 1.7143 | **111.1%** | 99.1% | **21.10** | 10.67 | 28 | 16 |
| 100 | 1.6807 | **108.9%** | 100.6% | **17.76** | 9.44 | 27 | 16 |
| 125 | 1.6474 | **106.8%** | 96.2% | **19.47** | 10.14 | 29 | 16 |
| 150 | 1.6096 | **104.3%** | 91.1% | **15.33** | 9.67 | 28 | 13 |
| 175 | 1.5573 | **100.9%** | 89.0% | **12.96** | 8.22 | 25 | 11 |
| 200 | 1.5368 | **99.6%** | 84.2% | **12.74** | 10.11 | 25 | 13 |

The relative predicate:

```text
floor                                    0.9257366873 nats (60% of 1.5428944789)
tandem readings below the floor          NONE
longest consecutive run on cadence       0
consecutive required to stop             3
```

**The 60% relative entropy predicate never came close to firing under the tandem
signal.** At setup iteration 200 the tandem policy sits at 99.6% of its baseline where
the standalone was at 84.2%, and the gap widened monotonically from 2.6 points at
iteration 25 to 15.4 points at iteration 200. Flag effective support tracked 1.3–2.3×
the standalone's throughout and flag square support roughly double.

Every absolute floor held, with room:

```text
minimum flag effective support     12.74   (floor 4.0)
reflection-class uniqueness         1.0000 (unchanged at every reading)
minimum class distance             20
legality / orientation / fallback failures   0 / 0 / 0
setup optimizer steps              5,000 over 200 updates
every update moved the raw digest  yes
every update consumed the budget   yes (320, exactly, every time)
setup KL max                       0.014072 against the 0.08 hard limit
```

### 13.3 The finding I did not go looking for: the D5 KL target is mis-calibrated for tandem

This is the part Agent 6 needs most, and it complicates the reading above.

```text
beta at its UPPER bound (1.0)      97.5% of setup iterations
beta first / min / max / final     0.15 / 0.15 / 1.0 / 1.0
control KL above the 0.0018 target 167 / 200 iterations  (83.5%)
observed mean KL                   0.002326   (1.29x the target)
observed max KL                    0.014072   (5.7x under the 0.08 hard limit)

Agent 3's v3 standalone, for contrast:
beta final 0.0171, beta max 0.3375, 14.5% at the LOWER bound
```

Beta saturates at its ceiling by setup iteration ~21 and stays there. This is the exact
mirror of the failure D5 was raised to fix — Agent 3's v1 pinned beta at its *lower*
bound for 100% of iterations, and Agent 3's own note applies unchanged: *"a controller
living at a bound is not regulating anything."* Its stated `pinned_fraction_limit` was
0.5; the tandem run sits at 0.975.

Two things follow, and they point in opposite directions:

1. The **safety** function is intact. The observed KL never approached the `0.08` hard
   limit — it stayed 5.7× under it — so nothing here is a stop condition, and `P3` never
   tripped.
2. The **regulation** function is gone. Beta has no upward headroom, so if the setup
   policy's drift rose, the controller could not respond. The target `0.0018` was
   calibrated on Agent 3's standalone KL scale; the tandem signal produces a different
   scale, and the frozen constants do not fit it.

**And this materially qualifies section 13.2.** A beta of 1.0 is roughly 59× Agent 3's
final 0.017, so the tandem setup policy is being pulled toward its behavior snapshot far
harder than the standalone one ever was. That restraint is a plausible partial cause of
the entropy holding up, and **this rehearsal cannot separate it from the outcome-signal
effect.** The honest statement is therefore narrower than "a real move signal prevents
concentration":

```text
Under the real current-policy tandem signal, with the D5 constants as frozen, the
setup policy did not concentrate over 200 setup iterations where the standalone one
clearly did. Part of that difference is attributable to a behavior-KL penalty that
saturated at its ceiling, and this rehearsal does not decompose the two.
```

Decision D9-B section 6 forbids Agent 4 tuning the KL, so the constants are unchanged
and the measurement is reported as it stands. This is carry-forward **A4-CF7**.

### 13.4 What this reading does not establish

- **No strength claim.** No benchmark lane, no opponent, no EWR. A setup distribution
  that stays diverse is not thereby a better one.
- **200 setup iterations, not 626.** Agent 3's standalone crossed the floor first at
  iteration 475. The tandem run was not carried that far, and a trajectory that is
  falling — 112.2% at iteration 50 to 99.6% at iteration 200 — may well cross later.
  What is established is that the two trajectories *separate*, and that the separation
  grows.
- **A reduced move budget.** The soak ran at 12,000 transitions per iteration rather
  than the production 65,536, so that 200 setup updates fit in a bounded rehearsal. The
  move policy is real, current, sampled and trained every iteration, but its windows are
  smaller than production's.
- **The beta confound in 13.3**, which is the largest single caveat.

## 14. Artifacts

| file | sha256 (first 32) | bytes |
|---|---|---|
| `agent_04_input_verification.json` | `d1987b1ed8a9f0f13c674ef5c588647a…` | 18,064 |
| `agent_04_throughput_probe.json` | `480832fa3103d6be5eaf413026398be2…` | 2,342 |
| `agent_04_throughput.json` | `1e098c307a596666f312e784086b7a82…` | 9,865 |
| `agent_04_schedule.json` | `a1271b40ac30619bac26dedd7e94d8c7…` | 119,490 |
| `agent_04_arrival_rate.json` | `1530d12ef8cdf93641d09eca1354a772…` | 6,702 |
| `agent_04_integration.json` | `9f9b3f13640406a6b6020e462763da3a…` | 20,740 |
| `agent_04_resume_rehearsal.json` | `c55cddde55132ee362e7258a31790b28…` | 4,507 |
| `agent_04_concentration.json` | `34d16ec6043297011ac88ffaaff68215…` | 274,351 |
| `phase17_tandem_handoff_v1.json` | `ceee4d2e4cac2da279df3e67490c948b…` | 323,276 |

```text
agent 4 source closure   109e2dacb823ba2a33e771e068ae54d611fa5443873c7aaa105bf1167997fc30
agent 4 test closure     05088aa64d25afef1a66ae937af27025d19ec8ce705200b87ed13575597a98c1
config digest            7af25953a763f115d681674bcb2c3527095d5365b0c09b2cbb565f154b1031ff
schedule digest          2607502309d51f2c6d7ecb9796d74d1f9fab5de43dd32bb590a2e5e9337787df
curve digest             e3bb29ac0744daa38a98126b684d85e9383c14163956a8f568035e5769f5bb8a
handoff digest           8c0416f420bd740864dcee3bb701d530e682897c59b727632a3839b67f2ebd71
integration baseline     22e967e908a951dd5ec25b378e362f8872feae2d
```

The artifact digests above are of the files as written; re-deriving the source and test
closures is `scripts/run_phase17_preflight.py --role handoff`, and re-deriving every
upstream input digest is `--role verify-inputs`.

## 15. Commands

```bash
.venv/bin/python scripts/run_phase17_preflight.py --role verify-inputs
```

```bash
.venv/bin/python scripts/run_phase17_preflight.py --role probe --probe-budget 65536 --probe-iterations 2 --devices mps --populations 96 256 512 --setup-budget 4096
```

```bash
.venv/bin/python scripts/run_phase17_preflight.py --role throughput --device mps --setup-device mps --budget 65536 --population 256 --pool-size 512 --setup-budget 572 --freeze-setup-budget 572 --iterations 8 --assumed-iterations 700
```

```bash
.venv/bin/python scripts/run_phase17_preflight.py --role integration --device mps --setup-device mps --budget 16384 --population 256 --pool-size 512 --setup-budget 64 --iterations 6 --assumed-iterations 640 --resume-warm-iterations 2
```

```bash
.venv/bin/python scripts/run_phase17_preflight.py --role concentration --device mps --setup-device mps --budget 12000 --population 256 --pool-size 512 --setup-budget 320 --setup-iterations 200 --reading-every 25 --samples 160 --assumed-iterations 640
```

```bash
.venv/bin/python scripts/run_phase17_training.py --describe
```

## 16. Carry-forward for Agent 6

| id | detail |
|---|---|
| A4-CF1 | **MPS is not bitwise reproducible.** Do not wire any production stop predicate to a bitwise digest comparison across a resume. Measured noise floor: 9.83e-07 advantage, 5.07e-07 W/D/L over 16,384 rows. |
| A4-CF2 | **CF1 (move schedule paper fidelity) is unresolved and now quantified at `N = 640`:** the LR floor is never reached; the entropy floor is reached at `n = 214` (33.4% of the run). Recorded, not changed. |
| A4-CF3 | **The production relative-entropy predicate is unresolved.** See section 14 and `setup_tandem_concentration_reading`. Agent 6 owns the decision, under contract section 13's rule that the threshold may not be silently moved to manufacture a pass. |
| A4-CF4 | **The setup-episode arrival rate rises during training and the frozen budget is sized against an early window.** Mean game length fell 42% in seventeen iterations (287.9 → 166.9 plies) and arrivals rose 68% with it. At the iteration-20 rate the frozen budget of 572 sits below arrivals; `P8` and the pre-window check turn that into a clean early stop rather than a dead process, but it is still an early stop. Decide before launch: accept it, re-size against a stated game-length floor, or resolve A4-CF6 first. **This is the item I would put in front of the operator first.** |
| A4-CF5 | **`divergence_rows_lost_to_resume`:** boundary-target divergence telemetry covers only post-resume rows for a game that spanned a resume. Non-gating under D2. |
| A4-CF6 | **What indexes the setup alpha is unresolved, and A4-CF4's fix depends on it.** `alpha` is indexed by the *setup* iteration; in tandem that counter runs slower than the move counter, so the anneal does not complete — at 125 setup updates it ends 2.24× above the paper endpoint. Contract section 8 defines the exponent against the twelve-hour iteration count, which reads like the move counter. Agent 4 may not tune alpha (D9-B §6), so this is surfaced, not decided. Flagged as `setup_alpha_indexing_unresolved`. |
| A4-CF7 | **The D5 setup KL target is mis-calibrated for the tandem signal.** Beta saturates at its upper bound `1.0` for **97.5%** of setup iterations — the mirror of the lower-bound pinning D5 was raised to fix, and far past Agent 3's own `pinned_fraction_limit` of 0.5. The safety function is intact (max KL 0.0141 against a 0.08 limit) but the regulation function is gone. This also **qualifies the section 13 result**: a beta 59× Agent 3's final value is restraining the setup policy, and the rehearsal cannot separate that restraint from the outcome-signal effect. Agent 4 may not tune the KL (D9-B §6). |

