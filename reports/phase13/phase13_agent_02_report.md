# Phase 13 — Agent 2: Exact Final-Training Integration

Run date: 2026-08-20. Task: `instructions/phase_13_final_training_integration/02_AGENT_2_FINAL_TRAINING_INTEGRATION.md`.

The Phase 14 final-training runner is built from Agent 1's frozen contract and the already-accepted
project components. **The implementation agrees with `phase13_final_training_contract_v1.json` on every
value it restates (0 disagreements, checked mechanically), the whole 168-hour machinery runs end to end
on the two declared test seams, and search is absent from the training import graph.** No 90-minute
rehearsal was run, no strength tournament was run, no Phase 14 run was started, and Agent 3 was not
started.

## 1. What this task did and did not do

Did: implemented the frozen contract as executable modules; integrated the accepted Phase 9 collector,
loss, KL controller, targets, loader and rollout store; built the Phase 14 population mixer, historical
archive and bounded active pool, checkpoint hierarchy, candidate evaluator, storage/retention
integration, telemetry/control surface, deadline controller and the test-only clock/scheduler seam;
proved the long-horizon events with a controllable clock; wrote the integrated config digest, tests and
artifacts.

Did not: tune the model, run the rehearsal, compare LRs, mixtures or budgets, change any Agent 1 value,
touch search, touch the spent Phase 11 bank, modify any accepted Phase 9/10/11/11B/12 artifact, or start
Phase 14. Every new file is additive and scoped to Phase 13/14.

## 2. Starting artifacts, verified before implementation (section 1)

Verified live this session, not copied from the document:

```text
phase13_final_training_contract_v1     sha256 65d1f941a326a1343dce597082c3b525203ef7182f73c759ac6eb04d87a12cdf
phase13_setup_census_v1                bound through phase14_setup_source_v1
phase14_setup_source_v1                identity phase14_setup_source_v1, selector config 6e227815bc3cb44f...
phase14_checkpoint_selection_pack_v1   pack_content_digest 896a753b3d568902e93e803f1a45de9e8834ff1cdf90bc08cfacf90bcf0c2bde (recomputed from the document)
phase14_checkpoint_selection_rule_v1   bound to the same pack digest
checkpoints/phase9/selfplay_c1_v1.pt   dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea (re-hashed)
checkpoints/phase8/warmstart_c1_v1.pt  f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca (re-hashed)
```

`phase14_contract.verify_against_frozen_contract()` compares the implementation against the frozen
document value by value — starting checkpoint and digests, LR9 and both continuation LRs and their
multipliers, segment hours and the transition, both mixtures including all five handcrafted families,
pool weights and anchor digests, cadences and candidate count, the pack digest, deadline seconds, the
setup-source identity, the storage volume and reserve, and the three search prohibitions. It returns
**0 disagreements**, and `assert_matches_frozen_contract()` runs again at run start, so a later edit to
either side stops the run rather than changing it.

One correction made during implementation: the accepted C1 parameter count is **863,959** (bound from
the accepted Phase 10B contract, together with the C1 config digest `31ca84ab140c523e...`), not the
value first written into the Phase 14 contract module. The trainer refuses any other architecture.

## 3. Module map (section 18 deliverables)

| module | what it owns |
|---|---|
| `stratego/training/phase14_contract.py` | every frozen value, the `Population` object, and the mechanical check against Agent 1's document |
| `stratego/training/phase14_seed.py` | Phase 14 roots, domains, game-id scheme and per-decision streams |
| `stratego/training/phase14_clock.py` | `SystemClock` / `ManualClock`, `RunWindow`, `DeadlineController` — the scheduler seam |
| `stratego/training/phase14_pool.py` | the durable archive and `phase14_active_pool_v1` (`f(k)`, weights, exact partition) |
| `stratego/training/phase14_schedule.py` | the population mixer and one scheduled logical game |
| `stratego/training/phase14_setup_source.py` | `phase14_setup_source_v1` as a Phase 9-compatible collection source |
| `stratego/training/phase14_collector.py` | population self-play on the accepted `GameRunner` and rollout store |
| `stratego/training/phase14_trainer.py` | the bulk-synchronous PPO/KL/value/belief learner |
| `stratego/training/phase14_checkpoint.py` | hot ring, durable archive, behavior snapshots, candidate marks, eval export |
| `stratego/training/phase14_storage.py` | layout, reserve monitor, disposable marks, the deletion guard |
| `stratego/training/phase14_telemetry.py` | the frozen metric list and the emergency-stop-only control surface |
| `stratego/training/phase14_runner.py` | the run loop: start, resume, units, cadences, deadline, finalize |
| `stratego/training/phase14_config.py` | `phase13_integrated_training_config_v1` and its digest |
| `stratego/evaluation/phase14_candidates.py` | the fixed-pack candidate evaluator, ledger and selection rule |
| `scripts/run_phase13_agent02.py` | the short integration checks and the artifacts |
| `tests/training/test_phase13_agent02.py` | 61 tests over all of the above |

## 4. Objectives, LR and entropy (sections 4–5)

The objective is **not reimplemented**. `phase14_trainer` calls the accepted
`phase9_loss.phase9_batch_loss` and the accepted `phase9_trainer.KLController`, and reaches advantages,
the filter, standardization and the WDL/belief targets through the accepted `phase9_targets`. The
belief auxiliary is present and contracted, so it runs unchanged at **weight 0.25** over every learner
decision — the integration run's belief loss is non-zero and finite in every unit, which is the proof
that it actually contributes rather than being carried as a dead field.

Phase 14 owns exactly three schedule facts: the learning rate (7.5e-5 main, 3.75e-5 late), the constant
0.001 entropy coefficient, and the minibatch shuffle stream. There is no dynamic LR edit anywhere: the
rate is set once per bound iteration from that iteration's **launch segment**, and the control surface
refuses `learning_rate` by name.

The transition is tied to `current_utc - run_start_utc` against the **original** start. Downtime counts.
An iteration bound under `main` finishes its epochs under `main` even if the transition passes
mid-training — the segment travels on the cursor, not on a clock read between minibatches.

## 5. Population mixer (section 6)

Contiguous frozen ordinal subranges, in the frozen order, carried by the four store buckets the accepted
rollout store already audits:

```text
main   current 1188 | historical 615 | rule 122 (61 strategic, 61 tactical) | stress 123 (41/41/41)   = 2048
late   current  819 | historical 984 | rule 122                            | stress 123               = 2048
```

Realized: main 58.008% / 30.029% / 11.963%; late 39.990% / 48.047% / 11.963% — the frozen percentages
exactly, inside the 85–90% neural and 10–15% handcrafted bands. Only the current/historical split moves
at the transition; the handcrafted share is byte-identical in both segments. Games are **scheduled
counts, never sampled**, and no result reaches the mixer, so adaptive reweighting is not disabled — it
is unrepresentable.

## 6. Archive and active pool (sections 7–8)

The **archive** is append-only and ordered; nothing is ever pruned from it, including snapshots that are
not active opponents. The **pool** is a pure function `f(k)` of that ordering: 2 permanent anchors plus
up to 14 snapshots, banded by age alone, recomputed after every durable write and at every resume. No
tournament, no result, no strength estimate is an input.

Verified across every archive size a 168-hour run can produce (k = 0…84): membership is bounded by 16,
distinct, and equals `min(k, 14)` snapshots; above k = 14 `recent` is exactly the six highest indices and
`older`/`middle` are the four quantile picks each.

Sampling weights are 20/25/25/30 with the frozen empty-category redistribution (all three empty → the
whole historical share to the anchors, split equally). The historical bucket is **partitioned exactly**
by largest remainder over *rational* shares — floats would let two members of one category differ in the
last bit and silently stop being a tie — with ties broken by canonical member order rotated left by
`iteration mod tie_group_size`. Every partition sums to H exactly (615 and 984 checked at k = 0, 1, 3,
14, 20, 84) and the remainder demonstrably rotates across iterations.

Resume compares `f(archive)` against the checkpointed pool by digest and **refuses to continue** on a
mismatch: a run that resumed against a differently composed pool would look healthy in every metric while
training against a different opponent distribution than its own checkpoint claims.

## 7. Setup source (section 3, contract `setup_source`)

`phase14_setup_source_v1` is the accepted `Phase10BSetupSource` — the frozen P10-D selector, whose 35/65
branch coin lives inside `LearnedSetupSource` — subclassed to change exactly two things: the per-side
selector seeds descend from the Phase 14 roots, and the provenance labels name Phase 14. The accepted
adapter already binds `SelectorDraw.oriented(player)` before `create_game`.

Agent 1's integration warning is enforced mechanically, not documented: the Phase 11B glue
(`stratego/belief/phase11b/corpus.py` `Phase11BSetupSources`) is not imported, and
`assert_orientation_path()` re-derives blue's draw and refuses a source whose engine setup is the
canonical tuple rather than the oriented one. It runs at every run start and in the test suite:
`engine_is_oriented = True`, `canonical_differs_from_oriented = True`.

## 8. Checkpoint hierarchy (section 9)

| cadence | what | where |
|---|---|---|
| 15 min | hot resume, ≥ 4 valid retained | fast internal disk (`checkpoints/phase14/hot`) |
| 2 h | durable archive snapshot, append-only | external volume (`/Volumes/Brandon_Washington/stratego_phase14/archive`) |
| 6 h | candidate **mark** on an archive snapshot (hours 0…168, 29 marks) | beside the archive |

All three carry the same complete payload, covering every field on the frozen resume list — weights,
optimizer state, the explicit statement that **no EMA exists**, the optimizer step, RNG stream state,
population schedule state, the pool with its categories, the archive cursor, the shard cursor, storage
state, the original start and deadline, the main/late schedule state and candidate-evaluation state. A
payload missing any of them is refused at write time.

The ring writes atomically, **reads the file back and validates it, and only then prunes**; a resume
takes the newest file that *validates*, not the newest file. A torn write therefore costs one cadence,
which the test proves by corrupting the newest file and recovering the one behind it.

A candidate is a *mark*, never a copy: the candidate is the archive snapshot (hour 0 is the accepted
starting checkpoint itself). Copying would double the archive footprint and create a second set of bytes
to keep consistent.

## 9. Candidate evaluation (section 10)

`stratego/evaluation/phase14_candidates.py` plays the frozen 128-game pack with direct greedy policies
through the accepted `play_match` / `RemoteNeuralPolicy`, pinning the pack's own per-game
`opponent_decision_seed` and `candidate_decision_seed` with the accepted `FrozenSeedPolicy` wrapper.
`load_pack()` recomputes `pack_content_digest` from the document at every use, so "the same pack" is
checked rather than assumed.

Isolation is structural: the module imports no trainer, collector, scheduler or clock, and no function it
exposes returns a value the training loop consults. Evaluation is **out of band by default** — 128 games
inside the loop would spend deadline time on monitoring — and the runner exposes
`evaluate_pending_candidates()` for an operator or for the post-run selection step. Every failure is
caught, recorded in the ledger as `failed` with its reason and `rerunnable: true`, and dropped: the
candidate bytes are preserved and re-evaluate later on the identical pack.

The selection rule is implemented exactly (mean EWR → min-stratum EWR → later hour) and **refuses
incomplete evaluations** rather than comparing a 4-game pass with a 128-game one.

## 10. Deadline controller (section 11)

```text
run_start_utc     stamped immediately before the loop, after input verification
run_deadline_utc  run_start_utc + 604800 s, computed once and persisted in every hot checkpoint
transition_utc    run_start_utc + 475200 s, likewise
```

`RunWindow` refuses to exist if the span is not exactly 168 h with the transition at exactly 132 h, and
`resume` rebuilds it from the persisted values — there is no code path that derives a new deadline on
restart. At or after the deadline: no new collection unit is launched, no optimizer step may begin (the
gate is consulted *before* each step, so a step already begun completes and lands), the final state is
written, the hour-168 candidate is marked, the run manifest is written and the run is closed. A recovery
that starts after the deadline finalizes immediately without any optimizer update.

## 11. The two test seams (section 12)

A short test cannot reach the 2-hour archive, the 6-hour candidate, the 132-hour transition or the
168-hour stop — and it cannot collect 2,048 games per unit either. Both gaps are closed by **declared
seams that production refuses**, rather than by leaving that code untested:

- **`ManualClock`** carries `production = False`; `require_production_clock()` refuses it, and the runner
  in `production` mode calls that on construction.
- **`Population.scaled(divisor)`** carries `production = False`; the runner in `production` mode refuses
  it and runs the frozen 2,048-game mixture.

Neither seam changes any production semantic: the logic under both clocks is one implementation, and the
scaled population keeps the frozen *shape* (four buckets, all five handcrafted families, the same colour
balance and ordinal layout) while changing only the size.

## 12. Storage and retention (section 13)

Full raw retention is the plan, as frozen. The reserve is monitored every committed iteration; rolling
deletion is pre-authorized **only** below 120 GiB free. Two structural guards make the frozen no-deletion
rule enforceable rather than remembered:

- `assert_not_project_evidence()` refuses any path that is not a `.stgshard` under
  `<external>/rollouts/phase14/…` — the accepted Phase 9 checkpoint, the archive and even a sibling
  `.meta.jsonl` all raise;
- a shard is only ever a deletion candidate if its iteration carries a `consumed / disposable /
  safe_to_delete` mark, written when the iteration reaches COMMITTED.

`plan_rolling_deletion()` retains a 1-in-16 representative sample of shard ranges and refuses to plan
anything while the reserve is intact; `execute_rolling_deletion()` re-verifies every path before removing
it.

## 13. Telemetry and controls (section 14)

Every one of the 23 frozen metrics has an explicit path in the snapshot, and `missing_metrics()` compares
a real snapshot against the frozen list — a metric that quietly stopped being emitted after a refactor
shows up as a finding instead of as a gap somebody discovers at hour 140. Every unit of the integration
run reported `missing_metrics == []`.

The control surface exposes **emergency stop and nothing else**. `set()` refuses each frozen key
(`learning_rate`, `loss_weights`, `opponent_mixture`, `setup_source`, `historical_pool_algorithm`,
`candidate_selection_rule`, `deadline`, `checkpoint_cadence`) by name with an explanation, and refuses
every other key as unwritable. Emergency stop is a *request*: the unit or step in flight finishes, a hot
checkpoint is written, and the loop exits at a safe boundary.

## 14. Failure handling (section 15)

Recoverable (`OSError`, `TimeoutError` — worker crash, transient MPS fault, torn shard write, reboot):
counted, hot-checkpointed, retried, with a consecutive-failure ceiling. A recoverable restart loads the
newest valid hot checkpoint and restores model, optimizer, KL controller, counters, RNG state, cursor,
pool, archive, shard cursor and the original window, continuing only if before the deadline.

Unrecoverable — no valid checkpoint, wrong starting model, checkpoint/pool inconsistency, configuration
digest mismatch, irrecoverable optimizer state, a breached KL or clip-fraction veto — stops the run with
`Phase14IntegrityError`. **No path starts a fresh logical run**: resuming into an empty hot directory is
a refusal, and starting over an existing valid checkpoint is a refusal too.

Two crash points inside a unit get explicit handling, because getting them wrong is silent rather than
loud:

- **Crash during collection.** The store reconciles, only the missing games are regenerated, and the
  segment and pool recorded in the iteration's state document are what the regeneration runs under — not
  the segment and pool the resume moment implies. The collector refuses to finish an iteration under a
  different behavior checkpoint, device, batch shape, segment or pool digest than it started with.
- **Crash during training.** The games are already sealed, so nothing is re-played: the rollout is bound
  with `resuming`, the *checkpointed cursor* decides where the epochs continue, and the on-policy digest
  check is deliberately not re-applied — the weights moved precisely because some of this iteration's
  updates already landed. Measured in the integration run: interrupted at step 21 with one update of
  iteration 4 done, resumed, finished at step 24, **0 games replayed**.

The narrow window where the store records COMMITTED but the checkpoint that would prove it was never
written is handled by advancing past that iteration rather than replaying it: those updates are gone with
the weights that made them, and PPO may not re-consume that rollout with drifted weights.

## 15. Short integration test results (section 16)

One scripted run, `Population.scaled(512)` (8 games/unit main, 7 late), CPU, single loader worker,
manual clock: start, three units across the transition, a clean resume, an interrupted unit resumed from
its cursor, and the deadline. 50.5 s for the whole walk.

| unit | segment | games | buckets | updates | LR | policy | value | belief | grad norm | filter | archive | candidate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | main | 8 | 2/1/2/3 | 6 | 7.5e-5 | 0.1160 | 0.5757 | 1.9571 | 1.000 | 0.145 | — | — |
| 2 | main | 8 | 2/1/2/3 | 8 | 7.5e-5 | 0.3030 | 0.5802 | 1.8247 | 1.000 | 0.254 | S0001 (mark 3) | hour 6 |
| 3 | late | 7 | 1/1/2/3 | 6 | 3.75e-5 | −0.0457 | 0.5171 | 2.0789 | 1.000 | 0.344 | S0002 (mark 66) | hour 132 |

Every section-16 item, and how it was shown:

```text
model updates occur                 20 optimizer steps; model-state digest != the accepted starting digest
losses are finite                   policy/value/belief/KL/grad-norm finite in every unit
belief auxiliary runs               loss_belief non-zero and finite in all three units (weight 0.25)
population mixer works              bucket counts equal the scheduled mixture in both segments
Phase 14 setup source works         orientation probe passes; provenance validates; draws are pure
hot checkpoint writes/loads         7 writes, ring keeps 4 valid, newest validates and reloads
archive checkpoint writes           S0001, S0002 during the run, S0003 at finalization
candidate marking works             hours 0, 6, 132, 168 marked; hour 168 preserved at close
candidate evaluator can run         4-game slice, one per stratum, 1.8 s, direct policies only
deadline controller works           new unit refused with reason "deadline" at 168.2 h elapsed
test-clock late transition works    LR and mixture switch at exactly 132 h elapsed
pool evolves deterministically      f(archive) recomputed == checkpointed pool digest
resume restores exact logical state before == after on step, iteration, pool, archive, window, model digest, beta, examples
crash mid-training resumes cleanly  interrupted at 1 update, resumed from the cursor, 0 games replayed, iteration completed
storage/telemetry paths work        telemetry JSONL + run manifest written; disposable marks written
search is absent from training      import-graph walk: no stratego.search module loaded
```

Two behaviours worth recording rather than discovering later:

- **KL beta damps to its floor early.** The controller took beta from 0.005 to the 1e-4 clamp over the
  run's epochs, because a learner playing snapshots of itself starts near-on-policy and every epoch KL sat
  below the 0.0075 decrease threshold. That is the accepted adaptive controller behaving exactly as
  specified; it is not a Phase 14 change.
- **Coalesced marks.** If several 2-hour or 6-hour marks pass while the machine is down or one long unit
  is in flight, they name no distinct weights, so they coalesce into the next snapshot/mark and the skipped
  indices are recorded on the artifact (`coalesced_marks`, `coalesced_hours`) rather than left as a gap.

## 16. Identity binding (section 17)

```text
phase14 contract digest             62ce6d4e04ffd25755717ef290f7486f2616927ddada59d8ea9fb05565c052b9
phase14 population digest           1281477150310bd5765a5635467758f48b6188f135febd32fd79d0b4107263ff
phase14 seed contract digest        3a19fa9bb10ad9f82d8e403889abe8f06f70388560174f15bf41ac35063505f8
phase13_integrated_training_config_v1
  integrated_config_digest          9c2a38e4335762997adbb33731dc619615fff713c2c60840c7c8d74a2f29da5e
```

The integrated config binds the starting checkpoint and its digests, the whole inherited objective, both
LRs, the transition, both mixtures, the pool algorithm with anchors and weights, the setup-source
identity and selector config digest, all three cadences and the candidate hours, the pack digest and
selection rule, the storage policy, the deadline semantics, the frozen contract's own file hash and every
Phase 14 module version. It deliberately excludes operational choices — device, loader workers,
games-in-flight, storage root, clock — because those change what a run costs, not what it is. **This
digest is the input Agent 4's immutable launch manifest binds.**

## 17. History preservation (section 2)

Nothing accepted was modified. Every file added is new and Phase 13/14-scoped; the accepted Phase 9
collector, behavior module, loss, targets, trainer internals, rollout store and checkpoint modules are
imported and subclassed, never edited. Phase 11 remains `FAIL` and unreinterpreted; Phase 11B remains an
engineering branch (Agent 1C is bound in every checkpoint as an explicit *non*-parent); Phase 12 search
stays outside Phase 14 entirely. Nothing was written to `/Volumes/Brandon_Washington` or to
`checkpoints/phase14`: the integration run lives in a temporary directory and is deleted.

## 18. Deliverables

```text
stratego/training/phase14_contract.py
stratego/training/phase14_seed.py
stratego/training/phase14_clock.py
stratego/training/phase14_pool.py
stratego/training/phase14_schedule.py
stratego/training/phase14_setup_source.py
stratego/training/phase14_collector.py
stratego/training/phase14_trainer.py
stratego/training/phase14_checkpoint.py
stratego/training/phase14_storage.py
stratego/training/phase14_telemetry.py
stratego/training/phase14_runner.py
stratego/training/phase14_config.py
stratego/evaluation/phase14_candidates.py
scripts/run_phase13_agent02.py
tests/training/test_phase13_agent02.py
reports/phase13/phase13_integrated_training_config_v1.json
reports/phase13/phase13_agent_02_summary.json
reports/phase13/phase13_agent_02_report.md
```

Tests: `tests/training/test_phase13_agent02.py` — **61 passed in 70.8 s**. Full suite —
**6,223 passed / 3 skipped in 423 s** (was 6,162/3 after Phase 13 Agent 1; +61).

`git diff --stat HEAD` over tracked files is **empty**: every file above is a new, untracked addition, so
no accepted Phase 9/10/11/11B/12 or Phase 13 Agent 1 artifact was touched.

## 19. Stop condition

```text
frozen Agent 1 contract implemented          yes (0 disagreements, checked mechanically)
short integration tests pass                 yes (60/60)
production search absent from training       yes (import-graph walk)
test-clock scheduler proved long horizons    yes (2 h, 6 h, 132 h, 168 h, restart)
90-minute rehearsal begun                    no
strength tournament run                      no
Phase 14 started                             no
Agent 3 started                              no
```

## 20. Handoff notes for Agent 3 / Agent 4

- The 90-minute rehearsal should run `Phase14Runner` in **production mode** (`SystemClock`,
  `PRODUCTION_POPULATION`) against a scratch storage root, not the production one, and will not reach any
  cadence beyond the 15-minute hot checkpoint — that is expected and is why the seams exist.
- Production throughput is unmeasured by this agent. The integration run used CPU, one loader worker and
  8 games in flight; the rehearsal is where `device`, `LoaderTopology.workers` and `games_in_flight` get
  chosen, and none of them enters the config digest.
- The candidate evaluator is out-of-band by default. Whoever operates the run decides when to call
  `evaluate_pending_candidates()`; the hour-168 selection step completes any missing evaluations on the
  identical pack, outside training time, before applying the frozen rule.
- `checkpoints/phase12/phase9_c1_readonly_copy.pt` remains an untracked Phase 12 re-export with different
  bytes from the original; Phase 14 binds the original path and SHA-256, as Agent 1 noted.
