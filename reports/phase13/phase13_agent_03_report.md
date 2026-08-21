# Phase 13 — Agent 3: the 90-Minute Crash/Resume Rehearsal

Run date: 2026-08-21. Task:
`instructions/phase_13_final_training_integration/03_AGENT_3_90_MINUTE_CRASH_RESUME_REHEARSAL.md`.

One deliberate 90-minute rehearsal of the exact Phase 14 training system was run, on the real wall
clock, at the **frozen 2,048-game production population**, on the real external volume. It survived a
forced process kill and a CPU worker kill and stopped itself at its original deadline. **Sixteen of
sixteen readiness checks pass.** One reliability defect was found — a killed CPU loader worker used to
kill the learner — and a narrow fix was applied and re-verified afterwards. No training value was
changed, and none may be changed on this evidence.

## 1. What this task did and did not do

Did: verified the six frozen identities live; built a rehearsal window seam that production refuses;
ran one 90-minute rehearsal under supervision; force-killed the training process group at 30 minutes and
restarted it through the production recovery path; killed a CPU loader worker at 55 minutes; let the run
reach its own deadline; measured storage; exercised the 2-hour, 6-hour, 132-hour and 168-hour events on
the declared clock seam; proved a post-deadline recovery refuses to train; and fixed the one defect the
rehearsal exposed.

Did not: start Phase 14; start Agent 4; change any learning rate, loss weight, opponent mixture, pool
algorithm, setup source, selection rule or cadence; use search; run a strength experiment; or touch any
accepted Phase 8/9/10/11/11B/12 artifact. Phase 9's `phase9_trainer.py` was **not** modified — the fix
lives entirely in Phase 14 code.

## 2. Prerequisites (section 1)

Verified live this session, recomputed rather than quoted
(`reports/phase13/agent03_evidence/stage_prerequisites.json`, `verified: true`, `problems: []`):

```text
Agent 1 contract         reports/phase13/phase13_final_training_contract_v1.json
                         sha256 65d1f941a326a1343dce597082c3b525203ef7182f73c759ac6eb04d87a12cdf
                         assert_matches_frozen_contract() -> 0 disagreements
Agent 2 integrated config integrated_config_digest 9c2a38e4335762997adbb33731dc619615fff713c2c60840c7c8d74a2f29da5e
                         recomputed == on disk; document == on-disk body
                         phase14 contract digest 62ce6d4e04ffd25755717ef290f7486f2616927ddada59d8ea9fb05565c052b9
Phase 9 start            checkpoints/phase9/selfplay_c1_v1.pt  dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
                         model state f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
Phase 14 setup source    phase14_setup_source_v1, selector config 6e227815bc3cb44f19cdeee55d00ec0ae75726fb411ee9131660aa712bb86668
Candidate pack           896a753b3d568902e93e803f1a45de9e8834ff1cdf90bc08cfacf90bcf0c2bde, 128 games
Selection rule           phase14_checkpoint_selection_rule_v1, bound to that pack digest
Pool anchors             P8 f7e9c40d..., P9 dfd698e5...  (both re-hashed, both match)
```

Both digests were re-checked **after** the post-rehearsal fix and are unchanged, so Agent 4's launch
manifest identity is untouched by anything done here.

## 3. The rehearsal, and its single deviation (sections 2–3)

```text
run_start_utc      2026-08-21T03:06:12.034Z
run_deadline_utc   2026-08-21T04:36:12.034Z     start + 5,400 s
transition_utc     2026-08-26T15:06:12.034Z     start + 475,200 s — the frozen 132 h
population         the frozen 2,048-game mixture, main segment (production = True)
learning rate      7.5e-05 throughout            the frozen main continuation rate
device / workers   mps, 6 CPU loader workers, 96 games in flight
storage            /Volumes/Brandon_Washington/stratego_phase13_rehearsal (external)
                   checkpoints/phase13_rehearsal/hot (fast internal disk)
```

The **only** value replaced is the deadline. `MODE_REHEARSAL` stamps a `RunWindow.rehearsal` window
against the real `SystemClock`; the 132-hour transition is left exactly where it is, so the rehearsal
spends all 90 of its minutes in `main` and never rehearses a schedule the real run will not follow.
That is deliberate — section 3 forbids moving the transition earlier, and section 9 verifies it
separately on the clock seam instead.

The seam is built the way Agent 2 built the other two. The shortened window carries `production =
False`, that flag rides in every checkpoint, and the runner **refuses** to resume a rehearsal window as
a production run or a production window as a rehearsal, and refuses a restart that declares a different
rehearsal deadline. `MODE_REHEARSAL` also calls `require_production_clock`: a 90-minute rehearsal whose
90 minutes were simulated would be worthless, so the manual clock is refused there too. Production
still refuses the manual clock, the scaled population and any shortened window.

The rehearsal was run under a supervisor (`stratego/training/phase13_rehearsal.py`) that observes the
run **only through what it writes to disk** — hot checkpoints, the telemetry JSONL, the run manifest,
the rollout store. Nothing recorded below was read out of a live Python object, because nothing read
that way would survive a `SIGKILL`, and because the property under test is precisely whether the
*persisted* state is enough.

Storage was pointed at a sibling directory on the same physical volume rather than at the frozen
`stratego_phase14` path: the measurements stay real, and the production directory stays empty for
Agent 4 — a rehearsal checkpoint sitting there would make the real run's `start()` refuse.

## 4. What happened (section 4)

Three launches, two injected failures, one natural stop. Wall clock 5,416 s.

```text
     0.0 s   launch 1 (pid 18401)  start: window stamped, hour-0 candidate marked
   304.2 s   iteration 1 SEALED    2,048 games collected
  1281.8 s   iteration 1 COMMITTED step 1744
  1594.0 s   iteration 2 SEALED
  1800.5 s   FAILURE 1 — SIGKILL of the whole process group (learner + 6 workers)
  1800.8 s   launch 2 (pid 43228)  resumed from hot_000003_step000001744.pt
  1806.0 s   iteration 2 TRAINING (again)
  2765.0 s   iteration 2 COMMITTED step 3480
  3062.7 s   iteration 3 SEALED
  3303.0 s   FAILURE 2 — SIGKILL of one CPU loader worker (60779 of six)
  3303.0 s   launch 2 DIED, rc=1, BrokenProcessPool          <-- the defect
  3308.3 s   launch 3 (pid 64076)  resumed from hot_000006_step000003480.pt
  3313.3 s   iteration 3 TRAINING (again)
  4217.2 s   iteration 3 COMMITTED step 5168
  4520.6 s   iteration 4 SEALED
  5400.0 s   DEADLINE — no optimizer step may begin
  5415.5 s   launch 3 exited rc=0 after finalizing
```

Segment A behaved as required: optimizer steps increased, parameters moved, every loss stayed finite,
workers generated games, hot checkpoints appeared and telemetry updated.

**Failure 1 — forced process termination.** The whole process group was killed with `SIGKILL`, taking
the six loader workers with it; `returncode -9`, **zero orphaned workers**. No checkpoint was created
beforehand: the newest valid one was the ordinary end-of-iteration write from 518 s earlier. The
restart went through the production path (`start_or_resume()` → `resume()`).

**Failure 2 — worker failure.** One `spawn_main` loader worker was killed while the learner trained
(the pool's `resource_tracker` child is filtered out by command line, so this is a real worker and not
bookkeeping). The victim died. **So did the learner**, with `BrokenProcessPool` out of
`future.result()`. See section 12.

**Segment C — natural deadline.** The run stopped itself at its original deadline despite 15.8 s of
downtime and two restarts, marked the hour-168 candidate on the final archive snapshot, wrote the
manifest and exited 0.

Per-iteration record (`phase14_telemetry.jsonl`, every row `missing_metrics: []`):

| iter | elapsed h | step | policy | value | belief | grad norm | LR | adv. retention |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.356 | 1744 | −0.0193 | 0.6458 | 1.8301 | 1.000 | 7.5e-05 | 0.291 |
| 2 | 0.768 | 3480 | −0.1359 | 0.6605 | 1.8354 | 1.000 | 7.5e-05 | 0.138 |
| 3 | 1.172 | 5168 | −0.0038 | 0.6727 | 1.8718 | 1.000 | 7.5e-05 | 0.217 |
| 4 | 1.502 | 6712 | +0.0926 | 0.6594 | 1.8389 | 1.000 | 7.5e-05 | 0.252 |

Four collections of exactly 2,048 games each; iteration 4's epochs were cut by the deadline
(`deadline_stops: 1`), which is why it is not COMMITTED. The belief auxiliary is non-zero and finite in
every iteration at the contracted weight 0.25 — it contributes rather than riding along as a dead
field. Every non-finite counter is 0; there were no KL or clip-fraction breaches.

**What the two failures cost.** Both crashes landed after their iteration's games were sealed, so no
game was ever regenerated. The cost is un-checkpointed optimizer work, bounded by the 15-minute hot
cadence:

```text
Failure 1   197.6 s of iteration 2's epochs redone,  5.5 s downtime,  0 games replayed
Failure 2   232.4 s of iteration 3's epochs redone, 10.3 s downtime,  0 games replayed
            ~7.4 min of the 90 (8.3 %), all of it charged against the deadline
```

The rollout store's own audit is the direct evidence that nothing was corrupted or double-counted:
every one of the four iterations records **committed 2048, decoded 2048, missing 0, duplicates 0**,
under one unchanging active-pool digest.

## 5. Resume boundary checks (section 5)

Each boundary compares the last full state the supervisor observed before a kill with the first *new*
checkpoint the restarted process wrote — not the same file read twice.

| | Failure 1 | Failure 2 |
|---|---|---|
| before | `hot_000003_step000001744.pt`, step 1744, iter 1 | `hot_000006_step000003480.pt`, step 3480, iter 2 |
| after | `hot_000004_step000001745.pt`, step **1745**, iter 1 | `hot_000007_step000004013.pt`, step 4013, iter 2 |

All nine checks hold at both boundaries: no duplicate optimizer work, no skipped logical optimizer
state, same active historical pool digest, archive cursor not reset, candidate scheduler not reset,
main/late scheduler not reset, start time not reset, deadline not reset, shard cursor not reset. The
step sequence observed across the whole rehearsal is monotonic and never restarts:

```text
0 -> 1049 -> 1744 -> 1745 -> 3359 -> 3480 -> 4013 -> 5168 -> 6712
```

**Bitwise continuity, and where it does not apply.** Model, optimizer, KL controller, counters, cursor,
pool, archive, shard cursor and the window are all restored exactly. **EMA is absent by contract** and
the checkpoint says so explicitly rather than omitting it (`ema_state.present = False`).

**RNG is the one field that is not restored, and that is correct.** Phase 14 captures every RNG stream
in every checkpoint and restores none of them. That is sound only if no global RNG cursor decides a
batch or a move — a claim about the code, which this rehearsal checked rather than repeated. The same
iteration was collected and trained twice, in two directories, under two deliberately different global
`torch` seeds. Identical: games, total decisions, terminal results, bucket counts, updates, examples
consumed, the entire epoch plan, and **the final model state digest**
(`f27ea740c4f13c7e5c8576adb1a789c84061cc6777ba7661d90f311cfa9cdf60` both times). The only field that
moved anywhere was `behavior_checkpoint_sha256`, because the behavior snapshot *file* embeds live
elapsed-time and free-space provenance; the trajectories it describes are the same. A fresh global RNG
therefore cannot change the logical run. (`stage_determinism.json`.)

## 6. Hot checkpoint verification (section 6)

```text
max hot-checkpoint age observed   895.4 s      against the frozen 900 s cadence
samples above 900 s               0            out of 1,067 supervisor samples
checkpoints written               12           4 retained, exactly HOT_CHECKPOINT_RETAIN
newest valid selected on recovery yes          both resumes took the newest file that validates
required resume fields            14 of 14 present in every retained checkpoint
readable after the rehearsal      4 of 4 hot + 1 of 1 archive snapshot, all re-read and re-validated
```

The cadence held across both crashes. Corrupt files are refused rather than selected: truncating the
newest checkpoint to half its length made the ring fall back to the one behind it and recover step
6712 — a torn write costs one cadence, not the run.

One honest limitation: no hot checkpoint is written *during* a collection, which ran 297–311 s here.
The exposure therefore stays inside the 15-minute bound, but an operator should read "checkpoint age"
as up to one cadence plus a collection, not up to one cadence.

## 7. Storage verification (section 7)

Measured over the rehearsal's 1.497 h on the real external volume:

```text
raw shards                0.2024 GiB/hour        (4 x ~78 MiB per 2,048-game iteration)
archive snapshot          10,457,515 B each      84 in a 168 h run
logs                      7.2e-06 GiB/hour
hot checkpoints           62.7 MiB, bounded      internal disk; ring of 4 + 2 behavior snapshots
external free-space change  -338,800,640 B over the rehearsal

projected 168 h raw       33.996 GiB
projected 168 h archive    0.818 GiB
projected 168 h total     34.815 GiB
external free now        925.33 GiB  ->  890.52 GiB after 168 h
reserve                  120 GiB, NOT threatened
```

Against Agent 1: the frozen projection used a **3.572 GiB/hour** planning rate and kept 600 GiB as a
conservative ceiling, alongside a Phase 9-derived basis of ~28 GiB. The measured rate is **0.2024
GiB/hour — 17.6× below the planning rate**, and the 34.8 GiB total sits just above the Phase 9 basis
and 17× under the ceiling. Full raw retention is comfortable; the rolling-deletion contingency is not
approached. **This is not a launch blocker.** No earlier accepted evidence was deleted; the rehearsal
wrote only under its own directory.

Hot checkpoints do not grow — the ring is bounded — so they are reported and excluded from the
projection rather than extrapolated as a rate.

## 8. Candidate checkpoint / evaluator plumbing (section 8)

On the Agent 2 clock seam, outside the production rehearsal. **No production cadence was changed.**

```text
2-hour archive event    fires at 2.05 h; archive k 0 -> 1; pool recomputed and equal to f(archive)
6-hour candidate event  fires at hour 6
candidate is a MARK     snapshot_path and sha256 equal the archive entry's — not a copy
fixed pack launches     pack digest 896a753b... matches the frozen pack
evaluation != training  trainer step, model digest, LR, contract digest and window byte-identical
                        before and after evaluation
failure is absorbed     a candidate with missing bytes records status "failed", rerunnable: true,
                        training step unchanged, and the next iteration trained normally
result cannot steer     the control surface refuses learning_rate, opponent_mixture,
                        candidate_selection_rule, deadline, checkpoint_cadence and mean_ewr by name
selection rule          refuses an incomplete evaluation (Phase14CandidateError) rather than
                        ranking a 2-game slice against a 128-game pack
hour-168 logic exists   finalize marks hour 168 on the final archive snapshot
```

No new candidate-strength experiment was run. The two evaluations performed were 2-game slices used
only to prove the evaluator executes; both are marked incomplete and the selection rule refuses them.

## 9. Late-phase transition plumbing (section 9)

```text
131.85 h   segment main, LR 7.5e-05
132.10 h   segment late, LR 3.75e-05
unit run under late      buckets current 1 / historical 1 / rule 2 / stress 3 == the late shape
transition_utc           equal to the value stamped at start — tied to the original wall clock
10 h of downtime         window unchanged, transition_utc unchanged, elapsed 142.1 h
restart across it        resumed run reports segment late and LR 3.75e-05
```

Downtime does not postpone the transition, and a restart preserves which segment should be active.

## 10. Deadline tests (section 10)

**Natural deadline.** At the original +90 minutes the run launched no new collection unit, began no
optimizer step (`deadline_stops: 1`, iteration 4 left at TRAINING with 0 further updates), wrote its
final state, marked the hour-168 candidate on the final archive snapshot, wrote the manifest and closed
with `closed_reason: deadline`.

**Recovery started after the deadline.** A real rehearsal checkpoint was loaded with the test clock set
**9.0 hours past** its persisted deadline:

```text
resume reports past_deadline: true, remaining_hours: -9.0
optimizer steps taken       0
units launched              none
finalized                   yes, immediately
deadline extended           no — window byte-identical before and after
```

## 11. Monitoring and controls (section 11)

The live status assembled from disk covers elapsed, remaining, deadline, optimizer step, current LR,
schedule segment, opponent mixture, active historical pool, worker status, checkpoint age, disk usage,
losses and failure/recovery events. All 23 frozen metrics were present in every telemetry row
(`missing_metrics: []` in 4 of 4). Emergency stop works against a real runner: `run()` stops with
`stopped_because: "emergency stop"`, a further `run_iteration()` refuses with reason
`emergency_stop`, a hot checkpoint is written, and clearing the request restores `should_continue()`.
Every frozen key is refused by name; the surface exposes no other mutable control.

Three monitoring observations, none of them a training-correctness problem, all of them things an
operator would want to know before hour 140. They are recorded rather than fixed, because fixing them
would change telemetry the frozen contract names and that is not this task's call:

1. **`games generated` under-reports after a crash.** The counter is restored from the last checkpoint,
   so a collection that completed but was never checkpointed before a crash is lost from it. The
   rehearsal committed **8,192 games** (4 × 2,048, confirmed in the store) but the counter reports
   **4,096**. The rollout store's per-iteration manifests are correct and authoritative; only the
   telemetry counter is low.
2. **`worker status` is not a health signal.** It reports the configured worker count and a static
   string (`{"loader_workers": 6, "status": "single-process bulk-synchronous loop"}`). It would not
   have shown the dead worker in Failure 2.
3. **The run's own telemetry cannot tell you it crashed.** `failures` stayed `{}` through two process
   deaths, because a killed process cannot record its own death and the resumed process records only
   failures it caught itself. The restart is visible in the manifest's `started_from` and in the
   elapsed/step discontinuity, but a watcher reading only the telemetry stream would see an unbroken
   run. Agent 4's launch supervisor should log restarts itself.

Externally observable state is only as fresh as the last hot checkpoint — up to 15 minutes — because
that is the cadence at which the run publishes. That is a property of the design, not a defect.

## 12. Defect found, and the narrow fix (sections 12, 15)

**Defect. A killed CPU loader worker killed the learner.** `ProcessPoolExecutor` marks itself
permanently broken when a worker dies and raises `BrokenProcessPool` — which subclasses `RuntimeError`,
and so was matched by neither `RECOVERABLE_ERRORS` (`OSError`, `TimeoutError`) nor
`UNRECOVERABLE_ERRORS`. It escaped `Phase14Runner.run()` and ended the process. Production runs six
loader workers by default for 168 unattended hours, so a single worker fault would have ended the real
run unless something outside it noticed.

The defect was reproduced directly before the rehearsal
(`agent03_evidence/worker_kill_probe_before_fix.log`) and then demonstrated end to end **inside** the
rehearsal at production scale, at 3,303 s. It was deliberately left unfixed for the rehearsal: section
12 frames fixes as post-rehearsal, and the rehearsal is meant to test the accepted Agent 2
implementation rather than a version cleaned up in anticipation.

**Fix**, entirely within Phase 14 code — `phase9_trainer.py` is untouched:

- `Phase14Trainer._next_minibatch` catches `BrokenExecutor`, shuts the dead pool down and rebuilds it
  **at the same cursor**. The minibatch plan is a pure function of the cursor, so the identical
  minibatch is rebuilt; the optimizer step, the epoch, the KL controller and the examples consumed are
  untouched, and only the processes that packed the bytes differ. Rebuilds are counted
  (`counters.loader_pool_rebuilds`) and capped at `MAX_LOADER_POOL_REBUILDS = 16`, so a sick host stops
  the run instead of hiding behind a healthy-looking log.
- `RECOVERABLE_ERRORS` gains `BrokenExecutor` as a backstop for a pool that breaks where the rebuild
  does not reach — costing the unit, not the run.

Neither digest moved: `contract_digest` is still `62ce6d4e...` and `integrated_config_digest` still
`9c2a38e4...`.

**Re-verification.** The identical probe that failed before now completes the iteration (216 updates,
sealed and trained): `agent03_evidence/worker_kill_probe_after_fix.log`. `tests/training/
test_phase13_agent03.py` adds 21 tests, including a worker killed mid-epoch from the trainer's own
per-step callback: the learner survives, exactly one rebuild is counted, and the `(epoch,
minibatch_index)` sequence and per-step losses are **identical** to a clean run of the same
iteration — no minibatch repeated, none skipped.

The fix changes the end-to-end worker-recovery path, so per section 12 that verification was re-run at
the **frozen 2,048-game production population on the real clock**, and nothing else was:

```text
423.4 s   one of six loader workers (8473) SIGKILLed mid-epoch; the victim died
          learner survived
          a fresh pool of six workers appeared in its place (10209..10214)
          loader_pool_rebuilds = 1
          iteration 1 completed with 1,744 updates
          no restart was needed: one launch, started_from "start", exit 0
```

**1,744 updates is exactly the count of the rehearsal's own uninterrupted iteration 1.** The rebuilt
pool consumed the same epoch plan at production scale — the strongest available evidence that recovery
changes nothing but the processes doing the packing. Evidence:
`agent03_evidence/stage_worker_reverify.json` and `agent03_evidence/worker_reverify/`.

**Tests.**

```text
tests/training/test_phase13_agent03.py     21 passed in 44.9 s
pytest tests (full suite)                6,244 passed, 3 skipped in 458.5 s
```

Agent 2's accepted baseline was 6,223 passed / 3 skipped; the difference is exactly the 21 tests added
here, so nothing regressed and nothing was lost. `git diff --stat HEAD` touches one tracked file,
`.gitignore` (four lines, so the rehearsal's hot checkpoints are not committed). Every other change is
an addition to the already-untracked Phase 13/14 set.

## 13. Nothing here changes training strategy (section 12)

No rehearsal EWR, short-term loss or game outcome was used to alter anything, and nothing was altered.
The learning rates, loss weights, opponent mixtures, historical-pool algorithm, setup source, candidate
selection rule and checkpoint cadences are exactly Agent 1's frozen values, mechanically re-checked at
every run start. Search is absent from the training import graph — verified again in a fresh process:
importing the runner, trainer, collector, setup source and rehearsal harness loads **no**
`stratego.search` module.

The rehearsal's losses are reported above because section 4 asks whether updates are finite, not
because they are evidence about strength. Four iterations at a 168-hour learning rate say nothing about
strength, and no one should read them that way.

## 14. Readiness (section 13)

**16 of 16 pass** (`stage_readiness.json`).

| check | result |
|---|---|
| training updates finite | PASS — 4/4 rows finite; all non-finite counters 0 |
| parameters change | PASS — final model digest `f5534a9d...` ≠ start `f1df694d...`, 6,712 steps |
| belief auxiliary functioning | PASS — non-zero and finite in every iteration at weight 0.25 |
| forced process crash recovered | PASS — group SIGKILL, 0 orphans, resumed from newest valid |
| optimizer state preserved | PASS — 1744 → 1744, optimizer state restored, EMA absent by contract |
| original rehearsal deadline preserved | PASS — one window across 3 launches and every checkpoint |
| active historical pool preserved | PASS — digest equal to recomputed `f(k)` at both boundaries |
| worker failure recovered | PASS — **after the narrow fix**; it did not survive during the rehearsal |

| storage remained safe | PASS — 34.8 GiB projected, reserve untouched |
| hot checkpoints readable | PASS — 4/4 hot + archive re-read; torn newest refused |
| test-clock 2 h archive event | PASS |
| test-clock 6 h candidate event | PASS — marked from the archive, frozen pack |
| test-clock late transition | PASS — segment, LR and mixture all switch at 132 h |
| test-clock 168 h shutdown | PASS — no unit launched, hour-168 candidate marked, window never extended |
| post-deadline recovery refuses training | PASS — 0 steps, immediate finalization, deadline not extended |
| search absent from training | PASS — no `stratego.search` module in the training import graph |

"Worker failure recovered" is the one line that needs its history stated plainly: **during the
rehearsal it failed**, the learner died, and the row passes only because of the fix applied and
re-verified afterwards.

## 15. Rehearsal identity

```text
phase13_rehearsal_v1
rehearsal_digest   d8ebae4e28500c27cad8e7c5c48932431e89c45c6d52b8c87da2bb1443a13d21
phase14 contract   62ce6d4e04ffd25755717ef290f7486f2616927ddada59d8ea9fb05565c052b9   (unchanged)
integrated config  9c2a38e4335762997adbb33731dc619615fff713c2c60840c7c8d74a2f29da5e   (unchanged)
```

## 16. Deliverables

```text
reports/phase13/phase13_rehearsal_v1.json           the rehearsal's identity and result
reports/phase13/phase13_agent_03_report.md          this document
reports/phase13/phase13_agent_03_summary.json       machine-readable summary
reports/phase13/agent03_evidence/
  stage_prerequisites.json    the six identities
  stage_rehearsal.json        plan, launches, marks, final state, manifest, 178 storage samples
  stage_scheduler.json        2 h / 6 h / transition / 168 h / evaluator / emergency stop
  stage_postdeadline.json     recovery 9 h past the deadline
  stage_readability.json      every checkpoint re-read; torn-write refusal
  stage_determinism.json      the global-RNG check
  stage_readiness.json        the 16 readiness checks, storage projection, resume boundaries
  stage_worker_reverify.json  the post-fix worker kill at production population
  rehearsal_supervisor_events.jsonl / rehearsal_child_events.jsonl / rehearsal_status.jsonl
  rehearsal_child_01..03.log  each training process's own output
  worker_kill_probe{,_before_fix.log,_after_fix.log}
stratego/training/phase13_rehearsal.py              the supervisor
scripts/run_phase13_agent03.py                      every stage
tests/training/test_phase13_agent03.py              21 tests
```

## 17. Stop condition (section 15)

The single 90-minute rehearsal completed; the crash/resume checks pass; the worker-failure check
completed, failed, was fixed narrowly and re-verified; the long-horizon scheduler tests completed; the
storage projection is updated; and all readiness evidence is written. **Phase 14 was not started and
Agent 4 was not begun.**
