# Phase 13 — Agent 4: the Immutable Phase 14 Launch Package and the GO / NO-GO Decision

Run date: 2026-08-21. Task:
`instructions/phase_13_final_training_integration/04_AGENT_4_IMMUTABLE_LAUNCH_PACKAGE.md`,
with the ten mandatory additions.

**Recommendation: `GO`.** All ten launch-readiness gates pass, 90 of 90 checks.
No frozen training value was changed, no rehearsal was rerun, and Phase 14 was
not started.

```text
phase14 contract digest    62ce6d4e04ffd25755717ef290f7486f2616927ddada59d8ea9fb05565c052b9   unchanged
integrated config digest   9c2a38e4335762997adbb33731dc619615fff713c2c60840c7c8d74a2f29da5e   unchanged
final training config      3770ebd4a0bb812f7ac49e8273f3485afb07e828915dbd767e47a2ca65bbff89   new
launch manifest            92bd041d6fa9954cd9734708c5c8106d046faac34b8cb86019d779815cf68cf6   new
phase14 code digest        beed81a8a1dfff30a514a123dfac10aa821216f0ec0b3089e766ae2e1777df64   new
bound git revision         e6daae8df7e1da697263635db0aadc70651b3dd8  (+ 4 modified tracked files, listed)
```

The manifest digest deliberately excludes `built_utc`, so rebuilding the package
over unchanged code reproduces the same identity: a digest that moved means the
bound *content* moved and nothing else.

## 1. What this agent did and did not do

Did: reviewed Agents 1–3; re-derived every upstream identity live; built the
four narrow monitoring/supervisor additions the task mandates; froze
`phase14_final_training_config_v1` and `phase14_launch_manifest_v1`; wrote
`PHASE_14_RUNBOOK.md`; evaluated Gates A–J mechanically; and returned `GO`.

Did not: start Phase 14; rerun the 90-minute rehearsal; change a learning rate,
loss weight, opponent mixture, pool algorithm, setup source, cadence, candidate
pack, selection rule or deadline; run a strength experiment; use search; or
redesign RNG handling. Both frozen digests are byte-identical to Agent 2's, and
that is checked by a test rather than asserted here.

## 2. Binding the post-Agent-3 code revision (addition 1)

The premise is exact and worth restating, because it is the reason this section
exists at all. The Agent 3 loader-pool repair changed **neither** frozen digest:
`contract_digest` is computed over frozen *values* and `integrated_config_digest`
over those plus module *version strings*, and the repair moved neither. A run
launched against the pre-repair code would present both correct digests and
would still die the first time a CPU loader worker was killed — which is exactly
what happened at 3,303 s of the rehearsal.

`phase14_launch_manifest_v1` therefore binds the code three ways, and
`assert_launch_code` recomputes all three before anything starts:

```text
git revision            e6daae8df7e1da697263635db0aadc70651b3dd8
tracked working tree    exactly 4 modified files, named:
                          .gitignore
                          stratego/training/phase14_runner.py
                          stratego/training/phase14_telemetry.py
                          stratego/training/phase14_trainer.py
code content digest     beed81a8...  over 113 files, per-file SHA-256
worker-pool repair      6 positive assertions, not a digest
operator scripts        7 entries, each bound by SHA-256
```

The 113 files are the **actual import closure** of the Phase 14 training graph,
computed by importing the declared entry points and reading `sys.modules` —
including the runner's several lazy in-method imports, which a static scan would
miss. The repair assertions are the part digests cannot do:

```text
BrokenExecutor in RECOVERABLE_ERRORS                     True
_next_minibatch catches BrokenExecutor                   True
_next_minibatch rebuilds at the same cursor              True
rebuilds are counted                                     True
MAX_LOADER_POOL_REBUILDS == 16                           True   (unchanged)
rebuild events are recorded                              True
```

The committed revision `e6daae8` was verified to carry the repair by reading it
out of git directly, not from the working tree.

**A defect in my own first design, found by running the full suite.** The
closure was originally computed in the calling process. Under `pytest`, another
test had already imported `stratego.search`, and the closure silently grew to
include eight search modules — making the bound file set depend on the caller.
The closure is now computed in a clean subprocess, and a test pins it by
deliberately importing `stratego.search` first and asserting the closure is
unchanged. That check doubles as the structural search-exclusion proof section 3
asks for: **no `stratego/search/` module is in the training closure.**

**Launch refuses a different tracked-code revision.** Three separate refusals,
each tested: a different commit; the same commit with edited bytes; a replaced
operator script. The only sanctioned way to move the binding is
`scripts/phase14_build_launch_package.py`, which is step 1.3 of the runbook.

> **Note for the reviewer.** The package on disk is built over the current
> working tree, whose four modified tracked files are Agent 4's own additions.
> When this work is committed, HEAD moves and the tree goes clean, so the bound
> revision no longer matches — **which is the mechanism working, not a fault.**
> Rebuild the package after committing, then run `--verify`. The test suite does
> not depend on the on-disk package being current; it tests the mechanism
> against a freshly built manifest.

## 3. Authoritative committed-game monitoring (addition 2)

Agent 3's observation, reproduced here against the real rehearsal store rather
than restated:

```text
/Volumes/Brandon_Washington/stratego_phase13_rehearsal/rollouts
  iteration_001  COMMITTED  2048   manifest.json
  iteration_002  COMMITTED  2048   manifest.json
  iteration_003  COMMITTED  2048   manifest.json
  iteration_004  TRAINING   2048   manifest.json
committed games (authoritative)   8192
process counter   (diagnostic)    4096
shortfall                         4096
```

`committed_game_census` reads the rollout store's per-iteration `manifest.json`,
which `seal_iteration` writes only after the scheduled set, the journals, every
payload digest and every metadata sidecar agree. That file survives a `SIGKILL`;
a counter in a Python object does not.

Training semantics are untouched. What changed is where a number comes from:

```text
collection.games_generated          now the store's committed total
collection.committed_games          authoritative, explicitly labelled
collection.committed_games_source   "rollout store iteration manifests"
collection.in_flight_games          the COLLECTING iteration, counted separately
collection.process_counter_games    diagnostic, retained
collection.process_counter_shortfall
```

The frozen metric named "games generated" now answers correctly rather than
sitting beside a correct field, because two numbers under one name is how an
operator reads the wrong one at hour 140. The process counter is kept, labelled
as diagnostic, and its shortfall is reported so the divergence is visible
instead of silent. An iteration still COLLECTING has no manifest yet, so it is
counted from its journals by **distinct** game id — a duplicate commit is a
defect the store's own reconciliation reports, and double-counting it here would
quietly inflate the authoritative total.

`FROZEN_METRIC_LIST` was **not** touched: it lives in the contract document and
moving it would move the contract digest. The additions are a separate
`EXTENDED_METRIC_PATHS` map with its own `missing_extended_metrics` check, and a
real run emits `missing_metrics: []` and `missing_extended_metrics: []`.

## 4. Actual loader health (addition 3)

The old field was `{"loader_workers": 6, "status": "single-process
bulk-synchronous loop"}` — a configured constant that, as Agent 3 wrote, would
not have shown the dead worker. It now reports:

```text
configured_loader_workers     from the topology
live_loader_workers           live OS children of the learner, spawn_main only
live_loader_worker_pids       the resource tracker filtered out by command line
pool_open                     whether a pool should have workers at all
loader_pool_rebuilds          counted
max_loader_pool_rebuilds      16, unchanged
last_pool_rebuild_utc         recorded
last_pool_rebuild_reason      the exception text
status                        a health sentence, the frozen metric's own path
```

Two things about this were not obvious and are worth stating plainly.

**Zero live workers is usually correct.** The `ProcessPoolExecutor` exists only
while an iteration trains. During the four to five minutes of every iteration
spent collecting, a healthy run has zero workers. Reporting that as a fault
would train an operator to ignore the field, which is how a real dead worker
gets missed — so `pool_open` is reported and the status sentence says "pool
idle" rather than "0 of 6".

**The live count is sampled where it means something.** The learner publishes
telemetry once per iteration, immediately *after* it closes its pool, so a live
count in a telemetry row would always read zero. The live probe therefore
happens in two places that can see a running learner mid-epoch: the supervisor
samples its child every 60 s and logs `worker_health_sample`, and
`scripts/phase14_status.py` probes the current learner PID at the moment the
operator asks. The cumulative rebuild facts come from the learner's own
persisted state, because those are facts it must carry across a restart.

`MAX_LOADER_POOL_REBUILDS` remains 16, as the accepted implementation defines.

## 5. The production launch supervisor (addition 4)

`stratego/training/phase14_supervisor.py`, driven by `scripts/phase14_launch.py`.
It contains no training logic, no schedule and no configuration.

Recorded, all of it fsync-ed per record to
`logs/phase14_supervisor.jsonl`:

```text
launch timestamp        every `launch` event
learner PID             every `launch` event
unexpected exit         `unexpected_exit` vs `expected_exit`, decided from disk
exit code / signal      split apart; -9 reads as SIGKILL
restart attempt         `restart_attempt`, with its backoff
checkpoint selected     path and optimizer step, read before the relaunch
restart success/failure `restart_success` when the step actually advances
final process exit      `final_process_exit`, with the reason
```

**It never creates a deadline.** It passes no window, no duration and no
deadline argument to the learner — a test asserts the argument vector contains
no such token — and the learner always calls `start_or_resume()`, which resumes
the window persisted in the checkpoint. Before every relaunch the supervisor
re-reads that window off disk and raises if it moved.

**Restart success is measured, not assumed.** The consecutive-restart count
clears only when the persisted optimizer step advances past the step observed at
the last launch. A crash loop in which each child survives a probation window
but makes no progress would still trip the bound; five such restarts stop the
supervisor.

**The five refusals**, checked in this order, each individually tested:

```text
1  an emergency stop is active
2  the run manifest says training is closed
3  an unrecoverable integrity failure has been recorded
4  no valid resume checkpoint exists
5  the deadline has passed
```

Order matters, and one case shows why: a learner that dies past the deadline
while an emergency stop is active is reported as an emergency stop, because that
is the fact the operator needs.

Refusal 4 deserves its own sentence. Without a valid hot checkpoint there is
nothing to resume, and a fresh `start()` would stamp a **new 168-hour deadline**.
The supervisor stops instead.

**One interpretation, flagged explicitly.** Requirement 4 both forbids
restarting after the deadline and says "a post-deadline restart attempt must
still take zero optimizer steps", which presupposes such an attempt can occur.
Taken strictly, a learner killed at hour 167.9 would leave the run permanently
unfinalized — no hour-168 candidate, no manifest, no closure. I have implemented
refusal 5 as: **training does not restart, and exactly one closeout launch is
made.** That launch runs `--role finalize`, which never calls `run()` at all: it
resumes, asserts the deadline has passed, and finalizes. It is logged as
`closeout_launch`, counted separately from restarts, and capped. Agent 3 verified
the underlying path nine hours past a real deadline, and a test here repeats the
runner-level property directly: resume past the deadline, `global_optimizer_step`
unchanged, window byte-identical.

**An unrecoverable integrity failure is now durable.** The learner catches
`Phase14IntegrityError` and writes `phase14_integrity_failure.json` before
exiting, so the supervisor refuses to relaunch over a run whose identity is in
doubt. A dead process cannot record its own death — that is the supervisor's job
— but a process dying of a *diagnosed* failure can, and now does.

**The emergency stop became durable too.** The accepted `ControlSurface` offered
emergency stop only in-process, which an operator at another terminal cannot
reach. It now also watches a stop file under the run's external root, written by
`scripts/phase14_emergency_stop.py`. It remains a stop and not a setting: every
frozen key is still refused by name, and a test walks the whole
`IMMUTABLE_CONTROL_KEYS` list to confirm it.

## 6. Frozen production topology (addition 5)

Frozen exactly as Agent 3 rehearsed it, and recorded in the launch manifest
**separately** from the logical config digest, which Agent 2 deliberately kept
free of operational choices:

```text
device                 MPS
inference device       MPS
CPU loader workers     6
games in flight        96
inference batch shape  64
population             the frozen 2,048-game production mixture
in_logical_config_digest   false
```

`assert_frozen_topology` refuses a production launch on anything else. No
alternative worker counts were benchmarked.

## 7. Candidate evaluation without manual memory (addition 6)

Frozen operational policy, implemented rather than described:

* the supervisor reads the candidate ledger **off disk** every 10 minutes and,
  when a mark is pending, launches `scripts/phase14_evaluate_candidates.py` as a
  **separate process**, one at a time, never inside the learner;
* results reach the ledger and nothing else. The evaluator imports no trainer,
  no runner, no scheduler and no clock — checked structurally in a subprocess —
  and **no search**;
* a failed evaluation preserves the candidate, records the reason, leaves
  training untouched and stays re-runnable on the identical pack;
* at hour 168, `assert_all_candidates_evaluated` **refuses** the frozen selection
  rule while any marked candidate lacks a complete 128-game result.
  `scripts/phase14_select_final.py` is that gate, and it also refuses to select
  before training is closed.

The last point is the one that matters most: without it, "hour 168 wins by
default" is exactly the outcome an unevaluated ledger produces, and it is the
outcome the frozen rule exists to prevent.

## 8. Checkpoint-age semantics, stated accurately (addition 7)

Not "every crash loses at most exactly 15 minutes". The design, as measured:

```text
nominal hot-checkpoint cadence      900 s
writes occur at                     safe training boundaries, not a timer
no write occurs during              a collection unit (297-311 s observed)
sealed rollout games                survive a later learner crash
what a crash can cost               un-checkpointed optimizer work, repeated
max checkpoint age observed         895.4 s, 0 of 1,067 samples above cadence
games replayed across two crashes   0
optimizer work repeated             197.6 s and 232.4 s
read checkpoint age as              up to one cadence PLUS a collection
```

The runbook says the same thing in the monitoring table, so an operator reading
`checkpoint age` at 1,100 s does not page anyone.

## 9. RNG documentation, preserved (addition 8)

```text
global RNG state captured = yes
global RNG state restored = no
reason: logical Phase 14 randomness uses explicit deterministic streams rather
        than ambient global RNG
```

Agent 3's determinism verification is bound as the evidence: the same iteration
collected and trained twice, in two directories, under two deliberately
different global `torch` seeds, produced identical games, decisions, terminal
results, bucket counts, updates, examples consumed, epoch plan and final model
state digest `f27ea740c4f13c7e5c8576adb1a789c84061cc6777ba7661d90f311cfa9cdf60`.
Nothing about RNG handling was redesigned.

## 10. No training changes (addition 9)

Mechanically verified, not asserted: `assert_matches_frozen_contract()` returns
zero disagreements, and both frozen digests are unchanged. Every value on the
prohibited list — main LR 7.5e-05, late LR 3.75e-05, the 132 h transition, both
population mixtures, belief weight 0.25, the historical-pool algorithm, the
Phase 14 setup source, the 15 m / 2 h / 6 h cadences, the candidate pack, the
selection rule, the 168-hour duration and the search prohibition — is checked in
Gate B against the live modules.

Three source files were modified, carrying 152 added lines between them, all of
it monitoring plumbing: telemetry field sources, the loader-rebuild recorder, the
durable stop file, and the production code-binding check. The fourth modified
tracked file is `.gitignore`, which gains one rule so the Phase 14 hot-checkpoint
ring — a bounded set of ~10 MB working files rewritten every 15 minutes for a
week — is not committed. No accepted upstream artifact was touched.

## 11. Launch-readiness gates

**90 of 90 checks pass** (`reports/phase13/agent04_evidence/gates.json`).

| gate | | checks | result |
|---|---|---|---|
| A | Upstream identity | 11 | PASS |
| B | Final training contract | 15 | PASS |
| C | Setup safety | 11 | PASS |
| D | Training correctness | 6 | PASS |
| E | Recovery | 7 | PASS |
| F | Wall-clock semantics | 6 | PASS |
| G | Historical system | 8 | PASS |
| H | Candidate system | 9 | PASS |
| I | Storage | 7 | PASS |
| J | Controls | 10 | PASS |

Selected evidence, all re-derived live this session rather than quoted:

**A.** Starting checkpoint `dfd698e5…` and both pool anchors re-hashed and
matching; frozen contract file `65d1f941…`; both frozen digests unchanged;
rehearsal identity `d8ebae4e…`; pack `896a753b…` with 128 games; the selection
rule's own `pack_binding` naming that same digest; Agent 1C is not the
policy/value checkpoint.

**B.** Zero contract disagreements. `7.5e-05 < 3e-04` and `3.75e-05 < 7.5e-05`,
exactly 0.25× and 0.125× LR9. Handcrafted share 11.96% in both segments, inside
the 10–15% band, with all five families present and unchanged between segments;
the late segment moves 369 games from current to historical and changes nothing
else. Transition tied to `run_start_utc`. `learning_rate` is on the immutable
control list, so no live LR tuning exists in the normal control path.

**C.** The alarm policy was written at 20:51:35 and the census at 21:00:56 —
policy first, and the policy file on disk hashes to the value the census names.
Zero defect observations across D1–D6; `P_trivial` 0.003296 < 0.05 and
`P_predecision` 0.001953 < 0.025. Reflection and perturbation changed the Flag
row **0** times in 32,768 draws. Forward-Flag exposure is attributed entirely to
one deliberate family (F15 `irregular_high_entropy`). No repair occurred, so no
Phase 10 evidence was rewritten.

One item deserves naming rather than burying: the Phase 12 setup-library
flag-on-front-row observation. Agent 1 traced it to the **Phase 11B glue**
returning canonical tuples and measured the production path clean (D4 = 0 over
8,192 engine boards), and warned Agent 2 not to reuse that glue. Gate C
re-verifies the live path structurally: the source is the P10-D learned
selector, every placement goes through `SelectorDraw.oriented(player)`, and the
canonical and oriented boards genuinely differ. The runner re-runs that same
probe before the window is stamped.

**D.** Finite updates and moving parameters at the production population; the
belief auxiliary non-zero and finite in every unit at weight 0.25; no telemetry
row missing a frozen metric; and no search module in the training import
closure.

**E.** Process-group `SIGKILL` recovered with zero orphaned workers; optimizer
state and pool preserved; worker failure recovered **after** the repair, which
is installed in this revision and was re-verified at the frozen 2,048-game
population (one of six workers killed, learner survived, 1,744 updates — exactly
the uninterrupted count). Plus the new fact: a killed process is now recorded by
something outside it.

**F.** One window across three rehearsal launches and every checkpoint; an
Agent 2 resume reproduced the window byte for byte; post-deadline recovery takes
zero optimizer steps; the deadline stop is automatic; the late transition reuses
the original start; the supervisor never creates a deadline.

**G.** 2-hour archive, append-only and never pruned; membership a pure function
of the ordered archive with no tournament admission; both permanent anchors;
bounded at 16 with weights 20/25/25/30; a resumed checkpoint recomputes the same
pool; pool continuity held across both rehearsal crashes.

**H.** 29 candidate hours at 6-hour cadence; one frozen 128-game pack for every
candidate; no search; the evaluator reaches no training module; every frozen key
refused by name; the hour-168 candidate marked `pending`, not selected; an
incomplete evaluation refused by the selection rule; pending marks detected from
disk.

**I.** Volume mounted and writable. Measured 168-hour projection **34.815 GiB**;
with the required 20% reserve, **41.78 GiB** against **925.17 GiB** free. Agent
1's conservative 600 GiB planning ceiling also fits with reserve. Full raw
retention is the plan; the 120 GiB contingency threshold is nowhere near.
Earlier accepted evidence can never be deleted. No Phase 14 run identity exists:
no hot checkpoint, no run manifest, no rollout iteration.

**J.** All 23 frozen metrics have snapshot paths and all 7 Agent 4 additions are
exposed and checked; the authoritative total comes from the store and worker
health from the OS; all 8 frozen keys refused by name; a durable emergency stop
can be requested, reaches the run, and can be cleared; the supervisor refuses to
restart over one; restart and resume are one documented procedure.

## 12. One host-configuration finding

Not a gate failure, but the most likely way to lose days, so it is stated
plainly.

```text
pmset sleep      1 minute
pmset disksleep  10 minutes
currently        "sleep prevented by powerd, Claude"
```

This Mac is configured to idle-sleep after **one minute**. It is awake right now
only because of a transient power assertion held by this Claude Code session,
which ends when the session does. The Phase 14 deadline is wall-clock: a machine
that sleeps at hour 3 does not pause the run, it **loses the hours**.

Two things were done about it. The runbook's launch command is
`caffeinate -dimsu python scripts/phase14_launch.py`, which holds the assertion
for the whole run without needing a password. And the supervisor's preflight
**refuses to launch** unless idle sleep is disabled or an assertion is
preventing it — so the failure mode cannot be reached by forgetting. An
unreadable `pmset` is recorded rather than treated as unsafe.

## 13. Deliverables

```text
reports/phase13/phase14_final_training_config_v1.json   3770ebd4...
reports/phase13/phase14_launch_manifest_v1.json         92bd041d...
PHASE_14_RUNBOOK.md
reports/phase13/phase13_agent_04_report.md              this document
reports/phase13/phase13_agent_04_summary.json
reports/phase13/agent04_evidence/gates.json             90 checks, evidence per check

stratego/training/phase14_launch.py         code binding, launch package, control files
stratego/training/phase14_status.py         authoritative census, live loader health
stratego/training/phase14_supervisor.py     the production supervisor
scripts/phase14_launch.py                   launch = resume; roles supervisor/learner/finalize
scripts/phase14_status.py                   read-only operator status
scripts/phase14_emergency_stop.py           the durable stop
scripts/phase14_evaluate_candidates.py      out-of-band candidate evaluation
scripts/phase14_select_final.py             the gated post-run selection
scripts/phase14_build_launch_package.py     the deliberate rebuild
scripts/run_phase13_agent04.py              gates A-J
tests/training/test_phase13_agent04.py      66 targeted tests

modified (152 added lines, monitoring only):
  stratego/training/phase14_runner.py
  stratego/training/phase14_telemetry.py
  stratego/training/phase14_trainer.py
  .gitignore                                  one rule: the phase14 hot ring
```

## 14. Tests

```text
tests/training/test_phase13_agent04.py      66 passed
pytest tests (full suite)                6,310 passed, 3 skipped in 471 s
```

Agent 3's accepted baseline was 6,244 passed / 3 skipped. The difference is
exactly the 66 tests added here: nothing regressed and nothing was lost. Agent 2's
61 tests and Agent 3's 21 tests all still pass unchanged, including the real
worker-kill tests that exercise the repair.

Targeted tests only, as required. **The 90-minute rehearsal was not rerun**, and
neither was the worker-kill verification: Agent 3 demonstrated the underlying
recovery path at the frozen production population and that evidence is bound
here rather than re-manufactured.

## 15. Recommendation

```text
GO
```

All ten launch-readiness gates pass. The launch package is frozen, the code
revision carrying the accepted worker-pool repair is bound and provably
installed, the three monitoring gaps Agent 3 recorded are closed, and a
production supervisor now records what a killed process cannot.

Before launching, do two things the runbook puts first: **rebuild the launch
package** once this work is committed (§1.3), and launch under `caffeinate`
(§1.1, §2).

Phase 14 was not started.
