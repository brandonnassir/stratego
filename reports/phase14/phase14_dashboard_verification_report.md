# Phase 14 local monitoring dashboard — verification report

**Status: complete. Phase 14 NOT started.**

A read-only local dashboard for the frozen 168-hour Phase 14 run, built and
verified before launch. It reads what the run already writes and changes
nothing. Recovery and control remain `PHASE_14_RUNBOOK.md` and the accepted
supervisor.

| | |
|---|---|
| address | `http://127.0.0.1:8714` (loopback only) |
| launch | `python scripts/phase14_dashboard.py` |
| implementation | `monitoring/phase14_dashboard/` (5 files) |
| launcher | `scripts/phase14_dashboard.py` |
| tests | `tests/monitoring/test_phase14_dashboard.py` — 62 passing; full suite 6,372 passed / 3 skipped |
| files under `stratego/` changed | **none** |

---

## 1. The one architectural decision worth reviewing

**The dashboard imports nothing from `stratego`, and therefore no torch.**

Every `phase14_*` module reaches torch through the contract chain —
`phase14_contract` → `phase9_contract` → `warmstart_contract` →
`stratego.model` → `torch` — measured at **205 MB resident and 0.46 s** to
import. Reusing `scripts/phase14_status.py` would have put a GPU-capable
runtime inside a process whose job is to read seven JSON files.

Worse, its `build_status()` calls `resume_checkpoint_state()`, which calls
`HotCheckpointRing.load_latest()` → `torch.load()` of a full model *and*
optimizer state, then recomputes an integrity digest over it — **per status
read**. That is correct for a resume decision and unusable at a ten-second
refresh.

So the dashboard is pure stdlib. The task forbids importing the model; making
it structurally impossible is stronger than promising not to.

**The cost of that choice, and how it is contained.** The frozen constants are
mirrored in `monitoring/phase14_dashboard/contract.py`, and duplicating a
frozen number is how a dashboard ends up confidently displaying last month's
deadline. `TestMirroredContract` therefore imports the real frozen modules —
in a test process, where torch is free — and asserts **all 26 mirrored values**
equal their source, that `segment_for_elapsed` and `learning_rate_for_elapsed`
agree with `phase14_clock` at the segment boundary, and that `RunPaths` resolves
to the same paths as `Phase14Storage`. If the contract moves, the test fails
rather than the dashboard lying.

## 2. Where each displayed metric comes from

Every value takes the source that survives a crash, and says which one it used
(`/api/sources`).

| field | source | why this source |
|---|---|---|
| **committed games** | rollout store `iteration_*/manifest.json` | authoritative; written at seal time and survives `SIGKILL` |
| in-flight games | the iteration's commit journals, distinct `phase9_game_id` | the collecting iteration has no manifest yet |
| **process counter** | `phase14_telemetry.jsonl` `collection.process_counter_games` | diagnostic only, labelled as such; low after any crash |
| elapsed / remaining / progress / segment | `phase14_run_state.json` `window`, falling back to the telemetry clock | the *original* window; downtime is lost time |
| LR, losses, grad norm, advantage retention, throughput, draw rate, game length | latest telemetry row | one row per iteration (~21 min) |
| optimizer step, population iteration | telemetry row / store | |
| **learner alive**, **live loader workers** | `pgrep -P` + `ps` against the learner PID, probed per request | a telemetry row written between iterations says nothing about *now* |
| learner PID, restarts, last exit signal, resumed-from checkpoint | `phase14_supervisor.jsonl` | the only party that can record a hard kill |
| supervisor PID and liveness | the learner's PPID, when the log does not carry it | see §3a — also detects an unsupervised learner |
| configured workers, pool rebuilds, last rebuild time/reason | telemetry row (cumulative facts the learner persists) | |
| hot / archive / candidate checkpoint age | filesystem `mtime`; **no checkpoint is opened** | |
| candidate ledger | `evaluations/phase14_candidate_ledger.json`, read as JSON | not via `CandidateLedger`, whose module imports torch |
| iteration-commit events | the store's `state.json` `history` transition stamps | so commits interleave with supervisor events in one time-ordered stream |
| disk | `statvfs` on the external volume | measured now, not copied from a plan |
| emergency stop / integrity failure | the two flag files | read, never written |

## 3. Required checks

All from `§20`, each verified:

| check | result | evidence |
|---|---|---|
| dashboard starts locally | pass | serves on `127.0.0.1:8714`; page rendered in light and dark |
| reads current approved telemetry | pass | read both real Phase 13 rehearsal run directories |
| displays authoritative game counts | pass | **reproduces the Phase 13 finding**: rehearsal store reports **8,192 committed** where the process counter says **4,096** |
| loader health is live | pass | detected 6/6 real `ProcessPoolExecutor` workers; correctly excluded the resource tracker; saw 6 → 0 when a worker was `SIGKILL`ed |
| supervisor failures/restarts visible | pass | restart count, last exit code/signal, resumed-from checkpoint, latest restart time; first launch is not counted as a restart |
| checkpoint ages correct | pass | newest-by-mtime, verified against a file aged 90 s vs one aged 400 s |
| deadline/elapsed/remaining correct | pass | 73h24m elapsed → 94h36m remaining at 43.7%; a 6-hour outage removes 6 hours of the 168 and moves neither deadline nor transition |
| disk status correct | pass | 925.2 GiB free on the real volume; graded against the frozen 120 GiB reserve |
| no MPS/model inference occurs | pass | see §4 |
| no frozen training state modified | pass | see §5 |
| trainer does not depend on dashboard | pass | see §6 |
| killing dashboard leaves trainer unaffected | pass | see §6 |

### 3a. One gap found and closed without touching frozen code

The accepted supervisor's `launch` record carries `learner_pid` but **not its
own PID**, so a naive reading leaves "is the supervisor alive?" permanently
`unknown` — and that is one of the questions `§10` specifically asks the
dashboard to answer.

Changing the supervisor to log its PID would move sealed Phase 14 code. It does
not need to: `spawn()` gives the learner a new *session*, not a new parent, so
while the supervisor lives it **is** the learner's PPID.

The derivation also answers something a logged PID could not. A learner whose
parent is PID 1 has been reparented to `launchd` — the supervisor died and left
it running. Training continues, but nothing is watching for the next crash, and
without this an operator would have to notice by hand.

Verified against real processes:

```text
actual supervisor 74359, learner 74361
derived           74359  "derived from the learner's parent process"

after SIGKILL of the supervisor:
learner still alive  True
derived              None  "learner reparented to launchd"  orphaned=True
```

An unsupervised learner is graded **red** and points at the runbook. A
`supervisor_pid` in the log, if one is ever added, is preferred over the
derivation. Three tests cover all three paths.

### 3b. Verified against a live run, not only against fixtures

The rehearsal directories on the external volume are *finished* runs, so they
exercise `COMPLETE` and not the state Phase 14 will spend 168 hours in. A
stand-in live run was therefore built to the accepted file formats — a
supervisor spawning a learner with a **real 6-worker `ProcessPoolExecutor`**,
209 committed iterations plus one collecting, a restart after `SIGKILL`, a
recorded pool rebuild, and telemetry appended every few seconds — and the
dashboard was pointed at it.

It read, live and correctly:

```text
PHASE 14  TRAINING   learner alive and inside the run window

████████████████░░░░░░░░░░░░░░░░  43.7%
73h 25m elapsed   94h 34m remaining     (the §5 worked example, exactly)

supervisor          ok      alive, pid 77794   (derived — not logged)
learner             ok      alive, pid 77796
loaders             ok      6 of 6 live        (real pool workers)
pool rebuilds       watch   1 of 16 — recovered, but count them
                            BrokenProcessPool: a loader worker exited
checkpoint          ok      1.6 min old
storage             watch   222.1 GiB free
nonfinite           ok      0
telemetry           ok      0 min old

committed games  428,032   AUTHORITATIVE
in flight          1,310   (journal-derived, the collecting iteration)
process counter  413,696   DIAGNOSTIC
supervisor restarts    1   last exit code -9 / SIGKILL
```

Then, with that run still going, the dashboard was `SIGKILL`ed:

| | before | after |
|---|---|---|
| supervisor 77794 | alive | **alive** |
| learner 77796 | alive | **alive** |
| loader workers | 6 (+tracker) | **6 (+tracker)** |
| telemetry rows | 40 | **42 — still advancing** |

A fresh dashboard was then started against the same live run and reconnected
immediately, reporting the same authoritative 428,032 games and `TRAINING`.

While serving that live browser at a 10 s refresh the dashboard used
**0.09 s of CPU and 24.7 MB resident**.

## 4. Zero model, zero MPS — confirmed

- A subprocess that imports the dashboard **and builds a full status document**
  reports `sorted(m for m in sys.modules if m.split('.')[0] in ('torch','stratego','numpy'))` == `[]`.
- No checkpoint file is ever opened. The hot panel reports `validated: false`
  with the note *"validity is the supervisor's call"*, and a test asserts the
  checkpoint's `mtime` is unchanged after a status read.
- `lsof` on the live server process: **0** torch or Metal files open.
- `vmmap` shows Metal/MPS framework mappings — and a bare
  `python -c "time.sleep(25)"` shows **exactly the same 69 mappings and 16
  IOAccelerator regions**. Those are the macOS dyld shared cache present in
  every process; the dashboard adds nothing and initialises no Metal device.

## 5. Read-only — confirmed

- **Empirical:** every file under the run root and hot ring is digested
  (size, `mtime_ns`, SHA-256), the dashboard is refreshed **100 times with all
  caches expired**, and the tree is digested again. Identical, with no file
  created or removed.
- **Structural:** the package source is parsed with `ast`; no call to `write`,
  `write_text`, `write_bytes`, `mkdir`, `makedirs`, `unlink`, `remove`,
  `rmtree`, `rename`, `chmod`, `utime`, `truncate`, `touch`, `symlink`, `link`,
  `fsync` or `copy` exists anywhere, and the only `open()` in the package is
  mode `"rb"`. (Writes to the HTTP response socket and `stderr` are exempted
  explicitly; neither is the run directory.)
- Reading a run directory that does not exist **does not create it** — a
  monitor that brought `/Volumes/.../stratego_phase14` into existence by
  looking at it would be a real defect. It reports `NOT STARTED`.
- **HTTP:** `POST`/`PUT`/`PATCH`/`DELETE` return 405 pointing at the runbook;
  any path outside the five-route allowlist returns 404. Binding anything but
  loopback is refused unless `--allow-nonlocal` is passed.
- **UI:** the page contains no `<input>`, `<button>`, `<form>`, `<select>`,
  `<textarea>` or `contenteditable`, and exactly one `fetch()`, which is a GET.
  There is no control for LR, opponent mixture, setup selector, batch size,
  loss weights, pool strategy, search mode, duration, deadline, optimizer
  settings, checkpoint deletion, a new run, or a restarted clock — absent, not
  disabled.

## 6. Training independence — confirmed

Against a stand-in trainer process that appends telemetry continuously:

1. the trainer runs with the dashboard absent;
2. the dashboard attaches to an already-running trainer and picks up its rows;
3. the dashboard is `SIGKILL`ed mid-run and the trainer keeps advancing — its
   step counter increases across the kill and the process stays alive. Repeated
   against the **live** stand-in run in §3b, where a supervisor, a learner and
   six real pool workers all survived the dashboard's death;
4. closing the browser changes nothing: the page stops polling when hidden, and
   the server does no work between requests;
5. a restarted dashboard reconnects and reports the same authoritative
   8,192 games, with the tree digest unchanged;
6. `lsof` confirms the dashboard holds **no descriptor** on any run file
   between requests — nothing for the trainer, supervisor, checkpoint writer or
   rollout store to contend with.

A missing run directory, a torn `manifest.json`, a truncated `state.json`, an
unparseable telemetry log and a half-written supervisor line are each handled
as reported absences rather than exceptions — the moment an operator most needs
the page is the moment the run is in the worst shape.

## 7. Measured overhead

On the M4 Pro, against a synthetic **480-iteration / 983,040-game** run — a
full 168 hours at the measured ~21 min/iteration:

| | |
|---|---|
| resident memory | **23.8 MB** (target: tens to low hundreds) |
| CPU idle | **0.0%** — nothing runs between requests |
| CPU over 5 min 23 s serving a browser | **0.06 s total** |
| cold read (480 manifests parsed) | 29.0 ms |
| refresh, **all caches expired** | **6.5 ms** median, 7.5 ms worst |
| refresh, warm (the 10 s case) | **~1.0 ms** |
| status payload | 45.8 KiB |
| MPS / GPU | **none** |

Three things keep it there: per-source TTLs (5–60 s); a sealed iteration is
parsed **once** and then costs two `stat` calls forever (verified: 20
iterations, 20 parses, unchanged across 10 further expired-cache refreshes);
and the telemetry, supervisor and journal files are read incrementally from a
byte offset, with inode and shrink detection so a replaced log is re-read from
the top rather than at a stale offset.

Browser history is bounded at 400 telemetry rows and 60 events.

## 8. Sealed Phase 14 code identity — unchanged

`§16` requires that any change to the sealed code identity be reported.
**There is none.**

- **Zero** files under `stratego/` were modified — `git status` shows only new
  untracked paths (`monitoring/`, `scripts/phase14_dashboard.py`,
  `tests/monitoring/`).
- All **113 files** in the launch manifest's code closure verify
  byte-identical against `phase14_launch_manifest_v1.json`.
- No `monitoring/` file is in that closure, and no module under `stratego/`
  imports the dashboard — so it cannot enter the closure and cannot move
  `code_digest`.
- Three tests guard this permanently.

`phase14_contract_digest`, `integrated_config_digest` and
`phase14_final_training_config_digest` are all untouched.

**Pre-existing, not caused by this work:**
`scripts/phase14_build_launch_package.py --verify` currently fails, because the
manifest was built at revision `e6daae8` with a dirty tree and Agent 4's commit
`0557fee` moved `HEAD`. This is the known, expected step already recorded for
launch: **rebuild the launch package after committing, before Phase 14
begins.** Committing the dashboard moves `HEAD` again, so do the rebuild after
that commit, not before.

## 9. Known display limitations

1. **Training numbers are up to one iteration (~21 min) old.** The learner
   publishes one telemetry row per completed iteration; that is the design, not
   a fault. It is stated on the page, and `telemetry` is graded `watch` past
   45 minutes. The fast-moving fields — the wall clock, learner and loader
   liveness, checkpoint ages, disk, supervisor events, in-flight game count —
   are all live and independent of that cadence.
2. **Checkpoint validity is not checked**, only presence, size and age.
   Validating means `torch.load`. `resume_checkpoint_state` in the accepted
   supervisor is the authority, and the panel says so.
3. *(resolved during the work — see §3a.)* The accepted supervisor does not log
   its own PID. The dashboard derives it instead, so no frozen code changes and
   the supervisor row is live.
4. **Before the first hot checkpoint and first telemetry row**, most training
   fields read `—` and health reads `unknown`. That is the honest state of a
   run in its first iteration.
5. **Games/hour is a run-average**, committed games over elapsed wall-clock, so
   it dilutes across downtime rather than reporting an instantaneous rate. The
   instantaneous rate is the separate `games/s` from the latest row.
6. **The first read after the external volume has been idle can take a few
   seconds** while the drive wakes — measured 2.96 s once, then 1.3 ms. Not a
   recurring cost.
7. **The dashboard reads one run at a time**; `--external-root` selects which.
8. Verified in the in-app Chromium at 1280 px in both colour schemes. The
   layout is responsive but has not been exercised on a phone.

## 10. Stop condition

```text
dashboard implementation complete      yes
read-only isolation verified           yes  (empirical + structural + HTTP + UI)
resource overhead acceptable           yes  (24 MB, 0.0% idle, 6.5 ms refresh)
training independence verified         yes  (6 of 6 checks in §15)
instructions documented                yes  (monitoring/README.md)
Phase 14 NOT STARTED                   correct
```

The safest dashboard is one that can disappear entirely and Phase 14 never
notices. This one can.
