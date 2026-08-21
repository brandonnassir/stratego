# Phase 14 local monitoring dashboard

A read-only web page that answers one question:

> Is Phase 14 alive, healthy, progressing, checkpointing, and staying inside its
> immutable seven-day contract?

It answers it without changing anything. If this dashboard crashes, is killed,
or is never opened, Phase 14 does not notice.

Recovery and control remain **`PHASE_14_RUNBOOK.md`** and the accepted
supervisor. This page has no button that does anything.

---

## Run it

```bash
python scripts/phase14_dashboard.py
```

Then open <http://127.0.0.1:8714>.

It binds `127.0.0.1` only. Start it before, during, or after the run — it reads
files, so it does not care whether a learner is alive.

Other forms:

```bash
python scripts/phase14_dashboard.py --port 9000
```

```bash
python scripts/phase14_dashboard.py --once
```

```bash
python scripts/phase14_dashboard.py --external-root /Volumes/Brandon_Washington/stratego_phase13_rehearsal --hot-root checkpoints/phase13_rehearsal/hot
```

`--once` prints one status document as JSON and exits, which is the form to use
from a script. `--verbose` logs every request. `--allow-nonlocal` is required
before it will bind anything but loopback, and there is no authentication, so
do not.

Stop it with Ctrl-C, or `kill` it — either is safe at any moment.

## Endpoints

| path | what it is |
|---|---|
| `/` | the page |
| `/api/status` | the whole status document (~46 KiB at full scale) |
| `/api/health` | the compact form: overall state, one word per check |
| `/api/sources` | which file each displayed value is read from |

Everything is GET. `POST`, `PUT`, `PATCH` and `DELETE` return 405 with a
pointer to the runbook; any other path returns 404.

For a monitoring agent, `/api/health` is the cheap one:

```bash
curl -s http://127.0.0.1:8714/api/health | python -m json.tool
```

## How to read it

| field | how to read it |
|---|---|
| **committed games** | authoritative — the rollout store's iteration manifests. This is the real number |
| **process counter** | diagnostic only — process-local and restored from the last hot checkpoint, so it is *low after any crash*. A shortfall is expected and is not an alarm |
| **elapsed / remaining** | measured from the original run window. Downtime is lost time. A restart never begins a fresh 168-hour clock |
| **loader workers** | `live / configured`, probed against the OS now. Zero live during a collection is normal — the pool exists only while an iteration trains |
| **pool rebuilds** | counted against the frozen ceiling of 16. One is a dead worker recovered; a rising count is a sick machine |
| **checkpoint age** | up to one 900 s cadence *plus* one collection (~300 s). Sustained past ~1,300 s shows as `watch` |
| **telemetry age** | the learner publishes one row per iteration (~21 min), so the training numbers are up to one iteration old **by design** |
| **supervisor restarts** | from the supervisor's log, which is the only party that can record a hard kill |

Statuses are `ok` / `watch` / `bad` / `unknown`. They are only ever applied to
conditions Phase 14's frozen contract already defines. Loss movement is
displayed and never graded — "policy loss went up, therefore training is bad"
is not in the contract, and an invented alarm is one an operator learns to
ignore.

`unknown` is a real answer, and it usually means "the run has not published
this yet", not "something is wrong".

## What it will not do

No control for learning rate, opponent mixture, setup selector, batch size,
loss weights, historical-pool strategy, search mode, training duration, the
deadline, optimizer settings, checkpoint deletion, a new run, or a restarted
clock. Not disabled — absent. The page contains no `<input>`, no `<button>`,
and one `fetch()`, which is a GET.

Emergency stop is shown, and is requested with the accepted control path:

```bash
python scripts/phase14_emergency_stop.py --reason "why"
```

## Why it is built the way it is

**It imports nothing from `stratego`.** Every `phase14_*` module reaches torch
through the contract chain — about 205 MB of resident memory and a
GPU-capable runtime inside a process whose job is to read JSON. Rather than
promise not to use the model, the dashboard is built so that it cannot: pure
stdlib, no `stratego` import anywhere. The cost is that
`monitoring/phase14_dashboard/contract.py` copies the frozen constants, and
`tests/monitoring/test_phase14_dashboard.py` checks every copy against the real
frozen module so the copy cannot drift.

**It never opens a checkpoint.** The accepted `resume_checkpoint_state` answers
"does the newest hot file *validate*" and pays a full `torch.load` to do it.
That is right for a resume decision and ruinous at a ten-second refresh, so the
dashboard reports mtime, size and count, and says validity is the supervisor's
call.

**Nothing runs between requests.** No thread, no timer, no background poll. An
open dashboard with no browser attached does no work at all, and the page stops
polling when its tab is hidden.

**Sealed iterations are read once.** A committed iteration's `committed_games`
never changes, so after the first parse it costs two `stat` calls. A full
168-hour run — 480 iterations, ~983,000 games — refreshes in about 6.5 ms with
every cache expired, and about 1 ms warm.

## Cost

Measured on the M4 Pro, against a synthetic full-length 480-iteration run:

| | |
|---|---|
| resident memory | ~24 MB |
| CPU, idle | 0.0% — no work happens between requests |
| CPU, 5 min serving a browser | 0.06 s total |
| refresh, all caches expired | 6.5 ms |
| refresh, warm | ~1 ms |
| status payload | 46 KiB |
| MPS / GPU | none — no torch is loaded in the process |

## Files

```text
monitoring/phase14_dashboard/contract.py   frozen values, mirrored and test-checked
monitoring/phase14_dashboard/sources.py    every read, and the caching
monitoring/phase14_dashboard/state.py      one status document, and the health grading
monitoring/phase14_dashboard/server.py     the localhost GET-only server
monitoring/phase14_dashboard/index.html    the page
scripts/phase14_dashboard.py               the launcher
tests/monitoring/test_phase14_dashboard.py 62 tests
```

Nothing under `stratego/` was changed to build this, and nothing under
`stratego/` imports it — so the launch manifest's 113-file code closure is
byte-identical.
