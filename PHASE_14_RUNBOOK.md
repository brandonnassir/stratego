# Phase 14 Runbook — the 168-hour final training run

Bound by `reports/phase13/phase14_launch_manifest_v1.json`.
Configuration: `reports/phase13/phase14_final_training_config_v1.json`.

> **Phase 14 runs for the full 168-hour deadline regardless of intermediate
> candidate scores. The deployed direct-policy checkpoint is selected only
> after training ends, using the predeclared fixed rule.**

Two rules govern everything below.

```text
never create a new 168-hour deadline
never delete earlier accepted project evidence
```

---

## 0. What is running

| | |
|---|---|
| starting model | `checkpoints/phase9/selfplay_c1_v1.pt` (`dfd698e5…`) — policy, value **and** belief head |
| learning rate | `7.5e-05` for 132 h, then `3.75e-05` for 36 h |
| transition | elapsed **132 h against the original start**; downtime counts |
| population | 2,048 games/iteration; main 1188/615/245, late 819/984/245 |
| setup source | `phase14_setup_source_v1` (35% neutral_v1 + 65% P10-D, train split) |
| checkpoints | hot 15 min · archive 2 h · candidate 6 h |
| candidates | 29 (hours 0, 6, …, 168), each on the frozen 128-game pack `896a753b…` |
| search | **not used anywhere in training** |
| topology | MPS, 6 CPU loader workers, 96 games in flight |
| external | `/Volumes/Brandon_Washington/stratego_phase14` |
| hot | `checkpoints/phase14/hot` (internal disk) |

---

## 1. Pre-launch

Work top to bottom. Every step is a check, not a change.

**1.1 — Power. The single most likely way to lose days.**

The deadline is wall-clock. A Mac that idle-sleeps at hour 3 does not pause the
run; it loses the hours. This machine is configured `sleep 1`, `disksleep 10`,
so **the launch command must hold a power assertion** — which is why §2 wraps it
in `caffeinate`. Confirm:

```bash
pmset -g | grep -E ' sleep| disksleep'
```

Either `sleep 0` / `disksleep 0`, or a `(sleep prevented by …)` assertion that
will still exist for the whole week. A Claude Code session's assertion does not
count — it ends with the session. The supervisor refuses to launch if neither
holds.

**1.2 — External volume mounted, and with room.**

```bash
df -H /Volumes/Brandon_Washington
```

Needs ≳ 42 GiB (the measured 34.8 GiB projection plus a 20% reserve). There
were 925 GiB free at freeze.

**1.3 — Clean, frozen code revision, and the launch package that binds it.**

```bash
git status --porcelain --untracked-files=no
```

If any tracked file has changed since the manifest was built, **rebuild the
launch package deliberately** — this is the only sanctioned way to move the
bound revision:

```bash
python scripts/phase14_build_launch_package.py
```

Then verify the binding, which also re-proves that the accepted Agent 3
worker-pool repair is installed (neither frozen digest can see it):

```bash
python scripts/phase14_build_launch_package.py --verify
```

Expect `"verified": true` and `"worker_pool_repair_installed": true`.

**1.4 — Every upstream identity, and every launch-readiness gate.**

```bash
python scripts/run_phase13_agent04.py --gates
```

Expect `"recommendation": "GO"` and no failed gate.

**1.5 — No existing Phase 14 run identity.**

```bash
ls checkpoints/phase14/hot /Volumes/Brandon_Washington/stratego_phase14
```

Both must be absent or empty. A leftover hot checkpoint makes `start()` refuse —
correctly, because starting over an existing run is how a run gets two
identities. If a *real* run is in progress, this is a **resume**, not a launch;
go to §4.

**1.6 — Monitoring is available.**

```bash
python scripts/phase14_status.py
```

---

## 2. Launch

One command. It stamps `run_start_utc` and `run_deadline_utc` exactly once and
persists them in every checkpoint from then on.

```bash
caffeinate -dimsu python scripts/phase14_launch.py
```

`caffeinate -dimsu` holds display, idle, disk and system sleep off for as long
as the supervisor runs. Leave the terminal open, or start it under `nohup`/`tmux`
if the session may end:

```bash
nohup caffeinate -dimsu python scripts/phase14_launch.py > /Volumes/Brandon_Washington/stratego_phase14_launch.log 2>&1 &
```

What happens: the supervisor runs preflight (code binding, worker-pool repair,
topology, power, control files), launches the learner in its own process group,
records the launch timestamp and PID, and watches it. The learner verifies every
frozen identity, loads the accepted Phase 9 checkpoint, stamps the window,
marks the hour-0 candidate and begins.

Confirm the window was stamped once:

```bash
python scripts/phase14_status.py | head -4
```

**Write down `run_deadline_utc`.** Nothing may ever change it.

---

## 3. Normal monitoring

Read-only. Nothing in this section changes training.

```bash
python scripts/phase14_status.py          # the operator view
python scripts/phase14_status.py --json   # everything
```

What to read, and how to read it:

| field | how to read it |
|---|---|
| `committed games` | **authoritative** — the rollout store's iteration manifests |
| `process counter` | **diagnostic only** — process-local, and low after any crash. A shortfall is expected, not an alarm |
| `checkpoint age` | up to one 900 s cadence **plus one collection** (~300 s). Sustained > ~1,300 s deserves a look |
| `workers` | `configured` vs `live`. Zero live during a collection is normal — the pool exists only while an iteration trains |
| `loader pool rebuilds` | counted, capped at 16. One is a dead worker recovered. A rising count is a sick machine |
| `learner alive` | probed now, not remembered |
| `candidates` | `unevaluated_hours` should drain on its own; the supervisor runs the evaluator out of band |
| `emergency stop` / `integrity failure` | both must be false |

Raw streams, if you want them:

```bash
tail -f /Volumes/Brandon_Washington/stratego_phase14/logs/phase14_supervisor.jsonl
tail -f /Volumes/Brandon_Washington/stratego_phase14/logs/phase14_telemetry.jsonl
```

The learner publishes telemetry once per iteration (~20 minutes), so externally
observable state is at most one iteration old. That is the design, not a fault.

---

## 4. Recoverable crash

**The supervisor handles this. Do nothing.** It records the death (exit code or
signal), selects the newest valid hot checkpoint, relaunches through
`start_or_resume()`, and clears its consecutive-restart count once the optimizer
step advances. Backoff grows: 15 s, 60 s, 180 s, 600 s, 900 s.

```text
never create a new 168-hour deadline
```

The resumed run reuses the window persisted in the checkpoint. The supervisor
re-reads that window before every relaunch and stops if it moved.

**If the supervisor itself died** (power loss, someone closed the terminal), run
exactly the launch command again:

```bash
caffeinate -dimsu python scripts/phase14_launch.py
```

It resumes: a valid hot checkpoint is present, so `start_or_resume()` resumes
rather than starts. Do **not** pass any deadline, duration or window argument —
there are none, deliberately.

What a crash costs: sealed games are never regenerated (both rehearsal crashes
replayed **0** games); what is repeated is un-checkpointed optimizer work, which
was 198 s and 232 s in the rehearsal.

The supervisor stops relaunching, and needs a human, when:

```text
an emergency stop is active
the run manifest says training is closed
an unrecoverable integrity failure has been recorded
no valid resume checkpoint exists
the deadline has passed          (one zero-step closeout launch is still made)
5 consecutive restarts made no optimizer-step progress
```

If it stopped for an integrity failure, read it and do not relaunch blind:

```bash
cat /Volumes/Brandon_Washington/stratego_phase14/phase14_integrity_failure.json
```

---

## 5. Worker failure

**Expected and handled.** A killed CPU loader worker raises `BrokenProcessPool`.
The trainer shuts the dead pool down and rebuilds it **at the same cursor**: the
identical minibatch is rebuilt, and the optimizer step, epoch, KL controller and
examples consumed are untouched. Only the processes that packed the bytes differ.

Visible as `loader pool rebuilds` incrementing, with a timestamp and reason.
Capped at **16 rebuilds per run** — past that the run stops rather than hiding a
failing machine behind a healthy-looking log.

Verified at the frozen 2,048-game production population: one of six workers
killed mid-epoch, the learner survived, a fresh pool of six appeared, and the
iteration completed with **1,744 updates — exactly the count of an
uninterrupted iteration**.

No action. If rebuilds climb past two or three in a day, the host needs
attention, not the run.

---

## 6. Storage warning

Frozen policy: **full raw retention**. Deletion is a contingency, pre-authorized
**only** below 120 GiB free, and only for Phase 14 raw shards already marked
consumed and disposable. The run does this itself.

```bash
df -H /Volumes/Brandon_Washington
python scripts/phase14_status.py --json | python3 -c "import json,sys; print(json.load(sys.stdin)['storage'])"
```

Allowed, if free space falls below 120 GiB: nothing manual — the run plans and
executes the rolling deletion under the frozen policy, keeping every checkpoint,
every metric, every historical snapshot and a 1-in-16 sample of deleted shard
ranges.

**Never permitted**, under any pressure: deleting earlier accepted project
evidence (Phase 8/9/10/11/11B/12/13 artifacts, reports, checkpoints) to make
room. If the volume genuinely fills, stop the run and escalate.

Measured rate: **0.2024 GiB/hour** → 34.8 GiB over 168 hours, 17.6× under the
planning rate. This is not expected to happen.

---

## 7. Emergency stop

```bash
python scripts/phase14_emergency_stop.py --reason "why"
```

A *request*, not a kill. The learner finishes the collection unit or optimizer
step in flight, writes a hot checkpoint and exits; the supervisor sees the same
file and does not relaunch.

**Consequences.** Stopping does not stop the clock. Downtime counts against the
168 hours, so every minute stopped is a minute of training lost from the run.

Check, and clear when you are ready to resume:

```bash
python scripts/phase14_emergency_stop.py --status
python scripts/phase14_emergency_stop.py --clear
caffeinate -dimsu python scripts/phase14_launch.py
```

The control surface offers this and nothing else. Learning rate, loss weights,
opponent mixture, setup source, historical-pool algorithm, selection rule,
deadline and checkpoint cadence are refused by name.

---

## 8. Deadline

Automatic, at `run_start_utc + 168 h`. No new collection unit is launched, no
optimizer step begins, the unit in flight is left where it is, a final archive
snapshot is written, the **hour-168 candidate is marked on it**, the manifest is
written with `closed_reason: deadline`, and the process exits 0. The supervisor
sees a closed manifest and stops.

If the learner dies just before the deadline, the supervisor makes exactly one
**closeout launch**: it resumes, observes the deadline has passed, takes **zero
optimizer steps**, and finalizes. Verified nine hours past a real deadline.

The hour-168 candidate is a candidate. It is **not** automatically the deployed
policy.

---

## 9. Post-run

```text
training is closed
the hour-168 candidate is preserved
complete any missing fixed-pack candidate evaluations
apply the frozen checkpoint-selection rule
select the final direct-policy checkpoint
do not train further
```

**9.1 — What is still unevaluated?**

```bash
python scripts/phase14_select_final.py --check
```

**9.2 — Complete every missing evaluation** on the identical frozen pack. No
search. Repeat until `unevaluated_hours` is empty; a failed evaluation preserves
its candidate and re-runs on the same pack.

```bash
python scripts/phase14_evaluate_candidates.py
```

**9.3 — Apply the frozen rule.** It refuses while anything is unevaluated,
because a candidate scored on 40 games is not comparable with one scored on 128.

```bash
python scripts/phase14_select_final.py
```

Rule: highest equal-weight mean EWR across the four strata; tie-break on highest
minimum stratum EWR; then the later candidate hour.

**9.4 — Do not train further.**

---

## 10. After selection

The next project sequence, in order. **None of it is part of Phase 14.**

```text
copy the selected final C1
dedicated Agent-1C-style belief specialization
freeze the final belief provider
rebind Phase 12 search
final machine evaluation
final human evaluation
```

---

## Quick reference

```bash
# launch or resume — the same command
caffeinate -dimsu python scripts/phase14_launch.py

# check without launching
python scripts/phase14_launch.py --preflight-only

# status
python scripts/phase14_status.py

# emergency stop / status / clear
python scripts/phase14_emergency_stop.py --reason "why"
python scripts/phase14_emergency_stop.py --status
python scripts/phase14_emergency_stop.py --clear

# candidate evaluation, out of band
python scripts/phase14_evaluate_candidates.py

# post-run selection
python scripts/phase14_select_final.py --check
python scripts/phase14_select_final.py

# rebuild the launch package after a deliberate code change
python scripts/phase14_build_launch_package.py
python scripts/phase14_build_launch_package.py --verify
```
