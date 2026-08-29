# Phase 17 — Agent 4C

## Narrow attribution and resume correction

**Correction commit `67b186a`.** Evidence parent `eab8a33`, implementation
commit `3be8bba`. Final production source-closure digest
`4aa5c3504f775f6b6963888ea74adb6e56c4d6a2cc5da6700955cbd79c349cc5` over 22
files. `ready_for_launch_freeze: true`.

Nothing in the recipe moved: the move model, setup model, optimizer equations,
schedules, sampling, transition budget (65,536), setup epochs (5), population
and every D10 constant are byte-identical to `3be8bba`. No run was started.

---

## 1. What was wrong, and what it would have cost

Six defects, in the order the instruction lists them.

### 1.1 Production would have run with no source identity at all

`TrainingSession.__init__` took `source_digest: str = ""` and `main()` never
passed one. The empty string was then written into every paired checkpoint,
every one of the 25 candidate manifests, and the resume comparison.

That is the dangerous shape of this defect: an empty digest is not a missing
check, it is a check that **passes**. `read_joint_checkpoint` compares the
stored digest to the process's, and `"" == ""`. Production would have produced
a full 12-hour candidate set that satisfied every identity check it had while
being unable to say which program made it.

There was a second, quieter half. The only closure that existed was a
hand-written tuple of eleven paths inside the smoke script:

| | pre-correction closure | corrected closure |
|---|---|---|
| files | 11 | 22 |
| `move_trainer.py`, `move_loss.py`, `move_start.py`, `move_snapshot.py` | **absent** | present |
| `transition_collector.py`, `transition_targets.py`, `transition_schema.py` | **absent** | present |
| `setup_model.py`, `setup_sampling.py`, `__init__.py` | **absent** | present |
| `run_phase17_d10_smoke.py` | **present** | deliberately absent |

So a change to the move learner did not move the source identity, and a change
to a smoke script did. Both directions were wrong.

### 1.2 A resume deleted the telemetry row of the iteration it resumed from

`step()` writes the checkpoint and then appends the row, because the row
carries the checkpoint's verified identity and that identity does not exist
until the file has landed. But the checkpoint recorded the **pre-append**
position, so `TelemetryWriter.resume` truncated the following row back as
excess. Agent 4B measured it: the smoke's log held iterations 1, 2, 4 at record
indices 0, 1, 2 — iteration 3's row deleted although its weights had been
checkpointed and restored.

D11 makes the 25-candidate learning curve the entire deliverable of the run.
One row lost per resume, with `record_index` no longer tracking anything, is
damage to exactly the artifact the run exists to produce.

### 1.3 Every resume skipped a checkpoint generation

`resume()` set `self.parent_checkpoint = payload["parent_checkpoint_identity"]`
— the loaded checkpoint's *own parent*. So generation g+1 linked to g−1, and
any paired export written after the resume bound the same wrong parent. A
reader walking the chain could not distinguish a resumed run from one that
never wrote generation g.

### 1.4 Four predicates were measured on a cadence and written nowhere

`_cadence_guards` runs after `run_iteration` has returned, so the P4/P5/P6/P7
verdicts it produces were never in `result.verdicts` and reached no telemetry
row. They could trip on every iteration of the 12-hour run and the log would
never mention it.

Fixing the wiring exposed a second problem underneath it. The row partitioned
verdicts on `fired`, but P5 and P7 have `consecutive_required = 1` — a tripped
warning **is** a fired verdict. A tripped P5 would therefore have appeared in
`system.stop_predicates`, a row claiming the run was stopping over something
D10 section 7 says can never stop it. The partition is now by consequence, on
the `stops_the_run` field each verdict already carries.

Separately, P6's first-hour move-entropy samples lived on the session object,
which a resume rebuilds from nothing. A run resumed inside its first hour would
have fixed the collapse floor from the post-resume readings alone; one resumed
after it would have started collecting a second, different baseline. Both
samples and the closed flag now live in the supervisor's persisted state.

### 1.5 A finished game's outcome could be dropped with a counter incremented

`_enqueue` appended refusals to `enqueue_rejections` and the window carried on.
Every refusal reason — incomplete, duplicate, already consumed — means a played
game's result will never reach the setup learner, which is D10 section 7's
"loss/duplication of setup outcomes". A run could have completed 12 hours with
a setup half trained on a different set of games than its telemetry described.

### 1.6 The v1 handoff attributed the work to the wrong commit — and froze the wrong config

`identities.commit` and `integration_base` both named `eab8a33`, the commit the
D10 conversion started *from*, so the entire 4B implementation was attributed
to its parent.

While recomputing the identities I found a third one v1 got wrong, not named in
the instruction. `identities.config_digest` was
`ee19d627…` — that is the **smoke session's** config (`RUN-SMOKE-D10`, setup
seed +1000), taken from `preflight["config_digest"]`. Production's is
`4e7abad8…`. Agent 6 must freeze the production value; v1's would have bound
the launch manifest to a config the run never uses.

---

## 2. What changed

New module `stratego/training/phase17/source_identity.py`. The closure is
**enumerated, not listed**: `PRODUCTION_SOURCE_ROOTS` names
`stratego/training/phase17` and `scripts/run_phase17_training.py`, and
directories expand to their sorted `*.py`. A module added later is covered
without anybody remembering to add it. The smoke stays outside on purpose —
production never loads it, and a source identity that moved when a smoke was
edited would force Agent 6 to refreeze for a change that cannot reach the run.

There is **no digest literal in the module**, and a test enforces that. Agent 6
freezes the value printed by `--describe`; Agent 7 passes it back on
`--source-digest`; `--start` re-hashes the tree and refuses anything else. The
check is "these bytes are the bytes the operator authorized", not "these bytes
are the bytes that were here when this file was written".

The telemetry fix keeps the existing ordering rather than reversing it. The
checkpointed position now names the row it is about to write
(`pending_row_iteration`), and the resume adopts **exactly one** row past the
offset, only if it is a complete line with the right run id, the right
`record_index`, and that iteration. Everything else past the offset is
truncated exactly as before. Reversing the order instead would have cost the
row its checkpoint identity — the thing that makes the row worth keeping.

`SUPERVISOR_VERSION` moves `v2 → v3`: the state document gained the first-hour
sample set, and `I7` widened to cover a lost or duplicated setup outcome. A v2
document is **refused**, not defaulted — a state that cannot say whether the
first hour closed would resume as "not closed" and rebuild a different baseline.

The production command now has a shape Agent 6 can bind:

```bash
.venv/bin/python scripts/run_phase17_training.py --describe
```

which prints the closure, the digest, and the exact command to freeze:

```text
nohup caffeinate -dimsu .venv/bin/python scripts/run_phase17_training.py \
  --run-id RUN-2026-B --start --i-am-agent-7 \
  --source-digest 4aa5c3504f775f6b6963888ea74adb6e56c4d6a2cc5da6700955cbd79c349cc5 > <log> 2>&1 &
```

`--start` without `--i-am-agent-7` exits 2. Without `--source-digest`, or with
one the tree does not reproduce, exits 3 naming both digests.

---

## 3. Verification

Targeted only, as instructed. **29 new tests** in
`tests/training/phase17/test_agent_4c_corrections.py`, one per required proof
plus the edge cases each correction opens:

| required proof | tests |
|---|---|
| an empty production source digest is refused | 6 |
| source identity changes the run identity and survives checkpoint/export/resume | 2 |
| checkpoint → append → resume retains every checkpointed row exactly once | 5 |
| the next checkpoint after resume links to the loaded checkpoint | 2 |
| a cadence warning appears in the corresponding telemetry row | 2 |
| the first-hour entropy baseline survives resume | 3 |
| a rejected completed setup episode arms the integrity stop | 3 |

The Phase 17 training package is **451 pass** (422 before). I ran the whole
package rather than only the individual affected tests — wider than the
instruction's economy asked for, and reported here as run. Not run, as
instructed: the production-shaped D10 smoke (either role), the repository
suite, any setup soak, any strength evaluation.

**Not established.** No strength claim; no benchmark lane was evaluated. The
D10 smoke was not rerun at production shape after these corrections. The resume
proofs are CPU — MPS is not bitwise reproducible run to run, so a resume
comparison there cannot distinguish a defect from the device.

---

## 4. Two things carried forward, deliberately unchanged

**The D10 advantage scale.** The printed advantage puts `alpha * (I - h)` in
nats against an outcome term bounded by 2, while `L_h` targets `I/10`. D10
section 4 forbids a compensating scale, floor, centering rule, horizon map or
controller, so it ships uncompensated. The component magnitudes are recorded
per iteration in `setup.advantage_components`; the ratio is part of the
experiment's reading, not a defect to fix here.

**A resume clears an armed stop.** `load_state_document` restores each
predicate's `consecutive` and `trips` but not the supervisor's `stopped`
record, so resuming from a checkpoint taken at an armed I7 starts the run
again. That is arguably right — a resume is a deliberate operator act after a
repair — but it is behavior, not an accident, and it is recorded here rather
than changed inside a narrow correction.

---

## 5. Handoff

```text
reports/phase17/agent_04c_report.md
reports/phase17/phase17_simple_tandem_handoff_v2.json
```

`ready_for_launch_freeze: true`. Production cannot start without an exact,
non-empty source identity that is re-hashed and compared before anything is
written under the run's name; that identity is bound into the run digest, every
paired checkpoint and every candidate manifest, and re-checked on resume;
cadence exports remain immutable; and a resume can no longer silently alter the
learning curve.

Agent 6 must bind correction commit `67b186a` — not `eab8a33`, not `3be8bba` —
and production config digest `4e7abad8…`, not v1's smoke digest.

The unrelated working-tree paths (the operator plan edits, Agent 5's evaluator,
benchmark, report and retired transport work, `data/phase17/`) were preserved
exactly and are neither staged nor committed by 4C.
