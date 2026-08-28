# Phase 17 — Agent 4B
## Narrow conversion to the simplified paper-shaped tandem recipe (operator decision D10)

_Written 2026-08-28. Starting commit `eab8a33`._

Agent 4's tandem system was converted to operator decision D10's
`phase17_simple_paper_tandem_v1`. The fixed-transition runner, current-policy
routing, exact active-game persistence, exports, telemetry and integrity
safeguards were reused, not rebuilt. Nothing here started production.

---

## 1. What changed

Eight conversions, one per D10 required change.

| # | Was | Is |
|---|---|---|
| 1 | no recipe identity; run `RUN-2026-A` | `phase17_simple_paper_tandem_v1`, run `RUN-2026-B`, refused on mismatch |
| 2 | production init unguarded | production refuses an injected setup model; Phase 9 digests + a from-scratch setup model are recorded |
| 3 | adaptive reverse-KL controller (D5) | fixed reverse coefficient `0.1`, named a coefficient everywhere |
| 4 | `alpha = max(0.1·n^-p, floor)`, `p` fitted to `N` (D3) | `alpha(n) = 0.1·n^-0.3` on the shared global iteration, no floor, no `N` |
| 5 | `0.9·alpha·(I/10)` uncentered bonus (D7-B) | the printed advantage `(o − E[v]) + alpha·(I − h)`; `L_h` still targets `I/10` |
| 6 | fixed quota 572 + two-budget warm-up + max-age + backlog alarm | every episode completed in the window, both sides, exactly once, five epochs |
| 7 | pool rebound only when the setup digest moved | 512 per side regenerated at every global iteration; refills from that snapshot |
| 8 | `P1`–`P8` could stop the run | `I1`–`I8` stop; `P1`–`P7` are warnings that never can |

Change 4 also settles Agent 4's carry-forward **`A4-CF6`** (setup-iteration
versus move-iteration indexing): D10 makes `n` the shared global tandem
iteration, so a skipped setup update still advances the anneal.

Deleted rather than left dormant, per D10 section 2's "one unambiguous path":
`stratego/training/phase17/queue.py` (the frozen budget policy),
`SetupKLController`, `scripts/run_phase17_preflight.py` (its throughput,
integration, concentration and refreeze-budget roles are all retired) and
`scripts/run_phase17_setup_gate.py` (the retired standalone setup gate). Their
findings remain in `reports/phase17/agent_0[1-4]_*.json`, which were not
touched.

Not changed, as instructed: the setup architecture, the move learner objective
and schedules, the fixed-transition target logic, policy sampling, the
current-policy rebind, EMA roles, external bundle semantics, and exact
active-game persistence.

## 2. Identities

```text
recipe                     phase17_simple_paper_tandem_v1
production run ID          RUN-2026-B
runner version             phase17_tandem_runner_v2
setup equation version     phase17_setup_update_paper_v1
completed-buffer version   phase17_completed_setup_buffer_v1
joint checkpoint schema    phase17_joint_checkpoint_v2
telemetry schema           phase17_simple_tandem_telemetry_v1
supervisor version         phase17_run_supervisor_v2
export schema              phase17_paired_export_v1   (unchanged)
```

Every schema version above was bumped because its content changed, which is
what makes a `RUN-2026-A` artifact refuse to load into `RUN-2026-B`.

Move start, unchanged and re-verified in smoke check 1:

```text
path                 checkpoints/phase9/selfplay_c1_v1.pt
file_sha256          dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
model_state_digest   f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
```

Run-specific digests are in
`reports/phase17/phase17_simple_tandem_preflight.json` and
`reports/phase17/phase17_simple_tandem_handoff_v1.json`.

## 3. Files changed

```text
modified  stratego/training/phase17/setup_contract.py
          stratego/training/phase17/setup_episode.py
          stratego/training/phase17/setup_learning.py
          stratego/training/phase17/setup_metrics.py
          stratego/training/phase17/runner.py
          stratego/training/phase17/checkpoint.py
          stratego/training/phase17/supervisor.py
          stratego/training/phase17/telemetry.py
          stratego/training/phase17/__init__.py
          scripts/run_phase17_training.py
deleted   stratego/training/phase17/queue.py
          scripts/run_phase17_preflight.py
          scripts/run_phase17_setup_gate.py
added     scripts/run_phase17_d10_smoke.py
          tests/training/phase17/test_simple_paper_recipe.py
```

## 4. Targeted tests

`tests/training/phase17/test_simple_paper_recipe.py` is the single-file D10
conformance module; the rest are updates to the existing suites.

| D10 item | Where |
|---|---|
| fixed KL coefficient and reverse direction | `test_simple_paper_recipe.py`, `test_setup_learning.py` |
| alpha at iterations 1, 2 and later | `test_simple_paper_recipe.py::test_alpha_at_the_iterations_d10_names` |
| printed advantage from recorded behavior fields | `test_simple_paper_recipe.py`, `test_setup_learning.py` |
| Phase 9 move + fresh setup init, checkpoint refusal | `test_runner_tandem.py`, `test_simple_paper_recipe.py` |
| all episodes consumed once, five epochs | `test_setup_learning.py`, `test_runner_tandem.py` |
| one short current-policy tandem iteration | `test_simple_paper_recipe.py` (legal sampled moves, legal oriented setups) |
| checkpoint round trip, no lost/duplicated outcomes | `test_simple_paper_recipe.py`, `test_setup_learning.py` |
| structural absence of prohibited participants | `test_simple_paper_recipe.py`, `test_setup_structure.py`, `test_move_no_search.py` |

## 5. The D10 smoke

One end-to-end run on the production code path at the frozen production shape
(65,536 transitions, population 256, pool 512 per side, MPS for both halves),
under run ID `RUN-SMOKE-D10` with a setup seed different from production's. Its
weights were discarded; `checkpoints/phase17/` is excluded from version control.

```text
role        scripts/run_phase17_d10_smoke.py smoke --iterations 3
duration    291.2 s of a 1,800 s cap
iterations  70.7 / 66.1 / 67.1 s   (mean 68.0 s against the frozen 67.4 s)
peak memory 17.7 GiB
result      9/9 checks passed, 0 warnings raised, 0 stop predicates fired
```

| # | D10 section 6 check | Result |
|---|---|---|
| 1 | exact Phase 9 move identity and fresh setup identity | pass — both start digests reproduce; the setup model equals `build_setup_model(seed)` and has had zero updates |
| 2 | both seats current raw policy, sampled legal actions | pass — 0 illegal actions in 196,608 rows, 0 rows under an unknown policy digest, 98,511 Red / 98,097 Blue; on a 6,000-row sample the drawn action was the distribution's mode only 40.9% of the time, so the draw is categorical rather than argmax |
| 3 | fresh legal, inventory-correct, oriented setups, no fallback | pass — 0 inventory, orientation, legality or fallback violations; pool regenerated at every iteration (0 / 320 / 592 candidates discarded) |
| 4 | exactly the configured transition count | pass — 65,536 harvested and 65,536 rows emitted each iteration, boundary + terminal accounting for all of them |
| 5 | a real completed game drives a five-epoch setup update | pass — 106 / 218 / 225 games finished, 212 / 436 / 450 episodes consumed, 5 epochs and 20 / 35 / 40 optimizer steps, buffer drained to 0 each time |
| 6 | fixed reverse-KL coefficient 0.1, no controller | pass — 0.1 on every iteration and every epoch, no `controller` attribute, telemetry names it `kl_coefficient` and carries no `kl_beta` |
| 7 | alpha equals `0.1·n^-0.3` at the global iteration | pass — 0.1000 / 0.0812 / 0.0719 against the closed form, setup index equal to the tandem iteration each time |
| 8 | checkpoint round trip, no lost or duplicated outcomes | pass — 256 games reseated and restored with their 256 episode pairs, 0 differing identity fields, 1,098 enqueued = 1,098 consumed before and after |
| 9 | no search / belief / historical / handcrafted participant | pass — clean participant ledger, no `stratego.search` or `stratego.policies` loaded, no setup-library import anywhere in the phase17 package |

Targeted tests: **422 pass** in `tests/training/phase17`, up from 386 at
`eab8a33`. The D10 conformance module contributes 19 and the reworked suites the
remainder. The broad repository suite was run once during this work and reported
7,028 passed / 3 skipped, unchanged from the pre-existing baseline; it is not a
D10 deliverable and was not repeated.

## 6. Issues that bear on the 12-hour run

**A4B-1 — the setup EMA landed on the wrong device after a resume. Found, fixed.**
`SetupEMA.load_state_dict` preserved the checkpoint payload's device. A paired
checkpoint serializes the setup EMA to CPU and `read_joint_checkpoint` loads
with `map_location="cpu"`, so a resume on the MPS production device left a CPU
shadow accumulating against MPS parameters and `SetupEMA.update` raised on the
**first setup update after the resume**. This is a defect in the retained
Agent 4/3 foundation, not in the D10 conversion. It survived because Agent 4's
resume rehearsal deliberately ran on CPU — MPS is not bitwise reproducible run
to run, so the exactness proof had to be there — and a CPU-to-CPU restore cannot
show it. The D10 smoke is the first thing to resume on the production device,
and it crashed on the first attempt. The shadow is now bound to the model's
device on load; pinned on CPU and, where a device exists, on MPS.

**A4B-2 — one telemetry row is lost per resume. Found, not changed.**
`TrainingSession.step` writes the checkpoint *before* it appends the telemetry
row, so the checkpoint's `telemetry_position` excludes the row of the iteration
it was taken at. `TelemetryWriter.resume` then truncates that row back and the
next iteration writes into its record slot. Measured in the smoke: the log holds
iterations 1, 2 and 4 at record indices 0, 1 and 2 — iteration 3's row is gone
even though its weights were checkpointed and correctly restored. Training and
checkpoints are unaffected; the cost is that the hour 6–12 learning curve loses
one row per resume and `record_index` no longer tracks the iteration number.
Every row carries `system.iteration`, so a reader realigns by that field. Left
alone because reversing the order is a restructure of the telemetry and
checkpoint machinery the 4B brief says to reuse — an operator call, not one to
make while converting a recipe.

**The printed advantage's entropy term dominates the outcome term.** Measured
across the three smoke iterations: entropy term 2.70 / 2.16 / 1.81 against an
outcome term of 1.00 / 1.00 / 1.00, a ratio of 2.84 / 2.25 / 1.88 falling only
as alpha anneals. This is the deliberate consequence of D10 keeping the paper's
printed `alpha·(I − h)` with `I` in nats while `L_h` targets `I/10`; it is the
opposite balance from the D7-B form it replaces, which was constructed
specifically to keep the two commensurate. D10 says not to add a compensating
scale, so nothing was added — but this is the substantive behavioural difference
between the two recipes and it is what the 12-hour curve will be testing. Every
row carries `setup.advantage_components` so the balance can be read directly.

**Resource exhaustion has no predicate.** D10 section 7 names it a stop
condition. There is no threshold to freeze and the runner has no resource
monitor, so it surfaces as a process-level failure, unchanged from Agent 4. It
is recorded in the handoff's `stop_policy.not_wired` rather than invented.

**The setup batch now tracks the arrival rate rather than a fixed quota.**
Under D10 the batch is whatever completed in the window: 212 → 436 → 450
episodes as the population reached steady state, giving 20 → 35 → 40 optimizer
steps. Agent 4's carry-forward `A4-CF4` measured game length falling 288 → 167
plies over 200 iterations, i.e. arrivals rising about 68%, so the batch and the
step count will grow through the run. This is not a stop and not a starvation
risk — the retired quota's overflow mode is gone with it — but it does mean the
setup half's cost is no longer pinned. Measured here at 0.61–0.87 s of a ~68 s
iteration, about 1%, so even a doubling stays negligible against the frozen
`N = 640`.

## 7. What was not established

- No strength claim of any kind. No benchmark lane was evaluated and no
  candidate was scored.
- The external 30-minute round trip. That is Agent 5's h0 handshake and it has
  not been performed.
- Setup diversity, entropy or concentration behaviour over 12 hours. The
  descriptive reading is wired and taken, but three iterations say nothing
  about it, and under D10 it could not gate anything anyway.
- Bitwise continuation equivalence across a resume **on the production device**.
  Smoke check 8 asserts identity and outcome accounting on MPS, which is what
  D10 asks for; the bitwise claim is proved on CPU by
  `test_runner_tandem.py::test_a_round_trip_reproduces_the_next_iteration_exactly`,
  because MPS is not bitwise reproducible run to run.
- Whether the printed advantage's entropy dominance helps or hurts. That is the
  experiment.

`ready_for_short_launch_check: true` in
`reports/phase17/phase17_simple_tandem_handoff_v1.json`. Production was not
started and Agent 6 was not begun.
