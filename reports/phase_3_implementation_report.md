# Phase 3 Implementation Report

Frozen reference: `phase2_1_reference_1.1.0`  
Rules: `stratego_project_v1`  
Observation: `observation_v2_1_127ch`

## 1. Agent 1 — Batch Wrapper and Reference Equivalence

### 1.1 Status

**PASS**

### 1.2 Implementation summary

`stratego/training/batch_simulation.py` adds a correctness-first, single-process
batch wrapper around the frozen reference engine. `stratego/engine/` was not
modified, and the wrapper contains no simulation logic: every rule decision,
observation channel, event and terminal reason is produced by the engine, so a
slot is exactly a `GameState` that the wrapper happens to hold.

`BatchSimulator(num_environments, root_seed=..., rules=...)` owns `N` independent
games and exposes the bulk-synchronous cycle the rest of Phase 3 is built on:
read observations/legality for every active slot, choose one action per slot,
apply them all, collect terminal results, reset the finished slots the caller
selects.

Required operations, all implemented:

- creating `N` games and deterministically seeding each slot;
- acting player per slot (`acting_players`, `NO_ACTING_PLAYER = -1` for a
  finished slot);
- observations, stacked as `(n, 127, 10, 10)` `float32` (`observations`) or per
  slot for either observer (`observation`);
- legal-action lists (`legal_action_lists`) and dense 10,000-entry `uint8` masks
  (`legal_action_masks`), both generated from the engine's own functions;
- one action per active slot per step (`step`), accepting either a
  `{slot: action_id}` mapping or a dense length-`N` vector in which a negative
  entry means "leave this slot alone";
- terminal result and reason (`SlotOutcome`, plus `BatchStepResult.newly_terminal`
  and `.outcomes` for the games that finished during a step);
- independent reset of selected or finished slots (`reset_slots`,
  `reset_finished`) with slot identity preserved.

Slot identity follows section 16 of `03_game_engine_spec.md`. `environment_id` is
fixed for the life of the slot; `generation` starts at `0` and increments by
exactly one per reset, so `(environment_id, generation)` names one game and a
trajectory record cannot cross a reset boundary. `reset_slots` builds a whole new
`EnvironmentSlot` from `derive_slot_seed(root_seed, environment_id, generation)`,
so no board, counter, recent move, behavioural event, knowledge flag, event log
or action history can survive a reset by construction rather than by clearing
fields.

Two decisions worth recording because Agent 2 inherits them:

1. **Illegal actions abort the whole batch step.** `step` validates every
   submitted action before it applies any of them, so a rejected step leaves the
   entire batch bit-identical, not merely the offending slot. This is stronger
   than the engine's own per-state atomicity and matches the "fail loudly"
   policy of `03_game_engine_spec.md` section 19. `BatchIllegalActionError` and
   `BatchTerminalStateError` also subclass the engine's `IllegalActionError` and
   `TerminalStateError`.
2. **Slot seeds are hashed from `(root_seed, environment_id, generation)`
   alone.** Any process can therefore rebuild any slot generation without
   replaying its neighbours, which is what Agent 2 needs inside a worker.

The wrapper also exposes the privileged and serialisable extras later agents
need, so Agent 3 should not have to change this interface: `belief_targets`
(training target only, never part of `observations`), `snapshot`,
`replay_record` (carrying the slot seed and identity in `seeds`),
`public_board`, `public_setup`, `public_events` and `render`. Legal-action lists
are cached per state and invalidated on every transition and reset; caching a
pure function of the state cannot change behaviour and avoids regenerating the
same list three times per batch step.

### 1.3 Files created / modified

Created:

```text
stratego/training/__init__.py
stratego/training/batch_simulation.py
tests/training/__init__.py
tests/training/differential.py
tests/training/test_batch_simulation.py
scripts/run_phase3_agent01.py
reports/phase_3_data/agent_01_batch_equivalence.json
reports/phase_3_implementation_report.md
```

Modified: none. `stratego/engine/` is untouched.

`tests/training/differential.py` holds the reference-side game builder and the
batch/reference comparator. The fast pytest suite and the acceptance harness both
import it, so the scaled-down and full-scale gates cannot drift apart.

### 1.4 Test summary

- `python -m pytest -q`: **1,292 passed, 0 failed, 0 skipped** (1,255 frozen
  Phase 2.1 tests plus 37 new batch tests). The Phase 2.1 suite was confirmed
  green before implementation began.
- `tests/training/test_batch_simulation.py` covers construction and slot
  identity, seed determinism and slot-local seeding, reproducibility from
  `root_seed`, equality with independently built reference games, acting players,
  stacked observations, list/mask agreement, explicit slot selection, unknown-slot
  rejection, single- and dense-form stepping, refusal to step a finished slot,
  terminal result/reason exposure including both limit-driven draw reasons,
  independent reset with byte-identical neighbours, generation and trajectory-key
  semantics, "nothing carried over" after reset, illegal-action inertness across
  several categories, and the privileged extras (belief targets, snapshot
  round-trip, replay reproduction, public views hiding unresolved identities).
- It also runs the differential comparator at small scale (four runs at batch
  sizes 1, 8, 16 and 32, roughly 3,600 comparisons) so a regression surfaces in
  the ordinary test run rather than only in the acceptance harness.
- `python scripts/run_phase3_agent01.py` is the full acceptance gate and exits
  non-zero unless every threshold below is met.

### 1.5 Key measured results

Single process, no multiprocessing, `Darwin 25.5.0 arm64`, Python 3.13.2,
NumPy 2.5.1, total harness wall clock 113.4 s.

| Gate | Requirement | Measured |
|---|---|---|
| Differential comparisons | >= 100,000 | **130,176** |
| Equivalence mismatches | 0 | **0** |
| Independent reset trials | pass | **320**, 0 mismatches |
| Generation errors | 0 | **0** |
| Illegal-action inert trials | pass | **4,487**, 0 failures |
| Phase 2.1 + new tests | all pass | **1,292 / 1,292** |

Batch sizes tested: 1, 8, 64 and 256, the last being the order of the Phase 3
engineering point. Every state a tested action produced was compared exactly
once, on: history-free state fingerprint, acting player (from the state and from
the batch API's dense vector), legal-action list, dense mask (both the per-slot
and the stacked read), both players' observations, both players' belief targets,
both players' public board views, and terminal reason/result. Each finished game
additionally compared the complete `include_history=True` fingerprint, which
covers the whole derived event log and action history, plus both players'
observer-filtered public event streams. Every step compared the derived events
the action emitted and their filtered forms.

Situation coverage inside the differential run: 122,176 ordinary moves, 8,000
Scout multi-square moves, 9,806 combats, 14,373 identity reveals, all five
behavioural event types (`declined_attack` 145,965, `threat` 18,506, `evade` 991,
`protect` 780, `was_protected` 780), 222 completed games and 222 resets.

Reset isolation used a batch held at substantially different plies (mean ply
spread 58.6, maximum 207): only the selected slots were driven to terminal and
reset, and all 1,920 full-history fingerprints of the unaffected slots were
identical before and after. Each reset incremented `generation` exactly once,
kept `environment_id` fixed, produced a game equal to an independently built
reference game for the new generation, and yielded 352 distinct
`(environment_id, generation)` keys with no reuse.

Illegal-action inertness covered twelve categories -- lake destination, standing
still, diagonal step, onto an own piece, two squares with a non-Scout, an
immovable Flag/Bomb, the opponent's piece, an empty source square, an identifier
above the 10,000-entry space, a negative identifier, an unknown slot index and a
finished slot. Each trial submitted the bad action alongside legal actions for
every other active slot; the step was rejected and the whole batch was unchanged
in all 4,487 trials, with 48 additional whole-batch full-history checks.

Throughput was roughly 1,510-1,620 fully-compared state/action pairs per second,
but that is the cost of the *comparison harness* (four observation builds, four
public views and two engine transitions per comparison), not a batch-wrapper
throughput measurement. Agent 2 owns the performance numbers.

### 1.6 Deviations and limitations

- Per-state fingerprints use `state_fingerprint(include_history=False)` and the
  full `include_history=True` fingerprint is compared once per finished game
  rather than at every ply. Hashing a growing event log at every ply is quadratic
  in game length; the per-step event comparison plus the per-game full-history
  comparison covers the same data. The in-loop reset-isolation check inside the
  differential run likewise uses history-free fingerprints, while the dedicated
  reset trials use full-history fingerprints.
- Only two terminal reasons occurred naturally in the 222 completed games:
  `flag_capture` (191) and `battleless_move_limit_draw` (31). The unit tests
  force `absolute_move_limit_draw` and `battleless_move_limit_draw` through tight
  rules configurations. `opponent_no_legal_move` and `both_no_legal_move_draw`
  need a constructed position rather than a seeded setup, so they remain covered
  by the engine's own terminal-condition tests; the batch layer copies whatever
  reason the engine set and that copy is compared at every terminal state.
- `acting_players` reports `-1` for a finished slot. The engine leaves
  `GameState.acting_player` naming the last mover after termination, which is not
  a player to move; the raw engine value is still compared and is part of the
  fingerprint.
- The engine always appends to `GameState.events`, so a long-running slot
  accumulates its own event log. That is engine behaviour and was not changed,
  but it is a memory consideration for 1,024+ persistent slots.
- No multiprocessing, shared memory, MPS or trajectory persistence, as specified.
  Slot seeding, per-slot rebuildability and the stacked buffer layouts were chosen
  so Agent 2 does not need an interface change.

### 1.7 Data files

```text
reports/phase_3_data/agent_01_batch_equivalence.json
```

Contains every required field plus the per-batch-size differential runs,
illegal-action category counts, reset statistics and coverage counters. No
reproduction file was written because there were no mismatches.

### 1.8 Handoff notes for Agent 2

- Agent 1 is `PASS`; the prerequisite is satisfied.
- Treat these as frozen batch semantics: fixed `environment_id`, `generation`
  incremented exactly once per reset, `(environment_id, generation)` as the
  trajectory key, all-or-nothing rejection of a batch step, and slot seeds
  derived only from `(root_seed, environment_id, generation)`.
- `derive_slot_seed` and `slot_game_id` are the reproducibility contract. A
  worker can rebuild any slot generation locally with
  `tests/training/differential.reference_game`, which is also how the shared
  memory path should be validated.
- Buffer layouts already match the shapes Agent 2's instructions ask for:
  observations `(N, 127, 10, 10)` `float32` from `observations`, dense masks
  `(N, 10000)` `uint8` from `legal_action_masks`, acting players as `int8` with
  `-1` for a finished slot, and a dense action vector of length `N` in which a
  negative entry skips the slot. Copying those arrays into shared memory needs no
  serialisation transform.
- Reuse `tests/training/differential.run_differential` and
  `compare_state`/`compare_step_events` for the cross-process equivalence gate
  rather than writing a second comparator.
- `belief_targets` is privileged. It must not be written into any model-facing
  shared buffer.
- Worker processes should not share a `BatchSimulator`; give each worker its own
  simulator over a disjoint `environment_id` range and keep the seed derivation
  global so slot identity stays unique across the whole run.

## 2. Agent 2 — Shared Memory and CPU Scaling

### 2.1 Status

**PASS**

### 2.2 Implementation summary

Agent 1's batch semantics now run across processes:

```text
coordinator process
    |
persistent shared-memory arrays      stratego/training/shared_buffers.py
    |
CPU simulation workers               stratego/training/worker_pool.py
```

`stratego/engine/` is still untouched, and the worker still contains no
simulation logic: each worker owns one `BatchSimulator` over a disjoint,
contiguous slot range, so a slot is exactly the `GameState` Agent 1 validated.

**Shared buffers.** One `multiprocessing.shared_memory` block, allocated once
per run and never resized, holds every field 64-byte aligned and field-major, so
`observations` is a genuine C-contiguous `(N, 127, 10, 10)` `float32` array and
`legal_mask` a genuine `(N, 10000)` `uint8` array. Both can be wrapped with
`torch.from_numpy` without a copy when Agent 4 arrives. Alongside the required
payload the block carries the terminal/status metadata a coordinator needs to
apply a reset policy and record results without ever receiving an object:
`status`, `terminal`, `ply`, `battleless_moves`, `worker_id`,
`publish_sequence`, `episode_count` and a `last_*` group holding the outcome of
the most recently finished game.

**No lock, by construction.** Every field has exactly one writer: workers write
the published state, the coordinator writes `actions` and `reset_request`, and a
slot belongs to one worker for the life of the pool. Within a bulk-synchronous
phase the two sides never write at the same time.

**No object-payload queues.** Only small fixed-shape dictionaries cross a pipe:
a command name, a sequence number, two flags, and a reply of counters and
timings. Measured at the Phase 3 engineering point, all control traffic for one
phase is smaller than a *single* observation. Reset requests travel as a shared
`uint8` column, so resetting 2,048 slots costs the same as resetting one.

**Failure surface.** The coordinator waits on the control pipes and the process
sentinels together. A worker that exits is reported as `WorkerCrashError`
naming the slot range that stopped being simulated, one that raises returns its
remote traceback as `WorkerFaultError`, and one that stops responding hits the
phase timeout as `WorkerTimeoutError`. After every phase the coordinator checks
that each slot's `publish_sequence` advanced; a slot that was not republished
raises `StaleBufferError` rather than being read. Production recovery is
deliberately not implemented.

**Determinism.** There is no process-local randomness. Every game still comes
from `derive_slot_seed(root_seed, environment_id, generation)`, so slot content
depends on identity alone — not on which worker holds the slot, how many workers
exist, or the order phases complete in. This is verified rather than asserted:
running the same seed under 2, 3 and 8 workers produces byte-identical buffers.

**Thread oversubscription.** Workers are pinned to single-threaded numerical
libraries through the five standard variables, set before the children are
spawned and restored in the coordinator afterwards. The workers report what they
actually see in their own environment (`1,1,1,1,1`), so this is measured rather
than assumed.

One backward-compatible interface extension was needed and is described in
2.6.

### 2.3 Files created / modified

Created:

```text
stratego/training/shared_buffers.py
stratego/training/worker_pool.py
tests/training/test_shared_buffers.py
tests/training/test_worker_pool.py
scripts/run_phase3_agent02.py
reports/phase_3_data/agent_02_shared_memory_scaling.json
reports/phase_3_data/agent_02_shared_memory_scaling_raw.csv
```

Modified:

```text
stratego/training/batch_simulation.py   (one keyword argument, see 2.6)
reports/phase_3_implementation_report.md (this section)
```

### 2.4 Test summary

- `python -m pytest -q`: **1,340 passed, 0 failed, 0 skipped** — Agent 1's 1,292
  plus 48 new tests. Agent 1's suite was confirmed green before and after the
  `batch_simulation.py` change.
- `tests/training/test_shared_buffers.py` (18 tests) covers the required shapes
  and dtypes, non-overlapping aligned layout, contiguity and non-ownership of
  the bulk views, single-writer field ownership, fill values, descriptor size
  and picklability, view semantics and range rejection, version and
  missing-block rejection, owner-only unlink, staleness detection, the
  isolation-snapshot helpers, and terminal-reason code round trips for every
  frozen reason.
- `tests/training/test_worker_pool.py` (30 tests) covers partitioning including
  uneven splits, the backward-compatible `first_environment_id` extension, the
  deterministic policy in both its dense-mask and legal-list forms, multiprocess
  equivalence against the single-process wrapper, partition independence,
  globally unique environment identifiers, requested and automatic reset
  isolation, outcome reporting, control-message size, single allocation, phase
  accounting, thread limits, and all four failure paths (killed worker, hung
  worker, worker exception, stale buffer).
- `python scripts/run_phase3_agent02.py` is the acceptance gate and exits
  non-zero unless every threshold in 2.5 is met. Full run: **274.3 s**.

### 2.5 Key measured results

`Darwin 25.5.0 arm64` (Apple M4 Pro, 10 performance + 4 efficiency cores,
48 GB), Python 3.13.2, NumPy 2.5.1, `spawn` start method, no PyTorch and no
Metal.

| Gate | Requirement | Measured |
|---|---|---|
| Cross-process environment steps | >= 25,000 | **30,272** |
| Equivalence mismatches | 0 | **0** |
| Reset events across workers | >= 5,000 | **5,120** |
| Reset mismatches / generation errors | 0 | **0 / 0** |
| Worker failure detection | detected | **3 of 3 cases** |
| Deadlocks | 0 | **0** |
| Benchmark errors | 0 | **0** |
| Test suite | all pass | **1,340 / 1,340** |

**Equivalence.** Three configurations (4 workers/32 environments, 8/128, 12/256)
were driven against reference `GameState`s built and advanced by the frozen
engine alone. Every one of 30,272 steps compared the shared-memory row against
the reference: acting-player observation, dense 10,000-entry mask, legal count,
acting player, environment identifier, generation, ply, battleless counter and
terminal flag; 33,376 row comparisons in total. Each of the 30,272 decisions
also checked that the action the coordinator picked *from the dense mask* equals
the action the reference picks *from its legal-action list*, which proves the
mask carries the legality the engine generated and that both sides then make the
same transition. All 58 completed games compared terminal reason, winner, draw
flag, final ply, both players' results and outcome generation. One configuration
ran with the coordinator-driven reset policy rather than worker auto-reset, so
the published terminal row was compared too.

**Reset isolation.** 5,120 reset events over 160 rounds at 256 environments and
8 workers, spread by a coprime stride so every round touched all 8 workers. The
batch was first staggered to a ply spread of 209 by holding slots back with the
dense action vector's skip entry, and rounds ran at a mean spread of 24.9 and a
maximum of 216. Each round took an independent copy of every field of all 224
untouched slots and required them byte-identical afterwards: 35,840 isolation
checks, zero changes outside `publish_sequence`. Each reset slot was checked for
exactly one generation increment, an unchanged environment identifier, a
restarted ply and battleless counter, non-terminal active status, at least one
legal action, and — the decisive check — an observation and mask equal to an
independently rebuilt game for its *new* generation. 5,376 distinct
`(environment_id, generation)` keys, no reuse.

**Worker failure.** A `SIGKILL`ed worker, a worker stalled past the phase
timeout and a worker raising on a malformed command were each reported as a
distinct, named infrastructure error naming the affected slot range. No case
returned stale buffers and none deadlocked.

**CPU scaling.** All 25 worker/environment pairs were screened, then the best
three re-measured for 45 s each. Positions/second (screen):

| workers | 256 | 512 | 1,024 | 1,536 | 2,048 | cores busy @1,024 |
|---|---|---|---|---|---|---|
| 4 | 50,371 | 49,238 | 49,170 | 49,738 | 49,506 | 3.63 |
| 6 | 69,095 | 69,412 | 69,305 | 69,896 | 70,008 | 5.14 |
| 8 | 84,690 | 87,539 | 87,653 | 85,651 | 84,751 | 6.51 |
| 10 | 94,192 | **97,985** | 96,845 | 95,976 | 94,679 | 7.45 |
| 12 | 85,106 | 87,177 | 87,772 | 86,593 | 89,772 | 7.28 |

Best sustained configuration: **10 workers x 1,536 environments = 91,482
positions/second** over 45 s, which is also 91,482 state transitions/second and
91,482 observation builds/second — this layer performs exactly one transition,
one dense mask and one observation per position. 10.9 microseconds per position,
13.5 ms mean step latency, 15.1 ms at the 95th percentile. The other two 45 s
measurements (10x512 and 10x1,024) landed within 0.7 percent, at 4.45 ms and
9.04 ms mean latency.

Two clear findings:

1. **Worker count is the throughput knob; environment count is not.** Across a
   factor of eight in environments, throughput moves by under 3 percent at every
   worker count. Environment count buys step latency and memory, not positions
   per second. Throughput peaks at 10 workers — the number of performance cores
   — and *falls* at 12, where workers spill onto efficiency cores and contend
   with the coordinator.
2. **Do not extrapolate from the single-process baseline.** Phase 2.1's
   component rates compose to roughly 16,850 positions/second in one process.
   Multiplying that by 10 workers would predict about 168,500; the measured
   sustained figure is 91,482, or 54 percent of the naive extrapolation. Per
   busy core the pipeline reaches 12,101 positions/second, 72 percent of the
   single-process composite, the remainder being shared-memory writes, phase
   synchronisation and efficiency-core scheduling.

Where the time goes at the best configuration: workers active 73.3 percent of
wall time, coordinator waiting on workers 79.8 percent, coordinator dense-mask
action selection 19.5 percent, barrier/straggler spread 9.6 percent, 7.56 cores
busy out of 14 (54.0 percent machine utilisation). The coordinator's 19.5
percent is serial dead time for the workers in a strictly bulk-synchronous loop
and is the largest single recoverable inefficiency measured.

**Memory.** The shared block is 60,861 bytes per environment: 15.6 MB at 256
environments, 62.3 MB at 1,024 and 124.6 MB at 2,048. Peak measured total
(shared block plus coordinator peak resident set plus the sum of worker peak
resident sets) was 1,477 MB at 10 workers x 1,536 environments. Worker startup
including process spawn, construction of every game and the first publish of
every slot was 0.09-0.16 s across all 28 benchmark runs.

### 2.6 Deviations and limitations

- **One backward-compatible interface extension.** `BatchSimulator.__init__`
  gained `first_environment_id: int = 0`, so a worker's simulator can own a
  disjoint `environment_id` range while the seed derivation stays global. The
  default reproduces Agent 1's behaviour exactly, no other batch semantics
  changed, and a test asserts an offset window is slot-for-slot identical to the
  matching range of one whole batch.
- **A terminal slot publishes zeros, not an observation.** A terminal state has
  no player to move, so there is no acting-player perspective to publish.
  `status` says `terminal` and `acting_player` is `-1`. Terminal *results* are
  published separately in the `last_*` fields, which a reset does not overwrite,
  so an outcome cannot be lost to an immediate reset.
- **`legal_mask` is stored as `uint8`**, matching Agent 1. The coordinator reads
  it through a zero-copy `bool` reinterpretation of the same bytes, which takes
  NumPy's boolean `nonzero` path and is about four times faster than scanning
  the identical memory as `uint8`. This was worth 60,000 -> 90,000 positions per
  second; the stored dtype and the published contract are unchanged.
- **`publish_sequence` advances on every slot every phase**, so the reset
  isolation check excludes it by design. That is the field's purpose: it is how
  a slot that a worker stopped maintaining becomes detectable.
- **Only the acting player's observation is published.** Both players'
  observations and the privileged `belief_targets` stay inside the worker. This
  matches the required `[N, 127, 10, 10]` payload and keeps hidden information
  out of the model-facing transport.
- **Worker failure is detected, not recovered**, as specified. A crashed pool
  must be shut down and rebuilt; nothing restarts a worker or rebuilds its slots.
- **Terminal reasons observed naturally** were `flag_capture` (57) and
  `battleless_move_limit_draw` (1). The other three need constructed positions
  and remain covered by the engine's own terminal-condition tests; this layer
  copies whatever reason the engine set and that copy is compared at every
  terminal state.
- **The benchmark excludes model inference and trajectory recording**, as
  specified. It uses a cheap deterministic legal-action policy, so the numbers
  cover observation building, legality generation, stepping, independent reset,
  shared-memory transport and worker synchronisation only.
- **No overlap between the coordinator and worker phases.** The loop is strictly
  bulk-synchronous, so the coordinator's action selection is dead time for the
  workers. Pipelining is Agent 5's ground.
- **Peak memory is an upper bound.** It sums per-process peak resident sets that
  did not necessarily occur at the same instant.
- **`--quick` mode** runs the same code paths at reduced scale and does not meet
  the acceptance thresholds; only the full run does.

### 2.7 Data files

```text
reports/phase_3_data/agent_02_shared_memory_scaling.json
reports/phase_3_data/agent_02_shared_memory_scaling_raw.csv
```

The JSON holds every required field plus the full field-by-field buffer
documentation (shape, dtype, writer, bytes, meaning), per-configuration
equivalence runs, reset statistics, the failure-case records, and the complete
25-point screening matrix with the 45 s measurements. The CSV is one row per
benchmark run with every latency, wait-fraction, CPU and memory column. No
reproduction file was written because there were no mismatches.

### 2.8 Handoff notes for Agent 3

- Agent 2 is `PASS`; the prerequisite is satisfied.
- **The coordinator has no game object, and it must stay that way.** Trajectory
  and snapshot records have to be produced inside the worker that owns the slot.
  Sending a state or an observation back through a pipe would undo the transport
  property this agent established. Add a worker command, not a return payload.
- `(environment_id, generation)` is available in shared memory every phase, and
  `episode_count` plus the `last_*` fields report a finished game without a
  round trip. `collect_finished` is the helper.
- A slot is still rebuildable anywhere from `(root_seed, environment_id,
  generation)`. `environment_id` equals the global slot index, so a worker's
  local slot `i` is global slot `assignment.start + i`.
- The coordinator's write surface is exactly `actions` and `reset_request`. A
  negative action skips a slot, which is also how the reset stage staggers plies
  and how Agent 3 can hold a slot still while it records something.
- `publish_sequence` is the staleness contract; keep checking it if you add
  phases.
- **Leave cores for the Metal coordinator.** Throughput peaks at 10 workers on
  this machine and already regresses at 12 with a nearly idle coordinator. Agent
  4's model will want a performance core, so the end-to-end optimum is likely to
  be below 10 workers — measure it rather than inheriting 10.
- The coordinator's dense-mask action selection is 19.5 percent of wall time at
  the best configuration and is pure serial dead time. That number is the
  concrete case for Agent 5's dense-versus-compact legality comparison; the
  worker already has the legal-action list, so publishing it compactly is cheap
  if the measurement favours it.
- `belief_targets` remains privileged and is still absent from every shared
  field. Keep it out of the model-facing buffers when you add trajectory
  storage.

## 3. Agent 3 — Trajectory Storage and Reconstruction

### 3.1 Status

**PASS**

### 3.2 Implementation summary

Three new modules add the compact self-play trajectory layer on top of the
frozen reference engine. `stratego/engine/` was not modified and contains no
call into `stratego/training/`; every rule decision, observation channel, legal
action, event and belief label is still produced by the engine. Agent 1 had
already provided the hooks this agent needed on `BatchSimulator` — `snapshot`,
`belief_targets`, `replay_record`, `public_board`, `public_setup` — so no
earlier agent's module needed a behavioural change.

**`stratego/training/serialization.py`** holds the byte primitives and knows
nothing about Stratego: LEB128 unsigned varints, zigzag signed varints, optional
integers, packed flag bytes, little-endian `float32`, delta-encoded ascending
integer sequences, a per-record string interning table, and `zlib` framing.
Standard library only; no new dependency was added.

**`stratego/training/trajectory.py`** defines the versioned record schema
(`trajectory_v1`, wire format `1`, magic `STJ1`) and the collector. A game
stores one header, the ordered action list, a compact engine snapshot every
`snapshot_interval` plies, and one sparse decision record per ply. The
`127 x 10 x 10` observation tensor is never stored, and neither is the dense
10,000-entry mask or a dense probability vector: a decision carries only the
ascending legal action identifiers and one `float32` per legal action.

The storage rule applied throughout is that nothing the header already
determines is stored a second time. A snapshot therefore omits every piece's
identifier, owner, true type and starting square — all four follow from the two
setups the header carries — and omits the 100-entry board array, which is
rebuilt from the living pieces. The encoder does not assume those derivations
hold: it recomputes each one and compares it against the snapshot it was handed,
refusing to encode on disagreement. It also refuses a snapshot whose field set
is not exactly what `create_snapshot(state, include_history=False)` produces. A
future engine change that invalidated an assumption would fail loudly at
collection time rather than silently write records that decode to the wrong
position.

Until Agent 4 supplies a network, `synthetic_policy` and `synthetic_value`
stand in for one: a `blake2b`-derived distribution over exactly the true legal
set and a three-class value, both pure functions of `(game_id, ply)`. The played
action is sampled from that distribution, so games are varied while the whole
run reproduces from the seeds. These are two function calls, not a schema
feature — real Agent 4/5 outputs drop into the same fields unchanged. Stored
probabilities are rounded to `float32` at the point of storage, so an in-memory
record compares exactly equal to its own decoded form and storage fidelity can
be asserted rather than approximated.

**`stratego/training/reconstruction.py`** rebuilds a historical position from
`game record + nearest snapshot at or before p + subsequent actions`, then
derives the frozen reference state, `observation_v2_1_127ch`, the legal-action
list, the dense mask on request, the public knowledge views and the privileged
belief target. `iter_reconstructed_decisions` walks a whole game by advancing a
single state instead of restoring per position; it hands out an independent copy
of the state by default so a materialised list of results stays valid. The
module also provides the digest surface the acceptance harness compares with.

Belief targets stay separate by construction. `ReconstructedDecision` exposes
`observation` and `belief_target` as two different fields, the observation comes
from `build_observation`, which has no code path to `belief_target`, and the
codec has no belief field at all — nothing privileged is ever serialised. A
consumer that feeds the observation to a network cannot reach the labels by
accident, and a consumer that wants the training target must name it.

### 3.3 Files created / modified

Created:

```text
stratego/training/serialization.py
stratego/training/trajectory.py
stratego/training/reconstruction.py
tests/training/test_trajectory.py
tests/training/test_reconstruction.py
scripts/run_phase3_agent03.py
reports/phase_3_data/agent_03_trajectory_reconstruction.json
reports/phase_3_data/agent_03_snapshot_interval_raw.csv
```

Modified:

```text
stratego/training/__init__.py            (new exports only, see 3.6)
reports/phase_3_implementation_report.md (this section)
```

`stratego/engine/` and `stratego/training/batch_simulation.py`,
`shared_buffers.py` and `worker_pool.py` were not modified.

### 3.4 Test summary

- `python -m pytest -q`: **1,424 passed, 0 failed, 0 skipped** — Agent 2's 1,340
  plus 84 new tests. Agent 1's and Agent 2's suites were green before and after.
- `tests/training/test_trajectory.py` (57 tests) covers the byte primitives and
  their rejection paths, exact and compressed record round trips, foreign-magic
  and future-version rejection, snapshot round trips against an independent
  replay, the four codec guards (foreign game, unknown field, history-bearing
  snapshot, setup disagreement), the versioning and required header fields,
  agreement with the frozen replay schema, the storage principle (no tensor, no
  dense vector, bytes per decision), configurable cadence at all three
  intervals, every sparse decision-storage rule with a parametrised table of
  twelve malformed decisions, `float32` storage exactness, determinism of the
  synthetic policy/value/selection, reproducibility of a whole collection run
  from its seeds, and the builder's ordering and terminal-state contracts.
- `tests/training/test_reconstruction.py` (27 tests) covers exact reconstruction
  of every stored decision, the same after a serialisation round trip, agreement
  with a full replay from ply 0, all three snapshot intervals, the replayed-action
  bound, nearest-snapshot selection, sequential-versus-random-access agreement,
  field-by-field agreement with the frozen engine for observation, legal list,
  dense mask, identity and public knowledge, the four belief-separation
  properties, the digest and comparison surface including a deliberately
  corrupted observation, and the error surface.
- `python scripts/run_phase3_agent03.py` is the acceptance gate and exits
  non-zero unless every threshold in 3.5 is met. Full run: **1,948.5 s**.

### 3.5 Key measured results

`Darwin 25.5.0 arm64` (Apple M4 Pro), Python 3.13.2, NumPy 2.5.1,
single-process, no PyTorch and no Metal. Peak resident memory 352 MB.

| Gate | Requirement | Measured |
|---|---|---|
| Historical decisions reconstructed | >= 1,000,000 | **1,000,162** |
| State fingerprint mismatches | 0 | **0** |
| Acting-player mismatches | 0 | **0** |
| `observation_v2_1_127ch` mismatches | 0 | **0** |
| Legal-action list mismatches | 0 | **0** |
| Dense legal-mask mismatches | 0 | **0** (99,978 compared) |
| Public-knowledge mismatches | 0 | **0** |
| Privileged belief-target mismatches | 0 | **0** |
| Stored selected-action mismatches | 0 | **0** |
| Environment/generation/game identity mismatches | 0 | **0** |
| Public event-stream mismatches | 0 | **0** (200 games) |
| Codec round-trip failures | 0 | **0** |
| Record validation problems | 0 | **0** of 2,020 games |
| Snapshot intervals measured | 16, 32, 64 | **all three** |
| Test suite | all pass | **1,424 / 1,424** |

**Reconstruction gate.** 2,020 complete games, 1,000,162 stored decisions, every
one of them rebuilt independently — snapshot restore plus replay, not a
sequential walk — and compared against a digest captured from the *live* game
immediately before its action was applied. Comparison covered the full state
fingerprint, acting player, observation tensor bytes, legal-action list, public
knowledge for both observers, privileged belief target, stored selected action,
and the game/environment/generation identity triple. Every comparison ran
against the record *after* an encode/decode round trip, so the codec is inside
the gate rather than beside it. Mean game length 495 plies, median 466; mean
24.9 legal actions per decision, maximum 62. Terminal reasons: 1,128 flag
captures, 851 battleless-limit draws, 41 no-legal-move wins.

**Dense-mask subset.** Comparing 1,000,000 dense 10,000-entry masks would have
dominated the run without testing anything the legal-action lists do not already
cover, so masks were compared on every tenth decision — 99,978 comparisons — and
the legal-action lists were compared for all 1,000,162. The subset is spread
across game length rather than clustered: 3,210 comparisons in plies 0-15, 3,105
in 16-31, 5,900 in 32-63, 11,029 in 64-127, 19,667 in 128-255, 29,669 in
256-511, and 27,398 at ply 512 and above.

**Storage.** At the default interval of 32, on the gate's 2,020-game corpus:

| Quantity | Raw | zlib |
|---|---|---|
| Mean bytes per game | **93,003** | **64,692** |
| Estimated bytes per million games | **93.0 GB** | **64.7 GB** |

Of the 93,003 raw bytes, 76,391 are decision records, 15,212 are snapshots and
the remaining ~1,400 are the game header and the action list. Mean 154 bytes
per decision. The equivalent dense storage — one `float32`
observation tensor per decision — would be 50,800 bytes per decision, about 330
times larger, and that is before a dense 10,000-entry probability vector.

**Snapshot interval benchmark.** All three intervals were measured on the
*identical* set of 200 games (verified by a corpus digest: the collection policy
is a function of `(game_id, ply)` only, so the same seed replays the same games
and only the snapshot cadence differs).

| Interval | Raw B/game | zlib B/game | Snapshot B/game | Decision B/game | Positions/s | Mean replay | p95 replay |
|---|---|---|---|---|---|---|---|
| 16 | 101,421 | 62,668 | 28,442 | 71,598 | 2,095 | 7.5 | 15 |
| **32** | **87,155** | **60,682** | **14,231** | **71,598** | **1,681** | **15.3** | **30** |
| 64 | 80,072 | 59,450 | 7,174 | 71,598 | 1,149 | 30.9 | 61 |

**Recommended interval: 32.** The choice is made against a stated budget —
15 percent more storage than the cheapest measured interval — and then takes the
fastest interval that fits, rather than taking either extreme automatically.
Interval 32 costs 8.8 percent more raw storage than interval 64 (2.1 percent
compressed) and reconstructs 1.46 times faster. Interval 16 is a further
1.25 times faster again but costs 26.7 percent more raw storage than interval
64, which is outside the budget; it was not chosen on speed alone. The decisive
context is that snapshots are the minority of the record at every cadence —
decision records are 71,598 bytes per game and are byte-identical at all three
intervals, since cadence changes nothing about them — so cadence buys
reconstruction speed at a much better rate than it costs storage, and the
compressed spread across the whole 16-to-64 range is only 5.4 percent. If a
future storage budget is tight, halving the interval to 16 remains cheap in
compressed terms; the reason not to do it now is raw resident size in a replay
buffer, not disk.

**Throughput.** Random-access reconstruction of state, observation, legal
actions and belief target runs at 1,681 positions/second per process at interval
32; state-only reconstruction is faster. The end-to-end gate rate was 850
verified decisions/second, which is lower because it also builds and hashes the
live-side digests and both public-knowledge views. Collection accounted for
643 s of the run and verification for 1,177 s.

### 3.6 Deviations and limitations

- **State fingerprints are compared with `include_history=False`.** A compact
  snapshot deliberately carries no derived event log and no action history
  (`08_internal_state_spec.md` section 15), so a history-inclusive fingerprint
  could not match by construction. The action history is verified separately and
  exactly: `validate_game_record` requires every decision's stored
  `selected_action_id` to equal the corresponding entry of the record's action
  list, and that ran clean on all 2,020 games.
- **"Public knowledge" in the 1,000,000-decision gate means both observers'
  board and setup views.** Those derive from per-piece knowledge flags and
  reconstruct exactly from a snapshot. The public *event stream* needs the
  derived event log and therefore a replay from ply 0, which is a whole-game
  operation; it was checked separately on 200 games, both observers, with zero
  mismatches.
- **`selected_action_id` is stored per decision although the record's action
  list already determines it.** Two bytes of deliberate redundancy: the schema
  requires the field, and the duplication gives a real cross-check that caught
  nothing here but would catch codec drift.
- **Probabilities are stored at `float32`.** That is the honest fidelity choice
  for accepting real model outputs unchanged, and it is also the single largest
  storage line: at 24.9 legal actions per decision, probabilities are about 100
  of the 154 bytes per decision. `float16` or a quantised integer form would cut
  roughly a third of total storage at a fidelity cost; that is a measurement for
  a later agent, not a change made silently here.
- **`stratego/training/__init__.py` was modified.** Additive only: the new
  modules were added to the package docstring and the export list. No existing
  export was removed or changed.
- **Collection is single-process.** Agent 2's handoff is right that production
  records must be produced inside the worker that owns the slot, and the
  collector is built for that — `GameTrajectoryBuilder` needs only a state and a
  slot's identity, both of which a worker already has. Wiring it into
  `worker_pool.py` would have meant modifying an earlier agent's module for no
  gain to this agent's gate, which is a determinism risk this agent had no
  reason to take. See 3.8.
- **No neural network.** As specified. The synthetic policy and value exist only
  to exercise storage fidelity.
- **No trajectory corpus is kept on disk.** Records are encoded, measured,
  verified and dropped, so the acceptance run's peak memory tracks games in
  flight (352 MB) rather than corpus size. Reproducing the corpus needs only the
  root seed.

### 3.7 Data files

```text
reports/phase_3_data/agent_03_trajectory_reconstruction.json
reports/phase_3_data/agent_03_snapshot_interval_raw.csv
```

The JSON holds every count in 3.5 plus the per-interval results, the
dense-mask ply-bucket histogram, terminal reason and result counts, the
recommendation rationale, timings and the platform record. The CSV is one row
per snapshot interval with every storage and reconstruction column. No
reproduction file was written because there were no mismatches.

### 3.8 Handoff notes for Agent 4

- Agent 3 is `PASS`; the prerequisite is satisfied.
- **The schema is ready for a real model; do not change it to fit one.**
  Replace the two calls `synthetic_policy(game_id, ply, legal_action_ids)` and
  `synthetic_value(game_id, ply)` with network outputs and pass them to
  `GameTrajectoryBuilder.record_decision` unchanged. Probabilities must arrive
  as one value per legal action, in the engine's ascending legal order.
  `collection_policy_version` is per decision, not per game, so a checkpoint
  switch mid-game is already representable; set `collection_checkpoint_id` on
  the builder.
- **`belief_targets` still never touches a model-facing structure.** It is
  absent from the codec entirely and appears only as
  `ReconstructedDecision.belief_target`. Keep it out of the encoder input when
  you add the belief head.
- **Wiring collection into the worker pool is a small job and is the right next
  step for whoever owns transport.** `builder_for_slot(simulator, slot, ...)`
  takes everything from the slot, and `record_decision` needs only the live
  state plus the model outputs — both already inside the worker. Nothing has to
  cross a pipe: a finished `GameRecord` is bytes, so
  `encode_game_record_compressed` output is what a worker should send back, not
  a state or an observation.
- **Reconstruction is 1,681 positions/second per process.** A training loop that
  wants more than that from one process will not get it from a smaller snapshot
  interval — the measured ceiling at interval 16 is 2,095 — so replay sampling
  should be parallel across processes, exactly like Agent 2's simulation
  workers. Budget it as a real cost alongside inference.
- **Storage is 93 GB raw / 65 GB compressed per million games** at the
  recommended interval. If that is over budget, the lever is the per-decision
  probability precision, not the snapshot cadence: probabilities are about two
  thirds of a decision record, and the whole 16-to-64 cadence range moves
  compressed size by only 5.4 percent.
- **`iter_reconstructed_decisions` copies the state by default.** For a hot
  sampler that consumes each result before pulling the next, pass
  `copy_state=False` and skip a full clone per position.
- The codec fails loudly rather than guessing. If a future change to
  `create_snapshot` adds or removes a field, `encode_snapshot` raises and tells
  you to revise the codec; do not widen the guard to make it pass.

## 4. Agent 4 — Representative MPS Inference Benchmark

### 4.1 Status

**PASS**

> **The network measured in this section is a benchmark probe, not a frozen
> model design.** It exists only to price Apple Metal inference, legality
> application and action sampling at the shapes Phase 4 will use. It is never
> trained, its weights are seeded noise, and nothing about it is a contract.
> The final architecture is still selected later, on playing strength.

### 4.2 Implementation summary

Two new modules and no change to the frozen engine. `stratego/engine/` was not
modified, and neither were the modules owned by Agents 1-3.

**`stratego/training/representative_model.py`** holds the probe and the
legality/sampling paths. The encoder is the `05_project_plan.md` section 5
target shape: 100 board-square tokens, 127 raw features per token, an input
projection to width 128, a learned position embedding, 4 pre-norm Transformer
encoder blocks with 4 attention heads and feedforward width 512. Three probe
heads share it:

- **policy** — source-query against destination-key scoring, producing a logical
  `[B, 100, 100]` matrix whose row-major flattening *is* the frozen
  `action_id = 100 * source + destination` encoding, so 10,000 logits come out
  with no permutation anywhere in the path;
- **value** — three logits for win, draw and loss over the pooled encoder state;
- **belief** — a deliberately lightweight shared per-square head producing
  `[B, 100, 12]` opponent-type logits. This is the placeholder the instructions
  asked for, not the paper's separate large belief Transformer.

Weights are built on the central processing unit from a fixed seed and only then
moved and cast, so every device and precision runs numerically identical
parameters and the cross-precision comparisons mean something.

Legality has two interchangeable implementations over the same engine-supplied
legal set. The **dense** path applies the engine's `(B, 10000)` mask directly.
The **compact** path pads legal action identifiers to `(B, capacity)` and gathers
them out of the policy logits. Both normalise over the legal set only, and
`build_compact_legality` raises rather than truncating when a state exceeds the
capacity — silently dropping a legal move would be a correctness bug sold as a
performance win. Sampling is Gumbel-max, which is exactly a categorical draw from
the masked softmax and is one kernel on Metal.

**`stratego/training/mps_benchmark.py`** owns the measurement. It detects Metal
rather than assuming it, builds the batch corpus from *real* frozen-engine
positions, and times the whole coordinator-side step:

```text
host observations -> device transfer -> encoder forward
    -> legality application -> action sampling -> chosen action ids back to host
```

Every timed region is bracketed by `torch.mps.synchronize()`. Metal dispatch is
asynchronous, so an unsynchronised timer measures queue submission rather than
work. Each configuration runs a warm-up, a pilot that sizes the iteration count,
then three trials of two passes: a **phased** pass that synchronises between
stages, which is what separates the model-only and legality+sampling figures,
and an **end-to-end** pass with a single synchronisation per step, which is the
honest rate. From batch 256 upward the phased stage sums agree with the
end-to-end means to within one percent, which is the internal consistency check
on the method; at batch 64 the phased sum runs about 5 percent high because the
extra synchronisations are no longer negligible against a 6 ms step. That is why
the phased split is used for attribution and the end-to-end pass for rates.

### 4.3 Files created and modified

Created:

```text
stratego/training/representative_model.py
stratego/training/mps_benchmark.py
tests/training/test_representative_model.py
scripts/run_phase3_agent04.py
requirements-training.txt
reports/phase_3_data/agent_04_mps_inference.json
reports/phase_3_data/agent_04_mps_inference_raw.csv
```

Modified:

```text
stratego/training/__init__.py            (docstring note only, see 4.6)
reports/phase_3_implementation_report.md (this section)
```

`stratego/engine/` was not modified. Neither were `batch_simulation.py`,
`shared_buffers.py`, `worker_pool.py`, `serialization.py`, `trajectory.py` or
`reconstruction.py`.

**Dependency change.** PyTorch was not previously installed. It is recorded in a
new `requirements-training.txt` rather than in `requirements.txt`, because that
file pins the exact versions used to produce `reports/phase_2_metrics.json` for
the frozen engine, and the engine itself has no PyTorch dependency. Phase 3 work
installs both:

```bash
pip install -r requirements.txt -r requirements-training.txt
```

### 4.4 Test summary

- `python -m pytest -q`: **1,453 passed, 0 failed, 0 skipped** — Agent 3's 1,424
  plus 29 new. Every earlier suite was green before and after.
- `tests/training/test_representative_model.py` (29 tests) runs on the central
  processing unit and on Metal, and at float32, float16 and bfloat16 on Metal.
  It covers output shapes and finiteness, the three-class value probe forming a
  distribution, tokenisation being a pure layout change verified channel by
  channel against `build_observation`, dense legality driving illegal
  probability to exactly zero, sampled actions being legal across 25 draws in
  both legality paths, dense and compact agreeing on the normalised legal-set
  distribution, compact padding carrying exactly zero mass, capacity overflow
  failing loudly, repeated forwards being stable per device and precision,
  distributional (explicitly *not* bitwise) agreement between the central
  processing unit and Metal, seed-determined weights, and the probe being
  machine-labelled as a benchmark probe.
- Two tests carry the hidden-information argument. One permutes hidden opponent
  identities with `permute_hidden_identities` and asserts that the model's
  tokens and all three outputs are bit-identical across the permutation, while
  `belief_targets_differ` confirms the privileged target really did change — so
  the check is not vacuous. The other monkeypatches
  `stratego.engine.observation.belief_target` to raise and then builds inputs
  and samples an action, proving the model path cannot reach the privileged
  labels. A third asserts the input projection is exactly 127 features wide, so
  no extra channel can be smuggled in.
- `python scripts/run_phase3_agent04.py` is the acceptance gate and exits
  non-zero unless every condition in 4.5 holds. Full run: **689.7 s**.

### 4.5 Key measured results

`Darwin 25.5.0 arm64` (macOS 26.5.2), **Apple M4 Pro**, 14 cores, 48 GiB unified
memory. Python 3.13.2, PyTorch 2.13.0, NumPy 2.5.1. Metal **is** available
(`torch.backends.mps` built and available; allocator reports a 37.4 GiB
recommended maximum). Probe parameter count: **873,999**.

Batches are drawn from a pool of **2,048 real frozen-engine positions** spanning
plies 0 to 480 (mean 202.9), with 24.27 legal actions per position on average
(minimum 2, maximum 49) — a legal density of 0.24 percent of the 10,000-entry
action space. Using real positions rather than synthetic masks is what makes the
dense-versus-compact comparison meaningful.

**Completion gate: all ten conditions hold.** 42 configurations (7 batch sizes x
3 precisions x 2 legality paths), **0 failures, 0 out-of-memory conditions, 0
illegal samples** out of every sampled action checked.

| Measure | Result |
|---|---|
| Model-only inference, best | **15,902 positions/s** (batch 2048, float16) |
| End-to-end model step, best | **14,761 positions/s** (batch 2048, float16, compact) |
| Sustained, recommended configuration | **14,922 positions/s** (float16 + dense, batch 2048) |
| Sustained, fastest measured configuration | 15,778 positions/s (float16 + compact) |
| Sustained, float32 + dense baseline | **12,185 positions/s** |
| float32 + dense baseline, sweep peak | 11,794 positions/s (batch 2048) |
| Peak process memory | 1.12 GB |
| Metal allocator, high-water across sweep | 3.26 GB driver-allocated |
| Metal allocator, at batch 2048 float16 | 190 MB currently allocated |

**Batch-size scaling.** Throughput climbs steeply to batch 512, then flattens.
At float16 with compact legality: 10,689 positions/s at batch 64, 13,578 at 512,
14,407 at 1,024, 14,581 at 1,536 and 14,761 at 2,048. Batch 64 is dispatch-bound
rather than compute-bound — its per-step latency is 6.0 ms of which the encoder
is 5.7 ms, and its run-to-run spread is the largest in the sweep. **Batch 1,024
captures about 98 percent of the achievable rate**; going to 2,048 buys roughly
2 percent for double the in-flight memory and double the bulk-synchronous batch
latency, which is what every simulation worker waits on. No batch size failed or
ran out of memory, which on 48 GiB of unified memory is unsurprising.

**Precision.** float32 is the baseline. Both reduced precisions are supported and
measured stable on the *complete* path, not on one operation:

| Precision | Best end-to-end | Against float32 | Greedy agreement | Max legal-probability difference |
|---|---:|---:|---:|---:|
| float32 | 12,345 positions/s | baseline | — | — |
| float16 | 14,761 positions/s | **+19.6 %** | **0.996** | 1.34e-04 |
| bfloat16 | 14,659 positions/s | +18.8 % | 0.941 | 1.27e-03 |

**Recommended precision: float16.** The two reduced modes are within about one
percent of each other on speed, so the choice is settled on fidelity to the
float32 distribution, where float16 is an order of magnitude closer and agrees
with the float32 greedy action on 99.6 percent of positions against bfloat16's
94.1 percent. Both are recorded as supported and stable; neither produced a
non-finite output at any batch size. The gain is a whole-path measurement:
normalisation and sampling run in float32 in every mode, and that cast sits
inside every timed region, so float16 is not flattered by skipping it.

**Dense versus compact legality.** The legality stage alone is dramatically
cheaper in the compact form — at batch 2,048 float16 it falls from 9.18 ms to
1.23 ms, a **7.5x** stage speed-up, and legality transport falls from 10,000
bytes per position to 1,152 at a padding capacity of 128. But the encoder
dominates the step at 129 ms of a 139 ms total, so that stage win converts to
only a few percent end to end.

Because a single sweep comparison sits close enough to the 5 percent adoption
margin to land on either side of it, the verdict is taken on **five interleaved
dense/compact pairs** at the decision configuration, which also cancels thermal
drift. That measurement was run twice, in two independent full acceptance runs:

| Repeatability, batch 2048, float16, capacity 128 | Final run | Preceding run |
|---|---:|---:|
| Mean compact gain | **+3.7 %** | +5.7 % |
| Median | +4.1 % | +5.7 % |
| Range | +1.0 % to +5.1 % | +5.0 % to +6.3 % |
| Standard deviation | 1.6 pp | 0.5 pp |
| Compact ahead in every pair | **yes** | **yes** |

Read together, these say something more useful than either alone. The *direction*
is robust: compact was faster in **10 of 10 pairs** across two independent runs.
The *magnitude* is not: the mean gain moved from +5.7 to +3.7 percent between
runs, straddling the pre-registered 5 percent margin, so which side of the
threshold a single run lands on is close to a coin flip.

**Recommended legality representation: dense.** The rule requires a reduced
representation to clear the margin *robustly*, and a gain that is reliably
positive but somewhere between 1 and 6 percent does not. The engine already
produces the dense mask, whereas the compact path adds a fixed padding capacity
and therefore a new way to fail. Per the instruction not to force sparse legality
downstream without a meaningful end-to-end benefit, **compact is offered to
Agent 5 as a validated option, not imposed**: it is measurably a little faster,
it is proven equivalent, and adopting it is a judgement call about complexity
rather than a throughput necessity.

The capacity question is settled independently and is not the reason for the
verdict. Compact was priced at a padding capacity of **128**, not at the pool's
observed maximum of 56, because a capacity fitted to one sample of positions is
not a bound on the legal-action count. This costs nothing: capacities 56, 64, 128
and 256 land within 0.6 percent of each other (14,785 / 14,797 / 14,702 / 14,791
positions/s at float16). The compact path is insensitive to padding width because
even 256 gathered entries are negligible against 10,000.

**Correctness of the measured paths.** Dense and compact agree on the normalised
legal-set distribution to a maximum absolute probability difference of 6.0e-08 at
float32 and 3.0e-08 at both reduced precisions, with illegal probability mass
exactly `0.0` and legal mass summing to 1.0. Repeated forwards on the same
device, precision and input were **bitwise identical** over five repeats at all
three precisions. No bitwise central-processing-unit-versus-Metal claim is made
or tested; the two agree distributionally to under 1e-04.

**Indicative ratio for the Phase 3 optimisation decision — not the decision.**
Against Agent 2's best measured simulation rate of 91,482 positions/s, the
`05_project_plan.md` ratio would be roughly `R = 6.1` at the recommended
configuration and `R = 7.5` against the float32 dense baseline, both far above
the 2.0 threshold at which the Python simulator is kept. This is stated only as a
sighting shot: both sides were measured **in isolation**, and the real `R` must
come from the combined pipeline with workers and coordinator contending for the
same machine. Agent 5 owns that measurement.

### 4.6 Deviations and limitations

- **The network is a benchmark probe.** It is untrained, its weights are seeded
  noise, and its 873,999 parameters sit just below the 1-2 million the planning
  document anticipated — the shape is exactly the planned one, so the shortfall
  is what the planned shape actually costs, not a scaled-down substitute. Every
  throughput number here scales with the final architecture and must be
  re-measured once that is chosen.
- **Normalisation and sampling always run in float32**, even when the encoder
  runs in float16 or bfloat16. A 10,000-way categorical draw taken in float16 is
  not numerically trustworthy. The cast is inside every timed region, so no
  reduced-precision result is flattered by it.
- **Sampling is Gumbel-max, not `torch.multinomial`.** It is an exact categorical
  draw from the masked softmax and behaves uniformly across backends. A
  `multinomial`-based path was not benchmarked.
- **Peak process memory is a process high-water mark**, not a per-configuration
  footprint: it is monotonic across the sweep and is reported per row only as
  the value at that point. The Metal allocator counters are per-configuration.
- **The compact padding capacity is not a proven bound.** The pool's maximum was
  49 legal actions and Agent 3 observed 62 across 1,000,162 decisions; 128 is a
  headroom choice, not a theorem. `build_compact_legality` raises above it, so
  the failure is loud rather than a silently dropped legal move — but a
  production compact path needs a real bound or a fallback.
- **Observation tokenisation is not inside the timed region.** The
  `(127, 10, 10)` to `(100, 127)` reshape happens when the pool is built, on the
  assumption that a coordinator would choose the token layout in shared memory.
  Host-to-device transfer *is* timed, and is about 7 percent of the step at
  batch 2,048.
- **This is a single-process, uncontended measurement.** No simulation workers
  were running. These are upper bounds for the coordinator with the machine to
  itself, so treat them as a ceiling rather than a production rate.
- **bfloat16 is supported and stable and is not recommended.** In the final run
  float16 was both marginally faster and an order of magnitude closer to the
  float32 distribution, so nothing is traded away by choosing it. The two are
  within about one percent on speed, though, and that ordering swapped between
  runs; if Phase 4 prefers bfloat16 for training-side reasons, nothing measured
  here argues against it.
- **PyTorch is a new dependency**, isolated to `requirements-training.txt` so
  that `requirements.txt` continues to describe the frozen engine exactly.
- **`stratego/training/__init__.py` gained a docstring note only.** The two new
  modules are deliberately *not* re-exported from the package, because importing
  them requires PyTorch and `import stratego.training` must keep working with
  `requirements.txt` alone. Import them directly.

### 4.7 Data files

```text
reports/phase_3_data/agent_04_mps_inference.json
reports/phase_3_data/agent_04_mps_inference_raw.csv
```

The JSON holds the platform and Metal device record, the architecture summary
and parameter count, every one of the 42 sweep results with full latency
distributions, the precision support and stability records, the dense/compact
equivalence and determinism checks, the capacity sweep, the five-pair
repeatability record, all three sustained-throughput runs, the recommendation
rationales, the completion gate and the test summary. The CSV is one row per
swept configuration with every latency, throughput and memory column.

### 4.8 Handoff notes for Agent 5

- Agent 4 is `PASS`; the prerequisite is satisfied. Metal is available and was
  used; no result here was substituted from the central processing unit.
- **Do not treat any of these rates as a production number.** They are
  single-process ceilings for a probe of the planned shape. The end-to-end
  coordinator will be slower, and the final model will not be this model.
- **Start the coordinator at batch 1,024, float16, dense legality.** Batch 1,024
  captures about 98 percent of the achievable rate at half the in-flight memory
  and half the bulk-synchronous latency of 2,048, which matters when a step's
  latency is what every simulation worker waits on.
- **Compact legality is an option, not a requirement.** It was faster in 10 of 10
  interleaved pairs, but by somewhere between 1 and 6 percent depending on the
  run, and it introduces a fixed padding capacity. Adopt it only if the
  coordinator can carry the extra representation cleanly; size the capacity
  conservatively (128 costs nothing measurable) and let it fail loudly. Nothing
  downstream depends on the choice — the two paths are proven equivalent.
- **The model is the bottleneck, by a wide margin.** The encoder is about 93
  percent of the step at batch 2,048; legality and sampling together are under 1
  percent in the compact path (6.6 percent dense), and host-to-device transfer is
  about 7 percent. Optimisation effort
  belongs in the encoder and in overlapping transfer with compute, not in the
  legality representation.
- The probe plugs straight into Agent 3's storage: `policy_logits` masked to the
  legal set gives the per-legal-action probabilities that
  `GameTrajectoryBuilder.record_decision` already expects, and `value_logits`
  gives the win/draw/loss triple. Agent 3's schema does not need to change.
- Only the coordinator may touch Metal (`03_game_engine_spec.md` section 18).
  Nothing in `representative_model.py` or `mps_benchmark.py` is safe to import
  into a simulation worker.

## 5. Agent 5 — End-to-End Pipeline, Soak, and Backend Decision

### 5.1 Status

**PASS**

### 5.2 Implementation summary

Agents 1-4 were integrated into the first end-to-end bulk-synchronous self-play
pipeline. The cycle is the required one, and nothing in it sends a game object,
an observation or a legality mask through a pipe:

```text
workers build observations and legality into shared memory
-> barrier: every worker reports its phase complete
-> the coordinator runs the model over the ready rows, in inference-batch chunks
-> the coordinator applies legality and samples one legal action per row
-> actions, policy and value are written back to shared memory
-> workers advance their environments
-> finished games are sealed and their slots independently reset
-> next global step
```

**`stratego/training/coordinator.py`** owns the cycle and is the only module in
the pipeline that imports the model or touches Metal. `SelfPlayCoordinator`
holds one `WorkerPool` and one model instance; a simulation worker never imports
it, and a test asserts that importing `worker_pool` does not pull PyTorch into
the process. Every active environment receives a decision on every global step:
`inference_batch_size` splits the ready rows into `ceil(active / batch)`
sequential Metal dispatches rather than capping how many environments advance.

**Decision transport.** Trajectory records must be built by the worker that owns
the slot, because the coordinator deliberately holds no game object — but the
policy and value that belong in a record come from the coordinator's model.
Three coordinator-written fields carry that decision back down the same shared
block the actions travel on: `policy_probabilities` (one probability per entry
of the slot's ascending legal-action list, zero-padded to 128),
`value_prediction` (win/draw/loss) and `decision_valid`. The ascending order is
the order `BatchSimulator.legal_actions` returns *and* the order
`numpy.flatnonzero(legal_mask)` produces, so neither side transmits the
identifiers. Recording costs no extra round trip, and the per-phase control
reply stays a small fixed-shape dictionary of counters.

**In-worker recording.** `worker_pool.py` gained a `RecordingConfig` and the
path that uses it. A worker builds a `GameTrajectoryBuilder` per slot, records
each decision *before* the action that leaves the position is applied, seals the
record on the terminal state before any reset can move the slot on, encodes it,
counts it and discards it. A sampled subset of games additionally carries live
digests through its whole life and is round-tripped through the trajectory codec
and rebuilt through Agent 3's path.

**`stratego/training/end_to_end_benchmark.py`** holds the integrated
differential gate, the screening and sustained measurement, the
simulation-pipeline measurement that forms the numerator of `R`, and the soak.

### 5.3 Files created / modified

Created:

```text
stratego/training/coordinator.py
stratego/training/end_to_end_benchmark.py
tests/training/test_coordinator.py
tests/training/test_end_to_end_pipeline.py
scripts/run_phase3_agent05.py
reports/phase_3_data/agent_05_end_to_end.json
reports/phase_3_data/agent_05_end_to_end_raw.csv
reports/phase_3_data/agent_05_soak_timeseries.csv
```

Modified:

```text
stratego/training/shared_buffers.py        three coordinator-written decision fields
stratego/training/worker_pool.py           in-worker trajectory recording
stratego/training/representative_model.py  illegal-action sampling fix (section 5.6)
tests/training/test_shared_buffers.py      widened coordinator-written field set
reports/phase_3_implementation_report.md   this section
```

`stratego_project_docs/` was **not** modified. `stratego/engine/` was not
modified.

### 5.4 Test summary

`tests/training/test_coordinator.py` and
`tests/training/test_end_to_end_pipeline.py`: **44 passed, 0 failed**, 13.0 s.
The whole repository suite is **1,497 passed**.

The pipeline tests run real worker processes, real shared memory and a real
model at small scale, and cover: every sampled action being legal in the
published mask; batch chunking; independent reset with exactly one generation
increment; terminal accounting; the absence of any privileged field from the
transport; the published observation being the acting player's perspective only;
`worker_pool` not importing PyTorch; the control pipe staying small with
recording on; exact reconstruction of recorded decisions; the collection policy
version being retained and distinct from Agent 3's; and stored probability rows
being distributions over exactly the legal prefix.

The coordinator tests cover the vectorised compact-legality build against a
per-row construction, configuration validation, the decision-rule boundaries at
2.0 and 1.25, the screening plan's coverage of every required dimension value,
and four regression tests pinning the sampling fix in section 5.6.

### 5.5 Key measured results

`Darwin 25.5.0 arm64` (Apple M4 Pro, 10 performance + 4 efficiency cores,
48 GB), Python 3.13.2, PyTorch 2.13.0, NumPy 2.5.1, Metal available. Total
harness time 2.18 h.

| Gate | Requirement | Measured |
|---|---|---|
| Integrated environment steps | >= 10,000 | **10,048** |
| Integrated mismatches | 0 | **0** |
| Stored decisions reconstructed | >= 10,000 | **11,251** |
| Reconstruction mismatches | 0 | **0** |
| Soak duration | >= 2 h | **2.00 h** |
| Soak errors / reconstruction mismatches | 0 | **0 / 0** |
| Test suite | all pass | **44 / 44** |

**Integrated differential gate.** An independent set of engine games was
advanced in lockstep with the live pipeline, using the actions the *model*
sampled. This is stronger than Agent 1's or Agent 2's differential runs: the
comparison spans the whole chain — worker publish, coordinator inference,
legality, sampling, write-back, worker step, republish — rather than the
simulation layer alone. 10,048 row comparisons covered the published
observation, dense mask, legal count, acting player, environment identifier,
generation, ply and battleless counter; 10,048 action-legality checks confirmed
every sampled action was legal in the independently generated legal set; 15
completed games compared terminal reason, winner, draw flag, final ply and both
players' results; 79 distinct `(environment_id, generation)` keys with no reuse.
Zero mismatches in every category.

**Reconstruction gate.** 172,800 decisions were recorded across 236 games, of
which 42 games carried live digests and were round-tripped through the
trajectory codec and rebuilt through Agent 3's path: **11,251 decisions
reconstructed, zero mismatches** across observation, legal list, dense mask,
state fingerprint, belief target, public knowledge, acting player, selected
action and identity. No game was joined mid-way.

**Configuration screening.** 24 points were measured rather than the full
5 x 5 x 7 = 175-point Cartesian product, chosen so that every required worker
count, environment count and inference batch size is measured at least once,
with deliberate off-diagonal probes where CPU generation and Metal consumption
are mismatched. Agent 2 had already shown throughput moving under 3 percent
across a factor of eight in environments and Agent 4 had shown batch sizes above
1,024 buying about 2 percent; screening all 175 would have spent most of an hour
re-measuring known-flat axes.

Batch sweep (8 workers, 2,048 environments), positions/second:

| batch | 64 | 128 | 256 | 512 | 1,024 | 1,536 | 2,048 |
|---|---|---|---|---|---|---|---|
| pos/s | 6,674 | 8,232 | 11,551 | 11,628 | **11,692** | 11,520 | 11,592 |

Worker sweep (2,048 environments, batch 2,048):

| workers | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| pos/s | 10,725 | 11,364 | 11,592 | **12,331** | 12,271 |

Environment sweep (8 workers, batch = environments):

| environments | 256 | 512 | 1,024 | 1,536 | 2,048 |
|---|---|---|---|---|---|
| pos/s | 11,472 | 11,875 | 12,272 | **12,537** | 11,592 |

Three findings:

1. **The batch knee is at 256, not 1,024.** Batch 64 and 128 are dispatch-bound
   (6,674 and 8,232 positions/second, with p95 latencies of 545 ms and 354 ms
   against a 177 ms mean above the knee); from 256 upward the curve is flat
   within 1.5 percent. Agent 4 measured the knee at 1,024 for a *single*
   dispatch. In the pipeline the positions per global step are fixed by the
   environment count, so a larger batch buys fewer, larger dispatches rather
   than more work, and the benefit saturates earlier.
2. **Worker count peaks at 10, the performance-core count**, exactly as Agent 2
   predicted, and 12 is no better. Agent 2's warning that the end-to-end optimum
   would fall below its standalone 10 did not hold: the coordinator is
   Metal-bound rather than core-bound, so it does not take a performance core
   away from the workers.
3. **Environment count matters here, where it did not for Agent 2.** More
   environments per step amortise the fixed per-step barrier and dispatch cost
   over more positions. It is a latency trade: 1,536 environments run at 12,537
   positions/second with a 123 ms step, 256 environments at 11,472 with a 22 ms
   step.

Baselines at 8 workers / 2,048 environments / batch 2,048:

| variant | pos/s | against float16 dense |
|---|---:|---:|
| float16, dense | 11,592 | — |
| float32, dense | 10,717 | **-7.5 %** |
| float16, compact | 10,548 | **-9.0 %** |

**Sustained finalists (60 s each).** Best: **`w10_e1536_b1536_float16_dense` at
12,838 positions/second**, 18.1 games/second, 119.6 ms mean step latency and
120.7 ms at the 95th percentile. The other two finalists landed at 12,570 and
12,160.

Where the time goes at the best configuration:

| stage | share | ms/step |
|---|---:|---:|
| Metal inference | **80.87 %** | 96.75 |
| worker phase (barrier) | 10.31 % | 12.34 |
| legality + action sampling | 5.60 % | 6.70 |
| host-to-device transfer | 3.12 % | 3.73 |
| straggler spread | 0.80 % | 0.96 |
| trajectory write-back | 0.07 % | 0.09 |
| observation gather | 0.01 % | 0.01 |

Workers are active **9.7 percent** of wall time and wait at the barrier
**90.4 percent**; the coordinator is active 89.7 percent and waits 10.3 percent;
Metal is active 80.9 percent. The pipeline is model-bound by a wide margin, and
the simulation side has very large headroom.

**Memory.** Coordinator peak resident set 475 MB, summed worker peak resident
sets 3.23 GB, shared block 94.3 MB at 1,536 environments (61,386 bytes per
environment, up 525 from Agent 2's 60,861 for the three decision fields). Metal
driver-allocated high-water 4.35 GB. System swap was **0 bytes at start and
0 bytes at end**; the machine has no swap in use at all.

**The decision ratio.** The numerator was measured, not extrapolated: the same
worker pool at the same 10 workers and 1,536 environments, with the model
removed and Agent 2's deterministic benchmark policy in its place, covering
observation building, legality generation, the engine transition, the
shared-memory transport, worker synchronisation and independent reset.

```text
R = 96,963 / 14,922 = 6.50
```

| quantity | value |
|---|---:|
| Simulation-pipeline positions/second (measured, no model) | **96,963** |
| Agent 4 sustainable representative-model inference positions/second | **14,922** |
| **R** | **6.50** |
| R against Agent 4's float32 dense baseline (12,185) | 7.96 |
| R against the measured contended end-to-end rate (12,838) | 7.55 |
| R with the recording-inclusive numerator (67,209) | 4.50 |

Every variant is far above the 2.0 threshold. The most conservative figure
available — the simulation pipeline carrying full trajectory recording, against
the uncontended model rate — is still 4.50, more than twice the threshold.

**End-to-end profiling independently supports the conclusion.** The ratio and
the profile are separate measurements and they agree: Metal is active 80.9
percent of the step and the workers are idle 90.4 percent of it. The simulation
side is not the bottleneck by any reading, and a faster simulation backend would
buy nothing until the model side moves.

**Soak: 2.00 hours continuous, 119 samples at 60 s.**

| Measure | Result |
|---|---|
| Duration | **7,200.1 s (2.00 h)** |
| Global steps | 41,583 |
| Positions | **63,871,488** |
| Games completed | **123,718** |
| Independent resets | 123,718 |
| Throughput | 8,871 positions/s, 17.2 games/s |
| Step latency | 173.1 ms mean, 166.1 p50, 180.7 p95, 1,142.9 max |
| Coordinator memory growth | **0 bytes** |
| Throughput change, first vs last quarter | **-0.76 %** |
| Worker liveness | **10 / 10 for all 119 samples** |
| Errors / restarts | **0 / 0** |
| Decisions recorded | 63,871,488 |
| Decisions reconstructed during the soak | **411,818** |
| Reconstruction mismatches | **0** |
| Swap used, start -> end | **0 -> 0 bytes** |

All five acceptance conditions hold: no invariant failure, no
observation/state/reconstruction mismatch, no deadlock, no memory growth trend
(resident set was flat at 453 MB from the first sample to the last), no swapping
at all, and no throughput collapse — the -0.76 percent drift between the first
and last quarter is inside the sample-to-sample noise. The single 1,143 ms step
against a 181 ms p95 is one outlier in 41,583 steps and is not attached to any
error, liveness change or memory event.

The soak runs 30.9 percent below the finalist rate because it records
trajectories and the finalist did not; 8,871 positions/second is the honest
production figure for a collecting pipeline.

Game statistics over 123,718 games: mean 516 plies, and **four of the five
terminal reasons occurred naturally** — `flag_capture` 55.1 percent,
`battleless_move_limit_draw` 43.0 percent, `opponent_no_legal_move` 1.9 percent,
and `both_no_legal_move_draw` twice. Only `absolute_move_limit_draw` did not
appear, which is expected: the battleless limit preempts it. Agent 2 saw only
two reasons naturally; this is the first Phase 3 run at a scale that surfaces
the rare ones.

**Storage.** 11.17 GiB of encoded trajectory recorded in two hours — 5.59 GiB/h,
96,965 bytes per game, 187.8 bytes per decision against Agent 3's measured
154.3. The difference is the policy row: these decisions store a real model
distribution over a mean 24-entry legal set, where Agent 3's synthetic policy
was sparser. Records were encoded, counted, sampled for verification and then
discarded, so the soak wrote nothing to disk.

### 5.6 Deviations and limitations

- **A latent illegal-action bug was found in Agent 4's sampler and fixed.** This
  is the most important non-performance result of this agent, so it is stated in
  full.

  `_gumbel_noise` drew `u` from `torch.rand`, which returns values in `[0, 1)` —
  zero included — and clamped only the upper end. When `u` was exactly `0`,
  `log1p(-u)` was exactly `0`, the outer `log` was `-inf`, and the noise was
  `+inf`. Added to the `-inf` that `apply_dense_legality` writes at every illegal
  entry, that gives `NaN`; `torch.argmax` ranks `NaN` above every finite value,
  so the sample landed on an action the engine had declared illegal.

  The rate is about one draw in 17 million (`2**-24`). At Agent 4's scale it is
  invisible — Agent 4's acceptance genuinely observed zero illegal samples, and
  that result stands for the number of draws it took. A sustained pipeline draws
  10,000 Gumbel values per position, so at 1,536 environments it passes 15
  million draws per *step*; the first failure here arrived at step 60. It
  reproduces on the central processing unit as well, so it is not a Metal issue.

  It was found by the frozen engine refusing the action —
  `BatchIllegalActionError`, with no slot in the batch modified. **The engine
  behaved exactly as specified and is not implicated**; it is the reason the bug
  surfaced as a loud stop rather than as corrupted training data.
  `representative_model.py` is Agent 4's Phase 3 benchmark probe, not the frozen
  reference engine, so it was fixed here rather than only reported: the uniform
  draw is now clamped at both ends, bounding the noise to about +/-16.1 and
  moving probability mass of `1e-7` at each tail. Four regression tests pin the
  mechanism, and the configuration that failed at step 60 now runs 4,000 steps —
  about 61 billion draws, where the old rate predicts roughly 3,600 failures —
  with none.

  A second, independent guard was added: the coordinator checks every sampled
  action against the mask it was drawn from before the workers see it, so a
  recurrence is named where it happens instead of surfacing as an opaque worker
  fault a phase later. It costs one gather per step.

  None of Agent 4's throughput or equivalence results are invalidated. The fix
  adds one clamp and changes the sampled distribution by `1e-7` per tail.

- **Compact legality is 9.0 percent slower end to end than dense, reversing the
  sign of Agent 4's stage measurement.** Agent 4 measured the compact legality
  *stage* as 7.5x cheaper on the device and compact end-to-end as a few percent
  faster, having built the compact representation outside the timed region on
  the reasonable assumption that a coordinator would hold the legal lists. It
  does not: the shared transport carries the dense mask the engine produced, so
  the coordinator has to *build* the compact form, and that build is 21.1
  percent of the step — more than the device stage saves. This is not a
  contradiction of Agent 4's numbers; it is the cost Agent 4 explicitly placed
  outside its timed region, measured where it actually falls. Agent 4's
  recommendation of dense stands, for a second and independent reason.

- **The coordinator always builds the compact legal set when it is recording**,
  whichever legality representation the model uses, because a stored decision
  carries one probability per legal action in ascending order. When recording is
  off the build is skipped entirely — it is 0.00 percent of the step in the
  finalist runs — so a pure throughput measurement does not pay for it.

- **float16 is worth less end to end than in isolation**: +7.5 percent over the
  float32 dense baseline here against Agent 4's +19.6 percent, because the
  pipeline step contains a worker phase and a transfer that the precision does
  not touch. It remains the recommended precision.

- **A terminal slot never reaches the model.** The pool resets a finished game
  inside the same phase it finished, so every published row is active with at
  least one legal action. The coordinator marks `decision_valid` only for active
  slots, and a worker raises if it is ever marked for a terminal one.

- **Recording costs about 13 microseconds per decision; digesting one for
  verification costs roughly 35 times that**, because a digest builds a full
  observation, belief target and public-knowledge view and hashes them. The
  correctness gate spends that deliberately; the soak samples it at one game per
  worker at a time, which kept it near 0.3 percent of worker time while still
  verifying 411,818 decisions spread across the whole two hours rather than only
  at the start.

- **Trajectory records are built, encoded, counted, sampled for verification and
  then discarded**, on the operator's instruction. A full-rate two-hour
  collection is 11.17 GiB and the machine had 45 GiB free. The storage path is
  exercised at its real cost and the byte totals are reported; the corpus is not
  persisted. A handful of retained records come back for inspection —
  `retain_games` is per worker, so the count returned is `retain_games x
  workers`.

- **The end-to-end rate is 86 percent of Agent 4's isolated ceiling**
  (12,838 against 14,922). The gap is contention plus the bulk-synchronous
  barrier, and it is the expected direction: Agent 4 explicitly warned its
  numbers were single-process ceilings.

- **The loop is strictly bulk-synchronous, as specified.** The coordinator is
  idle while the workers step and the workers are idle while the coordinator
  infers. At the best configuration the worker phase is 10.3 percent of the
  step, so overlapping the two is worth at most about 10 percent. That is the
  largest single recoverable inefficiency measured and it is not this agent's
  ground.

- **The shared block grew 525 bytes per environment**, from Agent 2's 60,861 to
  61,386, for the three decision fields — 94.3 MB at 1,536 environments.
  `COORDINATOR_WRITTEN_FIELDS` widened accordingly, so Agent 2's test pinning
  that set to exactly `{actions, reset_request}` was updated. Recording counters
  are emitted in the per-phase reply **only** when recording is enabled, so a
  pool that does not record sends exactly the reply Agent 2 specified.

- **Soak acceptance for memory growth, swap and throughput collapse is assessed
  from the reported measurements rather than auto-failed on a threshold.** The
  measurements are unambiguous here — 0 bytes of growth, 0 bytes of swap,
  -0.76 percent drift — but a future run nearer a boundary needs a human
  reading, not a magic number.

- **The model is Agent 4's untrained benchmark probe.** Every throughput number
  scales with the final architecture and must be re-measured once it is chosen.
  Game-length and terminal-reason statistics describe the probe's play, not
  Stratego. This agent owns integration and measurement, not playing strength.

- **`--quick` mode** runs the same code paths at reduced scale, reports status
  `QUICK`, and does not meet the acceptance thresholds; only the full run does.

### 5.7 Data files

```text
reports/phase_3_data/agent_05_end_to_end.json
reports/phase_3_data/agent_05_end_to_end_raw.csv
reports/phase_3_data/agent_05_soak_timeseries.csv
```

The JSON holds the platform and Metal device record, both correctness gates with
their full comparison counts and category breakdowns, all 24 screened and 3
finalist configurations with every stage timing and wait fraction, both
simulation-pipeline measurements, the ratio and decision with its inputs and
thresholds, and the soak summary. The first CSV is one row per measured
configuration; the second is one row per 60-second soak sample. No reproduction
file was written because there were no mismatches.

`files_created` and `files_modified` in the JSON are harness metadata; the
authoritative list is section 5.3 above.

### 5.8 Handoff and closing notes

- Agent 6 is **not** required. This is the normal end of Phase 3 implementation.
- **Start Phase 4 collection at 10 workers, 1,536 environments, inference batch
  1,536, float16, dense legality, snapshot interval 32.** Expect roughly 8,900
  positions/second with recording on and 5.6 GiB of encoded trajectory per hour.
- **Do not re-derive the batch size from Agent 4.** In the pipeline the knee is
  at 256, not 1,024, because positions per step are set by the environment
  count. Anything from 256 up is within 1.5 percent; 1,536 was chosen to match
  the environment count so each step is a single dispatch.
- **The model is the bottleneck and the simulator has 7.5x headroom.** When the
  real architecture replaces the probe, re-measure `R` before assuming the
  headroom survives — a model an order of magnitude cheaper would move the
  decision, and a model more expensive than the probe only widens it.
- **Keep the coordinator the sole owner of the device.** Nothing in
  `coordinator.py`, `representative_model.py` or `mps_benchmark.py` is safe to
  import into a simulation worker; a test enforces this for `worker_pool`.
- The compact legality path is proven equivalent and is available, but on this
  transport it costs more than it saves. Revisit only if workers publish the
  compact form directly.
- `belief_targets` and the true board remain absent from every shared field.

### 5.9 Recommended project-document updates

`stratego_project_docs/` was **not** modified. The following are recommended for
the later review that updates those documents once these results are accepted:

1. **`05_project_plan.md`** — record the Phase 3 backend decision
   (`KEEP_PYTHON`, `R = 6.50`) and that Agent 6 is not required; note the
   measured production configuration and the 8,871 positions/second collecting
   rate.
2. **`03_game_engine_spec.md` section 16** — document the three
   coordinator-written decision fields and the ascending-legal-order contract
   that lets a worker line probabilities up without transmitting identifiers.
3. **`09_public_event_and_replay_schema.md`** — record
   `end_to_end_representative_probe_v1` as a collection policy version alongside
   Agent 3's `synthetic_hash_policy_v1`, and the measured 187.8 bytes per
   decision for model-generated policy rows.
4. **`04_engine_validation_plan.md`** — add the Gumbel-max sampling failure as a
   named regression, and the general rule that a sampler mixing `-inf` masking
   with `argmax` must be proven to produce finite noise. Its discovery is also a
   good argument for keeping the engine's illegal-action rejection loud.
5. **A Phase 4 storage note** — 5.6 GiB per hour of collection at the
   recommended configuration is the planning figure for corpus sizing.

Phase 3 backend decision: KEEP_PYTHON
Measured R: 6.50
Agent 6 required: no
