# Phase 17 — Agent 6
## Integrated preflight, evidence audit, and launch authorization

## Mission

Decide whether the exact Phase 17 build is safe to launch for 12 hours. Integrate and
audit Agents 1–5, run only the bounded tests needed to catch silent false results,
verify one external round trip, and emit a digest-bound GO or NO-GO record.

You do not redesign the models, tune the schedules, repair large unrelated defects,
or start the production run. Read `00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md`
completely and require all five upstream handoffs.

## 1. Process and artifact boundary

Confirm no other learner or heavy evaluator owns the compute resource. Recompute every
handoff and artifact digest. Verify that the source commit/config differs from no file
after Agent 4/5 evidence was generated, except explicitly permitted report appendices.

If source, contract, checkpoint, benchmark, environment, or handoff identity is
ambiguous, issue NO-GO. Do not fix provenance by editing a report after the fact.

## 2. Correctness gate — at most 30 minutes

Run a single targeted gate covering:

- exact Phase 9 file/model-state identity and weights-only fresh-state initialization;
- project rules, battleless-100, absolute-4000, observation/action identities;
- Red and Blue canonical-to-engine setup orientation;
- exact inventory and legal action behavior;
- sampled-not-argmax move selection;
- both seats acting from the current raw snapshot;
- forced rebind of an already-running game after a move update;
- exactly fixed transition rows and unfinished-boundary bootstrapping;
- no search, belief auxiliary, historical, or handcrafted training participant;
- raw-versus-EMA separation;
- joint checkpoint/load/export identity and exact continuation.

Use existing Agent 2/4 evidence when it is directly digest-bound, but rerun the
smallest end-to-end path that proves the integrated build—not merely unit modules.

## 3. Setup evidence gate

Audit Agent 3's full 5,000-sample and short-soak artifacts against the exact setup
model/config now integrated. Recompute a small spot sample for identity. Do not repeat
the extended experiment unless evidence is missing, mismatched, or invalid.

Require:

- 4×128, 4-head, FF512 architecture and accepted parameter count;
- zero hard legality/orientation/masking/causality failures;
- real completed outcomes changed the raw setup model through nonzero gradients;
- reward-sign and behavior-snapshot binding passed;
- five setup epochs were retained, or the operator approved the measured alternative;
- calibrated diversity/entropy thresholds are present and supervisor-readable;
- setup raw/EMA/optimizer/KL/RNG/queue round trip passed.

## 4. External gate

Require Agent 5's operator-approved topology and a successful full paired h0 candidate
round trip. Recompute the returned receipt and verify:

- source candidate and parent checkpoint;
- move/setup EMA model-state identities;
- composite pack and both lane identities;
- evaluator source/environment/host identity;
- overall and stratum result-file digest;
- end-to-end cadence below the accepted limit;
- unattended queue behavior and return path.

If the remote MacBook requires continued manual action every 30 minutes, it does not
pass the unattended production requirement unless the operator explicitly changes the
contract.

## 5. Bounded integration rehearsal

Run a 20–30 minute rehearsal of the exact production command/config when feasible. It
must cross move and setup updates, a 30-minute export boundary may be accelerated only
through an explicit test clock, and checkpoint/resume must be exercised once.

Verify:

- stable fixed transition and optimizer sizes;
- setup queue supplies real five-epoch updates;
- every active decision has the current move digest;
- schedules match the frozen precomputed curve;
- raw and EMA states evolve in their intended roles;
- telemetry remains append-only across resume;
- one injected guard safely checkpoints and stops;
- external bundle enqueue does not block or mutate training.

Do not evaluate playing strength or alter a threshold based on this rehearsal.

## 6. Launch closure

Create an immutable launch manifest binding:

- run ID and source commit;
- every upstream handoff and code/config digest;
- exact Phase 9 identity;
- schedule horizon `N`, `n_ref`, all constants and epoch counts;
- population, transition, setup-pool, queue, and setup-update budgets;
- raw/EMA/checkpoint/export schemas;
- composite benchmark and external environment;
- guard thresholds and exact production command;
- expected h0 through h12 candidate schedule;
- all gate evidence and any accepted operator deviation.

The decision record is one of:

```text
GO     every mandatory gate passed and operator launch approval is still required
NO-GO  at least one mandatory gate failed, mismatched, or was not run
```

There is no conditional GO. A nonessential production-quality issue may be listed as
accepted risk only if it cannot produce silent false evidence and the operator agrees.

## 7. Handoff and report

Deliver:

```text
reports/phase17/phase17_launch_decision_v1.json
reports/phase17/phase17_launch_manifest_v1.json
reports/phase17/agent_06_report.md
```

Sign/digest the launch files after all referenced artifacts exist. A GO becomes invalid
if source, config, checkpoint, pack, environment, or command changes afterward. Agent 7
must re-verify it immediately before launch.
