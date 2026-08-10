# Batch Simulation and Shared-Memory Specification

## 1. Status

**Status: Accepted Phase 3 production interface.**

This document describes the training-facing batch/transport contract implemented and validated in Phase 3.

It does not redefine game rules or observation semantics.

Behavioral oracle:

- `phase2_1_reference_1.1.0`.

---

## 2. Batch semantics

A batch owns `N` independent Stratego environments.

Required operations:

- create deterministic environments;
- query acting players;
- obtain acting-player observations;
- obtain legal-action lists/masks;
- apply one action per active slot;
- expose terminal outcome/reason;
- independently reset selected finished slots;
- preserve slot identity across resets.

A submitted batch step is validated before mutation. Illegal actions must not partially mutate other slots.

---

## 3. Slot identity

Each slot contains:

```text
environment_id
generation
```

Contract:

- `environment_id` is fixed for that slot;
- `generation` starts at 0;
- every reset increments `generation` exactly once;
- `(environment_id, generation)` identifies one game instance.

Global worker partitioning must preserve globally unique environment identifiers.

Accepted worker implementation supports an explicit first-environment offset so workers own disjoint global ID ranges.

---

## 4. Deterministic seeding

Seed derivation depends only on:

```text
root_seed
environment_id
generation
```

It must not depend on:

- local worker index;
- number of workers;
- reset order;
- neighboring games;
- scheduling order.

This permits independent reconstruction of any slot generation.

---

## 5. Worker partitioning

Workers own disjoint, fixed contiguous slot ranges.

The worker owns:

- its `BatchSimulator`;
- all `GameState` objects in its range;
- legal-action lists;
- privileged belief targets;
- trajectory builders;
- game snapshots.

The coordinator owns no `GameState`.

---

## 6. Shared-memory architecture

Use one persistent preallocated shared-memory block per pool.

Properties:

- allocated once;
- field-major;
- bulk arrays C-contiguous where required;
- 64-byte aligned in the accepted implementation;
- no per-step resize;
- explicit writer ownership;
- stale-publication detection.

### 6.1 Worker-written model/public fields

Accepted fields include:

| Field | Shape | Dtype | Meaning |
|---|---|---|---|
| `observations` | `(N,127,10,10)` | `float32` | acting-player `observation_v2_1_127ch` |
| `legal_mask` | `(N,10000)` | `uint8` | engine legality |
| `legal_count` | `(N,)` | `int32` | number of legal actions |
| `acting_player` | `(N,)` | `int8` | player to move |
| `environment_id` | `(N,)` | `int32` | fixed slot ID |
| `generation` | `(N,)` | `int32` | current game generation |
| `ply` | `(N,)` | `int32` | current move count |
| `battleless_moves` | `(N,)` | `int32` | current no-battle counter |
| `terminal` | `(N,)` | `uint8` | terminal flag |
| `status` | `(N,)` | `int8` | slot lifecycle |
| `worker_id` | `(N,)` | `int16` | slot owner |
| `publish_sequence` | `(N,)` | `int64` | freshness/staleness counter |
| `episode_count` | `(N,)` | `int32` | completed games in slot |

The block also carries last-finished-game metadata required for terminal accounting without returning game objects.

### 6.2 Coordinator-written control fields

Accepted fields include:

| Field | Shape | Dtype | Meaning |
|---|---|---|---|
| `actions` | `(N,)` | `int32` | selected action; negative means skip |
| `reset_request` | `(N,)` | `uint8` | request slot reset |
| `policy_probabilities` | `(N,128)` | `float32` | legal-prefix policy probabilities, padded |
| `value_prediction` | `(N,3)` | `float32` | win/draw/loss |
| `decision_valid` | `(N,)` | small integer/bool | decision is valid for recording |

The 128 policy capacity is a transport/storage implementation capacity, not a legal-move theorem. Overflow must never truncate legal actions.

---

## 7. Hidden-information boundary

Shared model-facing fields must not contain:

- true unresolved opponent types;
- privileged opponent setup identities;
- belief targets;
- privileged full state.

Only the acting player's legally available 127-channel observation and engine legality are published for model use.

Belief targets remain inside the worker/reference-state side or are reconstructed later for training labels.

---

## 8. Synchronization and stale-buffer safety

One phase:

1. workers publish state;
2. coordinator waits for all worker phase completions;
3. coordinator verifies publication freshness;
4. model inference/action selection occurs;
5. coordinator writes decision fields;
6. workers consume decisions and step;
7. workers seal terminal games/reset as configured;
8. workers publish next state.

`publish_sequence` must advance for maintained rows.

If a worker:

- crashes;
- raises;
- hangs past timeout;
- fails to republish;

the coordinator must surface a named infrastructure failure and must not treat stale shared memory as current game state.

---

## 9. Control communication

Pipes/queues may carry only small control/status payloads such as:

- command;
- sequence;
- flags;
- counters;
- timing summaries;
- worker fault information;
- encoded finished trajectory bytes when explicitly retained/transferred.

Do not send per-position game objects, observation tensors, or dense masks through object serialization.

---

## 10. Worker threading

Simulation workers must prevent numerical-library oversubscription.

The accepted Phase 3 implementation constrained:

- `OMP_NUM_THREADS`;
- `OPENBLAS_NUM_THREADS`;
- `MKL_NUM_THREADS`;
- `VECLIB_MAXIMUM_THREADS`;
- `NUMEXPR_NUM_THREADS`;

to one thread in workers.

---

## 11. Model ownership

Only the coordinator may import/invoke the Metal model path.

Simulation workers must remain usable without PyTorch.

The accepted implementation has tests ensuring the worker-pool import path does not pull PyTorch into worker processes.

---

## 12. Decision-record ordering

The worker's legal-action list is ascending.

The dense mask's nonzero indices use the same ascending action-ID order.

Therefore:

```text
policy_probabilities[i]
```

for the legal prefix aligns with:

```text
legal_action_ids[i]
```

without re-sending the identifiers from coordinator to worker.

The worker records the decision **before** applying the action.

---

## 13. Accepted validation evidence

Phase 3 Agent 2:

- 30,272 cross-process steps;
- 0 equivalence mismatches;
- 5,120 reset events;
- 0 reset/generation errors;
- 3/3 worker-failure modes detected;
- 0 deadlocks.

Phase 3 Agent 5:

- 10,048 full integrated environment-step comparisons;
- 0 mismatches;
- 11,251 integrated trajectory reconstructions;
- 0 mismatches;
- two-hour soak with 0 errors/restarts.

---

## 14. Performance baseline

Accepted initial collection configuration:

```text
workers          10
environments     1536
inference batch  1536
precision        float16
legality         dense
```

Best integrated finalist without full recording:

- 12,838 positions/s.

Two-hour trajectory-recording soak:

- 8,871 positions/s.

These values must be re-measured with the final model.
