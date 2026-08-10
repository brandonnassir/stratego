# Phase 3 High-Throughput Self-Play Architecture

## 1. Status

**Status: Accepted — Phase 3 complete.**

Backend decision:

- `KEEP_PYTHON`
- measured \(R=6.50\)
- optimized-backend Agent 6 not required

Frozen behavioral source of truth:

- `phase2_1_reference_1.1.0`
- `stratego_project_v1`
- `observation_v2_1_127ch`
- 10,000-entry source-destination action encoding

The Phase 3 systems work did not change `stratego/engine/`.

---

## 2. Purpose

Phase 3 answered:

> Can the frozen Python simulator generate and transport self-play positions fast enough to keep the representative Metal model busy, or is a second optimized simulator necessary?

The accepted answer is:

> The current system is model/Metal-bound. The Python simulator has substantial headroom and should be retained.

---

## 3. Accepted architecture

```text
CPU worker 0 ─┐
CPU worker 1 ─┤
...           ├─> persistent shared-memory block
CPU worker N ─┘              |
                              v
                      single coordinator
                              |
                              v
                   PyTorch model on Metal
                              |
                              v
                 coordinator decision fields
                              |
                              v
                    CPU simulation workers
                              |
                              v
              compact trajectory construction
```

Design rules:

1. collection is bulk-synchronous;
2. only the coordinator touches Metal;
3. workers own disjoint game slots and game objects;
4. bulk observations/masks/actions use persistent shared memory;
5. pipes carry only small control/status messages;
6. completed environments reset independently;
7. trajectories are compact and reconstructed later;
8. hidden/belief targets are never model-facing shared-memory inputs.

---

## 4. Environment identity and determinism

Each slot has:

```text
environment_id
generation
```

`environment_id` is fixed.

`generation` increments once per reset.

A game/trajectory instance is identified by:

```text
(environment_id, generation)
```

Slot seeds derive from:

```text
(root_seed, environment_id, generation)
```

This makes slot generation reproducible independently of:

- worker count;
- worker assignment;
- neighboring game history;
- process completion order.

---

## 5. Accepted Phase 3 measurements

### 5.1 Simulation-only pipeline

Selected worker/environment configuration:

- 10 workers;
- 1,536 environments.

Measured simulation pipeline:

- **96,963 positions/s** for the accepted decision numerator.

Agent 2's standalone longer scaling measurement was around 91,482 positions/s at 10 workers x 1,536 environments; Agent 5 re-measured the decision numerator in the integrated harness rather than reusing that earlier rate.

### 5.2 Representative model

The Phase 3 model was an untrained systems probe:

- 100 square tokens;
- 127 input features/token;
- width 128;
- 4 Transformer blocks;
- 4 attention heads;
- feedforward width 512;
- source-query/destination-key policy head;
- three-class value head;
- lightweight shared belief probe;
- 873,999 parameters.

It is **not** the frozen final architecture.

Sustainable representative denominator used for the backend decision:

- **14,922 positions/s**.

### 5.3 Decision ratio

\[
R =
\frac{96{,}963}{14{,}922}
=
6.50.
\]

Decision thresholds:

- `R >= 2.0`: keep Python;
- `1.25 <= R < 2.0`: keep Python initially;
- `R < 1.25`: evaluate optimized backend.

Result:

- **KEEP_PYTHON**.

The recording-inclusive simulation numerator still yielded \(R=4.50\).

---

## 6. Integrated bottleneck profile

Best 60-second finalist:

- 10 workers;
- 1,536 environments;
- inference batch 1,536;
- float16;
- dense legality;
- **12,838 positions/s**.

Approximate step share:

| Stage | Share |
|---|---:|
| Metal inference | 80.87% |
| worker/barrier phase | 10.31% |
| legality + sampling | 5.60% |
| host-to-device transfer | 3.12% |
| straggler spread | 0.80% |
| trajectory write-back | 0.07% |
| observation gather | 0.01% |

Workers were idle for most of the model phase. Simulator optimization is therefore not the current priority.

---

## 7. Accepted starting configuration

For the next model-backed systems benchmark with a similarly sized model:

```text
workers             10
environments        1536
inference batch     1536
precision           float16
live legality       dense
snapshot interval   32 plies
```

This is a **starting point**, not a permanent training constant.

The final model architecture must re-benchmark:

- worker count;
- environment count;
- inference/training batch size;
- precision;
- memory;
- \(R\).

---

## 8. Dense versus compact legality

Agent 4 showed compact legality was cheap once already represented compactly.

Agent 5 measured the real production transport:

- workers publish dense engine masks;
- coordinator must construct compact legality before using it.

Integrated baseline at 8 workers / 2,048 environments / batch 2,048:

| Variant | Positions/s |
|---|---:|
| float16 dense | 11,592 |
| float16 compact | 10,548 |

Compact was approximately 9% slower end to end.

Accepted default:

- **dense legality**.

Compact remains a validated optional representation if the transport is later redesigned to publish compact legal IDs directly.

---

## 9. Two-hour soak

Accepted soak:

- duration: 7,200.1 s;
- positions: 63,871,488;
- games: 123,718;
- collecting throughput: 8,871 positions/s;
- workers alive: 10/10 throughout;
- errors/restarts: 0/0;
- reconstruction checks: 411,818;
- reconstruction mismatches: 0;
- coordinator memory growth: 0 bytes;
- swap: 0 -> 0 bytes;
- first-to-last-quarter throughput drift: -0.76%.

This passes the Phase 3 production-readiness stability gate.

---

## 10. Future optimized backend trigger

Do not implement a second backend merely because one might be faster.

Reconsider only if a future real model changes the measured balance enough that:

- \(R < 2.0\), or
- end-to-end profiling independently shows simulation materially limits throughput.

Any optimized backend must be separate from the frozen reference engine and pass its full differential acceptance suite before training use.
