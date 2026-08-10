# Game Engine Specification

## 1. Objective

Build a Stratego simulation layer that is:

- correct;
- deterministic when seeded;
- easy to inspect;
- fast enough for large-scale local self-play;
- compatible with reinforcement learning;
- compatible with decision-time search;
- usable by a browser interface through a higher-level service;
- replaceable by a faster backend without changing the model or training logic.

The first implementation is a **readable Python reference engine**. Phase 3 profiling demonstrated that this engine is sufficiently fast for the current model scale, so it is also the selected production simulation backend. A separate optimized backend remains a future option only if a later real-model benchmark changes the bottleneck decision.

### Frozen reference implementation and production decision

Phase 2.1 accepted and froze:

- implementation: `phase2_1_reference_1.1.0`;
- rules: `stratego_project_v1`;
- observation: `observation_v2_1_127ch`;
- action encoding: `action_id = 100 * source + destination`, 10,000 entries;
- deterministic replay semantics defined by the replay/event contracts.

Phase 3 left `stratego/engine/` unchanged and selected this same Python implementation as the production simulation backend:

- backend decision: `KEEP_PYTHON`;
- measured simulation/model ratio: \(R=6.50\);
- conditional optimized-backend Agent 6: **not required**.

The reference engine remains the behavioral oracle. It must not be optimized in place. If a separate faster backend is introduced later, it must remain interchangeable and pass differential testing against this frozen implementation.

---

## 2. Engine architecture

```text
Reference Python engine
        |
        |  authoritative behavior
        v
Engine interface / data contracts
        ^
        |
Optimized backend (later, only if profiling justifies it)
        |
        v
Training coordinator / evaluation / search / browser service
```

The optimized backend must be behaviorally interchangeable with the Python reference engine.

---

## 3. Design principles

1. **Correctness before speed.**
2. **Full state and player observation are separate objects/concepts.**
3. **No hidden information may leak through the policy observation interface.**
4. **Rules are configuration-driven where the project intentionally varies training and evaluation conditions.**
5. **All randomness is seedable and replayable.**
6. **Game histories are stored compactly.**
7. **The engine exposes batches of games rather than forcing one-game-at-a-time training.**
8. **The model-facing interface must not depend on whether the backend is Python or optimized native code.**

---

## 4. Canonical board representation

### Internal board

Use a fixed 10 by 10 coordinate system.

Recommended internal coordinates:

- row index: `0..9`;
- column index: `0..9`.

Recommended human notation:

- columns `a..j`;
- rows `1..10`.

A conversion utility must map exactly between the two.

### Lakes

Lake squares are constants and must never contain pieces.

The exact lake mask must have:

- 8 non-occupiable cells;
- 92 occupiable cells.

---

## 5. Piece representation

Every live piece must have at least:

- stable piece identifier unique within the game;
- owner: red or blue;
- piece type;
- current square;
- whether the piece has moved;
- whether its identity is public to the opponent;
- starting square.

Historical/public information that belongs to the information state should be stored or reconstructible separately from the minimal transition state.

---

## 6. Piece type identifiers

Use one stable enumeration across the engine, training data, interface, and model documentation.

Recommended order:

1. Spy
2. Scout
3. Miner
4. Sergeant
5. Lieutenant
6. Captain
7. Major
8. Colonel
9. General
10. Marshal
11. Flag
12. Bomb

The identifier order is an implementation convention; ranks are game rules and must not be inferred from identifier position for Flag or Bomb.

---

## 7. Setup representation

A setup consists of the 40 piece types assigned to the player's 40 setup squares in a fixed row-major order.

The engine must support:

- validating a supplied setup;
- loading a supplied setup;
- generating a random legal setup for testing;
- applying a setup produced by the project setup generator;
- reflecting a setup left-to-right;
- serializing and deserializing a setup.

A setup validator must reject incorrect piece counts, duplicate/missing placements, lake placements, or placements outside the legal setup zone.

---

## 8. Action representation

Use a source-destination move representation because it naturally supports both ordinary moves and long Scout moves.

Recommended canonical action:

- source square index `0..99`;
- destination square index `0..99`.

Recommended dense model action identifier:

\[
\text{action\_id}=100\times\text{source}+\text{destination}.
\]

This yields a fixed 10,000-entry source-destination space. Only legal actions receive nonzero probability after masking.

Advantages:

- simple;
- stable across model versions;
- compatible with a query-key move head;
- long Scout moves require no special action type;
- interface conversion is straightforward.

The engine remains responsible for legality. The model is never trusted to determine whether an action is legal.

---

## 9. Legal-action generation

Given a full state and acting player, the engine must return all legal actions.

Legal-action generation must correctly handle:

- one-square cardinal movement;
- Scout ray movement;
- friendly-piece blocking;
- enemy destination attack;
- lake blocking;
- board edges;
- immovable Flag and Bomb;
- terminal states;
- project no-battle rules only as termination rules, not move restrictions;
- deliberate absence of two-square and continuous-chasing restrictions.

The action mask must be deterministic and exactly consistent with the returned legal-action list.

---

## 10. State transition

Applying one legal action must update all relevant state atomically:

- source and destination occupancy;
- piece position;
- battle result, if any;
- captured-piece records;
- public identity/reveal records;
- moved status;
- Scout logical reveal when applicable;
- acting player;
- total move counter;
- no-battle counter;
- terminal state;
- terminal reason;
- game result.

Illegal actions must never partially mutate the state.

---

## 11. Terminal reasons

Use explicit terminal-reason labels rather than only a generic `done` flag.

Required reasons:

- `flag_capture`;
- `opponent_no_legal_move`;
- `both_no_legal_move_draw`;
- `battleless_move_limit_draw`;
- `absolute_move_limit_draw`;
- `not_terminal`.

This is important for diagnosing self-play behavior.

---

## 12. Deterministic replay

Every game must be reproducible from:

- rules version;
- red setup;
- blue setup;
- first player;
- action sequence.

If random setup generation or random actions are involved, the record must additionally store the relevant seed or generated result.

Replay must reproduce exactly:

- every board state;
- every reveal;
- every combat result;
- legal moves at every step;
- counters;
- terminal result.

---

## 13. Snapshot and restore

The engine must support creating a compact snapshot of a nonterminal state and restoring from it.

This is needed for later decision-time search.

A restored state must produce the same:

- legal actions;
- player observation;
- future transition under the same action sequence;
- terminal result.

Search support is a design requirement even though search will be implemented after the policy is trained.

---

## 14. Player observation contract

The model receives a player-relative observation, not the full state.

The acting player's own side must always be normalized to the same orientation before model input. This allows one model to play both colors.

### Approved compact representation: version 2

The authoritative observation identifier is:

- `observation_v2_1_127ch`

Shape:

\[
127 \times 10 \times 10.
\]

The representation contains:

- 12 current own-piece identity planes;
- 12 current known-opponent identity planes;
- 1 hidden-opponent occupancy plane;
- 1 own-piece-known-to-opponent plane;
- 2 moved-status planes;
- 4 live-piece starting-coordinate planes;
- 12 persistent own-setup planes;
- 12 known-opponent setup-identity planes;
- 12 unresolved-opponent-inventory planes;
- 20 own-piece behavioral planes;
- 20 opponent-piece behavioral planes;
- 16 recent-move planes;
- 3 global/static planes.

Final total: **127 planes**.

Behavioral features represent five event types for each player—threat, evade, declined attack, protect, and was protected—with four features per event: recency, counterpart rank when legally knowable, whether the actor knew the counterpart identity at event time, and special Bomb/Flag context when legally knowable.

The complete channel ordering, normalization formulas, hidden-information safety rules, and formal behavioral definitions are specified in:

- `06_observation_v2_127ch.md`

The channel-by-channel validation contract is specified in:

- `07_observation_validation_matrix.md`

### Separate model inputs

The following are supplied separately rather than counted as state planes:

- legal-action mask;
- acting-player metadata before perspective normalization;
- game identifier / training metadata.

### Representation versioning

Any change to channel semantics requires a new observation identifier. The earlier draft `observation_v1_68ch` is superseded and must not be used by the initial model implementation.

---

## 15. Belief targets

During training-data generation only, the engine must be able to produce ground-truth labels for the hidden opponent pieces.

For each opponent piece currently hidden from the acting player, provide:

- board square;
- true piece type.

These labels must never be included in the policy observation.

---

## 16. Batch simulation and production self-play interface

Phase 3 implemented and accepted a **bulk-synchronous** multiprocess collection architecture:

```text
CPU simulation workers
        |
        | worker-published shared-memory state
        v
persistent shared-memory block
        |
        v
single coordinator process
        |
        v
PyTorch model on Apple Metal
        |
        | coordinator-written decisions
        v
persistent shared-memory block
        |
        v
CPU simulation workers
```

### 16.1 Ownership

- exactly one coordinator process owns/invokes the PyTorch/Metal model;
- simulation workers remain central-processing-unit-only;
- each worker owns a fixed, disjoint range of environment slots;
- the coordinator owns no `GameState`;
- game objects, observations, and legality masks are never sent as per-step Python object payloads through pipes.

### 16.2 Environment identity

Each persistent slot has:

- fixed `environment_id`;
- monotonically increasing `generation`;
- trajectory identity `(environment_id, generation)`.

A reset increments `generation` exactly once and does not affect neighboring slots.

Slot seed derivation is deterministic from:

```text
(root_seed, environment_id, generation)
```

so any generation can be rebuilt independently.

### 16.3 Worker-published shared-memory fields

The accepted production transport includes model-facing/public control data such as:

- `observations`: `(N, 127, 10, 10)` `float32`, acting-player perspective only;
- `legal_mask`: `(N, 10000)` `uint8`;
- `legal_count`;
- `acting_player`;
- `environment_id`;
- `generation`;
- `ply`;
- `battleless_moves`;
- `terminal` / `status`;
- worker ownership and publish-sequence/staleness metadata;
- last-finished-game terminal/result metadata.

Privileged belief targets and true hidden identities are not present in model-facing shared memory.

### 16.4 Coordinator-written shared-memory fields

The coordinator writes:

- selected `actions`;
- `reset_request`;
- `policy_probabilities`: one `float32` probability for each ascending legal-action entry, zero-padded to an implementation capacity of 128;
- `value_prediction`: win/draw/loss;
- `decision_valid`.

The worker already owns the exact ascending legal-action list, so policy probabilities can be aligned without transmitting action identifiers back to the worker.

The padding capacity of 128 is an implementation capacity, **not a game-rule bound**. A position exceeding it must fail loudly or use a future correctness-preserving fallback; legal actions must never be truncated.

### 16.5 Synchronization

One production step is:

```text
workers publish observations/legal state
-> barrier
-> coordinator gathers ready rows
-> observation layout conversion/tokenization
-> host-to-device transfer
-> model inference
-> legality + action sampling
-> coordinator validates sampled actions against legality
-> action/policy/value write-back
-> workers record the decision before mutation
-> workers apply actions
-> completed games are sealed
-> completed slots independently reset
-> workers publish next state
```

`publish_sequence` is the stale-buffer contract. The coordinator must not consume a row that its owning worker failed to republish.

### 16.6 Accepted Phase 3 scaling result

The accepted starting configuration for similarly sized model-backed collection is:

- 10 simulation workers;
- 1,536 simultaneous environments;
- inference batch 1,536;
- `float16` representative inference;
- dense live legality;
- 32-ply trajectory snapshots.

Phase 3 screened all required worker/environment/batch dimensions and measured the full end-to-end path. These values remain tunable and must be re-benchmarked for the final model.

Detailed Phase 3 architecture and transport requirements are also summarized in:

- `10_phase_3_architecture.md`;
- `11_batch_simulation_spec.md`.

---

## 17. Training trajectory record

Training storage is compact, versioned, and reconstructible. The full `127 x 10 x 10` observation tensor and dense 10,000-entry policy vector are **not** stored per decision.

The accepted trajectory schema is documented in `12_trajectory_buffer_spec.md`.

### 17.1 Game-level record

Minimum game metadata includes:

- `game_id`;
- `environment_id`;
- `generation`;
- rules version;
- observation version;
- both true setups in the privileged record;
- first player;
- setup-generator/family identifiers when available;
- terminal result;
- terminal reason;
- final ply;
- collection policy/checkpoint identifiers.

### 17.2 Decision-level record

Each decision stores at least:

- game identifier;
- ply;
- acting player;
- selected action identifier;
- ascending legal action identifiers;
- old/behavior-policy probabilities over exactly those legal actions;
- win/draw/loss prediction;
- collection-policy version;
- snapshot reference.

Probabilities are stored as `float32` in the accepted Phase 3 baseline.

### 17.3 Snapshot cadence

Phase 3 measured snapshot intervals 16, 32, and 64 plies.

Accepted initial default:

- **32 plies**.

Measured tradeoff on the controlled comparison corpus:

| Interval | Raw bytes/game | Compressed bytes/game | Reconstruction positions/s |
|---:|---:|---:|---:|
| 16 | 101,421 | 62,668 | 2,095 |
| **32** | **87,155** | **60,682** | **1,681** |
| 64 | 80,072 | 59,450 | 1,149 |

Interval 32 was selected as the initial storage/reconstruction balance.

### 17.4 Reconstruction contract

A historical decision reconstructs from:

```text
game metadata
+ nearest earlier compact snapshot
+ subsequent actions
-> frozen reference engine
-> exact state
-> observation_v2_1_127ch
-> legal actions/mask
-> privileged belief target
```

Phase 3 reconstructed 1,000,162 historical decisions in the dedicated trajectory gate with zero required-field mismatches, then reconstructed another 11,251 decisions through the integrated pipeline and 411,818 sampled decisions during the two-hour soak with zero mismatches.

Belief targets remain privileged and separate from policy inputs/serialized model-facing data.

### 17.5 Measured storage

With real model-generated policy rows during the Phase 3 soak:

- 187.8 encoded bytes/decision;
- 96,965 encoded bytes/game;
- approximately 5.59 GiB/hour at the measured collecting rate.

The final training system should therefore use a rolling replay/trajectory buffer and selective archival rather than retaining every generated trajectory for the full 168-hour run.

---

## 18. Randomness

Randomness must be explicit and seedable for:

- random legal testing agents;
- random setup generation;
- setup-library sampling;
- any stochastic engine-level process.

The rules themselves contain no random combat results.

---

## 19. Error behavior

The engine and production self-play path must fail loudly for:

- illegal setup;
- illegal move;
- corrupted snapshot;
- impossible piece count;
- inconsistent board occupancy;
- invalid acting player;
- transition after terminal state without reset;
- stale shared-memory publication;
- non-finite action-sampling values;
- sampled action not present in the engine-published legal set;
- hidden-information leakage.

Correctness failures are global-stop errors during development/validation. Ordinary infrastructure failures may later support controlled recovery, but recovery must never silently discard or reinterpret a correctness failure.

### Action-sampling regression from Phase 3

Phase 3 found a rare boundary bug in the representative Gumbel-max sampler. A uniform draw at an endpoint could create non-finite Gumbel noise; combined with an illegal action's `-inf` mask, this could produce `NaN` before `argmax`.

The frozen engine rejected the resulting illegal action before any batch state was mutated.

Required regression policy:

1. random noise used with `-inf` legality masking must be finite;
2. uniform draws used in Gumbel transforms must be bounded away from mathematical singularities;
3. the coordinator must check every selected action against the exact engine legality before workers apply it;
4. an illegal sampled action is a hard correctness failure, not a fallback-to-another-move condition.

The corrected representative sampler is not a frozen game-semantic component; the engine legality contract remains authoritative.

---

## 20. Performance instrumentation and accepted backend decision

Phase 3 measured the production pipeline rather than extrapolating from single-process component rates.

### 20.1 Accepted measurements

Key measured rates:

- standalone multiprocess simulation pipeline at the selected worker/environment configuration: **96,963 positions/s**;
- representative model sustainable inference rate used for the decision denominator: **14,922 positions/s**;
- best 60-second integrated finalist without full trajectory recording: **12,838 positions/s**;
- two-hour production-style collection soak with trajectory recording: **8,871 positions/s**.

Decision ratio:

\[
R =
\frac{\text{sustainable simulation-pipeline positions/second}}
{\text{sustainable representative-model inference positions/second}}
=
\frac{96{,}963}{14{,}922}
=
6.50.
\]

Decision thresholds remain:

- `R >= 2.0`: retain Python as production simulator;
- `1.25 <= R < 2.0`: retain Python initially; optimized backend optional;
- `R < 1.25`: design/evaluate a separate optimized backend.

**Accepted Phase 3 result: `KEEP_PYTHON`.**

A recording-inclusive simulation numerator still gave \(R=4.50\), above the retention threshold.

### 20.2 Bottleneck profile

At the best integrated finalist, approximate step-time shares were:

- Metal inference: 80.87%;
- worker/barrier phase: 10.31%;
- legality + action sampling: 5.60%;
- host-to-device transfer: 3.12%;
- trajectory write-back: 0.07%;
- observation gather: 0.01%.

Workers waited most of the time while the coordinator/model path was active. A faster simulator therefore would not materially improve the current end-to-end system.

### 20.3 Dense versus compact legality

The production transport publishes the engine's dense legality mask.

Although compact legality was cheaper once already constructed on the coordinator/device benchmark, constructing compact legality from the dense shared-memory transport made the integrated compact path approximately **9% slower** than float16+dense in the measured baseline.

Accepted production choice:

- **dense live legality**.

Compact legality remains a validated optional representation only if a future transport publishes it directly or later profiling changes the cost.

### 20.4 Future remeasurement

The Phase 3 model is an untrained representative systems probe. When the final model architecture is selected, re-measure:

- model inference/training throughput;
- worker/environment/batch optimum;
- precision;
- memory;
- ratio \(R\).

The current `KEEP_PYTHON` decision should be revisited only if the real model changes the bottleneck materially.

---

## 21. Browser-interface compatibility

The engine itself does not serve the browser.

A higher-level application service will translate engine/model records into browser-safe data.

The engine must therefore provide serializable representations for:

- public board state;
- private player board state for human play;
- legal actions;
- game history;
- combat events;
- terminal result;
- setup placement and validation.

Training control is handled by the training-control service, not by direct browser calls into engine internals.

---

## 22. Backend replacement contract

Phase 3 selected the frozen Python reference engine as both:

- behavioral source of truth;
- current production simulation backend.

No separate optimized backend should be built for the current system merely because native code could be faster in isolation.

A future optimized backend may be reconsidered only after profiling demonstrates that simulation has become a material end-to-end bottleneck. The preferred trigger remains the measured ratio \(R\) in section 20.

If introduced, backend choice must remain configuration behind the same external interface, and the optimized backend must pass the full differential suite against `phase2_1_reference_1.1.0` before producing training data.

---

## 23. Internal state and public event contracts

The detailed internal-state contract is defined in:

- `08_internal_state_spec.md`

The replay, derived-event, and observer-filtered public-event contracts are defined in:

- `09_public_event_and_replay_schema.md`

These documents are authoritative for:

- stable type-independent piece identifiers;
- piece knowledge flags;
- recent-move state;
- active threat relations;
- behavioral-event memory;
- snapshot completeness;
- privileged replay versus public browser data;
- event ordering and hidden-information filtering.

The engine must store enough compact factual state to reconstruct `observation_v2_1_127ch`; it must not store model-ready observation channels as a second source of truth.
