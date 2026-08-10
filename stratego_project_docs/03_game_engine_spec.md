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

The first implementation is a **readable Python reference engine**. After it is validated and profiled, an optimized production backend may be introduced behind the same interface if required by the 168-hour training budget.

### Frozen reference implementation

Phase 2.1 accepted and froze:

- implementation: `phase2_1_reference_1.1.0`;
- rules: `stratego_project_v1`;
- observation: `observation_v2_1_127ch`;
- action encoding: `action_id = 100 * source + destination`, 10,000 entries;
- deterministic replay semantics defined by the current replay/event contracts.

Phase 3 validated this engine inside the complete multiprocessing, shared-memory, trajectory-recording, and Metal-inference pipeline. The production-backend decision is **KEEP_PYTHON**. Phase 4 then validated the same frozen engine under the permanent evaluation harness: reproducible paired matches, observer-safe policies, raw replay/result records, and large hidden-information differential audits. The frozen Python engine remains both the behavioral oracle and the production simulator for the current model scale. A separate optimized backend is deferred unless future model changes materially alter the measured throughput ratio.

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

## 16. Batch simulation interface

Phase 3 implemented and validated the production batch interface.

### Production architecture

```text
CPU simulation workers
        |
persistent shared-memory buffers
        |
single coordinator process
        |
PyTorch model on Apple Metal
```

The system is bulk-synchronous:

1. workers publish acting-player observations, legality, and slot metadata;
2. the coordinator waits for the worker phase;
3. the coordinator performs model inference over all active rows, possibly in inference-batch chunks;
4. the coordinator applies legality and samples one legal action per row;
5. actions and model outputs needed for trajectory recording are written back to shared memory;
6. workers record the decision, apply the action, seal terminal games, and independently reset finished slots;
7. the next global step begins.

Only the coordinator may own or invoke the Metal-backed model.

### Environment identity

Every persistent slot has:

- fixed `environment_id`;
- monotonically increasing `generation`.

The pair `(environment_id, generation)` uniquely identifies one game instance. Worker simulators own disjoint global environment-identifier ranges.

### Shared-memory contract

Worker-published model-facing fields include at minimum:

```text
observations      [N, 127, 10, 10] float32
legal_mask        [N, 10000]       uint8
legal_count       [N]              int32
acting_player     [N]              int8
environment_id    [N]              int32
generation        [N]              int32
status/terminal/counters
```

The coordinator writes:

```text
actions                    [N]          int32
reset_request              [N]          uint8
policy_probabilities       [N, 128]     float32
value_prediction           [N, 3]       float32
decision_valid             [N]          uint8
```

`policy_probabilities` aligns with the engine's deterministic ascending legal-action order and is zero-padded. It exists for in-worker trajectory recording and does not replace the authoritative dense legality mask.

The coordinator holds no game object and no privileged game state is transported to the model process.

### Measured Phase 3 starting point

With the representative compact Transformer probe, the strongest sustained integrated configuration was:

- 10 simulation workers;
- 1,536 simultaneous environments;
- inference batch 1,536;
- float16 inference;
- dense legality;
- snapshot interval 32.

Measured throughput:

- **12,838 positions/second** without full trajectory recording;
- **8,871 positions/second** during the two-hour collecting soak.

These are measured starting values, not frozen training constants. Re-benchmark when the actual model architecture changes.

### Legality representation

Dense legality is the accepted production default.

Compact legality was proven equivalent, but in the integrated pipeline it was slower because the shared transport naturally publishes the dense mask and the coordinator then had to build the compact form.

Measured comparison:

- float16 dense: about 11,592 positions/second;
- float16 compact: about 10,548 positions/second.

Compact legality may be revisited only if a future transport publishes compact legal actions directly or another architectural change removes the conversion cost.

---

## 17. Training trajectory record

Phase 3 implemented `trajectory_v1`, a compact versioned trajectory schema.

The system does not store the full 127 by 10 by 10 observation tensor or a dense 10,000-entry policy distribution for every decision.

### Game record

Minimum game-level fields include:

- `game_id`;
- `environment_id`;
- `generation`;
- rules and observation versions;
- red and blue setups;
- first player;
- setup-family/setup identifiers when available;
- ordered action sequence;
- periodic compact snapshots;
- terminal result and reason;
- final ply;
- collection checkpoint/policy metadata.

### Decision record

Each decision stores:

- game identifier;
- ply;
- acting player;
- selected action identifier;
- legal action identifiers in deterministic ascending order;
- behavior-policy probability for each legal action;
- win/draw/loss value prediction;
- collection-policy version;
- snapshot reference.

Policy probabilities are currently stored as `float32`.

### Snapshot cadence and reconstruction

Supported intervals are 16, 32, and 64 plies. Phase 3 selected **32** as the default compromise.

At interval 32, measured random-access reconstruction was approximately **1,681 positions/second per process**.

Historical positions reconstruct from:

```text
game metadata
+ nearest prior snapshot
+ subsequent actions
-> frozen reference engine
-> observation_v2_1_127ch
-> legal actions/mask
-> public knowledge
-> privileged belief target
```

Phase 3 reconstructed **1,000,162 historical decisions** in the dedicated trajectory gate with zero unexplained mismatches, then reconstructed additional model-generated decisions during integration and soak testing with zero mismatches.

The belief target remains a privileged training target and is not serialized as model input.

### Storage measurements

At 32-ply snapshots, Agent 3 measured approximately:

- 93 KB raw/game;
- 65 KB compressed/game;
- 64.7 GB compressed per million games on its synthetic-policy corpus.

The integrated model-backed soak measured approximately:

- 187.8 bytes/decision;
- 96,965 bytes/game;
- **5.59 GiB/hour** encoded trajectory at the accepted collection rate.

The production training system must therefore use a rolling/managed replay-storage policy rather than assuming every generated trajectory can be retained throughout the full 168-hour run.

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

The engine should fail loudly during development for:

- illegal setup;
- illegal move;
- corrupted snapshot;
- impossible piece count;
- inconsistent board occupancy;
- invalid acting player;
- transition after terminal state without reset.

The training coordinator may later choose a controlled error-recovery policy, but the reference engine must prioritize detectability over silent recovery.

Phase 3 development/production policy:

- any invariant failure, observation mismatch, illegal transition, or hidden-information leak is a global-stop error and must preserve a reproducible crash package;
- ordinary infrastructure failure such as a worker process exiting may be recoverable by restarting that worker from known coordinator state;
- infrastructure recovery must never convert a correctness failure into a silently discarded game;
- the coordinator must independently verify that every sampled model action is legal before workers receive it.

Phase 3 discovered a rare Gumbel-max sampling boundary bug in the representative model probe. Non-finite sampling noise combined with `-inf` legality masking could produce `NaN` and lead `argmax` to select an illegal action. The frozen engine rejected the action before mutation. The sampler was corrected and regression tests now require finite noise and legal sampled actions.

This is a permanent argument for preserving engine-side illegal-action rejection even when coordinator-side legality checks also exist.

---

## 20. Performance instrumentation and Phase 3 decision

Phase 3 measured the complete production path.

### Simulation pipeline

Measured no-model CPU/shared-memory simulation rate at 10 workers and 1,536 environments:

- **96,963 positions/second**.

With full trajectory recording enabled, the simulation side still measured:

- **67,209 positions/second**.

### Representative Metal model

Sustainable uncontended representative-model inference used for the production ratio:

- **14,922 positions/second**.

### Decision ratio

\[
R =
\frac{96{,}963}{14{,}922}
=
6.50.
\]

Using the pre-registered decision rule:

- `R >= 2.0`: retain Python;
- `1.25 <= R < 2.0`: retain Python initially;
- `R < 1.25`: optimized backend required.

Phase 3 result:

```text
backend decision = KEEP_PYTHON
R = 6.50
optimized Agent 6 = NOT REQUIRED
```

The recording-inclusive ratio remains approximately 4.50.

The integrated profile independently agrees:

- Metal inference ≈ 80.9% of the best step;
- workers active ≈ 9.7% of wall time;
- workers waiting ≈ 90.4%;
- coordinator active ≈ 89.7%.

The current pipeline is model-bound, not simulation-bound.

### Two-hour production soak

Accepted soak result:

- duration ≈ 7,200 seconds;
- 63,871,488 positions;
- 123,718 games;
- 123,718 independent resets;
- 8,871 positions/second with trajectory recording;
- 411,818 reconstructed decisions checked;
- zero reconstruction mismatches;
- zero worker errors/restarts;
- zero swap use;
- zero measured coordinator-memory growth;
- about -0.76% first-vs-last-quarter throughput drift.

This passed the Phase 3 production-training readiness gate.

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

Phase 3 decision: **no optimized backend is required for the current system**.

The frozen Python reference engine is both the behavioral oracle and the production simulator for the current model scale.

A future optimized backend remains an architectural option only if later profiling changes the bottleneck. If introduced, it must match the frozen reference engine on the full differential validation suite, and the model/training layer must not need semantic changes to switch between backends.

Backend choice remains configuration, not architecture.

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


## 24. Evaluation-harness compatibility

Phase 4 introduced an evaluation layer above the frozen engine. It does not alter engine semantics.

Accepted versions:

```text
policy interface: policy_interface_v1
match specification: match_spec_v1
evaluation setup bank: evaluation_setup_bank_v1
match runner: match_runner_v1
match result: match_result_v1
evaluation scheduler: evaluation_scheduler_v1
evaluation statistics: evaluation_statistics_v1
evaluation reporting: evaluation_reporting_v1
calibration: phase4_calibration_v1
```

### Observer-safe policy boundary

A policy must not receive a `GameState`, privileged replay, true unresolved opponent identities, belief target, or any object from which those values can be recovered.

Policy inputs are materialized observer-safe products only. Policies may request the products they require, including:

- legal action identifiers;
- `observation_v2_1_127ch`;
- legal-action mask;
- observer-safe `PublicView`;
- match and policy seed metadata.

Phase 4's ten catalogued baseline/stress policies use only `PublicView` plus legal actions. A checkpoint-shaped probe additionally consumed the 127-channel observation and legal-action mask through the same contract with zero illegal actions and zero reproduction mismatches.

### Evaluation setup bank and pairing

The accepted fixed evaluation-only bank is:

- `evaluation_setup_bank_v1`;
- generation family `structured_v1`;
- 1,024 deterministic setup pairs.

It is a reproducible evaluation instrument, **not** the Phase 7 training setup generator.

The accepted pairing mode is `color_swap_same_board`:

```text
Game A: candidate plays Red on fixed red/blue setups
Game B: candidate plays Blue on the same physical board setups
```

Each policy therefore receives one first-move game and one second-move game on the same board pair.

### Policy versions and calibration

Policy identity is versioned and included in match identity. A heuristic or weight change requires a policy-version bump before new results are considered comparable.

The calibrated core ladder after Phase 4 is:

1. `strategic_rule_based@1.1.0`;
2. `tactical_rule_based@1.0.0`;
3. `basic_heuristic@1.0.0`;
4. `random_legal@1.0.0`.

Strategic version 1.1.0 corrected the exposure heuristic to price publicly inferred vulnerability rather than raw material value. This was a policy implementation correction, not an engine/rule change.

### Engine authority during evaluation

The evaluation runner must still:

1. request legality from the frozen engine;
2. validate policy results;
3. submit the selected action to the frozen engine;
4. allow the engine to reject any illegal action atomically;
5. preserve the engine's terminal result and terminal reason without reinterpretation.

The evaluation layer must never replace an illegal or failed policy decision with a fallback legal move.
