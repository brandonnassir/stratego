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

- `observation_v2_127ch`

Shape:

\[
127 	imes 10 	imes 10.
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

The reference engine may begin with a small batch wrapper, but the production training interface must support many independent games.

Conceptual operations:

- create `N` games;
- reset selected games;
- get acting players;
- get observations;
- get legal-action masks;
- apply `N` actions or actions for the active subset;
- retrieve rewards/terminal results;
- independently reset finished games;
- retrieve compact trajectory data.

Finished games should be reset independently so parallel games naturally occupy different game phases.

---

## 17. Training trajectory record

A compact trajectory record should contain enough information to reconstruct model training examples without storing every 127 by 10 by 10 observation tensor.

Minimum record:

- rules version;
- observation version;
- red setup;
- blue setup;
- action sequence;
- public combat/reveal events if not trivially reconstructible;
- final result;
- terminal reason;
- model/checkpoint identifiers;
- action probabilities for sampled actions when required by policy optimization;
- value predictions when required for return estimation.

Prefer reconstructing observations from compact game history rather than storing full observations for every move.

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

---

## 20. Performance instrumentation

The engine must expose or make measurable:

- state transitions per second;
- games completed per second;
- average game length;
- observation construction time;
- legal-action generation time;
- batch size;
- memory usage;
- fraction of games ending by each terminal reason.

Optimization decisions will be based on these measurements.

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

A future optimized backend is accepted only if it matches the reference engine on the validation suite.

The model/training layer must not need changes to switch between:

- `reference` backend;
- `optimized` backend.

Backend choice should be configuration, not architecture.

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

The engine must store enough compact factual state to reconstruct `observation_v2_127ch`; it must not store model-ready observation channels as a second source of truth.
