# Public Event and Replay Schema

## 1. Purpose

This document defines how the engine records a game for:

- deterministic replay;
- debugging;
- browser visualization;
- model-training reconstruction;
- public/private observer views;
- validation of hidden-information safety.

The key design rule is that **privileged replay data and public event data are different products**.

A replay used internally may contain both secret setups. A public event stream shown to a player or browser client must never expose information that player is not allowed to know.

### Frozen reference and Phase 3 trajectory status

Phase 2.1 froze replay/event semantics in `phase2_1_reference_1.1.0`. Ten thousand complete games covering 5,078,406 plies replayed with zero state, observation, event, or terminal-result mismatches; 103,625 valid hidden-state permutations produced zero public-event/browser-view mismatches.

Phase 3 added a separate compact training-trajectory product without redefining the authoritative replay/public-event semantics.

Accepted Phase 3 trajectory/reconstruction evidence:

- schema: `trajectory_v1`, wire-format magic `STJ1`;
- 1,000,162 historical decisions reconstructed in the dedicated trajectory gate with zero required-field mismatches;
- 11,251 integrated pipeline decision reconstructions, zero mismatches;
- 411,818 sampled reconstruction checks during the two-hour soak, zero mismatches.

Training trajectory metadata, model outputs, and snapshots remain infrastructure/training data, not game-state authority.

---

## 2. Three record layers

```text
Privileged Replay Record
  - both true setups
  - action sequence
  - optional training metadata
          |
          +--> exact deterministic reconstruction
          |
          v
Derived Engine Events
  - move/combat/reveal/behavior/game-end facts
          |
          v
Observer-filtered Public Event Stream
  - only information legal for that observer
```

No browser or policy endpoint should receive the privileged replay record during an active game.

---

## 3. Privileged replay record

The minimal authoritative replay record contains:

| Field | Meaning |
|---|---|
| `replay_version` | Schema version |
| `rules_version` | Rules/configuration version |
| `observation_version` | Observation contract version |
| `game_id` | Stable game identifier |
| `red_setup` | Complete true red setup |
| `blue_setup` | Complete true blue setup |
| `first_player` | Normally red |
| `actions` | Ordered action identifiers or source/destination pairs |
| `terminal_result` | Win/loss/draw |
| `terminal_reason` | Explicit terminal reason |

If setup generation or opponent selection is stochastic, training metadata may additionally record:

- setup-generator version;
- setup-generator seed or selected setup identifier;
- policy/checkpoint identifiers;
- action sampling seeds, if exact stochastic reproduction is required;
- sampled action probability;
- value prediction;
- training iteration.

The rules of Stratego contain no random combat outcome, so the setups and action sequence are sufficient to reconstruct the game-state sequence.

---

## 4. Stable public piece identity

Public event records may use `piece_id` because a human can track a physical concealed piece as it moves.

The identifier must be type-independent, as defined in `08_internal_state_spec.md`.

A public piece identifier tells the observer **which physical piece** moved. It must never tell the observer **what type the piece is** unless that type has been legally revealed.

---

## 5. Move event

A move event records publicly observable movement.

Conceptual fields:

| Field | Meaning |
|---|---|
| `event_type` | `move` |
| `ply` | Ply index |
| `player` | Moving player |
| `piece_id` | Stable physical piece identifier |
| `source` | Source square |
| `destination` | Destination square |
| `distance` | Cardinal movement distance |
| `is_attack` | Whether destination held an opponent piece |
| `target_piece_id` | Target identifier when attacking; otherwise null |

The event does not include `true_type` unless an observer-filtered view is legally allowed to display it.

A multi-square move is itself public evidence that the moving piece is a Scout. The engine should therefore also generate an identity-reveal event when this movement newly reveals the Scout to the opponent.

---

## 6. Identity-reveal event

A reveal event represents a new legal disclosure of exact piece type.

Conceptual fields:

| Field | Meaning |
|---|---|
| `event_type` | `identity_reveal` |
| `ply` | Ply on which knowledge changed |
| `piece_id` | Revealed piece |
| `owner` | Piece owner |
| `piece_type` | Newly public type |
| `reason` | `combat` or `scout_multisquare` |
| `newly_known_to` | Which observer(s) gained the information |

A reveal event is emitted only when knowledge actually changes. Repeated combat involving an already known piece does not need a duplicate knowledge-change event, though combat still gets its own event.

---

## 7. Combat event

Every attack produces one combat event after resolution.

Conceptual fields:

| Field | Meaning |
|---|---|
| `event_type` | `combat` |
| `ply` | Combat ply |
| `attacker_piece_id` | Attacking piece |
| `defender_piece_id` | Defending piece |
| `attacker_type` | Public after combat |
| `defender_type` | Public after combat |
| `outcome` | `attacker_survives`, `defender_survives`, or `both_removed` |
| `flag_captured` | Boolean |

The combat event is public to both players because combat reveals both identities.

Separate capture events are optional and not authoritative; capture is already fully represented by combat outcome. Avoid storing the same fact in multiple authoritative forms.

---

## 8. Behavioral event record

Behavioral events are deterministic derived events used for observation reconstruction and debugging.

They contain no hidden type fields.

Conceptual fields:

| Field | Meaning |
|---|---|
| `event_type` | `behavior` |
| `behavior_type` | threat / evade / declined_attack / protect / was_protected |
| `ply` | Event ply |
| `actor_piece_id` | Piece receiving behavior memory |
| `counterpart_piece_id` | Selected counterpart |
| `actor_knew_counterpart_type` | Historical public-knowledge fact |
| `context_piece_id` | Optional extra context, such as threatener in protection |

The current 127-channel observation later resolves counterpart rank/special encoding using the **current observer's legal knowledge**.

### Why type is excluded

Suppose the counterpart was secretly a Marshal at event time. The full engine knows that fact, but the event record must not expose it merely because the privileged simulator knows it.

---

## 9. Threat-relation event/context

The engine may record all threat relations created by a move for debugging/reconstruction:

```text
threatener_piece_id
threatened_piece_id
creation_ply
```

These relations are public geometric facts and contain no hidden types.

They are primarily immediate state context for `Evade` and `Protect`, rather than long-term model events.

---

## 10. Game-end event

A terminal event contains:

| Field | Meaning |
|---|---|
| `event_type` | `game_end` |
| `ply` | Final ply |
| `winner` | Red, blue, or null for draw |
| `result` | Red win / blue win / draw |
| `terminal_reason` | Explicit project terminal reason |
| `total_moves` | Final move count |
| `moves_since_last_combat` | Final battleless count |

This event is public.

---

## 11. Setup visibility

The privileged replay contains both true setups.

An observer-specific browser/game-start view must instead behave as follows:

### Player's own setup

The observer may receive exact piece identities on all 40 own setup squares.

### Opponent setup

The observer receives:

- opponent piece occupancy on all 40 opponent setup squares;
- stable physical piece identifiers/origin tracking if the interface uses them;
- **no opponent true types** before legal revelation.

There is no public event at game start that exposes the opponent setup identities.

---

## 12. Observer-filtered public state

For any observer, the engine/application layer must be able to construct a serializable current board view containing only:

- square geometry;
- piece owner;
- stable piece identifier;
- exact type for the observer's own pieces;
- exact type for opponent pieces whose identities are legally known;
- hidden marker for unresolved opponent identities;
- moved status where publicly inferable from movement history;
- acting player;
- move counters;
- terminal status.

This view is appropriate for the browser. It is **not** the neural-network observation tensor.

---

## 13. Deterministic reconstruction rule

For a fixed:

- rules version;
- red setup;
- blue setup;
- first player;
- ordered action sequence;

reconstruction must produce the same:

- board after every ply;
- piece knowledge flags;
- reveal events;
- combat events;
- active threat relations;
- latest behavioral events;
- recent-move window;
- 127-channel observations;
- legal-action masks;
- terminal result.

Derived event logs must therefore be reproducible rather than treated as independent sources of truth.

---

## 14. Training metadata and trajectory record

The reinforcement-learning system may attach metadata to each sampled decision without altering game rules or replay semantics.

Accepted Phase 3 decision metadata includes:

- game identifier;
- environment identifier and generation;
- ply;
- acting player;
- policy/checkpoint identifier;
- collection-policy version;
- selected action;
- ascending legal action identifiers;
- behavior-policy probabilities aligned one-for-one with those legal actions;
- win/draw/loss value prediction;
- snapshot reference;
- setup family/setup identifier when available;
- training iteration/collection block when available.

Observed Phase 3 policy-version identifiers include:

- `synthetic_hash_policy_v1` for the deterministic Agent 3 storage/reconstruction probe;
- `end_to_end_representative_probe_v1` for the integrated representative-model collection path.

These identifiers document the collection policy that produced a record; they do not define game rules or the final model architecture.

### Model decision transport

In the accepted Phase 3 production path:

- the coordinator owns the model;
- the worker owns the game state and exact ascending legal-action list;
- coordinator writes selected action, legal-prefix probabilities, win/draw/loss prediction, and `decision_valid` into shared memory;
- worker records the decision before applying the action.

The model-facing shared transport never contains privileged belief targets or true hidden opponent identities.

### Belief targets

Belief targets are reconstructed separately from privileged state when training data is consumed. They are not serialized as model-facing trajectory input fields.

These fields are training data, not engine state.

---

## 15. Browser event stream

The browser-facing service should consume observer-filtered events rather than poll privileged engine objects directly.

Recommended browser event categories:

- game started;
- setup accepted;
- move played;
- piece identity revealed;
- combat resolved;
- game ended;
- training/evaluation metadata events from higher-level services.

The browser may request a full observer-filtered board snapshot at any time to recover from a dropped event connection.

Training control belongs to the training-control service, not the game-event stream.

---

## 16. Hidden-information anti-leak requirements

For two privileged games that differ only by a permutation of true types among unresolved opponent pieces while keeping public history identical:

### Must remain identical for the observer

- public board view;
- move events;
- behavior events and threat relations;
- legal action set, unless a public rule consequence legitimately differs;
- all `observation_v2_1_127ch` channels;
- opponent setup view;
- browser event stream.

### May differ

- privileged replay setup;
- privileged piece records;
- belief-learning targets.

When the different hidden identity later becomes legally revealed, public streams may then diverge.

---

## 17. Event ordering within a ply

For deterministic browser and replay behavior, events generated by one action use the following public ordering:

1. `move`;
2. `identity_reveal` events caused by the move/combat, ordered by stable piece identifier if more than one occurs;
3. `combat`, if applicable;
4. derived `behavior` events, ordered by behavior type then actor piece identifier;
5. `game_end`, if terminal.

This ordering is a project convention. It does not change game rules.

---

## 18. Versioning

Every serialized replay/event document must include schema versions.

Minimum version fields:

- `rules_version`;
- `replay_version`;
- `event_schema_version`;
- `observation_version` where observations may be reconstructed.

Changing channel semantics, action encoding, or event meaning requires a version change rather than silently changing old records.

---

## 19. Storage policy

For the 168-hour training run, store compact trajectories rather than full observations.

Accepted Phase 3 baseline:

- snapshot interval: 32 plies;
- sparse legal-action identifiers;
- `float32` old-policy probability per legal action;
- win/draw/loss prediction;
- compressed game record;
- exact reconstruction through the frozen reference engine.

Measured with real representative-model policy rows during the two-hour soak:

- 187.8 encoded bytes/decision;
- 96,965 encoded bytes/game;
- 11.17 GiB generated in two hours;
- approximately **5.59 GiB/hour**.

At the same rate for 168 hours, permanent retention of every trajectory would be approximately:

\[
5.59 \times 168 \approx 939\ \text{GiB}.
\]

That is too close to the project's 1-terabyte archive capacity once checkpoints, evaluations, logs, filesystem overhead, and other artifacts are included.

Therefore:

- use a rolling training/replay buffer for bulk self-play data;
- expire/delete bulk trajectories after they are no longer required by the training algorithm;
- retain selected representative training games;
- retain diagnostic/error/unusual games;
- retain evaluation games and manifests;
- retain checkpoints, run summaries, and enough data for reproducibility/debugging.

The exact buffer size is a later training-system decision; the principle that full-run bulk trajectories are not all permanently archived is now fixed.

---

## 20. Acceptance criteria

This schema is accepted when:

1. a complete game can be reconstructed from the privileged replay record alone;
2. public event streams are identical under hidden-type permutations until legal revelation;
3. browser views never require direct access to privileged types;
4. behavior events reconstruct the behavior channels exactly;
5. event ordering is deterministic;
6. replay/event schema versions are explicit;
7. training metadata remains separate from game-state authority.
