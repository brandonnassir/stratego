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

### Frozen reference status

Phase 2.1 froze these replay/event semantics in `phase2_1_reference_1.1.0`. Ten thousand complete games covering 5,078,406 plies replayed with zero state, observation, event, or terminal-result mismatches, and 103,625 valid hidden-state permutations produced zero public-event or browser-view mismatches.

Phase 3 may add training-trajectory metadata, periodic snapshots, sparse legal-action probabilities, and shared-memory transport records. Those additions are training/infrastructure products and must not redefine the authoritative replay or public-event semantics in this document.

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

## 14. Training metadata record

The reinforcement-learning system may attach metadata to each sampled decision without altering the game rules or replay semantics.

Possible fields:

- policy checkpoint identifier;
- opponent checkpoint identifier;
- setup family and setup identifier;
- action selected;
- behavior-policy action probability;
- win/draw/loss value prediction;
- belief-head output summary or loss target reference;
- training iteration;
- random-number generator state/seed where needed.

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

For the 168-hour training run, prefer storing:

- compact setups;
- action sequences;
- terminal results;
- selected training metadata;
- periodic checkpoints and evaluation summaries.

Do not store the full 127 by 10 by 10 observation tensor for every move unless profiling demonstrates that reconstruction is more expensive than the storage cost.

The 1-terabyte external solid-state drive should hold long-run trajectories, checkpoints, and evaluation artifacts. The 150-gigabyte internal disk should retain the active run, current checkpoints, logs, and sufficient restart data.

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
