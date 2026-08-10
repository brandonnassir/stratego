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

Phase 2.1 froze these replay/event semantics in `phase2_1_reference_1.1.0`.

Phase 3 added training-trajectory infrastructure without changing authoritative replay or public-event meaning:

- `trajectory_v1` stores compact snapshots, actions, sparse legal probabilities, and value predictions;
- belief labels remain reconstructed privileged targets rather than serialized model inputs;
- model-generated trajectory records reconstruct exactly through the frozen reference engine;
- 11,251 integrated stored decisions and 411,818 soak-time decisions reconstructed with zero mismatches.

Phase 3 collection-policy identifiers used during validation include:

- `synthetic_hash_policy_v1`;
- `end_to_end_representative_probe_v1`.

These identifiers describe data-generating policies, not game rules.

Phase 4 added a separate evaluation-result layer above the engine replay schema. It does not change authoritative replay/event semantics.

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
- collection-policy version;
- setup family and setup identifier;
- action selected;
- legal action identifiers in deterministic ascending order;
- behavior-policy probability for each legal action;
- win/draw/loss value prediction;
- belief-head output summary or loss target reference;
- training iteration;
- random-number generator state/seed where needed.

The integrated Phase 3 transport writes policy probabilities back to workers in the same deterministic ascending legal-action order used by the engine. This lets workers create compact trajectory records without transmitting privileged game objects to the coordinator.

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
- sparse legal-action probability rows;
- win/draw/loss predictions;
- periodic compact snapshots;
- terminal results;
- selected training metadata;
- checkpoints and evaluation summaries.

Do not store the full 127 by 10 by 10 observation tensor for every move.

Phase 3 measured model-backed trajectory encoding at approximately:

- 187.8 bytes per decision;
- 96,965 bytes per game;
- 5.59 GiB/hour at 8,871 collected positions/second.

Retaining every generated trajectory for all 168 hours would approach the capacity of the 1-terabyte external drive. The production training system must therefore use a rolling/managed trajectory buffer and selectively archive important and diagnostic material. Evaluation games are far smaller than training trajectories; accepted final evaluation leagues should normally retain their complete action histories when practical. The project preference is to preserve most games on the external drive when storage permits, while avoiding an unbounded hot training buffer.

---

## 20A. Phase 4 evaluation records

Phase 4 introduced versioned evaluation records that reference, but do not replace, the engine replay contract.

Accepted versions:

```text
policy_interface_v1
match_spec_v1
match_runner_v1
match_result_v1
evaluation_scheduler_v1
evaluation_statistics_v1
evaluation_reporting_v1
phase4_calibration_v1
```

### Match identity

`MatchSpec` deterministically binds at least:

- evaluation suite version;
- policy identifiers and versions;
- setup-bank version and setup-pair identifier;
- candidate color;
- replicate;
- root seed;
- complete rules configuration.

Worker number, schedule position, process identity, and wall-clock time must not affect match identity.

### Pair identity

The accepted paired mode is `color_swap_same_board`. The two games share a `paired_unit_id` and use the same physical red and blue setups, while the candidate/opponent policy assignment swaps colors.

The paired unit is the statistical resampling unit.

### MatchResult

A raw match result includes enough information to diagnose and reproduce the match, including:

- `match_id` and `paired_unit_id`;
- policy identifiers/versions;
- candidate color;
- setup-pair identifier;
- both resolved setup strings;
- rules payload/token;
- policy seeds;
- result/winner/draw;
- terminal reason;
- ply count;
- replay digest;
- timing and policy-error fields.

The engine replay remains the authority for the action sequence and final state.

### Replay sidecar

Phase 4 supports compact result rows plus a replay JSONL sidecar keyed by `match_id`. Agent 3 measured this split as substantially smaller and easier to summarize than embedding action histories in every result row.

The accepted final 44,544-game Phase 4 calibration league retained full raw rows and replay digests but did **not** archive all action histories. Because the estimated history archive is only on the order of 100 MB, the preferred preservation policy is to regenerate and retain the final calibration histories when convenient. This is a preservation improvement, not a Phase 4 correctness blocker.

For future accepted/citable evaluation leagues, retain complete action histories when practical.

### Evaluation setup bank

`evaluation_setup_bank_v1` contains 1,024 deterministic `structured_v1` setup pairs. It is an evaluation artifact only and must not be treated as the Phase 7 training setup generator.

## 20. Acceptance criteria

This schema is accepted when:

1. a complete game can be reconstructed from the privileged replay record alone;
2. public event streams are identical under hidden-type permutations until legal revelation;
3. browser views never require direct access to privileged types;
4. behavior events reconstruct the behavior channels exactly;
5. event ordering is deterministic;
6. replay/event schema versions are explicit;
7. training metadata remains separate from game-state authority.
