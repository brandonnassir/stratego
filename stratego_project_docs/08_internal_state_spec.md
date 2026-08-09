# Internal Game State Specification

## 1. Purpose

This document defines the minimum authoritative state that the Stratego engine must retain so that it can:

- enforce the project's rules;
- reconstruct `observation_v2_127ch` exactly;
- generate legal-action masks;
- support deterministic replay;
- support snapshot/restore for decision-time search;
- produce public browser events without leaking hidden information;
- provide privileged ground-truth targets for belief learning.

The engine must store **facts and event records**, not model-ready channel values. The 127-channel observation is derived from this state.

---

## 2. State layers

The engine separates information into four conceptual layers:

```text
Privileged full game state
        |
        +--> rules and state transitions
        +--> belief-learning ground truth
        +--> deterministic replay/debugging
        |
        v
Public knowledge + public history
        |
        v
Observer-specific player view
        |
        v
observation_v2_127ch + legal-action mask
```

The policy and value model must never receive the privileged full game state directly.

---

## 3. Rules configuration

Every game state must reference an immutable rules configuration containing at least:

- `rules_version`;
- board geometry version;
- first player;
- battleless-move draw limit;
- absolute-move safety limit;
- two-square rule enabled: **false**;
- continuous-chasing rule enabled: **false**.

Project defaults:

| Context | Battleless limit | Absolute move limit |
|---|---:|---:|
| Training | 100 plies | 4,000 plies |
| Evaluation / human play | 200 plies | configurable; normally 4,000 as an engineering safety limit |

A rules configuration must not change during an active game.

---

## 4. Canonical piece identifier

Every one of the 80 physical pieces receives a stable identifier at game creation.

The identifier must **not encode piece type**.

Recommended conceptual form:

```text
(owner, setup_slot_index)
```

where `setup_slot_index` is the row-major index `0..39` of the piece's original setup square.

Examples:

- `red:07`
- `blue:31`

This has three benefits:

1. a physical hidden piece can be followed as it moves;
2. the identifier does not leak its true type;
3. origin tracking is deterministic and requires no random identifier generation.

The model does not receive these identifiers as numeric features. They are engine/replay bookkeeping identifiers.

---

## 5. Piece record

The privileged state contains one record for each of the 80 physical pieces.

Minimum fields:

| Field | Meaning |
|---|---|
| `piece_id` | Stable type-independent identifier |
| `owner` | Red or blue |
| `true_type` | Privileged true type |
| `starting_square` | Immutable original square |
| `current_square` | Current square, or null if captured |
| `alive` | Whether piece remains on board |
| `has_moved` | Whether piece has ever legally moved |
| `known_to_red` | Whether red may legally know exact identity |
| `known_to_blue` | Whether blue may legally know exact identity |
| `reveal_reason_red` | Optional debugging metadata for how red learned the type |
| `reveal_reason_blue` | Optional debugging metadata for how blue learned the type |
| `capture_ply` | Null while alive; ply of capture otherwise |

### Knowledge invariants

- A player always knows the exact identities of their own pieces.
- Knowledge of an opponent identity is monotonic: once known, it cannot become hidden again.
- Combat makes the identities of both combatants public.
- A legal multi-square Scout move makes that moving piece identifiable as a Scout to the opponent.
- Capturing a piece does not erase previously acquired identity knowledge.

`reveal_reason_*` is debugging metadata and is not required by the model observation.

---

## 6. Board occupancy

The privileged state contains a fixed 100-cell board representation.

Each cell is exactly one of:

- lake;
- empty;
- occupied by one `piece_id`.

The board and piece table must agree bidirectionally:

- every live piece has exactly one current square;
- that square points back to the same piece identifier;
- captured pieces appear on no board square;
- lake squares never contain pieces.

Board occupancy must be treated as authoritative only together with these consistency invariants.

---

## 7. Game-control state

The state must retain:

- current phase: setup, play, or terminal;
- acting player;
- current ply index;
- total moves played;
- consecutive moves since last combat;
- terminal flag;
- terminal reason;
- winner, if any;
- draw flag, if any.

Required terminal reasons remain:

- `flag_capture`;
- `opponent_no_legal_move`;
- `both_no_legal_move_draw`;
- `battleless_move_limit_draw`;
- `absolute_move_limit_draw`;
- `not_terminal`.

---

## 8. Recent move history

The current observation requires the last 16 plies.

The engine state therefore retains a ring/deque containing at least the 16 most recent move records.

Each recent move record contains at minimum:

- ply index;
- acting player;
- moving piece identifier;
- source square;
- destination square;
- whether the destination was occupied by an opponent at move start;
- attacked piece identifier, if any.

The complete action sequence belongs to the replay record described in `09_public_event_and_replay_schema.md`; the core live state only needs the history window required for observation construction and immediate behavioral logic.

---

## 9. Immediate threat relations

Because `Evade` and `Protect` depend on the immediately preceding opponent move, the engine retains the complete set of **active threat relations** produced by the previous move.

Each relation is:

```text
(threatener_piece_id, threatened_piece_id, creation_ply)
```

Important distinction:

- the **behavioral Threat event** records at most one selected counterpart for the moving piece;
- `active_threat_relations` stores **all** opponent pieces newly threatened by that move.

This is necessary because several pieces can become adjacent to the moved piece, and any of them may evade or receive protection on the next turn.

The active relation set expires after the threatened player's response move is resolved and is replaced by threat relations created by that response.

---

## 10. Behavioral memory

For each physical piece, the engine retains the most recent event of each of five behavior types:

1. `threat`;
2. `evade`;
3. `declined_attack`;
4. `protect`;
5. `was_protected`.

The authoritative record is event metadata, not the four channel values.

Minimum behavioral event fields:

| Field | Meaning |
|---|---|
| `event_type` | One of the five behavior types |
| `actor_piece_id` | Piece receiving the behavior record |
| `counterpart_piece_id` | Piece used for the behavior context |
| `event_ply` | Ply on which event occurred |
| `actor_knew_counterpart_type` | Whether actor knew counterpart identity at event time |
| `context_piece_id` | Optional additional piece needed for debugging/future representations |

`context_piece_id` is particularly useful for protection events: the current 127-channel observation uses the protected/protector piece as counterpart, while the engine may also retain the opponent threatener for later analysis without expanding the current observation.

### Behavioral channel reconstruction

At observation time:

- recency is computed from `event_ply`;
- counterpart rank/special type is computed only if legal for the current observer;
- the historical actor-knew flag comes directly from the event;
- the values are written at the actor piece's current square if the actor is still alive.

---

## 11. Turn-start attack opportunities

`Declined Attack` depends on which adjacent attacks were available before the selected action.

The engine does not need to store these opportunities permanently. For each turn it must deterministically derive, before applying the action:

```text
piece_id -> ordered list of legally attackable adjacent opponent piece_ids
```

After the selected action is known, declined-attack events are generated according to `06_observation_v2_127ch.md`.

If several eligible declined targets remain for one piece, select the target occupying the lowest absolute board-square index at turn start.

No target ordering may inspect hidden piece type.

---

## 12. Setup information

The privileged state does not need separate copied setup tensors.

The original setups can be reconstructed from each piece's:

- owner;
- `true_type`;
- `starting_square`.

Observer-specific setup memory is derived as follows:

- own setup planes: use all own piece types and starting squares;
- opponent known-setup planes: include only pieces whose identity is legally known to that observer.

This avoids duplicate state while still reconstructing channels 32-55 exactly.

---

## 13. Unresolved inventory

The engine does not store unresolved-inventory channel values.

For observer `P` and opponent type `T`, derive:

\[
U_T=N_T-K_T,
\]

where `K_T` is the number of opponent physical pieces whose exact type is legally known to `P`, alive or captured.

The calculation must use the observer's knowledge flags, never privileged hidden types of unresolved pieces.

---

## 14. State transition order

For deterministic behavior, applying a legal action follows one conceptual order:

1. Validate game is nonterminal.
2. Capture turn-start legal actions and adjacent attack opportunities.
3. Preserve the previous move's active threat relations for evade/protect evaluation.
4. Validate the selected action.
5. Move/resolve combat atomically.
6. Update identity knowledge caused by combat or Scout movement.
7. Update alive/current-square/moved fields.
8. Update total-move and battleless counters.
9. Generate `evade`, `protect`, `was_protected`, and `declined_attack` behavioral events using the pre-move context.
10. Compute all new threat relations caused by the resolved move.
11. Record the selected `threat` behavioral event using the deterministic counterpart rule.
12. Append recent-move and public-event records.
13. Evaluate terminal conditions.
14. If nonterminal, change acting player.

The implementation may optimize this order internally, but externally observable results must be equivalent to it.

---

## 15. Snapshot requirements

A snapshot used for search must restore all information capable of changing future legality, observations, values, or events.

Minimum snapshot contents:

- rules configuration reference/version;
- board occupancy;
- all 80 piece records;
- acting player;
- ply and game counters;
- terminal status/result;
- recent 16-move history;
- active threat relations;
- latest five behavioral events for every piece.

The complete long-form replay record does not need to be copied into every search snapshot.

After restore, the following must match the original state exactly:

- legal-action mask;
- 127-channel observation for either player;
- public board view for either player;
- next transition under the same action;
- behavioral events generated by that transition;
- terminal result.

---

## 16. Privileged belief-learning target

The belief head may receive a training target constructed from privileged state, but this target must be kept separate from model inputs.

For each hidden opponent piece visible as an unresolved identity to the acting player, the target may contain:

- stable piece identifier;
- current square;
- true piece type.

The observation builder must not use the target to construct channels 0-127.

A test must verify that permuting true types among still-hidden opponent pieces changes the belief target but does not change the player observation when public history remains identical.

---

## 17. Derived versus stored information

### Store authoritatively

- true piece type;
- piece ownership and stable identifier;
- starting/current squares;
- alive/moved status;
- observer knowledge flags;
- board occupancy;
- counters and acting player;
- recent move records;
- active threat relations;
- latest behavior-event metadata.

### Derive when requested

- legal-action mask;
- all 127 observation channels;
- unresolved inventory;
- setup-memory planes;
- public board view;
- effective outcome value;
- browser display labels;
- belief-learning target tensors.

This keeps the core state compact and avoids multiple sources of truth.

---

## 18. Internal-state invariants

Every engine transition must preserve all of the following:

1. exactly 40 physical piece records exist per player for the entire game;
2. every live piece appears on exactly one occupiable square;
3. every captured piece appears on no square;
4. piece owner and identifier never change;
5. starting square and true type never change;
6. identity knowledge is monotonic;
7. each player always knows their own identities;
8. a hidden opponent identity cannot become known without a legal public cause;
9. Flag and Bomb never acquire `has_moved=true` in a reachable legal game;
10. active threat relations reference only currently live pieces at creation time;
11. behavior records never require privileged type information to decide whether an event occurred;
12. board occupancy and piece records agree exactly.

Any invariant failure is a hard engine error during development.

---

## 19. Phase 1 acceptance

This specification is ready for implementation when:

- all fields needed by `observation_v2_127ch` are accounted for;
- no model input requires privileged hidden information;
- snapshot contents are sufficient to reproduce observations exactly;
- public event schemas have been defined;
- validation tests cover state consistency, information security, behavior reconstruction, and replay.
