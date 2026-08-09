# Observation Specification: `observation_v2_127ch`

## 1. Status

**Status:** Approved Phase 1 design.

This document is the authoritative model-observation contract for the first Stratego system. It replaces the earlier draft identifier `observation_v1_68ch`.

Any future change to channel meaning, normalization, ordering, or perspective convention requires a new observation-version identifier.

---

## 2. Source basis versus project design

### Source-derived ideas

The Ataraxos paper, Appendix C (pages 20-25), represents Stratego information states with spatial feature planes covering current pieces, hidden-piece occupancy, movement history, threats, evasions, declined attacks, captures, causes of death, protection relationships, starting locations, and recent moves. Appendix C.2 represents recent moves with one plane per move using `-1` at the source and `+1` at the destination.

### Project design choices

This project intentionally uses a much smaller representation suitable for local training on an M4 Pro Mac mini. The 127-channel design below is **not** claimed to be the Ataraxos representation. It is our reduced representation inspired by the kinds of information the paper preserves.

We deliberately omit:

- the paper's hand-computed uniform-policy piece-probability channels;
- most cause-of-death channels;
- type-specific threat/evasion/protection planes;
- approximately 100 one-hot starting-location planes;
- 32-move history in favor of 16 recent moves plus persistent behavioral summaries.

---

## 3. Tensor shape and perspective

The observation is a floating-point tensor:

\[
X \in \mathbb{R}^{127 \times 10 \times 10}.
\]

The acting player's perspective is normalized before the tensor is returned to the model:

- the acting player's side is always presented in the same orientation;
- color itself is not needed by the model after normalization;
- all coordinates in the observation, including starting-location coordinates, use the normalized perspective.

The legal-action mask is a **separate model input** and is not included in the 127 planes.

---

## 4. Channel map

| Channels | Count | Meaning |
|---|---:|---|
| 0-11 | 12 | Current identities of own pieces |
| 12-23 | 12 | Current identities of known opponent pieces |
| 24 | 1 | Current hidden opponent-piece locations |
| 25 | 1 | Own pieces whose identities are known to the opponent |
| 26-27 | 2 | Own/opponent pieces that have moved |
| 28-31 | 4 | Starting coordinates of live pieces |
| 32-43 | 12 | Complete original own setup |
| 44-55 | 12 | Known opponent setup identities by original square |
| 56-67 | 12 | Unresolved opponent inventory |
| 68-87 | 20 | Behavioral history of own pieces |
| 88-107 | 20 | Behavioral history of opponent pieces |
| 108-123 | 16 | Most recent 16 moves |
| 124 | 1 | Lake mask |
| 125 | 1 | Normalized game progress |
| 126 | 1 | Normalized no-battle-rule progress |
| **Total** | **127** | |

---

# 5. Piece identity planes

## Channels 0-11 — current own piece identities

One binary plane per piece type in the following fixed order:

| Offset | Type |
|---:|---|
| 0 | Spy |
| 1 | Scout |
| 2 | Miner |
| 3 | Sergeant |
| 4 | Lieutenant |
| 5 | Captain |
| 6 | Major |
| 7 | Colonel |
| 8 | General |
| 9 | Marshal |
| 10 | Flag |
| 11 | Bomb |

For each living own piece, write `1` at its current square in the corresponding plane. All other cells are `0`.

Own piece identities are always legal information to the acting player.

## Channels 12-23 — current known opponent identities

The same 12-type order is used for opponent pieces whose exact identities are currently known to the acting player.

An opponent identity is known if it has become public through a legal game event, including:

- combat revelation;
- logically identifying a Scout through a multi-square Scout move.

Known identities remain known for the remainder of the piece's life.

## Channel 24 — hidden opponent occupancy

Binary plane:

- `1` at every square occupied by an opponent piece whose exact identity is unknown to the acting player;
- `0` elsewhere.

A square must never simultaneously be `1` in channel 24 and in one of channels 12-23.

## Channel 25 — own identity known to opponent

Binary plane:

- `1` at the current square of each own living piece whose exact identity is known to the opponent;
- `0` otherwise.

This channel lets the model reason about information it has revealed to the opponent.

---

# 6. Movement and piece-origin planes

## Channel 26 — own pieces that have moved

Binary plane marking living own pieces that have moved at least once.

The value follows the physical piece as it moves.

## Channel 27 — opponent pieces that have moved

Binary plane marking living opponent pieces that have moved at least once.

The value follows the physical piece as it moves.

This is public information and is strategically useful because a moved hidden piece cannot be a Flag or Bomb.

## Channels 28-31 — starting coordinates of live pieces

Each living piece carries its original setup square. That information is encoded at the piece's **current square**.

| Channel | Meaning |
|---:|---|
| 28 | Own starting row |
| 29 | Own starting column |
| 30 | Opponent starting row |
| 31 | Opponent starting column |

### Coordinate normalization

For normalized row or column index \(i \in \{0,\dots,9\}\):

\[
\operatorname{coord}(i)=2\frac{i}{9}-1.
\]

Therefore:

- index 0 maps to `-1`;
- index 9 maps to `+1`.

Empty squares and squares occupied by the other ownership category contain `0` in the corresponding coordinate planes.

This representation lets the model track the same concealed physical piece over time without encoding its true hidden identity.

---

# 7. Persistent setup memory

## Channels 32-43 — complete original own setup

Twelve binary planes in the fixed piece-type order.

These planes never change during a game. Each own piece remains represented at its **starting square**, even after moving or being captured.

This is legal memory available to the player.

## Channels 44-55 — known opponent original identities

Twelve binary planes in the same piece-type order.

When an opponent piece's identity becomes legally known, write `1` at that piece's **original setup square** in the corresponding type plane.

The entry persists even after the piece is captured.

Before the identity becomes known, no type is written for that original square.

This supports retrospective setup-family inference without leaking hidden identities.

---

# 8. Unresolved opponent inventory

## Channels 56-67

One broadcast scalar plane per piece type.

For type \(T\):

\[
U_T=N_T-K_T,
\]

where:

- \(N_T\) is the official initial count of type \(T\);
- \(K_T\) is the number of opponent pieces of type \(T\) whose identity is currently known to the acting player, whether alive or captured.

The channel value is normalized as:

\[
\hat U_T=\frac{U_T}{N_T}.
\]

The scalar \(\hat U_T\) is broadcast to all 100 board cells.

### Interpretation

These channels answer:

> How many copies of each type still need to be assigned among the opponent's unresolved hidden-piece identities?

They are not simply counts of pieces still alive.

### Example

If the opponent began with 8 Scouts and three Scout identities are now known, then:

\[
\hat U_{\text{Scout}}=\frac{8-3}{8}=0.625.
\]

---

# 9. Behavioral history representation

The behavioral representation is piece-centric. Behavioral values are written at the **current square of the living piece** and move with that physical piece.

Five behaviors are represented for both players:

1. threat;
2. evade;
3. declined attack;
4. protect;
5. was protected.

Each behavior uses four channels:

1. recency;
2. counterpart rank;
3. whether the actor knew the counterpart identity at the event time;
4. special counterpart encoding.

Thus:

\[
5\text{ behaviors}\times 4\text{ features}=20
\]

channels per player, or 40 total.

---

## 9.1 Behavioral channel ordering

### Own pieces: channels 68-87

| Channels | Behavior | Feature order |
|---|---|---|
| 68-71 | Threat | recency, rank, actor-knew, special |
| 72-75 | Evade | recency, rank, actor-knew, special |
| 76-79 | Declined attack | recency, rank, actor-knew, special |
| 80-83 | Protect | recency, rank, actor-knew, special |
| 84-87 | Was protected | recency, rank, actor-knew, special |

### Opponent pieces: channels 88-107

| Channels | Behavior | Feature order |
|---|---|---|
| 88-91 | Threat | recency, rank, actor-knew, special |
| 92-95 | Evade | recency, rank, actor-knew, special |
| 96-99 | Declined attack | recency, rank, actor-knew, special |
| 100-103 | Protect | recency, rank, actor-knew, special |
| 104-107 | Was protected | recency, rank, actor-knew, special |

---

## 9.2 Behavioral feature semantics

### Feature A — recency

If the event has never happened for that piece:

\[
R_B=0.
\]

Otherwise, if the most recent event occurred \(\Delta\) plies ago:

\[
R_B=\frac{1}{1+\Delta/32}.
\]

Examples:

| Plies since event | Value |
|---:|---:|
| 0 | 1.000 |
| 8 | 0.800 |
| 16 | 0.667 |
| 32 | 0.500 |
| 64 | 0.333 |
| 128 | 0.200 |

A ply is one player's move.

### Feature B — counterpart rank

If the counterpart identity may legally be represented in the current observation and the counterpart is ranked:

\[
Q_B=\frac{\text{rank}}{10}.
\]

Examples:

- Spy: `0.1`;
- Miner: `0.3`;
- Colonel: `0.8`;
- Marshal: `1.0`.

Otherwise use `0`.

Bomb and Flag use `0` because they do not have ordinary numeric ranks.

### Feature C — actor knew counterpart identity

Binary historical fact:

\[
A_B\in\{0,1\}.
\]

It records whether the **actor of the behavior** knew the counterpart's identity at the time the event occurred.

This does not itself reveal the identity.

### Feature D — special counterpart

If the counterpart identity may legally be represented in the current observation:

\[
S_B=
\begin{cases}
+1 & \text{Bomb}\\
-1 & \text{Flag}\\
0 & \text{otherwise.}
\end{cases}
\]

If the current observer is not legally allowed to know the counterpart identity, use `0`.

---

## 9.3 Hidden-information safety rule for behavior

Behavior events must be stored using stable piece identifiers and event metadata. The observation builder computes rank/special context from what is legally known **at observation time**.

Counterpart rank or special-type information may be exposed only if:

1. the actor knew the counterpart identity when the behavior occurred; and
2. the current observing player is legally allowed to know that identity now.

Otherwise the rank and special channels are `0`.

This allows legal retrospective reasoning without hidden-information leakage.

---

# 10. Formal behavioral event definitions

## 10.1 Threat

Piece \(P\) creates a **threat** against opponent piece \(Q\) when:

1. \(P\) is the piece moved by the completed action;
2. the action has fully resolved;
3. \(P\) remains alive;
4. \(P\) is orthogonally adjacent to \(Q\) after the move;
5. \(Q\) was not the piece directly attacked by that action.

If several opponent pieces satisfy the definition, choose one counterpart deterministically using the lowest absolute board-square index after the move.

Combat itself is not recorded as a threat against the attacked piece.

## 10.2 Evade

Piece \(P\) records an **evade** when:

1. the immediately preceding opponent move created a threat from opponent piece \(A\) against \(P\);
2. \(P\) is selected on the current move;
3. \(P\) moves to an empty square;
4. after movement, \(P\) is no longer orthogonally adjacent to \(A\).

An attack is not counted as an evade.

## 10.3 Declined attack

At the start of the player's turn, piece \(P\) has a declined-attack opportunity against opponent piece \(Q\) when:

1. \(Q\) is orthogonally adjacent to \(P\);
2. \(P\) is movable;
3. the attack \(P\rightarrow Q\) is legal.

If the player's chosen action is not \(P\rightarrow Q\), then \(P\) records a declined-attack event.

The piece \(P\) does not have to be the piece moved that turn.

If multiple legal adjacent attacks were declined by the same piece, choose the counterpart occupying the lowest absolute board-square index at the start of the turn.

The selection rule must not inspect hidden piece types.

## 10.4 Protect

Version 1 deliberately excludes protection of empty squares.

Piece \(C\) records a **protect** event for friendly piece \(B\) when:

1. the immediately preceding opponent move created a threat from opponent piece \(A\) against \(B\);
2. the current player moves \(C\);
3. \(C\neq B\);
4. \(C\) moves to an empty square;
5. \(C\) was not orthogonally adjacent to \(B\) immediately before the move;
6. \(C\) is orthogonally adjacent to \(B\) immediately after the move.

If several threatened friendly pieces satisfy the definition, choose the one at the lowest absolute board-square index after the move.

## 10.5 Was protected

Whenever \(C\) protects \(B\):

- \(C\) receives a `protect` event;
- \(B\) receives a `was_protected` event.

The counterpart for `protect` is \(B\).

The counterpart for `was_protected` is \(C\).

---

# 11. Behavioral event storage requirements

The engine must retain enough information to reconstruct the latest event of each type for each physical piece.

For each behavior event, store at minimum:

- event type;
- actor piece identifier;
- counterpart piece identifier;
- event ply index;
- whether actor knew counterpart identity at the event time.

The engine must not store a behavior channel value as the authoritative record. Channel values are derived from event history when the observation is constructed.

If a piece is captured, its live per-piece behavioral features disappear from board-token channels because it no longer occupies a square.

---

# 12. Recent-move planes

## Channels 108-123

Represent the most recent 16 plies.

Ordering:

- channel 108: immediately preceding move;
- channel 109: two plies ago;
- ...;
- channel 123: sixteen plies ago.

Each plane contains:

- `-1` at the normalized source square;
- `+1` at the normalized destination square;
- `0` elsewhere.

Unused history at the beginning of a game is all zeros.

This follows the basic recent-move encoding idea used in Ataraxos Appendix C.2, but uses 16 rather than 32 moves.

---

# 13. Global channels

## Channel 124 — lake mask

Binary static plane:

- `1` on all 8 lake squares;
- `0` elsewhere.

## Channel 125 — normalized game progress

Broadcast scalar:

\[
G=\min\left(\frac{\text{total moves}}{4000},1\right).
\]

The 4,000-move limit is a project engineering safety rule during training, modeled after the corresponding safety limit used in the Ataraxos training setup.

## Channel 126 — normalized no-battle progress

Broadcast scalar:

\[
D=\frac{\text{moves since last combat}}{\text{active no-battle limit}}.
\]

For the current project:

- training: denominator `100`;
- evaluation/human play: denominator `200`.

Clamp to `[0,1]`.

---

# 14. Separate legal-action mask

The engine returns a 10,000-entry source-destination legality mask separately from the observation.

The observation must not be relied upon to enforce legality.

The engine is the sole authority on legal actions.

---

# 15. Belief-learning targets

For training only, the engine may additionally provide ground-truth hidden-piece labels:

- stable opponent piece identifier;
- current square;
- true type.

These labels are **targets**, not observation channels.

They must never be accessible to the policy/value/belief encoder input.

---

# 16. Version acceptance conditions

`observation_v2_127ch` is accepted only when:

1. every channel has a deterministic construction rule;
2. perspective normalization is fully tested;
3. anti-leak differential tests pass;
4. behavioral-event definitions pass targeted tests;
5. observation reconstruction from compact replay is exact;
6. tensor values remain within documented ranges;
7. the optimized engine, if introduced, exactly matches the Python reference observation tensors.

The detailed test matrix is in `07_observation_validation_matrix.md`.
