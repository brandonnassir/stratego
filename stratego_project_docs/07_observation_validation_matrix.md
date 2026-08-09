# Observation Validation Matrix for `observation_v2_127ch`

## 1. Purpose

This document converts the observation specification into testable engine behavior.

The observation is not accepted because its tensor shape is correct. It is accepted only when each channel changes exactly when intended and never reveals information unavailable to the acting player.

All tests apply first to the Python reference engine. Any optimized backend must reproduce the reference tensor exactly for the same full state and public history.

---

## 2. Global acceptance rules

For every observation test:

- tensor shape must be exactly `(127, 10, 10)`;
- all values must be finite;
- binary planes must contain only `0` and `1`;
- normalized coordinate planes must remain in `[-1, 1]`;
- behavioral recency/rank/actor-knew planes must remain in `[0, 1]`;
- behavioral special planes must remain in `{-1, 0, +1}`;
- game-progress and no-battle-progress planes must remain in `[0, 1]`;
- perspective normalization must be deterministic;
- the same replay state must always produce byte-identical or numerically exact observations within the chosen tensor type.

---

# 3. Channel-group validation matrix

| Channels | Trigger / source | Expected behavior | Clearing / persistence | Critical failure tests |
|---|---|---|---|---|
| 0-11 own identities | Current own live pieces | Exactly one type plane marks each own piece at its current square | Moves with piece; disappears on capture | Wrong type, duplicate own type on same square, captured piece remains |
| 12-23 known opponent identities | Public opponent identity knowledge | Known live opponent appears in exact type plane | Moves with piece; disappears on capture | Hidden type exposed; known identity forgotten |
| 24 hidden opponent occupancy | Opponent live piece not exactly known | `1` on hidden opponent current square | Moves with piece; clears when identity becomes known/captured | Overlap with 12-23; hidden square omitted |
| 25 own known-to-opponent | Opponent has exact public knowledge of own piece | `1` on current square | Moves with piece; disappears on capture | Information not set after reveal; wrongly set for unrevealed piece |
| 26 own moved | Own piece has moved at least once | `1` follows piece | Persists until capture | Resets after later moves or changes square incorrectly |
| 27 opponent moved | Opponent piece has moved at least once | `1` follows piece | Persists until capture | Hidden Flag/Bomb marked moved in reachable legal state |
| 28-31 live-piece origins | Stable starting square of current live piece | Normalized origin row/column written at current square | Follows piece; disappears on capture | Origin changes when piece moves; perspective transform wrong |
| 32-43 own setup memory | Own original setup | Static type-by-start-square map | Persists entire game, including after capture | Setup memory altered by movement/capture |
| 44-55 known opponent setup memory | Opponent piece identity becomes legally known | Write type at original square | Persists after move/capture | Identity written before reveal; original square incorrect |
| 56-67 unresolved inventory | Official count minus number of known opponent identities | One normalized scalar per type broadcast everywhere | Changes only when a new identity of that type becomes known | Uses live count instead of unresolved count; hidden information used |
| 68-87 own behavior | Latest qualifying event per own live piece | Four values per behavior at current piece square | Recency decays; event replaced by later same-type event; disappears on capture | Event follows square instead of piece; context leaks hidden type |
| 88-107 opponent behavior | Latest qualifying event per opponent live piece | Same structure as own behavior | Same rules | Opponent-private knowledge leaked to observer |
| 108-123 recent moves | Public move history | `-1` source, `+1` destination, newest first | Shifts one plane per ply; old 16th drops | Ordering reversed; perspective transform wrong |
| 124 lake mask | Static board geometry | `1` on exactly 8 lake cells | Never changes | Wrong cell count/geometry |
| 125 game progress | Total move counter | Same normalized scalar on all cells | Increases monotonically until terminal | Nonuniform values; exceeds 1 |
| 126 no-battle progress | Consecutive moves since combat | Same normalized scalar on all cells | Increments without combat; resets on combat | Wrong denominator; fails to reset |

---

# 4. Piece-identity tests

## 4.1 Own piece identity planes 0-11

For each of the 12 types:

1. place one own live piece of that type at a known square;
2. construct observation;
3. verify exactly one `1` in the matching type plane at that square;
4. move the piece legally;
5. verify the `1` moves to the destination;
6. capture the piece;
7. verify no own type plane marks the captured piece.

**Gate:** zero failures across all piece types.

## 4.2 Known opponent planes 12-23 and hidden occupancy 24

Construct two full states with identical public history but different true hidden types at the same opponent square.

Expected:

- channels 12-23 identical;
- channel 24 identical;
- every other player-observation channel identical unless public consequences differ.

Then legally reveal the piece.

Expected:

- channel 24 clears at that square;
- exactly one correct channel among 12-23 becomes `1`;
- channel 44-55 records its original setup square.

**Anti-leak gate:** changing an unrevealed hidden type alone must not alter the observation.

## 4.3 Own known-to-opponent plane 25

Test three states for the same own physical piece:

1. never revealed;
2. revealed and still alive;
3. logically identified as Scout by multi-square move.

Expected:

- state 1: `0`;
- states 2 and 3: `1` at current square.

---

# 5. Movement and origin tests

## 5.1 Moved-status planes 26-27

For each side:

- before first move: `0`;
- after first legal move: `1`;
- after subsequent moves: remains `1`;
- after capture: disappears because piece no longer occupies a square.

## 5.2 Origin-coordinate planes 28-31

For each starting row and column index used by legal setup squares:

1. compute expected normalization \(2i/9-1\);
2. move the physical piece to several squares;
3. verify its original coordinates remain unchanged;
4. verify coordinate values are written at the piece's **current** square;
5. capture piece and verify values disappear.

### Perspective test

Create color-swapped/rotated equivalent games.

After normalization, corresponding pieces must have identical origin-coordinate values at equivalent normalized locations.

---

# 6. Setup-memory tests

## 6.1 Own original setup 32-43

For a complete legal setup:

- exactly 40 ones must exist across the 12 planes;
- every setup square must be represented once;
- moving pieces must not change these planes;
- captures must not change these planes.

## 6.2 Known opponent setup identity 44-55

For an opponent piece starting at square `s`:

- before identity is known: all 12 channels are `0` at `s`;
- once known: exactly one type channel becomes `1` at `s`;
- after subsequent movement: entry stays at `s`;
- after capture: entry stays at `s`.

### Anti-leak test

Two games with different true hidden setup identities but identical public history must produce identical channels 44-55 until the corresponding identities are revealed.

---

# 7. Unresolved-inventory tests 56-67

For every type \(T\):

1. initial state with no opponent identities known;
2. verify channel value is `1.0` everywhere;
3. reveal one identity of type \(T\);
4. verify value becomes \((N_T-1)/N_T\);
5. reveal all identities of type \(T\);
6. verify value becomes `0.0` everywhere.

### Critical distinction

Capture an opponent piece whose identity was already known before capture.

Expected:

- unresolved count must **not** change at capture time because that identity had already been removed from the unresolved pool.

Capture a hidden opponent piece through combat, revealing its identity at capture.

Expected:

- unresolved count decreases exactly once when the identity becomes known.

**Gate:** unresolved inventory must depend on public identity knowledge, not hidden live counts.

---

# 8. Behavioral event tests

The same test suite must be applied from both ownership perspectives.

---

## 8.1 Threat

### Positive case

1. move actor piece `P` to an empty square;
2. after the move, `P` is orthogonally adjacent to opponent `Q`;
3. no combat occurred between `P` and `Q`.

Expected:

- threat event stored for `P`;
- recency = `1.0` immediately after the event;
- counterpart = deterministic selected `Q`.

### Negative cases

No threat event when:

- adjacency is diagonal only;
- actor does not survive the move;
- `Q` is the piece attacked by the action;
- actor ends non-adjacent;
- only a friendly piece is adjacent.

### Multiple-counterpart determinism

If two opponent pieces are adjacent after the move, verify the counterpart with the lowest absolute board-square index is selected.

The selection must be unchanged if the two hidden true types are swapped.

---

## 8.2 Evade

### Positive case

1. previous opponent move creates a recorded threat from `A` to `P`;
2. on current turn, `P` moves to an empty square;
3. after movement, `P` is not adjacent to `A`.

Expected: evade event for `P`.

### Negative cases

No evade when:

- `P` attacks instead of moving to empty square;
- another friendly piece moves;
- `P` remains adjacent to `A`;
- the threat occurred more than one opponent move earlier;
- the previous move did not create a valid threat.

---

## 8.3 Declined attack

### Positive case

At turn start, `P` has a legal adjacent attack against `Q`, but the player chooses another action.

Expected: declined-attack event for `P`.

Important: this must occur even if `P` itself was not moved.

### Negative cases

No declined attack when:

- `P -> Q` is selected;
- `P` is immovable;
- `Q` is not adjacent;
- destination attack is illegal for another rule reason.

### Multiple opportunities

If `P` could legally attack two adjacent opponent pieces and attacks neither, verify the lower-index counterpart is chosen without inspecting hidden types.

---

## 8.4 Protect

### Positive case

1. previous opponent move threatens friendly `B`;
2. current player moves distinct piece `C` to an empty square;
3. `C` was not adjacent to `B` before the move;
4. `C` is adjacent to `B` after the move.

Expected:

- `protect` event for `C`, counterpart `B`;
- `was_protected` event for `B`, counterpart `C`.

### Negative cases

No protection when:

- `C == B`;
- `C` was already adjacent to `B` before the move;
- `C` attacks rather than moves to empty square;
- `B` was not threatened on immediately preceding move;
- `C` ends non-adjacent to `B`;
- only an empty square rather than a piece is notionally protected.

The final case verifies our deliberate version-1 exclusion of empty-square protection.

---

## 8.5 Behavioral recency

For a stored event at ply \(t_e\), at later ply \(t\):

\[
\Delta=t-t_e,
\qquad
R=\frac{1}{1+\Delta/32}.
\]

Verify exact expected values at at least:

- \(\Delta=0\);
- 8;
- 16;
- 32;
- 64;
- 128.

A later event of the same type for the same piece must replace the older event as the source of the four behavior channels.

---

## 8.6 Behavioral counterpart rank and special encoding

For known counterparts, verify:

- Spy -> `0.1` rank;
- Miner -> `0.3`;
- Colonel -> `0.8`;
- Marshal -> `1.0`;
- Bomb -> rank `0`, special `+1`;
- Flag -> rank `0`, special `-1`.

For unknown counterparts, verify rank and special are `0`.

---

## 8.7 Actor-knew flag

Construct behavior where:

1. actor knew counterpart identity at event time;
2. actor did not know counterpart identity at event time.

Verify `actor-knew` is `1` and `0`, respectively.

This flag is historical and must remain what was true at event time even if the counterpart becomes known later.

---

## 8.8 Retrospective legal reinterpretation

Construct an event where the current observing player did not initially know the counterpart identity.

Observation immediately after event:

- rank = `0`;
- special = `0`.

Later reveal the counterpart identity through a legal public event.

Reconstruct the later observation.

Expected:

- the old behavior event remains the latest event;
- if the actor knew the identity at event time and the observer now legally knows it, rank/special may now reflect that legally known identity;
- no event timestamp changes.

---

## 8.9 Behavioral anti-leak differential test

Create two full states with identical public history and identical behavior-event identifiers/timestamps, but swap the true types of still-hidden counterpart pieces.

Expected:

- all behavior channels visible to the acting player are identical.

This test must be repeated for all five behavior types.

**Gate:** zero differences.

---

# 9. Recent-move tests 108-123

For a sequence of at least 20 public moves:

- newest move must always occupy channel 108;
- previous entries shift upward by exactly one channel after each ply;
- after 16 plies, the oldest tracked move is dropped;
- each plane contains exactly one `-1` source and one `+1` destination;
- all other values are `0`.

For perspective-normalized equivalent games, recent-move planes must map identically after coordinate transformation.

---

# 10. Global-channel tests

## 10.1 Lake mask 124

- exactly 8 ones;
- exactly 92 zeros;
- static across all states;
- correctly transformed under the chosen perspective normalization.

## 10.2 Game progress 125

At total moves:

- 0 -> `0.0`;
- 1000 -> `0.25`;
- 2000 -> `0.5`;
- 4000 -> `1.0`.

The plane must be spatially constant.

## 10.3 No-battle progress 126

### Training configuration

- 0 battleless moves -> `0.0`;
- 50 -> `0.5`;
- 99 -> `0.99`;
- 100 -> terminal draw threshold / value clamped to `1.0` if observation exists for terminal diagnostic purposes.

### Evaluation configuration

- 100 -> `0.5`;
- 199 -> `0.995`;
- 200 -> terminal draw threshold.

Any combat resets the counter and channel to `0.0` for the subsequent nonterminal observation.

---

# 11. Full-observation anti-leak suite

This suite is mandatory before reinforcement-learning training.

For many randomly generated valid public histories:

1. clone the full hidden state;
2. permute true identities only among still-hidden opponent pieces while preserving all public constraints;
3. reconstruct acting-player observations;
4. require equality of all 127 channels unless the permutation changes something already legally deducible from public history.

Recommended Phase 1/2 acceptance target:

- at least **100,000 randomized hidden-state permutations**;
- **zero unexplained observation differences**.

This is one of the highest-priority correctness gates in the project.

---

# 12. Replay reconstruction suite

For at least 10,000 complete reference-engine games:

1. store only rules version, observation version, setups, and action history;
2. replay from the beginning;
3. reconstruct `observation_v2_127ch` at every ply;
4. compare against observations recorded during original play.

**Gate:** zero mismatches.

---

# 13. Optimized-backend differential suite

If an optimized engine is introduced:

For identical snapshots and action histories, Python reference and optimized backends must return identical:

- legal-action masks;
- full public event history;
- all 127 observation planes;
- behavioral event metadata;
- belief targets;
- terminal results.

Recommended pre-training gate:

- at least 1,000,000 randomly sampled state comparisons;
- zero rule or observation mismatches.

---

# 14. Phase gate

The observation representation is considered frozen for initial model development when:

- every test in this document is implemented;
- targeted tests pass;
- randomized anti-leak tests pass;
- deterministic replay reconstruction passes;
- the documented channel ordering is exported as machine-readable metadata by the engine package;
- no unresolved interpretation remains for any behavioral event.

At that point, changing the observation requires a new version identifier rather than silently altering `observation_v2_127ch`.
