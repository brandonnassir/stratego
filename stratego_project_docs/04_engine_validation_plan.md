# Game Engine Validation Plan

## 1. Acceptance principle

The model must not train on an engine whose rules have not been independently validated.

A rule bug is more dangerous than a weak neural network because it can contaminate every self-play trajectory.

The Python reference engine is the behavioral source of truth. Any later optimized backend must pass differential tests against it.

### Current acceptance status

The project has passed both the reference-engine correctness gate and the Phase 3 production-training readiness gate.

Frozen reference:

- `phase2_1_reference_1.1.0`;
- `stratego_project_v1`;
- `observation_v2_1_127ch`;
- 10,000-entry source-destination action encoding.

Phase 2.1 evidence:

- 1,255 automated tests passed at the freeze;
- 120 combat cases, 0 failures;
- 1,804 mirrored position pairs / 3,608 observation comparisons, 0 mismatches;
- 103,625 valid hidden-information permutations, 0 public-information mismatches;
- 10,000 deterministic replay games / 5,078,406 plies, 0 mismatches;
- 600 snapshot/restore cases, 0 mismatches;
- 1,045,111 invariant-checked transitions, 0 violations.

Phase 3 production evidence:

- repository suite reached 1,497 passing tests;
- 10,048 integrated end-to-end environment-step comparisons, 0 mismatches;
- 11,251 integrated stored-decision reconstructions, 0 mismatches;
- two-hour soak: 63,871,488 positions, 123,718 games, 0 errors/restarts;
- 411,818 reconstruction checks during the soak, 0 mismatches;
- 0 bytes swap at start/end;
- 0 unexplained coordinator memory growth;
- first-to-last-quarter throughput drift: -0.76%;
- backend decision: `KEEP_PYTHON`, measured \(R=6.50\).

Agent 6 / a separate optimized backend was not triggered.

---

## 2. Static board tests

Verify:

- board dimensions are 10 by 10;
- exactly 92 squares are occupiable;
- exactly 8 lake squares are non-occupiable;
- lake geometry is fixed and symmetric;
- coordinate conversion round-trips correctly for all 100 squares.

**Gate:** zero failures.

---

## 3. Piece inventory tests

For each player verify exact counts:

- Flag: 1;
- Spy: 1;
- Scout: 8;
- Miner: 5;
- Sergeant: 4;
- Lieutenant: 4;
- Captain: 4;
- Major: 3;
- Colonel: 2;
- General: 1;
- Marshal: 1;
- Bomb: 6.

Test setup validator rejection for every incorrect count class.

**Gate:** zero invalid setups accepted and zero valid setups rejected in the curated test set.

---

## 4. Ordinary movement tests

For every movable non-Scout piece test:

- one square north;
- one square south;
- one square east;
- one square west;
- no diagonal move;
- no two-square ordinary move;
- no move off board;
- no move onto friendly piece;
- no move onto lake;
- attack onto adjacent enemy square is legal.

Test from corners, edges, near lakes, and interior squares.

---

## 5. Immovable-piece tests

Flag and Bomb must have zero legal moves in every position.

They may still be attacked.

---

## 6. Scout movement tests

Test Scout rays in all four directions:

- one-square move;
- several-square move;
- stopping before a piece;
- attacking the first opponent piece on the ray;
- friendly piece blocks movement;
- opponent piece blocks movement beyond the attack square;
- lake blocks movement;
- board edge terminates the ray;
- no diagonal move;
- no jumping.

A multi-square move must update public knowledge so the opponent observation identifies the piece as a Scout.

---

## 7. Combat matrix tests

Create exhaustive attacker-versus-defender tests for every meaningful pair of piece types.

For ordinary ranked pieces verify:

- higher rank wins;
- lower rank loses;
- equal rank removes both.

Special cases:

- Spy attacks Marshal -> Spy survives;
- Marshal attacks Spy -> Marshal survives;
- Miner attacks Bomb -> Miner survives;
- every other movable piece attacks Bomb -> attacker is removed and Bomb survives;
- any movable piece attacks Flag -> attacker survives on Flag square and game ends.

Post-combat occupancy must match the project rules contract.

**Gate:** exhaustive combat table passes.

---

## 8. Reveal and information tests

Verify:

- hidden opponent identities are absent from normal observations;
- own identities are always present to the player;
- both combat identities become public;
- surviving revealed identities remain public on later turns;
- captured identities are reflected in remaining-piece counts;
- multi-square Scout movement produces logical Scout knowledge;
- red's observation never exposes unrevealed blue types;
- blue's observation never exposes unrevealed red types.

Include explicit anti-leak tests that compare observations from two full states that differ only in hidden opponent identities. The observations must be identical unless the difference is legally inferable through public history.

---

## 8A. Observation-version validation

The authoritative first model observation is `observation_v2_1_127ch`.

Validation must follow `07_observation_validation_matrix.md`, including:

- exact channel ordering and ranges;
- own/opponent identity state;
- disclosure state;
- movement status;
- piece-origin coordinates;
- persistent setup memory;
- unresolved opponent inventory;
- formal threat, evade, declined-attack, protect, and was-protected semantics;
- behavioral recency/context features;
- recent-move ordering;
- global progress planes;
- randomized hidden-information anti-leak differential tests;
- exact replay reconstruction.

Recommended anti-leak gate before reinforcement-learning training: at least **100,000 randomized hidden-state permutations** with zero unexplained observation differences.

---

## 9. Win-condition tests

Verify immediate termination for:

- red captures blue Flag;
- blue captures red Flag;
- acting player's move leaves opponent with no legal moves.

Result signs must be correct from both player perspectives.

---

## 10. Draw-condition tests

Verify:

- neither player has legal movement -> draw;
- 100 battleless moves in training configuration -> draw;
- combat resets the battleless counter;
- 200 battleless moves in evaluation configuration -> draw;
- absolute move limit -> draw;
- terminal reason is recorded correctly.

---

## 11. Deliberate exclusion tests

These tests ensure excluded rules do not accidentally reappear.

### Two-square exclusion

Construct repeated A-B-A-B movement sequences. They must remain legal until another termination condition is reached.

### Continuous-chasing exclusion

Construct a legal repeating chase. The engine must not reject a move solely because the chase recreates an earlier chase position.

---

## 12. Perspective normalization tests

Create a position and its color-swapped/rotated equivalent.

After acting-player normalization:

- own pieces must occupy equivalent coordinates;
- feature planes must transform consistently;
- legal actions must map correctly;
- value target sign/perspective must be correct.

---

## 13. Action encoding tests

For every legal source-destination pair tested:

- source/destination -> action identifier -> source/destination must round-trip exactly;
- legal list and legal mask must agree;
- illegal actions must be masked;
- no legal action may map to the same identifier as another legal action.

---

## 14. Snapshot/restore tests

At random nonterminal states:

1. save snapshot;
2. advance through a legal action sequence;
3. restore snapshot;
4. replay the same actions.

Require identical states, observations, legal actions, counters, and result.

---

## 15. Deterministic replay tests

For thousands of generated games, store only:

- rules version;
- setups;
- action sequence.

Replay from scratch and verify exact equality at every move.

Recommended initial gate:

- at least **10,000 complete random/heuristic games** with zero replay mismatch.

---

## 16. Randomized property tests

Across large numbers of randomly generated legal states verify invariants:

- total live + captured pieces equals 40 per player;
- no square contains more than one piece;
- no live piece occupies a lake;
- no piece exists in two squares;
- current player is valid;
- captured pieces cannot move;
- Flag/Bomb never move;
- legal actions never start from an opponent piece;
- terminal games have no further transition unless reset.

---

## 17. Long-run stability test — PASSED in Phase 3

Run the integrated batch/shared-memory/Metal/trajectory pipeline continuously while monitoring:

- memory growth;
- state corruption;
- exceptions;
- impossible piece counts;
- terminal-reason proportions;
- throughput drift;
- worker liveness;
- swap usage;
- trajectory reconstruction.

Accepted Phase 3 soak:

- duration: 7,200.1 seconds;
- positions: 63,871,488;
- games completed/resets: 123,718;
- workers alive: 10/10 throughout;
- errors/restarts: 0/0;
- reconstruction checks: 411,818;
- reconstruction mismatches: 0;
- coordinator memory growth: 0 bytes;
- system swap: 0 -> 0 bytes;
- first-vs-last-quarter throughput change: -0.76%.

Four terminal reasons occurred naturally at scale: Flag capture, battleless-limit draw, opponent-no-legal-move win, and both-no-legal-move draw. The absolute-move-limit draw did not occur naturally because the battleless limit ordinarily preempts it.

**Gate status:** passed.

---

## 18. Differential testing for optimized backend

If an optimized backend is added, run the same setups and actions through both backends.

Compare after every move:

- full board;
- acting player;
- public information;
- legal action mask;
- counters;
- terminal state;
- result.

Recommended acceptance gate:

- at least **100,000 randomized state/action comparisons** with zero behavioral mismatch;
- at least **10,000 full games** with identical replay.

Any mismatch blocks the optimized backend from training use.

---

## 19. Performance gate before model integration — PASSED in Phase 3

The accepted decision is based on measured end-to-end-relevant rates, not core-count extrapolation.

Measured:

- simulation pipeline: 96,963 positions/s;
- representative model sustainable rate used for denominator: 14,922 positions/s;
- ratio: \(R=6.50\);
- integrated finalist: 12,838 positions/s;
- production-style collecting soak: 8,871 positions/s.

Decision rule:

- `R >= 2.0`: retain Python;
- `1.25 <= R < 2.0`: retain Python initially, optimization optional;
- `R < 1.25`: evaluate separate optimized backend.

**Result:** `KEEP_PYTHON`.

The end-to-end profile independently supports this result: Metal inference dominates the integrated step and workers have substantial idle headroom.

This decision must be re-measured for the final model architecture rather than assumed permanently.

---

## 20. Engine readiness gates

### 20.1 Reference-engine correctness gate — PASSED in Phase 2.1

The Python reference engine passed:

- rule unit tests;
- exhaustive combat tests;
- legal-action list/mask consistency;
- observation and mirror equivalence;
- hidden-information anti-leak tests;
- 10,000-game deterministic replay;
- snapshot/restore;
- randomized invariant stress;
- frozen rules/observation/action versions.

Frozen implementation:

- `phase2_1_reference_1.1.0`.

### 20.2 Production-training readiness gate — PASSED in Phase 3

The integrated simulator pipeline passed:

- batched equivalence to the frozen engine;
- independent reset/generation validation;
- persistent shared-memory transport;
- representative Metal model integration;
- worker/environment/batch scaling;
- exact compact trajectory reconstruction;
- two-hour continuous soak;
- memory/swap stability;
- explicit backend-ratio measurement.

Backend decision:

- `KEEP_PYTHON`;
- \(R=6.50\);
- optimized backend not required.

**Status:** production simulation infrastructure is accepted for subsequent project phases. The final model and training loop still require their own future validation gates.

---

## 21. Internal-state and public-event tests

The requirements in `08_internal_state_spec.md` and `09_public_event_and_replay_schema.md` add the following mandatory tests.

### 21.1 Piece-table / board consistency

After every transition in randomized games:

- every live piece appears on exactly one board square;
- every occupied board square points to exactly one live piece;
- captured pieces occupy no square;
- starting square, owner, identifier, and true type never change;
- knowledge flags never revert from known to unknown.

**Reference-engine correctness gate:** zero invariant violations over at least **1,000,000 checked transitions spanning at least 2,000 complete randomized games**.

Phase 2.1 measured 1,045,111 checked transitions across 2,000 games with zero violations. The separate multi-hour batched soak in section 17 remains a Phase 3 production-readiness requirement.

### 21.2 Type-independent identifier test

Create two games with identical piece locations and public history but permute true types among hidden setup slots.

Expected:

- the same physical setup slots receive the same piece identifiers;
- identifiers reveal no type information;
- public events remain identical until a permuted identity is legally revealed.

### 21.3 Knowledge-monotonicity test

For combat and multi-square Scout revelation:

- verify legal observer knowledge changes exactly once;
- verify knowledge persists after later movement and capture;
- verify unrelated hidden pieces remain unresolved.

### 21.4 Active-threat relation test

Create a move that ends adjacent to several opponent pieces.

Expected:

- every qualifying threat relation is retained for the next response;
- the long-term `threat` behavior record selects only the deterministic counterpart with the lowest normalized square index, per `06_observation_v2_127ch.md` section 10.6;
- hidden type permutations do not change relation membership or selection;
- the colour-swapped, 180-degree-rotated equivalent position selects the mirror image of the same counterpart.

### 21.5 Snapshot completeness test

Construct a state containing:

- recent moves;
- known and hidden pieces;
- active threat relations;
- all five behavior types;
- a nonzero battleless counter.

Snapshot, advance, restore, then require exact equality of:

- full state;
- both observer views;
- `observation_v2_1_127ch`;
- legal-action mask;
- next generated events under the same action.

### 21.6 Public-event anti-leak test

For at least 100,000 hidden-type permutations preserving public history:

- observer-filtered event stream must remain identical;
- observer-filtered board state must remain identical;
- 127-channel observation must remain identical.

Only privileged replay data and belief targets may differ.

### 21.7 Replay-derived event test

For at least 10,000 complete games:

1. save only rules version, true setups, and action sequence;
2. reconstruct from scratch;
3. compare move, reveal, combat, behavior, and game-end events in order;
4. compare observer views and observations at every ply.

**Gate:** zero mismatch.

### 21.8 Browser-view privacy test

For each player perspective:

- own identities are exact;
- unresolved opponent identities are hidden;
- known opponent identities remain visible after reveal;
- no privileged type, belief target, or opponent setup identity appears in browser payloads before legal revelation.

A single unexplained hidden-information leak blocks model integration.
---

## 22. Model action-sampling safety regression

Any model/action sampler used with the frozen engine must satisfy:

1. sampled action is in the exact engine-generated legal set;
2. sampler intermediate values used for ranking/selection are finite where required;
3. masking with `-inf` cannot combine with non-finite random noise to produce a `NaN` winner;
4. uniform random draws passed through singular transforms are clamped/bounded away from singular endpoints;
5. coordinator performs an explicit sampled-action legality check before worker application;
6. illegal sampled action causes a loud correctness failure and no state mutation.

### Phase 3 regression basis

The representative Gumbel-max sampler originally allowed a boundary uniform draw to create non-finite Gumbel noise. Combined with an illegal action's `-inf` logit, this could create `NaN` before `argmax`.

The frozen engine caught the illegal action before mutation. The sampler was corrected and regression tests were added.

This validation requirement applies to future samplers even if they do not use Gumbel-max: model-side legality is never trusted over engine legality.
