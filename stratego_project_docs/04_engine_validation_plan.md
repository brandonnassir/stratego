# Game Engine Validation Plan

## 1. Acceptance principle

The model must not train on an engine whose rules have not been independently validated.

A rule bug is more dangerous than a weak neural network because it can contaminate every self-play trajectory.

The Python reference engine is the behavioral source of truth. Any later optimized backend must pass differential tests against it.

### Current acceptance status

The project has passed both engine-readiness layers.

#### Reference-engine correctness gate — PASSED in Phase 2.1

Frozen:

- `phase2_1_reference_1.1.0`;
- `stratego_project_v1`;
- `observation_v2_1_127ch`;
- 10,000-entry source-destination action encoding.

#### Production-training readiness gate — PASSED in Phase 3

Accepted Phase 3 evidence includes:

- 1,497 repository tests passing after integration;
- 10,048 integrated end-to-end differential environment steps, 0 mismatches;
- 11,251 stored decisions reconstructed in the integrated gate, 0 mismatches;
- 2-hour continuous soak;
- 63,871,488 soak positions;
- 411,818 soak-time reconstructed decisions, 0 mismatches;
- zero deadlock;
- zero worker errors/restarts;
- zero swap usage;
- zero measured coordinator-memory growth;
- measured `R = 6.50`;
- production backend decision: **KEEP_PYTHON**.

Agent 6 / optimized-backend implementation is not required.

#### Evaluation-harness readiness gate — PASSED in Phase 4

Phase 4 established a reproducible observer-safe evaluation system, including a 100,000-trial policy hidden-information audit, a 44,544-game calibration league, and exact cross-worker reproduction.

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

The production multiprocess + shared-memory + Metal pipeline must run continuously for at least two hours while monitoring memory, swap, throughput, worker liveness, reconstruction correctness, and reset behavior.

Accepted Phase 3 soak:

- 7,200.1 seconds;
- 63,871,488 positions;
- 123,718 games and resets;
- 8,871 positions/second;
- 0 bytes coordinator-memory growth;
- 0 -> 0 bytes swap;
- -0.76% first-vs-last-quarter throughput change;
- 10/10 workers alive for all samples;
- 0 errors/restarts;
- 411,818 reconstructed decisions checked;
- 0 reconstruction mismatches.

**Status:** PASS.

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

## 19. Performance gate before model integration — PASSED

Measured simulation numerator:

- 96,963 positions/second.

Measured representative-model denominator:

- 14,922 positions/second.

\[
R =
\frac{96{,}963}{14{,}922}
=
6.50.
\]

Decision:

```text
KEEP_PYTHON
```

The integrated profile confirms that the system is model-bound. Agent 6 is not required.

---

## 20. Engine readiness gates

### 20.1 Reference-engine correctness gate — PASSED

`phase2_1_reference_1.1.0` remains frozen and authoritative.

### 20.2 Production-training readiness gate — PASSED

Phase 3 established:

- batched equivalence to the frozen reference engine;
- correct independent environment reset;
- correct persistent shared-memory transport;
- representative Metal inference benchmarks;
- measured worker/environment/batch scaling;
- exact compact trajectory reconstruction;
- successful two-hour production soak;
- no sustained swapping or unexplained memory growth;
- `R = 6.50`;
- backend decision `KEEP_PYTHON`.

The Python simulator is approved for later model/training integration.

A materially different final model must re-measure the throughput relationship before assuming the same headroom.

---

## 20A. Sampler legality regression

Phase 3 discovered a rare failure mode in the representative model's Gumbel-max sampler:

- illegal logits were masked with `-inf`;
- a boundary random value could create non-finite Gumbel noise;
- combining that noise with a masked logit could yield `NaN`;
- `argmax` could then select an illegal action.

Permanent validation requirements:

1. sampling noise used with masked logits must be finite;
2. the coordinator must verify every sampled action against the legal mask;
3. the engine must still reject any illegal action atomically;
4. regression tests must cover the original boundary mechanism.

A model-side sampler bug must never silently corrupt engine state.

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


## 22. Phase 4 evaluation-harness validation — PASSED

Phase 4 adds a separate validation layer above the engine.

### 22.1 Policy information-security gate

Accepted audit:

- 100,000 valid hidden-state permutation trials;
- 1,000,000 policy comparisons across all 10 catalogued policies;
- 800,000 full score-vector comparisons for the 8 scoring policies;
- 0 action mismatches;
- 0 diagnostic mismatches;
- 0 score-vector mismatches;
- 0 `PublicView` mismatches;
- 0 legal-action-list mismatches;
- 100,000 positive-control trials with 0 failures.

The audit splits positions between random-walk states and states reached by baseline play and spans multiple game phases. A hidden-information audit is valid only when the privileged state actually changed; unchanged permutations do not count.

This audit is a **standing regression**. Any newly registered evaluation policy must enter the policy catalogue and pass the same observer-information boundary before being trusted.

### 22.2 Policy legality gate

Final calibration league:

- 44,544 games;
- 22,272 paired units;
- 45 matchups;
- 0 illegal policy actions;
- 0 policy errors.

Policy failure must remain loud. No evaluator may substitute a legal move for a failed or illegal decision.

### 22.3 Reproducibility gate

A representative final subset was rerun:

- serially;
- with 2 workers;
- with 4 workers;
- with 8 workers;
- with 4 workers and shuffled match order.

All five runs produced:

- 1 distinct results digest;
- 1 distinct replay-digest set;
- 0 field-level mismatches.

Match identity must be assigned before worker dispatch and must never depend on worker index, schedule order, process identity, or wall-clock time.

### 22.4 Statistical gate

Primary metric:

\[
\mathrm{EWR}=\frac{W+0.5D}{N}.
\]

Headline uncertainty uses a 95% percentile bootstrap over the **paired evaluation unit**, not individual games.

For important/citable comparisons, the Phase 4 operating recommendation is:

- 256 paired units: screening only;
- 1,024 paired units: important model-selection or citable comparison.

Reason: policies are stochastic and policy seeds derive from match identity. At 256 paired units, behavior-identical policy versions could differ by several percentage points from seed realization alone; at 1,024 units the measured replica spread fell to about 0.011 or less.

Bradley-Terry/Elo-like ratings are secondary ranking aids and must not replace the direct paired interval when deciding whether policies are statistically distinguishable.

### 22.5 Strength-tier gate

Accepted four-tier core ladder:

```text
strategic_rule_based@1.1.0
    > tactical_rule_based@1.0.0
    > basic_heuristic@1.0.0
    > random_legal@1.0.0
```

All 6 cross-tier core comparisons separated in the correct direction.

The direct Strategic-vs-Tactical result was:

- Strategic effective win rate: 0.5354;
- 95% paired interval: [0.5168, 0.5540];
- 1,024 paired units.

Tier assignment must use direct paired comparisons and direction. A scalar league rating alone is insufficient.

### 22.6 Stress-policy gate

All 6 stress policies were behaviorally distinct from the Strategic baseline on multiple measured metrics, including attack rate, piece-type usage, game length, draw rate, reveal behavior, and movement/piece entropy.

Stress policies need not form a strength ladder. Their purpose is diagnostic coverage.

### 22.7 Phase 4 gate

```text
Phase 4 = COMPLETE
Evaluation harness ready for future checkpoints = YES
```
