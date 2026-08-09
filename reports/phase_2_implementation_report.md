# Phase Two Implementation Report — Python Reference Stratego Engine

Implementation version: `phase2_1_reference_1.1.0`
Rules version: `stratego_project_v1`
Observation version: `observation_v2_1_127ch` (supersedes `observation_v2_127ch`)
Report generated from the acceptance run recorded in `reports/phase_2_metrics.json`.

This report covers Phase Two and the two corrections applied in Phase Two Point
One. The Phase Two Point One section near the end summarises exactly what
changed; every other section already reflects the corrected engine.

---

## 24.1 Executive status

```text
PASS — Reference Engine Frozen for Phase Three
```

Every acceptance gate was met with zero unexplained mismatches:

| Gate | Target | Measured |
|---|---|---|
| Automated tests | all pass | 1,255 passed, 0 failed, 0 skipped |
| Combat matrix | exhaustive, 0 failures | 120 cases, 0 failures |
| Legal list vs mask | 0 discrepancies | 9,285 positions, 0 discrepancies |
| Mirror equivalence (all 127 channels) | >= 1,000 pairs, 0 mismatches | 1,804 pairs / 3,608 comparisons, 0 mismatches |
| Hidden-information anti-leak | >= 100,000 valid trials, 0 mismatches | 103,625 valid trials, 0 mismatches |
| Deterministic replay | >= 10,000 games, 0 mismatches | 10,000 games / 5,078,406 plies, 0 mismatches |
| Snapshot / restore | equivalence | 600 snapshots across 6 phases, 0 mismatches |
| State invariants under stress | 0 violations | 2,000 games / 1,045,111 transitions, 0 violations |

The two open items carried by the original Phase Two report are now closed:

1. Perspective normalization is exact across **all 127 channels**. The
   behavioural counterpart tie-break now orders candidates by normalized rather
   than absolute square index, published as observation version
   `observation_v2_1_127ch`.
2. Terminal-condition precedence is specified, implemented and tested. Genuine
   Stratego game-ending conditions now outrank the project's own training
   termination limits, and the ordering is recorded in
   `02_project_ruleset.md` section 9A.

The reference engine is frozen. Rule semantics, `observation_v2_1_127ch`, the
action encoding and replay semantics must not change without a new version
identifier and a differential comparison against this implementation.

---

## 24.2 Implementation inventory

### Files created

Engine (`stratego/`, 3,823 lines of implementation code):

| File | Lines | Responsibility |
|---|---:|---|
| `stratego/__init__.py` | 7 | package marker |
| `stratego/engine/__init__.py` | 106 | public API surface |
| `stratego/engine/constants.py` | 300 | geometry, inventory, terminal labels, `RulesConfig` |
| `stratego/engine/coordinates.py` | 213 | index/name conversion, adjacency, rays, perspective |
| `stratego/engine/pieces.py` | 171 | `PieceRecord`, stable identifiers |
| `stratego/engine/setup.py` | 173 | setup validation, generation, serialisation |
| `stratego/engine/state.py` | 300 | `GameState`, recent moves, behaviour memory, rendering |
| `stratego/engine/actions.py` | 67 | 10,000-entry action encoding |
| `stratego/engine/legal_moves.py` | 143 | move generation, mask, attack opportunities |
| `stratego/engine/combat.py` | 76 | exhaustive combat resolver |
| `stratego/engine/transition.py` | 289 | atomic transition, terminal evaluation, event emission |
| `stratego/engine/behavior.py` | 353 | the five behavioural events and threat relations |
| `stratego/engine/observation.py` | 432 | `observation_v2_1_127ch`, metadata, belief targets |
| `stratego/engine/events.py` | 302 | derived events, observer filtering, browser views |
| `stratego/engine/replay.py` | 138 | privileged replay record and reconstruction |
| `stratego/engine/snapshot.py` | 162 | compact snapshot / restore |
| `stratego/engine/invariants.py` | 323 | invariant checking |
| `stratego/engine/random_play.py` | 94 | seeded uniform random legal agent |
| `stratego/engine/permutation.py` | 174 | hidden-identity permutation and public-surface comparison |

Tests (`tests/`, 5,561 lines including 252 lines of shared fixtures):

| Package | Files | Lines |
|---|---:|---:|
| `tests/engine/` | 12 | 2,100 |
| `tests/observation/` | 9 | 2,154 |
| `tests/information_security/` | 4 | 644 |
| `tests/replay/` | 2 | 406 |
| `tests/helpers.py` | 1 | 252 |

Harness and reporting (`scripts/`, 1,462 lines):

- `scripts/run_phase2_validation.py` — the acceptance harness that produces `reports/phase_2_metrics.json`
- `scripts/manual_inspection_examples.py` — generates the section 24.16 examples

Other files: `requirements.txt`, `reports/phase_2_implementation_report.md`,
`reports/phase_2_metrics.json`.

### Files modified in Phase Two Point One

Source (3 files):

| File | Change |
|---|---|
| `stratego/engine/behavior.py` | counterpart ties broken by normalized square index; new `normalized_square_key` helper |
| `stratego/engine/transition.py` | terminal-condition precedence reordered |
| `stratego/engine/constants.py` | `OBSERVATION_VERSION` bumped, `SUPERSEDED_OBSERVATION_VERSIONS` added, `IMPLEMENTATION_VERSION` bumped |

`stratego/engine/observation.py` had only its module docstring updated; the
channel construction is untouched.

Tests (4 files) and harness (1 file):

| File | Change |
|---|---|
| `tests/observation/test_perspective.py` | full 127-channel mirror equality; scripted mirrored pairs per behaviour type; tie case now asserts orientation *independence* |
| `tests/engine/test_terminal_conditions.py` | six new precedence tests |
| `tests/helpers.py` | filler setup squares assigned in normalized order; `mirror_name`, `mirror_placements`, `mirror_move` |
| `tests/observation/__init__.py` | docstring version |
| `scripts/run_phase2_validation.py` | new mirror-equivalence stage and metrics |

Phase One documentation (9 files): listed individually in section 24.18.

### Environment

| Item | Value |
|---|---|
| Python | 3.13.2 (CPython, arm64) |
| Platform | macOS (Darwin 25.5.0), Apple M4 Pro, 14 cores, 48 GB unified memory |
| numpy | 2.5.1 |
| pytest | 9.1.1 (with pluggy 1.6.0, iniconfig 2.3.0) |
| psutil | 7.2.2 |

`numpy` is used for the observation tensor and the legal-action mask. `pytest`
runs the suite. `psutil` samples memory for the performance baseline. The engine
itself imports only `numpy` and the standard library.

### Structural differences from the instruction file list

Two additions and one relocation, all permitted by instruction section 3:

- `stratego/engine/permutation.py` was added. The hidden-identity permutation and
  public-surface comparison are needed by both the test suite and the acceptance
  harness, so they live in the package rather than being duplicated in each.
- `scripts/` was added for the acceptance harness and the example generator.
  Neither is part of the engine, so neither belongs in `stratego/engine/`.
- The Phase One documents were already located in `stratego_project_docs/` rather
  than at the repository root. They were left exactly where they were, as
  instruction section 3 forbids moving or renaming them; the new directories were
  created at the repository root alongside them.

---

## 24.3 Requirements traceability

| Requirement | Specification source | Implementation location | Test location | Status |
|---|---|---|---|---|
| 10x10 board, 92 occupiable, 8 lakes | `01` §1, `03` §4 | `constants.py`, `coordinates.py` | `tests/engine/test_geometry.py` | PASS |
| Square index / human name round trip | `03` §4 | `coordinates.py` | `tests/engine/test_geometry.py` | PASS |
| Player-relative transforms invert | instruction §5 step 1 | `coordinates.py::PERSPECTIVE_TABLES` | `tests/engine/test_geometry.py` | PASS |
| Exact 40-piece inventory | `01` §2, `04` §3 | `constants.py::PIECE_COUNTS` | `tests/engine/test_setup.py` | PASS |
| Stable type-independent piece identifier | `08` §4 | `pieces.py::make_piece_id` | `tests/engine/test_setup.py` | PASS |
| Setup validation, no silent repair | `02` §6, `03` §7 | `setup.py` | `tests/engine/test_setup.py` | PASS |
| Setup reflection / serialisation | `03` §7 | `setup.py` | `tests/engine/test_setup.py` | PASS |
| Full internal state contract | `08` §5-§10 | `state.py`, `pieces.py` | `tests/engine/test_transition.py`, `test_snapshot.py` | PASS |
| One-square cardinal movement | `01` §3, `04` §4 | `legal_moves.py` | `tests/engine/test_movement.py` | PASS |
| Flag and Bomb immobility | `01` §5, `04` §5 | `legal_moves.py`, `invariants.py` | `tests/engine/test_movement.py`, `test_invariants.py` | PASS |
| Scout ray movement | `01` §4, `04` §6 | `legal_moves.py`, `coordinates.py::RAYS` | `tests/engine/test_scout.py` | PASS |
| Scout multi-square revelation | `02` §8, `09` §5 | `transition.py` | `tests/engine/test_scout.py` | PASS |
| Action encoding `100*src+dst` | `03` §8 | `actions.py` | `tests/engine/test_action_encoding.py` | PASS |
| Legal list and mask agree exactly | `03` §9, `04` §13 | `legal_moves.py` | `tests/engine/test_action_encoding.py`, harness stage 2 | PASS |
| Exhaustive combat resolution | `01` §6-§7, `04` §7 | `combat.py` | `tests/engine/test_combat.py`, harness stage 1 | PASS |
| Post-combat occupancy | `01` §6 | `transition.py` | `tests/engine/test_combat.py` | PASS |
| Atomic transition, illegal action inert | `03` §10, instruction §9 | `transition.py::apply_action` | `tests/engine/test_transition.py` | PASS |
| Battleless counter semantics | `02` §3, instruction §10 | `transition.py` | `tests/engine/test_terminal_conditions.py` | PASS |
| Six terminal reason labels | `03` §11 | `constants.py`, `transition.py` | `tests/engine/test_terminal_conditions.py` | PASS |
| Terminal-condition precedence | `02` §9A | `transition.py::_evaluate_terminal` | `tests/engine/test_terminal_conditions.py` | PASS |
| Two-square rule excluded | `02` §2, `04` §11 | not implemented; `RulesConfig` rejects it | `tests/engine/test_rule_exclusions.py` | PASS |
| Continuous chasing excluded | `02` §2, `04` §11 | not implemented; `RulesConfig` rejects it | `tests/engine/test_rule_exclusions.py` | PASS |
| Knowledge monotonicity | `08` §5, `04` §21.3 | `pieces.py::set_known_to`, `invariants.py` | `tests/information_security/test_knowledge.py` | PASS |
| Five behavioural events | `06` §10, `08` §10 | `behavior.py` | `tests/observation/test_behavior_channels.py` | PASS |
| Active threat relation set | `08` §9, `04` §21.4 | `behavior.py::compute_threat_relations` | `tests/observation/test_behavior_channels.py` | PASS |
| Turn-start attack opportunities | `08` §11 | `legal_moves.py::adjacent_attack_opportunities` | `tests/observation/test_behavior_channels.py` | PASS |
| `observation_v2_1_127ch` shape and order | `06` §3-§13 | `observation.py` | `tests/observation/` (all files) | PASS |
| Channel metadata export | `07` §14, instruction §14 | `observation.py::observation_channel_metadata` | `tests/observation/test_shape_and_ranges.py` | PASS |
| Perspective normalization | `06` §3, `04` §12 | `coordinates.py`, `observation.py` | `tests/observation/test_perspective.py` | PASS |
| Normalized counterpart tie-break | `06` §10.6 | `behavior.py::normalized_square_key` | `tests/observation/test_perspective.py` | PASS |
| Mirror-equivalence acceptance suite | `07` §9A | `scripts/run_phase2_validation.py::mirror_stage` | `tests/observation/test_perspective.py`, harness stage 5 | PASS |
| Public event schema and ordering | `09` §5-§10, §17 | `events.py`, `transition.py` | `tests/replay/test_event_stream.py` | PASS |
| Observer-filtered board / setup view | `09` §11-§12 | `events.py` | `tests/information_security/test_browser_privacy.py` | PASS |
| Deterministic replay | `03` §12, `09` §13 | `replay.py` | `tests/replay/test_replay.py`, harness stage 3 | PASS |
| Snapshot / restore | `03` §13, `08` §15 | `snapshot.py` | `tests/engine/test_snapshot.py`, harness stage 6 | PASS |
| State invariants | `08` §18, instruction §20 | `invariants.py` | `tests/engine/test_invariants.py`, harness stage 5 | PASS |
| Hidden-information anti-leak | `07` §11, `09` §16 | `permutation.py` | `tests/information_security/test_anti_leak.py`, harness stage 4 | PASS |
| Belief targets kept separate | `03` §15, `08` §16 | `observation.py::belief_target` | `tests/information_security/test_belief_targets.py` | PASS |
| Seeded random test agent | `03` §18, instruction §19 | `random_play.py` | `tests/engine/test_determinism.py` | PASS |
| Performance instrumentation | `03` §20, `04` §19 | `scripts/run_phase2_validation.py` | harness stage 7 | PASS |

---

## 24.4 Automated test summary

| Measure | Value |
|---|---:|
| Total tests | 1,255 |
| Passed | 1,255 |
| Failed | 0 |
| Skipped | 0 |
| Expected failures | 0 |
| Errors | 0 |
| Execution time | 17.7 s |

There are no individual failures or skipped tests to list.

Distribution by package:

| Package | Focus |
|---|---|
| `tests/engine/` | geometry, setup, movement, Scout, combat, transitions, terminal conditions, rule exclusions, action encoding, invariants, snapshots, determinism |
| `tests/observation/` | all 127 channels, behaviour semantics, recent moves, global planes, perspective |
| `tests/information_security/` | anti-leak permutations, knowledge monotonicity, browser privacy, belief-target separation |
| `tests/replay/` | replay reconstruction, event content and ordering |

Reproduce with `python -m pytest -q` from the repository root.

---

## 24.5 Rule validation

| Rule area | Result | Evidence |
|---|---|---|
| Board geometry | PASS | 10x10, exactly 92 occupiable and 8 lake squares; lakes verified against the independently written list `c5 d5 g5 h5 c6 d6 g6 h6`; all 100 indices round-trip through `(row, column)` and through human notation |
| Setup inventory | PASS | Both players receive exactly 1 Flag, 1 Spy, 8 Scouts, 5 Miners, 4 Sergeants, 4 Lieutenants, 4 Captains, 3 Majors, 2 Colonels, 1 General, 1 Marshal, 6 Bombs; one rejection test per piece type, plus lake placement, out-of-area placement, missing square and wrong-length rejections |
| Normal movement | PASS | Four cardinal directions from interior, edge and corner squares; diagonal, two-square, off-board, onto-friendly and onto-lake moves all rejected; adjacent enemy attack legal |
| Flag movement | PASS | Zero legal actions for a Flag on every one of the 92 occupiable squares |
| Bomb movement | PASS | Zero legal actions for a Bomb on every one of the 92 occupiable squares; a Bomb that survives combat keeps `has_moved = False` and its starting square |
| Scout movement | PASS | Rays in all four directions, single and multi-square, stopping before a friendly piece, attacking the first opponent and not passing it, blocked by lakes, terminated by board edges, never diagonal |
| Combat | PASS | 120-case exhaustive matrix (section 24.6) plus post-combat occupancy through real transitions |
| Flag capture | PASS | Immediate termination with `flag_capture`, verified for both colours; results `+1 / -1` from the correct perspectives |
| No-legal-move victory | PASS | `opponent_no_legal_move` when the mover strands the opponent; observed 170 times in the 10,000-game sample |
| Draw conditions | PASS | `battleless_move_limit_draw` at exactly 100 (training) and 200 (evaluation); `absolute_move_limit_draw` at the configured total; `both_no_legal_move_draw` when neither side can move; combat resets the counter to 0 |
| Two-square rule excluded | PASS | 20 repetitions of an A-B-A-B shuffle stay legal; the sequence only ends through the battleless draw counter at ply 100. `RulesConfig` raises if the rule is requested |
| Continuous chasing excluded | PASS | A chase that recreates an identical position 12 times is never rejected. `RulesConfig` raises if the rule is requested |

### Terminal-condition precedence

Specified in `02_project_ruleset.md` section 9A, added in Phase Two Point One.
The engine applies, highest priority first:

1. `flag_capture`
2. `opponent_no_legal_move`
3. `both_no_legal_move_draw`
4. `battleless_move_limit_draw`
5. `absolute_move_limit_draw`

Genuine Stratego game-ending conditions outrank the project's own training
termination limits, which exist only to guarantee practical termination once the
two anti-repetition rules were removed.

Six tests in `tests/engine/test_terminal_conditions.py` pin the ordering:

| Collision | Expected | Result |
|---|---|---|
| Battleless threshold reached on the move that strands the opponent | `opponent_no_legal_move` | PASS |
| Absolute limit reached on the move that strands the opponent | `opponent_no_legal_move` | PASS |
| Absolute limit reached on the move that strands both players | `both_no_legal_move_draw` | PASS |
| Flag captured on the move that also reaches the absolute limit and strands the opponent | `flag_capture` | PASS |
| Draw limits reached while both players can still move | the matching draw reason | PASS |
| Battleless limit colliding with a capture or a mutual stalemate | structurally unreachable | PASS |

The last row is a proof by construction rather than an example. Any combat
resets the no-battle counter to zero, so a capture cannot coincide with
`battleless_move_limit_draw`. A player also cannot strand itself with a
non-combat move, because the square it just vacated is always available to move
back into, so `both_no_legal_move_draw` always follows a combat move and
therefore cannot coincide with the battleless limit either. Both consequences
are recorded in `02_project_ruleset.md` section 9A.

---

## 24.6 Combat matrix

| Measure | Value |
|---|---:|
| Attacker-defender combinations tested | 120 (10 movable attackers x 12 defender types) |
| Passing | 120 |
| Failing | 0 |

The expected outcomes are held as a literal table transcribed from
`01_official_rules.md` sections 6 and 7 in `tests/engine/test_combat.py`, not
recomputed from the resolver, so a logic error in `combat.py` cannot be mirrored
by the test. The same table drives harness stage 1.

Specific cases:

| Case | Expected | Result |
|---|---|---|
| Spy attacks Marshal | Spy survives | PASS |
| Marshal attacks Spy | Marshal survives | PASS |
| Miner attacks Bomb | Miner survives, Bomb removed | PASS |
| Every other movable piece attacks Bomb | attacker removed, Bomb survives | PASS (9 of 9) |
| Equal ranks | both removed | PASS (10 of 10) |
| Any movable piece attacks Flag | Flag captured, attacker wins the game | PASS (10 of 10) |
| Flag or Bomb as attacker | impossible; `CombatError` raised | PASS |

Post-combat occupancy was verified through real transitions for all three
outcomes: the winning attacker occupies the destination and vacates the source,
a winning defender keeps the destination while the source empties, and a tie
empties both squares.

---

## 24.7 Legal-action consistency

| Measure | Value |
|---|---:|
| Positions tested (harness) | 9,285 |
| List / mask discrepancies | 0 |
| Largest legal-action count observed | 30 |
| Non-terminal positions with an empty action list | 0 |

Three independent checks run at every position: the set of mask indices equals
the returned list, the mask population count equals the list length, and the
list is strictly ascending with no duplicates. A further 96 positions are
covered by `tests/engine/test_action_encoding.py`, which also verifies that all
10,000 identifiers round-trip and that a terminal state produces an all-zero
mask.

Edge cases encountered and handled: Scout rays terminating on lakes and board
edges; a Scout blocked by a friendly piece on the first ray square; positions
where the acting player's only movable pieces are pinned between lakes; terminal
positions, which return an empty list rather than raising.

---

## 24.8 Observation validation

| Property | Result |
|---|---|
| Output shape | exactly `(127, 10, 10)` |
| Data type | `numpy.float32` |
| Finiteness | all values finite in every sampled position |
| Binary planes | contain only `{0.0, 1.0}` |
| Coordinate planes (28-31) | remain in `[-1, +1]`, computed as `2i/9 - 1` |
| Recent-move planes (108-123) | contain only `{-1.0, 0.0, +1.0}` |
| Behavioural recency/rank/actor-knew | remain in `[0, 1]` |
| Behavioural special planes | contain only `{-1.0, 0.0, +1.0}` |
| Progress and inventory planes | remain in `[0, 1]` |
| Channel metadata | one entry per channel, 127 unique names, every declared range verified to bound real observations |
| Determinism | byte-identical across repeated construction of the same state |

Group-by-group results:

- **Identity planes 0-23 and hidden occupancy 24.** Each of the 12 own types
  marks exactly one plane at the piece's current square; the mark follows the
  piece and disappears on capture. Channel 24 and channels 12-23 are provably
  disjoint: their sum is exactly 1 on every opponent-occupied square.
- **Disclosure plane 25.** Verified in all three required states: never
  revealed (`0`), revealed by combat (`1`), logically identified as a Scout by a
  multi-square move (`1`).
- **Moved planes 26-27.** Zero before the first move, set on the first move,
  persistent afterwards, travelling with the piece and vanishing on capture.
  Flag and Bomb are never marked.
- **Origin planes 28-31.** Values equal `2i/9 - 1` for the normalized starting
  row and column, written at the piece's *current* square, unchanged by
  movement, and cleared on capture. Opponent origins are present for hidden
  pieces because origin tracking is public information about which physical
  piece is where, not about its type.
- **Setup memory 32-55.** The own block always contains exactly 40 ones, one per
  setup square, unchanged by movement or capture. The opponent block starts
  empty and gains exactly one entry at the piece's original square when its
  identity becomes legally known, whether through combat or a multi-square Scout
  move, and keeps it after later movement and capture.
- **Unresolved inventory 56-67.** Starts at `1.0` for all twelve types, drops to
  `(N-1)/N` on the first revelation and to `0.0` when every copy is known. Both
  critical distinctions hold: capturing an *already known* piece does not change
  the value, while capturing a hidden piece through combat reduces it exactly
  once.
- **Behavioural channels 68-107.** All five event definitions were tested with
  their documented positive case and every documented negative case, plus
  deterministic counterpart selection, recency decay at Δ ∈ {0, 8, 16, 32, 64,
  128} matching `1/(1+Δ/32)` exactly, rank encoding for Spy `0.1`, Miner `0.3`,
  Colonel `0.8`, Marshal `1.0`, special encoding Bomb `+1` and Flag `-1`, the
  historical actor-knew flag, and retrospective reinterpretation after a later
  legal reveal.
- **Recent-move planes 108-123.** Newest move always in channel 108, planes
  shifting by exactly one per ply, exactly one `-1` and one `+1` per plane, and
  moves older than sixteen plies dropped. A Scout ray marks only its endpoints.
- **Global planes 124-126.** Lake mask has exactly 8 ones on the correct squares
  and never changes. Game progress is spatially constant with `0 -> 0.0`,
  `1000 -> 0.25`, `2000 -> 0.5`, `4000 -> 1.0`. No-battle progress gives
  `0/50/99/100 -> 0.0/0.5/0.99/1.0` under training rules and
  `0/100/199/200 -> 0.0/0.5/0.995/1.0` under evaluation rules, resetting to
  `0.0` after any combat.

### Perspective normalization

Red normalizes to the identity and blue to a 180-degree rotation
(`square -> 99 - square`), which maps the lake mask onto itself and places each
observer's own four setup rows at the bottom of the board.

The gate builds a game and its colour-swapped, rotated twin and requires the two
players in equivalent roles to receive identical observations across **all 127
channels**, in both directions of the pairing.

| Measure | Value |
|---|---:|
| Mirrored position pairs tested | 1,804 |
| Observation comparisons (two observer pairings per pair) | 3,608 |
| Channels compared | 127 of 127 |
| **Mismatches** | **0** |
| Pairs whose history contained more than one eligible counterpart | 1,557 |
| Multi-counterpart events observed during the walks | 5,395 |

Behavioural coverage of the sampled positions, counted as live behaviour records
present at a compared checkpoint:

| Behaviour | Occurrences |
|---|---:|
| threat | 6,635 |
| evade | 709 |
| declined attack | 10,738 |
| protect | 953 |
| was protected | 815 |

All five behaviour types appeared, so the gate's coverage requirement from
`07_observation_validation_matrix.md` section 9A is satisfied by the randomized
sample alone. Six scripted mirrored pairs in
`tests/observation/test_perspective.py` additionally guarantee coverage of each
behaviour type and of a genuine counterpart tie regardless of the random draw.

Under the superseded `observation_v2_127ch` this gate could not be met: the same
measurement over 60 mirrored pairs found differences in 22 of them, always
inside channels 68-107. Section "Phase Two Point One Changes" records the fix.

Legal actions map correctly between mirrored games: the mirrored image of one
game's legal set equals the other's, and both normalize to the same set in the
acting player's own frame.

---

## 24.9 Hidden-information leak report

| Measure | Value |
|---|---:|
| Permutation trials attempted (including rejected shuffles) | 363,461 |
| Valid trials | 103,625 |
| Trials that actually changed at least one hidden identity | 103,625 |
| Distinct sampled positions | 4,145 |
| Observation mismatches | **0** |
| Legal-action mismatches | **0** |
| Public-event mismatches | **0** |
| Browser / public-view mismatches (board view and setup view) | **0** |
| Belief-target positive-control checks | 103,625 |
| Belief-target control failures (a changed permutation that left the target unchanged) | **0** |

Acceptance target `>= 100,000 valid trials, 0 unexplained public-information
mismatches` is met.

### Method

Positions are drawn from seeded random games stopped at plies 8, 20, 35, 55, 80,
110, 150, 200 and 260. For each position the acting player is the observer. A
trial clones the privileged state and permutes the true types of the opponent
pieces that are alive and not legally known to the observer.

A permutation is valid only if it preserves every publicly deducible constraint.
Exactly one such constraint exists in this ruleset: a piece that has moved cannot
be a Flag or a Bomb, because both are immovable and movement is public. Piece
counts are preserved automatically because the permutation only rearranges the
hidden pieces' own types. Uniform shuffles are tried first and rejected when they
violate the constraint, which is why 363,461 shuffles were needed to obtain
103,625 accepted assignments; a constructive sampler is used as a fallback in
late-game positions where most hidden pieces have moved.

The four products compared per trial are exactly those listed in
`09_public_event_and_replay_schema.md` section 16: all 127 observation channels,
the observer's legal-action set, the observer-filtered board and setup views, and
the observer-filtered public event stream.

### Positive control

Privileged belief targets are *expected* to differ, and they did in all 103,625
changed trials. The targets keep the same piece identifiers and squares while
the true types move, which confirms the permutation actually altered hidden
state rather than being a no-op.

### Targeted tests

Beyond the randomized gate, `tests/information_security/test_anti_leak.py`
verifies the permutation machinery itself (multiset preservation, the
moved-piece constraint, own and known pieces left untouched), runs a
channel-by-channel comparison, and repeats the differential test once per
behaviour type using scripted positions that guarantee a live event of that
type.

---

## 24.10 Replay report

| Measure | Value |
|---|---:|
| Complete games generated | 10,000 |
| Complete games replayed | 10,000 |
| Total plies reconstructed | 5,078,406 |
| Board-state mismatches | **0** |
| Observation mismatches | **0** |
| Event mismatches | **0** |
| Terminal-result mismatches | **0** |

Acceptance target `>= 10,000 complete games, 0 unexplained mismatches` is met.

### Method

Each game is played once with the seeded random agent while a digest is captured
after every ply. The digest covers the board array, every piece record's square,
alive, moved, knowledge and reveal-reason fields, the acting player, both
counters, the terminal fields, the recent-move window, the active threat
relations, the behaviour memory, the legal-action list, the legal-action mask,
**both players' 127-channel observations** and that ply's events.

The game is then reconstructed from the replay record alone — rules version,
observation version, both setups, first player and the ordered action list — and
the digests are recomputed. Comparing the live pass against the replay pass, not
one replay against another, is what makes the check meaningful. All 5,078,406
plies matched.

The final states were additionally compared by full state fingerprint, event
list equality and observation equality for both players.

---

## 24.11 Snapshot/restore report

| Measure | Value |
|---|---:|
| Snapshots tested | 600 |
| Legal-action mismatches after restore | 0 |
| Observation mismatches (both observers) | 0 |
| Public-event mismatches | 0 |
| Public-view mismatches | 0 |
| Next-transition mismatches | 0 |
| Full state-fingerprint mismatches | 0 |

Coverage by required phase:

| Phase | Snapshots |
|---|---:|
| Early game | 119 |
| Middle game | 106 |
| Late game | 89 |
| Immediately before combat | 120 |
| Immediately after combat | 119 |
| Near the battleless draw limit | 47 |

Each snapshot is restored and then checked for: identical full state
fingerprint; identical legal-action list; identical 127-channel observation for
*both* players; identical observer-filtered board view and public event stream
for both players; and identical events generated by applying the same next
action to the original and the restored state.

`tests/engine/test_snapshot.py` adds the completeness case required by
`04_engine_validation_plan.md` section 21.5 — a state carrying recent moves,
known and hidden pieces, behaviour records of several types and a nonzero
battleless counter — plus a compact snapshot that deliberately omits the
long-form event log, confirming that the reduced contents permitted by
`08_internal_state_spec.md` section 15 still reproduce legality and observations
exactly.

---

## 24.12 State-invariant stress report

| Measure | Value |
|---|---:|
| Complete games checked | 2,000 |
| Total state transitions checked | 1,045,111 |
| Invariant violations | **0** |

Invariants are checked after every single transition, with the immutability
baseline and the previous knowledge snapshot supplied so the immutability and
monotonicity clauses are active. The checker verifies:

40 piece records per player and 80 in total; identifier equals table index;
owner consistent with the identifier; starting square consistent with the setup
slot; every live piece on exactly one occupiable non-lake square; captured
pieces on no square and carrying a capture ply; board and piece records agreeing
bidirectionally; no square holding two pieces; lake squares empty; every player
knowing all of their own identities; reveal reasons limited to `own_piece`,
`combat` and `scout_multisquare`; knowledge never reverting; Flag and Bomb never
marked moved and never leaving their starting square; counters non-negative with
the battleless count bounded by the total; a valid acting player; a valid and
internally consistent terminal state; threat relations crossing players and
belonging to the current ply; and behaviour-memory keys matching their events.

Sixteen negative-control tests in `tests/engine/test_invariants.py` deliberately
corrupt a state and confirm the checker names the correct violated invariant, so
the zero-violation result is not simply a checker that never fires.

---

## 24.13 Random-game statistics

Measured over the same 10,000 games used for the replay gate (training rules:
100 battleless moves, 4,000 absolute limit; uniform random legal play).

| Measure | Value |
|---|---:|
| Games | 10,000 |
| Mean moves | 507.8 |
| Median moves | 482 |
| Minimum moves | 1 |
| Maximum moves | 1,860 |
| Red wins | 2,831 (28.31 %) |
| Blue wins | 2,846 (28.46 %) |
| Draws | 4,323 (43.23 %) |

Terminal reasons:

| Reason | Count | Share |
|---|---:|---:|
| `flag_capture` | 5,507 | 55.07 % |
| `battleless_move_limit_draw` | 4,322 | 43.22 % |
| `opponent_no_legal_move` | 170 | 1.70 % |
| `both_no_legal_move_draw` | 1 | 0.01 % |
| `absolute_move_limit_draw` | 0 | 0.00 % |

Reading of these diagnostics:

- Red and blue win rates differ by 0.15 percentage points over 10,000 games,
  which is well inside sampling noise. There is no first-player artefact in the
  engine under random play.
- The one-move minimum is legitimate rather than a bug: red's front rank is row
  4 and blue's is row 7, so a red Scout on row 4 can reach row 7 in a single ray
  move and capture a Flag that happens to be placed there.
- `absolute_move_limit_draw` never triggered, which matches the Ataraxos
  observation quoted in `02_project_ruleset.md` section 4 that the 4,000-move
  safeguard almost never fires alongside the 100-move battleless rule. The label
  is nonetheless implemented and tested with a reduced limit.
- `both_no_legal_move_draw` is very rare (1 game in 10,000) because it requires
  the mover to strand both sides at once; it is covered by a deterministic unit
  test as well.

---

## 24.14 Performance baseline

Measured single-threaded on the target machine (Apple M4 Pro Mac mini, 14 cores,
48 GB, macOS 25.5.0, CPython 3.13.2). The acceptance harness uses multiple
processes to shorten wall-clock time, but all throughput figures below are
single-threaded.

| Measurement | Result |
|---|---:|
| Legal-action generations per second | 134,067 |
| Legal-action masks per second | 115,980 |
| State transitions per second | 73,269 |
| Observations generated per second | 26,975 |
| Snapshots per second | 66,184 |
| Complete random games per second | 121.9 |
| Random-game plies per second | 58,118 |
| Mean memory use | 49.9 MB |
| Peak memory use | 155.3 MB |

The two Phase Two Point One changes are performance neutral within measurement
noise: the tie-break replaced one comparison key with a table lookup, and the
terminal check reordered existing predicates.

Memory figures describe the coordinating process, in which the performance and
storage stages run, so they reflect what one engine instance costs. The harness
also records `children_rusage_maxrss_bytes` (1.41 GB for a 12-worker run) as a
separate diagnostic; `RUSAGE_CHILDREN` is platform dependent and macOS appears
to aggregate across reaped children rather than reporting the largest one, so
that value should be read as an upper bound on total worker memory — roughly
117 MB per worker — and not as any single process's peak.

Estimated share of engine time by component, from the same measurements:

| Component | Share |
|---|---:|
| Observation construction | 50.5 % |
| Behavioural processing | 20.8 % |
| State transition (excluding behaviour) | 18.6 % |
| Legal-action generation | 10.2 % |

Behavioural processing was timed directly by calling
`capture_pre_move_context`, `compute_threat_relations` and
`build_behavior_events` on sampled positions; its cost is a subset of the
transition cost in real play, so the two rows overlap and the split should be
read as an estimate rather than an exact partition.

Projection for Phase Three planning, using `04_engine_validation_plan.md`
section 19's formula and the single-threaded transition rate:

```text
73,269 transitions/second x 604,800 seconds = 4.43 x 10^10 transitions in 168 hours
```

That figure is engine-only and ignores neural-network inference, which will
dominate self-play. A self-play step also needs one observation, so the
observation rate of 26,975 per second is the more relevant single-core bound;
across 12 usable cores that is roughly 324,000 observations per second before
any optimization. No batch wrapper exists yet, so batch-size scaling was not
measured; that belongs to Phase Three.

No engine change was made to improve these numbers.

---

## 24.15 Storage baseline

| Item | Size |
|---|---:|
| One snapshot (JSON, compact separators) | 7,066 bytes |
| One typical complete replay (JSON) | 3,272 bytes mean, 3,290 bytes median |
| 1,000 typical replays | 3.27 MB |
| Estimated 1,000,000 games | 3.27 GB |

Replay records store only versions, both setups as 40-character strings, the
action list and the terminal summary, so size scales with game length rather
than with state size. A million games therefore fits comfortably on the 1 TB
external drive described in `05_project_plan.md` section 12, leaving the budget
free for checkpoints and evaluation artefacts. Storing full observation tensors
instead would cost 50.8 kB per ply, about 25.8 MB per game at the measured mean
length of 508 plies, or roughly 26 TB for a million games. That is why
reconstruction from the compact record is the right default, exactly as
`09_public_event_and_replay_schema.md` section 19 recommends.

---

## 24.16 Manual inspection examples

Generated by `python scripts/manual_inspection_examples.py`. Board codes:
`r`/`b` prefix for the owner, `1`-`9` for Marshal down to Scout, `S` Spy, `F`
Flag, `B` Bomb, `?` unresolved opponent piece, `~~` lake, `.` empty.

### Example 1 — Ordinary movement

```text
     a  b  c  d  e  f  g  h  i  j
 10  .  .  .  .  .  .  .  .  . bF
  7  .  .  .  . b7  .  .  .  .  .
  6  .  . ~~ ~~  .  . ~~ ~~  .  .
  5  .  . ~~ ~~  .  . ~~ ~~  .  .
  3  .  .  .  . r5  .  .  .  .  .
  1 rF  .  .  .  .  .  .  .  .  .
```

- Acting player: red
- Action: `e3->e4`, one square, no combat
- Expected: the captain occupies e4, its moved flag is set, the battleless
  counter becomes 1, and only a `move` event is emitted
- Actual: captain on e4; total moves 1, battleless moves 1; events `['move']`
- Observation changes (red's frame): channel 5 (own captain) e3 `1 -> 0` and e4
  `0 -> 1`; channel 26 (own moved) e4 `0 -> 1`; channel 108 (last move) e3 `-1`
  and e4 `+1`

### Example 2 — Scout revelation by a multi-square move

Blue's view before, then after:

```text
     a  b  c  d  e  f  g  h  i  j            a  b  c  d  e  f  g  h  i  j
 10 b7  .  .  .  .  .  .  .  . bF        10 b7  .  .  .  .  .  .  .  . bF
  4  .  .  .  .  .  .  .  .  .  .   ->    4 r9  .  .  .  .  .  .  .  .  .
  1 r?  .  .  .  .  .  .  .  . r?         1  .  .  .  .  .  .  .  .  . r?
```

- Acting player: red
- Action: `a1->a4`, four squares along the a-file
- Expected: only a Scout can move more than one square, so blue legally learns
  the type and an `identity_reveal` event with reason `scout_multisquare` is
  emitted
- Actual: known to blue `True` with reason `scout_multisquare`; events
  `['move', 'identity_reveal']`; reveal event names piece `red:00`, type
  `scout`, newly known to `['blue']`
- Observation changes: blue channel 13 (known opponent Scout) at a4 becomes `1`;
  blue channel 24 (hidden opponent) at a4 becomes `0`; red channel 25 (own piece
  known to opponent) at a4 becomes `1`

### Example 3 — Combat

Red's view before, then after:

```text
     a  b  c  d  e  f  g  h  i  j            a  b  c  d  e  f  g  h  i  j
 10  .  .  .  .  .  .  .  .  . b?        10  .  .  .  .  .  .  .  .  . b?
  4  .  .  .  . b?  .  .  .  .  .   ->    4  .  .  .  . b1  .  .  .  .  .
  3  .  .  .  . r5  .  .  .  .  .         3  .  .  .  .  .  .  .  .  .  .
  1 rF  .  .  .  .  .  .  .  .  .         1 rF  .  .  .  .  .  .  .  .  .
```

- Acting player: red
- Action: `e3->e4`, the captain attacks a hidden piece that turns out to be the
  Marshal
- Expected: the defender wins and keeps e4, the attacker is removed, both
  identities become public, and the battleless counter resets to 0
- Actual: attacker alive `False`, defender alive `True`; combat event
  `captain vs marshal -> defender_survives`; battleless moves `42 -> 0`; events
  `['move', 'identity_reveal', 'identity_reveal', 'combat']`
- Observation changes (red's frame): channel 21 (known opponent Marshal) at e4
  becomes `1`; channel 24 at e4 becomes `0`; unresolved Marshal inventory
  (channel 65) becomes `0.000` because the single Marshal is now identified

### Example 4 — Hidden-piece observation

Privileged view, then red's view of the same position:

```text
     a  b  c  d  e  f  g  h  i  j            a  b  c  d  e  f  g  h  i  j
 10  .  .  .  .  .  .  .  .  . bF        10  .  .  .  .  .  .  .  .  . b?
  6  .  . ~~ ~~ b1 bS ~~ ~~  .  .   ->    6  .  . ~~ ~~ b? b? ~~ ~~  .  .
  3  .  .  .  . r5  .  .  .  .  .         3  .  .  .  . r5  .  .  .  .  .
  1 rF  .  .  .  .  .  .  .  .  .         1 rF  .  .  .  .  .  .  .  .  .
```

- Acting player: red
- Action: none; this example inspects a static position's observation
- Expected: red sees generic occupancy on e6 and f6 and no type plane, even
  though the privileged state holds a Marshal and a Spy there
- Actual: e6 and f6 both have hidden-occupancy channel 24 `= 1` and a sum of
  `0` across known-identity channels 12-23
- Privileged belief target, which is a training label and never an observation
  input:
  `[{'piece_id': 'blue:00', 'square': 54, 'true_type': 'marshal'}, {'piece_id': 'blue:01', 'square': 55, 'true_type': 'spy'}, {'piece_id': 'blue:02', 'square': 99, 'true_type': 'flag'}]`

### Example 5 — Behavioural event tracking

Red's view before, then after:

```text
     a  b  c  d  e  f  g  h  i  j            a  b  c  d  e  f  g  h  i  j
  9  .  .  .  .  .  .  .  .  . b?         9  .  .  .  .  .  .  .  .  . b?
  5  .  . ~~ ~~ b7  . ~~ ~~  .  .   ->    5  .  . ~~ ~~  .  . ~~ ~~  .  .
  4  .  .  .  .  .  .  .  .  .  .         4  .  .  .  . b7  .  .  .  .  .
  3  .  . r8  . r5  .  .  .  .  .         3  .  .  . r8 r5  .  .  .  .  .
  1 rF  .  .  .  .  .  .  .  .  .         1 rF  .  .  .  .  .  .  .  .  .
```

- Acting player: blue, then red
- Actions: `e5->e4`, blue threatens the red captain; then `c3->d3`, red's miner
  moves newly adjacent to the threatened captain
- Expected: blue's sergeant records a `threat`; red's miner records a `protect`
  with the captain as counterpart and the threatener as context; the captain
  records a `was_protected`
- Actual:
  - threat event: actor `blue:00`, counterpart `red:02`, ply 1,
    `actor_knew_counterpart_type` `False`
  - protect: actor `red:01`, counterpart `red:02`, context `blue:00`, ply 2,
    actor knew counterpart `True`
  - was_protected: actor `red:02`, counterpart `red:01`, ply 2
- Observation changes (red's frame):
  - protect block channels 80-83 at d3: recency `1.000`, rank `0.600` (captain,
    6/10), actor-knew `1`, special `0`
  - was-protected block channels 84-87 at e3: recency `1.000`, rank `0.300`
    (miner, 3/10), actor-knew `1`
  - opponent threat block channel 88 at e4: recency `0.970`, which is
    `1/(1 + 1/32)` for a one-ply-old event; rank `0.000` and actor-knew `0`,
    because blue did not know the red captain's identity when the threat
    occurred. Exposing a counterpart rank needs both halves of the
    `06_observation_v2_127ch.md` section 9.3 rule, and this case fails the first
    half even though red obviously knows its own captain.

---

## 24.17 Known limitations

1. **The `evade` negative case "P remains adjacent to A" is unreachable.** One
   move creates threat relations from exactly one piece, and with orthogonal-only
   movement every square adjacent to the threatener other than the threatened
   square is a diagonal or blocked step away. So any legal non-attack move by a
   threatened piece necessarily breaks the adjacency. The test suite asserts this
   property by enumeration rather than pretending to construct the case.

2. **No batch simulation wrapper.** `03_game_engine_spec.md` section 16 describes
   a batch interface as a later requirement and instruction section 23 asks for
   batch measurements only "if a simple batch wrapper exists". None was built, so
   batch-size scaling is unmeasured. This is Phase Three work.

3. **No long-run stability soak.** `04_engine_validation_plan.md` section 17 asks
   for several hours of continuous batched play. The acceptance run applies more
   than sixteen million transitions across all stages — the replay gate alone
   plays each of its 10,000 games once and reconstructs it twice — without a
   memory growth trend or a single invariant violation, but a multi-hour soak
   has not been performed.

4. **Performance figures are single-machine and single-threaded.** They were
   measured on the target Mac mini but on an otherwise interactive desktop
   session, so they should be treated as approximate. The component time-share
   split overlaps between "behavioural processing" and "state transition" because
   the former is a subset of the latter.

5. **The random agent is a validation tool, not a baseline opponent.** Its
   statistics describe engine behaviour under random play and say nothing about
   playing strength. Phase Four owns the baseline agents.

6. **Game creation assumes the first player has a legal move.** Terminal
   conditions are only evaluated after a move, so a hypothetical start position
   in which the first player cannot move would not be detected as terminal. This
   is unreachable with legal setups: at most 7 of a player's pieces are
   immovable, the front rank holds 10 pieces and the rows in front of it are
   empty at game start, so at least 3 legal moves always exist. The random agent
   raises a `RuntimeError` rather than looping if this assumption is ever
   violated.

7. **Anti-leak coverage is broad but not exhaustive.** 103,625 trials across
   4,145 positions is a sampling argument, not a proof. The structural guard in
   `tests/information_security/test_belief_targets.py` — that
   `build_observation` never calls `belief_target` — plus the guarded
   `true_type` access pattern in `observation.py` are what make the sampling
   argument credible.

---

## 24.18 Specification deviations

```text
No known specification deviations.
```

The implementation matches the Phase One documents as they now stand. Phase Two
Point One changed the documents rather than deviating from them, so every
modification is listed below.

### Phase One documents modified in Phase Two Point One

| Document | Change | Approved |
|---|---|---|
| `README.md` | status raised to 0.3; notes the new observation identifier and the section 9A addition | Yes — Phase 2.1 instruction 1 and 2 |
| `02_project_ruleset.md` | new section 9A specifying terminal-condition precedence and its two unreachable collisions | Yes — Phase 2.1 instruction 2 |
| `03_game_engine_spec.md` | observation identifier updated in sections 14 and 23 | Yes — Phase 2.1 instruction 1 |
| `04_engine_validation_plan.md` | observation identifier updated in sections 8A and 21.5; section 21.4 now requires the normalized tie-break and mirror equivalence | Yes — Phase 2.1 instruction 1 |
| `05_project_plan.md` | observation identifier updated in sections 6, 13 (Phase 1) and 13 (Phase 5) | Yes — Phase 2.1 instruction 1 |
| `06_observation_v2_127ch.md` | now defines `observation_v2_1_127ch`; new section 1.1 version history and new section 10.6 specifying normalized counterpart selection; sections 10.1, 10.3 and 10.4 point at 10.6; acceptance clause updated | Yes — Phase 2.1 instruction 1 |
| `07_observation_validation_matrix.md` | observation identifier updated; sections 8.1 and 8.3 require the normalized tie-break; new section 9A defines the mirror-equivalence acceptance suite | Yes — Phase 2.1 instruction 1 |
| `08_internal_state_spec.md` | observation identifier updated in sections 1, 2 and 19; section 11 requires the normalized tie-break | Yes — Phase 2.1 instruction 1 |
| `09_public_event_and_replay_schema.md` | observation identifier updated in section 16 | Yes — Phase 2.1 instruction 1 |

The file `06_observation_v2_127ch.md` keeps its name even though it now defines
`observation_v2_1_127ch`. Renaming it would break the cross-references in the
eight other Phase One documents, and the original Phase Two instruction
forbids renaming Phase One files. The mismatch is called out in a note at the
top of the document and in the README.

Neither change requires a new `rules_version`. The observation change is
carried by the new observation identifier. The terminal precedence was never
stated in `stratego_project_v1`, so section 9A specifies previously unspecified
behaviour rather than altering a stated rule; section 9A itself records that any
*future* change to the ordering does require a new rules version.

### Remaining interpretations

Two points still required interpretation where the documents are silent rather
than contradictory. Each is implemented explicitly and documented in code.

| Interpretation | Where the documents are silent | Choice made | Reference |
|---|---|---|---|
| Ply numbering for recency | `06` §9.2 requires recency `1.0` "immediately after the event" but does not fix the ply origin | a completed move is stamped with the ply number it produces, so `Δ = total_moves - event_ply` is `0` right after the event, reproducing the documented table exactly | `state.py` module docstring |
| Meaning of "newly threatened" in the relation set | `08` §9 says "all opponent pieces newly threatened by that move" while `06` §10.1 defines a threat purely by post-move adjacency | the relation set uses the section 10.1 condition without an extra "was not already adjacent" filter, so it is always a superset of the recorded threat counterpart | `behavior.py::compute_threat_relations` |

Two further mechanical choices worth noting, neither of which changes observable
behaviour:

- The canonical piece identifier is stored as the integer `owner * 40 + slot`
  rather than the string `red:07`. `08_internal_state_spec.md` section 4 calls
  `(owner, setup_slot_index)` a *recommended conceptual form*; the packing is
  positional and encodes nothing about type. The string form is produced by
  `piece_id_name` for every event and public view.
- Channel 125 divides by `rules.absolute_move_limit` rather than a hardcoded
  4,000. Both documented configurations use 4,000, so the channel is numerically
  identical to the specification in every project configuration.

---

## 24.19 Open questions

The two questions carried by the original Phase Two report are now **closed**:
the counterpart tie-break was changed to the normalized index and published as
`observation_v2_1_127ch`, and the terminal-condition precedence was set and
recorded in `02_project_ruleset.md` section 9A. The remaining questions are all
Phase Three scoping decisions.

1. **Is a batch wrapper wanted in Phase Three, and at what shape?**
   `03_game_engine_spec.md` section 16 lists the conceptual operations but not
   the array layout or the reset policy. This determines whether the optimized
   backend decision is made against per-game or batched throughput.

2. **What throughput does the training coordinator actually need?** The Phase
   Three optimization decision needs a target, not just a measurement. The
   figures in section 24.14 are the input; the required rate depends on the model
   size chosen in Phase Six.

3. **Should the reference engine keep failing loudly in production?** The engine
   currently raises on illegal setups, illegal actions and post-terminal
   transitions, as `03_game_engine_spec.md` section 19 requires for development.
   `03` section 19 also anticipates a "controlled error-recovery policy" in the
   training coordinator. That policy is a Phase Three decision and was not
   invented here.

---

## 24.20 Acceptance checklist

```text
[x] All rule tests pass
[x] Complete combat matrix passes
[x] Legal-action list and mask agree
[x] 127-channel observation contract passes
[x] Behavioral representation passes
[x] Perspective normalization passes
[x] 100,000+ hidden-information anti-leak trials pass
[x] 10,000+ complete replay reconstructions pass
[x] Snapshot/restore passes
[x] State invariants pass under stress
[x] Deterministic seeded execution passes
[x] No unexplained specification deviations
[x] Performance baseline recorded
[x] Storage baseline recorded
```

Evidence for each item:

| Item | Evidence |
|---|---|
| All rule tests pass | section 24.4 (1,255 passed, 0 failed) and section 24.5 |
| Complete combat matrix passes | section 24.6 (120 of 120) |
| Legal-action list and mask agree | section 24.7 (9,285 positions, 0 discrepancies) |
| 127-channel observation contract passes | section 24.8 |
| Behavioral representation passes | section 24.8 behavioural paragraph and section 24.16 example 5 |
| Perspective normalization passes | section 24.8 (1,804 mirrored pairs, all 127 channels, 0 mismatches) |
| 100,000+ anti-leak trials pass | section 24.9 (103,625 valid trials, 0 mismatches) |
| 10,000+ replay reconstructions pass | section 24.10 (10,000 games, 5,078,406 plies, 0 mismatches) |
| Snapshot/restore passes | section 24.11 (600 snapshots across all six phases, 0 mismatches) |
| State invariants pass under stress | section 24.12 (1,045,111 transitions, 0 violations) |
| Deterministic seeded execution passes | `tests/engine/test_determinism.py`; every harness stage is seeded and worker-count independent |
| No unexplained specification deviations | section 24.18 |
| Performance baseline recorded | section 24.14 |
| Storage baseline recorded | section 24.15 |

---

## Phase Two Point One Changes

Two narrowly scoped specification corrections were applied to the validated
Phase Two engine, after which the whole acceptance suite was rerun and the
engine was frozen.

### Change 1 — behavioural counterpart tie-break

| Item | Value |
|---|---|
| Previous observation version | `observation_v2_127ch` |
| New observation version | `observation_v2_1_127ch` |
| Previous rule | eligible counterparts ordered by **absolute** board-square index |
| New rule | eligible counterparts ordered by square index **after normalization into the acting player's perspective** (`06_observation_v2_127ch.md` section 10.6) |
| Channels affected | none in number, order, range or meaning; only which counterpart is selected when several qualify |
| Red-to-move behaviour | unchanged, because red's normalization is the identity |

The absolute index is not preserved by the 180-degree rotation used for blue, so
under the old rule a position and its colour-swapped mirror could select
non-equivalent counterparts and behavioural channels 68-107 were not mirror
images. Since one network is intended to play both colours, that asymmetry was a
real defect in the representation rather than a cosmetic one.

The rule is applied at all four selection points in `behavior.py`: the `threat`
counterpart, the `evade` counterpart, the `declined_attack` counterpart and the
`protect` / `was_protected` counterpart.

Mirror-equivalence acceptance result:

| Measure | Before (`observation_v2_127ch`) | After (`observation_v2_1_127ch`) |
|---|---|---|
| Mirrored position pairs tested | 60 | 1,804 |
| Observation comparisons | 60 | 3,608 |
| Channels compared per comparison | 127 | 127 |
| Pairs with a mismatch | 22 | **0** |
| Channels that ever mismatched | 18, all in 68-107 | **none** |
| Pairs whose history contained multiple eligible counterparts | not measured | 1,557 |

Behaviour coverage of the sampled positions: threat 6,635, evade 709, declined
attack 10,738, protect 953, was protected 815 — all five present, plus 5,395
multi-counterpart events. Six scripted mirrored pairs in
`tests/observation/test_perspective.py` guarantee the same coverage
independently of the random draw.

### Change 2 — terminal-condition precedence

| Item | Value |
|---|---|
| Previous order | flag capture, battleless draw, absolute draw, no-legal-move outcomes |
| New order | flag capture, `opponent_no_legal_move`, `both_no_legal_move_draw`, `battleless_move_limit_draw`, `absolute_move_limit_draw` |
| Rationale | genuine Stratego game-ending conditions outrank the project's own training termination limits |
| Recorded in | `02_project_ruleset.md` section 9A |
| Rules version | unchanged (`stratego_project_v1`); section 9A specifies previously unspecified behaviour rather than altering a stated rule |

The reordering did not change the outcome of a single game in the 10,000-game
replay set: the terminal-reason distribution is byte-identical to the Phase Two
run. That is expected, because a collision requires one move to both reach a
draw threshold and settle the mobility question, and two of the four possible
collisions are structurally impossible (section 24.5).

### New tests added

| File | Tests added |
|---|---|
| `tests/engine/test_terminal_conditions.py` | `test_no_legal_move_victory_outranks_the_battleless_draw`, `test_no_legal_move_victory_outranks_the_absolute_move_limit_draw`, `test_mutual_stalemate_outranks_the_absolute_move_limit_draw`, `test_flag_capture_outranks_every_other_condition`, `test_draw_limits_still_apply_when_both_players_can_move`, `test_battleless_limit_can_never_coincide_with_a_capture_or_stalemate` |
| `tests/observation/test_perspective.py` | `test_mirrored_games_stay_identical_across_all_127_channels` (replaces the channel-restricted version), `test_mirrored_games_agree_at_every_ply`, `test_behaviour_planes_transform_consistently`, `test_scripted_mirrored_positions_match_on_all_channels` (6 parametrised cases), `test_scripted_tie_case_really_has_multiple_eligible_counterparts`, and `test_behaviour_counterpart_selection_is_orientation_independent` (replaces the test that pinned the old orientation-dependent behaviour) |
| `scripts/run_phase2_validation.py` | new mirror-equivalence stage feeding `mirror_*` metrics |

Test count moved from 1,250 to 1,255.

### Complete regression results after the changes

| Gate | Target | Result |
|---|---|---:|
| Automated tests | all pass | 1,255 passed, 0 failed, 0 skipped |
| Combat matrix | 0 failures | 120 cases, 0 failures |
| Legal-action list vs mask | 0 discrepancies | 9,285 positions, 0 |
| Mirror equivalence, all 127 channels | >= 1,000 pairs, 0 mismatches | 1,804 pairs, 0 |
| Hidden-information anti-leak | >= 100,000 valid trials, 0 mismatches | 103,625 valid trials (363,461 attempted, 4,145 positions), 0 |
| Deterministic replay | >= 10,000 games, 0 mismatches | 10,000 games, 5,078,406 plies, 0 |
| Snapshot / restore | 0 mismatches | 600 snapshots across 6 phases, 0 |
| Randomized invariant checking | 0 violations | 2,000 games, 1,045,111 transitions, 0 |

Every previously established threshold was met or exceeded:

| Gate | Phase Two | Phase Two Point One |
|---|---:|---:|
| Valid anti-leak trials | 103,625 | 103,625 |
| Complete replay games | 10,000 | 10,000 |
| Replayed plies | 5,078,406 | 5,078,406 |
| Invariant-checked transitions | 1,045,111 | 1,045,111 |
| Mirrored pairs, all 127 channels | not achievable | 1,804 |

Anti-leak detail: 0 observation mismatches, 0 legal-action mismatches, 0
public-event mismatches, 0 browser/public-view mismatches, and 103,625 of
103,625 belief-target positive controls behaved as expected.

Replay detail: 0 board-state mismatches, 0 observation mismatches, 0 event
mismatches, 0 terminal-result mismatches.

Invariant detail: 0 violations across 1,045,111 transitions, with every
transition checked against the immutability baseline and the previous knowledge
snapshot.

### Freeze

All acceptance gates pass, so the reference engine is frozen. From this point:

- no performance optimization of the reference engine;
- no change to rule semantics;
- no change to `observation_v2_1_127ch`;
- no change to the 10,000-entry source-destination action encoding;
- no change to replay semantics.

Any later behavioural change requires an explicit new version identifier and a
differential comparison against this implementation, which is now the
behavioural source of truth for the project.

---

## Repository tree

```text
.
├── PHASE_2_IMPLEMENTATION_INSTRUCTIONS.md
├── requirements.txt
├── .gitignore
│
├── stratego_project_docs/          # Phase One documentation, unmodified
│   ├── README.md
│   ├── 01_official_rules.md
│   ├── 02_project_ruleset.md
│   ├── 03_game_engine_spec.md
│   ├── 04_engine_validation_plan.md
│   ├── 05_project_plan.md
│   ├── 06_observation_v2_127ch.md
│   ├── 07_observation_validation_matrix.md
│   ├── 08_internal_state_spec.md
│   └── 09_public_event_and_replay_schema.md
│
├── stratego/
│   ├── __init__.py
│   └── engine/
│       ├── __init__.py
│       ├── constants.py
│       ├── coordinates.py
│       ├── pieces.py
│       ├── setup.py
│       ├── state.py
│       ├── actions.py
│       ├── legal_moves.py
│       ├── combat.py
│       ├── transition.py
│       ├── behavior.py
│       ├── observation.py
│       ├── events.py
│       ├── replay.py
│       ├── snapshot.py
│       ├── invariants.py
│       ├── random_play.py
│       └── permutation.py
│
├── tests/
│   ├── __init__.py
│   ├── helpers.py
│   ├── engine/
│   │   ├── test_geometry.py
│   │   ├── test_setup.py
│   │   ├── test_movement.py
│   │   ├── test_scout.py
│   │   ├── test_combat.py
│   │   ├── test_transition.py
│   │   ├── test_terminal_conditions.py
│   │   ├── test_rule_exclusions.py
│   │   ├── test_action_encoding.py
│   │   ├── test_invariants.py
│   │   ├── test_snapshot.py
│   │   └── test_determinism.py
│   ├── observation/
│   │   ├── test_shape_and_ranges.py
│   │   ├── test_identity_planes.py
│   │   ├── test_movement_and_origin.py
│   │   ├── test_setup_memory.py
│   │   ├── test_unresolved_inventory.py
│   │   ├── test_behavior_channels.py
│   │   ├── test_recent_moves.py
│   │   ├── test_global_channels.py
│   │   └── test_perspective.py
│   ├── information_security/
│   │   ├── test_anti_leak.py
│   │   ├── test_knowledge.py
│   │   ├── test_browser_privacy.py
│   │   └── test_belief_targets.py
│   └── replay/
│       ├── test_replay.py
│       └── test_event_stream.py
│
├── scripts/
│   ├── run_phase2_validation.py
│   └── manual_inspection_examples.py
│
└── reports/
    ├── phase_2_implementation_report.md
    └── phase_2_metrics.json
```

Each test package carries an `__init__.py` so `tests.helpers` is importable from
every test module; those files are omitted above for readability.

---

## Running the suite and regenerating this report

From the repository root, with the virtual environment created once via
`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`:

```bash
# Full automated test suite (about 17 seconds)
.venv/bin/python -m pytest -q

# One test package
.venv/bin/python -m pytest tests/observation -q

# Fast smoke run of the acceptance harness (about 10 seconds)
.venv/bin/python scripts/run_phase2_validation.py --quick

# Full acceptance run, regenerating reports/phase_2_metrics.json (about 6 minutes
# on the target Mac mini with 12 worker processes)
.venv/bin/python scripts/run_phase2_validation.py \
    --replay-games 10000 \
    --antileak-trials 120000 \
    --antileak-trials-per-position 25 \
    --mirror-seeds 250 \
    --invariant-games 2000 \
    --snapshot-seeds 120 \
    --legal-seeds 250 \
    --workers 12

# Regenerate the section 24.16 manual inspection examples
.venv/bin/python scripts/manual_inspection_examples.py
```

The harness exits with status `0` only when every gate reports zero unexplained
mismatches. `reports/phase_2_metrics.json` is written on every run; the prose in
this report is then updated from those numbers.
