# Phase Two Implementation Instructions — Python Reference Stratego Engine

## Purpose

You are implementing **Phase Two** of a Stratego artificial intelligence project.

Assume your current working directory is the **root directory of the extracted project documentation package** previously provided.

The Phase One Markdown files already present in this directory are the authoritative specification. Your task is to implement the Python reference engine, the complete automated validation suite, and a detailed informational report that allows another technical collaborator to determine whether Phase Two is working correctly and is ready for Phase Three.

Do **not** begin Phase Three optimization, machine-learning model development, setup-generator development, browser-interface development, or training-system development.

The reference engine prioritizes:

1. correctness;
2. readability;
3. deterministic behavior;
4. strict hidden-information separation;
5. comprehensive validation;
6. reproducibility.

It will later serve as the behavioral source of truth against which an optimized engine can be tested.

---

# 1. Read the Phase One documentation first

Before writing code, read these files completely from the current root directory:

```text
README.md
01_official_rules.md
02_project_ruleset.md
03_game_engine_spec.md
04_engine_validation_plan.md
05_project_plan.md
06_observation_v2_127ch.md
07_observation_validation_matrix.md
08_internal_state_spec.md
09_public_event_and_replay_schema.md
```

Treat them as the implementation contract.

The approved model observation is `observation_v2_127ch`, with:

- 127 spatial feature planes;
- board dimensions of 10 by 10;
- a separate 10,000-entry legal-action mask.

The internal state must store compact factual information. The 127-channel observation must be reconstructed from that factual state rather than stored as a second authoritative copy.

## Specification priority

If sources disagree, use this priority:

1. `02_project_ruleset.md`
2. the detailed Phase One specifications
3. `01_official_rules.md`
4. the Ataraxos research paper, if available
5. general Stratego knowledge

If two project documents conflict, do not silently choose one interpretation. Instead:

1. stop implementation of the affected behavior;
2. document the conflict;
3. identify the relevant files and sections;
4. explain the reasonable interpretations;
5. state your recommended interpretation;
6. continue only with unaffected work.

---

# 2. Important project-specific rule choices

This project intentionally excludes:

- the two-square rule;
- the continuous-chasing rule.

Do **not** implement either rule.

The project uses:

- 100 consecutive battleless moves as the training draw limit;
- 200 consecutive battleless moves as the evaluation/human-play draw limit;
- 4,000 total moves as the training safety limit.

These draw rules terminate games. They do not independently restrict ordinary legal movement.

---

# 3. Where all new files must be placed

Assume you are working from the root of the extracted documentation package.

Create the following project structure in that same root directory:

```text
.
├── README.md
├── 01_official_rules.md
├── 02_project_ruleset.md
├── 03_game_engine_spec.md
├── 04_engine_validation_plan.md
├── 05_project_plan.md
├── 06_observation_v2_127ch.md
├── 07_observation_validation_matrix.md
├── 08_internal_state_spec.md
├── 09_public_event_and_replay_schema.md
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
│       └── random_play.py
│
├── tests/
│   ├── engine/
│   ├── observation/
│   ├── information_security/
│   └── replay/
│
├── reports/
│   ├── phase_2_implementation_report.md
│   └── phase_2_metrics.json
│
└── PHASE_2_IMPLEMENTATION_INSTRUCTIONS.md
```

You may slightly change the implementation file decomposition if another structure is clearly simpler or more maintainable. If you do so, explain the difference in `reports/phase_2_implementation_report.md`.

Do not move or rename the Phase One documentation files.

Do not overwrite Phase One documentation unless an implementation-discovered ambiguity requires a documented correction. If any Phase One file is modified, list every modification in the final report.

---

# 4. Programming style

Use Python for the reference engine.

Prefer:

- clear functions;
- simple data structures;
- explicit state transitions;
- descriptive variable names;
- deterministic behavior;
- detailed comments for non-obvious logic;
- small focused modules;
- assertions and invariant checking;
- beginner-to-intermediate-readable Python.

Avoid classes when a functional design is practical.

Small data containers may be used when they materially improve correctness or clarity, but do not create an unnecessary object-oriented hierarchy around pieces, boards, or rules.

Do not optimize aggressively. This implementation is a correctness reference, not the final high-performance simulator.

---

# 5. Implementation order

Implement the engine incrementally in the order below.

## Step 1 — Constants and coordinate system

Implement and test:

- 10-by-10 board coordinates;
- 100 absolute square indices;
- 92 occupiable squares;
- 8 lake squares;
- human-readable square names;
- conversion between human-readable coordinates and internal indices;
- player-relative coordinate transformations;
- perspective normalization.

Required tests:

- all 100 square indices round-trip correctly;
- all human-readable coordinates round-trip correctly;
- all lake locations are correct;
- player-relative transformations invert correctly.

Do not proceed until these tests pass.

## Step 2 — Piece definitions and inventory

Implement the exact starting inventory for each player:

| Piece | Count |
|---|---:|
| Flag | 1 |
| Spy | 1 |
| Scout | 8 |
| Miner | 5 |
| Sergeant | 4 |
| Lieutenant | 4 |
| Captain | 4 |
| Major | 3 |
| Colonel | 2 |
| General | 1 |
| Marshal | 1 |
| Bomb | 6 |

Each player must have exactly 40 stable physical piece records.

A physical piece identifier must:

- never change during a game;
- be independent of the piece type;
- persist after the piece is captured.

## Step 3 — Setup validation

Implement setup validation before ordinary gameplay.

A legal setup must:

- contain exactly 40 pieces;
- contain the exact official inventory;
- occupy exactly the player's 40 legal setup squares;
- place no piece on a lake;
- place no piece outside the setup area;
- contain no duplicate occupancy.

Invalid setups must fail clearly. Do not silently repair invalid setups.

Create explicit fixtures for valid setups and intentionally invalid setups.

## Step 4 — Full internal state

Implement the authoritative internal state according to `08_internal_state_spec.md`.

The state must contain enough factual information to reconstruct:

- board occupancy;
- piece positions;
- true piece types;
- ownership;
- stable physical piece identifiers;
- starting squares;
- alive/captured state;
- moved state;
- identity knowledge for each player;
- acting player;
- total move count;
- battleless move count;
- recent move history;
- active threat relations;
- latest behavioral events;
- terminal state;
- terminal reason.

Do not store `observation_v2_127ch` as authoritative state. The observation must always be derived.

---

# 6. Legal movement

## Ordinary movable pieces

Implement:

- one-square cardinal movement;
- no diagonal movement;
- no movement outside the board;
- no movement onto lakes;
- no movement onto friendly pieces;
- movement onto an adjacent opponent piece as an attack.

## Flag and Bomb

They must never have legal movement actions.

## Scout

Implement cardinal ray movement:

- one or more squares;
- cannot jump pieces;
- cannot cross lakes;
- may stop on any unobstructed empty square;
- may attack the first opponent piece encountered;
- cannot continue past any occupied square.

A move of more than one square publicly identifies the moving piece as a Scout and must update knowledge state correctly.

---

# 7. Action encoding

Use the fixed source-destination action space.

For source index `0..99` and destination index `0..99`:

```text
action_id = 100 * source + destination
```

This creates exactly 10,000 possible action identifiers.

The engine must expose:

- a list of legal action identifiers;
- a 10,000-entry legal-action mask.

These must agree exactly for every state.

The game engine is always authoritative about action legality.

---

# 8. Combat

Implement an exhaustive combat resolver.

For ordinary ranked pieces:

- higher rank survives;
- lower rank is captured;
- equal ranks remove both.

Special cases:

- Spy attacking Marshal → Spy survives;
- Marshal attacking Spy → Marshal survives;
- Miner attacking Bomb → Miner survives;
- any other movable piece attacking Bomb → attacker is captured and Bomb survives;
- any movable piece attacking Flag → Flag is captured and the attacker wins immediately.

Combat must correctly update:

- board occupancy;
- captured state;
- surviving piece location;
- public identity knowledge;
- battleless-move counter;
- public events;
- terminal state.

Create an exhaustive attacker-versus-defender test matrix.

---

# 9. Atomic state transitions

Applying a legal action must update all relevant information as one coherent state transition.

Update, when applicable:

1. source occupancy;
2. destination occupancy;
3. moving piece position;
4. moved status;
5. Scout revelation;
6. combat outcome;
7. captured states;
8. identity revelations;
9. recent-move history;
10. threat state;
11. behavioral-event records;
12. total move count;
13. battleless move count;
14. terminal state;
15. terminal reason;
16. acting player.

An illegal action must leave the entire game state unchanged.

Create a test that captures the complete state, attempts an illegal move, and verifies exact equality afterward.

---

# 10. Battleless-move counter

Rules:

- every legal move increments the counter by 1;
- any combat resets the counter to 0;
- ordinary non-combat movement does not reset it;
- when the active configured threshold is reached, the game immediately becomes a draw.

Training:

```text
100 consecutive battleless moves → draw
```

Evaluation/human play:

```text
200 consecutive battleless moves → draw
```

---

# 11. Terminal conditions

Support the exact terminal labels:

```text
flag_capture
opponent_no_legal_move
both_no_legal_move_draw
battleless_move_limit_draw
absolute_move_limit_draw
not_terminal
```

Do not expose only a Boolean finished flag.

---

# 12. Hidden information and public identity knowledge

The internal engine knows every true piece type. The player observation must not.

Enforce these rules:

- players always know all of their own piece identities;
- unrevealed opponent identities remain hidden;
- combat revelations persist;
- multi-square Scout movement permanently reveals Scout identity;
- public knowledge never reverses;
- captured/revealed identities remain usable for inventory deduction.

Privileged information used for future belief-learning targets must remain separate from the policy observation.

No model-facing observation function may access hidden identities except when generating explicitly privileged belief targets.

---

# 13. Behavioral events

Implement exactly the five approved behavioral event types defined in `06_observation_v2_127ch.md`:

1. threat;
2. evade;
3. declined attack;
4. protect;
5. was protected.

Do not replace these definitions with intuitive alternatives.

Behavior records should retain factual metadata, not already-computed observation values.

At minimum store:

- event type;
- actor piece identifier;
- counterpart piece identifier;
- event ply;
- whether the actor knew the counterpart identity when the event occurred.

Behavior detection must not use privileged hidden identity information.

---

# 14. Observation construction

Implement `observation_v2_127ch` with exact shape:

```text
127 × 10 × 10
```

Implement the channel groups in exactly the documented order:

1. current own identities;
2. known opponent identities;
3. hidden opponent occupancy;
4. own disclosure state;
5. movement state;
6. starting coordinates;
7. own setup memory;
8. known opponent setup identities;
9. unresolved opponent inventory;
10. own behavioral history;
11. opponent behavioral history;
12. most recent 16 moves;
13. lake mask;
14. normalized game progress;
15. normalized battleless-move progress.

Do not reorder channels.

Also provide machine-readable metadata describing:

- observation version;
- channel index;
- channel name;
- valid range;
- short description.

The legal-action mask must remain separate.

---

# 15. Perspective normalization

One future neural network will play both colors.

The observation must normalize the board so the acting player always sees the game from the same orientation.

Test:

- red observations;
- blue observations;
- starting coordinates;
- recent-move planes;
- lake positions;
- source/destination transformations;
- setup-memory planes;
- equivalent mirrored states.

Equivalent positions from opposite player perspectives should normalize consistently where the specification expects them to.

---

# 16. Public event stream

Implement the event schema from `09_public_event_and_replay_schema.md`.

Within a move, event ordering must be deterministic:

1. move;
2. identity revelations;
3. combat;
4. derived behavioral events;
5. game end.

If several events occur in one category, use the deterministic ordering defined in the Phase One documentation.

Public/browser-safe events must never expose hidden information.

---

# 17. Deterministic replay

A complete game must be reproducible from:

- rules version;
- observation version;
- red setup;
- blue setup;
- first player;
- ordered action sequence;
- explicit random seeds used to generate stochastic inputs.

Replay must reproduce exactly:

- board state at every ply;
- legal-action mask;
- identity knowledge;
- behavior records;
- observations;
- public events;
- combat results;
- counters;
- terminal reason;
- winner or draw.

Phase Two acceptance target:

```text
At least 10,000 complete games replayed
with zero unexplained mismatches.
```

Observation reconstruction must also match at every ply.

---

# 18. Snapshot and restore

Implement compact state snapshot and restoration.

A restored state must reproduce exactly:

- legal actions;
- observations;
- public state;
- next transition under the same action;
- behavioral events;
- terminal result.

Test snapshots from:

- early game;
- middle game;
- late game;
- immediately before combat;
- immediately after combat;
- near the battleless-move limit.

This functionality is required now because later decision-time search will depend on it.

---

# 19. Random test-game generator

Implement a simple testing agent that selects uniformly from legal actions.

Its purpose is validation, not playing strength.

Use it to generate:

- complete random games;
- replay fixtures;
- randomized states;
- snapshot tests;
- hidden-information tests;
- terminal-condition tests.

All stochastic testing must support deterministic seeding.

---

# 20. State invariants

Create an invariant-checking function.

At minimum verify:

- exactly 40 physical piece records per player;
- every living piece occupies exactly one legal board square;
- captured pieces occupy no board square;
- no square contains more than one piece;
- owner never changes;
- stable piece identifier never changes;
- starting square never changes;
- true type never changes;
- players always know their own piece identities;
- public knowledge never reverses;
- hidden identities become public only through legal causes;
- Flag never moves;
- Bomb never moves;
- board occupancy matches piece records.

During development and stress testing, invariant failure should be treated as a hard error.

---

# 21. Hidden-information anti-leak testing

This is a mandatory acceptance test.

For a valid public game history:

1. clone the privileged full state;
2. permute true piece identities only among opponent pieces whose identities remain hidden;
3. preserve all publicly known constraints;
4. reconstruct the acting player's public observation;
5. compare all 127 channels;
6. compare legal actions;
7. compare browser/public state;
8. compare the public event stream where applicable.

These public-facing results must remain identical unless the modified state changes information that is already legally deducible.

Privileged belief-learning targets are expected to differ and should be checked as a positive control.

Acceptance target:

```text
At least 100,000 valid hidden-identity permutation trials
with zero unexplained public-information mismatches.
```

Any unexplained mismatch means Phase Two fails until the issue is resolved.

---

# 22. Automated testing

Implement the complete automated test requirements from:

```text
04_engine_validation_plan.md
07_observation_validation_matrix.md
```

Create tests under:

```text
tests/engine/
tests/observation/
tests/information_security/
tests/replay/
```

Use explicit known board positions where practical.

Do not simply recreate implementation logic inside the tests.

At minimum cover:

- board geometry;
- coordinate conversion;
- setup validation;
- movement;
- Scout movement;
- Flag/Bomb immobility;
- combat;
- reveal state;
- terminal conditions;
- battleless counter;
- action-list/action-mask agreement;
- observation channels;
- behavior channels;
- perspective normalization;
- hidden-information security;
- replay;
- snapshot/restore;
- state invariants;
- deterministic seeded execution.

---

# 23. Performance instrumentation

Phase Two is not an optimization phase, but record a baseline.

Measure separately:

- legal-action generation time;
- action-application time;
- observation-construction time;
- complete state transitions per second;
- observations per second;
- complete random games per second;
- average random-game length;
- memory use;
- terminal-reason distribution.

If a simple batch wrapper exists, measure several batch sizes.

Do not redesign the engine solely to improve these figures unless an obvious implementation error is causing pathological performance.

These measurements will guide Phase Three.

---

# 24. Required informational report

Create:

```text
reports/phase_2_implementation_report.md
```

This report must be detailed enough that another technical collaborator who did not write the implementation can determine whether Phase Two is correct and ready for Phase Three.

Do not write only a general statement such as `Everything works.` Provide evidence.

## 24.1 Executive status

State exactly one:

```text
PASS — recommended for Phase Three
```

or:

```text
CONDITIONAL PASS — specific issues remain
```

or:

```text
FAIL — Phase Two acceptance criteria not met
```

Then briefly explain the decision.

## 24.2 Implementation inventory

Report:

- files created;
- files modified;
- approximate lines of implementation code;
- approximate lines of test code;
- Python version;
- external packages;
- important package versions.

## 24.3 Requirements traceability

Create a table:

| Requirement | Specification source | Implementation location | Test location | Status |
|---|---|---|---|---|

Include all major Phase Two requirements.

## 24.4 Automated test summary

Report:

- total tests;
- passed;
- failed;
- skipped;
- expected failures;
- execution time.

List every failure or skipped test individually.

## 24.5 Rule validation

Report results for:

- board geometry;
- setup inventory;
- normal movement;
- Flag movement;
- Bomb movement;
- Scout movement;
- combat;
- Flag capture;
- no-legal-move victory;
- draw conditions;
- exclusion of the two-square rule;
- exclusion of continuous chasing.

## 24.6 Combat matrix

Report:

- number of attacker-defender combinations tested;
- passing cases;
- failing cases;
- Spy versus Marshal behavior;
- Marshal versus Spy behavior;
- Miner versus Bomb behavior;
- ordinary pieces versus Bomb;
- equal-rank combat;
- Flag capture.

## 24.7 Legal-action consistency

Report:

- number of positions tested;
- legal-action list/action-mask discrepancies;
- any edge cases encountered.

Target: `0 discrepancies`.

## 24.8 Observation validation

For `observation_v2_127ch`, report:

- exact output shape;
- data type;
- channel-range validation;
- channel metadata availability;
- perspective-normalization results;
- behavioral-channel validation;
- recent-move validation;
- inventory-channel validation;
- setup-memory validation.

Do not dump complete tensors into the report.

## 24.9 Hidden-information leak report

This section is mandatory.

Report:

- attempted hidden-state permutation trials;
- valid trials;
- observation mismatches;
- legal-action mismatches;
- public-event mismatches;
- browser/public-view mismatches;
- expected privileged belief-target differences.

Acceptance target:

```text
>= 100,000 valid trials
0 unexplained public-information mismatches
```

If an unexplained mismatch remains, final Phase Two status must be `FAIL`.

## 24.10 Replay report

Report:

- complete games generated;
- complete games replayed;
- total plies reconstructed;
- board-state mismatches;
- observation mismatches;
- event mismatches;
- terminal-result mismatches.

Acceptance target:

```text
>= 10,000 complete games
0 unexplained mismatches
```

## 24.11 Snapshot/restore report

Report:

- snapshots tested;
- legal-action mismatches after restore;
- observation mismatches;
- public-event mismatches;
- next-transition mismatches.

Include testing from early game, middle game, late game, before combat, after combat, and near draw termination.

## 24.12 State-invariant stress report

Report:

- complete games checked;
- total state transitions checked;
- invariant violations;
- exact violated invariant for any failure.

Target: `0 invariant violations`.

## 24.13 Random-game statistics

Report:

- number of games;
- average moves;
- median moves;
- minimum moves;
- maximum moves;
- red win rate;
- blue win rate;
- draw rate;
- counts and percentages for every terminal reason.

These are diagnostic values, not playing-strength targets. Investigate unusually skewed results.

## 24.14 Performance baseline

On the target Mac mini, if available, report:

| Measurement | Result |
|---|---:|
| Legal-action generations per second | |
| State transitions per second | |
| Observations generated per second | |
| Complete random games per second | |
| Mean memory use | |
| Peak memory use | |

Also estimate the fraction of engine time spent in:

- legal-action generation;
- observation construction;
- transitions;
- behavioral processing;
- other work.

Do not optimize yet.

## 24.15 Storage baseline

Report approximate serialized sizes for:

- one snapshot;
- one typical complete replay;
- 1,000 typical replays.

Estimate storage required for 1,000,000 games using the measured average replay size.

## 24.16 Manual inspection examples

Include at least five concise human-readable examples:

1. ordinary movement;
2. Scout revelation;
3. combat;
4. hidden-piece observation;
5. behavioral-event tracking.

For each provide:

- simplified board;
- acting player;
- action;
- expected result;
- actual result;
- relevant observation/event changes.

## 24.17 Known limitations

List every known incomplete feature, ambiguous interpretation, performance concern, testing limitation, and platform limitation.

## 24.18 Specification deviations

List every implementation difference from the Phase One documents.

For each state:

- what changed;
- why;
- whether it was explicitly approved;
- which documentation needs updating.

If there are none, write:

```text
No known specification deviations.
```

## 24.19 Open questions

List any decisions required before Phase Three. Do not make later-phase architecture decisions on your own.

## 24.20 Acceptance checklist

End the report with:

```text
[ ] All rule tests pass
[ ] Complete combat matrix passes
[ ] Legal-action list and mask agree
[ ] 127-channel observation contract passes
[ ] Behavioral representation passes
[ ] Perspective normalization passes
[ ] 100,000+ hidden-information anti-leak trials pass
[ ] 10,000+ complete replay reconstructions pass
[ ] Snapshot/restore passes
[ ] State invariants pass under stress
[ ] Deterministic seeded execution passes
[ ] No unexplained specification deviations
[ ] Performance baseline recorded
[ ] Storage baseline recorded
```

Every checked item must be supported by evidence earlier in the report.

---

# 25. Machine-readable metrics report

Also create:

```text
reports/phase_2_metrics.json
```

Include at minimum:

```text
implementation_version
rules_version
observation_version
python_version

tests_total
tests_passed
tests_failed
tests_skipped

combat_cases_tested
combat_cases_failed

legal_action_positions_tested
legal_action_mismatches

anti_leak_trials_attempted
anti_leak_trials_valid
anti_leak_observation_mismatches
anti_leak_action_mismatches
anti_leak_event_mismatches
anti_leak_public_view_mismatches

replay_games
replay_plies
replay_state_mismatches
replay_observation_mismatches
replay_event_mismatches
replay_result_mismatches

snapshot_tests
snapshot_mismatches

invariant_games
invariant_transitions
invariant_violations

random_games
random_game_mean_moves
random_game_median_moves
random_game_min_moves
random_game_max_moves
random_red_wins
random_blue_wins
random_draws

terminal_reason_counts

legal_actions_per_second
state_transitions_per_second
observations_per_second
random_games_per_second

mean_memory_bytes
peak_memory_bytes

snapshot_serialized_bytes
mean_replay_serialized_bytes
estimated_million_game_storage_bytes
```

Use valid JavaScript Object Notation. Do not add comments to this file.

---

# 26. Documentation updates

If implementation reveals a real ambiguity or error in the Phase One documents:

1. do not silently fix only the implementation;
2. update the relevant Markdown specification;
3. preserve the intended project design;
4. record the exact documentation change in the report.

If no specification corrections are required, leave all Phase One Markdown files unchanged.

---

# 27. Required completion package

When Phase Two is complete, the root directory should contain:

1. the original Phase One documentation;
2. `stratego/` with the complete reference engine;
3. `tests/` with the complete automated test suite;
4. `reports/phase_2_implementation_report.md`;
5. `reports/phase_2_metrics.json`;
6. this instruction document.

Also provide:

- an updated repository tree;
- concise instructions for running all tests;
- concise instructions for regenerating the Phase Two report;
- a compressed archive containing the complete updated project directory.

Do not return only patches or partial files.

---

# 28. Stop conditions

Stop and report the issue instead of inventing behavior if:

- two Phase One specifications contradict each other;
- a behavioral event cannot be implemented without privileged hidden information;
- an observation channel remains ambiguous;
- the rules do not determine an outcome;
- deterministic replay cannot represent required state;
- satisfying one acceptance criterion violates another.

When this happens, report:

- affected documents;
- affected sections;
- exact conflict;
- reasonable interpretations;
- recommended interpretation;
- blocked work.

Continue only with unrelated implementation work.

---

# 29. Definition of done

Phase Two is not complete merely because two agents can play a game.

The reference engine is complete only when it can serve as a trustworthy behavioral oracle for the rest of the project.

The minimum acceptance requirements are:

- all documented project rules implemented;
- all automated rule tests passing;
- exhaustive combat validation passing;
- legal-action list and action mask matching exactly;
- exact `observation_v2_127ch` construction;
- correct behavioral representation;
- correct perspective normalization;
- zero unexplained hidden-information leaks over at least 100,000 valid permutation trials;
- deterministic replay over at least 10,000 complete games with zero unexplained mismatches;
- snapshot/restore equivalence;
- state invariants holding under stress;
- deterministic seeded execution;
- performance baseline recorded;
- storage baseline recorded;
- complete informational report produced;
- machine-readable metrics report produced;
- no unresolved correctness problem.

Do **not** begin Phase Three yourself.

Return the complete Phase Two implementation and reports for review.
