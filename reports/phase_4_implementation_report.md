# Phase 4 Implementation Report

Frozen reference: `phase2_1_reference_1.1.0`  
Rules: `stratego_project_v1`  
Observation: `observation_v2_1_127ch`  
Phase 3 backend: `KEEP_PYTHON`

## 1. Agent 1 — Evaluation Foundations and Setup Bank

### 1.1 Status

**PASS**

Every completion gate is met:

| Gate | Result |
|---|---|
| Existing tests remain green | 1,630 passed, 0 failed (was 1,497 before this agent) |
| Policy API contains no privileged state | no `GameState`, `PieceRecord`, belief target or replay is reachable from a policy input |
| Setup bank has >= 512 valid deterministic pairs | 1,024 pairs, 0 validation failures |
| Match identities are deterministic | 412 determinism trials, 0 failures |
| Paired scheduling is explicit and reproducible | 1,024 paired units, 0 failures |
| All new tests pass | 133 new tests, 0 failures |

Prerequisites confirmed before implementation: `reports/phase_3_implementation_report.md`
ends with `Phase 3 backend decision: KEEP_PYTHON`, and the pre-existing suite was
green at 1,497 passed.

### 1.2 Implementation summary

`stratego/evaluation/` adds the contracts every later Phase 4 agent builds on.
`stratego/engine/` and `stratego/training/` were **not** modified. The package
contains no game rules: legality, combat, terminal precedence and knowledge
remain the frozen engine's, and every legality or observation product is
obtained by calling the engine rather than reimplementing it.

#### Policy interface (`policy.py`, `policy_interface_v1`)

One interface for Phase 4 baselines, future neural checkpoints and later search
policies. A `Policy` declares `policy_id`, `policy_version` and a
`PolicyRequirements` set, and implements `decide(request) -> PolicyResult`.

`PolicyInput` carries exactly the fields the instructions require — policy
identity, match identity (`match_id`, `paired_unit_id`, `game_id`,
`suite_version`), `ply`, `acting_player`, `legal_actions`, the
`observation_v2_1_127ch` tensor, the policy seed and a per-ply `decision_seed`,
plus public metadata — and nothing else.

**Materialisation is requirement-declared rather than lazy, and that is a
deliberate strengthening of the design.** The obvious way to make the
127-channel observation lazy is a closure over the `GameState`, but a closure
*is* a live reference to the privileged state and would violate the "no engine
object that permits reading hidden types" rule. Instead a policy declares what
it needs and `build_policy_input` materialises exactly that much; anything
undeclared arrives as `None` and `require_observation()` /
`require_legal_action_mask()` / `require_public_view()` raise
`PolicyContractError` if a policy reads what it did not declare. The practical
effect is identical — an unused observation is never built — and the structural
guarantee is stronger: `PolicyInput` is a frozen dataclass of scalars, tuples
and NumPy arrays with no path to `GameState`, `PieceRecord`, `BehaviorEvent`,
`RecentMove`, `ReplayRecord` or a belief target. A test walks the whole object
graph and asserts this.

Both materialised arrays are set read-only, so a policy cannot scribble on
engine-derived memory.

`PublicView` is the decoded observer-safe view Agent 2's rule-based baselines
will actually use: occupancy by square, all 80 pieces with unresolved opponent
types masked to `None`, own/opponent/unresolved piece identifier lists,
per-type unresolved opponent counts, which of the observer's own pieces the
opponent has legally learned, the 16-ply public move window, and the battleless
and absolute clocks. Every field is either public or the observer's own, and
every field is invariant under `permute_hidden_identities`. Building it costs
0.052 ms, against 0.034 ms for the observation tensor, so neither is a runtime
concern at league scale.

`PolicyResult` carries `selected_action_id`, the `PolicyRef`, the
`decision_seed` and optional serialisable diagnostics.
`validate_policy_result` rejects an illegal action, a wrong policy identity, a
wrong decision seed and a non-`PolicyResult` return value; `decide_checked`
applies it. Nothing substitutes a legal move for a failed decision — failures
are raised.

Three `contract_*` fixtures exist for testing only (`contract_first_legal`,
`contract_uniform_random`, `contract_observation_probe`). They implement no
strategy and their identifiers are prefixed so they can never be mistaken for a
ladder opponent. Agent 2 owns the real baseline suite, including Random Legal.

#### Match identity (`match_spec.py`, `match_spec_v1`)

`MatchSpec.match_id` is a BLAKE2b hash over exactly:

```text
match_spec_version, suite_version, pairing_mode,
candidate policy id@version, opponent policy id@version,
setup bank version, setup_pair_id, candidate colour, replicate, root_seed,
the complete rules configuration
```

The rules configuration is included field by field (`rules_token`), so a match
identifier can never silently mean two different rule sets; a test perturbs
every `RulesConfig` field and requires the token to change, and checks the two
deliberately-unsettable excluded-rule fields textually.

Deliberately absent: worker index, schedule position, shard, wall-clock time and
process identity. All derived values — `match_id`, `paired_unit_id`,
`candidate_seed`, `opponent_seed`, `game_id` — are computed properties, so a
specification cannot exist carrying an identifier that disagrees with its own
components. Policy seeds are `derive_policy_seed(match_id, role)`, so they
follow from identity alone.

`MatchSpec.from_dict` recomputes the identifier and raises if a stored
`match_id` disagrees, which catches both a tampered field and a mismatched rules
configuration.

#### Paired evaluation unit

The pairing rule is **`color_swap_same_board`**, stated explicitly rather than
assumed:

```text
Game A: red_setup = R, blue_setup = B, candidate plays RED  (moves first)
Game B: red_setup = R, blue_setup = B, candidate plays BLUE (moves second)
```

Both games start from the *identical physical position*; only the policy-to-colour
assignment flips. This matters because a setup is stored in each player's own
`SETUP_SQUARES` order and those orders are **not** symmetric — red's setup index
0 is board row 0, the row furthest from the lakes, while blue's index 0 is board
row 6, the row nearest them. So "give the candidate the same setup on the other
side" is a board transformation, not a relabelling. Holding the board fixed
avoids the transformation entirely and cancels two confounders exactly:

- setup-quality asymmetry, since both policies play both arrangements;
- first-move advantage, since `first_player` is red and each policy is red once.

The alternative — rotating the board so each policy physically replays its own
arrangement — is documented in `setup_bank.py` (the transform is the rank-block
reversal `orient_setup` implements) but is not used. `PAIRING_MODES` exists so a
second mode could be added later without changing any existing identifier.

`PairedUnit.from_match` recovers the whole unit, including the sibling game, from
either half; `sibling_match` is an involution.

#### Scheduling primitives

`build_paired_schedule`, `build_round_robin_schedule` (each unordered policy pair
once — the colour swap inside a unit is what balances a matchup),
`schedule_matches`, `schedule_digest` (order-independent content digest),
`validate_schedule` (duplicate matches, unpaired units, missing setup pairs) and
`shard_schedule`. Enumeration order and shard index affect nothing but the order
rows come back in.

#### Evaluation setup bank (`setup_bank.py`, `evaluation_setup_bank_v1`)

1,024 deterministic legal setup pairs, the preferred target rather than the 512
floor; generation costs 0.23 s, so runtime was never a reason to fall back.

Arrangements are generated in a **canonical own-orientation frame** (rank 0 = own
back row furthest from the lakes, rank 3 = front row, file = absolute column) and
mapped onto each player's setup indices by `orient_setup`, which is the identity
for red and a rank-block reversal for blue. This is what makes a structural rule
mean the same thing for both colours; a test proves rank 0 lands on board row 0
for red and row 9 for blue, and that canonical adjacency is real board adjacency
for both.

Structural rules of the `structured_v1` family, all fixed and hand-coded:

1. the flag sits on rank 0 or rank 1;
2. two or three bombs are placed orthogonally adjacent to the flag;
3. remaining bombs favour the back ranks;
4. scouts favour the front ranks;
5. marshal and general are kept off the front rank;
6. spy and miners favour the back ranks;
7. everything else is dealt uniformly to the remaining cells.

Nothing here is learned, scored or tuned against results. This is not the Phase 7
training setup generator, and a policy must not read the bank to infer an
opponent's arrangement.

Each entry records `setup_pair_id`, red setup, blue setup, `generation_seed` and
`bank_version` (plus `generation_family`). Seeds are
`derive_pair_seed(root_seed, setup_pair_id)`, hashed rather than mixed, so any
worker can rebuild any single pair without generating its neighbours — verified
for 128 sampled pairs.

Validation delegates to the frozen engine: `validate_setup` for exact inventory,
`validate_setup_placement` for legal setup squares and lakes, and `create_game`
plus `check_invariants` to prove no overlap and exactly 80 occupied squares.

### 1.3 Files created / modified

Created:

```text
stratego/evaluation/__init__.py
stratego/evaluation/policy.py
stratego/evaluation/match_spec.py
stratego/evaluation/setup_bank.py
tests/evaluation/__init__.py
tests/evaluation/test_policy_contract.py
tests/evaluation/test_match_spec.py
tests/evaluation/test_setup_bank.py
scripts/run_phase4_agent01.py
reports/phase_4_data/agent_01_evaluation_foundations.json
reports/phase_4_data/agent_01_setup_bank_v1.json
reports/phase_4_implementation_report.md
```

Modified: **none**. `stratego/engine/`, `stratego/training/`, the existing test
suite and `stratego_project_docs/` are untouched.

### 1.4 Tests run

```bash
python -m pytest -q
```

**1,630 passed, 0 failed, 0 errors** in 43.3 s. The 1,497 pre-existing tests all
still pass; 133 are new.

| Module | Tests | Covers |
|---|---|---|
| `tests/evaluation/test_setup_bank.py` | 41 | canonical frame and orientation, legality, inventory, determinism, isolated rebuild, identifier uniqueness, variation, structural rules, serialisation |
| `tests/evaluation/test_match_spec.py` | 47 | identity determinism, per-component sensitivity, rules coverage, seed derivation, paired-unit properties, shuffling and sharding invariance, schedule construction and validation, round trips |
| `tests/evaluation/test_policy_contract.py` | 45 | privileged-object unreachability, type masking, requirement plumbing, read-only arrays, seeds and reproducibility, legality and result validation, permutation invariance with a positive control |

Acceptance harness:

```bash
python scripts/run_phase4_agent01.py
```

64.8 s end to end (`--quick` for a smoke run; `--skip-pytest` for measurements
only). A `--quick` run reports FAIL by design, because its 64-pair bank is below
the 512-pair acceptance floor.

### 1.5 Measured results

#### Setup bank

| Metric | Value |
|---|---|
| Bank version / family | `evaluation_setup_bank_v1` / `structured_v1` |
| Root seed | 20260101 |
| Pairs | 1,024 (preferred target; floor is 512) |
| Validation failures | 0 |
| Structural-rule violations | 0 |
| Duplicate `setup_pair_id` | 0 |
| Distinct red / blue setups | 1,024 / 1,024 |
| Distinct positions | 1,024 |
| Content digest (SHA-256) | `5fe5f987…b674266` |
| Generation time | 0.23 s |

Variation across the 2,048 arrangements: 2,048 distinct front rows, 2,048
distinct back rows, all 20 legal flag cells used, all 10 files used, flag rank
histogram `{0: 1514, 1: 534}`, flag-guard bomb histogram `{2: 1020, 3: 1006,
4: 22}`. Front and back rows — where an opponent actually meets the setup — are
100% distinct, which is the meaningful "not one arrangement permuted" claim.

#### Determinism

412 trials, **0 failures**:

- 8 full-bank regenerations, each byte-identical (`to_json()` compared directly);
- 128 pairs rebuilt in isolation, each equal to its bank entry;
- 1 JSON round trip preserving the digest;
- a 4,096-match schedule (1,024 pairs × 2 replicates × 2 colours) recomputed and
  rebuilt from scratch with identical identifiers;
- 8 shuffles, each leaving the schedule digest and every specification unchanged;
- 7 worker counts (1, 2, 3, 4, 8, 16, 32), each leaving the digest unchanged with
  no match lost or duplicated;
- 256 dictionary round trips reproducing both policy seeds;
- collision checks: 4,096 distinct match identifiers and 8,192 distinct policy
  seeds.

Schedule digest: `a8d178d1…5da1b89d`.

#### Paired units

1,024 units tested, **0 failures**. Each unit was checked for both colour
assignments, a shared `paired_unit_id` with distinct `match_id`s, identical
resolved setups across its two games, the correct setup pair, exactly one
first-move game per policy, reconstruction from either half and a stable
involutive sibling. `validate_schedule` reported 0 problems.

#### Contract games

96 games played end to end through the policy interface only, 96,853 plies,
**0 illegal actions**, **0 replay mismatches**, **0 paired-board mismatches**.
Every game was replayed from a dictionary round trip of its own `MatchSpec` and
compared on both action history and full `state_fingerprint`. Terminal reasons:
48 battleless-limit draws, 28 opponent-no-legal-move, 20 flag captures. Effective
scores (candidate as red 0.542, as blue 0.521) are contract-fixture artefacts and
measure nothing about Stratego strength.

#### Hidden-information permutation

2,000 valid permutation trials, 6,000 policy comparisons (3 fixtures × 2,000),
**0 mismatches** and **0 positive-control failures** — every trial's belief
target genuinely changed, so no trial was vacuous. Compared per trial:
observation tensor, legal-action list, `PublicView`, filtered public events,
public setup view, selected action and public diagnostics. Agent 4 owns the
>= 100,000-trial audit across the real baseline suite.

### 1.6 Deviations and limitations

1. **Requirement-declared materialisation instead of closure-based laziness.**
   Described in 1.2. Same outcome — an unrequested observation is never built —
   with a stronger structural guarantee. Agent 2 must declare
   `PolicyRequirements` accurately or `require_*` will raise.
2. **`EVALUATION_RULES` is the default rules configuration** (battleless limit
   200, absolute limit 4,000, `context="evaluation"`), not the `TRAINING_RULES`
   used for Phase 3 collection. The rules configuration is part of match
   identity, so a specification is self-describing and a faster
   `TRAINING_RULES` smoke schedule is unambiguous rather than silently different.
3. **`structured_v1` is not uniform-random.** Uniformly shuffled setups are legal
   but frequently degenerate (flag on the front rank, bare flag), which would let
   a baseline win by punishing a setup blunder rather than by playing better.
   The structural rules are fixed and hand-coded, but they are a bias, and the
   bank is not a sample from the space of legal setups. If Agent 4 needs an
   unbiased control, a second family can be added under a new bank version
   without disturbing this one.
4. **Only one pairing mode is implemented.** `PAIRING_MODES` carries a single
   entry. The rotation transform for the alternative is documented and available
   as `orient_setup`, but no code path uses it.
5. **The 2,000 permutation trials use the three `contract_*` fixtures**, which is
   a regression floor, not an audit. Only `contract_observation_probe` actually
   reads the observation. The real coverage arrives with Agent 2's baselines and
   Agent 4's audit.
6. **Determinism rests on `random.Random` (Mersenne Twister) for the bank.**
   Consistent with `stratego/engine/setup.py`, and the frozen digest plus the
   committed bank artefact make any future drift loud rather than silent.
7. **No match runner, no parallelism, no statistics, no baseline strategy.** The
   game loop inside `contract_game_stage` is a throwaway acceptance check that
   the contract fits the engine, deliberately kept out of the package so it
   cannot be mistaken for Agent 3's runner.

### 1.7 Data files

```text
reports/phase_4_data/agent_01_evaluation_foundations.json
reports/phase_4_data/agent_01_setup_bank_v1.json
```

The first holds status and every check above, all version identifiers, the full
setup-bank summary with diversity metrics and digest, every determinism,
paired-unit, contract-game and permutation measurement, the test summary and the
environment record. The second is the materialised 1,024-pair bank in canonical
JSON — the digest input, and a frozen artefact any later checkpoint can compare
against. `files_created` and `files_modified` in the JSON are harness metadata;
section 1.3 is authoritative.

### 1.8 Handoff notes

**For Agent 2 (baselines and stress opponents):**

- Subclass `stratego.evaluation.Policy`. Set `policy_id`, `policy_version`,
  `requirements` and `stochastic`, then implement `decide`. Use
  `self.result(request, action_id, diagnostics)` to build a well-formed result.
- Work from `request.require_public_view()`. It already gives occupancy, masked
  piece types, `has_moved`, unresolved opponent counts per type, which own pieces
  are exposed, the 16-ply move window and the battleless clock — everything the
  instructions list as legal for a tactical or strategic baseline, without
  decoding 127 planes. Declare `observation=True` only if you actually need the
  tensor.
- Break ties deterministically **before** any seeded sampling, and draw
  randomness only from `request.random_stream()`. It is reseeded from
  `decision_seed` on every call, so two draws in one decision give the same
  numbers — take one stream and reuse it if you need a sequence.
- Diagnostics must be serialisable and must not name a hidden opponent type.
  They are compared for equality in the permutation tests, so a diagnostic that
  varies with hidden state is a hard failure, not a nuisance.
- Your differential regression suite can reuse
  `tests/evaluation/test_policy_contract.py::_permutation_cases` as a pattern:
  permute, require `info["changed"]`, and assert `belief_targets_differ` so the
  trial is not vacuous.
- Do not name a policy `contract_*`.

**For Agent 3 (match runner and statistics):**

- Build a game with `spec.resolve_setups(bank)` then
  `create_game(red, blue, rules=spec.rules, game_id=spec.game_id)`. Per ply use
  `spec.policy_ref_for(actor)`, `spec.policy_seed_for(actor)` and
  `build_policy_input(...)`, then `policy.decide_checked(request)` and
  `apply_action(state, result.selected_action_id, legal=legal)`.
  `contract_game_stage` in `scripts/run_phase4_agent01.py` is a working
  reference; it is not a runner and should be replaced, not imported.
- Assign identity before dispatch and never let a worker influence it. Shard with
  `shard_schedule` and verify with `schedule_digest`, which is order-independent
  by construction.
- **Bootstrap over `paired_unit_id`, not over individual games.** The two games
  of a unit share a board and a policy pair and are not independent.
- `validate_schedule(matches, bank)` catches duplicates, unpaired units and
  missing setup pairs before a long run starts.
- A raw match record can reproduce a game from `match_id` plus the bank version
  alone; `MatchSpec.to_dict()` / `from_dict` round-trips and re-verifies the
  identifier.
- Let `PolicyContractError` propagate. Do not substitute a legal move.

**For Agent 4 (calibration and audit):**

- The bank has 1,024 pairs, so 1,024 paired units per adjacent-tier comparison is
  available without replicates; `replicates` extends it further.
- Reuse `permutation_stage` in `scripts/run_phase4_agent01.py` as the audit
  shape. It already enforces the positive control and counts only trials where
  the permutation actually changed something.
- Compare the regenerated bank digest against
  `reports/phase_4_data/agent_01_setup_bank_v1.json` as a standing reproducibility
  check.

**Recommended project-document updates** (for the later review that updates those
documents once Phase 4 is accepted; `stratego_project_docs/` was not modified):

1. `05_project_plan.md` — record `policy_interface_v1`, `match_spec_v1` and
   `evaluation_setup_bank_v1`, and the `color_swap_same_board` pairing decision.
2. `09_public_event_and_replay_schema.md` — record `PublicView` as an
   observer-safe product alongside `public_board_view`, since it is now part of
   the surface the permutation gate protects.
3. A note that the evaluation setup bank is explicitly not the Phase 7 training
   setup generator, to keep the two from being conflated later.

## 2. Agent 2 — Baseline and Stress Opponents

### 2.1 Status

**PASS**

Every completion gate is met:

| Gate | Result |
|---|---|
| Random, Basic, Tactical and Strategic policies exist | 4-tier ladder, nested by construction |
| At least four stress policies exist | 6 stress policies |
| Every policy uses Agent 1's observer-safe interface | all 10 subclass `Policy`; none requests the observation tensor |
| No policy returns an illegal action | 50,000 acceptance decisions (5,000 positions x 10 policies), 0 illegal |
| Deterministic/stochastic reproducibility | 20,000 determinism trials, 0 failures |
| Local hidden-information differential tests | 1,500 permutation trials, 15,000 policy comparisons, 0 differences |
| Earlier tests remain green | 1,779 passed, 0 failed (was 1,630 before this agent) |

Prerequisite confirmed before implementation: Section 1 above reports `PASS`, and
`reports/phase_4_data/agent_01_evaluation_foundations.json` carries
`"status": "PASS"` with 0 determinism failures and 0 permutation mismatches.

Acceptance run: 383 s single-process.

### 2.2 Implementation summary

Four new modules under `stratego/evaluation/`. `stratego/engine/` and
`stratego/training/` were **not** modified, and neither were Agent 1's
`policy.py`, `match_spec.py` or `setup_bank.py`. This agent adds no rules: the
frozen engine remains the authority on legality, combat, terminal precedence and
knowledge, and the one engine rule the heuristics reach for is `resolve_combat`,
only ever with types the observer legally knows or with a type drawn from the
publicly deducible unresolved inventory.

#### `heuristics.py` (`phase4_heuristics_v1`)

Shared, observer-safe feature extraction. `DecisionContext` is built once per
ply from a `PublicView` plus the legal action list and precomputes everything a
scoring pass needs: decoded candidate moves, a known-attacker map per square,
adjacency counts for hidden pieces that have already moved, own support, free
space, unresolved inventory, material estimates, own Flag landmarks and a
16-ply repetition memory. Scoring a single move is then close to constant time,
which matters because a mid-game position routinely offers well over a hundred
actions.

Two public inferences are used throughout, and no others:

1. **Expected value over the unresolved inventory.** `unresolved_opponent_counts`
   says how many copies of each type remain unaccounted for, so attacking an
   unknown piece has a computable expected value rather than a hand-tuned
   constant. This is the single largest source of the Basic-to-Tactical gap.
2. **A piece that has moved is neither Flag nor Bomb.** Movement is public and
   both are immovable. This is precisely the constraint
   `permutation_is_valid()` enforces, so conditioning on it can never
   distinguish two permutations of the same public position.

#### `baselines.py` (`phase4_baseline_suite_v1`)

The ladder is a **nesting**, not four unrelated agents: `Strategic` subclasses
`Tactical` and calls the same `_tactical_terms()`, and each tier is defined by
which weights in a single `HeuristicWeights` table are nonzero. A strength
inversion therefore points at a specific added term, and disabling that term is
a one-line experiment.

| Tier | Adds |
|---|---|
| `random_legal` | nothing; uniform over the legal set |
| `basic_heuristic` | known captures, forward progress, anti-shuffling |
| `tactical_rule_based` | expected-value attacks, threat pricing and evasion, support, Miner/Bomb, Spy/Marshal approach, Scout probes, own-Flag defence |
| `strategic_rule_based` | mobility, Miner preservation scaled by unresolved Bombs and remaining Miners, exposure control, territorial pressure on never-moved blocks, Flag-guard retention, draw-counter awareness |

Weights live in three module-level tables rather than inline, so Agent 4 can
recalibrate by editing one table and bumping the affected `policy_version`.

#### `stress.py`

Six policies, each taking one idea to an extreme to widen the distribution of
games: `stress_scout_rush`, `stress_miner_rush`, `stress_draw_seeker`,
`stress_berserker`, `stress_information_miser`, `stress_chaos`. All six take a
known Flag capture regardless of their own objective, because a stress opponent
that declines a won game is noise rather than a useful distribution.
`stress_chaos` draws a fresh random weight vector over public features every
ply, so it is coherent within a ply and incoherent across plies — a regime no
fixed policy produces.

#### `registry.py`

One named catalogue that Agents 3 and 4 schedule from: `build_policy(policy_id)`,
`policy_ref`, `policy_catalog()` and the `LADDER_*`/`STRESS_*`/`ALL_*` id
tuples. The `contract_*` fixtures from `policy.py` are deliberately excluded and
the prefix is rejected at import, so an interface fixture can never enter a
league and report a meaningless strength number.

#### Determinism and information safety

Every policy reads `request.require_public_view()`, `request.legal_actions` and
`request.random_stream()`, and nothing else. Because Agent 1 proved `PublicView`
invariant under `permute_hidden_identities` and the legal action list is one of
the products the permutation gate protects, **a pure function of invariant
inputs is invariant** — the safety property is structural, and the differential
tests confirm it rather than being the only thing establishing it.

Ranking is by score and then by action identifier, so the candidate list is
deterministic *before* any sampling; only then does a policy draw from the
decision stream to break a near-tie.

### 2.3 Files created and modified

Created:

```text
stratego/evaluation/heuristics.py
stratego/evaluation/baselines.py
stratego/evaluation/stress.py
stratego/evaluation/registry.py
tests/evaluation/test_baselines.py
tests/evaluation/test_baseline_information_safety.py
scripts/run_phase4_agent02.py
reports/phase_4_data/agent_02_baseline_agents.json
reports/phase_4_data/agent_02_behavior_profile.csv
```

Modified:

```text
stratego/evaluation/__init__.py          (re-exports only; no existing symbol changed)
reports/phase_4_implementation_report.md (this section appended)
```

### 2.4 Tests run

```text
python -m pytest -q                      1,779 passed, 2 skipped, 0 failed
python scripts/run_phase4_agent02.py     PASS
```

149 new tests. The 2 skips are deliberate: `random_legal` and `stress_chaos` do
not expose a per-move score vector, so the score-vector invariance test does not
apply to them; both are still covered by every other invariance test.

`tests/evaluation/test_baselines.py` (112 tests) covers the catalogue, legality,
reproducibility, the closed diagnostic vocabulary, the heuristic primitives
against the engine's own combat resolver, crafted positions for each tier's
distinguishing rule, and behavioural assertions for every stress policy.

`tests/evaluation/test_baseline_information_safety.py` (37 tests) is the local
differential suite: a positive control that the permutation changed the
privileged state and the belief target, a **leak detector** proving the fixture
positions can still distinguish hidden state at all, invariance of the public
view, of every derived context field, of each policy's action, of each policy's
whole diagnostics payload, and of the entire per-move score vector — the last
being strictly stronger than comparing the argmax, since two score vectors can
share a maximum and differ everywhere else.

### 2.5 Measured results

**Legality.** 5,000 seeded positions spanning plies 6, 15, 30, 55, 85, 125 and
175, decided by all 10 policies: **0 illegal actions**, 0 empty legal sets in a
nonterminal state.

**Determinism.** 20,000 trials over 400 positions: **0 failures**. Repeated
decisions, fresh instances and seed sweeps all agree. Seed sensitivity is
measured rather than assumed, and the numbers show the sampler behaves exactly
as designed — for every scoring policy the count of positions where the seed
changed the action equals the count of positions with a genuine near-tie, so the
seed resolves every near-tie and never fires otherwise:

| Policy | Positions with a near-tie | Positions where the seed changed the action |
|---|---|---|
| `random_legal` | n/a | 400 / 400 |
| `basic_heuristic` | 159 | 159 |
| `tactical_rule_based` | 131 | 131 |
| `strategic_rule_based` | 93 | 93 |
| `stress_scout_rush` | 84 | 84 |
| `stress_miner_rush` | 120 | 120 |
| `stress_draw_seeker` | 395 | 395 |
| `stress_berserker` | 104 | 104 |
| `stress_information_miser` | 300 | 300 |
| `stress_chaos` | n/a | 400 / 400 |

**Hidden-information permutation.** 1,500 valid trials, 15,000 policy
comparisons, spread across all seven snapshot plies (183–238 trials each):
**0 action differences, 0 diagnostic differences, 0 score-vector differences**.
0 positive-control failures, 0 leak-detector failures and 0 positions skipped as
unchanged — every trial permuted something real.

**Behavioural fingerprints** (pooled over both reference opponents, 64 paired
units each). The stress policies separate from the ladder on every axis Agent 4
named:

| Policy | Role | Mean plies | Attack rate | Scout moves | Miner moves | Reveal rate | Draw rate | Flag-capture wins | Piece entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|
| `random_legal` | ladder | 348 | 0.048 | 0.120 | 0.131 | 0.800 | 0.023 | 0.016 | 3.22 |
| `basic_heuristic` | ladder | 279 | 0.209 | 0.109 | 0.147 | 0.747 | 0.027 | 0.480 | 3.21 |
| `tactical_rule_based` | ladder | 300 | 0.179 | 0.132 | 0.176 | 0.633 | 0.000 | 0.738 | 3.18 |
| `strategic_rule_based` | ladder | 335 | 0.191 | 0.170 | 0.181 | 0.584 | 0.016 | 0.875 | 3.14 |
| `stress_scout_rush` | stress | 598 | 0.059 | **0.793** | 0.032 | 0.678 | 0.324 | 0.219 | **1.37** |
| `stress_miner_rush` | stress | 499 | 0.061 | 0.025 | **0.841** | 0.566 | 0.305 | 0.336 | **1.12** |
| `stress_draw_seeker` | stress | **787** | **0.000** | 0.193 | 0.105 | 0.621 | **0.488** | 0.000 | 3.04 |
| `stress_berserker` | stress | **236** | **0.299** | 0.099 | 0.163 | 0.757 | 0.000 | 0.301 | 3.20 |
| `stress_information_miser` | stress | **866** | **0.000** | 0.064 | 0.054 | 0.710 | 0.281 | 0.000 | 3.10 |
| `stress_chaos` | stress | 714 | 0.077 | 0.080 | 0.116 | 0.779 | 0.230 | 0.062 | 3.25 |

`stress_information_miser` also has a Scout-run rate of exactly 0.000: combat and
multi-square Scout moves are the only two reveal paths in this ruleset, and it
declines both.

**Informational ladder screen** (192 paired units per matchup, 384 games,
resampled over the paired unit, normal approximation). Not calibration — see
2.6:

| Candidate | Opponent | Candidate EWR | ~95% interval | Separated from even |
|---|---|---|---|---|
| `random_legal` | `basic_heuristic` | 0.115 | [0.084, 0.145] | yes |
| `random_legal` | `tactical_rule_based` | 0.026 | [0.011, 0.041] | yes |
| `random_legal` | `strategic_rule_based` | 0.013 | [0.003, 0.023] | yes |
| `basic_heuristic` | `tactical_rule_based` | 0.273 | [0.235, 0.312] | yes |
| `basic_heuristic` | `strategic_rule_based` | 0.204 | [0.165, 0.244] | yes |
| `tactical_rule_based` | `strategic_rule_based` | 0.542 | [0.497, 0.587] | **no** |

Five of six matchups separate cleanly, giving at least three ordered tiers
(`random_legal` < `basic_heuristic` < {`tactical`, `strategic`}). The sixth is
discussed below.

### 2.6 Deviations and limitations

1. **`tactical_rule_based` and `strategic_rule_based` are not yet statistically
   separable head-to-head.** Strategic strictly dominates Tactical against both
   weaker tiers (0.013 vs 0.026 against Random; 0.204 vs 0.273 against Basic),
   but their direct matchup sits at 0.542 for Tactical with a 95% interval that
   contains 0.5. This is reported, not fixed: the Agent 2 instructions say
   "Do not tune policies to arbitrary target win percentages yet. Agent 4 owns
   calibration," and Agent 4's brief explicitly covers revising heuristic
   weights if the tiers are too similar. Phase 4's global gate asks for three
   distinguishable tiers, which the screen already provides.

   To make that handoff actionable rather than a puzzle, each Strategic-only
   term was ablated against Tactical over 192 paired units. This was a one-off
   diagnostic, not part of the acceptance harness; it is reproduced by
   `dataclasses.replace(STRATEGIC_WEIGHTS, <term>=0.0)` on a `StrategicRuleBasedPolicy`
   subclass and rerunning `screen_stage`:

   | Term disabled | Strategic EWR vs Tactical |
   |---|---|
   | none (full Strategic) | 0.487 |
   | `pressure` | 0.380 |
   | `miner_preservation` | 0.452 |
   | `flag_guard` | 0.456 |
   | `mobility` | 0.460 |
   | `battleless` | 0.486 |
   | `exposure` | **0.520** |

   `pressure` is by far the load-bearing term. `exposure` is the only term whose
   removal *improves* Strategic, making it the obvious first knob for Agent 4.
   `battleless` is effectively inert because it only engages past half the
   battleless window, which these games rarely reach.

2. **Every policy in the suite is stochastic.** There is no purely deterministic
   baseline. Two deterministic policies facing each other tend to lock into a
   repeating shuffle and hit the battleless-move draw limit, which would leave
   the ladder indistinguishable at draws. Every policy is still fully
   reproducible from `(public input, policy seed, ply)`, and the deterministic
   branch of the selector (`margin = 0`) is covered by its own test, so the
   "same public input -> same action" contract is exercised even though no
   catalogued policy uses it.

3. **A located opponent Flag is unreachable in practice.** Under
   `stratego_project_v1` the only reveal path is combat, and combat with a Flag
   ends the game on the same ply, so no policy is ever asked to decide against a
   Flag it has already identified. The `flag_capture` branch is kept as a
   correctness guard; the instructions' "immediate Flag capture" behaviour is
   delivered through the expected-value term, where the unresolved Flag carries
   its share of the value of attacking an unknown piece. Own-Flag *defence* is
   unaffected and does fire, since a policy always knows its own Flag.

4. **Hidden Scouts are not modelled as ray threats.** The threat map gives
   revealed opponent Scouts their full ray, but treats hidden pieces as
   adjacency threats only. Modelling every hidden piece as a possible Scout
   would make almost the whole board look threatened and cost a great deal of
   time for little signal. This makes the baselines slightly optimistic about
   long-range danger; it is a deliberate modelling choice, not an oversight, and
   it uses no privileged information either way.

5. **Behavioural metrics are opponent-dependent.** `stress_chaos` attacks on
   7.2% of moves against `random_legal` but 9.5% against
   `strategic_rule_based`. Every policy is therefore profiled against two
   references at opposite ends of the ladder and both rows are reported
   alongside the pooled row, so the spread is visible rather than averaged away.
   A reference cannot play itself, so `random_legal` and `strategic_rule_based`
   have one row each. Agent 4 recomputes all of this over the full league.

6. **The screen's confidence intervals are a normal approximation** over paired
   units, computed inline. They are a smoke-test aid, not the interval method
   Agent 3 will implement, and nothing in Phase 4 should cite them.

7. **No match runner, no parallelism, no statistics beyond the screen.** The
   game loop in `run_phase4_agent02.py` is a throwaway acceptance harness kept
   out of the package so it cannot be mistaken for Agent 3's runner.

### 2.7 Data files

```text
reports/phase_4_data/agent_02_baseline_agents.json
reports/phase_4_data/agent_02_behavior_profile.csv
```

The JSON holds status and every check above, all version identifiers, the policy
catalogue with declared requirements, the full legality, determinism,
permutation, behavioural and screen measurements, the test summary and the
environment record. The CSV is the behavioural table in a flat form — one row per
`(policy, reference opponent)` plus a pooled `all` row per policy — for Agent 4's
stress characterisation.

### 2.8 Handoff notes

**For Agent 3 (match runner and statistics):**

- Build opponents with `stratego.evaluation.registry.build_policy(policy_id)` and
  schedule from `policy_ref(policy_id)`. `ALL_POLICY_IDS`, `LADDER_POLICY_IDS`
  and `STRESS_POLICY_IDS` enumerate the suite; `policy_catalog()` is the
  serialisable description for a run manifest.
- A policy instance is stateless across decisions — proved by the acceptance run
  — so a worker may build one per process and reuse it for a whole shard.
- Pass `requirements=policy.requirements`. No baseline wants the observation
  tensor or the legality mask, so a runner that always materialises all products
  would waste most of its time building 127-plane observations nobody reads.
- Let `PolicyContractError` propagate. Do not substitute a legal move.
- Expect long games from `stress_draw_seeker` and `stress_information_miser`
  (mean 787 and 866 plies, and 48.8% and 28.1% draws respectively). Any
  per-match timeout must be sized against those, not against the ladder's ~300
  plies.

**For Agent 4 (calibration and audit):**

- The audit shape is already in `permutation_stage()` in
  `scripts/run_phase4_agent02.py`: it enforces the positive control, a leak
  detector, and compares actions, whole diagnostics payloads **and** full
  per-move score vectors. Scaling that loop past 100,000 trials needs only a
  larger position pool; `generate_positions()` snapshots one random game at
  seven plies rather than replaying a game per position, which is what makes the
  sweep cheap.
- Start tier tuning at deviation 1 above. `exposure` is the term to revisit
  first, `pressure` is the one to leave alone, and `battleless` is inert as
  configured. Weights are in `BASIC_WEIGHTS`, `TACTICAL_WEIGHTS` and
  `STRATEGIC_WEIGHTS` in `baselines.py`; **bump `policy_version` with any weight
  change**, since a match identity names the policy by `id@version` and a silent
  weight edit would invalidate every stored identifier that mentions it.
- `reports/phase_4_data/agent_02_behavior_profile.csv` is the starting point for
  the stress characterisation; the pooled rows already show separation of one to
  two orders of magnitude on attack rate, Scout share, Miner share, game length
  and draw rate.
- If a new policy is added, put it in `registry.py`. The information-safety
  suite parametrises over `ALL_POLICY_IDS`, so a policy that is not catalogued
  is silently absent from the audit — `test_every_policy_in_the_catalogue_is_covered`
  guards that, but only for policies that reached the registry.

## 3. Agent 3 — Match Runner and Statistics

### 3.1 Status

**PASS**

Every completion gate is met:

| Gate | Result |
|---|---|
| Exact match reproduction works | 64 in-process reproductions + 64 from stored rows alone, 0 mismatches |
| Parallel worker count does not change results | 1/2/4/8 workers + a shuffled rerun, **0 mismatches**, 1 distinct results digest |
| Paired match scheduling works | 256 paired units, 0 structural problems |
| EWR and 95% paired confidence intervals implemented and tested | 30/30 acceptance checks, 75 unit tests |
| Colour/setup/terminal summaries work | all four reported per matchup |
| Raw records are sufficient to reproduce a game | 64/64 reproduced with no setup bank present |
| Policy failures are loud, never substituted | 4/4 raised and classified, **0 substituted legal moves** |
| Earlier tests remain green | 1,931 passed, 0 failed (was 1,779 before this agent) |

Prerequisites confirmed before implementation: sections 1 and 2 above both report
`PASS`, both JSON files carry `"status": "PASS"`, and Agent 1's frozen setup-bank
artefact still regenerates byte-for-byte — digest `5fe5f987…` from a fresh
1,024-pair generation matches the committed
`agent_01_setup_bank_v1.json`.

Acceptance run: 147 s end to end, including the full 1,933-test suite.

### 3.2 Implementation summary

Four new modules under `stratego/evaluation/`. `stratego/engine/`,
`stratego/training/` and Agents 1–2's five modules were **not** modified; the only
edit to existing code is re-exports in `stratego/evaluation/__init__.py`. This
agent adds no rules: `legal_actions` decides what may be played, `apply_action`
applies it, and the engine's terminal precedence decides how a game ends. The
runner's only judgements are about process — whose turn it is, what a policy may
see, and whether a policy honoured its contract.

#### `match_runner.py` (`match_runner_v1`, `match_result_v1`)

`play_match` drives one fully determined game through Agent 1's observer-safe
interface and returns a raw `MatchResult`. Policies receive only
`build_policy_input` products, and only the ones they declared — no baseline
requests the 127-channel observation, so a league never builds one.

**Why the worker count cannot change a result** is structural rather than tested
into existence. Every input to a game is fixed by `MatchSpec` before dispatch:
setups come from `setup_pair_id` in a versioned bank, the colour assignment is
part of `match_id`, both policy seeds derive from `match_id` alone, and each
per-ply decision seed derives from the policy seed and the ply. Workers receive
already-built identities, never the parameters identities are computed from, so
parallelism is a scheduling concern only.

**The raw row is deliberately self-sufficient.** It carries both setups as
40-character strings, so a game can be rebuilt and replayed from the row with no
setup bank present — 80 characters per row to remove the only external dependency
a stored result would otherwise have. It also carries the complete rules
configuration as a payload *and* as a token, and `rules_config()` rebuilds the
former and checks it against the latter, so a truncated or edited payload fails
loudly instead of replaying the game under different limits.

**`comparable()` is the canonical equality form**, not the whole row.
`wall_clock_seconds` genuinely differs between runs, so a naive row comparison
would be useless for a reproducibility gate; `comparable()` drops timing and file
references, and `comparable_digest()` additionally drops the inline action
history because `replay_digest` already covers every action. The consequence is
that `results_digest` is identical whether a run stored histories inline or left
them in a sidecar, so two runs are comparable on one string regardless of that
choice. `compare_results` compares only fields present on both sides, for the
same reason.

**Failure classification.** `_decide` mirrors `Policy.decide_checked` but
distinguishes *how* a policy failed — `illegal_action`, `contract_violation`,
`policy_exception`, `engine_rejected_action` — so `illegal_policy_actions` is
counted directly rather than parsed out of an exception message. Agent 1's
`validate_policy_result` still runs last and remains authoritative: a violation
the classifier fails to recognise is raised by the validator, not swallowed.

#### `scheduler.py` (`evaluation_scheduler_v1`)

The reproducibility gate is not "the runner is deterministic" but "an evaluator
holding only the identifiers can rebuild the same work", which needs the *set* of
matches to be storable and checkable. `EvaluationSchedule` binds a match list to
the bank version, rules and suite version it was built for, and `from_dict`
re-verifies every `match_id` **and** the schedule digest on load — so a schedule
that was edited, or built against a different bank version, fails at load rather
than quietly running different games. Builders are `build_matchup_schedule`,
`build_gauntlet_schedule` (the shape a future checkpoint evaluation takes),
`build_league_schedule` and `build_ladder_schedule`, all resolving policy
identifiers through the catalogue so a schedule names the version it was built
against.

#### `statistics.py` (`evaluation_statistics_v1`)

`EWR = (W + 0.5D)/N`, with wins, draws and losses always reported beside it.

**The resampling unit is the paired unit, and this is enforced rather than
documented.** The two games of a unit share a board and a first-move assignment,
so they are not independent observations; bootstrapping games would treat 2N
correlated games as 2N independent ones and report an interval that is too
narrow — the exact failure the pairing was designed to remove. A unit contributes
one number, the mean of its two games, so a unit score lives in
`{0, 0.25, 0.5, 0.75, 1.0}`, and `unit_score_histogram` distinguishes "splits
every pair" from "sweeps half and loses half", which share an effective win rate
and mean very different things.

Intervals are percentile bootstrap over units, seeded explicitly, drawn from
NumPy's PCG64 in blocks so peak memory does not grow with the resample count.
Block size is verified not to move an endpoint — the generator is one stream, so
blocking cannot change the draws. A normal-approximation interval is reported
alongside as a cross-check, never as the headline; 3.5 shows it running outside
`[0, 1]` where the bootstrap does not.

Every aggregation sorts rows by `match_id` first, so a summary is a function of
the result *set* rather than of worker arrival order. Per-matchup bootstrap seeds
derive from a hash of the matchup name, not its position, so adding a matchup to
a run cannot shift any other matchup's interval.

`bradley_terry_ratings` is the secondary ranking: MM algorithm, draws as half a
win, deterministic from sorted tokens with a fixed tolerance and iteration cap,
renormalised to a unit geometric mean each sweep and reported on an Elo-like
scale. `prior_draws` adds virtual drawn games per pair actually played, because
without it the likelihood has no finite maximum for a policy that sweeps a
matchup — and Agent 2 measured `random_legal` at 0.013 against
`strategic_rule_based`, close enough that a smaller sample produces one.

#### `reporting.py` (`evaluation_reporting_v1`)

CSV for the row table, JSONL for replay records keyed by `match_id`, JSON for
summaries, and Markdown renderers for this report. The split follows the storage
decision recorded in 3.6: at sweep scale the 512-row CSV is 343 KB while the
replay sidecar is 1.16 MB, so histories are 77% of the bytes even before Agent 4
scales the league up.

### 3.3 Files created and modified

Created:

```text
stratego/evaluation/match_runner.py
stratego/evaluation/scheduler.py
stratego/evaluation/statistics.py
stratego/evaluation/reporting.py
tests/evaluation/test_match_runner.py
tests/evaluation/test_statistics.py
tests/evaluation/test_parallel_reproducibility.py
scripts/run_phase4_agent03.py
reports/phase_4_data/agent_03_match_runner_statistics.json
reports/phase_4_data/agent_03_reproducibility_raw.csv
reports/phase_4_data/agent_03_acceptance_results.csv
reports/phase_4_data/agent_03_acceptance_replays.jsonl
```

Modified:

```text
stratego/evaluation/__init__.py          (re-exports only; no existing symbol changed)
reports/phase_4_implementation_report.md (this section appended)
```

### 3.4 Tests run

```text
python -m pytest -q                      1,931 passed, 2 skipped, 0 failed
python scripts/run_phase4_agent03.py     PASS
```

152 new tests. The 2 skips are Agent 2's pre-existing deliberate skips.

| Module | Tests | Covers |
|---|---|---|
| `tests/evaluation/test_match_runner.py` | 56 | result schema and round trips, exact reproduction, row-only reproduction, engine replay, tamper detection on history/identity/rules, the four policy-failure classes, quarantine, schedule construction/storage/validation/merging, and a loop proving every catalogued policy can be scheduled and played |
| `tests/evaluation/test_statistics.py` | 75 | known-result tables, the paired-versus-game bootstrap contrast, seed reproducibility and block invariance, quantiles, the normal cross-check, colour/setup/terminal/ply diagnostics, duplicate and half-unit detection, errored-row handling, order invariance, and league ratings |
| `tests/evaluation/test_parallel_reproducibility.py` | 21 | four worker counts including serial, per-field identity of the gate fields, action histories, policy seeds, statistics, shuffled schedules, chunk counts, over-provisioned workers and a multi-matchup league |

The parallel module is kept separate because each pool costs about a second under
macOS `spawn`; it uses 16 matches, and worker count 3 deliberately does not divide
the match count so an off-by-one in chunking surfaces as a lost or duplicated
match rather than hiding behind even division.

`scripts/run_phase4_agent03.py` takes `--quick` for a smoke run and
`--skip-pytest` for measurements only.

### 3.5 Measured results

#### Exact reproduction

64 matches over four matchups, 19,006 plies, **0 problems** on every check:

| Check | Count | Failures |
|---|---|---|
| Reproduced identically in-process | 64 | 0 |
| Reproduced from the stored row with no setup bank | 64 | 0 |
| Stored history replays through the engine | 64 | 0 |
| Replay record rebuilds the same terminal state and digest | 64 | 0 |
| Raw schema carries all 20 required fields | 64 | 0 |
| Both games of a unit share one board, one colour each | 32 units | 0 |

Tamper detection is tested rather than assumed: an altered ply count, setup pair,
or rules payload each fail at a different, specific check.

#### Parallel reproducibility

512 matches (256 paired units, 4 matchups, 153,068 plies) run at four worker
counts plus a shuffled rerun. Machine: 14 cores.

| Workers | Chunks | Seconds | Speedup | Matches/s | Results digest | Mismatches |
|---|---|---|---|---|---|---|
| 1 (serial) | 1 | 31.5 | 1.00x | 16.2 | `dc824f46…` | 0 |
| 2 | 8 | 16.9 | 1.86x | 30.2 | `dc824f46…` | 0 |
| 4 | 16 | 8.7 | 3.61x | 58.6 | `dc824f46…` | 0 |
| 8 | 32 | 4.8 | 6.50x | 105.6 | `dc824f46…` | 0 |
| 4, schedule shuffled | 16 | — | — | — | `dc824f46…` | 0 |

**1 distinct results digest across all five executions.** Required identical and
verified identical field by field: match IDs, both setups, replay digests,
winners, terminal reasons, ply counts, action histories and both policy seeds.
Only row order and wall-clock timing differ. The same digest also appeared in an
earlier separate invocation of the script, so reproduction holds across process
lifetimes, not just within one.

This sweep additionally confirms a property Agent 2 asserted but could not test:
the serial path builds fresh policy instances per match, while a worker builds
one instance and reuses it for its whole shard. A stateful policy would therefore
make serial and parallel disagree. Zero mismatches over 512 matches is
independent evidence that policy instances carry no state between decisions.

`reports/phase_4_data/agent_03_reproducibility_raw.csv` holds the per-match
evidence — 2,560 rows, one per (execution, match), each built from that
execution's own results rather than copied from the baseline.

#### Policy failures

Four deliberately broken policies, each run in both error modes:

| Broken policy | Classified as | Raised in default mode | Quarantine score | Statistics refused it |
|---|---|---|---|---|
| returns an illegal action | `illegal_action` | yes | `None` | yes |
| raises `RuntimeError` | `policy_exception` | yes | `None` | yes |
| returns a wrong decision seed | `contract_violation` | yes | `None` | yes |
| returns a bare `int` | `contract_violation` | yes | `None` | yes |

**0 substituted legal moves.** In default mode all four abort the run. In
quarantine mode all four produce a row with `candidate_result="error"`,
`candidate_score=None`, no winner, `terminal_reason="policy_error"` and a
populated `policy_error`; the statistics then refuse to summarise that row until
the caller passes `allow_policy_errors=True`, and even then the errored match's
whole paired unit is dropped rather than counted as half a unit.

#### Statistical validation

30/30 checks over synthetic tables with hand-computed answers: EWR 1.0 / 0.0 /
0.5 / mixed, win-draw-loss preservation, seed reproducibility, seed sensitivity
across 12 seeds, point-estimate containment, interval narrowing with sample size,
colour splits, duplicate and half-unit and same-colour detection, refusal to
summarise a broken table, order invariance at matchup and run level, the unit
histogram, league order recovery, league determinism under shuffling, equal
ratings for equal policies, prior-regularised finiteness, and six guard-rail
rejections.

The decisive one is `paired_bootstrap_is_not_a_game_bootstrap`. In a table where
the candidate wins every red game and loses every blue one, every unit scores
exactly 0.5, so a paired bootstrap must return a **point interval of width
0.0000**; a bootstrap over the same 64 games sees 32 wins and 32 losses and
returns width **0.2500**, against a normal-theory expectation of 0.2450 for
independent observations. The two differ by the entire width of the interval, so
the test cannot pass if the resampling unit is ever silently changed to the game.

#### Acceptance statistics over the sweep

Informational only — these four matchups were chosen to exercise the statistics
(a mismatch, two adjacent tiers, the pair Agent 2 could not separate, and a
draw-heavy stress matchup), **not** to measure the ladder. Agent 4 owns
calibration. 64 paired units each, 10,000 resamples, seed 20260403.

| Candidate | Opponent | EWR | 95% paired bootstrap | Normal approximation | W/D/L | Separated from even |
|---|---|---|---|---|---|---|
| `random_legal` | `strategic_rule_based` | 0.027 | [0.008, 0.055] | [0.003, 0.0520] | 2/3/123 | yes |
| `basic_heuristic` | `tactical_rule_based` | 0.266 | [0.199, 0.332] | [0.198, 0.3337] | 33/2/93 | yes |
| `tactical_rule_based` | `strategic_rule_based` | 0.547 | [0.461, 0.625] | [0.463, 0.6306] | 70/0/58 | **no** |
| `strategic_rule_based` | `stress_draw_seeker` | 0.992 | [0.980, 1.000] | [0.981, **1.0029**] | 126/2/0 | yes |

The last row is the concrete argument for the bootstrap being the headline: the
normal approximation puts the upper bound at **1.0029**, outside the range the
statistic can take, while the bootstrap correctly stops at 1.000. On the interior
matchups the two agree to within 0.003, which is why the normal interval is still
worth reporting as a cross-check.

| Matchup | EWR as red | EWR as blue | Red − blue | Mean plies | Median | Range | Unit histogram 0/0.25/0.5/0.75/1 |
|---|---|---|---|---|---|---|---|
| `random_legal` vs `strategic_rule_based` | 0.047 | 0.008 | +0.039 | 348 | 320 | 57–1061 | 59/3/2/0/0 |
| `basic_heuristic` vs `tactical_rule_based` | 0.281 | 0.250 | +0.031 | 198 | 192 | 19–544 | 31/2/29/0/2 |
| `tactical_rule_based` vs `strategic_rule_based` | 0.625 | 0.469 | **+0.156** | 302 | 298 | 29–719 | 12/0/34/0/18 |
| `strategic_rule_based` vs `stress_draw_seeker` | 1.000 | 0.984 | +0.016 | 348 | 318 | 14–829 | 0/0/0/2/62 |

The +0.156 red advantage in the `tactical`/`strategic` matchup is the reason the
colour swap is inside the evaluation unit rather than averaged over. An unpaired
schedule that happened to give one policy red more often would move that
matchup's estimate by more than the width of its confidence interval.

Run-level: 512 games, mean 299 plies, median 272, range 14–1061. Terminal
reasons: 442 flag captures (86.3%), 63 opponent-no-legal-move (12.3%), 7
battleless-limit draws (1.4%). Setup-pair stratification found no matchup resting
on a handful of boards — the `basic`/`tactical` result, for instance, spreads
across 64 pairs with a per-pair EWR median of 0.25 and 31 pairs swept against.

#### League rating

Bradley-Terry MM, converged in 91 iterations, `prior_draws=1.0`.

| Rank | Policy | Rating | BT strength | Games |
|---|---|---|---|---|
| 1 | `tactical_rule_based` | 1828.4 | 6.6221 | 256 |
| 2 | `strategic_rule_based` | 1796.0 | 5.4949 | 384 |
| 3 | `basic_heuristic` | 1653.4 | 2.4176 | 128 |
| 4 | `random_legal` | 1198.0 | 0.1758 | 128 |
| 5 | `stress_draw_seeker` | 1024.2 | 0.0646 | 128 |

This table is itself the clearest argument for the instruction "do not replace
effective win rate with Elo as the project success metric". It reports a 32-point
gap between `tactical_rule_based` and `strategic_rule_based` — a confident-looking
ordering for two policies whose direct paired interval, [0.461, 0.625], contains
0.5 and therefore separates them not at all. A single rating number cannot express
"statistically indistinguishable", which is exactly what Phase 4's tier gate asks.
The ranking is a convenience for ordering a long list; the interval is the
evidence.

#### Artefacts

| File | Size | Contents |
|---|---|---|
| `agent_03_acceptance_results.csv` | 343 KB | 512 rows, 38 columns, no action histories |
| `agent_03_acceptance_replays.jsonl` | 1.16 MB | 512 engine replay records keyed by `match_id` |
| `agent_03_reproducibility_raw.csv` | 486 KB | 2,560 rows: per-match evidence per execution |
| `agent_03_match_runner_statistics.json` | 34 KB | status, every measurement above, manifests |

The sidecar round-trips: reloading all 512 records and recomputing their digests
reproduces the digest stored on every row, and stripping the inline histories
leaves `results_digest` unchanged.

### 3.6 Deviations and limitations

1. **Replay histories live in a sidecar, not inline** (chosen deliberately with
   the user). A match is already fully reproducible from `match_id` plus the bank
   version, so the action history is a verification artefact rather than a
   reproduction requirement. Each row keeps a 64-byte `replay_digest`, and
   histories go to a JSONL sidecar written on request. The measured split above —
   343 KB of rows against 1.16 MB of histories for the same 512 games — is why:
   Agent 4's league is tens of thousands of games. Divergence remains detectable
   at full strength; only *inspecting* a divergence needs the sidecar.

2. **Policy failures fail fast by default, with quarantine as an explicit
   opt-in** (also chosen with the user). `on_policy_error="raise"` aborts the run;
   `"quarantine"` records the failure, scores nothing, and lets a long league
   finish. Neither mode ever substitutes a legal move. The instructions require a
   `policy_error` field, and quarantine is what exercises it — without that mode
   the field would exist but never be populated in practice.

3. **Parallelism uses `ProcessPoolExecutor`, not threads.** The GIL makes threads
   useless for this workload. Consequence: the acceptance script and any caller
   using `worker_count > 1` must be `spawn`-safe, i.e. guarded by
   `if __name__ == "__main__":`. A schedule run from an interactive session or
   piped stdin will fail in the child process, which is a Python packaging
   constraint rather than a runner defect.

4. **Chunking trades load balance against nothing that matters.** Matches are
   dealt round-robin to `worker_count * chunks_per_worker` chunks, defaulting to
   4 chunks per worker, so a matchup with much longer games spreads across workers
   instead of landing on one. Chunk count is verified not to change any result.
   Speedup at 8 workers is 6.50x rather than 8x, partly from this and partly from
   the 1,061-ply tail games.

5. **Bootstrap draws come from NumPy's PCG64.** Exact interval endpoints are
   reproducible from the seed given the same generator; NumPy guarantees stream
   stability for `default_rng` + `integers`, and the sweep reproduced identical
   endpoints across NumPy 2.2.4 and 2.5.1 during development. `bootstrap_engine`
   is recorded in every interval so a future change would be visible rather than
   silent. A pure-Python fallback was rejected as roughly two orders of magnitude
   slower at 10,000 resamples per matchup.

6. **Only percentile bootstrap is implemented**, not BCa or studentised
   intervals. For unit scores confined to `[0, 1]` the percentile interval cannot
   leave the range, which is the property the 0.992 matchup above shows actually
   matters. BCa would give better coverage for a strongly skewed statistic; if
   Agent 4 needs it, it is an addition to `statistics.py` and not a change to
   anything.

7. **A tampered setup string on a stored row is not caught by identity alone.**
   `match_id` covers `setup_pair_id`, not the resolved setups, so editing
   `red_setup` produces a row whose identity still verifies. It is caught by both
   independent checks the harness runs — `replay_stored_match` finds the stored
   actions no longer produce the stored outcome, and re-running against the bank
   produces a different digest — but a caller who does neither could replay a
   different game. Passing `bank=` to `reproduce_match` resolves setups from the
   bank and sidesteps this entirely.

8. **The league ratings in 3.5 sit on an incomplete comparison graph** — 4 of the
   10 possible pairs among those 5 policies. Bradley-Terry is defined on a
   connected graph and this one is connected, but ratings from a sparse graph lean
   heavily on the few edges present, which is part of why `tactical` ranks above
   `strategic` there. Agent 4's full league will have far more edges.

9. **No baseline calibration, and none of 3.5's win rates should be cited as
   such.** The instructions reserve calibration for Agent 4 and the sweep's
   matchups were chosen for statistical variety. The `tactical`/`strategic`
   non-separation at 0.547 [0.461, 0.625] over 64 units does independently
   reproduce Agent 2's finding at 0.542 over 192 units, which is useful
   corroboration for Agent 4 but is not a calibration result.

10. **No per-match timeout.** The engine's `absolute_move_limit` of 4,000 bounds
    every game, and the longest game in 512 was 1,061 plies, so a timeout would
    add a failure mode without removing one. If Agent 4 adds much slower policies
    it becomes worth revisiting.

### 3.7 Data files

```text
reports/phase_4_data/agent_03_match_runner_statistics.json
reports/phase_4_data/agent_03_reproducibility_raw.csv
reports/phase_4_data/agent_03_acceptance_results.csv
reports/phase_4_data/agent_03_acceptance_replays.jsonl
```

The JSON holds status and all 19 status checks, every version identifier, the
prerequisite verification including Agent 1's bank digest, all runner-correctness
and failure-handling counts, the full parallel sweep with per-worker timings and
digests, all 30 statistical checks with their details, the complete acceptance
statistics including per-matchup intervals and per-opponent pooling, the league
ratings, the artefact inventory, the run manifest and the test summary.
`files_created` and `files_modified` in the JSON are harness metadata; section 3.3
is authoritative.

### 3.8 Handoff notes

**For Agent 4 (calibration and audit):**

- Build a league with
  `build_league_schedule(list(ALL_POLICY_IDS), 512, name="phase4_calibration")`,
  validate it with `require_valid_schedule(schedule, bank)` **before** starting —
  it catches duplicates, incomplete units and missing setup pairs in milliseconds
  rather than after an hour of play — then
  `run_schedule(schedule.matches, bank, worker_count=8)`.
- Budget from the measured rate: **105.6 matches/s at 8 workers** for these
  matchups. A full 10-policy round robin at 512 paired units is 45 pairs × 1,024
  matches = 46,080 matches, roughly 7–8 minutes of match time, plus more for the
  slow stress matchups — Agent 2 measured `stress_information_miser` at 866 mean
  plies against the ladder's ~300, so expect that matchup to cost about 3x the
  average. Consider `record_actions=False` for the league and a separate
  history-bearing run for whatever needs inspection.
- `summarize_run(results, resamples=10_000, seed=...)` gives per-matchup EWR with
  paired intervals, colour splits, setup stratification, terminal frequencies,
  per-opponent pooling and the league table in one call.
  `render_run_report(summary)` renders all of it as Markdown for section 4.
- **The tier gate is `MatchupSummary.separated_from_even`, or better, whether two
  adjacent tiers' intervals overlap** — not the league ratings. 3.5 shows the
  ratings confidently ordering two policies their direct interval cannot separate.
- If you revise heuristic weights, **bump `policy_version`**. A `match_id` names
  the policy by `id@version`, `resolve_policies` refuses to play a match against a
  differently-versioned catalogue entry, and `validate_evaluation_schedule` flags
  a stored schedule whose version no longer exists. That is deliberate: it makes a
  silent weight edit loud instead of invalidating stored identifiers invisibly.
- Reuse the reproducibility subset check directly: `parallel_stage` in
  `scripts/run_phase4_agent03.py` already runs a schedule serially, in parallel and
  shuffled, and compares field by field.
- For the hidden-information audit, note that `play_match` never hands a policy
  anything but `build_policy_input` products, so the audit's surface is unchanged
  from Agents 1 and 2 — the runner adds no new path to privileged state.
- Errored matches: run the league with the default `on_policy_error="raise"` while
  screening, so a broken policy stops you immediately; switch to `"quarantine"`
  only for a long final run, and report `policy_errors` explicitly if it is
  nonzero. `summarize_run` will refuse to proceed otherwise.

**Recommended project-document updates** (for the post-acceptance review;
`stratego_project_docs/` was not modified):

1. `05_project_plan.md` — record `match_runner_v1`, `match_result_v1`,
   `evaluation_scheduler_v1`, `evaluation_statistics_v1` and
   `evaluation_reporting_v1`, and that the paired unit is the resampling unit for
   every reported interval.
2. `09_public_event_and_replay_schema.md` — record that `MatchResult` carries a
   `replay_digest` over the engine's own `ReplayRecord`, and that Phase 4 stores
   replay records in a sidecar JSONL keyed by `match_id`.
3. A note that effective win rate with a paired confidence interval is the
   success metric and Bradley-Terry ratings are a secondary convenience, so the
   two are not confused in later phases.

## 4. Agent 4 — Calibration, Security Audit, and Phase 4 Acceptance

### 4.1 Status

**PASS**

All 17 gates in `completion_gates` are true:

| Gate | Result |
|---|---|
| `all_prior_agents_passed` | Agents 1-3 all `PASS`; Agent 1's bank digest still regenerates |
| `hidden_information_trials_met` | **100,000** valid trials, 1,000,000 policy comparisons |
| `hidden_information_zero_mismatches` | **0** action, diagnostic, score-vector, public-view or legal-action differences |
| `positive_controls_verified` | 100,000 control trials, **0 failures** |
| `leak_detector_verified` | **0** trials where the hidden types were identical |
| `every_policy_legal` | 44,544 league games, **0 illegal actions, 0 policy errors** |
| `league_reproducible` | 5 executions, **1 results digest, 1 replay-digest set, 0 mismatches** |
| `paired_confidence_intervals_work` | all 45 matchups resample the paired unit, `sample_size == paired_units` |
| `three_or_more_strength_tiers` | **4 tiers** |
| `random_is_the_ladder_floor` | `random_legal` is tier 4 of 4 |
| `ladder_tiers_fully_ordered` | 6/6 cross-tier pairs separated in the correct direction |
| `strategic_separates_from_tactical` | 0.5354 [0.5168, 0.5540] after the exposure recalibration |
| `screen_found_no_broken_policy` | 5,760 screening matches, 0 errors, 0 ladder inversions |
| `stress_policies_distinct` | **6/6** materially different from the Strategic baseline |
| `raw_results_permit_reproduction` | 44,544-row raw CSV carrying both setups per row |
| `test_suite_green` | 1,963 passed, 2 skipped, 0 failed |
| `published_record_verified` | 32/32 in `test_phase4_acceptance.py` re-run after the artefact was written |

Prerequisites confirmed before implementation: sections 1-3 above all report `PASS`,
all three JSON files carry `"status": "PASS"`, and a fresh 1,024-pair generation
still digests to `5fe5f987…b674266`, matching Agent 1's committed artefact.

Acceptance run: 1,071 s end to end, including the full test suite twice.

### 4.2 Implementation summary

One new module, `stratego/evaluation/calibration.py` (`phase4_calibration_v1`), one
acceptance harness, one test module. `stratego/engine/`, `stratego/training/` and
`stratego_project_docs/` were **not** modified, and neither were Agent 1's or
Agent 3's modules. Agent 2's `baselines.py` was changed in exactly one place —
described in 4.5.2, and the reason this agent exists rather than merely measuring.

This agent adds no rules. The frozen engine remains the authority on legality,
combat, terminal precedence and knowledge; `permute_hidden_identities` remains the
authority on what a valid hidden-state permutation is; Agent 3's `play_match`
remains the only thing that plays a game.

#### `calibration.py` — four decisions worth stating

**The audit's chunking is not the worker count.** The audit's whole claim is that a
policy cannot see hidden state, and that claim is worth nothing if the evidence
changes shape depending on how many cores were free. `audit_payloads` therefore cuts
a run into a fixed number of deterministic slices — 32 by default — and `workers`
only decides how many run at once. Everything a chunk does follows from
`(root_seed, chunk_index)`, so the merged report is a pure function of
`(root_seed, target_trials, sources, plies, policy_ids, chunks)`. A test asserts a
serial run and a four-process run agree on every field, including the per-ply and
per-policy trial histograms.

**Positions come from two sources.** `random_walk` is cheap and spreads widely;
`baseline_play` walks the game with two catalogued policies, costs about three times
as much, and produces the *kind* of position a league actually visits — more
revealed pieces, a partly resolved inventory, pieces clustered where baselines
fight. A leak that only manifested in realistic positions would hide from a purely
random sweep, so the audit splits its trials 50/50. Ten snapshot plies from 4 to 200
cover the phases, because how much is hidden — and therefore what a leak could look
like — changes completely across a game.

**Behaviour is profiled by replay, not by re-simulation.** `MatchResult` is an
outcome row; it carries no attack rate, Scout share or reveal count. Rather than
reimplement the game loop and risk profiling something subtly different from what
the league played, `profile_replay` replays a league row's own stored action history
through the engine and counts. No policy is consulted during a replay, so the
profile is by construction a description of the games the league actually played,
and a test asserts the two players' move counts sum to the row's own ply count. The
privileged state is read only for the mover's own type and, once the game is over,
for which of a player's pieces the opponent legally learned — facts *about* a
finished game, never inputs to a decision.

**`strength_tiers` requires direction, not just separation.** Two policies share a
tier when their direct paired comparison does not separate them. A new tier opens
only at a policy that every member of the tier above both beats **and** separates
from. Without the direction requirement a perfect rock-paper-scissors league would
report three cleanly ordered tiers; with it, that league correctly reports one tier
and no ordering, and a test builds exactly that cycle. This is the same argument
Agent 3 made about Bradley-Terry in 3.5 — a single number cannot express
"indistinguishable" — carried into the function the gate actually reads.

#### `scripts/run_phase4_agent04.py`

Nine stages, each returning its own `problems` list, with `--quick` for a smoke run,
`--skip-pytest` for measurements only and `--stage` for one stage at a time. Two
details worth noting:

The harness **re-runs its own acceptance module after writing the data file**. The
suite that feeds the `test_suite_green` gate necessarily runs before the artefact
exists, so `test_the_published_acceptance_record_is_internally_consistent` could
only ever skip during that pass. A second, narrow pass runs once the file is on
disk, and a failure there downgrades the decision rather than being reported as a
green run with a broken data file.

The **tuning ablation is re-derivable, not quoted**. `LegacyExposureStrategic`
restores the 1.0.0 exposure arithmetic on top of the current code, so the evidence
for the one change this agent made can be regenerated at any time instead of resting
on a number in a report.

### 4.3 Files created and modified

Created:

```text
stratego/evaluation/calibration.py
scripts/run_phase4_agent04.py
tests/evaluation/test_phase4_acceptance.py
reports/phase_4_data/agent_04_calibration_security.json
reports/phase_4_data/agent_04_hidden_information_audit.json
reports/phase_4_data/agent_04_baseline_league_raw.csv
reports/phase_4_data/agent_04_pairwise_calibration.csv
reports/phase_4_data/agent_04_stress_behavior.csv
```

Modified:

```text
stratego/evaluation/baselines.py           (the exposure rule; policy_version 1.0.0 -> 1.1.0)
stratego/evaluation/__init__.py            (re-exports only; no existing symbol changed)
tests/evaluation/test_match_runner.py      (the deliberately pinned Strategic version)
reports/phase_4_implementation_report.md   (this section appended)
```

### 4.4 Tests run

```text
python -m pytest -q                                          1,963 passed, 2 skipped, 0 failed
python -m pytest -q tests/evaluation/test_phase4_acceptance.py   32 passed, 0 skipped
python scripts/run_phase4_agent04.py                         PASS
```

32 new tests. The 2 skips are Agent 2's pre-existing deliberate skips.

| Group | Tests | Covers |
|---|---|---|
| exposure recalibration | 6 | the corrected direction on sampled positions, silence when nothing answers the piece, both preconditions load-bearing, the term still firing, permutation invariance of `expected_defence_value`, the version bump |
| strength tiers | 7 | a total order giving one tier each, ties collapsing, reversed matchup orientation, a missing comparison reported, a non-transitive league not reported as a ladder, a perfect cycle giving one tier, determinism under input order |
| audit determinism | 6 | chunk trials summing exactly to target, decomposition independent of workers, chunk replay, serial-versus-pooled equality field by field, a short no-leak sweep that is not vacuous, merge adding rather than overwriting |
| position sampling | 4 | determinism and non-terminality for both sources, the two sources differing, an unknown source rejected |
| behaviour profiling | 8 | replay agreeing with the row's own ply count, refusal without a history, the Scout rush separating, rate arithmetic, identical profiles not "different", zero-versus-positive reading as full separation, degenerate entropy, own-relative direction buckets |
| published artefact | 1 | every required field present, headline agreeing with the gates, intervals inside `[0, 1]` on the paired unit, Random as the floor |

The one test in the last group is why the harness makes a second pytest pass: it is
the only test that reads the published decision, and it is worthless if it can only
run before that decision is written.

### 4.5 Measured results

#### 4.5.1 Hidden-information audit

100,000 valid trials across all ten catalogued policies. Every trial clones the
privileged state, permutes the true identities of the unresolved hidden opponent
pieces while preserving every public constraint, and asks each policy to decide on
both states with the **same** policy seed.

| Measurement | Value |
|---|---|
| Valid trials | **100,000** |
| Policy comparisons | **1,000,000** (10 policies x 100,000) |
| Score-vector comparisons | 800,000 (the 8 scoring policies) |
| Action mismatches | **0** |
| Diagnostic mismatches | **0** |
| Score-vector mismatches | **0** |
| `PublicView` mismatches | **0** |
| Legal-action-list mismatches | **0** |
| Positive-control trials / failures | 100,000 / **0** |
| Leak-detector failures | **0** |
| Positions skipped as unchanged | 5 |
| Games sampled | 12,076 |
| Hidden pieces permuted | 3,410,571 (34.1 per trial) |
| Chunks / wall clock | 32 / 120 s at 10 workers |

Trials split exactly 50,000 / 50,000 between `random_walk` and `baseline_play`, and
across the ten snapshot plies:

```text
ply     4      8     15     24     40     60     85    115    150    200
     11,774 11,613 11,377 11,081 10,592 10,034  9,439  8,807  8,116  7,167
```

The taper is games ending before the later checkpoints, not a sampling bias; the
thinnest phase still carries 7,167 trials.

Three things make the zero meaningful rather than vacuous. The **positive control**
requires the privileged belief target to differ, and it fired on all 100,000 trials.
The **leak detector** requires the hidden true types themselves to differ, proving
each pair of states is one a leaking policy really could distinguish. And the
comparison is **stronger than the argmax**: for every scoring policy the entire
per-move score vector is compared, since two vectors can share a maximum and differ
everywhere else.

Only 5 sampled positions were rejected as unchanged out of 100,005, which is why the
mean of 34.1 permuted hidden pieces per trial matters — a trial is permuting most of
the opponent army, not one or two pieces.

#### 4.5.2 The one policy change: the exposure term was misimplemented

Agents 2 and 3 both measured `tactical_rule_based` versus `strategic_rule_based` at
about 0.54 for Tactical with an interval containing 0.5, and both correctly declined
to call it. Agent 2 additionally reported that `exposure` was the only Strategic-only
term whose *removal* improved Strategic. That is the thread this agent pulled.

**The diagnosis.** At 1.0.0 the exposure penalty was proportional to the identified
piece's material value. Measured against the quantity it was meant to price — the
publicly deducible expected cost of being attacked by an unknown mover — the two run
in opposite directions:

| Piece | Material value | 1.0.0 penalty at `advance=1` | Expected defence value | 1.1.0 penalty at `advance=1` |
|---|---|---|---|---|
| Spy | 25 | −1.50 | **−24.04** | **−1.44** |
| Scout | 10 | −0.60 | −6.35 | −0.38 |
| Miner | 30 | −1.80 | −12.88 | −0.77 |
| Sergeant | 14 | −0.84 | +3.12 | 0.00 |
| Captain | 24 | −1.44 | +7.19 | 0.00 |
| Colonel | 46 | −2.76 | +13.50 | 0.00 |
| General | 70 | −4.20 | +17.88 | 0.00 |
| Marshal | 100 | **−6.00** | **+18.46** | **0.00** |

The old rule taxed the Marshal four times as hard as the Spy, while the Marshal is
the piece the opponent can least answer and the Spy is the piece that actually dies
once named. Worse, under `stratego_project_v1` a piece becomes identified by
**winning a fight** — so a value-priced penalty froze precisely the attackers that
had just proved they win. That is a misimplemented rule, not a mistuned constant.

**The fix**, in `StrategicRuleBasedPolicy.score`, prices the same term by
`context.expected_defence_value(piece_type)`, which is negative only while the
opponent still holds something that beats this type. The conceptual role is
preserved — an identified piece the opponent can answer is a target — and no weight
moved: `STRATEGIC_WEIGHTS.exposure` is still 0.06. `policy_version` went 1.0.0 ->
1.1.0, which is what made the change loud: it broke the version Agent 3 deliberately
pinned in `test_match_runner.py`.

**The ablation**, re-derived by the harness at 1,024 paired units against Tactical:

| Strategic variant | EWR vs Tactical | 95% paired interval | Separated | W/D/L | Mean plies |
|---|---|---|---|---|---|
| 1.0.0 exposure (material-priced) | **0.4565** | [0.4365, 0.4758] | yes | 906/58/1084 | 310 |
| 1.1.0 exposure (vulnerability-priced) | **0.5457** | [0.5273, 0.5647] | yes | 1102/31/915 | 235 |

Both intervals exclude 0.5, in opposite directions. The old Strategic was
significantly **weaker** than Tactical — a genuine strength inversion that 64 to 192
paired units could only render as "not separated". The improvement is +0.089, and
the drop from 310 to 235 mean plies is the mechanism visible in the outcome: the
unfrozen attackers finish games.

**Tuning iterations**, all three recorded in the data file:

| # | Change | Result |
|---|---|---|
| 1 | diagnosis only, no code change | exposure pricing confirmed anti-correlated with vulnerability |
| 2 | re-price exposure by `expected_defence_value`; no weight moved | 0.4565 -> 0.5457 vs Tactical over 1,024 units |
| 3 | swept `pressure` x1.75/x2.5, `mobility` x1.5, `miner_preservation` x2, and dropping the Strategic scout-information bonus | none adopted — every variant landed inside the seed-noise band |

Iteration 3 is the one worth reading. At 256 paired units the variants scored 0.507
to 0.578 against Tactical, which looks like a ranking. But behaviour-identical
replicas that differed **only** in `policy_version` — and therefore only in derived
policy seeds — spanned 0.529, 0.542 and 0.578 at that same sample size. The apparent
ranking was noise. Rerun at 1,024 units the replica spread fell to 0.011 or less,
and the exposure fix stood alone at eight times that. **One rule changed in total**;
nothing was tuned toward a target percentage.

#### 4.5.3 Screening league

45 matchups over all ten policies at 64 paired units: 5,760 matches, 58 s at 99.8
matches/s, mean 405 plies. **0 policy errors, 0 illegal actions, 0 ladder
inversions** detectable at that sample size.

Two things the screen found, both expected and both reported rather than suppressed:

- **7 pathological-draw matchups** (battleless-limit draws >= 90% of games), every
  one involving `stress_draw_seeker` or `stress_information_miser`. Neither policy
  attacks, and combat is the only way the battleless counter resets, so a game
  between two non-attacking policies *must* reach the limit. This is the stress
  suite working as designed, not a defect.
- **1 near-identical matchup**: `stress_draw_seeker` vs `stress_information_miser` at
  exactly 0.500 with a **zero-width** interval. Every game is a battleless draw, so
  every paired unit scores exactly 0.5 and the bootstrap correctly returns a point.
  It is the cleanest possible demonstration that the interval tracks the data rather
  than a formula.

The screen could not see the Tactical/Strategic inversion at 64 units, which is
consistent with 4.5.2 and is exactly why the tier decision runs at 1,024.

#### 4.5.4 Final calibration league

Paired colour/setup evaluation, `color_swap_same_board`, sampled by matchup class:
1,024 paired units on all six core-ladder pairs, 512 on the 24 ladder-versus-stress
pairs, 256 on the 15 stress-versus-stress pairs.

| Measurement | Value |
|---|---|
| Matches / paired units / matchups | 44,544 / 22,272 / 45 |
| Wall clock | 393 s at 113.5 matches/s, 10 workers |
| Total plies | 17,280,973 (mean 388, median 275, range 3-3,325) |
| Terminal reasons | flag capture 59.8%, opponent no legal move 21.1%, battleless draw 19.1%, both no legal move 0.05% |
| Policy errors / illegal actions | **0 / 0** |
| Schedule digest / results digest | `996cedca…` / `29cebdca…` |

The six core-ladder comparisons, 1,024 paired units each, 10,000 resamples over the
paired unit:

| Candidate | Opponent | EWR | 95% paired bootstrap | Normal | Separated | W/D/L |
|---|---|---|---|---|---|---|
| `random_legal` | `basic_heuristic` | 0.1140 | [0.1013, 0.1270] | [0.1011, 0.1269] | yes | 206/55/1787 |
| `random_legal` | `tactical_rule_based` | 0.0212 | [0.0156, 0.0273] | [0.0153, 0.0271] | yes | 36/15/1997 |
| `random_legal` | `strategic_rule_based` | 0.0139 | [0.0093, 0.0190] | [0.0091, 0.0187] | yes | 24/9/2015 |
| `basic_heuristic` | `tactical_rule_based` | 0.2454 | [0.2285, 0.2620] | [0.2287, 0.2620] | yes | 485/35/1528 |
| `basic_heuristic` | `strategic_rule_based` | 0.2380 | [0.2219, 0.2546] | [0.2214, 0.2547] | yes | 469/37/1542 |
| `tactical_rule_based` | `strategic_rule_based` | **0.4646** | **[0.4460, 0.4832]** | [0.4459, 0.4833] | **yes** | 935/33/1080 |

The last row is the one Phase 4 was waiting for: Strategic takes 0.5354
[0.5168, 0.5540] against Tactical, and the interval clears 0.5. Compare Agent 2's
0.542 for Tactical over 192 units and Agent 3's 0.547 over 64 — the direction was
consistent across every measurement, and only the sample size ever changed.

Colour splits, and why the pairing is inside the unit rather than averaged over:

| Matchup | EWR as red | EWR as blue | Red − blue | Unit histogram 0 / .25 / .5 / .75 / 1 |
|---|---|---|---|---|
| `random_legal` vs `basic_heuristic` | 0.1240 | 0.1040 | +0.020 | 773/51/190/4/6 |
| `basic_heuristic` vs `tactical_rule_based` | 0.2529 | 0.2378 | +0.015 | 533/32/431/1/27 |
| `basic_heuristic` vs `strategic_rule_based` | 0.2456 | 0.2305 | +0.015 | 549/29/417/4/25 |
| `tactical_rule_based` vs `strategic_rule_based` | 0.4834 | 0.4458 | +0.038 | 223/20/615/11/155 |

Every matchup favours red — first move is worth something in every one — and the
`tactical`/`strategic` gap of +0.038 is twice the half-width of that matchup's
interval. An unpaired schedule that happened to hand one policy red more often would
have moved the tier decision.

The unit histograms carry information an effective win rate cannot. In
`basic` vs `tactical`, 431 of 1,024 units split 0.5 — Basic wins one colour and loses
the other on the same board — against 533 swept. That is a very different result from
the same 0.2454 produced by uniform quarter-losses, and it says the board matters
more than the colour.

#### 4.5.5 Strength tiers

**4 tiers, fully ordered, 6 of 6 cross-tier pairs separated in the correct
direction.**

| Tier | Member | Pooled EWR over the ladder | Separated from the tier below |
|---|---|---|---|
| 1 | `strategic_rule_based@1.1.0` | 0.7611 | yes, [0.5168, 0.5540] vs Tactical |
| 2 | `tactical_rule_based@1.0.0` | 0.7327 | yes, [0.7380, 0.7715] vs Basic |
| 3 | `basic_heuristic@1.0.0` | 0.4565 | yes, [0.8730, 0.8987] vs Random |
| 4 | `random_legal@1.0.0` | 0.0497 | — (the floor) |

Pooled over all nine opponents each faced in the league:

| Policy | Games | W/D/L | Pooled EWR |
|---|---|---|---|
| `strategic_rule_based@1.1.0` | 12,288 | 9844/570/1874 | 0.8243 |
| `tactical_rule_based@1.0.0` | 12,288 | 9528/694/2066 | 0.8036 |
| `basic_heuristic@1.0.0` | 12,288 | 7122/398/4768 | 0.5958 |
| `random_legal@1.0.0` | 12,288 | 1491/2796/8001 | 0.2351 |

Bradley-Terry ratings over the whole 45-matchup league, MM algorithm, converged in
126 iterations with `prior_draws=1.0` — reported as a convenience ranking only:

| Rank | Policy | Rating | BT strength | Games |
|---|---|---|---|---|
| 1 | `strategic_rule_based@1.1.0` | 1808.8 | 5.9152 | 12,288 |
| 2 | `tactical_rule_based@1.0.0` | 1787.6 | 5.2357 | 12,288 |
| 3 | `basic_heuristic@1.0.0` | 1605.8 | 1.8384 | 12,288 |
| 4 | `stress_miner_rush@1.0.0` | 1521.6 | 1.1325 | 6,656 |
| 5 | `stress_scout_rush@1.0.0` | 1467.1 | 0.8275 | 6,656 |
| 6 | `stress_berserker@1.0.0` | 1387.7 | 0.5239 | 6,656 |
| 7 | `stress_information_miser@1.0.0` | 1385.2 | 0.5163 | 6,656 |
| 8 | `stress_chaos@1.0.0` | 1374.9 | 0.4868 | 6,656 |
| 9 | `stress_draw_seeker@1.0.0` | 1339.8 | 0.3976 | 6,656 |
| 10 | `random_legal@1.0.0` | 1321.5 | 0.3580 | 12,288 |

The rating gap between Strategic and Tactical is 21 points. On Agent 3's sparse
4-edge graph the same table put Tactical *above* Strategic by 32 points. Neither
number is evidence; the paired interval is. This table is in the report because a
long list needs an order, not because it decides anything.

#### 4.5.6 Stress-policy characterisation

Profiled by replaying 96 paired units of every one of the 45 league matchups — a
prefix subset of the final league carrying the same `match_id`s — giving every policy
1,728 games against nine opponents. Both sides of every game are counted.

| Policy | Role | Mean plies | Attack | Scout | Miner | Reveal | Battleless draw | Flag-capture win | Piece entropy | Movement entropy |
|---|---|---|---|---|---|---|---|---|---|---|
| `random_legal` | ladder | 709 | 0.046 | 0.105 | 0.151 | 0.730 | 0.292 | 0.046 | 3.22 | 1.74 |
| `basic_heuristic` | ladder | 268 | 0.212 | 0.113 | 0.150 | 0.700 | 0.041 | 0.518 | 3.21 | 1.24 |
| `tactical_rule_based` | ladder | 286 | 0.187 | 0.128 | 0.175 | 0.595 | 0.073 | 0.620 | 3.18 | 1.36 |
| `strategic_rule_based` | ladder | 281 | 0.197 | 0.159 | 0.166 | 0.611 | 0.057 | 0.668 | 3.17 | 1.59 |
| `stress_scout_rush` | stress | 360 | 0.073 | **0.766** | 0.035 | 0.522 | 0.461 | 0.135 | **1.51** | 1.78 |
| `stress_miner_rush` | stress | 357 | 0.073 | 0.031 | **0.784** | 0.498 | 0.459 | 0.176 | **1.41** | 1.58 |
| `stress_draw_seeker` | stress | 449 | **0.000** | 0.215 | 0.092 | 0.480 | **0.559** | **0.000** | 3.04 | **0.65** |
| `stress_berserker` | stress | **242** | **0.311** | 0.098 | 0.163 | 0.767 | **0.000** | 0.293 | 3.20 | 1.09 |
| `stress_information_miser` | stress | 490 | **0.000** | 0.127 | 0.054 | 0.520 | 0.461 | **0.000** | 3.13 | 1.53 |
| `stress_chaos` | stress | 601 | 0.080 | 0.101 | 0.103 | 0.690 | 0.344 | 0.037 | **3.24** | 1.74 |

**6 of 6 stress policies are materially different** from the Strategic baseline, on
between four and six of the nine compared metrics each:

| Policy | Largest divergence | Metrics beyond the 0.35 relative threshold |
|---|---|---|
| `stress_scout_rush` | battleless draw rate (0.88) | attack, battleless draw, flag capture, Miner, piece entropy, Scout |
| `stress_miner_rush` | battleless draw rate (0.88) | attack, battleless draw, flag capture, Miner, piece entropy, Scout |
| `stress_draw_seeker` | flag-capture rate (1.00) | attack, battleless draw, flag capture, game length, Miner, movement entropy |
| `stress_berserker` | battleless draw rate (1.00) | attack, battleless draw, flag capture, Scout |
| `stress_information_miser` | attack rate (1.00) | attack, battleless draw, flag capture, game length, Miner |
| `stress_chaos` | flag-capture rate (0.94) | attack, battleless draw, flag capture, game length, Miner, Scout |

The extremes are the point: `berserker` attacks on 31.1% of moves and never draws by
the battleless limit, at the same time as `draw_seeker` and `information_miser`
attack on **0.000%** and draw 55.9% and 46.1% of the time. `scout_rush` and
`miner_rush` collapse piece-type entropy from 3.17 bits to 1.51 and 1.41 — they play
essentially one piece type. `draw_seeker` also has the lowest movement entropy at
0.65 bits, which is what "shuffle in place" looks like as a number.

For contrast the two ladder policies checked against the same threshold behave as
they should: `tactical_rule_based` diverges from Strategic on **nothing** (largest
0.22) and `basic_heuristic` on **nothing** (largest 0.29). The threshold is not
firing on ordinary variation between reasonable policies — which is what makes the
6/6 result mean something.

`random_legal` does cross the threshold on four metrics. It is a ladder policy, not a
stress policy, and the divergence is real rather than an artefact: at 709 mean plies
and a 0.046 flag-capture rate it genuinely plays differently from every other policy
in the suite.

#### 4.5.7 Final reproducibility

Four matchups chosen to span the suite — an adjacent ladder pair, the pair the tier
gate turns on, a ladder-versus-stress pair and a draw-heavy stress pair — at 64
paired units, 512 matches, run five times.

| Execution | Chunks | Seconds | Speedup | Results digest | Replay digest | Mismatches |
|---|---|---|---|---|---|---|
| workers=1 (serial) | 1 | 46.11 | 1.00x | `9aa54492…` | `c0be7096…` | 0 |
| workers=2 | 8 | 23.96 | 1.92x | `9aa54492…` | `c0be7096…` | 0 |
| workers=4 | 16 | 12.39 | 3.72x | `9aa54492…` | `c0be7096…` | 0 |
| workers=8 | 32 | 6.68 | 6.90x | `9aa54492…` | `c0be7096…` | 0 |
| workers=8, order shuffled | 32 | 7.15 | — | `9aa54492…` | `c0be7096…` | 0 |

**1 distinct results digest and 1 distinct replay-digest set across all five
executions**, and 0 field-level mismatches over 2,048 compared matches. The replay
digests are checked as their own statement rather than left implied by the field
comparison passing, because the instructions ask for both.

#### 4.5.8 Checkpoint readiness

`CheckpointShapedPolicy` declares `observation=True` and `legal_action_mask=True`,
receives the `observation_v2_1_127ch` tensor, projects it to a score per action
identifier through a fixed seeded projection, masks the illegal identifiers and takes
the argmax. It is the shape of a trained network's `decide` with the model replaced
by a constant — so the harness is exercised end to end without any of the
neural-network work Phase 4 excludes. It is deliberately not catalogued and is not an
opponent.

| Opponent | Paired units | Probe EWR | 95% interval | Mean plies | W/D/L |
|---|---|---|---|---|---|
| `random_legal` | 64 | 0.4805 | [0.4570, 0.5039] | 1,283 | 3/117/8 |
| `basic_heuristic` | 64 | 0.0938 | [0.0586, 0.1367] | 357 | 5/14/109 |
| `tactical_rule_based` | 64 | 0.1367 | [0.0977, 0.1797] | 445 | 1/33/94 |
| `strategic_rule_based` | 64 | 0.1016 | [0.0703, 0.1367] | 394 | 0/26/102 |

**0 illegal actions and 0 reproduction mismatches.** The win rates measure nothing
about Stratego; what they establish is that a tensor-consuming policy schedules,
plays, scores, and reproduces from its `match_id` exactly like a rule-based one. The
0.4805 against `random_legal` is 117 draws out of 128 — a deterministic policy and a
uniform one reach the battleless limit at 1,283 mean plies — which is itself a useful
warning that a fresh checkpoint will look like a draw machine against Random before
it looks like anything else.

### 4.6 Deviations and limitations

1. **The league is sampled by matchup class, not uniformly at the maximum.** 1,024
   paired units on all six core-ladder pairs, 512 on ladder-versus-stress, 256 on
   stress-versus-stress. The instructions ask for at least 512 units on "important
   adjacent-tier comparisons" and explicitly permit a league that does not evaluate
   every pair at the maximum. A uniform 1,024 would have been 92,160 matches instead
   of 44,544 — double the runtime and a 62 MB raw artefact instead of 30 MB — to buy
   tighter intervals on pairs no acceptance gate reads. Every class size is recorded
   under `league.units_by_class`, so the reduction is reported rather than applied
   silently. Consequence: several stress-versus-stress matchups do not separate from
   even, which is expected at 256 units and is not part of any gate.

2. **A paired interval conditions on the policy seeds it was computed from.** Every
   catalogued policy is stochastic, and a match's policy seeds derive from its
   `match_id`, which names the policy version — so two behaviourally identical
   policies differing only in version string play different games. Measured: at 256
   paired units, replicas differing *only* in `policy_version` scored 0.529, 0.542
   and 0.578 against the same opponent, with 95% intervals that barely overlapped. The
   bootstrap resamples boards, not seeds, so it cannot express that component. At
   1,024 units the same replica spread fell to <= 0.011. **Anyone evaluating a future
   checkpoint should treat a ~0.03 difference at 256 units as noise.** This is the
   single most transferable finding in this section, and it is why iteration 3 of the
   tuning record adopted nothing.

3. **Behaviour is profiled on a 96-unit prefix of each matchup, not the whole
   league.** Profiling by replay needs the action histories in memory and the full
   league is 44,544 games. 96 paired units across all 45 pairs gives each policy 1,728
   games against nine opponents, and because the prefix reuses the same setup-pair
   identifiers it is a genuine subset of the final league rather than a separate run.
   Consequence: the behavioural table and the league table are computed from
   different sample sizes, so effective win rates in 4.5.6 will not match 4.5.5
   exactly.

4. **The 0.35 relative threshold in `behavior_divergence` is a blunt line.** It is
   not derived from a null distribution; it is a stated cut with the raw numbers
   printed beside every verdict. Its calibration is checked in the only way available
   — the two ladder policies nearest the reference fail to cross it (largest 0.22 and
   0.29) while all six stress policies clear it on four or more metrics — but a
   different threshold would move a borderline policy, and none of the six is
   borderline.

5. **The parallel runner resolves policies through the catalogue, so an unregistered
   policy can only run serially.** This is Agent 3's design and it is right — it stops
   a re-versioned policy from silently playing games recorded under an old
   identifier — but it is a real constraint: both the tuning ablation and the
   checkpoint probe run serially because of it, and the ablation's 4,096 matches are
   most of the two minutes that stage costs.

6. **The legacy ablation restores the 1.0.0 exposure arithmetic on top of the 1.1.0
   code rather than checking out the old file.** Verified equivalent on 981 scored
   moves across eight sampled positions, to floating-point equality against 1.0.0's
   block computed independently. What remains is cosmetic: the exposure entry appears
   last in the diagnostics component tuple instead of in its original position, which
   cannot affect a score or a ranking.

7. **Tactical and Strategic separate, but not by much.** 0.5354 [0.5168, 0.5540].
   The two are a deliberate nesting — Strategic *is* Tactical plus longer-horizon
   terms — so a large gap was never the goal and would have been suspicious. The
   separation rests on 1,024 paired units and would not have been visible at 256.

8. **The setup-bank bias Agent 1 documented is inherited, not addressed.**
   `structured_v1` is a hand-coded structural family, not a sample from the space of
   legal setups. No unbiased control family was added, because every ladder
   comparison is paired on the *same* boards, so a bank bias cancels within a
   matchup. It would matter for an absolute claim about a policy's strength, which
   Phase 4 does not make.

9. **No neural-network work and no use of the Phase 3 representative Transformer.**
   The checkpoint-readiness evidence is a fixed-projection policy, not a model. PyTorch
   is installed in the environment for Phase 3's suite but this agent imports nothing
   from it.

10. **The screen's pathological-draw and near-identical findings are reported, not
    resolved.** Seven matchups draw by the battleless limit in >= 90% of games and one
    pair is statistically identical at exactly 0.500. Both follow necessarily from
    two non-attacking policies meeting, and neither is a defect to fix — but a future
    league that adds a passive policy should expect its stress-versus-stress cells to
    be uninformative.

### 4.7 Data files

```text
reports/phase_4_data/agent_04_calibration_security.json     the acceptance record
reports/phase_4_data/agent_04_hidden_information_audit.json  the audit, standalone
reports/phase_4_data/agent_04_baseline_league_raw.csv        44,544 rows, 30 MB
reports/phase_4_data/agent_04_pairwise_calibration.csv       45 matchups with intervals
reports/phase_4_data/agent_04_stress_behavior.csv            10 policies x 16 metrics
```

The primary JSON holds status and all completion gates, every version identifier, the
prerequisite verification including Agent 1's bank digest, the full audit with its
per-ply, per-source and per-policy histograms, the tuning record with the ablation and
the exposure pricing table, the screen, the complete league summary with every
matchup's interval, colour split, terminal frequencies and unit histogram, the tier
partition, the Bradley-Terry table, all ten behavioural profiles with their
divergences, the reproducibility executions, the checkpoint gauntlet, both test
summaries and the environment record.

The audit is written separately as well because it is the artefact a future run
compares against: `run_hidden_information_audit(100_000, workers=N)` reproduces it
byte for byte for any `N`.

The raw league CSV carries both setups per row, so any of the 44,544 games can be
rebuilt and replayed with no setup bank present. Replay histories are **not** stored
for the league — 30 MB of rows against an estimated 100 MB of histories — which
follows Agent 3's recorded storage decision; each row keeps its `replay_digest`, so a
divergence stays detectable at full strength and only *inspecting* one needs a rerun.

### 4.8 Handoff notes

**Evaluating a future checkpoint.**

- `build_gauntlet_schedule(candidate_id, LADDER_POLICY_IDS, units)` is the shape;
  `require_valid_schedule(schedule, bank)` before starting, then
  `run_schedule(...)`, then `summarize_run(results, resamples=10_000, seed=...)`.
- **Register the policy in `registry.py` before evaluating it in parallel.** Workers
  resolve policies through the catalogue. `CheckpointShapedPolicy` in
  `scripts/run_phase4_agent04.py` is a worked example of what a tensor-consuming
  policy declares and how it masks.
- **Use 1,024 paired units for anything you intend to cite, and treat 256 as
  screening.** See limitation 2: seed-only noise at 256 units is larger than the gap
  the tier gate rests on.
- **Read `strength_tier_count`, not the Bradley-Terry rating.** The tier partition
  requires separation *and* direction; the rating table has twice now ordered two
  policies more confidently than their direct interval supports.
- Expect a fresh checkpoint to draw heavily against `random_legal` before it
  loses to anything — 117 draws in 128 games for the probe — so read the ladder
  from `basic_heuristic` upward early on.

**Standing regressions worth keeping.**

- `run_hidden_information_audit(100_000, workers=N)` is machine-independent and
  compares directly against `agent_04_hidden_information_audit.json`. **A new policy
  must be added to `registry.py` to enter it** — the audit parametrises over the
  catalogue, so an uncatalogued policy is silently absent.
- Regenerating the 1,024-pair bank and comparing to Agent 1's digest costs 0.24 s.
- `tests/evaluation/test_phase4_acceptance.py` pins the exposure direction. A future
  edit that reprices exposure by material value fails there in seconds instead of
  costing another 15-minute league to rediscover.

**If a policy's weights or rules change**, bump `policy_version` and recalibrate.
Every stored `match_id` names the policy by `id@version`, `resolve_policies` refuses
a mismatch, and `test_match_runner.py` pins the Strategic version deliberately so the
bump cannot be silent. That mechanism worked exactly as Agent 3 designed it: the one
change made here broke that test immediately.

**Recommended project-document updates** (for the post-acceptance review;
`stratego_project_docs/` was not modified):

1. `05_project_plan.md` — record `phase4_calibration_v1`, the four calibrated tiers
   with their measured separations, and that `strategic_rule_based` is at 1.1.0 after
   the exposure recalibration.
2. `05_project_plan.md` — record that the paired unit is the resampling unit, and
   that a paired interval conditions on the realised policy seeds, so a sample size
   must be chosen against seed noise as well as board noise. Include the measured
   number: ~±0.025 at 256 units, <= 0.011 at 1,024.
3. `06_observation_v2_127ch.md` — note that a checkpoint-shaped policy consuming the
   127-channel tensor has been driven through the full evaluation path with 0 illegal
   actions and 0 reproduction mismatches.
4. A note that the hidden-information audit is reproducible from
   `(root_seed, target_trials, sources, plies, policy_ids, chunks)` and independent of
   worker count, making it a standing regression rather than a one-off Phase 4
   measurement.

```text
Phase 4 decision: PASS
Baseline tiers established: 4
Hidden-information audit trials: 100000
Hidden-information mismatches: 0
Evaluation harness ready for future checkpoints: yes
```

```text
Phase 4 = COMPLETE
```
