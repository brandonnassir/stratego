# Phase 7 Implementation Report

Production Stratego setup generator and setup library (`setup_library_v1`)
for the Phase 8 synthetic warm-start pipeline.

Phase 7 executes against the frozen post-Phase-6 stack: rules
`stratego_project_v1`, reference engine `phase2_1_reference_1.2.0`,
observation `observation_v2_1_127ch`, engine action encoding
`source_destination_10000_v1` in absolute engine squares, model contract
`model_contract_v2` (C1 primary, C0 fallback), backend `KEEP_PYTHON`,
trajectory `trajectory_v1`, and the untouched Phase 4 evaluation bank
`evaluation_setup_bank_v1`. Phase 7 is a setup-distribution phase: no
neural-network training, no outcome-based setup selection, and no engine
modification is authorized anywhere below.

## 1. Agent 1 — Setup Contract, Taxonomy, and Diversity Standard

**Status: PASS** — 11 / 11 completion gates true. Machine-readable record:
`reports/phase_7_data/agent_01_setup_contract.json` (contract + gates) and
`reports/phase_7_data/agent_01_diversity_thresholds.json` (frozen diversity
standard). Every threshold below was frozen while
`data/setups/setup_library_v1.jsonl` did not exist, and the artifact records
that fact (`frozen_before_generation: true`).

### 1.1 Prerequisite verification

Verified from the repository rather than assumed:

| Check | Required | Found |
|---|---|---|
| Phase 6 formally accepted | `PASS` | `agent_06b_final_decision.json` status `PASS`; report §6B-3 "SAFE TO FORMALLY CLOSE"; §6B-4 anti-leak cleanup `PASS` (107,750 ≥ 100,000 valid trials) |
| Reference engine | `phase2_1_reference_1.2.0` | `stratego/engine/constants.py` `IMPLEMENTATION_VERSION` |
| Rules version | `stratego_project_v1` | `RULES_VERSION`, unchanged |
| Phase 4 bank untouched | yes | `evaluation_setup_bank_v1` code and artifact unmodified by this agent |
| Production library absent | yes | `data/setups/` does not exist yet |

Pre-existing suite, measured at commit `9d8a950` **before any Phase 7 edit**:

```text
python -m pytest -q
2732 passed, 3 skipped, 0 failed in 198.84s
```

Identical to the totals recorded at Phase 6 formal closure (2732/3/0). The
three skips are the two pre-existing Phase 4 capability skips plus the
PASS-gated Phase 6B artifact check, all pre-dating Phase 7.

### 1.2 Contract versions

```text
setup_generator_contract_v1    generation/identity/serialization contract
setup_library_v1               the production library this contract defines
setup_family_v1                the 16 frozen family contracts
setup_trait_vector_v1          the 35-field structural trait schema
setup_diversity_standard_v1    metrics + numeric thresholds
master seed                    20260813   (date-seed, Phase 4 precedent)
```

A future semantic change to any of these requires a new version identifier.
Once this section reports PASS, Agents 2–6 must treat every definition and
threshold here as frozen: no weakening to make generation easier, no
reinterpretation after seeing the library, no post-hoc threshold movement.

### 1.3 Canonical representation

The library uses the canonical own-orientation frame already established by
the accepted Phase 4 evaluation bank:

```text
rank 0             own back row (furthest from the lakes)
rank 3             own front row (nearest the lakes)
file 0..9          absolute board column, left to right
canonical index    rank * 10 + file
storage            40 engine piece types, canonical row-major order
serialization      engine 40-character piece-code string
                   (stratego.engine.setup.serialize_setup)
orientation map    identity for red; rank-row reversal for blue
                   (stratego.setups.identity.orient_setup)
```

The frame helpers are reimplemented in `stratego/setups/identity.py` so the
production library does not import the frozen evaluation fixture;
`tests/setups/test_identity.py` pins the two implementations to each other
exhaustively (all 40 cells, all neighbour sets, both players' orientation
maps), so exactly one convention exists. No competing board-orientation
convention was invented.

### 1.4 Reflection and canonicalization

```text
reflection          file f -> 9 - f within every rank
                    (delegates to the frozen engine's reflect_setup)
involution          reflect(reflect(s)) == s        (tested, 0 failures)
class rule          {s, reflect(s)} is ONE base-library identity
representative      the lexicographically smaller piece-type tuple
fixed points        none exist: the single Flag cannot sit on file f and
                    file 9-f at once, so every class has exactly 2 members
```

Pinned representative examples (all tested): left corner ↔ right corner,
file 0 ↔ file 9, file 4 ↔ file 5, Flag/Bomb adjacency preserved, rank rows
unchanged, inventory unchanged. Agent 2 must materialize only canonical
representatives; Agent 3 recomputes canonicalization independently and counts
any stored non-representative as a hard failure.

### 1.5 Stable identity and fingerprints

```text
semantic id         setup_library_v1:<family>:<index %03d>
                    e.g. setup_library_v1:F07:042
class fingerprint   SHA-256("stratego_setup_class_v1:" + serialized
                    class representative) — shared by both class members;
                    THE duplicate/leakage identity
content fingerprint SHA-256("stratego_setup_content_v1:" + serialized
                    orientation) — distinguishes s from reflect(s)
golden anchor       sorted-inventory arrangement content fingerprint
                    97d88a98ac069373…  (pinned in test_identity.py)
```

Identity never depends on process enumeration order, and the package never
calls Python's process-randomized built-in hash (statically tested).

### 1.6 Seeding and isolated regeneration

```text
base seed       blake2b(person "strat-lb7") over
                "contract:library:master_seed:family_id:base_index"
attempt seed    blake2b(person "strat-at7") over "base_seed:attempt"
stream seed     blake2b(person "strat-st7") over "purpose:parts…"
                (for Agent 4's perturbation streams)
```

Generation for a base identity draws candidate attempts `0, 1, 2, …` and
accepts the first candidate satisfying the whole contract (family predicate,
legality, initial mobility), counting rejections. Rejection is therefore
local to the base identity: **no accepted setup may condition on any other
base's outcome**, which is what makes isolated regeneration exact:

```text
contract + library version + master seed + family id + base index
    -> exact same base setup and metadata, no prefix generation
```

Cross-base requirements (duplicates, distances) are acceptance gates, not
generation inputs; violating one under the frozen contract is a BLOCKED
finding, never a licence to reroll. The personalization tags differ from the
Phase 4 bank's (`strat-bnk`/`strat-sid`), so library streams can never
replay evaluation-bank randomness (tested).

Agent 2 must prove isolated rebuild for the fixed per-family sample
`{0–9, 395–404, 445–454, 490–499}` — 40 indices per family, 640 total,
covering both split boundaries.

### 1.7 Split rule

```text
base_index 0..399    train        400/family    6,400 total
base_index 400..449  validation    50/family      800 total
base_index 450..499  test          50/family      800 total
```

The split is a pure function of the base index, decided before any setup
exists, so it can never react to content, game strength, or model results.
Reflections and perturbed descendants inherit the base split verbatim; a
train-derived setup can never enter validation/test by reflection or
perturbation. Split identity is recorded in every base entry and is
recoverable during isolated regeneration without any lookup table.

### 1.8 Initial-mobility quality rule

Every accepted base setup and every generated descendant must have at least
one legal move for its owner in initial board geometry. This is a
**library acceptance criterion**, not a rules change: `stratego_project_v1`
and the corrected engine's handling of setups that are terminal at creation
are untouched, and no global setup restriction or reroll is introduced.

The executable check is `stratego.setups.mobility.setup_has_initial_mobility`:
it builds a real initial game through the frozen engine (`create_game`, the
arrangement mirrored for both colours) and asks the engine's own
`has_legal_action` — the same authority the engine's mobility-termination
rule consults. The module contains no movement geometry of its own
(statically tested), initial mobility is provably independent of the
opponent's arrangement (attack legality depends only on ownership), and the
verdict is reflection-invariant (the lake-facing file set is mirror-
symmetric; tested). A stranded fixture is detected by the check **and**
produces a game the 1.2.0 engine declares terminal at creation
(`both_no_legal_move_draw`), confirming one interpretation of the rule.
Agents 2–4 must use this function and count every stranded rejection.

### 1.9 The 16 family contracts

Every family is a measurable structural contract over the trait vector —
never "whatever the generator branch produced". A setup satisfies a family
iff every required clause is true and every forbidden clause is false. All
clauses reference only reflection-invariant traits (tested), so membership
is a property of the reflection class. Families overlap by design; each
library setup declares exactly one primary family whose contract is a hard
requirement, and Agent 3 reports cross-family satisfaction as a confusion
matrix. Full machine-readable clauses (with independently re-evaluable
expressions) live in the contract artifact.

| ID | Key | Required core (abridged) | Forbidden |
|---|---|---|---|
| F00 | corner_flag_fortress | Flag rank 0, corner file; both orthogonal neighbours Bombs | — |
| F01 | near_corner_flag_fortress | Flag rank 0, edge distance 1–2; ≥ 2 orthogonal Bomb guards | — |
| F02 | central_back_flag_fortress | Flag rank 0, edge distance ≥ 3; ≥ 2 orthogonal Bomb guards | — |
| F03 | partially_bombed_flag | Flag rank ≤ 1; exactly 1 orthogonal Bomb guard | ≥ 4 Bombs within Chebyshev 2 of Flag |
| F04 | lightly_defended_deceptive_flag | Flag rank ≤ 1; 0 orthogonal and 0 diagonal Bomb guards | ≥ 3 Bombs within Chebyshev 2 of Flag |
| F05 | false_fortress_bomb_decoy | Flag rank ≤ 1, ≤ 1 guard; decoy pocket: movable piece in ranks 0–1 at Manhattan ≥ 4 from Flag with ≥ 2 orthogonal Bombs | — |
| F06 | distributed_bomb_defense | Flag rank ≤ 1; Bombs on ≥ 5 distinct files; 0 orthogonally adjacent Bomb pairs | ≥ 2 orthogonal Flag guards |
| F07 | high_bomb_placement | Flag rank ≤ 1; ≥ 4 Bombs in ranks 2–3 | — |
| F08 | aggressive_high_rank_front | Flag rank ≤ 1; ≥ 5 of 7 rank≥7 pieces in ranks 2–3; Marshal and General both forward | — |
| F09 | conservative_high_rank_rear | Flag rank ≤ 1; ≥ 5 of 7 rank≥7 pieces in ranks 0–1; Marshal and General both back | — |
| F10 | scout_forward_information | Flag rank ≤ 1; ≥ 6 of 8 Scouts in ranks 2–3; ≥ 3 on the front rank | — |
| F11 | scout_preservation | Flag rank ≤ 1; ≥ 5 Scouts in ranks 0–1; 0 on the front rank | — |
| F12 | miner_forward | Flag rank ≤ 1; ≥ 3 of 5 Miners in ranks 2–3 | — |
| F13 | miner_preservation | Flag rank ≤ 1; ≥ 4 Miners in ranks 0–1; 0 on the front rank | — |
| F14 | balanced_conventional | Flag rank 0 with ≥ 2 guards; Marshal/General off the front rank; ≥ 3 forward Scouts; ≤ 2 front-rank Bombs; ≥ 2 reserved Miners; ≥ 8 movable front-rank pieces | — |
| F15 | irregular_high_entropy | ≥ 2 of 8 fixed unconventional-structure features | back-rank Flag with ≥ 2 orthogonal guards |

"High rank" means combat rank ≥ 7 from the engine's rank table (Major,
Colonel, General, Marshal — 7 pieces); Flag/Bomb are identified by their
piece-type constants, never enum ordinals. F15's eight unconventional
features are frozen in the trait schema (Flag forward / Flag unguarded /
≥ 3 front-rank Bombs / Marshal, General, or Spy on the front rank / no
front-rank Scouts / ≥ 3 front-rank Miners). F15 deliberately permits any
Flag rank including the front; F00–F02 and F14 pin the Flag to the back
rank; every other family requires the back two ranks.

Executability was proven with one hand-constructed positive fixture per
family (legal, mobile, contract-satisfying, reflection-invariant) and one
negative fixture per family (legal but contract-violating), all 16 + 16
green in `tests/setups/test_families.py`.

### 1.10 Structural trait vector

`setup_trait_vector_v1`: 35 deterministic fields computed from the canonical
arrangement alone — Flag location/edge distance/guards/zone density, Bomb
rank histogram + front/back/file-dispersion/adjacency/mean-pairwise-distance,
Scout and Miner rank histograms + front/back splits, high-rank histogram,
Marshal/General/Spy ranks, front-rank movability (total, immovable, and
open-file movable counts), decoy-pocket strength, three rank-entropy scalars,
and the unconventional-feature count. Every field has declared units and a
reflection-invariance flag; only `flag_file` is orientation-specific (it
mirrors under reflection — tested), and family clauses may not reference it.
The four float fields are rounded to 6 decimals at computation, so trait
equality is exact value equality. No trait involves playing strength or
neural outputs. Full schema: `trait_schema` section of the contract artifact.

### 1.11 Diversity metrics and frozen thresholds

Distance is `class_distance(a, b) = min(H(a,b), H(a,reflect(b)))` — Hamming
over the 40 canonical squares, well-defined on reflection classes. Support
metrics count "folded" cells `(rank, min(file, 9-file))` — 20 possible —
so positional coverage is reflection-invariant. All metrics are implemented
and tested now (`stratego/setups/diversity.py`), executable by Agent 3 on
the full 8,000-base library (vectorized; the full 8,000² distance audit is
a few seconds of numpy).

Hard identity/quality requirements (all zeros):

```text
exact duplicate classes                 0
reflection-equivalent duplicates        0
cross-split class duplicates            0
stored non-canonical representatives    0
stable-ID collisions                    0
fingerprint collisions                  0
engine-invalid bases                    0
stranded bases                          0
primary-family contract violations      0
family self-satisfaction                1.0 exactly (diagonal)
```

Frozen numeric floors:

```text
within-family nearest-neighbour class distance    >= 6
within-family pairs at class distance < 10        <= 0.1%
cross-split nearest-neighbour class distance      >= 8   (whole library)
global minimum pairwise class distance            >= 4
family mean per-square entropy                    >= 1.0 bits
global mean per-square entropy                    >= 1.5 bits  (ceiling ~3.28)
Bomb folded support per family                    >= 10 of 20
Scout folded support per family                   >= 8
Miner folded support per family                   >= 6
high-rank folded support per family               >= 6
Flag folded support per family                    family-specific:
    F00 1 · F01 2 · F02 2 · F14 3 · F03–F13 4 · F15 8
distinct trait vectors per family                 >= 250 of 500
distinct Bomb rank histograms per family          >= 8
distinct Scout rank histograms per family         >= 8
family overlap matrix off-diagonal                report-only
```

Rationale, recorded in the thresholds artifact: independent structured draws
concentrate near class distance ~30/40, so the floors (6/8/4) are far below
honest generation but immediately catch template repetition with cosmetic
swaps — the failure this standard exists to detect. Entropy/support floors
deliberately do **not** demand uniform randomness: families legitimately pin
strategically defining cells (F00 pins three), which is why the Flag-support
floor is family-specific and the entropy floor sits at 1.0 of ~3.28 bits.
The same identity/quality/family/split rules extend to the Agent 4
procedural stress corpus: ≥ 100,000 samples, zero legality / inventory /
stranded / family / split-leak failures, perturbation class distance within
[2, 12] of the base.

Detector correctness is itself tested: planted exact duplicates, mirrored
near-copies, cross-split leaks, and non-canonical entries are each caught by
the named check, and an honest random collection clears every statistical
floor while failing exactly the family self-satisfaction checks.

### 1.12 Perturbation invariants (for Agent 4, frozen now)

A perturbed descendant must preserve: exact inventory (engine-validated),
setup-zone occupancy, canonical orientation semantics, the base's split, the
base's primary family (all required clauses true, all forbidden false), the
Flag's exact cell, initial mobility, and class distance to its base within
`[2, 12]` of the 40 squares. It must be a pure function of
`(base_setup_id, sampler_version, perturbation_seed)`, carry full provenance
(base id, split, family, reflection flag, seed), and never becomes a new
base-library identity. An unchanged output must be recorded as unperturbed,
not as a perturbation. Everything else — non-Flag placements, non-defining
traits — may vary within those bounds. The executable checker
`validate_perturbation` returns the exact violation list and is tested
against passing, identity, excessive, Flag-moving, family-breaking,
inventory-corrupting, and stranding candidates (the stranding case isolated
via a family-valid F07 arrangement whose only violation is mobility). No
learned perturbation policy is prescribed or permitted.

### 1.13 Serialization and artifact contract

```text
data/setups/setup_library_v1.jsonl           one entry per line, families
                                             F00..F15, index ascending
data/setups/setup_library_v1_manifest.json   versions, master seed, counts,
                                             split map, digests
entry fields    base_setup_id, library_version, contract_version,
                family_contract_version, trait_schema_version, family_id,
                family_key, base_index, split, canonical_setup (40-char),
                fingerprint, generation_seed, generation_attempts,
                trait_vector
line format     json.dumps(entry, sort_keys=True, separators=(",", ":"))
library digest  SHA-256 over newline-joined "base_setup_id:fingerprint"
                lines in file order
```

Byte-determinism of the JSONL follows from the canonical line format, so
Agent 6's full regeneration must reproduce the file digest exactly.
Provenance is training/debug metadata only: it must never cross the
observer-safe model boundary, `observation_v2_1_127ch` and `trajectory_v1`
are unchanged, and the preferred provenance design remains a per-game
sidecar keyed by `game_id` (Agent 5's deliverable).

### 1.14 Tests

The acceptance harness (`scripts/run_phase7_agent01.py`) re-exercises every
fixture, identity rule, and the end-to-end diversity standard before writing
the artifacts, and the repository suite gained 166 tests in `tests/setups/`:

```text
test_identity.py     canonical frame pinned to the Phase 4 convention
                     (exhaustive), reflection involution, class
                     representative rules, fingerprint determinism +
                     golden anchor, seed derivation isolation, no
                     Phase 4 stream collision
test_mobility.py     stranded/mobile verdicts, direct create_game +
                     has_legal_action agreement, 1.2.0 terminal-at-creation
                     agreement, reflection invariance, no competing
                     movement implementation (static)
test_traits.py       schema completeness/order, determinism, hand-computed
                     values, histogram inventories, reflection invariance
                     of every invariant field, flag_file mirror law
test_families.py     16 IDs exactly once matching the instruction table,
                     positive/negative/reflected fixtures for all 16,
                     invariant-trait-only clause audit, clause algebra,
                     JSON round-trip, independent re-evaluation of
                     serialized expressions
test_contracts.py    exact counts/splits/boundaries, ID round-trips,
                     digest determinism, isolated-rebuild sample,
                     perturbation validator behaviour, artifact/code
                     consistency, no built-in hash (static)
test_diversity.py    distance hand cases, planted-violation detection,
                     entropy/support/trait metrics, threshold
                     serialization, end-to-end standard execution
```

```text
python -m pytest -q       (before edits)   2732 passed, 3 skipped, 0 failed
python -m pytest -q       (after edits)    2898 passed, 3 skipped, 0 failed
                                           (+166, none removed or weakened)
```

### 1.15 Files

Created: `stratego/setups/{__init__,identity,mobility,traits,families,contracts,diversity}.py`,
`tests/setups/{__init__,family_fixtures,test_identity,test_mobility,test_traits,test_families,test_contracts,test_diversity}.py`,
`scripts/run_phase7_agent01.py`,
`reports/phase_7_data/agent_01_setup_contract.json`,
`reports/phase_7_data/agent_01_diversity_thresholds.json`, this report.
Modified: nothing — in particular `stratego/engine/`, `stratego/evaluation/`,
the Phase 4 bank, and all prior reports/artifacts are untouched.

### 1.16 Deviations and design decisions

- Canonical-frame helpers are reimplemented in `stratego/setups/identity.py`
  rather than imported from the frozen Phase 4 evaluation fixture, keeping
  the production library independent of evaluation code; an exhaustive
  equivalence test pins the two implementations together, so one convention
  exists. No behavioural deviation.
- The class representative is the lexicographic minimum of the piece-type
  tuples — content-only and auditor-recomputable. Per-square statistics are
  therefore computed over stored representatives (both mirror orientations
  appear across a family); positional thresholds use reflection-invariant
  folded cells, so no metric depends on which member was stored.
- F15 permits any Flag rank including the front (that is the family's
  purpose); the mobility rule and engine legality still apply. All other
  families constrain the Flag to the back rank(s) as part of their identity.
- The master seed 20260813 follows the Phase 4 date-seed precedent and is
  frozen in the contract; changing it defines a different library.

### 1.17 Completion gates

```text
phase_6_accepted                          true
reference_engine_is_1_2_0                 true
sixteen_family_contracts_explicit         true
library_counts_exact                      true   (8,000 / 500 / 6,400-800-800)
split_rule_exact                          true
all_family_fixtures_executable            true   (16 positive + 16 negative)
reflection_involution_clean               true   (200-sample, 0 failures)
canonicalization_stable                   true   (0 failures)
class_fingerprint_reflection_invariant    true   (0 failures)
diversity_standard_executable             true   (0 non-family failures)
thresholds_frozen_before_generation       true
                                          11 / 11
```

No strength/outcome signal exists anywhere in the acceptance logic; no
frozen engine/evaluation/model/replay semantic changed.

### 1.18 Handoff to Agent 2

Everything Agent 2 needs is importable from `stratego.setups`:

```text
contract versions        SETUP_GENERATOR_CONTRACT_VERSION, SETUP_LIBRARY_VERSION,
                         SETUP_FAMILY_VERSION, SETUP_TRAIT_VECTOR_VERSION
master seed contract     DEFAULT_LIBRARY_MASTER_SEED = 20260813,
                         derive_base_seed(...), derive_attempt_seed(...)
family predicates        FAMILY_CONTRACTS / evaluate_family(family_id, setup)
trait computation        compute_trait_vector(setup)
canonicalization         reflect_canonical, canonical_class_representative,
                         is_canonical_representative
identity                 base_setup_id, parse_base_setup_id,
                         class_fingerprint, content_fingerprint
split rule               split_for_base_index(base_index)
mobility check           setup_has_initial_mobility(setup)
diversity thresholds     DIVERSITY_THRESHOLDS_V1, evaluate_against_thresholds
serialization            BASE_ENTRY_REQUIRED_FIELDS, base_entry_json_line,
                         library_digest, LIBRARY_JSONL_PATH,
                         LIBRARY_MANIFEST_PATH
isolated rebuild         isolated_rebuild_sample_indices()
```

Generation rules Agent 2 must obey without new family-design decisions:
accept the first attempt-stream candidate satisfying the full contract;
never condition one base on another base's outcome; store only canonical
representatives; record `generation_seed` and `generation_attempts` per
entry; count and report every rejection (family, mobility) per family; and
if any family cannot produce 500 acceptable canonical bases under this
frozen contract, or any cross-base acceptance gate fails under the frozen
master seed, report `BLOCKED` — do not weaken a contract and do not reroll.

## 2. Agent 2 — Deterministic Base Library Generator

**Status: PASS** — 22 / 22 completion gates true. Machine-readable record:
`reports/phase_7_data/agent_02_generation_summary.json` (gates, verification,
determinism proofs, performance) and
`reports/phase_7_data/agent_02_base_library_manifest.json` (the materialized
manifest plus the full generator-plan table). The production library itself is
`data/setups/setup_library_v1.jsonl` with
`data/setups/setup_library_v1_manifest.json`.

### 2.1 Prerequisite verification

Verified from the repository and the live code, not assumed:

| Check | Required | Found |
|---|---|---|
| Agent 1 status | `PASS` | `agent_01_setup_contract.json` status `PASS`, 11 / 11 gates |
| Agent 1 thresholds artifact | `PASS`, frozen pre-generation | status `PASS`, `frozen_before_generation: true` |
| Live contract == artifact | identical | `contract_document()` digest `4c25724e…` == artifact digest `4c25724e…` |
| Live thresholds == artifact | identical | `DIVERSITY_THRESHOLDS_V1.to_dict()` digest `189f4dbe…` == artifact digest `189f4dbe…` |
| Contract versions | 5 frozen ids | all match the artifact's `frozen_versions` |
| Master seed | `20260813` | matches the artifact |
| Reference engine | `phase2_1_reference_1.2.0` | unchanged |
| Rules version | `stratego_project_v1` | unchanged |

Pre-existing suite, measured at commit `3e54eae` **before any Agent 2 edit**:

```text
python -m pytest -q
2898 passed, 3 skipped, 0 failed in 175.41s
```

Identical to the totals Agent 1 recorded after its own changes. The three
skips are the two pre-existing Phase 4 capability skips plus the PASS-gated
Phase 6B artifact check; all pre-date Phase 7.

### 2.2 Master seed and seed derivation

```text
master seed        20260813          (Agent 1's frozen value, unchanged)
seed context       setup_library_seed_v1   (stratego/setups/seed.py)

base_seed          blake2b(person "strat-lb7") over
                   "contract:library:master_seed:family_id:base_index"
attempt_seed       blake2b(person "strat-at7") over "base_seed:attempt"
candidate rng      random.Random(attempt_seed)   — the only randomness source
```

The master seed was not chosen from any game outcome; it is Agent 1's frozen
date-seed, carried unchanged. `LibrarySeedContext` bundles the four identity
inputs so no call site mixes them by hand, and serializes the derivation into
the manifest. No seed input names another base, another family, or any
enumeration counter, so a rejected candidate in F03 cannot move any other
base setup — the property the enumeration-order regression pins directly.

### 2.3 Generator architecture: one framework, sixteen plans

There is exactly one construction procedure (`construct_candidate`), driven by
a declarative `FamilyPlan` per family. No literal setups are hard-coded, and
no family has its own generator program.

```text
FlagPlan     rank weights + permitted edge distances
BombPlan     flag-guard count, guard/zone/diagonal caps, front-half quota,
             front-rank cap, dispersion, distinct-file floor, decoy pocket
GroupPlan    per piece group: permitted counts in ranks 2-3 and on rank 3,
             plus rank weights

construction order
    flag -> flag guard bombs -> decoy pocket bombs -> free bombs
         -> scouts -> miners -> marshal -> general -> high_others -> spy
         -> uniform fill of every unnamed piece
```

Shared infrastructure — seed derivation, engine validation, family evaluation,
mobility, canonicalization, fingerprints, metadata, serialization — is common
to all sixteen. Every plan clause exists to realize a clause of Agent 1's
frozen family contract, and each is stated in the machine-readable plan table
with its rationale:

| ID | Plan constraint (abridged) | Realizes |
|---|---|---|
| F00 | Flag rank 0, edge distance 0; both orthogonal neighbours bombed | `flag_orth_bomb_guards == 2` |
| F01 | Flag rank 0, edge distance 1–2; 2–3 guards | `>= 2` guards near the corner |
| F02 | Flag rank 0, edge distance 3–4; 2–3 guards | central fortress |
| F03 | exactly 1 guard; free Bombs capped at 3 in the Chebyshev-2 zone | `== 1` guard, forbidden clause at 4 |
| F04 | 0 guards; no Bomb on a Flag diagonal; zone cap 2 | 0 orthogonal / 0 diagonal guards |
| F05 | ≤ 1 guard; 2–3 Bombs around a reserved movable back-half cell at Manhattan ≥ 4 | `decoy_pocket_bombs >= 2` |
| F06 | Bombs mutually non-adjacent over ≥ 5 distinct files; guard cap 1 | dispersion clauses |
| F07 | 4–6 Bombs drawn into ranks 2–3 | `bomb_front2_count >= 4` |
| F08 | Marshal and General forward + 3–5 of the other five heavies | `high_front2_count 5..7` |
| F09 | the mirror: heavies held in ranks 0–1 | `high_back2_count 5..7` |
| F10 | 6–8 Scouts in ranks 2–3, 3–5 of them on rank 3 | Scout-forward clauses |
| F11 | 0 Scouts on rank 3, ≤ 3 on rank 2 | `scout_back2_count >= 5` |
| F12 | 3–5 Miners in ranks 2–3 | `miner_front2_count >= 3` |
| F13 | 0 Miners on rank 3, ≤ 1 in the front half | `miner_back2_count >= 4` |
| F14 | guarded back-rank Flag, ≤ 2 front-rank Bombs, Marshal/General off rank 3, 3–6 forward Scouts, ≤ 3 forward Miners | the eight balanced clauses |
| F15 | every group uniform over the whole zone, Flag included | high-entropy draws, forbidden clause filters the fortress signature |

F14's front-rank Bomb cap is what secures `movable_front_rank_count >= 8`:
with the Flag on rank 0, Bombs are the only immovable pieces that can reach
the front rank.

### 2.4 Acceptance stack and rejection accounting

Plans propose; Agent 1's frozen predicates dispose, in this order:

```text
engine validate_setup(candidate, 0)      official inventory, legal form
family_contract(F).evaluate(traits)      the primary-family contract
setup_has_initial_mobility(candidate)    the curated library-quality rule
```

Attempts `0, 1, 2, …` are drawn from the frozen attempt streams and the first
candidate passing all three is accepted. A plan that drifted from its contract
could therefore only waste attempts, never smuggle in an invalid setup.

```text
accepted bases                8,000
total candidate attempts      8,274
attempts per accepted base    1.03425

rejections by reason
    construction_infeasible      55      (a plan step ran out of legal cells)
    engine_invalid                0
    family_predicate            219
    stranded                      0

attempt histogram   1: 7,796 · 2: 153 · 3: 39 · 4: 9 · 5: 2 · 9: 1
```

Rejections by family — every other family accepted its first candidate every
time:

| Family | Rejections | Reason | Attempts/base | Rejection rate |
|---|---:|---|---:|---:|
| F08 | 2 | construction_infeasible | 1.004 | 0.40 % |
| F09 | 3 | construction_infeasible | 1.006 | 0.60 % |
| F14 | 50 | construction_infeasible | 1.100 | 9.09 % |
| F15 | 219 | family_predicate | 1.438 | 30.46 % |

F08/F09/F14 exhaust cells in a half-board when several quotas compete for the
same twenty cells; F15 is pure rejection sampling against the frozen
unconventional-feature clause, which is the honest way to keep that family
high-entropy rather than constructing the features. No candidate was ever
repaired: rejection draws the next attempt stream, never an edit.

Production generation never exercised the `engine_invalid` or `stranded`
branches, so both are pinned by crafted regressions instead — an
inventory-corrupted arrangement and a legal, F07-satisfying arrangement whose
six Bombs occupy all six open front files (`tests/setups/test_generator.py`).

### 2.5 Global uniqueness and split handling

Uniqueness is enforced over the whole 8,000-entry library, across families,
before the library is materialized:

```text
distinct class fingerprints    8,000 / 8,000
distinct arrangements          8,000 / 8,000
distinct stable ids            8,000 / 8,000
stored set ∩ mirrored set      empty
```

The gate **raises** on collision rather than regenerating the colliding base.
That is Agent 1's frozen cross-base-independence rule: conditioning one base
on another base's outcome would destroy isolated rebuild, so a duplicate is a
BLOCKED finding for review. Under the frozen master seed there are none. The
gate's ability to fail is tested with planted exact, mirrored and stable-id
collisions.

Splits come only from Agent 1's frozen index rule — `0–399` train, `400–449`
validation, `450–499` test — never from acceptance order, and were never
adjusted after generation:

```text
per family      400 train / 50 validation / 50 test     (all 16 families)
library         6,400 train / 800 validation / 800 test
cross-split class duplicates                            0
```

### 2.6 Canonical stored form

Only the reflection-class representative is stored; the library is 8,000
entries, not 16,000. Acceptance is decided before canonicalization, which is
sound because legality, family membership and mobility are all
reflection-invariant under the frozen contract. Every stored entry satisfies

```text
canonicalize(setup)          == setup      8,000 / 8,000
canonicalize(reflect(setup)) == setup      8,000 / 8,000
class_fingerprint(reflect(setup)) == stored fingerprint
```

and records both the stored-orientation fingerprint and the reflected one.
Runtime orientation choice remains Agent 4's responsibility.

### 2.7 Materialized library and manifest

```text
data/setups/setup_library_v1.jsonl            14,110,382 bytes, 8,000 lines
data/setups/setup_library_v1_manifest.json         4,716 bytes

library_content_digest   7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777
entry_metadata_digest    d86f486182a820d546d470ef4ebce92ff60c6259aed80c481bc985bce8c64980
manifest_digest          53139ab7e21c4e8a31987507d6fb1eabf93f36cdc1221fe85d08042963488f31
```

`library_content_digest` is Agent 1's frozen identity digest over
`base_setup_id:fingerprint` in file order. `entry_metadata_digest` is stronger:
SHA-256 over the exact serialized lines, so a regeneration that changed any
recorded seed, attempt index or trait is caught even when the setups match.
`manifest_digest` covers the manifest minus its `generation_run` section, so
timestamps, wall time and host measurements stay outside the hash domain and
two independent regenerations agree exactly.

Each entry carries the frozen fourteen fields plus five required by the Agent 2
metadata contract:

```text
frozen      base_setup_id, library_version, contract_version,
            family_contract_version, trait_schema_version, family_id,
            family_key, base_index, split, canonical_setup (40-char),
            fingerprint, generation_seed, generation_attempts, trait_vector
added       generator_version, content_fingerprint,
            reflected_content_fingerprint, accepted_attempt_index,
            accepted_attempt_seed
```

`generation_seed` + `accepted_attempt_index` are exactly what an isolated
rebuild needs to reproduce the accepted candidate. No entry field names a game
outcome, win rate, Elo, value or policy score; the check is executable
(`FORBIDDEN_ENTRY_FIELD_TOKENS`) and reports zero offending fields, and an AST
scan of the three new modules reports zero offending identifiers.

### 2.8 Determinism

| Proof | Method | Result |
|---|---|---|
| Isolated rebuild | Agent 1's fixed sample `{0–9, 395–404, 445–454, 490–499}` × 16 families, rebuilt from identity alone and compared field-by-field | 640 / 640 exact, 0 mismatches |
| Enumeration independence | 256 bases regenerated in a shuffled order | 0 mismatches |
| Full regeneration | the entire library generated a second time | both digests identical |
| Serialization | write → read → rewrite | read-back exact, bytes identical |
| Master-seed sensitivity | 128 bases under master seed `20260814` | 128 / 128 changed, 0 shared fingerprints, ids and splits unchanged |

Generation consumes no global RNG state: reseeding Python's global stream
between two rebuilds of the same base yields the identical entry (tested).

### 2.9 Generation-time correctness checks

`verify_library` recomputes every check from stored content — re-validating
through the frozen engine, re-evaluating the frozen family contracts,
re-canonicalizing, re-fingerprinting, re-deriving identity and split — rather
than trusting the generator's counters:

```text
entry count                          8,000     exact
family counts                        500 × 16  exact
family split counts                  400/50/50 × 16  exact
engine-invalid bases                     0
incorrect inventories                    0
stranded bases                           0
primary-family violations                0
exact duplicate arrangements             0
reflection-equivalent duplicates         0
content-fingerprint collisions           0
stable-ID collisions                     0
stored non-canonical representatives     0
reflection round-trip failures           0
identity / split rule mismatches         0
entry metadata mismatches                0
outcome/strength fields                  0
```

Basic trait distributions are recorded per family in the summary artifact
(Flag rank and edge-distance histograms; min/max/mean for guard count, Bomb
front-half count, Bomb file dispersion, Scout front counts, Miner front count,
high-rank front count, front-rank movability, unconventional-feature count).
They confirm the plans did what they claim — for example F00's Flag rank
histogram is `[500, 0, 0, 0]` while F15's is `[92, 116, 146, 146]`.

### 2.10 Diversity preflight (Agent 3 owns the verdict)

Agent 1's full standard was executed on the finished library as a preflight so
a knowingly broken library is never handed on. **This is not a diversity
acceptance declaration** — Agent 3 owns that independent verdict.

```text
checks executed          199
failures                   0
global minimum pairwise class distance     20   (floor 4)
cross-split nearest-neighbour distance     21   (floor 8)
global mean per-square entropy          3.185 bits (floor 1.5)
family self-satisfaction                  1.0 on every diagonal cell
largest off-diagonal overlap             0.772  (F11 → F15, report-only)
```

| Family | min within-family NN | mean entropy (bits) | Flag folded support | distinct trait vectors |
|---|---:|---:|---:|---:|
| F00 | 22 | 2.94 | 1 (floor 1) | 500 |
| F01 | 22 | 3.01 | 2 (floor 2) | 500 |
| F02 | 22 | 3.02 | 2 (floor 2) | 500 |
| F03 | 24 | 3.14 | 10 | 500 |
| F04 | 22 | 3.15 | 10 | 500 |
| F05 | 21 | 3.13 | 10 | 500 |
| F06 | 22 | 3.17 | 10 | 500 |
| F07 | 22 | 3.11 | 10 | 500 |
| F08 | 22 | 3.03 | 10 | 500 |
| F09 | 22 | 3.03 | 10 | 500 |
| F10 | 24 | 3.08 | 10 | 500 |
| F11 | 22 | 3.01 | 10 | 500 |
| F12 | 24 | 3.10 | 10 | 500 |
| F13 | 22 | 3.06 | 10 | 500 |
| F14 | 20 | 3.04 | 5 (floor 3) | 500 |
| F15 | 25 | 3.24 | 20 (floor 8) | 500 |

Within-family nearest-neighbour distances sit at 20–25 of 40 against a floor
of 6, and every family's 500 bases have 500 distinct trait vectors against a
floor of 250 — the margins expected of independent structured draws rather
than template repetition.

### 2.11 Performance

```text
full-library generation wall time        2.958 s   (8,000 bases)
second full regeneration                 3.103 s
candidate attempts per accepted base     1.03425
peak RSS, generation only                   57 MB
peak RSS, generation + materialization     106 MB
peak RSS, full harness                   1,232 MB
materialized library                14,110,382 bytes
manifest                                 4,716 bytes
harness total (all proofs + preflight)  11.402 s
```

The harness peak is dominated by the optional diversity preflight's 8,000 ×
8,000 class-distance matrix and the `torch` import used only for environment
recording; the generator itself regenerates the whole library in three seconds
inside 60 MB, which makes routine regeneration during development practical.
Nothing was traded away for that: no determinism shortcut, no relaxed check.

### 2.12 Tests

`scripts/run_phase7_agent02.py` runs the whole acceptance path — prerequisite
verification, generation, materialization, all five determinism proofs,
content re-verification and the diversity preflight — before writing the
artifacts. The repository suite gained 78 tests in `tests/setups/`:

```text
test_generator.py    plan table completeness and validation, determinism from
                     identity, pure-function construction, master-seed
                     sensitivity, stream separation, isolated rebuild at both
                     split boundaries, engine validity, mobility, family
                     satisfaction, canonical storage, reflection round-trip,
                     recorded-metadata agreement, attempt accounting, crafted
                     engine-invalid / family-violating / stranded rejections,
                     AST scan for outcome or strength identifiers, no global
                     RNG consumption
test_library.py      exact 8,000 / 500 / 400-50-50 counts, file order, no
                     cross-split fingerprint, all fifteen verification checks,
                     planted exact / mirrored / stable-ID collisions rejected,
                     canonical JSON line form, write-read-rewrite byte
                     stability, digest determinism and sensitivity, manifest
                     fields and digest domain, shuffled-order regeneration,
                     alternative-master-seed divergence, and — once the
                     production files exist — the materialized library, its
                     manifest and both Agent 2 artifacts against a fresh
                     generation
```

```text
python -m pytest -q       (before edits)   2898 passed, 3 skipped, 0 failed
python -m pytest -q       (after edits)    2976 passed, 3 skipped, 0 failed
                                           (+78, none removed or weakened)
```

The nine artifact-gated tests in `test_library.py` skip while the production
library is absent and run once it exists, so they were green in the final run
against the materialized artifacts. No pre-existing test was removed, weakened
or re-scoped.

### 2.13 Files

Created: `stratego/setups/{seed,generator,library}.py`,
`tests/setups/{test_generator,test_library}.py`,
`scripts/run_phase7_agent02.py`, `data/setups/setup_library_v1.jsonl`,
`data/setups/setup_library_v1_manifest.json`,
`reports/phase_7_data/agent_02_base_library_manifest.json`,
`reports/phase_7_data/agent_02_generation_summary.json`.

Modified: `stratego/setups/__init__.py` (package docstring and re-exports for
the three new modules; no Agent 1 definition changed), this report (section 2
appended).

Untouched: `stratego/engine/`, `stratego/evaluation/` and the Phase 4 bank,
`observation_v2_1_127ch`, `trajectory_v1`, every Agent 1 contract module, and
all prior report sections and artifacts.

### 2.14 Deviations and design decisions

- **Global uniqueness is a raising gate, not a regeneration filter.** The
  Agent 2 instructions describe rejecting a candidate whose fingerprint
  already exists anywhere in the library; Agent 1's frozen contract forbids
  conditioning one base's acceptance on another base's outcome, because that
  breaks isolated rebuild. The two are reconciled by enforcing uniqueness as a
  hard pre-materialization gate that raises a BLOCKED finding on collision.
  The observable requirement — zero exact and zero reflection-equivalent
  duplicates across all 8,000 bases — is met exactly, and the frozen master
  seed produces no collision to resolve.
- **Entries carry five fields beyond Agent 1's frozen fourteen**
  (`generator_version`, `content_fingerprint`,
  `reflected_content_fingerprint`, `accepted_attempt_index`,
  `accepted_attempt_seed`), as the Agent 2 metadata contract requires. The
  frozen field list, line format and file order are unchanged, and the frozen
  `base_entry_json_line` still validates every line.
- **Acceptance is evaluated before canonicalization.** The three acceptance
  predicates are reflection-invariant under the frozen contract, so the
  verdict is identical either way; storing the representative afterwards keeps
  the library at 8,000 entries. Both directions are re-checked per entry.
- **Unconstrained piece groups share one mild rank preference** across all
  sixteen families (Scouts forward, Spy and Miners rearward, heavies off the
  very front — the accepted Phase 4 structural precedent), so families differ
  from one another only in the dimensions their contracts actually name.
- **F15 uses rejection sampling rather than feature construction.** Building
  the unconventional features directly would have made the highest-entropy
  family the most constructed one; drawing uniformly and letting the frozen
  clause filter costs 219 extra attempts and keeps the distribution honest.
- The library JSONL is 14.1 MB because Agent 1's frozen entry format stores
  the full 35-field trait vector per entry. It is deterministic and
  digest-checked, so it is a reproducible artifact rather than opaque data.

### 2.15 Completion gates

```text
agent_01_pass_verified                    true
agent_01_contract_matches_live_code       true
eight_thousand_bases_materialized         true
five_hundred_per_family                   true
split_counts_exact                        true   (400/50/50 × 16)
zero_engine_invalid                       true
zero_stranded                             true
zero_family_violations                    true
zero_exact_duplicates                     true
zero_reflection_duplicates                true
zero_stable_id_collisions                 true
all_entries_canonical                     true
reflection_roundtrip_clean                true
entry_metadata_consistent                 true
isolated_rebuild_exact                    true   (640 / 640)
enumeration_order_independent             true
master_seed_sensitive                     true
full_regeneration_digest_stable           true
serialization_roundtrip_exact             true
library_and_manifest_written              true
no_outcome_or_strength_signal             true
full_repository_suite_green               true   (2976 / 3 / 0)
                                          22 / 22
```

No outcome, win rate, Elo, value or policy signal participated in any
acceptance, rejection, weighting or split decision. No frozen
engine/evaluation/model/replay semantic changed, and no neural training
occurred.

### 2.16 Handoff to Agent 3

```text
library path        data/setups/setup_library_v1.jsonl
manifest path       data/setups/setup_library_v1_manifest.json
library digest      7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777
metadata digest     d86f486182a820d546d470ef4ebce92ff60c6259aed80c481bc985bce8c64980
manifest digest     53139ab7e21c4e8a31987507d6fb1eabf93f36cdc1221fe85d08042963488f31
master seed         20260813
generation command  python scripts/run_phase7_agent02.py

contract versions   setup_generator_contract_v1 · setup_library_v1 ·
                    setup_family_v1 · setup_trait_vector_v1 ·
                    setup_diversity_standard_v1 · setup_base_generator_v1
```

APIs Agent 3 needs, all importable from `stratego.setups` and all recomputing
from content rather than from Agent 2's counters:

```text
read the library      read_library_jsonl(path) -> [BaseSetupEntry]
                      read_manifest(path)
engine validation     stratego.engine.setup.validate_setup(setup, player)
                      setup_has_initial_mobility(setup)
identity              class_fingerprint, content_fingerprint,
                      canonical_class_representative,
                      is_canonical_representative, reflect_canonical,
                      base_setup_id / parse_base_setup_id,
                      split_for_base_index
traits                compute_trait_vector(setup), TRAIT_SCHEMA
families              evaluate_family(family_id, setup), FAMILY_CONTRACTS
diversity             DIVERSITY_THRESHOLDS_V1, evaluate_against_thresholds
                      (artifact: reports/phase_7_data/agent_01_diversity_thresholds.json)
digests               library_content_digest, entry_metadata_digest,
                      manifest_digest
independent rebuild   rebuild_base_setup(family_id, base_index)
```

Agent 2's acceptance counters are reported for transparency but are not needed
to audit anything: every claim in this section is recomputable from the
materialized JSONL plus Agent 1's frozen contracts. Agent 2 does not declare
diversity acceptance, and does not declare Phase 7 complete.

## 3. Agent 3 — Exhaustive Library Audit

**Status: PASS** — 25 / 25 completion gates true. Machine-readable record:
`reports/phase_7_data/agent_03_library_audit.json` (every stage, every gate,
the full overlap matrix and attributions), with
`reports/phase_7_data/agent_03_family_metrics.csv` (275 rows: every frozen
per-family threshold plus descriptive context) and
`reports/phase_7_data/agent_03_similarity_audit.csv` (68 rows: global,
within-family, and all cross-split scopes). The production library and its
manifest were not modified: byte-identity before/after the audit is itself a
recorded gate.

### 3.1 Prerequisite verification

Verified from artifacts and live code before any audit stage ran:

| Check | Required | Found |
|---|---|---|
| Agent 1 status | `PASS` | `agent_01_setup_contract.json` and `agent_01_diversity_thresholds.json` both `PASS`, thresholds `frozen_before_generation: true` |
| Agent 2 status | `PASS` | `agent_02_generation_summary.json` `PASS`, 22/22 gates |
| Live contract == artifact | identical | digest `4c25724e…` both sides |
| Live thresholds == artifact | identical | digest `189f4dbe…` both sides |
| Library digest before audit | `7b8a6660…` | recomputed from JSONL bytes: match |
| Metadata digest before audit | `d86f4861…` | recomputed: match |
| Manifest digest before audit | `53139ab7…` | recomputed: match |
| Reference engine | `phase2_1_reference_1.2.0` | unchanged |

A handoff-digest mismatch would have been `BLOCKED` (wrong input), not
`FAIL`; none occurred.

Pre-existing suite, measured at commit `974e5e9` **before any Agent 3 edit**:

```text
python -m pytest -q
2976 passed, 3 skipped, 0 failed in 179.04s
```

Identical to Agent 2's post-edit totals; the three skips pre-date Phase 7.

### 3.2 Audit design

`stratego/setups/audit.py` recomputes every fact from the materialized JSONL
plus the frozen engine and the frozen Agent 1 contracts. Agent 2's counters,
preflight values and manifest are audit subjects, never inputs. Reuse of
Agent 1's authoritative family/trait/identity definitions is per instruction;
similarity carries an additional audit-side implementation:

```text
distance matrix     dense 8,000 x 8,000 class-distance matrix built in
                    audit.py (blocked numpy uint8, matches-complement
                    formulation), independent of Agent 1's
                    diversity._pairwise_class_distances
cross-check 1       2,000 deterministically sampled pairs re-verified
                    against Agent 1's scalar class_distance: 0 mismatches,
                    matrix exactly symmetric
cross-check 2       every thresholded reduction (per-family min NN and
                    near-duplicate fraction, cross-split minimum, global
                    minimum) reconciled against the frozen distance_metrics:
                    exact agreement
```

Per-entry checks are exception-guarded: a malformed entry becomes a recorded
finding with its id, never an auditor crash and never a repair.

### 3.3 Exhaustive legality audit

For each of the 8,000 bases and each of the 8,000 reflected forms: engine
validation for both colours in both the row-major tuple form and the
square-oriented placement form (`validate_setup` + `validate_setup_placement`
— 32,000 placement validations in total), exact inventory recount, initial
mobility through the frozen engine (`create_game` + `has_legal_action`,
16,000 probes), independently recomputed trait vector, primary-family
predicate on base and reflection, serialization round trip, reflection round
trips, canonicalization, all three fingerprints, and identity/split/seed
re-derivation from the base index alone.

```text
base engine validation failures          0 / 8,000
reflected validation failures            0 / 8,000
inventory failures                       0
placement failures                       0
initial-mobility failures (base)         0
initial-mobility failures (reflected)    0
family-predicate failures (base)         0
family-predicate failures (reflected)    0
serialization failures                   0
reflection round-trip failures           0
canonicalization failures                0
fingerprint mismatches                   0   (class, content, reflected)
trait-vector mismatches                  0
identity / split-rule mismatches         0
seed re-derivation mismatches            0   (generation_seed and
                                              accepted_attempt_seed rebuilt
                                              from identity alone)
version-field mismatches                 0
```

The JSONL bytes themselves were audited line-by-line: all 8,000 lines parse,
carry the frozen required fields, deserialize to valid setups, and are
byte-identical to the frozen canonical serialization of their own payload.

### 3.4 Count audit

Recomputed independently and exact:

```text
total bases                     8,000
per family                      500 x 16
train / validation / test       6,400 / 800 / 800
per family per split            400 / 50 / 50 x 16
distinct stable ids             8,000
distinct class fingerprints     8,000
distinct arrangements           8,000
```

### 3.5 Duplicate audit

Groups were formed over the whole library — never within-family only:

```text
exact duplicate arrangements             0
reflection-equivalent duplicates         0   (recomputed class fingerprints)
stored ∩ mirrored arrangements           0
same stable id, different setup          0
different stable id, same class          0
cross-split class duplicates             0
```

### 3.6 Cross-split leakage audit

Distances are Agent 1's frozen reflection-class metric
`min(H(a,b), H(a,reflect(b)))`, so mirrored near-copies cannot hide. All
three split pairs, globally and within every family:

| Scope | Pairs | min | p1 | median | NN min A→B / B→A | pairs < 8 | pairs < 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train ↔ validation | 5,120,000 | 21 | 29 | 34 | 21 / 21 | 0 | 0 |
| train ↔ test | 5,120,000 | 22 | 29 | 34 | 22 / 22 | 0 | 0 |
| validation ↔ test | 640,000 | 22 | 29 | 34 | 22 / 22 | 0 | 0 |

Per-family cross-split minima (worst of the three split pairs): F00 23,
F01 22, F02 22, F03 24, F04 23, F05 24, F06 23, F07 22, F08 23, F09 22,
F10 24, F11 23, F12 24, F13 22, F14 22, F15 25.

```text
cross-split NN distance, whole library    21      floor 8      PASS
pairs below the near-duplicate distance    0
offending pairs                            none
```

Agent 1 declared no percentile thresholds; the p1/p5/p25/median columns here
and in the similarity CSV are nearest-rank descriptive context. Only the
frozen minima and ceilings gate.

### 3.7 Within-family diversity against the frozen thresholds

Every Agent 1 metric, recomputed from raw setups. All 199 frozen checks pass;
the full per-family table is `agent_03_family_metrics.csv`.

| Family | min NN (≥6) | near-dup frac (≤0.001) | entropy bits (≥1.0) | Flag support | Bomb (≥10) | Scout (≥8) | Miner (≥6) | high (≥6) | vectors (≥250) | NN med/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F00 | 22 | 0.0 | 2.937 | 1 (≥1) | 20 | 20 | 20 | 20 | 500 | 25 / 27 |
| F01 | 22 | 0.0 | 3.014 | 2 (≥2) | 20 | 20 | 20 | 20 | 500 | 26 / 28 |
| F02 | 22 | 0.0 | 3.015 | 2 (≥2) | 20 | 20 | 20 | 20 | 500 | 25 / 28 |
| F03 | 24 | 0.0 | 3.136 | 10 (≥4) | 20 | 20 | 20 | 20 | 500 | 27 / 29 |
| F04 | 22 | 0.0 | 3.155 | 10 (≥4) | 20 | 20 | 20 | 20 | 500 | 27 / 30 |
| F05 | 21 | 0.0 | 3.126 | 10 (≥4) | 20 | 20 | 20 | 20 | 500 | 27 / 29 |
| F06 | 22 | 0.0 | 3.166 | 10 (≥4) | 20 | 20 | 20 | 20 | 500 | 27 / 30 |
| F07 | 22 | 0.0 | 3.112 | 10 (≥4) | 20 | 20 | 20 | 20 | 500 | 27 / 29 |
| F08 | 22 | 0.0 | 3.031 | 10 (≥4) | 20 | 20 | 20 | 20 | 500 | 27 / 29 |
| F09 | 22 | 0.0 | 3.029 | 10 (≥4) | 20 | 20 | 20 | 20 | 500 | 27 / 29 |
| F10 | 24 | 0.0 | 3.081 | 10 (≥4) | 20 | 20 | 20 | 20 | 500 | 26 / 29 |
| F11 | 22 | 0.0 | 3.014 | 10 (≥4) | 20 | 15 | 20 | 20 | 500 | 26 / 28 |
| F12 | 24 | 0.0 | 3.101 | 10 (≥4) | 20 | 20 | 20 | 20 | 500 | 27 / 30 |
| F13 | 22 | 0.0 | 3.060 | 10 (≥4) | 20 | 20 | 15 | 20 | 500 | 26 / 29 |
| F14 | 20 | 0.0 | 3.038 | 5 (≥3) | 20 | 20 | 20 | 20 | 500 | 26 / 28 |
| F15 | 25 | 0.0 | 3.244 | 20 (≥8) | 20 | 20 | 20 | 20 | 500 | 28 / 30 |

```text
global mean per-square entropy      3.184683 bits     floor 1.5    PASS
global minimum pairwise distance    20                floor 4      PASS
distinct Bomb rank histograms       30–65 per family  floor 8      PASS
distinct Scout rank histograms      18–85 per family  floor 8      PASS
near-duplicate pairs (< 10), any family      0
```

Constrained families leave exactly the signatures their contracts pin, and
nothing more: F11's Scout folded support is 15/20 (its clause bars Scouts
from the five front-rank folded cells), F13's Miner support is likewise
15/20, F10's distinct Scout histograms drop to 18 (Scouts concentrated
forward by contract), F00/F01/F02/F14 pin Flag support at 1/2/2/5, and F15 —
the only family whose Flag may go anywhere — reaches 20/20. Every value sits
at or above its family-specific floor; low entropy appears only where
Agent 1 explicitly allowed it.

### 3.8 Between-family overlap / confusion matrix

`matrix[i][j]` = fraction of family `i` bases whose recomputed trait vector
satisfies family `j`'s frozen contract. The diagonal is the hard gate and is
exactly 1.0 for all 16 families. Off-diagonal overlap is **report-only**
under Agent 1's frozen standard (families are not required to be disjoint);
the full 16 × 16 matrix (fractions and integer counts) is in the audit
artifact, and every off-diagonal cell at or above 0.25 carries a clause-level
attribution.

```text
family self-satisfaction diagonal        1.0 x 16      required   PASS
off-diagonal cells > 0                   150 of 240    report-only
off-diagonal cells >= 0.25                67           report-only
largest                                  F11 -> F15    0.772  (386 / 500)
```

**The F11 → F15 overlap of 0.772, recomputed and attributed exactly.** The
audit reproduces Agent 2's preflight value independently: 386 of F11's 500
bases also satisfy F15. The mechanism is definitional, not accidental:

```text
F11 requires   scout_front_rank_count == 0
F15 requires   unconventional_feature_count >= 2, where feature #6 of the
               frozen eight is no_front_rank_scouts
               (scout_front_rank_count == 0)
```

F11's own defining clause **is** one of F15's eight unconventional features,
so every F11 base starts one feature toward F15's two-feature requirement —
the audit confirms feature-count ≥ 1 for all 500 (histogram over the eight
features: 1× 90, 2× 200, 3× 158, 4× 46, 5× 6; zero bases at 0). The exact
arithmetic of the cell:

```text
F11 bases with >= 2 features                    410 / 500   (0.820)
F11 bases with the forbidden fortress
signature (flag_rank 0 and >= 2 guards)          35 / 500   (0.070)
bases with both                                  24
satisfy F15  =  410 - 24  =                     386 / 500   (0.772)
```

Among the 386, the second feature is mostly `flag_unguarded` (63.5 %),
`marshal_on_front_rank` (32.9 %), `spy_on_front_rank` (32.1 %) or
`general_on_front_rank` (28.0 %): holding five-plus Scouts in the back two
ranks crowds the rear, so heavier pieces man the front rank and fortress
walls are rarer. The same crowding drives the second-largest F11 coupling,
F11 → F12 = 0.682 (Miners pushed forward). The direction is sharply
asymmetric — F15 → F11 is 0.034, because a high-entropy draw rarely holds
five Scouts back with none forward — and the F15 column as a whole confirms
the family's catch-all role: F07 → F15 = 0.574 (forward Bombs strip fortress
walls), F08 → F15 = 0.522 (Marshal and General forward are two features by
themselves).

None of this is content leakage: predicate co-satisfaction is a taxonomy
property. The nearest any two bases come to each other anywhere in the
library is class distance 20 of 40, every base satisfies its declared
primary family exactly, and every identity is unique. The matrix's
structural zeros land precisely where the contracts are arithmetically
exclusive — F08 ↔ F09 (≥5 of 7 heavies forward vs. rear), F12 ↔ F13 (≥3 of
5 Miners forward vs. ≥4 rear), F10 → F11 (≥3 front-rank Scouts vs. 0),
F00/F01/F02 mutually (Flag edge-distance partition {0}, {1,2}, {3,4}),
F03–F06 → F14 and F15 → F00/F01/F02/F14 (guard-count and fortress-signature
exclusions) — which is independent evidence the matrix measures the
contracts, not the generator. Column means show F05 (0.355) and F10 (0.413)
are the contracts most often satisfied incidentally, F11 (0.005) and
F00 (0.009) the most exclusive.

Observation, no action taken: if a later phase ever needs mutually exclusive
family labels (e.g. as classification targets), F15's permissive two-feature
contract — and this one definitional feature-sharing with F11 — would need a
new family-contract version. Under the frozen `setup_family_v1`, 0.772 is a
described property, not a defect; primary labels remain unambiguous.

### 3.9 Reflection audit

For every one of the 8,000 bases:

```text
reflect(reflect(s)) == s                    8,000 / 8,000
reflection inventory preserved              8,000 / 8,000   (engine-validated)
reflection engine-legal, both colours       8,000 / 8,000
reflection initial mobility                 8,000 / 8,000
reflection satisfies the primary family     8,000 / 8,000
class_fingerprint(reflect(s)) == stored     8,000 / 8,000
canonicalize(reflect(s)) == stored base     8,000 / 8,000
exactly reflection-symmetric bases          0
```

Zero symmetric bases is the only possible result — the single Flag would
need file `f == 9 - f`, which has no integer solution — and the audit
confirms the theorem holds on the data. Reflection changes no identity
semantics: every reflected form folds back to its stored representative and
fingerprint, so splits and family labels are inherited exactly as the
contract requires.

### 3.10 Independent metric implementation

```text
audit-side matrix vs frozen scalar metric      2,000 sampled pairs, 0 mismatches
audit-side matrix symmetry                     exact
audit-side reductions vs frozen reductions     exact agreement on every
                                               thresholded value
trait vectors                                  recomputed for all 8,000,
                                               0 mismatches vs stored
fingerprints                                   recomputed for all 8,000 bases
                                               and reflections, 0 mismatches
```

Agent 2's cached values were compared against, never consumed: the manifest
stage recomputed all three digests from entry content and matched the
handoff (`7b8a6660…`, `d86f4861…`, `53139ab7…`).

### 3.11 Performance

```text
audit wall time                    8.067 s   (full harness incl. artifacts)
peak RSS                           1,420 MB
similarity method                  dense blocked numpy uint8 class-distance
                                   matrix, direct + mirrored orientations
matrix build                       1.364 s
ordered pair comparisons           63,992,000   (31,996,000 unordered)
cell comparisons                   5,119,360,000
placement validations              32,000   (both colours, base + reflection)
mobility probes (create_game)      16,000
stage times                        per_base 2.435 s · similarity 2.053 s ·
                                   thresholds 2.825 s · overlap 0.113 s ·
                                   duplicates 0.062 s · line format 0.189 s ·
                                   manifest 0.108 s
```

Nothing was sampled or weakened: the full 8,000² comparison ran exactly as
the instructions intend for a library this size.

### 3.12 Tests

`tests/setups/test_audit.py` adds 42 tests. Positive controls prove the
auditor reports zero findings on clean isolated rebuilds of real production
entries and that both similarity implementations agree. Each
instruction-mandated defect class is deliberately injected and caught by the
named detector:

```text
wrong inventory              inventory_failures + engine_failures
illegal placement            engine placement validator + undeserializable line
stranded setup               mobility_failures, base and reflected,
                             with legality/family clean (isolated)
wrong family label           family_failures with the violated clause named,
                             and a broken overlap diagonal
exact duplicate              exact_duplicate_groups, cross-family too
reflected duplicate          reflection_class_duplicate_groups +
                             stored_mirror_overlap + canonicalization finding
stable-ID collision          stable_id_collisions + same_id_different_setup
split-count mismatch         count checks + identity_failures
cross-split near duplicate   distance-2 pair found, offenders named,
                             frozen threshold check fails
threshold failure            F00 min-NN floor and near-duplicate fraction
                             fail as findings while self-satisfaction passes
bad serialization            noncanonical / missing-field / unparseable lines
bad reflection               corrupted class and reflected fingerprints caught
altered manifest digest      manifest_digest_matches false; tampered counts
                             and forged handoff digests also caught
```

Plus: tampered generation seeds and trait vectors are caught, `audit_library`
on a forged mini-library returns FAIL with named gates, and — once the
production artifacts exist — artifact-gated tests pin the recorded PASS,
digests, zero-finding lists, 199/199 threshold checks, and both CSVs
(including zero cross-split pairs below 8 in every scope) against the files
on disk.

```text
python -m pytest -q       (before edits)   2976 passed, 3 skipped, 0 failed
python -m pytest -q       (after edits)    3018 passed, 3 skipped, 0 failed
                                           (+42, none removed or weakened)
```

### 3.13 Files

Created: `stratego/setups/audit.py`, `tests/setups/test_audit.py`,
`scripts/run_phase7_agent03.py`,
`reports/phase_7_data/agent_03_library_audit.json`,
`reports/phase_7_data/agent_03_family_metrics.csv`,
`reports/phase_7_data/agent_03_similarity_audit.csv`.

Modified: `stratego/setups/__init__.py` (docstring + audit re-exports), this
report (section 3 appended).

Untouched: `stratego/engine/`, every Agent 1 contract module and threshold,
Agent 2's generator/library modules, the production library and manifest
(byte-verified before and after the audit), the Phase 4 bank, and all prior
report sections and artifacts.

### 3.14 Deviations and observations

- Agent 1 defined no percentile thresholds, so the similarity report's
  p1/p5/p25/median columns are nearest-rank descriptive context; every
  gating comparison uses exactly the frozen minima/ceilings/exacts.
- In the canonical 40-tuple representation, placement illegality cannot be
  expressed while inventory holds; the audit therefore also runs the
  engine's square-oriented placement validator on every oriented form
  (32,000 validations), and the injected-placement proof exercises that
  validator directly plus the malformed-line path.
- Reflection invariance of family membership and mobility is a theorem
  Agent 1 tested; the audit still evaluated both on all 8,000 reflected
  forms directly, per the instruction to recompute rather than trust.
- The F11 → F15 = 0.772 overlap is fully attributed in §3.8 and in the
  artifact; it is definitional under the frozen contracts, report-only under
  the frozen standard, and no threshold, contract or library content was
  touched in response. The observation about a hypothetical future
  disjoint-label need is recorded for the reviewing chat, not acted on.
- The audit went beyond the letter in one place: `generation_seed` and
  `accepted_attempt_seed` were re-derived from identity inputs alone for all
  8,000 entries (0 mismatches), pinning recorded provenance without
  regenerating anything.

### 3.15 Completion gates

```text
agent_01_pass_verified                       true
agent_02_pass_verified                       true
contract_and_thresholds_frozen               true
handoff_digests_verified_before_audit        true
counts_exact                                 true   (8,000 / 500 x 16 / 6,400-800-800)
all_bases_engine_valid                       true   (0 / 8,000 failures)
all_reflections_engine_valid                 true   (0 / 8,000 failures)
zero_stranded_bases                          true   (base + reflected)
zero_family_contract_failures                true   (base + reflected)
zero_exact_duplicates                        true
zero_reflection_equivalent_duplicates        true
zero_stable_id_collisions                    true
zero_cross_split_equivalent_leakage          true
cross_split_nn_floor_met                     true   (21 >= 8)
all_diversity_thresholds_pass                true   (199 / 199)
serialization_exact                          true   (entries + JSONL lines)
reflection_roundtrips_exact                  true
canonicalization_exact                       true
fingerprints_and_traits_recomputed_exact     true
identity_split_seed_rederived_exact          true
independent_similarity_agrees                true   (0 mismatches)
family_self_satisfaction_diagonal_one        true   (1.0 x 16)
manifest_digests_verified                    true
production_library_untouched                 true   (byte-identical)
full_repository_suite_green                  true   (3018 / 3 / 0)
                                             25 / 25
```

No outcome, win rate, Elo, value or policy signal participated in any audit
decision. No frozen engine/evaluation/model/replay semantic changed, no
threshold moved, and no library content was repaired or regenerated.

### 3.16 Handoff to Agent 4

```text
audited library digest    7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777
entry metadata digest     d86f486182a820d546d470ef4ebce92ff60c6259aed80c481bc985bce8c64980
manifest digest           53139ab7e21c4e8a31987507d6fb1eabf93f36cdc1221fe85d08042963488f31
family metrics            agent_03_family_metrics.csv (every floor met)
similarity / leakage      agent_03_similarity_audit.csv (0 pairs < 10 across
                          splits; global min pairwise 20; cross-split min 21)
overlap matrix            audit artifact + §3.8 (largest F11 -> F15 = 0.772,
                          report-only, fully attributed)
reflection semantics      confirmed exact on all 8,000 (involution, class
                          fingerprints, family, mobility, split inheritance)
trait/family APIs         compute_trait_vector, evaluate_family,
                          class_fingerprint, reflect_canonical,
                          setup_has_initial_mobility, validate_perturbation —
                          all recomputation-verified against the library
```

Margins Agent 4 inherits: the minimum class distance between any two bases is
20 and the frozen perturbation bound is Hamming ≤ 12, so by the triangle
inequality no compliant descendant can come within class distance 8 of any
other base's equivalence class — perturbation cannot manufacture a duplicate
or a cross-split near-neighbour below the frozen floor. Agent 4 may not
change the 8,000-base library; the sampler must preserve base identity,
split, family, Flag cell, inventory, mobility, and the Hamming window
`[2, 12]` exactly as frozen.
