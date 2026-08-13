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
