# Phase 7 Agent 1 — Setup Contract, Taxonomy, and Diversity Standard

## Role

You are **Agent 1** in a sequential Phase 7 implementation for the Stratego project.

Work from the **root of the Stratego repository**.

Complete only the work assigned here.

Do not generate the final 8,000 production setups.

Do not begin Agent 2's base-library implementation.

---

## Required reading

Before editing, read:

```text
00_PHASE_7_SEQUENCE_AND_COMMON_CONTRACT.md

stratego_project_docs/02_project_ruleset.md
stratego_project_docs/03_game_engine_spec.md
stratego_project_docs/05_project_plan.md

reports/phase_4_implementation_report.md
reports/phase_4_data/agent_01_setup_bank_v1.json

reports/phase_6_implementation_report.md
reports/phase_6_data/

stratego/engine/
stratego/evaluation/setup_bank.py
tests/evaluation/test_setup_bank.py
```

If the project documentation lives under different but equivalent repository paths, use the actual repository structure and record it.

Verify from the live repository that Phase 6 has been formally accepted and that the frozen reference engine is `phase2_1_reference_1.2.0`.

Run the full repository suite before edits and record exact totals.

---

## Frozen contracts

Do not alter:

```text
rules                       stratego_project_v1
reference engine            phase2_1_reference_1.2.0
observation                 observation_v2_1_127ch
engine action encoding      source_destination_10000_v1
model contract              model_contract_v2
primary model               C1
fallback model              C0
backend                     KEEP_PYTHON
trajectory                  trajectory_v1
Phase 4 evaluation bank     evaluation_setup_bank_v1
Phase 4 evaluation semantics
```

Do not modify `stratego/engine/`.

---

## Mission

Define the **pre-generation contract** that every later Phase 7 agent must obey.

Your deliverable is not 8,000 setups.

Your deliverable is a sufficiently exact specification that Agent 2 can generate them and Agent 3 can independently decide whether they pass.

The contract must define:

1. setup-library identity;
2. canonical setup representation;
3. stable setup identifiers and fingerprints;
4. left-right reflection and canonicalization;
5. train/validation/test assignment;
6. the 16 primary family definitions;
7. structural trait-vector semantics;
8. base-setup quality requirements;
9. procedural-perturbation invariants;
10. diversity metrics;
11. numeric diversity thresholds;
12. final artifact/serialization contract.

Once you report `PASS`, Agents 2–6 must treat the thresholds and family contracts as frozen.

---

## Fixed library size and split

You must encode these exact requirements:

```text
library_version          setup_library_v1
base setups              8,000
families                 16
bases/family             500

train/family             400
validation/family         50
test/family               50

train total             6,400
validation total          800
test total                800
```

Do not change those counts.

---

## Fixed primary family IDs

Define formal contracts for:

```text
F00 corner_flag_fortress
F01 near_corner_flag_fortress
F02 central_back_flag_fortress
F03 partially_bombed_flag
F04 lightly_defended_deceptive_flag
F05 false_fortress_bomb_decoy
F06 distributed_bomb_defense
F07 high_bomb_placement
F08 aggressive_high_rank_front
F09 conservative_high_rank_rear
F10 scout_forward_information
F11 scout_preservation
F12 miner_forward
F13 miner_preservation
F14 balanced_conventional
F15 irregular_high_entropy
```

The display names may remain human-readable; the machine IDs above should be stable unless an existing repository naming rule requires a documented equivalent.

---

## Family-contract requirements

Every family definition must be inspectable from the setup itself.

Do not define a family as:

```text
"whatever generator branch F08 produced"
```

For every family define, at minimum:

```text
family_id
display_name
purpose
required structural predicates
allowed ranges
forbidden contradictions
primary diagnostic metrics
secondary trait expectations
reflection invariance rule
perturbation invariants
```

Examples of measurable concepts you may use:

```text
Flag rank/file region
Bomb adjacency to Flag
Bomb concentration / dispersion
Bomb front-rank count
high-rank front/back concentration
Scout front/back concentration
Miner front/back concentration
decoy fortress geometry
local defensive density
piece-type positional entropy
rank-weighted mobility support
```

Use the project's actual piece inventory and canonical own-orientation frame.

Do not infer Flag/Bomb rank from enum ordinal.

Families may overlap in secondary traits.

Every setup has one primary family; Agent 3 will report cross-family predicate overlap as a confusion/overlap matrix.

---

## Canonical representation

Specify one setup representation compatible with the frozen engine's setup contract.

Use the engine's established 40-square row-major setup representation unless there is a documented repository reason not to.

Define:

```text
canonical own-orientation frame
serialization
deserialization
content fingerprint
stable base_setup_id
family/base index mapping
```

Identity must not depend on process enumeration order.

Prefer content-addressed fingerprints plus semantic IDs, for example conceptually:

```text
semantic id      family + base index + library version
content digest   hash of canonical serialized setup
```

Do not rely on Python's process-randomized `hash()`.

---

## Reflection and canonicalization

Define left-right reflection exactly.

Require:

```text
reflect(reflect(setup)) == setup
```

A base setup and its reflection belong to one equivalence class.

Define one deterministic canonical representative of that class.

Agent 2 must not store both as separate base entries.

Agent 3 must be able to independently recompute canonicalization and detect reflection-equivalent duplicates.

Pin representative examples including:

```text
left corner <-> right corner
file 0 <-> file 9
file 4 <-> file 5
Flag/Bomb adjacency
front/back ranks unchanged
piece inventory unchanged
```

---

## Split assignment

Define a deterministic split rule that produces exactly:

```text
400 train / 50 validation / 50 test per family
```

Split identity must be stable and recorded in each base entry.

Do not assign splits after looking at game strength or model results.

Reflections and perturbations inherit the base split.

Define how split assignment interacts with base indices and isolated regeneration.

---

## Initial-mobility quality rule

Encode the Phase 7 library-quality requirement:

```text
every accepted base setup must have at least one legal move
available for its owner in initial board geometry
```

This is not a change to `stratego_project_v1`.

The frozen engine may still accept arbitrary legal setups that are terminal at creation.

The Phase 7 library simply excludes stranded templates.

Specify exactly how Agents 2–4 must test this using the frozen engine's legality authority.

Do not create a competing hand-written movement implementation solely for this check.

---

## Structural trait vector

Define a deterministic trait vector for every setup.

At minimum include enough components to support:

```text
family validation
diversity measurement
family overlap analysis
future stratified evaluation
```

Recommended categories:

```text
Flag location class
Bomb adjacency count
Bomb rank histogram
Bomb dispersion/concentration
Scout rank histogram
Miner rank histogram
Marshal/General rank locations
high-rank front/back counts
movable front-rank count
front-row immovable count
local defensive density around Flag
decoy/fortress indicators
piece-type positional entropy summary
```

Every trait must have explicit units/meaning.

Do not include playing strength or neural outputs.

---

## Diversity metrics

Define exact algorithms and numeric thresholds **now, before Agent 2 produces the library**.

The standard must include at least:

### Identity / leakage

```text
exact duplicates
reflection-equivalent duplicates
cross-split exact/reflection equivalents
stable-ID collisions
```

Hard threshold for all: zero.

### Setup distance

Define a canonical setup-distance measure, preferably piece-type Hamming distance over the 40 canonical setup squares, plus any secondary structural distance you find useful.

Predeclare thresholds for:

```text
within-family nearest-neighbor distance
cross-split nearest-neighbor distance
fraction of pairs below a near-duplicate threshold
```

The cross-split rule should be strong enough to prevent validation/test sets from being trivial one-swap variants of train setups.

### Entropy / positional coverage

Define numeric metrics and minimum/maximum acceptable ranges as appropriate for:

```text
per-square piece entropy
Flag location support
Bomb position support
Scout position support
Miner position support
high-rank position support
```

Do not demand uniform entropy in families whose identity deliberately constrains certain squares.

Thresholds may be family-specific.

### Trait diversity

Define within-family variation thresholds over the structural trait vector.

A family must not satisfy its count target by repeating one structural pattern with cosmetic swaps.

### Family overlap

Define a report-only overlap/confusion metric.

Primary-family predicate failure is a hard failure.

Satisfying another family's secondary predicates is not automatically a failure.

---

## Perturbation invariants

Agent 4 will design the sampler, but you must define what perturbation is allowed to preserve.

At minimum require:

```text
exact inventory
legal setup-zone occupancy
canonical orientation semantics
base split
base primary family
initial mobility
family required predicates
stable provenance
deterministic output from seed
```

Define whether specific traits may change within ranges and which defining traits are immutable.

Do not prescribe a learned perturbation policy.

---

## Versioned contracts

Expected version identities:

```text
setup_generator_contract_v1
setup_family_v1
setup_library_v1
```

You may introduce an explicit trait-schema version such as:

```text
setup_trait_vector_v1
```

if helpful.

A future semantic change to family predicates, split semantics, canonicalization, identity, or perturbation invariants must require a new version rather than silently reinterpreting `v1`.

---

## Suggested implementation ownership

Suggested files:

```text
stratego/setups/__init__.py
stratego/setups/contracts.py
stratego/setups/families.py
stratego/setups/traits.py
stratego/setups/identity.py

tests/setups/test_contracts.py
tests/setups/test_families.py
tests/setups/test_identity.py
tests/setups/test_traits.py

scripts/run_phase7_agent01.py
```

This list is guidance, not a requirement to create needless modules.

Do not implement the full 8,000-library generator.

A small number of hand-constructed fixtures or temporary generated examples are allowed only to prove the contracts are executable.

---

## Tests

Add tests for at least:

- all 16 family IDs present exactly once;
- contract serialization round-trip;
- exact fixed counts/splits;
- reflection involution;
- canonicalization stability;
- stable semantic ID generation;
- content fingerprint determinism;
- no use of process-randomized hash;
- family predicates on positive fixtures;
- negative fixture per family where practical;
- trait-vector deterministic values;
- mobility-quality rule delegates to engine legality;
- perturbation invariants are explicit;
- every diversity metric/threshold is serializable and executable;
- thresholds are defined before production library artifacts exist.

Run the full suite after changes.

---

## Shared reporting contract

Create:

```text
reports/phase_7_data/agent_01_setup_contract.json
reports/phase_7_data/agent_01_diversity_thresholds.json
```

Append only:

```markdown
## 1. Agent 1 — Setup Contract, Taxonomy, and Diversity Standard
```

to:

```text
reports/phase_7_implementation_report.md
```

The report must include:

```text
status
frozen-version verification
contract versions
family table
canonical representation
reflection/canonicalization
stable ID/fingerprint rule
split rule
mobility quality rule
trait schema
diversity metrics
all numeric thresholds
tests
files
deviations
completion gates
Agent 2 handoff
```

---

## Stop conditions

Report `BLOCKED` if:

- Phase 6 is not formally accepted;
- frozen engine/setup semantics are insufficient without modification;
- canonical orientation/reflection cannot be defined consistently with the engine;
- two project documents conflict materially on setup legality or setup representation;
- a family name cannot be converted into a measurable contract without an unresolved design decision;
- meaningful diversity thresholds cannot be defined before generation.

Do not resolve a material ambiguity by quietly inventing a rule.

---

## PASS gates

PASS only if:

- Phase 6 acceptance verified;
- pre-existing suite green;
- all 16 family contracts explicit;
- 8,000/500-per-family/400-50-50 split contract explicit;
- canonical representation explicit;
- stable identity/fingerprint explicit;
- reflection/canonicalization exact;
- initial-mobility quality rule executable through frozen engine authority;
- trait-vector schema explicit;
- diversity metrics defined;
- all numeric diversity thresholds frozen before production generation;
- perturbation invariants explicit;
- no strength/outcome input exists in acceptance logic;
- no frozen engine/evaluation/model/replay semantic changed;
- full repository suite green.

---

## Handoff to Agent 2

Give Agent 2:

```text
contract versions
master-seed input contract
all 16 family predicates
trait computation API
canonicalization API
stable ID/fingerprint API
split assignment rule
mobility-quality check
all frozen diversity thresholds
expected production library representation
```

Agent 2 must be able to generate the library without making a new family-design decision.
