# Phase 7 Sequential Agent Plan

## Goal

Build, validate, integrate, and freeze the first production **Stratego setup generator and setup library** that Phase 8 will use for synthetic warm-start data generation.

Phase 7 should answer:

- What exact contract defines a setup family, setup identity, split identity, reflection, and procedural perturbation?
- Can the project deterministically generate exactly 8,000 curated base setups across 16 strategic families?
- Are all curated setups legal under the frozen engine and strategically usable at game start?
- Is the base library sufficiently diverse under metrics and thresholds declared before generation?
- Can left-right reflection and constrained perturbation expand the effective setup support without changing family identity or leaking across train/validation/test splits?
- Can the accepted Phase 6 collection/persistence pipeline consume the setup library without changing engine, model, replay, or evaluation semantics?
- Can the whole library be regenerated bit-for-bit from its recorded contract, seed, and configuration?

Phase 7 is a **setup-distribution phase**, not a learning phase.

No meaningful neural-network training, reinforcement learning, setup-policy learning, setup-value learning, or outcome-based tuning is authorized.

---

## Agent sequence

| Agent | Task |
|---|---|
| 1 | Setup contract, canonical representation, 16-family taxonomy, split rules, diversity metrics and thresholds |
| 2 | Deterministic 8,000-base setup generator and materialized library |
| 3 | Independent exhaustive legality, duplicate/leakage, family, and diversity audit |
| 4 | Reflection + constrained procedural perturbation sampler and 100,000-sample stress validation |
| 5 | Integration with the accepted Phase 6 collection/persistence pipeline and provenance tracking |
| 6 | Independent regeneration, final acceptance audit, sampler-profile decision, and Phase 7 freeze recommendation |

Run strictly in order.

Do not begin a later agent unless every required earlier agent reports `PASS`.

No individual agent may formally declare Phase 7 complete. Agent 6 may recommend `PASS`, `FAIL`, or `BLOCKED`; the reviewing chat makes the final acceptance decision.

---

## Shared report

Create and maintain:

```text
reports/phase_7_implementation_report.md
```

Owned sections:

```text
# Phase 7 Implementation Report

## 1. Agent 1 — Setup Contract, Taxonomy, and Diversity Standard
## 2. Agent 2 — Deterministic Base Library Generator
## 3. Agent 3 — Exhaustive Library Audit
## 4. Agent 4 — Reflection and Procedural Perturbation
## 5. Agent 5 — Production Pipeline Integration
## 6. Agent 6 — Final Acceptance and Library Freeze
```

Agent 1 creates the report header if absent.

Later agents append only their own section. Do not rewrite accepted earlier sections except for an explicitly authorized factual correction; if a correction is necessary, preserve the original history and document the correction.

Every headline number in Markdown must also exist in a machine-readable artifact.

---

## Canonical Phase 7 data files

Required report artifacts:

```text
reports/phase_7_data/agent_01_setup_contract.json
reports/phase_7_data/agent_01_diversity_thresholds.json

reports/phase_7_data/agent_02_base_library_manifest.json
reports/phase_7_data/agent_02_generation_summary.json

reports/phase_7_data/agent_03_library_audit.json
reports/phase_7_data/agent_03_family_metrics.csv
reports/phase_7_data/agent_03_similarity_audit.csv

reports/phase_7_data/agent_04_sampler_contract.json
reports/phase_7_data/agent_04_procedural_stress.json
reports/phase_7_data/agent_04_procedural_family_metrics.csv

reports/phase_7_data/agent_05_pipeline_integration.json
reports/phase_7_data/agent_05_setup_provenance.csv

reports/phase_7_data/agent_06_final_acceptance.json
reports/phase_7_data/agent_06_library_regeneration.json
reports/phase_7_data/agent_06_sampler_profile.json
```

Optional raw CSV/JSON files may be added with the same `agent_0X_` prefix.

The materialized production library should live outside `reports/`. Preferred paths, unless the repository already has a stronger data convention:

```text
data/setups/setup_library_v1.jsonl
data/setups/setup_library_v1_manifest.json
```

Agent 1 may specify an equivalent deterministic representation if repository conventions justify it. Do not store the 8,000 production setups only inside a report artifact.

---

## Frozen project state

Phase 7 begins only after formal Phase 6 acceptance.

The frozen starting stack is:

```text
rules                       stratego_project_v1
reference engine            phase2_1_reference_1.2.0
observation                 observation_v2_1_127ch
engine action encoding      source_destination_10000_v1
engine action frame         absolute engine squares

model contract              model_contract_v2
model action frame          perspective_normalized_squares

primary model               C1
primary parameters          863,959
primary config digest       31ca84ab140c523e65567787b0289fe0dbdf5ab0344667410a5fda7060cfe07d

fallback model              C0
fallback parameters         123,223
fallback config digest      057d6c9242e328900f923d4e4c265eaba1bf95e57e1be120a024d2c42c143ddd

backend                     KEEP_PYTHON
trajectory                  trajectory_v1
zero-decision games         supported
collection precision        float16
evaluation reference        float32 + single_request

Phase 4 evaluation bank     evaluation_setup_bank_v1
Phase 4 pairing             color_swap_same_board
Phase 4 semantics           frozen
```

Phase 6's corrected reference engine remains the behavioral source of truth.

Do not modify `stratego/engine/` during Phase 7.

If a Phase 7 task appears to require an engine semantic change, report `BLOCKED` with a reproducer and exact conflict.

---

## Fixed Phase 7 library target

The base library target is fixed:

```text
library version             setup_library_v1
base setups                 8,000 exactly
strategic families          16 exactly
base setups / family        500 exactly
```

Fixed split allocation:

| Split | Per family | Total |
|---|---:|---:|
| train | 400 | 6,400 |
| validation | 50 | 800 |
| test | 50 | 800 |
| **total** | **500** | **8,000** |

Split assignment is permanent once Agent 2 materializes the accepted library.

All reflected or procedurally perturbed descendants inherit the split and primary-family identity of their base setup.

A train-derived setup must never become a validation/test setup merely because it is reflected or perturbed.

---

## Fixed Phase 7 family list

Use these 16 primary families:

| ID | Family |
|---|---|
| F00 | Corner Flag fortress |
| F01 | Near-corner Flag fortress |
| F02 | Central/back-row Flag fortress |
| F03 | Partially bombed Flag |
| F04 | Lightly defended / deceptive Flag |
| F05 | False fortress / Bomb decoy |
| F06 | Distributed Bomb defense |
| F07 | High Bomb placement |
| F08 | Aggressive high-rank front |
| F09 | Conservative high-rank rear |
| F10 | Scout-forward information |
| F11 | Scout-preservation |
| F12 | Miner-forward |
| F13 | Miner-preservation |
| F14 | Balanced conventional |
| F15 | Deliberately irregular / high-entropy |

Agent 1 must turn these names into explicit measurable family contracts before Agent 2 generates the production library.

The families do **not** have to be mathematically disjoint. A setup may satisfy secondary traits from other families.

Every setup must have exactly one declared primary family and a reproducible structural trait vector.

---

## Canonical setup orientation

Use the same conceptual own-orientation convention already established by the project:

```text
canonical own rank 0 = player's back row, furthest from lakes
canonical own rank 3 = player's front setup row, nearest lakes
files                 = 0..9 left-to-right in canonical own view
```

The frozen engine remains responsible for mapping a canonical supplied setup into the real board for Red or Blue.

Do not invent a competing board-orientation convention.

The setup itself is still the engine's 40 piece types assigned to the 40 setup squares in fixed row-major order.

---

## Reflection rule

Every accepted base setup must support deterministic **left-right reflection**.

Reflection does not create a new base-library identity.

Base-library uniqueness is defined over the reflection equivalence class:

```text
setup A
reflect(A)
```

must correspond to one canonical base representative, not two separately counted base entries.

Agent 1 must define the exact canonicalization/fingerprint rule.

Agent 2 must materialize only canonical representatives.

Runtime sampling may choose either orientation.

---

## Curated initial-mobility quality rule

The project rules allow any piece type anywhere within the legal four setup rows, and the corrected reference engine correctly handles a game that is terminal at creation because the first player has no legal move.

Phase 7 does **not** change those rules.

However, `setup_library_v1` is a curated strategic training library, so every accepted base setup and every generated descendant must satisfy an additional **library-quality** requirement:

```text
the setup has at least one legal move available for its owner
in the corresponding initial board geometry
```

This is a setup-library acceptance criterion, not an engine legality rule.

Do not reroll or redefine arbitrary engine setups globally.

If a generated Phase 7 setup is stranded, reject that generated candidate from the library/sampler and count the rejection.

Hard Phase 7 acceptance requires zero stranded setups in the final accepted base library and zero stranded outputs in the procedural stress corpus.

---

## Determinism and identity

Every source of setup randomness must be explicitly seeded.

Do not consume uncontrolled global RNG state.

The system must support isolated deterministic regeneration:

```text
contract + library version + master seed + family id + base index
    -> exact same base setup and metadata
```

without having to generate every preceding setup.

Agent 1 must define stable setup identity and fingerprint semantics.

Agent 2 must prove isolated rebuild for a broad deterministic sample.

Agent 6 must regenerate the complete library from scratch and require exact digest equality.

---

## Family-contract rule

Family definitions and diversity thresholds are **pre-generation contracts**.

Agent 1 owns them.

Once Agent 1 reports `PASS`:

- Agent 2 may not weaken them to make generation easier.
- Agent 3 may not reinterpret them after seeing the library.
- Agent 4 may not weaken them to make perturbation pass.
- Agent 6 may not move thresholds after seeing results.

If a family cannot produce 500 acceptable canonical bases under the approved contract, the correct outcome is `BLOCKED`, followed by review.

Do not silently relax a family definition.

---

## Diversity standard

Agent 1 must define exact metrics, methods, and numeric thresholds before the 8,000 production setups exist.

At minimum the standard must cover:

```text
exact duplicate identity
reflection-equivalent duplicate identity
cross-split equivalent leakage

within-family setup distance
cross-split nearest-neighbor distance
per-square piece-type entropy
Flag-position distribution
Bomb-position distribution
Scout front/rear distribution
Miner front/rear distribution
high-rank front/rear distribution
structural trait diversity
family overlap/confusion matrix
```

The metric must distinguish a strategically constrained family from a library that simply repeats a few arrangements.

It is acceptable and expected that some families have low entropy in strategically defining locations.

Do not require every family to look uniformly random.

---

## Phase 4 evaluation bank is separate

`evaluation_setup_bank_v1` remains frozen and untouched.

It is not replaced by the Phase 7 library and is not retroactively regenerated from it.

Phase 7 may inspect Phase 4 setup-bank code as implementation precedent for:

- canonical orientation;
- engine validation;
- deterministic seeding;
- serialization;
- isolated rebuild.

But the Phase 4 bank must remain a separate evaluation fixture and must not become a training source by accident.

Agent 5 must prove its digest/identity is unchanged.

---

## No outcome-based setup tuning

No agent may use:

```text
random-model win rate
C1/C0 win rate
baseline win rate
Elo
game outcome
value prediction
policy score
future learner score
human preference
```

to decide whether a base setup or family is accepted, rejected, weighted, or moved between splits.

Phase 7 selects setups using structural contracts and diversity only.

Learned setup selection belongs to a later phase.

---

## No setup neural network

Do not build:

- an autoregressive setup Transformer;
- a learned family selector;
- a learned template selector;
- a setup-value model;
- a setup-entropy model.

The Ataraxos-style learned setup process remains a later research direction.

The first move learner will consume this structured generator directly.

---

## Observer-safety and provenance

The training system may know the true setups because they are part of privileged replay/training state.

The move policy must not receive:

- opponent true setup;
- opponent setup-family id;
- opponent base-setup id;
- opponent perturbation seed;
- any setup provenance that reveals hidden identities.

Phase 7 provenance is training/debug metadata, not a model input.

Agent 5 must prove setup provenance does not cross the observer-safe model boundary.

Do not change `observation_v2_1_127ch` to include setup-family metadata.

---

## Trajectory and persistence

Do not change `trajectory_v1` merely to store Phase 7 provenance.

The trajectory already carries the actual setup needed for deterministic replay.

Preferred provenance design:

```text
game_id -> setup-library metadata sidecar / shard metadata
```

with, at minimum, per player:

```text
setup_library_version
primary_family_id
base_setup_id
split
reflected
perturbation_applied
perturbation_id_or_seed
final_setup_fingerprint
```

If the repository has a stronger existing sidecar/manifest convention, reuse it.

A Phase 7 provenance feature must not invalidate prior Phase 6 trajectory bytes.

---

## Shared correctness rules

- Read the real repository and every prerequisite report/artifact before editing.
- Run the full repository suite before each agent's changes and record exact totals.
- Preserve all prior tests.
- Add regression tests for every newly discovered bug.
- Do not weaken existing tests or acceptance thresholds.
- Do not modify `stratego/engine/`.
- Do not alter rules, combat, terminal precedence, observation semantics, action semantics, replay semantics, or Phase 4 evaluation semantics.
- The frozen engine remains the legality/setup-validation authority.
- No hidden setup truth or provenance may enter move-model inputs.
- Every RNG source must be seeded and reproducible.
- Do not use game strength/outcomes as setup-selection evidence.
- Do not begin Phase 8.
- No meaningful model training is authorized.
- Do not edit historical Phase 6 report sections except an explicitly authorized factual correction.
- Required external-drive work must honestly report whether the intended volume was actually available; never simulate external persistence and call it validated.

---

## Common machine-readable metadata

Every primary Agent 1–6 JSON should record at least:

```text
agent
phase
status
timestamp
commit
working_tree_state
platform
python_version
torch_version
mps_built
mps_available

prerequisite_status
frozen_versions
tests_before
tests_after
commands
durations
seeds

files_created
files_modified

completion_gates
problems
deviations
```

Where applicable also record:

```text
library_version
contract_version
family_contract_version
sampler_version
master_seed
library_digest
manifest_digest
setup_count
split_counts
family_counts
```

---

## General stop conditions

Mark `BLOCKED` and stop if:

- a prerequisite agent is not `PASS`;
- the pre-existing suite is red for a non-environmental reason;
- the frozen engine/rules/observation/replay/evaluation semantics would need modification;
- a family definition is ambiguous enough that two reasonable implementations would materially differ;
- a required family cannot produce 500 canonical legal mobile bases without weakening its accepted contract;
- stable identity/canonicalization is non-deterministic;
- train/validation/test split leakage cannot be prevented;
- perturbation cannot preserve family identity and legality;
- pipeline integration requires exposing setup truth/provenance to the move model;
- satisfying one acceptance condition necessarily violates another frozen guarantee.

For an ordinary bug in new Phase 7 code:

1. fix it;
2. add a regression test;
3. rerun focused tests;
4. rerun the full suite;
5. document it.

---

## Global Phase 7 acceptance

Phase 7 may be recommended `PASS` only if all of the following are true:

```text
base setups                              8,000 exactly
strategic families                       16 exactly
base setups/family                       500 exactly

train bases                              6,400 exactly
validation bases                           800 exactly
test bases                                 800 exactly

incorrect inventory                      0
engine-invalid bases                     0
stranded curated bases                   0

exact duplicate bases                    0
reflection-equivalent duplicate bases    0
cross-split equivalent leakage           0

family-contract violations               0
stable-ID collisions                     0

deterministic full-library regen diff    0
reflection round-trip failures           0

all Agent-1 diversity thresholds         PASS

procedural stress samples                >=100,000
procedural legality failures             0
procedural inventory failures            0
procedural stranded outputs              0
procedural family violations             0
procedural split leakage                  0

pipeline setup/provenance mismatches      0
trajectory/reconstruction mismatches      0
observer-safety violations                0

Phase 4 evaluation bank changed           NO
engine/rules changed                      NO
observation/model contract changed        NO
trajectory_v1 semantics changed           NO
meaningful neural training occurred       NO

full repository suite                     GREEN
```

Phase 7's final accepted manifest must record the exact contract/library/sampler versions, master seed, split map, library digest, and default Phase 8 training sampling profile.

---

## Expected Phase 7 output for Phase 8

After formal Phase 7 acceptance, Phase 8 should be able to request:

```text
sample_setup(
    split="train",
    seed=...,
)
```

and receive a deterministic setup plus provenance generated under the frozen Phase 7 sampler.

Phase 8 should not need to know how a family is constructed internally.

Phase 8 should not modify Phase 7 library definitions or split assignments.

Stop Phase 7 after Agent 6's handoff. Do not begin synthetic warm-start training.
