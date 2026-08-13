# Phase 7 Agent 4 — Reflection and Procedural Perturbation

## Role

You are **Agent 4** in sequential Phase 7.

Agents 1–3 have frozen and audited the 8,000-base library.

Your task is to build the deterministic runtime setup sampler that expands effective diversity through reflection and constrained procedural perturbation.

Do not edit the base library.

Do not learn setup probabilities.

Do not begin collection-pipeline integration.

---

## Prerequisite

Agents 1–3 must all report `PASS`.

Read:

```text
00_PHASE_7_SEQUENCE_AND_COMMON_CONTRACT.md

reports/phase_7_implementation_report.md
reports/phase_7_data/agent_01_setup_contract.json
reports/phase_7_data/agent_01_diversity_thresholds.json
reports/phase_7_data/agent_02_base_library_manifest.json
reports/phase_7_data/agent_03_library_audit.json
reports/phase_7_data/agent_03_family_metrics.csv

data/setups/setup_library_v1.jsonl
data/setups/setup_library_v1_manifest.json

stratego/setups/
stratego/engine/
```

Verify the exact library digest before work.

Run the full suite before edits.

---

## Frozen inputs

Do not alter:

```text
8,000 base entries
base setup IDs
base split assignments
base primary families
family predicates
diversity thresholds
canonicalization
engine/rules
```

All generated descendants inherit the base:

```text
base_setup_id
primary_family_id
split
```

---

## Mission

Implement a deterministic runtime sampler supporting:

```text
choose split
choose family
choose base
optional constrained perturbation
choose left/right orientation
final frozen-engine validation
return setup + provenance
```

The sampler must be suitable for Phase 8 without exposing generation internals to the learner.

---

## Sampling contract

Provide a stable public API conceptually equivalent to:

```text
sample_setup(split, seed, profile=...)
```

and a lower-level deterministic rebuild API capable of reconstructing a sampled descendant from provenance.

The exact Python signature may follow repository conventions.

Returned provenance must contain at minimum:

```text
setup_library_version
sampler_version
split
primary_family_id
base_setup_id

reflection_applied

perturbation_applied
perturbation_version
perturbation_seed_or_id

final_setup_fingerprint
```

Do not include game outcomes or model scores.

---

## Family/base sampling

The initial neutral training profile must not accidentally overweight a family because one family has more generator candidates.

At minimum support:

```text
family selection       uniform over 16 families
base selection         uniform over 500 bases within chosen family/split subset
orientation            seeded 50/50 left-right
```

Because each family has the same base count per split, this also gives uniform base mass within a split when family and base are both uniform.

If you propose a non-uniform default perturbation intensity mix, justify it only from structural-diversity/coverage evidence.

Do not tune against game strength.

Agent 6 makes the final sampler-profile freeze.

---

## Constrained perturbation

Perturbation must be **family-preserving**, not arbitrary shuffling.

Use Agent 1's perturbation invariants.

Allowed techniques may include:

```text
role-compatible piece swaps
bounded within-rank swaps
bounded cross-rank swaps
local fortress variation
controlled Scout/Miner relocation
controlled decoy variation
```

only when those transformations preserve the primary-family required predicates.

Every final output must be revalidated from scratch.

Do not assume a perturbation is legal because its operation looked harmless.

---

## Perturbation identity

Define a versioned perturbation contract, expected conceptually as:

```text
setup_perturbation_v1
setup_sampler_v1
```

Given:

```text
base_setup_id
split
perturbation seed
reflection bit
sampler/profile version
```

the exact final setup must be reproducible.

Do not rely on mutable global RNG state.

---

## Final-output validation

Every sampler output must pass:

```text
exact inventory
engine setup validation
initial-mobility quality check
base split unchanged
base primary family unchanged
family required predicates
serialization/fingerprint
```

If a perturbation candidate fails, the rejection/retry process must itself be deterministic.

Record rejection reasons.

Do not return an invalid setup as a fallback.

---

## 100,000-sample stress corpus

Run at least **100,000 final sampled setups**.

Use a deterministic balanced acceptance corpus.

Preferred exact coverage:

```text
6,250 outputs per family = 100,000 total
```

and within each family approximately the library split ratio:

```text
5,000 train
625 validation
625 test
```

Exercise both:

```text
unperturbed/reflection-only path
perturbed path
```

and both reflection orientations.

The stress test is an acceptance instrument; its sampling mix may deliberately balance branches even if the final Phase 8 default profile differs.

---

## Hard stress requirements

Across the >=100,000 outputs require:

```text
engine-invalid setups          0
incorrect inventories          0
stranded outputs               0
primary-family violations      0
split changes                  0
serialization failures         0
reflection failures            0
deterministic rebuild failures 0
stable provenance failures     0
```

Also recompute the Agent 1 family-defining metrics on the stress outputs and verify the perturbation process has not collapsed the family structure.

---

## Effective diversity measurement

Report, by family and overall:

```text
unique final setup fingerprints
collision/repeat rate
unique outputs/base
unique trait signatures
distance from base
distance among perturbed descendants
reflection balance
perturbation acceptance/rejection rate
```

The goal is to show perturbation expands effective support.

Do not invent a strength claim.

If the random sample contains repeated descendants, report them honestly.

---

## Split isolation

Procedural descendants must never cross split boundaries.

Run explicit tests showing that changing:

```text
split="train" -> "validation"
```

changes the eligible base set rather than relabeling an identical base.

Search the stress corpus for final setup equivalence across splits using Agent 1 canonical similarity/equivalence rules.

Hard gate:

```text
exact/reflection-equivalent cross-split descendant leakage = 0
```

Apply Agent 1's near-duplicate leakage thresholds where relevant to descendants and report the result.

---

## Suggested files

```text
stratego/setups/perturbation.py
stratego/setups/sampler.py

tests/setups/test_perturbation.py
tests/setups/test_sampler.py

scripts/run_phase7_agent04.py
```

Do not modify the production base-library JSONL.

---

## Artifacts

Create:

```text
reports/phase_7_data/agent_04_sampler_contract.json
reports/phase_7_data/agent_04_procedural_stress.json
reports/phase_7_data/agent_04_procedural_family_metrics.csv
```

Append only:

```markdown
## 4. Agent 4 — Reflection and Procedural Perturbation
```

to:

```text
reports/phase_7_implementation_report.md
```

Report exact sampler/perturbation versions and seeds.

---

## Tests

Add regressions for:

- uniform family/base selection;
- split-restricted base selection;
- deterministic reflection;
- 50/50 orientation instrument behavior;
- deterministic perturbation;
- deterministic rejection/retry;
- family-preserving positive examples;
- rejection of family-breaking perturbations;
- rejection of stranded perturbations;
- inventory preservation;
- provenance rebuild exactness;
- no split migration;
- no model/outcome dependency;
- stress artifact consistency.

Run full suite after artifacts exist.

---

## Stop conditions

Report `BLOCKED` if:

- Agents 1–3 are not PASS;
- the audited library digest changed;
- Agent 1's perturbation invariants cannot be implemented without changing family semantics;
- deterministic retries cannot be made reproducible;
- no practical perturbation method can materially expand support while preserving legality/family/split;
- frozen engine changes appear necessary.

---

## PASS gates

PASS only if:

- Agents 1–3 PASS verified;
- base library unchanged;
- deterministic sampler API implemented;
- uniform family/base sampling supported;
- reflection deterministic and correct;
- constrained perturbation implemented;
- provenance rebuild exact;
- >=100,000 stress outputs produced;
- 0 legality/inventory failures;
- 0 stranded outputs;
- 0 family violations;
- 0 split migrations/leaks;
- 0 serialization/reflection failures;
- effective-diversity metrics reported;
- procedural support materially exceeds static base support under Agent 1 metrics;
- no outcome/strength input exists;
- full repository suite green.

---

## Handoff to Agent 5

Provide:

```text
sampler version
perturbation version
public sampling API
provenance schema
default candidate profile(s)
stress results
library digest
observer-safety expectations
```

Agent 5 must integrate the sampler without changing its semantics.
