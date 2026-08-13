# Phase 7 Agent 2 — Deterministic Base Library Generator

## Role

You are **Agent 2** in sequential Phase 7.

Work from the repository root.

Complete only the deterministic base-library generation task.

Do not begin Agent 3's independent audit or Agent 4's procedural sampler.

---

## Prerequisite

Agent 1 must report `PASS`.

Read:

```text
00_PHASE_7_SEQUENCE_AND_COMMON_CONTRACT.md
reports/phase_7_implementation_report.md
reports/phase_7_data/agent_01_setup_contract.json
reports/phase_7_data/agent_01_diversity_thresholds.json

stratego/setups/
stratego/engine/
stratego/evaluation/setup_bank.py
```

Verify the live Agent 1 contract versions and threshold digests against its artifacts.

Run the full repository suite before edits and record exact totals.

---

## Frozen contracts

Do not alter:

- any Phase 6 frozen contract;
- Agent 1 family predicates;
- Agent 1 diversity metrics/thresholds;
- Agent 1 split semantics;
- Agent 1 canonicalization/identity rules.

If a family cannot produce 500 valid canonical bases under those rules, report `BLOCKED`.

Do not weaken the contract.

---

## Mission

Implement one deterministic generator capable of producing the complete:

```text
8,000-base setup_library_v1
```

with:

```text
16 families
500 canonical bases/family
400 train + 50 validation + 50 test per family
```

Materialize the production library and manifest.

The generator must support isolated rebuild of any entry directly from recorded identity/seed inputs.

---

## One generator framework, not 8,000 hard-coded boards

Do not hand-author 8,000 literal setup arrays.

Do not create 16 unrelated generator programs with inconsistent identity/randomness rules.

Use a common deterministic generation framework with family-specific constraints driven by Agent 1's frozen family definitions.

Family-specific logic is expected; identity, seed derivation, engine validation, canonicalization, deduplication, metadata, and serialization must be common infrastructure.

---

## Master seed and isolated regeneration

Declare one Phase 7 base-library master seed.

Do not choose it from game outcomes.

Use deterministic seed derivation so:

```text
master_seed
family_id
base_index / candidate_attempt_index
library_version
```

determine a candidate independently.

The exact derivation must be stable and recorded.

Do not use enumeration-order RNG consumption such that adding a rejected candidate in F03 changes every later family.

Required capability:

```text
rebuild_base_setup(family_id, base_index)
```

or an equivalent API that reproduces the exact accepted setup and metadata without generating the entire library.

If rejection sampling is used, record enough deterministic attempt information to rebuild the accepted entry exactly.

---

## Generation requirements

For each family, generate exactly 500 accepted canonical base setups.

Every accepted base must:

```text
match exact official piece inventory
pass frozen engine setup validation
occupy only the legal four-row setup zone
have no lake/non-setup placement
pass Agent 1 primary-family predicate
produce Agent 1 structural trait vector
pass Agent 1 initial-mobility quality check
be canonical under Agent 1 reflection rule
have a unique stable base_setup_id
have a unique canonical content fingerprint
have a stable split assignment
```

Reject invalid candidate attempts explicitly.

Track rejection reasons.

Do not silently repair an invalid candidate after the fact unless the deterministic family-generation algorithm itself specifies that repair before generation.

---

## Global uniqueness

Before accepting a base, reject if its canonical fingerprint is already present anywhere in the 8,000-base library, including another family.

This enforces:

```text
exact duplicate bases                  0
reflection-equivalent duplicate bases  0
```

globally.

Primary families may overlap structurally, but one canonical board may not be counted twice.

---

## Split handling

Materialize exactly:

```text
F00: 400 train / 50 validation / 50 test
...
F15: 400 train / 50 validation / 50 test
```

Do not assign splits from acceptance order unless Agent 1's frozen rule explicitly says so.

Do not alter split assignment to improve diversity after generation unless that behavior is already part of Agent 1's deterministic split contract.

No setup may appear in more than one split.

---

## Canonical stored form

Store only Agent 1's canonical left-right representative.

Do not double the library to 16,000 entries by materializing reflection as a second base.

The runtime reflection choice is Agent 4's responsibility.

For every base, verify:

```text
canonicalize(setup) == setup
canonicalize(reflect(setup)) == setup
```

and record the reflected fingerprint if the contract requires it.

---

## Production entry metadata

Every base-library entry must contain at least:

```text
setup_library_version
setup_generator_contract_version
setup_family_version
trait_schema_version if present

base_setup_id
primary_family_id
base_index
split

master_seed
derivation seed / accepted attempt identity

canonical setup serialization
canonical setup fingerprint
reflection-equivalence fingerprint

structural trait vector
```

Do not embed game outcomes or model scores.

---

## Materialized library

Preferred:

```text
data/setups/setup_library_v1.jsonl
data/setups/setup_library_v1_manifest.json
```

The manifest should contain at minimum:

```text
library_version
generator_contract_version
family_version
trait_schema_version
master_seed
entry_count
family_counts
split_counts
family_split_counts
library_content_digest
entry_metadata_digest
generation command
generation duration
rejection counts by family/reason
```

Choose deterministic serialization so rewriting the same library produces the same content digest.

If timestamps are included, keep them outside the content-hash domain.

---

## Base-generation diversity checks

Agent 3 owns the independent acceptance audit, but Agent 2 must run generation-time checks to avoid knowingly handing off a broken library.

At minimum compute and report:

```text
family counts
split counts
exact duplicates
reflection duplicates
stable-ID collisions
stranded bases
family-predicate failures
basic trait distributions
```

You may also compute Agent 1 diversity metrics as a preflight.

Do not declare final diversity acceptance; Agent 3 owns that independent verdict.

---

## Performance

This phase is not a throughput optimization contest.

Measure:

```text
full-library generation wall time
candidate attempts per accepted base
rejection rate by family
peak process RSS
materialized library bytes
manifest bytes
```

The generator should be practical to regenerate during development.

Do not compromise determinism or contract clarity for speed.

---

## Suggested files

Suggested:

```text
stratego/setups/generator.py
stratego/setups/library.py
stratego/setups/seed.py

tests/setups/test_generator.py
tests/setups/test_library.py

scripts/run_phase7_agent02.py

data/setups/setup_library_v1.jsonl
data/setups/setup_library_v1_manifest.json
```

Reuse Agent 1 modules rather than duplicating family/identity logic.

---

## Required tests

Add regressions for:

- deterministic master seed;
- isolated rebuild;
- regeneration independent of enumeration order;
- exact 500/family;
- exact 400/50/50 per family;
- all accepted bases engine-valid;
- all accepted bases initially mobile by the Agent 1 quality rule;
- all accepted bases satisfy primary-family contract;
- every stored setup is canonical;
- reflection canonicalizes back to same base;
- exact duplicate rejection;
- reflection-equivalent duplicate rejection;
- stable ID collision rejection;
- deterministic serialization;
- same run -> same library digest;
- different master seed -> changed library content;
- artifact manifest matches materialized library;
- no game-result/strength field participates in generation.

Run the full repository suite after all production artifacts exist so artifact-validation tests execute.

---

## Shared reporting contract

Create:

```text
reports/phase_7_data/agent_02_base_library_manifest.json
reports/phase_7_data/agent_02_generation_summary.json
```

Append only:

```markdown
## 2. Agent 2 — Deterministic Base Library Generator
```

to:

```text
reports/phase_7_implementation_report.md
```

Report:

```text
status
prerequisite verification
master seed
generator architecture
materialized paths
library digest
manifest digest
8,000 count
16 x 500 counts
split counts
rejection counts
generation duration
memory
generation-time correctness checks
tests
files
deviations
Agent 3 handoff
```

---

## Stop conditions

Report `BLOCKED` if:

- Agent 1 is not PASS;
- Agent 1 artifacts and live contract disagree;
- any family cannot produce 500 unique canonical legal mobile bases without weakening the frozen contract;
- deterministic isolated regeneration cannot be achieved;
- the library cannot satisfy exact split counts;
- canonicalization produces ambiguous identities;
- producing the library would require changing the frozen engine.

Ordinary candidate rejections are not blockers.

---

## PASS gates

PASS only if:

- Agent 1 PASS verified;
- pre-existing suite green;
- exactly 8,000 bases materialized;
- exactly 500/family;
- exactly 400/50/50 per family;
- 0 engine-invalid accepted bases;
- 0 incorrect inventories;
- 0 stranded accepted bases;
- 0 primary-family violations;
- 0 exact duplicate canonical bases;
- 0 reflection-equivalent duplicate bases;
- 0 stable-ID collisions;
- isolated rebuild exact;
- full regeneration digest stable;
- production library and manifest written;
- no outcome/strength signal used;
- full repository suite green.

---

## Handoff to Agent 3

Provide:

```text
library path
manifest path
library digest
master seed
generation command
contract versions
independent engine-validation API
identity/canonicalization API
trait API
Agent 1 diversity-threshold artifact
```

Agent 3 must be able to audit the library without relying on Agent 2's acceptance counters.
