# Phase 7 Agent 3 — Exhaustive Library Audit

## Role

You are **Agent 3** in sequential Phase 7.

Your role is independent validation.

Do not redesign the generator.

Do not modify Agent 1 thresholds to fit Agent 2's output.

Do not begin procedural perturbation.

---

## Prerequisite

Agents 1 and 2 must report `PASS`.

Read:

```text
00_PHASE_7_SEQUENCE_AND_COMMON_CONTRACT.md

reports/phase_7_implementation_report.md
reports/phase_7_data/agent_01_setup_contract.json
reports/phase_7_data/agent_01_diversity_thresholds.json
reports/phase_7_data/agent_02_base_library_manifest.json
reports/phase_7_data/agent_02_generation_summary.json

data/setups/setup_library_v1.jsonl
data/setups/setup_library_v1_manifest.json

stratego/setups/
stratego/engine/
```

Verify content digests before auditing.

Run the full repository suite before audit-code changes.

---

## Frozen inputs

Treat these as immutable audit inputs:

```text
Agent 1 family contracts
Agent 1 diversity metrics/thresholds
Agent 1 identity/canonicalization/split rules
Agent 2 materialized base library
Agent 2 manifest
```

If the library fails, report the failure.

Do not repair it inside the auditor.

---

## Mission

Independently audit **all 8,000 bases and all 8,000 left-right reflections** for:

1. exact engine legality/inventory;
2. initial mobility;
3. identity/canonicalization;
4. split correctness;
5. family-contract correctness;
6. duplicates and reflection duplicates;
7. cross-split near-duplicate leakage;
8. Agent 1 diversity thresholds;
9. serialization/reflection round-trips;
10. reproducible metrics.

The auditor should recompute facts from the library and frozen engine, not trust Agent 2 counters.

---

## Exhaustive legality audit

For each of the 8,000 bases:

```text
deserialize
validate exact inventory
validate setup placement
validate canonical form
compute stable fingerprint
compute trait vector independently
check primary-family predicate
check initial mobility through frozen engine authority
serialize -> deserialize exact round trip
reflect -> reflect exact round trip
canonicalize(reflection) -> same canonical base
```

Also validate the 8,000 reflected forms against the engine.

Required accepted results:

```text
base engine validation failures        0 / 8,000
reflected validation failures          0 / 8,000
inventory failures                     0
placement failures                     0
initial-mobility failures              0
family-predicate failures              0
serialization failures                 0
reflection round-trip failures         0
canonicalization failures              0
```

---

## Count audit

Independently recompute:

```text
total bases
family counts
split counts
family x split counts
stable ID uniqueness
canonical fingerprint uniqueness
```

Require exactly:

```text
8,000 total
500/family
6,400 train
800 validation
800 test
400/50/50 per family
```

---

## Duplicate audit

Detect globally:

```text
exact duplicate canonical setups
reflection-equivalent duplicate setups
same stable ID -> different setup
different stable ID -> same equivalence class
```

All must be zero.

Do not limit duplicate search to within-family comparisons.

---

## Cross-split leakage audit

Use Agent 1's frozen similarity metric and thresholds.

At minimum compute:

```text
train <-> validation nearest-neighbor distances
train <-> test nearest-neighbor distances
validation <-> test nearest-neighbor distances
```

within each family and globally where meaningful.

Report:

```text
minimum
median
percentiles specified by Agent 1
count below near-duplicate threshold
offending IDs if any
```

Require Agent 1's thresholds exactly.

Do not silently treat reflected forms as different when computing similarity.

Compare canonical equivalence classes.

---

## Within-family diversity

For every family compute all Agent 1 metrics, including at least:

```text
canonical setup-distance distribution
nearest-neighbor distance distribution
unique structural trait signatures
per-square piece-type entropy
Flag-position distribution
Bomb-position distribution
Scout rank distribution
Miner rank distribution
high-rank front/rear distribution
movable front-rank distribution
family-specific defining traits
```

Report metric values against the frozen thresholds.

A family may intentionally constrain defining squares.

Do not penalize legitimate low entropy that Agent 1 explicitly allowed.

---

## Between-family analysis

Compute a descriptive family overlap/confusion matrix.

For each base:

```text
declared primary family
which other family predicates also pass
```

Report:

```text
primary-family failures       hard gate
secondary overlaps            descriptive
```

Also report family-level trait centroids/distributions if Agent 1 specifies them.

Do not require families to be disjoint unless Agent 1 explicitly required that.

---

## Reflection audit

For every base verify:

```text
reflect twice = original
reflection preserves inventory
reflection preserves engine legality
reflection preserves mobility quality
reflection preserves primary-family validity
reflection preserves split
reflection changes no base identity semantics
```

Report how many bases are exactly reflection-symmetric.

Self-symmetric setups are allowed unless Agent 1 forbids them.

They still count as one base equivalence class.

---

## Independent metric implementation

Where practical, implement audit-side metric computation without calling Agent 2's precomputed acceptance result.

It is acceptable to reuse Agent 1's authoritative family/trait definitions.

Do not reuse Agent 2's cached metric values as the only proof.

At minimum recompute raw setup-derived inputs.

---

## Performance

Report:

```text
audit wall time
peak RSS
pairwise/similarity method
number of comparisons or index operations
```

Do not weaken the audit solely because exact comparison is somewhat expensive.

8,000 setups is intentionally small enough for thorough validation.

---

## Suggested files

```text
stratego/setups/audit.py

tests/setups/test_audit.py

scripts/run_phase7_agent03.py
```

Do not modify the production library.

---

## Required artifacts

Create:

```text
reports/phase_7_data/agent_03_library_audit.json
reports/phase_7_data/agent_03_family_metrics.csv
reports/phase_7_data/agent_03_similarity_audit.csv
```

Append only:

```markdown
## 3. Agent 3 — Exhaustive Library Audit
```

to:

```text
reports/phase_7_implementation_report.md
```

Every threshold should be shown as:

```text
metric
required threshold
measured value
PASS/FAIL
```

---

## Tests

Add tests proving the auditor catches deliberately injected:

- wrong inventory;
- illegal placement;
- stranded setup;
- wrong family label;
- exact duplicate;
- reflected duplicate;
- ID collision;
- split-count mismatch;
- cross-split near duplicate;
- threshold failure;
- bad serialization;
- bad reflection;
- altered manifest digest.

The production 8,000-base audit itself should also be exercised by the acceptance script.

Run the full suite after artifacts exist.

---

## Stop conditions

Report `BLOCKED` if:

- Agents 1 or 2 are not PASS;
- the production library digest does not match Agent 2's handoff before audit;
- Agent 1's thresholds are not executable or are internally contradictory;
- the audit requires changing frozen engine semantics.

A library failing a threshold is normally `FAIL`, not `BLOCKED`.

---

## PASS gates

PASS only if:

- Agents 1–2 PASS verified;
- full base count and split counts exact;
- all 8,000 bases engine-valid;
- all 8,000 reflections engine-valid;
- 0 inventory/placement failures;
- 0 stranded bases;
- 0 family-contract failures;
- 0 exact duplicates;
- 0 reflection-equivalent duplicates;
- 0 stable-ID collisions;
- 0 cross-split equivalent leakage;
- every Agent 1 similarity/leakage threshold passes;
- every Agent 1 family-diversity threshold passes;
- serialization/reflection/canonicalization exact;
- full suite green.

---

## Handoff to Agent 4

Provide:

```text
audited library digest
family metrics
similarity/leakage results
family-overlap matrix
confirmed reflection semantics
confirmed trait/family APIs
all thresholds and measured margins
```

Agent 4 may not change the 8,000 base library.
