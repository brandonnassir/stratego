# Phase 7 Agent 5 — Production Pipeline Integration

## Role

You are **Agent 5** in sequential Phase 7.

Your task is to connect the accepted Phase 7 setup sampler to the frozen Phase 6 collection/persistence pipeline.

Do not perform meaningful model training.

Do not change the base library, family definitions, or sampler semantics.

Do not begin final Phase 7 acceptance.

---

## Prerequisite

Agents 1–4 must all report `PASS`.

Read:

```text
00_PHASE_7_SEQUENCE_AND_COMMON_CONTRACT.md

reports/phase_7_implementation_report.md
reports/phase_7_data/
data/setups/setup_library_v1.jsonl
data/setups/setup_library_v1_manifest.json

reports/phase_6_implementation_report.md
reports/phase_6_data/

stratego/setups/
stratego/training/
stratego/evaluation/
stratego/model/
```

Verify:

```text
reference engine  phase2_1_reference_1.2.0
primary model     C1
model contract    model_contract_v2
trajectory        trajectory_v1
backend           KEEP_PYTHON
```

Run the full suite before edits.

---

## Frozen Phase 6 collection path

Preserve the accepted production shape:

```text
one MPS coordinator
CPU simulation workers
bulk synchronous collection
dense legality
perspective-normalized model actions
absolute engine actions
trajectory_v1
compressed shard persistence available
streaming verification
process recycling available
```

Do not redesign this architecture for Phase 7.

---

## Mission

Make the setup source pluggable so production collection can use:

```text
setup_library_v1 + setup_sampler_v1
```

rather than relying only on uniform random setups.

The setup sampler should be an input at game creation/reset.

It must not change game behavior after setup creation.

---

## Setup-pair sampling

For training mode, sample Red and Blue setups independently from the requested split/profile using deterministic per-game/per-side seeds.

Conceptually:

```text
game identity
    -> red setup sampling seed
    -> blue setup sampling seed
```

Changing worker count, environment slot, scheduling order, or recycle boundary must not change the setup pair assigned to a logical game identity when the same run manifest/seed semantics are replayed.

Do not consume one global setup RNG in worker-arrival order.

---

## Split behavior

The default future training mode must use:

```text
split = train
```

Validation/test splits must be accessible only through explicit evaluation/audit requests.

Do not silently mix validation/test bases into routine training collection.

Add a regression that the default production training path cannot sample validation/test entries.

---

## Provenance

Attach setup provenance to each completed game through a sidecar/manifest mechanism without changing `trajectory_v1`.

For each player preserve at least:

```text
setup_library_version
sampler_version
primary_family_id
base_setup_id
split
reflection_applied
perturbation_applied
perturbation_id_or_seed
final_setup_fingerprint
```

The actual setup stored in the trajectory remains the replay authority.

Provenance is diagnostic/training metadata.

---

## Observer-safety

Prove the model/inference path does not receive opponent setup provenance or true opponent setup.

Forbidden model inputs include:

```text
opponent family id
opponent base setup id
opponent perturbation seed
opponent reflected bit
true opponent setup
```

Do not add Phase 7 channels to `observation_v2_1_127ch`.

Do not add setup provenance to Agent 5's observer-safe neural inference request.

Add an object-graph/interface regression as appropriate.

---

## Phase 4 isolation

The accepted:

```text
evaluation_setup_bank_v1
```

must remain unchanged.

Do not replace its setup source with `setup_library_v1`.

Verify before/after:

```text
bank version
bank count
bank digest / reproducible content identity
```

Existing Phase 4 evaluation tests must pass unmodified.

---

## Integration correctness campaign

Run a deterministic campaign through the real collection/persistence path.

Target at least:

```text
4,096 completed games
```

with coverage of all:

```text
16 x 16 = 256 ordered Red-family / Blue-family combinations
```

Aim for at least 16 completed games per ordered family pair when using 4,096 games.

Exercise:

```text
both reflection orientations
perturbed and unperturbed outputs
zero-decision game support if encountered
ordinary completed games
compressed persistence
streaming verification
```

Use the train split for the headline training-path campaign.

Run separate explicit smoke requests for validation and test split access without mixing them into the training campaign.

---

## Model usage

Use the accepted C1 integration path or another existing deterministic Phase 6 fixture as required to exercise the real collection path.

No playing-strength interpretation is allowed.

No optimizer or learning update.

The model weights may remain fixed/random.

---

## Correctness requirements

Across the integration campaign require:

```text
setup engine-validation failures        0
setup provenance mismatches             0
wrong split samples                     0
family/base identity mismatches         0

illegal actions                         0
active-zero-legal anomalies             0
action-frame mismatches                 0
model/MPS failures                      0
non-finite output failures              0

trajectory decode failures              0
trajectory reconstruction mismatches    0
persisted-record corruption             0
duplicate game ids                      0
```

Verify that the setup serialized inside each sampled trajectory matches the setup fingerprint in provenance.

---

## Reconstruction sample

Reconstruct a meaningful sample from persisted trajectories.

Require at least:

```text
10,000 decisions
```

or all decisions if the campaign is smaller than that.

For every sampled game also verify:

```text
trajectory setup == provenance final fingerprint
family/base/split metadata corresponds to sampler rebuild
```

Zero mismatches.

Zero-decision games, if present, must be included in setup/provenance verification even though they contribute no decisions.

---

## Determinism

Repeat a deterministic subset under changed:

```text
worker count
schedule order
recycle boundary if practical
```

and require the setup assignments for the same logical game identities to remain identical.

Do not require model action histories to be identical across float16 batch-shape changes unless using the existing deterministic evaluation path; this gate is about **setup assignment identity**.

Record the exact deterministic comparison design.

---

## Performance impact

Measure enough to ensure setup generation is operationally cheap.

Report:

```text
setup sample calls/s
mean/p95 setup generation latency
setup-sampling fraction of wall time
positions/s
games/s
compressed GiB/hour for the integration campaign
process RSS
swap
```

Do not optimize aggressively unless setup sampling is unexpectedly material.

If sampler overhead is insignificant, state that with measurements.

---

## Suggested files

Possible integration ownership:

```text
stratego/training/setup_source.py
```

or a small extension to an existing reset/setup factory.

Tests:

```text
tests/training/test_phase7_setup_integration.py
tests/information_security/test_setup_provenance_boundary.py
```

Harness:

```text
scripts/run_phase7_agent05.py
```

Prefer a narrow injectable setup-source interface over family-specific logic inside workers.

---

## Artifacts

Create:

```text
reports/phase_7_data/agent_05_pipeline_integration.json
reports/phase_7_data/agent_05_setup_provenance.csv
```

Append only:

```markdown
## 5. Agent 5 — Production Pipeline Integration
```

to:

```text
reports/phase_7_implementation_report.md
```

Report:

```text
setup-source API
run seeds
game count
family-pair coverage
split coverage
provenance schema
observer-safety result
trajectory/reconstruction counts
Phase 4 bank unchanged proof
performance
tests
files
deviations
Agent 6 handoff
```

---

## Stop conditions

Report `BLOCKED` if:

- Agents 1–4 are not PASS;
- the library/sampler digest differs from handoff;
- integration requires modifying frozen engine semantics;
- setup provenance must enter model observations to make the pipeline work;
- `trajectory_v1` must be semantically changed rather than using sidecar metadata;
- deterministic setup assignment cannot be made schedule/worker independent;
- Phase 4 evaluation bank would need modification.

---

## PASS gates

PASS only if:

- Agents 1–4 PASS verified;
- setup sampler integrated into real collection reset path;
- train split is default and validation/test require explicit request;
- >=4,096 completed integration games;
- all 256 ordered family pairs represented;
- both reflection branches exercised;
- perturbed/unperturbed paths exercised;
- 0 setup/provenance/split mismatches;
- 0 illegal/frame/model failures;
- trajectory persistence/reconstruction clean;
- >=10,000 decisions reconstructed with 0 mismatches, where available;
- setup fingerprints match persisted setups;
- observer-safe model boundary unchanged;
- Phase 4 setup bank unchanged;
- setup assignment deterministic under scheduling variation;
- performance impact measured;
- no meaningful training;
- full repository suite green.

---

## Handoff to Agent 6

Provide:

```text
exact library/sampler versions and digests
integration API
default train split behavior
provenance sidecar schema
4,096-game campaign artifact
family-pair coverage
observer-safety proof
reconstruction counts
Phase 4 unchanged proof
measured sampler overhead
```
