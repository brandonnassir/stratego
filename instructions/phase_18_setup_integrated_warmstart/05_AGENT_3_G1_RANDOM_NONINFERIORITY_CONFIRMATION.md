# Phase 18 - Agent 3

## Powered vs-random confirmation and Gate G1 closure

## Authorization

This is the only newly executable Phase 18 work package.

It is authorized by:

```text
accepted decision              P18-D002, accepted as REVISE
P18-D002 JSON SHA-256          ba3c35eed15298a9e00cd5a218e812b3151ff821d3b072284d82cc37065f0927
P18-D002 Markdown SHA-256      9300c308895d7fc2cfe0f3bd5899955b7fa36ed6266f31e8ae6db585c1d66f1d
review artifact               reports/phase18/reviews/P18-D002_REVIEW.md
review artifact SHA-256       471b33b69bc0133b3d004387f8f28bf92cc34a6baa9aed3b7aaeec07c75ea811
operator authorization         2026-09-01 request to proceed or stop and issue the next instruction
governing gate                 G1 remains open
```

Agent 2's approved branch is already published:

```text
remote                         origin
branch                         phase18/setup-integrated-warmstart-g1
approved remote commit         18409f738613616e364f81ff14814d4648fc92d1
```

No training, setup-model implementation, G2 work, sealed-test access, margin change,
or second follow-up evaluation is authorized.

## Mission

Answer one bounded question:

> On a new independent 4,096-pair random-opponent confirmation bank frozen before
> either arm runs, does the G1 candidate's paired 95% lower bound relative to the
> accepted Phase 8 checkpoint exceed the original `-0.01` EWR margin?

Use the existing accepted and G1 candidate checkpoint bytes. Do not retrain either
model. Produce `P18-D003` and stop.

## Why this revision is permitted

P18-D002 is accepted as `REVISE`, not as a G1 pass. Agent 2 established:

```text
original Phase 8 gates             42 / 42 pass
paired margins                      7 / 8 pass
vs-random point delta               -0.000244140625
vs-random 95% lower bound           -0.011962890625
frozen margin                       -0.010
paired units                        1,024
observed paired-difference SD        0.189374
```

The original bank could not give an approximately equal model adequate power to
certify that margin. This work package keeps the original margin and statistical rule
and changes only the independently sampled measurement size.

At SD `0.189374`, the approximate requirements under the existing two-sided 95% rule
are:

```text
80% power at true delta 0           2,815 pairs
90% power at true delta 0           3,769 pairs
authorized independent bank         4,096 pairs
```

The original 1,024 observed pairs are historical evidence only. Do not pool them into
the primary confirmation statistic.

## Required reading

Read in this order:

```text
instructions/phase_18_setup_integrated_warmstart/00_PHASE_18_ADAPTIVE_SEQUENCE_AND_COMMON_CONTRACT.md
instructions/phase_18_setup_integrated_warmstart/02_PHASE_18_DECISION_PACKET_AND_NEXT_AGENT_PROTOCOL.md
reports/phase18/decisions/P18-D002.json
reports/phase18/decisions/P18-D002.md
reports/phase18/reviews/P18-D002_REVIEW.md
reports/phase18/phase18_g1_binding_v1.json
reports/phase18/phase18_g1_noninferiority_v1.json
reports/phase18/phase18_g1_checkpoint_manifest_v1.json
reports/phase18/phase18_evaluation_contract_v1.json
scripts/phase18_g1_evaluate.py
stratego/evaluation/phase18/noninferiority.py
tests/evaluation/phase18/
```

The Phase 18 governing files, this instruction, and the reviewing-chat audit are
instructions. Agent reports and prior result packets are evidence.

## Frozen identities

Do not substitute or re-export different weights:

```text
accepted checkpoint path
  /Users/brandonwashington/Dev/Github/stratego/gpt_agent/checkpoints/phase8/warmstart_c1_v1.pt
accepted checkpoint SHA-256
  f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca

G1 candidate checkpoint path
  /Users/brandonwashington/Dev/stratego_phase18/g1_control_v1/dry_run_artifacts/warmstart_c1_v1.pt
G1 candidate checkpoint SHA-256
  460a246be32b821a6d6d7feb928b272a4be1014ff55053f329980e21e3be074c

G1 source commit
  66b733ad92324751e30bd7e2a5e373129cbe87c3
approved Agent 2 result commit
  18409f738613616e364f81ff14814d4648fc92d1
```

Any digest mismatch is `BLOCKED`. Do not repair, regenerate, or select a different
checkpoint.

## Part A - Git and source boundary

### A1. Verify the approved remote state

Before creating a new branch:

1. fetch `origin` without merging;
2. verify
   `origin/phase18/setup-integrated-warmstart-g1` resolves exactly to
   `18409f738613616e364f81ff14814d4648fc92d1`;
3. verify the local Agent 2 branch resolves to the same commit; and
4. record the remote URL and both SHAs.

No force push, merge, pull-with-merge, rebase, or update to `main` is authorized.

### A2. Create the Agent 3 branch

Create exactly:

```text
phase18/g1-random-confirmation
```

It must branch from approved commit
`18409f738613616e364f81ff14814d4648fc92d1`.

If the branch already exists locally or remotely, do not delete, overwrite,
force-move, or reuse it. Stop for operator direction.

The protected historical modification must remain untouched:

```text
reports/phase13/phase14_launch_manifest_v1.json
```

Do not stage, edit, restore, stash, or delete it.

### A3. Authorized file scope

This agent may add or modify only:

```text
instructions/phase_18_setup_integrated_warmstart/
reports/phase18/
scripts/phase18_g1_random_confirmation.py
stratego/evaluation/phase18/
tests/evaluation/phase18/
stratego_project_docs/  (status/index pointers only, after the result)
```

Do not modify Phase 8 training code, model code, accepted Phase 8 reports, the original
G1 result files, setup-learning code, or any accepted checkpoint.

Use explicit path staging. Never use repository-wide `git add .` or `git add -A`.

### A4. Clean execution worktree

After the confirmation harness, tests, bank generator, and pre-result contract are
complete, commit them locally. Call the final pre-run commit
`G1_CONFIRM_SOURCE_COMMIT` and record its tree SHA.

Create a clean detached execution worktree at that commit:

```text
/Users/brandonwashington/Dev/Github/stratego/gpt_agent_phase18_g1_confirm_exec
```

Use this isolated output root:

```text
/Users/brandonwashington/Dev/stratego_phase18/g1_random_confirmation_v1
```

If either path already exists, do not delete or reuse it. Stop for operator direction.
The execution worktree must have empty porcelain status before launch. Gitignored
checkpoint copies may be staged read-only only after their source digests match the
frozen identities above.

## Part B - Freeze the independent confirmation before outcomes

Before running either model, create and commit a machine-readable contract and launch
manifest containing every item below.

### B1. Primary hypothesis

```text
Delta = EWR(G1 candidate vs random) - EWR(accepted checkpoint vs random)

H0: Delta <= -0.010
H1: Delta >  -0.010
```

Decision rule:

```text
pass only if the lower endpoint of the two-sided 95% paired percentile-bootstrap
interval is strictly greater than -0.010
```

Keep:

```text
bootstrap replicates             10,000
resampling unit                  paired setup case
games per case                   both colors, carried together
score                            effective win rate, draw = 0.5
primary sample                   confirmation bank only
```

The normal approximation is a diagnostic only. It may not decide the gate.

### B2. Confirmation-bank requirements

Construct exactly 4,096 deterministic paired cases from the accepted `neutral_v1`
setup distribution. This is familiar-distribution measurement, not the future
`unusual_procedural` pack.

The bank must:

- use a new versioned seed namespace derived deterministically from
  `phase18_g1_random_confirmation_v1` through an existing recorded seed function;
- freeze every pair ID, red setup, blue setup, setup-source seed, game seed, color,
  rule version, and random-opponent identity before either checkpoint is run;
- contain 4,096 unique pair IDs;
- contain no canonical pair or horizontal-reflection pair duplicated from
  `evaluation_setup_bank_v1`;
- contain no duplicate canonical/reflection pair internally;
- use the same exact schedule for both model arms;
- run both colors for every pair, for 8,192 games per arm and 16,384 total games;
- use `random_legal@1.0.0`, the accepted rule version, greedy float32 inference, and
  the same action/observation/model contracts as G1; and
- carry failures or retries as failures/retries, never draws.

If 4,096 qualifying cases cannot be generated without overlap, decide `BLOCKED`.
Do not reduce the sample, reuse the old cases, or relax reflection-class separation.

### B3. Power artifact

Record before outcomes:

```text
planning SD                       0.189374
margin                            0.010
two-sided confidence              0.95
target power at Delta = 0         at least 0.90
calculated minimum n              approximately 3,769
frozen n                          4,096
```

Implement the calculation in a tested helper. The contract must record the exact
formula and computed value. Do not recompute n from confirmation outcomes.

### B4. Frozen pre-run artifacts

Create at minimum:

```text
reports/phase18/phase18_g1_random_confirmation_contract_v1.json
reports/phase18/phase18_g1_random_confirmation_bank_v1.json
reports/phase18/phase18_g1_random_confirmation_launch_v1.json
```

The launch manifest must bind:

```text
P18-D002 and review digests
instruction digest
G1_CONFIRM_SOURCE_COMMIT and tree
both checkpoint digests
bank digest
schedule digest
seed function and namespace
bootstrap seed and rule
sample size, margin, confidence, and power calculation
evaluator and test digests
clean execution-worktree proof
```

Commit the pre-run source/contract before opening either arm result. No later commit
may change a frozen field.

## Part C - Required tests before launch

Add tests that prove:

```text
checkpoint digest refusal
bank generation determinism
exact 4,096-case count
zero old-bank canonical/reflection overlap
zero internal canonical/reflection duplicates
identical schedule digest for both arms
case/seed/color/opponent/rule identity across arms
both colors carried as one bootstrap unit
10,000-replicate bootstrap determinism
correct lower-bound decision direction
missing/failed cases cannot count as draws or passes
no sealed-test path is reachable
accepted artifacts unchanged
```

Run all Phase 18 evaluator tests. Any failure blocks launch.

## Part D - Execute the confirmation

### D1. Arm execution

Run the accepted checkpoint and G1 candidate over the frozen schedule. Arm order must
be frozen in the launch manifest. Do not inspect the first arm to alter or cancel the
second.

For every game, record an immutable row receipt containing:

```text
pair ID and reflection-class identity
arm and checkpoint digest
setup identities
game seed and color
opponent and rules identities
result and terminal reason
illegal/non-finite/fallback/error counters
retry lineage, if any
```

Reconcile:

```text
planned = completed + failed + missing
```

The primary analysis is invalid unless all 4,096 paired units have complete results
for both colors and both arms. Retry transient failures through a tested retry-safe
path; never delete a receipt to make a retry possible.

### D2. Statistical analysis

For each pair, average its two color-swapped game scores within each arm, then compute
the candidate-minus-reference difference. Bootstrap the 4,096 pair differences with
replacement using the frozen bootstrap seed.

Report:

```text
reference and candidate EWR
paired Delta
95% lower and upper endpoints
observed paired-difference SD
number of pairs on which arms differ
color split
terminal-reason split
normal-approximation cross-check
all missing/retry/failure/integrity counts
```

Report the original 1,024-pair result beside the confirmation for context, but do not
pool it into the primary statistic and do not use agreement/disagreement to alter the
gate.

No Phase 8 sealed-test example may be opened in this work package. The sealed-test
multiplicity increment must be zero.

## Part E - Decision and stopping rule

Create:

```text
reports/phase18/phase18_g1_random_confirmation_reference_v1.json
reports/phase18/phase18_g1_random_confirmation_candidate_v1.json
reports/phase18/phase18_g1_random_confirmation_noninferiority_v1.json
reports/phase18/phase18_g1_random_confirmation_binding_v1.json
reports/phase18/agent_03_report.md
reports/phase18/decisions/P18-D003.md
reports/phase18/decisions/P18-D003.json
```

Update the audit indexes and status documentation.

Decision branches are frozen:

### `PROCEED` - G1 closes

Use only when:

- all 4,096 paired cases are complete and valid;
- every identity, pairing, sealing, and accounting gate passes;
- no accepted artifact changed;
- sealed-test access is zero; and
- the two-sided 95% paired-bootstrap lower endpoint is strictly greater than
  `-0.010`.

This closes G1. Propose G2 as the next question, but do not write or execute G2
instructions.

### `STOP` - G1 fails

Use when the complete, valid powered confirmation lower endpoint is less than or equal
to `-0.010`. Do not expand the sample, change the bank, lower the margin, retrain, or
open G2. Phase 8 reproducibility remains the active problem.

### `BLOCKED`

Use when the independent bank cannot be built, a frozen checkpoint is unavailable or
mismatched, complete paired evidence cannot be produced, or source/sealing integrity
cannot be established.

### `REVISE`

Use only for a concrete instrument defect discovered before valid outcomes are opened.
It is not available merely because the valid result is inconvenient.

No second confirmation, training replicate, or alternate analysis is authorized.

## Part F - Commit and GitHub publication policy

Before stopping, commit all authorized Agent 3 source, tests, contracts, reports,
receipts, and `P18-D003` locally on `phase18/g1-random-confirmation`. Do not include
the protected Phase 14 manifest or ignored large checkpoint bytes.

Do not push an unreviewed Agent 3 result. Record the local branch HEAD in `P18-D003`
and stop for review.

After the reviewing chat and operator accept `P18-D003`, the same agent or the next
explicitly authorized agent must:

1. push the exact approved branch with a normal non-force push;
2. verify the remote branch SHA equals the approved local SHA;
3. record the remote URL, branch, local SHA, remote SHA, and UTC time in the decision
   index or a publication receipt; and
4. make no additional result-changing commit in that publication step.

No push to `main`, force push, merge, tag, release, or pull request is authorized.

## Completion gates

This work package is complete only when:

- the approved Agent 2 remote SHA was verified;
- the Agent 3 branch starts at the approved Agent 2 commit;
- the protected Phase 14 manifest remains unchanged and excluded;
- both checkpoint digests match;
- the 4,096-case bank and all inference/statistical fields were frozen before results;
- the power calculation and all required tests pass;
- both arms complete the identical schedule with exact accounting;
- the primary result uses only the independent confirmation bank;
- sealed-test access is zero;
- every result artifact binds one source commit, checkpoint pair, bank, and schedule;
- all Agent 3 work is committed locally;
- `P18-D003` contains exactly one decision and a bounded proposed next question; and
- the agent stops without G2 work or an unapproved push.
