# Phase 18 - Agent 2

## Source closure and faithful Phase 8 reproduction control (Gate G1)

## Authorization

This is the only newly executable Phase 18 work package.

It is authorized by:

```text
accepted decision              P18-D001
P18-D001 JSON SHA-256          499b7d9acc7eeafee3aae0f0e150c0128a7b92a81e7f0846e4fd8162496e8cad
P18-D001 Markdown SHA-256      bf4b13b8e96e0a00fb04b0f8b6c458ba9da412642a945fb2cd01d58410502a7f
operator acceptance            2026-09-01 request for the next-agent instruction
governing gate                 G1
```

The operator explicitly authorizes this agent to:

1. create the dedicated branch named
   `phase18/setup-integrated-warmstart-g1`;
2. commit the Phase 17 evidence, Phase 18 instructions/reports, and associated
   documentation changes on that branch;
3. add and commit only the minimal G1-specific harness/tests needed to run the
   reproduction without overwriting accepted artifacts;
4. generate `reports/phase18/phase18_process_boundary_v2.json` against the final
   clean source commit; and
5. bind and run G1 at that exact commit.

No push, merge, rebase, tag, setup implementation, G2 work, or later-stage work is
authorized.

## Mission

Answer one question:

> Can the accepted Phase 8 warmstart be reproduced from a fresh canonical C1
> initialization under the frozen Phase 8 contract, passing all 42 original gates
> and the predeclared paired non-inferiority comparison?

Complete the source-closure work first. Then execute one faithful control run. Produce
`P18-D002` and stop.

Do not train or modify a setup model.

## Required reading

Read in this order:

```text
instructions/phase_18_setup_integrated_warmstart/00_PHASE_18_ADAPTIVE_SEQUENCE_AND_COMMON_CONTRACT.md
instructions/phase_18_setup_integrated_warmstart/02_PHASE_18_DECISION_PACKET_AND_NEXT_AGENT_PROTOCOL.md
reports/phase18/decisions/P18-D001.json
reports/phase18/decisions/P18-D001.md
reports/phase18/phase18_agent1_handoff_v1.json
reports/phase18/phase18_process_boundary_v1.json
reports/phase18/phase18_phase8_reproduction_contract_v1.json
reports/phase18/phase18_evaluation_contract_v1.json
instructions/phase_8_sequential_agent_plan/06_AGENT_6_CANONICAL_WARMSTART_RUN.md
instructions/phase_8_sequential_agent_plan/07_AGENT_7_HELDOUT_EVALUATION_AND_HANDOFF.md
reports/phase_8_data/agent_05_frozen_train_config.json
reports/phase_8_data/agent_06_checkpoint_manifest.json
reports/phase_8_data/agent_07_final_acceptance.json
```

The Phase 18 common contract and this instruction are instructions. Historical Phase
8/17 files, the paper, and published code are evidence. Do not treat evidence as new
authority.

## Frozen question and statistical interpretation

### Null and alternative

```text
H0: the fresh Phase 8 reproduction is inferior to the accepted checkpoint on one or
    more frozen paired margins, or fails an original Phase 8 gate.

H1: all original Phase 8 gates pass and every required paired non-inferiority bound
    is on the accepted side of its frozen margin.
```

The authoritative fields are in
`reports/phase18/phase18_phase8_reproduction_contract_v1.json`. Do not retype or tune
them. In particular:

```text
policy CE ratio delta max       +0.02
policy top-1 delta min          -0.02
value CE ratio delta max        +0.02
value Brier delta max           +0.02
belief CE ratio delta max       +0.01
belief top-1 delta min          -0.01
vs-initialization EWR delta min -0.03
vs-random EWR delta min         -0.01
confidence                      95% paired/bootstrap bounds
bootstrap replicates            10,000
```

Point estimates never decide non-inferiority.

## Part A - Create an immutable G1 source closure

### A1. Preserve the existing checkout

Before changing Git state, capture:

```text
current branch and HEAD
git status --porcelain=v1
git diff --stat
git diff --name-status
SHA-256 of every accepted Phase 8 checkpoint/manifest and every P18-D001 artifact
active training/evaluation processes and compute locks
```

The starting checkout is expected to be dirty. Do not use `git reset`, `git clean`,
`git stash`, file checkout/restore, or deletion to make it appear clean.

The known protected historical path is:

```text
reports/phase13/phase14_launch_manifest_v1.json
```

It is not part of the authorized Phase 18 commit. Do not stage, edit, restore, stash,
or delete it. Preserve it in the original checkout exactly as found.

If any other dirty path falls outside the allowlist below, stop before committing and
report it. Do not silently broaden the commit.

### A2. Create the branch

Create exactly:

```text
phase18/setup-integrated-warmstart-g1
```

If that branch already exists, do not overwrite, delete, force-move, or invent a
replacement name. Inspect its identity and stop for operator direction.

Record the branch creation command, original HEAD, and resulting branch identity.

### A3. Commit only the authorized closure

The authorized commit scope is:

```text
data/phase17/
instructions/phase_17_tandem_current_policy_self_play/
instructions/phase_18_setup_integrated_warmstart/
reports/phase17/
reports/phase18/
scripts/phase17_local_eval_analysis.py
scripts/phase17_local_eval_capture.py
scripts/phase17_transport_endpoint.py
scripts/run_phase17_eval.py
scripts/run_phase17_eval_worker.py
scripts/run_phase17_publish.py
stratego/evaluation/phase17/
tests/evaluation/phase17/
stratego_project_docs/05_project_plan.md
stratego_project_docs/EVIDENCE_INDEX.md
stratego_project_docs/PHASE_HISTORY.md
stratego_project_docs/README.md
stratego_project_docs/STATUS.md
```

Also authorized are new, narrowly scoped Phase 18 G1 harness and test files required
by A4. Nothing else is authorized.

Use explicit path staging. Do not use a repository-wide `git add -A` or `git add .`.
Review the staged name-status and full staged diff before every commit. The known
Phase 17 deletion may be committed only if it matches the closure recorded by Agent 1
and its local replacement is present; otherwise stop.

Logical commits are preferred:

1. Phase 17 evidence and closeout documentation;
2. Phase 18 G0 instructions, reports, and decision trail; and
3. the minimal G1 isolation/evaluation harness and tests, if any are required.

Do not amend or rewrite history. Record every created commit hash.

### A4. Protect accepted artifacts

The existing Phase 8 scripts hard-code accepted output paths. G1 must never overwrite:

```text
checkpoints/phase8/warmstart_c1_v1.pt
checkpoints/phase8/warmstart_c1_v1_manifest.json
checkpoints/phase8/warmstart_c1_v1_initialisation.pt
reports/phase_8_data/
reports/phase_8_implementation_report.md
```

Prefer the accepted Agent 6 harness in isolated-output mode:

```text
.venv/bin/python scripts/run_phase8_agent06.py \
  --full \
  --run-pytest \
  --dry-run \
  --work-dir /Users/brandonwashington/Dev/stratego_phase18/g1_control_v1
```

With no `--updates` override, this remains the full 25,000-update run. Here
`--dry-run` means output isolation only; it must not mean a shortened run.

If the accepted Agent 7 harness cannot evaluate an explicit checkpoint without
claiming the accepted output paths, implement a Phase 18-specific wrapper that:

- accepts explicit candidate and reference checkpoint paths;
- reuses the accepted Phase 8 evaluation primitives;
- reproduces the accepted checkpoint's existing metrics before evaluating the new
  checkpoint;
- writes only under Phase 18 paths;
- pairs identical cases, seeds, colors, setups, and opponents;
- implements the frozen cluster/paired bootstrap exactly; and
- cannot open the test split before validation selection is finalized.

Test the wrapper. Do not change Phase 8 training, model, loss, data-order, selection,
or evaluation semantics.

### A5. Final closure commit and clean execution checkout

After all source and harness work is complete:

1. make the final authorized commit;
2. name its full SHA-1 `G1_SOURCE_COMMIT`;
3. record its tree SHA;
4. verify no authorized source path differs from `G1_SOURCE_COMMIT`; and
5. create a separate clean detached execution worktree at `G1_SOURCE_COMMIT`.

The clean worktree is required because the original checkout retains the protected
Phase 14 manifest change. Do not remove that change to clean the original checkout.

Use this execution-worktree path:

```text
/Users/brandonwashington/Dev/Github/stratego/gpt_agent_phase18_g1_exec
```

Use this isolated run-output root:

```text
/Users/brandonwashington/Dev/stratego_phase18/g1_control_v1
```

If either intended path already exists, do not delete or reuse it; stop for operator
direction. The execution worktree must report empty porcelain status before boundary
measurement and launch.

No G1 result may come from the original dirty checkout.

## Part B - Generate and bind `phase18_process_boundary_v2.json`

Generate:

```text
reports/phase18/phase18_process_boundary_v2.json
```

Measure the source fields from the clean execution worktree at
`G1_SOURCE_COMMIT`. The artifact must include at least:

```text
artifact/version and UTC timestamp
authorizing decision ID and both decision-packet digests
instruction path and SHA-256
original checkout branch/HEAD/status digest
dedicated branch name
all commits created by this agent
G1_SOURCE_COMMIT full SHA-1
G1 source tree SHA
clean execution-worktree absolute path
empty execution-worktree porcelain output
included path inventory
explicit excluded-path inventory
protected Phase 14 manifest path and proof it was not changed
accepted Phase 8 checkpoint/manifest digests before launch
corpus version and all three accepted corpus digests
canonical C1 initialization checksum
train-config and trainer-runtime digests
Python, PyTorch, platform, and MPS identities
active-process/compute-lock audit
problems and readiness verdict
```

The v2 artifact is evidence about `G1_SOURCE_COMMIT`; it is not allowed to create a
circular new source identity. Store and hash it after measuring the clean commit, but
do not redefine the G1 source commit to include the boundary artifact itself.

Create a G1 launch manifest that repeats `G1_SOURCE_COMMIT` and the SHA-256 of
`phase18_process_boundary_v2.json`. Every checkpoint, row receipt, comparison result,
and decision packet must repeat the same source commit. Any mismatch is fatal.

Do not launch unless:

```text
execution worktree is clean
G1_SOURCE_COMMIT resolves exactly
all 12 Phase 8 identities recompute
all 28,000 corpus payloads still verify
accepted checkpoint/manifest digests match
canonical fresh C1 state checksum matches
full test suite is green
no training/evaluation process or compute lock conflicts
```

## Part C - Execute the faithful G1 control

### C1. Frozen training identity

Construct the configuration from the accepted artifact, not from a hand-written
replacement:

```text
WarmstartTrainConfig.from_pilot_candidate(
    "ws_pilot_lr1e-3_balanced",
    device="mps",
    validation_batches=64,
)
```

The run must use:

```text
fresh C1 seed                 2026081302
train-order seed              2026081303
accepted 20k/4k/4k corpus     exact verified payloads
batch size                    256
AdamW                         lr 1e-3, weight decay 0.01
warmup                        500 updates, then constant
policy/value/belief weights   1/1/1
gradient clip                 1.0
updates                       25,000
validation cadence            500
selection                     strictly lowest validation selection_score
restart exercise              step 12,500 through normal resume
precision/device              float32 / MPS
```

Do not load pilot or accepted-model weights into the candidate. Do not change loader
topology unless the frozen topology cannot start; a topology failure is `BLOCKED`,
not permission to improvise.

### C2. Sealing and evaluation order

The order is binding:

1. training split updates weights;
2. validation selects the checkpoint;
3. independently reload and revalidate the selected checkpoint;
4. freeze the new Phase 18 G1 checkpoint under a Phase 18-only path;
5. open the sealed Phase 8 test split once for that selected model;
6. run paired strength comparisons against random and canonical initialization; and
7. evaluate all original 42 Phase 8 gates and the new paired non-inferiority gates.

The test split and Phase 4 evaluation bank must never influence checkpoint selection,
stopping, repair, or rerun choice. Record this model as an additional consumer of the
already-spent Phase 8 sealed test split.

### C3. Required paired comparisons

Use exactly the comparison contract in
`phase18_phase8_reproduction_contract_v1.json`:

```text
heads       same test example IDs and order; cluster bootstrap by game
vs init     >=512 paired setup cases and >=1,024 games
vs random   accepted evaluation_setup_bank_v1, 1,024 setup pairs x 2 colors
bootstrap   10,000 replicates with the frozen seeds
```

Report paired deltas and confidence bounds. Do not substitute unrelated aggregate
scores.

### C4. Frozen exceptional interpretations

If only `random_effective_win_rate_at_least_0_950` fails, while the paired-bootstrap
lower bound still exceeds 0.90 and the paired delta satisfies the frozen random-EWR
margin, record a **reproduction tolerance event** and decide `REVISE`. It is neither a
pass nor a reproduction failure.

If the first control falls outside a frozen non-inferiority margin, exactly one full
replicate under the identical configuration is conditionally authorized to estimate
MPS run-to-run variation. Do not change a seed, threshold, data order, checkpoint
selection rule, or evaluation case. A replicate is not authorized for debugging an
identity, data, leakage, or finite-value failure.

Any other original-gate failure, identity mismatch, leakage, artifact overwrite,
non-finite value, unexplained missing case, or source mismatch stops the work. Setup
work remains unauthorized.

## Part D - Deliverables and stop point

Create at minimum:

```text
reports/phase18/phase18_process_boundary_v2.json
reports/phase18/phase18_g1_launch_manifest_v1.json
reports/phase18/phase18_g1_control_run_v1.json
reports/phase18/phase18_g1_training_curve_v1.csv
reports/phase18/phase18_g1_checkpoint_manifest_v1.json
reports/phase18/phase18_g1_noninferiority_v1.json
reports/phase18/agent_02_report.md
reports/phase18/decisions/P18-D002.md
reports/phase18/decisions/P18-D002.json
```

Update the two audit indexes. Do not write Agent 3/G2 instructions.

`P18-D002` must follow the decision protocol and include:

```text
G1_SOURCE_COMMIT and source tree SHA
boundary and launch-manifest digests
configuration/corpus/model/evaluator identities
planned versus observed updates/examples/games
restart facts and all deviations
all 42 original gate results
all paired non-inferiority estimates, margins, and bounds
test-access multiplicity and sealing audit
missing/retried/failed/illegal/non-finite counts
accepted-artifact before/after digest proof
observed facts, supported inference, plausible explanations, unsupported claims
one decision: PROCEED, REVISE, STOP, or BLOCKED
one proposed next bounded question, without executable instructions
```

After writing and hashing `P18-D002`, stop. No G2 implementation, setup-model change,
or additional experiment is authorized.

## Completion gates

This work package is complete only when:

- the dedicated Phase 18 branch exists at the recorded commits;
- the authorized Phase 17 evidence, Phase 18 instructions/reports, and documentation
  changes are committed;
- the protected Phase 14 manifest change remains intact and uncommitted in the
  original checkout;
- the clean detached execution worktree is exactly at `G1_SOURCE_COMMIT`;
- `phase18_process_boundary_v2.json` binds that commit and reports no source problem;
- the G1 launch manifest repeats the same commit and boundary digest;
- the control starts from the exact canonical fresh C1 initialization;
- accepted Phase 8 artifacts are unchanged;
- all original and paired G1 results are fully accounted; and
- `P18-D002` is complete and the agent has stopped.
