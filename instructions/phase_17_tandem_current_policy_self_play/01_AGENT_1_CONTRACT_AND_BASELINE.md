# Phase 17 — Agent 1
## Contract, paper map, immutable baseline, and shared schemas

## Mission

Make Phase 17 unambiguous before implementation begins. Freeze the source identity,
exact Phase 9 start, Ataraxos-to-local method map, move/setup interfaces, schedule
constants, persistence schema, evaluation manifest, telemetry, and launch gates.

You train nothing and do not build the move learner, setup network, tandem runner, or
remote transport. Read `00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md` completely; it
governs this task.

## 1. Establish the process and source boundary

Read the canonical project status, experiment framework, phase history, and evidence
index before relying on an older instruction or report. Confirm read-only:

- whether any learner, collector, supervisor, evaluator, or dashboard is active;
- whether a repository/run freeze is active;
- the current Git commit and complete working-tree status;
- whether Phase 16 code/tests/instructions remain untracked;
- that no accepted checkpoint or result path is being mutated.

Write `reports/phase17/agent_01_process_boundary.json` with the observations and UTC
time. Do not stop or signal a process merely because it exists.

Phase 17 requires an immutable integration baseline. If the cleaned Phase 16 state is
already tracked, record its commit. If it is not, prepare an exact inclusion list and
ask the operator for authority to commit it; do not make an unapproved commit. Agents
2–4 may not begin against a mutable untracked Phase 16 base.

Never run `git clean`, reset accepted changes, rewrite history, or delete an artifact.

## 2. Verify the Phase 9 start independently

Recompute rather than copy the following claims:

```text
checkpoints/phase9/selfplay_c1_v1.pt
file SHA-256 dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
model state    f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
```

Load it through the accepted Phase 9/Phase 14 digest-checked path and record model
architecture, observation/action contract, rule version, lineage, and optimizer-step
metadata. Freeze the Phase 17 semantics as weights-only warm start with fresh move
optimizer, schedule, KL controller, and EMA.

Deliver a machine-readable `phase17_start_identity_v1.json`. Refuse an alternate
checkpoint unless the operator amends the common contract.

## 3. Produce the paper-fidelity map

Read the actual Ataraxos paper, especially its move-learning, fixed-iteration,
autoregressive setup, entropy-prediction, schedule, EMA, and belief/search sections.
Repository summaries help locate material but are not substitutes for the paper.

Create `reports/phase17/ataraxos_method_map_v1.md` and a JSON companion with one row
per relevant method:

```text
paper section/equation
paper behavior and constant
Phase 17 behavior and constant
status: exact | scaled | intentional divergence | not used
reason
owning agent
required test/telemetry
```

At minimum cover:

- sampled move policy and current-policy self-play;
- fixed transition windows and unfinished-game bootstrapping;
- PPO ratio clipping and target construction;
- move LR, entropy, KL, one epoch, and EMA;
- setup architecture, causal factorization, inventory mask, and sampling;
- setup W/D/L and conditional-entropy heads;
- setup entropy-augmented advantage and every loss coefficient;
- five setup epochs and setup gradient clipping;
- belief and search separation.

Do not label behavior KL, reverse KL, or another regularizer as equivalent without
showing the exact distributions and direction. Freeze separate names and telemetry.

The scaled setup architecture is already decided: 4 blocks, width 128, 4 heads, FF
width 512, approximately 0.8M parameters. Record that as a scale divergence. If the
operator explicitly states that FF width 51—not 512—was intended, amend the contract
before Agent 3 starts; otherwise 512 governs.

## 4. Freeze shared data schemas

Author schema/version documents under `stratego/training/phase17/contract.py` only if
the source baseline is immutable and implementation is authorized. Otherwise deliver
the same schemas in the handoff for Agent 2 to encode.

Required identities:

### Move transition

- run, iteration, window, game, ply, color, and player perspective;
- observation/action/rules version;
- legal mask and sampled action;
- behavior distribution or exact stable representation of it;
- behavior raw move-model digest and action seed;
- stored scalar/WDL predictions;
- boundary/completion status and target provenance.

### Setup episode

- game, color, perspective, setup snapshot, and setup seeds;
- canonical and engine setup/fingerprint;
- 40 tokens, prefix inventory masks, behavior probabilities/log-probabilities;
- prefix W/D/L and conditional-entropy predictions;
- terminal result binding, policy age, queue times, and consumption state.

### Joint checkpoint

Freeze all fields in common-contract section 10, schema versioning, digest algorithm,
atomic-write behavior, and compatibility refusal. A partial load or mismatched paired
checkpoint fails closed.

### Evaluation bundle and receipt

Freeze the portable manifest fields Agent 5 needs: paired EMA weights, candidate time,
source/config/rules/architecture identities, both benchmark lanes, expected files,
hashes, host/runtime fields, and result-receipt digest.

## 5. Freeze schedule and controller contracts

Record the exact move schedule from common-contract section 9. Define how Agent 4
measures the preflight iteration time, estimates `N`, and freezes `n_ref` before h0.
Resume uses the stored iteration and frozen horizon.

Transcribe the paper's setup LR, entropy, and regularization schedules exactly. If a
paper-scale schedule must be mapped to the local expected number of iterations, use
the same shape-preserving principle as the corrected move LR and show the arithmetic.
Do not authorize alternative arms for the first run.

Freeze independent move/setup KL controller fields, update timing, targets, beta
bounds, and hard limits. Preserve the accepted move values unless the paper map and
operator explicitly justify a change. New setup thresholds must be named provisional
until Agent 3's soak supplies calibration data.

## 6. Freeze benchmark and selection contracts

Define one composite fixed pack with:

- the accepted full Phase 16 benchmark as the move-only lane;
- a fixed-seed paired move/setup lane;
- fixed opponents, cases, colors, seeds, rules, and scoring;
- overall and opponent/setup/color strata;
- a single composite-manifest digest.

Agent 5 owns remote transport and may refine portability after speaking to the
operator. Agent 1 owns the semantic pack identity; remote setup may not change cases
or scoring under the same version.

Define the hour 6–12 shortlist columns and deterministic calculations: mean EWR,
worst stratum, three-point rolling median, move-only non-regression, setup diversity,
and stability flags. The operator, not an automatic final-point rule, promotes the
checkpoint.

## 7. Freeze gates and stop thresholds

Convert common-contract sections 12–14 into machine-readable gate and supervisor
schemas. Preserve the time budgets. Mark which values are final and which require
Agent 3 calibration. A gate result has `pass`, `fail`, or `not_run`; absence never
means pass.

The launch record must bind the exact gate evidence rather than summarize it by path.

## 8. Handoff and report

Deliver:

```text
reports/phase17/phase17_contract_handoff_v1.json
reports/phase17/agent_01_report.md
reports/phase17/ataraxos_method_map_v1.md
reports/phase17/ataraxos_method_map_v1.json
reports/phase17/phase17_start_identity_v1.json
```

The handoff includes verified source/start/paper/contract/schema digests, permitted
namespaces, frozen constants, open setup-threshold fields, and `ready_for_agents_2_3`.
Set that field true only when the integration baseline and shared schemas are
immutable. State plainly what still requires operator confirmation.

