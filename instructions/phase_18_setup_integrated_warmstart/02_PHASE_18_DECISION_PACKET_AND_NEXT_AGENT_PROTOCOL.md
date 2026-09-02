# Phase 18 - Decision Packet and Next-Agent Protocol

## Purpose

Phase 18 adapts after evidence without allowing result-driven contract drift. This
file defines the required stop point after every work package and the only process by
which a later agent becomes authorized.

Read `00_PHASE_18_ADAPTIVE_SEQUENCE_AND_COMMON_CONTRACT.md` first.

## 1. One packet per completed question

Every Phase 18 work package ends with:

```text
reports/phase18/decisions/<decision_id>.md
reports/phase18/decisions/<decision_id>.json
```

Use a monotonic decision ID:

```text
P18-D001
P18-D002
...
```

The Markdown and JSON must carry identical headline numbers and conclusions. The JSON
is authoritative for identities and calculations; the Markdown is authoritative for
the human-readable interpretation only when they do not conflict.

## 2. Required packet fields

### Identity

```text
decision ID
UTC timestamp
work package and agent
source closure digest
configuration digest
data/corpus/pack digests
model/checkpoint digests
evaluator version/digest
commands or exact entry points
```

### Question and hypothesis

State one bounded question. Include:

```text
null hypothesis
alternative hypothesis
primary metric
practical margin
confidence/precision rule
sample size and power basis
predeclared failure interpretations
```

Do not rewrite the hypothesis after the result.

### Execution integrity

Report:

```text
planned and observed samples/games/updates
missing/retried/failed cases
illegal/non-finite/orientation/fallback counts
sealing and data-access audit
resume or interruption facts
all deviations
```

### Results

Include:

- primary estimate and uncertainty;
- every required stratum and worst stratum;
- paired deltas rather than unrelated aggregate scores where the contract pairs cases;
- learning curves/trends when multiple checkpoints exist;
- negative and contradictory evidence;
- setup diversity and concentration diagnostics when setup weights changed; and
- original Phase 8 head metrics whenever C1 weights changed.

Do not use a best checkpoint's peak as evidence of a trend unless peak selection was
the predeclared rule and is evaluated on independent data.

### Interpretation

Separate:

```text
observed fact
supported inference
plausible but untested explanation
claim not supported
```

A useful negative result is not `BLOCKED`. `BLOCKED` means required evidence could not
be produced because of a concrete unresolved dependency.

### Decision

Exactly one:

```text
PROCEED
REVISE
STOP
BLOCKED
```

Include the gate name and a table of every sub-gate with `pass`, `fail`, or `not_run`.

### Proposed next question

Describe one bounded next question and why it discriminates between the remaining
explanations. Do not include executable implementation instructions unless the
operator has already approved the next work package.

## 3. Authorization workflow

1. Agent finishes the evidence packet and stops.
2. Reviewing chat independently checks identities, calculations, gate semantics, and
   whether the result answers the frozen question.
3. Operator accepts, rejects, or amends the interpretation.
4. Only then is a new numbered agent instruction written.
5. The new instruction cites the accepted decision ID and digest.

An agent may not infer authorization from a provisional stage label, a passing local
test, an available compute window, or a prior plan.

## 4. Provisional stage map

This is a map of likely questions, not executable instructions.

| Likely stage | Bounded question | Governing gate |
|---|---|---|
| Phase 8 control | Can the accepted warmstart be reproduced from fresh initialization? | G1 |
| Setup parity build | Does the local setup implementation match the published method and learn a known reward landscape? | G2 |
| Setup-only Stratego assay | Does learned setup selection beat both fresh initialization and the fixed library? | G3 |
| Tandem pilot | Does the live setup stream improve generalization without damaging Phase 8 learning? | G4 |
| Production rehearsal | Can the exact frozen topology start, resume, account, and seal correctly? | G5 |
| Full run and independent acceptance | Does the paired fresh model satisfy the complete Phase 18 goal? | G6 |

Stages may be split or replaced when the preceding evidence justifies it. The mission
and final acceptance claims do not change.

## 5. Default decision branches

### Phase 8 control fails

Next work remains Phase 8 reproducibility. Setup code is not authorized.

### Official parity or synthetic learning fails

Authorize one correction against a minimal failing fixture. Do not launch real-game
setup training.

### Synthetic learning passes; setup-only Stratego fails

Use attribution evidence to choose one bounded test among:

```text
outcome variance / outcomes per setup
pool lifetime and setup reuse
behavior-policy age
teacher/opponent distribution
entropy-to-outcome advantage balance
symmetry or canonicalization
```

Do not open a broad hyperparameter sweep.

### Setup beats initialization but not fixed library

Phase 18 has not established favorable setup selection. Do not proceed to a full
tandem run. A curriculum-only benefit is a different hypothesis and cannot silently
replace the mission.

### Setup-only passes; tandem move model regresses

The first candidates for bounded revision are the canonical/live stream mixture and
live-example cadence. The paper-faithful setup loss is not changed without evidence
that it is the source of the regression.

### Combined system improves while T-F regresses

Reject the combined result. A stronger own setup may not mask weaker unfamiliar-
opponent play.

### Familiar improves; unusual regresses

Reject as distribution specialization. Do not tune on the sealed unusual pack.

### All current gates pass

Authorize the next bounded stage. A full production instruction is written only after
G5.

## 6. Required audit trail

Maintain:

```text
reports/phase18/decision_index.json
reports/phase18/agent_instruction_index.json
```

The decision index binds each packet, its status, and the instruction it authorized.
The instruction index binds each numbered instruction to the accepted decision that
created it. An instruction with no authorizing decision is draft-only.

The original Agent 1 instruction is authorized by the operator request that created
this Phase 18 package and is the sole exception.

## 7. Commit and GitHub publication rule

The operator added this governing rule on 2026-09-01:

1. Every agent must commit its authorized source, tests, evidence, reports, and
   decision packet locally on its named phase branch before review.
2. Result commits remain local while the decision packet is unreviewed.
3. After the reviewing chat and operator accept the packet, the exact approved branch
   HEAD must be published to the configured GitHub `origin` with a normal non-force
   push.
4. The publication step must verify and record that the remote branch SHA equals the
   approved local SHA.
5. The publication step may not add a result-changing commit. A small publication
   receipt may be committed only when the authorizing instruction explicitly permits
   it and the resulting SHA is itself reviewed.
6. No agent may force push, push directly to `main`, merge, rebase, tag, create a
   release, or open a pull request without separate operator authorization.

When an agent has already stopped, the reviewing chat or the next explicitly
authorized agent may perform the non-force publication on its behalf. Publication is
not authorization for the next scientific stage.
