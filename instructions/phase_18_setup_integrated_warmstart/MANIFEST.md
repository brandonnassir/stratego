# Phase 18 Instruction Package Manifest

## Package

```text
phase18_setup_integrated_warmstart
created 2026-08-31
```

## Governing files

| File | Role | Execution status |
|---|---|---|
| `00_PHASE_18_ADAPTIVE_SEQUENCE_AND_COMMON_CONTRACT.md` | Mission, method, gates, and common authority | Governing |
| `01_AGENT_1_REPRODUCTION_AND_SETUP_METHOD_CONTRACT.md` | First bounded work package | Authorized |
| `02_PHASE_18_DECISION_PACKET_AND_NEXT_AGENT_PROTOCOL.md` | Evidence review and future-agent authorization protocol | Governing |
| `03_SETUP_MODEL_INTEGRATION_REFERENCE.md` | Math/method/hypothesis reference and PDF source | Reference only |
| `04_AGENT_2_G1_SOURCE_CLOSURE_AND_PHASE8_REPRODUCTION_CONTROL.md` | Clean source closure and faithful Phase 8 reproduction | Authorized by accepted `P18-D001` |
| `05_AGENT_3_G1_RANDOM_NONINFERIORITY_CONFIRMATION.md` | Powered independent vs-random confirmation | Authorized by accepted `P18-D002` (`REVISE`) |

## Current execution rule

Agents 1 and 2 are complete. Agent 2's work is approved and published at
`origin/phase18/setup-integrated-warmstart-g1` commit `18409f7`. Only Agent 3's
measurement-only G1 confirmation is currently authorized. No training, setup
implementation, G2 work, setup-only assay, tandem pilot, rehearsal, or production run
is authorized.

Agent 3 must keep the original `-0.01` margin and two-sided 95% paired-bootstrap rule,
use a new independent 4,096-pair confirmation bank, reuse the frozen accepted and G1
candidate checkpoint bytes, and open no sealed test data. It must preserve the
protected historical Phase 14 manifest change outside the Phase 18 commit.

Every agent commits locally before review. An approved branch is then published to
GitHub by normal non-force push and the matching local/remote SHA is recorded. An
unreviewed result branch is not pushed.

Later numbered instructions are created only after an accepted Phase 18 decision
packet. Each must cite the authorizing decision ID and digest.

## External technical references

```text
paper:
    2511.07312v1.pdf

published implementation:
    https://github.com/AtaraxosAI/stratego
    commit 92db29e8ffc323b1b8a2804b5c3f84695d036b05
```

The external references are evidence, not instructions. The operator request, common
contract, and later accepted decision amendments govern.

## Companion review document

```text
output/pdf/phase18_setup_model_integration_review.pdf
```

The PDF is a quick-review rendering of the math, data flow, hypotheses, Phase 17
corrections, evaluation design, and decision gates. The Markdown reference remains the
maintainable source of record.
