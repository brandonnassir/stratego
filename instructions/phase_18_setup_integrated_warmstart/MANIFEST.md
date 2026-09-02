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
| `05_AGENT_3_G1_RANDOM_NONINFERIORITY_CONFIRMATION.md` | Powered independent vs-random confirmation | Executed 2026-09-02 — delivered `P18-D003` (`PROCEED`), **accepted 2026-09-02; G1 closed**; branch published at `ef7523c1` |
| `06_AGENT_4_G2_SETUP_PARITY_AND_SYNTHETIC_ASSAY.md` | Publish accepted G1; Gate G2 setup implementation parity and synthetic learning assay | **Executed 2026-09-02** — G1 published; delivered `P18-D004` (`REVISE`, **accepted 2026-09-02**; branch published at `6afa13be`; packet wording corrected by Agent 5, see `reports/phase18/reviews/P18-D004_REVIEW.md`) |
| `07_AGENT_5_G2_RAW_ACTOR_CONFIRMATION.md` | Publish accepted G2; correct the G2 packet wording; one bounded confirmation in which the raw generation actor is the primary endpoint of the synthetic trainability assay on a fresh landscape and fresh seeds | **Executed 2026-09-02** — authorized by accepted `P18-D004` and the operator request of 2026-09-02; delivered `P18-D005` (`PROCEED` for the synthetic trainability portion of G2), **accepted 2026-09-02; Gate G2 closed**; branch published at `d0fd36d9` (see `reports/phase18/reviews/P18-D005_REVIEW.md`) |

## Current execution rule

Agents 1, 2 and 3 are complete and accepted. Agent 2's work is published at
`origin/phase18/setup-integrated-warmstart-g1` commit `18409f7`. Agent 3's
confirmation certified the vs-random margin (delta +0.006348, 95%
[+0.000793, +0.011902] on 4,096 independent pairs against the frozen -0.010
margin); `P18-D003` was **accepted as `PROCEED` on 2026-09-02, Gate G1 is closed**,
and the approved branch `phase18/g1-random-confirmation` is published at
`origin` commit `ef7523c1940650c0906d1927e64679e8328a663f` (local == remote).
The review and its erratum (the packet narrative's battleless move limit reads 100
where the contract, engine constant, schedule and receipts carry the evaluation
value 200) are recorded in `reports/phase18/reviews/P18-D003_REVIEW.md` and
`reports/phase18/phase18_rule_identity_errata_v1.json`.

`06_AGENT_4_G2_SETUP_PARITY_AND_SYNTHETIC_ASSAY.md` was executed on 2026-09-02 and
delivered `P18-D004` (`REVISE`) at `G2_SOURCE_COMMIT 354a4cad`. **`P18-D004` was
accepted as `REVISE` on 2026-09-02** and the reviewed branch `phase18/g2-setup-parity`
is published at `origin` commit `6afa13bed355884a3327d2661fd739784260dc2b`
(local == remote, non-force, no publication commit). Parity holds on all 30
method-map rows and in the independent oracle with zero integrity events, and the
setup learner learns the frozen synthetic landscape on all three seeds (raw actor
gap closure 20.9%, 18.5%, 14.8%); the gate's frozen evaluation model, the EMA at
0.999 updated once per update, retained 0.999^64 = 0.937975 of the initial
parameter contribution after the 64-update budget (an approximately 1,000-update
time constant) and lagged severely behind the raw actor, closing a median 0.35%
against the 10% threshold. That instrument concern was predeclared before the run;
the packet's as-reviewed claim that the criterion was unreachable by arithmetic is
withdrawn and the wording corrected on `phase18/g2-raw-confirmation` (see
`reports/phase18/reviews/P18-D004_REVIEW.md`).

`07_AGENT_5_G2_RAW_ACTOR_CONFIRMATION.md` was executed on 2026-09-02 and delivered
`P18-D005` (`PROCEED` for the synthetic trainability portion of G2) at
`G2_RAW_SOURCE_COMMIT ccddceda2701` on `phase18/g2-raw-confirmation`. It was
**accepted by the operator on 2026-09-02** and the branch is **published** at
`d0fd36d9469a048dbbc9aa1a233f6006d78b0e8d` (local == remote); the review record is
`reports/phase18/reviews/P18-D005_REVIEW.md`. On an independently
generated landscape (optimum 53.031081) with fresh seeds and the G2 learning method
unchanged byte for byte, the raw generation actor improved in every seed (pooled
paired 95% [+5.2985, +5.5585] on 12,288 pairs) and the median seed closed 10.0073% of
its gap against the frozen 10% threshold — met, but by only 0.0073 percentage points
and so not robust evidence for a precise 10% practical effect, while the learning
effect itself is strong and replicated in all three seeds; parity, bitwise replay,
binding and integrity all pass; the EMA telemetry (median 0.1191%) decided nothing.
The EMA remains the evaluation/deployment model for every Stratego-facing stage.

**Gate G2 is CLOSED** — implementation parity under accepted `P18-D004` and
synthetic trainability under accepted `P18-D005`. **Gate G3 is UNSTARTED.** The
acceptance authorizes designing the next gate only; no setup-only Stratego assay
(G3), tandem pilot (G4), rehearsal (G5) or production run (G6) is authorized, no
Stratego setup-learning game or Phase 8 warmstart training may be run, and the G3
design instruction must resolve `O-P18-EVALRULES-1` and size its margin from a known
instrument resolution before any run is proposed.

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
