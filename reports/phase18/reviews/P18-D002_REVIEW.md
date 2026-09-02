# Reviewing-chat audit of P18-D002

## Decision

**Accept Agent 2's `REVISE` decision. Proceed with one measurement-only G1
revision. Do not proceed to G2.**

Agent 2 produced a valid reproduction and a valid negative gate result under the
frozen rule:

```text
Phase 8 gates                         42 / 42 pass
Agent 6 completion gates              26 / 26 pass
paired non-inferiority margins         7 / 8 pass
failing margin                         vs-random EWR only
candidate-reference delta              -0.000244140625
95% paired interval                    [-0.011962890625, +0.01123046875]
frozen lower margin                    -0.010
missing / illegal / non-finite cases   0 / 0 / 0
```

The G1 control is strong evidence that Phase 8 reproduces, but G1 does not pass
because the predeclared decision reads the lower confidence bound, not the point
estimate. Lowering the margin after seeing the result would violate the Phase 18
process. Stopping the entire phase would also be premature because the only failure is
an underpowered comparison whose estimate is essentially zero.

## Independent verification

The review independently checked:

- all 11 files in `phase18_g1_binding_v1.json` against their SHA-256 values: zero
  mismatches;
- the instruction and decision audit trails;
- the exact source binding to
  `66b733ad92324751e30bd7e2a5e373129cbe87c3`;
- 40 Phase 18 evaluator tests: 40 passed;
- the paired interval, margin direction, and decision semantics;
- the protected Phase 14 manifest remains the only dirty path and was not committed;
  and
- the approved branch was published non-force to GitHub at
  `origin/phase18/setup-integrated-warmstart-g1`, commit
  `18409f738613616e364f81ff14814d4648fc92d1`.

## Sample-size correction

Agent 2 correctly diagnosed insufficient precision but overstated two points.

First, model quality can clear a fixed-width interval if the true delta is positive
enough; the problem is specifically that an approximately equal model has poor power
to certify a tight non-inferiority margin with 1,024 paired units.

Second, approximately 1,448 pairs makes the expected lower bound just cross `-0.01`
at the observed delta. It does not provide high probability of passing. With observed
paired-difference SD `0.189374` and the existing two-sided 95% rule:

```text
quantity                                      paired units
expected bound merely reaches margin          about 1,448 at the observed delta
80% power at true delta 0                      about 2,815
90% power at true delta 0                      about 3,769
frozen confirmation design                    4,096
```

The follow-up therefore uses 4,096 independent confirmation pairs. It keeps the
original `-0.01` margin, the original two-sided 95% paired-bootstrap rule, and 10,000
replicates. It does not pool the already-observed 1,024 pairs into the primary result.

## Authorized next question

> On a new independent 4,096-pair random-opponent confirmation bank frozen before
> either arm runs, does the G1 candidate's paired 95% lower bound relative to the
> accepted Phase 8 checkpoint exceed the original `-0.01` EWR margin?

This is a measurement revision, not a new training experiment. The accepted and G1
candidate checkpoint bytes remain fixed. No sealed test split is reopened. G2 and all
setup-model work remain unauthorized until this question closes.

## GitHub publication rule

Approved agent work must be published to its named GitHub branch with a normal,
non-force push. Unreviewed result commits are not pushed. The approval step records
the local and remote branch SHA and they must match before the next work package may
begin.
