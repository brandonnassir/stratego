# Review record of P18-D005

## Decision

**`P18-D005` is accepted as `PROCEED` for the synthetic trainability portion of
Gate G2 only.** On 2026-09-02 the operator reviewed the Agent 5 confirmation
packet, ordered the publication of the reviewed branch, and accepted the finding.
The decision, every identity and every number in the packet stand.

Together with the parity evidence accepted under `P18-D004` (30/30 method-map
rows, canned oracle PASS, 16/16 learning-method files byte-identical to the G2
launch manifest), **Gate G2 is closed.** Its two halves are now both answered:
the implementation matches the paper and the pinned published implementation, and
the parity-correct raw learner reliably learns a synthetic setup landscape from
outcome-only feedback.

**G3 remains unstarted.** This acceptance authorizes designing the next gate. It
does **not** authorize:

- executing a G3 run of any size, including a pilot, rehearsal or smoke;
- any Stratego setup-learning game, training run or tuning;
- the full Phase 8 setup-integrated warmstart;
- opening any sealed Phase 8 example.

## What the evidence shows

```text
parity                       86/86 evaluator, 131/131 setup tests; oracle PASS on this
                             run's own seed (loss 1.8e-15, gradients 6.7e-11);
                             30/30 method-map rows; 16/16 method files byte-identical
integrity                    0 legality / orientation / attribution / non-finite /
                             checkpoint-identity events; 12/12 endpoint arrays hold
                             exactly 4,096 finite values
replay                       landscape, initial models, first-period pool and all 4,096
                             of its outcomes, all 64 period digests, and all four
                             evaluation endpoints of every seed: bitwise
binding                      11/11 artifacts bind one source commit
raw gap closure              10.6479% / 10.0073% / 9.5987%    median 10.0073%
raw per-seed paired 95%      [+5.5450,+5.9877] [+5.1404,+5.6011] [+4.9260,+5.3750]
raw pooled paired delta      +5.4282   95% [+5.2985, +5.5585]   n = 12,288
EMA gap closure (telemetry)  0.0986% / 0.1191% / 0.1951%      median 0.1191%
EMA pooled paired delta      +0.0740   95% [+0.0116, +0.1386]  decides nothing
exact optimum                53.031081 (LP duality, re-verified by arithmetic)
Stratego games played        0         sealed Phase 8 access   0
```

**The learning effect itself is strong and replicated across all three seeds.**
This is the part of the result that is not in question. Every seed improved on the
primary endpoint; the three per-seed paired intervals are disjoint from zero with
lower ends above +4.9 utility units; the pooled paired lower bound of +5.2985 on
12,288 common-random-number pairs sits about 0.89 uniform-random SDs above zero
(uniform baseline mean −0.3423, SD 5.9529). The finding replicates the G2 result on
an independently generated table with fresh seed streams and a byte-unchanged
method. Nothing in the reservation recorded below weakens this.

## The margin: met as frozen, but at its edge

The frozen practical-margin criterion is that the **median** of the three seeds'
gap-closure fractions is at least 10%. The observed median is
`0.10007299279907324` — **10.0073%**, which passes.

It passes by **0.0073 percentage points of the gap** — on the median seed, roughly
0.004 utility units out of a 53.63-unit initial-to-optimum gap. Seeds 1 and 2 clear
10% individually; seed 3 (9.5987%) does not.

The rule is applied exactly as frozen, and the frozen rule returns `PROCEED`. But
this review records explicitly, as a durable caveat on how the result may be cited:

> **The 10.0073% median is not robust evidence for a precise 10% practical
> effect.** It is evidence that the learner learns strongly, plus a threshold
> comparison that happened to land on the passing side by 0.0073 percentage
> points. The margin is a fraction of a landscape-dependent gap: on the G2 table
> the same method closed 20.9% / 18.5% / 14.8%, and here it gained about half the
> utility (+5.43 versus +10.77 units) from a higher initial utility toward a lower
> optimum. A fourth table from the same family could fall on either side of 10%
> under the same strength of learning. No downstream document may treat "closes
> ≥ 10% of the gap" as a calibrated, reproducible property of the method.

The packet itself states this in its own "Claim not supported" section and does not
overclaim. The reviewer's requirement carried forward is the one the packet
recommends: **the next gate's margin must be sized in the units the gate is scored
in, from a known instrument resolution**, so that a pass or a fail is not decided
by the draw of the table. This is a design requirement on G3, not a defect in
G2 — `REVISE` was correctly not available, because no implementation or
measurement defect exists and an unfavourable valid result was not observed.

## The frozen contract's contradictory inherited metadata

`reports/phase18/phase18_g2_raw_confirmation_contract_v1.json`
(SHA-256 `9a33147090f035c840f2ebb59a78e73e212acc6b9db9f88a8b45035e1e57d012`)
carries two prose fields inherited from the G2 contract that still describe the
**EMA** as the decision endpoint, contradicting the raw-actor fields that actually
governed this confirmation.

**Authoritative — these fields governed the raw-actor decision:**

| field | content |
|---|---|
| `question.primary_endpoint` | "the RAW generation actor's held-out expected landscape utility at 0 updates and after update 64; this is the primary endpoint for this synthetic trainability assay only" |
| `question.checkpoint_rule` | "the raw actor after the final fixed update (update 64) decides; intermediate curve points are telemetry and never select a checkpoint" |
| `decision_rules.PROCEED` | "… final **raw-actor** mean utility > initial raw-actor mean utility in all three seeds; the lower bound of the pooled paired 95% bootstrap interval > 0; median **raw-actor** gap closure ≥ 0.10" |
| `decision_rules.ema_results` | "recorded as secondary mechanism telemetry; they cannot change the decision" |
| `question.ema_role` | EMA telemetry "cannot change the confirmation decision"; remains the required evaluation/deployment model for every later Stratego-facing stage (S28) |
| `question.evaluation_rule` | endpoint-neutral: common random numbers at snapshot 0 for **every** endpoint (raw and EMA, initial and final), 4,096 samples, sample-count integrity, S24 immediately-terminal handling |

**Inherited and NOT authoritative — stale EMA wording:**

| field | content |
|---|---|
| `design.checkpoint_rule` | "the decision reads the **EMA** after the final fixed update only; intermediate curve points are telemetry and never select a checkpoint" |
| `design.evaluation_rule` | "expected utility = mean landscape utility over the held-out **EMA** samples; the same evaluation stream at every endpoint gives paired per-sample differences" |

The defect is confined to those two prose fields. Every other `design` entry —
`updates` 64, `pool_size` 1024, `batch_size` 1024, `epochs_per_update` 5,
`outcomes_per_setup` 4, `evaluation_samples` 4096, `bootstrap_replicates` 10000,
`gap_closure_threshold` 0.1, the model/table/bootstrap seeds, the namespace and
`training_config_digest` — is a numeric or identity field, was correct, and
governed the run. `design` holds no EMA-decay field; the decay, retained fraction
and time constant live under `method_identity`.

Read together, `question.checkpoint_rule` and `design.checkpoint_rule` name
different endpoints for the same decision. **The `question` and `decision_rules`
blocks are authoritative; the two `design` prose fields are superseded and were not
applied.** The recorded execution confirms this is a documentation contradiction
and not an execution one: `phase18_g2_raw_confirmation_decision_input_v1.json`
records `decision.ema_results_considered = false`, and the decision basis names the
raw-actor criteria alone.

**The frozen contract must remain unchanged and has not been edited.** It was
frozen before any outcome existed (`question.frozen_before_outcomes = true`) and
committed at `G2_RAW_SOURCE_COMMIT ccddceda` ahead of the first outcome; its digest
above is the one the results file records. This review is the authoritative record
of which fields governed. The wording defect is carried forward as a drafting
requirement for the next gate's contract: the endpoint must be named once, in one
block, and inherited prose must be re-read field by field when a contract is cloned.

## Narrative erratum in commit `be4fc84`

The commit message of `be4fc84faed3a29dcf1e4669733a81fe384e813f` ("Record the G2
raw-actor confirmation") states the pooled paired 95% interval as

```text
[+5.2985, +5.6072]        <- commit message, upper endpoint WRONG
[+5.2985, +5.5585]        <- authoritative committed evidence
```

The authoritative value is `upper = 5.558476800891113`, recorded identically in
`phase18_g2_raw_confirmation_results_v1.json` (`raw_criteria.pooled_paired_interval`)
and `phase18_g2_raw_confirmation_decision_input_v1.json`
(`raw_pooled_paired_interval`), and rendered `+5.5585` in `P18-D005.md`,
`P18-D005.json`, `agent_05_report.md`, `decision_index.json`,
`agent_instruction_index.json`, `EVIDENCE_INDEX.md` and the instruction `MANIFEST.md`.

Scope, checked and bounded:

- **The lower endpoint is unaffected.** The message's `+5.2985` matches the
  authoritative `lower = 5.298537681459554` exactly.
- **The decision is unaffected.** The frozen criterion tests the pooled lower bound
  against zero; the upper endpoint enters no criterion.
- **No evidence file is affected.** `git grep '5\.6072'` over the tree at
  `d0fd36d` returns nothing: the error exists only in the commit message and in no
  file content, tracked or untracked.

This is a narrative metadata error of the same class as `E-P18-D003-RULES-1`. The
commit is **not** rewritten and no rerun is warranted; history stays as published
and this record is the correction, consistent with the handling of the earlier
erratum in `phase18_rule_identity_errata_v1.json`.

## Identities checked before publication

- `phase18/g2-raw-confirmation` resolved to
  `d0fd36d9469a048dbbc9aa1a233f6006d78b0e8d`, the reviewed HEAD, before the push.
  The only dirty path in the canonical checkout was the protected
  `reports/phase13/phase14_launch_manifest_v1.json`, which was neither staged,
  edited, restored nor stashed; there were no untracked files.
- At that commit `P18-D005.md` hashes to
  `e34c84baaf8d68a5b41a69135753006e6e73c295c28490fb0680e8ca6745eab4` and
  `P18-D005.json` to
  `38b0c0c06f453892142e539b24f8fbd7e0f98ed1d5069b18f7af446ddb72d795`, the digests
  the decision index recorded at delivery.
- The evidence chain re-verifies by content: the contract digest recorded in the
  results file, the results digest recorded in the decision input, and the
  landscape, launch-manifest and binding-ledger digests all match the files on the
  branch.
- The recorded commits are on the branch in the recorded order:
  `59ddc75` (documentation correction) → `ccddced` (`G2_RAW_SOURCE_COMMIT`) →
  `e980dc7` (verification) → `77ce90b` (launch manifest) → `be4fc84` (result
  evidence) → `d0fd36d` (delivery).

## GitHub publication

Published under section 7 of the decision-packet protocol with a normal non-force
push and no publication commit:

```text
remote        origin  https://github.com/brandonnassir/stratego.git
branch        phase18/g2-raw-confirmation
local SHA     d0fd36d9469a048dbbc9aa1a233f6006d78b0e8d
remote SHA    d0fd36d9469a048dbbc9aa1a233f6006d78b0e8d
local == remote  true
published     2026-09-02T18:20Z  by the Phase 18 Backend Steward
```

The remote branch was absent beforehand (`git ls-remote --heads origin` returned
nothing for it), so the push created the branch: no divergent history existed and
none was overwritten. The remote SHA was confirmed server-side with
`git ls-remote --heads origin` and again through `refs/remotes/origin/…` after a
fresh `git fetch`; all four references agree.

`phase18/g3-backend-foundation` was created from exactly that commit as a ref
pointer, with no working-tree change and no commit of its own. It is local and
unpublished.

## Status recorded by this review

`P18-D005` moves from `AWAITING REVIEW` to **accepted**, and the Phase 18 indexes
and status documents are updated to match:

```text
reports/phase18/decision_index.json                  D005 status -> accepted; publication block
reports/phase18/agent_instruction_index.json         instruction 07 status -> accepted
reports/phase18/agent_05_report.md                   publication line -> published/accepted
instructions/phase_18_setup_integrated_warmstart/MANIFEST.md
stratego_project_docs/STATUS.md                      G2 CLOSED; G3 unstarted
stratego_project_docs/EVIDENCE_INDEX.md              five G2-raw rows -> ACCEPTED
stratego_project_docs/PHASE_HISTORY.md
stratego_project_docs/05_project_plan.md
stratego_project_docs/README.md
```

The delivered decision packets (`P18-D005.md`, `P18-D005.json`) are **not** edited;
as with `P18-D003` and `P18-D004`, their internal publication field records the
state at delivery, and acceptance is recorded here and in the indexes. No frozen
contract, result, manifest, receipt or checkpoint was modified.

## Open item carried forward

`O-P18-EVALRULES-1` remains open and is unresolved by this acceptance:
`phase18_evaluation_contract_v1.json` names *training* rules for future play lanes,
and the training-versus-play-evaluation rule identity must be amended before any
real-game G3/G4 measurement. It did not affect G1 or G2. A proposed amendment is to
be drafted, reviewed and approved separately; until then no G3 evaluation design is
operative.
