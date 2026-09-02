# Phase 18 — Agent 4

## Publish accepted G1, then execute Gate G2 setup parity and synthetic learning

## Authorization and decision

`P18-D003` is accepted as `PROCEED`. Gate G1 is closed.

Approved identities:

```text
branch
  phase18/g1-random-confirmation

approved branch HEAD
  ef7523c1940650c0906d1927e64679e8328a663f

P18-D003 JSON SHA-256
  57281f6a5c460a9a524bd0538ed44050e2949dd12873401cc0a3f8521d6d8fdb

P18-D003 Markdown SHA-256
  7d930357e2c324d7c9ff3e0622637e5014fdbdba0390e4c06156e95b24266ab7

G1 confirmation source commit
  9392c6ec1c948a7c5c91278616f669340f4a6445

G1 result-evidence commit
  c1833ad7a36fb0c98d6fada02326316d50d7f284
```

Independent review reproduced the 4,096-pair result exactly:

```text
reference EWR       0.9559326171875
candidate EWR       0.9622802734375
paired delta       +0.00634765625
95% interval       [+0.00079345703125, +0.01190185546875]
margin             -0.010
decision            PASS
tests               86/86
```

This authorizes Gate G2 only. It does not authorize Stratego setup training, Phase 8
trainer integration, G3, a tandem pilot, or a production run.

## Part A — Publish and establish the source boundary

1. Fetch `origin` without merging.
2. Verify the local branch `phase18/g1-random-confirmation` resolves exactly to
   `ef7523c1940650c0906d1927e64679e8328a663f`.
3. If the remote branch already exists, require it to be either absent or at that
   exact commit. Do not overwrite divergent remote history.
4. Publish the exact approved head with a normal, non-force push.
5. Verify and record that local and remote SHAs are both
   `ef7523c1940650c0906d1927e64679e8328a663f`.
6. Do not add a publication commit to the approved G1 branch.
7. Create:

```text
phase18/g2-setup-parity
```

from the exact approved G1 commit. If it already exists locally or remotely, stop
rather than deleting or moving it.

Continue protecting this historical user modification:

```text
reports/phase13/phase14_launch_manifest_v1.json
```

Do not edit, stage, restore, stash, or delete it. Use explicit-path staging only.

## Part B — Record the G1 review and erratum

Create:

```text
reports/phase18/reviews/P18-D003_REVIEW.md
reports/phase18/phase18_rule_identity_errata_v1.json
```

Record that:

* `P18-D003` is accepted and G1 is closed.
* Its scientific result and all artifact identities were independently verified.
* `P18-D003.json` describes `battleless_move_limit` as 100, but the actual G1
  contract, engine constant, schedule, and receipts consistently use the accepted
  evaluation value of 200.
* This is a narrative metadata error and does not require rerunning G1.
* `phase18_evaluation_contract_v1.json` separately names training rules with a
  100-move limit. Preserve that frozen file, but mark the training-versus-evaluation
  rule choice as requiring an explicit amendment before any real-game G3/G4
  evaluation.

Do not rewrite the frozen G1 result packet. Correct it through the review and errata
trail.

Update the decision and instruction indexes, evidence index, status document, and
Phase 18 manifest to mark `P18-D003` accepted and published.

Save this work package as:

```text
instructions/phase_18_setup_integrated_warmstart/06_AGENT_4_G2_SETUP_PARITY_AND_SYNTHETIC_ASSAY.md
```

Commit the review, errata, instruction, and status changes before implementing or
training anything.

## Part C — Gate G2 mission

Answer one question:

Does the local scaled setup-policy implementation match the paper and pinned
published implementation at loss, gradient, sampling, aggregation, optimizer,
checkpoint, and EMA semantics—and can it reliably learn a known synthetic
setup-reward landscape from outcome-only feedback across three fresh seeds?

G2 has two inseparable sub-gates:

1. Method and implementation parity.
2. Synthetic learning with a known answer.

No Stratego games or sealed Phase 8 examples may be opened.

## Part D — Required sources

Read completely:

```text
instructions/phase_18_setup_integrated_warmstart/
  00_PHASE_18_ADAPTIVE_SEQUENCE_AND_COMMON_CONTRACT.md
  02_PHASE_18_DECISION_PACKET_AND_NEXT_AGENT_PROTOCOL.md
  03_SETUP_MODEL_INTEGRATION_REFERENCE.md

reports/phase18/ataraxos_setup_method_map_v2.md
reports/phase18/ataraxos_setup_method_map_v2.json
reports/phase18/decisions/P18-D003.json
reports/phase18/decisions/P18-D003.md
reports/phase18/reviews/P18-D003_REVIEW.md

Phase 17 setup-model source and tests
2511.07312v1.pdf

Authors’ implementation at:
92db29e8ffc323b1b8a2804b5c3f84695d036b05
```

The paper and published source are technical references. This instruction and the
Phase 18 governing contracts are authoritative.

## Part E — Implementation boundary

Add Phase 18 code under new namespaces, preferably:

```text
stratego/training/phase18/
tests/training/phase18/
scripts/phase18_g2_setup_parity.py
reports/phase18/g2/
```

Phase 17 code may be adapted but must not be edited in place. Do not modify the
Phase 8 trainer, move model, accepted checkpoints, or historical evidence.

The scaled setup model remains frozen at:

```text
decoder blocks                4
width                         128
attention heads               4
feed-forward width            512
trainable parameters          exactly 802,320
sequence                      start token + 40 piece tokens
piece head                    12 inventory-masked classes
value head                    loss/draw/win
entropy head                  normalized suffix entropy
positional initialization     standard deviation 0.1
```

Implement every setup-parity item S01–S30 in `ataraxos_setup_method_map_v2.json`.
Produce a machine-readable coverage table mapping every row to its test and
implementation location. No row may be marked complete based only on documentation.

Mandatory semantics include:

```text
entropy target       h_target = I / 10
advantage residual   I - 10h
setup outcome        mean of all completed outcomes for the exact setup/snapshot
zero outcomes        excluded, never converted into a draw
PPO clip             0.2
reverse KL           KL(current || behavior), coefficient 0.1
value coefficient    0.5
entropy coefficient  1.0
effective batch      1,024 setups
epochs               5
optimizer             AdamW, lr 5e-5, weight decay 0
gradient clipping    0.5
EMA                   0.999, once after the complete setup update
generation actor      raw model only
evaluation model      EMA only
```

Generation must use forced flag handedness followed by independently seeded 50%
horizontal reflection. All behavior probabilities must be gathered in canonical
network orientation.

## Part F — Independent canned parity oracle

Build an implementation-independent reference calculation for a small fixed batch.
It must verify:

* inventory masks and causal prefix alignment;
* realized suffix information;
* W/D/L ordering and expected value;
* repeated-outcome aggregation;
* `I - 10h`;
* PPO ratio and clipping;
* reverse-KL direction;
* all four loss coefficients;
* total loss;
* gradients for representative parameters;
* optimizer equivalence at zero weight decay;
* effective batch and five-epoch step counts;
* gradient clipping;
* raw/EMA separation and EMA cadence; and
* save/reload/one-more-update behavior on the production device.

The reference oracle must not call the production loss implementation to calculate
its expected values.

## Part G — Freeze the synthetic assay before training

Before the first optimizer step, create and commit:

```text
reports/phase18/phase18_g2_contract_v1.json
reports/phase18/phase18_g2_synthetic_landscape_v1.json
reports/phase18/phase18_g2_launch_manifest_v1.json
```

The landscape must:

* be deterministic and versioned;
* operate only on legal 40-piece canonical setups;
* be horizontally reflection-invariant;
* use a fixed additive piece-type-by-square utility table;
* have an independently calculated exact optimum;
* have a separately estimated fresh-policy baseline;
* expose no gradient or utility value to the learner;
* return only seeded W/D/L outcomes to the setup-learning path; and
* give every eligible setup multiple independently seeded outcomes so
  repeated-outcome aggregation is genuinely exercised.

Use three fresh model/training seeds derived through `derive_stream_seed` from:

```text
phase18_g2_setup_parity_v1
```

Freeze before training:

* all model, landscape, outcome, sampling, evaluation, and bootstrap seeds;
* the synthetic-outcome mapping;
* outcomes per setup;
* maximum update count;
* evaluation sample size;
* final-checkpoint rule;
* uncertainty calculation;
* practical improvement threshold; and
* exact `PROCEED`, `REVISE`, `STOP`, and `BLOCKED` interpretations.

Minimum assay requirements:

```text
training seeds                       3
ready setups per update              1,024
effective optimizer batch            1,024
epochs per update                    5
maximum setup updates                64
outcomes per eligible setup          at least 4
held-out EMA setups per endpoint     at least 4,096 per seed
checkpoint used for decision         final fixed update, not best-of-run
bootstrap replicates                 10,000
```

At minimum, `PROCEED` requires:

* every S01–S30 parity requirement passing;
* canned forward-loss and gradient parity passing;
* zero legality, orientation, attribution, non-finite, or checkpoint-identity
  failures;
* final EMA expected utility above initial EMA utility for all three seeds;
* the pooled paired 95% lower bound for final-minus-initial utility strictly above
  zero; and
* the median seed closing at least 10% of its initial-to-exact-optimum utility gap.

Do not change the landscape, threshold, update budget, or seed after an optimizer
step. Do not select the best intermediate checkpoint.

## Part H — Clean execution and artifact location

Commit the complete implementation, tests, frozen landscape, contract, and launch
machinery before training. Record this as `G2_SOURCE_COMMIT` with its tree SHA.

Create a clean detached execution worktree at:

```text
/Users/brandonwashington/Dev/Github/stratego/gpt_agent_phase18_g2_exec
```

Unlike G1, keep new runtime artifacts inside the project’s canonical tree:

```text
/Users/brandonwashington/Dev/Github/stratego/gpt_agent/artifacts/phase18/g2_setup_parity_v1
```

Ensure that runtime directory is explicitly Git-ignored before execution. Record both
its absolute path and repository-relative path. Do not relocate or delete the existing
G1 artifacts.

If either execution path already exists, stop instead of deleting or reusing it.

## Part I — Verification and decision

Run:

* the complete pre-existing Phase 18 evaluator suite;
* all new Phase 18 setup tests;
* independent loss/gradient parity;
* deterministic replay of the synthetic landscape;
* all three frozen synthetic training seeds; and
* artifact/hash/accounting verification.

Produce:

```text
reports/phase18/agent_04_report.md
reports/phase18/phase18_g2_results_v1.json
reports/phase18/phase18_g2_binding_v1.json
reports/phase18/decisions/P18-D004.md
reports/phase18/decisions/P18-D004.json
```

`P18-D004` must choose exactly one:

```text
PROCEED  parity passes and the synthetic learning criteria pass
REVISE   a concrete isolated implementation/instrument defect prevents a valid result
STOP     a valid parity-correct implementation fails the frozen synthetic learning criteria
BLOCKED  required evidence cannot be produced because of an unresolved dependency
```

Separate observed facts, supported inferences, plausible explanations, and unsupported
claims.

Commit all authorized source, tests, evidence, reports, and `P18-D004` locally on
`phase18/g2-setup-parity`. Do not push the unreviewed G2 result. Do not begin G3 or
play any Stratego setup-learning games. Stop for review.
