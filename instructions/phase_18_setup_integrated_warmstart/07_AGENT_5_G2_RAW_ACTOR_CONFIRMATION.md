# Phase 18 — Agent 5

## Bounded G2 raw-actor confirmation

_Saved verbatim from the operator's work-package message of 2026-09-02. The heading
above and this note are the only additions._

---

You are Agent 5 for Phase 18. Your assignment is a bounded G2 raw-actor confirmation. Do not begin G3, Stratego game experiments, or Phase 8 warmstart training.
Repository:
`/Users/brandonwashington/Dev/Github/stratego/gpt_agent`
1. Publish the approved Agent 4 work

1. Confirm that `phase18/g2-setup-parity` points exactly to:
`6afa13bed355884a3327d2661fd739784260dc2b`
2. Confirm that the only pre-existing uncommitted change is:
`reports/phase13/phase14_launch_manifest_v1.json`
3. Do not stage, edit, restore, or otherwise modify that protected historical file.
4. Push the exact reviewed `phase18/g2-setup-parity` branch to GitHub.
5. Create a new branch from that exact commit:
`phase18/g2-raw-confirmation`

2. Correct the G2 documentation
Preserve `P18-D004 = REVISE`, but correct these statements:

* Do not claim that the EMA endpoint was mathematically unreachable regardless of learner behavior.
* State that decay 0.999 retained `0.999^64 = 0.937975` of the initial parameter contribution, giving the EMA an approximately 1,000-update time constant.
* State that the frozen 64-update assay empirically showed severe EMA lag: the raw actor learned strongly while the EMA captured only a small portion of the observed change.
* State that development smokes informed the raw diagnostic and the instrument-defect interpretation. They did not change the frozen primary landscape, budget, threshold, or run seeds.

Do not rewrite or suppress the original results.
3. Freeze the bounded confirmation before running it
Create a new frozen contract and namespace for an independent confirmation.
Primary question:
“Using the parity-correct setup-learning method, does the raw generation actor reliably learn an independently generated synthetic setup landscape within 64 setup updates?”
The raw actor is the primary endpoint only for this synthetic trainability assay. The EMA remains the required evaluation/deployment model for every later Stratego-facing stage.
Freeze before observing any confirmation outcomes:

* A new synthetic landscape table generated from the same landscape family and methodology, but not the previously observed table.
* A new artifact namespace.
* Three entirely fresh model/training seed streams.
* Fresh pool, outcome, endpoint-evaluation, and bootstrap seeds.
* Deterministic seed derivation from the reviewed base commit, fixed namespace, and domain-separated labels.
* The exact landscape optimum and its independent certificate.
* All thresholds and decision rules below.

Do not hand-select a favorable table or seeds. Do not change any frozen field after the first outcome is generated.
4. Keep the learning method unchanged
Use the already verified G2 implementation without tuning:

* Existing 802,320-parameter setup model
* Existing inventory masking and forced-flag procedure
* Independent 50% reflection
* Raw actor for setup generation
* Four terminal outcomes per accepted setup
* Pool capacity 1,024
* Batch size 1,024
* Five optimizer epochs per setup update
* AdamW, learning rate `5e-5`, weight decay `0`
* Gradient clipping at `0.5`
* Existing PPO, value, entropy-target, and reverse-KL objectives and weights
* EMA decay `0.999`, updated exactly once after each complete setup update
* Maximum 64 setup updates
* Exactly 320 optimizer steps and 64 EMA updates per completed seed

Do not tune the model, optimizer, losses, decay, budget, or landscape after seeing results.
5. Evaluation design
For each of the three seeds:

* Evaluate the raw actor before training and after update 64.
* Use 4,096 held-out setups at each endpoint.
* Use common random numbers for the paired initial/final comparison.
* Exclude immediately terminal setups under the existing rule.
* Record legality, orientation, attribution, checkpoint-identity, nonfinite, and sample-count integrity checks.
* Continue recording EMA results as secondary mechanism telemetry. EMA results must not change the confirmation decision.

Use 10,000 paired bootstrap resamples for the pooled confidence interval.
6. Frozen decision rule
`PROCEED` only if all of the following hold:

1. All parity, replay, binding, and integrity checks pass.
2. Final raw-actor mean utility is greater than initial raw-actor mean utility in all three seeds.
3. The lower bound of the pooled paired 95% bootstrap interval is greater than zero.
4. Median raw-actor gap closure across the three seeds is at least 10%.

Use `STOP` if the parity-correct raw learner fails these criteria without a concrete defect.
Use `REVISE` only if a specific implementation or measurement defect invalidates the run. Do not reclassify an unfavorable valid result as an instrument problem.
A passing result closes only the synthetic trainability portion of G2. It authorizes designing the next gate; it does not authorize launching G3 or the full warmstart.
7. Execution discipline

* Run final verification from a clean, detached worktree bound to one frozen source commit.
* Generate a launch manifest before producing outcomes.
* If a post-freeze implementation correction is necessary, abandon the affected namespace and document it. Do not silently amend and continue.
* Do not access sealed Phase 8 evaluation evidence.
* Do not run any Stratego setup-learning games.
* Store large runtime artifacts under an ignored project-root artifact directory, for example:
`artifacts/phase18/g2_raw_confirmation_v1`

8. Deliverables
Produce:

* Frozen confirmation contract
* New synthetic landscape and optimum certificate
* Preflight/parity verification record
* Launch manifest
* Per-seed results and receipts
* Raw and EMA endpoint utility arrays
* Checkpoint manifests and identity digests
* Replay report
* Binding ledger
* Combined results report
* `P18-D005` decision in Markdown and JSON
* Agent 5 report describing exactly what ran, what did not run, and the recommended next question
* Relevant documentation/index updates

Commit all tracked work locally with a descriptive commit message. Report the branch name and exact commit hash. Do not push the new confirmation branch until it has been reviewed and approved.
