# Phase 15 — Agent 1
## Clean Belief-Corpus Generation and B18/B24 Fine-Tuning

## Mission

Create two belief specialists from the Phase 14 hour-18 and hour-24 candidates:

```text
P18 -> B18
P24 -> B24
```

`P18` and `P24` remain immutable policy/value players. `B18` and `B24` are separate belief-only models trained on a new, correctly oriented hidden-piece corpus.

This is an engineering phase. Optimize for a reliable handoff to search, not for a publication-style claim.

Do **not** implement search. Do **not** alter, pause, stop, kill, restart, or finalize any running Phase 14 task.

## 1. Process boundary

This instruction does not authorize process control.

Before any sustained corpus generation or training:

1. inspect the live process/status state read-only;
2. confirm that the requested work will not compete with a running Phase 14 learner, evaluator, supervisor, dashboard, or collector;
3. if Phase 14 is still consuming the required compute, finish implementation and CPU-only correctness tests, report `ready_for_compute`, and stop for operator review.

Never create an emergency-stop file, send a signal, edit live run state, rotate live checkpoints, or invoke a closeout/finalize command.

## 2. Preserve project history

Treat all accepted and live Phase 9–14 artifacts as read-only, including:

- accepted Phase 9 checkpoint and reports;
- all Phase 11 and Phase 11B data, checkpoints, reports, and closure records;
- all Phase 12 search code, candidate artifacts, and reports;
- all Phase 13 contracts and launch artifacts;
- the Phase 14 run state, hot checkpoints, archive, candidate ledger, candidate weights, evaluations, logs, and control files;
- the currently modified `reports/phase13/phase14_launch_manifest_v1.json`.

Do not clean, reset, rewrite, or include unrelated working-tree changes.

All implementation and evidence from this task must use additive Phase 15 namespaces, for example:

```text
stratego/belief/phase15/
tests/belief/phase15/
scripts/run_phase15_agent01.py
data/phase15/
checkpoints/phase15/
reports/phase15/
```

## 3. Resolve and freeze P18/P24

Resolve hour 18 and hour 24 from the Phase 14 candidate ledger and evaluation root. The ledger is authoritative; do not guess from the newest hot checkpoint.

Expected logical identities are:

```text
P18 = Phase 14 candidate hour_018
P24 = Phase 14 candidate hour_024
```

For each source:

1. verify that the candidate evaluation is complete;
2. verify the candidate file and embedded hour/optimizer-step identity;
3. compute file SHA-256 and model-state digest;
4. create a Phase 15 read-only source copy or immutable binding;
5. record the original path, copy/binding path, size, digest, optimizer step, and candidate evaluation reference.

Never train from a rolling `hot_*.pt` file. Never overwrite the Phase 14 candidate bytes.

Call the frozen direct models:

```text
P18 = hour-18 policy/value model
P24 = hour-24 policy/value model
```

## 4. Critical orientation correction

The old Phase 11B corpus and the Phase 12 match packs are contaminated by a Blue setup-orientation defect. `Phase11BSetupSources.draw()` returned canonical own-orientation tuples, and the old glue passed those tuples directly to `create_game()` for Blue.

Therefore:

- do not reuse the old Phase 11B corpus as training, calibration, or development data;
- do not call the old `corpus_plans()` path as the new collector;
- do not copy the old Phase 12 board-construction glue;
- do not treat old Phase 12 strength numbers as validation of the new system.

Every setup must pass through the accepted orientation path before `create_game()`:

```text
SelectorDraw.oriented(player)
SampledSetup.oriented(player)
or orient_setup(canonical, player)
```

Add a hard runtime assertion and tests proving:

```text
Red engine row  = canonical rank
Blue engine row = 9 - canonical rank
```

At minimum, verify Flag location, legal setup rows, complete inventory, and Red/Blue paired orientation on thousands of generated boards. Include a negative canary showing that passing Blue's canonical tuple directly is detected.

No corpus generation may begin until this gate passes.

## 5. Build `phase15_belief_corpus_v1`

Generate a new position-budgeted corpus. The stopping unit is eligible observer positions, not completed games.

Initial engineering target:

```text
train        120,000 positions
calibration   15,000 positions
development   20,000 positions
total        155,000 positions
```

If a throughput pilot shows this is impractical, preserve the pilot evidence and use no fewer than:

```text
train         80,000
calibration   10,000
development   10,000
```

Do not silently reduce the target.

An eligible position is one where:

- the observer is to act;
- at least one opponent piece is unresolved;
- the public observation and privileged replay agree exactly;
- a non-empty legal hidden-rank target exists.

Sample a bounded number of evenly spaced eligible decisions from each trajectory. A trajectory may be retired when its useful positions have been collected; it need not finish. Preserve the accepted game termination cap so a pathological long game cannot monopolize collection.

## 6. Corpus mixture

Balance the observer source exactly or as closely as the final position counts allow:

```text
50% observer P18
50% observer P24
```

Use this opponent mixture within each observer half:

```text
25% P18
25% P24
10% strategic_rule_based
10% tactical_rule_based
10% stress_scout_rush
10% stress_miner_rush
10% stress_information_miser
```

Balance Red/Blue observer color within every major cell.

Use a setup mixture that contains:

```text
35% neutral_v1
45% accepted P10-D / Phase 14 learned setup source
20% targeted unusual families
```

The targeted family share must cover, without inventing new setup semantics:

```text
high_bomb_placement
aggressive_high_rank_front
distributed_bomb_defense
corner_flag_fortress or near_corner_flag_fortress
scout_forward_information
miner_forward
irregular_high_entropy
```

Use the accepted setup library and orientation helpers. Use the train library split for training and non-overlapping validation identities for calibration/development. Record exact family/source/color/model counts after position sampling, not merely intended game counts.

## 7. Public/privileged boundary

Use the established two-pass pattern:

```text
public pass:
    model sees only legal observation/public state
    records action history, position identity, and observation digest

privileged replay pass:
    reconstructs the same state
    verifies the observation digest
    reads true hidden ranks only as labels
```

Store public inputs and privileged labels separately.

Every sample must contain at least:

```text
127 x 10 x 10 public observation
public-state document/id
observer color
observer model: P18 or P24
opponent identity
setup source and setup family
game id and decision index
hidden-piece mask / perspective squares
remaining public inventory
public legal-rank masks
true hidden-rank labels (privileged store only)
```

True ranks may never enter model inputs, setup-selection features visible to the player, search priors, or public metadata.

Make train/calibration/development identities disjoint and produce a manifest with file digests, sample counts, hidden-piece counts, orientation checks, and split-overlap checks.

## 8. Build B18 and B24

Reuse the successful Phase 11B 1C architecture pattern, but bind it independently to each new policy backbone.

For B18:

```text
frozen P18 prefix: first three C1 transformer blocks
trainable copy:   final C1 block + encoder norm
fresh belief MLP: 128 -> 512 -> 512 -> 12, GELU
```

For B24:

```text
frozen P24 prefix: first three C1 transformer blocks
trainable copy:   final C1 block + encoder norm
fresh belief MLP: 128 -> 512 -> 512 -> 12, GELU
```

Policy and value parameters must not be part of either optimizer. Prefer a structural design where B18/B24 checkpoints contain only:

- copied final block;
- copied encoder norm;
- belief MLP;
- calibration temperature;
- source policy identity and digests;
- corpus/config identity.

The deployed move models remain the untouched P18/P24 objects. Outputs from a fine-tuned belief copy's policy or value head are irrelevant and must never be used.

Add tests proving:

```text
P18 digest before training == P18 digest after training
P24 digest before training == P24 digest after training
policy/value parameter gradients are absent
belief checkpoint loads only with its recorded source identity
```

## 9. One fixed training recipe

Use one declared recipe for both specialists, not a sweep:

```text
loss                    hidden-piece cross-entropy
optimizer               AdamW
head learning rate      1.0e-3
final-block learning rate 1.0e-4
weight decay            1.0e-4
schedule                cosine
batch size              256 positions initially
maximum epochs          12
early-stop patience     3 development evaluations
selection               best development cross-entropy
```

Run a short throughput/memory pilot before the full jobs. Adjust batch size only for memory/throughput safety and record the change; do not change the statistical recipe separately for B18 and B24.

Evaluate every epoch and preserve the best checkpoint immediately. Do not allow a repeat/reproducibility pass to overwrite the selected bytes; every run must use a unique output path and digest.

## 10. Calibration

Search needs probabilities it can trust. Fit one positive scalar temperature per selected belief model using only the calibration split.

Report raw and temperature-calibrated metrics on development. Keep the calibrated version only if it improves development NLL and calibration error without changing legality or interface behavior. Temperature scaling must not change top-1 labels.

Record the fitted temperature inside the checkpoint and provider identity.

## 11. Required metrics

For B18 and B24, report at minimum:

```text
cross-entropy / NLL
remaining-count baseline cross-entropy
R_CE
top-1 accuracy
Brier score
expected calibration error
maximum calibration error
raw vs calibrated metrics

metrics by:
    observer color
    observer source P18/P24
    opponent class
    setup source/family
    early/middle/late game band

training wall-clock
time to best checkpoint
inference latency
peak memory
parameter counts
```

Also compare against the surviving old Agent 1C belief model as a reference on the **new** development corpus. Do not use the old contaminated development result as the comparison set.

## 12. Belief/sampler interface

Expose each selected specialist through the search-compatible interface:

```text
predict_marginals(public_state)
    -> 12-way rank probabilities for unresolved opponent pieces

sample_worlds(public_state, n, seed)
    -> complete legal hidden armies
```

Feed marginals through the existing accepted constrained-world sampler by adapter/import. Do not independently sample each piece and do not alter accepted inventory or movement-impossibility constraints.

Validate:

```text
probabilities finite and sum to one
fixed seed reproduces worlds
remaining piece counts exact
moved pieces never assigned Flag/Bomb
all sampled worlds pass accepted validation
no true rank accessible through public interface
```

## 13. Deliverables

Create at minimum:

```text
stratego/belief/phase15/...
tests/belief/phase15/...
scripts/run_phase15_agent01.py

data/phase15/phase15_belief_corpus_v1/...
data/phase15/phase15_belief_corpus_v1_manifest.json

checkpoints/phase15/p18_source_identity.json
checkpoints/phase15/p24_source_identity.json
checkpoints/phase15/b18_belief_v1.pt
checkpoints/phase15/b24_belief_v1.pt

reports/phase15/agent_01_learning_curves.json
reports/phase15/agent_01_metrics.json
reports/phase15/agent_01_report.md
reports/phase15/agent_01_summary.json
reports/phase15/phase15_search_handoff_v1.json
```

The handoff must bind exact digests for P18, P24, B18, B24, the corpus, calibration values, provider interface version, and accepted sampler version.

## 14. Completion and stop condition

Finish when:

- the new orientation-safe corpus is generated and verified;
- B18 and B24 are trained without changing P18/P24;
- calibrated development metrics are recorded;
- both providers generate legal deterministic worlds;
- the exact search handoff is written.

Then stop and report.

Do not implement search, choose the final combined player, modify Phase 12, or control any running Phase 14 task.
