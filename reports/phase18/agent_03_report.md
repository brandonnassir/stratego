# Phase 18 Agent 3 — G1 random non-inferiority confirmation

**Status: complete. Decision `P18-D003` = `PROCEED`; Gate G1 closes.**
Result: paired delta **+0.006348**, 95% [**+0.000793**, **+0.011902**] on 4,096
independent pairs against the frozen −0.010 margin. Not pushed — awaiting
review.

## What this work package was

P18-D002 reproduced Phase 8 (42/42 gates) but could not certify the vs-random
margin: ±0.0116 of instrument against a 0.010 margin at 1,024 pairs. The
accepted REVISE authorized exactly one measurement-only repair — same margin,
same two-sided 95% paired-bootstrap rule, same 10,000 replicates, same two
checkpoint byte-identities, a new independent 4,096-pair bank sized for 90%
power at true delta 0. This agent built that instrument, froze it before
outcomes, ran it, and stopped.

## Starting state, and what was adopted

The instruction (SHA verified `8ab53154…`) was first handed to other agents, so
the branch and three uncommitted files already existed. Verified before
continuing:

- `phase18/g1-random-confirmation` branches from exactly the approved Agent 2
  commit `18409f73`, which equals `origin/phase18/setup-integrated-warmstart-g1`;
  the one extra commit (`0756005`) is the operator's own review/authorization
  commit.
- The A4 execution paths did not exist — no run had been started anywhere.
- The three left-over files were reviewed line-by-line against the real APIs:
  `stratego/evaluation/phase18/confirmation_bank.py` and
  `stratego/evaluation/phase18/power.py` were correct and are kept as written;
  `scripts/phase18_g1_random_confirmation.py` ended mid-file (no analysis, no
  CLI) and was completed. All tests are new.

## The instrument

**Bank.** 4,096 pairs from the new `phase18_g1_random_confirmation_v1`
namespace through the recorded `derive_stream_seed`, digest `24b263d2…`.
Separation from `evaluation_setup_bank_v1` is audited in *reflection-class*
space per side, so a mirrored board — or a pair reusing one old side — collides
and is counted. All counters zero, on the full bank, as a test that runs in the
suite.

**Schedule.** One schedule for both arms (digest `2111848b…`): both arms share
the policy token, so match ids, seeds, colours and boards are identical by
construction; the run stage additionally proves rebuilt-digest identity, and
the analysis proves per-row identity across arms on all thirteen case fields.

**Power.** The audit's own arithmetic, implemented and pinned by tests:
2,815 pairs for 80%, 3,769 for 90%, 92.2% at the frozen 4,096; the same helper
reproduces Agent 2's ±0.011599 at 1,024. Frozen in the contract; never
recomputed from outcomes.

**Decision.** Strict `lower > −0.010` per the instruction, reported beside the
original `>=` dialect; a test pins the one place they can differ (exact
equality) and why it is unreachable on dyadic scores at this n.

**Safety rails.** Digest refusal on both checkpoints; chunked resume that
replays only missing chunks, refuses foreign chunk files, and never deletes a
receipt; `planned = completed + failed + missing` accounting where a failure or
gap can never score as a draw and invalidates the primary analysis outright;
per-row immutable receipts from which the analysis recomputes every unit score
and both EWRs exactly; before/after digests of the accepted Phase 8 artifacts;
and **no code path to the sealed Phase 8 test split at all**, pinned by a
source-scan test. 86/86 Phase 18 evaluator tests passed before launch.

## Execution

```text
freeze      contract + bank committed at 9392c6ec (G1_CONFIRM_SOURCE_COMMIT)
worktree    gpt_agent_phase18_g1_confirm_exec, detached at 9392c6ec, porcelain empty
manifest    launch manifest bound to commit+tree+proof, committed at 3a8b30d0
run         16,384 games in 10,953 s (~1.5 games/s), reference arm first
analysis    delta +0.006348, [+0.000793, +0.011902] -> PASS
evidence    committed at c1833ad7; binding ledger shows one source commit
```

Zero integrity events of any kind: no failed or missing game, no retry, no
policy error, no illegal action, no non-finite output, no worker torch import,
no worker checkpoint load. The accepted artifacts hashed identically before and
after both the run and the analysis. Sealed-test examples opened: **zero**.

## Findings worth carrying forward

1. **The power fix worked exactly as computed.** Half-width 0.0056 at 4,096
   pairs versus 0.0116 at 1,024 — the project's standing lesson (third
   margin-vs-power miss, per P18-D002) now has a positive example: margin
   certified on the first properly-sized attempt.
2. **The confirmation delta is +0.006 where the original bank gave −0.000244.**
   Same checkpoints, different setup distributions; the difference is about one
   standard error. Read it as per-bank variation, not as either model changing.
   With only two banks that distribution is unmeasured — worth remembering
   whenever a future gate compares deltas across banks.
3. **The candidate is more decisive against random play**: flag-capture share
   89.7% vs the reference's 86.7%, draws 86 vs 130. Descriptive only.
4. **The interval lies above zero on this bank.** Superiority was not the
   question, was not predeclared, and is not claimed. If anyone later wants a
   superiority reading, it needs its own frozen design.

## Commit trail (all local, none pushed)

```text
0756005  operator: review acceptance + Agent 3 authorization (pre-existing)
9392c6e  G1_CONFIRM_SOURCE_COMMIT: driver, modules, tests, contract, bank
3a8b30d  launch manifest bound to 9392c6e from the clean worktree
c1833ad  result evidence: 4 result artifacts + receipts + run/arm records
(next)   this report, P18-D003, index and documentation updates
```

The protected `reports/phase13/phase14_launch_manifest_v1.json` modification
was never staged, edited, or restored.

## Stop state

Stopped after committing. No G2 work, no setup-model code, no push. After
review accepts `P18-D003`, publication is a normal non-force push of the exact
approved branch HEAD with local/remote SHA verification recorded — per §7 of
the decision-packet protocol.
