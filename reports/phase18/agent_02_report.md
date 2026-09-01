# Phase 18 Agent 2 — G1 source closure and the Phase 8 reproduction control

**Decision: `REVISE`. Packet: `P18-D002`.**

The accepted Phase 8 warmstart reproduces. All 42 original gates pass, seven of
eight frozen paired margins clear, and the eighth fails for a reason that has
nothing to do with the model: its margin was set tighter than the evaluation can
resolve.

```text
G1_SOURCE_COMMIT  66b733ad92324751e30bd7e2a5e373129cbe87c3
source tree       2d1dcda2a05edf1121875709d4728a9b14ecc7a8
branch            phase18/setup-integrated-warmstart-g1
worktree          /Users/brandonwashington/Dev/Github/stratego/gpt_agent_phase18_g1_exec
run output        /Users/brandonwashington/Dev/stratego_phase18/g1_control_v1
run id            G1-CONTROL-2026-A
authorized by     P18-D001 (499b7d9a… / bf4b13b8…)
```

## 1. Source closure

The starting checkout was dirty by design: 41 entries, all Phase 17 evidence,
the Phase 18 package, or documentation. Every one was classified before any Git
action. Forty fell inside the instruction's allowlist; one — the Phase 14 launch
manifest — is the protected historical path, and it was left exactly as found.

Five commits were made on the dedicated branch. History was not rewritten.

| commit | paths | what |
|---|---|---|
| `121c799` | 125 | Phase 17 evidence closure and its closeout documentation |
| `1f1a5a9` | 17 | Phase 18 G0 research boundary and decision trail |
| `1bd674f` | 3 | corrected four tracking statements the first commit falsified |
| `a7c389e` | 7 | G1 harness — **superseded before anything bound it** |
| `66b733a` | 3 | **G1_SOURCE_COMMIT** |

The third commit exists because committing the Phase 17 evidence made four
sentences in the closeout documentation false — `STATUS.md` claims to answer
"what is true right now", and it said the evidence was untracked. The correction
is narrow and keeps the distinction that matters: the reports and code are now
committed, `checkpoints/phase17/` (33.5 GB) is still gitignored on purpose, and
the Phase 15–16 untracked risk is explicitly **not** resolved.

`a7c389e` was superseded, not amended. Its accepted-artifact protection looked
for the Phase 8 checkpoints under the repository root — the one place
`.gitignore` guarantees a worktree cannot have them. No boundary artifact, launch
manifest or run output existed at that commit, so nothing was bound to it.

**The protected Phase 14 manifest is untouched.** File `f300873e…`, working blob
`805f403a…`, committed blob `dd22a07c…` — all three identical to the values
recorded before the first Git command, and it appears in none of the five
commits.

## 2. Two structural findings about running from a worktree

Both were discovered by running, not by reading, and both are recorded rather
than smoothed over.

**Agent 6's corpus check is pinned to the original checkout directory.** It
asserts the corpus root three ways; the third compares against `REPOSITORY_ROOT`,
which no execution worktree can satisfy. That would be cosmetic except that
`verify_prerequisites` skips the corpus digest verification entirely when its
problem list is non-empty — so letting it fail would have silently dropped the
28,000-payload check that Part B names as a launch precondition. One constant is
rebound so the third assertion collapses onto the first. The resolver assertion,
the pointer assertion, and the full byte-level corpus verification all ran
untouched; `--prove-location-assertion` prints the untouched behaviour from
either location, and it is recorded from both.

**A git worktree is a source closure, not a data closure.** The first suite run
in the worktree reported 54 failures and 378 errors. Every one traced to a
gitignored artifact that simply is not in a fresh worktree. Rather than iterate
at ten minutes a round, I scanned the source for artifact path literals and found
all of them at once: 18 files, 78 MB, staged read-only with every digest verified
against the original. The suite then came back green.

## 3. Boundary and launch

`phase18_process_boundary_v2.json` (`6805254a…`) measures the source fields from
the clean detached worktree and is written afterwards, so it cannot create a
circular source identity. All ten readiness checks pass with zero problems:

```text
execution worktree clean          empty porcelain, detached at 66b733a
G1_SOURCE_COMMIT resolves         tree 2d1dcda
Phase 8 identities recompute      zero problems
28,000 corpus payloads verify     byte level, 76.9 s
accepted digests match            f7e9c40d… / 7e2af5dd… / 01c907ee…
canonical fresh C1 checksum       cfe60bb0…
full suite green                  7477 passed, 46 skipped, 0 failed, 0 errors
no process or lock conflicts      none running
protected manifest intact         three digests unchanged
staged artifacts identical        18/18
```

One honest qualification: 46 skips against the original checkout's 3. The 43
extra are artifact-gated tests for Phase 9/14/15/16/17 *run* data that is not in
the worktree and is not stageable at reasonable size. None is a Phase 8 test —
all 339 Phase 8 and warmstart tests pass with zero skips, as do all 40 new Phase
18 tests. It is a measured coverage difference, not an equivalence.

`phase18_g1_launch_manifest_v1.json` (`e86e2313…`) repeats the commit and the
boundary digest; every downstream artifact repeats the same commit.
`phase18_g1_binding_v1.json` is where that is checked in one place — it names all
eleven G1 artifacts, their digests, and the field through which each records the
commit. Three of them do so through the accepted Agent 7 environment field, which
stores git's abbreviated hash; the ledger accepts an abbreviation only after
verifying it is a prefix of the full commit.

## 4. The control

```text
updates            25,000 (not shortened; --dry-run means output isolation only)
examples consumed  6,399,705 — the same count as the accepted run
fresh init         cfe60bb0… verified before the first optimizer step
restart            real, at 12,500; resumed at epoch 2 position 705536
selected           step 25,000 by validation only (accepted selected 24,000)
reload             reproduced the selection score, absolute delta 0.0
training wall      3853.8 s (1.070 h; accepted 1.084 h)
Agent 6 gates      26/26
accepted artifacts unchanged before and after
```

The validation trajectory tracked the accepted run within ±0.004 selection score
at every checkpoint measured. The full-validation confirmation came out slightly
*better* than the accepted run on all three heads.

## 5. The evaluation, and why its fidelity is measured

The accepted Agent 7 harness cannot evaluate an explicit checkpoint without
claiming the accepted output paths, exactly as A4 anticipated. The wrapper drives
Agent 7's own stage functions twice over one schedule — same policy token, so the
match ids and seeds are identical by construction, and the schedule digests are
compared as proof.

The reference arm runs first, and this is the part worth emphasising: **it
reproduced the accepted checkpoint's published results exactly.** All ten
headline metrics at delta `0.0` — the sealed-test heads to ten decimal places,
the random gate EWR and its bootstrap lower bound, the vs-init EWR and its lower
bound. Any difference the candidate then shows is the candidate, not the harness.

That also settles something the contract had assumed the other way: the greedy
evaluation path **is** bitwise reproducible on MPS. Only the 25,000-step training
run carries the nondeterminism the contract warns about.

## 6. The result

All 42 original gates pass for the reproduction. Seven of eight margins clear.
The eighth, vs-random EWR, has a point estimate of **−0.000244** — one paired
unit in 1,024 — and a 95% interval of [−0.011963, +0.011230] against a −0.010
margin.

```text
paired units                      1,024
sd of paired differences          0.189374
95% half-width                    0.011599   the instrument's resolution
predeclared margin                0.010      what it had to certify
pairs needed to certify 0.010     1,448      the frozen bank holds 1,024
units on which the arms differ    171 of 1,024
```

The margin is 16% tighter than the evaluation can resolve. The bootstrap agrees
with the normal approximation to four decimals, so the interval is right — the
ruler is short. vs-init, with a *looser* margin on *half* the pairs, clears with
room; the tightening of vs-random was justified in the contract by gate headroom,
which is not the same quantity as a paired instrument's resolution.

P18-D001 flagged this in advance: the margins were predeclared judgments with no
measured run-to-run distribution behind them.

**The margin is not lowered and the gate is not converted into telemetry.** Under
the frozen rule G1 does not pass, and the packet says so.

The replicate that section C4 conditionally authorizes is declined, with reasons
stated in the packet: C4 exists to test whether an out-of-margin *delta* is
run-to-run noise, and this delta is −0.000244. A replicate cannot narrow a
within-run interval, would cost another sealed-split access, and would decide the
gate on the sign of a near-zero difference. It remains available on request.

## 7. Sealing

The control evaluated 0 test examples and played 0 Phase 4 neural games. The
sealed split opened twice, both after the selection was frozen and verified —
once to re-measure the accepted checkpoint, once for the candidate. Phase 8
Agent 7 spent it once already. All of it is counted in the packet.

## 8. What I did not do

No setup-model implementation, no G2 work, no later experiment, no push, merge,
rebase or tag, no Agent 3 instruction. Tier and stress diagnostics were not run:
no Phase 8 gate covers them and the paired contract does not name them.

## 9. A pattern worth naming

This is the third time the project has predeclared a margin without checking the
instrument's resolution — after the Phase 16 shootout's 0.03 margin at 0.53 SE,
and Phase 17's 120-board lane with a 0.1379 minimum detectable effect. A standing
rule that every predeclared margin ships with the **n** required to certify it
would have caught all three before the run instead of after.
